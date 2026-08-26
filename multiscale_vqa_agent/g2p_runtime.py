import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from data.dataset import load_feature_file
from models.g2p_toolbank import G2PHypergraphToolBank

from .registry import ToolBankRegistry


COORD_PATTERN = re.compile(r"(?P<x>\d+)_(?P<y>\d+)_(?P<size>\d+)(?:\.[A-Za-z0-9]+)?$")


def parse_patch_coordinate(value: Any, default_size: int) -> Tuple[Optional[int], Optional[int], int]:
    if value is None:
        return None, None, default_size
    if isinstance(value, (list, tuple, np.ndarray)) and len(value) >= 2:
        return int(value[0]), int(value[1]), int(value[2]) if len(value) > 2 else default_size
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="ignore")
    match = COORD_PATTERN.search(Path(str(value)).name)
    if not match:
        return None, None, default_size
    return int(match.group("x")), int(match.group("y")), int(match.group("size"))


def rmst_risk(hazards: np.ndarray, time_bins: List[float]) -> float:
    hazards = np.clip(np.asarray(hazards, dtype=float), 0.0, 1.0)
    ends = np.asarray(time_bins, dtype=float)
    widths = np.diff(np.concatenate([[0.0], ends]))
    survival_end = np.cumprod(1.0 - hazards)
    survival_start = np.concatenate([[1.0], survival_end[:-1]])
    return -float(np.sum(0.5 * (survival_start + survival_end) * widths))


class ScaleRuntime:
    def __init__(
        self,
        scale: int,
        tool_dir: Path,
        registry: ToolBankRegistry,
        device: torch.device,
        manifest_path: Optional[Path] = None,
    ):
        self.scale = int(scale)
        self.tool_dir = Path(tool_dir)
        self.registry = registry
        self.device = device
        self.ckpt = torch.load(self.tool_dir / "model.pt", map_location="cpu")
        self.relations = {k: v for k, v in np.load(self.tool_dir / "relations.npz").items()}
        self.manifest_path = Path(
            manifest_path or (self.tool_dir.parent / "aligned_manifest.csv")
        )
        self.manifest = pd.read_csv(self.manifest_path)
        required = {"case_id", "slide_id", "feature_path"}
        missing = required.difference(self.manifest.columns)
        if missing:
            raise ValueError(
                f"Scale-{self.scale} manifest {self.manifest_path} is missing "
                f"columns: {sorted(missing)}"
            )
        self.manifest["case_id"] = self.manifest["case_id"].astype(str).str[:12]
        self.model = self._load_model()

    def _load_model(self) -> G2PHypergraphToolBank:
        relation = self.relations
        vocab = self.registry.vocabs[self.scale]
        init_h = relation.get("H_gene_prior", relation["H_gene_to_program"])
        model = G2PHypergraphToolBank(
            self.ckpt["feature_dim"],
            self.ckpt["hidden_dim"],
            self.ckpt["phenotype_specs"],
            len(vocab["gene_list"]),
            self.ckpt["program_names"],
            init_h,
            relation["R_prior"],
            gene_phenotype_prior=relation.get("G_gene_to_phenotype_prior"),
            gene_names=vocab["gene_list"],
            relation_init_mode=self.ckpt.get("relation_init_mode", "prior"),
            relation_init_value=self.ckpt.get("relation_init_value", 0.5),
            relation_selection_mode=self.ckpt.get("relation_selection_mode", "prior_guided"),
        )
        state = dict(self.ckpt["state_dict"])
        if "gene_embeddings" in state and "gene_prototypes" not in state:
            state["gene_prototypes"] = state.pop("gene_embeddings")
        if "gene_prototypes" in state and "gene_identity_embeddings" not in state:
            state["gene_identity_embeddings"] = state.pop("gene_prototypes")
        missing, unexpected = model.load_state_dict(state, strict=False)
        material_missing = [k for k in missing if not k.startswith(("rna_", "gene_head."))]
        if material_missing or unexpected:
            raise RuntimeError(
                f"Scale {self.scale} checkpoint mismatch: missing={material_missing}, unexpected={unexpected}"
            )
        model.phenotype_mode = self.ckpt.get("phenotype_mode", "full")
        return model.to(self.device).eval()

    def infer_case(self, case_id: str) -> Dict[str, Any]:
        rows = self.manifest[self.manifest["case_id"] == case_id]
        if rows.empty:
            raise KeyError(f"No scale-{self.scale} feature found for {case_id}")
        slide_results = []
        for row in rows.to_dict("records"):
            features, raw_coords = load_feature_file(row["feature_path"])
            tensor = torch.from_numpy(features).to(self.device)
            with torch.inference_mode():
                output = self.model(tensor)
            coords = [
                parse_patch_coordinate(raw_coords[i] if raw_coords is not None else None, self.scale)
                for i in range(len(features))
            ]
            slide_results.append({
                "slide_id": str(row["slide_id"]),
                "feature_path": str(row["feature_path"]),
                "features": features.astype(np.float16, copy=False),
                "coords": coords,
                "gene_pred": output["gene_pred"].float().cpu().numpy(),
                "program_pred": output["program_pred"].float().cpu().numpy(),
                "phenotype_predictions": self._decode_phenotypes(output["phenotype_logits"]),
                "gene_attention": output["gene_attention"].half().cpu().numpy(),
                "program_attention": output["program_attention"].half().cpu().numpy(),
                "phenotype_attention": output["phenotype_attention"].half().cpu().numpy(),
            })
        return {
            "scale": self.scale,
            "case_id": case_id,
            "slides": slide_results,
            "patient_predictions": self._aggregate_predictions(slide_results),
            "gene_pred": np.mean([s["gene_pred"] for s in slide_results], axis=0),
            "program_pred": np.mean([s["program_pred"] for s in slide_results], axis=0),
        }

    def _decode_phenotypes(self, logits: List[torch.Tensor]) -> Dict[str, Dict[str, Any]]:
        result = {}
        for spec, logit in zip(self.ckpt["phenotype_specs"], logits):
            field = spec["field"]
            task_type = spec["task_type"]
            flat = logit.float().reshape(-1)
            if task_type == "binary":
                probability = float(torch.sigmoid(flat[0]).item())
                value = {"probability": probability, "predicted_class": int(probability >= 0.5)}
            elif task_type == "multiclass":
                probabilities = torch.softmax(flat, dim=0).cpu().tolist()
                predicted = int(np.argmax(probabilities))
                name = self.registry.field_to_name[field]
                inverse = {v: k for k, v in self.registry.label_encoders.get(name, {}).items()}
                value = {
                    "probabilities": probabilities,
                    "predicted_class": predicted,
                    "predicted_label": inverse.get(predicted, str(predicted)),
                }
            elif task_type == "discrete_survival":
                hazards = torch.sigmoid(flat[: int(spec["num_bins"])]).cpu().numpy()
                value = {
                    "hazards": hazards.tolist(),
                    "survival": np.cumprod(1.0 - hazards).tolist(),
                    "risk": rmst_risk(hazards, spec["time_bins"]),
                    "time_bins": spec["time_bins"],
                    "bin_labels": spec["bin_labels"],
                }
            else:
                value = {"value": float(flat[0].item())}
            result[field] = value
        return result

    @staticmethod
    def _aggregate_predictions(slides: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        fields = slides[0]["phenotype_predictions"]
        result = {}
        for field, first in fields.items():
            values = [s["phenotype_predictions"][field] for s in slides]
            if "probability" in first:
                probability = float(np.mean([v["probability"] for v in values]))
                result[field] = {"probability": probability, "predicted_class": int(probability >= 0.5)}
            elif "probabilities" in first:
                probabilities = np.mean([v["probabilities"] for v in values], axis=0)
                predicted = int(np.argmax(probabilities))
                result[field] = {
                    "probabilities": probabilities.tolist(),
                    "predicted_class": predicted,
                    "predicted_label": next(
                        (v["predicted_label"] for v in values if v["predicted_class"] == predicted), str(predicted)
                    ),
                }
            elif "hazards" in first:
                hazards = np.mean([v["hazards"] for v in values], axis=0)
                result[field] = dict(first)
                result[field].update({
                    "hazards": hazards.tolist(),
                    "survival": np.cumprod(1.0 - hazards).tolist(),
                    "risk": rmst_risk(hazards, first["time_bins"]),
                })
            else:
                result[field] = {"value": float(np.mean([v["value"] for v in values]))}
        return result


class MultiScaleG2PAgent:
    def __init__(self, config: Dict[str, Any], registry: ToolBankRegistry):
        requested = str(config.get("device", "auto"))
        if requested == "auto":
            requested = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(requested)
        self.registry = registry
        manifest_paths = config.get("scale_manifests", {})
        self.runtimes = {}
        for scale, directory in config["scales"].items():
            manifest_path = manifest_paths.get(str(scale))
            self.runtimes[int(scale)] = ScaleRuntime(
                int(scale),
                Path(directory),
                registry,
                self.device,
                Path(manifest_path) if manifest_path else None,
            )
        self.fusion_weights = config["fusion_weights"]

    def infer_case(self, case_id: str) -> Dict[int, Dict[str, Any]]:
        return {scale: runtime.infer_case(case_id) for scale, runtime in sorted(self.runtimes.items())}

    def fuse_task(self, results: Dict[int, Dict[str, Any]], field: str) -> Dict[str, Any]:
        name = self.registry.field_to_name[field]
        vocab = self.registry.vocabs[min(self.registry.vocabs)]
        group = vocab.get("phenotype_groups", {}).get(name, "morphology")
        if field == "OS":
            group = "survival"
        configured = self.fusion_weights[group]
        available = [(s, float(configured[str(s)])) for s in results if field in results[s]["patient_predictions"]]
        total = sum(weight for _, weight in available)
        weights = {s: weight / total for s, weight in available}
        predictions = {s: results[s]["patient_predictions"][field] for s in weights}
        first = next(iter(predictions.values()))
        if "probability" in first:
            probability = sum(weights[s] * predictions[s]["probability"] for s in weights)
            fused = {"probability": probability, "predicted_class": int(probability >= 0.5)}
        elif "probabilities" in first:
            probabilities = sum(weights[s] * np.asarray(predictions[s]["probabilities"]) for s in weights)
            predicted = int(np.argmax(probabilities))
            fused = {"probabilities": probabilities.tolist(), "predicted_class": predicted}
            label = next((p.get("predicted_label") for p in predictions.values() if p["predicted_class"] == predicted), None)
            if label is not None:
                fused["predicted_label"] = label
        elif "hazards" in first:
            hazards = sum(weights[s] * np.asarray(predictions[s]["hazards"]) for s in weights)
            fused = dict(first)
            fused.update({
                "hazards": hazards.tolist(),
                "survival": np.cumprod(1.0 - hazards).tolist(),
                "risk": rmst_risk(hazards, first["time_bins"]),
            })
        else:
            fused = {"value": sum(weights[s] * predictions[s]["value"] for s in weights)}
        semantics = self.registry.label_semantics(field)
        if "predicted_class" in fused and not fused.get("predicted_label"):
            fused["predicted_label"] = semantics["class_to_label"].get(str(fused["predicted_class"]))
        for prediction in predictions.values():
            if "predicted_class" in prediction and not prediction.get("predicted_label"):
                prediction["predicted_label"] = semantics["class_to_label"].get(str(prediction["predicted_class"]))
        return {
            "field": field,
            "label_semantics": semantics,
            "weights": {str(k): v for k, v in weights.items()},
            "per_scale": {str(k): v for k, v in predictions.items()},
            "fused": fused,
            "validation_metrics": self.registry.task_metrics(field),
        }
