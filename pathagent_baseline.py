import gc
import hashlib
import json
import re
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np


RETRIEVER_SEMANTICS = (
    "CONCH_v1_text_preprojection_768_vs_"
    "CONCH_v1.5_patch_768_heuristic"
)
BASELINE_NAME = "PathAgent-CONCH-MS"
BASELINE_VARIANT = "adapted_fixed20_5_single_zoom_conch768"


def normalize_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def normalized_key(case_id: Any, question: Any) -> Tuple[str, str]:
    return str(case_id)[:12], normalize_text(question).lower()


def _l2_normalize(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    return array / np.maximum(norms, 1e-12)


def _repo_modules() -> Tuple[Any, Any, Any, Any]:
    """Import mature read-only utilities without making the old repo our root."""
    import sys

    old_repo = Path(__file__).resolve().parents[1] / "g2p_toolbank_brca"
    if str(old_repo) not in sys.path:
        sys.path.insert(0, str(old_repo))
    from data.dataset import discover_feature_files, load_feature_file
    from multiscale_vqa_agent.clients import (
        OpenAICompatibleClient,
        parse_json_response,
    )
    from multiscale_vqa_agent.g2p_runtime import parse_patch_coordinate

    return (
        discover_feature_files,
        load_feature_file,
        OpenAICompatibleClient,
        (parse_json_response, parse_patch_coordinate),
    )


@dataclass
class PatchRecord:
    scale: int
    slide_id: str
    slide_key: str
    patch_index: int
    x: Optional[int]
    y: Optional[int]
    size: int
    similarity: float = 0.0
    retrieval_query: str = ""
    image_path: Optional[str] = None

    @property
    def identity(self) -> Tuple[int, str, int]:
        return self.scale, self.slide_id, self.patch_index

    @property
    def center(self) -> Optional[Tuple[float, float]]:
        if self.x is None or self.y is None:
            return None
        return self.x + self.size / 2.0, self.y + self.size / 2.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SlideFeatures:
    scale: int
    slide_id: str
    slide_key: str
    feature_path: str
    features: np.ndarray
    coords: List[Tuple[Optional[int], Optional[int], int]]


class CaseFeatures:
    def __init__(self, case_id: str, scales: Dict[int, List[SlideFeatures]]):
        self.case_id = case_id
        self.scales = scales

    def feature_dim(self, scale: int) -> Optional[int]:
        slides = self.scales.get(int(scale), [])
        return int(slides[0].features.shape[1]) if slides else None


class CaseFeatureStore:
    """Lightweight manifests plus one-case-at-a-time feature loading."""

    def __init__(self, feature_roots: Dict[Any, str]):
        discover, self._load_feature, _, utilities = _repo_modules()
        self._parse_coordinate = utilities[1]
        self.feature_roots = {int(scale): Path(path) for scale, path in feature_roots.items()}
        missing = [str(path) for path in self.feature_roots.values() if not path.is_dir()]
        if missing:
            raise FileNotFoundError(f"Missing feature roots: {missing}")
        self.manifests: Dict[int, Dict[str, List[Dict[str, str]]]] = {}
        for scale, root in self.feature_roots.items():
            frame = discover(root)
            grouped: Dict[str, List[Dict[str, str]]] = defaultdict(list)
            for row in frame.to_dict("records"):
                grouped[str(row["case_id"])[:12]].append(row)
            self.manifests[scale] = dict(grouped)

    @staticmethod
    def slide_key(slide_id: str) -> str:
        return str(slide_id).rsplit("_", 2)[0]

    def load_case(self, case_id: str) -> CaseFeatures:
        case_id = str(case_id)[:12]
        scales: Dict[int, List[SlideFeatures]] = {}
        for scale in sorted(self.feature_roots):
            slides = []
            for row in self.manifests[scale].get(case_id, []):
                features, raw_coords = self._load_feature(row["feature_path"])
                features = _l2_normalize(features)
                if raw_coords is None:
                    raise ValueError(
                        f"Feature coordinates are required: {row['feature_path']}"
                    )
                if len(raw_coords) != len(features):
                    raise ValueError(
                        f"Feature/coordinate count mismatch: {row['feature_path']}"
                    )
                coords = [
                    self._parse_coordinate(value, scale) for value in raw_coords
                ]
                slide_id = str(row["slide_id"])
                slides.append(SlideFeatures(
                    scale=scale,
                    slide_id=slide_id,
                    slide_key=self.slide_key(slide_id),
                    feature_path=str(row["feature_path"]),
                    features=features,
                    coords=coords,
                ))
            if not slides:
                raise KeyError(f"No scale-{scale} features found for {case_id}")
            scales[scale] = slides
        return CaseFeatures(case_id, scales)


class CONCHTextEncoder:
    """Online CONCH-v1 pre-projection text encoder with exact-string cache."""

    output_dim = 768

    def __init__(self, checkpoint: str, device: str = "cuda:0"):
        checkpoint_path = Path(checkpoint)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"CONCH checkpoint not found: {checkpoint_path}")
        import torch
        from conch.open_clip_custom import (
            create_model_from_pretrained,
            get_tokenizer,
            tokenize,
        )

        self.torch = torch
        self.tokenize = tokenize
        self.device = torch.device(device)
        self.model, _ = create_model_from_pretrained(
            "conch_ViT-B-16",
            checkpoint_path=str(checkpoint_path),
            device=self.device,
        )
        self.model.eval()
        self.tokenizer = get_tokenizer()
        self.cache: Dict[str, np.ndarray] = {}

    def encode(self, text: str) -> np.ndarray:
        key = normalize_text(text)
        if not key:
            raise ValueError("CONCH retrieval query cannot be empty")
        if key in self.cache:
            return self.cache[key]
        projection = self.model.text.text_projection
        if projection is None:
            raise RuntimeError("CONCH text tower has no text projection")
        token_ids = self.tokenize(self.tokenizer, [key]).to(self.device)
        input_ids = token_ids[:, :-1]
        self.model.text.text_projection = None
        try:
            with self.torch.inference_mode():
                output = self.model.text(input_ids)
        finally:
            self.model.text.text_projection = projection
        feature = output[0] if isinstance(output, tuple) else output
        feature = feature.float().cpu().numpy().reshape(-1)
        if feature.shape[0] != self.output_dim:
            raise RuntimeError(f"Unexpected CONCH query dimension: {feature.shape}")
        feature = _l2_normalize(feature)[0]
        self.cache[key] = feature
        return feature


class PathAgentRetriever:
    def __init__(self, text_encoder: Any):
        self.text_encoder = text_encoder

    def encode_query(self, text: str) -> np.ndarray:
        return _l2_normalize(self.text_encoder.encode(text))[0]

    @staticmethod
    def _validate_dim(query: np.ndarray, features: np.ndarray) -> None:
        if features.ndim != 2 or query.ndim != 1:
            raise ValueError("Query must be 1D and patch features must be 2D")
        if features.shape[1] != query.shape[0]:
            raise ValueError(
                "CONCH query/patch feature dimension mismatch: "
                f"query={query.shape[0]} patch={features.shape[1]}"
            )

    def retrieve(
        self,
        case: CaseFeatures,
        query_text: str,
        scale: int,
        top_k: int,
        visited: Optional[Set[Tuple[int, str, int]]] = None,
    ) -> List[PatchRecord]:
        query = self.encode_query(query_text)
        candidates: List[PatchRecord] = []
        for slide in case.scales.get(int(scale), []):
            self._validate_dim(query, slide.features)
            scores = slide.features @ query
            for index, score in enumerate(scores):
                identity = (int(scale), slide.slide_id, index)
                if visited and identity in visited:
                    continue
                x, y, size = slide.coords[index]
                candidates.append(PatchRecord(
                    scale=int(scale), slide_id=slide.slide_id,
                    slide_key=slide.slide_key, patch_index=index,
                    x=x, y=y, size=size, similarity=float(score),
                    retrieval_query=normalize_text(query_text),
                ))
        candidates.sort(key=lambda patch: patch.similarity, reverse=True)
        return candidates[:max(0, int(top_k))]

    @staticmethod
    def _record_feature(case: CaseFeatures, patch: PatchRecord) -> np.ndarray:
        for slide in case.scales.get(patch.scale, []):
            if slide.slide_id == patch.slide_id:
                return slide.features[patch.patch_index]
        raise KeyError(f"Patch feature not found: {patch.identity}")

    def select_zoom_parents(
        self,
        case: CaseFeatures,
        current_round: Sequence[PatchRecord],
        original_query: np.ndarray,
        top_k: int = 2,
    ) -> List[PatchRecord]:
        query = _l2_normalize(original_query)[0]
        scored = []
        for patch in current_round:
            if patch.scale != 4096:
                continue
            feature = self._record_feature(case, patch)
            self._validate_dim(query, feature.reshape(1, -1))
            scored.append((float(feature @ query), patch))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [patch for _, patch in scored[:max(0, int(top_k))]]

    def select_zoom_child(
        self,
        case: CaseFeatures,
        parents: Sequence[PatchRecord],
        target_scale: int,
        original_query: np.ndarray,
    ) -> Tuple[Optional[PatchRecord], int, Optional[PatchRecord]]:
        if int(target_scale) not in {1024, 2048}:
            raise ValueError("Zoom target scale must be 1024 or 2048")
        query = _l2_normalize(original_query)[0]
        eligible: List[Tuple[float, PatchRecord, PatchRecord]] = []
        for parent in parents:
            if parent.x is None or parent.y is None:
                continue
            for slide in case.scales.get(int(target_scale), []):
                if slide.slide_key != parent.slide_key:
                    continue
                self._validate_dim(query, slide.features)
                scores = slide.features @ query
                for index, (coords, score) in enumerate(zip(slide.coords, scores)):
                    x, y, size = coords
                    if x is None or y is None:
                        continue
                    center_x, center_y = x + size / 2.0, y + size / 2.0
                    if not (
                        parent.x <= center_x <= parent.x + parent.size
                        and parent.y <= center_y <= parent.y + parent.size
                    ):
                        continue
                    child = PatchRecord(
                        scale=int(target_scale), slide_id=slide.slide_id,
                        slide_key=slide.slide_key, patch_index=index,
                        x=x, y=y, size=size, similarity=float(score),
                        retrieval_query="original_question_zoom",
                    )
                    eligible.append((float(score), child, parent))
        if not eligible:
            return None, 0, None
        eligible.sort(key=lambda item: item[0], reverse=True)
        _, child, parent = eligible[0]
        return child, len(eligible), parent


class StrictWSICropper:
    def __init__(self, wsi_root: str, output_root: str):
        self.wsi_root = Path(wsi_root)
        self.output_root = Path(output_root)
        if not self.wsi_root.is_dir():
            raise FileNotFoundError(f"WSI root not found: {self.wsi_root}")
        self._index: Optional[Dict[str, Dict[str, Path]]] = None

    def _build_index(self) -> None:
        index: Dict[str, Dict[str, Path]] = defaultdict(dict)
        for path in self.wsi_root.rglob("*.svs"):
            index[path.name[:12]][path.stem] = path
        self._index = dict(index)

    def resolve(self, case_id: str, slide_key: str) -> Path:
        if self._index is None:
            self._build_index()
        path = self._index.get(str(case_id)[:12], {}).get(slide_key)
        if path is None:
            raise FileNotFoundError(
                f"Exact WSI not found for case={case_id} slide={slide_key}"
            )
        return path

    def crop(self, case_id: str, question: str, patch: PatchRecord) -> str:
        if patch.x is None or patch.y is None:
            raise ValueError(f"Patch has no coordinate: {patch.identity}")
        try:
            import openslide
        except ImportError as error:
            raise RuntimeError("openslide-python is required for WSI cropping") from error
        question_id = hashlib.sha1(question.encode("utf-8")).hexdigest()[:10]
        directory = self.output_root / str(case_id)[:12] / question_id
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / (
            f"{patch.scale}_{hashlib.sha1(patch.slide_id.encode()).hexdigest()[:8]}_"
            f"{patch.patch_index}_{patch.x}_{patch.y}.jpg"
        )
        if not target.exists() or target.stat().st_size == 0:
            wsi_path = self.resolve(case_id, patch.slide_key)
            with openslide.OpenSlide(str(wsi_path)) as slide:
                image = slide.read_region(
                    (patch.x, patch.y), 0, (patch.size, patch.size)
                ).convert("RGB")
            image.thumbnail((1024, 1024))
            image.save(target, "JPEG", quality=90)
        patch.image_path = str(target)
        return str(target)


PERCEPTOR_SYSTEM = """You are the visual perceptor in a pathology agent.
Inspect only the supplied H&E patch images. Do not use patient metadata, reports, reference answers, IHC, FISH, RNA, mutation, or other unavailable assays. Return JSON with a patches list. Each item must use the supplied image ordinal and contain concise pathology_features, relevance_to_question, and answer_hint. An answer_hint may name one supplied option only when visibly supported; otherwise say indeterminate."""


class PathAgentPerceptor:
    def __init__(self, client: Any, batch_size: int = 5):
        self.client = client
        self.batch_size = max(1, int(batch_size))

    def describe(
        self,
        question: str,
        choices: Sequence[str],
        patches: Sequence[PatchRecord],
        scale: int,
        retrieval_query: str,
    ) -> List[Dict[str, Any]]:
        descriptions = []
        for start in range(0, len(patches), self.batch_size):
            batch = list(patches[start:start + self.batch_size])
            parsed = self._request(question, choices, batch, scale, retrieval_query)
            if parsed is None:
                for patch in batch:
                    one = self._request(
                        question, choices, [patch], scale, retrieval_query
                    )
                    descriptions.extend(self._attach(one, [patch]))
            else:
                descriptions.extend(self._attach(parsed, batch))
        return descriptions

    def _request(
        self,
        question: str,
        choices: Sequence[str],
        patches: Sequence[PatchRecord],
        scale: int,
        retrieval_query: str,
    ) -> Optional[Dict[str, Any]]:
        payload = {
            "question": question,
            "choices": list(choices),
            "scale_fov": int(scale),
            "retrieval_query": retrieval_query,
            "images": [
                {"ordinal": index + 1, "instruction": "Inspect this H&E patch."}
                for index in range(len(patches))
            ],
            "output_schema": {
                "patches": [{
                    "ordinal": 1,
                    "pathology_features": "visible H&E morphology",
                    "relevance_to_question": "direct relevance or indeterminate",
                    "answer_hint": "one supplied option or indeterminate",
                }]
            },
        }
        parse_json, _ = _repo_modules()[3]
        for _ in range(2):
            raw = self.client.chat(
                PERCEPTOR_SYSTEM,
                json.dumps(payload, ensure_ascii=False),
                images=[str(patch.image_path) for patch in patches],
                temperature=0.0,
                max_tokens=1400,
                response_format={"type": "json_object"},
                retries=2,
                enable_thinking=False,
                image_max_size=1024,
            )
            parsed = parse_json(raw)
            rows = parsed.get("patches") if isinstance(parsed, dict) else None
            if isinstance(rows, list) and rows:
                return parsed
        return None

    @staticmethod
    def _attach(
        parsed: Optional[Dict[str, Any]], patches: Sequence[PatchRecord]
    ) -> List[Dict[str, Any]]:
        rows = parsed.get("patches", []) if isinstance(parsed, dict) else []
        by_ordinal = {
            int(row.get("ordinal")): row
            for row in rows
            if isinstance(row, dict) and str(row.get("ordinal", "")).isdigit()
        }
        attached = []
        for ordinal, patch in enumerate(patches, 1):
            row = by_ordinal.get(ordinal, {})
            attached.append({
                "patch": patch.to_dict(),
                "pathology_features": normalize_text(
                    row.get("pathology_features", "indeterminate")
                ),
                "relevance_to_question": normalize_text(
                    row.get("relevance_to_question", "indeterminate")
                ),
                "answer_hint": normalize_text(
                    row.get("answer_hint", "indeterminate")
                ),
                "perceptor_fallback": not bool(row),
            })
        return attached


EXECUTOR_SYSTEM = """You are the executor in a pure-visual pathology MCQ agent.
Use only the supplied question, choices, and accumulated H&E patch observations. Never use or request a reference answer. WSI morphology cannot become measured IHC, FISH, RNA, mutation, clinical history, or report facts. Follow the requested JSON schema exactly."""


class PathAgentExecutor:
    def __init__(self, client: Any):
        self.client = client
        self.prompt_log: List[Dict[str, Any]] = []

    def _call(self, stage: str, payload: Dict[str, Any], max_tokens: int = 700) -> Dict[str, Any]:
        self.prompt_log.append({"stage": stage, "payload": payload})
        parse_json = _repo_modules()[3][0]
        raw = self.client.chat(
            EXECUTOR_SYSTEM,
            json.dumps(payload, ensure_ascii=False),
            temperature=0.0,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
            retries=2,
            enable_thinking=False,
        )
        return parse_json(raw) or {}

    @staticmethod
    def _exact_choice(answer: Any, choices: Sequence[str]) -> Optional[str]:
        return str(answer) if isinstance(answer, str) and answer in choices else None

    @staticmethod
    def _compact_evidence(evidence: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [
            {
                "scale_fov": (row.get("patch") or {}).get("scale"),
                "pathology_features": row.get("pathology_features"),
                "relevance_to_question": row.get("relevance_to_question"),
                "answer_hint": row.get("answer_hint"),
            }
            for row in evidence
        ]

    def _repair_choice(self, generated: Any, choices: Sequence[str]) -> str:
        payload = {
            "instruction": "Map the generated answer to exactly one supplied choice.",
            "generated_answer": normalize_text(generated),
            "choices": list(choices),
            "output_schema": {"answer": "exact supplied choice text"},
        }
        parsed = self._call("option_repair", payload, max_tokens=120)
        return self._exact_choice(parsed.get("answer"), choices) or choices[0]

    def preliminary(
        self, question: str, choices: Sequence[str], evidence: Sequence[Dict[str, Any]]
    ) -> Dict[str, Any]:
        payload = {
            "task": "Choose a preliminary answer from accumulated visual evidence.",
            "question": question,
            "choices": list(choices),
            "accumulated_evidence": self._compact_evidence(evidence),
            "output_schema": {"answer": "exact supplied choice text", "reasoning": "concise"},
        }
        parsed = self._call("preliminary", payload)
        answer = self._exact_choice(parsed.get("answer"), choices)
        if answer is None:
            answer = self._repair_choice(parsed.get("answer"), choices)
        return {"answer": answer, "reasoning": normalize_text(parsed.get("reasoning"))}

    def sufficiency(
        self,
        question: str,
        choices: Sequence[str],
        evidence: Sequence[Dict[str, Any]],
        preliminary: Dict[str, Any],
    ) -> Dict[str, Any]:
        payload = {
            "task": "Judge evidence sufficiency only; do not choose a new answer.",
            "question": question,
            "choices": list(choices),
            "accumulated_evidence": self._compact_evidence(evidence),
            "preliminary_answer": preliminary,
            "output_schema": {"sufficient": "Yes or No", "reason": "concise"},
        }
        parsed = self._call("sufficiency", payload, max_tokens=350)
        sufficient = "Yes" if str(parsed.get("sufficient", "No")).lower() == "yes" else "No"
        return {"sufficient": sufficient, "reason": normalize_text(parsed.get("reason"))}

    def evidence_plan(
        self,
        question: str,
        choices: Sequence[str],
        evidence: Sequence[Dict[str, Any]],
    ) -> Dict[str, Any]:
        payload = {
            "task": "Identify missing visual evidence and whether a smaller FOV is needed.",
            "question": question,
            "choices": list(choices),
            "accumulated_evidence": self._compact_evidence(evidence),
            "fov_meaning": {"4096": "coarse", "2048": "intermediate", "1024": "fine"},
            "output_schema": {
                "missing_info": "visual morphology to seek",
                "zoom_recommendation": "Yes or No",
                "recommended_scale": "2048 or 1024 or null",
                "zoom_reason": "concise",
            },
        }
        parsed = self._call("evidence_plan", payload, max_tokens=450)
        zoom = "Yes" if str(parsed.get("zoom_recommendation", "No")).lower() == "yes" else "No"
        try:
            scale = int(parsed.get("recommended_scale"))
        except (TypeError, ValueError):
            scale = None
        if scale not in {1024, 2048}:
            scale = None
            zoom = "No"
        return {
            "missing_info": normalize_text(parsed.get("missing_info")) or question,
            "zoom_recommendation": zoom,
            "recommended_scale": scale,
            "zoom_reason": normalize_text(parsed.get("zoom_reason")),
        }

    def final(
        self, question: str, choices: Sequence[str], evidence: Sequence[Dict[str, Any]]
    ) -> Dict[str, Any]:
        payload = {
            "task": "Select the final answer from all accumulated visual evidence.",
            "question": question,
            "choices": list(choices),
            "accumulated_evidence": self._compact_evidence(evidence),
            "output_schema": {"answer": "exact supplied choice text", "explanation": "concise"},
        }
        parsed = self._call("final", payload)
        answer = self._exact_choice(parsed.get("answer"), choices)
        if answer is None:
            answer = self._repair_choice(parsed.get("answer"), choices)
        return {"answer": answer, "explanation": normalize_text(parsed.get("explanation"))}


class PathAgentBaseline:
    def __init__(
        self,
        retriever: PathAgentRetriever,
        cropper: Any,
        perceptor: PathAgentPerceptor,
        executor: PathAgentExecutor,
        initial_patches: int = 20,
        replenish_patches: int = 5,
        max_attempts: int = 5,
        zoom_parent_topk: int = 2,
        max_zoom_actions: int = 1,
    ):
        self.retriever = retriever
        self.cropper = cropper
        self.perceptor = perceptor
        self.executor = executor
        self.initial_patches = int(initial_patches)
        self.replenish_patches = int(replenish_patches)
        self.max_attempts = int(max_attempts)
        self.zoom_parent_topk = int(zoom_parent_topk)
        self.max_zoom_actions = int(max_zoom_actions)

    def answer(self, item: Dict[str, Any], case: CaseFeatures) -> Dict[str, Any]:
        case_id = str(item.get("Id", item.get("case_id", "")))[:12]
        question = normalize_text(item.get("Question", item.get("question", "")))
        choices = list(item.get("Choice", item.get("choices", [])) or [])
        if not case_id or not question or not choices:
            raise ValueError("PathAgent requires case_id, question, and choices")

        original_query = self.retriever.encode_query(question)
        visited: Set[Tuple[int, str, int]] = set()
        accumulated: List[Dict[str, Any]] = []
        rounds = []
        retrieval_query = question
        zoom_actions = 0
        final_answer = None

        for attempt in range(1, self.max_attempts + 1):
            budget = self.initial_patches if attempt == 1 else self.replenish_patches
            current = self.retriever.retrieve(
                case, retrieval_query, 4096, budget, visited
            )
            for patch in current:
                visited.add(patch.identity)
                self.cropper.crop(case_id, question, patch)
            descriptions = self.perceptor.describe(
                question, choices, current, 4096, retrieval_query
            )
            accumulated.extend(descriptions)
            preliminary = self.executor.preliminary(question, choices, accumulated)
            sufficiency = self.executor.sufficiency(
                question, choices, accumulated, preliminary
            )
            round_trace = {
                "attempt": attempt,
                "retrieval_query": retrieval_query,
                "new_patch_count": len(current),
                "accumulated_patch_count": len(accumulated),
                "retrieved_patches": [patch.to_dict() for patch in current],
                "patch_descriptions": descriptions,
                "preliminary_answer": preliminary,
                "sufficiency": sufficiency,
                "evidence_plan": None,
                "zoom": None,
            }
            rounds.append(round_trace)

            if sufficiency["sufficient"] == "Yes":
                final_answer = self.executor.final(question, choices, accumulated)
                break

            plan = self.executor.evidence_plan(question, choices, accumulated)
            round_trace["evidence_plan"] = plan
            if (
                plan["zoom_recommendation"] == "Yes"
                and zoom_actions < self.max_zoom_actions
            ):
                zoom_actions += 1
                parents = self.retriever.select_zoom_parents(
                    case, current, original_query, self.zoom_parent_topk
                )
                child, eligible_count, parent = self.retriever.select_zoom_child(
                    case, parents, plan["recommended_scale"], original_query
                )
                zoom_trace = {
                    "parents": [patch.to_dict() for patch in parents],
                    "requested_scale": plan["recommended_scale"],
                    "eligible_child_count": eligible_count,
                    "selected_child": child.to_dict() if child else None,
                    "zoom_description": None,
                }
                if child is not None:
                    self.cropper.crop(case_id, question, child)
                    zoom_description = self.perceptor.describe(
                        question,
                        choices,
                        [child],
                        plan["recommended_scale"],
                        question,
                    )
                    accumulated.extend(zoom_description)
                    zoom_trace["zoom_description"] = zoom_description
                    zoom_trace["parent_scale"] = 4096
                    zoom_trace["target_scale"] = child.scale
                    zoom_trace["parent_slide_id"] = parent.slide_id if parent else None
                    zoom_trace["parent_patch_index"] = parent.patch_index if parent else None
                    zoom_trace["child_patch_index"] = child.patch_index
                    zoom_trace["child_similarity"] = child.similarity
                round_trace["zoom"] = zoom_trace
                final_answer = self.executor.final(question, choices, accumulated)
                break
            retrieval_query = plan["missing_info"]

        if final_answer is None:
            final_answer = self.executor.final(question, choices, accumulated)
        if final_answer["answer"] not in choices:
            raise RuntimeError("Final PathAgent answer is not a supplied choice")

        return {
            "case_id": case_id,
            "question": question,
            "choices": choices,
            "agent_answer": final_answer,
            "baseline_name": BASELINE_NAME,
            "baseline_variant": BASELINE_VARIANT,
            "retriever_semantics": RETRIEVER_SEMANTICS,
            "initial_patch_count": self.initial_patches,
            "replenish_patch_count": self.replenish_patches,
            "max_attempts": self.max_attempts,
            "predicted_can_answer": True,
            "predicted_answerability": "answerable",
            "answerability_confidence": 1.0,
            "answerability_reason": "PathAgent baseline uses an always-answer policy.",
            "answerability_fallback_used": False,
            "abstained": False,
            "adaptations": {
                "plip_replaced_by_conch": True,
                "percentage_budget_replaced_by_fixed_budget": True,
                "initial_patch_count": self.initial_patches,
                "replenish_patch_count": self.replenish_patches,
                "official_digital_zoom_replaced_by_precomputed_multiscale_fov": True,
                "max_zoom_actions": self.max_zoom_actions,
                "offline_quilt_llava_descriptions": False,
                "pathor1_batched": True,
            },
            "process": {
                "rounds": rounds,
                "zoom_action_count": zoom_actions,
                "visited_4096_count": len(visited),
                "feature_dimensions": {
                    str(scale): case.feature_dim(scale)
                    for scale in (4096, 2048, 1024)
                },
                "query_dimension": int(original_query.shape[0]),
                "final_answer": final_answer,
            },
        }


def release_case(case: CaseFeatures) -> None:
    del case
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
