import hashlib
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

from .registry import ToolBankRegistry
from .schemas import EvidenceGroup, PatchCandidate


SOURCE_WEIGHTS = {
    "morphology": {"phenotype": 0.60, "program": 0.25, "gene": 0.15},
    "molecular": {"phenotype": 0.30, "program": 0.35, "gene": 0.35},
    "clinical": {"phenotype": 0.45, "program": 0.30, "gene": 0.25},
    "survival": {"phenotype": 0.35, "program": 0.35, "gene": 0.30},
}


def box_iou(left: PatchCandidate, right: PatchCandidate) -> float:
    if left.x is None or left.y is None or right.x is None or right.y is None:
        return 0.0
    x1, y1 = max(left.x, right.x), max(left.y, right.y)
    x2 = min(left.x + left.size, right.x + right.size)
    y2 = min(left.y + left.size, right.y + right.size)
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    union = left.size * left.size + right.size * right.size - intersection
    return intersection / union if union else 0.0


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    return float(np.dot(left, right) / denominator) if denominator else 0.0


class MultiScaleRetrievalAgent:
    def __init__(self, registry: ToolBankRegistry, config: Dict[str, Any]):
        self.registry = registry
        self.top_per_source = int(config.get("top_patches_per_source", 4))
        self.all_phenotype_top_per_prototype = int(
            config.get("all_phenotype_top_patches_per_prototype", 2)
        )
        self.max_groups = int(config.get("max_evidence_groups", 4))
        self.iou_threshold = float(config.get("same_scale_iou", 0.5))
        self.cosine_threshold = float(config.get("feature_cosine", 0.95))
        self.bypass = int(config.get("global_bypass_per_scale", 2))

    def retrieve(
        self,
        field: str,
        task_group: str,
        scale_results: Dict[int, Dict[str, Any]],
        relations: Dict[str, Any],
    ) -> List[EvidenceGroup]:
        phenotype_index = self.registry.field_to_index[field]
        program_indices = [item["index"] for item in relations["programs"]]
        gene_indices = [item["index"] for item in relations["genes"]]
        weights = SOURCE_WEIGHTS["survival" if field == "OS" else task_group]
        by_scale: Dict[int, List[PatchCandidate]] = {}
        for scale, result in scale_results.items():
            candidates = []
            for slide in result["slides"]:
                candidates.extend(self._from_attention(
                    scale, slide, "phenotype", [phenotype_index],
                    [field], weights["phenotype"]
                ))
                candidates.extend(self._from_attention(
                    scale, slide, "program", program_indices,
                    [self.registry.programs[i] for i in program_indices], weights["program"]
                ))
                candidates.extend(self._from_attention(
                    scale, slide, "gene", gene_indices,
                    [self.registry.genes[i] for i in gene_indices], weights["gene"]
                ))
            by_scale[scale] = self._deduplicate(candidates)
        return self._build_pyramids(by_scale)

    def retrieve_all_phenotypes(
        self,
        scale_results: Dict[int, Dict[str, Any]],
    ) -> List[EvidenceGroup]:
        indices = list(range(len(self.registry.phenotype_fields)))
        names = [
            f"{self.registry.field_to_prototype_id[field]}:{field}"
            for field in self.registry.phenotype_fields
        ]
        by_scale: Dict[int, List[PatchCandidate]] = {}
        for scale, result in scale_results.items():
            candidates = []
            for slide in result["slides"]:
                candidates.extend(self._from_attention(
                    scale,
                    slide,
                    "phenotype",
                    indices,
                    names,
                    1.0,
                    top_per_prototype=self.all_phenotype_top_per_prototype,
                ))
            # Consensus regions accumulate multiple prototype sources during deduplication.
            by_scale[scale] = self._deduplicate(candidates)
        return self._build_pyramids(by_scale)

    def _from_attention(
        self,
        scale: int,
        slide: Dict[str, Any],
        source_type: str,
        indices: List[int],
        names: List[str],
        source_weight: float,
        top_per_prototype: Optional[int] = None,
    ) -> List[PatchCandidate]:
        if not indices:
            return []
        attention = slide[f"{source_type}_attention"]
        candidates = []
        for prototype_index, prototype_name in zip(indices, names):
            values = np.asarray(attention[prototype_index], dtype=np.float32)
            if not len(values):
                continue
            count = min(
                self.top_per_source if top_per_prototype is None else top_per_prototype,
                len(values),
            )
            top = np.argpartition(values, -count)[-count:]
            top = top[np.argsort(values[top])[::-1]]
            selected = values[top]
            low, high = float(selected.min()), float(selected.max())
            normalized = (selected - low) / (high - low + 1e-8) if high > low else np.ones_like(selected)
            for rank, (patch_index, local_score) in enumerate(zip(top.tolist(), normalized.tolist())):
                x, y, size = slide["coords"][patch_index]
                score = source_weight * (0.7 + 0.3 * local_score) * (1.0 - 0.03 * rank)
                candidates.append(PatchCandidate(
                    scale=scale,
                    slide_id=slide["slide_id"],
                    patch_index=int(patch_index),
                    x=x,
                    y=y,
                    size=size,
                    score=float(score),
                    sources=[{
                        "type": source_type,
                        "name": prototype_name,
                        "attention": float(values[patch_index]),
                    }],
                    feature=np.asarray(slide["features"][patch_index], dtype=np.float32),
                ))
        return candidates

    def _deduplicate(self, candidates: List[PatchCandidate]) -> List[PatchCandidate]:
        kept: List[PatchCandidate] = []
        for candidate in sorted(candidates, key=lambda item: item.score, reverse=True):
            duplicate = next((
                item for item in kept
                if item.slide_id == candidate.slide_id and (
                    box_iou(item, candidate) >= self.iou_threshold
                    or cosine(item.feature, candidate.feature) >= self.cosine_threshold
                )
            ), None)
            if duplicate is None:
                kept.append(candidate)
                continue
            known = {(source["type"], source["name"]) for source in duplicate.sources}
            duplicate.sources.extend(
                source for source in candidate.sources
                if (source["type"], source["name"]) not in known
            )
            duplicate.score = max(duplicate.score, candidate.score) + 0.05 * min(len(duplicate.sources) - 1, 3)
        return sorted(kept, key=lambda item: item.score, reverse=True)

    def _build_pyramids(self, by_scale: Dict[int, List[PatchCandidate]]) -> List[EvidenceGroup]:
        groups: List[EvidenceGroup] = []
        used = set()
        for anchor in by_scale.get(4096, [])[: self.max_groups]:
            patches = {4096: anchor}
            used.add((4096, anchor.slide_id, anchor.patch_index))
            parent = anchor
            for scale in (2048, 1024):
                match = self._best_child(parent, by_scale.get(scale, []), used)
                if match:
                    patches[scale] = match
                    used.add((scale, match.slide_id, match.patch_index))
                    parent = match
            groups.append(EvidenceGroup(len(groups) + 1, self._group_score(patches), patches))
        for scale in (2048, 1024):
            added = 0
            for candidate in by_scale.get(scale, []):
                key = (scale, candidate.slide_id, candidate.patch_index)
                if key in used:
                    continue
                patches = {scale: candidate}
                if scale == 2048:
                    child = self._best_child(candidate, by_scale.get(1024, []), used)
                    if child:
                        patches[1024] = child
                        used.add((1024, child.slide_id, child.patch_index))
                groups.append(EvidenceGroup(len(groups) + 1, self._group_score(patches), patches))
                used.add(key)
                added += 1
                if added >= self.bypass:
                    break
        groups.sort(key=lambda item: item.score, reverse=True)
        groups = groups[: self.max_groups]
        for index, group in enumerate(groups, 1):
            group.group_id = index
        return groups

    @staticmethod
    def _best_child(
        parent: PatchCandidate,
        candidates: Iterable[PatchCandidate],
        used: set,
    ) -> Optional[PatchCandidate]:
        matches = []
        for candidate in candidates:
            if (candidate.scale, candidate.slide_id, candidate.patch_index) in used:
                continue
            if candidate.slide_id.rsplit("_", 2)[0] != parent.slide_id.rsplit("_", 2)[0]:
                continue
            center = candidate.center
            if center is None or parent.x is None or parent.y is None:
                continue
            if parent.x <= center[0] <= parent.x + parent.size and parent.y <= center[1] <= parent.y + parent.size:
                matches.append(candidate)
        return max(matches, key=lambda item: item.score, default=None)

    @staticmethod
    def _group_score(patches: Dict[int, PatchCandidate]) -> float:
        source_count = len({
            (source["type"], source["name"])
            for patch in patches.values() for source in patch.sources
        })
        return float(sum(patch.score for patch in patches.values()) + 0.1 * len(patches) + 0.03 * source_count)


class WSICropper:
    def __init__(self, wsi_root: Path, output_root: Path):
        self.wsi_root = Path(wsi_root)
        self.output_root = Path(output_root)
        self._index: Optional[Dict[str, List[Path]]] = None

    def crop_groups(self, case_id: str, question: str, groups: List[EvidenceGroup]) -> List[EvidenceGroup]:
        try:
            import openslide
        except ImportError as error:
            raise RuntimeError("openslide-python is required to crop WSI evidence") from error
        question_id = hashlib.sha1(question.encode("utf-8")).hexdigest()[:10]
        directory = self.output_root / case_id / question_id
        directory.mkdir(parents=True, exist_ok=True)
        for group in groups:
            for scale, patch in group.patches.items():
                if patch.x is None or patch.y is None:
                    continue
                wsi_path = self._resolve_wsi(case_id, patch.slide_id)
                if wsi_path is None:
                    continue
                target = directory / f"group{group.group_id}_{scale}_{patch.x}_{patch.y}.jpg"
                with openslide.OpenSlide(str(wsi_path)) as slide:
                    image = slide.read_region((patch.x, patch.y), 0, (patch.size, patch.size)).convert("RGB")
                image.thumbnail((1024, 1024))
                image.save(target, quality=90)
                patch.image_path = str(target)
        return groups

    def _resolve_wsi(self, case_id: str, slide_id: str) -> Optional[Path]:
        if self._index is None:
            self._index = {}
            for path in self.wsi_root.rglob("*.svs"):
                self._index.setdefault(path.name[:12], []).append(path)
        candidates = self._index.get(case_id, [])
        if not candidates:
            return None
        feature_stem = slide_id.rsplit("_", 2)[0]
        exact = next((path for path in candidates if path.stem == feature_stem), None)
        return exact or candidates[0]
