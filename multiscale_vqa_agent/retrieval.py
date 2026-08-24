import hashlib
import re
from copy import deepcopy
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
        self.morphology_context_per_slide = int(
            config.get("morphology_context_patches_per_slide", 2)
        )
        self.morphology_diverse_per_slide = int(
            config.get("morphology_diverse_patches_per_slide", 2)
        )
        self.question_top_per_slide = int(
            config.get("question_top_patches_per_slide", 4)
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
        groups = self._build_pyramids(by_scale)
        for group in groups:
            group.evidence_source = "selected_phenotype"
        return groups

    def retrieve_phenotype_only(
        self,
        field: str,
        scale_results: Dict[int, Dict[str, Any]],
        scale: int,
    ) -> List[EvidenceGroup]:
        """Retrieve one spatial scale without program or gene attention."""
        scale = self._require_scale(scale, scale_results)
        phenotype_index = self.registry.field_to_index[field]
        candidates = []
        for slide in scale_results[scale]["slides"]:
            candidates.extend(self._from_attention(
                scale,
                slide,
                "phenotype",
                [phenotype_index],
                [field],
                1.0,
            ))
        return self._single_scale_groups(
            self._deduplicate(candidates), scale, "selected_phenotype"
        )

    def retrieve_all_phenotypes_scale(
        self,
        scale_results: Dict[int, Dict[str, Any]],
        scale: int,
    ) -> List[EvidenceGroup]:
        """Retrieve broad morphology/context/diversity at exactly one scale."""
        scale = self._require_scale(scale, scale_results)
        indices = list(range(len(self.registry.phenotype_fields)))
        names = [
            f"{self.registry.field_to_prototype_id[field]}:{field}"
            for field in self.registry.phenotype_fields
        ]
        high_attention = []
        context = []
        diversity = []
        for slide in scale_results[scale]["slides"]:
            high_attention.extend(self._from_attention(
                scale,
                slide,
                "phenotype",
                indices,
                names,
                1.0,
                top_per_prototype=self.all_phenotype_top_per_prototype,
            ))
            context.extend(self._context_candidates(scale, slide))
            diversity.extend(self._feature_diversity_candidates(scale, slide))
        candidates = self._balanced_morphology_candidates(
            high_attention, context, diversity
        )
        return self._single_scale_groups(
            candidates, scale, "broad_phenotype"
        )

    def retrieve_question_scale(
        self,
        question_feature: np.ndarray,
        scale_results: Dict[int, Dict[str, Any]],
        scale: int,
    ) -> List[EvidenceGroup]:
        """Retrieve question-similar patches from exactly one spatial scale."""
        scale = self._require_scale(scale, scale_results)
        question_feature = np.asarray(question_feature, dtype=np.float32)
        if question_feature.ndim != 1:
            raise ValueError(
                f"Question feature must be 1D, got shape {question_feature.shape}"
            )
        norm = float(np.linalg.norm(question_feature))
        if not np.isfinite(norm) or norm <= 1e-8:
            raise ValueError("Question feature must be finite and non-zero")
        question_feature = question_feature / norm
        candidates = []
        for slide in scale_results[scale]["slides"]:
            candidates.extend(self._from_question_similarity(
                scale, slide, question_feature
            ))
        return self._single_scale_groups(
            self._deduplicate(candidates), scale, "question_similarity"
        )

    def retrieve_program_1024(
        self,
        program_index: int,
        scale_results: Dict[int, Dict[str, Any]],
    ) -> List[EvidenceGroup]:
        program_index = int(program_index)
        if not 0 <= program_index < len(self.registry.programs):
            raise IndexError(f"Invalid program index: {program_index}")
        scale = self._require_scale(1024, scale_results)
        candidates = []
        for slide in scale_results[scale]["slides"]:
            candidates.extend(self._from_attention(
                scale,
                slide,
                "program",
                [program_index],
                [self.registry.programs[program_index]],
                1.0,
            ))
        return self._single_scale_groups(
            self._deduplicate(candidates), scale, "program_support"
        )

    def retrieve_gene_1024(
        self,
        gene_index: int,
        scale_results: Dict[int, Dict[str, Any]],
    ) -> List[EvidenceGroup]:
        gene_index = int(gene_index)
        if not 0 <= gene_index < len(self.registry.genes):
            raise IndexError(f"Invalid gene index: {gene_index}")
        scale = self._require_scale(1024, scale_results)
        candidates = []
        for slide in scale_results[scale]["slides"]:
            candidates.extend(self._from_attention(
                scale,
                slide,
                "gene",
                [gene_index],
                [self.registry.genes[gene_index]],
                1.0,
            ))
        return self._single_scale_groups(
            self._deduplicate(candidates), scale, "gene_support"
        )

    def slice_groups_to_scale(
        self,
        groups: List[EvidenceGroup],
        scale: int,
        allowed_source_types: Optional[Iterable[str]] = None,
    ) -> List[EvidenceGroup]:
        """Copy one scale from existing groups with an optional source guard."""
        allowed = set(allowed_source_types or [])
        result = []
        for group in groups:
            patch = group.patches.get(int(scale))
            if patch is None:
                continue
            copied = deepcopy(patch)
            if allowed:
                copied.sources = [
                    source for source in copied.sources
                    if source.get("type") in allowed
                ]
                if not copied.sources:
                    continue
            result.append(EvidenceGroup(
                group_id=len(result) + 1,
                score=float(group.score),
                patches={int(scale): copied},
                evidence_source=group.evidence_source,
            ))
        return result[: self.max_groups]

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
            high_attention = []
            context = []
            diversity = []
            for slide in result["slides"]:
                high_attention.extend(self._from_attention(
                    scale,
                    slide,
                    "phenotype",
                    indices,
                    names,
                    1.0,
                    top_per_prototype=self.all_phenotype_top_per_prototype,
                ))
                context.extend(self._context_candidates(scale, slide))
                diversity.extend(self._feature_diversity_candidates(scale, slide))
            by_scale[scale] = self._balanced_morphology_candidates(
                high_attention,
                context,
                diversity,
            )
        groups = self._build_pyramids(by_scale)
        for group in groups:
            group.evidence_source = "broad_phenotype"
        return groups

    def retrieve_by_question(
        self,
        question_feature: np.ndarray,
        scale_results: Dict[int, Dict[str, Any]],
    ) -> List[EvidenceGroup]:
        question_feature = np.asarray(question_feature, dtype=np.float32)
        if question_feature.ndim != 1:
            raise ValueError(
                f"Question feature must be 1D, got shape {question_feature.shape}"
            )
        question_norm = float(np.linalg.norm(question_feature))
        if not np.isfinite(question_norm) or question_norm <= 1e-8:
            raise ValueError("Question feature must be finite and non-zero")
        question_feature = question_feature / question_norm

        by_scale: Dict[int, List[PatchCandidate]] = {}
        for scale, result in scale_results.items():
            candidates = []
            for slide in result["slides"]:
                candidates.extend(self._from_question_similarity(
                    scale, slide, question_feature
                ))
            by_scale[scale] = self._deduplicate(candidates)
        groups = self._build_pyramids(by_scale)
        for group in groups:
            group.evidence_source = "question_similarity"
        return groups

    def merge_hybrid_groups(
        self,
        question_groups: List[EvidenceGroup],
        prototype_groups: List[EvidenceGroup],
        question_limit: int = 3,
        prototype_limit: int = 2,
    ) -> List[EvidenceGroup]:
        """Combine question-first evidence with selected or broad support."""
        selected = deepcopy(question_groups[: max(0, int(question_limit))])
        for group in selected:
            group.evidence_source = "question_similarity"

        question_selected = list(selected)
        for prototype in deepcopy(
            prototype_groups[: max(0, int(prototype_limit))]
        ):
            support_source = prototype.evidence_source
            if support_source not in {"selected_phenotype", "broad_phenotype"}:
                support_source = "selected_phenotype"
            prototype.evidence_source = support_source
            duplicate = next((
                question_group for question_group in question_selected
                if self._groups_duplicate(question_group, prototype)
            ), None)
            if duplicate is None:
                selected.append(prototype)
                continue
            self._merge_group_sources(duplicate, prototype)
            duplicate.evidence_source = f"question_similarity+{support_source}"

        for index, group in enumerate(selected, 1):
            group.group_id = index
        return selected

    def merge_hybrid_scale_groups(
        self,
        question_groups: List[EvidenceGroup],
        phenotype_groups: List[EvidenceGroup],
    ) -> List[EvidenceGroup]:
        """Merge single-scale question and phenotype evidence."""
        return self.merge_hybrid_groups(
            question_groups,
            phenotype_groups,
            question_limit=3,
            prototype_limit=2,
        )[: self.max_groups]

    def link_child_groups(
        self,
        parent_groups: List[EvidenceGroup],
        child_groups: List[EvidenceGroup],
    ) -> List[EvidenceGroup]:
        """Keep only spatially linked children from the preceding visual round."""
        linked = []
        for child_group in child_groups:
            child = next(iter(child_group.patches.values()), None)
            if child is None:
                continue
            matches = []
            for parent_group in parent_groups:
                parent = next(iter(parent_group.patches.values()), None)
                if parent is None or not self._same_slide(parent, child):
                    continue
                relation = self._spatial_relation(parent, child)
                if relation is None:
                    continue
                priority = 1 if relation == "center_contained" else 0
                matches.append((priority, parent_group.score, parent_group, relation))
            if not matches:
                continue
            _, _, parent_group, relation = max(
                matches, key=lambda row: (row[0], row[1])
            )
            group = deepcopy(child_group)
            group.group_id = len(linked) + 1
            group.parent_group_id = parent_group.group_id
            group.anchor_group_id = (
                parent_group.anchor_group_id or parent_group.group_id
            )
            group.spatial_relation = relation
            linked.append(group)
            if len(linked) >= self.max_groups:
                break
        return linked

    @staticmethod
    def _same_slide(left: PatchCandidate, right: PatchCandidate) -> bool:
        return left.slide_id.rsplit("_", 2)[0] == right.slide_id.rsplit("_", 2)[0]

    @staticmethod
    def _spatial_relation(
        parent: PatchCandidate, child: PatchCandidate
    ) -> Optional[str]:
        if parent.x is None or parent.y is None or child.x is None or child.y is None:
            return None
        center = child.center
        if center and (
            parent.x <= center[0] <= parent.x + parent.size
            and parent.y <= center[1] <= parent.y + parent.size
        ):
            return "center_contained"
        x_overlap = min(parent.x + parent.size, child.x + child.size) - max(
            parent.x, child.x
        )
        y_overlap = min(parent.y + parent.size, child.y + child.size) - max(
            parent.y, child.y
        )
        return "bounding_box_overlap" if x_overlap > 0 and y_overlap > 0 else None

    def _groups_duplicate(
        self, left: EvidenceGroup, right: EvidenceGroup
    ) -> bool:
        for left_patch in left.patches.values():
            for right_patch in right.patches.values():
                if left_patch.slide_id != right_patch.slide_id:
                    continue
                if (
                    box_iou(left_patch, right_patch) >= self.iou_threshold
                    or cosine(left_patch.feature, right_patch.feature)
                    >= self.cosine_threshold
                ):
                    return True
        return False

    def _merge_group_sources(
        self, target: EvidenceGroup, duplicate: EvidenceGroup
    ) -> None:
        for scale, candidate in duplicate.patches.items():
            existing = target.patches.get(scale)
            if existing is None:
                target.patches[scale] = candidate
                continue
            known = {
                (source["type"], source["name"])
                for source in existing.sources
            }
            existing.sources.extend(
                source for source in candidate.sources
                if (source["type"], source["name"]) not in known
            )
            existing.score = max(existing.score, candidate.score)
        target.score = max(target.score, duplicate.score)

    def _from_question_similarity(
        self,
        scale: int,
        slide: Dict[str, Any],
        question_feature: np.ndarray,
    ) -> List[PatchCandidate]:
        features = np.asarray(slide["features"], dtype=np.float32)
        if features.ndim != 2:
            raise ValueError(
                f"Patch features must be 2D for scale={scale}, "
                f"slide={slide.get('slide_id')}: shape={features.shape}"
            )
        if features.shape[1] != question_feature.shape[0]:
            raise ValueError(
                "Patch/question feature dimension mismatch for "
                f"scale={scale}, slide={slide.get('slide_id')}: "
                f"patch_dim={features.shape[1]} question_dim={question_feature.shape[0]}"
            )
        if len(features) != len(slide["coords"]):
            raise ValueError(
                f"Patch feature/coordinate count mismatch for scale={scale}, "
                f"slide={slide.get('slide_id')}"
            )
        if self.question_top_per_slide <= 0 or not len(features):
            return []

        norms = np.linalg.norm(features, axis=1)
        valid = np.flatnonzero(
            np.isfinite(norms) & (norms > 1e-8) & np.isfinite(features).all(axis=1)
        )
        if not len(valid):
            return []
        unit_features = features[valid] / norms[valid, None]
        scores = unit_features @ question_feature
        count = min(self.question_top_per_slide, len(valid))
        order = np.argsort(scores)[::-1][:count]

        candidates = []
        for local_index in order.tolist():
            patch_index = int(valid[local_index])
            similarity = float(scores[local_index])
            x, y, size = slide["coords"][patch_index]
            candidates.append(PatchCandidate(
                scale=scale,
                slide_id=slide["slide_id"],
                patch_index=patch_index,
                x=x,
                y=y,
                size=size,
                score=similarity,
                sources=[{
                    "type": "question_similarity",
                    "name": "conch_v1_preprojection_768",
                    "attention": similarity,
                    "raw_similarity": similarity,
                    "retrieval_rank_score": similarity,
                    "score_semantics": "relative_retrieval_rank_only",
                    "similarity": similarity,
                }],
                feature=features[patch_index],
            ))
        return candidates

    def _context_candidates(
        self,
        scale: int,
        slide: Dict[str, Any],
    ) -> List[PatchCandidate]:
        count = self.morphology_context_per_slide
        attention = np.asarray(slide["phenotype_attention"], dtype=np.float32)
        features = np.asarray(slide["features"], dtype=np.float32)
        if count <= 0 or attention.ndim != 2 or not len(features):
            return []

        row_min = attention.min(axis=1, keepdims=True)
        row_range = attention.max(axis=1, keepdims=True) - row_min
        normalized = (attention - row_min) / (row_range + 1e-8)
        aggregate = normalized.max(axis=0)
        norms = np.linalg.norm(features, axis=1)
        valid = np.flatnonzero(
            np.isfinite(norms) & (norms > 1e-8) & np.isfinite(aggregate)
        )
        if not len(valid):
            return []
        selected = valid[np.argsort(aggregate[valid])[: min(count, len(valid))]]
        candidates = []
        for rank, patch_index in enumerate(selected.tolist()):
            x, y, size = slide["coords"][patch_index]
            candidates.append(PatchCandidate(
                scale=scale,
                slide_id=slide["slide_id"],
                patch_index=int(patch_index),
                x=x,
                y=y,
                size=size,
                score=float(0.90 - 0.03 * rank),
                sources=[{
                    "type": "context",
                    "name": "low_aggregate_phenotype_attention",
                    "attention": float(aggregate[patch_index]),
                    "raw_attention": float(aggregate[patch_index]),
                    "retrieval_rank_score": float(0.90 - 0.03 * rank),
                    "score_semantics": "relative_retrieval_rank_only",
                }],
                feature=features[patch_index],
            ))
        return candidates

    def _feature_diversity_candidates(
        self,
        scale: int,
        slide: Dict[str, Any],
    ) -> List[PatchCandidate]:
        count = self.morphology_diverse_per_slide
        features = np.asarray(slide["features"], dtype=np.float32)
        if count <= 0 or not len(features):
            return []

        norms = np.linalg.norm(features, axis=1)
        valid = np.flatnonzero(np.isfinite(norms) & (norms > 1e-8))
        if not len(valid):
            return []
        unit = features[valid] / norms[valid, None]
        center = unit.mean(axis=0)
        center_norm = np.linalg.norm(center)
        if center_norm > 1e-8:
            center = center / center_norm
            first = int(np.argmax(1.0 - unit @ center))
        else:
            first = 0

        selected_local = [first]
        min_distance = 1.0 - unit @ unit[first]
        min_distance[first] = -np.inf
        while len(selected_local) < min(count, len(valid)):
            next_local = int(np.argmax(min_distance))
            selected_local.append(next_local)
            distance = 1.0 - unit @ unit[next_local]
            min_distance = np.minimum(min_distance, distance)
            min_distance[selected_local] = -np.inf

        candidates = []
        for rank, local_index in enumerate(selected_local):
            patch_index = int(valid[local_index])
            x, y, size = slide["coords"][patch_index]
            candidates.append(PatchCandidate(
                scale=scale,
                slide_id=slide["slide_id"],
                patch_index=patch_index,
                x=x,
                y=y,
                size=size,
                score=float(0.86 - 0.03 * rank),
                sources=[{
                    "type": "diversity",
                    "name": "feature_farthest_point",
                    "attention": None,
                    "raw_attention": None,
                    "retrieval_rank_score": float(0.86 - 0.03 * rank),
                    "score_semantics": "relative_retrieval_rank_only",
                }],
                feature=features[patch_index],
            ))
        return candidates

    def _balanced_morphology_candidates(
        self,
        high_attention: List[PatchCandidate],
        context: List[PatchCandidate],
        diversity: List[PatchCandidate],
    ) -> List[PatchCandidate]:
        pools = {
            "high": self._deduplicate(high_attention),
            "context": self._deduplicate(context),
            "diversity": self._deduplicate(diversity),
        }
        positions = {name: 0 for name in pools}
        selected: List[PatchCandidate] = []
        target = max(self.max_groups * 3, 8)
        order = ("high", "context", "diversity", "high")

        while len(selected) < target:
            progress = False
            for name in order:
                pool = pools[name]
                while positions[name] < len(pool):
                    candidate = pool[positions[name]]
                    positions[name] += 1
                    duplicate = any(
                        item.slide_id == candidate.slide_id and (
                            box_iou(item, candidate) >= self.iou_threshold
                            or cosine(item.feature, candidate.feature) >= self.cosine_threshold
                        )
                        for item in selected
                    )
                    if duplicate:
                        continue
                    candidate.score = float(1.25 - 0.02 * len(selected))
                    selected.append(candidate)
                    progress = True
                    break
                if len(selected) >= target:
                    break
            if not progress:
                break
        return selected

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
                        "raw_attention": float(values[patch_index]),
                        "retrieval_rank_score": float(local_score),
                        "score_semantics": "relative_retrieval_rank_only",
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

    def _single_scale_groups(
        self,
        candidates: List[PatchCandidate],
        scale: int,
        evidence_source: str,
    ) -> List[EvidenceGroup]:
        groups = []
        for candidate in candidates[: self.max_groups]:
            groups.append(EvidenceGroup(
                group_id=len(groups) + 1,
                score=float(candidate.score),
                patches={int(scale): candidate},
                evidence_source=evidence_source,
            ))
        return groups

    @staticmethod
    def _require_scale(
        scale: int, scale_results: Dict[int, Dict[str, Any]]
    ) -> int:
        scale = int(scale)
        if scale not in scale_results:
            raise KeyError(f"Scale {scale} is not available")
        return scale

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

    def overview_thumbnails(
        self,
        case_id: str,
        size: int = 1536,
        max_slides: int = 2,
    ) -> List[str]:
        try:
            import openslide
        except ImportError:
            return []
        candidates = sorted(
            self._case_slides(case_id), key=self._slide_priority
        )[: max(0, int(max_slides))]
        directory = self.output_root / case_id / "overviews"
        paths = []
        for slide_path in candidates:
            target = directory / f"{slide_path.stem}.overview_{int(size)}.jpg"
            try:
                if not target.exists() or target.stat().st_size <= 0:
                    directory.mkdir(parents=True, exist_ok=True)
                    with openslide.OpenSlide(str(slide_path)) as slide:
                        image = slide.get_thumbnail((int(size), int(size))).convert("RGB")
                    image.save(target, "JPEG", quality=90)
                paths.append(str(target))
            except Exception:
                continue
        return paths

    @staticmethod
    def _slide_priority(path: Path):
        name = path.name.upper()
        priority = 0 if re.search(r"-DX\d", name) else 1 if "-TS" in name else 2
        return priority, name

    def _case_slides(self, case_id: str) -> List[Path]:
        if self._index is None:
            self._index = {}
            for path in self.wsi_root.rglob("*.svs"):
                self._index.setdefault(path.name[:12], []).append(path)
        return self._index.get(case_id, [])

    def _resolve_wsi(self, case_id: str, slide_id: str) -> Optional[Path]:
        candidates = self._case_slides(case_id)
        if not candidates:
            return None
        feature_stem = slide_id.rsplit("_", 2)[0]
        exact = next((path for path in candidates if path.stem == feature_stem), None)
        return exact or candidates[0]
