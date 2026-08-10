from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ExecutionPlan:
    case_id: str
    question: str
    target_phenotypes: List[str]
    task_type: str
    metrics: List[str]
    answer_mode: str
    supported: bool
    support_reason: str
    task_match: str = "direct"
    phenotype_relevance_score: float = 1.0
    scale_order: List[int] = field(default_factory=lambda: [4096, 2048, 1024])
    use_pathology_agent: bool = True
    evidence_route: str = "phenotype_direct"
    selected_prototype_ids: List[str] = field(default_factory=list)
    prototype_coverage: str = "none"
    local_morphology_useful: bool = False
    requires_unavailable_context: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PatchCandidate:
    scale: int
    slide_id: str
    patch_index: int
    x: Optional[int]
    y: Optional[int]
    size: int
    score: float
    sources: List[Dict[str, Any]] = field(default_factory=list)
    feature: Optional[Any] = field(default=None, repr=False)
    image_path: Optional[str] = None

    @property
    def center(self):
        if self.x is None or self.y is None:
            return None
        return self.x + self.size / 2.0, self.y + self.size / 2.0

    def to_dict(self) -> Dict[str, Any]:
        value = asdict(self)
        value.pop("feature", None)
        return value


@dataclass
class EvidenceGroup:
    group_id: int
    score: float
    patches: Dict[int, PatchCandidate] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "group_id": self.group_id,
            "score": self.score,
            "patches": {str(k): v.to_dict() for k, v in self.patches.items()},
        }
