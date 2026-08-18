from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class EvidenceObservation:
    round_index: int
    action: str
    evidence_type: str
    evidence_role: str
    scale: Optional[int]
    target_type: Optional[str]
    target_name: Optional[str]
    visual_description: str
    structured_support: Dict[str, Any] = field(default_factory=dict)
    group_ids: List[int] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class WorkingMemory:
    case_id: str
    question: str
    choices: List[str]
    plan: Dict[str, Any]
    knowledge: Dict[str, Any]
    structured_evidence: Dict[str, Any] = field(default_factory=dict)
    structured_candidate: Optional[Dict[str, Any]] = None
    structured_confidence: float = 0.0
    structured_reliability: float = 0.0
    option_alignment: Dict[str, Any] = field(default_factory=dict)
    direct_evidence_state: str = "unavailable"
    inspected_scales: List[int] = field(default_factory=list)
    observations: List[EvidenceObservation] = field(default_factory=list)
    direct_evidence: List[Dict[str, Any]] = field(default_factory=list)
    supportive_evidence: List[Dict[str, Any]] = field(default_factory=list)
    current_missing_evidence: List[str] = field(default_factory=list)
    missing_evidence_history: List[str] = field(default_factory=list)
    conflicts: List[str] = field(default_factory=list)
    inspected_programs: List[str] = field(default_factory=list)
    inspected_genes: List[str] = field(default_factory=list)
    action_history: List[Dict[str, Any]] = field(default_factory=list)
    final_verifier: Optional[Dict[str, Any]] = None

    def add_observation(self, observation: EvidenceObservation) -> None:
        self.observations.append(observation)
        if observation.scale in (1024, 2048, 4096):
            if observation.evidence_type in {"phenotype", "morphology"}:
                if observation.scale not in self.inspected_scales:
                    self.inspected_scales.append(observation.scale)
        row = observation.to_dict()
        if observation.evidence_role == "direct":
            self.direct_evidence.append(row)
        else:
            self.supportive_evidence.append(row)
        if observation.target_type == "program" and observation.target_name:
            if observation.target_name not in self.inspected_programs:
                self.inspected_programs.append(observation.target_name)
        if observation.target_type == "gene" and observation.target_name:
            if observation.target_name not in self.inspected_genes:
                self.inspected_genes.append(observation.target_name)

    def record_action(
        self,
        round_index: int,
        action: str,
        target: Optional[str] = None,
        verifier_fallback_used: bool = False,
        requested_action: Optional[str] = None,
        normalized_action: Optional[str] = None,
        fallback_reason: Optional[str] = None,
        target_resolution_fallback: bool = False,
    ) -> None:
        self.action_history.append({
            "round": int(round_index),
            "action": action,
            "requested_action": requested_action or action,
            "normalized_action": normalized_action or action,
            "executed_action": action,
            "target": target,
            "verifier_fallback_used": bool(verifier_fallback_used),
            "fallback_reason": fallback_reason,
            "target_resolution_fallback": bool(target_resolution_fallback),
        })

    def update_verifier(self, decision: Dict[str, Any]) -> None:
        missing = str(decision.get("missing_evidence_type") or "none")
        if missing in {"none", ""}:
            self.current_missing_evidence = []
        else:
            self.current_missing_evidence = [missing]
            self.missing_evidence_history.append(missing)
        if decision.get("conflict_detected"):
            reason = str(decision.get("reason") or "Evidence conflict detected.")
            if reason not in self.conflicts:
                self.conflicts.append(reason)
        self.final_verifier = dict(decision)

    @property
    def missing_evidence(self) -> List[str]:
        """Backward-compatible alias for the currently unresolved gap."""
        return self.current_missing_evidence

    def to_dict(self) -> Dict[str, Any]:
        return {
            "case_id": self.case_id,
            "question": self.question,
            "choices": list(self.choices),
            "plan": dict(self.plan),
            "knowledge": self.knowledge,
            "structured_evidence": self.structured_evidence,
            "structured_candidate": self.structured_candidate,
            "structured_confidence": self.structured_confidence,
            "structured_reliability": self.structured_reliability,
            "option_alignment": self.option_alignment,
            "direct_evidence_state": self.direct_evidence_state,
            "inspected_scales": list(self.inspected_scales),
            "observations": [item.to_dict() for item in self.observations],
            "direct_evidence": list(self.direct_evidence),
            "supportive_evidence": list(self.supportive_evidence),
            "missing_evidence": list(self.current_missing_evidence),
            "current_missing_evidence": list(self.current_missing_evidence),
            "missing_evidence_history": list(self.missing_evidence_history),
            "conflicts": list(self.conflicts),
            "inspected_programs": list(self.inspected_programs),
            "inspected_genes": list(self.inspected_genes),
            "action_history": list(self.action_history),
            "final_verifier": self.final_verifier,
        }
