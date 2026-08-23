import json
from typing import Any, Dict, Iterable, List, Optional

from .agent_memory import WorkingMemory
from .clients import OpenAICompatibleClient, parse_json_response


EVIDENCE_STATES = {
    "sufficient", "partial", "conflicting", "insufficient", "unavailable"
}
MISSING_TYPES = {
    "coarse_visual", "intermediate_visual", "fine_visual",
    "program_support", "gene_support", "unavailable", "none",
}
VERIFIER_SYSTEM_PROMPT = """You are the evidence acquisition controller for breast pathology WSI multiple-choice VQA.
Every question ultimately receives one supplied option from a separate Final Fusion model. You do not decide whether to answer, do not abstain, and never output an answer_id. Judge the current evidence and choose whether another acquisition action is useful.

Evidence hierarchy: direct patient-specific measurement/observation; validated structured phenotype evidence for its intended categorical target; target-specific visible morphology; explicitly permitted visual proxy evidence; supportive Program evidence; supportive Gene evidence; general knowledge and generic examples. Never promote weaker evidence into stronger evidence.

A WSI-derived categorical ER/PR/HER2 prediction may support that categorical target, but is not a measured assay: it cannot establish an exact percentage, measured IHC intensity, FISH/ISH ratio, amplification, copy number, or protein result. Patho-R1 is fallible visible-pixel evidence: an unlabeled edge is not a surgical margin, a local crop does not establish whole-specimen size, and breast-primary morphology does not establish nodal or distant status without represented tissue/context. Program and Gene signals are supportive WSI-derived predictions, never measured assays, treatment, procedure, history, gross measurement, or report metadata. Knowledge limitations, proxy rules, forced-choice rules, and generic examples are reasoning constraints, not patient facts.

Inspect another source only when it can plausibly resolve a concrete option-level distinction. If current evidence resolves it, choose answer with evidence_sufficient=true. If useful evidence remains, inspect it. If no remaining WSI/Program/Gene action can validly recover the missing fact, choose finalize with evidence_sufficient=false and search_exhausted=true. Finalize stops search and sends current evidence to Final Fusion; it never means abstain. Do not manufacture evidence to avoid finalize.

Return JSON only with evidence_sufficient, evidence_state, missing_evidence_type, conflict_detected, next_action, target, search_exhausted, and reason."""


class EvidenceVerifierAgent:
    def __init__(self, client: OpenAICompatibleClient):
        self.client = client

    @staticmethod
    def available_actions(
        last_action: str,
        has_program_candidates: bool,
        has_gene_candidates: bool,
    ) -> List[str]:
        if last_action == "round0":
            return ["answer", "inspect_4096", "inspect_2048", "inspect_1024", "finalize"]
        if last_action == "inspect_4096":
            return ["answer", "inspect_2048", "inspect_1024", "finalize"]
        if last_action == "inspect_2048":
            return ["answer", "inspect_1024", "finalize"]
        if last_action == "inspect_1024":
            actions = ["answer"]
            if has_program_candidates:
                actions.append("inspect_program")
            actions.append("finalize")
            return actions
        if last_action == "inspect_program":
            actions = ["answer"]
            if has_gene_candidates:
                actions.append("inspect_gene")
            actions.append("finalize")
            return actions
        if last_action == "inspect_gene":
            return ["answer", "finalize"]
        raise ValueError(f"Unsupported verifier state after {last_action!r}")

    def decide(
        self,
        question: str,
        choices: Iterable[str],
        plan: Dict[str, Any],
        knowledge: Dict[str, Any],
        memory: WorkingMemory,
        available_actions: List[str],
        program_candidates: Optional[List[Dict[str, Any]]] = None,
        gene_candidates: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        payload = {
            "question": question,
            "choices": list(choices),
            "plan": {
                "task_match": plan.get("task_match"),
                "evidence_route": plan.get("evidence_route"),
                "target_phenotypes": plan.get("target_phenotypes", []),
                "prototype_coverage": plan.get("prototype_coverage"),
                "requires_unavailable_context": plan.get(
                    "requires_unavailable_context", False
                ),
            },
            "knowledge": {
                "matched_concepts": knowledge.get("matched_concepts", []),
                "limitations": knowledge.get("limitations", []),
                "evidence_rules": knowledge.get("evidence_rules", []),
                "evidence_limitations": knowledge.get("evidence_limitations", [])[:5],
                "proxy_evidence_rules": knowledge.get("proxy_evidence_rules", [])[:5],
                "forced_choice_rules": knowledge.get("forced_choice_rules", [])[:12],
                "reasoning_examples": knowledge.get("reasoning_examples", [])[:3],
                "scale_specific_visual_guidance": knowledge.get(
                    "scale_specific_visual_guidance", {}
                ),
            },
            "working_memory": self._compact_memory(memory),
            "available_actions": list(available_actions),
            "program_candidates": list(program_candidates or []),
            "gene_candidates": list(gene_candidates or []),
            "output_schema": {
                "evidence_sufficient": False,
                "evidence_state": "sufficient|partial|conflicting|insufficient|unavailable",
                "missing_evidence_type": "coarse_visual|intermediate_visual|fine_visual|program_support|gene_support|unavailable|none",
                "conflict_detected": False,
                "next_action": "one supplied available action",
                "target": None,
                "search_exhausted": False,
                "reason": "one concise sentence",
            },
        }
        if not self.client or not self.client.enabled:
            return self._fallback(
                available_actions, "Verifier client is disabled.", memory
            )
        try:
            raw = self.client.chat(
                VERIFIER_SYSTEM_PROMPT,
                json.dumps(payload, ensure_ascii=False),
                temperature=0.0,
                max_tokens=650,
                response_format={"type": "json_object"},
                retries=2,
                enable_thinking=False,
            )
            parsed = parse_json_response(raw)
        except Exception as error:
            return self._fallback(
                available_actions,
                f"Verifier request failed: {type(error).__name__}.",
                memory,
            )
        return self._normalize(
            parsed,
            available_actions,
            program_candidates or [],
            gene_candidates or [],
            memory,
        )

    @staticmethod
    def _compact_memory(memory: WorkingMemory) -> Dict[str, Any]:
        return {
            "structured_evidence": EvidenceVerifierAgent._compact_structured(
                memory.structured_evidence
            ),
            "structured_candidate": memory.structured_candidate,
            "structured_confidence": memory.structured_confidence,
            "structured_reliability": memory.structured_reliability,
            "option_alignment": memory.option_alignment,
            "direct_evidence_state": memory.direct_evidence_state,
            "observations": [
                {
                    "round": row.round_index,
                    "action": row.action,
                    "evidence_role": row.evidence_role,
                    "scale": row.scale,
                    "target_type": row.target_type,
                    "visual_description": row.visual_description[:1000],
                    "structured_support": row.structured_support,
                }
                for row in memory.observations
            ],
            "current_missing_evidence": list(memory.current_missing_evidence),
            "missing_evidence_history": list(memory.missing_evidence_history),
            "conflicts": list(memory.conflicts),
            "inspected_scales": list(memory.inspected_scales),
            "inspected_programs": list(memory.inspected_programs),
            "inspected_genes": list(memory.inspected_genes),
        }

    @staticmethod
    def _compact_structured(structured: Dict[str, Any]) -> Dict[str, Any]:
        prediction_keys = (
            "field", "predicted_label",
            "fused_probability_for_predicted_class",
            "cross_scale_agreement", "validation_quality",
            "patient_evidence_strength", "reliability_adjusted_confidence",
        )
        return {
            "task_match": structured.get("task_match"),
            "prototype_coverage": structured.get("prototype_coverage"),
            "target_coverage": structured.get("target_coverage"),
            "predictions": [
                {key: row.get(key) for key in prediction_keys}
                for row in structured.get("predictions", [])
            ],
            "structured_candidate_id": structured.get(
                "structured_candidate_id"
            ),
            "structured_candidate_answer": structured.get(
                "structured_candidate_answer"
            ),
            "structured_candidate_confidence": structured.get(
                "structured_candidate_confidence"
            ),
            "overall_structured_reliability": structured.get(
                "overall_structured_reliability"
            ),
            "validation_reliability_source": structured.get(
                "validation_reliability_source"
            ),
            "reliability_adjusted_confidence": structured.get(
                "reliability_adjusted_confidence"
            ),
            "joint_fields": structured.get("joint_fields", []),
            "joint_state": structured.get("joint_state", {}),
            "joint_mapping_complete": structured.get(
                "joint_mapping_complete", False
            ),
            "option_alignment": {
                key: structured.get("option_alignment", {}).get(key)
                for key in ("source", "choice_id", "mapping_complete", "confidence")
            },
        }

    def _normalize(
        self,
        parsed: Any,
        available_actions: List[str],
        program_candidates: List[Dict[str, Any]],
        gene_candidates: List[Dict[str, Any]],
        memory: Optional[WorkingMemory] = None,
    ) -> Dict[str, Any]:
        if not isinstance(parsed, dict):
            return self._fallback(
                available_actions, "Verifier returned invalid JSON.", memory
            )
        state = str(parsed.get("evidence_state", "insufficient")).lower()
        if state not in EVIDENCE_STATES:
            state = "insufficient"
        missing = str(parsed.get("missing_evidence_type", "none")).lower()
        if missing not in MISSING_TYPES:
            missing = "none"
        action = str(parsed.get("next_action", "")).lower()
        requested_action = action
        sufficient = parsed.get("evidence_sufficient") is True
        if action == "answer" and not sufficient:
            return self._fallback(
                available_actions,
                "Verifier requested answer without sufficient evidence.",
                memory,
                requested_action=requested_action,
            )
        if action == "answer" and sufficient:
            state = "sufficient"
            missing = "none"
        if action == "finalize":
            sufficient = False
        if action not in available_actions:
            return self._fallback(
                available_actions,
                "Verifier requested an unavailable action.",
                memory,
                requested_action=requested_action,
            )
        target = parsed.get("target")
        candidates = (
            program_candidates if action == "inspect_program"
            else gene_candidates if action == "inspect_gene" else []
        )
        target_resolution_fallback = False
        if candidates:
            resolved = self._resolve_target(target, candidates)
            if resolved is None:
                target = candidates[0].get("name")
                target_resolution_fallback = True
            else:
                target = resolved
        elif action in {"inspect_program", "inspect_gene"}:
            return self._fallback(
                available_actions,
                "No constrained target was available.",
                memory,
                requested_action=requested_action,
            )
        else:
            target = None
            target_resolution_fallback = False
        return {
            "evidence_sufficient": sufficient,
            "evidence_state": state,
            "missing_evidence_type": missing,
            "conflict_detected": parsed.get("conflict_detected") is True,
            "next_action": action,
            "target": target,
            "search_exhausted": action == "finalize",
            "reason": " ".join(str(parsed.get("reason") or "").split())[:500],
            "verifier_fallback_used": False,
            "evidence_sufficiency_unverified": False,
            "requested_action": requested_action,
            "normalized_action": action,
            "executed_action": action,
            "fallback_reason": None,
            "target_resolution_fallback": target_resolution_fallback,
        }

    @staticmethod
    def _resolve_target(
        target: Any, candidates: List[Dict[str, Any]]
    ) -> Optional[str]:
        if isinstance(target, dict):
            target = target.get("name", target.get("index"))
        normalized_target = " ".join(str(target or "").lower().split())
        for row in candidates:
            normalized_name = " ".join(str(row.get("name") or "").lower().split())
            if (
                str(target or "") == str(row.get("index"))
                or normalized_target == normalized_name
            ):
                return str(row.get("name"))
        return None

    @staticmethod
    def _fallback(
        available_actions: List[str],
        reason: str,
        memory: Optional[WorkingMemory] = None,
        requested_action: Optional[str] = None,
    ) -> Dict[str, Any]:
        # Invalid model output must not certify evidence or force molecular
        # escalation. Stop acquisition and let deterministic Fusion recover.
        missing_to_action = {
            "coarse_visual": "inspect_4096",
            "intermediate_visual": "inspect_2048",
            "fine_visual": "inspect_1024",
        }
        unresolved = (
            memory.current_missing_evidence[-1]
            if memory and memory.current_missing_evidence else None
        )
        required_action = missing_to_action.get(unresolved)
        if required_action in available_actions:
            action = required_action
        elif "finalize" in available_actions:
            action = "finalize"
        else:
            action = available_actions[0]
        missing_map = {
            "inspect_2048": "intermediate_visual",
            "inspect_1024": "fine_visual",
            "inspect_program": "program_support",
            "inspect_gene": "gene_support",
            "finalize": "unavailable",
        }
        return {
            "evidence_sufficient": False,
            "evidence_state": "unavailable" if action == "finalize" else "insufficient",
            "missing_evidence_type": missing_map.get(action, "none"),
            "conflict_detected": False,
            "next_action": action,
            "target": None,
            "search_exhausted": action == "finalize",
            "reason": reason,
            "verifier_fallback_used": True,
            "evidence_sufficiency_unverified": True,
            "requested_action": requested_action,
            "normalized_action": action,
            "executed_action": action,
            "fallback_reason": reason,
            "target_resolution_fallback": False,
        }
