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
VERIFIER_SYSTEM_PROMPT = """You are an evidence sufficiency controller for breast pathology WSI multiple-choice VQA.
You do not choose an option and must never output an answer_id. Judge whether accumulated evidence resolves the option-level distinction and choose exactly one supplied available action.
Evidence priority is direct structured phenotype prediction, then target-specific visible morphology, then supportive program evidence, then supportive gene evidence. A reliable direct prediction with complete option alignment may be answered at Round 0 without visual inspection. Program, gene, and weak visual evidence must not casually overwrite a direct candidate.
A strong structured prediction requires both strong patient-level evidence and sufficiently reliable Tool validation. High softmax or cross-scale agreement alone must not make a weakly validated Tool conclusive. Use reliability-adjusted confidence as evidence context, not as a hard escalation threshold.
A WSI-derived categorical ER/PR/HER2 prediction is valid evidence for a categorical benchmark target such as positive versus negative. It is not a measured assay and cannot answer an exact percentage, intensity, FISH/ISH ratio, amplification, or other assay-specific quantity.
Visual observations describe pixels only. Treat a visual conflict as real only when spatially linked observations address the same target with clearly mutually exclusive morphology. Limited, indeterminate, different-region, or different-slide observations are not conflicts. Program and gene evidence is supportive WSI-derived evidence, never measured RNA, protein, mutation, IHC, FISH, amplification, or copy number. Learned relations are predictive associations, not causality.
Each visual scale is complementary: 4096 supports global architecture, 2048 intermediate tissue organization, and 1024 fine morphology/cytology. Absence of a fine feature at 4096 is not strong negative evidence, and global architecture must not be inferred from one 1024 patch. RAG visual guidance says what is useful to inspect at a scale; it is not patient evidence.
After appropriate morphology inspection, unresolved weak structured evidence may be corroborated by constrained Program and then Gene candidates. Their relation relevance, patient score, graph score, and cross-scale support are supportive context only, especially for morphology-dominant targets.
Use unused evidence only when it can resolve a stated missing distinction. Exact assay values, exact size/distance/count, clinical history, treatment, procedure, report wording, and other unavailable facts cannot be recovered by escalating program or gene evidence.
Return JSON only with evidence_sufficient, evidence_state, missing_evidence_type, conflict_detected, next_action, target, and a concise reason."""


class EvidenceVerifierAgent:
    def __init__(self, client: OpenAICompatibleClient):
        self.client = client

    @staticmethod
    def available_actions(
        last_action: str,
        has_program_candidates: bool,
        has_gene_candidates: bool,
        allow_early_abstain: bool = False,
    ) -> List[str]:
        if last_action == "round0":
            actions = ["answer", "inspect_4096", "inspect_2048", "inspect_1024"]
            if allow_early_abstain:
                actions.append("abstain")
            return actions
        if last_action == "inspect_4096":
            actions = ["answer", "inspect_2048", "inspect_1024"]
            if allow_early_abstain:
                actions.append("abstain")
            return actions
        if last_action == "inspect_2048":
            actions = ["answer", "inspect_1024"]
            if allow_early_abstain:
                actions.append("abstain")
            return actions
        if last_action == "inspect_1024":
            actions = ["answer"]
            if has_program_candidates:
                actions.append("inspect_program")
            actions.append("abstain")
            return actions
        if last_action == "inspect_program":
            actions = ["answer"]
            if has_gene_candidates:
                actions.append("inspect_gene")
            actions.append("abstain")
            return actions
        if last_action == "inspect_gene":
            return ["answer", "abstain"]
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
        # Invalid model output must not silently turn the agent into a fixed
        # Program/Gene escalation chain. Prefer the already mapped direct
        # candidate; otherwise honor only an explicitly unresolved legal gap.
        has_direct_candidate = bool(
            memory
            and memory.structured_candidate
            and memory.option_alignment.get("mapping_complete") is True
        )
        missing_to_action = {
            "coarse_visual": "inspect_4096",
            "intermediate_visual": "inspect_2048",
            "fine_visual": "inspect_1024",
            "program_support": "inspect_program",
            "gene_support": "inspect_gene",
        }
        unresolved = (
            memory.current_missing_evidence[-1]
            if memory and memory.current_missing_evidence else None
        )
        required_action = missing_to_action.get(unresolved)
        if has_direct_candidate and "answer" in available_actions:
            action = "answer"
        elif required_action in available_actions:
            action = required_action
        elif "answer" in available_actions:
            action = "answer"
        elif "abstain" in available_actions:
            action = "abstain"
        else:
            action = available_actions[0]
        missing_map = {
            "inspect_2048": "intermediate_visual",
            "inspect_1024": "fine_visual",
            "inspect_program": "program_support",
            "inspect_gene": "gene_support",
            "abstain": "unavailable",
        }
        return {
            "evidence_sufficient": False,
            "evidence_state": "unavailable" if action == "abstain" else "insufficient",
            "missing_evidence_type": missing_map.get(action, "none"),
            "conflict_detected": False,
            "next_action": action,
            "target": None,
            "reason": reason,
            "verifier_fallback_used": True,
            "evidence_sufficiency_unverified": action == "answer",
            "requested_action": requested_action,
            "normalized_action": action,
            "executed_action": action,
            "fallback_reason": reason,
            "target_resolution_fallback": False,
        }
