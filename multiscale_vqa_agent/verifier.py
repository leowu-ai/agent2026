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
ACTION_SEQUENCE = {
    "inspect_4096": "inspect_2048",
    "inspect_2048": "inspect_1024",
    "inspect_1024": "inspect_program",
    "inspect_program": "inspect_gene",
    "inspect_gene": "abstain",
}


VERIFIER_SYSTEM_PROMPT = """You are an evidence sufficiency controller for breast pathology WSI multiple-choice VQA.
You do not choose an option and must never output an answer_id. Judge whether accumulated evidence resolves the option-level distinction and choose exactly one supplied available action.
Direct phenotype predictions address only their trained target. Visual observations describe pixels. Program and gene evidence is supportive WSI-derived evidence, never measured RNA, protein, mutation, IHC, FISH, amplification, or copy number. Learned relations are predictive associations, not causality.
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
    ) -> List[str]:
        if last_action == "inspect_4096":
            return ["answer", "inspect_2048", "abstain"]
        if last_action == "inspect_2048":
            return ["answer", "inspect_1024", "abstain"]
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
            return self._fallback(available_actions, "Verifier client is disabled.")
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
            )
        return self._normalize(
            parsed,
            available_actions,
            program_candidates or [],
            gene_candidates or [],
        )

    @staticmethod
    def _compact_memory(memory: WorkingMemory) -> Dict[str, Any]:
        return {
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
            "missing_evidence": list(memory.missing_evidence),
            "conflicts": list(memory.conflicts),
            "inspected_scales": list(memory.inspected_scales),
            "inspected_programs": list(memory.inspected_programs),
            "inspected_genes": list(memory.inspected_genes),
        }

    def _normalize(
        self,
        parsed: Any,
        available_actions: List[str],
        program_candidates: List[Dict[str, Any]],
        gene_candidates: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        if not isinstance(parsed, dict):
            return self._fallback(available_actions, "Verifier returned invalid JSON.")
        state = str(parsed.get("evidence_state", "insufficient")).lower()
        if state not in EVIDENCE_STATES:
            state = "insufficient"
        missing = str(parsed.get("missing_evidence_type", "none")).lower()
        if missing not in MISSING_TYPES:
            missing = "none"
        action = str(parsed.get("next_action", "")).lower()
        sufficient = parsed.get("evidence_sufficient") is True
        if action == "answer" and not sufficient:
            return self._fallback(
                available_actions,
                "Verifier requested answer without sufficient evidence.",
            )
        if action == "answer" and sufficient:
            state = "sufficient"
            missing = "none"
        if action not in available_actions:
            return self._fallback(available_actions, "Verifier requested an unavailable action.")
        target = parsed.get("target")
        candidates = (
            program_candidates if action == "inspect_program"
            else gene_candidates if action == "inspect_gene" else []
        )
        if candidates:
            resolved = self._resolve_target(target, candidates)
            if resolved is None:
                result = self._fallback(
                    available_actions,
                    "Verifier target was not in the constrained candidate list.",
                )
                if result["next_action"] == action:
                    result["target"] = candidates[0].get("name")
                return result
            target = resolved
        elif action in {"inspect_program", "inspect_gene"}:
            return self._fallback(available_actions, "No constrained target was available.")
        else:
            target = None
        return {
            "evidence_sufficient": sufficient,
            "evidence_state": state,
            "missing_evidence_type": missing,
            "conflict_detected": parsed.get("conflict_detected") is True,
            "next_action": action,
            "target": target,
            "reason": " ".join(str(parsed.get("reason") or "").split())[:500],
            "verifier_fallback_used": False,
        }

    @staticmethod
    def _resolve_target(
        target: Any, candidates: List[Dict[str, Any]]
    ) -> Optional[str]:
        if isinstance(target, dict):
            target = target.get("name", target.get("index"))
        for row in candidates:
            if (
                str(target or "") == str(row.get("index"))
                or str(target or "") == str(row.get("name"))
            ):
                return str(row.get("name"))
        return None

    @staticmethod
    def _fallback(available_actions: List[str], reason: str) -> Dict[str, Any]:
        next_evidence = next(
            (action for action in available_actions if action.startswith("inspect_")),
            None,
        )
        action = next_evidence or ("abstain" if "abstain" in available_actions else available_actions[0])
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
        }
