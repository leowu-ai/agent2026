import json
import re
from typing import Any, Dict, List, Optional, Tuple

from .clients import OpenAICompatibleClient, parse_json_response
from .fusion_evidence import (
    UNAVAILABLE_PATTERNS,
    build_structured_summary,
    clinical_display_label,
    choice_id_for_answer,
    indexed_choices,
    load_fusion_prompt,
)
from .schemas import ExecutionPlan


MINIMAL_NONE_SYSTEM = """You answer breast pathology multiple-choice questions using broad WSI-derived context.
Select exactly one supplied option ID. Do not return null, refuse, or invent measurements.
There is no Router-selected direct phenotype prototype. Broad G2P phenotype predictions are contextual WSI-derived predictions, not measured assays and not direct target evidence. Patho-R1 is a fallible summary of directly visible morphology. Combine broad G2P context with visual morphology and choose the most defensible option. Do not treat an unrelated phenotype prediction as a measurement of the requested target.
If evidence is weak, choose with low confidence and say it is not confirmed.
Output only JSON: {\"answer_id\":\"<supplied option ID>\",\"confidence\":0.0,\"explanation\":\"one sentence\",\"limitations\":\"one sentence\"}"""


REPAIR_SYSTEM = """Repair a breast pathology MCQ answer. Output only short valid JSON.
The answer_id must exactly equal one supplied option ID. Do not return null or markdown.
Schema: {\"answer_id\":\"<supplied option ID>\",\"confidence\":0.0,\"explanation\":\"one sentence\",\"limitations\":\"one sentence\"}"""


ALIGNMENT_SYSTEM = """You align structured breast pathology predictions to multiple-choice options.
This is semantic option alignment, not diagnosis. Use only the question, clinical predicted labels, field meanings, and supplied choices.
Predicted labels are clinical values, never zero-based class indices. A primary field can establish a candidate by itself; missing supporting fields only lower mapping confidence.
Allow established clinical synonyms such as infiltrating=invasive. Never equate invasive with in situ, carcinoma with hyperplasia, or negative with no unless the question asks presence/status.
Return null when the prediction does not answer the question or an option requires unsupported extra facts.
Output only JSON with choice_id, mapping_complete, confidence, and a one-sentence reason."""


MORPHOLOGY_OVERRIDE_FIELDS = {
    "histological_type_label",
    "histologic_grade_label",
    "lymphovascular_invasion_label",
}


class FusionVerificationAgent:
    def __init__(self, client: OpenAICompatibleClient):
        self.client = client

    def answer(
        self,
        plan: ExecutionPlan,
        choices: List[str],
        phenotype_results: Any,
        relations: Any,
        pathology: Dict[str, Any],
        broad_g2p_predictions: Any = None,
    ) -> Dict[str, Any]:
        answer, _ = self.answer_with_summary(
            plan, choices, phenotype_results, relations, pathology,
            broad_g2p_predictions=broad_g2p_predictions,
        )
        return answer

    def answer_with_summary(
        self,
        plan: ExecutionPlan,
        choices: List[str],
        phenotype_results: Any,
        relations: Any,
        pathology: Dict[str, Any],
        broad_g2p_predictions: Any = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        structured = build_structured_summary(plan, choices, phenotype_results)
        self._attach_option_alignment(plan, choices, structured)
        answer = self._answer_prepared(
            plan, choices, structured, relations, pathology,
            broad_g2p_predictions,
        )
        return answer, structured

    def _answer_prepared(
        self,
        plan: ExecutionPlan,
        choices: List[str],
        structured: Dict[str, Any],
        relations: Any,
        pathology: Dict[str, Any],
        broad_g2p_predictions: Any = None,
    ) -> Dict[str, Any]:
        evidence = self._build_evidence_packet(
            plan, choices, structured, relations, pathology,
            broad_g2p_predictions,
        )
        system_prompt = MINIMAL_NONE_SYSTEM if structured.get("task_match") == "none" else load_fusion_prompt()
        if not self.client.enabled:
            return self._fallback(plan, choices, structured, None, "mock_fallback", 0)

        raw = None
        retry_count = 0
        try:
            raw = self.client.chat(
                system_prompt,
                json.dumps(evidence, ensure_ascii=False),
                temperature=0.6,
                max_tokens=4096,
                response_format={"type": "json_object"},
                retries=2,
                enable_thinking=True,
                top_p=0.95,
                top_k=20,
            )
            parsed = parse_json_response(raw)
            result = self._validate(parsed, choices, structured, raw, "parsed", retry_count)
            if result is not None:
                return result

            recovered = self._recover_answer_id(raw, choices)
            if recovered is not None:
                parsed = {
                    "answer_id": recovered,
                    "confidence": min(float(structured.get("structured_candidate_confidence") or 0.25), 0.55),
                    "explanation": "Answer recovered from a malformed model response.",
                    "limitations": "The original response was not valid JSON.",
                }
                return self._validate(parsed, choices, structured, raw, "recovered_answer", retry_count)

            retry_count = 1
            retry_prompt = json.dumps({
                "instruction": "Return only the required JSON object. Re-decide from this compact original context, not from the malformed answer.",
                "question": plan.question,
                "choices": indexed_choices(choices),
                "task_match": structured.get("task_match"),
                "evidence_route": evidence.get("evidence_route"),
                "broad_g2p_predictions": evidence.get("broad_g2p_predictions", []),
                "structured_candidate": evidence.get("structured_candidate"),
                "structured_evidence_summary": {
                    "answer_unit": evidence.get("answer_unit"),
                    "primary_predictions": evidence.get("primary_predictions", []),
                    "supporting_predictions": evidence.get("supporting_predictions", []),
                    "option_compatibility": evidence.get("option_compatibility", []),
                    "literal_match": evidence.get("literal_match", {}),
                    "conflicts": evidence.get("conflicts", []),
                },
                "visual_evidence_summary": evidence.get("visual_observations", evidence.get("available_visual_summary", "")),
                "output_schema": {
                    "answer_id": "<one supplied option ID>",
                    "confidence": 0.0,
                    "explanation": "<one sentence>",
                    "limitations": "<one sentence>",
                },
            }, ensure_ascii=False)
            retry_raw = self.client.chat(
                REPAIR_SYSTEM,
                retry_prompt,
                temperature=0.0,
                max_tokens=260,
                response_format={"type": "json_object"},
                retries=2,
                enable_thinking=False,
            )
            parsed = parse_json_response(retry_raw)
            result = self._validate(
                parsed,
                choices,
                structured,
                retry_raw,
                "retry_parsed",
                retry_count,
                initial_raw=raw,
            )
            if result is not None:
                return result
            raw = retry_raw or raw
        except Exception as error:
            fallback = self._fallback(plan, choices, structured, raw, "request_error", retry_count)
            fallback["limitations"] = self._limit(
                f"{fallback['limitations']} Qwen request failed: {type(error).__name__}.", 350
            )
            return fallback

        return self._fallback(
            plan, choices, structured, raw, "fallback_after_parse_failure", retry_count
        )

    def _build_evidence_packet(
        self,
        plan: ExecutionPlan,
        choices: List[str],
        structured: Dict[str, Any],
        relations: Any,
        pathology: Dict[str, Any],
        broad_g2p_predictions: Any = None,
    ) -> Dict[str, Any]:
        visual = self._visual_summary(pathology.get("description"))
        if structured.get("task_match") == "none":
            return {
                "question": plan.question,
                "choices": indexed_choices(choices),
                "task_match": "none",
                "evidence_route": getattr(plan, "evidence_route", "morphology_only"),
                "selected_prototype_ids": list(getattr(plan, "selected_prototype_ids", [])),
                "broad_g2p_predictions": list(broad_g2p_predictions or []),
                "available_visual_summary": visual,
                "evidence_availability": "broad_g2p_plus_visual",
            }
        primary_fields = set(structured.get("primary_fields", []))
        supporting_fields = set(structured.get("supporting_fields", []))
        predictions = structured.get("predictions", [])
        primary_predictions = [self._compact_prediction(row) for row in predictions if row.get("field") in primary_fields]
        supporting_predictions = [self._compact_prediction(row) for row in predictions if row.get("field") in supporting_fields]
        if not primary_predictions and predictions:
            primary_predictions = [self._compact_prediction(predictions[0])]
        option_rows = []
        for row in structured.get("option_compatibility", []):
            option_rows.append({
                "choice_id": row.get("choice_id"),
                "choice_text": row.get("choice"),
                "requirements": row.get("requirements", {}),
                "primary_requirements": row.get("primary_requirements", {}),
                "supporting_requirements": row.get("supporting_requirements", {}),
                "supported_fields": row.get("supported_fields", []),
                "missing_primary_fields": row.get("missing_primary_fields", []),
                "missing_supporting_fields": row.get("missing_supporting_fields", []),
                "contradicted_primary_fields": row.get("contradicted_primary_fields", []),
                "contradicted_supporting_fields": row.get("contradicted_supporting_fields", []),
                "uncertain_fields": row.get("uncertain_fields", []),
                "field_coverage": row.get("field_coverage", 0.0),
                "evidence_coverage": row.get("evidence_coverage", 0.0),
            })
        return {
            "question": plan.question,
            "choices": indexed_choices(choices),
            "task_match": structured.get("task_match"),
            "evidence_route": structured.get("evidence_route"),
            "selected_prototype_ids": structured.get("selected_prototype_ids", []),
            "answer_unit": structured.get("answer_unit"),
            "structured_candidate": {
                "choice_id": structured.get("structured_candidate_id"),
                "choice_text": structured.get("structured_candidate_answer"),
                "confidence": structured.get("structured_candidate_confidence"),
                "mapping_complete": structured.get("mapping_complete"),
                "alignment": structured.get("option_alignment", {}),
            },
            "primary_predictions": primary_predictions,
            "supporting_predictions": supporting_predictions,
            "option_compatibility": option_rows[:8],
            "literal_match": {
                "choice_id": structured.get("literal_match_id"),
                "matches": structured.get("literal_matches", []),
                "rule": structured.get("literal_match_rule"),
                "advisory_only": True,
            },
            "visual_observations": visual,
            "conflicts": [],
            "relation_summary": self._relation_summary(relations),
            "rules": [
                "clinical_predicted_label is a clinical value; never reinterpret it as a class index.",
                "A unique literal_match choice is a strong advisory hint, not an absolute answer.",
                "Patho-R1 can overturn structured evidence only with direct visible counterevidence.",
                "WSI-inferred gene/pathway scores are not measured RNA, IHC, FISH, mutation, or protein.",
            ],
            "code_generated_base_confidence": structured.get("structured_candidate_confidence"),
        }

    @staticmethod
    def _compact_prediction(row: Dict[str, Any]) -> Dict[str, Any]:
        field = row.get("field")
        label = clinical_display_label(field, row.get("predicted_label"))
        return {
            "prototype_id": row.get("prototype_id"),
            "field": field,
            "clinical_predicted_label": label,
            "probability": row.get("fused_probability_for_predicted_class"),
            "scale_agreement": row.get("cross_scale_agreement"),
            "validation_quality": row.get("validation_quality"),
            "selected_label_definition": {
                "field": field,
                "clinical_value": label,
                "is_internal_class_index": False,
            },
        }

    @staticmethod
    def _visual_summary(description: Any) -> str:
        text = str(description or "")
        text = re.sub(r"<think>.*?</think>", " ", text, flags=re.I | re.S)
        text = re.sub(r"\s+", " ", text).strip()
        return text[:1200]

    @staticmethod
    def _relation_summary(relations: Any) -> str:
        if not relations:
            return "No relation evidence is independently diagnostic."
        if isinstance(relations, dict):
            keys = list(relations.keys())[:4]
            return f"Relation tables are available for consistency only, not independently diagnostic; fields: {keys}."
        return "Relation evidence is available for consistency only, not independently diagnostic."

    def _validate(
        self,
        parsed: Optional[Dict[str, Any]],
        choices: List[str],
        structured: Dict[str, Any],
        raw: Optional[str],
        status: str,
        retry_count: int,
        initial_raw: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        if not isinstance(parsed, dict) or not isinstance(parsed.get("answer_id"), str):
            return None
        answer_id = parsed["answer_id"].strip().upper()
        options = {option["id"]: option["text"] for option in indexed_choices(choices)}
        if answer_id not in options:
            return None
        answer = options[answer_id]
        base = float(structured.get("structured_candidate_confidence") or 0.0)
        try:
            confidence = float(parsed.get("confidence", base))
        except (TypeError, ValueError):
            confidence = base
        confidence = max(0.0, min(confidence, 1.0))
        if structured.get("task_match") == "none":
            confidence = min(confidence, 0.35)
        elif base > 0:
            confidence = max(max(0.0, base - 0.2), min(confidence, min(1.0, base + 0.2)))
        candidate_id = structured.get("structured_candidate_id")
        proposed_override = bool(candidate_id and answer_id != candidate_id)
        override_rejected = False
        if (
            proposed_override
            and structured.get("task_match") == "direct"
            and not self._valid_counterevidence(parsed, structured)
        ):
            answer_id = candidate_id
            answer = options[candidate_id]
            confidence = max(confidence, max(0.0, base - 0.1))
            override_rejected = True
        override = proposed_override and not override_rejected
        limitations = parsed.get("limitations", "")
        if isinstance(limitations, list):
            limitations = " ".join(str(value) for value in limitations)
        explanation = self._limit(parsed.get("explanation", ""), 600)
        if override_rejected:
            explanation = "Retained the direct structured candidate because the proposed visual override lacked validated decisive counterevidence."
            limitations = f"Visual conflict was reported but did not satisfy override evidence requirements. {limitations}"
        result = {
            "answer_id": answer_id,
            "answer": answer,
            "confidence": round(confidence, 6),
            "explanation": explanation,
            "limitations": self._limit(limitations, 350),
            "raw_response": raw,
            "parse_status": status,
            "json_parse_success": status in {"parsed", "retry_parsed"},
            "retry_count": retry_count,
            "answer_in_choices": True,
            "override_occurred": override,
            "override_proposed": proposed_override,
            "override_rejected": override_rejected,
            "counterevidence": parsed.get("counterevidence"),
            "option_alignment": structured.get("option_alignment", {}),
            "override_reason": (
                self._limit(parsed.get("explanation", ""), 240) if override else None
            ),
            "structured_visual_conflict": override,
        }
        if initial_raw is not None:
            result["initial_raw_response"] = initial_raw
        return result

    def _fallback(
        self,
        plan: ExecutionPlan,
        choices: List[str],
        structured: Dict[str, Any],
        raw: Optional[str],
        status: str,
        retry_count: int,
    ) -> Dict[str, Any]:
        answer = structured.get("structured_candidate_answer")
        if answer not in choices:
            answer = self._unsupported_choice(plan.question, choices)
        answer_id = choice_id_for_answer(choices, answer)
        return {
            "answer_id": answer_id,
            "answer": answer,
            "confidence": round(min(float(structured.get("structured_candidate_confidence") or 0.0), 0.35), 6),
            "explanation": "Used a deterministic fallback after fusion JSON validation failed.",
            "limitations": self._limit(plan.support_reason or "Evidence was incomplete; fallback is low confidence.", 350),
            "raw_response": raw,
            "parse_status": status,
            "json_parse_success": False,
            "retry_count": retry_count,
            "answer_in_choices": answer in choices,
            "override_occurred": False,
            "override_reason": None,
            "structured_visual_conflict": False,
        }

    @staticmethod
    def _unsupported_choice(question: str, choices: List[str]) -> str:
        lowered = question.lower()
        for choice in choices:
            normalized = choice.lower().strip()
            if any(pattern in normalized for pattern in UNAVAILABLE_PATTERNS):
                return choice
        if any(term in lowered for term in ("mentioned", "report", "record", "documented")):
            for choice in choices:
                if "not mentioned" in choice.lower():
                    return choice
        normalized = {choice.lower().strip(): choice for choice in choices}
        if "no" in normalized and any(term in lowered for term in ("is there", "are there", "present", "positive", "identified")):
            return normalized["no"]
        if choices:
            return choices[-1]
        return ""

    @staticmethod
    def _recover_answer_id(raw: Optional[str], choices: List[str]) -> Optional[str]:
        if not raw:
            return None
        valid_ids = {option["id"] for option in indexed_choices(choices)}
        match = re.search(r'["\\\']answer_id["\\\']\s*:\s*["\\\']([A-Z]+)', raw, re.I)
        if match and match.group(1).upper() in valid_ids:
            return match.group(1).upper()
        stripped = raw.strip().upper()
        return stripped if stripped in valid_ids else None

    @staticmethod
    def _limit(value: Any, length: int) -> str:
        text = " ".join(str(value or "").split())
        return text[:length]

    def _attach_option_alignment(
        self,
        plan: ExecutionPlan,
        choices: List[str],
        structured: Dict[str, Any],
    ) -> None:
        if structured.get("task_match") == "none" or not structured.get("predictions"):
            structured["option_alignment"] = {
                "source": "not_applicable",
                "choice_id": None,
                "mapping_complete": False,
                "confidence": 0.0,
            }
            return

        literal_id = structured.get("literal_match_id")
        options = {row["id"]: row["text"] for row in indexed_choices(choices)}
        if literal_id in options:
            alignment = {
                "source": "literal_exact",
                "choice_id": literal_id,
                "mapping_complete": True,
                "confidence": 1.0,
                "reason": "The clinical predicted label exactly matches one supplied choice.",
            }
        elif self.client.enabled:
            predictions = structured.get("predictions", [])
            primary_fields = set(structured.get("primary_fields", []))
            primary = [
                self._compact_prediction(row)
                for row in predictions
                if row.get("field") in primary_fields
            ]
            supporting = [
                self._compact_prediction(row)
                for row in predictions
                if row.get("field") not in primary_fields
            ]
            payload = {
                "question": plan.question,
                "choices": indexed_choices(choices),
                "primary_predictions": primary or [
                    self._compact_prediction(predictions[0])
                ],
                "supporting_predictions": supporting,
            }
            try:
                raw = self.client.chat(
                    ALIGNMENT_SYSTEM,
                    json.dumps(payload, ensure_ascii=False),
                    max_tokens=220,
                    response_format={"type": "json_object"},
                    retries=2,
                )
                parsed = parse_json_response(raw) or {}
                choice_id = parsed.get("choice_id")
                if isinstance(choice_id, str):
                    choice_id = choice_id.strip().upper()
                else:
                    choice_id = None
                try:
                    confidence = max(
                        0.0, min(float(parsed.get("confidence", 0.0)), 1.0)
                    )
                except (TypeError, ValueError):
                    confidence = 0.0
                complete = bool(parsed.get("mapping_complete")) and choice_id in options
                alignment = {
                    "source": "llm_semantic_alignment",
                    "choice_id": choice_id if complete else None,
                    "mapping_complete": complete,
                    "confidence": round(confidence, 6),
                    "reason": self._limit(parsed.get("reason", ""), 240),
                    "raw_response": raw,
                }
            except Exception as error:
                alignment = {
                    "source": "alignment_error",
                    "choice_id": None,
                    "mapping_complete": False,
                    "confidence": 0.0,
                    "reason": f"{type(error).__name__}: option alignment failed.",
                }
        else:
            alignment = {
                "source": "client_disabled",
                "choice_id": None,
                "mapping_complete": False,
                "confidence": 0.0,
            }

        structured["option_alignment"] = alignment
        choice_id = alignment.get("choice_id")
        if alignment.get("mapping_complete") and choice_id in options:
            structured["structured_candidate_id"] = choice_id
            structured["structured_candidate_answer"] = options[choice_id]
            structured["mapping_complete"] = True
            structured["answer_unit"] = "choice_id"

    @staticmethod
    def _high_trust_candidate(structured: Dict[str, Any]) -> bool:
        if structured.get("task_match") != "direct":
            return False
        if not structured.get("structured_candidate_id"):
            return False
        alignment = structured.get("option_alignment", {})
        if float(alignment.get("confidence") or 0.0) < 0.65:
            return False
        if float(structured.get("structured_candidate_confidence") or 0.0) < 0.72:
            return False
        predictions = structured.get("predictions", [])
        if not predictions:
            return False
        primary = predictions[0]
        return (
            float(primary.get("fused_probability_for_predicted_class") or 0.0) >= 0.65
            and float(primary.get("cross_scale_agreement") or 0.0) >= (2.0 / 3.0)
            and float(primary.get("validation_quality") or 0.0) >= 0.5
        )

    @staticmethod
    def _valid_counterevidence(
        parsed: Dict[str, Any],
        structured: Dict[str, Any],
    ) -> bool:
        evidence = parsed.get("counterevidence")
        if not isinstance(evidence, dict):
            return False

        if evidence.get("is_decisive") is not True:
            return False
        if evidence.get("evidence_direction") != "supports_proposed":
            return False
        if evidence.get("supports_proposed") is not True:
            return False
        if evidence.get("contradicts_structured") is not True:
            return False

        try:
            confidence = float(evidence.get("confidence"))
        except (TypeError, ValueError):
            return False
        if not 0.75 <= confidence <= 1.0:
            return False

        required = ("visible_feature", "decisive_reason", "structured_failure")
        if any(not str(evidence.get(key) or "").strip() for key in required):
            return False

        predictions = structured.get("predictions", [])
        field = str(predictions[0].get("field") if predictions else "")
        return field in MORPHOLOGY_OVERRIDE_FIELDS
