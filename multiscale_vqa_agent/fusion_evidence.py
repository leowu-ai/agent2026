import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .schemas import ExecutionPlan


PROMPT_PATH = Path(__file__).with_name("prompts") / "fusion_arbiter.txt"


YES_WORDS = {"yes", "present", "positive", "detected", "identified"}
NO_WORDS = {"no", "absent", "negative", "not", "none"}
UNAVAILABLE_PATTERNS = (
    "cannot be determined", "unable to determine", "not provided", "not specified",
    "not mentioned", "unknown", "insufficient",
)
BINARY_UNCERTAIN_CONFIDENCE = 0.55


def load_fusion_prompt() -> str:
    return PROMPT_PATH.read_text(encoding="utf-8").strip()


def _confidence(prediction: Dict[str, Any]) -> Optional[float]:
    fused = prediction.get("fused", {})
    predicted_class = fused.get("predicted_class")
    probabilities = fused.get("probabilities")
    if probabilities:
        return float(max(probabilities))
    if "probability" in fused:
        value = float(fused["probability"])
        return value if predicted_class == 1 else 1.0 - value
    return None


def _scale_agreement(prediction: Dict[str, Any]) -> Optional[float]:
    predicted = prediction.get("fused", {}).get("predicted_class")
    values = [
        row.get("predicted_class")
        for row in prediction.get("per_scale", {}).values()
        if row.get("predicted_class") is not None
    ]
    if predicted is None or not values:
        return None
    return sum(value == predicted for value in values) / len(values)


def _validation_quality(prediction: Dict[str, Any]) -> Optional[float]:
    values = []
    for row in prediction.get("validation_metrics", {}).values():
        try:
            value = float(row.get("metric_value"))
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            values.append(max(0.0, min(value, 1.0)))
    return sum(values) / len(values) if values else None


def _normalize(value: Any) -> str:
    text = str(value or "").lower().replace("infiltrating", "invasive")
    text = text.replace("in-situ", "in situ")
    text = re.sub(r"\bgrade\s*1\b", " low ", text)
    text = re.sub(r"\bgrade\s*2\b", " intermediate ", text)
    text = re.sub(r"\bgrade\s*3\b", " high ", text)
    return " ".join(re.findall(r"[a-z0-9]+", text))


def _has_phrase(text: str, phrase: str) -> bool:
    return phrase in f" {text} "


def _label_for_choice(field: str, label: Any) -> str:
    value = str(label or "").strip().lower().replace("infiltrating", "invasive")
    if field == "histologic_grade_label":
        return {"1": "low", "2": "intermediate", "3": "high"}.get(value, value)
    if field.endswith("_binary"):
        if value in {"1", "true", "positive", "present"}:
            return "yes"
        if value in {"0", "false", "negative", "absent"}:
            return "no"
    return value


def _predicted_labels(predictions: Sequence[Dict[str, Any]]) -> Tuple[Dict[str, str], List[str]]:
    labels: Dict[str, str] = {}
    uncertain_fields: List[str] = []
    for row in predictions:
        field = row.get("field")
        if not field:
            continue
        confidence = _confidence(row)
        if field.endswith("_binary") and confidence is not None and confidence < BINARY_UNCERTAIN_CONFIDENCE:
            uncertain_fields.append(field)
            continue
        labels[field] = _label_for_choice(field, row.get("fused", {}).get("predicted_label"))
    return labels, uncertain_fields


def _is_yes_no_question(question: str, choices: Sequence[str]) -> bool:
    q = _normalize(question)
    normalized_choices = {_normalize(choice) for choice in choices}
    has_yes_no = {"yes", "no"}.issubset(normalized_choices)
    asks_presence = any(term in q for term in (" is ", " are ", " does ", " do ", " present", " positive", " identified", " mentioned", "documented"))
    return has_yes_no and asks_presence


def _choice_requirements(choice: str, question: str, choices: Sequence[str], fields: Sequence[str]) -> Dict[str, Any]:
    text = _normalize(choice)
    field_set = set(fields)
    requirements: Dict[str, str] = {}
    contradictions: Dict[str, str] = {}
    answer_unit = "unknown"
    question_text = _normalize(question)
    generic_in_situ = (
        ("any" in question_text and "in situ" in question_text)
        or "in situ component" in question_text
    )

    if any(pattern in str(choice).lower() for pattern in UNAVAILABLE_PATTERNS):
        return {"choice": choice, "answer_unit": "unavailable", "requirements": {}, "contradictions": {}}

    if "histologic_grade_label" in field_set:
        answer_unit = "grade"
        if _has_phrase(text, "low"):
            requirements["histologic_grade_label"] = "low"
        elif _has_phrase(text, "intermediate"):
            requirements["histologic_grade_label"] = "intermediate"
        elif _has_phrase(text, "high"):
            requirements["histologic_grade_label"] = "high"

    for field in ("ER_status_label", "PR_status_label", "HER2_status_label"):
        marker = field.split("_")[0].lower()
        if field not in field_set:
            continue
        question_text = _normalize(question)
        marker_relevant = marker in question_text or marker in text
        if not marker_relevant:
            continue
        answer_unit = "receptor_status"
        yes_no_mode = _is_yes_no_question(question, choices)
        if yes_no_mode:
            if text == "yes":
                requirements[field] = "positive"
            elif text == "no":
                requirements[field] = "negative"
        else:
            if _has_phrase(text, "positive"):
                requirements[field] = "positive"
            elif _has_phrase(text, "negative"):
                requirements[field] = "negative"
            elif _has_phrase(text, "equivocal"):
                requirements[field] = "equivocal"

    binary_fields = [field for field in field_set if field.endswith("_binary")]
    if binary_fields:
        yes_no_choice = text in {"yes", "no"}
        for field in binary_fields:
            # Generic in-situ presence is DCIS OR LCIS, not two required positives.
            if generic_in_situ and field in {"dcis_binary", "lcis_binary"}:
                continue
            stem = field[:-7].replace("_", " ")
            relevant = stem in text or stem in _normalize(question) or yes_no_choice
            if not relevant:
                continue
            answer_unit = "binary_presence"
            if yes_no_choice:
                requirements[field] = text
            elif any(word in text.split() for word in ("present", "positive", "detected", "identified")):
                requirements[field] = "yes"
            elif any(word in text.split() for word in ("absent", "negative")) or "without" in text:
                requirements[field] = "no"

    if {"dcis_binary", "lcis_binary"} & field_set and ("in situ" in _normalize(question) or "in situ" in text):
        answer_unit = "in_situ_component"
        if text == "yes":
            requirements["__any_in_situ__"] = "yes"
        elif text == "no":
            requirements["__any_in_situ__"] = "no"
        elif "ductal carcinoma in situ" in str(choice).lower() or "dcis" in text:
            requirements["dcis_binary"] = "yes"
            contradictions["lcis_only"] = "yes"
        elif "lobular carcinoma in situ" in str(choice).lower() or "lcis" in text:
            requirements["lcis_binary"] = "yes"
            contradictions["dcis_only"] = "yes"

    if "histological_type_label" in field_set:
        raw = str(choice).lower().replace("infiltrating", "invasive").replace("in-situ", "in situ")
        if "atypical" in raw or "hyperplasia" in raw:
            return {"choice": choice, "answer_unit": "unsupported_histology_option", "requirements": {}, "contradictions": {"unsupported_option": "yes"}}
        if "ductal carcinoma in situ" in raw or "dcis" in text:
            answer_unit = "in_situ_diagnosis"
            requirements["dcis_binary"] = "yes"
            contradictions["invasive_status"] = "yes"
        elif "lobular carcinoma in situ" in raw or "lcis" in text:
            answer_unit = "in_situ_diagnosis"
            requirements["lcis_binary"] = "yes"
            contradictions["invasive_status"] = "yes"
        elif "lobular carcinoma" in raw and "in situ" not in raw:
            answer_unit = "diagnosis_class"
            requirements["histological_type_label"] = "invasive lobular carcinoma"
            requirements["lobular_binary"] = "yes"
            if "ductal_binary" in field_set:
                requirements["ductal_binary"] = "no"
            contradictions["dcis_only"] = "yes"
        elif "ductal carcinoma" in raw and "in situ" not in raw:
            answer_unit = "diagnosis_class"
            requirements["histological_type_label"] = "invasive ductal carcinoma"
            requirements["ductal_binary"] = "yes"
            if "lobular_binary" in field_set:
                requirements["lobular_binary"] = "no"
            contradictions["dcis_only"] = "yes"
        elif "mixed" in raw and "carcinoma" in raw:
            answer_unit = "diagnosis_class"
            requirements["ductal_binary"] = "yes"
            requirements["lobular_binary"] = "yes"

    primary_requirements = dict(requirements)
    supporting_requirements: Dict[str, str] = {}
    if answer_unit == "diagnosis_class" and "histological_type_label" in requirements:
        primary_requirements = {
            "histological_type_label": requirements["histological_type_label"],
        }
        supporting_requirements = {
            field: expected
            for field, expected in requirements.items()
            if field != "histological_type_label"
        }

    return {
        "choice": choice,
        "answer_unit": answer_unit,
        "requirements": requirements,
        "primary_requirements": primary_requirements,
        "supporting_requirements": supporting_requirements,
        "contradictions": contradictions,
    }


def _requirement_met(field: str, expected: str, labels: Dict[str, str]) -> Optional[bool]:
    if field == "__any_in_situ__":
        values = [labels.get("dcis_binary"), labels.get("lcis_binary")]
        if any(value == "yes" for value in values):
            return expected == "yes"
        if any(value is None for value in values):
            return None
        return expected == "no"
    observed = labels.get(field)
    if observed is None:
        return None
    if field == "histological_type_label":
        return _normalize(expected) == _normalize(observed)
    return _normalize(expected) == _normalize(observed)


def _option_compatibility(
    question: str,
    choices: List[str],
    predictions: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    fields = [row.get("field") for row in predictions if row.get("field")]
    labels, uncertain_fields = _predicted_labels(predictions)
    rows = []
    for choice in choices:
        parsed = _choice_requirements(choice, question, choices, fields)
        supported = []
        missing = []
        contradicted = []
        primary_supported = []
        primary_missing = []
        primary_contradicted = []
        supporting_supported = []
        supporting_missing = []
        supporting_contradicted = []
        primary_requirements = parsed.get("primary_requirements", parsed["requirements"])
        supporting_requirements = parsed.get("supporting_requirements", {})
        for field, expected in parsed["requirements"].items():
            met = _requirement_met(field, expected, labels)
            is_primary = field in primary_requirements
            if met is True:
                supported.append(field)
                (primary_supported if is_primary else supporting_supported).append(field)
            elif met is False:
                contradicted.append(field)
                (primary_contradicted if is_primary else supporting_contradicted).append(field)
            else:
                missing.append(field)
                (primary_missing if is_primary else supporting_missing).append(field)
        hard_contradictions = []
        for field in parsed["contradictions"]:
            if field == "unsupported_option":
                contradicted.append(field)
                hard_contradictions.append(field)
            elif field == "dcis_only" and labels.get("dcis_binary") == "yes" and labels.get("histological_type_label") not in {"invasive ductal carcinoma", "invasive lobular carcinoma"}:
                contradicted.append(field)
                hard_contradictions.append(field)
            elif field == "invasive_status" and labels.get("histological_type_label", "").startswith("invasive"):
                contradicted.append(field)
                hard_contradictions.append(field)
        row_uncertain_fields = [
            field for field in parsed["requirements"] if field in uncertain_fields
        ]
        if "__any_in_situ__" in parsed["requirements"]:
            row_uncertain_fields.extend(
                field
                for field in ("dcis_binary", "lcis_binary")
                if field in uncertain_fields
            )
        primary_coverage = len(primary_supported) / max(len(primary_requirements), 1) if primary_requirements else 0.0
        evidence_coverage = len(supported) / max(len(parsed["requirements"]), 1) if parsed["requirements"] else 0.0
        rows.append({
            **parsed,
            "primary_requirements": primary_requirements,
            "supporting_requirements": supporting_requirements,
            "supported_fields": supported,
            "missing_required_fields": missing,
            "contradicted_fields": contradicted,
            "primary_supported_fields": primary_supported,
            "missing_primary_fields": primary_missing,
            "contradicted_primary_fields": primary_contradicted,
            "supporting_supported_fields": supporting_supported,
            "missing_supporting_fields": supporting_missing,
            "contradicted_supporting_fields": supporting_contradicted,
            "uncertain_fields": row_uncertain_fields,
            "hard_contradictions": hard_contradictions,
            "support_score": 2 * len(primary_supported) + len(supporting_supported),
            "contradiction_score": len(contradicted),
            "field_coverage": primary_coverage,
            "evidence_coverage": evidence_coverage,
        })
    return rows


def _candidate_answer(
    question: str,
    choices: List[str],
    predictions: List[Dict[str, Any]],
    missing_fields: List[str],
) -> Dict[str, Any]:
    compatibility = _option_compatibility(question, choices, predictions)
    eligible = [
        row for row in compatibility
        if row.get("primary_requirements")
        and not row["missing_primary_fields"]
        and not row["contradicted_primary_fields"]
        and not row["hard_contradictions"]
    ]
    eligible.sort(key=lambda row: (row["support_score"], row["field_coverage"]), reverse=True)
    answer = None
    used: List[str] = []
    complete = False
    answer_unit = "unknown"
    missing_supporting: List[str] = []
    contradicted_supporting: List[str] = []
    if eligible:
        best = eligible[0]
        tied = len(eligible) > 1 and (eligible[1]["support_score"], eligible[1]["field_coverage"]) == (best["support_score"], best["field_coverage"])
        if not tied:
            answer = best["choice"]
            used = list(best["supported_fields"])
            complete = True
            answer_unit = best["answer_unit"]
            missing_supporting = list(best["missing_supporting_fields"])
            contradicted_supporting = list(best["contradicted_supporting_fields"])

    return {
        "structured_candidate_answer": answer,
        "fields_used": used,
        "missing_fields": missing_fields,
        "mapping_complete": bool(answer is not None and complete),
        "answer_unit": answer_unit,
        "missing_supporting_fields": missing_supporting,
        "contradicted_supporting_fields": contradicted_supporting,
        "option_compatibility": compatibility,
    }


def _candidate_confidence(
    candidate: Dict[str, Any],
    prediction_by_field: Dict[str, Dict[str, Any]],
    relevance: float,
    coverage: float,
    task_match: str,
) -> Tuple[float, Dict[str, Any]]:
    if task_match == "none" or not candidate.get("structured_candidate_answer"):
        return 0.0, {
            "primary_fields": [],
            "supporting_fields": [],
            "modifier_fields": [],
            "contradictory_fields": [],
        }
    used = [field for field in candidate.get("fields_used", []) if field in prediction_by_field]
    primary = [field for field in used if field in {
        "histological_type_label", "histologic_grade_label", "ER_status_label", "PR_status_label", "HER2_status_label",
    }]
    if not primary and used:
        primary = [used[0]]
    supporting = [field for field in used if field not in primary]
    modifier = list(candidate.get("missing_supporting_fields", []))
    contradictory = list(candidate.get("contradicted_supporting_fields", []))
    primary_scores = [_confidence(prediction_by_field[field]) for field in primary]
    support_scores = [_confidence(prediction_by_field[field]) for field in supporting]
    agreement_scores = [_scale_agreement(prediction_by_field[field]) for field in used]
    validation_scores = [_validation_quality(prediction_by_field[field]) for field in used]
    primary_mean = sum(v for v in primary_scores if v is not None) / max(sum(v is not None for v in primary_scores), 1)
    support_mean = sum(v for v in support_scores if v is not None) / max(sum(v is not None for v in support_scores), 1) if support_scores else primary_mean
    agreement = sum(v for v in agreement_scores if v is not None) / max(sum(v is not None for v in agreement_scores), 1) if agreement_scores else 0.5
    validation = sum(v for v in validation_scores if v is not None) / max(sum(v is not None for v in validation_scores), 1) if validation_scores else 0.5
    confidence = (0.68 * primary_mean + 0.12 * support_mean + 0.10 * agreement + 0.10 * validation)
    confidence *= 0.96 ** len(modifier)
    confidence *= 0.88 ** len(contradictory)
    confidence *= relevance
    confidence = max(0.0, min(confidence, 0.98))
    return confidence, {
        "primary_fields": primary,
        "supporting_fields": supporting,
        "modifier_fields": modifier,
        "contradictory_fields": contradictory,
        "primary_mean_probability": primary_mean,
        "supporting_mean_probability": support_mean,
        "mean_multiscale_agreement": agreement,
        "mean_validation_quality": validation,
    }


def build_structured_summary(
    plan: ExecutionPlan,
    choices: List[str],
    phenotype_results: Any,
) -> Dict[str, Any]:
    if isinstance(phenotype_results, dict):
        predictions = [phenotype_results] if phenotype_results.get("field") else []
    else:
        predictions = list(phenotype_results or [])

    requested = list(plan.target_phenotypes)
    executed = [row.get("field") for row in predictions if row.get("field")]
    missing = [field for field in requested if field not in executed]
    coverage = len(executed) / len(requested) if requested else 0.0
    task_match = getattr(plan, "task_match", "direct" if plan.supported else "none")
    relevance = float(getattr(
        plan,
        "phenotype_relevance_score",
        {"direct": 1.0, "partial": 0.6, "indirect": 0.3, "none": 0.0}.get(task_match, 0.0),
    ))
    rows = []
    confidences = []
    agreements = []
    validations = []
    prediction_by_field = {}
    for prediction in predictions:
        fused = prediction.get("fused", {})
        field = prediction.get("field")
        confidence = _confidence(prediction)
        agreement = _scale_agreement(prediction)
        validation = _validation_quality(prediction)
        prediction_by_field[field] = prediction
        if confidence is not None:
            confidences.append(confidence)
        if agreement is not None:
            agreements.append(agreement)
        if validation is not None:
            validations.append(validation)
        rows.append({
            "field": field,
            "predicted_class_index_zero_based": fused.get("predicted_class"),
            "predicted_label": fused.get("predicted_label"),
            "clinical_label_semantics": prediction.get("label_semantics", {}),
            "fused_probability_for_predicted_class": confidence,
            "cross_scale_agreement": agreement,
            "validation_quality": validation,
        })

    structured_confidence = sum(confidences) / len(confidences) if confidences else 0.0
    scale_consistency = sum(agreements) / len(agreements) if agreements else 0.0
    validation_quality = sum(validations) / len(validations) if validations else 0.5
    candidate = _candidate_answer(plan.question, choices, predictions, missing)
    base_confidence, role_fields = _candidate_confidence(candidate, prediction_by_field, relevance, coverage, task_match)
    answerability = relevance * coverage * (1.0 if candidate["mapping_complete"] else 0.0)
    if task_match == "none":
        base_confidence = 0.0
        answerability = 0.0

    return {
        "available": bool(predictions and task_match != "none"),
        "task_match": task_match,
        "requested_fields": requested,
        "executed_fields": executed,
        "missing_fields": missing,
        "target_coverage": coverage,
        "answerability_score": round(answerability, 6),
        "phenotype_relevance_score": relevance,
        "predictions": rows if task_match != "none" else [],
        **candidate,
        **role_fields,
        "structured_candidate_confidence": round(base_confidence, 6),
        "confidence_factors": {
            "candidate_centered_confidence": round(base_confidence, 6),
            "mean_fused_probability_all_executed_fields": structured_confidence,
            "mean_multiscale_agreement_all_executed_fields": scale_consistency,
            "mean_validation_quality_all_executed_fields": validation_quality,
            "choice_mapping_complete": candidate["mapping_complete"],
        },
        "class_index_rule": (
            "Class indices are zero-based internal values. Clinical meaning comes only from "
            "predicted_label and clinical_label_semantics; never infer it from option position."
        ),
        "supplied_choices_verbatim": choices,
    }
