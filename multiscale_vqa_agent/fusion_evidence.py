import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


from .schemas import ExecutionPlan


PROMPT_PATH = Path(__file__).with_name("prompts") / "fusion_arbiter.txt"


UNAVAILABLE_PATTERNS = (
    "cannot be determined", "unable to determine", "not provided", "not specified",
    "not mentioned", "unknown", "insufficient",
)


def choice_id(index: int) -> str:
    """Return spreadsheet-style option IDs: A..Z, AA..AZ, ..."""
    value = int(index) + 1
    result = ""
    while value:
        value, remainder = divmod(value - 1, 26)
        result = chr(ord("A") + remainder) + result
    return result


def indexed_choices(choices: Sequence[str]) -> List[Dict[str, str]]:
    return [
        {"id": choice_id(index), "text": str(text)}
        for index, text in enumerate(choices)
    ]


def choice_id_for_answer(choices: Sequence[str], answer: Any) -> Optional[str]:
    for option in indexed_choices(choices):
        if answer == option["text"]:
            return option["id"]
    return None


def load_fusion_prompt() -> str:
    return PROMPT_PATH.read_text(encoding="utf-8").strip()


def _literal_normalize(value: Any) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", str(value or "").lower()))


def _literal_choice_matches(
    prediction_rows: List[Dict[str, Any]],
    choices: Sequence[str],
) -> Dict[str, Any]:
    options = indexed_choices(choices)
    matches = []
    for prediction in prediction_rows:
        label = prediction.get("predicted_label")
        normalized_label = _literal_normalize(label)
        if not normalized_label:
            continue
        matched_options = [
            option
            for option in options
            if _literal_normalize(option["text"]) == normalized_label
        ]
        if len(matched_options) != 1:
            continue
        option = matched_options[0]
        matches.append({
            "prototype_id": prediction.get("prototype_id"),
            "field": prediction.get("field"),
            "predicted_label": label,
            "choice_id": option["id"],
            "choice_text": option["text"],
        })
    unique_ids = {match["choice_id"] for match in matches}
    return {
        "literal_match_id": next(iter(unique_ids)) if len(unique_ids) == 1 else None,
        "literal_matches": matches,
        "literal_match_rule": (
            "Case, whitespace, and punctuation normalized equality only; "
            "no aliases, synonyms, or clinical transformations."
        ),
    }


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


def _mean(values: Sequence[Optional[float]], default: float) -> float:
    finite = [float(value) for value in values if value is not None and math.isfinite(value)]
    return sum(finite) / len(finite) if finite else default


def _structured_confidence(
    predictions: List[Dict[str, Any]],
    relevance: float,
    task_match: str,
) -> float:
    if task_match == "none" or not predictions:
        return 0.0
    primary = predictions[0]
    supporting = predictions[1:]
    primary_probability = _confidence(primary)
    support_probability = _mean([_confidence(row) for row in supporting], primary_probability or 0.5)
    agreement = _mean([_scale_agreement(row) for row in predictions], 0.5)
    validation = _mean([_validation_quality(row) for row in predictions], 0.5)
    confidence = (
        0.65 * (primary_probability if primary_probability is not None else 0.5)
        + 0.10 * support_probability
        + 0.15 * agreement
        + 0.10 * validation
    )
    return max(0.0, min(confidence * relevance, 0.98))


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
        {"direct": 1.0, "partial": 0.6, "none": 0.0}.get(task_match, 0.0),
    ))

    rows = []
    selected_prototype_ids = list(getattr(plan, "selected_prototype_ids", []))
    confidences = []
    agreements = []
    validations = []
    for prediction in predictions:
        fused = prediction.get("fused", {})
        confidence = _confidence(prediction)
        agreement = _scale_agreement(prediction)
        validation = _validation_quality(prediction)
        if confidence is not None:
            confidences.append(confidence)
        if agreement is not None:
            agreements.append(agreement)
        if validation is not None:
            validations.append(validation)
        rows.append({
            "prototype_id": (
                selected_prototype_ids[len(rows)]
                if len(selected_prototype_ids) > len(rows)
                else None
            ),
            "field": prediction.get("field"),
            "predicted_class_index_zero_based": fused.get("predicted_class"),
            "predicted_label": fused.get("predicted_label"),
            "clinical_label_semantics": prediction.get("label_semantics", {}),
            "fused_probability_for_predicted_class": confidence,
            "cross_scale_agreement": agreement,
            "validation_quality": validation,
        })

    literal_match = _literal_choice_matches(rows, choices)
    base_confidence = _structured_confidence(predictions, relevance, task_match)
    answerability = relevance * coverage if predictions and task_match != "none" else 0.0
    primary_fields = executed[:1]
    supporting_fields = executed[1:]

    return {
        "available": bool(predictions and task_match != "none"),
        "evidence_route": getattr(plan, "evidence_route", "phenotype_direct"),
        "selected_prototype_ids": selected_prototype_ids,
        "task_match": task_match,
        "requested_fields": requested,
        "executed_fields": executed,
        "missing_fields": missing,
        "target_coverage": coverage,
        "answerability_score": round(answerability, 6),
        "phenotype_relevance_score": relevance,
        "predictions": rows if task_match != "none" else [],
        "structured_candidate_answer": None,
        "structured_candidate_id": None,
        "fields_used": executed,
        "mapping_complete": False,
        "answer_unit": "llm_semantic_option_mapping",
        "option_compatibility": [],
        **literal_match,
        "primary_fields": primary_fields,
        "supporting_fields": supporting_fields,
        "modifier_fields": missing,
        "contradictory_fields": [],
        "structured_candidate_confidence": round(base_confidence, 6),
        "confidence_factors": {
            "prediction_centered_confidence": round(base_confidence, 6),
            "mean_fused_probability": _mean(confidences, 0.0),
            "mean_multiscale_agreement": _mean(agreements, 0.0),
            "mean_validation_quality": _mean(validations, 0.5),
            "choice_mapping_by": "final_fusion_llm",
        },
        "class_index_rule": (
            "Class indices are zero-based internal values. Clinical meaning comes only from "
            "predicted_label and clinical_label_semantics; never infer it from option position."
        ),
        "supplied_choices_verbatim": choices,
        "supplied_choice_options": indexed_choices(choices),
    }
