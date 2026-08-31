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


def clinical_display_label(field: Any, label: Any) -> str:
    """Render a predicted label as a clinical value, never an internal class ID."""
    value = str(label or "").strip()
    if str(field) == "histologic_grade_label" and value in {"1", "2", "3"}:
        return f"grade {value}"
    return value


FIELD_OPTION_TERMS = {
    "ER_status_label": ("er", "estrogen receptor", "estrogen receptors"),
    "PR_status_label": ("pr", "progesterone receptor", "progesterone receptors"),
    "HER2_status_label": ("her2", "her 2", "human epidermal growth factor receptor 2"),
}


def _contains_clinical_term(normalized_text: str, term: str) -> bool:
    normalized_term = _literal_normalize(term)
    return f" {normalized_term} " in f" {normalized_text} "


def primary_semantic_choice_alignment(
    structured: Dict[str, Any], choices: Sequence[str]
) -> Optional[Dict[str, Any]]:
    """Conservatively align only the primary clinical prediction to one choice."""
    predictions = structured.get("predictions", [])
    primary_fields = structured.get("primary_fields", [])
    primary = next(
        (
            row for row in predictions
            if row.get("field") in set(primary_fields)
        ),
        predictions[0] if predictions else None,
    )
    if not primary:
        return None
    field = str(primary.get("field") or "")
    label = clinical_display_label(field, primary.get("predicted_label"))
    normalized_label = _literal_normalize(label).replace("infiltrating", "invasive")
    if not normalized_label:
        return None
    field_terms = FIELD_OPTION_TERMS.get(field, ())
    matches = []
    for option in indexed_choices(choices):
        normalized = _literal_normalize(option["text"]).replace(
            "infiltrating", "invasive"
        )
        label_match = (
            normalized == normalized_label
            or f" {normalized_label} " in f" {normalized} "
            or f" {normalized} " in f" {normalized_label} "
        )
        if not label_match:
            continue
        if field_terms:
            other_receptor = any(
                _contains_clinical_term(normalized, term)
                for other_field, terms in FIELD_OPTION_TERMS.items()
                if other_field != field
                for term in terms
            )
            field_present = any(
                _contains_clinical_term(normalized, term)
                for term in field_terms
            )
            if other_receptor and not field_present:
                continue
        matches.append(option)
    if len(matches) != 1:
        return None
    return {
        "source": "deterministic_primary_clinical_semantics",
        "choice_id": matches[0]["id"],
        "mapping_complete": True,
        "confidence": 0.95,
        "reason": (
            "The primary clinical predicted label maps uniquely to this supplied choice; "
            "supporting fields do not control primary alignment."
        ),
    }


STATE_ALIASES = {
    "positive": "positive",
    "pos": "positive",
    "+": "positive",
    "yes": "positive",
    "present": "positive",
    "negative": "negative",
    "neg": "negative",
    "-": "negative",
    "no": "negative",
    "absent": "negative",
}


def _categorical_state(value: Any) -> Optional[str]:
    normalized = _literal_normalize(value)
    return STATE_ALIASES.get(normalized, STATE_ALIASES.get(str(value).strip().lower()))


def _field_aliases(field: str) -> List[str]:
    explicit = FIELD_OPTION_TERMS.get(field)
    if explicit:
        return list(explicit)
    base = re.sub(r"_(?:status_)?label$|_binary$", "", field, flags=re.I)
    normalized = " ".join(base.split("_")).strip().lower()
    aliases = [normalized]
    compact = normalized.replace(" ", "")
    if compact != normalized:
        aliases.append(compact)
    return [value for value in dict.fromkeys(aliases) if value]


def _choice_field_states(choice: str, field: str) -> List[str]:
    text = str(choice).lower()
    text = re.sub(r"her\s*-\s*2", "her2", text)
    text = re.sub(r"-\s*(positive|negative|pos|neg)\b", r" \1", text)
    state_pattern = r"(?:positive|negative|pos|neg|present|absent|yes|no|\+|-)"
    states = []
    for alias in _field_aliases(field):
        alias_pattern = re.escape(alias).replace(r"\ ", r"[\s-]+")
        alias_pattern = rf"(?<![a-z0-9]){alias_pattern}(?![a-z0-9])"
        patterns = (
            rf"{alias_pattern}\s*(?:status\s*)?(?:(?:is|=|:)\s*)?({state_pattern})",
            rf"({state_pattern})\s*(?:for\s+)?{alias_pattern}",
        )
        for pattern in patterns:
            for match in re.finditer(pattern, text, flags=re.I):
                state = _categorical_state(match.group(1))
                if state:
                    states.append(state)
    return list(dict.fromkeys(states))


def multi_field_semantic_choice_alignment(
    structured: Dict[str, Any], choices: Sequence[str]
) -> Optional[Dict[str, Any]]:
    """Align a complete joint categorical state to exactly one option."""
    requested = list(dict.fromkeys(structured.get("requested_fields", [])))
    if len(requested) < 2:
        return None
    prediction_by_field = {
        str(row.get("field")): row
        for row in structured.get("predictions", [])
        if row.get("field")
    }
    joint_state = {}
    for field in requested:
        row = prediction_by_field.get(field)
        state = _categorical_state(row.get("predicted_label")) if row else None
        if state:
            joint_state[field] = state
    result = {
        "source": "deterministic_multi_field_clinical_semantics",
        "joint_fields": requested,
        "joint_state": joint_state,
        "choice_id": None,
        "mapping_complete": False,
        "confidence": 0.0,
        "reason": "Joint categorical mapping requires every requested field and one unique compatible choice.",
    }
    if len(joint_state) != len(requested):
        return result

    matches = []
    for option in indexed_choices(choices):
        if any(
            _choice_field_states(option["text"], other_field)
            for other_field in FIELD_OPTION_TERMS
            if other_field not in requested
        ):
            continue
        observed = {
            field: _choice_field_states(option["text"], field)
            for field in requested
        }
        if all(observed[field] == [joint_state[field]] for field in requested):
            matches.append(option)
    if len(matches) == 1:
        result.update({
            "choice_id": matches[0]["id"],
            "mapping_complete": True,
            "confidence": 0.98,
            "reason": "All requested categorical fields map jointly and uniquely to this supplied choice.",
        })
    elif len(matches) > 1:
        result["reason"] = "More than one supplied choice represents the same joint categorical state."
    return result


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
    if prediction.get("scale_mode") == "single_scale":
        return None
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
        if not math.isfinite(value):
            continue
        metric = str(row.get("metric_name") or "").strip().lower()
        if metric in {"mae", "mse", "rmse", "loss"}:
            value = 1.0 / (1.0 + max(value, 0.0))
        elif metric in {"pearson", "spearman", "correlation"}:
            value = (value + 1.0) / 2.0
        values.append(max(0.0, min(value, 1.0)))
    return sum(values) / len(values) if values else None


def _mean(values: Sequence[Optional[float]], default: float) -> float:
    finite = [float(value) for value in values if value is not None and math.isfinite(value)]
    return sum(finite) / len(finite) if finite else default


def _structured_confidence(
    predictions: List[Dict[str, Any]],
    relevance: float,
    task_match: str,
) -> Dict[str, float]:
    if task_match == "none" or not predictions:
        return {
            "patient_evidence_strength": 0.0,
            "validation_reliability": 0.0,
            "reliability_adjusted_confidence": 0.0,
        }
    primary = predictions[0]
    supporting = predictions[1:]
    primary_probability = _confidence(primary)
    primary_probability = primary_probability if primary_probability is not None else 0.5
    support_probability = _mean(
        [_confidence(row) for row in supporting], primary_probability
    )
    agreement = _mean([_scale_agreement(row) for row in predictions], 0.5)
    validation = _mean([_validation_quality(row) for row in predictions], 0.5)
    patient_strength = (
        0.70 * primary_probability
        + 0.15 * support_probability
        + 0.15 * agreement
    )
    adjusted = patient_strength * validation * relevance
    return {
        "patient_evidence_strength": max(0.0, min(patient_strength, 1.0)),
        "validation_reliability": max(0.0, min(validation, 1.0)),
        "reliability_adjusted_confidence": max(0.0, min(adjusted, 1.0)),
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
        field_patient_strength = (
            0.85 * (confidence if confidence is not None else 0.5)
            + 0.15 * (agreement if agreement is not None else 0.5)
        )
        field_reliability_adjusted = (
            field_patient_strength
            * (validation if validation is not None else 0.5)
            * relevance
        )
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
            "evidence_scale": prediction.get("evidence_scale"),
            "scale_mode": prediction.get("scale_mode"),
            "predicted_class_index_zero_based": fused.get("predicted_class"),
            "predicted_label": fused.get("predicted_label"),
            "clinical_label_semantics": prediction.get("label_semantics", {}),
            "fused_probability_for_predicted_class": confidence,
            "cross_scale_agreement": agreement,
            "validation_quality": validation,
            "validation_reliability_source": (
                "prediction.validation_metrics_from_tool_metrics"
            ),
            "patient_evidence_strength": round(field_patient_strength, 6),
            "reliability_adjusted_confidence": round(
                max(0.0, min(field_reliability_adjusted, 1.0)), 6
            ),
        })

    literal_match = _literal_choice_matches(rows, choices)
    confidence_parts = _structured_confidence(predictions, relevance, task_match)
    base_confidence = confidence_parts["reliability_adjusted_confidence"]
    answerability = relevance * coverage if predictions and task_match != "none" else 0.0
    requested_executed = [field for field in requested if field in executed]
    if len(requested) > 1:
        primary_fields = requested_executed
        supporting_fields = [field for field in executed if field not in primary_fields]
    else:
        primary_fields = executed[:1]
        supporting_fields = executed[1:]

    return {
        "available": bool(predictions and task_match != "none"),
        "evidence_route": getattr(plan, "evidence_route", "phenotype_direct"),
        "selected_prototype_ids": selected_prototype_ids,
        "task_match": task_match,
        "prototype_coverage": getattr(plan, "prototype_coverage", "none"),
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
        "joint_fields": requested if len(requested) > 1 else [],
        "joint_state": {},
        "joint_alignment_source": None,
        "joint_mapping_complete": False,
        "joint_choice_id": None,
        **literal_match,
        "primary_fields": primary_fields,
        "supporting_fields": supporting_fields,
        "modifier_fields": missing,
        "contradictory_fields": [],
        "structured_candidate_confidence": round(base_confidence, 6),
        "overall_structured_reliability": round(
            confidence_parts["validation_reliability"], 6
        ),
        "validation_reliability_source": (
            "per-scale prediction.validation_metrics from active ToolBank tool_metrics.csv"
        ),
        "reliability_adjusted_confidence": round(base_confidence, 6),
        "confidence_factors": {
            "patient_evidence_strength": round(
                confidence_parts["patient_evidence_strength"], 6
            ),
            "validation_reliability": round(
                confidence_parts["validation_reliability"], 6
            ),
            "semantic_relevance": relevance,
            "reliability_adjusted_confidence": round(base_confidence, 6),
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
