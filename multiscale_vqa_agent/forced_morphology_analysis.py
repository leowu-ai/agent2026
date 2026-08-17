import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

from .answerability_evaluation import _load_gold
from .full_run_analysis import _correct, _load_jsonl, _route
from .precomputed_answerability import normalize_answerability_key


def _ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _summary(rows: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    rows = list(rows)
    correct = sum(_correct(row) for row in rows)
    return {
        "n": len(rows),
        "correct": correct,
        "accuracy": _ratio(correct, len(rows)),
    }


def write_forced_morphology_summary(
    answers_path: Path,
    labels_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Summarize forced Gate=False answers after inference has completed."""
    answers = _load_jsonl(Path(answers_path))
    rows = list(answers.values())
    gate_true = [row for row in rows if row.get("predicted_can_answer") is True]
    gate_false = [row for row in rows if row.get("predicted_can_answer") is False]
    forced = [
        row for row in rows
        if row.get("forced_morphology_from_gate_false") is True
    ]
    gate_true_routes = {
        route: _summary(row for row in gate_true if _route(row) == route)
        for route in ("direct", "partial", "morphology_only")
    }
    forced_parse_failures = sum(
        not bool(row.get(
            "json_parse_success",
            (row.get("agent_answer") or {}).get("json_parse_success", False)
            if isinstance(row.get("agent_answer"), dict) else False,
        ))
        for row in forced
        if not row.get("error")
    )
    summary = {
        "all_questions": _summary(rows),
        "gate_true": _summary(gate_true),
        "gate_false": _summary(gate_false),
        "gate_true_routes": gate_true_routes,
        "forced_morphology": {
            **_summary(forced),
            "gate_false_total": len(gate_false),
            "forced_morphology_answered": sum(
                not row.get("abstained", False)
                and row.get("agent_answer") is not None
                for row in forced
            ),
            "json_parse_failures": forced_parse_failures,
            "errors": sum(bool(row.get("error")) for row in forced),
        },
        "integrity": {
            "prediction_total": len(rows),
            "gate_true": len(gate_true),
            "gate_false": len(gate_false),
            "forced_morphology_total": len(forced),
            "abstained": sum(bool(row.get("abstained", False)) for row in rows),
            "errors": sum(bool(row.get("error")) for row in rows),
        },
        "gold_used_post_inference_only": labels_path is not None,
    }

    if labels_path is not None:
        gold_rows = _load_gold(Path(labels_path))
        gold = {
            normalize_answerability_key(row["Id"], row["Question"]): row
            for row in gold_rows
        }
        if set(answers) != set(gold):
            raise ValueError(
                "Forced morphology answers do not match Gold keys: "
                f"missing={len(set(gold) - set(answers))} "
                f"extra={len(set(answers) - set(gold))}"
            )
        valid: Iterable[Tuple[Tuple[str, str], Dict[str, Any]]] = (
            (key, row) for key, row in answers.items()
            if not gold[key].get("exclude_from_evaluation", False)
        )
        valid = list(valid)
        summary["gold_answerability"] = {
            "gold_answerable": _summary(
                row for key, row in valid if gold[key].get("can_answer") is True
            ),
            "gold_unanswerable": _summary(
                row for key, row in valid if gold[key].get("can_answer") is False
            ),
            "excluded_n": sum(
                bool(row.get("exclude_from_evaluation", False))
                for row in gold.values()
            ),
        }

    output_path = Path(answers_path).with_name("forced_morphology_summary.json")
    output_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary
