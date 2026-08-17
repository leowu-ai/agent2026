import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from .answerability_evaluation import _load_gold
from .live_metrics import normalize_answer
from .precomputed_answerability import normalize_answerability_key


def _load_jsonl(path: Path) -> Dict[Tuple[str, str], Dict[str, Any]]:
    rows = {}
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            key = normalize_answerability_key(
                row.get("case_id"), row.get("question")
            )
            if not all(key):
                raise ValueError(f"Result line {line_number} is missing its key")
            if key in rows:
                raise ValueError(f"Duplicate result key in {path}: {key!r}")
            rows[key] = row
    return rows


def _answer(row: Dict[str, Any]) -> Any:
    value = row.get("agent_answer")
    return value.get("answer") if isinstance(value, dict) else value


def _reference(row: Dict[str, Any]) -> Any:
    return row.get("reference_answer", (row.get("input") or {}).get("Answer"))


def _correct(row: Dict[str, Any]) -> bool:
    answer = _answer(row)
    reference = _reference(row)
    return bool(
        answer is not None
        and reference is not None
        and normalize_answer(answer) == normalize_answer(reference)
    )


def _task_match(row: Dict[str, Any]) -> str:
    plan = row.get("plan") or {}
    return str(row.get("task_match", plan.get("task_match", "abstained")))


def _evidence_route(row: Dict[str, Any]) -> str:
    plan = row.get("plan") or {}
    return str(row.get("evidence_route", plan.get("evidence_route", "abstained")))


def _route(row: Dict[str, Any]) -> str:
    task_match = _task_match(row)
    if task_match in {"direct", "partial"}:
        return task_match
    if _evidence_route(row) == "morphology_only" or task_match == "none":
        return "morphology_only"
    return "abstained"


def _classification(old_correct: bool, new_correct: bool) -> str:
    if not old_correct and new_correct:
        return "wrong_to_correct"
    if old_correct and not new_correct:
        return "correct_to_wrong"
    return "both_correct" if old_correct else "both_wrong"


def _ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _clean_cell(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    return "\n".join(line.rstrip() for line in value.splitlines()).strip()


def _write_csv(path: Path, rows: List[Dict[str, Any]], fields: Iterable[str]) -> None:
    fields = list(fields)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _clean_cell(row.get(field)) for field in fields})


def _flip_summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    counts = Counter(row["classification"] for row in rows)
    return {
        "total": len(rows),
        "wrong_to_correct": counts["wrong_to_correct"],
        "correct_to_wrong": counts["correct_to_wrong"],
        "both_correct": counts["both_correct"],
        "both_wrong": counts["both_wrong"],
        "net_correct_change": (
            counts["wrong_to_correct"] - counts["correct_to_wrong"]
        ),
    }


def _route_flip_summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    summary = {}
    for route in ("direct", "partial", "morphology_only"):
        same_route = [
            row for row in rows
            if row["old_route"] == route and row["new_route"] == route
        ]
        summary[route] = _flip_summary(same_route)
    changed = [row for row in rows if row["old_route"] != row["new_route"]]
    summary["route_changed"] = _flip_summary(changed)
    summary["route_changed"]["transitions"] = dict(sorted(Counter(
        f"{row['old_route']}->{row['new_route']}" for row in changed
    ).items()))
    return summary


def _route_gold_summary(
    rows: Dict[Tuple[str, str], Dict[str, Any]],
    gold: Dict[Tuple[str, str], Dict[str, Any]],
) -> Dict[str, Any]:
    output = {}
    for route in ("direct", "partial", "morphology_only"):
        selected = [
            (key, row) for key, row in rows.items() if _route(row) == route
        ]
        answerable = [
            (key, row) for key, row in selected
            if not gold[key].get("exclude_from_evaluation", False)
            and gold[key].get("can_answer") is True
        ]
        unanswerable = [
            (key, row) for key, row in selected
            if not gold[key].get("exclude_from_evaluation", False)
            and gold[key].get("can_answer") is False
        ]
        excluded = [
            (key, row) for key, row in selected
            if gold[key].get("exclude_from_evaluation", False)
        ]
        output[route] = {
            "total_answered_in_route": len(selected),
            "total_correct": sum(_correct(row) for _, row in selected),
            "raw_accuracy": _ratio(
                sum(_correct(row) for _, row in selected), len(selected)
            ),
            "gold_answerable_n": len(answerable),
            "gold_answerable_correct": sum(
                _correct(row) for _, row in answerable
            ),
            "gold_answerable_accuracy": _ratio(
                sum(_correct(row) for _, row in answerable), len(answerable)
            ),
            "gold_unanswerable_n": len(unanswerable),
            "gold_unanswerable_correct": sum(
                _correct(row) for _, row in unanswerable
            ),
            "excluded_n": len(excluded),
        }
    return output


def write_full_run_analysis(
    new_answers_path: Path,
    old_answers_path: Path,
    labels_path: Path,
) -> Dict[str, Any]:
    """Write paired route and correctness audits after full inference."""
    new_rows = _load_jsonl(Path(new_answers_path))
    old_rows = _load_jsonl(Path(old_answers_path))
    gold_rows = _load_gold(Path(labels_path))
    gold = {
        normalize_answerability_key(row["Id"], row["Question"]): row
        for row in gold_rows
    }
    expected = set(gold)
    for name, rows in (("new", new_rows), ("old", old_rows)):
        missing = expected - set(rows)
        extra = set(rows) - expected
        if missing or extra:
            raise ValueError(
                f"{name} answers do not match Gold keys: "
                f"missing={len(missing)} extra={len(extra)}"
            )

    comparison = []
    for key in sorted(expected):
        old = old_rows[key]
        new = new_rows[key]
        old_correct = _correct(old)
        new_correct = _correct(new)
        label = gold[key]
        comparison.append({
            "case_id": key[0],
            "question": key[1],
            "old_task_match": _task_match(old),
            "new_task_match": _task_match(new),
            "old_evidence_route": _evidence_route(old),
            "new_evidence_route": _evidence_route(new),
            "old_route": _route(old),
            "new_route": _route(new),
            "gold_can_answer": label.get("can_answer"),
            "excluded": bool(label.get("exclude_from_evaluation", False)),
            "old_answer": _answer(old),
            "new_answer": _answer(new),
            "reference_answer": _reference(new),
            "old_correct": old_correct,
            "new_correct": new_correct,
            "classification": _classification(old_correct, new_correct),
        })

    output_dir = Path(new_answers_path).parent
    fields = list(comparison[0])
    _write_csv(output_dir / "full_run_comparison.csv", comparison, fields)
    partial_rows = [
        row for row in comparison
        if row["old_route"] == "partial" or row["new_route"] == "partial"
    ]
    _write_csv(output_dir / "partial_comparison.csv", partial_rows, fields)

    valid = [row for row in comparison if not row["excluded"]]
    gold_answerable = [row for row in valid if row["gold_can_answer"] is True]
    old_gold_correct = sum(row["old_correct"] for row in gold_answerable)
    new_gold_correct = sum(row["new_correct"] for row in gold_answerable)
    old_gold_accuracy = _ratio(old_gold_correct, len(gold_answerable))
    new_gold_accuracy = _ratio(new_gold_correct, len(gold_answerable))
    summary = {
        "comparison_scope": (
            "New full system configuration versus old full system configuration; "
            "differences are not attributable to retrieval alone because Fusion "
            "thinking and stochastic sampling also differ."
        ),
        "total": _flip_summary(comparison),
        "gold_answerable": _flip_summary(gold_answerable),
        "gold_answerable_benchmark": {
            "denominator": len(gold_answerable),
            "old_correct": old_gold_correct,
            "new_correct": new_gold_correct,
            "absolute_correct_delta": new_gold_correct - old_gold_correct,
            "old_accuracy": old_gold_accuracy,
            "new_accuracy": new_gold_accuracy,
            "percentage_point_delta": 100.0 * (
                new_gold_accuracy - old_gold_accuracy
            ),
        },
        "same_route_flips": _route_flip_summary(comparison),
        "gold_answerable_same_route_flips": _route_flip_summary(gold_answerable),
        "task_match_transitions": dict(sorted(Counter(
            f"{row['old_task_match']}->{row['new_task_match']}"
            for row in comparison
        ).items())),
        "old_route_counts": dict(sorted(Counter(
            row["old_route"] for row in comparison
        ).items())),
        "new_route_counts": dict(sorted(Counter(
            row["new_route"] for row in comparison
        ).items())),
    }
    (output_dir / "full_run_comparison_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    partial_summary = {
        "all_old_or_new_partial": _flip_summary(partial_rows),
        "same_route_partial": _flip_summary([
            row for row in partial_rows
            if row["old_route"] == "partial" and row["new_route"] == "partial"
        ]),
        "gold_answerable_old_or_new_partial": _flip_summary([
            row for row in partial_rows
            if not row["excluded"] and row["gold_can_answer"] is True
        ]),
        "route_transitions": dict(sorted(Counter(
            f"{row['old_route']}->{row['new_route']}" for row in partial_rows
            if row["old_route"] != row["new_route"]
        ).items())),
    }
    (output_dir / "partial_comparison_summary.json").write_text(
        json.dumps(partial_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    route_summary = {
        "gold_used_post_inference_only": True,
        "routes": _route_gold_summary(new_rows, gold),
    }
    (output_dir / "route_gold_answerable_summary.json").write_text(
        json.dumps(route_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {
        "route_gold_answerable": route_summary,
        "comparison": summary,
        "partial": partial_summary,
    }
