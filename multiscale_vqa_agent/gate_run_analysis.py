import csv
import json
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


def _correct(row: Dict[str, Any]) -> bool:
    predicted = _answer(row)
    reference = row.get("reference_answer", row.get("input", {}).get("Answer"))
    return bool(
        predicted is not None
        and reference is not None
        and normalize_answer(predicted) == normalize_answer(reference)
    )


def _write_csv(path: Path, rows: List[Dict[str, Any]], fields: Iterable[str]) -> None:
    fields = list(fields)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_gate_run_analysis(
    new_answers_path: Path,
    old_answers_path: Path,
    labels_path: Path,
) -> Dict[str, Any]:
    """Write post-inference Gate transition and downstream call summaries."""
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

    accepted = []
    rejected = []
    for key in sorted(expected):
        new = new_rows[key]
        old = old_rows[key]
        label = gold[key]
        old_gate = bool(old.get("predicted_can_answer"))
        new_gate = bool(new.get("predicted_can_answer"))
        if not old_gate and new_gate:
            plan = new.get("plan", {}) or {}
            accepted.append({
                "case_id": new.get("case_id"),
                "question": new.get("question"),
                "reference_answer": new.get(
                    "reference_answer", new.get("input", {}).get("Answer")
                ),
                "gold_can_answer": label.get("can_answer"),
                "old_gate": old_gate,
                "new_gate": new_gate,
                "route": new.get("evidence_route", plan.get("evidence_route")),
                "task_match": new.get("task_match", plan.get("task_match")),
                "structured_candidate": new.get("structured_candidate_answer"),
                "final_answer": _answer(new),
                "correct": _correct(new),
                "abstained": bool(new.get("abstained", False)),
            })
        elif old_gate and not new_gate:
            rejected.append({
                "case_id": new.get("case_id"),
                "question": new.get("question"),
                "gold_can_answer": label.get("can_answer"),
                "old_final_answer": _answer(old),
                "old_correct": _correct(old),
                "new_gate_decision": new_gate,
            })

    output_dir = Path(new_answers_path).parent
    accepted_path = output_dir / "newly_accepted_results.csv"
    rejected_path = output_dir / "newly_rejected_questions.csv"
    _write_csv(accepted_path, accepted, (
        "case_id", "question", "reference_answer", "gold_can_answer",
        "old_gate", "new_gate", "route", "task_match",
        "structured_candidate", "final_answer", "correct", "abstained",
    ))
    _write_csv(rejected_path, rejected, (
        "case_id", "question", "gold_can_answer", "old_final_answer",
        "old_correct", "new_gate_decision",
    ))

    valid_keys = {
        key for key, row in gold.items()
        if not row.get("exclude_from_evaluation", False)
    }
    valid_new = [new_rows[key] for key in valid_keys]
    routed = [row for row in new_rows.values() if row.get("plan")]
    router_questions = [
        row for row in routed
        if not row.get("forced_morphology_from_gate_false", False)
    ]
    forced_morphology = [
        row for row in routed
        if row.get("forced_morphology_from_gate_false", False)
    ]
    pathology_questions = [
        row for row in routed
        if (row.get("pathology_evidence") or {}).get("backend")
        not in {None, "skipped", "no_visual_evidence", "mock"}
    ]
    final_fusion = [
        row for row in routed
        if not row.get("error") and row.get("agent_answer") is not None
    ]
    summary = {
        "dataset": {
            "total_questions": len(new_rows),
            "valid_questions": len(valid_keys),
            "excluded_questions": len(new_rows) - len(valid_keys),
        },
        "precomputed_gate": {
            "true_total": sum(bool(row.get("predicted_can_answer")) for row in new_rows.values()),
            "false_total": sum(not bool(row.get("predicted_can_answer")) for row in new_rows.values()),
            "true_valid": sum(bool(row.get("predicted_can_answer")) for row in valid_new),
            "false_valid": sum(not bool(row.get("predicted_can_answer")) for row in valid_new),
            "answerability_model_calls": 0,
        },
        "downstream": {
            "questions_entered": len(routed),
            "router_calls": len(router_questions),
            "router_skipped_forced_morphology": len(forced_morphology),
            "g2p_question_routes": len(routed),
            "g2p_patient_cases": len({row.get("case_id") for row in routed}),
            "pathology_agent_questions": len(pathology_questions),
            "pathor1_request_attempts": sum(
                int((row.get("pathology_evidence") or {}).get("request_attempts", 0))
                for row in routed
            ),
            "final_fusion_calls": len(final_fusion),
        },
        "newly_accepted": {
            "total": len(accepted),
            "gold_answerable": sum(row["gold_can_answer"] is True for row in accepted),
            "gold_unanswerable": sum(row["gold_can_answer"] is False for row in accepted),
            "answered_correct": sum(row["correct"] for row in accepted),
            "gold_answerable_correct": sum(
                row["correct"] for row in accepted if row["gold_can_answer"] is True
            ),
        },
        "newly_rejected": {
            "total": len(rejected),
            "gold_answerable": sum(row["gold_can_answer"] is True for row in rejected),
            "old_correct": sum(row["old_correct"] for row in rejected),
            "old_gold_answerable_correct": sum(
                row["old_correct"]
                for row in rejected if row["gold_can_answer"] is True
            ),
        },
    }
    summary["transition_net_gold_answerable_correct"] = (
        summary["newly_accepted"]["gold_answerable_correct"]
        - summary["newly_rejected"]["old_gold_answerable_correct"]
    )
    summary_path = output_dir / "gate_v2_runtime_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary
