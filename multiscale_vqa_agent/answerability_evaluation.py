import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .live_metrics import normalize_answer


EXPECTED_GOLD_TOTAL = 390
EXPECTED_GOLD_VALID = 382
EXPECTED_GOLD_EXCLUDED = 8
EXPECTED_GOLD_CAN_ANSWER = 186
EXPECTED_GOLD_CANNOT_ANSWER = 196


def _ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _load_gold(path: Path) -> List[Dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    labels = payload.get("labels") if isinstance(payload, dict) else None
    if not isinstance(labels, list):
        raise ValueError("Binary Gold must contain a labels list")
    if len(labels) != EXPECTED_GOLD_TOTAL:
        raise ValueError(f"Expected 390 Gold labels, got {len(labels)}")

    keys = []
    for index, row in enumerate(labels):
        if not isinstance(row, dict):
            raise ValueError(f"Gold row {index} is not an object")
        key = (str(row.get("Id", "")), str(row.get("Question", "")))
        if not all(key):
            raise ValueError(f"Gold row {index} is missing Id or Question")
        if not isinstance(row.get("exclude_from_evaluation", False), bool):
            raise ValueError(f"Gold row {key} has invalid exclusion flag")
        excluded = row.get("exclude_from_evaluation", False)
        if not excluded and (
            "can_answer" not in row or not isinstance(row["can_answer"], bool)
        ):
            raise ValueError(f"Non-excluded Gold row {key} has non-boolean can_answer")
        if excluded and row.get("can_answer") not in {None, True, False}:
            raise ValueError(f"Excluded Gold row {key} has invalid can_answer")
        keys.append(key)
    if len(set(keys)) != len(keys):
        raise ValueError("Binary Gold contains duplicate (Id, Question) keys")

    valid = [row for row in labels if not row.get("exclude_from_evaluation", False)]
    excluded = len(labels) - len(valid)
    can_answer = sum(row["can_answer"] for row in valid)
    cannot_answer = len(valid) - can_answer
    observed = (len(valid), excluded, can_answer, cannot_answer)
    expected = (
        EXPECTED_GOLD_VALID,
        EXPECTED_GOLD_EXCLUDED,
        EXPECTED_GOLD_CAN_ANSWER,
        EXPECTED_GOLD_CANNOT_ANSWER,
    )
    if observed != expected:
        raise ValueError(
            "Binary Gold counts do not match frozen benchmark: "
            f"observed valid/excluded/true/false={observed}, expected={expected}"
        )
    return labels


def _load_predictions(
    path: Path,
) -> Tuple[Dict[Tuple[str, str], Dict[str, Any]], int, int]:
    predictions = {}
    total = 0
    duplicates = 0
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            total += 1
            row = json.loads(line)
            key = (str(row.get("case_id", "")), str(row.get("question", "")))
            if not all(key):
                raise ValueError(f"Prediction line {line_number} is missing case_id or question")
            if key in predictions:
                duplicates += 1
            else:
                predictions[key] = row
    if duplicates:
        raise ValueError(f"Prediction file contains {duplicates} duplicate (case_id, question) keys")
    return predictions, total, duplicates


def _predicted_can_answer(row: Dict[str, Any]) -> Tuple[bool, bool]:
    value = row.get("predicted_can_answer")
    if isinstance(value, bool):
        return value, False
    legacy = str(row.get("predicted_answerability", "")).strip().lower()
    if legacy in {"directly_answerable", "inferable", "answerable"}:
        return True, True
    if legacy == "unanswerable":
        return False, True
    raise ValueError("Prediction is missing a boolean predicted_can_answer")


def _binary_auroc(labels: List[bool], scores: List[float]) -> Optional[float]:
    positives = sum(labels)
    negatives = len(labels) - positives
    if not positives or not negatives or len(scores) != len(labels):
        return None
    ordered = sorted(zip(scores, labels), key=lambda item: item[0])
    rank_sum = 0.0
    position = 1
    while position <= len(ordered):
        end = position
        while end < len(ordered) and ordered[end][0] == ordered[position - 1][0]:
            end += 1
        average_rank = (position + end) / 2.0
        rank_sum += average_rank * sum(label for _, label in ordered[position - 1:end])
        position = end + 1
    return _ratio(rank_sum - positives * (positives + 1) / 2.0, positives * negatives)


def evaluate_answerability(
    answers_path: Path,
    labels_path: Path,
    summary_path: Path = None,
    audit_path: Path = None,
) -> Dict[str, Any]:
    gold_rows = _load_gold(Path(labels_path))
    predictions, prediction_total, duplicate_predictions = _load_predictions(
        Path(answers_path)
    )
    gold_keys = {(str(row["Id"]), str(row["Question"])) for row in gold_rows}
    valid_gold = [row for row in gold_rows if not row["exclude_from_evaluation"]]
    valid_keys = {(str(row["Id"]), str(row["Question"])) for row in valid_gold}
    missing = sorted(valid_keys - set(predictions))
    extra = sorted(set(predictions) - gold_keys)
    if missing:
        raise ValueError(f"Missing {len(missing)} non-excluded predictions; first={missing[0]}")

    tp = fp = fn = tn = 0
    audit = []
    confidence_scores = []
    confidence_labels = []
    legacy_conversions = 0

    for gold in gold_rows:
        key = (str(gold["Id"]), str(gold["Question"]))
        excluded = bool(gold["exclude_from_evaluation"])
        predicted_row = predictions.get(key)
        predicted = None
        legacy_used = False
        if predicted_row is not None:
            predicted, legacy_used = _predicted_can_answer(predicted_row)
            legacy_conversions += int(legacy_used)

        answer_data = (predicted_row or {}).get("agent_answer") or {}
        agent_answer = answer_data.get("answer") if isinstance(answer_data, dict) else answer_data
        reference = (predicted_row or {}).get(
            "reference_answer", (predicted_row or {}).get("input", {}).get("Answer")
        )
        abstained = bool((predicted_row or {}).get("abstained", agent_answer is None))
        answered = not abstained and agent_answer is not None
        correct = bool(
            answered and reference is not None
            and normalize_answer(agent_answer) == normalize_answer(reference)
        )
        confidence = (predicted_row or {}).get("answerability_confidence", 0.0)
        try:
            confidence = max(0.0, min(float(confidence), 1.0))
        except (TypeError, ValueError):
            confidence = 0.0

        if not excluded:
            if gold["can_answer"] and predicted:
                tp += 1
            elif not gold["can_answer"] and predicted:
                fp += 1
            elif gold["can_answer"]:
                fn += 1
            else:
                tn += 1
            confidence_scores.append(confidence if predicted else 1.0 - confidence)
            confidence_labels.append(bool(gold["can_answer"]))

        audit.append({
            "case_id": key[0],
            "question": key[1],
            "gold_can_answer": gold["can_answer"],
            "predicted_can_answer": predicted,
            "answerability_confidence": confidence,
            "answerability_fallback_used": bool(
                (predicted_row or {}).get("answerability_fallback_used", False)
            ),
            "excluded": excluded,
            "abstained": abstained,
            "reference_answer": reference,
            "agent_answer": agent_answer,
            "correct": correct,
        })

    precision = _ratio(tp, tp + fp)
    recall = _ratio(tp, tp + fn)
    specificity = _ratio(tn, tn + fp)
    valid_audit = [row for row in audit if not row["excluded"]]
    gold_answerable = [row for row in valid_audit if row["gold_can_answer"]]
    answered = [row for row in valid_audit if not row["abstained"] and row["agent_answer"] is not None]
    summary = {
        "dataset": {
            "gold_total": EXPECTED_GOLD_TOTAL,
            "gold_valid": EXPECTED_GOLD_VALID,
            "gold_excluded": EXPECTED_GOLD_EXCLUDED,
            "gold_can_answer": EXPECTED_GOLD_CAN_ANSWER,
            "gold_cannot_answer": EXPECTED_GOLD_CANNOT_ANSWER,
        },
        "answerability": {
            "accuracy": _ratio(tp + tn, EXPECTED_GOLD_VALID),
            "precision": precision,
            "recall": recall,
            "f1": _ratio(2 * precision * recall, precision + recall),
            "specificity": specificity,
            "balanced_accuracy": (recall + specificity) / 2.0,
            "auroc": _binary_auroc(confidence_labels, confidence_scores),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        },
        "vqa": {
            "gold_answerable_n": len(gold_answerable),
            "gold_answerable_correct": sum(row["correct"] for row in gold_answerable),
            "gold_answerable_accuracy": _ratio(
                sum(row["correct"] for row in gold_answerable), len(gold_answerable)
            ),
            "answered_n": len(answered),
            "abstained_n": len(valid_audit) - len(answered),
            "selective_accuracy": _ratio(sum(row["correct"] for row in answered), len(answered)),
            "coverage": _ratio(len(answered), len(valid_audit)),
            "overall_correct": sum(row["correct"] for row in valid_audit),
            "overall_accuracy": _ratio(
                sum(row["correct"] for row in valid_audit), len(valid_audit)
            ),
        },
        "integrity": {
            "prediction_total": prediction_total,
            "missing_predictions": len(missing),
            "extra_predictions": len(extra),
            "duplicate_predictions": duplicate_predictions,
            "legacy_conversions": legacy_conversions,
        },
    }

    summary_path = Path(summary_path or Path(answers_path).with_name(
        f"{Path(answers_path).stem}_answerability_summary.json"
    ))
    audit_path = Path(audit_path or Path(answers_path).with_name(
        f"{Path(answers_path).stem}_answerability_audit.csv"
    ))
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with audit_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(audit[0]))
        writer.writeheader()
        writer.writerows(audit)
    return summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate binary WSI answerability")
    parser.add_argument("answers_jsonl")
    parser.add_argument("--answerability_labels", required=True)
    parser.add_argument("--summary", default=None)
    parser.add_argument("--audit", default=None)
    args = parser.parse_args()
    summary = evaluate_answerability(
        Path(args.answers_jsonl),
        Path(args.answerability_labels),
        Path(args.summary) if args.summary else None,
        Path(args.audit) if args.audit else None,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
