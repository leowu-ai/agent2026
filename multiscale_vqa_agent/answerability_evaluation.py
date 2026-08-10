import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from .answerability import ANSWERABILITY_CLASSES
from .live_metrics import normalize_answer


ORDERED_CLASSES = ["directly_answerable", "inferable", "unanswerable"]


def _load_predictions(path: Path) -> Dict[Tuple[str, str], Dict[str, Any]]:
    predictions = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            key = (str(row.get("case_id", "")), str(row.get("question", "")))
            predictions[key] = row
    return predictions


def _load_gold(path: Path) -> List[Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("exact_annotations") if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise ValueError("Gold answerability file must contain exact_annotations")
    return rows


def _ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _classification_metrics(pairs: Iterable[Tuple[str, str]]) -> Dict[str, Any]:
    pairs = list(pairs)
    matrix = [[0 for _ in ORDERED_CLASSES] for _ in ORDERED_CLASSES]
    index = {label: position for position, label in enumerate(ORDERED_CLASSES)}
    for gold, predicted in pairs:
        matrix[index[gold]][index[predicted]] += 1
    per_class = {}
    for label in ORDERED_CLASSES:
        position = index[label]
        tp = matrix[position][position]
        fp = sum(matrix[row][position] for row in range(len(matrix))) - tp
        fn = sum(matrix[position]) - tp
        precision = _ratio(tp, tp + fp)
        recall = _ratio(tp, tp + fn)
        f1 = _ratio(2 * precision * recall, precision + recall)
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": sum(matrix[position]),
        }
    return {
        "n_valid": len(pairs),
        "accuracy": _ratio(sum(matrix[i][i] for i in range(3)), len(pairs)),
        "macro_f1": sum(values["f1"] for values in per_class.values()) / 3.0,
        "per_class": per_class,
        "confusion_matrix_labels": ORDERED_CLASSES,
        "confusion_matrix": matrix,
    }


def evaluate_answerability(
    answers_path: Path,
    labels_path: Path,
    summary_path: Path = None,
    audit_path: Path = None,
) -> Dict[str, Any]:
    predictions = _load_predictions(Path(answers_path))
    gold_rows = _load_gold(Path(labels_path))
    audit = []
    pairs = []

    for gold in gold_rows:
        gold_label = str(gold.get("answerability", ""))
        if gold_label == "dataset_error":
            continue
        if gold_label not in ANSWERABILITY_CLASSES:
            continue
        key = (str(gold.get("Id", "")), str(gold.get("Question", "")))
        predicted_row = predictions.get(key, {})
        predicted_label = str(
            predicted_row.get("predicted_answerability", "unanswerable")
        )
        if predicted_label not in ANSWERABILITY_CLASSES:
            predicted_label = "unanswerable"
        pairs.append((gold_label, predicted_label))

        answer_data = predicted_row.get("agent_answer") or {}
        agent_answer = answer_data.get("answer") if isinstance(answer_data, dict) else answer_data
        reference = gold.get("Answer")
        answered = not bool(predicted_row.get("abstained", True)) and agent_answer is not None
        correct = bool(
            answered and normalize_answer(agent_answer) == normalize_answer(reference)
        )
        audit.append({
            "case_id": key[0],
            "question": key[1],
            "gold_answerability": gold_label,
            "predicted_answerability": predicted_label,
            "answerability_confidence": predicted_row.get("answerability_confidence", 0.0),
            "abstained": not answered,
            "reference_answer": reference,
            "agent_answer": agent_answer,
            "correct": correct,
        })

    detection = _classification_metrics(pairs)
    binary_tp = binary_fp = binary_fn = binary_tn = 0
    for gold, predicted in pairs:
        gold_answerable = gold != "unanswerable"
        predicted_answerable = predicted != "unanswerable"
        if gold_answerable and predicted_answerable:
            binary_tp += 1
        elif not gold_answerable and predicted_answerable:
            binary_fp += 1
        elif gold_answerable:
            binary_fn += 1
        else:
            binary_tn += 1
    binary_precision = _ratio(binary_tp, binary_tp + binary_fp)
    binary_recall = _ratio(binary_tp, binary_tp + binary_fn)

    primary = [row for row in audit if row["gold_answerability"] != "unanswerable"]
    strict = [row for row in audit if row["gold_answerability"] == "directly_answerable"]
    answered = [row for row in audit if not row["abstained"]]
    summary = {
        "answerability": detection,
        "binary_answerability": {
            "accuracy": _ratio(binary_tp + binary_tn, len(pairs)),
            "precision": binary_precision,
            "recall": binary_recall,
            "f1": _ratio(2 * binary_precision * binary_recall, binary_precision + binary_recall),
        },
        "vqa": {
            "gold_predictively_answerable_n": len(primary),
            "gold_predictively_answerable_correct": sum(row["correct"] for row in primary),
            "gold_predictively_answerable_accuracy": _ratio(
                sum(row["correct"] for row in primary), len(primary)
            ),
            "strict_visual_n": len(strict),
            "strict_visual_correct": sum(row["correct"] for row in strict),
            "strict_visual_accuracy": _ratio(sum(row["correct"] for row in strict), len(strict)),
            "answered_n": len(answered),
            "selective_accuracy": _ratio(sum(row["correct"] for row in answered), len(answered)),
            "coverage": _ratio(len(answered), len(audit)),
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
        fieldnames = list(audit[0]) if audit else [
            "case_id", "question", "gold_answerability", "predicted_answerability",
            "answerability_confidence", "abstained", "reference_answer", "agent_answer", "correct",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(audit)
    return summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate WSI answerability and fixed-subset VQA")
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
