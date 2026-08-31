#!/usr/bin/env python3
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


FIELD = "histological_type_label"
BCNB_LABELS = (
    "Invasive ductal carcinoma",
    "Invasive lobular carcinoma",
    "Other",
)
HEAD_TO_BCNB = {
    1: "Invasive ductal carcinoma",
    2: "Invasive lobular carcinoma",
}


def collapse_probabilities(probabilities: Iterable[float]) -> Dict[str, float]:
    values = [float(value) for value in probabilities]
    if len(values) != 9:
        raise ValueError(f"Expected 9 histology probabilities, found {len(values)}")
    return {
        BCNB_LABELS[0]: values[1],
        BCNB_LABELS[1]: values[2],
        BCNB_LABELS[2]: sum(value for index, value in enumerate(values) if index not in {1, 2}),
    }


def hard_collapse(predicted_class: int) -> str:
    return HEAD_TO_BCNB.get(int(predicted_class), "Other")


def grouped_prediction(probabilities: Iterable[float]) -> Dict[str, Any]:
    grouped = collapse_probabilities(probabilities)
    answer = max(BCNB_LABELS, key=lambda label: grouped[label])
    return {"answer": answer, "probabilities": grouped, "confidence": grouped[answer]}


def _load_tumor_items(path: Path) -> Dict[str, Dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        rows = json.load(handle)
    tumor = {}
    for row in rows:
        if row.get("Task") != "Tumor":
            continue
        key = str(row["Id"])
        if key in tumor:
            raise ValueError(f"Duplicate BCNB Tumor case: {key}")
        tumor[key] = row
    return tumor


def _histology_prediction(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    primary = row.get("phenotype_prediction") or {}
    if primary.get("field") == FIELD:
        return primary
    for prediction in row.get("phenotype_predictions", []):
        if prediction.get("field") == FIELD:
            return prediction
    return None


def _iter_jsonl(path: Path):
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from error


def _classification_metrics(records: List[Dict[str, Any]], prediction_key: str) -> Dict[str, Any]:
    confusion = {gold: {pred: 0 for pred in BCNB_LABELS} for gold in BCNB_LABELS}
    correct = 0
    predicted_counts = Counter()
    gold_counts = Counter()
    for record in records:
        gold = record["reference_answer"]
        predicted = record[prediction_key]
        confusion[gold][predicted] += 1
        correct += int(gold == predicted)
        predicted_counts[predicted] += 1
        gold_counts[gold] += 1

    per_class = {}
    f1_values = []
    recall_values = []
    for label in BCNB_LABELS:
        tp = confusion[label][label]
        fp = sum(confusion[gold][label] for gold in BCNB_LABELS if gold != label)
        fn = sum(confusion[label][pred] for pred in BCNB_LABELS if pred != label)
        precision = tp / (tp + fp) if tp + fp else None
        recall = tp / (tp + fn) if tp + fn else None
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision is not None and recall is not None and precision + recall
            else 0.0
        )
        per_class[label] = {
            "support": gold_counts[label],
            "predicted": predicted_counts[label],
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
        f1_values.append(f1)
        if recall is not None:
            recall_values.append(recall)

    return {
        "n": len(records),
        "correct": correct,
        "accuracy": correct / len(records) if records else None,
        "balanced_accuracy": sum(recall_values) / len(recall_values) if recall_values else None,
        "macro_f1": sum(f1_values) / len(f1_values) if f1_values else None,
        "gold_distribution": dict(gold_counts),
        "prediction_distribution": dict(predicted_counts),
        "per_class": per_class,
        "confusion_matrix": confusion,
    }


def evaluate(answers_path: Path, dataset_path: Path, output_dir: Path) -> Dict[str, Any]:
    tumor_items = _load_tumor_items(dataset_path)
    records = []
    seen = set()
    for row in _iter_jsonl(answers_path):
        case_id = str(row.get("case_id", ""))
        source = tumor_items.get(case_id)
        if source is None or str(row.get("question", "")) != str(source.get("Question", "")):
            continue
        if case_id in seen:
            raise ValueError(f"Duplicate prediction for BCNB Tumor case: {case_id}")
        prediction = _histology_prediction(row)
        if prediction is None:
            raise ValueError(f"Missing {FIELD} prediction for case {case_id}")

        record = {
            "question_id": str(source.get("QuestionId")),
            "case_id": case_id,
            "reference_answer": source["Answer"],
            "agent_answer": (row.get("agent_answer") or {}).get("answer"),
            "scales": {},
        }
        variants = dict(prediction.get("per_scale", {}))
        variants["fused"] = prediction["fused"]
        for scale, values in variants.items():
            probabilities = values["probabilities"]
            grouped = grouped_prediction(probabilities)
            record["scales"][str(scale)] = {
                "head_predicted_class": int(values["predicted_class"]),
                "head_predicted_label": values.get("predicted_label"),
                "head_probabilities": [float(value) for value in probabilities],
                "hard_collapsed_answer": hard_collapse(values["predicted_class"]),
                "grouped_probability_answer": grouped["answer"],
                "grouped_probabilities": grouped["probabilities"],
                "grouped_confidence": grouped["confidence"],
            }
        records.append(record)
        seen.add(case_id)

    missing = sorted(set(tumor_items).difference(seen))
    if missing:
        raise ValueError(f"Missing predictions for {len(missing)} Tumor cases; first={missing[:5]}")

    metrics = {
        "experiment": "BCNB Tumor classification using only the trained histological type head",
        "field": FIELD,
        "mapping": {
            "class_1": BCNB_LABELS[0],
            "class_2": BCNB_LABELS[1],
            "classes_0_3_4_5_6_7_8": BCNB_LABELS[2],
        },
        "source_answers": str(answers_path),
        "source_dataset": str(dataset_path),
        "n": len(records),
        "agent_tumor_baseline": _classification_metrics(records, "agent_answer"),
        "results": {},
    }
    for scale in ("1024", "2048", "4096", "fused"):
        flattened = []
        for record in records:
            flattened.append({
                "reference_answer": record["reference_answer"],
                "hard": record["scales"][scale]["hard_collapsed_answer"],
                "grouped": record["scales"][scale]["grouped_probability_answer"],
            })
        metrics["results"][scale] = {
            "hard_argmax_then_collapse": _classification_metrics(flattened, "hard"),
            "collapse_probabilities_then_argmax": _classification_metrics(flattened, "grouped"),
        }

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate the G2P histology head on BCNB Tumor MCQs")
    parser.add_argument("--answers", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    args = parser.parse_args()
    metrics = evaluate(args.answers, args.dataset, args.output_dir)
    compact = {"agent": metrics["agent_tumor_baseline"]["accuracy"]}
    for scale, result in metrics["results"].items():
        compact[scale] = {
            "hard": result["hard_argmax_then_collapse"]["accuracy"],
            "grouped": result["collapse_probabilities_then_argmax"]["accuracy"],
        }
    print(json.dumps(compact, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
