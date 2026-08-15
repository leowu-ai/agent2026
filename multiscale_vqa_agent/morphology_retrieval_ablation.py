#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .live_metrics import normalize_answer
from .precomputed_answerability import normalize_answerability_key


EXPECTED_SUBSET_SIZE = 19


def load_jsonl(path: Path) -> Dict[Tuple[str, str], Dict[str, Any]]:
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
                raise ValueError(f"Line {line_number} in {path} has no key")
            if key in rows:
                raise ValueError(f"Duplicate key in {path}: {key!r}")
            rows[key] = row
    return rows


def write_csv(path: Path, rows: List[Dict[str, Any]], fields: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def prepare_subset(
    old_answers_path: Path,
    vqa_path: Path,
    frozen_gate_path: Path,
    output_dir: Path,
) -> Dict[str, Any]:
    old_rows = load_jsonl(old_answers_path)
    selected_keys = {
        key for key, row in old_rows.items()
        if row.get("evidence_route") == "morphology_only"
    }
    if len(selected_keys) != EXPECTED_SUBSET_SIZE:
        raise ValueError(
            f"Expected {EXPECTED_SUBSET_SIZE} old morphology-only questions, "
            f"got {len(selected_keys)}"
        )

    source_items = json.loads(vqa_path.read_text(encoding="utf-8"))
    source = {}
    for item in source_items:
        key = normalize_answerability_key(
            str(item.get("Id", item.get("case_id", "")))[:12],
            item.get("Question", item.get("question", "")),
        )
        if key in source:
            raise ValueError(f"Duplicate source VQA key: {key!r}")
        source[key] = item
    missing_items = selected_keys - set(source)
    if missing_items:
        raise ValueError(f"Missing selected source item: {sorted(missing_items)[0]!r}")

    frozen = load_jsonl(frozen_gate_path)
    missing_gate = selected_keys - set(frozen)
    if missing_gate:
        raise ValueError(f"Missing frozen Gate row: {sorted(missing_gate)[0]!r}")
    selected_gate = [frozen[key] for key in selected_keys]
    if not all(row.get("predicted_can_answer") is True for row in selected_gate):
        raise ValueError("All selected morphology-only questions must have frozen Gate=True")

    ordered_keys = [
        normalize_answerability_key(
            str(item.get("Id", item.get("case_id", "")))[:12],
            item.get("Question", item.get("question", "")),
        )
        for item in source_items
        if normalize_answerability_key(
            str(item.get("Id", item.get("case_id", "")))[:12],
            item.get("Question", item.get("question", "")),
        ) in selected_keys
    ]
    subset = [source[key] for key in ordered_keys]
    gate_subset = [frozen[key] for key in ordered_keys]
    if len(subset) != EXPECTED_SUBSET_SIZE or len(set(ordered_keys)) != len(subset):
        raise ValueError("Morphology subset construction was not one-to-one")

    output_dir.mkdir(parents=True, exist_ok=True)
    subset_path = output_dir / "morphology_only_questions.json"
    gate_path = output_dir / "gate_predictions_morphology_only.jsonl"
    subset_path.write_text(
        json.dumps(subset, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    gate_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in gate_subset),
        encoding="utf-8",
    )
    summary = {
        "selection_source": str(old_answers_path),
        "selection_rule": "evidence_route == morphology_only",
        "uses_gold_for_selection": False,
        "subset_size": len(subset),
        "frozen_gate_true": sum(row["predicted_can_answer"] for row in gate_subset),
        "subset_path": str(subset_path),
        "gate_subset_path": str(gate_path),
    }
    (output_dir / "subset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def write_retrieval_audit(answers_path: Path, output_path: Path) -> Dict[str, Any]:
    answers = load_jsonl(answers_path)
    rows = []
    for result in answers.values():
        pathology = result.get("pathology_evidence") or {}
        if pathology.get("retrieval_mode") != "question_similarity":
            continue
        for group in pathology.get("evidence_groups", []):
            for scale, patch in (group.get("patches") or {}).items():
                source = next((
                    item for item in patch.get("sources", [])
                    if item.get("type") == "question_similarity"
                ), {})
                rows.append({
                    "case_id": result.get("case_id"),
                    "question": result.get("question"),
                    "group_id": group.get("group_id"),
                    "scale": int(scale),
                    "slide_id": patch.get("slide_id"),
                    "patch_index": patch.get("patch_index"),
                    "x": patch.get("x"),
                    "y": patch.get("y"),
                    "size": patch.get("size"),
                    "question_similarity": source.get(
                        "similarity", source.get("attention")
                    ),
                    "image_path": patch.get("image_path"),
                })
    write_csv(output_path, rows, [
        "case_id", "question", "group_id", "scale", "slide_id",
        "patch_index", "x", "y", "size", "question_similarity", "image_path",
    ])
    return {"questions": len(answers), "selected_patches": len(rows)}


def answer_text(row: Dict[str, Any]) -> Any:
    value = row.get("agent_answer")
    return value.get("answer") if isinstance(value, dict) else value


def clean_multiline_text(value: Any) -> str:
    return "\n".join(line.rstrip() for line in str(value or "").splitlines())


def is_correct(row: Dict[str, Any]) -> bool:
    answer = answer_text(row)
    reference = row.get("reference_answer", row.get("input", {}).get("Answer"))
    return bool(
        answer is not None and reference is not None
        and normalize_answer(answer) == normalize_answer(reference)
    )


def compare_runs(
    broad_path: Path,
    question_path: Path,
    output_path: Path,
) -> Dict[str, Any]:
    broad = load_jsonl(broad_path)
    question = load_jsonl(question_path)
    if set(broad) != set(question) or len(broad) != EXPECTED_SUBSET_SIZE:
        raise ValueError(
            "Paired runs must contain the same 19 case_id + question keys"
        )
    route_changes = []
    rows = []
    for key in sorted(broad):
        left, right = broad[key], question[key]
        for name, row in (("broad", left), ("question", right)):
            if row.get("evidence_route") != "morphology_only":
                route_changes.append({"run": name, "key": key})
        left_correct, right_correct = is_correct(left), is_correct(right)
        classification = (
            "wrong_to_correct" if not left_correct and right_correct
            else "correct_to_wrong" if left_correct and not right_correct
            else "both_correct" if left_correct
            else "both_wrong"
        )
        rows.append({
            "case_id": left.get("case_id"),
            "question": left.get("question"),
            "reference_answer": left.get(
                "reference_answer", left.get("input", {}).get("Answer")
            ),
            "broad_answer": answer_text(left),
            "broad_correct": left_correct,
            "question_answer": answer_text(right),
            "question_correct": right_correct,
            "broad_pathology_summary": clean_multiline_text(
                (left.get("pathology_evidence") or {}).get("description")
            ),
            "question_pathology_summary": clean_multiline_text(
                (right.get("pathology_evidence") or {}).get("description")
            ),
            "broad_json_parse_success": bool(left.get("json_parse_success")),
            "question_json_parse_success": bool(right.get("json_parse_success")),
            "classification": classification,
        })
    write_csv(output_path, rows, [
        "case_id", "question", "reference_answer", "broad_answer",
        "broad_correct", "question_answer", "question_correct",
        "broad_pathology_summary", "question_pathology_summary",
        "broad_json_parse_success", "question_json_parse_success", "classification",
    ])
    summary = {
        "total": len(rows),
        "broad_correct": sum(row["broad_correct"] for row in rows),
        "question_correct": sum(row["question_correct"] for row in rows),
        "wrong_to_correct": sum(row["classification"] == "wrong_to_correct" for row in rows),
        "correct_to_wrong": sum(row["classification"] == "correct_to_wrong" for row in rows),
        "net_correct_change": (
            sum(row["question_correct"] for row in rows)
            - sum(row["broad_correct"] for row in rows)
        ),
        "broad_json_parse_failures": sum(
            not row["broad_json_parse_success"] for row in rows
        ),
        "question_json_parse_failures": sum(
            not row["question_json_parse_success"] for row in rows
        ),
        "route_changes": route_changes,
    }
    output_path.with_name("morphology_retrieval_comparison_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Morphology retrieval ablation utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--old_answers", required=True)
    prepare.add_argument("--vqa_json", required=True)
    prepare.add_argument("--frozen_gate", required=True)
    prepare.add_argument("--output_dir", required=True)

    audit = subparsers.add_parser("audit")
    audit.add_argument("--answers", required=True)
    audit.add_argument("--output", required=True)

    compare = subparsers.add_parser("compare")
    compare.add_argument("--broad", required=True)
    compare.add_argument("--question", required=True)
    compare.add_argument("--output", required=True)
    args = parser.parse_args()

    if args.command == "prepare":
        summary = prepare_subset(
            Path(args.old_answers), Path(args.vqa_json), Path(args.frozen_gate),
            Path(args.output_dir),
        )
    elif args.command == "audit":
        summary = write_retrieval_audit(Path(args.answers), Path(args.output))
    else:
        summary = compare_runs(
            Path(args.broad), Path(args.question), Path(args.output)
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
