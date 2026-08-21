#!/usr/bin/env python3
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OLD_REPO = PROJECT_ROOT / "g2p_toolbank_brca"
for path in (PROJECT_ROOT, OLD_REPO):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from pathagent.pathagent_baseline import (
    CONCHTextEncoder,
    CaseFeatureStore,
    PathAgentBaseline,
    PathAgentExecutor,
    PathAgentPerceptor,
    PathAgentRetriever,
    StrictWSICropper,
    _repo_modules,
    normalized_key,
    release_case,
)


def load_config(path: str) -> Dict[str, Any]:
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config not found: {config_path}")
    return json.loads(config_path.read_text(encoding="utf-8"))


def load_completed(path: Path) -> Set[Tuple[str, str]]:
    completed: Set[Tuple[str, str]] = set()
    if not path.is_file():
        return completed
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            key = normalized_key(row.get("case_id"), row.get("question"))
            if key in completed:
                raise ValueError(f"Duplicate output key at line {line_number}: {key}")
            completed.add(key)
    return completed


def load_gold_answerable_keys(
    path: Path, expected_count: Optional[int] = 186
) -> Set[Tuple[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Gold labels not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    labels = payload.get("labels") if isinstance(payload, dict) else None
    if not isinstance(labels, list):
        raise ValueError("Gold answerability file must contain a labels list")
    keys = {
        normalized_key(row.get("Id"), row.get("Question"))
        for row in labels
        if (
            isinstance(row, dict)
            and not row.get("exclude_from_evaluation", False)
            and row.get("can_answer") is True
        )
    }
    if expected_count is not None and len(keys) != expected_count:
        raise ValueError(
            f"Expected {expected_count} Gold-answerable questions, got {len(keys)}"
        )
    return keys


def ordinary_metrics(path: Path) -> Dict[str, Any]:
    total = correct = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            answer = (row.get("agent_answer") or {}).get("answer")
            reference = row.get("reference_answer")
            total += 1
            correct += int(
                bool(answer is not None and reference is not None)
                and " ".join(str(answer).lower().split())
                == " ".join(str(reference).lower().split())
            )
    return {
        "processed": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
    }


def build_baseline(config: Dict[str, Any]) -> Tuple[PathAgentBaseline, CaseFeatureStore]:
    _, _, client_class, _ = _repo_modules()
    retrieval = config["retrieval"]
    text_encoder = CONCHTextEncoder(
        config["conch_checkpoint"], config.get("device", "cuda:0")
    )
    retriever = PathAgentRetriever(text_encoder)
    feature_store = CaseFeatureStore(config["features"])
    evidence_root = Path(config["pathagent_root"]) / "outputs" / "evidence_patches"
    cropper = StrictWSICropper(config["wsi_root"], str(evidence_root))
    perceptor = PathAgentPerceptor(
        client_class(config["pathor1"]), retrieval["perceptor_batch_size"]
    )
    executor = PathAgentExecutor(client_class(config["qwen"]))
    baseline = PathAgentBaseline(
        retriever=retriever,
        cropper=cropper,
        perceptor=perceptor,
        executor=executor,
        initial_patches=retrieval["initial_patches"],
        replenish_patches=retrieval["replenish_patches"],
        max_attempts=retrieval["max_attempts"],
        zoom_parent_topk=retrieval["zoom_parent_topk"],
        max_zoom_actions=retrieval["max_zoom_actions"],
    )
    return baseline, feature_store


def run(
    config: Dict[str, Any],
    limit: Optional[int] = None,
    resume: bool = True,
    baseline: Optional[PathAgentBaseline] = None,
    feature_store: Optional[CaseFeatureStore] = None,
    gold_answerable_only: bool = False,
) -> Path:
    source = Path(config["vqa_json"])
    if not source.is_file():
        raise FileNotFoundError(f"VQA file not found: {source}")
    rows = json.loads(source.read_text(encoding="utf-8"))
    rows = [row for row in rows if row.get("Choice", row.get("choices"))]
    if gold_answerable_only:
        answerable_keys = load_gold_answerable_keys(
            Path(config["answerability_labels"]), expected_count=186
        )
        rows = [
            row for row in rows
            if normalized_key(
                row.get("Id", row.get("case_id")),
                row.get("Question", row.get("question")),
            ) in answerable_keys
        ]
        if len(rows) != 186:
            raise ValueError(
                "VQA/Gold alignment did not produce exactly 186 answerable questions: "
                f"got {len(rows)}"
            )
    selected = rows[:limit] if limit is not None else rows

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "mc_answers.jsonl"
    if not resume and output.exists():
        output.unlink()
    completed = load_completed(output) if resume else set()
    remaining = [
        row for row in selected
        if normalized_key(
            row.get("Id", row.get("case_id")),
            row.get("Question", row.get("question")),
        ) not in completed
    ]
    print(
        f"pathagent selected={len(selected)} completed={len(selected)-len(remaining)} "
        f"remaining={len(remaining)}",
        flush=True,
    )
    if baseline is None or feature_store is None:
        baseline, feature_store = build_baseline(config)

    by_case: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in remaining:
        by_case[str(row.get("Id", row.get("case_id", "")))[:12]].append(row)
    processed = len(selected) - len(remaining)
    for case_index, (case_id, case_rows) in enumerate(by_case.items(), 1):
        print(
            f"[{case_index}/{len(by_case)}] load {case_id} "
            f"({len(case_rows)} questions)",
            flush=True,
        )
        case = feature_store.load_case(case_id)
        try:
            with output.open("a", encoding="utf-8") as handle:
                for item in case_rows:
                    inference_item = {
                        "Id": case_id,
                        "Question": item.get("Question", item.get("question")),
                        "Choice": item.get("Choice", item.get("choices")),
                    }
                    result = baseline.answer(inference_item, case)
                    result["evaluation_subset"] = (
                        "gold_can_answer_oracle_186"
                        if gold_answerable_only else "all_questions"
                    )
                    result["reference_answer"] = item.get(
                        "Answer", item.get("answer", item.get("reference_answer"))
                    )
                    handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                    handle.flush()
                    processed += 1
                    is_correct = (
                        " ".join(result["agent_answer"]["answer"].lower().split())
                        == " ".join(str(result["reference_answer"]).lower().split())
                    )
                    print(
                        f"pathagent processed={processed}/{len(selected)} "
                        f"last_correct={is_correct}",
                        flush=True,
                    )
        finally:
            release_case(case)

    metrics = ordinary_metrics(output)
    metrics["selection_policy"] = (
        "gold_can_answer_oracle_filter"
        if gold_answerable_only else "all_questions"
    )
    if gold_answerable_only:
        metrics.update({
            "gold_answerable_n": 186,
            "gold_answerable_correct": metrics["correct"],
            "gold_answerable_accuracy": metrics["correct"] / 186.0,
        })
    metrics_path = output_dir / "mc_answers_metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Gold is deliberately imported and read only after a complete full run.
    all_keys = {
        normalized_key(
            row.get("Id", row.get("case_id")),
            row.get("Question", row.get("question")),
        )
        for row in rows
    }
    output_keys = load_completed(output)
    if not gold_answerable_only and limit is None and all_keys.issubset(output_keys):
        labels = Path(config["answerability_labels"])
        if not labels.is_file():
            raise FileNotFoundError(f"Gold labels not found for evaluation: {labels}")
        from multiscale_vqa_agent.answerability_evaluation import (
            evaluate_answerability,
        )

        summary = evaluate_answerability(output, labels)
        print(f"post_inference_evaluation={json.dumps(summary, ensure_ascii=False)}")
    print(f"saved={output}", flush=True)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Run independent PathAgent-CONCH-MS")
    parser.add_argument(
        "--config", default=str(Path(__file__).with_name("config.json"))
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no_resume", action="store_true")
    parser.add_argument(
        "--gold_answerable_only",
        action="store_true",
        help=(
            "Oracle-filter inference to the 186 Gold can_answer=true questions. "
            "Gold is used for subset selection only and is never sent to models."
        ),
    )
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--initial_patches", type=int, default=None)
    parser.add_argument("--replenish_patches", type=int, default=None)
    parser.add_argument("--max_attempts", type=int, default=None)
    parser.add_argument("--perceptor_batch_size", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    if args.output_dir:
        config["output_dir"] = str(Path(args.output_dir).resolve())
    overrides = {
        "initial_patches": args.initial_patches,
        "replenish_patches": args.replenish_patches,
        "max_attempts": args.max_attempts,
        "perceptor_batch_size": args.perceptor_batch_size,
    }
    for key, value in overrides.items():
        if value is not None:
            if value <= 0:
                raise ValueError(f"--{key} must be positive")
            config["retrieval"][key] = value
    run(
        config,
        limit=args.limit,
        resume=not args.no_resume,
        gold_answerable_only=args.gold_answerable_only,
    )


if __name__ == "__main__":
    main()
