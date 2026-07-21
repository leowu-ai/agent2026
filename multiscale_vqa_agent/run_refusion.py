#!/usr/bin/env python3
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from multiscale_vqa_agent.clients import OpenAICompatibleClient
from multiscale_vqa_agent.fusion import FusionVerificationAgent
from multiscale_vqa_agent.fusion_evidence import build_structured_summary
from multiscale_vqa_agent.live_metrics import LiveAccuracyTracker
from multiscale_vqa_agent.schemas import ExecutionPlan


def completed_keys(path: Path):
    keys = set()
    if not path.exists():
        return keys
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
                keys.add((str(row.get("case_id", "")), str(row.get("question", ""))))
            except json.JSONDecodeError:
                continue
    return keys


def main():
    agent_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Re-run only final fusion on saved VQA evidence")
    parser.add_argument("--config", default=str(agent_dir / "config.servers.json"))
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--no_resume", action="store_true")
    args = parser.parse_args()

    with Path(args.config).open(encoding="utf-8") as handle:
        config = json.load(handle)
    source_path = Path(args.source)
    output_path = Path(args.output)
    metrics_path = Path(args.metrics)
    with source_path.open(encoding="utf-8") as handle:
        source_rows = [json.loads(line) for line in handle if line.strip()]

    resume = not args.no_resume
    done = completed_keys(output_path) if resume else set()
    mode = "a" if resume and output_path.exists() else "w"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    history_path = metrics_path.with_name(f"{metrics_path.stem}_history.csv")
    tracker = LiveAccuracyTracker(
        metrics_path,
        history_path,
        selected_total=len(source_rows),
        existing_answers=output_path if resume else None,
    )
    fusion = FusionVerificationAgent(OpenAICompatibleClient(config["qwen"]))
    remaining = [
        row for row in source_rows
        if (str(row.get("case_id", "")), str(row.get("question", ""))) not in done
    ]
    print(
        f"fusion_only selected={len(source_rows)} completed={len(done)} remaining={len(remaining)}",
        flush=True,
    )

    with output_path.open(mode, encoding="utf-8") as output:
        for index, source in enumerate(remaining, 1):
            row = dict(source)
            old_answer = row.get("agent_answer")
            try:
                plan = ExecutionPlan(**row["plan"])
                new_answer = fusion.answer(
                    plan,
                    list(row.get("choices", [])),
                    row.get("phenotype_predictions", row.get("phenotype_prediction", {})),
                    row.get("relation_evidence_by_field", row.get("relation_evidence", {})),
                    row["pathology_evidence"],
                )
                row["previous_agent_answer"] = old_answer
                row["agent_answer"] = new_answer
                structured = build_structured_summary(
                    plan, list(row.get("choices", [])), row.get("phenotype_predictions", row.get("phenotype_prediction", {}))
                )
                row.update({
                    "task_match": structured["task_match"],
                    "requested_fields": structured["requested_fields"],
                    "executed_fields": structured["executed_fields"],
                    "missing_fields": structured["missing_fields"],
                    "structured_candidate_answer": structured["structured_candidate_answer"],
                    "structured_candidate_confidence": structured["structured_candidate_confidence"],
                    "structured_evidence": structured,
                    "answer_in_choices": new_answer.get("answer") in row.get("choices", []),
                    "raw_response": new_answer.get("raw_response"),
                    "parse_status": new_answer.get("parse_status"),
                    "json_parse_success": new_answer.get("json_parse_success", False),
                    "retry_count": new_answer.get("retry_count", 0),
                    "override_occurred": new_answer.get("override_occurred", False),
                    "override_reason": new_answer.get("override_reason"),
                    "structured_visual_conflict": new_answer.get("structured_visual_conflict", False),
                })
                row.pop("error", None)
                row["refusion"] = {
                    "version": "multi_expert_arbiter_v3",
                    "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
                    "reused_g2p": True,
                    "reused_relations": True,
                    "reused_pathology_description": True,
                }
            except Exception as error:
                row["error"] = f"{type(error).__name__}: {error}"

            output.write(json.dumps(row, ensure_ascii=False) + "\n")
            output.flush()
            evaluation = tracker.update(row)
            snapshot = tracker.snapshot()
            accuracy = snapshot["accuracy"]
            supported = snapshot["supported_accuracy"]
            print(
                f"refusion {snapshot['processed']}/{snapshot['selected_total']} "
                f"acc={accuracy:.4f} supported_acc={supported:.4f} "
                f"errors={snapshot['errors']} last_correct={evaluation['correct']}",
                flush=True,
            )
    tracker.save_snapshot()
    print(f"Saved: {output_path}", flush=True)


if __name__ == "__main__":
    main()
