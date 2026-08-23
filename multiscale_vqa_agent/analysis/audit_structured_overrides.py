#!/usr/bin/env python3
import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict

from multiscale_vqa_agent.live_metrics import normalize_answer


def _correct(value: Any, reference: Any) -> bool:
    return bool(
        value is not None
        and reference is not None
        and normalize_answer(value) == normalize_answer(reference)
    )


def audit(path: Path) -> Dict[str, Any]:
    totals = defaultdict(lambda: {
        "proposed": 0,
        "accepted": 0,
        "rejected": 0,
        "structured_candidate_correct": 0,
        "final_answer_correct": 0,
    })
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if not row.get("override_proposed"):
                continue
            task_match = str(row.get("task_match") or "none")
            reference = row.get("reference_answer")
            answer = row.get("agent_answer") or {}
            final_answer = answer.get("answer") if isinstance(answer, dict) else answer
            accepted = bool(
                row.get("override_accepted", row.get("override_occurred", False))
            )
            for key in ("all", task_match):
                bucket = totals[key]
                bucket["proposed"] += 1
                bucket["accepted"] += int(accepted)
                bucket["rejected"] += int(not accepted)
                bucket["structured_candidate_correct"] += int(_correct(
                    row.get("structured_candidate_answer"), reference
                ))
                bucket["final_answer_correct"] += int(_correct(
                    final_answer, reference
                ))
    return {
        "source": str(path),
        "overall": dict(totals["all"]),
        "by_task_match": {
            key: dict(totals[key]) for key in ("direct", "partial", "none")
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit structured overrides in an existing MCQ result JSONL."
    )
    parser.add_argument("answers", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    result = audit(args.answers)
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
