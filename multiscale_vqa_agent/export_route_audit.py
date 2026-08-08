#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path


FIELDS = (
    "case_id",
    "question",
    "choices",
    "reference_answer",
    "evidence_route",
    "support_reason",
)


def export_route_audit(source: Path, output: Path) -> int:
    rows = []
    with source.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("task_match") != "none":
                continue
            rows.append({
                "case_id": record.get("case_id", ""),
                "question": record.get("question", ""),
                "choices": json.dumps(
                    record.get("choices", []), ensure_ascii=False
                ),
                "reference_answer": record.get("reference_answer", ""),
                "evidence_route": record.get("evidence_route", ""),
                "support_reason": (record.get("plan") or {}).get(
                    "support_reason", ""
                ),
            })

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export task_match=none VQA rows for route auditing."
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    count = export_route_audit(args.source, args.output)
    print(f"Exported {count} route-audit rows to {args.output}")


if __name__ == "__main__":
    main()
