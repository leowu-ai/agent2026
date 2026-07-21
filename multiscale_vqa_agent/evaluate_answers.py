#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path


def normalize(value):
    value = str(value or "").lower().replace("infiltrating", "invasive")
    return " ".join(re.findall(r"[a-z0-9]+", value))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("answers_jsonl")
    args = parser.parse_args()
    rows = []
    with Path(args.answers_jsonl).open(encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            if item.get("error"):
                continue
            predicted = item.get("agent_answer", {}).get("answer")
            reference = item.get("reference_answer")
            if predicted is None or reference is None:
                continue
            rows.append((
                normalize(predicted) == normalize(reference),
                bool(item.get("plan", {}).get("supported")),
            ))
    supported = [correct for correct, flag in rows if flag]
    print(f"answered={len(rows)}")
    print(f"exact_accuracy_all={sum(x for x, _ in rows) / len(rows):.4f}" if rows else "exact_accuracy_all=nan")
    print(
        f"exact_accuracy_supported={sum(supported) / len(supported):.4f} n={len(supported)}"
        if supported else "exact_accuracy_supported=nan n=0"
    )


if __name__ == "__main__":
    main()
