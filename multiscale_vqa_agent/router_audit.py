import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple


AUDIT_FIELDS = (
    "case_id",
    "question",
    "choices",
    "route",
    "task_match",
    "selected_prototype_ids",
    "target_phenotypes",
    "phenotype_relevance_score",
    "reason",
    "use_pathology_agent",
)


def write_router_audit(
    records: Iterable[Tuple[Dict[str, Any], Any]],
    output_jsonl: Path,
) -> Dict[str, Any]:
    """Write answer-free Router audit artifacts next to planner-only output."""
    rows = []
    counts = Counter()
    for item, plan in records:
        route = plan.evidence_route
        task_match = plan.task_match
        counts[route] += 1
        counts[f"{route}/{task_match}"] += 1
        rows.append({
            "case_id": plan.case_id,
            "question": plan.question,
            "choices": json.dumps(
                item.get("Choice", item.get("choices", [])) or [], ensure_ascii=False
            ),
            "route": route,
            "task_match": task_match,
            "selected_prototype_ids": json.dumps(plan.selected_prototype_ids),
            "target_phenotypes": json.dumps(plan.target_phenotypes),
            "phenotype_relevance_score": plan.phenotype_relevance_score,
            "reason": plan.support_reason,
            "use_pathology_agent": plan.use_pathology_agent,
        })

    csv_path = output_jsonl.with_name(f"{output_jsonl.stem}_route_audit.csv")
    summary_path = output_jsonl.with_name(f"{output_jsonl.stem}_route_summary.json")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=AUDIT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "total": len(rows),
        "phenotype_direct/direct": counts["phenotype_direct/direct"],
        "phenotype_direct/partial": counts["phenotype_direct/partial"],
        "morphology_only": counts["morphology_only"],
        "nonvisual": counts["nonvisual"],
        "route_counts": {
            route: counts[route]
            for route in ("phenotype_direct", "morphology_only", "nonvisual")
        },
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary
