import argparse
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
    "prototype_support_type",
    "prototype_coverage",
    "local_morphology_useful",
    "requires_unavailable_context",
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
    support_type_counts = Counter()
    for item, plan in records:
        route = plan.evidence_route
        task_match = plan.task_match
        counts[route] += 1
        counts[f"{route}/{task_match}"] += 1
        support_type_counts[plan.prototype_support_type] += 1
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
            "prototype_support_type": plan.prototype_support_type,
            "prototype_coverage": plan.prototype_coverage,
            "local_morphology_useful": plan.local_morphology_useful,
            "requires_unavailable_context": plan.requires_unavailable_context,
            "phenotype_relevance_score": plan.phenotype_relevance_score,
            "reason": plan.support_reason,
            "use_pathology_agent": plan.use_pathology_agent,
        })

    csv_path = output_jsonl.with_name(f"{output_jsonl.stem}_route_audit.csv")
    summary_path = output_jsonl.with_name(f"{output_jsonl.stem}_route_summary.json")
    support_path = output_jsonl.with_name("support_type_summary.json")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=AUDIT_FIELDS, lineterminator="\n")
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
        "support_type_counts": {
            support_type: support_type_counts[support_type]
            for support_type in ("target_evidence", "correlated_context", "none")
        },
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    support_path.write_text(
        json.dumps(summary["support_type_counts"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def _read_plans(path: Path) -> Dict[Tuple[str, str], Dict[str, Any]]:
    plans = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            plan = row.get("plan", row)
            key = (str(plan.get("case_id", "")), str(plan.get("question", "")))
            plans[key] = plan
    return plans


def _route_label(plan: Dict[str, Any]) -> str:
    route = str(plan.get("evidence_route", "nonvisual"))
    if route == "phenotype_direct":
        return f"{route}/{plan.get('task_match', 'none')}"
    return route


def write_route_transition(old_path: Path, new_path: Path, output: Path) -> Dict[str, Any]:
    """Compare answer-free planner outputs and save auditable route changes."""
    old_plans = _read_plans(old_path)
    new_plans = _read_plans(new_path)
    shared = sorted(old_plans.keys() & new_plans.keys())
    transitions = Counter()
    partial_cases = []
    requested_groups = {
        "v12_nonvisual_to_v13_partial": [],
        "v12_morphology_only_to_v13_partial": [],
        "v12_direct_to_v13_partial": [],
        "v12_phenotype_to_v13_nonvisual": [],
    }
    for key in shared:
        old = old_plans[key]
        new = new_plans[key]
        old_label = _route_label(old)
        new_label = _route_label(new)
        transitions[f"{old_label} -> {new_label}"] += 1
        detail = {
            "case_id": key[0],
            "question": key[1],
            "from": old_label,
            "to": new_label,
            "prototype_ids": new.get("selected_prototype_ids", []),
            "prototype_support_type": new.get("prototype_support_type", "none"),
            "prototype_coverage": new.get("prototype_coverage", "none"),
        }
        if new_label == "phenotype_direct/partial":
            partial_cases.append(detail)
            if old_label == "nonvisual":
                requested_groups["v12_nonvisual_to_v13_partial"].append(detail)
            if old_label == "morphology_only":
                requested_groups["v12_morphology_only_to_v13_partial"].append(detail)
            if old_label == "phenotype_direct/direct":
                requested_groups["v12_direct_to_v13_partial"].append(detail)
        if old_label.startswith("phenotype_direct/") and new_label == "nonvisual":
            requested_groups["v12_phenotype_to_v13_nonvisual"].append(detail)

    result = {
        "old_total": len(old_plans),
        "new_total": len(new_plans),
        "shared_total": len(shared),
        "transition_counts": dict(sorted(transitions.items())),
        "all_v13_partial": partial_cases,
        **requested_groups,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two answer-free Router plan files")
    parser.add_argument("--old", type=Path, required=True)
    parser.add_argument("--new", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = write_route_transition(args.old, args.new, args.output)
    print(json.dumps({
        "shared_total": result["shared_total"],
        "v13_partial": len(result["all_v13_partial"]),
        "output": str(args.output),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
