import csv
import json
import os
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


def normalize_answer(value: Any) -> str:
    text = str(value or "").lower().replace("infiltrating", "invasive")
    return " ".join(re.findall(r"[a-z0-9]+", text))


class LiveAccuracyTracker:
    def __init__(
        self,
        snapshot_path: Path,
        history_path: Path,
        selected_total: int,
        existing_answers: Optional[Path] = None,
    ):
        self.snapshot_path = Path(snapshot_path)
        self.history_path = Path(history_path)
        self.selected_total = int(selected_total)
        self.processed = self.scorable = self.correct = self.errors = 0
        self.supported_processed = self.supported_correct = 0
        self.unsupported_processed = self.unsupported_correct = 0
        self.json_parse_failures = self.answers_outside_choices = 0
        self.multifield_incomplete = self.overrides = 0
        self.override_structured_correct = self.override_agent_correct = 0
        self.per_task = defaultdict(self._empty_stats)
        self.per_task_match = defaultdict(self._empty_stats)
        self.last = None
        self.snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        if existing_answers and Path(existing_answers).exists():
            self._restore(Path(existing_answers))

    @staticmethod
    def _empty_stats():
        return {"processed": 0, "scorable": 0, "correct": 0, "errors": 0}

    def _restore(self, path: Path):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    self.update(json.loads(line), write_files=False)
                except json.JSONDecodeError:
                    continue
        self.save_snapshot()

    def update(self, row: Dict[str, Any], write_files: bool = True) -> Dict[str, Any]:
        plan = row.get("plan", {})
        fields = plan.get("target_phenotypes", []) or ["unrouted"]
        task = "+".join(str(value) for value in fields)
        supported = bool(plan.get("supported", False))
        task_match = str(row.get("task_match", plan.get("task_match", "direct" if supported else "none")))
        error = bool(row.get("error"))
        answer_data = row.get("agent_answer", {}) or {}
        predicted = answer_data.get("answer")
        reference = row.get("reference_answer", row.get("input", {}).get("Answer"))
        choices = list(row.get("choices", row.get("input", {}).get("Choice", [])) or [])
        scorable = not error and predicted is not None and reference is not None
        is_correct = bool(scorable and normalize_answer(predicted) == normalize_answer(reference))
        answer_in_choices = bool(predicted in choices) if choices else True
        json_success = bool(row.get("json_parse_success", answer_data.get("json_parse_success", False)))
        requested = row.get("requested_fields", fields)
        missing = row.get("missing_fields", [])
        incomplete = bool(len(requested or []) > 1 and missing)
        override = bool(row.get("override_occurred", answer_data.get("override_occurred", False)))
        structured = row.get("structured_candidate_answer")
        structured_correct = bool(
            structured is not None and reference is not None and
            normalize_answer(structured) == normalize_answer(reference)
        )

        self.processed += 1
        self.errors += int(error)
        self.scorable += int(scorable)
        self.correct += int(is_correct)
        self.supported_processed += int(supported)
        self.supported_correct += int(supported and is_correct)
        self.unsupported_processed += int(not supported)
        self.unsupported_correct += int((not supported) and is_correct)
        self.json_parse_failures += int(not error and not json_success)
        self.answers_outside_choices += int(not answer_in_choices)
        self.multifield_incomplete += int(incomplete)
        self.overrides += int(override)
        self.override_structured_correct += int(override and structured_correct)
        self.override_agent_correct += int(override and is_correct)

        for stats in (self.per_task[task], self.per_task_match[task_match]):
            stats["processed"] += 1
            stats["errors"] += int(error)
            stats["scorable"] += int(scorable)
            stats["correct"] += int(is_correct)

        self.last = {
            "case_id": row.get("case_id"),
            "question": row.get("question"),
            "task": task,
            "task_match": task_match,
            "supported": supported,
            "correct": is_correct,
            "error": row.get("error"),
            "predicted": predicted,
            "reference": reference,
            "answer_in_choices": answer_in_choices,
            "json_parse_success": json_success,
            "override_occurred": override,
        }
        evaluation = {
            "scorable": scorable,
            "correct": is_correct,
            "normalized_prediction": normalize_answer(predicted),
            "normalized_reference": normalize_answer(reference),
        }
        if write_files:
            self._append_history()
            self.save_snapshot()
        return evaluation

    @staticmethod
    def _ratio(numerator: int, denominator: int):
        return float(numerator / denominator) if denominator else None

    def _summarize_groups(self, groups: Dict[str, Dict[str, int]]) -> Dict[str, Any]:
        result = {}
        for key, stats in sorted(groups.items()):
            values = dict(stats)
            values["accuracy"] = self._ratio(values["correct"], values["processed"])
            values["accuracy_scorable"] = self._ratio(values["correct"], values["scorable"])
            result[key] = values
        return result

    def snapshot(self) -> Dict[str, Any]:
        return {
            "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "selected_total": self.selected_total,
            "processed": self.processed,
            "remaining": max(self.selected_total - self.processed, 0),
            "completion": self._ratio(self.processed, self.selected_total),
            "scorable": self.scorable,
            "correct": self.correct,
            "errors": self.errors,
            "accuracy": self._ratio(self.correct, self.processed),
            "accuracy_scorable": self._ratio(self.correct, self.scorable),
            "supported_processed": self.supported_processed,
            "supported_correct": self.supported_correct,
            "supported_accuracy": self._ratio(self.supported_correct, self.supported_processed),
            "unsupported_processed": self.unsupported_processed,
            "unsupported_correct": self.unsupported_correct,
            "unsupported_accuracy": self._ratio(self.unsupported_correct, self.unsupported_processed),
            "json_parse_failures": self.json_parse_failures,
            "answers_outside_choices": self.answers_outside_choices,
            "multifield_incomplete": self.multifield_incomplete,
            "structured_overrides": self.overrides,
            "override_structured_accuracy": self._ratio(self.override_structured_correct, self.overrides),
            "override_agent_accuracy": self._ratio(self.override_agent_correct, self.overrides),
            "per_task_match": self._summarize_groups(self.per_task_match),
            "per_task": self._summarize_groups(self.per_task),
            "last": self.last,
        }

    def save_snapshot(self):
        temporary = self.snapshot_path.with_suffix(self.snapshot_path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(self.snapshot(), handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(self.snapshot_path)

    def _append_history(self):
        snapshot = self.snapshot()
        row = {
            "updated_at": snapshot["updated_at"],
            "processed": self.processed,
            "selected_total": self.selected_total,
            "correct": self.correct,
            "errors": self.errors,
            "accuracy": snapshot["accuracy"],
            "accuracy_scorable": snapshot["accuracy_scorable"],
            "supported_accuracy": snapshot["supported_accuracy"],
            "unsupported_accuracy": snapshot["unsupported_accuracy"],
            "json_parse_failures": self.json_parse_failures,
            "answers_outside_choices": self.answers_outside_choices,
            "multifield_incomplete": self.multifield_incomplete,
            "structured_overrides": self.overrides,
            "case_id": self.last.get("case_id") if self.last else None,
            "task": self.last.get("task") if self.last else None,
            "task_match": self.last.get("task_match") if self.last else None,
            "last_correct": self.last.get("correct") if self.last else None,
            "last_error": self.last.get("error") if self.last else None,
        }
        exists = self.history_path.exists() and self.history_path.stat().st_size > 0
        with self.history_path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            if not exists:
                writer.writeheader()
            writer.writerow(row)
            handle.flush()
            os.fsync(handle.fileno())
