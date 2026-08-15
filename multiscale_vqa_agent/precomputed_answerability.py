import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple


AnswerabilityKey = Tuple[str, str]


def normalize_answerability_key(case_id: Any, question: Any) -> AnswerabilityKey:
    normalized_case = str(case_id or "").strip().lower()
    normalized_question = re.sub(r"\s+", " ", str(question or "").strip()).lower()
    return normalized_case, normalized_question


class PrecomputedAnswerabilityStore:
    """Strict lookup for frozen answerability decisions."""

    def __init__(self, path: str):
        self.path = Path(path)
        self.decisions: Dict[AnswerabilityKey, Dict[str, Any]] = {}
        with self.path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                key = normalize_answerability_key(
                    row.get("case_id"), row.get("question")
                )
                if not all(key):
                    raise ValueError(
                        f"Precomputed answerability line {line_number} is missing "
                        "case_id or question"
                    )
                if key in self.decisions:
                    raise ValueError(
                        "Duplicate precomputed answerability prediction for "
                        f"case_id={row.get('case_id')!r}, question={row.get('question')!r}"
                    )
                can_answer = row.get("predicted_can_answer")
                if not isinstance(can_answer, bool):
                    raise ValueError(
                        f"Precomputed answerability line {line_number} has no boolean "
                        "predicted_can_answer"
                    )
                try:
                    confidence = float(row.get("answerability_confidence", 0.0))
                except (TypeError, ValueError) as error:
                    raise ValueError(
                        f"Precomputed answerability line {line_number} has invalid confidence"
                    ) from error
                self.decisions[key] = {
                    "can_answer": can_answer,
                    "confidence": max(0.0, min(confidence, 1.0)),
                    "reason": str(row.get("answerability_reason", "")),
                    "fallback_used": bool(
                        row.get("answerability_fallback_used", False)
                    ),
                }
        if not self.decisions:
            raise ValueError(f"Precomputed answerability file is empty: {self.path}")

    def lookup(self, case_id: Any, question: Any) -> Dict[str, Any]:
        key = normalize_answerability_key(case_id, question)
        try:
            return dict(self.decisions[key])
        except KeyError as error:
            raise KeyError(
                "Missing precomputed answerability prediction for "
                f"case_id={case_id!r}, question={question!r}"
            ) from error

    def validate_items(self, items: Iterable[Dict[str, Any]]) -> None:
        expected: Dict[AnswerabilityKey, Tuple[str, str]] = {}
        for item in items:
            case_id = str(item.get("Id", item.get("case_id", "")))[:12]
            question = str(item.get("Question", item.get("question", "")))
            key = normalize_answerability_key(case_id, question)
            if not all(key):
                raise ValueError("VQA item is missing case_id or question")
            if key in expected:
                raise ValueError(
                    "Duplicate VQA item for precomputed answerability: "
                    f"case_id={case_id!r}, question={question!r}"
                )
            expected[key] = (case_id, question)

        missing = sorted(set(expected) - set(self.decisions))
        extra = sorted(set(self.decisions) - set(expected))
        if missing or extra:
            details = []
            if missing:
                case_id, question = expected[missing[0]]
                details.append(
                    f"missing={len(missing)} first_missing=({case_id!r}, {question!r})"
                )
            if extra:
                details.append(
                    f"extra={len(extra)} first_extra={extra[0]!r}"
                )
            raise ValueError(
                "Precomputed answerability does not exactly match the VQA items: "
                + "; ".join(details)
            )

    def summary(self) -> Dict[str, int]:
        true_count = sum(row["can_answer"] for row in self.decisions.values())
        return {
            "total": len(self.decisions),
            "predicted_true": true_count,
            "predicted_false": len(self.decisions) - true_count,
        }
