import json
from typing import Any, Dict, Iterable

from .clients import OpenAICompatibleClient, parse_json_response


ANSWERABILITY_CLASSES = {
    "directly_answerable",
    "inferable",
    "unanswerable",
}


ANSWERABILITY_SYSTEM_PROMPT = """You are the Answerability Gate for breast pathology WSI questions. Judge only whether the requested information can in principle be obtained from routine H&E WSI. Use only the supplied question and choices. Do not answer the question.

Return exactly one JSON object with answerability, confidence, and reason.

Classes:
- directly_answerable: the target is directly represented by H&E morphology, such as histologic type/grade, architecture, necrosis, microcalcification, lymphovascular invasion, in-situ disease, hyperplasia, adenosis, metaplasia, fibroadenoma, or fibrocystic change.
- inferable: a tissue-linked latent biological phenotype can be statistically predicted from morphology but is not directly observed, such as categorical ER, PR, or HER2 status.
- unanswerable: the target requires report, gross specimen, clinical history, treatment/procedure, follow-up, specimen orientation, assay records, or an exact value unavailable from H&E. This includes age, treatment, recommendations, exact size/distance/count/location/TNM/survival, report mentions, performed or pending tests, exact receptor percentages, IHC scores, and actual FISH results.

Judge information availability, not difficulty or current model capability. Categorical assay status may be inferable; an exact assay measurement is unanswerable. Choices clarify the requested information but are not evidence. Never infer or use a reference answer."""


class AnswerabilityAgent:
    def __init__(self, qwen: OpenAICompatibleClient):
        self.qwen = qwen

    def predict(self, question: str, choices: Iterable[str]) -> Dict[str, Any]:
        user = json.dumps(
            {
                "question": str(question),
                "choices": list(choices or []),
                "output_schema": {
                    "answerability": "directly_answerable|inferable|unanswerable",
                    "confidence": 0.0,
                    "reason": "one concise sentence",
                },
            },
            ensure_ascii=False,
        )
        try:
            parsed = parse_json_response(
                self.qwen.chat(
                    ANSWERABILITY_SYSTEM_PROMPT,
                    user,
                    max_tokens=160,
                    response_format={"type": "json_object"},
                    retries=2,
                )
            )
        except Exception as error:
            return self._continuation_fallback(f"Answerability service unavailable: {error}")
        return self._normalize(parsed)

    @staticmethod
    def _normalize(parsed: Any) -> Dict[str, Any]:
        if not isinstance(parsed, dict):
            return AnswerabilityAgent._continuation_fallback(
                "Answerability response was invalid; continuing without abstention."
            )
        label = str(parsed.get("answerability", "")).strip().lower()
        if label not in ANSWERABILITY_CLASSES:
            return AnswerabilityAgent._continuation_fallback(
                "Answerability label was invalid; continuing without abstention."
            )
        try:
            confidence = float(parsed.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        reason = str(parsed.get("reason", "")).strip()
        return {
            "answerability": label,
            "confidence": max(0.0, min(confidence, 1.0)),
            "reason": reason or "No reason was provided by the Answerability Agent.",
        }

    @staticmethod
    def _continuation_fallback(reason: str) -> Dict[str, Any]:
        # A gate failure must not masquerade as evidence that a question is unanswerable.
        return {
            "answerability": "directly_answerable",
            "confidence": 0.0,
            "reason": reason,
        }
