import json
from typing import Any, Dict, Iterable

from .clients import OpenAICompatibleClient, parse_json_response


ANSWERABILITY_SYSTEM_PROMPT = """You are the Answerability Gate for breast pathology WSI questions. Judge only whether the requested information can reasonably be answered from the current pathology WSI. Use only the supplied question and choices. Do not answer the question.

Return exactly one JSON object with can_answer, confidence, and reason. can_answer must be a JSON boolean, never a string.

- can_answer=true: the target is either directly represented by H&E morphology or is a categorical tissue-linked biological phenotype that can reasonably be predicted from morphology. This includes diagnosis, histologic type/subtype/grade, architecture, DCIS/LCIS, necrosis, microcalcification, lymphovascular invasion, hyperplasia, adenosis, metaplasia, fibroadenoma, fibrocystic change, and categorical ER/PR/HER2 status. Under this benchmark protocol, categorical ER/PR/HER2 positive/negative/equivocal questions must be treated as answerable through morphology-linked statistical prediction even though H&E does not provide a definitive measured assay result.
- can_answer=false: the target requires a pathology report, gross specimen, clinical metadata, treatment/procedure, follow-up, specimen orientation, exact measurement, or actual assay result. This includes age, symptoms, treatment, recommendations, exact size/percentage/distance/count/location/TNM/stage/survival, performed or pending tests, exact receptor percentages, quantitative IHC scores, and actual FISH or amplification results.

Judge information availability, not current thumbnail visibility, difficulty, or whether the answering model is certain it will be correct. The confidence field measures confidence in the can_answer classification itself, not confidence in the eventual clinical answer. Categorical ER/PR/HER2 status is answerable in this predictive task; exact percentages, quantitative IHC scores, FISH, amplification, or other actual assay measurements are not. Choices clarify the requested target but are not evidence."""


class AnswerabilityAgent:
    def __init__(self, qwen: OpenAICompatibleClient):
        self.qwen = qwen

    def predict(self, question: str, choices: Iterable[str]) -> Dict[str, Any]:
        user = json.dumps(
            {
                "question": str(question),
                "choices": list(choices or []),
                "output_schema": {
                    "can_answer": True,
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
        can_answer = parsed.get("can_answer")
        if not isinstance(can_answer, bool):
            return AnswerabilityAgent._continuation_fallback(
                "Answerability boolean was invalid; continuing without abstention."
            )
        try:
            confidence = float(parsed.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        reason = str(parsed.get("reason", "")).strip()
        return {
            "can_answer": can_answer,
            "confidence": max(0.0, min(confidence, 1.0)),
            "reason": reason or "No reason was provided by the Answerability Agent.",
            "fallback_used": False,
        }

    @staticmethod
    def _continuation_fallback(reason: str) -> Dict[str, Any]:
        # A gate failure must not masquerade as evidence that a question is unanswerable.
        return {
            "can_answer": True,
            "confidence": 0.0,
            "reason": reason,
            "fallback_used": True,
        }
