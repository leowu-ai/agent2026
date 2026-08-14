import json
from typing import Any, Dict, Iterable

from .clients import OpenAICompatibleClient, parse_json_response


ANSWERABILITY_SYSTEM_PROMPT = """You are the Answerability Gate for breast pathology WSI questions.

Your only task is to decide whether the requested target can reasonably be answered from the current pathology WSI under the benchmark protocol.

Use only the supplied question and choices.
Do not answer the medical question itself.

Return exactly one JSON object with:
- can_answer: JSON boolean
- confidence: number from 0 to 1
- reason: one concise sentence

IMPORTANT DECISION PRINCIPLE:
Classify the information TARGET being requested, not the wording or method used to describe it.

Words such as "test", "testing", "staining", "immunohistochemistry", "IHC",
"reaction", "examination", "analysis", or "results" do NOT by themselves make
a question unanswerable.

Use the answer choices to determine the semantic target and the required
granularity of the answer.

can_answer=true when the requested target is:

1. Directly represented by H&E morphology, including:
   - histologic diagnosis
   - tumor type or subtype
   - histologic or nuclear grade
   - Nottingham histologic grade or histology-derived Nottingham score
   - architectural pattern
   - DCIS or LCIS
   - necrosis or comedonecrosis
   - microcalcification
   - lymphovascular, vascular, venous, or angioinvasion
   - hyperplasia
   - adenosis
   - metaplasia
   - fibroadenoma
   - fibrocystic change
   - other categorical tissue morphology

2. A categorical tissue-linked biological phenotype that can reasonably be
   predicted statistically from morphology under this benchmark protocol.

   In particular, categorical ER, PR, and HER2 biological status is answerable.

   Examples of answerable output granularity:
   - ER positive vs negative
   - PR positive vs negative
   - HER2 positive vs negative vs equivocal
   - combinations such as ER+/PR+/HER2-
   - categorical receptor-status alternatives

   This remains true even if the question phrases the categorical phenotype as:
   - "the result of the test"
   - "the result of staining"
   - "the IHC result"
   - "the protein reaction"
   - "the examination result"

   If the requested output is still only the categorical biological state,
   classify it as answerable.

can_answer=false when answering requires information that is not reasonably
recoverable or predictable from the pathology WSI, including:

1. Clinical or patient metadata:
   - age
   - symptoms or history
   - medications
   - treatment
   - consent
   - performance status
   - follow-up
   - survival

2. Procedure, report, or specimen metadata:
   - which test or stain was performed
   - whether a test was performed
   - which test is pending
   - where a test was performed
   - treatment or procedure performed
   - recommendations
   - specimen orientation
   - specimen site or report wording

3. Exact case-level physical or anatomic quantities:
   - exact tumor size or dimensions
   - exact percentage
   - exact distance
   - exact count
   - exact location
   - margin distance/status when case-level specimen context is required
   - TNM, AJCC, or pathological stage

4. Actual assay-specific quantitative or molecular measurements:
   - exact receptor staining percentage
   - exact quantitative IHC score
   - H-score or other continuous assay score
   - exact FISH result
   - gene amplification result
   - assay-specific molecular measurement

CRITICAL DISTINCTIONS:

A. "What test was performed?"
   -> asks for procedure/report metadata
   -> can_answer=false

B. "Was HER2 positive or negative by immunohistochemistry?"
   with choices such as positive / negative / equivocal
   -> asks for categorical HER2 biological status
   -> can_answer=true

C. "What percentage of nuclei stained positive for ER?"
   -> asks for an exact assay measurement
   -> can_answer=false

D. "What was the ER/PR staining result?"
   with choices such as positive / negative / equivocal
   -> asks for categorical receptor phenotype
   -> can_answer=true

E. "Was HER2 gene amplification detected by FISH?"
   -> asks for an actual molecular assay result
   -> can_answer=false

F. "What is the Nottingham grade?"
   or a histology-derived Nottingham score/grade
   -> morphology-derived grading target
   -> can_answer=true

G. "Was the Nottingham score determined in the report?"
   -> asks about report availability rather than morphology
   -> can_answer=false

Judge information availability and target type, not current thumbnail visibility,
difficulty, or whether the eventual answering model is likely to be correct.

The confidence field measures confidence in this answerability classification,
not confidence in the eventual clinical answer.

When question wording and choices appear to conflict, identify what the choices
actually require the model to output and classify that requested output type.
Choices clarify the target but are not evidence for the correct medical answer."""


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
