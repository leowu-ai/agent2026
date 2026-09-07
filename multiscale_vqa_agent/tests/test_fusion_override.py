import json
import unittest
from types import SimpleNamespace

from multiscale_vqa_agent.fusion import (
    ALIGNMENT_SYSTEM,
    MINIMAL_NONE_SYSTEM,
    REPAIR_SYSTEM,
    FusionVerificationAgent,
)
from multiscale_vqa_agent.fusion_evidence import load_fusion_prompt


def structured(field):
    return {"predictions": [{"field": field}]}


def parsed_counterevidence(**overrides):
    counterevidence = {
        "is_decisive": True,
        "evidence_direction": "supports_proposed",
        "supports_proposed": True,
        "contradicts_structured": True,
        "confidence": 0.9,
        "visible_feature": "A morphology observation evaluated by the arbiter.",
        "decisive_reason": "The arbiter judged it to support the proposed answer.",
        "structured_failure": "The structured candidate may not represent this morphology.",
    }
    counterevidence.update(overrides)
    return {"counterevidence": counterevidence}


def candidate(field, task_match="direct", confidence=0.4):
    return {
        "task_match": task_match,
        "structured_candidate_id": "A",
        "structured_candidate_answer": "structured",
        "structured_candidate_confidence": confidence,
        "option_alignment": {"confidence": 0.5},
        "predictions": [{
            "field": field,
            "fused_probability_for_predicted_class": 0.55,
            "cross_scale_agreement": 0.5,
            "validation_quality": 0.4,
        }],
    }


def proposed_answer(counterevidence=None):
    parsed = {
        "answer_id": "B",
        "confidence": 0.6,
        "explanation": "The visual evidence favors the proposed answer.",
        "limitations": "Visual evidence may be imperfect.",
    }
    if counterevidence is not None:
        parsed["counterevidence"] = counterevidence
    return parsed


class CounterevidenceValidationTest(unittest.TestCase):
    def test_supports_structured_is_not_valid_override(self):
        parsed = parsed_counterevidence(
            is_decisive=False,
            evidence_direction="supports_structured",
            supports_proposed=False,
            contradicts_structured=False,
            confidence=0.95,
        )
        self.assertFalse(
            FusionVerificationAgent._valid_counterevidence(
                parsed, structured("histological_type_label")
            )
        )

    def test_decisive_support_for_proposed_is_valid_for_morphology(self):
        self.assertTrue(
            FusionVerificationAgent._valid_counterevidence(
                parsed_counterevidence(), structured("histological_type_label")
            )
        )

    def test_molecular_field_cannot_be_overridden_by_morphology(self):
        self.assertFalse(
            FusionVerificationAgent._valid_counterevidence(
                parsed_counterevidence(), structured("HER2_status_label")
            )
        )

    def test_low_counterevidence_confidence_is_rejected(self):
        self.assertFalse(
            FusionVerificationAgent._valid_counterevidence(
                parsed_counterevidence(confidence=0.4),
                structured("histological_type_label"),
            )
        )

    def test_mixed_direction_is_rejected(self):
        self.assertFalse(
            FusionVerificationAgent._valid_counterevidence(
                parsed_counterevidence(evidence_direction="mixed"),
                structured("histological_type_label"),
            )
        )


class DirectOverridePolicyTest(unittest.TestCase):
    def setUp(self):
        self.agent = FusionVerificationAgent(client=None)
        self.choices = ["structured", "proposed"]

    def validate(self, parsed, structured_summary):
        return self.agent._validate(
            parsed,
            self.choices,
            structured_summary,
            raw="{}",
            status="parsed",
            retry_count=0,
        )

    def test_low_confidence_direct_candidate_still_requires_counterevidence(self):
        structured_summary = candidate("histological_type_label", confidence=0.4)
        self.assertFalse(self.agent._high_trust_candidate(structured_summary))

        result = self.validate(proposed_answer(), structured_summary)

        self.assertEqual(result["answer_id"], "A")
        self.assertTrue(result["override_proposed"])
        self.assertTrue(result["override_rejected"])
        self.assertFalse(result["override_occurred"])

    def test_valid_morphology_counterevidence_allows_direct_override(self):
        evidence = parsed_counterevidence()["counterevidence"]

        result = self.validate(
            proposed_answer(evidence), candidate("histological_type_label")
        )

        self.assertEqual(result["answer_id"], "B")
        self.assertTrue(result["override_proposed"])
        self.assertFalse(result["override_rejected"])
        self.assertTrue(result["override_occurred"])

    def test_molecular_direct_candidate_rejects_morphology_override(self):
        evidence = parsed_counterevidence()["counterevidence"]

        result = self.validate(
            proposed_answer(evidence), candidate("HER2_status_label")
        )

        self.assertEqual(result["answer_id"], "A")
        self.assertTrue(result["override_proposed"])
        self.assertTrue(result["override_rejected"])
        self.assertFalse(result["override_occurred"])

    def test_partial_route_keeps_current_free_arbitration(self):
        result = self.validate(
            proposed_answer(),
            candidate("histological_type_label", task_match="partial"),
        )

        self.assertEqual(result["answer_id"], "B")
        self.assertTrue(result["override_proposed"])
        self.assertFalse(result["override_rejected"])
        self.assertTrue(result["override_occurred"])


class BlockedChoiceTest(unittest.TestCase):
    CHOICES = ["invasive carcinoma", "Not Mentioned", "benign tissue"]

    @staticmethod
    def plan():
        return SimpleNamespace(
            question="Which diagnosis is most likely?",
            support_reason="Evidence is limited.",
            evidence_route="morphology_only",
            selected_prototype_ids=[],
        )

    def test_visible_options_preserve_original_ids(self):
        options = FusionVerificationAgent._visible_choice_options(self.CHOICES)

        self.assertEqual(
            options,
            [
                {"id": "A", "text": "invasive carcinoma"},
                {"id": "C", "text": "benign tissue"},
            ],
        )

    def test_prompts_do_not_instruct_about_blocked_choice(self):
        prompts = (
            MINIMAL_NONE_SYSTEM,
            REPAIR_SYSTEM,
            ALIGNMENT_SYSTEM,
            load_fusion_prompt(),
        )

        for prompt in prompts:
            self.assertNotIn("not mentioned", prompt.casefold())
            self.assertNotIn("unavailable-style", prompt.casefold())

    def test_blocked_model_answer_is_rejected_on_initial_and_retry_calls(self):
        class CapturingClient:
            enabled = True

            def __init__(self):
                self.calls = []

            def chat(self, system, user, **kwargs):
                self.calls.append((system, user, kwargs))
                return json.dumps({
                    "answer_id": "B",
                    "confidence": 0.9,
                    "explanation": "Unavailable option selected.",
                    "limitations": "None.",
                })

        client = CapturingClient()
        agent = FusionVerificationAgent(client)
        result = agent._answer_prepared(
            self.plan(),
            self.CHOICES,
            {"task_match": "none", "structured_candidate_confidence": 0.0},
            relations=None,
            pathology={"description": "Visible breast tissue."},
        )

        self.assertEqual(len(client.calls), 2)
        self.assertNotEqual(result["answer_id"], "B")
        self.assertNotEqual(result["answer"].casefold(), "not mentioned")
        for system, user, _ in client.calls:
            self.assertNotIn("not mentioned", system.casefold())
            self.assertNotIn("not mentioned", user.casefold())

    def test_blocked_structured_candidate_is_not_used_by_fallback(self):
        agent = FusionVerificationAgent(client=None)
        structured_summary = {
            "task_match": "direct",
            "structured_candidate_id": "B",
            "structured_candidate_answer": "Not Mentioned",
            "structured_candidate_confidence": 0.9,
        }

        result = agent._fallback(
            self.plan(), self.CHOICES, structured_summary, None, "test", 0
        )

        self.assertNotEqual(result["answer_id"], "B")
        self.assertNotEqual(result["answer"].casefold(), "not mentioned")

    def test_option_alignment_cannot_see_or_restore_blocked_choice(self):
        class AlignmentClient:
            enabled = True

            def __init__(self):
                self.user_payload = None

            def chat(self, system, user, **kwargs):
                self.user_payload = user
                return json.dumps({
                    "choice_id": "B",
                    "mapping_complete": True,
                    "confidence": 0.95,
                    "reason": "Selected by the alignment model.",
                })

        client = AlignmentClient()
        agent = FusionVerificationAgent(client)
        structured_summary = {
            "task_match": "direct",
            "predictions": [{
                "field": "histological_type_label",
                "predicted_label": "unmapped lesion",
            }],
            "primary_fields": ["histological_type_label"],
            "requested_fields": ["histological_type_label"],
            "literal_match_id": None,
            "literal_matches": [],
        }

        agent._attach_option_alignment(
            self.plan(), self.CHOICES, structured_summary
        )

        self.assertNotIn("not mentioned", client.user_payload.casefold())
        self.assertFalse(structured_summary["option_alignment"]["mapping_complete"])
        self.assertIsNone(structured_summary["option_alignment"]["choice_id"])
        self.assertIsNone(structured_summary.get("structured_candidate_id"))

if __name__ == "__main__":
    unittest.main()
