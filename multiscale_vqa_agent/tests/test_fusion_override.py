import unittest

from multiscale_vqa_agent.fusion import FusionVerificationAgent


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


def candidate(field, task_match="direct", confidence=0.4, high_trust=True):
    return {
        "task_match": task_match,
        "structured_candidate_id": "A",
        "structured_candidate_answer": "structured",
        "structured_candidate_confidence": confidence,
        "option_alignment": {"confidence": 0.9 if high_trust else 0.5},
        "confidence_factors": {
            "patient_evidence_strength": 0.85 if high_trust else 0.6,
        },
        "predictions": [{
            "field": field,
            "fused_probability_for_predicted_class": 0.9 if high_trust else 0.55,
            "cross_scale_agreement": 1.0 if high_trust else 0.5,
            "validation_quality": 0.8 if high_trust else 0.4,
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

    def test_low_trust_direct_candidate_allows_free_arbitration(self):
        structured_summary = candidate(
            "histological_type_label", confidence=0.4, high_trust=False
        )
        self.assertFalse(self.agent._high_trust_candidate(structured_summary))

        result = self.validate(proposed_answer(), structured_summary)

        self.assertEqual(result["answer_id"], "B")
        self.assertTrue(result["override_proposed"])
        self.assertFalse(result["override_rejected"])
        self.assertTrue(result["override_occurred"])

    def test_high_trust_direct_candidate_rejects_invalid_counterevidence(self):
        structured_summary = candidate("histological_type_label")
        self.assertTrue(self.agent._high_trust_candidate(structured_summary))

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


if __name__ == "__main__":
    unittest.main()
