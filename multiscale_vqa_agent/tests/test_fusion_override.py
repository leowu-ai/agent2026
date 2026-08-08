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


if __name__ == "__main__":
    unittest.main()
