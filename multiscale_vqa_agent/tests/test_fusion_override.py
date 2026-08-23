import unittest

from multiscale_vqa_agent.fusion import FusionVerificationAgent
from multiscale_vqa_agent.fusion_evidence import structured_option_compatibility


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
        "mapping_complete": True,
        "option_alignment": {
            "choice_id": "A", "mapping_complete": True, "confidence": 0.5
        },
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

    def test_weak_pathology_suggestion_cannot_override_direct_candidate(self):
        evidence = parsed_counterevidence(
            is_decisive=False,
            evidence_direction="mixed",
            supports_proposed=False,
            contradicts_structured=False,
        )["counterevidence"]
        result = self.validate(
            proposed_answer(evidence), candidate("histological_type_label")
        )
        self.assertEqual(result["answer_id"], "A")
        self.assertTrue(result["override_rejected"])

    def test_molecular_direct_candidate_rejects_morphology_override(self):
        evidence = parsed_counterevidence()["counterevidence"]

        result = self.validate(
            proposed_answer(evidence), candidate("HER2_status_label")
        )

        self.assertEqual(result["answer_id"], "A")
        self.assertTrue(result["override_proposed"])
        self.assertTrue(result["override_rejected"])
        self.assertFalse(result["override_occurred"])

    def test_partial_mapped_candidate_rejects_unvalidated_override(self):
        result = self.validate(
            proposed_answer(),
            candidate("histological_type_label", task_match="partial"),
        )

        self.assertEqual(result["answer_id"], "A")
        self.assertTrue(result["override_proposed"])
        self.assertTrue(result["override_rejected"])
        self.assertFalse(result["override_occurred"])
        self.assertEqual(
            result["override_guard_reason"],
            "partial_requires_decisive_counterevidence",
        )

    def test_partial_decisive_visual_counterevidence_allows_override(self):
        evidence = parsed_counterevidence()["counterevidence"]
        result = self.validate(
            proposed_answer(evidence),
            candidate("histological_type_label", task_match="partial"),
        )
        self.assertEqual(result["answer_id"], "B")
        self.assertTrue(result["override_accepted"])
        self.assertEqual(
            result["override_evidence_type"], "patient_specific_visual"
        )

    def test_partial_generic_example_cannot_override(self):
        result = self.validate(
            proposed_answer(),
            candidate("histological_type_label", task_match="partial"),
        )
        self.assertEqual(result["answer_id"], "A")
        self.assertTrue(result["override_rejected"])

    def test_partial_proxy_rule_alone_cannot_override(self):
        result = self.validate(
            proposed_answer(),
            candidate("histological_type_label", task_match="partial"),
        )
        self.assertEqual(result["answer_id"], "A")
        self.assertFalse(result["override_accepted"])

    def test_unknown_partial_component_is_not_a_contradiction(self):
        summary = candidate("histological_type_label", task_match="partial")
        summary["option_compatibility"] = [{
            "choice_id": "A",
            "supported_fields": ["histological_type_label"],
            "missing_supporting_fields": ["tumor_size"],
            "uncertain_fields": ["tumor_size"],
            "contradicted_primary_fields": [],
            "contradicted_supporting_fields": [],
        }]
        result = self.validate(proposed_answer(), summary)
        self.assertEqual(result["answer_id"], "A")
        self.assertTrue(result["override_rejected"])

    def test_explicit_structured_contradiction_allows_partial_override(self):
        summary = candidate("histological_type_label", task_match="partial")
        summary["option_compatibility"] = [
            {
                "choice_id": "A",
                "contradicted_primary_fields": [],
                "contradicted_supporting_fields": ["HER2_status_label"],
            },
            {
                "choice_id": "B",
                "contradicted_primary_fields": [],
                "contradicted_supporting_fields": [],
            },
        ]
        result = self.validate(proposed_answer(), summary)
        self.assertEqual(result["answer_id"], "B")
        self.assertTrue(result["override_accepted"])
        self.assertEqual(
            result["override_guard_reason"],
            "partial_candidate_has_explicit_structured_contradiction",
        )

    def test_program_or_gene_context_alone_cannot_override_partial_candidate(self):
        result = self.validate(
            proposed_answer(),
            candidate("histological_type_label", task_match="partial"),
        )
        self.assertEqual(result["answer_id"], "A")
        self.assertEqual(
            result["override_evidence_type"], "insufficient_counterevidence"
        )

    def test_incomplete_partial_mapping_remains_free_to_choose(self):
        summary = candidate("histological_type_label", task_match="partial")
        summary["mapping_complete"] = False
        summary["option_alignment"]["mapping_complete"] = False
        result = self.validate(proposed_answer(), summary)
        self.assertEqual(result["answer_id"], "B")
        self.assertTrue(result["override_accepted"])
        self.assertEqual(result["override_guard_reason"], "mapping_incomplete")


class FusionReasoningModeTest(unittest.TestCase):
    @staticmethod
    def context(finalize=False):
        return {
            "knowledge": {
                "matched_concepts": [{"id": "concept"}],
                "evidence_limitations": [{"id": "limit"}],
                "proxy_evidence_rules": [{"id": "proxy"}],
                "forced_choice_rules": [{"id": "forced"}],
                "reasoning_examples": [{
                    "id": "example", "patient_specific": False
                }],
            },
            "final_verifier": {
                "next_action": "finalize" if finalize else "answer",
                "evidence_sufficient": not finalize,
            },
            "search_exhausted": finalize,
            "final_evidence_state": "insufficient" if finalize else "sufficient",
        }

    def test_supported_candidate_suppresses_generic_forced_choice_context(self):
        summary = candidate("histological_type_label", task_match="partial")
        context = FusionVerificationAgent._fusion_agent_context(
            summary, ["structured", "proposed"], self.context()
        )
        self.assertEqual(
            context["kb_usage_metadata"]["kb_reasoning_mode"],
            "structured_supported",
        )
        self.assertEqual(context["knowledge"]["forced_choice_rules"], [])
        self.assertEqual(context["knowledge"]["reasoning_examples"], [])
        self.assertEqual(context["knowledge"]["proxy_evidence_rules"], [])

    def test_none_or_no_candidate_gets_full_context(self):
        summary = {"task_match": "none"}
        context = FusionVerificationAgent._fusion_agent_context(
            summary, ["a", "b"], self.context(finalize=True)
        )
        metadata = context["kb_usage_metadata"]
        self.assertEqual(metadata["kb_reasoning_mode"], "evidence_exhausted")
        self.assertTrue(metadata["full_forced_choice_context_used"])
        self.assertEqual(
            metadata["supplied_to_final_fusion_reasoning_example_ids"],
            ["example"],
        )

    def test_finalize_partial_without_candidate_gets_full_context(self):
        summary = {"task_match": "partial", "mapping_complete": False}
        context = FusionVerificationAgent._fusion_agent_context(
            summary, ["a", "b"], self.context(finalize=True)
        )
        self.assertEqual(
            context["kb_usage_metadata"]["kb_reasoning_mode"],
            "evidence_exhausted",
        )
        self.assertTrue(
            context["kb_usage_metadata"]["full_forced_choice_context_used"]
        )

    def test_finalize_preserves_partial_anchor_and_only_gap_guidance(self):
        summary = candidate("histological_type_label", task_match="partial")
        context = FusionVerificationAgent._fusion_agent_context(
            summary, ["structured", "proposed"], self.context(finalize=True)
        )
        metadata = context["kb_usage_metadata"]
        self.assertEqual(metadata["kb_reasoning_mode"], "structured_supported")
        self.assertTrue(metadata["structured_components_remain_anchored"])
        self.assertFalse(metadata["full_forced_choice_context_used"])
        self.assertEqual(context["knowledge"]["reasoning_examples"], [])
        self.assertEqual(context["knowledge"]["forced_choice_rules"], [])
        self.assertEqual(
            metadata["supplied_to_final_fusion_proxy_rule_ids"], ["proxy"]
        )

    def test_reasoning_examples_remain_non_patient_specific(self):
        context = FusionVerificationAgent._fusion_agent_context(
            {"task_match": "none"}, ["a", "b"], self.context(finalize=True)
        )
        self.assertTrue(all(
            row.get("patient_specific") is False
            for row in context["knowledge"]["reasoning_examples"]
        ))


class AtomicOptionCompatibilityTest(unittest.TestCase):
    def test_unknown_and_contradicted_components_remain_distinct(self):
        rows = structured_option_compatibility(
            [
                {"field": "ER_status_label", "predicted_label": "positive"},
                {"field": "PR_status_label", "predicted_label": "negative"},
            ],
            ["ER+/PR+", "ER+/PR-", "ER positive"],
            ["ER_status_label"],
            ["PR_status_label"],
        )
        self.assertEqual(rows[0]["supported_fields"], ["ER_status_label"])
        self.assertEqual(
            rows[0]["contradicted_supporting_fields"], ["PR_status_label"]
        )
        self.assertEqual(rows[1]["contradicted_supporting_fields"], [])
        self.assertIn("PR_status_label", rows[2]["uncertain_fields"])
        self.assertNotIn(
            "PR_status_label", rows[2]["contradicted_supporting_fields"]
        )


if __name__ == "__main__":
    unittest.main()
