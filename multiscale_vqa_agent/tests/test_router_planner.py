import unittest

from multiscale_vqa_agent.registry import PrototypeAwarePlanner


class FakeRegistry:
    prototype_id_to_field = {
        "P001": "histological_type_label",
        "P006": "histologic_grade_label",
        "P007": "lymphovascular_invasion_label",
        "P010": "microcalcification_binary",
        "P013": "ER_status_label",
    }
    field_to_name = {
        "histological_type_label": "histological type",
        "histologic_grade_label": "histologic grade",
        "lymphovascular_invasion_label": "lymphovascular invasion",
        "microcalcification_binary": "microcalcification",
        "ER_status_label": "ER status",
    }
    vocabs = {
        1024: {
            "phenotype_task_types": {
                "histological type": "multiclass",
                "histologic grade": "multiclass",
                "lymphovascular invasion": "multiclass",
                "microcalcification": "binary",
                "ER status": "multiclass",
            }
        }
    }


class FakeClient:
    enabled = False


class RouterPlannerTest(unittest.TestCase):
    def setUp(self):
        self.planner = PrototypeAwarePlanner(FakeRegistry(), FakeClient())

    def normalize(self, parsed):
        return self.planner._normalize_llm_plan(
            "TCGA-XX-0001", "test question", ["A", "B"], parsed
        )

    def test_complete_maps_to_direct(self):
        plan = self.normalize({
            "prototype_ids": ["P006"],
            "prototype_support_type": "target_evidence",
            "prototype_coverage": "complete",
        })
        self.assertEqual(plan.target_phenotypes, ["histologic_grade_label"])
        self.assertEqual(plan.selected_prototype_ids, ["P006"])
        self.assertEqual(plan.task_match, "direct")

        self.assertEqual(plan.prototype_coverage, "complete")
        self.assertNotIn("local_morphology_useful", plan.to_dict())
        self.assertNotIn("requires_unavailable_context", plan.to_dict())

    def test_partial_keeps_single_prototype(self):
        plan = self.normalize({
            "prototype_ids": ["P007"],
            "prototype_support_type": "target_evidence",
            "prototype_coverage": "partial",
            "phenotype_relevance_score": 0.6,
        })
        self.assertEqual(
            plan.target_phenotypes, ["lymphovascular_invasion_label"]
        )
        self.assertEqual(plan.selected_prototype_ids, ["P007"])
        self.assertEqual(plan.task_match, "partial")
        self.assertTrue(plan.supported)

    def test_partial_keeps_multiple_prototypes(self):
        plan = self.normalize({
            "prototype_ids": ["P007", "P010"],
            "prototype_support_type": "target_evidence",
            "prototype_coverage": "partial",
            "phenotype_relevance_score": 0.7,
        })
        self.assertEqual(
            plan.target_phenotypes,
            ["lymphovascular_invasion_label", "microcalcification_binary"],
        )
        self.assertEqual(plan.task_match, "partial")
        self.assertTrue(plan.supported)

    def test_none_clears_valid_prototype_and_uses_morphology(self):
        plan = self.normalize({
            "prototype_ids": ["P001"],
            "prototype_support_type": "none",
            "prototype_coverage": "none",
            "use_pathology_agent": False,
        })
        self.assertEqual(plan.selected_prototype_ids, [])
        self.assertEqual(plan.target_phenotypes, [])
        self.assertEqual(plan.task_match, "none")
        self.assertTrue(plan.use_pathology_agent)

    def test_no_prototype_always_uses_morphology_only(self):
        plan = self.normalize({
            "prototype_ids": [],
            "prototype_support_type": "none",
            "prototype_coverage": "none",
            "use_pathology_agent": True,
        })
        self.assertEqual(plan.selected_prototype_ids, [])
        self.assertEqual(plan.evidence_route, "morphology_only")
        self.assertTrue(plan.use_pathology_agent)
        self.assertFalse(plan.supported)

    def test_invalid_partial_prototype_falls_back_to_morphology(self):
        plan = self.normalize({
            "prototype_ids": ["P999"],
            "prototype_support_type": "target_evidence",
            "prototype_coverage": "partial",
        })
        self.assertEqual(plan.evidence_route, "morphology_only")
        self.assertEqual(plan.selected_prototype_ids, [])
        self.assertEqual(plan.task_match, "none")

    def test_no_prototype_with_morphology_is_morphology_only(self):
        plan = self.normalize({
            "prototype_ids": [],
            "prototype_support_type": "none",
            "prototype_coverage": "none",
        })
        self.assertEqual(plan.evidence_route, "morphology_only")
        self.assertTrue(plan.use_pathology_agent)

    def test_partial_keeps_target_evidence_route(self):
        plan = self.normalize({
            "prototype_ids": ["P007"],
            "prototype_support_type": "target_evidence",
            "prototype_coverage": "partial",
        })
        self.assertEqual(plan.evidence_route, "phenotype_direct")
        self.assertEqual(plan.task_match, "partial")

    def test_complete_relevance_is_clamped_up(self):
        plan = self.normalize({
            "prototype_ids": ["P006"],
            "prototype_support_type": "target_evidence",
            "prototype_coverage": "complete",
            "phenotype_relevance_score": 0.2,
        })
        self.assertEqual(plan.phenotype_relevance_score, 0.85)

    def test_partial_relevance_is_clamped_down(self):
        plan = self.normalize({
            "prototype_ids": ["P007"],
            "prototype_support_type": "target_evidence",
            "prototype_coverage": "partial",
            "phenotype_relevance_score": 1.0,
        })
        self.assertEqual(plan.phenotype_relevance_score, 0.85)

    def test_correlated_partial_cannot_use_phenotype(self):
        plan = self.normalize({
            "prototype_ids": ["P013"],
            "prototype_support_type": "correlated_context",
            "prototype_coverage": "partial",
        })
        self.assertEqual(plan.selected_prototype_ids, [])
        self.assertEqual(plan.prototype_coverage, "none")
        self.assertEqual(plan.evidence_route, "morphology_only")

    def test_correlated_context_with_morphology_uses_morphology(self):
        plan = self.normalize({
            "prototype_ids": ["P001"],
            "prototype_support_type": "correlated_context",
            "prototype_coverage": "complete",
        })
        self.assertEqual(plan.evidence_route, "morphology_only")
        self.assertEqual(plan.prototype_coverage, "none")
        self.assertEqual(plan.selected_prototype_ids, [])

    def test_none_support_clears_llm_prototype_ids(self):
        plan = self.normalize({
            "prototype_ids": ["P001"],
            "prototype_support_type": "none",
            "prototype_coverage": "complete",
        })
        self.assertEqual(plan.selected_prototype_ids, [])
        self.assertEqual(plan.evidence_route, "morphology_only")

    def test_invalid_support_type_is_normalized_to_none(self):
        plan = self.normalize({
            "prototype_ids": ["P006"],
            "prototype_support_type": "possibly_related",
            "prototype_coverage": "complete",
        })
        self.assertEqual(plan.prototype_support_type, "none")
        self.assertEqual(plan.prototype_coverage, "none")
        self.assertEqual(plan.evidence_route, "morphology_only")

    def test_normalized_plans_only_use_two_runtime_routes(self):
        cases = [
            {
                "prototype_ids": ["P006"],
                "prototype_support_type": "target_evidence",
                "prototype_coverage": "complete",
            },
            {
                "prototype_ids": ["P007"],
                "prototype_support_type": "target_evidence",
                "prototype_coverage": "partial",
            },
            {
                "prototype_ids": [],
                "prototype_support_type": "none",
                "prototype_coverage": "none",
            },
        ]
        routes = {self.normalize(parsed).evidence_route for parsed in cases}
        self.assertEqual(routes, {"phenotype_direct", "morphology_only"})

    def test_rule_fallback_uses_morphology_only(self):
        plan = self.planner._rule_plan(
            "TCGA-XX-0001", "What treatment was documented?", ["A", "B"]
        )
        self.assertEqual(plan.evidence_route, "morphology_only")
        self.assertEqual(plan.task_type, "morphology")
        self.assertTrue(plan.use_pathology_agent)

    def test_label_space_keeps_categorical_er_but_downgrades_percentage(self):
        class SemanticRegistry(FakeRegistry):
            @staticmethod
            def label_semantics(field):
                assert field == "ER_status_label"
                return {"clinical_meaning": {"0": "negative", "1": "positive"}}

        planner = PrototypeAwarePlanner(SemanticRegistry(), FakeClient())
        parsed = {
            "prototype_ids": ["P013"],
            "prototype_support_type": "target_evidence",
            "prototype_coverage": "complete",
        }
        categorical = planner._normalize_llm_plan(
            "case", "Is ER positive or negative?", ["positive", "negative"], parsed
        )
        quantitative = planner._normalize_llm_plan(
            "case", "What percentage of cells are ER positive?",
            ["10%", "50%", "90%"], parsed,
        )
        self.assertEqual(categorical.prototype_coverage, "complete")
        self.assertEqual(quantitative.prototype_coverage, "partial")


if __name__ == "__main__":
    unittest.main()
