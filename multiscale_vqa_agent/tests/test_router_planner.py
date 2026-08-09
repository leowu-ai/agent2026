import unittest

from multiscale_vqa_agent.registry import PrototypeAwarePlanner


class FakeRegistry:
    prototype_id_to_field = {
        "P001": "histological_type_label",
        "P006": "histologic_grade_label",
        "P007": "lymphovascular_invasion_label",
    }
    field_to_name = {
        "histological_type_label": "histological type",
        "histologic_grade_label": "histologic grade",
        "lymphovascular_invasion_label": "lymphovascular invasion",
    }
    vocabs = {
        1024: {
            "phenotype_task_types": {
                "histological type": "multiclass",
                "histologic grade": "multiclass",
                "lymphovascular invasion": "multiclass",
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

    def test_direct_keeps_grade_prototype(self):
        plan = self.normalize({
            "route": "phenotype_direct",
            "prototype_ids": ["P006"],
            "task_match": "direct",
        })
        self.assertEqual(plan.target_phenotypes, ["histologic_grade_label"])
        self.assertEqual(plan.selected_prototype_ids, ["P006"])
        self.assertEqual(plan.task_match, "direct")

    def test_partial_keeps_available_prototypes(self):
        plan = self.normalize({
            "route": "phenotype_direct",
            "prototype_ids": ["P001", "P007"],
            "task_match": "partial",
            "phenotype_relevance_score": 0.7,
        })
        self.assertEqual(
            plan.target_phenotypes,
            ["histological_type_label", "lymphovascular_invasion_label"],
        )
        self.assertEqual(plan.task_match, "partial")
        self.assertTrue(plan.supported)

    def test_morphology_has_no_prototypes_and_uses_pathology(self):
        plan = self.normalize({
            "route": "morphology_only",
            "prototype_ids": ["P001"],
            "task_match": "direct",
            "use_pathology_agent": False,
        })
        self.assertEqual(plan.selected_prototype_ids, [])
        self.assertEqual(plan.target_phenotypes, [])
        self.assertEqual(plan.task_match, "none")
        self.assertTrue(plan.use_pathology_agent)

    def test_nonvisual_disables_pathology(self):
        plan = self.normalize({
            "route": "nonvisual",
            "prototype_ids": [],
            "task_match": "none",
            "use_pathology_agent": True,
        })
        self.assertEqual(plan.selected_prototype_ids, [])
        self.assertFalse(plan.use_pathology_agent)
        self.assertFalse(plan.supported)

    def test_invalid_prototype_is_safely_discarded(self):
        plan = self.normalize({
            "route": "phenotype_direct",
            "prototype_ids": ["P999"],
            "task_match": "direct",
        })
        self.assertEqual(plan.evidence_route, "morphology_only")
        self.assertEqual(plan.selected_prototype_ids, [])
        self.assertEqual(plan.task_match, "none")


if __name__ == "__main__":
    unittest.main()
