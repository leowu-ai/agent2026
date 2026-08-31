import unittest

from multiscale_vqa_agent.mc_pipeline import filter_multiple_choice_items


class MultipleChoiceTaskFilterTest(unittest.TestCase):
    def test_exact_normalized_task_filter(self):
        items = [
            {"Task": "Tumor", "Choice": ["A", "B"]},
            {"Task": " tumor ", "Choice": ["A", "B"]},
            {"Task": "HER2", "Choice": ["A", "B"]},
            {"Task": "Tumor", "Choice": []},
        ]

        selected = filter_multiple_choice_items(items, "TUMOR")

        self.assertEqual(len(selected), 2)
        self.assertTrue(all(item["Task"].strip().lower() == "tumor" for item in selected))

    def test_missing_task_fails_fast(self):
        with self.assertRaisesRegex(ValueError, "No multiple-choice items matched"):
            filter_multiple_choice_items(
                [{"Task": "Tumor", "Choice": ["A", "B"]}], "Unknown"
            )


if __name__ == "__main__":
    unittest.main()
