import tempfile
import unittest
from pathlib import Path

from multiscale_vqa_agent.live_metrics import LiveAccuracyTracker, slidebench_category


def result(task, predicted, reference):
    return {
        "case_id": "1",
        "question": f"{task}?",
        "choices": ["yes", "no"],
        "reference_answer": reference,
        "input": {"Task": task},
        "plan": {"target_phenotypes": ["field"], "supported": True},
        "agent_answer": {
            "answer": predicted,
            "json_parse_success": True,
        },
    }


class SlideBenchLiveMetricsTest(unittest.TestCase):
    def test_official_task_grouping(self):
        self.assertEqual(slidebench_category("Tumor"), "tumor_type")
        for task in ("ER", "PR", "HER2"):
            self.assertEqual(slidebench_category(task), "receptor_status")
        self.assertEqual(
            slidebench_category("HER2 Expression"), "her2_expression"
        )
        self.assertEqual(
            slidebench_category("Histological grading"),
            "histological_grading",
        )
        self.assertEqual(
            slidebench_category("Molecular subtype"), "molecular_subtype"
        )

    def test_snapshot_and_history_report_slidebench_accuracy(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tracker = LiveAccuracyTracker(
                root / "metrics.json", root / "history.csv", selected_total=3
            )
            tracker.update(result("ER", "yes", "yes"))
            tracker.update(result("PR", "yes", "no"))
            tracker.update(result("Tumor", "no", "no"))
            snapshot = tracker.snapshot()
            self.assertEqual(snapshot["accuracy"], 2 / 3)
            self.assertEqual(
                snapshot["slidebench_categories"]["receptor_status"]["accuracy"],
                0.5,
            )
            self.assertEqual(
                snapshot["slidebench_categories"]["tumor_type"]["accuracy"],
                1.0,
            )
            header = (root / "history.csv").read_text().splitlines()[0]
            self.assertIn("receptor_status_accuracy", header)
            self.assertIn("molecular_subtype_accuracy", header)

    def test_restore_looks_up_task_for_legacy_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            answers = root / "answers.jsonl"
            row = result("ER", "yes", "yes")
            row.pop("input")
            import json
            answers.write_text(json.dumps(row) + "\n")
            tracker = LiveAccuracyTracker(
                root / "metrics.json",
                root / "history.csv",
                selected_total=1,
                existing_answers=answers,
                task_by_key={("1", "ER?"): "ER"},
            )
            category = tracker.snapshot()["slidebench_categories"][
                "receptor_status"
            ]
            self.assertEqual(category["processed"], 1)
            self.assertEqual(category["accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
