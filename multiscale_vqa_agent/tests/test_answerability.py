import json
import tempfile
import unittest
from pathlib import Path

from multiscale_vqa_agent.answerability import AnswerabilityAgent
from multiscale_vqa_agent.answerability_evaluation import evaluate_answerability
from multiscale_vqa_agent.mc_pipeline import MultipleChoiceVQAPipeline
from multiscale_vqa_agent.pipeline import MultiScaleVQAPipeline
from multiscale_vqa_agent.schemas import ExecutionPlan


class CountingGate:
    def __init__(self, labels):
        self.labels = labels

    def predict(self, question, choices):
        return {
            "answerability": self.labels[question],
            "confidence": 0.9,
            "reason": "test decision",
        }


class CountingPlanner:
    def __init__(self, calls):
        self.calls = calls

    def plan(self, item):
        self.calls["planner"] += 1
        return ExecutionPlan(
            case_id=str(item["Id"])[:12],
            question=item["Question"],
            target_phenotypes=[],
            task_type="morphology",
            metrics=[],
            answer_mode="multiple_choice",
            supported=False,
            support_reason="test",
            task_match="none",
            evidence_route="morphology_only",
        )


class CountingG2P:
    def __init__(self, calls):
        self.calls = calls

    def infer_case(self, case_id):
        self.calls["g2p"] += 1
        return {}


class CountingPipeline(MultiScaleVQAPipeline):
    def _run_question(self, item, plan, scale_results, evidence_cache, crop_patches):
        self.calls["retrieval"] += 1
        self.calls["pathology"] += 1
        self.calls["fusion"] += 1
        return {
            "case_id": plan.case_id,
            "question": plan.question,
            "choices": item["Choice"],
            "reference_answer": item["Answer"],
            "plan": plan.to_dict(),
            "agent_answer": {"answer": item["Choice"][0]},
        }


class CountingMCPipeline(MultipleChoiceVQAPipeline, CountingPipeline):
    pass


class CapturingClient:
    enabled = True

    def __init__(self):
        self.system = None
        self.user = None

    def chat(self, system, user, **kwargs):
        self.system = system
        self.user = user
        return json.dumps({
            "answerability": "directly_answerable",
            "confidence": 0.8,
            "reason": "H&E morphology can provide the target.",
        })


class AnswerabilityPipelineTest(unittest.TestCase):
    def make_pipeline(self, labels):
        pipeline = CountingPipeline.__new__(CountingPipeline)
        pipeline.config = {"output_dir": "."}
        pipeline.planner_only = False
        pipeline.calls = {
            "planner": 0,
            "g2p": 0,
            "retrieval": 0,
            "pathology": 0,
            "fusion": 0,
        }
        pipeline.answerability = CountingGate(labels)
        pipeline.planner = CountingPlanner(pipeline.calls)
        pipeline.g2p = CountingG2P(pipeline.calls)
        return pipeline

    @staticmethod
    def item(question, case_id="TCGA-AA-0001"):
        return {
            "Id": case_id,
            "Question": question,
            "Choice": ["A", "B"],
            "Answer": "A",
        }

    def run_items(self, labels, items):
        pipeline = self.make_pipeline(labels)
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "questions.json"
            output = Path(directory) / "answers.jsonl"
            source.write_text(json.dumps(items), encoding="utf-8")
            pipeline.run(str(source), str(output), resume=False, crop_patches=False)
            rows = [json.loads(line) for line in output.read_text().splitlines()]
        return pipeline, rows

    def test_unanswerable_skips_planner(self):
        pipeline, _ = self.run_items({"q": "unanswerable"}, [self.item("q")])
        self.assertEqual(pipeline.calls["planner"], 0)

    def test_unanswerable_skips_g2p(self):
        pipeline, _ = self.run_items({"q": "unanswerable"}, [self.item("q")])
        self.assertEqual(pipeline.calls["g2p"], 0)

    def test_unanswerable_skips_retrieval(self):
        pipeline, _ = self.run_items({"q": "unanswerable"}, [self.item("q")])
        self.assertEqual(pipeline.calls["retrieval"], 0)

    def test_unanswerable_skips_pathology(self):
        pipeline, _ = self.run_items({"q": "unanswerable"}, [self.item("q")])
        self.assertEqual(pipeline.calls["pathology"], 0)

    def test_unanswerable_skips_fusion(self):
        pipeline, rows = self.run_items({"q": "unanswerable"}, [self.item("q")])
        self.assertEqual(pipeline.calls["fusion"], 0)
        self.assertTrue(rows[0]["abstained"])
        self.assertIsNone(rows[0]["agent_answer"])

    def test_directly_answerable_enters_old_pipeline(self):
        pipeline, rows = self.run_items(
            {"q": "directly_answerable"}, [self.item("q")]
        )
        self.assertEqual(pipeline.calls, {
            "planner": 1, "g2p": 1, "retrieval": 1, "pathology": 1, "fusion": 1,
        })
        self.assertFalse(rows[0]["abstained"])

    def test_inferable_enters_old_pipeline(self):
        pipeline, rows = self.run_items({"q": "inferable"}, [self.item("q")])
        self.assertEqual(pipeline.calls["fusion"], 1)
        self.assertEqual(rows[0]["predicted_answerability"], "inferable")

    def test_all_unanswerable_case_never_runs_infer_case(self):
        labels = {"q1": "unanswerable", "q2": "unanswerable"}
        pipeline, _ = self.run_items(labels, [self.item("q1"), self.item("q2")])
        self.assertEqual(pipeline.calls["g2p"], 0)

    def test_mixed_case_runs_only_answerable_questions(self):
        labels = {
            "q1": "directly_answerable",
            "q2": "inferable",
            "q3": "unanswerable",
            "q4": "unanswerable",
        }
        pipeline, rows = self.run_items(labels, [self.item(q) for q in labels])
        self.assertEqual(pipeline.calls["g2p"], 1)
        self.assertEqual(pipeline.calls["planner"], 2)
        self.assertEqual(pipeline.calls["fusion"], 2)
        self.assertEqual(sum(row["abstained"] for row in rows), 2)

    def test_mc_runner_applies_the_same_case_gate(self):
        labels = {"q1": "directly_answerable", "q2": "unanswerable"}
        pipeline = self.make_pipeline(labels)
        pipeline.__class__ = CountingMCPipeline
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "questions.json"
            output = Path(directory) / "answers.jsonl"
            source.write_text(
                json.dumps([self.item("q1"), self.item("q2")]), encoding="utf-8"
            )
            pipeline.run_multiple_choice(
                str(source), str(output), resume=False, crop_patches=False
            )
            rows = [json.loads(line) for line in output.read_text().splitlines()]
        self.assertEqual(pipeline.calls["planner"], 1)
        self.assertEqual(pipeline.calls["g2p"], 1)
        self.assertEqual(pipeline.calls["fusion"], 1)
        self.assertEqual(sum(row["abstained"] for row in rows), 1)

    def test_gold_or_reference_never_enters_answerability_prompt(self):
        client = CapturingClient()
        AnswerabilityAgent(client).predict("What is the grade?", ["low", "high"])
        packet = json.loads(client.user)
        self.assertEqual(set(packet), {"question", "choices", "output_schema"})
        self.assertNotIn("reference", client.user.lower())
        self.assertNotIn("gold", client.user.lower())


class AnswerabilityEvaluationTest(unittest.TestCase):
    def evaluate_fixture(self):
        predictions = [
            {
                "case_id": "C1",
                "question": "answerable abstain",
                "predicted_answerability": "unanswerable",
                "answerability_confidence": 0.9,
                "abstained": True,
                "agent_answer": None,
            },
            {
                "case_id": "C2",
                "question": "dataset error",
                "predicted_answerability": "directly_answerable",
                "abstained": False,
                "agent_answer": {"answer": "yes"},
            },
        ]
        gold = {
            "exact_annotations": [
                {
                    "Id": "C1", "Question": "answerable abstain", "Answer": "yes",
                    "answerability": "directly_answerable",
                },
                {
                    "Id": "C2", "Question": "dataset error", "Answer": "yes",
                    "answerability": "dataset_error",
                },
            ]
        }
        temporary = tempfile.TemporaryDirectory()
        directory = Path(temporary.name)
        answers_path = directory / "answers.jsonl"
        labels_path = directory / "labels.json"
        answers_path.write_text(
            "".join(json.dumps(row) + "\n" for row in predictions), encoding="utf-8"
        )
        labels_path.write_text(json.dumps(gold), encoding="utf-8")
        summary = evaluate_answerability(answers_path, labels_path)
        return temporary, summary

    def test_gold_answerable_abstention_counts_as_primary_error(self):
        temporary, summary = self.evaluate_fixture()
        self.addCleanup(temporary.cleanup)
        self.assertEqual(summary["vqa"]["gold_predictively_answerable_n"], 1)
        self.assertEqual(summary["vqa"]["gold_predictively_answerable_correct"], 0)
        self.assertEqual(summary["vqa"]["gold_predictively_answerable_accuracy"], 0.0)

    def test_dataset_error_is_excluded_from_every_metric(self):
        temporary, summary = self.evaluate_fixture()
        self.addCleanup(temporary.cleanup)
        self.assertEqual(summary["answerability"]["n_valid"], 1)
        self.assertEqual(summary["vqa"]["coverage"], 0.0)


if __name__ == "__main__":
    unittest.main()
