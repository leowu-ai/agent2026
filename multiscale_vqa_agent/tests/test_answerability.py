import json
import tempfile
import unittest
from pathlib import Path

from multiscale_vqa_agent.answerability import (
    ANSWERABILITY_SYSTEM_PROMPT,
    AnswerabilityAgent,
)
from multiscale_vqa_agent.answerability_evaluation import evaluate_answerability
from multiscale_vqa_agent.live_metrics import LiveAccuracyTracker
from multiscale_vqa_agent.mc_pipeline import MultipleChoiceVQAPipeline
from multiscale_vqa_agent.pipeline import MultiScaleVQAPipeline
from multiscale_vqa_agent.precomputed_answerability import (
    PrecomputedAnswerabilityStore,
)
from multiscale_vqa_agent.schemas import ExecutionPlan


PROJECT_ROOT = Path(__file__).resolve().parents[2]
GOLD_PATH = PROJECT_ROOT / "dataset" / "WsiVQA_answerability_binary_flat_v1.json"


class CountingGate:
    def __init__(self, labels):
        self.labels = labels

    def predict(self, question, choices):
        return {
            "can_answer": self.labels[question],
            "confidence": 0.9,
            "reason": "test decision",
            "fallback_used": False,
        }


class FailingGate:
    def predict(self, question, choices):
        raise AssertionError("Online AnswerabilityAgent must not be called")


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

    def __init__(self, response=None):
        self.response = response
        self.system = None
        self.user = None

    def chat(self, system, user, **kwargs):
        self.system = system
        self.user = user
        return self.response


class AnswerabilityPipelineTest(unittest.TestCase):
    def make_pipeline(self, labels, pipeline_class=CountingPipeline):
        pipeline = pipeline_class.__new__(pipeline_class)
        pipeline.config = {"output_dir": "."}
        pipeline.planner_only = False
        pipeline.answerability_only = False
        pipeline.calls = {
            "planner": 0,
            "g2p": 0,
            "retrieval": 0,
            "pathology": 0,
            "fusion": 0,
        }
        pipeline.answerability = FailingGate()
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

    def test_false_label_does_not_skip_planner(self):
        pipeline, _ = self.run_items({"q": False}, [self.item("q")])
        self.assertEqual(pipeline.calls["planner"], 1)

    def test_false_label_does_not_skip_g2p(self):
        pipeline, _ = self.run_items({"q": False}, [self.item("q")])
        self.assertEqual(pipeline.calls["g2p"], 1)

    def test_false_label_does_not_skip_retrieval(self):
        pipeline, _ = self.run_items({"q": False}, [self.item("q")])
        self.assertEqual(pipeline.calls["retrieval"], 1)

    def test_false_label_does_not_skip_pathology(self):
        pipeline, _ = self.run_items({"q": False}, [self.item("q")])
        self.assertEqual(pipeline.calls["pathology"], 1)

    def test_false_label_does_not_skip_fusion_or_abstain(self):
        pipeline, rows = self.run_items({"q": False}, [self.item("q")])
        self.assertEqual(pipeline.calls["fusion"], 1)
        self.assertFalse(rows[0].get("abstained", False))
        self.assertNotIn("predicted_can_answer", rows[0])
        self.assertIsNotNone(rows[0]["agent_answer"])

    def test_true_enters_old_pipeline(self):
        pipeline, rows = self.run_items({"q": True}, [self.item("q")])
        self.assertEqual(pipeline.calls, {
            "planner": 1, "g2p": 1, "retrieval": 1, "pathology": 1, "fusion": 1,
        })
        self.assertNotIn("predicted_can_answer", rows[0])
        self.assertFalse(rows[0].get("abstained", False))

    def test_all_false_case_still_runs_infer_case(self):
        pipeline, _ = self.run_items(
            {"q1": False, "q2": False}, [self.item("q1"), self.item("q2")]
        )
        self.assertEqual(pipeline.calls["g2p"], 1)

    def test_mixed_case_runs_all_questions(self):
        labels = {"q1": True, "q2": True, "q3": False, "q4": False}
        pipeline, rows = self.run_items(labels, [self.item(q) for q in labels])
        self.assertEqual(pipeline.calls["g2p"], 1)
        self.assertEqual(pipeline.calls["planner"], 4)
        self.assertEqual(pipeline.calls["fusion"], 4)
        self.assertEqual(sum(row.get("abstained", False) for row in rows), 0)

    def test_precomputed_gate_does_not_control_normal_inference(self):
        items = [self.item("q true"), self.item("q false")]
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            source = directory / "questions.json"
            output = directory / "answers.jsonl"
            frozen = directory / "gate.jsonl"
            source.write_text(json.dumps(items), encoding="utf-8")
            frozen.write_text("".join(
                json.dumps({
                    "case_id": item["Id"],
                    "question": item["Question"],
                    "predicted_can_answer": decision,
                    "answerability_confidence": 0.8,
                    "answerability_reason": "frozen",
                    "answerability_fallback_used": False,
                }) + "\n"
                for item, decision in zip(items, (True, False))
            ), encoding="utf-8")
            pipeline = self.make_pipeline({}, CountingMCPipeline)
            pipeline.answerability = FailingGate()
            pipeline.precomputed_answerability = PrecomputedAnswerabilityStore(
                str(frozen)
            )
            pipeline.run_multiple_choice(
                str(source), str(output), resume=False, crop_patches=False
            )
            rows = [json.loads(line) for line in output.read_text().splitlines()]
        self.assertEqual(pipeline.calls["planner"], 2)
        self.assertEqual(pipeline.calls["g2p"], 1)
        self.assertEqual(pipeline.calls["fusion"], 2)
        self.assertEqual(sum(row.get("abstained", False) for row in rows), 0)

    def test_missing_precomputed_key_is_fatal_without_online_fallback(self):
        item = self.item("missing question")
        with tempfile.TemporaryDirectory() as directory:
            frozen = Path(directory) / "gate.jsonl"
            frozen.write_text(json.dumps({
                "case_id": item["Id"],
                "question": "another question",
                "predicted_can_answer": True,
            }) + "\n", encoding="utf-8")
            store = PrecomputedAnswerabilityStore(str(frozen))
            with self.assertRaisesRegex(ValueError, "missing=1"):
                store.validate_items([item])

    def test_duplicate_precomputed_key_is_fatal(self):
        row = {
            "case_id": "TCGA-AA-0001",
            "question": "same question",
            "predicted_can_answer": True,
        }
        with tempfile.TemporaryDirectory() as directory:
            frozen = Path(directory) / "gate.jsonl"
            frozen.write_text(
                json.dumps(row) + "\n" + json.dumps(row) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "Duplicate precomputed"):
                PrecomputedAnswerabilityStore(str(frozen))

    def test_precomputed_lookup_normalizes_case_and_whitespace_only(self):
        with tempfile.TemporaryDirectory() as directory:
            frozen = Path(directory) / "gate.jsonl"
            frozen.write_text(json.dumps({
                "case_id": "TCGA-AA-0001",
                "question": "What   is the grade?",
                "predicted_can_answer": True,
                "answerability_confidence": 0.75,
                "answerability_reason": "frozen",
                "answerability_fallback_used": True,
            }) + "\n", encoding="utf-8")
            store = PrecomputedAnswerabilityStore(str(frozen))
            result = store.lookup("tcga-aa-0001", "  what is THE grade? ")
        self.assertTrue(result["can_answer"])
        self.assertEqual(result["confidence"], 0.75)
        self.assertTrue(result["fallback_used"])

    def test_prompt_contains_only_question_choices_and_schema(self):
        client = CapturingClient(json.dumps({
            "can_answer": True,
            "confidence": 0.8,
            "reason": "WSI morphology supports this target.",
        }))
        AnswerabilityAgent(client).predict("What is the grade?", ["low", "high"])
        packet = json.loads(client.user)
        self.assertEqual(set(packet), {"question", "choices", "output_schema"})
        combined = f"{client.system}\n{client.user}".lower()
        for forbidden in ("gold", "reference answer", "reason_code", "label_source"):
            self.assertNotIn(forbidden, combined)

    def test_prompt_defines_target_granularity_contract(self):
        prompt = ANSWERABILITY_SYSTEM_PROMPT
        self.assertIn("Classify the information TARGET", prompt)
        self.assertIn(
            'Words such as "test", "testing", "staining", '
            '"immunohistochemistry", "IHC"',
            prompt,
        )
        self.assertIn("do NOT by themselves make", prompt)
        self.assertIn(
            'B. "Was HER2 positive or negative by immunohistochemistry?"',
            prompt,
        )
        self.assertIn(
            'E. "Was HER2 gene amplification detected by FISH?"',
            prompt,
        )
        self.assertIn('F. "What is the Nottingham grade?"', prompt)
        self.assertIn(
            'G. "Was the Nottingham score determined in the report?"',
            prompt,
        )

    def test_fallback_is_true_with_zero_confidence(self):
        result = AnswerabilityAgent(CapturingClient("not json")).predict("q", ["A"])
        self.assertTrue(result["can_answer"])
        self.assertEqual(result["confidence"], 0.0)
        self.assertTrue(result["fallback_used"])

    def test_string_boolean_is_rejected(self):
        result = AnswerabilityAgent(CapturingClient(json.dumps({
            "can_answer": "false", "confidence": 1.0, "reason": "invalid"
        }))).predict("q", ["A"])
        self.assertTrue(result["can_answer"])
        self.assertTrue(result["fallback_used"])

    def test_answerability_only_calls_no_downstream_component(self):
        pipeline = CountingMCPipeline.__new__(CountingMCPipeline)
        pipeline.config = {"output_dir": "."}
        pipeline.planner_only = False
        pipeline.answerability_only = True
        pipeline.answerability = CountingGate({"q1": True, "q2": False})
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
        self.assertEqual(len(rows), 2)
        self.assertFalse(hasattr(pipeline, "planner"))
        self.assertFalse(hasattr(pipeline, "g2p"))
        self.assertTrue(all(row["answerability_only"] for row in rows))

    def test_abstention_is_not_counted_as_router_none(self):
        with tempfile.TemporaryDirectory() as directory:
            tracker = LiveAccuracyTracker(
                Path(directory) / "metrics.json",
                Path(directory) / "history.csv",
                selected_total=1,
            )
            tracker.update({"case_id": "C1", "question": "q", "plan": {},
                            "abstained": True, "agent_answer": None})
            snapshot = tracker.snapshot()
        self.assertEqual(snapshot["abstained"], 1)
        self.assertEqual(snapshot["answered"], 0)
        self.assertEqual(snapshot["per_task_match"], {})
        self.assertEqual(snapshot["per_task"], {})


class BinaryEvaluatorTest(unittest.TestCase):
    def setUp(self):
        self.gold_payload = json.loads(GOLD_PATH.read_text(encoding="utf-8"))

    def predictions(self):
        rows = []
        for gold in self.gold_payload["labels"]:
            predicted = (
                gold["can_answer"]
                if isinstance(gold["can_answer"], bool)
                else True
            )
            rows.append({
                "case_id": gold["Id"],
                "question": gold["Question"],
                "predicted_can_answer": predicted,
                "answerability_confidence": 0.9,
                "answerability_fallback_used": False,
                "abstained": not predicted,
                "reference_answer": "yes",
                "agent_answer": {"answer": "yes"} if predicted else None,
            })
        return rows

    def evaluate(self, rows, gold_payload=None):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        directory = Path(temporary.name)
        answers = directory / "answers.jsonl"
        labels = directory / "labels.json"
        answers.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        labels.write_text(json.dumps(gold_payload or self.gold_payload), encoding="utf-8")
        return evaluate_answerability(answers, labels)

    def test_gold_counts_are_strictly_390_382_186_196_8(self):
        summary = self.evaluate(self.predictions())
        self.assertEqual(summary["dataset"], {
            "gold_total": 390,
            "gold_valid": 382,
            "gold_excluded": 8,
            "gold_can_answer": 186,
            "gold_cannot_answer": 196,
        })

    def test_gold_answerable_false_abstention_is_primary_error(self):
        rows = self.predictions()
        valid_true_keys = {
            (row["Id"], row["Question"])
            for row in self.gold_payload["labels"]
            if not row["exclude_from_evaluation"] and row["can_answer"]
        }
        target = next(
            row for row in rows
            if (row["case_id"], row["question"]) in valid_true_keys
        )
        target["predicted_can_answer"] = False
        target["abstained"] = True
        target["agent_answer"] = None
        summary = self.evaluate(rows)
        self.assertEqual(summary["vqa"]["gold_answerable_n"], 186)
        self.assertEqual(summary["vqa"]["gold_answerable_correct"], 185)

    def test_excluded_rows_do_not_enter_any_metric(self):
        summary = self.evaluate(self.predictions())
        self.assertEqual(summary["answerability"]["tp"], 186)
        self.assertEqual(summary["answerability"]["tn"], 196)
        self.assertEqual(summary["vqa"]["answered_n"], 186)

    def test_missing_prediction_raises_instead_of_defaulting_false(self):
        rows = self.predictions()
        valid_keys = {
            (row["Id"], row["Question"])
            for row in self.gold_payload["labels"]
            if not row["exclude_from_evaluation"]
        }
        rows = [row for row in rows if (row["case_id"], row["question"]) != next(iter(valid_keys))]
        with self.assertRaisesRegex(ValueError, "Missing 1"):
            self.evaluate(rows)

    def test_duplicate_prediction_is_detected(self):
        rows = self.predictions()
        rows.append(dict(rows[0]))
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.evaluate(rows)

    def test_gold_must_have_exactly_390_rows(self):
        malformed = dict(self.gold_payload)
        malformed["labels"] = self.gold_payload["labels"][:-1]
        with self.assertRaisesRegex(ValueError, "Expected 390"):
            self.evaluate(self.predictions(), malformed)

    def test_legacy_three_class_predictions_are_convertible(self):
        rows = self.predictions()
        for row in rows:
            value = row.pop("predicted_can_answer")
            row["predicted_answerability"] = (
                "directly_answerable" if value else "unanswerable"
            )
        summary = self.evaluate(rows)
        self.assertEqual(summary["integrity"]["legacy_conversions"], 390)
        self.assertEqual(summary["answerability"]["accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
