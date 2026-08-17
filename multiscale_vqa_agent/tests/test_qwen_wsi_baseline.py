import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from multiscale_vqa_agent.answerability import ANSWERABILITY_SYSTEM_PROMPT
from multiscale_vqa_agent.run_qwen_wsi_baseline import (
    WSIThumbnailIndex,
    indexed_choices,
    parse_answer,
    run,
)


class QwenWSIBaselineTest(unittest.TestCase):
    def run_answer_case(self, answer_thinking, answer_responses):
        class FakeClient:
            calls = []
            responses = iter([
                '{"can_answer":true,"confidence":0.9,"reason":"morphology"}',
                *answer_responses,
            ])

            def __init__(self, config):
                pass

            def chat(self, *args, **kwargs):
                FakeClient.calls.append({
                    "system": kwargs.get("system", args[0] if args else None),
                    "user": kwargs.get("user", args[1] if len(args) > 1 else None),
                    **kwargs,
                })
                return next(FakeClient.responses)

        class FakeThumbnails:
            def __init__(self, *args, **kwargs):
                pass

            def thumbnails(self, case_id):
                return ["thumbnail.jpg"], ["slide.svs"]

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            questions = root / "questions.json"
            output = root / "answers.jsonl"
            config = root / "config.json"
            questions.write_text(json.dumps([{
                "Id": "TCGA-AA-0001",
                "Question": "What is the diagnosis?",
                "Choice": ["ductal", "lobular"],
                "Answer": "ductal",
            }]), encoding="utf-8")
            config.write_text(json.dumps({
                "vqa_json": str(questions),
                "output_dir": str(root),
                "wsi_root": str(root),
                "qwen": {},
            }), encoding="utf-8")
            args = argparse.Namespace(
                config=str(config),
                vqa_json=None,
                output=str(output),
                metrics=None,
                thumbnail_dir=None,
                thumbnail_size=1536,
                max_slides=4,
                limit=None,
                resume=False,
                answerability_labels=None,
                answer_thinking=answer_thinking,
            )
            with patch(
                "multiscale_vqa_agent.run_qwen_wsi_baseline.OpenAICompatibleClient",
                FakeClient,
            ), patch(
                "multiscale_vqa_agent.run_qwen_wsi_baseline.WSIThumbnailIndex",
                FakeThumbnails,
            ):
                run(args)
            row = json.loads(output.read_text().strip())
        return row, FakeClient.calls

    def test_indexed_choices(self):
        self.assertEqual(
            indexed_choices(["negative", "positive"]),
            [{"id": "A", "text": "negative"}, {"id": "B", "text": "positive"}],
        )

    def test_parse_valid_answer_id(self):
        parsed = parse_answer('{"answer_id":"B","confidence":0.8}', ["no", "yes"])
        self.assertEqual(parsed["answer"], "yes")

    def test_parse_rejects_answer_text_and_invalid_id(self):
        self.assertIsNone(parse_answer('{"answer_id":"yes"}', ["no", "yes"]))
        self.assertIsNone(parse_answer('{"answer_id":"C"}', ["no", "yes"]))

    def test_answerability_prompt_has_no_evaluation_leakage(self):
        lowered = ANSWERABILITY_SYSTEM_PROMPT.lower()
        for forbidden in (
            "reference_answer", "reference answer", "gold_can_answer",
            "exclude_from_evaluation", "reason_code", "label_source",
        ):
            self.assertNotIn(forbidden, lowered)

    def test_default_answer_call_is_explicitly_no_thinking(self):
        row, calls = self.run_answer_case(
            False, ['{"answer_id":"A","confidence":0.8}']
        )
        self.assertEqual(len(calls), 2)
        gate_call, answer_call = calls
        self.assertIsNone(gate_call.get("images"))
        self.assertNotIn("enable_thinking", gate_call)
        self.assertEqual(answer_call["images"], ["thumbnail.jpg"])
        self.assertIs(answer_call["enable_thinking"], False)
        self.assertEqual(answer_call["temperature"], 0.0)
        self.assertEqual(answer_call["max_tokens"], 256)
        self.assertFalse(row["answer_thinking"])
        self.assertFalse(row["repair_used"])

    def test_thinking_answer_call_uses_requested_sampling(self):
        row, calls = self.run_answer_case(
            True, ['{"answer_id":"A","confidence":0.8}']
        )
        answer_call = calls[1]
        self.assertEqual(answer_call["images"], ["thumbnail.jpg"])
        self.assertIs(answer_call["enable_thinking"], True)
        self.assertEqual(answer_call["temperature"], 0.6)
        self.assertEqual(answer_call["top_p"], 0.95)
        self.assertEqual(answer_call["top_k"], 20)
        self.assertEqual(answer_call["max_tokens"], 4096)
        self.assertTrue(row["answer_thinking"])
        self.assertFalse(row["repair_used"])
        self.assertEqual(
            row["initial_raw_response"],
            '{"answer_id":"A","confidence":0.8}',
        )

    def test_thinking_parse_failure_uses_one_deterministic_repair(self):
        row, calls = self.run_answer_case(
            True,
            [
                "not valid json",
                '{"answer_id":"B","confidence":0.4,"explanation":"B"}',
            ],
        )
        self.assertEqual(len(calls), 3)
        repair_call = calls[2]
        self.assertEqual(repair_call["images"], ["thumbnail.jpg"])
        self.assertIs(repair_call["enable_thinking"], False)
        self.assertEqual(repair_call["temperature"], 0.0)
        self.assertEqual(repair_call["max_tokens"], 512)
        self.assertEqual(repair_call["retries"], 0)
        self.assertTrue(row["repair_used"])
        self.assertEqual(row["initial_raw_response"], "not valid json")
        self.assertEqual(row["repair_raw_response"], '{"answer_id":"B","confidence":0.4,"explanation":"B"}')
        self.assertEqual(row["agent_answer"]["answer"], "lobular")
        self.assertTrue(row["json_parse_success"])

    def test_false_gate_skips_answer_stage(self):
        class FakeClient:
            calls = 0
            first_user = None
            first_images = None

            def __init__(self, config):
                pass

            def chat(self, *args, **kwargs):
                FakeClient.calls += 1
                if FakeClient.calls == 1:
                    FakeClient.first_user = args[1]
                    FakeClient.first_images = kwargs.get("images")
                return '{"can_answer":false,"confidence":0.9,"reason":"not visible"}'

        class FakeThumbnails:
            calls = 0

            def __init__(self, *args, **kwargs):
                pass

            def thumbnails(self, case_id):
                FakeThumbnails.calls += 1
                return ["thumbnail.jpg"], ["slide.svs"]

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            questions = root / "questions.json"
            output = root / "answers.jsonl"
            config = root / "config.json"
            questions.write_text(json.dumps([{
                "Id": "TCGA-AA-0001", "Question": "What is the age?",
                "Choice": ["40", "50"], "Answer": "50",
            }]), encoding="utf-8")
            config.write_text(json.dumps({
                "vqa_json": str(questions), "output_dir": str(root),
                "wsi_root": str(root), "qwen": {},
            }), encoding="utf-8")
            args = argparse.Namespace(
                config=str(config), vqa_json=None, output=str(output), metrics=None,
                thumbnail_dir=None, thumbnail_size=512, max_slides=1, limit=None,
                resume=False, answerability_labels=None,
            )
            with patch(
                "multiscale_vqa_agent.run_qwen_wsi_baseline.OpenAICompatibleClient",
                FakeClient,
            ), patch(
                "multiscale_vqa_agent.run_qwen_wsi_baseline.WSIThumbnailIndex",
                FakeThumbnails,
            ):
                run(args)
            row = json.loads(output.read_text().strip())
        self.assertEqual(FakeClient.calls, 1)
        self.assertIsNone(FakeClient.first_images)
        self.assertEqual(
            set(json.loads(FakeClient.first_user)),
            {"question", "choices", "output_schema"},
        )
        self.assertEqual(FakeThumbnails.calls, 0)
        self.assertFalse(row["predicted_can_answer"])
        self.assertTrue(row["abstained"])
        self.assertIsNone(row["agent_answer"])

    def test_diagnostic_slide_is_prioritized(self):
        paths = [Path("TCGA-XX-0001-01A-01-TS1.foo.svs"), Path("TCGA-XX-0001-01Z-00-DX1.bar.svs")]
        self.assertIn("DX1", sorted(paths, key=WSIThumbnailIndex._slide_priority)[0].name)


if __name__ == "__main__":
    unittest.main()
