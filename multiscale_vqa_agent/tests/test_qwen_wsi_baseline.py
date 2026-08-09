import tempfile
import unittest
from pathlib import Path

from multiscale_vqa_agent.run_qwen_wsi_baseline import (
    WSIThumbnailIndex,
    indexed_choices,
    parse_answer,
)


class QwenWSIBaselineTest(unittest.TestCase):
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

    def test_diagnostic_slide_is_prioritized(self):
        paths = [Path("TCGA-XX-0001-01A-01-TS1.foo.svs"), Path("TCGA-XX-0001-01Z-00-DX1.bar.svs")]
        self.assertIn("DX1", sorted(paths, key=WSIThumbnailIndex._slide_priority)[0].name)


if __name__ == "__main__":
    unittest.main()
