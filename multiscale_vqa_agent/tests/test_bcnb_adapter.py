import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

from multiscale_vqa_agent.prepare_bcnb import convert_vqa_csv, write_manifests
from multiscale_vqa_agent.retrieval import WSICropper


class BCNBAdapterTest(unittest.TestCase):
    def test_csv_conversion_uses_exact_choice_text(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input.csv"
            with source.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=(
                        "ID", "Slide", "Task", "Question", "A", "B", "C", "D", "Answer"
                    ),
                )
                writer.writeheader()
                writer.writerow({
                    "ID": "7", "Slide": "1", "Task": "ER", "Question": "ER status?",
                    "A": "Positive", "B": "Negative", "C": "", "D": "", "Answer": "B",
                })
            rows = convert_vqa_csv(source, root / "output.json")
            self.assertEqual(rows[0]["Id"], "1")
            self.assertEqual(rows[0]["Choice"], ["Positive", "Negative"])
            self.assertEqual(rows[0]["Answer"], "Negative")
            saved = json.loads((root / "output.json").read_text())
            self.assertEqual(saved[0]["AnswerId"], "B")

    def test_manifest_accepts_bcnb_object_npy(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for scale in (1024, 2048, 4096):
                scale_dir = root / "features" / str(scale)
                scale_dir.mkdir(parents=True)
                np.save(scale_dir / f"1_0_{scale}.npy", {
                    "feature": np.ones((2, 768), dtype=np.float32),
                    "index": [f"0_0_{scale}.png", f"{scale}_0_{scale}.png"],
                    "inst_label": [-1, -1],
                })
            manifests = write_manifests(
                [{"Id": "1"}], root / "features", root / "manifests"
            )
            for path in manifests.values():
                with open(path, encoding="utf-8") as handle:
                    row = next(csv.DictReader(handle))
                self.assertEqual(row["case_id"], "1")
                self.assertEqual(row["split"], "external")

    def test_jpeg_index_uses_numeric_stem(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new("RGB", (32, 32)).save(root / "17.jpg")
            cropper = WSICropper(root, root / "output")
            self.assertEqual(
                cropper._resolve_wsi("17", "17_0_1024"), root / "17.jpg"
            )

    def test_jpeg_crop_trims_mcu_expansion(self):
        expanded = Image.new("RGB", (1038, 1030), "white")
        with tempfile.NamedTemporaryFile(suffix=".jpg") as source:
            expanded.save(source.name)

            def fake_run(command, stdout, **kwargs):
                with open(source.name, "rb") as handle:
                    stdout.write(handle.read())
                return mock.Mock(returncode=0, stderr=b"")

            with mock.patch("subprocess.run", side_effect=fake_run), mock.patch.object(
                WSICropper, "_jpegtran_path", return_value="jpegtran"
            ):
                result = WSICropper._crop_jpeg(
                    Path("slide.jpg"), 14, 6, 1024
                )
            self.assertEqual(result.size, (1024, 1024))


if __name__ == "__main__":
    unittest.main()
