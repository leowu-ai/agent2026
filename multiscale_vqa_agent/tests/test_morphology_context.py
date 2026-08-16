import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from multiscale_vqa_agent.fusion import FusionVerificationAgent
from multiscale_vqa_agent.pathology import PathologyAgent
from multiscale_vqa_agent.retrieval import WSICropper
from multiscale_vqa_agent.schemas import EvidenceGroup, ExecutionPlan, PatchCandidate


class SequencedClient:
    enabled = True

    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def chat(self, system, user, **kwargs):
        self.calls.append({"system": system, "user": user, **kwargs})
        return next(self.responses)


def morphology_plan():
    return ExecutionPlan(
        case_id="TCGA-XX-0001",
        question="Which pattern is present?",
        target_phenotypes=[],
        task_type="morphology",
        metrics=[],
        answer_mode="multiple_choice",
        supported=False,
        support_reason="No direct prototype.",
        task_match="none",
        evidence_route="morphology_only",
        selected_prototype_ids=[],
    )


class MorphologyFusionTest(unittest.TestCase):
    def test_broad_context_reaches_initial_and_repair_prompts(self):
        client = SequencedClient([
            "not valid json",
            json.dumps({
                "answer_id": "B",
                "confidence": 0.3,
                "explanation": "The combined context favors B.",
                "limitations": "The evidence is indirect.",
            }),
        ])
        broad = [{
            "prototype_id": "P001",
            "field": "histological_type_label",
            "predicted_label": "ductal",
            "confidence": 0.8,
        }]
        answer, structured = FusionVerificationAgent(client).answer_with_summary(
            morphology_plan(),
            ["lobular", "ductal"],
            [],
            {},
            {"description": "Cohesive malignant glands are visible."},
            broad_g2p_predictions=broad,
        )

        self.assertEqual(answer["answer_id"], "B")
        self.assertEqual(structured["task_match"], "none")
        self.assertIsNone(structured.get("structured_candidate_answer"))
        self.assertEqual(len(client.calls), 2)
        self.assertEqual(client.calls[0]["temperature"], 0.6)
        self.assertEqual(client.calls[0]["max_tokens"], 4096)
        self.assertIs(client.calls[0]["enable_thinking"], True)
        self.assertEqual(client.calls[0]["top_p"], 0.95)
        self.assertEqual(client.calls[0]["top_k"], 20)
        self.assertEqual(client.calls[1]["temperature"], 0.0)
        self.assertIs(client.calls[1]["enable_thinking"], False)
        initial = json.loads(client.calls[0]["user"])
        repair = json.loads(client.calls[1]["user"])
        self.assertEqual(initial["broad_g2p_predictions"], broad)
        self.assertEqual(repair["broad_g2p_predictions"], broad)
        self.assertIn("Cohesive malignant glands", initial["available_visual_summary"])
        self.assertIn("Cohesive malignant glands", repair["visual_evidence_summary"])

    def test_direct_packet_does_not_use_broad_context(self):
        agent = FusionVerificationAgent(SequencedClient([]))
        plan = morphology_plan()
        plan.task_match = "direct"
        plan.evidence_route = "phenotype_direct"
        structured = {
            "task_match": "direct",
            "evidence_route": "phenotype_direct",
            "predictions": [],
        }
        packet = agent._build_evidence_packet(
            plan,
            ["yes", "no"],
            structured,
            {},
            {"description": "Visible morphology."},
            broad_g2p_predictions=[{"field": "unused"}],
        )
        self.assertNotIn("broad_g2p_predictions", packet)


class PathologyImageSelectionTest(unittest.TestCase):
    @staticmethod
    def patch(scale, group_id):
        return PatchCandidate(
            scale=scale,
            slide_id=f"slide_{group_id}",
            patch_index=scale,
            x=0,
            y=0,
            size=scale,
            score=1.0,
            image_path=f"/tmp/group{group_id}_{scale}.jpg",
        )

    def test_overviews_then_coarse_group_coverage_with_eight_image_cap(self):
        groups = [
            EvidenceGroup(
                group_id=group_id,
                score=1.0,
                patches={
                    4096: self.patch(4096, group_id),
                    1024: self.patch(1024, group_id),
                },
            )
            for group_id in range(1, 5)
        ]
        entries = PathologyAgent._image_entries(
            groups, ["/tmp/overview1.jpg", "/tmp/overview2.jpg"]
        )
        selected = PathologyAgent._select_entries(entries, 8)

        self.assertEqual(len(selected), 8)
        self.assertEqual([row["kind"] for row in selected[:2]], ["overview", "overview"])
        self.assertEqual(
            [(row["group_id"], row["scale"]) for row in selected[2:6]],
            [(1, 4096), (2, 4096), (3, 4096), (4, 4096)],
        )

    def test_direct_patches_restore_scale_then_group_order(self):
        entries = [
            {
                "kind": "patch",
                "group_id": 1,
                "scale": 1024,
                "image_path": "/tmp/group1_1024.jpg",
            },
            {
                "kind": "patch",
                "group_id": 1,
                "scale": 4096,
                "image_path": "/tmp/group1_4096.jpg",
            },
            {
                "kind": "patch",
                "group_id": 2,
                "scale": 2048,
                "image_path": "/tmp/group2_2048.jpg",
            },
        ]

        selected = PathologyAgent._select_entries(entries, 8)

        self.assertEqual(
            [(row["scale"], row["group_id"]) for row in selected],
            [(4096, 1), (2048, 2), (1024, 1)],
        )


class OverviewThumbnailTest(unittest.TestCase):
    def test_dx_then_ts_priority_and_cached_reuse(self):
        opened = []

        class FakeImage:
            def convert(self, mode):
                return self

            def save(self, path, *args, **kwargs):
                Path(path).write_bytes(b"jpeg")

        class FakeSlide:
            def __init__(self, path):
                self.path = path
                opened.append(Path(path).name)

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def get_thumbnail(self, size):
                return FakeImage()

        fake_openslide = types.SimpleNamespace(OpenSlide=FakeSlide)
        with tempfile.TemporaryDirectory() as root, tempfile.TemporaryDirectory() as output:
            case_id = "TCGA-XX-0001"
            for name in (
                f"{case_id}-01Z-00-DX1.svs",
                f"{case_id}-01Z-00-TS1.svs",
                f"{case_id}-01Z-00-AB1.svs",
            ):
                (Path(root) / name).touch()
            cropper = WSICropper(Path(root), Path(output))
            with mock.patch.dict(sys.modules, {"openslide": fake_openslide}):
                first = cropper.overview_thumbnails(case_id, size=256, max_slides=2)
                second = cropper.overview_thumbnails(case_id, size=256, max_slides=2)

        self.assertEqual(opened, [
            f"{case_id}-01Z-00-DX1.svs",
            f"{case_id}-01Z-00-TS1.svs",
        ])
        self.assertEqual(first, second)
        self.assertIn("-DX1", Path(first[0]).name)
        self.assertIn("-TS1", Path(first[1]).name)


if __name__ == "__main__":
    unittest.main()
