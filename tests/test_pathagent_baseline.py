import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from pathagent.pathagent_baseline import (
    CaseFeatures,
    PatchRecord,
    PathAgentBaseline,
    PathAgentExecutor,
    PathAgentPerceptor,
    PathAgentRetriever,
    SlideFeatures,
)
from pathagent.run_pathagent import (
    load_completed,
    load_gold_answerable_keys,
    run,
)


def slide(scale, slide_id, vectors, coords):
    return SlideFeatures(
        scale=scale,
        slide_id=f"{slide_id}_0_{scale}",
        slide_key=slide_id,
        feature_path=f"{slide_id}.npy",
        features=np.asarray(vectors, dtype=np.float32),
        coords=list(coords),
    )


def synthetic_case():
    return CaseFeatures("TCGA-AA-0001", {
        4096: [
            slide(4096, "slide-a", [[1, 0], [0.8, 0.2]], [(0, 0, 4096), (5000, 0, 4096)]),
            slide(4096, "slide-b", [[0.9, 0.1], [0, 1]], [(0, 0, 4096), (5000, 0, 4096)]),
        ],
        2048: [
            slide(2048, "slide-a", [[0.6, 0.8], [1, 0]], [(100, 100, 2048), (5200, 100, 2048)]),
            slide(2048, "slide-b", [[1, 0]], [(100, 100, 2048)]),
        ],
        1024: [
            slide(1024, "slide-a", [[0.2, 0.98], [1, 0], [0.7, 0.7]], [(100, 100, 1024), (600, 600, 1024), (9000, 9000, 1024)]),
            slide(1024, "slide-b", [[1, 0]], [(100, 100, 1024)]),
        ],
    })


class FakeEncoder:
    def __init__(self):
        self.calls = []

    def encode(self, text):
        self.calls.append(text)
        return np.asarray([1.0, 0.0] if "original" in text or "question" in text else [0.0, 1.0])


class FakeCropper:
    def __init__(self):
        self.patches = []

    def crop(self, case_id, question, patch):
        self.patches.append(patch)
        patch.image_path = f"/{patch.scale}_{patch.patch_index}.jpg"
        return patch.image_path


class FakePerceptor:
    def __init__(self):
        self.calls = []

    def describe(self, question, choices, patches, scale, retrieval_query):
        self.calls.append((scale, list(patches), retrieval_query))
        return [{
            "patch": patch.to_dict(),
            "pathology_features": "visible morphology",
            "relevance_to_question": "relevant",
            "answer_hint": choices[0],
        } for patch in patches]


class FakeExecutor:
    def __init__(self, sufficient=None, zoom=None):
        self.sufficient = list(sufficient or ["Yes"])
        self.zoom = list(zoom or [])
        self.prompts = []
        self.final_calls = 0

    def preliminary(self, question, choices, evidence):
        self.prompts.append(("preliminary", question, choices, evidence))
        return {"answer": choices[0], "reasoning": "visual"}

    def sufficiency(self, question, choices, evidence, preliminary):
        self.prompts.append(("sufficiency", question, choices, evidence, preliminary))
        return {"sufficient": self.sufficient.pop(0), "reason": "mock"}

    def evidence_plan(self, question, choices, evidence):
        self.prompts.append(("plan", question, choices, evidence))
        use_zoom = self.zoom.pop(0) if self.zoom else False
        return {
            "missing_info": "missing morphology",
            "zoom_recommendation": "Yes" if use_zoom else "No",
            "recommended_scale": 1024 if use_zoom else None,
            "zoom_reason": "mock",
        }

    def final(self, question, choices, evidence):
        self.prompts.append(("final", question, choices, evidence))
        self.final_calls += 1
        return {"answer": choices[0], "explanation": "visual"}


def make_baseline(executor, encoder=None, initial=20, replenish=5, attempts=5):
    encoder = encoder or FakeEncoder()
    return PathAgentBaseline(
        retriever=PathAgentRetriever(encoder),
        cropper=FakeCropper(),
        perceptor=FakePerceptor(),
        executor=executor,
        initial_patches=initial,
        replenish_patches=replenish,
        max_attempts=attempts,
        zoom_parent_topk=2,
        max_zoom_actions=1,
    )


class RetrievalTest(unittest.TestCase):
    def setUp(self):
        self.encoder = FakeEncoder()
        self.retriever = PathAgentRetriever(self.encoder)
        self.case = synthetic_case()

    def test_initial_budget_and_scale_are_fixed(self):
        rows = self.retriever.retrieve(self.case, "original question", 4096, 20)
        self.assertLessEqual(len(rows), 20)
        self.assertTrue(all(row.scale == 4096 for row in rows))

    def test_global_ranking_across_slides(self):
        rows = self.retriever.retrieve(self.case, "original question", 4096, 2)
        self.assertEqual([row.slide_key for row in rows], ["slide-a", "slide-b"])

    def test_replenish_budget_and_visited_filter(self):
        initial = self.retriever.retrieve(self.case, "original question", 4096, 2)
        visited = {row.identity for row in initial}
        extra = self.retriever.retrieve(
            self.case, "missing morphology", 4096, 5, visited
        )
        self.assertLessEqual(len(extra), 5)
        self.assertFalse(visited.intersection(row.identity for row in extra))

    def test_dynamic_missing_info_uses_online_encoder(self):
        self.retriever.retrieve(self.case, "missing morphology", 4096, 1)
        self.assertIn("missing morphology", self.encoder.calls)

    def test_dimension_mismatch_is_fatal(self):
        class BadEncoder:
            def encode(self, text):
                return np.ones(3)
        with self.assertRaisesRegex(ValueError, "dimension mismatch"):
            PathAgentRetriever(BadEncoder()).retrieve(
                self.case, "query", 4096, 1
            )

    def test_zoom_parents_only_use_current_round(self):
        current = self.retriever.retrieve(self.case, "original question", 4096, 3)
        unrelated = PatchRecord(4096, "missing", "missing", 99, 0, 0, 4096)
        parents = self.retriever.select_zoom_parents(
            self.case, current, np.asarray([1.0, 0.0]), 2
        )
        self.assertEqual(len(parents), 2)
        self.assertTrue(all(parent in current for parent in parents))
        self.assertNotIn(unrelated, parents)

    def test_zoom_child_is_same_slide_and_spatially_contained(self):
        parent = PatchRecord(4096, "slide-a_0_4096", "slide-a", 0, 0, 0, 4096)
        child, count, selected_parent = self.retriever.select_zoom_child(
            self.case, [parent], 1024, np.asarray([1.0, 0.0])
        )
        self.assertGreaterEqual(count, 2)
        self.assertEqual(child.slide_key, parent.slide_key)
        self.assertEqual(selected_parent, parent)
        cx, cy = child.center
        self.assertTrue(parent.x <= cx <= parent.x + parent.size)
        self.assertTrue(parent.y <= cy <= parent.y + parent.size)

    def test_zoom_uses_precomputed_features_and_selects_top_one(self):
        parent = PatchRecord(4096, "slide-a_0_4096", "slide-a", 0, 0, 0, 4096)
        child, count, _ = self.retriever.select_zoom_child(
            self.case, [parent], 1024, np.asarray([1.0, 0.0])
        )
        self.assertEqual(child.patch_index, 1)
        self.assertGreater(count, 1)
        self.assertIsInstance(child.similarity, float)


class ReasoningLoopTest(unittest.TestCase):
    item = {
        "Id": "TCGA-AA-0001",
        "Question": "original question",
        "Choice": ["A", "B"],
        "Answer": "SECRET_REFERENCE",
        "reference_answer": "SECRET_REFERENCE",
    }

    def test_replenish_encodes_missing_info_and_never_abstains(self):
        encoder = FakeEncoder()
        executor = FakeExecutor(sufficient=["No", "Yes"], zoom=[False])
        baseline = make_baseline(executor, encoder, initial=2, replenish=1, attempts=3)
        result = baseline.answer(self.item, synthetic_case())
        self.assertIn("missing morphology", encoder.calls)
        self.assertFalse(result["abstained"])
        self.assertTrue(result["predicted_can_answer"])
        self.assertIn(result["agent_answer"]["answer"], self.item["Choice"])
        self.assertEqual(result["process"]["rounds"][1]["new_patch_count"], 1)
        self.assertEqual(
            len(result["process"]["rounds"][0]["patch_descriptions"]), 2
        )
        self.assertEqual(result["process"]["query_dimension"], 2)
        self.assertEqual(result["process"]["feature_dimensions"]["4096"], 2)

    def test_zoom_happens_once_then_directly_finalizes(self):
        executor = FakeExecutor(sufficient=["No"], zoom=[True])
        baseline = make_baseline(executor, initial=2, attempts=5)
        result = baseline.answer(self.item, synthetic_case())
        self.assertEqual(result["process"]["zoom_action_count"], 1)
        self.assertEqual(len(result["process"]["rounds"]), 1)
        self.assertEqual(executor.final_calls, 1)
        zoom = result["process"]["rounds"][0]["zoom"]
        self.assertEqual(zoom["requested_scale"], 1024)
        self.assertEqual(zoom["selected_child"]["scale"], 1024)

    def test_max_attempts_still_produces_choice(self):
        executor = FakeExecutor(sufficient=["No", "No"], zoom=[False, False])
        baseline = make_baseline(executor, initial=1, replenish=1, attempts=2)
        result = baseline.answer(self.item, synthetic_case())
        self.assertEqual(len(result["process"]["rounds"]), 2)
        self.assertIn(result["agent_answer"]["answer"], self.item["Choice"])

    def test_reference_answer_never_enters_inference_prompts(self):
        executor = FakeExecutor(sufficient=["Yes"])
        baseline = make_baseline(executor, initial=1)
        baseline.answer(self.item, synthetic_case())
        serialized = json.dumps(executor.prompts)
        self.assertNotIn("SECRET_REFERENCE", serialized)

    def test_output_declares_adaptations_and_pure_visual_semantics(self):
        result = make_baseline(FakeExecutor()).answer(self.item, synthetic_case())
        self.assertEqual(result["baseline_name"], "PathAgent-CONCH-MS")
        self.assertFalse(result["adaptations"]["offline_quilt_llava_descriptions"])
        self.assertIn("heuristic", result["retriever_semantics"])


class IsolationAndResumeTest(unittest.TestCase):
    def test_gold_answerable_subset_loader_selects_true_nonexcluded_only(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gold.json"
            path.write_text(json.dumps({"labels": [
                {
                    "Id": "TCGA-AA-0001", "Question": "answerable",
                    "can_answer": True, "exclude_from_evaluation": False,
                },
                {
                    "Id": "TCGA-AA-0001", "Question": "not answerable",
                    "can_answer": False, "exclude_from_evaluation": False,
                },
                {
                    "Id": "TCGA-AA-0001", "Question": "excluded",
                    "can_answer": True, "exclude_from_evaluation": True,
                },
            ]}), encoding="utf-8")
            keys = load_gold_answerable_keys(path, expected_count=1)
        self.assertEqual(keys, {("TCGA-AA-0001", "answerable")})

    def test_real_gold_answerable_subset_has_186_questions(self):
        path = Path(
            "/home/wl/agent_2026/dataset/"
            "WsiVQA_answerability_binary_flat_v1.json"
        )
        self.assertEqual(len(load_gold_answerable_keys(path)), 186)

    def test_source_does_not_call_forbidden_agents_or_attention(self):
        source = Path(__file__).resolve().parents[1] / "pathagent_baseline.py"
        text = source.read_text(encoding="utf-8")
        forbidden = (
            "AnswerabilityAgent", "PrototypeAwarePlanner", "KnowledgeRAG",
            "EvidenceVerifierAgent", "FusionVerificationAgent",
            "phenotype_attention", "program_attention", "gene_attention",
        )
        for value in forbidden:
            self.assertNotIn(value, text)

    def test_resume_does_not_duplicate_rows(self):
        class Store:
            calls = 0
            def load_case(self, case_id):
                self.calls += 1
                return synthetic_case()

        class Baseline:
            calls = 0
            def answer(self, item, case):
                self.calls += 1
                return {
                    "case_id": item["Id"], "question": item["Question"],
                    "choices": item["Choice"],
                    "agent_answer": {"answer": item["Choice"][0]},
                    "predicted_can_answer": True, "abstained": False,
                }

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "vqa.json"
            source.write_text(json.dumps([
                {
                    "Id": "TCGA-AA-0001", "Question": "question one",
                    "Choice": ["A", "B"], "Answer": "A",
                },
                {
                    "Id": "TCGA-AA-0001", "Question": "question two",
                    "Choice": ["A", "B"], "Answer": "A",
                },
            ]), encoding="utf-8")
            config = {
                "vqa_json": str(source), "output_dir": str(root / "out"),
                "answerability_labels": str(root / "unused.json"),
            }
            baseline, store = Baseline(), Store()
            output = run(config, limit=2, resume=True, baseline=baseline, feature_store=store)
            run(config, limit=2, resume=True, baseline=baseline, feature_store=store)
            self.assertEqual(len(output.read_text().splitlines()), 2)
            self.assertEqual(baseline.calls, 2)
            self.assertEqual(store.calls, 1)
            self.assertEqual(len(load_completed(output)), 2)

    def test_output_shape_is_readable_by_existing_evaluator_loader(self):
        import sys
        sys.path.insert(0, "/home/wl/agent_2026/g2p_toolbank_brca")
        from multiscale_vqa_agent.answerability_evaluation import (
            _load_predictions,
            _predicted_can_answer,
        )
        result = make_baseline(FakeExecutor()).answer(
            ReasoningLoopTest.item, synthetic_case()
        )
        result["reference_answer"] = "A"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "answers.jsonl"
            path.write_text(json.dumps(result) + "\n", encoding="utf-8")
            rows, total, duplicates = _load_predictions(path)
        self.assertEqual(total, 1)
        self.assertEqual(duplicates, 0)
        self.assertEqual(_predicted_can_answer(next(iter(rows.values()))), (True, False))


class ModelInterfaceTest(unittest.TestCase):
    class Client:
        enabled = True

        def __init__(self, responses):
            self.responses = iter(responses)
            self.calls = []

        def chat(self, system, user, **kwargs):
            self.calls.append({"system": system, "user": user, **kwargs})
            return next(self.responses)

    def test_perceptor_batches_at_most_five_images(self):
        responses = [json.dumps({"patches": [
            {
                "ordinal": index + 1, "pathology_features": "visible",
                "relevance_to_question": "relevant", "answer_hint": "A",
            }
            for index in range(size)
        ]}) for size in (5, 2)]
        client = self.Client(responses)
        perceptor = PathAgentPerceptor(client, batch_size=5)
        patches = [
            PatchRecord(4096, "slide", "slide", index, 0, 0, 4096,
                        image_path=f"/{index}.jpg")
            for index in range(7)
        ]
        rows = perceptor.describe("question", ["A", "B"], patches, 4096, "query")
        self.assertEqual(len(rows), 7)
        self.assertEqual([len(call["images"]) for call in client.calls], [5, 2])

    def test_final_answer_repair_returns_exact_choice(self):
        client = self.Client([
            json.dumps({"answer": "A with explanation", "explanation": "x"}),
            json.dumps({"answer": "A"}),
        ])
        executor = PathAgentExecutor(client)
        result = executor.final("question", ["A", "B"], [])
        self.assertEqual(result["answer"], "A")
        self.assertEqual(client.calls[0]["temperature"], 0.0)
        self.assertIs(client.calls[0]["enable_thinking"], False)

    def test_executor_prompt_hides_patch_identity_and_paths(self):
        client = self.Client([json.dumps({"answer": "A", "explanation": "x"})])
        executor = PathAgentExecutor(client)
        executor.final("question", ["A", "B"], [{
            "patch": {
                "scale": 4096, "slide_id": "SECRET_SLIDE", "x": 123,
                "y": 456, "image_path": "/secret/path.jpg",
            },
            "pathology_features": "visible glands",
            "relevance_to_question": "relevant",
            "answer_hint": "A",
        }])
        prompt = client.calls[0]["user"]
        self.assertNotIn("SECRET_SLIDE", prompt)
        self.assertNotIn("/secret/path.jpg", prompt)
        self.assertNotIn('"x": 123', prompt)
        self.assertIn("visible glands", prompt)

    def test_config_keeps_all_outputs_under_independent_project(self):
        config = json.loads(
            (Path(__file__).resolve().parents[1] / "config.json").read_text()
        )
        self.assertTrue(config["output_dir"].startswith("/home/wl/agent_2026/pathagent/outputs/"))
        self.assertTrue(config["pathagent_root"].endswith("/pathagent"))


if __name__ == "__main__":
    unittest.main()
