import tempfile
import unittest
from pathlib import Path

import numpy as np

from multiscale_vqa_agent.pipeline import MultiScaleVQAPipeline
from multiscale_vqa_agent.question_features import QuestionFeatureStore
from multiscale_vqa_agent.retrieval import MultiScaleRetrievalAgent
from multiscale_vqa_agent.schemas import EvidenceGroup, ExecutionPlan, PatchCandidate


class QuestionFeatureStoreTest(unittest.TestCase):
    def make_store(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        path = Path(temporary.name) / "questions.npz"
        features = np.zeros((2, 768), dtype=np.float32)
        features[0, 0] = 2.0
        features[1, 1] = 3.0
        np.savez(
            path,
            unique_questions=np.asarray(["Question one?", "Question two?"]),
            preprojection_768=features,
        )
        return QuestionFeatureStore(str(path))

    def test_lookup_returns_normalized_768_feature(self):
        feature = self.make_store().lookup("  Question   one? ")
        self.assertEqual(feature.shape, (768,))
        self.assertAlmostEqual(float(np.linalg.norm(feature)), 1.0, places=6)
        self.assertEqual(float(feature[0]), 1.0)

    def test_missing_question_is_fatal(self):
        with self.assertRaisesRegex(ValueError, "Question not found"):
            self.make_store().lookup("question one?")


class QuestionRetrievalTest(unittest.TestCase):
    @staticmethod
    def agent(top=2):
        return MultiScaleRetrievalAgent(None, {
            "question_top_patches_per_slide": top,
            "max_evidence_groups": 4,
            "same_scale_iou": 0.5,
            "feature_cosine": 0.99,
            "global_bypass_per_scale": 2,
        })

    @staticmethod
    def scale_results(features):
        return {
            4096: {
                "slides": [{
                    "slide_id": "slide_0_4096",
                    "features": np.asarray(features, dtype=np.float32),
                    "coords": [
                        (0, 0, 4096),
                        (5000, 0, 4096),
                        (10000, 0, 4096),
                    ],
                }]
            }
        }

    def test_cosine_maximum_patch_ranks_first(self):
        groups = self.agent().retrieve_by_question(
            np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
            self.scale_results([
                [0.1, 1.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.5, 0.5, 0.0],
            ]),
        )
        self.assertEqual(groups[0].patches[4096].patch_index, 1)
        source = groups[0].patches[4096].sources[0]
        self.assertEqual(source["type"], "question_similarity")
        self.assertAlmostEqual(source["similarity"], 1.0, places=6)

    def test_dimension_mismatch_is_fatal(self):
        with self.assertRaisesRegex(ValueError, "dimension mismatch"):
            self.agent().retrieve_by_question(
                np.ones(4, dtype=np.float32),
                self.scale_results(np.eye(3, dtype=np.float32)),
            )

    @staticmethod
    def patch(source_type, name, x=0, feature=None):
        return PatchCandidate(
            scale=4096,
            slide_id="slide_0_4096",
            patch_index=x,
            x=x,
            y=0,
            size=4096,
            score=1.0,
            sources=[{
                "type": source_type,
                "name": name,
                "attention": 1.0,
            }],
            feature=np.asarray(
                feature if feature is not None else [1.0, 0.0],
                dtype=np.float32,
            ),
        )

    def test_hybrid_cross_source_dedup_merges_provenance(self):
        question = EvidenceGroup(
            1, 1.0, {4096: self.patch("question_similarity", "question")}
        )
        prototype = EvidenceGroup(
            1, 0.9, {4096: self.patch("phenotype", "histology")}
        )
        groups = self.agent().merge_hybrid_groups([question], [prototype])

        self.assertEqual(len(groups), 1)
        self.assertEqual(
            groups[0].evidence_source,
            "question_similarity+selected_phenotype",
        )
        self.assertEqual(
            {source["type"] for source in groups[0].patches[4096].sources},
            {"question_similarity", "phenotype"},
        )


class RetrievalSpy:
    def __init__(self):
        self.calls = {"question": 0, "broad": 0, "direct": 0}

    def retrieve_by_question(self, feature, scale_results):
        self.calls["question"] += 1
        return [EvidenceGroup(
            1, 1.0, {}, evidence_source="question_similarity"
        )]

    def retrieve_all_phenotypes(self, scale_results):
        self.calls["broad"] += 1
        return []

    def retrieve(self, field, group, scale_results, relations):
        self.calls["direct"] += 1
        return [EvidenceGroup(
            1, 0.9, {}, evidence_source="selected_phenotype"
        )]

    def merge_hybrid_groups(self, question_groups, prototype_groups):
        groups = list(question_groups) + list(prototype_groups)
        for index, group in enumerate(groups, 1):
            group.group_id = index
        return groups


class QuestionStoreSpy:
    feature_dim = 768

    def __init__(self):
        self.calls = 0

    def lookup(self, question):
        self.calls += 1
        return np.ones(768, dtype=np.float32)


class CropperSpy:
    def __init__(self):
        self.overview_calls = 0

    def crop_groups(self, case_id, label, groups):
        return groups

    def overview_thumbnails(self, case_id):
        self.overview_calls += 1
        return ["overview-1.jpg", "overview-2.jpg"]


class PathologySpy:
    def __init__(self):
        self.calls = []

    def describe(self, question, field, groups, overview_paths=None):
        overview_paths = list(overview_paths or [])
        self.calls.append({
            "question": question,
            "field": field,
            "overview_paths": overview_paths,
            "group_sources": [group.evidence_source for group in groups],
        })
        return {
            "description": "visible morphology",
            "evidence_groups": [],
            "image_metadata": [
                {"kind": "overview"} for _ in overview_paths
            ] + [
                {"kind": "patch", "evidence_source": group.evidence_source}
                for group in groups
            ],
        }


class G2PSpy:
    def fuse_task(self, scale_results, field):
        return {
            "field": field,
            "fused": {
                "predicted_class": 1,
                "predicted_label": "yes",
                "probability": 0.8,
            },
        }


class RelationSpy:
    def reason(self, field, scale_results):
        return {"programs": [], "genes": []}


class FusionSpy:
    def answer_with_summary(self, *args, **kwargs):
        return ({"answer": "yes", "json_parse_success": True}, {
            "task_match": args[0].task_match,
            "evidence_route": args[0].evidence_route,
            "selected_prototype_ids": [],
            "requested_fields": list(args[0].target_phenotypes),
            "executed_fields": list(args[0].target_phenotypes),
            "missing_fields": [],
            "structured_candidate_answer": None,
            "structured_candidate_id": None,
            "structured_candidate_confidence": 0.0,
            "option_alignment": {},
        })


class RegistrySpy:
    phenotype_fields = ["histological_type_label"]
    field_to_index = {"histological_type_label": 0}
    field_to_name = {"histological_type_label": "Histological Type"}
    field_to_prototype_id = {"histological_type_label": "P001"}
    vocabs = {1024: {"phenotype_groups": {"Histological Type": "morphology"}}}


class PipelineRouteTest(unittest.TestCase):
    def pipeline(
        self,
        morphology_mode,
        partial_mode="selected_phenotype",
        direct_mode="selected_phenotype",
    ):
        pipeline = MultiScaleVQAPipeline.__new__(MultiScaleVQAPipeline)
        pipeline.morphology_retrieval_mode = morphology_mode
        pipeline.partial_retrieval_mode = partial_mode
        pipeline.direct_retrieval_mode = direct_mode
        pipeline.question_features = QuestionStoreSpy()
        pipeline.retrieval = RetrievalSpy()
        pipeline.cropper = CropperSpy()
        pipeline.pathology = PathologySpy()
        pipeline.g2p = G2PSpy()
        pipeline.relation = RelationSpy()
        pipeline.fusion = FusionSpy()
        pipeline.registry = RegistrySpy()
        return pipeline

    @staticmethod
    def item(question="Which pattern is present?"):
        return {
            "Id": "TCGA-XX-0001",
            "Question": question,
            "Choice": ["yes", "no"],
            "Answer": "yes",
        }

    @staticmethod
    def plan(
        route="morphology_only",
        task_match="none",
        question="Which pattern is present?",
    ):
        fields = [] if route == "morphology_only" else ["histological_type_label"]
        return ExecutionPlan(
            case_id="TCGA-XX-0001",
            question=question,
            target_phenotypes=fields,
            task_type="morphology",
            metrics=[],
            answer_mode="multiple_choice",
            supported=bool(fields),
            support_reason="test",
            task_match=task_match,
            evidence_route=route,
        )

    def run_question(self, pipeline, plan):
        return pipeline._run_question(
            self.item(plan.question), plan, {}, {}, crop_patches=False
        )

    def test_question_mode_uses_only_question_retrieval_and_keeps_overviews(self):
        pipeline = self.pipeline("question_similarity")
        result = self.run_question(pipeline, self.plan())
        self.assertEqual(pipeline.retrieval.calls, {
            "question": 1, "broad": 0, "direct": 0,
        })
        self.assertEqual(pipeline.question_features.calls, 1)
        self.assertEqual(pipeline.cropper.overview_calls, 1)
        self.assertEqual(
            pipeline.pathology.calls[0]["overview_paths"],
            ["overview-1.jpg", "overview-2.jpg"],
        )
        self.assertEqual(
            result["pathology_evidence"]["retrieval_mode"],
            "question_similarity",
        )
        self.assertEqual(len(result["broad_g2p_predictions"]), 1)
        self.assertEqual(
            result["broad_g2p_predictions"][0]["field"],
            "histological_type_label",
        )

    def test_broad_mode_preserves_all_phenotype_retrieval(self):
        pipeline = self.pipeline("broad")
        self.run_question(pipeline, self.plan())
        self.assertEqual(pipeline.retrieval.calls, {
            "question": 0, "broad": 1, "direct": 0,
        })
        self.assertEqual(pipeline.question_features.calls, 0)

    def test_direct_default_mode_uses_selected_phenotype_without_overview(self):
        pipeline = self.pipeline("question_similarity", "question_similarity")
        self.run_question(pipeline, self.plan("phenotype_direct", "direct"))
        self.assertEqual(pipeline.retrieval.calls, {
            "question": 0, "broad": 0, "direct": 1,
        })
        self.assertEqual(pipeline.question_features.calls, 0)
        self.assertEqual(pipeline.cropper.overview_calls, 0)

    def test_direct_question_mode_keeps_structured_evidence_and_relations(self):
        pipeline = self.pipeline(
            "question_similarity", "question_similarity", "question_similarity"
        )
        result = self.run_question(
            pipeline, self.plan("phenotype_direct", "direct")
        )
        self.assertEqual(pipeline.retrieval.calls, {
            "question": 1, "broad": 0, "direct": 0,
        })
        self.assertEqual(pipeline.question_features.calls, 1)
        self.assertEqual(pipeline.cropper.overview_calls, 0)
        self.assertEqual(len(result["phenotype_predictions"]), 1)
        self.assertIn("histological_type_label", result["relation_evidence_by_field"])
        pathology = result["pathology_evidence"]
        self.assertEqual(pathology["primary_field"], "histological_type_label")
        self.assertEqual(
            pathology["structured_fields_covered"], ["histological_type_label"]
        )
        self.assertEqual(pathology["retrieval_mode"], "question_similarity")
        self.assertEqual(
            pathology["question_feature_source"], "CONCH_v1_preprojection_768"
        )
        self.assertEqual(pathology["question_feature_dim"], 768)

    def test_direct_question_cache_is_question_specific(self):
        pipeline = self.pipeline(
            "question_similarity", "question_similarity", "question_similarity"
        )
        cache = {}
        first = self.plan(
            "phenotype_direct", "direct", "Is ductal carcinoma present?"
        )
        second = self.plan(
            "phenotype_direct", "direct", "Is lobular carcinoma present?"
        )
        pipeline._run_question(self.item(first.question), first, {}, cache, False)
        pipeline._run_question(self.item(second.question), second, {}, cache, False)
        self.assertEqual(pipeline.retrieval.calls["question"], 2)
        self.assertEqual(pipeline.question_features.calls, 2)
        self.assertEqual(len([
            key for key in cache if key.startswith("__direct_question_groups__:")
        ]), 2)

    def test_direct_and_partial_hybrid_keep_all_evidence(self):
        for task_match in ("direct", "partial"):
            with self.subTest(task_match=task_match):
                pipeline = self.pipeline(
                    "question_similarity",
                    "hybrid_question_prototype",
                    "hybrid_question_prototype",
                )
                result = self.run_question(
                    pipeline, self.plan("phenotype_direct", task_match)
                )
                self.assertEqual(pipeline.retrieval.calls, {
                    "question": 1, "broad": 0, "direct": 1,
                })
                self.assertEqual(pipeline.cropper.overview_calls, 1)
                self.assertEqual(
                    pipeline.pathology.calls[0]["overview_paths"],
                    ["overview-1.jpg", "overview-2.jpg"],
                )
                self.assertEqual(
                    pipeline.pathology.calls[0]["group_sources"],
                    ["question_similarity", "selected_phenotype"],
                )
                self.assertEqual(len(result["phenotype_predictions"]), 1)
                self.assertIn(
                    "histological_type_label",
                    result["relation_evidence_by_field"],
                )
                pathology = result["pathology_evidence"]
                self.assertEqual(
                    pathology["retrieval_mode"],
                    "hybrid_question_prototype",
                )
                self.assertEqual(pathology["question_group_count"], 1)
                self.assertEqual(pathology["prototype_group_count"], 1)
                self.assertEqual(pathology["overview_count"], 2)
                self.assertEqual(pathology["overview_image_count"], 2)
                self.assertEqual(pathology["question_image_count"], 1)
                self.assertEqual(pathology["prototype_image_count"], 1)
                self.assertEqual(pathology["visual_evidence_order"], [
                    "overview", "question_similarity", "selected_phenotype"
                ])

    def test_hybrid_question_cache_is_question_specific(self):
        pipeline = self.pipeline(
            "question_similarity",
            "hybrid_question_prototype",
            "hybrid_question_prototype",
        )
        cache = {}
        first = self.plan(
            "phenotype_direct", "direct", "Is ductal carcinoma present?"
        )
        second = self.plan(
            "phenotype_direct", "direct", "Is lobular carcinoma present?"
        )
        pipeline._run_question(self.item(first.question), first, {}, cache, False)
        pipeline._run_question(self.item(second.question), second, {}, cache, False)
        self.assertEqual(pipeline.retrieval.calls["question"], 2)
        self.assertEqual(pipeline.retrieval.calls["direct"], 1)
        self.assertEqual(len([
            key for key in cache if key.startswith("__question_groups__:")
        ]), 2)

    def test_partial_default_mode_preserves_selected_phenotype_retrieval(self):
        pipeline = self.pipeline("question_similarity")
        self.run_question(pipeline, self.plan("phenotype_direct", "partial"))
        self.assertEqual(pipeline.retrieval.calls, {
            "question": 0, "broad": 0, "direct": 1,
        })
        self.assertEqual(pipeline.question_features.calls, 0)
        self.assertEqual(pipeline.cropper.overview_calls, 0)

    def test_partial_question_mode_keeps_structured_evidence_and_relations(self):
        pipeline = self.pipeline("question_similarity", "question_similarity")
        result = self.run_question(
            pipeline, self.plan("phenotype_direct", "partial")
        )
        self.assertEqual(pipeline.retrieval.calls, {
            "question": 1, "broad": 0, "direct": 0,
        })
        self.assertEqual(pipeline.question_features.calls, 1)
        self.assertEqual(pipeline.cropper.overview_calls, 0)
        self.assertEqual(len(result["phenotype_predictions"]), 1)
        self.assertIn("histological_type_label", result["relation_evidence_by_field"])
        pathology = result["pathology_evidence"]
        self.assertEqual(pathology["primary_field"], "histological_type_label")
        self.assertEqual(
            pathology["structured_fields_covered"], ["histological_type_label"]
        )
        self.assertEqual(pathology["retrieval_mode"], "question_similarity")
        self.assertEqual(
            pathology["question_feature_source"], "CONCH_v1_preprojection_768"
        )
        self.assertEqual(pathology["question_feature_dim"], 768)

    def test_partial_question_cache_is_question_specific(self):
        pipeline = self.pipeline("question_similarity", "question_similarity")
        cache = {}
        first = self.plan(
            "phenotype_direct", "partial", "Is focal invasion present?"
        )
        second = self.plan(
            "phenotype_direct", "partial", "Is extensive invasion present?"
        )
        pipeline._run_question(self.item(first.question), first, {}, cache, False)
        pipeline._run_question(self.item(second.question), second, {}, cache, False)
        self.assertEqual(pipeline.retrieval.calls["question"], 2)
        self.assertEqual(pipeline.question_features.calls, 2)
        self.assertEqual(len([
            key for key in cache if key.startswith("__partial_question_groups__:")
        ]), 2)


if __name__ == "__main__":
    unittest.main()
