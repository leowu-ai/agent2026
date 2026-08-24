import inspect
import json
import tempfile
import unittest
from dataclasses import fields
from pathlib import Path
from types import MethodType, SimpleNamespace

import numpy as np

from multiscale_vqa_agent.agent_memory import (
    EvidenceObservation,
    WorkingMemory,
)
from multiscale_vqa_agent.fusion import FusionVerificationAgent
from multiscale_vqa_agent.fusion_evidence import (
    build_structured_summary,
    multi_field_semantic_choice_alignment,
    primary_semantic_choice_alignment,
)
from multiscale_vqa_agent.knowledge_rag import KnowledgeRAG
from multiscale_vqa_agent.pathology import PATHOLOGY_SYSTEM_PROMPT, PathologyAgent
from multiscale_vqa_agent.pipeline import MultiScaleVQAPipeline
from multiscale_vqa_agent.registry import ToolBankRegistry
from multiscale_vqa_agent.retrieval import MultiScaleRetrievalAgent
from multiscale_vqa_agent.schemas import EvidenceGroup, ExecutionPlan, PatchCandidate
from multiscale_vqa_agent.verifier import (
    VERIFIER_SYSTEM_PROMPT,
    EvidenceVerifierAgent,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
KB_PATH = PROJECT_ROOT / "hybrid_pathology_knowledge_base_v1.zip"


class FakeRegistry:
    phenotype_fields = ["histological_type_label"]
    phenotype_names = ["Histologic type"]
    field_to_index = {"histological_type_label": 0}
    field_to_name = {"histological_type_label": "Histologic type"}
    field_to_prototype_id = {"histological_type_label": "P001"}
    programs = ["Program A", "Program B"]
    genes = ["GENE1", "GENE2"]


def synthetic_scale_results():
    results = {}
    for scale in (1024, 2048, 4096):
        results[scale] = {
            "slides": [{
                "slide_id": "TCGA-AA-0001-DX1_feature",
                "coords": [(0, 0, scale), (scale, 0, scale), (0, scale, scale)],
                "features": np.eye(3, dtype=np.float32),
                "phenotype_attention": np.asarray([[0.9, 0.2, 0.1]]),
                "program_attention": np.asarray([
                    [0.1, 0.8, 0.2], [0.2, 0.1, 0.7]
                ]),
                "gene_attention": np.asarray([
                    [0.3, 0.2, 0.9], [0.8, 0.1, 0.2]
                ]),
            }],
            "program_pred": np.asarray([0.8, -0.2]),
            "gene_pred": np.asarray([0.6, -0.1]),
        }
    return results


class KnowledgeRAGTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        config = json.loads(
            (PROJECT_ROOT / "multiscale_vqa_agent/config.servers.json").read_text()
        )
        registry = ToolBankRegistry({
            int(scale): Path(directory)
            for scale, directory in config["scales"].items()
        })
        cls.rag = KnowledgeRAG(str(KB_PATH), registry)

    def test_zip_manifest_and_all_declared_counts_load(self):
        self.assertEqual(len(self.rag.pathology_concepts), 54)
        self.assertEqual(len(self.rag.tool_semantics), 17)
        self.assertEqual(len(self.rag.evidence_rules), 10)
        self.assertEqual(len(self.rag.model_relations), 17)
        self.assertEqual(len(self.rag.program_gene_candidates), 22)
        self.assertEqual(self.rag.manifest["version"], "1.0")

    def test_target_tool_is_forced_and_rag_never_returns_answer(self):
        result = self.rag.retrieve(
            "What is the histological type?",
            ["ductal", "lobular"],
            ["histological_type_label"],
        )
        self.assertEqual(
            result["direct_tools"][0]["field"], "histological_type_label"
        )
        serialized = json.dumps(result).lower()
        self.assertNotIn("answer_id", serialized)
        self.assertNotIn("gold_can_answer", serialized)

    def test_api_does_not_accept_reference_information(self):
        parameters = inspect.signature(KnowledgeRAG.retrieve).parameters
        self.assertNotIn("reference_answer", parameters)
        self.assertNotIn("gold", parameters)

    def test_scale_guidance_is_resolution_specific_and_question_relevant(self):
        result = self.rag.retrieve(
            "Is the growth pattern lobular?", ["yes", "no"], ["lobular_binary"]
        )
        guidance = result["scale_specific_visual_guidance"]
        self.assertIn("global architecture", guidance["4096"][0])
        self.assertIn("intermediate", guidance["2048"][0])
        self.assertIn("cytology", guidance["1024"][0])
        joined = " ".join(
            row for rows in guidance.values() for row in rows
        ).lower()
        self.assertTrue("lobular" in joined or "single-file" in joined)


class RetrievalSeparationTest(unittest.TestCase):
    def setUp(self):
        self.agent = MultiScaleRetrievalAgent(FakeRegistry(), {
            "top_patches_per_source": 2,
            "all_phenotype_top_patches_per_prototype": 1,
            "max_evidence_groups": 3,
            "global_bypass_per_scale": 1,
            "question_top_patches_per_slide": 2,
        })
        self.results = synthetic_scale_results()

    @staticmethod
    def source_types(groups):
        return {
            source["type"]
            for group in groups
            for patch in group.patches.values()
            for source in patch.sources
        }

    def test_spatial_phenotype_round_contains_no_program_or_gene(self):
        for scale in (4096, 2048, 1024):
            groups = self.agent.retrieve_phenotype_only(
                "histological_type_label", self.results, scale
            )
            self.assertEqual(self.source_types(groups), {"phenotype"})
            self.assertTrue(all(set(group.patches) == {scale} for group in groups))

    def test_question_and_broad_rounds_contain_no_program_or_gene(self):
        question = self.agent.retrieve_question_scale(
            np.asarray([1.0, 0.0, 0.0]), self.results, 2048
        )
        broad = self.agent.retrieve_all_phenotypes_scale(self.results, 4096)
        self.assertNotIn("program", self.source_types(question))
        self.assertNotIn("gene", self.source_types(question))
        self.assertNotIn("program", self.source_types(broad))
        self.assertNotIn("gene", self.source_types(broad))

    def test_morphology_hybrid_contains_question_and_broad_sources(self):
        pipeline = MultiScaleVQAPipeline.__new__(MultiScaleVQAPipeline)
        pipeline.retrieval = self.agent
        pipeline.question_features = SimpleNamespace(
            lookup=lambda question: np.asarray([1.0, 0.0, 0.0])
        )
        pipeline.morphology_retrieval_mode = "question_similarity"
        pipeline.partial_retrieval_mode = "hybrid_question_prototype"
        pipeline.direct_retrieval_mode = "hybrid_question_prototype"
        plan = ExecutionPlan(
            case_id="case", question="What change is visible?",
            target_phenotypes=[], task_type="morphology", metrics=[],
            answer_mode="multiple_choice", supported=False, support_reason="",
            task_match="none", evidence_route="morphology_only",
            local_morphology_useful=True,
        )
        groups = pipeline._hierarchical_spatial_groups(
            plan, "morphology_only", self.results, {}, 4096, False
        )
        sources = {group.evidence_source for group in groups}
        self.assertTrue(any("question_similarity" in source for source in sources))
        self.assertTrue(any("broad_phenotype" in source for source in sources))

    def test_visual_phenotype_hybrid_contains_question_and_selected_sources(self):
        pipeline = MultiScaleVQAPipeline.__new__(MultiScaleVQAPipeline)
        pipeline.retrieval = self.agent
        pipeline.question_features = SimpleNamespace(
            lookup=lambda question: np.asarray([1.0, 0.0, 0.0])
        )
        pipeline.morphology_retrieval_mode = "question_similarity"
        pipeline.partial_retrieval_mode = "hybrid_question_prototype"
        pipeline.direct_retrieval_mode = "hybrid_question_prototype"
        plan = ExecutionPlan(
            case_id="case", question="What is the histological type?",
            target_phenotypes=["histological_type_label"],
            task_type="multiclass", metrics=[], answer_mode="multiple_choice",
            supported=True, support_reason="", task_match="direct",
            evidence_route="phenotype_direct",
        )
        groups = pipeline._hierarchical_spatial_groups(
            plan, "phenotype_direct", self.results, {}, 4096, False
        )
        sources = {group.evidence_source for group in groups}
        self.assertTrue(any("question_similarity" in source for source in sources))
        self.assertTrue(any("selected_phenotype" in source for source in sources))


    def test_program_and_gene_tools_are_1024_only(self):
        program = self.agent.retrieve_program_1024(0, self.results)
        gene = self.agent.retrieve_gene_1024(0, self.results)
        self.assertTrue(all(set(group.patches) == {1024} for group in program))
        self.assertTrue(all(set(group.patches) == {1024} for group in gene))
        self.assertEqual(self.source_types(program), {"program"})
        self.assertEqual(self.source_types(gene), {"gene"})

    def test_spatial_children_are_same_slide_and_linked(self):
        parents = self.agent.retrieve_phenotype_only(
            "histological_type_label", self.results, 4096
        )
        for group in parents:
            group.anchor_group_id = group.group_id
        children = self.agent.link_child_groups(
            parents,
            self.agent.retrieve_phenotype_only(
                "histological_type_label", self.results, 2048
            ),
        )
        self.assertTrue(children)
        self.assertTrue(all(group.parent_group_id for group in children))
        self.assertTrue(all(group.anchor_group_id for group in children))
        parent_by_id = {group.group_id: group for group in parents}
        for child_group in children:
            parent = next(iter(parent_by_id[child_group.parent_group_id].patches.values()))
            child = next(iter(child_group.patches.values()))
            self.assertTrue(self.agent._same_slide(parent, child))
            self.assertIn(
                child_group.spatial_relation,
                {"center_contained", "bounding_box_overlap"},
            )

    def test_raw_attention_is_distinct_from_rank_score(self):
        groups = self.agent.retrieve_phenotype_only(
            "histological_type_label", self.results, 1024
        )
        source = next(iter(groups[0].patches.values())).sources[0]
        self.assertIn("raw_attention", source)
        self.assertIn("retrieval_rank_score", source)
        self.assertEqual(source["score_semantics"], "relative_retrieval_rank_only")

    def test_legacy_retrieve_still_combines_all_source_types(self):
        relations = {
            "programs": [{"index": 0}],
            "genes": [{"index": 0}],
        }
        groups = self.agent.retrieve(
            "histological_type_label", "morphology", self.results, relations
        )
        self.assertTrue({"phenotype", "program", "gene"} <= self.source_types(groups))


class VisualSearchPolicyTest(unittest.TestCase):
    @staticmethod
    def plan(**overrides):
        values = {
            "case_id": "case", "question": "question",
            "target_phenotypes": [], "task_type": "morphology", "metrics": [],
            "answer_mode": "multiple_choice", "supported": False,
            "support_reason": "", "task_match": "none",
            "evidence_route": "morphology_only",
        }
        values.update(overrides)
        return ExecutionPlan(**values)

    def test_patch_assessable_morphology_is_eligible(self):
        policy = MultiScaleVQAPipeline._visual_search_policy(
            self.plan(local_morphology_useful=True), {}
        )
        self.assertTrue(policy["eligible"])

    def test_exact_nonvisual_context_skips_morphology(self):
        policy = MultiScaleVQAPipeline._visual_search_policy(
            self.plan(requires_unavailable_context=True), {}
        )
        self.assertFalse(policy["eligible"])

    def test_receptor_status_uses_structured_evidence_without_visual_search(self):
        policy = MultiScaleVQAPipeline._visual_search_policy(
            self.plan(
                target_phenotypes=["HER2_status_label"],
                supported=True, task_match="direct",
                evidence_route="phenotype_direct",
            ),
            {},
        )
        self.assertFalse(policy["eligible"])


class StateMachineTest(unittest.TestCase):
    def test_spatial_and_biological_order(self):
        self.assertEqual(
            EvidenceVerifierAgent.available_actions("inspect_4096", True, True),
            ["answer", "inspect_2048", "inspect_1024", "finalize"],
        )
        self.assertEqual(
            EvidenceVerifierAgent.available_actions("inspect_2048", True, True),
            ["answer", "inspect_1024", "finalize"],
        )
        self.assertNotIn(
            "inspect_gene",
            EvidenceVerifierAgent.available_actions("inspect_1024", True, True),
        )
        self.assertIn(
            "inspect_gene",
            EvidenceVerifierAgent.available_actions("inspect_program", True, True),
        )

    def test_verifier_never_exposes_abstain(self):
        for action in ("inspect_4096", "inspect_2048"):
            available = EvidenceVerifierAgent.available_actions(
                action, True, True
            )
            self.assertNotIn("abstain", available)
            self.assertIn("finalize", available)

    def test_1024_retains_optional_program_or_finalize(self):
        self.assertEqual(
            EvidenceVerifierAgent.available_actions("inspect_1024", True, False),
            ["answer", "inspect_program", "finalize"],
        )
        self.assertEqual(
            EvidenceVerifierAgent.available_actions("inspect_1024", False, False),
            ["answer", "finalize"],
        )

    def test_invalid_action_does_not_mechanically_deepen(self):
        verifier = EvidenceVerifierAgent(client=None)
        decision = verifier._normalize(
            {"next_action": "inspect_gene"},
            ["answer", "inspect_2048", "finalize"],
            [],
            [],
        )
        self.assertEqual(decision["next_action"], "finalize")
        self.assertTrue(decision["verifier_fallback_used"])
        self.assertTrue(decision["evidence_sufficiency_unverified"])

    def test_disabled_verifier_does_not_yield_fixed_five_round_trace(self):
        verifier = EvidenceVerifierAgent(client=None)
        decision = verifier._fallback(
            ["answer", "inspect_program", "finalize"], "synthetic"
        )
        self.assertEqual(decision["next_action"], "finalize")
        self.assertNotEqual(decision["next_action"], "inspect_program")

    def test_round0_allows_adaptive_visual_scale_or_answer(self):
        actions = EvidenceVerifierAgent.available_actions(
            "round0", True, False
        )
        self.assertEqual(
            actions, ["answer", "inspect_4096", "inspect_2048", "inspect_1024", "finalize"]
        )

    def test_fallback_prefers_mapped_direct_candidate(self):
        memory = WorkingMemory("case", "q", ["positive", "negative"], {}, {})
        memory.structured_candidate = {"choice_id": "A", "answer": "positive"}
        memory.option_alignment = {"mapping_complete": True}
        decision = EvidenceVerifierAgent._fallback(
            ["answer", "inspect_1024", "finalize"], "synthetic", memory
        )
        self.assertEqual(decision["next_action"], "finalize")

    def test_terminal_failure_finalizes_without_certifying_evidence(self):
        decision = EvidenceVerifierAgent._fallback(
            ["answer", "finalize"], "synthetic failure"
        )
        self.assertEqual(decision["next_action"], "finalize")
        self.assertTrue(decision["search_exhausted"])
        self.assertTrue(decision["verifier_fallback_used"])
        self.assertTrue(decision["evidence_sufficiency_unverified"])

    def test_verifier_prompt_and_memory_have_no_reference_fields(self):
        lowered = VERIFIER_SYSTEM_PROMPT.lower()
        self.assertNotIn("reference_answer", lowered)
        self.assertNotIn("gold_can_answer", lowered)
        memory_fields = {item.name for item in fields(WorkingMemory)}
        self.assertNotIn("reference_answer", memory_fields)
        self.assertNotIn("gold", memory_fields)


class GraphAndMemoryTest(unittest.TestCase):
    def pipeline_with_h(self):
        pipeline = MultiScaleVQAPipeline.__new__(MultiScaleVQAPipeline)
        h0 = np.asarray([[1.0, 0.0], [0.0, 1.0]])
        pipeline.g2p = SimpleNamespace(runtimes={
            scale: SimpleNamespace(relations={"H_gene_to_program": h0})
            for scale in (1024, 2048, 4096)
        })
        return pipeline

    def test_h_membership_accepts_member_and_rejects_nonmember(self):
        pipeline = self.pipeline_with_h()
        self.assertTrue(pipeline._gene_belongs_to_program(0, 0))
        self.assertFalse(pipeline._gene_belongs_to_program(1, 0))

    def test_memory_is_per_question_and_does_not_leak_targets(self):
        left = WorkingMemory("case", "q1", ["a"], {}, {})
        right = WorkingMemory("case", "q2", ["b"], {}, {})
        left.add_observation(EvidenceObservation(
            1, "inspect_program", "program", "supportive", 1024,
            "program", "Program A", "morphology", {}, [1],
        ))
        self.assertEqual(left.inspected_programs, ["Program A"])
        self.assertEqual(right.inspected_programs, [])
        self.assertIsNot(left.observations, right.observations)

    def test_same_program_and_gene_are_recorded_once(self):
        memory = WorkingMemory("case", "q", ["a"], {}, {})
        for round_index in (1, 2):
            memory.add_observation(EvidenceObservation(
                round_index, "inspect_program", "program", "supportive", 1024,
                "program", "Program A", "morphology", {}, [round_index],
            ))
        self.assertEqual(memory.inspected_programs, ["Program A"])

    def test_current_missing_evidence_replaces_stale_gap_and_keeps_history(self):
        memory = WorkingMemory("case", "q", ["a"], {}, {})
        memory.update_verifier({
            "missing_evidence_type": "intermediate_visual",
            "conflict_detected": False,
        })
        self.assertEqual(
            memory.current_missing_evidence, ["intermediate_visual"]
        )
        memory.update_verifier({
            "missing_evidence_type": "fine_visual",
            "conflict_detected": False,
        })
        self.assertEqual(memory.current_missing_evidence, ["fine_visual"])
        memory.update_verifier({
            "missing_evidence_type": "none",
            "conflict_detected": False,
        })
        self.assertEqual(memory.current_missing_evidence, [])
        self.assertEqual(memory.missing_evidence, [])
        self.assertEqual(memory.missing_evidence_history, [
            "intermediate_visual", "fine_visual",
        ])
        serialized = memory.to_dict()
        self.assertEqual(serialized["missing_evidence"], [])
        self.assertEqual(serialized["missing_evidence_history"], [
            "intermediate_visual", "fine_visual",
        ])


class EarlyAbstainAndProgramRankingTest(unittest.TestCase):
    def test_early_abstain_helper_is_removed(self):
        self.assertFalse(hasattr(MultiScaleVQAPipeline, "_early_abstain_allowed"))

    def test_morphology_program_ranking_uses_all_scales_and_consensus(self):
        pipeline = MultiScaleVQAPipeline.__new__(MultiScaleVQAPipeline)
        pipeline.registry = FakeRegistry()
        plan = SimpleNamespace(target_phenotypes=[])
        knowledge = {"candidate_programs": [
            {"name": "Program A", "relevance": 1.0},
            {"name": "Program B", "relevance": 1.0},
        ]}
        results = {
            1024: {"program_pred": np.asarray([0.9, 1.0])},
            2048: {"program_pred": np.asarray([0.9, 0.0])},
            4096: {"program_pred": np.asarray([0.9, 0.0])},
        }
        rows = pipeline._program_candidates(plan, knowledge, {}, results)
        self.assertEqual(rows[0]["name"], "Program A")
        self.assertEqual(rows[0]["per_scale_patient_score"], {
            "1024": 0.9, "2048": 0.9, "4096": 0.9,
        })
        self.assertAlmostEqual(rows[0]["patient_score"], 0.9)
        self.assertAlmostEqual(rows[0]["patient_activity"], 0.9)
        self.assertAlmostEqual(rows[0]["scale_consensus"], 1.0)
        self.assertEqual(
            rows[0]["selection_policy"],
            "knowledge_relevance_plus_multiscale_patient_activity_consensus",
        )

    def test_direct_target_relation_candidates_are_unchanged(self):
        pipeline = MultiScaleVQAPipeline.__new__(MultiScaleVQAPipeline)
        pipeline.registry = FakeRegistry()
        relation_row = {
            "index": 1, "name": "Program B", "score": 0.75,
            "patient_score": 0.2,
        }
        rows = pipeline._program_candidates(
            SimpleNamespace(target_phenotypes=["histological_type_label"]),
            {"candidate_programs": []},
            {"histological_type_label": {"programs": [relation_row]}},
            {},
        )
        self.assertEqual(rows[0]["index"], relation_row["index"])
        self.assertEqual(rows[0]["score"], relation_row["score"])
        self.assertEqual(rows[0]["patient_score"], relation_row["patient_score"])


class PathologyIsolationTest(unittest.TestCase):
    class Client:
        enabled = True

        def __init__(self):
            self.user = None

        def chat(self, system, user, **kwargs):
            self.user = user
            return "Visible morphology only."

    @staticmethod
    def group():
        patch = PatchCandidate(
            1024, "slide", 0, 0, 0, 1024, 1.0,
            sources=[{"type": "program", "name": "Secret Program"}],
            image_path="synthetic.jpg",
        )
        return EvidenceGroup(1, 1.0, {1024: patch}, "program_support")

    def test_hierarchical_prompt_metadata_hides_provenance(self):
        client = self.Client()
        result = PathologyAgent(client).describe(
            "question", "Secret Program", [self.group()], hide_provenance=True
        )
        prompt = client.user
        self.assertNotIn("Secret Program", prompt)
        self.assertNotIn("program_support", prompt)
        self.assertNotIn("evidence_source", prompt)
        self.assertEqual(result["evidence_groups"][0]["evidence_source"], "program_support")

    def test_legacy_metadata_keeps_evidence_source(self):
        client = self.Client()
        PathologyAgent(client).describe(
            "question", "field", [self.group()], hide_provenance=False
        )
        self.assertIn("program_support", client.user)

    def test_prompt_forbids_non_morphology_claims(self):
        prompt = PATHOLOGY_SYSTEM_PROMPT.lower()
        for forbidden in (
            "er status", "pr status", "her2 status", "triple-negative",
            "gene expression", "mutation", "ihc", "fish", "treatment",
            "clinical records",
        ):
            self.assertIn(forbidden, prompt)

    def test_guidance_is_labeled_as_lookup_not_observation(self):
        client = self.Client()
        PathologyAgent(client).describe(
            "question", "field", [self.group()],
            choices=["A", "B"], current_scale=1024,
            evidence_role="direct",
            visual_guidance=["Look for fine cellular cohesion."],
            hide_provenance=True,
        )
        payload = json.loads(client.user)
        self.assertEqual(payload["current_scale"], 1024)
        self.assertEqual(
            payload["scale_specific_visual_guidance"],
            ["Look for fine cellular cohesion."],
        )
        self.assertIn("not what is present", payload["evidence_rule"])
        self.assertIn("not patient evidence", PATHOLOGY_SYSTEM_PROMPT.lower())

    def test_parser_sanitizes_molecular_claim(self):
        parsed = PathologyAgent._normalize_morphology(json.dumps({
            "architecture": "single cell pattern",
            "cytology": "HER2 positive tumor cells",
            "stroma": "fibrous",
            "necrosis": "absent",
            "invasion_pattern": "infiltrative",
            "visible_findings": ["ER negative", "cohesive nests"],
            "target_visual_support": "supportive",
            "image_quality": "adequate",
        }))
        self.assertEqual(parsed["cytology"], "indeterminate")
        self.assertEqual(parsed["visible_findings"], ["cohesive nests"])


class FusionContextTest(unittest.TestCase):
    @staticmethod
    def categorical_prediction(field, label, probability, validation):
        predicted_class = 1 if label == "positive" else 0
        return {
            "field": field,
            "label_semantics": {
                "class_to_label": {"0": "negative", "1": "positive"}
            },
            "fused": {
                "probabilities": [1.0 - probability, probability],
                "predicted_class": predicted_class,
                "predicted_label": label,
            },
            "per_scale": {
                str(scale): {
                    "predicted_class": predicted_class,
                    "predicted_label": label,
                }
                for scale in (1024, 2048, 4096)
            },
            "validation_metrics": {
                str(scale): {
                    "metric_name": "AUC", "metric_value": validation,
                }
                for scale in (1024, 2048, 4096)
            },
        }

    @staticmethod
    def direct_plan(fields):
        return ExecutionPlan(
            case_id="case", question="status", target_phenotypes=list(fields),
            task_type="multiclass", metrics=[], answer_mode="multiple_choice",
            supported=True, support_reason="", task_match="direct",
            phenotype_relevance_score=1.0, prototype_coverage="complete",
        )

    def test_validation_reliability_attenuates_high_softmax(self):
        plan = self.direct_plan(["ER_status_label"])
        strong = build_structured_summary(plan, ["positive", "negative"], [
            self.categorical_prediction("ER_status_label", "positive", 0.95, 0.90)
        ])
        weak = build_structured_summary(plan, ["positive", "negative"], [
            self.categorical_prediction("ER_status_label", "positive", 0.95, 0.45)
        ])
        self.assertGreater(
            strong["structured_candidate_confidence"],
            weak["structured_candidate_confidence"],
        )
        self.assertGreater(strong["overall_structured_reliability"], 0.0)

    def test_validation_reliability_is_monotonic(self):
        plan = self.direct_plan(["ER_status_label"])
        values = []
        for validation in (0.3, 0.6, 0.9):
            summary = build_structured_summary(plan, ["positive", "negative"], [
                self.categorical_prediction(
                    "ER_status_label", "positive", 0.9, validation
                )
            ])
            values.append(summary["structured_candidate_confidence"])
        self.assertEqual(values, sorted(values))

    def test_primary_er_alignment_is_not_blocked_by_supporting_pr(self):
        structured = {
            "primary_fields": ["ER_status_label"],
            "predictions": [
                {"field": "ER_status_label", "predicted_label": "positive"},
                {"field": "PR_status_label", "predicted_label": "positive"},
            ],
        }
        alignment = primary_semantic_choice_alignment(structured, [
            "positive for estrogen receptors",
            "negative for estrogen receptors",
            "positive for progesterone receptors",
            "negative for progesterone receptors",
        ])
        self.assertEqual(alignment["choice_id"], "A")
        self.assertTrue(alignment["mapping_complete"])

    def test_joint_er_pr_her2_alignment(self):
        structured = {
            "requested_fields": [
                "ER_status_label", "PR_status_label", "HER2_status_label"
            ],
            "predictions": [
                {"field": "ER_status_label", "predicted_label": "positive"},
                {"field": "PR_status_label", "predicted_label": "negative"},
                {"field": "HER2_status_label", "predicted_label": "positive"},
            ],
        }
        alignment = multi_field_semantic_choice_alignment(structured, [
            "ER+/PR+/HER2+", "ER+/PR-/HER2+",
            "ER-/PR-/HER2-", "ER+/PR-/HER2-",
        ])
        self.assertTrue(alignment["mapping_complete"])
        self.assertEqual(alignment["choice_id"], "B")

    def test_joint_alignment_precedes_single_primary_in_fusion(self):
        fields = ["ER_status_label", "PR_status_label", "HER2_status_label"]
        predictions = [
            self.categorical_prediction(fields[0], "positive", 0.9, 0.8),
            self.categorical_prediction(fields[1], "negative", 0.1, 0.8),
            self.categorical_prediction(fields[2], "positive", 0.9, 0.8),
        ]
        fusion = FusionVerificationAgent(SimpleNamespace(enabled=False))
        structured = fusion.prepare_structured_summary(
            self.direct_plan(fields),
            ["ER+/PR+/HER2+", "ER+/PR-/HER2+", "ER-/PR-/HER2-"],
            predictions,
        )
        self.assertEqual(structured["structured_candidate_id"], "B")
        self.assertTrue(structured["joint_mapping_complete"])
        self.assertEqual(structured["primary_fields"], fields)

    def test_incomplete_joint_state_is_not_mapped(self):
        structured = {
            "requested_fields": [
                "ER_status_label", "PR_status_label", "HER2_status_label"
            ],
            "predictions": [
                {"field": "ER_status_label", "predicted_label": "positive"},
                {"field": "PR_status_label", "predicted_label": "negative"},
            ],
        }
        alignment = multi_field_semantic_choice_alignment(
            structured, ["ER+/PR-/HER2+", "ER+/PR-/HER2-"]
        )
        self.assertFalse(alignment["mapping_complete"])
        self.assertIsNone(alignment["choice_id"])

    def test_ambiguous_joint_options_are_not_mapped(self):
        structured = {
            "requested_fields": ["ER_status_label", "PR_status_label"],
            "predictions": [
                {"field": "ER_status_label", "predicted_label": "positive"},
                {"field": "PR_status_label", "predicted_label": "negative"},
            ],
        }
        alignment = multi_field_semantic_choice_alignment(
            structured, ["ER+/PR-", "ER positive and PR negative"]
        )
        self.assertFalse(alignment["mapping_complete"])
        self.assertIsNone(alignment["choice_id"])

    def test_verifier_distinguishes_categorical_status_from_assay(self):
        prompt = VERIFIER_SYSTEM_PROMPT.lower()
        self.assertIn("categorical er/pr/her2", prompt)
        self.assertIn("exact percentage", prompt)
        self.assertIn("not a measured assay", prompt)

    def test_accumulated_observations_and_supportive_role_reach_packet(self):
        plan = ExecutionPlan(
            case_id="case", question="question", target_phenotypes=[],
            task_type="morphology", metrics=[], answer_mode="multiple_choice",
            supported=True, support_reason="", task_match="none",
            evidence_route="morphology_only",
        )
        context = {
            "visual_observations": [
                {"action": "inspect_4096", "description": "coarse"},
                {"action": "inspect_1024", "description": "fine"},
            ],
            "supportive_biological_evidence": [{
                "type": "program", "name": "Program A",
                "scale": 1024, "evidence_role": "supportive",
            }],
            "limitations": ["not a measured assay"],
        }
        packet = FusionVerificationAgent(client=None)._build_evidence_packet(
            plan,
            ["a", "b"],
            {"task_match": "none"},
            {},
            {"description": "visual"},
            agent_context=context,
        )
        attached = packet["hierarchical_agent_context"]
        self.assertEqual(len(attached["visual_observations"]), 2)
        self.assertEqual(
            attached["supportive_biological_evidence"][0]["evidence_role"],
            "supportive",
        )
        rules = " ".join(packet["rules"]).lower()
        self.assertIn("wsi-derived evidence only", rules)
        self.assertIn("cannot become measured assay facts", rules)


class HierarchicalPipelineSyntheticTest(unittest.TestCase):
    class FakeG2P:
        def __init__(self):
            h = np.asarray([[1.0, 0.0], [0.0, 1.0]])
            self.runtimes = {
                scale: SimpleNamespace(relations={"H_gene_to_program": h})
                for scale in (1024, 2048, 4096)
            }

        @staticmethod
        def fuse_task(results, field):
            return {
                "field": field,
                "label_semantics": {
                    "class_to_label": {"0": "ductal", "1": "lobular"}
                },
                "weights": {"1024": 1 / 3, "2048": 1 / 3, "4096": 1 / 3},
                "per_scale": {
                    str(scale): {
                        "probabilities": [0.8, 0.2],
                        "predicted_class": 0,
                        "predicted_label": "ductal",
                    }
                    for scale in (1024, 2048, 4096)
                },
                "fused": {
                    "probabilities": [0.8, 0.2],
                    "predicted_class": 0,
                    "predicted_label": "ductal",
                },
                "validation_metrics": {},
            }

    class FakeRelation:
        @staticmethod
        def reason(field, results):
            return {
                "phenotype": field,
                "programs": [{
                    "index": 0,
                    "name": "Program A",
                    "score": 0.9,
                    "learned": 0.7,
                    "patient_score": 0.8,
                    "scale_consensus": 0.9,
                    "genes": [],
                }],
                "genes": [],
            }

        @staticmethod
        def _rank_genes(program_index, results, program_score):
            return [{
                "index": 0,
                "name": "GENE1",
                "score": 0.8,
                "gene_to_program": 1.0,
                "scale_consensus": 1.0,
                "patient_score": 0.6,
            }]

    class FakeKnowledge:
        @staticmethod
        def retrieve(question, choices, target_phenotypes):
            return {
                "matched_concepts": [],
                "direct_tools": [{"field": "histological_type_label"}],
                "supportive_tools": [],
                "scale_strategy": [],
                "candidate_programs": [{
                    "name": "Program A", "relevance": 1.0,
                    "sources": ["model_relation:histological_type_label"],
                }],
                "candidate_genes": [{
                    "name": "GENE1", "relevance": 1.0,
                    "sources": ["program:Program A"],
                }],
                "limitations": [],
                "evidence_rules": [],
                "retrieval_trace": {"method": "synthetic"},
            }

    class FakePathology:
        @staticmethod
        def describe(
            question, field, groups, overview_paths=None,
            hide_provenance=False, **kwargs,
        ):
            assert hide_provenance
            scales = sorted({scale for group in groups for scale in group.patches})
            return {
                "backend": "synthetic",
                "description": f"Visible morphology at {scales}.",
                "evidence_groups": [group.to_dict() for group in groups],
                "image_metadata": [],
            }

    class FakeCropper:
        @staticmethod
        def overview_thumbnails(case_id):
            return []

    @staticmethod
    def plan():
        return ExecutionPlan(
            case_id="TCGA-AA-0001",
            question="What is the histological type?",
            target_phenotypes=["histological_type_label"],
            task_type="multiclass",
            metrics=[],
            answer_mode="multiple_choice",
            supported=True,
            support_reason="synthetic",
            task_match="direct",
            evidence_route="phenotype_direct",
            selected_prototype_ids=["P001"],
            prototype_support_type="target_evidence",
            prototype_coverage="complete",
        )

    def configured_pipeline(self, verifier):
        pipeline = MultiScaleVQAPipeline.__new__(MultiScaleVQAPipeline)
        pipeline.agent_mode = "hierarchical_rag"
        pipeline.registry = FakeRegistry()
        pipeline.g2p = self.FakeG2P()
        pipeline.relation = self.FakeRelation()
        pipeline.retrieval = MultiScaleRetrievalAgent(FakeRegistry(), {
            "top_patches_per_source": 1,
            "max_evidence_groups": 2,
        })
        pipeline.direct_retrieval_mode = "selected_phenotype"
        pipeline.partial_retrieval_mode = "selected_phenotype"
        pipeline.morphology_retrieval_mode = "broad"
        pipeline.knowledge_rag = self.FakeKnowledge()
        pipeline.verifier = verifier
        pipeline.pathology = self.FakePathology()
        pipeline.cropper = self.FakeCropper()
        pipeline.fusion = FusionVerificationAgent(
            client=SimpleNamespace(enabled=False)
        )
        return pipeline

    @staticmethod
    def run_question(pipeline, plan):
        return pipeline._run_question(
            {
                "Id": "TCGA-AA-0001",
                "Question": plan.question,
                "Choice": ["ductal", "lobular"],
                "Answer": "ductal",
            },
            plan,
            synthetic_scale_results(),
            {},
            False,
        )

    def test_disabled_verifier_finalizes_at_round0_and_calls_fusion(self):
        pipeline = self.configured_pipeline(
            EvidenceVerifierAgent(client=None)
        )
        result = self.run_question(pipeline, self.plan())
        actions = [row["action"] for row in result["agent_trace"]["actions"]]
        self.assertEqual(actions, [])
        self.assertEqual(result["agent_trace"]["inspected_scales"], [])
        self.assertEqual(
            result["agent_trace"]["round0_decision"]["next_action"], "finalize"
        )
        self.assertEqual(
            result["agent_trace"]["structured_candidate_before_visual"]["choice_id"],
            "A",
        )
        self.assertTrue(result["search_exhausted"])
        self.assertFalse(result["final_evidence_sufficient"])
        self.assertFalse(result["abstained"])
        self.assertTrue(result["evidence_sufficiency_unverified"])
        self.assertEqual(result["verifier_failure_count"], 1)
        self.assertIsNotNone(result["agent_answer"])
        self.assertTrue(all(
            row["scale"] == 1024
            for row in result["working_memory"]["supportive_evidence"]
            if row["evidence_type"] in {"program", "gene"}
        ))

    def test_authoritative_finalize_still_invokes_fusion(self):
        class AuthoritativeVerifier(EvidenceVerifierAgent):
            def __init__(self):
                pass

            def decide(self, available_actions, **kwargs):
                next_evidence = next(
                    (
                        action for action in available_actions
                        if action.startswith("inspect_")
                    ),
                    None,
                )
                action = next_evidence or "finalize"
                return {
                    "evidence_sufficient": False,
                    "evidence_state": (
                        "unavailable" if action == "finalize" else "insufficient"
                    ),
                    "missing_evidence_type": (
                        "unavailable" if action == "finalize" else "fine_visual"
                    ),
                    "conflict_detected": False,
                    "next_action": action,
                    "target": None,
                    "reason": "authoritative synthetic decision",
                    "search_exhausted": action == "finalize",
                    "verifier_fallback_used": False,
                    "evidence_sufficiency_unverified": False,
                }

        pipeline = self.configured_pipeline(AuthoritativeVerifier())
        result = self.run_question(pipeline, self.plan())
        self.assertTrue(result["search_exhausted"])
        self.assertEqual(result["final_evidence_state"], "unavailable")
        self.assertFalse(result["evidence_sufficiency_unverified"])
        self.assertEqual(result["verifier_failure_count"], 0)
        self.assertFalse(result["abstained"])
        self.assertIsNotNone(result["agent_answer"])
        self.assertTrue(result["answer_in_choices"])

    def test_nonvisual_target_skips_pathology_and_still_invokes_fusion(self):
        class PreferInspectionVerifier(EvidenceVerifierAgent):
            def __init__(self):
                pass

            def decide(self, available_actions, **kwargs):
                visual = next((
                    action for action in available_actions
                    if action.startswith("inspect_")
                ), None)
                action = visual or "finalize"
                return {
                    "evidence_sufficient": False,
                    "evidence_state": "unavailable",
                    "missing_evidence_type": "unavailable",
                    "conflict_detected": False,
                    "next_action": action,
                    "target": None,
                    "reason": "synthetic decision",
                    "search_exhausted": action == "finalize",
                    "verifier_fallback_used": False,
                    "evidence_sufficiency_unverified": False,
                }

        class PathologyMustNotRun:
            @staticmethod
            def describe(*args, **kwargs):
                raise AssertionError("Patho-R1 must not run for a nonvisual target")

        plan = ExecutionPlan(
            case_id="TCGA-AA-0001", question="What is the HER2 status?",
            target_phenotypes=["HER2_status_label"], task_type="binary",
            metrics=[], answer_mode="multiple_choice", supported=True,
            support_reason="synthetic", task_match="direct",
            evidence_route="phenotype_direct", selected_prototype_ids=["P013"],
            prototype_support_type="target_evidence", prototype_coverage="complete",
        )
        pipeline = self.configured_pipeline(PreferInspectionVerifier())
        pipeline.pathology = PathologyMustNotRun()
        result = self.run_question(pipeline, plan)
        self.assertFalse(result["visual_search_eligible"])
        self.assertEqual(result["visual_retrieval_rounds"], [])
        self.assertEqual(result["agent_trace"]["inspected_scales"], [])
        self.assertIsNotNone(result["agent_answer"])
        self.assertTrue(result["answer_in_choices"])

    def test_same_case_multiple_questions_call_g2p_once(self):
        class CountingG2P:
            calls = 0

            def infer_case(self, case_id):
                self.calls += 1
                return {"case_id": case_id}

        pipeline = MultiScaleVQAPipeline.__new__(MultiScaleVQAPipeline)
        pipeline.agent_mode = "hierarchical_rag"
        pipeline.config = {"output_dir": "."}
        pipeline.answerability_only = False
        pipeline.planner_only = False
        pipeline.precomputed_answerability = None
        pipeline.answerability = SimpleNamespace(predict=lambda question, choices: {
            "can_answer": True,
            "confidence": 1.0,
            "reason": "synthetic",
            "fallback_used": False,
        })
        pipeline.planner = SimpleNamespace(plan=lambda item: ExecutionPlan(
            case_id="TCGA-AA-0001",
            question=item["Question"],
            target_phenotypes=[],
            task_type="morphology",
            metrics=[],
            answer_mode="multiple_choice",
            supported=True,
            support_reason="synthetic",
            task_match="none",
            evidence_route="morphology_only",
        ))
        pipeline.g2p = CountingG2P()

        def fake_hierarchical(self, item, plan, scale_results, evidence_cache, crop_patches):
            return {
                "case_id": plan.case_id,
                "question": plan.question,
                "agent_mode": "hierarchical_rag",
                "agent_answer": None,
                "abstained": True,
                "post_search_abstained": True,
            }

        pipeline._run_question_hierarchical = MethodType(fake_hierarchical, pipeline)
        rows = [
            {
                "Id": "TCGA-AA-0001",
                "Question": f"question {index}",
                "Choice": ["a", "b"],
            }
            for index in (1, 2)
        ]
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "vqa.json"
            output = Path(directory) / "answers.jsonl"
            source.write_text(json.dumps(rows), encoding="utf-8")
            pipeline.run(
                vqa_path=str(source),
                output_path=str(output),
                crop_patches=False,
                resume=False,
            )
        self.assertEqual(pipeline.g2p.calls, 1)


class SourceLeakageTest(unittest.TestCase):
    def test_new_modules_have_no_case_specific_or_gold_logic(self):
        for filename in (
            "knowledge_rag.py", "agent_memory.py", "verifier.py"
        ):
            text = (PROJECT_ROOT / "multiscale_vqa_agent" / filename).read_text()
            self.assertNotIn("TCGA-", text)
            self.assertNotIn("reference_answer", text)
            self.assertNotIn("gold_can_answer", text)


if __name__ == "__main__":
    unittest.main()
