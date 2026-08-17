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
from multiscale_vqa_agent.knowledge_rag import KnowledgeRAG
from multiscale_vqa_agent.pathology import PathologyAgent
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

    def test_program_and_gene_tools_are_1024_only(self):
        program = self.agent.retrieve_program_1024(0, self.results)
        gene = self.agent.retrieve_gene_1024(0, self.results)
        self.assertTrue(all(set(group.patches) == {1024} for group in program))
        self.assertTrue(all(set(group.patches) == {1024} for group in gene))
        self.assertEqual(self.source_types(program), {"program"})
        self.assertEqual(self.source_types(gene), {"gene"})

    def test_legacy_retrieve_still_combines_all_source_types(self):
        relations = {
            "programs": [{"index": 0}],
            "genes": [{"index": 0}],
        }
        groups = self.agent.retrieve(
            "histological_type_label", "morphology", self.results, relations
        )
        self.assertTrue({"phenotype", "program", "gene"} <= self.source_types(groups))


class StateMachineTest(unittest.TestCase):
    def test_spatial_and_biological_order(self):
        self.assertEqual(
            EvidenceVerifierAgent.available_actions("inspect_4096", True, True),
            ["answer", "inspect_2048"],
        )
        self.assertEqual(
            EvidenceVerifierAgent.available_actions("inspect_2048", True, True),
            ["answer", "inspect_1024"],
        )
        self.assertNotIn(
            "inspect_gene",
            EvidenceVerifierAgent.available_actions("inspect_1024", True, True),
        )
        self.assertIn(
            "inspect_gene",
            EvidenceVerifierAgent.available_actions("inspect_program", True, True),
        )

    def test_explicit_unavailable_semantics_allow_early_abstain(self):
        for action in ("inspect_4096", "inspect_2048"):
            available = EvidenceVerifierAgent.available_actions(
                action, True, True, allow_early_abstain=True
            )
            self.assertIn("abstain", available)

    def test_1024_retains_program_or_terminal_abstain(self):
        self.assertEqual(
            EvidenceVerifierAgent.available_actions("inspect_1024", True, False),
            ["answer", "inspect_program", "abstain"],
        )
        self.assertEqual(
            EvidenceVerifierAgent.available_actions("inspect_1024", False, False),
            ["answer", "abstain"],
        )

    def test_invalid_action_uses_deterministic_next_evidence(self):
        verifier = EvidenceVerifierAgent(client=None)
        decision = verifier._normalize(
            {"next_action": "inspect_gene"},
            ["answer", "inspect_2048", "abstain"],
            [],
            [],
        )
        self.assertEqual(decision["next_action"], "inspect_2048")
        self.assertTrue(decision["verifier_fallback_used"])

    def test_disabled_verifier_yields_full_five_round_trace(self):
        verifier = EvidenceVerifierAgent(client=None)
        sequence = []
        action = "inspect_4096"
        for _ in range(5):
            sequence.append(action)
            available = verifier.available_actions(
                action, has_program_candidates=True, has_gene_candidates=True
            )
            decision = verifier._fallback(available, "synthetic")
            action = decision["next_action"]
        self.assertEqual(sequence, [
            "inspect_4096", "inspect_2048", "inspect_1024",
            "inspect_program", "inspect_gene",
        ])
        self.assertEqual(action, "answer")

    def test_terminal_failure_is_unverified_answer_not_abstain(self):
        decision = EvidenceVerifierAgent._fallback(
            ["answer", "abstain"], "synthetic failure"
        )
        self.assertEqual(decision["next_action"], "answer")
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
    def test_early_abstain_uses_explicit_plan_or_rag_semantics(self):
        helper = MultiScaleVQAPipeline._early_abstain_allowed
        self.assertFalse(helper({}, {
            "matched_concepts": [{"evidence_role": "supportive_domain_knowledge"}],
            "evidence_rules": [{"id": "rule_morphology_coarse_to_fine"}],
        }))
        self.assertTrue(helper({"requires_unavailable_context": True}, {}))
        self.assertTrue(helper({}, {
            "matched_concepts": [{
                "evidence_role": "unavailable_from_local_visual_evidence"
            }],
        }))
        self.assertTrue(helper({}, {
            "evidence_rules": [{"id": "rule_assay_specific_target"}],
        }))
        self.assertFalse(helper(
            {"target_phenotypes": ["histological_type_label"]},
            {"evidence_rules": [{"id": "rule_stage_and_outcome"}]},
        ))

    def test_target_or_useful_morphology_prevents_early_abstain(self):
        helper = MultiScaleVQAPipeline._early_abstain_allowed
        self.assertFalse(helper({
            "target_phenotypes": ["lymphovascular_invasion_label"],
            "prototype_coverage": "partial",
            "requires_unavailable_context": True,
            "local_morphology_useful": True,
        }, {}))
        self.assertFalse(helper({
            "target_phenotypes": ["ER_status_label"],
            "prototype_coverage": "partial",
            "requires_unavailable_context": True,
            "local_morphology_useful": False,
        }, {}))
        self.assertFalse(helper({
            "target_phenotypes": [],
            "requires_unavailable_context": True,
            "local_morphology_useful": True,
        }, {}))
        self.assertFalse(helper({
            "target_phenotypes": [],
            "requires_unavailable_context": False,
            "local_morphology_useful": True,
        }, {
            "evidence_rules": [{"id": "rule_assay_specific_target"}],
        }))

    def test_only_genuinely_unavailable_without_useful_evidence_allows_early_abstain(self):
        helper = MultiScaleVQAPipeline._early_abstain_allowed
        self.assertTrue(helper({
            "target_phenotypes": [],
            "requires_unavailable_context": True,
            "local_morphology_useful": False,
        }, {}))
        self.assertTrue(helper({
            "target_phenotypes": [],
            "requires_unavailable_context": False,
            "local_morphology_useful": False,
        }, {
            "evidence_rules": [{"id": "rule_assay_specific_target"}],
        }))

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


class FusionContextTest(unittest.TestCase):
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
        def describe(question, field, groups, overview_paths=None, hide_provenance=False):
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

    def test_real_hierarchical_runner_acquires_five_distinct_rounds(self):
        pipeline = self.configured_pipeline(
            EvidenceVerifierAgent(client=None)
        )
        result = self.run_question(pipeline, self.plan())
        actions = [row["action"] for row in result["agent_trace"]["actions"]]
        self.assertEqual(actions, [
            "inspect_4096", "inspect_2048", "inspect_1024",
            "inspect_program", "inspect_gene",
        ])
        self.assertEqual(result["agent_trace"]["inspected_scales"], [4096, 2048, 1024])
        self.assertFalse(result["post_search_abstained"])
        self.assertIsNone(result["abstain_stage"])
        self.assertTrue(result["evidence_sufficiency_unverified"])
        self.assertEqual(result["verifier_failure_count"], 5)
        self.assertIsNotNone(result["agent_answer"])
        self.assertTrue(all(
            row["scale"] == 1024
            for row in result["working_memory"]["supportive_evidence"]
            if row["evidence_type"] in {"program", "gene"}
        ))

    def test_authoritative_verifier_abstain_remains_post_search_abstain(self):
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
                action = next_evidence or "abstain"
                return {
                    "evidence_sufficient": False,
                    "evidence_state": (
                        "unavailable" if action == "abstain" else "insufficient"
                    ),
                    "missing_evidence_type": (
                        "unavailable" if action == "abstain" else "fine_visual"
                    ),
                    "conflict_detected": False,
                    "next_action": action,
                    "target": None,
                    "reason": "authoritative synthetic decision",
                    "verifier_fallback_used": False,
                    "evidence_sufficiency_unverified": False,
                }

        pipeline = self.configured_pipeline(AuthoritativeVerifier())
        result = self.run_question(pipeline, self.plan())
        self.assertTrue(result["post_search_abstained"])
        self.assertEqual(result["abstain_stage"], "evidence_sufficiency")
        self.assertFalse(result["evidence_sufficiency_unverified"])
        self.assertEqual(result["verifier_failure_count"], 0)
        self.assertIsNone(result["agent_answer"])

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
