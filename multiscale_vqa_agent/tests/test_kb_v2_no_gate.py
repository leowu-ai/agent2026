import inspect
import json
import unittest
from pathlib import Path

from multiscale_vqa_agent.fusion import FusionVerificationAgent
from multiscale_vqa_agent.knowledge_rag import KnowledgeRAG
from multiscale_vqa_agent.mc_pipeline import MultipleChoiceVQAPipeline
from multiscale_vqa_agent.pipeline import MultiScaleVQAPipeline
from multiscale_vqa_agent.registry import ToolBankRegistry
from multiscale_vqa_agent.schemas import ExecutionPlan
from multiscale_vqa_agent.verifier import EvidenceVerifierAgent


PROJECT_ROOT = Path(__file__).resolve().parents[2]
KB_V2 = PROJECT_ROOT / "hybrid_pathology_knowledge_base_v2.zip"


class CapturingClient:
    enabled = True

    def __init__(self):
        self.calls = []

    def chat(self, system, user, **kwargs):
        self.calls.append((system, json.loads(user), kwargs))
        return json.dumps({
            "answer_id": "A",
            "confidence": 0.2,
            "explanation": "The first option is the most defensible forced choice.",
            "limitations": "Direct patient evidence is incomplete.",
        })


class KnowledgeRAGV2Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        config = json.loads(
            (PROJECT_ROOT / "multiscale_vqa_agent/config.servers.json").read_text()
        )
        registry = ToolBankRegistry({
            int(scale): Path(directory)
            for scale, directory in config["scales"].items()
        })
        cls.rag = KnowledgeRAG(str(KB_V2), registry)

    def test_v2_files_and_manifest_counts(self):
        self.assertEqual(self.rag.knowledge_base_version, "2.0")
        self.assertEqual(len(self.rag.evidence_limitations), 14)
        self.assertEqual(len(self.rag.proxy_evidence_rules), 15)
        self.assertEqual(len(self.rag.forced_choice_rules), 12)
        self.assertEqual(len(self.rag.reasoning_examples), 16)

    def _ids(self, question, choices):
        result = self.rag.retrieve(question, choices, [])
        return result, {
            key: {row["id"] for row in result[key]}
            for key in (
                "evidence_limitations", "proxy_evidence_rules",
                "reasoning_examples",
            )
        }

    def test_size_margin_tnm_and_fish_retrieval(self):
        _, ids = self._ids("What is the tumor size?", ["1 cm", "4 cm"])
        self.assertIn("limit_exact_tumor_size", ids["evidence_limitations"])
        self.assertIn("proxy_size_coarse_extent", ids["proxy_evidence_rules"])

        _, ids = self._ids("What is the closest margin distance?", ["1 mm", "5 mm"])
        self.assertIn("limit_margin", ids["evidence_limitations"])
        self.assertIn("proxy_margin_verified_edge", ids["proxy_evidence_rules"])

        _, ids = self._ids("What is the pTNM stage?", ["pT1N0", "pT2N1"])
        self.assertIn("limit_stage_tnm", ids["evidence_limitations"])
        self.assertIn("proxy_stage_componentwise", ids["proxy_evidence_rules"])

        _, ids = self._ids("What is the HER2 FISH ratio?", ["1.2", "2.4"])
        self.assertIn("limit_her2_ihc_ish", ids["evidence_limitations"])

    def test_examples_are_generic_and_query_has_no_patient_key(self):
        result = self.rag.retrieve(
            "What is the tumor size?", ["1 cm", "4 cm"], []
        )
        self.assertTrue(result["reasoning_examples"])
        self.assertTrue(all(
            row["evidence_role"] == "generic_reasoning_example"
            and row["patient_specific"] is False
            for row in result["reasoning_examples"]
        ))
        parameters = inspect.signature(self.rag.retrieve).parameters
        self.assertNotIn("case_id", parameters)
        self.assertNotIn("reference_answer", parameters)


class NoGateAndFusionTest(unittest.TestCase):
    def test_primary_mc_source_never_calls_answerability(self):
        source = inspect.getsource(MultipleChoiceVQAPipeline.run_multiple_choice)
        self.assertNotIn("_predict_answerability", source)
        self.assertNotIn("force_answer_all", source)

    def test_planner_payload_strips_gold_and_reference_fields(self):
        payload = MultiScaleVQAPipeline._planner_item({
            "Id": "TCGA-AA-0001",
            "Question": "What is the grade?",
            "Choice": ["low", "high"],
            "Answer": "high",
            "reference_answer": "high",
            "gold_can_answer": True,
            "exclude_from_evaluation": False,
        })
        self.assertEqual(
            set(payload), {"Id", "Question", "Choice"}
        )

    def test_verifier_finalize_is_terminal_and_not_abstain(self):
        actions = EvidenceVerifierAgent.available_actions("inspect_gene", True, True)
        self.assertEqual(actions, ["answer", "finalize"])
        verifier = EvidenceVerifierAgent(client=None)
        decision = verifier._normalize({
            "evidence_sufficient": False,
            "evidence_state": "unavailable",
            "missing_evidence_type": "unavailable",
            "next_action": "finalize",
            "search_exhausted": True,
        }, actions, [], [])
        self.assertFalse(decision["evidence_sufficient"])
        self.assertTrue(decision["search_exhausted"])

    def test_final_fusion_is_deterministic_and_forces_supplied_choice(self):
        client = CapturingClient()
        fusion = FusionVerificationAgent(client)
        plan = ExecutionPlan(
            case_id="TCGA-AA-0001",
            question="What is the exact tumor size?",
            target_phenotypes=[],
            task_type="morphology",
            metrics=[],
            answer_mode="multiple_choice",
            supported=False,
            support_reason="No direct phenotype prototype.",
            task_match="none",
            evidence_route="morphology_only",
        )
        answer = fusion.answer(
            plan, ["1 cm", "4 cm"], [], {}, {"description": "limited view"},
            agent_context={
                "final_evidence_state": "unavailable",
                "search_exhausted": True,
            },
        )
        self.assertEqual(answer["answer"], "1 cm")
        self.assertTrue(answer["answer_in_choices"])
        kwargs = client.calls[0][2]
        self.assertEqual(kwargs["temperature"], 0.0)
        self.assertEqual(kwargs["top_p"], 1.0)

    def test_runtime_kb_code_does_not_open_training_annotations(self):
        source = (PROJECT_ROOT / "multiscale_vqa_agent/knowledge_rag.py").read_text()
        self.assertNotIn("WsiVQA_train.json", source)
        self.assertNotIn("train_pattern_summary_v2", source)


if __name__ == "__main__":
    unittest.main()
