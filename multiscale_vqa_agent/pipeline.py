import gc
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import torch

from .agent_memory import EvidenceObservation, WorkingMemory
from .answerability import AnswerabilityAgent
from .clients import OpenAICompatibleClient
from .fusion import FusionVerificationAgent
from .fusion_evidence import indexed_choices
from .g2p_runtime import MultiScaleG2PAgent
from .pathology import PathologyAgent
from .knowledge_rag import KnowledgeRAG
from .precomputed_answerability import PrecomputedAnswerabilityStore
from .question_features import QuestionFeatureStore
from .registry import PrototypeAwarePlanner, ToolBankRegistry
from .relation import RelationReasoningAgent
from .retrieval import MultiScaleRetrievalAgent, WSICropper
from .router_audit import write_router_audit
from .verifier import EvidenceVerifierAgent


class MultiScaleVQAPipeline:
    def __init__(
        self,
        config_path: str,
        planner_only: bool = False,
        answerability_only: bool = False,
        precomputed_answerability: Optional[str] = None,
        morphology_retrieval_mode: Optional[str] = None,
        partial_retrieval_mode: Optional[str] = None,
        direct_retrieval_mode: Optional[str] = None,
        agent_mode: str = "legacy",
        knowledge_base: Optional[str] = None,
    ):
        self.config_path = Path(config_path)
        with self.config_path.open(encoding="utf-8") as handle:
            self.config = json.load(handle)
        self.qwen = OpenAICompatibleClient(self.config["qwen"])
        self.answerability = AnswerabilityAgent(self.qwen)
        self.precomputed_answerability = (
            PrecomputedAnswerabilityStore(precomputed_answerability)
            if precomputed_answerability
            else None
        )
        self.planner_only = planner_only
        self.answerability_only = answerability_only
        self.agent_mode = str(agent_mode)
        if self.agent_mode not in {"legacy", "hierarchical_rag"}:
            raise ValueError(
                "agent_mode must be legacy or hierarchical_rag, got "
                f"{self.agent_mode!r}"
            )
        if answerability_only:
            return
        scale_dirs = {int(k): Path(v) for k, v in self.config["scales"].items()}
        self.registry = ToolBankRegistry(scale_dirs)
        self.planner = PrototypeAwarePlanner(self.registry, self.qwen)
        if planner_only:
            return
        self.g2p = MultiScaleG2PAgent(self.config, self.registry)
        self.relation = RelationReasoningAgent(self.registry, self.g2p, self.config["retrieval"])
        self.retrieval = MultiScaleRetrievalAgent(self.registry, self.config["retrieval"])
        self.morphology_retrieval_mode = str(
            morphology_retrieval_mode
            or self.config["retrieval"].get("morphology_retrieval_mode", "broad")
        )
        if self.morphology_retrieval_mode not in {"broad", "question_similarity"}:
            raise ValueError(
                "morphology_retrieval_mode must be broad or question_similarity, got "
                f"{self.morphology_retrieval_mode!r}"
            )
        self.partial_retrieval_mode = str(
            partial_retrieval_mode
            or self.config["retrieval"].get(
                "partial_retrieval_mode", "selected_phenotype"
            )
        )
        if self.partial_retrieval_mode not in {
            "selected_phenotype", "question_similarity",
            "hybrid_question_prototype",
        }:
            raise ValueError(
                "partial_retrieval_mode must be selected_phenotype, "
                "question_similarity, or hybrid_question_prototype, got "
                f"{self.partial_retrieval_mode!r}"
            )
        self.direct_retrieval_mode = str(
            direct_retrieval_mode
            or self.config["retrieval"].get(
                "direct_retrieval_mode", "selected_phenotype"
            )
        )
        if self.direct_retrieval_mode not in {
            "selected_phenotype", "question_similarity",
            "hybrid_question_prototype",
        }:
            raise ValueError(
                "direct_retrieval_mode must be selected_phenotype, "
                "question_similarity, or hybrid_question_prototype, got "
                f"{self.direct_retrieval_mode!r}"
            )
        self.question_features = None
        if (
            self.morphology_retrieval_mode == "question_similarity"
            or self.partial_retrieval_mode in {
                "question_similarity", "hybrid_question_prototype"
            }
            or self.direct_retrieval_mode in {
                "question_similarity", "hybrid_question_prototype"
            }
        ):
            feature_path = self.config["retrieval"].get("question_feature_path")
            if not feature_path:
                raise ValueError(
                    "question_similarity retrieval requires "
                    "retrieval.question_feature_path"
                )
            self.question_features = QuestionFeatureStore(feature_path)
        self.cropper = WSICropper(
            Path(self.config["wsi_root"]), Path(self.config["output_dir"]) / "evidence_patches"
        )
        self.pathology = PathologyAgent(OpenAICompatibleClient(self.config["pathor1"]))
        self.fusion = FusionVerificationAgent(self.qwen)
        self.knowledge_rag = None
        self.verifier = None
        if self.agent_mode == "hierarchical_rag":
            if not knowledge_base:
                raise ValueError(
                    "hierarchical_rag mode requires --knowledge_base"
                )
            self.knowledge_rag = KnowledgeRAG(knowledge_base, self.registry)
            self.verifier = EvidenceVerifierAgent(self.qwen)

    def run(
        self,
        vqa_path: Optional[str] = None,
        output_path: Optional[str] = None,
        limit: Optional[int] = None,
        crop_patches: bool = True,
        resume: bool = True,
        multiple_choice_only: bool = False,
        answerability_labels: Optional[str] = None,
    ) -> Path:
        source = Path(vqa_path or self.config["vqa_json"])
        with source.open(encoding="utf-8") as handle:
            items = json.load(handle)
        if multiple_choice_only:
            items = [
                item for item in items
                if item.get("Choice", item.get("choices"))
            ]
        if getattr(self, "precomputed_answerability", None) is not None:
            self.precomputed_answerability.validate_items(items)
            print(
                "precomputed_answerability "
                + json.dumps(
                    self.precomputed_answerability.summary(), ensure_ascii=False
                ),
                flush=True,
            )
        if limit is not None:
            items = items[:limit]
        destination = Path(output_path or (Path(self.config["output_dir"]) / "answers.jsonl"))
        destination.parent.mkdir(parents=True, exist_ok=True)
        completed = self._completed_keys(destination) if resume else set()
        mode = "a" if resume and destination.exists() else "w"
        grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for item in items:
            case_id, question = self._item_key(item)
            if (case_id, question) not in completed:
                grouped[case_id].append(item)

        if self.answerability_only:
            with destination.open(mode, encoding="utf-8") as handle:
                for case_items in grouped.values():
                    for item in case_items:
                        assessment = self._predict_answerability(item)
                        handle.write(json.dumps(
                            self._gate_only_result(item, assessment), ensure_ascii=False
                        ) + "\n")
                        handle.flush()
            self._evaluate_if_requested(destination, answerability_labels)
            return destination

        if self.planner_only:
            plans = []
            with destination.open(mode, encoding="utf-8") as handle:
                for case_items in grouped.values():
                    for item in case_items:
                        assessment = self._predict_answerability(item)
                        if not assessment["can_answer"]:
                            result = self._abstained_result(item, assessment)
                        else:
                            plan = self.planner.plan(item)
                            plans.append((item, plan))
                            result = self._attach_answerability(
                                {
                                    "case_id": plan.case_id,
                                    "question": plan.question,
                                    "input": item,
                                    "plan": plan.to_dict(),
                                },
                                assessment,
                            )
                        handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            summary = write_router_audit(plans, destination)
            print(f"Router audit: {json.dumps(summary, ensure_ascii=False)}", flush=True)
            self._evaluate_if_requested(destination, answerability_labels)
            return destination

        with destination.open(mode, encoding="utf-8") as handle:
            for case_number, (case_id, case_items) in enumerate(grouped.items(), 1):
                print(f"[{case_number}/{len(grouped)}] gate {case_id} ({len(case_items)} questions)", flush=True)
                answerable = []
                for item in case_items:
                    assessment = self._predict_answerability(item)
                    if not assessment["can_answer"]:
                        handle.write(json.dumps(
                            self._abstained_result(item, assessment), ensure_ascii=False
                        ) + "\n")
                        handle.flush()
                        continue
                    answerable.append((item, assessment, self.planner.plan(item)))
                if not answerable:
                    print(f"skip G2P {case_id}: all questions unanswerable", flush=True)
                    continue
                print(f"infer {case_id} ({len(answerable)} answerable questions)", flush=True)
                try:
                    scale_results = self.g2p.infer_case(case_id)
                    evidence_cache = {}
                    for item, assessment, plan in answerable:
                        result = self._attach_answerability(
                            self._run_question(
                                item, plan, scale_results, evidence_cache, crop_patches
                            ),
                            assessment,
                        )
                        handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                        handle.flush()
                except Exception as error:
                    for item, assessment, plan in answerable:
                        result = self._attach_answerability(
                            {
                                "case_id": case_id,
                                "question": plan.question,
                                "input": item,
                                "plan": plan.to_dict(),
                                "error": f"{type(error).__name__}: {error}",
                            },
                            assessment,
                        )
                        handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                    handle.flush()
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        self._evaluate_if_requested(destination, answerability_labels)
        return destination

    @staticmethod
    def _item_key(item: Dict[str, Any]):
        return (
            str(item.get("Id", item.get("case_id", "")))[:12],
            str(item.get("Question", item.get("question", ""))),
        )

    def _predict_answerability(self, item: Dict[str, Any]) -> Dict[str, Any]:
        case_id, question = self._item_key(item)
        if getattr(self, "precomputed_answerability", None) is not None:
            return self.precomputed_answerability.lookup(case_id, question)
        choices = list(item.get("Choice", item.get("choices", [])) or [])
        return self.answerability.predict(question, choices)

    @staticmethod
    def _attach_answerability(
        result: Dict[str, Any], assessment: Dict[str, Any]
    ) -> Dict[str, Any]:
        result.update({
            "predicted_can_answer": assessment["can_answer"],
            "predicted_answerability": (
                "answerable" if assessment["can_answer"] else "unanswerable"
            ),
            "answerability_confidence": assessment["confidence"],
            "answerability_reason": assessment["reason"],
            "answerability_fallback_used": assessment["fallback_used"],
        })
        result.setdefault("abstained", False)
        return result

    def _gate_only_result(
        self, item: Dict[str, Any], assessment: Dict[str, Any]
    ) -> Dict[str, Any]:
        case_id, question = self._item_key(item)
        return {
            "case_id": case_id,
            "question": question,
            "predicted_can_answer": assessment["can_answer"],
            "predicted_answerability": (
                "answerable" if assessment["can_answer"] else "unanswerable"
            ),
            "answerability_confidence": assessment["confidence"],
            "answerability_reason": assessment["reason"],
            "answerability_fallback_used": assessment["fallback_used"],
            "answerability_only": True,
        }

    def _abstained_result(
        self, item: Dict[str, Any], assessment: Dict[str, Any]
    ) -> Dict[str, Any]:
        case_id, question = self._item_key(item)
        choices = list(item.get("Choice", item.get("choices", [])) or [])
        return {
            "case_id": case_id,
            "question": question,
            "choices": choices,
            "choice_options": indexed_choices(choices),
            "reference_answer": item.get("Answer", item.get("answer")),
            "input": item,
            "plan": {},
            "predicted_can_answer": assessment["can_answer"],
            "predicted_answerability": "unanswerable",
            "answerability_confidence": assessment["confidence"],
            "answerability_reason": assessment["reason"],
            "answerability_fallback_used": assessment["fallback_used"],
            "abstained": True,
            "agent_answer": None,
        }

    @staticmethod
    def _evaluate_if_requested(destination: Path, labels_path: Optional[str]):
        if labels_path:
            from .answerability_evaluation import evaluate_answerability

            summary = evaluate_answerability(destination, Path(labels_path))
            print(f"Answerability evaluation: {json.dumps(summary, ensure_ascii=False)}", flush=True)

    def _run_question(
        self,
        item: Dict[str, Any],
        plan: Any,
        scale_results: Dict[int, Dict[str, Any]],
        evidence_cache: Dict[str, Any],
        crop_patches: bool,
    ) -> Dict[str, Any]:
        if getattr(self, "agent_mode", "legacy") == "hierarchical_rag":
            return self._run_question_hierarchical(
                item, plan, scale_results, evidence_cache, crop_patches
            )
        return self._run_question_legacy(
            item, plan, scale_results, evidence_cache, crop_patches
        )

    def _run_question_legacy(
        self,
        item: Dict[str, Any],
        plan: Any,
        scale_results: Dict[int, Dict[str, Any]],
        evidence_cache: Dict[str, Any],
        crop_patches: bool,
    ) -> Dict[str, Any]:
        choices = list(item.get("Choice", item.get("choices", [])) or [])
        predictions = []
        relations_by_field = {}
        pathology_by_field = {}
        broad_g2p_predictions = None
        overview_paths = []

        for field in plan.target_phenotypes:
            if field not in evidence_cache:
                evidence_cache[field] = {
                    "phenotype": self.g2p.fuse_task(scale_results, field),
                    "relations": self.relation.reason(field, scale_results),
                }
            evidence = evidence_cache[field]
            predictions.append(evidence["phenotype"])
            relations_by_field[field] = evidence["relations"]

        evidence_route = getattr(
            plan,
            "evidence_route",
            "phenotype_direct" if plan.target_phenotypes else "morphology_only",
        )
        if evidence_route == "phenotype_direct" and plan.target_phenotypes:
            primary = plan.target_phenotypes[0]
            primary_evidence = evidence_cache[primary]
            retrieval_mode = (
                self.direct_retrieval_mode
                if plan.task_match == "direct"
                else self.partial_retrieval_mode
            )
            use_hybrid_retrieval = (
                retrieval_mode == "hybrid_question_prototype"
            )
            use_question_retrieval = retrieval_mode in {
                "question_similarity", "hybrid_question_prototype"
            }

            if use_hybrid_retrieval:
                question_key = f"__question_groups__:{plan.question}"
                prototype_key = f"__prototype_groups__:{primary}"
                groups_key = f"__hybrid_groups__:{primary}:{plan.question}"
                if question_key not in evidence_cache:
                    question_feature = self.question_features.lookup(plan.question)
                    evidence_cache[question_key] = self.retrieval.retrieve_by_question(
                        question_feature, scale_results
                    )
                if prototype_key not in evidence_cache:
                    name = self.registry.field_to_name[primary]
                    vocab = self.registry.vocabs[min(self.registry.vocabs)]
                    group = vocab.get("phenotype_groups", {}).get(
                        name, "morphology"
                    )
                    evidence_cache[prototype_key] = self.retrieval.retrieve(
                        primary, group, scale_results, primary_evidence["relations"]
                    )
                if groups_key not in evidence_cache:
                    groups = self.retrieval.merge_hybrid_groups(
                        evidence_cache[question_key],
                        evidence_cache[prototype_key],
                    )
                    if crop_patches:
                        groups = self.cropper.crop_groups(
                            plan.case_id, f"hybrid:{plan.question}", groups
                        )
                    evidence_cache[groups_key] = groups
                overview_key = "__overview_thumbnails__"
                if overview_key not in evidence_cache:
                    evidence_cache[overview_key] = self.cropper.overview_thumbnails(
                        plan.case_id
                    )
                overview_paths = evidence_cache[overview_key]
            else:
                groups_key = (
                    f"__{plan.task_match}_question_groups__:{plan.question}"
                    if use_question_retrieval
                    else f"__pathology_groups__:{primary}"
                )
                if groups_key not in evidence_cache:
                    if use_question_retrieval:
                        question_feature = self.question_features.lookup(plan.question)
                        groups = self.retrieval.retrieve_by_question(
                            question_feature, scale_results
                        )
                        crop_label = plan.question
                    else:
                        name = self.registry.field_to_name[primary]
                        vocab = self.registry.vocabs[min(self.registry.vocabs)]
                        group = vocab.get("phenotype_groups", {}).get(
                            name, "morphology"
                        )
                        groups = self.retrieval.retrieve(
                            primary, group, scale_results,
                            primary_evidence["relations"]
                        )
                        crop_label = primary
                    if crop_patches:
                        groups = self.cropper.crop_groups(
                            plan.case_id, crop_label, groups
                        )
                    evidence_cache[groups_key] = groups
            groups = evidence_cache[groups_key]
            pathology_key = f"__pathology__:{primary}:{plan.question}"
            if pathology_key not in evidence_cache:
                evidence_cache[pathology_key] = (
                    self.pathology.describe(
                        plan.question,
                        primary,
                        groups,
                        overview_paths=overview_paths,
                    )
                    if plan.use_pathology_agent
                    else {
                        "backend": "skipped",
                        "description": "Visual expert was not requested for this field.",
                        "image_count": 0,
                        "evidence_groups": [],
                    }
                )
            pathology = evidence_cache[pathology_key]
            pathology["primary_field"] = primary
            pathology["structured_fields_covered"] = list(plan.target_phenotypes)
            pathology["retrieval_mode"] = retrieval_mode
            if use_question_retrieval:
                pathology["question_feature_source"] = (
                    QuestionFeatureStore.FEATURE_NAME
                )
                pathology["question_feature_dim"] = self.question_features.feature_dim
            if use_hybrid_retrieval:
                pathology["visual_evidence_order"] = [
                    "overview", "question_similarity", "selected_phenotype"
                ]
                pathology["question_group_count"] = sum(
                    "question_similarity" in str(group.evidence_source)
                    for group in groups
                )
                pathology["prototype_group_count"] = sum(
                    "selected_phenotype" in str(group.evidence_source)
                    for group in groups
                )
                pathology["overview_count"] = len(overview_paths)
                pathology["thumbnail_paths"] = list(overview_paths)
                image_metadata = pathology.get("image_metadata", []) or []
                pathology["overview_image_count"] = sum(
                    item.get("kind") == "overview" for item in image_metadata
                )
                pathology["question_image_count"] = sum(
                    item.get("kind") == "patch"
                    and "question_similarity" in str(
                        item.get("evidence_source")
                    )
                    for item in image_metadata
                )
                pathology["prototype_image_count"] = sum(
                    item.get("kind") == "patch"
                    and "selected_phenotype" in str(
                        item.get("evidence_source")
                    )
                    for item in image_metadata
                )
        elif evidence_route == "morphology_only":
            predictions_key = "__all_g2p_predictions__"
            if predictions_key not in evidence_cache:
                evidence_cache[predictions_key] = self._compact_broad_predictions(
                    scale_results
                )
            broad_g2p_predictions = evidence_cache[predictions_key]
            morphology_mode = self.morphology_retrieval_mode
            groups_key = (
                "__all_phenotype_groups__"
                if morphology_mode == "broad"
                else f"__question_groups__:{plan.question}"
            )
            if groups_key not in evidence_cache:
                if morphology_mode == "broad":
                    groups = self.retrieval.retrieve_all_phenotypes(scale_results)
                    crop_label = "all_phenotypes"
                else:
                    question_feature = self.question_features.lookup(plan.question)
                    groups = self.retrieval.retrieve_by_question(
                        question_feature, scale_results
                    )
                    crop_label = plan.question
                if crop_patches:
                    groups = self.cropper.crop_groups(
                        plan.case_id, crop_label, groups
                    )
                evidence_cache[groups_key] = groups
            groups = evidence_cache[groups_key]
            overview_key = "__overview_thumbnails__"
            if overview_key not in evidence_cache:
                evidence_cache[overview_key] = self.cropper.overview_thumbnails(
                    plan.case_id
                )
            overview_paths = evidence_cache[overview_key]
            pathology_key = f"__pathology_all__:{plan.question}"
            if pathology_key not in evidence_cache:
                evidence_cache[pathology_key] = self.pathology.describe(
                    plan.question,
                    "all_phenotype_attention_fallback",
                    groups,
                    overview_paths=overview_paths,
                )
            pathology = evidence_cache[pathology_key]
            pathology["primary_field"] = None
            pathology["structured_fields_covered"] = []
            pathology["retrieval_mode"] = morphology_mode
            if morphology_mode == "question_similarity":
                pathology["question_feature_source"] = (
                    QuestionFeatureStore.FEATURE_NAME
                )
                pathology["question_feature_dim"] = self.question_features.feature_dim
            pathology["thumbnail_paths"] = list(overview_paths)
        else:
            raise ValueError(f"Unsupported evidence route: {evidence_route}")

        answer, structured = self.fusion.answer_with_summary(
            plan,
            choices,
            predictions,
            relations_by_field,
            pathology,
            broad_g2p_predictions=broad_g2p_predictions,
        )
        first_prediction = predictions[0] if predictions else {}
        first_relation = relations_by_field.get(plan.target_phenotypes[0], {}) if plan.target_phenotypes else {}
        return {
            "case_id": plan.case_id,
            "question": plan.question,
            "choices": choices,
            "choice_options": indexed_choices(choices),
            "reference_answer": item.get("Answer", item.get("answer")),
            "plan": plan.to_dict(),
            "task_match": structured["task_match"],
            "evidence_route": structured["evidence_route"],
            "selected_prototype_ids": structured["selected_prototype_ids"],
            "requested_fields": structured["requested_fields"],
            "executed_fields": structured["executed_fields"],
            "missing_fields": structured["missing_fields"],
            "structured_candidate_answer": structured["structured_candidate_answer"],
            "structured_candidate_id": structured["structured_candidate_id"],
            "structured_candidate_confidence": structured["structured_candidate_confidence"],
            "structured_evidence": structured,
            "phenotype_prediction": first_prediction,
            "phenotype_predictions": predictions,
            "relation_evidence": first_relation,
            "relation_evidence_by_field": relations_by_field,
            "broad_g2p_predictions": broad_g2p_predictions or [],
            "thumbnail_paths": list(overview_paths),
            "pathology_evidence": pathology,
            "agent_answer": answer,
            "answer_in_choices": answer.get("answer") in choices,
            "raw_response": answer.get("raw_response"),
            "parse_status": answer.get("parse_status"),
            "json_parse_success": answer.get("json_parse_success", False),
            "retry_count": answer.get("retry_count", 0),
            "option_alignment": structured.get("option_alignment", {}),
            "override_occurred": answer.get("override_occurred", False),
            "override_proposed": answer.get("override_proposed", False),
            "override_rejected": answer.get("override_rejected", False),
            "override_reason": answer.get("override_reason"),
            "structured_visual_conflict": answer.get("structured_visual_conflict", False),
        }

    def _run_question_hierarchical(
        self,
        item: Dict[str, Any],
        plan: Any,
        scale_results: Dict[int, Dict[str, Any]],
        evidence_cache: Dict[str, Any],
        crop_patches: bool,
    ) -> Dict[str, Any]:
        choices = list(item.get("Choice", item.get("choices", [])) or [])
        plan_dict = plan.to_dict()
        predictions = []
        relations_by_field = {}
        for field in plan.target_phenotypes:
            if field not in evidence_cache:
                evidence_cache[field] = {
                    "phenotype": self.g2p.fuse_task(scale_results, field),
                    "relations": self.relation.reason(field, scale_results),
                }
            predictions.append(evidence_cache[field]["phenotype"])
            relations_by_field[field] = evidence_cache[field]["relations"]

        evidence_route = getattr(
            plan,
            "evidence_route",
            "phenotype_direct" if plan.target_phenotypes else "morphology_only",
        )
        broad_g2p_predictions = None
        if evidence_route == "morphology_only":
            predictions_key = "__all_g2p_predictions__"
            if predictions_key not in evidence_cache:
                evidence_cache[predictions_key] = self._compact_broad_predictions(
                    scale_results
                )
            broad_g2p_predictions = evidence_cache[predictions_key]

        knowledge = self.knowledge_rag.retrieve(
            question=plan.question,
            choices=choices,
            target_phenotypes=plan.target_phenotypes,
        )
        structured_round0 = self.fusion.prepare_structured_summary(
            plan, choices, predictions
        )
        option_alignment = dict(
            structured_round0.get("option_alignment") or {}
        )
        structured_candidate = None
        if structured_round0.get("structured_candidate_id"):
            structured_candidate = {
                "choice_id": structured_round0.get("structured_candidate_id"),
                "answer": structured_round0.get("structured_candidate_answer"),
                "confidence": structured_round0.get(
                    "structured_candidate_confidence", 0.0
                ),
            }
        memory = WorkingMemory(
            case_id=plan.case_id,
            question=plan.question,
            choices=choices,
            plan=plan_dict,
            knowledge=knowledge,
            structured_evidence=structured_round0,
            structured_candidate=structured_candidate,
            structured_confidence=float(
                structured_round0.get("structured_candidate_confidence") or 0.0
            ),
            option_alignment=option_alignment,
            direct_evidence_state=(
                "mapped"
                if structured_candidate and option_alignment.get("mapping_complete")
                else "available_unmapped"
                if structured_round0.get("available")
                else "unavailable"
            ),
        )
        program_candidates = self._program_candidates(
            plan, knowledge, relations_by_field, scale_results
        )
        gene_candidates: List[Dict[str, Any]] = []
        selected_program = None
        selected_gene = None
        verifier_decisions = []
        pathology_rounds = []
        overview_paths: List[str] = []
        post_search_abstained = False
        verifier_failure_count = 0
        evidence_sufficiency_unverified = False
        allow_early_abstain = self._early_abstain_allowed(plan_dict, knowledge)
        spatial_parent_child_trace: List[Dict[str, Any]] = []
        visual_parent_groups: List[Any] = []
        visual_parent_scale: Optional[int] = None

        round0_actions = self.verifier.available_actions(
            "round0",
            has_program_candidates=bool(program_candidates),
            has_gene_candidates=False,
            allow_early_abstain=allow_early_abstain,
        )
        round0_decision = self.verifier.decide(
            question=plan.question,
            choices=choices,
            plan=plan_dict,
            knowledge=knowledge,
            memory=memory,
            available_actions=round0_actions,
            program_candidates=program_candidates,
            gene_candidates=[],
        )
        memory.update_verifier(round0_decision)
        verifier_decisions.append(dict(round0_decision))
        if round0_decision.get("verifier_fallback_used"):
            verifier_failure_count += 1
        action = round0_decision["next_action"]
        target = round0_decision.get("target")
        action_fallback = bool(round0_decision.get("verifier_fallback_used"))
        if action == "answer":
            evidence_sufficiency_unverified = bool(
                round0_decision.get("evidence_sufficiency_unverified")
            )
        elif action == "abstain":
            post_search_abstained = True

        for round_index in range(1, 6):
            if action in {"answer", "abstain"}:
                break
            if action.startswith("inspect_") and action.split("_")[-1].isdigit():
                scale = int(action.split("_")[-1])
                groups = self._hierarchical_spatial_groups(
                    plan,
                    evidence_route,
                    scale_results,
                    evidence_cache,
                    scale,
                    crop_patches,
                )
                if visual_parent_groups and visual_parent_scale and scale < visual_parent_scale:
                    groups = self.retrieval.link_child_groups(
                        visual_parent_groups, groups
                    )
                elif not visual_parent_groups:
                    for group in groups:
                        group.anchor_group_id = group.group_id
                        group.spatial_relation = "anchor"
                spatial_parent_child_trace.extend(
                    {
                        "round": round_index,
                        "scale": scale,
                        "group_id": group.group_id,
                        "parent_group_id": group.parent_group_id,
                        "anchor_group_id": group.anchor_group_id,
                        "spatial_relation": group.spatial_relation,
                        "slide_id": next(iter(group.patches.values())).slide_id
                        if group.patches else None,
                    }
                    for group in groups
                )
                visual_parent_groups = groups
                visual_parent_scale = scale
                if scale == 4096:
                    overview_key = "__overview_thumbnails__"
                    if overview_key not in evidence_cache:
                        evidence_cache[overview_key] = self.cropper.overview_thumbnails(
                            plan.case_id
                        )
                    overview_paths = list(evidence_cache[overview_key])
                round_overviews = overview_paths if scale == 4096 else []
                observation_type = (
                    "phenotype" if plan.target_phenotypes else "morphology"
                )
                observation_role = (
                    "direct"
                    if evidence_route == "phenotype_direct" and plan.target_phenotypes
                    else "supportive"
                )
                round_target_type = observation_type
                round_target_name = (
                    plan.target_phenotypes[0] if plan.target_phenotypes else None
                )
                structured_support = {
                    "target_phenotypes": list(plan.target_phenotypes),
                    "evidence_route": evidence_route,
                    "spatial_groups": [
                        {
                            "group_id": group.group_id,
                            "parent_group_id": group.parent_group_id,
                            "anchor_group_id": group.anchor_group_id,
                            "spatial_relation": group.spatial_relation,
                        }
                        for group in groups
                    ],
                }
            elif action == "inspect_program":
                selected_program, action_fallback = self._resolve_candidate(
                    target, program_candidates, action_fallback
                )
                if selected_program is None:
                    evidence_sufficiency_unverified = True
                    break
                groups = self.retrieval.retrieve_program_1024(
                    selected_program["index"], scale_results
                )
                groups = self._crop_hierarchical_groups(
                    plan, action, selected_program["name"], groups, crop_patches
                )
                scale = 1024
                round_overviews = []
                observation_type = "program"
                observation_role = "supportive"
                round_target_type = "program"
                round_target_name = selected_program["name"]
                structured_support = self._compact_program_support(selected_program)
                gene_candidates = self._gene_candidates(
                    selected_program, knowledge, scale_results
                )
            elif action == "inspect_gene":
                selected_gene, action_fallback = self._resolve_candidate(
                    target, gene_candidates, action_fallback
                )
                if selected_gene is None or selected_program is None:
                    evidence_sufficiency_unverified = True
                    break
                if not self._gene_belongs_to_program(
                    selected_gene["index"], selected_program["index"]
                ):
                    evidence_sufficiency_unverified = True
                    break
                groups = self.retrieval.retrieve_gene_1024(
                    selected_gene["index"], scale_results
                )
                groups = self._crop_hierarchical_groups(
                    plan, action, selected_gene["name"], groups, crop_patches
                )
                scale = 1024
                round_overviews = []
                observation_type = "gene"
                observation_role = "supportive"
                round_target_type = "gene"
                round_target_name = selected_gene["name"]
                structured_support = self._compact_gene_support(
                    selected_gene, selected_program
                )
            else:
                evidence_sufficiency_unverified = True
                break

            memory.record_action(
                round_index,
                action,
                round_target_name,
                verifier_fallback_used=action_fallback,
                requested_action=(
                    round0_decision.get("requested_action")
                    if round_index == 1
                    else verifier_decisions[-1].get("requested_action")
                ),
                normalized_action=(
                    round0_decision.get("normalized_action")
                    if round_index == 1
                    else verifier_decisions[-1].get("normalized_action")
                ),
                fallback_reason=(
                    round0_decision.get("fallback_reason")
                    if round_index == 1
                    else verifier_decisions[-1].get("fallback_reason")
                ),
                target_resolution_fallback=bool(
                    (round0_decision if round_index == 1 else verifier_decisions[-1])
                    .get("target_resolution_fallback")
                ),
            )
            pathology = (
                self.pathology.describe(
                    plan.question,
                    round_target_name or observation_type,
                    groups,
                    overview_paths=round_overviews,
                    hide_provenance=True,
                )
                if plan.use_pathology_agent
                else {
                    "backend": "skipped",
                    "description": "Visual expert was not requested.",
                    "image_count": 0,
                    "evidence_groups": [group.to_dict() for group in groups],
                    "image_metadata": [],
                }
            )
            pathology["action"] = action
            pathology["scale"] = scale
            pathology_rounds.append(pathology)
            observation = EvidenceObservation(
                round_index=round_index,
                action=action,
                evidence_type=observation_type,
                evidence_role=observation_role,
                scale=scale,
                target_type=round_target_type,
                target_name=round_target_name,
                visual_description=self.fusion._visual_summary(
                    pathology.get("description")
                ),
                structured_support=structured_support,
                group_ids=[group.group_id for group in groups],
            )
            memory.add_observation(observation)

            available_actions = self.verifier.available_actions(
                action,
                has_program_candidates=bool(program_candidates),
                has_gene_candidates=bool(gene_candidates),
                allow_early_abstain=allow_early_abstain,
            )
            decision = self.verifier.decide(
                question=plan.question,
                choices=choices,
                plan=plan_dict,
                knowledge=knowledge,
                memory=memory,
                available_actions=available_actions,
                program_candidates=program_candidates,
                gene_candidates=gene_candidates,
            )
            memory.update_verifier(decision)
            verifier_decisions.append(dict(decision))
            verifier_fallback_used = bool(decision.get("verifier_fallback_used"))
            if verifier_fallback_used:
                verifier_failure_count += 1
            next_action = decision["next_action"]
            if next_action == "answer":
                evidence_sufficiency_unverified = verifier_fallback_used
                break
            if next_action == "abstain":
                post_search_abstained = True
                break
            action = next_action
            target = decision.get("target")
            action_fallback = bool(decision.get("verifier_fallback_used"))
        else:
            evidence_sufficiency_unverified = True

        agent_context = self._hierarchical_fusion_context(
            memory, selected_program, selected_gene
        )
        combined_pathology = {
            "backend": "incremental_pathology",
            "description": "\n".join(
                f"{row.action}: {row.visual_description}"
                for row in memory.observations
            ),
            "rounds": pathology_rounds,
        }
        if post_search_abstained:
            structured = structured_round0
            answer = None
        else:
            answer, structured = self.fusion.answer_with_summary(
                plan,
                choices,
                predictions,
                relations_by_field,
                combined_pathology,
                broad_g2p_predictions=broad_g2p_predictions,
                agent_context=agent_context,
                prepared_structured=structured_round0,
            )

        first_prediction = predictions[0] if predictions else {}
        first_relation = (
            relations_by_field.get(plan.target_phenotypes[0], {})
            if plan.target_phenotypes else {}
        )
        if evidence_sufficiency_unverified:
            final_evidence_state = "unverified"
        elif memory.final_verifier:
            final_evidence_state = memory.final_verifier.get("evidence_state")
        else:
            final_evidence_state = "unavailable"
        agent_trace = {
            "round0_structured_evidence": structured_round0,
            "round0_decision": round0_decision,
            "structured_candidate_before_visual": structured_candidate,
            "round_count": len(memory.observations),
            "actions": list(memory.action_history),
            "inspected_scales": list(memory.inspected_scales),
            "program_candidates": program_candidates,
            "selected_program": selected_program,
            "gene_candidates": gene_candidates,
            "selected_gene": selected_gene,
            "verifier_decisions": verifier_decisions,
            "final_evidence_state": final_evidence_state,
            "post_search_abstained": post_search_abstained,
            "verifier_failure_count": verifier_failure_count,
            "verifier_fallback_count": verifier_failure_count,
            "evidence_sufficiency_unverified": evidence_sufficiency_unverified,
            "early_abstain_allowed": allow_early_abstain,
            "spatial_parent_child_trace": spatial_parent_child_trace,
            "visual_observations": [
                row.to_dict() for row in memory.observations
                if row.evidence_type in {"phenotype", "morphology"}
            ],
            "final_answer": answer or {},
        }
        result = {
            "case_id": plan.case_id,
            "question": plan.question,
            "choices": choices,
            "choice_options": indexed_choices(choices),
            "reference_answer": item.get("Answer", item.get("answer")),
            "plan": plan_dict,
            "agent_mode": "hierarchical_rag",
            "task_match": structured.get("task_match", plan.task_match),
            "evidence_route": structured.get("evidence_route", evidence_route),
            "selected_prototype_ids": structured.get(
                "selected_prototype_ids", list(plan.selected_prototype_ids)
            ),
            "requested_fields": structured.get(
                "requested_fields", list(plan.target_phenotypes)
            ),
            "executed_fields": structured.get(
                "executed_fields", list(plan.target_phenotypes)
            ),
            "missing_fields": structured.get("missing_fields", []),
            "structured_candidate_answer": structured.get(
                "structured_candidate_answer"
            ),
            "structured_candidate_id": structured.get("structured_candidate_id"),
            "structured_candidate_confidence": structured.get(
                "structured_candidate_confidence", 0.0
            ),
            "structured_evidence": structured,
            "phenotype_prediction": first_prediction,
            "phenotype_predictions": predictions,
            "relation_evidence": first_relation,
            "relation_evidence_by_field": relations_by_field,
            "broad_g2p_predictions": broad_g2p_predictions or [],
            "thumbnail_paths": list(overview_paths),
            "pathology_evidence": combined_pathology,
            "knowledge_rag": knowledge,
            "working_memory": memory.to_dict(),
            "agent_trace": agent_trace,
            "post_search_abstained": post_search_abstained,
            "evidence_sufficiency_unverified": evidence_sufficiency_unverified,
            "verifier_failure_count": verifier_failure_count,
            "abstain_stage": (
                "evidence_sufficiency" if post_search_abstained else None
            ),
            "abstained": post_search_abstained,
            "agent_answer": answer,
            "answer_in_choices": bool(
                answer and answer.get("answer") in choices
            ),
            "raw_response": answer.get("raw_response") if answer else None,
            "parse_status": answer.get("parse_status") if answer else "post_search_abstain",
            "json_parse_success": answer.get(
                "json_parse_success", False
            ) if answer else False,
            "retry_count": answer.get("retry_count", 0) if answer else 0,
            "option_alignment": structured.get("option_alignment", {}),
            "override_occurred": answer.get(
                "override_occurred", False
            ) if answer else False,
            "override_proposed": answer.get(
                "override_proposed", False
            ) if answer else False,
            "override_rejected": answer.get(
                "override_rejected", False
            ) if answer else False,
            "override_reason": answer.get("override_reason") if answer else None,
            "structured_visual_conflict": answer.get(
                "structured_visual_conflict", False
            ) if answer else False,
        }
        return result

    def _hierarchical_spatial_groups(
        self,
        plan: Any,
        evidence_route: str,
        scale_results: Dict[int, Dict[str, Any]],
        evidence_cache: Dict[str, Any],
        scale: int,
        crop_patches: bool,
    ) -> List[Any]:
        primary = plan.target_phenotypes[0] if plan.target_phenotypes else None
        if evidence_route == "phenotype_direct" and primary:
            retrieval_mode = (
                self.direct_retrieval_mode
                if plan.task_match == "direct"
                else self.partial_retrieval_mode
            )
        elif evidence_route == "morphology_only":
            retrieval_mode = self.morphology_retrieval_mode
        else:
            raise ValueError(f"Unsupported evidence route: {evidence_route}")

        key = (
            f"__hierarchical_spatial__:{retrieval_mode}:{primary}:"
            f"{plan.question}:{scale}"
        )
        if key in evidence_cache:
            return evidence_cache[key]

        if evidence_route == "phenotype_direct" and primary:
            if retrieval_mode == "selected_phenotype":
                groups = self.retrieval.retrieve_phenotype_only(
                    primary, scale_results, scale
                )
            elif retrieval_mode == "question_similarity":
                groups = self.retrieval.retrieve_question_scale(
                    self.question_features.lookup(plan.question),
                    scale_results,
                    scale,
                )
            elif retrieval_mode == "hybrid_question_prototype":
                question_groups = self.retrieval.retrieve_question_scale(
                    self.question_features.lookup(plan.question),
                    scale_results,
                    scale,
                )
                phenotype_groups = self.retrieval.retrieve_phenotype_only(
                    primary, scale_results, scale
                )
                groups = self.retrieval.merge_hybrid_scale_groups(
                    question_groups, phenotype_groups
                )
            else:
                raise ValueError(f"Unsupported retrieval mode: {retrieval_mode}")
        elif retrieval_mode == "question_similarity":
            groups = self.retrieval.retrieve_question_scale(
                self.question_features.lookup(plan.question),
                scale_results,
                scale,
            )
        else:
            groups = self.retrieval.retrieve_all_phenotypes_scale(
                scale_results, scale
            )

        self._assert_spatial_sources(groups)
        groups = self._crop_hierarchical_groups(
            plan, f"inspect_{scale}", primary or plan.question, groups, crop_patches
        )
        evidence_cache[key] = groups
        return groups

    def _crop_hierarchical_groups(
        self,
        plan: Any,
        action: str,
        target: str,
        groups: List[Any],
        crop_patches: bool,
    ) -> List[Any]:
        if not crop_patches:
            return groups
        return self.cropper.crop_groups(
            plan.case_id, f"hierarchical:{action}:{target}:{plan.question}", groups
        )

    @staticmethod
    def _assert_spatial_sources(groups: List[Any]) -> None:
        invalid = []
        for group in groups:
            for patch in group.patches.values():
                invalid.extend(
                    source.get("type") for source in patch.sources
                    if source.get("type") in {"program", "gene"}
                )
        if invalid:
            raise RuntimeError(
                "Hierarchical spatial retrieval leaked program/gene attention"
            )

    @staticmethod
    def _early_abstain_allowed(
        plan: Dict[str, Any], knowledge: Dict[str, Any]
    ) -> bool:
        """Use explicit Planner/RAG evidence semantics, never question keywords."""
        if plan.get("target_phenotypes"):
            return False
        if plan.get("local_morphology_useful") is True:
            return False
        if plan.get("requires_unavailable_context") is True:
            return True
        unavailable_roles = {
            "limited_context",
            "unavailable",
            "invalid_evidence",
            "unavailable_from_local_visual_evidence",
        }
        if any(
            str(row.get("evidence_role", "")).lower() in unavailable_roles
            for row in knowledge.get("matched_concepts", [])
        ):
            return True
        unavailable_rule_ids = {
            "rule_assay_specific_target",
            "rule_exact_quantity",
            "rule_stage_and_outcome",
        }
        return any(
            row.get("id") in unavailable_rule_ids
            for row in knowledge.get("evidence_rules", [])
        )

    def _program_candidates(
        self,
        plan: Any,
        knowledge: Dict[str, Any],
        relations_by_field: Dict[str, Any],
        scale_results: Dict[int, Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        knowledge_scores = {
            row["name"]: float(row.get("relevance") or 0.0)
            for row in knowledge.get("candidate_programs", [])
        }
        if plan.target_phenotypes:
            primary = plan.target_phenotypes[0]
            rows = []
            for relation in relations_by_field.get(primary, {}).get("programs", [])[:3]:
                row = dict(relation)
                row["knowledge_relevance"] = knowledge_scores.get(row["name"], 0.0)
                row["evidence_role"] = "supportive"
                rows.append(row)
            return rows

        scales = sorted(scale_results)
        rows = []
        for candidate in knowledge.get("candidate_programs", []):
            name = candidate.get("name")
            if name not in self.registry.programs:
                continue
            index = self.registry.programs.index(name)
            relevance = float(candidate.get("relevance") or 0.0)
            per_scale = {
                str(scale): float(
                    np.asarray(scale_results[scale]["program_pred"], dtype=float)[index]
                )
                for scale in scales
            }
            signed_scores = np.asarray(list(per_scale.values()), dtype=float)
            activities = np.abs(signed_scores)
            patient_score = float(signed_scores.mean())
            patient_activity = float(activities.mean())
            activity_consensus = float(np.clip(
                1.0 - activities.std() / (patient_activity + 1e-6),
                0.0,
                1.0,
            ))
            rows.append({
                "index": index,
                "name": name,
                "score": (
                    relevance
                    * (1.0 + min(patient_activity, 3.0) / 3.0)
                    * (0.5 + 0.5 * activity_consensus)
                ),
                "knowledge_relevance": relevance,
                "patient_score": patient_score,
                "patient_activity": patient_activity,
                "scale_consensus": activity_consensus,
                "per_scale_patient_score": per_scale,
                "evidence_role": "supportive",
                "selection_policy": (
                    "knowledge_relevance_plus_multiscale_patient_activity_consensus"
                ),
            })
        rows.sort(key=lambda row: (-row["score"], row["name"]))
        return rows[:3]

    def _gene_candidates(
        self,
        program: Dict[str, Any],
        knowledge: Dict[str, Any],
        scale_results: Dict[int, Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        program_index = int(program["index"])
        ranked = self.relation._rank_genes(
            program_index,
            scale_results,
            float(program.get("score") or 1.0),
        )
        knowledge_names = {
            row.get("name") for row in knowledge.get("candidate_genes", [])
            if any(
                source == f"program:{program['name']}"
                for source in row.get("sources", [])
            )
        }
        rows = []
        for row in ranked:
            if not self._gene_belongs_to_program(row["index"], program_index):
                continue
            value = dict(row)
            value["knowledge_candidate"] = value["name"] in knowledge_names
            value["evidence_role"] = "supportive"
            value["h_membership_validated"] = True
            rows.append(value)
        rows.sort(
            key=lambda row: (
                -int(row["knowledge_candidate"]), -float(row["score"]), row["name"]
            )
        )
        return rows[:5]

    def _gene_belongs_to_program(
        self, gene_index: int, program_index: int
    ) -> bool:
        memberships = [
            float(runtime.relations["H_gene_to_program"][gene_index, program_index])
            for runtime in self.g2p.runtimes.values()
        ]
        return bool(memberships) and all(value > 0.0 for value in memberships)

    @staticmethod
    def _resolve_candidate(
        target: Optional[str],
        candidates: List[Dict[str, Any]],
        fallback_used: bool,
    ) -> Any:
        selected = next(
            (row for row in candidates if str(row.get("name")) == str(target)),
            None,
        )
        if selected is not None:
            return selected, fallback_used
        if candidates:
            return candidates[0], True
        return None, True

    @staticmethod
    def _compact_program_support(program: Dict[str, Any]) -> Dict[str, Any]:
        return {
            key: program.get(key) for key in (
                "index", "name", "score", "prior", "initial", "learned",
                "change", "gate", "scale_consensus", "relation_type",
                "patient_score", "knowledge_relevance", "evidence_role",
            ) if key in program
        }

    @staticmethod
    def _compact_gene_support(
        gene: Dict[str, Any], program: Dict[str, Any]
    ) -> Dict[str, Any]:
        return {
            "index": gene.get("index"),
            "name": gene.get("name"),
            "score": gene.get("score"),
            "patient_score": gene.get("patient_score"),
            "gene_to_program": gene.get("gene_to_program"),
            "scale_consensus": gene.get("scale_consensus"),
            "program": program.get("name"),
            "h_membership_validated": gene.get("h_membership_validated", False),
            "evidence_role": "supportive",
        }

    @staticmethod
    def _hierarchical_fusion_context(
        memory: WorkingMemory,
        selected_program: Optional[Dict[str, Any]],
        selected_gene: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        biological = []
        for observation in memory.observations:
            if observation.evidence_type not in {"program", "gene"}:
                continue
            biological.append({
                "type": observation.evidence_type,
                "name": observation.target_name,
                "scale": observation.scale,
                "visual_description": observation.visual_description,
                "evidence_role": "supportive",
                "structured_support": observation.structured_support,
            })
        return {
            "knowledge": {
                "matched_concepts": memory.knowledge.get("matched_concepts", []),
                "limitations": memory.knowledge.get("limitations", []),
                "evidence_rules": memory.knowledge.get("evidence_rules", []),
            },
            "visual_observations": [
                {
                    "action": row.action,
                    "scale": row.scale,
                    "description": row.visual_description,
                }
                for row in memory.observations
            ],
            "supportive_biological_evidence": biological,
            "selected_program": (
                MultiScaleVQAPipeline._compact_program_support(selected_program)
                if selected_program else None
            ),
            "selected_gene": (
                MultiScaleVQAPipeline._compact_gene_support(
                    selected_gene, selected_program
                ) if selected_gene and selected_program else None
            ),
            "limitations": list(dict.fromkeys([
                *memory.knowledge.get("limitations", []),
                "Program and gene evidence is WSI-derived supportive evidence, not a measured assay.",
                "Learned program-to-phenotype relations are predictive associations, not causality.",
            ])),
        }

    def _compact_broad_predictions(
        self, scale_results: Dict[int, Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        rows = []
        for field in self.registry.phenotype_fields:
            prediction = self.g2p.fuse_task(scale_results, field)
            fused = prediction.get("fused", {})
            row = {
                "prototype_id": self.registry.field_to_prototype_id[field],
                "field": field,
                "name": self.registry.field_to_name[field],
                "confidence": None,
            }
            if "probability" in fused:
                probability = float(fused["probability"])
                predicted_class = int(fused["predicted_class"])
                row.update({
                    "predicted_class": predicted_class,
                    "predicted_label": fused.get("predicted_label"),
                    "positive_probability": probability,
                    "confidence": (
                        probability if predicted_class == 1 else 1.0 - probability
                    ),
                })
            elif "probabilities" in fused:
                probabilities = [float(value) for value in fused["probabilities"]]
                row.update({
                    "predicted_class": int(fused["predicted_class"]),
                    "predicted_label": fused.get("predicted_label"),
                    "probabilities": probabilities,
                    "confidence": max(probabilities) if probabilities else None,
                })
            elif "risk" in fused:
                row["risk"] = float(fused["risk"])
            elif "value" in fused:
                row["value"] = float(fused["value"])
            rows.append(row)
        return rows

    @staticmethod
    def _completed_keys(path: Path) -> set:
        if not path.exists():
            return set()
        result = set()
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    item = json.loads(line)
                    result.add((str(item.get("case_id", "")), str(item.get("question", ""))))
                except json.JSONDecodeError:
                    continue
        return result
