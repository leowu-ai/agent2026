import gc
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import torch

from .answerability import AnswerabilityAgent
from .clients import OpenAICompatibleClient
from .fusion import FusionVerificationAgent
from .fusion_evidence import indexed_choices
from .g2p_runtime import MultiScaleG2PAgent
from .pathology import PathologyAgent
from .precomputed_answerability import PrecomputedAnswerabilityStore
from .question_features import QuestionFeatureStore
from .registry import PrototypeAwarePlanner, ToolBankRegistry
from .relation import RelationReasoningAgent
from .retrieval import MultiScaleRetrievalAgent, WSICropper
from .router_audit import write_router_audit


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
            "selected_phenotype", "question_similarity"
        }:
            raise ValueError(
                "partial_retrieval_mode must be selected_phenotype or "
                f"question_similarity, got {self.partial_retrieval_mode!r}"
            )
        self.direct_retrieval_mode = str(
            direct_retrieval_mode
            or self.config["retrieval"].get(
                "direct_retrieval_mode", "selected_phenotype"
            )
        )
        if self.direct_retrieval_mode not in {
            "selected_phenotype", "question_similarity"
        }:
            raise ValueError(
                "direct_retrieval_mode must be selected_phenotype or "
                f"question_similarity, got {self.direct_retrieval_mode!r}"
            )
        self.question_features = None
        if (
            self.morphology_retrieval_mode == "question_similarity"
            or self.partial_retrieval_mode == "question_similarity"
            or self.direct_retrieval_mode == "question_similarity"
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
            "abstained": False,
        })
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
            question_retrieval_route = None
            if (
                plan.task_match == "direct"
                and self.direct_retrieval_mode == "question_similarity"
            ):
                question_retrieval_route = "direct"
            elif (
                plan.task_match == "partial"
                and self.partial_retrieval_mode == "question_similarity"
            ):
                question_retrieval_route = "partial"
            use_question_retrieval = question_retrieval_route is not None
            groups_key = (
                f"__{question_retrieval_route}_question_groups__:{plan.question}"
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
                        primary, group, scale_results, primary_evidence["relations"]
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
                    self.pathology.describe(plan.question, primary, groups)
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
            pathology["retrieval_mode"] = (
                "question_similarity"
                if use_question_retrieval
                else "selected_phenotype"
            )
            if use_question_retrieval:
                pathology["question_feature_source"] = (
                    QuestionFeatureStore.FEATURE_NAME
                )
                pathology["question_feature_dim"] = self.question_features.feature_dim
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
