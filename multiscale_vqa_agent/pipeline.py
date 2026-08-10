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
from .registry import PrototypeAwarePlanner, ToolBankRegistry
from .relation import RelationReasoningAgent
from .retrieval import MultiScaleRetrievalAgent, WSICropper
from .router_audit import write_router_audit


class MultiScaleVQAPipeline:
    def __init__(self, config_path: str, planner_only: bool = False):
        self.config_path = Path(config_path)
        with self.config_path.open(encoding="utf-8") as handle:
            self.config = json.load(handle)
        scale_dirs = {int(k): Path(v) for k, v in self.config["scales"].items()}
        self.registry = ToolBankRegistry(scale_dirs)
        self.qwen = OpenAICompatibleClient(self.config["qwen"])
        self.answerability = AnswerabilityAgent(self.qwen)
        self.planner = PrototypeAwarePlanner(self.registry, self.qwen)
        self.planner_only = planner_only
        if planner_only:
            return
        self.g2p = MultiScaleG2PAgent(self.config, self.registry)
        self.relation = RelationReasoningAgent(self.registry, self.g2p, self.config["retrieval"])
        self.retrieval = MultiScaleRetrievalAgent(self.registry, self.config["retrieval"])
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

        if self.planner_only:
            plans = []
            with destination.open(mode, encoding="utf-8") as handle:
                for case_items in grouped.values():
                    for item in case_items:
                        assessment = self._predict_answerability(item)
                        if assessment["answerability"] == "unanswerable":
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
                    if assessment["answerability"] == "unanswerable":
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
        _, question = self._item_key(item)
        choices = list(item.get("Choice", item.get("choices", [])) or [])
        return self.answerability.predict(question, choices)

    @staticmethod
    def _attach_answerability(
        result: Dict[str, Any], assessment: Dict[str, Any]
    ) -> Dict[str, Any]:
        result.update({
            "predicted_answerability": assessment["answerability"],
            "answerability_confidence": assessment["confidence"],
            "answerability_reason": assessment["reason"],
            "abstained": False,
        })
        return result

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
            "predicted_answerability": assessment["answerability"],
            "answerability_confidence": assessment["confidence"],
            "answerability_reason": assessment["reason"],
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
        evidence_cache: Dict[str, Dict[str, Any]],
        crop_patches: bool,
    ) -> Dict[str, Any]:
        choices = list(item.get("Choice", item.get("choices", [])) or [])
        predictions = []
        relations_by_field = {}
        pathology_by_field = {}

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
            "phenotype_direct" if plan.target_phenotypes else "nonvisual",
        )
        if evidence_route == "phenotype_direct" and plan.target_phenotypes:
            primary = plan.target_phenotypes[0]
            primary_evidence = evidence_cache[primary]
            groups_key = f"__pathology_groups__:{primary}"
            if groups_key not in evidence_cache:
                name = self.registry.field_to_name[primary]
                vocab = self.registry.vocabs[min(self.registry.vocabs)]
                group = vocab.get("phenotype_groups", {}).get(name, "morphology")
                groups = self.retrieval.retrieve(
                    primary, group, scale_results, primary_evidence["relations"]
                )
                if crop_patches:
                    groups = self.cropper.crop_groups(plan.case_id, primary, groups)
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
            pathology["retrieval_mode"] = "selected_phenotype"
        elif evidence_route == "morphology_only":
            groups_key = "__all_phenotype_groups__"
            if groups_key not in evidence_cache:
                groups = self.retrieval.retrieve_all_phenotypes(scale_results)
                if crop_patches:
                    groups = self.cropper.crop_groups(
                        plan.case_id, "all_phenotypes", groups
                    )
                evidence_cache[groups_key] = groups
            groups = evidence_cache[groups_key]
            pathology_key = f"__pathology_all__:{plan.question}"
            if pathology_key not in evidence_cache:
                evidence_cache[pathology_key] = self.pathology.describe(
                    plan.question,
                    "all_phenotype_attention_fallback",
                    groups,
                )
            pathology = evidence_cache[pathology_key]
            pathology["primary_field"] = None
            pathology["structured_fields_covered"] = []
            pathology["retrieval_mode"] = "all_phenotype_attention"
        else:
            pathology = {
                "backend": "unavailable",
                "description": "The question requires non-visual information that WSI patches cannot establish.",
                "image_count": 0,
                "primary_field": None,
                "structured_fields_covered": [],
                "retrieval_mode": "nonvisual",
            }

        answer, structured = self.fusion.answer_with_summary(
            plan, choices, predictions, relations_by_field, pathology
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
