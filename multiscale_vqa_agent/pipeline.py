import gc
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import torch

from .clients import OpenAICompatibleClient
from .fusion import FusionVerificationAgent
from .fusion_evidence import build_structured_summary
from .g2p_runtime import MultiScaleG2PAgent
from .pathology import PathologyAgent
from .registry import PrototypeAwarePlanner, ToolBankRegistry
from .relation import RelationReasoningAgent
from .retrieval import MultiScaleRetrievalAgent, WSICropper


class MultiScaleVQAPipeline:
    def __init__(self, config_path: str, planner_only: bool = False):
        self.config_path = Path(config_path)
        with self.config_path.open(encoding="utf-8") as handle:
            self.config = json.load(handle)
        scale_dirs = {int(k): Path(v) for k, v in self.config["scales"].items()}
        self.registry = ToolBankRegistry(scale_dirs)
        self.qwen = OpenAICompatibleClient(self.config["qwen"])
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
    ) -> Path:
        source = Path(vqa_path or self.config["vqa_json"])
        with source.open(encoding="utf-8") as handle:
            items = json.load(handle)
        if limit is not None:
            items = items[:limit]
        plans = [(item, self.planner.plan(item)) for item in items]
        destination = Path(output_path or (Path(self.config["output_dir"]) / "answers.jsonl"))
        destination.parent.mkdir(parents=True, exist_ok=True)
        completed = self._completed_keys(destination) if resume else set()
        mode = "a" if resume and destination.exists() else "w"
        if self.planner_only:
            with destination.open(mode, encoding="utf-8") as handle:
                for item, plan in plans:
                    key = (plan.case_id, plan.question)
                    if key in completed:
                        continue
                    handle.write(json.dumps({"input": item, "plan": plan.to_dict()}, ensure_ascii=False) + "\n")
            return destination
        grouped: Dict[str, List[Any]] = defaultdict(list)
        for item, plan in plans:
            if (plan.case_id, plan.question) not in completed:
                grouped[plan.case_id].append((item, plan))
        with destination.open(mode, encoding="utf-8") as handle:
            for case_number, (case_id, case_items) in enumerate(grouped.items(), 1):
                print(f"[{case_number}/{len(grouped)}] infer {case_id} ({len(case_items)} questions)", flush=True)
                try:
                    scale_results = self.g2p.infer_case(case_id)
                    evidence_cache = {}
                    for item, plan in case_items:
                        result = self._run_question(item, plan, scale_results, evidence_cache, crop_patches)
                        handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                        handle.flush()
                except Exception as error:
                    for item, plan in case_items:
                        handle.write(json.dumps({
                            "case_id": case_id,
                            "question": plan.question,
                            "input": item,
                            "plan": plan.to_dict(),
                            "error": f"{type(error).__name__}: {error}",
                        }, ensure_ascii=False) + "\n")
                    handle.flush()
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        return destination

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

        if plan.target_phenotypes:
            primary = plan.target_phenotypes[0]
            primary_evidence = evidence_cache[primary]
            pathology_key = f"__pathology__:{primary}"
            if pathology_key not in evidence_cache:
                name = self.registry.field_to_name[primary]
                vocab = self.registry.vocabs[min(self.registry.vocabs)]
                group = vocab.get("phenotype_groups", {}).get(name, "morphology")
                groups = self.retrieval.retrieve(
                    primary, group, scale_results, primary_evidence["relations"]
                )
                if crop_patches:
                    groups = self.cropper.crop_groups(plan.case_id, primary, groups)
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
        else:
            pathology = {
                "backend": "unavailable",
                "description": "No relevant trained phenotype field or field-specific visual evidence is available.",
                "image_count": 0,
                "primary_field": None,
                "structured_fields_covered": [],
            }

        answer = self.fusion.answer(
            plan, choices, predictions, relations_by_field, pathology
        )
        structured = build_structured_summary(plan, choices, predictions)
        first_prediction = predictions[0] if predictions else {}
        first_relation = relations_by_field.get(plan.target_phenotypes[0], {}) if plan.target_phenotypes else {}
        return {
            "case_id": plan.case_id,
            "question": plan.question,
            "choices": choices,
            "reference_answer": item.get("Answer", item.get("answer")),
            "plan": plan.to_dict(),
            "task_match": structured["task_match"],
            "requested_fields": structured["requested_fields"],
            "executed_fields": structured["executed_fields"],
            "missing_fields": structured["missing_fields"],
            "structured_candidate_answer": structured["structured_candidate_answer"],
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
            "override_occurred": answer.get("override_occurred", False),
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
