import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

from .clients import OpenAICompatibleClient, parse_json_response
from .schemas import ExecutionPlan


TASK_RULES = [
    ("HER2_status_label", ("her2", "erbb2")),
    ("ER_status_label", ("estrogen receptor", "er status", "er positive", "er negative")),
    ("PR_status_label", ("progesterone receptor", "pr status", "pr positive", "pr negative")),
    ("histologic_grade_label", ("histologic grade", "histological grade", "tumor grade", "nuclear grade", "分级")),
    ("nottingham_total_score", ("nottingham",)),
    ("mitotic_score", ("mitotic", "mitosis", "有丝分裂")),
    ("ajcc_pathologic_stage", ("ajcc", "pathologic stage", "pathological stage", "tumor stage", "分期")),
    ("lymphovascular_invasion_label", ("lymphovascular", "vascular invasion", "脉管侵犯")),
    ("comedonecrosis_binary", ("comedonecrosis", "comedo necrosis", "粉刺样坏死")),
    ("necrosis_binary", ("necrosis", "necrotic", "坏死")),
    ("microcalcification_binary", ("microcalcification", "calcification", "钙化")),
    ("dcis_binary", ("dcis", "ductal carcinoma in situ", "in-situ component", "in situ component")),
    ("lcis_binary", ("lcis", "lobular carcinoma in situ")),
    ("lobular_binary", ("lobular", "小叶")),
    ("ductal_binary", ("ductal", "导管")),
    ("OS", ("survival", "prognosis", "vital status", "death", "alive", "生存", "预后")),
    ("histological_type_label", ("diagnosis", "histological type", "histologic type", "histological_type", "histologic_type", "carcinoma type", "type of breast cancer", "诊断")),
]

UNSUPPORTED_HINTS = (
    "tumor size", "surgical margin", "margin status", "treatment", "chemotherapy",
    "radiotherapy", "radiation therapy", "exact survival", "survival time", "how many days",
)


class ToolBankRegistry:
    def __init__(self, scale_dirs: Dict[int, Path]):
        self.scale_dirs = {int(k): Path(v) for k, v in scale_dirs.items()}
        self.vocabs: Dict[int, Dict[str, Any]] = {}
        self.tools: Dict[int, Dict[str, Any]] = {}
        self.metrics: Dict[int, Dict[str, Dict[str, Any]]] = {}
        for scale, directory in self.scale_dirs.items():
            self.vocabs[scale] = self._read_json(directory / "vocab.json")
            self.tools[scale] = self._read_json(directory / "tool_registry.json")
            self.metrics[scale] = self._read_metrics(directory / "tool_metrics.csv")
        self._validate_vocabularies()
        first = self.vocabs[min(self.vocabs)]
        self.genes = first["gene_list"]
        self.programs = first["program_names"]
        self.phenotype_names = first["phenotype_names"]
        self.phenotype_fields = first["phenotype_fields"]
        self.field_to_index = {name: i for i, name in enumerate(self.phenotype_fields)}
        self.field_to_name = dict(zip(self.phenotype_fields, self.phenotype_names))
        self.label_encoders = first.get("label_encoders", {})

    @staticmethod
    def _read_json(path: Path) -> Dict[str, Any]:
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)

    @staticmethod
    def _read_metrics(path: Path) -> Dict[str, Dict[str, Any]]:
        if not path.exists():
            return {}
        frame = pd.read_csv(path)
        result = {}
        for row in frame.to_dict("records"):
            key = str(row.get("task_name", row.get("phenotype_field", "")))
            if key:
                result[key] = row
        return result

    def _validate_vocabularies(self):
        ordered = [self.vocabs[s] for s in sorted(self.vocabs)]
        for key in ("gene_list", "program_names", "phenotype_fields"):
            if any(item[key] != ordered[0][key] for item in ordered[1:]):
                raise ValueError(f"Scale ToolBanks disagree on {key}")

    def compact_catalog(self) -> List[Dict[str, Any]]:
        vocab = self.vocabs[min(self.vocabs)]
        task_types = vocab.get("phenotype_task_types", {})
        groups = vocab.get("phenotype_groups", {})
        return [
            {
                "field": field,
                "name": name,
                "task_type": task_types.get(name, "unknown"),
                "group": groups.get(name, "unknown"),
            }
            for field, name in zip(self.phenotype_fields, self.phenotype_names)
        ]

    def task_metrics(self, field: str) -> Dict[str, Any]:
        values = {}
        for scale, metrics in self.metrics.items():
            match = next((v for k, v in metrics.items() if field in k), None)
            if match:
                values[str(scale)] = match
        return values

    def label_semantics(self, field: str) -> Dict[str, Any]:
        name = self.field_to_name[field]
        encoder = self.label_encoders.get(name, {})
        class_to_label = {str(index): str(label) for label, index in encoder.items()}
        if not class_to_label and field.endswith("_binary"):
            class_to_label = {"0": "no", "1": "yes"}
        clinical_meaning = dict(class_to_label)
        if field == "histologic_grade_label":
            clinical_meaning = {"0": "grade 1 / low", "1": "grade 2 / intermediate", "2": "grade 3 / high"}
        return {
            "field": field,
            "predicted_class_is_zero_based": True,
            "class_to_label": class_to_label,
            "clinical_meaning": clinical_meaning,
            "positive_class": 1 if field.endswith("_binary") else None,
            "negative_class": 0 if field.endswith("_binary") else None,
        }


class PrototypeAwarePlanner:
    def __init__(self, registry: ToolBankRegistry, qwen: OpenAICompatibleClient):
        self.registry = registry
        self.qwen = qwen

    def plan(self, item: Dict[str, Any]) -> ExecutionPlan:
        question = str(item.get("Question", item.get("question", "")))
        case_id = str(item.get("Id", item.get("case_id", "")))[:12]
        choices = item.get("Choice", item.get("choices", [])) or []
        llm_plan = self._llm_plan(case_id, question, choices)
        if llm_plan:
            return llm_plan
        return self._rule_plan(case_id, question, choices)

    def _llm_plan(self, case_id: str, question: str, choices: Iterable[str]) -> Optional[ExecutionPlan]:
        if not self.qwen.enabled:
            return None
        system = (
            "You are the prototype-aware planner for a breast pathology VQA system. "
            "Select only fields in the supplied catalog. Return one JSON object with keys "
            "target_phenotypes, task_type, metrics, supported, support_reason, task_match, phenotype_relevance_score, use_pathology_agent. "
            "WSI cannot reliably provide treatment, surgical margin, exact tumor size, or exact survival days."
        )
        user = json.dumps({
            "question": question,
            "choices": list(choices),
            "available_phenotypes": self.registry.compact_catalog(),
        }, ensure_ascii=False)
        try:
            parsed = parse_json_response(self.qwen.chat(system, user, max_tokens=512))
        except Exception as error:
            print(f"Planner Qwen unavailable, using rules: {error}", flush=True)
            return None
        if not parsed:
            return None
        raw_fields = parsed.get("target_phenotypes", [])
        if isinstance(raw_fields, (str, dict)):
            raw_fields = [raw_fields]
        fields = []
        for value in raw_fields if isinstance(raw_fields, list) else []:
            if isinstance(value, dict):
                value = value.get("field", value.get("phenotype_field", value.get("name")))
            if isinstance(value, str) and value in self.registry.field_to_index:
                fields.append(value)
        fields = list(dict.fromkeys(fields))
        if not fields and parsed.get("supported", False):
            rule_plan = self._rule_plan(case_id, question, choices)
            fields = rule_plan.target_phenotypes

        supported_value = parsed.get("supported", bool(fields))
        if isinstance(supported_value, str):
            supported_value = supported_value.strip().lower() in {"true", "1", "yes"}
        supported = bool(supported_value) and bool(fields)
        task_match = str(parsed.get("task_match", "direct" if supported else "none")).lower()
        if task_match not in {"direct", "partial", "indirect", "none"}:
            task_match = "direct" if supported else "none"
        if not supported:
            task_match = "none"

        default_metrics = self._metrics_for(fields[0]) if fields else []
        metrics = parsed.get("metrics", default_metrics)
        if isinstance(metrics, str):
            metrics = [metrics]
        elif not isinstance(metrics, list):
            metrics = default_metrics
        default_relevance = {"direct": 1.0, "partial": 0.6, "indirect": 0.3, "none": 0.0}[task_match]
        try:
            relevance = float(parsed.get("phenotype_relevance_score", default_relevance))
        except (TypeError, ValueError):
            relevance = default_relevance
        relevance = max(0.0, min(relevance, 1.0))

        return ExecutionPlan(
            case_id=case_id,
            question=question,
            target_phenotypes=fields,
            task_type=str(parsed.get("task_type", "unknown")),
            metrics=metrics,
            answer_mode="multiple_choice" if choices else "open",
            supported=supported,
            support_reason=str(parsed.get("support_reason", "")),
            task_match=task_match,
            phenotype_relevance_score=relevance,
            use_pathology_agent=bool(parsed.get("use_pathology_agent", True)),
        )

    def _rule_plan(self, case_id: str, question: str, choices: Iterable[str]) -> ExecutionPlan:
        lowered = question.lower()
        unsupported = any(term in lowered for term in UNSUPPORTED_HINTS)
        fields = [field for field, terms in TASK_RULES if any(term in lowered for term in terms)]
        generic_in_situ = ("any" in lowered and ("in situ" in lowered or "in-situ" in lowered)) or "in situ component" in lowered or "in-situ component" in lowered
        if generic_in_situ:
            fields.extend(["dcis_binary", "lcis_binary"])
        fields = list(dict.fromkeys(fields))
        unsupported = unsupported or not fields
        vocab = self.registry.vocabs[min(self.registry.vocabs)]
        if fields:
            name = self.registry.field_to_name[fields[0]]
            task_type = vocab.get("phenotype_task_types", {}).get(name, "unknown")
        else:
            task_type = "unsupported"
        return ExecutionPlan(
            case_id=case_id,
            question=question,
            target_phenotypes=fields,
            task_type=task_type,
            metrics=self._metrics_for(fields[0]) if fields else [],
            answer_mode="multiple_choice" if list(choices) else "open",
            supported=not unsupported,
            support_reason=("Question is covered by a trained phenotype head." if not unsupported else
                            "No relevant trained phenotype head is available; structured evidence is unavailable."),
            task_match="direct" if not unsupported else "none",
            phenotype_relevance_score=1.0 if not unsupported else 0.0,
            use_pathology_agent=True,
        )

    @staticmethod
    def _metrics_for(field: str) -> List[str]:
        if field == "OS":
            return ["survival_risk", "time_dependent_AUC", "C-index"]
        if field in {"ductal_binary", "lobular_binary", "dcis_binary", "lcis_binary",
                     "necrosis_binary", "comedonecrosis_binary", "microcalcification_binary"}:
            return ["probability", "AUC"]
        return ["class_probabilities", "ACC"]
