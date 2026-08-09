import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

from .clients import OpenAICompatibleClient, parse_json_response
from .schemas import ExecutionPlan


NONVISUAL_HINTS = (
    "tumor size", "surgical margin", "margin status", "treatment", "chemotherapy",
    "radiotherapy", "radiation therapy", "exact survival", "survival time", "how many days",
    "patient age", "medical history", "clinical history", "mention", "documented",
    "report", "record",
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
        self.field_to_prototype_id = {
            field: f"P{index:03d}"
            for index, field in enumerate(self.phenotype_fields, 1)
        }
        self.prototype_id_to_field = {
            prototype_id: field
            for field, prototype_id in self.field_to_prototype_id.items()
        }
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
        catalog = []
        for field, name in zip(self.phenotype_fields, self.phenotype_names):
            semantics = self.label_semantics(field)
            catalog.append({
                "prototype_id": self.field_to_prototype_id[field],
                "field": field,
                "name": name,
                "task_type": task_types.get(name, "unknown"),
                "group": groups.get(name, "unknown"),
                "labels": semantics.get("clinical_meaning", {}),
                "limitation": (
                    "WSI-derived prediction, not a measured clinical assay or report fact."
                ),
            })
        return catalog

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
        system = """You are the Router for a breast pathology multiple-choice VQA system. Decide which evidence tools to call; never answer or guess the MCQ.

Infer the target semantics from the question and supplied choices. Choices may identify a grade, histologic type, receptor, invasion extent, architecture, or yes/no task, but may never be used to guess the correct answer.

Decision priority:
1. phenotype_direct/direct
2. phenotype_direct/partial
3. morphology_only
4. nonvisual
Before choosing nonvisual, verify that no prototype materially distinguishes the choices and that the target is not local H&E morphology.

DIRECT: use phenotype_direct/direct only when one to four selected prototypes cover enough core answer semantics to distinguish the relevant choices. The prediction must be capable of resolving the MCQ's main semantic distinction; mere topical relation or a shared phenotype word is insufficient. Grade 1/2/3 with the grade prototype, yes/no lymphovascular invasion with the LVI prototype, and positive/negative/equivocal HER2 status with the HER2 prototype are direct.

PARTIAL: use phenotype_direct/partial when selected prototypes materially narrow the choices but cannot distinguish all important choices, requested attributes, or modifiers. Use the specialized evidence, but do not treat it as a complete answer. Examples: LVI can distinguish absent from present but not focal from extensive; microcalcification plus LVI can help with an additional-findings question without covering benign findings; histologic type can help with additional carcinoma but cannot distinguish fibroadenoma from fibrocystic change. If prototypes explain only some choices or some requested attributes, use partial rather than direct. For a mixed visual-phenotype and unavailable component, use partial only if the phenotype truly narrows the supplied choices.

Prototype coverage must stay within its label space. The histological-type prototype can distinguish carcinoma types such as IDC, ILC, and DCIS. It does not directly resolve solid versus cribriform DCIS architecture, fibroadenoma versus fibrocystic change, benign background morphology, or stromal patterns. A question about DCIS architecture is morphology_only. A question contrasting fibrocystic change, IDC, both, or neither may use histological type only as partial evidence.

MORPHOLOGY_ONLY: use with task_match=none, no IDs, and use_pathology_agent=true when the answer is local H&E morphology assessable from retrieved patches but no prototype sufficiently covers it. Examples include cribriform/papillary/acinar architecture, nuclear atypia, inflammation, fibrosis, adenosis, hyperplasia pattern, fibroadenoma, fibrocystic change, and benign surrounding tissue. Generic words such as finding, diagnosis, or additional finding do not imply nonvisual.

NONVISUAL: use with task_match=none, no IDs, and use_pathology_agent=false when neither available prototypes nor retrieved local H&E patches provide substantive evidence. This includes exact macroscopic size; report-dependent margins; treatment; age/history; exact survival time; medication; reproductive history; performance status; specimen preparation; metadata-only laterality; and what a report documented. It also includes global exhaustive specimen facts such as exact tumor-focus count, whole-specimen multifocality, exact extent, distance to nipple or margin, and exact distribution across named specimen regions unless a dedicated tool exists.

Apply the assay boundary consistently. General status questions about ER, PR, HER2, hormone-receptor positivity, combined receptor statuses, or findings related to HER2 may use the corresponding WSI-derived phenotype prototypes. Do not infer an assay merely because ER, PR, or HER2 is named. Explicit measured-assay or report questions are nonvisual: FISH or amplification result, IHC score, immunohistochemical/immunostain result, percentage of positive nuclei, staining percentage, Allred or exact assay score, laboratory-test result, or documented pathology-report fact. WSI predictions estimate status; they are not measured IHC, FISH, or report results.

Return JSON with route, prototype_ids, task_match, phenotype_relevance_score, reason, and use_pathology_agent. route must be phenotype_direct, morphology_only, or nonvisual; task_match must be direct, partial, or none. For phenotype_direct choose one to four exact catalog IDs. Do not invent IDs."""
        user = json.dumps({
            "question": question,
            "choices": list(choices),
            "prototype_catalog": self.registry.compact_catalog(),
            "output_schema": {
                "route": "phenotype_direct|morphology_only|nonvisual",
                "prototype_ids": ["P001"],
                "task_match": "direct|partial|none",
                "phenotype_relevance_score": 0.0,
                "reason": "one sentence",
                "use_pathology_agent": True,
            },
        }, ensure_ascii=False)
        try:
            parsed = parse_json_response(self.qwen.chat(
                system,
                user,
                max_tokens=384,
                response_format={"type": "json_object"},
                retries=2,
            ))
        except Exception as error:
            print(f"Planner Qwen unavailable, using visual fallback: {error}", flush=True)
            return None
        if not isinstance(parsed, dict):
            return None

        return self._normalize_llm_plan(case_id, question, choices, parsed)

    def _normalize_llm_plan(
        self,
        case_id: str,
        question: str,
        choices: Iterable[str],
        parsed: Dict[str, Any],
    ) -> ExecutionPlan:
        """Validate Router structure without interpreting question semantics."""

        raw_ids = parsed.get("prototype_ids", [])
        if isinstance(raw_ids, (str, dict)):
            raw_ids = [raw_ids]
        prototype_ids = []
        for value in raw_ids if isinstance(raw_ids, list) else []:
            if isinstance(value, dict):
                value = value.get("prototype_id", value.get("id"))
            if isinstance(value, str):
                normalized = value.strip().upper()
                if normalized in self.registry.prototype_id_to_field:
                    prototype_ids.append(normalized)
        prototype_ids = list(dict.fromkeys(prototype_ids))[:4]

        route = str(parsed.get(
            "route", "phenotype_direct" if prototype_ids else "morphology_only"
        )).strip().lower()
        if route not in {"phenotype_direct", "morphology_only", "nonvisual"}:
            route = "phenotype_direct" if prototype_ids else "morphology_only"
        if route == "phenotype_direct" and not prototype_ids:
            route = "morphology_only"
        if route != "phenotype_direct":
            prototype_ids = []

        fields = [
            self.registry.prototype_id_to_field[prototype_id]
            for prototype_id in prototype_ids
        ]
        supported = route == "phenotype_direct"
        task_match = str(parsed.get(
            "task_match", "direct" if supported else "none"
        )).strip().lower()
        if task_match not in {"direct", "partial", "none"}:
            task_match = "direct" if supported else "none"
        if not supported:
            task_match = "none"

        default_relevance = {"direct": 1.0, "partial": 0.6, "none": 0.0}[task_match]
        try:
            relevance = float(parsed.get("phenotype_relevance_score", default_relevance))
        except (TypeError, ValueError):
            relevance = default_relevance
        relevance = max(0.0, min(relevance, 1.0)) if supported else 0.0

        vocab = self.registry.vocabs[min(self.registry.vocabs)]
        if fields:
            name = self.registry.field_to_name[fields[0]]
            task_type = vocab.get("phenotype_task_types", {}).get(name, "unknown")
        else:
            task_type = "morphology" if route == "morphology_only" else "nonvisual"

        return ExecutionPlan(
            case_id=case_id,
            question=question,
            target_phenotypes=fields,
            task_type=task_type,
            metrics=self._metrics_for(fields[0]) if fields else [],
            answer_mode="multiple_choice" if list(choices) else "open",
            supported=supported,
            support_reason=str(parsed.get("reason", "")),
            task_match=task_match,
            phenotype_relevance_score=relevance,
            use_pathology_agent=(
                True
                if route == "morphology_only"
                else False
                if route == "nonvisual"
                else bool(parsed.get("use_pathology_agent", True))
            ),
            evidence_route=route,
            selected_prototype_ids=prototype_ids,
        )

    def _rule_plan(self, case_id: str, question: str, choices: Iterable[str]) -> ExecutionPlan:
        lowered = question.lower()
        route = (
            "nonvisual"
            if any(term in lowered for term in NONVISUAL_HINTS)
            else "morphology_only"
        )
        return ExecutionPlan(
            case_id=case_id,
            question=question,
            target_phenotypes=[],
            task_type="nonvisual" if route == "nonvisual" else "morphology",
            metrics=[],
            answer_mode="multiple_choice" if list(choices) else "open",
            supported=False,
            support_reason=(
                "The question requires non-visual information unavailable from WSI."
                if route == "nonvisual"
                else "No numbered phenotype prototype was selected; use diverse all-phenotype visual evidence."
            ),
            task_match="none",
            phenotype_relevance_score=0.0,
            use_pathology_agent=route == "morphology_only",
            evidence_route=route,
            selected_prototype_ids=[],
        )

    @staticmethod
    def _metrics_for(field: str) -> List[str]:
        if field == "OS":
            return ["survival_risk", "time_dependent_AUC", "C-index"]
        if field in {"ductal_binary", "lobular_binary", "dcis_binary", "lcis_binary",
                     "necrosis_binary", "comedonecrosis_binary", "microcalcification_binary"}:
            return ["probability", "AUC"]
        return ["class_probabilities", "ACC"]
