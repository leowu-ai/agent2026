import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

from .clients import OpenAICompatibleClient, parse_json_response
from .schemas import ExecutionPlan


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
        system = """You decompose evidence needs for a breast pathology MCQ Router. Do not choose the answer and do not output a final route or task_match.

Use only question, choices, and the prototype catalog:
1. Infer the true target semantics from question and choices; choices clarify the task but never reveal the answer.
2. Select the smallest set of at most four exact prototype IDs whose predictions directly measure the requested variable or a true component of it.
3. Classify prototype support as target_evidence, correlated_context, or none, then set coverage.
4. Independently judge whether retrieved local H&E patches can provide useful morphology.
5. Independently judge whether unavailable records, measurements, assays, history, or exhaustive specimen context are required.

TARGET_EVIDENCE means knowing the prototype prediction gives the requested variable itself, measures a real component of that same variable, or eliminates choices for that target-level reason. Histologic grade for a grade question, LVI for invasion presence or extent, and receptor status for a receptor-status question are target evidence.

CORRELATED_CONTEXT means a prototype may influence, correlate with, or predict the answer without representing the requested fact. Histology, grade, and receptors do not state recommended treatment; histology does not state a performed surgery; receptor status does not state which stains were performed; HER2 status is not a FISH amplification result; phenotypes do not state documented recommendations or report wording. Correlated context is never phenotype evidence, even if clinically predictive.

NONE means catalog predictions have no direct target-level discrimination. Broad carcinoma type does not distinguish DCIS architecture or benign lesions such as fibroadenoma, fibrocystic change, adenosis, mastitis, hyperplasia, or stromal fibrosis.

Select only the minimum discriminative target-evidence set, not every clinically related prototype. If histologic grade alone resolves the target, select only that grade prototype. Do not add LVI, necrosis, receptor, or broad histology merely as supporting context.

COMPLETE means target evidence resolves the MCQ's core semantic distinction. Examples: grade 1/2/3 with histologic grade; yes/no LVI with LVI; positive/negative/equivocal general HER2 status with HER2.

PARTIAL means prototype predictions materially eliminate or favor some choices but leave an important distinction unresolved. LVI separates absent from present but not focal from extensive angioinvasion. LVI does not cover perineural invasion in a combined invasion question. Microcalcification plus LVI can support some additional-finding choices but not benign or hyperplasia alternatives. Use partial even when the unresolved remainder requires unavailable report context.

prototype_coverage is meaningful only for target_evidence. For correlated_context or none, set coverage=none and keep prototype_ids empty.

Assay granularity matters. General ER/PR/HER2 status is target_evidence/complete. ER status for choices negative, positive <10%, or positive >10% is target_evidence/partial because it separates negative from positive but not percentage ranges. HER2 status for explicit FISH/gene amplification is correlated_context or none, never partial. Receptor predictions for which stain was performed or pending are correlated_context.

local_morphology_useful=true for patch-assessable architecture, atypia, inflammation, fibrosis, adenosis, hyperplasia, fibroadenoma, fibrocystic change, benign tissue, local necrosis, or stroma when no target-level prototype exists. Set false for exact size, focus count, exhaustive multifocality, total-tumor percentage, margin distance, specimen extent/distribution, treatment/history/age, report wording, and exact TNM or gross facts. Do not force broad prototypes into partial merely to avoid local morphology.

requires_unavailable_context=true does not erase genuine target_evidence/partial, but correlated_context plus unavailable context is not partial. Return JSON only with prototype_ids, prototype_support_type, prototype_coverage, local_morphology_useful, requires_unavailable_context, phenotype_relevance_score, reason, and use_pathology_agent. Do not invent IDs."""
        user = json.dumps({
            "question": question,
            "choices": list(choices),
            "prototype_catalog": self.registry.compact_catalog(),
            "output_schema": {
                "prototype_ids": ["P001"],
                "prototype_support_type": "target_evidence|correlated_context|none",
                "prototype_coverage": "complete|partial|none",
                "local_morphology_useful": True,
                "requires_unavailable_context": False,
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

        support_type = str(parsed.get("prototype_support_type", "none")).strip().lower()
        if support_type not in {"target_evidence", "correlated_context", "none"}:
            support_type = "none"
        coverage = str(parsed.get("prototype_coverage", "none")).strip().lower()
        if coverage not in {"complete", "partial", "none"}:
            coverage = "none"
        local_morphology_useful = self._as_bool(
            parsed.get("local_morphology_useful", False)
        )
        requires_unavailable_context = self._as_bool(
            parsed.get("requires_unavailable_context", False)
        )

        if support_type != "target_evidence":
            prototype_ids = []
            coverage = "none"

        candidate_fields = [
            self.registry.prototype_id_to_field[prototype_id]
            for prototype_id in prototype_ids
        ]
        if (
            coverage == "complete"
            and candidate_fields
            and not self._label_space_covers_choices(candidate_fields, choices)
        ):
            coverage = "partial"

        if not prototype_ids or coverage == "none":
            coverage = "none"
            prototype_ids = []
            route = "morphology_only"
            task_match = "none"
        else:
            route = "phenotype_direct"
            task_match = "direct" if coverage == "complete" else "partial"

        fields = [
            self.registry.prototype_id_to_field[prototype_id]
            for prototype_id in prototype_ids
        ]
        supported = route == "phenotype_direct"
        default_relevance = {"complete": 1.0, "partial": 0.65, "none": 0.0}[coverage]
        try:
            relevance = float(parsed.get("phenotype_relevance_score", default_relevance))
        except (TypeError, ValueError):
            relevance = default_relevance
        if coverage == "complete":
            relevance = max(0.85, min(relevance, 1.0))
        elif coverage == "partial":
            relevance = max(0.45, min(relevance, 0.85))
        else:
            relevance = 0.0

        vocab = self.registry.vocabs[min(self.registry.vocabs)]
        if fields:
            name = self.registry.field_to_name[fields[0]]
            task_type = vocab.get("phenotype_task_types", {}).get(name, "unknown")
        else:
            task_type = "morphology"

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
                else True
                if task_match == "partial"
                else self._as_bool(parsed.get("use_pathology_agent", True))
            ),
            evidence_route=route,
            selected_prototype_ids=prototype_ids,
            prototype_support_type=support_type,
            prototype_coverage=coverage,
            local_morphology_useful=local_morphology_useful,
            requires_unavailable_context=requires_unavailable_context,
        )

    def _label_space_covers_choices(
        self, fields: List[str], choices: Iterable[str]
    ) -> bool:
        if not hasattr(self.registry, "label_semantics"):
            return True
        choice_list = [str(choice) for choice in choices]
        if not choice_list:
            return True
        clinical_choices = [
            choice for choice in choice_list
            if not any(
                phrase in self._normalize_label(choice)
                for phrase in (
                    "cannot determine", "not provided", "not specified",
                    "not mentioned", "unknown", "insufficient",
                )
            )
        ]
        if not clinical_choices:
            return False
        return all(
            any(
                self._field_label_matches_choice(field, label, choice)
                for field in fields
                for label in self.registry.label_semantics(field).get(
                    "clinical_meaning", {}
                ).values()
            )
            for choice in clinical_choices
        )

    @staticmethod
    def _normalize_label(value: Any) -> str:
        normalized = " ".join(
            __import__("re").findall(r"[a-z0-9]+", str(value or "").lower())
        )
        return normalized.replace("infiltrating", "invasive")

    def _field_label_matches_choice(
        self, field: str, label: Any, choice: str
    ) -> bool:
        normalized_label = self._normalize_label(label)
        normalized_choice = self._normalize_label(choice)
        if not normalized_label:
            return False
        field_terms = {
            "ER_status_label": ("er", "estrogen receptor", "estrogen receptors"),
            "PR_status_label": ("pr", "progesterone receptor", "progesterone receptors"),
            "HER2_status_label": (
                "her2", "her 2", "human epidermal growth factor receptor 2",
            ),
        }
        terms = field_terms.get(field, ())
        if terms:
            all_receptor_terms = {
                term for values in field_terms.values() for term in values
            }
            mentions_other = any(
                f" {self._normalize_label(term)} " in f" {normalized_choice} "
                and term not in terms
                for term in all_receptor_terms
            )
            mentions_field = any(
                f" {self._normalize_label(term)} " in f" {normalized_choice} "
                for term in terms
            )
            if mentions_other and not mentions_field:
                return False
        return (
            normalized_choice == normalized_label
            or f" {normalized_label} " in f" {normalized_choice} "
            or f" {normalized_choice} " in f" {normalized_label} "
        )

    @staticmethod
    def _as_bool(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() == "true"
        return False

    def _rule_plan(self, case_id: str, question: str, choices: Iterable[str]) -> ExecutionPlan:
        return ExecutionPlan(
            case_id=case_id,
            question=question,
            target_phenotypes=[],
            task_type="morphology",
            metrics=[],
            answer_mode="multiple_choice" if list(choices) else "open",
            supported=False,
            support_reason=(
                "No numbered phenotype prototype was selected; use broad G2P and "
                "diverse all-phenotype visual evidence."
            ),
            task_match="none",
            phenotype_relevance_score=0.0,
            use_pathology_agent=True,
            evidence_route="morphology_only",
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
