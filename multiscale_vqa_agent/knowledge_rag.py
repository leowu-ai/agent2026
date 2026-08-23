import json
import re
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple


TOKEN_PATTERN = re.compile(r"[a-z0-9]+")

SCALE_VISUAL_ROLES = {
    "4096": "Use this scale for question-relevant global architecture and broad tissue organization.",
    "2048": "Use this scale for question-relevant intermediate structural patterns and local tissue relationships.",
    "1024": "Use this scale for question-relevant fine morphology and cytology.",
}


def _tokens(value: Any) -> Set[str]:
    return set(TOKEN_PATTERN.findall(str(value or "").lower()))


def _text_values(row: Dict[str, Any], keys: Iterable[str]) -> List[str]:
    values: List[str] = []
    for key in keys:
        value = row.get(key)
        if isinstance(value, list):
            values.extend(str(item) for item in value)
        elif value:
            values.append(str(value))
    return values


class KnowledgeRAG:
    """Small deterministic retriever over the read-only pathology KB ZIP."""

    REQUIRED_FILES = (
        "pathology_concepts.jsonl",
        "tool_semantics.jsonl",
        "evidence_rules.jsonl",
        "model_relations.jsonl",
        "program_gene_candidates.jsonl",
        "manifest.json",
    )
    COUNT_KEYS = {
        "pathology_concepts.jsonl": "pathology_and_program_concepts",
        "tool_semantics.jsonl": "phenotype_tool_semantics",
        "evidence_rules.jsonl": "evidence_rules",
        "model_relations.jsonl": "model_relation_entries",
        "program_gene_candidates.jsonl": "program_gene_candidate_entries",
    }
    V2_FILES = {
        "evidence_limitations_v2.jsonl": "evidence_limitations_v2",
        "proxy_evidence_rules_v2.jsonl": "proxy_evidence_rules_v2",
        "forced_choice_reasoning_v2.jsonl": "forced_choice_reasoning_v2",
        "reasoning_examples_v2.jsonl": "reasoning_examples_v2",
    }

    def __init__(self, zip_path: str, registry: Any):
        self.zip_path = Path(zip_path)
        self.registry = registry
        if not self.zip_path.is_file():
            raise FileNotFoundError(f"Knowledge base ZIP not found: {self.zip_path}")
        self.prefix, self._members = self._inspect_archive()
        self.manifest = self._read_json("manifest.json")
        self.pathology_concepts = self._read_jsonl("pathology_concepts.jsonl")
        self.tool_semantics = self._read_jsonl("tool_semantics.jsonl")
        self.evidence_rules = self._read_jsonl("evidence_rules.jsonl")
        self.model_relations = self._read_jsonl("model_relations.jsonl")
        self.program_gene_candidates = self._read_jsonl(
            "program_gene_candidates.jsonl"
        )
        self.knowledge_base_version = str(self.manifest.get("version") or "1.0")
        self.evidence_limitations: List[Dict[str, Any]] = []
        self.proxy_evidence_rules: List[Dict[str, Any]] = []
        self.forced_choice_rules: List[Dict[str, Any]] = []
        self.reasoning_examples: List[Dict[str, Any]] = []
        if self._version_major(self.knowledge_base_version) >= 2:
            missing = [
                name for name in self.V2_FILES
                if f"{self.prefix}{name}" not in self._members
            ]
            if missing:
                raise ValueError(f"Knowledge base v2 ZIP is missing: {missing}")
            self.evidence_limitations = self._read_jsonl(
                "evidence_limitations_v2.jsonl"
            )
            self.proxy_evidence_rules = self._read_jsonl(
                "proxy_evidence_rules_v2.jsonl"
            )
            self.forced_choice_rules = self._read_jsonl(
                "forced_choice_reasoning_v2.jsonl"
            )
            self.reasoning_examples = self._read_jsonl(
                "reasoning_examples_v2.jsonl"
            )
        self._validate()
        self.tool_by_field = {
            str(row.get("field")): row for row in self.tool_semantics
        }
        self.relation_by_field = {
            str(row.get("phenotype")): row for row in self.model_relations
        }
        self.genes_by_program = {
            str(row.get("program")): row for row in self.program_gene_candidates
        }

    @staticmethod
    def _version_major(version: str) -> int:
        try:
            return int(str(version).split(".", 1)[0])
        except (TypeError, ValueError):
            return 1

    def _inspect_archive(self) -> Tuple[str, Set[str]]:
        with zipfile.ZipFile(self.zip_path) as archive:
            names = {
                name for name in archive.namelist()
                if name and not name.endswith("/")
            }
        manifest_matches = [
            name for name in names if PurePosixPath(name).name == "manifest.json"
        ]
        if len(manifest_matches) != 1:
            raise ValueError(
                "Knowledge base ZIP must contain exactly one manifest.json"
            )
        prefix_path = PurePosixPath(manifest_matches[0]).parent
        prefix = "" if str(prefix_path) == "." else f"{prefix_path.as_posix()}/"
        missing = [name for name in self.REQUIRED_FILES if f"{prefix}{name}" not in names]
        if missing:
            raise ValueError(f"Knowledge base ZIP is missing: {missing}")
        return prefix, names

    def _read_text(self, name: str) -> str:
        member = f"{self.prefix}{name}"
        with zipfile.ZipFile(self.zip_path) as archive:
            return archive.read(member).decode("utf-8")

    def _read_json(self, name: str) -> Dict[str, Any]:
        value = json.loads(self._read_text(name))
        if not isinstance(value, dict):
            raise ValueError(f"{name} must contain a JSON object")
        return value

    def _read_jsonl(self, name: str) -> List[Dict[str, Any]]:
        rows = []
        for line_number, line in enumerate(self._read_text(name).splitlines(), 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{name}:{line_number} must contain a JSON object")
            rows.append(value)
        return rows

    def _validate(self) -> None:
        expected = self.manifest.get("counts")
        if not isinstance(expected, dict):
            raise ValueError("Knowledge base manifest is missing counts")
        loaded = {
            "pathology_concepts.jsonl": self.pathology_concepts,
            "tool_semantics.jsonl": self.tool_semantics,
            "evidence_rules.jsonl": self.evidence_rules,
            "model_relations.jsonl": self.model_relations,
            "program_gene_candidates.jsonl": self.program_gene_candidates,
        }
        for filename, rows in loaded.items():
            count_key = self.COUNT_KEYS[filename]
            if int(expected.get(count_key, -1)) != len(rows):
                raise ValueError(
                    f"Knowledge base count mismatch for {filename}: "
                    f"manifest={expected.get(count_key)!r}, loaded={len(rows)}"
                )
        if self._version_major(self.knowledge_base_version) >= 2:
            v2_loaded = {
                "evidence_limitations_v2.jsonl": self.evidence_limitations,
                "proxy_evidence_rules_v2.jsonl": self.proxy_evidence_rules,
                "forced_choice_reasoning_v2.jsonl": self.forced_choice_rules,
                "reasoning_examples_v2.jsonl": self.reasoning_examples,
            }
            for filename, rows in v2_loaded.items():
                count_key = self.V2_FILES[filename]
                if int(expected.get(count_key, -1)) != len(rows):
                    raise ValueError(
                        f"Knowledge base count mismatch for {filename}: "
                        f"manifest={expected.get(count_key)!r}, loaded={len(rows)}"
                    )
        fields = {str(row.get("field")) for row in self.tool_semantics}
        registry_fields = set(getattr(self.registry, "phenotype_fields", []))
        if fields != registry_fields:
            raise ValueError(
                "Knowledge base phenotype tool semantics do not match ToolBank registry"
            )
        relation_fields = {
            str(row.get("phenotype")) for row in self.model_relations
        }
        if relation_fields != registry_fields:
            raise ValueError(
                "Knowledge base model relations do not match ToolBank registry"
            )
        registry_programs = set(getattr(self.registry, "programs", []))
        relation_programs = {
            str(program.get("name"))
            for row in self.model_relations
            for program in row.get("programs", [])
        }
        unknown_relation_programs = relation_programs - registry_programs
        if unknown_relation_programs:
            raise ValueError(
                "Knowledge base model relations contain unknown programs: "
                f"{sorted(unknown_relation_programs)}"
            )
        unknown_programs = {
            str(row.get("program")) for row in self.program_gene_candidates
        } - registry_programs
        if unknown_programs:
            raise ValueError(
                f"Knowledge base contains unknown programs: {sorted(unknown_programs)}"
            )

    def retrieve(
        self,
        question: str,
        choices: Iterable[str],
        target_phenotypes: Iterable[str],
    ) -> Dict[str, Any]:
        choice_list = [str(choice) for choice in choices]
        targets = [
            field for field in dict.fromkeys(str(value) for value in target_phenotypes)
            if field in self.tool_by_field
        ]
        query_text = " ".join([question, *choice_list])
        query_tokens = _tokens(query_text)
        ranked = []
        for row in self.pathology_concepts:
            score, trace = self._concept_score(row, query_text, query_tokens)
            if score > 0:
                ranked.append((score, str(row.get("id", "")), row, trace))
        ranked.sort(key=lambda item: (-item[0], item[1]))
        matched = [self._compact_concept(item[2], item[0]) for item in ranked[:6]]

        direct_tools = [
            self._compact_tool(self.tool_by_field[field], "direct")
            for field in targets
        ]
        related_fields = []
        for _, _, row, _ in ranked[:6]:
            related_fields.extend(row.get("related_phenotypes", []) or [])
        supportive_fields = [
            field for field in dict.fromkeys(str(value) for value in related_fields)
            if field in self.tool_by_field and field not in targets
        ][:5]
        supportive_tools = [
            self._compact_tool(self.tool_by_field[field], "supportive")
            for field in supportive_fields
        ]

        program_scores: Dict[str, float] = {}
        program_sources: Dict[str, Set[str]] = {}
        for score, _, row, _ in ranked[:6]:
            for name in row.get("candidate_programs", []) or []:
                self._add_candidate(
                    program_scores, program_sources, str(name), score, "concept"
                )
        for field in targets:
            relation = self.relation_by_field.get(field, {})
            for rank, program in enumerate(relation.get("programs", [])[:5]):
                name = str(program.get("name", ""))
                graph_score = float(program.get("graph_score") or 0.0)
                self._add_candidate(
                    program_scores,
                    program_sources,
                    name,
                    2.0 + graph_score - rank * 0.01,
                    f"model_relation:{field}",
                )
        candidate_programs = [
            {
                "name": name,
                "relevance": round(score, 6),
                "sources": sorted(program_sources.get(name, set())),
                "evidence_role": "supportive",
            }
            for name, score in sorted(
                program_scores.items(), key=lambda item: (-item[1], item[0])
            )[:8]
            if name in set(getattr(self.registry, "programs", []))
        ]

        gene_scores: Dict[str, float] = {}
        gene_sources: Dict[str, Set[str]] = {}
        for score, _, row, _ in ranked[:6]:
            for name in row.get("candidate_genes", []) or []:
                self._add_candidate(
                    gene_scores, gene_sources, str(name), score, "concept"
                )
        for program in candidate_programs:
            row = self.genes_by_program.get(program["name"], {})
            for gene in row.get("candidate_genes", []) or []:
                self._add_candidate(
                    gene_scores,
                    gene_sources,
                    str(gene),
                    float(program["relevance"]),
                    f"program:{program['name']}",
                )
        registry_genes = set(getattr(self.registry, "genes", []))
        candidate_genes = [
            {
                "name": name,
                "relevance": round(score, 6),
                "sources": sorted(gene_sources.get(name, set())),
                "evidence_role": "supportive",
                "requires_h_validation": True,
            }
            for name, score in sorted(
                gene_scores.items(), key=lambda item: (-item[1], item[0])
            )
            if name in registry_genes
        ][:12]

        limitations = self._unique(
            limitation
            for _, _, row, _ in ranked[:6]
            for limitation in row.get("limitations", []) or []
        )
        scale_strategy = self._unique(
            strategy
            for _, _, row, _ in ranked[:6]
            for strategy in row.get("scale_strategy", []) or []
        )
        relevant_rules = self._retrieve_rules(query_tokens, bool(targets))
        evidence_limitations = self._retrieve_v2_rows(
            self.evidence_limitations,
            query_text,
            query_tokens,
            ("applies_to", "target_type"),
            ("id", "target_type", "recoverability", "direct_evidence",
             "proxy_evidence", "invalid_inference", "forced_choice_policy"),
            limit=5,
        )
        proxy_evidence_rules = self._retrieve_v2_rows(
            self.proxy_evidence_rules,
            query_text,
            query_tokens,
            ("target", "stronger_evidence", "allowed_proxy", "option_use"),
            ("id", "target", "stronger_evidence", "allowed_proxy", "do_not",
             "option_use"),
            limit=5,
        )
        forced_choice_rules = [
            {
                "id": row.get("id"),
                "priority": row.get("priority"),
                "rule": row.get("rule"),
                "evidence_role": "general_reasoning_constraint",
                "patient_specific": False,
            }
            for row in sorted(
                self.forced_choice_rules,
                key=lambda row: (int(row.get("priority") or 999), str(row.get("id", ""))),
            )[:12]
        ]
        reasoning_examples = self._retrieve_v2_rows(
            self.reasoning_examples,
            query_text,
            query_tokens,
            ("category", "question_template", "choices_template", "teaches"),
            ("id", "category", "question_template", "choices_template", "evidence",
             "reasoning", "teaches"),
            limit=3,
            minimum_score=2.5,
        )
        for row in reasoning_examples:
            row["evidence_role"] = "generic_reasoning_example"
            row["patient_specific"] = False
        scale_guidance = self._scale_specific_visual_guidance(matched)
        return {
            "matched_concepts": matched,
            "direct_tools": direct_tools,
            "supportive_tools": supportive_tools,
            "scale_strategy": scale_strategy[:8],
            "scale_specific_visual_guidance": scale_guidance,
            "candidate_programs": candidate_programs,
            "candidate_genes": candidate_genes,
            "limitations": limitations[:10],
            "evidence_rules": relevant_rules,
            "evidence_limitations": evidence_limitations,
            "proxy_evidence_rules": proxy_evidence_rules,
            "forced_choice_rules": forced_choice_rules,
            "reasoning_examples": reasoning_examples,
            "retrieval_trace": {
                "method": "deterministic_lexical_v2" if self._version_major(
                    self.knowledge_base_version
                ) >= 2 else "deterministic_lexical_v1",
                "query_token_count": len(query_tokens),
                "target_phenotypes_forced": targets,
                "concept_candidates_scored": len(ranked),
                "knowledge_base_version": self.knowledge_base_version,
                "v2_limitation_ids": [row.get("id") for row in evidence_limitations],
                "v2_proxy_rule_ids": [row.get("id") for row in proxy_evidence_rules],
                "v2_example_ids": [row.get("id") for row in reasoning_examples],
            },
        }

    @staticmethod
    def _retrieve_v2_rows(
        rows: List[Dict[str, Any]],
        query_text: str,
        query_tokens: Set[str],
        search_keys: Iterable[str],
        output_keys: Iterable[str],
        limit: int,
        minimum_score: float = 0.1,
    ) -> List[Dict[str, Any]]:
        lowered = query_text.lower()
        ranked = []
        generic = {
            "a", "an", "and", "are", "as", "at", "be", "breast", "by",
            "cancer", "for", "from", "has", "in", "is", "it", "of", "or",
            "patient", "the", "this", "to", "tumor", "was", "what", "which",
            "with",
        }
        meaningful_query = {
            token for token in query_tokens - generic if not token.isdigit()
        }
        if "ptnm" in meaningful_query:
            meaningful_query.add("tnm")
        for row in rows:
            values = _text_values(row, search_keys)
            phrases = [value.lower().strip() for value in values if value.strip()]
            phrase_hits = sum(
                1 for phrase in phrases
                if len(_tokens(phrase) - generic) >= 1 and phrase in lowered
            )
            searchable = {
                token for token in _tokens(" ".join(values)) - generic
                if not token.isdigit()
            }
            primary_values = _text_values(row, tuple(search_keys)[:2])
            primary_tokens = _tokens(" ".join(primary_values)) - generic
            overlap = len(meaningful_query & searchable)
            primary_overlap = len(meaningful_query & primary_tokens)
            score = 3.0 * phrase_hits + float(overlap) + float(primary_overlap)
            if score >= minimum_score:
                ranked.append((score, str(row.get("id", "")), row))
        ranked.sort(key=lambda item: (-item[0], item[1]))
        return [
            {key: row.get(key) for key in output_keys if key in row}
            for _, _, row in ranked[:limit]
        ]

    @staticmethod
    def _concept_score(
        row: Dict[str, Any], query_text: str, query_tokens: Set[str]
    ) -> Tuple[float, Dict[str, Any]]:
        lowered = query_text.lower()
        concept = str(row.get("concept", "")).lower().strip()
        aliases = [str(value).lower().strip() for value in row.get("aliases", [])]
        intents = [str(value) for value in row.get("question_intents", [])]
        exact = bool(concept and concept in lowered)
        alias_matches = [alias for alias in aliases if alias and alias in lowered]
        intent_tokens = _tokens(" ".join(intents))
        concept_tokens = _tokens(" ".join([concept, *aliases]))
        intent_overlap = len(query_tokens & intent_tokens)
        concept_overlap = len(query_tokens & concept_tokens)
        score = (
            (5.0 if exact else 0.0)
            + min(4.0, 2.0 * len(alias_matches))
            + 0.8 * intent_overlap
            + 0.4 * concept_overlap
        )
        return score, {
            "concept_exact": exact,
            "alias_matches": alias_matches,
            "intent_overlap": intent_overlap,
            "token_overlap": concept_overlap,
        }

    def _retrieve_rules(
        self, query_tokens: Set[str], has_targets: bool
    ) -> List[Dict[str, Any]]:
        ranked = []
        for row in self.evidence_rules:
            searchable = _tokens(json.dumps(row.get("applies_to", [])))
            score = len(query_tokens & searchable)
            if row.get("id") == "rule_morphology_coarse_to_fine":
                score += 2
            if has_targets and row.get("id") == "rule_direct_phenotype_first":
                score += 2
            ranked.append((score, str(row.get("id", "")), row))
        ranked.sort(key=lambda item: (-item[0], item[1]))
        return [
            {
                "id": row.get("id"),
                "preferred_evidence_order": row.get("preferred_evidence_order", []),
                "allowed_escalation": row.get("allowed_escalation", []),
                "stop_rules": row.get("stop_rules", []),
            }
            for score, _, row in ranked[:5]
            if score > 0
        ]

    @staticmethod
    def _compact_concept(row: Dict[str, Any], score: float) -> Dict[str, Any]:
        return {
            "id": row.get("id"),
            "concept": row.get("concept"),
            "evidence_role": row.get("evidence_role"),
            "direct_visual_evidence": row.get("direct_visual_evidence", [])[:3],
            "supportive_visual_evidence": row.get(
                "supportive_visual_evidence", []
            )[:3],
            "scale_strategy": row.get("scale_strategy", [])[:3],
            "relevance": round(float(score), 6),
        }

    @staticmethod
    def _scale_specific_visual_guidance(
        matched_concepts: List[Dict[str, Any]],
    ) -> Dict[str, List[str]]:
        guidance = {
            scale: [role] for scale, role in SCALE_VISUAL_ROLES.items()
        }
        for concept in matched_concepts:
            name = str(concept.get("concept") or "matched concept")
            direct = list(concept.get("direct_visual_evidence", []) or [])
            supportive = list(
                concept.get("supportive_visual_evidence", []) or []
            )
            visible_cue = next(iter(direct or supportive), None)
            for strategy in concept.get("scale_strategy", []) or []:
                match = re.match(r"^\s*(4096|2048|1024)\s*:\s*(.+)$", str(strategy))
                if not match:
                    continue
                scale, focus = match.groups()
                text = f"{name}: focus on {focus.strip()}"
                if visible_cue:
                    text += f"; look for whether this is visible: {visible_cue}"
                guidance[scale].append(text[:600])
        return {
            scale: list(dict.fromkeys(rows))[:5]
            for scale, rows in guidance.items()
        }

    @staticmethod
    def _compact_tool(row: Dict[str, Any], role: str) -> Dict[str, Any]:
        return {
            "field": row.get("field"),
            "display_name": row.get("display_name"),
            "evidence_role": role,
            "direct_targets": row.get("direct_targets", []),
            "partial_targets": row.get("partial_targets", []),
            "supportive_targets": row.get("supportive_targets", []),
            "not_valid_for": row.get("not_valid_for", []),
        }

    @staticmethod
    def _add_candidate(
        scores: Dict[str, float],
        sources: Dict[str, Set[str]],
        name: str,
        score: float,
        source: str,
    ) -> None:
        if not name:
            return
        scores[name] = max(scores.get(name, float("-inf")), float(score))
        sources.setdefault(name, set()).add(source)

    @staticmethod
    def _unique(values: Iterable[Any]) -> List[str]:
        return list(dict.fromkeys(str(value) for value in values if value))
