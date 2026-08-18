import json
import re
from typing import Any, Dict, List, Optional

from .clients import OpenAICompatibleClient
from .clients import parse_json_response
from .schemas import EvidenceGroup


PATHOLOGY_SYSTEM_PROMPT = """You are a breast pathology morphology observer, not a diagnosis or molecular-status agent.
Describe only structures directly visible in the supplied H&E pixels. Do not choose an MCQ option and do not diagnose a disease or molecular subtype. Never infer ER status, PR status, HER2 status, triple-negative status, gene expression, pathways, mutation, RNA, protein, IHC, FISH/ISH, amplification, treatment, clinical records, or report facts.
Scale-specific knowledge guidance tells you what morphology may be useful to inspect; it is not patient evidence. Never claim a guided feature is present unless it is actually visible in the supplied pixels. Explicitly report absent, indeterminate, or limited findings when appropriate.
For program/gene-selected images, you are not told why a patch was retrieved and must not guess its provenance. Indeterminate or limited-quality findings are not contradictions. Output JSON only with architecture, cytology, stroma, necrosis, invasion_pattern, visible_findings, target_visual_support, and image_quality."""


MOLECULAR_OR_DIAGNOSTIC_PATTERN = re.compile(
    r"\b(?:er|pr|her2|triple[- ]negative|gene|pathway|rna|mutation|ihc|fish|ish|"
    r"amplification|copy number|protein|treatment|clinical record|diagnosis|"
    r"carcinoma|sarcoma|lymphoma|cancer)\b",
    re.I,
)


class PathologyAgent:
    def __init__(self, client: OpenAICompatibleClient):
        self.client = client

    def describe(
        self,
        question: str,
        field: str,
        groups: List[EvidenceGroup],
        overview_paths: Optional[List[str]] = None,
        hide_provenance: bool = False,
        choices: Optional[List[str]] = None,
        current_scale: Optional[int] = None,
        evidence_role: Optional[str] = None,
        visual_guidance: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        del field  # Retrieval provenance must not bias the visual expert.
        entries = self._image_entries(groups, overview_paths or [])
        evidence = [group.to_dict() for group in groups]
        if not entries:
            return {
                "backend": "no_visual_evidence",
                "description": "No cropped WSI evidence image was available.",
                "image_count": 0,
                "available_image_count": 0,
                "request_attempts": 0,
                "evidence_groups": evidence,
                "image_metadata": [],
            }
        if not self.client.enabled:
            return {
                "backend": "mock",
                "description": "Patho-R1 was not called; evidence patches were selected successfully.",
                "image_count": len(entries),
                "available_image_count": len(entries),
                "request_attempts": 0,
                "evidence_groups": evidence,
                "image_metadata": [
                    self._entry_metadata(entry, index, hide_provenance)
                    for index, entry in enumerate(entries, 1)
                ],
            }

        system = PATHOLOGY_SYSTEM_PROMPT
        attempts = self._attempt_schedule(len(entries))
        failures = []
        for attempt_index, (image_limit, image_max_size) in enumerate(attempts, 1):
            selected = self._select_entries(entries, image_limit)
            user = json.dumps({
                "question": question,
                "choices": list(choices or []),
                "current_scale": current_scale,
                "evidence_role": evidence_role,
                "scale_specific_visual_guidance": list(visual_guidance or []),
                "image_order": [
                    self._entry_metadata(entry, index, hide_provenance)
                    for index, entry in enumerate(selected, 1)
                ],
                "evidence_rule": (
                    "Guidance defines what to look for, not what is present. Report only observed "
                    "image morphology. Retrieval provenance and molecular predictions are hidden "
                    "and must not be guessed."
                ),
            }, ensure_ascii=False)
            try:
                description = self.client.chat(
                    system,
                    user,
                    images=[entry["image_path"] for entry in selected],
                    max_tokens=384,
                    retries=0,
                    image_max_size=image_max_size,
                    response_format={"type": "json_object"},
                    enable_thinking=False,
                )
                if not description:
                    raise RuntimeError("Patho-R1 returned an empty response")
                morphology = self._normalize_morphology(description)
                if morphology is None:
                    raise RuntimeError(
                        "Patho-R1 did not return valid morphology-only JSON"
                    )
                return {
                    "backend": "pathor1",
                    "description": json.dumps(morphology, ensure_ascii=False),
                    "morphology_observation": morphology,
                    "raw_response": description,
                    "image_count": len(selected),
                    "available_image_count": len(entries),
                    "request_attempts": attempt_index,
                    "image_max_size": image_max_size,
                    "retry_history": failures,
                    "evidence_groups": evidence,
                    "image_metadata": [
                        self._entry_metadata(entry, index, hide_provenance)
                        for index, entry in enumerate(selected, 1)
                    ],
                }
            except Exception as error:
                failures.append(
                    f"{type(error).__name__}: {str(error)[:160]}"
                )

        return {
            "backend": "pathor1_error",
            "description": "Patho-R1 request failed after adaptive image retries.",
            "image_count": attempts[-1][0] if attempts else 0,
            "available_image_count": len(entries),
            "request_attempts": len(attempts),
            "image_max_size": attempts[-1][1] if attempts else None,
            "retry_history": failures,
            "evidence_groups": evidence,
            "image_metadata": [
                self._entry_metadata(entry, index, hide_provenance)
                for index, entry in enumerate(
                    self._select_entries(entries, attempts[-1][0]), 1
                )
            ] if attempts else [],
        }

    @staticmethod
    def _normalize_morphology(raw: str) -> Optional[Dict[str, Any]]:
        parsed = parse_json_response(raw)
        if not isinstance(parsed, dict):
            return None

        def clean(value: Any, limit: int = 320) -> str:
            text = " ".join(str(value or "").split())[:limit]
            if MOLECULAR_OR_DIAGNOSTIC_PATTERN.search(text):
                return "indeterminate"
            return text or "indeterminate"

        visible = parsed.get("visible_findings", [])
        if not isinstance(visible, list):
            visible = [visible]
        visible = [clean(value, 220) for value in visible[:6]]
        visible = [value for value in visible if value != "indeterminate"]
        necrosis = str(parsed.get("necrosis", "indeterminate")).lower()
        if necrosis not in {"present", "absent", "indeterminate"}:
            necrosis = "indeterminate"
        support = str(
            parsed.get("target_visual_support", "indeterminate")
        ).lower()
        if support not in {"supportive", "contradictory", "indeterminate"}:
            support = "indeterminate"
        quality = str(parsed.get("image_quality", "limited")).lower()
        if quality not in {"adequate", "limited"}:
            quality = "limited"
        return {
            "architecture": clean(parsed.get("architecture")),
            "cytology": clean(parsed.get("cytology")),
            "stroma": clean(parsed.get("stroma")),
            "necrosis": necrosis,
            "invasion_pattern": clean(parsed.get("invasion_pattern")),
            "visible_findings": visible,
            "target_visual_support": support,
            "image_quality": quality,
        }

    @staticmethod
    def _image_entries(
        groups: List[EvidenceGroup], overview_paths: List[str]
    ) -> List[Dict[str, Any]]:
        overviews = [
            {
                "kind": "overview",
                "group_id": None,
                "scale": None,
                "image_path": str(path),
            }
            for path in overview_paths
            if path
        ]
        patches = [
            {
                "kind": "patch",
                "group_id": group.group_id,
                "scale": int(scale),
                "evidence_source": group.evidence_source,
                "image_path": patch.image_path,
            }
            for group in groups
            for scale, patch in sorted(group.patches.items(), reverse=True)
            if patch.image_path
        ]
        return overviews + patches

    @staticmethod
    def _entry_metadata(
        entry: Dict[str, Any], ordinal: int, hide_provenance: bool = False
    ) -> Dict[str, Any]:
        result = {"ordinal": ordinal, "kind": entry["kind"]}
        if entry["kind"] == "patch":
            result.update({
                "group_id": entry["group_id"],
                "scale": entry["scale"],
            })
            if not hide_provenance:
                result["evidence_source"] = entry.get("evidence_source")
        return result

    @staticmethod
    def _attempt_schedule(available: int) -> List[tuple]:
        if available <= 0:
            return [(0, 512)]
        schedule = [
            (min(8, available), 1024),
            (min(6, available), 768),
            (min(4, available), 512),
        ]
        result = []
        for item in schedule:
            if item not in result:
                result.append(item)
        return result

    @staticmethod
    def _select_entries(entries: List[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
        overviews = [entry for entry in entries if entry["kind"] == "overview"]
        if len(entries) <= limit:
            selected = list(entries)
            if not overviews:
                selected.sort(key=lambda item: (-item["scale"], item["group_id"]))
            return selected

        patches = [entry for entry in entries if entry["kind"] == "patch"]
        by_group: Dict[int, List[Dict[str, Any]]] = {}
        for entry in patches:
            by_group.setdefault(entry["group_id"], []).append(entry)
        for values in by_group.values():
            values.sort(key=lambda item: item["scale"], reverse=True)

        selected = []
        seen = set()

        def add(entry: Dict[str, Any]):
            key = (
                entry["kind"], entry["group_id"], entry["scale"],
                entry["image_path"],
            )
            if key not in seen and len(selected) < limit:
                selected.append(entry)
                seen.add(key)

        # Preserve broad WSI context before patch-level evidence.
        for entry in overviews[:2]:
            add(entry)
        # Cover every evidence group at architecture scale first.
        for values in by_group.values():
            add(values[0])
        # Add cellular detail for the strongest groups next.
        for values in by_group.values():
            if len(values) > 1:
                add(values[-1])
        # Fill remaining slots with intermediate or unused views.
        for entry in overviews[2:] + patches:
            add(entry)

        if not overviews:
            selected.sort(key=lambda item: (-item["scale"], item["group_id"]))
        return selected
