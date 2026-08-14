import json
from typing import Any, Dict, List, Optional

from .clients import OpenAICompatibleClient
from .schemas import EvidenceGroup


class PathologyAgent:
    def __init__(self, client: OpenAICompatibleClient):
        self.client = client

    def describe(
        self,
        question: str,
        field: str,
        groups: List[EvidenceGroup],
        overview_paths: Optional[List[str]] = None,
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
                    self._entry_metadata(entry, index)
                    for index, entry in enumerate(entries, 1)
                ],
            }

        system = (
            "You are a pathology vision evidence agent. Review the supplied images from coarse "
            "architecture to cellular detail. Describe only morphology directly visible in the images. "
            "Do not infer RNA, genes, pathways, receptor status, IHC, FISH, mutations, treatment, or "
            "clinical records. Do not output hidden reasoning or think tags. Give a concise observation, "
            "state whether scales agree, and state whether the visual evidence answers the question."
        )
        attempts = self._attempt_schedule(len(entries))
        failures = []
        for attempt_index, (image_limit, image_max_size) in enumerate(attempts, 1):
            selected = self._select_entries(entries, image_limit)
            user = json.dumps({
                "question": question,
                "image_order": [
                    self._entry_metadata(entry, index)
                    for index, entry in enumerate(selected, 1)
                ],
                "evidence_rule": (
                    "Use image pixels only. Retrieval provenance and molecular predictions are hidden "
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
                )
                if not description:
                    raise RuntimeError("Patho-R1 returned an empty response")
                return {
                    "backend": "pathor1",
                    "description": description,
                    "image_count": len(selected),
                    "available_image_count": len(entries),
                    "request_attempts": attempt_index,
                    "image_max_size": image_max_size,
                    "retry_history": failures,
                    "evidence_groups": evidence,
                    "image_metadata": [
                        self._entry_metadata(entry, index)
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
                self._entry_metadata(entry, index)
                for index, entry in enumerate(
                    self._select_entries(entries, attempts[-1][0]), 1
                )
            ] if attempts else [],
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
                "image_path": patch.image_path,
            }
            for group in groups
            for scale, patch in sorted(group.patches.items(), reverse=True)
            if patch.image_path
        ]
        return overviews + patches

    @staticmethod
    def _entry_metadata(entry: Dict[str, Any], ordinal: int) -> Dict[str, Any]:
        result = {"ordinal": ordinal, "kind": entry["kind"]}
        if entry["kind"] == "patch":
            result.update({
                "group_id": entry["group_id"],
                "scale": entry["scale"],
            })
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
