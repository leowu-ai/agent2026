import json
from typing import Any, Dict, List

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
    ) -> Dict[str, Any]:
        image_paths = [
            patch.image_path
            for group in groups
            for _, patch in sorted(group.patches.items(), reverse=True)
            if patch.image_path
        ]
        evidence = [group.to_dict() for group in groups]
        if not self.client.enabled:
            return {
                "backend": "mock",
                "description": "Patho-R1 was not called; evidence patches were selected successfully.",
                "image_count": len(image_paths),
                "evidence_groups": evidence,
            }
        system = (
            "You are a pathology vision evidence agent. Review images in coarse-to-fine order: "
            "4096 for architecture, 2048 for regional context, and 1024 for cellular detail. "
            "Describe only visible morphology. Never claim that a patch truly expresses a gene or pathway. "
            "State whether scale-level observations agree and whether evidence is sufficient."
        )
        user = json.dumps({
            "question": question,
            "target_phenotype": field,
            "image_order": [
                {
                    "group_id": group.group_id,
                    "scale": scale,
                    "sources": patch.sources,
                    "image_path": patch.image_path,
                }
                for group in groups
                for scale, patch in sorted(group.patches.items(), reverse=True)
                if patch.image_path
            ],
        }, ensure_ascii=False)
        try:
            description = self.client.chat(system, user, images=image_paths, max_tokens=384)
        except Exception as error:
            return {
                "backend": "pathor1_error",
                "description": f"Patho-R1 request failed: {type(error).__name__}: {error}",
                "image_count": len(image_paths),
                "evidence_groups": evidence,
            }
        return {
            "backend": "pathor1",
            "description": description,
            "image_count": len(image_paths),
            "evidence_groups": evidence,
        }
