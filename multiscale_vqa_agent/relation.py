from typing import Any, Dict, List

import numpy as np

from .g2p_runtime import MultiScaleG2PAgent
from .registry import ToolBankRegistry


class RelationReasoningAgent:
    """Ranks patient-aware phenotype <- program <- gene relation paths."""

    def __init__(self, registry: ToolBankRegistry, g2p_agent: MultiScaleG2PAgent, config: Dict[str, Any]):
        self.registry = registry
        self.g2p_agent = g2p_agent
        self.top_programs = int(config.get("top_programs", 3))
        self.genes_per_program = int(config.get("genes_per_program", 2))

    def reason(self, field: str, scale_results: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
        phenotype_index = self.registry.field_to_index[field]
        scales = sorted(scale_results)
        learned = np.stack([
            self.g2p_agent.runtimes[s].relations["R_program_to_phenotype"][:, phenotype_index]
            for s in scales
        ])
        initial = np.stack([
            self.g2p_agent.runtimes[s].relations.get(
                "R_initial", self.g2p_agent.runtimes[s].relations["R_prior"]
            )[:, phenotype_index]
            for s in scales
        ])
        prior = np.stack([
            self.g2p_agent.runtimes[s].relations["R_prior"][:, phenotype_index]
            for s in scales
        ])
        gates = np.stack([
            self.g2p_agent.runtimes[s].relations.get(
                "R_relation_gate", np.ones_like(learned[0])
            )[:, phenotype_index]
            for s in scales
        ])
        patient_program = np.stack([scale_results[s]["program_pred"] for s in scales])
        mean_learned = learned.mean(axis=0)
        mean_initial = initial.mean(axis=0)
        mean_prior = prior.mean(axis=0)
        mean_gate = gates.mean(axis=0)
        delta = mean_learned - mean_initial
        relation_consensus = np.clip(1.0 - learned.std(axis=0) / (np.abs(mean_learned) + 1e-6), 0.0, 1.0)
        patient_activity = np.mean(np.abs(patient_program), axis=0)
        activity_factor = 1.0 + np.minimum(patient_activity, 3.0) / 3.0
        scores = np.abs(mean_learned) * (0.5 + 0.5 * relation_consensus) * activity_factor
        program_indices = np.argsort(scores)[::-1][: self.top_programs]
        programs = []
        selected_gene_scores: Dict[int, float] = {}
        for program_index in program_indices:
            relation_type = self._relation_type(
                float(mean_prior[program_index]),
                float(mean_initial[program_index]),
                float(mean_learned[program_index]),
                float(mean_gate[program_index]),
            )
            program = {
                "index": int(program_index),
                "name": self.registry.programs[program_index],
                "score": float(scores[program_index]),
                "prior": float(mean_prior[program_index]),
                "initial": float(mean_initial[program_index]),
                "learned": float(mean_learned[program_index]),
                "change": float(delta[program_index]),
                "gate": float(mean_gate[program_index]),
                "scale_consensus": float(relation_consensus[program_index]),
                "relation_type": relation_type,
                "patient_score": float(patient_program[:, program_index].mean()),
                "per_scale": {
                    str(scale): {
                        "learned": float(learned[i, program_index]),
                        "initial": float(initial[i, program_index]),
                        "patient_score": float(patient_program[i, program_index]),
                    }
                    for i, scale in enumerate(scales)
                },
            }
            gene_rows = self._rank_genes(program_index, scale_results, float(scores[program_index]))
            program["genes"] = gene_rows[: self.genes_per_program]
            for gene in program["genes"]:
                selected_gene_scores[gene["index"]] = max(
                    selected_gene_scores.get(gene["index"], 0.0), gene["score"]
                )
            programs.append(program)
        genes = [
            {"index": index, "name": self.registry.genes[index], "score": score}
            for index, score in sorted(selected_gene_scores.items(), key=lambda item: item[1], reverse=True)
        ]
        return {
            "phenotype": field,
            "programs": programs,
            "genes": genes,
            "selection_policy": {
                "top_programs": self.top_programs,
                "genes_per_program": self.genes_per_program,
                "uses": ["R_prior", "R_initial", "R_learned", "relation_gate", "patient_activity", "scale_consensus"],
            },
        }

    def _rank_genes(
        self, program_index: int, scale_results: Dict[int, Dict[str, Any]], program_score: float
    ) -> List[Dict[str, Any]]:
        scales = sorted(scale_results)
        h_values = np.stack([
            self.g2p_agent.runtimes[s].relations["H_gene_to_program"][:, program_index]
            for s in scales
        ])
        patient_gene = np.stack([scale_results[s]["gene_pred"] for s in scales])
        mean_h = h_values.mean(axis=0)
        h_consensus = np.clip(1.0 - h_values.std(axis=0) / (np.abs(mean_h) + 1e-6), 0.0, 1.0)
        gene_activity = np.mean(np.abs(patient_gene), axis=0)
        score = np.abs(mean_h) * (0.5 + 0.5 * h_consensus) * (1.0 + np.minimum(gene_activity, 3.0) / 3.0)
        score *= program_score
        return [
            {
                "index": int(index),
                "name": self.registry.genes[index],
                "score": float(score[index]),
                "gene_to_program": float(mean_h[index]),
                "scale_consensus": float(h_consensus[index]),
                "patient_score": float(patient_gene[:, index].mean()),
            }
            for index in np.argsort(score)[::-1]
        ]

    @staticmethod
    def _relation_type(prior: float, initial: float, learned: float, gate: float) -> str:
        is_prior = prior >= 0.5
        change = learned - initial
        if is_prior and learned < 0.2 and gate < 0.25:
            return "prior_removed"
        if is_prior and change > 0.05:
            return "prior_strengthened"
        if is_prior and change < -0.05:
            return "prior_weakened"
        if is_prior:
            return "prior_retained"
        if learned >= 0.55 and change > 0.05:
            return "new_relation"
        return "weak_or_background"
