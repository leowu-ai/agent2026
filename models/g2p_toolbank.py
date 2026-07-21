import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class G2PHypergraphToolBank(nn.Module):
    def __init__(
        self,
        feature_dim,
        hidden_dim,
        phenotype_specs,
        num_genes,
        program_names,
        H_prior,
        R_prior,
        dropout=0.25,
        gene_phenotype_prior=None,
        gene_names=None,
        rna_mask_ratio=0.15,
        rna_encoder_layers=2,
        rna_encoder_heads=4,
        relation_init_mode="prior",
        relation_init_value=0.5,
        relation_selection_mode="prior_guided",
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.phenotype_specs = phenotype_specs
        self.program_names = program_names
        self.num_genes = num_genes
        self.num_programs = len(program_names)
        self.num_phenotypes = len(phenotype_specs)
        self.rna_mask_ratio = rna_mask_ratio
        self.phenotype_mode = "full"
        if relation_init_mode not in {"prior", "uniform"}:
            raise ValueError(f"Unsupported relation_init_mode: {relation_init_mode}")
        if relation_selection_mode not in {"prior_guided", "free_topk"}:
            raise ValueError(f"Unsupported relation_selection_mode: {relation_selection_mode}")
        if not -1.0 < float(relation_init_value) < 1.0:
            raise ValueError("relation_init_value must be strictly between -1 and 1")
        self.relation_init_mode = relation_init_mode
        self.relation_selection_mode = relation_selection_mode
        self.relation_init_value = float(relation_init_value)

        self.patch_projector = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.program_prototypes = nn.Parameter(torch.randn(self.num_programs, hidden_dim) * 0.02)
        self.phenotype_prototypes = nn.Parameter(torch.randn(self.num_phenotypes, hidden_dim) * 0.02)
        # Stable gene identity tokens are separated from WSI query prototypes.
        # The WSI query keeps a semantic link through projection(identity) + delta.
        self.gene_identity_embeddings = nn.Parameter(torch.randn(num_genes, hidden_dim) * 0.02)
        self.gene_query_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        nn.init.eye_(self.gene_query_projection.weight)
        self.gene_query_delta = nn.Parameter(torch.zeros(num_genes, hidden_dim))
        self.gene_query_norm = nn.LayerNorm(hidden_dim)

        self.rna_value_encoder = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.rna_mask_token = nn.Parameter(torch.randn(hidden_dim) * 0.02)
        self.rna_input_norm = nn.LayerNorm(hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=rna_encoder_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.rna_encoder = nn.TransformerEncoder(encoder_layer, num_layers=rna_encoder_layers)
        self.rna_reconstruction_head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 1))

        self.gene_program_context_update = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.alpha_gene_program_logit = nn.Parameter(torch.logit(torch.tensor(0.1, dtype=torch.float32)))
        self.pathway_fusion_gate = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.R_phenotype_update = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.alpha_r_logit = nn.Parameter(torch.logit(torch.tensor(0.1, dtype=torch.float32)))

        self.gene_head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 1))
        self.program_head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 1))
        self.pheno_heads = nn.ModuleList()
        for spec in phenotype_specs:
            if spec["task_type"] == "multiclass":
                out_dim = int(spec.get("num_classes", 1))
            elif spec["task_type"] == "discrete_survival":
                out_dim = int(spec.get("num_bins", 1))
            else:
                out_dim = 1
            self.pheno_heads.append(nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, out_dim)))

        gene_H_prior = torch.as_tensor(H_prior, dtype=torch.float32)
        R_prior = torch.as_tensor(R_prior, dtype=torch.float32)
        if gene_phenotype_prior is None:
            gene_phenotype_prior = torch.zeros((gene_H_prior.shape[0], R_prior.shape[1]), dtype=torch.float32)
        else:
            gene_phenotype_prior = torch.as_tensor(gene_phenotype_prior, dtype=torch.float32)
        self.register_buffer("gene_H_prior", gene_H_prior)
        self.register_buffer("gene_phenotype_prior", gene_phenotype_prior)
        # Fixed gene -> pathway membership. No learnable gene-to-pathway relation and
        # no pathway-to-gene message passing are used.
        self.register_buffer("H_prior", gene_H_prior)
        self.register_buffer("R_prior", R_prior)
        self.register_buffer("R_prior_mask", (R_prior > 0).float())

        # Initialize pathway -> phenotype relations from either the soft prior or
        # a uniform value while allowing signed residual updates through tanh(theta).
        eps = 1e-4
        if self.relation_init_mode == "uniform":
            R_init = torch.full_like(R_prior, self.relation_init_value)
        else:
            R_init = torch.where(
                R_prior > 0,
                torch.full_like(R_prior, 0.85),
                torch.full_like(R_prior, 0.30),
            )
        self.R_theta = nn.Parameter(torch.atanh(R_init.clamp(-1 + eps, 1 - eps)))
        # Iterative relation training keeps R_theta as the fixed prior anchor and
        # learns a residual. A zero scale preserves the legacy parameterization.
        self.R_delta = nn.Parameter(torch.zeros_like(R_init))
        self.register_buffer("relation_scale", torch.tensor(0.0, dtype=torch.float32))
        self.register_buffer("relation_gate", torch.ones_like(R_init))
        self.register_buffer("relation_ema", R_init.clone())
        self.register_buffer("relation_initial", R_init.clone())

    def gene_query_prototypes(self):
        return self.gene_query_norm(
            self.gene_query_projection(self.gene_identity_embeddings) + self.gene_query_delta
        )

    def gene_identity_tokens(self):
        return self.gene_identity_embeddings

    def gene_query_delta_regularization(self):
        return self.gene_query_delta.pow(2).mean()

    def gene_program_weights(self):
        return self.gene_H_prior

    def node_hypergraph_weights(self):
        return self.gene_H_prior

    def program_phenotype_weights(self):
        raw = torch.tanh(self.R_theta + self.relation_scale * self.R_delta)
        return raw * self.relation_gate

    def raw_program_phenotype_weights(self):
        return torch.tanh(self.R_theta + self.relation_scale * self.R_delta)

    @torch.no_grad()
    def set_relation_scale(self, scale):
        self.relation_scale.fill_(float(scale))

    @torch.no_grad()
    def reset_relation_gate(self):
        self.relation_gate.fill_(1.0)

    @torch.no_grad()
    def select_relation_edges(
        self, topk=6, new_edges=3, gate_floor=0.15, ema_decay=0.5,
        min_new_change=0.05, min_new_weight=0.35, selection_mode=None,
    ):
        """Select stable prior edges and newly learned non-prior residual edges."""
        topk = max(1, min(int(topk), self.num_programs))
        new_edges = max(0, min(int(new_edges), topk))
        gate_floor = float(min(max(gate_floor, 0.0), 1.0))
        ema_decay = float(min(max(ema_decay, 0.0), 1.0))
        min_new_change = float(max(min_new_change, 0.0))
        min_new_weight = float(min(max(min_new_weight, -1.0), 1.0))
        selection_mode = selection_mode or self.relation_selection_mode
        if selection_mode not in {"prior_guided", "free_topk"}:
            raise ValueError(f"Unsupported relation selection mode: {selection_mode}")

        raw = self.raw_program_phenotype_weights()
        self.relation_ema.mul_(ema_decay).add_(raw, alpha=1.0 - ema_decay)
        prior_mask = self.R_prior_mask.bool()
        new_score = self.relation_ema - self.relation_initial
        gate = torch.full_like(self.relation_gate, gate_floor)
        selected_prior = 0
        selected_new = 0
        eligible_new = 0

        for phenotype_index in range(self.num_phenotypes):
            if selection_mode == "free_topk":
                scores = self.relation_ema[:, phenotype_index].abs()
                chosen = torch.topk(scores, k=topk).indices
                gate[chosen, phenotype_index] = 1.0
                chosen_prior = prior_mask[chosen, phenotype_index]
                selected_prior += int(chosen_prior.sum().item())
                selected_new += int((~chosen_prior).sum().item())
                eligible_new += int((~chosen_prior).sum().item())
                continue

            prior_indices = torch.where(prior_mask[:, phenotype_index])[0]
            eligible_mask = (
                (~prior_mask[:, phenotype_index])
                & (new_score[:, phenotype_index] > min_new_change)
                & (self.relation_ema[:, phenotype_index] > min_new_weight)
            )
            eligible_indices = torch.where(eligible_mask)[0]
            eligible_new += int(eligible_indices.numel())
            new_slots = min(new_edges, int(eligible_indices.numel()))
            # When fewer new edges qualify, use remaining capacity for stable prior
            # edges. Weak non-prior edges are never used merely to fill top-k.
            prior_slots = min(topk - new_slots, int(prior_indices.numel()))

            if prior_slots > 0:
                scores = self.relation_ema[prior_indices, phenotype_index].abs()
                chosen = prior_indices[torch.topk(scores, k=prior_slots).indices]
                gate[chosen, phenotype_index] = 1.0
                selected_prior += int(chosen.numel())
            if new_slots > 0:
                scores = new_score[eligible_indices, phenotype_index]
                chosen = eligible_indices[torch.topk(scores, k=new_slots).indices]
                gate[chosen, phenotype_index] = 1.0
                selected_new += int(chosen.numel())

        self.relation_gate.copy_(gate)
        active = int((gate >= 1.0 - 1e-6).sum().item())
        attenuated_prior = int(((gate < 1.0 - 1e-6) & prior_mask).sum().item())
        mean_change = float((raw - self.relation_initial).abs().mean().item())
        return {
            "active_edges": active,
            "selected_prior_edges": selected_prior,
            "selected_new_edges": selected_new,
            "eligible_new_edges": eligible_new,
            "attenuated_prior_edges": attenuated_prior,
            "mean_abs_change": mean_change,
        }

    def alpha_gene_program(self):
        return 0.25 * torch.sigmoid(self.alpha_gene_program_logit)

    def alpha_r(self):
        return 0.5 * torch.sigmoid(self.alpha_r_logit)

    def effective_program_prototypes(self, gene_embeddings=None):
        if gene_embeddings is None:
            gene_embeddings = self.gene_query_prototypes()
        weights = self.gene_program_weights()
        weights = weights / weights.sum(dim=0, keepdim=True).clamp_min(1e-6)
        gene_context = weights.transpose(0, 1) @ gene_embeddings
        return self.program_prototypes + self.alpha_gene_program() * self.gene_program_context_update(gene_context)

    def _cross_attention(self, queries, patches, patch_mask=None):
        scale = math.sqrt(patches.shape[-1])
        if patches.dim() == 2:
            scores = queries @ patches.transpose(0, 1) / scale
            if patch_mask is not None:
                scores = scores.masked_fill(
                    ~patch_mask.bool().view(1, -1), torch.finfo(scores.dtype).min
                )
            attn = torch.softmax(scores, dim=-1)
            return attn @ patches, attn

        if queries.dim() == 2:
            queries = queries.unsqueeze(0).expand(patches.shape[0], -1, -1)
        scores = torch.matmul(queries, patches.transpose(-1, -2)) / scale
        if patch_mask is not None:
            scores = scores.masked_fill(
                ~patch_mask.bool().unsqueeze(1), torch.finfo(scores.dtype).min
            )
        attn = torch.softmax(scores, dim=-1)
        return torch.matmul(attn, patches), attn

    def _fixed_gene_to_pathway(self, z_gene):
        weights = self.gene_program_weights()
        weights = weights / weights.sum(dim=0, keepdim=True).clamp_min(1e-6)
        if z_gene.dim() == 3:
            return torch.einsum("gm,bgh->bmh", weights, z_gene)
        return weights.transpose(0, 1) @ z_gene

    def _fuse_pathway_embeddings(self, z_program_wsi, z_program_gene):
        gate = torch.sigmoid(self.pathway_fusion_gate(torch.cat([z_program_wsi, z_program_gene], dim=-1)))
        return z_program_wsi + gate * z_program_gene

    def _pathway_message(self, z_program, R):
        denom = R.abs().sum(dim=0).clamp_min(1.0)
        if z_program.dim() == 3:
            message = torch.einsum("mp,bmh->bph", R, z_program)
            return message / denom.view(1, -1, 1)
        return (R.transpose(0, 1) @ z_program) / denom.view(-1, 1)

    def _pathway_only_phenotype(self, z_program, R):
        msg_h = self._pathway_message(z_program, R)
        return self.R_phenotype_update(msg_h)

    def _pathway_to_phenotype_message(self, z_program, z_pheno, R):
        msg_h = self._pathway_message(z_program, R)
        return z_pheno + self.alpha_r() * self.R_phenotype_update(msg_h)

    def _sample_rna_mask(self, valid_mask):
        valid_mask = valid_mask.bool()
        if valid_mask.sum() < 2:
            return valid_mask.clone()
        rand = torch.rand_like(valid_mask.float())
        mask = (rand < self.rna_mask_ratio) & valid_mask
        if mask.sum() == 0:
            valid_idx = torch.where(valid_mask)[0]
            chosen = valid_idx[torch.randint(valid_idx.numel(), (1,), device=valid_idx.device)]
            mask[chosen] = True
        return mask

    def encode_rna_teacher(self, gene_expression, gene_mask=None):
        single_sample = gene_expression.dim() == 1
        if single_sample:
            gene_expression = gene_expression.unsqueeze(0)
        if gene_mask is None:
            gene_mask = torch.ones_like(gene_expression)
        elif gene_mask.dim() == 1:
            gene_mask = gene_mask.unsqueeze(0)
        gene_expression = gene_expression.float()
        gene_mask = gene_mask.float()
        reconstruct_mask = torch.stack([
            self._sample_rna_mask(valid > 0) for valid in gene_mask
        ], dim=0)
        visible_expression = gene_expression.masked_fill(reconstruct_mask, 0.0)
        value_tokens = self.rna_value_encoder(visible_expression.unsqueeze(-1))
        mask_tokens = reconstruct_mask.float().unsqueeze(-1) * self.rna_mask_token.view(1, 1, -1)
        identity_tokens = self.gene_identity_tokens().unsqueeze(0)
        tokens = self.rna_input_norm(identity_tokens + value_tokens + mask_tokens)
        padding_mask = gene_mask <= 0
        encoded = self.rna_encoder(tokens, src_key_padding_mask=padding_mask)
        recon = self.rna_reconstruction_head(encoded).squeeze(-1)
        if single_sample:
            return encoded.squeeze(0), recon.squeeze(0), reconstruct_mask.squeeze(0).float()
        return encoded, recon, reconstruct_mask.float()

    def forward(self, features, gene_expression=None, gene_mask=None, patch_mask=None):
        batched = features.dim() == 3
        patches = self.patch_projector(features.float())
        gene_program_weights = self.gene_program_weights()
        gene_queries = self.gene_query_prototypes()
        z_gene, gene_attention = self._cross_attention(gene_queries, patches, patch_mask)
        gene_pred = self.gene_head(z_gene).squeeze(-1)

        program_queries = self.effective_program_prototypes(gene_queries)
        z_program_wsi, program_attention = self._cross_attention(program_queries, patches, patch_mask)
        z_program_gene = self._fixed_gene_to_pathway(z_gene)
        z_program = self._fuse_pathway_embeddings(z_program_wsi, z_program_gene)

        R = self.program_phenotype_weights()
        if self.phenotype_mode == "pathway_only":
            z_pheno = self._pathway_only_phenotype(z_program, R)
            if batched:
                phenotype_attention = patches.new_zeros(
                    patches.shape[0], self.num_phenotypes, patches.shape[1]
                )
            else:
                phenotype_attention = patches.new_zeros(self.num_phenotypes, patches.shape[0])
        else:
            z_pheno, phenotype_attention = self._cross_attention(
                self.phenotype_prototypes, patches, patch_mask
            )
            z_pheno = self._pathway_to_phenotype_message(z_program, z_pheno, R)

        program_pred = self.program_head(z_program).squeeze(-1)
        if batched:
            pheno_logits = [head(z_pheno[:, i]).squeeze(-1) for i, head in enumerate(self.pheno_heads)]
        else:
            pheno_logits = [head(z_pheno[i]).squeeze(0) for i, head in enumerate(self.pheno_heads)]
        outputs = {
            "gene_pred": gene_pred,
            "gene_attention": gene_attention,
            "program_pred": program_pred,
            "program_attention": program_attention,
            "phenotype_logits": pheno_logits,
            "phenotype_attention": phenotype_attention,
            "H": gene_program_weights,
            "R": R,
            "gene_program_weights": gene_program_weights,
            "gene_embeddings": z_gene,
            "gene_query_embeddings": gene_queries,
            "gene_identity_embeddings": self.gene_identity_tokens(),
            "program_embeddings": z_program,
            "phenotype_embeddings": z_pheno,
            "phenotype_mode": self.phenotype_mode,
        }
        if gene_expression is not None:
            rna_embeddings, rna_recon, rna_recon_mask = self.encode_rna_teacher(gene_expression, gene_mask)
            outputs.update({
                "rna_gene_embeddings": rna_embeddings,
                "rna_recon_pred": rna_recon,
                "rna_recon_mask": rna_recon_mask,
            })
        return outputs

    def alignment_logits(self, tau=0.2):
        pp = F.normalize(self.effective_program_prototypes(), dim=-1)
        hp = F.normalize(self.phenotype_prototypes, dim=-1)
        return pp @ hp.t() / tau
