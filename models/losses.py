import torch
import torch.nn.functional as F


def masked_mean(losses, mask):
    mask = mask.float()
    denom = mask.sum().clamp_min(1.0)
    return (losses * mask).sum() / denom


def phenotype_loss(
    outputs,
    targets,
    masks,
    phenotype_specs,
    group_weights=None,
    task_weights=None,
    survival_pos_weights=None,
):
    group_weights = group_weights or {"molecular": 1.0, "morphology": 1.0, "clinical": 1.0}
    task_weights = task_weights or {}
    survival_pos_weights = survival_pos_weights or {}
    if targets.dim() == 2:
        targets = targets.unsqueeze(0)
        masks = masks.unsqueeze(0)
    total = targets.new_tensor(0.0)
    weight_sum = targets.new_tensor(0.0)
    per_task = {}
    group_totals = {}
    group_counts = {}
    for i, spec in enumerate(phenotype_specs):
        mask_i = masks[:, i]
        if mask_i.sum().item() <= 0:
            continue
        group = spec.get("group", "morphology")
        group_weight = float(group_weights.get(group, 1.0))
        task_weight = float(task_weights.get(spec["name"], 1.0))
        combined_weight = group_weight * task_weight
        if combined_weight <= 0:
            continue
        logit = outputs["phenotype_logits"][i]
        y = targets[:, i]
        if spec["task_type"] == "discrete_survival":
            n_bins = int(spec.get("num_bins", y.shape[-1]))
            hazard_logits = logit.reshape(-1, n_bins)
            bin_mask = mask_i[:, :n_bins].float()
            event_targets = y[:, :n_bins].float()
            pos_weight = torch.as_tensor(
                survival_pos_weights.get(spec["name"], [1.0] * n_bins),
                dtype=hazard_logits.dtype,
                device=hazard_logits.device,
            )[:n_bins]
            log_likelihood = (
                pos_weight.view(1, -1) * event_targets * F.logsigmoid(hazard_logits)
                + (1.0 - event_targets) * F.logsigmoid(-hazard_logits)
            )
            per_sample = -(log_likelihood * bin_mask).sum(dim=-1)
            valid_sample = (bin_mask.sum(dim=-1) > 0).float()
            loss = masked_mean(per_sample, valid_sample)
        elif spec["task_type"] in {"binary", "survival"}:
            sample_mask = mask_i[:, 0].float()
            losses = F.binary_cross_entropy_with_logits(
                logit.reshape(-1), y[:, 0].float(), reduction="none"
            )
            loss = masked_mean(losses, sample_mask)
        elif spec["task_type"] == "regression":
            sample_mask = mask_i[:, 0].float()
            losses = F.huber_loss(logit.reshape(-1), y[:, 0], reduction="none")
            loss = masked_mean(losses, sample_mask)
        else:
            sample_mask = mask_i[:, 0].float()
            class_logits = logit.unsqueeze(0) if logit.dim() == 1 else logit
            losses = F.cross_entropy(class_logits, y[:, 0].long(), reduction="none")
            loss = masked_mean(losses, sample_mask)
        total = total + combined_weight * loss
        weight_sum = weight_sum + combined_weight
        per_task[spec["name"]] = float(loss.detach().cpu())
        group_totals[group] = group_totals.get(group, 0.0) + float(loss.detach().cpu())
        group_counts[group] = group_counts.get(group, 0) + 1
    group_losses = {g: group_totals[g] / max(group_counts[g], 1) for g in group_totals}
    return total / weight_sum.clamp_min(1.0), per_task, group_losses


def program_loss(outputs, targets, masks):
    losses = F.huber_loss(outputs["program_pred"], targets, reduction="none")
    return masked_mean(losses, masks)


def masked_pearson_loss(pred, target, mask, eps=1e-6):
    mask = mask.float()
    pred = pred.float()
    target = target.float()
    if pred.dim() == 1:
        if mask.sum() < 2:
            return pred.new_tensor(0.0), pred.new_tensor(0.0)
        denom = mask.sum().clamp_min(1.0)
        pred_mean = (pred * mask).sum() / denom
        target_mean = (target * mask).sum() / denom
        pred_centered = (pred - pred_mean) * mask
        target_centered = (target - target_mean) * mask
        pred_scale = pred_centered.pow(2).sum().clamp_min(eps * eps).sqrt()
        target_scale = target_centered.pow(2).sum().clamp_min(eps * eps).sqrt()
        corr = (pred_centered * target_centered).sum() / (
            pred_scale * target_scale + eps
        )
        return 1.0 - corr, corr

    counts = mask.sum(dim=0)
    denom = counts.clamp_min(1.0)
    pred_mean = (pred * mask).sum(dim=0) / denom
    target_mean = (target * mask).sum(dim=0) / denom
    pred_centered = (pred - pred_mean.unsqueeze(0)) * mask
    target_centered = (target - target_mean.unsqueeze(0)) * mask
    numerator = (pred_centered * target_centered).sum(dim=0)
    pred_ss = pred_centered.pow(2).sum(dim=0)
    target_ss = target_centered.pow(2).sum(dim=0)
    pred_scale = pred_ss.clamp_min(eps * eps).sqrt()
    target_scale = target_ss.clamp_min(eps * eps).sqrt()
    valid = (counts >= 2) & (pred_ss > eps * eps) & (target_ss > eps * eps)
    if not valid.any():
        return pred.new_tensor(0.0), pred.new_tensor(0.0)
    correlations = numerator[valid] / (pred_scale[valid] * target_scale[valid] + eps)
    corr = correlations.mean()
    return 1.0 - corr, corr


def gene_expression_loss(outputs, targets, masks, corr_weight=0.1):
    huber = F.huber_loss(outputs["gene_pred"], targets, reduction="none")
    huber = masked_mean(huber, masks)
    corr_loss, corr = masked_pearson_loss(outputs["gene_pred"], targets, masks)
    if corr_weight <= 0:
        return huber, huber, corr_loss.detach(), corr.detach()
    return huber + corr_weight * corr_loss, huber, corr_loss, corr




def rna_teacher_losses(outputs, targets, gene_masks):
    if "rna_gene_embeddings" not in outputs:
        zero = outputs["gene_embeddings"].new_tensor(0.0)
        return zero, zero
    teacher = outputs["rna_gene_embeddings"].detach()
    student = outputs["gene_embeddings"]
    valid = gene_masks.float()
    if valid.sum() < 1:
        align = student.new_tensor(0.0)
    else:
        cosine = F.cosine_similarity(student, teacher, dim=-1)
        align = ((1.0 - cosine) * valid).sum() / valid.sum().clamp_min(1.0)

    recon_mask = outputs.get("rna_recon_mask", torch.zeros_like(valid)).float() * valid
    if recon_mask.sum() < 1:
        recon = student.new_tensor(0.0)
    else:
        recon_losses = F.huber_loss(outputs["rna_recon_pred"], targets, reduction="none")
        recon = (recon_losses * recon_mask).sum() / recon_mask.sum().clamp_min(1.0)
    return align, recon


def gene_program_phenotype_loss(model, outputs):
    # R is now a signed pathway -> phenotype contribution matrix, not a 0/1
    # probability graph. The old gene->phenotype probabilistic path BCE forced
    # prior edges toward 1 and non-prior edges toward 0, which conflicts with
    # soft, learnable non-prior contributions. Keep this hook as a zero-valued
    # compatibility term so the training loop and logs do not need to change.
    zero = outputs["R"].new_tensor(0.0)
    return zero, zero

def soft_prior_margin_loss(pred, prior, pos_floor=0.25, neg_ceiling=0.10, neg_weight=0.05):
    pred = pred.clamp(1e-6, 1 - 1e-6)
    prior = prior.to(pred.device)
    pos = pred[prior > 0]
    neg = pred[prior <= 0]
    # Only gently discourage prior links from collapsing below a floor.
    # Non-prior links are allowed to grow if task losses support them; above the ceiling
    # they pay a small sparsity-like cost rather than a hard MSE-to-zero penalty.
    pos_loss = F.relu(pos_floor - pos).pow(2).mean() if pos.numel() else pred.new_tensor(0.0)
    neg_loss = F.relu(neg - neg_ceiling).pow(2).mean() if neg.numel() else pred.new_tensor(0.0)
    return pos_loss + neg_weight * neg_loss


def gene_program_prior_loss(model, outputs):
    # Gene -> pathway membership is fixed from the canonical pathway JSON.
    return outputs["gene_program_weights"].new_tensor(0.0)


def relation_prior_loss(model, outputs):
    h_loss = outputs["H"].new_tensor(0.0)
    r = outputs["R"]
    if getattr(model, "relation_selection_mode", "prior_guided") == "free_topk":
        return h_loss, r.new_tensor(0.0)
    prior = model.R_prior.to(r.device)
    pos = r[prior > 0]
    # Prior links are softly encouraged to remain positive, but are not pushed to
    # 1. Non-prior links are left to phenotype loss plus flexible sparsity.
    r_loss = F.relu(0.10 - pos).pow(2).mean() if pos.numel() else r.new_tensor(0.0)
    return h_loss, r_loss


def flexible_sparsity_loss(model, outputs):
    # Prior-guided mode only discourages broad non-prior edges. Free mode
    # applies the same high-confidence penalty to every relation.
    h_prior = model.H_prior.to(outputs["H"].device)
    r_prior = model.R_prior.to(outputs["R"].device)
    g_prior = model.gene_H_prior.to(outputs["gene_program_weights"].device)
    h_non = outputs["H"][h_prior <= 0]
    if getattr(model, "relation_selection_mode", "prior_guided") == "free_topk":
        r_non = outputs["R"].abs().reshape(-1)
    else:
        r_non = outputs["R"].abs()[r_prior <= 0]
    g_non = outputs["gene_program_weights"][g_prior <= 0]
    h_loss = F.relu(h_non - 0.35).pow(2).mean() if h_non.numel() else outputs["H"].new_tensor(0.0)
    r_loss = F.relu(r_non - 0.75).pow(2).mean() if r_non.numel() else outputs["R"].new_tensor(0.0)
    g_loss = g_non.new_tensor(0.0) if g_non.numel() else outputs["gene_program_weights"].new_tensor(0.0)
    return h_loss + r_loss + g_loss


def attention_entropy(attn):
    return (-(attn.clamp_min(1e-8) * attn.clamp_min(1e-8).log()).sum(dim=-1)).mean()


def attention_diversity(attn):
    if attn.dim() == 2:
        attn = attn.unsqueeze(0)
    if attn.shape[1] < 2:
        return attn.new_tensor(0.0)
    norm = F.normalize(attn, dim=-1)
    sim = torch.matmul(norm, norm.transpose(-1, -2))
    eye = torch.eye(sim.shape[-1], device=sim.device, dtype=sim.dtype).unsqueeze(0)
    denom = max(sim.shape[0] * (sim.shape[-1] * sim.shape[-1] - sim.shape[-1]), 1)
    return ((sim * (1 - eye)).sum() / denom).abs()


def total_loss(model, outputs, batch, phenotype_specs, weights):
    group_weights = weights.get("phenotype_group_weights", None)
    lp, _, group_losses = phenotype_loss(
        outputs,
        batch["phenotype_targets"],
        batch["phenotype_masks"],
        phenotype_specs,
        group_weights,
        task_weights=weights.get("phenotype_task_weights"),
        survival_pos_weights=weights.get("survival_pos_weights"),
    )
    lg = program_loss(outputs, batch["program_targets"], batch["program_masks"])
    l_gene, l_gene_huber, l_gene_corr_loss, gene_corr = gene_expression_loss(
        outputs,
        batch["gene_targets"],
        batch["gene_masks"],
        corr_weight=weights.get("lambda_gene_corr", 0.1),
    )
    l_rna_align, l_rna_recon = rna_teacher_losses(outputs, batch["gene_targets"], batch["gene_masks"])
    if weights.get("lambda_align", 0.0) > 0:
        l_gene_pheno, l_gene_supported_r = gene_program_phenotype_loss(model, outputs)
        l_align = F.binary_cross_entropy_with_logits(model.alignment_logits(), model.R_prior)
    else:
        l_align = outputs["R"].new_tensor(0.0)
        l_gene_pheno = outputs["R"].new_tensor(0.0)
        l_gene_supported_r = outputs["R"].new_tensor(0.0)
    if weights.get("lambda_prior", 0.0) > 0:
        l_gene_prior = gene_program_prior_loss(model, outputs)
        l_h_prior, l_r_prior = relation_prior_loss(model, outputs)
        l_prior = l_h_prior + l_r_prior + l_gene_prior
    else:
        l_gene_prior = outputs["R"].new_tensor(0.0)
        l_h_prior = outputs["R"].new_tensor(0.0)
        l_r_prior = outputs["R"].new_tensor(0.0)
        l_prior = outputs["R"].new_tensor(0.0)
    l_query_delta = model.gene_query_delta_regularization() if hasattr(model, "gene_query_delta_regularization") else outputs["R"].new_tensor(0.0)
    l_sparse = flexible_sparsity_loss(model, outputs)
    l_attn = attention_entropy(outputs["program_attention"]) + attention_entropy(outputs["phenotype_attention"])
    l_div = attention_diversity(outputs["program_attention"]) + attention_diversity(outputs["phenotype_attention"])
    total = (
        weights.get("lambda_phenotype", 1.0) * lp
        + weights.get("lambda_gene", 0.0) * l_gene
        + weights.get("lambda_rna_align", 0.0) * l_rna_align
        + weights.get("lambda_rna_recon", 0.0) * l_rna_recon
        + weights.get("lambda_gene_query_delta", 0.0) * l_query_delta
        + weights["lambda_program"] * lg
        + weights["lambda_align"] * (l_align + l_gene_pheno + 0.5 * l_gene_supported_r)
        + weights["lambda_prior"] * l_prior
        + weights["lambda_sparse"] * (l_sparse + l_attn)
        + weights["lambda_diversity"] * l_div
    )
    return total, {
        "phenotype": float(lp.detach().cpu()),
        "phenotype_molecular": float(group_losses.get("molecular", 0.0)),
        "phenotype_morphology": float(group_losses.get("morphology", 0.0)),
        "phenotype_clinical": float(group_losses.get("clinical", 0.0)),
        "program": float(lg.detach().cpu()),
        "gene": float(l_gene.detach().cpu()),
        "gene_huber": float(l_gene_huber.detach().cpu()),
        "gene_corr_loss": float(l_gene_corr_loss.detach().cpu()),
        "gene_corr": float(gene_corr.detach().cpu()),
        "rna_align": float(l_rna_align.detach().cpu()),
        "rna_recon": float(l_rna_recon.detach().cpu()),
        "gene_query_delta": float(l_query_delta.detach().cpu()),
        "align": float(l_align.detach().cpu()),
        "gene_pheno": float(l_gene_pheno.detach().cpu()),
        "gene_supported_r": float(l_gene_supported_r.detach().cpu()),
        "prior": float(l_prior.detach().cpu()),
        "gene_prior": float(l_gene_prior.detach().cpu()),
        "h_prior": float(l_h_prior.detach().cpu()),
        "r_prior": float(l_r_prior.detach().cpu()),
        "sparse": float(l_sparse.detach().cpu()),
        "attn_entropy": float(l_attn.detach().cpu()),
        "diversity": float(l_div.detach().cpu()),
        "total": float(total.detach().cpu()),
    }
