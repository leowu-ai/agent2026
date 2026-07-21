#!/usr/bin/env python3
import argparse
import csv
import json
import os
import random
import shutil
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, roc_auc_score
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

sys.path.append(str(Path(__file__).resolve().parents[1]))
from data.dataset import G2PBRCAFeatureDataset, build_metadata, infer_feature_dim, parse_binary, prepare_manifest
from models.g2p_toolbank import G2PHypergraphToolBank
from models.losses import total_loss
from utils.device import select_device


FEATURE_DIR_BY_SIZE = {
    "1024": "/data_nas2/ljs/Share/TCGA_Embed/TCGA-BRCA/clam_gen_1024/conch_v1_5_new",
    "2048": "/data_nas2/ljs/Share/Share_Embedd_CLAM/TCGA-BRCA/clam_gen_2048/conch_v1_5",
    "4096": "/data_nas2/ljs/Share/Share_Embedd_CLAM/TCGA-BRCA/clam_gen_4096/conch_v1_5",
}


def resolve_feature_dir(args):
    if args.feature_dir:
        return args.feature_dir
    if args.feature_size:
        return FEATURE_DIR_BY_SIZE[args.feature_size]
    raise ValueError("Either --feature_dir or --feature_size must be provided.")


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def one_item(batch):
    return batch[0]


def mil_collate(items):
    batch_size = len(items)
    max_patches = max(item["features"].shape[0] for item in items)
    feature_dim = items[0]["features"].shape[1]
    features = items[0]["features"].new_zeros(batch_size, max_patches, feature_dim)
    patch_masks = torch.zeros(batch_size, max_patches, dtype=torch.bool)
    for i, item in enumerate(items):
        n_patches = item["features"].shape[0]
        features[i, :n_patches] = item["features"]
        patch_masks[i, :n_patches] = True

    tensor_keys = (
        "phenotype_targets", "phenotype_masks", "program_targets", "program_masks",
        "gene_targets", "gene_masks",
    )
    batch = {key: torch.stack([item[key] for item in items], dim=0) for key in tensor_keys}
    batch.update({
        "features": features,
        "patch_masks": patch_masks,
        "patch_count": torch.tensor([item["patch_count"] for item in items], dtype=torch.long),
        "original_patch_count": torch.tensor(
            [item["original_patch_count"] for item in items], dtype=torch.long
        ),
        "case_id": [item["case_id"] for item in items],
        "slide_id": [item["slide_id"] for item in items],
        "feature_path": [item["feature_path"] for item in items],
        "coords": [item["coords"] for item in items],
    })
    return batch



def move_batch(batch, device):
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device) if torch.is_tensor(v) else v
    return out




def setup_distributed():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return False, 0, 1, None
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    try:
        dist.init_process_group(backend="nccl", device_id=torch.device(f"cuda:{local_rank}"))
    except TypeError:
        dist.init_process_group(backend="nccl")
    return True, dist.get_rank(), dist.get_world_size(), local_rank


def cleanup_distributed(distributed):
    if distributed and dist.is_initialized():
        dist.destroy_process_group()


def reduce_mean(value, device, distributed):
    tensor = torch.tensor(float(value), device=device)
    if distributed:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor /= dist.get_world_size()
    return float(tensor.detach().cpu())


def all_ranks_finite(value, distributed):
    flag = torch.isfinite(value).all().to(dtype=torch.int32, device=value.device)
    if distributed:
        dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    return bool(flag.item())




def _set_trainable(obj, trainable):
    if obj is None:
        return
    if isinstance(obj, torch.nn.Parameter):
        obj.requires_grad_(trainable)
        return
    for param in obj.parameters():
        param.requires_grad_(trainable)


def apply_stage_freezing(model, stage_name):
    # Start permissive, then freeze modules that should not move in the current stage.
    for param in model.parameters():
        param.requires_grad_(True)

    if stage_name == "gene_pretrain":
        freeze_items = [
            model.program_prototypes,
            model.phenotype_prototypes,
            model.gene_program_context_update,
            model.alpha_gene_program_logit,
            model.pathway_fusion_gate,
            model.R_phenotype_update,
            model.alpha_r_logit,
            model.program_head,
            model.pheno_heads,
            model.R_theta,
            model.R_delta,
        ]
    elif stage_name == "gene_program":
        freeze_items = [
            model.phenotype_prototypes,
            model.pheno_heads,
            model.R_theta,
            model.R_phenotype_update,
            model.alpha_r_logit,
            model.R_delta,
        ]
    elif stage_name in {"relation_warmup", "relation_iter"}:
        # Learn pathway -> phenotype relations against a fixed WSI gene/pathway
        # representation so phenotype gradients cannot erase the gene predictor.
        freeze_items = [
            model.patch_projector,
            model.gene_identity_embeddings,
            model.gene_query_projection,
            model.gene_query_delta,
            model.gene_query_norm,
            model.gene_head,
            model.rna_value_encoder,
            model.rna_input_norm,
            model.rna_encoder,
            model.rna_reconstruction_head,
            model.rna_mask_token,
            model.program_prototypes,
            model.gene_program_context_update,
            model.alpha_gene_program_logit,
            model.pathway_fusion_gate,
            model.program_head,
            model.phenotype_prototypes,
        ]
        if stage_name == "relation_iter":
            freeze_items.append(model.R_theta)
    elif stage_name == "relation_adapt":
        # Keep learned gene and pathway representations fixed; adapt only
        # the phenotype side while relation parameters remain fixed.
        freeze_items = [
            model.patch_projector,
            model.gene_identity_embeddings,
            model.gene_query_projection,
            model.gene_query_delta,
            model.gene_query_norm,
            model.gene_head,
            model.rna_value_encoder,
            model.rna_input_norm,
            model.rna_encoder,
            model.rna_reconstruction_head,
            model.rna_mask_token,
            model.program_prototypes,
            model.gene_program_context_update,
            model.alpha_gene_program_logit,
            model.pathway_fusion_gate,
            model.program_head,
            model.R_theta,
            model.R_delta,
        ]
    elif stage_name in {"full", "iterative_full"}:
        # Keep the learned gene and pathway representations fixed during
        # phenotype fine-tuning so phenotype gradients cannot erode them.
        freeze_items = [
            model.patch_projector,
            model.gene_identity_embeddings,
            model.gene_query_projection,
            model.gene_query_delta,
            model.gene_query_norm,
            model.gene_head,
            model.rna_value_encoder,
            model.rna_input_norm,
            model.rna_encoder,
            model.rna_reconstruction_head,
            model.rna_mask_token,
            model.program_prototypes,
            model.gene_program_context_update,
            model.alpha_gene_program_logit,
            model.pathway_fusion_gate,
            model.program_head,
            model.R_theta,
            model.R_delta,
        ]
    else:
        # Preserve the previous standard-schedule behavior.
        freeze_items = [
            model.rna_value_encoder,
            model.rna_input_norm,
            model.rna_encoder,
            model.rna_reconstruction_head,
            model.rna_mask_token,
            model.gene_identity_embeddings,
        ]

    for item in freeze_items:
        _set_trainable(item, False)


def trainable_parameter_summary(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def set_phenotype_mode(model, stage_name):
    model.phenotype_mode = "pathway_only" if stage_name in {"relation_warmup", "relation_iter"} else "full"



def phenotype_group_weights_for_epoch(epoch, total_epochs, mode="staged"):
    if mode == "flat":
        return {"molecular": 1.0, "morphology": 1.0, "clinical": 1.0}
    if mode == "molecular_first":
        return {"molecular": 1.0, "morphology": 0.25, "clinical": 0.0}
    if mode == "no_clinical":
        return {"molecular": 1.0, "morphology": 1.0, "clinical": 0.0}
    # Default staged curriculum: molecular first, then morphology, then clinical with lower weight.
    if total_epochs <= 1:
        return {"molecular": 1.0, "morphology": 1.0, "clinical": 0.5}
    progress = (epoch - 1) / max(total_epochs - 1, 1)
    if progress < 0.34:
        return {"molecular": 1.0, "morphology": 0.25, "clinical": 0.0}
    if progress < 0.67:
        return {"molecular": 1.0, "morphology": 1.0, "clinical": 0.15}
    return {"molecular": 1.0, "morphology": 1.0, "clinical": 0.5}


def compute_survival_pos_weights(dataset, phenotype_specs, max_weight=6.0):
    case_ids = dataset.manifest["case_id"].drop_duplicates().tolist()
    result = {}
    for index, spec in enumerate(phenotype_specs):
        if spec["task_type"] != "discrete_survival":
            continue
        n_bins = int(spec.get("num_bins", 1))
        positives = np.zeros(n_bins, dtype=np.float64)
        valid_counts = np.zeros(n_bins, dtype=np.float64)
        for case_id in case_ids:
            targets, masks = dataset._phenotype_targets(case_id)
            y = targets[index, :n_bins].numpy()
            mask = masks[index, :n_bins].numpy()
            positives += y * mask
            valid_counts += mask
        negatives = valid_counts - positives
        weights = np.sqrt(negatives / np.maximum(positives, 1.0))
        weights = np.clip(weights, 1.0, float(max_weight))
        result[spec["name"]] = weights.astype(np.float32).tolist()
    return result


def format_survival_pos_weights(pos_weights):
    return ";".join(
        f"{name}=[" + ",".join(f"{value:.3g}" for value in values) + "]"
        for name, values in pos_weights.items()
    )


def format_group_weights(group_weights):
    return ",".join(f"{k}={v:g}" for k, v in group_weights.items())

def metric_safe(task_type, y_true, y_pred):
    if len(y_true) < 2:
        return 'NA', float('nan')
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    try:
        if task_type in {'binary', 'survival'}:
            if len(set(y_true.tolist())) < 2:
                return 'AUC', float('nan')
            return 'AUC', float(roc_auc_score(y_true, y_pred))
        if task_type == 'multiclass':
            return 'ACC', float(accuracy_score(y_true, y_pred))
        if task_type == 'regression':
            return 'MAE', float(mean_absolute_error(y_true, y_pred))
    except Exception:
        return 'NA', float('nan')
    return 'NA', float('nan')


def pearson_safe(y_true, y_pred):
    if len(y_true) < 3:
        return float('nan')
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if np.nanstd(y_true) < 1e-8 or np.nanstd(y_pred) < 1e-8:
        return float('nan')
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def iterative_relation_context(epoch, total_epochs, args):
    gene_pretrain_end = min(args.iterative_gene_pretrain_epochs, total_epochs)
    gene_program_end = min(
        gene_pretrain_end + args.iterative_gene_program_epochs, total_epochs
    )
    if epoch <= gene_pretrain_end:
        return "gene_pretrain", 0, 0.0
    if epoch <= gene_program_end:
        return "gene_program", 0, 0.0

    cycle_length = args.iterative_relation_epochs + args.iterative_adapt_epochs
    cycle_epoch = epoch - gene_program_end - 1
    iterative_epochs = args.iterative_relation_rounds * cycle_length
    if cycle_epoch < iterative_epochs:
        round_index = cycle_epoch // cycle_length + 1
        within_round = cycle_epoch % cycle_length
        scale = min(
            args.iterative_relation_scale_start
            * (args.iterative_relation_scale_multiplier ** (round_index - 1)),
            args.iterative_relation_scale_max,
        )
        stage = (
            "relation_iter"
            if within_round < args.iterative_relation_epochs
            else "relation_adapt"
        )
        return stage, round_index, scale
    return "iterative_full", args.iterative_relation_rounds, args.iterative_relation_scale_max


def iterative_stage_weights(stage_name, args):
    if stage_name == "gene_pretrain":
        return {
            "lambda_phenotype": 0.0, "lambda_gene": args.lambda_gene_stage1,
            "lambda_program": 0.0, "lambda_align": 0.0, "lambda_prior": 0.0,
            "lambda_sparse": 0.0, "lambda_diversity": 0.0,
        }
    if stage_name == "gene_program":
        return {
            "lambda_phenotype": 0.0, "lambda_gene": args.lambda_gene_stage2,
            "lambda_program": args.lambda_program, "lambda_align": 0.0,
            "lambda_prior": args.lambda_prior * 0.5,
            "lambda_sparse": args.lambda_sparse, "lambda_diversity": args.lambda_diversity,
        }
    if stage_name == "relation_iter":
        return {
            "lambda_phenotype": 1.0, "lambda_gene": 0.0, "lambda_program": 0.0,
            "lambda_align": 0.0, "lambda_prior": 0.0,
            "lambda_sparse": args.lambda_sparse, "lambda_diversity": args.lambda_diversity,
        }
    return {
        "lambda_phenotype": 1.0, "lambda_gene": 0.0,
        "lambda_program": args.lambda_program, "lambda_align": args.lambda_align,
        "lambda_prior": args.lambda_prior, "lambda_sparse": args.lambda_sparse,
        "lambda_diversity": args.lambda_diversity,
    }


def build_stage_optimizer(model, args, stage_name):
    base_lr = args.lr
    if stage_name == "iterative_full":
        base_lr *= args.iterative_final_lr_scale
    relation_ids = {id(model.R_delta)}
    relation_params = [
        p for p in model.parameters() if p.requires_grad and id(p) in relation_ids
    ]
    base_params = [
        p for p in model.parameters() if p.requires_grad and id(p) not in relation_ids
    ]
    groups = []
    if base_params:
        groups.append({"params": base_params, "lr": base_lr})
    if relation_params:
        multiplier = (
            args.iterative_relation_lr_multiplier
            if stage_name == "relation_iter" else 1.0
        )
        groups.append({"params": relation_params, "lr": base_lr * multiplier})
    return torch.optim.AdamW(groups, weight_decay=args.weight_decay)


def stage_weights_for_epoch(epoch, total_epochs, args):
    if args.training_schedule == "iterative_relation":
        stage_name, _, _ = iterative_relation_context(epoch, total_epochs, args)
        return stage_name, iterative_stage_weights(stage_name, args)
    if args.training_schedule == 'standard':
        return 'standard', {
            'lambda_phenotype': 1.0,
            'lambda_gene': args.lambda_gene,
            'lambda_program': args.lambda_program,
            'lambda_align': args.lambda_align,
            'lambda_prior': args.lambda_prior,
            'lambda_sparse': args.lambda_sparse,
            'lambda_diversity': args.lambda_diversity,
        }
    progress = epoch / max(total_epochs, 1)
    if args.training_schedule == 'relation_then_full':
        if progress <= args.gene_stage1_frac:
            return 'gene_pretrain', {
                'lambda_phenotype': 0.0,
                'lambda_gene': args.lambda_gene_stage1,
                'lambda_program': 0.0,
                'lambda_align': 0.0,
                'lambda_prior': 0.0,
                'lambda_sparse': 0.0,
                'lambda_diversity': 0.0,
            }
        if progress <= args.gene_stage2_frac:
            return 'gene_program', {
                'lambda_phenotype': 0.0,
                'lambda_gene': args.lambda_gene_stage2,
                'lambda_program': args.lambda_program,
                'lambda_align': 0.0,
                'lambda_prior': args.lambda_prior * 0.5,
                'lambda_sparse': args.lambda_sparse,
                'lambda_diversity': args.lambda_diversity,
            }
        if progress <= args.relation_warmup_frac:
            return 'relation_warmup', {
                'lambda_phenotype': 1.0,
                'lambda_gene': 0.0,
                'lambda_program': 0.0,
                'lambda_align': 0.0,
                'lambda_prior': 0.0,
                'lambda_sparse': args.lambda_sparse,
                'lambda_diversity': args.lambda_diversity,
            }
        return 'full', {
            'lambda_phenotype': 1.0,
            'lambda_gene': 0.0,
            'lambda_program': args.lambda_program,
            'lambda_align': args.lambda_align,
            'lambda_prior': args.lambda_prior,
            'lambda_sparse': args.lambda_sparse,
            'lambda_diversity': args.lambda_diversity,
        }
    if progress <= args.gene_stage1_frac:
        return 'gene_pretrain', {
            'lambda_phenotype': 0.0,
            'lambda_gene': args.lambda_gene_stage1,
            'lambda_program': 0.0,
            'lambda_align': 0.0,
            'lambda_prior': 0.0,
            'lambda_sparse': 0.0,
            'lambda_diversity': 0.0,
        }
    if progress <= args.gene_stage2_frac:
        return 'gene_program', {
            'lambda_phenotype': 0.0,
            'lambda_gene': args.lambda_gene_stage2,
            'lambda_program': args.lambda_program,
            'lambda_align': 0.0,
            'lambda_prior': args.lambda_prior * 0.5,
            'lambda_sparse': args.lambda_sparse,
            'lambda_diversity': args.lambda_diversity,
        }
    return 'full', {
        'lambda_phenotype': 1.0,
        'lambda_gene': args.lambda_gene,
        'lambda_program': args.lambda_program,
        'lambda_align': args.lambda_align,
        'lambda_prior': args.lambda_prior,
        'lambda_sparse': args.lambda_sparse,
        'lambda_diversity': args.lambda_diversity,
    }


def aggregate_survival_by_case(case_ids, targets, predictions, masks):
    grouped = {}
    for case_id, target, prediction, mask in zip(case_ids, targets, predictions, masks):
        item = grouped.setdefault(str(case_id), {"target": target, "mask": mask, "predictions": []})
        item["predictions"].append(prediction)
    ordered_cases = list(grouped)
    y = np.asarray([grouped[c]["target"] for c in ordered_cases], dtype=float)
    m = np.asarray([grouped[c]["mask"] for c in ordered_cases], dtype=float)
    p = np.asarray([
        np.mean(np.asarray(grouped[c]["predictions"], dtype=float), axis=0)
        for c in ordered_cases
    ])
    return ordered_cases, y, p, m


def hazards_to_rmst_risk(hazards, time_bins):
    hazards = np.clip(np.asarray(hazards, dtype=float), 0.0, 1.0)
    ends = np.asarray(time_bins, dtype=float)
    widths = np.diff(np.concatenate([[0.0], ends]))
    survival_end = np.cumprod(1.0 - hazards)
    survival_start = np.concatenate([[1.0], survival_end[:-1]])
    rmst = np.sum(0.5 * (survival_start + survival_end) * widths)
    return -float(rmst)


def harrell_c_index(times, events, risks):
    concordant = 0.0
    comparable = 0
    for i in range(len(times)):
        if events[i] <= 0:
            continue
        for j in range(len(times)):
            if times[j] <= times[i]:
                continue
            comparable += 1
            if risks[i] > risks[j]:
                concordant += 1.0
            elif risks[i] == risks[j]:
                concordant += 0.5
    return concordant / comparable if comparable else float("nan"), comparable


def evaluate(model, loader, device, phenotype_specs, program_names):
    model.eval()
    losses = []
    pheno_true = [[] for _ in phenotype_specs]; pheno_pred = [[] for _ in phenotype_specs]; pheno_mask = [[] for _ in phenotype_specs]
    pheno_cases = [[] for _ in phenotype_specs]
    prog_true = [[] for _ in program_names]; prog_pred = [[] for _ in program_names]
    gene_names = getattr(loader.dataset, 'vocab', {}).get('gene_list', [])
    gene_true = [[] for _ in gene_names]; gene_pred = [[] for _ in gene_names]
    weights = dict(lambda_program=.4, lambda_align=.2, lambda_prior=.01, lambda_sparse=.001, lambda_diversity=.001)
    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            out = model(batch['features'])
            loss, parts = total_loss(model, out, batch, phenotype_specs, weights)
            losses.append(parts['total'])
            for i, spec in enumerate(phenotype_specs):
                mask_i = batch['phenotype_masks'][i]
                logit = out['phenotype_logits'][i]
                if spec['task_type'] == 'discrete_survival':
                    n_bins = int(spec.get('num_bins', logit.numel()))
                    pred = torch.sigmoid(logit.reshape(-1)[:n_bins]).detach().cpu().numpy()
                    y = batch['phenotype_targets'][i, :n_bins].detach().cpu().numpy()
                    m = mask_i[:n_bins].detach().cpu().numpy()
                    pheno_true[i].append(y); pheno_pred[i].append(pred); pheno_mask[i].append(m)
                    pheno_cases[i].append(batch['case_id'])
                    continue
                if mask_i.sum().item() <= 0:
                    continue
                if spec['task_type'] in {'binary', 'survival'}:
                    pred = torch.sigmoid(logit).item(); y = batch['phenotype_targets'][i, 0].item(); m = 1.0
                elif spec['task_type'] == 'multiclass':
                    pred = int(torch.argmax(logit).item()); y = batch['phenotype_targets'][i, 0].item(); m = 1.0
                else:
                    pred = logit.item(); y = batch['phenotype_targets'][i, 0].item(); m = 1.0
                pheno_true[i].append(y); pheno_pred[i].append(pred); pheno_mask[i].append(m)
                pheno_cases[i].append(batch['case_id'])
            for i in range(len(program_names)):
                if batch['program_masks'][i].item() > 0:
                    prog_true[i].append(batch['program_targets'][i].item())
                    prog_pred[i].append(out['program_pred'][i].item())
            if 'gene_pred' in out and gene_names:
                for i in range(len(gene_names)):
                    if batch['gene_masks'][i].item() > 0:
                        gene_true[i].append(batch['gene_targets'][i].item())
                        gene_pred[i].append(out['gene_pred'][i].item())
    rows = []
    for i, spec in enumerate(phenotype_specs):
        if spec['task_type'] == 'discrete_survival':
            if pheno_true[i]:
                case_ids, y_arr, p_arr, m_arr = aggregate_survival_by_case(
                    pheno_cases[i], pheno_true[i], pheno_pred[i], pheno_mask[i]
                )
                m_arr = m_arr > 0
                aucs = []
                labels = spec.get('bin_labels', [f'bin{k+1}' for k in range(y_arr.shape[1])])
                for k in range(y_arr.shape[1]):
                    valid = m_arr[:, k]
                    val = float('nan')
                    if valid.sum() >= 2 and len(set(y_arr[valid, k].tolist())) >= 2:
                        val = float(roc_auc_score(y_arr[valid, k], p_arr[valid, k]))
                        aucs.append(val)
                    rows.append({'task_name': f"{spec['name']}::{labels[k]}", 'task_type': spec['task_type'], 'metric_name': 'AUC', 'metric_value': val, 'n': int(valid.sum())})
                rows.append({'task_name': f"{spec['name']}::mean", 'task_type': spec['task_type'], 'metric_name': 'AUC', 'metric_value': float(np.mean(aucs)) if aucs else float('nan'), 'n': len(case_ids)})
                horizon = float(spec.get('time_bins', [3650.0])[-1])
                observed_times, observed_events, risks = [], [], []
                for case_id, hazards in zip(case_ids, p_arr):
                    clinical_row = loader.dataset.pheno.loc[case_id]
                    if isinstance(clinical_row, pd.DataFrame):
                        clinical_row = clinical_row.iloc[0]
                    event = parse_binary(clinical_row[spec['name']])
                    time_value = pd.to_numeric(pd.Series([clinical_row[spec['time_name']]]), errors='coerce').iloc[0]
                    if pd.isna(event) or pd.isna(time_value):
                        continue
                    observed_times.append(min(float(time_value), horizon))
                    observed_events.append(float(event > 0 and float(time_value) <= horizon))
                    risks.append(hazards_to_rmst_risk(hazards, spec.get('time_bins', [horizon])))
                c_index, comparable_pairs = harrell_c_index(
                    np.asarray(observed_times), np.asarray(observed_events), np.asarray(risks)
                )
                rows.append({'task_name': f"{spec['name']}::c_index", 'task_type': spec['task_type'], 'metric_name': 'C-index', 'metric_value': c_index, 'n': comparable_pairs})
            else:
                rows.append({'task_name': spec['name'], 'task_type': spec['task_type'], 'metric_name': 'AUC', 'metric_value': float('nan'), 'n': 0})
            continue
        metric, val = metric_safe(spec['task_type'], pheno_true[i], pheno_pred[i])
        rows.append({'task_name': spec['name'], 'task_type': spec['task_type'], 'metric_name': metric, 'metric_value': val, 'n': len(pheno_true[i])})
    for i, name in enumerate(program_names):
        metric, val = metric_safe('regression', prog_true[i], prog_pred[i])
        rows.append({'task_name': name, 'task_type': 'gene_program_regression', 'metric_name': metric, 'metric_value': val, 'n': len(prog_true[i])})
    for i, name in enumerate(gene_names):
        rows.append({'task_name': name, 'task_type': 'gene_expression_regression', 'metric_name': 'Pearson', 'metric_value': pearson_safe(gene_true[i], gene_pred[i]), 'n': len(gene_true[i])})
    return float(np.mean(losses)) if losses else float('nan'), rows


def validation_summary(rows):
    def finite_values(predicate):
        values = [
            float(row["metric_value"]) for row in rows
            if predicate(row) and np.isfinite(row["metric_value"])
        ]
        return values

    binary_values = finite_values(lambda row: row["task_type"] == "binary")
    multiclass_values = finite_values(lambda row: row["task_type"] == "multiclass")
    os_mean_values = finite_values(
        lambda row: row["task_type"] == "discrete_survival"
        and row["task_name"].endswith("::mean")
    )
    c_index_values = finite_values(
        lambda row: row["task_type"] == "discrete_survival"
        and row["task_name"].endswith("::c_index")
    )
    gene_values = finite_values(lambda row: row["task_type"] == "gene_expression_regression")
    program_values = finite_values(lambda row: row["task_type"] == "gene_program_regression")
    phenotype_values = binary_values + multiclass_values + os_mean_values

    def mean_or_nan(values):
        return float(np.mean(values)) if values else float("nan")

    return {
        "binary_auc": mean_or_nan(binary_values),
        "multiclass_acc": mean_or_nan(multiclass_values),
        "os_mean_auc": mean_or_nan(os_mean_values),
        "c_index": mean_or_nan(c_index_values),
        "gene_pearson": mean_or_nan(gene_values),
        "program_mae": mean_or_nan(program_values),
        "phenotype_score": mean_or_nan(phenotype_values),
    }



def export_relation_csvs(tool_dir, metadata, H_gene, R, HR, G_prior=None):
    vocab = metadata['vocab']
    pd.DataFrame(H_gene, index=vocab['gene_list'], columns=vocab['program_names']).to_csv(
        tool_dir / 'H_gene_to_program.csv', encoding='utf-8-sig'
    )
    pd.DataFrame(R, index=vocab['program_names'], columns=vocab['phenotype_names']).to_csv(
        tool_dir / 'R_program_to_phenotype.csv', encoding='utf-8-sig'
    )
    pd.DataFrame(HR, index=vocab['gene_list'], columns=vocab['phenotype_names']).to_csv(
        tool_dir / 'HR_gene_to_phenotype.csv', encoding='utf-8-sig'
    )
    if G_prior is not None:
        pd.DataFrame(G_prior, index=vocab['gene_list'], columns=vocab['phenotype_names']).to_csv(
            tool_dir / 'G_gene_to_phenotype_prior.csv', encoding='utf-8-sig'
        )

def export_relation_dynamics(
    tool_dir, metadata, effective_r, raw_r, gate, initial_r,
    selection_mode="prior_guided",
):
    programs = metadata["vocab"]["program_names"]
    phenotypes = metadata["vocab"]["phenotype_names"]
    prior_mask = np.asarray(metadata["R_prior"]) > 0
    pd.DataFrame(raw_r, index=programs, columns=phenotypes).to_csv(
        tool_dir / "R_program_to_phenotype_raw.csv", encoding="utf-8-sig"
    )
    pd.DataFrame(gate, index=programs, columns=phenotypes).to_csv(
        tool_dir / "R_relation_gate.csv", encoding="utf-8-sig"
    )
    pd.DataFrame(effective_r - initial_r, index=programs, columns=phenotypes).to_csv(
        tool_dir / "R_program_to_phenotype_change.csv", encoding="utf-8-sig"
    )
    rows = []
    for i, program in enumerate(programs):
        for j, phenotype in enumerate(phenotypes):
            is_prior = bool(prior_mask[i, j])
            selected = bool(gate[i, j] >= 1.0 - 1e-6)
            if selected and not is_prior:
                status = (
                    "nonprior_selected" if selection_mode == "free_topk"
                    else "new_selected"
                )
            elif selected and is_prior:
                status = "prior_retained"
            elif is_prior:
                status = "prior_attenuated"
            else:
                status = "background_attenuated"
            rows.append({
                "program_name": program, "phenotype_name": phenotype,
                "prior_edge": int(is_prior), "selected": int(selected),
                "gate": float(gate[i, j]), "initial_weight": float(initial_r[i, j]),
                "raw_learned_weight": float(raw_r[i, j]),
                "effective_weight": float(effective_r[i, j]),
                "raw_change": float(raw_r[i, j] - initial_r[i, j]),
                "effective_change": float(effective_r[i, j] - initial_r[i, j]),
                "status": status,
            })
    pd.DataFrame(rows).to_csv(
        tool_dir / "R_program_to_phenotype_changes_long.csv",
        index=False, encoding="utf-8-sig"
    )


def export_gene_program_changes(tool_dir, metadata, gene_weights, keep_threshold=0.5, add_threshold=0.3, weaken_threshold=0.3):
    gene_list = metadata['vocab']['gene_list']
    program_names = metadata['vocab']['program_names']
    prior = metadata['H_prior']
    rows = []
    for i, gene in enumerate(gene_list):
        for j, program_name in enumerate(program_names):
            weight = float(gene_weights[i, j])
            prior_member = bool(prior[i, j] > 0)
            if prior_member and weight >= keep_threshold:
                status = 'retained'
            elif prior_member and weight < weaken_threshold:
                status = 'weakened_or_removed'
            elif (not prior_member) and weight >= add_threshold:
                status = 'new_candidate'
            else:
                status = 'background'
            rows.append({
                'gene': gene,
                'program_name': program_name,
                'learned_weight': weight,
                'prior_member': int(prior_member),
                'status': status,
            })
    df = pd.DataFrame(rows)
    df.to_csv(tool_dir / 'gene_program_changes.csv', index=False, encoding='utf-8-sig')
    df[df['status'] != 'background'].sort_values(['program_name', 'status', 'learned_weight'], ascending=[True, True, False]).to_csv(
        tool_dir / 'gene_program_changes_nonbackground.csv', index=False, encoding='utf-8-sig'
    )

def export_learned_gene_programs(tool_dir, metadata, gene_weights, top_k=12):
    gene_list = metadata['vocab']['gene_list']
    program_names = metadata['vocab']['program_names']
    rows = []
    programs = []
    prior = metadata['H_prior']
    for j, program_name in enumerate(program_names):
        order = np.argsort(-gene_weights[:, j])
        top_genes = []
        for rank, i in enumerate(order[:top_k], start=1):
            rows.append({
                'program_name': program_name,
                'rank': rank,
                'gene': gene_list[i],
                'learned_weight': float(gene_weights[i, j]),
                'prior_member': int(prior[i, j] > 0),
            })
            top_genes.append({'gene': gene_list[i], 'weight': float(gene_weights[i, j]), 'prior_member': bool(prior[i, j] > 0)})
        programs.append({'program_name': program_name, 'top_genes': top_genes})
    pd.DataFrame(rows).to_csv(tool_dir / 'learned_gene_programs.csv', index=False, encoding='utf-8-sig')
    with (tool_dir / 'learned_gene_programs.json').open('w', encoding='utf-8') as f:
        json.dump(programs, f, ensure_ascii=False, indent=2)

def make_registry(vocab, programs, learned_R, relation_gate=None, top_k=6):
    registry = {}
    learned_R = np.asarray(learned_R)
    relation_gate = None if relation_gate is None else np.asarray(relation_gate)
    for j, pheno in enumerate(vocab["phenotype_names"]):
        candidates = (
            np.arange(learned_R.shape[0])
            if relation_gate is None
            else np.where(relation_gate[:, j] >= 1.0 - 1e-6)[0]
        )
        order = candidates[np.argsort(-np.abs(learned_R[candidates, j]))[:top_k]]
        related_programs = [vocab["program_names"][i] for i in order]
        relation_weights = {
            vocab["program_names"][i]: float(learned_R[i, j]) for i in order
        }
        related_genes = sorted({
            g for p in programs if p["program_name"] in related_programs
            for g in p["member_genes"]
        })
        key = "".join(
            c if c.isalnum() else "_" for c in vocab["phenotype_fields"][j]
        ).strip("_") + "_tool"
        registry[key] = {
            "task_type": "phenotype_prediction", "phenotype": pheno,
            "phenotype_field": vocab["phenotype_fields"][j],
            "related_programs": related_programs,
            "relation_weights": relation_weights, "related_genes": related_genes,
            "output": ["probability_or_value", "program_score", "top_k_patches", "attention_heatmap"],
            "threshold": 0.5,
        }
    return registry


def export_toolbank(model, metadata, out_dir, args, val_rows, split_counts):
    tool_dir = Path(out_dir) / 'G2P_ToolBank_Minimal'
    tool_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        'state_dict': model.state_dict(),
        'feature_dim': model.feature_dim,
        'hidden_dim': model.hidden_dim,
        'phenotype_specs': metadata['phenotypes'],
        'program_names': metadata['vocab']['program_names'],
        'phenotype_mode': getattr(model, 'phenotype_mode', 'full'),
        'max_patches': args.max_patches,
        'relation_init_mode': args.relation_init_mode,
        'relation_init_value': args.relation_init_value,
        'relation_selection_mode': args.relation_selection_mode,
    }, tool_dir / 'model.pt')
    with (tool_dir / 'vocab.json').open('w', encoding='utf-8') as f: json.dump(metadata['vocab'], f, ensure_ascii=False, indent=2)
    with (tool_dir / 'normalization.json').open('w', encoding='utf-8') as f: json.dump(metadata['normalization'], f, ensure_ascii=False, indent=2)
    R = model.program_phenotype_weights().detach().cpu().numpy()
    R_raw = model.raw_program_phenotype_weights().detach().cpu().numpy()
    R_gate = model.relation_gate.detach().cpu().numpy()
    R_initial = model.relation_initial.detach().cpu().numpy()
    H_gene = model.gene_program_weights().detach().cpu().numpy()
    H_gene_prior = model.gene_H_prior.detach().cpu().numpy()
    gene_prototypes = model.gene_query_prototypes().detach().cpu().numpy()
    gene_identity_embeddings = model.gene_identity_tokens().detach().cpu().numpy()
    gene_query_delta = model.gene_query_delta.detach().cpu().numpy() if hasattr(model, 'gene_query_delta') else np.zeros_like(gene_prototypes)
    HR = H_gene @ R
    np.savez(
        tool_dir / 'relations.npz',
        H_gene_to_program=H_gene,
        H_gene_prior=H_gene_prior,
        gene_prototypes=gene_prototypes,
        gene_identity_embeddings=gene_identity_embeddings,
        gene_query_delta=gene_query_delta,
        G_gene_to_phenotype_prior=metadata['G_prior'],
        H_prior=H_gene_prior,
        R_program_to_phenotype=R,
        R_program_to_phenotype_raw=R_raw,
        R_relation_gate=R_gate,
        R_initial=R_initial,
        relation_init_mode=np.asarray(args.relation_init_mode),
        relation_selection_mode=np.asarray(args.relation_selection_mode),
        R_prior=metadata['R_prior'],
        HR_gene_to_phenotype=HR,
    )
    export_relation_csvs(tool_dir, metadata, H_gene, R, HR, metadata.get('G_prior'))
    export_relation_dynamics(
        tool_dir, metadata, R, R_raw, R_gate, R_initial,
        selection_mode=args.relation_selection_mode,
    )
    pd.DataFrame(gene_prototypes, index=metadata['vocab']['gene_list']).to_csv(
        tool_dir / 'gene_prototypes.csv', encoding='utf-8-sig'
    )
    pd.DataFrame(gene_identity_embeddings, index=metadata['vocab']['gene_list']).to_csv(
        tool_dir / 'gene_identity_embeddings.csv', encoding='utf-8-sig'
    )
    pd.DataFrame(gene_query_delta, index=metadata['vocab']['gene_list']).to_csv(
        tool_dir / 'gene_query_delta.csv', encoding='utf-8-sig'
    )
    export_learned_gene_programs(tool_dir, metadata, H_gene)
    export_gene_program_changes(tool_dir, metadata, H_gene)
    with (tool_dir / 'tool_registry.json').open('w', encoding='utf-8') as f: json.dump(make_registry(metadata['vocab'], metadata['programs'], R, R_gate, args.iterative_relation_topk), f, ensure_ascii=False, indent=2)
    metrics_path = tool_dir / 'tool_metrics.csv'
    with metrics_path.open('w', newline='', encoding='utf-8-sig') as f:
        fieldnames = ['task_name','task_type','num_train','num_val','num_test','metric_name','metric_value','missing_rate']
        w = csv.DictWriter(f, fieldnames=fieldnames); w.writeheader()
        for r in val_rows:
            w.writerow({'task_name': r['task_name'], 'task_type': r['task_type'], 'num_train': split_counts.get('train',0), 'num_val': split_counts.get('val',0), 'num_test': split_counts.get('test',0), 'metric_name': r['metric_name'], 'metric_value': r['metric_value'], 'missing_rate': ''})
    with (tool_dir / 'train_config.json').open('w', encoding='utf-8') as f: json.dump(vars(args), f, ensure_ascii=False, indent=2)
    return tool_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--label_dir', required=True); ap.add_argument('--program_scores_csv', required=True); ap.add_argument('--feature_dir', default=None); ap.add_argument('--feature_size', choices=['1024', '2048', '4096'], default=None); ap.add_argument('--out_dir', default='outputs/debug_run')
    ap.add_argument('--split_csv', default=None)
    ap.add_argument('--resume_checkpoint', default=None)
    ap.add_argument('--epochs', type=int, default=50); ap.add_argument('--batch_size', type=int, default=1); ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--gene_batch_size', type=int, default=None)
    ap.add_argument('--phenotype_batch_size', type=int, default=None)
    ap.add_argument('--max_patches', type=int, default=None)
    ap.add_argument('--grad_accum_steps', type=int, default=1)
    ap.add_argument('--gene_grad_accum_steps', type=int, default=None)
    ap.add_argument('--phenotype_grad_accum_steps', type=int, default=None)
    ap.add_argument('--weight_decay', type=float, default=1e-5); ap.add_argument('--hidden_dim', type=int, default=256); ap.add_argument('--dropout', type=float, default=.25)
    ap.add_argument('--seed', type=int, default=42); ap.add_argument('--device', default='auto'); ap.add_argument('--max_samples', type=int, default=None)
    ap.add_argument('--topk_patches', type=int, default=20)
    ap.add_argument('--log_every', type=int, default=50)
    ap.add_argument('--min_free_gb', type=float, default=6.0)
    ap.add_argument('--reserve_gb', type=float, default=4.0)
    ap.add_argument('--phenotype_schedule', choices=['staged', 'flat', 'molecular_first', 'no_clinical'], default='staged')
    ap.add_argument('--training_schedule', choices=['gene_curriculum', 'standard', 'relation_then_full', 'iterative_relation'], default='gene_curriculum')
    ap.add_argument('--gene_stage1_frac', type=float, default=0.2)
    ap.add_argument('--gene_stage2_frac', type=float, default=0.6)
    ap.add_argument('--relation_warmup_frac', type=float, default=0.65)
    ap.add_argument('--iterative_gene_pretrain_epochs', type=int, default=10)
    ap.add_argument('--iterative_gene_program_epochs', type=int, default=24)
    ap.add_argument('--iterative_relation_rounds', type=int, default=3)
    ap.add_argument('--relation_init_mode', choices=['prior', 'uniform'], default='prior')
    ap.add_argument('--relation_init_value', type=float, default=0.5)
    ap.add_argument('--relation_selection_mode', choices=['prior_guided', 'free_topk'], default='prior_guided')
    ap.add_argument('--iterative_relation_epochs', type=int, default=6)
    ap.add_argument('--iterative_adapt_epochs', type=int, default=10)
    ap.add_argument('--iterative_relation_topk', type=int, default=6)
    ap.add_argument('--iterative_relation_new_edges', type=int, default=3)
    ap.add_argument('--iterative_relation_gate_floor', type=float, default=0.15)
    ap.add_argument('--iterative_relation_ema_decay', type=float, default=0.5)
    ap.add_argument('--iterative_relation_min_new_change', type=float, default=0.05)
    ap.add_argument('--iterative_relation_min_new_weight', type=float, default=0.35)
    ap.add_argument('--iterative_relation_scale_start', type=float, default=0.5)
    ap.add_argument('--iterative_relation_scale_multiplier', type=float, default=2.0)
    ap.add_argument('--iterative_relation_scale_max', type=float, default=2.0)
    ap.add_argument('--iterative_relation_lr_multiplier', type=float, default=5.0)
    ap.add_argument('--iterative_final_lr_scale', type=float, default=0.5)
    ap.add_argument('--lambda_gene', type=float, default=0.1)
    ap.add_argument('--lambda_survival', type=float, default=1.0)
    ap.add_argument('--survival_pos_weight_max', type=float, default=1.0)
    ap.add_argument('--lambda_gene_stage1', type=float, default=1.0)
    ap.add_argument('--lambda_gene_stage2', type=float, default=0.5)
    ap.add_argument('--lambda_gene_corr', type=float, default=0.0)
    ap.add_argument('--lambda_gene_query_delta', type=float, default=0.0)
    ap.add_argument('--lambda_program', type=float, default=0.4)
    ap.add_argument('--lambda_align', type=float, default=0.2)
    ap.add_argument('--lambda_prior', type=float, default=0.01)
    ap.add_argument('--lambda_sparse', type=float, default=0.001)
    ap.add_argument('--lambda_diversity', type=float, default=0.001)
    ap.add_argument('--lambda_rna_align', type=float, default=0.05)
    ap.add_argument('--lambda_rna_recon', type=float, default=0.05)
    ap.add_argument('--gene_retention_ratio', type=float, default=0.98,
                    help='Minimum fraction of the final gene-stage validation Pearson required for phenotype checkpoint selection.')
    ap.add_argument('--gene_program_mae_tolerance', type=float, default=0.01,
                    help='Maximum program MAE gap from the best gene-program epoch when selecting the protected gene checkpoint.')
    args = ap.parse_args()
    if args.gene_batch_size is None:
        args.gene_batch_size = args.batch_size
    if args.phenotype_batch_size is None:
        args.phenotype_batch_size = args.batch_size
    if args.gene_grad_accum_steps is None:
        args.gene_grad_accum_steps = args.grad_accum_steps
    if args.phenotype_grad_accum_steps is None:
        args.phenotype_grad_accum_steps = args.grad_accum_steps
    if min(args.batch_size, args.gene_batch_size, args.phenotype_batch_size, args.grad_accum_steps, args.gene_grad_accum_steps, args.phenotype_grad_accum_steps) < 1:
        ap.error("batch sizes and all gradient accumulation settings must be positive")
    if args.max_patches is not None and args.max_patches < 1:
        ap.error("--max_patches must be positive")
    if args.training_schedule == "iterative_relation":
        positive_ints = [
            args.iterative_gene_pretrain_epochs, args.iterative_gene_program_epochs,
            args.iterative_relation_rounds, args.iterative_relation_epochs,
            args.iterative_adapt_epochs, args.iterative_relation_topk,
        ]
        if min(positive_ints) < 1:
            ap.error("iterative epoch, round, and top-k settings must be positive")
        if not -1.0 < args.relation_init_value < 1.0:
            ap.error("--relation_init_value must be strictly between -1 and 1")
        if not 0.0 <= args.iterative_relation_gate_floor <= 1.0:
            ap.error("--iterative_relation_gate_floor must be in [0, 1]")
        if not 0.0 <= args.iterative_relation_ema_decay < 1.0:
            ap.error("--iterative_relation_ema_decay must be in [0, 1)")
        if args.iterative_relation_min_new_change < 0.0:
            ap.error("--iterative_relation_min_new_change must be non-negative")
        if not -1.0 <= args.iterative_relation_min_new_weight <= 1.0:
            ap.error("--iterative_relation_min_new_weight must be in [-1, 1]")
        required = (
            args.iterative_gene_pretrain_epochs + args.iterative_gene_program_epochs
            + args.iterative_relation_rounds
            * (args.iterative_relation_epochs + args.iterative_adapt_epochs)
        )
        if args.epochs <= required:
            ap.error(
                f"--epochs must exceed {required} for iterative_relation so the final joint stage runs"
            )
    seed_all(args.seed)
    args.feature_dir = resolve_feature_dir(args)
    distributed, rank, world_size, local_rank = setup_distributed()
    out_dir = Path(args.out_dir)
    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f'Using feature_dir={args.feature_dir}', flush=True)
    if distributed:
        dist.barrier(device_ids=[local_rank])
    manifest_csv = out_dir / 'aligned_manifest.csv'
    if rank == 0 and not manifest_csv.exists():
        prepare_manifest(args.label_dir, args.feature_dir, out_dir, seed=args.seed, split_csv=args.split_csv)
    if distributed:
        dist.barrier(device_ids=[local_rank])
    gene_programs_json = Path(__file__).resolve().parents[1] / 'configs/gene_programs.json'
    metadata_tuple = build_metadata(
        args.label_dir, manifest_csv, gene_programs_json, out_dir,
        program_scores_csv=args.program_scores_csv,
    )
    metadata, _, _, _ = metadata_tuple
    feature_dim = infer_feature_dim(manifest_csv)
    manifest = pd.read_csv(manifest_csv); split_counts = manifest.split.value_counts().to_dict()
    if rank == 0:
        print(f'Aligned cases={manifest.case_id.nunique()} slides={len(manifest)} split={split_counts}', flush=True)
        group_counts = pd.Series([p.get('group', 'morphology') for p in metadata['phenotypes']]).value_counts().to_dict()
        print(f'Feature dim={feature_dim}; phenotypes={len(metadata["phenotypes"])} groups={group_counts} genes={len(metadata["vocab"]["gene_list"])} programs={len(metadata["vocab"]["program_names"])}', flush=True)
    datasets = {s: G2PBRCAFeatureDataset(manifest_csv, args.label_dir, gene_programs_json, out_dir, split=s, max_samples=args.max_samples, metadata=metadata_tuple, max_patches=args.max_patches) for s in ['train','val','test']}
    survival_pos_weights = compute_survival_pos_weights(
        datasets['train'], metadata['phenotypes'], args.survival_pos_weight_max
    )
    phenotype_task_weights = {
        spec['name']: (args.lambda_survival if spec['task_type'] == 'discrete_survival' else 1.0)
        for spec in metadata['phenotypes']
    }
    if rank == 0:
        print(f"Survival pos_weights={format_survival_pos_weights(survival_pos_weights)}", flush=True)
        print(f"Survival task weight={args.lambda_survival:g}", flush=True)
    train_sampler = DistributedSampler(datasets["train"], shuffle=True, seed=args.seed) if distributed else None
    loaders = {
        "train_standard": DataLoader(datasets["train"], batch_size=args.batch_size, shuffle=(train_sampler is None), sampler=train_sampler, collate_fn=mil_collate if args.batch_size > 1 else one_item),
        "train_gene": DataLoader(datasets["train"], batch_size=args.gene_batch_size, shuffle=(train_sampler is None), sampler=train_sampler, collate_fn=mil_collate if args.gene_batch_size > 1 else one_item),
        "train_phenotype": DataLoader(datasets["train"], batch_size=args.phenotype_batch_size, shuffle=(train_sampler is None), sampler=train_sampler, collate_fn=mil_collate if args.phenotype_batch_size > 1 else one_item),
        "val": DataLoader(datasets["val"], batch_size=1, shuffle=False, collate_fn=one_item),
        "test": DataLoader(datasets["test"], batch_size=1, shuffle=False, collate_fn=one_item),
    }
    if distributed:
        device = torch.device(f'cuda:{local_rank}')
        if rank == 0:
            print(f'DDP enabled: world_size={world_size}; one process per GPU; batch_size_gene={args.gene_batch_size}; batch_size_phenotype={args.phenotype_batch_size}; max_patches={args.max_patches}; grad_accum_gene={args.gene_grad_accum_steps}; grad_accum_phenotype={args.phenotype_grad_accum_steps}; effective_global_batch_gene={args.gene_batch_size * world_size * args.gene_grad_accum_steps}; effective_global_batch_phenotype={args.phenotype_batch_size * world_size * args.phenotype_grad_accum_steps}', flush=True)
    else:
        device = select_device(args.device, min_free_gb=args.min_free_gb, reserve_gb=args.reserve_gb, verbose=True)
    model = G2PHypergraphToolBank(
        feature_dim,
        args.hidden_dim,
        metadata['phenotypes'],
        len(metadata['vocab']['gene_list']),
        metadata['vocab']['program_names'],
        metadata['H_prior'],
        metadata['R_prior'],
        dropout=args.dropout,
        gene_phenotype_prior=metadata['G_prior'],
        gene_names=metadata['vocab']['gene_list'],
        relation_init_mode=args.relation_init_mode,
        relation_init_value=args.relation_init_value,
        relation_selection_mode=args.relation_selection_mode,
    ).to(device)
    train_model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True) if distributed else model
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    weights = dict(
        lambda_gene_corr=args.lambda_gene_corr,
        lambda_rna_align=args.lambda_rna_align,
        lambda_rna_recon=args.lambda_rna_recon,
        lambda_gene_query_delta=args.lambda_gene_query_delta,
        survival_pos_weights=survival_pos_weights,
        phenotype_task_weights=phenotype_task_weights,
    )
    history = []
    gene_reference_pearson = float('-inf')
    best_gene_program_mae = float('inf')
    best_gene_checkpoint_pearson = float('-inf')
    best_gene_checkpoint_path = out_dir / 'best_gene_stage_model.pt'
    best_phenotype_score = float('-inf')
    best_checkpoint_path = out_dir / 'best_gene_protected_model.pt'
    previous_stage_name = None
    relation_stats = {
        "active_edges": int(model.relation_gate.numel()),
        "selected_prior_edges": 0,
        "selected_new_edges": 0,
        "eligible_new_edges": 0,
        "attenuated_prior_edges": 0,
        "mean_abs_change": 0.0,
    }
    start_epoch = 1
    if args.resume_checkpoint:
        resume_path = Path(args.resume_checkpoint)
        if not resume_path.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        try:
            resume_checkpoint = torch.load(resume_path, map_location=device, weights_only=True)
        except TypeError:
            resume_checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(resume_checkpoint["state_dict"])
        resume_epoch = int(resume_checkpoint.get("epoch", 0))
        if resume_epoch >= args.epochs:
            raise ValueError(
                f"Resume epoch {resume_epoch} must be smaller than --epochs {args.epochs}"
            )
        start_epoch = resume_epoch + 1
        gene_reference_pearson = float(
            resume_checkpoint.get("gene_pearson", gene_reference_pearson)
        )
        best_gene_checkpoint_pearson = gene_reference_pearson
        best_phenotype_score = float(
            resume_checkpoint.get("phenotype_score", best_phenotype_score)
        )
        relation_stats["active_edges"] = int(
            (model.relation_gate >= 1.0 - 1e-6).sum().item()
        )
        if rank == 0:
            print(
                f"Resumed checkpoint={resume_path} epoch={resume_epoch} "
                f"stage={resume_checkpoint.get('stage', 'unknown')} "
                f"next_epoch={start_epoch} gene_pearson={gene_reference_pearson:.4f} "
                f"best_phenotype_score={best_phenotype_score:.4f}",
                flush=True,
            )
    try:
        for epoch in range(start_epoch, args.epochs + 1):
            if distributed:
                train_sampler.set_epoch(epoch)
            train_model.train(); losses=[]
            opt.zero_grad(set_to_none=True)
            pending_grad = False
            skipped_nonfinite_losses = 0
            skipped_nonfinite_gradients = 0
            epoch_start = time.time()
            stage_name, stage_weights = stage_weights_for_epoch(epoch, args.epochs, args)
            relation_round = 0
            relation_scale = float(model.relation_scale.item())
            if args.training_schedule == "iterative_relation":
                _, relation_round, relation_scale = iterative_relation_context(
                    epoch, args.epochs, args
                )
            leaving_gene_stage = (
                previous_stage_name in {'gene_pretrain', 'gene_program'}
                and stage_name not in {'gene_pretrain', 'gene_program'}
            )
            if leaving_gene_stage:
                if distributed:
                    dist.barrier(device_ids=[local_rank])
                if best_gene_checkpoint_path.exists():
                    try:
                        gene_checkpoint = torch.load(best_gene_checkpoint_path, map_location=device, weights_only=True)
                    except TypeError:
                        gene_checkpoint = torch.load(best_gene_checkpoint_path, map_location=device)
                    model.load_state_dict(gene_checkpoint['state_dict'])
                    gene_reference_pearson = gene_checkpoint['gene_pearson']
                    if rank == 0:
                        print(
                            f"Restored gene-stage checkpoint: epoch={gene_checkpoint['epoch']} "
                            f"gene_pearson={gene_checkpoint['gene_pearson']:.4f} "
                            f"program_mae={gene_checkpoint['program_mae']:.4f}",
                            flush=True,
                        )
                opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            if args.training_schedule == "iterative_relation":
                model.set_relation_scale(relation_scale)
                if stage_name != previous_stage_name:
                    if stage_name == "relation_iter":
                        model.reset_relation_gate()
                        if rank == 0:
                            print(
                                f"Relation round {relation_round}: exploration started "
                                f"scale={relation_scale:g} gate=all", flush=True
                            )
                    elif previous_stage_name == "relation_iter" and stage_name == "relation_adapt":
                        relation_stats = model.select_relation_edges(
                            topk=args.iterative_relation_topk,
                            new_edges=args.iterative_relation_new_edges,
                            gate_floor=args.iterative_relation_gate_floor,
                            ema_decay=args.iterative_relation_ema_decay,
                            min_new_change=args.iterative_relation_min_new_change,
                            min_new_weight=args.iterative_relation_min_new_weight,
                            selection_mode=args.relation_selection_mode,
                        )
                        if rank == 0:
                            print(
                                f"Relation round {relation_round}: selected "
                                f"active={relation_stats['active_edges']} "
                                f"prior={relation_stats['selected_prior_edges']} "
                                f"new={relation_stats['selected_new_edges']} "
                                f"eligible_new={relation_stats['eligible_new_edges']} "
                                f"attenuated_prior={relation_stats['attenuated_prior_edges']} "
                                f"mean_abs_change={relation_stats['mean_abs_change']:.4f}",
                                flush=True,
                            )
            if stage_name in {"gene_pretrain", "gene_program"}:
                active_batch_size = args.gene_batch_size
                active_grad_accum_steps = args.gene_grad_accum_steps
                train_loader = loaders["train_gene"]
            elif stage_name in {"relation_warmup", "relation_iter", "relation_adapt", "full", "iterative_full"}:
                active_batch_size = args.phenotype_batch_size
                active_grad_accum_steps = args.phenotype_grad_accum_steps
                train_loader = loaders["train_phenotype"]
            else:
                active_batch_size = args.batch_size
                active_grad_accum_steps = args.grad_accum_steps
                train_loader = loaders["train_standard"]
            num_train_steps = len(train_loader)
            apply_stage_freezing(model, stage_name)
            if args.training_schedule == "iterative_relation" and stage_name != previous_stage_name:
                opt = build_stage_optimizer(model, args, stage_name)
            set_phenotype_mode(model, stage_name)
            weights.update(stage_weights)
            gene_training_stage = stage_name in {'gene_pretrain', 'gene_program', 'standard'}
            weights["lambda_rna_align"] = args.lambda_rna_align if gene_training_stage else 0.0
            weights["lambda_rna_recon"] = args.lambda_rna_recon if gene_training_stage else 0.0
            group_weights = phenotype_group_weights_for_epoch(epoch, args.epochs, args.phenotype_schedule)
            if stage_weights.get('lambda_phenotype', 1.0) <= 0:
                group_weights = {k: 0.0 for k in group_weights}
            weights["phenotype_group_weights"] = group_weights
            if rank == 0:
                print(
                    f"epoch {epoch}/{args.epochs} started | stage={stage_name} "
                    f"relation_round={relation_round} relation_scale={relation_scale:g} "
                    f"relation_active={int((model.relation_gate >= 1.0 - 1e-6).sum().item())} "
                    f"train_steps_per_rank={num_train_steps} "
                    f"phenotype_weights={format_group_weights(group_weights)} phenotype_mode={getattr(model, 'phenotype_mode', 'full')} grad_accum={active_grad_accum_steps} batch_size={active_batch_size} effective_global_batch={active_batch_size * world_size * active_grad_accum_steps} "
                    f"gene_weight={weights.get('lambda_gene', 0):g} "
                    f"rna_align={weights.get('lambda_rna_align', 0):g} rna_recon={weights.get('lambda_rna_recon', 0):g} "
                    f"trainable_params={trainable_parameter_summary(model)[0]}/{trainable_parameter_summary(model)[1]}",
                    flush=True,
                )
            for step, batch in enumerate(train_loader, start=1):
                batch = move_batch(batch, device)
                group_start = ((step - 1) // active_grad_accum_steps) * active_grad_accum_steps
                accum_group_size = min(active_grad_accum_steps, num_train_steps - group_start)
                should_step = step % active_grad_accum_steps == 0 or step == num_train_steps
                sync_context = (
                    train_model.no_sync()
                    if distributed and not should_step
                    else nullcontext()
                )
                with sync_context:
                    out = train_model(
                        batch["features"],
                        batch["gene_targets"],
                        batch["gene_masks"],
                        batch.get("patch_masks"),
                    )
                    loss, parts = total_loss(model, out, batch, metadata["phenotypes"], weights)
                    if all_ranks_finite(loss.detach(), distributed):
                        (loss / accum_group_size).backward()
                        pending_grad = True
                        losses.append(parts["total"])
                    else:
                        skipped_nonfinite_losses += 1
                if should_step:
                    if pending_grad:
                        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                        if all_ranks_finite(grad_norm.detach(), distributed):
                            opt.step()
                        else:
                            skipped_nonfinite_gradients += 1
                            if rank == 0:
                                bad_names = [
                                    name for name, param in model.named_parameters()
                                    if param.grad is not None and not torch.isfinite(param.grad).all()
                                ]
                                print(
                                    f"epoch {epoch} step {step}: skipped optimizer step due to "
                                    f"non-finite gradients in {bad_names[:8]}",
                                    flush=True,
                                )
                    opt.zero_grad(set_to_none=True)
                    pending_grad = False
                if rank == 0 and args.log_every > 0 and (step == 1 or step % args.log_every == 0 or step == num_train_steps):
                    recent = float(np.mean(losses[-min(len(losses), args.log_every):])) if losses else float('nan')
                    elapsed = time.time() - epoch_start
                    print(
                        f"epoch {epoch}/{args.epochs} step {step}/{num_train_steps} "
                        f"recent_train_loss={recent:.4f} "
                        f"gene={parts.get('gene', 0.0):.4f} profile_r={parts.get('gene_corr', 0.0):.3f} "
                        f"mol={parts.get('phenotype_molecular', 0.0):.4f} "
                        f"morph={parts.get('phenotype_morphology', 0.0):.4f} "
                        f"clin={parts.get('phenotype_clinical', 0.0):.4f} "
                        f"elapsed={elapsed/60:.1f}min",
                        flush=True,
                    )
            local_train_loss = float(np.mean(losses)) if losses else float('nan')
            if rank == 0 and (skipped_nonfinite_losses or skipped_nonfinite_gradients):
                print(
                    f"epoch {epoch}: skipped_nonfinite_losses={skipped_nonfinite_losses} "
                    f"skipped_nonfinite_gradients={skipped_nonfinite_gradients}",
                    flush=True,
                )
            train_loss = reduce_mean(local_train_loss, device, distributed)
            if distributed:
                dist.barrier(device_ids=[local_rank])
            if rank == 0:
                val_loss, val_rows = evaluate(model, loaders['val'], device, metadata['phenotypes'], metadata['vocab']['program_names'])
                val_summary = validation_summary(val_rows)
                val_gene_pearson = val_summary['gene_pearson']
                val_phenotype_score = val_summary['phenotype_score']
                val_binary_auc = val_summary['binary_auc']
                val_multiclass_acc = val_summary['multiclass_acc']
                val_os_mean_auc = val_summary['os_mean_auc']
                val_c_index = val_summary['c_index']
                val_program_mae = val_summary['program_mae']
                if stage_name == 'gene_program' and np.isfinite(val_program_mae):
                    best_gene_program_mae = min(best_gene_program_mae, val_program_mae)
                if (
                    stage_name == 'gene_program'
                    and np.isfinite(val_gene_pearson)
                    and np.isfinite(val_program_mae)
                    and val_program_mae <= best_gene_program_mae + args.gene_program_mae_tolerance
                    and val_gene_pearson > best_gene_checkpoint_pearson
                ):
                    best_gene_checkpoint_pearson = val_gene_pearson
                    gene_reference_pearson = val_gene_pearson
                    torch.save({
                        'state_dict': model.state_dict(),
                        'epoch': epoch,
                        'stage': stage_name,
                        'gene_pearson': val_gene_pearson,
                        'program_mae': val_program_mae,
                        'relation_init_mode': args.relation_init_mode,
                        'relation_init_value': args.relation_init_value,
                        'relation_selection_mode': args.relation_selection_mode,
                    }, best_gene_checkpoint_path)
                gene_floor = gene_reference_pearson * args.gene_retention_ratio
                gene_protected = (
                    not np.isfinite(gene_reference_pearson)
                    or (np.isfinite(val_gene_pearson) and val_gene_pearson >= gene_floor)
                )
                if (
                    stage_name in {'full', 'standard', 'relation_adapt', 'iterative_full'}
                    and gene_protected
                    and np.isfinite(val_phenotype_score)
                    and val_phenotype_score > best_phenotype_score
                ):
                    best_phenotype_score = val_phenotype_score
                    torch.save({
                        'state_dict': model.state_dict(),
                        'epoch': epoch,
                        'stage': stage_name,
                        'gene_pearson': val_gene_pearson,
                        'phenotype_score': val_phenotype_score,
                        'gene_floor': gene_floor,
                        'relation_init_mode': args.relation_init_mode,
                        'relation_init_value': args.relation_init_value,
                        'relation_selection_mode': args.relation_selection_mode,
                    }, best_checkpoint_path)
                row = {
                    'epoch': epoch,
                    'stage': stage_name,
                    'relation_round': relation_round,
                    'relation_scale': relation_scale,
                    'relation_active_edges': int((model.relation_gate >= 1.0 - 1e-6).sum().item()),
                    'relation_selected_prior_edges': relation_stats['selected_prior_edges'],
                    'relation_selected_new_edges': relation_stats['selected_new_edges'],
                    'relation_eligible_new_edges': relation_stats['eligible_new_edges'],
                    'relation_attenuated_prior_edges': relation_stats['attenuated_prior_edges'],
                    'relation_mean_abs_change': float((
                        model.raw_program_phenotype_weights() - model.relation_initial
                    ).abs().mean().item()),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'lambda_gene': weights.get('lambda_gene', 0.0),
                    'lambda_program': weights.get('lambda_program', 0.0),
                    'lambda_phenotype': weights.get('lambda_phenotype', 1.0),
                    'molecular_weight': group_weights['molecular'],
                    'morphology_weight': group_weights['morphology'],
                    'clinical_weight': group_weights['clinical'],
                    'val_binary_auc': val_binary_auc,
                    'val_multiclass_acc': val_multiclass_acc,
                    'val_os_mean_auc': val_os_mean_auc,
                    'val_c_index': val_c_index,
                    'val_gene_pearson': val_gene_pearson,
                    'val_program_mae': val_program_mae,
                    'val_phenotype_score': val_phenotype_score,
                    'gene_retention_floor': gene_floor if np.isfinite(gene_floor) else float('nan'),
                    'gene_protected': int(gene_protected),
                }
                history.append(row); print(
                    f"epoch {epoch}: stage={stage_name} train_loss={row['train_loss']:.4f} val_loss={row['val_loss']:.4f} "
                    f"val_binary_auc={val_binary_auc:.4f} val_multiclass_acc={val_multiclass_acc:.4f} "
                    f"val_os_mean_auc={val_os_mean_auc:.4f} val_c_index={val_c_index:.4f} "
                    f"val_gene_pearson={val_gene_pearson:.4f} val_program_mae={val_program_mae:.4f} "
                    f"val_phenotype_score={val_phenotype_score:.4f} "
                    f"gene_protected={gene_protected} phenotype_weights={format_group_weights(group_weights)}",
                    flush=True,
                )
            if distributed:
                dist.barrier(device_ids=[local_rank])
            previous_stage_name = stage_name
        if rank == 0:
            pd.DataFrame(history).to_csv(out_dir / 'train_history.csv', index=False)
            if best_checkpoint_path.exists():
                try:
                    checkpoint = torch.load(best_checkpoint_path, map_location=device, weights_only=True)
                except TypeError:
                    checkpoint = torch.load(best_checkpoint_path, map_location=device)
                model.load_state_dict(checkpoint['state_dict'])
                set_phenotype_mode(model, checkpoint['stage'])
                print(
                    f"Restored protected checkpoint: epoch={checkpoint['epoch']} stage={checkpoint['stage']} "
                    f"gene_pearson={checkpoint['gene_pearson']:.4f} phenotype_score={checkpoint['phenotype_score']:.4f}",
                    flush=True,
                )
            _, val_rows = evaluate(model, loaders['val'], device, metadata['phenotypes'], metadata['vocab']['program_names'])
            tool_dir = export_toolbank(model, metadata, out_dir, args, val_rows, split_counts)
            print(f'ToolBank exported: {tool_dir}', flush=True)
    finally:
        cleanup_distributed(distributed)

if __name__ == '__main__':
    main()
