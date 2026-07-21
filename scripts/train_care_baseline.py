#!/usr/bin/env python3
import argparse
import csv
import json
import math
import random
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, mean_absolute_error, roc_auc_score
from torch.utils.data import DataLoader, Dataset

sys.path.append(str(Path(__file__).resolve().parents[1]))
from data.dataset import (
    CASE_COL,
    SURVIVAL_TIME_BINS,
    discrete_survival_labels,
    build_metadata,
    is_missing_value,
    parse_binary,
    prepare_manifest,
    score_class_label,
    read_label_tables,
)
from scripts.train_g2p_toolbank import (
    aggregate_survival_by_case,
    format_group_weights,
    harrell_c_index,
    hazards_to_rmst_risk,
    metric_safe,
    pearson_safe,
    phenotype_group_weights_for_epoch,
)
from utils.device import select_device


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def discover_care_files(feature_dir):
    rows = []
    for p in sorted(Path(feature_dir).expanduser().glob("*.npy")):
        case_id = p.name[:12]
        if re.match(r"TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}", case_id):
            rows.append({"case_id": case_id, "slide_id": p.stem, "feature_path": str(p)})
    return pd.DataFrame(rows)


def split_cases(case_ids, seed=42, ratios=(0.70, 0.15, 0.15)):
    case_ids = sorted(set(case_ids))
    rng = random.Random(seed)
    rng.shuffle(case_ids)
    n = len(case_ids)
    n_train = int(round(n * ratios[0]))
    n_val = int(round(n * ratios[1]))
    return {
        c: "train" if i < n_train else ("val" if i < n_train + n_val else "test")
        for i, c in enumerate(case_ids)
    }


def prepare_care_manifest(label_dir, feature_dir, out_dir, seed=42, split_csv=None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pheno, genes, _, _ = read_label_tables(label_dir)
    pheno_cases = set(pheno[CASE_COL].dropna().astype(str).str[:12])
    gene_cases = set(genes["case_id"].dropna().astype(str).str[:12])
    features = discover_care_files(feature_dir)
    aligned = features[features["case_id"].isin(pheno_cases & gene_cases)].copy()
    if split_csv:
        splits = pd.read_csv(split_csv)
        split_map = dict(zip(splits["case_id"].astype(str), splits["split"].astype(str)))
    else:
        split_map = split_cases(aligned["case_id"], seed=seed)
    aligned["split"] = aligned["case_id"].map(split_map)
    aligned = aligned[aligned["split"].isin(["train", "val", "test"])].reset_index(drop=True)
    aligned.to_csv(out_dir / "care_manifest.csv", index=False)
    pd.DataFrame({"case_id": list(split_map.keys()), "split": list(split_map.values())}).to_csv(
        out_dir / "case_splits.csv", index=False
    )
    return aligned


def load_care_vector(path):
    x = np.load(path, allow_pickle=True)
    if getattr(x, "shape", None) == () and getattr(x, "dtype", None) == object:
        obj = x.item()
        if isinstance(obj, dict):
            for key in ("feature", "features", "embedding", "embeddings", "x"):
                if key in obj:
                    x = obj[key]
                    break
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    return x


class CareDataset(Dataset):
    def __init__(self, manifest_csv, metadata_tuple, split, mean=None, std=None):
        self.manifest = pd.read_csv(manifest_csv)
        self.manifest = self.manifest[self.manifest["split"] == split].reset_index(drop=True)
        self.metadata, self.pheno, self.genes, self.program_scores = metadata_tuple
        self.phenotypes = self.metadata["phenotypes"]
        self.vocab = self.metadata["vocab"]
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.manifest)

    def _phenotype_targets(self, case_id):
        max_dim = max([int(p.get("num_bins", 1)) for p in self.phenotypes] + [1])
        values = np.zeros((len(self.phenotypes), max_dim), dtype=np.float32)
        masks = np.zeros((len(self.phenotypes), max_dim), dtype=np.float32)
        row = self.pheno.loc[case_id]
        for i, p in enumerate(self.phenotypes):
            v = row[p["name"]]
            if p["task_type"] == "discrete_survival":
                labels, bin_masks = discrete_survival_labels(v, row[p["time_name"]], p.get("time_bins", SURVIVAL_TIME_BINS))
                n = len(labels)
                values[i, :n] = labels
                masks[i, :n] = bin_masks
                continue
            if p["task_type"] in {"binary", "survival"}:
                y = parse_binary(v)
            elif p["task_type"] == "regression":
                y = pd.to_numeric(pd.Series([v]), errors="coerce").iloc[0]
            else:
                enc = self.vocab["label_encoders"].get(p["name"], {})
                if "score_class_field" in p:
                    label = score_class_label(p["score_class_field"], v)
                    y = enc.get(label, np.nan) if not pd.isna(label) else np.nan
                else:
                    y = enc.get(str(v), np.nan) if not is_missing_value(v) else np.nan
            values[i, 0] = 0.0 if pd.isna(y) else float(y)
            masks[i, 0] = 0.0 if pd.isna(y) else 1.0
        return torch.tensor(values, dtype=torch.float32), torch.tensor(masks, dtype=torch.float32)

    def __getitem__(self, idx):
        row = self.manifest.iloc[idx]
        x = load_care_vector(row["feature_path"])
        if self.mean is not None and self.std is not None:
            x = (x - self.mean) / np.maximum(self.std, 1e-6)
        case_id = row["case_id"]
        pheno_y, pheno_mask = self._phenotype_targets(case_id)
        program_values = self.program_scores.loc[case_id, self.vocab["program_names"]].astype(float)
        prog_y = torch.tensor(program_values.fillna(0.0).values, dtype=torch.float32)
        prog_mask = torch.tensor(program_values.notna().astype(float).values, dtype=torch.float32)
        gene_values = self.genes.loc[case_id, self.vocab["gene_list"]].astype(float)
        gene_y = torch.tensor(gene_values.fillna(0.0).values, dtype=torch.float32)
        gene_mask = torch.tensor(gene_values.notna().astype(float).values, dtype=torch.float32)
        return {
            "features": torch.from_numpy(x.astype(np.float32)),
            "phenotype_targets": pheno_y,
            "phenotype_masks": pheno_mask,
            "program_targets": prog_y,
            "program_masks": prog_mask,
            "gene_targets": gene_y,
            "gene_masks": gene_mask,
            "case_id": case_id,
            "slide_id": row["slide_id"],
        }


def collate(batch):
    out = {}
    for key in batch[0]:
        vals = [b[key] for b in batch]
        out[key] = torch.stack(vals) if torch.is_tensor(vals[0]) else vals
    return out


class CareBaseline(nn.Module):
    def __init__(self, input_dim, hidden_dim, phenotype_specs, num_programs, num_genes, model_type="mlp", dropout=0.25):
        super().__init__()
        if model_type == "linear":
            self.encoder = nn.Identity()
            rep_dim = input_dim
        else:
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
            )
            rep_dim = hidden_dim
        self.program_head = nn.Linear(rep_dim, num_programs)
        self.gene_head = nn.Linear(rep_dim, num_genes)
        self.pheno_heads = nn.ModuleList()
        for spec in phenotype_specs:
            if spec["task_type"] == "multiclass":
                out_dim = int(spec.get("num_classes", 1))
            elif spec["task_type"] == "discrete_survival":
                out_dim = int(spec.get("num_bins", 1))
            else:
                out_dim = 1
            self.pheno_heads.append(nn.Linear(rep_dim, out_dim))

    def forward(self, x):
        z = self.encoder(x.float())
        return {
            "program_pred": self.program_head(z),
            "gene_pred": self.gene_head(z),
            "phenotype_logits": [head(z) for head in self.pheno_heads],
        }


def phenotype_loss(outputs, targets, masks, phenotype_specs, group_weights):
    total = targets.new_tensor(0.0)
    weight_sum = targets.new_tensor(0.0)
    group_totals, group_counts = {}, {}
    for i, spec in enumerate(phenotype_specs):
        mask_i = masks[:, i]
        if not (mask_i > 0).any():
            continue
        group = spec.get("group", "morphology")
        group_weight = float(group_weights.get(group, 1.0))
        if group_weight <= 0:
            continue
        logits = outputs["phenotype_logits"][i]
        y = targets[:, i]
        if spec["task_type"] == "discrete_survival":
            n_bins = int(spec.get("num_bins", logits.shape[-1]))
            bin_mask = mask_i[:, :n_bins].float()
            hazard_logits = logits[:, :n_bins]
            event_targets = y[:, :n_bins].float()
            log_likelihood = (
                event_targets * F.logsigmoid(hazard_logits)
                + (1.0 - event_targets) * F.logsigmoid(-hazard_logits)
            )
            per_sample_nll = -(log_likelihood * bin_mask).sum(dim=1)
            valid_samples = bin_mask.sum(dim=1) > 0
            loss = per_sample_nll[valid_samples].mean()
        else:
            valid = mask_i[:, 0] > 0
            if not valid.any():
                continue
            logits = logits[valid]
            y = y[valid, 0]
            if spec["task_type"] in {"binary", "survival"}:
                loss = F.binary_cross_entropy_with_logits(logits.reshape(-1), y.reshape(-1))
            elif spec["task_type"] == "regression":
                loss = F.huber_loss(logits.reshape(-1), y.reshape(-1))
            else:
                loss = F.cross_entropy(logits, y.long())
        total = total + group_weight * loss
        weight_sum = weight_sum + group_weight
        group_totals[group] = group_totals.get(group, 0.0) + float(loss.detach().cpu())
        group_counts[group] = group_counts.get(group, 0) + 1
    group_losses = {g: group_totals[g] / max(group_counts[g], 1) for g in group_totals}
    return total / weight_sum.clamp_min(1.0), group_losses


def program_loss(outputs, targets, masks):
    losses = F.huber_loss(outputs["program_pred"], targets, reduction="none")
    denom = masks.float().sum().clamp_min(1.0)
    return (losses * masks.float()).sum() / denom


def gene_loss(outputs, targets, masks):
    losses = F.smooth_l1_loss(outputs["gene_pred"], targets, reduction="none")
    denom = masks.float().sum().clamp_min(1.0)
    return (losses * masks.float()).sum() / denom


def total_loss(outputs, batch, phenotype_specs, group_weights, lambda_program, lambda_gene):
    lp, group_losses = phenotype_loss(
        outputs, batch["phenotype_targets"], batch["phenotype_masks"], phenotype_specs, group_weights
    )
    lprog = program_loss(outputs, batch["program_targets"], batch["program_masks"])
    lgene = gene_loss(outputs, batch["gene_targets"], batch["gene_masks"])
    total = lp + lambda_program * lprog + lambda_gene * lgene
    return total, {"phenotype": lp, "program": lprog, "gene": lgene, **group_losses}


def move_batch(batch, device):
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def evaluate(model, loader, device, phenotype_specs, program_names):
    model.eval()
    pheno_true = [[] for _ in phenotype_specs]
    pheno_pred = [[] for _ in phenotype_specs]
    pheno_mask = [[] for _ in phenotype_specs]
    pheno_cases = [[] for _ in phenotype_specs]
    prog_true = [[] for _ in program_names]
    prog_pred = [[] for _ in program_names]
    gene_names = getattr(loader.dataset, "vocab", {}).get("gene_list", [])
    gene_true = [[] for _ in gene_names]
    gene_pred = [[] for _ in gene_names]
    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            out = model(batch["features"])
            for i, spec in enumerate(phenotype_specs):
                mask_i = batch["phenotype_masks"][:, i]
                if not (mask_i > 0).any():
                    continue
                logits = out["phenotype_logits"][i]
                if spec["task_type"] == "discrete_survival":
                    n_bins = int(spec.get("num_bins", logits.shape[-1]))
                    y = batch["phenotype_targets"][:, i, :n_bins].detach().cpu().numpy()
                    pred = torch.sigmoid(logits[:, :n_bins]).detach().cpu().numpy()
                    m = mask_i[:, :n_bins].detach().cpu().numpy()
                else:
                    valid = mask_i[:, 0] > 0
                    if not valid.any():
                        continue
                    y = batch["phenotype_targets"][valid, i, 0].detach().cpu().numpy().tolist()
                    logits = logits[valid]
                    if spec["task_type"] in {"binary", "survival"}:
                        pred = torch.sigmoid(logits.reshape(-1)).detach().cpu().numpy().tolist()
                    elif spec["task_type"] == "multiclass":
                        pred = torch.argmax(logits, dim=1).detach().cpu().numpy().tolist()
                    else:
                        pred = logits.reshape(-1).detach().cpu().numpy().tolist()
                    m = 1.0
                pheno_true[i].extend(y)
                pheno_pred[i].extend(pred)
                if spec["task_type"] == "discrete_survival":
                    pheno_mask[i].extend(m)
                    pheno_cases[i].extend(batch["case_id"])
            for i in range(len(program_names)):
                valid = batch["program_masks"][:, i] > 0
                if valid.any():
                    prog_true[i].extend(batch["program_targets"][valid, i].detach().cpu().numpy().tolist())
                    prog_pred[i].extend(out["program_pred"][valid, i].detach().cpu().numpy().tolist())
            for i in range(len(gene_names)):
                valid = batch["gene_masks"][:, i] > 0
                if valid.any():
                    gene_true[i].extend(batch["gene_targets"][valid, i].detach().cpu().numpy().tolist())
                    gene_pred[i].extend(out["gene_pred"][valid, i].detach().cpu().numpy().tolist())
    rows = []
    for i, spec in enumerate(phenotype_specs):
        if spec["task_type"] == "discrete_survival":
            if pheno_true[i]:
                case_ids, y_arr, p_arr, m_arr = aggregate_survival_by_case(
                    pheno_cases[i], pheno_true[i], pheno_pred[i], pheno_mask[i]
                )
                m_arr = m_arr > 0
                aucs = []
                labels = spec.get("bin_labels", [f"bin{k+1}" for k in range(y_arr.shape[1])])
                for k in range(y_arr.shape[1]):
                    valid = m_arr[:, k]
                    val = float("nan")
                    if valid.sum() >= 2 and len(set(y_arr[valid, k].tolist())) >= 2:
                        val = float(roc_auc_score(y_arr[valid, k], p_arr[valid, k]))
                        aucs.append(val)
                    rows.append({"task_name": f"{spec['name']}::{labels[k]}", "task_type": spec["task_type"], "metric_name": "AUC", "metric_value": val, "n": int(valid.sum())})
                rows.append({"task_name": f"{spec['name']}::mean", "task_type": spec["task_type"], "metric_name": "AUC", "metric_value": float(np.mean(aucs)) if aucs else float("nan"), "n": len(case_ids)})
                horizon = float(spec.get("time_bins", [3650.0])[-1])
                observed_times, observed_events, risks = [], [], []
                for case_id, hazards in zip(case_ids, p_arr):
                    clinical_row = loader.dataset.pheno.loc[case_id]
                    if isinstance(clinical_row, pd.DataFrame):
                        clinical_row = clinical_row.iloc[0]
                    event = parse_binary(clinical_row[spec["name"]])
                    time_value = pd.to_numeric(
                        pd.Series([clinical_row[spec["time_name"]]]), errors="coerce"
                    ).iloc[0]
                    if pd.isna(event) or pd.isna(time_value):
                        continue
                    observed_times.append(min(float(time_value), horizon))
                    observed_events.append(float(event > 0 and float(time_value) <= horizon))
                    risks.append(hazards_to_rmst_risk(hazards, spec.get("time_bins", [horizon])))
                c_index, comparable_pairs = harrell_c_index(
                    np.asarray(observed_times), np.asarray(observed_events), np.asarray(risks)
                )
                rows.append({
                    "task_name": f"{spec['name']}::c_index",
                    "task_type": spec["task_type"],
                    "metric_name": "C-index",
                    "metric_value": c_index,
                    "n": comparable_pairs,
                })
            else:
                rows.append({"task_name": spec["name"], "task_type": spec["task_type"], "metric_name": "AUC", "metric_value": float("nan"), "n": 0})
            continue
        metric, val = metric_safe(spec["task_type"], pheno_true[i], pheno_pred[i])
        rows.append({
            "task_name": spec["name"],
            "task_type": spec["task_type"],
            "metric_name": metric,
            "metric_value": val,
            "n": len(pheno_true[i]),
        })
    for i, name in enumerate(program_names):
        metric, val = metric_safe("regression", prog_true[i], prog_pred[i])
        rows.append({
            "task_name": name,
            "task_type": "gene_program_regression",
            "metric_name": metric,
            "metric_value": val,
            "n": len(prog_true[i]),
        })
    for i, name in enumerate(gene_names):
        rows.append({
            "task_name": name,
            "task_type": "gene_expression_regression",
            "metric_name": "Pearson",
            "metric_value": pearson_safe(gene_true[i], gene_pred[i]),
            "n": len(gene_true[i]),
        })
    return rows


def phenotype_selection_score(rows):
    values = []
    for row in rows:
        task_type = row["task_type"]
        if task_type not in {"binary", "multiclass", "discrete_survival"}:
            continue
        if task_type == "discrete_survival" and not row["task_name"].endswith("::mean"):
            continue
        value = float(row["metric_value"])
        if np.isfinite(value):
            values.append(value)
    return float(np.mean(values)) if values else float("nan")


def gene_pearson_score(rows):
    values = [
        float(row["metric_value"]) for row in rows
        if row["task_type"] == "gene_expression_regression"
        and np.isfinite(row["metric_value"])
    ]
    return float(np.mean(values)) if values else float("nan")


def care_validation_summary(rows):
    def mean_values(predicate):
        values = [
            float(row["metric_value"]) for row in rows
            if predicate(row) and np.isfinite(row["metric_value"])
        ]
        return float(np.mean(values)) if values else float("nan")

    return {
        "binary_auc": mean_values(lambda row: row["task_type"] == "binary"),
        "multiclass_acc": mean_values(lambda row: row["task_type"] == "multiclass"),
        "os_mean_auc": mean_values(
            lambda row: row["task_type"] == "discrete_survival"
            and row["task_name"].endswith("::mean")
        ),
        "c_index": mean_values(
            lambda row: row["task_type"] == "discrete_survival"
            and row["task_name"].endswith("::c_index")
        ),
        "gene_pearson": mean_values(lambda row: row["task_type"] == "gene_expression_regression"),
        "program_mae": mean_values(lambda row: row["task_type"] == "gene_program_regression"),
    }


def feature_stats(manifest):
    xs = [load_care_vector(p) for p in manifest.loc[manifest["split"] == "train", "feature_path"]]
    x = np.stack(xs).astype(np.float32)
    return x.mean(axis=0), x.std(axis=0)


def summarize_against_reference(out_dir, ref_metrics):
    if not ref_metrics:
        return
    ours = pd.read_csv(Path(out_dir) / "tool_metrics.csv")
    ref = pd.read_csv(ref_metrics)
    ours["metric_value"] = pd.to_numeric(ours["metric_value"], errors="coerce")
    ref["metric_value"] = pd.to_numeric(ref["metric_value"], errors="coerce")
    merged = ours.merge(
        ref[["task_name", "metric_value"]],
        on="task_name",
        how="inner",
        suffixes=("_care", "_reference"),
    )
    def delta(row):
        if pd.isna(row.metric_value_care) or pd.isna(row.metric_value_reference):
            return np.nan
        if row.metric_name == "MAE":
            return row.metric_value_reference - row.metric_value_care
        return row.metric_value_care - row.metric_value_reference
    merged["care_minus_reference"] = merged.apply(delta, axis=1)
    merged.to_csv(Path(out_dir) / "care_vs_reference_metrics.csv", index=False, encoding="utf-8-sig")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label_dir", required=True)
    ap.add_argument("--program_scores_csv", required=True)
    ap.add_argument("--care_feature_dir", required=True)
    ap.add_argument("--out_dir", default="outputs/care_baseline")
    ap.add_argument("--reference_metrics", default=None)
    ap.add_argument("--split_csv", default=None)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--hidden_dim", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.25)
    ap.add_argument("--model_type", choices=["linear", "mlp"], default="mlp")
    ap.add_argument("--lambda_program", type=float, default=0.5)
    ap.add_argument("--lambda_gene", type=float, default=0.1)
    ap.add_argument("--phenotype_schedule", choices=["staged", "flat", "molecular_first", "no_clinical"], default="staged")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--log_every", type=int, default=10)
    args = ap.parse_args()
    seed_all(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    split_csv = args.split_csv
    if split_csv is None:
        default_split = Path("outputs/full_run_ddp_low_prior/case_splits.csv")
        split_csv = str(default_split) if default_split.exists() else None
    manifest = prepare_care_manifest(args.label_dir, args.care_feature_dir, out_dir, seed=args.seed, split_csv=split_csv)
    gene_programs_json = Path(__file__).resolve().parents[1] / "configs/gene_programs.json"
    metadata_tuple = build_metadata(
        args.label_dir,
        out_dir / "care_manifest.csv",
        gene_programs_json,
        out_dir,
        program_scores_csv=args.program_scores_csv,
    )
    metadata, _, _, _ = metadata_tuple
    mean, std = feature_stats(manifest)
    np.savez(out_dir / "care_feature_normalization.npz", mean=mean, std=std)

    datasets = {
        s: CareDataset(out_dir / "care_manifest.csv", metadata_tuple, s, mean=mean, std=std)
        for s in ["train", "val", "test"]
    }
    loaders = {
        "train": DataLoader(datasets["train"], batch_size=args.batch_size, shuffle=True, collate_fn=collate),
        "val": DataLoader(datasets["val"], batch_size=args.batch_size, shuffle=False, collate_fn=collate),
        "test": DataLoader(datasets["test"], batch_size=args.batch_size, shuffle=False, collate_fn=collate),
    }
    input_dim = int(load_care_vector(manifest.iloc[0]["feature_path"]).shape[0])
    device = select_device(args.device, verbose=True)
    model = CareBaseline(
        input_dim,
        args.hidden_dim,
        metadata["phenotypes"],
        len(metadata["vocab"]["program_names"]),
        len(metadata["vocab"]["gene_list"]),
        model_type=args.model_type,
        dropout=args.dropout,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    history = []
    epoch_metrics = []
    best_phenotype_score = float("-inf")
    best_checkpoint_path = out_dir / "best_phenotype_model.pt"
    print(
        f"CARE baseline: slides={len(manifest)} split={manifest.split.value_counts().to_dict()} "
        f"input_dim={input_dim} model={args.model_type}",
        flush=True,
    )
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        group_weights = phenotype_group_weights_for_epoch(epoch, args.epochs, args.phenotype_schedule)
        for step, batch in enumerate(loaders["train"], start=1):
            batch = move_batch(batch, device)
            opt.zero_grad(set_to_none=True)
            out = model(batch["features"])
            loss, parts = total_loss(
                out, batch, metadata["phenotypes"], group_weights, args.lambda_program, args.lambda_gene
            )
            if torch.isfinite(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
                losses.append(float(loss.detach().cpu()))
            if args.log_every > 0 and (step == 1 or step % args.log_every == 0):
                print(f"epoch {epoch}/{args.epochs} step {step}/{len(loaders['train'])} loss={np.mean(losses[-args.log_every:]):.4f}", flush=True)
        val_rows = evaluate(model, loaders["val"], device, metadata["phenotypes"], metadata["vocab"]["program_names"])
        val_summary = pd.DataFrame(val_rows)
        val_phenotype_score = phenotype_selection_score(val_rows)
        val_gene_pearson = gene_pearson_score(val_rows)
        val_metrics = care_validation_summary(val_rows)
        epoch_metrics.extend({"epoch": epoch, **metric_row} for metric_row in val_rows)
        if np.isfinite(val_phenotype_score) and val_phenotype_score > best_phenotype_score:
            best_phenotype_score = val_phenotype_score
            torch.save({
                "state_dict": model.state_dict(),
                "epoch": epoch,
                "phenotype_score": val_phenotype_score,
            }, best_checkpoint_path)
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)) if losses else float("nan"),
            "val_auc_mean": pd.to_numeric(val_summary.loc[val_summary.metric_name == "AUC", "metric_value"], errors="coerce").mean(),
            "val_acc_mean": pd.to_numeric(val_summary.loc[val_summary.metric_name == "ACC", "metric_value"], errors="coerce").mean(),
            "val_mae_mean": pd.to_numeric(val_summary.loc[val_summary.metric_name == "MAE", "metric_value"], errors="coerce").mean(),
            "val_phenotype_score": val_phenotype_score,
            "val_gene_pearson": val_gene_pearson,
            "val_binary_auc": val_metrics["binary_auc"],
            "val_multiclass_acc": val_metrics["multiclass_acc"],
            "val_os_mean_auc": val_metrics["os_mean_auc"],
            "val_c_index": val_metrics["c_index"],
            "val_program_mae": val_metrics["program_mae"],
            "phenotype_weights": format_group_weights(group_weights),
        }
        history.append(row)
        print(
            f"epoch {epoch}: train_loss={row['train_loss']:.4f} "
            f"binary_auc={val_metrics['binary_auc']:.4f} multiclass_acc={val_metrics['multiclass_acc']:.4f} "
            f"os_mean_auc={val_metrics['os_mean_auc']:.4f} c_index={val_metrics['c_index']:.4f} "
            f"gene_pearson={val_gene_pearson:.4f} program_mae={val_metrics['program_mae']:.4f} "
            f"phenotype_score={val_phenotype_score:.4f}",
            flush=True,
        )

    pd.DataFrame(history).to_csv(out_dir / "train_history.csv", index=False)
    pd.DataFrame(epoch_metrics).to_csv(out_dir / "epoch_metrics.csv", index=False)
    if best_checkpoint_path.exists():
        try:
            checkpoint = torch.load(best_checkpoint_path, map_location=device, weights_only=True)
        except TypeError:
            checkpoint = torch.load(best_checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["state_dict"])
        print(
            f"Restored best CARE checkpoint: epoch={checkpoint['epoch']} "
            f"phenotype_score={checkpoint['phenotype_score']:.4f}",
            flush=True,
        )
    rows = evaluate(model, loaders["val"], device, metadata["phenotypes"], metadata["vocab"]["program_names"])
    split_counts = manifest.split.value_counts().to_dict()
    with (out_dir / "tool_metrics.csv").open("w", newline="", encoding="utf-8-sig") as f:
        fieldnames = ["task_name", "task_type", "num_train", "num_val", "num_test", "metric_name", "metric_value", "missing_rate"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({
                "task_name": r["task_name"],
                "task_type": r["task_type"],
                "num_train": split_counts.get("train", 0),
                "num_val": split_counts.get("val", 0),
                "num_test": split_counts.get("test", 0),
                "metric_name": r["metric_name"],
                "metric_value": r["metric_value"],
                "missing_rate": "",
            })
    torch.save({
        "state_dict": model.state_dict(),
        "args": vars(args),
        "input_dim": input_dim,
        "phenotype_specs": metadata["phenotypes"],
        "program_names": metadata["vocab"]["program_names"],
        "gene_names": metadata["vocab"]["gene_list"],
    }, out_dir / "care_baseline.pt")
    with (out_dir / "train_config.json").open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)
    summarize_against_reference(out_dir, args.reference_metrics)
    print(f"CARE baseline exported: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
