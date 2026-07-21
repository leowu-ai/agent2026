import json
import math
import random
import re
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


PHENO_FILE = "BRCA_表型标签__标签数据.csv"
GENE_FILE = "BRCA_表型相关基因标签__纯基因标签.csv"
STRICT_PATHWAY_GENE_FILE = "BRCA_strict_pathway_gene_labels.csv"
STRICT_PATHWAY_PHENOTYPE_COVERAGE_FILE = "BRCA_strict_pathway_phenotype_coverage.csv"
MAP_FILE = "BRCA_表型相关基因标签__标签-基因映射.csv"
MISSING_FILE = "BRCA_表型标签__字段缺失分组.csv"
CASE_COL = "病例ID（case_id）"
SKIP_PHENOTYPES = {
    CASE_COL,
    "癌种（cancer_type）",
    "组织学类型中文（histological_type_zh）",
    "生存状态（vital_status）",
    "肿瘤状态（tumor_status）",
}
ALLOWED_PHENOTYPE_FIELDS = {
    "histological_type_label",
    "ductal_binary",
    "lobular_binary",
    "dcis_binary",
    "lcis_binary",
    "histologic_grade_label",
    "lymphovascular_invasion_label",
    "necrosis_binary",
    "comedonecrosis_binary",
    "microcalcification_binary",
    "nottingham_total_score",
    "mitotic_score",
    "ER_status_label",
    "PR_status_label",
    "HER2_status_label",
    "ajcc_pathologic_stage",
    "OS",
    "OS_time",
}
SURVIVAL_TIME_BINS = [365.0, 1095.0, 1825.0, 3650.0]
SURVIVAL_BIN_LABELS = ["0-1y", "1-3y", "3-5y", "5-10y"]
UNAVAILABLE = {"", "nan", "none", "na", "n/a", "unknown", "stage x", "[discrepancy]", "discrepancy", "equivocal", "not reported", "not available", "not applicable"}


def stable_field_name(column):
    m = re.search(r"（([^（）]+)）", str(column))
    return m.group(1) if m else str(column)


def is_missing_value(value):
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return str(value).strip().lower() in UNAVAILABLE


def read_label_tables(label_dir):
    label_dir = Path(label_dir).expanduser()
    pheno = pd.read_csv(label_dir / PHENO_FILE)
    strict_gene_path = label_dir / STRICT_PATHWAY_GENE_FILE
    genes = pd.read_csv(strict_gene_path if strict_gene_path.exists() else label_dir / GENE_FILE)
    mapping = pd.read_csv(label_dir / MAP_FILE)
    missing = pd.read_csv(label_dir / MISSING_FILE)
    pheno[CASE_COL] = pheno[CASE_COL].astype(str).str[:12]
    genes["case_id"] = genes["case_id"].astype(str).str[:12]
    return pheno, genes, mapping, missing


def load_full_rna_expression(rna_expression_csv, min_background_genes=200):
    path = Path(rna_expression_csv).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Full RNA expression matrix not found: {path}")
    if path.suffix.lower() == ".parquet":
        table = pd.read_parquet(path)
    else:
        sep = "\t" if path.suffix.lower() in {".tsv", ".txt"} else ","
        table = pd.read_csv(path, sep=sep)

    case_candidates = ["case_id", CASE_COL, "sample_id", "expression_sample_id"]
    id_col = next((col for col in case_candidates if col in table.columns), None)
    if id_col is None:
        raise ValueError(
            "RNA matrix must be samples x genes and include one identifier column: "
            "case_id, sample_id, or expression_sample_id."
        )
    case_ids = table[id_col].astype(str).str[:12]
    id_columns = {col for col in case_candidates if col in table.columns}
    expression = table[[col for col in table.columns if col not in id_columns]].apply(
        pd.to_numeric, errors="coerce"
    )
    expression = expression.loc[:, expression.notna().any(axis=0)]
    if expression.shape[1] < int(min_background_genes):
        raise ValueError(
            f"ssGSEA background is too small; found only {expression.shape[1]} numeric genes "
            f"in {path}. Provide an RNA matrix with at least {min_background_genes} genes."
        )
    expression.insert(0, "case_id", case_ids.values)
    expression = expression.groupby("case_id", as_index=False).mean(numeric_only=True)
    return expression


def compute_ssgsea_program_scores(
    rna_expression_csv,
    gene_programs_json,
    out_csv,
    threads=4,
    min_gene_set_size=3,
    min_background_genes=200,
):
    try:
        import gseapy as gp
    except ImportError as exc:
        raise ImportError(
            "gseapy is required for ssGSEA. Install it in the training environment with: pip install gseapy"
        ) from exc

    expression = load_full_rna_expression(rna_expression_csv, min_background_genes)
    with open(gene_programs_json, encoding="utf-8") as handle:
        programs = json.load(handle)

    gene_columns = set(expression.columns) - {"case_id"}
    gene_sets = {}
    audit_rows = []
    for program in programs:
        name = program["program_name"]
        requested = list(dict.fromkeys(program.get("member_genes", [])))
        available = [gene for gene in requested if gene in gene_columns]
        audit_rows.append({
            "program_name": name,
            "n_requested_genes": len(requested),
            "n_available_genes": len(available),
            "missing_genes": ";".join(gene for gene in requested if gene not in gene_columns),
        })
        if len(available) < int(min_gene_set_size):
            raise ValueError(
                f"Pathway {name!r} has only {len(available)} genes in the full RNA matrix; "
                f"minimum is {min_gene_set_size}."
            )
        gene_sets[name] = available

    matrix = expression.set_index("case_id").transpose()
    matrix = matrix.apply(lambda row: row.fillna(row.median()), axis=1).fillna(0.0)
    result = gp.ssgsea(
        data=matrix,
        gene_sets=gene_sets,
        outdir=None,
        sample_norm_method="rank",
        correl_norm_type="rank",
        min_size=int(min_gene_set_size),
        max_size=max(len(genes) for genes in gene_sets.values()) + 1,
        permutation_num=0,
        weight=0.25,
        ascending=False,
        threads=max(1, int(threads)),
        no_plot=True,
        seed=42,
        verbose=False,
    )
    long_scores = result.res2d.copy()
    long_scores["ES"] = pd.to_numeric(long_scores["ES"], errors="coerce")
    scores = long_scores.pivot(index="Name", columns="Term", values="ES")
    program_names = [program["program_name"] for program in programs]
    scores = scores.reindex(columns=program_names)
    scores.index = scores.index.astype(str).str[:12]
    scores = scores.groupby(level=0).mean()
    scores.index.name = "case_id"

    out_path = Path(out_csv).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    scores.reset_index().to_csv(out_path, index=False)
    pd.DataFrame(audit_rows).to_csv(
        out_path.with_name(f"{out_path.stem}_gene_audit.csv"), index=False, encoding="utf-8-sig"
    )
    with out_path.with_name(f"{out_path.stem}_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump({
            "method": "ssGSEA",
            "background_scope": "full_transcriptome" if matrix.shape[0] >= 1000 else "restricted_pathway_panel",
            "implementation": f"gseapy {gp.__version__}",
            "score_column": "ES",
            "sample_norm_method": "rank",
            "correl_norm_type": "rank",
            "weight": 0.25,
            "n_cases": int(scores.shape[0]),
            "n_background_genes": int(matrix.shape[0]),
            "n_programs": int(scores.shape[1]),
            "rna_expression_csv": str(Path(rna_expression_csv).expanduser()),
            "gene_programs_json": str(Path(gene_programs_json).expanduser()),
        }, handle, ensure_ascii=False, indent=2)
    return scores.reset_index()


def discover_feature_files(feature_dir):
    feature_dir = Path(feature_dir).expanduser()
    paths = []
    for ext in ("*.npy", "*.pt", "*.pth", "*.h5", "*.hdf5"):
        paths.extend(feature_dir.rglob(ext))
    rows = []
    for p in sorted(paths):
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
    split = {}
    for i, case_id in enumerate(case_ids):
        split[case_id] = "train" if i < n_train else ("val" if i < n_train + n_val else "test")
    return split


def prepare_manifest(label_dir, feature_dir, out_dir, seed=42, split_csv=None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pheno, genes, _, _ = read_label_tables(label_dir)
    features = discover_feature_files(feature_dir)
    pheno_cases = set(pheno[CASE_COL].dropna().astype(str).str[:12])
    gene_cases = set(genes["case_id"].dropna().astype(str).str[:12])
    features["has_phenotype"] = features["case_id"].isin(pheno_cases)
    features["has_gene"] = features["case_id"].isin(gene_cases)
    aligned = features[features["has_phenotype"] & features["has_gene"]].copy()

    split_file = out_dir / "case_splits.csv"
    if split_csv is not None:
        splits = pd.read_csv(Path(split_csv).expanduser())
        required = {"case_id", "split"}
        if not required.issubset(splits.columns):
            raise ValueError(f"split_csv must contain columns {sorted(required)}")
        splits = splits[["case_id", "split"]].copy()
        splits["case_id"] = splits["case_id"].astype(str).str[:12]
        splits["split"] = splits["split"].astype(str)
        if splits["case_id"].duplicated().any():
            raise ValueError("split_csv contains duplicate case_id values")
        invalid = sorted(set(splits["split"]) - {"train", "val", "test"})
        if invalid:
            raise ValueError(f"split_csv contains invalid split values: {invalid}")
        split_map = dict(zip(splits["case_id"], splits["split"]))
        missing = sorted(set(aligned["case_id"]) - set(split_map))
        if missing:
            raise ValueError(f"split_csv is missing {len(missing)} aligned cases; examples: {missing[:5]}")
        pd.DataFrame({"case_id": sorted(set(aligned["case_id"]))}).assign(
            split=lambda frame: frame["case_id"].map(split_map)
        ).to_csv(split_file, index=False)
    elif split_file.exists():
        splits = pd.read_csv(split_file)
        split_map = dict(zip(splits["case_id"].astype(str), splits["split"].astype(str)))
    else:
        split_map = split_cases(aligned["case_id"], seed=seed)
        pd.DataFrame({"case_id": list(split_map.keys()), "split": list(split_map.values())}).to_csv(split_file, index=False)
    aligned["split"] = aligned["case_id"].map(split_map)
    aligned.to_csv(out_dir / "aligned_manifest.csv", index=False)
    missing_feature_cases = sorted((pheno_cases & gene_cases) - set(features["case_id"]))
    with (out_dir / "missing_feature_cases.txt").open("w", encoding="utf-8") as f:
        f.write("\n".join(missing_feature_cases))
    return aligned, {"feature_files": len(features), "missing_feature_cases": len(missing_feature_cases)}


def load_feature_file(path):
    path = Path(path)
    if path.suffix == ".npy":
        obj = np.load(path, allow_pickle=True)
        if getattr(obj, "shape", None) == () and obj.dtype == object:
            obj = obj.item()
        if isinstance(obj, dict):
            for key in ("feature", "features", "feats", "embeddings", "x"):
                if key in obj:
                    arr = obj[key]
                    break
            else:
                arr = next(v for v in obj.values() if hasattr(v, "shape") and len(v.shape) == 2)
            coords = obj.get("coords", obj.get("coordinates", obj.get("index")))
        else:
            arr, coords = obj, None
    elif path.suffix in {".pt", ".pth"}:
        obj = torch.load(path, map_location="cpu")
        if isinstance(obj, dict):
            for key in ("feature", "features", "feats", "embeddings", "x"):
                if key in obj:
                    arr = obj[key]
                    break
            else:
                arr = next(v for v in obj.values() if hasattr(v, "shape") and len(v.shape) == 2)
            coords = obj.get("coords", obj.get("coordinates", obj.get("index")))
        else:
            arr, coords = obj, None
        if torch.is_tensor(arr):
            arr = arr.numpy()
    elif path.suffix in {".h5", ".hdf5"}:
        with h5py.File(path, "r") as h:
            key = next((k for k in ("features", "feature", "feats", "embeddings", "x") if k in h), None)
            if key is None:
                key = next(k for k in h.keys() if len(h[k].shape) == 2)
            arr = h[key][:]
            coords = h["coords"][:] if "coords" in h else None
    else:
        raise ValueError(f"Unsupported feature file: {path}")
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 2:
        arr = arr.reshape(arr.shape[0], -1)
    return arr, coords


def infer_feature_dim(manifest_csv):
    manifest = pd.read_csv(manifest_csv)
    arr, _ = load_feature_file(manifest.iloc[0]["feature_path"])
    return int(arr.shape[1])


SCORE_CLASS_LABELS = ["low", "intermediate", "high"]
SCORE_CLASS_FIELDS = {"nottingham_total_score", "mitotic_score"}


def score_class_label(field, value):
    if is_missing_value(value):
        return np.nan
    v = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(v):
        return np.nan
    field = str(field)
    if field == "nottingham_total_score":
        # Nottingham total score is usually interpreted as low=3-5,
        # intermediate=6-7, high=8-9. Values below 3 are kept as low;
        # obvious extraction artifacts above 9 are treated as missing.
        if v > 9:
            return np.nan
        if v <= 5:
            return "low"
        if v <= 7:
            return "intermediate"
        return "high"
    if field == "mitotic_score":
        if v <= 1:
            return "low"
        if v <= 2:
            return "intermediate"
        return "high"
    return np.nan


def classify_task(column):
    field = stable_field_name(column)
    lower = field.lower()
    if lower in {"os", "dss", "dfi", "pfi"}:
        return "survival"
    if lower.endswith("_time"):
        return "regression"
    if lower.endswith("_binary") or "二值" in column or "阳性二值" in column:
        return "binary"
    if lower in {"nottingham_total_score", "mitotic_score"}:
        return "multiclass"
    if any(x in lower for x in ["percent", "score"]) and not lower.endswith("ihc_score"):
        return "regression"
    return "multiclass"


def infer_phenotype_group(column, task_type):
    field = stable_field_name(column).lower()
    name = str(column).lower()
    text = f"{field} {name}"
    molecular_terms = ["er", "pr", "her2", "ki67", "hormone", "ihc", "amplification"]
    clinical_terms = ["os", "dss", "dfi", "pfi", "survival", "stage", "ajcc", "tumor_status", "vital_status"]
    if task_type == "survival" or any(term in text for term in clinical_terms):
        return "clinical"
    if any(term in text for term in molecular_terms):
        return "molecular"
    return "morphology"


def parse_binary(value):
    if is_missing_value(value):
        return np.nan
    if isinstance(value, (int, float)) and not math.isnan(float(value)):
        return float(value)
    s = str(value).strip().lower()
    if s in {"1", "yes", "true", "positive", "pos", "present", "dead", "deceased", "event"}:
        return 1.0
    if s in {"0", "no", "false", "negative", "neg", "absent", "alive", "censored"}:
        return 0.0
    try:
        return float(s)
    except ValueError:
        return np.nan


def discrete_survival_labels(event_value, time_value, bins=SURVIVAL_TIME_BINS):
    event = parse_binary(event_value)
    time = pd.to_numeric(pd.Series([time_value]), errors="coerce").iloc[0]
    n_bins = len(bins)
    labels = np.zeros(n_bins, dtype=np.float32)
    masks = np.zeros(n_bins, dtype=np.float32)
    if pd.isna(event) or pd.isna(time) or float(time) < 0:
        return labels, masks

    time = float(time)
    ends = list(bins)
    if event > 0 and time <= ends[-1]:
        event_bin = next(i for i, end in enumerate(ends) if time <= end)
        masks[: event_bin + 1] = 1.0
        labels[event_bin] = 1.0
    else:
        # A censored patient contributes only intervals completed before censoring.
        # Events after the finite horizon are known event-free through every bin.
        for i, end in enumerate(ends):
            if time >= end:
                masks[i] = 1.0
    return labels, masks


def build_metadata(label_dir, manifest_csv, gene_programs_json, out_dir, min_valid=5, program_scores_csv=None):
    out_dir = Path(out_dir)
    pheno, genes, mapping, missing = read_label_tables(label_dir)
    manifest = pd.read_csv(manifest_csv)
    train_cases = set(manifest.loc[manifest["split"] == "train", "case_id"])
    gene_cols = [c for c in genes.columns if c not in {"case_id", "expression_sample_id"}]
    gene_train = genes[genes["case_id"].isin(train_cases)]
    gene_mean = gene_train[gene_cols].astype(float).mean().fillna(0.0)
    gene_std = gene_train[gene_cols].astype(float).std().replace(0, 1).fillna(1.0)

    with open(gene_programs_json, encoding="utf-8") as f:
        raw_programs = json.load(f)
    gene_index = {g: i for i, g in enumerate(gene_cols)}
    programs = []
    gene_program_audit = []
    for prog in raw_programs:
        original_members = list(dict.fromkeys(prog.get("member_genes", [])))
        available_members = [g for g in original_members if g in gene_index]
        missing_members = [g for g in original_members if g not in gene_index]
        gene_program_audit.append({
            "program_name": prog.get("program_name", ""),
            "n_original_genes": len(original_members),
            "n_available_genes": len(available_members),
            "n_missing_genes": len(missing_members),
            "available_genes": ";".join(available_members),
            "missing_genes": ";".join(missing_members),
        })
        if not available_members:
            continue
        clean_prog = dict(prog)
        clean_prog["member_genes"] = available_members
        programs.append(clean_prog)
    pd.DataFrame(gene_program_audit).to_csv(out_dir / "gene_program_expression_check.csv", index=False, encoding="utf-8-sig")
    program_names = [p["program_name"] for p in programs]
    H_prior = np.zeros((len(gene_cols), len(program_names)), dtype=np.float32)
    for k, prog in enumerate(programs):
        for g in prog["member_genes"]:
            H_prior[gene_index[g], k] = 1.0

    z = (genes[gene_cols].astype(float) - gene_mean) / gene_std
    z = z.clip(-5.0, 5.0)
    gene_z_scores = z.copy()
    gene_z_scores.insert(0, "case_id", genes["case_id"].values)
    if program_scores_csv is None:
        raise ValueError(
            "Program targets now require offline ssGSEA scores. Pass program_scores_csv pointing to "
            "the raw case-by-pathway ES table generated with: python data/dataset.py "
            "--rna_expression_csv FULL_RNA.csv --out_csv ssgsea_program_scores_raw.csv"
        )
    score_path = Path(program_scores_csv).expanduser()
    if not score_path.exists():
        raise FileNotFoundError(f"ssGSEA program score table not found: {score_path}")
    raw_program_scores = pd.read_csv(score_path)
    score_case_col = next(
        (col for col in ("case_id", CASE_COL, "sample_id", "expression_sample_id") if col in raw_program_scores.columns),
        None,
    )
    if score_case_col is None:
        raise ValueError(f"ssGSEA score table {score_path} has no case_id column.")
    raw_program_scores["case_id"] = raw_program_scores[score_case_col].astype(str).str[:12]
    missing_programs = [name for name in program_names if name not in raw_program_scores.columns]
    if missing_programs:
        raise ValueError(
            f"ssGSEA score table is missing {len(missing_programs)} pathways: {missing_programs[:5]}"
        )
    raw_program_scores[program_names] = raw_program_scores[program_names].apply(
        pd.to_numeric, errors="coerce"
    )
    raw_program_scores = raw_program_scores.groupby("case_id", as_index=False)[program_names].mean()
    manifest_cases = set(manifest["case_id"].astype(str).str[:12])
    missing_score_cases = sorted(manifest_cases - set(raw_program_scores["case_id"]))
    if missing_score_cases:
        raise ValueError(
            f"ssGSEA score table is missing {len(missing_score_cases)} aligned cases; "
            f"examples: {missing_score_cases[:5]}"
        )
    if raw_program_scores[program_names].isna().any().any():
        bad = raw_program_scores[program_names].isna().sum()
        bad = bad[bad > 0].to_dict()
        raise ValueError(f"ssGSEA score table contains missing pathway scores: {bad}")

    program_scores = pd.DataFrame({"case_id": genes["case_id"]}).merge(
        raw_program_scores[["case_id"] + program_names], on="case_id", how="left"
    )
    prog_train = program_scores[program_scores["case_id"].isin(train_cases)]
    program_mean = prog_train[program_names].mean().fillna(0.0)
    program_std = prog_train[program_names].std().replace(0, 1).fillna(1.0)
    program_scores[program_names] = (program_scores[program_names] - program_mean) / program_std
    program_scores.to_csv(out_dir / "program_ssgsea_scores_standardized.csv", index=False)

    phenotypes = []
    label_encoders = {}
    pheno_by_case = pheno.set_index(CASE_COL)
    field_to_column = {stable_field_name(c): c for c in pheno.columns}
    for col in pheno.columns:
        field = stable_field_name(col)
        if col in SKIP_PHENOTYPES or field not in ALLOWED_PHENOTYPE_FIELDS or field == "OS_time":
            continue
        task_type = "discrete_survival" if field == "OS" else classify_task(col)
        series = pheno[col]
        if task_type == "discrete_survival":
            time_col = field_to_column.get("OS_time")
            if time_col is None:
                continue
            y = series.map(parse_binary)
            t = pd.to_numeric(pheno[time_col], errors="coerce")
            valid = y.notna() & t.notna()
        elif task_type in {"binary", "survival"}:
            y = series.map(parse_binary)
            valid = y.notna()
        elif task_type == "regression":
            y = pd.to_numeric(series, errors="coerce")
            valid = y.notna()
        else:
            if field in SCORE_CLASS_FIELDS:
                y = series.map(lambda v: score_class_label(field, v))
                valid = y.notna()
                classes = SCORE_CLASS_LABELS
            else:
                valid = ~series.map(is_missing_value)
                classes = sorted(series[valid].astype(str).unique().tolist())
            if len(classes) < 2:
                continue
            label_encoders[col] = {v: i for i, v in enumerate(classes)}
        if int(valid.sum()) >= min_valid:
            spec = {"name": col, "field": field, "task_type": task_type, "group": infer_phenotype_group(col, task_type)}
            if task_type == "discrete_survival":
                spec["time_field"] = "OS_time"
                spec["time_name"] = field_to_column["OS_time"]
                spec["time_bins"] = SURVIVAL_TIME_BINS
                spec["bin_labels"] = SURVIVAL_BIN_LABELS
                spec["num_bins"] = len(SURVIVAL_TIME_BINS)
            if task_type == "multiclass":
                spec["num_classes"] = len(label_encoders[col])
                if field in SCORE_CLASS_FIELDS:
                    spec["class_labels"] = SCORE_CLASS_LABELS
                    spec["score_class_field"] = field
            phenotypes.append(spec)

    R_prior = np.zeros((len(program_names), len(phenotypes)), dtype=np.float32)
    G_prior = np.zeros((len(gene_cols), len(phenotypes)), dtype=np.float32)
    map_by_field = {}
    for _, row in mapping.iterrows():
        field = str(row.get("字段名", ""))
        genes_text = str(row.get("核心推荐基因标签", ""))
        map_by_field[field] = set(re.findall(r"[A-Za-z0-9]+", genes_text))

    coverage_by_field = {}
    coverage_path = Path(label_dir).expanduser() / STRICT_PATHWAY_PHENOTYPE_COVERAGE_FILE
    if coverage_path.exists():
        coverage = pd.read_csv(coverage_path)
        for _, row in coverage.iterrows():
            field = str(row.get("phenotype_label", ""))
            pathways = [x.strip() for x in str(row.get("related_pathways", "")).split("|") if x.strip()]
            coverage_by_field[field] = set(pathways)

    program_sets = [set(prog["member_genes"]) for prog in programs]
    for j, p in enumerate(phenotypes):
        related_genes = set(g for g in map_by_field.get(p["field"], set()) if g in gene_index)
        for g in related_genes:
            G_prior[gene_index[g], j] = 1.0
        coverage_programs = coverage_by_field.get(p["field"], set())
        for k, prog in enumerate(programs):
            related_by_config = p["field"] in set(prog.get("related_phenotypes", []))
            related_by_coverage = prog["program_name"] in coverage_programs
            related_by_old_gene_map = bool(related_genes & program_sets[k])
            if related_by_config or related_by_coverage or related_by_old_gene_map:
                R_prior[k, j] = 1.0
                for g in prog["member_genes"]:
                    G_prior[gene_index[g], j] = 1.0

    norm = {
        "gene_mean": gene_mean.to_dict(),
        "gene_std": gene_std.to_dict(),
        "program_mean": program_mean.to_dict(),
        "program_std": program_std.to_dict(),
        "program_score_method": "ssGSEA_ES_train_zscore",
        "program_scores_source": str(score_path),
    }
    vocab = {
        "gene_list": gene_cols,
        "program_names": program_names,
        "phenotype_names": [p["name"] for p in phenotypes],
        "phenotype_fields": [p["field"] for p in phenotypes],
        "phenotype_task_types": {p["name"]: p["task_type"] for p in phenotypes},
        "phenotype_groups": {p["name"]: p.get("group", "morphology") for p in phenotypes},
        "label_encoders": label_encoders,
    }
    metadata = {
        "phenotypes": phenotypes,
        "vocab": vocab,
        "normalization": norm,
        "H_prior": H_prior,
        "R_prior": R_prior,
        "G_prior": G_prior,
        "programs": programs,
        "missing_summary": missing.to_dict(orient="records"),
    }
    with (out_dir / "vocab.json").open("w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    with (out_dir / "normalization.json").open("w", encoding="utf-8") as f:
        json.dump(norm, f, ensure_ascii=False, indent=2)
    np.savez(out_dir / "priors.npz", H_prior=H_prior, R_prior=R_prior, G_prior=G_prior)
    return metadata, pheno_by_case, gene_z_scores.set_index("case_id"), program_scores.set_index("case_id")


class G2PBRCAFeatureDataset(Dataset):
    def __init__(
        self, manifest_csv, label_dir, gene_programs_json, out_dir, split="train",
        max_samples=None, metadata=None, program_scores_csv=None, max_patches=None,
    ):
        self.split = split
        self.max_patches = max_patches
        self.manifest = pd.read_csv(manifest_csv)
        self.manifest = self.manifest[self.manifest["split"] == split].reset_index(drop=True)
        if max_samples:
            self.manifest = self.manifest.iloc[:max_samples].reset_index(drop=True)
        if metadata is None:
            self.metadata, self.pheno, self.genes, self.program_scores = build_metadata(
                label_dir, manifest_csv, gene_programs_json, out_dir,
                program_scores_csv=program_scores_csv,
            )
        elif isinstance(metadata, tuple):
            self.metadata, self.pheno, self.genes, self.program_scores = metadata
        else:
            self.metadata, self.pheno, self.genes, self.program_scores = build_metadata(
                label_dir, manifest_csv, gene_programs_json, out_dir,
                program_scores_csv=program_scores_csv,
            )
            self.metadata = metadata
        self.phenotypes = self.metadata["phenotypes"]
        self.vocab = self.metadata["vocab"]

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

    def _sample_patches(self, features, coords):
        if self.max_patches is None or len(features) <= self.max_patches:
            return features, coords
        n_patches = len(features)
        if self.split == "train":
            indices = np.random.choice(len(features), size=self.max_patches, replace=False)
        else:
            indices = np.linspace(0, len(features) - 1, num=self.max_patches, dtype=np.int64)
        features = features[indices]
        if coords is not None and len(coords) == n_patches:
            coords = np.asarray(coords)[indices]
        return features, coords

    def __getitem__(self, idx):
        row = self.manifest.iloc[idx]
        features, coords = load_feature_file(row["feature_path"])
        original_patch_count = len(features)
        features, coords = self._sample_patches(features, coords)
        case_id = row["case_id"]
        pheno_y, pheno_mask = self._phenotype_targets(case_id)
        program_values = self.program_scores.loc[case_id, self.vocab["program_names"]].astype(float)
        prog_y = torch.tensor(program_values.fillna(0.0).values, dtype=torch.float32)
        prog_mask = torch.tensor(program_values.notna().astype(float).values, dtype=torch.float32)
        gene_values = self.genes.loc[case_id, self.vocab["gene_list"]].astype(float)
        gene_y = torch.tensor(gene_values.fillna(0.0).values, dtype=torch.float32)
        gene_mask = torch.tensor(gene_values.notna().astype(float).values, dtype=torch.float32)
        return {
            "features": torch.from_numpy(features),
            "patch_count": len(features),
            "original_patch_count": original_patch_count,
            "phenotype_targets": pheno_y,
            "phenotype_masks": pheno_mask,
            "program_targets": prog_y,
            "program_masks": prog_mask,
            "gene_targets": gene_y,
            "gene_masks": gene_mask,
            "case_id": case_id,
            "slide_id": row["slide_id"],
            "feature_path": row["feature_path"],
            "coords": coords,
        }



def _ssgsea_cli():
    import argparse

    parser = argparse.ArgumentParser(description="Compute raw ssGSEA pathway ES labels from a sample-by-gene RNA matrix.")
    parser.add_argument("--rna_expression_csv", required=True)
    parser.add_argument(
        "--gene_programs_json",
        default=str(Path(__file__).resolve().parents[1] / "configs/gene_programs.json"),
    )
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--min_gene_set_size", type=int, default=3)
    parser.add_argument("--min_background_genes", type=int, default=200)
    args = parser.parse_args()
    scores = compute_ssgsea_program_scores(
        args.rna_expression_csv,
        args.gene_programs_json,
        args.out_csv,
        threads=args.threads,
        min_gene_set_size=args.min_gene_set_size,
        min_background_genes=args.min_background_genes,
    )
    print(
        f"ssGSEA scores written to {args.out_csv}: cases={len(scores)} "
        f"programs={len(scores.columns) - 1}",
        flush=True,
    )


if __name__ == "__main__":
    _ssgsea_cli()
