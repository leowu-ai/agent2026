#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from data.dataset import prepare_manifest, infer_feature_dim, read_label_tables


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--label_dir', required=True)
    ap.add_argument('--feature_dir', required=True)
    ap.add_argument('--out_dir', default='outputs')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    manifest, stats = prepare_manifest(args.label_dir, args.feature_dir, out_dir, seed=args.seed)
    pheno, genes, mapping, missing = read_label_tables(args.label_dir)
    feature_dim = infer_feature_dim(out_dir / 'aligned_manifest.csv') if len(manifest) else None
    print(f'CSV check: phenotype={pheno.shape}, gene={genes.shape}, mapping={mapping.shape}, missing={missing.shape}')
    print(f'Feature files: {stats["feature_files"]}; missing feature cases: {stats["missing_feature_cases"]}')
    print(f'Aligned cases: {manifest.case_id.nunique()}')
    print(f'Aligned slides: {len(manifest)}')
    print(f'Splits: {manifest.split.value_counts().to_dict()}')
    print(f'Feature dim: {feature_dim}')
    print(f'Manifest saved: {out_dir / "aligned_manifest.csv"}')

if __name__ == '__main__':
    main()
