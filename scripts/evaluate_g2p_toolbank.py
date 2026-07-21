#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).resolve().parents[1]))
from data.dataset import G2PBRCAFeatureDataset, build_metadata, load_feature_file
from models.g2p_toolbank import G2PHypergraphToolBank
from utils.device import select_device


def one_item(batch): return batch[0]
def move_batch(batch, device): return {k: (v.to(device) if torch.is_tensor(v) else v) for k,v in batch.items()}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--label_dir', required=True); ap.add_argument('--program_scores_csv', required=True); ap.add_argument('--manifest_csv', required=True); ap.add_argument('--tool_dir', required=True)
    ap.add_argument('--out_dir', default=None); ap.add_argument('--split', default='test'); ap.add_argument('--topk_patches', type=int, default=20); ap.add_argument('--device', default='auto')
    ap.add_argument('--max_patches', type=int, default=None); ap.add_argument('--min_free_gb', type=float, default=6.0); ap.add_argument('--reserve_gb', type=float, default=4.0)
    args=ap.parse_args(); tool_dir=Path(args.tool_dir); out_dir=Path(args.out_dir) if args.out_dir else tool_dir.parent; out_dir.mkdir(parents=True, exist_ok=True)
    ckpt=torch.load(tool_dir/'model.pt', map_location='cpu')
    gene_programs_json=Path(__file__).resolve().parents[1]/'configs/gene_programs.json'
    metadata_tuple=build_metadata(
        args.label_dir,
        args.manifest_csv,
        gene_programs_json,
        out_dir,
        program_scores_csv=args.program_scores_csv,
    )
    metadata,_,_,_=metadata_tuple
    rel=np.load(tool_dir/'relations.npz')
    init_H = rel['H_gene_prior'] if 'H_gene_prior' in rel else rel['H_gene_to_program']
    model=G2PHypergraphToolBank(
        ckpt['feature_dim'],
        ckpt['hidden_dim'],
        ckpt['phenotype_specs'],
        len(metadata['vocab']['gene_list']),
        ckpt['program_names'],
        init_H,
        rel['R_prior'],
        gene_phenotype_prior=metadata.get('G_prior'),
        gene_names=metadata['vocab']['gene_list'],
        relation_init_mode=ckpt.get('relation_init_mode', 'prior'),
        relation_init_value=ckpt.get('relation_init_value', 0.5),
        relation_selection_mode=ckpt.get('relation_selection_mode', 'prior_guided'),
    )
    state = dict(ckpt['state_dict'])
    if 'gene_embeddings' in state and 'gene_prototypes' not in state:
        state['gene_prototypes'] = state.pop('gene_embeddings')
    if 'gene_prototypes' in state and 'gene_identity_embeddings' not in state:
        state['gene_identity_embeddings'] = state.pop('gene_prototypes')
    missing, unexpected = model.load_state_dict(state, strict=False)
    allowed_missing = {
        k for k in missing
        if k.startswith('gene_head.')
        or k.startswith('rna_value_encoder.')
        or k.startswith('rna_encoder.')
        or k.startswith('rna_reconstruction_head.')
        or k.startswith('gene_query_projection.')
        or k.startswith('gene_query_norm.')
        or k in {'rna_mask_token', 'gene_query_delta', 'R_delta', 'relation_scale', 'relation_gate', 'relation_ema', 'relation_initial'}
    }
    allowed_unexpected = {
        k for k in unexpected
        if k in {'gene_program_logits', 'H_logits', 'R_logits'}
        or k.startswith('R_program_update.')
        or k.startswith('H_update.')
    }
    if set(unexpected) - allowed_unexpected or set(missing) - allowed_missing:
        raise RuntimeError(f'Checkpoint mismatch: missing={missing}, unexpected={unexpected}')
    model.phenotype_mode = ckpt.get('phenotype_mode', 'full')
    device=select_device(args.device, min_free_gb=args.min_free_gb, reserve_gb=args.reserve_gb, verbose=True)
    model.to(device).eval()
    max_patches = args.max_patches if args.max_patches is not None else ckpt.get('max_patches')
    ds=G2PBRCAFeatureDataset(args.manifest_csv, args.label_dir, gene_programs_json, out_dir, split=args.split, metadata=metadata_tuple, max_patches=max_patches)
    loader=DataLoader(ds, batch_size=1, shuffle=False, collate_fn=one_item)
    evidence=[]
    with torch.no_grad():
        for batch in loader:
            batch=move_batch(batch, device); out=model(batch['features'])
            attention_sets = [
                ('gene', metadata['vocab']['gene_list'], out.get('gene_attention')),
                ('program', metadata['vocab']['program_names'], out['program_attention']),
                ('phenotype', metadata['vocab']['phenotype_names'], out['phenotype_attention']),
            ]
            for ptype, names, attn in attention_sets:
                if attn is None:
                    continue
                k=min(args.topk_patches, attn.shape[1]); vals, idx=torch.topk(attn, k=k, dim=1)
                for i,name in enumerate(names):
                    for score, patch_idx in zip(vals[i].cpu().tolist(), idx[i].cpu().tolist()):
                        evidence.append({'case_id': batch['case_id'], 'slide_id': batch['slide_id'], 'prototype_type': ptype, 'prototype_name': name, 'patch_index': patch_idx, 'attention_score': score})
    pd.DataFrame(evidence).to_csv(out_dir/'evidence_topk.csv', index=False)
    pd.DataFrame(rel['H_gene_to_program'], index=metadata['vocab']['gene_list'], columns=metadata['vocab']['program_names']).to_csv(out_dir/'H_gene_to_program.csv', encoding='utf-8-sig')
    if 'H_node_to_hyperedge' in rel:
        node_names = metadata['vocab']['program_names'] + metadata['vocab']['phenotype_names']
        hyperedge_names = [f'program_hyperedge:{name}' for name in metadata['vocab']['program_names']]
        pd.DataFrame(rel['H_node_to_hyperedge'], index=node_names, columns=hyperedge_names).to_csv(out_dir/'H_node_to_hyperedge.csv', encoding='utf-8-sig')
    pd.DataFrame(rel['R_program_to_phenotype'], index=metadata['vocab']['program_names'], columns=metadata['vocab']['phenotype_names']).to_csv(out_dir/'R_program_to_phenotype.csv', encoding='utf-8-sig')
    if 'HR_gene_to_phenotype' in rel:
        pd.DataFrame(rel['HR_gene_to_phenotype'], index=metadata['vocab']['gene_list'], columns=metadata['vocab']['phenotype_names']).to_csv(out_dir/'HR_gene_to_phenotype.csv', encoding='utf-8-sig')
    if 'G_gene_to_phenotype_prior' in rel:
        pd.DataFrame(rel['G_gene_to_phenotype_prior'], index=metadata['vocab']['gene_list'], columns=metadata['vocab']['phenotype_names']).to_csv(out_dir/'G_gene_to_phenotype_prior.csv', encoding='utf-8-sig')
    if 'gene_prototypes' in rel:
        pd.DataFrame(rel['gene_prototypes'], index=metadata['vocab']['gene_list']).to_csv(out_dir/'gene_prototypes.csv', encoding='utf-8-sig')
    if 'gene_identity_embeddings' in rel:
        pd.DataFrame(rel['gene_identity_embeddings'], index=metadata['vocab']['gene_list']).to_csv(out_dir/'gene_identity_embeddings.csv', encoding='utf-8-sig')
    if 'gene_query_delta' in rel:
        pd.DataFrame(rel['gene_query_delta'], index=metadata['vocab']['gene_list']).to_csv(out_dir/'gene_query_delta.csv', encoding='utf-8-sig')
    print(f'Evidence saved: {out_dir / "evidence_topk.csv"}')
    print(f'H/R matrices saved in: {out_dir}')

if __name__=='__main__': main()
