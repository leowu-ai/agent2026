#!/usr/bin/env bash
set -euo pipefail
trap '' HUP


cd /home/wl/agent_2026/g2p_toolbank_brca

mkdir -p outputs

GPUS=${GPUS:-$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -t, -k2,2n | head -4 | cut -d, -f1 | paste -sd, -)}
NPROC=${NPROC:-4}
PYTHON=${PYTHON:-/home/wl/anaconda3/envs/mil/bin/python}

COMMON_ARGS=(
  --label_dir /home/wl/agent_2026/dataset
  --program_scores_csv /home/wl/agent_2026/dataset/derived/ssgsea_278_program_scores_raw.csv
  --split_csv /home/wl/agent_2026/dataset/splits/multitask_stratified_split_seed42/case_splits_multitask_stratified_seed42.csv
  --epochs 100
  --hidden_dim 512
  --dropout 0.25
  --lr 1e-4
  --weight_decay 1e-5
  --batch_size 1
  --gene_batch_size 1
  --phenotype_batch_size 4
  --gene_grad_accum_steps 1
  --phenotype_grad_accum_steps 1
  --training_schedule iterative_relation
  --iterative_gene_pretrain_epochs 10
  --iterative_gene_program_epochs 24
  --iterative_relation_rounds 3
  --iterative_relation_epochs 6
  --iterative_adapt_epochs 10
  --iterative_relation_topk 6
  --relation_init_mode uniform
  --relation_init_value 0.5
  --relation_selection_mode free_topk
  --phenotype_schedule staged
  --lambda_gene_stage1 1.0
  --lambda_gene_stage2 0.5
  --lambda_gene 0.0
  --lambda_gene_corr 0.0
  --lambda_gene_query_delta 1e-5
  --lambda_rna_align 0.005
  --lambda_rna_recon 0.02
  --lambda_program 0.4
  --lambda_align 0.0
  --lambda_prior 0.0
  --lambda_sparse 0.001
  --lambda_diversity 0.001
  --gene_retention_ratio 0.98
  --device cuda:0
  --log_every 20
)

log_msg() {
  local msg="$1"
  local file="$2"
  echo "$msg"
  echo "$msg" >> "$file"
}

run_scale() {
  local scale="$1"
  local out_dir="outputs/full_run_h512_freeR05_earlyfreeze_sharedgate_${scale}"
  local log_file="${out_dir}.log"

  : > "${log_file}"
  log_msg "[$(date '+%F %T')] Starting feature_size=${scale} on GPUs=${GPUS}" "${log_file}"
  CUDA_VISIBLE_DEVICES="${GPUS}" nohup "${PYTHON}" -m torch.distributed.run --nproc_per_node="${NPROC}" scripts/train_g2p_toolbank.py \
    --feature_size "${scale}" \
    --out_dir "${out_dir}" \
    "${COMMON_ARGS[@]}" \
    >> "${log_file}" 2>&1
  log_msg "[$(date '+%F %T')] Finished feature_size=${scale}" "${log_file}"
}

SCALES=${SCALES:-"2048 4096"}
for scale in ${SCALES}; do
  run_scale "${scale}"
done

echo "[$(date '+%F %T')] All runs finished."
