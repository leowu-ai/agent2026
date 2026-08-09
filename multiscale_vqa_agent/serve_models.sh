#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/wl/agent_2026/g2p_toolbank_brca
AGENT_DIR="$ROOT/multiscale_vqa_agent"
VLLM=/home/wl/agent_2026/.venvs/qwen_pathor1_vllm/bin/vllm
MODEL_ROOT=/data_nas3/ycz/00_SHARE_WITH_COLLEAGUE_QWEN_PATHOR1_20260716
STATE_DIR="$ROOT/outputs/multiscale_vqa_agent/services"
GPU="${AGENT_GPU:-1}"

mkdir -p "$STATE_DIR"

start_service() {
  local name="$1"
  local port="$2"
  shift 2
  local pid_file="$STATE_DIR/$name.pid"
  local log_file="$STATE_DIR/$name.log"

  if [[ -f "$pid_file" ]] && kill -0 "$(cat "$pid_file")" 2>/dev/null; then
    echo "$name already running: PID=$(cat "$pid_file")"
    return
  fi
  if ss -ltn | grep -q ":$port "; then
    echo "port $port is already occupied; refusing to start $name" >&2
    return 1
  fi

  nohup env CUDA_VISIBLE_DEVICES="$GPU" VLLM_USE_FLASHINFER_SAMPLER=0 \
    "$VLLM" serve "$@" \
    --host 127.0.0.1 \
    --port "$port" \
    --enforce-eager \
    < /dev/null \
    > "$log_file" 2>&1 &
  echo "$!" > "$pid_file"
  echo "started $name: PID=$! log=$log_file"
}

start_service qwen 8000 \
  "$MODEL_ROOT/Qwen3.5-9B" \
  --served-model-name Qwen3.5 \
  --dtype bfloat16 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.35 \
  --language-model-only \
  --reasoning-parser qwen3

start_service pathor1 8001 \
  "$MODEL_ROOT/Patho-R1-7B" \
  --served-model-name Patho-R1-7B \
  --dtype bfloat16 \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.35 \
  --limit-mm-per-prompt '{"image": 12}'

echo "Use status_models.sh to watch startup. Initial model loading can take several minutes."
