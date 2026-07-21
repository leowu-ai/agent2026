#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/wl/agent_2026/g2p_toolbank_brca
AGENT_DIR="$ROOT/multiscale_vqa_agent"
PYTHON=/home/wl/anaconda3/envs/mil/bin/python
OUT_DIR="$ROOT/outputs/multiscale_vqa_agent/multiple_choice"
OUTPUT="$OUT_DIR/mc_answers.jsonl"
METRICS="$OUT_DIR/mc_live_metrics.json"
LOG="$OUT_DIR/run.log"
PID_FILE="$OUT_DIR/run.pid"

mkdir -p "$OUT_DIR"
if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo "already running: PID=$(cat "$PID_FILE")"
  exit 0
fi

CUDA_VISIBLE_DEVICES="${G2P_GPU:-0}" nohup "$PYTHON" "$AGENT_DIR/run_mc_vqa.py" \
  --config "$AGENT_DIR/config.servers.json" \
  --output "$OUTPUT" \
  --metrics "$METRICS" \
  > "$LOG" 2>&1 < /dev/null &

echo "$!" > "$PID_FILE"
echo "PID=$!"
echo "log=$LOG"
echo "answers=$OUTPUT"
echo "live_metrics=$METRICS"
