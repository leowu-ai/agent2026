#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/wl/agent_2026/g2p_toolbank_brca
PYTHON=/home/wl/anaconda3/envs/mil/bin/python
AGENT_DIR="$ROOT/multiscale_vqa_agent"
CONFIG="${1:-$AGENT_DIR/config.json}"
OUTPUT="$ROOT/outputs/multiscale_vqa_agent/answers.jsonl"
LOG="$ROOT/outputs/multiscale_vqa_agent/run.log"

mkdir -p "$(dirname "$OUTPUT")"
nohup "$PYTHON" "$AGENT_DIR/run_vqa.py" \
  --config "$CONFIG" \
  --output "$OUTPUT" \
  > "$LOG" 2>&1 &

echo "PID=$!"
echo "log=$LOG"
echo "output=$OUTPUT"
