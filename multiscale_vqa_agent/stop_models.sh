#!/usr/bin/env bash
set -euo pipefail

STATE_DIR=/home/wl/agent_2026/g2p_toolbank_brca/outputs/multiscale_vqa_agent/services

for name in qwen pathor1; do
  pid_file="$STATE_DIR/$name.pid"
  if [[ ! -f "$pid_file" ]]; then
    echo "$name: no PID file"
    continue
  fi
  pid="$(cat "$pid_file")"
  if kill -0 "$pid" 2>/dev/null; then
    kill "$pid"
    echo "$name: stopped PID=$pid"
  else
    echo "$name: PID=$pid is not running"
  fi
  rm -f "$pid_file"
done
