#!/usr/bin/env bash
set -euo pipefail

STATE_DIR=/home/wl/agent_2026/g2p_toolbank_brca/outputs/multiscale_vqa_agent/services

for spec in qwen:8000 pathor1:8001; do
  name="${spec%%:*}"
  port="${spec##*:}"
  pid_file="$STATE_DIR/$name.pid"
  pid="missing"
  process="stopped"
  if [[ -f "$pid_file" ]]; then
    pid="$(cat "$pid_file")"
    if kill -0 "$pid" 2>/dev/null; then
      process="running"
    fi
  fi
  health="not-ready"
  if curl -fsS --max-time 2 "http://127.0.0.1:$port/health" >/dev/null 2>&1; then
    health="ready"
  fi
  echo "$name: pid=$pid process=$process health=$health port=$port"
done
