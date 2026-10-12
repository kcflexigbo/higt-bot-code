#!/usr/bin/env bash
# Host-side GPU poller (Git Bash). Writes ../data/inspection_logs/gpu_status.txt every 15 s.
# Line 1: ISO-8601 timestamp; following lines: util %, mem used, mem total, temp C (one per GPU).
# Usage: bash dashboard/gpu_poller.sh &      (stop with: kill %1, or close the terminal)
DIR="$(cd "$(dirname "$0")/.." && pwd)/data/inspection_logs"
OUT="$DIR/gpu_status.txt"
INTERVAL="${INTERVAL:-15}"
while true; do
  {
    date -Iseconds
    nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader 2>&1
  } > "$OUT.tmp" && mv -f "$OUT.tmp" "$OUT"
  sleep "$INTERVAL"
done
