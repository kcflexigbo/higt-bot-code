#!/usr/bin/env bash
# Review item M6/M7: flow-record loss robustness (scripts/revision_flow_sampling.py).
# Starts after the LOSO chain (queue_R B) and the flow-sequence builds of queue_Q (memory). Idempotent.
cd "$(dirname "$0")/.."
export HIGT_GRAPHS_DIR=data/graphs_v3
L=data/inspection_logs/revision
until grep -q "DONE B" <(tail -n 2 $L/queue_R_status.txt) && [ "$(grep -c "flowseq_" $L/queue_Q_status.txt 2>/dev/null)" -ge 11 ]; do sleep 60; done
uv run python -m scripts.revision_flow_sampling >> $L/flow_sampling.log 2>&1 && echo "ok flow_sampling $(date +%H:%M)" >> $L/queue_R_status.txt || echo "FAIL flow_sampling $(date +%H:%M)" >> $L/queue_R_status.txt
