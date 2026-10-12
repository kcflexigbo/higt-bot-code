#!/usr/bin/env bash
# Detached GPU queue (2026-10-11): remaining ablation seeds, then the Option B looped-GINE screen.
cd "$(dirname "$0")/.."
L=data/inspection_logs
S=$L/queue_overnight_status.txt
echo "start $(date)" > $S
run() { name=$1; shift; "$@" > $L/$name.log 2>&1 && echo "ok $name $(date +%H:%M)" >> $S || echo "FAIL $name $(date +%H:%M)" >> $S; }
run phase6_sparse_baseline_seed2 uv run python scripts/train_phase6_sparse.py --seed 2 --results-suffix _baseline_seed2
run phase5_skip_seed1 uv run python scripts/train_phase5.py --raw-feature-skip --seed 1 --results-suffix _skip_seed1
run phase5_skip_seed2 uv run python scripts/train_phase5.py --raw-feature-skip --seed 2 --results-suffix _skip_seed2
for cfg in "loop2:--embed-loops 2" "loop4:--embed-loops 4" "loop8:--embed-loops 8" "untied4:--embed-layers 4"; do
  name=${cfg%%:*}; flags=${cfg#*:}
  run phase7_train_edge_$name uv run python scripts/train_phase7.py --gt-variant edge $flags --seed 42 --emb-dir data/flow_embeddings --results-suffix _edge_$name
done
echo "DONE $(date)" >> $S
