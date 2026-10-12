#!/usr/bin/env bash
# Infected-source relabelling experiment (review item C1): retrain the main
# ladder on data/graphs_src (a host is a bot only if it originates a bot flow).
# Original results are untouched; everything goes to */relabel/.
cd "$(dirname "$0")/.."
export HIGT_GRAPHS_DIR=data/graphs_src
L=data/inspection_logs/relabel; E=experiments/relabel; P=$L/predictions
EMB=data/flow_embeddings_relabel
mkdir -p $L $E $P
S=$L/queue_C_status.txt
echo "start $(date)" > $S
run() { name=$1; shift; "$@" > $L/$name.log 2>&1 && echo "ok $name $(date +%H:%M)" >> $S || echo "FAIL $name $(date +%H:%M)" >> $S; }

while [ ! -f data/inspection_logs/relabel_infected_src.json ]; do sleep 30; done
echo "relabel done $(date +%H:%M)" >> $S

# Group 1: flat baselines and the encoder, in parallel.
run rf uv run python scripts/baselines/run_rf.py --out $L/baseline_rf.json --save-preds $P/rf.npz &
for s in 42 1 2; do
  run gine_seed$s uv run python scripts/baselines/run_gnn.py --model gine --hidden 64 --layers 2 \
    --focal-gamma 2.0 --drop-edge-p 0.2 --seed $s --out $L/baseline_gine_seed$s.json \
    --save-preds $P/gine_seed$s.npz &
done
run phase5_skip uv run python scripts/train_phase5.py --raw-feature-skip --results-suffix _skip \
  --ckpt-dir $E/phase5 --results-dir $L
wait

run cache_embeddings uv run python scripts/cache_embeddings.py \
  --checkpoint $E/phase5/temporal_gine_skip.pt --out-dir $EMB

# Group 2: hierarchical models on the relabelled embeddings.
run phase6_sparse uv run python scripts/train_phase6_sparse.py --seed 42 --results-suffix _baseline \
  --emb-dir $EMB --ckpt-dir $E/phase6 --results-dir $L &
run phase7_edge uv run python scripts/train_phase7.py --gt-variant edge --seed 42 --results-suffix _edge \
  --emb-dir $EMB --ckpt-dir $E/phase7 --results-dir $L &
run phase7_edge_seed1 uv run python scripts/train_phase7.py --gt-variant edge --seed 1 --results-suffix _edge_seed1 \
  --emb-dir $EMB --ckpt-dir $E/phase7 --results-dir $L &
wait
run phase7_edge_seed2 uv run python scripts/train_phase7.py --gt-variant edge --seed 2 --results-suffix _edge_seed2 \
  --emb-dir $EMB --ckpt-dir $E/phase7 --results-dir $L

run reeval uv run python scripts/reeval_mixed_pool_fp32.py --exp-root $E --emb-dir $EMB \
  --results-root $L --save-preds-dir $P
echo "DONE $(date)" >> $S
