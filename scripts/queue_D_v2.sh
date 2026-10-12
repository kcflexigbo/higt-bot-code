#!/usr/bin/env bash
# v2 experiment: corrected labels v2 (data/graphs_src2: originators of bot flows to any
# destination; MedBIoT restricted to testbed devices minus C&C) + direction-aware,
# ego-separated GINE layers (HIGT_DIRECTED=1, src/models/dir_gine.py).
# At most 3 GPU jobs at once (the 5080 dropped off the bus when probes ran out of memory).
cd "$(dirname "$0")/.."
export HIGT_GRAPHS_DIR=data/graphs_src2
L=data/inspection_logs/v2; E=experiments/v2; P=$L/predictions
EMB=data/flow_embeddings_v2
mkdir -p $L $E $P
S=$L/queue_D_status.txt
echo "start $(date)" > $S
run() { name=$1; shift; "$@" > $L/$name.log 2>&1 && echo "ok $name $(date +%H:%M)" >> $S || echo "FAIL $name $(date +%H:%M)" >> $S; }
gine() {  # $1 = name, $2 = seed, $3 = HIGT_DIRECTED value
  HIGT_DIRECTED=$3 run $1 uv run python scripts/baselines/run_gnn.py --model gine --hidden 64 --layers 2 \
    --focal-gamma 2.0 --drop-edge-p 0.2 --seed $2 --out $L/baseline_$1.json --save-preds $P/$1.npz
}

# Group 1 (CPU + light GPU): tree baselines, flat GINE forward vs directed, the directed encoder.
run rf_tuned uv run python scripts/baselines/run_rf.py --tune-threshold --out $L/baseline_rf_tuned.json \
  --save-preds $P/rf_tuned.npz &
for s in 42 1 2; do gine gine_seed$s $s 0; done
for s in 42 1 2; do gine dirgine_seed$s $s 1; done
HIGT_DIRECTED=1 run phase5_dir uv run python scripts/train_phase5.py --raw-feature-skip --results-suffix _skip \
  --ckpt-dir $E/phase5 --results-dir $L
wait

HIGT_DIRECTED=1 run cache_embeddings uv run python scripts/cache_embeddings.py \
  --checkpoint $E/phase5/temporal_gine_skip.pt --out-dir $EMB
HYBRID_EMB_DIR=$EMB HYBRID_OUT_DIR=$L run hybrid_trees uv run python scripts/hybrid_trees.py &

# Group 2: hierarchical models, directed (3 seeds) + undirected ablation, max 3 on GPU.
p7() { HIGT_DIRECTED=$3 run $1 uv run python scripts/train_phase7.py --gt-variant edge --seed $2 \
  --results-suffix _$1 --emb-dir $EMB --ckpt-dir $E/phase7 --results-dir $L; }
HIGT_DIRECTED=1 run phase6_dir uv run python scripts/train_phase6_sparse.py --seed 42 --results-suffix _baseline \
  --emb-dir $EMB --ckpt-dir $E/phase6 --results-dir $L &
p7 edge 42 1 &
p7 edge_seed1 1 1 &
wait
p7 edge_seed2 2 1 &
p7 edge_undirected 42 0 &
HIGT_DIRECTED=1 run phase5_dir_seed1 uv run python scripts/train_phase5.py --raw-feature-skip --seed 1 \
  --results-suffix _skip_seed1 --ckpt-dir $E/phase5 --results-dir $L
wait

HIGT_DIRECTED=1 run reeval uv run python scripts/reeval_mixed_pool_fp32.py --exp-root $E --emb-dir $EMB \
  --results-root $L --save-preds-dir $P --also-val
echo "DONE $(date)" >> $S
