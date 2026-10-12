#!/usr/bin/env bash
# v3: retrain the graph models on data/graphs_v3 (v2 labels + node features from all
# window flows, scripts/refeature_graphs.py) so the GNN and tree rows of the final
# tables use the same features. At most two GPU training jobs at once (RAM + GPU budget).
cd "$(dirname "$0")/.."
export HIGT_GRAPHS_DIR=data/graphs_v3
L=data/inspection_logs/v3_gnn; E=experiments/v3; P=$L/predictions
EMB=data/flow_embeddings_v3
mkdir -p $L $E $P
S=$L/queue_E_status.txt
echo "start $(date)" > $S
run() { name=$1; shift; "$@" > $L/$name.log 2>&1 && echo "ok $name $(date +%H:%M)" >> $S || echo "FAIL $name $(date +%H:%M)" >> $S; }
gine() {  # $1 = name, $2 = seed, $3 = HIGT_DIRECTED value
  HIGT_DIRECTED=$3 run $1 uv run python scripts/baselines/run_gnn.py --model gine --hidden 64 --layers 2 \
    --focal-gamma 2.0 --drop-edge-p 0.2 --seed $2 --out $L/baseline_$1.json --save-preds $P/$1.npz
}
p7() { HIGT_DIRECTED=$3 run $1 uv run python scripts/train_phase7.py --gt-variant edge --seed $2 \
  --results-suffix _$1 --emb-dir $EMB --ckpt-dir $E/phase7 --results-dir $L; }

# Group 1: tuned RF (CPU), flat GINE chain (GPU) and the directed temporal encoder (GPU).
run rf_tuned uv run python scripts/baselines/run_rf.py --tune-threshold --out $L/baseline_rf_tuned.json \
  --save-preds $P/rf_tuned.npz &
( gine dirgine_seed42 42 1; gine dirgine_seed1 1 1; gine dirgine_seed2 2 1; gine gine_seed42 42 0 ) &
HIGT_DIRECTED=1 run phase5_dir uv run python scripts/train_phase5.py --raw-feature-skip --results-suffix _skip \
  --ckpt-dir $E/phase5 --results-dir $L
wait

HIGT_DIRECTED=1 run cache_embeddings uv run python scripts/cache_embeddings.py \
  --checkpoint $E/phase5/temporal_gine_skip.pt --out-dir $EMB

# Group 2: hierarchical models, directed 3 seeds + undirected ablation, two at a time.
p7 edge 42 1 &
p7 edge_seed1 1 1
wait
p7 edge_seed2 2 1 &
p7 edge_undirected 42 0
wait
HIGT_DIRECTED=1 run phase6_dir uv run python scripts/train_phase6_sparse.py --seed 42 --results-suffix _baseline \
  --emb-dir $EMB --ckpt-dir $E/phase6 --results-dir $L &
HIGT_DIRECTED=1 run phase5_dir_seed1 uv run python scripts/train_phase5.py --raw-feature-skip --seed 1 \
  --results-suffix _skip_seed1 --ckpt-dir $E/phase5 --results-dir $L
wait

HIGT_DIRECTED=1 run reeval uv run python scripts/reeval_mixed_pool_fp32.py --exp-root $E --emb-dir $EMB \
  --results-root $L --save-preds-dir $P --also-val
# Tree head with the v3 encoder embedding (emb rows); other rows should match data/inspection_logs/v3.
run trees_emb3 uv run python scripts/higt_v2_trees.py --out data/inspection_logs/v3_emb3 --emb-dir $EMB
echo "DONE $(date)" >> $S
