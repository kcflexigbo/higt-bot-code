#!/usr/bin/env bash
# Resume queue D after its last group was killed at 20:56 (system RAM ran out while
# several agents' jobs ran concurrently). At most two training jobs at once.
cd "$(dirname "$0")/.."
export HIGT_GRAPHS_DIR=data/graphs_src2
L=data/inspection_logs/v2; E=experiments/v2; P=$L/predictions
EMB=data/flow_embeddings_v2
S=$L/queue_D_status.txt
echo "resume $(date)" >> $S
run() { name=$1; shift; "$@" > $L/$name.log 2>&1 && echo "ok $name $(date +%H:%M)" >> $S || echo "FAIL $name $(date +%H:%M)" >> $S; }
p7() { HIGT_DIRECTED=$3 run $1 uv run python scripts/train_phase7.py --gt-variant edge --seed $2 \
  --results-suffix _$1 --emb-dir $EMB --ckpt-dir $E/phase7 --results-dir $L; }

p7 edge_seed2 2 1 &
p7 edge_undirected 42 0 &
wait
HIGT_DIRECTED=1 run phase5_dir_seed1 uv run python scripts/train_phase5.py --raw-feature-skip --seed 1 \
  --results-suffix _skip_seed1 --ckpt-dir $E/phase5 --results-dir $L
HIGT_DIRECTED=1 run reeval uv run python scripts/reeval_mixed_pool_fp32.py --exp-root $E --emb-dir $EMB \
  --results-root $L --save-preds-dir $P --also-val
echo "DONE $(date)" >> $S
