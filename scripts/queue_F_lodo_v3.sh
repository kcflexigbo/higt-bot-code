#!/usr/bin/env bash
# v3 LODO: leave-one-dataset-out on the corrected data (data/graphs_v3: v2 labels +
# node features from all window flows). Directed T-GINE (seed 42, FP32 training,
# --no-amp) per held-out dataset, two GPU jobs at a time, then the tree models and
# the summary (scripts/lodo_v3.py). Old-label LODO artefacts are not touched.
cd "$(dirname "$0")/.."
export HIGT_GRAPHS_DIR=data/graphs_v3 HIGT_DIRECTED=1
L=data/inspection_logs/v3_lodo; E=experiments/v3_lodo
mkdir -p $L $E
S=$L/queue_F_status.txt
echo "start $(date)" > $S
run() { name=$1; shift; "$@" > $L/$name.log 2>&1 && echo "ok $name $(date +%H:%M)" >> $S || echo "FAIL $name $(date +%H:%M)" >> $S; }
tgine() {  # $1 = held-out dataset
  run tgine_$1 uv run python scripts/train_phase5.py --raw-feature-skip --no-amp --seed 42 \
    --split-config configs/split_lodo_$1.yaml --ckpt-dir $E/$1/phase5 --results-dir $L \
    --results-suffix _lodo_$1_skip
}

# GPU: two T-GINE trainings in parallel; CPU: the tree models alongside.
tgine ctu13 &
tgine iot23 &
run trees uv run python -m scripts.lodo_v3 --part trees
wait
tgine medbiot
run tgine_eval uv run python -m scripts.lodo_v3 --part tgine
run report uv run python -m scripts.lodo_v3 --part report
echo "DONE $(date)" >> $S
