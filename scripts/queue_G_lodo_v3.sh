#!/usr/bin/env bash
# Continuation of queue_F_lodo_v3.sh after LODO-CTU13 crashed (batch of one single-host
# window hits BatchNorm; see scripts/train_phase5_droplast.py). CTU-13 and MedBIoT
# (also an odd number of training windows) are trained with the drop_last wrapper;
# IoT-23 (still running from queue F, original script) is left as is. $1 = its PID.
cd "$(dirname "$0")/.."
export HIGT_GRAPHS_DIR=data/graphs_v3 HIGT_DIRECTED=1
L=data/inspection_logs/v3_lodo; E=experiments/v3_lodo
S=$L/queue_F_status.txt
echo "queue G start $(date)" >> $S
run() { name=$1; shift; "$@" > $L/$name.log 2>&1 && echo "ok $name $(date +%H:%M)" >> $S || echo "FAIL $name $(date +%H:%M)" >> $S; }
tgine() {
  run tgine_$1 uv run python -m scripts.train_phase5_droplast --raw-feature-skip --no-amp --seed 42 \
    --split-config configs/split_lodo_$1.yaml --ckpt-dir $E/$1/phase5 --results-dir $L \
    --results-suffix _lodo_$1_skip
}
tgine ctu13 &
while tasklist //FI "PID eq $1" | grep -q python; do sleep 30; done
echo "iot23 training process exited $(date +%H:%M); ckpt: $(ls $E/iot23/phase5/ 2>&1)" >> $S
tgine medbiot
wait
run tgine_eval uv run python -m scripts.lodo_v3 --part tgine
run report uv run python -m scripts.lodo_v3 --part report
echo "DONE $(date)" >> $S
