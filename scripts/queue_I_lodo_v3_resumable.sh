#!/usr/bin/env bash
# Second restart of the MedBIoT LODO T-GINE (laptop blue-screened at 23:55 on 2026-10-12,
# epoch 63). Training now writes its full state every epoch (HIGT_TRAIN_STATE) and this
# queue is safe to re-launch after a crash: it resumes training, then evaluates and reports.
cd "$(dirname "$0")/.."
export HIGT_GRAPHS_DIR=data/graphs_v3 HIGT_DIRECTED=1
L=data/inspection_logs/v3_lodo; E=experiments/v3_lodo
S=$L/queue_F_status.txt
mkdir -p $E/medbiot/phase5
echo "queue I (resumable) start $(date)" >> $S
run() { name=$1; shift; "$@" >> $L/$name.log 2>&1 && echo "ok $name $(date +%H:%M)" >> $S || echo "FAIL $name $(date +%H:%M)" >> $S; }
if [ ! -f $E/medbiot/phase5/temporal_gine_lodo_medbiot_skip.pt ]; then
  [ -f $L/tgine_medbiot.log ] && [ ! -f $E/medbiot/phase5/train_state.pt ] && mv -f $L/tgine_medbiot.log $L/tgine_medbiot_killed_2.log
  HIGT_TRAIN_STATE=$E/medbiot/phase5/train_state.pt run tgine_medbiot uv run python -m scripts.train_phase5_droplast \
    --raw-feature-skip --no-amp --seed 42 --split-config configs/split_lodo_medbiot.yaml \
    --ckpt-dir $E/medbiot/phase5 --results-dir $L --results-suffix _lodo_medbiot_skip
fi
run tgine_eval uv run python -m scripts.lodo_v3 --part tgine
run report uv run python -m scripts.lodo_v3 --part report
echo "DONE $(date)" >> $S
