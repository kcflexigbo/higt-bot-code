#!/usr/bin/env bash
# Resume of queue_G after the machine restarted at 15:09 on 2026-10-12, killing the
# MedBIoT T-GINE training (no checkpoint is written until training ends). Trees and the
# CTU-13 / IoT-23 T-GINE runs had finished; this retrains MedBIoT, then evaluates and reports.
cd "$(dirname "$0")/.."
export HIGT_GRAPHS_DIR=data/graphs_v3 HIGT_DIRECTED=1
L=data/inspection_logs/v3_lodo; E=experiments/v3_lodo
S=$L/queue_F_status.txt
echo "queue H (resume after restart) start $(date)" >> $S
run() { name=$1; shift; "$@" > $L/$name.log 2>&1 && echo "ok $name $(date +%H:%M)" >> $S || echo "FAIL $name $(date +%H:%M)" >> $S; }
mv -f $L/tgine_medbiot.log $L/tgine_medbiot_killed_by_restart.log 2>/dev/null
run tgine_medbiot uv run python -m scripts.train_phase5_droplast --raw-feature-skip --no-amp --seed 42 \
    --split-config configs/split_lodo_medbiot.yaml --ckpt-dir $E/medbiot/phase5 --results-dir $L \
    --results-suffix _lodo_medbiot_skip
run tgine_eval uv run python -m scripts.lodo_v3 --part tgine
run report uv run python -m scripts.lodo_v3 --part report
echo "DONE $(date)" >> $S
