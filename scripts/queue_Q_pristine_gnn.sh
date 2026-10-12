#!/usr/bin/env bash
# Review item C1, graph models on the never-used captures (configs/split_pristine.yaml).
# 1) flow sequences (CPU, after chain A of queue_R so at most two heavy CPU jobs run)
# 2) waits until the LODO queue is DONE and no process uses the GPU, then encoder
#    embeddings + frozen-model inference (inference only). Idempotent: re-launch after a crash.
cd "$(dirname "$0")/.."
export HIGT_GRAPHS_DIR=data/graphs_v3 HIGT_DIRECTED=1
L=data/inspection_logs/revision; S=$L/queue_Q_status.txt
run() { name=$1; shift; "$@" >> $L/$name.log 2>&1 && echo "ok $name $(date +%H:%M)" >> $S || echo "FAIL $name $(date +%H:%M)" >> $S; }
echo "queue Q start $(date)" >> $S
until grep -q "^DONE A" $L/queue_R_status.txt 2>/dev/null; do sleep 60; done  # only python processes count as GPU jobs below
for sc in iot23-34-1 iot23-36-1 iot23-39-1 iot23-42-1 iot23-43-1 iot23-44-1 iot23-49-1 iot23-52-1 \
          medbiot-bashlite_leg medbiot-mirai_leg medbiot-torii_leg; do
  run flowseq_$sc uv run python scripts/build_flow_sequences.py --scenario $sc --skip-existing
done
until grep -q "^DONE" <(tail -n 3 data/inspection_logs/v3_lodo/queue_F_status.txt); do sleep 60; done
while nvidia-smi --query-compute-apps=process_name --format=csv,noheader 2>/dev/null | grep -qi python; do sleep 60; done
run embeddings uv run python scripts/cache_embeddings.py --checkpoint experiments/v3/phase5/temporal_gine_skip.pt \
  --out-dir data/flow_embeddings_v3 --split-config configs/split_pristine.yaml
while nvidia-smi --query-compute-apps=process_name --format=csv,noheader 2>/dev/null | grep -qi python; do sleep 60; done
run gnn_pristine uv run python -m scripts.revision_gnn_pristine
echo "DONE $(date)" >> $S
