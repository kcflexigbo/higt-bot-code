#!/usr/bin/env bash
# Option B screen on the shared RTX 5080, two runs at a time.
cd "$(dirname "$0")/.."
L=data/inspection_logs; S=$L/queue_A_status.txt
echo "start $(date)" > $S
run() { name=$1; shift; "$@" > $L/$name.log 2>&1 && echo "ok $name $(date +%H:%M)" >> $S || echo "FAIL $name $(date +%H:%M)" >> $S; }
p7() { run phase7_train_edge_$1 uv run python scripts/train_phase7.py --gt-variant edge $2 --seed 42 --emb-dir data/flow_embeddings --results-suffix _edge_$1; }
p7 loop2 "--embed-loops 2" & p7 loop4 "--embed-loops 4" & wait
p7 loop8 "--embed-loops 8" & p7 untied4 "--embed-layers 4" & wait
echo "DONE $(date)" >> $S
