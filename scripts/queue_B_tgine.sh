#!/usr/bin/env bash
# T-GINE + skip seed 2, after the running seed-1 job writes its results.
cd "$(dirname "$0")/.."
L=data/inspection_logs; S=$L/queue_B_status.txt
echo "waiting for seed 1 $(date)" > $S
while [ ! -f $L/phase5_temporal_gine_skip_seed1.json ]; do sleep 60; done
echo "seed1 done $(date +%H:%M)" >> $S
uv run python scripts/train_phase5.py --raw-feature-skip --seed 2 --results-suffix _skip_seed2 > $L/phase5_skip_seed2.log 2>&1 \
  && echo "ok seed2 $(date +%H:%M)" >> $S || echo "FAIL seed2 $(date +%H:%M)" >> $S
echo "DONE $(date)" >> $S
