#!/usr/bin/env bash
# E-GraphSAGE baseline under three label versions (v0 any-endpoint, v1 infected-src,
# v2 corrected infected-src) x 2 variants x 3 seeds, strictly sequential, plus
# tuned RF (CPU) per label version for comparison. Resumable: skips finished runs.
# GPU safety: one job at a time; waits until GPU memory used < 9 GB; on CUDA OOM /
# CUDA error, reruns that job on CPU (never retries on GPU).
set -u
cd "$(dirname "$0")/.."
ROOT=data/inspection_logs/egraphsage
STATUS=$ROOT/queue_status.txt
mkdir -p $ROOT
declare -A GDIR=( [v0]=data/graphs [v1]=data/graphs_src [v2]=data/graphs_src2 )

wait_gpu() {
  while true; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
    if [ -n "$used" ] && [ "$used" -lt 9000 ]; then echo "$used"; return 0; fi
    echo "$(date +%H:%M) waiting: gpu mem used=${used:-?} MiB" >> $STATUS
    sleep 60
  done
}

# Tuned RF (CPU only, deterministic) for every label version, with preds.
for V in v0 v1 v2; do
  D=$ROOT/$V; mkdir -p $D
  if [ ! -f $D/rf_tuned.json ]; then
    HIGT_GRAPHS_DIR=${GDIR[$V]} uv run python scripts/baselines/run_rf.py --tune-threshold \
      --out $D/rf_tuned.json --save-preds $D/rf_tuned_preds.npz > $D/rf_tuned.log 2>&1 \
      && echo "ok rf $V $(date +%H:%M)" >> $STATUS || echo "FAIL rf $V $(date +%H:%M)" >> $STATUS
  fi
done

for V in v0 v1 v2; do
  D=$ROOT/$V
  for VAR in node edge2node; do
    for SEED in 42 1 2; do
      STEM=$D/${VAR}_seed${SEED}
      [ -f $STEM.json ] && continue
      used=$(wait_gpu)
      echo "start $V $VAR seed$SEED gpu_mem=${used}MiB $(date +%H:%M)" >> $STATUS
      HIGT_GRAPHS_DIR=${GDIR[$V]} uv run python scripts/baselines/run_egraphsage.py \
        --variant $VAR --seed $SEED --device cuda --out $STEM.json \
        --save-preds ${STEM}_preds.npz > $STEM.log 2>&1
      rc=$?
      if [ $rc -ne 0 ] && grep -qiE "out of memory|CUDA error|cudaError|AcceleratorError" $STEM.log; then
        echo "CUDA failure $V $VAR seed$SEED -> CPU fallback $(date +%H:%M)" >> $STATUS
        mv $STEM.log $STEM.cuda_fail.log
        HIGT_GRAPHS_DIR=${GDIR[$V]} uv run python scripts/baselines/run_egraphsage.py \
          --variant $VAR --seed $SEED --device cpu --out $STEM.json \
          --save-preds ${STEM}_preds.npz > $STEM.log 2>&1
        rc=$?
      fi
      if [ $rc -eq 0 ]; then echo "ok $V $VAR seed$SEED $(date +%H:%M)" >> $STATUS
      else echo "FAIL($rc) $V $VAR seed$SEED $(date +%H:%M)" >> $STATUS; fi
    done
  done
done
echo "DONE $(date +%H:%M)" >> $STATUS
