#!/usr/bin/env bash
# Review-revision tree experiments (scripts/revision_trees.py). Idempotent: re-launch after a crash.
# $1 = chain A (seeds, importance) or B (botminer, loso)
cd "$(dirname "$0")/.."
export HIGT_GRAPHS_DIR=data/graphs_v3
L=data/inspection_logs/revision; S=$L/queue_R_status.txt
mkdir -p $L
run() { name=$1; shift; "$@" >> $L/$name.log 2>&1 && echo "ok $name $(date +%H:%M)" >> $S || echo "FAIL $name $(date +%H:%M)" >> $S; }
if [ "$1" = A ]; then
  run seeds uv run python -m scripts.revision_trees --part seeds
  run importance uv run python -m scripts.revision_trees --part importance
  echo "DONE A $(date)" >> $S
else
  run botminer uv run python -m scripts.revision_trees --part botminer
  run loso uv run python -m scripts.revision_trees --part loso
  echo "DONE B $(date)" >> $S
fi
