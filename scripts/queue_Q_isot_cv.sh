#!/usr/bin/env bash
# Within-ISOT window-grouped 5-fold CV (scripts/p2p_isot_cv.py). Re-launch after a crash;
# finished parts (json present) are skipped.
cd "$(dirname "$0")/.."
export HIGT_GRAPHS_DIR=data/graphs_v3
L=data/inspection_logs/p2p; S=$L/queue_Q_status.txt
echo "start $(date)" >> $S
run() { name=$1; shift; "$@" > $L/$name.log 2>&1 && echo "ok $name $(date +%H:%M)" >> $S || echo "FAIL $name $(date +%H:%M)" >> $S; }
[ -f $L/isot_cv_trees.json ] || run isot_cv_trees uv run python -m scripts.p2p_isot_cv --part trees
[ -f $L/isot_cv_gine.json ] || HIGT_DIRECTED=0 run isot_cv_gine uv run python -m scripts.p2p_isot_cv --part gine --tag gine
[ -f $L/isot_cv_dirgine.json ] || HIGT_DIRECTED=1 run isot_cv_dirgine uv run python -m scripts.p2p_isot_cv --part gine --tag dirgine
echo "DONE $(date)" >> $S
