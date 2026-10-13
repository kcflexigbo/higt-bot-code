#!/usr/bin/env bash
# P2P botnet experiments (ISOT 2010 + CTU-13 scenario 12). Idempotent and crash-resumable:
# every step checks for its output and is skipped if done, so after a crash simply re-launch:
#   bash scripts/queue_P_p2p.sh
# GPU steps wait for "DONE" in the LODO queue status and run strictly one at a time.
cd "$(dirname "$0")/.."
export HIGT_GRAPHS_DIR=data/graphs_v3 HIGT_DIRECTED=1
L=data/inspection_logs/p2p; R=data/raw/p2p_isot; S=$L/queue_P_status.txt
mkdir -p $L $R
echo "start $(date)" >> $S
run() { name=$1; shift; "$@" > $L/$name.log 2>&1 && echo "ok $name $(date +%H:%M)" >> $S || { echo "FAIL $name $(date +%H:%M)" >> $S; return 1; }; }
TGZ=$R/ISOT_Botnet_DataSet_2010.tar.gz; PCAP=$R/ISOT_Botnet_DataSet_2010/ISOT_Botnet_DataSet_2010.pcap
URL="https://drive.usercontent.google.com/download?id=1X1zPBJFPHU1ToQbpyd1Is1tJJuz2BeRd&export=download&confirm=t"

# 1. download (resumes a partial file), verify gzip integrity
if [ ! -f $R/.download_ok ]; then
  # resume only within one attempt; a file left by a crash can end in unflushed garbage,
  # so if the finished archive fails the integrity test it is deleted and fetched again
  for i in 1 2 3; do
    for j in 1 2 3 4 5; do curl -sL -C - -o $TGZ "$URL" && break; done
    gzip -t $TGZ 2>/dev/null && break
    echo "corrupt archive (attempt $i), refetching $(date +%H:%M)" >> $S; rm -f $TGZ
  done
  gzip -t $TGZ && sha256sum $TGZ > $R/sha256.txt && touch $R/.download_ok && echo "ok download $(date +%H:%M)" >> $S \
    || { echo "FAIL download $(date +%H:%M)" >> $S; exit 1; }
fi
# 2. extract the pcap (11,394,249,522 bytes)
if [ "$(stat -c %s $PCAP 2>/dev/null)" != "11394249522" ]; then
  tar xzf $TGZ -C $R && echo "ok extract $(date +%H:%M)" >> $S || { echo "FAIL extract" >> $S; exit 1; }
fi
# 3. parse to flows (atomic write)
if [ ! -f data/processed/isot-2010.parquet ]; then
  run parse_isot uv run python -m src.data.parse_isot --pcap $PCAP --out data/processed/isot-2010.tmp.parquet \
    && mv data/processed/isot-2010.tmp.parquet data/processed/isot-2010.parquet || exit 1
fi
# 4-7. graphs -> corrected labels (v2) -> full-flow features (v3) -> flow sequences (all skip when complete)
[ -f data/graphs/isot-2010/.complete ] || run graphs_isot uv run python scripts/build_graphs.py --parquet data/processed/isot-2010.parquet --streaming || exit 1
[ -f data/graphs_src2/isot-2010/.complete ] || run relabel_isot uv run python scripts/relabel_infected_src.py --any-dst \
  --medbiot-devices-only --out-dir data/graphs_src2 --only isot-2010 --out-json $L/relabel_isot.json || exit 1
[ -f data/graphs_v3/isot-2010/.complete ] || run refeature_isot uv run python scripts/refeature_graphs.py --only isot-2010 \
  --out-json $L/refeature_isot.json || exit 1
run flowseqs uv run python scripts/build_flow_sequences.py --scenario ctu13-12 --scenario isot-2010 --skip-existing || exit 1
# 8. CPU analyses
run graph_stats uv run python -m scripts.p2p_graph_stats
[ -f $L/.trees_done ] || { run trees uv run python -m scripts.p2p_transfer_trees --targets ctu13-12 isot-2010 && touch $L/.trees_done; }

# ---- GPU part: wait for the LODO queue, then one job at a time ----
until grep -q "^DONE" data/inspection_logs/v3_lodo/queue_F_status.txt; do sleep 60; done
run cache_emb uv run python scripts/cache_embeddings.py --checkpoint experiments/v3/phase5/temporal_gine_skip.pt \
  --out-dir data/flow_embeddings_v3 --split-config configs/split_p2p_transfer.yaml
run gnn_ckpt uv run python -m scripts.p2p_gnn_eval --targets ctu13-12 isot-2010
gine() {  # $1 name, $2 seed, $3 directed, $4 split config  (skipped when predictions exist)
  [ -f $L/$1.npz ] && return 0
  HIGT_DIRECTED=$3 run $1 uv run python scripts/baselines/run_gnn.py --model gine --hidden 64 --layers 2 \
    --focal-gamma 2.0 --drop-edge-p 0.2 --seed $2 --split-config $4 --out $L/$1.json --save-preds $L/$1.npz
}
gine dirgine_transfer_seed42 42 1 configs/split_p2p_transfer.yaml
gine dirgine_isot_seed42 42 1 configs/split_p2p_isot.yaml
gine gine_isot_seed42 42 0 configs/split_p2p_isot.yaml
gine dirgine_isot_seed1 1 1 configs/split_p2p_isot.yaml
gine dirgine_isot_seed2 2 1 configs/split_p2p_isot.yaml
echo "DONE $(date)" >> $S
