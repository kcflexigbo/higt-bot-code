# HiGT-Bot: Implementation Plan

**Hierarchical Graph-Transformer Learning for P2P Botnet Detection in IoT Networks**

Master's thesis implementation plan. Read this end-to-end before writing any code. The order of phases is not negotiable — every phase has a validation gate that must pass before the next phase begins. Skipping ahead is the single most common cause of wasted weeks in this kind of research.

---

## Table of contents

1. [Context and scope](#1-context-and-scope)
2. [Reference architecture](#2-reference-architecture)
3. [Phase 0 — Project skeleton and environment](#phase-0--project-skeleton-and-environment)
4. [Phase 1 — Acquire and inspect the data](#phase-1--acquire-and-inspect-the-data)
5. [Phase 2 — Flow parser](#phase-2--flow-parser)
6. [Phase 3 — Graph construction](#phase-3--graph-construction)
7. [Phase 4 — Baselines](#phase-4--baselines)
8. [Phase 5 — Temporal Transformer for node features](#phase-5--temporal-transformer-for-node-features)
9. [Phase 6 — Hierarchical pooling layer](#phase-6--hierarchical-pooling-layer)
10. [Phase 7 — Graph Transformer on the coarsened graph](#phase-7--graph-transformer-on-the-coarsened-graph)
11. [Phase 8 — Ablations, robustness, paper-ready story](#phase-8--ablations-robustness-and-the-paper-ready-story)
12. [Top 12 pain points and warnings](#12-top-12-pain-points-and-warnings)
13. [Reproducibility checklist](#13-reproducibility-checklist)
14. [Suggested cadence and milestones](#14-suggested-cadence-and-milestones)
15. [Useful references](#15-useful-references)

---

## 1. Context and scope

**Problem.** Detect P2P botnet traffic in IoT networks. Traditional flow-level features fail because individual P2P bot conversations look benign; the malice is in the *structure* of the communication graph at multiple scales.

**Approach.** Build a hybrid model that uses (a) a temporal Transformer to summarize each device's flow history into a rich node embedding, (b) GNN layers to capture local neighborhood structure, (c) hierarchical pooling (DiffPool or MinCutPool) to coarsen the graph into community-level super-nodes, and (d) a Graph Transformer on the coarsened graph for global, long-range reasoning across communities.

**Novelty claim.** Hierarchical pooling is what makes a Graph Transformer *computationally feasible* on host-level network graphs. O(K²) global attention is intractable on N hosts (thousands) but trivial on K << N community-level super-nodes. This reframes the contribution from "novel combination" (weak) to "principled architectural necessity" (strong): the hierarchy is the mechanism that lets long-range reasoning happen at all on realistic IoT host graphs. The 2024–2026 landscape confirms the gap: GConvTrans (PLOS One 2026) and IoE-GraphFormer (DCOSS-IoT 2025) combine GNN + Transformer but are *flat*; MalDMTP (2024) uses multi-tier pooling but for malware call graphs, not network traffic; the hybrid GNN-LSTM (Computers 2026) uses LSTM, not Transformer; PeerG, MalHAPGNN, BotLGT, and GraphSAINT+GIN each miss at least two of the three ingredients.

**Datasets.** Three real-malware datasets covering the full botnet lifecycle:

- **CTU-13** — general P2P botnet, attack execution phase (scenarios 9/10/11 labeled P2P).
- **IoT-23** — real IoT botnet, attack execution phase (Mirai, Mozi, Hakai captures).
- **MedBIoT** — real malware (Mirai, BashLite, Torii) on 83 real+emulated IoT devices, **propagation and C&C phases** (the lifecycle window the other two miss).

MedBIoT is sometimes mischaracterized as "synthetic." It is not: it deploys actual malware and captures real traffic; only the device mix is partly emulated. The 2026 cross-domain benchmark (arXiv 2602.23874) shows MedBIoT generalizes credibly to CICIoT23 under Zeek features. N-BaIoT is *not* used for graph construction because it ships pre-aggregated features.

**Tooling.** Python 3.11, PyTorch 2.4+, PyTorch Geometric 2.6+, scapy/nfstream/CICFlowMeter for parsing, NetworkX for graph utilities, Weights & Biases for experiment tracking, Hydra for configs.

**Target venues.** IEEE Internet of Things Journal, Computers & Security, Future Generation Computer Systems, IEEE TrustCom, IEEE ICC.

---

## 2. Reference architecture

```
Raw network traffic (pcap / argus / Zeek conn.log)
        │
        ▼
[ Per-device flow sequences in time window W ]
        │  (Temporal Transformer — Phase 5)
        ▼
[ Rich node features ]  +  [ Edge features ]
        │
        ▼
[ Constructed graph: nodes = hosts, edges = aggregated communications ]
        │
        ▼
[ GIN / GAT block — 2 layers ]            ← local structure
        │
        ▼
[ DiffPool / MinCutPool ]                 ← coarsen to community level
        │
        ▼
[ GIN block — 1 layer on coarsened graph ]
        │
        ▼
[ Graph Transformer — 2 layers ]          ← long-range reasoning
        │
        ▼
[ Global mean+max pool OR un-pool to original nodes ]
        │
        ▼
[ MLP classifier ]                        ← bot vs benign
```

Node-level classification path: after the Graph Transformer, multiply by the transpose of the DiffPool assignment matrices to project predictions back to original nodes (U-Net-style skip connections from earlier GIN block features help).

Graph-level classification path: global pooling → MLP → single label per window.

We will build both heads. Node-level is the harder, more useful task; graph-level is an easier ablation.

---

## Phase 0 — Project skeleton and environment

**Duration:** Day 1.
**Goal:** A reproducible, well-organized repo before any algorithm work.

### What to do

Create the project structure:

```
higt-bot/
├── configs/                # Hydra configs per experiment
├── data/
│   ├── raw/                # original pcap/argus files (gitignored)
│   ├── processed/          # parsed flows (parquet)
│   └── graphs/             # constructed PyG graph objects (.pt)
├── src/
│   ├── data/               # parsing, graph construction
│   ├── models/             # all model definitions
│   ├── training/           # train loop, eval, callbacks
│   ├── utils/              # logging, seeding, metrics
│   └── viz/                # plots and interpretability
├── notebooks/              # exploratory only — never trains models
├── experiments/            # logs and checkpoints (gitignored)
├── tests/                  # unit tests, especially data pipeline
├── pyproject.toml
├── requirements.txt
├── environment.yml         # conda alternative
├── .gitignore
└── README.md
```

Lock dependencies. Pin every version. Example `requirements.txt`:

```
torch==2.4.1
torch-geometric==2.6.1
torch-scatter==2.1.2
torch-sparse==0.6.18
numpy==1.26.4
pandas==2.2.3
scikit-learn==1.5.2
networkx==3.3
scapy==2.6.0
nfstream==6.5.3
pyarrow==17.0.0
hydra-core==1.3.2
wandb==0.18.5
matplotlib==3.9.2
seaborn==0.13.2
tqdm==4.66.5
pytest==8.3.3
```

Write a `set_seed()` utility and call it at the top of every entry point:

```python
def set_seed(seed: int = 42) -> None:
    import random, os, numpy as np, torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
```

Initialize git, commit the skeleton, push to a private repo. Every phase should end with a tagged commit (`v0.1-skeleton`, `v0.2-data`, `v0.3-parser`, etc.).

### Hardware and compute plan

Three machines, three distinct roles. The Mac has more capability here than you might expect — its Apple Silicon GPU (MPS backend) handles a meaningful subset of the workload.

- **MacBook Pro 2021 (Apple Silicon, 16 GB unified memory)** — *dev + secondary prototyping rig.* MPS works for stock PyTorch ops (`nn.TransformerEncoder`, `nn.MultiheadAttention`, dense matmul, `dense_diff_pool`), so you can prototype Phase 5 (temporal Transformer) and the dense parts of Phase 6 (DiffPool) at small scale on the GPU here. Real performance: ~2× CPU on transformer training, ~7–10× CPU on conv-heavy work. Effective ML memory ≈ 10–12 GB after macOS overhead. **The catch:** PyG's `torch-scatter` and `torch-sparse` custom kernels do *not* support MPS (confirmed by the PyG maintainer, late 2024). Sparse message-passing layers (`GINConv`, `GATConv` with sparse adjacency, etc.) either fail or fall back to CPU via `PYTORCH_ENABLE_MPS_FALLBACK=1` — which is slow enough that you'd rather just run them on the 3080. So: use the Mac freely for the temporal-Transformer prototype and dense DiffPool experiments; route sparse PyG layers to the 3080 instead.
- **RTX 5080 laptop (16 GB VRAM, Intel Ultra 9 HX)** — *primary training rig.* Blackwell architecture; needs CUDA 12.4+ wheels. Pin `torch==2.4.1+cu124`. Confirm `torch.cuda.get_device_capability()` ≥ `(12, 0)`. Expect a painful afternoon getting `torch-scatter`/`torch-sparse` to install on Blackwell — **do this on day 1, before any data code, so you know the env works**.
- **RTX 3080 desktop (10 GB VRAM)** — *secondary trainer.* Use for Phase 4 baselines (RF/GIN/GAT each finishes in <2 h) and seed replicates of finalized models. Keep a separate `requirements-cu121.txt` for it. 10 GB is too tight for DiffPool with N>300 — don't try the full HiGT-Bot here.

**Workflow:** code + Phase 5 prototype on Mac (MPS) → git push → train final HiGT-Bot on 5080 → run baselines and replicates on 3080 → unify results in W&B.

**Mac MPS quick verification gate:** `python -c "import torch; print(torch.backends.mps.is_available(), torch.backends.mps.is_built())"` should print `True True`. Set `PYTORCH_ENABLE_MPS_FALLBACK=1` in your shell rc — silent fallback for the few ops MPS still misses.

**Compute budget.** With mixed precision (`torch.cuda.amp`) and **cached temporal-Transformer embeddings** (see Phase 5), one full HiGT-Bot run on the 5080 ≈ 8–14 h. Total ~40 runs across baselines + ablations + robustness + 3 seeds → ~3–4 weeks of mostly-saturated training, *if all three machines stay busy*. Idle GPUs are the silent timeline killer.

### Gate

`python -c "import torch_geometric; import torch; print(torch.cuda.is_available(), torch_geometric.__version__)"` prints `True 2.6.x` on **both** the 5080 and the 3080.

### Pain points

- **Version mismatch between `torch` and `torch-geometric` extensions** (`torch-scatter`, `torch-sparse`) is the most common install failure. Use the official wheel index URL matching your CUDA version: 5080 needs `https://data.pyg.org/whl/torch-2.4.1+cu124.html`, 3080 needs `+cu121`. Do not let pip resolve these.
- **Blackwell (RTX 5080) is brand new.** PyG extension wheels for cu124 + sm_120 stabilized only recently. If pre-built wheels fail, fall back to `pip install torch-scatter --no-build-isolation` from source — slow but works.
- **Hydra + W&B together can be fiddly.** If short on time, start with simple YAML configs and argparse, add Hydra later. Do not skip W&B though — experiment tracking pays for itself within days, especially across three machines.

---

## Phase 1 — Acquire and inspect the data

**Duration:** Days 2–4.
**Goal:** Understand the data viscerally before writing parsers.

### What to do

Download:

- **CTU-13** from <https://www.stratosphereips.org/datasets-ctu13>. Focus on scenarios 9, 10, 11 (multi-bot, P2P-style). Download all 13 anyway for completeness.
- **IoT-23** from <https://www.stratosphereips.org/datasets-iot23>. The full dataset is 20+ GB; you can start with the lighter version. Look at captures labeled "Mirai," "Mozi," "Hakai," "Trojan."
- **MedBIoT** from its official page. Use as a generalization test set.

Write `src/data/inspect.py` that, given a scenario directory, prints:

- Number of distinct source / destination IPs
- Time range (first and last packet)
- Label distribution (bot / benign / background)
- Top 10 most communicative IPs and their roles

**Then — and this is critical — open the smallest scenario in Wireshark and look at it by hand.** Filter to a single bot IP. Look at the actual packets. Notice the timing (Mozi heartbeats every ~30 s, Mirai's scanning bursts, etc.), the typical payload sizes, the port patterns. This intuition will directly inform feature engineering in Phase 2 and graph construction in Phase 3. **Do not skip this step.**

Write `data/README.md` documenting: dataset path, scenario summary, label semantics, known quirks. Future-you and your advisor will thank you.

### Gate

You can run one script that answers: "How many P2P bot flows exist in CTU-13 scenario 10, and which source IPs are bots?" You can describe out loud what one Mozi bot's traffic looks like in Wireshark.

### Pain points

- **IoT-23 labels live in `conn.log.labeled` not `conn.log`.** Different scenarios use slightly different label vocabularies (`Malicious   PartOfAHorizontalPortScan`, `Benign   -`). Build a label normalization map early.
- **CTU-13 background traffic is huge** and unlabeled. You will need to decide whether to include it as "benign" (risky, may include unlabeled bots) or drop it (smaller, cleaner). Start by dropping background; add it back later as ablation.
- **Time zones.** CTU-13 timestamps are local Czech time, IoT-23 is UTC. Normalize to UTC early.
- **DHCP and NAT** mean the same physical device may have multiple IPs over time, and many devices may share one public IP. For CTU-13 (internal capture) this is mostly fine. For IoT-23 less so. Decide your node identity rule now: `(IP)` is simplest, `(IP, /24)` adds robustness, `(MAC)` is best when available.

---

## Phase 2 — Flow parser

**Duration:** Days 5–7.
**Goal:** A clean, tested DataFrame of bidirectional flows.

### What to do

Output schema (parquet, one row per bidirectional flow):

```
flow_id           int64       unique
src_ip            string
dst_ip            string
src_port          int
dst_port          int
protocol          string      tcp/udp/icmp/...
start_time        datetime64[ns, UTC]
end_time          datetime64[ns, UTC]
duration          float       seconds
bytes_fwd         int64
bytes_bwd         int64
pkts_fwd          int64
pkts_bwd          int64
mean_iat          float       inter-arrival time mean
std_iat           float
min_pkt_size      int
max_pkt_size      int
tcp_flag_dist     dict        per-flag counts
label             string      bot_<family> / benign / background
scenario          string      e.g. "ctu13-09"
```

Implementation paths by source format:

- **CTU-13 binetflow files** are CSV-like; pandas can read directly. Map their columns (StartTime, Dur, Proto, SrcAddr, Sport, Dir, DstAddr, Dport, State, sTos, dTos, TotPkts, TotBytes, SrcBytes, Label) into the schema above.
- **IoT-23 Zeek `conn.log.labeled`** is TSV; use the official `zeek-cut` or just pandas with proper field separators. Combine `conn.log` history with `labels` column.
- **Raw pcaps** (if you need to add an unseen dataset): use **NFStream** (`pip install nfstream`) — one line per flow with rich features. Faster and more correct than rolling your own.

Write tests in `tests/test_parser.py`:

```python
def test_parser_known_flow(small_pcap_fixture):
    flows = parse_pcap(small_pcap_fixture)
    expected = flows.query("src_ip == '10.0.0.5' and dst_port == 80").iloc[0]
    assert expected["bytes_fwd"] == 12500
    assert abs(expected["duration"] - 4.2) < 0.05
    assert expected["label"] == "benign"
```

Run these tests in CI (GitHub Actions free tier) from day one.

### Gate

Parsing 100 MB of CTU-13 scenario 10 takes under 2 minutes, produces a parquet file, and spot-checking 20 bot-labeled flows in pandas reveals they are genuinely bot traffic (consistent destination ports, suspicious peer counts, etc.).

### Pain points

- **Bidirectional vs unidirectional flows.** CTU-13's argus format gives bidirectional flows directly (each row is one A↔B conversation). Zeek conn.log is also bidirectional. NFStream is bidirectional by default. If you ever use a unidirectional source (raw NetFlow v5), you must merge `A→B` and `B→A` flows yourself by matching the 5-tuple and time window — easy to get wrong.
- **Label leakage from background.** Some CTU-13 background flows are actually bot flows that were never labeled. Do not assume "not labeled bot = definitely benign." Be explicit: treat unlabeled as `background` and exclude from training/eval, do not silently call it `benign`.
- **Very short flows (single-packet "flows")** wreak havoc on statistics — std and mean are undefined. Filter `pkts_total >= 2` for stats but keep them in counts.
- **Parsing speed.** A naive Python pcap parser is 100× slower than NFStream/CICFlowMeter. Use the C-backed tools.

---

## Phase 3 — Graph construction

**Duration:** Days 8–14. **This is the highest-risk phase. Budget a full week. Write tests.**

### What to do

We build a **host-centric, time-windowed, attributed graph**, one graph object per window.

Design decisions (lock these in a config file and document them in your thesis):

- **Window size W:** start with 300 seconds (5 min). Sweep {60, 180, 300, 600} as a later ablation.
- **Window stride:** non-overlapping (stride = W) is simplest. Overlapping (stride = W/2) gives more training data but creates dependence between samples — be careful with the train/test split.
- **Node identity:** start with `(IP)`. Drop bare-internet IPs that appear fewer than `min_flows = 3` times per window.
- **Max nodes per graph:** cap at **400**. Drop any window exceeding this, or aggressively raise `min_flows` for that window. This is a hard constraint imposed by 16 GB VRAM during DiffPool (dense `[B, N, N]` adjacency). Reviewers won't object — most graph-classification benchmarks cap at 300–500 nodes.
- **Edge definition:** one directed edge per `(src_ip, dst_ip)` pair within the window, aggregating all flows between them.
- **Edge features (per window):** total bytes, total packets, distinct dest ports, distinct source ports, mean flow duration, mean IAT, dominant protocol one-hot.
- **Node features (per window):** fan-out (distinct peers contacted), fan-in (distinct peers contacted by), port entropy, byte rate, packet rate, protocol diversity (Shannon entropy), out/in flow ratio, average flow duration.
- **Node label:** **bot if any flow involving this host in this window is bot-labeled**; benign otherwise. This matches the seminal Zhou et al. 2020 convention and is appropriate for P2P bots, which often communicate sparsely with peers — a 50%-of-flows threshold would systematically underlabel low-activity bot windows. The stricter `bot_flow_count / total_flow_count > 0.5` rule appears as a single sensitivity row in the Phase 8 ablation table, not as a competing primary.
- **Graph label:** bot if any node in the graph is bot; benign otherwise.

Save graphs as PyTorch Geometric `Data` objects:

```python
from torch_geometric.data import Data

data = Data(
    x=node_features,            # [num_nodes, num_node_feats]
    edge_index=edge_index,      # [2, num_edges]
    edge_attr=edge_features,    # [num_edges, num_edge_feats]
    y=node_labels,              # [num_nodes]   for node-level
    graph_y=graph_label,        # scalar         for graph-level
    scenario=scenario_id,       # for split control
    window_start=ts,            # for chronological splits
)
torch.save(data, f"data/graphs/{scenario}_{window_idx:05d}.pt")
```

**Split strategy — critical, often done wrong.** Use a **per-scenario chronological split** unless you have a strong reason otherwise:

- Train on scenarios {3, 4, 5, 9, 10}
- Val on scenarios {7, 11}
- Test on scenarios {2, 6, 8, 12}

This prevents label leakage where the same bot IPs appear in train and test. Do **not** randomly split graphs; bot IPs reappear across windows and you will get unrealistically high scores.

### Sanity check (build into the pipeline)

After construction, train a plain `sklearn.RandomForestClassifier` on the flat node features (ignoring graph structure) for node classification. You should get F1 ≥ 0.85 on a simple scenario. If you cannot, the labels are wrong. Fix before moving on.

### Gate

For CTU-13 scenario 10 you have hundreds of graph files. Visualize one with NetworkX + matplotlib: bot nodes should appear densely interconnected, benign nodes mostly peripheral. Random Forest on flat features hits F1 ≥ 0.85. Train/val/test scenarios documented in a config.

### Pain points

- **The split mistake.** Random splitting graphs across scenarios is the #1 way to write a paper that looks great and then collapses in review. Reviewers will ask for cross-scenario / cross-dataset generalization. Bake it in from day one.
- **Class imbalance.** P2P bot nodes are a small minority. Without weighting, your model will predict "benign" for everything and report 98% accuracy. Use `class_weight='balanced'` for sklearn, `pos_weight` in `BCEWithLogitsLoss` for PyTorch, and always report F1/PR-AUC, never just accuracy.
- **Empty windows.** Some windows have no traffic; skip them. Some have one node; also skip (a graph with one node has nothing to learn).
- **Node ordering must be consistent.** When you map IPs to node indices, do it deterministically (sorted IPs → 0, 1, 2…). Otherwise debugging is hell.
- **Memory.** Saving thousands of `.pt` files is fine. Loading them all into RAM is not. Use PyG's `Dataset` API (`InMemoryDataset` for small total size, `Dataset` lazy-loading for large).
- **Direction of edges.** Treat the graph as **directed** in construction (preserves info), then add a `make_undirected` flag for GNNs that need it. GAT and GIN work on undirected; some attention layers need explicit direction.

---

## Phase 4 — Baselines

**Duration:** Days 15–18 (trimmed from 7 → 4 days; the saved days go to Phase 6, where DiffPool stability eats time).
**Goal:** Three baselines that bracket the design space — no more. XGBoost and GCN are nice-to-haves; add them in Phase 8 if time permits.

### What to do

Implement, in order, each as a separate file under `src/models/`:

1. **Random Forest** on flat node features. `sklearn.ensemble.RandomForestClassifier(n_estimators=500, class_weight='balanced')`. The "non-graph" lower bound. Run on the 3080 — finishes in minutes.
2. **GAT** (2 layers, 4–8 heads, concatenated then averaged). The "attentional but flat" baseline.
3. **GIN** (3 layers, MLP epsilon-learned, sum aggregation). Your most serious flat baseline; many botnet-GNN papers use GIN, and the 2024 GraphSAINT+GIN paper on CTU-13 is the number to match.

*Optional, in Phase 8 if time:* XGBoost (often slightly beats RF), GCN (sanity middle ground).

Training loop should include: early stopping on val F1 (patience = 20), Adam optimizer with `lr=1e-3` and `weight_decay=1e-5`, learning rate scheduler (`ReduceLROnPlateau`), gradient clipping (`max_norm=1.0`), and W&B logging.

Evaluation: report on the test set only after model selection is final. Metrics: accuracy, precision, recall, F1, PR-AUC, ROC-AUC, confusion matrix, plus per-scenario breakdown.

### Gate

A results table with three rows × six metrics, logged to W&B. GIN should approach the numbers reported in the 2024 GraphSAINT+GIN paper on CTU-13. If it does not, your data pipeline is wrong — debug; do not proceed.

### Pain points

- **Comparing apples to apples.** Every baseline must use the same train/val/test split, the same feature normalization, the same evaluation script. Build a shared `evaluate()` function and call it everywhere.
- **Dropout placement.** In PyG, dropout goes between message-passing layers, not inside them. Misplacing it can lose 5+ F1 points.
- **GAT instability.** GAT can collapse to uniform attention if `negative_slope` and dropout aren't tuned. If GAT underperforms GCN, that's a sign — try different head counts and dropouts.
- **Inference batching for graphs.** PyG batches graphs by concatenating into one big disjoint graph (`Batch.from_data_list`). Make sure your evaluation respects per-graph boundaries.
- **Resist hyperparameter rabbit holes.** Each baseline should take half a day of tuning at most. Get them to "respectable," log everything, move on. You will tune the *final* model in Phase 8.

---

## Phase 5 — Temporal Transformer for node features

**Duration:** Days 22–28.
**Goal:** Replace hand-crafted node features with learned embeddings from per-device flow histories.

### What to do

Input: for each (node, window) pair, a tensor of shape `(num_flows, flow_feature_dim)` containing every flow involving that node, sorted by `start_time`.

Architecture:

```python
class TemporalFlowEncoder(nn.Module):
    def __init__(self, flow_feat_dim, d_model=64, nhead=4,
                 num_layers=2, max_flows=256, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(flow_feat_dim, d_model)
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len=max_flows)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=4*d_model,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

    def forward(self, flows, pad_mask):
        # flows: [B, L, F]   pad_mask: [B, L]  True for padded positions
        x = self.input_proj(flows)
        cls = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls, x], dim=1)
        cls_mask = torch.zeros(x.size(0), 1, dtype=torch.bool, device=x.device)
        pad_mask = torch.cat([cls_mask, pad_mask], dim=1)
        x = self.pos_enc(x)
        x = self.encoder(x, src_key_padding_mask=pad_mask)
        return x[:, 0]   # CLS token = node embedding
```

Cap `max_flows` at 256. For nodes with more flows, randomly sample 256 in training, keep evenly-spaced 256 in eval. Pad short sequences with zeros and use the mask.

Integration: plug `TemporalFlowEncoder` output as the node features `x` into the GIN baseline from Phase 4. Train end-to-end with the same loop.

### Gate

GIN + temporal-Transformer beats plain GIN by at least 1–2 F1 points on CTU-13 test. Log everything to W&B.

### Pain points

- **Variable-length sequences.** The most common bug is computing the mask wrong, then the model attends to padding positions and learns garbage. Print mask shapes on first forward pass; verify with a small synthetic case.
- **Overfitting.** With ~hundreds of windows per scenario the Transformer can memorize. Use dropout 0.1–0.3, weight decay 1e-5, and early stopping. Consider freezing the Transformer halfway through training.
- **Computational cost and caching.** Encoding every node's flow sequence per epoch is expensive — and on 16 GB VRAM with downstream DiffPool, you cannot afford to recompute it. **Once Phase 5 converges, freeze the temporal encoder and cache `[N, 64]` embeddings to parquet keyed by `(scenario, window_idx, node_idx)`.** Phase 6 and 7 then load embeddings instead of recomputing them — the difference is roughly 3× wall-clock and 2× peak VRAM.
- **`batch_first=True`** in PyTorch's Transformer is easy to forget. The default is `batch_first=False`, which silently does the wrong thing on batched data.

---

## Phase 6 — Hierarchical pooling layer

**Duration:** Days 29–38 (extended from 7 → 10 days; the original DiffPool authors themselves noted *"DIFFPOOL could be unstable to train, and there is significant variation in accuracy across different runs, even with the same hyperparameter setting"* — and trained for 3,000 epochs. Budget the time honestly).
**Goal:** Add multi-scale reasoning via DiffPool (or MinCutPool).

### Memory plan for 16 GB VRAM

DiffPool operates on **dense** `[B, N, N]` adjacency tensors — quadratic in node count. Without care this OOMs on the 5080. From day one:

- Keep `max_nodes_per_graph = 400` (enforced in Phase 3).
- Use mixed precision: wrap forward in `torch.cuda.amp.autocast()` and use `GradScaler`. Roughly halves VRAM.
- Batch size 4–8 (not 32). Use **gradient accumulation** to reach effective batch 32 if you need it for stability.
- Load **cached temporal-Transformer embeddings** from Phase 5 — never recompute the encoder during Phase 6 training.
- If memory still spikes, switch to **MinCutPool** earlier rather than later — it's leaner *and* the spectral interpretation reads cleaner in the thesis. Don't sink a week into rescuing DiffPool out of stubbornness.

### What to do

Start with **DiffPool**. Two parallel GNNs per pooling layer:

- **Embed GNN** computes node embeddings `Z = GNN_embed(X, A)` of shape `[N, d]`.
- **Pool GNN** computes a soft assignment `S = softmax(GNN_pool(X, A))` of shape `[N, K]`, where K = `num_clusters` (typically `ceil(0.25 * N)`).

Coarsened graph:

```
X_new = S^T @ Z         # [K, d]
A_new = S^T @ A @ S     # [K, K]
```

Auxiliary losses (add these to total loss, weights ~0.1 each):

```
L_link = ||A - S @ S^T||_F            # link prediction
L_ent  = (1/N) * sum_i H(S_i)         # entropy regularizer
```

Without these losses DiffPool collapses to a degenerate uniform assignment and gives no benefit.

Architecture (2-level hierarchy):

```
Input graph
    │
    ├─ GIN block (2 layers, 128-d)
    ▼
    DiffPool₁  → K = N/4 super-nodes
    │
    ├─ GIN block (1 layer, 128-d) on coarsened graph
    ▼
    DiffPool₂  → K = N/16 super-nodes
    │
    ▼
    Readout / classifier
```

If DiffPool is unstable (loss spikes, F1 plateaus at baseline), swap in **MinCutPool**:

```python
from torch_geometric.nn import dense_mincut_pool
x_new, adj_new, mincut_loss, ortho_loss = dense_mincut_pool(x, adj, s)
```

MinCutPool tends to be more numerically stable and the spectral interpretation makes for a cleaner thesis narrative.

### Gate

Hierarchical GIN matches or beats flat GIN on CTU-13. Visualize cluster assignments on one scenario (color nodes by argmax of `S`): bot nodes should cluster together. If they do not, DiffPool is collapsing — inspect, tune aux-loss weights.

### Pain points

- **Dense vs sparse.** PyG's DiffPool implementation operates on **dense** adjacency matrices (`[N, N]`). For graphs with thousands of nodes this is memory-heavy. Use `to_dense_adj` and `to_dense_batch`, and keep your max graph size bounded by node filtering in Phase 3.
- **Choice of K.** Too few clusters and you lose information; too many and pooling is meaningless. Start with K = N/4 and N/16 for the two levels. Sweep later.
- **Auxiliary loss weights.** If link/entropy losses dominate, classification stalls. If they're too small, pooling degenerates. Start at 0.1 each, tune via validation F1.
- **Initialization.** DiffPool is sensitive to init. Use Kaiming-normal for linear layers in the pool GNN; uniform won't work as well.
- **Forward pass logging.** Print mean and std of `S` at each layer for the first few epochs. If `S` becomes uniform (every entry ≈ 1/K), entropy regularizer is too weak or pooling is collapsing.

---

## Phase 7 — Graph Transformer on the coarsened graph

**Duration:** Days 36–42.
**Goal:** Add long-range global reasoning across communities.

### What to do

After the final DiffPool layer you have a small graph (typically 20–80 super-nodes). Apply a Graph Transformer.

**Two variants — implement both as an ablation:**

**Variant A: structural Graph Transformer (edge-aware).** Use PyG's `TransformerConv`:

```python
from torch_geometric.nn import TransformerConv

self.gt1 = TransformerConv(in_channels=d, out_channels=d, heads=4,
                            concat=False, dropout=0.1, edge_dim=edge_d)
self.gt2 = TransformerConv(d, d, heads=4, concat=False, dropout=0.1)
```

This respects edges in the coarsened graph — useful when community-to-community edges are meaningful.

**Variant B: pure global attention (edge-free).** Treat super-nodes as a sequence and apply `nn.MultiheadAttention`. Every super-node attends to every other, with no edge bias:

```python
class GlobalAttentionBlock(nn.Module):
    def __init__(self, d, nhead=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, nhead, dropout=dropout,
                                           batch_first=True)
        self.norm1 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 4*d), nn.GELU(),
                                nn.Linear(4*d, d))
        self.norm2 = nn.LayerNorm(d)

    def forward(self, x):
        h, _ = self.attn(x, x, x)
        x = self.norm1(x + h)
        x = self.norm2(x + self.ff(x))
        return x
```

Report both variants in the paper and let the empirical result speak.

### Full HiGT-Bot model

```python
class HiGTBot(nn.Module):
    def __init__(self, ...):
        super().__init__()
        self.temporal_enc = TemporalFlowEncoder(...)
        self.gin1 = GINBlock(d, 2)
        self.pool1 = DiffPoolLayer(d, ratio=0.25)
        self.gin2 = GINBlock(d, 1)
        self.pool2 = DiffPoolLayer(d, ratio=0.25)
        self.gt1 = GlobalAttentionBlock(d)
        self.gt2 = GlobalAttentionBlock(d)
        self.readout = nn.Sequential(nn.Linear(2*d, d), nn.ReLU(),
                                      nn.Linear(d, num_classes))

    def forward(self, data):
        x = self.temporal_enc(data.flows, data.flow_mask)  # node feats
        x = self.gin1(x, data.edge_index)
        x, adj, l1 = self.pool1(x, data.adj)
        x = self.gin2(x, adj_to_edge_index(adj))
        x, adj, l2 = self.pool2(x, adj)
        x = self.gt1(x)
        x = self.gt2(x)
        z = torch.cat([x.mean(dim=1), x.max(dim=1).values], dim=-1)
        return self.readout(z), l1 + l2
```

For **node-level classification**, instead of pooling at the end, project Graph-Transformer outputs back to original nodes using `S₁ @ S₂ @ h_gt`, then add U-Net-style skip connections from `gin1` features, then classify per node.

### Gate

Full HiGT-Bot beats every baseline on at least two of three datasets, in both F1 and PR-AUC. If not, run quick ablations (drop temporal Transformer? drop pooling? drop Graph Transformer?) to find the weak link and iterate.

### Kill criterion (decide *before* you need it)

If full HiGT-Bot does not beat `GIN + temporal-Transformer` by **≥1.5 F1 on CTU-13** by end of Phase 7 week 2, **pivot the contribution** rather than panic-tune for two more weeks:

- Drop the Graph Transformer from the headline claim.
- Reposition the paper around (a) interpretability via DiffPool community assignments — bot communities are visually identifiable and explanation-friendly, and (b) the hierarchical-efficiency argument from §1 (hierarchy makes any global mechanism tractable on host graphs).
- The `temporal-Transformer + DiffPool` combination is still novel for P2P-IoT and publishable at IEEE TrustCom or ICC even without the GT layer.

Pre-committing to this pivot now is the difference between a publishable result at week 12 and a panicked rewrite at week 14.

### Pain points

- **O(K²) attention is only manageable because K is small.** If you ever skip pooling and apply a Transformer to the original graph, you will OOM on anything realistic. Sell this in the paper as the *computational rationale* for hierarchy: pooling makes the Transformer feasible.
- **Pre-norm vs post-norm.** Modern Transformers use pre-norm (LayerNorm before attention/FFN). Stick with it; it trains more stably.
- **Residual connections required.** No residual = vanishing gradients in 2+ layer Transformers.
- **Un-pooling is fragile.** Skip connections from earlier GIN layers are not optional — they're how the node-level head recovers information that pooling smeared out.
- **Don't lose information during pooling.** Save `S₁`, `S₂` from the forward pass for use in un-pooling AND for visualization in Phase 8.

---

## Phase 8 — Ablations, robustness, and the paper-ready story

**Duration:** Days 43–60.
**Goal:** A results section a reviewer cannot tear apart.

### Ablations (one table)

Run, holding everything else equal:

| Variant | Components |
|---|---|
| Full model | Temporal-T + GIN + DiffPool + GIN + GT |
| − Temporal-T | hand-crafted features only |
| − Hierarchy | flat GIN + GT |
| − Graph Transformer | hierarchical GIN only |
| − Both pooling levels | flat GIN + temporal-T |
| Replace DiffPool with MinCutPool | |
| Replace DiffPool with SAGPool | |
| Variant A (TransformerConv) vs Variant B (global attention) | |

Each row must tell a story. If `− Hierarchy` drops F1 the most, your hierarchy claim is strong.

### Robustness

- **Edge perturbation:** randomly drop 5%, 10%, 20% of edges at test time. Plot F1 vs perturbation rate. Stable degradation = a robust model.
- **Label noise:** flip 1%, 5%, 10% of training labels. Measure test F1.
- **Cross-dataset:** train on CTU-13, test on IoT-23, and vice versa. This is the single most reviewer-pleasing result you can produce. Expect a drop; report it honestly.

### Efficiency

Report:

- Total parameter count.
- Peak GPU memory at inference.
- Inference time per graph (median of 1000 graphs).
- Compare against GIN baseline. The hierarchy-makes-Transformer-tractable argument needs these numbers.

### Interpretability

- Visualize DiffPool assignments on a known scenario: do clusters correspond to the actual bot subgraph? Use a force-directed layout and color by cluster.
- Run **GNN-Explainer** or **PGExplainer** on a few correctly-classified bot nodes; surface which edges contributed most.
- Visualize Graph Transformer attention weights on the coarsened graph.

### Final gate

You have, all logged to W&B and exported to a `results/` directory:

- A main results table (5+ baselines × 3 datasets × 6 metrics).
- An ablation table.
- Robustness plots.
- Efficiency comparison.
- 3–5 interpretability figures.

This is the spine of your paper's Section 4 and Section 5.

---

## 12. Top 12 pain points and warnings

1. **Random splitting of graphs leaks bot identities across train/test.** Use per-scenario chronological splits. This is the #1 reviewer trap.
2. **Class imbalance silently inflates accuracy.** Always report F1 and PR-AUC, never just accuracy. Use `class_weight='balanced'` / `pos_weight`.
3. **Phase 3 (graph construction) is where projects die.** Budget a full week. Write tests. Sanity-check with Random Forest on flat features before believing your graphs are correct.
4. **N-BaIoT cannot be used for graph construction.** It ships pre-aggregated features, not flows. Use it only for cross-domain node-level cross-checks.
5. **PyG version + torch-scatter/torch-sparse mismatches** are the #1 install failure. Use the official wheel index URL for your CUDA version.
6. **DiffPool collapses without auxiliary losses.** Always include link-prediction and entropy losses. Tune their weights.
7. **`batch_first=True`** in `nn.TransformerEncoderLayer` is easy to forget. Default `False` silently does the wrong thing.
8. **Padding masks must mask padded positions (not real ones).** Print mask shapes the first time; verify on a tiny synthetic example.
9. **Random seeds.** Set them everywhere (Python, NumPy, PyTorch, CUDA, `PYTHONHASHSEED`). Without this, you cannot reproduce results — including for the camera-ready paper.
10. **Don't tune on the test set.** Use val for model selection. Test is opened *once* per paper draft.
11. **W&B from day one.** Retro-fitting experiment tracking when writing the paper is misery. Log every run, no exceptions.
12. **Cross-dataset generalization will look worse than within-dataset.** Report it anyway. Reviewers respect honesty and punish suspiciously perfect numbers.

---

## 13. Reproducibility checklist

Before submitting a paper, confirm:

- [ ] `set_seed(42)` is called at the top of every entry point.
- [ ] `requirements.txt` and `environment.yml` pin every package version.
- [ ] Data download script is in the repo (or detailed instructions exist).
- [ ] Train/val/test split is deterministic and documented.
- [ ] All hyperparameters live in YAML configs, not magic numbers.
- [ ] W&B project link is in the README.
- [ ] At least one checkpoint per model variant is saved under `experiments/`.
- [ ] A `make reproduce` target (or shell script) runs the full pipeline end-to-end on a small subset.
- [ ] Unit tests cover the parser and graph constructor.
- [ ] Code is on GitHub (private during review, public on acceptance).

---

## 14. Suggested cadence and milestones

| Week | Phases | Milestone |
|---|---|---|
| 1 | Phase 0–1 | Repo skeleton on **all three machines**, PyG installs verified on 5080 + 3080, datasets downloaded |
| 2 | Phase 2 | Flow parser working with tests |
| 3 | Phase 3 (week 1) | First graphs constructed, `max_nodes=400` enforced, visualization sane |
| 4 | Phase 3 (week 2) | Sanity-check F1 ≥ 0.85 with Random Forest on flat features |
| 5 | Phase 4 (4 days) + Phase 5 start | Three baselines logged on 3080, results table v1; temporal Transformer scaffolding |
| 6 | Phase 5 | Temporal Transformer integrated, beats flat GIN, **embeddings cached to parquet** |
| 7 | Phase 6 (week 1) | DiffPool integrated with mixed precision, stable training on 5080 |
| 8 | Phase 6 (week 2) | Hierarchical GIN matches or beats flat GIN — *or* swap to MinCutPool and continue |
| 9 | Phase 6 buffer / Phase 7 start | Graph Transformer layer added, full model trains end-to-end |
| 10 | Phase 7 | HiGT-Bot beats baselines on CTU-13 — **or kill criterion fires, pivot scope** |
| 11 | Phase 8 (week 1) | Ablations (3080 in parallel) and robustness experiments |
| 12 | Phase 8 (week 2) | Interpretability, efficiency, figures |
| 13 | Thesis writing | Methods + Results |
| 14 | Thesis writing | Related Work + Introduction + Conclusion |
| 15 | Paper extraction | Trim thesis to 10–12 pages |
| 16 | Buffer | Reviewer revisions, polishing |

This is aggressive but achievable if Phases 3 and 6 don't slip.

---

## 15. Useful references

**Papers — competitors and inspiration**

- PeerG (Computers & Security 2024) — closest P2P-specific competitor, contrastive learning approach. <https://www.sciencedirect.com/science/article/abs/pii/S0167404824000762>
- MalHAPGNN (Sensors 2025) — hierarchical attention pooling for malware. <https://www.mdpi.com/1424-8220/25/2/374>
- GraphSAINT + GIN for IoT botnets (Mathematics 2024). <https://www.mdpi.com/2227-7390/12/9/1315>
- BotLGT (Neurocomputing 2025) — LLM + Graph Transformer for social bots. <https://www.sciencedirect.com/science/article/abs/pii/S0925231225021253>
- DiffPool (NeurIPS 2018) — the original hierarchical pooling paper. <https://cs.stanford.edu/people/jure/pubs/diffpool-neurips18.pdf>
- Graph construction insights (arXiv 2025). <https://arxiv.org/html/2603.06654>
- Automating Botnet Detection with GNNs (early seminal paper). <https://arxiv.org/pdf/2003.06344>

**Datasets**

- CTU-13: <https://www.stratosphereips.org/datasets-ctu13>
- IoT-23: <https://www.stratosphereips.org/datasets-iot23>
- N-BaIoT (UCI): <https://archive.ics.uci.edu/ml/datasets/detection_of_IoT_botnet_attacks_N_BaIoT>

**Tools and docs**

- PyTorch Geometric: <https://pytorch-geometric.readthedocs.io>
- NFStream: <https://www.nfstream.org>
- Weights & Biases: <https://docs.wandb.ai>
- Zeek: <https://docs.zeek.org>

---

**Final note.** This document is a contract with future-you. When you're three months in and tempted to skip a validation gate "just this once" because you're behind schedule — don't. Re-read Phase 3's pain points and remember that *every* skipped sanity check shows up later as a reviewer comment, a retracted claim, or a result that won't reproduce. Slow is smooth, smooth is fast.
