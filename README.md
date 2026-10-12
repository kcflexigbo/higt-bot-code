# HiGT-Bot

**Hierarchical Graph-Transformer Learning for Botnet Detection in IoT Networks**

Master's thesis implementation. The full plan, including phase-by-phase gates and pain points, lives in [`../HiGT-Bot_Implementation_Plan.md`](../HiGT-Bot_Implementation_Plan.md). Read it before changing the model.

## Corrected pipeline (October 2026) and HiGT-Bot v2

The results in the paper and thesis use a corrected pipeline. Three faults in the original one were fixed:

1. **Labels.** A host is a bot only if it *originates* malicious flows (not if it merely receives them); for MedBIoT only the testbed devices count. Relabel with `scripts/relabel_infected_src.py` (writes `data/graphs_src2`).
2. **Host features.** The nine node features are computed from *all* of a host's flows in the window, not only flows kept in the pruned graph. Recompute with `scripts/refeature_graphs.py` (writes `data/graphs_v3`).
3. **Message direction.** `HIGT_DIRECTED=1` switches the graph layers to a direction-aware GINE (`src/models/dir_gine.py`: ego, incoming and outgoing messages kept separate).

Select the corrected graphs with `HIGT_GRAPHS_DIR=data/graphs_v3`; without these variables every script behaves as before.

- **HiGT-Bot v2** (gradient-boosted trees on host features + in-window group similarity, `src/features/host_blocks.py`): `scripts/higt_v2_trees.py`
- Summary tables: `scripts/summarise_v3.py`; paired bootstrap: `scripts/bootstrap_significance.py`; robustness and latency: `scripts/robustness_efficiency_v3.py`; leave-one-dataset-out: `scripts/lodo_v3.py`; figures: `scripts/make_v3_tables_figures.py`
- Results (text/JSON) are under `data/inspection_logs/v3*`. Per-window prediction arrays, graphs and embeddings are not stored in the repository; the scripts regenerate them.

## Project layout

```
configs/                Hydra configs per experiment
data/
  raw/                  original pcap/argus files (gitignored)
  processed/            parsed flows (parquet, gitignored)
  graphs/               constructed PyG graph objects (.pt, gitignored)
src/
  data/                 parsing, graph construction
  models/               model definitions (RF, GIN, HiGT-Bot, ...)
  training/             train loop, eval, callbacks
  utils/                seeding, metrics, logging
  viz/                  plots and interpretability
notebooks/              exploratory only — never trains models
experiments/            logs and checkpoints (gitignored)
tests/                  unit tests (especially data pipeline)
scripts/                one-off entry points (smoke_test.py, ...)
```

## Setup

The project uses [`uv`](https://docs.astral.sh/uv/) for dependency management. Install uv first.

### MacBook Pro (Apple Silicon, MPS)

```bash
uv sync
uv pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.4.1+cpu.html
uv run python scripts/smoke_test.py
```

The Mac is for dev, the temporal-Transformer prototype, and dense DiffPool experiments. Sparse PyG message-passing layers fall back to CPU on MPS — route those to a CUDA box.

### RTX 5080 laptop (Blackwell, CUDA 12.4+)

```bash
uv sync
uv pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu124  # overrides Mac wheel
uv pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.4.1+cu124.html
uv run python scripts/smoke_test.py
```

Primary training rig. Use mixed precision (`torch.cuda.amp`) in Phase 6+ to fit DiffPool dense adjacency in 16 GB VRAM.

### RTX 3080 desktop (Ampere, CUDA 12.1)

```bash
uv sync
uv pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121
uv pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.4.1+cu121.html
uv run python scripts/smoke_test.py
```

Secondary rig: baselines, seed replicates, ablation rows that don't need DiffPool.

## Phase 0 gate

`uv run python scripts/smoke_test.py` should print `Phase 0 gate PASSED` on whichever machine you set up.

## Reproducibility

- `set_seed(42)` is called at the top of every entry point — see `src/utils/seeding.py`.
- Dependencies pinned in `pyproject.toml` and locked in `uv.lock`.
- Train/val/test split is per-scenario chronological; never random across windows (see Phase 3 of the plan).
