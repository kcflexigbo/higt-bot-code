# Cross-Dataset (LODO) Experiment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run three leave-one-dataset-out (LODO) training pipelines (hold out CTU-13 / IoT-23 / MedBIoT) using the full Phase 4–7 ladder, producing a cross-dataset comparison table without modifying any existing finalized result.

**Architecture:** Additive parameterization. `SplitSpec` and `load_split_files` in `src/data/dataset.py` gain an opt-in LODO branch. Every trainer/baseline script gains a `--split-config` flag (plus checkpoint/results directory flags where needed) whose defaults match current hardcoded paths, so existing reproducibility is byte-identical. A new `scripts/run_lodo_ladder.py` orchestrates the 8-stage ladder per LODO direction via clean subprocesses, writes `.done` markers and `manifest.json`, and validates pre/post that nothing in `experiments/phase{5,6,7}/` or `data/flow_embeddings/` was overwritten. Three new YAML configs (`configs/split_lodo_*.yaml`) describe each LODO split.

**Tech Stack:** Python 3.11, PyTorch 2.11, PyTorch Geometric 2.7, Hydra-style YAML configs (already in repo), pytest, uv for dependency management. All on Windows + RTX 5080.

**Spec reference:** `docs/superpowers/specs/2026-06-03-cross-dataset-lodo-experiment-design.md`

**Working directory:** All commands assume the shell is `cd`'d into `higt-bot-code/` (the inner project directory under `Yakub_thesis/`). Relative paths (`configs/`, `experiments/`, `tests/`, etc.) are relative to `higt-bot-code/`.

---

## File Structure

**New files:**
- `configs/split_lodo_ctu13.yaml` — LODO split spec, CTU-13 held out
- `configs/split_lodo_iot23.yaml` — LODO split spec, IoT-23 held out
- `configs/split_lodo_medbiot.yaml` — LODO split spec, MedBIoT held out
- `scripts/run_lodo_ladder.py` — subprocess orchestrator for one LODO direction
- `scripts/aggregate_lodo_results.py` — produces comparison table and per-scenario breakdown from result JSONs
- `tests/test_lodo_split.py` — LODO-branch unit tests for `dataset.py`
- `tests/test_run_lodo_ladder.py` — orchestrator unit tests (mocked subprocess)

**Modified files (additive only — defaults preserve current behavior):**
- `src/data/dataset.py` — `SplitSpec` gains `holdout_dataset` and `lodo_test_scenarios`; `load_split_files` gains LODO branch
- `scripts/train_phase5.py` — add `--split-config`, `--ckpt-dir`, `--results-dir`
- `scripts/train_phase6_sparse.py` — add `--split-config`, `--ckpt-dir`, `--results-dir`
- `scripts/train_phase7.py` — add `--split-config`, `--ckpt-dir`, `--results-dir`
- `scripts/cache_embeddings.py` — add `--split-config`
- `scripts/baselines/run_rf.py` — add `--split-config`
- `scripts/baselines/run_xgb.py` — add `--split-config`
- `scripts/baselines/run_gnn.py` — add `--split-config`

**Untouched:**
- All model code in `src/models/*`
- All training loops in `src/training/*`
- Evaluation code (`src/training/evaluate.py`)
- `configs/split.yaml`, `configs/phase5.yaml`, `configs/phase6.yaml`
- All existing checkpoints, embeddings, logs

---

## Task 1: Pre-implementation snapshot and safety baseline

**Files:**
- Create: `experiments/lodo/.pre_run_snapshot.txt`

This task captures the mod-times of every file in `experiments/phase{5,6,7}/` and `data/flow_embeddings/`. The orchestrator will use this snapshot to detect any accidental overwrite throughout the experiment.

- [ ] **Step 1: Run snapshot command**

```powershell
New-Item -ItemType Directory -Force -Path experiments/lodo | Out-Null
Get-ChildItem -Path experiments/phase5, experiments/phase6, experiments/phase7, data/flow_embeddings -Recurse -File -ErrorAction SilentlyContinue |
  ForEach-Object { "{0}`t{1}" -f $_.FullName, $_.LastWriteTimeUtc.ToString("o") } |
  Set-Content -Encoding utf8 experiments/lodo/.pre_run_snapshot.txt
Write-Host "Snapshot rows: $((Get-Content experiments/lodo/.pre_run_snapshot.txt).Count)"
```

Expected: a count > 0 printed; the file `experiments/lodo/.pre_run_snapshot.txt` exists.

- [ ] **Step 2: Commit**

```bash
cd higt-bot-code
git add experiments/lodo/.pre_run_snapshot.txt docs/superpowers/specs/2026-06-03-cross-dataset-lodo-experiment-design.md docs/superpowers/plans/2026-06-03-cross-dataset-lodo-experiment.md
git commit -m "chore(lodo): pre-run snapshot and design/plan docs

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: Add LODO fields to `SplitSpec` dataclass

**Files:**
- Modify: `src/data/dataset.py:30-56`
- Test: `tests/test_lodo_split.py` (new)

- [ ] **Step 1: Create the failing test file**

Create `tests/test_lodo_split.py` with this content:

```python
"""LODO (leave-one-dataset-out) unit tests for src/data/dataset.py.

These tests exercise the optional LODO branch added in the LODO experiment.
They never touch the canonical configs/split.yaml — instead they construct
SplitSpec objects directly so the tests are independent of disk state.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.data.dataset import SplitSpec


def test_split_spec_lodo_fields_default_to_none_and_empty() -> None:
    """SplitSpec should remain backward compatible: when LODO fields are not
    set, holdout_dataset is None and lodo_test_scenarios is empty."""
    spec = SplitSpec(
        version=1,
        seed=42,
        train_frac=0.6,
        val_frac=0.2,
        train_scenarios=["scenario-a", "scenario-b"],
        holdout_test_scenarios=["scenario-c"],
    )
    assert spec.holdout_dataset is None
    assert spec.lodo_test_scenarios == []


def test_split_spec_lodo_fields_settable() -> None:
    """LODO fields accept explicit values."""
    spec = SplitSpec(
        version=2,
        seed=42,
        train_frac=0.8,
        val_frac=0.2,
        train_scenarios=["a", "b"],
        holdout_test_scenarios=[],
        holdout_dataset="iot23",
        lodo_test_scenarios=["iot23-1-1", "iot23-3-1"],
    )
    assert spec.holdout_dataset == "iot23"
    assert spec.lodo_test_scenarios == ["iot23-1-1", "iot23-3-1"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_lodo_split.py -v`
Expected: FAIL — `TypeError: SplitSpec.__init__() got an unexpected keyword argument 'holdout_dataset'`

- [ ] **Step 3: Add fields to `SplitSpec`**

In `src/data/dataset.py`, modify the `SplitSpec` dataclass (currently at lines 30-56) to add the two new fields. The diff:

```python
@dataclass
class SplitSpec:
    version: int
    seed: int
    train_frac: float
    val_frac: float
    train_scenarios: list[str]
    holdout_test_scenarios: list[str]
    excluded: list[str] = field(default_factory=list)
    holdout_dataset: str | None = None
    lodo_test_scenarios: list[str] = field(default_factory=list)
```

The two new lines go after `excluded`. Defaults preserve all existing behavior.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_lodo_split.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Run the full existing dataset test suite to confirm no regression**

Run: `uv run pytest tests/test_dataset.py -v`
Expected: PASS (all 7 existing tests).

- [ ] **Step 6: Commit**

```bash
cd higt-bot-code
git add src/data/dataset.py tests/test_lodo_split.py
git commit -m "feat(dataset): add optional LODO fields to SplitSpec

Adds holdout_dataset and lodo_test_scenarios with defaults that preserve
the existing mixed-pool behavior byte-identically. Covered by new
test_lodo_split.py.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: `SplitSpec.load()` parses LODO YAML keys

**Files:**
- Modify: `src/data/dataset.py:44-56`
- Test: `tests/test_lodo_split.py`

- [ ] **Step 1: Append failing test to `tests/test_lodo_split.py`**

Append these tests to `tests/test_lodo_split.py`:

```python
def _write_yaml(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "split.yaml"
    p.write_text(body, encoding="utf-8")
    return p


def test_split_spec_load_without_lodo_fields(tmp_path: Path) -> None:
    """Loading a YAML without LODO keys yields holdout_dataset=None."""
    cfg = _write_yaml(tmp_path, """
split_version: 7
seed: 42
train_frac: 0.6
val_frac: 0.2
train: [a, b]
holdout_test: [c]
""")
    spec = SplitSpec.load(cfg)
    assert spec.version == 7
    assert spec.holdout_dataset is None
    assert spec.lodo_test_scenarios == []
    assert spec.train_scenarios == ["a", "b"]


def test_split_spec_load_with_lodo_fields(tmp_path: Path) -> None:
    """Loading a YAML with LODO keys populates the new fields."""
    cfg = _write_yaml(tmp_path, """
split_version: lodo-iot23-v1
seed: 42
train_frac: 0.8
val_frac: 0.2
holdout_dataset: iot23
train: [ctu13-1, ctu13-2, medbiot-bashlite_mal_CC_all]
lodo_test_scenarios: [iot23-1-1, iot23-3-1, iot23-8-1]
excluded: [ctu13-5]
""")
    spec = SplitSpec.load(cfg)
    assert spec.holdout_dataset == "iot23"
    assert spec.lodo_test_scenarios == ["iot23-1-1", "iot23-3-1", "iot23-8-1"]
    assert spec.train_scenarios == ["ctu13-1", "ctu13-2", "medbiot-bashlite_mal_CC_all"]
    assert spec.excluded == ["ctu13-5"]
```

Note: `SplitSpec.load()` currently coerces `version` with `int()`. The string `"lodo-iot23-v1"` will fail that. Step 3 will fix `load()` to accept string versions.

- [ ] **Step 2: Run tests to verify failure**

Run: `uv run pytest tests/test_lodo_split.py -v`
Expected: FAIL on `test_split_spec_load_with_lodo_fields` (string version raises ValueError) and FAIL on `test_split_spec_load_without_lodo_fields` (no `holdout_dataset` attribute set from yaml — actually this would silently default to None, so it might pass — but we still need to fix the version coercion to support both styles).

Actually verify by running. The without-LODO test passes immediately (defaults handle it); the with-LODO test fails at version coercion.

- [ ] **Step 3: Update `SplitSpec.load()` and the `version` field**

In `src/data/dataset.py`, replace the existing `version: int` field declaration and `load()` classmethod with:

```python
@dataclass
class SplitSpec:
    version: int | str            # accepts integer (legacy) or string (lodo-*-v1) labels
    seed: int
    train_frac: float
    val_frac: float
    train_scenarios: list[str]
    holdout_test_scenarios: list[str]
    excluded: list[str] = field(default_factory=list)
    holdout_dataset: str | None = None
    lodo_test_scenarios: list[str] = field(default_factory=list)

    @property
    def test_frac(self) -> float:
        return max(0.0, 1.0 - self.train_frac - self.val_frac)

    @classmethod
    def load(cls, path: Path = DEFAULT_SPLIT) -> "SplitSpec":
        with path.open() as f:
            cfg = yaml.safe_load(f)
        raw_version = cfg["split_version"]
        # Preserve int coercion for legacy integer versions; pass strings through.
        try:
            version: int | str = int(raw_version)
        except (TypeError, ValueError):
            version = str(raw_version)
        return cls(
            version=version,
            seed=int(cfg["seed"]),
            train_frac=float(cfg["train_frac"]),
            val_frac=float(cfg["val_frac"]),
            train_scenarios=list(cfg["train"]),
            holdout_test_scenarios=list(cfg.get("holdout_test", [])),
            excluded=list(cfg.get("excluded", [])),
            holdout_dataset=cfg.get("holdout_dataset"),
            lodo_test_scenarios=list(cfg.get("lodo_test_scenarios", [])),
        )
```

The changes from the original:
- `version` type annotation now `int | str`
- `version` parsed with int-first / fall-back-to-str (legacy integer YAML still produces an int)
- Two new lines populating `holdout_dataset` and `lodo_test_scenarios` from the YAML

- [ ] **Step 4: Run all LODO tests to verify pass**

Run: `uv run pytest tests/test_lodo_split.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Run the existing dataset test suite**

Run: `uv run pytest tests/test_dataset.py -v`
Expected: PASS. The legacy `version >= 1` check still works because `int()` is attempted first; existing `split.yaml` has integer 1.

- [ ] **Step 6: Commit**

```bash
cd higt-bot-code
git add src/data/dataset.py tests/test_lodo_split.py
git commit -m "feat(dataset): SplitSpec.load parses LODO YAML keys

Accept string version labels (lodo-iot23-v1 etc) by trying int() first
and falling back to str. Parse optional holdout_dataset and
lodo_test_scenarios keys with safe defaults.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: `load_split_files` LODO branch

**Files:**
- Modify: `src/data/dataset.py:119-136`
- Test: `tests/test_lodo_split.py`

- [ ] **Step 1: Append failing tests to `tests/test_lodo_split.py`**

These tests build a synthetic graphs directory in `tmp_path` so they do not depend on `data/graphs/` state.

Append:

```python
def _populate_graphs(tmp_graphs: Path, scenarios: dict[str, int]) -> None:
    """Create empty window_*.pt files for each scenario name."""
    for scenario, n in scenarios.items():
        d = tmp_graphs / scenario
        d.mkdir(parents=True, exist_ok=True)
        for i in range(n):
            (d / f"window_{i:05d}.pt").write_bytes(b"")


def test_load_split_files_lodo_mode_assigns_test_correctly(tmp_path: Path) -> None:
    """In LODO mode, every window of lodo_test_scenarios goes to test."""
    from src.data.dataset import load_split_files

    graphs_dir = tmp_path / "graphs"
    _populate_graphs(graphs_dir, {
        "ctu13-1": 10,
        "ctu13-2": 10,
        "iot23-1-1": 5,
        "iot23-3-1": 7,
    })
    spec = SplitSpec(
        version="lodo-iot23-v1",
        seed=42,
        train_frac=0.8,
        val_frac=0.2,
        train_scenarios=["ctu13-1", "ctu13-2"],
        holdout_test_scenarios=[],
        holdout_dataset="iot23",
        lodo_test_scenarios=["iot23-1-1", "iot23-3-1"],
    )

    tr = load_split_files("train", spec, graphs_dir)
    va = load_split_files("val", spec, graphs_dir)
    te = load_split_files("test", spec, graphs_dir)

    # Train scenarios chronologically split 0.8/0.2 each: ctu13-1 -> 8/2, ctu13-2 -> 8/2.
    assert len(tr) == 16
    assert len(va) == 4
    # Test = full window sequences of every LODO test scenario: 5 + 7 = 12.
    assert len(te) == 12
    # No test scenario file comes from the train_scenarios pool.
    assert all("ctu13" not in str(p) for p in te)
    # No train/val file comes from the LODO test pool.
    assert all("iot23" not in str(p) for p in tr + va)


def test_load_split_files_lodo_mode_ignores_holdout_test_scenarios(tmp_path: Path) -> None:
    """The existing holdout_test_scenarios field is ignored in LODO mode."""
    from src.data.dataset import load_split_files

    graphs_dir = tmp_path / "graphs"
    _populate_graphs(graphs_dir, {
        "ctu13-1": 10,
        "medbiot-torii_mal_all": 4,
        "iot23-1-1": 5,
    })
    spec = SplitSpec(
        version="lodo-iot23-v1",
        seed=42,
        train_frac=0.8,
        val_frac=0.2,
        train_scenarios=["ctu13-1", "medbiot-torii_mal_all"],
        holdout_test_scenarios=["should-not-appear-anywhere"],  # ignored in LODO mode
        holdout_dataset="iot23",
        lodo_test_scenarios=["iot23-1-1"],
    )

    te = load_split_files("test", spec, graphs_dir)
    # Only iot23-1-1 windows: 5 of them. holdout_test_scenarios is ignored.
    assert len(te) == 5
    assert all("iot23-1-1" in str(p) for p in te)


def test_load_split_files_non_lodo_mode_unchanged(tmp_path: Path) -> None:
    """Regression: when holdout_dataset is None, behavior matches the old code path
    (chronological 60/20/20 + holdout_test_scenarios appended to test)."""
    from src.data.dataset import load_split_files

    graphs_dir = tmp_path / "graphs"
    _populate_graphs(graphs_dir, {
        "a": 10,
        "b": 10,
        "hot": 5,
    })
    spec = SplitSpec(
        version=1,
        seed=42,
        train_frac=0.6,
        val_frac=0.2,
        train_scenarios=["a", "b"],
        holdout_test_scenarios=["hot"],
    )

    tr = load_split_files("train", spec, graphs_dir)
    va = load_split_files("val", spec, graphs_dir)
    te = load_split_files("test", spec, graphs_dir)

    # 60/20/20 chronological: 6/2/2 per scenario, ×2 = 12/4/4. Plus 5 holdout in test.
    assert len(tr) == 12
    assert len(va) == 4
    assert len(te) == 4 + 5
```

- [ ] **Step 2: Run tests to verify failure**

Run: `uv run pytest tests/test_lodo_split.py -v -k load_split_files`
Expected: FAIL — the LODO test cases fail because `load_split_files` only knows the mixed-pool branch. The non-LODO regression test passes because it exercises the unchanged code path.

- [ ] **Step 3: Add LODO branch to `load_split_files`**

In `src/data/dataset.py`, replace the existing `load_split_files` function (lines 119-136) with:

```python
def load_split_files(
    name: SplitName,
    spec: SplitSpec | None = None,
    graphs_dir: Path = GRAPHS_DIR,
) -> list[Path]:
    """Paths to window_*.pt for one split (no I/O).

    Two modes:
    - Mixed-pool (default, holdout_dataset is None): per-scenario chronological
      train_frac/val_frac/test_frac split of train_scenarios; test additionally
      includes the full window sequence of every holdout_test_scenario.
    - LODO (holdout_dataset is set): per-scenario chronological train_frac/val_frac
      split of train_scenarios (the dataset-X tail is unused — test comes from a
      different pool); test = full window sequence of every lodo_test_scenario.
      The holdout_test_scenarios field is ignored in LODO mode.
    """
    if spec is None:
        spec = SplitSpec.load()
    files: list[Path] = []
    if spec.holdout_dataset is None:
        # Mixed-pool: existing behavior, byte-identical.
        for sc in spec.train_scenarios:
            per_scenario = _chronological_split(
                _list_window_files(sc, graphs_dir), spec.train_frac, spec.val_frac
            )
            files.extend(per_scenario[name])
        if name == "test":
            for sc in spec.holdout_test_scenarios:
                files.extend(_list_window_files(sc, graphs_dir))
        return files

    # LODO mode.
    if name in ("train", "val"):
        for sc in spec.train_scenarios:
            per_scenario = _chronological_split(
                _list_window_files(sc, graphs_dir), spec.train_frac, spec.val_frac
            )
            files.extend(per_scenario[name])
        return files
    # name == "test"
    for sc in spec.lodo_test_scenarios:
        files.extend(_list_window_files(sc, graphs_dir))
    return files
```

The mixed-pool branch is byte-identical to the original. The LODO branch is purely additive.

- [ ] **Step 4: Run tests to verify pass**

Run: `uv run pytest tests/test_lodo_split.py -v`
Expected: PASS (7 tests).

- [ ] **Step 5: Run the full existing dataset test suite for regression**

Run: `uv run pytest tests/test_dataset.py -v`
Expected: PASS (7 existing tests).

- [ ] **Step 6: Commit**

```bash
cd higt-bot-code
git add src/data/dataset.py tests/test_lodo_split.py
git commit -m "feat(dataset): LODO branch in load_split_files

When SplitSpec.holdout_dataset is set, train/val use chronological
train_frac/val_frac of train_scenarios (no internal test) and test
draws the full window sequence of every lodo_test_scenarios entry.
The legacy holdout_test_scenarios field is ignored in LODO mode.
Mixed-pool branch byte-identical to previous behavior.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: Add `--split-config` to baseline scripts

**Files:**
- Modify: `scripts/baselines/run_rf.py`
- Modify: `scripts/baselines/run_xgb.py`
- Modify: `scripts/baselines/run_gnn.py`

These three scripts each call `SplitSpec.load()` with no path argument, falling back to `configs/split.yaml`. Add a single `--split-config` flag whose default keeps that exact behavior.

- [ ] **Step 1: Edit `scripts/baselines/run_rf.py`**

In `scripts/baselines/run_rf.py`:

Replace (around line 41-46):
```python
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-estimators", type=int, default=500)
    ap.add_argument("--max-depth", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=Path("data/inspection_logs/baseline_rf.json"))
    args = ap.parse_args()

    spec = SplitSpec.load()
```

With:
```python
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-estimators", type=int, default=500)
    ap.add_argument("--max-depth", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=Path("data/inspection_logs/baseline_rf.json"))
    ap.add_argument("--split-config", type=Path, default=Path("configs/split.yaml"),
                    help="path to split YAML; default matches the canonical mixed-pool split")
    args = ap.parse_args()

    spec = SplitSpec.load(args.split_config)
```

- [ ] **Step 2: Edit `scripts/baselines/run_xgb.py`**

Same change pattern. Replace (around line 39-48):
```python
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-estimators", type=int, default=500)
    ap.add_argument("--max-depth", type=int, default=6)
    ap.add_argument("--lr", type=float, default=0.1, dest="learning_rate")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=Path("data/inspection_logs/baseline_xgb.json"))
    args = ap.parse_args()

    spec = SplitSpec.load()
```

With:
```python
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-estimators", type=int, default=500)
    ap.add_argument("--max-depth", type=int, default=6)
    ap.add_argument("--lr", type=float, default=0.1, dest="learning_rate")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=Path("data/inspection_logs/baseline_xgb.json"))
    ap.add_argument("--split-config", type=Path, default=Path("configs/split.yaml"),
                    help="path to split YAML; default matches the canonical mixed-pool split")
    args = ap.parse_args()

    spec = SplitSpec.load(args.split_config)
```

- [ ] **Step 3: Edit `scripts/baselines/run_gnn.py`**

Replace (around lines 55-65):
```python
    ap.add_argument("--out", type=Path, default=None,
                    help="results json path; default data/inspection_logs/baseline_<model>.json")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device) if args.device else pick_device()
    print(f"device: {device}")

    spec = SplitSpec.load()
```

With:
```python
    ap.add_argument("--out", type=Path, default=None,
                    help="results json path; default data/inspection_logs/baseline_<model>.json")
    ap.add_argument("--split-config", type=Path, default=Path("configs/split.yaml"),
                    help="path to split YAML; default matches the canonical mixed-pool split")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device) if args.device else pick_device()
    print(f"device: {device}")

    spec = SplitSpec.load(args.split_config)
```

- [ ] **Step 4: Smoke check — each script still parses arguments and imports cleanly**

Run each in turn:

```powershell
uv run python scripts/baselines/run_rf.py --help
uv run python scripts/baselines/run_xgb.py --help
uv run python scripts/baselines/run_gnn.py --help
```

Expected: each prints help text including a `--split-config` line, exits 0.

- [ ] **Step 5: Commit**

```bash
cd higt-bot-code
git add scripts/baselines/run_rf.py scripts/baselines/run_xgb.py scripts/baselines/run_gnn.py
git commit -m "feat(baselines): add --split-config flag to RF/XGB/GNN scripts

Default matches the canonical configs/split.yaml so existing
reproducibility is preserved when called with no flags. Enables the
LODO experiment to point baseline scripts at different split YAMLs.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: Add `--split-config` to `cache_embeddings.py`

**Files:**
- Modify: `scripts/cache_embeddings.py:30-68`

`cache_embeddings.py` already has `--out-dir` and `--checkpoint`. It just needs `--split-config`.

- [ ] **Step 1: Edit `scripts/cache_embeddings.py`**

Replace (around lines 30-38):
```python
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path,
                    default=Path("experiments/phase5/temporal_gine_skip.pt"))
    ap.add_argument("--out-dir", type=Path, default=EMB_DIR)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--print-every", type=int, default=50)
    args = ap.parse_args()
```

With:
```python
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path,
                    default=Path("experiments/phase5/temporal_gine_skip.pt"))
    ap.add_argument("--out-dir", type=Path, default=EMB_DIR)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--print-every", type=int, default=50)
    ap.add_argument("--split-config", type=Path, default=Path("configs/split.yaml"),
                    help="path to split YAML; default matches canonical mixed-pool split")
    args = ap.parse_args()
```

Then replace (around line 65):
```python
    spec = SplitSpec.load()
```

With:
```python
    spec = SplitSpec.load(args.split_config)
```

- [ ] **Step 2: Smoke check**

```powershell
uv run python scripts/cache_embeddings.py --help
```

Expected: prints help including `--split-config`; exits 0.

- [ ] **Step 3: Commit**

```bash
cd higt-bot-code
git add scripts/cache_embeddings.py
git commit -m "feat(cache): add --split-config to cache_embeddings.py

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 7: Add flag triad to `train_phase5.py`

**Files:**
- Modify: `scripts/train_phase5.py:43-77, 155-177`

Add `--split-config`, `--ckpt-dir`, and `--results-dir`. Each defaults to the current hardcoded value so backward compat is byte-identical.

- [ ] **Step 1: Edit argparse block in `scripts/train_phase5.py`**

Replace (lines 43-67):
```python
CONFIG_PATH = Path("configs/phase5.yaml")
CKPT_DIR = Path("experiments/phase5")
RESULTS = Path("data/inspection_logs/phase5_temporal_gine.json")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=CONFIG_PATH)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--epochs", type=int, default=None,
                    help="override config max_epochs (useful for smoke tests)")
    ap.add_argument("--batch-size", type=int, default=None,
                    help="override config batch_size")
    ap.add_argument("--raw-feature-skip", action="store_true",
                    help="concat raw scaled node features into GIN input")
    ap.add_argument("--results-suffix", type=str, default="",
                    help="suffix for output files (e.g. '_skip')")
    ap.add_argument("--init-from-ssl", type=Path, default=None,
                    help="path to SSL-pretrained encoder checkpoint "
                         "(from scripts/pretrain_encoder.py); loads encoder_state "
                         "into model.encoder before training")
    args = ap.parse_args()
```

With:
```python
CONFIG_PATH = Path("configs/phase5.yaml")
CKPT_DIR = Path("experiments/phase5")
RESULTS = Path("data/inspection_logs/phase5_temporal_gine.json")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=CONFIG_PATH)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--epochs", type=int, default=None,
                    help="override config max_epochs (useful for smoke tests)")
    ap.add_argument("--batch-size", type=int, default=None,
                    help="override config batch_size")
    ap.add_argument("--raw-feature-skip", action="store_true",
                    help="concat raw scaled node features into GIN input")
    ap.add_argument("--results-suffix", type=str, default="",
                    help="suffix for output files (e.g. '_skip')")
    ap.add_argument("--init-from-ssl", type=Path, default=None,
                    help="path to SSL-pretrained encoder checkpoint "
                         "(from scripts/pretrain_encoder.py); loads encoder_state "
                         "into model.encoder before training")
    ap.add_argument("--split-config", type=Path, default=Path("configs/split.yaml"),
                    help="path to split YAML; default matches canonical mixed-pool split")
    ap.add_argument("--ckpt-dir", type=Path, default=CKPT_DIR,
                    help="checkpoint output directory; default experiments/phase5")
    ap.add_argument("--results-dir", type=Path, default=RESULTS.parent,
                    help="results JSON directory; default data/inspection_logs")
    args = ap.parse_args()
```

- [ ] **Step 2: Replace the `SplitSpec.load()` call**

Replace (around line 73):
```python
    spec = SplitSpec.load()
```

With:
```python
    spec = SplitSpec.load(args.split_config)
```

- [ ] **Step 3: Replace checkpoint and results path usage**

Replace (around lines 155-166):
```python
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt = CKPT_DIR / f"temporal_gine{args.results_suffix}.pt"
    save_blob = {"state_dict": model.state_dict(), "cfg": cfg,
                  "edge_scaler": (e_mean, e_std),
                  "raw_feat_dim": raw_feat_dim}
    if args.raw_feature_skip:
        save_blob["node_scaler"] = (n_mean, n_std)
    torch.save(save_blob, ckpt)
    print(f"saved {ckpt}")

    results_path = RESULTS.with_name(f"phase5_temporal_gine{args.results_suffix}.json")
    results_path.parent.mkdir(parents=True, exist_ok=True)
```

With:
```python
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt = args.ckpt_dir / f"temporal_gine{args.results_suffix}.pt"
    save_blob = {"state_dict": model.state_dict(), "cfg": cfg,
                  "edge_scaler": (e_mean, e_std),
                  "raw_feat_dim": raw_feat_dim}
    if args.raw_feature_skip:
        save_blob["node_scaler"] = (n_mean, n_std)
    torch.save(save_blob, ckpt)
    print(f"saved {ckpt}")

    results_path = args.results_dir / f"phase5_temporal_gine{args.results_suffix}.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
```

- [ ] **Step 4: Smoke check**

```powershell
uv run python scripts/train_phase5.py --help
```

Expected: prints help with `--split-config`, `--ckpt-dir`, `--results-dir`; exits 0.

- [ ] **Step 5: Commit**

```bash
cd higt-bot-code
git add scripts/train_phase5.py
git commit -m "feat(train): add split-config/ckpt-dir/results-dir flags to train_phase5

Defaults reproduce the previous hardcoded paths exactly.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 8: Add flag triad to `train_phase6_sparse.py`

**Files:**
- Modify: `scripts/train_phase6_sparse.py:43-94, 211-219`

- [ ] **Step 1: Edit argparse block in `scripts/train_phase6_sparse.py`**

Replace (lines 43-94):
```python
CKPT_DIR = Path("experiments/phase6")
RESULTS = Path("data/inspection_logs/phase6_sparse.json")
```

With:
```python
CKPT_DIR = Path("experiments/phase6")
RESULTS = Path("data/inspection_logs/phase6_sparse.json")
```

(unchanged) and then in the argparse block, after the existing `--emb-dir` line (around line 93), add:

```python
    ap.add_argument("--emb-dir", type=Path, default=Path("data/flow_embeddings"),
                    help="cached embedding directory")
    ap.add_argument("--split-config", type=Path, default=Path("configs/split.yaml"),
                    help="path to split YAML; default matches canonical mixed-pool split")
    ap.add_argument("--ckpt-dir", type=Path, default=CKPT_DIR,
                    help="checkpoint output directory; default experiments/phase6")
    ap.add_argument("--results-dir", type=Path, default=RESULTS.parent,
                    help="results JSON directory; default data/inspection_logs")
    args = ap.parse_args()
```

(replacing the existing `args = ap.parse_args()` line). The simplest way to specify this is: locate the existing line `ap.add_argument("--emb-dir", ...)` (around line 92) and insert the three new `add_argument` calls immediately after it.

- [ ] **Step 2: Replace `SplitSpec.load()` call**

Replace (around line 102):
```python
    spec = SplitSpec.load()
```

With:
```python
    spec = SplitSpec.load(args.split_config)
```

- [ ] **Step 3: Replace checkpoint and results paths**

Replace (around lines 211-219):
```python
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt = CKPT_DIR / f"hgt_sparse{args.results_suffix}.pt"
    torch.save({"state_dict": model.state_dict(), "cfg": cfg,
                 "args": vars(args), "edge_scaler": (e_mean, e_std),
                 "node_scaler": (n_mean, n_std)}, ckpt)
    print(f"saved {ckpt}")

    results_path = RESULTS.with_name(f"phase6_sparse{args.results_suffix}.json")
    results_path.parent.mkdir(parents=True, exist_ok=True)
```

With:
```python
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt = args.ckpt_dir / f"hgt_sparse{args.results_suffix}.pt"
    torch.save({"state_dict": model.state_dict(), "cfg": cfg,
                 "args": vars(args), "edge_scaler": (e_mean, e_std),
                 "node_scaler": (n_mean, n_std)}, ckpt)
    print(f"saved {ckpt}")

    results_path = args.results_dir / f"phase6_sparse{args.results_suffix}.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
```

- [ ] **Step 4: Smoke check**

```powershell
uv run python scripts/train_phase6_sparse.py --help
```

Expected: prints help with `--split-config`, `--ckpt-dir`, `--results-dir`; exits 0.

- [ ] **Step 5: Commit**

```bash
cd higt-bot-code
git add scripts/train_phase6_sparse.py
git commit -m "feat(train): add split-config/ckpt-dir/results-dir flags to train_phase6_sparse

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 9: Add flag triad to `train_phase7.py`

**Files:**
- Modify: `scripts/train_phase7.py:45-98, 216-224`

Mirror Task 8 exactly. The file structure of `train_phase7.py` is the same as `train_phase6_sparse.py`.

- [ ] **Step 1: Insert three new argparse lines after `--emb-dir`**

In `scripts/train_phase7.py`, locate this block (around lines 96-98):

```python
    ap.add_argument("--results-suffix", type=str, default="")
    ap.add_argument("--emb-dir", type=Path, default=Path("data/flow_embeddings"))
    args = ap.parse_args()
```

Replace with:

```python
    ap.add_argument("--results-suffix", type=str, default="")
    ap.add_argument("--emb-dir", type=Path, default=Path("data/flow_embeddings"))
    ap.add_argument("--split-config", type=Path, default=Path("configs/split.yaml"),
                    help="path to split YAML; default matches canonical mixed-pool split")
    ap.add_argument("--ckpt-dir", type=Path, default=CKPT_DIR,
                    help="checkpoint output directory; default experiments/phase7")
    ap.add_argument("--results-dir", type=Path, default=RESULTS.parent,
                    help="results JSON directory; default data/inspection_logs")
    args = ap.parse_args()
```

- [ ] **Step 2: Replace `SplitSpec.load()` call**

Replace (around line 108):
```python
    spec = SplitSpec.load()
```

With:
```python
    spec = SplitSpec.load(args.split_config)
```

- [ ] **Step 3: Replace checkpoint and results paths**

Replace (around lines 216-224):
```python
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt = CKPT_DIR / f"higt_bot{args.results_suffix}.pt"
    torch.save({"state_dict": model.state_dict(), "cfg": cfg,
                 "args": vars(args), "edge_scaler": (e_mean, e_std),
                 "node_scaler": (n_mean, n_std)}, ckpt)
    print(f"saved {ckpt}")

    results_path = RESULTS.with_name(f"phase7_higt_bot{args.results_suffix}.json")
    results_path.parent.mkdir(parents=True, exist_ok=True)
```

With:
```python
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt = args.ckpt_dir / f"higt_bot{args.results_suffix}.pt"
    torch.save({"state_dict": model.state_dict(), "cfg": cfg,
                 "args": vars(args), "edge_scaler": (e_mean, e_std),
                 "node_scaler": (n_mean, n_std)}, ckpt)
    print(f"saved {ckpt}")

    results_path = args.results_dir / f"phase7_higt_bot{args.results_suffix}.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
```

- [ ] **Step 4: Smoke check**

```powershell
uv run python scripts/train_phase7.py --help
```

Expected: prints help with `--split-config`, `--ckpt-dir`, `--results-dir`; exits 0.

- [ ] **Step 5: Commit**

```bash
cd higt-bot-code
git add scripts/train_phase7.py
git commit -m "feat(train): add split-config/ckpt-dir/results-dir flags to train_phase7

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 10: Regression run — verify mixed-pool defaults unchanged

**Files:**
- None (verification only)

A single mid-implementation gate: confirm all modified scripts still run from defaults without altering the canonical config's split assignment. We do this with a fast pytest assertion against `SplitSpec.load("configs/split.yaml")` because re-running training is too expensive; the path-default test is the byte-identical guarantee.

- [ ] **Step 1: Append regression test to `tests/test_lodo_split.py`**

```python
def test_default_split_config_loads_with_no_lodo() -> None:
    """Loading the canonical configs/split.yaml produces a mixed-pool spec.

    Regression guard: the additive code changes must not turn the canonical
    split into a LODO-mode spec by accident.
    """
    spec = SplitSpec.load(Path("configs/split.yaml"))
    assert spec.holdout_dataset is None
    assert spec.lodo_test_scenarios == []
    # Canonical split version 1, integer-typed.
    assert spec.version == 1
```

- [ ] **Step 2: Run**

```powershell
uv run pytest tests/test_lodo_split.py tests/test_dataset.py -v
```

Expected: all tests pass (8 LODO tests + 7 existing dataset tests = 15 total).

- [ ] **Step 3: Commit**

```bash
cd higt-bot-code
git add tests/test_lodo_split.py
git commit -m "test(lodo): regression guard for canonical split.yaml stays mixed-pool

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 11: Create LODO config YAMLs

**Files:**
- Create: `configs/split_lodo_ctu13.yaml`
- Create: `configs/split_lodo_iot23.yaml`
- Create: `configs/split_lodo_medbiot.yaml`

- [ ] **Step 1: Create `configs/split_lodo_ctu13.yaml`**

```yaml
# LODO split: hold out CTU-13 entirely.
# Train + val from IoT-23 + MedBIoT (per-scenario chronological 80/20).
# Test = all CTU-13 scenarios above the excluded threshold.

split_version: lodo-ctu13-v1
seed: 42
train_frac: 0.80
val_frac: 0.20

holdout_dataset: ctu13

train:
  - iot23-1-1
  - iot23-3-1
  - iot23-7-1
  - iot23-8-1
  - iot23-9-1
  - iot23-17-1
  - iot23-33-1
  - iot23-35-1
  - iot23-48-1
  - medbiot-bashlite_mal_spread_all
  - medbiot-bashlite_mal_CC_all
  - medbiot-mirai_mal_spread_all
  - medbiot-mirai_mal_CC_all
  - medbiot-torii_mal_all

lodo_test_scenarios:
  - ctu13-1
  - ctu13-2
  - ctu13-3
  - ctu13-4
  - ctu13-6
  - ctu13-8
  - ctu13-9
  - ctu13-10
  - ctu13-13

excluded:
  - ctu13-5
  - ctu13-7
  - ctu13-11
  - ctu13-12
  - iot23-20-1
  - iot23-21-1
```

- [ ] **Step 2: Create `configs/split_lodo_iot23.yaml`**

```yaml
# LODO split: hold out IoT-23 entirely.
# Train + val from CTU-13 + MedBIoT (per-scenario chronological 80/20).
# Test = all IoT-23 scenarios above the excluded threshold.

split_version: lodo-iot23-v1
seed: 42
train_frac: 0.80
val_frac: 0.20

holdout_dataset: iot23

train:
  - ctu13-1
  - ctu13-2
  - ctu13-3
  - ctu13-4
  - ctu13-6
  - ctu13-8
  - ctu13-9
  - ctu13-10
  - ctu13-13
  - medbiot-bashlite_mal_spread_all
  - medbiot-bashlite_mal_CC_all
  - medbiot-mirai_mal_spread_all
  - medbiot-mirai_mal_CC_all
  - medbiot-torii_mal_all

lodo_test_scenarios:
  - iot23-1-1
  - iot23-3-1
  - iot23-7-1
  - iot23-8-1
  - iot23-9-1
  - iot23-17-1
  - iot23-33-1
  - iot23-35-1
  - iot23-48-1

excluded:
  - ctu13-5
  - ctu13-7
  - ctu13-11
  - ctu13-12
  - iot23-20-1
  - iot23-21-1
```

- [ ] **Step 3: Create `configs/split_lodo_medbiot.yaml`**

```yaml
# LODO split: hold out MedBIoT entirely.
# Train + val from CTU-13 + IoT-23 (per-scenario chronological 80/20).
# Test = all MedBIoT malware scenarios.

split_version: lodo-medbiot-v1
seed: 42
train_frac: 0.80
val_frac: 0.20

holdout_dataset: medbiot

train:
  - ctu13-1
  - ctu13-2
  - ctu13-3
  - ctu13-4
  - ctu13-6
  - ctu13-8
  - ctu13-9
  - ctu13-10
  - ctu13-13
  - iot23-1-1
  - iot23-3-1
  - iot23-7-1
  - iot23-8-1
  - iot23-9-1
  - iot23-17-1
  - iot23-33-1
  - iot23-35-1
  - iot23-48-1

lodo_test_scenarios:
  - medbiot-bashlite_mal_spread_all
  - medbiot-bashlite_mal_CC_all
  - medbiot-mirai_mal_spread_all
  - medbiot-mirai_mal_CC_all
  - medbiot-torii_mal_all

excluded:
  - ctu13-5
  - ctu13-7
  - ctu13-11
  - ctu13-12
  - iot23-20-1
  - iot23-21-1
```

- [ ] **Step 4: Smoke check — every YAML parses cleanly**

```powershell
uv run python -c "from pathlib import Path; from src.data.dataset import SplitSpec; [print(p, SplitSpec.load(p).holdout_dataset, SplitSpec.load(p).lodo_test_scenarios[:2], '...') for p in [Path('configs/split_lodo_ctu13.yaml'), Path('configs/split_lodo_iot23.yaml'), Path('configs/split_lodo_medbiot.yaml')]]"
```

Expected: three lines printed, each showing the holdout_dataset value and the first two LODO test scenarios. No exceptions.

- [ ] **Step 5: Commit**

```bash
cd higt-bot-code
git add configs/split_lodo_ctu13.yaml configs/split_lodo_iot23.yaml configs/split_lodo_medbiot.yaml
git commit -m "feat(configs): add LODO split YAMLs for ctu13/iot23/medbiot

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 12: Build the LODO orchestrator skeleton

**Files:**
- Create: `scripts/run_lodo_ladder.py`

This is the largest single new file. Building it in two tasks: skeleton + stage definitions here, manifest/snapshot logic in Task 13.

- [ ] **Step 1: Create `scripts/run_lodo_ladder.py`**

```python
"""LODO orchestrator: runs the full Phase 4-7 ladder for one held-out dataset.

Per stage:
  1. Skip if --resume and <stage>.done marker exists.
  2. Capture pre-stage mod-time snapshot on the protected directories.
  3. Run the stage as a clean subprocess.
  4. Verify post-stage mod-times match (no accidental overwrite).
  5. Write <stage>.done marker with timestamp, exit code, command.

Stages:
  phase4    — RF, XGB, GAT, GIN, GINE (5 subprocesses)
  phase5    — Phase 5 encoder training (1 subprocess)
  cache     — Embedding cache for this LODO direction (1 subprocess)
  phase6    — SAGPool training (1 subprocess)
  phase7    — GT-edge training, one subprocess per seed

Usage:
    uv run python scripts/run_lodo_ladder.py --holdout medbiot
    uv run python scripts/run_lodo_ladder.py --holdout iot23 --resume
    uv run python scripts/run_lodo_ladder.py --holdout ctu13 --seeds 42 1 2
    uv run python scripts/run_lodo_ladder.py --holdout ctu13 --stages phase7
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

HOLDOUT_CHOICES = ("ctu13", "iot23", "medbiot")
DEFAULT_SEEDS = (42, 1, 2)
STAGES = ("phase4", "phase5", "cache", "phase6", "phase7")

# Directories whose mod-times must not change during a LODO run.
PROTECTED_DIRS = (
    Path("experiments/phase5"),
    Path("experiments/phase6"),
    Path("experiments/phase7"),
    Path("data/flow_embeddings"),
)


@dataclass
class LodoPaths:
    holdout: str
    split_config: Path
    ckpt_root: Path
    emb_dir: Path
    results_dir: Path
    manifest: Path
    log_dir: Path

    @classmethod
    def for_holdout(cls, holdout: str) -> "LodoPaths":
        ckpt_root = Path("experiments/lodo") / holdout
        return cls(
            holdout=holdout,
            split_config=Path(f"configs/split_lodo_{holdout}.yaml"),
            ckpt_root=ckpt_root,
            emb_dir=Path("data/flow_embeddings_lodo") / holdout,
            results_dir=Path("data/inspection_logs"),
            manifest=ckpt_root / "manifest.json",
            log_dir=ckpt_root / "logs",
        )


@dataclass
class StageResult:
    name: str
    cmd: list[str]
    rc: int
    started_iso: str
    finished_iso: str
    log_path: Path
    duration_s: float
    skipped: bool = False


def _utc_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _ensure_dirs(paths: LodoPaths) -> None:
    for d in (paths.ckpt_root, paths.emb_dir, paths.log_dir):
        d.mkdir(parents=True, exist_ok=True)
    for sub in ("phase4", "phase5", "phase6", "phase7"):
        (paths.ckpt_root / sub).mkdir(parents=True, exist_ok=True)


def _done_marker(paths: LodoPaths, stage: str, seed: int | None = None) -> Path:
    suffix = f"_seed{seed}" if seed is not None else ""
    return paths.ckpt_root / f".{stage}{suffix}.done"


def _result_suffix(paths: LodoPaths, seed: int | None = None) -> str:
    """Per-LODO results suffix that the existing scripts append to their output
    filenames. e.g., '_lodo_medbiot' or '_lodo_medbiot_seed1'."""
    base = f"_lodo_{paths.holdout}"
    return f"{base}_seed{seed}" if seed is not None else base


def stages_for_holdout(paths: LodoPaths, seeds: list[int]) -> list[tuple[str, list[str], int | None]]:
    """Return list of (stage_label, command, seed_or_None) tuples in execution order."""
    cmds: list[tuple[str, list[str], int | None]] = []
    suffix = _result_suffix(paths)
    p4 = paths.ckpt_root / "phase4"

    # Phase 4 — five subprocesses, each tagged with the LODO suffix in the --out filename.
    cmds.append(("phase4_rf", [
        sys.executable, "scripts/baselines/run_rf.py",
        "--split-config", str(paths.split_config),
        "--out", str(p4 / f"baseline_rf{suffix}.json"),
    ], None))
    cmds.append(("phase4_xgb", [
        sys.executable, "scripts/baselines/run_xgb.py",
        "--split-config", str(paths.split_config),
        "--out", str(p4 / f"baseline_xgb{suffix}.json"),
    ], None))
    for model in ("gat", "gin", "gine"):
        cmds.append((f"phase4_{model}", [
            sys.executable, "scripts/baselines/run_gnn.py",
            "--model", model,
            "--split-config", str(paths.split_config),
            "--out", str(p4 / f"baseline_{model}{suffix}.json"),
        ], None))

    # Phase 5 — encoder training (skip variant matches the final-model recipe).
    cmds.append(("phase5", [
        sys.executable, "scripts/train_phase5.py",
        "--raw-feature-skip",
        "--split-config", str(paths.split_config),
        "--ckpt-dir", str(paths.ckpt_root / "phase5"),
        "--results-dir", str(paths.results_dir),
        "--results-suffix", f"{suffix}_skip",
    ], None))

    # Cache — encode all windows under this LODO encoder.
    cmds.append(("cache", [
        sys.executable, "scripts/cache_embeddings.py",
        "--checkpoint", str(paths.ckpt_root / "phase5" / f"temporal_gine{suffix}_skip.pt"),
        "--split-config", str(paths.split_config),
        "--out-dir", str(paths.emb_dir),
    ], None))

    # Phase 6 — SAGPool over the LODO embeddings.
    cmds.append(("phase6", [
        sys.executable, "scripts/train_phase6_sparse.py",
        "--split-config", str(paths.split_config),
        "--ckpt-dir", str(paths.ckpt_root / "phase6"),
        "--results-dir", str(paths.results_dir),
        "--results-suffix", f"{suffix}_sparse",
        "--emb-dir", str(paths.emb_dir),
    ], None))

    # Phase 7 — GT-edge, once per seed.
    for seed in seeds:
        cmds.append((f"phase7_seed{seed}", [
            sys.executable, "scripts/train_phase7.py",
            "--gt-variant", "edge",
            "--seed", str(seed),
            "--split-config", str(paths.split_config),
            "--ckpt-dir", str(paths.ckpt_root / "phase7"),
            "--results-dir", str(paths.results_dir),
            "--results-suffix", f"{suffix}_edge_seed{seed}",
            "--emb-dir", str(paths.emb_dir),
        ], seed))

    return cmds


def run_stage(stage_label: str, cmd: list[str], paths: LodoPaths,
              resume: bool) -> StageResult:
    marker = paths.ckpt_root / f".{stage_label}.done"
    log_path = paths.log_dir / f"{stage_label}.log"

    if resume and marker.exists():
        return StageResult(
            name=stage_label, cmd=cmd, rc=0,
            started_iso=_utc_iso(), finished_iso=_utc_iso(),
            log_path=log_path, duration_s=0.0, skipped=True,
        )

    started = _utc_iso()
    t0 = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as lf:
        lf.write(f"# {stage_label}\n# started_utc={started}\n")
        lf.write(f"# cmd={' '.join(cmd)}\n\n")
        lf.flush()
        proc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT)
    dt = time.perf_counter() - t0
    finished = _utc_iso()

    if proc.returncode == 0:
        marker.write_text(
            f"started_utc={started}\nfinished_utc={finished}\n"
            f"duration_s={dt:.1f}\nrc=0\ncmd={' '.join(cmd)}\n",
            encoding="utf-8",
        )

    return StageResult(
        name=stage_label, cmd=cmd, rc=proc.returncode,
        started_iso=started, finished_iso=finished,
        log_path=log_path, duration_s=dt, skipped=False,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--holdout", choices=HOLDOUT_CHOICES, required=True)
    ap.add_argument("--resume", action="store_true",
                    help="skip stages whose .done marker already exists")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS),
                    help="seeds for Phase 7 GT-edge")
    ap.add_argument("--stages", type=str, default=",".join(STAGES),
                    help="comma-separated subset of: phase4,phase5,cache,phase6,phase7")
    args = ap.parse_args()

    paths = LodoPaths.for_holdout(args.holdout)
    _ensure_dirs(paths)

    requested = set(args.stages.split(","))
    invalid = requested - set(STAGES)
    if invalid:
        print(f"unknown stages: {invalid}", file=sys.stderr)
        return 2

    stage_plan = stages_for_holdout(paths, args.seeds)
    # Filter to requested stages — stage_label starts with one of phaseN/cache.
    plan = [(label, cmd, seed) for (label, cmd, seed) in stage_plan
             if label.split("_")[0] in requested
             or (label.startswith("cache") and "cache" in requested)]

    print(f"[lodo:{args.holdout}] planned stages: {[p[0] for p in plan]}",
          flush=True)

    results: list[StageResult] = []
    for label, cmd, seed in plan:
        print(f"[lodo:{args.holdout}] >>> {label}", flush=True)
        res = run_stage(label, cmd, paths, resume=args.resume)
        results.append(res)
        if res.skipped:
            print(f"[lodo:{args.holdout}]     skipped (resume marker present)",
                  flush=True)
            continue
        print(f"[lodo:{args.holdout}]     rc={res.rc}  {res.duration_s:.1f}s  "
              f"log={res.log_path}", flush=True)
        if res.rc != 0:
            print(f"[lodo:{args.holdout}] FAILED at {label}; see {res.log_path}",
                  file=sys.stderr, flush=True)
            return res.rc

    print(f"[lodo:{args.holdout}] all stages complete", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Smoke check argparse**

```powershell
uv run python scripts/run_lodo_ladder.py --help
```

Expected: prints usage including `--holdout`, `--resume`, `--seeds`, `--stages`; exits 0.

- [ ] **Step 3: Dry-run check that stage planning works**

```powershell
uv run python -c "from scripts.run_lodo_ladder import LodoPaths, stages_for_holdout; p = LodoPaths.for_holdout('medbiot'); [print(s[0], '->', ' '.join(s[1][:6]), '...') for s in stages_for_holdout(p, [42, 1, 2])]"
```

Expected: prints 11 lines (5 baselines + phase5 + cache + phase6 + 3 phase7 seeds), each with a sensible command preview.

- [ ] **Step 4: Commit**

```bash
cd higt-bot-code
git add scripts/run_lodo_ladder.py
git commit -m "feat(lodo): orchestrator skeleton with stage planning and subprocess exec

Per-stage subprocess execution with .done markers and per-stage logs.
Manifest writing and pre/post mod-time guard come in a follow-up task.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 13: Add non-overwrite guard and manifest to orchestrator

**Files:**
- Modify: `scripts/run_lodo_ladder.py`

- [ ] **Step 1: Add the snapshot, verification, and manifest functions**

In `scripts/run_lodo_ladder.py`, insert these helpers above `def run_stage(...)`:

```python
def snapshot_protected_dirs() -> dict[str, str]:
    """Return {file_path: lastWriteTimeUtc_iso} for every file under PROTECTED_DIRS.

    Used as a non-overwrite tripwire — if anything in here changes during a
    LODO run, we abort and report.
    """
    snap: dict[str, str] = {}
    for root in PROTECTED_DIRS:
        if not root.exists():
            continue
        for p in root.rglob("*"):
            if p.is_file():
                snap[str(p)] = time.strftime(
                    "%Y-%m-%dT%H:%M:%SZ", time.gmtime(p.stat().st_mtime)
                )
    return snap


def verify_no_overwrite(before: dict[str, str], after: dict[str, str],
                          stage_label: str) -> list[str]:
    """Return a list of human-readable mismatches; empty list = clean."""
    issues: list[str] = []
    # Files present before that changed mod-time.
    for fp, mt in before.items():
        if fp in after and after[fp] != mt:
            issues.append(f"MODIFIED during {stage_label}: {fp}  (was {mt}, now {after[fp]})")
    # New files inside protected dirs — also a violation.
    for fp in after:
        if fp not in before:
            issues.append(f"CREATED inside protected dir during {stage_label}: {fp}")
    # Disappeared files.
    for fp in before:
        if fp not in after:
            issues.append(f"DELETED during {stage_label}: {fp}")
    return issues


def write_manifest(paths: LodoPaths, results: list[StageResult],
                    seeds: list[int]) -> None:
    """Write experiments/lodo/<holdout>/manifest.json summarizing this LODO run."""
    payload = {
        "holdout": paths.holdout,
        "split_config": str(paths.split_config),
        "ckpt_root": str(paths.ckpt_root),
        "emb_dir": str(paths.emb_dir),
        "seeds": seeds,
        "stages": [
            {
                "name": r.name,
                "rc": r.rc,
                "duration_s": r.duration_s,
                "started_utc": r.started_iso,
                "finished_utc": r.finished_iso,
                "log": str(r.log_path),
                "skipped": r.skipped,
            }
            for r in results
        ],
        "completed_utc": _utc_iso(),
    }
    paths.manifest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
```

- [ ] **Step 2: Wire the snapshot check into `run_stage`**

Replace the existing `run_stage` function with:

```python
def run_stage(stage_label: str, cmd: list[str], paths: LodoPaths,
              resume: bool) -> StageResult:
    marker = paths.ckpt_root / f".{stage_label}.done"
    log_path = paths.log_dir / f"{stage_label}.log"

    if resume and marker.exists():
        return StageResult(
            name=stage_label, cmd=cmd, rc=0,
            started_iso=_utc_iso(), finished_iso=_utc_iso(),
            log_path=log_path, duration_s=0.0, skipped=True,
        )

    before = snapshot_protected_dirs()
    started = _utc_iso()
    t0 = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as lf:
        lf.write(f"# {stage_label}\n# started_utc={started}\n")
        lf.write(f"# cmd={' '.join(cmd)}\n\n")
        lf.flush()
        proc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT)
    dt = time.perf_counter() - t0
    finished = _utc_iso()

    # Non-overwrite tripwire. Run even if the subprocess failed; if it scribbled
    # on a protected dir before crashing we want to know.
    after = snapshot_protected_dirs()
    issues = verify_no_overwrite(before, after, stage_label)
    if issues:
        violations_log = paths.log_dir / f"{stage_label}.VIOLATIONS.txt"
        violations_log.write_text("\n".join(issues), encoding="utf-8")
        print(f"[lodo] ABORT: protected-dir violations in {stage_label}; "
              f"see {violations_log}", file=sys.stderr, flush=True)
        return StageResult(
            name=stage_label, cmd=cmd, rc=99,  # 99 = our non-overwrite sentinel
            started_iso=started, finished_iso=finished,
            log_path=log_path, duration_s=dt, skipped=False,
        )

    if proc.returncode == 0:
        marker.write_text(
            f"started_utc={started}\nfinished_utc={finished}\n"
            f"duration_s={dt:.1f}\nrc=0\ncmd={' '.join(cmd)}\n",
            encoding="utf-8",
        )

    return StageResult(
        name=stage_label, cmd=cmd, rc=proc.returncode,
        started_iso=started, finished_iso=finished,
        log_path=log_path, duration_s=dt, skipped=False,
    )
```

- [ ] **Step 3: Call `write_manifest` after the loop in `main()`**

In `main()`, replace the trailing:

```python
    print(f"[lodo:{args.holdout}] all stages complete", flush=True)
    return 0
```

With:

```python
    write_manifest(paths, results, args.seeds)
    print(f"[lodo:{args.holdout}] all stages complete; manifest at {paths.manifest}",
          flush=True)
    return 0
```

Also add `write_manifest` to the early-return on failure so partial manifests are still recorded:

Replace:
```python
        if res.rc != 0:
            print(f"[lodo:{args.holdout}] FAILED at {label}; see {res.log_path}",
                  file=sys.stderr, flush=True)
            return res.rc
```

With:
```python
        if res.rc != 0:
            print(f"[lodo:{args.holdout}] FAILED at {label}; see {res.log_path}",
                  file=sys.stderr, flush=True)
            write_manifest(paths, results, args.seeds)
            return res.rc
```

- [ ] **Step 4: Smoke check**

```powershell
uv run python scripts/run_lodo_ladder.py --help
```

Expected: still parses; exits 0.

- [ ] **Step 5: Commit**

```bash
cd higt-bot-code
git add scripts/run_lodo_ladder.py
git commit -m "feat(lodo): non-overwrite guard, per-stage logs, manifest

Pre/post mod-time snapshot of experiments/phase{5,6,7}/ and
data/flow_embeddings/ around every stage. Any modification triggers a
hard abort with a VIOLATIONS file. Manifest writes summarize seeds,
stage statuses, and log paths.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 14: Unit tests for the orchestrator

**Files:**
- Create: `tests/test_run_lodo_ladder.py`

- [ ] **Step 1: Write tests with mocked subprocess**

Create `tests/test_run_lodo_ladder.py`:

```python
"""Unit tests for scripts/run_lodo_ladder.py.

Subprocess execution is mocked — we test the orchestrator's stage planning,
non-overwrite guard, and manifest writing, not the underlying trainers.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from scripts.run_lodo_ladder import (
    LodoPaths,
    StageResult,
    snapshot_protected_dirs,
    stages_for_holdout,
    verify_no_overwrite,
    write_manifest,
)


def test_lodo_paths_for_each_holdout() -> None:
    for h in ("ctu13", "iot23", "medbiot"):
        p = LodoPaths.for_holdout(h)
        assert p.holdout == h
        assert p.split_config == Path(f"configs/split_lodo_{h}.yaml")
        assert p.ckpt_root == Path("experiments/lodo") / h
        assert p.emb_dir == Path("data/flow_embeddings_lodo") / h
        assert p.manifest == Path("experiments/lodo") / h / "manifest.json"


def test_stages_for_holdout_count_matches_ladder() -> None:
    """Three seeds means: 5 baselines + phase5 + cache + phase6 + 3 phase7 seeds = 11 stages."""
    p = LodoPaths.for_holdout("medbiot")
    plan = stages_for_holdout(p, seeds=[42, 1, 2])
    labels = [s[0] for s in plan]
    assert len(plan) == 11
    assert "phase4_rf" in labels
    assert "phase4_xgb" in labels
    assert "phase4_gat" in labels
    assert "phase4_gin" in labels
    assert "phase4_gine" in labels
    assert "phase5" in labels
    assert "cache" in labels
    assert "phase6" in labels
    assert "phase7_seed42" in labels
    assert "phase7_seed1" in labels
    assert "phase7_seed2" in labels


def test_stages_for_holdout_uses_holdout_split_config() -> None:
    p = LodoPaths.for_holdout("iot23")
    plan = stages_for_holdout(p, seeds=[42])
    for label, cmd, seed in plan:
        if "phase4" in label or "phase5" in label or "phase6" in label or "phase7" in label or "cache" in label:
            assert "configs/split_lodo_iot23.yaml" in " ".join(cmd), \
                f"stage {label} should reference iot23 split config"


def test_verify_no_overwrite_returns_empty_when_unchanged() -> None:
    before = {"a": "2026-01-01T00:00:00Z", "b": "2026-01-02T00:00:00Z"}
    after = dict(before)
    assert verify_no_overwrite(before, after, "phase5") == []


def test_verify_no_overwrite_flags_modifications() -> None:
    before = {"a": "2026-01-01T00:00:00Z"}
    after = {"a": "2026-06-03T18:00:00Z"}
    issues = verify_no_overwrite(before, after, "phase5")
    assert len(issues) == 1
    assert "MODIFIED" in issues[0]


def test_verify_no_overwrite_flags_new_files() -> None:
    before: dict[str, str] = {}
    after = {"x": "2026-06-03T18:00:00Z"}
    issues = verify_no_overwrite(before, after, "cache")
    assert any("CREATED" in i for i in issues)


def test_verify_no_overwrite_flags_deletions() -> None:
    before = {"removed": "2026-01-01T00:00:00Z"}
    after: dict[str, str] = {}
    issues = verify_no_overwrite(before, after, "phase6")
    assert any("DELETED" in i for i in issues)


def test_write_manifest_produces_valid_json(tmp_path: Path) -> None:
    paths = LodoPaths(
        holdout="medbiot",
        split_config=Path("configs/split_lodo_medbiot.yaml"),
        ckpt_root=tmp_path,
        emb_dir=tmp_path / "emb",
        results_dir=tmp_path / "results",
        manifest=tmp_path / "manifest.json",
        log_dir=tmp_path / "logs",
    )
    results = [
        StageResult(name="phase4_rf", cmd=["uv", "run", "rf"],
                     rc=0, started_iso="t0", finished_iso="t1",
                     log_path=tmp_path / "logs/phase4_rf.log", duration_s=12.3),
    ]
    write_manifest(paths, results, seeds=[42, 1, 2])

    assert paths.manifest.exists()
    blob = json.loads(paths.manifest.read_text())
    assert blob["holdout"] == "medbiot"
    assert blob["seeds"] == [42, 1, 2]
    assert len(blob["stages"]) == 1
    assert blob["stages"][0]["name"] == "phase4_rf"
    assert blob["stages"][0]["rc"] == 0
```

- [ ] **Step 2: Run tests**

```powershell
uv run pytest tests/test_run_lodo_ladder.py -v
```

Expected: PASS (8 tests).

- [ ] **Step 3: Commit**

```bash
cd higt-bot-code
git add tests/test_run_lodo_ladder.py
git commit -m "test(lodo): orchestrator unit tests for planning + guard + manifest

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 15: Smoke run LODO-MedBIoT end-to-end

**Files:**
- None (execution + verification only)

MedBIoT is the smallest LODO test set (5 scenarios, mostly small parquet files). It exposes any integration issues fastest. Two-stage approach: dry-run check of split assignment first, then real run.

- [ ] **Step 1: Split-assignment dry check**

```powershell
uv run python -c "from src.data.dataset import SplitSpec, load_split_files; from pathlib import Path; spec = SplitSpec.load(Path('configs/split_lodo_medbiot.yaml')); tr = load_split_files('train', spec); va = load_split_files('val', spec); te = load_split_files('test', spec); print(f'train_files={len(tr)} val_files={len(va)} test_files={len(te)}'); print('test scenarios:', sorted(set(p.parent.name for p in te)))"
```

Expected: prints non-zero train/val/test counts; the test scenarios list contains exactly the 5 medbiot entries from `configs/split_lodo_medbiot.yaml`, none from CTU-13 or IoT-23.

- [ ] **Step 2: Run the LODO ladder**

```powershell
uv run python scripts/run_lodo_ladder.py --holdout medbiot
```

This is the long-running stage (~2-3 hours wall-clock on RTX 5080). Monitor `experiments/lodo/medbiot/logs/` for per-stage progress.

Expected on success:
- All 11 `.done` markers present in `experiments/lodo/medbiot/`
- `experiments/lodo/medbiot/manifest.json` written, all stages `rc=0`
- New result JSONs in `data/inspection_logs/` with `_lodo_medbiot` suffix
- `data/flow_embeddings_lodo/medbiot/` populated
- `experiments/lodo/medbiot/phase{4,5,6,7}/` populated with the new artifacts

- [ ] **Step 3: Verify non-overwrite of existing directories**

```powershell
Get-ChildItem -Path experiments/phase5, experiments/phase6, experiments/phase7, data/flow_embeddings -Recurse -File -ErrorAction SilentlyContinue |
  ForEach-Object { "{0}`t{1}" -f $_.FullName, $_.LastWriteTimeUtc.ToString("o") } |
  Set-Content -Encoding utf8 experiments/lodo/.post_medbiot_snapshot.txt

Compare-Object (Get-Content experiments/lodo/.pre_run_snapshot.txt) (Get-Content experiments/lodo/.post_medbiot_snapshot.txt) |
  Tee-Object -Variable diff
if ($diff) { Write-Error "Protected dirs changed during LODO-MedBIoT run." } else { Write-Host "Protected dirs intact." }
```

Expected: "Protected dirs intact." printed; no errors. If the orchestrator already aborted on a violation, it would have done so during Step 2 — this is a final defense-in-depth check.

- [ ] **Step 4: Commit the smoke-run artifacts**

```bash
cd higt-bot-code
git add experiments/lodo/medbiot/manifest.json experiments/lodo/.post_medbiot_snapshot.txt data/inspection_logs/*lodo_medbiot*
git commit -m "exp(lodo): LODO-MedBIoT smoke run complete

Manifest, result JSONs, and post-run snapshot from the smallest LODO
direction. Used as integration test before running the larger
LODO-CTU13 and LODO-IoT23 directions.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 16: Run LODO-CTU13 and LODO-IoT23

**Files:**
- None (execution + verification only)

- [ ] **Step 1: Run LODO-CTU13**

```powershell
uv run python scripts/run_lodo_ladder.py --holdout ctu13
```

Expected on success: 11 `.done` markers, `manifest.json` written.

- [ ] **Step 2: Run LODO-IoT23**

```powershell
uv run python scripts/run_lodo_ladder.py --holdout iot23
```

Expected on success: 11 `.done` markers, `manifest.json` written. IoT-23 has the largest test scenarios (iot23-39-1, iot23-43-1) but they're not in the LODO test list (they're excluded or not used in the canonical split); inspect.py handles them via streaming if encountered.

- [ ] **Step 3: Final non-overwrite verification**

```powershell
Get-ChildItem -Path experiments/phase5, experiments/phase6, experiments/phase7, data/flow_embeddings -Recurse -File -ErrorAction SilentlyContinue |
  ForEach-Object { "{0}`t{1}" -f $_.FullName, $_.LastWriteTimeUtc.ToString("o") } |
  Set-Content -Encoding utf8 experiments/lodo/.post_all_snapshot.txt

Compare-Object (Get-Content experiments/lodo/.pre_run_snapshot.txt) (Get-Content experiments/lodo/.post_all_snapshot.txt) |
  Tee-Object -Variable diff
if ($diff) { Write-Error "Protected dirs changed during LODO runs." } else { Write-Host "All protected dirs intact." }
```

Expected: "All protected dirs intact."

- [ ] **Step 4: Commit**

```bash
cd higt-bot-code
git add experiments/lodo/ctu13/manifest.json experiments/lodo/iot23/manifest.json experiments/lodo/.post_all_snapshot.txt data/inspection_logs/*lodo_ctu13* data/inspection_logs/*lodo_iot23*
git commit -m "exp(lodo): LODO-CTU13 and LODO-IoT23 runs complete

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 17: Aggregation script and tables

**Files:**
- Create: `scripts/aggregate_lodo_results.py`

Mirrors the existing `scripts/phase8_ablation_table.py` style — slurps every `data/inspection_logs/*lodo_*.json` and writes markdown tables.

- [ ] **Step 1: Create `scripts/aggregate_lodo_results.py`**

```python
"""Aggregate LODO result JSONs into comparison tables and a per-scenario breakdown.

Reads every data/inspection_logs/*lodo_*.json and writes:
  - data/inspection_logs/lodo_comparison_table.md
  - data/inspection_logs/lodo_per_scenario_breakdown.md
  - data/inspection_logs/lodo_{ctu13,iot23,medbiot}_summary.md
  - One row per (model, LODO direction) with F1, PR-AUC, FN/FP, ROC-AUC.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

LOGS = Path("data/inspection_logs")
HOLDOUTS = ("ctu13", "iot23", "medbiot")
MODEL_ORDER = (
    "baseline_rf", "baseline_xgb", "baseline_gat", "baseline_gin", "baseline_gine",
    "phase5_temporal_gine", "phase6_sparse", "phase7_higt_bot",
)


def _model_pretty(name: str) -> str:
    return {
        "baseline_rf": "RandomForest",
        "baseline_xgb": "XGBoost",
        "baseline_gat": "GAT (flat)",
        "baseline_gin": "GIN (flat)",
        "baseline_gine": "GINE (flat)",
        "phase5_temporal_gine": "T-GINE + skip",
        "phase6_sparse": "SAGPool (sparse)",
        "phase7_higt_bot": "GT-edge (final)",
    }.get(name, name)


def _load_result(path: Path) -> dict:
    return json.loads(path.read_text())


def collect_by_holdout() -> dict[str, dict[str, list[dict]]]:
    """Return {holdout: {model_root: [result_dicts]}}.

    A model can appear multiple times when seed replicates exist
    (only Phase 7 GT-edge does this).
    """
    out: dict[str, dict[str, list[dict]]] = {
        h: defaultdict(list) for h in HOLDOUTS
    }
    for jf in LOGS.glob("*_lodo_*.json"):
        # Filename convention: <model_root>_lodo_<holdout>[_<extras>].json
        stem = jf.stem
        # Find which holdout this file is for.
        match = next((h for h in HOLDOUTS if f"_lodo_{h}" in stem), None)
        if match is None:
            continue
        head = stem.split(f"_lodo_{match}")[0]
        out[match][head].append(_load_result(jf))
    return out


def aggregate_metric(rows: list[dict], path: str) -> tuple[float, float | None]:
    """Pull `rows[i]['test'][path]` (path uses dot notation). Return (mean, std-or-None)."""
    vals = []
    for r in rows:
        cur = r.get("test", {})
        for key in path.split("."):
            cur = cur.get(key, {}) if isinstance(cur, dict) else cur
        if isinstance(cur, (int, float)):
            vals.append(float(cur))
    if not vals:
        return float("nan"), None
    if len(vals) == 1:
        return vals[0], None
    return mean(vals), stdev(vals)


def write_comparison_table(by_holdout: dict[str, dict[str, list[dict]]]) -> Path:
    lines = ["# LODO comparison table",
              "",
              "Test F1 (mean ± std across seeds where applicable). Mixed-pool column "
              "from existing `phase{4..7}_*.json`.",
              "",
              "| Model | LODO-CTU13 | LODO-IoT23 | LODO-MedBIoT |",
              "|---|---|---|---|"]
    for model_root in MODEL_ORDER:
        row = [_model_pretty(model_root)]
        for h in HOLDOUTS:
            rows = by_holdout[h].get(model_root, [])
            if not rows:
                row.append("—")
                continue
            m, s = aggregate_metric(rows, "f1")
            row.append(f"{m:.4f}" + (f" ± {s:.4f}" if s is not None else ""))
        lines.append("| " + " | ".join(row) + " |")
    out = LOGS / "lodo_comparison_table.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def write_per_scenario_breakdown(by_holdout: dict[str, dict[str, list[dict]]]) -> Path:
    lines = ["# LODO per-scenario test F1",
              "",
              "GT-edge results only (the headline model). Cells = F1 (mean across seeds).",
              ""]
    for h in HOLDOUTS:
        rows = by_holdout[h].get("phase7_higt_bot", [])
        if not rows:
            continue
        lines.append(f"## LODO-{h.upper()}")
        lines.append("")
        # Average per_scenario across seeds.
        per_scen: dict[str, list[float]] = defaultdict(list)
        for r in rows:
            for scen, m in r.get("test_per_scenario", {}).items():
                f1 = m.get("f1") if isinstance(m, dict) else None
                if isinstance(f1, (int, float)):
                    per_scen[scen].append(float(f1))
        if not per_scen:
            lines.append("_(no per-scenario data)_")
            lines.append("")
            continue
        lines.append("| Scenario | F1 (mean) | seeds |")
        lines.append("|---|---|---|")
        for scen in sorted(per_scen):
            vals = per_scen[scen]
            lines.append(f"| {scen} | {mean(vals):.4f} | {len(vals)} |")
        lines.append("")
    out = LOGS / "lodo_per_scenario_breakdown.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def write_per_holdout_summaries(by_holdout: dict[str, dict[str, list[dict]]]) -> list[Path]:
    paths = []
    for h in HOLDOUTS:
        models = by_holdout[h]
        lines = [f"# LODO-{h.upper()} summary",
                  "",
                  f"Held-out dataset: **{h}**. All models trained from scratch on "
                  f"the two remaining datasets using `configs/split_lodo_{h}.yaml`.",
                  "",
                  "| Model | F1 | PR-AUC | ROC-AUC |",
                  "|---|---|---|---|"]
        for model_root in MODEL_ORDER:
            rows = models.get(model_root, [])
            if not rows:
                lines.append(f"| {_model_pretty(model_root)} | — | — | — |")
                continue
            f1, _ = aggregate_metric(rows, "f1")
            pr, _ = aggregate_metric(rows, "pr_auc")
            roc, _ = aggregate_metric(rows, "roc_auc")
            lines.append(f"| {_model_pretty(model_root)} | {f1:.4f} | {pr:.4f} | {roc:.4f} |")
        out = LOGS / f"lodo_{h}_summary.md"
        out.write_text("\n".join(lines) + "\n", encoding="utf-8")
        paths.append(out)
    return paths


def main() -> None:
    by_holdout = collect_by_holdout()
    p1 = write_comparison_table(by_holdout)
    p2 = write_per_scenario_breakdown(by_holdout)
    ps = write_per_holdout_summaries(by_holdout)
    print(f"wrote {p1}")
    print(f"wrote {p2}")
    for p in ps:
        print(f"wrote {p}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run aggregation**

```powershell
uv run python scripts/aggregate_lodo_results.py
```

Expected: prints 5 "wrote ..." lines. The five generated markdown files exist in `data/inspection_logs/`.

- [ ] **Step 3: Sanity-check the comparison table**

```powershell
Get-Content data/inspection_logs/lodo_comparison_table.md
```

Expected: 8 model rows × 3 LODO columns; no empty cells unless a model failed to produce output. GT-edge row shows ` ± std`.

- [ ] **Step 4: Commit**

```bash
cd higt-bot-code
git add scripts/aggregate_lodo_results.py data/inspection_logs/lodo_comparison_table.md data/inspection_logs/lodo_per_scenario_breakdown.md data/inspection_logs/lodo_ctu13_summary.md data/inspection_logs/lodo_iot23_summary.md data/inspection_logs/lodo_medbiot_summary.md
git commit -m "feat(lodo): aggregation script and result tables

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 18: Write `lodo_master.md` narrative

**Files:**
- Create: `data/inspection_logs/lodo_master.md`

This file is hand-authored after results are in. It mirrors the structure of `phase8_master.md`.

- [ ] **Step 1: Build the narrative from the aggregated tables**

This is the one human-authored step in the plan. Open the aggregated tables (`lodo_comparison_table.md`, `lodo_per_scenario_breakdown.md`, `lodo_{x}_summary.md`) and the existing `data/inspection_logs/phase7_higt_bot_edge.json` for the mixed-pool number. Fill the skeleton below with the actual numbers from those files — the `<number ± std>` and `<paragraph>` placeholders below are intentional and must be replaced before commit. Compose `data/inspection_logs/lodo_master.md` with these sections:

1. **Headline finding (3 sentences)** — what does GT-edge actually do under LODO vs mixed-pool? Quote the numbers.
2. **Per-LODO direction** — for each of CTU-13, IoT-23, MedBIoT, one paragraph: best-case scenario in test, worst-case, what the failure pattern is.
3. **Comparison vs mixed-pool baseline** — table copied from `lodo_comparison_table.md` with mixed-pool column added by hand.
4. **What the asymmetry says about the model** — interpretive paragraph. Is the dataset-transfer monotonic (every direction loses by similar amount) or asymmetric (one direction tanks)?
5. **Threats to validity** — list what this LODO experiment does *not* show (e.g., transfer across capture instruments, transfer over time within a dataset, transfer with hyperparameter retuning).
6. **Reproduction recipe** — three commands plus expected wall-time. Reference the manifests under `experiments/lodo/<holdout>/manifest.json`.

Template skeleton:

```markdown
# LODO master narrative

## Headline

GT-edge mixed-pool baseline: F1 = 0.9677 (Phase 7).
Under leave-one-dataset-out:
- LODO-CTU13:   F1 = <number ± std> on held-out CTU-13 scenarios
- LODO-IoT23:   F1 = <number ± std> on held-out IoT-23 scenarios
- LODO-MedBIoT: F1 = <number ± std> on held-out MedBIoT scenarios

<one-sentence framing of what this means>

## Per-LODO direction

### LODO-CTU13
<one paragraph: train pool, test pool, headline number, best/worst scenario>

### LODO-IoT23
<same>

### LODO-MedBIoT
<same>

## Comparison vs mixed-pool baseline

<paste lodo_comparison_table.md content, then add the mixed-pool column
sourced from data/inspection_logs/phase{4..7}_*.json>

## What the asymmetry says

<interpretive paragraph>

## Threats to validity

- No hyperparameter retuning per LODO (intentional — see spec section 3.4).
- Excluded scenarios stay excluded (consistent with Phase 4-7 protocol).
- Test set class-imbalance shifts across LODO directions; PR-AUC is the
  prior-robust metric and is reported alongside F1.
- This experiment does not test cross-time transfer beyond what the
  internal 80/20 chronological split already tests.

## Reproduction

```bash
uv run python scripts/run_lodo_ladder.py --holdout medbiot
uv run python scripts/run_lodo_ladder.py --holdout ctu13
uv run python scripts/run_lodo_ladder.py --holdout iot23
uv run python scripts/aggregate_lodo_results.py
```

Wall-time on RTX 5080: ~2-3 hours per LODO; ~6-10 hours total.
Manifests at `experiments/lodo/<holdout>/manifest.json`.
```

- [ ] **Step 2: Commit**

```bash
cd higt-bot-code
git add data/inspection_logs/lodo_master.md
git commit -m "docs(lodo): master narrative summarizing all three LODO directions

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 19: Final acceptance verification

**Files:**
- None (verification only)

Acceptance criteria from spec section 13.

- [ ] **Step 1: Confirm all `.done` markers exist**

```powershell
$missing = @()
foreach ($h in 'ctu13', 'iot23', 'medbiot') {
    $root = "experiments/lodo/$h"
    foreach ($stage in 'phase4_rf', 'phase4_xgb', 'phase4_gat', 'phase4_gin', 'phase4_gine', 'phase5', 'cache', 'phase6', 'phase7_seed42', 'phase7_seed1', 'phase7_seed2') {
        $marker = Join-Path $root ".$stage.done"
        if (-not (Test-Path $marker)) { $missing += $marker }
    }
}
if ($missing) { Write-Error "Missing markers: $missing" } else { Write-Host "All 33 stage markers present (11 per LODO x 3 LODOs)." }
```

Expected: "All 33 stage markers present."

- [ ] **Step 2: Confirm all manifests are present and rc=0**

```powershell
foreach ($h in 'ctu13', 'iot23', 'medbiot') {
    $m = Get-Content "experiments/lodo/$h/manifest.json" -Raw | ConvertFrom-Json
    $fail = $m.stages | Where-Object { $_.rc -ne 0 }
    if ($fail) { Write-Error "Failed stages in $h LODO: $fail" } else { Write-Host "$h manifest: all rc=0" }
}
```

Expected: three "manifest: all rc=0" lines.

- [ ] **Step 3: Final protected-dir intactness check**

```powershell
Get-ChildItem -Path experiments/phase5, experiments/phase6, experiments/phase7, data/flow_embeddings -Recurse -File -ErrorAction SilentlyContinue |
  ForEach-Object { "{0}`t{1}" -f $_.FullName, $_.LastWriteTimeUtc.ToString("o") } |
  Set-Content -Encoding utf8 experiments/lodo/.final_snapshot.txt
Compare-Object (Get-Content experiments/lodo/.pre_run_snapshot.txt) (Get-Content experiments/lodo/.final_snapshot.txt) |
  Tee-Object -Variable diff
if ($diff) { Write-Error "ACCEPTANCE FAIL: existing experiments were modified." } else { Write-Host "ACCEPTANCE PASS: existing experiments untouched." }
```

Expected: "ACCEPTANCE PASS".

- [ ] **Step 4: Confirm dataset.py tests still pass**

```powershell
uv run pytest tests/test_dataset.py tests/test_lodo_split.py tests/test_run_lodo_ladder.py -v
```

Expected: all tests pass.

- [ ] **Step 5: Final commit**

```bash
cd higt-bot-code
git add experiments/lodo/.final_snapshot.txt
git commit -m "exp(lodo): final acceptance snapshot

All 33 stage markers present, three manifests written rc=0 throughout,
existing experiments/phase{5,6,7}/ and data/flow_embeddings/ intact.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Summary of commits (target order)

1. `chore(lodo): pre-run snapshot and design/plan docs`
2. `feat(dataset): add optional LODO fields to SplitSpec`
3. `feat(dataset): SplitSpec.load parses LODO YAML keys`
4. `feat(dataset): LODO branch in load_split_files`
5. `feat(baselines): add --split-config flag to RF/XGB/GNN scripts`
6. `feat(cache): add --split-config to cache_embeddings.py`
7. `feat(train): add split-config/ckpt-dir/results-dir flags to train_phase5`
8. `feat(train): add split-config/ckpt-dir/results-dir flags to train_phase6_sparse`
9. `feat(train): add split-config/ckpt-dir/results-dir flags to train_phase7`
10. `test(lodo): regression guard for canonical split.yaml stays mixed-pool`
11. `feat(configs): add LODO split YAMLs for ctu13/iot23/medbiot`
12. `feat(lodo): orchestrator skeleton with stage planning and subprocess exec`
13. `feat(lodo): non-overwrite guard, per-stage logs, manifest`
14. `test(lodo): orchestrator unit tests for planning + guard + manifest`
15. `exp(lodo): LODO-MedBIoT smoke run complete`
16. `exp(lodo): LODO-CTU13 and LODO-IoT23 runs complete`
17. `feat(lodo): aggregation script and result tables`
18. `docs(lodo): master narrative summarizing all three LODO directions`
19. `exp(lodo): final acceptance snapshot`
