# Cross-Dataset (Leave-One-Dataset-Out) Generalization Experiment

**Status:** Design — awaiting implementation plan
**Author:** Generated via brainstorming session, 2026-06-03
**Target:** `higt-bot-code/` thesis repo, post-Phase 8 addendum

## 1. Motivation

The existing HiGT-Bot evaluation mixes CTU-13, IoT-23, and MedBIoT in a single training pool and applies per-scenario chronological splits (60/20/20) plus three held-out scenarios. This demonstrates *intra-dataset forward-in-time generalization* and *held-out-scenario generalization within a mixed pool* — not "train on one dataset family, test on another."

The thesis implementation plan flagged cross-dataset (leave-one-dataset-out, "LODO") evaluation as the reviewer-pleasing result but it was never run. This spec defines an addendum experiment that runs it honestly without disturbing the existing finalized pipeline or its results.

## 2. Goal and success criteria

**Primary goal:** Produce a comparison table that quantifies how the full Phase 4–7 model ladder transfers across dataset families, with the GT-edge final model as the headline row.

**Success criteria:**
- All three LODO directions complete (hold out CTU-13, hold out IoT-23, hold out MedBIoT).
- Per-scenario test metrics (F1, PR-AUC, ROC-AUC, FN, FP) reported for every model in the ladder, every LODO direction.
- GT-edge headline reported as F1 ± std across three seeds per LODO direction.
- Zero modification to existing results: no overwrites of `experiments/phase{5,6,7}/`, `data/flow_embeddings/`, or `data/inspection_logs/phase{4..8}_*`. Verified by pre/post mod-time snapshot.
- One additional artifact, `lodo_master.md`, mirroring `phase8_master.md` style with the overall narrative.

The experiment is a contribution regardless of result direction:
- *Strong positive:* "Architecture transfers across capture environments" — headline upgrade.
- *Uniform degradation:* "Joint training is necessary to cover heterogeneity; LODO degradation quantifies the cost" — strengthens existing framing.
- *Asymmetric degradation:* The asymmetry itself is the finding (e.g., enterprise→IoT vs IoT→enterprise).

## 3. Experimental design

### 3.1 LODO directions

Three independent training runs, each from scratch:

| Direction | Train datasets | Test dataset |
|---|---|---|
| LODO-CTU13 | IoT-23 + MedBIoT | CTU-13 |
| LODO-IoT23 | CTU-13 + MedBIoT | IoT-23 |
| LODO-MedBIoT | CTU-13 + IoT-23 | MedBIoT |

### 3.2 Models per LODO direction

The full Phase 4–7 ladder, one trained model per cell:

| Phase | Models |
|---|---|
| 4 (flat baselines) | RandomForest, XGBoost, GAT, GIN, GINE |
| 5 (temporal encoder) | TemporalGINE + skip variant |
| 6 (hierarchical pooling) | SAGPool (sparse) |
| 7 (final) | GT-edge |

**Eight models × three LODO directions = 24 trained models** for the ladder, plus the GT-edge seed replicates below.

### 3.3 Seeds

- **Ladder:** single seed (42) per model, per LODO direction.
- **GT-edge headline:** seeds {42, 1, 2} per LODO direction, F1 ± std reported (matches Phase 7 seed protocol).

Total: 24 ladder runs + 6 extra GT-edge seeds = **30 trained models** across the experiment.

### 3.4 Hyperparameters

**No retuning.** All models use the existing Phase 4–7 final hyperparameters verbatim. This is a deliberate methodological choice: retuning would confound "architecture transfers" with "tuning was lucky for the joint pool," and the headline reading reviewers want is apples-to-apples. Reuse without retuning is the standard LODO protocol.

### 3.5 Metrics

Reported per scenario in the LODO test set, aggregated to a LODO-level number, and rolled up across LODO directions:

- F1 (binary node-level, threshold = 0.5)
- PR-AUC (primary metric under class imbalance)
- ROC-AUC
- False negatives, false positives (absolute counts)
- Confusion matrix per test scenario
- Per-malware-family breakdown where labels permit

## 4. Data split semantics (LODO mode)

Given a LODO direction X ∈ {ctu13, iot23, medbiot}:

| Role | Source |
|---|---|
| Train scenarios | All scenarios from the two non-X datasets. Includes the previously-held-out non-X scenarios (`ctu13-10`, `iot23-8-1`, `medbiot-torii_mal_all`) — reabsorbed into training because they are no longer the LODO target. |
| Per-train-scenario split | First 80% of windows (chronological) → train, last 20% → val. No internal test split. |
| Test scenarios | Every scenario from dataset X with ≥10 windows. Full chronological window sequence per scenario goes to test. |
| Excluded | The existing `excluded:` list (`ctu13-5/7/11/12`, `iot23-20-1/21-1`) stays excluded in every direction — consistent with the Phase 4–7 protocol. |

### 4.1 Concrete scenario assignments

**LODO-CTU13:**
- Train: `iot23-{1-1, 3-1, 7-1, 8-1, 9-1, 17-1, 33-1, 35-1, 48-1}` ∪ `medbiot-{bashlite_mal_spread_all, bashlite_mal_CC_all, mirai_mal_spread_all, mirai_mal_CC_all, torii_mal_all}`
- Test: `ctu13-{1, 2, 3, 4, 6, 8, 9, 10, 13}`

**LODO-IoT23:**
- Train: `ctu13-{1, 2, 3, 4, 6, 8, 9, 10, 13}` ∪ `medbiot-{bashlite_mal_spread_all, bashlite_mal_CC_all, mirai_mal_spread_all, mirai_mal_CC_all, torii_mal_all}`
- Test: `iot23-{1-1, 3-1, 7-1, 8-1, 9-1, 17-1, 33-1, 35-1, 48-1}`

**LODO-MedBIoT:**
- Train: `ctu13-{1, 2, 3, 4, 6, 8, 9, 10, 13}` ∪ `iot23-{1-1, 3-1, 7-1, 8-1, 9-1, 17-1, 33-1, 35-1, 48-1}`
- Test: `medbiot-{bashlite_mal_spread_all, bashlite_mal_CC_all, mirai_mal_spread_all, mirai_mal_CC_all, torii_mal_all}`

### 4.2 Rationale for 80/20 train/val (vs existing 60/20/20)

In LODO mode the dataset-X tail is no longer the test set, so the existing 20% test tail inside training scenarios is dead weight. Folding it into train would inflate the train set size relative to Phase 7 (confounds train-set-size with split-method comparisons). Folding into val gives the early-stopping signal more data without that confound.

## 5. File layout (non-overwrite guarantee)

Every LODO output goes to a new path. The existing paths are frozen.

```
higt-bot-code/
├── configs/
│   ├── split.yaml                          # UNTOUCHED
│   ├── split_lodo_ctu13.yaml               # NEW
│   ├── split_lodo_iot23.yaml               # NEW
│   └── split_lodo_medbiot.yaml             # NEW
├── experiments/
│   ├── phase5/  phase6/  phase7/           # UNTOUCHED
│   └── lodo/                               # NEW root
│       ├── ctu13/{phase4,phase5,phase6,phase7}/...
│       ├── iot23/{phase4,phase5,phase6,phase7}/...
│       └── medbiot/{phase4,phase5,phase6,phase7}/...
├── data/
│   ├── flow_embeddings/                    # UNTOUCHED
│   ├── flow_embeddings_lodo/               # NEW
│   │   ├── ctu13/  iot23/  medbiot/
│   └── inspection_logs/
│       ├── phase{4..8}_*                   # UNTOUCHED
│       └── lodo_{ctu13,iot23,medbiot}_*    # NEW (json + md)
└── scripts/
    └── run_lodo_ladder.py                  # NEW
```

**Structural guarantee:** No existing file is read-modified-written or deleted. Defaults on every modified script preserve the original output paths when called with no flags. A pre/post mod-time snapshot on `experiments/phase{5,6,7}/` and `data/flow_embeddings/` is run by the orchestrator to verify no accidental overwrite.

## 6. Code changes

Additive only — defaults preserve current behavior. No model code, training-loop, or evaluation logic is changed.

### 6.1 `src/data/dataset.py`

Extend `SplitSpec` with two optional fields:

```python
@dataclass
class SplitSpec:
    ...existing fields...
    holdout_dataset: str | None = None              # "ctu13" | "iot23" | "medbiot" | None
    lodo_test_scenarios: list[str] = field(default_factory=list)
```

In `load_split_files`, branch on `holdout_dataset`:
- If `None` (existing): identical behavior, byte-for-byte.
- If set: train and val draw from per-scenario chronological 80/20 of `train_scenarios`; test = full window sequence of every entry in `lodo_test_scenarios`. The existing `holdout_test_scenarios` field is ignored in LODO mode.

### 6.2 Scripts (additive flags only)

The following scripts gain a uniform flag triad. Defaults equal current hardcoded paths so existing reproducibility is preserved when scripts are called with no flags.

```python
# Pattern; each script's default matches its current hardcoded path/prefix.
# e.g. for train_phase5.py:
ap.add_argument("--split-config", type=Path, default=Path("configs/split.yaml"))
ap.add_argument("--out-dir", type=Path, default=Path("experiments/phase5"))
ap.add_argument("--results-prefix", type=str, default="phase5_temporal_gine")
```

Scripts to modify:
- `scripts/train_phase5.py`
- `scripts/train_phase6_sparse.py`
- `scripts/train_phase7.py`
- `scripts/cache_embeddings.py` (already has `--out-dir`; add `--split-config`)
- `scripts/baselines/run_rf.py`
- `scripts/baselines/run_xgb.py`
- `scripts/baselines/run_gnn.py`

Each edit is 3–5 lines (argparse declaration + replacing one hardcoded `Path(...)` with the parsed value).

### 6.3 New config files

`configs/split_lodo_ctu13.yaml`, `configs/split_lodo_iot23.yaml`, `configs/split_lodo_medbiot.yaml`.

Schema mirrors `configs/split.yaml` with new fields. Example for IoT-23 holdout:

```yaml
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

### 6.4 New orchestrator: `scripts/run_lodo_ladder.py`

Python orchestrator using `subprocess.run` so each phase executes in a clean process (no CUDA-context leakage, no module-state cross-talk).

```
Usage:
  uv run scripts/run_lodo_ladder.py --holdout {ctu13|iot23|medbiot}
                                    [--resume]
                                    [--seeds 42 1 2]
                                    [--stages phase4,phase5,cache,phase6,phase7]
```

Per-stage behavior:
1. Check for `<out-dir>/<stage>.done` marker — skip if `--resume` and marker present.
2. Capture pre-run mod-time snapshot on `experiments/phase{5,6,7}/` and `data/flow_embeddings/`.
3. Run stage via `subprocess.run` with the LODO config path and LODO output dir.
4. On success, write `<stage>.done` with timestamp, git SHA, exit code, command line.
5. Re-snapshot existing dirs and assert no mod-time changes (defensive non-overwrite check).
6. On failure, exit with non-zero status; resume continues from this stage on next run.

After all stages complete, write per-LODO `manifest.json` summarizing the run.

### 6.5 Tests

`tests/test_dataset.py` gains LODO-branch coverage:
- `holdout_dataset=None` produces byte-identical output to the existing implementation (regression guard).
- `holdout_dataset="ctu13"` puts all CTU-13 scenarios into test, no IoT-23/MedBIoT scenarios into test.
- `holdout_dataset="iot23"` mirror-image case.
- 80/20 internal chronological split is applied correctly.
- `excluded` list is honored in every direction.

## 7. Execution sequence

Per LODO direction, in order (each stage depends on the previous):

| Stage | Models | Est. time on RTX 5080 |
|---|---|---|
| Phase 4 | RF, XGB, GAT, GIN, GINE | ~10–20 min |
| Phase 5 | T-GINE + skip | ~30–60 min |
| Embedding cache | (uses existing streaming pattern) | ~5–15 min |
| Phase 6 | SAGPool | ~15–30 min |
| Phase 7 × 3 seeds | GT-edge {42, 1, 2} | ~30–90 min total |
| **Per LODO** | | **~2–3 hours** |
| **All 3 LODOs** | | **~6–10 hours (overnight)** |

LODO directions are independent. With a single RTX 5080 they run in series; running two in parallel would only context-switch on the same GPU and not actually save wall time. Series execution is the default and recommended path.

Stage dependencies within a LODO: Phase 4 baselines independent; Phase 5 → embedding cache → Phase 6 → Phase 7 chained.

## 8. Reporting deliverables

All in `data/inspection_logs/`, tracked in git:

| File | Content |
|---|---|
| `lodo_{ctu13,iot23,medbiot}_phase{4..7}_*.json` | Raw per-scenario metrics, mirrors existing `phase{4..7}_*.json` format |
| `lodo_{ctu13,iot23,medbiot}_summary.md` | Per-LODO narrative summary (one per direction) |
| `lodo_comparison_table.md` | Headline table: rows = ladder models, columns = mixed-pool + 3 LODO directions |
| `lodo_per_scenario_breakdown.md` | Per-test-scenario F1 to expose failure modes |
| `lodo_master.md` | Overall story, mirrors `phase8_master.md` style |

Headline table format:

| Model | Mixed-pool (existing) | LODO-CTU13 | LODO-IoT23 | LODO-MedBIoT |
|---|---|---|---|---|
| RF | (existing F1) | F1=? | F1=? | F1=? |
| XGB | (existing F1) | F1=? | F1=? | F1=? |
| GINE-flat | (existing F1) | F1=? | F1=? | F1=? |
| T-GINE+skip | 0.9661 | F1=? | F1=? | F1=? |
| SAGPool | 0.9674 | F1=? | F1=? | F1=? |
| **GT-edge** | **0.9677** | **F1 ± std** | **F1 ± std** | **F1 ± std** |

## 9. Streaming and resource use

No new streaming or memory-handling code is introduced. The LODO experiment reuses the existing patterns:

- **Graph construction streaming** (`src/data/build_graphs_streaming.py`) — not invoked by LODO; graphs in `data/graphs/` were built once and are reused across all experiments.
- **Lazy flow-sequence loading** (`src/data/flow_seq_dataset.FlowSeqGraphDataset`) — used by every trainer's `DataLoader`; loads flow tensors per-`__getitem__`. Trainer batches do not hold all flows in RAM.
- **Per-window embedding caching** (`scripts/cache_embeddings.py` lines 73–98) — load → encode → write parquet → `del` per window, with resume support (skip if output parquet exists).
- **Scaler-fit memory spike control** (`scripts/train_phase5.py` lines 81–87) — load graphs briefly for scaler fit, then `del tr_graphs; gc.collect()`.

These patterns are sufficient on RTX 5080 (16 GB VRAM) for the largest test scenarios (`iot23-39-1` at 2.1 GB parquet, `iot23-43-1` at 1.7 GB). No code path naively materializes a full dataset in memory.

## 10. Production-quality additions

| Addition | Purpose |
|---|---|
| Per-LODO `manifest.json` in `experiments/lodo/<holdout>/` | Captures config snapshot, all seeds, package versions (`uv pip freeze`), git SHA, stage statuses, output paths. One file reproduces the whole LODO without spelunking. |
| Resume markers (`.done` per stage) | Overnight runs failing mid-ladder resume from the failed stage instead of redoing hours of completed work. Marker contains timestamp, git SHA, exit code, command line. |
| Pre/post mod-time snapshot on existing dirs | Captures mod-times of `experiments/phase{5,6,7}/` and `data/flow_embeddings/` before and after each stage. Catches any accidental overwrite immediately. |
| `split_version: lodo-<x>-v1` field in every config + result JSON | Matches existing `split_version: 1` traceability convention. Bumping invalidates earlier runs cleanly. |
| LODO-branch tests in `tests/test_dataset.py` | Guards against regressions in the mixed-pool path and confirms LODO branch picks the right scenarios. |

## 11. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Phase 5 encoder fails to converge on reduced train set (especially LODO-CTU13, which drops the biggest dataset) | Monitor loss curve. If no convergence, document as a failure mode of the experiment itself rather than silently retrying with new hyperparams. |
| Flag-audit drift: a missed hardcoded path causes a LODO stage to silently write to `experiments/phase{5,6,7}/` | Pre/post mod-time check in orchestrator fails the run immediately. Implementation plan will include a `grep` checklist of every hardcoded path. |
| Class-imbalance shift in LODO test (different bot prevalence than mixed-pool test) | Report PR-AUC and confusion matrices alongside F1. PR-AUC is robust to prior shift. |
| GT-edge seed variance higher under LODO than mixed-pool (±0.001 observed at Phase 7) | Already covered by 3 seeds for GT-edge. If variance is large, this *is* a finding and gets reported. |

## 12. Out of scope (explicitly)

- Hyperparameter retuning per LODO direction.
- Window-size sensitivity under LODO.
- Visualization / interpretability under LODO (Phase 8 covers mixed-pool).
- Cross-fold variants or nested CV — only three LODO directions, no nesting.
- LODO over the SSL pretrain (Phase 6.3) — documented as a negative ablation, not on the critical path.
- LODO over DiffPool (Phase 6 dense) — SAGPool is the sparse pooling used in the final model; DiffPool was the dense alternative and is captured in mixed-pool Phase 8.

## 13. Acceptance criteria

The experiment is complete when:

1. Three LODO directions have all 8 ladder models trained, GT-edge ×3 seeds completed.
2. All `.done` markers present, `manifest.json` written for each direction.
3. Pre/post mod-time check confirms zero changes to `experiments/phase{5,6,7}/` and `data/flow_embeddings/`.
4. All five reporting deliverables produced and tracked in git.
5. `tests/test_dataset.py` passes including the new LODO-branch cases.
6. Regression guard: running any modified script with no flags and the original seed reproduces a result JSON byte-identical to the pre-modification run (the argparse defaults match the previous hardcoded values; no logic path changes).
