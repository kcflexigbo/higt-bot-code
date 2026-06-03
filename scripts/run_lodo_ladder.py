"""LODO orchestrator: runs the full Phase 4-7 ladder for one held-out dataset.

Per stage:
  1. Skip if --resume and <stage>.done marker exists.
  2. Capture pre-stage mod-time snapshot on the protected directories.
  3. Run the stage as a clean subprocess.
  4. Verify post-stage mod-times match (no accidental overwrite).
  5. Write <stage>.done marker with timestamp, exit code, command.

Stages:
  phase4    - RF, XGB, GAT, GIN, GINE (5 subprocesses)
  phase5    - Phase 5 encoder training (1 subprocess)
  cache     - Embedding cache for this LODO direction (1 subprocess)
  phase6    - SAGPool training (1 subprocess)
  phase7    - GT-edge training, one subprocess per seed

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

    # Phase 4 - five subprocesses, each tagged with the LODO suffix in the --out filename.
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

    # Phase 5 - encoder training (skip variant matches the final-model recipe).
    cmds.append(("phase5", [
        sys.executable, "scripts/train_phase5.py",
        "--raw-feature-skip",
        "--split-config", str(paths.split_config),
        "--ckpt-dir", str(paths.ckpt_root / "phase5"),
        "--results-dir", str(paths.results_dir),
        "--results-suffix", f"{suffix}_skip",
    ], None))

    # Cache - encode all windows under this LODO encoder.
    cmds.append(("cache", [
        sys.executable, "scripts/cache_embeddings.py",
        "--checkpoint", str(paths.ckpt_root / "phase5" / f"temporal_gine{suffix}_skip.pt"),
        "--split-config", str(paths.split_config),
        "--out-dir", str(paths.emb_dir),
    ], None))

    # Phase 6 - SAGPool over the LODO embeddings.
    cmds.append(("phase6", [
        sys.executable, "scripts/train_phase6_sparse.py",
        "--split-config", str(paths.split_config),
        "--ckpt-dir", str(paths.ckpt_root / "phase6"),
        "--results-dir", str(paths.results_dir),
        "--results-suffix", f"{suffix}_sparse",
        "--emb-dir", str(paths.emb_dir),
    ], None))

    # Phase 7 - GT-edge, once per seed.
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


def snapshot_protected_dirs() -> dict[str, str]:
    """Return {file_path: lastWriteTimeUtc_iso} for every file under PROTECTED_DIRS.

    Used as a non-overwrite tripwire - if anything in here changes during a
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
    # New files inside protected dirs - also a violation.
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
    # Filter to requested stages - stage_label starts with one of phaseN/cache.
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
            write_manifest(paths, results, args.seeds)
            return res.rc

    write_manifest(paths, results, args.seeds)
    print(f"[lodo:{args.holdout}] all stages complete; manifest at {paths.manifest}",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
