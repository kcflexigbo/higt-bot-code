"""Re-train the LODO-IoT23 ladder end-to-end with --no-amp at every stage.

Used to verify that the AMP-trained LODO checkpoints are not contaminated
by mixed-precision training. The training-time AMP path is theoretically
safe (GradScaler protects gradients, master weights stay FP32), but the
test-eval bug we just patched motivates an empirical check.

Outputs land in a *_fp32 namespace so the existing AMP-trained artifacts
stay untouched:
  experiments/lodo/iot23_fp32/{phase5,phase6,phase7}/...
  data/flow_embeddings_lodo/iot23_fp32/...
  data/inspection_logs/*_lodo_iot23_fp32_*.json

Phase 4 baselines are skipped — they don't use AMP, so their results
would be byte-identical to the existing LODO-IoT23 phase4 JSONs.

Wall-time on RTX 5080 (16 GB), based on the AMP LODO-IoT23 run:
~50 minutes total. FP32 may be ~1.5x slower on Phase 5 / Phase 7.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HOLDOUT = "iot23"
SPLIT_CFG = f"configs/split_lodo_{HOLDOUT}.yaml"
CKPT_ROOT = Path(f"experiments/lodo/{HOLDOUT}_fp32")
EMB_DIR = Path(f"data/flow_embeddings_lodo/{HOLDOUT}_fp32")
RESULTS_DIR = Path("data/inspection_logs")
LOG_DIR = CKPT_ROOT / "logs"
SUFFIX_BASE = f"_lodo_{HOLDOUT}_fp32"


def _utc_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _ensure_dirs():
    for d in (CKPT_ROOT, EMB_DIR, LOG_DIR,
               CKPT_ROOT / "phase5", CKPT_ROOT / "phase6", CKPT_ROOT / "phase7"):
        d.mkdir(parents=True, exist_ok=True)


def _run_stage(label, cmd, resume):
    marker = CKPT_ROOT / f".{label}.done"
    log_path = LOG_DIR / f"{label}.log"
    if resume and marker.exists():
        print(f"[fp32:{HOLDOUT}]     skipped (resume marker present)", flush=True)
        return 0, 0.0
    started = _utc_iso()
    t0 = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as lf:
        lf.write(f"# {label}\n# started_utc={started}\n")
        lf.write(f"# cmd={' '.join(cmd)}\n\n")
        lf.flush()
        proc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT)
    dt = time.perf_counter() - t0
    if proc.returncode == 0:
        marker.write_text(
            f"started_utc={started}\nfinished_utc={_utc_iso()}\n"
            f"duration_s={dt:.1f}\nrc=0\ncmd={' '.join(cmd)}\n",
            encoding="utf-8",
        )
    return proc.returncode, dt


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 1, 2])
    args = ap.parse_args()

    _ensure_dirs()

    # Phase 5 -- TemporalGINE + skip, --no-amp
    cmd_phase5 = [
        sys.executable, "scripts/train_phase5.py",
        "--raw-feature-skip",
        "--split-config", SPLIT_CFG,
        "--ckpt-dir", str(CKPT_ROOT / "phase5"),
        "--results-dir", str(RESULTS_DIR),
        "--results-suffix", f"{SUFFIX_BASE}_skip",
        "--no-amp",
    ]

    # Cache -- uses the Phase 5 fp32 encoder
    cmd_cache = [
        sys.executable, "scripts/cache_embeddings.py",
        "--checkpoint", str(CKPT_ROOT / "phase5" / f"temporal_gine{SUFFIX_BASE}_skip.pt"),
        "--split-config", SPLIT_CFG,
        "--out-dir", str(EMB_DIR),
    ]

    # Phase 6 -- SAGPool, --no-amp
    cmd_phase6 = [
        sys.executable, "scripts/train_phase6_sparse.py",
        "--split-config", SPLIT_CFG,
        "--ckpt-dir", str(CKPT_ROOT / "phase6"),
        "--results-dir", str(RESULTS_DIR),
        "--results-suffix", f"{SUFFIX_BASE}_sparse",
        "--emb-dir", str(EMB_DIR),
        "--no-amp",
    ]

    stages = [("phase5", cmd_phase5), ("cache", cmd_cache), ("phase6", cmd_phase6)]
    for seed in args.seeds:
        cmd_phase7 = [
            sys.executable, "scripts/train_phase7.py",
            "--gt-variant", "edge",
            "--seed", str(seed),
            "--split-config", SPLIT_CFG,
            "--ckpt-dir", str(CKPT_ROOT / "phase7"),
            "--results-dir", str(RESULTS_DIR),
            "--results-suffix", f"{SUFFIX_BASE}_edge_seed{seed}",
            "--emb-dir", str(EMB_DIR),
            "--no-amp",
        ]
        stages.append((f"phase7_seed{seed}", cmd_phase7))

    summary = {"seeds": args.seeds, "stages": []}
    for label, cmd in stages:
        print(f"[fp32:{HOLDOUT}] >>> {label}", flush=True)
        rc, dt = _run_stage(label, cmd, resume=args.resume)
        print(f"[fp32:{HOLDOUT}]     rc={rc}  {dt:.1f}s", flush=True)
        summary["stages"].append({"name": label, "rc": rc, "duration_s": dt})
        if rc != 0:
            print(f"[fp32:{HOLDOUT}] FAILED at {label}", file=sys.stderr, flush=True)
            (CKPT_ROOT / "manifest.json").write_text(json.dumps(summary, indent=2))
            return rc

    (CKPT_ROOT / "manifest.json").write_text(json.dumps(summary, indent=2))
    print(f"[fp32:{HOLDOUT}] all stages complete", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
