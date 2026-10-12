"""Aggregate E-GraphSAGE (+ tuned RF) results across label versions.

Reads data/inspection_logs/egraphsage/<v>/{node,edge2node}_seed*_preds.npz (+ .json
for the val-tuned threshold) and rf_tuned_preds.npz / rf_tuned.json, and prints
markdown tables: F1@0.5, F1@val-tuned threshold, PR-AUC, FN/FP (both thresholds)
for all test nodes, CTU-13 test nodes, and held-out-scenario test nodes.
Mean +- SD (ddof=1) over seeds; RF is deterministic (seed 42 only).

    uv run python scripts/egraphsage_summary.py > table.md
"""

from __future__ import annotations

import json
from pathlib import Path

import torch  # noqa: F401  (load before numpy-heavy libs on Windows)
import numpy as np
from sklearn.metrics import average_precision_score, f1_score

ROOT = Path("data/inspection_logs/egraphsage")
HELDOUT = {"ctu13-10", "iot23-8-1", "medbiot-torii_mal_all"}
VERSIONS = {"v0": "any-endpoint", "v1": "infected-source", "v2": "corrected infected-source"}
SUBSETS = {
    "all": lambda s: np.ones(len(s), bool),
    "CTU-13": lambda s: np.char.startswith(s.astype(str), "ctu13"),
    "held-out": lambda s: np.isin(s, list(HELDOUT)),
}


def metrics(y, proba, thr, pred05=None):
    p05 = (proba >= 0.5).astype(int) if pred05 is None else pred05
    pt = (proba >= thr).astype(int)
    pr = average_precision_score(y, proba) if y.sum() else float("nan")
    return {
        "f1_05": f1_score(y, p05, zero_division=0), "f1_t": f1_score(y, pt, zero_division=0),
        "pr_auc": pr,
        "fn_05": int(((p05 == 0) & (y == 1)).sum()), "fp_05": int(((p05 == 1) & (y == 0)).sum()),
        "fn_t": int(((pt == 0) & (y == 1)).sum()), "fp_t": int(((pt == 1) & (y == 0)).sum()),
        "n_pos": int(y.sum()), "n": int(len(y)),
    }


def runs_for(vdir: Path, model: str):
    out = []
    if model == "rf":
        npz, js = vdir / "rf_tuned_preds.npz", vdir / "rf_tuned.json"
        if npz.exists():
            d = np.load(npz, allow_pickle=True)
            proba = d["proba"]
            # RandomForest.predict == argmax, so @0.5 ties go to benign.
            out.append((d["y"], proba, d["scenario"], json.loads(js.read_text())["threshold"],
                        (proba > 0.5).astype(int), 42))
        return out
    for seed in (42, 1, 2):
        npz, js = vdir / f"{model}_seed{seed}_preds.npz", vdir / f"{model}_seed{seed}.json"
        if npz.exists() and js.exists():
            d = np.load(npz, allow_pickle=True)
            out.append((d["y"], d["proba"], d["scenario"], float(json.loads(js.read_text())["threshold_tuned"]),
                        d["pred"].astype(int), seed))
    return out


def fmt(vals, digits=3, integer=False):
    vals = np.asarray(vals, float)
    if len(vals) == 1:
        return f"{vals[0]:.0f}" if integer else f"{vals[0]:.{digits}f}"
    m, s = np.nanmean(vals), np.nanstd(vals, ddof=1)
    return f"{m:.0f} ± {s:.0f}" if integer else f"{m:.{digits}f} ± {s:.{digits}f}"


def main() -> None:
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    models = [("rf", "RF (tuned, seed 42)"), ("node", "E-GraphSAGE node (z_v)"),
              ("edge2node", "E-GraphSAGE edge→node (max)")]
    summary = {}
    for v, vname in VERSIONS.items():
        vdir = ROOT / v
        print(f"\n### {v} — {vname} labels\n")
        for sub, fn in SUBSETS.items():
            print(f"**{sub} test nodes**\n")
            print("| model | seeds | thr (val) | F1@0.5 | F1@thr | PR-AUC | FN@0.5 | FP@0.5 | FN@thr | FP@thr |")
            print("|---|---|---|---|---|---|---|---|---|---|")
            for key, name in models:
                rs = runs_for(vdir, key)
                if not rs:
                    print(f"| {name} | 0 | – | – | – | – | – | – | – | – |"); continue
                ms = []
                for y, proba, scen, thr, p05, _ in rs:
                    mk = fn(np.asarray(scen))
                    ms.append(metrics(y[mk], proba[mk], thr, p05[mk]))
                thrs = [r[3] for r in rs]
                col = {k: [m[k] for m in ms] for k in ms[0]}
                summary.setdefault(v, {}).setdefault(sub, {})[key] = {
                    k: (float(np.nanmean(x)), float(np.nanstd(x, ddof=1)) if len(x) > 1 else 0.0)
                    for k, x in col.items()} | {"seeds": [r[5] for r in rs], "thresholds": thrs}
                print(f"| {name} | {len(rs)} | {fmt(thrs, 2)} | {fmt(col['f1_05'])} | {fmt(col['f1_t'])} | "
                      f"{fmt(col['pr_auc'])} | {fmt(col['fn_05'], integer=True)} | {fmt(col['fp_05'], integer=True)} | "
                      f"{fmt(col['fn_t'], integer=True)} | {fmt(col['fp_t'], integer=True)} |")
            print(f"\n(n = {ms[0]['n']:,} nodes, {ms[0]['n_pos']:,} bots)\n")
    (ROOT / "egraphsage_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
