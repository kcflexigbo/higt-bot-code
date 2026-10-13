"""Test-set results on the main split, broken down by botnet family type (P2P vs centralised).

Uses the saved v3 test predictions (corrected labels and features), thresholds re-tuned on
validation where a *_val.npz exists (as in scripts/bootstrap_significance.py).
P2P families present in the corrected test set: IoT-23 1-1 (Hide and Seek) and 9-1 (Linux.Hajime).
Family names: https://www.stratosphereips.org/datasets-iot23 and the CTU-13 READMEs.

    uv run python -m scripts.p2p_family_eval
"""
from __future__ import annotations

import torch  # noqa: F401  (import before pandas on Windows)

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score

from scripts.bootstrap_significance import load

V3 = Path("data/inspection_logs/v3_emb3"); G3 = Path("data/inspection_logs/v3_gnn/predictions")
OUT = Path("data/inspection_logs/p2p")
MODELS = {
    "HiGT-Bot v2 (x+grp)": V3 / "pred_x_grp.npz",
    "XGB[x]": V3 / "pred_x.npz",
    "RF tuned": G3 / "rf_tuned.npz",
    "Directed GINE": G3 / "dirgine_seed42.npz",
    "Directed T-GINE": G3 / "t_gine_skip.npz",
    "HiGT-Bot directed": G3 / "higt_bot_edge.npz",
}
FAMILY = {  # scenario -> (family, architecture)
    "iot23-1-1": ("Hide and Seek", "P2P"), "iot23-9-1": ("Linux.Hajime", "P2P"),
    "iot23-3-1": ("Muhstik", "C&C"), "iot23-7-1": ("Linux.Mirai", "C&C"), "iot23-8-1": ("Hakai", "C&C"),
    "iot23-17-1": ("Kenjiro", "C&C"), "iot23-33-1": ("Kenjiro", "C&C"), "iot23-35-1": ("Mirai", "C&C"),
    "iot23-48-1": ("Mirai", "C&C"),
    "ctu13-1": ("Neris", "IRC"), "ctu13-2": ("Neris", "IRC"), "ctu13-9": ("Neris", "IRC"),
    "ctu13-3": ("Rbot", "IRC"), "ctu13-4": ("Rbot", "IRC"), "ctu13-10": ("Rbot", "IRC"),
    "ctu13-6": ("Menti", "C&C"), "ctu13-8": ("Murlo", "C&C"), "ctu13-13": ("Virut", "HTTP"),
}


def metrics(y, pred, proba):
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    return {"n": int(len(y)), "bots": int(y.sum()), "f1": 2 * tp / max(2 * tp + fp + fn, 1), "fn": fn, "fp": fp,
            "pr_auc": float(average_precision_score(y, proba)) if 0 < y.sum() < len(y) else None}


def main():
    M = {k: load(p) for k, p in MODELS.items()}
    sc = np.load(MODELS["XGB[x]"], allow_pickle=True)["scenario"].astype(str)
    groups = {"P2P (Hide and Seek, Hajime)": np.isin(sc, ["iot23-1-1", "iot23-9-1"]),
              "iot23-1-1 Hide and Seek": sc == "iot23-1-1", "iot23-9-1 Hajime": sc == "iot23-9-1",
              "Centralised (all other test scenarios)": ~np.isin(sc, ["iot23-1-1", "iot23-9-1"]),
              "Centralised IoT-23 only": np.char.startswith(sc, "iot23") & ~np.isin(sc, ["iot23-1-1", "iot23-9-1"])}
    res = {g: {k: metrics(v[0][m], v[1][m], v[2][m]) for k, v in M.items()} for g, m in groups.items()}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "family_eval.json").write_text(json.dumps(res, indent=2))
    for g, r in res.items():
        print(f"\n## {g}")
        for k, v in r.items():
            pr = f"{v['pr_auc']:.3f}" if v["pr_auc"] is not None else "n/a"
            print(f"  {k:22s} n {v['n']:6d} bots {v['bots']:5d} F1 {v['f1']:.3f} PR {pr} FN {v['fn']:4d} FP {v['fp']:4d}")


if __name__ == "__main__":
    main()
