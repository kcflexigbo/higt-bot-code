"""Tree-side experiments for the simulated review (manuscript/review/peer_review_v2_2026-10-13.md).

Parts (each idempotent: finished outputs are skipped, so the script can be re-launched
after a crash):
  tables    feature tables: the v3 split caches (data/inspection_logs/v3_emb3/cache_*.npz,
            built by scripts/higt_v2_trees.py) plus window ids, and the same x / grp blocks for
            the graphs_v3 scenarios never used by configs/split.yaml ("pristine", review C1)
  seeds     5 seeds of XGB[x], XGB[x+grp] (HiGT-Bot v2), RF tuned, the M2 variants
            (within-window normalisation, grp profile part, grp shared-target part), and the
            other block combinations of higt_v2_trees.py (validation-only selection, C1c);
            thresholds tuned on validation F1; predictions on val / test / pristine
  botminer  unsupervised BotMiner-style per-window cohort baseline (M2a)
  importance gain and permutation importance of the x+grp model (M2d)
  loso      CTU-13 leave-one-scenario-out with the quantile scaler refitted per fold (M9)

    uv run python -m scripts.revision_trees --part tables|seeds|botminer|importance|loso
Outputs: data/inspection_logs/revision/
"""
from __future__ import annotations

import os

os.environ.setdefault("HIGT_GRAPHS_DIR", "data/graphs_v3")
import torch  # noqa: E402  (import before pandas on Windows)

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score
from xgboost import XGBClassifier

from src.data.dataset import SplitSpec, load_split_files
from src.features.host_blocks import fit_group_scaler, grp_block

OUT = Path("data/inspection_logs/revision")
CACHE = Path("data/inspection_logs/v3_emb3")
G3 = Path("data/graphs_v3")
SEEDS = (42, 1, 2, 3, 4)
NJ = 8
PRISTINE = ["iot23-34-1", "iot23-36-1", "iot23-39-1", "iot23-42-1", "iot23-43-1", "iot23-44-1",
            "iot23-49-1", "iot23-52-1", "medbiot-bashlite_leg", "medbiot-mirai_leg", "medbiot-torii_leg"]
SECONDARY = ["ctu13-5", "ctu13-7", "ctu13-11", "iot23-20-1", "iot23-21-1"]  # listed as excluded, counted before
GRID = np.linspace(0.01, 0.99, 99)


def f1(y, p):
    tp = int(((p == 1) & (y == 1)).sum()); fp = int(((p == 1) & (y == 0)).sum()); fn = int(((p == 0) & (y == 1)).sum())
    return 2 * tp / max(2 * tp + fp + fn, 1)


def tune(y, p):
    return float(GRID[int(np.argmax([f1(y, (p >= t).astype(int)) for t in GRID]))])


def win_ids(files):
    return np.concatenate([np.full(int(torch.load(fp, weights_only=False).num_nodes), i) for i, fp in enumerate(files)])


# ---------------------------------------------------------------- tables
def part_tables(spec):
    path = OUT / "tables.npz"
    if path.exists():
        print("tables exist"); return
    T = {s: dict(np.load(CACHE / f"cache_{s}.npz", allow_pickle=True)) for s in ("train", "val", "test")}
    out = {}
    for s in ("train", "val", "test"):
        w = win_ids(load_split_files(s, spec))
        assert len(w) == len(T[s]["y"]), s
        for k in ("x", "grp", "seq", "nbr", "emb", "y", "scenario"):
            out[f"{s}_{k}"] = T[s][k]
        out[f"{s}_win"] = w
    qt = fit_group_scaler(T["train"]["x"])
    for name, scen in (("pristine", PRISTINE), ("secondary", SECONDARY)):
        X, Gp, Y, S, W, wid = [], [], [], [], [], 0
        for sc in scen:
            for fp in sorted((G3 / sc).glob("window_*.pt")):
                g = torch.load(fp, weights_only=False)
                x = g.x.numpy().astype(np.float32)
                X.append(x); Gp.append(grp_block(x, g.edge_index.numpy(), qt)); Y.append(g.y.numpy())
                S += [sc] * int(g.num_nodes); W.append(np.full(int(g.num_nodes), wid)); wid += 1
        out |= {f"{name}_x": np.vstack(X), f"{name}_grp": np.vstack(Gp), f"{name}_y": np.concatenate(Y),
                f"{name}_scenario": np.array(S), f"{name}_win": np.concatenate(W)}
        print(name, len(S), "host-windows,", int(np.concatenate(Y).sum()), "bots", flush=True)
    np.savez(path, **out)


def load_tables():
    d = np.load(OUT / "tables.npz", allow_pickle=True)
    T = {}
    for k in d.files:
        s, b = k.split("_", 1)
        T.setdefault(s, {})[b] = d[k]
    return T


def wnorm(x, win):
    """Within-window percentile rank and z-score of each host feature."""
    r = np.zeros_like(x, dtype=np.float32); z = np.zeros_like(x, dtype=np.float32)
    order = np.argsort(win, kind="stable"); ws = win[order]
    bounds = np.flatnonzero(np.diff(ws)) + 1
    for idx in np.split(order, bounds):
        v = x[idx]
        if len(idx) > 1:
            r[idx] = (np.argsort(np.argsort(v, 0), 0) / (len(idx) - 1)).astype(np.float32)
            z[idx] = ((v - v.mean(0)) / (v.std(0) + 1e-6)).astype(np.float32)
    return np.hstack([r, z])


def blocks(T, s, name):
    D = T[s]
    if name == "wnorm":
        key = f"_wn_{s}"
        if key not in T:
            T[key] = wnorm(D["x"], D["win"])
        return T[key]
    if name == "grp_prof":
        return D["grp"][:, :6]
    if name == "grp_share":
        return D["grp"][:, 6:]
    return D[name]


MODELS = {  # name -> (kind, blocks)
    "XGB[x]": ("xgb", ["x"]),
    "v2 XGB[x+grp]": ("xgb", ["x", "grp"]),
    "RF tuned": ("rf", ["x"]),
    "XGB[x+wnorm]": ("xgb", ["x", "wnorm"]),
    "XGB[x+grp_prof]": ("xgb", ["x", "grp_prof"]),
    "XGB[x+grp_share]": ("xgb", ["x", "grp_share"]),
    "XGB[x+wnorm+grp]": ("xgb", ["x", "wnorm", "grp"]),
}
SELECTION = {  # the other combinations of higt_v2_trees.py, validation-only selection (C1c); seeds 42,1,2
    "XGB[x+seq]": ("xgb", ["x", "seq"]), "XGB[x+seq+grp]": ("xgb", ["x", "seq", "grp"]),
    "XGB[x+emb]": ("xgb", ["x", "emb"]), "XGB[x+grp+emb]": ("xgb", ["x", "grp", "emb"]),
    "XGB[x+seq+grp+emb]": ("xgb", ["x", "seq", "grp", "emb"]), "XGB[x+nbr+seq+grp]": ("xgb", ["x", "nbr", "seq", "grp"]),
}


def make(kind, ytr, seed):
    if kind == "rf":
        return RandomForestClassifier(n_estimators=500, class_weight="balanced", random_state=seed, n_jobs=NJ)
    return XGBClassifier(n_estimators=600, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                         scale_pos_weight=(len(ytr) - ytr.sum()) / max(ytr.sum(), 1), tree_method="hist",
                         random_state=seed, n_jobs=NJ, eval_metric="aucpr")


def fname(model, seed):
    return OUT / "seed_preds" / (model.replace("[", "_").replace("]", "").replace("+", "_").replace(" ", "_") + f"_s{seed}.npz")


def part_seeds():
    T = load_tables(); (OUT / "seed_preds").mkdir(parents=True, exist_ok=True)
    jobs = [(m, s) for m in MODELS for s in SEEDS] + [(m, s) for m in SELECTION for s in (42, 1, 2)]
    for model, seed in jobs:
        f = fname(model, seed)
        if f.exists():
            continue
        kind, bl = (MODELS | SELECTION)[model]
        X = {s: np.hstack([blocks(T, s, b) for b in bl]) for s in ("train", "val", "test")}
        ytr = T["train"]["y"]
        clf = make(kind, ytr, seed).fit(X["train"], ytr)
        pr = {s: clf.predict_proba(X[s])[:, 1] for s in ("val", "test")}
        if model in MODELS:
            for s in ("pristine", "secondary"):
                pr[s] = clf.predict_proba(np.hstack([blocks(T, s, b) for b in bl]))[:, 1]
        thr = tune(T["val"]["y"], pr["val"])
        np.savez(f, threshold=thr, **{f"p_{k}": v.astype(np.float32) for k, v in pr.items()})
        print(f"{model:22s} seed {seed}: val thr {thr:.2f}  test F1 {f1(T['test']['y'], (pr['test'] >= thr).astype(int)):.4f}",
              flush=True)


# ---------------------------------------------------------------- BotMiner-style baseline
def botminer_scores(x, grp_unused, win, ei_by_win, qt):
    """Per window: DBSCAN (eps 1.0, min_samples 2) on quantile-normalised host features
    (communication-profile plane); a host's score is the number of other members of its
    cluster that share at least one outgoing destination with it (cross-plane overlap),
    divided by window size. Hosts outside any cluster score 0. No labels are used."""
    s = np.zeros(len(x), np.float32)
    q = qt.transform(x)
    for w in np.unique(win):
        idx = np.flatnonzero(win == w)
        if len(idx) < 2:
            continue
        lab = DBSCAN(eps=1.0, min_samples=2).fit_predict(q[idx])
        src, dst = ei_by_win[w]
        outs = [set() for _ in idx]
        for a, b in zip(src, dst):
            outs[a].add(b)
        for c in set(lab) - {-1}:
            mem = np.flatnonzero(lab == c)
            for i in mem:
                k = sum(1 for j in mem if j != i and outs[i] & outs[j])
                s[idx[i]] = (k + 0.5) / len(idx)  # +0.5: clustered hosts rank above unclustered ones
    return s


def part_botminer(spec):
    f = OUT / "botminer.npz"
    if f.exists():
        print("botminer exists"); return
    T = load_tables()
    qt = fit_group_scaler(T["train"]["x"])
    res = {}
    for s, files in (("val", load_split_files("val", spec)), ("test", load_split_files("test", spec)),
                     ("pristine", [fp for sc in PRISTINE for fp in sorted((G3 / sc).glob("window_*.pt"))])):
        ei = {i: torch.load(fp, weights_only=False).edge_index.numpy() for i, fp in enumerate(files)}
        res[s] = botminer_scores(T[s]["x"], None, T[s]["win"], ei, qt)
        print("botminer", s, "done", flush=True)
    np.savez(f, threshold=tune(T["val"]["y"], res["val"]), **{f"p_{k}": v for k, v in res.items()})


# ---------------------------------------------------------------- importance
def part_importance():
    f = OUT / "importance.json"
    if f.exists():
        print("importance exists"); return
    T = load_tables()
    names = [f"x{i}" for i in range(9)] + ["grp_nn1_dist", "grp_nn_mean_dist", "grp_nn5_dist", "grp_share_r1",
                                           "grp_share_r2", "grp_count_r2", "grp_n_cotarget_hosts", "grp_max_target_jaccard"]
    Xtr = np.hstack([T["train"]["x"], T["train"]["grp"]]); ytr = T["train"]["y"]
    clf = make("xgb", ytr, 42).fit(Xtr, ytr)
    gain = clf.get_booster().get_score(importance_type="gain")
    gain = {names[int(k[1:])]: float(v) for k, v in gain.items()}
    tot = sum(gain.values()); gain = {k: v / tot for k, v in sorted(gain.items(), key=lambda kv: -kv[1])}
    Xva = np.hstack([T["val"]["x"], T["val"]["grp"]]); yva = T["val"]["y"]
    base = average_precision_score(yva, clf.predict_proba(Xva)[:, 1])
    rng = np.random.default_rng(0); perm = {}
    for j, n in enumerate(names):  # permutation on the validation set (test set not used)
        drops = []
        for _ in range(3):
            Xp = Xva.copy(); Xp[:, j] = rng.permutation(Xp[:, j])
            drops.append(base - average_precision_score(yva, clf.predict_proba(Xp)[:, 1]))
        perm[n] = float(np.mean(drops))
    grp_blk = {}
    for nm, cols in (("all x", range(9)), ("all grp", range(9, 17)), ("grp profile (g1-6)", range(9, 15)),
                     ("grp shared-target (g7-8)", range(15, 17))):
        drops = []
        for _ in range(3):
            Xp = Xva.copy(); p = rng.permutation(len(Xp)); Xp[:, list(cols)] = Xp[p][:, list(cols)]
            drops.append(base - average_precision_score(yva, clf.predict_proba(Xp)[:, 1]))
        grp_blk[nm] = float(np.mean(drops))
    f.write_text(json.dumps({"gain_share": gain, "permutation_val_prauc_drop": perm, "block_permutation_val_prauc_drop": grp_blk,
                             "val_pr_auc": float(base)}, indent=2))
    print(json.dumps(grp_blk, indent=1))


# ---------------------------------------------------------------- LOSO with per-fold scaler
def part_loso(spec):
    """Same protocol as scripts/higt_v2_trees.run_loso (train / val windows of every other
    scenario of all three datasets; all windows of the left-out CTU-13 scenario as test),
    but the quantile scaler of the group block is refitted on each fold's training hosts."""
    f = OUT / "loso_refit.json"
    res = json.loads(f.read_text()) if f.exists() else {}
    parts = {}
    for s in ("train", "val", "test"):
        for fp in load_split_files(s, spec):
            g = torch.load(fp, weights_only=False)  # all datasets, as in higt_v2_trees.run_loso
            parts.setdefault(s, []).append((g.scenario, g.x.numpy().astype(np.float32), g.edge_index.numpy(), g.y.numpy()))
    allw = [(s, *w) for s, ws in parts.items() for w in ws]
    scen = sorted({w[1] for w in allw if w[1].startswith("ctu13")})
    for left in scen:
        if left in res and len(res[left]) == 3:
            continue
        tr = [w for w in allw if w[0] == "train" and w[1] != left]
        va = [w for w in allw if w[0] == "val" and w[1] != left]
        te = [w for w in allw if w[1] == left]
        if sum(int(w[4].sum()) for w in te) == 0:
            continue
        qt = fit_group_scaler(np.vstack([w[2] for w in tr]))  # refitted on the fold's training hosts only
        D = {}
        for nm, ws in (("tr", tr), ("va", va), ("te", te)):
            D[nm] = {"x": np.vstack([w[2] for w in ws]), "grp": np.vstack([grp_block(w[2], w[3], qt) for w in ws]),
                     "y": np.concatenate([w[4] for w in ws])}
        res[left] = {}
        for seed in (42, 1, 2):
            r = {}
            for model, bl in (("XGB[x]", ["x"]), ("v2 XGB[x+grp]", ["x", "grp"])):
                X = {k: np.hstack([D[k][b] for b in bl]) for k in D}
                clf = make("xgb", D["tr"]["y"], seed).fit(X["tr"], D["tr"]["y"])
                pv = clf.predict_proba(X["va"])[:, 1]; pt = clf.predict_proba(X["te"])[:, 1]
                t = tune(D["va"]["y"], pv); y = D["te"]["y"]
                neg = np.sort(pt[y == 0])[::-1]
                t1 = neg[int(0.01 * len(neg))] if len(neg) else 1.0  # threshold giving 1% FPR on the left-out scenario
                r[model] = {"pr_auc": float(average_precision_score(y, pt)), "f1_val_thr": f1(y, (pt >= t).astype(int)),
                            "val_thr": t, "f1_oracle": max(f1(y, (pt >= g).astype(int)) for g in GRID),
                            "recall_at_1pct_fpr": float(((pt > t1) & (y == 1)).sum() / max(y.sum(), 1))}
            res[left][str(seed)] = r
            f.write_text(json.dumps(res, indent=2))
        print(left, json.dumps({m: {k: round(v, 3) for k, v in res[left]["42"][m].items()} for m in res[left]["42"]}), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True, choices=("tables", "seeds", "botminer", "importance", "loso"))
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    spec = SplitSpec.load()
    {"tables": lambda: part_tables(spec), "seeds": part_seeds, "botminer": lambda: part_botminer(spec),
     "importance": part_importance, "loso": lambda: part_loso(spec)}[a.part]()


if __name__ == "__main__":
    main()
