"""Within-ISOT 2010 evaluation by window-grouped 5-fold cross-validation.

All 35 bot host-windows of ISOT fall in its last 18 windows (868-885 of 886), so a
chronological split cannot train on any bot. Instead windows are dealt into 5 folds,
stratified by whether the window contains a bot (seeded): for fold k, test = fold k,
validation = fold (k+1) mod 5 (threshold / early stopping), train = the other three.
Host-windows of one window never cross folds. The same fold assignment is used for every
model, so out-of-fold predictions are paired. Adjacent windows of the bot period are
correlated; this is a within-capture estimate, not a transfer result.

Questions: does neighbour information help when bots do contact each other (ISOT: 51% of
bot host-windows have a bot neighbour; CTU-13/IoT-23: none)? Trees on x, x+grp (HiGT-Bot v2),
x+nbr (mean neighbour features: a message-passing proxy), x+nbr+grp; GINE undirected vs
directed (HIGT_DIRECTED, same hyperparameters as the thesis GINE baseline).

    uv run python -m scripts.p2p_isot_cv --part trees
    HIGT_DIRECTED=0 uv run python -m scripts.p2p_isot_cv --part gine --tag gine
    HIGT_DIRECTED=1 uv run python -m scripts.p2p_isot_cv --part gine --tag dirgine
Output: data/inspection_logs/p2p/isot_cv_<part/tag>.json and isot_cv_oof_<model>.npz
"""
from __future__ import annotations

import os

os.environ.setdefault("HIGT_GRAPHS_DIR", "data/graphs_v3")
import torch  # noqa: E402  (import before pandas on Windows)

import argparse
import copy
import json
from pathlib import Path

import numpy as np

from scripts.higt_v2_trees import fit_predict, metrics, tune
from src.features.host_blocks import fit_group_scaler, grp_block, nbr_block

OUT = Path("data/inspection_logs/p2p")
SC = "isot-2010"
K = 5


def load():
    files = sorted((Path(os.environ["HIGT_GRAPHS_DIR"]) / SC).glob("window_*.pt"))
    return [torch.load(f, weights_only=False) for f in files]


def folds(graphs, seed):
    rng = np.random.default_rng(seed)
    has_bot = np.array([int(g.y.sum()) > 0 for g in graphs])
    fold = np.empty(len(graphs), dtype=int)
    for grp in (np.flatnonzero(has_bot), np.flatnonzero(~has_bot)):
        idx = rng.permutation(grp)
        fold[idx] = np.arange(len(idx)) % K
    return fold


def summarise(y, p, thr_per_fold, node_fold):
    pred = np.zeros_like(y)
    for k, t in thr_per_fold.items():
        m = node_fold == k
        pred[m] = (p[m] >= t).astype(int)
    return metrics(y, pred, p)


def part_trees(graphs, seeds):
    X = [g.x.numpy().astype(np.float32) for g in graphs]
    E = [g.edge_index.numpy() for g in graphs]
    A = [g.edge_attr.numpy().astype(np.float32) for g in graphs]
    Y = [g.y.numpy().astype(int) for g in graphs]
    NBR = [nbr_block(x, e, a) for x, e, a in zip(X, E, A)]
    res = {}
    for seed in seeds:
        fold = folds(graphs, seed)
        node_fold = np.concatenate([np.full(len(y), fold[i]) for i, y in enumerate(Y)])
        y_all = np.concatenate(Y)
        oof = {b: np.zeros(len(y_all)) for b in ("x", "x+grp", "x+nbr", "x+nbr+grp")}
        thr = {b: {} for b in oof}
        for k in range(K):
            te, va = fold == k, fold == (k + 1) % K
            tr = ~(te | va)
            qt = fit_group_scaler(np.vstack([X[i] for i in np.flatnonzero(tr)]), seed=seed)
            GRP = {i: grp_block(X[i], E[i], qt) for i in range(len(graphs))}

            def table(mask, blocks):
                ids = np.flatnonzero(mask)
                parts = []
                for i in ids:
                    cols = [X[i]]
                    if "nbr" in blocks:
                        cols.append(NBR[i])
                    if "grp" in blocks:
                        cols.append(GRP[i])
                    parts.append(np.hstack(cols))
                return np.vstack(parts), np.concatenate([Y[i] for i in ids])

            for b in oof:
                Xtr, ytr = table(tr, b); Xva, yva = table(va, b); Xte, _ = table(te, b)
                pva, pte = fit_predict(Xtr, ytr, Xva, Xte, seed)
                thr[b][k] = tune(yva, pva)
                oof[b][node_fold == k] = pte
        for b in oof:
            r = summarise(y_all, oof[b], thr[b], node_fold)
            res.setdefault(b, []).append(r)
            print(f"[trees seed {seed}] {b:10s} F1 {r['f1']:.3f} PR {r['pr_auc']:.3f} FN {r['fn']} FP {r['fp']}", flush=True)
            if seed == seeds[0]:
                np.savez(OUT / f"isot_cv_oof_{b.replace('+', '_')}.npz", y=y_all, proba=oof[b], fold=node_fold)
    return res


def part_gine(graphs, seeds, tag):
    from torch_geometric.loader import DataLoader

    from src.data.dataset import apply_edge_scaler, apply_feature_scaler, fit_edge_scaler, fit_feature_scaler
    from src.models.gnn_baselines import build_model
    from src.training.loop import TrainConfig, predict, train_one_model
    from src.utils.seeding import pick_device, set_seed
    device = pick_device()
    res = []
    for seed in seeds:
        fold = folds(graphs, seed)
        Y = [g.y.numpy().astype(int) for g in graphs]
        node_fold = np.concatenate([np.full(len(y), fold[i]) for i, y in enumerate(Y)])
        y_all = np.concatenate(Y); oof = np.zeros(len(y_all)); thr = {}
        for k in range(K):
            set_seed(seed)
            te_i, va_i = np.flatnonzero(fold == k), np.flatnonzero(fold == (k + 1) % K)
            tr_i = np.flatnonzero(~np.isin(fold, [k, (k + 1) % K]))
            tr = [copy.copy(graphs[i]) for i in tr_i]; va = [copy.copy(graphs[i]) for i in va_i]
            te = [copy.copy(graphs[i]) for i in te_i]
            mean, std = fit_feature_scaler(tr); e_mean, e_std = fit_edge_scaler(tr)
            for part in (tr, va, te):
                apply_feature_scaler(part, mean, std); apply_edge_scaler(part, e_mean, e_std)
            model = build_model("gine", int(tr[0].x.size(1)), edge_dim=int(tr[0].edge_attr.size(1)),
                                hidden=64, dropout=0.2, num_layers=2)
            cfg = TrainConfig(lr=1e-3, weight_decay=1e-4, batch_size=32, max_epochs=100, patience=20,
                              use_focal=True, focal_gamma=2.0, drop_edge_p=0.2)
            train_one_model(model, tr, va, cfg=cfg, device=device, log_prefix=f"[{tag} s{seed} f{k}] ")
            yv, _, pv, _ = predict(model, DataLoader(va, batch_size=32), device)
            _, _, pt, _ = predict(model, DataLoader(te, batch_size=32), device)
            thr[k] = tune(yv.astype(int), pv)
            oof[node_fold == k] = pt
        r = summarise(y_all, oof, thr, node_fold)
        res.append(r)
        print(f"[{tag} seed {seed}] F1 {r['f1']:.3f} PR {r['pr_auc']:.3f} FN {r['fn']} FP {r['fp']}", flush=True)
        if seed == seeds[0]:
            np.savez(OUT / f"isot_cv_oof_{tag}.npz", y=y_all, proba=oof, fold=node_fold)
    return {tag: res}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=("trees", "gine"), required=True)
    ap.add_argument("--tag", default="gine")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 1, 2])
    a = ap.parse_args()
    graphs = load()
    print(f"{SC}: {len(graphs)} windows, {sum(int(g.y.sum()) for g in graphs)} bot host-windows", flush=True)
    res = part_trees(graphs, a.seeds) if a.part == "trees" else part_gine(graphs, a.seeds, a.tag)
    name = "trees" if a.part == "trees" else a.tag
    (OUT / f"isot_cv_{name}.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
