"""Flow-record loss robustness for HiGT-Bot v2 (review item M6/M7).

The earlier edge-drop test only perturbed the group block, because the host features x
come from flow records. Here a fraction r of the window's flow RECORDS is dropped before
anything is computed: x is recomputed from the surviving records that touch each host
(same rows as scripts/refeature_graphs.py), an edge of the window graph survives if at
least one of its between-host records survives, and grp is recomputed from the new x and
edges. The node set is kept fixed (graph construction is not re-run), so hosts that would
lose their place in the graph are not simulated. Models are trained once on the clean
training split (XGB[x], XGB[x+grp], seeds 42, 1, 2; threshold from the clean validation set).

    uv run python -m scripts.revision_flow_sampling
Output: data/inspection_logs/revision/flow_sampling/{<scenario>.npz, results.json, results.md}
"""
from __future__ import annotations

import os

os.environ.setdefault("HIGT_GRAPHS_DIR", "data/graphs_v3")
import torch  # noqa: E402

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from sklearn.metrics import average_precision_score

from scripts.refeature_graphs import _SequentialWindows
from scripts.revision_trees import f1, make, tune
from src.data.build_graphs_streaming import FLOW_COLUMNS, _window_filter
from src.data.dataset import SplitSpec, load_split_files
from src.data.features import compute_node_features
from src.features.host_blocks import fit_group_scaler, grp_block

OUT = Path("data/inspection_logs/revision/flow_sampling")
RATES = (0.0, 0.05, 0.1, 0.2, 0.3)


def features_for_scenario(sc, files, qt):
    f = OUT / f"{sc}.npz"
    if f.exists():
        return
    pq_path = Path("data/processed") / f"{sc}.parquet"
    dataset = ds.dataset(pq_path, format="parquet")
    reader = _SequentialWindows(pq.ParquetFile(pq_path)) if pq_path.stat().st_size > 100e6 else None
    rng = np.random.default_rng(42)
    X = {r: [] for r in RATES}; Gp = {r: [] for r in RATES}; Y = []; mismatch = 0
    for fp in files:
        g = torch.load(fp, weights_only=False)
        t0 = pd.Timestamp(g.window_start)
        t0 = t0.tz_localize("UTC") if t0.tzinfo is None else t0
        t1 = t0 + pd.Timedelta(seconds=300)
        flows = reader.window(t0, t1) if reader is not None else             dataset.to_table(filter=_window_filter(t0, t1, drop_background=True), columns=FLOW_COLUMNS).to_pandas()
        nodes = list(g.node_ips); ns = set(nodes)
        touching = flows[flows["src_ip"].isin(ns) | flows["dst_ip"].isin(ns)].reset_index(drop=True)
        u = rng.random(len(touching))
        if not np.allclose(compute_node_features(touching, nodes, 300), g.x.numpy(), rtol=1e-4, atol=1e-4):
            mismatch += 1  # recomputation from the parquet must reproduce the stored x
        ei = g.edge_index.numpy(); idx = {ip: i for i, ip in enumerate(nodes)}
        for r in RATES:
            keep = touching[u >= r].reset_index(drop=True)
            x = compute_node_features(keep, nodes, 300).astype(np.float32) if r > 0 else g.x.numpy().astype(np.float32)
            if r > 0:
                btw = keep[keep["src_ip"].isin(ns) & keep["dst_ip"].isin(ns)]
                pairs = {(idx[a], idx[b]) for a, b in zip(btw["src_ip"], btw["dst_ip"])}
                m = np.array([(int(a), int(b)) in pairs or (int(b), int(a)) in pairs for a, b in zip(ei[0], ei[1])], bool) \
                    if ei.shape[1] else np.zeros(0, bool)
                e = ei[:, m]
            else:
                e = ei
            X[r].append(x); Gp[r].append(grp_block(x, e, qt))
        Y.append(g.y.numpy())
    np.savez(f, y=np.concatenate(Y), mismatch=mismatch, **{f"x_{r}": np.vstack(X[r]) for r in RATES}, **{f"grp_{r}": np.vstack(Gp[r]) for r in RATES})
    print(sc, "done, windows whose x was not reproduced:", mismatch, "of", len(files), flush=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    spec = SplitSpec.load()
    T = np.load("data/inspection_logs/revision/tables.npz", allow_pickle=True)
    qt = fit_group_scaler(T["train_x"])
    files = load_split_files("test", spec)
    by_sc = {}
    for fp in files:
        by_sc.setdefault(fp.parent.name, []).append(fp)
    for sc, fl in by_sc.items():
        features_for_scenario(sc, fl, qt)
    D = {sc: np.load(OUT / f"{sc}.npz") for sc in by_sc}
    y = np.concatenate([D[sc]["y"] for sc in by_sc])
    assert len(y) == len(T["test_y"]) and (y == T["test_y"]).all(), "row alignment with tables.npz"
    res = {}
    for name, bl in (("XGB[x]", ["x"]), ("v2 XGB[x+grp]", ["x", "grp"])):
        Xtr = np.hstack([T[f"train_{b}"] for b in bl]); ytr = T["train_y"]
        Xva = np.hstack([T[f"val_{b}"] for b in bl])
        res[name] = {}
        for seed in (42, 1, 2):
            clf = make("xgb", ytr, seed).fit(Xtr, ytr)
            thr = tune(T["val_y"], clf.predict_proba(Xva)[:, 1])
            for r in RATES:
                Xte = np.hstack([np.concatenate([D[sc][f"{b}_{r}"] for sc in by_sc]) for b in bl])
                p = clf.predict_proba(Xte)[:, 1]; pred = (p >= thr).astype(int)
                res[name].setdefault(str(r), []).append({"f1": f1(y, pred), "pr_auc": float(average_precision_score(y, p)),
                                                         "fn": int(((pred == 0) & (y == 1)).sum()),
                                                         "fp": int(((pred == 1) & (y == 0)).sum())})
            print(name, seed, "done", flush=True)
    (OUT / "results.json").write_text(json.dumps(res, indent=2))
    md = ["# Flow-record loss (records dropped before features and edges are computed; node set fixed)", "",
          "| Records dropped | " + " | ".join(f"{m} F1" for m in res) + " | " + " | ".join(f"{m} PR-AUC" for m in res) + " |",
          "|---|" + "---|" * (2 * len(res))]
    for r in RATES:
        cells = [f"{np.mean([v['f1'] for v in res[m][str(r)]]):.3f} ± {np.std([v['f1'] for v in res[m][str(r)]], ddof=1):.3f}" for m in res]
        cells += [f"{np.mean([v['pr_auc'] for v in res[m][str(r)]]):.4f}" for m in res]
        md.append(f"| {int(r * 100)}% | " + " | ".join(cells) + " |")
    (OUT / "results.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
