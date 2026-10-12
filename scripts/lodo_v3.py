"""Leave-one-dataset-out (LODO) on the corrected data (graphs_v3: v2 labels + node
features from all window flows).

Splits: configs/split_lodo_{ctu13,iot23,medbiot}.yaml (train/val = per-scenario
chronological 80/20 of the two source datasets; test = every window of the held-out
dataset). Models:
  RF tuned      RandomForest(500, class_weight=balanced) on x, as scripts/baselines/run_rf.py
  XGB[x]        XGBoost on the 9 host features, as scripts/higt_v2_trees.py
  HiGT-Bot v2   XGBoost on x + grp (grp's quantile scaler fitted on the source train x)
  T-GINE        directed temporal GINE + skip (scripts/train_phase5.py --no-amp, seed 42,
                trained by scripts/queue_F_lodo_v3.sh), re-predicted here in FP32
Trees: seeds 42, 1, 2. Every model: threshold tuned on the SOURCE validation F1, plus
threshold-free PR-AUC and the target prevalence; the "oracle" F1 (best threshold on
the target itself) is reported only as an upper bound for the threshold transfer.

    uv run python scripts/lodo_v3.py --part trees    # CPU
    uv run python scripts/lodo_v3.py --part tgine    # after the T-GINE trainings
    uv run python scripts/lodo_v3.py --part report   # md + figure + tables_v3.md section
Output: data/inspection_logs/v3_lodo/{lodo_<holdout>.json, lodo_v3.md, pred_*.npz},
        ../thesis_figures_v3/fig7_lodo_v3.png
"""
from __future__ import annotations

import os

os.environ.setdefault("HIGT_GRAPHS_DIR", "data/graphs_v3")
os.environ.setdefault("HIGT_DIRECTED", "1")
import torch  # noqa: E402  (import before pandas on Windows)

import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score

from scripts.higt_v2_trees import fit_predict, tune
from src.data.dataset import SplitSpec, load_split_files
from src.features.host_blocks import fit_group_scaler, grp_block

OUT = Path("data/inspection_logs/v3_lodo")
EXP = Path("experiments/v3_lodo")
FIG = Path("../thesis_figures_v3")
HOLDOUTS = ("ctu13", "iot23", "medbiot")
DNAME = {"ctu13": "CTU-13", "iot23": "IoT-23", "medbiot": "MedBIoT"}
SEEDS = (42, 1, 2)
GRID = np.linspace(0.01, 0.99, 99)
MODELS = ("RF tuned", "XGB[x]", "HiGT-Bot v2 (x+grp)", "Directed T-GINE")


def spec_for(h):
    return SplitSpec.load(Path(f"configs/split_lodo_{h}.yaml"))


def metrics(y, p, thr):
    pred = (p >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    f1 = lambda t: 2 * int(((p >= t) & (y == 1)).sum()) / max(int((p >= t).sum()) + int(y.sum()), 1)
    return {"f1": 2 * tp / max(2 * tp + fp + fn, 1), "fn": fn, "fp": fp, "tp": tp, "threshold": float(thr),
            "pr_auc": float(average_precision_score(y, p)) if 0 < y.sum() < len(y) else float("nan"),
            "oracle_f1": float(max(f1(t) for t in GRID)), "n": int(len(y)), "n_pos": int(y.sum()),
            "prevalence": float(y.mean())}


def evaluate_split(yva, pva, yte, pte, scen):
    thr = tune(yva, pva)
    per = {s: metrics(yte[scen == s], pte[scen == s], thr) for s in sorted(set(scen))}
    return {"all": metrics(yte, pte, thr), "per_scenario": per, "val_f1": metrics(yva, pva, thr)["f1"]}


def tables(h):
    """x, grp, y, scenario per split; grp scaler fitted on this direction's source train x."""
    path = OUT / f"cache_{h}.npz"
    if path.exists():
        z = np.load(path, allow_pickle=True)
        return {s: {k: z[f"{s}_{k}"] for k in ("x", "grp", "y", "scenario")} for s in ("train", "val", "test")}
    spec = spec_for(h)
    files = {s: load_split_files(s, spec) for s in ("train", "val", "test")}
    # graphs are streamed one at a time (RAM is shared with the T-GINE trainings)
    qt = fit_group_scaler(np.vstack([torch.load(fp, weights_only=False).x.numpy() for fp in files["train"]]))
    T = {}
    for s, fl in files.items():
        t0 = time.time()
        B = {"x": [], "grp": [], "y": [], "scenario": []}
        for fp in fl:
            g = torch.load(fp, weights_only=False)
            x = g.x.numpy().astype(np.float32)
            B["x"].append(x); B["grp"].append(grp_block(x, g.edge_index.numpy(), qt))
            B["y"].append(g.y.numpy().astype(int)); B["scenario"].append(np.full(int(g.num_nodes), g.scenario))
        T[s] = {"x": np.vstack(B["x"]), "grp": np.vstack(B["grp"]), "y": np.concatenate(B["y"]),
                "scenario": np.concatenate(B["scenario"]).astype(str)}
        print(f"[{h}] {s}: {len(fl)} windows, {len(T[s]['y'])} nodes, {T[s]['y'].sum()} bots ({time.time() - t0:.0f}s)",
              flush=True)
    np.savez(path, **{f"{s}_{k}": v for s, d in T.items() for k, v in d.items()})
    return T


def rf_fit_predict(Xtr, ytr, Xva, Xte, seed):
    clf = RandomForestClassifier(n_estimators=500, class_weight="balanced", random_state=seed, n_jobs=-1).fit(Xtr, ytr)
    return clf.predict_proba(Xva)[:, 1], clf.predict_proba(Xte)[:, 1]


def part_trees():
    for h in HOLDOUTS:
        T = tables(h)
        res = {}
        for name, blocks, fn in (("RF tuned", ["x"], rf_fit_predict), ("XGB[x]", ["x"], fit_predict),
                                 ("HiGT-Bot v2 (x+grp)", ["x", "grp"], fit_predict)):
            X = {s: np.hstack([T[s][b] for b in blocks]) for s in T}
            res[name] = []
            for seed in SEEDS:
                t0 = time.time()
                pva, pte = fn(X["train"], T["train"]["y"], X["val"], X["test"], seed)
                r = evaluate_split(T["val"]["y"], pva, T["test"]["y"], pte, T["test"]["scenario"])
                res[name].append(r | {"seed": seed, "fit_s": time.time() - t0})
                if seed == SEEDS[0]:
                    np.savez(OUT / f"pred_{h}_{name.split(' ')[0].replace('[', '_').replace(']', '')}.npz",
                             y=T["test"]["y"], proba=pte, scenario=T["test"]["scenario"])
                a = r["all"]
                print(f"[{h}] {name:22s} seed {seed}: F1 {a['f1']:.3f} PR {a['pr_auc']:.3f} FN {a['fn']} FP {a['fp']} "
                      f"thr {a['threshold']:.2f} oracle {a['oracle_f1']:.3f}", flush=True)
        prev = OUT / f"lodo_{h}.json"
        old = json.loads(prev.read_text()) if prev.exists() else {}
        prev.write_text(json.dumps(old | res, indent=2))


@torch.no_grad()
def part_tgine():
    from torch_geometric.loader import DataLoader
    from scripts.reeval_mixed_pool_fp32 import _predict_fp32
    from src.data.flow_seq_dataset import FlowSeqGraphDataset
    from src.models.hybrid import TemporalGINE
    from src.utils.seeding import pick_device
    device = pick_device()
    for h in HOLDOUTS:
        ckpt = EXP / h / "phase5" / f"temporal_gine_lodo_{h}_skip.pt"
        if not ckpt.exists():
            print(f"[{h}] missing {ckpt}", flush=True); continue
        blob = torch.load(ckpt, map_location="cpu", weights_only=False); cfg = blob["cfg"]
        e_mean, e_std = blob["edge_scaler"]; n_mean, n_std = blob["node_scaler"]
        kw = dict(edge_mean=e_mean.cpu(), edge_std=e_std.cpu(), node_mean=n_mean.cpu(), node_std=n_std.cpu())
        spec = spec_for(h); va_files = load_split_files("val", spec); te_files = load_split_files("test", spec)
        model = TemporalGINE(flow_feat_dim=cfg["encoder"]["flow_feat_dim"],
                             edge_dim=int(torch.load(te_files[0], weights_only=False).edge_attr.size(1)),
                             d_model=cfg["encoder"]["d_model"], nhead=cfg["encoder"]["nhead"],
                             num_layers=cfg["encoder"]["num_layers"], max_flows=cfg["encoder"]["max_flows"],
                             encoder_dropout=0.0, gin_hidden=cfg["gin"]["hidden"], gin_layers=cfg["gin"]["num_layers"],
                             dropout=0.0, out_dim=2, raw_feat_dim=blob.get("raw_feat_dim")).to(device)
        model.load_state_dict(blob["state_dict"])
        bs = int(cfg["train"]["batch_size"])
        yv, _, pv, _ = _predict_fp32(model, DataLoader(FlowSeqGraphDataset(va_files, **kw), batch_size=bs), device)
        yt, _, pt, st = _predict_fp32(model, DataLoader(FlowSeqGraphDataset(te_files, **kw), batch_size=bs), device)
        st = st.astype(str)
        r = evaluate_split(yv.astype(int), pv, yt.astype(int), pt, st) | {"seed": 42, "fp32": True}
        np.savez(OUT / f"pred_{h}_TGINE.npz", y=yt, proba=pt, scenario=st)
        a = r["all"]
        print(f"[{h}] Directed T-GINE: F1 {a['f1']:.3f} PR {a['pr_auc']:.3f} FN {a['fn']} FP {a['fp']} "
              f"thr {a['threshold']:.2f} oracle {a['oracle_f1']:.3f} nan-proba {int(np.isnan(pt).sum())}", flush=True)
        prev = OUT / f"lodo_{h}.json"
        old = json.loads(prev.read_text()) if prev.exists() else {}
        prev.write_text(json.dumps(old | {"Directed T-GINE": [r]}, indent=2))
        del model; torch.cuda.empty_cache()


def old_label_rows():
    """Old-label LODO (v0 labels, graphs v1 features): F1 at 0.5 and PR-AUC from the stored JSONs."""
    rows = {}
    for h in HOLDOUTS:
        for name, path in (("RF", f"experiments/lodo/{h}/phase4/baseline_rf_lodo_{h}.json"),
                           ("XGB[x]", f"experiments/lodo/{h}/phase4/baseline_xgb_lodo_{h}.json"),
                           ("T-GINE (undirected)", f"data/inspection_logs/phase5_temporal_gine_lodo_{h}"
                                                   f"{'_fp32' if h == 'iot23' else ''}_skip.json")):
            p = Path(path)
            if p.exists():
                t = json.loads(p.read_text())["test"]
                rows[(h, name)] = {"f1": t["f1"], "pr_auc": t["pr_auc"], "prevalence": t["n_pos"] / t["n"]}
    return rows


def msd(v, d=3):
    v = [x for x in v if x == x]
    return f"{np.mean(v):.{d}f}" + (f" ± {np.std(v, ddof=1):.{d}f}" if len(v) > 1 else "")


def part_report():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "figure.dpi": 200, "savefig.bbox": "tight"})
    R = {h: json.loads((OUT / f"lodo_{h}.json").read_text()) for h in HOLDOUTS if (OUT / f"lodo_{h}.json").exists()}
    tab = ["| Target (held out) | Prevalence | Model | F1 @ source-val threshold | PR-AUC | FN | FP | Oracle F1 |",
           "|---|---|---|---|---|---|---|---|"]
    for h, res in R.items():
        for m in MODELS:
            if m not in res:
                continue
            runs = [r["all"] for r in res[m]]
            tab.append(f"| {DNAME[h]} ({runs[0]['n_pos']:,} / {runs[0]['n']:,}) | {runs[0]['prevalence']:.3f} | {m}"
                       f"{' (1 seed)' if len(runs) == 1 else ''} | {msd([r['f1'] for r in runs])} | "
                       f"{msd([r['pr_auc'] for r in runs])} | {msd([r['fn'] for r in runs], 0)} | "
                       f"{msd([r['fp'] for r in runs], 0)} | {msd([r['oracle_f1'] for r in runs])} |")
    per = ["| Target scenario | Bots / host-windows | " + " | ".join(MODELS) + " |", "|---|---|" + "---|" * len(MODELS)]
    for h, res in R.items():
        ref = res[MODELS[1]][0]["per_scenario"]
        for s in ref:
            cells = []
            for m in MODELS:
                if m in res:
                    cells.append(msd([r["per_scenario"][s]["f1"] for r in res[m]]) + " / "
                                 + msd([r["per_scenario"][s]["pr_auc"] for r in res[m]]))
                else:
                    cells.append("—")
            per.append(f"| {s} | {ref[s]['n_pos']} / {ref[s]['n']} | " + " | ".join(cells) + " |")
    old = old_label_rows()
    cmp_ = ["| Target | Model | Old labels: prevalence | Old: F1 @ 0.5 | Old: PR-AUC | Corrected: prevalence | "
            "Corrected: F1 (source-val thr.) | Corrected: PR-AUC |", "|---|---|---|---|---|---|---|---|"]
    for h, res in R.items():
        for om, nm in (("RF", "RF tuned"), ("XGB[x]", "XGB[x]"), ("T-GINE (undirected)", "Directed T-GINE")):
            if (h, om) in old and nm in res:
                o = old[(h, om)]; runs = [r["all"] for r in res[nm]]
                cmp_.append(f"| {DNAME[h]} | {om} → {nm} | {o['prevalence']:.3f} | {o['f1']:.3f} | {o['pr_auc']:.3f} | "
                            f"{runs[0]['prevalence']:.3f} | {msd([r['f1'] for r in runs])} | {msd([r['pr_auc'] for r in runs])} |")
    body = ["Train and validate on the two source datasets (per-scenario chronological 80/20), test on every window "
            "of the held-out dataset. Corrected labels (bots = originators of bot traffic) and host features from all "
            "window flows (data/graphs_v3). Trees: mean ± sd over seeds 42, 1, 2; T-GINE: directed, seed 42, trained "
            "and evaluated in FP32. The decision threshold is tuned on the source validation set only; the oracle F1 "
            "(best threshold chosen on the target itself) is an upper bound, not a usable result.", "",
            *tab, "", "Per target scenario, F1 / PR-AUC:", "", *per, "",
            "Old-label LODO (any-flow labels, subgraph-restricted features; stored results) vs corrected:", "", *cmp_]
    md = ["# Leave-one-dataset-out on corrected labels (v3)", "", *body]
    (OUT / "lodo_v3.md").write_text("\n".join(md), encoding="utf-8")
    # figure: PR-AUC and F1 per target, per model
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    cols = {"RF tuned": "#9bbb59", "XGB[x]": "#4f81bd", "HiGT-Bot v2 (x+grp)": "#1f497d", "Directed T-GINE": "#f79646"}
    hs = [h for h in HOLDOUTS if h in R]; xs = np.arange(len(hs)); w = 0.2
    for ax, key, lab in ((axes[0], "pr_auc", "PR-AUC (threshold-free)"), (axes[1], "f1", "F1 at source-val threshold")):
        for i, m in enumerate(MODELS):
            v = [[r["all"][key] for r in R[h][m]] if m in R[h] else [np.nan] for h in hs]
            ax.bar(xs + (i - 1.5) * w, [np.nanmean(a) for a in v], w, color=cols[m], label=m,
                   yerr=[np.std(a, ddof=1) if len(a) > 1 else 0 for a in v], capsize=2)
        if key == "pr_auc":
            ax.scatter(xs, [R[h][MODELS[1]][0]["all"]["prevalence"] for h in hs], marker="_", s=400, color="k",
                       zorder=3, label="prevalence (no-skill PR-AUC)")
        ax.set_xticks(xs, [f"test on {DNAME[h]}" for h in hs], fontsize=8); ax.set_ylim(0, 1.05)
        ax.set_ylabel(lab); ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=7, loc="lower left")
    fig.tight_layout(); fig.savefig(FIG / "fig7_lodo_v3.png"); plt.close(fig)
    # append to tables_v3.md (replace an earlier copy of the section)
    tv = FIG / "tables_v3.md"; txt = tv.read_text(encoding="utf-8"); head = "## 7.x LODO, corrected labels"
    if head in txt:
        txt = txt[:txt.index(head)].rstrip() + "\n"
    tv.write_text(txt.rstrip() + "\n\n" + head + "\n\nSource: `data/inspection_logs/v3_lodo/lodo_v3.md` "
                  "(`scripts/lodo_v3.py`). Figure: `fig7_lodo_v3.png`.\n\n" + "\n".join(body) + "\n", encoding="utf-8")
    print("\n".join(md))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=("trees", "tgine", "report"), required=True)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    {"trees": part_trees, "tgine": part_tgine, "report": part_report}[args.part]()


if __name__ == "__main__":
    main()
