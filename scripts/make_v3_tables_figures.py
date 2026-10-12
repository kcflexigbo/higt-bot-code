"""Tables and figures for the rewritten paper / thesis (corrected labels, v3 features).

Reads only existing results (no training):
  data/inspection_logs/{relabel_infected_src_v2.json, refeature_v3.json}
  data/inspection_logs/v3/{higt_v2_trees.json, bootstrap_significance.json, pred_*.npz}
  data/inspection_logs/{v2,v3_gnn}/predictions/*.npz and v2/v3_gnn training logs
  data/graphs (old labels) and data/graphs_v3 (corrected labels, fixed features)
Writes ../thesis_figures_v3/*.png and ../thesis_figures_v3/tables_v3.md.

    uv run python scripts/make_v3_tables_figures.py
"""
from __future__ import annotations

import os

os.environ.setdefault("HIGT_GRAPHS_DIR", "data/graphs_v3")
import torch  # noqa: E402  (import before pandas on Windows)

import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import average_precision_score  # noqa: E402

from src.data.dataset import SplitSpec, load_split_files  # noqa: E402

LOGS = Path("data/inspection_logs")
OUT = Path("../thesis_figures_v3")
GRID = np.linspace(0.01, 0.99, 99)
DATASETS = ("ctu13", "iot23", "medbiot")
DNAME = {"ctu13": "CTU-13", "iot23": "IoT-23", "medbiot": "MedBIoT"}
plt.rcParams.update({"font.size": 10, "figure.dpi": 200, "savefig.bbox": "tight"})


def f1c(y, p):
    tp = int(((p == 1) & (y == 1)).sum()); fp = int(((p == 1) & (y == 0)).sum()); fn = int(((p == 0) & (y == 1)).sum())
    return 2 * tp / max(2 * tp + fp + fn, 1), fn, fp


def load_pred(path: Path):
    d = np.load(path, allow_pickle=True)
    y, pr, pred = d["y"].astype(int), d["proba"].astype(float), d["pred"].astype(int)
    vp = path.with_name(path.stem + "_val.npz")
    if vp.exists():
        v = np.load(vp, allow_pickle=True)
        t = GRID[int(np.argmax([f1c(v["y"].astype(int), (v["proba"] >= g).astype(int))[0] for g in GRID]))]
        pred = (pr >= t).astype(int)
    return y, pred, pr, d["scenario"].astype(str)


def scores(y, pred, pr):
    f, fn, fp = f1c(y, pred)
    return {"f1": f, "pr": float(average_precision_score(y, pr)) if 0 < y.sum() < len(y) else float("nan"), "fn": fn, "fp": fp}


def msd(vals, d=3):
    vals = [v for v in vals if v == v]
    return f"{np.mean(vals):.{d}f}" + (f" ± {np.std(vals, ddof=1):.{d}f}" if len(vals) > 1 else "")


def log_test_confusion(path: Path, n_test=59210):
    """(tn, fp, fn, tp) of the test block in a train_phase7 log."""
    txt = path.read_text(encoding="utf-8", errors="ignore")
    for tn, fp, fn, tp in re.findall(r"tn=(\d+) fp=(\d+)\]\s*\[fn=(\d+) tp=(\d+)", txt):
        tn, fp, fn, tp = map(int, (tn, fp, fn, tp))
        if tn + fp + fn + tp == n_test:
            return tn, fp, fn, tp
    raise ValueError(f"no test confusion in {path}")


# ---------------------------------------------------------------- tables
def t_label_counts(md):
    d = json.loads((LOGS / "relabel_infected_src_v2.json").read_text())
    agg = defaultdict(lambda: [0, 0, 0, 0])
    for sc, v in d.items():
        k = sc.split("-")[0]
        a = agg[k]; a[0] += v["windows"]; a[1] += v["nodes"]; a[2] += v["bot_any_endpoint"]; a[3] += v["bot_infected_src"]
    md += ["## Table L1. Positive host-windows under the old and the corrected labelling rule (all 40 scenarios)", "",
           "| Dataset | Windows | Host-windows | Bots, old rule (any endpoint) | Bots, corrected (originator) | Old / corrected |",
           "|---|---:|---:|---:|---:|---:|"]
    tot = [0, 0, 0, 0]
    for k in DATASETS:
        a = agg[k]; tot = [x + y for x, y in zip(tot, a)]
        md.append(f"| {DNAME[k]} | {a[0]:,} | {a[1]:,} | {a[2]:,} | {a[3]:,} | {a[2] / max(a[3], 1):.0f}× |")
    md += [f"| **All** | **{tot[0]:,}** | **{tot[1]:,}** | **{tot[2]:,}** | **{tot[3]:,}** | **{tot[2] / tot[3]:.0f}×** |", "",
           "Old rule: a host is a bot if it is the source or destination of any bot flow in the window. Corrected rule: "
           "a host is a bot if it originates a bot flow (MedBIoT: testbed devices only, C&C hosts excluded).", ""]
    return agg


def t_split(md, spec):
    md += ["## Table 3.6 (corrected). Split composition with corrected labels", "",
           "| Dataset | Split | Windows | Host-windows | Bots | Prevalence | Old-rule bots |", "|---|---|---:|---:|---:|---:|---:|"]
    rows = defaultdict(lambda: [0, 0, 0, 0])
    for s in ("train", "val", "test"):
        for fp in load_split_files(s, spec):
            g = torch.load(fp, weights_only=False)
            g0 = torch.load(Path("data/graphs") / fp.parent.name / fp.name, weights_only=False)
            r = rows[(g.scenario.split("-")[0], s)]
            r[0] += 1; r[1] += int(g.num_nodes); r[2] += int(g.y.sum()); r[3] += int(g0.y.sum())
    tot = [0, 0, 0, 0]
    for k in DATASETS:
        for s in ("train", "val", "test"):
            r = rows[(k, s)]
            if s == "test":
                tot = [a + b for a, b in zip(tot, r)]
            md.append(f"| {DNAME[k] if s == 'train' else ''} | {s.capitalize()} | {r[0]:,} | {r[1]:,} | {r[2]:,} | "
                      f"{r[2] / max(r[1], 1):.4f} | {r[3]:,} |")
    md += [f"| **All** | **Test** | **{tot[0]:,}** | **{tot[1]:,}** | **{tot[2]:,}** | **{tot[2] / tot[1]:.4f}** | **{tot[3]:,}** |", "",
           "Test includes the held-out scenarios (ctu13-10, iot23-8-1, medbiot-torii).", ""]


def model_rows(spec):
    """name -> list of per-seed (y, pred, proba, scenario) on the v3 test set."""
    g3, t3 = LOGS / "v3_gnn/predictions", LOGS / "v3"
    groups = {
        "Random forest (tuned)": [g3 / "rf_tuned.npz"],
        "GINE, undirected": [g3 / "gine_seed42.npz"],
        "GINE, directed": [g3 / f"dirgine_seed{s}.npz" for s in (42, 1, 2)],
        "T-GINE, directed": [g3 / "t_gine_skip.npz"],
        "SAGPool, directed": [g3 / "sagpool.npz"],
        "HiGT-Bot (original), directed": [g3 / f"higt_bot_edge{s}.npz" for s in ("", "_seed1", "_seed2")],
        "XGBoost, host features (x)": [t3 / "pred_x.npz"],
        "HiGT-Bot v2 (x + grp)": [t3 / "pred_x_grp.npz"],
    }
    return {k: [load_pred(p) for p in v if p.exists()] for k, v in groups.items()}


def t_main(md, M, spec, trees):
    md += ["## Table 5.1 (v3). Main comparison on the test split (corrected labels and features)", "",
           "Thresholds tuned on validation F1 for every model. Multi-seed rows: mean ± sd. Tree-head rows use the 3-seed "
           "results of `v3_emb3/higt_v2_trees.json`.", "",
           "| Model | Seeds | F1 | PR-AUC | Missed bots (FN) | False alarms (FP) | Held-out F1 | ctu13-10 F1 |",
           "|---|:-:|---|---|---|---|---|---|"]
    for name, runs in M.items():
        tk = {"XGBoost, host features (x)": "x", "HiGT-Bot v2 (x + grp)": "x+grp"}.get(name)
        if tk:
            r = trees["split"][tk]
            g = lambda sub, k: [x[sub][k] for x in r]
            md.append(f"| **{name}** | {len(r)} | {msd(g('all', 'f1'))} | {msd(g('all', 'pr_auc'))} | {msd(g('all', 'fn'), 0)} | "
                      f"{msd(g('all', 'fp'), 0)} | {msd(g('held-out', 'f1'))} | {msd(g('ctu13-10', 'f1'))} |"
                      if "v2" in name else
                      f"| {name} | {len(r)} | {msd(g('all', 'f1'))} | {msd(g('all', 'pr_auc'))} | {msd(g('all', 'fn'), 0)} | "
                      f"{msd(g('all', 'fp'), 0)} | {msd(g('held-out', 'f1'))} | {msd(g('ctu13-10', 'f1'))} |")
            continue
        S = []
        for y, pred, pr, sc in runs:
            h = np.isin(sc, spec.holdout_test_scenarios); c = sc == "ctu13-10"
            S.append(scores(y, pred, pr) | {"h": f1c(y[h], pred[h])[0], "c": f1c(y[c], pred[c])[0]})
        g = lambda k: [s[k] for s in S]
        md.append(f"| {name} | {len(S)} | {msd(g('f1'))} | {msd(g('pr'))} | {msd(g('fn'), 0)} | {msd(g('fp'), 0)} | "
                  f"{msd(g('h'))} | {msd(g('c'))} |")
    md.append("")


def t_per_scenario(md, M, spec):
    first = {k: v[0] for k, v in M.items() if v}
    y, _, _, sc = next(iter(first.values()))
    names = list(first)
    short = ["RF", "GINE", "dGINE", "dT-GINE", "dSAGPool", "HiGT-Bot", "XGB[x]", "**v2**"]
    for metric in ("F1", "PR-AUC"):
        md += [f"## Table A.{1 if metric == 'F1' else 2} (v3). Test {metric} per scenario (seed-42 runs; † held out)", "",
               "| Scenario | n | Bots | " + " | ".join(short) + " |", "|---|---:|---:|" + ":-:|" * len(names)]
        for s in sorted(np.unique(sc)):
            m = sc == s; ys = y[m]
            vals = []
            for k in names:
                _, pred, pr, _ = first[k]
                if metric == "F1":
                    vals.append(f1c(ys, pred[m])[0] if ys.sum() else float("nan"))
                else:
                    vals.append(average_precision_score(ys, pr[m]) if 0 < ys.sum() < len(ys) else float("nan"))
            best = np.nanmax(vals) if not all(v != v for v in vals) else None
            cells = ["–" if v != v else (f"**{v:.3f}**" if best is not None and abs(v - best) < 1e-9 else f"{v:.3f}") for v in vals]
            tag = " †" if s in spec.holdout_test_scenarios else ""
            md.append(f"| {s}{tag} | {m.sum():,} | {int(ys.sum())} | " + " | ".join(cells) + " |")
        md += ["", "– : no bots (F1) or a single class (PR-AUC) in this scenario's test portion.", ""]


def t_significance(md):
    b = json.loads((LOGS / "v3/bootstrap_significance.json").read_text())
    keep = ["XGB[x] (v3 features)", "RF tuned (v3 features)", "Directed GINE (v3)", "Directed T-GINE (v3)", "HiGT-Bot directed (v3)"]
    md += ["## Table 5.3 (v3). HiGT-Bot v2 (x + grp) minus comparator: paired window bootstrap", "",
           "2,000 resamples of test windows, stratified by scenario; 95% percentile CI; p = share of resamples with "
           "difference ≤ 0 (one-sided).", "",
           "| Comparator | Subset | ΔF1 [95% CI] | p | ΔPR-AUC [95% CI] | p |", "|---|---|---|---|---|---|"]
    for c in keep:
        for sub in ("all", "held-out", "ctu13-10"):
            v = b[sub]["pairs"].get(f"HiGT-Bot v2 (x+grp) - {c}")
            if not v:
                continue
            f = lambda m: f"{v[m]['delta']:+.3f} [{v[m]['ci95'][0]:+.3f}, {v[m]['ci95'][1]:+.3f}] | {v[m]['p_one_sided']:.3f}"
            md.append(f"| {c if sub == 'all' else ''} | {sub} | {f('f1')} | {f('pr_auc')} |")
    md.append("")


def t_tree_ablation(md, trees):
    md += ["## Table 5.4 (v3). Feature-block ablation of the tree head (3 seeds; thesis split)", "",
           "x = 9 host features, seq = flow-sequence statistics, grp = in-window group similarity, nbr = direction-aware "
           "neighbour aggregates, emb = learned temporal-encoder embedding (directed T-GINE retrained on the corrected data).", "",
           "| Blocks | F1 | PR-AUC | FN | FP | Held-out F1 | ctu13-10 F1 | LOSO mean PR-AUC | LOSO mean F1 |",
           "|---|---|---|---|---|---|---|---|---|"]
    for c, r in trees["split"].items():
        g = lambda sub, k: [x[sub][k] for x in r]
        lo = trees["loso"].get(c, {})
        lp = np.nanmean([v["pr_auc"] for v in lo.values()]) if lo else float("nan")
        lf = np.mean([v["f1"] for v in lo.values()]) if lo else float("nan")
        md.append(f"| {c} | {msd(g('all', 'f1'))} | {msd(g('all', 'pr_auc'))} | {msd(g('all', 'fn'), 0)} | {msd(g('all', 'fp'), 0)} | "
                  f"{msd(g('held-out', 'f1'))} | {msd(g('ctu13-10', 'f1'))} | {lp:.3f} | {lf:.3f} |")
    md.append("")


def t_direction(md):
    md += ["## Table D1. Message direction, before and after the feature fix (seed 42)", "",
           "v2 = corrected labels, old node features (between-node flows only); v3 = corrected labels, fixed features. "
           "HiGT-Bot rows: test confusion at the trainer's threshold from the training logs; GINE rows: threshold tuned "
           "on validation.", "",
           "| Model | Features | Undirected F1 | Directed F1 | Undirected FN / FP | Directed FN / FP |", "|---|---|---|---|---|---|"]
    for tag, d in (("v2 (old features)", LOGS / "v2"), ("v3 (fixed features)", LOGS / "v3_gnn")):
        u = log_test_confusion(d / "edge_undirected.log"); di = log_test_confusion(d / "edge.log")
        f = lambda c: 2 * c[3] / (2 * c[3] + c[1] + c[2])
        md.append(f"| HiGT-Bot (original) | {tag} | {f(u):.3f} | {f(di):.3f} | {u[2]} / {u[1]} | {di[2]} / {di[1]} |")
    for tag, d in (("v2 (old features)", LOGS / "v2/predictions"), ("v3 (fixed features)", LOGS / "v3_gnn/predictions")):
        u = scores(*load_pred(d / "gine_seed42.npz")[:3]); di = scores(*load_pred(d / "dirgine_seed42.npz")[:3])
        md.append(f"| GINE (flat) | {tag} | {u['f1']:.3f} | {di['f1']:.3f} | {u['fn']} / {u['fp']} | {di['fn']} / {di['fp']} |")
    md.append("")


def t_label_rankings(md):
    md += ["## Table L2. Test F1 of the same methods under each label version (validation-tuned thresholds)", "",
           "Sources: `tier1_reanalysis.md`, `relabel/`, `v2/`, `v3`, `egraphsage/egraphsage_summary.md`. "
           "v0 = old any-endpoint labels; v1 = first infected-source fix; v2 = corrected labels; v3 = v2 + fixed features.", "",
           "| Method | v0 (old) | v1 | v2 | v3 |", "|---|---|---|---|---|",
           "| Random forest (tuned) | 0.963 | 0.936 | 0.937 | 0.945 |",
           "| XGBoost, host features | — | 0.952 | — | 0.955 |",
           "| E-GraphSAGE (node readout) | 0.857 | 0.751 | 0.759 | — |",
           "| GINE (flat) | 0.955* | 0.897 | — | 0.845 (undirected) / 0.951 (directed) |",
           "| HiGT-Bot (original) | 0.967* | 0.863 ± 0.100 | 0.937 (directed) | 0.940 (directed) |",
           "| HiGT-Bot v2 (x + grp) | — | — | — | **0.978** |", "",
           "\\* threshold 0.5 (as published in the draft). — : not run under that label version. "
           "Under v0 an all-bot classifier scores F1 0.819.", ""]


def t_loso(md):
    txt = (LOGS / "v3_emb3/higt_v2_trees.md").read_text(encoding="utf-8")
    md += ["## Table 6.x (v3). CTU-13 leave-one-scenario-out (from `v3_emb3/higt_v2_trees.md`)", "",
           txt[txt.index("## Leave-one"):].split("\n", 1)[1].strip(), ""]


# ---------------------------------------------------------------- figures
def fig_label_flaw(agg):
    fig, ax = plt.subplots(figsize=(6, 3.2))
    xs = np.arange(len(DATASETS)); w = 0.38
    old = [agg[k][2] for k in DATASETS]; new = [agg[k][3] for k in DATASETS]
    ax.bar(xs - w / 2, old, w, label="old rule (any endpoint)", color="#c0504d")
    ax.bar(xs + w / 2, new, w, label="corrected (originator)", color="#4f81bd")
    for i, (o, n) in enumerate(zip(old, new)):
        ax.text(i - w / 2, o, f"{o:,}", ha="center", va="bottom", fontsize=8)
        ax.text(i + w / 2, n, f"{n:,}", ha="center", va="bottom", fontsize=8)
    ax.set_yscale("log"); ax.set_xticks(xs, [DNAME[k] for k in DATASETS]); ax.set_ylabel("bot host-windows (log scale)")
    ax.legend(frameon=False, fontsize=8); ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("Most 'bots' under the old rule are victims or servers", fontsize=10)
    fig.savefig(OUT / "fig_label_flaw.png"); plt.close(fig)


def fig_feature_fix():
    r = json.loads((LOGS / "refeature_v3.json").read_text())
    agg = defaultdict(lambda: [0, 0, 0, 0])
    for sc, v in r.items():
        a = agg[sc.split("-")[0]]
        a[0] += v["bots_fan_out_zero_before"]; a[1] += v["bots_fan_out_zero_after"]; a[2] += v["nodes_changed"]; a[3] += v["nodes"]
    fig, ax = plt.subplots(1, 2, figsize=(8, 3))
    xs = np.arange(len(DATASETS)); w = 0.38
    ax[0].bar(xs - w / 2, [agg[k][0] for k in DATASETS], w, label="before fix", color="#c0504d")
    ax[0].bar(xs + w / 2, [agg[k][1] for k in DATASETS], w, label="after fix", color="#4f81bd")
    ax[0].set_xticks(xs, [DNAME[k] for k in DATASETS]); ax[0].set_ylabel("bot host-windows")
    for i, k in enumerate(DATASETS):
        ax[0].text(i + w / 2, 2, str(agg[k][1]), ha="center", va="bottom", fontsize=8, color="#4f81bd")
    ax[0].legend(frameon=False, fontsize=8); ax[0].set_title("Bots with fan-out 0 (looked idle)", fontsize=10)
    ax[1].bar(xs, [100 * agg[k][2] / agg[k][3] for k in DATASETS], 0.5, color="#9bbb59")
    ax[1].set_xticks(xs, [DNAME[k] for k in DATASETS]); ax[1].set_ylabel("% of host-windows")
    ax[1].set_title("Host-windows whose features changed", fontsize=10)
    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(OUT / "fig_feature_fix.png"); plt.close(fig)


def fig_real_graph():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
    g0 = torch.load("data/graphs/ctu13-10/window_00056.pt", weights_only=False)
    g3 = torch.load("data/graphs_v3/ctu13-10/window_00056.pt", weights_only=False)
    n = int(g0.num_nodes); ei = g0.edge_index.numpy()
    import networkx as nx  # layout only
    G = nx.Graph(); G.add_nodes_from(range(n))
    und = {(min(a, b), max(a, b)) for a, b in ei.T if a != b}
    G.add_edges_from(und)
    lay = nx.kamada_kawai_layout(G)
    pos = np.array([lay[i] for i in range(n)])
    for ax, g, title in ((axes[0], g0, f"Old rule: {int(g0.y.sum())} 'bots'"),
                         (axes[1], g3, f"Corrected rule: {int(g3.y.sum())} bots")):
        y = g.y.numpy().astype(bool)
        for a, b in und:
            ax.plot(*pos[[a, b]].T, color="#bbbbbb", lw=0.8, zorder=1)
        ax.scatter(*pos[~y].T, s=90, c="#4f81bd", edgecolors="k", lw=0.5, zorder=2, label="benign")
        ax.scatter(*pos[y].T, s=90, c="#c0504d", edgecolors="k", lw=0.5, zorder=2, label="bot")
        for i, ip in enumerate(g.node_ips):
            if ip in ("147.32.80.9", "65.55.56.40"):
                ax.annotate(ip, pos[i], xytext=(6, 6), textcoords="offset points", fontsize=7,
                            fontweight="bold" if y[i] else "normal")
        ax.set_title(title, fontsize=10); ax.axis("off")
    axes[0].legend(frameon=False, loc="lower left", fontsize=8)
    # caption (in the thesis text): CTU-13 scenario 10, window 56, 22 hosts, 29 edges; the two hubs are a DNS server
    # and a Microsoft address, labelled bots under the old rule.
    fig.savefig(OUT / "fig3_2_real_graph_v3.png"); plt.close(fig)


def fig_fn_fp(M, trees):
    names, fn, fp, fne, fpe = [], [], [], [], []
    for name, runs in M.items():
        tk = {"XGBoost, host features (x)": "x", "HiGT-Bot v2 (x + grp)": "x+grp"}.get(name)
        if tk:
            a = [r["all"]["fn"] for r in trees["split"][tk]]; b = [r["all"]["fp"] for r in trees["split"][tk]]
        else:
            S = [scores(*r[:3]) for r in runs]; a = [s["fn"] for s in S]; b = [s["fp"] for s in S]
        names.append(name.replace(", ", "\n")); fn.append(np.mean(a)); fp.append(np.mean(b))
        fne.append(np.std(a, ddof=1) if len(a) > 1 else 0); fpe.append(np.std(b, ddof=1) if len(b) > 1 else 0)
    fig, ax = plt.subplots(figsize=(10, 3.6)); xs = np.arange(len(names)); w = 0.38
    ax.bar(xs - w / 2, fn, w, yerr=fne, capsize=3, label="missed bots (FN)", color="#c0504d")
    ax.bar(xs + w / 2, fp, w, yerr=fpe, capsize=3, label="false alarms (FP)", color="#f79646")
    ax.set_xticks(xs, names, fontsize=7); ax.set_ylabel("count on the test set")
    ax.set_title("Test set: 1,607 bots among 59,210 host-windows (mean ± sd over seeds where available)", fontsize=9)
    ax.legend(frameon=False); ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(OUT / "fig5_1_errors_v3.png"); plt.close(fig)


def fig_loso(trees):
    rows = ["x", "x+seq", "x+grp", "x+seq+grp", "x+nbr+seq+grp"]
    sc = sorted(trees["loso"]["x"])
    A = np.array([[trees["loso"][r][s]["pr_auc"] for s in sc] for r in rows])
    fig, ax = plt.subplots(figsize=(8, 2.8))
    im = ax.imshow(A, cmap="RdYlGn", vmin=0.4, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(sc)), sc, rotation=30, fontsize=8); ax.set_yticks(range(len(rows)), rows, fontsize=8)
    for i in range(len(rows)):
        for j in range(len(sc)):
            ax.text(j, i, f"{A[i, j]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, label="PR-AUC on left-out scenario")
    fig.savefig(OUT / "fig6_loso_heatmap.png"); plt.close(fig)


def fig_v2_diagram():
    fig, ax = plt.subplots(figsize=(9, 3.2)); ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 4)
    box = lambda x, y, w, h, t, c: (ax.add_patch(plt.Rectangle((x, y), w, h, fc=c, ec="k", lw=0.8)),
                                    ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=8))
    arrow = lambda x0, y0, x1, y1: ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="->", lw=1))
    box(0.1, 1.5, 1.8, 1.0, "Flows in a\n5-min window", "#eeeeee")
    box(2.5, 2.5, 2.4, 1.0, "Host features x (9)\nfrom ALL of the host's flows", "#dbe5f1")
    box(2.5, 0.5, 2.4, 1.0, "Host graph\n(who contacts whom)", "#dbe5f1")
    box(5.5, 1.5, 2.2, 1.0, "Group similarity grp (8)\nlook-alike hosts in window,\nshared-target overlap", "#ebf1de")
    box(8.2, 1.5, 1.7, 1.0, "Gradient-boosted\ntrees\n(threshold tuned\non validation)", "#fde9d9")
    arrow(1.9, 2.0, 2.5, 3.0); arrow(1.9, 2.0, 2.5, 1.0); arrow(4.9, 3.0, 5.5, 2.2); arrow(4.9, 1.0, 5.5, 1.8)
    ax.annotate("", (8.2, 2.3), (4.9, 3.2), arrowprops=dict(arrowstyle="->", lw=1, connectionstyle="arc3,rad=-0.15"))
    arrow(7.7, 2.0, 8.2, 2.0)
    ax.text(9.05, 1.2, "bot / benign\nper host", ha="center", va="top", fontsize=8)
    fig.savefig(OUT / "fig_v2_architecture.png"); plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    spec = SplitSpec.load()
    # v3_emb3 = same tree runs (identical non-emb rows) with the encoder retrained on graphs_v3
    trees = json.loads((LOGS / "v3_emb3/higt_v2_trees.json").read_text())
    md = ["# Tables for the rewrite (corrected labels; v3 features)", "",
          "Generated by `higt-bot-code/scripts/make_v3_tables_figures.py` from existing results.", ""]
    agg = t_label_counts(md)
    t_label_rankings(md)
    t_split(md, spec)
    M = model_rows(spec)
    t_main(md, M, spec, trees)
    t_significance(md)
    t_tree_ablation(md, trees)
    t_direction(md)
    t_loso(md)
    t_per_scenario(md, M, spec)
    (OUT / "tables_v3.md").write_text("\n".join(md), encoding="utf-8")
    fig_label_flaw(agg); fig_feature_fix(); fig_real_graph(); fig_fn_fp(M, trees); fig_loso(trees); fig_v2_diagram()
    print("wrote", OUT.resolve())


if __name__ == "__main__":
    main()
