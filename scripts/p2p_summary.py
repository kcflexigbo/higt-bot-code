"""Summary of the P2P botnet experiments (run after queue_P_p2p.sh and queue_Q_isot_cv.sh).

Collects: bot-bot structure per dataset (graph_stats.json), the in-house P2P families
(family_eval.json), transfer of the thesis-split models to CTU-13 scenario 12 and ISOT 2010
(trees_results.json, gnn_ckpt_results.json, dirgine_transfer_seed42*.npz), and the within-ISOT
window-grouped 5-fold CV (isot_cv_*.json / isot_cv_oof_*.npz) with a paired window-level
bootstrap. Writes data/inspection_logs/p2p/summary_p2p_tables.md (the tables of summary_p2p.md), thesis_figures_v3/fig_p2p.png and
the "P2P" section of thesis_figures_v3/tables_v3.md (replaced if present).

    uv run python -m scripts.p2p_summary
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score

from scripts.higt_v2_trees import metrics, tune

L = Path("data/inspection_logs/p2p")
G3 = Path("data/inspection_logs/v3_gnn/predictions")
FIG = Path("../thesis_figures_v3/fig_p2p.png")
TAB = Path("../thesis_figures_v3/tables_v3.md")
SECTION = "## P2P botnets (corrected labels)"


def ms(v, d=3):
    v = np.asarray(v, float)
    return f"{v.mean():.{d}f} ± {v.std(ddof=1):.{d}f}" if len(v) > 1 else f"{v[0]:.{d}f}"


def jl(p):
    p = L / p
    return json.loads(p.read_text()) if p.exists() else None


def f1_at(y, p, t):
    return metrics(y, (p >= t).astype(int), p)


def boot_diff(y, pa, pb, fold, win, n=2000, seed=0):
    """Paired bootstrap over windows of PR-AUC(a) - PR-AUC(b) (out-of-fold scores)."""
    rng = np.random.default_rng(seed)
    wins = np.unique(win); idx = {w: np.flatnonzero(win == w) for w in wins}
    d0 = average_precision_score(y, pa) - average_precision_score(y, pb)
    ds = []
    for _ in range(n):
        s = np.concatenate([idx[w] for w in rng.choice(wins, len(wins))])
        if y[s].sum() == 0:
            continue
        ds.append(average_precision_score(y[s], pa[s]) - average_precision_score(y[s], pb[s]))
    ds = np.array(ds)
    return d0, np.percentile(ds, 2.5), np.percentile(ds, 97.5), float((ds <= 0).mean())


def main():
    md = ["# P2P botnet experiments (corrected labels, v3 features)", ""]
    tab = [SECTION, ""]

    # 1. structure
    gs = jl("graph_stats.json")
    rows = []
    for ds, pre in (("CTU-13", "ctu13-"), ("IoT-23", "iot23-"), ("MedBIoT", "medbiot-")):
        sc = [k for k in gs if k.startswith(pre) and gs[k]["bot_host_windows"] > 0]
        b = sum(gs[k]["bot_host_windows"] for k in sc)
        bb = sum(gs[k]["bot_bot_edges"] for k in sc)
        nb = sum(gs[k]["frac_bots_with_bot_neighbour"] * gs[k]["bot_host_windows"] for k in sc)
        rows.append((ds, b, bb, nb / max(b, 1)))
    for k, name in (("ctu13-12", "CTU-13 s12 (NSIS.ay, P2P)"), ("iot23-1-1", "IoT-23 1-1 (Hide and Seek, P2P)"),
                    ("iot23-9-1", "IoT-23 9-1 (Hajime, P2P)"), ("isot-2010", "ISOT 2010 (Storm/Waledac P2P, Zeus)")):
        g = gs[k]; rows.append((name, g["bot_host_windows"], g["bot_bot_edges"], g["frac_bots_with_bot_neighbour"]))
    t1 = ["| Data | Bot host-windows | Bot-bot edges | Bots with a bot neighbour |", "|---|---|---|---|"]
    t1 += [f"| {r[0]} | {r[1]:,} | {r[2]:,} | {100 * r[3]:.1f}% |" for r in rows]
    md += ["## 1. Do bots contact each other inside the monitored windows?", "",
           "Window graphs (data/graphs_v3), corrected labels (bot = originator of bot traffic). Dataset rows pool "
           "every scenario with bots.", ""] + t1 + [""]
    tab += ["### Table P.1. Bot-bot structure in the window graphs", ""] + t1 + [""]

    # 2. in-house P2P families
    fe = jl("family_eval.json")
    t2 = ["| Capture | Bots / host-windows (test) | Models | F1 | PR-AUC |", "|---|---|---|---|---|"]
    for k in ("iot23-1-1 Hide and Seek", "iot23-9-1 Hajime"):
        v = fe[k]; any_m = next(iter(v.values()))
        f1s = sorted({round(m["f1"], 3) for m in v.values()}); prs = sorted({round(m["pr_auc"], 3) for m in v.values()})
        t2.append(f"| {k} | {any_m['bots']} / {any_m['n']:,} | {', '.join(v)} | {'/'.join(map(str, f1s))} | {'/'.join(map(str, prs))} |")
    md += ["## 2. P2P families already in the thesis split (main-split test windows, saved predictions)", ""] + t2 + [
        "", "Each capture has a single infected device whose labelled traffic is a horizontal port scan "
        "(raw_flow_check.txt); every model separates it perfectly. These captures do not test P2P behaviour.", ""]

    # 3. transfer
    tr = jl("trees_results.json"); gk = jl("gnn_ckpt_results.json") or {}
    vd = np.load(L / "dirgine_transfer_seed42_val.npz"); td = np.load(L / "dirgine_transfer_seed42.npz", allow_pickle=True)
    t_dir = tune(vd["y"].astype(int), vd["proba"].astype(float))
    t3 = ["| Target | Bots / host-windows | Model | F1 (thesis-val threshold) | PR-AUC | FN | FP |", "|---|---|---|---|---|---|---|"]
    for sc in ("ctu13-12", "isot-2010"):
        n, b = tr[sc]["host_windows"], tr[sc]["bots"]
        for m, rs in tr[sc]["transfer"].items():
            t3.append(f"| {sc} | {b} / {n:,} | {m}{' (3 seeds)' if len(rs) > 1 else ''} | {ms([r['f1'] for r in rs])} | "
                      f"{ms([r['pr_auc'] for r in rs])} | {np.mean([r['fn'] for r in rs]):.0f} | {np.mean([r['fp'] for r in rs]):,.0f} |")
        m = td["scenario"].astype(str) == sc
        r = f1_at(td["y"][m].astype(int), td["proba"][m].astype(float), t_dir)
        t3.append(f"| {sc} | {b} / {n:,} | Directed GINE (retrained, seed 42) | {r['f1']:.3f} | {r['pr_auc']:.3f} | {r['fn']} | {r['fp']:,} |")
        for mname, r in gk.get(sc, {}).items():
            t3.append(f"| {sc} | {b} / {n:,} | {mname} (thesis checkpoint) | {r['f1']:.3f} | {r['pr_auc']:.3f} | {r['fn']} | {r['fp']:,} |")
    md += ["## 3. Transfer: models trained on the thesis split, tested on unseen P2P captures", "",
           "No retraining except the directed GINE, which was retrained on the unchanged thesis split with the P2P "
           "captures added as extra held-out test scenarios. Thresholds tuned on the thesis validation set.", ""] + t3 + [""]
    tab += ["### Table P.2. Transfer to unseen P2P captures", ""] + t3 + [""]

    # 4. within-ISOT CV
    trees = jl("isot_cv_trees.json"); gn = jl("isot_cv_gine.json") or {}; dg = jl("isot_cv_dirgine.json") or {}
    res = {**{f"XGB[{b}]" if b != "x+grp" else "HiGT-Bot v2 (XGB[x+grp])": v for b, v in trees.items()},
           **{"GINE (undirected)": gn.get("gine", [])}, **{"Directed GINE": dg.get("dirgine", [])}}
    t4 = ["| Model | F1 | PR-AUC | FN | FP |", "|---|---|---|---|---|"]
    for mname, rs in res.items():
        if rs:
            t4.append(f"| {mname} | {ms([r['f1'] for r in rs])} | {ms([r['pr_auc'] for r in rs])} | "
                      f"{np.mean([r['fn'] for r in rs]):.1f} | {np.mean([r['fp'] for r in rs]):.1f} |")
    # paired bootstrap on seed-42 out-of-fold scores
    oof = {k: np.load(L / f"isot_cv_oof_{k}.npz") for k in ("x", "x_grp", "x_nbr", "x_nbr_grp", "gine", "dirgine")
           if (L / f"isot_cv_oof_{k}.npz").exists()}
    import torch
    files = sorted(Path("data/graphs_v3/isot-2010").glob("window_*.pt"))
    win = np.concatenate([np.full(int(torch.load(f, weights_only=False).num_nodes), i) for i, f in enumerate(files)])
    t5 = ["| Comparison (seed 42, out-of-fold) | ΔPR-AUC | 95% CI | one-sided p |", "|---|---|---|---|"]
    for a, b, lab in (("x_nbr", "x", "XGB[x+nbr] − XGB[x]"), ("x_grp", "x", "HiGT-Bot v2 − XGB[x]"),
                      ("x_nbr", "x_grp", "XGB[x+nbr] − HiGT-Bot v2"), ("dirgine", "gine", "Directed GINE − GINE"),
                      ("dirgine", "x", "Directed GINE − XGB[x]"), ("x_nbr", "dirgine", "XGB[x+nbr] − Directed GINE")):
        if a in oof and b in oof:
            y = oof[a]["y"].astype(int)
            d, lo, hi, p = boot_diff(y, oof[a]["proba"], oof[b]["proba"], oof[a]["fold"], win)
            t5.append(f"| {lab} | {d:+.3f} | [{lo:+.3f}, {hi:+.3f}] | {p:.3f} |")
    n_b = int(sum(int(torch.load(f, weights_only=False).y.sum()) for f in files[-18:]))
    md += ["## 4. Within ISOT 2010: window-grouped 5-fold cross-validation", "",
           f"All {n_b} bot host-windows lie in the last 18 of 886 windows, so a chronological split has no training "
           "bots. Windows are dealt into 5 folds stratified by bot presence (seeds 42, 1, 2); test = fold k, "
           "validation (threshold, early stopping) = fold k+1, train = the rest. ISOT merges traces recorded at "
           "different times: the bot windows contain only hosts of the honeynet/Ericsson portion (172.16.0.0/16 "
           "and their external peers), never LBNL background hosts, and adjacent bot windows are correlated. "
           "These are within-capture estimates on 35 positives and have wide uncertainty.", ""] + t4 + ["", "Paired "
           "bootstrap over windows (2,000 resamples) of the seed-42 out-of-fold scores:", ""] + t5 + [""]
    tab += ["### Table P.3. Within-ISOT 2010, window-grouped 5-fold CV (mean ± sd over 3 seeds)", ""] + t4 + [""] + t5 + [""]

    md += ["## Files", "", "graph_stats.json, raw_flow_check.txt, family_eval.json, trees_results.json, "
           "gnn_ckpt_results.json, dirgine_transfer_seed42.{json,npz}, isot_cv_*.json, isot_cv_oof_*.npz, inventory.md "
           "(dataset provenance, sha256), queue_P_status.txt, queue_Q_status.txt.", ""]
    (L / "summary_p2p_tables.md").write_text("\n".join(md), encoding="utf-8")

    # tables_v3.md section (replace if present)
    s = TAB.read_text(encoding="utf-8")
    s = re.sub(rf"\n{re.escape(SECTION)}.*?(?=\n## (?!#)|\Z)", "", s, flags=re.S).rstrip() + "\n\n" + "\n".join(tab)
    TAB.write_text(s, encoding="utf-8")

    # figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "figure.dpi": 200, "savefig.bbox": "tight"})
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={"width_ratios": [1, 1.3]})
    names = [r[0] for r in rows]; vals = [100 * r[3] for r in rows]
    cols = ["#9aa5b1" if "P2P" not in n else "#2b6cb0" for n in names]
    ax[0].barh(range(len(names)), vals, color=cols); ax[0].set_yticks(range(len(names)), names, fontsize=8)
    ax[0].invert_yaxis(); ax[0].set_xlabel("bot host-windows with a bot neighbour (%)")
    for i, v in enumerate(vals):
        ax[0].text(v + 1, i, f"{v:.1f}%", va="center", fontsize=8)
    ax[0].set_xlim(0, max(vals) * 1.25 + 1); ax[0].set_title("(a) Bot-bot contact in the window graphs", fontsize=10)
    mm = [m for m, rs in res.items() if rs]
    x = np.arange(len(mm)); w = 0.38
    for j, (key, lab, c) in enumerate((("f1", "F1", "#2b6cb0"), ("pr_auc", "PR-AUC", "#dd6b20"))):
        mu = [np.mean([r[key] for r in res[m]]) for m in mm]; sd = [np.std([r[key] for r in res[m]], ddof=1) for m in mm]
        ax[1].bar(x + (j - 0.5) * w, mu, w, yerr=sd, capsize=3, label=lab, color=c)
    ax[1].set_xticks(x, [m.replace(" (XGB[x+grp])", "").replace("XGB", "XGB") for m in mm], rotation=25, ha="right", fontsize=8)
    ax[1].set_ylim(0, 1.05); ax[1].legend(loc="upper left", fontsize=8, frameon=False)
    ax[1].set_title("(b) Within ISOT 2010, 5-fold CV (3 seeds)", fontsize=10)
    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
    fig.savefig(FIG)
    print("\n".join(md))


if __name__ == "__main__":
    main()
