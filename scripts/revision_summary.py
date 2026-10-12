"""Collects the review-revision experiments into data/inspection_logs/revision/summary_tables.md.

Inputs: revision/{tables.npz, seed_preds/, botminer.npz, importance.json, loso_refit.json},
v3_gnn/predictions, blind_spot_counts.json, v3/bootstrap_seeds_holm.json (if present).

    uv run python -m scripts.revision_summary
"""
from __future__ import annotations

import os

os.environ.setdefault("HIGT_GRAPHS_DIR", "data/graphs_v3")
import torch  # noqa: E402,F401

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score

from scripts.bootstrap_seeds_holm import gnn_runs
from src.data.dataset import SplitSpec

R = Path("data/inspection_logs/revision")
TREES = ["XGB[x]", "v2 XGB[x+grp]", "RF tuned", "XGB[x+wnorm]", "XGB[x+grp_prof]", "XGB[x+grp_share]", "XGB[x+wnorm+grp]"]
SELECT = ["XGB[x]", "XGB[x+seq]", "v2 XGB[x+grp]", "XGB[x+seq+grp]", "XGB[x+emb]", "XGB[x+grp+emb]",
          "XGB[x+seq+grp+emb]", "XGB[x+nbr+seq+grp]"]
HOST_WINDOWS_PER_DAY = 288  # 300-s windows


def stem(m):
    return m.replace("[", "_").replace("]", "").replace("+", "_").replace(" ", "_")


def conf(y, p):
    tp = int(((p == 1) & (y == 1)).sum()); fp = int(((p == 1) & (y == 0)).sum()); fn = int(((p == 0) & (y == 1)).sum())
    return {"f1": 2 * tp / max(2 * tp + fp + fn, 1), "tp": tp, "fp": fp, "fn": fn}


def runs(model, seeds=(42, 1, 2, 3, 4)):
    out = []
    for s in seeds:
        f = R / "seed_preds" / f"{stem(model)}_s{s}.npz"
        if f.exists():
            out.append(dict(np.load(f)))
    return out


def ms(v, d=3):
    v = [x for x in v if x == x]
    return f"{np.mean(v):.{d}f} ± {np.std(v, ddof=1):.{d}f}" if len(v) > 1 else (f"{v[0]:.{d}f}" if v else "n/a")


def main():
    spec = SplitSpec.load()
    T = np.load(R / "tables.npz", allow_pickle=True)
    yte = T["test_y"].astype(int); ste = T["test_scenario"].astype(str)
    yva = T["val_y"].astype(int)
    ypr = T["pristine_y"].astype(int); spr = T["pristine_scenario"].astype(str)
    mal = np.char.startswith(spr, "iot23"); ben = ~mal
    md = ["# Review-revision experiments: tables", ""]

    # ---------------------------------------------------------------- C1: pristine
    md += ["## C1. Confirmatory evaluation on captures never used before (frozen models, validation thresholds)", "",
           f"Malicious IoT-23 captures never evaluated before: {', '.join(sorted(set(spr[mal])))} "
           f"({int(mal.sum()):,} host-windows, {int(ypr[mal].sum())} bot host-windows). Benign-only MedBIoT captures: "
           f"{', '.join(sorted(set(spr[ben])))} ({int(ben.sum()):,} host-windows, no bots). Models trained once on the "
           "main training split (5 seeds), threshold from the main validation set, no retuning.", "",
           "| Model | F1 (malicious captures) | PR-AUC | FN | FP | false alarms per 1,000 benign host-windows (benign captures) | alarms per host-day |",
           "|---|---|---|---|---|---|---|"]
    pr_rows = {}
    for m in ["XGB[x]", "v2 XGB[x+grp]", "RF tuned", "XGB[x+wnorm]"]:
        rr = runs(m)
        if not rr:
            continue
        c = [conf(ypr[mal], (r["p_pristine"][mal] >= r["threshold"]).astype(int)) for r in rr]
        ap = [average_precision_score(ypr[mal], r["p_pristine"][mal]) for r in rr]
        fa = [1000 * int((r["p_pristine"][ben] >= r["threshold"]).sum()) / int(ben.sum()) for r in rr]
        pr_rows[m] = {"f1": [x["f1"] for x in c], "pr_auc": ap, "fn": [x["fn"] for x in c], "fp": [x["fp"] for x in c], "fa_per_1000": fa}
        md.append(f"| {m} | {ms([x['f1'] for x in c])} | {ms(ap)} | {ms([x['fn'] for x in c], 0)} | {ms([x['fp'] for x in c], 0)} "
                  f"| {ms(fa, 2)} | {ms([v * HOST_WINDOWS_PER_DAY / 1000 for v in fa], 3)} |")
    bm = np.load(R / "botminer.npz") if (R / "botminer.npz").exists() else None
    if bm is not None:
        c = conf(ypr[mal], (bm["p_pristine"][mal] >= bm["threshold"]).astype(int))
        fa = 1000 * int((bm["p_pristine"][ben] >= bm["threshold"]).sum()) / int(ben.sum())
        md.append(f"| BotMiner-style cohort (unsupervised) | {c['f1']:.3f} | {average_precision_score(ypr[mal], bm['p_pristine'][mal]):.3f} "
                  f"| {c['fn']} | {c['fp']} | {fa:.2f} | {fa * HOST_WINDOWS_PER_DAY / 1000:.3f} |")
    GP = R / "gnn_pristine"
    for name, files in (("Directed T-GINE", ["tgine_temporal_gine_skip.npz"]),
                        ("HiGT-Bot directed", ["higt_bot_edge.npz", "higt_bot_edge_seed1.npz", "higt_bot_edge_seed2.npz"])):
        rr = [dict(np.load(GP / f, allow_pickle=True)) for f in files if (GP / f).exists()]
        if not rr:
            continue
        for r in rr:  # graph-model rows follow the same window order; check alignment with the tree tables
            assert len(r["y"]) == len(ypr) and (r["y"].astype(int) == ypr).all(), name
        c = [conf(ypr[mal], (r["proba"][mal] >= r["threshold"]).astype(int)) for r in rr]
        ap = [average_precision_score(ypr[mal], r["proba"][mal]) for r in rr]
        fa = [1000 * int((r["proba"][ben] >= r["threshold"]).sum()) / int(ben.sum()) for r in rr]
        md.append(f"| {name} ({len(rr)} seed{'s' if len(rr) > 1 else ''}) | {ms([x['f1'] for x in c])} | {ms(ap)} | {ms([x['fn'] for x in c], 0)} "
                  f"| {ms([x['fp'] for x in c], 0)} | {ms(fa, 2)} | {ms([v * HOST_WINDOWS_PER_DAY / 1000 for v in fa], 3)} |")
    md += ["", "Per-scenario recall on the malicious captures (mean over seeds):", "",
           "| Scenario | bot host-windows | " + " | ".join(pr_rows) + " |", "|---|---|" + "---|" * len(pr_rows)]
    for s in sorted(set(spr[mal])):
        k = spr == s
        cells = []
        for m in pr_rows:
            rec = [((r["p_pristine"][k] >= r["threshold"]) & (ypr[k] == 1)).sum() / max(ypr[k].sum(), 1) for r in runs(m)]
            cells.append(f"{np.mean(rec):.3f}")
        md.append(f"| {s} | {int(ypr[k].sum())} | " + " | ".join(cells) + " |")
    # secondary
    ysec = T["secondary_y"].astype(int)
    md += ["", f"Secondary set (captures listed as excluded for size; their bot counts were inspected earlier): "
           f"{len(ysec):,} host-windows, {int(ysec.sum())} bots.", "", "| Model | F1 | FN | FP |", "|---|---|---|---|"]
    for m in ["XGB[x]", "v2 XGB[x+grp]", "RF tuned"]:
        c = [conf(ysec, (r["p_secondary"] >= r["threshold"]).astype(int)) for r in runs(m)]
        md.append(f"| {m} | {ms([x['f1'] for x in c])} | {ms([x['fn'] for x in c], 0)} | {ms([x['fp'] for x in c], 0)} |")

    # ---------------------------------------------------------------- C1c: validation-only selection
    md += ["", "## C1c. Block selection on the validation set alone (seeds 42, 1, 2)", "",
           "| Blocks | val PR-AUC | val F1 (val-tuned threshold) | test F1 | test PR-AUC |", "|---|---|---|---|---|"]
    sel = {}
    for m in SELECT:
        rr = runs(m, (42, 1, 2))
        if not rr:
            continue
        vap = [average_precision_score(yva, r["p_val"]) for r in rr]
        vf = [conf(yva, (r["p_val"] >= r["threshold"]).astype(int))["f1"] for r in rr]
        tf = [conf(yte, (r["p_test"] >= r["threshold"]).astype(int))["f1"] for r in rr]
        tap = [average_precision_score(yte, r["p_test"]) for r in rr]
        sel[m] = (np.mean(vap), np.mean(vf))
        md.append(f"| {m} | {ms(vap, 4)} | {ms(vf, 4)} | {ms(tf)} | {ms(tap)} |")
    if sel:
        md += ["", f"Best by validation PR-AUC: **{max(sel, key=lambda k: sel[k][0])}**; best by validation F1: "
               f"**{max(sel, key=lambda k: sel[k][1])}**. Note: the validation set contains no held-out scenario."]

    # ---------------------------------------------------------------- M2
    md += ["", "## M2. Cohort baselines and decomposition of the group block (main test split, 5 seeds)", "",
           "| Model | F1 | PR-AUC | FN | FP | held-out F1 | ctu13-10 F1 |", "|---|---|---|---|---|---|---|"]
    ho = np.isin(ste, spec.holdout_test_scenarios); c10 = ste == "ctu13-10"
    for m in TREES:
        rr = runs(m)
        if not rr:
            continue
        P = [(r["p_test"] >= r["threshold"]).astype(int) for r in rr]
        c = [conf(yte, p) for p in P]
        md.append(f"| {m} | {ms([x['f1'] for x in c])} | {ms([average_precision_score(yte, r['p_test']) for r in rr])} "
                  f"| {ms([x['fn'] for x in c], 0)} | {ms([x['fp'] for x in c], 0)} | {ms([conf(yte[ho], p[ho])['f1'] for p in P])} "
                  f"| {ms([conf(yte[c10], p[c10])['f1'] for p in P])} |")
    if bm is not None:
        p = (bm["p_test"] >= bm["threshold"]).astype(int); c = conf(yte, p)
        md.append(f"| BotMiner-style cohort (unsupervised) | {c['f1']:.3f} | {average_precision_score(yte, bm['p_test']):.3f} | {c['fn']} "
                  f"| {c['fp']} | {conf(yte[ho], p[ho])['f1']:.3f} | {conf(yte[c10], p[c10])['f1']:.3f} |")
    if (R / "importance.json").exists():
        imp = json.loads((R / "importance.json").read_text())
        md += ["", "Feature importance of the x+grp model (seed 42): gain share, and drop in validation PR-AUC when the "
               f"feature is permuted (validation PR-AUC {imp['val_pr_auc']:.4f}).", "",
               "| Feature | gain share | permutation drop |", "|---|---|---|"]
        for k, v in imp["gain_share"].items():
            md.append(f"| {k} | {v:.3f} | {imp['permutation_val_prauc_drop'].get(k, float('nan')):.4f} |")
        md += ["", "Block permutation (validation PR-AUC drop): " +
               "; ".join(f"{k} {v:.4f}" for k, v in imp["block_permutation_val_prauc_drop"].items())]

    # ---------------------------------------------------------------- M6: co-active bots
    win = T["test_win"]
    nb = np.zeros(len(yte), int)
    for w in np.unique(win):
        k = win == w; nb[k] = yte[k].sum()
    bins = [(1, 1, "1 (single infected host)"), (2, 3, "2-3"), (4, 9, "4-9"), (10, 10**6, ">= 10")]
    md += ["", "## M6. Recall by number of infected hosts in the window (main test split, means over seeds)", "",
           "| Bots in window | bot host-windows | " + " | ".join(["XGB[x]", "v2 XGB[x+grp]", "XGB[x+wnorm]", "BotMiner-style"]) + " |",
           "|---|---|---|---|---|---|"]
    for lo, hi, lab in bins:
        k = (yte == 1) & (nb >= lo) & (nb <= hi)
        if not k.any():
            continue
        cells = []
        for m in ["XGB[x]", "v2 XGB[x+grp]", "XGB[x+wnorm]"]:
            cells.append(f"{np.mean([(r['p_test'][k] >= r['threshold']).mean() for r in runs(m)]):.3f}")
        cells.append(f"{(bm['p_test'][k] >= bm['threshold']).mean():.3f}" if bm is not None else "n/a")
        md.append(f"| {lab} | {int(k.sum())} | " + " | ".join(cells) + " |")

    # ---------------------------------------------------------------- M8
    blind = json.loads(Path("data/inspection_logs/blind_spot_counts.json").read_text())
    keep = {k: v["src_kept"] / v["src_pairs"] for k, v in blind.items() if isinstance(v, dict) and v.get("src_pairs")}
    models = {m: [((r["p_test"] >= r["threshold"]).astype(int), r["p_test"]) for r in runs(m)] for m in ["v2 XGB[x+grp]", "XGB[x]", "RF tuned"]}
    G = Path("data/inspection_logs/v3_gnn/predictions")
    for name, stems in (("Directed GINE", ["dirgine_seed42", "dirgine_seed1", "dirgine_seed2"]),
                        ("HiGT-Bot directed", ["higt_bot_edge", "higt_bot_edge_seed1", "higt_bot_edge_seed2"]),
                        ("Directed T-GINE", ["t_gine_skip"]), ("SAGPool", ["sagpool"])):
        if all((G / f"{s}.npz").exists() for s in stems):
            models[name] = gnn_runs(stems)
    scen = [s for s in sorted(set(ste)) if yte[ste == s].sum() > 0]
    md += ["", "## M8. Macro-averaged metrics and end-to-end recall (main test split; means over seeds)", "",
           f"Macro = unweighted mean over the {len(scen)} test scenarios with at least one bot host-window. End-to-end "
           "recall counts infected host-windows dropped by graph construction as misses; for split scenarios the dropped "
           "count in the test period is estimated from the scenario's kept fraction (blind_spot_counts), for held-out "
           "scenarios (all windows in test) it is exact.", "",
           "| Model | pooled F1 | macro F1 | macro PR-AUC | CTU-13 F1 | IoT-23 F1 | MedBIoT F1 | recall (in graph) | end-to-end recall | held-out end-to-end recall | FP per 1,000 benign host-windows |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, rr in models.items():
        vals = {k: [] for k in ("pool", "mf1", "map", "ctu", "iot", "med", "rec", "e2e", "e2e_ho", "fa")}
        for pred, prob in rr:
            vals["pool"].append(conf(yte, pred)["f1"])
            vals["mf1"].append(np.mean([conf(yte[ste == s], pred[ste == s])["f1"] for s in scen]))
            vals["map"].append(np.mean([average_precision_score(yte[ste == s], prob[ste == s]) if (yte[ste == s] == 0).any() else 1.0 for s in scen]))
            for key, pre in (("ctu", "ctu13"), ("iot", "iot23"), ("med", "medbiot")):
                k = np.char.startswith(ste, pre); vals[key].append(conf(yte[k], pred[k])["f1"])
            tp = ((pred == 1) & (yte == 1))
            vals["rec"].append(tp.sum() / yte.sum())
            dropped = sum(yte[ste == s].sum() * (1 - keep.get(s, 1.0)) / max(keep.get(s, 1.0), 1e-9) for s in scen)
            vals["e2e"].append(tp.sum() / (yte.sum() + dropped))
            hos = [s for s in scen if s in spec.holdout_test_scenarios]
            dh = sum(yte[ste == s].sum() * (1 - keep.get(s, 1.0)) / max(keep.get(s, 1.0), 1e-9) for s in hos)
            kh = np.isin(ste, hos)
            vals["e2e_ho"].append(tp[kh].sum() / (yte[kh].sum() + dh))
            vals["fa"].append(1000 * ((pred == 1) & (yte == 0)).sum() / (yte == 0).sum())
        md.append(f"| {name} ({len(rr)} seed{'s' if len(rr) > 1 else ''}) | " +
                  " | ".join(ms(vals[k], 2 if k == "fa" else 3) for k in vals) + " |")
    md += ["", f"Benign host-windows in the test split: {(yte == 0).sum():,}. One host-window = 300 s, so alarms per "
           f"host-day = (alarms per 1,000 host-windows) x {HOST_WINDOWS_PER_DAY}/1000."]

    # ---------------------------------------------------------------- M9
    if (R / "loso_refit.json").exists():
        L = json.loads((R / "loso_refit.json").read_text())
        md += ["", "## M9. CTU-13 leave-one-scenario-out, quantile scaler refitted per fold (3 seeds)", "",
               "Note: the ctu13-10 fold trains on the same scenarios as the main split (ctu13-10 is held out there too), "
               "so it largely repeats the main held-out result.", "",
               "| Left-out | model | PR-AUC | F1 at validation threshold | F1 at best threshold (oracle) | recall at 1% FPR |",
               "|---|---|---|---|---|---|"]
        agg = {}
        for s, bys in L.items():
            for m in ("XGB[x]", "v2 XGB[x+grp]"):
                v = {k: [bys[sd][m][k] for sd in bys] for k in ("pr_auc", "f1_val_thr", "f1_oracle", "recall_at_1pct_fpr")}
                for k, x in v.items():
                    agg.setdefault(m, {}).setdefault(k, []).append(np.mean(x))
                md.append(f"| {s} | {m} | {ms(v['pr_auc'])} | {ms(v['f1_val_thr'])} | {ms(v['f1_oracle'])} | {ms(v['recall_at_1pct_fpr'])} |")
        md += ["", "| Model | mean PR-AUC | mean F1 (val thr) | mean F1 (oracle thr) | mean recall at 1% FPR |", "|---|---|---|---|---|"]
        for m, v in agg.items():
            md.append(f"| {m} | {np.mean(v['pr_auc']):.3f} | {np.mean(v['f1_val_thr']):.3f} | {np.mean(v['f1_oracle']):.3f} | {np.mean(v['recall_at_1pct_fpr']):.3f} |")

    (R / "summary_tables.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
