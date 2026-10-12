"""LODO decision-threshold recalibration (post-hoc, no retraining).

Reads val (source) + test (held-out target) probabilities written by
  scripts/diagnose_lodo_predictions.py --save-preds data/inspection_logs/predictions/lodo/<run>_seed<s>.npz
and compares ways of choosing the bot/benign decision rule on the target:

  fixed_0.5     argmax, as trained
  src_val       threshold maximising MCC on the source validation split
  em_prior      Saerens et al. (2002) EM re-estimation of the target bot prior from
                unlabelled target scores, posteriors re-weighted, threshold 0.5
  bbse          black-box shift estimation of the target prior (Lipton et al., 2018)
                from the source-val confusion matrix, then the same re-weighting
  quantile      threshold set so the predicted-bot rate on the target equals the
                EM prior estimate (unsupervised)
  kshot_k       threshold maximising MCC on k labelled target windows, evaluated on
                the remaining windows (mean over random draws)
  oracle        threshold maximising MCC on the whole target set (upper bound)
  chrono_k      per target capture, threshold tuned on its chronologically first k
                windows and applied to its later windows (deployment-like); compared
                with fixed_0.5 and an oracle threshold on exactly the same later windows

Temperature / Platt scaling followed by a threshold is a monotone transform of the
score, so for binary decisions it is equivalent to threshold selection and is not
listed separately.

Output: data/inspection_logs/lodo_recalibration.{md,json}
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

INSPECTION = Path("data/inspection_logs")
RUNS = {"ctu13": "LODO-CTU13", "iot23": "LODO-IoT23 (AMP-trained)",
        "iot23_fp32": "LODO-IoT23 (FP32-trained)", "medbiot": "LODO-MedBIoT"}
GRID = np.unique(np.concatenate([np.linspace(0.001, 0.999, 400), [0.5]]))


def confusion(y, pred):
    tp = int(((pred == 1) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    return tp, tn, fp, fn


def metrics(y, pred):
    tp, tn, fp, fn = confusion(y, pred)
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    tpr = tp / (tp + fn) if tp + fn else 0.0
    tnr = tn / (tn + fp) if tn + fp else 0.0
    den = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / den if den else 0.0
    return {"f1": f1, "mcc": mcc, "bal_acc": (tpr + tnr) / 2, "recall": tpr, "fpr": 1 - tnr}


def best_threshold(y, p):
    if len(y) == 0 or y.min() == y.max():
        return None
    scores = [metrics(y, (p >= t).astype(int))["mcc"] for t in GRID]
    return float(GRID[int(np.argmax(scores))])


def reweight(p, pi_s, pi_t):
    a = (pi_t / pi_s) * p
    b = ((1 - pi_t) / (1 - pi_s)) * (1 - p)
    return a / np.maximum(a + b, 1e-12)


def em_prior(p, pi_s, iters=200, tol=1e-7):
    pi_t = pi_s
    for _ in range(iters):
        new = float(np.clip(reweight(p, pi_s, pi_t).mean(), 1e-4, 1 - 1e-4))
        if abs(new - pi_t) < tol:
            break
        pi_t = new
    return pi_t


def bbse_prior(yv, pv, pt):
    # C[i, j] = P(pred = i | y = j) on source val; q = predicted-class rates on target
    pred_v = (pv >= 0.5).astype(int)
    C = np.array([[np.mean(pred_v[yv == j] == i) for j in (0, 1)] for i in (0, 1)])
    q = np.array([np.mean(pt < 0.5), np.mean(pt >= 0.5)])
    try:
        w = np.linalg.solve(C, q)  # target class proportions
    except np.linalg.LinAlgError:
        return None
    w = np.clip(w, 0, None)
    return float(np.clip(w[1] / w.sum(), 1e-4, 1 - 1e-4)) if w.sum() > 0 else None


def run_one(d, rng, ks, n_draws):
    yv, pv = d["val_y"].astype(int), np.nan_to_num(d["val_proba"], nan=0.5)
    yt, pt = d["test_y"].astype(int), np.nan_to_num(d["test_proba"], nan=0.5)
    gt = d["test_graph"]
    pi_s = float(yv.mean())
    res = {"prevalence": float(yt.mean()), "roc_auc": float(roc_auc_score(yt, pt)),
           "pr_auc": float(average_precision_score(yt, pt))}
    res["no_skill_f1"] = 2 * res["prevalence"] / (1 + res["prevalence"])
    res["fixed_0.5"] = metrics(yt, (pt >= 0.5).astype(int))
    t = best_threshold(yv, pv)
    res["src_val"] = {**metrics(yt, (pt >= t).astype(int)), "threshold": t}
    pi_em = em_prior(pt, pi_s)
    res["em_prior"] = {**metrics(yt, (reweight(pt, pi_s, pi_em) >= 0.5).astype(int)), "pi_hat": pi_em}
    pi_bb = bbse_prior(yv, pv, pt)
    if pi_bb is not None:
        res["bbse"] = {**metrics(yt, (reweight(pt, pi_s, pi_bb) >= 0.5).astype(int)), "pi_hat": pi_bb}
    tq = float(np.quantile(pt, 1 - pi_em))
    res["quantile"] = {**metrics(yt, (pt >= tq).astype(int)), "threshold": tq}
    t_or = best_threshold(yt, pt)
    res["oracle"] = {**metrics(yt, (pt >= t_or).astype(int)), "threshold": t_or}
    graphs = np.unique(gt)
    for k in ks:
        acc = defaultdict(list)
        for _ in range(n_draws):
            pick = rng.choice(graphs, size=min(k, len(graphs) - 1), replace=False)
            cal = np.isin(gt, pick)
            tk = best_threshold(yt[cal], pt[cal])
            if tk is None:  # calibration windows hold only one class: fall back to source threshold
                tk = t
            for m, v in metrics(yt[~cal], (pt[~cal] >= tk).astype(int)).items():
                acc[m].append(v)
        res[f"kshot_{k}"] = {m: float(np.mean(v)) for m, v in acc.items()} | {"mcc_std": float(np.std(acc["mcc"]))}
    if "test_scenario" in d.files:
        st = d["test_scenario"]
        for k in ks:
            cal = np.zeros(len(yt), dtype=bool)
            pred = np.zeros(len(yt), dtype=int)
            pred_or = np.zeros(len(yt), dtype=int)
            macro = defaultdict(list)
            for sc in np.unique(st):
                m = st == sc
                first = np.unique(gt[m])[:k]
                c = m & np.isin(gt, first)
                tk = best_threshold(yt[c], pt[c])
                tk = t if tk is None else tk
                r = m & ~c
                t_or_sc = best_threshold(yt[r], pt[r])
                t_or_sc = 0.5 if t_or_sc is None else t_or_sc
                pred[r] = (pt[r] >= tk).astype(int)
                pred_or[r] = (pt[r] >= t_or_sc).astype(int)
                cal |= c
                if r.any() and 0 < yt[r].sum() < r.sum():  # capture has both classes left
                    macro["chrono"].append(metrics(yt[r], pred[r])["mcc"])
                    macro["fixed"].append(metrics(yt[r], (pt[r] >= 0.5).astype(int))["mcc"])
                    macro["oracle"].append(metrics(yt[r], pred_or[r])["mcc"])
            rest = ~cal
            res[f"chrono_{k}"] = {**metrics(yt[rest], pred[rest]),
                                  "fixed_0.5_same_rest": metrics(yt[rest], (pt[rest] >= 0.5).astype(int)),
                                  "oracle_same_rest": metrics(yt[rest], pred_or[rest]),
                                  "macro_mcc": {m: float(np.mean(v)) for m, v in macro.items()},
                                  "macro_n_captures": len(macro["chrono"]),
                                  "labelled_windows": int(len(np.unique(gt[cal])))}
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preds-dir", type=Path, default=INSPECTION / "predictions" / "lodo")
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 2, 5, 10, 20])
    ap.add_argument("--draws", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    out = {}
    for run in RUNS:
        for s in (42, 1, 2):
            f = args.preds_dir / f"{run}_seed{s}.npz"
            if f.exists():
                out.setdefault(run, {})[str(s)] = run_one(np.load(f), rng, args.ks, args.draws)
                print(f"done {run} seed {s}", flush=True)
    (INSPECTION / "lodo_recalibration.json").write_text(json.dumps(out, indent=2))

    methods = ["fixed_0.5", "src_val", "em_prior", "bbse", "quantile"] + [f"kshot_{k}" for k in args.ks] + ["oracle"]
    L = ["# LODO decision-threshold recalibration (post-hoc, no retraining)", "",
         "Final GT-edge model; mean ± std over seeds. Thresholds tuned for MCC. "
         "k-shot: k labelled target windows used for tuning, metrics on the remaining windows "
         f"({args.draws} random draws per seed).", ""]
    for run, seeds in out.items():
        r0 = list(seeds.values())
        ms = lambda key, sub=None: np.array([(r[key][sub] if sub else r[key]) for r in r0 if key in r])
        L += [f"## {RUNS[run]} ({len(r0)} seeds)", "",
              f"Prevalence {ms('prevalence').mean():.3f} · no-skill F1 {ms('no_skill_f1').mean():.3f} · "
              f"ROC-AUC per seed {', '.join(f'{v:.3f}' for v in ms('roc_auc'))} · "
              f"PR-AUC lift {np.mean(ms('pr_auc') - ms('prevalence')):+.3f}", "",
              "| Method | F1 | MCC | Balanced acc. | Recall | FPR |", "|---|---|---|---|---|---|"]
        for m in methods:
            if not all(m in r for r in r0):
                continue
            cells = []
            for k in ("f1", "mcc", "bal_acc", "recall", "fpr"):
                v = ms(m, k)
                cells.append(f"{v.mean():.3f} ± {v.std():.3f}")
            L.append(f"| {m} | " + " | ".join(cells) + " |")
        L.append("")
        ck = [k for k in args.ks if all(f"chrono_{k}" in r for r in r0)]
        if ck:
            L += ["Chronological k-shot (per target capture: first k windows labelled, later windows "
                  "evaluated). Oracle = best threshold per capture on the same later windows. "
                  "Pooled MCC over all later windows, and macro MCC (mean over captures with both classes); mean ± std over seeds:", "",
                  "| k | labelled windows | chrono_k pooled | fixed 0.5 pooled | oracle pooled | chrono_k macro | fixed 0.5 macro | oracle macro | captures |",
                  "|---|---|---|---|---|---|---|---|---|"]
            for k in ck:
                c = [r[f"chrono_{k}"] for r in r0]
                f = lambda xs: f"{np.mean(xs):.3f} ± {np.std(xs):.3f}"
                L.append(f"| {k} | {int(np.mean([x['labelled_windows'] for x in c]))} | {f([x['mcc'] for x in c])} | "
                         f"{f([x['fixed_0.5_same_rest']['mcc'] for x in c])} | {f([x['oracle_same_rest']['mcc'] for x in c])} | "
                         + " | ".join(f([x['macro_mcc'].get(mm, np.nan) for x in c]) for mm in ("chrono", "fixed", "oracle"))
                         + f" | {c[0]['macro_n_captures']} |")
            L.append("")
    pooled = [r for run in ("iot23", "iot23_fp32") for r in out.get(run, {}).values()]
    if pooled:
        L += [f"## LODO-IoT23, all {len(pooled)} runs pooled (AMP- and FP32-trained)", "",
              f"ROC-AUC {np.mean([r['roc_auc'] for r in pooled]):.3f} ± {np.std([r['roc_auc'] for r in pooled]):.3f} "
              f"(range {min(r['roc_auc'] for r in pooled):.3f}–{max(r['roc_auc'] for r in pooled):.3f}); "
              f"PR-AUC lift {np.mean([r['pr_auc'] - r['prevalence'] for r in pooled]):+.3f}", "",
              "| Method | MCC | F1 | FPR |", "|---|---|---|---|"]
        for m in methods + [f"chrono_{k}" for k in args.ks]:
            if all(m in r for r in pooled):
                L.append(f"| {m} | " + " | ".join(f"{np.mean([r[m][x] for r in pooled]):.3f} ± {np.std([r[m][x] for r in pooled]):.3f}"
                                               for x in ("mcc", "f1", "fpr")) + " |")
        L.append("")
    (INSPECTION / "lodo_recalibration.md").write_text("\n".join(L), encoding="utf8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
