"""Prevalence-adjusted LODO comparison table.

F1@0.5 is not comparable across LODO directions because bot prevalence swings
from ~0.36 (CTU-13) to ~0.99 (MedBIoT). This script reads the FP32-corrected
LODO result JSONs and reports, per (model, direction), metrics that are robust
to prior shift:

  - F1@0.5                (kept for continuity)
  - No-skill F1           = 2*prev/(1+prev)  -- the "predict all bot" baseline
  - F1 over no-skill      = F1 - no-skill F1  (negative => below trivial)
  - Balanced accuracy     = 0.5*(TPR + TNR)
  - MCC                   (robust to imbalance; 0 == no skill)
  - PR-AUC and PR-AUC lift = PR-AUC - prevalence (PR no-skill line)
  - ROC-AUC               (0.5 == no skill)

Sources: data/inspection_logs/*_lodo_*.json (Phase 5/6/7, FP32-corrected) and
experiments/lodo/<h>/phase4/*_lodo_*.json (flat baselines). AMP-backup files
are excluded. Seed replicates (GT-edge) are averaged over the identical test set.
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

LOGS = Path("data/inspection_logs")
HOLDOUTS = ("ctu13", "iot23", "medbiot")
MODEL_ORDER = (
    "baseline_rf", "baseline_xgb", "baseline_gat", "baseline_gin", "baseline_gine",
    "phase5_temporal_gine", "phase6_sparse", "phase7_higt_bot",
)
PRETTY = {
    "baseline_rf": "RandomForest", "baseline_xgb": "XGBoost", "baseline_gat": "GAT (flat)",
    "baseline_gin": "GIN (flat)", "baseline_gine": "GINE (flat)",
    "phase5_temporal_gine": "T-GINE + skip", "phase6_sparse": "SAGPool (sparse)",
    "phase7_higt_bot": "GT-edge (final)",
}


def collect() -> dict[str, dict[str, list[dict]]]:
    out: dict[str, dict[str, list[dict]]] = {h: defaultdict(list) for h in HOLDOUTS}
    cand = list(LOGS.glob("*_lodo_*.json"))
    for h in HOLDOUTS:
        cand += list((Path("experiments/lodo") / h / "phase4").glob("*_lodo_*.json"))
    cand = [c for c in cand if "_amp_backup" not in c.stem]
    for jf in cand:
        stem = jf.stem
        h = next((x for x in HOLDOUTS if f"_lodo_{x}" in stem), None)
        if h is None:
            continue
        head = stem.split(f"_lodo_{h}")[0]
        out[h][head].append(json.loads(jf.read_text()))
    return out


def avg(rows: list[dict], key: str) -> float:
    vals = [r["test"][key] for r in rows
            if key in r["test"] and isinstance(r["test"][key], (int, float))
            and not math.isnan(r["test"][key])]
    return mean(vals) if vals else float("nan")


def metrics(rows: list[dict]) -> dict:
    tn, fp = avg(rows, "tn"), avg(rows, "fp")
    fn, tp = avg(rows, "fn"), avg(rows, "tp")
    n = tn + fp + fn + tp
    pos, neg = tp + fn, tn + fp
    prev = pos / n if n else float("nan")
    tpr = tp / pos if pos else float("nan")
    tnr = tn / neg if neg else float("nan")
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = 2 * prec * tpr / (prec + tpr) if (prec + tpr) else 0.0
    bal_acc = 0.5 * (tpr + tnr)
    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / denom if denom else 0.0
    noskill_f1 = 2 * prev / (1 + prev) if not math.isnan(prev) else float("nan")
    pr = avg(rows, "pr_auc")
    roc = avg(rows, "roc_auc")
    f1s = [r["test"]["f1"] for r in rows if not math.isnan(r["test"].get("f1", float("nan")))]
    f1_std = stdev(f1s) if len(f1s) > 1 else None
    return dict(n=n, prev=prev, f1=f1, f1_std=f1_std, noskill_f1=noskill_f1,
                f1_over=f1 - noskill_f1, bal_acc=bal_acc, mcc=mcc,
                pr=pr, pr_lift=pr - prev if not math.isnan(pr) else float("nan"), roc=roc)


def fmt(v, nd=3):
    if v is None:
        return ""
    if isinstance(v, float) and math.isnan(v):
        return "--"
    return f"{v:+.{nd}f}" if isinstance(v, float) and v < 0 else f"{v:.{nd}f}"


def main() -> None:
    by = collect()
    L: list[str] = ["# LODO results — prevalence-adjusted (FP32-corrected)", "",
                    "F1@0.5 is confounded by prevalence shift across directions "
                    "(CTU-13 ~0.36 positive, IoT-23 ~0.74, MedBIoT ~0.99). Read "
                    "**MCC**, **balanced accuracy**, and **PR-AUC lift** (PR-AUC minus "
                    "prevalence) as the prior-robust metrics. *F1 vs no-skill* is F1 "
                    "minus the predict-all-bot baseline; negative means the model "
                    "scores below trivially predicting the majority class.", ""]

    for h in HOLDOUTS:
        # Establish the direction's prevalence / no-skill line from any model present.
        ref = next((metrics(by[h][m]) for m in MODEL_ORDER if by[h].get(m)), None)
        prev = ref["prev"] if ref else float("nan")
        L += [f"## LODO-{h.upper()}  (test prevalence = {fmt(prev)}, "
              f"no-skill F1 = {fmt(ref['noskill_f1']) if ref else '--'}, "
              f"no-skill MCC/BalAcc/ROC = 0 / 0.5 / 0.5)", "",
              "| Model | F1@0.5 | F1 vs no-skill | Bal.Acc | MCC | PR-AUC | PR-AUC lift | ROC-AUC |",
              "|---|---|---|---|---|---|---|---|"]
        for m in MODEL_ORDER:
            rows = by[h].get(m)
            if not rows:
                L.append(f"| {PRETTY[m]} | -- | -- | -- | -- | -- | -- | -- |")
                continue
            d = metrics(rows)
            f1cell = fmt(d["f1"]) + (f" ± {d['f1_std']:.3f}" if d["f1_std"] else "")
            L.append(f"| {PRETTY[m]} | {f1cell} | {fmt(d['f1_over'])} | {fmt(d['bal_acc'])} "
                     f"| {fmt(d['mcc'])} | {fmt(d['pr'])} | {fmt(d['pr_lift'])} | {fmt(d['roc'])} |")
        L.append("")

    # Headline GT-edge across directions.
    L += ["## GT-edge (final model) across directions", "",
          "| Metric | LODO-CTU13 | LODO-IoT23 | LODO-MedBIoT |", "|---|---|---|---|"]
    gt = {h: metrics(by[h]["phase7_higt_bot"]) for h in HOLDOUTS if by[h].get("phase7_higt_bot")}
    for label, key in [("Test prevalence", "prev"), ("F1@0.5", "f1"),
                       ("No-skill F1 (all-bot)", "noskill_f1"), ("F1 vs no-skill", "f1_over"),
                       ("Balanced accuracy", "bal_acc"), ("MCC", "mcc"),
                       ("PR-AUC", "pr"), ("PR-AUC lift", "pr_lift"), ("ROC-AUC", "roc")]:
        L.append(f"| {label} | " + " | ".join(fmt(gt[h][key]) for h in HOLDOUTS) + " |")
    L.append("")
    L += ["## Reading", "",
          "- **CTU-13 (enterprise, held out of an IoT-only training pool):** genuine "
          "failure — MCC ~0, ROC-AUC ~0.5, PR-AUC at the prevalence line. The model "
          "has no discriminative signal in this direction.",
          "- **IoT-23:** F1@0.5 sits on the no-skill line, so F1 is uninformative here; "
          "the real evidence of transfer is the positive **PR-AUC lift** and **MCC > 0**.",
          "- **MedBIoT:** high-prevalence regime. F1@0.5 looks high but is *below* the "
          "all-bot baseline (negative F1-vs-no-skill); PR-AUC lift is small. Use this "
          "direction only to show the FP32 result did not collapse, not as strong transfer.",
          ""]

    out = LOGS / "lodo_prevalence_adjusted.md"
    out.write_text("\n".join(L), encoding="utf-8")
    print(out.read_text())
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
