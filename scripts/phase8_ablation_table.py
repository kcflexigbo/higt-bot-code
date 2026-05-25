"""Phase 8.2 — compile the final ablation table from all logged JSON results.

Reads every result file under data/inspection_logs/ that matches a known
phase pattern and assembles a single ablation table covering:
  - Phase 4 baselines (GAT, GIN, GINE, RF, XGB)
  - Phase 5 variants (T-GINE, T-GINE+skip, T-GINE+SSL-init)
  - Phase 6 variants (DiffPool, +TAM, +GraphSHA, +SSL-FT)
  - Phase 6.4 SAGPool (and SAGPool × SSL-FT)
  - Phase 7 GT variants (edge, global, hybrid-2L, hybrid-4L)

Output:
  - data/inspection_logs/phase8_ablation_table.md (the table)
  - data/inspection_logs/phase8_ablation_table.json (machine-readable)
"""
from __future__ import annotations

import json
from pathlib import Path

LOG_DIR = Path("data/inspection_logs")

# (label, json filename, phase, ablation-table category)
RUNS: list[tuple[str, str, str, str]] = [
    # Phase 4 baselines
    ("GAT (flat)",                "baseline_gat.json",                       "4",  "Phase 4 baseline"),
    ("GIN (flat)",                "baseline_gin.json",                       "4",  "Phase 4 baseline"),
    ("GINE (flat)",               "baseline_gine.json",                      "4",  "Phase 4 baseline"),
    ("GINE matched-alpha",        "baseline_gine_matched.json",              "4",  "Phase 4 baseline"),
    ("RandomForest",              "baseline_rf.json",                        "4",  "Phase 4 baseline (tabular)"),
    ("XGBoost",                   "baseline_xgb.json",                       "4",  "Phase 4 baseline (tabular)"),
    # Phase 5
    ("T-GINE",                    "phase5_temporal_gine.json",               "5",  "Phase 5 (temporal encoder)"),
    ("T-GINE + raw-skip",         "phase5_temporal_gine_skip.json",          "5",  "Phase 5 (temporal encoder)"),
    ("T-GINE + SSL-init",         "phase5_temporal_gine_ssl_init.json",      "5",  "Phase 5 (temporal encoder)"),
    # Phase 6
    ("HiGT-Bot DiffPool",         "phase6_diffpool.json",                    "6",  "Phase 6 (hierarchical, dense pool)"),
    ("DiffPool + TAM v1",         "phase6_diffpool_tam.json",                "6.1","Phase 6 ablations"),
    ("DiffPool + TAM v2",         "phase6_diffpool_tam_bal.json",            "6.1","Phase 6 ablations"),
    ("DiffPool + GraphSHA",       "phase6_diffpool_sha.json",                "6.2","Phase 6 ablations"),
    ("DiffPool + SSL alone",      "phase6_diffpool_ssl.json",                "6.3","Phase 6 ablations"),
    ("DiffPool + SSL→FT",         "phase6_diffpool_ssl_ft.json",             "6.3","Phase 6 ablations"),
    # Phase 6.4 / 6.5
    ("HiGT-Bot SAGPool",          "phase6_sparse_baseline.json",             "6.4","Phase 6 (hierarchical, sparse pool)"),
    ("SAGPool × SSL-FT",          "phase6_sparse_ssl_ft.json",               "6.5","Phase 6 ablations"),
    # Phase 7
    ("HiGT-Bot full (GT-edge)",   "phase7_higt_bot_edge.json",               "7",  "Phase 7 (full HiGT-Bot)"),
    ("HiGT-Bot full (GT-global)", "phase7_higt_bot_global.json",             "7",  "Phase 7 (full HiGT-Bot)"),
    ("HiGT-Bot full (hybrid 2L)", "phase7_higt_bot_hybrid.json",             "7",  "Phase 7 (full HiGT-Bot)"),
    ("HiGT-Bot full (hybrid 4L)", "phase7_higt_bot_hybrid4.json",            "7",  "Phase 7 (full HiGT-Bot)"),
]


def fmt(v, n: int = 4) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.{n}f}"
    return str(v)


def main() -> None:
    rows = []
    for label, fname, phase, category in RUNS:
        p = LOG_DIR / fname
        if not p.exists():
            rows.append({"label": label, "phase": phase, "category": category,
                          "missing": True})
            continue
        d = json.loads(p.read_text())
        t = d.get("test", {})
        rows.append({
            "label": label, "phase": phase, "category": category,
            "missing": False,
            "f1": t.get("f1"),
            "precision": t.get("precision"),
            "recall": t.get("recall"),
            "pr_auc": t.get("pr_auc"),
            "roc_auc": t.get("roc_auc"),
            "fn": t.get("fn"),
            "fp": t.get("fp"),
            "tp": t.get("tp"),
            "tn": t.get("tn"),
            "iot23_35_1_f1": (d.get("test_per_scenario", {})
                                .get("iot23-35-1", {})
                                .get("f1")),
            "ctu13_10_f1": (d.get("test_per_scenario", {})
                              .get("ctu13-10", {})
                              .get("f1")),
            "params": d.get("params"),
            "fit_seconds": d.get("fit_seconds"),
        })

    # Markdown table grouped by category
    md = ["# Phase 8.2 — Final Ablation Table", "",
          "Auto-compiled from `data/inspection_logs/phase{4-7}_*.json`.",
          "All metrics on the held-out test set (n = 59,210; n_pos = 41,063).",
          ""]
    categories: list[str] = []
    for r in rows:
        if r["category"] not in categories:
            categories.append(r["category"])
    for cat in categories:
        md.append(f"## {cat}")
        md.append("")
        md.append("| Model | F1 | PR-AUC | Recall | FN | iot23-35-1 | ctu13-10 | Params |")
        md.append("|---|---|---|---|---|---|---|---|")
        for r in rows:
            if r["category"] != cat:
                continue
            if r["missing"]:
                md.append(f"| {r['label']} | (missing) | | | | | | |")
                continue
            md.append(
                f"| {r['label']} | {fmt(r['f1'])} | {fmt(r['pr_auc'])} | "
                f"{fmt(r['recall'])} | {fmt(r['fn'],0)} | "
                f"{fmt(r['iot23_35_1_f1'])} | {fmt(r['ctu13_10_f1'])} | "
                f"{r['params'] if r['params'] is not None else '—'} |"
            )
        md.append("")

    # Headline row
    final = next((r for r in rows
                   if r["label"] == "HiGT-Bot full (GT-edge)"), None)
    if final and not final["missing"]:
        md.append("## Headline")
        md.append("")
        md.append(f"- **Final model**: {final['label']}")
        md.append(f"- **Test F1**: {fmt(final['f1'])}")
        md.append(f"- **PR-AUC**: {fmt(final['pr_auc'])}")
        md.append(f"- **Recall**: {fmt(final['recall'])}")
        md.append(f"- **FN**: {final['fn']}")
        md.append("")

    out_md = LOG_DIR / "phase8_ablation_table.md"
    out_json = LOG_DIR / "phase8_ablation_table.json"
    out_md.write_text("\n".join(md), encoding="utf-8")
    out_json.write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")
    print(f"wrote {out_md}")
    print(f"wrote {out_json}")


if __name__ == "__main__":
    main()
