"""Aggregate LODO result JSONs into comparison tables and a per-scenario breakdown.

Reads every data/inspection_logs/*lodo_*.json and writes:
  - data/inspection_logs/lodo_comparison_table.md
  - data/inspection_logs/lodo_per_scenario_breakdown.md
  - data/inspection_logs/lodo_{ctu13,iot23,medbiot}_summary.md
  - One row per (model, LODO direction) with F1, PR-AUC, FN/FP, ROC-AUC.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

LOGS = Path("data/inspection_logs")
HOLDOUTS = ("ctu13", "iot23", "medbiot")
MODEL_ORDER = (
    "baseline_rf", "baseline_xgb", "baseline_gat", "baseline_gin", "baseline_gine",
    "phase5_temporal_gine", "phase6_sparse", "phase7_higt_bot",
)


def _model_pretty(name: str) -> str:
    return {
        "baseline_rf": "RandomForest",
        "baseline_xgb": "XGBoost",
        "baseline_gat": "GAT (flat)",
        "baseline_gin": "GIN (flat)",
        "baseline_gine": "GINE (flat)",
        "phase5_temporal_gine": "T-GINE + skip",
        "phase6_sparse": "SAGPool (sparse)",
        "phase7_higt_bot": "GT-edge (final)",
    }.get(name, name)


def _load_result(path: Path) -> dict:
    return json.loads(path.read_text())


def collect_by_holdout() -> dict[str, dict[str, list[dict]]]:
    """Return {holdout: {model_root: [result_dicts]}}.

    A model can appear multiple times when seed replicates exist
    (only Phase 7 GT-edge does this).
    """
    out: dict[str, dict[str, list[dict]]] = {
        h: defaultdict(list) for h in HOLDOUTS
    }
    for jf in LOGS.glob("*_lodo_*.json"):
        # Filename convention: <model_root>_lodo_<holdout>[_<extras>].json
        stem = jf.stem
        # Find which holdout this file is for.
        match = next((h for h in HOLDOUTS if f"_lodo_{h}" in stem), None)
        if match is None:
            continue
        head = stem.split(f"_lodo_{match}")[0]
        out[match][head].append(_load_result(jf))
    return out


def aggregate_metric(rows: list[dict], path: str) -> tuple[float, float | None]:
    """Pull `rows[i]['test'][path]` (path uses dot notation). Return (mean, std-or-None)."""
    vals = []
    for r in rows:
        cur = r.get("test", {})
        for key in path.split("."):
            cur = cur.get(key, {}) if isinstance(cur, dict) else cur
        if isinstance(cur, (int, float)):
            vals.append(float(cur))
    if not vals:
        return float("nan"), None
    if len(vals) == 1:
        return vals[0], None
    return mean(vals), stdev(vals)


def write_comparison_table(by_holdout: dict[str, dict[str, list[dict]]]) -> Path:
    lines = ["# LODO comparison table",
              "",
              "Test F1 (mean +/- std across seeds where applicable). Mixed-pool column "
              "from existing `phase{4..7}_*.json`.",
              "",
              "| Model | LODO-CTU13 | LODO-IoT23 | LODO-MedBIoT |",
              "|---|---|---|---|"]
    for model_root in MODEL_ORDER:
        row = [_model_pretty(model_root)]
        for h in HOLDOUTS:
            rows = by_holdout[h].get(model_root, [])
            if not rows:
                row.append("--")
                continue
            m, s = aggregate_metric(rows, "f1")
            row.append(f"{m:.4f}" + (f" +/- {s:.4f}" if s is not None else ""))
        lines.append("| " + " | ".join(row) + " |")
    out = LOGS / "lodo_comparison_table.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def write_per_scenario_breakdown(by_holdout: dict[str, dict[str, list[dict]]]) -> Path:
    lines = ["# LODO per-scenario test F1",
              "",
              "GT-edge results only (the headline model). Cells = F1 (mean across seeds).",
              ""]
    for h in HOLDOUTS:
        rows = by_holdout[h].get("phase7_higt_bot", [])
        if not rows:
            continue
        lines.append(f"## LODO-{h.upper()}")
        lines.append("")
        # Average per_scenario across seeds.
        per_scen: dict[str, list[float]] = defaultdict(list)
        for r in rows:
            for scen, m in r.get("test_per_scenario", {}).items():
                f1 = m.get("f1") if isinstance(m, dict) else None
                if isinstance(f1, (int, float)):
                    per_scen[scen].append(float(f1))
        if not per_scen:
            lines.append("_(no per-scenario data)_")
            lines.append("")
            continue
        lines.append("| Scenario | F1 (mean) | seeds |")
        lines.append("|---|---|---|")
        for scen in sorted(per_scen):
            vals = per_scen[scen]
            lines.append(f"| {scen} | {mean(vals):.4f} | {len(vals)} |")
        lines.append("")
    out = LOGS / "lodo_per_scenario_breakdown.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def write_per_holdout_summaries(by_holdout: dict[str, dict[str, list[dict]]]) -> list[Path]:
    paths = []
    for h in HOLDOUTS:
        models = by_holdout[h]
        lines = [f"# LODO-{h.upper()} summary",
                  "",
                  f"Held-out dataset: **{h}**. All models trained from scratch on "
                  f"the two remaining datasets using `configs/split_lodo_{h}.yaml`.",
                  "",
                  "| Model | F1 | PR-AUC | ROC-AUC |",
                  "|---|---|---|---|"]
        for model_root in MODEL_ORDER:
            rows = models.get(model_root, [])
            if not rows:
                lines.append(f"| {_model_pretty(model_root)} | -- | -- | -- |")
                continue
            f1, _ = aggregate_metric(rows, "f1")
            pr, _ = aggregate_metric(rows, "pr_auc")
            roc, _ = aggregate_metric(rows, "roc_auc")
            lines.append(f"| {_model_pretty(model_root)} | {f1:.4f} | {pr:.4f} | {roc:.4f} |")
        out = LOGS / f"lodo_{h}_summary.md"
        out.write_text("\n".join(lines) + "\n", encoding="utf-8")
        paths.append(out)
    return paths


def main() -> None:
    by_holdout = collect_by_holdout()
    p1 = write_comparison_table(by_holdout)
    p2 = write_per_scenario_breakdown(by_holdout)
    ps = write_per_holdout_summaries(by_holdout)
    print(f"wrote {p1}")
    print(f"wrote {p2}")
    for p in ps:
        print(f"wrote {p}")


if __name__ == "__main__":
    main()
