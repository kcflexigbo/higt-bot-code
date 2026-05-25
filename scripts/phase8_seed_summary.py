"""Phase 8.1 — compile mean ± std across seeds for the final model.

Reads phase7_higt_bot_edge.json (seed 42) + _edge_seed1.json + _edge_seed2.json
and writes mean/std/min/max for each test metric, plus per-scenario stats.
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

LOG_DIR = Path("data/inspection_logs")
RUNS = [
    ("seed=42", "phase7_higt_bot_edge.json"),
    ("seed=1",  "phase7_higt_bot_edge_seed1.json"),
    ("seed=2",  "phase7_higt_bot_edge_seed2.json"),
]
METRICS = ["f1", "precision", "recall", "pr_auc", "roc_auc", "fn"]
SCENARIOS_WATCH = ["iot23-35-1", "ctu13-10", "iot23-7-1", "ctu13-3",
                    "medbiot-bashlite_mal_spread_all"]


def fmt(values: list[float], digits: int = 4) -> str:
    if not values:
        return "—"
    if len(values) == 1:
        return f"{values[0]:.{digits}f}"
    mean = statistics.mean(values)
    std = statistics.pstdev(values)
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def main() -> None:
    rows = []
    for tag, fname in RUNS:
        p = LOG_DIR / fname
        if not p.exists():
            print(f"missing: {p}")
            rows.append({"tag": tag, "missing": True})
            continue
        d = json.loads(p.read_text())
        rows.append({"tag": tag, "missing": False, "data": d})

    md = ["# Phase 8.1 — Seed Sweep Summary", "",
          "Final HiGT-Bot (GT-edge) trained with three seeds. Reported as "
          "mean ± std across available seeds. Higher seed count gives tighter "
          "bars; even a 2-seed pair lets us check stability.",
          ""]

    md.append("## Aggregate metrics")
    md.append("")
    md.append("| Metric | " + " | ".join(r["tag"] for r in rows) + " | mean ± std |")
    md.append("|---|" + "|".join(["---"] * len(rows)) + "|---|")
    for m in METRICS:
        cells = []
        vals = []
        for r in rows:
            if r["missing"]:
                cells.append("—")
                continue
            t = r["data"]["test"]
            v = t.get(m)
            if v is None:
                cells.append("—")
                continue
            cells.append(f"{v:.4f}" if isinstance(v, float) else str(v))
            vals.append(float(v))
        md.append(f"| {m} | " + " | ".join(cells) + " | " + fmt(vals) + " |")

    md.append("")
    md.append("## Per-scenario test F1")
    md.append("")
    md.append("| Scenario | " + " | ".join(r["tag"] for r in rows) + " | mean ± std |")
    md.append("|---|" + "|".join(["---"] * len(rows)) + "|---|")
    for sc in SCENARIOS_WATCH:
        cells = []
        vals = []
        for r in rows:
            if r["missing"]:
                cells.append("—")
                continue
            per = r["data"].get("test_per_scenario", {}).get(sc, {})
            v = per.get("f1")
            if v is None:
                cells.append("—")
                continue
            cells.append(f"{v:.4f}")
            vals.append(float(v))
        md.append(f"| {sc} | " + " | ".join(cells) + " | " + fmt(vals) + " |")

    md.append("")
    md.append("Per-scenario stability is the headline check — variance >0.02 "
              "on iot23-35-1 (31 positives) is expected, but easy scenarios "
              "should be tight across seeds.")

    out_md = LOG_DIR / "phase8_seed_summary.md"
    out_json = LOG_DIR / "phase8_seed_summary.json"
    out_md.write_text("\n".join(md), encoding="utf-8")
    out_json.write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")
    print(f"wrote {out_md}")
    print(f"wrote {out_json}")


if __name__ == "__main__":
    main()
