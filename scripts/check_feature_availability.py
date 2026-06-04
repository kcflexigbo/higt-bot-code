"""Quantify the feature-availability mismatch across CTU-13, IoT-23, MedBIoT.

The flow schema (src/data/schema.py) marks four fields as NaN-able because
CTU-13 (binetflow) and IoT-23 (Zeek conn.log) do not expose per-packet
information: mean_iat_ms, std_iat_ms, min_pkt_size, max_pkt_size. MedBIoT
(parsed from pcap) does. This script samples a representative scenario from
each dataset and reports the fraction of flows for which each of those
features is NaN.

Output: data/inspection_logs/lodo_feature_availability.json
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

NANABLE = ["mean_iat_ms", "std_iat_ms", "min_pkt_size", "max_pkt_size"]
OUT = Path("data/inspection_logs/lodo_feature_availability.json")

SAMPLES = [
    ("ctu13-1", "CTU-13"),
    ("ctu13-13", "CTU-13"),
    ("iot23-1-1", "IoT-23"),
    ("iot23-48-1", "IoT-23"),
    ("medbiot-mirai_mal_CC_all", "MedBIoT"),
    ("medbiot-bashlite_mal_spread_all", "MedBIoT"),
    ("medbiot-torii_mal_all", "MedBIoT"),
]

records = []
print(
    f'{"Scenario":<32s}  {"Dataset":<8s}  '
    + "  ".join(f"{c[:8]:>8s}" for c in NANABLE)
    + f'  {"n_flows":>10s}'
)
for scen, ds in SAMPLES:
    p = Path(f"data/processed/{scen}.parquet")
    if not p.exists():
        print(f"{scen:<32s}  {ds:<8s}  (missing)")
        continue
    df = pd.read_parquet(p, columns=NANABLE)
    fracs = {c: float(df[c].isna().mean()) for c in NANABLE}
    n = int(len(df))
    records.append({"scenario": scen, "dataset": ds, "n_flows": n, **fracs})
    print(
        f"{scen:<32s}  {ds:<8s}  "
        + "  ".join(f"{fracs[c]:>8.4f}" for c in NANABLE)
        + f"  {n:>10,}"
    )

# Per-dataset averages
agg = {}
for ds in {r["dataset"] for r in records}:
    rows = [r for r in records if r["dataset"] == ds]
    agg[ds] = {
        c: round(sum(r[c] for r in rows) / len(rows), 4) for c in NANABLE
    }

OUT.write_text(
    json.dumps({"per_scenario": records, "per_dataset_mean": agg}, indent=2)
)
print(f"\nwrote {OUT}")
print("Per-dataset NaN fractions (mean across sampled scenarios):")
for ds, fracs in agg.items():
    print(
        f"  {ds:<10s}  "
        + "  ".join(f"{c}={fracs[c]:.3f}" for c in NANABLE)
    )
