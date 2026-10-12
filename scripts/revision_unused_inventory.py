"""Inventory of graphs_v3 scenarios not used by configs/split.yaml (review item C1)."""
import torch  # noqa: F401
import json
from pathlib import Path
from src.data.dataset import SplitSpec

G = Path("data/graphs_v3"); OUT = Path("data/inspection_logs/revision")
spec = SplitSpec.load()
used = set(spec.train_scenarios) | set(spec.holdout_test_scenarios)
rows = []
for d in sorted(p for p in G.iterdir() if p.is_dir()):
    if d.name in used:
        continue
    files = sorted(d.glob("window_*.pt"))
    n = b = gw = 0
    for f in files:
        g = torch.load(f, weights_only=False)
        n += int(g.y.numel()); bb = int(g.y.sum()); b += bb; gw += bb > 0
    rows.append({"scenario": d.name, "windows": len(files), "hosts": n, "bots": b, "bot_windows": gw,
                 "listed_excluded": d.name in spec.excluded})
    print(rows[-1], flush=True)
(OUT / "unused_inventory.json").write_text(json.dumps(rows, indent=2))
