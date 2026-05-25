"""Analyze why TemporalGINE regresses vs RF on ctu13-10 and iot23-35-1."""
import os
import glob
import torch
import numpy as np
from collections import defaultdict

ROOT = r"C:\Users\Kenneth Igbo\Documents\Yakub_thesis\higt-bot-code"
FLOW_DIR = os.path.join(ROOT, "data", "flow_seqs")
GRAPH_DIR = os.path.join(ROOT, "data", "graphs")

# Regression scenarios + "good" comparison scenarios
SCENARIOS = ["ctu13-10", "iot23-35-1", "ctu13-2", "iot23-48-1"]

def analyze_scenario(scen):
    flow_files = sorted(glob.glob(os.path.join(FLOW_DIR, scen, "window_*.pt")))
    graph_files = sorted(glob.glob(os.path.join(GRAPH_DIR, scen, "window_*.pt")))
    # match by basename
    g_map = {os.path.basename(p): p for p in graph_files}

    bot_flow_counts = []
    benign_flow_counts = []
    nodes_per_window = []
    edges_per_window = []
    bot_feats = []
    benign_feats = []
    n_pos = 0
    n_total = 0

    for fp in flow_files:
        bn = os.path.basename(fp)
        if bn not in g_map:
            continue
        try:
            fdata = torch.load(fp, weights_only=False, map_location="cpu")
            gdata = torch.load(g_map[bn], weights_only=False, map_location="cpu")
        except Exception as e:
            print(f"load err {fp}: {e}")
            continue

        if isinstance(fdata, dict):
            flow_mask = fdata.get("flow_mask", None)
        else:
            flow_mask = getattr(fdata, "flow_mask", None)
        if flow_mask is None:
            continue
        # flow_mask: [N, max_flows] bool
        # flow_mask: True = padded; real flows = ~mask
        counts = (~flow_mask).sum(dim=1).cpu().numpy()
        y = gdata.y.cpu().numpy() if hasattr(gdata, "y") else None
        x = gdata.x.cpu().numpy() if hasattr(gdata, "x") else None

        nodes_per_window.append(gdata.num_nodes)
        edges_per_window.append(gdata.edge_index.shape[1] if hasattr(gdata, "edge_index") else 0)

        if y is not None:
            bot_idx = np.where(y == 1)[0]
            benign_idx = np.where(y == 0)[0]
            n_pos += len(bot_idx)
            n_total += len(y)
            if len(counts) == len(y):
                bot_flow_counts.extend(counts[bot_idx].tolist())
                benign_flow_counts.extend(counts[benign_idx].tolist())
            if x is not None and x.shape[0] == len(y):
                if len(bot_idx) > 0:
                    bot_feats.append(x[bot_idx])
                if len(benign_idx) > 0:
                    benign_feats.append(x[benign_idx])

    bot_flow_counts = np.array(bot_flow_counts) if bot_flow_counts else np.array([0])
    benign_flow_counts = np.array(benign_flow_counts) if benign_flow_counts else np.array([0])
    bot_feats = np.concatenate(bot_feats, axis=0) if bot_feats else np.zeros((0, 9))
    benign_feats = np.concatenate(benign_feats, axis=0) if benign_feats else np.zeros((0, 9))

    print(f"\n=== {scen} ===")
    print(f"  windows: {len(nodes_per_window)}")
    print(f"  n_total nodes: {n_total}, n_pos: {n_pos}, pos_frac: {n_pos/max(1,n_total):.4f}")
    print(f"  nodes/window: mean={np.mean(nodes_per_window):.1f} median={np.median(nodes_per_window):.1f} min={np.min(nodes_per_window)} max={np.max(nodes_per_window)}")
    print(f"  edges/window: mean={np.mean(edges_per_window):.1f} median={np.median(edges_per_window):.1f}")
    print(f"  BOT  flows/node: mean={bot_flow_counts.mean():.2f} median={np.median(bot_flow_counts):.1f} p10={np.percentile(bot_flow_counts,10):.1f} p90={np.percentile(bot_flow_counts,90):.1f} (n={len(bot_flow_counts)})")
    print(f"  BENIGN flows/node: mean={benign_flow_counts.mean():.2f} median={np.median(benign_flow_counts):.1f} p10={np.percentile(benign_flow_counts,10):.1f} p90={np.percentile(benign_flow_counts,90):.1f} (n={len(benign_flow_counts)})")
    # fraction of bots with <=1 flow
    if len(bot_flow_counts) > 0:
        frac_le1 = (bot_flow_counts <= 1).mean()
        frac_le3 = (bot_flow_counts <= 3).mean()
        print(f"  BOT frac with <=1 flow: {frac_le1:.3f}, <=3 flows: {frac_le3:.3f}")

    if bot_feats.shape[0] > 0:
        print(f"  BOT feat means:    {np.round(bot_feats.mean(axis=0), 3).tolist()}")
        print(f"  BOT feat stds:     {np.round(bot_feats.std(axis=0), 3).tolist()}")
    if benign_feats.shape[0] > 0:
        print(f"  BENIGN feat means: {np.round(benign_feats.mean(axis=0), 3).tolist()}")
        print(f"  BENIGN feat stds:  {np.round(benign_feats.std(axis=0), 3).tolist()}")

    # separability: cohen's d per feature
    if bot_feats.shape[0] > 1 and benign_feats.shape[0] > 1:
        mu_b, mu_n = bot_feats.mean(0), benign_feats.mean(0)
        sd_b, sd_n = bot_feats.std(0), benign_feats.std(0)
        pooled = np.sqrt((sd_b**2 + sd_n**2) / 2 + 1e-9)
        d = (mu_b - mu_n) / pooled
        print(f"  cohen_d per feat:  {np.round(d, 3).tolist()}")
        print(f"  max |cohen_d|: {np.max(np.abs(d)):.3f}")

    return {
        "scen": scen,
        "bot_flow_mean": bot_flow_counts.mean(),
        "benign_flow_mean": benign_flow_counts.mean(),
    }

results = []
for s in SCENARIOS:
    results.append(analyze_scenario(s))

print("\n=== SUMMARY ===")
for r in results:
    print(f"{r['scen']:20s} bot_flows/node={r['bot_flow_mean']:.2f}  benign_flows/node={r['benign_flow_mean']:.2f}")
