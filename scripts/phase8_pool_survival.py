"""Phase 8.7 — Test-set-wide SAGPool survival vs. detection for the final model.

Phase 8.5 inspected one window per scenario. This pass covers the whole
canonical test set: for every node it records whether SAGPool kept it and
whether the per-node head classified it correctly, then reports recall of
bots that were kept vs. bots that were dropped (the skip-path claim).

Reuses the checkpoint/dataset loading of scripts/reeval_mixed_pool_fp32.py.
Read-only. Output: data/inspection_logs/phase8_pool_survival.{md,json}
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from src.data.dataset import SplitSpec, load_split_files
from src.data.embedding_dataset import EmbeddingGraphDataset
from src.models.higt_bot import HiGTBot
from src.utils.seeding import pick_device, set_seed

INSPECTION = Path("data/inspection_logs")
_spec = importlib.util.spec_from_file_location("reeval", Path(__file__).with_name("reeval_mixed_pool_fp32.py"))
reeval = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(reeval)


def _load(ckpt_path, te_files, device):
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    a = blob["args"]
    (e_mean, e_std), (n_mean, n_std) = blob["edge_scaler"], blob["node_scaler"]
    te_ds = EmbeddingGraphDataset(te_files, emb_root=Path("data/flow_embeddings"),
                                  edge_mean=e_mean.cpu(), edge_std=e_std.cpu(),
                                  node_mean=n_mean.cpu(), node_std=n_std.cpu())
    probe = te_ds[0]
    model = HiGTBot(d_model=int(probe.node_emb.size(1)), raw_feat_dim=int(probe.x.size(1)),
                    edge_dim=int(probe.edge_attr.size(1)), hidden=a["hidden"], pool_ratio=a["pool_ratio"],
                    gt_variant=a["gt_variant"], gt_layers=a["gt_layers"], gt_heads=a["gt_heads"],
                    num_classes=2, dropout=a["dropout"]).to(device)
    model.load_state_dict(blob["state_dict"])
    return model, DataLoader(te_ds, batch_size=int(a["batch_size"]), shuffle=False, num_workers=0)


@torch.no_grad()
def main():
    set_seed(42)
    device = pick_device()
    te_files = load_split_files("test", SplitSpec.load(reeval.CANONICAL_SPLIT))
    model, loader = _load(Path("experiments/phase7/higt_bot_edge.pt"), te_files, device)
    model.eval()

    captured = {}
    model.pool.register_forward_hook(lambda m, i, o: captured.__setitem__("perm", o[4]))
    ys, preds, kept = [], [], []
    for batch in loader:
        batch = batch.to(device)
        out = model(batch)
        logits = (out["logits"] if isinstance(out, dict) else out).float()
        k = torch.zeros(batch.num_nodes, dtype=torch.bool, device=device)
        k[captured["perm"]] = True
        ys.append(batch.y.cpu().numpy()); preds.append(logits.argmax(-1).cpu().numpy()); kept.append(k.cpu().numpy())
    y, p, kept = np.concatenate(ys), np.concatenate(preds), np.concatenate(kept)

    def grp(mask):
        n = int(mask.sum())
        return {"n": n, "detected": int((p[mask] == 1).sum()), "recall": float((p[mask] == 1).mean()) if n else None}

    bot, ben = y == 1, y == 0
    res = {
        "n_nodes": int(len(y)),
        "bot_survival": float(kept[bot].mean()), "benign_survival": float(kept[ben].mean()),
        "bots_kept": grp(bot & kept), "bots_dropped": grp(bot & ~kept),
        "benign_kept_fp": int((p[ben & kept] == 1).sum()), "benign_dropped_fp": int((p[ben & ~kept] == 1).sum()),
    }
    (INSPECTION / "phase8_pool_survival.json").write_text(json.dumps(res, indent=2))
    md = ["# Phase 8.7 — SAGPool survival vs. detection (whole test set, final model)", "",
          f"Nodes: {res['n_nodes']:,}. Bot survival {res['bot_survival']:.3f}; benign survival {res['benign_survival']:.3f}.", "",
          "| Bot group | n | detected | recall |", "|---|---|---|---|",
          f"| kept by SAGPool | {res['bots_kept']['n']:,} | {res['bots_kept']['detected']:,} | {res['bots_kept']['recall']:.4f} |",
          f"| dropped by SAGPool | {res['bots_dropped']['n']:,} | {res['bots_dropped']['detected']:,} | {res['bots_dropped']['recall']:.4f} |",
          "", f"False positives among benign hosts: kept {res['benign_kept_fp']:,}, dropped {res['benign_dropped_fp']:,}."]
    (INSPECTION / "phase8_pool_survival.md").write_text("\n".join(md) + "\n", encoding="utf8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
