"""Phase 6 dataset: load graphs + cached encoder embeddings.

Reads `data/flow_embeddings/<scenario>/window_<NNNN>.parquet` (written by
`scripts/cache_embeddings.py`) and attaches the [N, d_model] embedding as
`g.node_emb`. The original `g.x` (9-d raw features) is preserved so the GIN
can still see the raw skip; `g.flows` is NOT loaded — that's the whole point
of caching.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

EMB_DIR = Path("data/flow_embeddings")


def emb_path(scenario: str, window_idx: int, root: Path = EMB_DIR) -> Path:
    return root / scenario / f"window_{window_idx:05d}.parquet"


class EmbeddingGraphDataset(Dataset):
    """Lazy dataset returning (Data with .node_emb, .x, .edge_index, .edge_attr, .y).

    `node_mean/std` and `edge_mean/std` standardize raw features per the
    training-set statistics. Embeddings are NOT scaled (encoder output was
    already normalized via LayerNorm).
    """

    def __init__(
        self,
        graph_files: list[Path],
        *,
        emb_root: Path = EMB_DIR,
        edge_mean: torch.Tensor | None = None,
        edge_std: torch.Tensor | None = None,
        node_mean: torch.Tensor | None = None,
        node_std: torch.Tensor | None = None,
    ) -> None:
        self.graph_files = list(graph_files)
        self.emb_root = emb_root
        self.edge_mean = edge_mean
        self.edge_std = edge_std
        self.node_mean = node_mean
        self.node_std = node_std

    def __len__(self) -> int:
        return len(self.graph_files)

    def __getitem__(self, idx: int) -> Data:
        g = torch.load(self.graph_files[idx], weights_only=False)
        p = emb_path(g.scenario, int(g.window_idx), self.emb_root)
        df = pd.read_parquet(p)
        if list(df["node_ip"]) != list(g.node_ips):
            raise ValueError(
                f"embedding/graph node_ips mismatch for {g.scenario} w{g.window_idx}"
            )
        dim_cols = [c for c in df.columns if c.startswith("dim_")]
        g.node_emb = torch.from_numpy(df[dim_cols].to_numpy()).float()
        if self.edge_mean is not None and self.edge_std is not None and g.edge_attr is not None:
            g.edge_attr = ((g.edge_attr - self.edge_mean) / self.edge_std).float()
        if self.node_mean is not None and self.node_std is not None and g.x is not None:
            g.x = ((g.x - self.node_mean) / self.node_std).float()
        return g
