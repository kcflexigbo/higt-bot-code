"""scripts/train_phase5.py with drop_last=True on the shuffled training loader.

LODO-CTU13 on graphs_v3 crashed in epoch 5 (BatchNorm "Expected more than 1 value
per channel"): 2,709 training windows at batch size 2 leave a final batch of one
window, and when that window has a single host the GIN BatchNorm sees one row.
Dropping the incomplete last batch skips one random window per epoch and nothing
else; validation and test loaders are unchanged. Same arguments as train_phase5.py.
"""
from __future__ import annotations

import torch  # noqa: F401  (import before pandas on Windows)

from torch_geometric.loader import DataLoader

import src.training.loop_phase5 as loop_phase5
from scripts import train_phase5


class _DropLastTrain(DataLoader):
    def __init__(self, *args, **kwargs):
        if kwargs.get("shuffle"):
            kwargs["drop_last"] = True
        super().__init__(*args, **kwargs)


loop_phase5.DataLoader = _DropLastTrain

# Crash-safe training: if HIGT_TRAIN_STATE is set, the per-epoch training state is
# written there and resumed after a crash (see train_one_model(state_path=...)).
import functools  # noqa: E402
import os  # noqa: E402

if os.environ.get("HIGT_TRAIN_STATE"):
    train_phase5.train_one_model = functools.partial(
        loop_phase5.train_one_model, state_path=os.environ["HIGT_TRAIN_STATE"])

if __name__ == "__main__":
    train_phase5.main()
