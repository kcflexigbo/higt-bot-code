"""Per-host feature blocks for the HiGT-Bot v2 tree head.

  nbr  direction-aware neighbour aggregates (ego kept separate): out- and in-degree,
       mean / max of neighbours' x, signed-log sum of edge features, per direction
  seq  summary statistics of the host's own flow sequence (<= 256 flows from ALL
       window flows, so flows to pruned hosts count): counts and shares of outgoing
       / incoming flows, mean / std / min / max of protocol, duration, bytes,
       packets, IAT and max packet size, size repetition, inter-start-time
       regularity (beaconing), first / last flow time
  grp  in-window group similarity (BotMiner's cohort principle): distances to the
       nearest look-alike hosts in quantile-normalised x space, share of hosts
       within radius 1 and 2, number of hosts sharing an outgoing target and the
       maximum Jaccard overlap of target sets

Design follows manuscript/review/fable_beating_trees_2026-10-11.md (Sections 5–6).
"""
from __future__ import annotations

import collections

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import QuantileTransformer


def nbr_block(x: np.ndarray, edge_index: np.ndarray, edge_attr: np.ndarray) -> np.ndarray:
    n = x.shape[0]
    src, dst = edge_index
    blk = []
    for a, b in ((src, dst), (dst, src)):  # out-neighbours, then in-neighbours
        deg = np.bincount(a, minlength=n).astype(np.float32)
        s = np.zeros_like(x); np.add.at(s, a, x[b])
        mx = np.full_like(x, -np.inf); np.maximum.at(mx, a, x[b]); mx[np.isinf(mx)] = 0
        es = np.zeros((n, edge_attr.shape[1]), np.float32); np.add.at(es, a, edge_attr)
        blk += [deg[:, None], s / np.maximum(deg, 1)[:, None], mx, np.log1p(np.abs(es)) * np.sign(es)]
    return np.hstack(blk).astype(np.float32)


def seq_block(flows: np.ndarray, pad_mask: np.ndarray) -> np.ndarray:
    """flows [N, L, 13]; pad_mask [N, L] True = padding. Column 4 = outgoing bit,
    0-2 protocol one-hot, 5 duration, 6 bytes, 7 packets, 8 IAT, 10 max packet, 12 time."""
    n = flows.shape[0]
    valid = ~pad_mask
    cnt = valid.sum(1).astype(np.float32)
    out_bit = flows[:, :, 4]
    feats = [cnt[:, None]]
    for sel in (out_bit > 0.5, out_bit < 0.5):
        m = valid & sel
        k = m.sum(1).astype(np.float32)
        feats += [k[:, None], (k / np.maximum(cnt, 1))[:, None]]
        w = m.astype(np.float32)
        for col in (0, 1, 2, 5, 6, 7, 8, 10):
            v = flows[:, :, col]
            mu = (v * w).sum(1) / np.maximum(k, 1)
            var = (((v - mu[:, None]) ** 2) * w).sum(1) / np.maximum(k, 1)
            vmax = np.where(m, v, -np.inf).max(1); vmax[np.isinf(vmax)] = 0
            vmin = np.where(m, v, np.inf).min(1); vmin[np.isinf(vmin)] = 0
            feats += [mu[:, None], np.sqrt(var)[:, None], vmin[:, None], vmax[:, None]]
        rb = np.round(flows[:, :, 6] * 4) / 4
        rep = np.zeros(n, np.float32); dist = np.zeros(n, np.float32)
        gap_cv = np.zeros(n, np.float32); gap_mean = np.zeros(n, np.float32); gap_min = np.zeros(n, np.float32)
        for i in range(n):
            mi = m[i]
            if not mi.any():
                continue
            _, c = np.unique(rb[i][mi], return_counts=True)
            rep[i] = c.max() / mi.sum(); dist[i] = len(c) / mi.sum()
            t = np.sort(flows[i, :, 12][mi])
            if len(t) >= 3:
                gp = np.diff(t); gm = gp.mean()
                gap_mean[i] = gm; gap_min[i] = gp.min(); gap_cv[i] = gp.std() / (gm + 1e-6)
            else:
                gap_cv[i] = -1
        feats += [rep[:, None], dist[:, None], gap_cv[:, None], gap_mean[:, None], gap_min[:, None]]
    t = np.where(valid, flows[:, :, 12], np.nan)
    with np.errstate(all="ignore"):
        feats += [np.nan_to_num(np.nanmin(t, 1))[:, None], np.nan_to_num(np.nanmax(t, 1))[:, None]]
    return np.hstack(feats).astype(np.float32)


def fit_group_scaler(x_train: np.ndarray, seed: int = 0) -> QuantileTransformer:
    return QuantileTransformer(output_distribution="normal", n_quantiles=1000,
                               subsample=200_000, random_state=seed).fit(x_train)


def grp_block(x: np.ndarray, edge_index: np.ndarray, qt: QuantileTransformer) -> np.ndarray:
    n = x.shape[0]
    q = qt.transform(x)
    f = np.zeros((n, 8), np.float32)
    k = min(6, n)
    dist, _ = NearestNeighbors(n_neighbors=k).fit(q).kneighbors(q)
    d = dist[:, 1:] if k > 1 else np.full((n, 1), 10.0)
    f[:, 0] = d[:, 0]; f[:, 1] = d.mean(1); f[:, 2] = d[:, -1]
    if n > 1:
        rn = NearestNeighbors(radius=2.0).fit(q)
        cnt2 = np.array([len(a) - 1 for a in rn.radius_neighbors(q, radius=2.0, return_distance=False)])
        cnt1 = np.array([len(a) - 1 for a in rn.radius_neighbors(q, radius=1.0, return_distance=False)])
        f[:, 3] = cnt1 / (n - 1); f[:, 4] = cnt2 / (n - 1); f[:, 5] = cnt2
    outs: dict[int, set] = collections.defaultdict(set)
    for a, b in zip(edge_index[0], edge_index[1]):
        outs[int(a)].add(int(b))
    by_t: dict[int, set] = collections.defaultdict(set)
    for a, ts in outs.items():
        for t in ts:
            by_t[t].add(a)
    for a, ts in outs.items():
        co = set().union(*(by_t[t] for t in ts)) - {a}
        f[a, 6] = len(co)
        if co:
            f[a, 7] = max(len(ts & outs[c]) / len(ts | outs[c]) for c in co)
    return f
