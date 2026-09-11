"""Entity-graph features over a causal sliding window.

A per-flow model sees one connection at a time, which is the wrong unit for the attacks that matter
most. A port sweep is not an unusual flow — each individual connection looks ordinary. What makes
it a sweep is *one source touching many destinations in a short time*, and that fact does not exist
in any single row. These features put it there.

The graph is implicit: sources and destinations are nodes, flows are edges, and what we compute is
each node's local degree over a trailing time window. Fan-out (one source, many destinations) is
scanning. Fan-in (many sources, one destination) is distributed flooding. Neither is visible
per-flow.

## Causality is the entire correctness argument

Every feature for flow *t* is computed from flows that arrived **strictly before** *t*. Not before
*or at* — a flow must not contribute to its own fan-out count, or every first-of-its-kind flow
would look different from every subsequent one for a reason that has nothing to do with the
traffic.

Getting this wrong produces a model that scores beautifully and cannot be deployed, because at
inference time the future has not happened yet. It is the single easiest way to fake a result in
this whole project, so `test_entity_graph.py` asserts it directly: appending future traffic must
leave every earlier row's features bit-identical.

## Why this is CICIDS2017 only

It needs `Src IP`, `Dst IP`, `Dst Port` and `Timestamp`. UNSW-NB15's published split has none of
them, and its `ct_*` columns are already entity-window aggregates computed over a 100-connection
window by the original pipeline — so on UNSW this would be recomputing, worse, something that is
already in the data.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

# Ordered, and the order is the column order of the emitted frame.
GRAPH_FEATURES: list[str] = [
    "g_src_flows",
    "g_src_distinct_dst",
    "g_src_distinct_dport",
    "g_src_dport_entropy",
    "g_src_mean_gap",
    "g_dst_flows",
    "g_dst_distinct_src",
    "g_pair_flows",
    "g_src_fanout_ratio",
]


@dataclass(frozen=True)
class GraphConfig:
    """Window geometry.

    `window_seconds` is the trailing horizon. 60s is the conventional choice for scan detection and
    is short enough that a busy server does not accumulate a permanently high fan-in, which would
    make the feature constant and therefore useless.

    `max_events` caps per-entity memory. A host generating a million flows in the window is either
    a load balancer or a flood; either way the 5,000th event tells us nothing the 4,999th did not,
    and without the cap one entity can hold the whole capture in memory.
    """

    window_seconds: float = 60.0
    max_events: int = 5_000


class _EntityWindow:
    """A trailing window of events for one node, with distinct-value counts maintained in step.

    Recomputing `len(set(...))` per row would make this quadratic. Incremental counters make each
    row O(1) amortised, which is the difference between 40 seconds and an afternoon on 2.7M flows.
    """

    __slots__ = ("times", "keys", "counts", "secondary", "secondary_counts")

    def __init__(self) -> None:
        self.times: deque[float] = deque()
        self.keys: deque[Any] = deque()
        self.counts: Counter[Any] = Counter()
        self.secondary: deque[Any] = deque()
        self.secondary_counts: Counter[Any] = Counter()

    def evict(self, cutoff: float, max_events: int) -> None:
        while self.times and (self.times[0] < cutoff or len(self.times) > max_events):
            self.times.popleft()
            key = self.keys.popleft()
            self.counts[key] -= 1
            if self.counts[key] <= 0:
                del self.counts[key]
            if self.secondary:
                skey = self.secondary.popleft()
                self.secondary_counts[skey] -= 1
                if self.secondary_counts[skey] <= 0:
                    del self.secondary_counts[skey]

    def push(self, time: float, key: Any, secondary: Any = None) -> None:
        self.times.append(time)
        self.keys.append(key)
        self.counts[key] += 1
        if secondary is not None:
            self.secondary.append(secondary)
            self.secondary_counts[secondary] += 1

    def __len__(self) -> int:
        return len(self.times)

    @property
    def span(self) -> float:
        return (self.times[-1] - self.times[0]) if len(self.times) > 1 else 0.0


def _entropy(counts: Counter[Any]) -> float:
    """Shannon entropy of the distinct-value distribution, in bits.

    Count alone cannot separate "200 connections to one port" from "200 connections spread over 200
    ports". The first is a busy service; the second is a sweep. Entropy is what tells them apart.
    """
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    p = np.fromiter(counts.values(), dtype=float, count=len(counts)) / total
    return float(-(p * np.log2(p)).sum())


def compute(meta: pd.DataFrame, *, config: GraphConfig | None = None) -> pd.DataFrame:
    """Causal entity-graph features, one row per input row, in the input's row order.

    `meta` needs `Timestamp`, `Src IP`, `Dst IP` and `Dst Port` — the columns CICIDS2017 carries
    alongside its features and which never reach the model as features themselves.

    Rows are processed in timestamp order and written back to their original positions, so the
    caller can concatenate the result to a feature frame without reindexing.
    """
    config = config or GraphConfig()
    n = len(meta)
    out = np.zeros((n, len(GRAPH_FEATURES)), dtype=np.float32)
    if n == 0:
        return pd.DataFrame(out, columns=GRAPH_FEATURES)

    required = {"Timestamp", "Src IP", "Dst IP", "Dst Port"}
    missing = required - set(meta.columns)
    if missing:
        raise ValueError(f"entity graph needs {sorted(missing)}; this dataset does not carry them")

    times = pd.to_datetime(meta["Timestamp"], errors="coerce")
    seconds = (times - pd.Timestamp("1970-01-01")).dt.total_seconds().to_numpy(dtype=float)
    # A row with no usable timestamp cannot be placed on the timeline at all, so it takes no part:
    # it reads zeros and contributes to nobody's window. Dropping it instead would shift every
    # other row's position in the frame, and giving it a sentinel time would silently group every
    # unparseable row into one dense fake burst.
    usable = ~np.isnan(seconds)

    src = meta["Src IP"].astype(str).to_numpy()
    dst = meta["Dst IP"].astype(str).to_numpy()
    dport = meta["Dst Port"].astype(str).to_numpy()

    # Stable sort: flows sharing a timestamp keep their file order, so a run is reproducible.
    order = np.argsort(np.where(usable, seconds, np.inf), kind="stable")[: int(usable.sum())]

    src_windows: dict[str, _EntityWindow] = {}
    dst_windows: dict[str, _EntityWindow] = {}
    pair_windows: dict[tuple[str, str], _EntityWindow] = {}

    for position in order:
        now = float(seconds[position])
        cutoff = now - config.window_seconds
        s, d, p = src[position], dst[position], dport[position]

        s_win = src_windows.get(s)
        if s_win is None:
            s_win = src_windows[s] = _EntityWindow()
        d_win = dst_windows.get(d)
        if d_win is None:
            d_win = dst_windows[d] = _EntityWindow()
        pair_key = (s, d)
        p_win = pair_windows.get(pair_key)
        if p_win is None:
            p_win = pair_windows[pair_key] = _EntityWindow()

        for window in (s_win, d_win, p_win):
            window.evict(cutoff, config.max_events)

        # READ BEFORE WRITE. This ordering is the causality guarantee: the current flow is measured
        # against its history and only afterwards becomes part of it.
        n_src = len(s_win)
        distinct_dst = len(s_win.counts)
        distinct_dport = len(s_win.secondary_counts)
        out[position] = (
            n_src,
            distinct_dst,
            distinct_dport,
            _entropy(s_win.secondary_counts),
            (s_win.span / (n_src - 1)) if n_src > 1 else 0.0,
            len(d_win),
            len(d_win.counts),
            len(p_win),
            (distinct_dst / n_src) if n_src else 0.0,
        )

        s_win.push(now, d, p)
        d_win.push(now, s)
        p_win.push(now, p)

    return pd.DataFrame(out, columns=GRAPH_FEATURES)


def attach(X: pd.DataFrame, meta: pd.DataFrame, *, config: GraphConfig | None = None) -> pd.DataFrame:
    """Feature frame with the graph columns appended, metadata left out."""
    graph = compute(meta, config=config)
    return pd.concat([X.reset_index(drop=True), graph], axis=1)
