"""Per-entity flow sequences for the sequence head.

The per-flow model asks "is this connection unusual". The sequence model asks "is what this host
has been *doing* unusual", which is a different question and the only one that can catch a slow
multi-stage attack. Every individual flow in a low-and-slow scan is unremarkable. The sequence is
not.

## What a window is

For each flow *t* from source *s*, the window is the **K flows from *s* that arrived immediately
before *t*, plus *t* itself**, oldest first. Shorter histories are left-padded with zeros and
carry a mask, because a host's first flow genuinely has no history and pretending otherwise (by
repeating the current flow backwards, say) manufactures a pattern that was never on the wire.

## Causality, again

Position K-1 is the current flow and positions 0..K-2 are strictly earlier flows from the same
source. Nothing after *t* is ever in *t*'s window. The temptation to centre the window on *t* is
real — it performs better — and it is a time-travel leak, because at inference time the flows
after *t* do not exist yet.

`test_windows.py` asserts this the same way the graph features do: appending future traffic must
leave earlier windows bit-identical.

## Memory, and the wrong way to solve it

A (n, K, d) float32 tensor at n=1.2M, K=16, d=78 is 5.9 GB and this machine has 16, so the output
has to be thinned. The obvious thinning is wrong: truncating the time-ordered stream would train
CICIDS2017 almost entirely on Monday, which is 371,624 flows containing zero attacks.

`keep_every` thins by stride instead. History accumulates from every flow — only the materialised
windows are subsampled — so the whole time range survives and class prevalence is unchanged, which
leaves TPR and FPR unbiased.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class WindowConfig:
    """Sequence geometry.

    `length` 16 is a compromise: long enough to contain a port sweep's shape, short enough that
    most hosts in a 2-day capture actually have that many flows. `max_gap_seconds` ends a sequence
    when a host goes quiet — two bursts separated by six hours are two sessions, and splicing them
    into one window invents a transition that never happened.
    """

    length: int = 16
    max_gap_seconds: float = 300.0


@dataclass
class Sequences:
    """The tensor, plus what is needed to interpret it."""

    X: np.ndarray  # (n, length, d) float32, oldest first, zero-padded at the front
    mask: np.ndarray  # (n, length) bool - True where a real flow sits
    y: np.ndarray  # (n,) label of the CURRENT flow, i.e. the last unmasked position
    index: np.ndarray  # (n,) row positions in the source frame, for joining results back
    features: list[str]

    @property
    def length(self) -> int:
        return int(self.X.shape[1])

    @property
    def history_depth(self) -> np.ndarray:
        """Real flows per window, current flow included. 1 means no history at all."""
        return self.mask.sum(axis=1)

    def summary(self) -> str:
        depth = self.history_depth
        return (
            f"{len(self.X):,} sequences of {self.length} x {self.X.shape[2]} features\n"
            f"  median history {int(np.median(depth))} flows, "
            f"{float((depth == 1).mean()):.1%} have no history at all\n"
            f"  {float(self.y.mean()):.1%} of current flows are attacks"
        )


def build(
    X: pd.DataFrame,
    y: np.ndarray,
    meta: pd.DataFrame,
    *,
    config: WindowConfig | None = None,
    max_rows: int | None = None,
    keep_every: int = 1,
) -> Sequences:
    """Assemble one causal window per row, grouped by source entity.

    Rows are visited in timestamp order; the returned tensor is in that same order, and `index`
    carries the original row positions so predictions can be joined back.

    `keep_every` subsamples which rows get a **materialised** window. History still accumulates
    from every flow — only the output tensor is thinned. This matters more than it looks: a
    (1.2M, 16, 78) float32 tensor is 5.9 GB and this machine has 16, so some cap is unavoidable,
    and the obvious one is wrong. Truncating in time order would have trained CICIDS2017 almost
    entirely on Monday, which is 371,624 rows containing zero attacks. A stride keeps the whole
    time range and leaves class prevalence unchanged, so TPR and FPR are unbiased by it.

    `max_rows` is a hard safety cap applied after the stride. It DOES truncate in time order, so
    leave it None unless you want that.
    """
    config = config or WindowConfig()
    required = {"Timestamp", "Src IP"}
    missing = required - set(meta.columns)
    if missing:
        raise ValueError(f"sequence windows need {sorted(missing)}; this dataset does not carry them")

    features = list(X.columns)
    values = X.to_numpy(dtype=np.float32, na_value=0.0, copy=False)
    labels = np.asarray(y).astype(np.int8)

    times = pd.to_datetime(meta["Timestamp"], errors="coerce")
    seconds = (times - pd.Timestamp("1970-01-01")).dt.total_seconds().to_numpy(dtype=float)
    usable = ~np.isnan(seconds)
    src = meta["Src IP"].astype(str).to_numpy()

    order = np.argsort(np.where(usable, seconds, np.inf), kind="stable")[: int(usable.sum())]

    # Which rows get an output window. Every row still contributes to its source's history.
    materialise = np.zeros(len(order), dtype=bool)
    materialise[:: max(1, keep_every)] = True
    if max_rows is not None and int(materialise.sum()) > max_rows:
        cutoff = int(np.searchsorted(np.cumsum(materialise), max_rows, side="left")) + 1
        materialise[cutoff:] = False

    k, d = config.length, values.shape[1]
    n = int(materialise.sum())
    out = np.zeros((n, k, d), dtype=np.float32)
    mask = np.zeros((n, k), dtype=bool)
    out_y = np.zeros(n, dtype=np.int8)
    out_index = np.zeros(n, dtype=np.int64)

    # Per-source history of (time, row position). Only the last K-1 are ever needed.
    history: dict[str, deque[tuple[float, int]]] = {}

    slot = 0
    for step, position in enumerate(order):
        now = float(seconds[position])
        s = src[position]
        past = history.get(s)
        if past is None:
            past = history[s] = deque(maxlen=config.length - 1)

        # A long silence ends the session. Two bursts six hours apart are two sessions, and
        # splicing them makes the model learn a transition that never occurred.
        if past and (now - past[-1][0]) > config.max_gap_seconds:
            past.clear()

        if materialise[step]:
            # Oldest first, current flow last. Left-padded, so the final position is always `now`
            # and a recurrent layer always ends on the flow being classified.
            window = [*past, (now, int(position))]
            start = k - len(window)
            for offset, (_, row) in enumerate(window):
                out[slot, start + offset] = values[row]
            mask[slot, start:] = True
            out_y[slot] = labels[position]
            out_index[slot] = position
            slot += 1

        # READ BEFORE WRITE, as in entity_graph: the current flow joins the history only after it
        # has been measured against it.
        past.append((now, int(position)))

    return Sequences(X=out, mask=mask, y=out_y, index=out_index, features=features)
