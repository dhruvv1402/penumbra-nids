"""Thread count by batch size, for scikit-learn forests at predict time.

A forest fitted with `n_jobs=-1` dispatches every predict through a joblib thread pool. For a large
batch that pays for itself; for one flow it is most of the cost. Measured on the UNSW detector (300
trees, 8 cores): one row takes 59 ms with all cores and 31 ms with one, and the crossover sits
between 256 rows (74 vs 114 ms, one core wins) and 2048 (176 vs 165 ms, all cores win).

Only the thread count changes, never the trees. The per-tree probabilities are summed in a
different order, so p_attack can move in the last bit (at most 4.4e-16 across the UNSW test split);
every verdict, family and alert field was identical when checked on 19,799 alerts. The attribute is set on the shared estimator; two concurrent calls of different sizes
can see each other's setting, which costs speed and never correctness.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from sklearn.ensemble import BaseEnsemble

SMALL_BATCH = 1024


def _forest(model: Any) -> Any:
    steps = getattr(model, "named_steps", None)
    est = steps.get("clf") if steps is not None else model
    return est if isinstance(est, BaseEnsemble) and hasattr(est, "n_jobs") else None


@contextmanager
def threads_for(model: Any, n_rows: int) -> Iterator[None]:
    """One thread for a small batch, the fitted setting otherwise."""
    forest = _forest(model)
    if forest is None or n_rows >= SMALL_BATCH or forest.n_jobs == 1:
        yield
        return
    fitted = forest.n_jobs
    forest.n_jobs = 1
    try:
        yield
    finally:
        forest.n_jobs = fitted
