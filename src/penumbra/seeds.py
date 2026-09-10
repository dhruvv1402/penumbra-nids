"""One seed, set in one place.

Reproducibility is a claim this project makes explicitly ("every number regenerable with one
command"), so the seed is not scattered through call sites. Estimators take `random_state=SEED`
from here; anything stochastic that cannot be seeded is documented as such in the report.
"""

from __future__ import annotations

import os
import random

import numpy as np

from penumbra.config import settings

SEED: int = settings().seed


def seed_everything(seed: int | None = None) -> int:
    """Seed Python, NumPy and hash randomisation. Returns the seed actually used."""
    s = SEED if seed is None else seed
    random.seed(s)
    np.random.seed(s)
    # Affects set/dict iteration order in subprocesses; harmless here, and it removes one source
    # of run-to-run variation when a script shells out.
    os.environ.setdefault("PYTHONHASHSEED", str(s))
    return s
