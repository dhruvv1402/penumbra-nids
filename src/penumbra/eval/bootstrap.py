"""Confidence intervals and significance tests.

"Honest evaluation" means error bars. A single decimal with no interval is a claim, not a
measurement, and on a class with 44 test rows it is barely even a claim.

Two details that are easy to get wrong and quietly matter:

  * Resampling is STRATIFIED by class. An unstratified bootstrap over a set containing 44 Worms rows
    produces resamples where Worms is absent entirely, and the resulting interval is nonsense.
  * McNemar reports the discordant pair counts and the odds ratio, not just a p-value. At 82,000
    samples every difference is "significant"; the discordant counts are what say whether it is
    large enough to care about.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy import stats

from penumbra.seeds import SEED

ArrayLike = np.ndarray
MetricFn = Callable[[np.ndarray, np.ndarray], float]


@dataclass(frozen=True)
class Interval:
    point: float
    lower: float
    upper: float
    level: float = 0.95
    n_resamples: int = 0

    @property
    def width(self) -> float:
        return self.upper - self.lower

    def __str__(self) -> str:
        return f"{self.point:.4f} [{self.lower:.4f}, {self.upper:.4f}]"

    def format_pct(self) -> str:
        return f"{self.point:.1%} [{self.lower:.1%}, {self.upper:.1%}]"


def stratified_bootstrap(
    y_true: ArrayLike,
    scores: ArrayLike,
    metric: MetricFn,
    *,
    n_resamples: int = 1000,
    level: float = 0.95,
    seed: int = SEED,
) -> Interval:
    """Percentile bootstrap CI, resampling within each class separately.

    Stratifying preserves the class balance of the original sample in every resample, which is what
    keeps a rare class from vanishing and taking the interval with it.
    """
    y_true = np.asarray(y_true)
    scores = np.asarray(scores, dtype=float)
    rng = np.random.default_rng(seed)

    class_indices = [np.flatnonzero(y_true == c) for c in np.unique(y_true)]
    point = float(metric(y_true, scores))

    estimates = np.empty(n_resamples, dtype=float)
    n_valid = 0
    for _ in range(n_resamples):
        drawn = np.concatenate(
            [rng.choice(idx, size=len(idx), replace=True) for idx in class_indices if len(idx)]
        )
        try:
            estimates[n_valid] = metric(y_true[drawn], scores[drawn])
            n_valid += 1
        except ValueError:
            # A metric can be undefined on a degenerate resample (e.g. AUC when one class is
            # missing). Skip rather than counting it, so the interval reflects usable resamples.
            continue

    if n_valid == 0:
        return Interval(point, float("nan"), float("nan"), level, 0)

    alpha = (1.0 - level) / 2.0
    lo, hi = np.quantile(estimates[:n_valid], [alpha, 1.0 - alpha])
    return Interval(point, float(lo), float(hi), level, n_valid)


def moving_block_bootstrap(
    y_true: ArrayLike,
    scores: ArrayLike,
    metric: MetricFn,
    *,
    block_size: int = 200,
    n_resamples: int = 1000,
    level: float = 0.95,
    seed: int = SEED,
) -> Interval:
    """Bootstrap for temporally ordered data.

    Network traffic is autocorrelated: consecutive flows are not independent draws. An iid bootstrap
    over such a sequence treats each row as independent evidence and produces intervals that are too
    narrow. Resampling contiguous blocks preserves local dependence.

    Use this on CICIDS2017's time-ordered evaluation; the stratified version is right for the
    shuffled UNSW-NB15 and NSL-KDD splits.
    """
    y_true = np.asarray(y_true)
    scores = np.asarray(scores, dtype=float)
    n = len(y_true)
    rng = np.random.default_rng(seed)

    point = float(metric(y_true, scores))
    n_blocks = max(1, n // block_size)
    starts_max = max(1, n - block_size)

    estimates = np.empty(n_resamples, dtype=float)
    n_valid = 0
    for _ in range(n_resamples):
        starts = rng.integers(0, starts_max, size=n_blocks)
        drawn = np.concatenate([np.arange(s, min(s + block_size, n)) for s in starts])
        try:
            estimates[n_valid] = metric(y_true[drawn], scores[drawn])
            n_valid += 1
        except ValueError:
            continue

    if n_valid == 0:
        return Interval(point, float("nan"), float("nan"), level, 0)

    alpha = (1.0 - level) / 2.0
    lo, hi = np.quantile(estimates[:n_valid], [alpha, 1.0 - alpha])
    return Interval(point, float(lo), float(hi), level, n_valid)


# =================================================================================================
# Significance
# =================================================================================================


@dataclass(frozen=True)
class McNemarResult:
    """Paired comparison of two classifiers on the same test set."""

    n01: int  # model A wrong, model B right
    n10: int  # model A right, model B wrong
    statistic: float
    p_value: float
    odds_ratio: float
    n_discordant: int

    def summary(self, name_a: str = "A", name_b: str = "B") -> str:
        direction = name_b if self.n01 > self.n10 else name_a
        return "\n".join(
            [
                f"  McNemar {name_a} vs {name_b}",
                f"    {name_a} wrong / {name_b} right : {self.n01:,}",
                f"    {name_a} right / {name_b} wrong : {self.n10:,}",
                f"    discordant pairs              : {self.n_discordant:,}",
                f"    odds ratio                    : {self.odds_ratio:.3f}  (favours {direction})",
                f"    p                             : {self.p_value:.3e}",
                "    Note: at this sample size p is near-meaningless on its own. The discordant",
                "    counts and the odds ratio are what say whether the difference is worth having.",
            ]
        )


def mcnemar(y_true: ArrayLike, pred_a: ArrayLike, pred_b: ArrayLike) -> McNemarResult:
    """Exact McNemar test on the discordant pairs.

    Binomial exact rather than the chi-square approximation: with a small discordant count the
    approximation is poor, and the exact test costs nothing at these sizes.
    """
    y_true = np.asarray(y_true)
    correct_a = np.asarray(pred_a) == y_true
    correct_b = np.asarray(pred_b) == y_true

    n01 = int(np.sum(~correct_a & correct_b))
    n10 = int(np.sum(correct_a & ~correct_b))
    n_disc = n01 + n10

    if n_disc == 0:
        return McNemarResult(0, 0, 0.0, 1.0, 1.0, 0)

    p = float(stats.binomtest(min(n01, n10), n_disc, 0.5).pvalue)
    odds = (n01 / n10) if n10 else float("inf")
    statistic = float((abs(n01 - n10) - 1) ** 2 / n_disc) if n_disc else 0.0
    return McNemarResult(n01, n10, statistic, p, odds, n_disc)


def benjamini_hochberg(p_values: ArrayLike, alpha: float = 0.05) -> np.ndarray:
    """Boolean mask of rejected hypotheses under BH false-discovery-rate control.

    Needed wherever many tests run at once - notably the per-feature KS drift tests, where 42
    features at alpha=0.05 produce roughly two false alarms every window, forever, unless corrected.
    """
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    if n == 0:
        return np.array([], dtype=bool)

    order = np.argsort(p)
    ranked = p[order]
    thresholds = alpha * np.arange(1, n + 1) / n
    below = ranked <= thresholds

    rejected = np.zeros(n, dtype=bool)
    if below.any():
        cutoff = int(np.max(np.flatnonzero(below)))
        rejected[order[: cutoff + 1]] = True
    return rejected
