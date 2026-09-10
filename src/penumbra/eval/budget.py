"""Matched budget: comparing detectors at equal cost to the analyst.

The mistake this module exists to prevent is easy to make and hard to see in a results table:

    Adding any second detector to a system raises recall.

If configuration A raises 2,000 alerts and B raises 6,000, and B catches more attacks, that says
nothing about B. It might be better. It might just be louder. A SOC has fixed analyst-hours, so the
only meaningful question is: **at the same cost, which catches more?**

## What "the same cost" means, and why the obvious answer is wrong here

The intuitive budget is total alert count. On these datasets that is actively misleading, and we
found out by running it: UNSW-NB15's test set is 55% attack and NSL-KDD's is 57%. Capping total
alerts at 50 per 1,000 flows on a set where 570 per 1,000 rows ARE attacks caps recall at about 9%
before any detector has said anything. The budget gets spent on true positives.

A real network is 99.9%+ benign, so operational alert volume is almost entirely FALSE positives.
The faithful analogue on a prevalence-inflated test set is therefore to match the number of BENIGN
rows flagged - i.e. to match FPR. That is:

  * prevalence-invariant, so it means the same thing here and on a real network;
  * the actual driver of analyst workload;
  * still a strictly fair comparison, since both configurations flag the same number of benign rows.

`match="fpr"` is the default for that reason. `match="alerts"` is kept for datasets whose prevalence
is realistic, and for showing the contrast.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BudgetedResult:
    """One configuration at a fixed cost."""

    name: str
    threshold: float
    n_alerts: int
    n_benign_flagged: int
    recall: float
    precision: float
    fpr: float
    tp: int
    fp: int
    fn: int

    def summary(self) -> str:
        return (
            f"  {self.name:<28} alerts={self.n_alerts:>7,}  benign_flagged={self.n_benign_flagged:>6,}  "
            f"recall={self.recall:.4f}  precision={self.precision:.4f}  FPR={self.fpr:.4f}"
        )


def threshold_for_alert_count(scores: np.ndarray, budget: int) -> float:
    """Threshold admitting approximately `budget` rows overall."""
    scores = np.asarray(scores, dtype=float)
    if budget <= 0:
        return float(np.inf)
    if budget >= len(scores):
        return float(-np.inf)
    return float(np.partition(scores, -budget)[-budget])


def threshold_for_benign_count(scores: np.ndarray, y_true: np.ndarray, budget: int) -> float:
    """Threshold admitting approximately `budget` BENIGN rows.

    Attacks above the threshold come along for free - which is precisely the point. On a real
    network the analyst's day is set by how many benign flows get flagged, not by how many alerts
    happen to be genuine.
    """
    benign_scores = np.asarray(scores, dtype=float)[np.asarray(y_true) == 0]
    if budget <= 0:
        return float(np.inf)
    if budget >= len(benign_scores):
        return float(-np.inf)
    return float(np.partition(benign_scores, -budget)[-budget])


def evaluate_at_threshold(
    name: str, y_true: np.ndarray, scores: np.ndarray, threshold: float
) -> BudgetedResult:
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    flagged = scores >= threshold

    tp = int(np.sum(flagged & (y_true == 1)))
    fp = int(np.sum(flagged & (y_true == 0)))
    fn = int(np.sum(~flagged & (y_true == 1)))
    n_neg = int(np.sum(y_true == 0))
    n_alerts = int(flagged.sum())

    return BudgetedResult(
        name=name,
        threshold=threshold,
        n_alerts=n_alerts,
        n_benign_flagged=fp,
        recall=tp / (tp + fn) if (tp + fn) else float("nan"),
        precision=tp / n_alerts if n_alerts else float("nan"),
        fpr=fp / n_neg if n_neg else float("nan"),
        tp=tp,
        fp=fp,
        fn=fn,
    )


@dataclass
class BudgetComparison:
    """Several configurations at one matched cost."""

    match: str  # "fpr" | "alerts"
    budget_value: float  # target FPR, or alerts per 1,000 flows
    budget_count: int  # realised benign-row budget, or alert budget
    results: dict[str, BudgetedResult]

    def best(self) -> BudgetedResult:
        return max(self.results.values(), key=lambda r: r.recall)

    def delta(self, baseline: str, candidate: str) -> float:
        """Recall gained by `candidate` over `baseline` at equal cost.

        The only number in the LOAFO experiment permitted to be called an improvement.
        """
        return self.results[candidate].recall - self.results[baseline].recall

    def summary(self) -> str:
        if self.match == "fpr":
            head = (
                f"  Matched FPR: {self.budget_value:.2%} of benign rows flagged "
                f"({self.budget_count:,} benign rows)"
            )
        else:
            head = f"  Matched alert volume: {self.budget_count:,} alerts ({self.budget_value:.1f} per 1,000)"
        return "\n".join([head, "", *(r.summary() for r in self.results.values())])


def compare_at_matched_budget(
    y_true: np.ndarray,
    scored: Mapping[str, np.ndarray],
    *,
    match: str = "fpr",
    budget_fpr: float = 0.01,
    budget_per_1k: float | None = None,
) -> BudgetComparison:
    """Compare configurations at equal cost to the analyst.

    `match="fpr"` (default) equalises the number of benign rows flagged - the honest choice on a
    prevalence-inflated test set. `match="alerts"` equalises total alert count.
    """
    y_true = np.asarray(y_true).astype(int)
    n = len(y_true)

    if match == "fpr":
        n_benign = int(np.sum(y_true == 0))
        budget_count = max(1, int(round(budget_fpr * n_benign)))
        results = {
            name: evaluate_at_threshold(name, y_true, s, threshold_for_benign_count(s, y_true, budget_count))
            for name, s in scored.items()
        }
        return BudgetComparison("fpr", budget_fpr, budget_count, results)

    if match == "alerts":
        if budget_per_1k is None:
            raise ValueError('match="alerts" needs budget_per_1k')
        budget_count = max(1, int(round(budget_per_1k * n / 1000.0)))
        results = {
            name: evaluate_at_threshold(name, y_true, s, threshold_for_alert_count(s, budget_count))
            for name, s in scored.items()
        }
        return BudgetComparison("alerts", budget_per_1k, budget_count, results)

    raise ValueError(f'unknown match {match!r}; expected "fpr" or "alerts"')


def budget_curve(
    y_true: np.ndarray,
    scored: Mapping[str, np.ndarray],
    *,
    fprs: tuple[float, ...] = (0.001, 0.005, 0.01, 0.02, 0.05, 0.10),
) -> list[BudgetComparison]:
    """Recall across a range of matched FPRs.

    A single operating point invites the objection that it was chosen to flatter one configuration.
    If the ordering holds across the whole curve, it is a property of the detectors rather than of
    the threshold.
    """
    return [compare_at_matched_budget(y_true, scored, match="fpr", budget_fpr=f) for f in fprs]


def format_curve(curve: list[BudgetComparison], *, baseline: str, candidate: str) -> str:
    lines = [
        f"  {'FPR':>8} {'benign flagged':>15} {baseline:>20} {candidate:>22} {'delta':>9}",
        f"  {'-' * 8} {'-' * 15} {'-' * 20} {'-' * 22} {'-' * 9}",
    ]
    for comp in curve:
        b, c = comp.results.get(baseline), comp.results.get(candidate)
        if b is None or c is None:
            continue
        lines.append(
            f"  {comp.budget_value:>8.3%} {comp.budget_count:>15,} {b.recall:>20.4f} "
            f"{c.recall:>22.4f} {c.recall - b.recall:>+9.4f}"
        )
    return "\n".join(lines)
