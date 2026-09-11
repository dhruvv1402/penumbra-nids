"""Choosing an operating point from cost, not from 0.5.

0.5 is an artifact of how classifiers are written, not a decision about how many false positives a
team can absorb. The threshold that matters is the one minimising expected cost, and that requires
saying out loud what a miss costs relative to a false alarm.

Those numbers are assumptions and this module treats them as such: every result carries the cost
ratio it was computed at, and `sensitivity_analysis` shows how far the chosen point moves when the
assumption is wrong - which is the honest way to present a number nobody can actually measure.

The cost of a false positive is easy and defensible: analyst minutes, which a SOC can price. The
cost of a false negative is a distribution over outcomes from "nothing" to "ransomware", and any
single figure is a guess. So the deliverable is the curve plus the sensitivity, not a point.

Costs scale with PREVALENCE, so this operates on the prevalence-corrected numbers from
`eval/prevalence.py` rather than on test-set counts. At 55% attack the arithmetic is meaningless.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import roc_curve

from penumbra.config import FLOWS_PER_DAY, MINUTES_PER_ALERT


@dataclass(frozen=True)
class CostModel:
    """What a mistake costs. Every field is an assumption and is reported alongside the result."""

    # Microsoft/Omdia State of the SOC 2026 puts median investigation at ~45 minutes.
    minutes_per_false_positive: float = MINUTES_PER_ALERT
    analyst_cost_per_hour: float = 50.0
    # The honest wide guess. A missed scan costs little; a missed ransomware foothold costs a great
    # deal. Sensitivity analysis exists because this number cannot be known.
    cost_per_missed_attack: float = 5_000.0
    flows_per_day: int = FLOWS_PER_DAY
    prevalence: float = 1e-4

    @property
    def cost_per_false_positive(self) -> float:
        return self.minutes_per_false_positive / 60.0 * self.analyst_cost_per_hour

    @property
    def cost_ratio(self) -> float:
        """How many false positives one missed attack is worth."""
        fp = self.cost_per_false_positive
        return self.cost_per_missed_attack / fp if fp else float("inf")

    def describe(self) -> str:
        return (
            f"  false positive: {self.minutes_per_false_positive:.0f} analyst-minutes "
            f"= ${self.cost_per_false_positive:,.2f}\n"
            f"  missed attack : ${self.cost_per_missed_attack:,.0f}\n"
            f"  ratio         : 1 miss = {self.cost_ratio:,.0f} false positives\n"
            f"  prevalence    : 1 in {1 / self.prevalence:,.0f} over "
            f"{self.flows_per_day:,} flows/day"
        )


@dataclass(frozen=True)
class CostPoint:
    threshold: float
    tpr: float
    fpr: float
    false_positives_per_day: float
    missed_attacks_per_day: float
    fp_cost_per_day: float
    fn_cost_per_day: float

    @property
    def total_cost_per_day(self) -> float:
        return self.fp_cost_per_day + self.fn_cost_per_day

    @property
    def analyst_hours_per_day(self) -> float:
        return self.false_positives_per_day * MINUTES_PER_ALERT / 60.0


@dataclass
class CostCurve:
    model: CostModel
    points: list[CostPoint] = field(default_factory=list)

    @property
    def optimal(self) -> CostPoint:
        return min(self.points, key=lambda p: p.total_cost_per_day)

    def at_threshold(self, threshold: float) -> CostPoint:
        return min(self.points, key=lambda p: abs(p.threshold - threshold))

    def summary(self) -> str:
        best = self.optimal
        default = self.at_threshold(0.5)
        saving = default.total_cost_per_day - best.total_cost_per_day

        lines = [
            "Cost-optimal operating point",
            "",
            "  Assumptions (all of them, stated):",
            self.model.describe(),
            "",
            f"  {'threshold':>10} {'TPR':>7} {'FPR':>8} {'FP/day':>10} {'missed/day':>11} "
            f"{'$/day':>12} {'analyst-h':>10}",
            f"  {'-' * 10} {'-' * 7} {'-' * 8} {'-' * 10} {'-' * 11} {'-' * 12} {'-' * 10}",
        ]
        step = max(1, len(self.points) // 12)
        for p in self.points[::step]:
            mark = "  <-- optimal" if p is best else ""
            lines.append(
                f"  {p.threshold:>10.4f} {p.tpr:>7.4f} {p.fpr:>8.5f} "
                f"{p.false_positives_per_day:>10,.0f} {p.missed_attacks_per_day:>11.1f} "
                f"{p.total_cost_per_day:>12,.0f} {p.analyst_hours_per_day:>10.1f}{mark}"
            )

        lines += [
            "",
            f"  Optimal threshold {best.threshold:.4f}: TPR {best.tpr:.4f}, FPR {best.fpr:.5f}",
            f"    {best.false_positives_per_day:,.0f} false alerts/day "
            f"({best.analyst_hours_per_day:.1f} analyst-hours), "
            f"{best.missed_attacks_per_day:.1f} attacks missed",
            f"  Default threshold 0.5: ${default.total_cost_per_day:,.0f}/day",
            f"  Choosing from the curve saves ${saving:,.0f}/day under these assumptions.",
            "",
            "  These are MODELLED costs at a stated prevalence, not measured ones. The cost of a",
            "  missed attack is a guess; see sensitivity_analysis for how much the choice depends",
            "  on it.",
        ]
        return "\n".join(lines)

    def to_dict(self) -> dict[str, object]:
        best = self.optimal
        return {
            "cost_model": {
                "minutes_per_false_positive": self.model.minutes_per_false_positive,
                "analyst_cost_per_hour": self.model.analyst_cost_per_hour,
                "cost_per_missed_attack": self.model.cost_per_missed_attack,
                "cost_ratio": self.model.cost_ratio,
                "prevalence": self.model.prevalence,
                "flows_per_day": self.model.flows_per_day,
            },
            "optimal": {
                "threshold": best.threshold,
                "tpr": best.tpr,
                "fpr": best.fpr,
                "false_positives_per_day": best.false_positives_per_day,
                "missed_attacks_per_day": best.missed_attacks_per_day,
                "total_cost_per_day": best.total_cost_per_day,
                "analyst_hours_per_day": best.analyst_hours_per_day,
            },
            "modelled_not_measured": True,
        }


def build_curve(y_true: np.ndarray, scores: np.ndarray, model: CostModel | None = None) -> CostCurve:
    """Expected daily cost across every threshold the ROC curve visits.

    TPR and FPR are measured on the test set; volumes are projected onto the stated operational
    prevalence. That split matters - the rates transfer between prevalences, the counts do not.
    """
    model = model or CostModel()
    fpr, tpr, thresholds = roc_curve(np.asarray(y_true).astype(int), np.asarray(scores, dtype=float))

    attacks_per_day = model.flows_per_day * model.prevalence
    benign_per_day = model.flows_per_day * (1.0 - model.prevalence)

    points = [
        CostPoint(
            threshold=float(t),
            tpr=float(tp),
            fpr=float(fp),
            false_positives_per_day=benign_per_day * float(fp),
            missed_attacks_per_day=attacks_per_day * (1.0 - float(tp)),
            fp_cost_per_day=benign_per_day * float(fp) * model.cost_per_false_positive,
            fn_cost_per_day=attacks_per_day * (1.0 - float(tp)) * model.cost_per_missed_attack,
        )
        for fp, tp, t in zip(fpr, tpr, thresholds, strict=True)
        if np.isfinite(t)
    ]
    return CostCurve(model=model, points=points)


def sensitivity_analysis(
    y_true: np.ndarray,
    scores: np.ndarray,
    *,
    miss_costs: tuple[float, ...] = (500, 1_000, 5_000, 25_000, 100_000),
    base: CostModel | None = None,
) -> str:
    """How much the optimal threshold moves when the guessed cost is wrong.

    The point of the exercise. If the chosen point is stable across two orders of magnitude of
    assumed breach cost, the choice is defensible despite resting on a number nobody knows. If it
    swings wildly, the honest report is that the operating point is undetermined by the evidence -
    and that is worth knowing before putting it on a slide.
    """
    base = base or CostModel()
    lines = [
        "Sensitivity of the operating point to the assumed cost of a miss",
        "",
        f"  {'miss cost':>12} {'ratio':>12} {'threshold':>11} {'TPR':>8} {'FPR':>9} {'FP/day':>10}",
        f"  {'-' * 12} {'-' * 12} {'-' * 11} {'-' * 8} {'-' * 9} {'-' * 10}",
    ]
    thresholds: list[float] = []
    for cost in miss_costs:
        model = CostModel(
            minutes_per_false_positive=base.minutes_per_false_positive,
            analyst_cost_per_hour=base.analyst_cost_per_hour,
            cost_per_missed_attack=cost,
            flows_per_day=base.flows_per_day,
            prevalence=base.prevalence,
        )
        best = build_curve(y_true, scores, model).optimal
        thresholds.append(best.threshold)
        lines.append(
            f"  ${cost:>11,.0f} {model.cost_ratio:>12,.0f} {best.threshold:>11.4f} "
            f"{best.tpr:>8.4f} {best.fpr:>9.5f} {best.false_positives_per_day:>10,.0f}"
        )

    spread = max(thresholds) - min(thresholds)
    lines += [
        "",
        f"  Threshold spans {spread:.4f} across a 200x range of assumed miss cost.",
    ]
    if spread < 0.1:
        lines.append(
            "  The operating point is stable under the assumption, so the choice is defensible "
            "despite\n  resting on a number nobody can measure."
        )
    else:
        lines.append(
            "  The operating point moves materially with the assumption. It is therefore NOT "
            "determined\n  by the evidence, and should be set by policy with the trade-off stated "
            "rather than presented\n  as an optimum."
        )
    return "\n".join(lines)
