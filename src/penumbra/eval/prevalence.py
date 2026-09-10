"""Translating measured detection rates into operational alert volume.

UNSW-NB15's test set is ~55% attack. A real enterprise segment runs somewhere between 1-in-10^3 and
1-in-10^5 flows. Precision, PPV, PR-AUC and "alerts per 1,000 flows" all move with base rate, so
every one of those numbers measured on this data is optimistic by two to four orders of magnitude.

TPR and FPR do not move with base rate. So the honest procedure is: measure TPR and FPR, then
re-weight to a stated deployment prevalence and derive everything else from there.

Everything this module returns is MODELLED, not measured, and says so.

Axelsson, "The Base-Rate Fallacy and the Difficulty of Intrusion Detection", ACM TISSEC 2000.
"""

from __future__ import annotations

from dataclasses import dataclass

from penumbra.config import FLOWS_PER_DAY, MINUTES_PER_ALERT, OPERATIONAL_PREVALENCE


@dataclass(frozen=True)
class OperationalEstimate:
    """What a measured (TPR, FPR) pair implies at a given deployment prevalence."""

    prevalence: float
    flows_per_day: int
    tpr: float
    fpr: float

    true_positives_per_day: float
    false_positives_per_day: float
    alerts_per_day: float
    ppv: float  # precision at THIS prevalence - the number that actually matters operationally
    missed_attacks_per_day: float
    analyst_hours_per_day: float

    measured: bool = False  # always False: these are projections, never observations

    def summary(self) -> str:
        return (
            f"  prevalence 1 in {1 / self.prevalence:,.0f}  "
            f"({self.true_positives_per_day:,.0f} true / {self.false_positives_per_day:,.0f} false "
            f"per day)  PPV {self.ppv:.4f}  {self.analyst_hours_per_day:,.1f} analyst-hours/day"
        )


def reweight(
    tpr: float,
    fpr: float,
    prevalence: float,
    *,
    flows_per_day: int = FLOWS_PER_DAY,
    minutes_per_alert: float = MINUTES_PER_ALERT,
) -> OperationalEstimate:
    """Project a measured TPR/FPR onto a deployment with the given base rate."""
    attacks = flows_per_day * prevalence
    benign = flows_per_day * (1.0 - prevalence)

    tp = attacks * tpr
    fp = benign * fpr
    alerts = tp + fp

    return OperationalEstimate(
        prevalence=prevalence,
        flows_per_day=flows_per_day,
        tpr=tpr,
        fpr=fpr,
        true_positives_per_day=tp,
        false_positives_per_day=fp,
        alerts_per_day=alerts,
        ppv=(tp / alerts) if alerts else float("nan"),
        missed_attacks_per_day=attacks - tp,
        analyst_hours_per_day=alerts * minutes_per_alert / 60.0,
    )


def sweep(
    tpr: float,
    fpr: float,
    *,
    prevalences: tuple[float, ...] = OPERATIONAL_PREVALENCE,
    flows_per_day: int = FLOWS_PER_DAY,
    minutes_per_alert: float = MINUTES_PER_ALERT,
) -> list[OperationalEstimate]:
    return [
        reweight(tpr, fpr, p, flows_per_day=flows_per_day, minutes_per_alert=minutes_per_alert)
        for p in prevalences
    ]


def base_rate_narrative(
    tpr: float,
    fpr: float,
    *,
    prevalence: float = 1e-4,
    flows_per_day: int = FLOWS_PER_DAY,
) -> str:
    """The base-rate fallacy, spelled out in this model's own numbers.

    This is the slide. Most teams will present a precision figure from a 55%-attack test set; this
    paragraph is what that figure becomes on a real network.
    """
    est = reweight(tpr, fpr, prevalence, flows_per_day=flows_per_day)
    return "\n".join(
        [
            f"At {flows_per_day:,} flows/day and a prevalence of 1 in {1 / prevalence:,.0f}, this "
            f"segment carries {est.flows_per_day * prevalence:,.0f} genuine attack flows a day.",
            f"Measured TPR {tpr:.3f} and FPR {fpr:.4f} therefore produce:",
            f"  {est.true_positives_per_day:,.0f} true alerts",
            f"  {est.false_positives_per_day:,.0f} false alerts",
            f"  -> positive predictive value {est.ppv:.2%}",
            f"  -> {est.analyst_hours_per_day:,.0f} analyst-hours/day to triage",
            "",
            "That is the base-rate fallacy (Axelsson, 2000), and it is why a detector cannot be",
            "judged by precision measured on a balanced test set - and why the novelty lane is",
            "capped at a fixed daily budget instead of generating tickets.",
        ]
    )
