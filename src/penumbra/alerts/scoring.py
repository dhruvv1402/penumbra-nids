"""Turning model outputs into Alerts, and routing them to the right lane.

This is where ADR-0002's two-lane design becomes code:

  KNOWN_THREAT lane   supervised head is confident -> incident queue, SLA'd, ticket-generating
  HUNTING lane        novelty head fired where the supervised head saw nothing -> ranked queue with
                      a FIXED DAILY BUDGET, metric is Precision@k
  REVIEW lane         conformal set was ambiguous -> human decides

The budget on the hunting lane is not a nicety. The autoencoder runs at 1.5-4% FPR on held-out
benign traffic, which at realistic prevalence cannot produce tickets - a 1% FPR against 1e-4
prevalence gives a ~1% positive predictive value. A queue capped at N items per day cannot cause
alert fatigue no matter what its FPR is, because the cap is the contract. That is the difference
between designing around a false-positive rate and hiding one.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from penumbra.alerts.models import (
    Alert,
    Contribution,
    Lane,
    NetworkContext,
    Severity,
    Verdict,
)
from penumbra.explain import attack_map


@dataclass(frozen=True)
class ScoringPolicy:
    """Thresholds and budgets. Every one of these should come from measurement, not a round number.

    `known_threat_threshold` is chosen from the cost curve (eval/cost.py) or from a target FPR on
    held-out benign traffic - never left at 0.5, which is an artifact of how classifiers are written
    rather than a decision about how many false positives a team can absorb.
    """

    known_threat_threshold: float = 0.80
    novelty_percentile_threshold: float = 0.99
    # Items a hunting analyst can actually work in a shift. The queue is truncated to this, and the
    # truncation is the feature.
    hunting_daily_budget: int = 50
    # Below this the supervised head is not committing to anything.
    uncertain_band: tuple[float, float] = (0.40, 0.80)
    min_detector_agreement: int = 1

    def severity_for(self, p_attack: float, novelty: float) -> Severity:
        if p_attack >= 0.90 or novelty >= 0.999:
            return Severity.HIGH
        if p_attack >= self.known_threat_threshold or novelty >= self.novelty_percentile_threshold:
            return Severity.MEDIUM
        if p_attack >= self.uncertain_band[0]:
            return Severity.LOW
        return Severity.INFORMATIONAL


def priority_score(p_attack: float, novelty_percentile: float, *, agreement: int = 0) -> int:
    """The 0-100 triage sort key.

    **This is an ordering, not a probability**, and it is deliberately not calibrated. Blending a
    calibrated probability with an anomaly percentile and calling the result calibrated is a
    category error (ADR-0002), so the blend lives here under a name that cannot be mistaken for one.

    Weighted toward whichever head is more confident, with a small bonus when several independent
    novelty detectors agree - an analyst treats "all three flagged it" differently from "one did",
    and the fused score alone hides that.
    """
    base = max(p_attack, novelty_percentile)
    corroboration = 0.04 * min(agreement, 3)
    return int(round(min(1.0, base + corroboration) * 100))


def classify(
    p_attack: float,
    novelty_percentile: float,
    *,
    policy: ScoringPolicy,
    agreement: int = 0,
    conformal_ambiguous: bool = False,
    suppressed: bool = False,
) -> tuple[Verdict, Lane]:
    """Decide the verdict and which queue it belongs in."""
    if suppressed:
        return Verdict.BENIGN_BY_POLICY, Lane.KNOWN_THREAT

    if p_attack >= policy.known_threat_threshold:
        return Verdict.KNOWN_ATTACK, Lane.KNOWN_THREAT

    # The product claim: unusual traffic the classifier did NOT recognise. If the supervised head
    # had recognised it, it would be a known attack and belong in the other lane.
    if (
        novelty_percentile >= policy.novelty_percentile_threshold
        and agreement >= policy.min_detector_agreement
    ):
        return Verdict.SUSPECTED_NOVEL, Lane.HUNTING

    if conformal_ambiguous or policy.uncertain_band[0] <= p_attack < policy.uncertain_band[1]:
        return Verdict.UNCERTAIN, Lane.REVIEW

    return Verdict.BENIGN, Lane.KNOWN_THREAT


def suggested_action(verdict: Verdict, family: str | None) -> str:
    """Text. Always text.

    Nothing in this package consumes these strings as instructions, and `requires_analyst_approval`
    is unconditionally True on every Alert. ADR-0001.
    """
    if verdict is Verdict.SUSPECTED_NOVEL:
        return (
            "Investigate the source host. This traffic does not match a known attack family, so "
            "confirm whether it is a new legitimate service before treating it as hostile."
        )
    if verdict is Verdict.KNOWN_ATTACK:
        base = "Confirm the source is not an authorised tool, then follow the runbook for "
        return base + (f"{family}." if family else "this family.")
    if verdict is Verdict.UNCERTAIN:
        return "Model declined to commit. Manual review required."
    return ""


def build_alert(
    *,
    p_attack: float,
    novelty_percentile: float,
    policy: ScoringPolicy,
    family: str | None = None,
    family_confidence: float | None = None,
    dataset: str = "unsw",
    fine_label: str | None = None,
    agreement: int = 0,
    conformal_ambiguous: bool = False,
    conformal_set: Sequence[str] = (),
    suppressed: bool = False,
    suppression_rule_id: str | None = None,
    network: NetworkContext | None = None,
    contributions: Sequence[Contribution] = (),
    model_name: str = "penumbra",
    model_version: str = "0.1.0",
) -> Alert:
    """Assemble one Alert from model outputs."""
    verdict, lane = classify(
        p_attack,
        novelty_percentile,
        policy=policy,
        agreement=agreement,
        conformal_ambiguous=conformal_ambiguous,
        suppressed=suppressed,
    )

    # A novelty finding has no family by construction: if the model could name it, it would not be
    # novel. Passing one through here would contradict the verdict.
    effective_family = None if verdict is Verdict.SUSPECTED_NOVEL else family
    technique = (
        attack_map.lookup(effective_family, dataset=dataset, fine_label=fine_label)
        if effective_family
        else None
    )

    return Alert(
        verdict=verdict,
        lane=lane,
        severity=policy.severity_for(p_attack, novelty_percentile),
        p_attack=float(np.clip(p_attack, 0.0, 1.0)),
        novelty_percentile=float(np.clip(novelty_percentile, 0.0, 1.0)),
        priority=priority_score(p_attack, novelty_percentile, agreement=agreement),
        family=effective_family,
        family_confidence=family_confidence,
        attack=technique,
        network=network or NetworkContext(),
        contributions=list(contributions),
        conformal_set=list(conformal_set),
        detector_agreement=agreement,
        suggested_action=suggested_action(verdict, effective_family),
        suppression_rule_id=suppression_rule_id,
        model_name=model_name,
        model_version=model_version,
    )


def route(alerts: Sequence[Alert], policy: ScoringPolicy) -> dict[Lane, list[Alert]]:
    """Split alerts into lanes and apply the hunting budget.

    The hunting lane is sorted by priority and TRUNCATED. Items below the cut are not lost - they
    are recorded and counted, and `hunting_overflow` reports how many there were - but they do not
    enter a queue, because a queue nobody can finish is the thing this design exists to avoid.
    """
    lanes: dict[Lane, list[Alert]] = {Lane.KNOWN_THREAT: [], Lane.HUNTING: [], Lane.REVIEW: []}
    for alert in alerts:
        if alert.is_actionable_alert:
            lanes[alert.lane].append(alert)

    lanes[Lane.KNOWN_THREAT].sort(key=lambda a: a.priority, reverse=True)
    lanes[Lane.REVIEW].sort(key=lambda a: a.priority, reverse=True)
    lanes[Lane.HUNTING].sort(key=lambda a: (a.novelty_percentile, a.priority), reverse=True)
    lanes[Lane.HUNTING] = lanes[Lane.HUNTING][: policy.hunting_daily_budget]
    return lanes


def hunting_overflow(alerts: Sequence[Alert], policy: ScoringPolicy) -> int:
    """Novelty findings that did not fit inside today's budget.

    Reported rather than hidden. A persistently large overflow means the novelty threshold or the
    benign reference window needs attention, and an operator should be able to see that.
    """
    n = sum(1 for a in alerts if a.lane is Lane.HUNTING and a.is_actionable_alert)
    return max(0, n - policy.hunting_daily_budget)
