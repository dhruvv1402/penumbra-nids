"""Which alerts to ask an analyst about next.

Uncertainty sampling. An analyst's hour is the scarcest resource in a SOC, and spending it confirming
detections the model is already sure of teaches the model nothing. The label worth having is the one
the model could not produce itself.

Order:
  1. conformal abstentions - the model formally declined to commit (the review lane)
  2. then by margin |p_attack - 0.5|, smallest first
  3. ties broken by novelty, because an unusual flow the model is unsure about is doubly informative

Already-judged alerts are excluded. The queue is capped: active learning that hands a human 4,000
items is a backlog, not a strategy.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from penumbra.alerts.models import Alert


def uncertainty(alert: Alert) -> float:
    """0 is maximally uncertain, 0.5 is certain."""
    return abs(alert.p_attack - 0.5)


def queue(
    alerts: Iterable[Alert], *, judged: set[str] | None = None, limit: int = 25
) -> list[dict[str, Any]]:
    judged = judged or set()
    seen: set[str] = set()
    candidates = []
    for a in alerts:
        # Callers fetch the review lane on its own AND the top of all lanes, so the same alert can
        # arrive twice; it must be offered once.
        if a.alert_id in judged or a.alert_id in seen or a.verdict.value == "BENIGN_BY_POLICY":
            continue
        seen.add(a.alert_id)
        candidates.append(a)
    candidates.sort(
        key=lambda a: (
            0 if a.lane.value == "review" else 1,
            uncertainty(a),
            -a.novelty_percentile,
        )
    )
    return [
        {
            "alert_id": a.alert_id,
            "lane": a.lane.value,
            "verdict": a.verdict.value,
            "p_attack": a.p_attack,
            "novelty_percentile": a.novelty_percentile,
            "margin": round(uncertainty(a), 4),
            "why": (
                "the model abstained: its conformal prediction set was ambiguous"
                if a.lane.value == "review"
                else f"p_attack {a.p_attack:.2f} is {uncertainty(a):.2f} from the decision boundary"
            ),
        }
        for a in candidates[:limit]
    ]
