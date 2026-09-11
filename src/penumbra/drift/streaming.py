"""Streaming drift detection.

The windowed comparison in `detectors.py` answers "has the distribution moved between these two
windows?". This module answers "has it moved *yet*?" - one observation at a time, which is how a
monitor actually runs.

Two detector families, and the distinction between them matters operationally:

  UNSUPERVISED (ADWIN, Page-Hinkley on the score stream)
      Run continuously. Detect that the model's OUTPUT distribution has moved. Available in real
      time because they need no labels.

  SUPERVISED (DDM, EDDM on the error stream)
      Detect that the model's ERROR RATE has moved, which is the thing you actually care about. But
      they need ground truth, and in a SOC ground truth is an analyst verdict that arrives days
      later, on a biased sample of alerts that someone chose to investigate.

So the operating pattern is: unsupervised detectors are the alarm, supervised detectors are the
confirmation, and the lag between them is a property of the SOC rather than of the model.

ADWIN (Bifet & Gavalda, SDM 2007) is the strongest citation available here: it maintains two adaptive
sub-windows and signals when their means differ beyond a Hoeffding bound, with provable false-positive
and false-negative guarantees rather than a tuned threshold.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from river.drift import ADWIN, PageHinkley
from river.drift.binary import DDM, EDDM


@dataclass
class DriftAlarm:
    index: int
    detector: str
    kind: str  # "warning" | "drift"
    value: float

    def __str__(self) -> str:
        return f"[{self.index:>7,}] {self.detector:<14} {self.kind:<8} value={self.value:.4f}"


@dataclass
class StreamResult:
    detector: str
    n_seen: int
    alarms: list[DriftAlarm] = field(default_factory=list)
    warnings: list[DriftAlarm] = field(default_factory=list)

    @property
    def first_alarm(self) -> int | None:
        return self.alarms[0].index if self.alarms else None

    def detection_delay(self, true_change_point: int) -> int | None:
        """Observations between the actual change and the first alarm after it.

        The number that matters for a monitor: a detector that fires eventually is not the same as
        one that fires soon. Returns None if it never fired after the change.
        """
        after = [a.index for a in self.alarms if a.index >= true_change_point]
        return (after[0] - true_change_point) if after else None

    def false_alarms_before(self, true_change_point: int) -> int:
        return sum(1 for a in self.alarms if a.index < true_change_point)

    def summary(self, true_change_point: int | None = None) -> str:
        lines = [
            f"  {self.detector:<16} {len(self.alarms):>3} alarms, {len(self.warnings):>3} warnings "
            f"over {self.n_seen:,} observations"
        ]
        if true_change_point is not None:
            delay = self.detection_delay(true_change_point)
            false_early = self.false_alarms_before(true_change_point)
            lines.append(
                f"  {'':16} change at {true_change_point:,}: detected after {delay:,} obs"
                if delay is not None
                else f"  {'':16} change at {true_change_point:,}: NEVER DETECTED"
            )
            if false_early:
                lines.append(f"  {'':16} {false_early} alarm(s) before the change (false)")
        return "\n".join(lines)


def monitor_scores(scores: Iterable[float], *, detector: str = "adwin", **kwargs: Any) -> StreamResult:
    """Run an unsupervised detector over a stream of model scores.

    No labels required, so this is what runs in production continuously.
    """
    scores = list(scores)
    det: Any
    if detector == "adwin":
        # delta is the confidence bound; 0.002 is river's default and corresponds to a low
        # false-alarm rate over long streams.
        det = ADWIN(delta=kwargs.get("delta", 0.002))
    elif detector == "page_hinkley":
        det = PageHinkley(
            min_instances=kwargs.get("min_instances", 30),
            delta=kwargs.get("delta", 0.005),
            threshold=kwargs.get("threshold", 50.0),
        )
    else:
        raise ValueError(f"unknown unsupervised detector {detector!r}")

    result = StreamResult(detector=detector, n_seen=len(scores))
    for i, value in enumerate(scores):
        det.update(float(value))
        if det.drift_detected:
            result.alarms.append(DriftAlarm(i, detector, "drift", float(value)))
    return result


def monitor_errors(correct: Iterable[bool], *, detector: str = "ddm", **kwargs: Any) -> StreamResult:
    """Run a supervised detector over a stream of correct/incorrect outcomes.

    Requires ground truth, so in a SOC this runs on the trickle of analyst-confirmed verdicts rather
    than on live traffic - and on a sample biased toward alerts somebody chose to investigate. Both
    facts are why this is the confirmation signal, not the alarm.
    """
    outcomes = list(correct)
    det: Any
    if detector == "ddm":
        det = DDM(
            warm_start=kwargs.get("warm_start", 30),
            warning_threshold=kwargs.get("warning_threshold", 2.0),
            drift_threshold=kwargs.get("drift_threshold", 3.0),
        )
    elif detector == "eddm":
        det = EDDM()
    else:
        raise ValueError(f"unknown supervised detector {detector!r}")

    result = StreamResult(detector=detector, n_seen=len(outcomes))
    for i, ok in enumerate(outcomes):
        # river's binary detectors take 1 for an error.
        det.update(0 if ok else 1)
        if getattr(det, "warning_detected", False):
            result.warnings.append(DriftAlarm(i, detector, "warning", 0.0))
        if det.drift_detected:
            result.alarms.append(DriftAlarm(i, detector, "drift", 0.0))
    return result


def compare_detectors(
    scores: Iterable[float], *, true_change_point: int | None = None
) -> dict[str, StreamResult]:
    """Run every unsupervised detector over the same stream.

    Detection delay and false alarms before the change are the two axes that matter, and they trade
    against each other - a detector tuned to fire fast fires early on noise too.
    """
    scores = list(scores)
    return {name: monitor_scores(scores, detector=name) for name in ("adwin", "page_hinkley")}


def sliding_windows(
    values: np.ndarray, *, size: int, step: int | None = None
) -> Iterator[tuple[int, np.ndarray]]:
    """Yield (start_index, window) pairs for windowed monitoring."""
    step = step or size
    values = np.asarray(values)
    for start in range(0, max(len(values) - size + 1, 0), step):
        yield start, values[start : start + size]
