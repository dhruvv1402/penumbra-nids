"""Probability calibration.

`p_attack` is the only calibrated number Penumbra emits, and calibration is what makes it mean
something: among alerts scored 0.80, roughly 80% should actually be attacks. Without that, a "risk
score" is decoration - an analyst cannot reason about it, a cost curve cannot be built on it, and
Sentinel's `ThreatConfidence` field is being fed a number that does not match its own definition.

Three things here that are easy to get wrong:

**Calibrate on held-out data.** Fitting the calibrator on the same rows the model trained on gives a
calibrator fitted to memorised predictions, which reports excellent calibration and delivers none.

**Resampling decalibrates.** SMOTE and friends alter the training prior, so the model's outputs no
longer correspond to observed frequencies. Any imbalance strategy used alongside a calibrated
probability needs recalibration afterwards, and that ordering is not optional.

**Report the reliability diagram, not just the Brier score.** Brier conflates calibration with
discrimination: a model can improve its Brier score by becoming more accurate while getting *worse*
at calibration. The decomposition and the curve are what actually answer the question.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss

from penumbra.seeds import SEED


@dataclass
class ReliabilityBin:
    lower: float
    upper: float
    n: int
    mean_predicted: float
    observed_frequency: float

    @property
    def gap(self) -> float:
        """Signed calibration error. Positive means over-confident."""
        return self.mean_predicted - self.observed_frequency


@dataclass
class CalibrationReport:
    method: str
    brier: float
    expected_calibration_error: float
    max_calibration_error: float
    bins: list[ReliabilityBin] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"  Calibration ({self.method})",
            f"    Brier score  {self.brier:.5f}   (lower is better; 0.25 is a coin flip)",
            f"    ECE          {self.expected_calibration_error:.5f}   (mean |predicted - observed|)",
            f"    MCE          {self.max_calibration_error:.5f}   (worst bin)",
            "",
            f"    {'bin':<14} {'n':>8} {'predicted':>11} {'observed':>10} {'gap':>9}",
            f"    {'-' * 14} {'-' * 8} {'-' * 11} {'-' * 10} {'-' * 9}",
        ]
        for b in self.bins:
            if b.n == 0:
                continue
            lines.append(
                f"    [{b.lower:.2f},{b.upper:.2f}) {b.n:>8,} {b.mean_predicted:>11.4f} "
                f"{b.observed_frequency:>10.4f} {b.gap:>+9.4f}"
            )
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "brier": self.brier,
            "ece": self.expected_calibration_error,
            "mce": self.max_calibration_error,
            "bins": [
                {
                    "lower": b.lower,
                    "upper": b.upper,
                    "n": b.n,
                    "mean_predicted": b.mean_predicted,
                    "observed_frequency": b.observed_frequency,
                    "gap": b.gap,
                }
                for b in self.bins
            ],
        }


def reliability(
    y_true: np.ndarray, probabilities: np.ndarray, *, n_bins: int = 10, method: str = "uncalibrated"
) -> CalibrationReport:
    """Bin predictions and compare predicted probability to observed frequency.

    Equal-width bins, which is the convention. Note that they leave the high-confidence bins sparsely
    populated on a well-separated problem - the `n` column is there so a large gap computed from
    eleven samples is visibly not the same claim as one computed from eleven thousand.
    """
    y_true = np.asarray(y_true).astype(int)
    p = np.asarray(probabilities, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)

    bins: list[ReliabilityBin] = []
    ece = 0.0
    mce = 0.0
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        mask = (p >= lo) & (p < hi) if hi < 1.0 else (p >= lo) & (p <= hi)
        n = int(mask.sum())
        if n == 0:
            bins.append(ReliabilityBin(float(lo), float(hi), 0, float("nan"), float("nan")))
            continue
        mean_pred = float(p[mask].mean())
        observed = float(y_true[mask].mean())
        gap = abs(mean_pred - observed)
        ece += (n / len(p)) * gap
        mce = max(mce, gap)
        bins.append(ReliabilityBin(float(lo), float(hi), n, mean_pred, observed))

    return CalibrationReport(
        method=method,
        brier=float(brier_score_loss(y_true, p)),
        expected_calibration_error=float(ece),
        max_calibration_error=float(mce),
        bins=bins,
    )


def calibrate(estimator: Any, X_cal: Any, y_cal: np.ndarray, *, method: str = "isotonic") -> Any:
    """Wrap a fitted estimator in a calibrator fitted on held-out data.

    `cv="prefit"` is the important argument: it tells sklearn the estimator is already trained and
    that `X_cal`/`y_cal` are calibration data it has not seen.

    isotonic is non-parametric and generally better with enough calibration data; sigmoid (Platt) is
    a two-parameter fit that is more stable when calibration data is scarce. Both are reported by
    `compare_methods` rather than one being assumed.
    """
    if method not in {"isotonic", "sigmoid"}:
        raise ValueError(f"unknown calibration method {method!r}")
    calibrated = CalibratedClassifierCV(estimator, method=method, cv="prefit")
    calibrated.fit(X_cal, y_cal)
    return calibrated


def compare_methods(
    estimator: Any,
    X_cal: Any,
    y_cal: np.ndarray,
    X_test: Any,
    y_test: np.ndarray,
    *,
    n_bins: int = 10,
) -> dict[str, CalibrationReport]:
    """Uncalibrated vs isotonic vs Platt, all scored on the test set."""
    reports: dict[str, CalibrationReport] = {}

    raw = estimator.predict_proba(X_test)[:, 1]
    reports["uncalibrated"] = reliability(y_test, raw, n_bins=n_bins, method="uncalibrated")

    for method in ("isotonic", "sigmoid"):
        model = calibrate(estimator, X_cal, y_cal, method=method)
        p = model.predict_proba(X_test)[:, 1]
        reports[method] = reliability(y_test, p, n_bins=n_bins, method=method)

    return reports


def brier_decomposition(
    y_true: np.ndarray, probabilities: np.ndarray, *, n_bins: int = 10
) -> dict[str, float]:
    """Murphy's decomposition: Brier = reliability - resolution + uncertainty.

    This is why the Brier score alone is not enough. `reliability` is the calibration error and
    lower is better; `resolution` is how far predictions move away from the base rate and HIGHER is
    better; `uncertainty` is a property of the data that no model can change.

    A model can lower its Brier score purely by increasing resolution while becoming worse
    calibrated. Reporting the parts keeps the two claims separate.
    """
    y_true = np.asarray(y_true).astype(int)
    p = np.asarray(probabilities, dtype=float)
    base_rate = float(y_true.mean())
    edges = np.linspace(0.0, 1.0, n_bins + 1)

    reliability_term = 0.0
    resolution_term = 0.0
    n_total = len(p)

    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        mask = (p >= lo) & (p < hi) if hi < 1.0 else (p >= lo) & (p <= hi)
        n = int(mask.sum())
        if n == 0:
            continue
        mean_pred = float(p[mask].mean())
        observed = float(y_true[mask].mean())
        reliability_term += (n / n_total) * (mean_pred - observed) ** 2
        resolution_term += (n / n_total) * (observed - base_rate) ** 2

    uncertainty = base_rate * (1.0 - base_rate)
    return {
        "reliability": reliability_term,
        "resolution": resolution_term,
        "uncertainty": uncertainty,
        "brier_reconstructed": reliability_term - resolution_term + uncertainty,
        "base_rate": base_rate,
    }


def split_for_calibration(
    X: Any, y: np.ndarray, *, fraction: float = 0.2, seed: int = SEED
) -> tuple[Any, Any, np.ndarray, np.ndarray]:
    """Carve a calibration set out of training data, stratified.

    Stratification matters at these class ratios: an unstratified 20% slice of a set containing 130
    Worms rows can easily contain none of them.
    """
    from sklearn.model_selection import train_test_split

    X_fit, X_cal, y_fit, y_cal = train_test_split(X, y, test_size=fraction, stratify=y, random_state=seed)
    return X_fit, X_cal, y_fit, y_cal
