"""Drift detection.

Four things here that most implementations get wrong, and each is a one-line fix that changes the
answer:

**PSI needs FIXED bin edges taken from the reference window.** Recomputing quantile bins on each new
window makes PSI read near zero no matter how far the distribution has moved, because the bins move
with it. This is the single most common PSI bug and it silently disables the monitor.

**42 features x a KS test per window is 42 hypotheses.** At alpha=0.05 that is roughly two false
alarms every window, forever. Benjamini-Hochberg, or at minimum report observed-versus-expected flag
counts so an operator can see that two flags is the null result.

**KS loses all meaning at scale.** With 10^5 samples per window every feature is "significantly"
drifted. The p-value stops discriminating, so the D statistic is reported as an effect size and
gated on separately.

**Concept drift is not detectable without labels.** Unlabelled monitoring detects *covariate* shift
(P(x) moving) and *prediction* shift (P(y-hat) moving). P(y|x) moving is invisible without ground
truth. In a SOC labels arrive days late or never, so the unsupervised signals are the alarm and
analyst verdicts are the confirmation. Saying so is not a caveat, it is the design.

PSI's 0.1 / 0.25 thresholds are a convention from Lewis (1994) and Siddiqi's credit-scoring work,
not a derived result, and they are sample-size dependent. Treated as a prompt to look, not a verdict.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from penumbra.eval.bootstrap import benjamini_hochberg

PSI_NO_SHIFT = 0.10
PSI_SIGNIFICANT = 0.25
_EPSILON = 1e-6


@dataclass(frozen=True)
class FeatureDrift:
    feature: str
    psi: float
    ks_statistic: float
    ks_pvalue: float
    js_distance: float
    flagged_ks: bool = False

    @property
    def psi_verdict(self) -> str:
        if self.psi < PSI_NO_SHIFT:
            return "stable"
        if self.psi < PSI_SIGNIFICANT:
            return "moderate"
        return "significant"


@dataclass
class DriftReport:
    n_reference: int
    n_current: int
    features: list[FeatureDrift] = field(default_factory=list)
    alpha: float = 0.05

    @property
    def significant(self) -> list[FeatureDrift]:
        return [f for f in self.features if f.psi >= PSI_SIGNIFICANT]

    @property
    def moderate(self) -> list[FeatureDrift]:
        return [f for f in self.features if PSI_NO_SHIFT <= f.psi < PSI_SIGNIFICANT]

    @property
    def ks_flagged(self) -> list[FeatureDrift]:
        return [f for f in self.features if f.flagged_ks]

    @property
    def expected_false_flags(self) -> float:
        """How many KS flags the null hypothesis predicts.

        Printed next to the observed count so "three features flagged" can be read against "two were
        expected anyway".
        """
        return self.alpha * len(self.features)

    def should_retrain(self) -> bool:
        return bool(self.significant)

    def summary(self, top: int = 12) -> str:
        ranked = sorted(self.features, key=lambda f: f.psi, reverse=True)[:top]
        lines = [
            f"Drift report - reference {self.n_reference:,} rows vs current {self.n_current:,}",
            "",
            f"  {'feature':<24} {'PSI':>8} {'verdict':<12} {'KS D':>8} {'KS p':>10} {'JS':>8}",
            f"  {'-' * 24} {'-' * 8} {'-' * 12} {'-' * 8} {'-' * 10} {'-' * 8}",
        ]
        for f in ranked:
            mark = " *" if f.flagged_ks else "  "
            lines.append(
                f"  {f.feature:<24} {f.psi:>8.4f} {f.psi_verdict:<12} {f.ks_statistic:>8.4f} "
                f"{f.ks_pvalue:>10.2e} {f.js_distance:>8.4f}{mark}"
            )
        lines += [
            "",
            f"  PSI: {len(self.significant)} significant (>={PSI_SIGNIFICANT}), "
            f"{len(self.moderate)} moderate (>={PSI_NO_SHIFT})",
            f"  KS after Benjamini-Hochberg: {len(self.ks_flagged)} of {len(self.features)} flagged "
            f"({self.expected_false_flags:.1f} expected under the null)",
            "",
            "  PSI thresholds are a convention (Lewis 1994; Siddiqi), not a derived result, and they",
            "  are sample-size dependent. They are a prompt to investigate, not a verdict.",
        ]
        if self.should_retrain():
            lines.append("")
            lines.append(f"  -> RETRAIN REVIEW: {', '.join(f.feature for f in self.significant[:6])}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_reference": self.n_reference,
            "n_current": self.n_current,
            "alpha": self.alpha,
            "n_significant": len(self.significant),
            "n_moderate": len(self.moderate),
            "n_ks_flagged": len(self.ks_flagged),
            "expected_false_flags": self.expected_false_flags,
            "features": [
                {
                    "feature": f.feature,
                    "psi": f.psi,
                    "verdict": f.psi_verdict,
                    "ks_statistic": f.ks_statistic,
                    "ks_pvalue": f.ks_pvalue,
                    "js_distance": f.js_distance,
                    "ks_flagged": f.flagged_ks,
                }
                for f in self.features
            ],
        }


class PSIBins:
    """Bin edges frozen from the reference window.

    Existing as a class rather than a function is the point: the edges are computed once, stored,
    and reused for every subsequent window. Recomputing them per window is the bug described at the
    top of this module.
    """

    def __init__(self, reference: np.ndarray, n_bins: int = 10) -> None:
        ref = np.asarray(reference, dtype=float)
        ref = ref[np.isfinite(ref)]
        if len(ref) == 0:
            self.edges = np.array([0.0, 1.0])
            return
        quantiles = np.linspace(0, 100, n_bins + 1)
        edges = np.unique(np.percentile(ref, quantiles))
        if len(edges) < 2:
            edges = np.array([edges[0] - 0.5, edges[0] + 0.5])
        # Open-ended tails so values outside the reference range still land in a bin rather than
        # being silently dropped - a new value beyond anything seen before is exactly the kind of
        # drift worth catching.
        self.edges = np.concatenate([[-np.inf], edges[1:-1], [np.inf]])
        self.reference_proportions = self._proportions(ref)

    def _proportions(self, values: np.ndarray) -> np.ndarray:
        counts, _ = np.histogram(values[np.isfinite(values)], bins=self.edges)
        total = counts.sum()
        if total == 0:
            return np.full(len(counts), _EPSILON)
        return np.maximum(counts / total, _EPSILON)

    def psi(self, current: np.ndarray) -> float:
        cur = np.asarray(current, dtype=float)
        actual = self._proportions(cur)
        expected = self.reference_proportions
        return float(np.sum((actual - expected) * np.log(actual / expected)))


def population_stability_index(reference: np.ndarray, current: np.ndarray, *, n_bins: int = 10) -> float:
    """One-shot PSI. Prefer `PSIBins` for repeated monitoring so the edges stay fixed."""
    return PSIBins(reference, n_bins=n_bins).psi(current)


def jensen_shannon(reference: np.ndarray, current: np.ndarray, *, n_bins: int = 20) -> float:
    """Jensen-Shannon distance, bounded in [0, 1] with log base 2.

    Preferred over KL for monitoring: it is symmetric, bounded, and defined when a bin is empty on
    one side - KL is not, and needs smoothing that changes the answer.
    """
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    ref, cur = ref[np.isfinite(ref)], cur[np.isfinite(cur)]
    if len(ref) == 0 or len(cur) == 0:
        return float("nan")

    lo = min(ref.min(), cur.min())
    hi = max(ref.max(), cur.max())
    if lo == hi:
        return 0.0
    edges = np.linspace(lo, hi, n_bins + 1)

    p, _ = np.histogram(ref, bins=edges)
    q, _ = np.histogram(cur, bins=edges)
    p = np.maximum(p / max(p.sum(), 1), _EPSILON)
    q = np.maximum(q / max(q.sum(), 1), _EPSILON)
    return _js_distance(p, q)


def _js_distance(p: np.ndarray, q: np.ndarray) -> float:
    m = 0.5 * (p + q)
    divergence = 0.5 * stats.entropy(p, m, base=2) + 0.5 * stats.entropy(q, m, base=2)
    return float(np.sqrt(max(divergence, 0.0)))


def compare(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    *,
    features: list[str] | None = None,
    alpha: float = 0.05,
    bins: dict[str, PSIBins] | None = None,
) -> DriftReport:
    """Per-feature drift between a reference window and a current window.

    Pass `bins` from `fit_bins` when monitoring repeatedly, so the PSI edges stay frozen across
    windows rather than following the data.
    """
    cols = features or [c for c in reference.columns if c in current.columns]
    numeric = [c for c in cols if pd.api.types.is_numeric_dtype(reference[c])]

    drifts: list[FeatureDrift] = []
    pvalues: list[float] = []

    for col in numeric:
        ref = reference[col].to_numpy(dtype=float)
        cur = current[col].to_numpy(dtype=float)
        ref_f, cur_f = ref[np.isfinite(ref)], cur[np.isfinite(cur)]
        if len(ref_f) < 2 or len(cur_f) < 2:
            continue

        psi_bins = (bins or {}).get(col) or PSIBins(ref_f)
        ks = stats.ks_2samp(ref_f, cur_f)

        drifts.append(
            FeatureDrift(
                feature=col,
                psi=psi_bins.psi(cur_f),
                ks_statistic=float(ks.statistic),
                ks_pvalue=float(ks.pvalue),
                js_distance=jensen_shannon(ref_f, cur_f),
            )
        )
        pvalues.append(float(ks.pvalue))

    # Multiplicity correction across the whole feature set, not per feature.
    if pvalues:
        rejected = benjamini_hochberg(np.array(pvalues), alpha=alpha)
        drifts = [
            FeatureDrift(d.feature, d.psi, d.ks_statistic, d.ks_pvalue, d.js_distance, bool(flag))
            for d, flag in zip(drifts, rejected, strict=True)
        ]

    return DriftReport(n_reference=len(reference), n_current=len(current), features=drifts, alpha=alpha)


def fit_bins(
    reference: pd.DataFrame, *, features: list[str] | None = None, n_bins: int = 10
) -> dict[str, PSIBins]:
    """Freeze PSI bin edges from a reference window, once."""
    cols = features or [c for c in reference.columns if pd.api.types.is_numeric_dtype(reference[c])]
    out: dict[str, PSIBins] = {}
    for col in cols:
        values = reference[col].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if len(values) >= 2:
            out[col] = PSIBins(values, n_bins=n_bins)
    return out


# =================================================================================================
# Label-free monitoring
# =================================================================================================


@dataclass
class PredictionDrift:
    """What can be monitored when no labels exist - which in a SOC is most of the time.

    Analyst verdicts arrive days later or never, so the alert rate and the score distribution are the
    signals available in real time. A sudden change in either is actionable before anyone knows
    whether the alerts were correct.
    """

    reference_alert_rate: float
    current_alert_rate: float
    reference_mean_score: float
    current_mean_score: float
    score_psi: float

    @property
    def alert_rate_ratio(self) -> float:
        return (
            self.current_alert_rate / self.reference_alert_rate if self.reference_alert_rate else float("inf")
        )

    @property
    def is_anomalous(self) -> bool:
        # A doubling or halving of alert volume is worth a human look regardless of cause. Usually
        # the cause is a network change rather than model failure, and the runbook says to check
        # that first.
        return self.alert_rate_ratio > 2.0 or self.alert_rate_ratio < 0.5 or self.score_psi >= PSI_SIGNIFICANT

    def summary(self) -> str:
        return "\n".join(
            [
                "Prediction-distribution drift (no labels required)",
                f"  alert rate   {self.reference_alert_rate:.4%} -> {self.current_alert_rate:.4%} "
                f"({self.alert_rate_ratio:.2f}x)",
                f"  mean score   {self.reference_mean_score:.4f} -> {self.current_mean_score:.4f}",
                f"  score PSI    {self.score_psi:.4f}",
                f"  -> {'INVESTIGATE' if self.is_anomalous else 'within normal variation'}",
                "",
                "  This detects covariate and prediction shift. True concept drift - P(y|x) moving -",
                "  is not observable without labels, and labels in a SOC arrive days late or never.",
            ]
        )


def prediction_drift(
    reference_scores: np.ndarray, current_scores: np.ndarray, *, threshold: float = 0.5
) -> PredictionDrift:
    ref = np.asarray(reference_scores, dtype=float)
    cur = np.asarray(current_scores, dtype=float)
    return PredictionDrift(
        reference_alert_rate=float(np.mean(ref >= threshold)),
        current_alert_rate=float(np.mean(cur >= threshold)),
        reference_mean_score=float(ref.mean()),
        current_mean_score=float(cur.mean()),
        score_psi=population_stability_index(ref, cur),
    )
