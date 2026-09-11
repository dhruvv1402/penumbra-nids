"""Conformal prediction: an abstention lane with a stated error rate.

Split conformal turns any score into a prediction *set* with a finite-sample coverage guarantee: at
alpha = 0.1 the true label falls inside the set at least 90% of the time. Where the set is ambiguous
— it contains both labels, or neither — the model is declining to commit, and that is what feeds the
`UNCERTAIN` verdict and the review lane.

That is human-in-the-loop with a number attached rather than a gesture.

## The caveat, which is also the feature

**The guarantee holds under exchangeability**, and drift violates exchangeability by definition. So
a project that claims "statistically guaranteed 95% coverage" on one slide and "we monitor concept
drift" on the next has contradicted itself, and a stats-literate reviewer will collapse both.

We state the caveat and then measure it: `coverage_under_drift` runs the conformal predictor over a
drifting stream and reports empirical coverage falling below nominal as PSI rises. That turns the
contradiction into a second, label-free drift signal — coverage loss is observable without ground
truth, which is exactly the regime a SOC operates in.

## Mondrian, not marginal

Marginal conformal guarantees coverage *on average over all classes*. On a set that is 55% attack
that number is dominated by the majority and says almost nothing about `Worms`. Class-conditional
(Mondrian) conformal calibrates a separate quantile per class, so the guarantee holds within each —
which is the only version worth having when the classes you care about are the rare ones.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

# Labels used in the binary setting. The set is over these, not over attack families.
BENIGN, ATTACK = 0, 1


@dataclass
class ConformalSets:
    """Prediction sets for a batch of rows."""

    contains_benign: np.ndarray
    contains_attack: np.ndarray
    alpha: float

    @property
    def size(self) -> np.ndarray:
        return self.contains_benign.astype(int) + self.contains_attack.astype(int)

    @property
    def is_ambiguous(self) -> np.ndarray:
        """Both labels admitted: the model will not choose."""
        return self.contains_benign & self.contains_attack

    @property
    def is_empty(self) -> np.ndarray:
        """Neither label admitted.

        An empty set is not a failure of the method - it means the row is atypical of *both*
        calibration classes, which is itself informative and is a natural companion to the novelty
        head. These route to review as well.
        """
        return ~self.contains_benign & ~self.contains_attack

    @property
    def abstains(self) -> np.ndarray:
        return self.is_ambiguous | self.is_empty

    @property
    def abstention_rate(self) -> float:
        return float(self.abstains.mean())

    def labels(self) -> np.ndarray:
        """Singleton prediction where the set has one member, -1 where it abstains."""
        out = np.full(len(self.contains_benign), -1, dtype=int)
        out[self.contains_attack & ~self.contains_benign] = ATTACK
        out[self.contains_benign & ~self.contains_attack] = BENIGN
        return out

    def as_strings(self) -> list[list[str]]:
        """Set membership rendered for an alert payload."""
        out: list[list[str]] = []
        for b, a in zip(self.contains_benign, self.contains_attack, strict=True):
            members = []
            if b:
                members.append("BENIGN")
            if a:
                members.append("ATTACK")
            out.append(members)
        return out


@dataclass
class CoverageResult:
    alpha: float
    nominal: float
    empirical: float
    n: int
    per_class: dict[int, float] = field(default_factory=dict)
    abstention_rate: float = 0.0
    mean_set_size: float = 0.0

    @property
    def gap(self) -> float:
        """Empirical minus nominal. Negative means the guarantee is not holding."""
        return self.empirical - self.nominal

    @property
    def holds(self) -> bool:
        # A small shortfall is finite-sample noise; a large one is exchangeability failing.
        return self.gap > -0.02

    def summary(self) -> str:
        lines = [
            f"  alpha {self.alpha:.2f}  nominal coverage {self.nominal:.1%}",
            f"  empirical {self.empirical:.1%} over {self.n:,} rows  (gap {self.gap:+.1%})",
            f"  abstention {self.abstention_rate:.1%}  mean set size {self.mean_set_size:.2f}",
        ]
        for cls, cov in sorted(self.per_class.items()):
            name = "attack" if cls == ATTACK else "benign"
            lines.append(f"    {name:<7} {cov:.1%}")
        lines.append(f"  -> {'guarantee holds' if self.holds else 'COVERAGE BELOW NOMINAL'}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "alpha": self.alpha,
            "nominal": self.nominal,
            "empirical": self.empirical,
            "gap": self.gap,
            "n": self.n,
            "per_class": {str(k): v for k, v in self.per_class.items()},
            "abstention_rate": self.abstention_rate,
            "mean_set_size": self.mean_set_size,
            "holds": self.holds,
        }


class MondrianConformal:
    """Class-conditional split conformal over a binary probability.

    Nonconformity is 1 - p(true class): a row the model scored confidently and correctly is
    conforming, one it scored confidently and wrongly is not.
    """

    def __init__(self, alpha: float = 0.1) -> None:
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be in (0, 1)")
        self.alpha = alpha
        self.quantiles_: dict[int, float] = {}
        self.n_calibration_: dict[int, int] = {}

    def fit(self, p_attack_cal: np.ndarray, y_cal: np.ndarray) -> MondrianConformal:
        """Calibrate on held-out data the model did not train on.

        Calibrating on training rows would fit the quantile to memorised predictions and the
        guarantee would be vacuous.
        """
        p = np.asarray(p_attack_cal, dtype=float)
        y = np.asarray(y_cal).astype(int)

        for cls in (BENIGN, ATTACK):
            mask = y == cls
            n = int(mask.sum())
            if n == 0:
                self.quantiles_[cls] = 1.0
                self.n_calibration_[cls] = 0
                continue

            # Nonconformity of the true class for rows of this class.
            scores = 1.0 - (p[mask] if cls == ATTACK else 1.0 - p[mask])

            # The finite-sample correction. Using the plain (1-alpha) quantile undershoots coverage
            # on small calibration sets, which is exactly where the rare classes live.
            level = min(1.0, np.ceil((n + 1) * (1.0 - self.alpha)) / n)
            self.quantiles_[cls] = float(np.quantile(scores, level, method="higher"))
            self.n_calibration_[cls] = n
        return self

    def predict_sets(self, p_attack: np.ndarray) -> ConformalSets:
        """Admit each label whose nonconformity falls below that class's calibrated quantile."""
        if not self.quantiles_:
            raise RuntimeError("conformal predictor is not calibrated")
        p = np.asarray(p_attack, dtype=float)
        return ConformalSets(
            contains_benign=(1.0 - (1.0 - p)) <= self.quantiles_[BENIGN],
            contains_attack=(1.0 - p) <= self.quantiles_[ATTACK],
            alpha=self.alpha,
        )

    def evaluate(self, p_attack: np.ndarray, y_true: np.ndarray) -> CoverageResult:
        """Empirical coverage: how often the true label is actually in the set."""
        sets = self.predict_sets(p_attack)
        y = np.asarray(y_true).astype(int)

        covered = np.where(y == ATTACK, sets.contains_attack, sets.contains_benign)
        per_class = {cls: float(covered[y == cls].mean()) for cls in (BENIGN, ATTACK) if (y == cls).any()}

        return CoverageResult(
            alpha=self.alpha,
            nominal=1.0 - self.alpha,
            empirical=float(covered.mean()),
            n=len(y),
            per_class=per_class,
            abstention_rate=sets.abstention_rate,
            mean_set_size=float(sets.size.mean()),
        )

    def summary(self) -> str:
        return "\n".join(
            [
                f"  Mondrian split conformal, alpha={self.alpha:.2f}",
                *(
                    f"    {'attack' if c == ATTACK else 'benign':<7} quantile {q:.4f} "
                    f"({self.n_calibration_[c]:,} calibration rows)"
                    for c, q in sorted(self.quantiles_.items())
                ),
            ]
        )


# =================================================================================================
# Coverage as a drift signal
# =================================================================================================


@dataclass
class DriftCoveragePoint:
    window: int
    psi: float
    empirical_coverage: float
    nominal: float
    abstention_rate: float

    @property
    def gap(self) -> float:
        return self.empirical_coverage - self.nominal


@dataclass
class DriftCoverageResult:
    points: list[DriftCoveragePoint] = field(default_factory=list)
    change_point_window: int | None = None

    def first_breach(self, tolerance: float = 0.02) -> int | None:
        """First window where coverage falls meaningfully below nominal."""
        for p in self.points:
            if p.gap < -tolerance:
                return p.window
        return None

    def summary(self) -> str:
        lines = [
            "Conformal coverage under injected drift",
            "",
            "  Coverage guarantees hold under exchangeability. Drift violates exchangeability, so",
            "  coverage falling below nominal is not the method failing - it is the method",
            "  reporting that its own assumption no longer holds. That is observable WITHOUT",
            "  labels, which is the regime a SOC actually operates in.",
            "",
            f"  {'window':>7} {'PSI':>8} {'coverage':>10} {'nominal':>9} {'gap':>8} {'abstain':>9}",
            f"  {'-' * 7} {'-' * 8} {'-' * 10} {'-' * 9} {'-' * 8} {'-' * 9}",
        ]
        for p in self.points:
            mark = "  <-- drift injected" if p.window == self.change_point_window else ""
            lines.append(
                f"  {p.window:>7} {p.psi:>8.4f} {p.empirical_coverage:>10.1%} {p.nominal:>9.1%} "
                f"{p.gap:>+8.1%} {p.abstention_rate:>9.1%}{mark}"
            )
        breach = self.first_breach()
        lines.append("")
        if breach is not None:
            lines.append(f"  Coverage first breached nominal at window {breach}.")
            if self.change_point_window is not None:
                lines.append(
                    f"  Drift was injected at window {self.change_point_window} "
                    f"(detected {breach - self.change_point_window} windows later)."
                )
        else:
            lines.append("  Coverage held throughout - conformal did not detect this drift.")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "change_point_window": self.change_point_window,
            "first_breach_window": self.first_breach(),
            "points": [
                {
                    "window": p.window,
                    "psi": p.psi,
                    "coverage": p.empirical_coverage,
                    "nominal": p.nominal,
                    "gap": p.gap,
                    "abstention_rate": p.abstention_rate,
                }
                for p in self.points
            ],
        }


def coverage_under_drift(
    predictor: MondrianConformal,
    p_attack_stream: np.ndarray,
    y_stream: np.ndarray,
    reference_scores: np.ndarray,
    *,
    window: int = 2000,
    change_point: int | None = None,
) -> DriftCoverageResult:
    """Track empirical coverage and score PSI across a stream.

    Coverage needs labels to measure, so this is the offline validation of the idea. The operational
    version watches the *abstention rate*, which needs no labels and moves with coverage — an
    abstention rate that climbs is the same signal read from the side you can actually see.
    """
    from penumbra.drift.detectors import PSIBins

    p = np.asarray(p_attack_stream, dtype=float)
    y = np.asarray(y_stream).astype(int)
    bins = PSIBins(np.asarray(reference_scores, dtype=float))

    result = DriftCoverageResult(
        change_point_window=(change_point // window) if change_point is not None else None
    )
    for i, start in enumerate(range(0, len(p) - window + 1, window)):
        chunk_p, chunk_y = p[start : start + window], y[start : start + window]
        cov = predictor.evaluate(chunk_p, chunk_y)
        result.points.append(
            DriftCoveragePoint(
                window=i,
                psi=bins.psi(chunk_p),
                empirical_coverage=cov.empirical,
                nominal=cov.nominal,
                abstention_rate=cov.abstention_rate,
            )
        )
    return result
