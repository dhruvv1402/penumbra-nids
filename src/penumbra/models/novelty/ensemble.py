"""Fusing novelty detectors into one score, thresholded once.

Two rules, and both exist because the obvious approach is wrong:

**Rank, do not rescale.** Autoencoder reconstruction error and IsolationForest's decision function
are not commensurable quantities - one is a mean squared error in scaled feature units, the other a
path-length statistic. Min-max or z-scoring them onto a shared range produces a number with no
meaning, dominated by whichever detector has the fatter tail. Instead each score is converted to its
PERCENTILE against the benign reference distribution: "more unusual than 99.7% of known-benign
traffic" means the same thing for all three, and it is a sentence an analyst can act on.

**Threshold once, at the end.** Three detectors each thresholded at their own 1% false-positive
point and OR-ed together do not give 1% FPR - they give something closer to 3%, because their false
positives only partly overlap. Fuse first, threshold once, and the operating point means what it
says.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from penumbra.models.novelty.detectors import NoveltyDetector, default_detectors


@dataclass
class NoveltyEnsemble:
    """Benign-only ensemble producing a single novelty percentile per flow."""

    detectors: list[NoveltyDetector] = field(default_factory=default_detectors)
    # Sorted benign scores per detector, used to convert a raw score into a percentile.
    reference_: dict[str, np.ndarray] = field(default_factory=dict, repr=False)
    fitted_: bool = False

    def fit(self, X_benign: np.ndarray) -> NoveltyEnsemble:
        """Fit every detector on benign traffic and record its benign score distribution.

        The reference distribution is built from the SAME rows the detectors were fitted on. That
        makes the resulting percentiles slightly optimistic - a detector fits its training data
        better than it fits unseen benign traffic - and the honest consequence is that the operating
        threshold is calibrated on a held-out benign split by `calibrate_on`, not here.
        """
        for det in self.detectors:
            det.fit(X_benign)
            self.reference_[det.name] = np.sort(det.score(X_benign))
        self.fitted_ = True
        return self

    def calibrate_on(self, X_benign_holdout: np.ndarray) -> NoveltyEnsemble:
        """Rebuild the reference distributions from benign data the detectors never saw.

        Without this the percentiles are measured against training reconstruction error, which is
        systematically lower than it would be on fresh benign traffic - so every real flow looks
        more novel than it is and the false-positive rate at a given percentile is understated.
        """
        if not self.fitted_:
            raise RuntimeError("calibrate_on called before fit")
        for det in self.detectors:
            self.reference_[det.name] = np.sort(det.score(X_benign_holdout))
        return self

    def raw_scores(self, X: np.ndarray) -> dict[str, np.ndarray]:
        return {det.name: det.score(X) for det in self.detectors}

    def percentiles(self, X: np.ndarray) -> dict[str, np.ndarray]:
        """Per-detector percentile rank against the benign reference, in [0, 1]."""
        out: dict[str, np.ndarray] = {}
        for det in self.detectors:
            ref = self.reference_[det.name]
            # searchsorted gives how many benign scores this one exceeds.
            idx = np.searchsorted(ref, det.score(X), side="left")
            out[det.name] = idx / max(len(ref), 1)
        return out

    def score(self, X: np.ndarray, *, how: str = "mean") -> np.ndarray:
        """Fused novelty percentile in [0, 1]. Higher = more unusual.

        `mean` averages the per-detector percentiles: a flow has to look unusual to more than one
        detector to score highly, which suppresses each detector's idiosyncratic false positives.

        `max` fires if ANY detector finds it unusual - higher recall, considerably more noise. Kept
        because it is the right choice for the hunting lane's ranked queue, where the budget caps
        the volume anyway.
        """
        if not self.fitted_:
            raise RuntimeError("score called before fit")
        stacked = np.column_stack([self.percentiles(X)[d.name] for d in self.detectors])
        if how == "mean":
            return np.asarray(stacked.mean(axis=1), dtype=float)
        if how == "max":
            return np.asarray(stacked.max(axis=1), dtype=float)
        if how == "median":
            return np.asarray(np.median(stacked, axis=1), dtype=float)
        raise ValueError(f"unknown fusion {how!r}; expected mean, max or median")

    def threshold_for_fpr(
        self, X_benign_holdout: np.ndarray, target_fpr: float, *, how: str = "mean"
    ) -> float:
        """The fused-score threshold that yields `target_fpr` on held-out benign traffic.

        This is how an operating point gets chosen: from measured benign behaviour, not from a
        round number.
        """
        scores = self.score(X_benign_holdout, how=how)
        return float(np.quantile(scores, 1.0 - target_fpr))

    def agreement(self, X: np.ndarray, *, percentile: float = 0.99) -> np.ndarray:
        """How many detectors independently place a flow above the given percentile.

        Useful in the console: "all three detectors flagged this" is a different message to an
        analyst than "one of three did", and the fused score alone hides the distinction.
        """
        pcts = self.percentiles(X)
        return np.asarray(np.sum([pcts[d.name] >= percentile for d in self.detectors], axis=0), dtype=int)
