"""Fusing the supervised and novelty heads.

## The bug this module exists to fix

The first implementation fused with `max(p_attack, novelty_percentile)` and produced a strongly
NEGATIVE LOAFO result: mean recall delta -0.36, eight of nine families worse. Adding a detector at
matched false-positive rate should not be able to do that, so the result was a bug rather than a
finding. Measured on UNSW-NB15 benign test rows:

    p_attack    median 0.0044   p99 0.9072   <- bimodal, mass near zero
    novelty     median 0.8237   p99 0.9960   <- percentile rank, ~uniform by construction
    max(p, n)   median 0.8268   p99 0.9961   <- entirely dominated by novelty

Matching on FPR sets the threshold at the 99th benign percentile. For the supervised head alone
that is 0.907; for the max-fusion it is 0.996. **The fused configuration was therefore discarding
every supervised detection scoring between 0.907 and 0.996** - which is most of them.

The cause is a scale error. `NoveltyEnsemble` rank-normalises its three detectors against each
other, precisely because reconstruction error and `decision_function` are not commensurable. That
same reasoning applies across the supervised/novelty boundary and was not applied there: a
calibrated probability and a percentile rank are different kinds of number, and `max` over them
silently means "whichever is measured on the larger scale".

## The fix

Rank-normalise **both** heads against the same benign reference before combining. Then `max` is
comparing like with like: "this flow is above the 99.5th benign percentile according to at least
one head".

## What OR-ing actually costs, made visible

Two independent detectors each thresholded at 1% FPR do not OR to 1% - they OR to nearly 2%. So at
a matched 1% budget, each head can only run at roughly 0.5%. That is a real cost of combining
detectors and it is exactly what the matched-budget comparison is for: the fused configuration has
to buy its extra coverage out of the same fixed allowance.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def rank_normalise(scores: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Convert raw scores to percentile ranks against a benign reference distribution.

    The output is "fraction of benign traffic this flow scores above", in [0, 1], which means the
    same thing regardless of whether the input was a probability, a reconstruction error or a path
    length. That is what makes two heads combinable at all.
    """
    ref = np.sort(np.asarray(reference, dtype=float))
    idx = np.searchsorted(ref, np.asarray(scores, dtype=float), side="left")
    return np.asarray(idx / max(len(ref), 1), dtype=float)


@dataclass(frozen=True)
class FusionPolicy:
    """How the two heads combine.

    `mode="max"` is an OR: either head can raise the flow. This is the right default for a detector
    whose purpose is coverage of things the other head cannot see.

    `mode="weighted"` favours the supervised head, which is better calibrated on families it knows,
    and uses novelty as a tiebreak. Better precision, worse coverage of genuinely novel traffic.
    """

    mode: str = "max"
    supervised_weight: float = 0.7

    def combine(self, rank_supervised: np.ndarray, rank_novelty: np.ndarray) -> np.ndarray:
        if self.mode == "max":
            return np.maximum(rank_supervised, rank_novelty)
        if self.mode == "weighted":
            w = self.supervised_weight
            return w * rank_supervised + (1.0 - w) * rank_novelty
        if self.mode == "supervised_only":
            return rank_supervised
        if self.mode == "novelty_only":
            return rank_novelty
        raise ValueError(f"unknown fusion mode {self.mode!r}")


@dataclass
class FusedScores:
    """Both heads, on a common scale, plus the combination."""

    rank_supervised: np.ndarray
    rank_novelty: np.ndarray
    fused: np.ndarray
    p_attack_raw: np.ndarray
    novelty_raw: np.ndarray

    def __len__(self) -> int:
        return len(self.fused)


def fuse(
    p_attack: np.ndarray,
    novelty: np.ndarray,
    *,
    benign_p_attack: np.ndarray,
    benign_novelty: np.ndarray,
    policy: FusionPolicy | None = None,
) -> FusedScores:
    """Put both heads on a benign-percentile scale and combine them.

    `benign_p_attack` and `benign_novelty` are the heads' scores on held-out BENIGN traffic. They
    define what "unusual" means for each head, and they must come from data neither head trained on
    - otherwise the reference is optimistic and every real flow looks more unusual than it is.
    """
    policy = policy or FusionPolicy()
    rs = rank_normalise(p_attack, benign_p_attack)
    rn = rank_normalise(novelty, benign_novelty)
    return FusedScores(
        rank_supervised=rs,
        rank_novelty=rn,
        fused=policy.combine(rs, rn),
        p_attack_raw=np.asarray(p_attack, dtype=float),
        novelty_raw=np.asarray(novelty, dtype=float),
    )


@dataclass
class OrGate:
    """Two heads, each thresholded on its own raw scale, OR-ed together.

    ## Why not rank-normalise and threshold once

    Rank normalisation against a finite benign reference **saturates**. Measured on NSL-KDD: 2.41%
    of benign test rows score above every one of the 13,468 reference rows, so their rank is exactly
    1.0. The 99th percentile of that distribution is therefore also 1.0, and a `>= 1.0` threshold
    admits the whole tied block identically for every configuration - which made
    supervised-vs-fused come out exactly equal, digit for digit, on every attack type.

    Thresholding each head on its own raw score avoids ties entirely, and it is also what a SOC
    actually deploys: two detectors, each tuned to its own budget.

    ## Thresholds come from held-out benign data, not from the test set

    Deriving an operating point from test-set labels is test-set peeking, even when both
    configurations get the same advantage - it cannot be reproduced at deployment time, where no
    labels exist. So thresholds are fitted on a benign reference split and then applied blind. The
    realised test FPR is reported rather than assumed, which is the honest version of "we targeted
    1%".

    ## The cost of OR-ing is charged, not hidden

    Two independent detectors at 1% each OR to nearly 2%. At a matched 1% total, each head therefore
    runs at about 0.5%. The fused configuration buys its extra coverage out of the same allowance -
    it does not get it for free.
    """

    supervised_threshold: float
    novelty_threshold: float
    use_novelty: bool = True

    @classmethod
    def fit(
        cls,
        benign_p_attack: np.ndarray,
        benign_novelty: np.ndarray,
        *,
        total_fpr: float = 0.01,
        use_novelty: bool = True,
    ) -> OrGate:
        """Choose per-head thresholds from held-out benign traffic."""
        if not use_novelty:
            return cls(
                supervised_threshold=float(np.quantile(benign_p_attack, 1.0 - total_fpr)),
                novelty_threshold=float("inf"),
                use_novelty=False,
            )
        per_head = per_head_budget(total_fpr, 2)
        return cls(
            supervised_threshold=float(np.quantile(benign_p_attack, 1.0 - per_head)),
            novelty_threshold=float(np.quantile(benign_novelty, 1.0 - per_head)),
            use_novelty=True,
        )

    def flags(self, p_attack: np.ndarray, novelty: np.ndarray) -> np.ndarray:
        """Boolean: did either head fire?"""
        fired = np.asarray(p_attack, dtype=float) >= self.supervised_threshold
        if self.use_novelty:
            fired = fired | (np.asarray(novelty, dtype=float) >= self.novelty_threshold)
        return fired

    def which_fired(self, p_attack: np.ndarray, novelty: np.ndarray) -> np.ndarray:
        """Per-row attribution: 0 neither, 1 supervised only, 2 novelty only, 3 both.

        Feeds the verdict lattice - `SUSPECTED_NOVEL` is precisely the 2 case, where the novelty
        head fired and the supervised head did not.
        """
        s = (np.asarray(p_attack, dtype=float) >= self.supervised_threshold).astype(int)
        n = (
            (np.asarray(novelty, dtype=float) >= self.novelty_threshold).astype(int)
            if self.use_novelty
            else np.zeros_like(s)
        )
        return s + 2 * n


def expected_or_fpr(fpr_a: float, fpr_b: float, *, correlation: float = 0.0) -> float:
    """FPR of two OR-ed detectors.

    Independent: 1 - (1-a)(1-b), which is nearly a+b for small rates. Perfectly correlated: max(a,b).
    `correlation` interpolates between them.

    Here to make a specific point concrete: three detectors each tuned to 1% do not give 1%, they
    give close to 3%. That is why `NoveltyEnsemble` fuses and thresholds ONCE instead of OR-ing
    three independently-thresholded detectors.
    """
    independent = 1.0 - (1.0 - fpr_a) * (1.0 - fpr_b)
    correlated = max(fpr_a, fpr_b)
    c = float(np.clip(correlation, 0.0, 1.0))
    return float((1.0 - c) * independent + c * correlated)


def per_head_budget(total_fpr: float, n_heads: int = 2) -> float:
    """Per-head FPR that ORs to approximately `total_fpr`.

    Solves 1 - (1-x)^n = total for x. With two heads at a 1% total budget each head runs at ~0.5%:
    the coverage a fused configuration gains is bought out of the allowance, not added on top.
    """
    return float(1.0 - (1.0 - total_fpr) ** (1.0 / n_heads))
