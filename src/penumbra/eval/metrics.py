"""Honest metrics.

Design rules, each of which exists because the obvious alternative is misleading:

  * The PRIMARY metrics are TPR and FPR. They are prevalence-invariant, so they mean the same thing
    on UNSW-NB15's 55%-attack test set as they would on a real network at 1-in-10,000.
  * PR-AUC is never returned bare. It always carries the prevalence it was computed at and its
    no-skill baseline (which equals the prevalence). "PR-AUC 0.97" is not a statement; "PR-AUC 0.97
    at prevalence 0.55, no-skill 0.55" is.
  * `recall_at_fpr` and `fpr_at_recall` are what a SOC actually buys on, and both refuse to report
    below the estimation floor - with 37,000 benign rows, FPR = 0.01% is 3.7 false positives and is
    not a measurable quantity.

See docs/EVALUATION.md.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)

ArrayLike = np.ndarray


# =================================================================================================
# Estimation floor
# =================================================================================================


def estimation_floor(n_negatives: int) -> float:
    """The smallest FPR distinguishable from zero given this many benign rows.

    One false positive out of n. Any FPR target below this is asking for a fraction of an event, and
    the honest response is to say so rather than to report a number.
    """
    return 1.0 / n_negatives if n_negatives else float("nan")


class BelowEstimationFloor(ValueError):
    def __init__(self, target: float, floor: float, n_negatives: int) -> None:
        super().__init__(
            f"FPR target {target:.2e} is below the estimation floor {floor:.2e} "
            f"({n_negatives:,} benign rows = one false positive). Not estimable; report an "
            "interval or a larger target instead."
        )


# =================================================================================================
# Operating-point metrics
# =================================================================================================


def recall_at_fpr(y_true: ArrayLike, scores: ArrayLike, target_fpr: float) -> tuple[float, float]:
    """Recall achievable while holding FPR at or below target. Returns (recall, threshold).

    This is the number a SOC buys on: "at one false positive per hundred benign flows, what fraction
    of attacks do we catch?"
    """
    y_true = np.asarray(y_true)
    n_neg = int((y_true == 0).sum())
    floor = estimation_floor(n_neg)
    if target_fpr < floor:
        raise BelowEstimationFloor(target_fpr, floor, n_neg)

    fpr, tpr, thresholds = roc_curve(y_true, scores)
    eligible = fpr <= target_fpr
    if not eligible.any():
        return 0.0, float("inf")
    idx = int(np.argmax(tpr * eligible))
    return float(tpr[idx]), float(thresholds[idx])


def fpr_at_recall(y_true: ArrayLike, scores: ArrayLike, target_recall: float) -> tuple[float, float]:
    """Lowest FPR at which the target recall is still met. Returns (fpr, threshold)."""
    fpr, tpr, thresholds = roc_curve(np.asarray(y_true), scores)
    eligible = tpr >= target_recall
    if not eligible.any():
        return 1.0, float("-inf")
    idx = int(np.argmin(np.where(eligible, fpr, np.inf)))
    return float(fpr[idx]), float(thresholds[idx])


def precision_at_k(y_true: ArrayLike, scores: ArrayLike, k: int) -> float:
    """Fraction of the top-k scored rows that are genuinely attacks.

    The honest metric for the hunting lane, which works a fixed daily budget rather than everything
    above a threshold. FPR is the wrong question for a queue that is capped by construction.
    """
    y_true = np.asarray(y_true)
    k = min(k, len(y_true))
    if k == 0:
        return float("nan")
    top = np.argsort(scores)[::-1][:k]
    return float(y_true[top].mean())


# =================================================================================================
# The report object
# =================================================================================================


@dataclass
class BinaryMetrics:
    """Everything we are willing to say about a binary detector at one operating point."""

    n: int
    n_positive: int
    n_negative: int
    prevalence: float
    threshold: float

    # Prevalence-invariant. These are the headline numbers.
    tpr: float  # recall / sensitivity
    fpr: float
    tnr: float
    fnr: float
    roc_auc: float

    # Prevalence-DEPENDENT. Meaningful only alongside `prevalence`.
    precision: float
    f1: float
    pr_auc: float
    pr_auc_no_skill: float  # equals prevalence; a PR-AUC at this level is worthless

    tp: int
    fp: int
    tn: int
    fn: int

    brier: float | None = None
    recall_at_fpr: dict[str, float] = field(default_factory=dict)
    fpr_at_recall: dict[str, float] = field(default_factory=dict)
    precision_at_k: dict[str, float] = field(default_factory=dict)
    estimation_floor_fpr: float = float("nan")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @property
    def pr_auc_lift(self) -> float:
        """How much better than chance the PR-AUC actually is."""
        return self.pr_auc - self.pr_auc_no_skill

    def summary(self) -> str:
        return "\n".join(
            [
                f"  n={self.n:,}  attacks={self.n_positive:,}  benign={self.n_negative:,}  "
                f"prevalence={self.prevalence:.4f}",
                f"  threshold={self.threshold:.4f}",
                "",
                "  prevalence-invariant (report these):",
                f"    TPR (recall)  {self.tpr:.4f}",
                f"    FPR           {self.fpr:.4f}",
                f"    ROC-AUC       {self.roc_auc:.4f}",
                "",
                f"  prevalence-dependent (only meaningful at prevalence={self.prevalence:.4f}):",
                f"    precision     {self.precision:.4f}",
                f"    F1            {self.f1:.4f}",
                f"    PR-AUC        {self.pr_auc:.4f}   (no-skill {self.pr_auc_no_skill:.4f}, "
                f"lift {self.pr_auc_lift:+.4f})",
                "",
                f"  confusion: TP={self.tp:,} FP={self.fp:,} TN={self.tn:,} FN={self.fn:,}",
                f"  estimation floor for FPR: {self.estimation_floor_fpr:.2e} "
                f"(one false positive in {self.n_negative:,} benign rows)",
            ]
        )


def binary_metrics(
    y_true: ArrayLike,
    scores: ArrayLike,
    *,
    threshold: float = 0.5,
    fpr_targets: tuple[float, ...] = (0.001, 0.01, 0.05),
    recall_targets: tuple[float, ...] = (0.90, 0.95, 0.99),
    k_values: tuple[int, ...] = (50, 100, 500),
    is_probability: bool = True,
) -> BinaryMetrics:
    """Compute the full honest metric set at one threshold."""
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    y_pred = (scores >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    n_pos, n_neg = int((y_true == 1).sum()), int((y_true == 0).sum())
    prevalence = n_pos / len(y_true) if len(y_true) else float("nan")

    precision, _, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)

    r_at_f: dict[str, float] = {}
    for t in fpr_targets:
        try:
            r_at_f[f"{t:g}"] = recall_at_fpr(y_true, scores, t)[0]
        except BelowEstimationFloor:
            # Not an error - a target we decline to report. Recorded as NaN so the report can say
            # "not estimable" rather than silently omitting it.
            r_at_f[f"{t:g}"] = float("nan")

    return BinaryMetrics(
        n=len(y_true),
        n_positive=n_pos,
        n_negative=n_neg,
        prevalence=prevalence,
        threshold=threshold,
        tpr=float(tp / (tp + fn)) if (tp + fn) else float("nan"),
        fpr=float(fp / (fp + tn)) if (fp + tn) else float("nan"),
        tnr=float(tn / (tn + fp)) if (tn + fp) else float("nan"),
        fnr=float(fn / (fn + tp)) if (fn + tp) else float("nan"),
        roc_auc=float(roc_auc_score(y_true, scores)),
        precision=float(precision),
        f1=float(f1),
        pr_auc=float(average_precision_score(y_true, scores)),
        pr_auc_no_skill=prevalence,
        tp=int(tp),
        fp=int(fp),
        tn=int(tn),
        fn=int(fn),
        brier=float(brier_score_loss(y_true, scores)) if is_probability else None,
        recall_at_fpr=r_at_f,
        fpr_at_recall={f"{t:g}": fpr_at_recall(y_true, scores, t)[0] for t in recall_targets},
        precision_at_k={str(k): precision_at_k(y_true, scores, k) for k in k_values},
        estimation_floor_fpr=estimation_floor(n_neg),
    )


# =================================================================================================
# Per-class
# =================================================================================================


@dataclass
class ClassMetrics:
    label: str
    support: int
    precision: float
    recall: float
    f1: float
    # True when support is too small for a point estimate to mean anything. UNSW's Worms has 44 test
    # rows, which puts its recall interval at roughly +/-15 points.
    below_floor: bool

    def format_recall(self) -> str:
        return f"{self.recall:.3f}{' (n too small - interval only)' if self.below_floor else ''}"


def per_class_metrics(
    y_true: ArrayLike, y_pred: ArrayLike, labels: list[str], *, min_support: int = 100
) -> list[ClassMetrics]:
    """Per-family precision/recall/F1, flagging classes too rare to quote as point estimates."""
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    return [
        ClassMetrics(
            label=str(lab),
            support=int(s),
            precision=float(p),
            recall=float(r),
            f1=float(f),
            below_floor=int(s) < min_support,
        )
        for lab, p, r, f, s in zip(labels, precision, recall, f1, support, strict=True)
    ]


def macro_f1(class_metrics: list[ClassMetrics]) -> float:
    """Unweighted mean F1.

    The honest aggregate under imbalance: weighted-F1 lets Generic's 18,871 test rows drown out
    Worms' 44, which is precisely the failure the brief asks us not to hide.
    """
    return float(np.mean([c.f1 for c in class_metrics])) if class_metrics else float("nan")
