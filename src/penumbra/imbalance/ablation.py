"""The imbalance ablation, and a deliberate demonstration of the leak it is designed to avoid.

`run()` compares every strategy on the natural test distribution. `leakage_demonstration()` runs the
same pipeline twice - once correctly, once with resampling applied before the train/test split - and
reports both numbers side by side.

The leakage demo is the more useful half. "Apply SMOTE inside the CV fold" is advice everyone has
read and many people skip, because the consequence is invisible: the leaky version does not crash,
it just reports a better number. Printing both is the only argument that lands.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

from penumbra.data.loaders.base import Dataset
from penumbra.eval.metrics import binary_metrics, recall_at_fpr
from penumbra.features.preprocess import supervised_pipeline
from penumbra.imbalance import strategies
from penumbra.models.supervised import _estimator
from penumbra.seeds import SEED, seed_everything


@dataclass
class StrategyResult:
    strategy: str
    roc_auc: float
    pr_auc: float
    recall: float
    fpr: float
    precision: float
    f1: float
    recall_at_1pct_fpr: float
    minority_recall: float
    train_seconds: float
    n_train_rows_after_resampling: int

    def summary(self) -> str:
        return (
            f"  {self.strategy:<18} {self.roc_auc:>8.4f} {self.recall:>8.4f} {self.fpr:>8.4f} "
            f"{self.recall_at_1pct_fpr:>10.4f} {self.minority_recall:>10.4f} "
            f"{self.train_seconds:>8.1f}"
        )


@dataclass
class AblationResult:
    dataset: str
    model: str
    prevalence: float
    minority_family: str
    results: list[StrategyResult] = field(default_factory=list)

    def best_by(self, attr: str) -> StrategyResult:
        return max(self.results, key=lambda r: getattr(r, attr))

    def summary(self) -> str:
        lines = [
            "=" * 88,
            f"  Class-imbalance ablation - {self.dataset} ({self.model})",
            "=" * 88,
            "",
            f"  Evaluated on the NATURAL test distribution (prevalence {self.prevalence:.4f}).",
            f"  'minority recall' is the {self.minority_family} family specifically.",
            "",
            f"  {'strategy':<18} {'ROC-AUC':>8} {'recall':>8} {'FPR':>8} {'R@1%FPR':>10} "
            f"{'minority':>10} {'fit s':>8}",
            f"  {'-' * 18} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 10} {'-' * 10} {'-' * 8}",
        ]
        lines.extend(r.summary() for r in self.results)

        best_op = self.best_by("recall_at_1pct_fpr")
        best_min = self.best_by("minority_recall")
        lines += [
            "",
            f"  Best at the operating point (recall @1% FPR): {best_op.strategy} "
            f"({best_op.recall_at_1pct_fpr:.4f})",
            f"  Best on the minority family:                  {best_min.strategy} "
            f"({best_min.minority_recall:.4f})",
        ]
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "model": self.model,
            "prevalence": self.prevalence,
            "minority_family": self.minority_family,
            "strategies": [r.__dict__ for r in self.results],
        }


def _minority_family(ds: Dataset) -> str:
    counts = ds.fam_train[ds.y_train == 1].value_counts()
    return str(counts.index[-1]) if len(counts) else "unknown"


def run(
    ds: Dataset,
    *,
    model: str = "rf",
    names: tuple[str, ...] | None = None,
    on_progress: Any = None,
) -> AblationResult:
    """Fit every strategy and score it on the natural test distribution."""
    names = names or tuple(strategies.STRATEGIES)
    minority = _minority_family(ds)
    minority_mask = (ds.fam_test == minority).to_numpy()
    y_test = ds.y_test.to_numpy()

    out = AblationResult(
        dataset=ds.name,
        model=model,
        prevalence=float(np.mean(y_test)),
        minority_family=minority,
    )

    for name in names:
        if on_progress:
            on_progress(name)
        seed_everything()

        pipe = strategies.build(name, ds, model=model)
        start = time.perf_counter()
        pipe.fit(ds.X_train, ds.y_train)
        elapsed = time.perf_counter() - start

        scores = pipe.predict_proba(ds.X_test)[:, 1]

        # threshold_moving is the one strategy whose operating point is not 0.5: it trains
        # unweighted and then moves the boundary to hit a target FPR.
        threshold = 0.5
        if name == "threshold_moving":
            threshold = recall_at_fpr(y_test, scores, 0.01)[1]

        m = binary_metrics(y_test, scores, threshold=threshold)
        flagged = scores >= threshold

        # How many rows the resampler produced, so the cost of each strategy is visible.
        n_after = len(ds.X_train)
        if "resample" in getattr(pipe, "named_steps", {}):
            try:
                Xp = pipe.named_steps["prep"].transform(ds.X_train)
                _, y_res = pipe.named_steps["resample"].fit_resample(Xp, ds.y_train)
                n_after = len(y_res)
            except Exception:  # noqa: BLE001 - reporting only; never fail the ablation for this
                pass

        out.results.append(
            StrategyResult(
                strategy=name,
                roc_auc=m.roc_auc,
                pr_auc=m.pr_auc,
                recall=m.tpr,
                fpr=m.fpr,
                precision=m.precision,
                f1=m.f1,
                recall_at_1pct_fpr=m.recall_at_fpr.get("0.01", float("nan")),
                minority_recall=float(np.mean(flagged[minority_mask]))
                if minority_mask.any()
                else float("nan"),
                train_seconds=elapsed,
                n_train_rows_after_resampling=n_after,
            )
        )
    return out


# =================================================================================================
# The leakage demonstration
# =================================================================================================


@dataclass
class LeakageResult:
    """The same pipeline, done right and done wrong."""

    target_family: str
    n_positive: int
    prevalence: float
    correct_f1: float
    correct_pr_auc: float
    correct_roc_auc: float
    leaky_f1: float
    leaky_pr_auc: float
    leaky_roc_auc: float
    n_train: int
    n_val: int
    impossible_rows: list[dict[str, float]] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            "=" * 88,
            "  SMOTE leakage demonstration",
            "=" * 88,
            "",
            f"  Target: {self.target_family} vs everything else - {self.n_positive:,} positive",
            f"  rows, prevalence {self.prevalence:.4%}. A rare family, not attack-vs-normal, because",
            "  that is where resampling does real work and therefore where leaking it matters.",
            "",
            "  Identical model, identical data, identical resampler. The only difference is WHERE",
            "  the resampling happens relative to the split.",
            "",
            f"  {'':<34} {'F1':>10} {'PR-AUC':>10} {'ROC-AUC':>10}",
            f"  {'-' * 34} {'-' * 10} {'-' * 10} {'-' * 10}",
            f"  {'correct (resample inside fold)':<34} {self.correct_f1:>10.4f} "
            f"{self.correct_pr_auc:>10.4f} {self.correct_roc_auc:>10.4f}",
            f"  {'leaky (resample before split)':<34} {self.leaky_f1:>10.4f} "
            f"{self.leaky_pr_auc:>10.4f} {self.leaky_roc_auc:>10.4f}",
            "",
            f"  inflation: F1 {self.leaky_f1 - self.correct_f1:+.4f}, "
            f"PR-AUC {self.leaky_pr_auc - self.correct_pr_auc:+.4f}",
            "",
            "  The leaky version does not fail. It reports a better number, which is exactly why",
            "  this mistake survives review. SMOTE synthesises minority rows by interpolating",
            "  between neighbours; done before the split, a validation row can be interpolated",
            "  from its own neighbours and the model has effectively seen it.",
        ]
        if self.impossible_rows:
            lines += [
                "",
                "  And a second objection, independent of leakage: SMOTE on flow data generates",
                "  rows that could not exist on a wire. One synthetic minority sample:",
                "",
            ]
            for feat, val in list(self.impossible_rows[0].items())[:5]:
                lines.append(f"    {feat:<34} {val:>12.4f}   <- counts are integers")
            lines += [
                "",
                "  A flow with a fractional packet count is not a rare flow. It is not a flow.",
                "  This is the argument for threshold-moving over resampling on network data.",
            ]
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "correct": {
                "f1": self.correct_f1,
                "pr_auc": self.correct_pr_auc,
                "roc_auc": self.correct_roc_auc,
            },
            "leaky": {"f1": self.leaky_f1, "pr_auc": self.leaky_pr_auc, "roc_auc": self.leaky_roc_auc},
            "f1_inflation": self.leaky_f1 - self.correct_f1,
            "pr_auc_inflation": self.leaky_pr_auc - self.correct_pr_auc,
            "impossible_rows": self.impossible_rows,
        }


def leakage_demonstration(
    ds: Dataset, *, model: str = "rf", target_family: str | None = None
) -> LeakageResult:
    """Run the same pipeline correctly and leakily, and report both.

    The target is **one rare family against everything else**, not attack-vs-normal. That choice is
    necessary rather than cosmetic: NSL-KDD's binary target is 46.5% positive and UNSW's is 68%, so
    SMOTE generates almost nothing and leaking it changes almost nothing. The first version of this
    demonstration used the binary target and reported an inflation of -0.0000, which was a true
    measurement of an uninteresting question.

    Leakage inflates exactly where resampling does real work: a family with a few hundred rows,
    where SMOTE manufactures most of the minority class and a validation row can be interpolated
    from its own neighbours.

    Uses a held-out slice of TRAINING data as the validation set, so the real test set is untouched
    by a demonstration of how to misuse it.
    """
    seed_everything()

    family = target_family or _minority_family(ds)
    y_binary = (ds.fam_train == family).astype(int)

    X_tr, X_val, y_tr, y_val = train_test_split(
        ds.X_train, y_binary, test_size=0.25, stratify=y_binary, random_state=SEED
    )

    # --- correct: resampling lives inside the pipeline, so it never sees X_val
    correct = strategies.build("smote", ds, model=model)
    correct.fit(X_tr, y_tr)
    p_correct = correct.predict_proba(X_val)[:, 1]

    # --- leaky: resample the WHOLE training pool first, then split
    prep = supervised_pipeline(ds, scale=model == "logreg")
    X_all = prep.fit_transform(ds.X_train)
    X_res, y_res = SMOTE(random_state=SEED).fit_resample(X_all, y_binary)
    Xl_tr, Xl_val, yl_tr, yl_val = train_test_split(
        X_res, y_res, test_size=0.25, stratify=y_res, random_state=SEED
    )
    leaky = _estimator(model, n_classes=2, class_weight=None, scale_pos_weight=None)
    leaky.fit(Xl_tr, yl_tr)
    p_leaky = leaky.predict_proba(Xl_val)[:, 1]

    return LeakageResult(
        target_family=family,
        n_positive=int(y_binary.sum()),
        prevalence=float(y_binary.mean()),
        correct_f1=float(f1_score(y_val, p_correct >= 0.5, zero_division=0)),
        correct_pr_auc=float(average_precision_score(y_val, p_correct)),
        correct_roc_auc=float(roc_auc_score(y_val, p_correct)),
        leaky_f1=float(f1_score(yl_val, p_leaky >= 0.5, zero_division=0)),
        leaky_pr_auc=float(average_precision_score(yl_val, p_leaky)),
        leaky_roc_auc=float(roc_auc_score(yl_val, p_leaky)),
        n_train=len(X_tr),
        n_val=len(X_val),
        impossible_rows=strategies.physically_impossible_example(ds, n=2),
    )
