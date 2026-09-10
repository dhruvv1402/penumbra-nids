"""Leave-One-Attack-Family-Out: does a benign-only novelty head recover recall on attacks the
supervised model never saw?

The pre-registered central experiment. Hypothesis, predicted outcome and falsification condition
are in docs/EXPERIMENTS.md E1, committed before this code ran.

Protocol, per attack family F:

  1. Remove every row of F from training. Benign rows are untouched, so the novelty head is
     unaffected by the removal - which is the point.
  2. Train the supervised head on what remains.
  3. Train the novelty head on benign training rows only.
  4. Score the FULL test set with both.
  5. Fix an alert budget over the full test set, identically for both configurations.
  6. Measure recall on F's test rows alone.

Three metrics, in priority order:

  primary    binary attack-vs-normal recall on F, at matched budget. "Did the SOC get told?"
  secondary  family-attribution accuracy on F. Expected to be ~0 - you cannot name a class you have
             never seen - and reported separately because "we detected it, we could not name it" is
             the honest and more interesting finding.
  tertiary   SUSPECTED_NOVEL rate on F: how often the system correctly says "an attack I have no
             name for". The actual product claim.

Multiclass recall is deliberately NOT the primary metric. UNSW-NB15's family boundaries overlap
heavily; if `Backdoor` is held out and the model labels those flows `Exploits`, the SOC still got an
alert. Scoring that as a miss would rig the experiment in our favour.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from penumbra.data.loaders.base import Dataset
from penumbra.eval.budget import BudgetComparison, compare_at_matched_budget
from penumbra.features.preprocess import assert_benign_only_fit, benign_only_pipeline
from penumbra.models import supervised
from penumbra.models.novelty.ensemble import NoveltyEnsemble
from penumbra.seeds import seed_everything

# Fraction of benign training rows held back to calibrate novelty percentiles. Fitting the reference
# distribution on the same rows the detectors trained on makes every real flow look more novel than
# it is, and understates the false-positive rate at any chosen percentile.
CALIBRATION_FRACTION = 0.2


@dataclass
class FamilyResult:
    family: str
    n_train_removed: int
    n_test_rows: int

    recall_supervised: float
    recall_fused: float
    delta: float

    attribution_accuracy: float
    novel_verdict_rate: float

    budget_alerts: int
    budget_fpr: float

    def summary(self) -> str:
        return (
            f"  {self.family:<16} {self.n_test_rows:>7,} {self.recall_supervised:>12.4f} "
            f"{self.recall_fused:>12.4f} {self.delta:>+9.4f} {self.attribution_accuracy:>12.4f} "
            f"{self.novel_verdict_rate:>10.4f}"
        )


@dataclass
class LoafoMatrix:
    dataset: str
    results: list[FamilyResult] = field(default_factory=list)
    budget_fpr: float = 0.0

    @property
    def mean_delta(self) -> float:
        return float(np.mean([r.delta for r in self.results])) if self.results else float("nan")

    @property
    def n_improved(self) -> int:
        return sum(1 for r in self.results if r.delta > 0.001)

    @property
    def n_unchanged(self) -> int:
        return sum(1 for r in self.results if abs(r.delta) <= 0.001)

    @property
    def n_degraded(self) -> int:
        return sum(1 for r in self.results if r.delta < -0.001)

    def summary(self) -> str:
        lines = [
            "=" * 92,
            f"  LOAFO - {self.dataset}   (matched FPR {self.budget_fpr:.2%} of benign rows flagged)",
            "=" * 92,
            "",
            f"  {'held-out family':<16} {'test n':>7} {'supervised':>12} {'+novelty':>12} "
            f"{'delta':>9} {'attribution':>12} {'NOVEL':>10}",
            f"  {'-' * 16} {'-' * 7} {'-' * 12} {'-' * 12} {'-' * 9} {'-' * 12} {'-' * 10}",
        ]
        lines.extend(r.summary() for r in self.results)
        lines.append("")
        lines.append(
            f"  mean delta {self.mean_delta:+.4f}   improved {self.n_improved}   "
            f"unchanged {self.n_unchanged}   degraded {self.n_degraded}   of {len(self.results)}"
        )
        lines.append("")
        lines.append(
            "  'supervised' and '+novelty' are compared at IDENTICAL alert volume. Without that "
            "constraint\n  the comparison would measure nothing but the extra alerts."
        )
        lines.append(
            "  'attribution' is how often the model names the held-out family correctly. It cannot "
            "name a\n  class it never saw, so a near-zero column here is the expected result, not a "
            "failure."
        )
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "budget_fpr": self.budget_fpr,
            "mean_delta": self.mean_delta,
            "n_improved": self.n_improved,
            "n_unchanged": self.n_unchanged,
            "n_degraded": self.n_degraded,
            "families": [
                {
                    "family": r.family,
                    "n_train_removed": r.n_train_removed,
                    "n_test_rows": r.n_test_rows,
                    "recall_supervised": r.recall_supervised,
                    "recall_fused": r.recall_fused,
                    "delta": r.delta,
                    "attribution_accuracy": r.attribution_accuracy,
                    "novel_verdict_rate": r.novel_verdict_rate,
                }
                for r in self.results
            ],
        }


def _fit_novelty(
    ds: Dataset, X_train_subset: pd.DataFrame, y_train_subset: pd.Series
) -> tuple[NoveltyEnsemble, Any]:
    """Fit the benign-only pipeline and novelty ensemble.

    The benign/calibration split matters: the detectors fit on one part, the percentile reference is
    built from the other. Sharing them would make the reference distribution optimistic.
    """
    benign = X_train_subset.loc[y_train_subset == 0]
    assert_benign_only_fit(ds, benign)

    rng = np.random.default_rng(0)
    shuffled = rng.permutation(len(benign))
    n_cal = int(len(benign) * CALIBRATION_FRACTION)
    cal_idx, fit_idx = shuffled[:n_cal], shuffled[n_cal:]

    prep = benign_only_pipeline(ds)
    Z_fit = prep.fit_transform(benign.iloc[fit_idx])
    Z_cal = prep.transform(benign.iloc[cal_idx])

    ens = NoveltyEnsemble().fit(Z_fit)
    ens.calibrate_on(Z_cal)
    return ens, prep


def run_family(
    ds: Dataset,
    family: str,
    *,
    model_name: str = "rf",
    budget_fpr: float = 0.01,
    novel_percentile: float = 0.99,
) -> FamilyResult:
    """One LOAFO fold: hold out `family`, train, and compare at matched budget."""
    seed_everything()

    keep = ds.fam_train != family
    n_removed = int((~keep).sum())
    X_tr, y_tr = ds.X_train[keep], ds.y_train[keep]

    # --- supervised head, blind to this family
    binary = supervised.build(model_name, ds, n_classes=2, balanced=True)
    binary.fit(X_tr, y_tr)
    p_attack = supervised.attack_scores(binary, ds.X_test)

    # --- novelty head, benign only, unaffected by the removal
    ens, prep = _fit_novelty(ds, X_tr, y_tr)
    novelty = ens.score(prep.transform(ds.X_test), how="max")

    # --- fused: the union, expressed as a single score so it can be thresholded once
    fused = np.maximum(p_attack, novelty)

    y_test = ds.y_test.to_numpy()
    comparison = compare_at_matched_budget(
        y_test,
        {"supervised": p_attack, "supervised+novelty": fused},
        match="fpr",
        budget_fpr=budget_fpr,
    )

    # Recall restricted to the held-out family's rows, at the budget-derived thresholds.
    mask = (ds.fam_test == family).to_numpy()
    n_test = int(mask.sum())

    def recall_on_family(scores: np.ndarray, threshold: float) -> float:
        if n_test == 0:
            return float("nan")
        return float(np.mean(scores[mask] >= threshold))

    r_sup = recall_on_family(p_attack, comparison.results["supervised"].threshold)
    r_fused = recall_on_family(fused, comparison.results["supervised+novelty"].threshold)

    # --- secondary: can the model name it? (It has never seen it, so this should be ~0.)
    multi, encoder = supervised.fit_multiclass(model_name, _subset(ds, keep), balanced=True)
    pred_fam = supervised.predict_families(multi, encoder, ds.X_test)
    attribution = float(np.mean(pred_fam[mask] == family)) if n_test else float("nan")

    # --- tertiary: how often does it say "attack I cannot name"?
    novel_thr = float(np.quantile(novelty, novel_percentile))
    sup_thr = comparison.results["supervised"].threshold
    is_novel = (novelty >= novel_thr) & (p_attack < sup_thr)
    novel_rate = float(np.mean(is_novel[mask])) if n_test else float("nan")

    return FamilyResult(
        family=family,
        n_train_removed=n_removed,
        n_test_rows=n_test,
        recall_supervised=r_sup,
        recall_fused=r_fused,
        delta=r_fused - r_sup,
        attribution_accuracy=attribution,
        novel_verdict_rate=novel_rate,
        budget_alerts=comparison.budget_count,
        budget_fpr=comparison.budget_value,
    )


def _subset(ds: Dataset, keep: pd.Series) -> Dataset:
    return Dataset(
        name=f"{ds.name}-loafo",
        X_train=ds.X_train[keep],
        y_train=ds.y_train[keep],
        fam_train=ds.fam_train[keep],
        X_test=ds.X_test,
        y_test=ds.y_test,
        fam_test=ds.fam_test,
        categorical=ds.categorical,
        numeric=ds.numeric,
        suspected_artifacts=ds.suspected_artifacts,
        capabilities=ds.capabilities,
    )


def run(
    ds: Dataset,
    *,
    families: list[str] | None = None,
    model_name: str = "rf",
    budget_fpr: float = 0.01,
    on_progress: Any = None,
) -> LoafoMatrix:
    """Run LOAFO across every attack family."""
    if families is None:
        families = sorted(f for f in ds.fam_train.unique() if str(f).lower() not in {"normal", "benign"})

    matrix = LoafoMatrix(dataset=ds.name, budget_fpr=budget_fpr)
    for fam in families:
        if on_progress:
            on_progress(fam)
        matrix.results.append(run_family(ds, fam, model_name=model_name, budget_fpr=budget_fpr))
    return matrix


# =================================================================================================
# NSL-KDD's natural experiment
# =================================================================================================


@dataclass
class UnseenResult:
    """The 17 attack types present in KDDTest+ but absent from KDDTrain+.

    No holdout construction by us - the dataset's authors built this. It is the external-validity
    half of E1: LOAFO on UNSW is controlled and repeatable, this is real.
    """

    n_unseen: int
    n_seen_attacks: int
    recall_supervised_unseen: float
    recall_fused_unseen: float
    delta_unseen: float
    recall_supervised_seen: float
    recall_fused_seen: float
    budget_fpr: float
    per_type: dict[str, tuple[int, float, float]] = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            "=" * 92,
            "  NSL-KDD - the 17 attack types that appear in KDDTest+ but never in KDDTrain+",
            "=" * 92,
            "",
            f"  matched FPR {self.budget_fpr:.2%} - the same number of benign rows flagged by both",
            "",
            f"  {'population':<28} {'n':>8} {'supervised':>12} {'+novelty':>12} {'delta':>9}",
            f"  {'-' * 28} {'-' * 8} {'-' * 12} {'-' * 12} {'-' * 9}",
            f"  {'genuinely unseen types':<28} {self.n_unseen:>8,} "
            f"{self.recall_supervised_unseen:>12.4f} {self.recall_fused_unseen:>12.4f} "
            f"{self.delta_unseen:>+9.4f}",
            f"  {'attack types seen in train':<28} {self.n_seen_attacks:>8,} "
            f"{self.recall_supervised_seen:>12.4f} {self.recall_fused_seen:>12.4f} "
            f"{self.recall_fused_seen - self.recall_supervised_seen:>+9.4f}",
        ]
        if self.per_type:
            lines.extend(["", f"  {'attack type':<20} {'n':>6} {'supervised':>12} {'+novelty':>12}"])
            lines.append(f"  {'-' * 20} {'-' * 6} {'-' * 12} {'-' * 12}")
            for name, (n, sup, fus) in sorted(self.per_type.items(), key=lambda kv: -kv[1][0]):
                lines.append(f"  {name:<20} {n:>6,} {sup:>12.4f} {fus:>12.4f}")
        return "\n".join(lines)


def run_nslkdd_unseen(*, model_name: str = "rf", budget_fpr: float = 0.01) -> UnseenResult:
    """Score the naturally unseen attack types, at matched alert budget."""
    seed_everything()
    from penumbra.data.loaders import nsl_kdd

    ds, _, fine_test = nsl_kdd.load_with_fine_labels()
    unseen = nsl_kdd.unseen_mask(fine_test).to_numpy()

    binary = supervised.build(model_name, ds, n_classes=2, balanced=True)
    binary.fit(ds.X_train, ds.y_train)
    p_attack = supervised.attack_scores(binary, ds.X_test)

    ens, prep = _fit_novelty(ds, ds.X_train, ds.y_train)
    novelty = ens.score(prep.transform(ds.X_test), how="max")
    fused = np.maximum(p_attack, novelty)

    y_test = ds.y_test.to_numpy()
    comp: BudgetComparison = compare_at_matched_budget(
        y_test,
        {"supervised": p_attack, "supervised+novelty": fused},
        match="fpr",
        budget_fpr=budget_fpr,
    )
    thr_sup = comp.results["supervised"].threshold
    thr_fus = comp.results["supervised+novelty"].threshold

    seen_attacks = (y_test == 1) & ~unseen

    per_type: dict[str, tuple[int, float, float]] = {}
    for name in sorted(set(fine_test[unseen])):
        m = (fine_test == name).to_numpy()
        per_type[str(name)] = (
            int(m.sum()),
            float(np.mean(p_attack[m] >= thr_sup)),
            float(np.mean(fused[m] >= thr_fus)),
        )

    r_sup_unseen = float(np.mean(p_attack[unseen] >= thr_sup))
    r_fus_unseen = float(np.mean(fused[unseen] >= thr_fus))

    return UnseenResult(
        n_unseen=int(unseen.sum()),
        n_seen_attacks=int(seen_attacks.sum()),
        recall_supervised_unseen=r_sup_unseen,
        recall_fused_unseen=r_fus_unseen,
        delta_unseen=r_fus_unseen - r_sup_unseen,
        recall_supervised_seen=float(np.mean(p_attack[seen_attacks] >= thr_sup)),
        recall_fused_seen=float(np.mean(fused[seen_attacks] >= thr_fus)),
        budget_fpr=comp.budget_value,
        per_type=per_type,
    )
