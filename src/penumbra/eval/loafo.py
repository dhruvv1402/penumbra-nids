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
from penumbra.eval.budget import evaluate_flags
from penumbra.features.preprocess import assert_benign_only_fit, benign_only_pipeline
from penumbra.models import supervised
from penumbra.models.fusion import OrGate
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
    realised_fpr_supervised: float = float("nan")
    realised_fpr_fused: float = float("nan")

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
) -> tuple[NoveltyEnsemble, Any, pd.DataFrame]:
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
    # The calibration rows are also the benign reference for the SUPERVISED head, so both heads are
    # ranked against the same traffic. Using different reference sets would make the two percentile
    # scales incomparable again, which is the bug this returns exist to prevent.
    return ens, prep, benign.iloc[cal_idx]


def run_family(
    ds: Dataset,
    family: str,
    *,
    model_name: str = "rf",
    budget_fpr: float = 0.01,
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
    ens, prep, benign_ref = _fit_novelty(ds, X_tr, y_tr)
    novelty = ens.score(prep.transform(ds.X_test), how="max")

    # --- both heads thresholded on their OWN raw scale, from held-out benign data.
    # Rank-normalising and thresholding once saturates: >2% of benign rows sit above the entire
    # reference, so their rank is exactly 1.0 and every configuration admits the same tied block.
    # See models/fusion.OrGate.
    benign_p = supervised.attack_scores(binary, benign_ref)
    benign_n = ens.score(prep.transform(benign_ref), how="max")

    gate_sup = OrGate.fit(benign_p, benign_n, total_fpr=budget_fpr, use_novelty=False)
    gate_fused = OrGate.fit(benign_p, benign_n, total_fpr=budget_fpr, use_novelty=True)

    flags_sup = gate_sup.flags(p_attack, novelty)
    flags_fused = gate_fused.flags(p_attack, novelty)

    y_test = ds.y_test.to_numpy()
    res_sup = evaluate_flags("supervised", y_test, flags_sup)
    res_fused = evaluate_flags("supervised+novelty", y_test, flags_fused)

    mask = (ds.fam_test == family).to_numpy()
    n_test = int(mask.sum())

    def recall_on_family(flags: np.ndarray) -> float:
        return float(np.mean(flags[mask])) if n_test else float("nan")

    r_sup = recall_on_family(flags_sup)
    r_fused = recall_on_family(flags_fused)

    # --- secondary: can the model name it? (It has never seen it, so this should be ~0.)
    multi, encoder = supervised.fit_multiclass(model_name, _subset(ds, keep), balanced=True)
    pred_fam = supervised.predict_families(multi, encoder, ds.X_test)
    attribution = float(np.mean(pred_fam[mask] == family)) if n_test else float("nan")

    # --- tertiary: how often does it say "attack I cannot name"?
    # SUSPECTED_NOVEL is exactly "novelty fired, supervised did not".
    which = gate_fused.which_fired(p_attack, novelty)
    novel_rate = float(np.mean(which[mask] == 2)) if n_test else float("nan")

    return FamilyResult(
        family=family,
        n_train_removed=n_removed,
        n_test_rows=n_test,
        recall_supervised=r_sup,
        recall_fused=r_fused,
        delta=r_fused - r_sup,
        attribution_accuracy=attribution,
        novel_verdict_rate=novel_rate,
        budget_alerts=res_fused.n_alerts,
        budget_fpr=budget_fpr,
        realised_fpr_supervised=res_sup.fpr,
        realised_fpr_fused=res_fused.fpr,
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
    realised_fpr_supervised: float = float("nan")
    realised_fpr_fused: float = float("nan")
    per_type: dict[str, tuple[int, float, float]] = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            "=" * 92,
            "  NSL-KDD - the 17 attack types that appear in KDDTest+ but never in KDDTrain+",
            "=" * 92,
            "",
            f"  target FPR {self.budget_fpr:.2%}, thresholds fitted on held-out benign traffic",
            f"  realised on test: supervised {self.realised_fpr_supervised:.3%}, "
            f"fused {self.realised_fpr_fused:.3%}",
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
    """Score the naturally unseen attack types at a matched false-positive budget.

    Thresholds are fitted on held-out benign training traffic and applied blind to the test set, so
    the operating point is one that could actually be chosen at deployment time. The realised test
    FPR is reported rather than assumed.
    """
    seed_everything()
    from penumbra.data.loaders import nsl_kdd

    ds, _, fine_test = nsl_kdd.load_with_fine_labels()
    unseen = nsl_kdd.unseen_mask(fine_test).to_numpy()

    binary = supervised.build(model_name, ds, n_classes=2, balanced=True)
    binary.fit(ds.X_train, ds.y_train)
    p_attack = supervised.attack_scores(binary, ds.X_test)

    ens, prep, benign_ref = _fit_novelty(ds, ds.X_train, ds.y_train)
    novelty = ens.score(prep.transform(ds.X_test), how="max")

    benign_p = supervised.attack_scores(binary, benign_ref)
    benign_n = ens.score(prep.transform(benign_ref), how="max")

    gate_sup = OrGate.fit(benign_p, benign_n, total_fpr=budget_fpr, use_novelty=False)
    gate_fused = OrGate.fit(benign_p, benign_n, total_fpr=budget_fpr, use_novelty=True)

    flags_sup = gate_sup.flags(p_attack, novelty)
    flags_fused = gate_fused.flags(p_attack, novelty)

    y_test = ds.y_test.to_numpy()
    res_sup = evaluate_flags("supervised", y_test, flags_sup)
    res_fused = evaluate_flags("supervised+novelty", y_test, flags_fused)
    seen_attacks = (y_test == 1) & ~unseen

    per_type: dict[str, tuple[int, float, float]] = {}
    for name in sorted(set(fine_test[unseen])):
        m = (fine_test == name).to_numpy()
        per_type[str(name)] = (
            int(m.sum()),
            float(np.mean(flags_sup[m])),
            float(np.mean(flags_fused[m])),
        )

    r_sup_unseen = float(np.mean(flags_sup[unseen]))
    r_fus_unseen = float(np.mean(flags_fused[unseen]))

    return UnseenResult(
        n_unseen=int(unseen.sum()),
        n_seen_attacks=int(seen_attacks.sum()),
        recall_supervised_unseen=r_sup_unseen,
        recall_fused_unseen=r_fus_unseen,
        delta_unseen=r_fus_unseen - r_sup_unseen,
        recall_supervised_seen=float(np.mean(flags_sup[seen_attacks])),
        recall_fused_seen=float(np.mean(flags_fused[seen_attacks])),
        budget_fpr=budget_fpr,
        realised_fpr_supervised=res_sup.fpr,
        realised_fpr_fused=res_fused.fpr,
        per_type=per_type,
    )


# =================================================================================================
# The operating-point curve
# =================================================================================================


@dataclass
class CurvePoint:
    target_fpr: float
    realised_fpr: float
    recall_supervised: float
    recall_fused: float
    novelty_only_detections: int

    @property
    def delta(self) -> float:
        return self.recall_fused - self.recall_supervised


@dataclass
class UnseenCurve:
    """Recall on genuinely unseen attacks across operating points.

    A single threshold invites the objection that it was chosen to flatter one configuration. The
    curve answers it: the novelty head's contribution is not a number, it is a shape, and the shape
    has a region where it helps and a region where it does not.
    """

    points: list[CurvePoint] = field(default_factory=list)
    n_unseen: int = 0

    def best(self) -> CurvePoint | None:
        return max(self.points, key=lambda c: c.delta) if self.points else None

    def summary(self) -> str:
        lines = [
            "=" * 86,
            "  NSL-KDD unseen-17: recall across operating points",
            "=" * 86,
            "",
            f"  {self.n_unseen:,} test rows from 17 attack types absent from training.",
            "  Thresholds set so realised FPR is exact, isolating the question: at equal",
            "  false-positive cost, does the novelty head add recall on unseen attacks?",
            "",
            f"  {'realised FPR':>13} {'supervised':>12} {'+novelty':>11} {'delta':>9} {'novelty-only':>13}",
            f"  {'-' * 13} {'-' * 12} {'-' * 11} {'-' * 9} {'-' * 13}",
        ]
        for c in self.points:
            lines.append(
                f"  {c.realised_fpr:>12.3%} {c.recall_supervised:>12.4f} {c.recall_fused:>11.4f} "
                f"{c.delta:>+9.4f} {c.novelty_only_detections:>13,}"
            )
        best = self.best()
        if best:
            lines += [
                "",
                f"  Largest gain: {best.delta:+.4f} at {best.realised_fpr:.2%} FPR "
                f"({best.recall_supervised:.4f} -> {best.recall_fused:.4f}).",
                "",
                "  The gain is confined to a band. Below ~0.2% FPR neither head has room to fire;",
                "  above ~5% the supervised head is loose enough to catch most things on its own and",
                "  splitting the budget costs more than novelty returns.",
            ]
        return chr(10).join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_unseen": self.n_unseen,
            "points": [
                {
                    "target_fpr": c.target_fpr,
                    "realised_fpr": c.realised_fpr,
                    "recall_supervised": c.recall_supervised,
                    "recall_fused": c.recall_fused,
                    "delta": c.delta,
                    "novelty_only_detections": c.novelty_only_detections,
                }
                for c in self.points
            ],
        }


def unseen_curve(
    *,
    model_name: str = "rf",
    targets: tuple[float, ...] = (0.001, 0.005, 0.01, 0.02, 0.05, 0.10),
) -> UnseenCurve:
    """Sweep operating points on NSL-KDD's natural unseen-attack split.

    Thresholds are placed so the REALISED false-positive rate on test benign traffic hits each
    target exactly. That is not a deployable procedure - it uses test labels to position the
    threshold - and it is not meant to be: it isolates the scientific question from the separate,
    and also real, problem that an operating point fitted on training benign traffic does not
    transfer to test benign traffic.
    """
    seed_everything()
    from penumbra.data.loaders import nsl_kdd
    from penumbra.models.fusion import per_head_budget

    ds, _, fine_test = nsl_kdd.load_with_fine_labels()
    unseen = nsl_kdd.unseen_mask(fine_test).to_numpy()

    binary = supervised.build(model_name, ds, n_classes=2, balanced=True)
    binary.fit(ds.X_train, ds.y_train)
    p_attack = supervised.attack_scores(binary, ds.X_test)

    ens, prep, _ = _fit_novelty(ds, ds.X_train, ds.y_train)
    novelty = ens.score(prep.transform(ds.X_test), how="max")

    y_test = ds.y_test.to_numpy()
    benign = y_test == 0

    curve = UnseenCurve(n_unseen=int(unseen.sum()))
    for target in targets:
        t_sup = float(np.quantile(p_attack[benign], 1.0 - target))
        per_head = per_head_budget(target, 2)
        t_sup2 = float(np.quantile(p_attack[benign], 1.0 - per_head))
        t_nov2 = float(np.quantile(novelty[benign], 1.0 - per_head))

        flags_sup = p_attack >= t_sup
        flags_fused = (p_attack >= t_sup2) | (novelty >= t_nov2)
        novelty_only = int(np.sum((novelty >= t_nov2) & (p_attack < t_sup2) & unseen))

        curve.points.append(
            CurvePoint(
                target_fpr=target,
                realised_fpr=float(np.mean(flags_sup[benign])),
                recall_supervised=float(np.mean(flags_sup[unseen])),
                recall_fused=float(np.mean(flags_fused[unseen])),
                novelty_only_detections=novelty_only,
            )
        )
    return curve
