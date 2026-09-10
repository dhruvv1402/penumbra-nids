"""Train, evaluate, and report - the Phase 1 end-to-end path.

Produces every number the report quotes, so that "regenerate the results" is one command and no
figure in the deck is typed by hand.

The with/without-artifacts comparison is built in rather than bolted on: `evaluate_dataset` runs
the whole suite twice, once on all features and once with the audit's quarantined set removed. The
second number is the one we stand behind, and having both in the same object makes it impossible to
quote the flattering one by accident.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score

from penumbra.config import settings
from penumbra.data.loaders.base import Dataset
from penumbra.eval import prevalence as prev_mod
from penumbra.eval.bootstrap import Interval, mcnemar, stratified_bootstrap
from penumbra.eval.metrics import BinaryMetrics, binary_metrics, macro_f1, per_class_metrics
from penumbra.models import supervised
from penumbra.seeds import seed_everything


@dataclass
class ModelResult:
    model: str
    metrics: BinaryMetrics
    roc_auc_ci: Interval
    recall_ci: Interval
    train_seconds: float
    predictions: np.ndarray = field(repr=False, default_factory=lambda: np.array([]))
    scores: np.ndarray = field(repr=False, default_factory=lambda: np.array([]))

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "train_seconds": round(self.train_seconds, 2),
            "roc_auc": {
                "point": self.roc_auc_ci.point,
                "ci95": [self.roc_auc_ci.lower, self.roc_auc_ci.upper],
            },
            "recall": {
                "point": self.recall_ci.point,
                "ci95": [self.recall_ci.lower, self.recall_ci.upper],
            },
            "metrics": self.metrics.to_dict(),
        }


@dataclass
class DatasetResult:
    dataset: str
    with_artifacts: dict[str, ModelResult]
    without_artifacts: dict[str, ModelResult]
    quarantined: list[str]

    def champion(self, *, artifacts: bool = False) -> ModelResult:
        pool = self.with_artifacts if artifacts else self.without_artifacts
        return max(pool.values(), key=lambda r: r.metrics.roc_auc)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "quarantined_features": self.quarantined,
            "with_artifacts": {k: v.to_dict() for k, v in self.with_artifacts.items()},
            "without_artifacts": {k: v.to_dict() for k, v in self.without_artifacts.items()},
        }


def train_and_score(name: str, ds: Dataset, *, n_boot: int = 500) -> ModelResult:
    """Fit one model on train, score test, and attach confidence intervals."""
    seed_everything()
    model = supervised.build(name, ds, n_classes=2, balanced=True)

    start = time.perf_counter()
    model.fit(ds.X_train, ds.y_train)
    elapsed = time.perf_counter() - start

    scores = supervised.attack_scores(model, ds.X_test)
    y_test = ds.y_test.to_numpy()
    metrics = binary_metrics(y_test, scores, threshold=0.5)

    auc_ci = stratified_bootstrap(y_test, scores, lambda y, s: float(roc_auc_score(y, s)), n_resamples=n_boot)
    recall_ci = stratified_bootstrap(
        y_test,
        scores,
        lambda y, s: float(((s >= 0.5) & (y == 1)).sum() / max((y == 1).sum(), 1)),
        n_resamples=n_boot,
    )

    return ModelResult(
        model=name,
        metrics=metrics,
        roc_auc_ci=auc_ci,
        recall_ci=recall_ci,
        train_seconds=elapsed,
        predictions=(scores >= 0.5).astype(int),
        scores=scores,
    )


def evaluate_dataset(
    dataset: str,
    *,
    models: tuple[str, ...] = ("logreg", "rf", "xgb"),
    n_boot: int = 500,
    on_progress: Any = None,
) -> DatasetResult:
    """Run the full suite twice - all features, then with quarantined features removed."""
    from penumbra.data import audit as audit_mod

    loader = _loader_for(dataset)
    ds_full = loader(drop_artifacts=False)

    report = audit_mod.run_audit(ds_full)
    quarantined = report.quarantine

    def run(ds: Dataset, tag: str) -> dict[str, ModelResult]:
        out: dict[str, ModelResult] = {}
        for name in models:
            if on_progress:
                on_progress(f"{tag}: {name}")
            out[name] = train_and_score(name, ds, n_boot=n_boot)
        return out

    with_art = run(ds_full, "all features")

    ds_clean = _drop_columns(ds_full, quarantined)
    without_art = run(ds_clean, "artifacts removed")

    return DatasetResult(
        dataset=dataset,
        with_artifacts=with_art,
        without_artifacts=without_art,
        quarantined=quarantined,
    )


def _loader_for(dataset: str) -> Any:
    key = dataset.lower().replace("-", "").replace("_", "")
    if key in {"unsw", "unswnb15"}:
        from penumbra.data.loaders import unsw

        return unsw.load
    if key in {"nslkdd", "nsl", "kdd"}:
        from penumbra.data.loaders import nsl_kdd

        return nsl_kdd.load
    raise ValueError(f"unknown dataset {dataset!r}")


def _drop_columns(ds: Dataset, columns: list[str]) -> Dataset:
    """A copy of the dataset with the given features removed."""
    keep = [c for c in ds.feature_names if c not in columns]
    return Dataset(
        name=f"{ds.name}-quarantined",
        X_train=ds.X_train[keep],
        y_train=ds.y_train,
        fam_train=ds.fam_train,
        X_test=ds.X_test[keep],
        y_test=ds.y_test,
        fam_test=ds.fam_test,
        categorical=[c for c in ds.categorical if c in keep],
        numeric=[c for c in ds.numeric if c in keep],
        suspected_artifacts=[],
        capabilities=ds.capabilities,
    )


# =================================================================================================
# Reporting
# =================================================================================================


def format_result(result: DatasetResult) -> str:
    lines: list[str] = []
    lines.append("=" * 78)
    lines.append(f"  {result.dataset.upper()} - honest evaluation")
    lines.append("=" * 78)

    for tag, pool in (
        ("ALL FEATURES", result.with_artifacts),
        ("ARTIFACTS REMOVED", result.without_artifacts),
    ):
        lines.append("")
        lines.append(f"  {tag}")
        if tag.startswith("ARTIFACTS") and result.quarantined:
            lines.append(f"  removed: {', '.join(result.quarantined)}")
        lines.append("")
        lines.append(
            f"  {'model':<10} {'ROC-AUC (95% CI)':<28} {'recall':>8} {'FPR':>8} "
            f"{'PR-AUC':>8} {'R@1%FPR':>9} {'fit s':>7}"
        )
        lines.append(f"  {'-' * 10} {'-' * 28} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 9} {'-' * 7}")
        for name, r in pool.items():
            m = r.metrics
            r1 = m.recall_at_fpr.get("0.01", float("nan"))
            lines.append(
                f"  {name:<10} {str(r.roc_auc_ci):<28} {m.tpr:>8.4f} {m.fpr:>8.4f} "
                f"{m.pr_auc:>8.4f} {r1:>9.4f} {r.train_seconds:>7.1f}"
            )

    champ_a = result.champion(artifacts=True)
    champ_c = result.champion(artifacts=False)
    delta = champ_a.metrics.roc_auc - champ_c.metrics.roc_auc

    lines.append("")
    lines.append("-" * 78)
    lines.append("  THE NUMBER WE STAND BEHIND")
    lines.append("")
    lines.append(
        f"  With suspected testbed artifacts : ROC-AUC {champ_a.metrics.roc_auc:.4f}  ({champ_a.model})"
    )
    lines.append(
        f"  With them removed                : ROC-AUC {champ_c.metrics.roc_auc:.4f}  ({champ_c.model})"
    )
    lines.append(f"  Attributable to artifacts        : {delta:+.4f}")
    lines.append("")
    lines.append(
        f"  PR-AUC is reported at prevalence {champ_c.metrics.prevalence:.4f} "
        f"(no-skill baseline {champ_c.metrics.pr_auc_no_skill:.4f}). This test set is not a real "
        "network."
    )

    lines.append("")
    lines.append("-" * 78)
    lines.append("  WHAT THIS MEANS OPERATIONALLY  (modelled, not measured)")
    lines.append("")
    for est in prev_mod.sweep(champ_c.metrics.tpr, champ_c.metrics.fpr):
        lines.append(est.summary())
    lines.append("")
    lines.append(prev_mod.base_rate_narrative(champ_c.metrics.tpr, champ_c.metrics.fpr))

    return "\n".join(lines)


def compare_models(result: DatasetResult, y_true: np.ndarray) -> str:
    """Pairwise McNemar between the models, on the artifact-free run."""
    pool = result.without_artifacts
    names = list(pool)
    lines = ["", "-" * 78, "  PAIRWISE SIGNIFICANCE (artifacts removed)", ""]
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            res = mcnemar(y_true, pool[a].predictions, pool[b].predictions)
            lines.append(res.summary(a, b))
            lines.append("")
    return "\n".join(lines)


def save(result: DatasetResult, *, name: str | None = None) -> Path:
    settings().ensure_dirs()
    out = settings().report_dir / f"eval_{name or result.dataset}.json"
    out.write_text(json.dumps(result.to_dict(), indent=2, default=str), encoding="utf-8")
    return out


def per_family_report(ds: Dataset, model_name: str = "xgb") -> str:
    """Per-family recall from a multiclass model, with rare classes flagged.

    Reported separately from the binary numbers because it is where the honest bad news lives:
    Worms has 44 test rows, so its recall is an interval, not a decimal.
    """
    seed_everything()
    families = sorted(ds.fam_train.unique())
    model, encoder = supervised.fit_multiclass(model_name, ds, balanced=True)
    pred = supervised.predict_families(model, encoder, ds.X_test)

    rows = per_class_metrics(ds.fam_test.to_numpy(), pred, families)
    lines = [
        "",
        "-" * 78,
        f"  PER-FAMILY ({model_name}, multiclass)",
        "",
        f"  {'family':<18} {'support':>8} {'precision':>10} {'recall':>10} {'F1':>8}",
        f"  {'-' * 18} {'-' * 8} {'-' * 10} {'-' * 10} {'-' * 8}",
    ]
    for r in rows:
        flag = "  <- too few rows for a point estimate" if r.below_floor else ""
        lines.append(
            f"  {r.label:<18} {r.support:>8,} {r.precision:>10.4f} {r.recall:>10.4f} {r.f1:>8.4f}{flag}"
        )
    lines.append("")
    lines.append(f"  macro-F1 {macro_f1(rows):.4f}   (unweighted: rare families count equally)")
    return "\n".join(lines)
