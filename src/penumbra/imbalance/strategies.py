"""Class-imbalance strategies, each wired so it cannot leak.

Every strategy here is an `imblearn.pipeline.Pipeline`, not a sklearn one. The difference matters:
imblearn's pipeline applies resamplers during `fit` and **skips them during `predict`**, which is
the correct semantics. A sklearn Pipeline would try to resample at prediction time.

Two rules that this module enforces structurally rather than by convention:

  1. **Resampling happens inside the pipeline**, so during cross-validation it is refitted per fold
     and never sees the validation half. Applying SMOTE before the split is the classic leak, and it
     is visible in `ablation.py` as a deliberate demonstration.
  2. **Nothing here touches the test set.** Evaluation is always on the natural distribution.
     Resampling the test set would be measuring a world that does not exist.

A third point that is easy to miss: **resampling decalibrates a model.** After SMOTE the predicted
probabilities no longer correspond to observed frequencies, because the training prior has been
altered. Any strategy used alongside a calibrated probability needs recalibration on a held-out set
afterwards - see models/calibration.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from imblearn.combine import SMOTETomek
from imblearn.ensemble import BalancedRandomForestClassifier, EasyEnsembleClassifier
from imblearn.over_sampling import ADASYN, SMOTE, SMOTENC, BorderlineSMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler

from penumbra.data.loaders.base import Dataset
from penumbra.features.preprocess import supervised_pipeline
from penumbra.models.supervised import _estimator
from penumbra.seeds import SEED


@dataclass(frozen=True)
class Strategy:
    name: str
    description: str
    caveat: str = ""


STRATEGIES: dict[str, Strategy] = {
    "none": Strategy("none", "No handling. The floor for this comparison."),
    "class_weight": Strategy(
        "class_weight",
        "Reweight the loss by inverse class frequency. No synthetic data, no extra training time.",
    ),
    "threshold_moving": Strategy(
        "threshold_moving",
        "Train unweighted, then move the decision threshold. The cheapest option and frequently "
        "competitive - it changes where the boundary is drawn rather than what the model learned.",
        caveat="Requires a calibrated score to be meaningful.",
    ),
    "smote": Strategy(
        "smote",
        "Synthesise minority examples by interpolating between neighbours.",
        caveat="On flow data this invents physically impossible rows: fractional packet counts, "
        "byte totals inconsistent with those counts, interpolated one-hot categoricals.",
    ),
    "smotenc": Strategy(
        "smotenc",
        "SMOTE that treats categorical columns as categorical instead of interpolating them.",
        caveat="Still invents impossible numeric combinations; only the categorical half is fixed.",
    ),
    "borderline_smote": Strategy(
        "borderline_smote",
        "SMOTE restricted to minority points near the decision boundary.",
    ),
    "adasyn": Strategy(
        "adasyn",
        "Like SMOTE, but generates more samples where the minority class is hardest to learn.",
    ),
    "smote_tomek": Strategy(
        "smote_tomek",
        "SMOTE followed by Tomek-link removal to clean the overlap it creates.",
    ),
    "undersample": Strategy(
        "undersample",
        "Randomly discard majority examples.",
        caveat="Throws away real data. On 56,000 benign rows that is a lot of information to "
        "discard in order to fix a ratio.",
    ),
    "balanced_rf": Strategy(
        "balanced_rf",
        "Random Forest that balances each bootstrap sample rather than the whole dataset.",
    ),
    "easy_ensemble": Strategy(
        "easy_ensemble",
        "Ensemble of learners each trained on a balanced undersample - uses the majority data that "
        "plain undersampling would discard.",
    ),
}


def _categorical_indices(ds: Dataset) -> list[int]:
    """Positions of categorical columns AFTER preprocessing.

    One-hot encoding means SMOTE-NC needs the post-transform indices, which depend on the encoder's
    fitted categories. Computed by fitting the preprocessor once.
    """
    prep = supervised_pipeline(ds, scale=False)
    prep.fit(ds.X_train)
    names = [str(n) for n in prep.get_feature_names_out()]
    return [i for i, n in enumerate(names) if any(n.startswith(f"{c}_") for c in ds.categorical)]


def build(name: str, ds: Dataset, *, model: str = "rf") -> ImbPipeline:
    """Assemble preprocessing + resampling + estimator as one leak-proof pipeline."""
    if name not in STRATEGIES:
        raise ValueError(f"unknown strategy {name!r}; choose from {sorted(STRATEGIES)}")

    prep = supervised_pipeline(ds, scale=model == "logreg")
    steps: list[tuple[str, Any]] = [("prep", prep)]

    # These two rebalance internally; adding a resampler on top would double-count.
    if name == "balanced_rf":
        steps.append(
            (
                "clf",
                BalancedRandomForestClassifier(
                    n_estimators=300,
                    max_depth=24,
                    n_jobs=-1,
                    random_state=SEED,
                    sampling_strategy="all",
                    replacement=True,
                    bootstrap=False,
                ),
            )
        )
        return ImbPipeline(steps)

    if name == "easy_ensemble":
        steps.append(("clf", EasyEnsembleClassifier(n_estimators=10, n_jobs=-1, random_state=SEED)))
        return ImbPipeline(steps)

    sampler = _sampler(name, ds)
    if sampler is not None:
        steps.append(("resample", sampler))

    # class_weight is the only strategy that changes the estimator itself.
    weighted = name == "class_weight"
    n_pos = int((ds.y_train == 1).sum())
    n_neg = int((ds.y_train == 0).sum())
    steps.append(
        (
            "clf",
            _estimator(
                model,
                n_classes=2,
                class_weight="balanced" if weighted else None,
                scale_pos_weight=(n_neg / n_pos) if (weighted and n_pos) else None,
            ),
        )
    )
    return ImbPipeline(steps)


def _sampler(name: str, ds: Dataset) -> Any:
    k = {"k_neighbors": 5, "random_state": SEED}
    if name in {"none", "class_weight", "threshold_moving"}:
        return None
    if name == "smote":
        return SMOTE(**k)
    if name == "smotenc":
        idx = _categorical_indices(ds)
        # With no categorical columns SMOTE-NC is undefined; fall back rather than crash.
        return SMOTENC(categorical_features=idx, **k) if idx else SMOTE(**k)
    if name == "borderline_smote":
        return BorderlineSMOTE(**k)
    if name == "adasyn":
        return ADASYN(n_neighbors=5, random_state=SEED)
    if name == "smote_tomek":
        return SMOTETomek(random_state=SEED)
    if name == "undersample":
        return RandomUnderSampler(random_state=SEED)
    raise ValueError(f"unhandled strategy {name!r}")


def physically_impossible_example(ds: Dataset, *, n: int = 1) -> list[dict[str, Any]]:
    """Generate SMOTE rows and return ones that could not exist on a wire.

    The argument against resampling network flows, made concrete rather than asserted. SMOTE
    interpolates between neighbours, so it produces fractional packet counts and byte totals
    inconsistent with them - a flow with 3.7 packets is not a rare flow, it is not a flow.
    """
    prep = supervised_pipeline(ds, scale=False)
    X = prep.fit_transform(ds.X_train)
    names = [str(v) for v in prep.get_feature_names_out()]

    Xr, _ = SMOTE(random_state=SEED).fit_resample(X, ds.y_train)
    synthetic = Xr[len(X) :]

    integral = [
        i
        for i, nm in enumerate(names)
        if any(tok in nm for tok in ("pkts", "packets", "loss", "count", "ct_"))
    ]

    out: list[dict[str, Any]] = []
    for row in synthetic:
        bad = {names[i]: float(row[i]) for i in integral if not float(row[i]).is_integer()}
        if bad:
            out.append(dict(list(bad.items())[:6]))
        if len(out) >= n:
            break
    return out


def natural_distribution_check(y_eval: np.ndarray, y_train_original: np.ndarray) -> None:
    """Fail if evaluation is happening on a resampled distribution.

    Guards the rule that matters most here: resampling belongs to training only. A test set balanced
    to 50/50 reports a world that does not exist and inflates every precision-family number.
    """
    eval_prev = float(np.mean(y_eval))
    if abs(eval_prev - 0.5) < 0.01 and abs(float(np.mean(y_train_original)) - 0.5) > 0.05:
        raise ValueError(
            f"evaluation set is balanced (prevalence {eval_prev:.3f}) while the natural training "
            "prevalence is not. Resampling must never touch the evaluation data."
        )
