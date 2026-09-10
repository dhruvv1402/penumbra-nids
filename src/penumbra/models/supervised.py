"""The known-threat head.

Three models, and the first one is not decoration: LogisticRegression is the floor every other
claim is measured against. A gradient-boosted ensemble that cannot beat a linear model on this data
is a finding, not a detail.

There is no transformer here. Gradient-boosted trees outperform deep architectures on tabular data
of this size and shape (Grinsztajn et al., NeurIPS 2022), and on 8 CPU cores without a GPU we
cannot afford the hyperparameter search a transformer needs to be competitive - so we would be
shipping an undertuned model that loses to XGBoost and demonstrates nothing. See ADR-0002.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from penumbra.data.loaders.base import Dataset
from penumbra.features.preprocess import supervised_pipeline
from penumbra.seeds import SEED


@dataclass(frozen=True)
class ModelSpec:
    name: str
    kind: str  # "linear" | "tree"
    description: str


SPECS: dict[str, ModelSpec] = {
    "logreg": ModelSpec(
        "logreg",
        "linear",
        "The floor. Every other model is measured against this, and a complex model that does not "
        "clear it has not earned its complexity.",
    ),
    "rf": ModelSpec(
        "rf",
        "tree",
        "Random Forest with balanced subsample weighting - the brief's own suggestion, and TreeSHAP "
        "makes per-alert attribution cheap.",
    ),
    "xgb": ModelSpec(
        "xgb",
        "tree",
        "Gradient boosting. Expected champion on tabular flow features.",
    ),
}


def _estimator(name: str, *, n_classes: int, class_weight: str | None, scale_pos_weight: float | None) -> Any:
    if name == "logreg":
        # n_jobs is a no-op since sklearn 1.8 and removed in 1.10; the lbfgs solver is
        # single-threaded regardless.
        return LogisticRegression(
            max_iter=2000,
            class_weight=class_weight,
            random_state=SEED,
        )
    if name == "rf":
        return RandomForestClassifier(
            n_estimators=300,
            # Unbounded depth on 175k rows memorises; 24 keeps it honest without underfitting.
            max_depth=24,
            min_samples_leaf=2,
            n_jobs=-1,
            # balanced_subsample recomputes weights per bootstrap rather than once globally, which
            # matters when a class has 130 rows.
            class_weight="balanced_subsample" if class_weight else None,
            random_state=SEED,
        )
    if name == "xgb":
        params: dict[str, Any] = {
            "n_estimators": 400,
            "max_depth": 8,
            "learning_rate": 0.1,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "n_jobs": -1,
            "random_state": SEED,
            "tree_method": "hist",
            "eval_metric": "logloss",
        }
        if n_classes > 2:
            params.update(objective="multi:softprob", num_class=n_classes)
        else:
            params.update(objective="binary:logistic")
            if scale_pos_weight is not None:
                params["scale_pos_weight"] = scale_pos_weight
        return XGBClassifier(**params)
    raise ValueError(f"unknown model {name!r}; choose from {sorted(SPECS)}")


def build(
    name: str,
    ds: Dataset,
    *,
    n_classes: int = 2,
    balanced: bool = True,
) -> Pipeline:
    """Assemble preprocessing + estimator as one Pipeline.

    Returning a Pipeline rather than a bare estimator is deliberate: it means preprocessing is
    fitted inside every cross-validation fold, which is the difference between an honest CV score
    and a leaked one.
    """
    spec = SPECS[name]

    scale_pos_weight = None
    if balanced and n_classes == 2:
        n_pos = int((ds.y_train == 1).sum())
        n_neg = int((ds.y_train == 0).sum())
        scale_pos_weight = (n_neg / n_pos) if n_pos else 1.0

    return Pipeline(
        [
            ("prep", supervised_pipeline(ds, scale=spec.kind == "linear")),
            (
                "clf",
                _estimator(
                    name,
                    n_classes=n_classes,
                    class_weight="balanced" if balanced else None,
                    scale_pos_weight=scale_pos_weight,
                ),
            ),
        ]
    )


def attack_scores(model: Pipeline, X: Any) -> np.ndarray:
    """P(attack) as a 1-D array, for binary and multiclass models alike.

    For a multiclass model the attack score is 1 - P(benign) rather than the max attack-class
    probability. That distinction is the whole point of the LOAFO experiment: a flow the model
    spreads thinly across several attack families is still an attack the SOC should be told about,
    and taking the max would discard exactly that evidence.
    """
    proba = model.predict_proba(X)
    if proba.shape[1] == 2:
        return np.asarray(proba[:, 1], dtype=float)

    classes = list(model.named_steps["clf"].classes_)
    benign_idx = _benign_index(classes)
    if benign_idx is None:
        return np.asarray(proba.max(axis=1), dtype=float)
    return np.asarray(1.0 - proba[:, benign_idx], dtype=float)


def _benign_index(classes: list[Any]) -> int | None:
    for i, c in enumerate(classes):
        if str(c).strip().lower() in {"normal", "benign", "0"}:
            return i
    return None


def fit_multiclass(name: str, ds: Dataset, *, balanced: bool = True) -> tuple[Pipeline, LabelEncoder]:
    """Fit a family classifier, encoding string labels to integers.

    XGBoost requires contiguous integer classes and rejects string labels outright, so the encoder
    travels with the model rather than being reconstructed at predict time - rebuilding it from a
    different label ordering would silently permute every family prediction.
    """
    encoder = LabelEncoder().fit(ds.fam_train)
    model = build(name, ds, n_classes=len(encoder.classes_), balanced=balanced)
    model.fit(ds.X_train, encoder.transform(ds.fam_train))
    return model, encoder


def predict_families(model: Pipeline, encoder: LabelEncoder, X: Any) -> np.ndarray:
    """Predicted family labels as strings."""
    return np.asarray(encoder.inverse_transform(model.predict(X)))
