"""Preprocessing pipelines.

Two of them, and keeping them apart is load-bearing:

  supervised_pipeline()   fitted on all training rows
  benign_only_pipeline()  fitted on benign training rows ONLY

The novelty head's entire claim is that it has never seen an attack. If its scaler were fitted over
all training data, the attack distribution would reach it through the feature means and variances -
a quiet, plausible-looking leak that would invalidate every novelty number we report. So the benign
pipeline is a separate object, fitted separately, and `assert_benign_only_fit` exists to prove it.

Everything here is a scikit-learn Pipeline so that resampling and cross-validation compose without
any step being fitted outside a fold.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler

from penumbra.data.loaders.base import Dataset

# `proto` carries ~130 distinct values in UNSW-NB15. One-hot encoding all of them adds a wide, very
# sparse block that swamps the 39 numeric columns - and for the autoencoder it would let categorical
# noise dominate reconstruction error. Rare levels collapse into a single "infrequent" column.
MAX_CATEGORIES = 16


class FiniteSanitiser(BaseEstimator, TransformerMixin):
    """Replace inf/-inf with NaN so the imputer can see them.

    Rate-style features (`rate`, `sload`, `dload`, and CICIDS2017's `Flow Bytes/s`) divide by flow
    duration, and zero-duration flows exist. Left alone these arrive at the scaler as inf and
    silently poison the column mean for every row.
    """

    def fit(self, X: pd.DataFrame, y: object = None) -> FiniteSanitiser:  # noqa: ARG002
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.replace([np.inf, -np.inf], np.nan)

    def get_feature_names_out(self, input_features: list[str] | None = None) -> np.ndarray:
        return np.asarray(input_features if input_features is not None else [])


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """Encode a categorical column as the training frequency of each level.

    Used for the novelty head instead of one-hot. A 130-column sparse block would dominate an
    autoencoder's reconstruction error with categorical noise, while frequency encoding keeps the
    signal that matters for anomaly detection - "this protocol is rare" - in one dense column.

    Unseen levels map to 0.0, which is correct: a level never observed in benign training traffic
    has a benign frequency of zero, and that is exactly the kind of thing the novelty head exists
    to notice.
    """

    def __init__(self) -> None:
        self.frequencies_: dict[str, dict[object, float]] = {}
        self.columns_: list[str] = []

    def fit(self, X: pd.DataFrame, y: object = None) -> FrequencyEncoder:  # noqa: ARG002
        self.columns_ = list(X.columns)
        self.frequencies_ = {col: (X[col].value_counts(normalize=True).to_dict()) for col in self.columns_}
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        out = np.column_stack(
            [X[col].map(self.frequencies_[col]).fillna(0.0).to_numpy(dtype=float) for col in self.columns_]
        )
        return out

    def get_feature_names_out(self, input_features: list[str] | None = None) -> np.ndarray:  # noqa: ARG002
        return np.asarray([f"{c}__freq" for c in self.columns_])


def _numeric_branch(*, scale: bool, robust: bool = False) -> Pipeline:
    steps: list[tuple[str, object]] = [
        ("finite", FiniteSanitiser()),
        # Median rather than mean: these distributions are heavy-tailed enough that a mean
        # imputation would insert values no real flow would produce.
        ("impute", SimpleImputer(strategy="median")),
    ]
    if scale:
        # RobustScaler for the novelty head - flow features have extreme outliers, and a
        # StandardScaler fitted on them compresses the bulk of the data into a narrow band where
        # reconstruction error stops discriminating.
        steps.append(("scale", RobustScaler() if robust else StandardScaler()))
    return Pipeline(steps)


def supervised_pipeline(ds: Dataset, *, scale: bool = False) -> ColumnTransformer:
    """Preprocessing for the known-threat head.

    `scale=True` for linear models. Tree ensembles are invariant to monotone rescaling, so scaling
    them costs time and buys nothing.
    """
    return ColumnTransformer(
        transformers=[
            ("num", _numeric_branch(scale=scale), ds.numeric),
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    max_categories=MAX_CATEGORIES,
                    min_frequency=0.001,
                    sparse_output=False,
                ),
                ds.categorical,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def benign_only_pipeline(ds: Dataset) -> ColumnTransformer:
    """Preprocessing for the novelty head. FIT THIS ON BENIGN ROWS ONLY.

    Frequency-encodes categoricals and robust-scales numerics. See `assert_benign_only_fit` for the
    check that the caller actually honoured the contract.
    """
    return ColumnTransformer(
        transformers=[
            ("num", _numeric_branch(scale=True, robust=True), ds.numeric),
            ("cat", FrequencyEncoder(), ds.categorical),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def assert_benign_only_fit(ds: Dataset, fitted_on: pd.DataFrame) -> None:
    """Fail loudly if a novelty pipeline was fitted on anything but benign rows.

    Compares by index against the dataset's own benign mask. Called from the novelty trainer and
    asserted again in the test suite, because this is the single easiest way to invalidate the
    novelty result while everything still appears to work.
    """
    benign_index = set(ds.X_train.index[ds.y_train == 0])
    got = set(fitted_on.index)
    contamination = got - benign_index
    if contamination:
        raise ValueError(
            f"novelty pipeline was fitted on {len(contamination):,} non-benign rows. "
            "The novelty head's claim is that it has never seen an attack; fitting its scaler on "
            "attack rows leaks the attack distribution through the feature statistics."
        )


def feature_names(ct: ColumnTransformer) -> list[str]:
    """Post-transform column names, for SHAP and for the plain-English attribution renderer."""
    try:
        return [str(n) for n in ct.get_feature_names_out()]
    except Exception:  # noqa: BLE001 - name recovery is best-effort, never fatal
        return []
