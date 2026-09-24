"""Two invariants whose violation produces a better-looking number and no error.

1. The novelty head's pipeline is fitted on benign rows only (ADR-0002). If attack rows reach it,
   the "never seen an attack" claim is false and nothing crashes.
2. A resampler is only ever fitted on the training fold (MODEL_CARD §6). SMOTE before the split
   inflated minority F1 from 0.12 to 0.999 in our own measurement (EVALUATION §5).

Property-based where the input space matters: hypothesis varies the data, the attack rate and the
seed, and each property must hold for every draw.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedKFold, cross_validate

from penumbra.data.loaders.base import Dataset
from penumbra.features import preprocess
from penumbra.imbalance import strategies
from penumbra.models import detector as detector_module
from penumbra.models.detector import PenumbraDetector


def dataset(n: int, attack_rate: float, seed: int) -> Dataset:
    rng = np.random.default_rng(seed)

    def split(m: int) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
        y = (rng.random(m) < attack_rate).astype(int)
        y[:2], y[2:4] = 0, 1  # both classes present, whatever the draw
        X = pd.DataFrame(
            {
                # Unique per row, and the first numeric column after preprocessing: lets a spy tell
                # exactly which rows reached it.
                "row_id": np.arange(m, dtype=float) + seed * 1e6,
                "b": rng.normal(0, 1, m) + 2 * y,
                "proto": rng.choice(["tcp", "udp"], m),
            }
        )
        return X, pd.Series(y), pd.Series(np.where(y == 1, "dos", "normal"))

    Xtr, ytr, ftr = split(n)
    Xte, yte, fte = split(max(n // 3, 20))
    return Dataset("prop", Xtr, ytr, ftr, Xte, yte, fte, categorical=["proto"], numeric=["row_id", "b"])


@settings(max_examples=6, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    n=st.integers(min_value=120, max_value=300),
    attack_rate=st.floats(min_value=0.1, max_value=0.6),
    seed=st.integers(min_value=0, max_value=10_000),
)
def test_novelty_pipeline_never_fits_on_an_attack_row(n: int, attack_rate: float, seed: int) -> None:
    ds = dataset(n, attack_rate, seed)
    seen: list[pd.DataFrame] = []
    real = preprocess.benign_only_pipeline

    def spying(d: Dataset):
        pipe = real(d)
        original = pipe.fit_transform

        def fit_transform(X, y=None, **kw):
            seen.append(X)
            return original(X, y, **kw)

        pipe.fit_transform = fit_transform
        return pipe

    mp = pytest.MonkeyPatch()
    mp.setattr(detector_module, "benign_only_pipeline", spying)
    try:
        PenumbraDetector(target_fpr=0.1).fit(ds, fit_family_model=False)
    finally:
        mp.undo()

    assert seen, "the novelty pipeline was never fitted"
    fitted_ids = set(np.concatenate([x["row_id"].to_numpy() for x in seen]))
    attack_ids = set(ds.X_train.loc[ds.y_train == 1, "row_id"])
    assert not fitted_ids & attack_ids


def test_contamination_is_detected_for_any_single_attack_row() -> None:
    ds = dataset(200, 0.3, 1)
    benign = ds.X_train[ds.y_train == 0]
    for i in ds.X_train.index[ds.y_train == 1][:10]:
        with pytest.raises(ValueError, match="non-benign"):
            preprocess.assert_benign_only_fit(ds, pd.concat([benign, ds.X_train.loc[[i]]]))


class SpySMOTE(SMOTE):
    """SMOTE that records which original rows it was fitted on (by the row_id column)."""

    fitted: list[set[float]] = []

    def fit_resample(self, X, y, **kw):
        SpySMOTE.fitted.append({float(v) for v in np.asarray(X)[:, 0]})
        return super().fit_resample(X, y, **kw)


@settings(max_examples=4, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(seed=st.integers(min_value=0, max_value=10_000))
def test_resampler_is_fitted_inside_the_fold(seed: int) -> None:
    ds = dataset(240, 0.2, seed)
    pipe = strategies.build("smote", ds)
    pipe.steps[[n for n, _ in pipe.steps].index("resample")] = (
        "resample",
        SpySMOTE(random_state=0, k_neighbors=3),
    )
    SpySMOTE.fitted = []

    folds = list(StratifiedKFold(n_splits=3, shuffle=True, random_state=seed).split(ds.X_train, ds.y_train))
    cross_validate(pipe, ds.X_train, ds.y_train, cv=folds)

    assert len(SpySMOTE.fitted) == len(folds)
    ids = ds.X_train["row_id"].to_numpy()
    for fitted, (train_idx, test_idx) in zip(SpySMOTE.fitted, folds, strict=True):
        assert fitted <= set(ids[train_idx])
        assert not fitted & set(ids[test_idx]), "a validation row reached the resampler"
