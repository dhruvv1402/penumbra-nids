"""Flat forests must reproduce scikit-learn exactly, and refuse what they cannot reproduce."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.ensemble import IsolationForest, RandomForestClassifier

from penumbra.models.flat_forest import FlatBinaryForest, FlatIsolationForest, Unsupported


@pytest.fixture(scope="module")
def data() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(3000, 6))
    X[:, 2] = np.round(X[:, 2])  # ties on thresholds are where float handling goes wrong
    y = ((X[:, 0] + X[:, 1] ** 2 + rng.normal(0, 0.5, len(X))) > 1).astype(int)
    return X, y


def test_random_forest_matches_predict_proba(data) -> None:
    X, y = data
    rf = RandomForestClassifier(n_estimators=40, max_depth=12, random_state=0).fit(X[:2000], y[:2000])
    flat = FlatBinaryForest(rf)
    assert np.max(np.abs(flat.p_class1(X[2000:]) - rf.predict_proba(X[2000:])[:, 1])) < 1e-12


def test_unbounded_depth_forest_matches(data) -> None:
    X, y = data
    rf = RandomForestClassifier(n_estimators=10, random_state=1).fit(X[:2000], y[:2000])
    assert np.max(np.abs(FlatBinaryForest(rf).p_class1(X) - rf.predict_proba(X)[:, 1])) < 1e-12


def test_float32_split_semantics(data) -> None:
    # sklearn compares float32(x) <= float64(threshold). A value that is below the threshold in
    # float64 but rounds up past it in float32 must go the way sklearn sends it.
    X, y = data
    rf = RandomForestClassifier(n_estimators=5, max_depth=6, random_state=0).fit(X[:2000], y[:2000])
    thresholds = rf.estimators_[0].tree_.threshold
    t = float(thresholds[thresholds != -2][0])
    probe = np.tile(X[:1], (3, 1))
    probe[:, rf.estimators_[0].tree_.feature[0]] = [np.nextafter(t, -np.inf), t, np.nextafter(t, np.inf)]
    assert np.max(np.abs(FlatBinaryForest(rf).p_class1(probe) - rf.predict_proba(probe)[:, 1])) < 1e-12


def test_isolation_forest_matches_decision_function(data) -> None:
    X, _ = data
    iso = IsolationForest(n_estimators=60, random_state=0).fit(X[:2000])
    flat = FlatIsolationForest(iso)
    assert np.max(np.abs(flat.decision_function(X[2000:]) - iso.decision_function(X[2000:]))) < 1e-12


def test_isolation_forest_with_feature_subsampling(data) -> None:
    # Trees fitted on a feature subset index into that subset; the flat arrays must map back.
    X, _ = data
    iso = IsolationForest(n_estimators=30, max_features=0.5, random_state=3).fit(X[:2000])
    flat = FlatIsolationForest(iso)
    assert np.max(np.abs(flat.decision_function(X) - iso.decision_function(X))) < 1e-12


def test_single_row_matches(data) -> None:
    X, y = data
    rf = RandomForestClassifier(n_estimators=20, random_state=0).fit(X[:2000], y[:2000])
    row = X[2500:2501]
    assert FlatBinaryForest(rf).p_class1(row) == pytest.approx(rf.predict_proba(row)[:, 1], abs=1e-12)


def test_refuses_multiclass_and_non_forests(data) -> None:
    X, _ = data
    y3 = np.arange(len(X)) % 3
    with pytest.raises(Unsupported):
        FlatBinaryForest(RandomForestClassifier(n_estimators=3, random_state=0).fit(X, y3))
    with pytest.raises(Unsupported):
        FlatBinaryForest(IsolationForest(n_estimators=3, random_state=0).fit(X))
    with pytest.raises(Unsupported):
        FlatIsolationForest(RandomForestClassifier(n_estimators=3, random_state=0).fit(X, y3 > 0))


def test_refuses_nan_input(data) -> None:
    X, y = data
    rf = RandomForestClassifier(n_estimators=3, random_state=0).fit(X, y)
    bad = X[:2].copy()
    bad[0, 0] = np.nan
    with pytest.raises(Unsupported, match="NaN"):
        FlatBinaryForest(rf).p_class1(bad)
