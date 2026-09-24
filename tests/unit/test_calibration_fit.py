"""calibration.calibrate actually runs on the installed scikit-learn.

It used cv="prefit", removed in scikit-learn 1.9, and nothing called it until `penumbra calibrate`,
so it was broken without anyone knowing. This test is the call that would have noticed.
"""

from __future__ import annotations

import numpy as np

from penumbra.models import calibration, supervised
from tests.unit.test_registry import tiny_dataset


def test_isotonic_and_platt_fit_and_return_probabilities() -> None:
    ds = tiny_dataset(n=900)
    X_fit, X_cal, y_fit, y_cal = calibration.split_for_calibration(ds.X_train, ds.y_train.to_numpy())
    est = supervised.build("rf", ds, n_classes=2, balanced=True).fit(X_fit, y_fit)
    reports = calibration.compare_methods(est, X_cal, y_cal, ds.X_test, ds.y_test.to_numpy())
    assert set(reports) == {"uncalibrated", "isotonic", "sigmoid"}
    for method in ("isotonic", "sigmoid"):
        p = calibration.calibrate(est, X_cal, y_cal, method=method).predict_proba(ds.X_test)[:, 1]
        assert np.all((p >= 0) & (p <= 1))
        assert 0 <= reports[method].brier <= 1
