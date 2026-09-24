"""Promoted verdicts -> augmented training set -> gated challenger, on a synthetic dataset."""

from __future__ import annotations

import numpy as np

from penumbra.eval import canary, retraining
from penumbra.models.detector import PenumbraDetector
from tests.unit.test_registry import tiny_dataset


def row(i: int, label: int, family: str | None = None, **features) -> dict:
    base = {"a": float(i), "b": 0.5, "c": 1.0, "proto": "tcp"}
    base.update(features)
    return {"features": base, "label": label, "family": family, "approver": "senior2"}


def test_augment_appends_labels_and_families() -> None:
    ds = tiny_dataset()
    rows = [row(1, 1, "dos"), row(2, 0), row(3, 1, None)]
    augmented, summary = retraining.augment(ds, rows)
    assert len(augmented.X_train) == len(ds.X_train) + 3
    assert list(augmented.y_train.tail(3)) == [1, 0, 1]
    assert list(augmented.fam_train.tail(3)) == ["dos", "normal", retraining.CONFIRMED_UNNAMED]
    assert summary.attack == 2 and summary.benign == 1 and summary.approvers == ["senior2"]


def test_rows_from_another_dataset_are_dropped_not_imputed() -> None:
    ds = tiny_dataset()
    foreign = {"features": {"sttl": 31.0, "sbytes": 100.0}, "label": 0, "family": None, "approver": "x"}
    augmented, summary = retraining.augment(ds, [foreign, row(1, 1, "dos")])
    assert summary.rows_offered == 2 and summary.rows_usable == 1
    assert len(augmented.X_train) == len(ds.X_train) + 1


def test_non_finite_features_become_nan_for_the_imputer() -> None:
    ds = tiny_dataset()
    frame, _ = retraining.feedback_frame([row(1, 0, c=None)], ds)
    assert np.isnan(frame.loc[0, "c"])


def test_benign_family_uses_the_datasets_own_spelling() -> None:
    ds = tiny_dataset()
    ds.fam_train = ds.fam_train.replace("normal", "Normal")
    augmented, _ = retraining.augment(ds, [row(1, 0)])
    assert augmented.fam_train.iloc[-1] == "Normal"


def test_challenge_returns_a_gated_challenger() -> None:
    ds = tiny_dataset()
    champion = PenumbraDetector(target_fpr=0.05).fit(ds)
    augmented, _ = retraining.augment(ds, [row(i, 1, "dos", a=4.0 + i / 10) for i in range(20)])
    challenger, result = retraining.challenge(champion, augmented, ds.X_test, ds.y_test, ds.fam_test)
    assert isinstance(result, canary.GateResult)
    assert challenger is not champion and challenger.target_fpr == champion.target_fpr
