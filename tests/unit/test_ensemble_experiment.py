"""The E9 protocol end to end on tiny data: splits, selection, checkpoints, the report's shape.

The numbers mean nothing at this size; what is tested is that selection never sees test, that S
is the detector's own calibration slice, and that a resumed run reproduces a fresh one.
"""

from __future__ import annotations

import numpy as np
import pytest

from penumbra.eval import ensemble as ens
from penumbra.models.detector import CALIBRATION_FRACTION, calibration_split
from tests.unit.test_registry import tiny_dataset

TINY = ens.Profile(
    "tiny",
    k_small=30,
    k_large=20,
    cv=2,
    cv_large=2,
    n_boot=20,
    n_boot_large=20,
    exact_svc_rows=300,
    large_full_grid=False,
)


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    ck = tmp_path_factory.mktemp("ck")
    ds = tiny_dataset(n=1500)
    return ds, ck, ens.run(ds, profile=TINY, checkpoint_dir=ck)


def test_splits_are_disjoint_and_s_is_the_detectors_slice() -> None:
    ds = tiny_dataset(n=1500)
    F, H, S = ens.splits(ds)
    assert not (set(F) & set(H)) and not (set(F) & set(S)) and not (set(H) & set(S))
    assert len(F) + len(H) + len(S) == len(ds.X_train)
    _, s_detector = calibration_split(ds.fam_train, CALIBRATION_FRACTION)
    assert np.array_equal(np.sort(S), np.sort(s_detector))


def test_report_has_every_registered_section(run) -> None:
    _, _, r = run
    assert r["experiment"] == "E9"
    assert set(r["stage_a"]["arms"]) == set(ens.E.ARMS)
    assert {"ensemble", "rf", "xgb", "logreg", "ensemble_P3a"} <= set(r["test"]["models"])
    assert "vs_rf" in r["test"]["models"]["ensemble"]
    assert set(r["test"]["members"]) == set(ens.E.MEMBERS)
    assert {"tree-tree", "tree-svm", "svm-svm"} == set(r["test"]["diversity"]["mean_q"])
    assert r["fingerprint"]["profile"]["name"] == "tiny"
    assert "check_exact_svc" in r and "check_sgd" in r
    assert r["summary"]


def test_selection_is_made_on_the_holdout_not_test(run) -> None:
    _, _, r = run
    sel = r["selection"]["selected"]
    best = max(
        (row for row in r["selection"]["rows"] if row["combiner"] != "hard"),
        key=lambda row: (row["recall"], -row["member_seconds"]),
    )
    assert (sel["config"], sel["combiner"]) == (best["config"], best["combiner"])
    assert sel["combiner"] != "hard"


def test_realised_fpr_is_printed_and_matches_the_budget(run) -> None:
    _, _, r = run
    for row in r["test"]["models"].values():
        assert abs(row["realised_fpr"] - ens.FPR) < 0.02


def test_a_resumed_run_reproduces_the_fresh_one(run) -> None:
    ds, ck, first = run
    again = ens.run(ds, profile=TINY, checkpoint_dir=ck, resume=True)
    a = first["test"]["models"]["ensemble"]["recall_at_1pct"]
    b = again["test"]["models"]["ensemble"]["recall_at_1pct"]
    assert a == b
    assert first["selection"]["selected"] == again["selection"]["selected"]


def test_yules_q() -> None:
    a = np.array([1, 1, 0, 0, 1, 0], dtype=bool)
    assert ens.yules_q(a, a) == 1.0
    assert ens.yules_q(a, ~a) == -1.0
