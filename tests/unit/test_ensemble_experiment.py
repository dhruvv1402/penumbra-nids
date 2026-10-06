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


@pytest.mark.parametrize("mode", ["fresh", "no-directory"])
def test_a_run_reads_back_what_it_wrote(tmp_path, mode) -> None:
    # The smoke run on NSL-KDD found it: with --fresh the store ignored its own writes, so P3b was
    # skipped ("P2 did not run") and the PCA counts H9c is judged on were empty.
    ds = tiny_dataset(n=1200)
    ck = None if mode == "no-directory" else tmp_path
    r = ens.run(ds, profile=TINY, checkpoint_dir=ck, resume=False)
    arms = r["stage_a"]["arms"]
    assert "P3b" in arms
    assert arms["P2"]["transform"]["pca_components"] >= 1
    assert arms["P3a"]["transform"]["pca_components"] >= 1


def test_smoke_subsample_keeps_groups_and_unseen_aligned() -> None:
    ds = tiny_dataset(n=1500)
    groups = np.arange(len(ds.X_train)) % 3
    unseen = np.arange(len(ds.X_test)) % 2 == 0
    small, g, u = ens.subsample(ds, train=400, test=200, groups=groups, unseen=unseen)
    assert len(small.X_train) == len(g) and len(small.X_test) == len(u)
    assert abs(len(small.X_train) - 400) <= 5


def test_tied_scores_are_shared_not_spent_in_row_order() -> None:
    # The E9 run found it: UNSW's training CSV lists benign rows first, so a tree's tied block at
    # p = 1.0 went entirely to benign under row-order tie-breaking and its recall read 0.0000.
    from penumbra.eval.budget import flags_at_benign_budget

    y = np.r_[np.zeros(2000, int), np.ones(2000, int)]  # sorted by class, benign first
    s = np.r_[np.zeros(1000), np.ones(1000), np.ones(2000)]  # half the benign rows tie with every attack
    budget = int(round(0.01 * 2000))
    assert flags_at_benign_budget(s, y, budget)[y == 1].mean() == 0.0  # the artifact
    flags = ens.at_budget(s, y)
    assert flags[y == 0].sum() == budget  # the budget is still exact
    # Fair share of the tied block: budget / tied benign = 2%, so about 2% of the attacks.
    assert 0.01 < flags[y == 1].mean() < 0.03


def test_untied_scores_match_the_row_order_rule() -> None:
    from penumbra.eval.budget import flags_at_benign_budget

    rng = np.random.default_rng(0)
    y = (rng.random(3000) < 0.5).astype(int)
    s = rng.random(3000) + y
    assert np.array_equal(
        ens.at_budget(s, y), flags_at_benign_budget(s, y, int(round(0.01 * (y == 0).sum())))
    )


def test_yules_q() -> None:
    a = np.array([1, 1, 0, 0, 1, 0], dtype=bool)
    assert ens.yules_q(a, a) == 1.0
    assert ens.yules_q(a, ~a) == -1.0
