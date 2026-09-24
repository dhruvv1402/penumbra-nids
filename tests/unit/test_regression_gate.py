"""The ML regression gate fails on regressions AND on unreviewed improvements."""

from __future__ import annotations

import json

import numpy as np

from penumbra.eval import regression

BASE = {"metrics": {"roc_auc": 0.96, "recall_at_1pct": 0.46, "unseen_recall_at_1pct": 0.05}}


def current(**over: float) -> dict[str, float]:
    return {**BASE["metrics"], "realised_fpr": 0.01, "n_test": 100, "n_unseen": 10, **over}


def test_within_tolerance_passes() -> None:
    assert regression.compare(current(roc_auc=0.958), BASE).passed


def test_regression_fails_and_names_the_metric() -> None:
    outcome = regression.compare(current(unseen_recall_at_1pct=0.01), BASE)
    assert not outcome.passed
    assert outcome.regressions and "unseen_recall_at_1pct" in outcome.regressions[0]


def test_unreviewed_improvement_fails_softly() -> None:
    outcome = regression.compare(current(recall_at_1pct=0.60), BASE)
    assert not outcome.passed
    assert outcome.improvements and not outcome.regressions


def test_baseline_tolerance_overrides_default() -> None:
    loose = {**BASE, "tolerance": {"roc_auc": 0.1}}
    assert regression.compare(current(roc_auc=0.90), loose).passed


def test_measure_hits_the_budget_exactly_despite_ties() -> None:
    # 1,000 benign rows all tied at 0.0: a quantile threshold would flag none or all of them.
    y = np.array([0] * 1000 + [1] * 100)
    scores = np.concatenate([np.zeros(1000), np.linspace(0.5, 1, 100)])
    unseen = np.array([False] * 1050 + [True] * 50)
    m = regression.measure(scores, y, unseen)
    assert m["realised_fpr"] == 0.01
    assert m["recall_at_1pct"] == 1.0


def test_write_then_compare_round_trip(tmp_path) -> None:
    path = tmp_path / "b.json"
    regression.write_baseline(path, current(), note="t")
    assert regression.compare(current(), json.loads(path.read_text())).passed
