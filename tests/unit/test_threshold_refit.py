"""E8's machinery, on data small enough for CI. The experiment itself runs on NSL-KDD."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from penumbra.eval import threshold_refit as tr
from penumbra.models.detector import PenumbraDetector
from tests.unit.test_rebaseline import new_network
from tests.unit.test_registry import tiny_dataset


@pytest.fixture(scope="module")
def detector() -> PenumbraDetector:
    return PenumbraDetector(target_fpr=0.05).fit(tiny_dataset(n=900))


def test_thresholds_only_changes_the_gate_and_nothing_else(detector) -> None:
    out = tr.thresholds_only(detector, new_network(600))
    assert out.gate is not detector.gate
    assert out.novelty is detector.novelty
    assert out.conformal is detector.conformal  # A1 as registered: thresholds only


def test_relearn_normal_splits_five_to_three(detector) -> None:
    out = tr.relearn_normal(detector, new_network(800), seed=0)
    assert out.metadata.baseline["fit_rows"] == 500
    assert out.metadata.baseline["calibration_rows"] == 300


def test_evaluate_counts_the_right_rows() -> None:
    scored = pd.DataFrame(
        {"fired": [0, 1, 0, 2, 3, 0], "conformal_abstains": [True, False, False, False, False, False]}
    )
    y = np.array([0, 0, 0, 1, 1, 1])
    unseen = np.array([False, False, False, True, False, True])
    out = tr.evaluate(scored, y, unseen)
    assert out["fpr"]["k"] == 1 and out["fpr"]["n"] == 3
    assert out["recall_unseen17"]["k"] == 1 and out["recall_unseen17"]["n"] == 2
    assert out["recall_seen"]["k"] == 1 and out["recall_seen"]["n"] == 1
    assert out["benign_reach"]["k"] == 2


def _arm(fpr: float, unseen: float) -> dict:
    return {"fpr": {"rate": fpr}, "recall_unseen17": {"rate": unseen}}


def test_summary_checks_the_registered_predictions() -> None:
    report = {
        "A0": _arm(0.10, 0.77),
        "windows": [
            {"n": 3400, "seed": None, "A1": _arm(0.0106, 0.43), "A2": _arm(0.0127, 0.38)},
            *(
                {"n": 500, "seed": s, "A1": _arm(0.01 + 0.004 * s, 0.4), "A2": _arm(0.02, 0.5)}
                for s in range(5)
            ),
            *(
                {"n": 2000, "seed": s, "A1": _arm(0.01 + 0.001 * s, 0.4), "A2": _arm(0.02, 0.5)}
                for s in range(5)
            ),
        ],
        "dirty": [
            {"dose": 0.01, "A2_clean": _arm(0.01, 0.38), "A2_dirty": _arm(0.006, 0.27)},
            {"dose": 0.05, "A2_clean": _arm(0.01, 0.38), "A2_dirty": _arm(0.003, 0.15)},
        ],
    }
    s = tr.summarise(report)
    assert s["P2_H8a_A1_fpr_in_band"]["held"] is True
    assert s["P3_H8b_A1_unseen_loss"]["held"] is True
    assert s["P4_H8c_A2_over_A1_unseen"]["held"] is False
    assert s["P5_H8d_spread"]["held"] is True  # 0.0063 vs 0.0016
    assert s["P6_H8e_dirty"]["held"] is True
