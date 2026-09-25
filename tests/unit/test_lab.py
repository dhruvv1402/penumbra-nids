"""Lab scoring: ground truth by address pair, and a re-baselined novelty head that separates."""

from __future__ import annotations

import numpy as np
import pandas as pd

from penumbra.eval import lab


def test_attack_mask_matches_both_directions_only() -> None:
    meta = pd.DataFrame({"Src IP": ["a", "b", "a", "c"], "Dst IP": ["b", "a", "c", "b"]})
    assert lab.attack_mask(meta, "a", "b").tolist() == [True, True, False, False]


def test_rebaseline_separates_what_the_stock_head_cannot() -> None:
    rng = np.random.default_rng(0)
    benign = pd.DataFrame(
        {"dur": rng.normal(5, 1, 400), "sbytes": rng.normal(2000, 200, 400), "proto": "tcp"}
    )
    attack = pd.DataFrame(
        {"dur": rng.normal(0.01, 0.005, 200), "sbytes": rng.normal(60, 5, 200), "proto": "tcp"}
    )
    X = pd.concat([benign, attack], ignore_index=True)
    is_attack = np.r_[np.zeros(400, bool), np.ones(200, bool)]
    stock = rng.random(len(X))  # an uninformative head
    out = lab.rebaseline(X, is_attack, stock)
    assert out["stock_unsw_novelty"]["roc_auc"] < 0.65
    assert out["local_rebaselined_novelty"]["roc_auc"] > 0.95
    assert out["fit_benign_flows"] + out["heldout_benign_flows"] == 400
