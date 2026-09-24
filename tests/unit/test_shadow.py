"""Shadow scoring: volume, agreement and the drop/add breakdown, without needing labels."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from penumbra.eval import shadow


def scored(fired: list[int], p: float = 0.5) -> pd.DataFrame:
    return pd.DataFrame({"fired": fired, "p_attack": [p] * len(fired)})


def test_identical_models_agree_perfectly() -> None:
    s = scored([0, 1, 2, 3, 0, 1])
    r = shadow.compare(s, s)
    assert r["agreement"] == 1.0 and r["cohen_kappa"] == 1.0
    assert r["lost"]["n"] == r["gained"]["n"] == 0


def test_drops_and_adds_are_attributed_to_the_head_that_fired() -> None:
    champ = scored([1, 2, 0, 0])
    chall = scored([0, 2, 3, 0])
    r = shadow.compare(champ, chall)
    assert r["lost"] == {"n": 1, "champion_head": {"supervised": 1, "novelty_only": 0, "both": 0}}
    assert r["gained"] == {"n": 1, "challenger_head": {"supervised": 0, "novelty_only": 0, "both": 1}}


def test_labels_are_optional_and_only_add_a_column() -> None:
    champ, chall = scored([1, 0]), scored([0, 1])
    assert "labelled" not in shadow.compare(champ, chall)
    lab = shadow.compare(champ, chall, np.array([1, 1]))["labelled"]
    assert lab["lost_that_were_attacks"] == 1 and lab["gained_that_were_attacks"] == 1


def test_kappa_discounts_chance_agreement() -> None:
    # Both alert on almost nothing: raw agreement is high, kappa should not be.
    rng = np.random.default_rng(0)
    a = (rng.random(10_000) < 0.02).astype(int)
    b = (rng.random(10_000) < 0.02).astype(int)
    r = shadow.compare(scored(list(a)), scored(list(b)))
    assert r["agreement"] > 0.95
    assert r["cohen_kappa"] == pytest.approx(0.0, abs=0.05)
