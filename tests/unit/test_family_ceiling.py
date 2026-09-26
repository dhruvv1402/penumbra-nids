"""The family ceiling: identical inputs cannot get different answers."""

from __future__ import annotations

import pandas as pd

from penumbra.data.loaders.base import Dataset
from penumbra.eval import family_ceiling


def _ds(test_rows: list[tuple[int, str]], train_rows: list[tuple[int, str]]) -> Dataset:
    def frame(rows):
        X = pd.DataFrame({"f": [r[0] for r in rows], "g": [0] * len(rows)})
        fam = pd.Series([r[1] for r in rows])
        y = (fam != "Normal").astype(int)
        return X, y, fam

    Xtr, ytr, ftr = frame(train_rows)
    Xte, yte, fte = frame(test_rows)
    return Dataset("t", Xtr, ytr, ftr, Xte, yte, fte, categorical=[], numeric=["f", "g"])


def test_identical_vectors_cap_recall() -> None:
    # Vector 1 is labelled A twice and B once: the best labelling says A, so B's rows on it are lost.
    test = [(1, "A"), (1, "A"), (1, "B"), (2, "B"), (3, "Normal")]
    train = [(1, "B"), (1, "B"), (2, "B")]
    r = family_ceiling.run(_ds(test, train))
    assert r["attack_test_rows"] == 4
    assert r["overall_ceiling"] == 0.75
    assert r["families"]["A"]["ceiling"] == 1.0
    assert r["families"]["B"]["ceiling"] == 0.5
    assert r["families"]["B"]["shared"] == 0.5
    # A's vector exists in training, where its majority is B: the labels disagree across splits.
    assert r["families"]["A"]["train_twin"] == 1.0
    assert r["families"]["A"]["train_majority_agrees"] == 0.0


def test_unique_vectors_have_no_ceiling() -> None:
    r = family_ceiling.run(_ds([(1, "A"), (2, "B")], [(9, "A")]))
    assert r["overall_ceiling"] == 1.0
    assert r["families"]["A"]["train_twin"] == 0.0
