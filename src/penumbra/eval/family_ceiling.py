"""How well can ANY classifier name attack families on these features? Measured from duplicates.

Two rows with identical features must get the same prediction from any function of those features.
So wherever the same feature vector carries different family labels, some of those labels are
unreachable, whatever the model. Grouping rows by their exact feature vector gives, per family:

  shared      share of its rows whose vector also carries another family's label
  ceiling     its recall if every vector were given its majority label - the labelling that
              maximises overall accuracy, and therefore the best any single classifier can do
              without trading other families away
  train_twin  share of its TEST rows whose exact vector appears in TRAINING, and among those, how
              often the training majority is this family. A model trained on the training labels
              follows the training majority; where that disagrees with the test labels, the gap is
              in the labels, not the model.

The overall ceiling is the accuracy of the majority labelling across all attack rows.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from penumbra.data.loaders.base import Dataset


def _keys(X: pd.DataFrame) -> pd.Series:
    return pd.Series(pd.util.hash_pandas_object(X, index=False).to_numpy(), index=X.index)


def run(ds: Dataset, *, benign_label: str = "Normal") -> dict[str, Any]:
    cols = list(ds.X_test.columns)
    test = pd.DataFrame({"key": _keys(ds.X_test[cols]).to_numpy(), "fam": ds.fam_test.astype(str).to_numpy()})
    train = pd.DataFrame(
        {"key": _keys(ds.X_train[cols]).to_numpy(), "fam": ds.fam_train.astype(str).to_numpy()}
    )
    attacks = test[test.fam != benign_label]

    groups = attacks.groupby(["key", "fam"]).size().unstack(fill_value=0)
    majority = groups.idxmax(axis=1)
    mixed = groups.gt(0).sum(axis=1) > 1
    train_groups = train.groupby(["key", "fam"]).size().unstack(fill_value=0)
    train_majority = train_groups.idxmax(axis=1)

    families: dict[str, Any] = {}
    for fam in groups.columns:
        rows = attacks[attacks.fam == fam]
        in_train = rows["key"].isin(train_majority.index)
        agree = rows.loc[in_train, "key"].map(train_majority) == fam
        families[fam] = {
            "test_rows": int(len(rows)),
            "shared": float(groups.loc[mixed, fam].sum() / groups[fam].sum()),
            "ceiling": float(groups.loc[majority == fam, fam].sum() / groups[fam].sum()),
            "train_twin": float(in_train.mean()) if len(rows) else 0.0,
            "train_majority_agrees": float(agree.mean()) if in_train.any() else float("nan"),
        }
    return {
        "dataset": ds.name,
        "attack_test_rows": int(len(attacks)),
        "distinct_vectors": int(len(groups)),
        "overall_ceiling": float(groups.max(axis=1).sum() / len(attacks)) if len(attacks) else float("nan"),
        "families": families,
    }
