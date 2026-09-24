"""ADR-0004: KDDTrain+ and KDDTest+ are never concatenated and re-split.

Re-splitting would destroy the 17 attack types that appear only in the test file - the only
naturally occurring zero-day holdout in this project. Needs the dataset, so CI deselects it.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.needs_data


def test_loader_keeps_the_published_split() -> None:
    from penumbra.data import schema
    from penumbra.data.loaders import nsl_kdd

    try:
        ds, fine_train, fine_test = nsl_kdd.load_with_fine_labels()
    except FileNotFoundError:
        pytest.skip("NSL-KDD not fetched")
    assert (len(ds.X_train), len(ds.X_test)) == (125_973, 22_544)

    unseen = set(schema.NSLKDD_UNSEEN_IN_TEST)
    train_types = set(fine_train.astype(str).str.lower())
    test_types = set(fine_test.astype(str).str.lower())
    assert not unseen & train_types, "an unseen-in-train attack type leaked into training"
    assert unseen <= test_types
    assert int(nsl_kdd.unseen_mask(fine_test).sum()) == 3_750
