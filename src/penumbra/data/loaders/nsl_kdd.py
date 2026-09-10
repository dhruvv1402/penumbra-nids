"""NSL-KDD loader.

This dataset earns its place for one reason: `KDDTest+` contains 17 attack types that never appear
in `KDDTrain+` (3,750 rows, 16.6% of the test set). That is a genuine unseen-attack holdout built by
the dataset's authors rather than constructed by us, and it is the external-validity half of the
LOAFO experiment.

Traps handled here, each documented in docs/DATA_TRAPS.md:

  * Files are headerless. Without explicit names pandas promotes row 1 to a header, losing a row and
    inventing column names.
  * `difficulty` (column 43) is a LEAK - it encodes how many of 21 classic learners classified the
    row correctly, i.e. a function of the answer. Dropped.
  * `num_outbound_cmds` is identically zero in both splits. Dropped.
  * `su_attempted` is documented as binary but contains a third value, 2. Clamped.
  * Train and test are NEVER concatenated. Merging and re-splitting destroys the unseen-17 holdout,
    which is the entire reason this dataset is here.
"""

from __future__ import annotations

import pandas as pd

from penumbra.config import settings
from penumbra.data import schema
from penumbra.data.loaders.base import Capabilities, Dataset, DatasetNotFetched

_FILES = {
    "train": "KDDTrain+.txt",
    "test": "KDDTest+.txt",
    "train_20pct": "KDDTrain+_20Percent.txt",
    "test_21": "KDDTest-21.txt",
}

_EXPECTED_ROWS = {
    "train": 125_973,
    "test": 22_544,
    "train_20pct": 25_192,
    "test_21": 11_850,
}


def _read_split(split: str) -> pd.DataFrame:
    path = settings().raw_dir / "nslkdd" / _FILES[split]
    if not path.exists():
        raise DatasetNotFetched(str(path), "nslkdd")

    df = pd.read_csv(path, header=None, names=schema.NSLKDD_COLUMNS, low_memory=False)

    if len(df) != _EXPECTED_ROWS[split]:
        raise ValueError(f"NSL-KDD {split}: {len(df):,} rows, expected {_EXPECTED_ROWS[split]:,}")

    df["label"] = df["label"].astype(str).str.strip().str.lower().str.rstrip(".")
    return df


def _prepare(df: pd.DataFrame, *, drop_artifacts: bool) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    fam = df["label"].map(schema.nslkdd_category)
    y = (fam != "normal").astype(int)

    drop = [*schema.NSLKDD_DROP, "label"]
    X = df.drop(columns=drop, errors="ignore")

    # `su_attempted` is documented {0,1} but ships a 2. Left alone it becomes a spurious ordinal
    # level that the model will happily use.
    for col, (lo, hi) in schema.NSLKDD_CLAMP.items():
        if col in X.columns:
            X[col] = X[col].clip(lower=lo, upper=hi)

    if drop_artifacts:
        X = X.drop(columns=_SUSPECTED_ARTIFACTS, errors="ignore")

    return X.reset_index(drop=True), y.reset_index(drop=True), fam.reset_index(drop=True)


# NSL-KDD has no documented testbed artifact equivalent to UNSW's `sttl`; data/audit.py determines
# empirically whether any single feature is doing suspicious amounts of work.
_SUSPECTED_ARTIFACTS: list[str] = []


def load(
    *,
    train_split: str = "train",
    test_split: str = "test",
    drop_artifacts: bool = False,
) -> Dataset:
    """Load NSL-KDD.

    `train_split="train_20pct"` gives a 25k-row training set for fast iteration.
    `test_split="test_21"` gives the harder subset - rows most classic learners got wrong.
    """
    if train_split not in _FILES or test_split not in _FILES:
        raise ValueError(f"unknown split; choose from {sorted(_FILES)}")

    train_raw = _read_split(train_split)
    test_raw = _read_split(test_split)

    X_train, y_train, fam_train = _prepare(train_raw, drop_artifacts=drop_artifacts)
    X_test, y_test, fam_test = _prepare(test_raw, drop_artifacts=drop_artifacts)

    categorical = [c for c in schema.NSLKDD_CATEGORICAL if c in X_train.columns]
    numeric = [c for c in X_train.columns if c not in categorical]

    for col in categorical:
        X_train[col] = X_train[col].astype(str).str.strip().str.lower()
        X_test[col] = X_test[col].astype(str).str.strip().str.lower()

    return Dataset(
        name=f"nsl-kdd({train_split}/{test_split})",
        X_train=X_train,
        y_train=y_train,
        fam_train=fam_train,
        X_test=X_test,
        y_test=y_test,
        fam_test=fam_test,
        categorical=categorical,
        numeric=numeric,
        suspected_artifacts=[] if drop_artifacts else list(_SUSPECTED_ARTIFACTS),
        capabilities=Capabilities(
            has_entities=False,
            has_timestamps=False,
            has_natural_unseen_split=True,  # the 17 unseen attack types
            families=("normal", "dos", "probe", "r2l", "u2r"),
        ),
    )


def unseen_mask(fine_labels: pd.Series) -> pd.Series:
    """Boolean mask over test rows whose attack type never appears in training.

    3,750 rows, 16.6% of `KDDTest+`. This is the population for
    `penumbra eval --dataset nslkdd --unseen-only`: real zero-days, not a holdout we invented.
    """
    normalised = fine_labels.astype(str).str.strip().str.lower().str.rstrip(".")
    return normalised.isin(schema.NSLKDD_UNSEEN_IN_TEST)


def load_with_fine_labels(
    *, train_split: str = "train", test_split: str = "test"
) -> tuple[Dataset, pd.Series, pd.Series]:
    """Load, and also return the fine-grained attack names.

    The unseen-17 experiment needs the specific attack name (`mscan`, `apache2`, ...), which the
    coarse dos/probe/r2l/u2r categories throw away.
    """
    ds = load(train_split=train_split, test_split=test_split)
    fine_train = _read_split(train_split)["label"].reset_index(drop=True)
    fine_test = _read_split(test_split)["label"].reset_index(drop=True)
    return ds, fine_train, fine_test
