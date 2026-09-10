"""UNSW-NB15 loader (the published 175,341 / 82,332 split).

Traps handled here, each documented in docs/DATA_TRAPS.md:

  * UTF-8 BOM on the `id` column -> read with `utf-8-sig` or every lookup by name fails.
  * The training file is LARGER than the testing file. That is correct; row counts are asserted so
    that a swapped mirror (Mireu-Lab has train/test reversed) is caught rather than trained on.
  * `id` is a row counter and is dropped.
  * `attack_cat` uses inconsistent casing/whitespace across releases ("Backdoor" vs "Backdoors").
  * No IP addresses and no timestamps exist in this split, so `Capabilities` declares them absent.
"""

from __future__ import annotations

import pandas as pd

from penumbra.config import settings
from penumbra.data import schema
from penumbra.data.loaders.base import Capabilities, Dataset, DatasetNotFetched

_FILES = {
    "train": "UNSW_NB15_training-set.csv",
    "test": "UNSW_NB15_testing-set.csv",
}

_EXPECTED_ROWS = {"train": 175_341, "test": 82_332}

# `attack_cat` is not spelled consistently between UNSW-NB15 releases. Normalising to the canonical
# ten avoids a silent eleventh class that would break per-family reporting.
_FAMILY_ALIASES = {
    "backdoors": "Backdoor",
    "backdoor": "Backdoor",
    "dos": "DoS",
    "exploits": "Exploits",
    "fuzzers": "Fuzzers",
    "generic": "Generic",
    "normal": "Normal",
    "reconnaissance": "Reconnaissance",
    "shellcode": "Shellcode",
    "worms": "Worms",
    "analysis": "Analysis",
}


def _normalise_family(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "Normal"
    key = str(value).strip().lower()
    if not key or key == "-":
        return "Normal"
    try:
        return _FAMILY_ALIASES[key]
    except KeyError as exc:
        # Do not fall back to Normal: an unrecognised family would become a fabricated benign row.
        raise KeyError(f"unmapped UNSW attack_cat value: {value!r}") from exc


def _read_split(split: str) -> pd.DataFrame:
    path = settings().raw_dir / "unsw" / _FILES[split]
    if not path.exists():
        raise DatasetNotFetched(str(path), "unsw")

    # utf-8-sig strips the BOM that would otherwise make the first column name "﻿id".
    df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    df.columns = [c.strip() for c in df.columns]

    if len(df) != _EXPECTED_ROWS[split]:
        raise ValueError(
            f"UNSW {split}: {len(df):,} rows, expected {_EXPECTED_ROWS[split]:,}. "
            "Some mirrors serve the splits under swapped filenames - check data/manifest.json."
        )
    missing = set(schema.UNSW_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"UNSW {split}: missing expected columns {sorted(missing)}")
    return df


def load(*, drop_artifacts: bool = False) -> Dataset:
    """Load UNSW-NB15.

    `drop_artifacts=True` removes the features that data/audit.py has flagged as probable testbed
    artifacts (`sttl` and friends). Every headline metric is reported both ways; this flag is how
    the second number gets produced.
    """
    train = _read_split("train")
    test = _read_split("test")

    fam_train = train[schema.UNSW_LABEL_FAMILY].map(_normalise_family)
    fam_test = test[schema.UNSW_LABEL_FAMILY].map(_normalise_family)
    y_train = train[schema.UNSW_LABEL_BINARY].astype(int)
    y_test = test[schema.UNSW_LABEL_BINARY].astype(int)

    # The binary label and the family label must agree, or one of them is wrong and every downstream
    # number inherits the disagreement.
    for name, y, fam in (("train", y_train, fam_train), ("test", y_test, fam_test)):
        implied = (fam != "Normal").astype(int)
        mismatch = int((implied != y).sum())
        if mismatch:
            raise ValueError(f"UNSW {name}: {mismatch:,} rows where `label` disagrees with `attack_cat`")

    drop = [*schema.UNSW_DROP, schema.UNSW_LABEL_FAMILY, schema.UNSW_LABEL_BINARY]
    if drop_artifacts:
        drop += schema.UNSW_SUSPECTED_ARTIFACTS

    X_train = train.drop(columns=drop, errors="ignore")
    X_test = test.drop(columns=drop, errors="ignore")

    categorical = [c for c in schema.UNSW_CATEGORICAL if c in X_train.columns]
    numeric = [c for c in X_train.columns if c not in categorical]

    # Categorical columns arrive with inconsistent case and stray whitespace across releases.
    for col in categorical:
        X_train[col] = X_train[col].astype(str).str.strip().str.lower()
        X_test[col] = X_test[col].astype(str).str.strip().str.lower()

    # Numeric columns occasionally parse as object when a stray token appears in one row. Coercing
    # here (rather than letting sklearn fail later) puts the error next to its cause.
    for col in numeric:
        if X_train[col].dtype == object or X_test[col].dtype == object:
            X_train[col] = pd.to_numeric(X_train[col], errors="coerce")
            X_test[col] = pd.to_numeric(X_test[col], errors="coerce")

    return Dataset(
        name="unsw-nb15" + ("-noartifacts" if drop_artifacts else ""),
        X_train=X_train.reset_index(drop=True),
        y_train=y_train.reset_index(drop=True),
        fam_train=fam_train.reset_index(drop=True),
        X_test=X_test.reset_index(drop=True),
        y_test=y_test.reset_index(drop=True),
        fam_test=fam_test.reset_index(drop=True),
        categorical=categorical,
        numeric=numeric,
        suspected_artifacts=[] if drop_artifacts else list(schema.UNSW_SUSPECTED_ARTIFACTS),
        capabilities=Capabilities(
            # No srcip/dstip/stime in the published split. This is why correlation, graph features
            # and temporal splits run on CICIDS2017 instead (ADR-0004).
            has_entities=False,
            has_timestamps=False,
            has_natural_unseen_split=False,
            families=tuple(schema.UNSW_FAMILIES),
        ),
    )
