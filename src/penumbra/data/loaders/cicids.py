"""CICIDS2017 loader (the Engelen et al. corrected re-release).

This is the only dataset here carrying `Src IP`, `Dst IP` and `Timestamp`, which makes it the only
one that can support alert-to-incident correlation, entity-graph features, per-host sequence windows
and a genuine temporal split. See ADR-0004.

We use the **improved** release rather than the original because the original's labels are known to
be wrong: a CICFlowMeter bug terminated TCP flows on a single FIN rather than a mutual exchange, and
more than 20% of flows changed label or boundary in the correction (Engelen et al., WTMC 2021; IEEE
CNS 2022).

Traps handled here, each documented in docs/DATA_TRAPS.md:

  * Headers carry leading/trailing whitespace. Strip before anything else or every lookup fails.
  * `Destination Port` / `Dst Port` is a LABEL LEAK - attacks targeted fixed victim ports, so one
    decision stump on it scores near-perfectly. Dropped by default; `keep_leaks=True` exists only to
    quantify the inflation.
  * Zero-duration flows produce NaN and +/-Inf in the rate columns.
  * Several columns are identically zero.
  * Web-attack labels carry a CP1252 en-dash that is not valid UTF-8, hence `encoding="latin1"`.
  * The improved release adds "Attempted" labels. The authors are explicit that these should be
    folded into the real attack class or relabelled benign, never treated as a third class.
  * Monday is benign-only, which makes it the correct source for the novelty head's reference.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from penumbra.config import settings
from penumbra.data import schema
from penumbra.data.loaders.base import Capabilities, Dataset, DatasetNotFetched

DAYS = ("monday", "tuesday", "wednesday", "thursday", "friday")

ENTITY_COLUMNS = ["Flow ID", "Src IP", "Src Port", "Dst IP", "Dst Port", "Protocol", "Timestamp"]
LABEL = "Label"
BENIGN_LABEL = "BENIGN"

# Attacks were generated against fixed victim ports, so this column identifies the attack rather
# than describing it. Dropped for any honest evaluation.
LEAK_COLUMNS = ["Dst Port", "Destination Port"]

# The improved release marks flows where an attack was attempted but did not complete. Per the
# authors these are folded into the attack class rather than forming a third one.
# The release spells these as a suffix ("Botnet - Attempted"), and occasionally as a prefix.
# Both forms are folded into the parent attack class, per the authors' own guidance - treating
# "attempted" as a third class would invent a category the data does not have.
_ATTEMPTED = re.compile(r"(^\s*attempted\s+|\s*[-–—]?\s*attempted\s*$)", re.IGNORECASE)


def _dir() -> Path:
    return settings().raw_dir / "cicids" / "improved"


def _normalise_label(value: object) -> str:
    text = str(value).strip()
    text = _ATTEMPTED.sub("", text)
    # Web-attack labels contain a CP1252 en-dash that survives as U+FFFD in some mirrors. Match on
    # a pattern rather than a literal so both spellings normalise the same way.
    text = re.sub(r"[�–—]", "-", text)
    text = re.sub(r"\s+", " ", text)
    return text or BENIGN_LABEL


def load_day(day: str, *, nrows: int | None = None) -> pd.DataFrame:
    """Read one day, cleaned."""
    path = _dir() / f"{day}.csv"
    if not path.exists():
        raise DatasetNotFetched(str(path), "cicids")

    df = pd.read_csv(path, encoding="latin1", low_memory=False, nrows=nrows)
    df.columns = [c.strip() for c in df.columns]

    # pandas silently renames a duplicated column to `.1`; it carries no information.
    df = df.loc[:, ~df.columns.str.endswith(".1")]

    df[LABEL] = df[LABEL].map(_normalise_label)
    if "Timestamp" in df.columns:
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce", format="mixed")
    df["__day"] = day
    return df


def load(
    *,
    train_days: tuple[str, ...] = ("monday", "tuesday", "wednesday"),
    test_days: tuple[str, ...] = ("thursday", "friday"),
    nrows_per_day: int | None = None,
    keep_leaks: bool = False,
    drop_duplicates: bool = True,
) -> Dataset:
    """Load CICIDS2017 with a day-based temporal split.

    Train Mon-Wed, test Thu-Fri. **Not a random split** - the data is temporally ordered and a
    random split leaks future into past, inflating every number.
    """
    train_raw = pd.concat([load_day(d, nrows=nrows_per_day) for d in train_days], ignore_index=True)
    test_raw = pd.concat([load_day(d, nrows=nrows_per_day) for d in test_days], ignore_index=True)

    if drop_duplicates:
        # Before splitting would be wrong here - the split is temporal and fixed. De-duplicating
        # within each side removes the memorisation that internal duplicates cause, and the
        # cross-boundary overlap is measured separately by data/audit.py.
        train_raw = train_raw.drop_duplicates()
        test_raw = test_raw.drop_duplicates()

    train_x, train_y, train_fam, train_meta = _prepare(train_raw, keep_leaks=keep_leaks)
    test_x, test_y, test_fam, test_meta = _prepare(test_raw, keep_leaks=keep_leaks)

    # Align columns: a day can be missing a column another day has.
    shared = [c for c in train_x.columns if c in test_x.columns]
    train_x, test_x = train_x[shared], test_x[shared]

    ds = Dataset(
        name="cicids2017" + ("" if keep_leaks else "-noleak"),
        X_train=train_x,
        y_train=train_y,
        fam_train=train_fam,
        X_test=test_x,
        y_test=test_y,
        fam_test=test_fam,
        categorical=[],
        numeric=list(train_x.columns),
        suspected_artifacts=[] if keep_leaks else list(LEAK_COLUMNS),
        capabilities=Capabilities(
            has_entities=True,
            has_timestamps=True,
            has_natural_unseen_split=False,
            families=tuple(sorted(set(train_fam.unique()) | set(test_fam.unique()))),
        ),
    )
    # Entity and time columns travel alongside rather than as features: they are needed for
    # correlation and must never reach the model.
    ds.X_train.attrs["meta"] = train_meta
    ds.X_test.attrs["meta"] = test_meta
    return ds


def _prepare(
    df: pd.DataFrame, *, keep_leaks: bool
) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.DataFrame]:
    fam = df[LABEL].where(df[LABEL] != BENIGN_LABEL, "Normal").reset_index(drop=True)
    y = (df[LABEL] != BENIGN_LABEL).astype(int).reset_index(drop=True)

    meta_cols = [c for c in ENTITY_COLUMNS if c in df.columns]
    meta = df[meta_cols].reset_index(drop=True) if meta_cols else pd.DataFrame()

    drop = [*meta_cols, LABEL, "__day", *schema.CICIDS_CONSTANT_COLUMNS]
    if not keep_leaks:
        drop += LEAK_COLUMNS

    X = df.drop(columns=[c for c in drop if c in df.columns], errors="ignore")
    X = X.select_dtypes(include=[np.number]).reset_index(drop=True)

    # Zero-duration flows divide by zero in the rate columns. Left alone these reach the scaler as
    # inf and poison the column statistics for every other row.
    X = X.replace([np.inf, -np.inf], np.nan)

    # Identically-constant columns carry no information and break scalers.
    nunique = X.nunique(dropna=True)
    X = X.loc[:, nunique[nunique > 1].index]

    return X, y, fam, meta


def entities(ds: Dataset, split: str = "test") -> list[str]:
    """Pseudonymised source addresses, aligned with the feature rows.

    Correlation groups on these. They are real identities from the capture, pseudonymised before
    they reach an alert - so the compression ratio measured from them is a measured number, unlike
    anything derivable from UNSW-NB15.
    """
    from penumbra.api.security.pii import pseudonymise_ip

    X = ds.X_test if split == "test" else ds.X_train
    meta = X.attrs.get("meta")
    if meta is None or "Src IP" not in getattr(meta, "columns", []):
        return []
    return [pseudonymise_ip(str(ip)) for ip in meta["Src IP"]]


def timestamps(ds: Dataset, split: str = "test") -> list[pd.Timestamp]:
    X = ds.X_test if split == "test" else ds.X_train
    meta = X.attrs.get("meta")
    if meta is None or "Timestamp" not in getattr(meta, "columns", []):
        return []
    return list(meta["Timestamp"])


def destinations(ds: Dataset, split: str = "test") -> list[tuple[str | None, int | None]]:
    """(pseudonymised destination IP, destination port) per row, for fan-out features."""
    from penumbra.api.security.pii import pseudonymise_ip

    X = ds.X_test if split == "test" else ds.X_train
    meta = X.attrs.get("meta")
    if meta is None:
        return []
    dst_ip = meta["Dst IP"] if "Dst IP" in meta.columns else None
    dst_port = meta["Dst Port"] if "Dst Port" in meta.columns else None
    out: list[tuple[str | None, int | None]] = []
    for i in range(len(meta)):
        ip = pseudonymise_ip(str(dst_ip.iloc[i])) if dst_ip is not None else None
        port = int(dst_port.iloc[i]) if dst_port is not None and pd.notna(dst_port.iloc[i]) else None
        out.append((ip, port))
    return out
