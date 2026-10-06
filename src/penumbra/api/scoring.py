"""Live scoring with the registered champions: sampled test flows, an uploaded CSV, or a pcap.

ADR-0006. Until this existed the API never scored anything - it displayed alerts scored offline -
and THREAT_MODEL T3 listed "no scoring endpoint at all" as a mitigation, because a public scorer
is an oracle: query it enough and you learn its thresholds. What replaces that mitigation:

  * every call is rate-limited per caller (role + client address; guests share one identity, so
    the address is what separates them) and capped in rows;
  * scores come back rounded to two decimals, so a threshold cannot be bisected to the last digit;
  * every call is written to the hash-chained audit log, with its size and what it found;
  * results are returned to the caller and stored nowhere - the alert queue, its verdicts and its
    training pool never see a scored upload, so an upload cannot reach retraining (T1);
  * the output is alerts, as everywhere else. There is still no code path that blocks traffic.

The models are the registry champions, bundled into the deployment image beside their manifests.
"""

from __future__ import annotations

import io
import tempfile
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from penumbra.config import settings

DATASETS: dict[str, dict[str, str]] = {
    "unsw": {"title": "UNSW-NB15", "test": "UNSW-NB15 test split, 82,332 flows"},
    "nslkdd": {"title": "NSL-KDD", "test": "KDDTest+, 22,544 connections; 17 attack types never in training"},
    "cicids": {
        "title": "CICIDS2017",
        "test": "Thursday and Friday; every attack family is absent from training",
    },
}
PCAP_DATASET = "unsw"  # the only schema `pcap.assemble` produces
SAMPLE_COLUMNS = ("_label", "_family", "_unseen")
MAX_CSV_BYTES = 5 * 2**20
MAX_PCAP_BYTES = 8 * 2**20
# A capture is scored whole (its summary covers every flow); only the per-flow rows returned are
# capped by the caller's row limit, which is what bounds how much an oracle can harvest per call.
MAX_PCAP_FLOWS = 20_000
ATTACK_SHARES = (0.05, 0.10, 0.50)
ROUND = 2


@dataclass(frozen=True)
class Limits:
    calls_per_hour: int
    max_rows: int


GUEST_LIMITS = Limits(calls_per_hour=30, max_rows=500)
USER_LIMITS = Limits(calls_per_hour=120, max_rows=5_000)


class ScoringError(ValueError):
    """A request the scorer refuses, with a message fit to show the caller."""


# --- rate limiting -------------------------------------------------------------------------------


class RateLimiter:
    """Sliding one-hour window per key. In memory: the API runs as a single replica."""

    def __init__(self, window_seconds: float = 3600.0) -> None:
        self.window = window_seconds
        self._calls: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def allow(self, key: str, limit: int) -> tuple[bool, int]:
        now = time.monotonic()
        with self._lock:
            q = self._calls.setdefault(key, deque())
            while q and now - q[0] > self.window:
                q.popleft()
            if len(q) >= limit:
                return False, len(q)
            q.append(now)
            return True, len(q)


# --- models and sample pools -----------------------------------------------------------------------


class ModelStore:
    """Champions loaded on first use and kept. UNSW is ~750 MB in memory, the others far less."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = root
        self._models: dict[str, Any] = {}
        self._pools: dict[str, pd.DataFrame] = {}
        self._lock = threading.Lock()

    def _models_dir(self) -> Path:
        return (self.root or settings().artifact_root) / "models"

    def _samples_dir(self) -> Path:
        return (self.root or settings().artifact_root) / "samples"

    def available(self) -> list[str]:
        return [d for d in DATASETS if (self._models_dir() / d / "detector.joblib").exists()]

    def detector(self, dataset: str) -> Any:
        if dataset not in DATASETS:
            raise ScoringError(f"unknown dataset {dataset!r}; expected one of {sorted(DATASETS)}")
        with self._lock:
            if dataset not in self._models:
                path = self._models_dir() / dataset
                if not (path / "detector.joblib").exists():
                    raise ScoringError(f"no {dataset} model is deployed here")
                from penumbra.models.detector import PenumbraDetector

                self._models[dataset] = PenumbraDetector.load(path)
            return self._models[dataset]

    def pool(self, dataset: str) -> pd.DataFrame:
        with self._lock:
            if dataset not in self._pools:
                path = self._samples_dir() / f"{dataset}_test_sample.csv.gz"
                if not path.exists():
                    raise ScoringError(f"no {dataset} sample pool is deployed here")
                self._pools[dataset] = pd.read_csv(path, low_memory=False)
            return self._pools[dataset]


def categorical_columns(detector: Any) -> list[str]:
    prep = detector.supervised_model.named_steps["prep"]
    for name, _, cols in getattr(prep, "transformers_", []):
        if name == "cat":
            return [str(c) for c in cols]
    return []


def model_info(store: ModelStore, dataset: str) -> dict[str, Any]:
    det = store.detector(dataset)
    meta = det.metadata
    return {
        "dataset": dataset,
        "title": DATASETS[dataset]["title"],
        "test_pool": DATASETS[dataset]["test"],
        "registry_version": getattr(meta, "registry_version", None),
        "model": getattr(meta, "model_name", None),
        "calibration": getattr(meta, "calibration", None),
        "trained_rows": getattr(meta, "n_train_rows", None),
        "target_fpr": det.target_fpr,
        "features": list(det._feature_names),
        "categorical": categorical_columns(det),
        "quarantined": list(getattr(meta, "quarantined_features", []) or []),
    }


# --- inputs ----------------------------------------------------------------------------------------


def coerce(detector: Any, frame: pd.DataFrame) -> pd.DataFrame:
    """The detector's columns in its order and types; a clear error naming what is missing."""
    frame = frame.rename(columns=lambda c: str(c).strip())
    need = list(detector._feature_names)
    missing = [c for c in need if c not in frame.columns]
    if missing:
        shown = ", ".join(missing[:8]) + (f" and {len(missing) - 8} more" if len(missing) > 8 else "")
        raise ScoringError(f"{len(missing)} required column(s) missing: {shown}. Download the template.")
    X = frame[need].copy()
    cats = set(categorical_columns(detector))
    for col in need:
        if col in cats:
            X[col] = X[col].astype(str).str.strip().str.lower()
        else:
            X[col] = pd.to_numeric(X[col], errors="coerce")
    return X.reset_index(drop=True)


def sample(store: ModelStore, dataset: str, n: int, attack_share: float, seed: int | None) -> pd.DataFrame:
    """`n` held-out flows at the chosen attack share, drawn from the bundled pool."""
    if attack_share not in ATTACK_SHARES:
        raise ScoringError(f"attack_share must be one of {ATTACK_SHARES}")
    pool = store.pool(dataset)
    rng = np.random.default_rng(seed)
    attack = np.flatnonzero(pool["_label"].to_numpy() == 1)
    benign = np.flatnonzero(pool["_label"].to_numpy() == 0)
    n_att = min(len(attack), int(round(n * attack_share)))
    n_ben = min(len(benign), n - n_att)
    rows = np.concatenate(
        [rng.choice(attack, n_att, replace=False), rng.choice(benign, n_ben, replace=False)]
    )
    rng.shuffle(rows)
    return pool.iloc[rows].reset_index(drop=True)


def read_csv_upload(content: bytes) -> pd.DataFrame:
    if len(content) > MAX_CSV_BYTES:
        raise ScoringError(f"CSV is {len(content) / 2**20:.1f} MB; the limit is {MAX_CSV_BYTES // 2**20} MB")
    try:
        return pd.read_csv(io.BytesIO(content), low_memory=False)
    except (ValueError, UnicodeDecodeError, pd.errors.ParserError) as exc:
        raise ScoringError(f"could not read the CSV: {exc}") from exc


def read_pcap_upload(content: bytes) -> pd.DataFrame:
    """Flows from a capture, with addresses kept aside in attrs, never in the features."""
    if len(content) > MAX_PCAP_BYTES:
        raise ScoringError(
            f"capture is {len(content) / 2**20:.1f} MB; the limit is {MAX_PCAP_BYTES // 2**20} MB"
        )
    from penumbra.pcap import assemble

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "upload.pcap"
        path.write_bytes(content)
        try:
            frame = assemble.assemble(path)
        except Exception as exc:  # noqa: BLE001 - any parser failure is the upload's problem, reported as such
            raise ScoringError(f"could not read the capture as pcap/pcapng: {type(exc).__name__}") from exc
    if frame.empty:
        raise ScoringError("the capture contains no IP flows")
    return frame


# --- scoring ---------------------------------------------------------------------------------------

ALERTING = {"KNOWN_ATTACK", "SUSPECTED_NOVEL", "UNCERTAIN"}
HEADS = {0: "none", 1: "supervised", 2: "novelty", 3: "both"}


def score(
    detector: Any,
    X: pd.DataFrame,
    *,
    dataset: str,
    truth: pd.DataFrame | None = None,
    endpoints: list[tuple[str, str]] | None = None,
    max_rows: int,
    max_returned: int | None = None,
) -> dict[str, Any]:
    """Score `X` with the full two-head detector and return alerts, never a decision to block.

    At most `max_rows` flows are scored and summarised; at most `max_returned` (default: the same)
    come back as per-flow rows.
    """
    from penumbra.alerts.builder import alerts_with_positions

    truncated = len(X) > max_rows
    if truncated:
        X = X.iloc[:max_rows]
        truth = truth.iloc[:max_rows] if truth is not None else None
    returned = max_rows if max_returned is None else max_returned
    scored = detector.score(X)
    positioned = alerts_with_positions(detector, X, scored, dataset=dataset, include_benign=True)
    fired = scored["fired"].to_numpy()
    rows: list[dict[str, Any]] = []
    for pos, alert in positioned:
        row: dict[str, Any] = {
            "row": pos,
            "verdict": alert.verdict.value,
            "p_attack": round(float(alert.p_attack), ROUND),
            "novelty_percentile": round(float(alert.novelty_percentile), ROUND),
            "family": alert.family,
            "head": HEADS.get(int(fired[pos]), "none"),
            "conformal_set": list(alert.conformal_set),
            "why": [
                {"feature": c.feature, "narrative": c.narrative, "direction": c.direction}
                for c in alert.contributions[:3]
            ],
            "attack_technique": getattr(getattr(alert, "attack", None), "technique_id", None),
        }
        if truth is not None:
            row["truth"] = {
                "label": int(truth["_label"].iloc[pos]),
                "family": str(truth["_family"].iloc[pos]),
                "unseen_in_training": bool(truth["_unseen"].iloc[pos]) if "_unseen" in truth else None,
            }
        if endpoints is not None and pos < len(endpoints):
            row["src"], row["dst"] = endpoints[pos]
        rows.append(row)

    verdicts = pd.Series([r["verdict"] for r in rows])
    summary: dict[str, Any] = {
        "flows": len(rows),
        "alerts": int(verdicts.isin(ALERTING).sum()),
        "by_verdict": {str(k): int(v) for k, v in verdicts.value_counts().items()},
        "truncated_to": max_rows if truncated else None,
        "rows_returned": min(len(rows), returned),
        "model": getattr(detector.metadata, "registry_version", None),
    }
    if truth is not None:
        y = truth["_label"].to_numpy().astype(int)
        flagged = verdicts.isin(ALERTING).to_numpy()
        summary["truth"] = {
            "attacks": int(y.sum()),
            "attacks_reaching_an_analyst": int((flagged & (y == 1)).sum()),
            "benign": int((y == 0).sum()),
            "benign_reaching_an_analyst": int((flagged & (y == 0)).sum()),
        }
        if "_unseen" in truth:
            unseen = truth["_unseen"].to_numpy().astype(bool) & (y == 1)
            summary["truth"]["unseen_attacks"] = int(unseen.sum())
            summary["truth"]["unseen_reaching_an_analyst"] = int((flagged & unseen).sum())
    # Alerts first, so a capped list shows what fired rather than the first N benign flows.
    rows.sort(key=lambda r: (r["verdict"] not in ALERTING, -r["p_attack"]))
    return {"summary": summary, "rows": rows[:returned]}


def template_csv(store: ModelStore, dataset: str, rows: int = 5) -> str:
    """The detector's columns, with a few real held-out rows to show the format. No labels."""
    det = store.detector(dataset)
    pool = store.pool(dataset)
    return pool[list(det._feature_names)].head(rows).to_csv(index=False)


def build_pool(
    X: pd.DataFrame,
    y: pd.Series,
    families: pd.Series,
    *,
    unseen: np.ndarray | None = None,
    n_benign: int = 2_000,
    n_attack: int = 2_000,
    seed: int = 42,
) -> pd.DataFrame:
    """A fixed, labelled slice of a TEST split for `sample`: benign at random, attacks by family.

    Attack rows are spread across families (each gets an equal share, capped by what it has, the
    remainder topped up at random) so a 200-row draw can show a rare family and not only the
    dominant one. Every row is from the held-out split; none was ever trained on.
    """
    rng = np.random.default_rng(seed)
    y_arr = y.to_numpy().astype(int)
    fam = families.astype(str).to_numpy()
    benign = np.flatnonzero(y_arr == 0)
    take_b = rng.choice(benign, min(n_benign, len(benign)), replace=False)
    attack_fams = sorted(set(fam[y_arr == 1]))
    share = max(1, n_attack // max(len(attack_fams), 1))
    chosen: list[int] = []
    for f in attack_fams:
        idx = np.flatnonzero((y_arr == 1) & (fam == f))
        chosen.extend(rng.choice(idx, min(share, len(idx)), replace=False).tolist())
    rest = np.setdiff1d(np.flatnonzero(y_arr == 1), np.asarray(chosen, dtype=int))
    top_up = max(0, n_attack - len(chosen))
    if top_up and len(rest):
        chosen.extend(rng.choice(rest, min(top_up, len(rest)), replace=False).tolist())
    rows = np.concatenate([take_b, np.asarray(chosen, dtype=int)])
    pool = X.iloc[rows].reset_index(drop=True).copy()
    pool["_label"] = y_arr[rows]
    pool["_family"] = fam[rows]
    if unseen is not None:
        pool["_unseen"] = np.asarray(unseen)[rows].astype(bool)
    return pool
