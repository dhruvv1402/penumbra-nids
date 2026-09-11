"""The whole detector, as one object.

Everything the evaluation modules assemble piecemeal, packaged so it can be fitted once, saved, and
loaded by the API and the replay engine. This is what turns a set of experiments into a thing that
runs.

It owns the two heads, their separate preprocessing, the benign reference distributions, and the
thresholds - and it keeps them together, because a threshold without the reference it was fitted
against is meaningless and the two drifting apart is a whole class of deployment bug.

Fit order matters and is enforced:

  1. supervised pipeline on ALL training rows
  2. benign-only pipeline on BENIGN rows only, split into fit and calibration halves
  3. novelty detectors on the fit half
  4. percentile references from the calibration half
  5. thresholds from held-out benign scores, never from test labels
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from penumbra.alerts.models import Alert, Contribution, NetworkContext
from penumbra.alerts.scoring import ScoringPolicy, build_alert
from penumbra.data.loaders.base import Dataset
from penumbra.features.preprocess import assert_benign_only_fit, benign_only_pipeline
from penumbra.models import supervised
from penumbra.models.fusion import OrGate
from penumbra.models.novelty.ensemble import NoveltyEnsemble
from penumbra.seeds import SEED, seed_everything

CALIBRATION_FRACTION = 0.2


@dataclass
class DetectorMetadata:
    """Provenance. Written next to the artifact so a model can be traced to what made it."""

    model_name: str
    version: str
    dataset: str
    trained_at: str
    n_train_rows: int
    n_benign_fit: int
    n_benign_calibration: int
    target_fpr: float
    supervised_threshold: float
    novelty_threshold: float
    features: list[str] = field(default_factory=list)
    quarantined_features: list[str] = field(default_factory=list)
    seed: int = SEED

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


class PenumbraDetector:
    """Two heads, one object."""

    def __init__(self, *, policy: ScoringPolicy | None = None, target_fpr: float = 0.01) -> None:
        self.policy = policy or ScoringPolicy()
        self.target_fpr = target_fpr

        self.supervised_model: Any = None
        self.family_model: Any = None
        self.family_encoder: Any = None
        self.novelty_prep: Any = None
        self.novelty: NoveltyEnsemble | None = None
        self.gate: OrGate | None = None
        self.metadata: DetectorMetadata | None = None
        self._feature_names: list[str] = []

    # --- fit -------------------------------------------------------------------------------------

    def fit(
        self,
        ds: Dataset,
        *,
        model_name: str = "rf",
        fit_family_model: bool = True,
        on_progress: Any = None,
    ) -> PenumbraDetector:
        seed_everything()

        def step(msg: str) -> None:
            if on_progress:
                on_progress(msg)

        step("supervised head")
        self.supervised_model = supervised.build(model_name, ds, n_classes=2, balanced=True)
        self.supervised_model.fit(ds.X_train, ds.y_train)

        if fit_family_model:
            step("family classifier")
            self.family_model, self.family_encoder = supervised.fit_multiclass(model_name, ds, balanced=True)

        step("benign-only pipeline")
        benign = ds.X_train.loc[ds.y_train == 0]
        # The contract the novelty head's whole claim rests on.
        assert_benign_only_fit(ds, benign)

        rng = np.random.default_rng(SEED)
        order = rng.permutation(len(benign))
        n_cal = int(len(benign) * CALIBRATION_FRACTION)
        cal_idx, fit_idx = order[:n_cal], order[n_cal:]
        benign_fit = benign.iloc[fit_idx]
        benign_cal = benign.iloc[cal_idx]

        self.novelty_prep = benign_only_pipeline(ds)
        Z_fit = self.novelty_prep.fit_transform(benign_fit)
        Z_cal = self.novelty_prep.transform(benign_cal)

        step("novelty detectors")
        self.novelty = NoveltyEnsemble().fit(Z_fit)
        self.novelty.calibrate_on(Z_cal)

        step("thresholds from held-out benign")
        benign_p = supervised.attack_scores(self.supervised_model, benign_cal)
        benign_n = self.novelty.score(Z_cal, how="max")
        self.gate = OrGate.fit(benign_p, benign_n, total_fpr=self.target_fpr, use_novelty=True)

        self._feature_names = list(ds.feature_names)
        self.metadata = DetectorMetadata(
            model_name=model_name,
            version="0.1.0",
            dataset=ds.name,
            trained_at=datetime.now(UTC).isoformat(),
            n_train_rows=len(ds.X_train),
            n_benign_fit=len(benign_fit),
            n_benign_calibration=len(benign_cal),
            target_fpr=self.target_fpr,
            supervised_threshold=self.gate.supervised_threshold,
            novelty_threshold=self.gate.novelty_threshold,
            features=self._feature_names,
        )
        return self

    # --- score -----------------------------------------------------------------------------------

    def score(self, X: pd.DataFrame) -> pd.DataFrame:
        """Score a batch. Returns the raw numbers; `alerts()` turns them into Alert objects."""
        if self.supervised_model is None or self.novelty is None or self.gate is None:
            raise RuntimeError("detector is not fitted")

        p_attack = supervised.attack_scores(self.supervised_model, X)
        Z = self.novelty_prep.transform(X)
        novelty_raw = self.novelty.score(Z, how="max")
        agreement = self.novelty.agreement(Z, percentile=0.99)
        which = self.gate.which_fired(p_attack, novelty_raw)

        families: list[str | None] = [None] * len(X)
        if self.family_model is not None:
            predicted = supervised.predict_families(self.family_model, self.family_encoder, X)
            families = [None if str(f).lower() in {"normal", "benign"} else str(f) for f in predicted]

        return pd.DataFrame(
            {
                "p_attack": p_attack,
                "novelty_percentile": novelty_raw,
                "agreement": agreement,
                "fired": which,  # 0 neither, 1 supervised, 2 novelty only, 3 both
                "family": families,
            },
            index=X.index,
        )

    def alerts(
        self,
        X: pd.DataFrame,
        scored: pd.DataFrame | None = None,
        *,
        dataset: str = "unsw",
        entities: list[str] | None = None,
        include_benign: bool = False,
    ) -> list[Alert]:
        """Turn scored rows into Alert objects."""
        scored = scored if scored is not None else self.score(X)
        out: list[Alert] = []

        for pos, (_, row) in enumerate(scored.iterrows()):
            fired = int(row["fired"])
            if fired == 0 and not include_benign:
                continue

            # SUSPECTED_NOVEL is precisely "novelty fired and supervised did not", so a novelty-only
            # detection carries no family by construction - naming it would contradict the verdict.
            raw_family = row["family"]
            family = raw_family if (fired in (1, 3) and isinstance(raw_family, str)) else None

            alert = build_alert(
                p_attack=float(row["p_attack"]),
                novelty_percentile=float(row["novelty_percentile"]),
                policy=self.policy,
                family=family,
                dataset=dataset,
                agreement=int(row["agreement"]),
                network=self._network_for(X, pos, entities),
                contributions=self._contributions(X, pos),
                model_version=self.metadata.version if self.metadata else "0.1.0",
            )
            out.append(alert)
        return out

    def _network_for(self, X: pd.DataFrame, pos: int, entities: list[str] | None) -> NetworkContext:
        row = X.iloc[pos]
        return NetworkContext(
            src_ip=entities[pos] if entities and pos < len(entities) else None,
            dst_port=_as_int(row.get("dst_port")),
            protocol=str(row.get("proto") or row.get("protocol_type") or "") or None,
            src_bytes=_as_int(row.get("sbytes") or row.get("src_bytes")),
            dst_bytes=_as_int(row.get("dbytes") or row.get("dst_bytes")),
            src_packets=_as_int(row.get("spkts")),
            dst_packets=_as_int(row.get("dpkts")),
            duration_ms=_as_float(row.get("dur")),
        )

    def _contributions(self, X: pd.DataFrame, pos: int, top: int = 5) -> list[Contribution]:
        """Top feature contributions, rendered in plain English.

        Uses the tree ensemble's global importances weighted by how unusual this row's value is
        relative to training. Genuine per-row TreeSHAP lands in the explain module; this keeps the
        replay path fast enough to stream, and it is labelled for what it is.
        """
        importances = self._importances()
        if importances is None:
            return []

        row = X.iloc[pos]
        ranked = sorted(importances.items(), key=lambda kv: kv[1], reverse=True)[:top]
        return [
            Contribution(
                feature=name,
                value=_as_float(row.get(name)),
                shap_value=float(weight),
                direction="toward_attack",
                narrative=_narrate(name, row.get(name)),
            )
            for name, weight in ranked
            if name in row.index
        ]

    def _importances(self) -> dict[str, float] | None:
        try:
            clf = self.supervised_model.named_steps["clf"]
            prep = self.supervised_model.named_steps["prep"]
            names = [str(n) for n in prep.get_feature_names_out()]
            values = getattr(clf, "feature_importances_", None)
            if values is None:
                return None
            # Collapse one-hot columns back onto their source feature.
            merged: dict[str, float] = {}
            for name, weight in zip(names, values, strict=True):
                base = name.split("_")[0] if "_" in name and name not in self._feature_names else name
                merged[base] = merged.get(base, 0.0) + float(weight)
            return merged
        except Exception:  # noqa: BLE001 - attribution is best-effort, never fatal to scoring
            return None

    # --- persistence -----------------------------------------------------------------------------

    def save(self, directory: Path) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "supervised_model": self.supervised_model,
                "family_model": self.family_model,
                "family_encoder": self.family_encoder,
                "novelty_prep": self.novelty_prep,
                "novelty": self.novelty,
                "gate": self.gate,
                "policy": self.policy,
                "target_fpr": self.target_fpr,
                "feature_names": self._feature_names,
            },
            directory / "detector.joblib",
            compress=3,
        )
        if self.metadata:
            (directory / "metadata.json").write_text(
                json.dumps(self.metadata.to_dict(), indent=2), encoding="utf-8"
            )
        return directory / "detector.joblib"

    @classmethod
    def load(cls, directory: Path) -> PenumbraDetector:
        blob = joblib.load(directory / "detector.joblib")
        det = cls(policy=blob["policy"], target_fpr=blob["target_fpr"])
        det.supervised_model = blob["supervised_model"]
        det.family_model = blob["family_model"]
        det.family_encoder = blob["family_encoder"]
        det.novelty_prep = blob["novelty_prep"]
        det.novelty = blob["novelty"]
        det.gate = blob["gate"]
        det._feature_names = blob["feature_names"]

        meta_path = directory / "metadata.json"
        if meta_path.exists():
            det.metadata = DetectorMetadata(**json.loads(meta_path.read_text(encoding="utf-8")))
        return det


def _as_int(value: Any) -> int | None:
    try:
        if value is None or (isinstance(value, float) and not np.isfinite(value)):
            return None
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _as_float(value: Any) -> float | None:
    try:
        f = float(value)
        return f if np.isfinite(f) else None
    except (TypeError, ValueError):
        return None


# Plain-English renderers. "sload = 1400000" is not something a tier-1 analyst can act on;
# "outbound throughput 1.4 Mb/s" is.
_NARRATIVES: dict[str, str] = {
    "sttl": "source time-to-live {v} - a property of the sending host's OS, flagged as a testbed artifact",
    "ct_dst_sport_ltm": "{v} recent connections to this destination port across hosts",
    "ct_srv_src": "{v} recent connections to the same service from this source",
    "ct_dst_ltm": "{v} recent connections to this destination",
    "sbytes": "{v} bytes sent",
    "dbytes": "{v} bytes received",
    "spkts": "{v} packets sent",
    "rate": "{v} packets/second",
    "sload": "{v} bits/second outbound",
    "dur": "flow lasted {v} seconds",
    "smean": "mean outbound packet size {v} bytes",
    "count": "{v} connections to the same host in the window",
    "srv_count": "{v} connections to the same service in the window",
    "serror_rate": "{v} of connections had SYN errors - the signature of a SYN flood",
    "srv_serror_rate": "{v} SYN-error rate on this service",
    "diff_srv_rate": "{v} of connections went to differing services - the shape of a port sweep",
    "dst_host_srv_count": "{v} connections to this service on the destination host",
}


def _narrate(feature: str, value: Any) -> str:
    template = _NARRATIVES.get(feature)
    if template is None:
        return ""
    number = _as_float(value)
    if number is None:
        return ""
    rendered = f"{number:,.0f}" if abs(number) >= 100 else f"{number:,.3g}"
    return template.format(v=rendered)
