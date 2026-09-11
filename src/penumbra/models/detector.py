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

from penumbra.data.loaders.base import Dataset
from penumbra.features.preprocess import assert_benign_only_fit, benign_only_pipeline
from penumbra.models import supervised
from penumbra.models.conformal import MondrianConformal
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

    def __init__(self, *, policy: Any = None, target_fpr: float = 0.01) -> None:
        # Typed as Any rather than importing ScoringPolicy: models/ must not depend on alerts/, and
        # the policy is carried for the alert builder rather than used during inference.
        self.policy = policy
        self.target_fpr = target_fpr

        self.supervised_model: Any = None
        self.family_model: Any = None
        self.family_encoder: Any = None
        self.novelty_prep: Any = None
        self.novelty: NoveltyEnsemble | None = None
        self.gate: OrGate | None = None
        self.conformal: MondrianConformal | None = None
        self.metadata: DetectorMetadata | None = None
        self._feature_names: list[str] = []
        # Ranked global importances, computed once. Recomputing per row made scoring O(rows x
        # features) and pinned replay throughput at ~38 flows/s.
        self._ranked_importances: list[tuple[str, float]] | None = None

    # --- fit -------------------------------------------------------------------------------------

    def fit(
        self,
        ds: Dataset,
        *,
        model_name: str = "rf",
        fit_family_model: bool = True,
        conformal_alpha: float = 0.10,
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

        step("conformal calibration")
        # Calibrated on a held-out slice of TRAINING data, stratified. Calibrating on rows the model
        # fitted would tune the quantile to memorised predictions and the guarantee would be
        # vacuous; calibrating on test would be peeking.
        from sklearn.model_selection import train_test_split

        try:
            _, X_conf, _, y_conf = train_test_split(
                ds.X_train, ds.y_train, test_size=0.2, stratify=ds.y_train, random_state=SEED
            )
            self.conformal = MondrianConformal(alpha=conformal_alpha).fit(
                supervised.attack_scores(self.supervised_model, X_conf), y_conf.to_numpy()
            )
        except ValueError:
            # Too few rows in some class to stratify. Better to have no conformal predictor than a
            # miscalibrated one claiming a guarantee it cannot keep.
            self.conformal = None

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

        # Conformal abstention: an ambiguous or empty prediction set means the model declines to
        # commit, which is a different statement from a low score and routes to human review.
        if self.conformal is not None:
            sets = self.conformal.predict_sets(p_attack)
            abstains = sets.abstains
            set_labels = sets.as_strings()
        else:
            abstains = np.zeros(len(X), dtype=bool)
            set_labels = [[] for _ in range(len(X))]

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
                "conformal_abstains": abstains,
                "conformal_set": set_labels,
            },
            index=X.index,
        )

    def ranked_importances(self, top: int = 8) -> list[tuple[str, float]]:
        """Top global feature importances, computed once and cached.

        Importances are a property of the fitted model, not of the row being scored. Recomputing
        them per row made the alert path O(rows x features) and pinned replay at 38 flows/s; caching
        took it to 670.
        """
        if self._ranked_importances is None:
            importances = self._importances()
            self._ranked_importances = (
                sorted(importances.items(), key=lambda kv: kv[1], reverse=True)[:16] if importances else []
            )
        return self._ranked_importances[:top]

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
                "conformal": self.conformal,
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
        det.conformal = blob.get("conformal")
        det._feature_names = blob["feature_names"]
        det._ranked_importances = None

        meta_path = directory / "metadata.json"
        if meta_path.exists():
            det.metadata = DetectorMetadata(**json.loads(meta_path.read_text(encoding="utf-8")))
        return det
