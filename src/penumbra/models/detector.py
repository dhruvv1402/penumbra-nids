"""The whole detector, as one object.

Everything the evaluation modules assemble piecemeal, packaged so it can be fitted once, saved, and
loaded by the API and the replay engine. This is what turns a set of experiments into a thing that
runs.

It owns the two heads, their separate preprocessing, the benign reference distributions, and the
thresholds - and it keeps them together, because a threshold without the reference it was fitted
against is meaningless and the two drifting apart is a whole class of deployment bug.

Fit order matters and is enforced:

  0. training rows split into a fit part and a stratified calibration slice S
  1. supervised pipeline on the fit part only
  2. benign-only pipeline on BENIGN fit-part rows only
  3. novelty detectors on those rows
  4. percentile references from S's benign rows
  5. thresholds and the conformal layer from S, never from rows a model trained on, never from test
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

from penumbra import __version__
from penumbra.data.loaders.base import Dataset
from penumbra.features.preprocess import assert_benign_only_fit, benign_only_pipeline
from penumbra.models import supervised
from penumbra.models.conformal import MondrianConformal
from penumbra.models.fusion import OrGate
from penumbra.models.novelty.ensemble import NoveltyEnsemble
from penumbra.seeds import SEED, seed_everything

CALIBRATION_FRACTION = 0.2
# The novelty head's MLP autoencoder costs O(rows) per epoch; on CICIDS's ~800k benign training
# flows it would dominate the whole fit. A seeded sample of this size is plenty to learn "normal",
# and `n_benign_fit` records what was actually used.
NOVELTY_MAX_FIT_ROWS = 200_000


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
    # Set by `rebaselined`: where this detector's idea of normal came from.
    baseline: dict[str, Any] | None = None
    # Defaults describe every detector saved before E9a, so old metadata.json files still load and
    # still say truthfully how they were calibrated.
    calibration: str = "in_sample"
    n_calibration_rows: int | None = None
    n_supervised_fit_rows: int | None = None
    family_model_name: str | None = None
    # Set by the registry on registration, so an alert can be traced to the version that raised it.
    registry_version: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def calibration_split(
    families: pd.Series, fraction: float, *, seed: int = SEED, keep_in_fit: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Positional (fit, calibration) indices, stratified by family by hand.

    By hand for the reason `canary.live_split` is: a library stratifier refuses a family with one
    row. Here such a family simply stays in the fit part, so no rare family vanishes from training
    to calibrate a threshold that does not need it.
    """
    values = families.astype(str).to_numpy()
    eligible = np.ones(len(values), dtype=bool)
    if keep_in_fit is not None:
        eligible[np.asarray(keep_in_fit, dtype=int)] = False
    rng = np.random.default_rng(seed)
    calibration: list[int] = []
    for label in sorted(set(values)):
        idx = np.flatnonzero((values == label) & eligible)
        rng.shuffle(idx)
        calibration.extend(idx[: int(round(len(idx) * fraction))].tolist())
    cal = np.sort(np.asarray(calibration, dtype=int))
    in_fit = np.ones(len(values), dtype=bool)
    in_fit[cal] = False
    return np.flatnonzero(in_fit), cal


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
        family_model_name: str | None = None,
        fit_family_model: bool = True,
        conformal_alpha: float = 0.10,
        calibration: str = "held_out",
        keep_in_fit: np.ndarray | None = None,
        quarantined: list[str] | None = None,
        on_progress: Any = None,
    ) -> PenumbraDetector:
        """Fit both heads, the thresholds and the conformal layer.

        `calibration="held_out"` (the default) fits the supervised model on training rows minus a
        stratified slice S and fits every threshold and the conformal predictor on S. The
        supervised model has never seen S, so its scores there are the scores it would give
        unseen traffic.

        `calibration="in_sample"` is the behaviour every number published before E9a was measured
        with. The novelty head's calibration rows were held out, but the supervised head trained on
        every row, including the benign rows its threshold is a quantile of. A forest scores its
        own training rows lower than unseen ones, so that threshold sits too low; for a single
        unpruned tree the in-sample scores are all 0 and every row fires. Kept so the commands
        behind E7, E8 and the correlation run still reproduce what they published.

        `keep_in_fit` holds positions that must train the model and never land in S (analyst
        feedback rows: a verdict is a training label, not a calibration sample).
        """
        if calibration not in {"held_out", "in_sample"}:
            raise ValueError(f"calibration must be 'held_out' or 'in_sample', got {calibration!r}")
        seed_everything()
        family_model_name = family_model_name or supervised.family_model_for(model_name)

        def step(msg: str) -> None:
            if on_progress:
                on_progress(msg)

        if calibration == "held_out":
            fit_pos, cal_pos = calibration_split(ds.fam_train, CALIBRATION_FRACTION, keep_in_fit=keep_in_fit)
            X_fit, y_fit = ds.X_train.iloc[fit_pos], ds.y_train.iloc[fit_pos]
            X_cal, y_cal = ds.X_train.iloc[cal_pos], ds.y_train.iloc[cal_pos]
        else:
            X_fit, y_fit = ds.X_train, ds.y_train

        step("supervised head")
        self.supervised_model = supervised.build(model_name, ds, n_classes=2, balanced=True)
        self.supervised_model.fit(X_fit, y_fit)
        clf = self.supervised_model.named_steps["clf"]
        if hasattr(clf, "fit_importances") and calibration == "held_out":
            # A model without feature_importances_ (the E9 ensemble) gets permutation importances,
            # measured on rows it never trained on.
            step("permutation importances")
            clf.fit_importances(self.supervised_model.named_steps["prep"].transform(X_cal), y_cal.to_numpy())

        if fit_family_model:
            step("family classifier")
            self.family_model, self.family_encoder = supervised.fit_multiclass(
                family_model_name, ds, balanced=True
            )

        step("benign-only pipeline")
        if calibration == "held_out":
            benign_fit = X_fit.loc[y_fit == 0]
            benign_cal = X_cal.loc[y_cal == 0]
            # The contract the novelty head's whole claim rests on.
            assert_benign_only_fit(ds, benign_fit)
            assert_benign_only_fit(ds, benign_cal)
        else:
            benign = ds.X_train.loc[ds.y_train == 0]
            assert_benign_only_fit(ds, benign)
            rng = np.random.default_rng(SEED)
            order = rng.permutation(len(benign))
            n_cal = int(len(benign) * CALIBRATION_FRACTION)
            cal_idx, fit_idx = order[:n_cal], order[n_cal:]
            benign_fit = benign.iloc[fit_idx]
            benign_cal = benign.iloc[cal_idx]

        if len(benign_fit) > NOVELTY_MAX_FIT_ROWS:
            benign_fit = benign_fit.sample(NOVELTY_MAX_FIT_ROWS, random_state=SEED)
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
        # Calibrating on rows the model fitted tunes the quantile to memorised predictions and the
        # guarantee is vacuous; calibrating on test would be peeking. Held out, it is S itself.
        try:
            if calibration == "held_out":
                X_conf, y_conf = X_cal, y_cal
            else:
                from sklearn.model_selection import train_test_split

                _, X_conf, _, y_conf = train_test_split(
                    ds.X_train, ds.y_train, test_size=0.2, stratify=ds.y_train, random_state=SEED
                )
            self.conformal = MondrianConformal(alpha=conformal_alpha).fit(
                supervised.attack_scores(self.supervised_model, X_conf), y_conf.to_numpy()
            )
        except ValueError:
            # Too few rows in some class. Better to have no conformal predictor than a miscalibrated
            # one claiming a guarantee it cannot keep.
            self.conformal = None

        self._feature_names = list(ds.feature_names)
        self.metadata = DetectorMetadata(
            model_name=model_name,
            version=__version__,
            dataset=ds.name,
            trained_at=datetime.now(UTC).isoformat(),
            n_train_rows=len(ds.X_train),
            n_benign_fit=len(benign_fit),
            n_benign_calibration=len(benign_cal),
            target_fpr=self.target_fpr,
            supervised_threshold=self.gate.supervised_threshold,
            novelty_threshold=self.gate.novelty_threshold,
            features=self._feature_names,
            quarantined_features=sorted(quarantined or []),
            calibration=calibration,
            n_calibration_rows=len(X_conf) if self.conformal is not None else 0,
            n_supervised_fit_rows=len(X_fit),
            family_model_name=family_model_name if fit_family_model else None,
        )
        return self

    # --- re-baseline -----------------------------------------------------------------------------

    def rebaselined(
        self,
        X_fit: pd.DataFrame,
        X_cal: pd.DataFrame,
        *,
        target_fpr: float | None = None,
        source: str = "local benign traffic",
    ) -> PenumbraDetector:
        """A copy whose idea of normal is learnt from a new network's benign traffic.

        What changes, and why each piece has to:

          * **the novelty head** (its preprocessing, the three detectors, their benign references),
            refitted on `X_fit` and referenced on `X_cal`. "Unusual" is only meaningful against the
            network being watched; against the training testbed, everything real is unusual.
          * **both thresholds**, from `X_cal` at the same total budget. The supervised head's scores
            shift on a new network too - on the lab capture it fired on 519 of 912 benign flows at
            its UNSW threshold - so an operating point fitted elsewhere does not hold here.
          * **the benign side of the conformal layer**, re-fitted on `X_cal`. Mondrian calibration is
            per class, so benign rows alone are enough to re-fit that class honestly.

        What does not change: the supervised model and the family model. Benign traffic carries no
        information about attacks, so nothing learnt about attacks is touched, and the attack-side
        conformal quantile stays as calibrated.

        `X_fit` and `X_cal` must be disjoint and must be benign. That second condition is an
        assumption about the capture window, not something this method can check - a baseline
        recorded while an intruder was active teaches the novelty head that the intruder is normal
        (THREAT_MODEL T9). The CLI reports how much of the window the champion itself flags.
        """
        if self.supervised_model is None or self.novelty is None or self.gate is None:
            raise RuntimeError("detector is not fitted")
        if len(X_fit) == 0 or len(X_cal) == 0:
            raise ValueError("need benign rows to fit and to calibrate on")
        if set(X_fit.index) & set(X_cal.index):
            raise ValueError("fit and calibration rows overlap; the thresholds would be optimistic")
        import copy

        target = self.target_fpr if target_fpr is None else target_fpr
        categorical = [c for c in X_fit.columns if not pd.api.types.is_numeric_dtype(X_fit[c])]
        numeric = [c for c in X_fit.columns if c not in categorical]
        zeros = pd.Series(np.zeros(len(X_fit), dtype=int), index=X_fit.index)
        normal = pd.Series(["normal"] * len(X_fit), index=X_fit.index)
        local = Dataset(
            "local", X_fit, zeros, normal, X_fit, zeros, normal, categorical=categorical, numeric=numeric
        )

        new = copy.copy(self)
        new.target_fpr = target
        new.novelty_prep = benign_only_pipeline(local)
        Z_fit = new.novelty_prep.fit_transform(X_fit)
        Z_cal = new.novelty_prep.transform(X_cal)
        new.novelty = NoveltyEnsemble().fit(Z_fit)
        new.novelty.calibrate_on(Z_cal)

        benign_p = supervised.attack_scores(self.supervised_model, X_cal)
        benign_n = new.novelty.score(Z_cal, how="max")
        new.gate = OrGate.fit(benign_p, benign_n, total_fpr=target, use_novelty=True)
        if self.conformal is not None:
            from penumbra.models.conformal import BENIGN

            new.conformal = self.conformal.recalibrated(BENIGN, benign_p)

        if self.metadata is not None:
            new.metadata = copy.copy(self.metadata)
            new.metadata.trained_at = datetime.now(UTC).isoformat()
            new.metadata.n_benign_fit = len(X_fit)
            new.metadata.n_benign_calibration = len(X_cal)
            new.metadata.target_fpr = target
            new.metadata.supervised_threshold = new.gate.supervised_threshold
            new.metadata.novelty_threshold = new.gate.novelty_threshold
            new.metadata.baseline = {
                "mode": "full",
                "source": source,
                "fit_rows": len(X_fit),
                "calibration_rows": len(X_cal),
                "previous_supervised_threshold": self.gate.supervised_threshold,
                "previous_novelty_threshold": self.gate.novelty_threshold,
            }
        return new

    def rethresholded(
        self,
        X_cal: pd.DataFrame,
        *,
        target_fpr: float | None = None,
        source: str = "recent benign traffic",
    ) -> PenumbraDetector:
        """A copy with the same heads and a new operating point, for a network that has drifted.

        Re-fits both thresholds and the benign conformal quantile on `X_cal`; the novelty head,
        its references and every model are untouched. E8 (EXPERIMENTS.md) is why this exists
        beside `rebaselined`: on NSL-KDD's shifted test split, moving the thresholds brought a
        realised 10.2% FPR back to 1.06%, stably from 500 rows, while re-learning normal from the
        same rows did no better on unseen attacks and was unstable below ~1,000 calibration rows.
        Re-learning normal is for a network the model has never seen; this is for the same network,
        moved.

        It cannot rescue traffic outside everything the novelty head was referenced on: those flows
        all score above the whole benign reference, their percentiles saturate at 1.0, and no
        threshold separates them. The held-out check in `eval.rebaseline` (R2) refuses that case.
        """
        if self.supervised_model is None or self.novelty is None or self.gate is None:
            raise RuntimeError("detector is not fitted")
        if len(X_cal) == 0:
            raise ValueError("need benign rows to calibrate on")
        import copy

        target = self.target_fpr if target_fpr is None else target_fpr
        new = copy.copy(self)
        new.target_fpr = target
        benign_p = supervised.attack_scores(self.supervised_model, X_cal)
        benign_n = self.novelty.score(self.novelty_prep.transform(X_cal), how="max")
        new.gate = OrGate.fit(benign_p, benign_n, total_fpr=target, use_novelty=True)
        if self.conformal is not None:
            from penumbra.models.conformal import BENIGN

            new.conformal = self.conformal.recalibrated(BENIGN, benign_p)
        if self.metadata is not None:
            new.metadata = copy.copy(self.metadata)
            new.metadata.trained_at = datetime.now(UTC).isoformat()
            new.metadata.target_fpr = target
            new.metadata.supervised_threshold = new.gate.supervised_threshold
            new.metadata.novelty_threshold = new.gate.novelty_threshold
            new.metadata.baseline = {
                "mode": "thresholds",
                "source": source,
                "fit_rows": 0,
                "calibration_rows": len(X_cal),
                "previous_supervised_threshold": self.gate.supervised_threshold,
                "previous_novelty_threshold": self.gate.novelty_threshold,
            }
        return new

    # --- score -----------------------------------------------------------------------------------

    def score(self, X: pd.DataFrame) -> pd.DataFrame:
        """Score a batch. Returns the raw numbers; `alerts()` turns them into Alert objects."""
        if self.supervised_model is None or self.novelty is None or self.gate is None:
            raise RuntimeError("detector is not fitted")

        p_attack = supervised.attack_scores(self.supervised_model, X)
        # Each detector runs once: the fused score and the agreement count read the same percentiles.
        raw = self.novelty.raw_scores(self.novelty_prep.transform(X))
        return self.assemble(X.index, p_attack, raw, self.family_predictor(X))

    def family_predictor(self, X: pd.DataFrame) -> Any:
        """Family labels for chosen row positions of `X`, or None without a family model."""
        if self.family_model is None:
            return None

        def predict(rows: np.ndarray) -> np.ndarray:
            subset = X if len(rows) == len(X) else X.iloc[rows]
            return supervised.predict_families(self.family_model, self.family_encoder, subset)

        return predict

    def assemble(
        self,
        index: pd.Index,
        p_attack: np.ndarray,
        raw_novelty: dict[str, np.ndarray],
        family_predictor: Any,
    ) -> pd.DataFrame:
        """The scored frame from the heads' raw outputs, however they were computed.

        Split out of `score` so the compiled path (`models.compiled`) produces its frame through the
        same fusion, gate and conformal code rather than a copy of it.

        Families are predicted only for rows that become alerts (either head fired, or the conformal
        layer abstained). Nothing reads a family anywhere else - the alert builder drops rows that
        are neither - and on live traffic, which is overwhelmingly benign, that skips the second
        forest for almost every flow.
        """
        if self.novelty is None or self.gate is None:
            raise RuntimeError("detector is not fitted")
        pcts = self.novelty.percentiles_from_raw(raw_novelty)
        novelty_raw = self.novelty.fuse(pcts, how="max")
        agreement = self.novelty.agreement_from(pcts, percentile=0.99)
        which = self.gate.which_fired(p_attack, novelty_raw)
        n = len(index)

        # Conformal abstention: an ambiguous or empty prediction set means the model declines to
        # commit, which is a different statement from a low score and routes to human review.
        if self.conformal is not None:
            sets = self.conformal.predict_sets(p_attack)
            abstains = sets.abstains
            set_labels = sets.as_strings()
        else:
            abstains = np.zeros(n, dtype=bool)
            set_labels = [[] for _ in range(n)]

        families: list[str | None] = [None] * n
        alertable = np.flatnonzero((which > 0) | abstains)
        if family_predictor is not None and len(alertable):
            for row, f in zip(alertable.tolist(), family_predictor(alertable), strict=True):
                families[row] = None if str(f).lower() in {"normal", "benign"} else str(f)

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
            index=index,
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
