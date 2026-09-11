"""The sequence head — a 1D-CNN into a bidirectional GRU over per-host flow windows.

This is the only deep model in the project, and it is here for a reason the trees cannot cover. A
per-flow classifier sees one connection and asks whether it looks unusual. A slow port sweep has no
unusual connection in it — each flow is a single, ordinary, short TCP attempt. What makes it an
attack is the *shape of the sequence*: fifteen ordinary flows to fifteen different destinations.
The trees cannot see that, because the row they are given does not contain it.

## Architecture, and why each piece

```
(K, d) window  ->  + mask channel  ->  Conv1D x2 (causal)  ->  BiGRU  ->  Dense  ->  p(attack)
```

**Causal convolutions.** `padding="causal"` means position *i*'s receptive field is positions
`i-2..i`, never `i+1`. Within the window this is belt-and-braces — the whole window is already
strictly causal with respect to the flow being classified — but it means the same weights would
work unchanged in a streaming setting.

**A bidirectional GRU, which looks like a leak and is not.** The backward pass reads from the
current flow towards the oldest flow in the window. Every position it visits is in the *past* of
the flow being classified, and the whole window exists at inference time. Bidirectionality over a
trailing window is not lookahead; bidirectionality over a window centred on the current flow would
be, which is why `features/windows.py` never centres one.

**A mask channel instead of a `Masking` layer.** Short histories are zero-padded at the front, and
`Conv1D` does not consume Keras masks — the combination is a well-known source of silent wrongness.
Appending the mask as a `d+1`th channel is explicit, works through every layer, and lets the
network learn that "no history" is itself informative.

## What this is not

It is not the novelty head and it does not replace it. It is a supervised model, so it learns the
sequence shapes of attacks it was *trained on*, and it will label an unseen family's sequence
`normal` with the same confidence the per-flow model does. The claim being tested here is narrower
and checkable: **does sequence context buy recall that per-flow features cannot?**

`docs/EVALUATION.md` records the answer, including if it is no. Trees beat deep models on tabular
data of this size and shape (Grinsztajn et al., NeurIPS 2022) and the interesting question is
whether the *temporal* axis is the exception.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from penumbra.features.windows import Sequences
from penumbra.seeds import SEED

# Keras is noisy on import and most of it is about hardware we do not have.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

# Scaled features are clipped to this many robust deviations. Generous enough to keep the shape of
# a genuinely extreme flow, tight enough that one row cannot saturate the network.
CLIP_SIGMAS = 10.0

# A validation split needs at least this many of the rarer class for ROC-AUC to mean anything.
# Below it, Keras reports 0.5 and EarlyStopping has nothing to monitor.
MIN_VALIDATION_MINORITY = 500


@dataclass
class SequenceConfig:
    """Deliberately small.

    8 CPU cores and no GPU. A model that takes an hour per epoch cannot be tuned, and an untuned
    deep model losing to XGBoost demonstrates nothing except that it was untuned. These sizes train
    in single-digit minutes on a few hundred thousand windows, which is what makes the comparison
    in EVALUATION.md an actual experiment rather than one run.
    """

    conv_filters: int = 64
    conv_kernel: int = 3
    gru_units: int = 48
    dense_units: int = 64
    dropout: float = 0.3
    learning_rate: float = 1e-3
    batch_size: int = 256
    epochs: int = 12
    patience: int = 3
    validation_fraction: float = 0.15


@dataclass
class TrainingHistory:
    epochs_run: int = 0
    best_val_auc: float = 0.0
    stopped_early: bool = False
    collapsed_epochs: int = 0
    validation_split: str = "temporal"
    validation_minority: int = 0
    curve: list[dict[str, float]] = field(default_factory=list)

    @property
    def collapsed(self) -> bool:
        """Did training diverge to a constant output?

        Worth its own flag because early stopping hides it perfectly: best weights are restored,
        the run exits cleanly, and a one-epoch model gets reported as the architecture's verdict.
        """
        return self.collapsed_epochs > 0

    def summary(self) -> str:
        stop = "early-stopped" if self.stopped_early else "ran to the epoch cap"
        line = f"  {self.epochs_run} epochs ({stop}), best validation ROC-AUC {self.best_val_auc:.4f}"
        line += (
            f"{chr(10)}  validation split: {self.validation_split}, "
            f"{self.validation_minority:,} rows in the rarer class"
        )
        if self.collapsed:
            line += (
                chr(10) + "  WARNING: validation AUC sat at 0.500 for "
                f"{self.collapsed_epochs} epoch(s) - "
                "training diverged to a constant output. Any result below describes that, not the "
                "architecture."
            )
        return line


def _require_keras() -> Any:
    try:
        import keras
    except ImportError as exc:  # pragma: no cover - exercised by the extra being absent
        raise ImportError(
            "The sequence head needs the `dl` extra: uv sync --extra dl. It is optional on purpose "
            "- TensorFlow is ~600 MB and nothing else in Penumbra needs it."
        ) from exc
    return keras


def build_model(length: int, n_features: int, config: SequenceConfig | None = None) -> Any:
    """The Keras model. `n_features` excludes the mask channel, which is added here."""
    keras = _require_keras()
    config = config or SequenceConfig()

    inputs = keras.Input(shape=(length, n_features + 1), name="window")
    x = keras.layers.Conv1D(config.conv_filters, config.conv_kernel, padding="causal", activation="relu")(
        inputs
    )
    x = keras.layers.Conv1D(config.conv_filters, config.conv_kernel, padding="causal", activation="relu")(x)
    x = keras.layers.Bidirectional(keras.layers.GRU(config.gru_units))(x)
    x = keras.layers.Dropout(config.dropout)(x)
    x = keras.layers.Dense(config.dense_units, activation="relu")(x)
    x = keras.layers.Dropout(config.dropout)(x)
    outputs = keras.layers.Dense(1, activation="sigmoid", name="p_attack")(x)

    model = keras.Model(inputs, outputs, name="penumbra_sequence")
    model.compile(
        # Gradient clipping is the second line of defence behind the robust scaler. Recurrent
        # layers on bursty traffic produce occasional enormous gradients, and one of those is
        # enough to move the weights somewhere the model never recovers from.
        optimizer=keras.optimizers.Adam(config.learning_rate, clipnorm=1.0),
        loss="binary_crossentropy",
        metrics=[keras.metrics.AUC(name="auc"), keras.metrics.AUC(name="pr_auc", curve="PR")],
    )
    return model


class SequenceDetector:
    """Fit / score / save / load, with the scaler fitted on real flows only.

    Fitting the scaler over the padded tensor would let the zero-padding — which is an artifact of
    how many flows a host happened to have, not of the traffic — drag every column's mean towards
    zero, and by a different amount for busy and quiet hosts.
    """

    def __init__(self, config: SequenceConfig | None = None) -> None:
        self.config = config or SequenceConfig()
        self.model: Any = None
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None
        self.features: list[str] = []
        self.history = TrainingHistory()

    # -- internals -------------------------------------------------------------------------

    def _fit_scaler(self, seq: Sequences) -> None:
        """Median and IQR, not mean and standard deviation.

        CICIDS2017's flow features are violently heavy-tailed: `Total TCP Flow Time` reaches 7.2e9
        and several columns have a maximum more than 200 standard deviations from their own mean.
        Z-scaling those leaves a handful of rows at +200 and everything else squashed against zero,
        and +200 through two convolutions into a GRU saturates the network.

        That is not hypothetical. The first run of this experiment trained to validation ROC-AUC
        0.9999 in epoch 1, then collapsed to a constant output - val AUC exactly 0.500 for every
        remaining epoch. Early stopping dutifully restored the epoch-1 weights and the experiment
        reported a number, which is the dangerous version of this failure: it does not crash, it
        just quietly reports a one-epoch model as though it were the architecture's verdict.
        """
        real = seq.X[seq.mask]
        median = np.median(real, axis=0).astype(np.float32)
        q75, q25 = np.percentile(real, [75, 25], axis=0)
        iqr = (q75 - q25).astype(np.float32)
        # A column with a degenerate IQR (constant, or over 75% identical) falls back to std, and
        # then to 1.0 - a zero divisor produces NaN through the whole network.
        std = real.std(axis=0, dtype=np.float64).astype(np.float32)
        scale = np.where(iqr > 1e-8, iqr, std)
        self.mean_ = median
        self.scale_ = np.where(scale > 1e-8, scale, 1.0).astype(np.float32)

    def _prepare(self, seq: Sequences) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("SequenceDetector is not fitted")
        # One allocation. Written naively this is `subtract`, `divide`, `where`, `nan_to_num` and
        # `concatenate` - five full-size copies of a tensor that is already 2 GB, which is how you
        # turn a 4 GB job into a 12 GB one and get killed by the OS mid-experiment.
        n, k, d = seq.X.shape
        out = np.zeros((n, k, d + 1), dtype=np.float32)
        view = out[:, :, :d]
        np.subtract(seq.X, self.mean_, out=view)
        np.divide(view, self.scale_, out=view)
        np.nan_to_num(view, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        # Robust scaling narrows the tails; it does not remove them. A flow at 4,000 IQRs still
        # exists, and it only has to reach the network once to saturate it.
        np.clip(view, -CLIP_SIGMAS, CLIP_SIGMAS, out=view)
        # Padded positions must stay exactly zero. Scaling moved them off it, and a padded position
        # carrying the negative of the training mean is not padding any more.
        view *= seq.mask[..., None]
        out[:, :, d] = seq.mask
        return out

    def _validation_cut(self, y: np.ndarray) -> tuple[int, str]:
        """Where to split for validation, and by what rule.

        A TEMPORAL split is the right default. The sequences arrive in time order, so holding out
        the tail is the only split that answers "does this work on traffic that arrives after what
        it was trained on" - the only question deployment asks.

        But a temporal tail can be degenerate, and on CICIDS2017 it is. The last 15% of Mon-Wed in
        time order has an attack rate of **0.0001** - four attacks in 44,638 windows, because
        Wednesday evening is quiet after the DoS traffic stops. Keras computes ROC-AUC of 0.5 on a
        validation set that is effectively one class, EarlyStopping monitors that, and the model
        trains with no usable stopping signal at all. The run exits cleanly reporting
        `val_auc 0.500`, which reads like divergence and is actually a broken split.

        So the tail is widened until the minority class is large enough to support an AUC, and only
        if no tail up to half the data works does it fall back to a stratified shuffle. Which rule
        was used is recorded on the history, because a random split answers a weaker question and
        the report should not silently substitute one for the other.
        """
        n = len(y)
        for fraction in (self.config.validation_fraction, 0.25, 0.35, 0.5):
            cut = int(n * (1.0 - fraction))
            tail = y[cut:]
            minority = min(int((tail == 1).sum()), int((tail == 0).sum()))
            if minority >= MIN_VALIDATION_MINORITY:
                return cut, "temporal" if fraction == self.config.validation_fraction else "temporal-widened"
        return int(n * (1.0 - self.config.validation_fraction)), "random"

    # -- api -------------------------------------------------------------------------------

    def fit(self, seq: Sequences, *, verbose: int = 0) -> SequenceDetector:
        keras = _require_keras()
        keras.utils.set_random_seed(SEED)

        self.features = list(seq.features)
        self._fit_scaler(seq)
        X = self._prepare(seq)
        y = seq.y.astype(np.float32)

        cut, split_kind = self._validation_cut(y)
        if split_kind == "random":
            # Stratified shuffle, used only when no temporal tail works. Recorded, not hidden.
            rng = np.random.default_rng(SEED)
            order = rng.permutation(len(X))
            X, y = X[order], y[order]
        X_fit, X_val = X[:cut], X[cut:]
        y_fit, y_val = y[:cut], y[cut:]

        n_pos = float(y_fit.sum())
        n_neg = float(len(y_fit) - n_pos)
        class_weight = (
            {0: 1.0, 1: n_neg / n_pos} if n_pos > 0 and n_neg > 0 else None  # noqa: SIM108
        )

        self.model = build_model(seq.length, seq.X.shape[2], self.config)
        stopper = keras.callbacks.EarlyStopping(
            monitor="val_auc", mode="max", patience=self.config.patience, restore_best_weights=True
        )
        fitted = self.model.fit(
            X_fit,
            y_fit,
            validation_data=(X_val, y_val) if len(X_val) else None,
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            class_weight=class_weight,
            callbacks=[stopper],
            verbose=verbose,
        )

        val_auc = fitted.history.get("val_auc", [])
        # A validation AUC pinned at 0.5 means the model emits one constant value. EarlyStopping
        # restores the best weights and the run looks fine, so this has to be surfaced explicitly
        # or a collapsed model gets reported as an architecture result.
        collapsed = sum(1 for a in val_auc[1:] if abs(float(a) - 0.5) < 1e-3)
        self.history = TrainingHistory(
            epochs_run=len(fitted.history.get("loss", [])),
            best_val_auc=float(max(val_auc)) if val_auc else 0.0,
            stopped_early=stopper.stopped_epoch > 0,
            collapsed_epochs=collapsed,
            validation_split=split_kind,
            validation_minority=int(min((y_val == 1).sum(), (y_val == 0).sum())),
            curve=[
                {"epoch": i + 1, "loss": float(loss), "val_auc": float(auc)}
                for i, (loss, auc) in enumerate(zip(fitted.history.get("loss", []), val_auc, strict=False))
            ],
        )
        return self

    def score(self, seq: Sequences) -> np.ndarray:
        """P(attack) for the current flow of each window."""
        if self.model is None:
            raise RuntimeError("SequenceDetector is not fitted")
        X = self._prepare(seq)
        return np.asarray(self.model.predict(X, batch_size=1024, verbose=0)).ravel()

    def save(self, directory: Path) -> Path:
        if self.model is None or self.mean_ is None or self.scale_ is None:
            raise RuntimeError("SequenceDetector is not fitted")
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.model.save(directory / "sequence.keras")
        np.savez(
            directory / "sequence_scaler.npz",
            mean=self.mean_,
            scale=self.scale_,
            features=np.array(self.features, dtype=object),
        )
        return directory

    @classmethod
    def load(cls, directory: Path, config: SequenceConfig | None = None) -> SequenceDetector:
        keras = _require_keras()
        directory = Path(directory)
        detector = cls(config)
        detector.model = keras.models.load_model(directory / "sequence.keras")
        blob = np.load(directory / "sequence_scaler.npz", allow_pickle=True)
        detector.mean_ = blob["mean"]
        detector.scale_ = blob["scale"]
        detector.features = list(blob["features"])
        return detector
