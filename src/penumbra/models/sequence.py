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
    curve: list[dict[str, float]] = field(default_factory=list)

    def summary(self) -> str:
        stop = "early-stopped" if self.stopped_early else "ran to the epoch cap"
        return f"  {self.epochs_run} epochs ({stop}), best validation ROC-AUC {self.best_val_auc:.4f}"


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
        optimizer=keras.optimizers.Adam(config.learning_rate),
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
        real = seq.X[seq.mask]
        self.mean_ = real.mean(axis=0, dtype=np.float64).astype(np.float32)
        std = real.std(axis=0, dtype=np.float64).astype(np.float32)
        # A constant column would divide by zero and produce NaN through the whole network.
        self.scale_ = np.where(std > 1e-8, std, 1.0).astype(np.float32)

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
        # Padded positions must stay exactly zero. Scaling moved them off it, and a padded position
        # carrying the negative of the training mean is not padding any more.
        view *= seq.mask[..., None]
        out[:, :, d] = seq.mask
        return out

    # -- api -------------------------------------------------------------------------------

    def fit(self, seq: Sequences, *, verbose: int = 0) -> SequenceDetector:
        keras = _require_keras()
        keras.utils.set_random_seed(SEED)

        self.features = list(seq.features)
        self._fit_scaler(seq)
        X = self._prepare(seq)
        y = seq.y.astype(np.float32)

        # A TEMPORAL validation split, not a random one. The sequences arrive in time order, so
        # holding out the tail is the only split that answers "does this work on traffic that
        # arrives after what it was trained on" - which is the only question deployment asks.
        cut = int(len(X) * (1.0 - self.config.validation_fraction))
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
        self.history = TrainingHistory(
            epochs_run=len(fitted.history.get("loss", [])),
            best_val_auc=float(max(val_auc)) if val_auc else 0.0,
            stopped_early=stopper.stopped_epoch > 0,
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
