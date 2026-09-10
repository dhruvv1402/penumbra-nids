"""Novelty detectors. Every one of these is fitted on BENIGN TRAFFIC ONLY.

That constraint is the whole point. A supervised classifier trained on families {A..I} has no
representation for family J, so it assigns J to whichever known class it resembles - and when J
resembles benign traffic more than any attack class it answers `normal`, confidently. A detector
that has only ever seen benign traffic cannot make that particular mistake: everything unfamiliar
looks unfamiliar.

Three detectors with different notions of "unusual", because they fail differently:

  Autoencoder   reconstruction error - "I cannot compress this the way I compress normal traffic"
  IsolationForest  how few splits it takes to isolate a point - "this sits alone"
  Mahalanobis   distance from the benign centroid under the benign covariance - "this is far out"

All three return a score where HIGHER MEANS MORE NOVEL, on whatever scale they like. Making those
scales comparable is `ensemble.py`'s job, and it does it by rank rather than by rescaling, because
reconstruction error and `decision_function` output are not commensurable quantities.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.neural_network import MLPRegressor

from penumbra.seeds import SEED


@runtime_checkable
class NoveltyDetector(Protocol):
    name: str

    def fit(self, X_benign: np.ndarray) -> NoveltyDetector: ...

    def score(self, X: np.ndarray) -> np.ndarray:
        """Novelty score, higher = more unusual."""
        ...


class Autoencoder:
    """Undercomplete autoencoder; novelty = reconstruction error.

    42 -> 32 -> 16 -> 8 -> 16 -> 32 -> 42. This is a small MLP and we do not call it deep - the
    bottleneck is what matters, not the depth. Trained to reproduce benign traffic, it reconstructs
    benign flows well and unfamiliar ones badly, and the gap is the signal.

    Implemented with MLPRegressor rather than Keras deliberately: it trains in a couple of minutes
    on 8 CPU cores, it adds no dependency, and importing TensorFlow to fit a 5-layer MLP would cost
    600 MB for no benefit. The sequence head in Phase 7 is where a DL framework earns its place.
    """

    name = "autoencoder"

    def __init__(
        self,
        hidden: tuple[int, ...] = (32, 16, 8, 16, 32),
        *,
        max_iter: int = 60,
        seed: int = SEED,
    ) -> None:
        self.hidden = hidden
        self.model = MLPRegressor(
            hidden_layer_sizes=hidden,
            activation="relu",
            solver="adam",
            learning_rate_init=1e-3,
            batch_size=256,
            max_iter=max_iter,
            early_stopping=True,
            n_iter_no_change=5,
            validation_fraction=0.1,
            random_state=seed,
            verbose=False,
        )

    def fit(self, X_benign: np.ndarray) -> Autoencoder:
        self.model.fit(X_benign, X_benign)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        recon = self.model.predict(X)
        return np.asarray(np.mean((X - recon) ** 2, axis=1), dtype=float)


class IsolationForestDetector:
    """Isolation Forest; novelty = negated decision function.

    `contamination` is left at a nominal small value and never used as a threshold - thresholding
    happens once, downstream, on the fused score. Setting it here would bake an operating point
    into a component.
    """

    name = "iforest"

    def __init__(self, n_estimators: int = 200, *, seed: int = SEED) -> None:
        self.model = IsolationForest(
            n_estimators=n_estimators,
            max_samples="auto",
            contamination="auto",
            n_jobs=-1,
            random_state=seed,
        )

    def fit(self, X_benign: np.ndarray) -> IsolationForestDetector:
        self.model.fit(X_benign)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        # decision_function is high for inliers, so negate to make high = novel.
        return np.asarray(-self.model.decision_function(X), dtype=float)


class MahalanobisDetector:
    """Distance from the benign centroid under the benign covariance.

    Expected to be the weakest of the three, and kept anyway. Forty-odd correlated, heavy-tailed,
    non-Gaussian flow features produce a near-singular covariance matrix, so this needs help:
    PCA-whitening to a modest number of components first, then Ledoit-Wolf shrinkage on top. An
    ablation in which one detector is visibly worse is more informative than one in which three
    detectors look interchangeable.
    """

    name = "mahalanobis"

    def __init__(self, n_components: int = 15, *, seed: int = SEED) -> None:
        self.n_components = n_components
        self.pca = PCA(n_components=n_components, random_state=seed)
        self.cov = LedoitWolf(store_precision=True)
        self.mean_: np.ndarray | None = None

    def fit(self, X_benign: np.ndarray) -> MahalanobisDetector:
        n_comp = min(self.n_components, X_benign.shape[1], max(X_benign.shape[0] - 1, 1))
        self.pca = PCA(n_components=n_comp, random_state=SEED)
        Z = self.pca.fit_transform(X_benign)
        self.cov.fit(Z)
        self.mean_ = Z.mean(axis=0)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None:
            raise RuntimeError("MahalanobisDetector.score called before fit")
        Z = self.pca.transform(X)
        delta = Z - self.mean_
        precision = self.cov.get_precision()
        # Row-wise quadratic form without materialising an n x n matrix.
        return np.asarray(np.einsum("ij,jk,ik->i", delta, precision, delta), dtype=float)


def default_detectors() -> list[NoveltyDetector]:
    return [Autoencoder(), IsolationForestDetector(), MahalanobisDetector()]
