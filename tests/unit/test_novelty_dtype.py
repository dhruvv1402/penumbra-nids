"""The novelty detectors must not overflow on float32 input with CICIDS-sized values.

CICIDS loads as float32, and its byte-rate columns reach 1e20 after robust scaling of a near-zero
IQR. Squared in float32 that is inf; Ledoit-Wolf returned NaN and the CICIDS detector fit failed.
"""

from __future__ import annotations

import numpy as np

from penumbra.models.novelty.detectors import Autoencoder, MahalanobisDetector


def heavy(n: int = 400, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 6))
    X[:, 0] *= 1e20  # a flow-bytes/s column after scaling by a tiny IQR
    return X.astype(np.float32)


def test_mahalanobis_fits_and_scores_float32_without_nan() -> None:
    X = heavy()
    s = MahalanobisDetector(n_components=4).fit(X).score(X)
    assert np.isfinite(s).all()


def test_autoencoder_scores_are_finite_on_float32() -> None:
    X = heavy()
    X[:, 0] /= 1e19  # keep the MLP trainable; the point is the dtype path, not the scale
    s = Autoencoder(max_iter=5).fit(X).score(X)
    assert np.isfinite(s).all()
