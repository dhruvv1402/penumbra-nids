"""The compiled detector must score exactly as the detector it was compiled from."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from penumbra.alerts.builder import alerts_with_positions
from penumbra.models import compiled as compiled_mod
from penumbra.models.compiled import CompileRefused, compile_detector, parity
from penumbra.models.detector import PenumbraDetector
from tests.unit.test_registry import tiny_dataset


@pytest.fixture(scope="module")
def ds():
    return tiny_dataset(n=900)


@pytest.fixture(scope="module")
def detector(ds) -> PenumbraDetector:
    return PenumbraDetector(target_fpr=0.05).fit(ds)


@pytest.fixture(scope="module")
def compiled(detector, ds):
    return compile_detector(detector, ds.X_test)


def test_parity_is_exact_on_every_decision(compiled, detector, ds) -> None:
    assert compiled.parity["decisions_changed"] == 0
    assert compiled.compiled == ["supervised", "iforest"]
    forced = compiled._score(ds.X_test, force=True)
    reference = detector.score(ds.X_test)
    np.testing.assert_allclose(forced["p_attack"], reference["p_attack"], atol=1e-12)
    np.testing.assert_array_equal(forced["fired"], reference["fired"])
    assert forced["family"].tolist() == reference["family"].tolist()


def test_limits_are_measured_and_reported(compiled) -> None:
    report = compiled.parity
    assert set(report["flat_up_to_rows"]) == {"supervised", "iforest"}
    for name, timings in report["timings"].items():
        assert timings, name
        assert all({"rows", "flat_ms", "sklearn_ms"} <= set(t) for t in timings)
        assert report["flat_up_to_rows"][name] in {0, *(t["rows"] for t in timings)}


def test_score_matches_across_the_crossover(compiled, detector, ds) -> None:
    # Whichever path a batch size selects, the answer is the same.
    for size in (1, 7, 64, len(ds.X_test)):
        X = ds.X_test.head(size)
        a, b = compiled.score(X), detector.score(X)
        np.testing.assert_array_equal(a["fired"], b["fired"])
        np.testing.assert_allclose(a["novelty_percentile"], b["novelty_percentile"])


def test_alerts_from_the_compiled_detector_are_the_same(compiled, detector, ds) -> None:
    X = ds.X_test.head(200)

    def strip(pairs):
        return [(p, a.model_dump(exclude={"alert_id", "timestamp", "p_attack"})) for p, a in pairs]

    assert strip(alerts_with_positions(compiled, X, compiled.score(X))) == strip(
        alerts_with_positions(detector, X, detector.score(X))
    )


def test_delegates_everything_it_does_not_override(compiled, detector) -> None:
    assert compiled.gate is detector.gate
    assert compiled.metadata is detector.metadata
    assert compiled.ranked_importances(3) == detector.ranked_importances(3)


def test_refuses_when_decisions_disagree(detector, ds, monkeypatch) -> None:
    real = compiled_mod.CompiledDetector._p_attack

    def skewed(self, X):
        return np.clip(real(self, X) + 0.5, 0, 1)

    monkeypatch.setattr(compiled_mod.CompiledDetector, "_p_attack", skewed)
    with pytest.raises(CompileRefused):
        compile_detector(detector, ds.X_test)


def test_unsupported_input_falls_back_to_sklearn(detector, ds, monkeypatch) -> None:
    fresh = compile_detector(detector, ds.X_test)

    def refuse(*_args, **_kwargs):
        raise compiled_mod.Unsupported("NaN in the input")

    monkeypatch.setattr(fresh, "_p_attack", refuse)
    monkeypatch.setattr(fresh, "_iforest_novelty", refuse)
    X = ds.X_test.head(4)
    out = fresh._score(X, force=True)
    assert fresh.fallbacks == 2
    np.testing.assert_array_equal(out["fired"], detector.score(X)["fired"])


def test_parity_treats_missing_families_as_equal() -> None:
    a = pd.DataFrame(
        {
            "fired": [0, 1],
            "family": [None, "dos"],
            "conformal_abstains": [False, False],
            "p_attack": [0.1, 0.9],
            "novelty_percentile": [0.2, 0.3],
            "agreement": [0, 1],
        }
    )
    b = a.copy()
    b["family"] = [np.nan, "dos"]
    assert parity(a, b)["family_changed"] == 0
