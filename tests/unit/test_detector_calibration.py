"""E9a: held-out calibration, tie-safe thresholds, and the matched-FPR gate (ADR-0005).

Each of these fixes a way the detector could report one operating point and run at another, so
each test states the failure it prevents rather than only the behaviour it expects.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from penumbra.eval import canary
from penumbra.models.detector import PenumbraDetector, calibration_split
from penumbra.models.fusion import OrGate, benign_threshold
from tests.unit.test_registry import tiny_dataset


class TestCalibrationSplit:
    def test_disjoint_complete_and_stratified(self) -> None:
        fam = pd.Series(["normal"] * 500 + ["dos"] * 300 + ["probe"] * 200)
        fit, cal = calibration_split(fam, 0.2)
        assert len(set(fit) & set(cal)) == 0
        assert len(fit) + len(cal) == len(fam)
        for name, n in (("normal", 500), ("dos", 300), ("probe", 200)):
            assert (fam.iloc[cal] == name).sum() == round(n * 0.2)

    def test_a_one_row_family_stays_in_training(self) -> None:
        fam = pd.Series(["normal"] * 100 + ["worms"])
        fit, cal = calibration_split(fam, 0.2)
        assert 100 in set(fit)
        assert 100 not in set(cal)

    def test_kept_rows_never_calibrate(self) -> None:
        fam = pd.Series(["normal"] * 200)
        keep = np.arange(150, 200)
        _, cal = calibration_split(fam, 0.5, keep_in_fit=keep)
        assert not set(cal) & set(keep)

    def test_deterministic(self) -> None:
        fam = pd.Series(["normal"] * 300 + ["dos"] * 100)
        a, b = calibration_split(fam, 0.2), calibration_split(fam, 0.2)
        assert all(np.array_equal(x, y) for x, y in zip(a, b, strict=True))


@pytest.fixture(scope="module")
def held_out() -> PenumbraDetector:
    return PenumbraDetector(target_fpr=0.05).fit(tiny_dataset(n=1500))


class TestHeldOutFit:
    def test_supervised_model_never_saw_the_calibration_slice(self, held_out) -> None:
        meta = held_out.metadata
        assert meta.calibration == "held_out"
        assert meta.n_supervised_fit_rows + meta.n_calibration_rows == meta.n_train_rows
        assert meta.n_calibration_rows == pytest.approx(0.2 * meta.n_train_rows, abs=3)

    def test_in_sample_is_still_available_and_says_so(self) -> None:
        det = PenumbraDetector(target_fpr=0.05).fit(tiny_dataset(n=900), calibration="in_sample")
        assert det.metadata.calibration == "in_sample"
        assert det.metadata.n_supervised_fit_rows == det.metadata.n_train_rows

    def test_unknown_mode_is_refused(self) -> None:
        with pytest.raises(ValueError, match="calibration"):
            PenumbraDetector().fit(tiny_dataset(n=300), calibration="test")

    def test_quarantine_is_recorded(self) -> None:
        det = PenumbraDetector(target_fpr=0.05).fit(tiny_dataset(n=600), quarantined=["sttl", "dttl"])
        assert det.metadata.quarantined_features == ["dttl", "sttl"]

    def test_metadata_saved_before_e9a_still_loads_as_in_sample(self, held_out, tmp_path) -> None:
        held_out.save(tmp_path)
        meta = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))
        for key in ("calibration", "n_calibration_rows", "n_supervised_fit_rows", "family_model_name"):
            meta.pop(key)
        (tmp_path / "metadata.json").write_text(json.dumps(meta), encoding="utf-8")
        assert PenumbraDetector.load(tmp_path).metadata.calibration == "in_sample"


class TestTieSafeThreshold:
    def test_a_tied_block_is_not_admitted_whole(self) -> None:
        # 99.6% of benign rows at exactly 0: a plain quantile puts the 99.5th percentile at 0 and,
        # with `>=`, fires on every row - a 0.5% budget realising 100%.
        ref = np.r_[np.zeros(996), np.ones(4)]
        assert float(np.quantile(ref, 0.995)) == 0.0
        t = benign_threshold(ref, 0.005)
        assert t > 0.0
        assert np.mean(ref >= t) <= 0.005 + 1 / len(ref)

    def test_untied_scores_get_the_plain_quantile(self) -> None:
        ref = np.random.default_rng(0).random(5000)
        assert benign_threshold(ref, 0.01) == float(np.quantile(ref, 0.99))

    def test_or_gate_on_tied_scores_respects_its_budget(self) -> None:
        p = np.r_[np.zeros(995), np.linspace(0.5, 1, 5)]
        n = np.random.default_rng(1).random(1000)
        gate = OrGate.fit(p, n, total_fpr=0.01)
        assert gate.flags(p, n).mean() <= 0.02


class FakeDetector:
    """Scores are fixed per row; only the thresholds differ. Enough to test a gate."""

    def __init__(self, p: np.ndarray, novelty: np.ndarray, threshold: float, target_fpr: float = 0.01):
        self.p, self.novelty, self.threshold, self.target_fpr = p, novelty, threshold, target_fpr

    def score(self, X: pd.DataFrame) -> pd.DataFrame:
        fired = (self.p >= self.threshold).astype(int)
        return pd.DataFrame(
            {"p_attack": self.p, "novelty_percentile": self.novelty, "fired": fired}, index=X.index
        )


def canary_set(n_benign: int = 4000, n_attack: int = 1000, seed: int = 0):
    rng = np.random.default_rng(seed)
    y = np.r_[np.zeros(n_benign, int), np.ones(n_attack, int)]
    fam = np.r_[["normal"] * n_benign, rng.choice(["dos", "probe"], n_attack)]
    p = np.r_[rng.beta(2, 8, n_benign), rng.beta(8, 2, n_attack)]
    novelty = rng.random(len(y)) * 0.5  # an uninformative second head, never above its threshold
    X = pd.DataFrame({"row": np.arange(len(y))})
    return X, pd.Series(y), pd.Series(fam), p, novelty


class TestMatchedFprGate:
    def test_an_over_alerting_champion_does_not_block_an_honest_challenger(self) -> None:
        X, y, fam, p, nov = canary_set()
        # The same model at two operating points: the champion over-alerts (low threshold).
        champion = FakeDetector(p, nov, threshold=0.30)
        challenger = FakeDetector(p, nov, threshold=0.55)
        result = canary.gate_detectors(champion, challenger, X, y, fam)
        assert result.at_own_threshold is not None
        assert result.at_own_threshold["passed"] is False  # the old gate: G1 refuses
        assert any(r.startswith("G1") for r in result.at_own_threshold["reasons"])
        assert result.passed, result.reasons  # ADR-0005: same model, matched FPR, no regression
        assert result.advisory["matched_fpr"] == pytest.approx(result.champion.fpr, abs=0.01)

    def test_a_poisoned_family_still_fails_at_matched_fpr(self) -> None:
        X, y, fam, p, nov = canary_set()
        champion = FakeDetector(p, nov, threshold=0.30)
        poisoned = p.copy()
        poisoned[(fam == "probe").to_numpy()] *= 0.2  # taught that probes look benign
        result = canary.gate_detectors(champion, FakeDetector(poisoned, nov, threshold=0.55), X, y, fam)
        assert not result.passed
        assert any(r.startswith("G2") and "probe" in r for r in result.reasons)

    def test_a_challenger_that_alerts_more_still_fails_g3(self) -> None:
        X, y, fam, p, nov = canary_set()
        result = canary.gate_detectors(
            FakeDetector(p, nov, threshold=0.55), FakeDetector(p, nov, threshold=0.30), X, y, fam
        )
        assert any(r.startswith("G3") for r in result.reasons)

    def test_both_verdicts_and_g4_are_recorded(self) -> None:
        X, y, fam, p, nov = canary_set()
        d = canary.gate_detectors(
            FakeDetector(p, nov, threshold=0.30), FakeDetector(p, nov, threshold=0.55), X, y, fam
        ).to_dict()
        assert d["kind"] == "matched_fpr"
        assert {"at_own_threshold", "challenger_at_matched_fpr", "advisory"} <= set(d)
        lo, hi = d["advisory"]["G4_interval_99"]
        assert lo <= d["advisory"]["G4_realised_fpr"] <= hi

    def test_binomial_interval(self) -> None:
        lo, hi = canary.binomial_interval(10, 1000)
        assert lo < 0.01 < hi
        assert canary.binomial_interval(0, 100)[0] == 0.0
        assert canary.binomial_interval(100, 100)[1] == 1.0
