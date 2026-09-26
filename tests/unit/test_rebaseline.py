"""Re-baselining: what changes, what must not, and the gate in front of it."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from penumbra.eval import rebaseline as rb
from penumbra.models.conformal import ATTACK, BENIGN
from penumbra.models.detector import PenumbraDetector
from tests.unit.test_registry import tiny_dataset


@pytest.fixture(scope="module")
def detector() -> PenumbraDetector:
    return PenumbraDetector(target_fpr=0.05).fit(tiny_dataset(n=900))


def new_network(n: int, seed: int = 1) -> pd.DataFrame:
    """Benign traffic from a network unlike the training one: every feature shifted and rescaled."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "a": rng.normal(1.5, 0.3, n),
            "b": rng.normal(4.0, 0.5, n),
            "c": rng.exponential(3.0, n),
            "proto": rng.choice(["tcp", "udp"], n, p=[0.9, 0.1]),
        }
    )


def attacks_on_new_network(n: int, seed: int = 2) -> pd.DataFrame:
    X = new_network(n, seed)
    X["b"] = X["b"] + 6.0  # far from this network's normal
    return X


class TestRebaselined:
    def test_supervised_and_family_models_are_untouched(self, detector) -> None:
        X = new_network(600)
        new = detector.rebaselined(X.iloc[:300], X.iloc[300:])
        assert new.supervised_model is detector.supervised_model
        assert new.family_model is detector.family_model
        assert new.novelty is not detector.novelty
        assert new.novelty_prep is not detector.novelty_prep

    def test_original_detector_is_not_mutated(self, detector) -> None:
        before = (detector.gate.supervised_threshold, detector.gate.novelty_threshold, detector.target_fpr)
        X = new_network(600)
        detector.rebaselined(X.iloc[:300], X.iloc[300:], target_fpr=0.2)
        assert (
            detector.gate.supervised_threshold,
            detector.gate.novelty_threshold,
            detector.target_fpr,
        ) == before

    def test_only_the_benign_conformal_quantile_moves(self, detector) -> None:
        X = new_network(600)
        new = detector.rebaselined(X.iloc[:300], X.iloc[300:])
        assert new.conformal.quantiles_[ATTACK] == detector.conformal.quantiles_[ATTACK]
        assert new.conformal.n_calibration_[BENIGN] == 300

    def test_thresholds_hold_on_the_new_network(self, detector) -> None:
        X = new_network(3000)
        new = detector.rebaselined(X.iloc[:1500], X.iloc[1500:2400], target_fpr=0.05)
        hold = X.iloc[2400:]
        stock_rate = (detector.score(hold)["fired"] > 0).mean()
        new_rate = (new.score(hold)["fired"] > 0).mean()
        assert new_rate <= 0.10
        assert new_rate < stock_rate

    def test_overlapping_slices_are_refused(self, detector) -> None:
        X = new_network(400)
        with pytest.raises(ValueError, match="overlap"):
            detector.rebaselined(X.iloc[:300], X.iloc[200:])

    def test_metadata_records_where_normal_came_from(self, detector) -> None:
        X = new_network(600)
        new = detector.rebaselined(X.iloc[:300], X.iloc[300:], source="lab.pcap")
        assert new.metadata.baseline["source"] == "lab.pcap"
        assert new.metadata.baseline["previous_novelty_threshold"] == detector.gate.novelty_threshold
        assert detector.metadata.baseline is None

    def test_survives_save_and_load(self, detector, tmp_path) -> None:
        X = new_network(600)
        new = detector.rebaselined(X.iloc[:300], X.iloc[300:])
        new.save(tmp_path)
        loaded = PenumbraDetector.load(tmp_path)
        pd.testing.assert_frame_equal(loaded.score(X.head(50)), new.score(X.head(50)))
        assert loaded.metadata.baseline == new.metadata.baseline


class TestGate:
    def test_minimum_rows(self) -> None:
        # 5% total over two heads is 2.53% per head; five exceedances need 198 calibration flows.
        assert rb.minimum_calibration_rows(0.05) == 198
        assert rb.minimum_calibration_rows(0.01) == 998
        assert rb.minimum_window(0.01) == 3327

    def test_holdout_limit_allows_sampling_noise_but_not_much(self) -> None:
        assert rb.holdout_limit(200, 0.05) > 10
        assert rb.holdout_limit(200, 0.05) < 25

    def test_split_is_disjoint_and_complete(self) -> None:
        f, c, h = rb.split(1000)
        assert len(f) + len(c) + len(h) == 1000
        assert len(set(f) | set(c) | set(h)) == 1000
        assert (len(f), len(c), len(h)) == (500, 300, 200)

    def test_short_window_is_refused_before_fitting(self, detector) -> None:
        out, report = rb.run(detector, new_network(300), target_fpr=0.01, source="short")
        assert out is None
        assert report["passed"] is False
        assert "R1" in report["reasons"][0]
        assert report["gates"]["R1_calibration_size"]["window_needed"] == 3327

    def test_a_good_window_passes_and_reports_evidence(self, detector) -> None:
        out, report = rb.run(
            detector,
            new_network(2000),
            target_fpr=0.05,
            source="synthetic",
            X_attack=attacks_on_new_network(300),
            known=(tiny_dataset(n=900).X_test, tiny_dataset(n=900).y_test.to_numpy()),
        )
        assert out is not None
        assert report["passed"] is True, report["reasons"]
        ev = report["evidence"]
        assert ev["holdout"]["rebaselined"]["fired_rate"] < ev["holdout"]["current"]["fired_rate"]
        assert ev["attack_flows"]["rebaselined"]["fired_rate"] > 0.5
        assert {"attack_recall", "benign_fpr"} <= set(ev["known_dataset"]["rebaselined"])
        json.dumps(report, default=float)  # the report is what gets attached to the registry

    def test_a_threshold_that_does_not_hold_fails_r2(self, detector, monkeypatch) -> None:
        real = PenumbraDetector.rebaselined

        def lax(self, X_fit, X_cal, **kw):
            out = real(self, X_fit, X_cal, **kw)
            out.gate.novelty_threshold = 0.0  # fires on everything
            return out

        monkeypatch.setattr(PenumbraDetector, "rebaselined", lax)
        out, report = rb.run(detector, new_network(2000), target_fpr=0.05, source="lax")
        assert report["passed"] is False
        assert report["gates"]["R2_holdout_fpr"]["passed"] is False
        assert any(r.startswith("R2") for r in report["reasons"])


def test_cli_registers_a_gated_candidate_without_promoting_it(tmp_path, monkeypatch) -> None:
    from typer.testing import CliRunner

    from penumbra import config
    from penumbra.cli import app
    from penumbra.models.registry import ModelRegistry

    monkeypatch.setenv("PENUMBRA_ARTIFACT_ROOT", str(tmp_path / "artifacts"))
    monkeypatch.setenv("PENUMBRA_DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.delenv("PENUMBRA_STATE_ROOT", raising=False)
    config.settings.cache_clear()
    try:
        PenumbraDetector(target_fpr=0.05).fit(tiny_dataset(n=900)).save(
            tmp_path / "artifacts" / "models" / "tiny"
        )
        flows = tmp_path / "benign.csv"
        new_network(2000).to_csv(flows, index=False)

        result = CliRunner().invoke(
            app,
            ["rebaseline", str(flows), "--model", "tiny", "--no-known-check", "--by", "alice"],
            env={"COLUMNS": "300"},
        )
        assert result.exit_code == 0, result.output
        reg = ModelRegistry(tmp_path / "artifacts" / "registry", "tiny")
        versions = reg.versions()
        assert len(versions) == 1
        assert versions[0].gate["kind"] == "rebaseline" and versions[0].gate["passed"] is True
        assert reg.champion() is None  # registered, not promoted
        assert not reg.verify(versions[0].version)
        loaded = reg.load(versions[0].version)
        # The verdict policy follows the new operating point.
        assert loaded.policy.novelty_percentile_threshold == loaded.gate.novelty_threshold
        audit = (tmp_path / "artifacts" / "audit.jsonl").read_text(encoding="utf-8")
        assert "model.rebaseline" in audit
        report = json.loads(
            (tmp_path / "artifacts" / "reports" / "rebaseline_tiny.json").read_text(encoding="utf-8")
        )
        assert report["window"]["benign_flows"] == 2000
    finally:
        config.settings.cache_clear()
