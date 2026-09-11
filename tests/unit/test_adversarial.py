"""Problem-space constraint tests.

Every assertion here is about a row that could exist on a wire. That is the whole difference
between this module and the adversarial numbers usually reported for network IDS: an attack that
produces `spkts = 3.2`, or reduces the byte count of the attacker's own exploit, or sets `sload`
independently of `sbytes` and `dur`, has evaded nothing — it has left the set of transmittable
flows, and the detection it "beat" was never defending that region.

So the constraints are tested directly rather than documented and hoped for.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from penumbra.adversarial import evasion


def attack_rows(n: int = 200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dur = rng.uniform(0.01, 2.0, n)
    sbytes = rng.integers(100, 5000, n).astype(float)
    dbytes = rng.integers(50, 3000, n).astype(float)
    spkts = rng.integers(2, 40, n).astype(float)
    dpkts = rng.integers(1, 30, n).astype(float)
    return pd.DataFrame(
        {
            "dur": dur,
            "sbytes": sbytes,
            "dbytes": dbytes,
            "spkts": spkts,
            "dpkts": dpkts,
            "rate": (spkts + dpkts) / dur,
            "sload": sbytes * 8.0 / dur,
            "dload": dbytes * 8.0 / dur,
            "smean": sbytes / spkts,
            "dmean": dbytes / dpkts,
        }
    )


class TestConstraints:
    @pytest.mark.parametrize("effort", [0.5, 1.0, 4.0, 9.0])
    def test_packet_counts_never_decrease(self, effort: float) -> None:
        """An attacker cannot un-send a packet."""
        X = attack_rows()
        out = evasion.problem_space_attack(X, effort, evasion.UNSW_SPACE)
        for col in ("spkts", "dpkts"):
            assert np.all(out[col].to_numpy() >= X[col].to_numpy())

    @pytest.mark.parametrize("effort", [0.5, 1.0, 4.0, 9.0])
    def test_byte_counts_never_decrease(self, effort: float) -> None:
        """Padding is available; un-sending the exploit's own payload is not."""
        X = attack_rows()
        out = evasion.problem_space_attack(X, effort, evasion.UNSW_SPACE)
        for col in ("sbytes", "dbytes"):
            assert np.all(out[col].to_numpy() >= X[col].to_numpy())

    def test_packet_counts_stay_integral(self) -> None:
        """`spkts = 3.2` is not a thing that happens on a wire."""
        out = evasion.problem_space_attack(attack_rows(), 4.0, evasion.UNSW_SPACE)
        for col in ("spkts", "dpkts"):
            values = out[col].to_numpy(dtype=float)
            assert np.all(values == np.floor(values))

    def test_duration_never_shrinks(self) -> None:
        """Stretching a flow is free; making it shorter than its own content is not."""
        X = attack_rows()
        out = evasion.problem_space_attack(X, 2.0, evasion.UNSW_SPACE)
        assert np.all(out["dur"].to_numpy() >= X["dur"].to_numpy())

    def test_derived_features_stay_consistent_with_their_primitives(self) -> None:
        """The constraint that separates this from every ε-ball attack.

        `sload` IS `sbytes * 8 / dur`. A row where it is not describes no flow, and a detection it
        slips past was not defeated.
        """
        out = evasion.problem_space_attack(attack_rows(), 4.0, evasion.UNSW_SPACE)
        dur = out["dur"].to_numpy()
        assert np.allclose(out["sload"], out["sbytes"] * 8.0 / dur, rtol=1e-5)
        assert np.allclose(out["dload"], out["dbytes"] * 8.0 / dur, rtol=1e-5)
        assert np.allclose(out["rate"], (out["spkts"] + out["dpkts"]) / dur, rtol=1e-5)
        assert np.allclose(out["smean"], out["sbytes"] / out["spkts"], rtol=1e-5)

    def test_zero_effort_changes_nothing(self) -> None:
        X = attack_rows()
        assert np.allclose(evasion.problem_space_attack(X, 0.0, evasion.UNSW_SPACE).to_numpy(), X.to_numpy())

    def test_stretching_lowers_rate(self) -> None:
        """Slow-rate mimicry has to actually slow the rate, or it is not mimicry."""
        X = attack_rows()
        out = evasion.problem_space_attack(X, 9.0, evasion.UNSW_SPACE)
        assert out["rate"].mean() < X["rate"].mean()

    def test_effort_is_monotone_in_duration(self) -> None:
        """Effort has to be a price. If more effort is not more slowdown, the x-axis is meaningless."""
        X = attack_rows()
        durations = [
            evasion.problem_space_attack(X, e, evasion.UNSW_SPACE)["dur"].mean()
            for e in (0.0, 1.0, 4.0, 9.0)
        ]
        assert durations == sorted(durations)

    def test_nslkdd_space_leaves_window_aggregates_alone(self) -> None:
        """`serror_rate` is computed over OTHER connections; one flow's attacker cannot set it."""
        X = pd.DataFrame(
            {
                "duration": [1.0] * 50,
                "src_bytes": [500.0] * 50,
                "dst_bytes": [300.0] * 50,
                "serror_rate": [1.0] * 50,
                "count": [200.0] * 50,
            }
        )
        out = evasion.problem_space_attack(X, 9.0, evasion.NSLKDD_SPACE)
        assert np.allclose(out["serror_rate"], X["serror_rate"])
        assert np.allclose(out["count"], X["count"])


class TestFeatureSpaceStrawman:
    def test_it_breaks_the_constraints_on_purpose(self) -> None:
        """It is included to be beaten, so it must actually be unconstrained."""
        X = attack_rows()
        out = evasion.feature_space_attack(X, 9.0, evasion.UNSW_SPACE, benign_reference=X * 0.1)
        integral = out["spkts"].to_numpy(dtype=float)
        assert not np.all(integral == np.floor(integral)), "fractional packet counts are the point"
        assert np.any(out["sbytes"].to_numpy() < X["sbytes"].to_numpy()), "un-sends bytes"
        assert not np.allclose(out["sload"], out["sbytes"] * 8.0 / out["dur"].to_numpy())

    def test_it_moves_toward_the_benign_reference(self) -> None:
        """Symmetric noise is not an attack. An earlier version was, and it inverted the finding."""
        X = attack_rows()
        benign = attack_rows(seed=1) * 0.05
        out = evasion.feature_space_attack(X, 9.0, evasion.UNSW_SPACE, benign_reference=benign)
        centroid = benign.mean().to_numpy()
        before = np.abs(X.to_numpy() - centroid).mean()
        after = np.abs(out.to_numpy() - centroid).mean()
        assert after < before


class TestEvaluate:
    def test_detection_falls_as_effort_rises(self) -> None:
        X = attack_rows(400)
        benign = attack_rows(400, seed=7) * 0.05

        # A detector that keys on rate, which is exactly what slow-rate mimicry attacks.
        def score(frame: pd.DataFrame) -> np.ndarray:
            return (frame["rate"].to_numpy(dtype=float) / 1000.0).clip(0, 1)

        report = evasion.evaluate(
            score, X, dataset="unsw", threshold=0.02, benign_reference=benign, efforts=(0.0, 1.0, 9.0)
        )
        constrained = next(r for r in report.results if r.constrained)
        assert constrained.points[0].detection_rate >= constrained.points[-1].detection_rate

    def test_both_attacks_are_reported(self) -> None:
        """The gap between them is the finding, so neither may be quietly omitted."""
        X = attack_rows(100)
        report = evasion.evaluate(
            lambda f: np.full(len(f), 0.9), X, dataset="unsw", threshold=0.5, efforts=(0.0, 1.0)
        )
        assert {r.constrained for r in report.results} == {True, False}
        assert "PROBLEM-SPACE (realisable)" in report.summary()
        assert "feature-space (strawman)" in report.summary()

    def test_a_family_with_too_few_rows_gets_no_rate(self) -> None:
        """A detection rate from 6 rows is not a detection rate."""
        X = attack_rows(100)
        families = pd.Series(["Common"] * 94 + ["Rare"] * 6)
        report = evasion.evaluate(
            lambda f: np.full(len(f), 0.9),
            X,
            dataset="unsw",
            threshold=0.5,
            families=families,
            efforts=(0.0, 1.0),
        )
        assert "Common" in report.per_family
        assert "Rare" not in report.per_family

    def test_an_unknown_dataset_raises_rather_than_guessing(self) -> None:
        with pytest.raises(ValueError, match="no manipulation space"):
            evasion.space_for("cicids")
