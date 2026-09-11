"""Conformal prediction tests.

The first three assert the guarantee actually holds, because a conformal implementation that quietly
under-covers is worse than none — it attaches a number to a claim it does not keep.

The last group asserts the *caveat*: coverage degrades as exchangeability breaks. That is the
property the drift story rests on, and it is only interesting if it is measured rather than assumed.
"""

from __future__ import annotations

import numpy as np
import pytest

from penumbra.models.conformal import (
    ATTACK,
    BENIGN,
    MondrianConformal,
    coverage_under_drift,
)


def make_scores(n: int, prevalence: float = 0.3, separation: float = 6.0, seed: int = 0):
    """A well-separated binary problem with calibrated-ish scores."""
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < prevalence).astype(int)
    p = np.where(y == 1, rng.beta(separation, 2, n), rng.beta(2, separation, n))
    return np.clip(p, 0.0, 1.0), y


class TestGuarantee:
    @pytest.mark.parametrize("alpha", [0.05, 0.10, 0.20])
    def test_coverage_meets_nominal_under_exchangeability(self, alpha: float) -> None:
        p_cal, y_cal = make_scores(5000, seed=1)
        p_test, y_test = make_scores(8000, seed=2)

        result = MondrianConformal(alpha=alpha).fit(p_cal, y_cal).evaluate(p_test, y_test)
        # Finite-sample slack, but it must not under-cover materially.
        assert result.empirical >= result.nominal - 0.02
        assert result.holds

    def test_coverage_holds_per_class_not_just_on_average(self) -> None:
        """Why Mondrian rather than marginal.

        On an imbalanced set, marginal coverage is dominated by the majority class and can look
        healthy while the minority is badly under-covered - which is the class anyone actually cares
        about in intrusion detection.
        """
        p_cal, y_cal = make_scores(6000, prevalence=0.1, seed=3)
        p_test, y_test = make_scores(6000, prevalence=0.1, seed=4)

        result = MondrianConformal(alpha=0.1).fit(p_cal, y_cal).evaluate(p_test, y_test)
        for cls in (BENIGN, ATTACK):
            assert result.per_class[cls] >= 0.86, f"class {cls} under-covered"

    def test_smaller_alpha_gives_wider_sets(self) -> None:
        p_cal, y_cal = make_scores(4000, seed=5)
        p_test, _ = make_scores(4000, seed=6)

        tight = MondrianConformal(alpha=0.20).fit(p_cal, y_cal).predict_sets(p_test)
        loose = MondrianConformal(alpha=0.02).fit(p_cal, y_cal).predict_sets(p_test)
        assert loose.size.mean() > tight.size.mean()

    def test_alpha_is_validated(self) -> None:
        for bad in (0.0, 1.0, -0.1, 1.5):
            with pytest.raises(ValueError):
                MondrianConformal(alpha=bad)

    def test_predict_before_fit_raises(self) -> None:
        with pytest.raises(RuntimeError, match="not calibrated"):
            MondrianConformal().predict_sets(np.array([0.5]))


class TestSets:
    def test_ambiguous_and_empty_both_abstain(self) -> None:
        p_cal, y_cal = make_scores(3000, seed=7)
        sets = MondrianConformal(alpha=0.1).fit(p_cal, y_cal).predict_sets(np.linspace(0, 1, 500))
        assert np.array_equal(sets.abstains, sets.is_ambiguous | sets.is_empty)

    def test_empty_set_is_not_the_same_as_ambiguous(self) -> None:
        # An empty set means "atypical of BOTH calibration classes" - informative, and a natural
        # companion to the novelty head. Collapsing the two would lose that.
        p_cal, y_cal = make_scores(3000, seed=8)
        sets = MondrianConformal(alpha=0.1).fit(p_cal, y_cal).predict_sets(np.linspace(0, 1, 500))
        assert not np.any(sets.is_empty & sets.is_ambiguous)

    def test_singleton_sets_yield_a_label(self) -> None:
        p_cal, y_cal = make_scores(3000, seed=9)
        p_test, _ = make_scores(1000, seed=10)
        sets = MondrianConformal(alpha=0.1).fit(p_cal, y_cal).predict_sets(p_test)
        labels = sets.labels()
        assert np.all(labels[sets.abstains] == -1)
        assert np.all(np.isin(labels[~sets.abstains], [BENIGN, ATTACK]))

    def test_set_membership_renders_for_an_alert(self) -> None:
        p_cal, y_cal = make_scores(2000, seed=11)
        rendered = (
            MondrianConformal(alpha=0.1).fit(p_cal, y_cal).predict_sets(np.array([0.01, 0.99])).as_strings()
        )
        assert all(set(r) <= {"BENIGN", "ATTACK"} for r in rendered)


class TestCoverageUnderDrift:
    def test_coverage_degrades_as_exchangeability_breaks(self) -> None:
        """The property the drift story rests on.

        Measured on real data (docs/EVALUATION.md §10.6): 89.8% coverage when exchangeable, 59.5%
        on NSL-KDD's natural train/test shift, 29.7% under injected drift.
        """
        p_cal, y_cal = make_scores(5000, seed=12)
        cp = MondrianConformal(alpha=0.1).fit(p_cal, y_cal)

        same = cp.evaluate(*make_scores(5000, seed=13))
        # Shift the score distribution: the model is now confident about different things.
        p_shift, y_shift = make_scores(5000, separation=1.5, seed=14)
        shifted = cp.evaluate(p_shift, y_shift)

        assert same.empirical > shifted.empirical + 0.05
        assert same.holds and not shifted.holds

    def test_abstention_rises_with_coverage_loss(self) -> None:
        """The label-free read of the same signal.

        Coverage needs ground truth. Abstention does not, which is what makes it deployable in a SOC
        where verdicts arrive days late or never.
        """
        p_cal, y_cal = make_scores(5000, seed=15)
        cp = MondrianConformal(alpha=0.1).fit(p_cal, y_cal)

        same = cp.evaluate(*make_scores(5000, seed=16))
        shifted = cp.evaluate(*make_scores(5000, separation=1.5, seed=17))
        assert shifted.abstention_rate > same.abstention_rate

    def test_windowed_tracking_reports_psi_and_coverage(self) -> None:
        p_cal, y_cal = make_scores(4000, seed=18)
        cp = MondrianConformal(alpha=0.1).fit(p_cal, y_cal)

        clean_p, clean_y = make_scores(6000, seed=19)
        drift_p, drift_y = make_scores(6000, separation=1.5, seed=20)
        stream_p = np.concatenate([clean_p, drift_p])
        stream_y = np.concatenate([clean_y, drift_y])

        result = coverage_under_drift(cp, stream_p, stream_y, p_cal, window=2000, change_point=6000)
        assert len(result.points) >= 5
        before = [pt for pt in result.points if pt.window < result.change_point_window]
        after = [pt for pt in result.points if pt.window >= result.change_point_window]
        assert np.mean([p.empirical_coverage for p in after]) < np.mean(
            [p.empirical_coverage for p in before]
        )

    def test_breach_before_injection_is_not_reported_as_a_detection(self) -> None:
        """A negative detection delay is nonsense and must not be printed as one.

        If coverage is already below nominal before the injection, the baseline was never
        exchangeable and no delay can be claimed from that stream.
        """
        p_cal, y_cal = make_scores(3000, separation=8.0, seed=21)
        cp = MondrianConformal(alpha=0.1).fit(p_cal, y_cal)

        # Whole stream drawn from a different distribution: breached from window 0.
        bad_p, bad_y = make_scores(8000, separation=1.2, seed=22)
        result = coverage_under_drift(cp, bad_p, bad_y, p_cal, window=2000, change_point=4000)

        text = result.summary()
        assert "ALREADY below nominal" in text
        assert "windows later" not in text


class TestDetectorIntegration:
    def test_scored_frame_carries_conformal_columns(self) -> None:
        # A detector fitted without conformal must still produce a usable scored frame rather than
        # raising - the columns are present and simply never abstain.
        import pandas as pd

        from penumbra.alerts.builder import alerts_from_scores

        scored = pd.DataFrame(
            {
                "p_attack": [0.9, 0.1],
                "novelty_percentile": [0.5, 0.5],
                "agreement": [0, 0],
                "fired": [1, 0],
                "family": ["DoS", None],
                "conformal_abstains": [False, True],
                "conformal_set": [["ATTACK"], []],
            }
        )
        X = pd.DataFrame({"src_bytes": [100, 200], "dst_bytes": [50, 60]})

        alerts = alerts_from_scores(object(), X, scored, dataset="nslkdd")
        # Row 1 did not fire but abstained, so it still reaches an analyst.
        assert len(alerts) == 2
        assert alerts[0].conformal_set == ["ATTACK"]
