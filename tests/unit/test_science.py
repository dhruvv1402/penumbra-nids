"""Tests for the evaluation and detection machinery.

Several of these encode bugs that were actually hit during development and produced plausible,
publishable-looking numbers. They are here so those bugs cannot come back quietly:

  * fusing a calibrated probability with a percentile using max()
  * rank-normalising against a reference small enough to saturate at 1.0
  * PSI with bin edges recomputed per window
  * matching an alert budget on total alerts rather than on false positives
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from penumbra.alerts.correlate import Correlator, EntitiesRequired, fan_out_features
from penumbra.alerts.models import Lane, NetworkContext, Severity, Verdict
from penumbra.alerts.scoring import ScoringPolicy, build_alert, classify, priority_score, route
from penumbra.drift import detectors as drift
from penumbra.eval import budget, cost, prevalence
from penumbra.eval.bootstrap import benjamini_hochberg, mcnemar, stratified_bootstrap
from penumbra.eval.metrics import BelowEstimationFloor, binary_metrics, estimation_floor, recall_at_fpr
from penumbra.models.fusion import OrGate, expected_or_fpr, per_head_budget, rank_normalise


class TestMetrics:
    def test_estimation_floor(self) -> None:
        assert estimation_floor(37_000) == pytest.approx(1 / 37_000)

    def test_below_floor_refuses_rather_than_guessing(self) -> None:
        y = np.array([0] * 1000 + [1] * 100)
        s = np.random.default_rng(0).random(1100)
        # 0.01% of 1,000 benign rows is a tenth of a false positive. Not a number.
        with pytest.raises(BelowEstimationFloor):
            recall_at_fpr(y, s, 0.0001)

    def test_pr_auc_carries_its_no_skill_baseline(self) -> None:
        rng = np.random.default_rng(0)
        y = (rng.random(2000) < 0.3).astype(int)
        s = np.clip(rng.random(2000) + y * 0.4, 0, 1)
        m = binary_metrics(y, s)
        # The baseline IS the prevalence - a PR-AUC at that level is worthless, and reporting the
        # number without it invites reading 0.3 as a result.
        assert m.pr_auc_no_skill == pytest.approx(m.prevalence)
        assert m.pr_auc_lift == pytest.approx(m.pr_auc - m.prevalence)

    def test_tpr_and_fpr_are_prevalence_invariant(self) -> None:
        """The reason they are the primary metrics."""
        rng = np.random.default_rng(1)
        base_pos = rng.normal(1.0, 1.0, 500)
        base_neg = rng.normal(-1.0, 1.0, 500)

        balanced_y = np.r_[np.ones(500), np.zeros(500)]
        balanced_s = np.r_[base_pos, base_neg]

        # Same score distributions, a tenth of the positives.
        rare_y = np.r_[np.ones(50), np.zeros(500)]
        rare_s = np.r_[base_pos[:50], base_neg]

        a = binary_metrics(balanced_y, balanced_s, threshold=0.0, is_probability=False)
        b = binary_metrics(rare_y, rare_s, threshold=0.0, is_probability=False)

        assert a.fpr == pytest.approx(b.fpr, abs=1e-9)  # identical: same negatives
        assert a.tpr == pytest.approx(b.tpr, abs=0.12)  # stable under resampling
        # Precision, by contrast, collapses - which is the whole point.
        assert b.precision < a.precision - 0.2


class TestPrevalence:
    def test_reweighting_changes_ppv_not_rates(self) -> None:
        high = prevalence.reweight(0.9, 0.01, 0.5)
        low = prevalence.reweight(0.9, 0.01, 1e-4)
        assert high.tpr == low.tpr and high.fpr == low.fpr
        assert low.ppv < 0.05 < high.ppv

    def test_base_rate_fallacy_arithmetic(self) -> None:
        est = prevalence.reweight(0.9, 0.01, 1e-4, flows_per_day=1_000_000)
        assert est.true_positives_per_day == pytest.approx(90, abs=1)
        assert est.false_positives_per_day == pytest.approx(9999, abs=5)
        assert est.ppv < 0.01  # ~1% PPV at a 90% detection rate

    def test_estimates_are_labelled_modelled(self) -> None:
        assert prevalence.reweight(0.9, 0.01, 1e-3).measured is False


class TestBudget:
    def test_matching_on_fpr_equalises_benign_flagged(self) -> None:
        rng = np.random.default_rng(0)
        y = np.r_[np.ones(500), np.zeros(500)].astype(int)
        a = np.r_[rng.beta(6, 2, 500), rng.beta(2, 6, 500)]
        b = rng.random(1000)

        comp = budget.compare_at_matched_budget(y, {"a": a, "b": b}, match="fpr", budget_fpr=0.1)
        assert comp.results["a"].n_benign_flagged == comp.results["b"].n_benign_flagged

    def test_alert_count_matching_is_spent_on_true_positives(self) -> None:
        """Why match='fpr' is the default.

        On a 50%-attack set an alert-count cap is consumed by true positives and caps recall long
        before any detector has had a say. This is the bug that made the first LOAFO run read flat.
        """
        rng = np.random.default_rng(0)
        y = np.r_[np.ones(500), np.zeros(500)].astype(int)
        scores = np.r_[rng.beta(8, 2, 500), rng.beta(2, 8, 500)]

        capped = budget.compare_at_matched_budget(y, {"m": scores}, match="alerts", budget_per_1k=50)
        assert capped.results["m"].recall <= 0.12  # 50 alerts / 500 attacks

        by_fpr = budget.compare_at_matched_budget(y, {"m": scores}, match="fpr", budget_fpr=0.05)
        assert by_fpr.results["m"].recall > capped.results["m"].recall


class TestFusion:
    def test_rank_normalise_gives_percentiles(self) -> None:
        ref = np.arange(100, dtype=float)
        assert rank_normalise(np.array([50.0]), ref)[0] == pytest.approx(0.5, abs=0.02)

    def test_max_over_incommensurable_scales_is_dominated_by_the_wider_one(self) -> None:
        """The fusion bug, pinned.

        p_attack is bimodal near 0; a novelty percentile is ~uniform. max() over the two inherits
        the novelty distribution's threshold and discards supervised detections below it.
        """
        rng = np.random.default_rng(0)
        # BENIGN rows, which is where the threshold is set. Measured on UNSW: p_attack p99 = 0.907
        # while the novelty percentile is uniform by construction with p99 = 0.996.
        p_attack = rng.beta(1, 30, 5000)
        novelty = rng.random(5000)
        naive = np.maximum(p_attack, novelty)

        sup_threshold = float(np.quantile(p_attack, 0.99))
        fused_threshold = float(np.quantile(naive, 0.99))

        # The fused threshold is dragged up to the novelty distribution's, so every supervised
        # detection scoring between the two is silently discarded.
        assert fused_threshold > sup_threshold + 0.5
        assert fused_threshold > 0.95

    def test_or_gate_thresholds_each_head_separately(self) -> None:
        rng = np.random.default_rng(0)
        benign_p, benign_n = rng.beta(1, 20, 5000), rng.random(5000)
        gate = OrGate.fit(benign_p, benign_n, total_fpr=0.02, use_novelty=True)
        realised = gate.flags(benign_p, benign_n).mean()
        assert realised == pytest.approx(0.02, abs=0.01)

    def test_or_cost_is_charged(self) -> None:
        # Two heads at a 1% total budget each run at ~0.5%; the fused config buys coverage out of
        # the same allowance rather than getting it free.
        per_head = per_head_budget(0.01, 2)
        assert 0.004 < per_head < 0.006
        assert expected_or_fpr(per_head, per_head) == pytest.approx(0.01, abs=0.001)

    def test_supervised_only_gate_uses_the_whole_budget(self) -> None:
        rng = np.random.default_rng(0)
        benign_p, benign_n = rng.beta(1, 20, 5000), rng.random(5000)
        gate = OrGate.fit(benign_p, benign_n, total_fpr=0.02, use_novelty=False)
        assert gate.flags(benign_p, benign_n).mean() == pytest.approx(0.02, abs=0.006)


class TestDrift:
    def test_psi_with_frozen_bins_sees_a_shift(self) -> None:
        rng = np.random.default_rng(0)
        ref, cur = rng.normal(0, 1, 5000), rng.normal(2, 1, 5000)
        assert drift.PSIBins(ref).psi(cur) > drift.PSI_SIGNIFICANT

    def test_recomputing_bins_per_window_is_blind(self) -> None:
        """The PSI bug, pinned. Bins that follow the data cannot see the data move."""
        rng = np.random.default_rng(0)
        cur = rng.normal(2, 1, 5000)
        assert drift.PSIBins(cur).psi(cur) == pytest.approx(0.0, abs=1e-6)

    def test_psi_is_near_zero_on_stable_data(self) -> None:
        rng = np.random.default_rng(0)
        assert drift.PSIBins(rng.normal(0, 1, 5000)).psi(rng.normal(0, 1, 5000)) < drift.PSI_NO_SHIFT

    def test_multiplicity_is_corrected(self) -> None:
        rng = np.random.default_rng(0)
        stable = pd.DataFrame({f"f{i}": rng.normal(0, 1, 2000) for i in range(40)})
        other = pd.DataFrame({f"f{i}": rng.normal(0, 1, 2000) for i in range(40)})
        report = drift.compare(stable, other)
        # Under the null ~2 of 40 flag at alpha=0.05. BH should keep it at or below that.
        assert len(report.ks_flagged) <= 3
        assert report.expected_false_flags == pytest.approx(2.0)

    def test_benjamini_hochberg_rejects_the_obvious(self) -> None:
        p = np.array([1e-10, 1e-9, 0.5, 0.8, 0.9])
        assert benjamini_hochberg(p).tolist() == [True, True, False, False, False]

    def test_prediction_drift_needs_no_labels(self) -> None:
        rng = np.random.default_rng(0)
        result = drift.prediction_drift(rng.beta(2, 8, 5000), rng.beta(6, 4, 5000))
        assert result.is_anomalous


class TestBootstrap:
    def test_interval_brackets_the_point(self) -> None:
        rng = np.random.default_rng(0)
        y = np.r_[np.ones(300), np.zeros(300)].astype(int)
        s = np.r_[rng.beta(6, 2, 300), rng.beta(2, 6, 300)]
        from sklearn.metrics import roc_auc_score

        ci = stratified_bootstrap(y, s, lambda a, b: float(roc_auc_score(a, b)), n_resamples=200)
        assert ci.lower <= ci.point <= ci.upper
        assert ci.width < 0.2

    def test_mcnemar_reports_discordant_counts(self) -> None:
        y = np.array([1, 1, 1, 0, 0, 0, 1, 0])
        a = np.array([1, 0, 1, 0, 1, 0, 1, 0])
        b = np.array([1, 1, 1, 0, 0, 0, 0, 0])
        r = mcnemar(y, a, b)
        assert r.n_discordant == r.n01 + r.n10
        assert 0.0 <= r.p_value <= 1.0


class TestScoring:
    def test_novel_verdict_requires_novelty_without_supervised(self) -> None:
        policy = ScoringPolicy()
        verdict, lane = classify(0.1, 0.995, policy=policy, agreement=2)
        assert verdict is Verdict.SUSPECTED_NOVEL
        assert lane is Lane.HUNTING

    def test_supervised_confidence_wins_over_novelty(self) -> None:
        verdict, lane = classify(0.95, 0.999, policy=ScoringPolicy(), agreement=3)
        assert verdict is Verdict.KNOWN_ATTACK
        assert lane is Lane.KNOWN_THREAT

    def test_priority_is_not_a_probability(self) -> None:
        # Agreement pushes priority above either input, which a probability could not do.
        assert priority_score(0.9, 0.9, agreement=3) > 90

    def test_novelty_alert_carries_no_family(self) -> None:
        alert = build_alert(
            p_attack=0.1,
            novelty_percentile=0.999,
            policy=ScoringPolicy(),
            family="Exploits",
            agreement=3,
        )
        assert alert.verdict is Verdict.SUSPECTED_NOVEL
        # Naming a family would contradict the verdict that we do not recognise it.
        assert alert.family is None
        assert alert.attack is None

    def test_hunting_lane_is_budget_capped(self) -> None:
        policy = ScoringPolicy(hunting_daily_budget=10)
        alerts = [
            build_alert(p_attack=0.1, novelty_percentile=0.99 + i * 1e-5, policy=policy, agreement=2)
            for i in range(50)
        ]
        lanes = route(alerts, policy)
        assert len(lanes[Lane.HUNTING]) == 10


class TestCorrelation:
    @staticmethod
    def _alert(entity: str, dst: str, port: int, family: str = "Reconnaissance"):
        return build_alert(
            p_attack=0.95,
            novelty_percentile=0.4,
            policy=ScoringPolicy(),
            family=family,
            network=NetworkContext(src_ip=entity, dst_ip=dst, dst_port=port),
        )

    def test_refuses_without_entities(self) -> None:
        # UNSW-NB15 has no IPs; a compression ratio from synthesised ones would be invented.
        alerts = [
            build_alert(p_attack=0.95, novelty_percentile=0.4, policy=ScoringPolicy(), family="DoS")
            for _ in range(10)
        ]
        with pytest.raises(EntitiesRequired, match="synthesised identifiers"):
            Correlator().correlate(alerts, dataset="unsw")

    def test_groups_a_scan_into_one_incident(self) -> None:
        alerts = [self._alert("pseudo:aa", f"pseudo:d{i}", 445) for i in range(400)]
        result = Correlator().correlate(alerts, dataset="cicids")
        assert len(result.incidents) == 1
        assert result.incidents[0].event_count == 400
        assert result.compression_ratio == 400

    def test_separate_families_stay_separate(self) -> None:
        alerts = [self._alert("pseudo:aa", "pseudo:d1", 445, "Reconnaissance") for _ in range(5)]
        alerts += [self._alert("pseudo:aa", "pseudo:d1", 445, "Exploits") for _ in range(5)]
        result = Correlator().correlate(alerts, dataset="cicids")
        # A host scanning AND exploiting is two incidents with different responses.
        assert len(result.incidents) == 2

    def test_fan_out_is_measured(self) -> None:
        alerts = [self._alert("pseudo:aa", f"pseudo:d{i}", 1000 + i) for i in range(50)]
        stats = fan_out_features(alerts)
        assert stats["pseudo:aa"]["distinct_destinations"] == 50
        assert stats["pseudo:aa"]["distinct_ports"] == 50


class TestCost:
    def test_optimal_beats_the_default_threshold(self) -> None:
        rng = np.random.default_rng(0)
        y = (rng.random(5000) < 0.4).astype(int)
        s = np.clip(rng.beta(2, 5, 5000) + y * 0.5, 0, 1)
        curve = cost.build_curve(y, s)
        assert curve.optimal.total_cost_per_day <= curve.at_threshold(0.5).total_cost_per_day

    def test_sensitivity_reports_when_the_point_is_undetermined(self) -> None:
        rng = np.random.default_rng(0)
        y = (rng.random(3000) < 0.4).astype(int)
        s = np.clip(rng.beta(2, 5, 3000) + y * 0.5, 0, 1)
        text = cost.sensitivity_analysis(y, s)
        assert "assumed miss cost" in text
        assert ("NOT" in text) or ("stable" in text)

    def test_cost_ratio_is_explicit(self) -> None:
        model = cost.CostModel(cost_per_missed_attack=5000, minutes_per_false_positive=45)
        assert model.cost_ratio > 100
        assert "1 miss =" in model.describe()


class TestSeverity:
    def test_severity_matches_asim_enum(self) -> None:
        # ASIM validates EventSeverity as a 4-value string enum; a mismatch fails ingestion.
        from penumbra.alerts.schemas.asim import EVENT_SEVERITIES

        assert {s.value for s in Severity} == EVENT_SEVERITIES
