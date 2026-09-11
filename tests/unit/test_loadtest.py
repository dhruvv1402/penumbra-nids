"""Load-test reporting tests.

The measurements themselves are timings and cannot be asserted — a p99 on a shared laptop is not a
reproducible quantity. What *can* be asserted is that the report does not mislead, and that is what
these cover: the fixed-versus-marginal split has to be arithmetic rather than vibes, the bandwidth
conversion has to carry both the mean and the median, and the summary has to keep saying that flows
per second is not link speed.

The last one is a text assertion, which is unusual in a test suite and deliberate here. That
sentence is the entire difference between an honest throughput claim and a marketing one, and it is
exactly the kind of line that gets trimmed for brevity six months from now.
"""

from __future__ import annotations

import pandas as pd
import pytest

from penumbra.eval.loadtest import (
    LoadReport,
    Percentiles,
    ThroughputPoint,
    estimate_flow_bytes,
)


def point(batch: int, flows_per_second: float, p50: float) -> ThroughputPoint:
    return ThroughputPoint(
        batch_size=batch,
        flows_per_second=flows_per_second,
        batch_latency_ms=Percentiles(p50=p50, p95=p50 * 1.2, p99=p50 * 1.4, mean=p50, max=p50 * 2),
        n_batches=10,
    )


class TestPercentiles:
    def test_nearest_rank_does_not_invent_precision(self) -> None:
        """With 20 samples there is no meaningful p99; interpolating one would manufacture it."""
        result = Percentiles.of([float(i) for i in range(1, 21)])
        assert result.p50 == 10.0
        assert result.p99 == 20.0
        assert result.max == 20.0

    def test_empty_samples_are_zeros_not_a_crash(self) -> None:
        result = Percentiles.of([])
        assert (result.p50, result.p99, result.max) == (0.0, 0.0, 0.0)

    def test_a_single_sample_is_every_percentile(self) -> None:
        result = Percentiles.of([42.0])
        assert result.p50 == result.p99 == result.max == 42.0


class TestCostSplit:
    def test_fixed_and_marginal_recover_a_known_cost_model(self) -> None:
        """Given latency = 100ms + 0.05ms/flow, the split must recover 100 and 0.05."""
        report = LoadReport(dataset="synthetic", n_rows=4096, cores=8, mean_flow_bytes=1000)
        report.points = [
            point(1, 10.0, 100.0 + 0.05 * 1),
            point(2048, 20_000.0, 100.0 + 0.05 * 2048),
        ]
        assert report.fixed_overhead_ms == pytest.approx(100.0, abs=0.5)
        assert report.marginal_ms_per_flow == pytest.approx(0.05, abs=0.001)

    def test_a_dominant_fixed_cost_is_visible_as_such(self) -> None:
        """The finding that matters: a 1-row and a 2048-row call costing the same.

        That is what makes this a batch scorer, and it is what rules out inline enforcement
        regardless of policy.
        """
        report = LoadReport(dataset="synthetic", n_rows=4096, cores=8, mean_flow_bytes=1000)
        report.points = [point(1, 7.0, 139.0), point(2048, 7296.0, 271.5)]
        assert report.fixed_overhead_ms > 130.0
        assert report.marginal_ms_per_flow < 0.1

    def test_one_point_cannot_be_split(self) -> None:
        report = LoadReport(dataset="synthetic", n_rows=10, cores=8, mean_flow_bytes=1000)
        report.points = [point(32, 100.0, 50.0)]
        assert report.marginal_ms_per_flow == 0.0


class TestConversion:
    def test_both_mean_and_median_are_estimated(self) -> None:
        """On UNSW they are 21,227 and 880 - a factor of 24, and only reporting one misleads."""
        X = pd.DataFrame({"sbytes": [100] * 99 + [1_000_000], "dbytes": [0] * 100})
        mean, median = estimate_flow_bytes(X)
        assert median == 100
        assert mean > 10 * median

    def test_nslkdd_column_names_also_resolve(self) -> None:
        X = pd.DataFrame({"src_bytes": [200, 400], "dst_bytes": [100, 100]})
        mean, median = estimate_flow_bytes(X)
        assert (mean, median) == (400, 400)

    def test_a_dataset_without_byte_columns_falls_back(self) -> None:
        mean, median = estimate_flow_bytes(pd.DataFrame({"dur": [1.0, 2.0]}))
        assert mean > 0
        assert median == 0, "no median is honest; a fabricated one is not"

    def test_mbps_is_a_pure_conversion(self) -> None:
        report = LoadReport(dataset="x", n_rows=1, cores=8, mean_flow_bytes=1_000)
        # 1,000 flows/s x 1,000 bytes x 8 bits = 8,000,000 bits/s = 8 Mbps.
        assert report.monitored_mbps(1_000) == 8.0


class TestSummaryKeepsItsCaveats:
    def make(self) -> LoadReport:
        report = LoadReport(
            dataset="unsw-nb15",
            n_rows=20_000,
            cores=8,
            mean_flow_bytes=21_227,
            median_flow_bytes=880,
        )
        report.points = [point(1, 7.0, 139.0), point(2048, 7296.0, 271.5)]
        return report

    def test_it_says_flows_per_second_is_not_link_speed(self) -> None:
        """The line that separates an honest throughput claim from a marketing one."""
        assert "FLOWS PER SECOND IS NOT LINK SPEED" in self.make().summary()

    def test_it_reports_both_conversions_not_the_flattering_one(self) -> None:
        text = self.make().summary()
        assert "MEAN" in text and "MEDIAN" in text
        assert "21,227" in text and "880" in text

    def test_it_states_the_batch_scorer_consequence(self) -> None:
        """Per-flow latency rules out inline enforcement, which is ADR-0001 as a performance fact."""
        text = self.make().summary()
        assert "BATCH SCORER" in text
        assert "ADR-0001" in text

    def test_it_records_that_this_is_not_a_benchmark(self) -> None:
        report = self.make()
        report.notes = ["Measured on a developer machine that was not otherwise idle."]
        assert "not otherwise idle" in report.summary()

    def test_a_report_with_no_points_still_renders(self) -> None:
        report = LoadReport(dataset="x", n_rows=0, cores=8, mean_flow_bytes=1000)
        assert "THROUGHPUT AND LATENCY" in report.summary()
