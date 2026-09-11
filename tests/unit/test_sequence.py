"""Entity-graph, windowing and sequence-head tests.

The centre of gravity here is causality. Both feature builders compute a row from that row's past,
and if either ever reads forward the result is a model that scores beautifully in evaluation and
cannot be deployed, because at inference time the future has not happened. That failure is silent:
nothing crashes, the number just gets better.

So it is tested the only way that actually settles it — append future traffic and assert every
earlier row is bit-identical. A model that reads ahead cannot pass that.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from penumbra.features import entity_graph, windows

keras = pytest.importorskip("keras", reason="sequence head needs the optional `dl` extra")


def make_meta(
    n: int = 60, *, sources: int = 3, destinations: int = 20, start: str = "2017-07-05 09:00:00"
) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    base = pd.Timestamp(start)
    return pd.DataFrame(
        {
            "Timestamp": [base + pd.Timedelta(seconds=float(i)) for i in range(n)],
            "Src IP": [f"10.0.0.{i % sources}" for i in range(n)],
            "Dst IP": [f"192.168.1.{int(rng.integers(destinations))}" for _ in range(n)],
            "Dst Port": [int(rng.integers(1, 1024)) for _ in range(n)],
        }
    )


class TestGraphCausality:
    def test_future_traffic_cannot_change_a_past_row(self) -> None:
        """The test that settles it. Everything else here is detail."""
        meta = make_meta(80)
        prefix = entity_graph.compute(meta.iloc[:40])
        full = entity_graph.compute(meta)
        assert np.array_equal(prefix.to_numpy(), full.iloc[:40].to_numpy())

    def test_the_first_flow_from_a_host_has_no_history(self) -> None:
        """A flow must not be in its own fan-out count."""
        meta = make_meta(9, sources=1)
        graph = entity_graph.compute(meta)
        assert graph.loc[0, "g_src_flows"] == 0
        assert graph.loc[0, "g_src_distinct_dst"] == 0

    def test_row_order_is_preserved_when_input_is_not_time_sorted(self) -> None:
        meta = make_meta(40)
        ordered = entity_graph.compute(meta)
        shuffled_index = np.random.default_rng(1).permutation(len(meta))
        shuffled = entity_graph.compute(meta.iloc[shuffled_index].reset_index(drop=True))
        # Same flow, same features, wherever it sits in the frame.
        assert np.allclose(ordered.to_numpy()[shuffled_index], shuffled.to_numpy())


class TestGraphSemantics:
    def test_a_sweep_produces_high_fanout(self) -> None:
        """One source, many destinations, tight window — the thing per-flow features cannot see."""
        base = pd.Timestamp("2017-07-05 09:00:00")
        meta = pd.DataFrame(
            {
                "Timestamp": [base + pd.Timedelta(seconds=i * 0.1) for i in range(30)],
                "Src IP": ["10.0.0.9"] * 30,
                "Dst IP": [f"192.168.1.{i}" for i in range(30)],
                "Dst Port": [445] * 30,
            }
        )
        graph = entity_graph.compute(meta)
        assert graph["g_src_fanout_ratio"].iloc[-1] == pytest.approx(1.0)
        assert graph["g_src_distinct_dst"].iloc[-1] == 29

    def test_repeated_contact_with_one_host_does_not(self) -> None:
        base = pd.Timestamp("2017-07-05 09:00:00")
        meta = pd.DataFrame(
            {
                "Timestamp": [base + pd.Timedelta(seconds=i * 0.1) for i in range(30)],
                "Src IP": ["10.0.0.9"] * 30,
                "Dst IP": ["192.168.1.5"] * 30,
                "Dst Port": [443] * 30,
            }
        )
        graph = entity_graph.compute(meta)
        assert graph["g_src_distinct_dst"].iloc[-1] == 1
        assert graph["g_src_fanout_ratio"].iloc[-1] < 0.05

    def test_port_entropy_separates_a_sweep_from_a_busy_service(self) -> None:
        """Count alone cannot: 200 connections to one port and to 200 ports are both 200."""
        base = pd.Timestamp("2017-07-05 09:00:00")
        common = {"Src IP": ["10.0.0.9"] * 40, "Dst IP": ["192.168.1.5"] * 40}
        times = [base + pd.Timedelta(seconds=i * 0.1) for i in range(40)]
        one_port = entity_graph.compute(pd.DataFrame({**common, "Timestamp": times, "Dst Port": [443] * 40}))
        many_ports = entity_graph.compute(
            pd.DataFrame({**common, "Timestamp": times, "Dst Port": list(range(40))})
        )
        assert one_port["g_src_dport_entropy"].iloc[-1] == pytest.approx(0.0)
        assert many_ports["g_src_dport_entropy"].iloc[-1] > 5.0

    def test_events_outside_the_window_are_forgotten(self) -> None:
        base = pd.Timestamp("2017-07-05 09:00:00")
        meta = pd.DataFrame(
            {
                # Two bursts an hour apart; a 60s window must not join them.
                "Timestamp": [base + pd.Timedelta(seconds=i) for i in range(5)]
                + [base + pd.Timedelta(hours=1, seconds=i) for i in range(5)],
                "Src IP": ["10.0.0.1"] * 10,
                "Dst IP": [f"192.168.1.{i}" for i in range(10)],
                "Dst Port": [80] * 10,
            }
        )
        graph = entity_graph.compute(meta, config=entity_graph.GraphConfig(window_seconds=60.0))
        assert graph.loc[4, "g_src_flows"] == 4
        assert graph.loc[5, "g_src_flows"] == 0, "the first burst is an hour outside the window"

    def test_missing_columns_raise_rather_than_return_zeros(self) -> None:
        """UNSW-NB15 has none of these. Silently emitting zeros would add nine dead features."""
        with pytest.raises(ValueError, match="entity graph needs"):
            entity_graph.compute(pd.DataFrame({"Timestamp": [pd.Timestamp("2017-07-05")]}))


class TestWindows:
    @pytest.fixture
    def frame(self):
        meta = make_meta(60)
        X = pd.DataFrame({"a": np.arange(60.0), "b": np.arange(60.0) * 2})
        y = (np.arange(60) % 3 == 0).astype(int)
        return X, y, meta

    def test_future_traffic_cannot_change_a_past_window(self, frame) -> None:
        X, y, meta = frame
        prefix = windows.build(X.iloc[:30], y[:30], meta.iloc[:30])
        full = windows.build(X, y, meta)
        assert np.array_equal(prefix.X, full.X[:30])
        assert np.array_equal(prefix.mask, full.mask[:30])

    def test_the_current_flow_is_always_last(self, frame) -> None:
        """A recurrent layer must finish on the flow being classified, not somewhere before it."""
        X, y, meta = frame
        seq = windows.build(X, y, meta)
        values = X.to_numpy(dtype=np.float32)
        for slot in range(len(seq.X)):
            assert np.allclose(seq.X[slot, -1], values[seq.index[slot]])
            assert seq.mask[slot, -1]

    def test_short_history_is_padded_at_the_front_and_masked(self, frame) -> None:
        X, y, meta = frame
        seq = windows.build(X, y, meta, config=windows.WindowConfig(length=8))
        # Row 0 is the first flow from its source, so only the last position is real.
        assert seq.mask[0].sum() == 1
        assert np.all(seq.X[0, :-1] == 0)

    def test_a_long_silence_ends_the_sequence(self) -> None:
        base = pd.Timestamp("2017-07-05 09:00:00")
        meta = pd.DataFrame(
            {
                "Timestamp": [base + pd.Timedelta(seconds=i) for i in range(4)]
                + [base + pd.Timedelta(hours=6)],
                "Src IP": ["10.0.0.1"] * 5,
            }
        )
        X = pd.DataFrame({"a": np.arange(5.0)})
        seq = windows.build(X, np.zeros(5, dtype=int), meta)
        assert seq.mask[3].sum() == 4
        assert seq.mask[4].sum() == 1, "six hours later is a new session, not flow five of one"

    def test_stride_thins_the_output_without_thinning_the_history(self, frame) -> None:
        """The property the memory cap rests on.

        If the stride dropped flows from the history too, a strided run would be a different
        experiment on sparser traffic rather than a subsample of the same one.
        """
        X, y, meta = frame
        full = windows.build(X, y, meta)
        strided = windows.build(X, y, meta, keep_every=3)
        assert len(strided.X) == len(range(0, len(full.X), 3))
        # Each kept window is bit-identical to the one the unstrided run produced for that row.
        positions = {int(row): slot for slot, row in enumerate(full.index)}
        for slot, row in enumerate(strided.index):
            assert np.array_equal(strided.X[slot], full.X[positions[int(row)]])

    def test_stride_preserves_prevalence(self, frame) -> None:
        """Which is what leaves TPR and FPR unbiased by the memory cap."""
        X, y, meta = frame
        full = windows.build(X, y, meta)
        strided = windows.build(X, y, meta, keep_every=2)
        assert abs(float(strided.y.mean()) - float(full.y.mean())) < 0.1

    def test_labels_follow_the_current_flow(self, frame) -> None:
        X, y, meta = frame
        seq = windows.build(X, y, meta)
        assert np.array_equal(seq.y, y[seq.index])

    def test_windows_only_contain_flows_from_the_same_source(self, frame) -> None:
        X, y, meta = frame
        seq = windows.build(X, y, meta)
        sources = meta["Src IP"].to_numpy()
        # Column `a` equals the row position, so a window's values name their own rows.
        for slot in range(0, len(seq.X), 7):
            rows = seq.X[slot, seq.mask[slot], 0].astype(int)
            assert len(set(sources[rows])) == 1


class TestSequenceDetector:
    @pytest.fixture
    def learnable(self):
        """A signal that exists ONLY in the sequence.

        Every flow is identical in isolation; what separates the classes is whether the preceding
        flows from that source went to many destinations or to one. A per-flow model cannot do
        better than chance here, which is the point of the fixture.
        """
        from penumbra.features.windows import Sequences

        rng = np.random.default_rng(0)
        n, k, d = 900, 8, 4
        X = rng.normal(0, 1, (n, k, d)).astype(np.float32)
        y = (rng.random(n) < 0.4).astype(np.int8)
        # Attacks: the history ramps. Benign: the history is flat. The last row is identical.
        ramp = np.linspace(0, 3, k, dtype=np.float32)[None, :, None]
        X[y == 1] += ramp
        X[:, -1, :] = 0.5
        return Sequences(
            X=X,
            mask=np.ones((n, k), dtype=bool),
            y=y,
            index=np.arange(n),
            features=[f"f{i}" for i in range(d)],
        )

    def test_it_learns_a_signal_that_lives_only_in_the_sequence(self, learnable) -> None:
        from sklearn.metrics import roc_auc_score

        from penumbra.models.sequence import SequenceConfig, SequenceDetector

        detector = SequenceDetector(SequenceConfig(epochs=6, batch_size=128, gru_units=16)).fit(learnable)
        auc = roc_auc_score(learnable.y, detector.score(learnable))
        assert auc > 0.85, f"the sequence signal is learnable by construction; got {auc:.3f}"

    def test_padding_stays_zero_after_scaling(self, learnable) -> None:
        """A padded position carrying the negative of the training mean is not padding."""
        from penumbra.models.sequence import SequenceDetector

        masked = learnable
        masked.mask[:, :3] = False
        detector = SequenceDetector()
        detector._fit_scaler(masked)
        prepared = detector._prepare(masked)
        assert np.all(prepared[:, :3, :] == 0.0)

    def test_one_extreme_row_does_not_squash_everything_else(self, learnable) -> None:
        """Why median/IQR instead of mean/std.

        CICIDS2017 has columns whose maximum is over 200 standard deviations from their own mean
        (`Total TCP Flow Time` reaches 7.2e9). Z-scaling leaves one row at +200 and the rest piled
        against zero, and +200 through two convolutions into a GRU saturates the network - which is
        exactly what happened: validation AUC 0.9999 in epoch 1, then a constant 0.500 forever.
        """
        from penumbra.models.sequence import CLIP_SIGMAS, SequenceDetector

        spiked = learnable
        spiked.X[0, 0, 0] = 1e9

        detector = SequenceDetector()
        detector._fit_scaler(spiked)
        prepared = detector._prepare(spiked)

        assert np.abs(prepared).max() <= CLIP_SIGMAS + 1e-6
        # The ordinary rows must keep their spread rather than collapsing onto zero.
        assert float(prepared[1:, :, 0].std()) > 0.1

    def test_a_collapsed_run_is_flagged_rather_than_reported(self) -> None:
        """Early stopping hides divergence perfectly: best weights restored, clean exit, a number.

        The first CICIDS run reported a one-epoch model as the architecture's verdict because of
        exactly this. The flag is what stops that being publishable by accident.
        """
        from penumbra.models.sequence import TrainingHistory

        healthy = TrainingHistory(epochs_run=6, best_val_auc=0.97, collapsed_epochs=0)
        assert not healthy.collapsed
        assert "WARNING" not in healthy.summary()

        collapsed = TrainingHistory(epochs_run=4, best_val_auc=0.9999, collapsed_epochs=3)
        assert collapsed.collapsed
        assert "diverged to a constant output" in collapsed.summary()

    def test_scoring_before_fitting_raises(self, learnable) -> None:
        from penumbra.models.sequence import SequenceDetector

        with pytest.raises(RuntimeError, match="not fitted"):
            SequenceDetector().score(learnable)

    def test_round_trip_through_disk_preserves_scores(self, learnable, tmp_path) -> None:
        from penumbra.models.sequence import SequenceConfig, SequenceDetector

        detector = SequenceDetector(SequenceConfig(epochs=2, batch_size=128, gru_units=8)).fit(learnable)
        before = detector.score(learnable)
        detector.save(tmp_path)
        after = SequenceDetector.load(tmp_path).score(learnable)
        assert np.allclose(before, after, atol=1e-5)
