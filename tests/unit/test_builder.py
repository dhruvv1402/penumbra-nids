"""Alert construction: positions, feature payloads, and the timestamp misalignment they fixed."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd

from penumbra.alerts.builder import alerts_with_positions, feature_payload
from penumbra.replay import engine


class StubDetector:
    """Flags exactly the rows whose `src_bytes` is odd."""

    metadata = None
    policy = None

    def score(self, X: pd.DataFrame) -> pd.DataFrame:
        fired = (X["src_bytes"].to_numpy() % 2 == 1).astype(int)
        return pd.DataFrame(
            {
                "p_attack": np.where(fired == 1, 0.99, 0.01),
                "novelty_percentile": 0.1,
                "agreement": 0,
                "fired": fired,
                "family": np.where(fired == 1, "dos", "normal"),
                "conformal_abstains": False,
                "conformal_set": [[] for _ in range(len(X))],
            },
            index=X.index,
        )


def frame(n: int = 10) -> pd.DataFrame:
    return pd.DataFrame({"src_bytes": np.arange(n), "service": ["http"] * n, "rate": np.linspace(0, 1, n)})


def test_positions_point_at_the_flagged_rows() -> None:
    positioned = alerts_with_positions(StubDetector(), frame())
    assert [pos for pos, _ in positioned] == [1, 3, 5, 7, 9]


def test_alerts_carry_their_feature_row() -> None:
    positioned = alerts_with_positions(StubDetector(), frame())
    pos, alert = positioned[2]
    assert alert.raw_features["src_bytes"] == pos
    assert alert.raw_features["service"] == "http"


def test_feature_payload_is_json_safe() -> None:
    row = pd.Series({"a": np.int64(3), "b": np.float32(np.nan), "c": np.bool_(True), "d": "tcp"})
    assert feature_payload(row) == {"a": 3, "b": None, "c": True, "d": "tcp"}


def test_replay_timestamps_follow_the_row_not_the_alert_index() -> None:
    # Before the fix, the k-th alert got the k-th row's timestamp. With every other row unflagged,
    # alert 1 (row 3) was stamped with row 1's time.
    start = datetime(2026, 1, 1, tzinfo=UTC)
    stamps = [start + timedelta(seconds=i) for i in range(10)]
    alerts, _ = engine.replay(
        StubDetector(),
        frame(),
        config=engine.ReplayConfig(delay=0.0, batch_size=4),
        timestamps=stamps,
    )
    rows = [int(a.raw_features["src_bytes"]) for a in alerts]
    assert [a.timestamp for a in alerts] == [stamps[r] for r in rows]


def test_fixture_round_trip_keeps_features(tmp_path) -> None:
    alerts = [a for _, a in alerts_with_positions(StubDetector(), frame())]
    path = engine.write_fixture(alerts, [], tmp_path / "f.json")
    back, incidents = engine.read_fixture(path)
    assert [a.raw_features for a in back] == [a.raw_features for a in alerts]
    assert incidents == []
