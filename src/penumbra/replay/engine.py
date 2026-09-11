"""Replay: stream a dataset through the detector as if it were live traffic.

This is what the console watches. It is also how the drift demonstration is staged - the injector
perturbs the stream at a known change point, the monitor fires, and the measured recall visibly
falls, with a ground-truth change point so the detection delay is a measurement rather than an
impression.

Three modes:

  offline   score everything, correlate, print. Fastest path to the numbers.
  ingest    POST batches to a running API, which persists and pushes to connected consoles.
  fixture   replay pre-scored alerts from a checked-in JSON file. No model, no dataset, no
            network. This is the demo-day fallback and it exists because three independent things
            that can fail on stage is three too many.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from penumbra.alerts.builder import alerts_from_scores
from penumbra.alerts.correlate import Correlator, EntitiesRequired
from penumbra.alerts.models import Alert
from penumbra.models.detector import PenumbraDetector


@dataclass
class ReplayConfig:
    batch_size: int = 200
    # Wall-clock seconds between batches. 0.4 looks alive without being hard to follow; 0 is for
    # scoring runs where nobody is watching.
    delay: float = 0.4
    max_rows: int | None = None
    # Spread synthetic timestamps across this window so correlation has something to group by.
    span: timedelta = timedelta(hours=2)
    inject_drift: str | None = None


@dataclass
class ReplayStats:
    rows_scored: int = 0
    alerts_emitted: int = 0
    by_verdict: dict[str, int] = field(default_factory=dict)
    by_lane: dict[str, int] = field(default_factory=dict)
    incidents: int = 0
    elapsed_seconds: float = 0.0

    @property
    def rows_per_second(self) -> float:
        return self.rows_scored / self.elapsed_seconds if self.elapsed_seconds else 0.0

    @property
    def alert_rate(self) -> float:
        return self.alerts_emitted / self.rows_scored if self.rows_scored else 0.0

    def summary(self) -> str:
        lines = [
            f"  scored {self.rows_scored:,} flows in {self.elapsed_seconds:.1f}s "
            f"({self.rows_per_second:,.0f} flows/s)",
            f"  emitted {self.alerts_emitted:,} alerts ({self.alert_rate:.2%} of flows)",
        ]
        if self.by_verdict:
            lines.append(f"  verdicts: {self.by_verdict}")
        if self.by_lane:
            lines.append(f"  lanes:    {self.by_lane}")
        if self.incidents:
            lines.append(
                f"  correlated into {self.incidents:,} incidents "
                f"({self.alerts_emitted / self.incidents:.1f} events each)"
            )
        lines.append("")
        lines.append(
            "  Throughput is flows/second, not link speed. A flow record summarises many packets, "
            "so\n  the two are not interchangeable and this number should not be quoted as bandwidth."
        )
        return "\n".join(lines)


def batches(df: pd.DataFrame, size: int) -> Iterator[tuple[int, pd.DataFrame]]:
    for start in range(0, len(df), size):
        yield start, df.iloc[start : start + size]


def synthetic_timestamps(n: int, span: timedelta, *, end: datetime | None = None) -> list[datetime]:
    """Spread n events evenly across a window ending now.

    Only for datasets with no timestamps of their own. The correlation window needs *some* time
    axis; this one is clearly synthetic and is never used to compute a reported metric.
    """
    end = end or datetime.now(UTC)
    start = end - span
    step = span / max(n - 1, 1)
    return [start + step * i for i in range(n)]


def replay(
    detector: PenumbraDetector,
    X: pd.DataFrame,
    *,
    config: ReplayConfig | None = None,
    dataset: str = "unsw",
    entities: list[str] | None = None,
    timestamps: list[datetime] | None = None,
    on_batch: Any = None,
) -> tuple[list[Alert], ReplayStats]:
    """Score a dataset in batches, pacing to look live."""
    config = config or ReplayConfig()
    if config.max_rows:
        X = X.head(config.max_rows)
        if entities:
            entities = entities[: config.max_rows]

    stamps = timestamps or synthetic_timestamps(len(X), config.span)
    stats = ReplayStats()
    collected: list[Alert] = []
    started = time.perf_counter()

    for start, chunk in batches(X, config.batch_size):
        scored = detector.score(chunk)
        chunk_entities = entities[start : start + len(chunk)] if entities else None
        alerts = alerts_from_scores(detector, chunk, scored, dataset=dataset, entities=chunk_entities)

        for offset, alert in enumerate(alerts):
            idx = start + offset
            if idx < len(stamps):
                alert.timestamp = stamps[idx]

        collected.extend(alerts)
        stats.rows_scored += len(chunk)
        stats.alerts_emitted += len(alerts)
        for alert in alerts:
            stats.by_verdict[alert.verdict.value] = stats.by_verdict.get(alert.verdict.value, 0) + 1
            stats.by_lane[alert.lane.value] = stats.by_lane.get(alert.lane.value, 0) + 1

        if on_batch:
            on_batch(start, alerts, stats)
        if config.delay:
            time.sleep(config.delay)

    stats.elapsed_seconds = time.perf_counter() - started
    return collected, stats


def correlate_if_possible(alerts: list[Alert], *, dataset: str) -> tuple[list[Any], str | None]:
    """Correlate, or explain why not.

    Returns (incidents, reason_it_was_skipped). The reason is surfaced rather than swallowed,
    because "we did not measure this" and "this measured zero" are different statements.
    """
    try:
        result = Correlator().correlate(alerts, dataset=dataset)
        return result.incidents, None
    except EntitiesRequired as exc:
        return [], str(exc)


# =================================================================================================
# Ingest
# =================================================================================================


class IngestClient:
    """POSTs alerts to a running API.

    urllib rather than httpx so the replay works from a bare install. Failures are reported and
    skipped rather than raised: a console that disconnects mid-demo should not stop the stream.
    """

    # B310 on the urlopen calls below is suppressed because __init__ rejects any scheme but
    # http(s), which is the mitigation the rule asks for.
    def __init__(self, base_url: str = "http://127.0.0.1:8000", token: str = "") -> None:
        if urllib.parse.urlparse(base_url).scheme.lower() not in {"http", "https"}:
            raise ValueError(f"ingest URL must be http(s), got {base_url!r}")
        self.base_url = base_url.rstrip("/")
        self.token = token

    def login(self, username: str, password: str) -> bool:
        try:
            body = json.dumps({"username": username, "password": password}).encode()
            req = urllib.request.Request(
                f"{self.base_url}/auth/login",
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=10) as resp:  # nosec B310
                self.token = json.loads(resp.read())["access_token"]
            return True
        except (urllib.error.URLError, KeyError, TimeoutError):
            return False

    def send(self, alerts: list[Alert]) -> int:
        if not alerts or not self.token:
            return 0
        payload = json.dumps({"alerts": [json.loads(a.model_dump_json()) for a in alerts]}).encode()
        req = urllib.request.Request(
            f"{self.base_url}/ingest",
            data=payload,
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.token}"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:  # nosec B310
                return int(json.loads(resp.read()).get("ingested", 0))
        except (urllib.error.URLError, TimeoutError, ValueError):
            return 0


# =================================================================================================
# Fixture mode - the demo-day fallback
# =================================================================================================


def write_fixture(alerts: list[Alert], incidents: list[Any], path: Path) -> Path:
    """Freeze a scored run to disk.

    Replaying this needs no model, no dataset and no network, which is the entire point: on stage,
    the fewest moving parts wins.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "generated_at": datetime.now(UTC).isoformat(),
                "alerts": [json.loads(a.model_dump_json()) for a in alerts],
                "incidents": [json.loads(i.model_dump_json()) for i in incidents],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return path


def read_fixture(path: Path) -> tuple[list[Alert], list[dict[str, Any]]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [Alert.model_validate(a) for a in data["alerts"]], data.get("incidents", [])


# =================================================================================================
# Drift-injected replay
# =================================================================================================


def replay_with_drift(
    detector: PenumbraDetector,
    X: pd.DataFrame,
    *,
    scenario: str = "abrupt",
    config: ReplayConfig | None = None,
    dataset: str = "unsw",
) -> dict[str, Any]:
    """Replay a stream that changes halfway through, and measure whether the monitor notices.

    The change point is known, so detection delay is measured rather than asserted - which is the
    difference between demonstrating drift monitoring and claiming it.
    """
    from penumbra.drift import injector, streaming

    config = config or ReplayConfig(delay=0.0)
    plan = injector.scenario(scenario, len(X))
    drifted = injector.inject(X, plan, numeric_features=list(X.select_dtypes("number").columns))

    clean_scores = detector.score(X)["p_attack"].to_numpy()
    drifted_scores = detector.score(drifted)["p_attack"].to_numpy()

    # The stream a monitor would actually see: normal traffic, then the perturbed traffic.
    stream = np.concatenate([clean_scores[: plan.change_point], drifted_scores[plan.change_point :]])
    detectors = streaming.compare_detectors(stream)

    return {
        "scenario": scenario,
        "change_point": plan.change_point,
        "n_rows": len(X),
        "mean_score_before": float(stream[: plan.change_point].mean()),
        "mean_score_after": float(stream[plan.change_point :].mean()),
        "detectors": {
            name: {
                "alarms": len(r.alarms),
                "detection_delay": r.detection_delay(plan.change_point),
                "false_alarms_before": r.false_alarms_before(plan.change_point),
            }
            for name, r in detectors.items()
        },
    }
