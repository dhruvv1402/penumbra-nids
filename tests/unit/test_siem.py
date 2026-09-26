"""SIEM connectors: the mock is as strict as Sentinel; the Sentinel client is tested with no network."""

from __future__ import annotations

import json

import pytest

from penumbra.alerts.models import NetworkContext
from penumbra.alerts.schemas import asim
from penumbra.alerts.scoring import ScoringPolicy, build_alert
from penumbra.integrations.siem import NotConfigured, connector
from penumbra.integrations.siem.mock import LocalMockSiem
from penumbra.integrations.siem.sentinel import AzureSentinelSiem, SentinelConfig, batches

ENV = {
    "PENUMBRA_SENTINEL_ENDPOINT": "https://example-dcr.eastus-1.ingest.monitor.azure.com",
    "PENUMBRA_SENTINEL_DCR_ID": "dcr-0123456789abcdef",
    "PENUMBRA_AZURE_TENANT_ID": "tenant",
    "PENUMBRA_AZURE_CLIENT_ID": "client",
    "PENUMBRA_AZURE_CLIENT_SECRET": "s3cr3t-value",
}


def record(i: int = 0) -> dict:
    a = build_alert(
        p_attack=0.95,
        novelty_percentile=0.4,
        policy=ScoringPolicy(),
        family="dos",
        dataset="nslkdd",
        network=NetworkContext(src_ip=f"pseudo:{i:04d}", dst_port=80, protocol="tcp"),
    )
    return asim.to_asim(a)


class FakeTransport:
    def __init__(self, statuses: list[int]) -> None:
        self.statuses = statuses
        self.calls: list[tuple[str, str, dict, bytes]] = []

    def __call__(self, method, url, headers, body):
        self.calls.append((method, url, headers, body))
        if "login.microsoftonline.com" in url:
            return 200, {}, json.dumps({"access_token": f"tok{len(self.calls)}", "expires_in": 3600}).encode()
        status = self.statuses.pop(0) if self.statuses else 204
        return status, {"Retry-After": "2"}, b""

    def ingest_calls(self):
        return [c for c in self.calls if "dataCollectionRules" in c[1]]


class TestMock:
    def test_valid_records_are_written(self, tmp_path) -> None:
        siem = LocalMockSiem(tmp_path)
        result = siem.send([record(1), record(2)])
        assert (result.accepted, result.rejected) == (2, 0)
        assert siem.count() == 2

    def test_invalid_record_is_rejected_with_a_reason(self, tmp_path) -> None:
        bad = record()
        bad["EventSeverity"] = 3  # ASIM wants a string enum; Sentinel would reject this
        result = LocalMockSiem(tmp_path).send([bad])
        assert result.rejected == 1 and result.problems


class TestSentinel:
    def test_not_configured_names_what_is_missing_and_never_the_secret(self) -> None:
        with pytest.raises(NotConfigured, match="PENUMBRA_SENTINEL_DCR_ID"):
            SentinelConfig.from_env({k: v for k, v in ENV.items() if k != "PENUMBRA_SENTINEL_DCR_ID"})
        assert "s3cr3t" not in repr(SentinelConfig.from_env(ENV))

    def test_http_endpoint_is_refused(self) -> None:
        with pytest.raises(NotConfigured):
            SentinelConfig.from_env({**ENV, "PENUMBRA_SENTINEL_ENDPOINT": "http://plain"})

    def test_posts_to_the_dcr_stream_with_a_bearer_token(self) -> None:
        t = FakeTransport([204])
        siem = AzureSentinelSiem(SentinelConfig.from_env(ENV), transport=t, sleep=lambda s: None)
        result = siem.send([record()])
        assert result.accepted == 1
        method, url, headers, body = t.ingest_calls()[0]
        assert url.endswith(
            "/dataCollectionRules/dcr-0123456789abcdef/streams/Custom-PenumbraAlerts?api-version=2023-01-01"
        )
        assert headers["Authorization"].startswith("Bearer tok")
        assert json.loads(body)[0]["DvcAction"] == "Allow"

    def test_throttling_is_retried_once_then_reported(self) -> None:
        slept: list[float] = []
        t = FakeTransport([429, 429])
        siem = AzureSentinelSiem(SentinelConfig.from_env(ENV), transport=t, sleep=slept.append)
        result = siem.send([record()])
        assert slept == [2.0]
        assert result.rejected == 1 and "429" in result.problems[0]

    def test_forbidden_says_what_to_check(self) -> None:
        # What the first live run hit: a new app's role on the DCR had not applied yet.
        t = FakeTransport([403])
        siem = AzureSentinelSiem(SentinelConfig.from_env(ENV), transport=t, sleep=lambda s: None)
        result = siem.send([record()])
        assert result.rejected == 1
        assert "403" in result.problems[0] and "Monitoring Metrics Publisher" in result.problems[0]

    def test_expired_token_is_refreshed_once(self) -> None:
        t = FakeTransport([401, 204])
        siem = AzureSentinelSiem(SentinelConfig.from_env(ENV), transport=t, sleep=lambda s: None)
        assert siem.send([record()]).accepted == 1
        tokens = {c[2]["Authorization"] for c in t.ingest_calls()}
        assert len(tokens) == 2

    def test_batches_stay_under_the_request_cap(self) -> None:
        records = [record(i) for i in range(200)]
        chunks = list(batches(records, limit=20_000))
        assert sum(len(c) for c in chunks) == 200
        assert all(len(json.dumps(c).encode()) <= 20_000 for c in chunks)


def test_connector_selection(tmp_path, monkeypatch) -> None:
    assert isinstance(connector(tmp_path, "mock"), LocalMockSiem)
    assert connector(tmp_path, "none") is None
    for key in ENV:
        monkeypatch.delenv(key, raising=False)
    with pytest.raises(NotConfigured):
        connector(tmp_path, "sentinel")  # never falls back to the mock silently


def test_network_failure_is_reported_not_raised() -> None:
    import urllib.error

    def down(method, url, headers, body):
        raise urllib.error.URLError("connection refused")

    siem = AzureSentinelSiem(SentinelConfig.from_env(ENV), transport=down, sleep=lambda s: None)
    result = siem.send([record()])
    assert result.rejected == 1 and "URLError" in result.problems[0]
