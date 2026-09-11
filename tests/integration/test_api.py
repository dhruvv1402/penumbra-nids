"""API integration tests.

These assert the access-control boundaries end to end rather than at the unit level, because a
permission that is enforced in `rbac.py` and forgotten in a router is not enforced.

The last test in this file is the important one: it walks the whole OpenAPI surface and fails if any
route appears that could block traffic.
"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("PENUMBRA_ALLOW_DEMO_USERS", "1")

from fastapi.testclient import TestClient  # noqa: E402

from penumbra.alerts.models import Incident, Lane, NetworkContext, Severity  # noqa: E402
from penumbra.alerts.scoring import ScoringPolicy, build_alert  # noqa: E402
from penumbra.api.app import app, state  # noqa: E402
from penumbra.api.security.audit import AuditLog  # noqa: E402
from penumbra.storage.sqlite import SqliteRepository  # noqa: E402


@pytest.fixture(scope="module")
def client() -> TestClient:
    # Hermetic: an in-memory repository and a fresh audit log, so the suite does not inherit
    # whatever a previous run left in artifacts/penumbra.db.
    state.repo = SqliteRepository(":memory:")
    state.audit = AuditLog()

    policy = ScoringPolicy()
    alerts = []
    for i in range(30):
        alert = build_alert(
            p_attack=0.95 if i % 3 else 0.05,
            novelty_percentile=0.997 if i % 3 == 0 else 0.3,
            policy=policy,
            family="Reconnaissance" if i % 3 else None,
            agreement=3,
            network=NetworkContext(src_ip=f"pseudo:host{i % 5:02d}", dst_port=445, protocol="tcp"),
        )
        alert.raw_features = {"segment": "dmz" if i % 2 else "ot"}
        alerts.append(alert)
    state.repo.save_alerts(alerts)
    state.repo.save_incident(
        Incident(
            incident_id="INC-TEST-1",
            title="Port scan from pseudo:host01",
            severity=Severity.HIGH,
            lane=Lane.KNOWN_THREAT,
            priority=88,
            entity="pseudo:host01",
            family="Reconnaissance",
            alert_ids=[a.alert_id for a in alerts[:12]],
            event_count=3812,
            distinct_destinations=254,
        )
    )
    return TestClient(app)


def _token(client: TestClient, user: str) -> dict[str, str]:
    resp = client.post("/auth/login", json={"username": user, "password": user})
    assert resp.status_code == 200
    return {"Authorization": f"Bearer {resp.json()['access_token']}"}


class TestAuth:
    def test_unauthenticated_is_rejected(self, client: TestClient) -> None:
        assert client.get("/alerts").status_code == 401

    def test_bad_credentials_rejected(self, client: TestClient) -> None:
        assert (
            client.post("/auth/login", json={"username": "analyst", "password": "wrong"}).status_code == 401
        )

    def test_garbage_token_rejected(self, client: TestClient) -> None:
        assert client.get("/alerts", headers={"Authorization": "Bearer nonsense"}).status_code == 401

    def test_login_returns_role_and_segments(self, client: TestClient) -> None:
        body = client.post("/auth/login", json={"username": "analyst", "password": "analyst"}).json()
        assert body["role"] == "analyst"
        assert set(body["segments"]) == {"dmz", "corp"}

    def test_health_needs_no_auth(self, client: TestClient) -> None:
        assert client.get("/health").status_code == 200


class TestRowLevelScoping:
    def test_analyst_sees_only_permitted_segments(self, client: TestClient) -> None:
        analyst = len(client.get("/alerts?limit=1000", headers=_token(client, "analyst")).json())
        senior = len(client.get("/alerts?limit=1000", headers=_token(client, "senior")).json())
        assert analyst < senior
        assert analyst == 15  # the dmz half

    def test_scoping_applies_to_the_api_not_just_the_console(self, client: TestClient) -> None:
        # Same query, direct to the API, no UI involved.
        for alert in client.get("/alerts?limit=1000", headers=_token(client, "analyst")).json():
            assert alert["raw_features"].get("segment") in {"dmz", "corp", None}


class TestPermissionBoundaries:
    def test_analyst_may_record_a_verdict(self, client: TestClient) -> None:
        resp = client.post(
            "/incidents/INC-TEST-1/verdict",
            headers=_token(client, "analyst"),
            json={"verdict": "true_positive", "note": "confirmed scan"},
        )
        assert resp.status_code == 200
        assert resp.json()["queued_for_training"] is True
        # Recorded, but NOT training anything yet.
        assert resp.json()["promoted"] is False

    def test_analyst_cannot_promote_a_verdict(self, client: TestClient) -> None:
        # THREAT_MODEL T1: poisoning the model requires a second, senior account.
        resp = client.post(
            "/feedback/promote",
            headers=_token(client, "analyst"),
            json={"incident_ids": ["INC-TEST-1"]},
        )
        assert resp.status_code == 403

    def test_senior_can_promote(self, client: TestClient) -> None:
        client.post(
            "/incidents/INC-TEST-1/verdict",
            headers=_token(client, "analyst"),
            json={"verdict": "true_positive"},
        )
        resp = client.post(
            "/feedback/promote",
            headers=_token(client, "senior"),
            json={"incident_ids": ["INC-TEST-1"]},
        )
        assert resp.status_code == 200

    def test_analyst_cannot_read_the_audit_log(self, client: TestClient) -> None:
        assert client.get("/audit", headers=_token(client, "analyst")).status_code == 403

    def test_senior_can_read_the_audit_log(self, client: TestClient) -> None:
        body = client.get("/audit", headers=_token(client, "senior")).json()
        assert body["chain_intact"] is True

    def test_denied_requests_are_audited(self, client: TestClient) -> None:
        client.get("/audit", headers=_token(client, "analyst"))  # 403
        entries = client.get("/audit", headers=_token(client, "senior")).json()["entries"]
        assert any(e["action"] == "auth.denied" for e in entries)


class TestAuditIntegrity:
    def test_health_reports_chain_state(self, client: TestClient) -> None:
        body = client.get("/health").json()
        assert body["audit_chain_intact"] is True

    def test_verdicts_appear_in_the_audit_log(self, client: TestClient) -> None:
        client.post(
            "/incidents/INC-TEST-1/verdict",
            headers=_token(client, "analyst"),
            json={"verdict": "false_positive", "note": "authorised scanner"},
        )
        entries = client.get("/audit?limit=200", headers=_token(client, "senior")).json()["entries"]
        verdicts = [e for e in entries if e["action"] == "analyst.verdict"]
        assert verdicts
        assert verdicts[-1]["actor"] == "analyst"


class TestAlertNotBlock:
    """ADR-0001, enforced against the live API surface."""

    def test_health_declares_it_does_not_block(self, client: TestClient) -> None:
        assert client.get("/health").json()["blocks_traffic"] is False

    def test_no_route_can_block_traffic(self, client: TestClient) -> None:
        # The whole OpenAPI surface, not a curated list - so a route added later is caught too.
        paths = client.get("/openapi.json").json()["paths"]
        forbidden = ("block", "deny", "drop", "quarantine", "firewall", "reset", "isolate")
        for path in paths:
            assert not any(word in path.lower() for word in forbidden), (
                f"route {path} looks like enforcement. Penumbra alerts; it does not block. "
                "See docs/adr/0001-alert-not-block.md"
            )

    def test_alerts_always_require_analyst_approval(self, client: TestClient) -> None:
        for alert in client.get("/alerts?limit=50", headers=_token(client, "senior")).json():
            assert alert["requires_analyst_approval"] is True
