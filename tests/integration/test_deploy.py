"""What a public deployment relies on: no known logins, a guest that can only look, and demo data
that comes back after a restart."""

from __future__ import annotations

import json
import os

import pytest

os.environ.setdefault("PENUMBRA_ALLOW_DEMO_USERS", "1")

from fastapi.testclient import TestClient  # noqa: E402

from penumbra.alerts.models import NetworkContext  # noqa: E402
from penumbra.alerts.scoring import ScoringPolicy, build_alert  # noqa: E402
from penumbra.api import app as app_module  # noqa: E402
from penumbra.api.app import app, state  # noqa: E402
from penumbra.api.security.audit import AuditLog  # noqa: E402
from penumbra.api.security.auth import UserStore  # noqa: E402
from penumbra.api.security.rbac import Principal, Role  # noqa: E402
from penumbra.integrations.siem.mock import LocalMockSiem  # noqa: E402
from penumbra.storage.sqlite import SqliteRepository  # noqa: E402


def _alert(i: int):
    alert = build_alert(
        p_attack=0.95,
        novelty_percentile=0.3,
        policy=ScoringPolicy(),
        family="Reconnaissance",
        network=NetworkContext(src_ip=f"pseudo:h{i}", dst_port=445, protocol="tcp"),
    )
    alert.raw_features = {"segment": "dmz"}
    return alert


@pytest.fixture()
def client(tmp_path) -> TestClient:
    state.repo = SqliteRepository(":memory:")
    state.audit = AuditLog()
    state.siem = LocalMockSiem(tmp_path / "siem")
    state.repo.save_alert(_alert(1))
    return TestClient(app)


def test_guest_access_is_off_unless_enabled(client, monkeypatch) -> None:
    monkeypatch.delenv("PENUMBRA_GUEST_ACCESS", raising=False)
    assert client.post("/auth/guest").status_code == 404


def test_guest_can_look_and_cannot_touch(client, monkeypatch) -> None:
    monkeypatch.setenv("PENUMBRA_GUEST_ACCESS", "1")
    resp = client.post("/auth/guest")
    assert resp.status_code == 200 and resp.json()["role"] == "guest"
    headers = {"Authorization": f"Bearer {resp.json()['access_token']}"}

    assert client.get("/alerts", headers=headers).status_code == 200
    assert client.get("/incidents", headers=headers).status_code == 200
    alert_id = client.get("/alerts", headers=headers).json()[0]["alert_id"]

    writes = [
        ("post", f"/alerts/{alert_id}/verdict", {"verdict": "false_positive"}),
        ("post", "/ingest", {"alerts": [json.loads(_alert(2).model_dump_json())]}),
        ("post", "/incidents/bulk", []),
        ("post", "/feedback/promote", {"verdict_ids": [1]}),
        ("get", "/audit", None),
    ]
    for method, path, body in writes:
        r = getattr(client, method)(path, headers=headers, **({"json": body} if body is not None else {}))
        assert r.status_code in (403, 422), (path, r.status_code)
        assert r.status_code == 403 or path == "/feedback/promote", (path, r.status_code)
    assert "auth.guest" in [e.action for e in state.audit.tail(20)]


def test_guest_cannot_log_in_with_a_password(client) -> None:
    assert client.post("/auth/login", json={"username": "guest", "password": ""}).status_code == 401


def test_configured_users_come_from_the_environment() -> None:
    store = UserStore()
    created = store.seed_configured_users(
        {"PENUMBRA_SENIOR_PASSWORD": "correct-horse-battery-staple", "PENUMBRA_ANALYST_PASSWORD": ""}
    )
    assert created == ["senior"]
    assert store.authenticate("senior", "correct-horse-battery-staple") is not None
    assert store.authenticate("analyst", "analyst") is None


def test_a_short_password_refuses_startup() -> None:
    with pytest.raises(RuntimeError, match="shorter"):
        UserStore().seed_configured_users({"PENUMBRA_ADMIN_PASSWORD": "admin"})


def test_seed_fills_an_empty_store_once(tmp_path) -> None:
    state.repo = SqliteRepository(":memory:")
    state.audit = AuditLog()
    fixture = tmp_path / "f.json"
    alerts = [_alert(i) for i in range(3)]
    fixture.write_text(
        json.dumps(
            {
                "alerts": [json.loads(a.model_dump_json()) for a in alerts],
                "incidents": [
                    {
                        "incident_id": "INC-1",
                        "title": "t",
                        "severity": "High",
                        "lane": "known_threat",
                        "priority": 80,
                        "entity": "pseudo:h1",
                        "alert_ids": [alerts[0].alert_id],
                    },
                    {
                        "incident_id": "INC-ORPHAN",
                        "title": "t",
                        "severity": "High",
                        "lane": "known_threat",
                        "priority": 80,
                        "entity": "pseudo:h9",
                        "alert_ids": ["not-shipped"],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    assert app_module.seed_fixtures(tmp_path) == (3, 1)
    assert app_module.seed_fixtures(tmp_path) == (0, 0)  # already populated: untouched
    admin = Principal(username="t", role=Role.ADMIN)
    assert state.repo.counts(admin)["alerts_total"] == 3
