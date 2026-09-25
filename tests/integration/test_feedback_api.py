"""The feedback loop end to end: alert verdicts, suppression, the two-person rule, rate limits.

Each of these is a control against a specific failure, and each is asserted at the HTTP boundary
because a control enforced in a helper and forgotten in a route is not enforced.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta

import pytest

os.environ.setdefault("PENUMBRA_ALLOW_DEMO_USERS", "1")

from fastapi.testclient import TestClient  # noqa: E402

from penumbra.alerts.models import NetworkContext  # noqa: E402
from penumbra.alerts.scoring import ScoringPolicy, build_alert  # noqa: E402
from penumbra.api import app as app_module  # noqa: E402
from penumbra.api.app import app, state  # noqa: E402
from penumbra.api.security.audit import AuditLog  # noqa: E402
from penumbra.integrations.siem.mock import LocalMockSiem  # noqa: E402
from penumbra.storage.sqlite import SqliteRepository  # noqa: E402


def _alert(i: int, *, p: float = 0.95, service: str = "private", segment: str = "dmz"):
    a = build_alert(
        p_attack=p,
        novelty_percentile=0.3,
        policy=ScoringPolicy(),
        family="probe" if p >= 0.5 else None,
        agreement=0,
        network=NetworkContext(dst_port=None, protocol="tcp"),
    )
    a.raw_features = {"segment": segment, "service": service, "src_bytes": float(i), "flag": "S0"}
    return a


@pytest.fixture()
def client(tmp_path) -> TestClient:
    state.repo = SqliteRepository(":memory:")
    state.audit = AuditLog()
    state.siem = LocalMockSiem(tmp_path / "siem")
    return TestClient(app)


def _token(client: TestClient, user: str) -> dict[str, str]:
    resp = client.post("/auth/login", json={"username": user, "password": user})
    return {"Authorization": f"Bearer {resp.json()['access_token']}"}


class TestAlertVerdicts:
    def test_uncorrelated_alert_accepts_a_verdict(self, client: TestClient) -> None:
        # The bug this replaces: NSL-KDD has no IPs, so nothing correlates, so every verdict button
        # in the console answered "not yet correlated into an incident".
        alert = _alert(1)
        state.repo.save_alert(alert)
        resp = client.post(
            f"/alerts/{alert.alert_id}/verdict",
            json={"verdict": "false_positive"},
            headers=_token(client, "analyst"),
        )
        assert resp.status_code == 200
        assert resp.json()["queued_for_training"] is True
        pending = client.get("/feedback/pending", headers=_token(client, "senior")).json()
        assert [p["target_id"] for p in pending] == [alert.alert_id]
        assert pending[0]["target_kind"] == "alert"

    def test_verdict_is_segment_scoped(self, client: TestClient) -> None:
        hidden = _alert(2, segment="ot")  # the analyst sees dmz and corp only
        state.repo.save_alert(hidden)
        resp = client.post(
            f"/alerts/{hidden.alert_id}/verdict",
            json={"verdict": "false_positive"},
            headers=_token(client, "analyst"),
        )
        assert resp.status_code == 404

    def test_benign_by_policy_proposes_a_rule_and_is_not_a_label(self, client: TestClient) -> None:
        alert = _alert(3)
        state.repo.save_alert(alert)
        body = client.post(
            f"/alerts/{alert.alert_id}/verdict",
            json={"verdict": "benign_by_policy"},
            headers=_token(client, "analyst"),
        ).json()
        assert body["queued_for_training"] is False
        assert body["proposed_suppression"]["service"] == "private"


class TestTwoPersonRule:
    def test_cannot_promote_own_verdict(self, client: TestClient) -> None:
        alert = _alert(4)
        state.repo.save_alert(alert)
        senior = _token(client, "senior")
        client.post(f"/alerts/{alert.alert_id}/verdict", json={"verdict": "false_positive"}, headers=senior)

        own = client.post("/feedback/promote", json={"incident_ids": [alert.alert_id]}, headers=senior).json()
        assert own["promoted"] == 0
        assert own["refused_self_approval"] == [alert.alert_id]

        other = client.post(
            "/feedback/promote", json={"incident_ids": [alert.alert_id]}, headers=_token(client, "admin")
        ).json()
        assert other["promoted"] == 1

    def test_promoted_verdict_becomes_a_training_row(self, client: TestClient) -> None:
        alert = _alert(5, p=0.97)
        state.repo.save_alert(alert)
        client.post(
            f"/alerts/{alert.alert_id}/verdict",
            json={"verdict": "false_positive"},
            headers=_token(client, "analyst"),
        )
        client.post(
            "/feedback/promote", json={"incident_ids": [alert.alert_id]}, headers=_token(client, "senior")
        )
        rows = state.repo.promoted_training_rows()
        assert len(rows) == 1
        assert rows[0]["label"] == 0
        assert rows[0]["features"]["src_bytes"] == 5.0

    def test_benign_by_policy_never_reaches_training(self, client: TestClient) -> None:
        alert = _alert(6)
        state.repo.save_alert(alert)
        client.post(
            f"/alerts/{alert.alert_id}/verdict",
            json={"verdict": "benign_by_policy"},
            headers=_token(client, "analyst"),
        )
        client.post(
            "/feedback/promote", json={"incident_ids": [alert.alert_id]}, headers=_token(client, "senior")
        )
        assert state.repo.promoted_training_rows() == []


class TestIntegrityFlags:
    def test_confident_contradiction_is_flagged(self, client: TestClient) -> None:
        alert = _alert(7, p=0.99)
        state.repo.save_alert(alert)
        client.post(
            f"/alerts/{alert.alert_id}/verdict",
            json={"verdict": "false_positive"},
            headers=_token(client, "analyst"),
        )
        pending = client.get("/feedback/pending", headers=_token(client, "senior")).json()
        assert "confident_contradiction" in pending[0]["flags"]
        assert pending[0]["flag_reasons"]

    def test_analyst_cannot_read_the_promotion_queue(self, client: TestClient) -> None:
        assert client.get("/feedback/pending", headers=_token(client, "analyst")).status_code == 403


class TestRateLimit:
    def test_bulk_relabelling_is_throttled_and_audited(self, client: TestClient, monkeypatch) -> None:
        monkeypatch.setattr(app_module, "VERDICT_RATE_LIMIT", 3)
        analyst = _token(client, "analyst")
        codes = []
        for i in range(5):
            a = _alert(100 + i)
            state.repo.save_alert(a)
            codes.append(
                client.post(
                    f"/alerts/{a.alert_id}/verdict", json={"verdict": "false_positive"}, headers=analyst
                ).status_code
            )
        assert codes == [200, 200, 200, 429, 429]
        actions = [e.action for e in state.audit.tail(20)]
        assert "verdict.rate_limited" in actions


class TestSuppression:
    def _rule(self, client: TestClient, **overrides) -> tuple[int, dict]:
        body = {
            "match": {"service": "private", "family": "probe"},
            "reason": "nightly vuln scanner",
            "days": 14,
        }
        body.update(overrides)
        resp = client.post("/suppressions", json=body, headers=_token(client, "senior"))
        return resp.status_code, resp.json()

    def test_analyst_cannot_create(self, client: TestClient) -> None:
        resp = client.post(
            "/suppressions",
            json={"match": {"service": "private"}, "reason": "nightly vuln scanner", "days": 14},
            headers=_token(client, "analyst"),
        )
        assert resp.status_code == 403

    def test_matching_ingest_is_reclassified_not_dropped(self, client: TestClient) -> None:
        code, rule = self._rule(client)
        assert code == 200 and rule["active"]

        hit, miss = _alert(8, service="private"), _alert(9, service="http")
        resp = client.post(
            "/ingest",
            json={"alerts": [hit.model_dump(mode="json"), miss.model_dump(mode="json")]},
            headers=_token(client, "senior"),
        )
        body = resp.json()
        assert (body["ingested"], body["suppressed"]) == (2, 1)
        # Both reach the SIEM: the suppressed one as BENIGN_BY_POLICY with its rule id, not dropped.
        assert body["siem"] == {"connector": "local-mock", "accepted": 2, "rejected": 0}

        senior = _token(client, "senior")
        stored = {a["alert_id"]: a for a in client.get("/alerts", headers=senior).json()}
        assert stored[hit.alert_id]["verdict"] == "BENIGN_BY_POLICY"
        assert stored[hit.alert_id]["suppression_rule_id"] == rule["rule_id"]
        assert stored[miss.alert_id]["verdict"] == "KNOWN_ATTACK"
        assert client.get("/stats", headers=senior).json()["alerts_suppressed"] == 1

    def test_creation_is_audited(self, client: TestClient) -> None:
        self._rule(client)
        assert "suppression.create" in [e.action for e in state.audit.tail(5)]

    @pytest.mark.parametrize(
        "overrides",
        [
            {"days": 365},  # longer than the ceiling
            {"match": {"family": "probe"}},  # a whole family: switching the detector off
            {"match": {"service": "*"}},  # wildcard
            {"match": {"password": "x"}},  # not a matchable field
            {"reason": "ok"},  # no real reason given
        ],
    )
    def test_unsafe_rules_are_refused(self, client: TestClient, overrides: dict) -> None:
        code, _ = self._rule(client, **overrides)
        assert code == 422


class TestLabellingQueue:
    def test_most_uncertain_first_and_judged_excluded(self, client: TestClient) -> None:
        sure, unsure, judged = _alert(10, p=0.99), _alert(11, p=0.55), _alert(12, p=0.52)
        state.repo.save_alerts([sure, unsure, judged])
        analyst = _token(client, "analyst")
        client.post(f"/alerts/{judged.alert_id}/verdict", json={"verdict": "true_positive"}, headers=analyst)
        items = client.get("/feedback/queue", headers=analyst).json()["items"]
        ids = [i["alert_id"] for i in items]
        assert judged.alert_id not in ids
        assert ids.index(unsure.alert_id) < ids.index(sure.alert_id)


def test_expired_rule_stops_matching() -> None:
    from penumbra.alerts.suppression import SuppressionRule, first_match

    past = datetime.now(UTC) - timedelta(days=40)
    rule = SuppressionRule(
        match={"service": "private"},
        reason="scanner allowlisted in March",
        created_by="senior",
        created_at=past,
        expires_at=past + timedelta(days=30),
    )
    assert first_match([rule], _alert(13)) is None


def test_labelling_queue_reaches_the_review_lane_under_load(client: TestClient) -> None:
    # 2,100 confident alerts outrank every abstention and exceed the old 2,000-row fetch.
    confident = [_alert(1000 + i, p=0.99) for i in range(2100)]
    unsure = build_alert(
        p_attack=0.5,
        novelty_percentile=0.3,
        policy=ScoringPolicy(),
        conformal_ambiguous=True,
        network=NetworkContext(),
    )
    unsure.raw_features = {"segment": "dmz"}
    state.repo.save_alerts([*confident, unsure])
    items = client.get("/feedback/queue", headers=_token(client, "analyst")).json()["items"]
    assert items[0]["alert_id"] == unsure.alert_id


def test_model_registry_is_readable_and_scoped(client: TestClient, tmp_path, monkeypatch) -> None:
    from penumbra.api import app as app_mod

    class S:
        artifact_root = tmp_path

    monkeypatch.setattr(app_mod, "settings", lambda: S())
    analyst = _token(client, "analyst")
    empty = client.get("/models/nslkdd", headers=analyst).json()
    assert empty == {"dataset": "nslkdd", "champion": None, "history": [], "versions": []}
    assert client.get("/models/..%2F..%2Fetc", headers=analyst).status_code == 404
    assert client.get("/models/nslkdd").status_code == 401


def test_siem_status_says_the_mock_is_a_mock(client: TestClient) -> None:
    body = client.get("/siem/status", headers=_token(client, "senior")).json()
    assert body["connector"] == "local-mock" and "not sent" in body["note"]
    assert client.get("/siem/status", headers=_token(client, "analyst")).status_code == 403


def test_triage_note_is_served_and_scoped(client: TestClient) -> None:
    from penumbra.rag.copilot import Copilot, Retriever
    from penumbra.rag.corpus import Technique

    state._copilot = Copilot(
        Retriever(
            [
                Technique(
                    "T1498",
                    "Network Denial of Service",
                    "Floods.",
                    url="https://attack.mitre.org/techniques/T1498",
                )
            ]
        )
    )
    try:
        visible, hidden = _alert(900), _alert(901, segment="ot")
        state.repo.save_alerts([visible, hidden])
        analyst = _token(client, "analyst")
        body = client.get(f"/alerts/{visible.alert_id}/triage", headers=analyst).json()
        assert body["note"]["headline"] and "[generated by: deterministic-template]" in body["text"]
        assert client.get(f"/alerts/{hidden.alert_id}/triage", headers=analyst).status_code == 404
    finally:
        state._copilot = None


def test_triage_without_a_corpus_says_how_to_build_one(client: TestClient, monkeypatch) -> None:
    a = _alert(902)
    state.repo.save_alert(a)

    def missing():
        raise FileNotFoundError("no corpus")

    monkeypatch.setattr(state, "copilot", missing)
    resp = client.get(f"/alerts/{a.alert_id}/triage", headers=_token(client, "analyst"))
    assert resp.status_code == 503 and "penumbra copilot build" in resp.json()["detail"]


def test_pii_key_status_is_admin_only(client: TestClient) -> None:
    assert client.get("/governance/pii-keys", headers=_token(client, "senior")).status_code == 403
    body = client.get("/governance/pii-keys", headers=_token(client, "admin")).json()
    assert set(body) == {"current", "previous", "overlap_open"}


def test_siem_outage_does_not_fail_the_ingest(client: TestClient) -> None:
    class Broken:
        name = "broken"

        def send(self, records):
            raise ConnectionError("SIEM down")

    state.siem = Broken()
    a = _alert(950)
    resp = client.post(
        "/ingest", json={"alerts": [a.model_dump(mode="json")]}, headers=_token(client, "senior")
    )
    assert resp.status_code == 200
    assert resp.json()["siem"]["error"] == "ConnectionError"
    assert "siem.forward_failed" in [e.action for e in state.audit.tail(5)]


class TestVerdictRaces:
    """Review findings: a re-recorded label could slip past the approver, or out of the pool."""

    def test_changed_verdict_is_not_promoted_under_a_pinned_approval(self, client: TestClient) -> None:
        a = _alert(960)
        state.repo.save_alert(a)
        analyst, senior = _token(client, "analyst"), _token(client, "senior")
        client.post(f"/alerts/{a.alert_id}/verdict", json={"verdict": "true_positive"}, headers=analyst)
        seen = {p["target_id"]: p["verdict"] for p in client.get("/feedback/pending", headers=senior).json()}
        client.post(f"/alerts/{a.alert_id}/verdict", json={"verdict": "false_positive"}, headers=analyst)
        resp = client.post(
            "/feedback/promote", json={"incident_ids": [a.alert_id], "expected": seen}, headers=senior
        ).json()
        assert resp["promoted"] == 0
        assert state.repo.promoted_training_rows() == []

    def test_promoted_verdict_cannot_be_re_recorded(self, client: TestClient) -> None:
        a = _alert(961)
        state.repo.save_alert(a)
        analyst = _token(client, "analyst")
        client.post(f"/alerts/{a.alert_id}/verdict", json={"verdict": "true_positive"}, headers=analyst)
        client.post(
            "/feedback/promote", json={"incident_ids": [a.alert_id]}, headers=_token(client, "senior")
        )
        again = client.post(
            f"/alerts/{a.alert_id}/verdict", json={"verdict": "false_positive"}, headers=analyst
        )
        assert again.status_code == 409
        assert state.repo.promoted_training_rows()[0]["label"] == 1


def test_placeholder_values_cannot_scope_a_suppression(client: TestClient) -> None:
    for value in ("-", "None", "UNKNOWN", "any"):
        resp = client.post(
            "/suppressions",
            json={"match": {"service": value}, "reason": "nightly vuln scanner", "days": 7},
            headers=_token(client, "senior"),
        )
        assert resp.status_code == 422, value


def test_labelling_queue_has_no_duplicates(client: TestClient) -> None:
    unsure = build_alert(
        p_attack=0.5,
        novelty_percentile=0.3,
        policy=ScoringPolicy(),
        conformal_ambiguous=True,
        network=NetworkContext(),
    )
    unsure.raw_features = {"segment": "dmz"}
    state.repo.save_alert(unsure)
    ids = [
        i["alert_id"]
        for i in client.get("/feedback/queue", headers=_token(client, "analyst")).json()["items"]
    ]
    assert len(ids) == len(set(ids))


def test_incident_segment_is_derived_and_enforced(client: TestClient) -> None:
    # Review finding: incidents were stored with segment NULL, which passes every segment filter.
    from penumbra.alerts.models import Incident, Lane, Severity

    ot = [_alert(970 + i, segment="ot") for i in range(3)]
    state.repo.save_alerts(ot)
    state.repo.save_incident(
        Incident(
            incident_id="INC-OT",
            title="ot incident",
            severity=Severity.HIGH,
            lane=Lane.KNOWN_THREAT,
            priority=90,
            entity="pseudo:ot",
            alert_ids=[a.alert_id for a in ot],
            event_count=3,
        )
    )
    analyst = _token(client, "analyst")  # dmz + corp only
    assert client.get("/incidents/INC-OT", headers=analyst).status_code == 404
    resp = client.post("/incidents/INC-OT/verdict", json={"verdict": "false_positive"}, headers=analyst)
    assert resp.status_code == 404
    assert client.get("/incidents/INC-OT", headers=_token(client, "senior")).status_code == 200


@pytest.mark.parametrize("fmt", ["asim", "ocsf", "ecs"])
def test_alert_exports_in_each_schema(client: TestClient, fmt: str) -> None:
    a = _alert(990)
    state.repo.save_alert(a)
    body = client.get(f"/alerts/{a.alert_id}/export?format={fmt}", headers=_token(client, "analyst")).json()
    assert body["format"] == fmt and body["record"]


def test_export_rejects_unknown_formats_and_hidden_alerts(client: TestClient) -> None:
    a, hidden = _alert(991), _alert(992, segment="ot")
    state.repo.save_alerts([a, hidden])
    analyst = _token(client, "analyst")
    assert client.get(f"/alerts/{a.alert_id}/export?format=cef", headers=analyst).status_code == 422
    assert client.get(f"/alerts/{hidden.alert_id}/export", headers=analyst).status_code == 404


def test_suppression_can_be_revoked_and_stops_matching(client: TestClient) -> None:
    senior = _token(client, "senior")
    rule = client.post(
        "/suppressions",
        json={
            "match": {"service": "private", "family": "probe"},
            "reason": "nightly vuln scanner",
            "days": 30,
        },
        headers=senior,
    ).json()
    assert (
        client.post(f"/suppressions/{rule['rule_id']}/revoke", headers=_token(client, "analyst")).status_code
        == 403
    )
    revoked = client.post(f"/suppressions/{rule['rule_id']}/revoke", headers=senior).json()
    assert revoked["active"] is False
    hit = _alert(993, service="private")
    body = client.post("/ingest", json={"alerts": [hit.model_dump(mode="json")]}, headers=senior).json()
    assert body["suppressed"] == 0
    assert client.post(f"/suppressions/{rule['rule_id']}/revoke", headers=senior).status_code == 409
    assert "suppression.revoke" in [e.action for e in state.audit.tail(10)]


def test_placeholder_is_dropped_when_a_real_field_scopes_the_rule(client: TestClient) -> None:
    # Second review: the console's own pre-filled rule (real address + service "-") was refused.
    resp = client.post(
        "/suppressions",
        json={
            "match": {"src_ip": "pseudo:abc", "dst_port": "80", "service": "-", "protocol": "any"},
            "reason": "nightly vuln scanner",
            "days": 7,
        },
        headers=_token(client, "senior"),
    )
    assert resp.status_code == 200
    assert resp.json()["match"] == {"src_ip": "pseudo:abc", "dst_port": "80", "protocol": "any"}


def test_proposal_never_offers_a_placeholder() -> None:
    from penumbra.alerts.suppression import proposed_match

    assert "service" not in proposed_match(_alert(994, service="-"))


def test_live_stream_respects_segments(client: TestClient) -> None:
    # Second review: publish_alert pushed every alert to every socket regardless of segment.
    token = client.post("/auth/login", json={"username": "analyst", "password": "analyst"}).json()[
        "access_token"
    ]
    ot, dmz = _alert(995, segment="ot"), _alert(996, segment="dmz")
    with client.websocket_connect(f"/stream?token={token}") as ws:
        assert ws.receive_json()["type"] == "connected"
        senior = _token(client, "senior")
        client.post("/ingest", json={"alerts": [ot.model_dump(mode="json")]}, headers=senior)
        client.post("/ingest", json={"alerts": [dmz.model_dump(mode="json")]}, headers=senior)
        first = ws.receive_json()
        assert first["type"] == "alert" and first["alert"]["alert_id"] == dmz.alert_id


def test_incident_without_stored_alerts_is_unresolved_not_public(client: TestClient) -> None:
    from penumbra.alerts.models import Incident, Lane, Severity

    state.repo.save_incident(
        Incident(
            incident_id="INC-ORPHAN",
            title="orphan",
            severity=Severity.LOW,
            lane=Lane.KNOWN_THREAT,
            priority=10,
            entity="pseudo:x",
            alert_ids=["never-stored"],
            event_count=1,
        )
    )
    assert client.get("/incidents/INC-ORPHAN", headers=_token(client, "analyst")).status_code == 404
    assert client.get("/incidents/INC-ORPHAN", headers=_token(client, "senior")).status_code == 200


def test_pre_fix_incidents_are_backfilled_on_open(tmp_path) -> None:
    from penumbra.alerts.models import Incident, Lane, Severity

    db = tmp_path / "old.db"
    repo = SqliteRepository(db)
    a = _alert(997, segment="ot")
    repo.save_alert(a)
    inc = Incident(
        incident_id="INC-OLD",
        title="old",
        severity=Severity.LOW,
        lane=Lane.KNOWN_THREAT,
        priority=10,
        entity="pseudo:y",
        alert_ids=[a.alert_id],
        event_count=1,
    )
    repo.save_incident(inc)
    repo._conn.execute("UPDATE incidents SET segment = NULL")  # what every pre-fix row looked like
    repo._conn.commit()
    repo.close()
    reopened = SqliteRepository(db)
    assert reopened._conn.execute("SELECT segment FROM incidents").fetchone()["segment"] == "ot"


class TestIngestCapacity:
    """THREAT_MODEL T8: a flood is refused predictably and visibly, and nothing is dropped silently."""

    def test_over_capacity_is_refused_whole_with_retry_after(self, client: TestClient, monkeypatch) -> None:
        monkeypatch.setattr(app_module, "INGEST_CAPACITY", 2)
        state.last_shed = state.last_shed_audit = None
        batch = [_alert(1100 + i).model_dump(mode="json") for i in range(3)]
        resp = client.post("/ingest", json={"alerts": batch}, headers=_token(client, "senior"))
        assert resp.status_code == 503 and resp.headers["Retry-After"]
        from penumbra.api.security.rbac import Principal, Role

        stored = state.repo.list_alerts(Principal(username="t", role=Role.ADMIN))
        assert stored == []  # nothing from the refused batch was stored: the sensor still has it

    def test_shedding_turns_health_degraded_and_is_audited(self, client: TestClient, monkeypatch) -> None:
        monkeypatch.setattr(app_module, "INGEST_CAPACITY", 1)
        state.last_shed = state.last_shed_audit = None
        batch = [_alert(1200 + i).model_dump(mode="json") for i in range(2)]
        client.post("/ingest", json={"alerts": batch}, headers=_token(client, "senior"))
        assert client.get("/health").json()["ingest"]["status"] == "degraded"
        assert "ingest.shed" in [e.action for e in state.audit.tail(5)]

    def test_within_capacity_is_unaffected_and_inflight_returns_to_zero(self, client: TestClient) -> None:
        state.last_shed = None
        resp = client.post(
            "/ingest",
            json={"alerts": [_alert(1300).model_dump(mode="json")]},
            headers=_token(client, "senior"),
        )
        assert resp.status_code == 200
        assert state.ingest_inflight == 0
        assert client.get("/health").json()["ingest"]["status"] == "ok"

    def test_oversized_batch_is_rejected(self, client: TestClient, monkeypatch) -> None:
        big = [_alert(0).model_dump(mode="json")] * (app_module.INGEST_MAX_BATCH + 1)
        assert (
            client.post("/ingest", json={"alerts": big}, headers=_token(client, "senior")).status_code == 422
        )


def test_events_per_incident_uses_true_event_counts(client: TestClient) -> None:
    # The demo fixture ships a sample of each incident's alerts; the header read 20x against a
    # measured 463.6x because it divided stored alerts by incidents.
    from penumbra.alerts.models import Incident, Lane, Severity

    sample = _alert(1400)
    state.repo.save_alert(sample)
    state.repo.save_incident(
        Incident(
            incident_id="INC-BIG",
            title="big",
            severity=Severity.HIGH,
            lane=Lane.KNOWN_THREAT,
            priority=90,
            entity="pseudo:big",
            alert_ids=[sample.alert_id],
            event_count=1000,
        )
    )
    stats = client.get("/stats", headers=_token(client, "senior")).json()
    assert stats["compression_ratio"] == 1000 and stats["incident_events"] == 1000
