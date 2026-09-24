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
def client() -> TestClient:
    state.repo = SqliteRepository(":memory:")
    state.audit = AuditLog()
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
        assert resp.json() == {"ingested": 2, "suppressed": 1}

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
