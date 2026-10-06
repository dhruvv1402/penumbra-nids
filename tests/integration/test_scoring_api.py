"""Live scoring (ADR-0006): what a guest may do, and every guard that replaced "no scoring endpoint".

Run against a tiny detector so CI needs no dataset. The success path on the real champions,
including a real pcap, is exercised by hand before each deploy; what is pinned here are the
controls: the permission, the rate limit, the row cap, rounding, the upload limits, the audit
record, and the promise that a scored upload is stored nowhere.
"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("PENUMBRA_ALLOW_DEMO_USERS", "1")

from fastapi.testclient import TestClient  # noqa: E402

from penumbra.api import app as app_module  # noqa: E402
from penumbra.api import scoring  # noqa: E402
from penumbra.api.app import app, state  # noqa: E402
from penumbra.api.security.audit import AuditLog  # noqa: E402
from penumbra.integrations.siem.mock import LocalMockSiem  # noqa: E402
from penumbra.models.detector import PenumbraDetector  # noqa: E402
from penumbra.storage.sqlite import SqliteRepository  # noqa: E402
from tests.unit.test_registry import tiny_dataset  # noqa: E402


@pytest.fixture(scope="module")
def bundle(tmp_path_factory):
    """A deployment bundle in miniature: one model and its held-out sample pool."""
    root = tmp_path_factory.mktemp("bundle")
    ds = tiny_dataset(n=1200)
    PenumbraDetector(target_fpr=0.05).fit(ds).save(root / "models" / "unsw")
    (root / "samples").mkdir()
    pool = scoring.build_pool(ds.X_test, ds.y_test, ds.fam_test, n_benign=200, n_attack=200)
    pool.to_csv(root / "samples" / "unsw_test_sample.csv.gz", index=False, compression="gzip")
    return root


@pytest.fixture()
def client(bundle, tmp_path, monkeypatch) -> TestClient:
    monkeypatch.setenv("PENUMBRA_GUEST_ACCESS", "1")
    monkeypatch.setattr(app_module, "SCORER", scoring.ModelStore(root=bundle))
    monkeypatch.setattr(app_module, "SCORE_LIMITER", scoring.RateLimiter())
    state.repo = SqliteRepository(":memory:")
    state.audit = AuditLog()
    state.siem = LocalMockSiem(tmp_path / "siem")
    return TestClient(app)


def guest(client: TestClient) -> dict[str, str]:
    return {"Authorization": f"Bearer {client.post('/auth/guest').json()['access_token']}"}


def user(client: TestClient, name: str) -> dict[str, str]:
    tok = client.post("/auth/login", json={"username": name, "password": name}).json()["access_token"]
    return {"Authorization": f"Bearer {tok}"}


def test_a_guest_scores_held_out_flows_and_sees_the_truth(client) -> None:
    body = client.post(
        "/score/sample",
        headers=guest(client),
        json={"dataset": "unsw", "n": 100, "attack_share": 0.5, "seed": 1},
    ).json()
    s = body["summary"]
    assert s["flows"] == 100
    assert s["truth"]["attacks"] == 50
    assert {r["verdict"] for r in body["rows"]} <= {"KNOWN_ATTACK", "SUSPECTED_NOVEL", "UNCERTAIN", "BENIGN"}
    assert all("truth" in r for r in body["rows"])


def test_scores_are_rounded_so_a_threshold_cannot_be_bisected(client) -> None:
    rows = client.post(
        "/score/sample", headers=guest(client), json={"dataset": "unsw", "n": 50, "seed": 2}
    ).json()["rows"]
    for r in rows:
        assert r["p_attack"] == round(r["p_attack"], scoring.ROUND)
        assert r["novelty_percentile"] == round(r["novelty_percentile"], scoring.ROUND)


def test_a_guest_is_capped_in_rows(client) -> None:
    body = client.post(
        "/score/sample", headers=guest(client), json={"dataset": "unsw", "n": 5000, "seed": 3}
    ).json()
    assert body["summary"]["flows"] <= scoring.GUEST_LIMITS.max_rows


def test_the_rate_limit_applies_and_is_audited(client, monkeypatch) -> None:
    monkeypatch.setattr(scoring, "GUEST_LIMITS", scoring.Limits(calls_per_hour=2, max_rows=50))
    h = guest(client)
    for _ in range(2):
        assert client.post("/score/sample", headers=h, json={"n": 10}).status_code == 200
    refused = client.post("/score/sample", headers=h, json={"n": 10})
    assert refused.status_code == 429
    assert "oracle" in refused.json()["detail"]
    assert len(state.audit.by_action("model.score")) == 2
    assert len(state.audit.by_action("model.score.rate_limited")) == 1


def test_guests_on_different_addresses_have_separate_budgets(client, monkeypatch) -> None:
    monkeypatch.setattr(scoring, "GUEST_LIMITS", scoring.Limits(calls_per_hour=1, max_rows=50))
    h = guest(client)
    assert (
        client.post(
            "/score/sample", headers={**h, "x-forwarded-for": "198.51.100.1"}, json={"n": 5}
        ).status_code
        == 200
    )
    assert (
        client.post(
            "/score/sample", headers={**h, "x-forwarded-for": "198.51.100.1"}, json={"n": 5}
        ).status_code
        == 429
    )
    assert (
        client.post(
            "/score/sample", headers={**h, "x-forwarded-for": "198.51.100.2"}, json={"n": 5}
        ).status_code
        == 200
    )


def test_a_scored_upload_is_stored_nowhere(client) -> None:
    admin = user(client, "admin")
    before = client.get("/stats", headers=admin).json()
    assert (
        client.post(
            "/score/sample", headers=user(client, "senior"), json={"n": 100, "attack_share": 0.5}
        ).status_code
        == 200
    )
    assert client.get("/stats", headers=admin).json() == before


def test_template_round_trips_through_csv_scoring(client) -> None:
    h = guest(client)
    template = client.get("/score/template/unsw", headers=h)
    assert template.status_code == 200 and template.headers["content-type"].startswith("text/csv")
    scored = client.post(
        "/score/csv?dataset=unsw", headers={**h, "Content-Type": "text/csv"}, content=template.content
    )
    assert scored.status_code == 200
    assert scored.json()["summary"]["flows"] == len(template.text.strip().splitlines()) - 1


def test_a_csv_missing_columns_is_refused_with_the_names(client) -> None:
    r = client.post(
        "/score/csv?dataset=unsw",
        headers={**guest(client), "Content-Type": "text/csv"},
        content=b"x,y\n1,2\n",
    )
    assert r.status_code == 422
    assert "missing" in r.json()["detail"]


def test_an_oversized_upload_is_refused_before_it_is_parsed(client, monkeypatch) -> None:
    monkeypatch.setattr(scoring, "MAX_CSV_BYTES", 100)
    r = client.post(
        "/score/csv?dataset=unsw", headers={**guest(client), "Content-Type": "text/csv"}, content=b"a" * 500
    )
    assert r.status_code == 413


def test_junk_is_not_a_capture(client) -> None:
    pytest.importorskip("dpkt")
    r = client.post(
        "/score/pcap",
        headers={**guest(client), "Content-Type": "application/octet-stream"},
        content=b"not a pcap",
    )
    assert r.status_code == 422


def test_an_unknown_dataset_is_a_clear_error(client) -> None:
    r = client.post("/score/sample", headers=guest(client), json={"dataset": "kdd99"})
    assert r.status_code == 422


def test_scoring_needs_a_session(client) -> None:
    assert client.post("/score/sample", json={"n": 5}).status_code == 401


def test_models_endpoint_reports_schema_and_the_callers_limits(client) -> None:
    body = client.get("/score/models", headers=guest(client)).json()
    assert [m["dataset"] for m in body["models"]] == ["unsw"]
    assert body["limits"]["max_rows"] == scoring.GUEST_LIMITS.max_rows
    assert "proto" in body["models"][0]["categorical"]
    analyst = client.get("/score/models", headers=user(client, "analyst")).json()
    assert analyst["limits"]["max_rows"] == scoring.USER_LIMITS.max_rows
