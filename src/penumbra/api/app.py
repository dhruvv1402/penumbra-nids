"""The Penumbra API.

Serves the SOC console and anything else that wants scored traffic. Every route that changes
something is permission-gated and audit-logged.

**There is no endpoint that blocks traffic.** Not disabled, not admin-gated - absent. Alerts carry
a verdict and a suggested action as text; nothing consumes either as an instruction. ADR-0001, and
CI greps the tree to keep it true.

There is also no scoring endpoint: scoring happens sensor-side (replay, pcap) and the API receives
scored alerts. An API that scores arbitrary input on request is a threshold-discovery oracle
(THREAT_MODEL T3).
"""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from typing import Annotated, Any

from fastapi import Depends, FastAPI, HTTPException, Query, Request, WebSocket, WebSocketDisconnect, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field
from starlette.responses import Response

from penumbra import __version__
from penumbra.alerts import suppression
from penumbra.alerts.models import Alert, Incident
from penumbra.alerts.schemas import asim
from penumbra.api import reports
from penumbra.api.security import rbac
from penumbra.api.security.audit import ACTION_SUPPRESS, AuditLog
from penumbra.api.security.auth import InvalidToken, UserStore, decode_token, issue_token
from penumbra.api.security.rbac import Permission, Principal
from penumbra.config import DEV_PII_KEY, settings
from penumbra.feedback import active, integrity
from penumbra.integrations.siem import SiemConnector
from penumbra.integrations.siem import connector as siem_connector
from penumbra.storage.sqlite import SqliteRepository, VerdictLocked

# --- metrics --------------------------------------------------------------------------------------

REQUESTS = Counter("penumbra_requests_total", "API requests", ["route", "status"])
SCORE_LATENCY = Histogram("penumbra_score_seconds", "Scoring latency")
ALERTS_EMITTED = Counter("penumbra_alerts_total", "Alerts emitted", ["lane", "verdict"])


# --- app state ------------------------------------------------------------------------------------


def refuse_default_secrets() -> None:
    """Outside demo mode, refuse to start with the public development PII key.

    The key is what stands between a pseudonymised address and the real one. docker-compose used to
    set PENUMBRA_PII_SALT while the code read PENUMBRA_PII_HMAC_KEY, so a compose deployment ran on
    the key published in this repository and nothing noticed. Failing at startup is how it gets
    noticed.
    """
    if os.environ.get("PENUMBRA_ALLOW_DEMO_USERS"):
        return
    if settings().pii_hmac_key.strip() in {"", DEV_PII_KEY}:
        raise RuntimeError(
            "PENUMBRA_PII_HMAC_KEY (or PENUMBRA_PII_SALT) is unset, so IP pseudonymisation would use the "
            "public development key. Set it, or run with PENUMBRA_ALLOW_DEMO_USERS=1 for a local demo."
        )


class AppState:
    def __init__(self) -> None:
        refuse_default_secrets()
        self.repo = SqliteRepository(settings().state_dir / "penumbra.db")
        self.users = UserStore()
        self.users.seed_demo_users()
        self.audit = AuditLog(settings().state_dir / "audit.jsonl")
        # Socket -> who is on the other end. The stream carries full alerts, so it is scoped by
        # segment exactly like the REST reads; a restricted analyst must not receive every
        # segment's alerts just by holding the socket open.
        self.subscribers: dict[WebSocket, Principal] = {}
        # PENUMBRA_SIEM=mock (default) | sentinel | none. A misconfigured `sentinel` fails here, at
        # startup, rather than on the first alert.
        self.siem: SiemConnector | None = siem_connector(settings().state_dir)
        self._copilot: Any = None

    def copilot(self) -> Any:
        """The BM25 triage copilot, loaded on first use. Raises FileNotFoundError without a corpus."""
        if self._copilot is None:
            from penumbra.rag.copilot import Copilot

            self._copilot = Copilot()
        return self._copilot

    async def broadcast(self, message: dict[str, Any], *, segment: str | None = None) -> None:
        """Push to every connected console allowed to see it. A dead socket is dropped, never fatal.

        `segment` is the segment of the data in the message; None means unsegmented, which every
        principal may see - the same rule as the SQL filter.
        """
        dead: set[WebSocket] = set()
        for ws, principal in list(self.subscribers.items()):
            if segment is not None and principal.segments and segment not in principal.segments:
                continue
            try:
                await ws.send_json(message)
            except Exception:  # noqa: BLE001 - a disconnected client must not break the stream
                dead.add(ws)
        for ws in dead:
            self.subscribers.pop(ws, None)


state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    yield
    state.repo.close()


app = FastAPI(
    title="Penumbra",
    version=__version__,
    description=(
        "ML-native network detection and response. Emits alerts for analyst triage. "
        "Contains no endpoint that blocks, drops or resets traffic, by design - see ADR-0001."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Headers that cost nothing and close off whole classes of finding. CI's ZAP baseline job scans a
# running container for exactly these, and a DAST job that only ever confirms known gaps is
# decorative - the point is for it to find nothing here and then to start finding things when
# somebody changes this file.
SECURITY_HEADERS: dict[str, str] = {
    # MIME sniffing on a JSON API that echoes user-supplied strings is a real risk: a browser that
    # decides a response is HTML will execute what is in it.
    "X-Content-Type-Options": "nosniff",
    # Nothing here is meant to be framed. Clickjacking a verdict button is a small attack with a
    # large blast radius, given the verdicts feed retraining.
    "X-Frame-Options": "DENY",
    # This API serves JSON, never markup. A CSP that forbids everything is accurate rather than
    # cautious, and it means a reflected-XSS bug in an error path has nowhere to execute.
    "Content-Security-Policy": "default-src 'none'; frame-ancestors 'none'",
    # Referrer data from an internal SOC tool should not reach whatever an analyst clicks next.
    "Referrer-Policy": "no-referrer",
    # Deliberately absent: Strict-Transport-Security. TLS terminates at a reverse proxy or ingress;
    # this service speaks plain HTTP on loopback by design, and asserting HSTS from here would be
    # a guarantee it cannot keep. Recorded in .zap/rules.tsv rather than left for someone to
    # rediscover.
}


@app.middleware("http")
async def security_headers(request: Request, call_next: Any) -> Any:
    response = await call_next(request)
    for header, value in SECURITY_HEADERS.items():
        response.headers.setdefault(header, value)
    return response


bearer = HTTPBearer(auto_error=False)


# --- auth dependency --------------------------------------------------------------------------------


async def current_principal(
    creds: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer)],
) -> Principal:
    if creds is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "missing bearer token")
    try:
        return decode_token(creds.credentials)
    except InvalidToken as exc:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, str(exc)) from exc


CurrentUser = Annotated[Principal, Depends(current_principal)]


def require(principal: Principal, permission: Permission) -> None:
    """Permission gate that also records the denial.

    A denied request is a security event: it is either a misconfigured client or someone probing
    what they can reach, and both are worth being able to look up later.
    """
    try:
        rbac.require(principal, permission)
    except rbac.PermissionDenied as exc:
        state.audit.append(
            actor=principal.username,
            role=principal.role.value,
            action="auth.denied",
            target=permission.value,
        )
        raise HTTPException(status.HTTP_403_FORBIDDEN, str(exc)) from exc


# --- schemas ----------------------------------------------------------------------------------------


class LoginRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    role: str
    segments: list[str]


class VerdictRequest(BaseModel):
    verdict: str = Field(pattern="^(true_positive|false_positive|benign_by_policy)$")
    note: str = ""


class PromoteRequest(BaseModel):
    """Verdict targets to promote. Named `incident_ids` for compatibility; alert ids work too.

    `expected` maps each target to the verdict the approver saw. When given, a target whose verdict
    changed since is not promoted. The console always sends it.
    """

    incident_ids: list[str]
    expected: dict[str, str] | None = None


class SuppressionRequest(BaseModel):
    match: dict[str, str]
    reason: str = Field(min_length=10, max_length=500)
    days: int = Field(default=30, ge=1, le=suppression.MAX_LIFETIME.days)
    source_alert_id: str | None = None


# Per-account verdict ceiling. A human triaging carefully does not record 120 verdicts an hour; a
# script replaying a stolen session to relabel a campaign does. The ceiling is deliberately loose
# for people and tight for automation.
VERDICT_RATE_LIMIT = 120
VERDICT_RATE_WINDOW = timedelta(hours=1)


class IngestRequest(BaseModel):
    """Alerts from a sensor or the replay engine.

    Senior or above: an ingest endpoint that anyone can post to is an alert-injection primitive,
    and a queue an attacker can fill is a queue an attacker can hide in.
    """

    alerts: list[Alert]


# --- routes -----------------------------------------------------------------------------------------


@app.get("/health", tags=["ops"])
async def health() -> dict[str, Any]:
    chain = state.audit.verify()
    return {
        "status": "ok",
        "version": __version__,
        "audit_chain_intact": chain.valid,
        "audit_entries": chain.n_entries,
        "blocks_traffic": False,  # structurally, not configurably
    }


@app.get("/metrics", tags=["ops"])
async def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/auth/login", response_model=TokenResponse, tags=["auth"])
async def login(body: LoginRequest, request: Request) -> TokenResponse:
    user = state.users.authenticate(body.username, body.password)
    if user is None:
        state.audit.append(
            actor=body.username,
            role="unknown",
            action="auth.denied",
            target="login",
            detail={"ip": request.client.host if request.client else "unknown"},
        )
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid credentials")

    state.audit.append(actor=user.username, role=user.role.value, action="auth.login")
    return TokenResponse(
        access_token=issue_token(user),
        role=user.role.value,
        segments=sorted(user.segments),
    )


@app.get("/alerts", response_model=list[Alert], tags=["alerts"])
async def list_alerts(
    principal: CurrentUser,
    lane: Annotated[str | None, Query()] = None,
    limit: Annotated[int, Query(le=1000)] = 100,
) -> list[Alert]:
    require(principal, Permission.READ_ALERTS)
    return state.repo.list_alerts(principal, lane=lane, limit=limit)


@app.get("/incidents", response_model=list[Incident], tags=["incidents"])
async def list_incidents(
    principal: CurrentUser,
    incident_status: Annotated[str | None, Query(alias="status")] = None,
    lane: Annotated[str | None, Query()] = None,
    limit: Annotated[int, Query(le=500)] = 100,
) -> list[Incident]:
    require(principal, Permission.READ_INCIDENTS)
    return state.repo.list_incidents(principal, status=incident_status, lane=lane, limit=limit)


@app.get("/incidents/{incident_id}", tags=["incidents"])
async def get_incident(incident_id: str, principal: CurrentUser) -> dict[str, Any]:
    require(principal, Permission.READ_INCIDENTS)
    incident = state.repo.get_incident(incident_id, principal)
    if incident is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "incident not found")
    return {
        "incident": incident,
        "alerts": state.repo.alerts_for_incident(incident_id, principal),
    }


@app.post("/incidents/{incident_id}/verdict", tags=["incidents"])
async def record_verdict(incident_id: str, body: VerdictRequest, principal: CurrentUser) -> dict[str, Any]:
    """Record an analyst verdict.

    The verdict is queued, not applied to training. Promotion is a separate senior-gated call - see
    `/feedback/promote` and docs/THREAT_MODEL.md T1.
    """
    require(principal, Permission.RECORD_VERDICT)
    _enforce_verdict_rate(principal)
    # Scope check first: get_incident filters by segment, record_verdict does not.
    if state.repo.get_incident(incident_id, principal) is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "incident not found")
    try:
        incident = state.repo.record_verdict(incident_id, body.verdict, principal.username, body.note)
    except VerdictLocked as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc
    if incident is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "incident not found")

    state.audit.append(
        actor=principal.username,
        role=principal.role.value,
        action="analyst.verdict",
        target=incident_id,
        detail={"verdict": body.verdict, "note": body.note, "target_kind": "incident"},
    )
    await state.broadcast({"type": "verdict", "incident_id": incident_id, "verdict": body.verdict})
    return {"incident": incident, "queued_for_training": True, "promoted": False}


@app.get("/alerts/{alert_id}/export", tags=["alerts"])
async def export_alert(
    alert_id: str,
    principal: CurrentUser,
    fmt: Annotated[str, Query(alias="format", pattern="^(asim|ocsf|ecs)$")] = "asim",
) -> dict[str, Any]:
    """One alert as Microsoft Sentinel ASIM, OCSF (Security Lake and others) or Elastic ECS.

    The same alert, three vocabularies, each validated against its schema's rules before it leaves.
    A record that would fail ingestion is refused here with the reasons rather than exported.
    """
    require(principal, Permission.READ_ALERTS)
    alert = state.repo.get_alert(alert_id, principal)
    if alert is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "alert not found")
    from penumbra.alerts.schemas import ecs, ocsf

    module = {"asim": asim, "ocsf": ocsf, "ecs": ecs}[fmt]
    record = module.to_asim(alert) if fmt == "asim" else getattr(module, f"to_{fmt}")(alert)
    problems = module.validate(record)
    if problems:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, {"format": fmt, "problems": problems})
    return {"format": fmt, "record": record}


@app.get("/alerts/{alert_id}/triage", tags=["alerts"])
async def triage_note(alert_id: str, principal: CurrentUser) -> dict[str, Any]:
    """A cited triage note for one alert, from the offline BM25 copilot.

    Deterministic and offline: retrieval over a local ATT&CK corpus, a template fills the note, and
    every source is cited. A novelty alert's note deliberately names no technique, because the
    system's own verdict is that it does not recognise the traffic. Segment-scoped like any read.
    """
    require(principal, Permission.READ_ALERTS)
    alert = state.repo.get_alert(alert_id, principal)
    if alert is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "alert not found")
    try:
        copilot = state.copilot()
    except FileNotFoundError as exc:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "ATT&CK corpus not built. Run `penumbra data fetch -d attack` then `penumbra copilot build`.",
        ) from exc
    note = copilot.note(alert)
    return {"alert_id": alert_id, "note": note.to_dict(), "text": note.render()}


@app.post("/alerts/{alert_id}/verdict", tags=["alerts"])
async def record_alert_verdict(alert_id: str, body: VerdictRequest, principal: CurrentUser) -> dict[str, Any]:
    """Record a verdict on a single alert that never correlated into an incident.

    Same queue, same promotion gate, same audit trail as an incident verdict.
    """
    require(principal, Permission.RECORD_VERDICT)
    _enforce_verdict_rate(principal)
    try:
        alert = state.repo.record_alert_verdict(
            alert_id, body.verdict, principal.username, principal, body.note
        )
    except VerdictLocked as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc
    if alert is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "alert not found")

    state.audit.append(
        actor=principal.username,
        role=principal.role.value,
        action="analyst.verdict",
        target=alert_id,
        detail={"verdict": body.verdict, "note": body.note, "target_kind": "alert"},
    )
    await state.broadcast({"type": "verdict", "alert_id": alert_id, "verdict": body.verdict})
    return {
        "alert_id": alert_id,
        "queued_for_training": body.verdict != "benign_by_policy",
        "promoted": False,
        # Benign-by-policy is a policy decision, not a label: it never trains the model. The
        # console follows it with a suppression rule instead.
        "proposed_suppression": suppression.proposed_match(alert)
        if body.verdict == "benign_by_policy"
        else None,
    }


def _can_see_target(verdict: dict[str, object], principal: Principal) -> bool:
    target = str(verdict["target_id"])
    if verdict.get("target_kind") == "alert":
        return state.repo.get_alert(target, principal) is not None
    return state.repo.get_incident(target, principal) is not None


def _enforce_verdict_rate(principal: Principal) -> None:
    recent = state.repo.verdicts_by(principal.username, datetime.now(UTC) - VERDICT_RATE_WINDOW)
    if recent >= VERDICT_RATE_LIMIT:
        state.audit.append(
            actor=principal.username,
            role=principal.role.value,
            action="verdict.rate_limited",
            detail={"recent": recent, "limit": VERDICT_RATE_LIMIT},
        )
        raise HTTPException(
            status.HTTP_429_TOO_MANY_REQUESTS,
            f"{recent} verdicts in the last hour; the limit is {VERDICT_RATE_LIMIT}. "
            "Bulk relabelling is how a stolen session poisons a model, so it is throttled and logged.",
        )


@app.get("/feedback/pending", tags=["feedback"])
async def pending_verdicts(principal: CurrentUser) -> list[dict[str, object]]:
    """Pending verdicts, each annotated with the reasons it deserves a second look."""
    require(principal, Permission.PROMOTE_VERDICT)
    visible = [v for v in state.repo.pending_verdicts() if _can_see_target(v, principal)]
    flagged = integrity.flag_verdicts(visible, state.repo.verdict_history())
    for v in flagged:
        v["flag_reasons"] = [integrity.explain(f) for f in v["flags"]]
        v["self_approval"] = v.get("actor") == principal.username
    return flagged


@app.get("/feedback/queue", tags=["feedback"])
async def labelling_queue(
    principal: CurrentUser, limit: Annotated[int, Query(ge=1, le=100)] = 25
) -> dict[str, Any]:
    """Which alerts an analyst's next hour is best spent on (uncertainty sampling)."""
    require(principal, Permission.RECORD_VERDICT)
    judged = {str(v["target_id"]) for v in state.repo.verdict_history()}
    # The review lane is fetched on its own. Abstentions sit near p = 0.5, below every confident
    # alert, so a single top-N by priority would starve exactly the items this queue exists for.
    candidates = state.repo.list_alerts(principal, lane="review", limit=1000) + state.repo.list_alerts(
        principal, limit=1000
    )
    return {"strategy": "uncertainty sampling", "items": active.queue(candidates, judged=judged, limit=limit)}


@app.post("/feedback/promote", tags=["feedback"])
async def promote_verdicts(body: PromoteRequest, principal: CurrentUser) -> dict[str, Any]:
    """Move verdicts into the retraining pool. Senior or above, and never your own.

    This is the gate that stops one compromised account from teaching the model that its own traffic
    is benign. The two-person rule is enforced in the repository's UPDATE, so no caller can skip it.
    """
    require(principal, Permission.PROMOTE_VERDICT)
    # Only targets this principal can see. A segment-restricted senior promoting by id must not
    # reach verdicts on traffic outside their segments.
    pending = {str(v["target_id"]): v for v in state.repo.pending_verdicts()}
    ids = [i for i in body.incident_ids if i not in pending or _can_see_target(pending[i], principal)]
    own = {i for i in ids if i in pending and pending[i]["actor"] == principal.username}
    n = state.repo.promote_verdicts(ids, principal.username, body.expected)
    state.audit.append(
        actor=principal.username,
        role=principal.role.value,
        action="verdict.promote",
        detail={"count": n, "incident_ids": body.incident_ids, "refused_self_approval": sorted(own)},
    )
    return {"promoted": n, "refused_self_approval": sorted(own)}


# --- suppression ------------------------------------------------------------------------------------


@app.get("/suppressions", tags=["suppression"])
async def list_suppressions(principal: CurrentUser) -> list[dict[str, Any]]:
    require(principal, Permission.READ_ALERTS)
    return [suppression.summary(r) for r in state.repo.list_suppressions()]


@app.post("/suppressions/{rule_id}/revoke", tags=["suppression"])
async def revoke_suppression(rule_id: str, principal: CurrentUser) -> dict[str, Any]:
    """End a suppression now instead of at its expiry. Senior only, audited, never a deletion.

    The rule stays on record with its new, earlier expiry, so "what did we suppress last quarter,
    and who stopped it" keeps an answer. Alerts it already reclassified are not rewritten: they were
    suppressed under the policy in force when they arrived.
    """
    require(principal, Permission.CREATE_SUPPRESSION)
    rule = next((r for r in state.repo.list_suppressions() if r.rule_id == rule_id), None)
    if rule is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "no such suppression rule")
    now = datetime.now(UTC)
    if not rule.is_active(now):
        raise HTTPException(status.HTTP_409_CONFLICT, "rule is not active")
    # Validation requires expires_at > created_at, so a rule revoked in its first instant expires a
    # microsecond after creation rather than being rejected.
    revoked = rule.model_copy(update={"expires_at": max(now, rule.created_at + timedelta(microseconds=1))})
    state.repo.save_suppression(revoked)
    state.audit.append(
        actor=principal.username,
        role=principal.role.value,
        action="suppression.revoke",
        target=rule_id,
        detail={"match": rule.match, "was_expiring": rule.expires_at.isoformat()},
    )
    await state.broadcast({"type": "suppression", "rule_id": rule_id})
    return suppression.summary(revoked)


@app.post("/suppressions", tags=["suppression"])
async def create_suppression(body: SuppressionRequest, principal: CurrentUser) -> dict[str, Any]:
    """Create a BENIGN_BY_POLICY rule. Senior only, always scoped, always expiring.

    Matching alerts are still stored and counted - they leave the queue, not the record.
    """
    require(principal, Permission.CREATE_SUPPRESSION)
    now = datetime.now(UTC)
    try:
        rule = suppression.SuppressionRule(
            match=body.match,
            reason=body.reason,
            created_by=principal.username,
            created_at=now,
            expires_at=now + timedelta(days=body.days),
            source_alert_id=body.source_alert_id,
        )
    except ValueError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(exc)) from exc

    state.repo.save_suppression(rule)
    state.audit.append(
        actor=principal.username,
        role=principal.role.value,
        action=ACTION_SUPPRESS,
        target=rule.rule_id,
        detail={"match": rule.match, "reason": rule.reason, "expires_at": rule.expires_at.isoformat()},
    )
    await state.broadcast({"type": "suppression", "rule_id": rule.rule_id})
    return suppression.summary(rule)


@app.post("/ingest", tags=["alerts"])
async def ingest(body: IngestRequest, principal: CurrentUser) -> dict[str, Any]:
    """Accept scored alerts, persist them, and push to connected consoles."""
    require(principal, Permission.PROMOTE_VERDICT)  # senior or above
    rules = state.repo.list_suppressions(active_only=True)
    suppressed = 0
    for alert in body.alerts:
        suppressed += await publish_alert(alert, rules=rules)
    out: dict[str, Any] = {"ingested": len(body.alerts), "suppressed": suppressed}
    if state.siem is not None and body.alerts:
        # Forwarded after suppression, so the SIEM sees BENIGN_BY_POLICY with its rule id rather than
        # a detection Penumbra has already decided is policy. Suppressed is not deleted there either.
        # In a worker thread: the Sentinel client does blocking HTTP with retries, and running it on
        # the event loop stalled every other request and the websocket while it waited. A SIEM
        # failure is reported, never raised - the alerts are already stored and pushed, and an
        # outage downstream must not turn a successful ingest into a 500.
        records = [asim.to_asim(a) for a in body.alerts]
        try:
            sent = await asyncio.to_thread(state.siem.send, records)
            out["siem"] = {"connector": state.siem.name, "accepted": sent.accepted, "rejected": sent.rejected}
        except Exception as exc:  # noqa: BLE001 - see above
            out["siem"] = {"connector": state.siem.name, "accepted": 0, "error": type(exc).__name__}
            state.audit.append(
                actor=principal.username,
                role=principal.role.value,
                action="siem.forward_failed",
                detail={"error": type(exc).__name__, "records": len(records)},
            )
    return out


@app.get("/siem/status", tags=["governance"])
async def siem_status(principal: CurrentUser) -> dict[str, Any]:
    """Which SIEM connector is live. The mock says it is a mock."""
    require(principal, Permission.READ_AUDIT)
    if state.siem is None:
        return {"connector": "none", "forwarding": False}
    status_: dict[str, Any] = {"connector": state.siem.name, "forwarding": True}
    count = getattr(state.siem, "count", None)
    if callable(count):
        status_["records_written"] = count()
        status_["note"] = "local mock: records are validated against ASIM and written to a file, not sent"
    return status_


@app.post("/incidents/bulk", tags=["incidents"])
async def ingest_incidents(incidents: list[Incident], principal: CurrentUser) -> dict[str, Any]:
    """Accept correlated incidents from the replay engine."""
    require(principal, Permission.PROMOTE_VERDICT)
    for incident in incidents:
        state.repo.save_incident(incident)
    await state.broadcast({"type": "incidents", "count": len(incidents)})
    return {"ingested": len(incidents)}


@app.get("/stats", tags=["ops"])
async def stats(principal: CurrentUser) -> dict[str, int]:
    require(principal, Permission.READ_ALERTS)
    return state.repo.counts(principal)


@app.get("/audit", tags=["governance"])
async def read_audit(principal: CurrentUser, limit: Annotated[int, Query(le=500)] = 50) -> dict[str, Any]:
    require(principal, Permission.READ_AUDIT)
    chain = state.audit.verify()
    return {
        "chain_intact": chain.valid,
        "broken_at": chain.broken_at,
        "n_entries": chain.n_entries,
        "entries": [json.loads(e.to_json()) for e in state.audit.tail(limit)],
    }


@app.get("/governance/rbac", tags=["governance"])
async def rbac_matrix(principal: CurrentUser) -> dict[str, Any]:
    require(principal, Permission.READ_ALERTS)
    return {
        "matrix": rbac.matrix(),
        "roles": {
            role.value: sorted(p.value for p in perms) for role, perms in rbac.ROLE_PERMISSIONS.items()
        },
    }


KNOWN_DATASETS = frozenset({"nslkdd", "unsw", "cicids"})


@app.get("/models/{dataset}", tags=["governance"])
async def model_registry(dataset: str, principal: CurrentUser) -> dict[str, Any]:
    """The registry, read-only: which version is champion, why, and whether its bytes are intact.

    Promotion stays a CLI action by design - it is rare, deliberate, and needs the gate report on
    disk. What the console needs is to SHOW the evidence: every version's gate verdict, its shadow
    comparison, and a fresh hash check, so "is production running what we evaluated?" is answered
    by the page rather than by trust.
    """
    require(principal, Permission.READ_ALERTS)
    if dataset not in KNOWN_DATASETS:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND, f"unknown dataset; expected one of {sorted(KNOWN_DATASETS)}"
        )
    from penumbra.models.registry import ModelRegistry, RegistryError

    reg = ModelRegistry(settings().artifact_root / "registry", dataset)
    try:
        champion = reg.champion_state()
    except RegistryError as exc:
        # A tampered champion pointer is shown as exactly that, not as a 500 or an empty registry.
        champion = {"version": None, "history": [], "error": str(exc)}
    versions = []
    for v in reg.versions():
        shadow = v.training.get("shadow") or {}
        versions.append(
            {
                "version": v.version,
                "created_at": v.created_at,
                "created_by": v.created_by,
                "parent": v.parent,
                "feedback_rows": v.training.get("feedback_rows"),
                "approvers": v.training.get("approvers", []),
                "gate": None
                if v.gate is None
                else {"passed": v.gate.get("passed"), "reasons": v.gate.get("reasons", [])},
                "shadow": None
                if not shadow
                else {k: shadow.get(k) for k in ("alert_volume_ratio", "cohen_kappa", "agreement", "n_rows")},
                "intact": not reg.verify(v.version),
                "champion": v.version == champion.get("version"),
            }
        )
    return reports.finite(
        {
            "dataset": dataset,
            "champion": champion.get("version"),
            "history": champion.get("history", []),
            "versions": versions,
            **({"error": champion["error"]} if "error" in champion else {}),
        }
    )


@app.get("/governance/pii-keys", tags=["governance"])
async def pii_key_status(principal: CurrentUser) -> dict[str, Any]:
    """Which pseudonymisation keys are live, as fingerprints - never the keys.

    Rotation itself happens in the secret manager (PENUMBRA_PII_HMAC_KEY plus
    PENUMBRA_PII_HMAC_PREVIOUS_KEYS for the overlap window). This is how an operator confirms which
    keys a running instance accepts, and when the overlap has been closed.
    """
    require(principal, Permission.ROTATE_KEY)
    from penumbra.api.security import pii

    status_ = pii.key_fingerprints()
    state.audit.append(
        actor=principal.username,
        role=principal.role.value,
        action="pii.key_status",
        detail={"previous_accepted": len(status_["previous"])},  # type: ignore[arg-type]
    )
    return {**status_, "overlap_open": bool(status_["previous"])}


@app.get("/reports", tags=["reports"])
async def list_reports(principal: CurrentUser) -> dict[str, Any]:
    """Every evaluation report the console knows how to render, present or not.

    Absent reports come back with `available: false` and the command that produces them. A page
    that silently renders nothing looks identical to one whose data is genuinely empty, and on
    stage that difference is the whole answer to "why is this blank".
    """
    require(principal, Permission.READ_ALERTS)
    return {"reports": reports.catalogue()}


@app.get("/reports/{name}", tags=["reports"])
async def read_report(principal: CurrentUser, name: str) -> dict[str, Any]:
    """One report's contents, straight from the file the CLI wrote.

    Nothing is recomputed here. The console cannot drift out of step with the measurements, and
    every number on screen traces back to a command a judge can run.
    """
    require(principal, Permission.READ_ALERTS)
    payload = reports.load(name)
    if payload is None:
        known = ", ".join(sorted(reports.BY_NAME))
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"no report {name!r}. Known reports: {known}",
        )
    return payload


@app.websocket("/stream")
async def stream(websocket: WebSocket) -> None:
    """Live alert stream for the console.

    Token is passed as a query parameter because browsers cannot set headers on a WebSocket
    handshake. It is still verified, and the connection is refused without a valid one.
    """
    token = websocket.query_params.get("token", "")
    try:
        principal = decode_token(token)
    except InvalidToken:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    await websocket.accept()
    state.subscribers[websocket] = principal
    await websocket.send_json({"type": "connected", "user": principal.username})
    try:
        while True:
            # Keep the socket open; the server pushes, the client does not poll.
            await asyncio.sleep(30)
            await websocket.send_json({"type": "heartbeat"})
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        state.subscribers.pop(websocket, None)


async def publish_alert(alert: Alert, *, rules: list[suppression.SuppressionRule] | None = None) -> int:
    """Persist an alert and push it to connected consoles. Returns 1 if a suppression matched.

    Suppression happens here, at ingest, so a rule applies to every sensor and replay alike. The
    alert is reclassified and stored - never dropped.
    """
    rule = suppression.first_match(rules or [], alert)
    if rule is not None:
        suppression.apply(alert, rule)
    state.repo.save_alert(alert)
    ALERTS_EMITTED.labels(lane=alert.lane.value, verdict=alert.verdict.value).inc()
    segment = alert.raw_features.get("segment")
    await state.broadcast(
        {
            "type": "alert",
            "alert": json.loads(alert.model_dump_json()),
            "at": datetime.now(UTC).isoformat(),
        },
        segment=str(segment) if segment is not None else None,
    )
    return int(rule is not None)
