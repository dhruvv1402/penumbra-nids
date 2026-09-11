"""The Penumbra API.

Serves the SOC console and anything else that wants scored traffic. Every route that changes
something is permission-gated and audit-logged.

**There is no endpoint that blocks traffic.** Not disabled, not admin-gated - absent. `/score`
returns a verdict and a suggested action as text; nothing consumes it as an instruction. ADR-0001,
and CI greps the tree to keep it true.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Annotated, Any

from fastapi import Depends, FastAPI, HTTPException, Query, Request, WebSocket, WebSocketDisconnect, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field
from starlette.responses import Response

from penumbra import __version__
from penumbra.alerts.models import Alert, Incident
from penumbra.api import reports
from penumbra.api.security import rbac
from penumbra.api.security.audit import AuditLog
from penumbra.api.security.auth import InvalidToken, UserStore, decode_token, issue_token
from penumbra.api.security.rbac import Permission, Principal
from penumbra.config import settings
from penumbra.storage.sqlite import SqliteRepository

# --- metrics --------------------------------------------------------------------------------------

REQUESTS = Counter("penumbra_requests_total", "API requests", ["route", "status"])
SCORE_LATENCY = Histogram("penumbra_score_seconds", "Scoring latency")
ALERTS_EMITTED = Counter("penumbra_alerts_total", "Alerts emitted", ["lane", "verdict"])


# --- app state ------------------------------------------------------------------------------------


class AppState:
    def __init__(self) -> None:
        self.repo = SqliteRepository(settings().artifact_root / "penumbra.db")
        self.users = UserStore()
        self.users.seed_demo_users()
        self.audit = AuditLog(settings().artifact_root / "audit.jsonl")
        self.subscribers: set[WebSocket] = set()

    async def broadcast(self, message: dict[str, Any]) -> None:
        """Push to every connected console. A dead socket is dropped, never fatal."""
        dead: set[WebSocket] = set()
        for ws in self.subscribers:
            try:
                await ws.send_json(message)
            except Exception:  # noqa: BLE001 - a disconnected client must not break the stream
                dead.add(ws)
        self.subscribers -= dead


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
    incident_ids: list[str]


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
    incident = state.repo.record_verdict(incident_id, body.verdict, principal.username, body.note)
    if incident is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "incident not found")

    state.audit.append(
        actor=principal.username,
        role=principal.role.value,
        action="analyst.verdict",
        target=incident_id,
        detail={"verdict": body.verdict, "note": body.note},
    )
    await state.broadcast({"type": "verdict", "incident_id": incident_id, "verdict": body.verdict})
    return {"incident": incident, "queued_for_training": True, "promoted": False}


@app.get("/feedback/pending", tags=["feedback"])
async def pending_verdicts(principal: CurrentUser) -> list[dict[str, object]]:
    require(principal, Permission.PROMOTE_VERDICT)
    return state.repo.pending_verdicts()


@app.post("/feedback/promote", tags=["feedback"])
async def promote_verdicts(body: PromoteRequest, principal: CurrentUser) -> dict[str, Any]:
    """Move verdicts into the retraining pool. Senior or above.

    This is the gate that stops one compromised analyst account from teaching the model that its own
    traffic is benign.
    """
    require(principal, Permission.PROMOTE_VERDICT)
    n = state.repo.promote_verdicts(body.incident_ids, principal.username)
    state.audit.append(
        actor=principal.username,
        role=principal.role.value,
        action="verdict.promote",
        detail={"count": n, "incident_ids": body.incident_ids},
    )
    return {"promoted": n}


@app.post("/ingest", tags=["alerts"])
async def ingest(body: IngestRequest, principal: CurrentUser) -> dict[str, Any]:
    """Accept scored alerts, persist them, and push to connected consoles."""
    require(principal, Permission.PROMOTE_VERDICT)  # senior or above
    for alert in body.alerts:
        await publish_alert(alert)
    return {"ingested": len(body.alerts)}


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
    state.subscribers.add(websocket)
    await websocket.send_json({"type": "connected", "user": principal.username})
    try:
        while True:
            # Keep the socket open; the server pushes, the client does not poll.
            await asyncio.sleep(30)
            await websocket.send_json({"type": "heartbeat"})
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        state.subscribers.discard(websocket)


async def publish_alert(alert: Alert) -> None:
    """Persist an alert and push it to connected consoles."""
    state.repo.save_alert(alert)
    ALERTS_EMITTED.labels(lane=alert.lane.value, verdict=alert.verdict.value).inc()
    await state.broadcast(
        {
            "type": "alert",
            "alert": json.loads(alert.model_dump_json()),
            "at": datetime.now(UTC).isoformat(),
        }
    )
