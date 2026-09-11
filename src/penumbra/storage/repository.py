"""Storage interface.

A Protocol rather than a base class, so the API layer depends on the shape and never on a concrete
backend. Swapping SQLite for Postgres is one new file implementing this, not a refactor.

SQLite is the default deliberately. On demo day, every container that has to be running is another
thing that can be down, and a database that lives in a file cannot fail to accept connections.
Postgres arrives if and when the platform phase does.

**Row-level scoping is enforced here, not in the UI.** Every read takes a `Principal` and filters by
the segments that principal may see. An analyst calling the API directly gets exactly what the
console would have shown them, because it is the same query.
"""

from __future__ import annotations

from datetime import datetime
from typing import Protocol, runtime_checkable

from penumbra.alerts.models import Alert, Incident
from penumbra.api.security.rbac import Principal


@runtime_checkable
class Repository(Protocol):
    """Everything the API needs from storage."""

    # --- alerts
    def save_alert(self, alert: Alert) -> None: ...
    def save_alerts(self, alerts: list[Alert]) -> None: ...
    def get_alert(self, alert_id: str, principal: Principal) -> Alert | None: ...
    def list_alerts(
        self,
        principal: Principal,
        *,
        lane: str | None = None,
        limit: int = 100,
        since: datetime | None = None,
    ) -> list[Alert]: ...

    # --- incidents
    def save_incident(self, incident: Incident) -> None: ...
    def get_incident(self, incident_id: str, principal: Principal) -> Incident | None: ...
    def list_incidents(
        self,
        principal: Principal,
        *,
        status: str | None = None,
        lane: str | None = None,
        limit: int = 100,
    ) -> list[Incident]: ...
    def alerts_for_incident(self, incident_id: str, principal: Principal) -> list[Alert]: ...

    # --- analyst feedback
    def record_verdict(
        self, incident_id: str, verdict: str, actor: str, note: str = ""
    ) -> Incident | None: ...

    # --- retraining queue
    def pending_verdicts(self) -> list[dict[str, object]]: ...
    def promote_verdicts(self, incident_ids: list[str], approver: str) -> int: ...

    # --- counts for the console header
    def counts(self, principal: Principal) -> dict[str, int]: ...


class SuppressionRule(Protocol):
    """A rule that reclassifies matching traffic as BENIGN_BY_POLICY.

    Every rule carries an expiry. A permanent suppression is a permanent blind spot, and the
    scanner allowlisted in March is the C2 channel missed in September - so the type makes the
    expiry non-optional rather than leaving it to discipline.
    """

    rule_id: str
    entity: str
    family: str | None
    reason: str
    created_by: str
    created_at: datetime
    expires_at: datetime  # not optional, on purpose

    def matches(self, alert: Alert) -> bool: ...
    def is_active(self, now: datetime) -> bool: ...
