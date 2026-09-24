"""SQLite implementation of `Repository`.

One file, no server, no container. Alerts and incidents are stored with their JSON payload alongside
a few indexed columns, which keeps the schema small while leaving the full `Alert` retrievable
exactly as it was scored.

Row-level scoping happens in the WHERE clause, not in the caller. An analyst restricted to the `dmz`
and `corp` segments gets a query that cannot return `ot` rows, whether they came through the console
or straight to the API.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from penumbra.alerts.models import Alert, Incident
from penumbra.alerts.suppression import SuppressionRule
from penumbra.api.security.rbac import Principal

SCHEMA = """
CREATE TABLE IF NOT EXISTS alerts (
    alert_id            TEXT PRIMARY KEY,
    timestamp           TEXT NOT NULL,
    verdict             TEXT NOT NULL,
    lane                TEXT NOT NULL,
    severity            TEXT NOT NULL,
    priority            INTEGER NOT NULL,
    p_attack            REAL NOT NULL,
    novelty_percentile  REAL NOT NULL,
    family              TEXT,
    segment             TEXT,
    incident_id         TEXT,
    payload             TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_alerts_lane     ON alerts(lane);
CREATE INDEX IF NOT EXISTS idx_alerts_priority ON alerts(priority DESC);
CREATE INDEX IF NOT EXISTS idx_alerts_incident ON alerts(incident_id);
CREATE INDEX IF NOT EXISTS idx_alerts_segment  ON alerts(segment);

CREATE TABLE IF NOT EXISTS incidents (
    incident_id     TEXT PRIMARY KEY,
    opened_at       TEXT NOT NULL,
    last_seen_at    TEXT NOT NULL,
    title           TEXT NOT NULL,
    severity        TEXT NOT NULL,
    lane            TEXT NOT NULL,
    priority        INTEGER NOT NULL,
    entity          TEXT NOT NULL,
    family          TEXT,
    segment         TEXT,
    status          TEXT NOT NULL,
    analyst_verdict TEXT,
    closed_by       TEXT,
    event_count     INTEGER NOT NULL DEFAULT 0,
    payload         TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_incidents_status   ON incidents(status);
CREATE INDEX IF NOT EXISTS idx_incidents_priority ON incidents(priority DESC);
CREATE INDEX IF NOT EXISTS idx_incidents_segment  ON incidents(segment);

-- Verdicts land here first and require senior promotion before they can train anything.
-- See docs/THREAT_MODEL.md T1.
CREATE TABLE IF NOT EXISTS verdict_queue (
    incident_id  TEXT PRIMARY KEY,
    verdict      TEXT NOT NULL,
    actor        TEXT NOT NULL,
    note         TEXT,
    recorded_at  TEXT NOT NULL,
    promoted     INTEGER NOT NULL DEFAULT 0,
    promoted_by  TEXT,
    promoted_at  TEXT
);

-- The full rule is the payload; expires_at is a column so expiry can be queried without parsing.
CREATE TABLE IF NOT EXISTS suppression_rules (
    rule_id     TEXT PRIMARY KEY,
    created_by  TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    expires_at  TEXT NOT NULL,
    payload     TEXT NOT NULL
);
"""

# Columns added to verdict_queue after databases already existed in the field. CREATE TABLE IF NOT
# EXISTS does not add columns to an existing table, so they are added here, idempotently.
VERDICT_QUEUE_MIGRATIONS = {
    "target_kind": "TEXT NOT NULL DEFAULT 'incident'",  # 'incident' | 'alert'
    "p_attack": "REAL",  # the model's score on what was judged, for the integrity checks
    "family": "TEXT",
}

# How a promoted verdict becomes a training label. BENIGN_BY_POLICY is deliberately absent: the
# internal scanner IS a port scan, and teaching the model otherwise would blind it to the next real
# one. Policy lives in suppression rules, where it expires; never in the weights, where it does not.
TRAINING_LABEL = {"true_positive": 1, "false_positive": 0}


class SqliteRepository:
    """File-backed repository. Satisfies `Repository`."""

    def __init__(self, path: Path | str = ":memory:") -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False because uvicorn serves from a threadpool. Writes are serialised by
        # SQLite itself; the connection is not shared across processes.
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA)
        self._migrate()
        self._conn.commit()

    def _migrate(self) -> None:
        present = {r["name"] for r in self._conn.execute("PRAGMA table_info(verdict_queue)")}
        for column, ddl in VERDICT_QUEUE_MIGRATIONS.items():
            if column not in present:
                # Column names and DDL come from the constant above, never from input.
                self._conn.execute(f"ALTER TABLE verdict_queue ADD COLUMN {column} {ddl}")

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Connection]:
        try:
            yield self._conn
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

    def close(self) -> None:
        self._conn.close()

    # --- scoping ---------------------------------------------------------------------------------

    @staticmethod
    def _segment_clause(principal: Principal, table: str) -> tuple[str, list[Any]]:
        """Row-level filter. Empty segment set means unrestricted.

        Returns a clause containing only `?` placeholders plus the values to bind. No caller-supplied
        string is ever interpolated into SQL - the f-strings below splice this fixed clause, and
        every value travels as a bound parameter. Bandit flags the f-string construction (B608) and
        cannot see that distinction, hence the nosec markers with this as their justification.
        """
        if not principal.segments:
            return "", []
        placeholders = ",".join("?" for _ in principal.segments)
        return (
            f" AND ({table}.segment IS NULL OR {table}.segment IN ({placeholders}))",
            list(principal.segments),
        )

    # --- alerts ----------------------------------------------------------------------------------

    def save_alert(self, alert: Alert) -> None:
        self.save_alerts([alert])

    def save_alerts(self, alerts: list[Alert]) -> None:
        rows = [
            (
                a.alert_id,
                a.timestamp.isoformat(),
                a.verdict.value,
                a.lane.value,
                a.severity.value,
                a.priority,
                a.p_attack,
                a.novelty_percentile,
                a.family,
                a.raw_features.get("segment"),
                a.incident_id,
                a.model_dump_json(),
            )
            for a in alerts
        ]
        with self._tx() as conn:
            conn.executemany("INSERT OR REPLACE INTO alerts VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", rows)

    def get_alert(self, alert_id: str, principal: Principal) -> Alert | None:
        clause, params = self._segment_clause(principal, "alerts")
        row = self._conn.execute(
            f"SELECT payload FROM alerts WHERE alert_id = ?{clause}",  # nosec B608
            [alert_id, *params],
        ).fetchone()
        return Alert.model_validate_json(row["payload"]) if row else None

    def list_alerts(
        self,
        principal: Principal,
        *,
        lane: str | None = None,
        limit: int = 100,
        since: datetime | None = None,
    ) -> list[Alert]:
        sql = "SELECT payload FROM alerts WHERE 1=1"
        params: list[Any] = []
        if lane:
            sql += " AND lane = ?"
            params.append(lane)
        if since:
            sql += " AND timestamp >= ?"
            params.append(since.isoformat())
        clause, seg = self._segment_clause(principal, "alerts")
        sql += clause + " ORDER BY priority DESC, timestamp DESC LIMIT ?"
        params.extend([*seg, limit])
        return [Alert.model_validate_json(r["payload"]) for r in self._conn.execute(sql, params)]

    # --- incidents -------------------------------------------------------------------------------

    def save_incident(self, incident: Incident) -> None:
        with self._tx() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO incidents VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    incident.incident_id,
                    incident.opened_at.isoformat(),
                    incident.last_seen_at.isoformat(),
                    incident.title,
                    incident.severity.value,
                    incident.lane.value,
                    incident.priority,
                    incident.entity,
                    incident.family,
                    None,
                    incident.status,
                    incident.analyst_verdict,
                    incident.closed_by,
                    incident.event_count,
                    incident.model_dump_json(),
                ),
            )

    def get_incident(self, incident_id: str, principal: Principal) -> Incident | None:
        clause, params = self._segment_clause(principal, "incidents")
        row = self._conn.execute(
            f"SELECT payload FROM incidents WHERE incident_id = ?{clause}", [incident_id, *params]
        ).fetchone()
        return Incident.model_validate_json(row["payload"]) if row else None

    def list_incidents(
        self,
        principal: Principal,
        *,
        status: str | None = None,
        lane: str | None = None,
        limit: int = 100,
    ) -> list[Incident]:
        sql = "SELECT payload FROM incidents WHERE 1=1"
        params: list[Any] = []
        if status:
            sql += " AND status = ?"
            params.append(status)
        if lane:
            sql += " AND lane = ?"
            params.append(lane)
        clause, seg = self._segment_clause(principal, "incidents")
        sql += clause + " ORDER BY priority DESC, last_seen_at DESC LIMIT ?"
        params.extend([*seg, limit])
        return [Incident.model_validate_json(r["payload"]) for r in self._conn.execute(sql, params)]

    def alerts_for_incident(self, incident_id: str, principal: Principal) -> list[Alert]:
        clause, params = self._segment_clause(principal, "alerts")
        rows = self._conn.execute(
            f"SELECT payload FROM alerts WHERE incident_id = ?{clause} ORDER BY timestamp",
            [incident_id, *params],
        )
        return [Alert.model_validate_json(r["payload"]) for r in rows]

    # --- feedback --------------------------------------------------------------------------------

    def record_verdict(self, incident_id: str, verdict: str, actor: str, note: str = "") -> Incident | None:
        """Record a verdict and queue it. It does NOT enter the training pool here.

        Promotion is a separate, senior-gated step. That split is what stops one compromised
        analyst account from teaching the model to ignore its own traffic.
        """
        row = self._conn.execute(
            "SELECT payload FROM incidents WHERE incident_id = ?", [incident_id]
        ).fetchone()
        if row is None:
            return None

        incident = Incident.model_validate_json(row["payload"])
        incident.analyst_verdict = verdict
        incident.status = "triaging"

        top = self._conn.execute(
            "SELECT MAX(p_attack) AS p FROM alerts WHERE incident_id = ?", [incident_id]
        ).fetchone()["p"]
        self._queue_verdict(
            target=incident_id,
            kind="incident",
            verdict=verdict,
            actor=actor,
            note=note,
            p_attack=top,
            family=incident.family,
        )
        self.save_incident(incident)
        return incident

    def record_alert_verdict(
        self, alert_id: str, verdict: str, actor: str, principal: Principal, note: str = ""
    ) -> Alert | None:
        """A verdict on one alert, for traffic that never correlated into an incident.

        Correlation needs source IPs, and two of the three datasets have none (ADR-0004). Without
        this, every verdict button on an uncorrelated alert was a dead end. Scoped like any read:
        an analyst cannot judge an alert in a segment they cannot see.
        """
        alert = self.get_alert(alert_id, principal)
        if alert is None:
            return None
        self._queue_verdict(
            target=alert_id,
            kind="alert",
            verdict=verdict,
            actor=actor,
            note=note,
            p_attack=alert.p_attack,
            family=alert.family,
        )
        return alert

    def _queue_verdict(
        self,
        *,
        target: str,
        kind: str,
        verdict: str,
        actor: str,
        note: str,
        p_attack: float | None,
        family: str | None,
    ) -> None:
        with self._tx() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO verdict_queue "
                "(incident_id, verdict, actor, note, recorded_at, promoted, target_kind, p_attack, family) "
                "VALUES (?,?,?,?,?,0,?,?,?)",
                [target, verdict, actor, note, datetime.now(UTC).isoformat(), kind, p_attack, family],
            )

    def verdicts_by(self, actor: str, since: datetime) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) AS n FROM verdict_queue WHERE actor = ? AND recorded_at >= ?",
            [actor, since.isoformat()],
        ).fetchone()
        return int(row["n"])

    def pending_verdicts(self) -> list[dict[str, object]]:
        rows = self._conn.execute(
            "SELECT incident_id AS target_id, target_kind, verdict, actor, note, recorded_at, "
            "p_attack, family FROM verdict_queue WHERE promoted = 0 ORDER BY recorded_at"
        )
        return [dict(r) for r in rows]

    def verdict_history(self) -> list[dict[str, object]]:
        """Every verdict, promoted or not. The integrity checks judge an account on all of it."""
        rows = self._conn.execute(
            "SELECT incident_id AS target_id, target_kind, verdict, actor, recorded_at, p_attack, "
            "family, promoted, promoted_by FROM verdict_queue ORDER BY recorded_at"
        )
        return [dict(r) for r in rows]

    def promote_verdicts(self, incident_ids: list[str], approver: str) -> int:
        """Move verdicts into the training pool. Senior only - enforced at the router.

        **Two-person rule:** a verdict cannot be promoted by the account that recorded it. Seniors
        record verdicts too, and without this one compromised senior account could label its own
        traffic benign and then approve the label - the exact poisoning path the promotion step
        exists to close (THREAT_MODEL T1).
        """
        if not incident_ids:
            return 0
        placeholders = ",".join("?" for _ in incident_ids)
        with self._tx() as conn:
            cur = conn.execute(
                f"UPDATE verdict_queue SET promoted = 1, promoted_by = ?, promoted_at = ? "
                f"WHERE incident_id IN ({placeholders}) AND promoted = 0 AND actor != ?",
                [approver, datetime.now(UTC).isoformat(), *incident_ids, approver],
            )
            return int(cur.rowcount)

    def promoted_training_rows(self) -> list[dict[str, Any]]:
        """Promoted verdicts as labelled feature rows, ready for `penumbra retrain`.

        An incident verdict labels every alert in the incident. Alerts with no features (scored
        before features travelled with alerts) are skipped rather than guessed.
        """
        out: list[dict[str, Any]] = []
        rows = self._conn.execute(
            "SELECT incident_id AS target_id, target_kind, verdict, actor, promoted_by FROM verdict_queue "
            "WHERE promoted = 1 ORDER BY promoted_at"
        ).fetchall()
        for r in rows:
            label = TRAINING_LABEL.get(r["verdict"])
            if label is None:
                continue
            # One of two fixed column names, never input.
            column = "alert_id" if r["target_kind"] == "alert" else "incident_id"
            payloads = self._conn.execute(
                f"SELECT payload FROM alerts WHERE {column} = ?",  # nosec B608
                [r["target_id"]],
            )
            for p in payloads:
                alert = Alert.model_validate_json(p["payload"])
                if not alert.raw_features:
                    continue
                out.append(
                    {
                        "features": alert.raw_features,
                        "label": label,
                        "family": alert.family if label == 1 else None,
                        "p_attack": alert.p_attack,
                        "verdict": r["verdict"],
                        "actor": r["actor"],
                        "approver": r["promoted_by"],
                        "alert_id": alert.alert_id,
                    }
                )
        return out

    # --- suppression -----------------------------------------------------------------------------

    def save_suppression(self, rule: SuppressionRule) -> None:
        with self._tx() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO suppression_rules VALUES (?,?,?,?,?)",
                [
                    rule.rule_id,
                    rule.created_by,
                    rule.created_at.isoformat(),
                    rule.expires_at.isoformat(),
                    rule.model_dump_json(),
                ],
            )

    def list_suppressions(self, *, active_only: bool = False) -> list[SuppressionRule]:
        rules = [
            SuppressionRule.model_validate_json(r["payload"])
            for r in self._conn.execute("SELECT payload FROM suppression_rules ORDER BY created_at DESC")
        ]
        if active_only:
            now = datetime.now(UTC)
            rules = [r for r in rules if r.is_active(now)]
        return rules

    # --- counts ----------------------------------------------------------------------------------

    def counts(self, principal: Principal) -> dict[str, int]:
        clause, seg = self._segment_clause(principal, "alerts")
        alerts = self._conn.execute(
            f"SELECT lane, COUNT(*) AS n FROM alerts WHERE 1=1{clause} GROUP BY lane", seg
        )
        out = {f"alerts_{r['lane']}": int(r["n"]) for r in alerts}

        iclause, iseg = self._segment_clause(principal, "incidents")
        incidents = self._conn.execute(
            f"SELECT status, COUNT(*) AS n FROM incidents WHERE 1=1{iclause} GROUP BY status", iseg
        )
        out.update({f"incidents_{r['status']}": int(r["n"]) for r in incidents})

        total_alerts = self._conn.execute(
            f"SELECT COUNT(*) AS n FROM alerts WHERE 1=1{clause}", seg
        ).fetchone()["n"]
        total_incidents = self._conn.execute(
            f"SELECT COUNT(*) AS n FROM incidents WHERE 1=1{iclause}", iseg
        ).fetchone()["n"]

        out["alerts_total"] = int(total_alerts)
        out["incidents_total"] = int(total_incidents)

        # Compression is alerts-per-incident, and it is only meaningful over alerts that were
        # actually correlated INTO those incidents. Dividing every alert by every incident would
        # report a ratio built from unrelated rows - the same fabricated-metric problem the
        # correlator refuses to commit, so the repository must not commit it either.
        correlated = self._conn.execute(
            f"SELECT COUNT(*) AS n FROM alerts WHERE incident_id IS NOT NULL{clause}", seg
        ).fetchone()["n"]
        out["alerts_correlated"] = int(correlated)
        out["compression_ratio"] = int(correlated // total_incidents) if total_incidents and correlated else 0
        out["pending_verdicts"] = len(self.pending_verdicts())
        suppressed = self._conn.execute(
            f"SELECT COUNT(*) AS n FROM alerts WHERE verdict = 'BENIGN_BY_POLICY'{clause}", seg
        ).fetchone()["n"]
        out["alerts_suppressed"] = int(suppressed)
        return out

    def export_json(self) -> str:
        """Everything, for the fixture-mode demo fallback."""
        alerts = [json.loads(r["payload"]) for r in self._conn.execute("SELECT payload FROM alerts")]
        incidents = [json.loads(r["payload"]) for r in self._conn.execute("SELECT payload FROM incidents")]
        return json.dumps({"alerts": alerts, "incidents": incidents}, indent=2)
