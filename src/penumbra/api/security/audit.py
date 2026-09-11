"""Append-only, hash-chained audit log.

Each entry commits to the digest of the previous one, so editing or deleting any entry breaks the
chain from that point forward and `verify()` reports exactly where. An attacker with database write
access can still destroy the log; what they cannot do is quietly rewrite one line of it.

This is not a blockchain and does not pretend to be. It is a hash chain, which is the useful 1% of
that idea: cheap, dependency-free, and it turns silent tampering into loud tampering.

What gets logged, and why each matters:

  analyst verdicts        they feed retraining, so they are a poisoning vector (THREAT_MODEL T1)
  incident closures       who decided, and when
  suppression rules       a permanent blind spot needs an owner and an expiry
  PII re-identification   admin-only, and a re-identification with no corresponding incident is
                          itself an incident
  model promotions        which artifact became champion, on whose authority
  threshold changes       an operating point moved, quietly, is a change in how much gets missed
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

GENESIS = "0" * 64


@dataclass(frozen=True)
class AuditEntry:
    sequence: int
    timestamp: str
    actor: str
    role: str
    action: str
    target: str
    detail: dict[str, Any]
    previous_hash: str
    entry_hash: str

    def payload(self) -> dict[str, Any]:
        """The fields the hash covers. `entry_hash` itself is necessarily excluded."""
        return {
            "sequence": self.sequence,
            "timestamp": self.timestamp,
            "actor": self.actor,
            "role": self.role,
            "action": self.action,
            "target": self.target,
            "detail": self.detail,
            "previous_hash": self.previous_hash,
        }

    def compute_hash(self) -> str:
        # sort_keys so the digest does not depend on dict insertion order, which would make the
        # chain fail to verify across Python versions or after a round-trip through JSON.
        canonical = json.dumps(self.payload(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def is_intact(self) -> bool:
        return self.entry_hash == self.compute_hash()

    def to_json(self) -> str:
        return json.dumps({**self.payload(), "entry_hash": self.entry_hash}, sort_keys=True)

    @classmethod
    def from_json(cls, line: str) -> AuditEntry:
        data = json.loads(line)
        return cls(
            sequence=data["sequence"],
            timestamp=data["timestamp"],
            actor=data["actor"],
            role=data["role"],
            action=data["action"],
            target=data["target"],
            detail=data["detail"],
            previous_hash=data["previous_hash"],
            entry_hash=data["entry_hash"],
        )


@dataclass
class VerificationResult:
    valid: bool
    n_entries: int
    broken_at: int | None = None
    reason: str = ""

    def summary(self) -> str:
        if self.valid:
            return f"  Audit chain intact across {self.n_entries:,} entries."
        return (
            f"  AUDIT CHAIN BROKEN at entry {self.broken_at}: {self.reason}\n"
            f"  Entries after this point cannot be trusted."
        )


class AuditLog:
    """Append-only hash-chained log, persisted as JSONL.

    JSONL rather than a database table on purpose: it is append-only at the filesystem level, it
    survives the application, and it can be verified by anything that can read lines.
    """

    def __init__(self, path: Path | None = None) -> None:
        self.path = path
        self._entries: list[AuditEntry] = []
        if path and path.exists():
            self._load()

    def _load(self) -> None:
        assert self.path is not None
        with self.path.open("r", encoding="utf-8") as fh:
            self._entries = [AuditEntry.from_json(line) for line in fh if line.strip()]

    @property
    def head(self) -> str:
        return self._entries[-1].entry_hash if self._entries else GENESIS

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self) -> Iterator[AuditEntry]:
        return iter(self._entries)

    def append(
        self,
        *,
        actor: str,
        role: str,
        action: str,
        target: str = "",
        detail: dict[str, Any] | None = None,
    ) -> AuditEntry:
        """Record an action. There is no update or delete - that is the point."""
        entry = AuditEntry(
            sequence=len(self._entries),
            timestamp=datetime.now(UTC).isoformat(),
            actor=actor,
            role=role,
            action=action,
            target=target,
            detail=detail or {},
            previous_hash=self.head,
            entry_hash="",
        )
        # Hash the payload, then rebuild the frozen record carrying its own digest.
        sealed = AuditEntry(**{**entry.payload(), "entry_hash": entry.compute_hash()})
        self._entries.append(sealed)

        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(sealed.to_json() + "\n")
        return sealed

    def verify(self) -> VerificationResult:
        """Walk the chain and report the first break.

        Two independent checks per entry: the digest still matches its own contents (nothing was
        edited), and its `previous_hash` matches the entry before it (nothing was inserted, removed
        or reordered).
        """
        previous = GENESIS
        for entry in self._entries:
            if not entry.is_intact():
                return VerificationResult(
                    False, len(self._entries), entry.sequence, "entry content was modified"
                )
            if entry.previous_hash != previous:
                return VerificationResult(
                    False,
                    len(self._entries),
                    entry.sequence,
                    "chain link mismatch - an entry was inserted, removed or reordered",
                )
            previous = entry.entry_hash
        return VerificationResult(True, len(self._entries))

    def by_action(self, action: str) -> list[AuditEntry]:
        return [e for e in self._entries if e.action == action]

    def by_actor(self, actor: str) -> list[AuditEntry]:
        return [e for e in self._entries if e.actor == actor]

    def tail(self, n: int = 20) -> list[AuditEntry]:
        return self._entries[-n:]


# --- The actions worth recording -----------------------------------------------------------------

ACTION_VERDICT = "analyst.verdict"
ACTION_CLOSE = "incident.close"
ACTION_SUPPRESS = "suppression.create"
ACTION_REIDENTIFY = "pii.reidentify"
ACTION_PROMOTE = "model.promote"
ACTION_THRESHOLD = "threshold.change"
ACTION_LOGIN = "auth.login"
ACTION_DENIED = "auth.denied"


def record_verdict(
    log: AuditLog, *, actor: str, role: str, incident_id: str, verdict: str, note: str = ""
) -> AuditEntry:
    """Log an analyst verdict.

    Provenance matters here more than anywhere else: verdicts feed the retraining pool, so a
    compromised account can teach the model to ignore its own traffic. The audit trail is what makes
    that attack attributable after the fact, and the senior-approval gate is what makes it hard
    before. See docs/THREAT_MODEL.md T1.
    """
    return log.append(
        actor=actor,
        role=role,
        action=ACTION_VERDICT,
        target=incident_id,
        detail={"verdict": verdict, "note": note},
    )
