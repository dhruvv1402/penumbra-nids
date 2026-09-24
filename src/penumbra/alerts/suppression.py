"""Suppression rules: the BENIGN_BY_POLICY half of the verdict lattice.

An internal vulnerability scanner fires the port-scan detector every night. It is not a false
positive - it IS a port scan - and it is not a threat either. The honest answer is a policy
decision, recorded as one: who decided, why, what it matches, and when it stops matching.

Three constraints are enforced by the type rather than left to discipline:

  * **Every rule expires, and no later than `MAX_LIFETIME`.** A permanent suppression is a
    permanent blind spot: the scanner allowlisted in March is the C2 channel missed in September.
  * **Every rule is scoped to something narrower than a family.** "Suppress all DoS" is not a
    policy, it is switching the detector off. A rule must name at least one of source, destination,
    port or service.
  * **A match reclassifies; it never deletes.** The alert is still stored, still counted and still
    audit-traceable to the rule that suppressed it. "What did we suppress last quarter?" has an
    answer.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from penumbra.alerts.models import Alert

MAX_LIFETIME = timedelta(days=90)

# Fields a rule may match on. `family` narrows; the rest scope. A rule needs at least one scoping
# field, so family alone is rejected.
SCOPING_FIELDS = ("src_ip", "dst_ip", "dst_port", "service")
MATCH_FIELDS = (*SCOPING_FIELDS, "family", "protocol")


class SuppressionRule(BaseModel):
    """One analyst-authored policy exception. Satisfies `storage.repository.SuppressionRule`."""

    model_config = ConfigDict(extra="forbid")

    rule_id: str = Field(default_factory=lambda: f"sup-{uuid.uuid4().hex[:12]}")
    match: dict[str, str]
    reason: str = Field(min_length=10, max_length=500)
    created_by: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    expires_at: datetime  # not optional, on purpose
    source_alert_id: str | None = None

    @field_validator("match")
    @classmethod
    def _known_fields(cls, v: dict[str, str]) -> dict[str, str]:
        unknown = sorted(set(v) - set(MATCH_FIELDS))
        if unknown:
            raise ValueError(f"cannot match on {unknown}; allowed: {list(MATCH_FIELDS)}")
        cleaned = {k: str(val).strip() for k, val in v.items() if str(val).strip()}
        if not any(k in cleaned for k in SCOPING_FIELDS):
            raise ValueError(
                f"a suppression must be scoped by at least one of {list(SCOPING_FIELDS)}; "
                "suppressing a whole family is switching the detector off"
            )
        # Wildcards, and the datasets' "no value" placeholders. UNSW writes "-" for "no service" on
        # roughly half its flows, so {"service": "-"} passed as a scoped rule and would have switched
        # detection off across every family. A placeholder scopes nothing.
        blocked = {
            "*",
            "any",
            "all",
            "0.0.0.0/0",
            "::/0",
            "-",
            "--",
            "none",
            "null",
            "nan",
            "n/a",
            "unknown",
            "other",
        }
        if any(val.lower() in blocked for val in cleaned.values()):
            raise ValueError("wildcards and placeholder values ('-', 'none', ...) cannot scope a suppression")
        return cleaned

    @model_validator(mode="after")
    def _bounded_expiry(self) -> SuppressionRule:
        created = _aware(self.created_at)
        expires = _aware(self.expires_at)
        if expires <= created:
            raise ValueError("expires_at must be after created_at")
        if expires - created > MAX_LIFETIME:
            raise ValueError(
                f"a suppression may last at most {MAX_LIFETIME.days} days; renew it deliberately"
            )
        return self

    # --- the Protocol ----------------------------------------------------------------------------

    @property
    def entity(self) -> str:
        return self.match.get("src_ip") or self.match.get("dst_ip") or self.match.get("service") or "-"

    @property
    def family(self) -> str | None:
        return self.match.get("family")

    def is_active(self, now: datetime | None = None) -> bool:
        now = _aware(now or datetime.now(UTC))
        return _aware(self.created_at) <= now < _aware(self.expires_at)

    def matches(self, alert: Alert) -> bool:
        observed = alert_fields(alert)
        return all((observed.get(k) or "").lower() == v.lower() for k, v in self.match.items())


def alert_fields(alert: Alert) -> dict[str, str | None]:
    """The matchable view of an alert. Missing values never match anything."""
    net = alert.network
    service = alert.raw_features.get("service")
    return {
        "src_ip": net.src_ip,
        "dst_ip": net.dst_ip,
        "dst_port": str(net.dst_port) if net.dst_port is not None else None,
        "protocol": net.protocol,
        "service": str(service) if service is not None else None,
        "family": alert.family,
    }


def proposed_match(alert: Alert) -> dict[str, str]:
    """The narrowest sensible rule for an alert, for the console to pre-fill.

    Everything the alert carries that scopes it, plus its family when it has one. The analyst can
    loosen it; the default should never be looser than the evidence.
    """
    return {k: v for k, v in alert_fields(alert).items() if v is not None and k in MATCH_FIELDS}


def first_match(
    rules: list[SuppressionRule], alert: Alert, now: datetime | None = None
) -> SuppressionRule | None:
    now = now or datetime.now(UTC)
    for rule in rules:
        if rule.is_active(now) and rule.matches(alert):
            return rule
    return None


def apply(alert: Alert, rule: SuppressionRule) -> Alert:
    """Reclassify a matched alert. Scores are untouched: the model's opinion is still on record."""
    from penumbra.alerts.models import Verdict

    alert.verdict = Verdict.BENIGN_BY_POLICY
    alert.suppression_rule_id = rule.rule_id
    alert.suggested_action = (
        f"Suppressed by policy {rule.rule_id} ({rule.reason}) until {rule.expires_at:%Y-%m-%d}. "
        "Stored and counted; not queued."
    )
    return alert


def _aware(dt: datetime) -> datetime:
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)


def summary(rule: SuppressionRule) -> dict[str, Any]:
    return {**rule.model_dump(mode="json"), "active": rule.is_active()}
