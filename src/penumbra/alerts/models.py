"""The canonical Alert.

One object, three serialisers (ASIM / OCSF / ECS). Everything downstream - the API, the console, the
SIEM connectors, the correlation engine - agrees on this shape, which is why it is frozen early.

Three separate numbers, deliberately, per ADR-0002:

  p_attack            calibrated P(attack | x). The ONLY calibrated quantity we emit; Brier score
                      and the reliability diagram describe this and nothing else.
  novelty_percentile  "more unusual than 99.7% of known-benign traffic". A percentile, not a
                      probability, and self-explaining to an analyst in a way that a raw
                      reconstruction error is not.
  priority            0-100 triage ordering. NOT a probability. Blending a calibrated probability
                      with an anomaly score and calling the result calibrated is a category error,
                      so the sort key is kept separate and labelled.

And one thing that is deliberately absent: any field that could execute. `suggested_action` is text,
`requires_analyst_approval` is always True, and nothing in this package can act on either. ADR-0001.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Verdict(StrEnum):
    """The verdict lattice. See ADR-0002.

    There is no BLOCK, DENY or DROP member, because there is nothing in this system to deny traffic
    with. A test asserts that.
    """

    KNOWN_ATTACK = "KNOWN_ATTACK"
    SUSPECTED_NOVEL = "SUSPECTED_NOVEL"
    UNCERTAIN = "UNCERTAIN"
    BENIGN = "BENIGN"
    BENIGN_BY_POLICY = "BENIGN_BY_POLICY"


class Severity(StrEnum):
    """Four levels, matching ASIM's EventSeverity enum exactly.

    ASIM validates this as a string enum, not a number, and a mismatch fails ingestion - so the
    internal representation is the one the SIEM already accepts.
    """

    INFORMATIONAL = "Informational"
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"


class Lane(StrEnum):
    """Which queue an alert belongs to. See ADR-0002.

    The known-threat lane is SLA'd and ticket-generating. The hunting lane is a ranked queue with a
    fixed daily budget, whose honest metric is Precision@k rather than FPR - a budgeted queue cannot
    cause alert fatigue by construction.
    """

    KNOWN_THREAT = "known_threat"
    HUNTING = "hunting"
    REVIEW = "review"  # conformal abstention


class Contribution(BaseModel):
    """One feature's contribution, rendered for a tier-1 analyst.

    `narrative` is the point. "sload = 1.4e6" means nothing on a triage queue; "outbound throughput
    1.4 MB/s, typical for this host is 12 KB/s" is actionable by someone who is not an ML engineer.
    """

    feature: str
    value: float | str | None = None
    shap_value: float = 0.0
    direction: str = Field(default="toward_attack", pattern="^(toward_attack|toward_benign)$")
    narrative: str = ""


class AttackTechnique(BaseModel):
    """A MITRE ATT&CK mapping, with the confidence we actually have in it.

    `confidence` is load-bearing. No authoritative mapping from these dataset taxonomies to ATT&CK
    exists, so every mapping is our judgment. Two UNSW families (`Generic`, `Analysis`) have no
    defensible mapping at all and are represented by omitting this object rather than by inventing
    one. A wrong technique ID in front of a security judge is worse than an absent one.
    """

    tactic_id: str  # TA####
    tactic_name: str
    technique_id: str  # T####[.###]
    technique_name: str
    confidence: str = Field(default="medium", pattern="^(strong|good|medium|stretch)$")
    rationale: str = ""

    @property
    def reference(self) -> str:
        base = self.technique_id.split(".")[0]
        sub = self.technique_id.split(".")
        tail = f"{base}/{sub[1]}/" if len(sub) > 1 else f"{base}/"
        return f"https://attack.mitre.org/techniques/{tail}"


class NetworkContext(BaseModel):
    """The five-tuple, pseudonymised.

    IP addresses are personal data (GDPR Art. 4; Breyer, CJEU C-582/14). `src_ip`/`dst_ip` hold the
    HMAC pseudonym, never the raw address. Re-identification is an admin-only, audited operation.

    This is pseudonymisation and not anonymisation - IPv4 is 2**32 values and anyone holding the key
    can enumerate the whole mapping - so the output remains personal data and we do not claim
    otherwise.
    """

    src_ip: str | None = None
    src_port: int | None = None
    dst_ip: str | None = None
    dst_port: int | None = None
    protocol: str | None = None
    duration_ms: float | None = None
    src_bytes: int | None = None
    dst_bytes: int | None = None
    src_packets: int | None = None
    dst_packets: int | None = None
    is_pseudonymised: bool = True


class Alert(BaseModel):
    """A single detection, ready for an analyst or a SIEM."""

    model_config = ConfigDict(frozen=False, extra="forbid")

    alert_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))

    verdict: Verdict
    lane: Lane
    severity: Severity

    # --- the three numbers, kept apart on purpose
    p_attack: float = Field(ge=0.0, le=1.0, description="Calibrated P(attack). The only calibrated value.")
    novelty_percentile: float = Field(
        ge=0.0, le=1.0, description="Percentile against known-benign traffic. Not a probability."
    )
    priority: int = Field(ge=0, le=100, description="Triage sort key. NOT a probability.")

    family: str | None = None
    family_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    attack: AttackTechnique | None = None

    network: NetworkContext = Field(default_factory=NetworkContext)
    contributions: list[Contribution] = Field(default_factory=list)

    # --- conformal abstention
    conformal_set: list[str] = Field(default_factory=list)
    conformal_coverage: float | None = None

    # --- model provenance, so an alert can be traced to the exact artifact that produced it
    model_name: str = "penumbra"
    model_version: str = "0.1.0"
    detector_agreement: int = 0  # how many novelty detectors independently flagged it

    # --- ADR-0001: text, and only text
    suggested_action: str = ""
    requires_analyst_approval: bool = True

    # --- correlation
    incident_id: str | None = None
    suppression_rule_id: str | None = None

    raw_features: dict[str, Any] = Field(default_factory=dict)

    @field_validator("requires_analyst_approval")
    @classmethod
    def _approval_is_mandatory(cls, v: bool) -> bool:
        """This field exists to be True.

        It is not configurable and there is no code path that consumes it as False. Enforced here so
        that a future caller cannot construct an auto-actionable alert even by accident. See
        ADR-0001.
        """
        if not v:
            raise ValueError(
                "requires_analyst_approval cannot be False. Penumbra alerts; it does not act. "
                "See docs/adr/0001-alert-not-block.md"
            )
        return True

    @property
    def is_actionable_alert(self) -> bool:
        """True when this belongs in a queue at all."""
        return self.verdict in {Verdict.KNOWN_ATTACK, Verdict.SUSPECTED_NOVEL, Verdict.UNCERTAIN}

    @property
    def explanation(self) -> str:
        """One paragraph an analyst can read without opening the model."""
        if self.verdict is Verdict.SUSPECTED_NOVEL:
            head = (
                f"Traffic unlike anything in the benign baseline - more unusual than "
                f"{self.novelty_percentile:.1%} of known-normal flows. The classifier did not "
                f"recognise it as a known attack family, which is what this lane is for."
            )
        elif self.verdict is Verdict.KNOWN_ATTACK:
            head = f"Matches the {self.family} pattern with calibrated probability {self.p_attack:.2f}."
        elif self.verdict is Verdict.UNCERTAIN:
            head = "The model declined to commit: the conformal prediction set was ambiguous."
        else:
            head = "No detection."

        if self.contributions:
            why = " ".join(c.narrative for c in self.contributions[:3] if c.narrative)
            if why:
                head = f"{head} {why}"
        return head


class Incident(BaseModel):
    """Correlated alerts sharing an entity, family and time window.

    The anti-alert-fatigue mechanism: a scan producing thousands of flows becomes one incident with
    thousands of events, not thousands of tickets. The compression ratio is measured on CICIDS2017,
    which is the only dataset carrying real source IPs (ADR-0004) - quoting it from synthesised
    identifiers would be quoting a fabricated number.
    """

    model_config = ConfigDict(extra="forbid")

    incident_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    opened_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    last_seen_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    title: str
    severity: Severity
    lane: Lane
    priority: int = Field(ge=0, le=100)

    entity: str  # pseudonymised source
    family: str | None = None
    attack: AttackTechnique | None = None

    alert_ids: list[str] = Field(default_factory=list)
    event_count: int = 0
    distinct_destinations: int = 0
    distinct_ports: int = 0

    status: str = Field(default="open", pattern="^(open|triaging|closed|suppressed)$")
    analyst_verdict: str | None = Field(
        default=None, pattern="^(true_positive|false_positive|benign_by_policy)$"
    )
    closed_by: str | None = None

    @property
    def compression_ratio(self) -> float:
        """Events folded into this one incident."""
        return float(self.event_count) if self.event_count else 0.0
