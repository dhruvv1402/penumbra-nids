"""Microsoft Sentinel ASIM `NetworkSession`, schema version 0.2.7.

ASIM's own documentation defines `EventType: IDS` as "a network session reported as suspicious" -
which is exactly what Penumbra emits, so we use the vendor's own event type rather than inventing a
custom table shape.

**The alert-not-block decision, expressed in Microsoft's vocabulary:**

    DvcAction        = "Allow"     <- we did not block
    ThreatConfidence = 93          <- and we are confident
    EventSeverity    = "High"

That field combination *is* ADR-0001, stated in the SIEM's normalized schema instead of in prose.
A reviewer can read it off the payload.

Validation traps this module exists to get right, each of which fails ingestion silently or noisily:

  * `EventSeverity` is a four-value STRING enum, not a number.
  * `ThreatField` is CONDITIONAL - required whenever `ThreatIpAddr` is set.
  * `ThreatConfidence` and `ThreatRiskLevel` are integers 0-100, not floats 0-1.
  * `EventSchemaVersion` must match the schema the parser declares.
  * Column names must start with a letter and be at most 45 alphanumeric-or-underscore characters.

`EventVendor`/`EventProduct` are drawn from a closed Microsoft allow-list that Penumbra is not on.
We use our own designator and say so; production use would need Microsoft to allocate one. Knowing
that constraint is better than pretending it does not exist.
"""

from __future__ import annotations

from typing import Any, Final

from penumbra.alerts.models import Alert, Verdict

SCHEMA: Final[str] = "NetworkSession"
SCHEMA_VERSION: Final[str] = "0.2.7"
EVENT_VENDOR: Final[str] = "Penumbra"
EVENT_PRODUCT: Final[str] = "Penumbra NIDS"

# Mandatory at the schema level - absence fails ASIM conformance.
MANDATORY_FIELDS: Final[tuple[str, ...]] = (
    "TimeGenerated",
    "EventCount",
    "EventStartTime",
    "EventEndTime",
    "EventType",
    "EventResult",
    "EventProduct",
    "EventVendor",
    "EventSchema",
    "EventSchemaVersion",
    "Dvc",
)

EVENT_TYPES: Final[frozenset[str]] = frozenset(
    {"NetworkSession", "L2NetworkSession", "Flow", "EndpointNetworkSession", "IDS"}
)
EVENT_RESULTS: Final[frozenset[str]] = frozenset({"Success", "Partial", "Failure", "NA"})
EVENT_SEVERITIES: Final[frozenset[str]] = frozenset({"Informational", "Low", "Medium", "High"})
DVC_ACTIONS: Final[frozenset[str]] = frozenset(
    {
        "Allow",
        "Deny",
        "Drop",
        "Drop ICMP",
        "Reset",
        "Reset Source",
        "Reset Destination",
        "Encrypt",
        "Decrypt",
        "VPNroute",
    }
)
THREAT_FIELDS: Final[frozenset[str]] = frozenset({"SrcIpAddr", "DstIpAddr"})


def to_asim(alert: Alert, *, device: str = "penumbra-sensor-01") -> dict[str, Any]:
    """Project an Alert onto ASIM NetworkSession."""
    ts = alert.timestamp.isoformat()

    record: dict[str, Any] = {
        # --- mandatory
        "TimeGenerated": ts,
        "EventCount": 1,
        "EventStartTime": ts,
        "EventEndTime": ts,
        "EventType": "IDS",
        # ASIM's rule: DvcAction Allow implies the session succeeded. We never deny, so this is
        # always Success - which is itself the point being made.
        "EventResult": "Success",
        "EventProduct": EVENT_PRODUCT,
        "EventVendor": EVENT_VENDOR,
        "EventSchema": SCHEMA,
        "EventSchemaVersion": SCHEMA_VERSION,
        "Dvc": device,
        # --- recommended
        "EventSeverity": alert.severity.value,
        "EventUid": alert.alert_id,
        "DvcHostname": device,
        # ADR-0001, in one field. We observed; we did not act.
        "DvcAction": "Allow",
        "EventMessage": alert.explanation,
        # --- inspection fields: the model's verdict
        "ThreatId": alert.alert_id,
        "ThreatName": _threat_name(alert),
        "ThreatCategory": alert.family or _category(alert),
        # Integers 0-100. ThreatConfidence carries the CALIBRATED probability, never `priority`,
        # which is a sort key and would misrepresent itself here.
        "ThreatConfidence": int(round(alert.p_attack * 100)),
        "ThreatRiskLevel": int(alert.priority),
        "ThreatIsActive": True,
        "NetworkRuleName": f"{alert.model_name}-{alert.model_version}/{alert.family or 'novel'}",
    }

    net = alert.network
    if net.src_ip:
        record["SrcIpAddr"] = net.src_ip
        # CONDITIONAL: ThreatField is required whenever ThreatIpAddr is present.
        record["ThreatIpAddr"] = net.src_ip
        record["ThreatField"] = "SrcIpAddr"
    if net.dst_ip:
        record["DstIpAddr"] = net.dst_ip
    if net.src_port is not None:
        record["SrcPortNumber"] = net.src_port
    if net.dst_port is not None:
        record["DstPortNumber"] = net.dst_port
    if net.protocol:
        record["NetworkProtocol"] = net.protocol.upper()
    if net.duration_ms is not None:
        record["NetworkDuration"] = int(net.duration_ms)
    if net.src_bytes is not None:
        record["SrcBytes"] = net.src_bytes
    if net.dst_bytes is not None:
        record["DstBytes"] = net.dst_bytes
    if net.src_packets is not None:
        record["SrcPackets"] = net.src_packets
    if net.dst_packets is not None:
        record["DstPackets"] = net.dst_packets
    if net.src_bytes is not None and net.dst_bytes is not None:
        record["NetworkBytes"] = net.src_bytes + net.dst_bytes

    # AdditionalFields is ASIM's documented escape hatch. Model metadata that has no normalized
    # home goes here rather than being forced into a field that means something else.
    record["AdditionalFields"] = {
        "Verdict": alert.verdict.value,
        "Lane": alert.lane.value,
        # Explicitly NOT ThreatConfidence: a percentile is not a probability.
        "NoveltyPercentile": round(alert.novelty_percentile, 4),
        "Priority": alert.priority,
        "PriorityIsNotAProbability": True,
        "DetectorAgreement": alert.detector_agreement,
        "RequiresAnalystApproval": alert.requires_analyst_approval,
        "SuggestedAction": alert.suggested_action,
        "ModelVersion": alert.model_version,
        "TopContributions": [
            {"Feature": c.feature, "Shap": round(c.shap_value, 5), "Why": c.narrative}
            for c in alert.contributions[:5]
        ],
    }
    if alert.attack:
        record["AdditionalFields"].update(
            {
                "AttackTactic": alert.attack.tactic_id,
                "AttackTechnique": alert.attack.technique_id,
                "AttackTechniqueName": alert.attack.technique_name,
                "AttackMappingConfidence": alert.attack.confidence,
            }
        )
    if alert.incident_id:
        record["AdditionalFields"]["IncidentId"] = alert.incident_id

    return record


def _threat_name(alert: Alert) -> str:
    if alert.verdict is Verdict.SUSPECTED_NOVEL:
        return "Anomalous network session (no known family)"
    if alert.family:
        return f"{alert.family} network session"
    return "Anomalous network session"


def _category(alert: Alert) -> str:
    return {
        Verdict.SUSPECTED_NOVEL: "Anomaly",
        Verdict.UNCERTAIN: "Anomaly",
        Verdict.KNOWN_ATTACK: "Malicious",
    }.get(alert.verdict, "Informational")


# =================================================================================================
# Conformance
# =================================================================================================


class ConformanceError(ValueError):
    pass


def validate(record: dict[str, Any]) -> list[str]:
    """Check an ASIM record. Returns a list of problems; empty means conformant.

    A green suite proving "our output is Sentinel-ingestible" is more persuasive, and more
    verifiable by a reader, than a screenshot of a portal.
    """
    problems: list[str] = []

    for field in MANDATORY_FIELDS:
        if field not in record or record[field] in (None, ""):
            problems.append(f"missing mandatory field {field}")

    def enum_check(field: str, allowed: frozenset[str]) -> None:
        if field in record and record[field] not in allowed:
            problems.append(f"{field}={record[field]!r} not in {sorted(allowed)}")

    enum_check("EventType", EVENT_TYPES)
    enum_check("EventResult", EVENT_RESULTS)
    enum_check("EventSeverity", EVENT_SEVERITIES)
    enum_check("DvcAction", DVC_ACTIONS)
    enum_check("ThreatField", THREAT_FIELDS)

    if record.get("EventSchema") != SCHEMA:
        problems.append(f"EventSchema must be {SCHEMA!r}")
    if record.get("EventSchemaVersion") != SCHEMA_VERSION:
        problems.append(f"EventSchemaVersion must be {SCHEMA_VERSION!r}")

    for field in ("ThreatConfidence", "ThreatRiskLevel"):
        if field in record:
            v = record[field]
            if not isinstance(v, int) or not (0 <= v <= 100):
                problems.append(f"{field} must be an int in [0,100], got {v!r}")

    # The conditional that is easiest to miss.
    if record.get("ThreatIpAddr") and not record.get("ThreatField"):
        problems.append("ThreatField is required whenever ThreatIpAddr is set")

    # Log Analytics column-name rules.
    for name in record:
        if not name[:1].isalpha():
            problems.append(f"column {name!r} must start with a letter")
        if len(name) > 45:
            problems.append(f"column {name!r} exceeds 45 characters")

    # Penumbra never blocks. If a denying action ever appears here, something upstream is wrong in
    # a way that matters more than schema conformance.
    if record.get("DvcAction") in {"Deny", "Drop", "Drop ICMP", "Reset", "Reset Source", "Reset Destination"}:
        problems.append(
            f"DvcAction={record['DvcAction']!r} implies enforcement. Penumbra alerts; it does not "
            "block. See docs/adr/0001-alert-not-block.md"
        )

    return problems


def assert_conformant(record: dict[str, Any]) -> None:
    problems = validate(record)
    if problems:
        raise ConformanceError("ASIM NetworkSession conformance failed:\n  " + "\n  ".join(problems))
