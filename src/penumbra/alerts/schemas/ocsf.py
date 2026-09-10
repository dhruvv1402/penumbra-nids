"""OCSF Detection Finding (class_uid 2004), schema 1.5.0.

Vendor-neutral, and the target for data lakes and AWS Security Lake.

The gotcha this module exists to get right: **Detection Finding has no top-level `src_endpoint` or
`dst_endpoint`.** The network five-tuple lives inside `evidences[]`, which additionally carries an
"at least one of" constraint - an Evidence object with none of the recognised keys is invalid.
Emitting the endpoints at the top level produces a document that looks correct and is not.

`type_uid` is derived, never hardcoded: `class_uid * 100 + activity_id`.
"""

from __future__ import annotations

from typing import Any, Final

from penumbra.alerts.models import Alert, Severity, Verdict

SCHEMA_VERSION: Final[str] = "1.5.0"
CLASS_UID: Final[int] = 2004  # Detection Finding
CATEGORY_UID: Final[int] = 2  # Findings

ACTIVITY_CREATE: Final[int] = 1

SEVERITY_ID: Final[dict[Severity, int]] = {
    Severity.INFORMATIONAL: 1,
    Severity.LOW: 2,
    Severity.MEDIUM: 3,
    Severity.HIGH: 4,
}

# 0 Unknown, 1 Low, 2 Medium, 3 High
CONFIDENCE_BANDS: Final[tuple[tuple[float, int], ...]] = ((0.34, 1), (0.67, 2), (1.01, 3))

STATUS_NEW: Final[int] = 1


def _confidence_id(p: float) -> int:
    for upper, band in CONFIDENCE_BANDS:
        if p < upper:
            return band
    return 3


def to_ocsf(alert: Alert) -> dict[str, Any]:
    """Project an Alert onto an OCSF Detection Finding."""
    activity_id = ACTIVITY_CREATE

    evidence: dict[str, Any] = {}
    net = alert.network
    if net.src_ip:
        evidence["src_endpoint"] = {"ip": net.src_ip, **({"port": net.src_port} if net.src_port else {})}
    if net.dst_ip:
        evidence["dst_endpoint"] = {"ip": net.dst_ip, **({"port": net.dst_port} if net.dst_port else {})}
    if net.protocol:
        evidence["connection_info"] = {"protocol_name": net.protocol.lower()}
    if not evidence:
        # The Evidence object requires at least one recognised key. `data` is the documented
        # fallback and is better than emitting an invalid empty object.
        evidence["data"] = {"note": "flow features only; no endpoint identifiers in this dataset"}

    finding_info: dict[str, Any] = {
        "uid": alert.alert_id,
        "title": _title(alert),
        "analytic": {
            "name": alert.model_name,
            "type_id": 3,  # Learning (ML)
            "version": alert.model_version,
        },
        "desc": alert.explanation,
    }
    if alert.attack:
        finding_info["attacks"] = [
            {
                "tactic": {"uid": alert.attack.tactic_id, "name": alert.attack.tactic_name},
                "technique": {"uid": alert.attack.technique_id, "name": alert.attack.technique_name},
                "version": "16",
            }
        ]

    record: dict[str, Any] = {
        "class_uid": CLASS_UID,
        "category_uid": CATEGORY_UID,
        "activity_id": activity_id,
        "type_uid": CLASS_UID * 100 + activity_id,
        "time": int(alert.timestamp.timestamp() * 1000),
        "severity_id": SEVERITY_ID[alert.severity],
        "status_id": STATUS_NEW,
        "confidence_id": _confidence_id(alert.p_attack),
        "confidence_score": int(round(alert.p_attack * 100)),
        "metadata": {
            "version": SCHEMA_VERSION,
            "product": {
                "name": "Penumbra NIDS",
                "vendor_name": "Penumbra",
                "version": alert.model_version,
            },
        },
        "finding_info": finding_info,
        "evidences": [evidence],
        "message": alert.explanation,
        "unmapped": {
            "verdict": alert.verdict.value,
            "lane": alert.lane.value,
            "novelty_percentile": round(alert.novelty_percentile, 4),
            "priority": alert.priority,
            "priority_is_not_a_probability": True,
            "requires_analyst_approval": alert.requires_analyst_approval,
            "suggested_action": alert.suggested_action,
            "detector_agreement": alert.detector_agreement,
        },
    }

    if net.src_ip:
        record["observables"] = [{"name": "src_endpoint.ip", "type_id": 2, "value": net.src_ip}]
    return record


def _title(alert: Alert) -> str:
    if alert.verdict is Verdict.SUSPECTED_NOVEL:
        src = alert.network.src_ip or "unknown source"
        return f"Anomalous network session from {src} with no known attack family"
    return f"{alert.family or 'Anomalous'} network session detected"


def validate(record: dict[str, Any]) -> list[str]:
    problems: list[str] = []

    for field in (
        "class_uid",
        "category_uid",
        "activity_id",
        "type_uid",
        "time",
        "severity_id",
        "metadata",
        "finding_info",
    ):
        if field not in record:
            problems.append(f"missing required field {field}")

    if record.get("class_uid") != CLASS_UID:
        problems.append(f"class_uid must be {CLASS_UID}")
    if record.get("category_uid") != CATEGORY_UID:
        problems.append(f"category_uid must be {CATEGORY_UID}")

    # The derivation is easy to get wrong by hardcoding.
    expected = record.get("class_uid", 0) * 100 + record.get("activity_id", 0)
    if record.get("type_uid") != expected:
        problems.append(f"type_uid must equal class_uid*100 + activity_id ({expected})")

    if record.get("severity_id") not in range(0, 7) and record.get("severity_id") != 99:
        problems.append(f"severity_id {record.get('severity_id')!r} out of range")

    meta = record.get("metadata", {})
    if "product" not in meta:
        problems.append("metadata.product is required")

    if "uid" not in record.get("finding_info", {}):
        problems.append("finding_info.uid is required")

    evidences = record.get("evidences", [])
    if not evidences:
        problems.append("evidences must contain at least one Evidence object")
    else:
        allowed = {
            "actor",
            "api",
            "connection_info",
            "data",
            "database",
            "databucket",
            "device",
            "dst_endpoint",
            "email",
            "file",
            "job",
            "process",
            "query",
            "reg_key",
            "reg_value",
            "script",
            "src_endpoint",
            "url",
            "user",
            "win_service",
        }
        for i, ev in enumerate(evidences):
            if not (set(ev) & allowed):
                problems.append(f"evidences[{i}] has none of the required Evidence attributes")

    # Detection Finding does not define these at the top level; emitting them there is the common
    # mistake and produces a document that validates nowhere.
    for wrong in ("src_endpoint", "dst_endpoint"):
        if wrong in record:
            problems.append(f"{wrong} does not belong at the top level; it goes inside evidences[]")

    return problems


def assert_conformant(record: dict[str, Any]) -> None:
    problems = validate(record)
    if problems:
        raise ValueError("OCSF Detection Finding conformance failed:\n  " + "\n  ".join(problems))
