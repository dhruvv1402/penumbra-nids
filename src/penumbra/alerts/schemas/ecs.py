"""Elastic Common Schema.

The gotcha: **never emit `event.kind: "signal"`.** Elastic reserves that value for Kibana's own
detection engine, and the ECS documentation states plainly that ingestion pipelines must not
populate it. `alert` is the correct value for an IDS finding, and the docs name intrusion detection
systems explicitly when defining it.
"""

from __future__ import annotations

from typing import Any, Final

from penumbra.alerts.models import Alert, Severity, Verdict

ECS_VERSION: Final[str] = "8.11.0"

# Reserved for Kibana's alerting framework. Emitting it from a pipeline is a documented error.
FORBIDDEN_EVENT_KINDS: Final[frozenset[str]] = frozenset({"signal"})

EVENT_KINDS: Final[frozenset[str]] = frozenset(
    {"alert", "asset", "enrichment", "event", "metric", "state", "pipeline_error", "signal"}
)
EVENT_CATEGORIES: Final[frozenset[str]] = frozenset(
    {
        "api",
        "authentication",
        "configuration",
        "database",
        "driver",
        "email",
        "file",
        "host",
        "iam",
        "intrusion_detection",
        "library",
        "malware",
        "network",
        "package",
        "process",
        "registry",
        "session",
        "threat",
        "vulnerability",
        "web",
    }
)

SEVERITY_NUM: Final[dict[Severity, int]] = {
    Severity.INFORMATIONAL: 1,
    Severity.LOW: 2,
    Severity.MEDIUM: 3,
    Severity.HIGH: 4,
}


def to_ecs(alert: Alert) -> dict[str, Any]:
    """Project an Alert onto ECS."""
    net = alert.network

    record: dict[str, Any] = {
        "@timestamp": alert.timestamp.isoformat(),
        "ecs": {"version": ECS_VERSION},
        "event": {
            "kind": "alert",
            "category": ["intrusion_detection", "network"],
            "type": ["info"],
            "action": "penumbra-detection",
            "outcome": "success",
            "severity": SEVERITY_NUM[alert.severity],
            "risk_score": alert.priority,
            "dataset": "penumbra.alerts",
            "module": "penumbra",
            "provider": "penumbra",
            "id": alert.alert_id,
        },
        "rule": {
            "name": f"Penumbra {alert.family or 'novelty'} detector",
            "version": alert.model_version,
            "ruleset": "penumbra",
        },
        "message": alert.explanation,
        "labels": {
            "verdict": alert.verdict.value,
            "lane": alert.lane.value,
            "requires_analyst_approval": str(alert.requires_analyst_approval).lower(),
        },
        "penumbra": {
            "p_attack": round(alert.p_attack, 6),
            "novelty_percentile": round(alert.novelty_percentile, 6),
            "priority": alert.priority,
            "priority_is_not_a_probability": True,
            "detector_agreement": alert.detector_agreement,
            "suggested_action": alert.suggested_action,
            "contributions": [
                {"feature": c.feature, "shap": round(c.shap_value, 5), "narrative": c.narrative}
                for c in alert.contributions[:5]
            ],
        },
    }

    if net.src_ip:
        record["source"] = {
            "ip": net.src_ip,
            **({"port": net.src_port} if net.src_port is not None else {}),
            **({"bytes": net.src_bytes} if net.src_bytes is not None else {}),
            **({"packets": net.src_packets} if net.src_packets is not None else {}),
        }
    if net.dst_ip:
        record["destination"] = {
            "ip": net.dst_ip,
            **({"port": net.dst_port} if net.dst_port is not None else {}),
            **({"bytes": net.dst_bytes} if net.dst_bytes is not None else {}),
            **({"packets": net.dst_packets} if net.dst_packets is not None else {}),
        }
    if net.protocol or net.src_bytes is not None:
        record["network"] = {
            **({"transport": net.protocol.lower()} if net.protocol else {}),
            **(
                {"bytes": (net.src_bytes or 0) + (net.dst_bytes or 0)}
                if net.src_bytes is not None or net.dst_bytes is not None
                else {}
            ),
        }

    if alert.attack:
        record["threat"] = {
            "framework": "MITRE ATT&CK",
            "tactic": {
                "id": alert.attack.tactic_id,
                "name": alert.attack.tactic_name,
                "reference": f"https://attack.mitre.org/tactics/{alert.attack.tactic_id}/",
            },
            "technique": {
                "id": alert.attack.technique_id,
                "name": alert.attack.technique_name,
                "reference": alert.attack.reference,
            },
        }

    if alert.verdict is Verdict.SUSPECTED_NOVEL:
        record["event"]["reason"] = "unlike known-benign traffic; no known attack family matched"

    return record


def validate(record: dict[str, Any]) -> list[str]:
    problems: list[str] = []
    event = record.get("event", {})

    if "@timestamp" not in record:
        problems.append("missing @timestamp")

    kind = event.get("kind")
    if kind not in EVENT_KINDS:
        problems.append(f"event.kind={kind!r} not an ECS-allowed value")
    if kind in FORBIDDEN_EVENT_KINDS:
        problems.append(
            f"event.kind={kind!r} is reserved for Kibana's detection engine; ingestion pipelines "
            "must not emit it"
        )

    categories = event.get("category", [])
    if not isinstance(categories, list) or not categories:
        problems.append("event.category must be a non-empty list")
    else:
        for c in categories:
            if c not in EVENT_CATEGORIES:
                problems.append(f"event.category {c!r} not an ECS-allowed value")

    threat = record.get("threat")
    if threat:
        for path in ("tactic", "technique"):
            node = threat.get(path, {})
            if node and "id" not in node:
                problems.append(f"threat.{path}.id is required when threat.{path} is present")

    return problems


def assert_conformant(record: dict[str, Any]) -> None:
    problems = validate(record)
    if problems:
        raise ValueError("ECS conformance failed:\n  " + "\n  ".join(problems))
