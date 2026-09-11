"""Alerts into incidents.

The anti-alert-fatigue mechanism, and the one number in this project most likely to be quoted: a
port scan producing four thousand flows should reach an analyst as one incident with four thousand
events, not four thousand tickets.

**This requires real source identities.** UNSW-NB15's published split has no IP addresses and no
timestamps (ADR-0004), so a compression ratio computed from identifiers we invented for display
would be an invented number. `correlate` therefore refuses to run without real entities, and
`Correlator.requires_entities` says so rather than silently grouping on something meaningless.

CICIDS2017 carries `Source IP` and `Timestamp`, which is why the measured compression figure comes
from there.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from penumbra.alerts.models import Alert, Incident, Lane, Severity, Verdict

SEVERITY_ORDER = [Severity.INFORMATIONAL, Severity.LOW, Severity.MEDIUM, Severity.HIGH]


class EntitiesRequired(ValueError):
    """Raised when correlation is attempted on data with no real source identities."""

    def __init__(self, dataset: str) -> None:
        super().__init__(
            f"{dataset} carries no source identities, so alerts cannot be grouped by entity. "
            "Any compression ratio computed here would be derived from synthesised identifiers. "
            "Correlation runs on CICIDS2017; see docs/adr/0004-dataset-roles.md."
        )


@dataclass
class CorrelationPolicy:
    """How alerts are grouped.

    Grouping by (entity, family) rather than by entity alone is deliberate: one host running a port
    scan AND exfiltrating data is two incidents with different responses, and merging them hides the
    second behind the first.
    """

    window: timedelta = timedelta(minutes=15)
    group_by_family: bool = True
    # Below this an "incident" is a single alert wearing a hat, and the extra object costs the
    # analyst a click for nothing.
    min_alerts: int = 1


@dataclass
class CorrelationResult:
    incidents: list[Incident] = field(default_factory=list)
    n_alerts: int = 0
    unassigned: list[Alert] = field(default_factory=list)

    @property
    def compression_ratio(self) -> float:
        """Alerts per incident. The headline anti-alert-fatigue number."""
        return self.n_alerts / len(self.incidents) if self.incidents else 0.0

    def summary(self) -> str:
        lines = [
            f"  {self.n_alerts:,} alerts -> {len(self.incidents):,} incidents "
            f"({self.compression_ratio:.1f} events per incident)",
        ]
        if self.incidents:
            biggest = max(self.incidents, key=lambda i: i.event_count)
            lines.append(
                f"  largest: {biggest.title} - {biggest.event_count:,} events, "
                f"{biggest.distinct_destinations:,} destinations, {biggest.distinct_ports:,} ports"
            )
        return "\n".join(lines)


class Correlator:
    """Groups alerts into incidents by entity, family and time window."""

    def __init__(self, policy: CorrelationPolicy | None = None) -> None:
        self.policy = policy or CorrelationPolicy()

    @staticmethod
    def requires_entities(alerts: list[Alert]) -> bool:
        """True when the alerts carry real source identities."""
        return any(a.network.src_ip for a in alerts)

    def correlate(self, alerts: list[Alert], *, dataset: str = "unknown") -> CorrelationResult:
        actionable = [a for a in alerts if a.is_actionable_alert]
        if not actionable:
            return CorrelationResult(n_alerts=0)

        if not self.requires_entities(actionable):
            raise EntitiesRequired(dataset)

        buckets: dict[tuple[str, str | None], list[Alert]] = defaultdict(list)
        unassigned: list[Alert] = []

        for alert in sorted(actionable, key=lambda a: a.timestamp):
            entity = alert.network.src_ip
            if not entity:
                unassigned.append(alert)
                continue
            family = alert.family if self.policy.group_by_family else None
            buckets[(entity, family)].append(alert)

        incidents: list[Incident] = []
        for (entity, family), group in buckets.items():
            incidents.extend(self._split_by_window(entity, family, group))

        incidents = [i for i in incidents if i.event_count >= self.policy.min_alerts]
        # Priority first, then size. Two incidents at equal priority are not equally urgent when
        # one represents 29,562 flows and the other represents one.
        incidents.sort(key=lambda i: (i.priority, i.event_count), reverse=True)

        return CorrelationResult(incidents=incidents, n_alerts=len(actionable), unassigned=unassigned)

    def _split_by_window(self, entity: str, family: str | None, alerts: list[Alert]) -> list[Incident]:
        """Break a bucket wherever the gap exceeds the window.

        Without this, a host scanned on Monday and again on Friday becomes one incident spanning
        four days, which is not a thing an analyst can act on.
        """
        out: list[Incident] = []
        current: list[Alert] = []

        for alert in alerts:
            if current and (alert.timestamp - current[-1].timestamp) > self.policy.window:
                out.append(self._build(entity, family, current))
                current = []
            current.append(alert)
        if current:
            out.append(self._build(entity, family, current))
        return out

    def _build(self, entity: str, family: str | None, group: list[Alert]) -> Incident:
        severity = max(
            (a.severity for a in group), key=lambda s: SEVERITY_ORDER.index(s), default=Severity.LOW
        )
        priority = max(a.priority for a in group)
        destinations = {a.network.dst_ip for a in group if a.network.dst_ip}
        ports = {a.network.dst_port for a in group if a.network.dst_port is not None}

        novel = sum(1 for a in group if a.verdict is Verdict.SUSPECTED_NOVEL)
        lane = Lane.HUNTING if novel > len(group) / 2 else Lane.KNOWN_THREAT

        incident = Incident(
            title=self._title(entity, family, group, len(destinations), len(ports)),
            severity=severity,
            lane=lane,
            priority=priority,
            entity=entity,
            family=family,
            attack=next((a.attack for a in group if a.attack), None),
            alert_ids=[a.alert_id for a in group],
            event_count=len(group),
            distinct_destinations=len(destinations),
            distinct_ports=len(ports),
            opened_at=min(a.timestamp for a in group),
            last_seen_at=max(a.timestamp for a in group),
        )
        for alert in group:
            alert.incident_id = incident.incident_id
        return incident

    @staticmethod
    def _title(entity: str, family: str | None, group: list[Alert], n_dest: int, n_ports: int) -> str:
        if family is None:
            return f"Unrecognised traffic from {entity} ({len(group):,} flows)"
        # Fan-out is what makes a scan a scan, so it goes in the title where an analyst sees it
        # without opening anything.
        if n_dest > 10 or n_ports > 20:
            return f"{family} from {entity} - {n_dest:,} hosts, {n_ports:,} ports"
        return f"{family} from {entity} ({len(group):,} flows)"


def fan_out_features(alerts: list[Alert]) -> dict[str, dict[str, int]]:
    """Per-entity fan-out, the topological signature of scanning.

    Contacting 254 hosts on one port is a sweep; contacting one host on 1,024 ports is a port scan.
    Neither is visible in a single flow record, which is why per-flow models miss both and why these
    features exist.
    """
    stats: dict[str, dict[str, set[object]]] = defaultdict(
        lambda: {"destinations": set(), "ports": set(), "families": set()}
    )
    for alert in alerts:
        entity = alert.network.src_ip
        if not entity:
            continue
        if alert.network.dst_ip:
            stats[entity]["destinations"].add(alert.network.dst_ip)
        if alert.network.dst_port is not None:
            stats[entity]["ports"].add(alert.network.dst_port)
        if alert.family:
            stats[entity]["families"].add(alert.family)

    return {
        entity: {
            "distinct_destinations": len(v["destinations"]),
            "distinct_ports": len(v["ports"]),
            "distinct_families": len(v["families"]),
        }
        for entity, v in stats.items()
    }


def now() -> datetime:
    return datetime.now(UTC)
