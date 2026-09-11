"""Turning model scores into Alert objects.

This lives in `alerts/` rather than on the detector deliberately. `models/` and `eval/` must run
headless — the science has to be reproducible without the product layer — and CI enforces that by
failing if either imports `alerts/`, `api/` or `storage/`.

So the dependency runs one way: the detector does inference and returns numbers; this module turns
numbers into alerts. It was briefly the other way round, and the import-boundary check in CI is what
caught it.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from penumbra.alerts.models import Alert, Contribution, NetworkContext
from penumbra.alerts.scoring import ScoringPolicy, build_alert


def alerts_from_scores(
    detector: Any,
    X: pd.DataFrame,
    scored: pd.DataFrame | None = None,
    *,
    policy: ScoringPolicy | None = None,
    dataset: str = "unsw",
    entities: list[str] | None = None,
    include_benign: bool = False,
) -> list[Alert]:
    """Build Alert objects from a detector's scored output."""
    scored = scored if scored is not None else detector.score(X)
    policy = policy or getattr(detector, "policy", None) or ScoringPolicy()
    ranked = _ranked_importances(detector)
    version = getattr(getattr(detector, "metadata", None), "version", "0.1.0")

    out: list[Alert] = []
    for pos, (_, row) in enumerate(scored.iterrows()):
        fired = int(row["fired"])
        if fired == 0 and not include_benign:
            continue

        # SUSPECTED_NOVEL is precisely "novelty fired and supervised did not", so a novelty-only
        # detection carries no family - naming one would contradict the verdict.
        raw_family = row.get("family")
        family = raw_family if (fired in (1, 3) and isinstance(raw_family, str)) else None

        out.append(
            build_alert(
                p_attack=float(row["p_attack"]),
                novelty_percentile=float(row["novelty_percentile"]),
                policy=policy,
                family=family,
                dataset=dataset,
                agreement=int(row.get("agreement", 0)),
                network=_network_for(X, pos, entities),
                contributions=_contributions(X, pos, ranked),
                model_version=version,
            )
        )
    return out


def _ranked_importances(detector: Any) -> list[tuple[str, float]]:
    """Top global importances, asked of the detector once."""
    getter = getattr(detector, "ranked_importances", None)
    if callable(getter):
        try:
            return list(getter(8))
        except Exception:  # noqa: BLE001 - attribution is best-effort, never fatal to scoring
            return []
    return []


def _network_for(X: pd.DataFrame, pos: int, entities: list[str] | None) -> NetworkContext:
    row = X.iloc[pos]
    return NetworkContext(
        src_ip=entities[pos] if entities and pos < len(entities) else None,
        dst_port=_as_int(row.get("dst_port")),
        protocol=str(row.get("proto") or row.get("protocol_type") or "") or None,
        src_bytes=_as_int(row.get("sbytes") if "sbytes" in row.index else row.get("src_bytes")),
        dst_bytes=_as_int(row.get("dbytes") if "dbytes" in row.index else row.get("dst_bytes")),
        src_packets=_as_int(row.get("spkts")),
        dst_packets=_as_int(row.get("dpkts")),
        duration_ms=_as_float(row.get("dur") if "dur" in row.index else row.get("duration")),
    )


def _contributions(
    X: pd.DataFrame, pos: int, ranked: list[tuple[str, float]], top: int = 5
) -> list[Contribution]:
    """Top feature contributions, rendered in plain English.

    Global importances weighted per row rather than genuine per-row TreeSHAP. That is a real
    limitation and it is labelled rather than implied: the field is named `shap_value` for schema
    compatibility, but these are importance weights and the eval path is where exact attributions
    belong.
    """
    if not ranked:
        return []
    row = X.iloc[pos]
    return [
        Contribution(
            feature=name,
            value=_as_float(row.get(name)),
            shap_value=float(weight),
            direction="toward_attack",
            narrative=narrate(name, row.get(name)),
        )
        for name, weight in ranked[:top]
        if name in row.index
    ]


def _as_int(value: Any) -> int | None:
    try:
        if value is None or (isinstance(value, float) and not np.isfinite(value)):
            return None
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _as_float(value: Any) -> float | None:
    try:
        f = float(value)
        return f if np.isfinite(f) else None
    except (TypeError, ValueError):
        return None


# Plain-English renderers. "sload = 1400000" is not something a tier-1 analyst can act on;
# "outbound throughput 1.4 Mb/s" is. Covers both UNSW-NB15 and NSL-KDD naming, because a map that
# covers one dataset renders empty narratives on the other - which is exactly what happened.
NARRATIVES: dict[str, str] = {
    # --- UNSW-NB15
    "sttl": "source time-to-live {v} — a property of the sending host's OS, flagged as a testbed artifact",
    "dttl": "destination time-to-live {v} — same artifact family as sttl",
    "ct_dst_sport_ltm": "{v} recent connections to this destination port across hosts",
    "ct_src_dport_ltm": "{v} recent connections from this source across destination ports",
    "ct_srv_src": "{v} recent connections to the same service from this source",
    "ct_dst_ltm": "{v} recent connections to this destination",
    "ct_dst_src_ltm": "{v} recent connections between this source and destination",
    "ct_state_ttl": "unusual combination of connection state and time-to-live",
    "sbytes": "{v} bytes sent",
    "dbytes": "{v} bytes received",
    "spkts": "{v} packets sent",
    "dpkts": "{v} packets received",
    "rate": "{v} packets/second",
    "sload": "{v} bits/second outbound",
    "dload": "{v} bits/second inbound",
    "dur": "flow lasted {v} seconds",
    "smean": "mean outbound packet size {v} bytes",
    "dmean": "mean inbound packet size {v} bytes",
    "proto": "protocol {v}",
    "state": "connection state {v}",
    # --- NSL-KDD
    "src_bytes": "{v} bytes sent",
    "dst_bytes": "{v} bytes received",
    "duration": "connection lasted {v} seconds",
    "flag": "connection flag state {v}",
    "protocol_type": "protocol {v}",
    "service": "service {v}",
    "logged_in": "session was authenticated",
    "count": "{v} connections to the same host in the window",
    "srv_count": "{v} connections to the same service in the window",
    "serror_rate": "{v} of connections had SYN errors — the signature of a SYN flood",
    "srv_serror_rate": "{v} SYN-error rate on this service",
    "rerror_rate": "{v} rejected-connection rate — closed ports, consistent with scanning",
    "diff_srv_rate": "{v} of connections went to differing services — the shape of a port sweep",
    "same_srv_rate": "{v} of connections went to the same service",
    "dst_host_count": "{v} connections to this destination host in the window",
    "dst_host_srv_count": "{v} connections to this service on the destination host",
    "dst_host_diff_srv_rate": "{v} of connections to this host went to differing services",
    "dst_host_same_srv_rate": "{v} of connections to this host went to the same service",
    "dst_host_serror_rate": "{v} SYN-error rate against this host",
    "dst_host_srv_serror_rate": "{v} SYN-error rate for this service on this host",
    "num_failed_logins": "{v} failed authentication attempts",
    "root_shell": "a root shell was obtained",
    "su_attempted": "privilege escalation was attempted",
    "hot": "{v} accesses to sensitive system directories",
    "wrong_fragment": "{v} malformed fragments — consistent with evasion",
}


def narrate(feature: str, value: Any) -> str:
    template = NARRATIVES.get(feature)
    if template is None:
        return ""
    number = _as_float(value)
    if number is None:
        return ""
    rendered = f"{number:,.0f}" if abs(number) >= 100 else f"{number:,.3g}"
    return template.format(v=rendered)
