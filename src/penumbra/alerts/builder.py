"""Turning model scores into Alert objects.

This lives in `alerts/` rather than on the detector deliberately. `models/` and `eval/` must run
headless — the science has to be reproducible without the product layer — and CI enforces that by
failing if either imports `alerts/`, `api/` or `storage/`.

So the dependency runs one way: the detector does inference and returns numbers; this module turns
numbers into alerts. It was briefly the other way round, and the import-boundary check in CI is what
caught it.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
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
    return [
        alert
        for _, alert in alerts_with_positions(
            detector,
            X,
            scored,
            policy=policy,
            dataset=dataset,
            entities=entities,
            include_benign=include_benign,
        )
    ]


def alerts_with_positions(
    detector: Any,
    X: pd.DataFrame,
    scored: pd.DataFrame | None = None,
    *,
    policy: ScoringPolicy | None = None,
    dataset: str = "unsw",
    entities: list[str] | None = None,
    include_benign: bool = False,
) -> list[tuple[int, Alert]]:
    """As `alerts_from_scores`, paired with each alert's row position in `X`.

    Rows nobody flagged are dropped, so the n-th alert is not the n-th row. Anything that lines
    alerts back up with per-row data (timestamps, entities) needs the position, and indexing by the
    alert's place in the list silently attaches row 3's timestamp to row 40's alert.
    """
    # pandas deep-copies `attrs` on every row access and on most operations. The CICIDS loader keeps
    # a ~1.2M-row entity frame there, so each `X.iloc[pos]` copied the whole thing: a 240k-flow
    # replay ran 30+ minutes pinned in `DataFrame.__deepcopy__`. Even `X.copy(deep=False)` copies
    # attrs, so they are detached in place for the duration and restored afterwards. Entities arrive
    # as an argument; nothing here needs the metadata.
    saved = dict(X.attrs)
    X.attrs.clear()
    try:
        return _build_positioned(
            detector,
            X,
            scored,
            policy=policy,
            dataset=dataset,
            entities=entities,
            include_benign=include_benign,
        )
    finally:
        X.attrs.update(saved)


def _build_positioned(
    detector: Any,
    X: pd.DataFrame,
    scored: pd.DataFrame | None,
    *,
    policy: ScoringPolicy | None,
    dataset: str,
    entities: list[str] | None,
    include_benign: bool,
) -> list[tuple[int, Alert]]:
    scored = scored if scored is not None else detector.score(X)
    policy = policy or getattr(detector, "policy", None) or ScoringPolicy()
    ranked = _ranked_importances(detector)
    version = getattr(getattr(detector, "metadata", None), "version", "0.1.0")

    n = len(scored)
    fired = scored["fired"].to_numpy(dtype=int)
    abstains = (
        scored["conformal_abstains"].to_numpy(dtype=bool)
        if "conformal_abstains" in scored
        else np.zeros(n, dtype=bool)
    )
    # A row neither head flagged still reaches an analyst if the model declined to commit on it.
    # That is the point of the abstention lane: "I don't know" is a reportable answer.
    keep = np.arange(n) if include_benign else np.flatnonzero((fired != 0) | abstains)
    if not len(keep):
        return []

    # Column arrays and one pass over the kept rows. Per-row `iterrows` plus three `X.iloc[pos]`
    # lookups per alert made building 20k alerts take 16 s, several times longer than scoring them.
    p_attack = scored["p_attack"].to_numpy(dtype=float)
    novelty = scored["novelty_percentile"].to_numpy(dtype=float)
    agreement = scored["agreement"].to_numpy(dtype=int) if "agreement" in scored else np.zeros(n, dtype=int)
    families = scored["family"].tolist() if "family" in scored else [None] * n
    sets = scored["conformal_set"].tolist() if "conformal_set" in scored else [None] * n
    rows = X.iloc[keep].to_dict("records")

    out: list[tuple[int, Alert]] = []
    for pos, row in zip(keep.tolist(), rows, strict=True):
        f = int(fired[pos])
        # SUSPECTED_NOVEL is precisely "novelty fired and supervised did not", so a novelty-only
        # detection carries no family - naming one would contradict the verdict.
        raw_family = families[pos]
        family = raw_family if (f in (1, 3) and isinstance(raw_family, str)) else None

        alert = build_alert(
            p_attack=float(p_attack[pos]),
            novelty_percentile=float(novelty[pos]),
            policy=policy,
            family=family,
            dataset=dataset,
            agreement=int(agreement[pos]),
            conformal_ambiguous=bool(abstains[pos]),
            conformal_set=list(sets[pos] or []),
            network=_network_for(row, entities[pos] if entities and pos < len(entities) else None),
            contributions=_contributions(row, ranked),
            model_version=version,
        )
        # The full feature row travels with the alert. An analyst verdict is only a training
        # example if the features it labels can be recovered later, and the alert is the one
        # record that survives from scoring to promotion.
        alert.raw_features = feature_payload(row)
        out.append((pos, alert))
    return out


def feature_payload(row: Mapping[str, Any]) -> dict[str, Any]:
    """A feature row as JSON-safe scalars. Non-finite numbers become None."""
    return {str(name): _json_scalar(value) for name, value in row.items()}


def _json_scalar(value: Any) -> Any:
    # Exact-type checks first: `to_dict("records")` hands back plain Python scalars, and these cover
    # nearly every value. The isinstance chain below is the general case (numpy scalars, NA).
    kind = type(value)
    if kind is float:
        return value if math.isfinite(value) else None
    if kind is int or kind is bool:
        return value
    if kind is str:
        return value
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if value is None:
        return None
    return str(value)


def _ranked_importances(detector: Any) -> list[tuple[str, float]]:
    """Top global importances, asked of the detector once."""
    getter = getattr(detector, "ranked_importances", None)
    if callable(getter):
        try:
            return list(getter(8))
        except Exception:  # noqa: BLE001 - attribution is best-effort, never fatal to scoring
            return []
    return []


def _network_for(row: Mapping[str, Any], src_ip: str | None) -> NetworkContext:
    return NetworkContext(
        src_ip=src_ip,
        dst_port=_as_int(row.get("dst_port")),
        protocol=str(row.get("proto") or row.get("protocol_type") or "") or None,
        src_bytes=_as_int(row.get("sbytes") if "sbytes" in row else row.get("src_bytes")),
        dst_bytes=_as_int(row.get("dbytes") if "dbytes" in row else row.get("dst_bytes")),
        src_packets=_as_int(row.get("spkts")),
        dst_packets=_as_int(row.get("dpkts")),
        duration_ms=_as_float(row.get("dur") if "dur" in row else row.get("duration")),
    )


def _contributions(
    row: Mapping[str, Any], ranked: list[tuple[str, float]], top: int = 5
) -> list[Contribution]:
    """Top feature contributions, rendered in plain English.

    Global importances weighted per row rather than genuine per-row TreeSHAP. That is a real
    limitation and it is labelled rather than implied: the field is named `shap_value` for schema
    compatibility, but these are importance weights and the eval path is where exact attributions
    belong.
    """
    return [
        Contribution(
            feature=name,
            value=_as_float(row.get(name)),
            shap_value=float(weight),
            direction="toward_attack",
            narrative=narrate(name, row.get(name)),
        )
        for name, weight in ranked[:top]
        if name in row
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
        return f if math.isfinite(f) else None
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
