"""Integrity checks on analyst verdicts, before any of them can train a model.

The feedback loop is a poisoning vector we built ourselves (THREAT_MODEL T1): an attacker holding an
analyst account marks their own traffic `false_positive`, the label is promoted, and the next model
learns the attack is benign. Promotion by a second senior account is the first control. These checks
are the second - they tell that approver *which* verdicts deserve a second look.

Three signals, each cheap and each explainable to the person reading it:

  confident_contradiction  the verdict clears a detection the model scored >= 0.90. Sometimes the
                           analyst is right and the model is wrong; that is exactly the case a
                           second pair of eyes should confirm.
  actor_outlier            one account clears detections far more often than its peers, beyond
                           what binomial noise explains.
  family_campaign          one account's clearances concentrate on a single attack family - the
                           shape of whitewashing one technique rather than triaging a queue.

None of these blocks anything. They are flags on a review screen, and the approver decides. A flag
that silently dropped verdicts would be a second, unaudited decision-maker.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

CLEARING_VERDICTS = frozenset({"false_positive", "benign_by_policy"})

CONFIDENT = 0.90
OUTLIER_Z = 3.0
MIN_ACTOR_VERDICTS = 10
CAMPAIGN_SHARE = 0.80
MIN_CAMPAIGN_CLEARANCES = 10


@dataclass
class ActorProfile:
    actor: str
    n_verdicts: int
    n_cleared: int
    clear_rate: float
    peer_clear_rate: float
    z: float
    top_family: str | None
    top_family_share: float
    flags: list[str] = field(default_factory=list)


def _clears(v: Mapping[str, Any]) -> bool:
    return str(v.get("verdict")) in CLEARING_VERDICTS


def actor_profiles(history: Iterable[Mapping[str, Any]]) -> dict[str, ActorProfile]:
    """Per-account clearance behaviour against the pooled rate of everyone else.

    Leave-one-out pooling: comparing an actor to a pool that includes themselves dilutes exactly the
    signal being looked for, and with three analysts it dilutes it a lot.
    """
    rows = list(history)
    by_actor: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for r in rows:
        by_actor[str(r.get("actor"))].append(r)

    total_n = len(rows)
    total_cleared = sum(_clears(r) for r in rows)

    out: dict[str, ActorProfile] = {}
    for actor, mine in by_actor.items():
        n = len(mine)
        cleared = sum(_clears(r) for r in mine)
        peers_n = total_n - n
        peer_rate = (total_cleared - cleared) / peers_n if peers_n else float("nan")
        rate = cleared / n if n else 0.0

        z = 0.0
        if peers_n and 0.0 < peer_rate < 1.0 and n:
            z = (rate - peer_rate) / math.sqrt(peer_rate * (1.0 - peer_rate) / n)

        families = Counter(str(r.get("family")) for r in mine if _clears(r) and r.get("family"))
        top, top_n = families.most_common(1)[0] if families else (None, 0)
        share = top_n / cleared if cleared else 0.0

        profile = ActorProfile(
            actor=actor,
            n_verdicts=n,
            n_cleared=cleared,
            clear_rate=rate,
            peer_clear_rate=peer_rate,
            z=z,
            top_family=top,
            top_family_share=share,
        )
        if n >= MIN_ACTOR_VERDICTS and z >= OUTLIER_Z:
            profile.flags.append("actor_outlier")
        if top_n >= MIN_CAMPAIGN_CLEARANCES and share >= CAMPAIGN_SHARE:
            profile.flags.append("family_campaign")
        out[actor] = profile
    return out


def flag_verdicts(
    pending: Iterable[Mapping[str, Any]], history: Iterable[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """Annotate each pending verdict with the reasons it deserves a second look."""
    profiles = actor_profiles(history)
    out: list[dict[str, Any]] = []
    for v in pending:
        flags: list[str] = []
        p = v.get("p_attack")
        if _clears(v) and isinstance(p, (int, float)) and p >= CONFIDENT:
            flags.append("confident_contradiction")
        profile = profiles.get(str(v.get("actor")))
        if profile is not None and _clears(v):
            flags.extend(profile.flags)
            if "family_campaign" in flags and v.get("family") != profile.top_family:
                flags.remove("family_campaign")
        out.append({**v, "flags": flags})
    return out


def explain(flag: str) -> str:
    return {
        "confident_contradiction": f"clears a detection the model scored >= {CONFIDENT:.2f}",
        "actor_outlier": f"this account clears detections at a rate >= {OUTLIER_Z:.0f} SE above its peers",
        "family_campaign": (
            f"over {CAMPAIGN_SHARE:.0%} of this account's clearances target one attack family"
        ),
    }.get(flag, flag)
