"""The canary gate: a challenger must not be worse than the champion on frozen, trusted labels.

Retraining on analyst feedback is how the system improves and also how it is poisoned (THREAT_MODEL
T1). A model taught to ignore one attack does not get worse on average - the other thousands of rows
are unchanged - so an aggregate metric waves it through. The gate therefore checks **every family
separately**, because a targeted poisoning attack is a per-family regression by construction.

Three criteria, each compared at the model's OWN operating point (what would actually alert), not at
an AUC nobody deploys:

  G1  overall attack recall does not fall by more than `max_overall_drop`
  G2  no family with enough canary support loses more than `max_family_drop` recall
  G3  benign false-positive rate does not rise by more than `max_fpr_rise`

The canary set is frozen and never trained on. In production it is a curated, access-controlled
labelled set; here it is a fixed, seeded slice of the dataset's test split (`live_split`), and the
numbers reported alongside a gate decision come from a DIFFERENT slice, so no reported number was
measured on rows that chose the model.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from penumbra.seeds import SEED

MIN_FAMILY_SUPPORT = 20


@dataclass(frozen=True)
class GatePolicy:
    max_overall_drop: float = 0.02
    max_family_drop: float = 0.10
    max_fpr_rise: float = 0.01
    min_family_support: int = MIN_FAMILY_SUPPORT


@dataclass
class OperatingReport:
    """What a detector does on a labelled set, at the threshold it would deploy with."""

    n_attack: int
    n_benign: int
    recall: float
    fpr: float
    per_family: dict[str, dict[str, float]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GateResult:
    passed: bool
    reasons: list[str]
    champion: OperatingReport
    challenger: OperatingReport
    policy: GatePolicy
    family_deltas: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "reasons": self.reasons,
            "policy": asdict(self.policy),
            "champion": self.champion.to_dict(),
            "challenger": self.challenger.to_dict(),
            "family_deltas": self.family_deltas,
        }


def operating_report(fired: np.ndarray, y: np.ndarray, families: np.ndarray) -> OperatingReport:
    fired = np.asarray(fired).astype(bool)
    y = np.asarray(y).astype(int)
    families = np.asarray(families).astype(str)
    attack, benign = y == 1, y == 0

    per_family: dict[str, dict[str, float]] = {}
    for fam in sorted({str(f) for f in families[attack]}):
        mask = attack & (families == fam)
        per_family[fam] = {"n": int(mask.sum()), "recall": float(fired[mask].mean())}

    return OperatingReport(
        n_attack=int(attack.sum()),
        n_benign=int(benign.sum()),
        recall=float(fired[attack].mean()) if attack.any() else float("nan"),
        fpr=float(fired[benign].mean()) if benign.any() else float("nan"),
        per_family=per_family,
    )


def fired(detector: Any, X: pd.DataFrame) -> np.ndarray:
    """Rows the detector would alert on: either head fired."""
    return np.asarray(detector.score(X)["fired"].to_numpy() > 0)


def evaluate(detector: Any, X: pd.DataFrame, y: pd.Series, families: pd.Series) -> OperatingReport:
    return operating_report(fired(detector, X), y.to_numpy(), families.to_numpy())


def gate(
    champion: OperatingReport, challenger: OperatingReport, policy: GatePolicy | None = None
) -> GateResult:
    policy = policy or GatePolicy()
    reasons: list[str] = []

    if challenger.recall < champion.recall - policy.max_overall_drop:
        reasons.append(
            f"G1 overall recall {champion.recall:.4f} -> {challenger.recall:.4f} "
            f"(allowed drop {policy.max_overall_drop})"
        )

    deltas: dict[str, float] = {}
    for fam, champ in champion.per_family.items():
        if champ["n"] < policy.min_family_support or fam not in challenger.per_family:
            continue
        delta = challenger.per_family[fam]["recall"] - champ["recall"]
        deltas[fam] = delta
        if delta < -policy.max_family_drop:
            reasons.append(
                f"G2 family {fam!r} recall {champ['recall']:.4f} -> "
                f"{challenger.per_family[fam]['recall']:.4f} on {int(champ['n'])} canary rows "
                f"(allowed drop {policy.max_family_drop})"
            )

    if challenger.fpr > champion.fpr + policy.max_fpr_rise:
        reasons.append(
            f"G3 benign FPR {champion.fpr:.4f} -> {challenger.fpr:.4f} (allowed rise {policy.max_fpr_rise})"
        )

    return GateResult(
        passed=not reasons,
        reasons=reasons,
        champion=champion,
        challenger=challenger,
        policy=policy,
        family_deltas=deltas,
    )


def live_split(
    labels: pd.Series, fractions: tuple[float, float, float] = (0.30, 0.35, 0.35), seed: int = SEED
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Deterministically split rows into (canary, feedback, evaluation), stratified by label.

    Stratified by hand rather than with sklearn because NSL-KDD's test split has attack types with
    one or two rows, and a library stratifier refuses those outright. Here a rare label simply lands
    wherever the seeded shuffle puts it. Positional indices are returned.
    """
    if abs(sum(fractions) - 1.0) > 1e-9:
        raise ValueError("fractions must sum to 1")
    rng = np.random.default_rng(seed)
    values = labels.astype(str).to_numpy()
    parts: list[list[int]] = [[], [], []]
    for label in sorted(set(values)):
        idx = np.flatnonzero(values == label)
        rng.shuffle(idx)
        cut1 = int(round(len(idx) * fractions[0]))
        cut2 = cut1 + int(round(len(idx) * fractions[1]))
        parts[0].extend(idx[:cut1])
        parts[1].extend(idx[cut1:cut2])
        parts[2].extend(idx[cut2:])
    return tuple(np.sort(np.asarray(p, dtype=int)) for p in parts)  # type: ignore[return-value]
