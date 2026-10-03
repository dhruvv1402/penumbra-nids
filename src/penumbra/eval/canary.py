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

## ADR-0005: G1 and G2 at a matched false-positive rate

Comparing recall at each model's own threshold is only fair when both realise about the same FPR.
A champion that fires on 18.5% of benign traffic buys recall with those false positives, and an
honestly calibrated challenger then fails G1 while being better at every matched FPR. So
`gate_detectors` reads the challenger's recall with EACH HEAD placed at the champion's realised
canary FPR for that head (revision 1: matching only the total moved the head mix too - the UNSW
champion spends 18.6 of its 18.8 points on the supervised head, and an equal split starved the
challenger's), keeps G3 at the challenger's own threshold, and reports G4 - the challenger's
realised FPR against its target - as advisory. The verdicts at own thresholds and at the original
total-FPR match are recorded beside it. `gate` itself is unchanged: E7 measured it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from penumbra.models.fusion import OrGate, benign_threshold
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
    # ADR-0005. Left at their defaults by `gate`, which compares at own thresholds only.
    kind: str = "own_threshold"
    challenger_matched: OperatingReport | None = None
    at_own_threshold: dict[str, Any] | None = None
    advisory: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "kind": self.kind,
            "passed": self.passed,
            "reasons": self.reasons,
            "policy": asdict(self.policy),
            "champion": self.champion.to_dict(),
            "challenger": self.challenger.to_dict(),
            "family_deltas": self.family_deltas,
        }
        if self.challenger_matched is not None:
            out["challenger_at_matched_fpr"] = self.challenger_matched.to_dict()
        if self.at_own_threshold is not None:
            out["at_own_threshold"] = self.at_own_threshold
        if self.advisory is not None:
            out["advisory"] = self.advisory
        return out


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
    """G1-G3, each at the model's own threshold. The gate E7 measured; see `gate_detectors`."""
    policy = policy or GatePolicy()
    reasons, deltas = _recall_checks(champion, challenger, policy)
    reasons += _fpr_check(champion, challenger, policy)
    return GateResult(
        passed=not reasons,
        reasons=reasons,
        champion=champion,
        challenger=challenger,
        policy=policy,
        family_deltas=deltas,
    )


def _recall_checks(
    champion: OperatingReport, challenger: OperatingReport, policy: GatePolicy
) -> tuple[list[str], dict[str, float]]:
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

    return reasons, deltas


def _fpr_check(champion: OperatingReport, challenger: OperatingReport, policy: GatePolicy) -> list[str]:
    if challenger.fpr > champion.fpr + policy.max_fpr_rise:
        return [
            f"G3 benign FPR {champion.fpr:.4f} -> {challenger.fpr:.4f} (allowed rise {policy.max_fpr_rise})"
        ]
    return []


def binomial_interval(k: int, n: int, confidence: float = 0.99) -> tuple[float, float]:
    """Clopper-Pearson interval for a rate of k in n."""
    if n == 0:
        return float("nan"), float("nan")
    a = 1.0 - confidence
    lo = 0.0 if k == 0 else float(stats.beta.ppf(a / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(stats.beta.ppf(1 - a / 2, k + 1, n - k))
    return lo, hi


def fired_at_fpr(scored: pd.DataFrame, y: np.ndarray, total_fpr: float) -> np.ndarray:
    """Rows a detector fires on with both thresholds refitted on these benign rows at `total_fpr`.

    The same per-head split the detector itself uses (`OrGate.fit`), so the comparison moves only
    the operating point, never how the two heads are combined.
    """
    p = scored["p_attack"].to_numpy(dtype=float)
    n = scored["novelty_percentile"].to_numpy(dtype=float)
    benign = np.asarray(y).astype(int) == 0
    refit = OrGate.fit(p[benign], n[benign], total_fpr=float(total_fpr), use_novelty=True)
    return np.asarray(refit.flags(p, n))


def head_fprs(scored: pd.DataFrame, y: np.ndarray) -> tuple[float, float]:
    """(supervised, novelty) realised FPR on the benign rows: how a detector spends its budget."""
    fired = scored["fired"].to_numpy()
    benign = np.asarray(y).astype(int) == 0
    if not benign.any():
        return float("nan"), float("nan")
    return float(np.isin(fired[benign], (1, 3)).mean()), float(np.isin(fired[benign], (2, 3)).mean())


def fired_at_head_fprs(
    scored: pd.DataFrame, y: np.ndarray, supervised_fpr: float, novelty_fpr: float
) -> np.ndarray:
    """Rows a detector fires on with each head placed at its own benign rate on these rows.

    A head the reference never fired stays off: matching a 0% head means not using it.
    """
    p = scored["p_attack"].to_numpy(dtype=float)
    n = scored["novelty_percentile"].to_numpy(dtype=float)
    benign = np.asarray(y).astype(int) == 0
    t_sup = benign_threshold(p[benign], supervised_fpr) if supervised_fpr > 0 else float("inf")
    t_nov = benign_threshold(n[benign], novelty_fpr) if novelty_fpr > 0 else float("inf")
    return np.asarray((p >= t_sup) | (n >= t_nov))


def _own_columns(detector: Any, X: pd.DataFrame) -> pd.DataFrame:
    names = getattr(detector, "_feature_names", None)
    return X[list(names)] if names else X


def gate_detectors(
    champion: Any,
    challenger: Any,
    X: pd.DataFrame,
    y: pd.Series,
    families: pd.Series,
    policy: GatePolicy | None = None,
) -> GateResult:
    """The promotion gate (ADR-0005): G1/G2 at the champion's realised FPR, G3 at own thresholds."""
    policy = policy or GatePolicy()
    y_arr, f_arr = y.to_numpy(), families.to_numpy()
    # Each detector reads its own columns: a challenger trained with quarantined features is
    # scored on the same canary rows as a champion that kept them.
    champ_scored, chall_scored = (
        champion.score(_own_columns(champion, X)),
        challenger.score(_own_columns(challenger, X)),
    )
    champ = operating_report(champ_scored["fired"].to_numpy() > 0, y_arr, f_arr)
    chall = operating_report(chall_scored["fired"].to_numpy() > 0, y_arr, f_arr)
    own = gate(champ, chall, policy)

    sup_fpr, nov_fpr = head_fprs(champ_scored, y_arr)
    total_match: dict[str, Any] | None = None
    if np.isfinite(champ.fpr) and 0.0 < champ.fpr < 1.0:
        matched = operating_report(fired_at_head_fprs(chall_scored, y_arr, sup_fpr, nov_fpr), y_arr, f_arr)
        # The original ADR-0005 rule, kept on the record: both heads at an equal split of the total.
        total = operating_report(fired_at_fpr(chall_scored, y_arr, champ.fpr), y_arr, f_arr)
        t_reasons, _ = _recall_checks(champ, total, policy)
        t_reasons += _fpr_check(champ, chall, policy)
        total_match = {
            "passed": not t_reasons,
            "reasons": t_reasons,
            "recall": total.recall,
            "fpr": total.fpr,
        }
    else:
        # Nothing to match against (a champion that never fires, or always does): own thresholds.
        matched = chall
    reasons, deltas = _recall_checks(champ, matched, policy)
    reasons += _fpr_check(champ, chall, policy)

    k = int(round(chall.fpr * chall.n_benign)) if chall.n_benign else 0
    lo, hi = binomial_interval(k, chall.n_benign)
    target = float(getattr(challenger, "target_fpr", float("nan")))
    advisory = {
        "G4_target_fpr": target,
        "G4_realised_fpr": chall.fpr,
        "G4_interval_99": [lo, hi],
        "G4_consistent_with_target": bool(lo <= target <= hi) if np.isfinite(target) else None,
        "champion_realised_fpr": champ.fpr,
        "champion_head_fprs": {"supervised": sup_fpr, "novelty": nov_fpr},
        "matched_fpr": matched.fpr,
        "at_matched_total_fpr": total_match,
    }
    return GateResult(
        passed=not reasons,
        reasons=reasons,
        champion=champ,
        challenger=chall,
        policy=policy,
        family_deltas=deltas,
        kind="matched_fpr_per_head",
        challenger_matched=matched,
        at_own_threshold={"passed": own.passed, "reasons": own.reasons},
        advisory=advisory,
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
