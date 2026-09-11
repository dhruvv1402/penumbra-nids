"""Constrained evasion — how much effort does it take to slip past the detector?

Most published adversarial-robustness numbers for network IDS are meaningless, and the reason is
mechanical. They perturb the *feature vector* inside an ε-ball: nudge `spkts` from 3 to 3.2, drop
`sbytes` by 40, set `sload` independently of `sbytes` and `dur`. None of those rows can exist. A
packet count of 3.2 is not a thing that happens on a wire, an attacker cannot un-send bytes their
own exploit needs, and `sload` is *defined* as `sbytes * 8 / dur` — setting it independently
describes no flow at all.

So the attack succeeds against a model evaluated on rows that could never be transmitted, and the
resulting "95% evasion rate" measures the geometry of the feature space rather than the security of
the system.

Pierazzi et al. (IEEE S&P 2020) call the alternative **problem-space** attacks: perturb only what
an attacker actually controls, in directions they can actually move, and recompute everything
downstream. That is what this module does.

## The manipulation space

| an attacker CAN | an attacker CANNOT |
|---|---|
| add padding bytes | un-send bytes the exploit requires |
| send more packets | reduce their own packet count |
| stretch a flow's duration | make a flow shorter than its content |
| widen inter-packet gaps | make a scan contact fewer hosts and still be a scan |

Three invariants are enforced on every perturbed row, and each is a test:

1. **Monotone.** Every primitive moves in the attacker-feasible direction only.
2. **Integral.** Packet counts are whole numbers.
3. **Derived features are recomputed, never perturbed.** `rate`, `sload`, `dload`, `smean`, `dmean`
   are functions of `dur`, `sbytes`, `dbytes`, `spkts`, `dpkts`. Perturbing them independently is
   how a row leaves the manifold, and a detection missed for that reason was not evaded.

## What gets reported

A **detection-versus-effort curve**, not a single evasion rate. Effort is the attacker's cost —
stretching a DoS flow ten-fold makes it ten times slower, which is a real price in attack
throughput. "Detection falls from 0.97 to 0.61 when the attacker accepts a 10x slowdown" is a
statement a defender can act on. "95% evasion" is not.

Both attacks are run: the constrained one, and the unconstrained feature-space strawman. The gap
between them is the finding.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ManipulationSpace:
    """What the attacker controls on this dataset, and what follows from it.

    `grow_only` features may only increase: padding is always available, un-sending is not.
    `integral` features must stay whole. `derived` is recomputed from primitives after every
    perturbation and never touched directly.
    """

    grow_only: tuple[str, ...]
    integral: tuple[str, ...]
    duration: str | None
    derived: tuple[str, ...]

    def available(self, columns: set[str]) -> tuple[str, ...]:
        return tuple(c for c in self.grow_only if c in columns)


# UNSW-NB15. `rate`, `sload`, `dload`, `smean`, `dmean` are arithmetic consequences of the rest.
UNSW_SPACE = ManipulationSpace(
    grow_only=("sbytes", "dbytes", "spkts", "dpkts"),
    integral=("spkts", "dpkts", "sbytes", "dbytes"),
    duration="dur",
    derived=("rate", "sload", "dload", "smean", "dmean"),
)

# NSL-KDD. Its connection-window rates (`serror_rate`, `count`, `srv_count`, ...) are aggregates
# over OTHER connections, so a single flow's attacker cannot set them directly at all - they are
# neither manipulable nor recomputable here, and pretending otherwise would overstate the attack.
NSLKDD_SPACE = ManipulationSpace(
    grow_only=("src_bytes", "dst_bytes"),
    integral=("src_bytes", "dst_bytes"),
    duration="duration",
    derived=(),
)

SPACES: dict[str, ManipulationSpace] = {"unsw": UNSW_SPACE, "nslkdd": NSLKDD_SPACE}


def space_for(dataset: str) -> ManipulationSpace:
    key = dataset.lower().replace("-", "").replace("_", "")
    for name, space in SPACES.items():
        if key.startswith(name):
            return space
    raise ValueError(f"no manipulation space defined for {dataset!r}")


# =================================================================================================
# The attacks
# =================================================================================================


def _recompute(df: pd.DataFrame, space: ManipulationSpace) -> None:
    """Restore the arithmetic identities the perturbation just broke.

    Every one of these is a definition, not a model: `sload` IS `sbytes * 8 / dur`. Leaving a
    stretched flow with its original `sload` produces a row that claims to have moved 400 Mbit in
    40 seconds through 12 packets of 200 bytes, which no network did.
    """
    if space.duration is None or space.duration not in df.columns:
        return
    dur = df[space.duration].to_numpy(dtype=float)
    safe = np.where(dur > 0, dur, np.nan)

    packets = [c for c in ("spkts", "dpkts") if c in df.columns]
    if packets and "rate" in df.columns:
        total = sum(df[c].to_numpy(dtype=float) for c in packets)
        df["rate"] = np.nan_to_num(total / safe, nan=0.0, posinf=0.0)

    for byte_col, load_col in (("sbytes", "sload"), ("dbytes", "dload")):
        if byte_col in df.columns and load_col in df.columns:
            df[load_col] = np.nan_to_num(df[byte_col].to_numpy(dtype=float) * 8.0 / safe, nan=0.0, posinf=0.0)

    for byte_col, pkt_col, mean_col in (("sbytes", "spkts", "smean"), ("dbytes", "dpkts", "dmean")):
        if {byte_col, pkt_col, mean_col} <= set(df.columns):
            p = df[pkt_col].to_numpy(dtype=float)
            df[mean_col] = np.nan_to_num(
                df[byte_col].to_numpy(dtype=float) / np.where(p > 0, p, np.nan), nan=0.0
            )


def problem_space_attack(
    X: pd.DataFrame, effort: float, space: ManipulationSpace, *, seed: int = 0
) -> pd.DataFrame:
    """Slow-rate mimicry plus padding, within what an attacker can actually do.

    `effort` is the attacker's cost in units of slowdown: 0.0 changes nothing, 1.0 roughly doubles
    the flow's duration, 9.0 makes it ten times slower. That is a real price — a ten-fold slower
    scan takes ten times as long to complete — which is why it is the x-axis rather than a
    dimensionless epsilon.
    """
    rng = np.random.default_rng(seed)
    out = X.copy()
    n = len(out)
    if effort <= 0 or n == 0:
        return out

    if space.duration and space.duration in out.columns:
        # Jittered so every flow is not stretched by the identical factor, which would itself be a
        # detectable signature and would flatter the attacker.
        factor = 1.0 + effort * rng.uniform(0.5, 1.5, size=n)
        out[space.duration] = out[space.duration].to_numpy(dtype=float) * factor

    for col in space.available(set(out.columns)):
        values = out[col].to_numpy(dtype=float)
        # Padding scales with effort but sub-linearly: an attacker padding a payload tenfold is
        # paying bandwidth for it, and past a point the padding is more conspicuous than the flow.
        grown = values * (1.0 + 0.3 * np.log1p(effort) * rng.uniform(0.0, 1.0, size=n))
        out[col] = np.ceil(grown) if col in space.integral else grown

    _recompute(out, space)
    return out


def feature_space_attack(
    X: pd.DataFrame,
    effort: float,
    space: ManipulationSpace,
    *,
    benign_reference: pd.DataFrame | None = None,
    seed: int = 0,
) -> pd.DataFrame:
    """The strawman, included so the comparison can be made rather than asserted.

    An unconstrained move through feature space **towards the benign centroid**: every numeric
    column slides whichever way helps, fractional packet counts are fine, and derived features
    drift free of the primitives that define them.

    It has to actually be an attack for the comparison to mean anything. An earlier version of this
    function added symmetric Gaussian noise, which is not an attack at all - it barely moved
    detection, and it would have made the unconstrained baseline look *weaker* than the realisable
    one, inverting the finding. Interpolating towards the benign centroid is the cheap honest
    version: no gradients, no model access, and it still evades far more than anything realisable.

    This is roughly what most of the literature reports, and it is here **to be beaten**.
    """
    rng = np.random.default_rng(seed)
    out = X.copy()
    numeric = out.select_dtypes(include="number").columns
    if effort <= 0 or len(numeric) == 0:
        return out

    block = out[numeric].to_numpy(dtype=float)
    if benign_reference is not None and len(benign_reference):
        target = (
            benign_reference.reindex(columns=numeric).to_numpy(dtype=float)
            if set(numeric) - set(benign_reference.columns)
            else benign_reference[numeric].to_numpy(dtype=float)
        )
        centroid = np.nanmean(target, axis=0)
    else:
        centroid = np.nanmean(block, axis=0)
    centroid = np.nan_to_num(centroid, nan=0.0)

    # `effort` here buys a fraction of the distance to the centroid. Capped just below 1 so the
    # attack never simply replaces the row with the benign mean, which would be a statement about
    # arithmetic rather than about the detector.
    alpha = min(0.95, 0.1 * effort)
    jitter = rng.uniform(0.8, 1.2, size=block.shape)
    moved = block + alpha * jitter * (centroid - block)
    out[numeric] = np.nan_to_num(moved, nan=0.0, posinf=0.0, neginf=0.0)
    return out


# =================================================================================================
# Measurement
# =================================================================================================


@dataclass
class EffortPoint:
    effort: float
    detection_rate: float
    n_evaded: int
    mean_duration_multiple: float = 1.0


@dataclass
class AttackResult:
    name: str
    constrained: bool
    points: list[EffortPoint] = field(default_factory=list)

    @property
    def baseline(self) -> float:
        return self.points[0].detection_rate if self.points else float("nan")

    def detection_at(self, effort: float) -> float:
        for point in self.points:
            if point.effort >= effort:
                return point.detection_rate
        return self.points[-1].detection_rate if self.points else float("nan")


@dataclass
class EvasionReport:
    dataset: str
    n_attacks: int
    threshold: float
    results: list[AttackResult] = field(default_factory=list)
    per_family: dict[str, dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "n_attacks": self.n_attacks,
            "threshold": self.threshold,
            "results": [
                {"name": r.name, "constrained": r.constrained, "points": [asdict(p) for p in r.points]}
                for r in self.results
            ],
            "per_family": self.per_family,
        }

    def summary(self) -> str:
        lines = [
            "=" * 88,
            "  ADVERSARIAL EVASION UNDER PROBLEM-SPACE CONSTRAINTS",
            "=" * 88,
            "",
            f"  {self.n_attacks:,} attack flows, detector threshold {self.threshold:.4f}",
            "",
            "  Effort is the attacker's own cost: 9.0 means accepting a roughly 10x slower flow.",
            "  That price is why it is the x-axis. An epsilon is not a price.",
            "",
        ]
        for result in self.results:
            label = "PROBLEM-SPACE (realisable)" if result.constrained else "feature-space (strawman)"
            lines += [f"  {label}", f"    {'effort':>8} {'detection':>11} {'evaded':>9} {'x slower':>10}"]
            for point in result.points:
                lines.append(
                    f"    {point.effort:>8.1f} {point.detection_rate:>11.4f} "
                    f"{point.n_evaded:>9,} {point.mean_duration_multiple:>10.2f}"
                )
            lines.append("")

        constrained = next((r for r in self.results if r.constrained), None)
        strawman = next((r for r in self.results if not r.constrained), None)
        if constrained and strawman and constrained.points:
            effort = constrained.points[-1].effort
            real = constrained.detection_at(effort)
            fake = strawman.detection_at(effort)
            lines += [
                "  The gap between those two tables is the whole point:",
                "",
                f"    at equal effort, the unconstrained attack leaves detection at {fake:.4f}",
                f"    and the realisable attack leaves it at {real:.4f}.",
                "",
                "  The first number is the one most papers report. It describes perturbations that",
                "  include fractional packet counts and throughputs inconsistent with their own byte",
                "  counts - rows no network can carry. Detections it 'evades' were never defended.",
                "",
            ]

        if self.per_family:
            lines += ["  Detection under the realisable attack, per family:", ""]
            lines.append(f"  {'family':22} {'baseline':>10} {'attacked':>10} {'delta':>10} {'n':>8}")
            for family, row in sorted(self.per_family.items(), key=lambda kv: kv[1]["delta"]):
                lines.append(
                    f"  {family[:20]:22} {row['baseline']:>10.4f} {row['attacked']:>10.4f} "
                    f"{row['delta']:>+10.4f} {int(row['n']):>8,}"
                )
            lines.append("")
        return "\n".join(lines)


def evaluate(
    score_fn: Any,
    X_attacks: pd.DataFrame,
    *,
    dataset: str,
    threshold: float,
    benign_reference: pd.DataFrame | None = None,
    families: pd.Series | None = None,
    efforts: tuple[float, ...] = (0.0, 0.5, 1.0, 2.0, 4.0, 9.0),
    seed: int = 0,
    on_progress: Any = None,
) -> EvasionReport:
    """Measure detection decay under both attacks.

    `score_fn` takes a feature frame and returns P(attack) per row; `threshold` is the detector's
    operating point. Attack rows only — evasion is about what happens to traffic that *is* an
    attack, and mixing benign rows in would dilute the rate with something the attacker never
    touched.
    """
    space = space_for(dataset)
    results: list[AttackResult] = []
    duration_col = space.duration if space.duration in X_attacks.columns else None
    base_duration = float(np.nanmean(X_attacks[duration_col].to_numpy(dtype=float))) if duration_col else 1.0

    for name, constrained in (("problem-space", True), ("feature-space", False)):
        points: list[EffortPoint] = []
        for effort in efforts:
            perturbed = (
                problem_space_attack(X_attacks, effort, space, seed=seed)
                if constrained
                else feature_space_attack(
                    X_attacks, effort, space, benign_reference=benign_reference, seed=seed
                )
            )
            scores = np.asarray(score_fn(perturbed)).ravel()
            detected = scores >= threshold
            multiple = 1.0
            if duration_col and base_duration > 0:
                multiple = float(np.nanmean(perturbed[duration_col].to_numpy(dtype=float))) / base_duration
            points.append(
                EffortPoint(
                    effort=float(effort),
                    detection_rate=float(detected.mean()),
                    n_evaded=int((~detected).sum()),
                    mean_duration_multiple=multiple,
                )
            )
            if on_progress:
                on_progress(f"{name} effort {effort:>4.1f}: detection {detected.mean():.4f}")
        results.append(AttackResult(name=name, constrained=constrained, points=points))

    per_family: dict[str, dict[str, float]] = {}
    if families is not None:
        strongest = max(efforts)
        attacked = problem_space_attack(X_attacks, strongest, space, seed=seed)
        base_scores = np.asarray(score_fn(X_attacks)).ravel() >= threshold
        post_scores = np.asarray(score_fn(attacked)).ravel() >= threshold
        fam = families.reset_index(drop=True).to_numpy()
        for family in sorted(set(fam.tolist())):
            rows = fam == family
            n = int(rows.sum())
            if n < 20:  # a detection rate from fewer rows than this is not a rate
                continue
            before, after = float(base_scores[rows].mean()), float(post_scores[rows].mean())
            per_family[str(family)] = {
                "baseline": before,
                "attacked": after,
                "delta": after - before,
                "n": float(n),
            }

    return EvasionReport(
        dataset=dataset,
        n_attacks=len(X_attacks),
        threshold=threshold,
        results=results,
        per_family=per_family,
    )


def write_report(report: EvasionReport, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    return path
