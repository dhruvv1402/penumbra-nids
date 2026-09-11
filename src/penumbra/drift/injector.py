"""Deliberate drift injection.

Drift monitoring is usually presented as a paragraph. This makes it a thing an audience watches
happen: the replay stream is perturbed on a schedule, the monitor fires, the measured recall falls,
and the "retrain" banner lights up - live, on stage, with a known ground-truth change point so the
detection delay can be quoted rather than estimated.

Four injection modes, because they are genuinely different failures and a monitor that catches one
does not necessarily catch another:

  COVARIATE   feature distributions shift; P(y|x) unchanged. The model is still right, the data has
              moved. Detectable without labels.
  NEW_FAMILY  an attack family the model never trained on enters the stream. The model is wrong and
              cannot know it. This is the zero-day case.
  SEASONAL    a gradual, reversible shift - a working-hours pattern, a backup window. The failure
              mode here is a monitor that fires every night at 2am.
  ADVERSARIAL slow-rate mimicry: an attacker stretches flow duration and scales rate to look benign.
              Semantics-preserving, and the only mode where the change is hostile rather than
              incidental.

Perturbations are **problem-space constrained** (Pierazzi et al., IEEE S&P 2020): packet counts stay
integral and never decrease, byte counts never decrease, and derived features are recomputed from
the primitives rather than perturbed independently. A flow with 38.3 packets is not a stealthier
attack, it is not a flow.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np
import pandas as pd

from penumbra.seeds import SEED


class DriftMode(StrEnum):
    COVARIATE = "covariate"
    NEW_FAMILY = "new_family"
    SEASONAL = "seasonal"
    ADVERSARIAL = "adversarial"


@dataclass
class InjectionPlan:
    """What to inject, where, and how hard."""

    mode: DriftMode
    change_point: int  # row index where drift begins
    magnitude: float = 1.0  # multiples of the reference standard deviation
    features: list[str] = field(default_factory=list)
    new_family: str | None = None
    ramp_rows: int = 0  # 0 = abrupt; >0 = gradual over this many rows

    def intensity_at(self, index: int) -> float:
        """How much drift applies at this row. Ramping makes gradual drift gradual."""
        if index < self.change_point:
            return 0.0
        if self.ramp_rows <= 0:
            return self.magnitude
        progress = min(1.0, (index - self.change_point) / self.ramp_rows)
        return self.magnitude * progress


def inject(
    df: pd.DataFrame,
    plan: InjectionPlan,
    *,
    numeric_features: list[str] | None = None,
    seed: int = SEED,
) -> pd.DataFrame:
    """Return a copy of `df` with drift applied from `plan.change_point` onward."""
    out = df.copy()
    rng = np.random.default_rng(seed)

    features = (
        plan.features or numeric_features or [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    )
    features = [f for f in features if f in out.columns]
    if not features:
        return out

    if plan.mode is DriftMode.COVARIATE:
        _inject_covariate(out, plan, features)
    elif plan.mode is DriftMode.SEASONAL:
        _inject_seasonal(out, plan, features)
    elif plan.mode is DriftMode.ADVERSARIAL:
        _inject_adversarial(out, plan, rng)
    # NEW_FAMILY is a row-level operation handled by `splice_new_family`, not a perturbation.

    return out


def _inject_covariate(df: pd.DataFrame, plan: InjectionPlan, features: list[str]) -> None:
    """Shift feature means by `magnitude` reference standard deviations."""
    idx = np.arange(len(df))
    intensity = np.array([plan.intensity_at(int(i)) for i in idx])
    for col in features:
        values = df[col].to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        if len(finite) < 2:
            continue
        sigma = float(np.std(finite)) or 1.0
        shifted = values + intensity * sigma
        # Counts and byte totals are non-negative. A shift that produces -3 packets is not drift,
        # it is a broken simulator.
        if any(tok in col.lower() for tok in ("pkt", "byte", "count", "loss", "dur", "ct_")):
            shifted = np.maximum(shifted, 0.0)
        df[col] = shifted


def _inject_seasonal(df: pd.DataFrame, plan: InjectionPlan, features: list[str]) -> None:
    """Smooth, reversible oscillation - the shape of a daily cycle.

    Included because it is the drift that should NOT trigger a retrain. A monitor that cannot
    distinguish a nightly backup window from genuine degradation will cry wolf every night, and an
    operator will switch it off.
    """
    idx = np.arange(len(df), dtype=float)
    period = max(len(df) / 4.0, 1.0)
    wave = np.sin(2 * np.pi * idx / period)
    active = (idx >= plan.change_point).astype(float)
    for col in features:
        values = df[col].to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        if len(finite) < 2:
            continue
        sigma = float(np.std(finite)) or 1.0
        df[col] = values + active * wave * plan.magnitude * sigma * 0.5


def _inject_adversarial(df: pd.DataFrame, plan: InjectionPlan, rng: np.random.Generator) -> None:
    """Slow-rate mimicry, with problem-space constraints respected.

    An attacker stretching a flow can: increase duration, add padding bytes, add packets, increase
    inter-packet gaps. An attacker CANNOT: un-send a packet, reduce the bytes their own exploit
    requires, or make a scan contact fewer hosts while still scanning them.

    Derived quantities are recomputed rather than perturbed. `rate`, `sload` and `smean` are
    functions of `dur`, `sbytes` and `spkts`; perturbing them independently produces rows that are
    off the data manifold, and an "attack" the model misses for that reason has not evaded anything.
    """
    mask = np.arange(len(df)) >= plan.change_point
    if not mask.any():
        return

    stretch = 1.0 + plan.magnitude * rng.uniform(2.0, 10.0, size=int(mask.sum()))

    if "dur" in df.columns:
        dur = df.loc[mask, "dur"].to_numpy(dtype=float)
        df.loc[mask, "dur"] = dur * stretch

    # Padding: bytes only ever increase.
    for col in ("sbytes", "dbytes"):
        if col in df.columns:
            v = df.loc[mask, col].to_numpy(dtype=float)
            df.loc[mask, col] = v * (1.0 + plan.magnitude * rng.uniform(0.0, 0.3, size=len(v)))

    # Packet counts stay integral and never decrease.
    for col in ("spkts", "dpkts"):
        if col in df.columns:
            v = df.loc[mask, col].to_numpy(dtype=float)
            df.loc[mask, col] = np.ceil(v * (1.0 + plan.magnitude * rng.uniform(0.0, 0.2, size=len(v))))

    _recompute_derived(df, mask)


def _recompute_derived(df: pd.DataFrame, mask: np.ndarray) -> None:
    """Recompute features that are functions of the primitives we just changed."""
    dur = df.loc[mask, "dur"].to_numpy(dtype=float) if "dur" in df.columns else None
    if dur is None:
        return
    safe_dur = np.where(dur > 0, dur, np.nan)

    # Recompute from whichever packet columns exist. Requiring both would silently skip the
    # recompute on a frame carrying only one, leaving `rate` inconsistent with the duration we just
    # stretched - which is exactly the off-manifold row this function exists to prevent.
    packet_cols = [c for c in ("spkts", "dpkts") if c in df.columns]
    if packet_cols and "rate" in df.columns:
        total = sum(df.loc[mask, c].to_numpy(float) for c in packet_cols)
        df.loc[mask, "rate"] = np.nan_to_num(total / safe_dur, nan=0.0, posinf=0.0)

    for byte_col, load_col in (("sbytes", "sload"), ("dbytes", "dload")):
        if byte_col in df.columns and load_col in df.columns:
            b = df.loc[mask, byte_col].to_numpy(float)
            df.loc[mask, load_col] = np.nan_to_num(b * 8.0 / safe_dur, nan=0.0, posinf=0.0)

    for byte_col, pkt_col, mean_col in (("sbytes", "spkts", "smean"), ("dbytes", "dpkts", "dmean")):
        if {byte_col, pkt_col, mean_col} <= set(df.columns):
            b = df.loc[mask, byte_col].to_numpy(float)
            p = df.loc[mask, pkt_col].to_numpy(float)
            df.loc[mask, mean_col] = np.nan_to_num(b / np.where(p > 0, p, np.nan), nan=0.0)


def splice_new_family(
    stream: pd.DataFrame,
    stream_labels: pd.Series,
    source: pd.DataFrame,
    source_labels: pd.Series,
    *,
    change_point: int,
    n_rows: int = 500,
    seed: int = SEED,
) -> tuple[pd.DataFrame, pd.Series, np.ndarray]:
    """Splice rows of an unseen attack family into the stream at a known point.

    Returns the new stream, its labels, and a boolean mask marking the injected rows - so the demo
    can show exactly which alerts were the injected family and measure detection on them alone.

    This is the zero-day scenario: the model has no representation for these rows and no way to know
    that. Only the novelty head has any chance, which makes it the honest live test of the thesis.
    """
    rng = np.random.default_rng(seed)
    take = min(n_rows, len(source))
    picks = rng.choice(len(source), size=take, replace=False)

    injected = source.iloc[picks].reset_index(drop=True)
    injected_labels = source_labels.iloc[picks].reset_index(drop=True)

    head = stream.iloc[:change_point]
    tail = stream.iloc[change_point:]
    new_stream = pd.concat([head, injected, tail], ignore_index=True)
    new_labels = pd.concat(
        [stream_labels.iloc[:change_point], injected_labels, stream_labels.iloc[change_point:]],
        ignore_index=True,
    )

    mask = np.zeros(len(new_stream), dtype=bool)
    mask[change_point : change_point + take] = True
    return new_stream, new_labels, mask


def scenario(name: str, n_rows: int) -> InjectionPlan:
    """Preset scenarios for the live demo."""
    mid = n_rows // 2
    presets = {
        "abrupt": InjectionPlan(DriftMode.COVARIATE, change_point=mid, magnitude=1.5),
        "gradual": InjectionPlan(DriftMode.COVARIATE, change_point=mid, magnitude=2.0, ramp_rows=n_rows // 4),
        "seasonal": InjectionPlan(DriftMode.SEASONAL, change_point=mid, magnitude=1.0),
        "zero_day": InjectionPlan(DriftMode.NEW_FAMILY, change_point=mid, new_family="Worms"),
        "evasion": InjectionPlan(DriftMode.ADVERSARIAL, change_point=mid, magnitude=1.0),
    }
    if name not in presets:
        raise ValueError(f"unknown scenario {name!r}; choose from {sorted(presets)}")
    return presets[name]


def windowed(df: pd.DataFrame, size: int) -> Iterator[tuple[int, pd.DataFrame]]:
    for start in range(0, len(df), size):
        yield start, df.iloc[start : start + size]
