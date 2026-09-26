"""Re-baselining on a new network, and the gate that decides whether the result may be promoted.

Deploy a detector trained on a 2015 testbed onto a real network and everything alerts: on our own
lab capture, all 2,979 flows reached an analyst (EVALUATION §10.7h). "Normal" has to be learnt where
the detector runs. `PenumbraDetector.rebaselined` does the learning; this module decides whether the
result is fit to use, from the new network's own benign traffic and nothing else.

The benign window is split three ways, by a fixed seed:

  fit          50%   the novelty head is refitted on it
  calibration  30%   benign references, both thresholds and the benign conformal quantile come from it
  holdout      20%   touched once, to check the operating point actually holds

The gate - blocking checks first, then evidence:

  R1  enough calibration flows. A threshold at the per-head FPR q is an order statistic; below
      5 / q flows it rests on fewer than five benign exceedances and is noise. The check refuses
      BEFORE fitting and says how many flows the window would need.
  R2  the held-out benign alert rate is within what the target allows. Holdouts are small, so the
      bound is the binomial 99% upper quantile at the target, not the target itself; a rate above
      it is evidence the threshold does not hold.

  evidence (not blocking)
      - how much of the window the CURRENT detector's supervised head fires on. On a network the
        supervised head does not transfer to this is high by construction (57% on the lab capture),
        so it cannot be a gate - but it is also the only signal that an intrusion was in progress
        while the baseline was recorded (THREAT_MODEL T9), so it is printed, never hidden.
      - with labelled attack flows (a lab capture): how many reach an analyst, before and after.
      - with a labelled dataset: known-attack recall at the new thresholds, before and after. The
        supervised model is untouched but its threshold moves, and that trade is stated.

Nothing here promotes anything. The CLI registers the result as a new, non-champion version with this
report attached as its gate; promotion is the registry's existing, separate, audited step.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import binom

from penumbra.models.fusion import per_head_budget
from penumbra.seeds import SEED

SPLIT = (0.5, 0.3, 0.2)
MIN_EXCEEDANCES = 5
HOLDOUT_CONFIDENCE = 0.99


def split(n: int, seed: int = SEED) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Positions of the fit, calibration and holdout slices of an n-row benign window."""
    order = np.random.default_rng(seed).permutation(n)
    a = int(round(n * SPLIT[0]))
    b = a + int(round(n * SPLIT[1]))
    return np.sort(order[:a]), np.sort(order[a:b]), np.sort(order[b:])


def minimum_calibration_rows(target_fpr: float) -> int:
    """Calibration flows needed for each head's threshold to rest on MIN_EXCEEDANCES benign rows."""
    return math.ceil(MIN_EXCEEDANCES / per_head_budget(target_fpr, 2))


def minimum_window(target_fpr: float) -> int:
    """Benign flows a window needs in total for R1 to pass at this target."""
    return math.ceil(minimum_calibration_rows(target_fpr) / SPLIT[1])


def holdout_limit(n: int, target_fpr: float) -> int:
    """Most alerts n held-out benign flows may raise before the target is evidently not holding."""
    return int(binom.ppf(HOLDOUT_CONFIDENCE, n, target_fpr))


def _rates(scored: pd.DataFrame) -> dict[str, Any]:
    fired = scored["fired"].to_numpy() > 0
    abstains = scored["conformal_abstains"].to_numpy().astype(bool)
    n = len(scored)
    return {
        "flows": n,
        "fired": int(fired.sum()),
        "fired_rate": float(fired.mean()) if n else 0.0,
        "supervised_fired": int(np.isin(scored["fired"].to_numpy(), (1, 3)).sum()),
        "novelty_fired": int(np.isin(scored["fired"].to_numpy(), (2, 3)).sum()),
        "abstained": int(abstains.sum()),
        "reached_an_analyst": int((fired | abstains).sum()),
        "reach_rate": float((fired | abstains).mean()) if n else 0.0,
    }


def run(
    detector: Any,
    X_benign: pd.DataFrame,
    *,
    target_fpr: float,
    source: str,
    seed: int = SEED,
    X_attack: pd.DataFrame | None = None,
    known: tuple[pd.DataFrame, np.ndarray] | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Re-baseline `detector` on `X_benign` and gate the result. Returns (detector or None, report).

    `X_attack` - labelled attack flows from the same network, if there are any (a lab capture).
    `known` - (X, y) from a labelled dataset, for the known-attack trade-off. Both are evidence only.
    """
    X_benign = X_benign.reset_index(drop=True)
    fit_idx, cal_idx, hold_idx = split(len(X_benign), seed)
    need_cal = minimum_calibration_rows(target_fpr)
    report: dict[str, Any] = {
        "kind": "rebaseline",
        "source": source,
        "target_fpr": target_fpr,
        "seed": seed,
        "window": {
            "benign_flows": len(X_benign),
            "fit": len(fit_idx),
            "calibration": len(cal_idx),
            "holdout": len(hold_idx),
        },
        "gates": {},
        "evidence": {},
        "reasons": [],
    }

    r1 = len(cal_idx) >= need_cal
    report["gates"]["R1_calibration_size"] = {
        "passed": r1,
        "calibration_flows": len(cal_idx),
        "needed": need_cal,
        "window_needed": minimum_window(target_fpr),
    }
    if not r1:
        report["reasons"].append(
            f"R1: {len(cal_idx):,} calibration flows cannot place a {per_head_budget(target_fpr, 2):.2%} "
            f"per-head threshold (need {need_cal:,}; a window of {minimum_window(target_fpr):,} benign "
            f"flows). Record a longer window, or re-baseline at a higher target FPR."
        )
        report["passed"] = False
        return None, report

    X_fit, X_cal, X_hold = (X_benign.iloc[i] for i in (fit_idx, cal_idx, hold_idx))
    rebased = detector.rebaselined(X_fit, X_cal, target_fpr=target_fpr, source=source)

    stock_window = detector.score(pd.concat([X_fit, X_cal]))
    report["evidence"]["current_detector_on_window"] = _rates(stock_window)

    stock_hold, new_hold = detector.score(X_hold), rebased.score(X_hold)
    limit = holdout_limit(len(X_hold), target_fpr)
    new_rates = _rates(new_hold)
    r2 = new_rates["fired"] <= limit
    report["gates"]["R2_holdout_fpr"] = {
        "passed": r2,
        "holdout_flows": len(X_hold),
        "alerts": new_rates["fired"],
        "limit": limit,
        "confidence": HOLDOUT_CONFIDENCE,
    }
    if not r2:
        report["reasons"].append(
            f"R2: {new_rates['fired']} of {len(X_hold):,} held-out benign flows fired; at a "
            f"{target_fpr:.1%} target at most {limit} would (binomial {HOLDOUT_CONFIDENCE:.0%} bound)."
        )
    report["evidence"]["holdout"] = {"current": _rates(stock_hold), "rebaselined": new_rates}
    report["thresholds"] = {
        "supervised": {
            "current": detector.gate.supervised_threshold,
            "rebaselined": rebased.gate.supervised_threshold,
        },
        "novelty": {
            "current": detector.gate.novelty_threshold,
            "rebaselined": rebased.gate.novelty_threshold,
        },
    }
    if detector.conformal is not None and rebased.conformal is not None:
        report["thresholds"]["conformal_benign_quantile"] = {
            "current": detector.conformal.quantiles_.get(0),
            "rebaselined": rebased.conformal.quantiles_.get(0),
        }

    if X_attack is not None and len(X_attack):
        report["evidence"]["attack_flows"] = {
            "current": _rates(detector.score(X_attack)),
            "rebaselined": _rates(rebased.score(X_attack)),
        }

    if known is not None:
        Xk, yk = known
        yk = np.asarray(yk).astype(int)

        def recall_fpr(scored: pd.DataFrame) -> dict[str, float]:
            fired = scored["fired"].to_numpy() > 0
            return {
                "attack_recall": float(fired[yk == 1].mean()) if (yk == 1).any() else float("nan"),
                "benign_fpr": float(fired[yk == 0].mean()) if (yk == 0).any() else float("nan"),
                "known_attack_recall": float(np.isin(scored["fired"].to_numpy()[yk == 1], (1, 3)).mean())
                if (yk == 1).any()
                else float("nan"),
            }

        report["evidence"]["known_dataset"] = {
            "rows": len(Xk),
            "current": recall_fpr(detector.score(Xk)),
            "rebaselined": recall_fpr(rebased.score(Xk)),
        }

    report["passed"] = r1 and r2
    return rebased, report
