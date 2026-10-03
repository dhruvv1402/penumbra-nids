"""E9a: how much of the realised FPR is our own in-sample optimism? (EXPERIMENTS.md)

Before E9a, `PenumbraDetector.fit` fitted the supervised threshold and the conformal layer on rows
the supervised model had trained on, and the shipped UNSW detector kept the testbed TTL columns the
audit quarantines. This drill measures both, on the full test splits:

  UNSW      2 x 2    TTL kept / quarantined  x  in-sample / held-out calibration
  NSL-KDD   1 x 2    in-sample / held-out (nothing is quarantined on NSL-KDD)
  lab       out of the box, no re-baseline: the shipped detector and the two held-out UNSW arms
  gate      the TTL-quarantined held-out arm against the registry champion, old gate and ADR-0005

Headless: the CLI loads datasets, the capture and the champion and hands them in, so nothing here
touches storage, the registry or the alert layer.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from penumbra.data.loaders.base import Dataset
from penumbra.models.detector import PenumbraDetector

TARGET_FPR = 0.01


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def _scored(det: PenumbraDetector, X: pd.DataFrame) -> pd.DataFrame:
    return det.score(X[det._feature_names] if det._feature_names else X)


def rates(scored: pd.DataFrame, y: np.ndarray | None = None) -> dict[str, Any]:
    """Fired, abstained and reached-an-analyst rates; split by class when labels are given."""
    fired = scored["fired"].to_numpy() > 0
    reach = fired | scored["conformal_abstains"].to_numpy().astype(bool)
    if y is None:
        return {
            "flows": int(len(fired)),
            "fired": float(fired.mean()),
            "reached_an_analyst": float(reach.mean()),
        }
    y = np.asarray(y).astype(int)
    benign, attack = y == 0, y == 1
    k, n = int(fired[benign].sum()), int(benign.sum())
    return {
        "benign_rows": n,
        "attack_rows": int(attack.sum()),
        "realised_fpr": k / n if n else float("nan"),
        "realised_fpr_ci95": list(wilson(k, n)),
        "recall": float(fired[attack].mean()) if attack.any() else float("nan"),
        "benign_reaching_an_analyst": float(reach[benign].mean()) if n else float("nan"),
    }


def describe(det: PenumbraDetector) -> dict[str, Any]:
    assert det.gate is not None and det.metadata is not None
    out: dict[str, Any] = {
        "calibration": det.metadata.calibration,
        "quarantined": det.metadata.quarantined_features,
        "supervised_threshold": det.gate.supervised_threshold,
        "novelty_threshold": det.gate.novelty_threshold,
        "conformal_benign_quantile": None,
    }
    if det.conformal is not None:
        out["conformal_benign_quantile"] = det.conformal.quantiles_.get(0)
    return out


def fit_arm(
    ds: Dataset, *, calibration: str, quarantined: list[str], on_progress: Any = None
) -> PenumbraDetector:
    # The family model names alerts; it plays no part in what fires, so the drill skips it.
    return PenumbraDetector(target_fpr=TARGET_FPR).fit(
        ds,
        model_name="rf",
        fit_family_model=False,
        calibration=calibration,
        quarantined=quarantined,
        on_progress=on_progress,
    )


def arm(det: PenumbraDetector, ds: Dataset) -> dict[str, Any]:
    return {**describe(det), "test": rates(_scored(det, ds.X_test), ds.y_test.to_numpy())}


def lab_rates(det: PenumbraDetector, X: pd.DataFrame, attack: np.ndarray) -> dict[str, Any]:
    scored = _scored(det, X)
    attack = np.asarray(attack).astype(bool)
    return {
        "ordinary_flows": rates(scored[~attack]),
        "attack_flows": rates(scored[attack]),
    }


def summarise(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Each pre-registered prediction, marked held or not. The predictions are not edited."""
    out: dict[str, dict[str, Any]] = {}
    unsw = report.get("unsw", {})
    nsl = report.get("nslkdd", {})

    def fpr(arms: dict[str, Any], key: str) -> float:
        return float(arms[key]["test"]["realised_fpr"]) if key in arms else float("nan")

    if unsw:
        shipped = fpr(unsw, "kept/in_sample")
        held = fpr(unsw, "kept/held_out")
        out["P1 shipped arm realises 15-22% on UNSW"] = {"value": shipped, "held": 0.15 <= shipped <= 0.22}
        out["P2 H9f held-out lowers UNSW FPR by >= 2 points and stays > 2%"] = {
            "value": {"in_sample": shipped, "held_out": held},
            "held": (shipped - held) >= 0.02 and held > 0.02,
        }
        q_ho = fpr(unsw, "quarantined/held_out")
        rec = {k: float(v["test"]["recall"]) for k, v in unsw.items()}
        out["P5 quarantine moves UNSW FPR and recall by < 1 point"] = {
            "value": {
                "fpr_delta": q_ho - held,
                "recall_delta": rec["quarantined/held_out"] - rec["kept/held_out"],
            },
            "held": abs(q_ho - held) < 0.01
            and abs(rec["quarantined/held_out"] - rec["kept/held_out"]) < 0.01,
        }
    if nsl:
        held_n = fpr(nsl, "held_out")
        out["P3 H9f NSL-KDD held-out realises 5-9.5%"] = {
            "value": {"in_sample": fpr(nsl, "in_sample"), "held_out": held_n},
            "held": 0.05 <= held_n <= 0.095 and held_n < fpr(nsl, "in_sample"),
        }
    if unsw and nsl:

        def reach(arms: dict[str, Any], key: str) -> float:
            return float(arms[key]["test"]["benign_reaching_an_analyst"])

        out["P4 held-out lowers benign reaching an analyst on both"] = {
            "value": {
                "unsw": [reach(unsw, "kept/in_sample"), reach(unsw, "kept/held_out")],
                "nslkdd": [reach(nsl, "in_sample"), reach(nsl, "held_out")],
            },
            "held": reach(unsw, "kept/held_out") < reach(unsw, "kept/in_sample")
            and reach(nsl, "held_out") < reach(nsl, "in_sample"),
        }
    lab = report.get("lab")
    if lab:
        kept = lab["kept/held_out"]["ordinary_flows"]["fired"]
        quar = lab["quarantined/held_out"]["ordinary_flows"]["fired"]
        out["P6 H9h quarantine lowers lab ordinary fired rate by >= 10 points, stays > 5%"] = {
            "value": {
                "kept": kept,
                "quarantined": quar,
                "shipped": lab["shipped"]["ordinary_flows"]["fired"],
            },
            "held": (kept - quar) >= 0.10 and quar > 0.05,
        }
    gate = report.get("gate")
    if gate:
        out["P7 H9g old gate refuses (G1), ADR-0005 gate passes"] = {
            "value": {
                "old": gate["at_own_threshold"],
                "amended": {"passed": gate["passed"], "reasons": gate["reasons"]},
            },
            "held": (not gate["at_own_threshold"]["passed"])
            and any(r.startswith("G1") for r in gate["at_own_threshold"]["reasons"])
            and bool(gate["passed"]),
        }
    return out
