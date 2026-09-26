"""E8: when the data drifts, is it the thresholds or the notion of normal that is wrong?

Pre-registered in docs/EXPERIMENTS.md (E8) before this module existed. The design is implemented as
written there; where the text is precise, the constant below carries the same number.

NSL-KDD. A champion is fitted on KDDTrain+ at a 1% target. KDDTest+ is split by `canary.live_split`
into canary (unused), recent (windows are drawn from it) and evaluation (every reported number).
A window is N ground-truth-benign rows of the recent pool, standing in for a vetted quiet period.

  A0        champion as shipped
  A1        thresholds only: OrGate.fit on the window's scores, heads unchanged
  A2        re-learn normal: PenumbraDetector.rebaselined, window split 5:3 into fit and calibration
  A2-dirty  A2 on all recent benign rows plus attack rows amounting to 5% (and 1%) of the window
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import pandas as pd

from penumbra.data.loaders.base import Dataset
from penumbra.eval import canary
from penumbra.eval.poisoning import wilson
from penumbra.models import supervised
from penumbra.models.detector import PenumbraDetector
from penumbra.models.fusion import OrGate
from penumbra.seeds import SEED

TARGET_FPR = 0.01
WINDOW_SIZES = (500, 1000, 2000)
WINDOW_SEEDS = (0, 1, 2, 3, 4)
DIRTY_DOSES = (0.01, 0.05)
DIRTY_SEEDS = (0, 1, 2)
FIT_SHARE = 5 / 8


def thresholds_only(detector: PenumbraDetector, window: pd.DataFrame) -> PenumbraDetector:
    """A1: same heads, both thresholds re-fitted on the window at the champion's target."""
    if detector.novelty is None:
        raise RuntimeError("detector is not fitted")
    out = copy.copy(detector)
    p = supervised.attack_scores(detector.supervised_model, window)
    n = detector.novelty.score(detector.novelty_prep.transform(window), how="max")
    out.gate = OrGate.fit(p, n, total_fpr=detector.target_fpr, use_novelty=True)
    return out


def relearn_normal(detector: PenumbraDetector, window: pd.DataFrame, seed: int) -> PenumbraDetector:
    """A2: `rebaselined`, with the window split 5:3 into fit and calibration."""
    order = np.random.default_rng(seed).permutation(len(window))
    cut = int(round(len(window) * FIT_SHARE))
    return detector.rebaselined(
        window.iloc[np.sort(order[:cut])], window.iloc[np.sort(order[cut:])], source="e8 window"
    )


def evaluate(scored: pd.DataFrame, y: np.ndarray, unseen: np.ndarray) -> dict[str, Any]:
    fired = scored["fired"].to_numpy() > 0
    reach = fired | scored["conformal_abstains"].to_numpy().astype(bool)
    benign, attack = y == 0, y == 1
    seen = attack & ~unseen

    def rate(mask: np.ndarray, hits: np.ndarray) -> dict[str, Any]:
        k, n = int(hits[mask].sum()), int(mask.sum())
        return {"rate": k / n if n else float("nan"), "k": k, "n": n, "ci95": wilson(k, n)}

    return {
        "fpr": rate(benign, fired),
        "recall_all": rate(attack, fired),
        "recall_unseen17": rate(unseen, fired),
        "recall_seen": rate(seen, fired),
        "benign_reach": rate(benign, reach),
    }


def run(
    ds: Dataset,
    fine_train: pd.Series,
    fine_test: pd.Series,
    unseen_test: np.ndarray,
    *,
    model_name: str = "rf",
    seed: int = SEED,
    on_progress: Any = None,
) -> dict[str, Any]:
    def step(msg: str) -> None:
        if on_progress:
            on_progress(msg)

    fine_test = fine_test.astype(str).reset_index(drop=True)
    _, recent_idx, eval_idx = canary.live_split(fine_test, seed=seed)
    y_test = ds.y_test.reset_index(drop=True).to_numpy().astype(int)
    X_test = ds.X_test.reset_index(drop=True)

    recent_benign = X_test.iloc[recent_idx[y_test[recent_idx] == 0]].reset_index(drop=True)
    recent_attack = X_test.iloc[recent_idx[y_test[recent_idx] == 1]].reset_index(drop=True)
    X_ev, y_ev, unseen_ev = X_test.iloc[eval_idx], y_test[eval_idx], np.asarray(unseen_test)[eval_idx]

    step(f"champion ({model_name}, target {TARGET_FPR:.0%})")
    champion = PenumbraDetector(target_fpr=TARGET_FPR).fit(ds, model_name=model_name)

    def score(det: Any) -> dict[str, Any]:
        out = evaluate(det.score(X_ev), y_ev, unseen_ev)
        out["thresholds"] = {
            "supervised": det.gate.supervised_threshold,
            "novelty": det.gate.novelty_threshold,
        }
        return out

    report: dict[str, Any] = {
        "experiment": "E8",
        "dataset": ds.name,
        "target_fpr": TARGET_FPR,
        "seed": seed,
        "pools": {
            "recent_benign": len(recent_benign),
            "recent_attack": len(recent_attack),
            "eval_rows": len(eval_idx),
            "eval_benign": int((y_ev == 0).sum()),
            "eval_unseen17": int(unseen_ev.sum()),
        },
        "A0": score(champion),
        "windows": [],
        "dirty": [],
    }
    step(f"A0 realised FPR {report['A0']['fpr']['rate']:.2%}")

    sizes: list[tuple[int, int | None]] = [(n, s) for n in WINDOW_SIZES for s in WINDOW_SEEDS]
    sizes.append((len(recent_benign), None))
    for n, window_seed in sizes:
        if window_seed is None:
            window = recent_benign
        else:
            rows = np.random.default_rng(window_seed).choice(len(recent_benign), size=n, replace=False)
            window = recent_benign.iloc[np.sort(rows)]
        a1 = score(thresholds_only(champion, window))
        a2 = score(relearn_normal(champion, window, window_seed or 0))
        report["windows"].append({"n": len(window), "seed": window_seed, "A1": a1, "A2": a2})
        step(
            f"N={len(window):>5} seed={window_seed}  A1 FPR {a1['fpr']['rate']:.2%} unseen "
            f"{a1['recall_unseen17']['rate']:.3f} | A2 FPR {a2['fpr']['rate']:.2%} unseen {a2['recall_unseen17']['rate']:.3f}"
        )

    for dose in DIRTY_DOSES:
        for dirty_seed in DIRTY_SEEDS:
            n_attack = int(round(dose * len(recent_benign) / (1 - dose)))
            rows = np.random.default_rng(1000 + dirty_seed).choice(
                len(recent_attack), size=n_attack, replace=False
            )
            window = pd.concat([recent_benign, recent_attack.iloc[np.sort(rows)]], ignore_index=True)
            dirty = score(relearn_normal(champion, window, dirty_seed))
            clean = score(relearn_normal(champion, recent_benign, dirty_seed))
            report["dirty"].append(
                {
                    "dose": dose,
                    "seed": dirty_seed,
                    "attack_rows": n_attack,
                    "A2_dirty": dirty,
                    "A2_clean": clean,
                }
            )
            step(
                f"dirty {dose:.0%} seed={dirty_seed}: unseen {clean['recall_unseen17']['rate']:.3f} -> "
                f"{dirty['recall_unseen17']['rate']:.3f}"
            )

    report["summary"] = summarise(report)
    return report


def _rate(entry: dict[str, Any], metric: str) -> float:
    return float(entry[metric]["rate"])


def summarise(report: dict[str, Any]) -> dict[str, Any]:
    """The six registered predictions, each checked against the numbers."""
    a0 = report["A0"]
    full = next(w for w in report["windows"] if w["seed"] is None)
    a1, a2 = full["A1"], full["A2"]

    def spread(n: int) -> float:
        values = [_rate(w["A1"], "fpr") for w in report["windows"] if w["n"] == n and w["seed"] is not None]
        return float(np.std(values, ddof=1)) if len(values) > 1 else float("nan")

    def dirty_loss(dose: float) -> float:
        rows = [d for d in report["dirty"] if d["dose"] == dose]
        return float(
            np.mean(
                [
                    _rate(d["A2_clean"], "recall_unseen17") - _rate(d["A2_dirty"], "recall_unseen17")
                    for d in rows
                ]
            )
        )

    in_band = lambda r: 0.005 <= r <= 0.02  # noqa: E731 - the registered band, stated once
    sd500, sd2000 = spread(500), spread(2000)
    loss1, loss5 = dirty_loss(0.01), dirty_loss(0.05)
    return {
        "P1_A0_fpr": {"value": _rate(a0, "fpr"), "predicted": "roughly 10%"},
        "P2_H8a_A1_fpr_in_band": {"value": _rate(a1, "fpr"), "held": in_band(_rate(a1, "fpr"))},
        "P3_H8b_A1_unseen_loss": {
            "value": _rate(a0, "recall_unseen17") - _rate(a1, "recall_unseen17"),
            "held": _rate(a0, "recall_unseen17") - _rate(a1, "recall_unseen17") >= 0.10,
        },
        "P4_H8c_A2_over_A1_unseen": {
            "value": _rate(a2, "recall_unseen17") - _rate(a1, "recall_unseen17"),
            "A2_fpr": _rate(a2, "fpr"),
            "held": (_rate(a2, "recall_unseen17") - _rate(a1, "recall_unseen17") >= 0.05)
            and in_band(_rate(a2, "fpr")),
        },
        "P5_H8d_spread": {"sd_500": sd500, "sd_2000": sd2000, "held": sd500 >= 2 * sd2000},
        "P6_H8e_dirty": {"loss_1pct": loss1, "loss_5pct": loss5, "held": loss5 >= 0.05 and loss1 < loss5},
    }
