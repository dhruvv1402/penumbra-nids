"""Shadow scoring: run the challenger beside the champion on the same traffic, alerting on nothing.

The canary gate answers "is the challenger worse on labels we trust?". Shadow answers the questions a
SOC lead actually asks before a model change, none of which need labels:

  * how many more (or fewer) alerts will my analysts get, per lane?
  * how often do the two models disagree, and beyond chance (Cohen's kappa)?
  * where they disagree, what was the champion saying - is the challenger dropping known-attack
    detections, or trimming noise from the hunting lane?

Labels, where the stream has them, are used for one extra column and nothing else: in production
they arrive days late or never, and a shadow report that only works with labels is not one.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

FIRED_NAMES = {0: "none", 1: "supervised", 2: "novelty_only", 3: "both"}


def cohen_kappa(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a).astype(bool), np.asarray(b).astype(bool)
    observed = float((a == b).mean())
    pa, pb = a.mean(), b.mean()
    expected = float(pa * pb + (1 - pa) * (1 - pb))
    return (observed - expected) / (1 - expected) if expected < 1 else 1.0


def compare(champion: pd.DataFrame, challenger: pd.DataFrame, y: np.ndarray | None = None) -> dict[str, Any]:
    """Compare two `PenumbraDetector.score` outputs on the same rows."""
    fc, fx = champion["fired"].to_numpy(), challenger["fired"].to_numpy()
    ac, ax = fc > 0, fx > 0
    n = len(fc)

    lost = ac & ~ax  # champion alerted, challenger would not
    gained = ~ac & ax

    def by_head(mask: np.ndarray, fired: np.ndarray) -> dict[str, int]:
        return {FIRED_NAMES[k]: int(((fired == k) & mask).sum()) for k in (1, 2, 3)}

    out: dict[str, Any] = {
        "n_rows": int(n),
        "alerts": {"champion": int(ac.sum()), "challenger": int(ax.sum())},
        "alert_rate": {"champion": float(ac.mean()), "challenger": float(ax.mean())},
        "alert_volume_ratio": float(ax.sum() / ac.sum()) if ac.sum() else float("nan"),
        "by_head": {"champion": by_head(ac, fc), "challenger": by_head(ax, fx)},
        "agreement": float((ac == ax).mean()),
        "cohen_kappa": cohen_kappa(ac, ax),
        "lost": {"n": int(lost.sum()), "champion_head": by_head(lost, fc)},
        "gained": {"n": int(gained.sum()), "challenger_head": by_head(gained, fx)},
        "p_attack_shift": float(np.mean(challenger["p_attack"].to_numpy() - champion["p_attack"].to_numpy())),
    }
    if y is not None:
        y = np.asarray(y).astype(int)
        out["labelled"] = {
            "note": "labels are rarely available in production; shown because this stream has them",
            "lost_that_were_attacks": int((lost & (y == 1)).sum()),
            "gained_that_were_attacks": int((gained & (y == 1)).sum()),
            "lost_that_were_benign": int((lost & (y == 0)).sum()),
            "gained_that_were_benign": int((gained & (y == 0)).sum()),
        }
    return out


def summary(report: dict[str, Any]) -> str:
    a = report["alerts"]
    lines = [
        f"shadow over {report['n_rows']:,} flows - nothing alerted, both models scored",
        f"  alerts        champion {a['champion']:,}   challenger {a['challenger']:,}   "
        f"(x{report['alert_volume_ratio']:.2f})",
        f"  agreement     {report['agreement']:.4f}   Cohen's kappa {report['cohen_kappa']:.4f}",
        f"  challenger drops {report['lost']['n']:,} champion alerts {report['lost']['champion_head']}",
        f"  challenger adds  {report['gained']['n']:,} new alerts      {report['gained']['challenger_head']}",
    ]
    if "labelled" in report:
        lab = report["labelled"]
        lines.append(
            f"  (labelled stream) dropped attacks {lab['lost_that_were_attacks']:,}, "
            f"dropped benign {lab['lost_that_were_benign']:,}, new attacks {lab['gained_that_were_attacks']:,}, "
            f"new benign {lab['gained_that_were_benign']:,}"
        )
    return "\n".join(lines)
