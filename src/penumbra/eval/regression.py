"""The ML regression gate: model quality as a merge blocker, like a failing test.

CI fits the supervised head on NSL-KDD and compares three numbers against a committed baseline:

  roc_auc                  ranking quality, threshold-free
  recall_at_1pct           attack recall with exactly 1% of benign test rows flagged
  unseen_recall_at_1pct    the same, on the 17 attack types absent from training - the thesis
                           number, and the one a well-meaning change to the supervised head is
                           most likely to move without anyone noticing

Why not PR-AUC, which the original plan named: it moves with prevalence, and NSL-KDD's test set is
~57% attack. A gate on a prevalence-dependent number would fire on a data change and stay quiet on
a model change. Recall at a fixed benign budget is prevalence-invariant, and it is the operating
point this project reports everywhere else.

The budget is matched by rank (`budget.flags_at_benign_budget`), not by a quantile threshold: a
forest puts thousands of rows at exactly 0.0, and a quantile lands inside the tie and silently
delivers a different FPR from run to run.

A metric that IMPROVES beyond tolerance also fails the gate, softly: the baseline must be updated
deliberately in the same change, so an improvement is a reviewed claim rather than drift nobody
looked at.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score

from penumbra.eval.budget import flags_at_benign_budget

METRICS = ("roc_auc", "recall_at_1pct", "unseen_recall_at_1pct")
DEFAULT_TOLERANCE = {"roc_auc": 0.005, "recall_at_1pct": 0.02, "unseen_recall_at_1pct": 0.03}


def measure(scores: np.ndarray, y: np.ndarray, unseen: np.ndarray, *, fpr: float = 0.01) -> dict[str, float]:
    y = np.asarray(y).astype(int)
    unseen = np.asarray(unseen).astype(bool)
    budget = int(round((y == 0).sum() * fpr))
    flagged = flags_at_benign_budget(np.asarray(scores), y, budget)
    attack = y == 1
    return {
        "roc_auc": float(roc_auc_score(y, scores)),
        "recall_at_1pct": float(flagged[attack].mean()),
        "unseen_recall_at_1pct": float(flagged[attack & unseen].mean())
        if (attack & unseen).any()
        else float("nan"),
        "realised_fpr": float(flagged[~attack].mean()),
        "n_test": int(len(y)),
        "n_unseen": int((attack & unseen).sum()),
    }


@dataclass
class GateOutcome:
    passed: bool
    regressions: list[str]
    improvements: list[str]
    current: dict[str, float]
    baseline: dict[str, float]

    def summary(self) -> str:
        lines = [f"{'metric':<24} {'baseline':>9} {'current':>9} {'delta':>8}"]
        for m in METRICS:
            b, c = self.baseline[m], self.current[m]
            lines.append(f"{m:<24} {b:>9.4f} {c:>9.4f} {c - b:>+8.4f}")
        lines.append(f"{'realised_fpr':<24} {'':>9} {self.current['realised_fpr']:>9.4f}")
        for r in self.regressions:
            lines.append(f"REGRESSION  {r}")
        for i in self.improvements:
            lines.append(f"IMPROVED    {i} - update the baseline in this change so the claim is reviewed")
        lines.append("PASSED" if self.passed else "FAILED")
        return "\n".join(lines)


def compare(current: dict[str, float], baseline: dict[str, Any]) -> GateOutcome:
    tolerance = {**DEFAULT_TOLERANCE, **baseline.get("tolerance", {})}
    regressions, improvements = [], []
    for m in METRICS:
        b, c, tol = float(baseline["metrics"][m]), float(current[m]), float(tolerance[m])
        if c < b - tol:
            regressions.append(f"{m} {b:.4f} -> {c:.4f} (tolerance {tol})")
        elif c > b + tol:
            improvements.append(f"{m} {b:.4f} -> {c:.4f} (tolerance {tol})")
    return GateOutcome(
        passed=not regressions and not improvements,
        regressions=regressions,
        improvements=improvements,
        current=current,
        baseline={m: float(baseline["metrics"][m]) for m in METRICS},
    )


def write_baseline(path: Path, current: dict[str, float], *, note: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "dataset": "nslkdd",
                "model": "rf",
                "note": note,
                "metrics": {m: round(current[m], 6) for m in METRICS},
                "context": {k: current[k] for k in ("realised_fpr", "n_test", "n_unseen")},
                "tolerance": DEFAULT_TOLERANCE,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
