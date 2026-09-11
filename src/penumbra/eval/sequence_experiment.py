"""Does sequence context buy recall that per-flow features cannot?

Three arms, scored on **exactly the same rows**, compared at a **matched false-positive budget**:

1. **per-flow** — a random forest on the dataset's own features. The floor.
2. **per-flow + graph** — the same forest plus nine causal entity-graph features. Cheap: no deep
   model, no GPU, ~40 lines of windowing.
3. **sequence** — the 1D-CNN/BiGRU over per-host flow windows. The deep model.

Two disciplines make this an experiment rather than a demo.

**The same rows.** The sequence builder drops rows with unparseable timestamps and reorders by
time, so its predictions are a subset in a different order. Every arm is evaluated on `seq.index`
— the intersection — and never on its own convenient subset. Comparing arm 3 on the rows it kept
against arm 1 on all rows would be a different dataset, not a different model.

**A matched budget.** "The deep model has higher recall" is empty if it also alerts more. Each
arm's threshold is chosen so it flags the same fraction of *benign* rows, and recall is read off
there. Matching on total alert count instead would cap recall at the attack prevalence, which is
the bug this project already hit once (EVALUATION.md §10.4).

The hypothesis is pre-registered in `docs/EXPERIMENTS.md`. If sequence context buys nothing, that
is the result and it gets reported — the trees-beat-deep-nets literature on tabular data
(Grinsztajn et al., NeurIPS 2022) says it is the likely one, and the question worth asking is
whether the temporal axis is the exception.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from penumbra.data.loaders.base import Dataset
from penumbra.eval import budget as budget_mod
from penumbra.features import entity_graph, windows
from penumbra.seeds import SEED


@dataclass
class ArmResult:
    name: str
    roc_auc: float
    recall_at_budget: float
    threshold: float
    realised_fpr: float
    n_alerts: int
    notes: str = ""

    def row(self) -> str:
        return (
            f"  {self.name:24} {self.roc_auc:>8.4f} {self.recall_at_budget:>10.4f} "
            f"{self.realised_fpr:>10.4f} {self.n_alerts:>9,}"
        )


@dataclass
class SequenceExperiment:
    dataset: str
    fpr_budget: float
    n_train: int
    n_test: int
    n_features: int
    window_length: int
    arms: list[ArmResult] = field(default_factory=list)
    training: dict[str, Any] = field(default_factory=dict)
    per_family: dict[str, dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "fpr_budget": self.fpr_budget,
            "n_train": self.n_train,
            "n_test": self.n_test,
            "n_features": self.n_features,
            "window_length": self.window_length,
            "arms": [asdict(a) for a in self.arms],
            "training": self.training,
            "per_family": self.per_family,
        }

    def summary(self) -> str:
        lines = [
            "=" * 88,
            "  DOES SEQUENCE CONTEXT BUY RECALL PER-FLOW FEATURES CANNOT?",
            "=" * 88,
            "",
            f"  {self.n_train:,} train / {self.n_test:,} test windows, {self.n_features} features, "
            f"K={self.window_length}",
            f"  All arms scored on the SAME rows, at a matched {self.fpr_budget:.1%} false-positive budget.",
            "",
            f"  {'arm':24} {'ROC-AUC':>8} {'recall@fpr':>10} {'realised':>10} {'alerts':>9}",
        ]
        lines.extend(a.row() for a in self.arms)
        lines.append("")

        if len(self.arms) >= 2:
            base = self.arms[0]
            for arm in self.arms[1:]:
                delta = arm.recall_at_budget - base.recall_at_budget
                verdict = (
                    "buys recall" if delta > 0.01 else ("costs recall" if delta < -0.01 else "no change")
                )
                lines.append(
                    f"  {arm.name} vs {base.name}: {delta:+.4f} recall at the same budget - {verdict}"
                )
            lines.append("")

        if self.per_family:
            lines += ["  Per family, recall at the same budget:", ""]
            names = [a.name for a in self.arms]
            lines.append(f"  {'family':24}" + "".join(f"{n:>16}" for n in names))
            for family, scores in sorted(self.per_family.items(), key=lambda kv: -kv[1].get("n", 0)):
                row = f"  {family[:22]:24}"
                row += "".join(f"{scores.get(n, float('nan')):>16.4f}" for n in names)
                lines.append(f"{row}   (n={int(scores.get('n', 0)):,})")
            lines.append("")

        lines += [
            "  Caveat that belongs next to the numbers: this is CICIDS2017, a 2017 capture from a",
            "  testbed. What transfers is the RELATIVE comparison between arms on identical rows,",
            "  not the absolute recall.",
            "",
        ]
        return "\n".join(lines)


def _evaluate(
    name: str, scores: np.ndarray, y: np.ndarray, budget: float, notes: str = ""
) -> tuple[ArmResult, np.ndarray]:
    """Score one arm at the matched budget, and return its per-row decision.

    Rank selection rather than a quantile threshold. Forest probabilities put thousands of rows at
    exactly 0.0, so a quantile lands inside that tied block and flags a fraction of the budget -
    the first run of this experiment matched a 1% budget and realised 0.2% on one arm and 1.0% on
    another, which silently made them incomparable.
    """
    n_attacks = int((y == 1).sum())
    n_benign = int((y == 0).sum())
    flagged = budget_mod.flags_at_benign_budget(scores, y, int(round(budget * n_benign)))
    result = ArmResult(
        name=name,
        roc_auc=float(roc_auc_score(y, scores)) if len(set(y.tolist())) > 1 else float("nan"),
        recall_at_budget=float((flagged & (y == 1)).sum() / n_attacks) if n_attacks else float("nan"),
        threshold=float(scores[flagged].min()) if flagged.any() else float("inf"),
        realised_fpr=float((flagged & (y == 0)).sum() / n_benign) if n_benign else float("nan"),
        n_alerts=int(flagged.sum()),
        notes=notes,
    )
    return result, flagged


def _forest(X: pd.DataFrame, y: np.ndarray) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=200,
        max_depth=16,
        min_samples_leaf=5,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=SEED,
    ).fit(X.to_numpy(dtype=np.float32, na_value=0.0), y)


def _scores(model: RandomForestClassifier, X: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(X.to_numpy(dtype=np.float32, na_value=0.0))[:, 1]


def run(
    ds: Dataset,
    *,
    fpr_budget: float = 0.01,
    window_length: int = 16,
    max_train_rows: int | None = None,
    max_test_rows: int | None = None,
    train_stride: int = 4,
    test_stride: int = 3,
    epochs: int = 12,
    on_progress: Any = None,
) -> SequenceExperiment:
    """Run all three arms and compare them."""

    def say(message: str) -> None:
        if on_progress:
            on_progress(message)

    meta_train = ds.X_train.attrs.get("meta")
    meta_test = ds.X_test.attrs.get("meta")
    if meta_train is None or meta_test is None or "Timestamp" not in getattr(meta_train, "columns", []):
        raise ValueError(
            f"{ds.name} carries no entity/timestamp metadata. The sequence head needs CICIDS2017; "
            "UNSW-NB15's published split has no IPs and no timestamps at all."
        )

    config = windows.WindowConfig(length=window_length)
    y_train_all = np.asarray(ds.y_train).astype(int)
    y_test_all = np.asarray(ds.y_test).astype(int)

    say("building causal windows")
    seq_train = windows.build(
        ds.X_train,
        y_train_all,
        meta_train,
        config=config,
        max_rows=max_train_rows,
        keep_every=train_stride,
    )
    seq_test = windows.build(
        ds.X_test,
        y_test_all,
        meta_test,
        config=config,
        max_rows=max_test_rows,
        keep_every=test_stride,
    )
    say(f"  train {seq_train.summary().splitlines()[0]}")
    say(f"  test  {seq_test.summary().splitlines()[0]}")

    # Every arm is evaluated on exactly these rows, in exactly this order.
    train_rows, test_rows = seq_train.index, seq_test.index
    y_train, y_test = seq_train.y.astype(int), seq_test.y.astype(int)

    X_train = ds.X_train.iloc[train_rows].reset_index(drop=True)
    X_test = ds.X_test.iloc[test_rows].reset_index(drop=True)

    arms: list[ArmResult] = []
    per_arm_scores: dict[str, np.ndarray] = {}
    per_arm_flags: dict[str, np.ndarray] = {}

    say("arm 1: per-flow forest")
    flat = _forest(X_train, y_train)
    per_arm_scores["per-flow"] = _scores(flat, X_test)
    result, flags = _evaluate("per-flow", per_arm_scores["per-flow"], y_test, fpr_budget, "the floor")
    arms.append(result)
    per_arm_flags["per-flow"] = flags

    say("arm 2: per-flow + entity graph")
    graph_train = entity_graph.compute(meta_train).iloc[train_rows].reset_index(drop=True)
    graph_test = entity_graph.compute(meta_test).iloc[test_rows].reset_index(drop=True)
    Xg_train = pd.concat([X_train, graph_train], axis=1)
    Xg_test = pd.concat([X_test, graph_test], axis=1)
    graphed = _forest(Xg_train, y_train)
    per_arm_scores["per-flow + graph"] = _scores(graphed, Xg_test)
    result, flags = _evaluate(
        "per-flow + graph",
        per_arm_scores["per-flow + graph"],
        y_test,
        fpr_budget,
        "nine causal entity-graph features, no deep model",
    )
    arms.append(result)
    per_arm_flags["per-flow + graph"] = flags

    say("arm 3: sequence head (CNN + BiGRU)")
    from penumbra.models.sequence import SequenceConfig, SequenceDetector

    detector = SequenceDetector(SequenceConfig(epochs=epochs)).fit(seq_train)
    say(detector.history.summary())
    per_arm_scores["sequence"] = detector.score(seq_test)
    result, flags = _evaluate(
        "sequence",
        per_arm_scores["sequence"],
        y_test,
        fpr_budget,
        f"K={window_length} causal window per source host",
    )
    arms.append(result)
    per_arm_flags["sequence"] = flags

    # Per-family recall at each arm's own matched threshold, so the comparison stays budget-matched
    # inside every family rather than only in aggregate.
    families = pd.Series(ds.fam_test).reset_index(drop=True).iloc[test_rows].to_numpy()
    per_family: dict[str, dict[str, float]] = {}
    for family in sorted(set(families.tolist())):
        if family == "Normal":
            continue
        rows = families == family
        n = int(rows.sum())
        if n < 20:  # a recall estimate from fewer rows than this is not an estimate
            continue
        entry: dict[str, float] = {"n": float(n)}
        for arm in arms:
            entry[arm.name] = float(per_arm_flags[arm.name][rows].mean())
        per_family[str(family)] = entry

    return SequenceExperiment(
        dataset=ds.name,
        fpr_budget=fpr_budget,
        n_train=len(train_rows),
        n_test=len(test_rows),
        n_features=int(ds.X_train.shape[1]),
        window_length=window_length,
        arms=arms,
        training={
            "epochs_run": detector.history.epochs_run,
            "best_val_auc": detector.history.best_val_auc,
            "stopped_early": detector.history.stopped_early,
            "collapsed_epochs": detector.history.collapsed_epochs,
            "validation_split": detector.history.validation_split,
            "validation_minority": detector.history.validation_minority,
            "curve": detector.history.curve,
        },
        per_family=per_family,
    )


def write_report(experiment: SequenceExperiment, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(experiment.to_dict(), indent=2), encoding="utf-8")
    return path
