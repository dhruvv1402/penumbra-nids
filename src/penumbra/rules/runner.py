"""End-to-end rule mining: fit, mine, validate, emit.

Three splits, and the separation between them is the whole point:

- **fit** — a forest is grown here, and nowhere else. Its leaves are the candidate rules.
- **holdout** — carved out of *training* data the trees never saw. Every precision number attached
  to a rule comes from here. Using the test set would make the rules and the model's own reported
  metrics share a denominator, and the rule pack would stop being independent evidence.
- **test** — touched once, at the end, only to ask whether the surviving set still holds up. It
  changes no threshold and discards no rule.

The run is done twice: once with the quarantined features available, once without. The pair is the
finding. A rule resting on `sttl` will validate at 1.000 precision on held-out data from the same
testbed and detect nothing on a real network, because the thing it learned is which machine
generated the packet. Reporting only the second number hides how much of the first was that.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from penumbra.data import schema
from penumbra.data.loaders.base import Dataset
from penumbra.rules import emit, mining
from penumbra.seeds import SEED

# Categorical values rarer than this are folded away rather than given an indicator column. UNSW's
# `proto` has ~133 values, almost all with a handful of rows; a rule built on one of them is an
# anecdote with a threshold on it.
MIN_CATEGORY_ROWS = 200

NEWLINE = chr(10)


def rule_frame(X: pd.DataFrame, *, categories: dict[str, list[str]] | None = None) -> pd.DataFrame:
    """Reshape a feature frame into something a mined rule can be written against.

    Numeric columns pass through untouched. Categoricals become 0/1 indicators named `proto=tcp`,
    which is what lets a mined condition round-trip back to `proto == "tcp"` in KQL instead of
    surfacing as a threshold on an anonymous one-hot index.
    """
    numeric = X.select_dtypes(include="number")
    object_cols = [c for c in X.columns if c not in numeric.columns]

    frames: list[pd.DataFrame] = [numeric.reset_index(drop=True)]
    for col in object_cols:
        values = X[col].astype("string").fillna("NA")
        keep = (
            categories[col]
            if categories is not None
            else sorted(values.value_counts().loc[lambda s: s >= MIN_CATEGORY_ROWS].index.tolist())
        )
        if not keep:
            continue
        block = pd.DataFrame(
            {f"{col}={v}": (values == v).astype("int8").to_numpy() for v in keep},
        )
        frames.append(block)

    return pd.concat(frames, axis=1)


def frame_categories(X: pd.DataFrame) -> dict[str, list[str]]:
    """The category vocabulary to freeze across splits.

    Recomputing it per split would give train and holdout different columns, and a rule would then
    silently match nothing on a frame that lacks its column.
    """
    out: dict[str, list[str]] = {}
    numeric = set(X.select_dtypes(include="number").columns)
    for col in X.columns:
        if col in numeric:
            continue
        values = X[col].astype("string").fillna("NA").value_counts()
        out[col] = sorted(values.loc[lambda s: s >= MIN_CATEGORY_ROWS].index.tolist())
    return out


@dataclass
class MiningRun:
    """One arm of the experiment: rules mined with or without the quarantined features."""

    label: str
    ruleset: mining.RuleSet
    holdout_coverage: dict[str, float] = field(default_factory=dict)
    test_coverage: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            **self.ruleset.to_dict(),
            "holdout_coverage": self.holdout_coverage,
            "test_coverage": self.test_coverage,
        }


@dataclass
class MiningReport:
    dataset: str
    quarantined: list[str]
    n_fit: int
    n_holdout: int
    n_test: int
    with_artifacts: MiningRun
    without_artifacts: MiningRun

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "quarantined": self.quarantined,
            "n_fit": self.n_fit,
            "n_holdout": self.n_holdout,
            "n_test": self.n_test,
            "with_artifacts": self.with_artifacts.to_dict(),
            "without_artifacts": self.without_artifacts.to_dict(),
        }

    def summary(self) -> str:
        keep = self.without_artifacts
        drop = self.with_artifacts
        if not self.quarantined:
            # Both arms are the same run. Printing them side by side would imply a comparison was
            # made and came out even, which is a different claim from "there was nothing to drop".
            return NEWLINE.join(
                [
                    keep.ruleset.summary(),
                    "",
                    "  Nothing was quarantined on this dataset: every leaky value the audit found",
                    "  has a recorded verdict of `signal`. A SYN flood genuinely produces a SYN",
                    "  error rate of 1.0, so a rule resting on that is the detection working.",
                    "",
                    f"  set coverage, holdout: recall {keep.holdout_coverage.get('recall', 0):.4f} "
                    f"at precision {keep.holdout_coverage.get('precision', 0):.4f}",
                    f"  set coverage, test:    recall {keep.test_coverage.get('recall', 0):.4f} "
                    f"at precision {keep.test_coverage.get('precision', 0):.4f}",
                    "",
                ]
            )
        lines = [
            keep.ruleset.summary(),
            "",
            "=" * 88,
            "  WHAT THE QUARANTINE COST",
            "=" * 88,
            "",
            f"  Quarantined: {', '.join(self.quarantined) or 'nothing'}",
            "",
            f"  {'':26} {'with artifacts':>16} {'without':>16}",
            f"  {'rules surviving':26} {len(drop.ruleset.survivors):>16,} {len(keep.ruleset.survivors):>16,}",
            f"  {'set recall (holdout)':26} {drop.holdout_coverage.get('recall', 0):>16.4f} "
            f"{keep.holdout_coverage.get('recall', 0):>16.4f}",
            f"  {'set precision (holdout)':26} {drop.holdout_coverage.get('precision', 0):>16.4f} "
            f"{keep.holdout_coverage.get('precision', 0):>16.4f}",
            f"  {'set recall (test)':26} {drop.test_coverage.get('recall', 0):>16.4f} "
            f"{keep.test_coverage.get('recall', 0):>16.4f}",
            f"  {'set precision (test)':26} {drop.test_coverage.get('precision', 0):>16.4f} "
            f"{keep.test_coverage.get('precision', 0):>16.4f}",
            "",
            "  The left column is the number we could have reported. The right column is the one we",
            "  stand behind. Both were measured on data the trees never saw - which is exactly why",
            "  held-out validation alone does not catch a testbed artifact.",
            "",
        ]
        return NEWLINE.join(lines)


def _coverage(ruleset: mining.RuleSet, X: pd.DataFrame, y: np.ndarray) -> dict[str, float]:
    return ruleset.coverage(X, y)


def run(
    ds: Dataset,
    *,
    dataset_key: str,
    holdout_fraction: float = 0.35,
    n_estimators: int = 300,
    max_depth: int = 8,
    min_support: int = 50,
    min_purity: float = 0.98,
    max_conditions: int = 4,
    min_precision: float = 0.98,
    max_rules: int = 2000,
    on_progress: Any = None,
) -> MiningReport:
    """Mine, validate and compare both arms.

    `max_rules` defaults high on purpose. Capping it would truncate the arm that mines more paths -
    which is the arm with the artifacts in it - and the comparison would then be measuring the cap
    rather than the quarantine.
    """

    def say(message: str) -> None:
        if on_progress:
            on_progress(message)

    categories = frame_categories(ds.X_train)
    X_all = rule_frame(ds.X_train, categories=categories)
    y_all = np.asarray(ds.y_train).astype(int)

    X_fit, X_hold, y_fit, y_hold = train_test_split(
        X_all, y_all, test_size=holdout_fraction, random_state=SEED, stratify=y_all
    )
    X_test = rule_frame(ds.X_test, categories=categories)
    y_test = np.asarray(ds.y_test).astype(int)
    # Column order must match; a missing indicator would make every rule on it match nothing.
    X_test = X_test.reindex(columns=X_all.columns, fill_value=0)

    say(f"fitting forest on {len(X_fit):,} rows, {X_all.shape[1]} rule-visible features")
    forest = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_support,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=SEED,
    ).fit(X_fit, y_fit)

    quarantine = schema.quarantined(dataset_key)
    feature_names = list(X_all.columns)

    arms: dict[str, MiningRun] = {}
    for label, excluded in (("with_artifacts", set()), ("without_artifacts", quarantine)):
        candidates = mining.mine(
            forest,
            feature_names,
            min_support=min_support,
            min_purity=min_purity,
            max_depth=max_conditions,
            max_rules=max_rules,
            exclude_features=excluded or None,
        )
        ruleset = mining.validate(candidates, X_hold, y_hold, min_precision=min_precision)
        say(
            f"{label}: {len(candidates):,} candidates "
            f"({candidates.n_artifact_excluded:,} artifact-excluded) "
            f"-> {len(ruleset.survivors):,} survivors"
        )
        arms[label] = MiningRun(
            label=label,
            ruleset=ruleset,
            holdout_coverage=_coverage(ruleset, X_hold, y_hold),
            test_coverage=_coverage(ruleset, X_test, y_test),
        )

    return MiningReport(
        dataset=ds.name,
        quarantined=sorted(quarantine),
        n_fit=len(X_fit),
        n_holdout=len(X_hold),
        n_test=len(X_test),
        with_artifacts=arms["with_artifacts"],
        without_artifacts=arms["without_artifacts"],
    )


def write_artifacts(
    report: MiningReport,
    *,
    report_path: Path,
    kql_path: Path,
    sigma_dir: Path | None = None,
) -> dict[str, Any]:
    """Persist the report, the KQL pack and any Sigma rules.

    Only the quarantined-free arm is emitted as runnable content. The other arm exists to be
    reported, not deployed - shipping a KQL pack built on `sttl` would hand a detection engineer a
    rule that fires on an operating system.
    """
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")

    ruleset = report.without_artifacts.ruleset
    kql_path.parent.mkdir(parents=True, exist_ok=True)
    kql_path.write_text(emit.to_kql_pack(ruleset), encoding="utf-8")

    docs, skipped = emit.to_sigma_pack(ruleset)
    written: list[str] = []
    if sigma_dir is not None and docs:
        sigma_dir.mkdir(parents=True, exist_ok=True)
        for i, doc in enumerate(docs):
            path = sigma_dir / f"penumbra_mined_{i:03d}.yml"
            path.write_text(doc, encoding="utf-8")
            written.append(str(path))

    return {
        "report": str(report_path),
        "kql": str(kql_path),
        "sigma_written": written,
        "sigma_skipped": skipped,
    }
