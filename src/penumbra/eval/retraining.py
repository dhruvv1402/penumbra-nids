"""From promoted analyst verdicts to a gated challenger.

Headless on purpose: the CLI's `retrain` wires storage and the registry around this, and the tests
drive it with a synthetic dataset. The path from "a senior promoted a verdict" to "a model that may
be promoted" is the one the whole feedback-loop threat model is about, so it has to be testable
without a 126k-row dataset on disk.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any

import pandas as pd

from penumbra.data.loaders.base import Dataset
from penumbra.eval import canary
from penumbra.models.detector import PenumbraDetector


@dataclass
class FeedbackSummary:
    rows_offered: int
    rows_usable: int
    attack: int
    benign: int
    approvers: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "feedback_rows": self.rows_usable,
            "feedback_offered": self.rows_offered,
            "feedback_attack": self.attack,
            "feedback_benign": self.benign,
            "approvers": self.approvers,
        }


def feedback_frame(rows: Sequence[dict[str, Any]], ds: Dataset) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Promoted rows as a frame shaped exactly like the training features.

    Rows whose features do not cover this dataset's columns are dropped, not imputed: an alert
    scored by a different model on a different dataset is not a training example for this one.
    Numeric columns are coerced (None -> NaN, which the pipeline's imputer handles); categorical
    columns stay strings.
    """
    usable = [r for r in rows if set(ds.feature_names) <= set(r["features"])]
    frame = pd.DataFrame([r["features"] for r in usable], columns=ds.feature_names)
    for col in ds.feature_names:
        if col in ds.categorical:
            frame[col] = frame[col].astype(str)
        else:
            frame[col] = pd.to_numeric(frame[col], errors="coerce").astype(float)
    return frame, usable


# A confirmed attack the model could not name (a novelty or abstention alert). Labelling it with
# the benign family would contradict its binary label and teach the family classifier that attacks
# look normal; inventing a family would be worse. It gets an explicit class of its own.
CONFIRMED_UNNAMED = "confirmed_unnamed"


def benign_family_label(ds: Dataset) -> str:
    """The dataset's own spelling of the benign family ('normal' in NSL-KDD, 'Normal' elsewhere)."""
    benign = ds.fam_train[ds.y_train == 0]
    return str(benign.mode().iloc[0]) if len(benign) else "normal"


def family_for(row: dict[str, Any], benign_family: str) -> str:
    if int(row["label"]) == 0:
        return benign_family
    family = row.get("family")
    return str(family) if family else CONFIRMED_UNNAMED


def augment(ds: Dataset, rows: Sequence[dict[str, Any]]) -> tuple[Dataset, FeedbackSummary]:
    frame, usable = feedback_frame(rows, ds)
    labels = pd.Series([int(r["label"]) for r in usable], dtype=int)
    benign_family = benign_family_label(ds)
    families = pd.Series([family_for(r, benign_family) for r in usable], dtype=object)
    X_train = pd.concat(
        [ds.X_train, frame.astype(ds.X_train.dtypes.to_dict(), errors="ignore")], ignore_index=True
    )
    augmented = replace(
        ds,
        X_train=X_train,
        y_train=pd.concat([ds.y_train, labels], ignore_index=True),
        fam_train=pd.concat([ds.fam_train, families], ignore_index=True),
    )
    summary = FeedbackSummary(
        rows_offered=len(rows),
        rows_usable=len(usable),
        attack=int(labels.sum()),
        benign=int(len(labels) - labels.sum()),
        approvers=sorted({str(r.get("approver")) for r in usable}),
    )
    return augmented, summary


def challenge(
    champion: PenumbraDetector,
    augmented: Dataset,
    canary_X: pd.DataFrame,
    canary_y: pd.Series,
    canary_families: pd.Series,
    *,
    model_name: str = "rf",
    policy: canary.GatePolicy | None = None,
) -> tuple[PenumbraDetector, canary.GateResult]:
    """Fit a challenger on the augmented data and gate it against the champion on the canary."""
    challenger = PenumbraDetector(target_fpr=champion.target_fpr).fit(augmented, model_name=model_name)
    result = canary.gate(
        canary.evaluate(champion, canary_X, canary_y, canary_families),
        canary.evaluate(challenger, canary_X, canary_y, canary_families),
        policy,
    )
    return challenger, result
