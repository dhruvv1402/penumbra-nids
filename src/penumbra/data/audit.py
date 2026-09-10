"""Data audit: what is the model actually learning from?

This runs BEFORE any model is trained, because it constrains what we are allowed to claim
afterwards. Four checks:

  1. single_feature_auc  - how well does ONE feature separate attack from benign? A feature that
                           alone reaches 0.90+ AUC is almost certainly a testbed artifact rather
                           than attack behaviour. On UNSW-NB15 we expect `sttl` to qualify: attack
                           and benign traffic were generated from different hosts, so time-to-live
                           encodes the generator.
  2. train_test_overlap  - rows appearing in both splits mean the test score is partly memorisation.
  3. shuffled_label_control - train on permuted labels; anything above chance is a pipeline leak.
  4. index_control       - train on row position alone; anything above chance is an ordering leak.

Checks 3 and 4 exist to catch a bug masquerading as a result. They are cheap and they eliminate a
whole class of "too good to be true" from suspicion.

See docs/EXPERIMENTS.md E3 and E4.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier

from penumbra.config import ARTIFACT_AUC_THRESHOLD
from penumbra.data import schema
from penumbra.data.loaders.base import Dataset
from penumbra.seeds import SEED

# =================================================================================================
# 1. Single-feature AUC
# =================================================================================================


@dataclass
class FeatureAudit:
    feature: str
    auc: float
    kind: str  # "numeric" | "categorical"
    n_unique: int

    @property
    def is_suspect(self) -> bool:
        return self.auc >= ARTIFACT_AUC_THRESHOLD


@dataclass
class ArtifactReport:
    dataset: str
    features: list[FeatureAudit] = field(default_factory=list)
    threshold: float = ARTIFACT_AUC_THRESHOLD

    @property
    def suspects(self) -> list[FeatureAudit]:
        return [f for f in self.features if f.is_suspect]

    @property
    def suspect_names(self) -> list[str]:
        return [f.feature for f in self.suspects]

    def top(self, n: int = 15) -> list[FeatureAudit]:
        return self.features[:n]

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "feature": f.feature,
                    "solo_auc": f.auc,
                    "kind": f.kind,
                    "n_unique": f.n_unique,
                    "suspect": f.is_suspect,
                }
                for f in self.features
            ]
        )

    def summary(self) -> str:
        lines = [
            f"Single-feature AUC audit - {self.dataset}",
            f"  a feature scoring >= {self.threshold:.2f} alone is treated as a suspected artifact",
            "",
            f"  {'feature':<24} {'solo AUC':>9}  {'kind':<12} {'unique':>7}",
            f"  {'-' * 24} {'-' * 9}  {'-' * 12} {'-' * 7}",
        ]
        for f in self.top(15):
            flag = "  <-- SUSPECT" if f.is_suspect else ""
            lines.append(f"  {f.feature:<24} {f.auc:>9.4f}  {f.kind:<12} {f.n_unique:>7,}{flag}")
        lines.append("")
        if self.suspects:
            lines.append(f"  {len(self.suspects)} suspected artifact(s): {', '.join(self.suspect_names)}")
            lines.append(
                "  Every headline metric must be reported with AND without these. The second "
                "number is the one we stand behind."
            )
        else:
            lines.append("  No single feature clears the threshold.")
        return "\n".join(lines)


def _numeric_solo_auc(x_tr: pd.Series, y_tr: pd.Series, x_te: pd.Series, y_te: pd.Series) -> float:
    """AUC from one numeric feature.

    A depth-3 tree rather than the raw value, because a raw-value AUC only sees monotonic
    relationships and would understate a feature that separates the classes through a band (which
    is exactly the shape a TTL artifact takes).
    """
    tr = x_tr.to_numpy(dtype=float).reshape(-1, 1)
    te = x_te.to_numpy(dtype=float).reshape(-1, 1)
    finite = np.isfinite(tr).ravel()
    if finite.sum() < 10 or y_tr[finite].nunique() < 2:
        return 0.5
    tree = DecisionTreeClassifier(max_depth=3, random_state=SEED)
    tree.fit(np.nan_to_num(tr[finite]), y_tr[finite])
    proba = tree.predict_proba(np.nan_to_num(te))[:, 1]
    return float(roc_auc_score(y_te, proba))


def _categorical_solo_auc(x_tr: pd.Series, y_tr: pd.Series, x_te: pd.Series, y_te: pd.Series) -> float:
    """AUC from one categorical feature, via a target rate fitted on TRAIN only.

    Fitting the encoding on train and scoring test is the honest version. Encoding with test labels
    included would manufacture the very leak we are looking for.
    """
    rates = y_tr.groupby(x_tr.to_numpy(), observed=True).mean()
    prior = float(y_tr.mean())
    scores = x_te.map(rates).fillna(prior).astype(float)
    if scores.nunique() < 2:
        return 0.5
    return float(roc_auc_score(y_te, scores))


def single_feature_auc(ds: Dataset) -> ArtifactReport:
    """Rank every feature by how well it separates attack from benign on its own."""
    audits: list[FeatureAudit] = []
    for col in ds.feature_names:
        kind = "categorical" if col in ds.categorical else "numeric"
        fn = _categorical_solo_auc if kind == "categorical" else _numeric_solo_auc
        auc = fn(ds.X_train[col], ds.y_train, ds.X_test[col], ds.y_test)
        # AUC below 0.5 is the same separating power with the sign flipped.
        auc = max(auc, 1.0 - auc)
        audits.append(FeatureAudit(feature=col, auc=auc, kind=kind, n_unique=int(ds.X_train[col].nunique())))
    audits.sort(key=lambda f: f.auc, reverse=True)
    return ArtifactReport(dataset=ds.name, features=audits)


# =================================================================================================
# 1b. Leaky values
# =================================================================================================
#
# Solo AUC is a RANKING metric, and it misses a specific and important artifact shape: a single
# feature VALUE that identifies a large subpopulation of one class perfectly, while the feature's
# remaining values carry no signal.
#
# UNSW-NB15 is exactly this case. `sttl == 31` accounts for ~70% of benign training rows and 0% of
# attack rows - an unambiguous fingerprint of the traffic generator - yet `sttl` scores only 0.76
# solo AUC and never trips the 0.90 artifact threshold.
#
# We found this by inspecting the crosstab after the AUC audit came back clean, which is itself the
# lesson: one artifact detector is not enough, and a negative result from one is not evidence of
# absence.


@dataclass
class LeakyValue:
    feature: str
    value: object
    support: int  # training rows carrying this value
    support_frac: float
    purity: float  # max class share among rows with this value
    majority_class: int

    def __str__(self) -> str:
        cls = "attack" if self.majority_class == 1 else "benign"
        return (
            f"{self.feature}={self.value!r}: {self.support_frac:.1%} of train rows, {self.purity:.1%} {cls}"
        )


@dataclass
class LeakyValueReport:
    dataset: str
    values: list[LeakyValue] = field(default_factory=list)
    min_support_frac: float = 0.01
    min_purity: float = 0.99

    @property
    def features(self) -> list[str]:
        seen: dict[str, None] = {}
        for v in self.values:
            seen.setdefault(v.feature, None)
        return list(seen)

    def summary(self) -> str:
        lines = [
            f"Leaky-value audit - {self.dataset}",
            f"  values held by >= {self.min_support_frac:.0%} of train rows that are "
            f">= {self.min_purity:.0%} one class",
            "",
        ]
        if not self.values:
            lines.append("  None found.")
            return "\n".join(lines)
        for v in self.values[:15]:
            lines.append(f"  {v}")
        lines.append("")
        lines.append(f"  {len(self.features)} feature(s) carry a leaky value: {', '.join(self.features)}")
        lines.append(
            "  A single value that fingerprints one class is a testbed artifact, not behaviour. "
            "Solo AUC does not detect this shape."
        )
        return "\n".join(lines)


def leaky_values(
    ds: Dataset,
    *,
    min_support_frac: float = 0.01,
    min_purity: float = 0.99,
    max_cardinality: int = 1000,
) -> LeakyValueReport:
    """Find feature values that fingerprint a class.

    Restricted to features with at most `max_cardinality` distinct values: a continuous feature has
    thousands of values each held by a handful of rows, and "this exact float appears twice, both
    attack" is noise rather than an artifact.
    """
    found: list[LeakyValue] = []
    n = len(ds.X_train)
    min_support = max(int(min_support_frac * n), 20)

    for col in ds.feature_names:
        series = ds.X_train[col]
        if series.nunique() > max_cardinality:
            continue
        grouped = ds.y_train.groupby(series.to_numpy(), observed=True).agg(["count", "mean"])
        for value, (count, mean) in grouped.iterrows():
            if count < min_support:
                continue
            purity = max(float(mean), 1.0 - float(mean))
            if purity >= min_purity:
                found.append(
                    LeakyValue(
                        feature=col,
                        value=value,
                        support=int(count),
                        support_frac=float(count) / n,
                        purity=purity,
                        majority_class=int(round(float(mean))),
                    )
                )

    found.sort(key=lambda v: v.support_frac * v.purity, reverse=True)
    return LeakyValueReport(
        dataset=ds.name,
        values=found,
        min_support_frac=min_support_frac,
        min_purity=min_purity,
    )


# =================================================================================================
# 2. Train/test overlap
# =================================================================================================


@dataclass
class OverlapReport:
    dataset: str
    n_train: int
    n_test: int
    exact_duplicates_within_train: int
    exact_duplicates_within_test: int
    test_rows_seen_in_train: int
    round_dp: int

    @property
    def leaked_fraction(self) -> float:
        return self.test_rows_seen_in_train / self.n_test if self.n_test else 0.0

    def summary(self) -> str:
        return "\n".join(
            [
                f"Train/test overlap - {self.dataset}  (features rounded to {self.round_dp} dp)",
                f"  duplicate rows within train : {self.exact_duplicates_within_train:,}",
                f"  duplicate rows within test  : {self.exact_duplicates_within_test:,}",
                f"  test rows also seen in train: {self.test_rows_seen_in_train:,} "
                f"({self.leaked_fraction:.2%} of test)",
                "",
                "  Rows crossing the split boundary make the test score partly memorisation.",
            ]
        )


def _row_hashes(X: pd.DataFrame, round_dp: int) -> pd.Series:
    """Stable per-row digest.

    Rounding first so that float noise below the rounding point does not hide a duplicate. The hash
    is over a canonical string rather than pandas' own hashing so the result does not depend on
    column dtype inference.
    """
    numeric = X.select_dtypes(include=[np.number]).round(round_dp)
    other = X.select_dtypes(exclude=[np.number]).astype(str)
    joined = pd.concat([numeric, other], axis=1)[X.columns]
    as_text = joined.astype(str).agg("\x1f".join, axis=1)
    return as_text.map(lambda s: hashlib.blake2b(s.encode("utf-8"), digest_size=16).hexdigest())


def train_test_overlap(ds: Dataset, *, round_dp: int = 6) -> OverlapReport:
    h_train = _row_hashes(ds.X_train, round_dp)
    h_test = _row_hashes(ds.X_test, round_dp)
    train_set = set(h_train)
    return OverlapReport(
        dataset=ds.name,
        n_train=len(h_train),
        n_test=len(h_test),
        exact_duplicates_within_train=int(h_train.duplicated().sum()),
        exact_duplicates_within_test=int(h_test.duplicated().sum()),
        test_rows_seen_in_train=int(h_test.isin(train_set).sum()),
        round_dp=round_dp,
    )


# =================================================================================================
# 3 & 4. Negative controls
# =================================================================================================


@dataclass
class ControlReport:
    dataset: str
    shuffled_label_auc: float
    index_only_auc: float
    majority_baseline_accuracy: float

    @property
    def passes(self) -> bool:
        # Both controls must sit near chance. 0.55 is a deliberately generous bound: anything above
        # it on shuffled labels or row position means information is reaching the model that should
        # not be.
        return self.shuffled_label_auc < 0.55 and self.index_only_auc < 0.55

    def summary(self) -> str:
        verdict = "PASS" if self.passes else "FAIL - investigate before trusting any result"
        return "\n".join(
            [
                f"Negative controls - {self.dataset}",
                f"  shuffled labels  AUC : {self.shuffled_label_auc:.4f}  (expect ~0.50)",
                f"  row index only   AUC : {self.index_only_auc:.4f}  (expect ~0.50)",
                f"  majority baseline acc: {self.majority_baseline_accuracy:.4f}",
                f"  -> {verdict}",
            ]
        )


def negative_controls(ds: Dataset, *, sample: int = 40_000) -> ControlReport:
    """Train on nonsense and confirm the pipeline cannot learn from it."""
    rng = np.random.default_rng(SEED)

    n = min(sample, len(ds.X_train))
    idx = rng.choice(len(ds.X_train), size=n, replace=False)
    X = ds.X_train.iloc[idx]
    y = ds.y_train.iloc[idx].to_numpy()

    # Numeric-only view: the controls test the pipeline, not the encoder.
    Xn = X.select_dtypes(include=[np.number]).to_numpy(dtype=float)
    Xn = np.nan_to_num(Xn, nan=0.0, posinf=0.0, neginf=0.0)

    tree = DecisionTreeClassifier(max_depth=6, random_state=SEED)

    y_shuffled = rng.permutation(y)
    shuffled_auc = float(cross_val_score(tree, Xn, y_shuffled, cv=3, scoring="roc_auc", n_jobs=-1).mean())

    index_only = np.arange(len(X), dtype=float).reshape(-1, 1)
    index_auc = float(cross_val_score(tree, index_only, y, cv=3, scoring="roc_auc", n_jobs=-1).mean())

    dummy = DummyClassifier(strategy="most_frequent")
    majority = float(cross_val_score(dummy, Xn, y, cv=3, scoring="accuracy").mean())

    return ControlReport(
        dataset=ds.name,
        shuffled_label_auc=max(shuffled_auc, 1 - shuffled_auc),
        index_only_auc=max(index_auc, 1 - index_auc),
        majority_baseline_accuracy=majority,
    )


# =================================================================================================
# Full audit
# =================================================================================================


@dataclass
class AuditReport:
    artifacts: ArtifactReport
    leaks: LeakyValueReport
    overlap: OverlapReport
    controls: ControlReport

    # Which dataset's verdict table to consult: "unsw" or "nslkdd".
    verdict_key: str = ""

    @property
    def candidates(self) -> list[str]:
        """Everything either detector flagged, before judgment.

        Neither detector alone is sufficient. On UNSW-NB15 the solo-AUC test finds nothing at all,
        while the leaky-value test finds `sttl=31` fingerprinting 22.5% of benign rows.
        """
        names = list(self.artifacts.suspect_names)
        for f in self.leaks.features:
            if f not in names:
                names.append(f)
        return names

    @property
    def quarantine(self) -> list[str]:
        """Candidates judged to be genuine artifacts.

        Only `artifact` verdicts. A feature the detector flagged but domain reasoning says is real
        behaviour - NSL-KDD's `srv_serror_rate=1.0`, which is what a SYN flood physically is -
        stays in. Excluding it would delete detection capability to improve a statistic.
        """
        return [f for f in self.candidates if schema.artifact_verdict(self.verdict_key, f)[0] == "artifact"]

    @property
    def unresolved(self) -> list[str]:
        return [f for f in self.candidates if schema.artifact_verdict(self.verdict_key, f)[0] == "unresolved"]

    def verdict_table(self) -> str:
        lines = [
            "Artifact verdicts - a detector surfaces candidates; deciding artifact vs signal is a",
            "question about network semantics, so each one is judged explicitly.",
            "",
        ]
        for f in self.candidates:
            verdict, reason = schema.artifact_verdict(self.verdict_key, f)
            lines.append(f"  [{verdict:<10}] {f}")
            lines.append(f"               {reason}")
        return "\n".join(lines)

    def summary(self) -> str:
        return "\n\n".join(
            [
                self.artifacts.summary(),
                self.leaks.summary(),
                self.overlap.summary(),
                self.controls.summary(),
                f"QUARANTINE: {', '.join(self.quarantine) or 'nothing'}\n"
                "  Every headline metric is reported with and without these features.",
            ]
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "dataset": self.artifacts.dataset,
            "artifact_threshold": self.artifacts.threshold,
            "suspected_artifacts": self.artifacts.suspect_names,
            "candidates": self.candidates,
            "quarantine": self.quarantine,
            "unresolved": self.unresolved,
            "verdicts": {
                f: {
                    "verdict": schema.artifact_verdict(self.verdict_key, f)[0],
                    "reason": schema.artifact_verdict(self.verdict_key, f)[1],
                }
                for f in self.candidates
            },
            "leaky_values": [
                {
                    "feature": v.feature,
                    "value": str(v.value),
                    "support": v.support,
                    "support_frac": round(v.support_frac, 6),
                    "purity": round(v.purity, 6),
                    "majority_class": v.majority_class,
                }
                for v in self.leaks.values
            ],
            "single_feature_auc": {f.feature: round(f.auc, 6) for f in self.artifacts.features},
            "overlap": {
                "n_train": self.overlap.n_train,
                "n_test": self.overlap.n_test,
                "duplicates_within_train": self.overlap.exact_duplicates_within_train,
                "duplicates_within_test": self.overlap.exact_duplicates_within_test,
                "test_rows_seen_in_train": self.overlap.test_rows_seen_in_train,
                "leaked_fraction": round(self.overlap.leaked_fraction, 6),
            },
            "controls": {
                "shuffled_label_auc": round(self.controls.shuffled_label_auc, 6),
                "index_only_auc": round(self.controls.index_only_auc, 6),
                "majority_baseline_accuracy": round(self.controls.majority_baseline_accuracy, 6),
                "passes": self.controls.passes,
            },
        }


def run_audit(ds: Dataset, *, verdict_key: str = "") -> AuditReport:
    """Run all four checks.

    `verdict_key` selects which dataset's artifact-verdict table to consult ("unsw" / "nslkdd").
    Inferred from the dataset name when not given.
    """
    if not verdict_key:
        verdict_key = "nslkdd" if "nsl" in ds.name.lower() else "unsw"
    return AuditReport(
        artifacts=single_feature_auc(ds),
        leaks=leaky_values(ds),
        overlap=train_test_overlap(ds),
        controls=negative_controls(ds),
        verdict_key=verdict_key,
    )
