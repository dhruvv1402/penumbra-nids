"""Mining human-readable detection rules out of the trained forest.

The premise of the brief is that a signature IDS misses novel attacks. The usual response is to
replace it. This is the other direction: **the model mines the signature, and the existing SIEM
enforces it** — for attacks no vendor has shipped a rule for yet.

A decision path from root to a pure leaf is already a rule. `sttl <= 62 AND ct_dst_ltm > 16` is a
conjunction of thresholds that any SIEM can evaluate without a model, a Python runtime, or a GPU.
Extracting those turns a black box into something a detection engineer can read, argue with, and
own.

## Every mined rule is validated before it is emitted

An unvalidated mined rule is worse than no rule: it carries the authority of "the model found this"
with none of the evidence. So each candidate is scored on **held-out data the tree never saw**, and
its own precision, recall and false-positive count travel with it. Rules below the purity floor are
discarded, and the ones that survive say what they cost.

## KQL is primary; Sigma only where it can be honest

Sigma's taxonomy is **log-based** — process creation, firewall, DNS, web. It has no field for
`sload`, `ct_srv_src` or `tcprtt`, so emitting `sload > 1400000` as a Sigma rule produces something
no SIEM can evaluate. KQL against a flow table can express all of it. Sigma is emitted only for the
subset of conditions that map to fields Sigma actually defines.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.tree import _tree

# Fields Sigma's network taxonomy actually defines. Anything outside this cannot be expressed as a
# Sigma rule honestly, however tempting it is to emit one.
SIGMA_MAPPABLE: dict[str, str] = {
    "dst_port": "DestinationPort",
    "dsport": "DestinationPort",
    "Dst Port": "DestinationPort",
    "src_port": "SourcePort",
    "proto": "Protocol",
    "protocol_type": "Protocol",
    "service": "Application",
    "state": "ConnectionState",
    "sbytes": "BytesSent",
    "src_bytes": "BytesSent",
    "dbytes": "BytesReceived",
    "dst_bytes": "BytesReceived",
    "duration": "Duration",
    "dur": "Duration",
}


@dataclass(frozen=True)
class Condition:
    feature: str
    operator: str  # "<=" | ">"
    threshold: float

    def __str__(self) -> str:
        cat = self.categorical
        if cat is not None:
            column, value = cat
            return f"{column} {'==' if self.is_positive else '!='} {value!r}"
        return f"{self.feature} {self.operator} {self.threshold:.6g}"

    @property
    def categorical(self) -> tuple[str, str] | None:
        """`(column, value)` when this is an indicator on a categorical value, else None.

        Categoricals reach the forest as 0/1 indicator columns named `proto=tcp`. A split on one is
        always at 0.5, so the tree's `proto=tcp > 0.5` is really `proto == "tcp"` - and that is what
        a SIEM query and a detection engineer both need to see.
        """
        column, sep, value = self.feature.partition("=")
        return (column, value) if sep else None

    @property
    def is_positive(self) -> bool:
        """For an indicator condition: does it assert the value, or negate it?"""
        return self.operator == ">"

    @property
    def sigma_field(self) -> str | None:
        cat = self.categorical
        return SIGMA_MAPPABLE.get(cat[0] if cat else self.feature)


def _drop_implied_negations(conditions: list[Condition]) -> list[Condition]:
    """Remove categorical negations that a positive on the same column already implies.

    A tree can descend through `proto != "udp"` and later land on `proto == "mobile"`. Both are in
    the path, but the second makes the first vacuous - and a detection engineer reading
    `proto != "udp" and proto == "mobile"` will reasonably assume the negation is load-bearing and
    spend time working out why.
    """
    asserted = {c.categorical[0] for c in conditions if c.categorical and c.is_positive}
    return [
        c for c in conditions if not (c.categorical and not c.is_positive and c.categorical[0] in asserted)
    ]


@dataclass
class MinedRule:
    """One decision path, plus what it actually costs."""

    conditions: list[Condition]
    train_support: int
    train_purity: float
    predicted_class: int

    # Filled by validate(); None means the rule has not been validated and must not be emitted.
    holdout_support: int | None = None
    holdout_precision: float | None = None
    holdout_recall: float | None = None
    holdout_false_positives: int | None = None

    @property
    def validated(self) -> bool:
        return self.holdout_precision is not None

    @property
    def depth(self) -> int:
        return len(self.conditions)

    @property
    def features(self) -> list[str]:
        seen: dict[str, None] = {}
        for c in self.conditions:
            seen.setdefault(c.feature, None)
        return list(seen)

    @property
    def sigma_expressible(self) -> bool:
        """True only if EVERY condition maps to a field Sigma defines.

        A partial translation is a different rule, not the same one with fewer clauses.
        """
        return bool(self.conditions) and all(c.sigma_field for c in self.conditions)

    def matches(self, X: pd.DataFrame) -> np.ndarray:
        mask = np.ones(len(X), dtype=bool)
        for c in self.conditions:
            if c.feature not in X.columns:
                return np.zeros(len(X), dtype=bool)
            values = pd.to_numeric(X[c.feature], errors="coerce").to_numpy(dtype=float)
            with np.errstate(invalid="ignore"):
                mask &= (values <= c.threshold) if c.operator == "<=" else (values > c.threshold)
        return mask

    def simplify(self) -> MinedRule:
        """Collapse redundant conditions on the same feature.

        A root-to-leaf path can split on one feature repeatedly, producing
        `ct_dst_sport_ltm > 1.5 AND ct_dst_sport_ltm > 2.5`. Both are true whenever the tighter one
        is, so the rule is the tighter one - and a detection engineer reading the looser clause
        would reasonably wonder what it is doing there.

        For `>` keep the largest threshold; for `<=` keep the smallest. A feature carrying both
        becomes a range, which is meaningful and is kept.
        """
        upper: dict[str, float] = {}
        lower: dict[str, float] = {}
        for c in self.conditions:
            if c.operator == ">":
                lower[c.feature] = max(lower.get(c.feature, float("-inf")), c.threshold)
            else:
                upper[c.feature] = min(upper.get(c.feature, float("inf")), c.threshold)

        merged: list[Condition] = []
        for feature in self.features:
            if feature in lower:
                merged.append(Condition(feature, ">", lower[feature]))
            if feature in upper:
                merged.append(Condition(feature, "<=", upper[feature]))

        merged = _drop_implied_negations(merged)

        return MinedRule(
            conditions=merged,
            train_support=self.train_support,
            train_purity=self.train_purity,
            predicted_class=self.predicted_class,
            holdout_support=self.holdout_support,
            holdout_precision=self.holdout_precision,
            holdout_recall=self.holdout_recall,
            holdout_false_positives=self.holdout_false_positives,
        )

    def depends_on(self, features: set[str]) -> bool:
        """True if any condition references one of these features, one-hot expansions included.

        A rule built on `sttl` and one built on `proto_tcp` both inherit whatever is wrong with the
        source column, so the match is on the prefix as well as the exact name.
        """
        for c in self.conditions:
            if c.feature in features:
                return True
            if any(c.feature.startswith(f"{f}_") for f in features):
                return True
            # A column-level quarantine covers every indicator derived from that column.
            cat = c.categorical
            if cat is not None and cat[0] in features:
                return True
        return False

    def signature(self) -> tuple[tuple[str, str, float], ...]:
        """Canonical form, for de-duplicating paths that differ only in ordering."""
        return tuple(sorted((c.feature, c.operator, round(c.threshold, 6)) for c in self.conditions))

    def describe(self) -> str:
        head = " AND ".join(str(c) for c in self.conditions)
        if not self.validated:
            return f"  [UNVALIDATED] {head}"
        return (
            f"  {head}\n"
            f"      held out: precision {self.holdout_precision:.3f}  "
            f"recall {self.holdout_recall:.4f}  "
            f"matches {self.holdout_support:,}  false positives {self.holdout_false_positives:,}"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "conditions": [
                {"feature": c.feature, "op": c.operator, "threshold": c.threshold} for c in self.conditions
            ],
            "train_support": self.train_support,
            "train_purity": self.train_purity,
            "holdout_support": self.holdout_support,
            "holdout_precision": self.holdout_precision,
            "holdout_recall": self.holdout_recall,
            "holdout_false_positives": self.holdout_false_positives,
            "sigma_expressible": self.sigma_expressible,
        }


# =================================================================================================
# Extraction
# =================================================================================================


def extract_from_tree(
    tree: Any,
    feature_names: list[str],
    *,
    min_support: int = 50,
    min_purity: float = 0.98,
    max_depth: int = 4,
    target_class: int = 1,
) -> list[MinedRule]:
    """Walk one tree and return the paths to leaves that are pure enough to be worth a rule.

    `max_depth` is a readability constraint, not a statistical one. A seven-condition conjunction may
    be perfectly predictive and no detection engineer will ever adopt it.
    """
    t = tree.tree_
    rules: list[MinedRule] = []

    def walk(node: int, path: list[Condition]) -> None:
        if len(path) > max_depth:
            return

        if t.children_left[node] == _tree.TREE_LEAF:
            counts = t.value[node][0]
            total = float(counts.sum())
            if total <= 0 or target_class >= len(counts):
                return
            # sklearn normalises value[] for some estimators, so recover counts via weighted samples.
            support = int(round(t.weighted_n_node_samples[node]))
            purity = float(counts[target_class] / total)
            if support >= min_support and purity >= min_purity:
                rules.append(
                    MinedRule(
                        conditions=list(path),
                        train_support=support,
                        train_purity=purity,
                        predicted_class=target_class,
                    )
                )
            return

        name = feature_names[t.feature[node]]
        threshold = float(t.threshold[node])
        walk(t.children_left[node], [*path, Condition(name, "<=", threshold)])
        walk(t.children_right[node], [*path, Condition(name, ">", threshold)])

    walk(0, [])
    return rules


@dataclass
class Candidates:
    """Mined candidates, plus what was discarded and why.

    A bare list would let the artifact-exclusion count vanish, and "47 rules" reads very differently
    from "47 rules, 145 discarded because they rest on quarantined features".
    """

    rules: list[MinedRule] = field(default_factory=list)
    n_artifact_excluded: int = 0
    n_duplicates: int = 0

    def __len__(self) -> int:
        return len(self.rules)

    def __iter__(self) -> Any:
        return iter(self.rules)


def mine(
    forest: Any,
    feature_names: list[str],
    *,
    min_support: int = 50,
    min_purity: float = 0.98,
    max_depth: int = 4,
    target_class: int = 1,
    max_rules: int = 200,
    exclude_features: set[str] | None = None,
) -> Candidates:
    """Mine candidate rules across every tree, de-duplicated.

    A forest of 300 trees produces thousands of near-identical paths. De-duplicating on the
    canonical condition set collapses them; without it the output is unreadable and the "we mined N
    rules" number is meaningless.
    """
    estimators = getattr(forest, "estimators_", None)
    if estimators is None:
        estimators = [forest]

    seen: set[tuple[tuple[str, str, float], ...]] = set()
    out: list[MinedRule] = []
    excluded = exclude_features or set()
    n_artifact_rules = 0
    n_duplicates = 0

    for tree in estimators:
        for raw in extract_from_tree(
            tree,
            feature_names,
            min_support=min_support,
            min_purity=min_purity,
            max_depth=max_depth,
            target_class=target_class,
        ):
            # Simplify BEFORE de-duplicating, so two paths that differ only in a redundant clause
            # collapse to the same signature instead of both surviving.
            rule = raw.simplify()

            # De-duplicate FIRST. 300 trees rediscover the same path many times over, and counting
            # exclusions before the de-dup would report "110 artifact rules excluded" against "101
            # candidates kept" - two numbers in different units, which is worse than no number.
            sig = rule.signature()
            if sig in seen:
                n_duplicates += 1
                continue
            seen.add(sig)

            # A rule resting on a quarantined feature encodes the testbed, not the behaviour. It
            # will validate beautifully on held-out data from the same testbed and transfer to
            # nothing, which is exactly the failure this project exists to avoid reproducing.
            if excluded and rule.depends_on(excluded):
                n_artifact_rules += 1
                continue

            out.append(rule)

    # Prefer the rules that cover the most traffic; a pure rule matching 51 flows is not worth
    # shipping to a SIEM.
    out.sort(key=lambda r: (r.train_support, r.train_purity), reverse=True)
    return Candidates(
        rules=out[:max_rules],
        n_artifact_excluded=n_artifact_rules,
        n_duplicates=n_duplicates,
    )


# =================================================================================================
# Validation
# =================================================================================================


@dataclass
class RuleSet:
    rules: list[MinedRule] = field(default_factory=list)
    min_precision: float = 0.98
    n_candidates: int = 0
    n_holdout: int = 0
    n_artifact_excluded: int = 0

    @property
    def survivors(self) -> list[MinedRule]:
        return [r for r in self.rules if r.validated and (r.holdout_precision or 0) >= self.min_precision]

    def coverage(self, X: pd.DataFrame, y: np.ndarray) -> dict[str, float]:
        """What the surviving rules catch as a set, without the model."""
        if not self.survivors:
            return {"recall": 0.0, "precision": float("nan"), "n_matched": 0}
        matched = np.zeros(len(X), dtype=bool)
        for rule in self.survivors:
            matched |= rule.matches(X)
        y = np.asarray(y).astype(int)
        tp = int(np.sum(matched & (y == 1)))
        return {
            "recall": tp / max(int((y == 1).sum()), 1),
            "precision": tp / max(int(matched.sum()), 1),
            "n_matched": int(matched.sum()),
        }

    def summary(self, top: int = 8) -> str:
        lines = [
            "=" * 88,
            "  MINED DETECTION RULES",
            "=" * 88,
            "",
            f"  {self.n_candidates:,} candidate paths -> {len(self.survivors):,} survived validation",
            f"  floor: precision >= {self.min_precision:.2f} on {self.n_holdout:,} held-out rows",
            "",
            f"  {self.n_artifact_excluded:,} further paths were discarded BEFORE validation because",
            "  they rest on features our own audit quarantined as testbed artifacts. Those rules",
            "  would have validated perfectly on held-out data from the same testbed and",
            "  transferred to nothing.",
            "",
            "  Every rule below was scored on data the tree never saw. An unvalidated mined rule",
            "  carries the authority of 'the model found this' with none of the evidence.",
            "",
        ]
        for rule in self.survivors[:top]:
            lines.append(rule.describe())
            lines.append("")
        sigma = sum(1 for r in self.survivors if r.sigma_expressible)
        lines.append(
            f"  {sigma} of {len(self.survivors)} are expressible as Sigma rules; the rest reference "
            f"flow\n  features Sigma's log-based taxonomy does not define, and are emitted as KQL only."
        )
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_candidates": self.n_candidates,
            "n_artifact_excluded": self.n_artifact_excluded,
            "n_survivors": len(self.survivors),
            "min_precision": self.min_precision,
            "n_holdout": self.n_holdout,
            "rules": [r.to_dict() for r in self.survivors],
        }


def validate(
    rules: list[MinedRule] | Candidates,
    X_holdout: pd.DataFrame,
    y_holdout: np.ndarray,
    *,
    min_precision: float = 0.98,
    min_holdout_support: int = 20,
) -> RuleSet:
    """Score every candidate on held-out data and keep only what survives.

    This is the step that separates a mined rule from a plausible-looking conjunction. A rule that
    was pure on the training split and is 60% precise on held-out data is not a detection, it is
    an overfit path.
    """
    candidates = rules if isinstance(rules, Candidates) else Candidates(rules=list(rules))
    rules = candidates.rules

    y = np.asarray(y_holdout).astype(int)
    n_attacks = int((y == 1).sum())

    for rule in rules:
        matched = rule.matches(X_holdout)
        n_matched = int(matched.sum())
        tp = int(np.sum(matched & (y == 1)))

        rule.holdout_support = n_matched
        rule.holdout_false_positives = n_matched - tp
        rule.holdout_precision = (tp / n_matched) if n_matched else 0.0
        rule.holdout_recall = (tp / n_attacks) if n_attacks else 0.0

        # A rule matching three rows has a precision estimate built from three rows.
        if n_matched < min_holdout_support:
            rule.holdout_precision = 0.0

    result = RuleSet(
        rules=rules,
        min_precision=min_precision,
        n_candidates=len(rules),
        n_holdout=len(X_holdout),
        n_artifact_excluded=candidates.n_artifact_excluded,
    )
    result.rules.sort(key=lambda r: (r.holdout_precision or 0, r.holdout_recall or 0), reverse=True)
    return result
