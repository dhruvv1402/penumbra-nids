"""Emitting mined rules as KQL and Sigma.

KQL is the primary target because it can express every condition the forest produces. Sigma is
emitted only for rules whose conditions all map to fields Sigma's log-based taxonomy actually
defines — a partial translation is a different rule, not a shorter one.

Every emitted rule carries its held-out precision and false-positive count in a comment. A detection
engineer inheriting this needs to know what it costs before they enable it, and burying that in a
separate report is how a rule gets enabled without it.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from penumbra.rules.mining import Condition, MinedRule, RuleSet

# Column names in the Sentinel custom table the ASIM parser reads from.
KQL_TABLE = "PenumbraFlows_CL"

NEWLINE = chr(10)


def _column(name: str) -> str:
    # Column names in Log Analytics cannot contain spaces; the ingestion DCR normalises them.
    return name.replace(" ", "_").replace("-", "_")


def _kql_condition(condition: Condition) -> str:
    """One condition as KQL.

    Categorical indicators are rendered back as equality against the source column. The forest sees
    `proto=tcp > 0.5`; a detection engineer needs to read `proto == "tcp"`, and the SIEM's flow
    table stores the string, not the indicator.
    """
    cat = condition.categorical
    if cat is not None:
        column, value = cat
        return f'{_column(column)} {"==" if condition.is_positive else "!="} "{value}"'
    return f"{_column(condition.feature)} {condition.operator} {condition.threshold:.6g}"


def to_kql(rule: MinedRule, *, name: str = "", table: str = KQL_TABLE) -> str:
    """One mined rule as a KQL predicate, with its cost in a comment."""
    where = "\n  and ".join(_kql_condition(c) for c in rule.conditions)
    header = [f"// {name or 'Penumbra mined rule'}"]
    if rule.validated:
        header.append(
            f"// held-out precision {rule.holdout_precision:.3f}, "
            f"recall {rule.holdout_recall:.4f}, "
            f"{rule.holdout_false_positives:,} false positives in {rule.holdout_support:,} matches"
        )
    else:
        header.append("// WARNING: not validated on held-out data. Do not enable.")
    header.append(f"// mined from a random forest leaf, train support {rule.train_support:,}")

    return "\n".join([*header, table, f"| where {where}"])


def to_kql_pack(ruleset: RuleSet, *, table: str = KQL_TABLE) -> str:
    """The surviving rules as one runnable KQL function."""
    survivors = ruleset.survivors
    lines = [
        "// Penumbra mined detection rules",
        f"// Generated {datetime.now(UTC).isoformat()}",
        "//",
        f"// {ruleset.n_candidates:,} candidate decision paths were extracted from the trained",
        f"// forest; {len(survivors):,} survived validation at precision >= {ruleset.min_precision:.2f}",
        f"// on {ruleset.n_holdout:,} held-out rows the trees never saw.",
        "//",
        "// These run WITHOUT the model. That is the point: the ML mines the detection, the SIEM",
        "// enforces it, and a detection engineer can read, argue with and own the result.",
        "//",
        "// Each rule's own held-out precision and false-positive count is stated inline. Enable",
        "// them individually on that basis rather than as a block.",
        "",
        "let PenumbraMinedRules = (",
        f"    {table}",
        "    | extend MatchedRule = case(",
    ]

    for i, rule in enumerate(survivors):
        predicate = " and ".join(_kql_condition(c) for c in rule.conditions)
        lines.append(
            f"        // precision {rule.holdout_precision:.3f}, "
            f"{rule.holdout_false_positives:,} FP in {rule.holdout_support:,} matches"
        )
        lines.append(f'        {predicate}, "penumbra-mined-{i:03d}",')

    lines += [
        '        ""',
        "    )",
        "    | where isnotempty(MatchedRule)",
        ");",
        "PenumbraMinedRules",
    ]
    return "\n".join(lines)


def _sigma_scalar(value: str) -> str:
    """Quote a categorical value so YAML reads it as a string.

    Unquoted, protocol values like `no`, `on` and `y` are parsed as booleans by YAML 1.1, and
    numeric-looking service names become ints. The rule would then never match anything and would
    give no indication why.
    """
    escaped = value.replace(chr(92), chr(92) * 2).replace('"', chr(92) + '"')
    return f'"{escaped}"'


def to_sigma(rule: MinedRule, *, title: str = "", index: int = 0) -> str | None:
    """A mined rule as Sigma, or None when it cannot be expressed honestly.

    Returns None rather than a partial translation. Sigma has no field for `sload`, `ct_srv_src` or
    `tcprtt`; emitting one anyway produces a rule that looks portable and evaluates nowhere.
    """
    if not rule.sigma_expressible:
        return None

    # Grouped by key, because a YAML mapping cannot carry the same key twice. Two negations on one
    # field become a list, which in Sigma means OR - and `not (Protocol in [arp, ospf])` is exactly
    # the conjunction of the two negations we are translating.
    selection: dict[str, list[str]] = {}
    negated: dict[str, list[str]] = {}
    for c in rule.conditions:
        field = c.sigma_field
        if field is None:
            return None
        cat = c.categorical
        if cat is not None:
            target = selection if c.is_positive else negated
            target.setdefault(field, []).append(_sigma_scalar(cat[1]))
            continue
        # Sigma expresses ranges with |lte / |gt modifiers; there is no bare comparison operator.
        modifier = "lte" if c.operator == "<=" else "gt"
        selection.setdefault(f"{field}|{modifier}", []).append(f"{c.threshold:.6g}")

    # A repeated positive key means the conjunction asserts two values for one field at once. A
    # list there would silently become OR and invert the rule's meaning, so refuse it instead.
    if any(len(v) > 1 for v in selection.values()):
        return None
    if not selection:
        # Every condition was a negation. "Anything that is not DNS" is not a detection.
        return None

    def block(entries: dict[str, list[str]]) -> list[str]:
        out: list[str] = []
        for key, values in entries.items():
            out.append(f"        {key}: {values[0]}" if len(values) == 1 else f"        {key}:")
            if len(values) > 1:
                out.extend(f"            - {v}" for v in values)
        return out

    cost = (
        f"Held-out precision {rule.holdout_precision:.3f}, "
        f"{rule.holdout_false_positives} false positives in {rule.holdout_support} matches."
        if rule.validated
        else "NOT VALIDATED - do not enable."
    )

    return NEWLINE.join(
        [
            f"title: {title or f'Penumbra mined rule {index:03d}'}",
            f"id: {uuid.uuid5(uuid.NAMESPACE_URL, str(rule.signature()))}",
            "status: experimental",
            "description: |",
            "    Mined from a decision path in a trained random forest and validated on held-out",
            f"    data. {cost}",
            "    Raised for analyst triage; this rule does not block traffic.",
            "author: Penumbra",
            f"date: {datetime.now(UTC).date().isoformat()}",
            "logsource:",
            "    category: firewall",
            "    product: network",
            "detection:",
            "    selection:",
            *block(selection),
            *(["    filter:", *block(negated)] if negated else []),
            f"    condition: selection{' and not filter' if negated else ''}",
            "falsepositives:",
            "    - Authorised scanners and monitoring agents matching the same flow shape",
            "level: medium",
        ]
    )


def to_sigma_pack(ruleset: RuleSet) -> tuple[list[str], int]:
    """Sigma documents for every expressible rule, plus how many were skipped.

    The skip count is returned rather than swallowed: "we emitted 6 Sigma rules" and "we emitted 6
    of 41 because Sigma cannot express the other 35" are different claims.
    """
    docs: list[str] = []
    skipped = 0
    for i, rule in enumerate(ruleset.survivors):
        doc = to_sigma(rule, index=i)
        if doc is None:
            skipped += 1
        else:
            docs.append(doc)
    return docs, skipped
