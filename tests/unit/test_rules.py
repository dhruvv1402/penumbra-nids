"""Rule mining tests.

Two things are being asserted here, and only one of them is about code working.

The first is mechanical: a mined rule must mean the same thing after simplification, and must
translate to KQL and Sigma without changing meaning. The second is a claim about honesty — an
unvalidated rule must never be emitted, a rule resting on a quarantined feature must never survive,
and a Sigma document must never be a partial translation that looks portable and evaluates nowhere.

The Sigma indentation test exists because the emitter shipped with the detection keys as siblings
of `selection` rather than children. It produced valid YAML describing a different rule, and no
test caught it because the run that exercised it produced zero Sigma-expressible rules.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import yaml
from sklearn.ensemble import RandomForestClassifier

from penumbra.data import schema
from penumbra.rules import emit
from penumbra.rules.mining import (
    Candidates,
    Condition,
    MinedRule,
    mine,
    validate,
)
from penumbra.rules.runner import frame_categories, rule_frame


def rule(*conditions: tuple[str, str, float], support: int = 100, purity: float = 1.0) -> MinedRule:
    return MinedRule(
        conditions=[Condition(f, op, t) for f, op, t in conditions],
        train_support=support,
        train_purity=purity,
        predicted_class=1,
    )


class TestSimplify:
    def test_repeated_greater_than_keeps_the_tightest(self) -> None:
        r = rule(("ct_dst_ltm", ">", 1.5), ("ct_dst_ltm", ">", 2.5)).simplify()
        assert len(r.conditions) == 1
        assert r.conditions[0].threshold == 2.5

    def test_repeated_less_equal_keeps_the_tightest(self) -> None:
        r = rule(("dur", "<=", 4.0), ("dur", "<=", 1.0)).simplify()
        assert len(r.conditions) == 1
        assert r.conditions[0].threshold == 1.0

    def test_a_genuine_range_is_kept(self) -> None:
        """Both directions on one feature is a band, not a redundancy."""
        r = rule(("sbytes", ">", 100.0), ("sbytes", "<=", 500.0)).simplify()
        assert len(r.conditions) == 2
        assert {c.operator for c in r.conditions} == {">", "<="}

    def test_simplification_does_not_change_which_rows_match(self) -> None:
        """The point of the exercise: shorter, identical meaning."""
        X = pd.DataFrame({"a": np.arange(200.0), "b": np.arange(200.0) % 7})
        original = rule(("a", ">", 10.0), ("a", ">", 50.0), ("b", "<=", 3.0), ("b", "<=", 5.0))
        assert np.array_equal(original.matches(X), original.simplify().matches(X))

    def test_implied_categorical_negation_is_dropped(self) -> None:
        r = rule(("proto=mobile", ">", 0.5), ("proto=arp", "<=", 0.5)).simplify()
        assert [str(c) for c in r.conditions] == ["proto == 'mobile'"]

    def test_unimplied_negation_survives(self) -> None:
        # No positive on `proto`, so "not udp" is doing real work.
        r = rule(("proto=udp", "<=", 0.5), ("sbytes", ">", 10.0)).simplify()
        assert any(c.categorical == ("proto", "udp") for c in r.conditions)


class TestCategoricalConditions:
    def test_indicator_renders_as_equality(self) -> None:
        assert str(Condition("service=dns", ">", 0.5)) == "service == 'dns'"
        assert str(Condition("service=dns", "<=", 0.5)) == "service != 'dns'"

    def test_numeric_condition_is_untouched(self) -> None:
        assert str(Condition("sload", ">", 1400.0)) == "sload > 1400"

    def test_sigma_field_resolves_through_the_indicator(self) -> None:
        assert Condition("proto=tcp", ">", 0.5).sigma_field == "Protocol"


class TestQuarantine:
    def test_only_artifact_verdicts_are_quarantined(self) -> None:
        q = schema.quarantined("unsw")
        assert "sttl" in q and "dttl" in q
        # `unresolved` is not `artifact`. Quarantining on suspicion is the same error as keeping on
        # convenience, pointed the other way.
        assert "ct_dst_ltm" not in q

    def test_value_level_quarantine_keeps_the_column(self) -> None:
        q = schema.quarantined("unsw")
        assert "proto=unas" in q
        assert "proto" not in q, "quarantining all of proto would discard TCP to remove one marker"

    def test_nslkdd_quarantines_nothing(self) -> None:
        """Its leaky values are real behaviour, and the verdict table says so."""
        assert schema.quarantined("nslkdd") == set()

    def test_depends_on_matches_indicator_and_column(self) -> None:
        assert rule(("sttl", ">", 60.0)).depends_on({"sttl"})
        assert rule(("proto=unas", ">", 0.5)).depends_on({"proto=unas"})
        # A column-level quarantine covers every indicator built from it.
        assert rule(("proto=tcp", ">", 0.5)).depends_on({"proto"})
        assert not rule(("sload", ">", 1.0)).depends_on({"sttl"})


class TestMining:
    @pytest.fixture
    def forest_and_frame(self):
        rng = np.random.default_rng(0)
        n = 3000
        X = pd.DataFrame(
            {
                "sload": rng.gamma(2.0, 1000.0, n),
                "sttl": rng.choice([31, 62, 254], n),
                "dur": rng.random(n),
            }
        )
        # sttl == 31 is a perfect benign marker, exactly like the real artifact.
        y = np.where(X["sttl"] == 31, 0, (X["sload"] > 2500).astype(int))
        forest = RandomForestClassifier(
            n_estimators=20, max_depth=5, min_samples_leaf=20, random_state=0
        ).fit(X, y)
        return forest, X, y

    def test_mining_produces_candidates(self, forest_and_frame) -> None:
        forest, X, _ = forest_and_frame
        got = mine(forest, list(X.columns), min_support=20)
        assert len(got) > 0
        assert all(r.train_purity >= 0.98 for r in got)

    def test_excluded_features_never_appear_in_a_candidate(self, forest_and_frame) -> None:
        forest, X, _ = forest_and_frame
        got = mine(forest, list(X.columns), min_support=20, exclude_features={"sttl"})
        assert got.n_artifact_excluded > 0, "the fixture guarantees sttl-based paths exist"
        assert all("sttl" not in r.features for r in got)

    def test_exclusion_and_kept_counts_are_in_the_same_units(self, forest_and_frame) -> None:
        """Both counted after de-duplication, so they add up to the unexcluded candidate count.

        Counted before the de-dup, the exclusion figure is inflated by however many trees happened
        to rediscover the same path - and "107 excluded, 101 kept" out of 208 stops being true.
        """
        forest, X, _ = forest_and_frame
        everything = mine(forest, list(X.columns), min_support=20)
        filtered = mine(forest, list(X.columns), min_support=20, exclude_features={"sttl"})
        assert len(filtered) + filtered.n_artifact_excluded == len(everything)

    def test_candidates_are_deduplicated(self, forest_and_frame) -> None:
        forest, X, _ = forest_and_frame
        got = mine(forest, list(X.columns), min_support=20)
        signatures = [r.signature() for r in got]
        assert len(signatures) == len(set(signatures))


class TestValidation:
    def test_an_overfit_rule_does_not_survive(self) -> None:
        X = pd.DataFrame({"a": np.arange(1000.0)})
        y = (np.arange(1000) % 2).astype(int)  # `a` carries no signal about y
        result = validate([rule(("a", ">", 500.0))], X, y, min_precision=0.98)
        assert result.survivors == []

    def test_a_real_rule_survives_and_carries_its_cost(self) -> None:
        X = pd.DataFrame({"a": np.arange(1000.0)})
        y = (np.arange(1000) > 500).astype(int)
        result = validate([rule(("a", ">", 500.0))], X, y, min_precision=0.98)
        assert len(result.survivors) == 1
        survivor = result.survivors[0]
        assert survivor.holdout_precision == 1.0
        assert survivor.holdout_false_positives == 0

    def test_a_rule_matching_too_few_rows_is_rejected(self) -> None:
        """A precision estimate built from three rows is not a precision estimate."""
        X = pd.DataFrame({"a": np.arange(1000.0)})
        y = (np.arange(1000) > 996).astype(int)
        result = validate([rule(("a", ">", 996.0))], X, y, min_holdout_support=20)
        assert result.survivors == []

    def test_artifact_exclusion_count_reaches_the_ruleset(self) -> None:
        X = pd.DataFrame({"a": np.arange(1000.0)})
        y = (np.arange(1000) > 500).astype(int)
        result = validate(Candidates(rules=[rule(("a", ">", 500.0))], n_artifact_excluded=42), X, y)
        assert result.n_artifact_excluded == 42
        assert "42" in result.summary()


class TestEmit:
    def test_kql_renders_a_categorical_as_equality(self) -> None:
        text = emit.to_kql(rule(("service=dns", ">", 0.5), ("sbytes", "<=", 500.0)))
        assert 'service == "dns"' in text
        assert "sbytes <= 500" in text

    def test_kql_marks_an_unvalidated_rule(self) -> None:
        assert "WARNING: not validated" in emit.to_kql(rule(("sload", ">", 1.0)))

    def test_sigma_is_none_when_a_field_is_not_expressible(self) -> None:
        """A partial translation is a different rule, not a shorter one."""
        r = rule(("ct_srv_src", ">", 4.0))
        r.holdout_precision, r.holdout_support, r.holdout_false_positives = 1.0, 100, 0
        assert emit.to_sigma(r) is None

    def test_sigma_detection_keys_are_nested_under_selection(self) -> None:
        """The bug this test exists for: keys emitted as siblings of `selection`.

        The document still parsed. It just described a different rule.
        """
        r = rule(("proto=tcp", ">", 0.5), ("sbytes", "<=", 500.0))
        r.holdout_precision, r.holdout_support, r.holdout_false_positives = 1.0, 100, 0
        doc = yaml.safe_load(emit.to_sigma(r))
        assert doc["detection"]["selection"] == {"Protocol": "tcp", "BytesSent|lte": 500}

    def test_sigma_negation_becomes_a_filter_block(self) -> None:
        r = rule(("proto=udp", "<=", 0.5), ("sbytes", ">", 500.0))
        r.holdout_precision, r.holdout_support, r.holdout_false_positives = 1.0, 100, 0
        doc = yaml.safe_load(emit.to_sigma(r))
        assert doc["detection"]["filter"] == {"Protocol": "udp"}
        assert doc["detection"]["condition"] == "selection and not filter"

    def test_sigma_refuses_a_rule_that_is_only_negations(self) -> None:
        r = rule(("proto=udp", "<=", 0.5))
        r.holdout_precision, r.holdout_support, r.holdout_false_positives = 1.0, 100, 0
        assert emit.to_sigma(r) is None, "'anything that is not UDP' is not a detection"

    def test_sigma_values_are_quoted(self) -> None:
        """YAML 1.1 reads bare `no` and `on` as booleans; a protocol named `no` would never match."""
        r = rule(("proto=no", ">", 0.5), ("sbytes", "<=", 10.0))
        r.holdout_precision, r.holdout_support, r.holdout_false_positives = 1.0, 100, 0
        doc = yaml.safe_load(emit.to_sigma(r))
        assert doc["detection"]["selection"]["Protocol"] == "no"


class TestRuleFrame:
    def test_categoricals_become_named_indicators(self) -> None:
        X = pd.DataFrame({"proto": ["tcp"] * 300 + ["udp"] * 300, "sbytes": range(600)})
        frame = rule_frame(X, categories=frame_categories(X))
        assert "proto=tcp" in frame.columns and "proto=udp" in frame.columns
        assert frame["proto=tcp"].sum() == 300

    def test_rare_categories_are_dropped(self) -> None:
        """A rule built on a value with five rows is an anecdote with a threshold on it."""
        X = pd.DataFrame({"proto": ["tcp"] * 595 + ["arp"] * 5, "sbytes": range(600)})
        frame = rule_frame(X, categories=frame_categories(X))
        assert "proto=tcp" in frame.columns
        assert "proto=arp" not in frame.columns

    def test_the_vocabulary_is_frozen_across_splits(self) -> None:
        """A split missing an indicator would make every rule on it match nothing, silently."""
        train = pd.DataFrame({"proto": ["tcp"] * 300 + ["udp"] * 300, "sbytes": range(600)})
        test = pd.DataFrame({"proto": ["tcp"] * 10, "sbytes": range(10)})
        categories = frame_categories(train)
        assert "proto=udp" in rule_frame(test, categories=categories).columns
