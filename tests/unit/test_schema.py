"""Regression tests for the dataset knowledge in `data/schema.py`.

Every assertion here corresponds to a trap in docs/DATA_TRAPS.md. They are cheap, they run without
the datasets present, and they exist because each of these constants is the kind of thing that gets
"tidied" by someone who does not know why it is the way it is.
"""

from __future__ import annotations

import pytest

from penumbra.data import schema


class TestNSLKDD:
    def test_column_count(self) -> None:
        # 41 features + label + difficulty. The files are headerless, so if this list is wrong
        # every column is silently misnamed and nothing raises.
        assert len(schema.NSLKDD_COLUMNS) == 43
        assert schema.NSLKDD_COLUMNS[-2:] == ["label", "difficulty"]

    def test_difficulty_is_dropped(self) -> None:
        # `difficulty` encodes how many of 21 classic learners got the row right - a function of
        # the answer. Leaving it in is a published-paper-grade mistake.
        assert "difficulty" in schema.NSLKDD_DROP

    def test_zero_variance_column_is_dropped(self) -> None:
        assert "num_outbound_cmds" in schema.NSLKDD_DROP

    def test_su_attempted_is_clamped(self) -> None:
        # Documented as binary; the data contains a third value, 2.
        assert schema.NSLKDD_CLAMP["su_attempted"] == (0, 1)

    def test_every_attack_name_maps_to_a_known_category(self) -> None:
        valid = {"normal", "dos", "probe", "r2l", "u2r"}
        assert set(schema.NSLKDD_ATTACK_CATEGORY.values()) <= valid

    def test_attack_map_covers_both_splits(self) -> None:
        # 22 attack types in train + 17 that appear only in test = 39 attack names, plus `normal`.
        # The map must cover every name in either split or the loader raises mid-parse.
        attack_names = {k for k in schema.NSLKDD_ATTACK_CATEGORY if k != "normal"}
        assert len(attack_names) == 39
        assert schema.NSLKDD_UNSEEN_IN_TEST.issubset(attack_names)
        assert schema.NSLKDD_ABSENT_FROM_TEST.issubset(attack_names)

    def test_unseen_seventeen(self) -> None:
        # The zero-day holdout the dataset authors built for us: 17 attack types in KDDTest+ that
        # never appear in KDDTrain+. 3,750 rows, 16.6% of the test set.
        assert len(schema.NSLKDD_UNSEEN_IN_TEST) == 17
        assert "mscan" in schema.NSLKDD_UNSEEN_IN_TEST
        assert "sqlattack" in schema.NSLKDD_UNSEEN_IN_TEST

    def test_unseen_and_absent_sets_are_disjoint(self) -> None:
        assert not (schema.NSLKDD_UNSEEN_IN_TEST & schema.NSLKDD_ABSENT_FROM_TEST)

    def test_category_lookup_normalises(self) -> None:
        assert schema.nslkdd_category("Neptune") == "dos"
        assert schema.nslkdd_category("neptune.") == "dos"
        assert schema.nslkdd_category(" portsweep ") == "probe"

    def test_unknown_attack_name_raises(self) -> None:
        # Silently defaulting to `normal` would turn a parsing bug into a fabricated benign row.
        with pytest.raises(KeyError):
            schema.nslkdd_category("definitely_not_an_attack")


class TestUNSW:
    def test_column_count(self) -> None:
        assert len(schema.UNSW_COLUMNS) == 45  # id + 42 features + attack_cat + label

    def test_has_no_ip_or_timestamp_columns(self) -> None:
        # This absence is load-bearing: it is why graph features, sequence windows, IP-based
        # correlation and temporal splits all live on CICIDS2017 instead (ADR-0004). If someone
        # swaps in the full four-part release, this test should fail and the ADR be revisited.
        absent = {"srcip", "dstip", "sport", "dsport", "stime", "ltime"}
        assert not (absent & set(schema.UNSW_COLUMNS))

    def test_id_is_dropped(self) -> None:
        assert "id" in schema.UNSW_DROP

    def test_ten_families(self) -> None:
        assert len(schema.UNSW_FAMILIES) == 10
        assert len(schema.UNSW_ATTACK_FAMILIES) == 9
        assert "Normal" not in schema.UNSW_ATTACK_FAMILIES

    def test_family_counts_cover_every_family(self) -> None:
        assert set(schema.UNSW_TRAIN_COUNTS) == set(schema.UNSW_FAMILIES)
        assert set(schema.UNSW_TEST_COUNTS) == set(schema.UNSW_FAMILIES)

    def test_published_split_sizes(self) -> None:
        # 175,341 / 82,332. The training file being LARGER than the testing file is correct and
        # surprises people into "fixing" it.
        assert sum(schema.UNSW_TRAIN_COUNTS.values()) == 175_341
        assert sum(schema.UNSW_TEST_COUNTS.values()) == 82_332

    def test_test_set_prevalence_is_not_operational(self) -> None:
        # ~55% attack. Every precision-family number computed here is inflated by orders of
        # magnitude relative to a real network, which is why eval/prevalence.py exists.
        total = sum(schema.UNSW_TEST_COUNTS.values())
        attack = total - schema.UNSW_TEST_COUNTS["Normal"]
        assert 0.50 < attack / total < 0.60

    def test_worms_is_below_the_estimation_floor(self) -> None:
        # 44 test rows means a recall interval of roughly +/-15 points. Documented so that nobody
        # quotes a Worms point estimate.
        assert schema.UNSW_TEST_COUNTS["Worms"] < 50

    def test_suspected_artifacts_are_real_columns(self) -> None:
        assert set(schema.UNSW_SUSPECTED_ARTIFACTS) <= set(schema.UNSW_COLUMNS)

    def test_entity_window_features_are_real_columns(self) -> None:
        # The ct_* columns are already entity-window aggregates from the original pipeline, which
        # is why we do not recompute graph features on this dataset.
        assert set(schema.UNSW_ENTITY_WINDOW_FEATURES) <= set(schema.UNSW_COLUMNS)


class TestCICIDS:
    def test_destination_port_is_quarantined(self) -> None:
        # Attacks were generated against fixed victim ports; a single stump on this column scores
        # near-perfectly.
        assert "Destination Port" in schema.CICIDS_LEAK_COLUMNS

    def test_duplicate_column_is_dropped(self) -> None:
        # `Fwd Header Length` appears at columns 35 and 56; pandas renames the second to `.1`.
        assert "Fwd Header Length.1" in schema.CICIDS_DUPLICATE_COLUMNS

    def test_temporal_split_is_chronological_and_disjoint(self) -> None:
        assert not set(schema.CICIDS_TRAIN_DAYS) & set(schema.CICIDS_TEST_DAYS)
        assert schema.CICIDS_TRAIN_DAYS[0] == "monday"  # benign-only: trains the novelty head


class TestVerdicts:
    def test_lattice(self) -> None:
        assert set(schema.VERDICTS) == {
            "KNOWN_ATTACK",
            "SUSPECTED_NOVEL",
            "UNCERTAIN",
            "BENIGN",
            "BENIGN_BY_POLICY",
        }

    def test_no_blocking_verdict_exists(self) -> None:
        # ADR-0001. There is no verdict that denies traffic, because there is nothing to deny it
        # with.
        assert not any("BLOCK" in v or "DENY" in v or "DROP" in v for v in schema.VERDICTS)
