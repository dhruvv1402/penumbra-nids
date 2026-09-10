"""Schema conformance for the three SIEM serialisers.

A green suite here is the artifact that says "our output is Sentinel-ingestible", and it is
verifiable by a reader in a way a screenshot of a portal is not. It also costs nothing to run and
needs no Azure subscription.

Several of these tests encode traps that fail silently in production: ASIM's conditional
`ThreatField`, OCSF's derived `type_uid`, OCSF putting endpoints inside `evidences[]`, and ECS
reserving `event.kind: "signal"` for Kibana.
"""

from __future__ import annotations

import pytest

from penumbra.alerts.models import (
    Alert,
    Contribution,
    Incident,
    Lane,
    NetworkContext,
    Severity,
    Verdict,
)
from penumbra.alerts.schemas import asim, ecs, ocsf
from penumbra.explain import attack_map


@pytest.fixture
def known_attack() -> Alert:
    return Alert(
        verdict=Verdict.KNOWN_ATTACK,
        lane=Lane.KNOWN_THREAT,
        severity=Severity.HIGH,
        p_attack=0.93,
        novelty_percentile=0.71,
        priority=87,
        family="Reconnaissance",
        family_confidence=0.88,
        attack=attack_map.lookup("Reconnaissance", dataset="unsw"),
        network=NetworkContext(
            src_ip="pseudo:a1b2c3d4",
            src_port=51422,
            dst_ip="pseudo:e5f6a7b8",
            dst_port=445,
            protocol="tcp",
            src_bytes=46536,
            dst_bytes=32455,
            src_packets=6478,
            dst_packets=446,
        ),
        contributions=[
            Contribution(
                feature="ct_dst_sport_ltm",
                value=312,
                shap_value=0.31,
                narrative="312 distinct destination ports in the window; typical for this host is 3.",
            )
        ],
        suggested_action="Review the source host and confirm whether an authorised scan is scheduled.",
        detector_agreement=2,
    )


@pytest.fixture
def novel() -> Alert:
    """A novelty-lane alert with no endpoints - the UNSW-NB15 shape.

    UNSW's published split has no IP addresses, so this exercises the path where the five-tuple is
    absent entirely. Both OCSF and ASIM have to stay conformant without it.
    """
    return Alert(
        verdict=Verdict.SUSPECTED_NOVEL,
        lane=Lane.HUNTING,
        severity=Severity.MEDIUM,
        p_attack=0.11,
        novelty_percentile=0.997,
        priority=64,
        family=None,
        attack=None,
        detector_agreement=3,
    )


class TestAlertContract:
    def test_approval_cannot_be_disabled(self, known_attack: Alert) -> None:
        # ADR-0001 enforced at the type level: there is no way to construct an auto-actionable
        # alert, even deliberately.
        with pytest.raises(ValueError, match="does not act"):
            Alert(
                verdict=Verdict.KNOWN_ATTACK,
                lane=Lane.KNOWN_THREAT,
                severity=Severity.HIGH,
                p_attack=0.9,
                novelty_percentile=0.5,
                priority=80,
                requires_analyst_approval=False,
            )

    def test_verdict_lattice_has_no_enforcement_member(self) -> None:
        for v in Verdict:
            assert not any(word in v.value for word in ("BLOCK", "DENY", "DROP", "QUARANTINE"))

    def test_three_numbers_are_separate_fields(self, known_attack: Alert) -> None:
        # Conflating these is the category error ADR-0002 exists to prevent.
        assert known_attack.p_attack != known_attack.novelty_percentile
        assert isinstance(known_attack.priority, int)
        assert 0 <= known_attack.priority <= 100

    def test_probabilities_are_bounded(self) -> None:
        with pytest.raises(ValueError):
            Alert(
                verdict=Verdict.BENIGN,
                lane=Lane.KNOWN_THREAT,
                severity=Severity.LOW,
                p_attack=1.4,
                novelty_percentile=0.5,
                priority=10,
            )

    def test_novel_alert_explains_itself_without_a_family(self, novel: Alert) -> None:
        text = novel.explanation
        assert "unlike anything" in text
        assert "99.7%" in text  # the percentile rendered for a human

    def test_ip_fields_are_marked_pseudonymised(self, known_attack: Alert) -> None:
        assert known_attack.network.is_pseudonymised
        assert known_attack.network.src_ip is not None
        assert not known_attack.network.src_ip.count(".")  # not a raw dotted quad


class TestASIM:
    def test_conformant(self, known_attack: Alert) -> None:
        asim.assert_conformant(asim.to_asim(known_attack))

    def test_conformant_without_endpoints(self, novel: Alert) -> None:
        asim.assert_conformant(asim.to_asim(novel))

    def test_mandatory_fields_present(self, known_attack: Alert) -> None:
        record = asim.to_asim(known_attack)
        for field in asim.MANDATORY_FIELDS:
            assert field in record, field

    def test_alert_not_block_is_expressed_in_the_schema(self, known_attack: Alert) -> None:
        # ADR-0001 stated in Microsoft's own vocabulary: high confidence, high severity, and we
        # still did not act.
        record = asim.to_asim(known_attack)
        assert record["DvcAction"] == "Allow"
        assert record["ThreatConfidence"] == 93
        assert record["EventSeverity"] == "High"

    def test_threat_confidence_carries_calibrated_probability_not_priority(self, known_attack: Alert) -> None:
        record = asim.to_asim(known_attack)
        assert record["ThreatConfidence"] == int(round(known_attack.p_attack * 100))
        assert record["ThreatConfidence"] != known_attack.priority

    def test_threat_field_is_set_whenever_threat_ip_is(self, known_attack: Alert) -> None:
        record = asim.to_asim(known_attack)
        assert record["ThreatIpAddr"]
        assert record["ThreatField"] in asim.THREAT_FIELDS

    def test_missing_threat_field_is_caught(self, known_attack: Alert) -> None:
        record = asim.to_asim(known_attack)
        del record["ThreatField"]
        assert any("ThreatField is required" in p for p in asim.validate(record))

    def test_severity_is_a_string_enum_not_a_number(self, known_attack: Alert) -> None:
        record = asim.to_asim(known_attack)
        assert isinstance(record["EventSeverity"], str)
        assert record["EventSeverity"] in asim.EVENT_SEVERITIES

    def test_enforcement_action_is_rejected(self, known_attack: Alert) -> None:
        record = asim.to_asim(known_attack)
        record["DvcAction"] = "Drop"
        assert any("alerts; it does not block" in p for p in asim.validate(record))

    def test_column_names_respect_log_analytics_rules(self, known_attack: Alert) -> None:
        for name in asim.to_asim(known_attack):
            assert name[:1].isalpha()
            assert len(name) <= 45

    def test_novelty_percentile_is_not_reported_as_confidence(self, novel: Alert) -> None:
        record = asim.to_asim(novel)
        assert record["ThreatConfidence"] == int(round(novel.p_attack * 100))
        assert record["AdditionalFields"]["NoveltyPercentile"] == pytest.approx(0.997)


class TestOCSF:
    def test_conformant(self, known_attack: Alert) -> None:
        ocsf.assert_conformant(ocsf.to_ocsf(known_attack))

    def test_conformant_without_endpoints(self, novel: Alert) -> None:
        # Evidence has an "at least one of" constraint; the no-endpoint path must still satisfy it.
        ocsf.assert_conformant(ocsf.to_ocsf(novel))

    def test_type_uid_is_derived(self, known_attack: Alert) -> None:
        record = ocsf.to_ocsf(known_attack)
        assert record["type_uid"] == record["class_uid"] * 100 + record["activity_id"]
        assert record["type_uid"] == 200401

    def test_endpoints_live_inside_evidences(self, known_attack: Alert) -> None:
        record = ocsf.to_ocsf(known_attack)
        assert "src_endpoint" not in record
        assert "dst_endpoint" not in record
        assert record["evidences"][0]["src_endpoint"]["ip"]

    def test_top_level_endpoint_is_rejected(self, known_attack: Alert) -> None:
        record = ocsf.to_ocsf(known_attack)
        record["src_endpoint"] = {"ip": "10.0.0.1"}
        assert any("does not belong at the top level" in p for p in ocsf.validate(record))

    def test_empty_evidence_is_rejected(self, known_attack: Alert) -> None:
        record = ocsf.to_ocsf(known_attack)
        record["evidences"] = [{}]
        assert any("Evidence attributes" in p for p in ocsf.validate(record))


class TestECS:
    def test_conformant(self, known_attack: Alert) -> None:
        ecs.assert_conformant(ecs.to_ecs(known_attack))

    def test_kind_is_alert_never_signal(self, known_attack: Alert) -> None:
        record = ecs.to_ecs(known_attack)
        assert record["event"]["kind"] == "alert"
        assert record["event"]["kind"] not in ecs.FORBIDDEN_EVENT_KINDS

    def test_signal_is_rejected(self, known_attack: Alert) -> None:
        record = ecs.to_ecs(known_attack)
        record["event"]["kind"] = "signal"
        assert any("reserved for Kibana" in p for p in ecs.validate(record))

    def test_categories_are_allowed_values(self, known_attack: Alert) -> None:
        for c in ecs.to_ecs(known_attack)["event"]["category"]:
            assert c in ecs.EVENT_CATEGORIES

    def test_threat_technique_has_id_and_reference(self, known_attack: Alert) -> None:
        threat = ecs.to_ecs(known_attack)["threat"]
        assert threat["technique"]["id"].startswith("T")
        assert threat["technique"]["reference"].startswith("https://attack.mitre.org/")


class TestAllThreeAgree:
    def test_same_underlying_probability(self, known_attack: Alert) -> None:
        """One detection, three schemas, one calibrated probability behind all of them."""
        a = asim.to_asim(known_attack)
        o = ocsf.to_ocsf(known_attack)
        e = ecs.to_ecs(known_attack)
        assert a["ThreatConfidence"] == o["confidence_score"] == 93
        assert e["penumbra"]["p_attack"] == pytest.approx(known_attack.p_attack)

    def test_none_of_them_can_express_a_block(self, known_attack: Alert) -> None:
        blob = (
            str(asim.to_asim(known_attack)) + str(ocsf.to_ocsf(known_attack)) + str(ecs.to_ecs(known_attack))
        )
        for word in ("Deny", "Drop", "quarantine", "blocked"):
            assert word not in blob


class TestAttackMap:
    def test_unmappable_families_return_none(self) -> None:
        # Refusing to map is the point. A plausible-looking wrong technique ID is worse than none.
        assert attack_map.lookup("Generic", dataset="unsw") is None
        assert attack_map.lookup("Analysis", dataset="unsw") is None
        assert attack_map.is_unmappable("Generic")

    def test_every_mapping_carries_a_confidence_and_rationale(self) -> None:
        for table in (attack_map.UNSW_ATTACK_MAP, attack_map.NSLKDD_ATTACK_MAP):
            for fam, tech in table.items():
                assert tech.confidence in {"strong", "good", "medium", "stretch"}, fam
                assert tech.rationale, fam

    def test_technique_ids_are_well_formed(self) -> None:
        for table in (
            attack_map.UNSW_ATTACK_MAP,
            attack_map.NSLKDD_ATTACK_MAP,
            attack_map.NSLKDD_FINE_OVERRIDES,
        ):
            for fam, tech in table.items():
                assert tech.technique_id.startswith("T"), fam
                assert tech.tactic_id.startswith("TA"), fam

    def test_fuzzers_is_flagged_as_a_stretch(self) -> None:
        # ATT&CK has no adversary-traffic fuzzing technique. Saying so is the honest option.
        assert attack_map.UNSW_ATTACK_MAP["Fuzzers"].confidence == "stretch"

    def test_reconnaissance_uses_pre_attack_scanning(self) -> None:
        # T1046 is TA0007 Discovery, not TA0043 Reconnaissance. The external-vantage mapping is
        # T1595; the internal alternative is noted in the rationale.
        recon = attack_map.UNSW_ATTACK_MAP["Reconnaissance"]
        assert recon.tactic_id == "TA0043"
        assert recon.technique_id.startswith("T1595")
        assert "T1046" in recon.rationale

    def test_fine_label_override_beats_category(self) -> None:
        coarse = attack_map.lookup("r2l", dataset="nslkdd")
        fine = attack_map.lookup("r2l", dataset="nslkdd", fine_label="guess_passwd")
        assert coarse is not None and fine is not None
        assert fine.technique_id == "T1110.001"
        assert fine.technique_id != coarse.technique_id


class TestIncident:
    def test_compression_is_recorded(self) -> None:
        inc = Incident(
            title="Port scan from pseudo:a1b2c3d4",
            severity=Severity.HIGH,
            lane=Lane.KNOWN_THREAT,
            priority=88,
            entity="pseudo:a1b2c3d4",
            family="Reconnaissance",
            alert_ids=[f"a{i}" for i in range(3812)],
            event_count=3812,
            distinct_destinations=254,
            distinct_ports=1024,
        )
        assert inc.compression_ratio == 3812
        assert inc.status == "open"

    def test_analyst_verdict_is_constrained(self) -> None:
        with pytest.raises(ValueError):
            Incident(
                title="x",
                severity=Severity.LOW,
                lane=Lane.HUNTING,
                priority=1,
                entity="e",
                analyst_verdict="looks_fine",
            )
