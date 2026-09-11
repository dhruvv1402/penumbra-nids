"""Security-layer tests: audit chain, RBAC boundaries, PII pseudonymisation.

Several of these assert things the README claims, which is the point of having them: a claim that
is only in prose drifts away from the code, and a claim with a test behind it does not.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from penumbra.api.security import pii, rbac
from penumbra.api.security.audit import GENESIS, AuditLog, record_verdict
from penumbra.api.security.rbac import Permission, PermissionDenied, Principal, Role


class TestAuditChain:
    def test_empty_log_head_is_genesis(self) -> None:
        assert AuditLog().head == GENESIS

    def test_chain_verifies(self) -> None:
        log = AuditLog()
        for i in range(10):
            log.append(actor="alice", role="analyst", action="test", target=f"t{i}")
        assert log.verify().valid

    def test_each_entry_links_to_the_previous(self) -> None:
        log = AuditLog()
        a = log.append(actor="a", role="analyst", action="one")
        b = log.append(actor="a", role="analyst", action="two")
        assert a.previous_hash == GENESIS
        assert b.previous_hash == a.entry_hash

    def test_editing_an_entry_breaks_the_chain(self) -> None:
        log = AuditLog()
        log.append(actor="alice", role="analyst", action="auth.login")
        record_verdict(log, actor="alice", role="analyst", incident_id="INC-1", verdict="true_positive")
        log.append(actor="bob", role="senior", action="incident.close", target="INC-1")
        assert log.verify().valid

        # An attacker flips a verdict to hide that they marked their own traffic benign.
        log._entries[1] = replace(log._entries[1], detail={"verdict": "false_positive", "note": ""})

        result = log.verify()
        assert not result.valid
        assert result.broken_at == 1
        assert "modified" in result.reason

    def test_deleting_an_entry_breaks_the_chain(self) -> None:
        log = AuditLog()
        for i in range(5):
            log.append(actor="a", role="analyst", action=f"act{i}")
        del log._entries[2]
        assert not log.verify().valid

    def test_reordering_breaks_the_chain(self) -> None:
        log = AuditLog()
        for i in range(5):
            log.append(actor="a", role="analyst", action=f"act{i}")
        log._entries[1], log._entries[2] = log._entries[2], log._entries[1]
        assert not log.verify().valid

    def test_hash_is_stable_across_a_json_round_trip(self) -> None:
        # sort_keys in compute_hash exists for this. Without it the digest depends on dict ordering
        # and a log written by one process fails to verify in another.
        log = AuditLog()
        entry = log.append(actor="a", role="admin", action="pii.reidentify", detail={"z": 1, "a": 2})
        from penumbra.api.security.audit import AuditEntry

        assert AuditEntry.from_json(entry.to_json()).is_intact()

    def test_persists_and_reloads(self, tmp_path) -> None:
        path = tmp_path / "audit.jsonl"
        log = AuditLog(path)
        for i in range(3):
            log.append(actor="a", role="analyst", action=f"act{i}")
        assert AuditLog(path).verify().valid
        assert len(AuditLog(path)) == 3


class TestRBAC:
    def test_analyst_cannot_close_an_incident(self) -> None:
        analyst = Principal("alice", Role.ANALYST)
        assert not analyst.can(Permission.CLOSE_INCIDENT)
        with pytest.raises(PermissionDenied):
            rbac.require(analyst, Permission.CLOSE_INCIDENT)

    def test_analyst_cannot_promote_a_verdict_into_training(self) -> None:
        # THREAT_MODEL T1: separating "record a verdict" from "let it train the model" means
        # poisoning needs two compromised accounts, not one.
        analyst = Principal("alice", Role.ANALYST)
        assert analyst.can(Permission.RECORD_VERDICT)
        assert not analyst.can(Permission.PROMOTE_VERDICT)

    def test_only_admin_can_reidentify(self) -> None:
        assert not Principal("a", Role.ANALYST).can(Permission.REIDENTIFY_PII)
        assert not Principal("b", Role.SENIOR).can(Permission.REIDENTIFY_PII)
        assert Principal("c", Role.ADMIN).can(Permission.REIDENTIFY_PII)

    def test_roles_are_nested(self) -> None:
        analyst = rbac.ROLE_PERMISSIONS[Role.ANALYST]
        senior = rbac.ROLE_PERMISSIONS[Role.SENIOR]
        admin = rbac.ROLE_PERMISSIONS[Role.ADMIN]
        assert analyst < senior < admin

    def test_row_level_segment_scoping(self) -> None:
        scoped = Principal("alice", Role.ANALYST, segments=frozenset({"dmz", "corp"}))
        assert scoped.may_see_segment("dmz")
        assert not scoped.may_see_segment("ot")

        unrestricted = Principal("bob", Role.SENIOR)
        assert unrestricted.may_see_segment("ot")

    def test_denial_message_names_the_required_role(self) -> None:
        with pytest.raises(PermissionDenied, match="senior"):
            rbac.require(Principal("a", Role.ANALYST), Permission.CLOSE_INCIDENT)

    def test_matrix_renders_every_permission(self) -> None:
        text = rbac.matrix()
        for perm in Permission:
            assert perm.value in text


class TestPII:
    def test_pseudonym_is_deterministic(self) -> None:
        # Correlation by entity depends on this.
        assert pii.pseudonymise_ip("10.0.0.1") == pii.pseudonymise_ip("10.0.0.1")

    def test_different_addresses_differ(self) -> None:
        assert pii.pseudonymise_ip("10.0.0.1") != pii.pseudonymise_ip("10.0.0.2")

    def test_output_is_not_an_address(self) -> None:
        out = pii.pseudonymise_ip("192.168.1.50")
        assert pii.is_pseudonymised(out)
        assert "192.168" not in out

    def test_subnet_preservation_keeps_hosts_together(self) -> None:
        a = pii.pseudonymise_ip("10.0.5.10", preserve_subnet=True)
        b = pii.pseudonymise_ip("10.0.5.99", preserve_subnet=True)
        c = pii.pseudonymise_ip("10.0.9.10", preserve_subnet=True)
        assert a.rsplit(".", 1)[0] == b.rsplit(".", 1)[0]
        assert a.rsplit(".", 1)[0] != c.rsplit(".", 1)[0]

    def test_ipv6_handled(self) -> None:
        assert pii.is_pseudonymised(pii.pseudonymise_ip("2001:db8::1"))

    def test_reidentification_denied_below_admin(self) -> None:
        index = pii.build_reverse_index(["10.0.0.1"])
        token = next(iter(index))
        for role in ("analyst", "senior"):
            with pytest.raises(pii.ReidentificationDenied):
                pii.reidentify(token, index, role=role, actor="x", reason="curiosity")

    def test_reidentification_requires_a_reason(self) -> None:
        index = pii.build_reverse_index(["10.0.0.1"])
        token = next(iter(index))
        with pytest.raises(ValueError, match="reason"):
            pii.reidentify(token, index, role="admin", actor="root", reason="   ")

    def test_reidentification_is_audited(self) -> None:
        log = AuditLog()
        index = pii.build_reverse_index(["10.0.0.1"])
        token = next(iter(index))

        class Sink:
            def __init__(self) -> None:
                self.entries: list[dict] = []

            def append(self, entry: dict) -> None:
                self.entries.append(entry)

        sink = Sink()
        assert (
            pii.reidentify(token, index, role="admin", actor="root", reason="INC-42", audit_sink=sink)
            == "10.0.0.1"
        )
        assert sink.entries[0]["action"] == "pii.reidentify"
        assert sink.entries[0]["reason"] == "INC-42"
        _ = log

    def test_no_raw_addresses_escape_a_serialised_alert(self) -> None:
        clean = {"src": pii.pseudonymise_ip("10.0.0.1"), "nested": {"dst": pii.pseudonymise_ip("10.0.0.2")}}
        assert pii.assert_no_raw_addresses(clean) == []

        leaky = {"src": pii.pseudonymise_ip("10.0.0.1"), "nested": {"dst": "10.0.0.2"}}
        assert pii.assert_no_raw_addresses(leaky) == ["10.0.0.2"]

    def test_key_fingerprint_does_not_reveal_the_key(self) -> None:
        fp = pii.fingerprint_key()
        assert len(fp) == 16
        from penumbra.config import settings

        assert settings().pii_hmac_key not in fp
