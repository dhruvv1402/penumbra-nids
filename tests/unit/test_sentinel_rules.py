"""The Sentinel solution files obey the validation rules the Sentinel repo enforces.

Payload rules live in test_alert_schemas.py. These are the rules on the solution artifacts
themselves: analytics-rule descriptions must be ASCII (an em dash fails validation), technique IDs
must be well-formed and, where the ATT&CK corpus is available locally, real; and the parser pair the
docs promise must both exist with the standard filtering parameters.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2] / "sentinel"
RULES = sorted((ROOT / "Analytic Rules").glob("*.yaml"))
TECHNIQUE = re.compile(r"^T\d{4}(\.\d{3})?$")
TACTICS = {
    "Reconnaissance", "ResourceDevelopment", "InitialAccess", "Execution", "Persistence",
    "PrivilegeEscalation", "DefenseEvasion", "CredentialAccess", "Discovery", "LateralMovement",
    "Collection", "CommandAndControl", "Exfiltration", "Impact",
}  # fmt: skip
VIM_PARAMS = (
    "starttime", "endtime", "srcipaddr_has_any_prefix", "dstipaddr_has_any_prefix",
    "ipaddr_has_any_prefix", "dstportnumber", "hostname_has_any", "dvcaction", "eventresult", "disabled",
)  # fmt: skip


def load(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_there_are_rules() -> None:
    assert RULES


@pytest.mark.parametrize("path", RULES, ids=lambda p: p.name)
def test_description_is_ascii(path: Path) -> None:
    rule = load(path)
    for field in ("name", "description"):
        bad = [c for c in rule[field] if ord(c) > 127]
        assert not bad, f"{path.name} {field} has non-ASCII {bad!r}; Sentinel validation rejects it"


@pytest.mark.parametrize("path", RULES, ids=lambda p: p.name)
def test_techniques_and_tactics_are_well_formed(path: Path) -> None:
    rule = load(path)
    assert all(TECHNIQUE.match(t) for t in rule["relevantTechniques"])
    assert set(rule["tactics"]) <= TACTICS


@pytest.mark.parametrize("path", RULES, ids=lambda p: p.name)
def test_techniques_exist_in_attack(path: Path) -> None:
    corpus = pytest.importorskip("penumbra.rag.corpus")
    try:
        known = {t.technique_id for t in corpus.load()}
    except FileNotFoundError:
        pytest.skip("ATT&CK corpus not built on this machine")
    missing = [t for t in load(path)["relevantTechniques"] if t not in known]
    assert not missing, f"{path.name}: {missing} are not in the local ATT&CK corpus"


@pytest.mark.parametrize("path", RULES, ids=lambda p: p.name)
def test_rule_queries_go_through_the_asim_parser(path: Path) -> None:
    assert "ASimNetworkSessionPenumbra" in load(path)["query"]


def test_parser_pair_exists_with_standard_filter_parameters() -> None:
    assert (ROOT / "Parsers" / "ASimNetworkSessionPenumbra.kql").exists()
    vim = (ROOT / "Parsers" / "vimNetworkSessionPenumbra.kql").read_text(encoding="utf-8")
    for param in VIM_PARAMS:
        assert re.search(rf"\b{param}\s*:", vim), f"filtering parser lacks the {param} parameter"
    assert "ASimNetworkSessionPenumbra" in vim  # one normalisation, not two that can drift
