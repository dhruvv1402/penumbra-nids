"""The Sentinel workbook parses, queries only our table, and reads fields the ASIM record emits."""

from __future__ import annotations

import json
import re
from pathlib import Path

from penumbra.alerts.models import NetworkContext
from penumbra.alerts.schemas import asim
from penumbra.alerts.scoring import ScoringPolicy, build_alert

WORKBOOK = Path(__file__).resolve().parents[2] / "sentinel" / "Workbooks" / "PenumbraOverview.json"


def queries() -> list[str]:
    wb = json.loads(WORKBOOK.read_text(encoding="utf-8"))
    return [i["content"]["query"] for i in wb["items"] if i["type"] == 3]


def test_workbook_is_valid_and_has_queries() -> None:
    wb = json.loads(WORKBOOK.read_text(encoding="utf-8"))
    assert wb["version"] == "Notebook/1.0"
    assert len(queries()) >= 5


def test_every_query_reads_penumbra_table() -> None:
    assert all(q.startswith("PenumbraAlerts_CL") for q in queries())


def test_adr0001_tile_is_present() -> None:
    assert any("DvcAction != 'Allow'" in q for q in queries())


def _record() -> dict:
    alert = build_alert(
        p_attack=0.95,
        novelty_percentile=0.4,
        policy=ScoringPolicy(),
        family="dos",
        dataset="nslkdd",  # a mapped family, so the conditional ATT&CK fields are present
        network=NetworkContext(src_ip="pseudo:ab", dst_port=80, protocol="tcp"),
    )
    alert.suppression_rule_id = "sup-123"
    return asim.to_asim(alert)


def test_additional_fields_the_workbook_reads_are_emitted() -> None:
    emitted = set(_record()["AdditionalFields"])
    read = set()
    for q in queries():
        read |= set(re.findall(r"AdditionalFields\.(\w+)", q))
    assert read <= emitted, read - emitted


def test_suppression_rule_travels_to_the_siem() -> None:
    assert _record()["AdditionalFields"]["SuppressionRuleId"] == "sup-123"
