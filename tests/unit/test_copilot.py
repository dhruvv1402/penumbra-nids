"""The triage copilot: grounded, cited, and never overriding the curated ATT&CK mapping."""

from __future__ import annotations

from penumbra.alerts.models import Contribution, NetworkContext, Verdict
from penumbra.alerts.scoring import ScoringPolicy, build_alert
from penumbra.rag.copilot import Copilot, Retriever
from penumbra.rag.corpus import Technique

CORPUS = [
    Technique(
        "T1498",
        "Network Denial of Service",
        "Adversaries may perform Network Denial of Service attacks to degrade availability. Floods exhaust bandwidth.",
        tactics=["impact"],
        detection="Monitor network traffic for SYN floods and sudden volume. Alert on connection error rates.",
        url="https://attack.mitre.org/techniques/T1498",
    ),
    Technique(
        "T1046",
        "Network Service Discovery",
        "Adversaries may scan for services on remote hosts. Port scans probe many ports.",
        tactics=["discovery"],
        detection="Watch for one source touching many ports or hosts in a short window.",
        url="https://attack.mitre.org/techniques/T1046",
    ),
]


def copilot() -> Copilot:
    return Copilot(Retriever(CORPUS))


def alert(p: float, novelty: float, family: str | None):
    a = build_alert(
        p_attack=p,
        novelty_percentile=novelty,
        policy=ScoringPolicy(),
        family=family,
        dataset="nslkdd",
        agreement=3,
        network=NetworkContext(src_ip="pseudo:ab", protocol="tcp"),
    )
    a.contributions = [
        Contribution(
            feature="serror_rate",
            value=1.0,
            shap_value=0.2,
            direction="toward_attack",
            narrative="syn errors",
        )
    ]
    return a


def test_known_attack_note_cites_the_curated_technique_first() -> None:
    a = alert(0.97, 0.3, "dos")
    assert a.attack is not None
    note = copilot().note(a)
    assert note.citations and note.citations[0].technique_id == a.attack.technique_id.split(".")[0]
    assert note.what_not_to_conclude  # every note states what it does not establish


def test_novel_note_names_no_technique_as_the_explanation() -> None:
    a = alert(0.05, 0.999, None)
    assert a.verdict is Verdict.SUSPECTED_NOVEL
    note = copilot().note(a)
    assert "T1" not in note.what_this_might_be and "T1" not in note.headline
    assert any("that this is an attack" in c.lower() for c in note.what_not_to_conclude)


def test_every_citation_resolves_to_the_corpus() -> None:
    known = {t.technique_id for t in CORPUS}
    for a in (alert(0.97, 0.3, "dos"), alert(0.05, 0.999, None)):
        assert {c.technique_id for c in copilot().note(a).citations} <= known


def test_note_is_deterministic() -> None:
    a = alert(0.97, 0.3, "dos")
    assert copilot().note(a).render() == copilot().note(a).render()
