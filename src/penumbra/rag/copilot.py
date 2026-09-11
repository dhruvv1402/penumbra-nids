"""The analyst copilot: retrieval over ATT&CK, and a cited triage note.

**BM25, not embeddings, and the deterministic renderer is the primary path.** Two reasons, both
practical rather than ideological:

  * There is no LLM key in this environment, so the offline path is not a fallback - it is what
    runs. A demo that depends on a network call is a demo that can fail on stage.
  * A template renderer **cannot hallucinate a technique ID**. Given a security audience, a
    confidently wrong T-number is worse than a plainly mechanical answer.

What the copilot does NOT do: choose the ATT&CK mapping. That comes from the hand-curated table in
`explain/attack_map.py`. Retrieval decides which ATT&CK *prose* to quote for a technique we already
identified; it never decides which technique applies.

Every claim in the note carries its source. That is what the jargon buster means by RAG - grounded
in retrieved documents with citations, not a summary that sounds authoritative.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from penumbra.alerts.models import Alert, Verdict
from penumbra.rag.corpus import Technique, load

_TOKEN = re.compile(r"[a-z0-9]+")

# Flow feature names are meaningless to a corpus written in English about adversary behaviour.
# Searching ATT&CK for "ct_dst_sport_ltm" returns nothing, which is what the first version did.
# This maps the feature to the behaviour it evidences, so retrieval has something to match on.
FEATURE_CONCEPTS: dict[str, str] = {
    "ct_dst_sport_ltm": "port scanning enumeration of destination ports",
    "ct_src_dport_ltm": "scanning many destination ports from one source",
    "ct_dst_ltm": "connections to many destination hosts sweep",
    "ct_srv_src": "repeated connections to the same service",
    "ct_srv_dst": "many sources contacting one service",
    "ct_dst_src_ltm": "repeated host to host connections beaconing",
    "ct_state_ttl": "unusual connection state and time to live",
    "dst_host_srv_count": "many connections to one service on a host",
    "dst_host_count": "many connections to one destination host",
    "count": "burst of connections in a short window",
    "srv_count": "burst of connections to one service",
    "serror_rate": "SYN errors half open connections flood denial of service",
    "srv_serror_rate": "SYN error rate flood denial of service",
    "diff_srv_rate": "connections spread across differing services port sweep",
    "same_srv_rate": "connections concentrated on one service",
    "rerror_rate": "rejected connections closed ports scanning",
    "sbytes": "large outbound data transfer exfiltration",
    "dbytes": "large inbound data transfer download",
    "sload": "high outbound throughput bulk transfer",
    "dur": "long lived connection persistent channel",
    "spkts": "high packet count flood",
    "rate": "high packet rate flood denial of service",
    "sttl": "unusual time to live value spoofed source",
    "swin": "tcp window size anomaly",
    "trans_depth": "repeated http transactions web requests",
    "is_ftp_login": "ftp authentication remote service login",
    "ct_ftp_cmd": "ftp commands remote file transfer",
    "ct_flw_http_mthd": "http methods web application requests",
    "num_failed_logins": "failed authentication attempts brute force password guessing",
    "logged_in": "successful authentication valid accounts",
    "root_shell": "root shell privilege escalation",
    "su_attempted": "privilege escalation attempt elevation",
    "num_file_creations": "file creation on target host",
    "hot": "suspicious indicators access to system directories",
    "wrong_fragment": "malformed fragmented packets evasion",
}


def concept_query(features: list[str]) -> str:
    """Turn feature names into a phrase ATT&CK prose can actually match."""
    phrases = [FEATURE_CONCEPTS[f] for f in features if f in FEATURE_CONCEPTS]
    return " ".join(phrases) if phrases else "anomalous network traffic reconnaissance"


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


@dataclass
class Citation:
    technique_id: str
    name: str
    url: str
    quoted: str

    def render(self) -> str:
        return f"[{self.technique_id}] {self.name} — {self.url}"


@dataclass
class TriageNote:
    """A grounded triage note. Every paragraph traceable to a source."""

    headline: str
    what_fired: list[str] = field(default_factory=list)
    what_this_might_be: str = ""
    what_to_check: list[str] = field(default_factory=list)
    what_not_to_conclude: list[str] = field(default_factory=list)
    citations: list[Citation] = field(default_factory=list)
    generated_by: str = "deterministic-template"

    def render(self) -> str:
        lines = [self.headline, ""]
        if self.what_fired:
            lines.append("Why it fired")
            lines += [f"  - {w}" for w in self.what_fired]
            lines.append("")
        if self.what_this_might_be:
            lines += ["What this might be", f"  {self.what_this_might_be}", ""]
        if self.what_to_check:
            lines.append("What to check next")
            lines += [f"  {i}. {c}" for i, c in enumerate(self.what_to_check, 1)]
            lines.append("")
        if self.what_not_to_conclude:
            lines.append("What this does NOT establish")
            lines += [f"  - {c}" for c in self.what_not_to_conclude]
            lines.append("")
        if self.citations:
            lines.append("Sources")
            lines += [f"  {c.render()}" for c in self.citations]
            lines.append("")
        lines.append(f"[generated by: {self.generated_by}]")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, object]:
        return {
            "headline": self.headline,
            "what_fired": self.what_fired,
            "what_this_might_be": self.what_this_might_be,
            "what_to_check": self.what_to_check,
            "what_not_to_conclude": self.what_not_to_conclude,
            "citations": [c.__dict__ for c in self.citations],
            "generated_by": self.generated_by,
        }


class Retriever:
    """BM25 over the ATT&CK corpus.

    Falls back to a simple term-overlap score if `rank_bm25` is absent, so the copilot degrades
    rather than disappearing when the optional extra is not installed.
    """

    def __init__(self, techniques: list[Technique] | None = None) -> None:
        self.techniques = techniques or load()
        self._corpus = [tokenize(t.searchable) for t in self.techniques]
        self._bm25 = None
        try:
            from rank_bm25 import BM25Okapi

            self._bm25 = BM25Okapi(self._corpus)
        except ImportError:
            pass

    def search(self, query: str, *, top: int = 3) -> list[tuple[Technique, float]]:
        tokens = tokenize(query)
        if not tokens:
            return []

        if self._bm25 is not None:
            scores = self._bm25.get_scores(tokens)
        else:
            wanted = set(tokens)
            scores = [len(wanted & set(doc)) / max(len(wanted), 1) for doc in self._corpus]

        ranked = sorted(zip(self.techniques, scores, strict=True), key=lambda p: p[1], reverse=True)
        return [(t, float(s)) for t, s in ranked[:top] if s > 0]

    def by_id(self, technique_id: str) -> Technique | None:
        for t in self.techniques:
            if t.technique_id == technique_id:
                return t
        return None


class Copilot:
    """Produces cited triage notes."""

    def __init__(self, retriever: Retriever | None = None) -> None:
        self.retriever = retriever or Retriever()

    def note(self, alert: Alert, *, top_sources: int = 2) -> TriageNote:
        if alert.verdict is Verdict.SUSPECTED_NOVEL:
            return self._novel_note(alert, top_sources)
        return self._known_note(alert, top_sources)

    # --- novelty ---------------------------------------------------------------------------------

    def _novel_note(self, alert: Alert, top_sources: int) -> TriageNote:
        """A note for traffic with no known family.

        Deliberately does NOT name a technique. The system's own verdict is that it does not
        recognise this, and a triage note that then confidently names an ATT&CK technique would
        contradict the verdict it is explaining.
        """
        query = concept_query([c.feature for c in alert.contributions[:5]])
        hits = self.retriever.search(query, top=top_sources)

        return TriageNote(
            headline=(
                f"Unrecognised traffic — more unusual than {alert.novelty_percentile:.1%} of the "
                f"benign baseline, and matched no known attack family."
            ),
            what_fired=[c.narrative or c.feature for c in alert.contributions[:4] if c.feature],
            what_this_might_be=(
                "The classifier did not recognise this, which means it is either an attack pattern "
                "absent from training or a legitimate service the baseline has never seen. Those "
                "two look identical to the model and are distinguished by asking the asset owner, "
                "not by asking the model again."
            ),
            what_to_check=[
                f"Is {alert.network.src_ip or 'the source host'} a known system, and is its owner "
                "expecting this traffic?",
                "Has a new service, scanner or backup job been deployed recently on this segment?",
                "Do other hosts show the same pattern? One host is a change; many is a rollout.",
                "If it is legitimate and recurring, record a benign-by-policy verdict with an "
                "expiry rather than a permanent suppression.",
            ],
            what_not_to_conclude=[
                "That this is an attack. The only claim made is statistical unusualness relative to "
                "a benign baseline.",
                "That a low p(attack) means it is safe — the supervised head has no representation "
                "for families it never saw, which is the whole reason this lane exists.",
            ],
            citations=[
                Citation(t.technique_id, t.name, t.url, _first_sentence(t.description)) for t, _ in hits
            ],
        )

    # --- known family ----------------------------------------------------------------------------

    def _known_note(self, alert: Alert, top_sources: int) -> TriageNote:
        technique = None
        citations: list[Citation] = []

        if alert.attack:
            # Look the curated technique up in the corpus for its detection prose. Retrieval is
            # used for context, never to pick the technique.
            technique = self.retriever.by_id(alert.attack.technique_id) or self.retriever.by_id(
                alert.attack.technique_id.split(".")[0]
            )
            if technique:
                citations.append(
                    Citation(
                        technique.technique_id,
                        technique.name,
                        technique.url,
                        _first_sentence(technique.description),
                    )
                )

        query = f"{alert.family or ''} {concept_query([c.feature for c in alert.contributions[:3]])}"
        for t, _ in self.retriever.search(query, top=top_sources):
            if all(c.technique_id != t.technique_id for c in citations):
                citations.append(Citation(t.technique_id, t.name, t.url, _first_sentence(t.description)))

        checks = _detection_steps(technique)
        caveats = [
            "A family label is the model's best match among families it was trained on. It is not "
            "an identification of the tool or the actor.",
        ]
        if alert.attack and alert.attack.confidence in {"medium", "stretch"}:
            caveats.append(
                f"The ATT&CK mapping for {alert.family} is rated '{alert.attack.confidence}': "
                f"{alert.attack.rationale}"
            )

        return TriageNote(
            headline=(
                f"{alert.family or 'Attack'} pattern matched, calibrated probability {alert.p_attack:.2f}."
            ),
            what_fired=[c.narrative or c.feature for c in alert.contributions[:4] if c.feature],
            what_this_might_be=(
                f"Consistent with {technique.name} ({technique.technique_id})."
                if technique
                else "Consistent with the trained pattern for this family."
            ),
            what_to_check=checks,
            what_not_to_conclude=caveats,
            citations=citations,
        )


def _detection_steps(technique: Technique | None) -> list[str]:
    """Turn ATT&CK detection prose into numbered steps, or fall back to generic triage.

    The prose is quoted rather than paraphrased, because paraphrasing is where a grounded answer
    stops being grounded.
    """
    steps = [
        "Confirm the source host is not a sanctioned scanner or monitoring agent.",
        "Check whether the destination hosts are related (same subnet, same service).",
        "Look for the same source in earlier windows — a repeat is a campaign, a one-off is noise.",
    ]
    if technique and technique.detection:
        sentences = [s.strip() for s in technique.detection.split(". ") if len(s.strip()) > 30]
        steps = [f"{s.rstrip('.')}." for s in sentences[:3]] + steps[:2]
    if technique and technique.data_sources:
        steps.append(f"Relevant telemetry per ATT&CK: {', '.join(technique.data_sources[:3])}.")
    return steps


def _first_sentence(text: str, limit: int = 220) -> str:
    text = " ".join(text.split())
    stop = text.find(". ")
    quoted = text[: stop + 1] if 0 < stop < limit else text[:limit]
    return quoted.rstrip() + ("" if quoted.endswith(".") else " …")
