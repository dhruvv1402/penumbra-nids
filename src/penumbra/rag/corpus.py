"""ATT&CK corpus: STIX bundle in, compact JSONL out.

The enterprise STIX bundle is 53.8 MB of deeply nested JSON. Parsing it per request would be absurd,
so it is reduced once at build time to a few megabytes of flat records carrying only the fields a
triage note needs: id, name, tactics, description, detection guidance, data sources, platforms.

The reduction happens offline and the result is checked in, which means the copilot has **no network
dependency at all**. That is deliberate: the demo must work with the cable pulled.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from penumbra.config import settings

CORPUS_FILENAME = "attack_techniques.jsonl"


@dataclass
class Technique:
    technique_id: str
    name: str
    description: str
    tactics: list[str] = field(default_factory=list)
    platforms: list[str] = field(default_factory=list)
    detection: str = ""
    data_sources: list[str] = field(default_factory=list)
    is_subtechnique: bool = False
    url: str = ""

    @property
    def searchable(self) -> str:
        """The text BM25 indexes.

        Detection guidance is included because the questions an analyst actually asks are about what
        to look for next, not what the technique is called.
        """
        return " ".join(
            [self.technique_id, self.name, " ".join(self.tactics), self.description, self.detection]
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "technique_id": self.technique_id,
            "name": self.name,
            "description": self.description,
            "tactics": self.tactics,
            "platforms": self.platforms,
            "detection": self.detection,
            "data_sources": self.data_sources,
            "is_subtechnique": self.is_subtechnique,
            "url": self.url,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Technique:
        return cls(**data)


def _attack_id(obj: dict[str, Any]) -> str | None:
    for ref in obj.get("external_references", []):
        if ref.get("source_name") == "mitre-attack":
            return str(ref.get("external_id", "")) or None
    return None


def _attack_url(obj: dict[str, Any]) -> str:
    for ref in obj.get("external_references", []):
        if ref.get("source_name") == "mitre-attack":
            return str(ref.get("url", ""))
    return ""


def build(
    stix_path: Path | None = None,
    out_path: Path | None = None,
    *,
    max_description_chars: int = 1200,
) -> Path:
    """Reduce the STIX bundle to a JSONL corpus.

    Descriptions are truncated because ATT&CK prose runs to several thousand characters and a triage
    note quotes a sentence or two. Truncation happens at a sentence boundary where one is available,
    so a citation never ends mid-clause.
    """
    stix_path = stix_path or (settings().raw_dir / "attack" / "enterprise-attack.json")
    out_path = out_path or (settings().interim_dir / CORPUS_FILENAME)

    if not stix_path.exists():
        raise FileNotFoundError(f"{stix_path} not found. Run `penumbra data fetch --dataset attack` first.")

    bundle = json.loads(stix_path.read_text(encoding="utf-8"))
    objects = bundle.get("objects", [])
    detection_by_technique = _detection_index(objects)
    techniques: list[Technique] = []

    for obj in objects:
        if obj.get("type") != "attack-pattern" or obj.get("revoked") or obj.get("x_mitre_deprecated"):
            continue
        tid = _attack_id(obj)
        if not tid:
            continue

        techniques.append(
            Technique(
                technique_id=tid,
                name=str(obj.get("name", "")),
                description=_truncate(str(obj.get("description", "")), max_description_chars),
                tactics=[
                    str(p.get("phase_name", "")).replace("-", " ")
                    for p in obj.get("kill_chain_phases", [])
                    if p.get("kill_chain_name") == "mitre-attack"
                ],
                platforms=[str(p) for p in obj.get("x_mitre_platforms", [])],
                detection=_truncate(
                    detection_by_technique.get(str(obj.get("id", "")), ""), max_description_chars
                ),
                data_sources=[str(d) for d in obj.get("x_mitre_data_sources", [])],
                is_subtechnique=bool(obj.get("x_mitre_is_subtechnique")),
                url=_attack_url(obj),
            )
        )

    techniques.sort(key=lambda t: t.technique_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for technique in techniques:
            fh.write(json.dumps(technique.to_dict()) + "\n")
    return out_path


def _detection_index(objects: list[dict[str, Any]]) -> dict[str, str]:
    """Map each technique's STIX id to MITRE's own detection guidance.

    The current ATT&CK STIX schema no longer carries `x_mitre_detection` on the technique. Detection
    guidance lives in `x-mitre-detection-strategy` objects, which link to techniques through a
    `detects` relationship and reference `x-mitre-analytic` objects that hold the actual prose.

    Reading it from the old field silently yields empty detection for every technique - which is
    what the first version of this module did, and the symptom was a corpus reporting
    `n_with_detection: 0`.
    """
    analytics = {
        obj["id"]: str(obj.get("description") or "")
        for obj in objects
        if obj.get("type") == "x-mitre-analytic"
    }
    strategies = {
        obj["id"]: obj
        for obj in objects
        if obj.get("type") == "x-mitre-detection-strategy" and not obj.get("x_mitre_deprecated")
    }

    out: dict[str, list[str]] = {}
    for rel in objects:
        if rel.get("type") != "relationship" or rel.get("relationship_type") != "detects":
            continue
        strategy = strategies.get(rel.get("source_ref", ""))
        technique_id = rel.get("target_ref", "")
        if not strategy or not technique_id:
            continue

        parts = [str(strategy.get("name") or "")]
        parts += [analytics[ref] for ref in strategy.get("x_mitre_analytic_refs", []) if ref in analytics]
        text = " ".join(part for part in parts if part).strip()
        if text:
            out.setdefault(technique_id, []).append(text)

    return {tid: " ".join(texts) for tid, texts in out.items()}


def _truncate(text: str, limit: int) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    cut = text[:limit]
    stop = cut.rfind(". ")
    return (cut[: stop + 1] if stop > limit // 2 else cut).rstrip() + " …"


def load(path: Path | None = None) -> list[Technique]:
    path = path or (settings().interim_dir / CORPUS_FILENAME)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `penumbra copilot build` first.")
    with path.open("r", encoding="utf-8") as fh:
        return [Technique.from_dict(json.loads(line)) for line in fh if line.strip()]


def stats(techniques: list[Technique]) -> dict[str, Any]:
    return {
        "n_techniques": len(techniques),
        "n_subtechniques": sum(1 for t in techniques if t.is_subtechnique),
        "n_with_detection": sum(1 for t in techniques if t.detection),
        "tactics": sorted({tactic for t in techniques for tactic in t.tactics}),
    }
