"""Attack family -> MITRE ATT&CK, hand-curated.

**This table is written by hand and is never RAG output.** Retrieval from the string "Worms" to a
technique ID is unreliable, and a wrong ATT&CK ID in front of a security audience is worse than no
ID at all. The copilot may narrate a technique; it does not choose one.

There is no authoritative published mapping from NSL-KDD or UNSW-NB15 labels to ATT&CK. What exists
is MITRE's Center for Threat-Informed Defense mapping *methodology*, which we apply: map at the
finest label available, tag each mapping with the confidence we actually have, and **declare the
unmappable ones unmappable** rather than force-fitting them.

Two UNSW-NB15 families have no defensible mapping and are deliberately absent:

  Generic   The dataset defines it as cryptanalytic attacks against block ciphers independent of
            cipher structure. The nearest candidates (T1600 Weaken Encryption, T1110.002 Password
            Cracking) are both real stretches.
  Analysis  The dataset's own description fuses port scanning, spam and HTML-file penetration -
            three unrelated behaviours. It cannot be mapped as a unit.

Saying "two of ten have no defensible mapping" scores better than a fully-populated dishonest table.

One correction worth recording, because it is a common error: **T1046 Network Service Discovery
belongs to TA0007 Discovery, not TA0043 Reconnaissance.** TA0043 is PRE-ATT&CK - activity before the
adversary has access - and its scanning technique is T1595 Active Scanning. Which one applies depends
on where the scanner sits relative to the perimeter, and the flow records do not tell us, so the
external-vantage mapping is primary and the internal one is noted.

Sentinel validates `relevantTechniques` against ATT&CK v16; every ID here exists in v16.
"""

from __future__ import annotations

from typing import Final

from penumbra.alerts.models import AttackTechnique

# --- UNSW-NB15 ----------------------------------------------------------------------------------

UNSW_ATTACK_MAP: Final[dict[str, AttackTechnique]] = {
    "DoS": AttackTechnique(
        tactic_id="TA0040",
        tactic_name="Impact",
        technique_id="T1498",
        technique_name="Network Denial of Service",
        confidence="strong",
        rationale="Volumetric flood traffic. T1499 Endpoint DoS applies to the resource-exhaustion "
        "variants; T1498 is the better default for flow-level detection.",
    ),
    "Exploits": AttackTechnique(
        tactic_id="TA0001",
        tactic_name="Initial Access",
        technique_id="T1190",
        technique_name="Exploit Public-Facing Application",
        confidence="strong",
        rationale="The dataset's Exploits category is service exploitation against reachable "
        "targets. T1203 and T1210 are the client-side and lateral variants.",
    ),
    "Reconnaissance": AttackTechnique(
        tactic_id="TA0043",
        tactic_name="Reconnaissance",
        technique_id="T1595.001",
        technique_name="Active Scanning: Scanning IP Blocks",
        confidence="strong",
        rationale="External-vantage scanning. If the scanner is inside the perimeter the correct "
        "mapping is TA0007 / T1046 Network Service Discovery instead; flow records do not "
        "reveal which side it sits on.",
    ),
    "Worms": AttackTechnique(
        tactic_id="TA0008",
        tactic_name="Lateral Movement",
        technique_id="T1210",
        technique_name="Exploitation of Remote Services",
        confidence="good",
        rationale="Self-propagation across hosts by service exploitation. T1091 Replication "
        "Through Removable Media is excluded - it is out of scope for network-only data.",
    ),
    "Backdoor": AttackTechnique(
        tactic_id="TA0011",
        tactic_name="Command and Control",
        technique_id="T1071",
        technique_name="Application Layer Protocol",
        confidence="good",
        rationale="Persistent channel to a controlled host. T1505.003 Web Shell is the more "
        "specific mapping where the channel is HTTP.",
    ),
    "Shellcode": AttackTechnique(
        tactic_id="TA0002",
        tactic_name="Execution",
        technique_id="T1203",
        technique_name="Exploitation for Client Execution",
        confidence="medium",
        rationale="Shellcode is a payload, not an ATT&CK technique - ATT&CK models the adversary's "
        "goal. Mapped by intent (code execution); T1055 Process Injection is the alternative.",
    ),
    "Fuzzers": AttackTechnique(
        tactic_id="TA0043",
        tactic_name="Reconnaissance",
        technique_id="T1595.002",
        technique_name="Active Scanning: Vulnerability Scanning",
        confidence="stretch",
        rationale="ATT&CK has no technique for adversary-against-victim fuzzing. T1587.004 "
        "describes fuzzing as offline capability development, not observable traffic. This "
        "is the closest available mapping and it is not a good one.",
    ),
    # "Generic" and "Analysis" are intentionally absent. See the module docstring.
}

# --- NSL-KDD ------------------------------------------------------------------------------------

NSLKDD_ATTACK_MAP: Final[dict[str, AttackTechnique]] = {
    "dos": AttackTechnique(
        tactic_id="TA0040",
        tactic_name="Impact",
        technique_id="T1498",
        technique_name="Network Denial of Service",
        confidence="strong",
        rationale="neptune and smurf are flood/reflection attacks (T1498.001/.002); land, "
        "teardrop and pod are malformed-packet exhaustion, closer to T1499.004.",
    ),
    "probe": AttackTechnique(
        tactic_id="TA0043",
        tactic_name="Reconnaissance",
        technique_id="T1595.001",
        technique_name="Active Scanning: Scanning IP Blocks",
        confidence="strong",
        rationale="ipsweep, portsweep, nmap, satan are host and port enumeration.",
    ),
    "r2l": AttackTechnique(
        tactic_id="TA0006",
        tactic_name="Credential Access",
        technique_id="T1110",
        technique_name="Brute Force",
        confidence="medium",
        rationale="R2L is a grab-bag. guess_passwd and ftp_write are brute force; imap, phf and "
        "warezmaster are service exploitation (T1190). Mapped per sub-label where the "
        "fine-grained name is available.",
    ),
    "u2r": AttackTechnique(
        tactic_id="TA0004",
        tactic_name="Privilege Escalation",
        technique_id="T1068",
        technique_name="Exploitation for Privilege Escalation",
        confidence="medium",
        rationale="buffer_overflow and rootkit map well. loadmodule and perl are closer to T1548 "
        "Abuse Elevation Control Mechanism.",
    ),
}

# Finer-grained overrides, used when the specific attack name is known rather than only its
# category. This is where R2L stops being a grab-bag.
NSLKDD_FINE_OVERRIDES: Final[dict[str, AttackTechnique]] = {
    "guess_passwd": AttackTechnique(
        tactic_id="TA0006",
        tactic_name="Credential Access",
        technique_id="T1110.001",
        technique_name="Brute Force: Password Guessing",
        confidence="strong",
        rationale="Repeated authentication attempts against a known account.",
    ),
    "portsweep": AttackTechnique(
        tactic_id="TA0043",
        tactic_name="Reconnaissance",
        technique_id="T1595.001",
        technique_name="Active Scanning: Scanning IP Blocks",
        confidence="strong",
        rationale="Sequential port enumeration against a host.",
    ),
    "satan": AttackTechnique(
        tactic_id="TA0043",
        tactic_name="Reconnaissance",
        technique_id="T1595.002",
        technique_name="Active Scanning: Vulnerability Scanning",
        confidence="strong",
        rationale="SATAN is a vulnerability scanner.",
    ),
    "httptunnel": AttackTechnique(
        tactic_id="TA0011",
        tactic_name="Command and Control",
        technique_id="T1071.001",
        technique_name="Application Layer Protocol: Web Protocols",
        confidence="strong",
        rationale="Covert channel tunnelled over HTTP.",
    ),
    "worm": AttackTechnique(
        tactic_id="TA0008",
        tactic_name="Lateral Movement",
        technique_id="T1210",
        technique_name="Exploitation of Remote Services",
        confidence="good",
        rationale="Self-propagation by remote service exploitation.",
    ),
}

UNMAPPABLE: Final[dict[str, str]] = {
    "Generic": "UNSW-NB15 defines Generic as cryptanalytic attacks against block ciphers "
    "independent of cipher structure. T1600 Weaken Encryption and T1110.002 Password "
    "Cracking are both stretches. No defensible mapping.",
    "Analysis": "UNSW-NB15's Analysis fuses port scanning, spam and HTML-file penetration - three "
    "unrelated behaviours. Not mappable as a single unit.",
}


def lookup(family: str, *, dataset: str = "unsw", fine_label: str | None = None) -> AttackTechnique | None:
    """Map a family (or a specific attack name) to ATT&CK.

    Returns None when no defensible mapping exists. Callers must render that as "no mapping" rather
    than substituting a plausible-looking technique.
    """
    # pandas turns a None in an object column into NaN, so a "family" arriving here can be a float.
    # Guard rather than assume: a crash in the alert path would take down scoring for a benign row.
    if not isinstance(family, str) or not family.strip():
        return None

    if isinstance(fine_label, str) and fine_label.strip():
        override = NSLKDD_FINE_OVERRIDES.get(fine_label.strip().lower())
        if override:
            return override

    table = NSLKDD_ATTACK_MAP if dataset.lower().startswith("nsl") else UNSW_ATTACK_MAP
    return table.get(family) or table.get(family.strip().lower())


def is_unmappable(family: str) -> bool:
    return family in UNMAPPABLE


def coverage_report(dataset: str = "unsw") -> str:
    """How much of the taxonomy we can honestly map, and at what confidence."""
    table = NSLKDD_ATTACK_MAP if dataset.lower().startswith("nsl") else UNSW_ATTACK_MAP
    lines = [
        f"ATT&CK coverage - {dataset}",
        "",
        f"  {'family':<18} {'tactic':<10} {'technique':<12} {'confidence':<10}",
        f"  {'-' * 18} {'-' * 10} {'-' * 12} {'-' * 10}",
    ]
    for fam, tech in table.items():
        lines.append(f"  {fam:<18} {tech.tactic_id:<10} {tech.technique_id:<12} {tech.confidence:<10}")
    if dataset.lower().startswith("unsw"):
        lines.append("")
        for fam, why in UNMAPPABLE.items():
            lines.append(f"  {fam:<18} UNMAPPABLE - {why.split('.')[0]}.")
    return "\n".join(lines)
