"""IP pseudonymisation.

**This is pseudonymisation, not anonymisation, and the distinction is legally load-bearing.**

IPv4 has 2^32 values. Anyone holding the key can enumerate the entire mapping in seconds, so the
output is reversible to an attacker who obtains the key - which means that under GDPR Art. 4(5) and
Recital 26 the pseudonymised data **remains personal data**. IP addresses are personal data in the
first place per *Breyer v Bundesrepublik Deutschland* (CJEU C-582/14).

We do not claim otherwise, and the consequences are real rather than decorative:

  * The key lives outside the data store and never in the repository.
  * Re-identification is an admin-only operation and every use is written to the audit log.
  * The motivated-intruder test is acknowledged in docs/THREAT_MODEL.md (T5), not argued away.

What this does buy: an analyst triaging a queue does not need to see raw addresses, correlation
still works because the mapping is deterministic, and a leaked alert store is materially less
useful than one holding plaintext addresses.

Subnet structure is preserved optionally, because a SOC frequently needs to know that two hosts are
on the same /24 without needing to know which /24. That is a deliberate confidentiality trade: it
leaks topology in exchange for keeping the alerts usable.
"""

from __future__ import annotations

import hmac
import ipaddress
from hashlib import blake2b, sha256

from penumbra.config import settings

PREFIX = "pseudo:"
_DIGEST_CHARS = 12


class ReidentificationDenied(PermissionError):
    """Raised when a caller without the admin role attempts to reverse a pseudonym."""


def _key() -> bytes:
    return settings().pii_hmac_key.encode("utf-8")


def pseudonymise_ip(ip: str, *, preserve_subnet: bool = False) -> str:
    """Map an IP address to a stable pseudonym.

    Deterministic, so correlation by entity still works. Keyed, so the mapping cannot be recomputed
    without the secret - unlike a bare hash, which is trivially reversible for a 32-bit space by
    anyone with a rainbow table and ten minutes.
    """
    try:
        addr = ipaddress.ip_address(ip)
    except ValueError:
        # Not an address - hash it anyway rather than passing an unexpected value through to a
        # store that is supposed to contain no identifiers.
        return PREFIX + _mac(ip.encode("utf-8"))

    if preserve_subnet and addr.version == 4:
        net = ipaddress.ip_network(f"{addr}/24", strict=False)
        host = int(addr) & 0xFF
        return f"{PREFIX}{_mac(str(net.network_address).encode())}.{host}"

    return PREFIX + _mac(addr.packed)


def _mac(payload: bytes) -> str:
    return hmac.new(_key(), payload, sha256).hexdigest()[:_DIGEST_CHARS]


def is_pseudonymised(value: str | None) -> bool:
    return bool(value) and str(value).startswith(PREFIX)


def build_reverse_index(addresses: list[str], *, preserve_subnet: bool = False) -> dict[str, str]:
    """Pseudonym -> address map, for the audited admin re-identification path.

    Built on demand from a known address list rather than stored, so there is no standing table
    linking pseudonyms back to addresses sitting next to the alerts. An attacker who takes the
    alert store gets pseudonyms and nothing to join them against.
    """
    return {pseudonymise_ip(a, preserve_subnet=preserve_subnet): a for a in addresses}


def reidentify(
    pseudonym: str,
    index: dict[str, str],
    *,
    role: str,
    actor: str,
    reason: str,
    audit_sink: object | None = None,
) -> str:
    """Reverse a pseudonym. Admin only, and always audited.

    `reason` is mandatory and is recorded. An audit log showing re-identification without a
    corresponding incident is itself an incident - see docs/RUNBOOK.md.
    """
    if role != "admin":
        raise ReidentificationDenied(
            f"role {role!r} cannot re-identify. IP addresses are personal data (GDPR Art. 4; "
            "Breyer, CJEU C-582/14) and reversal is restricted to admin."
        )
    if not reason.strip():
        raise ValueError("re-identification requires a stated reason; it is written to the audit log")

    address = index.get(pseudonym)
    if address is None:
        raise KeyError(f"unknown pseudonym {pseudonym!r}")

    if audit_sink is not None and hasattr(audit_sink, "append"):
        audit_sink.append(
            {
                "action": "pii.reidentify",
                "actor": actor,
                "role": role,
                "pseudonym": pseudonym,
                "reason": reason,
            }
        )
    return address


def fingerprint_key() -> str:
    """A short, non-reversing fingerprint of the active key.

    Lets an operator confirm which key an alert store was written with - during rotation, mostly -
    without the fingerprint itself being useful for reversing anything.
    """
    return blake2b(_key(), digest_size=8).hexdigest()


def assert_no_raw_addresses(payload: dict[str, object]) -> list[str]:
    """Scan a serialised alert for anything that parses as an IP address.

    The belt-and-braces check behind the test suite: pseudonymisation applied in one code path and
    forgotten in another is exactly the kind of defect that ships. Returns the offending values.
    """
    found: list[str] = []

    def walk(node: object) -> None:
        if isinstance(node, str):
            if node.startswith(PREFIX):
                return
            try:
                ipaddress.ip_address(node)
            except ValueError:
                return
            found.append(node)
        elif isinstance(node, dict):
            for v in node.values():
                walk(v)
        elif isinstance(node, list | tuple):
            for v in node:
                walk(v)

    walk(payload)
    return found
