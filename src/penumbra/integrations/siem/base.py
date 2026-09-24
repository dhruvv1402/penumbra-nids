"""The SIEM seam.

Everything that leaves Penumbra for a SIEM goes through `SiemConnector`. Two implementations:

  LocalMockSiem      the default. Validates every record against the ASIM rules Sentinel enforces
                     and appends what WOULD have been sent to a local JSONL file. No network.
  AzureSentinelSiem  the Logs Ingestion API against a real workspace. Raises `NotConfigured` until
                     credentials exist, and never falls back to the mock silently: a demo that
                     claims to be talking to Sentinel while writing a file would be a lie.

The decision recorded in the plan was "simulate now, real credits later". The seam is what makes
the second half a configuration change rather than a rewrite.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


class NotConfigured(RuntimeError):
    """The connector lacks what it needs, and says exactly what that is."""


@dataclass
class SendResult:
    accepted: int = 0
    rejected: int = 0
    batches: int = 0
    problems: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


@runtime_checkable
class SiemConnector(Protocol):
    name: str

    def send(self, records: list[dict[str, Any]]) -> SendResult: ...
