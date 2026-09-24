"""SIEM connectors. `connector()` picks one from configuration; the mock is the default."""

from __future__ import annotations

import os
from pathlib import Path

from penumbra.integrations.siem.base import NotConfigured, SendResult, SiemConnector
from penumbra.integrations.siem.mock import LocalMockSiem


def connector(artifact_root: Path, kind: str | None = None) -> SiemConnector | None:
    """PENUMBRA_SIEM = mock (default) | sentinel | none.

    `sentinel` without credentials raises NotConfigured instead of quietly writing to the mock:
    silently pretending to talk to a SIEM is worse than failing to start.
    """
    kind = (kind or os.environ.get("PENUMBRA_SIEM") or "mock").lower()
    if kind == "none":
        return None
    if kind == "mock":
        return LocalMockSiem(artifact_root / "siem")
    if kind == "sentinel":
        from penumbra.integrations.siem.sentinel import AzureSentinelSiem

        return AzureSentinelSiem()
    raise NotConfigured(f"PENUMBRA_SIEM={kind!r}; expected mock, sentinel or none")


__all__ = ["LocalMockSiem", "NotConfigured", "SendResult", "SiemConnector", "connector"]
