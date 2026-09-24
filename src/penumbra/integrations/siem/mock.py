"""LocalMockSiem: exactly what Sentinel would receive, written to a file, validated first.

The mock is only worth having if it is as strict as the real thing. A mock that accepts anything
turns "ingestible by Sentinel" into an untested claim, so every record is run through the same ASIM
validation the unit tests use - string-enum severity, conditional ThreatField, integer confidence -
and a record that would fail ingestion is rejected here with the reason, not written.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

from penumbra.alerts.schemas import asim
from penumbra.integrations.siem.base import SendResult

TABLE = "PenumbraAlerts_CL"


class LocalMockSiem:
    name = "local-mock"

    def __init__(self, directory: Path) -> None:
        self.path = directory / f"{TABLE}.jsonl"
        self._lock = threading.Lock()

    def send(self, records: list[dict[str, Any]]) -> SendResult:
        result = SendResult(batches=1 if records else 0)
        good: list[str] = []
        for record in records:
            problems = asim.validate(record)
            if problems:
                result.rejected += 1
                result.problems.extend(f"{record.get('EventUid', '?')}: {p}" for p in problems[:3])
            else:
                good.append(json.dumps(record, default=str))
        if good:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self._lock, self.path.open("a", encoding="utf-8") as f:
                f.write("\n".join(good) + "\n")
        result.accepted = len(good)
        return result

    def count(self) -> int:
        if not self.path.exists():
            return 0
        with self.path.open(encoding="utf-8") as f:
            return sum(1 for _ in f)
