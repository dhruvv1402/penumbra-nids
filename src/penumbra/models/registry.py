"""Model registry: versioned detectors, a champion pointer, and hashes checked before every load.

The retraining policy in docs/RUNBOOK.md was written as prose - "the registry keeps both artifacts
with SHA-256 digests", "rollback is re-pointing the registry". This is that registry.

Layout, one tree per dataset:

    registry/<dataset>/
        champion.json                 current version + full promotion history
        versions/<version>/
            detector.joblib           the artifact
            metadata.json             DetectorMetadata
            manifest.json             sha256 of every file, parent version, training provenance,
                                      and the canary gate report that justified promotion

Three rules, each closing a specific hole:

  * **Hashes are verified before unpickling.** A joblib file is a pickle, and loading a pickle is
    code execution. An attacker who can write to the model directory otherwise gets RCE in the API
    process on its next restart. The manifest is the allow-list.
  * **Promotion requires a passed gate report**, stored in the manifest of the version promoted.
    "Why is this model in production?" has a recorded answer.
  * **Rollback is a pointer change**, not a retrain and not a redeploy, and it is itself recorded in
    the history. Every version stays on disk.

Nothing here imports the API or the audit log (models/ must run headless; CI enforces it). The CLI
writes the audit entry around each promotion.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from penumbra.models.detector import PenumbraDetector

MANIFEST = "manifest.json"
CHAMPION = "champion.json"


class RegistryError(RuntimeError):
    pass


class TamperedArtifact(RegistryError):
    """A file's digest does not match its manifest. The file is not loaded."""


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class VersionInfo:
    version: str
    created_at: str
    created_by: str
    parent: str | None
    files: dict[str, str]
    training: dict[str, Any] = field(default_factory=dict)
    gate: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


class ModelRegistry:
    def __init__(self, root: Path, dataset: str) -> None:
        self.dataset = dataset.lower()
        self.root = root / self.dataset
        self.versions_dir = self.root / "versions"

    # --- reading ---------------------------------------------------------------------------------

    def versions(self) -> list[VersionInfo]:
        if not self.versions_dir.exists():
            return []
        out = [self.info(p.name) for p in sorted(self.versions_dir.iterdir()) if (p / MANIFEST).exists()]
        return sorted(out, key=lambda v: v.version)

    def info(self, version: str) -> VersionInfo:
        path = self._version_dir(version) / MANIFEST
        if not path.exists():
            raise RegistryError(f"no version {version!r} in the {self.dataset} registry")
        return VersionInfo(**json.loads(path.read_text(encoding="utf-8")))

    def champion_state(self) -> dict[str, Any]:
        path = self.root / CHAMPION
        if not path.exists():
            return {"version": None, "history": []}
        state: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        return state

    def champion(self) -> str | None:
        version = self.champion_state().get("version")
        return str(version) if version else None

    def path(self, version: str) -> Path:
        return self._version_dir(version)

    def verify(self, version: str) -> list[str]:
        """Files whose digest no longer matches the manifest. Empty means intact."""
        info = self.info(version)
        directory = self._version_dir(version)
        bad = []
        for name, digest in info.files.items():
            f = directory / name
            if not f.exists() or sha256(f) != digest:
                bad.append(name)
        return bad

    def load(self, version: str | None = None) -> PenumbraDetector:
        """Load a version (default: the champion), refusing anything whose hashes do not match."""
        version = version or self.champion()
        if version is None:
            raise RegistryError(f"the {self.dataset} registry has no champion")
        tampered = self.verify(version)
        if tampered:
            raise TamperedArtifact(
                f"{self.dataset}/{version}: {tampered} do not match the manifest. Refusing to unpickle."
            )
        return PenumbraDetector.load(self._version_dir(version))

    # --- writing ---------------------------------------------------------------------------------

    def register(
        self,
        detector: PenumbraDetector,
        *,
        created_by: str,
        parent: str | None = None,
        training: dict[str, Any] | None = None,
    ) -> VersionInfo:
        """Save a detector as a new, non-champion version."""
        n = len(self.versions()) + 1
        stamp = datetime.now(UTC)
        version = f"v{n:03d}-{stamp:%Y%m%dT%H%M%S}"
        directory = self._version_dir(version)
        detector.save(directory)

        files = {p.name: sha256(p) for p in sorted(directory.iterdir()) if p.name != MANIFEST}
        info = VersionInfo(
            version=version,
            created_at=stamp.isoformat(),
            created_by=created_by,
            parent=parent,
            files=files,
            training=training or {},
        )
        self._write_manifest(info)
        return info

    def attach_gate(self, version: str, report: dict[str, Any]) -> None:
        info = self.info(version)
        info.gate = report
        self._write_manifest(info)

    def promote(self, version: str, *, approver: str, allow_without_gate: bool = False) -> dict[str, Any]:
        """Make `version` the champion.

        Refused unless the version carries a PASSED gate report - except for the very first
        champion, which has nothing to be compared against and must say so (`allow_without_gate`).
        """
        info = self.info(version)
        if self.verify(version):
            raise TamperedArtifact(f"{version} failed hash verification; it cannot be promoted")

        state = self.champion_state()
        if state.get("version") == version:
            raise RegistryError(f"{version} is already the champion")
        gate_passed = bool(info.gate and info.gate.get("passed"))
        first_champion = allow_without_gate and state.get("version") is None
        if not gate_passed and not first_champion:
            raise RegistryError(
                f"{version} has no passed canary gate report. Run the gate; promotion is never "
                "automatic and never ungated once a champion exists."
            )

        state.setdefault("history", []).append(
            {
                "action": "promote",
                "version": version,
                "previous": state.get("version"),
                "by": approver,
                "at": datetime.now(UTC).isoformat(),
                "gate_passed": gate_passed,
            }
        )
        state["version"] = version
        self._write_champion(state)
        return state

    def rollback(self, *, approver: str) -> dict[str, Any]:
        """Re-point the champion at the version it replaced. A pointer change, recorded."""
        state = self.champion_state()
        current = state.get("version")
        previous = next(
            (h["previous"] for h in reversed(state.get("history", [])) if h["version"] == current),
            None,
        )
        if not previous:
            raise RegistryError("nothing to roll back to")
        # Rolling back is usually done in a hurry, which is exactly when nobody checks what they are
        # rolling back TO. A tampered previous version is refused here as it would be at load.
        if self.verify(previous):
            raise TamperedArtifact(f"cannot roll back to {previous}: it fails hash verification")
        state["history"].append(
            {
                "action": "rollback",
                "version": previous,
                "previous": current,
                "by": approver,
                "at": datetime.now(UTC).isoformat(),
            }
        )
        state["version"] = previous
        self._write_champion(state)
        return state

    # --- internals -------------------------------------------------------------------------------

    def _version_dir(self, version: str) -> Path:
        # Versions are names we generate; refuse anything that could walk out of the tree.
        if not version or "/" in version or "\\" in version or ".." in version:
            raise RegistryError(f"invalid version name {version!r}")
        return self.versions_dir / version

    def _write_manifest(self, info: VersionInfo) -> None:
        path = self._version_dir(info.version) / MANIFEST
        path.write_text(json.dumps(info.to_dict(), indent=2, default=str), encoding="utf-8")

    def _write_champion(self, state: dict[str, Any]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / CHAMPION).write_text(json.dumps(state, indent=2), encoding="utf-8")
