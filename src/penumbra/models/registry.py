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
import hmac
import json
import os
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from penumbra.models.detector import PenumbraDetector

MANIFEST = "manifest.json"
CHAMPION = "champion.json"
# A manifest that omits the artifact would verify vacuously. These must always be listed.
REQUIRED_FILES = ("detector.joblib",)
SIGNING_KEY_ENV = "PENUMBRA_MODEL_SIGNING_KEY"


def _signing_key() -> bytes | None:
    key = os.environ.get(SIGNING_KEY_ENV, "")
    return key.encode("utf-8") if key else None


def _sign(key: bytes, document: dict[str, Any]) -> str:
    """HMAC over a whole JSON document, minus its own signature field.

    The whole document, not just the file digests: the promotion decision reads `gate.passed` from
    the same manifest, so signing only the digests left the gate verdict (and the feedback
    provenance) forgeable by anyone who could write the directory.
    """
    body = {k: v for k, v in document.items() if k != "signature"}
    payload = json.dumps(body, sort_keys=True, default=str).encode()
    return hmac.new(key, payload, hashlib.sha256).hexdigest()


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
    # HMAC over (version, files) with a key held OUTSIDE the registry (PENUMBRA_MODEL_SIGNING_KEY).
    # Without it, anyone who can write the version directory can rewrite the manifest to match a
    # swapped artifact; with it, they would also need the key.
    signature: str | None = None

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
        key = _signing_key()
        if key is not None and not hmac.compare_digest(str(state.get("signature", "")), _sign(key, state)):
            # The champion pointer decides what production unpickles. Unsigned, it would allow a
            # forced "rollback" to any version by editing one file.
            raise TamperedArtifact(
                f"{self.dataset} champion pointer fails its signature; refusing to trust it"
            )
        return state

    def champion(self) -> str | None:
        version = self.champion_state().get("version")
        return str(version) if version else None

    def path(self, version: str) -> Path:
        return self._version_dir(version)

    def verify(self, version: str) -> list[str]:
        """Everything wrong with a version's integrity. Empty means intact.

        Four checks, because a digest comparison alone verifies only what the manifest chooses to
        list, and the manifest sits in the same writable directory as the artifact:
          - the artifact must BE listed (an emptied manifest verifies vacuously otherwise)
          - no unlisted file may sit in the version directory
          - every listed digest must match
          - when PENUMBRA_MODEL_SIGNING_KEY is set, the manifest must carry a valid signature, so a
            rewritten manifest fails even when its digests match the swapped file
        """
        info = self.info(version)
        directory = self._version_dir(version)
        bad = [f"{name} (not in manifest)" for name in REQUIRED_FILES if name not in info.files]
        on_disk = {p.name for p in directory.iterdir() if p.is_file() and p.name != MANIFEST}
        bad += [f"{name} (unlisted file)" for name in sorted(on_disk - set(info.files))]
        for name, digest in info.files.items():
            f = directory / name
            if not f.exists() or sha256(f) != digest:
                bad.append(name)
        key = _signing_key()
        if key is not None and (
            not info.signature or not hmac.compare_digest(info.signature, _sign(key, info.to_dict()))
        ):
            bad.append("manifest signature")
        return bad

    def signed(self, version: str) -> bool:
        return bool(self.info(version).signature)

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

    def attach_shadow(self, version: str, report: dict[str, Any]) -> None:
        """Record a shadow-scoring comparison. Evidence for the promoter, not a gate."""
        info = self.info(version)
        info.training = {**info.training, "shadow": report}
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

    @staticmethod
    def lineage(history: list[dict[str, Any]]) -> list[str]:
        """The stack of champions, oldest first, replayed from the history.

        A promotion pushes, a rollback pops. Looking up "the entry whose version is current" instead
        finds the previous ROLLBACK when rolling back twice, and its `previous` is the bad model:
        v1 -> v2 -> v3, rollback -> v2, rollback -> v3 again. Replaying the stack cannot do that.
        """
        stack: list[str] = []
        for h in history:
            if h.get("action") == "promote":
                stack.append(str(h["version"]))
            elif h.get("action") == "rollback" and len(stack) > 1:
                stack.pop()
        return stack

    def rollback(self, *, approver: str) -> dict[str, Any]:
        """Re-point the champion at the version it replaced. A pointer change, recorded."""
        state = self.champion_state()
        current = state.get("version")
        stack = self.lineage(state.get("history", []))
        previous = stack[-2] if len(stack) > 1 else None
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
        # Re-signed on every write, so attaching a gate or shadow report keeps the manifest valid and
        # an edit made anywhere else does not.
        key = _signing_key()
        info.signature = _sign(key, info.to_dict()) if key is not None else None
        path = self._version_dir(info.version) / MANIFEST
        path.write_text(json.dumps(info.to_dict(), indent=2, default=str), encoding="utf-8")

    def _write_champion(self, state: dict[str, Any]) -> None:
        key = _signing_key()
        state.pop("signature", None)
        if key is not None:
            state["signature"] = _sign(key, state)
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / CHAMPION).write_text(json.dumps(state, indent=2), encoding="utf-8")
