"""Fetch datasets, verify them, and pin what we got.

Uses urllib rather than requests so that `penumbra data fetch` works before the optional extras are
installed. Downloads stream to a `.part` file and are renamed only after the size check passes, so
an interrupted transfer can never masquerade as a complete dataset.
"""

from __future__ import annotations

import shutil
import urllib.error
import urllib.request
import zipfile
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

from penumbra.config import settings
from penumbra.data.manifest import Manifest, RemoteFile

_UA = "penumbra-nids/0.1 (academic; hackathon project)"
_CHUNK = 1 << 20


@dataclass
class FetchResult:
    spec: RemoteFile
    path: Path
    status: str  # "downloaded" | "cached" | "repaired" | "failed"
    detail: str = ""

    @property
    def ok(self) -> bool:
        return self.status != "failed"


def _download(url: str, dest: Path, on_progress: Callable[[int, int], None] | None = None) -> int:
    """Stream a URL to dest. Returns bytes written. Raises on transport failure."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    written = 0
    # HuggingFace `resolve/main` answers 302 to a CDN URL; urlopen follows redirects by default.
    with urllib.request.urlopen(req, timeout=120) as resp:  # noqa: S310 - fixed allowlist of URLs
        total = int(resp.headers.get("Content-Length") or 0)
        with part.open("wb") as fh:
            while chunk := resp.read(_CHUNK):
                fh.write(chunk)
                written += len(chunk)
                if on_progress:
                    on_progress(written, total)
    part.replace(dest)
    return written


def fetch_one(
    spec: RemoteFile,
    manifest: Manifest,
    *,
    force: bool = False,
    on_progress: Callable[[int, int], None] | None = None,
) -> FetchResult:
    dest = settings().raw_dir / spec.dataset / spec.filename

    if dest.exists() and not force:
        ok, why = manifest.verify(spec, dest)
        if ok:
            return FetchResult(spec, dest, "cached", why)
        # A file that is present but wrong is worse than one that is absent, because it will be
        # used. Replace it rather than reporting success.
        detail_prefix = f"local copy rejected ({why}); re-downloading"
    else:
        detail_prefix = ""

    try:
        written = _download(spec.url, dest, on_progress)
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
        return FetchResult(spec, dest, "failed", f"{type(exc).__name__}: {exc}")

    if spec.expected_bytes and written != spec.expected_bytes:
        # The classic symptom of a redirect-to-landing-page: a 200 with a plausible-looking body
        # that is three orders of magnitude too small.
        dest.unlink(missing_ok=True)
        return FetchResult(
            spec,
            dest,
            "failed",
            f"got {written:,} bytes, expected {spec.expected_bytes:,}. "
            "A large shortfall usually means the URL redirected to an HTML page.",
        )

    manifest.record(spec, dest)
    status = "repaired" if detail_prefix else "downloaded"
    return FetchResult(spec, dest, status, detail_prefix or f"{written:,} bytes")


def fetch_all(
    specs: Iterable[RemoteFile],
    *,
    force: bool = False,
    on_progress: Callable[[int, int], None] | None = None,
    on_result: Callable[[FetchResult], None] | None = None,
) -> list[FetchResult]:
    settings().ensure_dirs()
    manifest = Manifest.load()
    results: list[FetchResult] = []
    for spec in specs:
        res = fetch_one(spec, manifest, force=force, on_progress=on_progress)
        results.append(res)
        if on_result:
            on_result(res)
    manifest.save()
    return results


def extract_zip(archive: Path, dest_dir: Path, *, members: list[str] | None = None) -> list[Path]:
    """Extract an archive, refusing any entry that would escape dest_dir.

    zipfile.extractall does not protect against absolute paths or `..` traversal in member names.
    Our archives are trusted, but a downloader that writes wherever a remote file tells it to is a
    bug regardless of the current source.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    root = dest_dir.resolve()
    with zipfile.ZipFile(archive) as zf:
        names = members or zf.namelist()
        for name in names:
            if name.endswith("/"):
                continue
            target = (root / Path(name).name).resolve()
            if not str(target).startswith(str(root)):
                raise ValueError(f"refusing to extract outside destination: {name!r}")
            if target.exists():
                written.append(target)
                continue
            with zf.open(name) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst, _CHUNK)
            written.append(target)
    return written
