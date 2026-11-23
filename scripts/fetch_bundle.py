"""Download the deployment bundle from the Hugging Face model repo. Standard library only.

    python scripts/fetch_bundle.py --repo <user>/penumbra-nids --dest /app [--models nslkdd cicids]

The repo is written by scripts/publish_hf.py. Lays it out the way the API reads it:

    <dest>/bundle/models/<dataset>/{detector.joblib,metadata.json}   PENUMBRA_ARTIFACT_ROOT=<dest>/bundle
    <dest>/bundle/samples/<dataset>_test_sample.csv.gz
    <dest>/bundle/reports/*.json
    <dest>/data/interim/attack_techniques.jsonl                       PENUMBRA_DATA_ROOT=<dest>/data

Every file stored in LFS (the detectors) is checked against the SHA-256 the Hub reports for it; a
mismatch stops the build rather than shipping bytes nobody verified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.request
from pathlib import Path

HUB = "https://huggingface.co"


def get(url: str, attempts: int = 4) -> bytes:
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(url, timeout=120) as resp:  # noqa: S310 - fixed https host
                return resp.read()
        except OSError as exc:
            if attempt == attempts:
                raise
            print(f"  retry {attempt} for {url.rsplit('/', 1)[-1]}: {exc}", file=sys.stderr)
            time.sleep(2 * attempt)
    raise AssertionError("unreachable")


def wanted(path: str, models: set[str]) -> str | None:
    """Where a repo file goes under <dest>, or None to skip it."""
    parts = path.split("/")
    if parts[0] == "models" and len(parts) == 3:
        return f"bundle/{path}" if parts[1] in models else None
    if parts[0] == "samples" and len(parts) == 2:
        return f"bundle/{path}" if parts[1].split("_", 1)[0] in models else None
    if parts[0] == "reports" and len(parts) == 2:
        return f"bundle/{path}"
    if path.startswith("data/interim/"):
        return path
    return None


def main(repo: str, dest: Path, models: set[str], revision: str) -> None:
    info = json.loads(get(f"{HUB}/api/models/{repo}/revision/{revision}?blobs=true"))
    fetched = 0
    for sibling in info["siblings"]:
        rel = wanted(sibling["rfilename"], models)
        if rel is None:
            continue
        data = get(f"{HUB}/{repo}/resolve/{info['sha']}/{sibling['rfilename']}")
        lfs = sibling.get("lfs")
        if lfs and hashlib.sha256(data).hexdigest() != lfs["sha256"]:
            sys.exit(f"{sibling['rfilename']}: SHA-256 does not match the Hub's record")
        out = dest / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(data)
        fetched += 1
    missing = [m for m in models if not (dest / "bundle" / "models" / m / "detector.joblib").exists()]
    if missing:
        sys.exit(f"no detector in {repo} for: {', '.join(missing)}")
    print(f"{fetched} files from {repo}@{info['sha'][:10]} (models: {', '.join(sorted(models))})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--dest", type=Path, required=True)
    parser.add_argument("--models", nargs="+", default=["nslkdd", "cicids"])
    parser.add_argument("--revision", default="main")
    args = parser.parse_args()
    main(args.repo, args.dest, set(args.models), args.revision)
