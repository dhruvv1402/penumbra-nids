"""The dataset registry: where each file comes from, how big it should be, and what it hashes to.

Every URL here was probed live before being written down. Two of the obvious sources are dead or
wrong and are recorded as such, because the failure modes are silent:

  * CICIDS2017's official link (205.174.165.80/.../MachineLearningCSV.zip) answers 200 and then
    redirects to the UNB index page. You save 108 KB of HTML named `.zip` and find out later.
  * The HuggingFace mirror `Mireu-Lab/UNSW-NB15` has train and test filenames SWAPPED. Its
    "train.csv" is the 82k testing split. A run against it trains on the small half and looks
    plausible the whole way through.

Hence: size is checked on download, SHA256 is pinned after the first fetch, and row counts are
asserted after parsing. Three independent chances to notice we got the wrong bytes.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

MANIFEST_PATH: Final[Path] = Path(__file__).resolve().parents[3] / "data" / "manifest.json"


@dataclass(frozen=True)
class RemoteFile:
    key: str
    url: str
    filename: str
    expected_bytes: int
    dataset: str
    # Rows after parsing, when known. Asserted by the loader — this is what catches a mirror that
    # serves the right-sized file with the wrong contents.
    expected_rows: int | None = None
    note: str = ""


# --- NSL-KDD -------------------------------------------------------------------------------------
_NSLKDD_BASE = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master"

NSLKDD_FILES: Final[list[RemoteFile]] = [
    RemoteFile(
        key="nslkdd_train",
        # The '+' MUST stay percent-encoded. Some clients decode a literal '+' in a path as a
        # space and the request 404s.
        url=f"{_NSLKDD_BASE}/KDDTrain%2B.txt",
        filename="KDDTrain+.txt",
        expected_bytes=19_109_424,
        expected_rows=125_973,
        dataset="nslkdd",
    ),
    RemoteFile(
        key="nslkdd_test",
        url=f"{_NSLKDD_BASE}/KDDTest%2B.txt",
        filename="KDDTest+.txt",
        expected_bytes=3_441_513,
        expected_rows=22_544,
        dataset="nslkdd",
        note="Contains 17 attack types absent from train (3,750 rows). Never merge with train.",
    ),
    RemoteFile(
        key="nslkdd_train_20pct",
        url=f"{_NSLKDD_BASE}/KDDTrain%2B_20Percent.txt",
        filename="KDDTrain+_20Percent.txt",
        expected_bytes=3_822_033,
        expected_rows=25_192,
        dataset="nslkdd",
        note="Fast iteration during development.",
    ),
    RemoteFile(
        key="nslkdd_test21",
        url=f"{_NSLKDD_BASE}/KDDTest-21.txt",
        filename="KDDTest-21.txt",
        expected_bytes=1_814_092,
        expected_rows=11_850,
        dataset="nslkdd",
        note="The harder subset: rows that most classic learners got wrong.",
    ),
]

# --- UNSW-NB15 -----------------------------------------------------------------------------------
_UNSW_BASE = "https://huggingface.co/datasets/dileepa0011/unsw-nb15/resolve/main"

UNSW_FILES: Final[list[RemoteFile]] = [
    RemoteFile(
        key="unsw_train",
        url=f"{_UNSW_BASE}/UNSW_NB15_training-set.csv",
        filename="UNSW_NB15_training-set.csv",
        expected_bytes=32_293_018,
        expected_rows=175_341,
        dataset="unsw",
        note="Larger than the testing split - that is correct and surprises people. UTF-8 BOM.",
    ),
    RemoteFile(
        key="unsw_test",
        url=f"{_UNSW_BASE}/UNSW_NB15_testing-set.csv",
        filename="UNSW_NB15_testing-set.csv",
        expected_bytes=15_380_800,
        expected_rows=82_332,
        dataset="unsw",
    ),
]

# --- CICIDS2017 ----------------------------------------------------------------------------------
# The Engelen/Rimmer/Joosen re-release (WTMC 2021 + IEEE CNS 2022). Built with a fixed
# CICFlowMeter and corrected ground truth; more than 20% of flows changed label or boundary.
# Preferred over the original precisely because the original's labels are known-wrong.
CICIDS_FILES: Final[list[RemoteFile]] = [
    RemoteFile(
        key="cicids_improved",
        url="https://intrusion-detection.distrinet-research.be/CNS2022/Datasets/CICIDS2017_improved.zip",
        filename="CICIDS2017_improved.zip",
        expected_bytes=343_549_013,
        dataset="cicids",
        note="Zip of monday..friday.csv. Carries Src/Dst IP + Timestamp, unlike UNSW.",
    ),
]

# --- MITRE ATT&CK corpus (for the RAG copilot) ---------------------------------------------------
ATTACK_FILES: Final[list[RemoteFile]] = [
    RemoteFile(
        key="attack_stix",
        url=(
            "https://raw.githubusercontent.com/mitre-attack/attack-stix-data/master/"
            "enterprise-attack/enterprise-attack.json"
        ),
        filename="enterprise-attack.json",
        expected_bytes=53_835_637,
        dataset="attack",
        note="Reduced to a <5 MB JSONL at build time; never parsed per request.",
    ),
]

ALL_FILES: Final[list[RemoteFile]] = [*NSLKDD_FILES, *UNSW_FILES, *CICIDS_FILES, *ATTACK_FILES]

BY_DATASET: Final[dict[str, list[RemoteFile]]] = {}
for _f in ALL_FILES:
    BY_DATASET.setdefault(_f.dataset, []).append(_f)


# --- Known-bad sources, recorded so nobody re-discovers them the hard way ------------------------
@dataclass(frozen=True)
class PoisonedSource:
    url: str
    problem: str


KNOWN_BAD_SOURCES: Final[list[PoisonedSource]] = [
    PoisonedSource(
        url="http://205.174.165.80/CICDataset/CIC-IDS-2017/Dataset/CIC-IDS-2017/CSVs/MachineLearningCSV.zip",
        problem=(
            "Answers HTTP 200 but 301->302 redirects to the UNB datasets index. You save 108 KB "
            "of HTML with a .zip extension and no error."
        ),
    ),
    PoisonedSource(
        url="https://huggingface.co/datasets/Mireu-Lab/UNSW-NB15",
        problem=(
            "train.csv and test.csv are SWAPPED - their 'train' is the 82k testing split. "
            "Training against it silently uses the small half."
        ),
    ),
    PoisonedSource(
        url="https://huggingface.co/datasets/bastyje/UNSW-NB15",
        problem=(
            "Not the pre-split edition: 882 MB / 220 MB is the full 2.54M-record UNSW_NB15_1-4 "
            "raw set. Different schema, not comparable to published 175k/82k results."
        ),
    ),
    PoisonedSource(
        url="https://cloudstor.aarnet.edu.au/plus/index.php/s/2DhnLGDdEECo4ys",
        problem="AARNet retired CloudStor. Connection failure, not a 404.",
    ),
]


def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


@dataclass
class ManifestEntry:
    key: str
    filename: str
    url: str
    bytes: int
    sha256: str
    expected_rows: int | None = None


@dataclass
class Manifest:
    """The checked-in record of exactly which bytes produced our numbers."""

    entries: dict[str, ManifestEntry] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path = MANIFEST_PATH) -> Manifest:
        if not path.exists():
            return cls()
        raw = json.loads(path.read_text(encoding="utf-8"))
        return cls(entries={k: ManifestEntry(**v) for k, v in raw.get("entries", {}).items()})

    def save(self, path: Path = MANIFEST_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "_comment": (
                "Generated by `penumbra data fetch`. Pins the exact bytes behind every number in "
                "the report. Do not hand-edit; re-fetch instead."
            ),
            "entries": {k: v.__dict__ for k, v in sorted(self.entries.items())},
        }
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    def record(self, spec: RemoteFile, path: Path) -> ManifestEntry:
        entry = ManifestEntry(
            key=spec.key,
            filename=spec.filename,
            url=spec.url,
            bytes=path.stat().st_size,
            sha256=sha256_of(path),
            expected_rows=spec.expected_rows,
        )
        self.entries[spec.key] = entry
        return entry

    def verify(self, spec: RemoteFile, path: Path) -> tuple[bool, str]:
        """Check a local file against the pinned hash. Returns (ok, human-readable reason)."""
        if not path.exists():
            return False, "missing"
        size = path.stat().st_size
        known = self.entries.get(spec.key)
        if known is None:
            if size != spec.expected_bytes:
                return False, f"size {size:,} != expected {spec.expected_bytes:,}"
            return True, "size ok (not yet pinned)"
        if size != known.bytes:
            return False, f"size {size:,} != pinned {known.bytes:,}"
        actual = sha256_of(path)
        if actual != known.sha256:
            return False, f"sha256 {actual[:12]}... != pinned {known.sha256[:12]}..."
        return True, "sha256 ok"
