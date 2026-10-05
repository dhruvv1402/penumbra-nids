"""Standalone dataset downloader for Penumbra NIDS.
Requires standard library only (no pip dependencies needed).

Downloads:
- UNSW-NB15
- NSL-KDD
- CICIDS2017 (Improved release, automatically unzipped to raw/cicids/improved)
- MITRE ATT&CK (Optional, for copilot)
"""

from __future__ import annotations

import argparse
import os
import sys
import urllib.request
import zipfile
from pathlib import Path

# URLs and definitions
DATASETS = [
    # NSL-KDD
    {
        "name": "NSL-KDD Train",
        "dataset": "nslkdd",
        "filename": "KDDTrain+.txt",
        "url": "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain%2B.txt",
        "size": 19_109_424,
    },
    {
        "name": "NSL-KDD Test",
        "dataset": "nslkdd",
        "filename": "KDDTest+.txt",
        "url": "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest%2B.txt",
        "size": 3_441_513,
    },
    {
        "name": "NSL-KDD Train 20%",
        "dataset": "nslkdd",
        "filename": "KDDTrain+_20Percent.txt",
        "url": "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain%2B_20Percent.txt",
        "size": 3_822_033,
    },
    {
        "name": "NSL-KDD Test-21",
        "dataset": "nslkdd",
        "filename": "KDDTest-21.txt",
        "url": "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest-21.txt",
        "size": 1_814_092,
    },
    # UNSW-NB15
    {
        "name": "UNSW-NB15 Train",
        "dataset": "unsw",
        "filename": "UNSW_NB15_training-set.csv",
        "url": "https://huggingface.co/datasets/dileepa0011/unsw-nb15/resolve/main/UNSW_NB15_training-set.csv",
        "size": 32_293_018,
    },
    {
        "name": "UNSW-NB15 Test",
        "dataset": "unsw",
        "filename": "UNSW_NB15_testing-set.csv",
        "url": "https://huggingface.co/datasets/dileepa0011/unsw-nb15/resolve/main/UNSW_NB15_testing-set.csv",
        "size": 15_380_800,
    },
    # CICIDS2017 Improved
    {
        "name": "CICIDS2017 Improved Zip",
        "dataset": "cicids",
        "filename": "CICIDS2017_improved.zip",
        "url": "https://intrusion-detection.distrinet-research.be/CNS2022/Datasets/CICIDS2017_improved.zip",
        "size": 343_549_013,
        "is_zip": True,
        "extract_to": "improved",
    },
]

REPO_ROOT = Path(__file__).resolve().parents[1]


def get_default_data_root() -> Path:
    env_root = os.getenv("PENUMBRA_DATA_ROOT")
    if env_root:
        return Path(env_root)
    g = Path("G:/penumbra-data")
    if g.drive and Path(g.drive + "/").exists():
        return g
    return REPO_ROOT / "data"


def download_file(url: str, dest_path: Path, expected_size: int | None = None) -> None:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if dest_path.exists():
        actual_size = dest_path.stat().st_size
        if expected_size and actual_size == expected_size:
            print(f"  [CACHED] {dest_path.name} ({actual_size / 1e6:.1f} MB)")
            return
        elif not expected_size and actual_size > 0:
            print(f"  [CACHED] {dest_path.name} ({actual_size / 1e6:.1f} MB)")
            return

    print(f"  [DOWNLOADING] {dest_path.name} from {url}...")
    headers = {"User-Agent": "PenumbraDatasetFetcher/1.0"}
    req = urllib.request.Request(url, headers=headers)

    temp_path = dest_path.with_suffix(dest_path.suffix + ".part")
    try:
        with urllib.request.urlopen(req, timeout=60) as response, open(temp_path, "wb") as out_file:
            total_size = int(response.headers.get("content-length", 0)) or expected_size or 0
            downloaded = 0
            block_size = 1024 * 1024  # 1MB blocks

            while True:
                buffer = response.read(block_size)
                if not buffer:
                    break
                downloaded += len(buffer)
                out_file.write(buffer)

                if total_size > 0:
                    percent = downloaded * 100 / total_size
                    mb_down = downloaded / (1024 * 1024)
                    mb_tot = total_size / (1024 * 1024)
                    sys.stdout.write(f"\r    -> {mb_down:.1f}/{mb_tot:.1f} MB ({percent:.1f}%)")
                    sys.stdout.flush()
                else:
                    mb_down = downloaded / (1024 * 1024)
                    sys.stdout.write(f"\r    -> {mb_down:.1f} MB downloaded")
                    sys.stdout.flush()
            print()

        if temp_path.exists():
            temp_path.replace(dest_path)
        print(f"  [DONE] Saved to {dest_path}")
    except Exception as e:
        if temp_path.exists():
            temp_path.unlink()
        raise RuntimeError(f"Failed to download {url}: {e}") from e


def extract_zip(archive: Path, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    print(f"  [EXTRACTING] {archive.name} -> {dest_dir}...")
    with zipfile.ZipFile(archive, "r") as zf:
        for member in zf.infolist():
            if member.is_dir():
                continue
            filename = Path(member.filename).name
            target = dest_dir / filename
            if target.exists() and target.stat().st_size > 0:
                continue
            with zf.open(member) as source, open(target, "wb") as target_file:
                target_file.write(source.read())
    print(f"  [DONE] Extracted files to {dest_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Penumbra datasets.")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Target data directory (defaults to $PENUMBRA_DATA_ROOT, G:/penumbra-data, or ./data)",
    )
    parser.add_argument(
        "--dataset",
        choices=["all", "unsw", "nslkdd", "cicids"],
        default="all",
        help="Which dataset to download",
    )
    args = parser.parse_args()

    data_root = Path(args.data_dir) if args.data_dir else get_default_data_root()
    raw_root = data_root / "raw"
    print(f"Target data raw root: {raw_root.resolve()}")
    print("=" * 60)

    for item in DATASETS:
        if args.dataset != "all" and item["dataset"] != args.dataset:
            continue

        print(f"\nProcessing: {item['name']}")
        dest_dir = raw_root / item["dataset"]
        dest_file = dest_dir / item["filename"]

        download_file(item["url"], dest_file, item.get("size"))

        if item.get("is_zip") and item.get("extract_to"):
            extract_dir = dest_dir / item["extract_to"]
            extract_zip(dest_file, extract_dir)

    print("\n" + "=" * 60)
    print("All datasets are ready in:")
    print(f"  {raw_root.resolve()}")


if __name__ == "__main__":
    main()
