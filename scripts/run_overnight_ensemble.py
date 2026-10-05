"""Overnight E9 Ensemble Runner for Penumbra NIDS.

Runs UNSW, NSL-KDD, and CICIDS sequentially with:
1. Live console tee-logging to both terminal and `logs/ensemble_<timestamp>.log`.
2. Checkpoint auto-resumption and elapsed timing per dataset.
3. Verification of final report artifacts upon completion.
"""

from __future__ import annotations

import argparse
import datetime
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
LOGS_DIR = REPO_ROOT / "logs"


def get_penumbra_command(dataset: str, profile: str) -> list[str]:
    # Check if 'penumbra' executable is in PATH
    penumbra_bin = shutil.which("penumbra")
    if penumbra_bin:
        return [penumbra_bin, "ensemble", "-d", dataset, "--profile", profile]
    # Fallback to python -m penumbra.cli
    return [sys.executable, "-m", "penumbra.cli", "ensemble", "-d", dataset, "--profile", profile]


def tee_run(cmd: list[str], log_file: Path) -> int:
    """Run command, teeing output to both stdout and log_file with immediate flush."""
    with open(log_file, "a", encoding="utf-8") as f:
        msg = f"\n[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Executing: {' '.join(cmd)}\n"
        print(msg, flush=True)
        f.write(msg)
        f.flush()

        process = subprocess.Popen(
            cmd,
            cwd=str(REPO_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            encoding="utf-8",
            errors="replace",
        )

        if process.stdout:
            for line in iter(process.stdout.readline, ""):
                sys.stdout.write(line)
                sys.stdout.flush()
                f.write(line)
                f.flush()
            process.stdout.close()

        ret = process.wait()
        end_msg = f"\n[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Exit code: {ret}\n"
        print(end_msg, flush=True)
        f.write(end_msg)
        f.flush()
        return ret


def main() -> None:
    parser = argparse.ArgumentParser(description="Overnight E9 Ensemble Runner")
    parser.add_argument("--profile", default="laptop", choices=["laptop", "workstation", "gpu"])
    args = parser.parse_args()

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    main_log = LOGS_DIR / f"ensemble_overnight_{timestamp}.log"

    print("=" * 70)
    print("PENUMBRA OVERNIGHT ENSEMBLE TRAINING")
    print(f"Profile:     {args.profile}")
    print(f"Start time:  {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Logging to:  {main_log}")
    print("=" * 70)

    datasets = ["unsw", "nslkdd", "cicids"]
    overall_start = time.time()
    results = {}

    for idx, ds in enumerate(datasets, start=1):
        print(f"\n{'='*30} STEP {idx}/3: DATASET [{ds.upper()}] {'='*30}")
        ds_start = time.time()
        cmd = get_penumbra_command(ds, args.profile)
        code = tee_run(cmd, main_log)
        elapsed = time.time() - ds_start

        hours, rem = divmod(elapsed, 3600)
        mins, secs = divmod(rem, 60)
        duration_str = f"{int(hours)}h {int(mins)}m {int(secs)}s"

        results[ds] = {"code": code, "duration": duration_str}

        if code != 0:
            print(f"\n[ERROR] Dataset {ds} exited with non-zero code {code}.")
            print("[NOTICE] Checkpoints are saved. You can re-run to resume.")
            break
        else:
            print(f"[SUCCESS] {ds} finished in {duration_str}")

    total_elapsed = time.time() - overall_start
    thours, trem = divmod(total_elapsed, 3600)
    tmins, tsecs = divmod(trem, 60)

    summary = [
        "\n" + "=" * 70,
        "OVERNIGHT RUN SUMMARY",
        f"Completed at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Total duration: {int(thours)}h {int(tmins)}m {int(tsecs)}s",
        "-" * 70,
    ]
    for ds, info in results.items():
        status = "PASSED" if info["code"] == 0 else f"FAILED (code {info['code']})"
        summary.append(f"  - {ds.upper():<10}: {status:<15} (took {info['duration']})")

    summary.append("-" * 70)
    summary.append("Artifact Reports Check:")
    reports_dir = REPO_ROOT / "artifacts" / "reports"
    for ds in datasets:
        rep_file = reports_dir / f"ensemble_{ds}.json"
        if rep_file.exists():
            summary.append(f"  [OK] {rep_file} ({rep_file.stat().st_size / 1024:.1f} KB)")
        else:
            summary.append(f"  [MISSING] {rep_file}")
    summary.append("=" * 70 + "\n")

    summary_text = "\n".join(summary)
    print(summary_text)
    with open(main_log, "a", encoding="utf-8") as f:
        f.write(summary_text)


if __name__ == "__main__":
    main()
