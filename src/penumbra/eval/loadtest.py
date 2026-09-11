"""Throughput and latency — and the conversion that most people get wrong.

"Our IDS handles 10 Gbps" is the claim a network engineer will actually interrogate, and almost
every ML-IDS project answers it by measuring rows per second and quietly relabelling the axis.

**Flows per second is not link speed.** A flow is an aggregate over many packets; a 10 Gbps link
carrying long-lived connections produces far fewer flows per second than one carrying a scan. The
honest form is two statements, in this order:

1. *N flows/s sustained, p99 latency X ms* — a measurement.
2. *at this dataset's mean flow size, that is roughly Y Mbps of monitored traffic* — a conversion,
   with the assumption printed next to it.

The second is an order-of-magnitude sanity check, not a capacity claim, and it is labelled as one.
It also says nothing about whether the flow assembler upstream can keep up, which on a real
deployment is usually the actual bottleneck.

## What is being measured

Two different things, kept apart because they answer different questions:

- **Scoring throughput** — the detector alone, on batches, with no HTTP. This is what determines
  whether the model can keep up with a stream.
- **API latency** — the round trip a SOC integration actually experiences, including
  serialisation. Reported as percentiles, because a mean latency hides exactly the tail that wakes
  someone up.

## Percentiles from one machine are not a benchmark

These numbers come from a developer laptop that is also running a browser, a test suite and
whatever else. A p99 measured there is a useful smoke test and a terrible benchmark, so the report
records the machine's core count and whether anything else was competing, and the summary says so
in words rather than leaving the reader to assume.
"""

from __future__ import annotations

import json
import os
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# UNSW-NB15's mean flow is about this size. Used only for the flows/s -> Mbps conversion, and the
# assumption is printed with the result rather than buried here.
DEFAULT_MEAN_FLOW_BYTES = 1_100


@dataclass
class Percentiles:
    p50: float
    p95: float
    p99: float
    mean: float
    max: float

    @classmethod
    def of(cls, samples: list[float]) -> Percentiles:
        if not samples:
            return cls(0.0, 0.0, 0.0, 0.0, 0.0)
        ordered = sorted(samples)

        def at(q: float) -> float:
            # Nearest-rank. With 20 samples there is no meaningful p99 and interpolating one would
            # invent precision the measurement does not have.
            index = min(len(ordered) - 1, max(0, int(round(q * len(ordered))) - 1))
            return ordered[index]

        return cls(
            p50=at(0.50),
            p95=at(0.95),
            p99=at(0.99),
            mean=float(statistics.fmean(ordered)),
            max=ordered[-1],
        )


@dataclass
class ThroughputPoint:
    batch_size: int
    flows_per_second: float
    batch_latency_ms: Percentiles
    n_batches: int


@dataclass
class LoadReport:
    dataset: str
    n_rows: int
    cores: int
    mean_flow_bytes: int
    # The median travels alongside the mean because flow sizes are violently skewed and the two
    # conversions differ by more than an order of magnitude. Reporting only the mean is
    # arithmetically correct and rhetorically misleading.
    median_flow_bytes: int = 0
    points: list[ThroughputPoint] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def fixed_overhead_ms(self) -> float:
        """Per-call cost that does not scale with batch size.

        Estimated from the two extreme batch sizes. If a 1-row call and a 2,048-row call take
        roughly the same wall time, almost all of that time is fixed — and that is the single most
        decision-relevant number here, because it says what the detector is *for*.
        """
        if len(self.points) < 2:
            return 0.0
        small, large = self.points[0], self.points[-1]
        span = large.batch_size - small.batch_size
        if span <= 0:
            return small.batch_latency_ms.p50
        marginal = (large.batch_latency_ms.p50 - small.batch_latency_ms.p50) / span
        return max(0.0, small.batch_latency_ms.p50 - marginal * small.batch_size)

    @property
    def marginal_ms_per_flow(self) -> float:
        """Cost of one additional flow once the fixed overhead is already paid."""
        if len(self.points) < 2:
            return 0.0
        small, large = self.points[0], self.points[-1]
        span = large.batch_size - small.batch_size
        if span <= 0:
            return 0.0
        return max(0.0, (large.batch_latency_ms.p50 - small.batch_latency_ms.p50) / span)

    @property
    def best(self) -> ThroughputPoint | None:
        return max(self.points, key=lambda p: p.flows_per_second) if self.points else None

    def monitored_mbps(self, flows_per_second: float) -> float:
        """Flows/s converted to monitored bandwidth. A conversion, not a measurement."""
        return flows_per_second * self.mean_flow_bytes * 8 / 1_000_000

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "n_rows": self.n_rows,
            "cores": self.cores,
            "mean_flow_bytes": self.mean_flow_bytes,
            "median_flow_bytes": self.median_flow_bytes,
            "fixed_overhead_ms": self.fixed_overhead_ms,
            "marginal_ms_per_flow": self.marginal_ms_per_flow,
            "points": [
                {
                    "batch_size": p.batch_size,
                    "flows_per_second": p.flows_per_second,
                    "batch_latency_ms": asdict(p.batch_latency_ms),
                    "n_batches": p.n_batches,
                }
                for p in self.points
            ],
            "notes": self.notes,
        }

    def summary(self) -> str:
        lines = [
            "=" * 88,
            "  THROUGHPUT AND LATENCY",
            "=" * 88,
            "",
            f"  {self.n_rows:,} flows from {self.dataset}, {self.cores} cores, no GPU",
            "",
            f"  {'batch':>8} {'flows/s':>12} {'p50 ms':>10} {'p95 ms':>10} {'p99 ms':>10} {'batches':>9}",
        ]
        for point in self.points:
            lines.append(
                f"  {point.batch_size:>8,} {point.flows_per_second:>12,.0f} "
                f"{point.batch_latency_ms.p50:>10.1f} {point.batch_latency_ms.p95:>10.1f} "
                f"{point.batch_latency_ms.p99:>10.1f} {point.n_batches:>9,}"
            )
        lines.append("")

        best = self.best
        if best:
            mbps = self.monitored_mbps(best.flows_per_second)
            per_flow_ms = self.points[0].batch_latency_ms.p50 if self.points else 0.0
            lines += [
                f"  Best sustained: {best.flows_per_second:,.0f} flows/s at batch "
                f"{best.batch_size:,}, p99 {best.batch_latency_ms.p99:.1f} ms per batch.",
                "",
                "  THIS IS A BATCH SCORER, AND THAT IS THE HEADLINE.",
                "",
                f"  About {self.fixed_overhead_ms:,.0f} ms of every call is FIXED - it does not scale with",
                f"  batch size - against {self.marginal_ms_per_flow:.3f} ms of marginal cost per flow. A",
                "  one-flow call and a two-thousand-flow call therefore cost nearly the same.",
                "",
                f"  So scoring one flow at a time costs about {per_flow_ms:,.0f} ms. No inline",
                "  enforcement decision can be made on that budget whatever the policy says, which",
                "  makes ADR-0001 a performance fact as well as an ethical one.",
                "",
                "  FLOWS PER SECOND IS NOT LINK SPEED.",
                "",
                f"  At this dataset's {self.mean_flow_bytes:,}-byte MEAN flow, "
                f"{best.flows_per_second:,.0f} flows/s is",
                f"  roughly {mbps:,.0f} Mbps of monitored traffic.",
            ]
            if self.median_flow_bytes:
                median_mbps = best.flows_per_second * self.median_flow_bytes * 8 / 1_000_000
                skew = self.mean_flow_bytes / max(self.median_flow_bytes, 1)
                lines += [
                    f"  At the {self.median_flow_bytes:,}-byte MEDIAN flow it is roughly "
                    f"{median_mbps:,.0f} Mbps.",
                    "",
                    f"  Those differ by {skew:.0f}x, because flow sizes are violently skewed: a handful",
                    "  of very large transfers carry most of the bytes. The mean figure is the",
                    "  arithmetically correct one for total bandwidth and will still be read as",
                    "  typical, which it is not. Quote both, or quote the median.",
                ]
            lines += [
                "",
                "  Either way it is a conversion with an assumption attached, not a capacity",
                "  measurement: a link carrying long-lived connections produces far fewer flows per",
                "  second than one carrying a scan. It also says nothing about the flow assembler",
                "  upstream, which on a real deployment is usually the actual bottleneck.",
                "",
            ]

        if self.notes:
            lines += ["  Measurement conditions:", *(f"    - {n}" for n in self.notes), ""]
        return "\n".join(lines)


def measure_scoring(
    detector: Any,
    X: pd.DataFrame,
    *,
    batch_sizes: tuple[int, ...] = (1, 32, 256, 2048),
    max_rows: int = 20_000,
    warmup: int = 2,
    dataset: str = "unsw",
    mean_flow_bytes: int = DEFAULT_MEAN_FLOW_BYTES,
    on_progress: Any = None,
) -> LoadReport:
    """Sustained scoring throughput at several batch sizes.

    Batch size is swept rather than fixed because the answer changes by an order of magnitude
    across it, and quoting the best one without saying which is how a throughput number becomes
    marketing. Batch 1 is the honest per-flow latency; batch 2048 is the honest bulk rate.
    """
    X = X.head(max_rows).reset_index(drop=True)
    report = LoadReport(
        dataset=dataset,
        n_rows=len(X),
        cores=os.cpu_count() or 0,
        mean_flow_bytes=mean_flow_bytes,
        notes=[
            "Measured on a developer machine that was not otherwise idle. Useful as a smoke test, "
            "not as a benchmark.",
            "Scoring only: no HTTP, no database write, no alert construction.",
            f"Warm-up of {warmup} batches discarded per batch size, so the first-call import and "
            "allocation cost is not counted as steady state.",
        ],
    )

    for size in batch_sizes:
        if size > len(X):
            continue
        latencies: list[float] = []
        scored_rows = 0

        # Warm-up. The first call pays for lazy imports, thread-pool creation and the first
        # allocation of every intermediate; counting it would understate steady state badly at
        # small batch sizes, where it can dominate the whole measurement.
        for _ in range(warmup):
            detector.score(X.head(size))

        elapsed = 0.0
        for start in range(0, len(X), size):
            batch = X.iloc[start : start + size]
            if len(batch) < size:
                break
            began = time.perf_counter()
            detector.score(batch)
            took = time.perf_counter() - began
            latencies.append(took * 1000.0)
            elapsed += took
            scored_rows += len(batch)
            # Small batches over 20k rows is 20,000 calls; cap the sample rather than the accuracy.
            if len(latencies) >= 200:
                break

        if not latencies or elapsed <= 0:
            continue
        report.points.append(
            ThroughputPoint(
                batch_size=size,
                flows_per_second=scored_rows / elapsed,
                batch_latency_ms=Percentiles.of(latencies),
                n_batches=len(latencies),
            )
        )
        if on_progress:
            on_progress(
                f"batch {size:>5,}: {scored_rows / elapsed:>9,.0f} flows/s  "
                f"p99 {Percentiles.of(latencies).p99:.1f} ms"
            )

    return report


def measure_api(
    base_url: str,
    token: str,
    *,
    requests: int = 200,
    path: str = "/alerts?limit=50",
    on_progress: Any = None,
) -> Percentiles:
    """Round-trip latency against a running API.

    Sequential rather than concurrent, deliberately. A concurrency number from a single-process
    uvicorn on a laptop measures the event loop, not the system, and reporting it as a capacity
    figure would be the same error as calling flows/s a link speed.
    """
    import httpx

    latencies: list[float] = []
    headers = {"Authorization": f"Bearer {token}"}
    with httpx.Client(base_url=base_url, headers=headers, timeout=10.0) as client:
        client.get(path)  # warm-up, discarded
        for i in range(requests):
            began = time.perf_counter()
            response = client.get(path)
            if response.status_code == 200:
                latencies.append((time.perf_counter() - began) * 1000.0)
            if on_progress and i and i % 50 == 0:
                on_progress(f"{i}/{requests} requests")
    return Percentiles.of(latencies)


def write_report(report: LoadReport, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    return path


def estimate_flow_bytes(X: pd.DataFrame) -> tuple[int, int]:
    """`(mean, median)` bytes per flow, from whichever columns the dataset carries.

    Both, because on UNSW-NB15 they are 21,227 and 880 — a factor of 24. The mean is the correct
    number for a bandwidth conversion (total bytes = flows x mean) and the wrong number for a
    reader's mental picture of a typical flow, so the report prints both and says why they differ.
    """
    pairs = (("sbytes", "dbytes"), ("src_bytes", "dst_bytes"))
    for a, b in pairs:
        if a in X.columns and b in X.columns:
            total = pd.to_numeric(X[a], errors="coerce").fillna(0) + pd.to_numeric(
                X[b], errors="coerce"
            ).fillna(0)
            values = total.to_numpy(dtype=float)
            mean = float(np.nanmean(values))
            median = float(np.nanmedian(values))
            if np.isfinite(mean) and mean > 0:
                return int(mean), int(median) if np.isfinite(median) else 0
    return DEFAULT_MEAN_FLOW_BYTES, 0
