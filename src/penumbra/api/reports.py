"""Serving the generated evaluation reports to the console.

Every number on the console's evaluation and drift pages comes from a file in `artifacts/reports/`
that a CLI command wrote. Nothing is recomputed here and nothing is hardcoded in the frontend —
which means the console cannot drift out of step with the measurements, and a judge asking "where
did that number come from" gets the command that produced it rather than a shrug.

## Why this is not `StaticFiles`

Mounting the directory would serve whatever is in it, including files that arrive there later by
other means. This reads a **known set of report names**, resolves each to a path, and verifies the
resolved path is still inside the report directory. A name like `../../.env` therefore returns 404
rather than a secret.

That check is not theoretical politeness: path traversal through a filename parameter is the
single most common way a read-only endpoint becomes an arbitrary-file-read endpoint.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from penumbra.config import settings


@dataclass(frozen=True)
class ReportSpec:
    """One report the console knows how to render."""

    name: str
    filename: str
    title: str
    command: str
    description: str


# The catalogue is explicit rather than a directory listing, so the console renders reports it
# understands and a stray file in artifacts/ is never served.
CATALOGUE: tuple[ReportSpec, ...] = (
    ReportSpec(
        "eval-unsw",
        "eval_unsw.json",
        "UNSW-NB15 evaluation",
        "penumbra eval --dataset unsw",
        "Binary and per-family metrics with bootstrap CIs, reported with and without the "
        "quarantined testbed features.",
    ),
    ReportSpec(
        "eval-nslkdd",
        "eval_nslkdd.json",
        "NSL-KDD evaluation",
        "penumbra eval --dataset nslkdd",
        "The same metric suite on NSL-KDD.",
    ),
    ReportSpec(
        "audit-unsw",
        "audit_unsw.json",
        "UNSW-NB15 leak audit",
        "penumbra audit --dataset unsw",
        "Single-feature AUC, leaky values, train/test overlap and negative controls. Runs before "
        "any model, because it constrains what may be claimed afterwards.",
    ),
    ReportSpec(
        "audit-nslkdd",
        "audit_nslkdd.json",
        "NSL-KDD leak audit",
        "penumbra audit --dataset nslkdd",
        "The same four checks on NSL-KDD, where the leaky values turn out to be real behaviour.",
    ),
    ReportSpec(
        "unseen17",
        "unseen17_curve.json",
        "Unseen-attack curve",
        "penumbra loafo --dataset nslkdd",
        "Recall on the 3,750 NSL-KDD test rows whose attack type never appears in training, "
        "across false-positive budgets. The headline experiment.",
    ),
    ReportSpec(
        "loafo-unsw",
        "loafo_unsw.json",
        "Leave-one-family-out",
        "penumbra loafo --dataset unsw",
        "Per-family holdout at a matched budget. Reported as a control: the supervised head "
        "already detects the held-out families, so the holdout held nothing out.",
    ),
    ReportSpec(
        "ablation",
        "ablation_unsw.json",
        "Class-imbalance ablation",
        "penumbra ablate --dataset unsw",
        "Every resampling strategy evaluated on the natural distribution.",
    ),
    ReportSpec(
        "smote-leak",
        "smote_leakage.json",
        "SMOTE leakage demonstration",
        "penumbra ablate --dataset unsw",
        "Minority F1 with the sampler fitted inside versus outside the CV fold.",
    ),
    ReportSpec(
        "correlation",
        "correlation_cicids.json",
        "Alert-to-incident correlation",
        "penumbra replay --dataset cicids",
        "Measured on real CICIDS2017 source addresses - the only dataset here that has any.",
    ),
    ReportSpec(
        "conformal",
        "conformal_drift.json",
        "Conformal coverage under drift",
        "penumbra replay --inject-drift abrupt",
        "Empirical coverage against nominal as PSI rises, and abstention as the label-free "
        "version of the same signal.",
    ),
    ReportSpec(
        "conformal-control",
        "conformal_coverage.json",
        "Conformal coverage, three-way control",
        "penumbra eval --dataset nslkdd",
        "Coverage and abstention under exchangeable data, under NSL-KDD's natural shift, and "
        "under injected drift. The control is what makes the drift claim a measurement.",
    ),
    ReportSpec(
        "rules-unsw",
        "mined_rules_unsw.json",
        "Mined detection rules (UNSW)",
        "penumbra rules --dataset unsw",
        "Decision paths validated on held-out data, with and without the quarantined features.",
    ),
    ReportSpec(
        "rules-nslkdd",
        "mined_rules_nslkdd.json",
        "Mined detection rules (NSL-KDD)",
        "penumbra rules --dataset nslkdd",
        "The same, on a dataset where nothing was quarantined.",
    ),
    ReportSpec(
        "sequence",
        "sequence_cicids.json",
        "Sequence context (E6)",
        "penumbra sequence",
        "Per-flow, per-flow plus entity graph, and the CNN/BiGRU sequence head, on identical rows "
        "at a matched false-positive budget.",
    ),
    ReportSpec(
        "loadtest",
        "loadtest_unsw.json",
        "Throughput and latency",
        "penumbra loadtest --dataset unsw",
        "Flows per second and p99 across batch sizes, with the fixed-versus-marginal cost split "
        "and the flows-to-Mbps conversion stated as a conversion.",
    ),
    ReportSpec(
        "adversarial",
        "adversarial_unsw.json",
        "Constrained evasion",
        "penumbra adversarial --dataset unsw",
        "Detection against attacker effort, under realisable perturbations and under the "
        "unconstrained feature-space attack.",
    ),
)

BY_NAME: dict[str, ReportSpec] = {spec.name: spec for spec in CATALOGUE}


def _resolve(spec: ReportSpec) -> Path | None:
    """Resolve a spec to a path inside the report directory, or None.

    The containment check is what keeps this endpoint read-only in the sense that matters. Even
    though every name here comes from a constant, the check runs anyway: the catalogue is the kind
    of thing that grows a caller-supplied entry six months from now.
    """
    root = settings().report_dir.resolve()
    candidate = (root / spec.filename).resolve()
    if not candidate.is_relative_to(root):
        return None
    return candidate if candidate.is_file() else None


def catalogue() -> list[dict[str, Any]]:
    """Every known report, present or not.

    Absent reports are listed with `available: false` and the command that would produce them,
    rather than omitted. A console page that silently shows nothing is indistinguishable from one
    whose data is genuinely empty, and the difference matters on stage.
    """
    out: list[dict[str, Any]] = []
    for spec in CATALOGUE:
        path = _resolve(spec)
        out.append(
            {
                "name": spec.name,
                "title": spec.title,
                "description": spec.description,
                "command": spec.command,
                "available": path is not None,
                "generated_at": path.stat().st_mtime if path else None,
                "bytes": path.stat().st_size if path else 0,
            }
        )
    return out


def load(name: str) -> dict[str, Any] | None:
    """One report's contents, or None if it is unknown or has not been generated."""
    spec = BY_NAME.get(name)
    if spec is None:
        return None
    path = _resolve(spec)
    if path is None:
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        "name": spec.name,
        "title": spec.title,
        "description": spec.description,
        "command": spec.command,
        "generated_at": path.stat().st_mtime,
        "data": payload,
    }
