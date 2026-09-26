"""The detector with its forests flattened: same scores, a fraction of the fixed cost per call.

Where a small batch spent its time in `PenumbraDetector.score` (UNSW, one flow, 8 cores): the
supervised forest and the IsolationForest each walked 200-300 trees in a Python loop, 11-17 ms
apiece however few rows there were. `models.flat_forest` pushes a batch down every tree at once, one
vectorised step per depth level: 0.29 ms and 0.07 ms for one row. This module swaps those two
forests in and routes everything else (preprocessing, the autoencoder and Mahalanobis, percentile
fusion, the gate, conformal, the family model) through the detector's own code via
`PenumbraDetector.assemble`, so there is one implementation of the decision logic, not two.

**Only below a measured batch size.** Flattening wins on small batches and loses on large ones,
where the (rows x trees) working set outgrows the cache and sklearn's per-tree Cython is faster -
on UNSW the supervised forest crosses over between 128 and 512 rows, the IsolationForest between
512 and 2048. `compile_detector` times both paths on the caller's rows and records the largest batch
size at which the flat path is at least as fast; above it, sklearn runs. Measured, not hard-coded,
because it depends on the model and the machine.

**Parity is a precondition.** `compile_detector` scores a check set with the flat forests forced on
for every row and refuses (`CompileRefused`) if more than `max_flip_rate` of rows change `fired`,
predicted family or conformal outcome. The flat forests are exact up to summation order, so the
expected count is zero; the check is there so that stays a measurement.

The family forest stays in sklearn: `assemble` only asks it about rows that became alerts, and a
ten-class copy of four million nodes is 320 MB to save milliseconds on a minority of calls. So does
anything the flat forests cannot score exactly (XGBoost, multiclass, NaN input): that component is
simply not compiled, or the batch that hits it falls back.

Compiled in memory from a detector already loaded - through the registry, already hash-verified -
in a fraction of a second. Nothing new is written to disk, so there is no second artifact to sign.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd

from penumbra.models import supervised
from penumbra.models.flat_forest import FlatBinaryForest, FlatIsolationForest, Unsupported

# Batch sizes at which each component is timed both ways during compilation.
PROBES = (1, 8, 32, 128, 512, 2048)


class CompileRefused(RuntimeError):
    """The compiled detector disagreed with the original on too many decisions."""


class CompiledDetector:
    """`PenumbraDetector.score`, with flat forests below a measured batch size. Else delegated."""

    def __init__(self, detector: Any) -> None:
        self.detector = detector
        self._supervised: FlatBinaryForest | None = None
        self._iforest: FlatIsolationForest | None = None
        self.skipped: dict[str, str] = {}
        try:
            self._supervised = FlatBinaryForest(detector.supervised_model.named_steps["clf"])
        except Unsupported as exc:
            self.skipped["supervised"] = str(exc)
        iforest = self._iforest_detector()
        if iforest is not None:
            try:
                self._iforest = FlatIsolationForest(iforest.model)
            except Unsupported as exc:
                self.skipped["iforest"] = str(exc)
        self.compiled = [n for n, f in (("supervised", self._supervised), ("iforest", self._iforest)) if f]
        # Largest batch each flat forest is used for; set by `compile_detector` from measurement.
        self.limits: dict[str, int] = {name: 0 for name in self.compiled}
        self.fallbacks = 0
        self.parity: dict[str, Any] = {}

    def __getattr__(self, name: str) -> Any:
        # policy, metadata, gate, novelty, ranked_importances ... - anything not overridden here.
        if name in {"detector", "limits", "_supervised", "_iforest"}:
            raise AttributeError(name)
        return getattr(self.detector, name)

    def _iforest_detector(self) -> Any:
        return next((d for d in self.detector.novelty.detectors if d.name == "iforest"), None)

    def score(self, X: pd.DataFrame) -> pd.DataFrame:
        return self._score(X, force=False)

    def _use(self, name: str, n_rows: int, force: bool) -> bool:
        return name in self.limits and (force or n_rows <= self.limits[name])

    def _score(self, X: pd.DataFrame, *, force: bool) -> pd.DataFrame:
        det = self.detector
        n = len(X)
        p_attack: np.ndarray | None = None
        if self._use("supervised", n, force):
            try:
                p_attack = self._p_attack(X)
            except Unsupported:
                self.fallbacks += 1
        if p_attack is None:
            p_attack = supervised.attack_scores(det.supervised_model, X)

        Z = det.novelty_prep.transform(X)
        raw: dict[str, np.ndarray] = {}
        for d in det.novelty.detectors:
            if d.name == "iforest" and self._use("iforest", n, force):
                try:
                    raw[d.name] = self._iforest_novelty(Z)
                    continue
                except Unsupported:
                    self.fallbacks += 1
            raw[d.name] = d.score(Z)
        return det.assemble(X.index, p_attack, raw, det.family_predictor(X))

    def _p_attack(self, X: pd.DataFrame) -> np.ndarray:
        if self._supervised is None:
            raise Unsupported("supervised forest not compiled")
        return self._supervised.p_class1(self.detector.supervised_model[:-1].transform(X))

    def _iforest_novelty(self, Z: np.ndarray) -> np.ndarray:
        if self._iforest is None:
            raise Unsupported("IsolationForest not compiled")
        # Same sign convention as IsolationForestDetector.score: high = novel.
        return -self._iforest.decision_function(Z)


def _timed(fn: Any, repeats: int = 5) -> float:
    fn()
    best = float("inf")
    for _ in range(repeats):
        started = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - started)
    return best


def _crossover(flat_fn: Any, sklearn_fn: Any, X: pd.DataFrame) -> tuple[int, list[dict[str, float]]]:
    """Largest probed batch size up to which the flat path is at least as fast at every probe."""
    limit, timings = 0, []
    for size in PROBES:
        if size > len(X):
            break
        chunk = X.head(size)
        t_flat, t_sk = _timed(lambda c=chunk: flat_fn(c)), _timed(lambda c=chunk: sklearn_fn(c))
        timings.append(
            {"rows": size, "flat_ms": round(1000 * t_flat, 3), "sklearn_ms": round(1000 * t_sk, 3)}
        )
        if t_flat > t_sk:
            break
        limit = size
    return limit, timings


def compile_detector(detector: Any, X_check: pd.DataFrame, *, max_flip_rate: float = 0.0) -> CompiledDetector:
    """Flatten, check parity on `X_check`, then measure where each flat forest beats sklearn.

    `X_check` is also the timing sample, so it must have the columns the detector was fitted on and
    at least a couple of thousand rows; parity on ten rows proves nothing. Parity is checked with the
    flat forests forced on for every row, whatever the measured limits turn out to be.
    """
    started = time.perf_counter()
    compiled = CompiledDetector(detector)
    compile_seconds = time.perf_counter() - started

    reference = detector.score(X_check)
    candidate = compiled._score(X_check, force=True)
    report = parity(reference, candidate)
    if report["decision_flip_rate"] > max_flip_rate:
        raise CompileRefused(
            f"compiled detector changed {report['decisions_changed']} of {len(X_check):,} "
            f"decisions (limit {max_flip_rate:.2%}); keep the sklearn path"
        )

    timings: dict[str, list[dict[str, float]]] = {}
    if "supervised" in compiled.limits:
        compiled.limits["supervised"], timings["supervised"] = _crossover(
            compiled._p_attack, lambda c: supervised.attack_scores(detector.supervised_model, c), X_check
        )
    iforest = compiled._iforest_detector()
    if "iforest" in compiled.limits and iforest is not None:
        compiled.limits["iforest"], timings["iforest"] = _crossover(
            lambda c: compiled._iforest_novelty(detector.novelty_prep.transform(c)),
            lambda c: iforest.score(detector.novelty_prep.transform(c)),
            X_check,
        )
    compiled.parity = report | {
        "compile_seconds": round(compile_seconds, 3),
        "compiled": compiled.compiled,
        "skipped": compiled.skipped,
        "flat_up_to_rows": dict(compiled.limits),
        "timings": timings,
    }
    return compiled


def _family(value: Any) -> str | None:
    return value if isinstance(value, str) else None


def parity(reference: pd.DataFrame, candidate: pd.DataFrame) -> dict[str, Any]:
    """Row-level agreement between two scored frames."""
    fired = reference["fired"].to_numpy() != candidate["fired"].to_numpy()
    family = np.array(
        [_family(a) != _family(b) for a, b in zip(reference["family"], candidate["family"], strict=True)],
        dtype=bool,
    )
    abstain = reference["conformal_abstains"].to_numpy() != candidate["conformal_abstains"].to_numpy()
    changed = fired | family | abstain
    return {
        "n_rows": int(len(reference)),
        "max_abs_diff_p_attack": float(
            np.max(np.abs(reference["p_attack"].to_numpy() - candidate["p_attack"].to_numpy()))
        ),
        "max_abs_diff_novelty": float(
            np.max(
                np.abs(
                    reference["novelty_percentile"].to_numpy() - candidate["novelty_percentile"].to_numpy()
                )
            )
        ),
        "fired_changed": int(fired.sum()),
        "family_changed": int(family.sum()),
        "abstention_changed": int(abstain.sum()),
        "agreement_changed": int(
            (reference["agreement"].to_numpy() != candidate["agreement"].to_numpy()).sum()
        ),
        "decisions_changed": int(changed.sum()),
        "decision_flip_rate": float(changed.mean()) if len(changed) else 0.0,
    }
