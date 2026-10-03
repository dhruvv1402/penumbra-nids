"""E9: does a 3-tree + 3-SVM ensemble beat a 300-tree forest? (EXPERIMENTS.md)

Protocol, as registered:

  splits     S   20% of train, stratified by family - the slice the detector calibrates on, so
                 nothing here trains on it either
             H   20% of the rest - the holdout every selection is made on
             F   the remaining 64% - Stage A fits here
  Stage A    every member under every arm, fitted once on F and scored on H; SVM C chosen on H;
             combiners scored on H by cross-validating the (cheap) combiner over H's score matrix.
             The configuration with the best recall at an exact 1% benign budget on H is selected.
  Stage B    the selected configuration and the requested P3 configuration refitted on F + H with
             out-of-fold stacking (by day on CICIDS), next to RF-300, XGBoost and LogReg fitted on
             the same rows. Test is scored once, here.

Every expensive intermediate is checkpointed (`Checkpoint`), so a run that dies resumes, and a run
moved to a bigger machine picks up where the smaller one stopped. Profiles change compute settings
only; the report records which profile and which machine produced it.
"""

from __future__ import annotations

import io
import json
import os
import platform
import subprocess
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from penumbra.data.loaders.base import Dataset
from penumbra.eval.bootstrap import mcnemar, moving_block_bootstrap, stratified_bootstrap
from penumbra.eval.budget import flags_at_benign_budget
from penumbra.features.preprocess import supervised_pipeline
from penumbra.models import ensemble as E
from penumbra.models import supervised
from penumbra.models.detector import CALIBRATION_FRACTION, calibration_split
from penumbra.seeds import SEED

FPR = 0.01
C_GRID = (0.1, 1.0)


@dataclass(frozen=True)
class Profile:
    name: str
    k_small: int  # Nystroem landmarks, UNSW / NSL-KDD
    k_large: int  # Nystroem landmarks, CICIDS
    cv: int
    cv_large: int
    n_boot: int
    n_boot_large: int
    exact_svc_rows: int  # 0 = full data (GPU)
    large_full_grid: bool


PROFILES: dict[str, Profile] = {
    "laptop": Profile("laptop", 1000, 256, 5, 3, 500, 200, 30_000, False),
    "workstation": Profile("workstation", 2000, 1000, 5, 3, 1000, 500, 30_000, True),
    "gpu": Profile("gpu", 2000, 1000, 5, 3, 1000, 500, 0, True),
}


# --- provenance --------------------------------------------------------------------------------


def fingerprint(profile: Profile) -> dict[str, Any]:
    """The machine and code a report came from. Timings are only comparable within one of these."""
    import sklearn

    def run(cmd: list[str]) -> str | None:
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=10, check=False)  # noqa: S603
            return out.stdout.strip() or None
        except (OSError, subprocess.SubprocessError):
            return None

    ram = None
    try:
        import importlib

        ram = round(importlib.import_module("psutil").virtual_memory().total / 2**30, 1)
    except ImportError:
        pass
    return {
        "profile": asdict(profile),
        "os": platform.platform(),
        "cpu": platform.processor() or platform.machine(),
        "cores": os.cpu_count(),
        "ram_gb": ram,
        "gpu": run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"]),
        "git_sha": run(["git", "rev-parse", "HEAD"]),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "sklearn": sklearn.__version__,
    }


class Checkpoint:
    """Arrays and small JSON blobs keyed by name, one file each. `None` from `get` means recompute."""

    def __init__(self, root: Path | None, resume: bool = True) -> None:
        self.root = root
        self.resume = resume
        if root is not None:
            root.mkdir(parents=True, exist_ok=True)

    def get(self, key: str) -> dict[str, Any] | None:
        if self.root is None or not self.resume:
            return None
        arr, meta = self.root / f"{key}.npz", self.root / f"{key}.json"
        if not (arr.exists() and meta.exists()):
            return None
        with np.load(arr) as z:
            out: dict[str, Any] = {k: z[k] for k in z.files}
        out.update(json.loads(meta.read_text(encoding="utf-8")))
        return out

    def put(self, key: str, arrays: dict[str, Any], meta: dict[str, Any]) -> None:
        if self.root is None:
            return
        np.savez_compressed(self.root / f"{key}.npz", **arrays)
        (self.root / f"{key}.json").write_text(json.dumps(meta, default=float), encoding="utf-8")


# --- metrics -----------------------------------------------------------------------------------


def at_budget(scores: np.ndarray, y: np.ndarray, fpr: float = FPR) -> np.ndarray:
    y = np.asarray(y).astype(int)
    return flags_at_benign_budget(scores, y, int(round(fpr * int((y == 0).sum()))))


def recall_at(scores: np.ndarray, y: np.ndarray, fpr: float = FPR) -> tuple[float, float]:
    """(recall, realised FPR) at an exact benign budget. Realised is printed so the match is checkable."""
    y = np.asarray(y).astype(int)
    flags = at_budget(scores, y, fpr)
    return float(flags[y == 1].mean()), float(flags[y == 0].mean())


def _auc(y: np.ndarray, s: np.ndarray) -> float:
    try:
        return float(roc_auc_score(y, s))
    except ValueError:
        return float("nan")


def yules_q(a: np.ndarray, b: np.ndarray) -> float:
    """Q on correctness indicators: 1 = the two always err together, 0 = independent, <0 = opposite."""
    n11 = float(np.sum(a & b))
    n00 = float(np.sum(~a & ~b))
    n10 = float(np.sum(a & ~b))
    n01 = float(np.sum(~a & b))
    denom = n11 * n00 + n01 * n10
    return float((n11 * n00 - n01 * n10) / denom) if denom else float("nan")


def diversity(member_scores: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    """Pairwise disagreement and Yule's Q, each member at its own matched 1% budget."""
    y = np.asarray(y).astype(int)
    flags = np.column_stack([at_budget(member_scores[:, j], y) for j in range(member_scores.shape[1])])
    correct = flags == (y[:, None] == 1)
    pairs: dict[str, dict[str, float]] = {}
    groups: dict[str, list[float]] = {"tree-tree": [], "tree-svm": [], "svm-svm": []}
    for i in range(len(E.MEMBERS)):
        for j in range(i + 1, len(E.MEMBERS)):
            a, b = E.MEMBERS[i], E.MEMBERS[j]
            q = yules_q(correct[:, i], correct[:, j])
            pairs[f"{a}-{b}"] = {"q": q, "disagreement": float((flags[:, i] != flags[:, j]).mean())}
            kind = "-".join(sorted(("tree" if m.startswith("T") else "svm") for m in (a, b)))
            groups[kind.replace("svm-tree", "tree-svm")].append(q)
    return {
        "pairs": pairs,
        "mean_q": {k: float(np.nanmean(v)) if v else float("nan") for k, v in groups.items()},
    }


# --- splits ------------------------------------------------------------------------------------


def splits(ds: Dataset) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Positions of (F, H, S) in ds.X_train. S is exactly the detector's calibration slice."""
    fit_pos, s_pos = calibration_split(ds.fam_train, CALIBRATION_FRACTION)
    rest = ds.fam_train.iloc[fit_pos].reset_index(drop=True)
    f_rel, h_rel = calibration_split(rest, 0.2, seed=SEED + 1)
    return fit_pos[f_rel], fit_pos[h_rel], s_pos


# --- Stage A -----------------------------------------------------------------------------------


def stage_a(
    XF: np.ndarray,
    yF: np.ndarray,
    XH: np.ndarray,
    yH: np.ndarray,
    *,
    arms: tuple[str, ...],
    k: int,
    solver: str,
    feature_names: list[str],
    ck: Checkpoint,
    step: Callable[[str], None],
) -> dict[str, Any]:
    """Every member under every arm, fitted on F, scored on H."""
    out: dict[str, Any] = {"arms": {}, "scores": {}}
    pca_k: int | None = None
    for arm in arms:
        info: dict[str, Any] = {}
        cached = {m: ck.get(f"A_{arm}_{m}") for m in E.TREES} | {
            f"{s}_C{c}": ck.get(f"A_{arm}_{s}_C{c}") for s in E.SVMS for c in C_GRID
        }
        need_fit = any(v is None for v in cached.values())
        t = E.arm_transform(arm, pca_k=pca_k) if arm != "P3b" or pca_k else None
        if arm == "P3b" and pca_k is None:
            step("  P3b skipped: P2 did not run")
            continue
        ZF = ZH = None
        if need_fit or ck.get(f"A_{arm}_transform") is None:
            step(f"  arm {arm}: transform")
            ZF = XF if t is None else t.fit_transform(XF)
            ZH = XH if t is None else t.transform(XH)
            meta: dict[str, Any] = {"pca_components": E.pca_components(t)}
            if t is not None and "pca" in getattr(t, "named_steps", {}):
                pca = t.named_steps["pca"]
                meta["explained_variance_ratio"] = [float(v) for v in pca.explained_variance_ratio_[:10]]
                top = np.argsort(-np.abs(pca.components_[0]))[:5]
                meta["first_component_top_loadings"] = [
                    {"feature": feature_names[i], "loading": float(pca.components_[0][i])} for i in top
                ]
            ck.put(f"A_{arm}_transform", {}, meta)
        tmeta = ck.get(f"A_{arm}_transform") or {}
        info["transform"] = {k_: v for k_, v in tmeta.items()}
        if arm == "P2" and tmeta.get("pca_components"):
            pca_k = int(tmeta["pca_components"])

        def fitted(key: str, make: Callable[[], Any], Z_fit: Any, Z_h: Any) -> dict[str, Any]:
            got = ck.get(key)
            if got is not None:
                return got
            model = make()
            rec = E.fit_member(model, Z_fit, yF)
            s = E.member_score(model, Z_h)
            res = {"seconds": rec.seconds, "convergence_warnings": rec.convergence_warnings}
            ck.put(key, {"scores": s}, res)
            return {"scores": s, **res}

        max_iter = 200 if arm == "P0" else 1000
        for m in E.TREES:
            step(f"  arm {arm}: {m}")
            r = fitted(f"A_{arm}_{m}", partial(E.member, m), ZF, ZH)
            out["scores"][f"{arm}/{m}"] = r["scores"]
            rec, fpr = recall_at(r["scores"], yH)
            info[m] = {
                "recall": rec,
                "realised_fpr": fpr,
                "auc": _auc(yH, r["scores"]),
                "seconds": r["seconds"],
            }
        for s in E.SVMS:
            best = None
            for c in C_GRID:
                step(f"  arm {arm}: {s} C={c}")
                r = fitted(
                    f"A_{arm}_{s}_C{c}",
                    partial(
                        E.member, s, C=c, n_components=k, solver=solver, n_rows=len(XF), max_iter=max_iter
                    ),
                    ZF,
                    ZH,
                )
                rec, fpr = recall_at(r["scores"], yH)
                row = {
                    "C": c,
                    "recall": rec,
                    "realised_fpr": fpr,
                    "auc": _auc(yH, r["scores"]),
                    "seconds": r["seconds"],
                    "convergence_warnings": int(r["convergence_warnings"]),
                }
                info.setdefault(f"{s}_grid", []).append(row)
                if best is None or (rec, row["auc"]) > (best["recall"], best["auc"]):
                    best = row
                    out["scores"][f"{arm}/{s}"] = r["scores"]
            info[s] = best
        out["arms"][arm] = info
        del ZF, ZH
    out["pca_k"] = pca_k
    return out


def combine_on_holdout(matrix: np.ndarray, y: np.ndarray, kind: str) -> np.ndarray:
    """Combiner scores on H without fitting the combiner on the rows it scores (5-fold over H)."""
    if kind == "hard":
        return E.Combiner("hard", E.MEMBERS).proba(matrix)
    out = np.zeros(len(y))
    for tr, va in StratifiedKFold(5, shuffle=True, random_state=SEED).split(matrix, y):
        out[va] = E.Combiner(kind, E.MEMBERS).fit(matrix[tr], y[tr]).proba(matrix[va])
    return out


def select(a: dict[str, Any], yH: np.ndarray) -> dict[str, Any]:
    """Score every (tree arm, svm arm) x combiner on H; pick the best by recall at 1% FPR."""
    arms = list(a["arms"])
    configs: list[tuple[str, str, str]] = [(arm, arm, arm) for arm in arms]
    scaled = [x for x in ("P1", "P2") if x in arms]
    if "P0" in arms and scaled:

        def mean_svm(arm: str) -> float:
            return float(np.mean([a["arms"][arm][s]["recall"] for s in E.SVMS]))

        configs.append(("hetero", "P0", max(scaled, key=mean_svm)))
    rows = []
    for label, tree_arm, svm_arm in configs:
        matrix = np.column_stack(
            [a["scores"][f"{tree_arm}/{m}"] for m in E.TREES]
            + [a["scores"][f"{svm_arm}/{s}"] for s in E.SVMS]
        )
        seconds = sum(a["arms"][tree_arm][m]["seconds"] for m in E.TREES) + sum(
            a["arms"][svm_arm][s]["seconds"] for s in E.SVMS
        )
        for kind in E.COMBINERS:
            p = combine_on_holdout(matrix, yH, kind)
            rec, fpr = recall_at(p, yH)
            row: dict[str, Any] = {
                "config": label,
                "tree_arm": tree_arm,
                "svm_arm": svm_arm,
                "combiner": kind,
                "recall": rec,
                "realised_fpr": fpr,
                "auc": _auc(yH, p),
                "member_seconds": seconds,
            }
            if kind == "hard":
                votes = np.round(p * len(E.MEMBERS)).astype(int)
                benign = np.asarray(yH) == 0
                row["vote_levels_fpr"] = {int(v): float((votes[benign] >= v).mean()) for v in range(1, 7)}
            rows.append(row)
    best = max(
        (r for r in rows if r["combiner"] != "hard"), key=lambda r: (r["recall"], -r["member_seconds"])
    )
    hard = [r for r in rows if r["combiner"] == "hard"]
    best_any = max(rows, key=lambda r: (r["recall"], -r["member_seconds"]))
    # Hard voting is scored and reported, but cannot be selected: it has no operating point near
    # 1% to offer an analyst (prediction 9). If it would have won, the report says so.
    return {
        "rows": rows,
        "selected": best,
        "hard_vote_would_have_won": best_any["combiner"] == "hard",
        "hard": hard,
    }


# --- checks --------------------------------------------------------------------------------------


def exact_svc_check(
    XF: np.ndarray, yF: np.ndarray, XH: np.ndarray, yH: np.ndarray, *, rows: int, k: int, ck: Checkpoint
) -> dict[str, Any]:
    """Exact SVC(rbf) vs its Nystroem twin, on the same rows, scored on H (prediction 7, H9i)."""
    got = ck.get("check_exact_svc")
    if got is not None:
        return {k_: v for k_, v in got.items() if not isinstance(v, np.ndarray)}
    t = E.arm_transform("P1")
    assert t is not None
    rng = np.random.default_rng(SEED)
    if rows and rows < len(XF):
        take = np.concatenate(
            [
                rng.choice(
                    np.flatnonzero(yF == c), size=max(1, int(round(rows * np.mean(yF == c)))), replace=False
                )
                for c in (0, 1)
            ]
        )
    else:
        take = np.arange(len(XF))
    Z = t.fit_transform(XF[take])
    ZH = t.transform(XH)
    out: dict[str, Any] = {"rows": int(len(take))}
    engine = "sklearn"
    if not rows:
        try:
            from cuml.svm import SVC as cuSVC  # type: ignore[import-not-found]

            exact: Any = cuSVC(kernel="rbf", gamma="scale", C=1.0, class_weight="balanced")
            engine = "cuml"
        except ImportError:
            out["note"] = "full-data exact SVC needs cuML; not installed, fell back to a 30,000-row subsample"
            sub = rng.choice(len(take), size=min(30_000, len(take)), replace=False)
            take, Z = take[sub], Z[sub]
            out["rows"] = int(len(take))
    if engine == "sklearn":
        from sklearn.svm import SVC

        exact = SVC(kernel="rbf", gamma="scale", C=1.0, class_weight="balanced", cache_size=2000)
    start = time.perf_counter()
    exact.fit(Z, yF[take])
    exact_s = np.asarray(exact.decision_function(ZH), dtype=float)
    out["exact"] = {
        "engine": engine,
        "seconds": time.perf_counter() - start,
        "n_support": int(np.sum(getattr(exact, "n_support_", [0]))),
    }
    twin = E.member("S2", C=1.0, n_components=k, n_rows=len(take))
    start = time.perf_counter()
    twin.fit(Z, yF[take])
    twin_s = E.member_score(twin, ZH)
    out["nystroem"] = {"seconds": time.perf_counter() - start, "n_components": k}
    for name, s in (("exact", exact_s), ("nystroem", twin_s)):
        rec, fpr = recall_at(s, yH)
        out[name].update(recall=rec, realised_fpr=fpr, auc=_auc(yH, s))
    out["recall_delta"] = out["nystroem"]["recall"] - out["exact"]["recall"]
    ck.put("check_exact_svc", {}, out)
    return out


def sgd_check(
    XF: np.ndarray,
    yF: np.ndarray,
    XH: np.ndarray,
    yH: np.ndarray,
    *,
    k: int,
    a: dict[str, Any],
    ck: Checkpoint,
) -> dict[str, Any]:
    """SGD vs liblinear on the same objective, P1, the C each member chose (prediction 8)."""
    got = ck.get("check_sgd")
    if got is not None:
        return {k_: v for k_, v in got.items() if not isinstance(v, np.ndarray)}
    if "P1" not in a["arms"]:
        return {"skipped": "P1 did not run"}
    t = E.arm_transform("P1")
    assert t is not None
    Z, ZH = t.fit_transform(XF), t.transform(XH)
    out: dict[str, Any] = {}
    for s in E.SVMS:
        c = a["arms"]["P1"][s]["C"]
        model = E.member(s, C=c, n_components=k, solver="sgd", n_rows=len(Z))
        model.fit(Z, yF)
        auc_sgd = _auc(yH, E.member_score(model, ZH))
        out[s] = {"C": c, "auc_liblinear": a["arms"]["P1"][s]["auc"], "auc_sgd": auc_sgd}
        out[s]["delta"] = auc_sgd - out[s]["auc_liblinear"]
    ck.put("check_sgd", {}, out)
    return out


# --- Stage B and test ----------------------------------------------------------------------------


def _latency(model: Any, X: pd.DataFrame) -> dict[str, float]:
    one = X.iloc[[0]]
    batch = X.iloc[: min(2048, len(X))]
    t1 = []
    for _ in range(20):
        start = time.perf_counter()
        model.predict_proba(one)
        t1.append(time.perf_counter() - start)
    tb = []
    for _ in range(3):
        start = time.perf_counter()
        model.predict_proba(batch)
        tb.append(time.perf_counter() - start)
    return {
        "ms_batch_1": 1000 * float(np.median(t1)),
        "ms_per_flow_batch_2048": 1000 * float(np.median(tb)) / len(batch),
    }


def _size_mb(model: Any) -> float:
    buf = io.BytesIO()
    joblib.dump(model, buf, compress=3)
    return buf.tell() / 2**20


def _scores_chunked(model: Any, X: pd.DataFrame) -> np.ndarray:
    return np.concatenate(
        [supervised.attack_scores(model, X.iloc[i : i + E.CHUNK]) for i in range(0, len(X), E.CHUNK)]
    )


def test_metrics(
    scores: dict[str, np.ndarray],
    y: np.ndarray,
    fam: np.ndarray,
    *,
    unseen: np.ndarray | None,
    n_boot: int,
    temporal: bool,
) -> dict[str, Any]:
    y = np.asarray(y).astype(int)
    boot = moving_block_bootstrap if temporal else stratified_bootstrap
    out: dict[str, Any] = {"prevalence": float(y.mean()), "benign_rows": int((y == 0).sum()), "models": {}}
    rf = scores.get("rf")
    rf_flags = at_budget(rf, y) if rf is not None else None
    for name, s in scores.items():
        flags = at_budget(s, y)
        rec, fpr = float(flags[y == 1].mean()), float(flags[y == 0].mean())
        auc_ci = boot(y, s, _auc, n_resamples=n_boot)
        row: dict[str, Any] = {
            "recall_at_1pct": rec,
            "realised_fpr": fpr,
            "roc_auc": {"point": auc_ci.point, "ci95": [auc_ci.lower, auc_ci.upper]},
            "per_family_recall": {
                str(f): float(flags[(fam == f) & (y == 1)].mean()) for f in sorted(set(fam[y == 1].tolist()))
            },
        }
        if unseen is not None:
            row["unseen_recall_at_1pct"] = float(flags[np.asarray(unseen) & (y == 1)].mean())
        if rf is not None and rf_flags is not None and name != "rf":

            def diff(yy: np.ndarray, ss: np.ndarray) -> float:
                return recall_at(ss[:, 0], yy)[0] - recall_at(ss[:, 1], yy)[0]

            d = boot(y, np.column_stack([s, rf]), diff, n_resamples=n_boot)
            m = mcnemar(y, flags.astype(int), rf_flags.astype(int))
            row["vs_rf"] = {
                "recall_delta": d.point,
                "ci95": [d.lower, d.upper],
                "mcnemar": {"n01": m.n01, "n10": m.n10, "p": m.p_value, "odds_ratio": m.odds_ratio},
            }
        out["models"][name] = row
    return out


def run(
    ds: Dataset,
    *,
    profile: Profile,
    checkpoint_dir: Path | None,
    resume: bool = True,
    groups: np.ndarray | None = None,
    unseen: np.ndarray | None = None,
    inherit: dict[str, Any] | None = None,
    on_progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """The whole E9 protocol on one dataset. `groups` = day of each training row (CICIDS)."""

    def step(msg: str) -> None:
        if on_progress:
            on_progress(msg)

    ck = Checkpoint(checkpoint_dir, resume)
    large = len(ds.X_train) > supervised.LARGE_DATASET_ROWS
    k = profile.k_large if large else profile.k_small
    solver = "sgd" if large else "liblinear"
    report: dict[str, Any] = {
        "experiment": "E9",
        "dataset": ds.name,
        "fingerprint": fingerprint(profile),
        "settings": {"nystroem_k": k, "svm_solver": solver, "c_grid": list(C_GRID), "fpr": FPR},
    }

    F, H, S = splits(ds)
    report["splits"] = {"F": int(len(F)), "H": int(len(H)), "S": int(len(S)), "test": int(len(ds.X_test))}
    prep = supervised_pipeline(ds, scale=False).fit(ds.X_train.iloc[F])
    names = [str(n) for n in prep.get_feature_names_out()]
    dtype = np.float32 if large else np.float64
    XF = np.asarray(prep.transform(ds.X_train.iloc[F]), dtype=dtype)
    XH = np.asarray(prep.transform(ds.X_train.iloc[H]), dtype=dtype)
    yF, yH = ds.y_train.iloc[F].to_numpy(), ds.y_train.iloc[H].to_numpy()

    if large and not profile.large_full_grid and inherit:
        sel = inherit["selected"]
        arms = tuple(dict.fromkeys(a for a in (sel["tree_arm"], sel["svm_arm"]) if a in E.ARMS))
        report["grid"] = f"inherited from {inherit.get('dataset')}: {list(arms)}"
    else:
        arms = E.ARMS
        report["grid"] = "full"
    step(f"Stage A on {len(F):,} rows, holdout {len(H):,}, arms {list(arms)}")
    a = stage_a(XF, yF, XH, yH, arms=arms, k=k, solver=solver, feature_names=names, ck=ck, step=step)
    report["stage_a"] = {"arms": a["arms"], "pca_k": a["pca_k"]}
    sel = select(a, yH)
    if large and inherit and not profile.large_full_grid:
        inherited = inherit["selected"]
        sel["selected"] = next(
            r
            for r in sel["rows"]
            if r["tree_arm"] == inherited["tree_arm"]
            and r["svm_arm"] == inherited["svm_arm"]
            and r["combiner"] == inherited["combiner"]
        )
    report["selection"] = sel
    chosen = sel["selected"]
    step(f"selected {chosen['config']} / {chosen['combiner']}: holdout recall {chosen['recall']:.4f}")

    if not large:
        step("exact SVC vs Nystroem")
        report["check_exact_svc"] = exact_svc_check(XF, yF, XH, yH, rows=profile.exact_svc_rows, k=k, ck=ck)
        step("SGD vs liblinear")
        report["check_sgd"] = sgd_check(XF, yF, XH, yH, k=k, a=a, ck=ck)
    del XF, XH

    # Stage B: F + H, out-of-fold stacking, then test once.
    FH = np.sort(np.concatenate([F, H]))
    X_fit, y_fit = ds.X_train.iloc[FH], ds.y_train.iloc[FH]
    g_fit = groups[FH] if groups is not None else None
    cv = profile.cv_large if large else profile.cv

    def ensemble_pipeline(tree_arm: str, svm_arm: str, combiner: str) -> Any:
        model = supervised.build("rf", ds)  # for the prep step and the Pipeline shape
        cs = tuple(float(a["arms"][svm_arm][s]["C"]) for s in E.SVMS)
        model.steps[-1] = (
            "clf",
            E.PenumbraEnsemble(
                tree_arm=tree_arm,
                svm_arm=svm_arm,
                combiner=combiner,
                n_components=k,
                svm_solver=solver,
                svm_C=cs,  # type: ignore[arg-type]
                cv=cv,
                pca_k=a["pca_k"],
            ),
        )
        return model

    candidates: dict[str, Callable[[], Any]] = {
        "ensemble": lambda: ensemble_pipeline(chosen["tree_arm"], chosen["svm_arm"], chosen["combiner"]),
        "rf": lambda: supervised.build("rf", ds),
        "xgb": lambda: supervised.build("xgb", ds),
        "logreg": lambda: supervised.build("logreg", ds),
    }
    if "P3a" in a["arms"] and (not large or profile.large_full_grid):
        candidates["ensemble_P3a"] = lambda: ensemble_pipeline("P3a", "P3a", chosen["combiner"])

    scores: dict[str, np.ndarray] = {}
    stage_b: dict[str, Any] = {}
    member_test: np.ndarray | None = None
    for name, make in candidates.items():
        got = ck.get(f"B_{name}")
        if got is None:
            step(f"Stage B: {name} on {len(FH):,} rows")
            model = make()
            start = time.perf_counter()
            if name.startswith("ensemble") and g_fit is not None:
                model.fit(X_fit, y_fit, clf__groups=g_fit)
            else:
                model.fit(X_fit, y_fit)
            seconds = time.perf_counter() - start
            s = _scores_chunked(model, ds.X_test)
            meta: dict[str, Any] = {
                "fit_seconds": seconds,
                "size_mb": _size_mb(model),
                **_latency(model, ds.X_test),
            }
            arrays = {"scores": s}
            if name.startswith("ensemble"):
                clf = model.named_steps["clf"]
                Xp = model.named_steps["prep"].transform(ds.X_test)
                arrays["members"] = np.concatenate(
                    [clf.member_scores(Xp[i : i + E.CHUNK]) for i in range(0, len(Xp), E.CHUNK)]
                )
                meta.update(
                    member_seconds=clf.fit_seconds_,
                    convergence_warnings=clf.convergence_warnings_,
                    pca_components=clf.pca_components_,
                )
            ck.put(f"B_{name}", arrays, meta)
            got = {**arrays, **meta}
        scores[name] = np.asarray(got["scores"])
        stage_b[name] = {k_: v for k_, v in got.items() if not isinstance(v, np.ndarray)}
        if name == "ensemble":
            member_test = np.asarray(got["members"])
    report["stage_b"] = stage_b

    y_test = ds.y_test.to_numpy()
    fam = ds.fam_test.astype(str).to_numpy()
    n_boot = profile.n_boot_large if large else profile.n_boot
    step("test metrics")
    report["test"] = test_metrics(scores, y_test, fam, unseen=unseen, n_boot=n_boot, temporal=large)
    if member_test is not None:
        members: dict[str, Any] = {}
        for j, m in enumerate(E.MEMBERS):
            flags = at_budget(member_test[:, j], y_test)
            row = {
                "recall_at_1pct": float(flags[y_test == 1].mean()),
                "realised_fpr": float(flags[y_test == 0].mean()),
            }
            if unseen is not None:
                row["unseen_recall_at_1pct"] = float(flags[np.asarray(unseen) & (y_test == 1)].mean())
            members[m] = row
        report["test"]["members"] = members
        report["test"]["diversity"] = diversity(member_test, y_test)
    report["summary"] = summarise(report)
    return report


# --- the registered predictions ------------------------------------------------------------------


def summarise(r: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Each prediction from EXPERIMENTS.md E9 that this dataset can speak to. Not edited after."""
    out: dict[str, dict[str, Any]] = {}
    arms = r.get("stage_a", {}).get("arms", {})
    models = r.get("test", {}).get("models", {})
    if "ensemble" in models and "vs_rf" in models["ensemble"]:
        hi = models["ensemble"]["vs_rf"]["ci95"][1]
        out["H9a ensemble - RF upper CI <= +0.005"] = {
            "value": models["ensemble"]["vs_rf"],
            "held": bool(hi <= 0.005),
        }
    if "P0" in arms and "P1" in arms:
        gaps = {s: arms["P1"][s]["recall"] - arms["P0"][s]["recall"] for s in E.SVMS}
        out["H9b each SVM >= 0.10 worse under P0 than P1"] = {
            "value": gaps,
            "held": all(g >= 0.10 for g in gaps.values()),
        }
    if {"P0", "P1", "P2"} <= set(arms):
        tree_drop = float(np.mean([arms["P0"][t]["recall"] - arms["P2"][t]["recall"] for t in E.TREES]))
        svm_moves = {s: arms["P2"][s]["recall"] - arms["P1"][s]["recall"] for s in E.SVMS}
        out["H9c trees lose >= 0.02 under P2; SVMs within 0.01 of P1"] = {
            "value": {"tree_mean_drop": tree_drop, "svm_moves": svm_moves},
            "held": tree_drop >= 0.02 and all(abs(v) <= 0.01 for v in svm_moves.values()),
        }
    if "P3a" in arms:
        n = arms["P3a"]["transform"].get("pca_components")
        out["H9c P3a keeps <= 3 components"] = {
            "value": {
                "components": n,
                "top_loadings": arms["P3a"]["transform"].get("first_component_top_loadings"),
            },
            "held": n is not None and n <= 3,
        }
    members = r.get("test", {}).get("members", {})
    if members and "unseen_recall_at_1pct" in members.get("S2", {}):
        v = members["S2"]["unseen_recall_at_1pct"]
        out["H9d S2 unseen-17 recall at 1% > 0.053"] = {"value": v, "held": bool(v > 0.053)}
    div = r.get("test", {}).get("diversity")
    if div:
        q = div["mean_q"]
        out["H9e mean Q tree-svm < tree-tree"] = {"value": q, "held": bool(q["tree-svm"] < q["tree-tree"])}
    ex = r.get("check_exact_svc")
    if ex and "recall_delta" in ex:
        out["P7 exact vs Nystroem within 0.01 recall"] = {
            "value": ex["recall_delta"],
            "held": abs(ex["recall_delta"]) <= 0.01,
        }
    sg = r.get("check_sgd")
    if sg and "S1" in sg:
        deltas = {s: sg[s]["delta"] for s in E.SVMS}
        out["P8 SGD within 0.005 AUC of liblinear"] = {
            "value": deltas,
            "held": all(abs(d) <= 0.005 for d in deltas.values()),
        }
    hard = r.get("selection", {}).get("hard")
    if hard:
        levels = [lvl for h in hard for lvl in h["vote_levels_fpr"].values()]
        out["P9 no hard-vote level realises 0.75-1.25% FPR"] = {
            "value": [h["vote_levels_fpr"] for h in hard],
            "held": not any(0.0075 <= v <= 0.0125 for v in levels),
        }
    return out
