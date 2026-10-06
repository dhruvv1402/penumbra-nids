"""E9, member level: the selected ensemble's six members refitted, scored on test, ties shared fairly.

The E9 run on a teammate's machine (commit 50e1eb6) computed member recalls and H9e (Yule's Q) with
row-order tie-breaking, which on UNSW's class-sorted file read single trees as 0.0000. Its checkpointed
scores stayed on that machine. The members are deterministic - same splits, seeds, code and library
versions - so they are refitted here on the same F + H rows, with the configuration and C values the
run selected (read from its report), and test is scored once.

    uv run python scripts/e9_members.py [unsw nslkdd cicids]

Writes artifacts/reports/ensemble_members.json. Nothing here changes a selection or a headline number.
"""

from __future__ import annotations

import json
import sys
import time

import numpy as np

from penumbra.config import settings
from penumbra.eval import ensemble as ens
from penumbra.models import ensemble as E
from penumbra.models import supervised


def load(name: str):
    from penumbra.cli import _load, _load_cicids

    if name == "nslkdd":
        from penumbra.data.loaders import nsl_kdd

        ds, _, fine_test = nsl_kdd.load_with_fine_labels()
        return ds, nsl_kdd.unseen_mask(fine_test).to_numpy()
    if name == "cicids":
        return _load_cicids()[0], None
    return _load("unsw"), None


def members(name: str) -> dict:
    report = json.loads((settings().report_dir / f"ensemble_{name}.json").read_text(encoding="utf-8"))
    sel = report["selection"]["selected"]
    arms = report["stage_a"]["arms"]
    cs = tuple(float(arms[sel["svm_arm"]][s]["C"]) for s in E.SVMS)
    k, solver = report["settings"]["nystroem_k"], report["settings"]["svm_solver"]
    ds, unseen = load(name)
    F, H, _ = ens.splits(ds)
    FH = np.sort(np.concatenate([F, H]))
    prep = supervised.build("rf", ds).named_steps["prep"].fit(ds.X_train.iloc[FH])
    large = len(ds.X_train) > supervised.LARGE_DATASET_ROWS
    dtype = np.float32 if large else np.float64
    X = np.asarray(prep.transform(ds.X_train.iloc[FH]), dtype=dtype)
    Xt = np.asarray(prep.transform(ds.X_test), dtype=dtype)
    y, yt = ds.y_train.iloc[FH].to_numpy(), ds.y_test.to_numpy()
    model = E.PenumbraEnsemble(
        tree_arm=sel["tree_arm"],
        svm_arm=sel["svm_arm"],
        n_components=k,
        svm_solver=solver,
        svm_C=cs,  # type: ignore[arg-type]
        pca_k=report["stage_a"].get("pca_k"),
    )
    start = time.perf_counter()
    transforms, fitted, _ = model._fit_members(X, y)
    seconds = time.perf_counter() - start
    scores = np.concatenate(
        [model._scores(transforms, fitted, Xt[i : i + E.CHUNK]) for i in range(0, len(Xt), E.CHUNK)]
    )
    out: dict = {
        "config": {
            "tree_arm": sel["tree_arm"],
            "svm_arm": sel["svm_arm"],
            "svm_C": cs,
            "nystroem_k": k,
            "solver": solver,
        },
        "fit_seconds_here": seconds,
        "members": {},
    }
    for j, m in enumerate(E.MEMBERS):
        rec, fpr = ens.recall_at(scores[:, j], yt)
        row = {"recall_at_1pct": rec, "realised_fpr": fpr, "roc_auc": ens._auc(yt, scores[:, j])}
        if unseen is not None:
            flags = ens.at_budget(scores[:, j], yt)
            row["unseen_recall_at_1pct"] = float(flags[np.asarray(unseen) & (yt == 1)].mean())
        out["members"][m] = row
    out["diversity"] = ens.diversity(scores, yt)
    q = out["diversity"]["mean_q"]
    out["H9e"] = {
        "tree-svm": q["tree-svm"],
        "tree-tree": q["tree-tree"],
        "held": bool(q["tree-svm"] < q["tree-tree"]),
    }
    return out


def main(names: list[str]) -> None:
    path = settings().report_dir / "ensemble_members.json"
    result = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    result["fingerprint"] = ens.fingerprint(ens.PROFILES["laptop"])
    for name in names:
        print(f"{name}: refitting the selected members...", flush=True)
        result[name] = members(name)
        r = result[name]
        print(
            f"  {name}: H9e tree-svm {r['H9e']['tree-svm']:.3f} vs tree-tree {r['H9e']['tree-tree']:.3f} "
            f"-> {'held' if r['H9e']['held'] else 'refuted'}  ({r['fit_seconds_here']:.0f}s)",
            flush=True,
        )
        print("  members:", {m: round(v["recall_at_1pct"], 4) for m, v in r["members"].items()}, flush=True)
        path.write_text(json.dumps(result, indent=2, default=float), encoding="utf-8")
    print(f"written {path}")


if __name__ == "__main__":
    main(sys.argv[1:] or ["unsw", "nslkdd", "cicids"])
