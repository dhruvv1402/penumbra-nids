"""Three decision trees and three SVMs, combined (E9, EXPERIMENTS.md).

## Why these six

A tree draws axis-aligned boxes; an SVM draws a smooth margin. They make different mistakes, and
an ensemble is only worth its members' disagreement. So the members differ by model family first
and by settings second:

  T1  tree, gini, unlimited depth, min_samples_leaf=5, balanced classes
  T2  tree, entropy, max_depth=20, max_features="sqrt"
  T3  tree, gini, max_depth=12, min_samples_leaf=20
  S1  linear SVM
  S2  RBF SVM      Nystroem(rbf)  -> linear SVM
  S3  poly SVM     Nystroem(poly, degree 2) -> linear SVM

## The SVMs are kernel approximations

An exact kernel SVC needs O(n^2) memory and O(n^2..n^3) time: on 140,000 to 1.2 million flows and
16 GB that is not a slow fit, it is no fit. Nystroem maps the data through k landmark points into a
space where a LINEAR SVM approximates the kernel one, at O(n k) cost. E9 measures what that costs
against an exact SVC on a subsample (and on a GPU machine, on the full data). Where liblinear's
float64 copy of an n x k matrix will not fit (CICIDS), the same hinge objective is minimised by SGD
(`svm_solver="sgd"`, alpha = 1 / (C n)).

## Preprocessing arms

Each member family gets its own transform, chosen by arm, fitted inside the training fold:

  P0   none (one-hot + imputation only, from the detector's `prep` step)
  P1   signed log  sign(x) log1p(|x|), then z-score      - "normalise"
  P2   P1, then PCA keeping 95% of the variance          - "normalise -> PCA"
  P3a  PCA on unscaled features (95%), then z-score      - "PCA -> normalise", as literally asked
  P3b  as P3a with P2's number of components

Trees are invariant to the per-feature monotone transforms in P1, and PCA rotates away the axes
trees split on; running them on every arm is the control that shows it.

## Combiners

All three read one out-of-fold score matrix, so no combiner is fitted on scores a member produced
for rows it trained on:

  stack   logistic regression on the members' out-of-fold scores
  soft    each member Platt-scaled on its out-of-fold scores, probabilities averaged
  hard    the fraction of members voting attack (tree p >= 0.5, SVM margin >= 0): seven levels
"""

from __future__ import annotations

import time
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, clone
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from penumbra.seeds import SEED

ARMS = ("P0", "P1", "P2", "P3a", "P3b")
COMBINERS = ("stack", "soft", "hard")
TREES = ("T1", "T2", "T3")
SVMS = ("S1", "S2", "S3")
MEMBERS = TREES + SVMS
PCA_VARIANCE = 0.95
# Rows per chunk when mapping through Nystroem: an n x k float32 block is n * k * 4 bytes.
CHUNK = 100_000

# What `supervised.build("ens")` ships. Set from E9's selection rule (holdout recall at 1% FPR),
# never from test; EXPERIMENTS.md E9 RESULT records the selection that produced it.
PRODUCT_CONFIG: dict[str, Any] = {"tree_arm": "P0", "svm_arm": "P1", "combiner": "stack"}


def signed_log(X: np.ndarray) -> np.ndarray:
    """sign(x) log1p(|x|). Byte and packet counts span seven orders of magnitude; z-scoring them raw
    leaves one flow in a million defining the scale. Module-level so the transform pickles."""
    X = np.asarray(X, dtype=np.float64)
    return np.sign(X) * np.log1p(np.abs(X))


def arm_transform(arm: str, *, pca_k: int | None = None) -> TransformerMixin | None:
    """The transform an arm applies before a member sees the data. None means identity."""
    log = ("log", FunctionTransformer(signed_log))
    if arm == "P0":
        return None
    if arm == "P1":
        return Pipeline([log, ("z", StandardScaler())])
    if arm == "P2":
        return Pipeline(
            [
                log,
                ("z", StandardScaler()),
                ("pca", PCA(n_components=PCA_VARIANCE, svd_solver="full", random_state=SEED)),
            ]
        )
    if arm == "P3a":
        return Pipeline(
            [
                ("pca", PCA(n_components=PCA_VARIANCE, svd_solver="full", random_state=SEED)),
                ("z", StandardScaler()),
            ]
        )
    if arm == "P3b":
        if pca_k is None:
            raise ValueError("P3b needs pca_k: P2's number of components")
        return Pipeline(
            [
                ("pca", PCA(n_components=int(pca_k), svd_solver="full", random_state=SEED)),
                ("z", StandardScaler()),
            ]
        )
    raise ValueError(f"unknown arm {arm!r}; choose from {ARMS}")


def pca_components(transform: Any) -> int | None:
    if isinstance(transform, Pipeline) and "pca" in transform.named_steps:
        return int(transform.named_steps["pca"].n_components_)
    return None


class ScaleRuleNystroem(TransformerMixin, BaseEstimator):
    """Nystroem with gamma set by sklearn's SVC "scale" rule, 1 / (d Var X), at fit time.

    The same rule `SVC(gamma="scale")` uses, so the exact-SVC check compares like with like. Output
    is float32 and computed in chunks: the map is the memory peak of the whole ensemble.
    """

    def __init__(
        self,
        kernel: str = "rbf",
        n_components: int = 1000,
        degree: int = 2,
        coef0: float = 1.0,
        random_state: int = SEED,
    ) -> None:
        self.kernel = kernel
        self.n_components = n_components
        self.degree = degree
        self.coef0 = coef0
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: Any = None) -> ScaleRuleNystroem:
        X = np.asarray(X)
        var = float(X.var())
        self.gamma_ = 1.0 / (X.shape[1] * var) if var > 0 else 1.0
        params: dict[str, Any] = {"gamma": self.gamma_}
        if self.kernel == "poly":
            params.update(degree=self.degree, coef0=self.coef0)
        self.nystroem_ = Nystroem(
            kernel=self.kernel,
            n_components=min(self.n_components, len(X)),
            random_state=self.random_state,
            kernel_params=None,
            **params,
        ).fit(X)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        out = np.empty((len(X), self.nystroem_.n_components), dtype=np.float32)
        for start in range(0, len(X), CHUNK):
            out[start : start + CHUNK] = self.nystroem_.transform(X[start : start + CHUNK])
        return out


def svm_classifier(C: float, *, solver: str, n_rows: int, max_iter: int) -> Any:
    if solver == "liblinear":
        return LinearSVC(C=C, class_weight="balanced", dual="auto", max_iter=max_iter, random_state=SEED)
    if solver == "sgd":
        # The same hinge objective: LinearSVC minimises 0.5|w|^2 + C sum(hinge); SGD minimises
        # mean(hinge) + alpha/2 |w|^2, so alpha = 1 / (C n).
        return SGDClassifier(
            loss="hinge",
            alpha=1.0 / (C * max(n_rows, 1)),
            average=True,
            class_weight="balanced",
            max_iter=max(max_iter // 50, 20),
            tol=1e-4,
            random_state=SEED,
        )
    raise ValueError(f"unknown svm solver {solver!r}")


def member(
    name: str,
    *,
    C: float = 1.0,
    n_components: int = 1000,
    solver: str = "liblinear",
    n_rows: int = 0,
    max_iter: int = 1000,
) -> Any:
    """An unfitted member. SVM members carry their own kernel map; the arm transform is shared."""
    if name == "T1":
        return DecisionTreeClassifier(
            criterion="gini", min_samples_leaf=5, class_weight="balanced", random_state=SEED
        )
    if name == "T2":
        return DecisionTreeClassifier(
            criterion="entropy", max_depth=20, max_features="sqrt", class_weight="balanced", random_state=SEED
        )
    if name == "T3":
        return DecisionTreeClassifier(
            criterion="gini", max_depth=12, min_samples_leaf=20, class_weight="balanced", random_state=SEED
        )
    clf = svm_classifier(C, solver=solver, n_rows=n_rows, max_iter=max_iter)
    if name == "S1":
        return clf
    if name == "S2":
        return Pipeline([("map", ScaleRuleNystroem("rbf", n_components)), ("svm", clf)])
    if name == "S3":
        return Pipeline([("map", ScaleRuleNystroem("poly", n_components, degree=2, coef0=1.0)), ("svm", clf)])
    raise ValueError(f"unknown member {name!r}; choose from {MEMBERS}")


def member_score(model: Any, Z: np.ndarray) -> np.ndarray:
    """A tree's P(attack), an SVM's signed margin."""
    if hasattr(model, "predict_proba") and not hasattr(model, "decision_function"):
        return np.asarray(model.predict_proba(Z)[:, 1], dtype=float)
    out = np.empty(len(Z), dtype=float)
    for start in range(0, len(Z), CHUNK):
        out[start : start + CHUNK] = model.decision_function(Z[start : start + CHUNK])
    return out


def member_votes(scores: np.ndarray, names: Sequence[str]) -> np.ndarray:
    cut = np.array([0.5 if n.startswith("T") else 0.0 for n in names])
    return (np.asarray(scores) >= cut).astype(float)


@dataclass
class FitRecord:
    seconds: float
    convergence_warnings: int


def fit_member(model: Any, Z: np.ndarray, y: np.ndarray) -> FitRecord:
    start = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(Z, y)
    n_conv = sum(1 for w in caught if issubclass(w.category, ConvergenceWarning))
    return FitRecord(time.perf_counter() - start, n_conv)


class Combiner:
    """Maps a (n, 6) member-score matrix to P(attack). Fitted on out-of-fold scores only."""

    def __init__(self, kind: str, names: Sequence[str]) -> None:
        if kind not in COMBINERS:
            raise ValueError(f"unknown combiner {kind!r}; choose from {COMBINERS}")
        self.kind = kind
        self.names = list(names)
        self.models: list[Any] = []

    def fit(self, scores: np.ndarray, y: np.ndarray) -> Combiner:
        scores, y = np.asarray(scores, dtype=float), np.asarray(y).astype(int)
        if self.kind == "stack":
            self.models = [
                Pipeline(
                    [("z", StandardScaler()), ("lr", LogisticRegression(max_iter=1000, random_state=SEED))]
                ).fit(scores, y)
            ]
        elif self.kind == "soft":
            self.models = [
                LogisticRegression(max_iter=1000, random_state=SEED).fit(scores[:, [j]], y)
                for j in range(scores.shape[1])
            ]
        return self

    def proba(self, scores: np.ndarray) -> np.ndarray:
        scores = np.asarray(scores, dtype=float)
        if self.kind == "stack":
            return np.asarray(self.models[0].predict_proba(scores)[:, 1])
        if self.kind == "soft":
            return np.mean([m.predict_proba(scores[:, [j]])[:, 1] for j, m in enumerate(self.models)], axis=0)
        return member_votes(scores, self.names).mean(axis=1)


class PenumbraEnsemble(ClassifierMixin, BaseEstimator):
    """3 trees + 3 SVMs behind the usual `predict_proba`, so it drops into the detector's Pipeline.

    Input is the detector's `prep` output (one-hot + imputed numerics, unscaled). `tree_arm` and
    `svm_arm` pick each family's transform; `hetero` in E9 is simply tree_arm="P0" with the SVMs on
    a scaled arm.
    """

    def __init__(
        self,
        tree_arm: str = "P0",
        svm_arm: str = "P1",
        combiner: str = "stack",
        n_components: int = 1000,
        svm_solver: str = "liblinear",
        svm_C: tuple[float, float, float] = (1.0, 1.0, 1.0),
        cv: int = 5,
        pca_k: int | None = None,
        max_iter: int = 1000,
        random_state: int = SEED,
    ) -> None:
        self.tree_arm = tree_arm
        self.svm_arm = svm_arm
        self.combiner = combiner
        self.n_components = n_components
        self.svm_solver = svm_solver
        self.svm_C = svm_C
        self.cv = cv
        self.pca_k = pca_k
        self.max_iter = max_iter
        self.random_state = random_state

    # --- fitting -----------------------------------------------------------------------------------

    def _members(self, n_rows: int) -> list[Any]:
        out = [member(t) for t in TREES]
        for name, C in zip(SVMS, self.svm_C, strict=True):
            out.append(
                member(
                    name,
                    C=C,
                    n_components=self.n_components,
                    solver=self.svm_solver,
                    n_rows=n_rows,
                    max_iter=self.max_iter,
                )
            )
        return out

    def _fit_members(self, X: np.ndarray, y: np.ndarray) -> tuple[dict[str, Any], list[Any], list[FitRecord]]:
        transforms: dict[str, Any] = {}
        Z: dict[str, np.ndarray] = {}
        for group, arm in (("tree", self.tree_arm), ("svm", self.svm_arm)):
            t = arm_transform(arm, pca_k=self.pca_k)
            if t is None:
                transforms[group], Z[group] = None, X
            else:
                transforms[group] = clone(t)
                Z[group] = transforms[group].fit_transform(X)
        models = self._members(len(X))
        records = [
            fit_member(m, Z["tree" if n.startswith("T") else "svm"], y)
            for n, m in zip(MEMBERS, models, strict=True)
        ]
        return transforms, models, records

    @staticmethod
    def _scores(transforms: dict[str, Any], models: list[Any], X: np.ndarray) -> np.ndarray:
        Z = {g: (X if t is None else t.transform(X)) for g, t in transforms.items()}
        return np.column_stack(
            [
                member_score(m, Z["tree" if n.startswith("T") else "svm"])
                for n, m in zip(MEMBERS, models, strict=True)
            ]
        )

    def fit(self, X: Any, y: Any, groups: Any = None) -> PenumbraEnsemble:
        X = np.asarray(X)
        y = np.asarray(y).astype(int)
        self.classes_ = np.array([0, 1])
        oof = np.zeros((len(X), len(MEMBERS)), dtype=float)
        if self.combiner != "hard":
            if groups is not None:
                from sklearn.model_selection import GroupKFold

                folds = list(GroupKFold(n_splits=self.cv).split(X, y, groups))
            else:
                folds = list(
                    StratifiedKFold(self.cv, shuffle=True, random_state=self.random_state).split(X, y)
                )
            for tr, va in folds:
                transforms, models, _ = self._fit_members(X[tr], y[tr])
                oof[va] = self._scores(transforms, models, X[va])
        self.oof_scores_ = oof
        self.transforms_, self.members_, records = self._fit_members(X, y)
        self.fit_seconds_ = {n: r.seconds for n, r in zip(MEMBERS, records, strict=True)}
        self.convergence_warnings_ = {
            n: r.convergence_warnings for n, r in zip(MEMBERS, records, strict=True)
        }
        self.combiner_ = Combiner(self.combiner, MEMBERS).fit(oof, y)
        self.pca_components_ = {g: pca_components(t) for g, t in self.transforms_.items()}
        return self

    # --- inference ---------------------------------------------------------------------------------

    def member_scores(self, X: Any) -> np.ndarray:
        return self._scores(self.transforms_, self.members_, np.asarray(X))

    def predict_proba(self, X: Any) -> np.ndarray:
        p = np.clip(self.combiner_.proba(self.member_scores(X)), 0.0, 1.0)
        return np.column_stack([1.0 - p, p])

    def predict(self, X: Any) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    # --- attribution -------------------------------------------------------------------------------

    def fit_importances(self, X: Any, y: Any, *, n_rows: int = 10_000, n_repeats: int = 3) -> np.ndarray:
        """Permutation importance in `prep`-output space, on rows the model did not train on.

        Six models of two families have no shared `feature_importances_`; the drop in ROC-AUC when a
        column is shuffled is the model-agnostic substitute. The detector calls this on its
        held-out calibration slice. Stored as `feature_importances_` so the alert path reads it
        the way it reads a forest's.
        """
        from sklearn.metrics import roc_auc_score

        X, y = np.asarray(X), np.asarray(y).astype(int)
        rng = np.random.default_rng(self.random_state)
        if len(X) > n_rows:
            take = rng.choice(len(X), size=n_rows, replace=False)
            X, y = X[take], y[take]
        if len(np.unique(y)) < 2:
            self.feature_importances_ = np.zeros(X.shape[1])
            return self.feature_importances_
        base = roc_auc_score(y, self.predict_proba(X)[:, 1])
        drops = np.zeros(X.shape[1])
        for j in range(X.shape[1]):
            for _ in range(n_repeats):
                Xp = X.copy()
                Xp[:, j] = rng.permutation(Xp[:, j])
                drops[j] += base - roc_auc_score(y, self.predict_proba(Xp)[:, 1])
        drops = np.clip(drops / n_repeats, 0.0, None)
        total = drops.sum()
        self.feature_importances_ = drops / total if total > 0 else drops
        return self.feature_importances_
