"""A fitted forest as flat arrays, walked one depth level at a time for every tree at once.

scikit-learn scores a forest tree by tree: a Python-level loop over 200-300 estimators, each
calling into Cython. That is fast per row and slow per call - 16-24 ms for one flow on the UNSW
detector, whatever the batch size - and on a live stream the calls are small.

Here every tree's nodes live in one set of concatenated arrays, and a batch is pushed down all trees
together: one vectorised step per depth level (8 for the IsolationForest, 24 for the supervised
forest) instead of one Python call per tree. For one row that is about 1 ms. For large batches the
(rows x trees) working set outgrows the cache and sklearn's per-tree Cython wins, which is why
`models.compiled` measures the crossover and uses this only below it.

Exactness: the split test is sklearn's own - features cast to float32, compared `<=` against the
float64 threshold - so every row reaches the same leaf. The leaf values are read from the fitted
model. Only the order in which per-tree values are summed differs, which moves results in the last
bit (measured: 1e-16) and changed no decision on the UNSW test split.

Supported: binary RandomForestClassifier (P(class 1)) and IsolationForest (`decision_function`).
Anything else - XGBoost, multiclass, NaN in the input, which sklearn routes per node with learnt
missing-value directions - is refused with `Unsupported`, and the caller keeps sklearn.
"""

from __future__ import annotations

from typing import Any

import numpy as np

_LEAF = -1


class Unsupported(ValueError):
    """This model or input cannot be scored exactly by a flat forest."""


class FlatForest:
    """Concatenated tree arrays plus one scalar value per node."""

    def __init__(
        self, trees: list[Any], values: list[np.ndarray], features: list[np.ndarray] | None = None
    ) -> None:
        sizes = [t.node_count for t in trees]
        offsets = np.concatenate([[0], np.cumsum(sizes)[:-1]]).astype(np.int64)
        feature, threshold, left, right = [], [], [], []
        for i, (tree, off) in enumerate(zip(trees, offsets, strict=True)):
            leaf = tree.children_left == _LEAF
            node_ids = np.arange(tree.node_count, dtype=np.int64) + off
            f = tree.feature.astype(np.int64)
            if features is not None:
                # Trees fitted on a feature subset index into that subset.
                f = np.where(leaf, 0, np.asarray(features[i], dtype=np.int64)[np.where(leaf, 0, f)])
            feature.append(np.where(leaf, 0, f))
            threshold.append(np.where(leaf, np.inf, tree.threshold))
            # A leaf points at itself, so rows that arrive early simply stay put.
            left.append(np.where(leaf, node_ids, tree.children_left + off))
            right.append(np.where(leaf, node_ids, tree.children_right + off))
        # int32 indices: half the memory of int64, and 2**31 nodes is far beyond any forest here.
        self.feature = np.concatenate(feature).astype(np.int32)
        self.threshold = np.concatenate(threshold).astype(np.float64)
        self.left = np.concatenate(left).astype(np.int32)
        self.right = np.concatenate(right).astype(np.int32)
        self.value = np.concatenate(values).astype(np.float64)
        self.roots = offsets.astype(np.int32)
        self.n_trees = len(trees)
        self.depth = max(int(t.max_depth) for t in trees)
        self.missing_left = any(bool(np.any(getattr(t, "missing_go_to_left", np.zeros(1)))) for t in trees)

    def leaf_values(self, X: np.ndarray) -> np.ndarray:
        """(rows, trees): the value of the leaf each row reaches in each tree."""
        X = np.asarray(X, dtype=np.float32)
        if np.isnan(X).any():
            raise Unsupported("NaN in the input; sklearn routes missing values per node")
        n = X.shape[0]
        node = np.broadcast_to(self.roots, (n, self.n_trees)).copy()
        rows = np.arange(n)[:, None]
        for _ in range(self.depth):
            go_left = X[rows, self.feature[node]] <= self.threshold[node]
            node = np.where(go_left, self.left[node], self.right[node])
        return self.value[node]

    @property
    def nbytes(self) -> int:
        return sum(a.nbytes for a in (self.feature, self.threshold, self.left, self.right, self.value))


class FlatBinaryForest:
    """`predict_proba(X)[:, 1]` of a binary RandomForestClassifier."""

    def __init__(self, model: Any) -> None:
        from sklearn.ensemble import RandomForestClassifier

        if not isinstance(model, RandomForestClassifier):
            raise Unsupported(f"{type(model).__name__} is not a RandomForestClassifier")
        if getattr(model, "n_outputs_", 1) != 1 or len(model.classes_) != 2:
            raise Unsupported("only single-output binary forests are supported")
        trees = [e.tree_ for e in model.estimators_]
        values = []
        for t in trees:
            v = t.value[:, 0, :]
            # sklearn >= 1.4 stores class fractions; normalise anyway so older pickles agree.
            values.append(v[:, 1] / v.sum(axis=1))
        self.forest = FlatForest(trees, values)

    def p_class1(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self.forest.leaf_values(X).mean(axis=1), dtype=float)


class FlatIsolationForest:
    """`decision_function(X)` of a fitted IsolationForest."""

    def __init__(self, model: Any) -> None:
        from sklearn.ensemble import IsolationForest
        from sklearn.ensemble._iforest import _average_path_length

        if not isinstance(model, IsolationForest):
            raise Unsupported(f"{type(model).__name__} is not an IsolationForest")
        trees = [e.tree_ for e in model.estimators_]
        # Per node, what sklearn adds to a row's depth when it ends there.
        values = [
            np.asarray(d, dtype=np.float64) + np.asarray(a, dtype=np.float64) - 1.0
            for d, a in zip(model._decision_path_lengths, model._average_path_length_per_tree, strict=True)
        ]
        subsample = model._max_features != model.n_features_in_
        self.forest = FlatForest(trees, values, list(model.estimators_features_) if subsample else None)
        self.denominator = len(trees) * float(_average_path_length([model._max_samples])[0])
        self.offset = float(model.offset_)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        depths = self.forest.leaf_values(X).sum(axis=1)
        scores = np.ones_like(depths) if self.denominator == 0 else 2.0 ** (-depths / self.denominator)
        # sklearn: score_samples is the NEGATED anomaly score, and decision_function subtracts offset_.
        return np.asarray(-scores - self.offset, dtype=float)
