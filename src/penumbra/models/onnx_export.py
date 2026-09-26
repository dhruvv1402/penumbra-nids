"""ONNX export of the supervised head, with parity measured rather than assumed.

Two reasons to ship an ONNX artifact next to the joblib one:

  * **No pickle.** A joblib file is a pickle and loading it executes code; the registry guards that
    with hashes. An ONNX graph is data - loading it runs no Python - so a scoring service that only
    needs the supervised head can drop the pickle attack surface entirely.
  * **Portability.** onnxruntime runs where Python and scikit-learn do not: a sensor, a container
    without the training stack, a different language.

What is exported: the supervised pipeline (preprocessing + forest). What is not: the novelty head
(its FrequencyEncoder and MLP autoencoder would each need custom converters) and the conformal
layer (a pair of thresholds, trivially re-implemented). The report says so.

Parity is checked on the full test split, not a sample: maximum absolute probability difference,
and how many rows land on the other side of the deployed threshold. float32 inside ONNX against
float64 in sklearn means a handful of rows sitting exactly at a threshold CAN flip; the count is
reported instead of hidden behind a tolerance.

Requires the `onnx` extra (skl2onnx, onnxruntime).
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd

from penumbra.features.preprocess import FiniteSanitiser

_REGISTERED = False


def _register() -> None:
    """Teach skl2onnx our one custom transformer: inf -> NaN, then the imputer takes over."""
    global _REGISTERED
    if _REGISTERED:
        return
    from skl2onnx import update_registered_converter
    from skl2onnx.algebra.onnx_ops import OnnxIsInf, OnnxWhere

    def shape(operator: Any) -> None:
        operator.outputs[0].type = operator.inputs[0].type.__class__(operator.inputs[0].type.shape)

    def convert(scope: Any, operator: Any, container: Any) -> None:
        opv = container.target_opset
        x = operator.inputs[0]
        nan = np.array([np.nan], dtype=np.float32)
        OnnxWhere(
            OnnxIsInf(x, op_version=opv), nan, x, op_version=opv, output_names=operator.outputs[:1]
        ).add_to(scope, container)

    update_registered_converter(FiniteSanitiser, "PenumbraFiniteSanitiser", shape, convert)
    _REGISTERED = True


def export(pipeline: Any, X_example: pd.DataFrame, categorical: list[str]) -> bytes:
    """Convert a fitted supervised Pipeline. One graph input per column, as the pipeline sees them."""
    _register()
    pipeline = faithful_copy(pipeline, categorical)
    from skl2onnx import to_onnx
    from skl2onnx.common.data_types import FloatTensorType, StringTensorType

    inputs = [
        (c, StringTensorType([None, 1]) if c in categorical else FloatTensorType([None, 1]))
        for c in X_example.columns
    ]
    clf = pipeline.named_steps["clf"]
    model = to_onnx(pipeline, initial_types=inputs, options={id(clf): {"zipmap": False}}, target_opset=17)
    return bytes(model.SerializeToString())


def feed(
    X: pd.DataFrame,
    categorical: list[str],
    categories: dict[str, tuple[set[str], bool]] | None = None,
) -> dict[str, np.ndarray]:
    """One (n, 1) array per graph input.

    The numeric block is converted once, column-major, so each input is a contiguous column view.
    Converting column by column through `X[[c]]` cost ~35 ms per call whatever the batch size -
    150 times the 0.2 ms the graph itself takes to score one row.

    With `categories`, rare and unseen values go to the rare bucket the way sklearn's encoder sends
    them to its infrequent column; a value with nowhere to go raises `UnseenCategory`.
    """
    categories = categories or {}
    numeric = [c for c in X.columns if c not in categorical]
    out: dict[str, np.ndarray] = {}
    if numeric:
        block = np.asfortranarray(X[numeric].to_numpy(dtype=np.float32))
        out.update({c: block[:, i : i + 1] for i, c in enumerate(numeric)})
    for c in categorical:
        values = X[c].astype(str).to_numpy(dtype=object)
        if c in categories:
            known, has_rare = categories[c]
            other = ~np.isin(values, list(known))
            if other.any():
                if not has_rare:
                    raise UnseenCategory(
                        f"{c}={sorted(set(values[other]))[:5]} never seen in training and there is no "
                        "rare bucket; sklearn scores it as an all-zero indicator, which the ONNX encoder "
                        "cannot express. Use the sklearn detector for these rows."
                    )
                values = np.where(other, RARE, values).astype(object)
        out[c] = values.reshape(-1, 1)
    return out


class UnseenCategory(ValueError):
    """A categorical value the ONNX graph cannot encode.

    sklearn's encoder scores an unseen value as an all-zero indicator. ONNX's OneHotEncoder has no
    all-zero mode for unknowns; it fails the whole batch. We refuse the rows explicitly rather than
    let a batch fail somewhere less legible, or score something different from what was evaluated.
    """


class ExportUnsupported(RuntimeError):
    """The fitted pipeline uses something skl2onnx cannot express faithfully."""


RARE = "__penumbra_rare__"


def category_map(pipeline: Any, categorical: list[str]) -> dict[str, tuple[set[str], bool]]:
    """Per categorical column: the frequent values the graph encodes, and whether a rare bucket exists."""
    if not categorical:
        return {}
    encoder = pipeline.named_steps["prep"].named_transformers_["cat"]
    infrequent = getattr(encoder, "infrequent_categories_", None) or [None] * len(categorical)
    out: dict[str, tuple[set[str], bool]] = {}
    for col, cats, rare in zip(categorical, encoder.categories_, infrequent, strict=True):
        rare_set = {str(r) for r in rare} if rare is not None else set()
        out[col] = ({str(c) for c in cats} - rare_set, rare is not None)
    return out


def faithful_copy(pipeline: Any, categorical: list[str]) -> Any:
    """A copy of the pipeline whose one-hot encoder skl2onnx can express exactly.

    sklearn groups rare categories into one trailing "infrequent" column per feature; skl2onnx has
    no converter for that. The fitted forest only sees column POSITIONS, so we swap in a plain
    encoder with an explicit category list - the frequent values in sklearn's order, then a sentinel
    in the infrequent column's place - and let the scorer map rare and unseen values to the
    sentinel, which is precisely what `handle_unknown="infrequent_if_exist"` does. Parity on the
    full test split is what proves the swap is exact; it is measured, not assumed.
    """
    import copy

    from sklearn.preprocessing import OneHotEncoder

    if not categorical:
        return pipeline
    clone = copy.deepcopy(pipeline)
    prep = clone.named_steps["prep"]
    old = prep.named_transformers_["cat"]
    infrequent = getattr(old, "infrequent_categories_", None) or [None] * len(categorical)
    categories = []
    for cats, rare in zip(old.categories_, infrequent, strict=True):
        rare_set = set(rare) if rare is not None else set()
        frequent = [c for c in cats if c not in rare_set]
        categories.append(frequent + ([RARE] if rare is not None else []))
    new = OneHotEncoder(categories=categories, handle_unknown="error", sparse_output=False)
    new.fit(pd.DataFrame({c: [cats[0]] for c, cats in zip(categorical, categories, strict=True)}))
    if new.transform(
        pd.DataFrame({c: [cats[0]] for c, cats in zip(categorical, categories, strict=True)})
    ).shape[1] != len(old.get_feature_names_out()):
        raise ExportUnsupported("rebuilt encoder does not reproduce the fitted column layout")
    prep.transformers_ = [(n, new if n == "cat" else t, cols) for n, t, cols in prep.transformers_]
    return clone


class OnnxScorer:
    """p_attack from an ONNX graph, with no scikit-learn and no pickle involved."""

    def __init__(
        self,
        blob: bytes,
        categorical: list[str],
        categories: dict[str, tuple[set[str], bool]] | None = None,
    ) -> None:
        import onnxruntime as ort

        self.session = ort.InferenceSession(blob, providers=["CPUExecutionProvider"])
        self.categorical = categorical
        self.categories = categories or {}

    def attack_scores(self, X: pd.DataFrame) -> np.ndarray:
        probabilities = self.session.run(None, feed(X, self.categorical, self.categories))[1]
        return np.asarray(probabilities[:, 1], dtype=np.float64)


def parity(
    pipeline: Any, scorer: OnnxScorer, X: pd.DataFrame, *, threshold: float, batch: int = 2048
) -> dict[str, Any]:
    """Compare ONNX against sklearn on every row, and time both at a realistic batch size."""
    p_sk = pipeline.predict_proba(X)[:, 1]
    p_ox = scorer.attack_scores(X)
    diff = np.abs(p_sk - p_ox)
    flipped = (p_sk >= threshold) != (p_ox >= threshold)

    chunk = X.head(batch)

    def timed(fn: Any, repeats: int = 5) -> float:
        fn(chunk)  # warm-up
        t = time.perf_counter()
        for _ in range(repeats):
            fn(chunk)
        return (time.perf_counter() - t) / repeats

    t_sk = timed(lambda c: pipeline.predict_proba(c))
    t_ox = timed(scorer.attack_scores)
    return {
        "n_rows": int(len(X)),
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "threshold": float(threshold),
        "decisions_flipped": int(flipped.sum()),
        "latency_batch": batch,
        "sklearn_ms_per_batch": 1000 * t_sk,
        "onnx_ms_per_batch": 1000 * t_ox,
        "speedup": t_sk / t_ox if t_ox else float("nan"),
        "exported": "supervised head (preprocessing + forest)",
        "not_exported": "novelty head (custom encoder + autoencoder), conformal layer (two thresholds)",
    }
