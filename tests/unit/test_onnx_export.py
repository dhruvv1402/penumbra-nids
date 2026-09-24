"""ONNX export: the graph scores like the sklearn pipeline, including through inf values."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("skl2onnx")
pytest.importorskip("onnxruntime")

from penumbra.models import onnx_export, supervised  # noqa: E402
from tests.unit.test_registry import tiny_dataset  # noqa: E402


@pytest.fixture(scope="module")
def fitted():
    ds = tiny_dataset()
    model = supervised.build("rf", ds, n_classes=2, balanced=True)
    model.fit(ds.X_train, ds.y_train)
    return ds, model


def test_probabilities_match(fitted) -> None:
    ds, model = fitted
    scorer = onnx_export.OnnxScorer(
        onnx_export.export(model, ds.X_test.head(5), ds.categorical), ds.categorical
    )
    report = onnx_export.parity(model, scorer, ds.X_test, threshold=0.5, batch=64)
    assert report["max_abs_diff"] < 1e-5
    assert report["decisions_flipped"] == 0


def test_inf_is_sanitised_inside_the_graph(fitted) -> None:
    # The custom FiniteSanitiser converter: inf must become NaN and then the imputed median, exactly
    # as in sklearn, or zero-duration flows score differently in production than in evaluation.
    ds, model = fitted
    X = ds.X_test.head(20).copy()
    X.loc[X.index[:5], "c"] = np.inf
    scorer = onnx_export.OnnxScorer(onnx_export.export(model, X.head(5), ds.categorical), ds.categorical)
    np.testing.assert_allclose(scorer.attack_scores(X), model.predict_proba(X)[:, 1], atol=1e-5)


def _scorer(model, ds, X):
    blob = onnx_export.export(model, X.head(5), ds.categorical)
    return onnx_export.OnnxScorer(blob, ds.categorical, onnx_export.category_map(model, ds.categorical))


def test_unseen_category_without_a_bucket_is_refused_not_misscored(fitted) -> None:
    # tcp and udp are both frequent, so there is no infrequent bucket: sklearn would score an
    # all-zero indicator, which the ONNX encoder cannot express. Refuse, loudly.
    ds, model = fitted
    X = ds.X_test.head(10).copy()
    X["proto"] = "sctp"
    with pytest.raises(onnx_export.UnseenCategory):
        _scorer(model, ds, X).attack_scores(X)


def test_infrequent_grouping_is_reproduced_exactly() -> None:
    # One protocol in 3,000 rows falls under min_frequency, so sklearn groups it as "infrequent".
    # skl2onnx cannot convert that encoder; the faithful copy must reproduce it bit-for-bit, including
    # for a value never seen at all (which sklearn also sends to the infrequent bucket).
    ds = tiny_dataset(n=3000)
    ds.X_train.loc[ds.X_train.index[:1], "proto"] = "icmp"
    model = supervised.build("rf", ds, n_classes=2, balanced=True).fit(ds.X_train, ds.y_train)
    X = ds.X_test.head(30).copy()
    X.loc[X.index[:5], "proto"] = "icmp"
    X.loc[X.index[5:10], "proto"] = "sctp"
    np.testing.assert_allclose(
        _scorer(model, ds, X).attack_scores(X), model.predict_proba(X)[:, 1], atol=1e-5
    )
