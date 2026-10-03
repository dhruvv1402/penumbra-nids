"""The E9 ensemble: 3 trees + 3 SVMs, its arms, its combiners, and its place in the detector.

The failure modes worth testing are the quiet ones: a combiner fitted on scores its members gave
their own training rows (it would trust the memorising tree), a transform fitted outside the fold,
and an ensemble that scores but silently loses its attribution or its pickle.
"""

from __future__ import annotations

import io

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from penumbra.models import ensemble as E
from penumbra.models import supervised
from penumbra.models.detector import PenumbraDetector
from penumbra.seeds import SEED
from tests.unit.test_registry import tiny_dataset


def data(n: int = 600, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.4).astype(int)
    X = np.column_stack(
        [
            rng.normal(0, 1, n) + 2 * y,
            rng.exponential(1, n) * (1 + 2 * y) * 1e4,  # heavy-tailed, byte-count-like
            rng.normal(0, 1, n),
            rng.integers(0, 2, n),  # a one-hot column
        ]
    )
    flip = rng.random(n) < 0.05
    return X, np.where(flip, 1 - y, y)


def small(**kw) -> E.PenumbraEnsemble:
    return E.PenumbraEnsemble(n_components=50, cv=3, **kw)


class TestEstimator:
    def test_fits_and_scores_like_a_classifier(self) -> None:
        X, y = data()
        m = small().fit(X, y)
        p = m.predict_proba(X)
        assert p.shape == (len(X), 2)
        assert np.allclose(p.sum(axis=1), 1.0)
        assert list(m.classes_) == [0, 1]
        assert m.member_scores(X).shape == (len(X), 6)

    def test_combiner_reads_out_of_fold_scores_only(self) -> None:
        # Each OOF row must come from members fitted without it. Rebuild fold 0 by hand and check
        # its validation rows' scores are exactly what a model blind to them produces.
        X, y = data()
        m = small().fit(X, y)
        tr, va = next(StratifiedKFold(3, shuffle=True, random_state=SEED).split(X, y))
        transforms, models, _ = m._fit_members(X[tr], y[tr])
        np.testing.assert_allclose(m.oof_scores_[va], m._scores(transforms, models, X[va]))
        # And they differ from in-sample scores, which is the point: T1 memorises.
        in_sample = m.member_scores(X)[:, 0]
        assert not np.allclose(in_sample, m.oof_scores_[:, 0])

    def test_stacking_matches_sklearn(self) -> None:
        X, y = data(n=500)
        ours = E.PenumbraEnsemble(tree_arm="P0", svm_arm="P0", n_components=40, cv=3).fit(X, y)
        members = [E.member(n, n_components=40, n_rows=len(X)) for n in E.MEMBERS]
        ref = StackingClassifier(
            estimators=list(zip(E.MEMBERS, members, strict=True)),
            final_estimator=Pipeline(
                [("z", StandardScaler()), ("lr", LogisticRegression(max_iter=1000, random_state=SEED))]
            ),
            cv=StratifiedKFold(3, shuffle=True, random_state=SEED),
        ).fit(X, y)
        np.testing.assert_allclose(ours.predict_proba(X), ref.predict_proba(X), atol=1e-6)

    def test_trees_do_not_care_about_scaling(self) -> None:
        # P1 is monotone per feature, so tree partitions are unchanged. Not exactly 100%: trees cast
        # to float32 and the signed log can merge near-equal values.
        X, y = data(n=1200)
        Xh, _ = data(n=400, seed=1)
        raw = E.member("T3").fit(X, y)
        t = E.arm_transform("P1")
        scaled = E.member("T3").fit(t.fit_transform(X), y)
        agree = (raw.predict(Xh) == scaled.predict(t.transform(Xh))).mean()
        assert agree >= 0.99

    def test_pca_arms_report_their_components(self) -> None:
        X, y = data()
        m = small(tree_arm="P0", svm_arm="P2").fit(X, y)
        assert m.pca_components_["tree"] is None
        assert 1 <= m.pca_components_["svm"] <= X.shape[1]
        with pytest.raises(ValueError, match="pca_k"):
            E.arm_transform("P3b")
        assert E.pca_components(E.arm_transform("P3b", pca_k=2).fit(X)) == 2

    def test_pca_before_scaling_finds_the_units(self) -> None:
        # Unscaled, the byte-count column carries almost all the variance: P3a keeps one component.
        X, _ = data()
        assert E.pca_components(E.arm_transform("P3a").fit(X)) == 1

    def test_hard_vote_has_seven_levels(self) -> None:
        X, y = data()
        p = small(combiner="hard").fit(X, y).predict_proba(X)[:, 1]
        assert set(np.round(p * 6).astype(int)) <= set(range(7))

    def test_sgd_solver_runs(self) -> None:
        X, y = data()
        p = small(svm_solver="sgd", svm_arm="P1").fit(X, y).predict_proba(X)[:, 1]
        assert np.isfinite(p).all()

    def test_round_trips_through_joblib(self) -> None:
        X, y = data()
        m = small(svm_arm="P2").fit(X, y)
        buf = io.BytesIO()
        joblib.dump(m, buf)
        buf.seek(0)
        np.testing.assert_allclose(joblib.load(buf).predict_proba(X), m.predict_proba(X))

    def test_permutation_importances_find_the_signal(self) -> None:
        X, y = data(n=800)
        m = small().fit(X, y)
        imp = m.fit_importances(X, y, n_repeats=2)
        assert imp.shape == (X.shape[1],)
        assert imp[2] < max(imp[0], imp[1])  # column 2 is pure noise


class TestInDetector:
    @pytest.fixture(scope="class")
    def det(self) -> PenumbraDetector:
        return PenumbraDetector(target_fpr=0.05).fit(tiny_dataset(n=900), model_name="ens")

    def test_family_model_is_a_forest(self, det) -> None:
        assert det.metadata.family_model_name == "rf"
        assert supervised.family_model_for("ens") == "rf"

    def test_alerts_keep_their_attribution(self, det) -> None:
        assert det.ranked_importances()

    def test_scores_and_survives_save_and_load(self, det, tmp_path) -> None:
        X = tiny_dataset(n=900).X_test.head(60)
        det.save(tmp_path)
        pd.testing.assert_frame_equal(PenumbraDetector.load(tmp_path).score(X), det.score(X))

    def test_threshold_is_not_collapsed(self, det) -> None:
        # The bug held-out calibration exists for: an in-sample tree threshold at ~0 fires on all.
        ds = tiny_dataset(n=900)
        fired = det.score(ds.X_test)["fired"].to_numpy() > 0
        assert fired[ds.y_test.to_numpy() == 0].mean() < 0.5
