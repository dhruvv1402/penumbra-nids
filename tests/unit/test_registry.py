"""Model registry and canary gate.

The registry's security property is that nothing is unpickled unless its hash matches the manifest:
a joblib file is a pickle, and loading a pickle is code execution. Its governance property is that
nothing becomes champion without a passed gate. Both are tested by trying to break them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from penumbra.data.loaders.base import Dataset
from penumbra.eval import canary
from penumbra.models.detector import PenumbraDetector
from penumbra.models.registry import ModelRegistry, RegistryError, TamperedArtifact


def tiny_dataset(n: int = 600, seed: int = 0) -> Dataset:
    rng = np.random.default_rng(seed)

    def split(m: int) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
        y = (rng.random(m) < 0.4).astype(int)
        X = pd.DataFrame(
            {
                "a": rng.normal(0, 1, m) + 3 * y,
                "b": rng.normal(0, 1, m),
                "c": rng.exponential(1, m) * (1 + y),
                "proto": rng.choice(["tcp", "udp"], m),
            }
        )
        fam = pd.Series(np.where(y == 1, rng.choice(["dos", "probe"], m), "normal"))
        return X, pd.Series(y), fam

    Xtr, ytr, ftr = split(n)
    Xte, yte, fte = split(n // 2)
    return Dataset(
        name="tiny",
        X_train=Xtr,
        y_train=ytr,
        fam_train=ftr,
        X_test=Xte,
        y_test=yte,
        fam_test=fte,
        categorical=["proto"],
        numeric=["a", "b", "c"],
    )


@pytest.fixture(scope="module")
def detector() -> PenumbraDetector:
    return PenumbraDetector(target_fpr=0.05).fit(tiny_dataset())


@pytest.fixture()
def registry(tmp_path) -> ModelRegistry:
    return ModelRegistry(tmp_path, "tiny")


PASSED = {"passed": True, "reasons": []}
FAILED = {"passed": False, "reasons": ["G2 family 'dos' recall 0.9 -> 0.2"]}


class TestRegistry:
    def test_first_champion_is_ungated_and_says_so(self, registry, detector) -> None:
        v1 = registry.register(detector, created_by="admin")
        with pytest.raises(RegistryError, match="no passed canary gate"):
            registry.promote(v1.version, approver="admin")
        state = registry.promote(v1.version, approver="admin", allow_without_gate=True)
        assert state["history"][-1]["gate_passed"] is False
        assert registry.champion() == v1.version

    def test_later_promotions_need_a_passed_gate(self, registry, detector) -> None:
        v1 = registry.register(detector, created_by="admin")
        registry.promote(v1.version, approver="admin", allow_without_gate=True)
        v2 = registry.register(detector, created_by="admin", parent=v1.version)

        # allow_without_gate only ever applies to the FIRST champion.
        with pytest.raises(RegistryError):
            registry.promote(v2.version, approver="admin", allow_without_gate=True)
        registry.attach_gate(v2.version, FAILED)
        with pytest.raises(RegistryError):
            registry.promote(v2.version, approver="admin")
        registry.attach_gate(v2.version, PASSED)
        registry.promote(v2.version, approver="senior")
        assert registry.champion() == v2.version

    def test_rollback_is_a_recorded_pointer_change(self, registry, detector) -> None:
        v1 = registry.register(detector, created_by="admin")
        registry.promote(v1.version, approver="admin", allow_without_gate=True)
        v2 = registry.register(detector, created_by="admin", parent=v1.version)
        registry.attach_gate(v2.version, PASSED)
        registry.promote(v2.version, approver="senior")

        state = registry.rollback(approver="admin")
        assert registry.champion() == v1.version
        assert state["history"][-1]["action"] == "rollback"
        assert (registry.path(v2.version) / "detector.joblib").exists()  # nothing deleted

    def test_tampered_artifact_is_never_unpickled(self, registry, detector) -> None:
        v1 = registry.register(detector, created_by="admin")
        registry.promote(v1.version, approver="admin", allow_without_gate=True)
        with (registry.path(v1.version) / "detector.joblib").open("ab") as f:
            f.write(b"\x00")
        assert registry.verify(v1.version) == ["detector.joblib"]
        with pytest.raises(TamperedArtifact):
            registry.load()

    def test_rollback_refuses_a_tampered_target(self, registry, detector) -> None:
        v1 = registry.register(detector, created_by="admin")
        registry.promote(v1.version, approver="admin", allow_without_gate=True)
        v2 = registry.register(detector, created_by="admin", parent=v1.version)
        registry.attach_gate(v2.version, PASSED)
        registry.promote(v2.version, approver="senior")
        with (registry.path(v1.version) / "detector.joblib").open("ab") as f:
            f.write(b"\x00")
        with pytest.raises(TamperedArtifact):
            registry.rollback(approver="admin")
        assert registry.champion() == v2.version

    def test_loaded_champion_scores_identically(self, registry, detector) -> None:
        v1 = registry.register(detector, created_by="admin")
        registry.promote(v1.version, approver="admin", allow_without_gate=True)
        X = tiny_dataset(seed=1).X_test
        pd.testing.assert_frame_equal(registry.load().score(X), detector.score(X))

    @pytest.mark.parametrize("name", ["../escape", "a/b", "..", ""])
    def test_version_names_cannot_walk_out_of_the_tree(self, registry, name) -> None:
        with pytest.raises(RegistryError):
            registry.path(name)


class TestGate:
    def report(self, recall: float, fpr: float, fam: dict[str, float]) -> canary.OperatingReport:
        return canary.OperatingReport(
            n_attack=1000,
            n_benign=1000,
            recall=recall,
            fpr=fpr,
            per_family={k: {"n": 100, "recall": v} for k, v in fam.items()},
        )

    def test_targeted_regression_fails_even_when_aggregate_is_flat(self) -> None:
        # The case the per-family check exists for: overall recall barely moves, one family collapses.
        champ = self.report(0.90, 0.02, {"dos": 0.95, "r2l": 0.60})
        chall = self.report(0.89, 0.02, {"dos": 0.96, "r2l": 0.20})
        result = canary.gate(champ, chall)
        assert not result.passed
        assert any("r2l" in r and r.startswith("G2") for r in result.reasons)

    def test_honest_improvement_passes(self) -> None:
        champ = self.report(0.90, 0.02, {"dos": 0.95, "r2l": 0.60})
        chall = self.report(0.92, 0.015, {"dos": 0.95, "r2l": 0.75})
        assert canary.gate(champ, chall).passed

    def test_fpr_rise_fails(self) -> None:
        champ = self.report(0.90, 0.02, {})
        assert not canary.gate(champ, self.report(0.95, 0.05, {})).passed

    def test_small_families_are_not_gated(self) -> None:
        champ = canary.OperatingReport(10, 10, 0.9, 0.0, {"u2r": {"n": 5, "recall": 1.0}})
        chall = canary.OperatingReport(10, 10, 0.9, 0.0, {"u2r": {"n": 5, "recall": 0.0}})
        assert canary.gate(champ, chall).passed  # 5 rows cannot distinguish a drop from noise

    def test_live_split_is_disjoint_stratified_and_deterministic(self) -> None:
        labels = pd.Series(["normal"] * 100 + ["dos"] * 40 + ["rare"] * 2)
        a = canary.live_split(labels)
        b = canary.live_split(labels)
        assert all(np.array_equal(x, y) for x, y in zip(a, b, strict=True))
        allidx = np.concatenate(a)
        assert len(allidx) == len(set(allidx)) == len(labels)
        assert (labels.iloc[a[0]] == "dos").sum() == 12  # 30% of 40
