"""The shape every loader returns.

One container for both datasets so that `eval/`, `models/` and the LOAFO harness never branch on
which dataset they were handed. Anything genuinely dataset-specific (does it have IP addresses? can
it be split temporally?) is a declared capability rather than something callers infer.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass(frozen=True)
class Capabilities:
    """What a dataset can physically support.

    UNSW-NB15's published split has no IP addresses and no timestamps, so correlation, entity-graph
    features and temporal splitting are impossible on it. Callers ask here rather than assuming, and
    the LOAFO/correlation code raises a clear error instead of silently producing a number computed
    from nothing. See ADR-0004.
    """

    has_entities: bool = False  # source/destination IP present
    has_timestamps: bool = False
    has_natural_unseen_split: bool = False  # test contains classes absent from train
    families: tuple[str, ...] = ()


@dataclass
class Dataset:
    """A loaded train/test pair with both label granularities."""

    name: str
    X_train: pd.DataFrame
    y_train: pd.Series  # binary: 0 benign, 1 attack
    fam_train: pd.Series  # family / category label as a string
    X_test: pd.DataFrame
    y_test: pd.Series
    fam_test: pd.Series
    categorical: list[str] = field(default_factory=list)
    numeric: list[str] = field(default_factory=list)
    # Features whose single-feature AUC is expected to be suspiciously high. Confirmed or refuted by
    # data/audit.py; every headline metric is reported with and without whatever it confirms.
    suspected_artifacts: list[str] = field(default_factory=list)
    capabilities: Capabilities = field(default_factory=Capabilities)

    def __post_init__(self) -> None:
        if list(self.X_train.columns) != list(self.X_test.columns):
            raise ValueError(
                f"{self.name}: train and test columns differ - "
                f"train-only={set(self.X_train.columns) - set(self.X_test.columns)}, "
                f"test-only={set(self.X_test.columns) - set(self.X_train.columns)}"
            )
        for split, X, y, fam in (
            ("train", self.X_train, self.y_train, self.fam_train),
            ("test", self.X_test, self.y_test, self.fam_test),
        ):
            if not (len(X) == len(y) == len(fam)):
                raise ValueError(f"{self.name}/{split}: length mismatch X={len(X)} y={len(y)} fam={len(fam)}")

    @property
    def feature_names(self) -> list[str]:
        return list(self.X_train.columns)

    @property
    def n_features(self) -> int:
        return self.X_train.shape[1]

    def benign_train(self) -> pd.DataFrame:
        """Training rows labelled benign.

        The novelty head fits on this and nothing else. Sharing a preprocessing pipeline fitted over
        all rows would leak the attack distribution into a model whose entire claim is that it has
        never seen an attack.
        """
        return self.X_train.loc[self.y_train == 0]

    def prevalence(self, split: str = "test") -> float:
        y = self.y_test if split == "test" else self.y_train
        return float(y.mean())

    def family_counts(self, split: str = "test") -> pd.Series:
        fam = self.fam_test if split == "test" else self.fam_train
        return fam.value_counts()

    def describe(self) -> str:
        lines = [
            f"{self.name}: {len(self.X_train):,} train / {len(self.X_test):,} test, "
            f"{self.n_features} features",
            f"  prevalence  train {self.prevalence('train'):.3f}  test {self.prevalence('test'):.3f}",
            f"  categorical {len(self.categorical)}  numeric {len(self.numeric)}",
        ]
        if self.suspected_artifacts:
            lines.append(f"  suspected artifacts: {', '.join(self.suspected_artifacts)}")
        caps = []
        if self.capabilities.has_entities:
            caps.append("entities")
        if self.capabilities.has_timestamps:
            caps.append("timestamps")
        if self.capabilities.has_natural_unseen_split:
            caps.append("natural-unseen-split")
        lines.append(f"  capabilities: {', '.join(caps) or 'flow features only'}")
        return "\n".join(lines)


class DatasetNotFetched(FileNotFoundError):
    """Raised with an actionable message rather than a bare path error."""

    def __init__(self, path: str, dataset: str) -> None:
        super().__init__(f"{path} not found. Run `penumbra data fetch --dataset {dataset}` first.")
