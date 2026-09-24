"""Where things live, and the constants that must not drift between modules.

Data does not live under the repository. C: has roughly 23 GB free on the development machine and
the CICIDS2017 improved release alone unpacks to 2-3 GB, so `data_root` resolves to G: when G:
exists and falls back to `./data` when it does not. Nothing else in the codebase is allowed to
hardcode a dataset path; ask `settings()` instead.
"""

from __future__ import annotations

import functools
from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]


def _default_data_root() -> Path:
    """G: if it is mounted, otherwise a directory inside the repo.

    The development machine keeps bulk data on G: (78 GB free vs 23 GB on C:). A teammate cloning
    on a single-drive laptop gets ./data and nothing breaks; only the free-space headroom differs.
    """
    g = Path("G:/penumbra-data")
    if g.drive and Path(g.drive + "/").exists():
        return g
    return REPO_ROOT / "data"


DEV_PII_KEY = "dev-only-not-a-secret"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="PENUMBRA_", env_file=".env", extra="ignore")

    data_root: Path = Field(default_factory=_default_data_root)
    artifact_root: Path = REPO_ROOT / "artifacts"
    # Where the API WRITES: the SQLite store, the audit chain, the SIEM mock. Separate from the
    # artifact root because a deployment mounts models read-only (docker-compose does) and the
    # service must still be able to record what it saw. Defaults to the artifact root locally.
    state_root: Path | None = None

    # One seed, threaded through every estimator and split. `penumbra.seeds.seed_everything`
    # is the only thing that should read it.
    seed: int = 42

    # HMAC key for IP pseudonymisation. Pseudonymisation, not anonymisation: IPv4 is 2**32 values,
    # so anyone holding this key can enumerate the whole mapping. It is therefore a secret, it is
    # stored apart from the data it protects, and re-identification is a role-gated audited call.
    # The default exists so tests run; production must set PENUMBRA_PII_HMAC_KEY.
    pii_hmac_key: str = Field(
        default=DEV_PII_KEY,
        # PENUMBRA_PII_SALT is what docker-compose and CI set; for a long time nothing read it, so the
        # key silently stayed at the public default. Both names are accepted now.
        validation_alias=AliasChoices("PENUMBRA_PII_HMAC_KEY", "PENUMBRA_PII_SALT"),
    )

    # Keys being rotated OUT, comma-separated, newest first. New pseudonyms always use the current
    # key; these are still accepted for matching and audited re-identification so pre-rotation data
    # stays queryable during the overlap window. Empty them to finish a rotation.
    pii_hmac_previous_keys: str = ""

    @property
    def previous_pii_keys(self) -> list[str]:
        return [k.strip() for k in self.pii_hmac_previous_keys.split(",") if k.strip()]

    @property
    def state_dir(self) -> Path:
        return self.state_root or self.artifact_root

    @property
    def raw_dir(self) -> Path:
        return self.data_root / "raw"

    @property
    def interim_dir(self) -> Path:
        return self.data_root / "interim"

    @property
    def model_dir(self) -> Path:
        return self.artifact_root / "models"

    @property
    def report_dir(self) -> Path:
        return self.artifact_root / "reports"

    @property
    def figure_dir(self) -> Path:
        return self.artifact_root / "figures"

    def ensure_dirs(self) -> None:
        for d in (self.raw_dir, self.interim_dir, self.model_dir, self.report_dir, self.figure_dir):
            d.mkdir(parents=True, exist_ok=True)


@functools.lru_cache(maxsize=1)
def settings() -> Settings:
    return Settings()


# --- Operational constants -------------------------------------------------------------------
#
# These are assumptions, not measurements, and every report that uses them says so. They are here
# rather than inline so that a reviewer can find and challenge all of them in one place.

# Deployment prevalence used to translate a measured TPR/FPR into an operational alert volume.
# UNSW-NB15's test set is ~55% attack; a real network is nowhere near that. See eval/prevalence.py.
OPERATIONAL_PREVALENCE: tuple[float, ...] = (1e-2, 1e-3, 1e-4)

# Flows/day on a mid-size enterprise segment, for the alert-volume arithmetic.
FLOWS_PER_DAY: int = 1_000_000

# Microsoft/Omdia State of the SOC 2026: mean time to investigate ~75 min, median ~45 min. We use
# the median, and we use it only to convert alert counts into analyst-hours.
MINUTES_PER_ALERT: float = 45.0

# A feature whose single-feature AUC exceeds this is treated as a suspected testbed artifact and
# every headline number is reported with and without it. See data/audit.py.
ARTIFACT_AUC_THRESHOLD: float = 0.90
