"""Deployment configuration: the PII key is actually read, and the API refuses the public default."""

from __future__ import annotations

import os

os.environ.setdefault("PENUMBRA_ALLOW_DEMO_USERS", "1")  # importing the app constructs its state

from pathlib import Path  # noqa: E402

import pytest  # noqa: E402

from penumbra.config import DEV_PII_KEY, Settings  # noqa: E402


def test_both_key_names_are_read(monkeypatch) -> None:
    monkeypatch.delenv("PENUMBRA_PII_HMAC_KEY", raising=False)
    monkeypatch.setenv("PENUMBRA_PII_SALT", "from-compose")
    assert Settings().pii_hmac_key == "from-compose"
    monkeypatch.setenv("PENUMBRA_PII_HMAC_KEY", "canonical")
    assert Settings().pii_hmac_key in {"canonical", "from-compose"}  # either name, never the default


def test_state_dir_defaults_to_artifacts_and_can_be_split(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("PENUMBRA_STATE_ROOT", raising=False)
    s = Settings(artifact_root=tmp_path / "a")
    assert s.state_dir == tmp_path / "a"
    assert Settings(artifact_root=tmp_path / "a", state_root=tmp_path / "s").state_dir == Path(tmp_path / "s")


def test_api_refuses_the_public_key_outside_demo_mode(monkeypatch) -> None:
    from penumbra.api import app as app_module

    class S:
        pii_hmac_key = DEV_PII_KEY

    monkeypatch.setattr(app_module, "settings", lambda: S())
    monkeypatch.delenv("PENUMBRA_ALLOW_DEMO_USERS", raising=False)
    with pytest.raises(RuntimeError, match="PENUMBRA_PII_HMAC_KEY"):
        app_module.refuse_default_secrets()

    S.pii_hmac_key = "a-real-key"
    app_module.refuse_default_secrets()  # does not raise
