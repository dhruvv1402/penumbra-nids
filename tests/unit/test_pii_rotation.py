"""PII key rotation: an overlap window keeps old pseudonyms usable, and closing it retires them."""

from __future__ import annotations

import pytest

from penumbra import config
from penumbra.api.security import pii


@pytest.fixture()
def keys(monkeypatch):
    def use(current: str, previous: str = "") -> None:
        monkeypatch.setenv("PENUMBRA_PII_HMAC_KEY", current)
        monkeypatch.setenv("PENUMBRA_PII_HMAC_PREVIOUS_KEYS", previous)
        monkeypatch.delenv("PENUMBRA_PII_SALT", raising=False)
        config.settings.cache_clear()

    yield use
    config.settings.cache_clear()


def test_new_pseudonyms_use_only_the_current_key(keys) -> None:
    keys("old")
    before = pii.pseudonymise_ip("10.0.0.5")
    keys("new", "old")
    assert pii.pseudonymise_ip("10.0.0.5") != before


def test_overlap_window_keeps_old_alerts_reidentifiable(keys) -> None:
    keys("old")
    stored_last_week = pii.pseudonymise_ip("10.0.0.5")
    keys("new", "old")
    index = pii.build_reverse_index(["10.0.0.5"])
    assert pii.reidentify(stored_last_week, index, role="admin", actor="a", reason="INC-1") == "10.0.0.5"
    assert stored_last_week in pii.pseudonyms_for("10.0.0.5")


def test_closing_the_window_retires_the_old_key(keys) -> None:
    keys("old")
    stored_last_week = pii.pseudonymise_ip("10.0.0.5")
    keys("new")
    with pytest.raises(KeyError):
        pii.reidentify(
            stored_last_week, pii.build_reverse_index(["10.0.0.5"]), role="admin", actor="a", reason="x"
        )


def test_fingerprints_never_reveal_keys(keys) -> None:
    keys("new-secret", "old-secret")
    fp = pii.key_fingerprints()
    assert len(fp["previous"]) == 1
    assert "secret" not in str(fp)
