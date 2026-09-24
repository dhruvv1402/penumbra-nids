"""Verdict integrity checks: each flag fires on its pattern and stays quiet on honest triage."""

from __future__ import annotations

from penumbra.feedback import integrity


def _v(actor: str, verdict: str, family: str | None = "dos", p: float = 0.6) -> dict:
    return {"actor": actor, "verdict": verdict, "family": family, "p_attack": p}


def honest_history() -> list[dict]:
    rows = []
    for actor in ("alice", "bob", "carol"):
        rows += [_v(actor, "true_positive") for _ in range(16)]
        rows += [_v(actor, "false_positive", family=f) for f in ("dos", "probe", "r2l", "u2r")]
    return rows


def test_honest_triage_raises_no_actor_flags() -> None:
    profiles = integrity.actor_profiles(honest_history())
    assert all(not p.flags for p in profiles.values())


def test_outlier_account_is_flagged() -> None:
    history = honest_history() + [_v("mallory", "false_positive", family=f) for f in ["dos", "probe"] * 10]
    profiles = integrity.actor_profiles(history)
    assert "actor_outlier" in profiles["mallory"].flags
    assert not profiles["alice"].flags


def test_family_campaign_is_flagged() -> None:
    history = honest_history() + [_v("mallory", "false_positive", family="r2l") for _ in range(12)]
    assert "family_campaign" in integrity.actor_profiles(history)["mallory"].flags


def test_leave_one_out_pooling() -> None:
    # Two actors: one clears everything, one clears nothing. Pooled WITH self, the clearer's own
    # rows drag the baseline toward it; leave-one-out compares it to the other actor only.
    history = (
        [_v("a", "false_positive") for _ in range(20)]
        + [_v("b", "true_positive") for _ in range(19)]
        + [_v("b", "false_positive")]
    )
    profile = integrity.actor_profiles(history)["a"]
    assert profile.peer_clear_rate == 1 / 20


def test_confident_contradiction_only_on_clearing_verdicts() -> None:
    flagged = integrity.flag_verdicts(
        [_v("alice", "false_positive", p=0.97), _v("alice", "true_positive", p=0.97)], honest_history()
    )
    assert flagged[0]["flags"] == ["confident_contradiction"]
    assert flagged[1]["flags"] == []


def test_campaign_flag_only_on_the_campaign_family() -> None:
    history = honest_history() + [_v("mallory", "false_positive", family="r2l") for _ in range(12)]
    flagged = integrity.flag_verdicts(
        [_v("mallory", "false_positive", family="r2l"), _v("mallory", "false_positive", family="dos")],
        history,
    )
    assert "family_campaign" in flagged[0]["flags"]
    assert "family_campaign" not in flagged[1]["flags"]
