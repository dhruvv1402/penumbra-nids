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


def test_family_skew_survives_cover_work() -> None:
    # The E7 failure of family_campaign: lots of honest clearances on other families dilute the
    # attacker's concentration below 80%. family_skew compares per family against peers instead.
    history = honest_history()
    # Peers see r2l alerts too, and mostly confirm them. Skew is only measurable against peers who
    # have judged the same family.
    history += [_v(a, "true_positive", family="r2l") for a in ("alice", "bob", "carol") for _ in range(8)]
    history += [_v("mallory", "false_positive", family=f) for f in ("dos", "probe", "u2r") * 5]  # cover
    history += [_v("mallory", "true_positive", family="dos") for _ in range(30)]
    history += [_v("mallory", "false_positive", family="r2l") for _ in range(20)]  # the campaign
    profile = integrity.actor_profiles(history)["mallory"]
    assert "family_campaign" not in profile.flags  # the registered flag misses it
    assert "r2l" in profile.skewed_families
    flagged = integrity.flag_verdicts([_v("mallory", "false_positive", family="r2l")], history)
    assert "family_skew" in flagged[0]["flags"]


def test_family_skew_quiet_on_honest_triage() -> None:
    profiles = integrity.actor_profiles(honest_history())
    assert all(not p.skewed_families for p in profiles.values())


def test_family_less_verdicts_form_their_own_bucket() -> None:
    # E7: 244 of 269 targeted alerts had no predicted family. Skipping them made the flag blind.
    history = honest_history()
    history += [_v(a, "true_positive", family=None) for a in ("alice", "bob", "carol") for _ in range(10)]
    history += [_v("mallory", "false_positive", family=None) for _ in range(15)]
    history += [_v("mallory", "true_positive", family="dos") for _ in range(40)]
    assert integrity.UNRECOGNISED in integrity.actor_profiles(history)["mallory"].skewed_families
    flagged = integrity.flag_verdicts([_v("mallory", "false_positive", family=float("nan"))], history)
    assert "family_skew" in flagged[0]["flags"]
