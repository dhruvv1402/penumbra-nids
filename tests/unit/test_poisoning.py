"""E7 drill mechanics: the target rule, the verdict simulation, and the flag accounting."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from penumbra.eval import poisoning


def alerted_frame() -> pd.DataFrame:
    fine = ["warezmaster"] * 60 + ["neptune"] * 200 + ["normal"] * 80
    y = [0 if f == "normal" else 1 for f in fine]
    return pd.DataFrame(
        {
            "pos": np.arange(len(fine)),
            "y": y,
            "fine": fine,
            "p_attack": np.where(np.array(y) == 1, 0.97, 0.6),
            # Benign alerts are misread as a spread of families, so honest clearances do not
            # concentrate. (When they do, family_campaign fires on honest accounts - see E7.)
            "pred_family": ["r2l"] * 60
            + ["dos"] * 200
            + list(np.random.default_rng(0).choice(["dos", "probe", "r2l", "u2r"], 80)),
        }
    )


def test_target_rule_picks_smallest_support_with_enough_alerts() -> None:
    train = pd.Series(["neptune"] * 5000 + ["warezmaster"] * 20 + ["rootkit"] * 5)
    alerted = pd.Series(["neptune"] * 300 + ["warezmaster"] * 60 + ["rootkit"] * 10)
    # rootkit has less support, but only 10 alerts: below the pre-registered floor of 50.
    assert poisoning.choose_target(train, alerted) == "warezmaster"


def test_target_rule_refuses_when_nothing_qualifies() -> None:
    with pytest.raises(RuntimeError):
        poisoning.choose_target(pd.Series(["a"] * 10), pd.Series(["a"] * 3))


@pytest.mark.parametrize("dose", [0.0, 0.25, 1.0])
def test_only_the_target_is_flipped_and_only_by_the_attacker(dose: float) -> None:
    records = poisoning.verdicts(alerted_frame(), target="warezmaster", dose=dose)
    flipped = [r for r in records if r["poisoned"]]
    assert len(flipped) == round(60 * dose)
    assert all(r["actor"] == poisoning.ATTACKER and r["verdict"] == "false_positive" for r in flipped)
    # Every unflipped verdict is truthful.
    frame = alerted_frame()
    for r in records:
        if not r["poisoned"]:
            truth = int(frame.loc[frame["pos"] == r["row"], "y"].iloc[0])
            assert r["verdict"] == ("true_positive" if truth else "false_positive")


def test_attacker_also_does_honest_cover_work() -> None:
    records = poisoning.verdicts(alerted_frame(), target="warezmaster", dose=0.0)
    assert any(r["actor"] == poisoning.ATTACKER for r in records)


def test_full_dose_attacker_is_flagged_and_honest_accounts_are_not() -> None:
    rates = poisoning.flag_rates(poisoning.verdicts(alerted_frame(), target="warezmaster", dose=1.0))
    assert rates["poisoned_flagged_any"] == 1.0
    # Her honest cover work clears benign alerts across families, which dilutes the campaign share
    # below 80% here - but the clearance RATE still gives her away.
    assert "actor_outlier" in rates["attacker_profile"]["flags"]
    assert all(not flags for flags in rates["honest_actor_flags"].values())


def test_wilson_interval_brackets_the_estimate() -> None:
    lo, hi = poisoning.wilson(8, 10)
    assert lo < 0.8 < hi
    assert poisoning.wilson(0, 0) != poisoning.wilson(0, 0)  # nan, not a fake certainty
