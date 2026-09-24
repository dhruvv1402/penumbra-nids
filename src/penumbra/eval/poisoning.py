"""E7 - the poisoning drill. Pre-registered in docs/EXPERIMENTS.md before it was run.

The feedback loop lets analysts teach the model. An attacker holding one analyst account can use it
to teach the model that their own attack is benign. This measures, end to end on real data:

  1. how much damage a given number of flipped verdicts does, as a dose-response curve
  2. whether the verdict integrity flags notice the flipped verdicts, and how often they cry wolf on
     honest ones
  3. whether the canary gate refuses to promote the poisoned model, and whether it waves the
     honestly retrained one through

Data flow, all on NSL-KDD:

  KDDTrain+  ->  champion (the deployed model)
  KDDTest+   ->  split three ways, stratified by attack type, seeded:
                   canary     frozen trusted labels; the gate decides on these and nothing else
                   feedback   "live traffic": the champion alerts, analysts record verdicts
                   evaluation every reported number comes from here, never from the canary

Only rows the champion ALERTED on get verdicts, because an analyst never sees the rest. The attacker
can only poison a detection that fires - which is exactly the situation they are in: their traffic
is being caught, and they want it not to be.
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd

from penumbra.data.loaders.base import Dataset
from penumbra.eval import canary
from penumbra.feedback import integrity
from penumbra.models.detector import PenumbraDetector
from penumbra.seeds import SEED

HONEST = ("alice", "bob", "carol")
ATTACKER = "mallory"

# Pre-registered target rule (EXPERIMENTS.md E7): the attack type with the SMALLEST training support
# among those the champion alerts on at least this many times in the feedback pool. Poisoning risk
# concentrates where the model has the least honest evidence to outvote the flipped labels.
MIN_TARGET_ALERTS = 50
DOSES = (0.10, 0.25, 0.50, 1.00)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def choose_target(fine_train: pd.Series, alerted_fine: pd.Series, exclude: tuple[str, ...] = ()) -> str:
    support = fine_train.value_counts()
    alerts = alerted_fine[alerted_fine != "normal"].value_counts()
    eligible = [
        t for t, n in alerts.items() if n >= MIN_TARGET_ALERTS and support.get(t, 0) > 0 and t not in exclude
    ]
    if not eligible:
        raise RuntimeError("no attack type meets the pre-registered target rule")
    return str(min(eligible, key=lambda t: (support[t], t)))


def verdicts(alerted: pd.DataFrame, *, target: str, dose: float, seed: int = SEED) -> list[dict[str, Any]]:
    """Analyst verdicts on alerted feedback rows.

    Honest analysts - and the attacker, who does her share of honest work as cover - label rows by
    their true class. On top of that the attacker marks `dose` of the target type's alerts
    `false_positive`. Round-robin assignment, so every account sees a similar mix.
    """
    rng = np.random.default_rng(seed)
    target_rows = np.flatnonzero(alerted["fine"].to_numpy() == target)
    n_flip = int(round(len(target_rows) * dose))
    flipped = set(rng.choice(target_rows, size=n_flip, replace=False).tolist()) if n_flip else set()

    actors = (*HONEST, ATTACKER)
    out: list[dict[str, Any]] = []
    for i, (_, row) in enumerate(alerted.iterrows()):
        truth = int(row["y"])
        if i in flipped:
            out.append(_verdict(ATTACKER, "false_positive", row, poisoned=True))
        else:
            out.append(
                _verdict(
                    actors[i % len(actors)],
                    "true_positive" if truth else "false_positive",
                    row,
                    poisoned=False,
                )
            )
    return out


def _verdict(actor: str, verdict: str, row: pd.Series, *, poisoned: bool) -> dict[str, Any]:
    return {
        "actor": actor,
        "verdict": verdict,
        "p_attack": float(row["p_attack"]),
        # What the analyst sees on the alert: the model's predicted family, not the ground truth.
        "family": row["pred_family"] if isinstance(row["pred_family"], str) else None,
        "poisoned": poisoned,
        "row": int(row["pos"]),
    }


def flag_rates(records: list[dict[str, Any]]) -> dict[str, Any]:
    """How the integrity flags fare against known-poisoned and known-honest verdicts."""
    flagged = integrity.flag_verdicts(records, records)
    poisoned = [f for f in flagged if f["poisoned"]]
    honest_clear = [f for f in flagged if not f["poisoned"] and f["verdict"] == "false_positive"]
    honest_all = [f for f in flagged if not f["poisoned"]]

    def rate(rows: list[dict[str, Any]], flag: str | None = None) -> float:
        if not rows:
            return float("nan")
        hit = [r for r in rows if (r["flags"] if flag is None else flag in r["flags"])]
        return len(hit) / len(rows)

    names = ("confident_contradiction", "actor_outlier", "family_campaign", "family_skew")
    profiles = integrity.actor_profiles(records)
    # The attacker's cover work is truthful but done from the flagged account, so it is flagged
    # too - correctly. The false-flag rate that matters is on the honest ACCOUNTS.
    innocent = [f for f in flagged if f["actor"] in HONEST]
    innocent_clear = [f for f in innocent if f["verdict"] == "false_positive"]
    return {
        "n_poisoned": len(poisoned),
        "n_honest": len(honest_all),
        "n_honest_clearances": len(honest_clear),
        "poisoned_flagged_any": rate(poisoned),
        "honest_flagged_any": rate(honest_all),
        "honest_clearances_flagged_any": rate(honest_clear),
        "honest_accounts_flagged_any": rate(innocent),
        "honest_accounts_clearances_flagged_any": rate(innocent_clear),
        "honest_accounts_clearances_flagged_by_flag": {n: rate(innocent_clear, n) for n in names},
        "by_flag": {
            n: {
                "poisoned": rate(poisoned, n),
                "honest": rate(honest_all, n),
                "honest_clearances": rate(honest_clear, n),
            }
            for n in names
        },
        "attacker_profile": {
            "flags": profiles[ATTACKER].flags if ATTACKER in profiles else [],
            "clear_rate": profiles[ATTACKER].clear_rate if ATTACKER in profiles else None,
            "peer_clear_rate": profiles[ATTACKER].peer_clear_rate if ATTACKER in profiles else None,
            "z": profiles[ATTACKER].z if ATTACKER in profiles else None,
        },
        "honest_actor_flags": {a: profiles[a].flags for a in HONEST if a in profiles},
    }


def augmented(ds: Dataset, X_fb: pd.DataFrame, records: list[dict[str, Any]], coarse: pd.Series) -> Dataset:
    """Training set plus the feedback rows, labelled by verdict (not by truth)."""
    rows = [r["row"] for r in records]
    labels = np.array([1 if r["verdict"] == "true_positive" else 0 for r in records])
    fam = np.where(labels == 1, coarse.iloc[rows].to_numpy(), "normal")
    X_new = X_fb.iloc[rows].reset_index(drop=True)
    return replace(
        ds,
        X_train=pd.concat([ds.X_train, X_new], ignore_index=True),
        y_train=pd.concat([ds.y_train, pd.Series(labels)], ignore_index=True),
        fam_train=pd.concat([ds.fam_train, pd.Series(fam)], ignore_index=True),
    )


def _head_breakdown(scored: pd.DataFrame, mask: np.ndarray) -> dict[str, float]:
    fired = scored["fired"].to_numpy()[mask]
    n = max(len(fired), 1)
    return {
        "supervised": float(np.isin(fired, (1, 3)).sum() / n),
        "novelty_only": float((fired == 2).sum() / n),
        "missed": float((fired == 0).sum() / n),
    }


def run(
    ds: Dataset,
    fine_train: pd.Series,
    fine_test: pd.Series,
    *,
    model_name: str = "rf",
    target_fpr: float = 0.01,
    doses: tuple[float, ...] = DOSES,
    flags_only: bool = False,
    exclude: tuple[str, ...] = (),
    seed: int = SEED,
    on_progress: Any = None,
) -> dict[str, Any]:
    def step(msg: str) -> None:
        if on_progress:
            on_progress(msg)

    fine_test = fine_test.astype(str).reset_index(drop=True)
    can_idx, fb_idx, ev_idx = canary.live_split(fine_test, seed=seed)

    step("champion")
    champion = PenumbraDetector(target_fpr=target_fpr).fit(ds, model_name=model_name, fit_family_model=True)

    X_fb = ds.X_test.iloc[fb_idx].reset_index(drop=True)
    scored_fb = champion.score(X_fb)
    alert_mask = scored_fb["fired"].to_numpy() > 0
    alerted = pd.DataFrame(
        {
            "pos": np.flatnonzero(alert_mask),
            "y": ds.y_test.iloc[fb_idx].to_numpy()[alert_mask],
            "fine": fine_test.iloc[fb_idx].to_numpy()[alert_mask],
            "p_attack": scored_fb["p_attack"].to_numpy()[alert_mask],
            "pred_family": scored_fb["family"].to_numpy()[alert_mask],
        }
    )
    target = choose_target(fine_train.astype(str), alerted["fine"], exclude)
    step(f"target: {target} ({int((fine_train == target).sum())} training rows)")

    X_can, y_can, f_can = (
        ds.X_test.iloc[can_idx],
        ds.y_test.iloc[can_idx],
        fine_test.iloc[can_idx],
    )
    X_ev, y_ev, f_ev = ds.X_test.iloc[ev_idx], ds.y_test.iloc[ev_idx], fine_test.iloc[ev_idx]
    target_ev = f_ev.to_numpy() == target
    coarse_fb = ds.fam_test.iloc[fb_idx].reset_index(drop=True)

    champ_canary = canary.evaluate(champion, X_can, y_can, f_can)
    champ_scored_ev = champion.score(X_ev)

    def summarise(det: PenumbraDetector, scored_ev: pd.DataFrame) -> dict[str, Any]:
        rep = canary.operating_report(scored_ev["fired"].to_numpy() > 0, y_ev.to_numpy(), f_ev.to_numpy())
        k = int((scored_ev["fired"].to_numpy()[target_ev] > 0).sum())
        n = int(target_ev.sum())
        return {
            "evaluation": {
                "recall": rep.recall,
                "fpr": rep.fpr,
                "target_recall": k / n if n else float("nan"),
                "target_recall_ci95": wilson(k, n),
                "target_n": n,
                "target_heads": _head_breakdown(scored_ev, target_ev),
            },
        }

    arms: dict[str, Any] = {"champion": summarise(champion, champ_scored_ev)}
    arms["champion"]["canary"] = champ_canary.to_dict()

    plan = [("honest", 0.0), *[(f"poisoned_{int(d * 100)}pct", d) for d in doses]]
    for name, dose in plan:
        step(f"challenger: {name}")
        records = verdicts(alerted, target=target, dose=dose, seed=seed)
        if flags_only:
            # The flags need only the verdicts, not a retrained model: minutes instead of the
            # six full fits, for iterating on the integrity checks.
            arms[name] = {
                "dose": dose,
                "n_flipped": sum(r["poisoned"] for r in records),
                "integrity": flag_rates(records),
            }
            continue
        challenger = PenumbraDetector(target_fpr=target_fpr).fit(
            augmented(ds, X_fb, records, coarse_fb), model_name=model_name, fit_family_model=True
        )
        result = canary.gate(champ_canary, canary.evaluate(challenger, X_can, y_can, f_can))
        arm = summarise(challenger, challenger.score(X_ev))
        arm["dose"] = dose
        arm["n_verdicts"] = len(records)
        arm["n_flipped"] = sum(r["poisoned"] for r in records)
        arm["gate"] = {
            "passed": result.passed,
            "reasons": result.reasons,
            "target_canary_delta": result.family_deltas.get(target),
        }
        arm["integrity"] = flag_rates(records)
        arms[name] = arm

    return {
        "dataset": ds.name,
        "model": model_name,
        "target_fpr": target_fpr,
        "seed": seed,
        "excluded_targets": list(exclude),
        "target": target,
        "target_train_support": int((fine_train == target).sum()),
        "split": {"canary": len(can_idx), "feedback": len(fb_idx), "evaluation": len(ev_idx)},
        "feedback_alerts": int(alert_mask.sum()),
        "target_alerts_in_feedback": int((alerted["fine"] == target).sum()),
        "gate_policy": canary.GatePolicy().__dict__,
        "arms": arms,
    }
