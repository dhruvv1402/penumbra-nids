"""A real capture from owned hardware, scored against known ground truth.

The lab: two laptops on a phone hotspot (not a shared network), one capturing with tcpdump while it
browses normally and then nmap-scans the other. Ground truth is the address pair: every flow between
the attacker and the target is attack traffic, everything else is not.

Two measurements:

  out_of_the_box   the UNSW-NB15 detector as shipped. What does a model trained on a 2015 testbed
                   make of a real network it has never seen?
  rebaselined      the benign-only novelty head refitted on a random 70% of this capture's own
                   non-attack flows - no labels, which is the head's whole deployment story - and
                   scored on the held-out 30% against the attack flows.

Caveats are part of the result: one session, one attacker, a benign split made within the same
session, and scan flows that are highly uniform. Recall at 1% FPR on ~270 held-out benign flows is
3 flows and is below the estimation floor.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from penumbra.data.loaders.base import Dataset
from penumbra.eval.budget import flags_at_benign_budget
from penumbra.features.preprocess import benign_only_pipeline
from penumbra.models.novelty.ensemble import NoveltyEnsemble
from penumbra.seeds import SEED

FIT_FRACTION = 0.7


def attack_mask(meta: pd.DataFrame, attacker: str, target: str) -> np.ndarray:
    src = meta["Src IP"].astype(str).to_numpy()
    dst = meta["Dst IP"].astype(str).to_numpy()
    return ((src == attacker) & (dst == target)) | ((src == target) & (dst == attacker))


def out_of_the_box(scored: pd.DataFrame, verdicts: list[str], attack: np.ndarray) -> dict[str, Any]:
    fired = scored["fired"].to_numpy()
    abstain = scored["conformal_abstains"].to_numpy()
    verdict = np.asarray(verdicts)

    def group(mask: np.ndarray) -> dict[str, Any]:
        return {
            "flows": int(mask.sum()),
            "verdicts": dict(Counter(verdict[mask].tolist())),
            "reached_an_analyst": int(
                np.isin(verdict[mask], ["KNOWN_ATTACK", "SUSPECTED_NOVEL", "UNCERTAIN"]).sum()
            ),
            "supervised_head_fired": int(np.isin(fired[mask], (1, 3)).sum()),
            "abstained": int(abstain[mask].sum()),
        }

    return {"attack_flows": group(attack), "other_flows": group(~attack)}


def _summarise(scores: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    out: dict[str, Any] = {"roc_auc": float(roc_auc_score(y, scores))}
    for fpr in (0.01, 0.05, 0.10):
        budget = int(round((y == 0).sum() * fpr))
        flagged = flags_at_benign_budget(scores, y, budget)
        out[f"recall_at_{int(fpr * 100)}pct_fpr"] = float(flagged[y == 1].mean())
        out[f"benign_flagged_at_{int(fpr * 100)}pct_fpr"] = int(flagged[y == 0].sum())
    return out


def rebaseline(
    X: pd.DataFrame, attack: np.ndarray, stock_novelty: np.ndarray, seed: int = SEED
) -> dict[str, Any]:
    """Refit the novelty head on local benign flows; compare with the stock head on the same rows."""
    rng = np.random.default_rng(seed)
    benign = np.flatnonzero(~attack)
    rng.shuffle(benign)
    cut = int(len(benign) * FIT_FRACTION)
    fit_idx, hold_idx = np.sort(benign[:cut]), np.sort(benign[cut:])
    eval_idx = np.concatenate([hold_idx, np.flatnonzero(attack)])
    y = np.concatenate([np.zeros(len(hold_idx), int), np.ones(int(attack.sum()), int)])

    categorical = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    numeric = [c for c in X.columns if c not in categorical]
    Xf = X.iloc[fit_idx].reset_index(drop=True)
    zeros, normal = pd.Series(np.zeros(len(Xf), int)), pd.Series(["normal"] * len(Xf))
    ds = Dataset("lab", Xf, zeros, normal, Xf, zeros, normal, categorical=categorical, numeric=numeric)
    prep = benign_only_pipeline(ds)
    Z_fit = prep.fit_transform(Xf)
    local = NoveltyEnsemble().fit(Z_fit)
    local.calibrate_on(Z_fit)
    local_scores = local.score(prep.transform(X.iloc[eval_idx]), how="max")

    return {
        "fit_benign_flows": int(len(fit_idx)),
        "heldout_benign_flows": int(len(hold_idx)),
        "attack_flows": int(attack.sum()),
        "stock_unsw_novelty": _summarise(stock_novelty[eval_idx], y),
        "local_rebaselined_novelty": _summarise(local_scores, y),
    }
