# Evaluation methodology

> **Status: pre-results.** The methodology below is final and was fixed before any model was trained.
> Numbers are `[pending]` and will be written by `penumbra eval --report`, not by hand.
> Hypotheses are pre-registered in [`EXPERIMENTS.md`](EXPERIMENTS.md).

---

## Why this document is longer than the results section

Accuracy on an intrusion dataset is close to meaningless, and most of the ways to get a flattering
number are accidental rather than dishonest. This document exists to say, in advance, which
flattering numbers we have decided not to produce.

---

## 1. Prevalence — the thing that invalidates most reported IDS metrics

UNSW-NB15's test set is **~55% attack**. Real enterprise networks run somewhere between 1-in-10³ and
1-in-10⁵ flows.

Precision, PPV, PR-AUC and "alerts per 1,000 flows" are **all prevalence-dependent**. Every one of
them, computed on this data, is optimistic by two to four orders of magnitude relative to deployment.

Consequences:

- **Primary metrics are the prevalence-invariant pair: TPR and FPR**, plus ROC-AUC and recall at fixed
  FPR. These do not move with base rate.
- **PR-AUC is never reported alone.** It always carries the prevalence it was computed at and its
  no-skill baseline, which equals the prevalence. `PR-AUC 0.97 (π = 0.55, no-skill 0.55)`.
- `eval/prevalence.py` re-weights measured TPR/FPR to stated operational prevalences (1e-2, 1e-3, 1e-4)
  and emits modelled PPV, alerts/day and analyst-hours/day. Output is labelled **modelled, not
  measured.**

**The base-rate arithmetic, done explicitly:** 1,000,000 flows/day at π = 1e-4 means 100 attack flows.
At 1% FPR that is ~9,999 false alerts against ~100 true ones — PPV ≈ 1%. This is Axelsson's 2000 result
and it is the reason signature IDSs still exist. It is also why the novelty lane is budget-capped
rather than ticket-generating.

## 2. Artifact audit — how much of the score is the testbed?

`data/audit.py` fits a single-feature decision stump per column and ranks by AUC. Any feature with solo
AUC > 0.90 is quarantined as a suspected testbed artifact.

Declared suspects, in advance: `sttl`, `ct_state_ttl`, `is_sm_ips_ports`, `dttl` on UNSW-NB15;
`Destination Port` on CICIDS2017.

The reasoning for `sttl`: attack and benign traffic in UNSW-NB15 were generated from different hosts,
so time-to-live encodes the generator rather than the behaviour. A model leaning on it has learned the
testbed, not the attack.

**Every headline number is reported twice — with and without the quarantined set — and the second one
is what we stand behind.**

| | with artifacts | artifacts removed |
|---|---|---|
| `sttl` solo AUC | `[pending]` | — |
| Binary ROC-AUC | `[pending]` | `[pending]` |

## 3. Negative controls

Three checks whose only purpose is to catch a pipeline bug masquerading as a result.

| Control | Expected | Result |
|---|---|---|
| Shuffled labels | collapses to chance | `[pending]` |
| Row index / `id` only | chance (no ordering leak) | `[pending]` |
| Train/test duplicate overlap | reported, both raw and de-duplicated scores | `[pending]` |

## 4. Splits

| Dataset | Split | Why |
|---|---|---|
| UNSW-NB15 | published 175,341 / 82,332 | comparable to the literature |
| NSL-KDD | `KDDTrain+` / `KDDTest+`, **never re-shuffled** | preserves 17 attack types that appear only in test — the sole natural zero-day holdout |
| CICIDS2017 | day-based: train Mon–Wed, test Thu–Fri | random splitting leaks future into past |

De-duplication happens **before** splitting. Roughly 20% of CICIDS2017 rows are exact duplicates; if
they straddle the boundary, the test score is partly memorisation.

## 5. Class imbalance

Full ablation, **every cell evaluated on the natural distribution**: `class_weight` · SMOTE · SMOTE-NC ·
ADASYN · BorderlineSMOTE · RandomUnderSampler · BalancedRandomForest · EasyEnsemble · focal loss ·
threshold-moving on calibrated output.

Discipline enforced by test:
- Samplers fit **inside CV folds, on training data only**. A pipeline-inspection test fails the build
  otherwise.
- **SMOTE-NC where categoricals exist** — interpolating between one-hot columns produces flows that
  cannot exist.
- **Resampling decalibrates.** Recalibration on a held-out set happens after resampling, or the
  calibration claim is void.

Two results we intend to publish either way: the deliberate **leakage demonstration** (SMOTE before the
split vs inside folds, side by side), and a single SMOTE-generated flow row with fractional packet
counts — the physical argument against resampling network data.

## 6. Statistics

- **Stratified bootstrap** 95% CIs on every headline number. Unstratified resampling makes rare classes
  vanish and their intervals meaningless. Temporal data uses a **moving-block** bootstrap, not iid —
  iid resampling of autocorrelated traffic produces intervals that are too narrow.
- **McNemar** between model pairs, reporting **discordant pair counts and the odds ratio**, not only a
  p-value. At 82,000 samples a p-value declares everything significant.
- **Estimation floor.** ~37,000 benign test rows means FPR = 0.1% is 37 false positives; FPR = 0.01% is
  3.7 and **is not estimable**. `Worms` has 44 test rows, so its recall interval is roughly ±15 points.
  **No point estimates below the floor** — those get intervals.

## 7. Calibration

Isotonic and Platt compared, reported by Brier score and a reliability diagram. Calibration describes
**`p_attack` only** — `priority` is a sort key and `novelty_percentile` is a percentile, and neither is
a probability. Conflating them would be the same category error as calling a fused score calibrated.

## 8. Drift

- **PSI with fixed reference bin edges.** Recomputing bins per window is a common bug that makes PSI
  read near zero regardless of actual shift.
- PSI's 0.1 / 0.25 thresholds are a **convention** (Lewis 1994; Siddiqi), not a derived result, and are
  sample-size dependent. Used as a review trigger, not as law.
- **Multiplicity correction.** 42 features × KS per window yields ~2 false flags every window at
  α = 0.05, forever. Benjamini-Hochberg applied; observed-vs-expected flag counts reported.
- **Concept drift is undetectable without labels.** Unlabelled monitoring detects covariate shift
  (P(x)) and prediction shift (P(ŷ)). P(y|x) moving is not observable. In a SOC, labels arrive days
  late or never — which is precisely why unsupervised monitoring is the primary signal and delayed
  analyst verdicts are treated as confirmation.

## 9. Self-audit against Arp et al. (2022)

*Dos and Don'ts of Machine Learning in Computer Security*, USENIX Security 2022. We audit ourselves
against all ten pitfalls **and publish the ones we still fail.**

| # | Pitfall | Our position |
|---|---|---|
| P1 | Sampling bias | **Partially fails.** All three datasets are synthetic or simulated; none represents 2026 traffic. Stated in the model card as the headline caveat. Unfixable within this scope. |
| P2 | Label inaccuracy | **Mitigated.** CICIDS2017's original labels are known-wrong; we use the corrected re-release and cite the errata. NSL-KDD/UNSW labels are taken as given. |
| P3 | Data snooping | **Mitigated.** Preprocessing fitted on training folds only; novelty pipeline fitted on benign training rows only; de-dup before split; NSL-KDD splits never merged. Enforced by tests. |
| P4 | Spurious correlations | **Directly addressed** — this is §2, the artifact audit. |
| P5 | Biased parameter selection | **Mitigated.** Hyperparameters selected by CV on training data; the test split is touched once, at the end. |
| P6 | Inappropriate baseline | **Mitigated.** LogisticRegression is the floor and every claim is measured against it. |
| P7 | Inappropriate performance measures | **Directly addressed** — §1 and §6. |
| P8 | Base-rate fallacy | **Directly addressed** — §1, with the arithmetic shown. |
| P9 | Lab-only evaluation | **Partially fails.** Mitigated by the owned-hardware pcap capture, but that is a small lab, not a production network. Stated plainly. |
| P10 | Inappropriate threat model | **Addressed** in `THREAT_MODEL.md`, including attacks on Penumbra itself — the feedback-loop poisoning vector we introduced ourselves. |

Two of ten are partial failures, both for the same underlying reason: we do not have real production
traffic. Saying so is more useful than a table of ten green ticks.

## 10. Results

`[pending]` — populated by `penumbra eval --report`. Every figure is stamped with the git commit that
produced it, and the whole set regenerates with `penumbra reproduce-all`.

---

## References

Axelsson, *The Base-Rate Fallacy and the Difficulty of Intrusion Detection*, ACM TISSEC 2000 ·
Sommer & Paxson, *Outside the Closed World*, IEEE S&P 2010 · Arp et al., USENIX Security 2022 ·
Lewis 1994 / Siddiqi, *Intelligent Credit Scoring* (PSI) · Pierazzi et al., IEEE S&P 2020
