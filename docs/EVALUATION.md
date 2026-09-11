# Evaluation methodology

> **Status: results for Phase 1 and the NSL-KDD unseen-17 experiment are in.** The methodology was
> fixed before any model was trained. Numbers come from `penumbra eval`, not from hand-editing.
> Hypotheses are pre-registered in [`EXPERIMENTS.md`](EXPERIMENTS.md), committed before these runs.

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

### Result: the prediction was wrong, and the way it was wrong is the interesting part

**The solo-AUC test found nothing.** No feature on either dataset reached 0.90. `sttl` — the feature
we named in advance as the likely artifact — came 17th of 42 at **0.7603**.

That looked like a clean bill of health. It was not. Inspecting the crosstab directly:

```
sttl == 31   →  22.5% of training rows,  100.0% benign
dttl == 29   →  22.5% of training rows,  100.0% benign
proto == 'unas' →  6.9% of training rows, 100.0% attack
is_sm_ips_ports == 1 → 1.6% of training rows, 100.0% benign
```

A single TTL value identifies a fifth of the benign class perfectly. **Solo AUC could not see it**,
because AUC is a *ranking* metric: `sttl=31` separates a large subpopulation cleanly while the
feature's other values rank poorly, and the two effects cancel in the aggregate.

So a second detector was added — `leaky_values`, which looks for feature values with high class
purity and non-trivial support — and it finds all four decisively. **One artifact detector is not
enough, and a negative result from one is not evidence of absence.**

### Result: artifact vs signal is a semantic question, not a statistical one

The leaky-value detector also fires on NSL-KDD: `srv_serror_rate == 1.0` is 99.1% attack across
27.7% of training rows. But a SYN flood *is* a 100% SYN-error condition — that is the detection
working, not a leak. Quarantining it would delete real capability to improve a statistic.

The test we apply: **is there a causal path from attack behaviour to this feature value?** A flood
causes a SYN error rate of 1.0. Nothing about an attack causes a packet's initial TTL to be 31 —
that is the host that generated it. Verdicts and their reasoning are recorded per feature in
`schema.ARTIFACT_VERDICTS`; the detector only surfaces candidates.

Quarantined on UNSW-NB15: `sttl`, `dttl`, `proto`, `is_sm_ips_ports`. Quarantined on NSL-KDD:
**nothing** — all eleven candidates were judged genuine behaviour.

### Result: the trees route around the artifact

| model | ROC-AUC with artifacts | artifacts removed | Δ | recall @1% FPR, with | without |
|---|---|---|---|---|---|
| logreg | 0.9556 | 0.9276 | −0.0280 | 0.7369 | **0.5176** |
| **rf** | **0.9844** | **0.9833** | **−0.0011** | 0.8561 | 0.8302 |
| xgb | 0.9827 | 0.9817 | −0.0010 | 0.8580 | 0.8532 |

*(UNSW-NB15, 95% CIs in `artifacts/reports/eval_unsw.json`; RF's AUC interval is ±0.0006.)*

**E3 predicted 0.98 → 0.90–0.93. That is refuted for the tree models: RF loses 0.0011.** The
artifacts are real — `sttl=31` is not a coincidence — but the information is redundantly encoded
across the other 38 features, and an ensemble simply routes around the removal.

Two things follow, and both are more useful than the prediction would have been:

1. **Removing an artifact is not sufficient to make an evaluation honest.** The correlated
   information survives. Reporting "we dropped the leaky column" is weaker evidence than teams
   generally assume.
2. **The linear model tells a different story.** LogReg loses 0.028 AUC and **22 points of recall at
   1% FPR** (0.737 → 0.518). A model without the capacity to route around the artifact shows how
   much was resting on it. The AUC delta understates the dependence; the operating-point delta
   reveals it.

## 3. Negative controls

| Control | Expected | UNSW-NB15 | NSL-KDD |
|---|---|---|---|
| Shuffled labels | ~0.50 | **0.5022** | **0.5021** |
| Row index only | ~0.50 | **0.5001** | **0.5001** |

Both pass on both datasets. The pipeline cannot learn from noise or from row position.

### Result: train/test duplicate overlap is substantial and was not documented anywhere we looked

| | UNSW-NB15 | NSL-KDD |
|---|---|---|
| duplicate rows within train | **74,301** (42.4%) | 16 |
| duplicate rows within test | **28,386** (34.5%) | 57 |
| test rows also present in train | **8,541 (10.4% of test)** | 664 (2.9%) |

NSL-KDD is clean — de-duplication is the entire reason it exists as a refinement of KDD Cup '99.

**UNSW-NB15's published split is not.** Over 10% of its test rows appear verbatim in training, so
that fraction of the test score is memorisation rather than generalisation. We report it because it
is a property of the benchmark that anyone quoting a UNSW-NB15 number is inheriting whether they
know it or not.

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

### Result: every SMOTE variant performs worse than doing nothing

UNSW-NB15, RandomForest, all eleven strategies evaluated on the **natural** test distribution
(prevalence 0.5506). Ranked by **recall at 1% FPR** — the operationally meaningful column, because
it is where a SOC actually runs.

| strategy | ROC-AUC | recall @1% FPR | minority recall † | train rows after resampling | fit time |
|---|---|---|---|---|---|
| undersample | 0.9846 | **0.8613** | 1.0000 | 112,000 | 20 s |
| class_weight | 0.9844 | 0.8561 | 1.0000 | 175,341 | 31 s |
| balanced_rf | 0.9839 | 0.8501 | 1.0000 | 175,341 | 29 s |
| **none** | 0.9828 | **0.8488** | 1.0000 | 175,341 | 27 s |
| **threshold_moving** | 0.9828 | **0.8488** | 0.8636 | 175,341 | 28 s |
| smotenc | 0.9821 | 0.8284 | 1.0000 | 238,682 | **426 s** |
| smote_tomek | 0.9816 | 0.8280 | 1.0000 | 228,086 | 268 s |
| smote | 0.9824 | 0.8258 | 1.0000 | 238,682 | 59 s |
| borderline_smote | 0.9816 | 0.8251 | 1.0000 | 238,682 | 85 s |
| adasyn | 0.9822 | 0.8237 | 1.0000 | 238,470 | 88 s |
| easy_ensemble | 0.9559 | 0.6572 | 0.9773 | 175,341 | 74 s |

† `Worms`, 44 test rows. **Below the estimation floor** — a 1.0000 here means 44 of 44, which is not
evidence of discrimination. It is in the table for completeness and should not be read as a result.

**All five SMOTE-family strategies land below the no-handling baseline.** SMOTE-NC spent 426 seconds
and 63,000 synthetic rows to arrive 2 points *worse* than doing nothing at all. E2 predicted
threshold-moving would match or beat resampling; the data says something stronger — on this dataset
resampling is not merely unnecessary, it is actively counterproductive.

The mechanism is in §10.2's companion result: SMOTE interpolates in feature space with no notion of
which quantities are integral or which are derived from others, so the synthetic rows it adds are
off the data manifold. The model spends capacity learning a region of feature space that no real
flow occupies.

#### Why `none` and `threshold_moving` are identical in that column

They are the same fitted model. Recall-at-fixed-FPR is read off the ROC curve, so it cannot
distinguish two configurations that differ only in where the threshold sits.

The difference is in what each one *does at its operating point*:

| | recall | FPR |
|---|---|---|
| none, at the default 0.5 | 0.9885 | **0.2728** |
| threshold_moving, targeting 1% FPR | 0.8488 | **0.0100** |

Same model, same training, no synthetic data. **Moving the threshold took FPR from 27% to 1%** for
14 points of recall. Nothing in the resampling column achieves anything comparable, and the
untuned default is the configuration a SOC could least afford to run.

*Reproduce: `uv run penumbra ablate --dataset unsw`. Raw output in
`artifacts/reports/ablation_unsw.json`.*

### Result: leaking SMOTE inflates minority F1 by 8×

Target: **UNSW-NB15 `Worms` vs everything else** — 130 positive rows, prevalence **0.074%**.
Identical model, identical data, identical resampler. The only difference is *where* the resampling
happens relative to the split.

| | F1 | PR-AUC | ROC-AUC |
|---|---|---|---|
| correct — SMOTE inside the fold | **0.1231** | 0.2276 | 0.9916 |
| leaky — SMOTE before the split | **0.9986** | 0.9999 | 1.0000 |
| inflation | **+0.8756** | +0.7722 | +0.0084 |

**The leaky pipeline does not fail. It reports a near-perfect number.** That is why this mistake
survives review: there is no error, no warning, and no symptom other than a result good enough to
publish. SMOTE synthesises minority rows by interpolating between neighbours, so applied before the
split a validation row can be interpolated from its own neighbours — the model has effectively seen
it.

Note also that ROC-AUC barely moves (+0.008) while F1 moves by 0.876. **A leak this severe is
invisible in the metric most papers lead with**, and obvious in the one that matters at this
prevalence.

#### A second objection, independent of leakage

One synthetic `Worms` row SMOTE generated, verbatim:

```
spkts       38.3251     dpkts       33.8113
sloss       12.7300     dloss        6.8244
ct_dst_ltm   3.6081
```

A flow with 38.3 packets sent is not a rare flow. It is not a flow. SMOTE interpolates in feature
space with no notion of which quantities are integral, which are derived from others (`rate`,
`sload` and `smean` are functions of `dur`, `sbytes` and `spkts`), or which combinations are
physically realisable.

This is the argument for **threshold-moving over resampling on network data**: it changes where the
boundary is drawn rather than manufacturing traffic that could not exist.

*Reproduce: `uv run penumbra ablate --leakage-demo`. Raw output in
`artifacts/reports/smote_leakage.json`.*

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

### 10.1 Binary detection

| dataset | model | ROC-AUC (95% CI) | TPR | FPR | recall @1% FPR | macro-F1 |
|---|---|---|---|---|---|---|
| UNSW-NB15 | rf | 0.9833 [0.9828, 0.9839] | 0.9795 | 0.2105 | 0.8302 | 0.5086 |
| UNSW-NB15 | xgb | 0.9817 [0.9811, 0.9824] | 0.9681 | 0.2001 | 0.8532 | — |
| UNSW-NB15 | logreg | 0.9276 [0.9262, 0.9291] | 0.9134 | 0.2983 | 0.5176 | — |
| NSL-KDD | xgb | 0.9685 | 0.6655 | 0.0282 | 0.3818 | 0.5474 |
| NSL-KDD | rf | 0.9668 | 0.6170 | 0.0264 | 0.4644 | — |
| NSL-KDD | logreg | 0.8433 [0.8383, 0.8483] | 0.6256 | 0.0763 | 0.3614 | — |

*UNSW-NB15 figures are the artifact-removed run.* PR-AUC is reported in the JSON with its
prevalence attached; it is omitted here because quoting it without that context is exactly the
practice §1 objects to.

The FPR at threshold 0.5 is high (0.20–0.30 on UNSW) because class weighting pushes the decision
boundary toward recall. That is why the threshold-0.5 column is not the headline: **recall at 1% FPR
is the operationally meaningful number**, and the cost-based operating point supersedes both.

McNemar, artifact-free UNSW run: RF beats LogReg on 8,049 discordant pairs against 1,807 (odds 4.45).
RF vs XGB is 1,598 against 1,729 (odds 0.92, p = 0.024) — statistically distinguishable at 82,332
samples, operationally a coin flip. This is exactly why the discordant counts are reported rather
than the p-value alone.

**The linear floor is cleared.** RF beats LogReg by 0.056 AUC on UNSW and 0.125 on NSL-KDD, so the
ensemble has earned its complexity. That was not guaranteed.

### 10.2 Per-family — where the honest bad news lives

**UNSW-NB15** (xgb, multiclass, macro-F1 **0.5086**):

| family | support | recall | | family | support | recall |
|---|---|---|---|---|---|---|
| Generic | 18,871 | 0.9723 | | Fuzzers | 6,062 | 0.5551 |
| Reconnaissance | 3,496 | 0.8112 | | DoS | 4,089 | **0.1250** |
| Exploits | 11,132 | 0.8213 | | Backdoor | 583 | **0.0926** |
| Shellcode | 378 | 0.7778 | | Analysis | 677 | **0.0901** |
| Normal | 37,000 | 0.7541 | | Worms | 44 | 0.4318 † |

† 44 test rows. Interval only; no point estimate is defensible at that support.

**NSL-KDD** (xgb, multiclass, macro-F1 **0.5474**): dos 0.8435, probe 0.6270, **r2l 0.0589**,
**u2r 0.1343** (67 rows †).

Macro-F1 in the 0.51–0.55 band is consistent with the 0.55–0.75 predicted in `EXPERIMENTS.md` at
its lower edge. DoS at 0.125 recall on UNSW is the notable failure and is a family-confusion
artifact: DoS flows are being absorbed into Exploits, whose precision (0.62) is correspondingly
depressed. This is the boundary-overlap problem that makes multiclass recall the wrong primary
metric for LOAFO — the SOC still receives an alert for those flows.

**NSL-KDD r2l at 0.0589 recall is the single most informative number in this table.** Eight of the
seventeen attack types that appear only in the test set are r2l. The model is not failing to
classify r2l; it is failing to classify attacks it was never shown. That is the problem this project
exists to address, visible in a per-class recall column.

### 10.3 The thesis experiment — NSL-KDD's natural unseen-attack split

3,750 test rows (16.6% of `KDDTest+`) belong to 17 attack types that never appear in `KDDTrain+`.

**The answer is a curve, not a number.** A single operating point invites the objection that it was
chosen to flatter one configuration; the curve shows where the technique works and where it does
not. Thresholds here are placed so the realised FPR hits each target exactly, isolating the
scientific question from the separate deployment problem in §10.4.

| realised FPR | supervised | + novelty | Δ | detections novelty found alone |
|---|---|---|---|---|
| 0.12% | 0.0453 | 0.0512 | +0.0059 | 24 |
| 0.51% | 0.0515 | 0.1536 | **+0.1021** | 388 |
| 1.01% | 0.0525 | 0.3464 | **+0.2939** | 1,106 |
| **2.01%** | **0.0848** | **0.5405** | **+0.4557** | **1,830** |
| 5.03% | 0.6437 | 0.7520 | +0.1083 | 2,321 |
| 10.01% | 0.8640 | 0.7693 | **−0.0947** | 470 |

Three things this says, and the third is the one worth arguing about:

**The supervised head fails on attacks it has never seen.** At 1% FPR it recalls 0.0525 of the
unseen types. Its recall on attack types it *was* trained on, at the same threshold, is an order of
magnitude higher. That gap is the failure mode this project was built around, measured rather than
asserted.

**The novelty head recovers a large part of it, in a band.** +0.29 at 1% FPR, +0.46 at 2%. It does
this while flagging the same number of benign rows, because the OR cost is charged: two heads at a
1% total budget each run at ~0.5%.

**Outside that band it does nothing, or harm.** Below ~0.2% FPR neither head has room to fire. Above
~5% the supervised head is already loose enough to catch most things, and splitting the budget costs
more than novelty returns — at 10% FPR the fused configuration is **worse** by 9.5 points. A novelty
head is not a free addition; it is a trade that pays off only where the supervised head is starved.

At the per-type level the picture is equally uneven. `mscan`, `apache2`, `processtable` and
`httptunnel` are recovered substantially; `snmpgetattack` and `worm` stay at zero regardless of
threshold. The types it misses are as informative as the ones it catches.

### 10.3b LOAFO on UNSW-NB15 — a negative result about the *experiment*, not the method

The same protocol on UNSW-NB15, holding out each of the nine attack families in turn, matched at 2%
FPR:

| held-out family | train rows removed | test n | supervised | + novelty | Δ | attribution |
|---|---|---|---|---|---|---|
| Backdoor | 1,746 | 583 | **1.0000** | 1.0000 | +0.0000 | 0.0000 |
| Worms | 130 | 44 | **1.0000** | 1.0000 | +0.0000 | 0.0000 |
| DoS | 12,264 | 4,089 | 0.9985 | 0.9985 | +0.0000 | 0.0000 |
| Generic | 40,000 | 18,871 | 0.9999 | 0.9998 | −0.0001 | 0.0000 |
| Analysis | 2,000 | 677 | 0.9970 | 0.9882 | −0.0089 | 0.0000 |
| Reconnaissance | 10,491 | 3,496 | 0.9928 | 0.9857 | −0.0072 | 0.0000 |
| Exploits | 33,393 | 11,132 | 0.9874 | 0.9788 | −0.0086 | 0.0000 |
| Shellcode | 1,133 | 378 | 0.9947 | 0.9894 | −0.0053 | 0.0000 |
| Fuzzers | 18,184 | 6,062 | 0.6485 | 0.5826 | −0.0658 | 0.0000 |

**mean Δ −0.0107 — the novelty head adds nothing.** But read the `supervised` column first.

**Delete every `Backdoor` row from training and the model still detects 100% of Backdoor flows at
test time.** Same for `Worms`. Eight of nine families sit above 0.98.

That is not a model succeeding at zero-day detection. It means **the held-out families were never
novel.** UNSW-NB15's categories overlap so heavily in feature space that a flow labelled `Backdoor`
is still recognisable as *an attack* from what `Exploits`, `DoS` and `Generic` taught the model. The
label was removed; the behaviour was still represented.

So: **leave-one-class-out does not simulate a zero-day when the classes overlap.** It simulates
removing a *name*, not a *mechanism*. A novelty head cannot add recall on top of 1.0000, and its
small negative deltas are just the OR-cost being charged for coverage that was not needed.

This is why NSL-KDD's natural split (§10.3) is the load-bearing experiment and UNSW's LOAFO matrix is
reported as a control. The contrast is the evidence:

| | supervised recall on held-out attacks |
|---|---|
| UNSW-NB15, synthetic family holdout | **0.99–1.00** — not actually unseen |
| NSL-KDD, 17 genuinely unseen types | **0.05** — actually unseen |

An experiment where the baseline already scores 1.0000 cannot measure an improvement. Had we run
only the UNSW matrix, the honest conclusion would have been "novelty detection does not help" — and
it would have been an artifact of a holdout that held nothing out.

The `attribution` column is 0.0000 for every family, exactly as pre-registered: the model cannot
name a class it has never seen, even while detecting it perfectly. *"We detected it, we could not
name it"* is the finding, and it is why binary recall rather than multiclass recall is E1's primary
metric.

### 10.4 Three methodological corrections, recorded because each changed the answer

These are in the results section rather than hidden in a commit log because two of them produced
publishable-looking numbers that were wrong, and the sequence is the actual finding.

**(a) Matching on alert count instead of false positives.** The first run capped total alerts at 50
per 1,000 flows and gave Δ = +0.0003 — flat. But these test sets are 55–57% attack, so that cap is
consumed by true positives and limits recall to ~9% before any detector speaks. A real network is
>99% benign, so operational alert volume is essentially *false-positive* volume. Matching on benign
rows flagged is both prevalence-invariant and the actual driver of analyst workload.

**(b) Fusing incommensurable scores.** The second run used `max(p_attack, novelty_percentile)` and
gave Δ = **+0.3560** — a flattering number produced by a bug. On benign rows `p_attack` has
p99 = 0.907 while the novelty percentile is uniform by construction with p99 = 0.996, so the max
inherited a 0.996 threshold and **discarded every supervised detection between the two**. The same
error, run on UNSW-NB15, gave mean Δ = −0.36 with eight of nine families worse.

The irony is instructive: `NoveltyEnsemble` already rank-normalises its three detectors against each
other, for precisely this reason. The reasoning was simply never carried across the
supervised/novelty boundary.

Rank-normalising both heads then failed a third way — **2.41% of benign rows score above the entire
reference set**, so their rank is exactly 1.0, the 99th percentile is 1.0, and every configuration
admits the identical tied block. Supervised and fused came out digit-for-digit equal on every attack
type, which is what prompted looking again.

The working design thresholds each head on its own raw scale and charges the OR cost explicitly.

**(c) Test-set peeking in the operating point.** Deriving a threshold from test-set benign rows is
not deployable — at inference time there are no labels to derive it from — even when every
configuration gets the same advantage. Thresholds are now fitted on held-out benign data and applied
blind.

**That change surfaced a genuine finding.** Targeting 1% FPR from training-benign traffic realises
**10.17%** on test benign. The operating point does not transfer, because `KDDTest+`'s benign traffic
is distributed differently from `KDDTrain+`'s — and 10% is precisely the band where fusion *hurts*.
An operating point chosen honestly on training data lands in the one region where the technique
fails. That is a drift problem, it is measurable before deployment, and it is the strongest
available argument for the drift monitoring in §8.

### 10.5 Alert-to-incident correlation — measured, on real source IPs

This is the anti-alert-fatigue number, and it is reported here rather than alongside the UNSW
results for a specific reason: **UNSW-NB15 has no IP addresses**, so any compression ratio computed
there would be derived from identifiers we invented. `Correlator.correlate` raises rather than
grouping on synthesised entities, and the repository's `/stats` endpoint refuses to divide alerts by
unrelated incidents.

CICIDS2017, 240,000 test flows from the Thursday–Friday temporal split, grouped by
(source IP, family, 15-minute window):

| | |
|---|---|
| alerts emitted | **30,780** |
| incidents presented | **273** |
| events per incident | **112.7** |

The largest single incident: **a DoS Hulk flood from one source — 29,562 flows, one destination, one
port — arrives as one thing to look at.** Without correlation that is 29,562 tickets for one event
that a human understands in ten seconds.

Grouping is by `(entity, family)` rather than by entity alone, deliberately: a host running a port
scan *and* exfiltrating data is two incidents with different responses, and merging them hides the
second behind the first. Incidents are ranked by priority and then by event count, because two
incidents at equal priority are not equally urgent when one represents 29,562 flows and the other
represents one.

*Reproduce: see `artifacts/reports/correlation_cicids.json`.*

### 10.6 Conformal coverage as a label-free drift signal

Split conformal gives a finite-sample coverage guarantee — **under exchangeability**. Drift violates
exchangeability by definition, so a project claiming "guaranteed 95% coverage" on one slide and
"we monitor drift" on the next has contradicted itself.

We state the caveat and then measure it. Mondrian (class-conditional) conformal at α = 0.10,
calibrated on held-out *training* rows, evaluated under three conditions:

| condition | score PSI | empirical coverage | gap vs nominal | abstention |
|---|---|---|---|---|
| **A** — exchangeable (held-out train) | 0.0036 | **89.8%** | −0.2% | 10.2% |
| **B** — natural shift (`KDDTest+`) | 0.6532 | **59.5%** | −30.5% | 39.6% |
| **C** — injected covariate drift | 1.9326 | **29.7%** | −60.3% | 69.9% |

**A is the control and it works**: 89.8% against 90.0% nominal, per-class, on data drawn from the
calibration distribution. The method is correctly implemented.

**B is not an injected condition.** It is NSL-KDD's own train/test split, and coverage collapses by
30 points on it. `KDDTest+` is deliberately not exchangeable with `KDDTrain+` — 17 attack types
appear only in test — so the conformal predictor is *correctly reporting that its own assumption
does not hold on this data*. This independently corroborates §10.4(c): an operating point fitted on
training benign realises 10.2% FPR on test rather than the targeted 1%. Two different methods,
measuring the same shift, agreeing.

**C escalates it further**, as it should.

#### Why this matters operationally

Coverage needs labels to compute, so it is not directly available in production. But look at the
abstention column: **10.2% → 39.6% → 69.9%**, tracking coverage monotonically. Abstention rate needs
no labels at all.

So the deployable signal is: *the fraction of flows on which the model declines to commit*. When it
climbs, exchangeability is breaking, and that is observable in real time — in the regime where
ground truth arrives days late or never.

Coverage loss is the offline validation; abstention rate is the operational read of the same thing.

#### What this buys the analyst

Ambiguous and empty prediction sets route to the `UNCERTAIN` verdict and the review lane. That is
human-in-the-loop with a stated error rate attached rather than a gesture — and an empty set (the
row is atypical of *both* calibration classes) is a natural companion to the novelty head rather
than a failure mode.

*Reproduce: `artifacts/reports/conformal_coverage.json`.*

### 10.7 Throughput

670 flows/second sustained on 8 CPU cores, single process, scoring both heads.

**That is flows per second, not link speed, and the two are not interchangeable** — a flow record
summarises many packets, so converting one to the other requires an assumption about mean flow size
that we would rather state than bury. At CICIDS2017's observed mean flow size this corresponds very
roughly to a few hundred Mbps of *monitored* traffic, and that figure should be treated as an
order-of-magnitude sanity check rather than a capacity claim.

The first measurement was 38 flows/s. The cause was `_importances()` rebuilding and re-sorting the
model's global feature-importance dictionary once per scored row — a quantity that depends on the
fitted model and not on the row. Caching it gave an 18× speedup with no change to output.

### 10.7b Mined detection rules — the model writes signatures for the SIEM

The brief's premise is that a signature IDS misses novel attacks. The usual answer is to replace
it. This is the other direction: extract the high-purity decision paths out of the forest and emit
them as KQL and Sigma, so the existing SIEM enforces detections nobody has shipped a rule for.

A path from root to a pure leaf already *is* a rule. `ct_dst_sport_ltm > 3.5 and sload > 3.49e7` is
a conjunction any SIEM can evaluate with no model, no Python and no GPU.

**Three splits, and the separation is the point.** The forest is grown on a `fit` split. Every
precision number attached to a rule comes from a `holdout` carved out of *training* data the trees
never saw — not the test set, because then the rules and the model's own reported metrics would
share a denominator and the rule pack would stop being independent evidence. The test set is
touched once, at the end, and changes no threshold and discards no rule.

#### Result: half the mined rules rested on a feature our own audit had quarantined

| UNSW-NB15 | with artifacts | quarantined |
|---|---:|---:|
| unique decision paths mined | 207 | 101 |
| discarded as artifact-dependent | 0 | **106** |
| survived held-out validation (≥ 0.98 precision) | 193 | **95** |
| set recall, holdout (61,370 rows) | 0.6628 | 0.6470 |
| set precision, holdout | 0.9924 | 0.9925 |
| set recall, **test** (82,332 rows) | 0.6786 | **0.6626** |
| set precision, **test** | 0.9635 | **0.9657** |

`sttl`, `dttl`, `ct_state_ttl`, `is_sm_ips_ports` and the value `proto='unas'` are quarantined —
the verdicts and their reasoning are in §2. 106 of 207 mined paths depend on one of them. Every one
of those 106 would have validated at high precision on held-out data from the same testbed and
detected nothing on a real network, because what they encode is which machine generated the packet.

**Held-out validation does not catch a testbed artifact.** That is the finding. The 193-rule pack
and the 95-rule pack were validated identically, on the same rows, against the same floor. Only the
artifact audit separates them, and it runs on the data rather than on the model.

The cost of the quarantine is **1.6 points of set recall** (0.6786 → 0.6626) and half the rules.
The artifact-dependent rules were almost entirely redundant: the behavioural rules already covered
the same attacks. Test precision is fractionally *better* without them.

**The headline we stand behind: 95 rules, no model, 66.3% of attacks on the UNSW test set at 0.966
precision.** Quoted at the test set's own prevalence (see §1) — precision on a real network would
be far lower, and the prevalence-invariant reading is the relevant one.

#### The transfer drop is visible in both datasets, and it is larger on NSL-KDD

Nothing is quarantined on NSL-KDD: every leaky value the audit found has a verdict of `signal`
(§2). A SYN flood genuinely produces a SYN error rate of 1.0 — a rule resting on that is the
detection working, not a leak. So there is one arm, and 317 of 361 mined paths survive.

| | holdout | test | Δ |
|---|---:|---:|---:|
| UNSW set recall | 0.6470 | 0.6626 | +0.016 |
| UNSW set precision | 0.9925 | 0.9657 | **−0.027** |
| NSL-KDD set recall | 0.9823 | **0.7312** | **−0.251** |
| NSL-KDD set precision | 0.9880 | 0.9140 | **−0.074** |

NSL-KDD's rule set loses **25 points of recall** between a held-out slice of training data and the
official test set, against UNSW's +1.6. The difference is the dataset: `KDDTest+` contains 17
attack types absent from `KDDTrain+` (§10.3), and a conjunction of thresholds mined from a family
it has never seen does not fire on that family. This is the thesis experiment showing up in the
rule pack, and it is the honest limit of rule mining — mined rules are excellent at compressing
known behaviour into something a SIEM can run, and they are not a novelty detector. That is what
the second head is for.

#### Sigma is emitted only where it can be honest

**1 of 95 UNSW rules and 7 of 317 NSL-KDD rules are expressible as Sigma.** Sigma's taxonomy is
log-based — process creation, firewall, DNS, web — and has no field for `sload`, `ct_srv_src`,
`sinpkt` or `tcprtt`. Emitting `sload > 1400000` as a Sigma rule produces something that looks
portable and evaluates nowhere, so `to_sigma()` returns `None` rather than a partial translation,
and the skip count is reported alongside the emitted count. "We emitted 7 Sigma rules" and "we
emitted 7 of 317 because Sigma cannot express the other 310" are different claims.

KQL against a flow table can express all of it, which is why KQL is the primary target.

#### Two bugs in this pipeline, both worth recording

**76% of the first run's rules were artifact-dependent, and the first count of them was wrong.**
The exclusion was applied before de-duplication, so the same path rediscovered by 40 trees counted
40 times: it reported "110 excluded" against "101 kept" — two numbers in different units. De-duping
first makes them add up (106 + 101 = 207). A test now asserts that identity.

**The Sigma emitter indented detection keys as siblings of `selection` rather than children.** The
document parsed as valid YAML and described a different rule. No test caught it because the run
that would have exercised it produced zero Sigma-expressible rules — the emitter was only reached
once categorical conditions were supported. Categoricals now reach the forest as indicators named
`proto=tcp`, which round-trip back to `proto == "tcp"` in KQL rather than surfacing as a threshold
on an anonymous one-hot index, and that is what made the first real Sigma document appear.

Reproduce: `uv run penumbra rules --dataset unsw`. Outputs:
`artifacts/reports/mined_rules_{unsw,nslkdd}.json`,
`sentinel/Analytic Rules/PenumbraMinedRules_{unsw,nslkdd}.kql`, `sentinel/Sigma/{unsw,nslkdd}/`.

---

### 10.7c Sequence context — the deep model, and why it loses (E6)

Pre-registered in `docs/EXPERIMENTS.md` §E6 and committed before the run. Three arms on **identical
rows** at a **matched benign-flag budget**, on CICIDS2017 — the only dataset here with source
addresses and timestamps.

| arm | ROC-AUC | recall @ 1% FPR | Δ | realised FPR |
|---|---:|---:|---:|---:|
| per-flow (RF) | 0.9976 | **0.9959** | — | 0.0100 |
| per-flow + entity graph | **0.9989** | 0.9867 | −0.0092 | 0.0100 |
| sequence (1D-CNN → BiGRU) | 0.9582 | 0.6084 | −0.3874 | 0.0100 |

297,586 train / 303,211 test windows, K=16. **H6 refuted, H6b confirmed.** Neither context arm
clears per-flow. The nine cheap graph features beat the deep model by **0.378 recall** at the same
budget.

Most of the explanation is that there was nothing to win: CICIDS2017's per-flow features already
reach 0.9959 recall at 1% FPR. We predicted that in advance, and we also predicted a small gain for
the graph arm that turned out to be a small loss.

#### AUC and the operating point disagree, and both are right

The graph arm has the **highest ROC-AUC** and **lower recall at the threshold we would deploy**.
ROC-AUC averages ranking quality over every threshold; recall at 1% FPR is one threshold. A model
can rank better overall and be worse at the point you actually use.

This is §1's argument made concrete. If we reported AUC alone, the graph arm wins. If we report
recall at a fixed FPR — the number a SOC actually buys on — it does not.

#### The sequence head trained fine and generalised badly

Validation ROC-AUC **0.9968**, no collapse, early-stopped at 4 epochs. Test ROC-AUC **0.9582**.

| family | per-flow | + graph | sequence | n |
|---|---:|---:|---:|---:|
| Portscan | 0.9998 | 1.0000 | **0.4237** | 53,002 |
| DDoS | 1.0000 | 1.0000 | 1.0000 | 31,717 |
| Infiltration – Portscan | 0.9920 | 0.9482 | 0.5518 | 23,955 |
| Botnet | 0.8436 | 0.8548 | **0.0555** | 1,605 |
| Web Attack – Brute Force | 0.9977 | 1.0000 | **0.0655** | 443 |
| Web Attack – XSS | 1.0000 | 1.0000 | **0.0317** | 221 |

Training is Mon–Wed (dominated by `DoS Hulk`); test is Thu–Fri (Portscan, Botnet, Web attacks). The
sequence head learned Wednesday's temporal shapes. `DDoS` — the one family whose shape carries
across the boundary — it detects perfectly. Everything else it largely misses.

> **A model that learns temporal shape overfits the temporal shapes in its training window. A
> per-flow model has no temporal shape to overfit, so it does not have that failure mode.**

A random split would have hidden this completely: Wednesday's sequences would appear on both sides
and this arm would look excellent. It is the same argument as §4's temporal-split rule, arriving
from a different direction.

#### Two silent failures on the way here

Recorded because neither raised an exception and both produced a plausible table.

**A quantile does not deliver a matched budget.** Forest probabilities put thousands of rows at
exactly 0.0, so the quantile lands inside the tied block. Run one matched a 1% budget and realised
**0.2% on one arm and 1.0% on another**, reporting both as 1%. Rank selection
(`eval.budget.flags_at_benign_budget`) is exact; realised FPR is now printed beside the budget in
every table above so the match is checkable.

**A collapsed training run reported a number.** Run two hit validation AUC 0.9999 in epoch one then
exactly 0.500 for every epoch after — a constant output. Early stopping restored the epoch-one
weights and the run exited cleanly. The first diagnosis was exploding feature values (CICIDS reaches
7.2e9, with maxima over 200σ from their own mean), and that fix — median/IQR scaling clipped to 10
robust deviations, plus gradient clipping — was worth making but was **not the cause**.

The cause: **the last 15% of Mon–Wed in time order has an attack rate of 0.0001** — four attacks in
44,638 windows, because Wednesday evening is quiet once the DoS traffic stops. Keras reports
ROC-AUC 0.5 on an effectively single-class validation set, and early stopping was monitoring exactly
that. The temporal tail is now widened until the rarer class clears 500 rows, falling back to a
stratified shuffle only if nothing up to half the data works, and **which rule was used is recorded
on every run** — a random split answers a weaker question and must not be substituted silently.

---

### 10.7d Constrained adversarial evasion

`adversarial/evasion.py` measures detection decay under perturbations an attacker could actually
transmit: pad bytes up, stretch duration, widen inter-packet gaps, never un-send a packet, keep
counts integral, and recompute every derived feature from its primitives (Pierazzi et al., IEEE S&P
2020). The unconstrained feature-space attack runs on the same rows so the gap is measured rather
than asserted.

The x-axis is **attacker effort in units of slowdown**, not a dimensionless epsilon, because effort
here has a price: a ten-times slower scan takes ten times as long to finish. "Detection falls from
0.97 to 0.61 when the attacker accepts a 10× slowdown" is a sentence a defender can act on.

#### Result: slow-rate mimicry does not evade this detector, and we do not claim robustness from that

20,000 UNSW attack flows, detector threshold 0.5359 (fitted on held-out benign, never on these rows).

| effort | × slower | detection, realisable | detection, unconstrained | × slower, unconstrained |
|---:|---:|---:|---:|---:|
| 0.0 | 1.00 | 0.9797 | 0.9797 | 1.00 |
| 1.0 | 2.00 | 0.9852 | 0.9957 | 0.99 |
| 4.0 | 5.02 | 0.9937 | 0.9976 | 0.95 |
| 9.0 | 10.03 | **0.9943** | **0.3186** | **0.88** |

**Neither curve decays.** Under the realisable attack detection *rises* at all five steps; under the
unconstrained attack it rises at four of five and then collapses at the last.

That is not robustness and the tool refuses to report it as such. A perturbation that increases
detection much more likely means the perturbed rows have left the region the model was fitted on,
and a tree ensemble's prediction in an extrapolation region is whichever leaf the path happens to
reach — not a judgment about the traffic. The summary prints that caveat whenever the curve is
non-monotone, which here is always.

The honest claim is therefore narrow: **slow-rate mimicry, as implemented within the problem-space
constraints, does not evade this model on this dataset.** We cannot claim the model is robust,
because we cannot show the perturbed rows are still rows the model reasons about meaningfully.

#### The two effort axes are not a common currency

Look at the last column. The unconstrained attack's flows come out at **0.88× their original
duration — faster**, which no amount of padding or added delay can achieve. It is not paying a price
at all.

So "at equal effort" is a sentence this table cannot support, and an earlier draft of the summary
said it. Realisable effort is a slowdown the attacker actually suffers. Unconstrained effort is a
fraction of the distance to the benign centroid; at effort 9 that fraction is 0.9.

Which gives a stronger finding than "constrained attacks are weaker":

> The unconstrained attack's evasion is bought by moving the row **90% of the way to the benign
> mean**. A row that is mostly benign has not evaded detection — it has stopped being the attack.

That is what a reported "68% evasion rate" can mean, and it is why an evasion number without a
constraint model is not a security measurement.

#### Per-family, and the one genuinely interesting row

| family | baseline | attacked | Δ | n |
|---|---:|---:|---:|---:|
| Fuzzers | 0.6538 | 0.9110 | **+0.2572** | 1,011 |
| Exploits | 0.9909 | 1.0000 | +0.0091 | 5,723 |
| Reconnaissance | 0.9994 | 0.9873 | **−0.0121** | 1,737 |
| DoS · Analysis · Backdoor · Worms | 1.0000 | 1.0000 | 0.0000 | 3,142 |

`Reconnaissance` is the only family the realisable attack helps at all, and it buys 1.2 points for a
tenfold slowdown — a terrible trade for an attacker whose scan now takes ten times as long.

`Fuzzers` moves the other way by 26 points. Fuzzers are the family the model is *worst* at (0.65
baseline, consistent with §10.2), and stretching them apparently pushes them out of whatever region
made them ambiguous. Again: that is extrapolation behaviour, not evidence of anything defensive.

#### A trap we fell into first

The strawman, written as symmetric Gaussian noise, barely moved detection at all — which would have
made the *unconstrained* attack look weaker than the realisable one and inverted the entire finding.
It interpolates toward the benign centroid now. An unconstrained attack has to actually be an attack
for the comparison to mean anything.

Reproduce: `penumbra fit -d unsw && penumbra adversarial -d unsw`. Output:
`artifacts/reports/adversarial_unsw.json`, and the console's evaluation page.

---

### 10.8 Reproduction

```bash
uv run penumbra audit --dataset unsw        # artifact + leak audit
uv run penumbra eval  --dataset unsw        # binary + per-family, with/without artifacts
uv run penumbra loafo --dataset nslkdd      # the unseen-17 experiment
uv run penumbra rules --dataset unsw        # mine + validate KQL/Sigma rules
uv run penumbra sequence                    # E6: does sequence context buy recall?
uv run penumbra reproduce-all               # everything, with reasons for what it skips
```

Raw outputs: `artifacts/reports/{audit,eval}_{unsw,nslkdd}.json`,
`artifacts/reports/unseen17_nslkdd.json`, `artifacts/reports/loafo_unsw.json`.

---

## References

Axelsson, *The Base-Rate Fallacy and the Difficulty of Intrusion Detection*, ACM TISSEC 2000 ·
Sommer & Paxson, *Outside the Closed World*, IEEE S&P 2010 · Arp et al., USENIX Security 2022 ·
Lewis 1994 / Siddiqi, *Intelligent Credit Scoring* (PSI) · Pierazzi et al., IEEE S&P 2020
