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

### 10.5 Reproduction

```bash
uv run penumbra audit --dataset unsw        # artifact + leak audit
uv run penumbra eval  --dataset unsw        # binary + per-family, with/without artifacts
uv run penumbra loafo --dataset nslkdd      # the unseen-17 experiment
```

Raw outputs: `artifacts/reports/{audit,eval}_{unsw,nslkdd}.json`,
`artifacts/reports/unseen17_nslkdd.json`, `artifacts/reports/loafo_unsw.json`.

---

## References

Axelsson, *The Base-Rate Fallacy and the Difficulty of Intrusion Detection*, ACM TISSEC 2000 ·
Sommer & Paxson, *Outside the Closed World*, IEEE S&P 2010 · Arp et al., USENIX Security 2022 ·
Lewis 1994 / Siddiqi, *Intelligent Credit Scoring* (PSI) · Pierazzi et al., IEEE S&P 2020
