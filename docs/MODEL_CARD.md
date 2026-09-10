# Model Card — Penumbra network intrusion detector

Structured per Mitchell et al., *Model Cards for Model Reporting* (FAT\* 2019).

> **Status: pre-results.** Every section below is complete except the numbers, which are marked
> `[pending]` and will be filled from `penumbra eval --report` rather than typed by hand. Sections that
> state limitations are already final — they do not depend on how the model performs.

---

## 1. Model details

| | |
|---|---|
| Name | Penumbra |
| Version | 0.1.0 (pre-results) |
| Type | Two-stage: supervised multiclass classifier + benign-only novelty detector |
| Architecture | Known-threat head: LogisticRegression (floor), RandomForest, XGBoost. Novelty head: 42-32-16-8-16-32-42 MLP autoencoder, IsolationForest, Mahalanobis with Ledoit-Wolf shrinkage. Fusion: rank-normalised, single threshold |
| Calibration | Isotonic on a held-out split; Platt compared |
| Uncertainty | Mondrian (class-conditional) split-conformal prediction |
| Owner | Dhruv Gupta |
| Licence | MIT |
| Repository | https://github.com/dhruvv1402/penumbra-nids |

### Training data vintage — read this before quoting any number

**NSL-KDD descends from a 1998/99 DARPA simulation. UNSW-NB15 is synthetic IXIA PerfectStorm traffic
from 2015. CICIDS2017 is from 2017, and its original labels were wrong enough that a corrected
re-release exists.**

These datasets are between 9 and 28 years old and none of them are real production traffic. The
absolute numbers this model produces **do not transfer to a 2026 enterprise network.** What transfers
is the relative comparison between methods and the evaluation methodology.

We state this first rather than in a footnote because it is the single most important caveat on the
entire artifact.

---

## 2. Intended use

**Primary use.** Generating prioritised alerts for human SOC analyst triage, in two lanes: a
high-precision known-threat incident queue, and a fixed-budget hunting queue for statistically unusual
traffic.

**Intended users.** Tier-1 and tier-2 SOC analysts. Output is designed to be actionable without ML
expertise — feature attributions are rendered in plain English, and novelty is expressed as a
percentile against normal traffic rather than as a reconstruction error.

**Explicitly out of scope:**

- **Automated blocking or traffic denial of any kind.** There is no blocking code path in the
  repository and a CI test fails the build if one appears (ADR-0001).
- Automated firewall or ACL rule insertion.
- Sole basis for account suspension, disciplinary action, or any decision about a person.
- Attribution of attacks to individuals or organisations.
- Detection inside encrypted payloads — the model sees flow metadata only, never packet contents.
- Compliance evidence or legal proceedings.

---

## 3. Factors

Performance is reported disaggregated by, not only in aggregate over:

- **Attack family** — the primary breakdown. Per-family recall varies by an order of magnitude.
- **Protocol** (TCP/UDP/ICMP) and **service**.
- **Traffic volume regime** — short/low-byte flows behave differently from bulk transfers.
- **Network segment**, where the dataset distinguishes one.
- **Flow duration regime** — relevant to slow-and-low evasion.

Reported using the same framing as the Azure ML Responsible AI dashboard's **Error Analysis**
component, run locally via the open-source `erroranalysis` package.

---

## 4. Metrics

**Primary metrics are prevalence-invariant: TPR (recall) and FPR**, plus ROC-AUC and recall at fixed
FPR. This is deliberate. UNSW-NB15's test set is ~55% attack while real networks run 1e-3 to 1e-5, so
precision, PPV and PR-AUC computed on this data are inflated by two to four orders of magnitude
relative to deployment.

PR-AUC is reported **always annotated with the prevalence it was computed at and its no-skill baseline**
(which equals the prevalence). "PR-AUC 0.97" alone is not a meaningful statement; "PR-AUC 0.97 at
π=0.55, no-skill 0.55" is.

Also reported: per-class precision/recall/F1 (macro **and** weighted), full confusion matrix,
Brier score with reliability diagram, Precision@k for the hunting lane, and stratified bootstrap 95%
confidence intervals on every headline number.

**Estimation floor.** With ~37,000 benign test rows, FPR = 0.1% is 37 false positives and FPR = 0.01%
is 3.7 — not estimable. `Worms` has 44 test rows, giving a recall interval of roughly ±15 points.
**No point estimate is published below the floor.** Those get intervals.

**Operational translation.** `eval/prevalence.py` re-weights measured TPR/FPR to a stated deployment
prevalence and emits PPV, alerts/day and analyst-hours/day. Those outputs are labelled **modelled, not
measured**, because they are.

| Metric | Value |
|---|---|
| Binary ROC-AUC (with suspected artifact features) | `[pending]` |
| **Binary ROC-AUC (artifact features removed)** | `[pending]` — **this is the number we stand behind** |
| Recall @ 1% FPR | `[pending]` |
| Macro-F1 across 10 classes | `[pending]` |
| Brier score, pre/post isotonic | `[pending]` |
| Per-family recall matrix | `[pending]` |
| LOAFO ΔRecall at matched alert budget | `[pending]` |

---

## 5. Evaluation data

| Dataset | Split | Rationale |
|---|---|---|
| UNSW-NB15 | the authors' published 175,341 / 82,332 split | comparability with published results |
| NSL-KDD | `KDDTrain+` / `KDDTest+`, **never re-shuffled** | preserves the 17 attack types present only in test — 3,750 rows, 16.6% of test — which is the only naturally occurring zero-day holdout available |
| CICIDS2017 | day-based temporal: train Mon–Wed, test Thu–Fri | random splitting leaks future into past on inherently temporal data |

De-duplication is performed **before** splitting. Train/test duplicate overlap is measured and reported
(`data/audit.py`); on CICIDS2017 roughly 20% of rows are exact duplicates, and if they straddle the
split boundary the test score is partly memorisation.

---

## 6. Training data

Class distribution (UNSW-NB15 training split):

| Family | n | | Family | n |
|---|---|---|---|---|
| Normal | 56,000 | | Reconnaissance | 10,491 |
| Generic | 40,000 | | Analysis | 2,000 |
| Exploits | 33,393 | | Backdoor | 1,746 |
| Fuzzers | 18,184 | | Shellcode | 1,133 |
| DoS | 12,264 | | **Worms** | **130** |

**Imbalance handling, stated precisely because this is where results become fiction:**

Resampling is applied **inside cross-validation folds, on training data only, never before the split**.
A test fails the build if a sampler is ever fitted outside a fold. SMOTE-NC rather than plain SMOTE
where categoricals are present, because interpolating between one-hot columns generates rows that
cannot physically exist.

**Resampling decalibrates a model.** Since we report calibrated probabilities, recalibration on a
held-out set is performed *after* any resampling. This is easy to miss and invalidates the calibration
claim if missed.

The full ablation — `class_weight`, SMOTE, SMOTE-NC, ADASYN, BorderlineSMOTE, RandomUnderSampler,
BalancedRandomForest, EasyEnsemble, focal loss, threshold-moving — is evaluated **on the natural
distribution** in every cell.

**Suspected testbed artifacts.** `data/audit.py` fits a single-feature stump per column and quarantines
anything with solo AUC > 0.90. On UNSW-NB15 we expect `sttl` to qualify: attack and benign traffic were
generated from different hosts, so time-to-live records the generator rather than the behaviour. Every
headline number is reported with and without the quarantined set.

---

## 7. Quantitative analyses

`[pending]` — per-family recall, disaggregation across the factors in §3, the LOAFO matrix, calibration
curves, and the with/without-artifact comparison.

Expected in advance and published regardless: **U2R and R2L recall on NSL-KDD will be poor** (52 U2R
instances in test). That is reported with an explanation rather than omitted.

---

## 8. Ethical considerations

**The base-rate fallacy is the dominant risk, and it is arithmetic, not opinion.** At 1,000,000 flows
per day and a prevalence of 1e-4 (100 genuine attack flows), a 1% false positive rate produces ~10,000
false alerts against ~100 true ones — a positive predictive value near 1%. This is why signature IDSs
still exist and why the novelty lane is budget-capped rather than ticket-generating (Axelsson, 2000).

**Dual use.** A detector's feature attributions describe what makes traffic detectable, which is also a
map of what an evader would need to change. Mitigated by keeping attributions inside an authenticated,
role-gated, audited interface.

**Disparate impact.** A model keyed on subnet, device class or department can systematically
over-flag one team's traffic — research groups, contractors, or anyone whose normal work looks unusual.
Mitigated by disaggregated reporting (§3), and it is a genuine residual risk rather than a solved one.

**Privacy.** IP addresses are personal data (GDPR Art. 4; *Breyer*, CJEU C-582/14). Penumbra
pseudonymises them with a keyed HMAC — **pseudonymisation, not anonymisation**: IPv4 is 2³² values and
trivially enumerable by anyone holding the key, so the output remains personal data. No packet payloads
are inspected.

**Adversarial evasion.** Detection degrades under deliberate evasion. Measured with *problem-space*
constraints (Pierazzi et al., IEEE S&P 2020) rather than meaningless feature-space perturbations —
an attacker can add padding and delay, but cannot un-send a packet.

**Poisoning through the feedback loop.** The analyst-verdict path is an attack surface we introduced
ourselves. See `docs/THREAT_MODEL.md`.

---

## 9. Caveats and recommendations

- **A human decides.** EU AI Act Article 14 requires high-risk AI — Annex III includes critical
  infrastructure — to be designed so a natural person can oversee, interpret, override and halt it.
  Penumbra alerts; it never blocks.
- **Retrain on a drift trigger, not a calendar.** PSI ≥ 0.25 on any monitored feature, or a sustained
  shift in the prediction distribution. Policy in `docs/RUNBOOK.md`.
- **PSI's 0.1/0.25 thresholds are a convention** (Lewis 1994; Siddiqi), not a derived result, and they
  are sample-size dependent. Treated as a trigger for review, not as law.
- **True concept drift is undetectable without labels.** Without ground truth you can detect covariate
  shift and prediction shift. P(y|x) moving is not observable. Labels in a SOC arrive days late or
  never, which is why unsupervised monitoring is the primary signal.
- **Conformal coverage assumes exchangeability**, which drift violates by definition. We state the
  caveat and then measure coverage degrading as PSI rises.
- **Known failure modes:** encrypted traffic, tunnelling, low-and-slow attacks below the flow-aggregate
  signal, insider misuse of legitimate credentials, and any attack family whose network signature
  resembles benign traffic.
- **Do not deploy on a network whose traffic mix differs materially from the training data** without
  re-evaluating. Given §1, that means essentially every real network.

---

## References

Mitchell et al., *Model Cards for Model Reporting*, FAT\* 2019 · Gebru et al., *Datasheets for
Datasets*, CACM 2021 · Axelsson, *The Base-Rate Fallacy and the Difficulty of Intrusion Detection*,
2000 · Sommer & Paxson, *Outside the Closed World*, IEEE S&P 2010 · Arp et al., *Dos and Don'ts of
Machine Learning in Computer Security*, USENIX Security 2022 · Pierazzi et al., *Intriguing Properties
of Adversarial ML Attacks in the Problem Space*, IEEE S&P 2020 · Grinsztajn et al., *Why do tree-based
models still outperform deep learning on tabular data?*, NeurIPS 2022
