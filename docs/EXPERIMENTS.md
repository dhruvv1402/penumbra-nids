# Pre-registered experiments

**This file is committed before the experiments run.** Its git timestamp is the point. A hypothesis
written after seeing the result is not a hypothesis, and a project whose headline claim is "we measured
honestly" has to be able to prove it did not move the goalposts.

Nothing below will be edited once results exist. Results go in `docs/EVALUATION.md`, and where they
contradict what is written here, the contradiction is reported as the finding.

---

## E1 — Does a benign-only novelty head recover recall on attack families the supervised model never saw?

This is the project's central claim and the literal restatement of the problem brief.

### Hypothesis

A supervised classifier trained without family *F* will fail to flag *F* at test time. A novelty
detector trained on benign traffic only, which has no notion of families at all, will flag some of *F*
because *F* is statistically unlike benign traffic. Adding the novelty head therefore recovers
attack-vs-normal recall on unseen families **at no additional alert cost**.

### Design

**Leave-One-Attack-Family-Out (LOAFO).** For each of the 9 UNSW-NB15 attack families: remove every row
of that family from training, train both configurations, score the full test set, measure recall on the
held-out family's test rows.

Second, independent instance of the same question: **NSL-KDD's natural unseen-17.** `KDDTest+` contains
17 attack types absent from `KDDTrain+` (3,750 rows, 16.6% of the test set). No holdout construction by
us; the dataset authors built it.

### Metrics — in this order, and the first one is primary

1. **Binary attack-vs-normal recall on the held-out family, at a matched alert budget.** This answers
   "did the SOC get told." Primary.
2. **Family-attribution accuracy on the held-out family.** Expected to be approximately zero — you
   cannot name a class you have never seen. Reported separately because "we detected it, we could not
   name it" is the honest and more interesting result.
3. **`SUSPECTED_NOVEL` verdict rate on the held-out family.** How often the system correctly says
   "this is an attack I have no name for." The actual product claim.

### The matched-budget constraint

**Any comparison not made at a fixed alert budget is void.** OR-ing an additional detector into a
system always raises recall; a comparison that lets alert volume float measures nothing but the extra
alerts. So: fix alerts-per-1000-flows identically for `supervised-alone` and `supervised + novelty`,
then compare held-out recall.

`eval/budget.py` is built before `eval/loafo.py` for this reason.

### Why multiclass recall is *not* the primary metric

UNSW-NB15's category boundaries overlap heavily — Exploits, DoS, Fuzzers, Backdoor and Analysis are
not cleanly separable in feature space, which is one of the most replicated findings about the dataset.
If `Backdoor` is held out and the model labels those flows `Exploits`, **the SOC still received an
alert.** That is a detection success. Reporting near-zero multiclass recall and concluding "supervised
detection fails on novel attacks" would be a rigged comparison, and measuring it that way would make
the headline result an artifact of our own metric choice.

### Predicted outcome, recorded in advance

We expect a **mixed result**, and we expect the mix to be unflattering in a specific way:

- Novelty **adds measurable recall** on high-volume, statistically extreme families — DoS, Generic,
  Fuzzers, Reconnaissance. These are also the families the supervised head already catches. It helps
  most where it is needed least.
- Novelty **adds approximately nothing** on Worms, Shellcode, Analysis and Backdoor — low-volume
  families that do not look extreme at the flow-aggregate level. These are exactly the ones we want it
  for.
- Net expectation: positive macro-average ΔRecall at matched budget on roughly 4–6 of 9 families,
  ~0 on 2–3, **negative on 1–2**.

If that is what the data says, that is what gets presented. A matrix with mixed signs is a more
credible deliverable than a single flattering number, and Sommer & Paxson (2010) predicted this shape
of result fifteen years ago: machine learning is good at finding similarity and bad at finding novelty.

### Falsification

E1 is **refuted** if, at matched alert budget, mean ΔRecall across the 9 families is ≤ 0 with a
bootstrap 95% CI containing 0, **and** the NSL-KDD unseen-17 result shows the same.

If refuted, the headline becomes "an honest measurement of the limits of supervised intrusion
detection," the evaluation-rigour results (E3, E4, E5) carry the presentation, and this file is cited
as evidence that the negative result was predicted rather than discovered by accident.

### Schedule

Run as a **smoke test in Phase 1.5**, not at the end. NSL-KDD unseen-17 first — it has the most
headroom, since supervised binary recall on those rows is typically 0.15–0.45. We need the answer at
roughly hour 16, while there is still time to react to it.

---

## E2 — Does threshold-moving on a calibrated model match or beat resampling?

### Hypothesis

For class imbalance, tuning the decision threshold on a well-calibrated model performs at least as well
as SMOTE-family resampling, at a fraction of the cost — and resampling additionally *decalibrates* the
model, requiring recalibration afterwards.

### Design

Ablation over: `class_weight` · SMOTE · SMOTE-NC · ADASYN · BorderlineSMOTE · RandomUnderSampler ·
BalancedRandomForest · EasyEnsemble · focal loss · threshold-moving on isotonic-calibrated output.

**Every cell is evaluated on the natural distribution.** Resampling touches training folds only. A
test asserts that no sampler is ever fitted outside a CV fold.

SMOTE-NC rather than plain SMOTE where categoricals are present: `proto`, `service` and `state` are
categorical, and interpolating between one-hot columns produces rows that cannot exist.

### Secondary result we intend to publish either way

A **deliberate leakage demonstration**: the same pipeline with SMOTE applied before the train/test
split versus inside the folds. We expect the leaky version to report dramatically better minority-class
F1. Publishing both numbers side by side is the clearest possible statement of why the discipline
matters.

We also intend to show a single synthetic minority row generated by SMOTE on flow data — with
fractional packet counts, and byte totals inconsistent with those counts — as the physical argument
against resampling network flows at all.

---

## E3 — How much of our headline performance is a dataset artifact?

### Hypothesis

A substantial fraction of reported UNSW-NB15 performance is attributable to testbed artifacts rather
than learned attack behaviour. Specifically, `sttl` alone reaches high binary AUC because attack and
benign traffic were generated from different hosts and time-to-live records the generator, not the
behaviour.

### Design

Fit a single-feature decision stump per column; rank by AUC. Any feature with solo AUC > 0.90 is
quarantined as a suspected artifact. **Every headline number is then reported twice — with and without
the quarantined set.**

Suspects declared in advance: `sttl`, `ct_state_ttl`, `is_sm_ips_ports`, `dttl` on UNSW-NB15;
`Destination Port` on CICIDS2017.

### Predicted outcome

`sttl` solo AUC ≈ 0.90+. Full-model binary ROC-AUC ≈ 0.98 with artifacts, ≈ 0.90–0.93 without.

**The second number is the one we stand behind and the one that goes on the slide.**

### Why this runs first

It runs in Phase 0, before any model is trained, because it constrains what we are permitted to claim
afterwards.

---

## E4 — Negative controls

Three checks that exist to catch a pipeline bug masquerading as a result:

1. **Shuffled labels.** Train on permuted labels; performance must collapse to chance. If it does not,
   there is leakage in the pipeline.
2. **Row index only.** Train on `id`/row position alone; must be chance. Detects ordering leaks.
3. **Train/test duplicate overlap.** Hash rounded feature vectors and count exact and near-duplicate
   rows crossing the split boundary. Report both raw and de-duplicated scores.

Whatever the overlap count turns out to be, it is a finding and it is published.

---

## E5 — Does conformal coverage degrade measurably under injected drift?

### Hypothesis

Split-conformal coverage guarantees hold under exchangeability. Concept drift violates exchangeability.
Therefore empirical coverage should fall below its nominal level as drift increases — making coverage
loss usable as a *label-free drift signal*.

### Design

Mondrian (class-conditional) conformal predictor. Replay the test stream through the drift injector
while shifting feature distributions and introducing an unseen family mid-stream. Plot empirical
coverage against PSI over time.

### Predicted outcome

Coverage tracks nominal while PSI < 0.1 and falls below it as PSI crosses 0.25.

If it does not, we report that conformal coverage is *not* a usable drift signal on this data, which is
also a result.

---

## E6 — Does sequence context buy recall that per-flow features cannot?

**Registered before the run. Nothing below was written after seeing a number.**

### Hypothesis

A per-flow classifier is given one connection and asked whether it looks unusual. For a whole class
of attacks that is the wrong unit: a slow port sweep contains no unusual connection, only an
unusual *sequence* of ordinary ones. So:

> **H6.** Adding per-host temporal context raises recall at a matched false-positive budget on
> CICIDS2017, and the gain concentrates in the families whose signature is distributional rather
> than per-flow — `PortScan`, the slow DoS variants (`DoS Slowloris`, `DoS Slowhttptest`),
> `FTP-Patator` / `SSH-Patator`.

And the question we actually care about, because it decides whether the deep model earns its place:

> **H6b.** Most of that gain is available from **nine cheap causal entity-graph features** and does
> not require a deep sequence model. Trees beat deep nets on tabular data of this size and shape
> (Grinsztajn et al., NeurIPS 2022); the open question is only whether the *temporal* axis is the
> exception.

### Design

Three arms, scored on **exactly the same rows**, at a **matched benign-flag budget** (1% FPR):

| arm | what it sees |
|---|---|
| `per-flow` | CICIDS2017's own features. The floor. |
| `per-flow + graph` | plus 9 causal entity-graph features (fan-out, fan-in, port entropy, pair count, inter-arrival) |
| `sequence` | 1D-CNN → BiGRU over a K=16 causal window of the source host's preceding flows |

- **CICIDS2017 only.** It is the one dataset here with `Src IP`, `Dst IP` and `Timestamp`. UNSW's
  published split has none of them, and its `ct_*` columns are already entity-window aggregates
  computed by the original pipeline — a graph head there would recompute what is in the data.
- **Temporal split throughout**: train Mon–Wed, test Thu–Fri. The sequence head's own validation
  split is the tail of train, not a random slice.
- **The same rows.** The window builder drops unparseable timestamps and reorders by time, so all
  three arms are evaluated on its index. Comparing the deep arm on its convenient subset against
  the floor on everything would be a different dataset, not a different model.
- **Matched budget, on benign rows.** "Higher recall" is empty if it also alerts more. Matching on
  total alert count would cap recall at the prevalence — the bug already measured in
  EVALUATION.md §10.4.
- **Causality is tested, not asserted.** Appending future traffic must leave every earlier row's
  features bit-identical (`tests/unit/test_sequence.py`). A forward-reading window scores better
  and cannot be deployed, and nothing crashes when it happens.

### Predicted outcome, recorded in advance

1. `per-flow + graph` beats `per-flow` by a **small but real** margin — call it +0.01 to +0.05
   recall at 1% FPR — concentrated almost entirely in `PortScan`.
2. `sequence` lands **within noise of `per-flow + graph`**, and may well come in below it. If so,
   that is the finding: the temporal signal on this dataset is captured by nine aggregate features,
   and the deep model is not paying for itself.
3. CICIDS2017's per-flow features are already strong (this is the dataset where `Destination Port`
   alone nearly separates the classes, which is why it is dropped), so the headroom for any arm is
   small. A large gain would be more suspicious than a small one.

### Falsification

H6 is refuted if neither context arm clears `per-flow` by more than 0.01 recall at the matched
budget. H6b is refuted if `sequence` beats `per-flow + graph` by more than 0.02. **Both refutations
get published.** The value of this experiment does not depend on the deep model winning — it
depends on the comparison being fair, and the fairness is in the matched rows and matched budget.

---

## Standing rules for all experiments

- **Prevalence is stated with every precision-family number.** UNSW-NB15's test set is ~55% attack;
  real networks run 1e-3 to 1e-5. Precision, PPV and PR-AUC all move with base rate. Primary metrics
  are the prevalence-invariant pair, TPR and FPR. PR-AUC is always printed alongside the prevalence it
  was computed at and its no-skill baseline (which equals the prevalence).
- **Estimation floor.** With 37,000 benign test rows, FPR = 0.1% is 37 false positives and FPR = 0.01%
  is 3.7 — not estimable. `Worms` has 44 test rows, so its recall interval is roughly ±15 points. No
  point estimate is given below the floor; those get intervals.
- **Bootstrap CIs are stratified by class**, or rare classes vanish from resamples. Temporal data uses
  a moving-block bootstrap, not iid.
- **McNemar reports discordant pair counts and the odds ratio**, not only a p-value — at 82,000 samples
  a p-value calls everything significant.
- **Drift tests are multiplicity-corrected.** 42 features × KS per window means ~2 false flags every
  window at α = 0.05. Benjamini-Hochberg, or report observed-versus-expected flag counts.
- **Seeds are fixed and recorded**; every figure is stamped with the git commit that produced it.
