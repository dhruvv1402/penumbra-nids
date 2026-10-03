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

### RESULT — H6 refuted, H6b confirmed

Recorded after the run. The predictions above were committed first and are not edited.

| arm | ROC-AUC | recall @ 1% FPR | Δ vs per-flow | alerts |
|---|---:|---:|---:|---:|
| per-flow | 0.9976 | **0.9959** | — | 112,440 |
| per-flow + graph | **0.9989** | 0.9867 | −0.0092 | 111,422 |
| sequence (CNN+BiGRU) | 0.9582 | 0.6084 | **−0.3874** | 69,443 |

297,586 train / 303,211 test windows, identical rows, realised FPR 0.0100 on all three arms.

**H6 is refuted.** Neither context arm clears per-flow by more than 0.01 recall at the matched
budget. Adding per-host temporal context did not buy recall on this dataset; it cost some.

**H6b is confirmed, and by a wider margin than predicted.** The graph arm beats the sequence arm by
**0.378 recall** at the same budget. Nine causal aggregate features — fan-out, fan-in, port entropy,
pair count, inter-arrival — are not merely competitive with the deep model here, they dominate it.

**Prediction 1 was wrong.** We predicted the graph arm would clear per-flow by +0.01 to +0.05,
concentrated in `PortScan`. It came in at −0.0092. Recording it: a small predicted gain became a
small measured loss.

**Prediction 3 was right, and it is most of the explanation.** CICIDS2017's per-flow features leave
almost no headroom — 0.9959 recall at 1% FPR before any context is added. There was nothing for the
extra features to win.

#### The dissociation worth noticing

The graph arm has the **highest ROC-AUC of the three (0.9989 vs 0.9976)** and slightly **lower**
recall at the operating point. Those are not in conflict: AUC averages ranking quality over every
threshold, and recall at 1% FPR is one threshold. A model can rank better overall and be worse at
the single point you actually deploy. This is the concrete version of the argument in §1 for
reporting recall at a fixed FPR rather than AUC alone.

#### Why the sequence head loses, and it is not the architecture

It trained cleanly: validation ROC-AUC **0.9968**, no collapse, early-stopped at 4 epochs. Then it
scored **0.9582** on test. The per-flow forest scored 0.9976 on the same rows.

That gap is a temporal generalisation failure, and the per-family table says where:

| family | per-flow | + graph | sequence | n |
|---|---:|---:|---:|---:|
| Portscan | 0.9998 | 1.0000 | **0.4237** | 53,002 |
| DDoS | 1.0000 | 1.0000 | 1.0000 | 31,717 |
| Infiltration – Portscan | 0.9920 | 0.9482 | 0.5518 | 23,955 |
| Botnet | 0.8436 | 0.8548 | **0.0555** | 1,605 |
| Web Attack – Brute Force | 0.9977 | 1.0000 | **0.0655** | 443 |
| Web Attack – XSS | 1.0000 | 1.0000 | **0.0317** | 221 |
| Infiltration | 0.9600 | 1.0000 | 0.0400 | 25 |

Training is Mon–Wed, which is dominated by `DoS Hulk`. Test is Thu–Fri: Portscan, Botnet, Web
attacks. The sequence head learned Wednesday's *sequence shapes* and they did not transfer;
`DDoS` — the one family whose temporal signature is shared across the boundary — it detects
perfectly.

So the honest conclusion is narrower than "deep learning does not work here":

> **A model that learns temporal shape overfits the temporal shapes present in training, and a
> per-flow model does not have that failure mode because it has no temporal shape to overfit.**

That is a real cost of sequence modelling and the kind of thing a random split would have hidden
entirely — on a shuffled split, Wednesday's sequences appear in both train and test, and this arm
would have looked excellent.

#### Process note: three runs, two of them broken

Only the third run is reported. The first two are recorded here because the failures were silent.

1. **Run one** matched the FPR budget with a quantile. Forest probabilities put thousands of rows
   at exactly 0.0, so the quantile landed inside the tied block: one arm realised 0.2% FPR while
   another realised 1.0%, both printed as "1%". Fixed with rank selection
   (`budget.flags_at_benign_budget`). Realised FPR is now printed beside the budget so the match is
   checkable by eye.
2. **Run two** trained a sequence head that had collapsed. Validation AUC 0.9999 in epoch one, then
   exactly 0.500 for every epoch after; early stopping restored the epoch-one weights and the run
   exited cleanly with a number attached. The first diagnosis — exploding feature values — was
   wrong. The actual cause: **the last 15% of Mon–Wed in time order has an attack rate of 0.0001**,
   four attacks in 44,638 windows, because Wednesday evening is quiet after the DoS traffic stops.
   Keras reports ROC-AUC 0.5 on a single-class validation set and early stopping monitored exactly
   that. The temporal tail is now widened until the rarer class clears 500 rows, and which rule was
   used is recorded on every run.

Neither failure raised an exception. Both produced a plausible table.

---

## E7 — Can one analyst account poison the model through the feedback loop, and do our controls notice?

**Registered before the run. Nothing below was written after seeing a number.**

The human-in-the-loop is the one attack surface this project created itself (THREAT_MODEL T1). An
attacker whose traffic is being detected, and who holds one analyst account, records
`false_positive` on their own alerts. If those labels are promoted, the next model learns the attack
is benign. The controls are: a second senior account must promote every verdict (procedural, not
measurable offline), integrity flags on pending verdicts, and a canary gate on the retrained model.
This experiment measures the last two, and the damage they are meant to prevent.

### Hypotheses

> **H7a — the attack works, and it works where support is thin.** Retraining on flipped verdicts
> lowers recall on the targeted attack type in proportion to the dose, because the model has little
> honest evidence about that type to outvote the flipped labels.

> **H7b — the canary gate refuses the poisoned model and accepts the honest one.** A targeted
> poisoning attack is a per-family regression by construction; an aggregate gate would miss it, a
> per-family gate should not.

> **H7c — the integrity flags notice the attacker, and the cheapest flag is the noisiest.**

### Design

- **NSL-KDD.** `KDDTrain+` fits the champion. `KDDTest+` is split three ways, stratified by attack
  type with a fixed seed: **canary** 30% (frozen trusted labels; the gate decides on these and
  nothing else), **feedback** 35% (live traffic: the champion alerts, analysts record verdicts),
  **evaluation** 35% (every reported number). No reported number comes from the rows that chose a
  model.
- **Only alerted rows get verdicts.** An analyst never sees what did not fire, and the attacker can
  only poison a detection that fires — which is exactly their situation.
- **Target, by a rule fixed now:** the attack type with the **smallest training support** among
  those the champion alerts on at least 50 times in the feedback pool. We have not looked at which
  type that selects; we expect `warezmaster` (20 training rows) or `guess_passwd` (53).
- **Four accounts**, round-robin: three honest analysts and `mallory`, who also does her share of
  honest triage as cover. On top of that she marks a fraction *d* of the target's alerts
  `false_positive`, for *d* ∈ {0.10, 0.25, 0.50, 1.00}.
- **Arms:** champion; **honest** challenger (all verdicts truthful, *d* = 0); four poisoned
  challengers. Every challenger is `KDDTrain+` plus the verdict-labelled feedback rows, fitted with
  the same full two-head pipeline — so flipped rows also enter the **benign-only** training set of
  the novelty head.
- **Gate** (`eval/canary.py`), each at the model's own deployed threshold: G1 overall recall may not
  drop > 0.02; G2 no attack type with ≥ 20 canary rows may lose > 0.10 recall; G3 benign FPR may
  not rise > 0.01.
- **Flags** (`feedback/integrity.py`): `confident_contradiction` (clears a detection scored
  ≥ 0.90), `actor_outlier` (clearance rate ≥ 3 SE above leave-one-out peers), `family_campaign`
  (≥ 80% of an account's clearances on one predicted family). The analyst sees the *predicted*
  family, not ground truth, so that is what the flags see too.

### Predicted outcome, recorded in advance

1. **Honest feedback helps.** The honest challenger passes the gate, and its target recall on the
   evaluation split is at least the champion's — more labelled examples of a thin type.
2. **Dose-response.** Target recall falls as *d* rises. At *d* = 1.0 it falls by **≥ 0.30** absolute
   against the champion.
3. **The gate has a floor.** It fails the 50% and 100% arms on **G2, naming the target type**, and
   passes the 10% arm, whose damage we expect to be below the 0.10 tolerance. A gate that cannot see
   a small attack is still worth having; we will report where the floor sits rather than implying
   there is none.
4. **Flags.** At *d* = 1.0, **≥ 90%** of flipped verdicts carry at least one flag, driven by
   `family_campaign` and `actor_outlier`. `confident_contradiction` will **also** fire on a large
   share (> 30%) of *honest* clearances: `KDDTest+` is shifted, so the model is often confidently
   wrong about benign traffic. It is a review cue, not a detector, and the numbers should show why.
5. **The novelty head does not rescue the poisoned model.** Flipped rows are labelled benign, so
   they also enter the novelty head's benign reference. We expect the novelty-only share of target
   detections not to rise enough to compensate. Both heads are exposed to the same poisoned label.

### Falsification

H7a is refuted if the 100% arm loses less than 0.10 target recall — the attack would then be
ineffective against this model, and that is the finding. H7b is refuted if the honest challenger
fails the gate (a gate that blocks honest improvement gets switched off), or if any poisoned arm
that lost ≥ 0.10 target recall on the evaluation split passes. H7c is refuted if fewer than half
the flipped verdicts at *d* = 1.0 are flagged. **Every refutation gets published.**

---

### RESULT — H7a, H7b, H7c all survive; two of five predictions were wrong

Recorded after the run. The predictions above were committed first (`f653ee9`) and are not edited.

The rule selected **`warezmaster`**: 20 training rows, 269 champion alerts in the feedback pool, 331
rows in the evaluation split. KDDTest+ split 6,764 canary / 7,893 feedback / 7,887 evaluation; the
champion alerted on 4,028 feedback rows and every one of them received a verdict.

| arm | flipped | target recall [95% CI] | overall recall | FPR | gate | flipped verdicts flagged |
|---|---:|---|---:|---:|---|---:|
| champion | — | 0.843 [0.800, 0.878] | 0.8311 | 0.1018 | — | — |
| honest | 0 | **0.979** [0.957, 0.990] | **0.9352** | **0.0432** | pass | — |
| poisoned 10% | 27 | 0.979 [0.957, 0.990] | 0.9358 | 0.0438 | pass | **0%** |
| poisoned 25% | 67 | 0.970 [0.945, 0.984] | 0.9107 | 0.0374 | pass | 100% |
| poisoned 50% | 134 | 0.952 [0.923, 0.970] | 0.9100 | 0.0382 | pass | 100% |
| poisoned 100% | 269 | **0.193** [0.154, 0.239] | 0.8859 | 0.0435 | **FAIL** (G2) | 100% |

Evaluation-split numbers throughout; the gate saw only the canary. Two runs, identical to the
fourth decimal.

**Prediction 1 — honest feedback helps: confirmed, and by more than expected.** Target recall
0.843 → 0.979, overall 0.831 → 0.935, and FPR falls from 10.2% to 4.3%: the `false_positive`
verdicts teach the model what this network's benign traffic looks like. Read the size with care.
The feedback pool is drawn from the same shifted `KDDTest+` distribution as the evaluation split, so
this is the loop doing domain adaptation, which is its job, but the magnitude reflects how far
`KDDTest+` sits from `KDDTrain+` (§10.4c). It does not transfer as a number.

**Prediction 2 — dose-response: confirmed, but the shape is a cliff, not a slope.** 0.979, 0.970,
0.952, then **0.193** at 100%, a loss of 0.65 against the champion. Flipping a quarter or half of the
target's verdicts barely registers, because the *remaining honest* `true_positive` verdicts on the
same attack type outvote the flipped ones. The attack only works if the attacker clears
essentially **every** alert of their own traffic. That is what makes it visible (prediction 4).

**Prediction 3 — the gate fails 50% and 100%: wrong on 50%.** It failed only the 100% arm, on G2,
naming `warezmaster` (canary recall 0.802 → 0.233 on 283 rows). The 50% arm passed, but it did no
damage *relative to the champion*: 0.952 against 0.843. It lost 0.027 relative to the honest
retrain, and no deployment has that counterfactual to compare against. **This is a structural limit
of champion-relative gating: poisoning that rides alongside honest improvement is hidden by it.**
Here that coincides with poisoning that does little harm. We have not shown it always will.

**Prediction 4 — flags: confirmed on the headline, wrong on the mechanism.**
- ≥ 90% of flipped verdicts flagged at *d* = 1.0: **100%**. Also 100% at 25% and 50%.
- **All of it came from `actor_outlier`**, which never fired on any of the three honest accounts in
  any arm. Mallory's clearance rate rose from 9.2% to 28.1% against a peer rate of 8.4–8.9%
  (z = 0.9 at 10%, 5.0 at 25%, 11.4 at 50%, 23.2 at 100%).
- **`family_campaign` never fired.** Predicted to drive detection; it contributed nothing. Mallory's
  honest cover work adds benign clearances of her own, and at 100% even a single predicted family
  across all 269 flips would cap her share at 269 / ≈340 ≈ 79%, just under the 80% threshold. Cover
  work is cheap and defeats a fixed concentration threshold.
- **`confident_contradiction` flagged 0 of the 269 flipped verdicts** and 22.9% of the honest
  accounts' clearances (predicted > 30%; it was lower). The attack succeeds on the attack type the
  model knows least about, so the model is never confident about it, so this flag cannot fire.
  **The cheapest flag is blind in exactly the case that matters.**
- At 10% nothing was flagged: 27 flips moved her clearance rate by less than noise. That dose also
  did no measurable damage. **In this drill, detection and damage have the same floor.**

**Prediction 5 — the novelty head does not rescue the poisoned model: confirmed.** At 100%, 80.7%
of `warezmaster` rows go undetected. The novelty-only share rises from 3.6% (champion) to 12.4%,
because the flipped rows also enter the benign-only reference the novelty head is fitted on.
Both heads trained on the same poisoned label.

#### What this changes

- The outlier check is the control that earns its place. It is also the easiest to evade by
  spreading flips across accounts, which is what the two-person rule and account security are for.
  The drill models one compromised account, not a coordinated insider group.
- `family_campaign` should be judged relative to peers, not against an absolute 80%. It is kept
  as registered here, and any revision will be reported as a revision.
- `confident_contradiction` stays as a review cue for honest mistakes. It is not a poisoning
  control, and the review screen should not present it as one.

*Reproduce: `penumbra poison-drill` (≈ 13 minutes, six full two-head fits) →
`artifacts/reports/poisoning_nslkdd.json`.*

#### Addendum — two revisions to `family_campaign`, made AFTER seeing the result

Everything above this line is the pre-registered flag and its result, unedited. This addendum is
post hoc, and its numbers are **in-sample**: the revised flag was designed on the same drill it is
scored on. It is labelled `family_skew` and runs *alongside* the registered flag rather than
replacing it, so the registered result stays reproducible.

**Revision 1: compare per family against peers, not against a fixed 80%.** "You clear r2l alerts
at 45%; your colleagues clear r2l alerts at 3%." Cover work on *other* families cannot dilute that
comparison. Result: **9% of flips flagged.** Better than zero, and still nearly blind.

**Why so low:** 244 of the 269 targeted `warezmaster` alerts carried **no predicted family**. The
family classifier said `normal`, and the alert fired on the novelty or abstention path instead. The
flag skipped family-less verdicts, so it could see at most 25 of 269.

**Revision 2: "the model could not name it" is a bucket of its own.** The analyst sees an alert
with no family, so the flag groups those like any other family. Peers clear family-less alerts 23%
of the time; the attacker cleared hers 100% of the time.

| dose | flipped | `actor_outlier` | `family_skew` (rev. 2) | honest-account clearances flagged by `family_skew` |
|---|---:|---:|---:|---:|
| 10% | 27 | 0% | 0% | 0% |
| 25% | 67 | 100% | 87% | 0% |
| 50% | 134 | 100% | 100% | 0% |
| 100% | 269 | 100% | 100% | 0% |

`family_skew` is the more *specific* of the two. `actor_outlier` flags every clearance the
attacker makes, cover work included. `family_skew` flags the targeted bucket and touched 1.2% of
her honest cover verdicts. For an approver reviewing a queue, that is the difference between
"distrust this account" and "distrust these verdicts".

**What would make this a finding rather than a fitted curve:** a second drill with a different
target type and seed, run before anyone looks at it. Not done; `penumbra poison-drill --flags-only`
runs one in minutes.

*Reproduce: `penumbra poison-drill --flags-only` → `artifacts/reports/poisoning_flags_nslkdd.json`.*

---

### E7b — replication, registered before it runs

The addendum's revised flag was fitted to the drill that scores it. This is the out-of-sample test.

**Design:** identical to E7, with two changes fixed now: the target rule runs with `warezmaster`
excluded, so it selects the next-smallest-support type with ≥ 50 champion alerts (we expect
`guess_passwd`, 53 training rows, without having checked), and every seed (split and verdict
assignment) is 7 instead of 42. `penumbra poison-drill --exclude-target warezmaster --seed 7 --tag _rep`.

**Predictions:**
1. `family_skew` flags **≥ 80%** of flipped verdicts at the 50% and 100% doses, and **≤ 2%** of the
   honest accounts' clearances.
2. `actor_outlier` flags **≥ 90%** of flips at every dose ≥ 25%.
3. `confident_contradiction` flags **< 10%** of flips: the rule picks a thin type, so the model is
   not confident about it.
4. The damage is again a cliff: target recall at 50% dose stays within 0.10 of the honest retrain,
   and at 100% falls by ≥ 0.30 against the champion.
5. The gate fails the 100% arm on G2 naming the target, and passes the honest arm.

**Falsification:** prediction 1 failing means `family_skew` was fitted to one drill and does not
generalise; the addendum's numbers then describe E7 only, and we say so.

#### RESULT (E7b) — `family_skew` replicates; two predictions about the other flags were wrong

Recorded after the run; the registration above (`ac1f0c0`) is unedited.

The rule, with `warezmaster` excluded, selected **`back`**, not `guess_passwd` as we expected:
956 training rows, 126 champion alerts in the feedback pool. The champion does not alert on
`guess_passwd` 50 times. That is a very different target, a well-supported DoS type the model
already detects perfectly, which makes it a harder test of generality than the one planned.

| arm | flipped | target recall [95% CI] | gate | `family_skew` | `actor_outlier` | `confident_contradiction` |
|---|---:|---|---|---:|---:|---:|
| champion | — | 1.000 [0.970, 1.000] | — | — | — | — |
| honest | 0 | 1.000 [0.970, 1.000] | pass | — | — | — |
| 10% | 13 | 1.000 [0.970, 1.000] | pass | 100% | 0% | 38% |
| 25% | 32 | 1.000 [0.970, 1.000] | pass | 100% | **0%** | 41% |
| 50% | 63 | 0.976 [0.932, 0.992] | pass | 100% | 100% | 30% |
| 100% | 126 | **0.472** [0.387, 0.559] | **FAIL** | 100% | 100% | 40% |

Honest accounts' clearances flagged: `family_skew` **0%**, `actor_outlier` 0%,
`confident_contradiction` 22%.

1. **`family_skew` ≥ 80% at 50/100% with ≤ 2% honest flags: confirmed, and stronger.** 100% at
   every dose, including 10%, with no honest false flags. The E7 addendum's revision generalises to
   a target it was not fitted on. `back` alerts carry a predicted family (`dos`) that honest
   analysts almost never clear, so even 13 clearances stand out.
2. **`actor_outlier` ≥ 90% at every dose ≥ 25%: wrong.** It missed the 25% dose entirely: 32
   flips among ~1,000 verdicts do not move an account's overall clearance rate enough. A per-account
   rate is diluted by the account's own honest volume. A per-family rate is not.
3. **`confident_contradiction` < 10%: wrong.** 30–41%. The prediction assumed a thin type; the rule
   picked a well-supported one, and the model is confident about `back`. This is E7's mechanism
   confirmed from the other side. The flag tracks model confidence, not attacker intent, and it
   still flags 22% of honest clearances.
4. **The cliff: confirmed.** 50% stays within 0.024 of the honest retrain; 100% falls 0.53 against
   the champion.
5. **The gate: confirmed.** It fails 100% on G2 (`back` canary recall 1.000 → 0.426 on 108 rows)
   and passes the honest retrain.

**Across both drills:** `family_skew` is the one integrity check that caught every damaging dose
on two different targets without flagging an honest account. It is also the only one we designed
after seeing data, which is why E7b exists.

*Reproduce: `penumbra poison-drill --exclude-target warezmaster --seed 7 --tag _rep`.*

---

## E8 — When the data drifts, is it the thresholds or the notion of normal that is wrong?

**Registered before the run. Nothing below was written after seeing a number.** The code that runs
it (`eval/threshold_refit.py`, `penumbra refit-drill`) does not exist at the time of this commit.

The detector targets 1% FPR. On NSL-KDD's own shifted test split it realises **10.2%**
(EVALUATION §7, condition B): the operating point was fitted on training benign traffic and the
benign traffic moved. There are two different repairs, and they are not the same claim:

- **move the thresholds** - keep both heads, re-fit only the two thresholds on recent benign traffic;
- **re-learn normal** - re-fit the novelty head too (`PenumbraDetector.rebaselined`, what
  `penumbra rebaseline` ships), then the thresholds.

This experiment measures what each buys, what each costs, how much recent benign traffic each
needs, and what happens when that traffic is not clean.

### Hypotheses

> **H8a — the 10.2% is a threshold problem.** Re-fitting only the thresholds on benign rows drawn
> from the shifted distribution brings the realised FPR on held-out shifted benign traffic back to
> the 1% target.

> **H8b — the repair is not free, and the price is paid in unseen-attack recall.** The shipped
> thresholds are permissive on shifted data; part of what they catch is caught *because* they
> over-alert. Moving them to a true 1% gives some of that back.

> **H8c — re-learning normal beats moving thresholds at the same false-positive rate.** A novelty
> head fitted to the benign traffic it is actually watching separates unseen attacks from it better
> than one fitted to a different network's benign traffic, so at the same realised FPR it recalls
> more of them.

> **H8d — the window-size rule is right.** `penumbra rebaseline` refuses a window whose
> calibration slice rests on fewer than five benign exceedances per head (R1: 998 calibration rows,
> a 3,327-row window, at 1%). Below that size the realised FPR should be visibly unstable from one
> window to the next.

> **H8e — a dirty baseline costs detection (THREAT_MODEL T9, measured).** Re-learning normal from a
> window that is 5% attack traffic teaches the detector some attacks are normal.

### Design

- **NSL-KDD.** A champion is fitted on `KDDTrain+` at a 1% target, seeded, the same full two-head
  pipeline as everywhere else. `KDDTest+` is split three ways by `canary.live_split` (stratified by
  attack type, fixed seed), as in E7: **canary** 30% unused here, **recent** 35% is the pool
  windows are drawn from, **evaluation** 35% is where every reported number comes from.
- **A window is N benign rows of the recent pool.** Ground-truth benign stands in for a vetted
  window (an operator's quiet week, or benign-confirmed traffic); this measures what a clean window
  buys, not a label-free way of obtaining one. N ∈ {500, 1,000, 2,000, all ≈ 3,400}; the three
  smaller sizes are drawn with five seeds each.
- **Arms**, each at a 1% total target:
  - **A0** champion as shipped.
  - **A1 thresholds only**: `OrGate.fit` on the window's supervised and novelty scores. Heads
    unchanged.
  - **A2 re-learn normal**: `PenumbraDetector.rebaselined` with the window split 5:3 into fit and
    calibration (the same ratio as `penumbra rebaseline`'s 50/30).
  - **A2-dirty**: A2 on a window of all recent benign rows plus attack rows from the recent pool
    amounting to **5%** of the window (and, as a dose check, 1%), sampled uniformly from the pool's
    attack rows, three seeds each.
- **Metrics on the evaluation slice**, each at the arm's own thresholds: realised FPR on its benign
  rows (with a Wilson 95% interval); recall on all attacks; recall on the **unseen-17** attack types
  (types absent from `KDDTrain+`) and on the seen types; and the share of benign rows reaching an
  analyst (fired or abstained). Recall means "fired", either head.

### Predicted outcome, recorded in advance

1. **A0** realises roughly 10% FPR on the evaluation slice (it is the same shift as §7, on a 35%
   sample).
2. **A1 at N = all** realises FPR in **[0.5%, 2.0%]**. (H8a)
3. **A1's unseen-17 recall is at least 0.10 below A0's**, at N = all. (H8b)
4. **A2's unseen-17 recall exceeds A1's by at least 0.05** at N = all, with A2's realised FPR also
   in [0.5%, 2.0%]. (H8c) This is the prediction we are least sure of: NSL-KDD's shift is partly
   new attack types rather than new benign behaviour, and if benign traffic barely moved, re-fitting
   the novelty head buys nothing.
5. **Across the five seeds, the standard deviation of A1's realised FPR at N = 500 is at least twice
   that at N = 2,000.** (H8d)
6. **A2-dirty at 5% loses at least 0.05 unseen-17 recall against A2 on the same clean rows**; the
   1% dose loses less than the 5% dose. (H8e)

### Falsification

H8a is refuted if A1 at N = all realises FPR outside [0.5%, 2.0%]: then the drift is not a
threshold problem and something else is broken. H8b is refuted if A1 loses less than 0.10
unseen-17 recall - the shipped operating point was not buying detection with its false positives.
H8c is refuted if A2 does not beat A1 by 0.05 at comparable FPR; the finding would then be that on
this data, re-learning normal is not worth more than moving thresholds, and `penumbra rebaseline`'s
case rests on the lab capture alone. H8d is refuted if the N = 500 spread is under twice the
N = 2,000 spread, which would say R1 is stricter than it needs to be. H8e is refuted if the 5% dose
costs under 0.05 unseen-17 recall. **Every refutation gets published.**

---

### RESULT — H8a, H8b, H8e survive; H8c and H8d are refuted

Recorded after the run. The predictions above were committed first (`e1ed326`) and are not edited.
Evaluation slice: 7,887 rows, 3,399 benign, 1,309 unseen-17. Recent pool: 3,399 benign, 4,494 attack.
`penumbra refit-drill` reproduces every number (`artifacts/reports/threshold_refit_nslkdd.json`).

At N = all 3,399 recent benign rows, 1% target:

| arm | realised FPR [95% CI] | recall | unseen-17 recall | seen recall | benign reaching an analyst |
|---|---|---:|---:|---:|---:|
| A0 shipped | 10.18% [9.21, 11.24] | 0.831 | 0.769 | 0.857 | 16.3% |
| A1 thresholds only | **1.06%** [0.77, 1.46] | 0.606 | 0.428 | 0.679 | 16.3% |
| A2 re-learn normal | 1.27% [0.94, 1.70] | 0.588 | 0.377 | 0.675 | **11.0%** |

1. **P1 held.** A0 realised 10.18%.
2. **H8a survives.** Moving the thresholds alone puts the realised FPR at 1.06%, inside the band,
   CI covering the target. The 10.2% was a threshold problem.
3. **H8b survives, and the price is large.** Unseen-17 recall 0.769 → 0.428 (−0.34). Most of what
   the shipped operating point caught of the unseen types, it caught *because* it was over-alerting
   ten-fold. That is the honest reading of the headline 0.77: at the FPR the detector claims, it is
   0.43.
4. **H8c is refuted.** Re-learning normal recalls *less* of the unseen types than moving thresholds
   (0.377 vs 0.428) at a slightly higher FPR. The hedge in the prediction was the right one:
   NSL-KDD's shift is mostly new attack types, not new benign behaviour, so there is little new
   "normal" to learn, and a head fitted on 2,124 rows (5/8 of the window) is noisier than one fitted
   on 53,875. Re-learning normal did one thing better: with its benign conformal quantile
   re-fitted, **a third fewer benign rows reach an analyst** (16.3% → 11.0%), because the conformal
   layer stops abstaining on ordinary traffic it was not calibrated for.
5. **H8d is refuted, as registered.** The A1 FPR spread across five windows is 0.24 points at
   N = 500 and 0.19 at N = 2,000: a ratio of 1.3, not 2. Re-fitting only thresholds is stable even on
   500 rows, so R1 is stricter than thresholds-only needs.

   *Exploratory, not registered:* the rule is not too strict for what `penumbra rebaseline`
   actually does. A2 calibrates on 3/8 of the window, which is below R1's 998 rows for every window
   up to 2,000. There its FPR is both biased and unstable: mean 2.35% / 1.65% / 1.72% at N = 500 /
   1,000 / 2,000, spread 1.28 / 1.23 / 0.66 points, one window at 4.59%. At N = all (1,275
   calibration rows, above R1) it lands at 1.27%. The failure R1 exists to prevent is real; it is
   the novelty head's, not the thresholds'.
6. **H8e survives, and the attack is silent.** A window that is 1% attack traffic (34 rows) costs
   0.109 unseen-17 recall; at 5% (179 rows) it costs 0.235 (one seed lost 0.33, leaving 0.046).
   **And the realised FPR falls** - to 0.6% at the 1% dose and 0.3% at 5%. An operator watching the
   false-positive rate sees the re-baseline as an improvement. That is the whole danger of T9 in
   one number: poisoning the baseline looks like tuning.

**What changes because of this.** `penumbra rebaseline` gains `--mode thresholds`, which is A1 (plus
the benign conformal quantile, the one thing A2 did better): the repair for a network that is the
same network, drifting. `--mode full` stays the default and stays the tool for a network the model
has never seen, where the lab capture showed the stock novelty head is no better than chance (ROC-AUC
0.60 → 0.97 re-baselined, §10.7h). THREAT_MODEL T9 gains the measured cost and the observation that
a falling FPR after a re-baseline is a reason to look harder, not to relax.

---

## E9a — How much of the realised FPR is our own in-sample optimism?

**Registered before the run.** The code that runs it does not exist at the time of this commit.

Three defects were found while planning E9, all in the shipped detector rather than in the data:

1. **The supervised threshold and the conformal predictor are fitted on rows the supervised model
   trained on.** `PenumbraDetector.fit` holds benign rows out of the *novelty* head only; the
   supervised head saw every training row, including the "held-out" benign rows its threshold is the
   99.5th percentile of and the 20% slice conformal is calibrated on. A forest scores its own
   training rows lower than unseen ones, so the threshold sits too low and test benign traffic
   clears it more often than the target says. For a single unpruned decision tree the in-sample
   scores collapse to 0 and every row would fire. E9 cannot be run on top of this.
2. **The shipped UNSW detector keeps the quarantined testbed TTL columns** (`sttl`, `dttl`,
   `ct_state_ttl`, `is_sm_ips_ports`). `penumbra fit` has no `--drop-artifacts`; the evaluation
   runner has always quarantined, the product never did. On real traffic TTLs are 64 or 128, values
   UNSW's benign rows never take.
3. **On a 20,000-row slice of UNSW test, the shipped detector fires on 18.5% of benign rows**
   against a 1% target (registry v002 manifest, `known_dataset.current.benign_fpr`).

### Hypotheses

> **H9f — part of the realised FPR is in-sample optimism.** Fitting the supervised model on train
> minus a stratified 20% calibration slice S, and fitting thresholds and conformal on S, lowers the
> realised FPR on test benign traffic. Not to the target: the train/test shift that E8 measured
> remains.

> **H9g — the gate rewards over-alerting.** The canary gate compares recall at each model's own
> threshold. A champion that fires on 18.5% of benign traffic buys recall with those false positives,
> so an honestly calibrated challenger fails G1 even if it is better at every matched FPR. ADR-0005
> amends the gate to compare at the champion's realised FPR.

> **H9h — the TTL artifact is part of why real traffic alerts.** Out of the box, the shipped UNSW
> detector fired on 69.8% of the lab capture's held-out ordinary flows, and every one of them
> reached an analyst (§10.7h). Quarantining the TTL
> columns removes a feature on which every real flow looks unlike UNSW benign traffic.

### Design

- **UNSW, 2×2:** {TTL kept, TTL quarantined (the loader's set, `schema.py:282`)} × {in-sample
  calibration (as shipped), held-out calibration}. RF-300, seed 42, 1% target, the full two-head
  detector. S is 20% of train, stratified by family.
- **NSL-KDD, 1×2:** in-sample vs held-out calibration. Nothing is quarantined (§2: every NSL-KDD
  candidate was judged signal).
- **Metrics on the full test split:** realised FPR with a Wilson 95% interval, attack recall, the
  share of benign rows reaching an analyst (fired or abstained), the supervised threshold and the
  benign conformal quantile.
- **Lab:** the TTL-kept and TTL-quarantined held-out detectors score `lab.pcap` out of the box, with
  no re-baseline. Metrics: fired rate on ordinary flows, fired rate on the attacker's flows, and
  the reached-an-analyst rate.
- **Gate:** the TTL-quarantined held-out detector is registered as a challenger to UNSW champion
  v001 and scored by both the old gate and the ADR-0005 gate, on the same canary slice.

### Predicted outcome, recorded in advance

1. The shipped arm (TTL kept, in-sample) realises **between 15% and 22%** FPR on UNSW test.
2. **H9f:** held-out calibration lowers UNSW realised FPR by **at least 2 points** (TTL kept), and
   the held-out arm still realises **above 2%**.
3. **H9f, NSL-KDD:** held-out calibration lowers the 10.2% realised FPR, but the held-out arm still
   realises **between 5% and 9.5%**. Most of NSL-KDD's excess is shift, which E8 repaired with
   thresholds fitted on shifted benign traffic.
4. On both datasets, held-out calibration lowers the share of benign rows reaching an analyst.
   Conformal's benign quantile is fitted on scores the model has not memorised.
5. On UNSW test, quarantining TTL moves realised FPR and recall by **less than 1 point each**.
   Trees route around the columns, as the audit found (−0.0011 AUC).
6. **H9h:** on the lab capture, quarantining TTL lowers the out-of-the-box fired rate on ordinary
   flows by **at least 10 points**, and it stays **above 5%**. The re-baseline remains necessary.
7. **H9g:** the old gate refuses the honest challenger (G1). The ADR-0005 gate passes it.

### Falsification

H9f is refuted if held-out calibration lowers UNSW realised FPR by less than 2 points. Then the
18.5% is shift or something else, not optimism. H9h is refuted if quarantining TTL moves the lab
fired rate by less than 10 points. Then the out-of-the-box failure is not about TTL, and the
quarantine is justified by the audit alone. H9g is refuted if the old gate passes the challenger, or
if the amended gate refuses it. **Every refutation gets published.** Published E7 and E8 numbers
are reproduced with the old calibration, which their commands pin, and are not edited. E8 gets an
addendum if prediction 3 holds.

---

### RESULT — H9f survives, and it is the biggest fix in the project. H9g and H9h are refuted.

Recorded after the run. The predictions above were committed first (`751b5aa`) and are not edited.
`penumbra calibration-drill` reproduces every number (`artifacts/reports/calibration_drill.json`).
Full test splits, 1% target, RF-300, seed 42.

| arm | realised FPR [95% CI] | recall | benign reaching an analyst | supervised threshold |
|---|---:|---:|---:|---:|
| UNSW, TTL kept, in-sample (as shipped) | **18.79%** [18.40, 19.20] | 0.973 | 38.6% | 0.536 |
| UNSW, TTL kept, held-out | **2.91%** [2.74, 3.08] | 0.885 | 28.5% | 0.860 |
| UNSW, TTL quarantined, in-sample | 18.76% [18.37, 19.16] | 0.971 | 38.9% | 0.540 |
| UNSW, TTL quarantined, held-out | 2.82% [2.65, 2.99] | 0.879 | 28.0% | 0.865 |
| NSL-KDD, in-sample | 9.62% [9.05, 10.22] | 0.828 | 16.3% | 0.123 |
| NSL-KDD, held-out | 9.21% [8.65, 9.80] | 0.804 | 16.0% | 0.154 |

1. **P1 held.** The shipped arm realises 18.79%.
2. **H9f survives, far beyond its prediction.** Held-out calibration cuts UNSW's realised FPR from
   18.8% to 2.9%. That is 15.9 points, against a predicted 2 or more. The shipped detector's
   threshold was the 99.5th percentile of scores the forest gave its own training rows. Those
   scores are low because the forest memorised them, so the threshold sat at 0.54 when unseen
   benign traffic needed 0.86. **Most of the shipped UNSW operating point's recall was bought with
   that over-alerting.** At an honest threshold recall is 0.885, not 0.973. It is E8's lesson
   again, and this time the cause was our own code, not drift.
3. **P3 held.** NSL-KDD moves only from 9.62% to 9.21%. Its excess is the train/test shift E8
   measured, not optimism.
   - *Not explained here:* this run's in-sample arm realises 9.62% on the full test split, while §10.4
     reported 10.17%. We checked that the new tie-safe threshold does not cause it: plain and tie-safe
     thresholds give identical numbers on this fit. The source of the 0.55-point gap is not
     identified.
4. **P4 held, barely on NSL-KDD.** Benign rows reaching an analyst fall from 38.6% to 28.5% on UNSW,
   and from 16.3% to 16.0% on NSL-KDD.
5. **P5 held.** Quarantining TTL moves UNSW realised FPR by 0.09 points and recall by 0.006. Trees
   route around the columns.
6. **H9h is refuted.** On the lab capture out of the box, quarantining TTL moves the ordinary-flow
   fired rate from 38.4% to 38.7%: nothing. **What moved it was calibration.** The shipped detector
   fired on 70.3% of the 912 ordinary flows; held out, it fires on 38.4%. Reached-an-analyst goes
   from 100% to 91.4%. The TTL columns are an audit finding, not the reason real traffic alerts.
   The quarantine stays, because the audit says those values are testbed artifacts; it is not sold
   as a lab fix. The attacker's scan flows still fire at 0.3%. Re-baselining remains the only thing
   that detects them (§10.7h).
7. **H9g is refuted, in an instructive way.** On the UNSW canary (30% of test), the champion fires on
   18.8% of benign rows at recall 0.971. The challenger (TTL quarantined, held-out) fires on 2.9% at
   recall 0.881.
   - **As predicted, the old gate refuses** on G1, and on G2 for Fuzzers (0.81 → 0.35) and
     Shellcode.
   - **The ADR-0005 gate also refuses:** recall 0.971 → 0.946 at the matched 18.0% FPR, and
     Fuzzers 0.81 → 0.66.

#### Diagnosis (post-hoc, after the refusal)

The ADR-0005 rule refits the challenger's two thresholds at the champion's realised **total** FPR,
using the detector's equal per-head split. The champion does not spend its budget equally. On the
canary's benign rows its supervised head fires on 18.6% and its novelty head on 0.23%. The
"matched" challenger therefore ran its supervised head at about 9.4%, half the champion's, and
spent the other half on a novelty head that buys little on UNSW. The comparison changed the
operating point **and** the head mix. Only the first was intended.

This is a flaw in a rule we registered, found by the run that tested it. The revision below is
registered before it is run, and this refusal stays in the record.

### E9a-r — the gate revised to match per head (registered before the re-run)

**Revision (ADR-0005, revision 1):** G1 and G2 compare the challenger with **each head thresholded
at the champion's realised canary FPR for that head**. The supervised threshold is placed where the
champion's supervised head fired on the canary's benign rows, and likewise for novelty. G3 and G4
are unchanged.

**Prediction:** the same honest UNSW challenger, refitted by `penumbra registry challenge -d unsw
--drop-artifacts` (same seed and data, so the same model), **passes** the revised gate. Overall
recall at the matched per-head operating point is within 0.02 of the champion's 0.971. No family
with at least 20 canary rows loses more than 0.10, Fuzzers included.

**Falsification:** if it refuses, the champion stays, the refusal is published, and the gate is not
revised again for this challenger.

#### RESULT — the prediction held. The challenger passed and was promoted.

The revision and prediction were committed first (`b0d929f`), and the code after that (`feb7717`).
Run with `penumbra registry challenge -d unsw -m rf --drop-artifacts`, which registered
**v003-20261003T231527**.

| UNSW canary (24,676 rows) | recall | realised FPR |
|---|---:|---:|
| champion v001 at its own thresholds | 0.971 | 18.78% (supervised 18.59%, novelty 0.23%) |
| challenger at its own thresholds | 0.881 | **2.86%** (99% interval [2.47%, 3.30%]) |
| challenger, each head at the champion's rate | **0.970** | 18.73% |

- **At the champion's per-head operating point the challenger is the same detector.** Recall is
  0.970 against 0.971. The largest family change is Backdoor at −0.006. Fuzzers, the family that
  failed both earlier rules, moves from 0.805 to 0.803.
- **What the challenger changes is where it operates.** It runs at a 2.9% FPR instead of 18.8%,
  with the TTL artifacts quarantined.
- On the separate evaluation slice the numbers are 0.974 at 18.9% for the champion, against 0.879
  at 2.8% for the challenger.
- **The verdicts recorded beside it:**
  - at own thresholds: refuse (G1; G2 for Fuzzers and Shellcode);
  - at the matched total FPR: refuse (G1; G2 for Fuzzers);
  - per head: pass.
- **G4 is advisory and says what is still true.** 2.86% is not 1%. The remaining excess is
  UNSW's train/test benign shift, which `rebaseline --mode thresholds` (E8) is the tool for. It
  is not more calibration.
- Promoted by a different account (`admin`) from the one that registered it (`senior`), and
  recorded in the audit log.

**Not predicted, recorded for completeness: NSL-KDD.** `registry init -d nslkdd` registered the
shipped in-sample detector as the first champion. It was ungated, and the manifest says so.
`registry challenge -d nslkdd -m rf` then gated a held-out challenger with the same revised rule:

- **Refused.** `processtable`, one of the 17 attack types training never contains, fell from 0.79
  to 0.63 recall on 206 canary rows at the matched per-head operating point.
- At its own threshold the challenger runs at 8.72% against the champion's 9.17%. NSL-KDD's excess
  is shift, so calibration buys little there and a family regression is a real cost.
- The NSL-KDD champion is unchanged.

---

## E9 — Does a 3-tree + 3-SVM ensemble beat a 300-tree forest?

**Registered before the run.** The code that runs it (`models/ensemble.py`, `eval/ensemble.py`,
`penumbra ensemble`) does not exist at the time of this commit. Run on top of E9a's held-out
calibration.

The question a judge asks of every tree-based detector: would a different family of model do
better? An SVM draws smooth margins where a tree draws boxes, so the two make different mistakes,
and different mistakes are what an ensemble needs. This experiment builds the smallest honest version
(three decision trees, three SVMs) and measures it against the shipped RF-300 on the full training
data of all three datasets. It also measures the preprocessing an SVM needs, and a tree does not:
normalisation and PCA, in both orders.

### Members

| | member | |
|---|---|---|
| T1 | decision tree | gini, unlimited depth, `min_samples_leaf=5`, balanced class weights |
| T2 | decision tree | entropy, `max_depth=20`, `max_features="sqrt"` |
| T3 | decision tree | gini, `max_depth=12`, `min_samples_leaf=20` |
| S1 | linear SVM | `LinearSVC` |
| S2 | RBF SVM | `Nystroem(rbf, gamma="scale" rule)` → `LinearSVC` |
| S3 | polynomial SVM | `Nystroem(poly, degree 2, coef0 1)` → `LinearSVC` |

**The SVMs are kernel approximations, and that is a stated limitation.** An exact kernel SVC costs
O(n²) memory and O(n²–n³) time. On 140,000 to 1.2 million rows and 16 GB, that is not a slow run.
It is no run. Nystroem with k landmarks makes a linear SVM in the approximate kernel space:
k = 1,000 on UNSW and NSL-KDD, and 256 on CICIDS. On CICIDS the SVMs are trained with
`SGDClassifier(hinge, average=True)` in float32, because liblinear's copy of a 1.2M × 256 matrix
does not fit. An exact `SVC(rbf)` is fitted on a 30,000-row stratified subsample beside its
Nystroem twin to measure the approximation's cost. On a GPU machine, if one is available, a full-data
exact SVC is fitted as well (cuML). `C` ∈ {0.1, 1} is chosen per member on the holdout. Trees differ
by criterion, depth and `max_features`. Nothing is bootstrapped, so every member sees the full data.

### Preprocessing arms (each SVM member's own pipeline; trees run on the same arms as a control)

- **P0 raw:** one-hot and imputation only. `max_iter=200`, with convergence failures recorded.
- **P1 normalise:** signed log `sign(x)·log1p(|x|)`, then z-score.
- **P2 normalise → PCA:** P1, then PCA keeping 95% of the variance.
- **P3 PCA → normalise:** PCA on the unscaled features, then z-score the components. P3a keeps 95%
  of variance. P3b keeps P2's number of components.
- **hetero:** trees on P0, SVMs on the better of P1 and P2 (chosen on the holdout).

Every transform is fitted inside the training fold, never on the holdout or test.

### Combiners, built from one out-of-fold score matrix

Stacking (logistic regression on out-of-fold member scores), soft voting (each member Platt-scaled
on its out-of-fold scores), and hard voting (6 votes give 7 score levels).

### Selection protocol (test is never used to choose)

- **Stage A:** members are fitted on train minus S (E9a) minus a stratified 20% holdout H, and
  scored on H. The configuration (arm × combiner) with the highest recall at an exact 1% benign
  budget on H is selected (`flags_at_benign_budget`). Ties are broken by lower fit time.
- **Stage B:** the selected configuration, and the user-requested P3 configuration, are refitted
  with 5-fold out-of-fold stacking. On CICIDS this is 3 folds **grouped by day**, because shuffled
  folds on time-ordered flows reward memorisation.
- **CICIDS** runs the configuration UNSW selected, plus RF and XGBoost baselines, unless a larger
  machine runs its full grid. Either way, the report says which.
- **Test is scored once**, at the end.

### Metrics

Recall at an exact 1% benign budget, with the realised FPR printed beside it. The **paired
bootstrap of the difference** against RF-300 on the same rows: 500 stratified resamples, and 200
moving-block resamples on CICIDS. ROC-AUC with an interval, per-family recall, NSL-KDD unseen-17
recall at the matched budget, and CICIDS Thursday–Friday recall. McNemar's test with discordant
counts. Member diversity: pairwise disagreement and Yule's Q on test errors. Fit time, latency at
batch 1 and batch 2,048, and artifact size. Every report carries the machine it ran on, the git
commit and the compute profile. Timings are only compared within a machine.

### Hypotheses and predicted outcome, recorded in advance

1. **H9a — the ensemble does not beat the forest.** On UNSW and CICIDS, the paired 95% interval of
   (ensemble − RF-300) recall at 1% FPR has an upper bound **at or below +0.005**. RF-300 is already
   an ensemble of 300 decorrelated trees, and three trees plus three linear-margin models add
   little it lacks.
2. **H9b — SVMs need normalisation.** Each of S1–S3 under P0 is **at least 0.10** below the same
   member under P1, in holdout recall at 1% FPR.
3. **H9c — PCA is for the SVMs, not the trees.** On UNSW the trees lose **at least 0.02** recall under
   P2 against P0 (rotation destroys the axis-aligned splits trees exploit). The SVMs under P2 land
   **within ±0.01** of P1. **P3a keeps three components or fewer, loaded on byte, rate and load
   columns**: PCA on unscaled flow data finds the units, not the structure.
4. **Sanity, not hypothesis:** trees under P1 agree with trees under P0 on **at least 99%** of
   holdout rows. Scaling is monotone and trees are invariant to it. Agreement is not exactly 100%
   because trees cast to float32.
5. **H9d — a smooth margin extrapolates differently.** On NSL-KDD at a matched 1% test-benign
   budget, S2's (RBF) recall on the 17 unseen attack types **exceeds RF's 0.053** (the CI
   baseline). This is the prediction we are least sure of.
6. **H9e — diversity comes from mixing families.** On all three datasets, the mean Yule's Q between
   tree–SVM pairs is **lower** than between tree–tree pairs.
7. **Approximation check:** on the 30,000-row subsample, exact `SVC(rbf)` and its Nystroem twin
   differ by **at most 0.01** in recall at 1% FPR. If a GPU run happens, the same holds on full UNSW
   (H9i).
8. **SGD precondition:** on UNSW, the SGD-trained S1–S3 match the LinearSVC-trained ones within
   **0.005 ROC-AUC**. If not, every CICIDS SVM number is labelled SGD-only.
9. **Hard voting cannot hit the budget:** on the UNSW holdout, no vote level realises an FPR inside
   [0.75%, 1.25%]. With seven levels the operating point is chosen by the vote count, not by the
   analyst.

### Falsification

H9a is refuted if the paired interval's upper bound exceeds +0.005 on either dataset: then the
ensemble is better, and it is offered to the gate on that basis. H9b is refuted if any SVM member
loses less than 0.10 without normalisation. H9c is refuted if the trees lose under 0.02 to PCA, if
the SVMs move more than 0.01, or if P3a keeps more than three components. H9d is refuted if S2's
unseen-17 recall is at or below 0.053. H9e is refuted if tree–SVM Q is not lower on any dataset.

**What ships is decided by the gate, not by this experiment.** The selected ensemble is registered
as a challenger on each dataset and promoted only if it passes ADR-0005's gate. If H9a holds,
RF-300 stays the champion, and the reason is a measured one. **Every refutation gets published.**

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
