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
