# Questions we expect, and the answers we can defend

Each answer points at the measurement behind it. If a question is not here and the answer is not in
`docs/EVALUATION.md`, the honest reply is "we did not measure that".

---

## The model

**Why a random forest and not deep learning?**
- Trees beat deep networks on tabular data of this size (Grinsztajn et al., 2022).
- We measured it anyway, with a CNN + BiGRU sequence head on CICIDS2017 (E6). The forest scored
  0.9959 recall at 1% FPR and the deep model 0.6084. The deep model learned Wednesday's traffic
  shapes and missed Thursday's attacks.
- See EVALUATION §10.7c.

**Did you try anything other than the forest?**
- Logistic regression (the floor), XGBoost, and E9: three decision trees plus three SVMs, stacked.
  We ran it with every preprocessing arm (raw, z-score, z-score then PCA, PCA then z-score) on all
  three datasets.
- The forest won on every dataset, with paired bootstrap intervals:
  - UNSW: 0.859 against 0.826 recall at 1% FPR;
  - NSL-KDD's shifted test: 0.461 against 0.191;
  - CICIDS's unseen Thursday–Friday: 0.852 against 0.311, where the ensemble's ranking was at chance
    (AUC 0.50).
- The further from the training data, the wider the gap. Six models that memorise their training
  distribution do not degrade as gracefully as 300 averaged trees.
- XGBoost tied the forest on UNSW.
- See EXPERIMENTS.md E9 and EVALUATION §10.7k.

**Are those really SVMs?**
- They are linear SVMs on a Nystroem kernel approximation: 1,000 landmark points on UNSW and
  NSL-KDD, 256 on CICIDS.
- An exact kernel SVM needs O(n²) memory. On 140,000 to 1.2 million flows and 16 GB of RAM it is not
  a slow run; it is no run.
- We measured what the approximation costs by fitting an exact `SVC(rbf)` beside its Nystroem twin
  on the same 30,000 rows. On UNSW the approximation was *better* (holdout recall 0.828 against
  0.733), and on NSL-KDD they agreed within 0.001. The approximation is not what holds the SVMs back.

**PCA before or after normalisation?**
- After. We ran both because both were asked for. PCA on unscaled flow data finds the units, not
  the structure: byte counts span seven orders of magnitude, so the first component is just
  "bytes".
- Measured: PCA on unscaled UNSW keeps 2 components. The first (82% of the variance) is the two TCP
  initial sequence numbers, random 32-bit integers. The PCA-then-normalise ensemble drops to 0.537
  recall.
- Even after normalisation, PCA cost the trees 0.06 and the linear SVM 0.34. Normalisation alone
  was the right preprocessing for the SVMs.
- Trees gain nothing from either. Scaling is monotone, so they ignore it, and PCA rotates away the
  axes they split on.

## The numbers

**Your detector targets 1% false positives. What does it actually run at?**
- UNSW: **2.9%** on the test split, as shipped from v003. Until E9a it was **18.8%**: we fitted the
  threshold on rows the forest had memorised. We found that ourselves, measured it, fixed it, and
  published both numbers (EVALUATION §10.7j).
- NSL-KDD: 9.6%, because its test traffic is distributed differently from its training traffic.
  Re-fitting the thresholds on recent benign traffic brings it to 1.06% (E8).
- **A target is not a realised rate. We print both everywhere.**

**Why not report accuracy or AUC?**
- UNSW's test set is 55% attack; a real network is around 0.1% or less. Accuracy and precision move
  with that base rate.
- Recall and FPR do not, so those are the primary numbers. Precision is always printed with its
  prevalence.
- AUC ranks over operating points nobody deploys. E6 has a model with the higher AUC and the lower
  recall at 1% FPR.

**Does it catch attacks it has never seen?**
- On NSL-KDD's 17 attack types that never appear in training: the supervised head catches 5% at
  1% FPR, and the novelty head lifts that to 35%.
- At 10% FPR the novelty head costs 9 points.
- On a shifted test split the honest unseen-attack recall at the FPR we claim is 0.43, not 0.77
  (E8). We report the curve, not the best point.

**Your best UNSW number used features you then removed?**
- No. The audit quarantined four testbed-TTL columns. Quarantining them costs 0.0011 AUC, and in E9a
  0.09 points of FPR.
- Half of our mined detection rules rested on them, so those were discarded (§10.7b). The
  quarantine does not explain why real traffic alerts: on our own lab capture it changed nothing
  (H9h refuted).

## The product

**Why does it never block?**
- ADR-0001. A false positive that blocks is an outage that someone else caused.
- It is also a performance fact: scoring is a batch operation, about 12.6 ms for one flow on the
  forest. No inline enforcement fits that budget.
- CI fails the build if a blocking code path appears.

**How do you stop a bad model reaching production?**
- A canary gate on a frozen labelled set, checked per attack family, so a model poisoned to ignore
  one family cannot hide behind its average (E7).
- Since ADR-0005, recall is compared at the champion's operating point, head by head, so an
  over-alerting champion cannot win on false positives.
- Our first version of that rule was wrong, and the run that tested it showed it. The revision was
  registered before its re-run.
- Promotion is a separate human step by a different account, recorded in a hash-chained audit log.

**Can I run the models myself?**
- Yes, on the live site, even as a guest. The **Try the model** page scores three kinds of input
  with the deployed champions:
  - held-out test flows, with the true label beside each;
  - a CSV in a model's schema (a template is downloadable);
  - your own pcap.
- It is rate-limited, capped, rounded, audited and stores nothing (ADR-0006). A public scorer is an
  oracle, and we say how we bounded it.

**What happens on a network it was not trained on?**
- Out of the box it does not work. On our lab capture, even after the calibration fix, 91% of
  ordinary flows reached an analyst.
- `penumbra rebaseline` learns that network's normal from its own benign traffic, with no labels.
  Ordinary flows alerting go from 100% to 4.9%, scan flows detected 92.9%. It is gated on a held-out
  slice and registered as a candidate (§10.7h).

**Where does Microsoft fit?**
- Alerts are emitted in ASIM NetworkSession format through the Logs Ingestion API into Sentinel. We
  ran it against a live workspace: 8,332 records, 13 incidents, every record `DvcAction=Allow`.
- The console runs on Azure Container Apps.

## The data

**These datasets are old. Why trust any of it?**
- Don't trust the absolute numbers. NSL-KDD is from 1998/99, UNSW-NB15 from 2015, CICIDS2017 from
  2017, and none of it is production traffic. The model card says so first.
- What transfers is the method: the comparison at matched false-positive budgets, the leak audit,
  the pre-registration, and the refusal to report a number we cannot reproduce.

**What is the weakest part?**
- Family names on UNSW: DoS 0.125, Backdoor 0.093, Analysis 0.090. These flows are still detected
  as attacks (0.997-1.000). It is the family label that is unreliable.
- EVALUATION §10.2 measures why. About 90% of Analysis and Backdoor test rows share an exact
  feature vector with another family's row, so 0.086 and 0.111 is the most any classifier can reach.
- DoS is held down by the two official splits labelling the same flows differently.
