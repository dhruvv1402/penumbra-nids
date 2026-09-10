# ADR-0002 — Two detection heads, two alert lanes, three output numbers

**Status:** accepted · **Date:** 2026-09-11

## Context

The brief's title is *"Catch the Attack the Signatures Miss."* The obvious response is a supervised
classifier over a labelled intrusion dataset. That response does not solve the stated problem.

A supervised model trained on families {A…I} and shown family J has no representation for "unknown."
Softmax assigns J to whichever known class it resembles, and when J resembles benign traffic more than
any attack class, the model outputs `normal` — confidently, because confidence is calibrated over the
classes it knows, not over the ones it does not. Shipping that and describing it as zero-day detection
reproduces the exact failure the brief describes, one abstraction level up.

## Decision

### 1. Two heads with different failure modes

| | Known-threat head | Novelty head |
|---|---|---|
| Trained on | labelled attacks + benign | **benign only** |
| Question | "which known family is this?" | "is this unlike anything I've seen?" |
| Fails when | family is new → answers `normal` | benign traffic legitimately shifts |
| Models | LogReg (floor), RandomForest, XGBoost | autoencoder, IsolationForest, Mahalanobis |

The novelty head's preprocessing — scaler, encoders, reference distribution — is fitted on **benign
rows only, in a pipeline separate from the supervised one**. Sharing a scaler fitted across all data
would leak the attack distribution into a model whose entire claim is that it has never seen an attack.
Subtle, and exactly what a reviewer looks for.

### 2. Novelty detectors are fused, then thresholded once

Rank-normalise each detector's score against the benign reference distribution, combine, threshold the
combination. **Not** three independently thresholded 1% detectors OR-ed together — that stacks to
roughly 3% combined FPR and makes any calibration statement meaningless. Raw reconstruction error and
`IsolationForest.decision_function` are on incomparable scales; percentile ranks are comparable by
construction.

### 3. Two lanes, because 1–4% FPR is a fact and not a bug

An autoencoder over benign flow features runs at roughly 1.5–4% FPR on held-out benign, worse on real
traffic, and drifts. At realistic prevalence that cannot produce tickets.

- **Known-threat lane** → the incident queue. High precision, SLA'd, alert-shaped.
- **Unknown-behaviour lane** → a **ranked hunting queue with a fixed daily budget**. Top-N by novelty
  percentile. Its honest metric is **Precision@k, not FPR.**

A fixed-budget queue cannot cause alert fatigue by construction. This is the difference between hiding
a false positive rate and designing around it — and it is why "the autoencoder has a 3% FPR" is a
statement we can make on stage without flinching.

### 4. Three output numbers, not one risk score

Blending a calibrated probability with an anomaly score and calling the result calibrated is wrong;
calibration is a property of probabilities. So:

| Field | Meaning | Calibrated |
|---|---|---|
| `p_attack` | P(attack \| x), isotonic-calibrated on a held-out split | **yes** — Brier and the reliability diagram describe this alone |
| `novelty_percentile` | "more unusual than 99.7% of known-benign traffic" | n/a — it is a percentile |
| `priority` | 0–100 triage sort key | **no** — documented as an ordering |

`novelty_percentile` is deliberately a percentile and not a reconstruction error. "MSE 0.0431" means
nothing to a tier-1 analyst; "more unusual than 99.7% of normal traffic" is self-explaining.

### 5. Verdict lattice

```
KNOWN_ATTACK(family) · SUSPECTED_NOVEL · UNCERTAIN · BENIGN · BENIGN_BY_POLICY
```

`UNCERTAIN` is the conformal abstention lane: where the Mondrian conformal prediction set is ambiguous,
the alert routes to human review rather than being silently guessed. Human-in-the-loop with a stated
coverage level rather than a gesture.

## Rationale for the specific models

**LogReg** is the floor. Every claim is measured against it; a complex model that does not beat a
linear one is a finding.

**XGBoost is the champion, and there is no transformer.** Gradient-boosted trees outperform deep
architectures on tabular data of this size and shape (Grinsztajn et al., 2022), and on 8 CPU cores
without a GPU we cannot afford the hyperparameter search a transformer needs to be competitive. We
would ship an undertuned FT-Transformer that loses to XGBoost and proves nothing. If time permits it
returns as a one-slide honest ablation, never as a headline model.

**The autoencoder is a 42-32-16-8-16-32-42 MLP.** It is not deep and we do not call it deep.
Categorical columns are frequency-encoded or reduced to top-15 plus `OTHER` rather than one-hot —
`proto` has roughly 130 distinct values and one-hot encoding would let categorical noise dominate the
reconstruction error.

**Mahalanobis is expected to lose.** 42 correlated, heavy-tailed, non-Gaussian features give a
near-singular covariance matrix; it needs Ledoit-Wolf shrinkage or PCA-whitening to be usable at all.
It stays because an ablation in which one of three detectors is clearly worse is more informative than
one in which all three look interchangeable.

**The sequence head runs only on CICIDS2017**, because it needs source IPs and timestamps that the
UNSW-NB15 split CSVs do not contain (ADR-0004). Its windows are strictly **causal** — features for
flow *t* use only flows before *t*. A non-causal window is a time-travel leak that produces a beautiful
and completely fake result.

## Consequences

- The headline claim is now an empirical question rather than an assertion, which is why it is
  pre-registered in `docs/EXPERIMENTS.md` with a predicted **mixed** outcome and a falsification
  condition.
- Two lanes mean two sets of metrics and two UI surfaces. Accepted; the alternative is one queue that
  is either too noisy to work or too narrow to catch anything new.
- Maintaining two preprocessing pipelines is a real cost and a real source of bugs. Mitigated by
  property tests asserting the benign-only pipeline never sees an attack row during fit.

## Related

ADR-0001 (alert, don't block) · ADR-0004 (dataset roles) · `docs/EXPERIMENTS.md` E1
