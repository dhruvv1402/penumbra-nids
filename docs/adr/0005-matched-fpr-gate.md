# ADR-0005 — The promotion gate compares models at a matched false-positive rate

**Status:** accepted · **Date:** 2026-10-04

## Context

Promotion requires a passed canary gate (`eval/canary.py`, `models/registry.py`). The gate has three
criteria, each measured at the model's **own** operating point:

- **G1:** overall attack recall must not fall by more than 0.02.
- **G2:** no family with at least 20 canary rows may lose more than 0.10 recall.
- **G3:** benign FPR must not rise by more than 0.01.

G1 and G2 compare recall at two different false-positive rates. That is only fair when the two
models realise roughly the same FPR. Until now they always did: the gate was built for E7, where
champion and challenger differ by a few hundred feedback labels.

**The UNSW champion v001 fires on 18.5% of benign rows** on a 20,000-row slice of UNSW test,
against a 1% target. That is E9a's in-sample calibration problem. It was promoted as the
registry's first champion, which the registry allows without a gate and records as
`gate_passed: false`. Part of its recall comes from those false positives.

Now suppose a challenger is honestly calibrated, alerting on 2% of benign traffic.

- It loses recall at its own threshold, simply because it alerts less.
- G1 refuses it, even if it is better than the champion at every matched FPR.
- G3 never objects to the champion, because G3 only penalises a *rise*.

**The gate as written rewards over-alerting.** That is the opposite of what a SOC needs: alert
fatigue is the failure the whole product is designed against.

## Decision

1. **G1 and G2 are measured at the champion's realised canary FPR.**
   - The challenger's two thresholds are refitted on the canary's benign rows at a total budget equal
     to the champion's realised canary FPR (`OrGate.fit`, the same per-head split the detector uses).
   - Its recall, overall and per family, is read at that operating point.
   - The canary is never reported and never trained on, so fitting a comparison threshold on its
     benign rows leaks nothing into any published number.
   - The per-family check, which is the defence against targeted poisoning (THREAT_MODEL T1), is
     unchanged in kind. It compares like with like now.
2. **G3 is unchanged.** At its own threshold, the challenger may not alert on more than 1 point more
   benign traffic than the champion.
3. **G4 is new and advisory.** The challenger's realised canary FPR at its own threshold is reported
   against the target, with a 99% binomial interval.
   - It does not block, because on a shifted test split even an honest model misses its target (E8).
   - It is printed beside every verdict, so "fires on 18.5% of benign traffic" can never again be
     promoted silently.
4. **Both verdicts are recorded.** The manifest stores the old gate's result
   (`gate.at_own_threshold`) beside the amended one (`gate.passed`, `gate.at_matched_fpr`).
   `registry.promote` reads the amended verdict.
5. **E7's published numbers keep the old gate.** `penumbra poison-drill` measures the old criteria,
   as registered. Under E7's conditions champion and challenger realise nearly the same FPR, so the
   two gates agree there by construction.

## Consequences

- An honest challenger can replace an over-alerting champion. A challenger that is worse at a matched
  FPR still cannot.
- A poisoned challenger still fails G2: a family taught to look benign loses recall at any threshold.
  E7's mechanism is untouched.
- The gate now needs scores, not just fired flags, so `canary.evaluate` gains a matched variant.
  `GateResult` grows two fields. Old manifests without them still load.
- G4 makes the realised FPR of every promoted model visible in the registry and on the console's
  governance page. That number was previously only in a manifest's evidence block.
- Pre-registered in EXPERIMENTS.md E9a (H9g) before any challenger was scored: the old gate is
  predicted to refuse the honest UNSW challenger, and this one to pass it.

## Revision 1 — match per head (2026-10-04, after the first run refused)

**What went wrong.** E9a's run refused the honest UNSW challenger under this rule as well. The
cause was found afterwards, and it is a flaw in decision 1.

- The rule refitted both heads at the champion's *total* FPR, using the detector's equal per-head
  split.
- The champion does not split its budget equally. On the canary's benign rows its supervised head
  fires on 18.6% and its novelty head on 0.23%.
- So the "matched" challenger ran its supervised head at about half the champion's rate. The
  comparison moved the head mix as well as the operating point.

**Decision.** G1 and G2 place each challenger head at the champion's realised canary FPR **for that
head**: the supervised threshold where the champion's supervised head fired on the canary's benign
rows, and likewise for novelty. A champion head that never fired stays off in the challenger. G3
and G4 are unchanged.

**Why this is not tuning the gate until it passes.**

- The revision and its prediction were committed before the re-run (EXPERIMENTS.md E9a-r).
- The refusal under the original rule stays in the record.
- If the revised gate also refuses, the champion stays and the gate is not revised again for this
  challenger.
- Per-head matching is also the stricter reading of "compare at the same operating point". Two
  detectors at the same total FPR with different head mixes are not at the same operating point.
