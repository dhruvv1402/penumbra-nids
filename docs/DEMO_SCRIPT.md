# Demo script

Eight minutes. The spine is one question — *what happens when the attack isn't in the training
set?* — and every section answers part of it with a measured number.

**Rehearse against a frozen tag, not `main`.** Build the images the night before. Run the whole
thing once with the network cable pulled.

---

## Before you start

```bash
uv run penumbra fit -d nslkdd              # ~2 min, do this beforehand
uv run penumbra serve                      # terminal 1
cd console && npm run dev                  # terminal 2
uv run penumbra replay -d nslkdd --ingest  # terminal 3, when you want the queue to fill
```

Have `artifacts/reports/` open in a second window. Every number quoted below is in a JSON file there
and regenerates from one command.

**Fallback, tested:** `penumbra replay --from-fixture tests/fixtures/demo_alerts.json` replays
pre-scored alerts with no model, no dataset and no network. If anything is wrong on the day, use it
and say nothing about it.

---

## 1 — The problem, in one slide (60s)

> "Signature IDS misses novel attacks. So people train a classifier. But a classifier trained on
> nine attack families has no bucket for the tenth — it puts it in `normal`, confidently. We didn't
> assume that. We measured it."

**NSL-KDD's test set contains 17 attack types that never appear in training.** 3,750 rows, 16.6% of
the test set. The dataset's authors built the experiment; we just ran it.

| | recall at 1% FPR |
|---|---|
| supervised classifier, attack types it *was* trained on | 0.63 |
| same classifier, the 17 types it was **never** shown | **0.05** |

> "Five percent. That's the gap this project exists to close."

---

## 2 — The two-lane architecture (60s)

Show the console. Two lanes side by side.

> "Left lane: a supervised head, high precision, SLA'd — normal tickets. Right lane: a second
> detector trained on **benign traffic only**. It has never seen an attack, so it can't fail to
> recognise one — everything unfamiliar looks unfamiliar.
>
> The novelty head runs at 1–4% false positive rate. At realistic prevalence that can't produce
> tickets, so it doesn't: the hunting lane is a **ranked queue with a fixed daily budget**. A queue
> with a cap can't cause alert fatigue. That's not a workaround, it's the design."

Click an alert. Three numbers: `p_attack` (calibrated), `novelty_percentile` (a percentile, not a
probability), `priority` (a sort key, explicitly not a probability).

> "Three different kinds of number, shown as three different things. You can't blend a calibrated
> probability with an anomaly score and still call it calibrated."

---

## 3 — The result, as a curve (90s) — **the centrepiece**

| realised FPR | supervised | + novelty head | Δ |
|---|---|---|---|
| 0.5% | 0.052 | 0.154 | +0.10 |
| **1.0%** | **0.053** | **0.346** | **+0.29** |
| **2.0%** | **0.085** | **0.541** | **+0.46** |
| 5.0% | 0.644 | 0.752 | +0.11 |
| **10.0%** | **0.864** | **0.769** | **−0.09** |

> "At 2% false positives we recover 46 points of recall on attacks the model has never seen —
> flagging the same number of benign flows, because the OR cost is charged: two detectors on a 1%
> budget each run at half a percent.
>
> **And at 10% it makes things worse.** A novelty head is a trade, not a free addition. If we'd
> reported one operating point we could have picked whichever one flattered us. The curve is the
> honest object."

`mscan` 0 → 0.71. `apache2` 0 → 0.53. `snmpgetattack` and `worm` stay at zero regardless. Say that
too.

---

## 4 — What we got wrong (90s) — **the credibility slide**

> "Three times this experiment gave us a number we liked, and three times it was wrong."

1. **We matched on alert count.** On a 57%-attack test set the budget gets spent on true positives
   and caps recall at 9% before any detector speaks. Result: Δ = +0.0003, apparently flat.
2. **We fused with `max(p_attack, novelty_percentile)`.** On benign rows p_attack sits near zero and
   the novelty percentile is uniform — the max inherits a 0.996 threshold and throws away most
   supervised detections. Result: a flattering **+0.356** that was a bug.
3. **We set the threshold using test-set labels.** Not deployable. Fixing it surfaced something
   real: a 1% target fitted on *training* benign realises **10.2%** on test — and 10% is exactly
   the band where fusion hurts.

> "Each of those is now a test. The third one isn't a bug we fixed, it's a finding we kept: your
> operating point doesn't transfer, and that's the argument for drift monitoring."

---

## 5 — Honest evaluation, three ways (90s)

**Prevalence.** UNSW's test set is 55% attack; real networks are 1-in-10,000.

> "At a million flows a day, 1% FPR and realistic prevalence: 100 real attacks, 10,000 false
> alarms. Positive predictive value about 1%. That's Axelsson's base-rate fallacy and it's why
> every precision number measured on a balanced test set is fiction. We report TPR and FPR, which
> don't move with base rate, and we label the projections *modelled*."

**The artifact audit.**

> "We predicted `sttl` would be a testbed artifact. Our AUC test said no — it ranked 17th. But
> `sttl = 31` is 22.5% of training rows and **100% benign**. AUC is a ranking metric; it can't see
> one value that fingerprints a subpopulation. One detector isn't enough, and a negative result
> from one isn't evidence of absence."

**The SMOTE demo.** UNSW `Worms`, 130 rows:

| | F1 |
|---|---|
| SMOTE inside the fold | **0.12** |
| SMOTE before the split | **0.999** |

> "Same model, same data, same resampler. The only difference is *where*. The leaky version doesn't
> error — it reports a near-perfect number. And ROC-AUC moves by 0.008, so the leak is invisible in
> the metric most papers lead with."

Then the ablation: **every SMOTE variant scored below doing nothing.** SMOTE-NC spent 426 seconds to
finish two points worse.

---

## 6 — Alert, never block (45s)

> "There is no blocking code path in this repository. Not disabled — absent. CI greps the tree and
> fails the build if one appears, and a test walks the entire API surface looking for a route that
> could enforce."

Show the ASIM payload:

```
DvcAction        = "Allow"      <- we did not block
ThreatConfidence = 93           <- and we were confident
EventSeverity    = "High"
```

> "That's the design decision stated in Microsoft's own normalized schema. At 1% FPR on a busy link,
> auto-blocking takes the business offline faster than any attacker would. EU AI Act Article 14
> requires a human be able to override and halt high-risk AI. We alert. A human decides."

---

## 7 — Drift, live (45s)

```bash
uv run penumbra replay -d nslkdd --ingest --inject-drift abrupt
```

> "Known change point, so detection delay is measured rather than claimed. ADWIN fires 39
> observations after the change; Page-Hinkley takes 122."

> "And the PSI bug worth knowing: if you recompute your bin edges each window, PSI reads 0.0000 on a
> two-sigma shift, because the bins follow the data. Ours are frozen to the reference window. Same
> data, PSI 0.9156."

---

## 8 — Close (30s)

> "A supervised IDS labels the unknown as `normal`. We measured exactly how much — per attack
> family, with confidence intervals, at matched false-positive cost. We bought back 29 points of
> recall at 1% FPR on genuinely unseen attacks, and we'll tell you the operating points where it
> doesn't work.
>
> We also measured how much of our own headline AUC is a dataset artifact, and told you that too."

---

## Questions you will get

**"Why is your accuracy only ~95%?"**
> Accuracy on a 55%-attack test set is close to meaningless. Ask about recall at 1% FPR — 0.83 with
> the suspected artifact features removed.

**"Did you try deep learning?"**
> The novelty head is a neural autoencoder. We deliberately didn't ship a transformer: on 175k rows
> of tabular data, gradient-boosted trees win (Grinsztajn et al. 2022), and on eight CPU cores we
> couldn't afford the search to make one competitive. We'd have shipped an undertuned model that
> loses to XGBoost and proves nothing.

**"Would this work on a real network?"**
> Unknown, and we say so in the model card. These datasets are 10 and 28 years old and synthetic.
> What transfers is the relative comparison between methods and the methodology, not the absolute
> numbers.

**"Why doesn't the UNSW leave-one-family-out show a gain?"**
> Because the holdout held nothing out. Delete every `Backdoor` row and the model still detects 100%
> of Backdoor flows — the families overlap enough that removing a label doesn't remove the
> behaviour. We report it as a control, and it's why the NSL-KDD natural split is the load-bearing
> experiment.

**"What's the false positive rate?"**
> At the deployed threshold, and on which data? On held-out benign, 1% by construction. On the test
> set, 10.2% — because the operating point doesn't transfer between the two, which is measurable
> before deployment and is why the drift monitor exists.

---

## Do not

- Quote flows/second as link speed. A flow record summarises many packets.
- Quote a `Worms` recall point estimate. 44 test rows; give the interval.
- Quote the alerts→incidents ratio from any dataset without real source IPs.
- Claim the system catches zero-days. It surfaces statistically unusual traffic at a fixed budget.
