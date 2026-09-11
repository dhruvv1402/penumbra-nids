# PENUMBRA

**ML-native network detection and response for the SOC.** Catches what the signatures miss — and measures
honestly how much it actually catches.

> Bennett University Hackathon 2026 · Microsoft track · Problem 26, *Catch the Attack the Signatures Miss*

The penumbra is the region between full shadow and full light. It is where novel attacks live: not matching a
known-bad signature, not looking like known-good either.

---

## The result

NSL-KDD's test set contains 17 attack types that never appear in its training set — 3,750 rows the
model has genuinely never seen. At a matched false-positive cost:

| realised FPR | supervised alone | + novelty head | delta |
|---|---|---|---|
| 1.0% | 0.053 | 0.346 | **+0.29** |
| 2.0% | 0.085 | 0.541 | **+0.46** |
| 10.0% | 0.864 | 0.769 | **-0.09** |

A supervised classifier recalls **5%** of attacks it has never seen. The novelty head recovers a
large part of that — but only in a band, and at 10% FPR it makes things *worse*. A novelty head is a
trade, not a free addition.

Full curve, plus the three methodological corrections that got us there (two of which produced
flattering numbers that were wrong): [`docs/EVALUATION.md`](docs/EVALUATION.md).

## The thesis

A supervised classifier does not catch novel attacks. Trained on known families, it labels an unseen family
`normal` — because "normal" is the only bucket it has for "nothing I recognise." Shipping a Random Forest and
calling it a zero-day detector is the same failure the problem statement describes, one level up.

So Penumbra runs **two heads**:

| | Known-threat head | Novelty head |
|---|---|---|
| Trained on | labelled attacks + benign | **benign only** |
| Answers | "which known family is this?" | "is this unlike anything I've seen?" |
| Fails when | the family is new — it says `normal` | benign traffic legitimately shifts |
| Emits | `KNOWN_ATTACK(family)` | `SUSPECTED_NOVEL` |

...and then **measures whether the second head actually helps**, per attack family, with confidence intervals,
**at a matched alert budget**. That last clause is the whole experiment: OR-ing in any extra detector raises
recall, so a comparison that does not hold alert volume constant is meaningless.

The hypothesis was written down in [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) and git-committed **before** the
run. Whatever the matrix says, it is a finding rather than a claim.

## Two lanes, because a 1–3% false-positive rate is a fact of life

An autoencoder over benign flow features runs at roughly 1.5–4% FPR on held-out benign traffic. At realistic
attack prevalence that is an unusable ticket queue — so novelty findings do not become tickets:

- **Known-threat lane** → the incident queue. High precision, SLA'd.
- **Unknown-behaviour lane** → a ranked **hunting queue with a fixed daily budget**. Its honest metric is
  Precision@k, not FPR. A fixed-budget queue cannot cause alert fatigue by construction.

## Alert fatigue, measured

On CICIDS2017's Thursday–Friday split, with real source IPs from the capture:

**30,780 alerts → 273 incidents.** 112.7 events each.

The largest: a DoS Hulk flood producing **29,562 flows arrives as one incident**. Without
correlation that is 29,562 tickets for one event a human understands in ten seconds.

This number is reported from CICIDS2017 and nowhere else, because it is the only dataset here with
real source identities. On UNSW-NB15 the correlator *raises* rather than grouping on identifiers we
invented.

## It alerts. It never blocks.

There is no blocking code path in this repository, and [a test fails the build if one appears](tests/). The
reasoning, including the blast-radius analysis, is [ADR-0001](docs/adr/0001-alert-not-block.md). At 1% FPR on a
busy link, auto-blocking takes the business offline faster than any attacker would.

Every alert carries a suggested containment action flagged `requires_analyst_approval`. A human decides.

---

## Honest evaluation

Three things this project refuses to do, each of which is the normal thing to do:

**1. Quote precision or PR-AUC without its prevalence.** UNSW-NB15's test set is ~55% attack; real networks run
1-in-10³ to 1-in-10⁵. Precision, PPV and PR-AUC all move with base rate, so those numbers on this data flatter the
model by orders of magnitude. Primary metrics here are the prevalence-invariant pair — **TPR and FPR** — plus
ROC-AUC and recall@fixed-FPR. PR-AUC is always printed with the prevalence it was computed at and its no-skill
baseline. `penumbra.eval.prevalence` re-weights measured rates to a stated operational prevalence and labels the
output **modelled, not measured**.

**2. Report a headline AUC without checking how much of it is a testbed artifact.** `penumbra.data.audit` fits a
one-feature decision stump per column and ranks by AUC. On UNSW-NB15 we expect `sttl` alone to reach ~0.9 —
attack and benign traffic were generated from different hosts, and TTL records that. Every headline number is
reported **with and without** the suspect features, and the second number is the one we stand behind.

**3. Give a point estimate below the estimation floor.** With 37k benign test rows, FPR = 0.1% is 37 false
positives and FPR = 0.01% is 3.7 — not estimable. `Worms` has 44 test rows, so its recall interval is roughly
±15 points. Those get intervals, not decimals.

Plus: stratified bootstrap CIs on every headline number, McNemar with discordant-pair counts (not just a p-value
that 82k samples make meaningless), calibration by Brier score and reliability diagram, and a self-audit against
**Arp et al. (2022), *Dos and Don'ts of Machine Learning in Computer Security*** — including the pitfalls we
still fail.

## About the data

UNSW-NB15 is synthetic IXIA traffic from 2015. NSL-KDD descends from a 1998 DARPA simulation. CICIDS2017's
original labels were buggy enough that a corrected re-release exists, and that is the one used here.

**These datasets are 10 and 28 years old and synthetic. The absolute numbers do not transfer to a real network.**
What transfers is the relative comparison between methods, and the methodology.

Each dataset does the job it is actually capable of:

| Dataset | Job | Why |
|---|---|---|
| UNSW-NB15 | tabular benchmark, imbalance ablation, LOAFO | 10 labelled families, modern flow features |
| NSL-KDD | the zero-day experiment | 17 attack types appear in test but not train — 3,750 rows, 16.6% of the test set |
| CICIDS2017 | correlation, graph features, temporal validity | the only one with source IPs and timestamps |

## Scope and ethics

Public datasets and traffic captured on hardware the team owns. No scanning, probing or capture against any
system we do not own. `penumbra.pcap` parses capture files read-only and never transmits a packet. See
[`docs/ETHICS_SCOPE.md`](docs/ETHICS_SCOPE.md).

---

## Quickstart

```bash
uv sync --extra eval --extra api --extra rag --extra drift

uv run penumbra data fetch                 # downloads + verifies SHA256 against data/manifest.json
uv run penumbra audit  --dataset unsw      # run this BEFORE trusting any model number
uv run penumbra eval   --dataset unsw      # honest metrics, with and without artifact features
uv run penumbra ablate --dataset unsw      # imbalance ablation on the natural distribution
uv run penumbra ablate --leakage-demo      # SMOTE before vs inside the fold, side by side
uv run penumbra loafo  --dataset nslkdd    # the unseen-17 experiment, across operating points
```

### The demo

```bash
uv run penumbra fit -d nslkdd              # train and save the two-head detector
uv run penumbra serve                      # API on :8000
cd console && npm install && npm run dev   # console on :3000
uv run penumbra replay -d nslkdd --ingest  # stream alerts into it
```

Sign in as `analyst`, `senior` or `admin` (password = username). The roles differ: an analyst sees
only its own network segments and can record a verdict but **cannot promote it into the training
pool** — that separation is what stops one compromised account from teaching the model to ignore
its own traffic.

Add `--inject-drift abrupt` to the replay to watch the drift monitor fire at a known change point.

Every number in the report regenerates from a command. None are typed by hand.

## Layout

| | |
|---|---|
| `src/penumbra/data/` | loaders, the dataset-trap registry, and the leak audit |
| `src/penumbra/models/` | supervised head, benign-only novelty head, fusion, calibration |
| `src/penumbra/eval/` | metrics, prevalence correction, matched budgets, LOAFO, cost curves |
| `src/penumbra/drift/` | PSI with frozen bins, KS with BH correction, ADWIN, drift injector |
| `src/penumbra/alerts/` | the canonical Alert, two-lane routing, correlation, ASIM/OCSF/ECS |
| `src/penumbra/rag/` | ATT&CK corpus and the cited triage copilot — fully offline |
| `src/penumbra/api/` | FastAPI, JWT, RBAC, hash-chained audit log, PII pseudonymisation |
| `console/` | Next.js SOC console |
| `docs/` | evaluation, model card, datasheet, threat model, ADRs, pre-registered experiments |

## Status

Phases 0–2 complete and pushed. 141 tests. Working: data pipeline with leak auditing, both detector
heads, the full honest-evaluation suite, imbalance ablation, drift detection and injection, the
alert contract with three SIEM serialisers, RBAC + audit + PII, the API, and the console.

Not yet built: conformal abstention, the sequence head, KQL rule mining, the pcap converter, and the
Sentinel solution files.
