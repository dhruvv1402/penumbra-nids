# PENUMBRA

**ML-native network detection and response for the SOC.** Catches what the signatures miss — and measures
honestly how much it actually catches.

> Bennett University Hackathon 2026 · Microsoft track · Problem 26, *Catch the Attack the Signatures Miss*

**At a glance** (every number regenerates from a command; details in [`docs/EVALUATION.md`](docs/EVALUATION.md)):

| | |
|---|---|
| **The thesis, measured** | On 17 attack types never seen in training, a supervised classifier recalls **5%** at 1% FPR; adding a benign-only novelty head: **35%**. At 10% FPR the same head *costs* 9 points. Both reported. |
| **Alert fatigue** | Real CICIDS2017 source IPs: **94,115 alerts → 203 incidents**; 99.93% of attack flows reach an analyst. |
| **Real traffic** | nmap between two of our laptops: out of the box everything alerts; the novelty head re-baselined on the network's own traffic, with no labels, reaches **AUC 0.97**. |
| **Human in the loop, attacked** | One stolen analyst account poisons a thin attack type (0.84 → 0.19 recall), but only at full dose; a per-family peer check flagged every flip on two targets. Pre-registered. |
| **Honesty checks** | Leak audit, SMOTE-leakage demo, calibration that breaks under shift, a deep model that lost, 13 bugs found by review and fixed. |
| **Speed** | One flow scored in **12.6 ms** (was 139), ~11,000 flows/s batched, 11,000 alerts/s built. Flat-forest scorer, exact: 0 decisions changed, checked at every startup. |
| **Never blocks** | No enforcement code path, CI-enforced. Alerts a SOC; a human decides. |
| **Try it** | `uv run penumbra demo` - API, console and real incidents in one command, no dataset needed. |

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

**94,115 alerts → 203 incidents.** 463.6 events each, and 99.93% of attack flows reached an analyst.

The largest: a PortScan of **41,862 flows across 1,001 ports arrives as one incident**. Without
correlation that is 41,862 tickets for one event a human understands in ten seconds.

Almost none of it came through the supervised head. Thursday–Friday's attack families never appear
on Monday–Wednesday, so the classifier does not recognise them, and it says so: 91,543 of the alerts
are conformal abstentions routed for review. The abstention lane carried the detection; correlation
is what made 91,543 review items into 203 things to look at. `penumbra correlate` reproduces it.

This number is reported from CICIDS2017 and nowhere else, because it is the only dataset here with
real source identities. On UNSW-NB15 the correlator *raises* rather than grouping on identifiers we
invented.

## The human in the loop, attacked

Analyst verdicts retrain the model, so a stolen analyst account can teach it that its own traffic is
benign. We built that attack and measured it (E7, pre-registered, then replicated on a second target):

- One account clearing its own attack type took recall on it from **0.84 to 0.19**, but only by
  clearing *every* alert. At a quarter or a half of them, the honest verdicts outvoted it.
- A per-family comparison against peer analysts flagged **every flipped verdict at every damaging
  dose, on both targets, and never an honest analyst**.
- A per-family canary gate refused the poisoned model while an aggregate gate would have promoted it:
  overall recall went *up*, because the honest feedback on everything else improved the model at the
  same time.
- Honest feedback alone cut the false-positive rate from 10.2% to 4.3%.

Promotion needs a second senior (never the verdict's author), verdicts are rate-limited, and a model
reaches production only through a hash-verified registry with a gate report attached.

## Calibrated, until the data moves

Isotonic calibration cuts expected calibration error 4× (UNSW) and 13× (NSL-KDD) on held-out
training data. On each dataset's shifted test split it makes calibration **worse**. NSL-KDD's
shifted split also breaks conformal coverage by 30 points. So `p_attack` is labelled "calibrated on held-out
training data", and the label-free abstention rate is the monitored warning that it no longer is.

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
# optional: --extra pcap (penumbra pcap) · --extra onnx (export-onnx) · --extra dl (the E6 sequence head)

uv run penumbra data fetch                 # downloads + verifies SHA256 against data/manifest.json
uv run penumbra audit  --dataset unsw      # run this BEFORE trusting any model number
uv run penumbra eval   --dataset unsw      # honest metrics, with and without artifact features
uv run penumbra ablate --dataset unsw      # imbalance ablation on the natural distribution
uv run penumbra ablate --leakage-demo      # SMOTE before vs inside the fold, side by side
uv run penumbra loafo  --dataset nslkdd    # the unseen-17 experiment, across operating points
```

### The demo

```bash
uv sync --extra eval --extra api --extra rag --extra drift
cd console && npm install && cd ..
uv run penumbra demo                       # API + console + 3,154 alerts + 203 real incidents
```

Open http://localhost:3000. No trained model, no dataset download and no network needed: the demo
runs from checked-in fixtures. To score live instead, `penumbra fit -d nslkdd`, then `penumbra
serve` with `npm run dev` in `console/`, and stream with `penumbra replay -d nslkdd --ingest`.

Sign in as `analyst`, `senior` or `admin` (password = username). The roles differ: an analyst sees
only its own network segments and can record a verdict but **cannot promote it into the training
pool**. A senior can, but **never their own verdict**: promotion takes a second account. That
separation is what stops one compromised account from teaching the model to ignore its own traffic.

The feedback loop, end to end:

1. **Queue page:** record a verdict on any alert. *Benign by policy* opens a suppression form for
   seniors: a rule scoped narrower than a family, with an expiry of at most 90 days. Matching alerts
   are reclassified at ingest, stored, counted, and kept out of the queue.
2. **Feedback page:** pending verdicts with integrity flags, a *label next* queue ranked by model
   uncertainty, and the live suppression rules.
3. `penumbra retrain -d nslkdd` trains a challenger on promoted verdicts and runs the per-family
   canary gate. `penumbra registry shadow <v>` scores it beside the champion with nothing
   alerting. `penumbra registry promote <v>` is refused without a passed gate, and `rollback` is a
   pointer change.

Add `--inject-drift abrupt` to the replay to watch the drift monitor fire at a known change point.

Every number in the report regenerates from a command. None are typed by hand.

## Layout

| | |
|---|---|
| `src/penumbra/data/` | loaders, the dataset-trap registry, and the leak audit |
| `src/penumbra/features/` | preprocessing (benign-only for novelty), causal entity-graph and sequence windows |
| `src/penumbra/models/` | supervised head, benign-only novelty head, fusion, calibration, conformal, sequence, registry |
| `src/penumbra/eval/` | metrics, prevalence, matched budgets, LOAFO, canary gate, shadow, regression gate, E7 drill |
| `src/penumbra/drift/` | PSI with frozen bins, KS with BH correction, ADWIN, drift injector |
| `src/penumbra/alerts/` | the canonical Alert, two-lane routing, correlation, suppression rules, ASIM/OCSF/ECS |
| `src/penumbra/feedback/` | verdict integrity flags and the uncertainty-sampling labelling queue |
| `src/penumbra/rules/` | KQL/Sigma rule mining from forest leaves, validated on held-out data |
| `src/penumbra/pcap/` · `adversarial/` | pcap → flow features; problem-space evasion |
| `src/penumbra/rag/` | ATT&CK corpus and the cited triage copilot, fully offline |
| `src/penumbra/api/` · `storage/` | FastAPI, JWT, RBAC, hash-chained audit, PII pseudonymisation; SQLite behind a Protocol |
| `console/` | Next.js SOC console: queue, feedback, evaluation, drift, governance |
| `sentinel/` | ASIM parser, analytics rules, DCR, mined rule packs, overview workbook |
| `ci/` | the ML regression gate's committed baseline |
| `docs/` | evaluation, pre-registered experiments E1–E7, model card, datasheet, threat model, ADRs |

## Status

347 tests. CI runs lint, types, tests, architectural invariants, SAST, dependency audit, a
full-history secret scan, an image build and boot, ZAP DAST, and an ML regression gate on NSL-KDD.

Built: every phase in `docs/ROADMAP.md`. The supervised head also exports to ONNX with exact
parity. Not built, and deliberately so: OpenTelemetry
spans (Prometheus `/metrics` exists), Postgres (SQLite behind the `Repository` Protocol), and Grafana,
all on the cut list. A real capture from our own laptops is measured
(EVALUATION §10.7h). Not done: Sentinel against a live workspace (the solution files exist; running
them needs Azure credentials).
