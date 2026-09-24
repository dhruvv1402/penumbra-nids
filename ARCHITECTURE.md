# Architecture

## The problem this shape solves

A signature IDS misses novel attacks. That is the premise of the brief. The tempting answer — train a
supervised classifier on labelled attack traffic — reproduces the same failure one level up: a model
trained on known families has no bucket for "something I have never seen," so it assigns unseen attacks
to `normal` with high confidence.

Penumbra therefore runs two detectors with different failure modes, and spends most of its evaluation
budget measuring whether the second one actually helps.

```
              flows  (dataset replay, or a pcap captured on hardware we own)
                                        │
                        ┌───────────────┴────────────────┐
                        │  feature pipeline              │
                        │  · per-flow, dataset-native    │
                        │  · entity-graph  (CICIDS only) │
                        │  · IP pseudonymisation         │
                        └───────────────┬────────────────┘
                                        │
              ┌─────────────────────────┼─────────────────────────┐
              │                         │                         │
    ┌─────────┴─────────┐   ┌───────────┴──────────┐   ┌──────────┴─────────┐
    │  KNOWN-THREAT     │   │  SEQUENCE            │   │  NOVELTY           │
    │  LogReg · RF ·    │   │  1D-CNN + BiGRU      │   │  autoencoder       │
    │  XGBoost          │   │  (CICIDS only)       │   │  + IsolationForest │
    │                   │   │                      │   │  + Mahalanobis     │
    │  labelled data    │   │  causal windows      │   │  BENIGN ONLY       │
    └─────────┬─────────┘   └───────────┬──────────┘   └──────────┬─────────┘
              │                         │                         │
              └─────────────────────────┼─────────────────────────┘
                                        │
                        ┌───────────────┴────────────────┐
                        │  FUSION                        │
                        │  p_attack           calibrated │
                        │  novelty_percentile  rank-norm │
                        │  priority            sort key  │
                        │  conformal set      (Mondrian) │
                        └───────────────┬────────────────┘
                                        │
                 ┌──────────────────────┴──────────────────────┐
                 │                                             │
       ┌─────────┴──────────┐                      ┌───────────┴───────────┐
       │  KNOWN-THREAT LANE │                      │ UNKNOWN-BEHAVIOUR LANE│
       │  incident queue    │                      │ hunting queue         │
       │  SLA'd, high prec. │                      │ fixed daily budget    │
       │  metric: precision │                      │ metric: Precision@k   │
       └─────────┬──────────┘                      └───────────┬───────────┘
                 └──────────────────────┬──────────────────────┘
                                        │
                        ┌───────────────┴────────────────┐
                        │  correlate · suppress · rank   │
                        │  SHAP → plain English          │
                        │  ATT&CK map · cited RAG note   │
                        └───────────────┬────────────────┘
                                        │
        ┌───────────────┬───────────────┴───────┬────────────────────┐
        │  SOC console  │  ASIM / OCSF / ECS    │  mined KQL rules   │
        └───────┬───────┴───────────────────────┴────────────────────┘
                │
      analyst verdict  (true positive / false positive / benign-by-policy)
                │
                └──→ senior approval ──→ training pool ──→ champion/challenger

      There is no edge from this diagram to a firewall. See ADR-0001.
```

## Why two heads

|  | Known-threat head | Novelty head |
|---|---|---|
| Trained on | labelled attacks + benign | **benign only** |
| Question | "which known family is this?" | "is this unlike anything I've seen?" |
| Fails when | the family is new — it answers `normal` | benign traffic legitimately shifts |
| Emits | `KNOWN_ATTACK(family)` | `SUSPECTED_NOVEL` |
| Expected FPR | low, tunable | 1.5–4% on held-out benign |

The novelty head's scaler and encoders are fitted on benign rows **only**, separately from the
supervised pipeline. Sharing a preprocessing pipeline fitted across all data would leak the attack
distribution into a model whose entire claim is that it has never seen an attack.

## Why two lanes

An autoencoder at 1.5–4% FPR cannot produce tickets. At an operational prevalence of 1e-4, a 1% false
positive rate on a million flows a day buys 10,000 false alerts against roughly 100 true ones. That is
the base-rate fallacy (Axelsson, 2000), and it is the reason signature IDSs still exist.

So novelty findings never become tickets. They become a **ranked hunting queue with a fixed daily
budget**, whose honest metric is Precision@k rather than FPR. A queue with a fixed budget cannot cause
alert fatigue by construction. The known-threat lane keeps the SLA and the incident queue.

This is the difference between hiding the false positive rate and designing around it.

## Three numbers, not one risk score

You cannot blend a calibrated probability with an anomaly score and still call the result calibrated.
Calibration is a property of probabilities. So the system emits:

| Field | Meaning | Calibrated? |
|---|---|---|
| `p_attack` | P(attack \| x) from the supervised head, isotonic-calibrated on a held-out split | **yes** — Brier score and the reliability diagram describe this and only this |
| `novelty_percentile` | "more unusual than 99.7% of known-benign traffic" — rank against the benign reference | n/a, it is a percentile |
| `priority` | 0–100 triage sort key | **no.** Documented as an ordering, never as a probability |

Novelty detectors are fused by **rank-normalising each against the benign reference and thresholding
once**, not by OR-ing three independently thresholded detectors — that would stack to roughly 3%
combined FPR and destroy any calibration claim. Raw reconstruction error and
`IsolationForest.decision_function` are on incomparable scales; percentiles are not.

## Verdict lattice

```
KNOWN_ATTACK(family)   supervised head recognised a trained family
SUSPECTED_NOVEL        novelty fired where supervised saw nothing
UNCERTAIN              conformal set ambiguous → routed to human review
BENIGN
BENIGN_BY_POLICY       matched an analyst-authored suppression rule
```

`UNCERTAIN` is the abstention lane. Mondrian (class-conditional) conformal prediction gives a coverage
guarantee under exchangeability; temporal splits and drift violate exchangeability, so we state the
caveat and then **measure empirical coverage degrading as PSI rises** — which turns the caveat into a
second drift signal.

## Dataset roles

Each dataset does the job it is actually capable of. See ADR-0004.

| Dataset | Role | Why it and not the others |
|---|---|---|
| **UNSW-NB15** | tabular benchmark, imbalance ablation, LOAFO | 10 labelled families, modern flow features, 175k/82k |
| **NSL-KDD** | the zero-day experiment | 17 attack types in test but not train — 3,750 rows, 16.6% of test |
| **CICIDS2017** | correlation, graph features, sequence head, temporal split | the only one carrying source IPs and timestamps |

The UNSW-NB15 split CSVs have **no IP addresses and no timestamps** — those exist only in the full
four-part release. Entity-graph features, per-host sequence windows, source-IP alert correlation and
temporal splitting are therefore impossible on that dataset, and any "alerts collapsed into incidents"
figure computed from fabricated IPs would be a fabricated metric. Those capabilities live on
CICIDS2017 and are measured there.

Related: UNSW's `ct_*` columns are already entity-window aggregates computed over a 100-connection
window by the original pipeline, so a graph head on that dataset would recompute what is already in
the table.

## Module boundaries

```
src/penumbra/
  data/         loaders/ · schema (every dataset trap encoded) · manifest · download · audit (leaks)
  features/     preprocess (benign-only for novelty) · entity_graph · windows (causal)
  models/       supervised · novelty/ · sequence · fusion · calibration · conformal · detector
                registry · onnx_export
  eval/         metrics · prevalence · budget · bootstrap · cost · loafo · runner · loadtest
                canary · shadow · retraining · regression (CI gate) · poisoning (E7) · sequence_experiment
  imbalance/    strategies · ablation
  drift/        detectors (PSI with frozen bins, KS + BH, JS) · streaming (ADWIN, Page-Hinkley) · injector
  explain/      attack_map (hand-curated)
  rules/        mining (leaf extraction) · emit (KQL/Sigma) · runner (validate + compare)
  rag/          corpus · copilot (BM25, cited, offline)
  alerts/       models · scoring · builder · correlate · suppression · schemas/{asim,ocsf,ecs}
  feedback/     integrity (verdict flags) · active (uncertainty-sampling queue)
  integrations/ siem/{base,mock,sentinel}
  api/          app · reports · security/{auth,rbac,pii,audit}
  storage/      repository (Protocol) · sqlite
  replay/       engine (live replay, fixture fallback, drift-injected replay)
  adversarial/  evasion (problem-space constrained; the strawman runs alongside it)
  pcap/         assemble  (read-only; never transmits)
```

Enforced by a grep-based import check in CI (`.github/workflows/ci.yml`, job `invariant`):

- `models/`, `eval/` and `rules/` never import `api/`, `alerts/` or `storage/`. The science must
  run headless, and a mined rule pack has to be reproducible without the product layer.
- `api/` uses storage through the `storage.repository.Repository` Protocol. The one concrete
  reference is where `AppState` constructs the SQLite backend. This is a convention, not CI-enforced.
- Everything SIEM-facing goes through `integrations.siem.base.SiemConnector`. `LocalMockSiem` is the
  default binding; `AzureSentinelSiem` raises `NotConfigured` until credentials exist.
- `explain/attack_map.py` is a hand-written constant table with a justification per entry. A wrong
  ATT&CK technique ID in front of a security judge is fatal, so that mapping is never RAG output.

## Causality, and the leak it prevents

Three modules compute a row's features from that row's *past*: `features/entity_graph`,
`features/windows`, and the `ct_*` counters in `pcap/assemble`. All three read before they write —
the current flow is measured against its history and only then joins it.

This is the easiest place in the whole system to fake a result. A window centred on the current
flow, or a fan-out count that includes the flow doing the fanning, scores better and cannot be
deployed, because at inference time the future has not happened yet. Nothing crashes when it goes
wrong; the number just improves.

So it is asserted rather than documented: appending future traffic must leave every earlier row's
features bit-identical (`tests/unit/test_sequence.py`, `tests/unit/test_pcap.py`).

## Feature scaling in the sequence head

Median and IQR, clipped to 10 robust deviations — not mean and standard deviation.

CICIDS2017's flow features reach 7.2e9, and several columns have a maximum more than 200 standard
deviations from their own mean. Z-scaling leaves a handful of rows at +200 and everything else
squashed against zero, and +200 through two convolutions into a GRU saturates the network. The
first run of the sequence experiment did exactly that: validation ROC-AUC 0.9999 in epoch one, then
exactly 0.500 for every epoch after. Early stopping restored the epoch-one weights and the run
exited cleanly with a number attached, which is the dangerous form of the failure.

`TrainingHistory.collapsed` now flags a constant-output run so it cannot be reported as an
architecture result.

## Storage and deployment

SQLite behind a `Repository` Protocol, Postgres only if the platform phase completes. Fewer moving
parts on demo day; swapping backends is one file, not a refactor.

Models are trained **on the host**, never inside Docker, and versioned artifacts are bind-mounted into
the image. Training in a container on a 16 GB machine that is also running Next, FastAPI, and a
browser is how a demo dies.

## The feedback loop

```
 analyst verdict ──► verdict_queue ──► second senior promotes ──► promoted_training_rows
  (alert or incident;   (integrity flags    (two-person rule, in the        │
   rate-limited,         shown, never        SQL UPDATE; audited)           ▼
   audited)              dropped)                                  eval/retraining.augment
                                                                            │
 benign_by_policy ──► suppression rule (scoped, ≤ 90 days)                  ▼
                      applied at /ingest; reclassify, never drop    challenger fit ──► canary gate
                      NEVER a training label                        (per attack type) │
                                                                                      ▼
                        registry: hash-verified load ◄── promote (CLI, gate required) ◄── shadow
                                  rollback = pointer change, refuses a tampered target
```

Every arrow that changes what the model learns needs a second human, and every one writes to the
audit chain. E7 measured the attack this whole path exists to resist; see `docs/EXPERIMENTS.md`.

## Security posture of the system itself

The analyst-feedback loop is a poisoning vector we introduced ourselves: a compromised analyst account
can teach the model that its own traffic is benign. Mitigations are in `docs/THREAT_MODEL.md` —
a second senior's approval before a verdict enters the training pool (never the recorder's own),
verdict provenance in the hash-chained audit log, integrity flags on pending verdicts, a per-family
canary gate before promotion, and per-account rate limits. Each is implemented, and E7 measures
which of them actually catch the attack. Referenced against **MITRE ATLAS**, not just ATT&CK.

IP addresses are pseudonymised with a keyed HMAC. That is pseudonymisation, not anonymisation — IPv4
is 2³² values and trivially enumerable by anyone holding the key — so under GDPR Art. 4(5) the output
is still personal data, and re-identification is a role-gated, audited call.
