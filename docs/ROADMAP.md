# Roadmap

Phased so that **every phase ends with a working demo**. No phase may leave the system broken, and
`v0.1-baseline` is a complete, submittable answer to the brief on its own.

Team: 3 developers across four tracks. The API contract and the canonical `Alert` object are frozen in
Phase 0 precisely so the tracks can move independently from Phase 2 onward.

| Track | Scope |
|---|---|
| **A — ML & evaluation** | features, model stack, imbalance ablation, LOAFO, calibration, conformal, adversarial |
| **B — Backend & integrations** | FastAPI, auth/RBAC/audit, correlation, replay + drift injector, ASIM/OCSF/ECS emitters, rule mining, pcap |
| **C — Console** | the five pages, WebSocket, charts, design system |
| **D — Governance, CI, demo** | docs, CI/CD + SAST/DAST + ML regression gate, Sentinel Solution packaging, copilot, demo |

---

## Phase 0 — Foundations and the leak audit ✅

Repo skeleton, `uv` project on Python 3.12, dataset manifest with SHA-256 verification, typed loaders,
`data/schema.py` encoding every known dataset trap, the pydantic API contract plus a checked-in alert
fixture, and **`data/audit.py`**.

The audit runs **first**, before any model exists, because it constrains what we are allowed to claim
afterwards.

## Phase 1 — The simple version that runs

Load UNSW-NB15 → LogisticRegression + RandomForest → per-class precision/recall/F1, FPR, ROC-AUC,
PR-AUC with prevalence annotation, confusion matrix, stratified bootstrap CIs → alerts to JSONL.
One command. No UI, no services.

**Tagged `v0.1-baseline`.** Everything after this is upside.

## Phase 1.5 — LOAFO smoke test ⚠️ do not defer

NSL-KDD unseen-17 first, then one held-out UNSW family, with a crude autoencoder.

This answers *"does the headline claim hold?"* early rather than the night before the deadline. If both
come back flat, the headline pivots to "an honest measurement of the limits of supervised intrusion
detection" and the evaluation-rigour results carry the presentation. That decision needs to be
available while there is still time to act on it.

## Phase 2 — Novelty head and the LOAFO harness — *critical path*

Benign-only preprocessing pipeline, autoencoder + IsolationForest + Mahalanobis, rank-normalised score
fusion with a single threshold, the verdict lattice.

**`eval/budget.py` is built before `eval/loafo.py`.** Without a matched alert budget the entire
comparison is void.

Then the full 9-family matrix: binary recall, family attribution, `SUSPECTED_NOVEL` rate, bootstrap CIs.

## Phase 3 — Alerts, correlation, console — *parallel from Phase 1*

Alert and Incident models, correlation on CICIDS2017 (source IP + family + time window), SQLite behind
a `Repository` Protocol, replay engine, WebSocket stream, incident queue and detail pages with SHAP
waterfall, ATT&CK chip and verdict buttons.

## Phase 4 — Imbalance, calibration, conformal — *parallel*

The full ablation, always evaluated on the natural distribution and always inside CV folds. The
deliberate leakage demonstration. Isotonic calibration with Brier and reliability diagram. Mondrian
conformal with the abstention lane. Prevalence-corrected reporting. Cost curve with the operating
point chosen from it.

## Phase 5 — Drift

PSI with fixed reference bins, KS with Benjamini-Hochberg, JS divergence, ADWIN and Page-Hinkley on the
streaming score, prediction-distribution monitoring. The **drift injector** — distribution shift plus a
new family mid-stream — so the monitor visibly fires during the demo. Conformal-coverage degradation
overlaid on PSI, tying Phase 4 to Phase 5.

## Phase 6 — Enterprise — *parallel, backend track*

JWT + RBAC (analyst / senior / admin; only senior closes an incident **and** only senior promotes a
verdict into the training pool), hash-chained audit log, HMAC PII pseudonymisation with role-gated
re-identification, `/metrics`, OpenTelemetry, CI with the ML regression gate, Docker Compose, model
registry.

## Phase 7 — RAG copilot

ATT&CK STIX reduced to a compact JSONL at build time. BM25 retrieval — no embedding model, no download,
and it cannot hallucinate, which is better for a demo. Hand-curated family→ATT&CK map (**never** RAG
output — a wrong technique ID in front of a security judge is fatal). Template-filled cited triage
notes, with the LLM path behind an interface and off by default.

## Phase 8 — Differentiators ✅ (mostly)

- ✅ **KQL/Sigma rule mining**, every rule validated on held-out data (purity ≥ 0.98, support ≥ 50).
  `penumbra rules`. 95 rules catch 66.3% of UNSW test attacks at 0.966 precision with no model —
  and 106 of the 207 mined paths were discarded because they rested on quarantined features.
- ✅ **Problem-space adversarial evaluation** with slow-rate mimicry, plus the unconstrained
  feature-space strawman run alongside it so the gap is measured. `penumbra adversarial`.
- ✅ **Sequence head on CICIDS2017** with causal windows, and ✅ **entity-graph features**.
  `penumbra sequence`, pre-registered as E6 in EXPERIMENTS.md.
- ✅ **pcap → flow converter**, read-only, with the payload-dependent features named rather than
  guessed. `penumbra pcap`.
- ✅ **`reproduce-all`**, which reports SKIPPED with a reason rather than passing quietly.
- ✅ **Console**: queue, evaluation, drift and governance pages, all reading measured reports.
- ⬜ ONNX export · ⬜ load test · ⬜ Docker Compose · ⬜ ZAP DAST
- ⬜ Sentinel connector against a live workspace (the Protocol and the mock exist; the plan was
  always mock-first, and a live run needs credentials).
- ⬜ A real lab capture. The converter works and is tested on synthetic captures; no capture from
  owned hardware has been taken yet.

## Phase 9 — Governance, docs, rehearsal — **never cut**

Model card and datasheet finalised with real numbers, `EXPERIMENTS.md` results, Arp et al. self-audit,
governance page. **Fixture-mode demo fallback, recorded screen capture, two full rehearsals.**

Reserve the final hours for this.

---

## Cut list, in order

Drop from the top if time collapses:

1. Grafana · 2. ONNX export · 3. FT-Transformer · 4. graph feature head · 5. sequence head ·
6. Postgres (stay on SQLite) · 7. load test · 8. LLM-backed RAG (keep BM25) · 9. Sigma emission (keep
KQL) · 10. adversarial robustness · 11. active-learning retraining (keep the feedback store + queue) ·
12. champion/challenger (keep the written policy) · 13. OpenTelemetry (keep `/metrics`) ·
14. governance *page* (keep the markdown)

**Conformal is cut last of the optionals** — it is a genuine differentiator and the
coverage-under-drift result ties two workstreams together.

### Never cut

The working simple version · honest metrics with CIs · the imbalance ablation · drift monitoring that
visibly fires · alert-not-block as an enforced ADR · LOAFO **or** NSL-KDD unseen-17 · **the artifact
audit and base-rate honesty slide** · incident correlation measured on real IPs · fixture-mode fallback
and rehearsal.
