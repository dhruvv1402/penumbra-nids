# Runbook

Operational procedures: retraining, drift response, key rotation, incident handling, and what to do
when the model is wrong.

---

## Retraining policy

### Triggers — retrain on a signal, not on a calendar

| Trigger | Threshold | Action |
|---|---|---|
| Feature drift | PSI ≥ 0.25 on any monitored feature, sustained over 3 windows | review, then retrain |
| Feature drift, moderate | 0.1 ≤ PSI < 0.25 | investigate; no automatic retrain |
| Prediction shift | alert rate deviates > 2× from the 30-day median | investigate first — usually a network change, not a model failure |
| Conformal coverage loss | empirical coverage falls below nominal − 5 points | strong drift indicator; review |
| Confirmed misses | any analyst-confirmed attack the model scored benign | root-cause before retraining |
| Feedback volume | ≥ 500 senior-approved verdicts accumulated | scheduled retrain |
| Elapsed time | 90 days with no other trigger | scheduled retrain |

PSI's 0.1 / 0.25 thresholds are a convention (Lewis 1994; Siddiqi), not a derived result, and they are
sample-size dependent. Treat them as a prompt to look, not as a verdict.

### Promotion — champion / challenger

1. Train the challenger on the extended dataset. Fixed seed, recorded.
2. **Canary gate** (`eval/canary.py`). On a frozen canary set, at each model's own deployed
   threshold: overall recall may not drop more than 0.02, no attack type with ≥ 20 canary rows may
   lose more than 0.10 recall, and benign FPR may not rise more than 0.01. The per-type check is the
   one that matters. A model poisoned to ignore one attack loses almost nothing on average, and in
   E7 its overall recall went *up*.
3. **Shadow scoring** (`penumbra registry shadow`). Run the challenger alongside the champion on live
   traffic for ≥ 7 days, scoring without alerting. Compare alert volume, agreement rate, and per-family recall on anything the
   analysts confirm.
4. **Promote** only with sign-off recorded in the audit log. The registry keeps both artifacts with
   SHA-256 digests.
5. **Rollback** is re-pointing the registry at the previous version — a config change, not a redeploy.
   Keep the previous two versions always.

### What is never done automatically

Promotion. Threshold changes. Suppression rule creation. Each requires a human, and each writes to the
audit log.

---

## Deploying on a new network

A detector trained on a public dataset does not know your network: on our own lab capture, every
flow alerted (EVALUATION §10.7h). Re-baseline before relying on the queues.

1. Record a window of traffic you believe is ordinary: a quiet week if you can, at least the size
   the tool asks for. At a 5% target that is 660 flows; at 1%, 3,327.
2. `penumbra rebaseline <window.pcap> --model unsw [--exclude <known-scanner>] --by <you>`.
   It refuses up front (R1) if the window is too short for the target, and says how long it needs
   to be. Otherwise it fits, checks the held-out false-positive rate (R2) and registers a
   **candidate** version with the report attached.
3. Read the report before promoting, especially the line saying how much of the window the current
   detector fires on. If it is much higher than you expect for this network, the window may contain
   an intrusion (THREAT_MODEL T9). Do not promote; record another window.
4. `penumbra registry promote <version> -d unsw --by <someone else>`. Rollback is the usual
   pointer change.

Re-baseline again after a large, deliberate change to the network (a new site, a new class of
service). Not on a schedule: see "Triggers" above.

**Same network, drifted: `--mode thresholds`.** When the realised FPR has walked away from the
target on a network the detector already knows, move the operating point instead of re-learning
normal. E8 measured both on NSL-KDD's shifted split: thresholds-only restored 1% from 10.2% and was
stable from 500 rows; re-learning normal was no better on unseen attacks and unstable below about
1,000 calibration rows. Expect recall to fall when the FPR is brought back down, since part of
what the drifted operating point caught, it caught by over-alerting. That is the honest operating
point, not a regression.

**If a re-baseline lowers the FPR by more than you asked for, look at the window before
promoting.** A window containing attack traffic raises the thresholds and looks exactly like
successful tuning (E8: at 5% contamination the FPR fell to 0.3% while unseen-attack recall fell by
0.235).

## Drift response

**Drift alarm fires — first question: did the network change, or did the model degrade?**

They look identical on a dashboard and have opposite responses.

1. **Check for a network event first.** New subnet, new application, a migration, a scanner deployment,
   an office reopening. Most PSI spikes are the business doing something, not the model failing.
2. **Identify which features moved.** `penumbra drift -d <dataset>` ranks by PSI with
   reference-frozen bins and Benjamini-Hochberg-corrected KS; the drift console page shows the
   ranking. **Then check the score PSI too** — on NSL-KDD no feature crosses 0.25 while the score
   distribution moves by 0.65 (EVALUATION §10.6). Note that ~2 features flag spuriously per window at α = 0.05, which
   is why the correction and the observed-vs-expected count exist.
3. **Check whether prediction distribution moved with it.** Feature drift without prediction drift is
   often benign. Both moving together is the concerning case.
4. **If a network change explains it:** update the reference window, do not retrain. Record the
   decision.
5. **If it does not:** collect analyst verdicts on recent alerts to get a labelled sample, measure
   actual degradation, and retrain if confirmed.

**A caveat that matters:** without labels you are detecting *covariate shift* and *prediction shift*.
True concept drift — P(y|x) changing — is not observable without ground truth. Labels arrive days late
or never. Unsupervised monitoring is the alarm; analyst verdicts are the confirmation.

---

## Alert triage

**Known-threat lane** (incident queue, SLA'd):
1. Read the plain-English attribution before the score. "312 distinct destination ports in 60s,
   typical is 3" is more useful than `p_attack = 0.94`.
2. Check the ATT&CK mapping **and its confidence tier**. `stretch` mappings are suggestions.
3. Read the cited copilot triage note (alert detail panel, or `GET /alerts/{id}/triage`). Follow
   the citations; do not trust the summary alone.
   To hand an alert to another tool, `GET /alerts/{id}/export?format=asim|ocsf|ecs` returns it in
   that schema, validated first.
4. Verdict: true positive → escalate; false positive → record why; benign-by-policy → create a
   suppression rule **with an expiry date**.

**Unknown-behaviour lane** (hunting queue, fixed daily budget):
- This lane is worked top-down until the budget is exhausted, then stopped. That is the design. An
  unworked item at position 400 is not a missed alert; it is a queue functioning as intended.
- Its metric is Precision@k. Track it — if the top 20 are consistently uninteresting, the novelty
  threshold or the reference window needs attention.

**`UNCERTAIN` (abstention lane):** the conformal prediction set was ambiguous. These are the items
where the model is explicitly declining to guess, and they are the highest-value labelling targets for
the active-learning loop.

---

## Suppression rules

- **Every suppression has an expiry date.** No exceptions. A permanent suppression is a permanent blind
  spot, and the vulnerability scanner you allowlisted in March is the C2 channel you missed in
  September.
- Requires `senior` role. Records who, why, and what it matches.
- Reviewed monthly; expired rules are not auto-renewed.
- A rule can be ended early (`POST /suppressions/{id}/revoke`, or *end now* on the feedback page).
  It is not deleted: its expiry moves to now and the revocation is audited.
- Suppressed events are still **stored and counted**, just not queued. Suppression is not deletion —
  you must be able to answer "what did we suppress last quarter?"

---

## Key rotation — PII pseudonymisation

The HMAC key is a secret. Anyone holding it can enumerate the entire IPv4 mapping in seconds.

1. Generate the new key; store it in the secret manager, never in the repository.
2. Open the overlap window: new key in `PENUMBRA_PII_HMAC_KEY`, the old one in
   `PENUMBRA_PII_HMAC_PREVIOUS_KEYS`. New alerts get new pseudonyms; entity lookups
   (`pii.pseudonyms_for`) and audited re-identification still accept the old key, so historical data
   stays queryable. `GET /governance/pii-keys` (admin) shows the fingerprints in force.
3. Re-pseudonymise retained data, or accept that pre-rotation data is only readable with the archived
   key.
4. Close the window: empty `PENUMBRA_PII_HMAC_PREVIOUS_KEYS` and restart. Old pseudonyms can no
   longer be re-identified from then on. The fingerprint check is written to the audit log.

Rotate on: suspected compromise, an operator with key access leaving, or annually.

Re-identification is `admin`-only and audited on every use. If the audit log shows re-identification
without a corresponding incident, that is itself an incident.

---

## If the model is compromised

Suspected training-data poisoning (see `THREAT_MODEL.md` T1):

1. **Roll back** to the last model version predating the suspect verdicts. Registry pointer change.
2. **Quarantine** the feedback pool from the suspect period; do not delete it — it is evidence.
3. **Audit** verdict provenance: which accounts, what patterns, what did they consistently mark benign.
4. **Re-evaluate** the current champion against the frozen canary set.
5. **Rebuild** the training pool from verified verdicts only.
6. Treat the analyst account compromise as its own security incident.

---

## Demo-day operations

Rehearsed, in this order:

1. Freeze the build well before the deadline. Tag it. Rehearse against the frozen tag, not against
   `main`. `penumbra demo` brings the whole stack up with both fixtures in one command.
2. `penumbra replay --from-fixture tests/fixtures/demo_alerts.json --ingest` pushes pre-scored
   alerts from checked-in JSON into the API. **No model inference, no dataset, no network beyond
   loopback.** This is the fallback.
3. Verify the full path works with the network disconnected.
4. Record a screen capture of the complete run as the last-resort fallback.
5. Build container images the night before, not on the day.

Three independent failure points on stage — WebSocket, container, live model — is three too many. The
fixture path removes all three.

---

## Commands

```bash
penumbra data fetch [--force]           # download + verify against manifest
penumbra audit --dataset unsw           # artifact audit; run before trusting any number
penumbra train --dataset unsw
penumbra eval  --dataset unsw           # writes artifacts/reports/eval_unsw.json
penumbra loafo --dataset unsw           # the pre-registered experiment
penumbra replay -d nslkdd --inject-drift abrupt   # abrupt | gradual | seasonal | evasion
penumbra serve                          # API + WebSocket
penumbra copilot build                  # ATT&CK STIX -> local corpus (after `data fetch -d attack`)
penumbra copilot ask "<what you see>"   # what the triage copilot would cite
penumbra registry init -d nslkdd        # the fitted model becomes the first (ungated) champion
penumbra retrain -d nslkdd              # challenger from PROMOTED verdicts + canary gate
penumbra registry shadow <version> -d nslkdd   # challenger beside champion, nothing alerts
penumbra registry list|promote|rollback|verify -d nslkdd
penumbra gate                           # ML regression gate against ci/baseline_nslkdd.json
penumbra correlate                      # CICIDS alert->incident correlation on real source IPs
penumbra poison-drill                   # E7: the feedback-loop poisoning drill
penumbra poison-drill --flags-only      # the integrity flags alone, no refits (minutes)
penumbra export-onnx -d nslkdd          # supervised head to ONNX; parity on every test row
penumbra rebaseline <window.pcap> --model unsw  # learn this network's normal; gated candidate, not promoted
penumbra rebaseline <window.pcap> --mode thresholds  # same network, drifted: move the operating point only
penumbra refit-drill                    # E8: thresholds vs re-learning normal, measured under drift
uv run python scripts/deploy_sentinel.py  # Sentinel workspace, DCR, parsers, workbook, rules (az login first)
penumbra drift -d nslkdd [--inject abrupt]  # per-feature PSI + BH-corrected KS
penumbra reproduce-all                  # regenerates every number in the report
```
