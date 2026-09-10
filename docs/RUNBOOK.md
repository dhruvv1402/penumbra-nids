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
2. **Canary gate.** The challenger must clear a frozen held-out evaluation set and beat the champion's
   committed PR-AUC floor. A model poisoned to ignore a traffic pattern fails here.
3. **Shadow scoring.** Run the challenger alongside the champion on live traffic for ≥ 7 days, scoring
   without alerting. Compare alert volume, agreement rate, and per-family recall on anything the
   analysts confirm.
4. **Promote** only with sign-off recorded in the audit log. The registry keeps both artifacts with
   SHA-256 digests.
5. **Rollback** is re-pointing the registry at the previous version — a config change, not a redeploy.
   Keep the previous two versions always.

### What is never done automatically

Promotion. Threshold changes. Suppression rule creation. Each requires a human, and each writes to the
audit log.

---

## Drift response

**Drift alarm fires — first question: did the network change, or did the model degrade?**

They look identical on a dashboard and have opposite responses.

1. **Check for a network event first.** New subnet, new application, a migration, a scanner deployment,
   an office reopening. Most PSI spikes are the business doing something, not the model failing.
2. **Identify which features moved.** `penumbra drift report --since <window>` ranks by PSI with
   Benjamini-Hochberg correction. Note that ~2 features flag spuriously per window at α = 0.05, which
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
3. Read the cited copilot triage note. Follow the citations; do not trust the summary alone.
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
- Suppressed events are still **stored and counted**, just not queued. Suppression is not deletion —
  you must be able to answer "what did we suppress last quarter?"

---

## Key rotation — PII pseudonymisation

The HMAC key is a secret. Anyone holding it can enumerate the entire IPv4 mapping in seconds.

1. Generate the new key; store it in the secret manager, never in the repository.
2. Both keys are accepted during a defined overlap window — historical data stays queryable.
3. Re-pseudonymise retained data, or accept that pre-rotation data is only readable with the archived
   key.
4. Retire the old key; record the rotation in the audit log.

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
   `main`.
2. `penumbra demo --from-fixture` replays pre-scored alerts from checked-in JSON. **No model
   inference, no network, no database.** This is the fallback.
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
penumbra eval  --dataset unsw --report
penumbra loafo --dataset unsw           # the pre-registered experiment
penumbra drift report --since 7d
penumbra replay --inject-drift          # drift injector for the live demo
penumbra serve                          # API + WebSocket
penumbra reproduce-all                  # regenerates every number in the report
```
