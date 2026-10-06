# ADR-0006 — A rate-limited live scoring endpoint

**Status:** accepted · **Date:** 2026-10-06 · **Supersedes:** the "no scoring endpoint at all"
mitigation in THREAT_MODEL T3

## Context

Until now the API never scored anything. It displayed alerts that were scored offline and
ingested, and THREAT_MODEL T3 listed the absence of a scoring endpoint as a mitigation. A public
scorer is an oracle: anyone who can query it enough can find its thresholds and walk a malicious
flow down to just below them.

The product's demo, though, has to let a visitor run the trained models, not only watch a recording
of them. The decision is how to do that without handing out the oracle T3 worried about.

## Decision

`/score/sample`, `/score/csv` and `/score/pcap` run the registered champions (UNSW v003, NSL-KDD
v001, CICIDS v001) on, respectively:

- held-out test flows from a bundled pool, with the true label shown beside each;
- an uploaded CSV in a model's schema;
- an uploaded capture, assembled into flows and scored by the UNSW champion.

The guards that replace "no endpoint":

1. **Permission.** `model:score`, held by every role including the read-only guest, so a judge can
   use it from "View as guest".
2. **Rate limit per caller.** Guests: 30 calls per hour. Signed-in users: 120.
   - The key is role + username + client address (`X-Forwarded-For` behind the ingress), because
     every guest shares one identity.
   - Refusals are 429s, and each is audited.
3. **Row caps.** A guest gets at most 500 rows scored per sample or CSV call and 500 per-flow rows
   back from a capture; signed-in users get 5,000. A capture's *summary* covers all of it, up to
   20,000 flows.
4. **Rounded scores.** `p_attack` and the novelty percentile come back to two decimals, so a
   threshold cannot be bisected to the last digit within the call budget.
5. **Audit.** Every call is logged in the hash-chained audit log: mode, dataset, flows, alerts,
   model version and address.
6. **Nothing is stored.**
   - Results go back to the caller only.
   - The alert queue, its verdicts and the training pool never see a scored upload, so an upload
     cannot reach retraining (T1) and cannot disturb the pre-recorded demo queue.
7. **Uploads are bounded.** CSV ≤ 5 MB and capture ≤ 8 MB, checked on `Content-Length` before the
   body is read and again after. A capture that does not parse is a 422 that names nothing about
   the parser.
8. **Still alerts, never blocks.** The response is verdicts and reasons. ADR-0001 is untouched, and
   the CI check for enforcement code paths still runs.

## Consequences

- **The oracle exists, bounded.** At 30 calls of 500 rows an hour per address, with scores to two
  decimals, threshold discovery is slow and loud rather than impossible.
  - A determined attacker with many addresses gets further.
  - The audit log is how they are seen, and the deployment can turn guest scoring off
    (`PENUMBRA_GUEST_ACCESS` unset) without a code change.
- **The deployment image carries the champions** (about 225 MB) and a 4,000-row labelled pool per
  dataset. The API container goes from 2 to 4 GiB, since UNSW alone is about 750 MB once loaded.
- **Captures are parsed on a public server.** The parser is `dpkt` on bounded input in a worker
  thread. This is new attack surface, accepted for the demo, and listed in THREAT_MODEL T3.
- **Live captures from a network the model was not trained on mostly alert.** Out of the box, the
  UNSW champion flags almost every flow of our own lab capture. The scoring page says so and points
  at `penumbra rebaseline` (EVALUATION §10.7h), rather than letting a visitor conclude the model is
  either magic or broken.
