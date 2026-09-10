# ADR-0001 — Penumbra alerts. It does not block.

**Status:** accepted · **Date:** 2026-09-11

## Context

An intrusion detector that can act has two options when it fires: raise an alert for a human, or block
the traffic itself. Blocking is superficially attractive — it closes the gap between detection and
response, and demos well.

The brief asks for alerts to a SOC rather than auto-blocks. This ADR records that we agree, and why,
so that the decision survives contact with the first person who asks "why not just block it?"

## Decision

**There is no blocking code path in this repository.** No firewall client, no ACL writer, no
`iptables`/`netsh`/NSG call, no quarantine action, no session reset. Not disabled, not feature-flagged
off — absent.

Alerts carry a *suggested* containment action in a field explicitly named
`requires_analyst_approval`. The suggestion is text. Nothing executes it.

A CI test walks the source tree for blocking-shaped call sites and **fails the build** if one appears.
The architectural claim is enforced mechanically rather than by convention, because "we decided not to"
degrades over a codebase's life and "the build breaks" does not.

## Rationale

### 1. The arithmetic

At an operational prevalence of 1e-4 on a million flows per day — 100 genuine attack flows — a 1% false
positive rate produces roughly 10,000 false alerts. Auto-blocking on that signal severs ten thousand
legitimate connections a day. The detector becomes a more effective denial-of-service tool than
anything it is defending against, and the outage is self-inflicted, continuous, and hard to attribute.

Our novelty head runs at 1.5–4% FPR on held-out benign traffic by design (see ADR-0002). Wiring that to
an enforcement point is not a tuning problem. It is a category error.

### 2. Blast radius is asymmetric

A false negative costs dwell time on one intrusion. A false positive that blocks costs availability for
whatever it blocked — and the traffic most likely to look statistically anomalous is often the most
operationally important: backup windows, replication, batch transfers, vulnerability scanners,
monitoring agents, a new deployment. The failure mode is biased toward severing the things a business
notices within minutes.

### 3. An attacker can weaponise the enforcement point

If the model blocks, an adversary who can shape traffic to resemble a detection can induce the
defender to block chosen destinations. Auto-blocking converts the detector into an attacker-controllable
availability primitive. This is a known pattern against reactive defences and is the reason mature
deployments keep enforcement manual or heavily rate-limited.

### 4. Novel detections are exactly the wrong thing to automate

The system's purpose is surfacing traffic it has *never seen before*. Confidence is lowest precisely
where the product claims most value. Acting automatically on the least-understood signal inverts the
correct risk posture.

### 5. Regulation says a human decides

EU AI Act **Article 14** requires high-risk AI systems — Annex III includes critical infrastructure — to
be designed so that a natural person can effectively oversee them, interpret their output, and
**override or halt** them. NIST AI RMF 1.0 assigns the same responsibilities under GOVERN 3.2 and
MEASURE 2.8.

Alerting is not merely a design preference here. It is the compliance posture.

## Consequences

**Accepted:**
- Mean time to respond depends on analyst availability. Penumbra shortens time-to-*decision* (ranked
  incidents, plain-English attributions, cited triage notes) rather than removing the decision.
- A genuine attack can proceed while a human triages it. That is the cost, and it is the correct cost.

**Gained:**
- No self-inflicted outage path exists.
- No attacker-controllable enforcement primitive exists.
- The analyst feedback loop has something to feed on — verdicts on alerts a human actually reviewed.
- The claim "we never auto-block" is verifiable by reading the code, not by trusting the README.

## Alternatives considered

**Auto-block above a high confidence threshold.** Rejected. A threshold high enough to be safe fires so
rarely that the automation is worthless, and calibration drifts — the threshold that was safe last month
is not safe today, silently.

**Auto-block only known-family detections, never novelty.** Rejected, though it is the strongest
alternative. It would still require an enforcement integration to exist in the codebase, and the
existence of that path is most of the risk. A future deployment can add it downstream of the alert
stream, where it is somebody's explicit, audited decision rather than our default.

**Human-approved one-click block in the console.** Deferred, not rejected. It preserves the human
decision. It is out of scope because it requires an enforcement integration we will not build for a
hackathon, and shipping a half-tested firewall client is worse than shipping none.

## Related

ADR-0002 (two-stage detection), `docs/THREAT_MODEL.md`, `docs/MODEL_CARD.md` §Intended Use — which
lists automated blocking, automated firewall rule insertion, and use as the sole basis for account
suspension as explicitly **out of scope**.
