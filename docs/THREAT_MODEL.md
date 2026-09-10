# Threat model

Two systems are in scope: the network Penumbra defends, and **Penumbra itself**. The second half is the
part most projects skip, and it is where the interesting findings are — our own human-in-the-loop
design introduces a poisoning vector that did not exist before we added it.

Referenced against **MITRE ATT&CK** for the defended network and **MITRE ATLAS** for attacks against
the ML system.

---

## Part 1 — What Penumbra is expected to detect

### In scope

Network-observable attack behaviour visible in flow metadata: reconnaissance and scanning, denial of
service, exploitation of remote services, worm-like lateral movement, backdoor and C2 beaconing
patterns, and brute-force authentication attempts.

### Explicitly out of scope

| Not detected | Why |
|---|---|
| Anything inside encrypted payloads | metadata-only by design; no TLS interception, which is also a privacy choice |
| Host-resident activity | no endpoint telemetry — process creation, registry, file writes are invisible |
| Insider misuse of legitimate credentials | looks exactly like authorised use at the flow level |
| Low-and-slow activity below flow-aggregate resolution | the signal is averaged away; partially addressed by the sequence head, not solved |
| Application-layer logic abuse | requires payload and application context |
| Supply-chain and physical compromise | out of the sensor's reach entirely |

Publishing this list matters: a detector's coverage claim is only meaningful alongside its blind spots.

---

## Part 2 — Attacks against Penumbra

### T1 — Training-data poisoning through the analyst feedback loop

**Severity: high. This is the one we created ourselves.**

Analyst verdicts (true positive / false positive / benign-by-policy) feed an active-learning queue that
feeds retraining. A compromised analyst account — or a malicious insider — can therefore label their own
traffic benign repeatedly and teach the model to ignore it. The feedback loop that makes the system
improve is the same mechanism that makes it corruptible.

**Mitigations:**
- A verdict does not enter the training pool on an analyst's authority. **Senior approval is required
  for promotion**, and the two roles are separated.
- Full verdict provenance — who, when, from where — in the hash-chained audit log.
- **Canary gate:** a candidate model must clear a held-out evaluation set before promotion. A model
  taught to ignore one traffic pattern fails the canary.
- Per-account verdict rate limits, and anomaly detection over verdict patterns (an account whose
  false-positive rate diverges sharply from its peers is itself a signal).
- Champion/challenger — a new model shadow-scores before it replaces anything.

ATLAS: `AML.T0020` Poison Training Data · `AML.T0018` Backdoor ML Model.

### T2 — Evasion / adversarial examples

**Severity: high. Inherent to the problem.**

An attacker shapes traffic to score below threshold: padding bytes, injecting delay, stretching flow
duration, mimicking benign inter-packet timing.

**Mitigations:**
- Measured, not assumed — **problem-space constrained** evaluation only (Pierazzi et al., IEEE S&P
  2020). Feature-space ε-balls are meaningless here: `spkts = 3.2` does not correspond to any packet
  sequence that can exist on a wire. Perturbations must respect integrality, non-negativity and
  monotonicity, and derived features (`rate`, `sload`, `smean`) are recomputed from primitives rather
  than perturbed independently.
- The two-head design raises the cost: evading the supervised head by looking unlike known attacks
  tends to look *more* unusual to the benign-only novelty head. Evading both simultaneously is a
  tighter constraint than evading either.
- **Slow-rate mimicry** is measured explicitly — stretch a DoS flow's duration ×10, scale rate
  accordingly, report the detection collapse.

ATLAS: `AML.T0043` Craft Adversarial Data.

### T3 — Model inversion / attribution leakage

**Severity: medium.**

Feature attributions describe what makes traffic detectable, which doubles as a map of what an evader
must change. Repeated querying of a scoring endpoint allows threshold discovery.

**Mitigations:** attributions only inside authenticated, role-gated, audited interfaces; rate limiting
on `/score`; no public scoring endpoint; SHAP values never returned to an unauthenticated caller.

ATLAS: `AML.T0024` Exfiltration via ML Inference API · `AML.T0002` Acquire Public ML Artifacts.

### T4 — Alert-flooding as cover

**Severity: medium.**

Generate obvious, noisy attacks to saturate the queue while the real intrusion proceeds quietly.

**Mitigations:** alert-to-incident correlation collapses volumetric noise into a small number of
entities by design; the hunting lane's **fixed daily budget** means a flood cannot displace it; ranking
is by priority, not arrival order, so noise does not push signal off the first page.

### T5 — Compromise of the pseudonymisation key

**Severity: medium.**

IP pseudonymisation is a **keyed HMAC**. IPv4 has 2³² values, so anyone holding the key can enumerate
the entire mapping in seconds. This is pseudonymisation, not anonymisation, and the distinction is
legally load-bearing: under GDPR Art. 4(5) and Recital 26 the output remains **personal data**, and IP
addresses are personal data per *Breyer* (CJEU C-582/14).

**Mitigations:** key stored separately from the data it protects and never in the repository;
re-identification is an `admin`-only operation, audited on every use; the motivated-intruder test is
acknowledged rather than argued away; key rotation documented in `docs/RUNBOOK.md`.

### T6 — Audit log tampering

**Severity: medium.**

An attacker with database access rewrites history to erase their own actions.

**Mitigations:** the audit log is append-only and **hash-chained** — each entry commits to the digest of
the previous one, so any edit breaks the chain from that point forward. A test asserts that tampering
with any entry is detected.

### T7 — Supply chain

**Severity: medium.**

Compromise via a dependency, or a tampered model artifact.

**Mitigations:** `uv.lock` pins every transitive dependency; `pip-audit` runs on every push **and on a
weekly schedule**, because a dependency does not have to change to become vulnerable; secret scanning
over full history, since a key committed and later deleted is still published; model artifacts carry
SHA-256 in the registry; SAST via `bandit` and `semgrep`.

### T8 — Denial of service against the detector

**Severity: low-medium.**

Flood the scoring path to exhaust capacity and create a blind spot.

**Mitigations:** rate limiting; bounded queues that shed load predictably rather than collapsing;
measured throughput and p99 latency published so capacity is a known number rather than a hope;
degraded-mode behaviour is to alert on the backlog rather than to fail silent.

---

## Part 3 — Risks the architecture deliberately does not have

**Auto-blocking.** There is no enforcement path, so there is no way to induce Penumbra into blocking
chosen traffic by shaping input to look malicious. A detector wired to a firewall is an
attacker-controllable availability primitive; this one is not. See ADR-0001.

**Payload retention.** No packet contents are stored, so there is no payload corpus to breach.

**Public inference.** No unauthenticated scoring endpoint exists, so threshold discovery requires
credentials first.

---

## Part 4 — Testing scope and ethics

All evaluation uses public research datasets and traffic captured on hardware the team owns, on an
isolated host-only network. **No scanning, probing, or capture against any system we do not own.**
`penumbra.pcap` parses capture files read-only and never transmits a packet.

Full statement: `docs/ETHICS_SCOPE.md`.

---

## References

MITRE ATT&CK · MITRE ATLAS · Pierazzi et al., IEEE S&P 2020 · Axelsson 2000 · Arp et al., USENIX
Security 2022 · GDPR Art. 4(5), Recital 26 · *Breyer v Bundesrepublik Deutschland*, CJEU C-582/14
