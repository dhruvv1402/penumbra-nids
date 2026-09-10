# Testing scope and ethics

## The rule

**We test only systems we own.**

No scanning, probing, exploitation, or traffic capture against any host, network, or service that is
not ours. Not a "safe-looking" public target, not a friend's server, not a university system, not a
site with a bug bounty we have not read.

The hackathon brief states this rule explicitly for the cybersecurity track. We state it here
unprompted because a network security project that cannot articulate its own boundaries has not
understood the field.

---

## What we actually use

### Public research datasets

All model training and evaluation uses published academic datasets, downloaded from their maintainers
or documented mirrors, with citation:

| Dataset | Terms | Citation |
|---|---|---|
| NSL-KDD | free for research with citation | Tavallaee et al., CISDA 2009 |
| UNSW-NB15 | "free use for academic research purposes in perpetuity"; commercial use requires the authors' agreement | Moustafa & Slay, MilCIS 2015 |
| CICIDS2017 (improved) | CIC research-use terms | Sharafaldin et al., ICISSP 2018; Engelen et al., WTMC 2021 / IEEE CNS 2022 |
| MITRE ATT&CK (STIX) | Apache 2.0 / MITRE terms | The MITRE Corporation |

This is a student hackathon project. Use is academic. Attribution is in `docs/DATASHEET.md`, the
README, and the model card.

### Traffic we generate ourselves

The live-capture demonstration runs on **virtual machines the team owns, on an isolated host-only
network with no route to the internet or to any campus network.**

Both endpoints are ours. The scanner and the scanned host are ours. Nothing leaves the lab. The scope
is written down before the lab is built, and it is a closed set of IP addresses.

### What the code can and cannot do

`penumbra.pcap` is a **read-only** capture-file parser. It reads `.pcap`/`.pcapng` from disk and
converts flows into feature vectors.

**It does not open a socket. It does not put an interface into promiscuous mode. It does not transmit a
packet.** There is no packet-crafting code, no scanner, and no exploit anywhere in this repository.

Likewise there is no blocking, firewall, or traffic-manipulation path — for the separate reasons in
ADR-0001.

---

## Privacy

**IP addresses are personal data.** GDPR Art. 4 and *Breyer v Bundesrepublik Deutschland* (CJEU
C-582/14) settle this for dynamic IPs.

Penumbra pseudonymises addresses with a keyed HMAC before storage or display. That is
**pseudonymisation, not anonymisation** — IPv4 is 2³² values, so anyone holding the key can enumerate
the whole mapping. Under GDPR Art. 4(5) and Recital 26, pseudonymised data remains personal data, and
we do not claim otherwise.

Consequences we accept:

- The key lives outside the data store and never in the repository.
- Re-identification is an `admin`-only operation and every use is written to the audit log.
- The motivated-intruder test is acknowledged, not argued away.
- **No packet payloads are inspected or stored** — the model uses flow metadata only. This is a
  privacy decision as much as an engineering one, and it means there is no payload corpus to leak.

---

## Responsible disclosure

If work on this project incidentally reveals a vulnerability in third-party software, we report it to
the maintainer privately and publish nothing until it is fixed or a reasonable disclosure window has
passed. We do not test the finding against anyone else's deployment.

---

## Dual-use acknowledgement

Feature attributions explain what makes traffic detectable — which is also a description of what an
evader would need to change. This is inherent to explainable detection and cannot be designed away.

Mitigation is access control rather than obscurity: attributions are available only through an
authenticated, role-gated, audited interface, and there is no public scoring endpoint.

---

## Scope statement for the demonstration

> Every result in this project comes from public research datasets or from traffic generated on virtual
> machines we own, on an isolated network. No system outside that lab was scanned, probed, or captured.
> The software reads capture files; it never transmits.
