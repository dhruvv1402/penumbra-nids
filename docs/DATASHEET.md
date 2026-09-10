# Datasheet for datasets

Following Gebru et al., *Datasheets for Datasets* (CACM 64(12), 2021). Covers the three datasets
Penumbra consumes. The **Uses** section is where the value is: it records each dataset's published
criticisms rather than only its advertised properties.

---

## Motivation

**Why were these datasets chosen?** They are the three the brief names, they are publicly available
without authentication, and together they cover complementary evaluation needs (ADR-0004): a modern
tabular benchmark, a natural zero-day holdout, and the only one carrying IPs and timestamps.

**Who created them?** UNSW-NB15: Moustafa & Slay, UNSW Canberra / ACCS. NSL-KDD: Tavallaee et al., UNB.
CICIDS2017: Sharafaldin, Lashkari & Ghorbani, Canadian Institute for Cybersecurity; the improved
re-release is by Engelen, Rimmer & Joosen, DistriNet / KU Leuven.

**Who funded them?** Academic and institutional research programmes at the respective universities.

---

## Composition

| | NSL-KDD | UNSW-NB15 (pre-split) | CICIDS2017 (improved) |
|---|---|---|---|
| Instances | 125,973 train / 22,544 test | 175,341 train / 82,332 test | ~2.7M |
| Unit | one network connection record | one network flow | one bidirectional flow |
| Features | 41 | 42 | ~78 |
| Labels | 39 attack names → 4 categories | 9 attack families + Normal | 14 attack labels + BENIGN |
| Carries IPs / timestamps | no | **no** | **yes** |
| Uncompressed | ~28 MB | ~48 MB | ~2–3 GB |

**Does it contain personal data?** No direct identifiers. CICIDS2017 contains IP addresses of a
simulated testbed, which are not real people's addresses — but Penumbra pseudonymises them regardless,
because the same code path handles real capture data where they would be.

**Is anything missing?** Materially, yes. The UNSW-NB15 pre-split CSVs omit `srcip`, `dstip`, `sport`,
`dsport`, `stime`, `ltime`, which exist only in the full four-part release. That absence determines the
architecture (ADR-0004).

**Class balance.** Severe and deliberate. UNSW-NB15 `Worms`: 130 train / 44 test. CICIDS2017
Heartbleed: 11 rows total; SQL Injection: 21; Infiltration: 36. These are not learnable as their own
classes at those counts, and we report them accordingly rather than quoting a per-class recall computed
on eleven rows.

**Errors and noise.** Extensively documented in [`DATA_TRAPS.md`](DATA_TRAPS.md) — 20% duplicate rows
and a duplicated column in CICIDS2017, a leaking `difficulty` column and an out-of-range `su_attempted`
in NSL-KDD, non-finite values, encoding defects, and label errors in the original CICIDS2017 release.

---

## Collection process

**All three are synthetic or simulated.** None is captured production traffic.

- **NSL-KDD** is a de-duplicated refinement of KDD Cup '99, itself derived from the **1998/99 DARPA
  intrusion detection evaluation** — a simulated military network.
- **UNSW-NB15** was generated in 2015 using the **IXIA PerfectStorm** traffic generator in a testbed at
  ACCS, with synthetic benign traffic and scripted attacks.
- **CICIDS2017** was generated over five days in 2017 in a testbed with profiled "benign" user
  behaviour and scheduled attacks.

**Labelling.** By construction — the generator knows which traffic it produced as an attack. This is
why CICIDS2017's original labels could be systematically wrong: the *labelling* was sound but the
*flow extraction* was buggy (CICFlowMeter terminated TCP flows on a single FIN rather than a mutual
exchange), so labels attached to incorrectly bounded flows. More than 20% changed in the correction.

---

## Preprocessing / cleaning / labelling

What Penumbra does, and why, per dataset — the full table is in [`DATA_TRAPS.md`](DATA_TRAPS.md).
Summary:

- Drop leaking columns (`difficulty`, `Destination Port`), zero-variance columns, and duplicated columns.
- Clamp `su_attempted` to {0,1}.
- De-duplicate **before** splitting.
- Strip whitespace-padded headers; handle the UTF-8 BOM and the CP1252 en-dash.
- Mask non-finite values arising from zero-duration flows.
- Frequency-encode high-cardinality categoricals rather than one-hot.
- **Quarantine suspected testbed artifacts** identified by single-feature AUC, and report every result
  with and without them.

Raw files are retained unmodified; all cleaning is applied at load time and is reproducible from the
manifest.

---

## Uses

### What these datasets have been used for

Benchmarking intrusion detection models, extensively, for two decades.

### What we use them for

Comparative evaluation of detection methods, and a methodology demonstration. **Not** as evidence about
absolute performance on a real network.

### Published criticisms — the part that matters

**NSL-KDD.** Descends from 1998/99 simulated traffic. The attack mix, protocol distribution and service
mix bear little resemblance to a modern network. Its feature set predates encrypted traffic being the
norm. It remains useful for one specific reason: `KDDTest+` contains 17 attack types absent from
`KDDTrain+`, which is a genuine unseen-attack holdout constructed by the authors rather than by us.

**UNSW-NB15.** Synthetic IXIA-generated traffic from 2015 with no host or browser telemetry. Its
attack-category boundaries overlap substantially in feature space — Exploits, DoS, Fuzzers, Backdoor
and Analysis are not cleanly separable, which is among the most replicated findings about the dataset,
and it is why our LOAFO experiment measures binary detection rather than family attribution as its
primary metric. `Generic` denotes cryptanalytic attacks against block ciphers and has **no defensible
MITRE ATT&CK mapping**; `Analysis` fuses three unrelated behaviours. We mark both unmappable rather
than force-fitting them. Attack and benign traffic were generated from different hosts, which is the
suspected source of the `sttl` artifact.

**CICIDS2017.** The original release has documented label and flow-extraction errors, which is why the
improved re-release exists and why we use it. `Destination Port` is a severe leak: attacks targeted
fixed victim ports, so a single decision stump on that column scores near-perfectly. Roughly 20% of
rows are exact duplicates.

### Uses we consider inappropriate

- Claiming production-representative detection rates.
- Averaging metrics across the three — they measure different traffic from different decades.
- Random splitting of CICIDS2017, which is temporally ordered.
- Re-shuffling NSL-KDD train and test together, which destroys the unseen-attack holdout.
- Reporting precision or PR-AUC without stating the evaluation prevalence.

---

## Distribution

Publicly downloadable without authentication from the sources pinned in
`src/penumbra/data/manifest.py`. Penumbra redistributes **no data** — only the manifest of URLs,
expected sizes and SHA-256 digests. Several widely-circulated mirrors are wrong (swapped train/test
files, redirect-to-HTML links); the known-bad ones are recorded in the manifest so nobody rediscovers
them.

---

## Maintenance

The datasets are static academic releases; we do not maintain them. Penumbra maintains the manifest,
and `penumbra data fetch` re-verifies digests on every run, so a silently changed upstream file is
detected rather than absorbed.

---

## Licences and citation

| Dataset | Terms |
|---|---|
| NSL-KDD | free for research with citation — Tavallaee, Bagheri, Lu & Ghorbani, CISDA 2009 |
| UNSW-NB15 | "free use for academic research purposes in perpetuity"; commercial use requires the authors' agreement — Moustafa & Slay, MilCIS 2015 |
| CICIDS2017 | CIC research-use terms — Sharafaldin, Lashkari & Ghorbani, ICISSP 2018; improved release: Engelen, Rimmer & Joosen, WTMC 2021 and IEEE CNS 2022 |
| MITRE ATT&CK | © The MITRE Corporation, used per MITRE's terms |

This project is academic and non-commercial.
