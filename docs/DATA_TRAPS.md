# Data traps

Every item here was verified against the actual bytes, not copied from a paper. Each one is encoded in
`src/penumbra/data/schema.py` and covered by a regression test.

This document exists because naming the traps you avoided is itself an evaluation artifact. Most of
these fail *silently* — they do not raise, they just quietly inflate a number.

---

## Sources that are wrong rather than broken

A 404 is harmless; you notice it. These do not raise.

| Source | What actually happens |
|---|---|
| `205.174.165.80/.../MachineLearningCSV.zip` — CICIDS2017's official link | Answers **HTTP 200**, then 301 → 302 → the UNB datasets index page. You save **108 KB of HTML named `.zip`**. |
| HF `Mireu-Lab/UNSW-NB15` | `train.csv` and `test.csv` are **swapped** — their "train" is the 82k *testing* split. You train on the small half and everything looks plausible. |
| HF `bastyje/UNSW-NB15` | Not the pre-split edition — it is the full 2.54M-record raw set with a different schema. Not comparable to any published 175k/82k result. |
| `cloudstor.aarnet.edu.au/...` | AARNet retired CloudStor. Connection failure, not a 404. |

**Defence:** `data/manifest.py` pins URL, byte size and SHA256; `data/download.py` deletes any download
whose size disagrees with the manifest and says so; loaders assert row counts after parsing. Three
independent chances to notice we have the wrong bytes.

---

## NSL-KDD

| Trap | Effect if missed | Handling |
|---|---|---|
| **`difficulty` (column 43) is a leak** | It encodes how many of 21 classic learners classified that row correctly — a function of the answer. Published papers have shipped results with it left in. | dropped in `NSLKDD_DROP` |
| `num_outbound_cmds` is identically zero in both splits | zero-variance column, no information, breaks some scalers | dropped |
| `su_attempted` contains a **third value, `2`** | documented as binary; left alone it becomes a spurious ordinal level | clamped to {0,1} via `NSLKDD_CLAMP` |
| Files are **headerless** | pandas silently promotes the first data row to a header — you lose a row and gain nonsense column names | explicit `NSLKDD_COLUMNS` |
| `Attack Types.csv` uses **bare-CR (`\r`) line endings** and omits every test-only attack name | `read_csv` returns one mangled row | the 39-name → 4-category map is hardcoded in `schema.py` |
| `+` in `KDDTrain+.txt` | some clients decode a literal `+` in a URL path as a space; the request 404s | percent-encoded as `%2B` in the manifest |
| **Concatenating and re-shuffling train + test** | destroys the 17 naturally unseen attack types — the only real zero-day holdout we have | forbidden; enforced by test |

The unseen-17 (`mscan`, `apache2`, `processtable`, `snmpguess`, `saint`, `mailbomb`, `snmpgetattack`,
`httptunnel`, `named`, `ps`, `sendmail`, `xterm`, `xlock`, `xsnoop`, `worm`, `udpstorm`, `sqlattack`)
are 3,750 rows, 16.6% of `KDDTest+`. This is why the famous KDDTest+ accuracy ceiling sits near 0.77
while per-class U2R/R2L recall sits at 5–15%. It is a feature, not a defect, and it is the point.

---

## UNSW-NB15

| Trap | Effect if missed | Handling |
|---|---|---|
| **UTF-8 BOM on the `id` column** | first column name becomes `﻿id`; every lookup by name fails | `encoding="utf-8-sig"` |
| Training file is **larger** than the testing file (175k vs 82k) | looks like a swap; people "fix" it and break the comparison | asserted row counts, documented |
| **No IPs, no timestamps** | graph features, sequence windows, IP correlation and temporal splits are all impossible — and an incident-compression figure computed from synthesised IPs would be fabricated | those capabilities moved to CICIDS2017 (ADR-0004) |
| Test set is **~55% attack** | every precision, PPV and PR-AUC number is inflated by 2–4 orders of magnitude relative to deployment | `eval/prevalence.py` re-weights; PR-AUC always printed with its prevalence and no-skill baseline |
| `proto` has ~130 distinct values | one-hot explodes dimensionality and lets categorical noise dominate autoencoder reconstruction error | frequency encoding, or top-15 + `OTHER` |
| Plain SMOTE on categorical columns | interpolates between one-hot columns and generates rows that cannot exist | SMOTE-NC where categoricals are present |
| **`sttl` is a suspected testbed artifact** | attack and benign traffic were generated from different hosts; TTL records the generator, not the behaviour. Expected solo AUC ≈ 0.90+ | `data/audit.py` quarantines any feature with solo AUC > 0.90; every headline number reported with and without |
| `id` kept as a feature | invites memorisation of row order | dropped |
| `Worms` has **44 test rows** | a recall point estimate carries roughly ±15 points of interval | intervals only, never a point estimate |

---

## CICIDS2017

| Trap | Effect if missed | Handling |
|---|---|---|
| **`Destination Port` is a label leak** | attacks were generated against fixed victim ports; a single decision stump scores near-perfectly | dropped; with/without delta reported to quantify the inflation |
| **~20% exact duplicate rows** | duplicates straddle the split, test leaks into train, accuracy inflates toward 99.9% | de-duplicate **before** splitting |
| **`Fwd Header Length` appears twice** (columns 35 and 56) | pandas silently renames the second to `.1`; it carries no information | dropped |
| Headers carry **leading spaces** (`" Destination Port"`) | every lookup by name fails, mysteriously | `df.columns.str.strip()` before anything else |
| 8 identically-zero columns | zero variance, breaks scalers | dropped |
| `Flow Bytes/s` and `Flow Packets/s` carry **NaN and ±Inf** | zero-duration flows divide by zero; sklearn raises far downstream | non-finite masking at load |
| Web-attack labels contain a raw **`0x96` CP1252 en-dash** | `UnicodeDecodeError` on the Thursday morning file with `encoding="utf-8"` | `encoding="latin1"` + regex-normalise the dash. Some mirrors have already transcoded it to `U+FFFD`, so match on a pattern, not a literal |
| Filename casing is load-bearing | `Wednesday-workingHours` has a lowercase `w`; `Infilteration` is misspelled that way in the original | exact names in the manifest |
| **Random rather than day-based splitting** | leaks future into past across an inherently temporal dataset | train Mon–Wed, test Thu–Fri |
| Heartbleed (11 rows), SQL Injection (21), Infiltration (36) | unlearnable as their own classes; per-class recall on 11 rows is noise | folded into a coarser label, reported as such |
| Original labels are **known-wrong** | a CICFlowMeter bug terminated TCP flows on a single FIN rather than a mutual exchange; >20% of flows changed in the correction | we use the Engelen et al. **improved** re-release and cite the errata |

---

## Traps in our own pipeline

Not dataset defects — mistakes we could make, each with a test that fails if we do.

| Trap | Test |
|---|---|
| Resampler fitted outside a CV fold (the classic SMOTE leak) | pipeline inspection test fails the build |
| Novelty-head scaler fitted on data containing attack rows | property test asserts the benign-only pipeline never sees an attack row during `fit` |
| Non-causal sequence windows — features for flow *t* using flows after *t* | window construction test |
| PSI bins recomputed per window instead of fixed to the reference | drift unit test; recomputed bins make PSI read ~0 regardless of actual shift |
| 42 KS tests per window with no multiplicity correction | ~2 false flags every window forever at α=0.05; Benjamini-Hochberg applied |
| Unstratified bootstrap | rare classes vanish from resamples and their CIs become nonsense |
| A blocking code path appearing anywhere in the tree | source-tree scan fails the build (ADR-0001) |

---

## Sources

Engelen, Rimmer & Joosen, *Troubleshooting an Intrusion Detection Dataset* (WTMC 2021) and the IEEE
CNS 2022 follow-up — the CICIDS2017 corrections. Tavallaee et al. (CISDA 2009) — NSL-KDD.
Moustafa & Slay (MilCIS 2015) — UNSW-NB15. Sharafaldin, Lashkari & Ghorbani (ICISSP 2018) — CICIDS2017.
