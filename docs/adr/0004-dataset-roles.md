# ADR-0004 — Three datasets, three distinct jobs

**Status:** accepted · **Date:** 2026-09-11

## Context

The brief offers NSL-KDD, CICIDS2017 and UNSW-NB15. The usual approach is to pick one, or to train on
all three and report an average.

While planning we checked the actual column layouts rather than the papers, and found a constraint that
changes the architecture: **the UNSW-NB15 pre-split CSVs contain no IP addresses and no timestamps.**

The 45 columns are `id` + 42 features + `attack_cat` + `label`. There is no `srcip`, `dstip`, `sport`,
`dsport`, `stime` or `ltime`. Those exist only in the full four-part release.

Four planned capabilities were therefore impossible on the dataset we had chosen as primary:

- entity-graph features (per-source fan-out, distinct destination ports, port entropy)
- per-host sequence windows
- alert-to-incident correlation grouped by source IP and time window
- temporal splits

The fourth item mattered most for honesty. The intended headline — *"the raw model emits thousands of
alerts; Penumbra presents tens of incidents"* — **cannot be measured without source IPs.** Synthesising
IPs in the replay engine and then quoting the compression ratio would have been a fabricated metric,
and the kind a security judge finds in thirty seconds of questioning.

## Decision

Each dataset does the job it is actually capable of.

| Dataset | Job | Why it and not the others |
|---|---|---|
| **UNSW-NB15** (175,341 / 82,332) | tabular benchmark · imbalance ablation · **LOAFO** | 10 labelled families, modern flow features, small enough to iterate on |
| **NSL-KDD** (125,973 / 22,544) | **the zero-day experiment** | 17 attack types appear in `KDDTest+` but not `KDDTrain+` — 3,750 rows, 16.6% of test |
| **CICIDS2017 improved** (~2.7M) | **correlation · graph features · sequence head · temporal validity** | the only one carrying `Source IP`, `Destination IP`, `Timestamp` |

CICIDS2017 is consequently **promoted out of "optional"**. It is no longer a nice-to-have second act;
it is load-bearing for four capabilities.

### Consequences per dataset

**UNSW-NB15.** Its `ct_*` columns (`ct_srv_src`, `ct_dst_ltm`, `ct_src_dport_ltm`, `ct_dst_src_ltm`, …)
are *already* entity-window aggregates, computed over a 100-connection window by the original
Argus/Bro pipeline. A graph head here would recompute what is already in the table. We say so on the
slide; knowing it reads as sophistication rather than as a missing feature.

Its test set is also **~55% attack**, which is why every precision-family number computed on it is
annotated with prevalence and re-weighted by `eval/prevalence.py` before anyone quotes an alert volume.

**NSL-KDD.** `KDDTrain+` and `KDDTest+` are **never concatenated and re-shuffled**. Doing so would
destroy the only naturally occurring unseen-attack holdout available to us — the dataset authors
constructed the zero-day experiment, and re-splitting throws it away. A test enforces this.

**CICIDS2017.** We use the Engelen/Rimmer/Joosen **improved re-release** (WTMC 2021, IEEE CNS 2022)
rather than the original, because the original's labels are known-wrong: a CICFlowMeter bug terminated
TCP flows on a single FIN rather than a mutual exchange, and more than 20% of flows changed label or
boundary in the correction. Using a dataset whose errata are published, and citing the errata, is
strictly better than using the version everyone else uses.

Its natural day-based split — Monday benign-only, attacks Tuesday through Friday — gives a genuine
temporal split (train Mon–Wed, test Thu–Fri) rather than a random one. Monday being benign-only also
makes it the correct training source for the novelty head.

## Alternatives considered

**Download the full four-part UNSW-NB15 release (~600 MB, has `srcip`/`stime`) and derive our own
splits.** Rejected: more work, and it forfeits comparability with the standard 175k/82k split that the
literature reports against.

**Fabricate plausible IPs for the UNSW replay.** Rejected outright. Any metric derived from invented
entities is invented. The replay engine may *display* synthetic identifiers for visual realism, but no
number computed from them appears in any report, and the display is labelled.

**Use CICIDS2017 alone.** Rejected: 2.8M rows and six separate cleaning traps make it a poor iteration
loop, and it has no naturally unseen attack families, so the central experiment would have to be
entirely synthetic.

## Consequences

- Three loaders, three schemas, three sets of traps to encode. Cost accepted; `data/schema.py` holds
  all of it in one auditable place.
- Results are reported per dataset. No averaging across datasets — they measure different things on
  different traffic from different decades, and a mean over them would be meaningless.
- Cross-dataset transfer (train on one, test on another) stays a stretch goal, and if it fails it is
  published as a negative result.

## Related

`src/penumbra/data/schema.py` · `docs/DATA_TRAPS.md` · `docs/DATASHEET.md` · ADR-0002
