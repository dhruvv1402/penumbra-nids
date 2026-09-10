# ADR-0003 — One canonical alert, three serialisers

**Status:** accepted · **Date:** 2026-09-11

## Context

A detector that a SOC cannot ingest is a research artifact. The output has to land in whatever the team
already runs — Microsoft Sentinel, an OCSF-consuming data lake, or an Elastic stack — without a
rewrite per destination.

This is a Microsoft-sponsored event, so Sentinel is the priority. But building only for Sentinel would
be both strategically narrow and a worse engineering answer.

## Decision

One internal `Alert` dataclass is the single source of truth. Three thin serialisers project it onto:

| Target | Schema | Notes |
|---|---|---|
| Microsoft Sentinel | **ASIM `NetworkSession`**, `EventType = "IDS"`, schema version 0.2.7 | primary |
| Data lakes / AWS Security Lake | **OCSF Detection Finding**, `class_uid 2004` | vendor-neutral |
| Elastic | **ECS**, `event.kind: "alert"` | vendor-neutral |

*One detection, three schemas, zero vendor lock-in.* Roughly 150 lines of serialiser.

### Sentinel specifics

ASIM's documentation defines `EventType: IDS` as "a network session reported as suspicious" — literally
our event. The model verdict maps onto the inspection fields:

- `ThreatConfidence` — integer 0–100, from the **calibrated** `p_attack` × 100. Not `priority`;
  `priority` is a sort key, not a probability, and putting it here would misrepresent it.
- `ThreatName`, `ThreatCategory`, `ThreatId`, `ThreatRiskLevel`
- `ThreatField` — **conditional**: required whenever `ThreatIpAddr` is set
- `NetworkRuleName` — `Penumbra-XGB-v{version}/{family}`

**The alert-not-block decision, expressed in Microsoft's own vocabulary:**

```
DvcAction     = "Allow"      ← we did not block
ThreatConfidence = 93
EventSeverity = "High"
```

ADR-0001 stated in the SIEM's normalized schema rather than in prose.

### Parsers, not just payloads

We ship two KQL functions following Microsoft's documented naming convention:

- `ASimNetworkSessionPenumbra` — parameter-less normalising parser
- `vimNetworkSessionPenumbra` — parametrized filtering parser

The `_Im_NetworkSession` unifying parser then picks our table up automatically, which means
**Microsoft-authored ASIM analytics rules run against Penumbra output without us writing them**. That
is the difference between integrating with Sentinel and joining its detection ecosystem.

The repository carries a real **Sentinel Solution layout** — `Analytic Rules/`, `Parsers/`,
`Data Connectors/`, `Workbooks/` — so the artifact is shaped like an actual ISV contribution.

### Ingestion path

**Logs Ingestion API via a `"kind": "Direct"` DCR**, OAuth2 client credentials with audience
`https://monitor.azure.com`, the app granted **Monitoring Metrics Publisher** on the DCR, custom table
`PenumbraAlerts_CL`.

Two deliberate choices worth stating: a **DCE is no longer required** (since March 2024 the DCR exposes
its own `logsIngestion` endpoint), and the legacy **HTTP Data Collector API retires 14 September 2026**
— we built on the DCR path rather than the dying shared-key one.

### Mock by default

`integrations.siem.base.SiemConnector` is a Protocol. `LocalMockSiem` writes JSONL to disk and is the
default binding. `AzureSentinelSiem` raises `NotConfigured` until credentials exist.

**The demo must never depend on a live network call.** Schema conformance is verified by a test suite —
all mandatory ASIM fields present, every enum value legal, `ThreatConfidence ∈ [0,100]`, `ThreatField`
set whenever `ThreatIpAddr` is, OCSF `type_uid == class_uid × 100 + activity_id`, ECS never emitting
`event.kind: "signal"`.

A green suite proving "our output is Sentinel-ingestible" is more persuasive, and more verifiable, than
a screenshot of a portal.

## Known constraints we are not pretending away

- `EventVendor` / `EventProduct` come from a **closed allow-list** that Penumbra is not on. Production
  use would require Microsoft to allocate a designator. We use our own and say so.
- Analytics rule `description` fields must be **ASCII only** — em dashes and smart quotes fail
  validation.
- `EventSeverity` is a four-value **string** enum (`Informational|Low|Medium|High`), not a number.
- Sentinel validates `relevantTechniques` against **ATT&CK v16**.
- OCSF Detection Finding has **no top-level `src_endpoint`/`dst_endpoint`** — the network 5-tuple goes
  inside `evidences[]`.
- ECS reserves `event.kind: "signal"` for Kibana's own alerting engine; ingestion pipelines must not
  emit it.

## Consequences

- Adding a fourth destination is one serialiser plus one conformance test.
- The three schemas disagree on how to express confidence (`ThreatConfidence` 0–100, OCSF
  `confidence_score` plus a `confidence_id` band, ECS `event.risk_score`). All three are projections of
  the same calibrated probability, and the mapping is centralised so they cannot drift apart.
- Serialisers must be updated when a schema version moves. Pinned versions are asserted in tests.

## Related

ADR-0001 · ADR-0002 · `sentinel/` · `src/penumbra/alerts/schemas/`
