# Sentinel solution

Laid out the way a real Sentinel Solution is, so the artifacts can be read against
[Azure/Azure-Sentinel](https://github.com/Azure/Azure-Sentinel) rather than taken on trust.

```
Analytic Rules/   scheduled KQL detection rules, plus the mined rule packs
Parsers/          ASIM normalizing + filtering parsers
Data Connectors/  the DCR and connector definition
Sigma/            mined rules that Sigma can express honestly
Workbooks/        (not built)
```

## The claim worth making

Shipping `ASimNetworkSessionPenumbra` under Microsoft's documented parser naming convention means
the `_Im_NetworkSession` unifying parser picks Penumbra's table up automatically — and **every
Microsoft-authored ASIM analytics rule then runs against Penumbra output without us writing any of
them**. That is the difference between integrating with Sentinel and joining its detection
ecosystem.

## Alert-not-block, in Microsoft's own vocabulary

```
DvcAction        = "Allow"     <- we observed and did not act
ThreatConfidence = 93          <- and we were confident
EventSeverity    = "High"
```

ADR-0001 expressed as three normalized fields rather than as a sentence in a README. A reviewer can
read it straight off the payload.

## Ingestion

Logs Ingestion API through a `"kind": "Direct"` data collection rule:

```
POST {dcr-endpoint}/dataCollectionRules/{immutableId}/streams/Custom-PenumbraAlerts?api-version=2023-01-01
Authorization: Bearer <token audience https://monitor.azure.com>
```

The app registration needs **Monitoring Metrics Publisher** on the DCR. Two deliberate choices:
a DCE is **no longer required** (since March 2024 the DCR exposes its own `logsIngestion` endpoint),
and the legacy HTTP Data Collector API **retires 14 September 2026** — this is built on the DCR
path, not the dying shared-key one.

## Validation traps these files respect

- `EventSeverity` is a four-value **string** enum (`Informational|Low|Medium|High`), not a number.
- `ThreatField` is **conditional** — required whenever `ThreatIpAddr` is set. Omitting it is the
  easiest way to fail schema validation.
- `ThreatConfidence` and `ThreatRiskLevel` are integers **0–100**, not floats 0–1.
- Analytics rule `description` must be **ASCII only**; em dashes and smart quotes fail validation.
- `relevantTechniques` is validated against **ATT&CK v16**.
- Log Analytics column names must start with a letter and be ≤45 characters.

`tests/unit/test_alert_schemas.py` asserts all of these against generated payloads. A green suite
proving "this is Sentinel-ingestible" is more verifiable than a screenshot of a portal.

## Constraint we are not pretending away

`EventVendor` / `EventProduct` come from a closed Microsoft allow-list that Penumbra is not on.
Production use would require Microsoft to allocate a designator. We use our own and say so.

## Running this for real

`integrations/siem/` binds `LocalMockSiem` by default and `AzureSentinelSiem` raises
`NotConfigured` until credentials exist — the demo never depends on a live Azure call. Azure for
Students provides $100 with no credit card, and enabling Sentinel grants 10 GB/day free for 31 days,
which is orders of magnitude more than this alert volume needs.


## Mined rules

`PenumbraMinedRules_{unsw,nslkdd}.kql` are not hand-written. They are decision paths lifted out of
a trained random forest, validated on held-out data the trees never saw, and emitted with their own
precision and false-positive count inline. They run **without the model** — that is the point. The
ML mines the detection; Sentinel enforces it; a detection engineer can read, argue with, and own
the result.

Enable them individually on the basis of the cost stated in each rule's comment, not as a block.

Two things are deliberately absent:

**Rules resting on testbed artifacts.** 106 of 207 mined UNSW paths depended on a feature our own
audit quarantined (`sttl`, `dttl`, `ct_state_ttl`, `is_sm_ips_ports`, `proto='unas'`). Every one of
them validated at high precision on held-out data from the same testbed and would fire on an
operating system rather than on an attack. They are counted in `docs/EVALUATION.md` §10.7b and
shipped nowhere.

**Sigma translations that Sigma cannot express.** 1 of 95 UNSW rules and 7 of 317 NSL-KDD rules are
in `Sigma/`. The rest reference flow features Sigma's log-based taxonomy has no field for — `sload`,
`ct_srv_src`, `sinpkt` — and a partial translation is a different rule, not a shorter one.

Regenerate with `uv run penumbra rules --dataset unsw`.
