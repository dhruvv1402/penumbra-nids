# Sentinel solution

Laid out the way a real Sentinel Solution is, so the artifacts can be read against
[Azure/Azure-Sentinel](https://github.com/Azure/Azure-Sentinel) rather than taken on trust.

```
Analytic Rules/   scheduled KQL detection rules, plus the mined rule packs
Parsers/          ASIM normalizing + filtering parsers
Data Connectors/  the DCR (PenumbraDCR.json); no portal connector definition yet
Sigma/            mined rules that Sigma can express honestly
Workbooks/        PenumbraOverview.json: lanes, verdict mix, suppressions, abstention, ADR-0001 check
```

## Live, not simulated (2026-09-26)

Deployed into a real Sentinel workspace (Azure for Students, Central India) with
`scripts/deploy_sentinel.py`, fed by `penumbra demo` with `PENUMBRA_SIEM=sentinel`, and queried back:

| | measured in the workspace |
|---|---|
| records ingested through the DCR | 8,332 (4,160 distinct alerts, each sent twice) |
| `DvcAction = "Allow"` | **8,332 of 8,332** |
| records with a source IP that carry a pseudonym (`pseudo:...`), not an address | **2,022 of 2,022** |
| normalised by `ASimNetworkSessionPenumbra` | 8,332, `EventSchema = NetworkSession`, `EventType = IDS` |
| `vimNetworkSessionPenumbra(dvcaction=dynamic(['Deny']))` | 0, as designed: Penumbra never denies |
| incidents raised by `PenumbraNovelTraffic.yaml` | **13**, "Unrecognised network behaviour from pseudo:...", one per source |

Unique alerts by severity: High 1,769, Informational 1,654, Medium 643, Low 100.

**What the live run found, that the test suite had not:**

1. **The analytics rule was invalid.** Sentinel refuses an `alertDescriptionFormat` with more than
   three `{{column}}` placeholders; ours had five. Fixed, and
   `tests/unit/test_sentinel_rules.py` now enforces the limit, and checks that every column an
   override or custom detail names is produced by the query.
2. **The "joins the ASIM ecosystem automatically" claim was false.** This README used to say that
   shipping `ASimNetworkSessionPenumbra` under Microsoft's naming convention makes the built-in
   `_Im_NetworkSession` unifying parser pick Penumbra up. Measured: `_Im_NetworkSession` returns
   **0** Penumbra rows, and still 0 with `ASimNetworkSessionCustom` / `vimNetworkSessionCustom`
   hooks deployed (which do work when called directly: 8,332 rows). What holds is narrower:
   Penumbra's parsers normalise to ASIM NetworkSession 0.2.7, so an ASIM rule works against them when
   it queries `ASimNetworkSessionPenumbra` (or the Custom hooks) instead of the built-in parser.
3. **A new app's permission takes minutes to apply.** Until it does, Sentinel answers 403 and the
   first batches are refused. `/ingest` now returns the reason with the count, `siem.forward_rejected`
   is audit-logged, and a 403 names the likely cause (role missing or not yet applied).

Reproduce: `az login`, `uv run python scripts/deploy_sentinel.py`, then the environment block it
prints (the client secret is created straight into your shell and never printed or stored).

## Alert-not-block, in Microsoft's own vocabulary

```
DvcAction        = "Allow"     <- we observed and did not act
ThreatConfidence = 93          <- and we were confident
EventSeverity    = "High"
```

ADR-0001 expressed as three normalized fields rather than as a sentence in a README. A reviewer can
read it straight off the payload.

## Ingestion

`src/penumbra/integrations/siem/`: `/ingest` forwards every alert, suppressed ones included, as an
ASIM record through a `SiemConnector`. `PENUMBRA_SIEM=mock` (the default) validates each record and
appends it to `<state dir>/siem/PenumbraAlerts_CL.jsonl` (`PENUMBRA_STATE_ROOT`, default `artifacts/`), which is exactly what Sentinel would
receive. `PENUMBRA_SIEM=sentinel` posts to the Logs Ingestion API and refuses to start without
`PENUMBRA_SENTINEL_ENDPOINT`, `PENUMBRA_SENTINEL_DCR_ID` and the three `PENUMBRA_AZURE_*` values.
It never falls back to the mock silently. The Sentinel client is tested against a fake transport,
and has been run against a live workspace (above).

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

`tests/unit/test_alert_schemas.py` asserts the payload rules against generated payloads, and
`tests/unit/test_sentinel_rules.py` asserts the analytics-rule rules (ASCII-only description,
technique IDs) and that both parsers exist. A green suite
proving "this is Sentinel-ingestible" is more verifiable than a screenshot of a portal.

## Constraint we are not pretending away

`EventVendor` / `EventProduct` come from a closed Microsoft allow-list that Penumbra is not on.
Production use would require Microsoft to allocate a designator. We use our own and say so.

## Running this for real

`integrations/siem/` binds `LocalMockSiem` by default and `AzureSentinelSiem` raises
`NotConfigured` until credentials exist, so the demo never *depends* on a live Azure call; with
`PENUMBRA_SIEM=sentinel` it makes them (see "Live" above). Azure for
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
