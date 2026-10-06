# Deployment

Penumbra runs publicly on **Azure Container Apps** (Central India), in the same subscription and
resource group as the live Sentinel workspace (sentinel/README.md "Live").

```
browser ──HTTPS──> penumbra-console (Next.js, public)
                        │  /api/* proxied
                        └──HTTPS──> penumbra-api (FastAPI, 1 replica, SQLite on the container)
```

Recreate it with one command (needs `az login` and Docker):

```bash
uv run python scripts/deploy_azure.py --base-image penumbra-api:local   # reuse a local API image's dependencies
uv run python scripts/deploy_azure.py                                    # or build everything
```

The script creates the registry, builds and pushes both images, creates the environment and both
apps, and prints the URL. Re-running updates in place and keeps existing secrets.

## What a public instance changes

| | local `penumbra demo` | public deployment |
|---|---|---|
| logins | analyst/analyst, senior/senior, admin/admin | **no known passwords**: generated 24-character passwords held as Container Apps secrets, written only to `artifacts/deploy/credentials.env` (gitignored) |
| read-only access | - | **"View as guest"**: role `guest` holds alerts:read and incidents:read and nothing else; every write returns 403 (tested in `tests/integration/test_deploy.py`) |
| JWT secret, PII key | per-process / development key | generated, stored as secrets; the API refuses to start on the public development PII key |
| data | fixtures pushed by `penumbra demo` | the same fixtures, seeded on start **only into an empty store** (`PENUMBRA_SEED_FIXTURES`) - a restart comes back with the demo data, and a populated store is never touched |
| SIEM | local mock | local mock (switch to Sentinel below) |
| live scoring (ADR-0006) | `/score/*` against `artifacts/models/` and `artifacts/samples/` | the **registry champions**, copied out of the registry only after their manifests verify, plus the held-out sample pools (`penumbra samples`). Guests: 30 calls/hour, 500 flows a call. API container 2 CPU / 4 GiB (UNSW alone is ~750 MB loaded). Nothing scored is stored |

## Things measured, not assumed

- **The API is reachable from the internet.** It is created with *internal* ingress, which a
  standard Container Apps environment honours. The environment an Azure for Students subscription
  gets is an "express" environment, and it does not: the API answers at its own FQDN. App-name
  addressing (`http://penumbra-api`) does not resolve there either, so the console proxies to the
  API's HTTPS FQDN. This is acceptable because every route except `/health`, `/metrics`,
  `/auth/login` and `/auth/guest` requires a token, and there are no known passwords. It is stated
  here rather than implied by the word "internal".
- **Images are built locally.** ACR Tasks (`az acr build`) is refused on this subscription type
  (`TasksOperationsNotAllowed`), and so is pulling with a managed identity
  (`ExpressEnvironmentFeatureNotSupported`); the script uses the registry's admin credential, which
  it passes straight to Azure without printing or storing it.
- **State is per-container.** One API replica, SQLite on the container's disk. A verdict a judge
  records survives until the container restarts; the demo data always comes back.

Checked after deploying: console 200; guest sees 4,165 alerts and 203 incidents; a guest ingest or
audit read returns 403; `analyst/analyst` returns 401; the evaluation reports load; the live alert
stream connects; a senior login works; no page errors.

## Forwarding to the live Sentinel workspace

The deployed API uses the SIEM mock. To forward to Sentinel, set the Sentinel variables on the API
app, with the client secret created straight into a secret (never printed):

```powershell
$az = "C:\Program Files\Microsoft SDKs\Azure\CLI2\wbin\az.cmd"
$s = & $az ad app credential reset --id <app-id> --display-name penumbra-aca --append --years 1 --query password -o tsv
& $az containerapp secret set -g penumbra-rg -n penumbra-api --secrets "sentinel-secret=$s"
& $az containerapp update -g penumbra-rg -n penumbra-api --set-env-vars PENUMBRA_SIEM=sentinel `
    PENUMBRA_SENTINEL_ENDPOINT=<dcr endpoint> PENUMBRA_SENTINEL_DCR_ID=<dcr immutable id> `
    PENUMBRA_AZURE_TENANT_ID=<tenant> PENUMBRA_AZURE_CLIENT_ID=<app-id> `
    PENUMBRA_AZURE_CLIENT_SECRET=secretref:sentinel-secret
```

(`scripts/deploy_sentinel.py` prints the endpoint, DCR id, tenant and app id.) Seeded data is not
forwarded; alerts ingested after the switch are.

## Cost and teardown

Two always-on replicas (1 vCPU / 2 GiB and 0.5 vCPU / 1 GiB) and a Basic registry: roughly a few
dollars a day against the $100 student credit. Scale to zero after judging with
`az containerapp update -n <app> -g penumbra-rg --min-replicas 0`, or remove everything with
`az group delete -n penumbra-rg`.
