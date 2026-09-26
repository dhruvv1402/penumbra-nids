"""Deploy the Sentinel side of Penumbra into an Azure subscription, with the Azure CLI.

    uv run python scripts/deploy_sentinel.py [--resource-group penumbra-rg] [--location centralindia]

Idempotent: every step is a PUT, so re-running updates rather than duplicates. What it creates:

  resource group, Log Analytics workspace, Microsoft Sentinel on it
  PenumbraAlerts_CL, from the columns in `sentinel/Data Connectors/PenumbraDCR.json`
  the data collection rule (kind Direct: its own ingestion endpoint, no DCE)
  app registration `penumbra-ingest` with Monitoring Metrics Publisher on the DCR - and NO secret
  the ASIM parser pair, plus the ASimNetworkSessionCustom / vimNetworkSessionCustom hooks
  the Penumbra overview workbook and every scheduled rule in `sentinel/Analytic Rules/*.yaml`

It never creates or prints a client secret. The last thing it prints is the command that creates
one straight into the calling shell's environment, and the other variables `PENUMBRA_SIEM=sentinel`
needs. Run against a live workspace on 2026-09-26; see sentinel/README.md "Live".

Requires `az login` (device code works: `az login --use-device-code`).
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1] / "sentinel"
TYPES = {"datetime": "dateTime", "int": "int", "long": "long", "string": "string", "boolean": "boolean",
         "dynamic": "dynamic", "real": "real"}  # fmt: skip
VIM_PARAMS = (
    "starttime:datetime=datetime(null), endtime:datetime=datetime(null), "
    "srcipaddr_has_any_prefix:dynamic=dynamic([]), dstipaddr_has_any_prefix:dynamic=dynamic([]), "
    "ipaddr_has_any_prefix:dynamic=dynamic([]), dstportnumber:int=int(null), "
    "hostname_has_any:dynamic=dynamic([]), dvcaction:dynamic=dynamic([]), eventresult:string='*', disabled:bool=false"
)
VIM_ARGS = ", ".join(f"{p.split(':')[0]}={p.split(':')[0]}" for p in VIM_PARAMS.split(", "))


def az_binary() -> str:
    found = shutil.which("az") or shutil.which("az.cmd")
    fallback = Path(r"C:\Program Files\Microsoft SDKs\Azure\CLI2\wbin\az.cmd")
    if found:
        return found
    if fallback.exists():
        return str(fallback)
    sys.exit("Azure CLI not found. Install it and run `az login` first.")


AZ = az_binary()


def az(*args: str) -> str:
    done = subprocess.run([AZ, *args], capture_output=True, text=True)  # noqa: S603 - fixed binary
    if done.returncode != 0:
        sys.exit(f"az {' '.join(args[:3])} failed:\n{done.stderr.strip()[:1500]}")
    return done.stdout.strip()


def put(url: str, body: dict[str, Any]) -> dict[str, Any]:
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as f:
        json.dump(body, f)
        path = f.name
    try:
        out = az(
            "rest", "--method", "put", "--url", f"https://management.azure.com{url}", "--body", f"@{path}"
        )
    finally:
        Path(path).unlink(missing_ok=True)
    return json.loads(out) if out else {}


def iso(duration: str) -> str:
    return "PT" + duration.upper() if duration[-1].lower() in "mh" else duration


def main(rg: str, location: str, workspace: str, app_name: str) -> None:
    account = json.loads(az("account", "show"))
    sub, tenant = account["id"], account["tenantId"]
    print(f"subscription {account['name']} ({sub})")
    for provider in ("Microsoft.OperationalInsights", "Microsoft.Insights", "Microsoft.SecurityInsights",
                     "Microsoft.OperationsManagement"):  # fmt: skip
        az("provider", "register", "--wait", "-n", provider)

    az("group", "create", "-n", rg, "-l", location, "-o", "none")
    ws = json.loads(az("monitor", "log-analytics", "workspace", "create", "-g", rg, "-n", workspace,
                       "-l", location, "--retention-time", "30"))  # fmt: skip
    ws_id = ws["id"]
    put(
        f"{ws_id}/providers/Microsoft.SecurityInsights/onboardingStates/default?api-version=2024-03-01",
        {"properties": {}},
    )
    print(f"workspace {workspace}, Sentinel on")

    dcr = json.loads((ROOT / "Data Connectors" / "PenumbraDCR.json").read_text(encoding="utf-8"))
    columns = dcr["properties"]["streamDeclarations"]["Custom-PenumbraAlerts"]["columns"]
    put(f"{ws_id}/tables/PenumbraAlerts_CL?api-version=2022-10-01", {"properties": {
        "schema": {"name": "PenumbraAlerts_CL", "columns": [{"name": c["name"], "type": TYPES[c["type"]]} for c in columns]},
        "retentionInDays": 30}})  # fmt: skip
    dcr.pop("_comment", None)
    dcr["location"] = location
    dcr["properties"]["destinations"]["logAnalytics"][0]["workspaceResourceId"] = ws_id
    dcr_path = f"/subscriptions/{sub}/resourceGroups/{rg}/providers/Microsoft.Insights/dataCollectionRules/penumbra-dcr"
    rule = put(f"{dcr_path}?api-version=2023-03-11", dcr)
    endpoint, immutable = rule["properties"]["endpoints"]["logsIngestion"], rule["properties"]["immutableId"]
    print(f"table PenumbraAlerts_CL, DCR {immutable}")

    app_id = az("ad", "app", "list", "--display-name", app_name, "--query", "[0].appId", "-o", "tsv")
    if not app_id:
        app_id = az("ad", "app", "create", "--display-name", app_name, "--sign-in-audience", "AzureADMyOrg",
                    "--query", "appId", "-o", "tsv")  # fmt: skip
    if not az("ad", "sp", "list", "--filter", f"appId eq '{app_id}'", "--query", "[0].id", "-o", "tsv"):
        az("ad", "sp", "create", "--id", app_id, "-o", "none")
    # JSON parsed here rather than with --query: on Windows az is a .cmd wrapper, and cmd.exe mangles
    # JMESPath containing parentheses.
    existing = json.loads(az("role", "assignment", "list", "--assignee", app_id, "--scope", dcr_path) or "[]")
    if not existing:
        az("role", "assignment", "create", "--assignee", app_id, "--role", "Monitoring Metrics Publisher",
           "--scope", dcr_path, "-o", "none")  # fmt: skip
    print(f"app {app_name} ({app_id}), Monitoring Metrics Publisher on the DCR")

    functions = {
        "ASimNetworkSessionPenumbra": ((ROOT / "Parsers" / "ASimNetworkSessionPenumbra.kql").read_text(encoding="utf-8"), None),
        "vimNetworkSessionPenumbra": ((ROOT / "Parsers" / "vimNetworkSessionPenumbra.kql").read_text(encoding="utf-8"), VIM_PARAMS),
        "ASimNetworkSessionCustom": ("ASimNetworkSessionPenumbra | where not(disabled)", "disabled:bool=false"),
        "vimNetworkSessionCustom": (f"vimNetworkSessionPenumbra({VIM_ARGS})", VIM_PARAMS),
    }  # fmt: skip
    for alias, (query, params) in functions.items():
        props = {
            "category": "ASIM",
            "displayName": alias,
            "functionAlias": alias,
            "query": query,
            "version": 2,
        }
        if params:
            props["functionParameters"] = params
        put(f"{ws_id}/savedSearches/{alias.lower()}?api-version=2020-08-01", {"properties": props})
    print("parsers: " + ", ".join(functions))

    workbook_id = str(uuid.uuid5(uuid.NAMESPACE_URL, ws_id + "/penumbra-overview"))
    put(f"/subscriptions/{sub}/resourceGroups/{rg}/providers/Microsoft.Insights/workbooks/{workbook_id}?api-version=2022-04-01", {
        "location": location, "kind": "shared", "properties": {
            "displayName": "Penumbra overview", "category": "sentinel", "sourceId": ws_id, "version": "1.0",
            "serializedData": (ROOT / "Workbooks" / "PenumbraOverview.json").read_text(encoding="utf-8")}})  # fmt: skip
    print("workbook: Penumbra overview")

    for path in sorted((ROOT / "Analytic Rules").glob("*.yaml")):
        r = yaml.safe_load(path.read_text(encoding="utf-8"))
        props = {
            "displayName": r["name"], "description": r["description"].strip(), "severity": r["severity"],
            "enabled": True, "query": r["query"], "queryFrequency": iso(r["queryFrequency"]),
            "queryPeriod": iso(r["queryPeriod"]), "triggerOperator": "GreaterThan",
            "triggerThreshold": r["triggerThreshold"], "suppressionDuration": r["suppressionDuration"],
            "suppressionEnabled": r["suppressionEnabled"], "tactics": r["tactics"],
            "techniques": r["relevantTechniques"],
        }  # fmt: skip
        for key in (
            "eventGroupingSettings",
            "customDetails",
            "alertDetailsOverride",
            "entityMappings",
            "incidentConfiguration",
        ):
            if key in r:
                props[key] = r[key]
        put(f"{ws_id}/providers/Microsoft.SecurityInsights/alertRules/{r['id']}?api-version=2024-03-01",
            {"kind": "Scheduled", "properties": props})  # fmt: skip
        print(f"analytics rule: {r['name']}")

    print("\nDone. In the shell that will run Penumbra (PowerShell shown):\n")
    print(
        f"$env:PENUMBRA_AZURE_CLIENT_SECRET = (az ad app credential reset --id {app_id} --display-name penumbra --years 1 --query password -o tsv)"
    )
    for key, value in (("PENUMBRA_SIEM", "sentinel"), ("PENUMBRA_SENTINEL_ENDPOINT", endpoint),
                       ("PENUMBRA_SENTINEL_DCR_ID", immutable), ("PENUMBRA_AZURE_TENANT_ID", tenant),
                       ("PENUMBRA_AZURE_CLIENT_ID", app_id)):  # fmt: skip
        print(f'$env:{key} = "{value}"')
    print("uv run penumbra demo")
    print(
        "\nA new app's permission on the DCR can take a few minutes to apply; until then Sentinel answers 403"
    )
    print(
        f"and /ingest reports the batch as rejected. Remove everything: az group delete -n {rg}; az ad app delete --id {app_id}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--resource-group", default="penumbra-rg")
    parser.add_argument("--location", default="centralindia")
    parser.add_argument("--workspace", default="penumbra-law")
    parser.add_argument("--app-name", default="penumbra-ingest")
    a = parser.parse_args()
    main(a.resource_group, a.location, a.workspace, a.app_name)
