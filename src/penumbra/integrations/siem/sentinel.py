"""AzureSentinelSiem: the Logs Ingestion API, with no Azure SDK.

    POST {endpoint}/dataCollectionRules/{dcr_immutable_id}/streams/{stream}?api-version=2023-01-01
    Authorization: Bearer <token for https://monitor.azure.com/.default>

Built on the DCR path deliberately: the legacy HTTP Data Collector API retires 14 September 2026,
and since March 2024 a DCR exposes its own `logsIngestion` endpoint, so no DCE is needed
(sentinel/README.md). The app registration needs **Monitoring Metrics Publisher** on the DCR.

Configuration, all from the environment, none of it ever logged:

    PENUMBRA_SENTINEL_ENDPOINT      the DCR's logsIngestion endpoint (https://...)
    PENUMBRA_SENTINEL_DCR_ID        the DCR immutable id (dcr-...)
    PENUMBRA_AZURE_TENANT_ID
    PENUMBRA_AZURE_CLIENT_ID
    PENUMBRA_AZURE_CLIENT_SECRET

Batches are capped under the API's 1 MB request limit. A 429 or 503 is retried once after the
server's Retry-After (capped), then reported as rejected rather than retried forever: a SIEM outage
must not stall alerting.

HTTP goes through an injectable `transport` so tests run with no network and no credentials.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

from penumbra.integrations.siem.base import NotConfigured, SendResult

STREAM = "Custom-PenumbraAlerts"
API_VERSION = "2023-01-01"
SCOPE = "https://monitor.azure.com/.default"
MAX_BATCH_BYTES = 900_000  # under the 1 MB request cap, with room for the envelope
MAX_RETRY_AFTER = 10.0

# (method, url, headers, body) -> (status, headers, body)
Transport = Callable[[str, str, dict[str, str], bytes], tuple[int, dict[str, str], bytes]]


def urllib_transport(
    method: str, url: str, headers: dict[str, str], body: bytes
) -> tuple[int, dict[str, str], bytes]:
    if urllib.parse.urlparse(url).scheme != "https":
        raise ValueError("Sentinel ingestion is https only")
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    # B310 is suppressed because the scheme is checked to be https two lines up.
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:  # nosec B310
            return resp.status, dict(resp.headers), resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, dict(exc.headers or {}), exc.read()


@dataclass(frozen=True)
class SentinelConfig:
    endpoint: str
    dcr_id: str
    tenant_id: str
    client_id: str
    client_secret: str

    ENV = {
        "endpoint": "PENUMBRA_SENTINEL_ENDPOINT",
        "dcr_id": "PENUMBRA_SENTINEL_DCR_ID",
        "tenant_id": "PENUMBRA_AZURE_TENANT_ID",
        "client_id": "PENUMBRA_AZURE_CLIENT_ID",
        "client_secret": "PENUMBRA_AZURE_CLIENT_SECRET",
    }

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> SentinelConfig:
        env = dict(os.environ) if env is None else env
        missing = [var for var in cls.ENV.values() if not env.get(var)]
        if missing:
            raise NotConfigured(
                f"Sentinel connector needs {missing}. The local mock is the default until then."
            )
        values = {key: env[var] for key, var in cls.ENV.items()}
        if not values["endpoint"].startswith("https://"):
            raise NotConfigured("PENUMBRA_SENTINEL_ENDPOINT must be an https URL")
        return cls(**values)

    def __repr__(self) -> str:  # never print the secret, even by accident in a traceback
        return f"SentinelConfig(endpoint={self.endpoint!r}, dcr_id={self.dcr_id!r}, client_id={self.client_id!r})"


class AzureSentinelSiem:
    name = "azure-sentinel"

    def __init__(
        self,
        config: SentinelConfig | None = None,
        *,
        transport: Transport = urllib_transport,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.config = config or SentinelConfig.from_env()
        self.transport = transport
        self.sleep = sleep
        self._token: str | None = None
        self._token_expires = 0.0

    # --- auth ------------------------------------------------------------------------------------

    def _bearer(self) -> str:
        if self._token and time.time() < self._token_expires - 60:
            return self._token
        body = urllib.parse.urlencode(
            {
                "grant_type": "client_credentials",
                "client_id": self.config.client_id,
                "client_secret": self.config.client_secret,
                "scope": SCOPE,
            }
        ).encode()
        url = (
            f"https://login.microsoftonline.com/{urllib.parse.quote(self.config.tenant_id)}/oauth2/v2.0/token"
        )
        status, _, raw = self.transport(
            "POST", url, {"Content-Type": "application/x-www-form-urlencoded"}, body
        )
        if status != 200:
            raise NotConfigured(f"token request failed with HTTP {status}; check the app registration")
        payload = json.loads(raw)
        self._token = str(payload["access_token"])
        self._token_expires = time.time() + float(payload.get("expires_in", 3600))
        return self._token

    # --- send ------------------------------------------------------------------------------------

    def url(self) -> str:
        base = self.config.endpoint.rstrip("/")
        dcr = urllib.parse.quote(self.config.dcr_id)
        return f"{base}/dataCollectionRules/{dcr}/streams/{STREAM}?api-version={API_VERSION}"

    def send(self, records: list[dict[str, Any]]) -> SendResult:
        result = SendResult()
        for batch in batches(records):
            body = json.dumps(batch, default=str).encode()
            status = self._post(body)
            result.batches += 1
            if status in (200, 204):
                result.accepted += len(batch)
            else:
                result.rejected += len(batch)
                result.problems.append(f"batch of {len(batch)} rejected with HTTP {status}")
        return result

    def _post(self, body: bytes) -> int:
        for attempt in (1, 2):
            headers = {"Authorization": f"Bearer {self._bearer()}", "Content-Type": "application/json"}
            status, resp_headers, _ = self.transport("POST", self.url(), headers, body)
            if status in (429, 503) and attempt == 1:
                wait = _retry_after(resp_headers)
                self.sleep(wait)
                continue
            if status == 401 and attempt == 1:
                self._token = None  # expired or revoked: refresh once
                continue
            return status
        return status


def batches(records: list[dict[str, Any]], limit: int = MAX_BATCH_BYTES) -> Iterator[list[dict[str, Any]]]:
    """Split records into JSON arrays under `limit` bytes. A single oversized record goes alone."""
    current: list[dict[str, Any]] = []
    size = 2
    for record in records:
        n = len(json.dumps(record, default=str).encode()) + 1
        if current and size + n > limit:
            yield current
            current, size = [], 2
        current.append(record)
        size += n
    if current:
        yield current


def _retry_after(headers: dict[str, str]) -> float:
    for key, value in headers.items():
        if key.lower() == "retry-after":
            try:
                return min(float(value), MAX_RETRY_AFTER)
            except ValueError:
                break
    return 1.0
