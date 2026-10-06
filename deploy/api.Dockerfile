# Deploy-only layer over the API image: the evaluation reports the console renders, and the demo
# fixtures a public instance seeds itself with. Neither is in git (artifacts/ is generated), so the
# main Dockerfile - which CI builds from a clean checkout - cannot carry them.
#
# Built by scripts/deploy_azure.py with BASE set to the API image it has just built.

ARG BASE
FROM ${BASE}

# The source, always from this checkout. The base installs the project editable from /app/src, so
# this is what runs even when BASE is an older build whose dependencies are still current (which is
# what `deploy_azure.py --base-image` relies on to skip re-downloading every dependency).
COPY --chown=penumbra:penumbra src/ /app/src/

# Dependencies as locked. A no-op on a freshly built base; on a reused one (`--base-image`) it brings
# whatever the lockfile has moved since that base was built, so a security bump (pyjwt 2.15.1,
# urllib3 2.8.0) does not wait for a full rebuild. --inexact: add or change, never remove.
COPY --from=ghcr.io/astral-sh/uv:0.5.11 /uv /bin/uv
COPY --chown=penumbra:penumbra pyproject.toml uv.lock README.md /tmp/lock/
RUN cd /tmp/lock \
    && UV_PROJECT_ENVIRONMENT=/app/.venv UV_CACHE_DIR=/tmp/uv-cache UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never \
       uv sync --frozen --inexact --no-install-project --extra eval --extra api --extra rag --extra drift --extra pcap \
    && rm -rf /tmp/uv-cache /tmp/lock

# Not /app/artifacts: the base image declares that a VOLUME, and content added under a volume path
# after the declaration is not guaranteed to survive. A separate root, pointed at by the env var.
COPY --chown=penumbra:penumbra artifacts/reports/ /app/bundle/reports/
COPY --chown=penumbra:penumbra tests/fixtures/demo_alerts.json tests/fixtures/incidents_cicids.json /app/seed/

# The copilot's ATT&CK corpus, staged by scripts/deploy_azure.py (may be an empty directory).
COPY --chown=penumbra:penumbra deploy/data/ /app/data/

# Live scoring (ADR-0006): the registry champions, verified against their manifests when staged,
# and the held-out sample pools the scorer draws from.
COPY --chown=penumbra:penumbra deploy/models/ /app/bundle/models/
COPY --chown=penumbra:penumbra deploy/samples/ /app/bundle/samples/

ENV PENUMBRA_ARTIFACT_ROOT=/app/bundle \
    PENUMBRA_SEED_FIXTURES=/app/seed
