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

# Not /app/artifacts: the base image declares that a VOLUME, and content added under a volume path
# after the declaration is not guaranteed to survive. A separate root, pointed at by the env var.
COPY --chown=penumbra:penumbra artifacts/reports/ /app/bundle/reports/
COPY --chown=penumbra:penumbra tests/fixtures/demo_alerts.json tests/fixtures/incidents_cicids.json /app/seed/

# The copilot's ATT&CK corpus, staged by scripts/deploy_azure.py (may be an empty directory).
COPY --chown=penumbra:penumbra deploy/data/ /app/data/

ENV PENUMBRA_ARTIFACT_ROOT=/app/bundle \
    PENUMBRA_SEED_FIXTURES=/app/seed
