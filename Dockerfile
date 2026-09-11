# Penumbra API image.
#
# Two-stage so the runtime layer carries no build toolchain and no uv cache.
#
# What this image deliberately does NOT do: train a model. Training on a 16 GB machine that is also
# running a browser, a Next dev server and a database is how a demo dies, and a container that
# trains on start has an unbounded, unpredictable startup time. Models are fitted on the host with
# `penumbra fit` and the versioned artifact is mounted in at /app/artifacts.
#
# It also does not ship demo users: PENUMBRA_ALLOW_DEMO_USERS is deliberately unset, so an image
# built from this repository has no known credentials.

FROM python:3.12-slim-bookworm AS builder

# uv resolves and installs from the lockfile, so the image gets exactly the dependency set the
# tests ran against rather than whatever PyPI serves today.
COPY --from=ghcr.io/astral-sh/uv:0.5.11 /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# Dependencies first, in their own layer: they change far less often than the source, so a source
# edit does not re-resolve 189 packages.
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --extra eval --extra api --extra rag --extra drift

COPY src/ ./src/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --extra eval --extra api --extra rag --extra drift


FROM python:3.12-slim-bookworm AS runtime

# Runs unprivileged. Nothing here needs root, and a detector that cannot write outside its own
# directory is one fewer thing to reason about if it is ever compromised.
RUN useradd --create-home --uid 10001 penumbra

WORKDIR /app

COPY --from=builder --chown=penumbra:penumbra /app/.venv /app/.venv
COPY --from=builder --chown=penumbra:penumbra /app/src /app/src
COPY --chown=penumbra:penumbra pyproject.toml README.md ./

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PENUMBRA_DATA_ROOT=/app/data \
    PENUMBRA_ARTIFACT_ROOT=/app/artifacts

# Bind-mount a trained model in here. Empty is a valid state: the API starts, /health answers, and
# scoring endpoints report that no detector is loaded.
RUN mkdir -p /app/data /app/artifacts && chown -R penumbra:penumbra /app/data /app/artifacts
VOLUME ["/app/artifacts"]

USER penumbra
EXPOSE 8000

# /health reports the audit chain state as well as liveness, so an unhealthy container and a
# tampered audit log surface through the same probe.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"

CMD ["uvicorn", "penumbra.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
