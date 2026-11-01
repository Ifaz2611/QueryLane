# Cloud-only image.
# Base tag python:3.11-slim is pinned by its multi-arch manifest digest
# (Docker Hub, resolved 2026-10-28) so every rebuild uses the same base.
FROM python:3.11-slim@sha256:673719e071ecf27434fd2635bbd8d00f49f4fdd81f243143438a42ced8bedf60
LABEL org.opencontainers.image.title="QueryLane" \
      org.opencontainers.image.description="Cost-aware LLM query routing (cloud-only image)"

ENV HOME=/home/querylane \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Run as an unprivileged user. /output is created here so the harness can
# mount over it; when bind-mounting a host directory, make that directory
# writable by uid 10001 (the README shows the chmod for local runs).
RUN groupadd --gid 10001 querylane \
 && useradd --uid 10001 --gid querylane --create-home \
      --home-dir /home/querylane --shell /usr/sbin/nologin querylane \
 && mkdir -p /output \
 && chown -R querylane:querylane /app /output /home/querylane

# Fully pinned + hash-verified runtime dependencies (see requirements.lock).
COPY requirements.txt requirements.lock ./
RUN pip install --no-cache-dir --require-hashes -r requirements.lock

COPY --chown=querylane:querylane agent/ ./agent/

USER querylane

# Phase 2 uncomments the model bundle:
# COPY --chown=querylane:querylane models/ ./models/

ENTRYPOINT ["python", "-m", "agent.main"]
