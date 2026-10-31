FROM python:3.11-slim
LABEL org.opencontainers.image.title="QueryLane" \
      org.opencontainers.image.description="Cost-aware LLM query routing (cloud-only image)"

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY agent/ ./agent/

# Phase 2 uncomments the model bundle:
# COPY models/ ./models/

ENTRYPOINT ["python", "-m", "agent.main"]
