# Build context must be the project root (docker-compose sets context: ..)

FROM python:3.11-slim

# ── Python dependencies ────────────────────────────────────────────────────────
COPY Layer3/guardrails/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# ── Project structure ──────────────────────────────────────────────────────────
WORKDIR /workspace

# Guardrails package (service + logic in one directory)
COPY Layer3/guardrails/ Layer3/guardrails/

# ── Runtime ────────────────────────────────────────────────────────────────────
EXPOSE 8003

WORKDIR /workspace/Layer3/guardrails
CMD ["python", "api.py"]
