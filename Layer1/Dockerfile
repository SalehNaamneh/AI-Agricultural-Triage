# Build context must be the project root (docker-compose sets context: ..)
# so that crop_config.py and data/data/<crop>/crop.yaml are reachable.

FROM python:3.11-slim

# ── Python dependencies ────────────────────────────────────────────────────────
COPY Layer1/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# ── Project structure ──────────────────────────────────────────────────────────
WORKDIR /workspace

# Shared config module (project root)
COPY crop_config.py .

# Crop metadata: YAML + CSVs only (training images are not included in the image)
COPY data/data/onion/crop.yaml    data/data/onion/crop.yaml
COPY data/data/onion/disease_info data/data/onion/disease_info/

# App source
COPY Layer1/app.py Layer1/

# ── Runtime ────────────────────────────────────────────────────────────────────
# Service URLs are injected at runtime via environment variables:
#   RAG_URL            (default: http://localhost:8000)
#   IMAGE_ANALYZER_URL (default: http://localhost:8002)
#   N8N_WEBHOOK_URL    (default: http://localhost:5678/webhook/agritriage)
EXPOSE 7860

WORKDIR /workspace/Layer1
CMD ["python", "app.py"]
