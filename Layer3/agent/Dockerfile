# Build context must be the project root (docker-compose sets context: ..)

FROM python:3.11-slim

# System libraries required by Pillow / OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# ── Python dependencies ────────────────────────────────────────────────────────
COPY Layer3/agent/requirements.txt /tmp/requirements.txt

# CPU-only torch — EC2 standard instances have no GPU; saves ~3.5 GB vs CUDA build
RUN pip install --no-cache-dir \
        torch torchvision \
        --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Pre-download the multilingual embedding model so first request is instant
RUN python -c \
    "from sentence_transformers import SentenceTransformer; \
     SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')"

# ── Project structure ──────────────────────────────────────────────────────────
WORKDIR /workspace

# Shared config module (project root)
COPY crop_config.py .

# Crop metadata: YAML + CSVs only
COPY data/data/onion/crop.yaml      data/data/onion/crop.yaml
COPY data/data/onion/disease_info   data/data/onion/disease_info/

# RAG source files (imported directly by agent nodes)
COPY Layer3/RAG/rag_chain.py  Layer3/RAG/
COPY Layer3/RAG/retriever.py  Layer3/RAG/
COPY Layer3/RAG/ingest.py     Layer3/RAG/

# ImageAnalyzer source files (imported directly by agent nodes)
COPY Layer3/ImageAnalyzer/predict.py  Layer3/ImageAnalyzer/
COPY Layer3/ImageAnalyzer/model.py    Layer3/ImageAnalyzer/
COPY Layer3/ImageAnalyzer/dataset.py  Layer3/ImageAnalyzer/

# Guardrails package
COPY Layer3/guardrails/  Layer3/guardrails/

# Agent source files
COPY Layer3/agent/  Layer3/agent/

# ── Runtime ────────────────────────────────────────────────────────────────────
# ChromaDB index — mount Layer3/RAG/chroma_db/ as a volume
# Model weights — mount Layer3/ImageAnalyzer/models/ as a volume
EXPOSE 8001

WORKDIR /workspace/Layer3/agent
CMD ["python", "api.py"]
