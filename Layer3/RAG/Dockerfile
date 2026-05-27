# Build context must be the project root (docker-compose sets context: ..)

FROM python:3.11-slim

# ── Python dependencies ────────────────────────────────────────────────────────
COPY Layer3/RAG/requirements.txt /tmp/requirements.txt

# CPU-only torch — RAG only needs sentence-transformers for embeddings (no GPU)
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

# Service source files
COPY Layer3/RAG/api.py       Layer3/RAG/
COPY Layer3/RAG/rag_chain.py Layer3/RAG/
COPY Layer3/RAG/retriever.py Layer3/RAG/
COPY Layer3/RAG/ingest.py    Layer3/RAG/

# Guardrails package (imported by api.py via sys.path pointing at Layer3/)
COPY Layer3/guardrails/      Layer3/guardrails/

# ── Runtime ────────────────────────────────────────────────────────────────────
# ChromaDB index is NOT baked in — mount Layer3/RAG/chroma_db/ as a volume.
# On first run (empty volume), retriever.py will call build_index() automatically.
EXPOSE 8000

WORKDIR /workspace/Layer3/RAG
CMD ["python", "api.py"]
