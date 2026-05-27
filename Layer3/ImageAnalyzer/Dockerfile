# Build context must be the project root (docker-compose sets context: ..)
# so that crop_config.py and data/data/<crop>/crop.yaml are reachable.

FROM python:3.11-slim

# System libraries required by Pillow / OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# ── Python dependencies ────────────────────────────────────────────────────────
COPY Layer3/ImageAnalyzer/requirements.txt /tmp/requirements.txt

# Install PyTorch (CUDA 12.8) first so pip doesn't fight over it
RUN pip install --no-cache-dir \
        torch torchvision \
        --index-url https://download.pytorch.org/whl/cu128

RUN pip install --no-cache-dir -r /tmp/requirements.txt

# ── Project structure ──────────────────────────────────────────────────────────
# Mirror the on-disk layout so sys.path.insert(0, parents[2]) resolves to /workspace
WORKDIR /workspace

# Shared config module (project root)
COPY crop_config.py .

# Crop metadata: YAML + CSVs only — images are large and not needed for inference
COPY data/data/onion/crop.yaml      data/data/onion/crop.yaml
COPY data/data/onion/disease_info   data/data/onion/disease_info/

# Service source files
COPY Layer3/ImageAnalyzer/api.py     Layer3/ImageAnalyzer/
COPY Layer3/ImageAnalyzer/dataset.py Layer3/ImageAnalyzer/
COPY Layer3/ImageAnalyzer/model.py   Layer3/ImageAnalyzer/
COPY Layer3/ImageAnalyzer/predict.py Layer3/ImageAnalyzer/

# ── Runtime ────────────────────────────────────────────────────────────────────
# Model weights are NOT baked in — mount Layer3/ImageAnalyzer/models/ as a volume.
EXPOSE 8002

WORKDIR /workspace/Layer3/ImageAnalyzer
CMD ["python", "api.py"]
