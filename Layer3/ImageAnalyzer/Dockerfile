FROM python:3.11-slim

WORKDIR /app

# System deps for Pillow
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Install CPU-only torch first (small), then swap for GPU at runtime if needed
COPY requirements.txt .
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu128 && \
    pip install --no-cache-dir -r requirements.txt

# Copy source (model weights mounted as a volume at runtime)
COPY api.py dataset.py model.py predict.py ./

EXPOSE 8002

CMD ["python", "api.py"]
