# Backend deployment image for SATCOM NGP RAG (FastAPI + ChromaDB + sentence-transformers)
# Intended for a container host with persistent disk for RAG_PERSIST_DIR/model cache.
# Uses CPU-only PyTorch to reduce image size; preserves existing RAG behavior/config.

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface \
    TRANSFORMERS_CACHE=/app/.cache/huggingface \
    SENTENCE_TRANSFORMERS_HOME=/app/.cache/sentence-transformers \
    RAG_PERSIST_DIR=/app/chroma_db_production

WORKDIR /app

# System deps (minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only torch first to avoid large CUDA wheels
RUN pip install --upgrade pip && \
    pip install --index-url https://download.pytorch.org/whl/cpu torch

# Runtime dependencies (minimal for serving)
COPY requirements.runtime.txt requirements.txt ./
RUN if [ -f requirements.runtime.txt ]; then pip install -r requirements.runtime.txt; else pip install -r requirements.txt; fi

# Copy application
COPY . .

# Pre-create writable/cache dirs (persist_dir should be mounted for true persistence)
RUN mkdir -p /app/chroma_db_production /app/.cache/huggingface /app/.cache/sentence-transformers

EXPOSE 8000

# Default: serve API. Ensure RAG_PERSIST_DIR points to persistent storage in production.
# Use PORT when provided by container platforms (e.g. Render Docker).
CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
