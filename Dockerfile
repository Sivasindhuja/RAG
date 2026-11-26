# Backend deployment image for SATCOM NGP RAG (FastAPI + Weaviate Cloud production path)
# Production migration (Task 38): Weaviate selected for Render Free 512MB constraint.
# This image installs lightweight Weaviate runtime deps and does NOT install torch/sentence-transformers/chromadb.
# For Chroma/local-index path, use requirements.runtime.txt and install CPU torch separately.

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    RAG_RETRIEVER_BACKEND=weaviate

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

COPY requirements.weaviate.runtime.txt ./
RUN pip install -r requirements.weaviate.runtime.txt

COPY . .

EXPOSE 8000

# Use PORT when provided by container platforms (e.g. Render Docker).
CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
