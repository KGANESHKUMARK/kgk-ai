# KGK AI — Dockerfile
# Optimized for Hugging Face Spaces and local Docker deployment.
# Multi-stage build for smaller image size.

# --- Stage 1: Builder ---
FROM python:3.12-slim AS builder

WORKDIR /build

# System dependencies for building
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies to a separate directory
COPY requirements.txt .
RUN pip install --no-cache-dir --target=/install -r requirements.txt

# --- Stage 2: Runtime ---
FROM python:3.12-slim

WORKDIR /app

# Install runtime system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder
COPY --from=builder /install /usr/local/lib/python3.12/site-packages/

# Copy application code
COPY . .

# Create non-root user for security
RUN useradd -m -u 1000 kgk && \
    mkdir -p /app/data /app/logs /app/knowledge && \
    chown -R kgk:kgk /app

USER kgk

# Expose port (Gradio/FastAPI default)
EXPOSE 7860

# Environment
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV HOME=/home/kgk

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/')" || exit 1

# Launch Gradio UI (default)
# For API mode: docker run ... python -m app.main --api
CMD ["python", "app.py"]
