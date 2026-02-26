# ── Stage 1: Build ───────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder
ARG BUILDKIT_INLINE_CACHE=1

WORKDIR /app

# Install system deps needed for some packages (python-docx, fastembed, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (layer caching — only reinstalls if requirements change)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ── Stage 2: Runtime ─────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime
# Enable inline cache metadata for --cache-from to work across builds
ARG BUILDKIT_INLINE_CACHE=1

WORKDIR /app

# Install runtime system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder stage
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY main.py .
COPY cache_manager.py .

# Copy legal corpus for Gemini context caching (12 files, ~4.1MB)
COPY cache_corpus/ ./cache_corpus/

# Cloud Run injects $PORT at runtime — uvicorn must listen on it
# BM25 model loads asynchronously in background at startup (see main.py lifespan)
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080} --workers 4 --timeout-keep-alive 120 --log-level info"]
