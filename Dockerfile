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

# Entrypoint script: writes GCP credentials from env var to file, then starts uvicorn
# This avoids committing the SA key to git
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

CMD ["./entrypoint.sh"]
