#!/bin/sh
# ── Iurexia API Entrypoint ──────────────────────────────────────────────────
# Writes GCP Service Account credentials from env var to file
# so that google-cloud SDKs can authenticate via GOOGLE_APPLICATION_CREDENTIALS

if [ -n "$GCP_SA_KEY_JSON" ]; then
    echo "$GCP_SA_KEY_JSON" > /app/gcp-sa-key.json
    export GOOGLE_APPLICATION_CREDENTIALS=/app/gcp-sa-key.json
    echo "[entrypoint] GCP credentials written to /app/gcp-sa-key.json"
else
    echo "[entrypoint] WARNING: GCP_SA_KEY_JSON not set — Vertex AI will not work"
fi

# Start uvicorn with the port Render injects
exec uvicorn main:app \
    --host 0.0.0.0 \
    --port ${PORT:-8080} \
    --workers 4 \
    --timeout-keep-alive 120 \
    --log-level info
