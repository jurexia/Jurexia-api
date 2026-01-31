# Jurexia API

Backend FastAPI para la plataforma Jurexia - IA legal para México.

## Deploy en Railway

1. Conecta este repo a Railway
2. Configura las variables de entorno (ver abajo)
3. Railway detectará automáticamente Python

## Variables de Entorno

```
QDRANT_URL=tu_url_qdrant
QDRANT_API_KEY=tu_api_key_qdrant
OPENAI_API_KEY=tu_api_key_openai
```

## Endpoints

- `GET /health` - Estado del sistema
- `POST /search` - Búsqueda híbrida
- `POST /chat` - Chat con streaming
- `POST /audit` - Auditoría de documentos
