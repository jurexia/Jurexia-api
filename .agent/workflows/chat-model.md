---
description: Current chat model configuration and history of model changes for Iurexia
---

# Chat Model Configuration

## Current Setup (as of Feb 2026)

| Function | Model | Provider | Notes |
|---|---|---|---|
| **Chat RAG** | `gpt-5-mini` | OpenAI | Best balance of quality + cost for legal chat |
| **Sentencia Analysis** | `o3-mini` | OpenAI | Reasoning-focused for judicial analysis |
| **Document Drafting** | `deepseek-reasoner` | DeepSeek | Creative legal drafting with deep reasoning |
| **Embeddings** | `text-embedding-3-large` | OpenAI | 3072-dim vectors for semantic search |

## Model History (Chat)

1. **DeepSeek Chat** → initial model
2. **o4-mini** → switched for cost efficiency, but responses became shallow
3. **gpt-5-mini** → current model, excellent quality with rich legal analysis ✅

## Where to Change

- **Chat model**: `main.py` line ~72, variable `CHAT_MODEL`
- **Sentencia model**: `main.py` line ~73, variable `SENTENCIA_MODEL`
- **DeepSeek**: `main.py`, `DEEPSEEK_CHAT_MODEL` and `DEEPSEEK_REASONER_MODEL`

## Deployment

After changing the model:
```
cd jurexia-api-git
git add main.py
git commit -m "feat(model): change CHAT_MODEL to <new_model>"
git push origin main
```
Render auto-deploys from `main` branch (~2-3 min).

## Notes

- `gpt-5-mini` produces significantly richer responses than `o4-mini` — more constitutional law citations, deeper jurisprudence analysis, better structured arguments
- DeepSeek stays for document generation — it has superior creative reasoning for legal drafting
- If upgrading further within OpenAI: `gpt-5-mini` → `gpt-5` (higher cost but maximum quality)
