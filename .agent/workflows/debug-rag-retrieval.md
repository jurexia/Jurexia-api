---
description: How to debug persistent RAG retrieval issues where expected documents are NOT being returned to the LLM
---

# Debug RAG Retrieval Issues

## When to Use
When the LLM says "no se recuperó" or "no se encontró" a specific law/document, but you believe it exists in Qdrant.

## Step-by-Step Debugging

### 1. Confirm documents exist in Qdrant
Run a direct scroll query on the collection to verify the documents are actually ingested:
```python
from qdrant_client import QdrantClient
client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
results = client.scroll(
    collection_name="leyes_estatales",
    scroll_filter=Filter(must=[
        FieldCondition(key="entidad", match=MatchValue(value="CIUDAD_DE_MEXICO"))
    ]),
    limit=10
)
```

### 2. Add production logging (CRITICAL)
// turbo
Add print statements in the chat endpoint AFTER `format_results_as_xml` to see EXACTLY what documents reach the LLM context:
```python
estatales_in_context = [r for r in search_results if r.silo == "leyes_estatales"]
print(f"🔬 CONTEXT AUDIT (estado={effective_estado}):")
print(f"   Total docs: {len(search_results)}")
print(f"   Leyes estatales: {len(estatales_in_context)}")
for r in estatales_in_context[:5]:
    print(f"   → ref={r.ref}, origen={r.origen[:50]}, score={r.score:.4f}")
```

### 3. Read logs from Render CAREFULLY
The logs will reveal the REAL problem. Common patterns:

**Pattern A: Wrong documents returned (MOST COMMON)**
```
⭐ [leyes_estatales] ref=Art. 6 origen=Reglamento del Comité Técnico...
```
If you see WRONG documents with LOW scores (0.3-0.4), there's a FILTER BUG excluding the right ones.

**Pattern B: Zero estatales returned**
```
Leyes estatales: 0
```
The estado filter doesn't match. Check `normalize_estado()` and the `entidad` field in Qdrant.

**Pattern C: Correct documents but LLM ignores them**
```
⭐ [leyes_estatales] ref=Art. 43 origen=Ley de Propiedad...  score=0.73
```
If correct docs are there but LLM still says "no se recuperó", it's a prompting issue. Check document ORDER in context — LLM pays more attention to first documents.

### 4. Known Root Causes

#### ⚠️ Qdrant `should` + `must` = HARD FILTER (NOT soft boost!)
**This was the root cause of the CDMX condominium law bug.**

In Qdrant's Filter API:
- `must` alone = hard filter (all conditions must match)
- `should` alone = at least one must match  
- **`must` + `should` together = ALL must conditions AND AT LEAST ONE should condition must match**

This means `should` is NOT a soft boost when combined with `must`. It EXCLUDES documents that don't match any `should` condition.

**Example of the bug:**
```python
# BROKEN: This EXCLUDES documents where tipo_codigo != "URBANO"
Filter(
    must=[FieldCondition(key="entidad", match=MatchValue(value="CIUDAD_DE_MEXICO"))],
    should=[FieldCondition(key="tipo_codigo", match=MatchValue(value="URBANO"))]
)
```
The Ley de Propiedad en Condominio had `tipo_codigo=CIVIL`, so it was filtered OUT.

**Fix:** Never combine `should` with `must` unless you intentionally want hard filtering on the `should` conditions too.

#### Document ordering matters
LLMs pay more attention to documents at the START of the context. When estado is selected, put `leyes_estatales` FIRST in the merged results, not after federal/constitutional documents.

#### Prompt fixes rarely work for retrieval issues  
If the LLM says "document not found" and you try to fix it by changing the system prompt, it usually means the document genuinely isn't in the context. Don't waste time on prompt engineering — look at the LOGS to verify what documents are actually being sent.

### 5. Quick Test Script Template
```python
# Test the exact production search flow
from main import hybrid_search_all_silos, format_results_as_xml
results = await hybrid_search_all_silos(
    query="YOUR EXACT QUERY HERE",
    estado="ESTADO_VALUE",
    top_k=30,
)
estatales = [r for r in results if r.silo == "leyes_estatales"]
print(f"Estatales: {len(estatales)}")
for r in estatales:
    print(f"  {r.ref} | {r.origen[:50]} | score={r.score:.4f}")
```
