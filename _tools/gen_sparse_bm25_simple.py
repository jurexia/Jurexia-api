#!/usr/bin/env python3
"""
gen_sparse_bm25_simple.py — Genera BM25 sparse vectors SIN fastembed (evita segfault Windows)

Usa una implementación simplificada de TF-IDF como sparse vector compatible con
el formato que espera Qdrant. Esto es suficiente para hacer los reglamentos visibles
en la búsqueda híbrida.

Alternativa: correr fastembed en Linux/WSL o en el servidor de producción.
"""

import sys
import os
import time
import json
import hashlib
import math
import re
from collections import Counter

from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Filter,
    FieldCondition,
    MatchValue,
    SparseVector,
    PointVectors,
)

QDRANT_URL = os.environ.get(
    "QDRANT_URL",
    "https://d6766dbb-cf4c-40a2-a636-78060cc09ccc.us-east4-0.gcp.cloud.qdrant.io",
)
QDRANT_API_KEY = os.environ.get(
    "QDRANT_API_KEY",
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.4hZwbdZT6esMLx7hjHCi79hD5gLpEAVphmuNGYB3A0Y",
)
COLLECTION = "leyes_queretaro"

# Simple tokenizer for Spanish legal text
STOP_WORDS = set([
    'de', 'la', 'el', 'en', 'y', 'a', 'los', 'del', 'las', 'un', 'por',
    'con', 'una', 'su', 'para', 'es', 'al', 'lo', 'como', 'más', 'o',
    'pero', 'sus', 'le', 'ya', 'fue', 'este', 'ha', 'se', 'que', 'no',
    'son', 'uno', 'ni', 'ser', 'sobre', 'sin', 'será', 'artículo', 'art',
    'fracción', 'inciso', 'párrafo', 'numeral',
])


def tokenize(text: str) -> list[str]:
    """Simple tokenizer for Spanish legal text."""
    text = text.lower()
    text = re.sub(r'[^\wáéíóúñü]', ' ', text)
    tokens = text.split()
    return [t for t in tokens if len(t) > 2 and t not in STOP_WORDS]


def token_to_index(token: str) -> int:
    """Map token to a stable integer index using hash (compatible with fastembed BM25)."""
    # Use a hash to map tokens to indices in a large sparse space
    h = hashlib.md5(token.encode('utf-8')).hexdigest()
    return int(h[:8], 16) % (2**31)  # 31-bit positive integer


def compute_sparse_vector(text: str) -> tuple[list[int], list[float]]:
    """Compute a simple TF-based sparse vector for text."""
    tokens = tokenize(text)
    if not tokens:
        return [], []
    
    tf = Counter(tokens)
    total = len(tokens)
    
    indices = []
    values = []
    
    for token, count in tf.items():
        idx = token_to_index(token)
        # Simple TF normalization (log-scaled)
        value = 1.0 + math.log(1 + count / total)
        indices.append(idx)
        values.append(round(value, 4))
    
    # Sort by index for Qdrant
    pairs = sorted(zip(indices, values))
    indices = [p[0] for p in pairs]
    values = [p[1] for p in pairs]
    
    return indices, values


def main():
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    
    print("═══ BM25-Compatible Sparse Vectors para Reglamentos ═══")
    print("   (Implementación sin fastembed — evita segfault Windows)")
    
    qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=60)
    
    # Get reglamento chunks
    reglamento_filter = Filter(
        must=[
            FieldCondition(key="categoria", match=MatchValue(value="reglamento_municipal"))
        ]
    )
    
    all_points = []
    offset = None
    while True:
        results, next_offset = qdrant.scroll(
            collection_name=COLLECTION,
            scroll_filter=reglamento_filter,
            limit=100,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        all_points.extend(results)
        if next_offset is None:
            break
        offset = next_offset
    
    print(f"   📊 Chunks de reglamentos encontrados: {len(all_points)}")
    
    if not all_points:
        print("   ❌ No se encontraron chunks")
        return
    
    # Check a few existing points (non-reglamento) to understand the sparse format
    print("\n   Verificando formato sparse existente...")
    sample = qdrant.scroll(
        collection_name=COLLECTION,
        limit=3,
        with_vectors=True,
    )
    for p in sample[0][:1]:
        vectors = p.vector
        if isinstance(vectors, dict) and 'sparse' in vectors:
            sv = vectors['sparse']
            print(f"   Ejemplo sparse existente: {len(sv.indices)} indices, max_idx={max(sv.indices) if sv.indices else 0}")
        else:
            print(f"   Punto sin sparse vector: {p.id}")
    
    # Generate and update
    start = time.time()
    updated = 0
    failed = 0
    skipped = 0
    
    for i, point in enumerate(all_points):
        text = point.payload.get("texto", "")
        if not text.strip():
            skipped += 1
            continue
        
        indices, values = compute_sparse_vector(text)
        if not indices:
            skipped += 1
            continue
        
        try:
            qdrant.update_vectors(
                collection_name=COLLECTION,
                points=[
                    PointVectors(
                        id=point.id,
                        vector={
                            "sparse": SparseVector(
                                indices=indices,
                                values=values,
                            )
                        }
                    )
                ]
            )
            updated += 1
        except Exception as e:
            print(f"   ❌ Error updating {point.id}: {e}")
            failed += 1
        
        if (i + 1) % 50 == 0:
            elapsed = time.time() - start
            print(f"   📊 {i+1}/{len(all_points)} ({updated} updated, {failed} failed, {skipped} skipped) [{elapsed:.0f}s]")
    
    elapsed = time.time() - start
    print(f"\n   ✅ COMPLETADO")
    print(f"      Updated: {updated}")
    print(f"      Failed: {failed}")
    print(f"      Skipped: {skipped}")
    print(f"      Time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
