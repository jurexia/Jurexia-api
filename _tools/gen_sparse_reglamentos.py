#!/usr/bin/env python3
"""
gen_sparse_reglamentos.py — Genera BM25 sparse vectors para los reglamentos municipales

Los reglamentos fueron ingestados con dense-only. Este script:
1. Consulta Qdrant por chunks con categoria="reglamento_municipal"
2. Genera sparse vectors con fastembed (Qdrant/bm25)
3. Actualiza cada punto con el sparse vector
"""

import sys
import os
import time

from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Filter,
    FieldCondition,
    MatchValue,
    NamedSparseVector,
    SparseVector,
    PointVectors,
    SetPayloadOperation,
    PointIdsList,
)
from fastembed import SparseTextEmbedding

QDRANT_URL = os.environ.get(
    "QDRANT_URL",
    "https://d6766dbb-cf4c-40a2-a636-78060cc09ccc.us-east4-0.gcp.cloud.qdrant.io",
)
QDRANT_API_KEY = os.environ.get(
    "QDRANT_API_KEY",
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.4hZwbdZT6esMLx7hjHCi79hD5gLpEAVphmuNGYB3A0Y",
)
COLLECTION = "leyes_queretaro"
BATCH_SIZE = 20


def main():
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

    print("═══ Generando BM25 Sparse Vectors para Reglamentos Municipales ═══")
    
    # Init sparse encoder
    print("   Cargando modelo BM25 (fastembed)...")
    sparse_encoder = SparseTextEmbedding(model_name="Qdrant/bm25")
    print("   ✅ Modelo cargado")
    
    # Connect to Qdrant
    qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    
    # Get all reglamento_municipal chunks
    print(f"\n   Consultando chunks con categoria='reglamento_municipal' en {COLLECTION}...")
    
    reglamento_filter = Filter(
        must=[
            FieldCondition(key="categoria", match=MatchValue(value="reglamento_municipal"))
        ]
    )
    
    # Scroll through all matching points
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
    
    print(f"   📊 Total chunks encontrados: {len(all_points)}")
    
    if not all_points:
        print("   ❌ No se encontraron chunks de reglamentos. ¿Ya se ingestaron?")
        return
    
    # Generate sparse vectors and update
    start = time.time()
    updated = 0
    failed = 0
    
    for batch_start in range(0, len(all_points), BATCH_SIZE):
        batch = all_points[batch_start:batch_start + BATCH_SIZE]
        texts = [p.payload.get("texto", "") for p in batch]
        
        try:
            sparse_embeddings = list(sparse_encoder.passage_embed(texts))
            
            for point, sp_emb in zip(batch, sparse_embeddings):
                if len(sp_emb.indices) == 0:
                    continue
                
                try:
                    qdrant.update_vectors(
                        collection_name=COLLECTION,
                        points=[
                            PointVectors(
                                id=point.id,
                                vector={
                                    "sparse": SparseVector(
                                        indices=sp_emb.indices.tolist(),
                                        values=sp_emb.values.tolist(),
                                    )
                                }
                            )
                        ]
                    )
                    updated += 1
                except Exception as e:
                    print(f"   ❌ Error updating point {point.id}: {e}")
                    failed += 1
            
            progress = min(batch_start + BATCH_SIZE, len(all_points))
            print(f"   📊 Processed {progress}/{len(all_points)} ({updated} updated, {failed} failed)")
            
        except Exception as e:
            print(f"   ❌ Batch error at {batch_start}: {e}")
            failed += len(batch)
    
    elapsed = time.time() - start
    print(f"\n   ✅ COMPLETADO")
    print(f"      Updated: {updated}/{len(all_points)}")
    print(f"      Failed: {failed}")
    print(f"      Time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
