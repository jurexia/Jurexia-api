#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
═══════════════════════════════════════════════════════════════════
 Re-ingesta BM25: Genera sparse vectors reales para leyes_estatales
═══════════════════════════════════════════════════════════════════
 
 El script de ingesta original subió sparse vacíos ({indices:[], values:[]}).
 Este script:
 1. Scrollea TODOS los puntos de leyes_estatales (filtro opcional por entidad)
 2. Genera BM25 sparse vectors REALES usando fastembed Qdrant/bm25
 3. Actualiza cada punto in-place (preserva dense vector y payload)
 
 COSTO: $0 — fastembed corre localmente, no usa API de OpenAI
 TIEMPO: ~15-30 min para 15K puntos (Querétaro)
         ~2-3 horas para 173K puntos (toda la colección)
"""

import os
import sys
import time
import argparse
from typing import List, Optional

from qdrant_client import QdrantClient
from qdrant_client.models import (
    PointVectors,
    Filter,
    FieldCondition,
    MatchValue,
    NamedSparseVector,
    SparseVector,
)
from fastembed import SparseTextEmbedding
from tqdm import tqdm

# ════════════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ════════════════════════════════════════════════════════════════

QDRANT_URL = os.environ.get(
    "QDRANT_URL",
    "https://d6766dbb-cf4c-40a2-a636-78060cc09ccc.us-east4-0.gcp.cloud.qdrant.io"
)
QDRANT_API_KEY = os.environ.get(
    "QDRANT_API_KEY",
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.4hZwbdZT6esMLx7hjHCi79hD5gLpEAVphmuNGYB3A0Y"
)

COLLECTION = "leyes_estatales"
BATCH_SIZE = 50  # puntos por update batch
SCROLL_LIMIT = 100  # puntos por scroll request


def get_sparse_embedding(encoder: SparseTextEmbedding, text: str) -> SparseVector:
    """Genera BM25 sparse vector para un texto de documento."""
    # passage_embed para documentos (no query_embed que es para búsquedas)
    embeddings = list(encoder.passage_embed([text]))
    if not embeddings:
        return SparseVector(indices=[], values=[])
    
    sparse = embeddings[0]
    return SparseVector(
        indices=sparse.indices.tolist(),
        values=sparse.values.tolist(),
    )


def count_points(client: QdrantClient, entidad: Optional[str] = None) -> int:
    """Cuenta puntos en la colección, opcionalmente filtrados por entidad."""
    filter_ = None
    if entidad:
        filter_ = Filter(
            must=[FieldCondition(key="entidad", match=MatchValue(value=entidad))]
        )
    result = client.count(collection_name=COLLECTION, count_filter=filter_)
    return result.count


def process_batch(
    client: QdrantClient,
    encoder: SparseTextEmbedding,
    points: list,
    dry_run: bool = False
) -> int:
    """Genera sparse vectors y actualiza un batch de puntos."""
    updates = []
    
    for point in points:
        payload = point.payload or {}
        texto = payload.get("texto", payload.get("text", ""))
        
        if not texto:
            continue
        
        # Generar BM25 sparse vector real
        sparse = get_sparse_embedding(encoder, texto)
        
        if len(sparse.indices) == 0:
            continue
        
        updates.append(
            PointVectors(
                id=point.id,
                vector={
                    "sparse": sparse,
                }
            )
        )
    
    if updates and not dry_run:
        client.update_vectors(
            collection_name=COLLECTION,
            points=updates,
        )
    
    return len(updates)


def main():
    parser = argparse.ArgumentParser(
        description="Re-ingesta BM25 sparse vectors para leyes_estatales"
    )
    parser.add_argument(
        "--entidad",
        type=str,
        default=None,
        help="Filtrar por entidad (ej: QUERETARO, CIUDAD_DE_MEXICO). Si no se especifica, procesa TODA la colección."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Solo genera sparse vectors pero no actualiza Qdrant"
    )
    args = parser.parse_args()
    
    print("╔" + "═" * 68 + "╗")
    print("║   RE-INGESTA BM25 — Sparse Vectors Reales                        ║")
    print("╚" + "═" * 68 + "╝")
    
    # Inicializar
    print("\n[1/3] Inicializando...")
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    encoder = SparseTextEmbedding(model_name="Qdrant/bm25")
    
    # Contar puntos
    total = count_points(client, args.entidad)
    entidad_label = args.entidad or "TODA LA COLECCIÓN"
    print(f"   Colección: {COLLECTION}")
    print(f"   Entidad: {entidad_label}")
    print(f"   Puntos a procesar: {total:,}")
    print(f"   Dry run: {args.dry_run}")
    
    if total == 0:
        print("\n✅ No hay puntos que procesar.")
        return
    
    # Confirmar
    if not args.dry_run:
        print(f"\n⚠️  Se actualizarán {total:,} puntos en Qdrant.")
        confirm = input("   ¿Continuar? (s/N): ").strip().lower()
        if confirm != "s":
            print("   Cancelado.")
            return
    
    # Scroll y procesar
    print(f"\n[2/3] Procesando {total:,} puntos...")
    
    scroll_filter = None
    if args.entidad:
        scroll_filter = Filter(
            must=[FieldCondition(key="entidad", match=MatchValue(value=args.entidad))]
        )
    
    offset = None
    processed = 0
    updated = 0
    errors = 0
    start_time = time.time()
    
    pbar = tqdm(total=total, desc="Sparse BM25", unit="pts")
    
    while True:
        try:
            results, next_offset = client.scroll(
                collection_name=COLLECTION,
                scroll_filter=scroll_filter,
                limit=SCROLL_LIMIT,
                offset=offset,
                with_payload=True,
                with_vectors=False,  # No necesitamos los dense vectors
            )
        except Exception as e:
            print(f"\n❌ Error en scroll: {e}")
            errors += 1
            if errors > 5:
                print("   Demasiados errores, abortando.")
                break
            time.sleep(2)
            continue
        
        if not results:
            break
        
        # Procesar batch
        try:
            batch_updated = process_batch(client, encoder, results, args.dry_run)
            updated += batch_updated
            processed += len(results)
            pbar.update(len(results))
        except Exception as e:
            print(f"\n❌ Error en batch: {e}")
            errors += 1
            if errors > 5:
                print("   Demasiados errores, abortando.")
                break
            time.sleep(2)
        
        # Avanzar offset
        if next_offset is None:
            break
        offset = next_offset
    
    pbar.close()
    elapsed = time.time() - start_time
    
    # Resumen
    print(f"\n[3/3] Resumen:")
    print(f"   Puntos procesados: {processed:,}/{total:,}")
    print(f"   Vectores sparse actualizados: {updated:,}")
    print(f"   Errores: {errors}")
    print(f"   Tiempo: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"   Velocidad: {processed/elapsed:.0f} pts/s" if elapsed > 0 else "")
    
    if args.dry_run:
        print("\n📋 DRY RUN — No se actualizó Qdrant.")
    else:
        print(f"\n✅ Re-ingesta BM25 completada para {entidad_label}.")


if __name__ == "__main__":
    main()
