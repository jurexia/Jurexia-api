#!/usr/bin/env python3
"""
ingest_sentencias.py — Ingesta de Sentencias de Ejemplo a Qdrant

Lee PDFs de sentencias reales de TCC desde carpetas locales y las ingesta
en 4 colecciones Qdrant dedicadas (una por tipo de resolución).

Colecciones:
    sentencias_amparo_directo    (201 PDFs)
    sentencias_amparo_revision   (94 PDFs)
    sentencias_recurso_queja     (89 PDFs)
    sentencias_revision_fiscal   (23 PDFs)

Uso:
    python ingest_sentencias.py                    # Ingesta completa
    python ingest_sentencias.py --dry-run          # Solo procesar, sin subir
    python ingest_sentencias.py --tipo amparo_directo  # Solo un tipo
"""

import asyncio
import os
import sys
import time
import uuid
import argparse
from pathlib import Path

import pymupdf
from openai import AsyncOpenAI
from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    PointStruct,
    SparseVector,
    VectorParams,
    SparseVectorParams,
    SparseIndexParams,
)

from terminator_sentencias import procesar_sentencia, ChunkSentencia, diagnostico


# ══════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════

QDRANT_URL = os.environ.get(
    "QDRANT_URL",
    "https://d6766dbb-cf4c-40a2-a636-78060cc09ccc.us-east4-0.gcp.cloud.qdrant.io",
)
QDRANT_API_KEY = os.environ.get(
    "QDRANT_API_KEY",
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.4hZwbdZT6esMLx7hjHCi79hD5gLpEAVphmuNGYB3A0Y",
)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536
EMBED_BATCH_SIZE = 50
QDRANT_BATCH_SIZE = 50

# Source directory with example sentencias
SOURCE_DIR = Path(r"C:\Proyectos\EJEMPLOS DE SENTENCIAS INGESTAR")

# Mapping: folder name → (tipo_sentencia, collection_name)
SENTENCIA_TYPES = {
    "Amparos Directos": ("amparo_directo", "sentencias_amparo_directo"),
    "Amparo en Revisión": ("amparo_revision", "sentencias_amparo_revision"),
    "Quejas": ("recurso_queja", "sentencias_recurso_queja"),
    "Revisión Fiscal": ("revision_fiscal", "sentencias_revision_fiscal"),
}

# BM25 Sparse Encoder (optional)
HAS_SPARSE = False
sparse_encoder = None

def _init_sparse_encoder():
    global HAS_SPARSE, sparse_encoder
    if sparse_encoder is not None:
        return
    try:
        from fastembed import SparseTextEmbedding
        sparse_encoder = SparseTextEmbedding(model_name="Qdrant/bm25")
        HAS_SPARSE = True
        print("   [OK] BM25 sparse encoder loaded")
    except ImportError:
        sparse_encoder = None
        HAS_SPARSE = False
        print("   [WARN] fastembed not installed — BM25 sparse vectors will be skipped")


# ══════════════════════════════════════════════════════════════════════
# PDF EXTRACTION
# ══════════════════════════════════════════════════════════════════════

def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract text from PDF using PyMuPDF."""
    try:
        doc = pymupdf.open(str(pdf_path))
        pages = []
        for page in doc:
            text = page.get_text("text")
            if text.strip():
                pages.append(text)
        doc.close()
        return "\n".join(pages)
    except Exception as e:
        print(f"   ❌ Error reading PDF {pdf_path.name}: {e}")
        return ""


# ══════════════════════════════════════════════════════════════════════
# EMBEDDING
# ══════════════════════════════════════════════════════════════════════

openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)


async def get_dense_embeddings(texts: list[str]) -> list[list[float]]:
    """Get dense embeddings from OpenAI in batch."""
    truncated = [t[:30000] for t in texts]
    resp = await openai_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=truncated,
    )
    return [d.embedding for d in resp.data]


async def embed_all_chunks(chunks: list[ChunkSentencia]) -> list[list[float]]:
    """Embed all chunks with batching and error recovery."""
    all_embeddings: list[list[float]] = [[] for _ in chunks]
    
    for batch_start in range(0, len(chunks), EMBED_BATCH_SIZE):
        batch_end = min(batch_start + EMBED_BATCH_SIZE, len(chunks))
        batch_texts = [c.texto for c in chunks[batch_start:batch_end]]
        
        try:
            batch_embeddings = await get_dense_embeddings(batch_texts)
            for i, emb in enumerate(batch_embeddings):
                all_embeddings[batch_start + i] = emb
            
            progress = min(batch_end, len(chunks))
            print(f"   📊 Embedded {progress}/{len(chunks)} chunks")
            
        except Exception as e:
            print(f"   ❌ Embedding error at batch {batch_start}: {e}")
            # Retry one by one
            for j in range(batch_start, batch_end):
                try:
                    embs = await get_dense_embeddings([chunks[j].texto])
                    all_embeddings[j] = embs[0]
                except Exception as e2:
                    print(f"   ❌ Skip chunk {j}: {e2}")
                    all_embeddings[j] = [0.0] * EMBEDDING_DIM
        
        await asyncio.sleep(0.15)
    
    return all_embeddings


# ══════════════════════════════════════════════════════════════════════
# QDRANT COLLECTION MANAGEMENT
# ══════════════════════════════════════════════════════════════════════

def ensure_collection(qdrant: QdrantClient, collection_name: str):
    """Create collection if it doesn't exist."""
    try:
        info = qdrant.get_collection(collection_name)
        print(f"   ✅ Collection '{collection_name}' exists ({info.points_count} points)")
        return
    except Exception:
        pass
    
    print(f"   📦 Creating collection '{collection_name}'...")
    qdrant.create_collection(
        collection_name=collection_name,
        vectors_config={
            "dense": VectorParams(
                size=EMBEDDING_DIM,
                distance=Distance.COSINE,
            ),
        },
        sparse_vectors_config={
            "sparse": SparseVectorParams(
                index=SparseIndexParams(on_disk=False)
            )
        },
    )
    print(f"   ✅ Collection '{collection_name}' created (with dense + sparse)")


def delete_existing_data(qdrant: QdrantClient, collection_name: str):
    """Delete all data in the collection for clean re-ingest."""
    try:
        from qdrant_client.http.models import Filter
        count = qdrant.count(collection_name=collection_name)
        if count.count == 0:
            print("   ✅ Collection is empty, nothing to delete")
            return
        
        print(f"   🗑️  Deleting {count.count} existing points...")
        qdrant.delete(
            collection_name=collection_name,
            points_selector=Filter(must=[]),
        )
        print(f"   ✅ Deleted all existing data")
    except Exception as e:
        print(f"   ⚠️  Delete skipped: {e}")


# ══════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════

def generate_point_id(tipo: str, archivo: str, seccion: str, 
                      considerando: str, chunk_index: int) -> str:
    """Generate deterministic UUID for a chunk."""
    raw = f"sentencia::{tipo}::{archivo}::{seccion}::{considerando}::{chunk_index}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, raw))


async def ingest_one_type(
    qdrant: QdrantClient,
    folder_name: str,
    tipo_sentencia: str,
    collection_name: str,
    dry_run: bool = False,
):
    """Ingest all sentencias of one type."""
    folder_path = SOURCE_DIR / folder_name
    
    if not folder_path.exists():
        print(f"   ❌ Folder not found: {folder_path}")
        return 0
    
    # Find all PDFs
    pdf_files = sorted(folder_path.glob("*.pdf"))
    if not pdf_files:
        pdf_files = sorted(folder_path.glob("*.PDF"))
    
    print(f"\n   📂 {folder_name}: {len(pdf_files)} PDFs encontrados")
    
    if not pdf_files:
        return 0
    
    # Phase 1: Extract + Terminator
    all_chunks: list[ChunkSentencia] = []
    files_processed = 0
    files_failed = 0
    
    for pdf_path in pdf_files:
        texto_raw = extract_text_from_pdf(pdf_path)
        
        if not texto_raw.strip() or len(texto_raw) < 100:
            print(f"   ⚠️  Empty/tiny PDF: {pdf_path.name}")
            files_failed += 1
            continue
        
        chunks = procesar_sentencia(texto_raw, pdf_path.name, tipo_sentencia)
        all_chunks.extend(chunks)
        files_processed += 1
        
        if files_processed % 50 == 0:
            print(f"   📄 Processed {files_processed}/{len(pdf_files)} PDFs, "
                  f"{len(all_chunks)} chunks so far")
    
    print(f"\n   📊 TERMINATOR SUMMARY ({folder_name}):")
    print(f"      Files processed: {files_processed}")
    print(f"      Files failed: {files_failed}")
    print(f"      Total chunks: {len(all_chunks)}")
    
    stats = diagnostico(all_chunks)
    if stats.get("total_chunks", 0) > 0:
        print(f"      Chars min/avg/max: {stats['chars_min']}/{stats['chars_avg']}/{stats['chars_max']}")
        print(f"      Total chars: {stats['chars_total']:,}")
        if 'por_seccion' in stats:
            for sec, count in sorted(stats['por_seccion'].items()):
                print(f"      {sec}: {count} chunks")
    
    if not all_chunks:
        print(f"   ❌ No chunks generated for {folder_name}!")
        return 0
    
    if dry_run:
        print(f"\n   🏁 DRY RUN — showing sample chunks:")
        for c in all_chunks[:5]:
            preview = c.texto[:120].replace('\n', ' ')
            print(f"      [{c.seccion}] {c.considerando_num or '-'} ({len(c.texto)} chars)")
            print(f"         {c.jerarquia_txt}")
            print(f"         {preview}...")
        return len(all_chunks)
    
    # Phase 2: Setup collection
    ensure_collection(qdrant, collection_name)
    delete_existing_data(qdrant, collection_name)
    
    # Phase 3: Generate embeddings
    print(f"\n   🔢 Generating embeddings for {len(all_chunks)} chunks...")
    embeddings = await embed_all_chunks(all_chunks)
    
    # Phase 4: Build and upsert points
    print(f"\n   ☁️  Upserting to '{collection_name}'...")
    points = []
    for i, (chunk, embedding) in enumerate(zip(all_chunks, embeddings)):
        point_id = generate_point_id(
            tipo_sentencia, chunk.archivo_origen,
            chunk.seccion, chunk.considerando_num, chunk.chunk_index
        )
        
        payload = {
            "tipo_sentencia": chunk.tipo_sentencia,
            "seccion": chunk.seccion,
            "considerando_num": chunk.considerando_num,
            "archivo_origen": chunk.archivo_origen,
            "texto": chunk.texto,
            "jerarquia_txt": chunk.jerarquia_txt,
            "chunk_index": chunk.chunk_index,
            "collection": collection_name,
            "fuente": f"Sentencia Ejemplo: {chunk.archivo_origen}",
        }
        
        # Generate BM25 sparse vector if available
        sparse_dict = {}
        if HAS_SPARSE and sparse_encoder:
            try:
                embeddings_sparse = list(sparse_encoder.passage_embed([chunk.texto]))
                if embeddings_sparse and len(embeddings_sparse[0].indices) > 0:
                    sp = embeddings_sparse[0]
                    sparse_dict["sparse"] = SparseVector(
                        indices=sp.indices.tolist(),
                        values=sp.values.tolist(),
                    )
            except Exception:
                pass
        
        vector = {"dense": embedding}
        if sparse_dict:
            vector.update(sparse_dict)
        
        point = PointStruct(
            id=point_id,
            vector=vector,
            payload=payload,
        )
        points.append(point)
    
    # Batch upsert
    for batch_start in range(0, len(points), QDRANT_BATCH_SIZE):
        batch = points[batch_start:batch_start + QDRANT_BATCH_SIZE]
        try:
            qdrant.upsert(
                collection_name=collection_name,
                points=batch,
            )
            progress = min(batch_start + QDRANT_BATCH_SIZE, len(points))
            print(f"   ✅ Upserted {progress}/{len(points)} points")
        except Exception as e:
            print(f"   ❌ Upsert error at batch {batch_start}: {e}")
            for p in batch:
                try:
                    qdrant.upsert(collection_name=collection_name, points=[p])
                except Exception as e2:
                    print(f"      ❌ Point {p.id}: {e2}")
    
    # Verify
    final_count = qdrant.count(collection_name=collection_name)
    print(f"   ✅ Collection '{collection_name}': {final_count.count} points")
    
    return final_count.count


async def run_full_ingestion(dry_run: bool = False, tipos_filter: list[str] = None):
    """Run the full ingestion pipeline for all (or filtered) sentencia types."""
    start_time = time.time()
    
    print("═══════════════════════════════════════════════════════════════")
    print("  🏛️  INGESTA DE SENTENCIAS DE EJEMPLO")
    print("═══════════════════════════════════════════════════════════════")
    print(f"  Directorio fuente: {SOURCE_DIR}")
    print(f"  Dry run: {dry_run}")
    print(f"  Tipos: {tipos_filter or 'TODOS'}")
    print("═══════════════════════════════════════════════════════════════")
    
    if not SOURCE_DIR.exists():
        print(f"\n   ❌ Source directory not found: {SOURCE_DIR}")
        return
    
    # Connect to Qdrant (skip for dry-run if no connection needed)
    qdrant = None
    if not dry_run:
        qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        print(f"\n   ✅ Connected to Qdrant: {QDRANT_URL[:50]}...")
    
    total_points = 0
    
    for folder_name, (tipo, collection) in SENTENCIA_TYPES.items():
        # Skip if filtering by type
        if tipos_filter and tipo not in tipos_filter:
            continue
        
        print(f"\n{'═' * 63}")
        print(f"  📁 {folder_name} → {collection}")
        print(f"{'═' * 63}")
        
        count = await ingest_one_type(
            qdrant=qdrant,
            folder_name=folder_name,
            tipo_sentencia=tipo,
            collection_name=collection,
            dry_run=dry_run,
        )
        total_points += count
    
    elapsed = time.time() - start_time
    
    print(f"\n{'═' * 63}")
    print(f"  ✅ INGESTA COMPLETADA")
    print(f"  📊 Total points: {total_points:,}")
    print(f"  ⏱️  Time: {elapsed:.1f}s ({elapsed/60:.1f}m)")
    if not dry_run:
        print(f"  💡 BM25 sparse: {'YES' if HAS_SPARSE else 'NO (run reingest-sparse later)'}")
    print(f"{'═' * 63}")


# ══════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════

def main():
    # Fix Windows encoding
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    
    parser = argparse.ArgumentParser(
        description="Ingest example sentencias into Qdrant collections"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Process PDFs and show chunks without embedding/upserting"
    )
    parser.add_argument(
        "--tipo",
        choices=["amparo_directo", "amparo_revision", "recurso_queja", "revision_fiscal"],
        help="Only ingest one specific type (default: all)"
    )
    
    args = parser.parse_args()
    
    tipos_filter = [args.tipo] if args.tipo else None
    
    asyncio.run(run_full_ingestion(
        dry_run=args.dry_run,
        tipos_filter=tipos_filter,
    ))


if __name__ == "__main__":
    main()
