#!/usr/bin/env python3
"""
ingest_qro_reglamentos.py — Ingesta Suplementaria de Reglamentos Municipales de Querétaro

Lee PDFs locales de reglamentos municipales y los agrega a la colección
leyes_queretaro existente (NO borra datos previos).

Uso:
    python ingest_qro_reglamentos.py --dry-run
    python ingest_qro_reglamentos.py

Fuente: C:\Proyectos\LEYES_MEXICO_MASTER\03_LEYES_ESTATALES\QUERETARO\REGLAMENTOS MUNICIPIO QRO
"""

import asyncio
import os
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
import argparse

import pymupdf
from openai import AsyncOpenAI
from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    PointStruct,
)

from terminator_leyes import procesar_ley, ArticuloLimpio, diagnostico


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

COLLECTION = "leyes_queretaro"
ENTIDAD = "QUERETARO"

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536
EMBED_BATCH_SIZE = 50
QDRANT_BATCH_SIZE = 50

# Source directory with local PDFs
PDF_DIR = Path(r"C:\Proyectos\LEYES_MEXICO_MASTER\03_LEYES_ESTATALES\QUERETARO\REGLAMENTOS MUNICIPIO QRO")


# ══════════════════════════════════════════════════════════════════════
# REGLAMENTO DEFINITIONS
# ══════════════════════════════════════════════════════════════════════

@dataclass
class ReglamentoDef:
    nombre: str
    filename: str
    tipo_codigo: str = ""


REGLAMENTOS = [
    ReglamentoDef(
        nombre="Código de Conducta de los Servidores Públicos Municipales de Querétaro 2021",
        filename="CODIGO DE CONDUCTA-2021.pdf",
        tipo_codigo="codigo_conducta",
    ),
    ReglamentoDef(
        nombre="Código Municipal de Querétaro",
        filename="CÓDIGO MUNPAL DE QRO.pdf",
        tipo_codigo="codigo_municipal",
    ),
    ReglamentoDef(
        nombre="Lineamientos en Materia de Obra Pública del Municipio de Querétaro",
        filename="LINEAMIENTOS EN MATERIA DE OBRA PÚBLICA DEL MUNICIPIO DE QUERÉTARO.pdf",
        tipo_codigo="lineamiento",
    ),
    ReglamentoDef(
        nombre="Protocolo para la Implementación de los Puntos de Control de Alcoholimetría para el Municipio de Querétaro",
        filename="PROTOCOLO PARA LA IMPLEMENTACIÓN DE LOS PUNTOS DE CONTROL DE ALCOHOLIMETRÍA PARA EL MUNICIPIO DE QUERÉTARO.pdf",
        tipo_codigo="protocolo",
    ),
    ReglamentoDef(
        nombre="Reglamento para la Movilidad y el Tránsito del Municipio de Querétaro",
        filename="REGLAMENTO PARA LA MOVILIDAD Y EL TRÁNSITO.pdf",
        tipo_codigo="reglamento_transito",
    ),
    ReglamentoDef(
        nombre="Reglamento de Justicia Cívica del Municipio de Querétaro (Marzo 2025)",
        filename="Reglamento de Justicia Civica marzo2025.pdf",
        tipo_codigo="reglamento_justicia_civica",
    ),
]


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


async def embed_all_chunks(articulos: list[ArticuloLimpio]) -> list[list[float]]:
    """Embed all articles with batching and error recovery."""
    all_embeddings: list[list[float]] = [[] for _ in articulos]
    
    for batch_start in range(0, len(articulos), EMBED_BATCH_SIZE):
        batch_end = min(batch_start + EMBED_BATCH_SIZE, len(articulos))
        batch_texts = [a.texto for a in articulos[batch_start:batch_end]]
        
        try:
            batch_embeddings = await get_dense_embeddings(batch_texts)
            for i, emb in enumerate(batch_embeddings):
                all_embeddings[batch_start + i] = emb
            
            progress = min(batch_end, len(articulos))
            print(f"   📊 Embedded {progress}/{len(articulos)} chunks")
            
        except Exception as e:
            print(f"   ❌ Embedding error at batch {batch_start}: {e}")
            for j in range(batch_start, batch_end):
                try:
                    embs = await get_dense_embeddings([articulos[j].texto])
                    all_embeddings[j] = embs[0]
                except Exception as e2:
                    print(f"   ❌ Skip chunk {j}: {e2}")
                    all_embeddings[j] = [0.0] * EMBEDDING_DIM
        
        await asyncio.sleep(0.15)
    
    return all_embeddings


# ══════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════

def generate_point_id(entidad: str, law_name: str, ref: str, chunk_index: int) -> str:
    """Generate deterministic UUID for a chunk."""
    raw = f"{entidad}::REGLAMENTO_MUNICIPAL::{law_name}::{ref}::{chunk_index}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, raw))


# ══════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════

async def run_ingestion(dry_run: bool = False):
    """Supplementary ingestion — adds reglamentos to existing leyes_queretaro."""
    
    start_time = time.time()
    
    print("═══════════════════════════════════════════════════════════════")
    print("  INGESTA SUPLEMENTARIA: REGLAMENTOS MUNICIPALES QUERÉTARO")
    print(f"  Colección destino: {COLLECTION}")
    print(f"  Fuente: {PDF_DIR}")
    print(f"  Reglamentos: {len(REGLAMENTOS)}")
    print("═══════════════════════════════════════════════════════════════")
    
    # Connect to Qdrant and check existing data
    qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    
    print("\n═══════════════════════════════════════════════════════════════")
    print("  PHASE 1: CHECKING EXISTING DATA IN QDRANT")
    print("═══════════════════════════════════════════════════════════════")
    
    existing_count = qdrant.count(collection_name=COLLECTION)
    print(f"   📊 Existing {COLLECTION} chunks: {existing_count.count}")
    
    # Phase 2: Extract + Terminator
    print("\n═══════════════════════════════════════════════════════════════")
    print("  PHASE 2: EXTRACT + TERMINATOR (artículos limpios)")
    print("═══════════════════════════════════════════════════════════════")
    
    all_articulos: list[ArticuloLimpio] = []
    all_reglamentos: list[ReglamentoDef] = []
    processed = 0
    failed = 0
    
    for reg in REGLAMENTOS:
        filepath = PDF_DIR / reg.filename
        
        if not filepath.exists():
            print(f"   ❌ NOT FOUND: {reg.filename}")
            failed += 1
            continue
        
        print(f"\n   📄 Processing: {reg.nombre}")
        print(f"      File: {reg.filename} ({filepath.stat().st_size / 1024:.0f} KB)")
        
        texto_raw = extract_text_from_pdf(filepath)
        
        if not texto_raw.strip():
            print(f"   ⚠️  Empty PDF: {reg.nombre}")
            failed += 1
            continue
        
        # 🤖 TERMINATOR: Produce artículos limpios
        articulos = procesar_ley(texto_raw, reg.nombre)
        
        stats = diagnostico(articulos)
        print(f"      → {stats['articulos_unicos']} artículos, "
              f"{stats['total']} chunks, "
              f"avg {stats['chars_avg']:.0f} chars")
        
        all_articulos.extend(articulos)
        all_reglamentos.extend([reg] * len(articulos))
        processed += 1
    
    print(f"\n   📊 TERMINATOR SUMMARY:")
    print(f"      Reglamentos processed: {processed}/{len(REGLAMENTOS)}")
    print(f"      Reglamentos failed: {failed}")
    print(f"      Total article chunks: {len(all_articulos)}")
    
    if not all_articulos:
        print("   ❌ No articles generated! Aborting.")
        return
    
    if dry_run:
        print("\n   🏁 DRY RUN — Skipping embedding and upsert")
        print("\n   📝 SAMPLE ARTICLES:")
        for a in all_articulos[:8]:
            preview = a.texto[:120].replace('\n', ' ')
            print(f"      [{a.ref}] ({len(a.texto)} chars)")
            print(f"         {a.jerarquia_txt}")
            print(f"         {preview}...")
        return
    
    # Phase 3: Embeddings
    print("\n═══════════════════════════════════════════════════════════════")
    print("  PHASE 3: GENERATING DENSE EMBEDDINGS")
    print("═══════════════════════════════════════════════════════════════")
    
    embeddings = await embed_all_chunks(all_articulos)
    
    # Phase 4: Upsert (SUPPLEMENTARY — no delete)
    print("\n═══════════════════════════════════════════════════════════════")
    print(f"  PHASE 4: UPSERTING TO '{COLLECTION}' (supplementary)")
    print("═══════════════════════════════════════════════════════════════")
    
    points = []
    for i, (art, reg, embedding) in enumerate(zip(all_articulos, all_reglamentos, embeddings)):
        point_id = generate_point_id(ENTIDAD, art.origen, art.ref, art.chunk_index)
        
        payload = {
            "entidad": ENTIDAD,
            "origen": art.origen,
            "ref": art.ref,
            "texto": art.texto,
            "jerarquia_txt": art.jerarquia_txt,
            "tipo_codigo": reg.tipo_codigo,
            "jurisdiccion": "municipal",
            "categoria": "reglamento_municipal",
            "chunk_index": art.chunk_index,
            "titulo": art.titulo,
            "capitulo": art.capitulo,
            "seccion": art.seccion,
        }
        
        point = PointStruct(
            id=point_id,
            vector={"dense": embedding},
            payload=payload,
        )
        points.append(point)
    
    # Batch upsert
    for batch_start in range(0, len(points), QDRANT_BATCH_SIZE):
        batch = points[batch_start:batch_start + QDRANT_BATCH_SIZE]
        try:
            qdrant.upsert(
                collection_name=COLLECTION,
                points=batch,
            )
            progress = min(batch_start + QDRANT_BATCH_SIZE, len(points))
            print(f"   ✅ Upserted {progress}/{len(points)} points")
        except Exception as e:
            print(f"   ❌ Upsert error at batch {batch_start}: {e}")
            for p in batch:
                try:
                    qdrant.upsert(collection_name=COLLECTION, points=[p])
                except Exception as e2:
                    print(f"      ❌ Point {p.id}: {e2}")
    
    # Verification
    print("\n═══════════════════════════════════════════════════════════════")
    print("  VERIFICATION")
    print("═══════════════════════════════════════════════════════════════")
    
    final_count = qdrant.count(collection_name=COLLECTION)
    elapsed = time.time() - start_time
    new_points = final_count.count - existing_count.count
    
    print(f"\n   ✅ SUPPLEMENTARY INGESTION COMPLETE")
    print(f"      Collection: {COLLECTION}")
    print(f"      Previous points: {existing_count.count}")
    print(f"      New points: {new_points}")
    print(f"      Total points: {final_count.count}")
    print(f"      Time: {elapsed:.1f}s")
    print(f"      BM25 sparse: NO (run reingest-sparse later)")


# ══════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════

def main():
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    
    parser = argparse.ArgumentParser(
        description="Ingest Querétaro municipal reglamentos (supplementary)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Process PDFs and show chunks without embedding/upserting"
    )
    
    args = parser.parse_args()
    asyncio.run(run_ingestion(dry_run=args.dry_run))


if __name__ == "__main__":
    main()
