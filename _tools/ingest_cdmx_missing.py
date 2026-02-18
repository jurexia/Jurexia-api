#!/usr/bin/env python3
"""
ingest_cdmx_missing.py — Supplementary Ingestion for Missing CDMX Laws
═══════════════════════════════════════════════════════════════════════

Uses cdmx_urls_current.json (scraped from portal) to find and ingest
only the laws that are NOT yet in Qdrant. Does NOT delete existing data.

Usage:
    python ingest_cdmx_missing.py               # Full pipeline (download + ingest missing)
    python ingest_cdmx_missing.py --dry-run      # Show what would be ingested without doing it
    python ingest_cdmx_missing.py --download-only # Only download PDFs, skip ingestion
"""

import asyncio
import hashlib
import json
import os
import re
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import httpx
import pymupdf  # PyMuPDF
from openai import AsyncOpenAI
from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    FieldCondition,
    Filter,
    MatchValue,
    NamedVector,
    PointStruct,
    ScrollRequest,
)

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

COLLECTION = "leyes_cdmx"
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536
ENTIDAD = "CDMX"  # Must match normalize_estado() in main.py

# Article-aware chunking parameters
MAX_CHUNK_TOKENS = 1200
OVERLAP_CHARS = 400
MIN_CHUNK_LEN = 50

# PDF download directory
PDF_DIR = Path(__file__).parent / "pdfs_cdmx"

# Rate limiting
EMBED_BATCH_SIZE = 50
QDRANT_BATCH_SIZE = 50

# Scraped URLs file
URLS_FILE = Path(__file__).parent / "cdmx_urls_current.json"

# ══════════════════════════════════════════════════════════════════════
# LAW DEFINITION
# ══════════════════════════════════════════════════════════════════════

@dataclass
class LawDef:
    nombre: str
    url: str
    categoria: str
    tipo_codigo: str = ""
    filename: str = ""


def _infer_tipo_codigo(nombre: str, categoria: str) -> str:
    """Infer tipo_codigo from law name."""
    n = nombre.lower()
    if "penal" in n: return "PENAL"
    if "civil" in n and "procedimiento" not in n: return "CIVIL"
    if "procedimientos civiles" in n: return "PROCESAL_CIVIL"
    if "fiscal" in n: return "FISCAL"
    if "urbano" in n: return "URBANO"
    if "ambiental" in n: return "AMBIENTAL"
    if "familiar" in n: return "FAMILIAR"
    if "electoral" in n: return "ELECTORAL"
    if "constitución" in n or "constituc" in n: return "CONSTITUCION"
    if "tránsito" in n or "transito" in n: return "TRANSITO"
    if "notarial" in n or "notariado" in n: return "NOTARIAL"
    if "salud" in n: return "SALUD"
    if "educación" in n or "educacion" in n: return "EDUCACION"
    if "transparencia" in n: return "TRANSPARENCIA"
    if "laboral" in n or "trabajadores" in n: return "LABORAL"
    if "hacienda" in n: return "HACIENDA"
    if "seguridad" in n: return "SEGURIDAD"
    if "derechos humanos" in n: return "DERECHOS_HUMANOS"
    if "movilidad" in n: return "MOVILIDAD"
    if "construccion" in n or "construcciones" in n: return "CONSTRUCCION"
    return "GENERAL"


def _infer_jurisdiccion(nombre: str) -> str:
    """Infer jurisdiccion from law name."""
    n = nombre.lower()
    if "penal" in n: return "penal"
    if "civil" in n: return "civil"
    if "familiar" in n: return "familiar"
    if "laboral" in n or "trabajadores" in n: return "laboral"
    if "mercantil" in n or "comerci" in n: return "mercantil"
    if "fiscal" in n or "hacienda" in n or "tributari" in n: return "fiscal"
    if "administrativ" in n: return "administrativo"
    if "electoral" in n: return "electoral"
    if "ambiental" in n: return "ambiental"
    if "constitución" in n or "constituc" in n: return "constitucional"
    return "general"


def _name_from_filename(filename: str) -> str:
    """Convert a PDF filename to a readable law name."""
    name = filename.replace(".pdf", "")
    # Remove version numbers at the end (e.g., _14.4, _7.3, _2.5)
    name = re.sub(r'_\d+(?:\.\d+)*$', '', name)
    # Replace underscores with spaces
    name = name.replace("_", " ").strip()
    # Capitalize properly
    name = name.title()
    # Fix common abbreviations
    name = name.replace("Cdmx", "CDMX").replace("Df", "DF").replace("Nna", "NNA")
    name = name.replace("Tsjdf", "TSJDF").replace("Tsj", "TSJ")
    name = name.replace("De La", "de la").replace("De Los", "de los")
    name = name.replace("De Las", "de las").replace("Del ", "del ")
    name = name.replace("Para El", "para el").replace("Para La", "para la")
    name = name.replace("En El", "en el").replace("En La", "en la")
    name = name.replace("Y De", "y de").replace("Y La", "y la")
    name = name.replace(" La ", " la ").replace(" El ", " el ")
    # Ensure first letter is capitalized
    if name:
        name = name[0].upper() + name[1:]
    return name


def load_laws_from_json() -> list[LawDef]:
    """Load law definitions from scraped JSON."""
    if not URLS_FILE.exists():
        print(f"❌ {URLS_FILE} not found. Run scrape_cdmx_urls.py first.")
        sys.exit(1)

    with open(URLS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    laws = []
    for entry in data:
        filename = entry.get("filename", entry["url"].split("/")[-1])
        nombre = entry.get("nombre", "")

        # If nombre is just the filename or empty, generate a better name
        if not nombre or nombre == filename.replace(".pdf", "").replace("_", " "):
            nombre = _name_from_filename(filename)

        law = LawDef(
            nombre=nombre,
            url=entry["url"],
            categoria=entry.get("categoria", "ley"),
            filename=filename,
        )
        law.tipo_codigo = _infer_tipo_codigo(law.nombre, law.categoria)
        laws.append(law)

    return laws


# ══════════════════════════════════════════════════════════════════════
# ARTICLE-AWARE CHUNKING (same as ingest_cdmx.py)
# ══════════════════════════════════════════════════════════════════════

ARTICLE_PATTERN = re.compile(
    r'(?:^|\n)'
    r'(Art[ií]culo\s+\d+[\w]*'
    r'(?:\s+(?:BIS|TER|QUÁTER|QUINQUIES))?'
    r'[\.\-\s])',
    re.IGNORECASE | re.MULTILINE
)

ARTICLE_REF_PATTERN = re.compile(
    r'Art[ií]culo\s+(\d+[\w]*(?:\s+(?:BIS|TER|QUÁTER|QUINQUIES))?)',
    re.IGNORECASE
)

SECTION_PATTERN = re.compile(
    r'(?:^|\n)\s*((?:TÍTULO|CAPITULO|CAPÍTULO|SECCIÓN|SECCION|LIBRO)\s+[IVXLCDM\d]+)',
    re.IGNORECASE | re.MULTILINE
)


@dataclass
class Chunk:
    """A single chunk ready for embedding."""
    text: str
    origin: str
    ref: str
    jerarquia_txt: str
    tipo_codigo: str
    jurisdiccion: str
    categoria: str
    chunk_index: int = 0


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


def article_aware_chunk(text: str, law: LawDef) -> list[Chunk]:
    """Split legal text into article-aware chunks."""
    if not text.strip():
        return []

    text = re.sub(r'\r\n', '\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)

    chunks: list[Chunk] = []
    jurisdiccion = _infer_jurisdiccion(law.nombre)
    current_section = ""

    splits = ARTICLE_PATTERN.split(text)

    if len(splits) <= 1:
        return _fixed_size_chunk(text, law, jurisdiccion)

    # Preamble
    preamble = splits[0].strip()
    if preamble and len(preamble) > MIN_CHUNK_LEN:
        sec_match = SECTION_PATTERN.search(preamble)
        if sec_match:
            current_section = sec_match.group(1).strip()
        for i, sub in enumerate(_split_long_text(preamble)):
            chunks.append(Chunk(
                text=sub, origin=law.nombre, ref="Preámbulo",
                jerarquia_txt=f"{law.nombre} > Preámbulo",
                tipo_codigo=law.tipo_codigo, jurisdiccion=jurisdiccion,
                categoria=law.categoria, chunk_index=i,
            ))

    # Articles
    i = 1
    while i < len(splits):
        art_header = splits[i].strip() if i < len(splits) else ""
        art_body = splits[i + 1].strip() if (i + 1) < len(splits) else ""
        full_article = f"{art_header} {art_body}".strip()

        if len(full_article) < MIN_CHUNK_LEN:
            i += 2
            continue

        ref_match = ARTICLE_REF_PATTERN.search(art_header)
        art_ref = f"Art. {ref_match.group(1)}" if ref_match else art_header[:30]

        sec_match = SECTION_PATTERN.search(art_body)
        if sec_match:
            current_section = sec_match.group(1).strip()

        jerarquia = f"{law.nombre} > {current_section} > {art_ref}" if current_section else f"{law.nombre} > {art_ref}"

        sub_texts = _split_long_text(full_article)
        for j, sub in enumerate(sub_texts):
            chunks.append(Chunk(
                text=sub, origin=law.nombre, ref=art_ref,
                jerarquia_txt=jerarquia, tipo_codigo=law.tipo_codigo,
                jurisdiccion=jurisdiccion, categoria=law.categoria,
                chunk_index=j,
            ))
        i += 2

    # Transitorios
    if splits and len(splits) > 2:
        last_text = splits[-1]
        trans_match = re.search(r'(?:^|\n)(TRANSITORIOS?)\s*\n', last_text, re.IGNORECASE)
        if trans_match:
            trans_text = last_text[trans_match.start():].strip()
            if len(trans_text) > MIN_CHUNK_LEN:
                for j, sub in enumerate(_split_long_text(trans_text)):
                    chunks.append(Chunk(
                        text=sub, origin=law.nombre, ref="Transitorios",
                        jerarquia_txt=f"{law.nombre} > Transitorios",
                        tipo_codigo=law.tipo_codigo, jurisdiccion=jurisdiccion,
                        categoria=law.categoria, chunk_index=j,
                    ))

    return chunks


def _split_long_text(text: str, max_chars: int = 4800) -> list[str]:
    """Split text that exceeds max_chars into overlapping chunks."""
    if len(text) <= max_chars:
        return [text]
    parts = []
    start = 0
    while start < len(text):
        end = start + max_chars
        if end >= len(text):
            parts.append(text[start:])
            break
        split_point = text.rfind('\n\n', start + max_chars // 2, end)
        if split_point == -1:
            split_point = text.rfind('. ', start + max_chars // 2, end)
        if split_point == -1:
            split_point = text.rfind(' ', start + max_chars // 2, end)
        if split_point == -1:
            split_point = end
        else:
            split_point += 1
        parts.append(text[start:split_point])
        start = split_point - OVERLAP_CHARS
    return parts


def _fixed_size_chunk(text: str, law: LawDef, jurisdiccion: str) -> list[Chunk]:
    """Fallback: chunk text into fixed-size pieces."""
    chunks = []
    parts = _split_long_text(text, max_chars=3200)
    for i, part in enumerate(parts):
        if len(part.strip()) < MIN_CHUNK_LEN:
            continue
        chunks.append(Chunk(
            text=part.strip(), origin=law.nombre, ref=f"Sección {i + 1}",
            jerarquia_txt=f"{law.nombre} > Sección {i + 1}",
            tipo_codigo=law.tipo_codigo, jurisdiccion=jurisdiccion,
            categoria=law.categoria, chunk_index=0,
        ))
    return chunks


# ══════════════════════════════════════════════════════════════════════
# EMBEDDING + QDRANT
# ══════════════════════════════════════════════════════════════════════

openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)


async def get_dense_embeddings(texts: list[str]) -> list[list[float]]:
    """Get dense embeddings from OpenAI in batch."""
    truncated = [t[:30000] for t in texts]
    resp = await openai_client.embeddings.create(
        model=EMBEDDING_MODEL, input=truncated,
    )
    return [d.embedding for d in resp.data]


async def embed_all_chunks(chunks: list[Chunk]) -> list[list[float]]:
    """Embed all chunks with rate limiting and batching."""
    all_embeddings: list[list[float]] = [[] for _ in chunks]

    for batch_start in range(0, len(chunks), EMBED_BATCH_SIZE):
        batch_end = min(batch_start + EMBED_BATCH_SIZE, len(chunks))
        batch_texts = [c.text for c in chunks[batch_start:batch_end]]

        try:
            batch_embeddings = await get_dense_embeddings(batch_texts)
            for i, emb in enumerate(batch_embeddings):
                all_embeddings[batch_start + i] = emb

            progress = min(batch_end, len(chunks))
            print(f"   📊 Embedded {progress}/{len(chunks)} chunks")

        except Exception as e:
            print(f"   ❌ Embedding error at batch {batch_start}: {e}")
            for j in range(batch_start, batch_end):
                try:
                    embs = await get_dense_embeddings([chunks[j].text])
                    all_embeddings[j] = embs[0]
                except Exception as e2:
                    print(f"   ❌ Skip chunk {j}: {e2}")
                    all_embeddings[j] = [0.0] * EMBEDDING_DIM

        await asyncio.sleep(0.15)

    return all_embeddings


def generate_point_id(law_name: str, ref: str, chunk_index: int) -> str:
    """Generate a deterministic UUID for a chunk."""
    raw = f"{ENTIDAD}::{law_name}::{ref}::{chunk_index}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, raw))


# ══════════════════════════════════════════════════════════════════════
# FIND WHAT'S MISSING
# ══════════════════════════════════════════════════════════════════════

def get_existing_origins(qdrant: QdrantClient) -> set[str]:
    """Get all unique 'origen' values for CDMX in Qdrant."""
    origins = set()
    offset = None

    while True:
        results = qdrant.scroll(
            collection_name=COLLECTION,
            scroll_filter=Filter(
                must=[FieldCondition(key="entidad", match=MatchValue(value=ENTIDAD))]
            ),
            limit=100,
            offset=offset,
            with_payload=["origen"],
            with_vectors=False,
        )

        points, next_offset = results
        for p in points:
            if p.payload and "origen" in p.payload:
                origins.add(p.payload["origen"])

        if next_offset is None:
            break
        offset = next_offset

    return origins


# ══════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════

async def download_missing_pdfs(laws: list[LawDef]) -> dict[str, Path]:
    """Download only PDFs that don't exist locally."""
    PDF_DIR.mkdir(exist_ok=True)

    downloaded = {}
    failed = []
    skipped = 0

    async with httpx.AsyncClient(timeout=60, follow_redirects=True, verify=False) as client:
        for i, law in enumerate(laws):
            filename = law.filename or law.url.split("/")[-1]
            filepath = PDF_DIR / filename

            # Skip if already downloaded
            if filepath.exists() and filepath.stat().st_size > 100:
                downloaded[law.nombre] = filepath
                skipped += 1
                continue

            try:
                resp = await client.get(law.url)
                if resp.status_code == 200 and len(resp.content) > 100:
                    filepath.write_bytes(resp.content)
                    downloaded[law.nombre] = filepath
                    if (i + 1) % 20 == 0:
                        print(f"   📥 Progress: {i + 1}/{len(laws)} PDFs")
                else:
                    failed.append((law.nombre, f"HTTP {resp.status_code}", law.url))
            except Exception as e:
                failed.append((law.nombre, str(e), law.url))

            await asyncio.sleep(0.1)

    print(f"\n   ✅ Downloaded: {len(downloaded)} ({skipped} cached)")
    if failed:
        print(f"   ❌ Failed: {len(failed)}")
        for name, err, url in failed[:10]:
            print(f"      • {name}: {err}")
            print(f"        URL: {url}")

    return downloaded


async def run_ingestion():
    """Main ingestion pipeline — supplementary mode (no delete)."""
    start_time = time.time()

    # Load all laws from scraped JSON
    all_laws = load_laws_from_json()
    print(f"\n═══════════════════════════════════════════════════════════════")
    print(f"  CDMX SUPPLEMENTARY INGESTION")
    print(f"  Total laws from portal: {len(all_laws)}")
    print(f"═══════════════════════════════════════════════════════════════\n")

    # Connect to Qdrant
    qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

    # Phase 1: Check what's already in Qdrant
    print("═══════════════════════════════════════════════════════════════")
    print("  PHASE 1: CHECKING EXISTING DATA IN QDRANT")
    print("═══════════════════════════════════════════════════════════════")

    existing_count = qdrant.count(
        collection_name=COLLECTION,
        count_filter=Filter(
            must=[FieldCondition(key="entidad", match=MatchValue(value=ENTIDAD))]
        ),
    )
    print(f"   📊 Existing CDMX chunks: {existing_count.count}")

    existing_origins = get_existing_origins(qdrant)
    print(f"   📊 Existing CDMX law origins: {len(existing_origins)}")
    if existing_origins:
        for o in sorted(existing_origins)[:5]:
            print(f"      • {o}")
        if len(existing_origins) > 5:
            print(f"      ... and {len(existing_origins) - 5} more")

    # Determine which laws are missing
    # Match by checking if the filename (without version) is already present
    missing_laws = []
    already_have = []

    for law in all_laws:
        # Check if this law's name (or similar) already exists in Qdrant origins
        found = False
        for origin in existing_origins:
            # Fuzzy match: check if key words match
            law_words = set(re.findall(r'\w{4,}', law.nombre.lower()))
            origin_words = set(re.findall(r'\w{4,}', origin.lower()))
            common = law_words & origin_words
            if len(common) >= max(2, min(len(law_words), len(origin_words)) * 0.5):
                found = True
                break
        if found:
            already_have.append(law)
        else:
            missing_laws.append(law)

    print(f"\n   📊 Laws already in Qdrant: {len(already_have)}")
    print(f"   📊 Laws to ingest: {len(missing_laws)}")

    if "--dry-run" in sys.argv:
        print(f"\n   🔍 DRY RUN — Would ingest these {len(missing_laws)} laws:")
        for law in missing_laws:
            print(f"      [{law.categoria}] {law.nombre}")
            print(f"        URL: {law.url}")
        return

    if not missing_laws:
        print("\n   ✅ All laws already ingested! Nothing to do.")
        return

    # Phase 2: Download missing PDFs
    print(f"\n═══════════════════════════════════════════════════════════════")
    print(f"  PHASE 2: DOWNLOADING {len(missing_laws)} MISSING PDFs")
    print(f"═══════════════════════════════════════════════════════════════")

    downloaded = await download_missing_pdfs(missing_laws)

    if "--download-only" in sys.argv:
        print(f"\n   ✅ Download complete. Run without --download-only to ingest.")
        return

    # Phase 3: Extract + Chunk
    print(f"\n═══════════════════════════════════════════════════════════════")
    print(f"  PHASE 3: EXTRACTING TEXT + ARTICLE-AWARE CHUNKING")
    print(f"═══════════════════════════════════════════════════════════════")

    all_chunks: list[Chunk] = []
    laws_processed = 0
    laws_failed = 0

    for law in missing_laws:
        if law.nombre not in downloaded:
            laws_failed += 1
            continue

        filepath = downloaded[law.nombre]
        text = extract_text_from_pdf(filepath)

        if not text.strip():
            print(f"   ⚠️  Empty PDF: {law.nombre}")
            laws_failed += 1
            continue

        chunks = article_aware_chunk(text, law)
        all_chunks.extend(chunks)
        laws_processed += 1

        if laws_processed % 20 == 0:
            print(f"   📄 Processed {laws_processed} laws, {len(all_chunks)} chunks so far")

    print(f"\n   📊 CHUNKING SUMMARY:")
    print(f"      Laws processed: {laws_processed}")
    print(f"      Laws failed/skipped: {laws_failed}")
    print(f"      Total new chunks: {len(all_chunks)}")

    # Stats by category
    cats = {}
    for c in all_chunks:
        cats[c.categoria] = cats.get(c.categoria, 0) + 1
    for cat, count in sorted(cats.items()):
        print(f"      {cat}: {count} chunks")

    if not all_chunks:
        print("   ❌ No new chunks generated! Aborting.")
        return

    # Phase 4: Generate embeddings
    print(f"\n═══════════════════════════════════════════════════════════════")
    print(f"  PHASE 4: GENERATING DENSE EMBEDDINGS")
    print(f"═══════════════════════════════════════════════════════════════")

    embeddings = await embed_all_chunks(all_chunks)

    # Phase 5: Upsert to Qdrant (APPEND — no delete!)
    print(f"\n═══════════════════════════════════════════════════════════════")
    print(f"  PHASE 5: UPSERTING TO QDRANT (SUPPLEMENTARY — NO DELETE)")
    print(f"═══════════════════════════════════════════════════════════════")

    points = []
    for i, (chunk, embedding) in enumerate(zip(all_chunks, embeddings)):
        point_id = generate_point_id(chunk.origin, chunk.ref, chunk.chunk_index)

        payload = {
            "entidad": ENTIDAD,
            "origen": chunk.origin,
            "ref": chunk.ref,
            "texto": chunk.text,
            "jerarquia_txt": chunk.jerarquia_txt,
            "tipo_codigo": chunk.tipo_codigo,
            "jurisdiccion": chunk.jurisdiccion,
            "categoria": chunk.categoria,
            "chunk_index": chunk.chunk_index,
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
            qdrant.upsert(collection_name=COLLECTION, points=batch)
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
    print(f"\n═══════════════════════════════════════════════════════════════")
    print(f"  VERIFICATION")
    print(f"═══════════════════════════════════════════════════════════════")

    final_count = qdrant.count(
        collection_name=COLLECTION,
        count_filter=Filter(
            must=[FieldCondition(key="entidad", match=MatchValue(value=ENTIDAD))]
        ),
    )

    elapsed = time.time() - start_time

    print(f"\n   ✅ SUPPLEMENTARY INGESTION COMPLETE")
    print(f"      Previous CDMX chunks: {existing_count.count}")
    print(f"      New chunks added: {len(points)}")
    print(f"      Total CDMX chunks now: {final_count.count}")
    print(f"      Time elapsed: {elapsed:.1f}s")
    print(f"\n   ⚠️  NEXT STEP: Trigger BM25 sparse vector generation via:")
    print(f"      POST https://api.iurexia.com/admin/reingest-sparse")
    print(f'      Body: {{"admin_key": "...", "entidad": "CDMX"}}')


async def main():
    await run_ingestion()


if __name__ == "__main__":
    asyncio.run(main())
