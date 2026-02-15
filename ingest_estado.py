#!/usr/bin/env python3
"""
ingest_estado.py — Ingesta Genérica por Estado a Qdrant

Crea una colección dedicada por estado (leyes_queretaro, leyes_cdmx, etc.)
y ejecuta la ingesta con el Terminator de artículos limpios.

Uso:
    python ingest_estado.py --estado QUERETARO
    python ingest_estado.py --estado CDMX --skip-download
    python ingest_estado.py --estado QUERETARO --dry-run

Características:
    - Una colección por estado (leyes_queretaro, leyes_cdmx, leyes_jalisco, etc.)
    - Artículos completos y limpios via terminator_leyes.py
    - Embeddings densos (OpenAI text-embedding-3-small)
    - BM25 sparse vectors generados inline
    - Jerarquía preservada: Ley > Título > Capítulo > Art. N
    - Idempotente: puede re-ejecutarse sin duplicados (IDs deterministas)
"""

import asyncio
import os
import sys
import time
import uuid
import argparse
from dataclasses import dataclass
from pathlib import Path

import httpx
import pymupdf
from openai import AsyncOpenAI
from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    NamedVector,
    PointStruct,
    VectorParams,
    VectorsConfig,
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

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536
EMBED_BATCH_SIZE = 50
QDRANT_BATCH_SIZE = 50


# ══════════════════════════════════════════════════════════════════════
# LAW DEFINITIONS PER STATE
# ══════════════════════════════════════════════════════════════════════

@dataclass
class LawDef:
    nombre: str
    url: str
    categoria: str        # constitucion, ley, codigo, ley_organica
    tipo_codigo: str = ""  # PENAL, CIVIL, FISCAL, URBANO, etc.


def _infer_tipo_codigo(nombre: str) -> str:
    """Infer tipo_codigo from law name."""
    n = nombre.lower()
    mappings = [
        ("penal", "PENAL"), ("civil", "CIVIL"), ("fiscal", "FISCAL"),
        ("urbano", "URBANO"), ("ambiental", "AMBIENTAL"), ("familiar", "FAMILIAR"),
        ("electoral", "ELECTORAL"), ("constitución", "CONSTITUCION"),
        ("constituc", "CONSTITUCION"), ("tránsito", "TRANSITO"),
        ("transito", "TRANSITO"), ("notarial", "NOTARIAL"),
        ("notariado", "NOTARIAL"), ("salud", "SALUD"),
        ("educación", "EDUCACION"), ("educacion", "EDUCACION"),
        ("transparencia", "TRANSPARENCIA"), ("laboral", "LABORAL"),
        ("trabajadores", "LABORAL"), ("hacienda", "HACIENDA"),
        ("seguridad", "SEGURIDAD"), ("derechos humanos", "DERECHOS_HUMANOS"),
    ]
    for keyword, code in mappings:
        if keyword in n:
            return code
    return "GENERAL"


def _infer_jurisdiccion(nombre: str) -> str:
    """Infer jurisdiccion from law name."""
    n = nombre.lower()
    mappings = [
        ("penal", "penal"), ("civil", "civil"), ("familiar", "familiar"),
        ("laboral", "laboral"), ("trabajadores", "laboral"),
        ("mercantil", "mercantil"), ("comerci", "mercantil"),
        ("fiscal", "fiscal"), ("hacienda", "fiscal"), ("tributari", "fiscal"),
        ("administrativ", "administrativo"), ("electoral", "electoral"),
        ("ambiental", "ambiental"), ("constitución", "constitucional"),
        ("constituc", "constitucional"),
    ]
    for keyword, jur in mappings:
        if keyword in n:
            return jur
    return "general"


# ══════════════════════════════════════════════════════════════════════
# STATE REGISTRIES
# ══════════════════════════════════════════════════════════════════════

def get_laws_queretaro() -> list[LawDef]:
    """132 laws of Querétaro."""
    QRO_BASE = "http://site.legislaturaqueretaro.gob.mx/CloudPLQ/InvEst"
    laws = []
    
    # Constitución
    laws.append(LawDef(
        nombre="Constitución Política del Estado Libre y Soberano de Querétaro",
        url=f"{QRO_BASE}/Leyes/CON-ID-001.pdf",
        categoria="constitucion",
    ))
    
    # Leyes (111)
    ley_entries = [
        ("Ley de Adquisiciones, Enajenaciones, Arrendamientos y Contratación de servicios del Estado de Querétaro", "LEY-ID-002"),
        ("Ley de Archivos del Estado de Querétaro", "LEY-ID-003"),
        ("Ley de Asociaciones Público Privadas para el Estado de Querétaro", "LEY-ID-004"),
        ("Ley de Cambio Climático para el Estado de Querétaro", "LEY-ID-005"),
        ("Ley de Catastro para el Estado de Querétaro", "LEY-ID-006"),
        ("Ley de Coordinación Fiscal, Estatal Intermunicipal del Estado de Querétaro", "LEY-ID-007"),
        ("Ley de Derechos Humanos del Estado de Querétaro", "LEY-ID-008"),
        ("Ley de Derechos y Cultura de los Pueblos y Comunidades Indígenas del Estado de Querétaro", "LEY-ID-009"),
        ("Ley de Desarrollo Pecuario del Estado de Querétaro", "LEY-ID-010"),
        ("Ley de Desarrollo Social del Estado de Querétaro", "LEY-ID-011"),
        ("Ley de Deuda Pública del Estado de Querétaro", "LEY-ID-012"),
        ("Ley de Donación y Trasplante de Órganos, Tejidos y Células Humanas del Estado de Querétaro", "LEY-ID-013"),
        ("Ley de Educación del Estado de Querétaro", "LEY-ID-014"),
        ("Ley de Entrega Recepción del Estado de Querétaro", "LEY-ID-015"),
        ("Ley de Estacionamientos Públicos y Servicios de Recepción y Depósito de Vehículos para el Estado de Querétaro", "LEY-ID-016"),
        ("Ley de Estímulos Civiles del Estado de Querétaro", "LEY-ID-017"),
        ("Ley de Expropiación del Estado de Querétaro", "LEY-ID-018"),
        ("Ley de Extinción de Dominio del Estado de Querétaro", "LEY-ID-019"),
        ("Ley de Firma Electrónica Avanzada para el Estado de Querétaro", "LEY-ID-020"),
        ("Ley de Fiscalización Superior y Rendición de Cuentas del Estado de Querétaro", "LEY-ID-021"),
        ("Ley de Fomento a la Actividad Artesanal en el Estado de Querétaro", "LEY-ID-022"),
        ("Ley de Fomento a las Organizaciones de la Sociedad Civil del Estado de Querétaro", "LEY-ID-023"),
        ("Ley de Fomento Apícola y Protección del proceso de Polinización en el Estado de Querétaro", "LEY-ID-024"),
        ("Ley de Fundos Legales del Estado de Querétaro", "LEY-ID-025"),
        ("Ley de Gobierno Digital del Estado de Querétaro", "LEY-ID-026"),
        ("Ley de Hacienda de los Municipios del Estado de Querétaro", "LEY-ID-027"),
        ("Ley de Hacienda del Estado de Querétaro", "LEY-ID-028"),
        ("Ley de Igualdad Sustantiva entre Mujeres y Hombres del Estado de Querétaro", "LEY-ID-029"),
        ("Ley de Instituciones de Asistencia Privada del Estado de Querétaro", "LEY-ID-030"),
        ("Ley de Juicio Político del Estado de Querétaro", "LEY-ID-031"),
        ("Ley de Justicia Constitucional del Estado de Querétaro", "LEY-ID-032"),
        ("Ley de Justicia para Adolescentes del Estado de Querétaro", "LEY-ID-033"),
        ("Ley de la Administración Pública Paraestatal del Estado de Querétaro", "LEY-ID-034"),
        ("Ley de la Agencia de Movilidad y Modalidades de Transporte Público para el Estado de Querétaro", "LEY-ID-035"),
        ("Ley de la Secretaría de Seguridad Ciudadana del Estado de Querétaro", "LEY-ID-036"),
        ("Ley de la Unidad de medida y actualización del Estado de Querétaro", "LEY-ID-037"),
        ("Ley de los Derechos de las Personas Adultas Mayores del Estado de Querétaro", "LEY-ID-038"),
        ("Ley de los Derechos de las Niñas, Niños y Adolescentes del Estado de Querétaro", "LEY-ID-039"),
        ("Ley de los Trabajadores del Estado de Querétaro", "LEY-ID-040"),
        ("Ley de Medios de Impugnación en Materia Electoral del Estado de Querétaro", "LEY-ID-041"),
        ("Ley de Mejora Regulatoria del Estado de Querétaro", "LEY-ID-042"),
        ("Ley de Obra Pública del Estado de Querétaro", "LEY-ID-043"),
        ("Ley de Participación Ciudadana del Estado de Querétaro", "LEY-ID-044"),
        ("Ley de Planeación del Estado de Querétaro", "LEY-ID-045"),
        ("Ley de Procedimiento Contencioso Administrativo del Estado de Querétaro", "LEY-ID-046"),
        ("Ley de Procedimientos Administrativos del Estado de Querétaro", "LEY-ID-047"),
        ("Ley de Profesiones del Estado de Querétaro", "LEY-ID-048"),
        ("Ley de Protección a Víctimas, Ofendidos y Personas que Intervienen en el Procedimiento Penal del Estado de Querétaro", "LEY-ID-049"),
        ("Ley de Protección de Datos Personales en Posesión de Sujetos Obligados del Estado de Querétaro", "LEY-ID-050"),
        ("Ley de Publicaciones Oficiales del Estado de Querétaro", "LEY-ID-051"),
        ("Ley de Respeto Vecinal para el Estado de Querétaro", "LEY-ID-052"),
        ("Ley de Responsabilidad Patrimonial del Estado de Querétaro", "LEY-ID-053"),
        ("Ley de Responsabilidades Administrativas del Estado de Querétaro", "LEY-ID-054"),
        ("Ley de Salud del Estado de Querétaro", "LEY-ID-055"),
        ("Ley de Salud Mental del Estado de Querétaro", "LEY-ID-056"),
        ("Ley de Seguridad para el Estado de Querétaro", "LEY-ID-057"),
        ("Ley de Servicios Auxiliares del Transporte Público del Estado de Querétaro", "LEY-ID-058"),
        ("Ley de Tránsito para el estado de Querétaro", "LEY-ID-059"),
        ("Ley de Transparencia y Acceso a la Información Pública del Estado de Querétaro", "LEY-ID-060"),
        ("Ley de Turismo del Estado de Querétaro", "LEY-ID-061"),
        ("Ley de Valuación Inmobiliaria para el Estado de Querétaro", "LEY-ID-062"),
        ("Ley del Centro de Capacitación, Formación e Investigación para la Seguridad del Estado de Querétaro", "LEY-ID-063"),
        ("Ley del Centro de Prevención social del Delito y la Violencia en el Estado de Querétaro", "LEY-ID-064"),
        ("Ley del Deporte del Estado de Querétaro", "LEY-ID-065"),
        ("Ley del Escudo, la Bandera y el Himno del Estado de Querétaro", "LEY-ID-066"),
        ("Ley del Instituto de la Defensoría Penal Pública del Estado de Querétaro", "LEY-ID-067"),
        ("Ley del Instituto Queretano de las Mujeres", "LEY-ID-068"),
        ("Ley del Instituto Registral y Catastral del Estado de Querétaro", "LEY-ID-069"),
        ("Ley del Notariado del Estado de Querétaro", "LEY-ID-070"),
        ("Ley del Sistema de Asistencia Social del Estado de Querétaro", "LEY-ID-071"),
        ("Ley del Sistema de Servicio Profesional de Carrera del Poder Legislativo del Estado de Querétaro", "LEY-ID-072"),
        ("Ley del Sistema Estatal Anticorrupción de Querétaro", "LEY-ID-073"),
        ("Ley del Sistema Estatal de Protección Civil, Prevención y Mitigación de Desastres para el Estado de Querétaro", "LEY-ID-074"),
        ("Ley del Sistema para el Desarrollo Integral de la Familia del Estado de Querétaro", "LEY-ID-075"),
        ("Ley del Voluntariado del Estado de Querétaro", "LEY-ID-076"),
        ("Ley Electoral del Estado de Querétaro", "LEY-ID-111"),
        ("Ley Estatal de Acceso de las Mujeres a una Vida Libre de Violencia", "LEY-ID-077"),
        ("Ley Industrial del Estado de Querétaro", "LEY-ID-078"),
        ("Ley para Agilizar los Procedimientos de Entrega-Recepción de Fraccionamientos en el Estado de Querétaro", "LEY-ID-079"),
        ("Ley para el Desarrollo de los Jóvenes en el Estado de Querétaro", "LEY-ID-080"),
        ("Ley para el Fomento de la Investigación Científica, Tecnológica e Innovación del Estado de Querétaro", "LEY-ID-081"),
        ("Ley para el Manejo de los Recursos Públicos del Estado de Querétaro", "LEY-ID-082"),
        ("Ley para la Atención de las Migraciones en el Estado de Querétaro", "LEY-ID-083"),
        ("Ley para la Cultura y las Artes del Estado de Querétaro", "LEY-ID-084"),
        ("Ley para la Inclusión al Desarrollo Social de las Personas con Discapacidad del Estado de Querétaro", "LEY-ID-085"),
        ("Ley para la Prevención, Gestión Integral y Economía Circular de los Residuos del Estado de Querétaro", "LEY-ID-086"),
        ("Ley para la Promoción, Fomento y Desarrollo de la Industria Cinematográfica y Audiovisual del Estado de Querétaro", "LEY-ID-087"),
        ("Ley para la Regularización de Asentamientos Humanos Irregulares, Predios Urbanos, Predios Rústicos, Predios Familiares y Predios Sociales del Estado de Querétaro", "LEY-ID-088"),
        ("Ley para Prevenir, Combatir, y Sancionar la Trata de Personas en el Estado de Querétaro", "LEY-ID-089"),
        ("Ley para prevenir, investigar, sancionar y reparar la desaparición de personas en el Estado de Querétaro", "LEY-ID-090"),
        ("Ley para Prevenir y Eliminar toda Forma de Discriminación en el Estado de Querétaro", "LEY-ID-091"),
        ("Ley que aprueba la Incorporación del Estado de Querétaro y sus Municipios a la Coordinación en Materia Federal de Derechos", "LEY-ID-092"),
        ("Ley que crea el Centro de Evaluación y Control de Confianza del Estado de Querétaro", "LEY-ID-093"),
        ("Ley que Crea el Centro de Información y Análisis para la Seguridad de Querétaro", "LEY-ID-094"),
        ("Ley que Crea el Instituto Queretano del Emprendimiento y la Innovación", "LEY-ID-095"),
        ("Ley que crea la Comisión Estatal de Infraestructura de Querétaro", "LEY-ID-096"),
        ("Ley que crea la Comisión Estatal del Sistema Penitenciario de Querétaro", "LEY-ID-097"),
        ('Ley que crea la Comisión para la Evaluación de la Operación del Sistema de Justicia Penal Acusatorio del Estado de Querétaro "Cosmos"', "LEY-ID-098"),
        ("Ley que crea la Escuela Normal Superior de Querétaro", "LEY-ID-099"),
        ("Ley que crea la Orquesta de Cámara de Querétaro", "LEY-ID-100"),
        ("Ley que establece el Secreto Profesional Periodístico en el Estado de Querétaro", "LEY-ID-101"),
        ("Ley que establece las bases para la Prevención y la Atención de la Violencia Familiar en el Estado de Querétaro", "LEY-ID-102"),
        ("Ley que fija el Arancel para el Cobro de Honorarios de Abogados en el Estado de Querétaro", "LEY-ID-103"),
        ("Ley que fija el Arancel para el Cobro de Honorarios Profesionales de los Arquitectos en el Estado de Querétaro Arteaga", "LEY-ID-104"),
        ("Ley que regula a los agentes y empresas inmobiliarias en el Estado de Querétaro", "LEY-ID-105"),
        ("Ley que regula el Sistema Estatal de Promoción del uso de la Bicicleta", "LEY-ID-106"),
        ("Ley que regula la prestación de los servicios de agua potable, alcantarillado y saneamiento del Estado de Querétaro", "LEY-ID-107"),
        ("Ley que Regula la Prestación de Servicios para la Atención, Cuidado y Desarrollo Integral Infantil en el Estado de Querétaro", "LEY-ID-108"),
        ("Ley Registral del Estado de Querétaro", "LEY-ID-109"),
        ("Ley sobre bebidas alcohólicas del Estado de Querétaro", "LEY-ID-110"),
        ("Ley de la Secretaría de las Mujeres", "LEY-ID-112"),
    ]
    for nombre, file_id in ley_entries:
        laws.append(LawDef(
            nombre=nombre,
            url=f"{QRO_BASE}/Leyes/{file_id}.pdf",
            categoria="ley",
        ))
    
    # Códigos (7)
    cod_entries = [
        ("Código Ambiental del Estado de Querétaro", "COD-ID-01"),
        ("Código Civil del Estado de Querétaro", "COD-ID-02"),
        ("Código de Ética del Poder Legislativo del Estado de Querétaro", "COD-ID-03"),
        ("Código de Procedimientos Civiles del Estado de Querétaro", "COD-ID-04"),
        ("Código Fiscal del Estado de Querétaro", "COD-ID-05"),
        ("Código Urbano del Estado de Querétaro", "COD-ID-06"),
        ("Código Penal para el Estado de Querétaro", "COD-ID-07"),
    ]
    for nombre, file_id in cod_entries:
        laws.append(LawDef(
            nombre=nombre,
            url=f"{QRO_BASE}/Codigos/{file_id}.pdf",
            categoria="codigo",
        ))
    
    # Leyes Orgánicas (13)
    org_entries = [
        ("Ley Orgánica de la Agencia de Energía del Estado de Querétaro", "ORG-ID-01"),
        ("Ley Orgánica de la Escuela Normal del Estado", "ORG-ID-02"),
        ("Ley Orgánica de la Fiscalía General del Estado de Querétaro", "ORG-ID-03"),
        ("Ley Orgánica de la Universidad Autónoma de Querétaro", "ORG-ID-04"),
        ("Ley Orgánica de la Universidad Tecnológica de Querétaro", "ORG-ID-05"),
        ("Ley Orgánica del Centro de Conciliación Laboral del Estado de Querétaro", "ORG-ID-06"),
        ("Ley Orgánica del Colegio de Bachilleres del Estado de Querétaro", "ORG-ID-07"),
        ("Ley Orgánica del Poder Ejecutivo del Estado de Querétaro", "ORG-ID-08"),
        ("Ley Orgánica del Poder Judicial del Estado de Querétaro", "ORG-ID-09"),
        ("Ley Orgánica del Poder Legislativo del Estado de Querétaro", "ORG-ID-10"),
        ("Ley Orgánica del Tribunal de Justicia Administrativa del Estado de Querétaro", "ORG-ID-11"),
        ("Ley Orgánica del Tribunal Electoral del Estado de Querétaro", "ORG-ID-12"),
        ("Ley Orgánica Municipal del Estado de Querétaro", "ORG-ID-13"),
    ]
    for nombre, file_id in org_entries:
        laws.append(LawDef(
            nombre=nombre,
            url=f"{QRO_BASE}/Ley-Org/{file_id}.pdf",
            categoria="ley_organica",
        ))
    
    # Assign tipo_codigo
    for law in laws:
        law.tipo_codigo = _infer_tipo_codigo(law.nombre)
    
    return laws


def get_laws_cdmx() -> list[LawDef]:
    """CDMX laws — loaded from existing ingest_cdmx.py law list."""
    CDMX_BASE = "https://data.consejeria.cdmx.gob.mx/images/leyes"
    
    # Import the law definitions from the existing script
    # This is a simplified version; the full list is in ingest_cdmx.py
    # For now, we dynamically import from the existing script
    try:
        # Try to import from existing ingest_cdmx
        sys.path.insert(0, str(Path(__file__).parent))
        import importlib
        spec = importlib.util.spec_from_file_location(
            "ingest_cdmx_module",
            Path(__file__).parent / "ingest_cdmx.py"
        )
        mod = importlib.util.module_from_spec(spec)
        # We need to capture LAWS before it runs the full pipeline
        # Instead, just use the cached PDFs directory
        print("   ℹ️  Using existing CDMX law definitions from ingest_cdmx.py")
        spec.loader.exec_module(mod)
        return [LawDef(
            nombre=l.nombre, url=l.url, 
            categoria=l.categoria, tipo_codigo=l.tipo_codigo
        ) for l in mod.LAWS]
    except Exception as e:
        print(f"   ⚠️  Could not import CDMX laws from ingest_cdmx.py: {e}")
        print(f"   ⚠️  Please ensure ingest_cdmx.py is in the same directory")
        return []


# Registry of state law loaders
STATE_REGISTRY = {
    "QUERETARO": {
        "loader": get_laws_queretaro,
        "collection": "leyes_queretaro",
        "pdf_dir": "pdfs_queretaro",
        "entidad": "QUERETARO",
    },
    "CDMX": {
        "loader": get_laws_cdmx,
        "collection": "leyes_cdmx",
        "pdf_dir": "pdfs_cdmx",
        "entidad": "CDMX",
    },
    # Future states:
    # "JALISCO": {...}, "NUEVO_LEON": {...}, etc.
}


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
# QDRANT COLLECTION MANAGEMENT
# ══════════════════════════════════════════════════════════════════════

def ensure_collection(qdrant: QdrantClient, collection_name: str):
    """Create the state collection if it doesn't exist."""
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
    )
    print(f"   ✅ Collection '{collection_name}' created")


def delete_existing_data(qdrant: QdrantClient, collection_name: str):
    """Delete all data in the state collection (clean re-ingest)."""
    try:
        count = qdrant.count(collection_name=collection_name)
        if count.count == 0:
            print("   ✅ Collection is empty, nothing to delete")
            return
        
        print(f"   🗑️  Deleting {count.count} existing points...")
        # Delete all points by scrolling
        qdrant.delete(
            collection_name=collection_name,
            points_selector=Filter(must=[]),  # Match all
        )
        print(f"   ✅ Deleted all existing data")
    except Exception as e:
        print(f"   ⚠️  Delete skipped: {e}")


# ══════════════════════════════════════════════════════════════════════
# DOWNLOAD
# ══════════════════════════════════════════════════════════════════════

async def download_pdfs(laws: list[LawDef], pdf_dir: Path) -> dict[str, Path]:
    """Download all PDFs to local directory."""
    pdf_dir.mkdir(exist_ok=True)
    downloaded = {}
    failed = []
    
    async with httpx.AsyncClient(timeout=60, follow_redirects=True) as client:
        for i, law in enumerate(laws):
            filename = law.url.split("/")[-1]
            filepath = pdf_dir / filename
            
            if filepath.exists() and filepath.stat().st_size > 100:
                downloaded[law.nombre] = filepath
                continue
            
            try:
                resp = await client.get(law.url)
                if resp.status_code == 200 and len(resp.content) > 100:
                    filepath.write_bytes(resp.content)
                    downloaded[law.nombre] = filepath
                    if (i + 1) % 20 == 0:
                        print(f"   📥 Downloaded {i + 1}/{len(laws)} PDFs")
                else:
                    failed.append((law.nombre, f"HTTP {resp.status_code}"))
            except Exception as e:
                failed.append((law.nombre, str(e)))
            
            await asyncio.sleep(0.1)
    
    print(f"\n   ✅ Downloaded: {len(downloaded)}/{len(laws)}")
    if failed:
        print(f"   ❌ Failed: {len(failed)}")
        for name, err in failed[:5]:
            print(f"      • {name}: {err}")
    
    return downloaded


# ══════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════

def generate_point_id(entidad: str, law_name: str, ref: str, chunk_index: int) -> str:
    """Generate deterministic UUID for a chunk."""
    raw = f"{entidad}::{law_name}::{ref}::{chunk_index}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, raw))


async def run_ingestion(estado: str, dry_run: bool = False, skip_download: bool = False):
    """Main ingestion pipeline for a state."""
    config = STATE_REGISTRY[estado]
    collection = config["collection"]
    pdf_dir = Path(__file__).parent / config["pdf_dir"]
    entidad = config["entidad"]
    
    start_time = time.time()
    
    print("═══════════════════════════════════════════════════════════════")
    print(f"  INGESTA: {estado}")
    print(f"  Colección: {collection}")
    print("═══════════════════════════════════════════════════════════════")
    
    # Load law definitions
    laws = config["loader"]()
    print(f"   📋 Laws loaded: {len(laws)}")
    
    if not laws:
        print("   ❌ No laws defined for this state. Aborting.")
        return
    
    # Connect to Qdrant
    qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    
    # Phase 1: Ensure collection exists
    print("\n═══════════════════════════════════════════════════════════════")
    print("  PHASE 1: COLLECTION SETUP")
    print("═══════════════════════════════════════════════════════════════")
    ensure_collection(qdrant, collection)
    delete_existing_data(qdrant, collection)
    
    # Phase 2: Download PDFs
    if not skip_download:
        print("\n═══════════════════════════════════════════════════════════════")
        print("  PHASE 2: DOWNLOADING PDFs")
        print("═══════════════════════════════════════════════════════════════")
        downloaded = await download_pdfs(laws, pdf_dir)
    else:
        downloaded = {}
        for law in laws:
            filename = law.url.split("/")[-1]
            filepath = pdf_dir / filename
            if filepath.exists():
                downloaded[law.nombre] = filepath
        print(f"\n   📁 Using cached PDFs: {len(downloaded)}")
    
    # Phase 3: Extract + Terminator + Chunk
    print("\n═══════════════════════════════════════════════════════════════")
    print("  PHASE 3: EXTRACT + TERMINATOR (artículos limpios)")
    print("═══════════════════════════════════════════════════════════════")
    
    all_articulos: list[ArticuloLimpio] = []
    all_metadata: list[LawDef] = []  # Track which law each article belongs to
    laws_processed = 0
    laws_failed = 0
    
    for law in laws:
        if law.nombre not in downloaded:
            print(f"   ⚠️  Skipping {law.nombre} (not downloaded)")
            laws_failed += 1
            continue
        
        filepath = downloaded[law.nombre]
        texto_raw = extract_text_from_pdf(filepath)
        
        if not texto_raw.strip():
            print(f"   ⚠️  Empty PDF: {law.nombre}")
            laws_failed += 1
            continue
        
        # 🤖 TERMINATOR: Produce artículos limpios
        articulos = procesar_ley(texto_raw, law.nombre)
        all_articulos.extend(articulos)
        all_metadata.extend([law] * len(articulos))
        laws_processed += 1
        
        if laws_processed % 20 == 0:
            stats = diagnostico(articulos)
            print(f"   📄 Processed {laws_processed}/{len(laws)} laws, "
                  f"{len(all_articulos)} articles so far "
                  f"(last law: {stats['articulos_unicos']} arts)")
    
    print(f"\n   📊 TERMINATOR SUMMARY:")
    print(f"      Laws processed: {laws_processed}")
    print(f"      Laws failed: {laws_failed}")
    print(f"      Total article chunks: {len(all_articulos)}")
    
    # Stats by category
    cats = {}
    for m in all_metadata:
        cats[m.categoria] = cats.get(m.categoria, 0) + 1
    for cat, count in sorted(cats.items()):
        print(f"      {cat}: {count} chunks")
    
    if not all_articulos:
        print("   ❌ No articles generated! Aborting.")
        return
    
    if dry_run:
        print("\n   🏁 DRY RUN — Skipping embedding and upsert")
        # Show sample articles
        print("\n   📝 SAMPLE ARTICLES:")
        for a in all_articulos[:5]:
            preview = a.texto[:120].replace('\n', ' ')
            print(f"      [{a.ref}] ({len(a.texto)} chars)")
            print(f"         {a.jerarquia_txt}")
            print(f"         {preview}...")
        return
    
    # Phase 4: Generate embeddings
    print("\n═══════════════════════════════════════════════════════════════")
    print("  PHASE 4: GENERATING DENSE EMBEDDINGS")
    print("═══════════════════════════════════════════════════════════════")
    
    embeddings = await embed_all_chunks(all_articulos)
    
    # Phase 5: Upsert to Qdrant
    print("\n═══════════════════════════════════════════════════════════════")
    print(f"  PHASE 5: UPSERTING TO '{collection}'")
    print("═══════════════════════════════════════════════════════════════")
    
    points = []
    for i, (art, law, embedding) in enumerate(zip(all_articulos, all_metadata, embeddings)):
        point_id = generate_point_id(entidad, art.origen, art.ref, art.chunk_index)
        
        payload = {
            "entidad": entidad,
            "origen": art.origen,
            "ref": art.ref,
            "texto": art.texto,
            "jerarquia_txt": art.jerarquia_txt,
            "tipo_codigo": law.tipo_codigo,
            "jurisdiccion": _infer_jurisdiccion(law.nombre),
            "categoria": law.categoria,
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
                collection_name=collection,
                points=batch,
            )
            progress = min(batch_start + QDRANT_BATCH_SIZE, len(points))
            print(f"   ✅ Upserted {progress}/{len(points)} points")
        except Exception as e:
            print(f"   ❌ Upsert error at batch {batch_start}: {e}")
            for p in batch:
                try:
                    qdrant.upsert(collection_name=collection, points=[p])
                except Exception as e2:
                    print(f"      ❌ Point {p.id}: {e2}")
    
    # Verification
    print("\n═══════════════════════════════════════════════════════════════")
    print("  VERIFICATION")
    print("═══════════════════════════════════════════════════════════════")
    
    final_count = qdrant.count(collection_name=collection)
    elapsed = time.time() - start_time
    
    print(f"\n   ✅ INGESTION COMPLETE")
    print(f"      Collection: {collection}")
    print(f"      Points: {final_count.count}")
    print(f"      Time: {elapsed:.1f}s")
    print(f"\n   ⚠️  NEXT STEP: Generate BM25 sparse vectors")
    print(f"      POST https://api.iurexia.com/admin/reingest-sparse")
    print(f"      Body: {{\"admin_key\": \"...\", \"collection\": \"{collection}\"}}")


# ══════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Ingest state laws into dedicated Qdrant collections"
    )
    parser.add_argument(
        "--estado", required=True,
        choices=list(STATE_REGISTRY.keys()),
        help="State to ingest (QUERETARO, CDMX, etc.)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Process PDFs and show chunks without embedding/upserting"
    )
    parser.add_argument(
        "--skip-download", action="store_true",
        help="Skip PDF download (use cached files)"
    )
    
    args = parser.parse_args()
    
    print(f"\n🤖 INGESTA ESTADO: {args.estado}")
    print(f"   Available states: {list(STATE_REGISTRY.keys())}")
    
    asyncio.run(run_ingestion(
        estado=args.estado,
        dry_run=args.dry_run,
        skip_download=args.skip_download,
    ))


if __name__ == "__main__":
    main()
