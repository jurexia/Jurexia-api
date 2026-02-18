#!/usr/bin/env python3
"""
ingest_queretaro.py — Article-Aware Ingestion for Querétaro State Laws
═══════════════════════════════════════════════════════════════════════

Downloads 132 official PDFs from legislaturaqueretaro.gob.mx,
applies article-aware chunking, generates dense embeddings (text-embedding-3-small),
and upserts to the leyes_estatales Qdrant collection.

BM25 sparse vectors are generated AFTER ingestion via /admin/reingest-sparse.

Usage:
    python ingest_queretaro.py                    # Full pipeline
    python ingest_queretaro.py --delete-only      # Only delete existing Querétaro data
    python ingest_queretaro.py --skip-download     # Skip PDF download (use cached)
"""

import asyncio
import hashlib
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
    SparseVector,
    NamedSparseVector,
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

COLLECTION = "leyes_estatales"
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536
ENTIDAD = "QUERETARO"

# Article-aware chunking parameters
MAX_CHUNK_TOKENS = 1200       # Max tokens per chunk (~4800 chars)
OVERLAP_CHARS = 400           # Overlap between sub-chunks for long articles
MIN_CHUNK_LEN = 50            # Skip chunks shorter than this (noise)

# PDF download directory
PDF_DIR = Path(__file__).parent / "pdfs_queretaro"

# Rate limiting
EMBED_BATCH_SIZE = 50         # OpenAI allows up to 2048 inputs
EMBED_CONCURRENCY = 5         # Concurrent embedding batches
QDRANT_BATCH_SIZE = 50        # Points per upsert batch

# ══════════════════════════════════════════════════════════════════════
# ALL 132 QUERÉTARO LAWS
# ══════════════════════════════════════════════════════════════════════

QRO_BASE = "http://site.legislaturaqueretaro.gob.mx/CloudPLQ/InvEst"

@dataclass
class LawDef:
    nombre: str
    url: str
    categoria: str  # constitucion, ley, codigo, ley_organica
    tipo_codigo: str = ""  # PENAL, CIVIL, FISCAL, AMBIENTAL, URBANO, etc.

def _infer_tipo_codigo(nombre: str, categoria: str) -> str:
    """Infer tipo_codigo from law name for Da Vinci filtering."""
    n = nombre.lower()
    if "penal" in n:
        return "PENAL"
    if "civil" in n and "procedimiento" not in n:
        return "CIVIL"
    if "procedimientos civiles" in n:
        return "PROCESAL_CIVIL"
    if "fiscal" in n:
        return "FISCAL"
    if "urbano" in n:
        return "URBANO"
    if "ambiental" in n:
        return "AMBIENTAL"
    if "familiar" in n:
        return "FAMILIAR"
    if "electoral" in n:
        return "ELECTORAL"
    if "constitución" in n or "constituc" in n:
        return "CONSTITUCION"
    if "tránsito" in n or "transito" in n:
        return "TRANSITO"
    if "notarial" in n or "notariado" in n:
        return "NOTARIAL"
    if "salud" in n:
        return "SALUD"
    if "educación" in n or "educacion" in n:
        return "EDUCACION"
    if "transparencia" in n:
        return "TRANSPARENCIA"
    if "laboral" in n or "trabajadores" in n:
        return "LABORAL"
    if "hacienda" in n:
        return "HACIENDA"
    if "seguridad" in n:
        return "SEGURIDAD"
    if "derechos humanos" in n:
        return "DERECHOS_HUMANOS"
    return "GENERAL"

def _infer_jurisdiccion(nombre: str) -> str:
    """Infer jurisdiccion (legal matter) from law name."""
    n = nombre.lower()
    if "penal" in n:
        return "penal"
    if "civil" in n:
        return "civil"
    if "familiar" in n:
        return "familiar"
    if "laboral" in n or "trabajadores" in n:
        return "laboral"
    if "mercantil" in n or "comerci" in n:
        return "mercantil"
    if "fiscal" in n or "hacienda" in n or "tributari" in n:
        return "fiscal"
    if "administrativ" in n:
        return "administrativo"
    if "electoral" in n:
        return "electoral"
    if "ambiental" in n:
        return "ambiental"
    if "constitución" in n or "constituc" in n:
        return "constitucional"
    return "general"

# Build the complete law registry
LAWS: list[LawDef] = []

def _build_laws():
    """Build the full list of 132 laws."""
    # Constitución (1)
    LAWS.append(LawDef(
        nombre="Constitución Política del Estado Libre y Soberano de Querétaro",
        url=f"{QRO_BASE}/Leyes/CON-ID-001.pdf",
        categoria="constitucion",
    ))

    # Leyes (111) — IDs from estadosData.ts
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
        ("Ley que crea la Comisión para la Evaluación de la Operación del Sistema de Justicia Penal Acusatorio del Estado de Querétaro \"Cosmos\"", "LEY-ID-098"),
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
        LAWS.append(LawDef(
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
        LAWS.append(LawDef(
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
        LAWS.append(LawDef(
            nombre=nombre,
            url=f"{QRO_BASE}/Ley-Org/{file_id}.pdf",
            categoria="ley_organica",
        ))

    # Assign tipo_codigo and jurisdiccion
    for law in LAWS:
        law.tipo_codigo = _infer_tipo_codigo(law.nombre, law.categoria)


_build_laws()

# ══════════════════════════════════════════════════════════════════════
# ARTICLE-AWARE CHUNKING
# ══════════════════════════════════════════════════════════════════════

# Regex to detect article boundaries in Mexican legal text
ARTICLE_PATTERN = re.compile(
    r'(?:^|\n)'                           # Start of line
    r'(Art[ií]culo\s+\d+[\w]*'            # "Artículo 15", "Artículo 15 Bis"
    r'(?:\s+(?:BIS|TER|QUÁTER|QUINQUIES))?' # Optional suffixes
    r'[\.\-\s])',                          # Followed by period, dash or space
    re.IGNORECASE | re.MULTILINE
)

# Regex to extract article reference
ARTICLE_REF_PATTERN = re.compile(
    r'Art[ií]culo\s+(\d+[\w]*(?:\s+(?:BIS|TER|QUÁTER|QUINQUIES))?)',
    re.IGNORECASE
)

# Regex to detect section headers (títulos, capítulos, secciones)
SECTION_PATTERN = re.compile(
    r'(?:^|\n)\s*((?:TÍTULO|CAPITULO|CAPÍTULO|SECCIÓN|SECCION|LIBRO)\s+[IVXLCDM\d]+)',
    re.IGNORECASE | re.MULTILINE
)

@dataclass
class Chunk:
    """A single chunk ready for embedding."""
    text: str
    origin: str            # Law name
    ref: str               # Article reference (e.g., "Art. 15")
    jerarquia_txt: str     # Hierarchical context
    tipo_codigo: str       # Code type for Da Vinci
    jurisdiccion: str      # Legal matter
    categoria: str         # constitucion, ley, codigo, ley_organica
    chunk_index: int = 0   # Sub-chunk index (for long articles)


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
    """
    Split legal text into article-aware chunks.
    
    Strategy:
    1. Split by article boundaries
    2. Each article = 1 chunk (if < MAX_CHUNK_TOKENS)
    3. Long articles → sub-chunks with overlap
    4. Non-article text (preambles, transitorios) → fixed-size chunks
    """
    if not text.strip():
        return []
    
    # Clean the text
    text = re.sub(r'\r\n', '\n', text)
    text = re.sub(r'[ \t]+', ' ', text)  # Collapse horizontal whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)  # Max 2 newlines
    
    chunks: list[Chunk] = []
    jurisdiccion = _infer_jurisdiccion(law.nombre)
    
    # Track the current section header
    current_section = ""
    
    # Split by article boundaries
    splits = ARTICLE_PATTERN.split(text)
    
    if len(splits) <= 1:
        # No articles found — chunk as fixed-size
        return _fixed_size_chunk(text, law, jurisdiccion)
    
    # Process preamble (text before first article)
    preamble = splits[0].strip()
    if preamble and len(preamble) > MIN_CHUNK_LEN:
        # Extract any section headers from preamble
        sec_match = SECTION_PATTERN.search(preamble)
        if sec_match:
            current_section = sec_match.group(1).strip()
        
        for i, sub in enumerate(_split_long_text(preamble)):
            chunks.append(Chunk(
                text=sub,
                origin=law.nombre,
                ref="Preámbulo",
                jerarquia_txt=f"{law.nombre} > Preámbulo",
                tipo_codigo=law.tipo_codigo,
                jurisdiccion=jurisdiccion,
                categoria=law.categoria,
                chunk_index=i,
            ))
    
    # Process article-text pairs
    # splits alternates: [preamble, "Artículo X.", article_text, "Artículo Y.", article_text, ...]
    i = 1
    while i < len(splits):
        art_header = splits[i].strip() if i < len(splits) else ""
        art_body = splits[i + 1].strip() if (i + 1) < len(splits) else ""
        
        # Full article text
        full_article = f"{art_header} {art_body}".strip()
        
        if len(full_article) < MIN_CHUNK_LEN:
            i += 2
            continue
        
        # Extract article reference
        ref_match = ARTICLE_REF_PATTERN.search(art_header)
        art_ref = f"Art. {ref_match.group(1)}" if ref_match else art_header[:30]
        
        # Update section if we find a header
        sec_match = SECTION_PATTERN.search(art_body)
        if sec_match:
            current_section = sec_match.group(1).strip()
        
        jerarquia = f"{law.nombre} > {current_section} > {art_ref}" if current_section else f"{law.nombre} > {art_ref}"
        
        # Split long articles into sub-chunks
        sub_texts = _split_long_text(full_article)
        for j, sub in enumerate(sub_texts):
            chunks.append(Chunk(
                text=sub,
                origin=law.nombre,
                ref=art_ref,
                jerarquia_txt=jerarquia,
                tipo_codigo=law.tipo_codigo,
                jurisdiccion=jurisdiccion,
                categoria=law.categoria,
                chunk_index=j,
            ))
        
        i += 2
    
    # Check for transitorios after the last article
    if splits and len(splits) > 2:
        last_text = splits[-1]
        trans_match = re.search(r'(?:^|\n)(TRANSITORIOS?)\s*\n', last_text, re.IGNORECASE)
        if trans_match:
            trans_text = last_text[trans_match.start():].strip()
            if len(trans_text) > MIN_CHUNK_LEN:
                for j, sub in enumerate(_split_long_text(trans_text)):
                    chunks.append(Chunk(
                        text=sub,
                        origin=law.nombre,
                        ref="Transitorios",
                        jerarquia_txt=f"{law.nombre} > Transitorios",
                        tipo_codigo=law.tipo_codigo,
                        jurisdiccion=jurisdiccion,
                        categoria=law.categoria,
                        chunk_index=j,
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
        
        # Try to split at a paragraph or sentence boundary
        split_point = text.rfind('\n\n', start + max_chars // 2, end)
        if split_point == -1:
            split_point = text.rfind('. ', start + max_chars // 2, end)
        if split_point == -1:
            split_point = text.rfind(' ', start + max_chars // 2, end)
        if split_point == -1:
            split_point = end
        else:
            split_point += 1  # Include the delimiter
        
        parts.append(text[start:split_point])
        start = split_point - OVERLAP_CHARS  # Overlap
    
    return parts


def _fixed_size_chunk(text: str, law: LawDef, jurisdiccion: str) -> list[Chunk]:
    """Fallback: chunk text into fixed-size pieces when no articles are detected."""
    chunks = []
    parts = _split_long_text(text, max_chars=3200)
    for i, part in enumerate(parts):
        if len(part.strip()) < MIN_CHUNK_LEN:
            continue
        chunks.append(Chunk(
            text=part.strip(),
            origin=law.nombre,
            ref=f"Sección {i + 1}",
            jerarquia_txt=f"{law.nombre} > Sección {i + 1}",
            tipo_codigo=law.tipo_codigo,
            jurisdiccion=jurisdiccion,
            categoria=law.categoria,
            chunk_index=0,
        ))
    return chunks


# ══════════════════════════════════════════════════════════════════════
# EMBEDDING + QDRANT
# ══════════════════════════════════════════════════════════════════════

openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)


async def get_dense_embeddings(texts: list[str]) -> list[list[float]]:
    """Get dense embeddings from OpenAI in batch."""
    # Truncate texts that are too long for the API (max ~8191 tokens)
    truncated = [t[:30000] for t in texts]  # ~7500 tokens
    
    resp = await openai_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=truncated,
    )
    return [d.embedding for d in resp.data]


async def embed_all_chunks(chunks: list[Chunk]) -> list[list[float]]:
    """Embed all chunks with rate limiting and batching."""
    all_embeddings: list[list[float]] = [[] for _ in chunks]
    
    # Process in batches
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
            # Retry with smaller batches
            for j in range(batch_start, batch_end):
                try:
                    embs = await get_dense_embeddings([chunks[j].text])
                    all_embeddings[j] = embs[0]
                except Exception as e2:
                    print(f"   ❌ Skip chunk {j}: {e2}")
                    all_embeddings[j] = [0.0] * EMBEDDING_DIM
        
        # Rate limit: 500 RPM for text-embedding-3-small
        await asyncio.sleep(0.15)
    
    return all_embeddings


def generate_point_id(law_name: str, ref: str, chunk_index: int) -> str:
    """Generate a deterministic UUID for a chunk."""
    raw = f"{ENTIDAD}::{law_name}::{ref}::{chunk_index}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, raw))


# ══════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════

async def download_pdfs() -> dict[str, Path]:
    """Download all PDFs to local directory."""
    PDF_DIR.mkdir(exist_ok=True)
    
    downloaded = {}
    failed = []
    
    async with httpx.AsyncClient(timeout=60, follow_redirects=True) as client:
        for i, law in enumerate(LAWS):
            filename = law.url.split("/")[-1]
            filepath = PDF_DIR / filename
            
            # Skip if already downloaded
            if filepath.exists() and filepath.stat().st_size > 100:
                downloaded[law.nombre] = filepath
                continue
            
            try:
                resp = await client.get(law.url)
                if resp.status_code == 200 and len(resp.content) > 100:
                    filepath.write_bytes(resp.content)
                    downloaded[law.nombre] = filepath
                    if (i + 1) % 20 == 0:
                        print(f"   📥 Downloaded {i + 1}/{len(LAWS)} PDFs")
                else:
                    failed.append((law.nombre, f"HTTP {resp.status_code}"))
            except Exception as e:
                failed.append((law.nombre, str(e)))
            
            await asyncio.sleep(0.1)  # Rate limit
    
    print(f"\n   ✅ Downloaded: {len(downloaded)}/{len(LAWS)}")
    if failed:
        print(f"   ❌ Failed: {len(failed)}")
        for name, err in failed[:5]:
            print(f"      • {name}: {err}")
    
    return downloaded


def delete_queretaro_data(qdrant: QdrantClient):
    """Delete all existing Querétaro chunks from leyes_estatales."""
    print("\n═══════════════════════════════════════════════════════════════")
    print("  PHASE 1: DELETING EXISTING QUERÉTARO DATA")
    print("═══════════════════════════════════════════════════════════════")
    
    try:
        # Count existing
        total = qdrant.count(collection_name=COLLECTION)
        print(f"   📊 Total points in collection: {total.count}")
        
        if total.count == 0:
            print("   ✅ Collection is empty, nothing to delete")
            return
        
        # Try filtered count
        count = qdrant.count(
            collection_name=COLLECTION,
            count_filter=Filter(
                must=[FieldCondition(key="entidad", match=MatchValue(value=ENTIDAD))]
            ),
        )
        print(f"   📊 Existing Querétaro chunks: {count.count}")
        
        if count.count == 0:
            print("   ✅ No Querétaro data to delete")
            return
        
        # Delete by filter
        qdrant.delete(
            collection_name=COLLECTION,
            points_selector=Filter(
                must=[FieldCondition(key="entidad", match=MatchValue(value=ENTIDAD))]
            ),
        )
        print(f"   ✅ Deleted {count.count} Querétaro chunks")
        
    except Exception as e:
        print(f"   ⚠️ Delete phase skipped (collection may be empty/new): {e}")


async def run_ingestion():
    """Main ingestion pipeline."""
    start_time = time.time()
    
    print("═══════════════════════════════════════════════════════════════")
    print("  QUERÉTARO LAW INGESTION — Article-Aware Pipeline")
    print(f"  Laws: {len(LAWS)} | Collection: {COLLECTION}")
    print("═══════════════════════════════════════════════════════════════")
    
    # Connect to Qdrant
    qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    
    # Phase 1: Delete existing data
    delete_queretaro_data(qdrant)
    
    # Phase 2: Download PDFs
    if "--skip-download" not in sys.argv:
        print("\n═══════════════════════════════════════════════════════════════")
        print("  PHASE 2: DOWNLOADING PDFs")
        print("═══════════════════════════════════════════════════════════════")
        downloaded = await download_pdfs()
    else:
        # Use cached PDFs
        downloaded = {}
        for law in LAWS:
            filename = law.url.split("/")[-1]
            filepath = PDF_DIR / filename
            if filepath.exists():
                downloaded[law.nombre] = filepath
        print(f"\n   📁 Using cached PDFs: {len(downloaded)}")
    
    # Phase 3: Extract + Chunk
    print("\n═══════════════════════════════════════════════════════════════")
    print("  PHASE 3: EXTRACTING TEXT + ARTICLE-AWARE CHUNKING")
    print("═══════════════════════════════════════════════════════════════")
    
    all_chunks: list[Chunk] = []
    laws_processed = 0
    laws_failed = 0
    
    for law in LAWS:
        if law.nombre not in downloaded:
            print(f"   ⚠️  Skipping {law.nombre} (not downloaded)")
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
            print(f"   📄 Processed {laws_processed}/{len(LAWS)} laws, {len(all_chunks)} chunks so far")
    
    print(f"\n   📊 CHUNKING SUMMARY:")
    print(f"      Laws processed: {laws_processed}")
    print(f"      Laws failed: {laws_failed}")
    print(f"      Total chunks: {len(all_chunks)}")
    
    # Stats by category
    cats = {}
    for c in all_chunks:
        cats[c.categoria] = cats.get(c.categoria, 0) + 1
    for cat, count in sorted(cats.items()):
        print(f"      {cat}: {count} chunks")
    
    if not all_chunks:
        print("   ❌ No chunks generated! Aborting.")
        return
    
    # Phase 4: Generate embeddings
    print("\n═══════════════════════════════════════════════════════════════")
    print("  PHASE 4: GENERATING DENSE EMBEDDINGS")
    print("═══════════════════════════════════════════════════════════════")
    
    embeddings = await embed_all_chunks(all_chunks)
    
    # Phase 5: Upsert to Qdrant
    print("\n═══════════════════════════════════════════════════════════════")
    print("  PHASE 5: UPSERTING TO QDRANT")
    print("═══════════════════════════════════════════════════════════════")
    
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
        
        # Use named vectors (matching collection schema)
        point = PointStruct(
            id=point_id,
            vector={
                "dense": embedding,
            },
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
            # Try individual upserts
            for p in batch:
                try:
                    qdrant.upsert(collection_name=COLLECTION, points=[p])
                except Exception as e2:
                    print(f"      ❌ Point {p.id}: {e2}")
    
    # Verification
    print("\n═══════════════════════════════════════════════════════════════")
    print("  VERIFICATION")
    print("═══════════════════════════════════════════════════════════════")
    
    final_count = qdrant.count(
        collection_name=COLLECTION,
        count_filter=Filter(
            must=[FieldCondition(key="entidad", match=MatchValue(value=ENTIDAD))]
        ),
    )
    
    elapsed = time.time() - start_time
    
    print(f"\n   ✅ INGESTION COMPLETE")
    print(f"      Querétaro chunks in Qdrant: {final_count.count}")
    print(f"      Time elapsed: {elapsed:.1f}s")
    print(f"\n   ⚠️  NEXT STEP: Trigger BM25 sparse vector generation via:")
    print(f"      POST https://api.iurexia.com/admin/reingest-sparse")
    print(f"      Body: {{\"admin_key\": \"...\", \"entidad\": \"QUERETARO\"}}")


async def main():
    if "--delete-only" in sys.argv:
        qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        delete_queretaro_data(qdrant)
    else:
        await run_ingestion()


if __name__ == "__main__":
    asyncio.run(main())
