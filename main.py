"""
api_jurexia_core.py - Motor de Producción Jurexia
──────────────────────────────────────────────────
FastAPI backend para plataforma LegalTech con:
- Búsqueda Híbrida (BM25 + Dense OpenAI)
- Filtros estrictos de jurisdicción
- Inyección de contexto XML
- Agente Centinela para auditoría legal
- Memoria conversacional stateless con streaming
- Grounding con citas documentales
"""

import asyncio
import html
import json
import os
import re
import uuid
from typing import AsyncGenerator, List, Literal, Optional, Dict, Set, Tuple
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient, AsyncQdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import (
    FieldCondition,
    Filter,
    MatchValue,
    NamedVector,
    NamedSparseVector,
    Prefetch,
    Query,
    SparseVector,
)
from fastembed import SparseTextEmbedding
from openai import AsyncOpenAI

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ══════════════════════════════════════════════════════════════════════════════

# Cargar variables de entorno desde .env
from dotenv import load_dotenv
load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL", "https://your-cluster.qdrant.tech")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")

# DeepSeek API Configuration
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
CHAT_MODEL = "deepseek-chat"  # or deepseek-reasoner for R1

# For embeddings, we still use OpenAI (DeepSeek doesn't have embeddings)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Silos V4.2 de Jurexia (incluye Bloque de Constitucionalidad)
SILOS = {
    "federal": "leyes_federales",
    "estatal": "leyes_estatales",
    "jurisprudencia": "jurisprudencia_nacional",
    "constitucional": "bloque_constitucional",  # Constitución, Tratados DDHH, Jurisprudencia CoIDH
}

# Estados mexicanos válidos (normalizados a mayúsculas)
ESTADOS_MEXICO = [
    "AGUASCALIENTES", "BAJA_CALIFORNIA", "BAJA_CALIFORNIA_SUR", "CAMPECHE",
    "CHIAPAS", "CHIHUAHUA", "CIUDAD_DE_MEXICO", "COAHUILA", "COLIMA",
    "DURANGO", "GUANAJUATO", "GUERRERO", "HIDALGO", "JALISCO", "MEXICO",
    "MICHOACAN", "MORELOS", "NAYARIT", "NUEVO_LEON", "OAXACA", "PUEBLA",
    "QUERETARO", "QUINTANA_ROO", "SAN_LUIS_POTOSI", "SINALOA", "SONORA",
    "TABASCO", "TAMAULIPAS", "TLAXCALA", "VERACRUZ", "YUCATAN", "ZACATECAS",
]

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536

# ══════════════════════════════════════════════════════════════════════════════
# SYSTEM PROMPTS
# ══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT_CHAT = """Eres JUREXIA, IA Jurídica especializada en Derecho Mexicano.

PRINCIPIO PRO PERSONA (Art. 1° CPEUM):
En temas de derechos humanos, aplica SIEMPRE la interpretación más favorable a la persona.
Cuando existan normas en conflicto, prioriza: Bloque Constitucional > Leyes Federales > Leyes Estatales.

INSTRUCCIONES:
- Cita con formato: [Doc ID: uuid_completo]
- Para temas de DDHH, SIEMPRE prioriza documentos del silo="bloque_constitucional":
  * Constitución (CPEUM): tipo="CONSTITUCION" → Artículos 1-29 son fundamentales
  * Tratados DDHH (CADH, PIDCP, CAT, CDN): tipo="TRATADO_INTERNACIONAL"
  * Jurisprudencia CoIDH: tipo="JURISPRUDENCIA_INTERAMERICANA" → Vinculante para México
- La jurisprudencia de la CoIDH es OBLIGATORIA para México desde el Caso Radilla Pacheco (2009)
- NUNCA digas "no hay jurisprudencia" si hay documentos relevantes en cualquier silo

ESTRUCTURA DE RESPUESTA:

## 1. Conceptualización
Define la figura jurídica consultada.

## 2. Marco Constitucional y Convencional
> "Artículo X.- [contenido]" — *CPEUM* [Doc ID: uuid]
> "Artículo X.- [contenido]" — *CADH/PIDCP* [Doc ID: uuid]

## 3. Jurisprudencia Interamericana (si aplica)
> "Párrafo X.- [contenido]" — *CoIDH, Caso [Nombre]* [Doc ID: uuid]

## 4. Fundamento Legal Secundario
> "Artículo X.- [contenido]" — *[Ley/Código]* [Doc ID: uuid]

## 5. Jurisprudencia Nacional
> "[Rubro de tesis]" — *SCJN, Registro [X]* [Doc ID: uuid]

## 6. Conclusión Pro Persona
Síntesis práctica aplicando la interpretación más favorable.
"""




SYSTEM_PROMPT_AUDIT = """Eres un Auditor Legal Experto. Tu tarea es analizar documentos legales contra la evidencia jurídica proporcionada.

INSTRUCCIONES:
1. Extrae los "Puntos Controvertidos" del documento analizado.
2. Evalúa cada punto contra la evidencia proporcionada en las etiquetas <documento>.
3. Identifica Fortalezas, Debilidades y Sugerencias.
4. SIEMPRE cita usando [Doc ID: X].

RETORNA TU ANÁLISIS EN EL SIGUIENTE FORMATO JSON ESTRICTO:
{
    "puntos_controvertidos": ["..."],
    "fortalezas": [{"punto": "...", "fundamento": "...", "citas": ["Doc ID: X"]}],
    "debilidades": [{"punto": "...", "problema": "...", "citas": ["Doc ID: X"]}],
    "sugerencias": [{"accion": "...", "justificacion": "...", "citas": ["Doc ID: X"]}],
    "riesgo_general": "BAJO|MEDIO|ALTO",
    "resumen_ejecutivo": "..."
}
"""

# ══════════════════════════════════════════════════════════════════════════════
# MODELOS PYDANTIC
# ══════════════════════════════════════════════════════════════════════════════

class Message(BaseModel):
    """Mensaje del historial conversacional"""
    role: Literal["user", "assistant", "system"]
    content: str


class SearchRequest(BaseModel):
    """Request para búsqueda híbrida"""
    query: str = Field(..., min_length=1, max_length=2000)
    estado: Optional[str] = Field(None, description="Estado mexicano (ej: NUEVO_LEON)")
    top_k: int = Field(10, ge=1, le=50)
    alpha: float = Field(0.7, ge=0.0, le=1.0, description="Balance dense/sparse (1=solo dense)")


class SearchResult(BaseModel):
    """Resultado individual de búsqueda"""
    id: str
    score: float
    texto: str
    ref: Optional[str] = None
    origen: Optional[str] = None
    jurisdiccion: Optional[str] = None
    entidad: Optional[str] = None
    silo: str


class SearchResponse(BaseModel):
    """Response de búsqueda"""
    query: str
    estado_filtrado: Optional[str]
    resultados: List[SearchResult]
    total: int


class ChatRequest(BaseModel):
    """Request para chat conversacional"""
    messages: List[Message] = Field(..., min_items=1)
    estado: Optional[str] = Field(None, description="Estado para filtrado jurisdiccional")
    top_k: int = Field(4, ge=1, le=30)  # Reduced to 4 to stay within 8k token limit


class AuditRequest(BaseModel):
    """Request para auditoría de documento legal"""
    documento: str = Field(..., min_length=50, description="Texto de la demanda/sentencia")
    estado: Optional[str] = Field(None)
    profundidad: Literal["rapida", "exhaustiva"] = "rapida"


class AuditResponse(BaseModel):
    """Response estructurada del agente centinela"""
    puntos_controvertidos: List[str]
    fortalezas: List[dict]
    debilidades: List[dict]
    sugerencias: List[dict]
    riesgo_general: str
    resumen_ejecutivo: str


class CitationValidation(BaseModel):
    """Resultado de validación de una cita individual"""
    doc_id: str
    exists_in_context: bool
    status: Literal["valid", "invalid", "not_found"]
    source_ref: Optional[str] = None  # Referencia del documento si existe


class ValidationResult(BaseModel):
    """Resultado completo de validación de citas"""
    total_citations: int
    valid_count: int
    invalid_count: int
    citations: List[CitationValidation]
    confidence_score: float  # Porcentaje de citas válidas (0-1)



# ══════════════════════════════════════════════════════════════════════════════
# CLIENTES GLOBALES (Lifecycle)
# ══════════════════════════════════════════════════════════════════════════════

sparse_encoder: SparseTextEmbedding = None
qdrant_client: AsyncQdrantClient = None
openai_client: AsyncOpenAI = None  # For embeddings only
deepseek_client: AsyncOpenAI = None  # For chat/reasoning


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Inicialización y cleanup de recursos"""
    global sparse_encoder, qdrant_client, openai_client, deepseek_client
    
    # Startup
    print("⚡ Inicializando Jurexia Core Engine...")
    
    # BM25 Sparse Encoder
    sparse_encoder = SparseTextEmbedding(model_name="Qdrant/bm25")
    print("  ✓ BM25 Encoder cargado")
    
    # Qdrant Async Client
    qdrant_client = AsyncQdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        timeout=30,
    )
    print("  ✓ Qdrant Client conectado")
    
    # OpenAI Client (for embeddings only)
    openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
    print("  ✓ OpenAI Client inicializado (embeddings)")
    
    # DeepSeek Client (for chat/reasoning)
    deepseek_client = AsyncOpenAI(
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE_URL,
    )
    print("  ✓ DeepSeek Client inicializado (chat)")
    
    print("🚀 Jurexia Core Engine LISTO")
    
    yield
    
    # Shutdown
    print("🔻 Cerrando conexiones...")
    await qdrant_client.close()


# ══════════════════════════════════════════════════════════════════════════════
# UTILIDADES
# ══════════════════════════════════════════════════════════════════════════════

def normalize_estado(estado: Optional[str]) -> Optional[str]:
    """Normaliza el nombre del estado a formato esperado"""
    if not estado:
        return None
    normalized = estado.upper().replace(" ", "_").replace("-", "_")
    # Corrección de variantes comunes
    if normalized in ["NUEVO_LEON", "NL", "NUEVOLEON"]:
        return "NUEVO_LEON"
    if normalized in ["CDMX", "DF", "CIUDAD_DE_MEXICO"]:
        return "CIUDAD_DE_MEXICO"
    if normalized in ESTADOS_MEXICO:
        return normalized
    return None


def build_state_filter(estado: Optional[str]) -> Optional[Filter]:
    """
    Construye filtro para leyes estatales SOLO.
    REGLA: Si hay estado seleccionado, filtra por ese estado específico.
    Este filtro solo se aplica a la colección leyes_estatales.
    """
    if not estado:
        return None
    
    normalized = normalize_estado(estado)
    if not normalized:
        return None
    
    # Filtro simple: solo documentos del estado seleccionado
    return Filter(
        must=[
            FieldCondition(key="entidad", match=MatchValue(value=normalized)),
        ]
    )


def get_filter_for_silo(silo_name: str, estado: Optional[str]) -> Optional[Filter]:
    """
    Retorna el filtro apropiado para cada silo.
    - leyes_estatales: Filtra por estado seleccionado
    - leyes_federales: Sin filtro (todo es aplicable a cualquier estado)
    - jurisprudencia_nacional: Sin filtro (toda es aplicable)
    - bloque_constitucional: Sin filtro (CPEUM, tratados y CoIDH aplican a todo)
    """
    if silo_name == "leyes_estatales" and estado:
        return build_state_filter(estado)
    # Para federales, jurisprudencia y bloque constitucional, no se aplica filtro de estado
    return None


# Sinónimos legales para query expansion (mejora recall BM25)
LEGAL_SYNONYMS = {
    "derecho del tanto": [
        "derecho de preferencia", "preferencia adquisición", 
        "socios gozarán del tanto", "enajenar partes sociales",
        "copropiedad preferencia", "colindantes vía pública",
        "propietarios predios colindantes", "retracto legal",
        "usufructuario goza del tanto", "copropiedad indivisa",
        "rescisión contrato ocho días", "aparcería enajenar"
    ],
    "amparo indirecto": [
        "juicio de amparo", "amparo ante juez de distrito", 
        "demanda de amparo", "acto reclamado"
    ],
    "pensión alimenticia": [
        "alimentos", "obligación alimentaria", "derechos alimentarios",
        "manutención", "asistencia familiar"
    ],
    "prescripción": [
        "caducidad", "extinción de acción", "término prescriptorio"
    ],
    "contrato": [
        "convenio", "acuerdo", "obligaciones contractuales"
    ],
    "arrendamiento": [
        "alquiler", "renta", "locación", "arrendador arrendatario"
    ],
    "compraventa": [
        "enajenación", "transmisión de dominio", "adquisición"
    ],
    "sucesión": [
        "herencia", "testamento", "herederos", "legado", "intestado"
    ],
    "divorcio": [
        "disolución matrimonial", "separación conyugal", "convenio de divorcio"
    ],
    "delito": [
        "ilícito penal", "hecho punible", "conducta típica"
    ],
}


def expand_legal_query(query: str) -> str:
    """
    Expande la consulta con sinónimos legales para mejorar el recall de BM25.
    Mantiene la consulta original y añade términos relacionados.
    """
    query_lower = query.lower()
    expanded_terms = [query]  # Siempre incluir la consulta original
    
    for key_term, synonyms in LEGAL_SYNONYMS.items():
        if key_term in query_lower:
            # Añadir más sinónimos para mejor cobertura
            expanded_terms.extend(synonyms[:6])  # Aumentado de 3 a 6
            break  # Solo expandir el primer match
    
    return " ".join(expanded_terms)


# Términos que indican query sobre derechos humanos
DDHH_KEYWORDS = {
    # Derechos fundamentales
    "derecho humano", "derechos humanos", "ddhh", "garantía", "garantías",
    "libertad", "igualdad", "dignidad", "integridad", "vida",
    # Principios
    "pro persona", "pro homine", "principio de progresividad", "no regresión",
    "interpretación conforme", "control de convencionalidad", "control difuso",
    # Tratados
    "convención americana", "cadh", "pacto de san josé", "pidcp",
    "convención contra la tortura", "cat", "convención del niño", "cedaw",
    # Corte IDH
    "corte interamericana", "coidh", "cidh", "comisión interamericana",
    # Violaciones
    "tortura", "desaparición forzada", "detención arbitraria", "discriminación",
    "debido proceso", "presunción de inocencia", "acceso a la justicia",
    # Artículos constitucionales DDHH
    "artículo 1", "art. 1", "artículo primero", "artículo 14", "artículo 16",
    "artículo 17", "artículo 19", "artículo 20", "artículo 21", "artículo 22",
}

def is_ddhh_query(query: str) -> bool:
    """
    Detecta si la consulta está relacionada con derechos humanos.
    Retorna True si la query contiene términos de DDHH.
    """
    query_lower = query.lower()
    return any(keyword in query_lower for keyword in DDHH_KEYWORDS)


async def get_dense_embedding(text: str) -> List[float]:
    """Genera embedding denso usando OpenAI"""
    response = await openai_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text,
    )
    return response.data[0].embedding


def get_sparse_embedding(text: str) -> SparseVector:
    """Genera embedding sparse usando BM25"""
    embeddings = list(sparse_encoder.query_embed(text))
    if not embeddings:
        return SparseVector(indices=[], values=[])
    
    sparse = embeddings[0]
    return SparseVector(
        indices=sparse.indices.tolist(),
        values=sparse.values.tolist(),
    )


# Maximum characters per document to prevent token overflow
MAX_DOC_CHARS = 600

def format_results_as_xml(results: List[SearchResult]) -> str:
    """
    Formatea resultados en XML para inyección de contexto.
    Escapa caracteres HTML para seguridad.
    Trunca documentos largos para evitar exceder límite de tokens.
    """
    if not results:
        return "<documentos>Sin resultados relevantes encontrados.</documentos>"
    
    xml_parts = ["<documentos>"]
    for r in results:
        # Truncate long documents to fit within token limits
        texto = r.texto
        if len(texto) > MAX_DOC_CHARS:
            texto = texto[:MAX_DOC_CHARS] + "... [truncado]"
        
        escaped_texto = html.escape(texto)
        escaped_ref = html.escape(r.ref or "N/A")
        escaped_origen = html.escape(r.origen or "Desconocido")
        
        xml_parts.append(
            f'<documento id="{r.id}" ref="{escaped_ref}" '
            f'origen="{escaped_origen}" silo="{r.silo}" score="{r.score:.4f}">\n'
            f'{escaped_texto}\n'
            f'</documento>'
        )
    xml_parts.append("</documentos>")
    
    return "\n".join(xml_parts)


# ══════════════════════════════════════════════════════════════════════════════
# VALIDADOR DE CITAS (Citation Grounding Verification)
# ══════════════════════════════════════════════════════════════════════════════

# Regex para extraer Doc IDs del formato [Doc ID: uuid]
DOC_ID_PATTERN = re.compile(r'\[Doc ID:\s*([a-f0-9\-]{36})\]', re.IGNORECASE)


def extract_doc_ids(text: str) -> List[str]:
    """
    Extrae todos los Doc IDs citados en el texto.
    Formato esperado: [Doc ID: uuid]
    """
    matches = DOC_ID_PATTERN.findall(text)
    return list(set(matches))  # Únicos


def build_doc_id_map(search_results: List[SearchResult]) -> Dict[str, SearchResult]:
    """
    Construye un diccionario de Doc ID -> SearchResult para validación rápida.
    """
    return {result.id: result for result in search_results}


def validate_citations(
    response_text: str,
    retrieved_docs: Dict[str, SearchResult]
) -> ValidationResult:
    """
    Valida que todas las citas en la respuesta del LLM correspondan
    a documentos realmente recuperados de Qdrant.
    
    Args:
        response_text: Texto de respuesta del LLM
        retrieved_docs: Diccionario de Doc ID -> SearchResult de docs recuperados
    
    Returns:
        ValidationResult con estadísticas y detalle de cada cita
    """
    cited_ids = extract_doc_ids(response_text)
    
    if not cited_ids:
        # Sin citas - permitido pero sin verificación
        return ValidationResult(
            total_citations=0,
            valid_count=0,
            invalid_count=0,
            citations=[],
            confidence_score=1.0  # Sin citas = no hay errores
        )
    
    validations = []
    valid_count = 0
    invalid_count = 0
    
    for doc_id in cited_ids:
        if doc_id in retrieved_docs:
            doc = retrieved_docs[doc_id]
            validations.append(CitationValidation(
                doc_id=doc_id,
                exists_in_context=True,
                status="valid",
                source_ref=doc.ref
            ))
            valid_count += 1
        else:
            validations.append(CitationValidation(
                doc_id=doc_id,
                exists_in_context=False,
                status="invalid",
                source_ref=None
            ))
            invalid_count += 1
    
    total = valid_count + invalid_count
    confidence = valid_count / total if total > 0 else 1.0
    
    return ValidationResult(
        total_citations=total,
        valid_count=valid_count,
        invalid_count=invalid_count,
        citations=validations,
        confidence_score=confidence
    )


def annotate_invalid_citations(response_text: str, invalid_ids: Set[str]) -> str:
    """
    Anota las citas inválidas en el texto con una advertencia visual.
    
    Ejemplo:
        [Doc ID: abc123] -> [Doc ID: abc123] ⚠️ *[Cita no verificada]*
    """
    if not invalid_ids:
        return response_text
    
    def replace_invalid(match):
        doc_id = match.group(1)
        original = match.group(0)
        if doc_id.lower() in [i.lower() for i in invalid_ids]:
            return f"{original} ⚠️ *[Cita no verificada]*"
        return original
    
    return DOC_ID_PATTERN.sub(replace_invalid, response_text)


def get_valid_doc_ids_prompt(retrieved_docs: Dict[str, SearchResult]) -> str:
    """
    Genera una lista de Doc IDs válidos para incluir en prompts de regeneración.
    """
    if not retrieved_docs:
        return "No hay documentos disponibles para citar."
    
    lines = ["DOCUMENTOS DISPONIBLES PARA CITAR (usa SOLO estos Doc IDs):"]
    for doc_id, doc in list(retrieved_docs.items())[:15]:  # Limitar a 15 para no saturar
        ref = doc.ref or "Sin referencia"
        lines.append(f"  - [Doc ID: {doc_id}] → {ref[:80]}")
    
    return "\n".join(lines)




async def hybrid_search_single_silo(
    collection: str,
    query: str,
    dense_vector: List[float],
    sparse_vector: SparseVector,
    filter_: Optional[Filter],
    top_k: int,
    alpha: float,
) -> List[SearchResult]:
    """
    Ejecuta búsqueda en un silo.
    Auto-detecta si la colección tiene sparse vectors para usar híbrido o solo dense.
    """
    try:
        # Verificar que la colección existe
        collections = await qdrant_client.get_collections()
        existing = [c.name for c in collections.collections]
        if collection not in existing:
            return []
        
        # Obtener info de la colección para detectar tipos de vectores
        col_info = await qdrant_client.get_collection(collection)
        vectors_config = col_info.config.params.vectors
        
        # Detectar si tiene vectores sparse
        has_sparse = isinstance(vectors_config, dict) and "sparse" in vectors_config
        
        if has_sparse:
            # Búsqueda Híbrida: Prefetch Sparse -> Rerank Dense
            # IMPORTANTE: El filtro se aplica tanto en prefetch como en query principal
            results = await qdrant_client.query_points(
                collection_name=collection,
                prefetch=[
                    Prefetch(
                        query=sparse_vector,
                        using="sparse",
                        limit=top_k * 3,
                        filter=filter_,
                    ),
                ],
                query=dense_vector,
                using="dense",
                limit=top_k,
                query_filter=filter_,  # CRÍTICO: Filtro también en rerank denso
                with_payload=True,
                score_threshold=0.1,
            )

        else:
            # Búsqueda Solo Dense (colecciones sin sparse)
            results = await qdrant_client.query_points(
                collection_name=collection,
                query=dense_vector,
                using="dense",
                limit=top_k,
                query_filter=filter_,
                with_payload=True,
                score_threshold=0.1,
            )
        
        search_results = []
        for point in results.points:
            payload = point.payload or {}
            search_results.append(SearchResult(
                id=str(point.id),
                score=point.score,
                texto=payload.get("texto", payload.get("text", "")),
                ref=payload.get("ref"),
                origen=payload.get("origen"),
                jurisdiccion=payload.get("jurisdiccion"),
                entidad=payload.get("entidad"),
                silo=collection,
            ))
        
        return search_results
    
    except Exception as e:
        print(f"⚠️ Error en búsqueda sobre {collection}: {e}")
        return []


async def hybrid_search_all_silos(
    query: str,
    estado: Optional[str],
    top_k: int,
    alpha: float = 0.7,
) -> List[SearchResult]:
    """
    Ejecuta búsqueda híbrida paralela en todos los silos relevantes.
    Aplica filtros de jurisdicción y fusiona resultados.
    """
    # Expandir query para mejorar recall de BM25 Y semántico
    expanded_query = expand_legal_query(query)
    
    # Generar embeddings: AMBOS usan query expandido para consistencia
    dense_task = get_dense_embedding(expanded_query)  # Expandido para mejor comprensión semántica
    sparse_vector = get_sparse_embedding(expanded_query)  # Expandido para mejor recall
    dense_vector = await dense_task
    
    # Búsqueda paralela en los 3 silos CON FILTROS ESPECÍFICOS POR SILO
    tasks = []
    for silo_name in SILOS.values():
        # Obtener filtro específico para este silo
        silo_filter = get_filter_for_silo(silo_name, estado)
        tasks.append(
            hybrid_search_single_silo(
                collection=silo_name,
                query=query,
                dense_vector=dense_vector,
                sparse_vector=sparse_vector,
                filter_=silo_filter,
                top_k=top_k,
                alpha=alpha,
            )
        )

    
    all_results = await asyncio.gather(*tasks)
    
    # Separar resultados por silo para garantizar representación balanceada
    federales = []
    estatales = []
    jurisprudencia = []
    constitucional = []  # Nuevo silo: Constitución, Tratados DDHH, Jurisprudencia CoIDH
    
    for results in all_results:
        for r in results:
            if r.silo == "leyes_federales":
                federales.append(r)
            elif r.silo == "leyes_estatales":
                estatales.append(r)
            elif r.silo == "jurisprudencia_nacional":
                jurisprudencia.append(r)
            elif r.silo == "bloque_constitucional":
                constitucional.append(r)
    
    # Ordenar cada grupo por score
    federales.sort(key=lambda x: x.score, reverse=True)
    estatales.sort(key=lambda x: x.score, reverse=True)
    jurisprudencia.sort(key=lambda x: x.score, reverse=True)
    constitucional.sort(key=lambda x: x.score, reverse=True)
    
    # Fusión balanceada DINÁMICA según tipo de query
    # Para queries de DDHH, priorizar agresivamente el bloque constitucional
    if is_ddhh_query(query):
        # Modo DDHH: Prioridad máxima a bloque constitucional
        min_constitucional = min(15, len(constitucional))  # ALTA prioridad
        min_jurisprudencia = min(5, len(jurisprudencia))   
        min_federales = min(5, len(federales))             
        min_estatales = min(3, len(estatales))             
    else:
        # Modo estándar: Balance entre todos los silos
        min_constitucional = min(8, len(constitucional))   
        min_jurisprudencia = min(7, len(jurisprudencia))   
        min_federales = min(8, len(federales))             
        min_estatales = min(5, len(estatales))             
    
    merged = []
    
    # Primero añadir los mejores de cada categoría garantizada
    # Bloque constitucional primero (mayor jerarquía normativa)
    merged.extend(constitucional[:min_constitucional])
    merged.extend(federales[:min_federales])
    merged.extend(estatales[:min_estatales])
    merged.extend(jurisprudencia[:min_jurisprudencia])
    
    # Llenar el resto con los mejores scores combinados
    already_added = {r.id for r in merged}
    remaining = [r for results in all_results for r in results if r.id not in already_added]
    remaining.sort(key=lambda x: x.score, reverse=True)
    
    slots_remaining = top_k - len(merged)
    if slots_remaining > 0:
        merged.extend(remaining[:slots_remaining])
    
    # Ordenar el resultado final por score para presentación
    merged.sort(key=lambda x: x.score, reverse=True)
    return merged[:top_k]


# ══════════════════════════════════════════════════════════════════════════════
# APP FASTAPI
# ══════════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="Jurexia Core API",
    description="Motor de Producción para Plataforma LegalTech con RAG Híbrido",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS para Next.js frontend (allow all origins for production flexibility)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,  # Must be False when allow_origins=["*"]
    allow_methods=["*"],
    allow_headers=["*"],
)


# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINT: HEALTH CHECK
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/health")
async def health_check():
    """Verifica estado del servicio y conexiones"""
    try:
        # Test Qdrant
        collections = await qdrant_client.get_collections()
        qdrant_status = "connected"
        silos_activos = [c.name for c in collections.collections if c.name in SILOS.values()]
    except Exception as e:
        qdrant_status = f"error: {e}"
        silos_activos = []
    
    return {
        "status": "healthy" if qdrant_status == "connected" else "degraded",
        "qdrant": qdrant_status,
        "silos_activos": silos_activos,
        "sparse_encoder": "Qdrant/bm25",
        "dense_model": EMBEDDING_MODEL,
    }


# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINT: OBTENER DOCUMENTO POR ID
# ══════════════════════════════════════════════════════════════════════════════

class DocumentResponse(BaseModel):
    """Respuesta con documento completo"""
    id: str
    texto: str
    ref: Optional[str] = None
    origen: Optional[str] = None
    jurisdiccion: Optional[str] = None
    entidad: Optional[str] = None
    silo: str
    found: bool = True


@app.get("/document/{doc_id}", response_model=DocumentResponse)
async def get_document(doc_id: str):
    """
    Obtiene el contenido completo de un documento por su ID de Qdrant.
    Busca en todos los silos hasta encontrarlo.
    """
    try:
        # Buscar en cada silo
        for silo_name in SILOS.values():
            try:
                # Intentar obtener el punto por ID
                points = await qdrant_client.retrieve(
                    collection_name=silo_name,
                    ids=[doc_id],
                    with_payload=True,
                )
                
                if points:
                    point = points[0]
                    payload = point.payload or {}
                    
                    return DocumentResponse(
                        id=str(point.id),
                        texto=payload.get("texto", payload.get("text", "Contenido no disponible")),
                        ref=payload.get("ref", payload.get("referencia", None)),
                        origen=payload.get("origen", payload.get("fuente", None)),
                        jurisdiccion=payload.get("jurisdiccion", None),
                        entidad=payload.get("entidad", payload.get("estado", None)),
                        silo=silo_name,
                        found=True,
                    )
            except Exception:
                # ID no encontrado en este silo, continuar
                continue
        
        # No encontrado en ningún silo
        raise HTTPException(
            status_code=404, 
            detail=f"Documento {doc_id} no encontrado en ningún silo"
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al obtener documento: {str(e)}")


# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINT: BÚSQUEDA HÍBRIDA
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/search", response_model=SearchResponse)
async def search_endpoint(request: SearchRequest):
    """
    Búsqueda Híbrida Real (BM25 + Dense).
    
    Estrategia: Prefetch Sparse → Rerank Dense (RRF).
    Filtros MUST para seguridad jurisdiccional.
    """
    try:
        results = await hybrid_search_all_silos(
            query=request.query,
            estado=request.estado,
            top_k=request.top_k,
            alpha=request.alpha,
        )
        
        return SearchResponse(
            query=request.query,
            estado_filtrado=normalize_estado(request.estado),
            resultados=results,
            total=len(results),
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en búsqueda: {str(e)}")


# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINT: CHAT (STREAMING SSE CON VALIDACIÓN DE CITAS)
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """
    Chat conversacional con memoria stateless, streaming SSE y VALIDACIÓN DE CITAS.
    
    NUEVO: Genera respuesta completa, valida Doc IDs contra documentos recuperados,
    anota citas inválidas, y luego hace streaming del resultado validado.
    
    - Recibe historial completo en el body.
    - Ejecuta RAG híbrido sobre la última pregunta.
    - Valida que las citas correspondan a documentos reales.
    - Retorna stream de texto con citas validadas.
    """
    if not request.messages:
        raise HTTPException(status_code=400, detail="Se requiere al menos un mensaje")
    
    # Extraer última pregunta del usuario
    last_user_message = None
    for msg in reversed(request.messages):
        if msg.role == "user":
            last_user_message = msg.content
            break
    
    if not last_user_message:
        raise HTTPException(status_code=400, detail="No se encontró mensaje del usuario")
    
    try:
        # ─────────────────────────────────────────────────────────────────────
        # PASO 1: Búsqueda Híbrida en Qdrant
        # ─────────────────────────────────────────────────────────────────────
        search_results = await hybrid_search_all_silos(
            query=last_user_message,
            estado=request.estado,
            top_k=request.top_k,
        )
        
        # Construir mapa de Doc IDs para validación
        doc_id_map = build_doc_id_map(search_results)
        
        # Inyectar contexto XML
        context_xml = format_results_as_xml(search_results)
        
        # ─────────────────────────────────────────────────────────────────────
        # PASO 2: Construir mensajes para LLM
        # ─────────────────────────────────────────────────────────────────────
        llm_messages = [
            {"role": "system", "content": SYSTEM_PROMPT_CHAT},
            {"role": "system", "content": f"CONTEXTO JURÍDICO RECUPERADO:\n{context_xml}"},
        ]
        
        # Agregar historial conversacional
        for msg in request.messages:
            llm_messages.append({"role": msg.role, "content": msg.content})
        
        # ─────────────────────────────────────────────────────────────────────
        # PASO 3: Generar Respuesta COMPLETA (sin streaming para validar)
        # ─────────────────────────────────────────────────────────────────────
        response = await deepseek_client.chat.completions.create(
            model=CHAT_MODEL,
            messages=llm_messages,
            stream=False,  # Generación completa para validación
            temperature=0.3,
            max_tokens=4000,
        )
        
        full_response = response.choices[0].message.content
        
        # ─────────────────────────────────────────────────────────────────────
        # PASO 4: Validar Citas
        # ─────────────────────────────────────────────────────────────────────
        validation = validate_citations(full_response, doc_id_map)
        
        # Log de validación para debugging
        if validation.invalid_count > 0:
            print(f"⚠️ CITAS INVÁLIDAS: {validation.invalid_count}/{validation.total_citations}")
            for cit in validation.citations:
                if cit.status == "invalid":
                    print(f"   ❌ Doc ID no encontrado: {cit.doc_id}")
        else:
            print(f"✅ Validación OK: {validation.valid_count} citas verificadas")
        
        # ─────────────────────────────────────────────────────────────────────
        # PASO 5: Anotar Citas Inválidas (si hay)
        # ─────────────────────────────────────────────────────────────────────
        if validation.invalid_count > 0:
            invalid_ids = {c.doc_id for c in validation.citations if c.status == "invalid"}
            full_response = annotate_invalid_citations(full_response, invalid_ids)
            
            # Agregar nota al final
            full_response += f"\n\n---\n⚠️ *Nota: {validation.invalid_count} cita(s) no pudieron ser verificadas contra los documentos fuente.*"
        
        # ─────────────────────────────────────────────────────────────────────
        # PASO 6: Streaming Simulado del Resultado Validado
        # ─────────────────────────────────────────────────────────────────────
        async def generate_validated_stream() -> AsyncGenerator[str, None]:
            """Streaming simulado del texto ya validado"""
            # Velocidad de streaming: ~50 caracteres por yield para balance
            chunk_size = 50
            for i in range(0, len(full_response), chunk_size):
                yield full_response[i:i + chunk_size]
                await asyncio.sleep(0.02)  # 20ms de delay para simular streaming
        
        return StreamingResponse(
            generate_validated_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
                "X-Citation-Validation": f"{validation.valid_count}/{validation.total_citations}",
            },
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en chat: {str(e)}")


# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINT: AGENTE CENTINELA (AUDITORÍA)
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/audit", response_model=AuditResponse)
async def audit_endpoint(request: AuditRequest):
    """
    Agente Centinela para auditoría de documentos legales.
    
    WORKFLOW:
    1. LLM extrae Puntos Controvertidos del documento.
    2. Búsquedas paralelas en Qdrant por cada punto.
    3. Consolidación de evidencia.
    4. LLM audita documento vs evidencia.
    5. Retorna JSON estructurado.
    """
    try:
        # ─────────────────────────────────────────────────────────────────────
        # PASO 1: Extraer Puntos Controvertidos
        # ─────────────────────────────────────────────────────────────────────
        extraction_prompt = f"""Analiza el siguiente documento legal y extrae una lista de máximo 5 "Puntos Controvertidos" (los temas jurídicos clave que requieren fundamentación).

DOCUMENTO:
{request.documento[:8000]}

Responde SOLO con un JSON array de strings:
["punto 1", "punto 2", ...]
"""
        
        extraction_response = await deepseek_client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[{"role": "user", "content": extraction_prompt}],
            temperature=0.2,
            max_tokens=500,
        )
        
        try:
            puntos_text = extraction_response.choices[0].message.content
            # Limpiar markdown si existe
            puntos_text = puntos_text.strip()
            if puntos_text.startswith("```"):
                puntos_text = puntos_text.split("```")[1]
                if puntos_text.startswith("json"):
                    puntos_text = puntos_text[4:]
            puntos_controvertidos = json.loads(puntos_text)
        except json.JSONDecodeError:
            puntos_controvertidos = ["Análisis general del documento"]
        
        # ─────────────────────────────────────────────────────────────────────
        # PASO 2: Búsquedas Paralelas por Punto
        # ─────────────────────────────────────────────────────────────────────
        top_k_per_punto = 5 if request.profundidad == "rapida" else 10
        
        search_tasks = []
        for punto in puntos_controvertidos[:5]:  # Máximo 5 puntos
            search_tasks.append(
                hybrid_search_all_silos(
                    query=punto,
                    estado=request.estado,
                    top_k=top_k_per_punto,
                )
            )
        
        all_evidence = await asyncio.gather(*search_tasks)
        
        # ─────────────────────────────────────────────────────────────────────
        # PASO 3: Consolidar Evidencia
        # ─────────────────────────────────────────────────────────────────────
        seen_ids = set()
        consolidated_results = []
        for evidence_list in all_evidence:
            for result in evidence_list:
                if result.id not in seen_ids:
                    seen_ids.add(result.id)
                    consolidated_results.append(result)
        
        # Ordenar por score y limitar
        consolidated_results.sort(key=lambda x: x.score, reverse=True)
        consolidated_results = consolidated_results[:30]
        
        evidence_xml = format_results_as_xml(consolidated_results)
        
        # ─────────────────────────────────────────────────────────────────────
        # PASO 4: Auditoría por LLM
        # ─────────────────────────────────────────────────────────────────────
        audit_prompt = f"""DOCUMENTO A AUDITAR:
{request.documento[:6000]}

PUNTOS CONTROVERTIDOS IDENTIFICADOS:
{json.dumps(puntos_controvertidos, ensure_ascii=False, indent=2)}

EVIDENCIA JURÍDICA:
{evidence_xml}

Realiza la auditoría siguiendo las instrucciones del sistema."""
        
        audit_response = await deepseek_client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_AUDIT},
                {"role": "user", "content": audit_prompt},
            ],
            temperature=0.2,
            max_tokens=3000,
        )
        
        # Parsear respuesta JSON
        audit_text = audit_response.choices[0].message.content
        try:
            audit_data = json.loads(audit_text)
        except json.JSONDecodeError:
            # Fallback si falla el parsing
            audit_data = {
                "puntos_controvertidos": puntos_controvertidos,
                "fortalezas": [],
                "debilidades": [],
                "sugerencias": [],
                "riesgo_general": "INDETERMINADO",
                "resumen_ejecutivo": audit_text[:500],
            }
        
        return AuditResponse(
            puntos_controvertidos=audit_data.get("puntos_controvertidos", puntos_controvertidos),
            fortalezas=audit_data.get("fortalezas", []),
            debilidades=audit_data.get("debilidades", []),
            sugerencias=audit_data.get("sugerencias", []),
            riesgo_general=audit_data.get("riesgo_general", "INDETERMINADO"),
            resumen_ejecutivo=audit_data.get("resumen_ejecutivo", "Análisis completado"),
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en auditoría: {str(e)}")


# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINT: MEJORAR TEXTO LEGAL
# ══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT_ENHANCE = """Eres JUREXIA, un experto redactor jurídico especializado en mejorar documentos legales mexicanos.

Tu tarea es MEJORAR el texto legal proporcionado, integrando fundamentos normativos y jurisprudenciales de los documentos de contexto.

REGLAS DE MEJORA:
1. MANTÉN la estructura y esencia del documento original
2. INTEGRA citas de artículos relevantes usando formato: [Doc ID: uuid]
3. REFUERZA argumentos con jurisprudencia cuando sea aplicable
4. MEJORA la redacción manteniendo formalidad jurídica
5. CORRIGE errores ortográficos o de sintaxis
6. AÑADE fundamentación normativa donde haga falta

FORMATO DE CITAS:
- Para artículos: "...conforme al artículo X del [Ordenamiento] [Doc ID: uuid]..."
- Para jurisprudencia: "...como lo ha sostenido la [Tesis/Jurisprudencia] [Doc ID: uuid]..."

TIPO DE DOCUMENTO: {doc_type}

DOCUMENTOS DE REFERENCIA (usa sus IDs para citar):
{context}

Responde ÚNICAMENTE con el texto mejorado, sin explicaciones adicionales.
"""

class EnhanceRequest(BaseModel):
    """Request para mejorar texto legal"""
    texto: str = Field(..., min_length=50, max_length=50000, description="Texto legal a mejorar")
    tipo_documento: str = Field(default="demanda", description="Tipo: demanda, amparo, impugnacion, contestacion, contrato, otro")
    estado: Optional[str] = Field(default=None, description="Estado para filtrar legislación estatal")


class EnhanceResponse(BaseModel):
    """Response con texto mejorado"""
    texto_mejorado: str
    documentos_usados: int
    tokens_usados: int


@app.post("/enhance", response_model=EnhanceResponse)
async def enhance_legal_text(request: EnhanceRequest):
    """
    Mejora texto legal usando RAG.
    Busca artículos y jurisprudencia relevantes e integra citas en el texto.
    """
    try:
        # Normalizar estado si viene
        estado_norm = normalize_estado(request.estado)
        
        # Buscar documentos relevantes basados en el texto
        # Extraer conceptos clave del texto para búsqueda
        search_query = request.texto[:1000]  # Primeros 1000 chars para embedding
        
        search_results = await hybrid_search_all_silos(
            query=search_query,
            estado=estado_norm,
            top_k=15,  # Menos documentos para enhance, más enfocados
            alpha=0.7,
        )
        
        if not search_results:
            # Retornar texto sin cambios si no hay contexto
            return EnhanceResponse(
                texto_mejorado=request.texto,
                documentos_usados=0,
                tokens_usados=0,
            )
        
        # Construir contexto XML
        context_parts = []
        for result in search_results:
            context_parts.append(
                f'<documento id="{result.id}" silo="{result.silo}" ref="{result.ref or "N/A"}">\n'
                f'{result.texto[:800]}\n'
                f'</documento>'
            )
        context_xml = "\n\n".join(context_parts)
        
        # Mapear tipo de documento a descripción
        doc_type_map = {
            "demanda": "DEMANDA JUDICIAL",
            "amparo": "DEMANDA DE AMPARO",
            "impugnacion": "RECURSO DE IMPUGNACIÓN",
            "contestacion": "CONTESTACIÓN DE DEMANDA",
            "contrato": "CONTRATO",
            "otro": "DOCUMENTO LEGAL",
        }
        doc_type_desc = doc_type_map.get(request.tipo_documento, "DOCUMENTO LEGAL")
        
        # Construir prompt
        system_prompt = SYSTEM_PROMPT_ENHANCE.format(
            doc_type=doc_type_desc,
            context=context_xml,
        )
        
        # Llamar a DeepSeek
        response = await deepseek_client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Mejora el siguiente texto legal:\n\n{request.texto}"},
            ],
            temperature=0.3,  # Más conservador para mantener fidelidad
            max_tokens=8000,
        )
        
        enhanced_text = response.choices[0].message.content
        tokens_used = response.usage.total_tokens if response.usage else 0
        
        return EnhanceResponse(
            texto_mejorado=enhanced_text,
            documentos_usados=len(search_results),
            tokens_usados=tokens_used,
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al mejorar texto: {str(e)}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    
    print("═" * 60)
    print("  JUREXIA CORE API - Motor de Producción")
    print("═" * 60)
    
    uvicorn.run(
        "api_jurexia_core:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
