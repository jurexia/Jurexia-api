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
import uuid
from typing import AsyncGenerator, List, Literal, Optional
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
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Silos V4 de Jurexia
SILOS = {
    "federal": "leyes_federales",
    "estatal": "leyes_estatales",
    "jurisprudencia": "jurisprudencia_nacional",
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

SYSTEM_PROMPT_CHAT = """Eres un Consultor Jurídico Élite especializado en Derecho Mexicano.

INSTRUCCIONES CRÍTICAS DE CITAS:
1. SIEMPRE fundamenta tus respuestas usando EXCLUSIVAMENTE los documentos proporcionados en las etiquetas <documento>.
2. Al citar, usa el formato: [Doc ID: X] donde X es el id del documento XML.
3. Si no encuentras información relevante en los documentos, indícalo expresamente.
4. Sigue el razonamiento jurídico: Constitución → Leyes Secundarias → Jurisprudencia → Aplicación.

ESTRUCTURA DE RESPUESTA:
1. **Fundamento Normativo**: Cita las normas aplicables con sus IDs.
2. **Interpretación Jurisprudencial**: Cita tesis relevantes con sus IDs.
3. **Conclusión Jurídica**: Síntesis aplicada al caso concreto.
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
    top_k: int = Field(10, ge=1, le=30)


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


# ══════════════════════════════════════════════════════════════════════════════
# CLIENTES GLOBALES (Lifecycle)
# ══════════════════════════════════════════════════════════════════════════════

sparse_encoder: SparseTextEmbedding = None
qdrant_client: AsyncQdrantClient = None
openai_client: AsyncOpenAI = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Inicialización y cleanup de recursos"""
    global sparse_encoder, qdrant_client, openai_client
    
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
    
    # OpenAI Client
    openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
    print("  ✓ OpenAI Client inicializado")
    
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


def build_jurisdiction_filter(estado: Optional[str]) -> Optional[Filter]:
    """
    Construye filtro MUST para seguridad jurisdiccional.
    REGLA: Si selecciona un estado, SOLO trae ese estado + Federal/Jurisprudencia.
    """
    if not estado:
        return None
    
    normalized = normalize_estado(estado)
    if not normalized:
        return None
    
    # Solo permite resultados del estado específico O federales/jurisprudencia
    return Filter(
        should=[
            # Leyes del estado específico
            Filter(
                must=[
                    FieldCondition(key="jurisdiccion", match=MatchValue(value="ESTATAL")),
                    FieldCondition(key="entidad", match=MatchValue(value=normalized)),
                ]
            ),
            # Leyes federales (siempre aplicables)
            FieldCondition(key="jurisdiccion", match=MatchValue(value="FEDERAL")),
            # Jurisprudencia nacional (siempre aplicable)
            FieldCondition(key="silo", match=MatchValue(value="jurisprudencia_nacional")),
        ]
    )


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


def format_results_as_xml(results: List[SearchResult]) -> str:
    """
    Formatea resultados en XML para inyección de contexto.
    Escapa caracteres HTML para seguridad.
    """
    if not results:
        return "<documentos>Sin resultados relevantes encontrados.</documentos>"
    
    xml_parts = ["<documentos>"]
    for r in results:
        escaped_texto = html.escape(r.texto)
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
                with_payload=True,
                score_threshold=0.3,
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
                score_threshold=0.3,
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
    # Generar embeddings
    dense_task = get_dense_embedding(query)
    sparse_vector = get_sparse_embedding(query)
    dense_vector = await dense_task
    
    # Construir filtro jurisdiccional
    filter_ = build_jurisdiction_filter(estado)
    
    # Búsqueda paralela en los 3 silos
    tasks = []
    for silo_name in SILOS.values():
        tasks.append(
            hybrid_search_single_silo(
                collection=silo_name,
                query=query,
                dense_vector=dense_vector,
                sparse_vector=sparse_vector,
                filter_=filter_,
                top_k=top_k,
                alpha=alpha,
            )
        )
    
    all_results = await asyncio.gather(*tasks)
    
    # Fusionar y ordenar por score (Reciprocal Rank Fusion simplificado)
    merged = []
    for results in all_results:
        merged.extend(results)
    
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

# CORS para Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3001",
        "https://jurexiagtp.com",
        "https://www.jurexiagtp.com",
        "https://v0-jurexia-dashboard-design.vercel.app",
        "https://jurexia-frontend.vercel.app",
    ],
    allow_credentials=True,
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
# ENDPOINT: CHAT (STREAMING SSE)
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """
    Chat conversacional con memoria stateless y streaming SSE.
    
    - Recibe historial completo en el body.
    - Ejecuta RAG híbrido sobre la última pregunta.
    - Retorna stream de texto letra por letra.
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
        # Ejecutar búsqueda híbrida
        search_results = await hybrid_search_all_silos(
            query=last_user_message,
            estado=request.estado,
            top_k=request.top_k,
        )
        
        # Inyectar contexto XML
        context_xml = format_results_as_xml(search_results)
        
        # Construir mensajes para LLM
        llm_messages = [
            {"role": "system", "content": SYSTEM_PROMPT_CHAT},
            {"role": "system", "content": f"CONTEXTO JURÍDICO RECUPERADO:\n{context_xml}"},
        ]
        
        # Agregar historial conversacional
        for msg in request.messages:
            llm_messages.append({"role": msg.role, "content": msg.content})
        
        async def generate_stream() -> AsyncGenerator[str, None]:
            """Generador de streaming SSE"""
            try:
                stream = await openai_client.chat.completions.create(
                    model="gpt-4o",
                    messages=llm_messages,
                    stream=True,
                    temperature=0.3,
                    max_tokens=4000,
                )
                
                async for chunk in stream:
                    if chunk.choices and chunk.choices[0].delta.content:
                        yield chunk.choices[0].delta.content
                        
            except Exception as e:
                yield f"\n\n[Error en generación: {str(e)}]"
        
        return StreamingResponse(
            generate_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
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
        
        extraction_response = await openai_client.chat.completions.create(
            model="gpt-4o",
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
        
        audit_response = await openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_AUDIT},
                {"role": "user", "content": audit_prompt},
            ],
            temperature=0.2,
            max_tokens=3000,
            response_format={"type": "json_object"},
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
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    
    print("═" * 60)
    print("  JUREXIA CORE API - Motor de Producción")
    print("═" * 60)
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=False,
        log_level="info",
    )

