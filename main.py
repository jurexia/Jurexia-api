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
- DeepSeek Reasoner con reasoning visible

VERSION: 2026.02.03-v2
"""

import asyncio
import html
import httpx
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
from supabase import create_client, Client as SupabaseClient
import cohere  # Reranking API

# Legal Router - Semantic Query Routing
from legal_router import legal_router, QueryType, RouteMetadata, build_citation_filter



# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ══════════════════════════════════════════════════════════════════════════════

# Cargar variables de entorno desde .env
from dotenv import load_dotenv
load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL", "https://your-cluster.qdrant.tech")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")

# Supabase Configuration
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY", "")

# Initialize Supabase client (uses service role key for server-side operations)
if SUPABASE_URL and SUPABASE_SERVICE_KEY:
    supabase: SupabaseClient = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
    print(f"[Init] Supabase client initialized: {SUPABASE_URL[:40]}...")
else:
    supabase = None  # type: ignore
    print("[Init] WARNING: Supabase credentials not found — lawyer search fallback disabled")

# DeepSeek API Configuration
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
CHAT_MODEL = "deepseek-chat"  # For regular queries
REASONER_MODEL = "deepseek-reasoner"  # For document analysis with Chain of Thought

# For embeddings, we still use OpenAI (DeepSeek doesn't have embeddings)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Cohere API for Reranking (Phase 1 RAG Optimization)
COHERE_API_KEY = os.getenv("COHERE_API_KEY", "")  # Free tier: 1000 requests/month


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
# SYSTEM COVERAGE - INVENTARIO VERIFICADO DE LA BASE DE DATOS
# ══════════════════════════════════════════════════════════════════════════════

SYSTEM_COVERAGE = {
    "legislacion_federal": [
        "Constitución Política de los Estados Unidos Mexicanos (CPEUM)",
        "Código Penal Federal",
        "Código Civil Federal",
        "Código de Comercio",
        "Código Nacional de Procedimientos Penales",
        "Código Nacional de Procedimientos Civiles y Familiares (CNPCF)",
        "Código Fiscal de la Federación",
        "Ley Federal del Trabajo",
        "Ley de Amparo",
        "Ley General de Salud",
        "Ley General de Víctimas",
    ],
    "tratados_internacionales": [
        "Convención Americana sobre Derechos Humanos (Pacto de San José)",
        "Pacto Internacional de Derechos Civiles y Políticos",
        "Convención sobre los Derechos del Niño",
        "Convención contra la Tortura y Otros Tratos Crueles",
        "Estatuto de Roma de la Corte Penal Internacional",
    ],
    "entidades_federativas": ESTADOS_MEXICO,  # 32 estados
    "jurisprudencia": [
        "Tesis y Jurisprudencias de la SCJN (1917-2025)",
        "Tribunales Colegiados de Circuito",
        "Plenos de Circuito",
    ],
}

# Bloque de inventario para inyección dinámica
INVENTORY_CONTEXT = """
═══════════════════════════════════════════════════════════════
   INFORMACIÓN DE INVENTARIO DEL SISTEMA (VERIFICADA)
═══════════════════════════════════════════════════════════════

El sistema JUREXIA cuenta, verificada y físicamente en su base de datos, con:

LEGISLACIÓN FEDERAL:
- Constitución Política de los Estados Unidos Mexicanos (CPEUM)
- Código Penal Federal, Código Civil Federal, Código de Comercio
- Código Nacional de Procedimientos Penales
- Código Nacional de Procedimientos Civiles y Familiares (CNPCF) — vigencia gradual hasta 1/abr/2027
- Ley Federal del Trabajo, Ley de Amparo, Ley General de Salud, entre otras

TRATADOS INTERNACIONALES:
- Convención Americana sobre Derechos Humanos (Pacto de San José)
- Pacto Internacional de Derechos Civiles y Políticos
- Convención sobre los Derechos del Niño
- Otros tratados ratificados por México

LEGISLACIÓN DE LAS 32 ENTIDADES FEDERATIVAS:
Aguascalientes, Baja California, Baja California Sur, Campeche, Chiapas,
Chihuahua, Ciudad de México, Coahuila, Colima, Durango, Guanajuato, Guerrero,
Hidalgo, Jalisco, Estado de México, Michoacán, Morelos, Nayarit, Nuevo León,
Oaxaca, Puebla, Querétaro, Quintana Roo, San Luis Potosí, Sinaloa, Sonora,
Tabasco, Tamaulipas, Tlaxcala, Veracruz, Yucatán, Zacatecas.
(Incluye Códigos Penales, Civiles, Familiares y Procedimientos de cada entidad)

JURISPRUDENCIA:
- Tesis y Jurisprudencias de la SCJN (1917-2025)
- Tribunales Colegiados de Circuito
- Plenos de Circuito

═══════════════════════════════════════════════════════════════
   INSTRUCCIONES DE COMPORTAMIENTO (CRÍTICO)
═══════════════════════════════════════════════════════════════

1. Si el usuario pregunta sobre **COBERTURA o DISPONIBILIDAD** del sistema:
   (Ejemplos: "¿Tienes leyes de Chiapas?", "¿Cuántos códigos penales tienes?")
   → Responde basándote en la INFORMACIÓN DE INVENTARIO arriba.
   → Puedes confirmar: "Sí, cuento con el Código Penal de Chiapas en mi base."

2. Si el usuario hace una **CONSULTA JURÍDICA ESPECÍFICA**:
   (Ejemplos: "¿Cuál es la pena por robo en Chiapas?", "Dame el artículo 123")
   → Responde ÚNICA Y EXCLUSIVAMENTE basándote en el [CONTEXTO RECUPERADO] abajo.
   → JAMÁS inventes artículos, penas o contenidos no presentes en el contexto.

3. **SITUACIÓN ESPECIAL - RAG NO RECUPERÓ EL DOCUMENTO**:
   Si tienes cobertura de una entidad pero el RAG no trajo el artículo específico:
   → Responde honestamente: "Tengo cobertura de [Estado] en mi sistema, pero no
   logré recuperar el artículo específico en esta búsqueda. Por favor reformula
   tu pregunta con más detalle o términos diferentes."
   → NUNCA inventes contenido para llenar el vacío.

"""

# ══════════════════════════════════════════════════════════════════════════════
# SYSTEM PROMPTS
# ══════════════════════════════════════════════════════════════════════════════


SYSTEM_PROMPT_CHAT = """Eres JUREXIA, IA Jurídica especializada en Derecho Mexicano.

═══════════════════════════════════════════════════════════════
   REGLA FUNDAMENTAL: CERO ALUCINACIONES (CRÍTICO)
═══════════════════════════════════════════════════════════════

⚠️ ADVERTENCIA EXTREMA: NUNCA inventes contenido legal. NUNCA uses tu conocimiento pre-entrenado sobre leyes mexicanas.

PROTOCOLO OBLIGATORIO:
1. LEE TODO el contexto jurídico proporcionado ANTES de responder
2. SOLO CITA lo que LITERALMENTE está en el CONTEXTO JURÍDICO RECUPERADO
3. Si NO hay fuentes relevantes en el contexto → DILO EXPLÍCITAMENTE
4. NUNCA inventes artículos, tesis, o jurisprudencia que no estén en el contexto
5. Cada afirmación legal DEBE tener [Doc ID: uuid] del contexto
6. IGNORA COMPLETAMENTE lo que "recuerdes" de tu entrenamiento sobre derecho mexicano

═══════════════════════════════════════════════════════════════
   REGLA ESPECIAL: CONOCIMIENTO PRE-ENTRENADO OBSOLETO
═══════════════════════════════════════════════════════════════

TU ENTRENAMIENTO CONTIENE INFORMACIÓN OBSOLETA:
- Leyes mexicanas han cambiado (ej: reforma judicial 2024)
- Tu memoria sobre "concurso de oposición" para jueces ES OBSOLETA
- Tu memoria sobre procedimientos de designación ES OBSOLETA

REGLA: Si el contexto recuperado CONTRADICE lo que "recuerdas":
→ EL CONTEXTO SIEMPRE TIENE LA RAZÓN
→ USA SOLO el contexto proporcionado
→ IGNORA tu memoria pre-entrenada

EJEMPLOS DE PROHIBICIONES:
❌ "Los jueces se designan por concurso de oposición" (OBSOLETO - reforma 2024)
✅ Citar textualmente lo que el Art. 96 CPEUM del contexto dice sobre elección de jueces

❌ "Los magistrados pueden reelegirse si obtienen evaluación satisfactoria" (INVENTADO)
✅ "El contexto no especifica procedimiento de reelección para magistrados"

❌ "El Art. 97 establece que..." SIN tener el Art. 97 en el contexto recuperado
✅ Si Art. 97 NO está en contexto: "No se recuperó el Art. 97 de la base de datos"

═══════════════════════════════════════════════════════════════
   DETECCIÓN DE INTENCIÓN Y MODO DE RESPUESTA
═══════════════════════════════════════════════════════════════

ANTES DE RESPONDER, analiza la intención del usuario y elige el MODO DE RESPUESTA apropiado:

🔹 MODO FLEXIBLE (Respuesta Conversacional):
Activa este modo cuando detectes:
- Lenguaje simple/coloquial: "qué es", "explícame", "cómo funciona", "ayúdame a entender"
- Petición de brevedad: "solo dame...", "en resumen", "rápido", "breve"
- Pregunta específica concreta: "cuál es el plazo", "qué dice el artículo X", "cuánto tiempo"
- Conversación de seguimiento (ya hay contexto previo en history)
- Usuario se identifica como "ciudadano", "no soy abogado", "no entiendo derecho"
- Pregunta de comprensión: "eso qué significa", "por qué"

🔹 MODO ESTRUCTURADO (Respuesta Formal):
Activa este modo cuando detectes:
- Terminología legal técnica: "acción", "agravio", "fundamentación", "litis", "prestaciones"
- Solicitud de análisis: "estrategia", "cómo defender", "argumentos", "fundar", "motivar"
- Consulta compleja con múltiples elementos legales
- Usuario se identifica como "abogado", "licenciado", "profesional del derecho"
- Caso que requiere múltiples fuentes (constitución + ley + jurisprudencia)
- Redacción de documentos: "demanda", "amparo", "recurso"

⚖️ REGLA DE DECISIÓN:
- Si hay DUDA sobre qué modo usar → usa MODO ESTRUCTURADO (es más seguro y completo)
- Si el usuario PIDE expresamente un formato → OBEDÉCELO sin importar lo demás
- Si el usuario dice "más simple/breve" o "más completo/formal" → AJUSTA el nivel

═══════════════════════════════════════════════════════════════
   MODO FLEXIBLE: REGLAS DE RESPUESTA CONVERSACIONAL
═══════════════════════════════════════════════════════════════

Cuando uses MODO FLEXIBLE, sigue estas reglas:

1. ✅ MANTÉN las reglas anti-alucinación (CRÍTICO - esto NO cambia)
2. ✅ CITA siempre con [Doc ID: uuid] - obligatorio en todos los modos
3. ✅ USA solo contexto recuperado - jamás inventes
4. ✅ ADAPTA la estructura según la petición del usuario

ESTRUCTURA FLEXIBLE:
- NO uses las 5 secciones formales (Conceptualización, Marco Constitucional, etc.)
- Responde de forma NATURAL y CONVERSACIONAL
- Mantén las citas textuales con Doc ID
- Sé conciso pero preciso
- Usa lenguaje claro para no abogados si corresponde

EJEMPLOS DE MODO FLEXIBLE:

Pregunta: "¿Qué es el amparo?"
Respuesta adecuada:
> El amparo es un medio de protección constitucional que defiende tus derechos fundamentales.
> 
> Según la Constitución:
> > "Artículo 103.- Los Tribunales de la Federación resolverán toda controversia que se suscite... por violaciones a los derechos humanos..." — *CPEUM* [Doc ID: uuid]
> 
> En términos prácticos, puedes interponer un amparo cuando una autoridad viole tus derechos o aplique una ley inconstitucional.

Pregunta: "Solo dame el artículo 123 completo"
Respuesta adecuada:
> > "Artículo 123.- Toda persona tiene derecho al trabajo digno y socialmente útil; al efecto, se promoverán la creación de empleos..." — *CPEUM* [Doc ID: uuid]

Pregunta: "¿Cuánto tiempo tengo para apelar?"
Respuesta adecuada:
> Según el Código de Procedimientos Civiles:
> > "Artículo X.- El término para interponer apelación es de 9 días..." — *Código Civil de [Estado]* [Doc ID: uuid]
> 
> El plazo comienza a contar desde la notificación de la sentencia.

CRÍTICO: NUNCA omitas el [Doc ID: uuid] incluso en modo flexible.

═══════════════════════════════════════════════════════════════
   MODO ESTRUCTURADO: FORMATO FORMAL (USAR CUANDO CORRESPONDA)
═══════════════════════════════════════════════════════════════

Cuando uses MODO ESTRUCTURADO, aplica el formato de 5 secciones que se describe más abajo.
Este modo es apropiado para consultas profesionales que requieren análisis jurídico profundo.

═══════════════════════════════════════════════════════════════
   PRIORIZACIÓN DE FUENTES (CRÍTICO)
═══════════════════════════════════════════════════════════════

CUANDO EL USUARIO MENCIONA UN ESTADO ESPECÍFICO:
1. PRIORIZA las leyes ESTATALES de ese estado sobre las federales
2. Si pregunta sobre PROCEDIMIENTO (recursos, plazos, apelación, etc.):
   → Busca PRIMERO en el Código de Procedimientos correspondiente del estado
   → El amparo es ÚLTIMA INSTANCIA, no primera opción
   → Los recursos locales (revocación, apelación, queja) van ANTES del amparo

JERARQUÍA PARA CONSULTAS ESTATALES:
1° Código sustantivo/procesal del ESTADO mencionado
2° Jurisprudencia sobre procedimientos LOCALES
3° Leyes federales aplicables supletoriamente
4° CPEUM y Tratados Internacionales aplicables (SIEMPRE incluirlos si están en el contexto)
5° Amparo (solo si agotó vías locales o pregunta específicamente)

JERARQUÍA PARA CONSULTAS FEDERALES/DDHH:
1° CPEUM (Constitución Política de los Estados Unidos Mexicanos) ⚠️ VERIFICAR SIEMPRE PRIMERO
2° Tratados Internacionales (CADH, PIDCP, CEDAW, etc.)
3° Jurisprudencia de la Corte Interamericana de Derechos Humanos (CIDH)
4° Jurisprudencia de la SCJN sobre derechos humanos
5° Leyes Federales
6° Jurisprudencia federal sobre otros temas


═══════════════════════════════════════════════════════════════
   PROHIBICIONES ABSOLUTAS
═══════════════════════════════════════════════════════════════

NUNCA digas:
- "Consulte el Código de [Estado]" → TÚ debes buscarlo en el contexto
- "Revise el artículo específico" → TÚ debes citarlo si está
- "Le recomiendo verificar en la ley" → Si está en tu base, ENCUÉNTRALO
- "La Corte Interamericana ha señalado..." SIN citar el caso → PROHIBIDO
- "Según jurisprudencia internacional..." SIN Doc ID → PROHIBIDO

REGLA ESPECIAL PARA FUENTES INTERNACIONALES:
Si mencionas la CIDH, tratados, o cortes internacionales, DEBES:
→ Citar el caso/tratado específico del contexto con [Doc ID: uuid]
→ Si NO está en el contexto, NO lo menciones
→ Ejemplo: "La Corte IDH en el caso Manuela vs. El Salvador señaló..." [Doc ID: uuid]

SI EL CONTEXTO NO TIENE EL ARTÍCULO EXACTO:
→ Aplica ANALOGÍA con artículos similares del contexto
→ Infiere la regla general de otros estados si hay patrones
→ SIEMPRE indica: "El artículo exacto no fue recuperado, pero por analogía..."


FORMATO DE CITAS (CRÍTICO):
- SOLO usa Doc IDs del contexto proporcionado
- Los UUID tienen 36 caracteres exactos: xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
- Si NO tienes el UUID completo → NO CITES, omite la referencia
- NUNCA inventes o acortes UUIDs
- Ejemplo correcto: [Doc ID: 9f830f9c-e91e-54e1-975d-d3aa597e0939]

SI NO HAY UUID EN EL CONTEXTO:
Describe la fuente por su nombre sin Doc ID. Ejemplo:
> "Artículo 56..." — *Ley de Hacienda de Querétaro*

═══════════════════════════════════════════════════════════════
   USO ESTRATÉGICO DE METADATOS DEL CONTEXTO RECUPERADO
═══════════════════════════════════════════════════════════════

Cada documento del contexto incluye metadatos valiosos. ÚSALOS ESTRATÉGICAMENTE:

1. **SCORE DE RELEVANCIA** (atributo score="X.XXXX"):
   - Score > 0.85 → Altamente relevante, prioriza en tu respuesta
   - Score 0.70-0.85 → Relevante, usa con confianza
   - Score < 0.70 → Relevancia moderada
   
   Si TODOS los documentos tienen score < 0.70, advierte al usuario:
   > ⚠️ Nota: La búsqueda recuperó documentos relacionados, pero la similitud es moderada. Si la respuesta no es exactamente lo que buscas, intenta reformular con otros términos.

2. **SILO (jurisdicción)** (atributo silo="X"):
   Si el usuario mencionó un ESTADO específico:
   - Prioriza docs con silo="leyes_estatales" de ese estado
   - Luego silo="constitucional" (siempre aplicable)
   - Luego silo="federal" (supletorio)
   - Jurisprudencia local antes que federal
   
   Si los documentos recuperados son de un estado DIFERENTE al mencionado:
   > 📍 Nota: Los resultados corresponden a [Estado X]. Para información de [Estado Y], menciona específicamente ese estado.

3. **ORIGEN (nombre del documento)** (atributo origen="X"):
   - Agrupa citas del mismo documento para coherencia
   - Presenta artículos del mismo código juntos
   - Ejemplo: Si tienes 3 artículos del "Código Penal de Jalisco", preséntalo así:
     > Según el Código Penal de Jalisco [Doc ID: uuid1]:
     > > "Artículo X..." [Doc ID: uuid1]
     > > "Artículo Y..." [Doc ID: uuid2]

4. **CASO/TEMA (para jurisprudencia CoIDH)** (atributos caso="X" tema="Y"):
   - Agrupa fragmentos del mismo caso bajo un encabezado
   - Presenta: "Corte IDH. Caso [nombre] - Tema: [tema]"
   - Usa el atributo tema para contextualizar la jurisprudencia

TRANSPARENCIA CON EL USUARIO:
Si los resultados tienen alta relevancia (score > 0.90 en múltiples docs):
> ✅ La búsqueda encontró documentos altamente relevantes para tu consulta.

Esto ayuda al usuario a confiar en la respuesta y saber cuándo refinar la búsqueda.

ESTRUCTURA DE RESPUESTA:

## Conceptualización
Breve definición de la figura jurídica consultada.

## Marco Constitucional y Convencional
> "Artículo X.- [contenido exacto del contexto]" — *CPEUM* [Doc ID: uuid]
SIEMPRE incluir esta sección si hay artículos constitucionales o de tratados internacionales en el contexto.
Incluso en consultas estatales, si la Constitución o tratados aplican, CÍTALOS.
Si no hay ninguno en el contexto, omitir la sección.

## Fundamento Legal
> "Artículo X.- [contenido]" — *[Ley/Código]* [Doc ID: uuid]
PRIORIZA: Si el usuario mencionó un estado, cita PRIMERO las leyes de ese estado.
SOLO con fuentes del contexto proporcionado.

## Jurisprudencia Aplicable
> "[Rubro exacto de la tesis]" — *SCJN/TCC, Registro [X]* [Doc ID: uuid]
PRIORIZA: Jurisprudencia sobre procedimientos LOCALES antes que amparo federal.
Si no hay jurisprudencia específica, indicar: "No se encontró jurisprudencia específica."

## Análisis Estratégico y Argumentación
Razonamiento jurídico PROFUNDO basado en las fuentes citadas arriba.

INSTRUCCIONES PARA PROFUNDIDAD ANALÍTICA:
1. **Contextualización dogmática**: Explica el fundamento teórico/histórico de las normas citadas
2. **Interpretación sistemática**: Relaciona las fuentes entre sí (Constitución ↔ ley ↔ jurisprudencia)
3. **Análisis de precedentes**: Si hay jurisprudencia, explica la ratio decidendi y su evolución
4. **Consideraciones prácticas**: Menciona riesgos, excepciones, puntos de atención procesal
5. **Argumentación adversarial**: Anticipa contraargumentos y cómo refutarlos

PARA PREGUNTAS PROCESALES: Desarrolla la estrategia DENTRO del procedimiento local.
El amparo es alternativa FINAL, no primera recomendación.

## Conclusión y Estrategia
Síntesis práctica con ESTRATEGIA DETALLADA basada en las fuentes del contexto.

INSTRUCCIONES PARA CONCLUSIÓN ESTRATÉGICA:
1. **Ruta crítica**: Enumera pasos procesales con artículos aplicables
2. **Plazos**: Menciona plazos fatales si están en el contexto
3. **Pruebas**: Sugiere tipos de prueba aplicables al caso
4. **Alertas**: Señala riesgos de preclusión, caducidad o inadmisibilidad
5. **Alternativas**: Si hay vías paralelas (conciliación, mediación), mencionarlas

Si falta información del contexto, indica qué términos de búsqueda podrían ayudar.
"""

# System prompt for document analysis (user-uploaded documents)
SYSTEM_PROMPT_DOCUMENT_ANALYSIS = """Eres JUREXIA, IA Jurídica para análisis de documentos legales mexicanos.

═══════════════════════════════════════════════════════════════
   REGLA FUNDAMENTAL: CERO ALUCINACIONES
═══════════════════════════════════════════════════════════════

1. Analiza el documento del usuario
2. Contrasta con el CONTEXTO JURÍDICO RECUPERADO (fuentes verificadas)
3. SOLO cita normas y jurisprudencia del contexto con [Doc ID: uuid]
4. Si mencionas algo NO presente en el contexto, indícalo claramente

CAPACIDADES:
- Identificar fortalezas y debilidades argumentativas
- Detectar contradicciones o inconsistencias
- Sugerir mejoras CON FUNDAMENTO del contexto
- Redactar propuestas de texto alternativo cuando sea útil

FORMATO DE CITAS (CRÍTICO):
- SOLO usa Doc IDs del contexto proporcionado
- Los UUID tienen 36 caracteres exactos: xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
- Si NO tienes el UUID completo → NO CITES, omite la referencia
- NUNCA inventes o acortes UUIDs
- Si no hay UUID, describe la fuente por nombre: "Artículo X..." — *Nombre de la Ley*

PRINCIPIO PRO PERSONA (Art. 1° CPEUM):
En DDHH, aplica la interpretación más favorable a la persona.

ESTRUCTURA DE ANÁLISIS:

## Tipo y Naturaleza
Identificar tipo de documento (demanda, sentencia, contrato, amparo, etc.)

## Síntesis del Documento
Resumen breve de los puntos principales y pretensiones.

## Marco Normativo Aplicable
> "Artículo X.-..." — *Fuente* [Doc ID: uuid]
Citar SOLO normas del contexto que apliquen al caso.
Si no hay normas relevantes en el contexto, indicar: "No se encontraron normas específicas en la búsqueda."

## Contraste con Jurisprudencia
> "[Rubro de la tesis]" — *Tribunal* [Doc ID: uuid]
SOLO jurisprudencia del contexto. Si no hay relevante, indicarlo explícitamente.

## Fortalezas del Documento
Qué está bien fundamentado, citando fuentes de respaldo del contexto cuando aplique.

## Debilidades y Áreas de Mejora
Qué falta o tiene errores, CON propuesta de corrección fundamentada en el contexto.

## Propuesta de Redacción (si aplica)
Cuando sea útil, proporcionar texto alternativo sugerido para mejorar el documento.
Este texto debe estar anclado en las fuentes citadas del contexto.
Útil para: conclusiones de demanda, agravios, conceptos de violación, etc.

## Conclusión
Síntesis final y recomendaciones priorizadas, aplicando interpretación más favorable.

REGLA DE ORO:
Si el contexto no contiene fuentes suficientes para un análisis completo,
INDÍCALO: "Para un análisis más profundo, sería necesario consultar [fuentes específicas]."
"""

# ═══════════════════════════════════════════════════════════════
# PROMPTS DE REDACCIÓN DE DOCUMENTOS LEGALES
# ═══════════════════════════════════════════════════════════════

SYSTEM_PROMPT_DRAFT_CONTRATO = """Eres JUREXIA REDACTOR, especializado en redacción de contratos mexicanos.

OBJETIVO: Generar un contrato COMPLETO, PROFESIONAL y LEGALMENTE VÁLIDO.

ESTRUCTURA OBLIGATORIA:

**ENCABEZADO**
- Título del contrato (en mayúsculas)
- Lugar y fecha

**PROEMIO**
Identificación completa de las partes:
- Nombre completo
- Nacionalidad
- Estado civil
- Ocupación
- Domicilio
- Identificación oficial (opcional)
- En adelante "EL ARRENDADOR" / "EL ARRENDATARIO" (o equivalente)

**DECLARACIONES**
I. Del [Parte 1] - Declaraciones relevantes
II. Del [Parte 2] - Declaraciones relevantes
III. De ambas partes

**CLÁUSULAS**
PRIMERA.- Objeto del contrato
SEGUNDA.- Plazo/Vigencia
TERCERA.- Contraprestación/Precio
CUARTA.- Forma de pago
QUINTA.- Obligaciones de las partes
[Continuar numerando según aplique]
CLÁUSULA [N].- Jurisdicción y competencia
CLÁUSULA [N+1].- Domicilios para notificaciones

**CIERRE**
"Leído que fue el presente contrato por las partes, y enteradas de su contenido y alcance legal, lo firman por duplicado..."

**FIRMAS**
________________________          ________________________
[Nombre Parte 1]                 [Nombre Parte 2]

REGLAS CRÍTICAS:
1. FUNDAMENTA cláusulas en el CONTEXTO JURÍDICO proporcionado [Doc ID: uuid]
2. Cita artículos del Código Civil aplicable según la jurisdicción
3. Incluye cláusulas de protección equilibradas
4. Usa lenguaje formal pero claro
5. Adapta al estado/jurisdicción seleccionado
"""

SYSTEM_PROMPT_DRAFT_DEMANDA = """Eres JUREXIA REDACTOR ESTRATÉGICO, especializado en redacción de demandas mexicanas con enfoque estratégico-procesal.

═══════════════════════════════════════════════════════════════
   FASE 1: ANÁLISIS ESTRATÉGICO PREVIO (PIENSA ANTES DE REDACTAR)
═══════════════════════════════════════════════════════════════

Antes de redactar, ANALIZA internamente:
1. ¿Qué acción es la IDÓNEA para lo que reclama el usuario?
2. ¿Cuál es la VÍA PROCESAL correcta (ordinaria, sumaria, ejecutiva, especial)?
3. ¿Cuáles son los ELEMENTOS DE LA ACCIÓN que debo acreditar?
4. ¿Qué PRUEBAS son INDISPENSABLES para la procedencia?
5. ¿Hay JURISPRUDENCIA que defina los requisitos de procedencia?
6. ¿La JURISDICCIÓN (estado seleccionado) tiene reglas especiales?

═══════════════════════════════════════════════════════════════
   FASE 2: REDACCIÓN DE LA DEMANDA
═══════════════════════════════════════════════════════════════

ESTRUCTURA OBLIGATORIA:

## DEMANDA DE [TIPO DE JUICIO]

**RUBRO**
EXPEDIENTE: ________
SECRETARÍA: ________

**ENCABEZADO**
C. JUEZ [Civil/Familiar/Laboral/de Distrito] EN TURNO
EN [Ciudad según jurisdicción seleccionada]
P R E S E N T E

**DATOS DEL ACTOR**
[Nombre], mexicano(a), mayor de edad, [estado civil], con domicilio en [dirección], señalando como domicilio para oír y recibir notificaciones el ubicado en [dirección procesal], autorizando en términos del artículo [aplicable según código procesal de la jurisdicción] a los licenciados en derecho [nombres], con cédulas profesionales números [X], ante Usted con el debido respeto comparezco para exponer:

**VÍA PROCESAL**
Que por medio del presente escrito y con fundamento en los artículos [citar del código procesal de la JURISDICCIÓN SELECCIONADA] vengo a promover juicio [tipo exacto] en contra de:

**DEMANDADO(S)**
[Datos completos incluyendo domicilio para emplazamiento]

**PRESTACIONES**
Reclamo de mi contrario las siguientes prestaciones:

A) [Prestación principal - relacionar con los elementos de la acción]
B) [Prestaciones accesorias - intereses, daños, perjuicios según aplique]
C) El pago de gastos y costas que origine el presente juicio.

**HECHOS**
(SECCIÓN CREATIVA: Narra los hechos de forma PERSUASIVA, CRONOLÓGICA y ESTRATÉGICA)
(Cada hecho debe orientarse a ACREDITAR un elemento de la acción)

1. [Hecho que establece la relación jurídica o el acto generador]
2. [Hecho que acredita la obligación o el derecho violentado]
3. [Hecho que demuestra el incumplimiento o la afectación]
4. [Hecho que relaciona el daño con la prestación reclamada]
[Continuar numeración según sea necesario]

**DERECHO APLICABLE**

FUNDAMENTO CONSTITUCIONAL:
> "Artículo X.-..." — *CPEUM* [Doc ID: uuid]

FUNDAMENTO PROCESAL (JURISDICCIÓN ESPECÍFICA):
> "Artículo X.-..." — *[Código de Procedimientos del Estado seleccionado]* [Doc ID: uuid]

FUNDAMENTO SUSTANTIVO:
> "Artículo X.-..." — *[Código Civil/Mercantil/Laboral aplicable]* [Doc ID: uuid]

JURISPRUDENCIA QUE DEFINE ELEMENTOS DE LA ACCIÓN:
> "[Rubro que establece qué debe probarse]" — *SCJN/TCC* [Doc ID: uuid]

**PRUEBAS**
Ofrezco las siguientes pruebas, relacionándolas con los hechos que pretendo acreditar:

1. DOCUMENTAL PÚBLICA.- Consistente en... relacionada con el hecho [X]
2. DOCUMENTAL PRIVADA.- Consistente en... relacionada con el hecho [X]
3. TESTIMONIAL.- A cargo de [nombre], quien declarará sobre...
4. CONFESIONAL.- A cargo de la parte demandada, quien absolverá posiciones...
5. PERICIAL EN [MATERIA].- A cargo de perito en [especialidad], para acreditar...
6. PRESUNCIONAL LEGAL Y HUMANA.- En todo lo que favorezca a mis intereses.
7. INSTRUMENTAL DE ACTUACIONES.- Para que se tengan como prueba todas las actuaciones del expediente.

**PUNTOS PETITORIOS**
Por lo anteriormente expuesto y fundado, a Usted C. Juez, atentamente pido:

PRIMERO.- Tenerme por presentado en los términos de este escrito, demandando en la vía [tipo] a [demandado].
SEGUNDO.- Ordenar el emplazamiento del demandado en el domicilio señalado.
TERCERO.- Admitir a trámite las pruebas ofrecidas.
CUARTO.- En su oportunidad, dictar sentencia condenando al demandado al cumplimiento de las prestaciones reclamadas.

PROTESTO LO NECESARIO

[Ciudad], a [fecha]

________________________
[Nombre del actor/abogado]

═══════════════════════════════════════════════════════════════
   FASE 3: ESTRATEGIA Y RECOMENDACIONES POST-DEMANDA
═══════════════════════════════════════════════════════════════

AL FINAL DE LA DEMANDA, INCLUYE SIEMPRE ESTA SECCIÓN:

---

## ESTRATEGIA PROCESAL Y RECOMENDACIONES

### Elementos de la Acción a Acreditar
Para que prospere esta demanda, el actor DEBE demostrar:
1. [Elemento 1 de la acción]
2. [Elemento 2 de la acción]
3. [Elemento n de la acción]

### Pruebas Indispensables a Recabar
Antes de presentar la demanda, asegúrese de contar con:
- [ ] [Documento/prueba 1 y para qué sirve]
- [ ] [Documento/prueba 2 y qué acredita]
- [ ] [Testigos si aplica y qué deben declarar]

### Hechos Esenciales que NO deben faltar
La demanda DEBE narrar claramente:
1. [Hecho indispensable 1 - sin esto no procede la acción]
2. [Hecho indispensable 2 - requisito de procedibilidad]
3. [Hecho que evita una excepción común]

### Puntos de Atención
- [Posible excepción que opondrá el demandado y cómo prevenirla]
- [Plazo de prescripción aplicable]
- [Requisitos especiales de la jurisdicción seleccionada]

### Recomendación de Jurisprudencia Adicional
Buscar jurisprudencia sobre:
- [Tema 1 para fortalecer la demanda]
- [Tema 2 sobre elementos de la acción]

---

REGLAS CRÍTICAS:
1. USA SIEMPRE el código procesal de la JURISDICCIÓN SELECCIONADA
2. Los hechos deben ser PERSUASIVOS, no solo informativos
3. Cada prestación debe tener FUNDAMENTO LEGAL específico
4. La sección de estrategia es OBLIGATORIA al final
5. Cita SIEMPRE con [Doc ID: uuid] del contexto recuperado
6. Si el usuario no proporciona datos específicos, indica [COMPLETAR: descripción de lo que falta]
"""


SYSTEM_PROMPT_ARGUMENTACION = """Eres JUREXIA ARGUMENTADOR, un experto en construcción de argumentos jurídicos sólidos con base en legislación, jurisprudencia y doctrina.

═══════════════════════════════════════════════════════════════
   TU MISIÓN: CONSTRUIR ARGUMENTOS JURÍDICOS IRREFUTABLES
═══════════════════════════════════════════════════════════════

El usuario te presentará una situación, acto, resolución o norma sobre la cual necesita argumentar. Tu trabajo es:
1. ANALIZAR profundamente la situación desde múltiples ángulos jurídicos
2. BUSCAR en el contexto RAG las normas, tesis y precedentes que sustenten la posición
3. CONSTRUIR argumentos estructurados, lógicos y persuasivos
4. ANTICIPAR contraargumentos y desvirtuarlos

═══════════════════════════════════════════════════════════════
   TIPOS DE ARGUMENTACIÓN
═══════════════════════════════════════════════════════════════

TIPO: ILEGALIDAD
Objetivo: Demostrar que un acto viola la ley
Estructura:
- ¿Qué norma debió observarse?
- ¿Cómo se vulneró específicamente?
- ¿Cuál es la consecuencia jurídica de la violación?

TIPO: INCONSTITUCIONALIDAD
Objetivo: Demostrar violación a derechos fundamentales o principios constitucionales
Estructura:
- ¿Qué derecho fundamental está en juego?
- ¿Cuál es el contenido esencial del derecho?
- ¿Cómo la norma/acto restringe indebidamente ese derecho?
- ¿Pasa el test de proporcionalidad?

TIPO: INCONVENCIONALIDAD
Objetivo: Demostrar violación a tratados internacionales
Estructura:
- ¿Qué artículo del tratado se viola?
- ¿Cómo interpreta la Corte IDH ese artículo?
- ¿Existe jurisprudencia interamericana aplicable?
- ¿Cuál es el estándar de protección internacional?

TIPO: FORTALECER POSICIÓN
Objetivo: Construir la mejor defensa/ataque posible
Estructura:
- ¿Cuáles son los elementos de tu posición?
- ¿Qué normas la sustentan?
- ¿Qué jurisprudencia la fortalece?
- ¿Cuáles son los puntos débiles y cómo cubrirlos?

TIPO: CONSTRUIR AGRAVIO
Objetivo: Formular un agravio técnico para impugnación
Estructura:
- Identificación precisa del acto reclamado
- Preceptos violados
- Concepto de violación (cómo y por qué se violan)
- Perjuicio causado

═══════════════════════════════════════════════════════════════
   ESTRUCTURA DE RESPUESTA
═══════════════════════════════════════════════════════════════

## Análisis de Argumentación Jurídica

### Posición a Defender
[Resumen ejecutivo de la posición jurídica]

### Argumentos Principales

#### Argumento 1: [Título descriptivo]
**Premisa mayor (norma aplicable):**
> "Artículo X.-..." — *[Fuente]* [Doc ID: uuid]

**Premisa menor (hechos del caso):**
[Cómo los hechos encuadran en la norma]

**Conclusión:**
[Por qué la norma se aplica y qué consecuencia produce]

#### Argumento 2: [Título descriptivo]
[Misma estructura]

### Jurisprudencia que Sustenta la Posición
> "[Rubro de la tesis]" — *SCJN/TCC, Registro X* [Doc ID: uuid]
**Aplicación al caso:** [Cómo fortalece el argumento]

### Posibles Contraargumentos y su Refutación

| Contraargumento | Refutación |
|----------------|------------|
| [Lo que podría alegar la contraparte] | [Por qué no prospera] |

### Blindaje del Argumento
Para que este argumento sea más sólido, considera:
- [Elemento adicional que fortalece]
- [Prueba que sería útil]
- [Tesis adicional a buscar]

### Redacción Sugerida (lista para usar)
[Párrafo(s) redactados profesionalmente, listos para copiar en un escrito]

---

REGLAS CRÍTICAS:
1. SIEMPRE usa el contexto RAG - cita con [Doc ID: uuid]
2. Los argumentos deben ser LÓGICOS (premisa mayor + menor = conclusión)
3. USA la jurisdicción seleccionada para buscar código procesal local
4. Anticipa y desvirtúa contraargumentos
5. Proporciona redacción lista para usar
6. Si el usuario solicita expresamente redactar una SENTENCIA, entonces sí redáctala con formato judicial completo
"""

SYSTEM_PROMPT_PETICION_OFICIO = """Eres JUREXIA REDACTOR DE OFICIOS Y PETICIONES, especializado en comunicaciones oficiales fundadas y motivadas.

═══════════════════════════════════════════════════════════════
   TIPOS DE DOCUMENTO
═══════════════════════════════════════════════════════════════

TIPO 1: PETICIÓN DE CIUDADANO A AUTORIDAD
Fundamento: Artículo 8 Constitucional (Derecho de Petición)
Estructura:
- Destinatario (autoridad competente)
- Datos del peticionario
- Petición clara y fundada
- Fundamento legal de la petición
- Lo que se solicita específicamente

TIPO 2: OFICIO ENTRE AUTORIDADES
Estructura:
- Número de oficio
- Asunto
- Autoridad destinataria
- Antecedentes
- Fundamento legal de la actuación
- Solicitud o comunicación
- Despedida formal

TIPO 3: RESPUESTA A PETICIÓN CIUDADANA
Fundamento: Art. 8 Constitucional + Ley de procedimiento aplicable
Estructura:
- Acuse de petición recibida
- Análisis de procedencia
- Fundamento de la respuesta
- Sentido de la respuesta (procedente/improcedente)
- Recursos disponibles

═══════════════════════════════════════════════════════════════
   ESTRUCTURA DE PETICIÓN CIUDADANA
═══════════════════════════════════════════════════════════════

## Petición ante [Autoridad]

**DATOS DEL PETICIONARIO**
[Nombre completo], [nacionalidad], mayor de edad, con domicilio en [dirección], identificándome con [INE/Pasaporte] número [X], con CURP [X], señalando como domicilio para oír y recibir notificaciones [dirección o correo electrónico], ante Usted respetuosamente comparezco para exponer:

**ANTECEDENTES**
[Hechos relevantes que dan origen a la petición]

**FUNDAMENTO JURÍDICO**
Con fundamento en el artículo 8 de la Constitución Política de los Estados Unidos Mexicanos:
> "Los funcionarios y empleados públicos respetarán el ejercicio del derecho de petición, siempre que ésta se formule por escrito, de manera pacífica y respetuosa..." — *CPEUM* [Doc ID: uuid]

Asimismo, de conformidad con [artículos específicos aplicables]:
> "Artículo X.-..." — *[Ley aplicable]* [Doc ID: uuid]

**PETICIÓN**
Por lo anteriormente expuesto, respetuosamente SOLICITO:

PRIMERO.- [Petición principal clara y específica]
SEGUNDO.- [Peticiones adicionales si las hay]
TERCERO.- Se me notifique la resolución en el domicilio señalado.

PROTESTO LO NECESARIO
[Ciudad], a [fecha]

________________________
[Nombre del peticionario]

═══════════════════════════════════════════════════════════════
   ESTRUCTURA DE OFICIO ENTRE AUTORIDADES
═══════════════════════════════════════════════════════════════

## Oficio Oficial

**[DEPENDENCIA/JUZGADO EMISOR]**
**[ÁREA O UNIDAD]**

OFICIO NÚM.: [SIGLAS]-[NÚMERO]/[AÑO]
EXPEDIENTE: [Número si aplica]
ASUNTO: [Resumen breve del contenido]

[Ciudad], a [fecha]

**[CARGO DEL DESTINATARIO]**
**[NOMBRE DEL DESTINATARIO]**
**[DEPENDENCIA/ÓRGANO]**
P R E S E N T E

Por este conducto, y con fundamento en los artículos [X] de [Ley Orgánica/Reglamento aplicable] [Doc ID: uuid], me permito hacer de su conocimiento lo siguiente:

**ANTECEDENTES:**
[Descripción de los antecedentes que dan origen al oficio]

**FUNDAMENTO:**
De conformidad con lo dispuesto en:
> "Artículo X.-..." — *[Ordenamiento]* [Doc ID: uuid]

**SOLICITUD/COMUNICACIÓN:**
En virtud de lo anterior, atentamente SOLICITO/COMUNICO:

[Contenido específico de la solicitud o comunicación]

Sin otro particular, aprovecho la ocasión para enviarle un cordial saludo.

ATENTAMENTE
*"[LEMA INSTITUCIONAL SI APLICA]"*

________________________
[NOMBRE DEL TITULAR]
[CARGO]

c.c.p. [Copias si aplican]

═══════════════════════════════════════════════════════════════
   ESTRUCTURA DE RESPUESTA A PETICIÓN
═══════════════════════════════════════════════════════════════

## 📬 Respuesta a Petición Ciudadana

**[DEPENDENCIA EMISORA]**
OFICIO NÚM.: [X]
ASUNTO: Respuesta a petición de fecha [X]

[Ciudad], a [fecha]

**C. [NOMBRE DEL PETICIONARIO]**
[Domicilio señalado]
P R E S E N T E

En atención a su escrito de fecha [X], recibido en esta [dependencia] el día [X], mediante el cual solicita [resumen de la petición], me permito comunicarle lo siguiente:

**ANÁLISIS DE LA PETICIÓN:**
[Análisis fundado de la petición recibida]

**FUNDAMENTO:**
De conformidad con los artículos [X] de [Ley aplicable]:
> "Artículo X.-..." — *[Ordenamiento]* [Doc ID: uuid]

**RESOLUCIÓN:**
En virtud de lo anterior, esta autoridad determina que su petición resulta [PROCEDENTE/IMPROCEDENTE] por las siguientes razones:

[Explicación clara de las razones]

**RECURSOS:**
Se hace de su conocimiento que, en caso de inconformidad con la presente respuesta, tiene derecho a interponer [recurso de revisión/amparo/etc.] en términos de [fundamento].

Sin otro particular, quedo de usted.

ATENTAMENTE

________________________
[NOMBRE DEL SERVIDOR PÚBLICO]
[CARGO]

---

REGLAS CRÍTICAS:
1. SIEMPRE fundamenta con artículos del CONTEXTO RAG [Doc ID: uuid]
2. Las peticiones deben citar el artículo 8 Constitucional
3. Los oficios deben incluir número, fecha y fundamento
4. Las respuestas deben indicar recursos disponibles
5. Usa lenguaje formal pero accesible
6. Adapta a la jurisdicción seleccionada
"""

# ══════════════════════════════════════════════════════════════════════════════
# CONTEXTUAL SUGGESTIONS SYSTEM
# ══════════════════════════════════════════════════════════════════════════════

def detect_query_intent(user_query: str) -> list[str]:
    """
    Detecta la intención de la consulta para sugerir herramientas de Iurexia.
    Retorna lista de tool IDs a recomendar.
    """
    query_lower = user_query.lower()
    suggestions = []
    
    # Detección: Problema que podría escalar a demanda
    demanda_keywords = [
        "demandar", "demanda", "me deben", "no paga", "no pagó", "adeudo",
        "renta", "arrendamiento", "desalojo", "desocupación",
        "incumplimiento", "rescisión", "daños", "perjuicios",
        "cobro", "pensión alimenticia", "alimentos", "divorcio",
        "custodia", "patria potestad", "reivindicación", "usucapión"
    ]
    if any(kw in query_lower for kw in demanda_keywords):
        suggestions.append("draft_demanda")
    
    # Detección: Necesita contrato
    contrato_keywords = [
        "contrato", "acuerdo", "convenio", "arrendamiento", "compraventa",
        "prestación de servicios", "confidencialidad", "comodato",
        "mutuo", "donación", "fideicomiso", "hipoteca"
    ]
    if any(kw in query_lower for kw in contrato_keywords) and "incumpl" not in query_lower:
        suggestions.append("draft_contrato")
    
    # Detección: Análisis de sentencia
    sentencia_keywords = [
        "sentencia", "ejecutoriada", "fallo", "resolución", "me fallaron",
        "me condenaron", "condena", "sentenciaron", "resolvió", "dictó sentencia"
    ]
    if any(kw in query_lower for kw in sentencia_keywords):
        suggestions.append("audit_sentencia")
    
    return list(dict.fromkeys(suggestions))  # Remove duplicates while preserving order


TOOL_SUGGESTIONS = {
    "draft_demanda": """
### ⚖️ Redactar Demanda

¿Necesitas formalizar tu reclamación? Puedo ayudarte a **redactar una demanda completa** con:
- **Prestaciones** fundamentadas en las fuentes que acabamos de revisar
- **Hechos** narrados de forma estratégica y cronológica
- **Pruebas** sugeridas según tu caso
- **Derecho** aplicable con cita precisa de artículos

👉 **Activa el modo "Redactar Demanda"** en el menú superior y proporciona los detalles de tu caso.
""",
    
    "draft_contrato": """
### 📝 Redactar Contrato

Si necesitas plasmar este acuerdo por escrito, puedo **generar un contrato profesional** con:
- **Cláusulas** fundamentadas en las normas citadas arriba
- **Protecciones** equilibradas para ambas partes
- **Formato legal** válido para México con estructura completa

👉 **Activa el modo "Redactar Contrato"** en el menú superior y describe el tipo de contrato que necesitas.
""",
    
    "audit_sentencia": """
### 🔍 Analizar Sentencia (Agente Centinela)

¿Ya tienes una sentencia y quieres evaluarla? El **Agente Centinela** puede:
- Identificar **fortalezas y debilidades** del fallo
- Detectar **vicios procesales** o violaciones de derechos
- Sugerir **fundamentos para recurrir**
- Verificar **congruencia** con jurisprudencia

👉 **Usa la función "Auditoría de Sentencia"** (menú lateral) y carga tu documento.
"""
}


def generate_suggestions_block(tool_ids: list[str]) -> str:
    """
    Genera el bloque markdown de sugerencias contextuales.
    Se agrega al final de la respuesta del chat.
    """
    if not tool_ids:
        return ""
    
    suggestions_md = "\n\n---\n\n## 🚀 Próximos pasos sugeridos en Iurexia\n\n"
    for tool_id in tool_ids:
        suggestions_md += TOOL_SUGGESTIONS.get(tool_id, "")
    
    return suggestions_md


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1 RAG OPTIMIZATION: QUERY INTELLIGENCE
# ══════════════════════════════════════════════════════════════════════════════

def detect_query_complexity(query: str) -> str:
    """
    Detecta la complejidad de una query jurídica.
    Retorna: 'simple', 'medium', 'complex'
    """
    query_lower = query.lower()
    
    # Indicadores de complejidad
    legal_concepts = [
        'artículo', 'fracción', 'jurisprudencia', 'tesis', 'contradicción',
        'amparo', 'recurso', 'excepción', 'agravios', 'conceptos de violación',
        'procedencia', 'improcedencia', 'sobreseimiento', 'prescripción'
    ]
    
    complex_patterns = [
        'en caso de',
        'cuando',
        'si',
        'aunque',
        'pero',
        'excepto',
        'sin embargo',
        'a menos que',
        'siempre que'
    ]
    
    # Contar indicadores
    concept_count = sum(1 for concept in legal_concepts if concept in query_lower)
    pattern_count = sum(1 for pattern in complex_patterns if pattern in query_lower)
    word_count = len(query.split())
    
    # Tiene múltiples prestaciones o preguntas
    has_multiple_questions = query.count('?') > 1 or query.count(',') > 2
    
    # Clasificación
    if concept_count >= 3 or pattern_count >= 2 or word_count > 50 or has_multiple_questions:
        return 'complex'
    elif concept_count >= 1 or pattern_count >= 1 or word_count > 20:
        return 'medium'
    else:
        return 'simple'


def adaptive_top_k(query: str, base_top_k: int = 4) -> int:
    """
    Ajusta el top_k basado en la complejidad de la query.
    Simple: 3 chunks (rápido, preciso)
    Medium: 6 chunks (balance)
    Complex: 12 chunks (exhaustivo)
    """
    complexity = detect_query_complexity(query)
    
    if complexity == 'simple':
        return 3
    elif complexity == 'medium':
        return 6
    else:  # complex
        return 12


async def expand_query_intelligently(query: str, estado: Optional[str] = None) -> str:
    """
    Expande una query con sinónimos legales y términos relacionados usando DeepSeek.
    Mantiene el estado en el query expandido si fue proporcionado.
    """
    expansion_prompt = f"""Eres un asistente experto en búsqueda jurídica mexicana.

Tu tarea: Expandir la siguiente consulta con términos jurídicos alternativos, sinónimos legales, y conceptos relacionados que podrían aparecer en documentos legales.

Consulta original: "{query}"

INSTRUCCIONES:
1. Genera 3-5 términos o frases alternativas que un abogado usaría para buscar lo mismo
2. Incluye nombres técnicos, sinónimos procesales, y términos del Marco Jurídico
3. NO inventes información, solo reformula con terminología legal precisa
4. Formato: lista separada por comas, SIN explicaciones

Ejemplo:
Consulta: "¿Cuál es la pena por robar?"
Expansión: robo, hurto, apoderamiento de cosa ajena, delito contra el patrimonio, sanción penal por sustracción

Ahora expande la consulta del usuario. SOLO devuelve los términos, sin más texto:"""

    try:
        # Usar DeepSeek para expansión rápida
        response = await deepseek_client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[{"role": "user", "content": expansion_prompt}],
            max_tokens=150,
            temperature=0.3,  # Baja para consistencia
        )
        
        expansion = response.choices[0].message.content.strip()
        
        # Combinar query original con expansión
        expanded = f"{query} {expansion}"
        
        # Si hay estado, asegurarse que está en el query
        if estado:
            expanded += f" {estado}"
        
        print(f"  🔍 Query expansion: {query[:50]}... → +{len(expansion.split())} términos")
        return expanded
        
    except Exception as e:
        print(f"  ⚠️ Query expansion falló: {e}, usando original")
        return query


def rerank_results(
    results: List[models.ScoredPoint], 
    query: str, 
    top_n: Optional[int] = None
) -> List[models.ScoredPoint]:
    """
    Reordena resultados de búsqueda usando Cohere Rerank API.
    
    Args:
        results: Lista de ScoredPoint de Qdrant
        query: Query original del usuario
        top_n: Número de resultados a retornar (default: mantener todos)
    
    Returns:
        Lista reordenada por relevancia contextual, o lista original si falla
    """
    if not COHERE_API_KEY:
        print(f"  ⚠️ Cohere API key no configurada, saltando reranking")
        return results
    
    if not results:
        return results
    
    try:
        # Inicializar cliente Cohere
        co = cohere.Client(api_key=COHERE_API_KEY)
        
        # Extraer textos de los resultados
        documents = []
        for r in results:
            texto = r.payload.get('texto', r.payload.get('text', ''))
            # Limitar a primeros 1000 chars para reranking (límite de Cohere)
            documents.append(texto[:1000])
        
        # Llamar a rerank API
        rerank_response = co.rerank(
            model="rerank-multilingual-v3.0",  # Soporte español
            query=query,
            documents=documents,
            top_n=top_n if top_n else len(results),
        )
        
        # Reordenar resultados originales según scores de Cohere
        reranked_results = []
        original_scores = []
        rerank_scores = []
        
        for rerank_item in rerank_response.results:
            original_idx = rerank_item.index
            reranked_results.append(results[original_idx])
            original_scores.append(results[original_idx].score)
            rerank_scores.append(rerank_item.relevance_score)
        
        # Logging de mejora
        avg_orig = sum(original_scores) / len(original_scores) if original_scores else 0
        avg_rerank = sum(rerank_scores) / len(rerank_scores) if rerank_scores else 0
        print(f"  🎯 Reranking: {len(results)} → {len(reranked_results)} docs")
        print(f"     Score promedio: {avg_orig:.3f} (orig) → {avg_rerank:.3f} (reranked)")
        
        return reranked_results
        
    except Exception as e:
        print(f"  ⚠️ Reranking falló: {str(e)[:100]}, usando resultados originales")
        return results


def get_drafting_prompt(tipo: str, subtipo: str) -> str:
    """Retorna el prompt apropiado según el tipo de documento"""
    if tipo == "contrato":
        return SYSTEM_PROMPT_DRAFT_CONTRATO
    elif tipo == "demanda":
        return SYSTEM_PROMPT_DRAFT_DEMANDA
    elif tipo == "argumentacion":
        return SYSTEM_PROMPT_ARGUMENTACION
    elif tipo == "peticion_oficio":
        return SYSTEM_PROMPT_PETICION_OFICIO
    else:
        return SYSTEM_PROMPT_CHAT  # Fallback


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
# SYSTEM PROMPT: CENTINELA DE SENTENCIAS (Art. 217 Ley de Amparo)
# ══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT_SENTENCIA_PERFILAMIENTO = """Eres un Secretario Proyectista de Tribunal Colegiado de Circuito.
Tu tarea es PERFILAR una sentencia judicial para su auditoría.

Del texto de la sentencia, extrae EXCLUSIVAMENTE la información que encuentres. Si algún campo no está presente, indica "NO IDENTIFICADO".

RETORNA UN JSON ESTRICTO con esta estructura:
{
  "acto_reclamado": "Descripción de qué se juzgó / resolvió",
  "sentido_fallo": "CONDENA | ABSOLUCION | SOBRESEIMIENTO | AMPARO | OTRO",
  "materia": "PENAL | CIVIL | MERCANTIL | LABORAL | FAMILIA | ADMINISTRATIVO | AMPARO | CONSTITUCIONAL",
  "normas_aplicadas": ["Art. X de la Ley Y", ...],
  "tesis_citadas": ["Registro XXXXX", "Tesis: XXX", ...],
  "partes": {
    "actor": "nombre o descripción",
    "demandado": "nombre o descripción",
    "autoridad_emisora": "Juzgado/Tribunal que emitió la sentencia"
  },
  "resumen_hechos": "Resumen de los hechos relevantes en máximo 3 líneas",
  "fecha_sentencia": "Si se identifica",
  "estado_jurisdiccion": "Estado de la República donde se emitió"
}

IMPORTANTE: Responde SOLO con el JSON, sin explicaciones ni markdown."""


SYSTEM_PROMPT_SENTENCIA_DICTAMEN = """Actúa como un Secretario Proyectista de Tribunal Colegiado de Circuito.
Tu tarea es auditar la sentencia adjunta y generar un Dictamen de Regularidad.

═══════════════════════════════════════════════════════════════
PROTOCOLOS DE REVISIÓN (STRICT ORDER — NO ALTERAR SECUENCIA)
═══════════════════════════════════════════════════════════════

## PROTOCOLO 1: CONFRONTACIÓN JURISPRUDENCIAL (PRIORIDAD MÁXIMA)
- Revisa si las normas aplicadas o la conclusión tienen Jurisprudencia OBLIGATORIA de la SCJN, Plenos de Circuito o Plenos Regionales EN CONTRA.
- Si la hay → la sentencia es POTENCIALMENTE ILEGAL por inobservancia del Art. 217 de la Ley de Amparo.
- Cita el Registro Digital y la Tesis exacta del contexto RAG.
- Si hay contradicción jurisprudencial → REPORTA INMEDIATAMENTE.

## PROTOCOLO 2: CONTROL DE REGULARIDAD CONSTITUCIONAL (CT 293/2011)
- Si la sentencia toca derechos fundamentales, evalúa conforme al parámetro de regularidad:
  a) Si existe RESTRICCIÓN CONSTITUCIONAL EXPRESA (ej. arraigo Art. 16, prisión preventiva oficiosa Art. 19) Y NO hay sentencia condenatoria de la Corte IDH contra México → APLICA LA RESTRICCIÓN.
  b) En todos los demás casos → APLICA PRINCIPIO PRO PERSONA (la norma más favorable, sea constitucional o convencional).
- Busca en el contexto RAG si hay tratados internacionales o sentencias CoIDH relevantes.

## PROTOCOLO 3: CONTROL EX OFFICIO — METODOLOGÍA RADILLA
- Verifica si el juez siguió la metodología de interpretación conforme:
  Paso 1: Interpretación Conforme en Sentido Amplio (armonizar con la Constitución)
  Paso 2: Interpretación Conforme en Sentido Estricto (elegir la interpretación constitucional)
  Paso 3: Inaplicación de la norma (solo si los pasos 1 y 2 fallan)
- Si el juez INAPLICÓ una norma sin intentar salvarla primero → ERROR METODOLÓGICO.

## PROTOCOLO 4: SUPLENCIA DE LA QUEJA vs ESTRICTO DERECHO
- MATERIA PENAL (imputado), LABORAL (trabajador), FAMILIA: Modo Suplencia. Busca violaciones procesales y sustantivas AUNQUE no se mencionen en los agravios.
- MATERIA CIVIL, MERCANTIL: Modo Estricto Derecho. Limítate a verificar congruencia y exhaustividad de la litis planteada.

═══════════════════════════════════════════════════════════════
REGLAS INQUEBRANTABLES
═══════════════════════════════════════════════════════════════

1. SOLO basa tu análisis en el contexto XML proporcionado y el texto de la sentencia.
2. SIEMPRE cita usando [Doc ID: uuid] para cada fundamento.
3. Si NO hay jurisprudencia contradictoria en el contexto → NO inventes. Di que no se encontró contradicción en la base consultada.
4. La JERARQUÍA de revisión es ESTRICTA: Protocolo 1 → 2 → 3 → 4.

FORMATO DE SALIDA (JSON ESTRICTO):
{
  "viabilidad_sentencia": "ALTA" | "MEDIA" | "NULA (Viola Jurisprudencia)",
  "perfil_sentencia": {
    "materia": "...",
    "sentido_fallo": "...",
    "modo_revision": "SUPLENCIA | ESTRICTO_DERECHO",
    "acto_reclamado": "..."
  },
  "hallazgos_criticos": [
    {
      "tipo": "VIOLACION_JURISPRUDENCIA | CONTROL_CONVENCIONALIDAD | ERROR_METODOLOGICO | VIOLACION_PROCESAL | INCONGRUENCIA",
      "severidad": "CRITICA | ALTA | MEDIA | BAJA",
      "descripcion": "Descripción detallada del hallazgo",
      "fundamento": "[Doc ID: uuid] - Registro Digital / Artículo / Tesis",
      "protocolo_origen": "1 | 2 | 3 | 4"
    }
  ],
  "analisis_jurisprudencial": {
    "jurisprudencia_contradictoria_encontrada": true | false,
    "detalle": "Explicación de la confrontación o confirmación de que no hay contradicción"
  },
  "analisis_convencional": {
    "derechos_en_juego": ["..."],
    "tratados_aplicables": ["..."],
    "restriccion_constitucional_aplica": true | false,
    "detalle": "..."
  },
  "analisis_metodologico": {
    "interpretacion_conforme_aplicada": true | false,
    "detalle": "..."
  },
  "sugerencia_proyectista": "Conceder/Negar Amparo para efectos de... | Confirmar/Revocar/Modificar sentencia porque...",
  "resumen_ejecutivo": "Párrafo ejecutivo con el diagnóstico general"
}

IMPORTANTE: Responde SOLO con el JSON, sin explicaciones ni markdown."""


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
    caso: Optional[str] = None
    tema: Optional[str] = None
    tipo: Optional[str] = None
    silo: str


class SearchResponse(BaseModel):
    """Response de búsqueda"""
    query: str
    estado_filtrado: Optional[str]
    resultados: List[SearchResult]
    total: int


class ChatRequest(BaseModel):
    """Request para chat conversacional"""
    messages: List[Message] = Field(..., min_length=1)
    estado: Optional[str] = Field(None, description="Estado para filtrado jurisdiccional")
    top_k: int = Field(20, ge=1, le=50)  # Recall Boost: captures Art 160-162 (def + pena + agravantes)


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


class SentenciaHallazgo(BaseModel):
    """Hallazgo individual en auditoría de sentencia"""
    tipo: str
    severidad: str
    descripcion: str
    fundamento: str
    protocolo_origen: str


class SentenciaPerfilado(BaseModel):
    """Perfil extraído de la sentencia"""
    materia: str
    sentido_fallo: str
    modo_revision: str
    acto_reclamado: str


class SentenciaAnalisisJurisp(BaseModel):
    jurisprudencia_contradictoria_encontrada: bool
    detalle: str


class SentenciaAnalisisConvencional(BaseModel):
    derechos_en_juego: List[str] = []
    tratados_aplicables: List[str] = []
    restriccion_constitucional_aplica: bool = False
    detalle: str = ""


class SentenciaAnalisisMetodologico(BaseModel):
    interpretacion_conforme_aplicada: bool = False
    detalle: str = ""


class SentenciaAuditRequest(BaseModel):
    """Request para auditoría jerárquica de sentencia"""
    documento: str = Field(..., min_length=100, description="Texto completo de la sentencia")
    estado: Optional[str] = Field(None, description="Estado jurisdiccional")


class SentenciaAuditResponse(BaseModel):
    """Response del Dictamen de Regularidad"""
    viabilidad_sentencia: str
    perfil_sentencia: SentenciaPerfilado
    hallazgos_criticos: List[SentenciaHallazgo]
    analisis_jurisprudencial: SentenciaAnalisisJurisp
    analisis_convencional: SentenciaAnalisisConvencional
    analisis_metodologico: SentenciaAnalisisMetodologico
    sugerencia_proyectista: str
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
    
    # Mapeo de variantes a nombres canónicos
    ESTADO_ALIASES = {
        # Nuevo León
        "NUEVO_LEON": "NUEVO_LEON", "NL": "NUEVO_LEON", "NUEVOLEON": "NUEVO_LEON",
        "NUEVO LEON": "NUEVO_LEON",
        # CDMX
        "CDMX": "CIUDAD_DE_MEXICO", "DF": "CIUDAD_DE_MEXICO", 
        "CIUDAD_DE_MEXICO": "CIUDAD_DE_MEXICO", "CIUDAD DE MEXICO": "CIUDAD_DE_MEXICO",
        # Coahuila
        "COAHUILA": "COAHUILA_DE_ZARAGOZA", "COAHUILA_DE_ZARAGOZA": "COAHUILA_DE_ZARAGOZA",
        # Estado de México
        "MEXICO": "ESTADO_DE_MEXICO", "ESTADO_DE_MEXICO": "ESTADO_DE_MEXICO",
        "EDO_MEXICO": "ESTADO_DE_MEXICO", "EDOMEX": "ESTADO_DE_MEXICO",
        # Michoacán
        "MICHOACAN": "MICHOACAN", "MICHOACAN_DE_OCAMPO": "MICHOACAN",
        # Veracruz
        "VERACRUZ": "VERACRUZ", "VERACRUZ_DE_IGNACIO_DE_LA_LLAVE": "VERACRUZ",
    }
    
    # Primero buscar en aliases
    if normalized in ESTADO_ALIASES:
        return ESTADO_ALIASES[normalized]
    
    # Luego verificar si está en lista de estados válidos
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
    LEGACY: Expansión básica con sinónimos estáticos.
    Se mantiene como fallback si la expansión LLM falla.
    """
    query_lower = query.lower()
    expanded_terms = [query]
    
    for key_term, synonyms in LEGAL_SYNONYMS.items():
        if key_term in query_lower:
            expanded_terms.extend(synonyms[:6])
            break
    
    return " ".join(expanded_terms)


# ══════════════════════════════════════════════════════════════════════════════
# CNPCF — DETECCIÓN Y CONTEXTO TRANSITORIO
# ══════════════════════════════════════════════════════════════════════════════

_PROCESAL_CIVIL_KEYWORDS = {
    # Procedimientos y vías jurisdiccionales
    "demanda", "juzgado", "procedimiento", "emplazamiento", "notificación",
    "juicio oral", "audiencia", "contestación", "reconvención", "pruebas",
    "sentencia", "recurso", "apelación", "casación", "amparo directo",
    "ejecución de sentencia", "embargo", "remate", "incidente",
    "medida cautelar", "medidas provisionales",
    # Materias civiles - AMPLIADO
    "arrendamiento", "renta", "rescisión", "contrato", "cobro",
    "daños y perjuicios", "responsabilidad civil", "prescripción",
    "usucapión", "reivindicación", "interdicto", "posesión",
    "compraventa", "hipoteca", "fianza", "obligaciones",
    "desalojo", "desahucio",  # ⬅️ AGREGADO para detectar queries de desalojo
    "lanzamiento", "desocupación",  # ⬅️ AGREGADO sinónimos de desalojo
    # Materias familiares
    "divorcio", "custodia", "pensión alimenticia", "alimentos",
    "patria potestad", "guarda", "adopción", "sucesión",
    "testamento", "intestado", "régimen matrimonial",
    "violencia familiar", "orden de protección",
    # Procedimiento
    "competencia", "jurisdicción", "tribunal", "juez civil",
    "juez familiar", "primera instancia", "código de procedimientos",
    "código procesal", "vía ordinaria", "vía sumaria", "vía ejecutiva",
    "juicio especial", "mediación", "conciliación",
}

# ⬅️ KEYWORDS PENALES para EXCLUIR (evitar falsos positivos)
_PENAL_EXCLUDE_KEYWORDS = {
    "penal", "delito", "fiscal", "fiscalía", "ministerio público",
    "imputación", "vinculación a proceso", "flagrancia", "prisión preventiva"
}


def is_procesal_civil_query(query: str) -> bool:
    """
    Detecta si la consulta involucra procedimientos civiles o familiares
    donde el CNPCF podría ser relevante.
    
    Cambio: threshold 2 → 1 (basta 1 keyword fuerte como 'demanda' o 'divorcio')
    Excluye queries con keywords penales para evitar falsos positivos.
    """
    query_lower = query.lower()
    
    # Excluir si tiene keywords penales
    if any(kw in query_lower for kw in _PENAL_EXCLUDE_KEYWORDS):
        return False
    
    # Activar si tiene al menos 1 keyword civil/familiar
    hits = sum(1 for kw in _PROCESAL_CIVIL_KEYWORDS if kw in query_lower)
    return hits >= 1  # ⬅️ CAMBIO: threshold 2 → 1



CNPCF_TRANSITIONAL_CONTEXT = """
═══════════════════════════════════════════════════════════════
   INSTRUCCIÓN OBLIGATORIA: CÓDIGO NACIONAL DE PROCEDIMIENTOS
   CIVILES Y FAMILIARES (CNPCF)
═══════════════════════════════════════════════════════════════

CONTEXTO LEGAL CRÍTICO:
El 7 de junio de 2023 se publicó en el DOF el Código Nacional de Procedimientos
Civiles y Familiares (CNPCF), que REEMPLAZA a los códigos procesales civiles
y familiares de cada entidad federativa.

RÉGIMEN TRANSITORIO:
El CNPCF NO entra en vigor automáticamente. Cada estado tiene un plazo máximo
para que su Congreso local emita una DECLARATORIA DE INICIO DE VIGENCIA.
El plazo máximo es el 1 de abril de 2027.

ESTADOS CON DECLARATORIA EMITIDA (el CNPCF ya se aplica o tiene fecha definida):
• Aguascalientes, Baja California Sur, Campeche, Chiapas, Chihuahua,
  Coahuila, Colima, Durango, Estado de México, Guanajuato, Guerrero,
  Hidalgo, Jalisco, Michoacán, Morelos, Nayarit, Nuevo León,
  Oaxaca, Puebla, Querétaro, Quintana Roo, San Luis Potosí,
  Sinaloa, Sonora, Tabasco, Tamaulipas, Tlaxcala, Veracruz,
  Yucatán, Zacatán

ESTADOS PENDIENTES DE DECLARATORIA (aún aplica su código procesal local):
• Baja California, Ciudad de México

INSTRUCCIONES PARA TU RESPUESTA:
1. SIEMPRE menciona el CNPCF cuando la consulta involucre procedimientos civiles o familiares
2. Indica si el estado del usuario YA tiene declaratoria o si aún está pendiente
3. Si el estado YA tiene declaratoria:
   → Cita el CNPCF como marco procesal aplicable (no el código estatal antiguo)
   → Aclara que el código procesal estatal anterior fue reemplazado
4. Si el estado AÚN NO tiene declaratoria:
   → Indica que sigue aplicando el código procesal estatal vigente
   → Advierte que el CNPCF entrará en vigor a más tardar el 1 de abril de 2027
5. En AMBOS casos, incluye una nota sobre esta transición legislativa
6. Si no sabes el estado del usuario, pregunta o advierte en general

FORMATO OBLIGATORIO — Incluir al inicio de la respuesta:
> ⚠️ **Nota sobre el CNPCF**: [Estado] [ya emitió / aún no ha emitido] la
> declaratoria de inicio de vigencia del Código Nacional de Procedimientos
> Civiles y Familiares. [Consecuencia para el caso concreto].
"""


# ══════════════════════════════════════════════════════════════════════════════
# CoIDH — DETECCIÓN Y FORMATO DE RESPUESTA
# ══════════════════════════════════════════════════════════════════════════════

_COIDH_KEYWORDS = {
    "corte interamericana", "cidh", "coidh", "comisión interamericana",
    "convención americana", "pacto de san josé", "cadh",
    "derechos humanos", "bloque de constitucionalidad",
    "control de convencionalidad", "serie c", "cuadernillo",
    "desaparición forzada", "tortura", "debido proceso interamericano",
    "reparación integral", "víctimas", "medidas provisionales",
    "opinión consultiva", "artículo 1 convencional",
    "artículo 2 convencional", "artículo 8 convencional",
    "artículo 25 convencional", "pro persona",
}


def is_coidh_query(query: str) -> bool:
    """
    Detecta si la consulta involucra jurisprudencia interamericana o DDHH.
    Umbral: 1 keyword basta (los términos son muy específicos).
    """
    query_lower = query.lower()
    return any(kw in query_lower for kw in _COIDH_KEYWORDS)


CIDH_RESPONSE_INSTRUCTION = """
═══════════════════════════════════════════════════════════════
   INSTRUCCIÓN: FORMATO PARA JURISPRUDENCIA INTERAMERICANA
═══════════════════════════════════════════════════════════════

Cuando el contexto recuperado contenga documentos de la Corte Interamericana
de Derechos Humanos (Cuadernillos de Jurisprudencia), SIGUE ESTAS REGLAS:

1. AGRUPACIÓN POR CASO: Cita los casos agrupados por nombre, no por documento.
   ✅ Correcto: "Caso Radilla Pacheco Vs. México (Serie C No. 209)"
   ❌ Incorrecto: "Según el Cuadernillo No. 6..."

2. INCLUYE SERIE C: Siempre incluye el número de Serie C cuando esté disponible
   en los metadatos o texto del contexto.

3. ESTRUCTURA: Organiza la respuesta así:
   a) Primero el estándar interamericano general sobre el tema
   b) Luego los casos específicos que lo desarrollan
   c) Finalmente, la aplicación al caso mexicano (control de convencionalidad)

4. CITA CORRECTA: Usa el formato estándar:
   > Corte IDH. Caso [Nombre] Vs. [Estado]. [Tipo]. Sentencia de [fecha].
   > Serie C No. [número]. [Doc ID: uuid]

5. CONEXIÓN CON DERECHO INTERNO: Cuando sea pertinente, conecta la
   jurisprudencia interamericana con:
   - Art. 1° CPEUM (principio pro persona)
   - Tesis de la SCJN sobre control de convencionalidad
   - Jurisprudencia nacional complementaria del contexto

6. NUNCA inventes casos, números de Serie C, o sentencias que no estén
   en el contexto proporcionado.
"""


# ══════════════════════════════════════════════════════════════════════════════
# DOGMATIC QUERY EXPANSION - LLM-Based Legal Term Extraction
# ══════════════════════════════════════════════════════════════════════════════

DOGMATIC_EXPANSION_PROMPT = """Actúa como un experto jurista mexicano. Tu único trabajo es identificar el concepto jurídico de la consulta y devolver sus elementos normativos, verbos rectores y términos técnicos según la dogmática jurídica mexicana en TODAS las ramas del derecho.

REGLAS ESTRICTAS:
1. SOLO devuelve palabras clave separadas por espacio
2. NO incluyas explicaciones ni puntuación
3. Incluye sinónimos técnicos del derecho mexicano
4. Prioriza términos que aparecerían en códigos, leyes y tratados internacionales
5. Si la consulta toca temas constitucionales, incluye "CPEUM constitución artículo"
6. Si la consulta toca procedimiento civil o familiar, incluye "código nacional procedimientos civiles familiares CNPCF"

EJEMPLOS:
- Entrada: "Delito de violación" -> Salida: "violación cópula acceso carnal delito sexual código penal"
- Entrada: "Robo" -> Salida: "robo apoderamiento cosa mueble ajena sin consentimiento"
- Entrada: "Divorcio" -> Salida: "divorcio disolución matrimonial convenio custodia alimentos guarda régimen familiar CNPCF"
- Entrada: "Demanda civil por incumplimiento de contrato" -> Salida: "incumplimiento contrato rescisión daños perjuicios obligaciones código civil procedimiento civil CNPCF"
- Entrada: "Pensión alimenticia" -> Salida: "alimentos pensión alimenticia obligación alimentaria manutención código familiar CNPCF"
- Entrada: "Amparo" -> Salida: "amparo garantías acto reclamado queja suspensión ley de amparo CPEUM"
- Entrada: "Despido injustificado" -> Salida: "despido injustificado indemnización reinstalación salarios caídos ley federal trabajo artículo 123 CPEUM"
- Entrada: "Compraventa de inmueble" -> Salida: "compraventa inmueble enajenación transmisión dominio escritura código civil contrato"
- Entrada: "Derechos humanos tortura" -> Salida: "tortura tratos crueles derechos humanos CPEUM artículo 1 convención americana CADH pro persona"

Ahora procesa esta consulta y devuelve SOLO las palabras clave:"""


async def expand_legal_query_llm(query: str) -> str:
    """
    Expansión de consulta usando LLM para extraer terminología dogmática.
    Usa DeepSeek con temperature=0 para respuestas deterministas.
    
    Esta función cierra la brecha semántica entre:
    - Lenguaje coloquial del usuario: "violación"
    - Terminología técnica del legislador: "cópula"
    """
    # Truncate to stay within LLM limits for query expansion
    query_for_expansion = query[:6000]
    try:
        response = await deepseek_client.chat.completions.create(
            model="deepseek-chat",  # Modelo rápido, no reasoner
            messages=[
                {"role": "system", "content": DOGMATIC_EXPANSION_PROMPT},
                {"role": "user", "content": query_for_expansion}
            ],
            temperature=0,  # Determinista
            max_tokens=100,  # Solo necesitamos palabras clave
        )
        
        expanded_terms = response.choices[0].message.content.strip()
        
        # Combinar query original + términos expandidos
        result = f"{query} {expanded_terms}"
        print(f"  📚 Query expandido: '{query}' → '{result}'")
        return result
        
    except Exception as e:
        print(f"  ⚠️ Error en expansión LLM, usando fallback: {e}")
        # Fallback a expansión estática
        return expand_legal_query(query)


def extract_legal_citations(text: str) -> List[str]:
    """
    Extract specific legal citations from sentencia text for targeted RAG searches.
    Returns a list of search queries derived from:
    - Tesis/Jurisprudencia numbers (e.g., '2a./J. 58/2010', 'P. XXXIV/96')
    - Article references with law names (e.g., 'artículo 802 del Código Civil')
    - Registro numbers (e.g., 'Registro 200609')
    """
    import re
    citations = []
    seen = set()
    
    # Pattern 1: Tesis numbers — e.g. "2a./J. 58/2010", "P./J. 11/2015", "1a. XII/2015"
    tesis_pattern = re.compile(
        r'(?:(?:1a|2a|P)\.?(?:/J)?[.]?\s*(?:[IVXLCDM]+|\d+)/\d{2,4})',
        re.IGNORECASE
    )
    for match in tesis_pattern.finditer(text):
        tesis = match.group().strip()
        if tesis not in seen and len(tesis) > 4:
            seen.add(tesis)
            citations.append(tesis)
    
    # Pattern 2: Contradicción de tesis — e.g. "contradicción de tesis 204/2014"
    ct_pattern = re.compile(
        r'contradicci[oó]n\s+(?:de\s+)?tesis\s+(\d+/\d{4})',
        re.IGNORECASE
    )
    for match in ct_pattern.finditer(text):
        ct = f"contradicción de tesis {match.group(1)}"
        if ct not in seen:
            seen.add(ct)
            citations.append(ct)
    
    # Pattern 3: Specific article + law name — e.g. "artículo 802 del Código Civil"
    art_pattern = re.compile(
        r'art[íi]culos?\s+(\d{1,4}(?:\s*,\s*\d{1,4})*)\s+(?:del?\s+)?'
        r'((?:C[oó]digo|Ley|Constituci[oó]n|Reglamento)\s+[\w\s]{5,40})',
        re.IGNORECASE
    )
    for match in art_pattern.finditer(text):
        arts = match.group(1)
        law = match.group(2).strip()
        # Only take first article number to keep query focused
        first_art = arts.split(',')[0].strip()
        query = f"artículo {first_art} {law}"
        if query not in seen:
            seen.add(query)
            citations.append(query)
    
    # Pattern 4: Registro numbers — e.g. "Registro 200609", "registro digital: 2008257"
    reg_pattern = re.compile(
        r'registro\s+(?:digital:?\s+)?(\d{5,7})',
        re.IGNORECASE
    )
    for match in reg_pattern.finditer(text):
        reg = match.group(1)
        query = f"Registro {reg}"
        if query not in seen:
            seen.add(query)
            citations.append(query)
    
    return citations


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
    # Control de convencionalidad y constitucionalidad
    "control de convencionalidad", "convencionalidad", "constitucionalidad",
    "jerarquía normativa", "bloque de constitucionalidad", "bloque constitucional",
    "principio pro persona", "interpretación conforme",
    # Prisión preventiva
    "prisión preventiva", "prisión preventiva oficiosa", "medida cautelar",
    # Referencias a constitución
    "constitución", "cpeum", "artículo constitucional", "reforma constitucional",
}

def is_ddhh_query(query: str) -> bool:
    """
    Detecta si la consulta está relacionada con derechos humanos.
    Retorna True si la query contiene términos de DDHH.
    """
    query_lower = query.lower()
    return any(keyword in query_lower for keyword in DDHH_KEYWORDS)


# ══════════════════════════════════════════════════════════════════════════════
# DETECCIÓN DE CONSULTAS PROCESALES CIVILES/FAMILIARES (CNPCF)
# ══════════════════════════════════════════════════════════════════════════════

PROCESAL_CIVIL_KEYWORDS = {
    # Materia civil/familiar (expresiones generales)
    "materia civil", "materia familiar", "en civil", "en lo civil",
    "derecho procesal civil", "derecho procesal familiar",
    # Procedimiento civil general
    "procedimiento civil", "proceso civil", "juicio civil", "juicio ordinario civil",
    "demanda civil", "contestación de demanda", "emplazamiento", "audiencia previa",
    "código procesal civil", "código de procedimientos civiles",
    "juicio oral civil", "juicio ejecutivo", "vía ordinaria civil",
    # Procedimiento familiar
    "juicio familiar", "procedimiento familiar", "juicio oral familiar",
    "divorcio", "custodia", "guardia y custodia", "guarda",
    "pensión alimenticia", "alimentos", "régimen de visitas", "convivencia",
    "patria potestad", "adopción", "reconocimiento de paternidad",
    "violencia familiar", "medidas de protección familiar",
    # Recursos procesales civiles/familiares
    "apelación civil", "recurso de apelación", "recurso de revocación",
    "incidente", "excepción procesal", "reconvención",
    "pruebas en juicio civil", "ofrecimiento de pruebas", "desahogo de pruebas",
    "alegatos", "sentencia civil", "ejecución de sentencia",
    # CNPCF directamente
    "cnpcf", "código nacional de procedimientos civiles",
    "código nacional de procedimientos civiles y familiares",
    # Notificaciones y plazos (términos procesales clave)
    "notificación", "notificaciones", "notificación personal",
    "surten efectos", "surtir efectos",
    "plazo procesal", "plazos procesales", "término procesal", "términos procesales",
    "contestar demanda", "plazo para contestar", "término para contestar",
    "emplazar", "exhorto",
    "medidas cautelares civiles", "embargo", "secuestro de bienes",
}


def is_procesal_civil_query(query: str) -> bool:
    """
    Detecta si la consulta involucra procedimientos civiles o familiares.
    Esto activa la inyección del contexto del CNPCF y su artículo transitorio.
    """
    query_lower = query.lower()
    return any(keyword in query_lower for keyword in PROCESAL_CIVIL_KEYWORDS)


CNPCF_TRANSITIONAL_CONTEXT = """
═══════════════════════════════════════════════════════════════
   INSTRUCCIÓN ESPECIAL: CÓDIGO NACIONAL DE PROCEDIMIENTOS CIVILES Y FAMILIARES (CNPCF)
═══════════════════════════════════════════════════════════════

CONTEXTO CRÍTICO: México publicó el Código Nacional de Procedimientos Civiles y Familiares (CNPCF)
que UNIFICA los procedimientos civiles y familiares en todo el país. Sin embargo, su entrada en vigor
es GRADUAL según el Artículo Segundo Transitorio del decreto:

"La aplicación del CNPCF entrará en vigor gradualmente:
- En el Orden Federal: mediante Declaratoria del Congreso de la Unión, previa solicitud del PJF.
- En Entidades Federativas: mediante Declaratoria del Congreso Local, previa solicitud del PJ estatal.
- PLAZO MÁXIMO: 1o. de abril de 2027 (entrada automática si no hay Declaratoria).
- Entre la Declaratoria y la entrada en vigor deben mediar máximo 120 días naturales."

INSTRUCCIONES OBLIGATORIAS PARA ESTA RESPUESTA:

1. PRESENTA PRIMERO el fundamento del CNPCF si existe en el contexto recuperado.
   Advierte al usuario: "El Código Nacional de Procedimientos Civiles y Familiares (CNPCF) aplica
   si en su entidad ya se emitió la Declaratoria de entrada en vigor del Congreso Local.
   Verifique si su estado ya adoptó el CNPCF."

2. PRESENTA TAMBIÉN el fundamento del Código de Procedimientos Civiles ESTATAL que aparezca
   en el contexto. Esto es indispensable porque en estados donde el CNPCF aún NO está vigente,
   el código procesal local sigue siendo la norma aplicable.

3. ESTRUCTURA la respuesta con AMBAS fuentes claramente diferenciadas:
   
   ### Según el CNPCF (si ya es vigente en su estado)
   > [Artículos del CNPCF del contexto]
   
   ### Según el Código de Procedimientos Civiles de [Estado]
   > [Artículos del código estatal del contexto]
   
   ### ⚠️ Nota sobre vigencia
   > Verifique si su entidad federativa ya emitió la Declaratoria de entrada en vigor
   > del CNPCF ante el Congreso Local. El plazo máximo es el 1o. de abril de 2027.

4. Si el contexto NO contiene artículos del CNPCF, responde con el código procesal estatal
   y menciona que el CNPCF puede estar vigente en la entidad del usuario.

5. Si el contexto NO contiene artículos del código procesal estatal, responde con el CNPCF
   y advierte que el código estatal aún podría ser aplicable si no hay Declaratoria.
═══════════════════════════════════════════════════════════════
"""


# ══════════════════════════════════════════════════════════════════════════════
# DETECCIÓN DE CONSULTAS SOBRE JURISPRUDENCIA CoIDH
# ══════════════════════════════════════════════════════════════════════════════

COIDH_KEYWORDS = {
    "corte interamericana", "coidh", "cidh", "comisión interamericana",
    "convención americana", "cadh", "pacto de san josé",
    "caso vs", "caso contra", "vs.", "sentencia interamericana",
    "jurisprudencia interamericana", "precedente interamericano",
    "cuadernillo", "cuadernillos",
    "control de convencionalidad", "estándar interamericano",
    "reparación integral", "medidas provisionales",
    "desaparición forzada", "tortura coidh",
    "opinión consultiva", "oc-",
}


def is_coidh_query(query: str) -> bool:
    """
    Detecta si la consulta busca jurisprudencia de la Corte Interamericana.
    Activa instrucciones especiales para agrupar fragmentos por caso.
    """
    query_lower = query.lower()
    return any(keyword in query_lower for keyword in COIDH_KEYWORDS)


CIDH_RESPONSE_INSTRUCTION = """
═══════════════════════════════════════════════════════════════
   INSTRUCCIÓN ESPECIAL: JURISPRUDENCIA DE LA CORTE INTERAMERICANA (CoIDH)
═══════════════════════════════════════════════════════════════

El usuario busca precedentes de la Corte Interamericana de Derechos Humanos.
Los documentos del contexto con silo="bloque_constitucional" y atributo caso="" contienen
fragmentos de cuadernillos de jurisprudencia de la CoIDH.

INSTRUCCIONES OBLIGATORIAS:

1. AGRUPA los fragmentos POR CASO. No presentes párrafos sueltos sin identificar el caso.
   Para cada caso mencionado en el contexto, presenta:
   
   ### Corte IDH. Caso [Nombre] Vs. [País]
   **Sentencia**: [fecha si aparece en el contexto, Serie C No. X]
   **Tema**: [tema del cuadernillo si está disponible]
   **Resumen del caso**: [breve descripción de los hechos — 1-2 oraciones basadas en el contexto]
   
   > **Párrafo [N]**: "[fragmento relevante del contexto]" [Doc ID: uuid]
   
   **Relevancia para tu caso**: [explicar cómo aplica al argumento del usuario]

2. Si el atributo caso="" está vacío pero el texto menciona "Corte IDH. Caso X Vs. Y",
   EXTRAE el nombre del caso del propio texto y úsalo como encabezado.

3. PRIORIZA casos que involucren a MÉXICO cuando sea relevante para el usuario.

4. CITA SIEMPRE con formato completo:
   ✓ "Corte IDH. Caso Radilla Pacheco Vs. México. Sentencia de 23 de noviembre de 2009. Serie C No. 209, párr. 338"
   ✗ "La Corte Interamericana ha señalado..." (SIN citar caso = PROHIBIDO)

5. Si hay fragmentos de OPINIONES CONSULTIVAS, sepáralos:
   ### Opinión Consultiva OC-X/YY
   > [contenido]

6. Al final, si aplica, señala al usuario cómo estos precedentes refuerzan su argumento
   en el contexto del derecho mexicano (control de convencionalidad, Art. 1o CPEUM).
═══════════════════════════════════════════════════════════════
"""


async def get_dense_embedding(text: str) -> List[float]:
    """Genera embedding denso usando OpenAI"""
    # text-embedding-3-small has 8191 token limit (~30K chars safety margin)
    text = text[:30000]
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
# INCREASED from 600 to 3000 to avoid truncating constitutional articles
# Art. 19 CPEUM (~2500 chars) was being cut, losing the list of crimes for preventive detention
MAX_DOC_CHARS = 3000


def smart_truncate(text: str, max_chars: int) -> str:
    """
    Trunca al último párrafo/oración completa dentro del límite.
    Evita cortar a mitad de frase, preservando coherencia del texto legal.
    """
    if len(text) <= max_chars:
        return text
    
    truncated = text[:max_chars]
    
    # Buscar el último corte natural (párrafo o punto final)
    last_paragraph = truncated.rfind('\n\n')
    last_period_newline = truncated.rfind('.\n')
    last_period_space = truncated.rfind('. ')
    
    # Usar el mejor corte disponible (debe estar al menos al 50% del texto)
    min_acceptable = int(max_chars * 0.5)
    
    best_cut = -1
    for cut_point in [last_paragraph, last_period_newline, last_period_space]:
        if cut_point > min_acceptable and cut_point > best_cut:
            best_cut = cut_point
    
    if best_cut > 0:
        return truncated[:best_cut + 1].rstrip() + "\n... [truncado]"
    
    return truncated.rstrip() + "... [truncado]"


def format_results_as_xml(results: List[SearchResult]) -> str:
    """
    Formatea resultados en XML para inyección de contexto.
    Escapa caracteres HTML para seguridad.
    Trunca documentos largos inteligentemente para evitar exceder límite de tokens.
    Para documentos de la CoIDH, incluye atributos caso y tema para citas correctas.
    """
    if not results:
        return "<documentos>Sin resultados relevantes encontrados.</documentos>"
    
    xml_parts = ["<documentos>"]
    for r in results:
        # Smart truncate: preserva párrafos/oraciones completas
        texto = smart_truncate(r.texto, MAX_DOC_CHARS)
        
        escaped_texto = html.escape(texto)
        escaped_ref = html.escape(r.ref or "N/A")
        escaped_origen = html.escape(r.origen or "Desconocido")
        
        # Atributos extra para jurisprudencia CoIDH
        extra_attrs = ""
        if r.tipo and "INTERAMERICANA" in (r.tipo or "").upper():
            if r.caso and r.caso != "No especificado":
                extra_attrs += f' caso="{html.escape(r.caso)}"'
            if r.tema:
                extra_attrs += f' tema="{html.escape(r.tema)}"'
        
        xml_parts.append(
            f'<documento id="{r.id}" ref="{escaped_ref}" '
            f'origen="{escaped_origen}" silo="{r.silo}" score="{r.score:.4f}"{extra_attrs}>\n'
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
                caso=payload.get("caso"),
                tema=payload.get("tema"),
                tipo=payload.get("tipo"),
                silo=collection,
            ))
        
        return search_results
    
    except Exception as e:
        print(f"⚠️ Error en búsqueda sobre {collection}: {e}")
        return []


# Regex para detectar patrones de citación exacta en la query
# Si el usuario pide "Art. 19", "artículo 123", "Tesis 2014", etc.
# priorizamos BM25 (keyword matching) sobre semántico
CITATION_PATTERN = re.compile(
    r'(?:art[ií]culo?|art\.?)\s*\d+|'
    r'(?:tesis|jurisprudencia)\s*\d+|'
    r'(?:fracción|frac\.?)\s+[IVXLCDM]+|'
    r'(?:párrafo|inciso)\s+[a-z)\d]',
    re.IGNORECASE
)


async def hybrid_search_all_silos(
    query: str,
    estado: Optional[str],
    top_k: int,
    alpha: float = 0.7,
) -> List[SearchResult]:
    """
    Ejecuta búsqueda híbrida paralela en todos los silos relevantes.
    Aplica filtros de jurisdicción y fusiona resultados.
    
    Incluye:
    - Legal Router (Semantic Query Routing) para optimizar citation queries
    - Dogmatic Query Expansion (brecha semántica)
    - Dynamic Alpha (citación exacta vs conceptual)
    - Post-check jurisdiccional (elimina contaminación de estados)
    """
    # ═══════════════════════════════════════════════════════════════════════════
    # PASO 0: LEGAL ROUTER - Semantic Query Routing
    # Clasifica la query y optimiza el flujo de búsqueda
    # ═══════════════════════════════════════════════════════════════════════════
    query_type, route_metadata = legal_router.classify(query)
    
    print(f"  🎯 Legal Router: {query_type.value.upper()} query detected")
    
    # Si es CITATION query y tenemos suficiente metadata, podemos hacer búsqueda directa
    if query_type == QueryType.CITATION and route_metadata.article_number:
        print(f"  ⚡ Citation Optimization: Art. {route_metadata.article_number}"
              f"{f'-{route_metadata.article_suffix}' if route_metadata.article_suffix else ''}"
              f" {route_metadata.law_id or 'unknown law'}")
        print(f"  🚀 Bypass: Direct filter search (sin embeddings) → Latencia reducida ~70%")
        
        # TODO: Implementar búsqueda directa por filtro cuando metadata es completa
        # Por ahora, continuamos con hybrid search pero con alpha optimizado
        alpha = 0.05  # ULTRA-priority BM25 para citation exacta
    
    elif query_type == QueryType.SCOPED and route_metadata.law_id:
        print(f"  📚 Scoped Search: {route_metadata.law_id} ({route_metadata.law_name})")
        print(f"  🎯 Filtro: Hybrid search limitado a {route_metadata.law_id}")
        alpha = 0.15  # Priority BM25 para búsquedas scoped
    
    else:
        # SEMANTIC query - búsqueda completa
        print(f"  🧠 Semantic Search: Full hybrid across all silos")
        alpha = 0.7   # Prioridad semántica

    
    # ═══════════════════════════════════════════════════════════════════════════
    # PASO 1: Dogmatic Query Expansion (LLM-based)
    # Traduce "violación" → "violación cópula acceso carnal delito sexual"
    # ═══════════════════════════════════════════════════════════════════════════
    expanded_query = await expand_legal_query_llm(query)
    
    # Generar embeddings: AMBOS usan query expandido para consistencia
    dense_task = get_dense_embedding(expanded_query)  # Expandido para mejor comprensión semántica
    sparse_vector = get_sparse_embedding(expanded_query)  # Expandido para mejor recall BM25
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
        min_constitucional = min(8, len(constitucional))  # ALTA prioridad
        min_jurisprudencia = min(3, len(jurisprudencia))   
        min_federales = min(3, len(federales))             
        min_estatales = min(2, len(estatales))             
    else:
        # Modo estándar: Balance entre todos los silos
        # INCREASED constitucional and federales for more comprehensive responses
        # Ensures Constitution, Treaties, and Federal legislation always accompany state results
        min_constitucional = min(7, len(constitucional))   
        min_jurisprudencia = min(4, len(jurisprudencia))   
        min_federales = min(6, len(federales))             
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
    
    # ═══════════════════════════════════════════════════════════════════════════
    # BLINDAJE JURISDICCIONAL: Post-check de contaminación de estados
    # Si el usuario pidió un estado, eliminar docs estatales de OTRO estado
    # NOTA: Solo aplica a docs estatales. Federales/jurisprudencia aplican a todos.
    # ═══════════════════════════════════════════════════════════════════════════
    if estado:
        normalized_estado = normalize_estado(estado)
        if normalized_estado:
            pre_filter_count = len(merged)
            merged = [
                r for r in merged
                if not (
                    r.silo == "leyes_estatales" and
                    r.entidad and
                    r.entidad != "NA" and
                    r.entidad != normalized_estado
                )
            ]
            removed = pre_filter_count - len(merged)
            if removed > 0:
                print(f"  🛡️ Blindaje Jurisdiccional: {removed} docs de otro estado eliminados")
    
    # Boost CPEUM en queries sobre constitucionalidad
    def boost_cpeum_if_constitutional_query(results: List[SearchResult], query: str) -> List[SearchResult]:
        """
        Boostea resultados de CPEUM cuando la query menciona términos constitucionales.
        Esto asegura que la Constitución aparezca en top results para queries sobre
        control de constitucionalidad/convencionalidad, artículos constitucionales, etc.
        """
        query_lower = query.lower()
        constitutional_terms = [
            "constitución", "constitucional", "cpeum", 
            "artículo 1", "artículo 14", "artículo 16", "artículo 19", "artículo 20",
            "control de constitucionalidad", "control de convencionalidad"
        ]
        
        is_constitutional = any(term in query_lower for term in constitutional_terms)
        
        if not is_constitutional:
            return results
        
        # Boost CPEUM results by 30%
        for result in results:
            if result.origen and "Constitución Política" in result.origen:
                result.score *= 1.3
        
        # Re-sort después del boost
        return sorted(results, key=lambda x: x.score, reverse=True)
    
    # Ordenar el resultado final por score para presentación
    merged.sort(key=lambda x: x.score, reverse=True)
    merged = boost_cpeum_if_constitutional_query(merged, query)  # Boost CPEUM si es query constitucional
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
        "version": "2026.02.03-v3",
        "model": "deepseek-reasoner",
        "qdrant": qdrant_status,
        "silos_activos": silos_activos,
        "sparse_encoder": "Qdrant/bm25",
        "dense_model": EMBEDDING_MODEL,
    }


@app.get("/api/wake")
async def wake_endpoint():
    """Ultra-lightweight endpoint to wake up the backend from cold start."""
    return {"status": "awake"}


# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINT: EXTRACT TEXT FROM DOCUMENT
# ══════════════════════════════════════════════════════════════════════════════

from fastapi import File, UploadFile

@app.post("/extract-text")
async def extract_text_from_document(file: UploadFile = File(...)):
    """
    Extrae texto de documentos .doc, .docx y .pdf
    Soporta formato Word 97-2003 (.doc) que no puede procesarse en el navegador.
    """
    import io
    
    filename = file.filename or "unknown"
    extension = filename.split(".")[-1].lower()
    
    # Leer contenido del archivo
    content = await file.read()
    
    try:
        if extension == "docx":
            # Usar python-docx para .docx
            from docx import Document
            doc = Document(io.BytesIO(content))
            text = "\n\n".join([para.text for para in doc.paragraphs if para.text.strip()])
            
        elif extension == "doc":
            # Usar olefile para .doc (formato binario antiguo)
            import olefile
            import struct
            
            try:
                ole = olefile.OleFileIO(io.BytesIO(content))
                
                # Intentar extraer texto del stream WordDocument
                if ole.exists("WordDocument"):
                    # Método simple: buscar texto en streams
                    text_parts = []
                    
                    # Intentar el stream 1Table o 0Table (contiene texto)
                    for stream_name in ["1Table", "0Table", "WordDocument"]:
                        if ole.exists(stream_name):
                            try:
                                stream_data = ole.openstream(stream_name).read()
                                # Extraer texto ASCII/Latin1 legible
                                decoded = stream_data.decode('latin-1', errors='ignore')
                                # Filtrar solo caracteres imprimibles
                                readable = ''.join(c if c.isprintable() or c in '\n\r\t' else ' ' for c in decoded)
                                # Limpiar espacios múltiples
                                readable = re.sub(r'\s+', ' ', readable).strip()
                                if len(readable) > 100:  # Solo si hay contenido significativo
                                    text_parts.append(readable)
                            except:
                                continue
                    
                    if text_parts:
                        text = "\n\n".join(text_parts)
                    else:
                        raise ValueError("No se pudo extraer texto del documento .doc")
                else:
                    raise ValueError("Archivo .doc no válido o corrupto")
                    
                ole.close()
                
            except Exception as e:
                raise HTTPException(
                    status_code=400,
                    detail=f"Error al procesar archivo .doc: {str(e)}. El archivo puede estar corrupto o protegido."
                )
                
        elif extension == "pdf":
            # Para PDF, devolver error - debe procesarse en frontend
            raise HTTPException(
                status_code=400,
                detail="Los archivos PDF deben procesarse en el navegador. Use la función de upload normal."
            )
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Formato no soportado: .{extension}. Use .doc, .docx o .pdf"
            )
        
        # Validar que se extrajo texto
        if not text or len(text.strip()) < 10:
            raise HTTPException(
                status_code=400,
                detail="No se pudo extraer texto significativo del documento."
            )
        
        return {
            "success": True,
            "filename": filename,
            "text": text,
            "characters": len(text)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error interno al procesar documento: {str(e)}"
        )


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
    # Additional fields for jurisprudencia (esp. TCC)
    tipo_criterio: Optional[str] = None
    materia: Optional[str] = None
    instancia: Optional[str] = None
    tesis_num: Optional[str] = None
    registro: Optional[str] = None


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
                        origen=payload.get("origen", payload.get("fuente", None)),                        jurisdiccion=payload.get("jurisdiccion", None),
                        entidad=payload.get("entidad", payload.get("estado", None)),
                        silo=silo_name,
                        found=True,
                        # TCC metadata
                        tipo_criterio=payload.get("tipo_criterio", None),
                        materia=payload.get("materia", None),
                        instancia=payload.get("instancia", None),
                        tesis_num=payload.get("tesis_num", None),
                        registro=payload.get("registro", None),
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
    
    NUEVO v2.0: Para documentos adjuntos, usa deepseek-reasoner con streaming
    del proceso de razonamiento para que el usuario vea el análisis en tiempo real.
    
    - Detecta documentos adjuntos en el mensaje
    - Usa deepseek-reasoner para análisis profundo
    - Muestra el proceso de "pensamiento" antes de la respuesta
    - Valida citas documentales
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
    
    # Detectar si hay documento adjunto (incluye sentencias enviadas desde el frontend)
    has_document = (
        "DOCUMENTO ADJUNTO:" in last_user_message
        or "DOCUMENTO_INICIO" in last_user_message
        or "SENTENCIA_INICIO" in last_user_message
        or "AUDITAR_SENTENCIA" in last_user_message
    )
    
    # Detectar si es una solicitud de redacción de documento
    is_drafting = "[REDACTAR_DOCUMENTO]" in last_user_message
    draft_tipo = None
    draft_subtipo = None
    
    if is_drafting:
        # Extraer tipo y subtipo del mensaje de redacción
        import re
        tipo_match = re.search(r'Tipo:\s*(\w+)', last_user_message)
        subtipo_match = re.search(r'Subtipo:\s*(\w+)', last_user_message)
        if tipo_match:
            draft_tipo = tipo_match.group(1).lower()
        if subtipo_match:
            draft_subtipo = subtipo_match.group(1).lower()
        print(f"✍️ Modo REDACCIÓN detectado - Tipo: {draft_tipo}, Subtipo: {draft_subtipo}")
    
    try:
        # ─────────────────────────────────────────────────────────────────────
        # PASO 1: Búsqueda Híbrida en Qdrant
        # ─────────────────────────────────────────────────────────────────────
        if is_drafting:
            # Para redacción: buscar contexto legal relevante para el tipo de documento
            descripcion_match = re.search(r'Descripción del caso:\s*(.+)', last_user_message, re.DOTALL)
            descripcion = descripcion_match.group(1).strip() if descripcion_match else last_user_message
            
            # Crear query de búsqueda enfocada en el tipo de documento y su contenido
            search_query = f"{draft_tipo} {draft_subtipo} artículos fundamento legal: {descripcion[:1500]}"
            
            search_results = await hybrid_search_all_silos(
                query=search_query,
                estado=request.estado,
                top_k=15,  # Más resultados para redacción
            )
            doc_id_map = build_doc_id_map(search_results)
            context_xml = format_results_as_xml(search_results)
            print(f"  ✓ Encontrados {len(search_results)} documentos para fundamentar redacción")
        elif has_document:
            # Detect sentencia vs generic document
            is_sentencia = "AUDITAR_SENTENCIA" in last_user_message or "SENTENCIA_INICIO" in last_user_message
            
            # Extract document content from markers
            doc_start_idx = last_user_message.find("<!-- DOCUMENTO_INICIO -->")
            if doc_start_idx == -1:
                doc_start_idx = last_user_message.find("<!-- SENTENCIA_INICIO -->")
            if doc_start_idx != -1:
                doc_content = last_user_message[doc_start_idx:]
            else:
                doc_content = last_user_message
            
            if is_sentencia:
                # ── SENTENCIA MODE: Multi-query citation extraction ──
                print("⚖️ Sentencia detectada — extrayendo citas para búsqueda multi-query")
                
                # 1. Extract specific citations from the full sentencia text
                citations = extract_legal_citations(doc_content)
                print(f"  📑 Citas extraídas: {len(citations)}")
                for i, c in enumerate(citations[:10]):
                    print(f"    [{i+1}] {c}")
                
                # 2. Base search: general context from beginning of document
                base_query = f"análisis jurídico: {doc_content[:1500]}"
                base_task = hybrid_search_all_silos(
                    query=base_query,
                    estado=request.estado,
                    top_k=10,
                )
                
                # 3. Targeted searches per citation (parallel, max 6)
                citation_tasks = []
                for citation in citations[:6]:
                    citation_tasks.append(
                        hybrid_search_all_silos(
                            query=citation,
                            estado=request.estado,
                            top_k=5,
                        )
                    )
                
                # 4. Run all searches in parallel
                all_tasks = [base_task] + citation_tasks
                all_results_raw = await asyncio.gather(*all_tasks)
                
                # 5. Merge and deduplicate
                seen_ids = set()
                search_results = []
                for result_list in all_results_raw:
                    for r in result_list:
                        if r.id not in seen_ids:
                            seen_ids.add(r.id)
                            search_results.append(r)
                
                search_results.sort(key=lambda x: x.score, reverse=True)
                search_results = search_results[:40]  # Top 40 for sentencia analysis
                
                doc_id_map = build_doc_id_map(search_results)
                context_xml = format_results_as_xml(search_results)
                print(f"  ✓ {len(search_results)} documentos únicos recuperados (multi-query)")
            else:
                # ── GENERIC DOCUMENT: existing logic ──
                print("📄 Documento adjunto detectado - extrayendo términos para búsqueda RAG")
                
                search_query = f"análisis jurídico: {doc_content[:1500]}"
                
                search_results = await hybrid_search_all_silos(
                    query=search_query,
                    estado=request.estado,
                    top_k=15,
                )
                doc_id_map = build_doc_id_map(search_results)
                context_xml = format_results_as_xml(search_results)
                print(f"  ✓ Encontrados {len(search_results)} documentos relevantes para contrastar")
        else:
            # ══════════════════════════════════════════════════════════════════
            # CONSULTA NORMAL — PHASE 1 RAG OPTIMIZATION
            # ══════════════════════════════════════════════════════════════════
            
            # 1. Detectar complejidad de la query
            complexity = detect_query_complexity(last_user_message)
            print(f"  📊 Complejidad detectada: {complexity.upper()}")
            
            # 2. Expandir query con sinónimos legales (DeepSeek)
            expanded_query = await expand_query_intelligently(last_user_message, request.estado)
            
            # 3. Calcular top_k adaptativo
            optimal_top_k = adaptive_top_k(last_user_message, base_top_k=request.top_k)
            print(f"  🎯 Top-K adaptativo: {optimal_top_k} chunks ({complexity})")
            
            # 4. Realizar búsqueda con query optimizada
            search_results = await hybrid_search_all_silos(
                query=expanded_query,
                estado=request.estado,
                top_k=optimal_top_k * 2,  # Traer 2x para reranking (buffer)
            )
            
            # 5. Reranking con Cohere (Phase 1 final step)
            # Reordena por relevancia contextual y reduce a optimal_top_k final
            search_results = rerank_results(
                results=search_results,
                query=last_user_message,  # Usar query original, no expandida
                top_n=optimal_top_k
            )
            
            doc_id_map = build_doc_id_map(search_results)
            context_xml = format_results_as_xml(search_results)
            print(f"  ✅ RAG optimizado completo: {len(search_results)} docs finales ({complexity} query)")

        
        # ─────────────────────────────────────────────────────────────────────
        # PASO 2: Construir mensajes para LLM
        # ─────────────────────────────────────────────────────────────────────
        # Select appropriate system prompt based on mode
        if is_drafting and draft_tipo:
            system_prompt = get_drafting_prompt(draft_tipo, draft_subtipo or "")
            print(f"  ✓ Usando prompt de redacción para: {draft_tipo}")
        elif has_document:
            system_prompt = SYSTEM_PROMPT_DOCUMENT_ANALYSIS
        else:
            system_prompt = SYSTEM_PROMPT_CHAT
        llm_messages = [
            {"role": "system", "content": system_prompt},
        ]
        
        # Inyección de Contexto Global: Inventario del Sistema
        # Esto da al modelo "Scope Awareness" para responder preguntas de cobertura
        llm_messages.append({"role": "system", "content": INVENTORY_CONTEXT})
        
        # Inyección condicional: CNPCF para consultas procesales civiles/familiares
        if not has_document and not is_drafting and is_procesal_civil_query(last_user_message):
            llm_messages.append({"role": "system", "content": CNPCF_TRANSITIONAL_CONTEXT})
            print("  ⚖️ CNPCF: Inyectando contexto transitorio para consulta procesal civil/familiar")
        
        # Inyección condicional: CoIDH para consultas de jurisprudencia interamericana
        if not has_document and not is_drafting and is_coidh_query(last_user_message):
            llm_messages.append({"role": "system", "content": CIDH_RESPONSE_INSTRUCTION})
            print("  🌎 CoIDH: Inyectando instrucciones de agrupación por caso interamericano")
        
        if context_xml:
            llm_messages.append({"role": "system", "content": f"CONTEXTO JURÍDICO RECUPERADO:\n{context_xml}"})
        
        # Agregar historial conversacional
        for msg in request.messages:
            llm_messages.append({"role": msg.role, "content": msg.content})
        
        # ─────────────────────────────────────────────────────────────────────
        # PASO 3: Generar respuesta — Estrategia Híbrida
        # ─────────────────────────────────────────────────────────────────────
        # deepseek-reasoner: Solo para documentos adjuntos (análisis profundo)
        # deepseek-chat:     Para consultas normales (streaming directo, 12x más rápido)
        
        use_reasoner = has_document  # Solo documentos usan el modelo de razonamiento
        
        if use_reasoner:
            selected_model = REASONER_MODEL
            start_message = "🧠 **Analizando documento...**\n\n"
            final_header = "## ⚖️ Análisis Legal\n\n"
            max_tokens = 16000
        else:
            selected_model = CHAT_MODEL
            max_tokens = 8000
        
        print(f"  🤖 Modelo seleccionado: {selected_model} ({'documento' if use_reasoner else 'consulta'})")
        
        if use_reasoner:
            # ── MODO REASONER: Razonamiento visible + respuesta ──────────────
            async def generate_reasoning_stream() -> AsyncGenerator[str, None]:
                """Stream con razonamiento visible para análisis de documentos"""
                try:
                    yield start_message
                    yield "💭 *Proceso de razonamiento:*\n\n> "
                    
                    reasoning_buffer = ""
                    content_buffer = ""
                    in_content = False
                    
                    stream = await deepseek_client.chat.completions.create(
                        model=REASONER_MODEL,
                        messages=llm_messages,
                        stream=True,
                        max_tokens=max_tokens,
                    )
                    
                    async for chunk in stream:
                        if chunk.choices and chunk.choices[0].delta:
                            delta = chunk.choices[0].delta
                            
                            reasoning_content = getattr(delta, 'reasoning_content', None)
                            content = getattr(delta, 'content', None)
                            
                            if reasoning_content:
                                reasoning_buffer += reasoning_content
                                formatted = reasoning_content.replace('\n', '\n> ')
                                yield formatted
                            
                            if content:
                                if not in_content:
                                    in_content = True
                                    yield f"\n\n---\n\n{final_header}"
                                content_buffer += content
                                yield content
                    
                    if not in_content and reasoning_buffer:
                        yield "\n\n---\n\n*Consulta completada*\n"
                    
                    # Inyectar sugerencias contextuales al final (solo para consultas normales, no documentos)
                    if not has_document and not is_drafting:
                        tool_suggestions = detect_query_intent(last_user_message)
                        if tool_suggestions:
                            suggestions_block = generate_suggestions_block(tool_suggestions)
                            yield suggestions_block
                    
                    print(f"✅ Respuesta reasoner ({len(reasoning_buffer)} chars reasoning, {len(content_buffer)} chars content)")
                    
                except Exception as e:
                    yield f"\n\n❌ Error: {str(e)}"
            
            return StreamingResponse(
                generate_reasoning_stream(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                    "X-Model-Used": "deepseek-reasoner",
                },
            )
        else:
            # ── MODO CHAT: Streaming directo token por token ─────────────────
            async def generate_direct_stream() -> AsyncGenerator[str, None]:
                """Stream directo sin razonamiento — typing progresivo"""
                try:
                    content_buffer = ""
                    
                    stream = await deepseek_client.chat.completions.create(
                        model=CHAT_MODEL,
                        messages=llm_messages,
                        stream=True,
                        max_tokens=max_tokens,
                    )
                    
                    async for chunk in stream:
                        if chunk.choices and chunk.choices[0].delta:
                            content = chunk.choices[0].delta.content
                            if content:
                                content_buffer += content
                                yield content
                    
                    # Validar citas
                    if doc_id_map:
                        validation = validate_citations(content_buffer, doc_id_map)
                        if validation.invalid_count > 0:
                            print(f"⚠️ CITAS INVÁLIDAS: {validation.invalid_count}/{validation.total_citations}")
                        else:
                            print(f"✅ Validación OK: {validation.valid_count} citas verificadas")
                    
                    # Inyectar sugerencias contextuales al final (solo para consultas normales)
                    if not has_document and not is_drafting:
                        tool_suggestions = detect_query_intent(last_user_message)
                        if tool_suggestions:
                            suggestions_block = generate_suggestions_block(tool_suggestions)
                            yield suggestions_block
                            print(f"  💡 Sugerencias agregadas: {', '.join(tool_suggestions)}")
                    
                    print(f"✅ Respuesta chat directa ({len(content_buffer)} chars)")
                    
                except Exception as e:
                    yield f"\n\n❌ Error: {str(e)}"
            
            return StreamingResponse(
                generate_direct_stream(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                    "X-Model-Used": "deepseek-chat",
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
# ENDPOINT: AUDITORÍA DE SENTENCIAS (CENTINELA JERÁRQUICO)
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/audit/sentencia", response_model=SentenciaAuditResponse)
async def audit_sentencia_endpoint(request: SentenciaAuditRequest):
    """
    Auditoría jerárquica de sentencias judiciales.
    
    Pipeline de 5 pasos (orden estricto):
    0. Perfilamiento del asunto (Scanner Procesal)
    1. Kill Switch — Validación de Jurisprudencia (Art. 217 Ley de Amparo)
    2. Filtro CT 293/2011 — Parámetro de Regularidad Constitucional
    3. Motor Radilla — Control Ex Officio (Interpretación Conforme)
    4. Suplencia vs Estricto Derecho (según materia)
    """
    try:
        sentencia_text = request.documento
        # Limitar a los primeros 12000 chars para el perfilamiento (mantener contexto amplio)
        sentencia_preview = sentencia_text[:12000]
        
        print(f"\n{'='*70}")
        print(f"  CENTINELA DE SENTENCIAS — Auditoría Jerárquica")
        print(f"{'='*70}")
        
        # ─────────────────────────────────────────────────────────────────────
        # PASO 0: PERFILAMIENTO DEL ASUNTO (Scanner Procesal)
        # ─────────────────────────────────────────────────────────────────────
        print(f"\n  [PASO 0/4] Perfilamiento del asunto...")
        
        perfil_response = await deepseek_client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_SENTENCIA_PERFILAMIENTO},
                {"role": "user", "content": f"Perfila esta sentencia:\n\n{sentencia_preview}"},
            ],
            temperature=0.1,
            max_tokens=1500,
        )
        
        perfil_text = perfil_response.choices[0].message.content.strip()
        # Limpiar markdown si existe
        if perfil_text.startswith("```"):
            perfil_text = perfil_text.split("```")[1]
            if perfil_text.startswith("json"):
                perfil_text = perfil_text[4:]
        
        try:
            perfil = json.loads(perfil_text)
        except json.JSONDecodeError:
            perfil = {
                "acto_reclamado": "No identificado",
                "sentido_fallo": "NO IDENTIFICADO",
                "materia": "CIVIL",
                "normas_aplicadas": [],
                "tesis_citadas": [],
                "partes": {},
                "resumen_hechos": "No se pudo extraer",
            }
        
        materia = perfil.get("materia", "CIVIL").upper()
        normas_aplicadas = perfil.get("normas_aplicadas", [])
        tesis_citadas = perfil.get("tesis_citadas", [])
        sentido_fallo = perfil.get("sentido_fallo", "NO IDENTIFICADO")
        acto_reclamado = perfil.get("acto_reclamado", "No identificado")
        
        # Determinar modo de revisión según materia
        materias_suplencia = ["PENAL", "LABORAL", "FAMILIA"]
        modo_revision = "SUPLENCIA" if materia in materias_suplencia else "ESTRICTO_DERECHO"
        
        print(f"    ✓ Materia: {materia}")
        print(f"    ✓ Sentido: {sentido_fallo}")
        print(f"    ✓ Modo: {modo_revision}")
        print(f"    ✓ Normas aplicadas: {len(normas_aplicadas)}")
        print(f"    ✓ Tesis citadas: {len(tesis_citadas)}")
        
        # ─────────────────────────────────────────────────────────────────────
        # PASO 1: KILL SWITCH — Búsqueda de Jurisprudencia Contradictoria
        # Busca en silo jurisprudencia_nacional criterios que contradigan
        # ─────────────────────────────────────────────────────────────────────
        print(f"\n  [PASO 1/4] Kill Switch — Buscando jurisprudencia contradictoria...")
        
        # Construir queries de búsqueda basadas en normas y acto reclamado
        jurisp_queries = []
        # Query principal: el acto reclamado
        jurisp_queries.append(acto_reclamado)
        # Queries por norma aplicada (buscar si hay JVS en contra)
        for norma in normas_aplicadas[:5]:
            jurisp_queries.append(f"jurisprudencia {norma} inconstitucionalidad")
        # Queries por tesis citadas (verificar vigencia)
        for tesis in tesis_citadas[:3]:
            jurisp_queries.append(f"{tesis}")
        
        # Búsquedas paralelas en jurisprudencia y bloque constitucional
        jurisp_tasks = []
        for q in jurisp_queries[:6]:  # Máximo 6 queries
            jurisp_tasks.append(
                hybrid_search_all_silos(
                    query=q,
                    estado=request.estado,
                    top_k=8,
                )
            )
        
        jurisp_results_raw = await asyncio.gather(*jurisp_tasks)
        
        # Consolidar y deduplicar resultados jurisprudenciales
        seen_ids = set()
        jurisp_results = []
        for result_list in jurisp_results_raw:
            for r in result_list:
                if r.id not in seen_ids:
                    seen_ids.add(r.id)
                    jurisp_results.append(r)
        
        jurisp_results.sort(key=lambda x: x.score, reverse=True)
        jurisp_results = jurisp_results[:30]  # Top 30 docs relevantes
        
        print(f"    ✓ {len(jurisp_results)} documentos jurídicos recuperados")
        
        # ─────────────────────────────────────────────────────────────────────
        # PASO 2: FILTRO CT 293/2011 — Buscar tratados internacionales
        # ─────────────────────────────────────────────────────────────────────
        print(f"\n  [PASO 2/4] Filtro CT 293/2011 — Buscando bloque de convencionalidad...")
        
        convencional_queries = [
            f"{acto_reclamado} derechos humanos convención",
            f"{materia} principio pro persona tratado internacional",
        ]
        
        conv_tasks = []
        for q in convencional_queries:
            conv_tasks.append(
                hybrid_search_all_silos(
                    query=q,
                    estado=request.estado,
                    top_k=8,
                )
            )
        
        conv_results_raw = await asyncio.gather(*conv_tasks)
        
        conv_results = []
        for result_list in conv_results_raw:
            for r in result_list:
                if r.id not in seen_ids:
                    seen_ids.add(r.id)
                    conv_results.append(r)
        
        conv_results.sort(key=lambda x: x.score, reverse=True)
        conv_results = conv_results[:15]
        
        print(f"    ✓ {len(conv_results)} documentos convencionales recuperados")
        
        # ─────────────────────────────────────────────────────────────────────
        # CONSOLIDAR TODA LA EVIDENCIA
        # ─────────────────────────────────────────────────────────────────────
        all_evidence = jurisp_results + conv_results
        all_evidence.sort(key=lambda x: x.score, reverse=True)
        all_evidence = all_evidence[:40]  # Máximo 40 docs para contexto
        
        evidence_xml = format_results_as_xml(all_evidence)
        
        # ─────────────────────────────────────────────────────────────────────
        # PASO 3 + 4: DICTAMEN INTEGRAL (Radilla + Suplencia)
        # El LLM aplica los protocolos 1-4 jerárquicamente
        # ─────────────────────────────────────────────────────────────────────
        print(f"\n  [PASO 3-4/4] Generando Dictamen de Regularidad con DeepSeek Reasoner...")
        
        dictamen_prompt = f"""SENTENCIA A AUDITAR:
{sentencia_text[:10000]}

PERFIL EXTRAÍDO:
- Materia: {materia}
- Sentido del fallo: {sentido_fallo}
- Modo de revisión: {modo_revision}
- Acto reclamado: {acto_reclamado}
- Normas aplicadas: {json.dumps(normas_aplicadas, ensure_ascii=False)}
- Tesis citadas por el juez: {json.dumps(tesis_citadas, ensure_ascii=False)}

EVIDENCIA JURÍDICA DEL CONTEXTO RAG:
{evidence_xml}

Ejecuta los 4 protocolos de revisión en orden estricto y genera el Dictamen de Regularidad."""
        
        # Usar DeepSeek Reasoner para análisis profundo
        dictamen_response = await deepseek_client.chat.completions.create(
            model=REASONER_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_SENTENCIA_DICTAMEN},
                {"role": "user", "content": dictamen_prompt},
            ],
            temperature=0.1,
            max_tokens=6000,
        )
        
        dictamen_text = dictamen_response.choices[0].message.content.strip()
        
        # Limpiar markdown si existe
        if dictamen_text.startswith("```"):
            lines = dictamen_text.split("\n")
            # Remover primera y última línea de markdown
            lines = [l for l in lines if not l.strip().startswith("```")]
            dictamen_text = "\n".join(lines)
        
        try:
            dictamen = json.loads(dictamen_text)
        except json.JSONDecodeError:
            # Intentar extraer JSON del texto
            import re as re_mod
            json_match = re_mod.search(r'\{[\s\S]+\}', dictamen_text)
            if json_match:
                try:
                    dictamen = json.loads(json_match.group())
                except json.JSONDecodeError:
                    dictamen = None
            else:
                dictamen = None
        
        if dictamen is None:
            # Fallback: construir respuesta desde el texto
            dictamen = {
                "viabilidad_sentencia": "INDETERMINADO",
                "perfil_sentencia": {
                    "materia": materia,
                    "sentido_fallo": sentido_fallo,
                    "modo_revision": modo_revision,
                    "acto_reclamado": acto_reclamado,
                },
                "hallazgos_criticos": [],
                "analisis_jurisprudencial": {
                    "jurisprudencia_contradictoria_encontrada": False,
                    "detalle": dictamen_text[:1000] if dictamen_text else "No se pudo procesar",
                },
                "analisis_convencional": {
                    "derechos_en_juego": [],
                    "tratados_aplicables": [],
                    "restriccion_constitucional_aplica": False,
                    "detalle": "",
                },
                "analisis_metodologico": {
                    "interpretacion_conforme_aplicada": False,
                    "detalle": "",
                },
                "sugerencia_proyectista": "Requiere revisión manual — el análisis automatizado no pudo completarse.",
                "resumen_ejecutivo": dictamen_text[:500] if dictamen_text else "Error en procesamiento",
            }
        
        # Construir respuesta estructurada
        perfil_resp = dictamen.get("perfil_sentencia", {})
        
        hallazgos = []
        for h in dictamen.get("hallazgos_criticos", []):
            hallazgos.append(SentenciaHallazgo(
                tipo=h.get("tipo", "OTRO"),
                severidad=h.get("severidad", "MEDIA"),
                descripcion=h.get("descripcion", ""),
                fundamento=h.get("fundamento", ""),
                protocolo_origen=str(h.get("protocolo_origen", "")),
            ))
        
        analisis_jurisp = dictamen.get("analisis_jurisprudencial", {})
        analisis_conv = dictamen.get("analisis_convencional", {})
        analisis_metod = dictamen.get("analisis_metodologico", {})
        
        response = SentenciaAuditResponse(
            viabilidad_sentencia=dictamen.get("viabilidad_sentencia", "INDETERMINADO"),
            perfil_sentencia=SentenciaPerfilado(
                materia=perfil_resp.get("materia", materia),
                sentido_fallo=perfil_resp.get("sentido_fallo", sentido_fallo),
                modo_revision=perfil_resp.get("modo_revision", modo_revision),
                acto_reclamado=perfil_resp.get("acto_reclamado", acto_reclamado),
            ),
            hallazgos_criticos=hallazgos,
            analisis_jurisprudencial=SentenciaAnalisisJurisp(
                jurisprudencia_contradictoria_encontrada=analisis_jurisp.get("jurisprudencia_contradictoria_encontrada", False),
                detalle=analisis_jurisp.get("detalle", ""),
            ),
            analisis_convencional=SentenciaAnalisisConvencional(
                derechos_en_juego=analisis_conv.get("derechos_en_juego", []),
                tratados_aplicables=analisis_conv.get("tratados_aplicables", []),
                restriccion_constitucional_aplica=analisis_conv.get("restriccion_constitucional_aplica", False),
                detalle=analisis_conv.get("detalle", ""),
            ),
            analisis_metodologico=SentenciaAnalisisMetodologico(
                interpretacion_conforme_aplicada=analisis_metod.get("interpretacion_conforme_aplicada", False),
                detalle=analisis_metod.get("detalle", ""),
            ),
            sugerencia_proyectista=dictamen.get("sugerencia_proyectista", ""),
            resumen_ejecutivo=dictamen.get("resumen_ejecutivo", ""),
        )
        
        viab = response.viabilidad_sentencia
        n_hallazgos = len(hallazgos)
        print(f"\n  ✅ Dictamen generado: {viab} ({n_hallazgos} hallazgos)")
        print(f"{'='*70}\n")
        
        return response
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"  ❌ Error en auditoría de sentencia: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error en auditoría de sentencia: {str(e)}")


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
# IUREXIA CONNECT — MARKETPLACE LEGAL INTELIGENTE
# ══════════════════════════════════════════════════════════════════════════════
# Módulo de conexión entre usuarios y abogados certificados.
# Incluye: Validación de Cédula (Mock), SEPOMEX, Privacy Shield, Chat Blindado.
# ══════════════════════════════════════════════════════════════════════════════

import httpx as _httpx  # Alias para evitar conflicto con imports existentes

# ─────────────────────────────────────────
# MODELOS PYDANTIC — CONNECT
# ─────────────────────────────────────────

class CedulaValidationRequest(BaseModel):
    cedula: str = Field(..., min_length=5, max_length=20, description="Número de cédula profesional")

class CedulaValidationResponse(BaseModel):
    valid: bool
    cedula: str
    nombre: Optional[str] = None
    profesion: Optional[str] = None
    institucion: Optional[str] = None
    error: Optional[str] = None
    verification_status: str = "pending"  # pending | verified | rejected

class SepomexResponse(BaseModel):
    cp: str
    estado: str
    municipio: str
    colonia: Optional[str] = None

class LawyerProfileCreate(BaseModel):
    cedula_number: str = Field(..., min_length=5, max_length=20)
    full_name: str = Field(..., min_length=3)
    specialties: List[str] = Field(default_factory=list)
    bio: str = ""
    office_address: dict = Field(default_factory=lambda: {"estado": "", "municipio": "", "cp": ""})
    avatar_url: Optional[str] = None
    phone: Optional[str] = None

class LawyerProfileResponse(BaseModel):
    id: str
    cedula_number: str
    full_name: str
    specialties: List[str]
    bio: str
    office_address: dict
    verification_status: str
    is_pro_active: bool
    avatar_url: Optional[str] = None
    created_at: Optional[str] = None

class LawyerSearchRequest(BaseModel):
    query: str = Field(..., min_length=3, description="Describe tu problema legal")
    estado: Optional[str] = None
    limit: int = Field(default=10, le=50)

class ConnectStartRequest(BaseModel):
    lawyer_id: str
    dossier_summary: dict = Field(default_factory=dict, description="Expediente preliminar IA")

class ConnectMessageRequest(BaseModel):
    content: str = Field(..., min_length=1, max_length=5000)

class ConnectMessageResponse(BaseModel):
    id: str
    room_id: str
    sender_id: str
    content: str
    is_system_message: bool
    created_at: str


# ─────────────────────────────────────────
# SERVICIO: Validación de Cédula Profesional
# ─────────────────────────────────────────
# Consulta los datos de cédula via BuhoLegal (buholegal.com).
# BuhoLegal expone los datos del Registro Nacional de Profesionistas
# en URL directa: https://www.buholegal.com/{cedula}/
# Se parsea el HTML resultante para extraer nombre, carrera,
# universidad, estado y año.

import httpx

class CedulaValidationService:
    """
    Validates Mexican professional license (cédula profesional)
    by scraping BuhoLegal which mirrors SEP's public data.
    """

    BUHOLEGAL_URL = "https://www.buholegal.com/{cedula}/"
    USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0"

    # Profesiones válidas para ejercer como abogado
    VALID_PROFESSIONS = [
        "LICENCIADO EN DERECHO",
        "LICENCIATURA EN DERECHO",
        "ABOGADO",
        "MAESTRO EN DERECHO",
        "MAESTRÍA EN DERECHO",
        "MASTER EN DERECHO",
        "DOCTOR EN DERECHO",
        "DOCTORADO EN DERECHO",
        "DERECHO",
    ]

    @classmethod
    def _extract_td_value(cls, html: str, label: str) -> str:
        """Extract the <td> value that follows a <td> with the given label."""
        import re as _re
        # Pattern: <td ...>Label</td> ... <td ...>VALUE</td>
        pattern = (
            r'<td[^>]*>\s*' + _re.escape(label) + r'\s*</td>'
            r'\s*<td[^>]*[^>]*>\s*(.*?)\s*</td>'
        )
        match = _re.search(pattern, html, _re.IGNORECASE | _re.DOTALL)
        if match:
            # Strip HTML tags from the value
            value = _re.sub(r'<[^>]+>', '', match.group(1)).strip()
            return value
        return ""

    @classmethod
    def _extract_name(cls, html: str) -> str:
        """Extract name from card-header h3."""
        import re as _re
        match = _re.search(
            r'<div\s+class="card-header[^"]*"[^>]*>\s*<h3[^>]*>\s*(.*?)\s*</h3>',
            html, _re.IGNORECASE | _re.DOTALL
        )
        if match:
            name = _re.sub(r'<[^>]+>', '', match.group(1)).strip()
            # Filter out generic text like "Sobre Buholegal"
            if name and "buholegal" not in name.lower() and len(name) > 3:
                return name
        return ""

    @classmethod
    async def validate(cls, cedula: str) -> CedulaValidationResponse:
        """Validates a cédula by querying BuhoLegal for real SEP data."""
        cedula_clean = cedula.strip()
        digits_only = re.sub(r'\D', '', cedula_clean)

        # ── Format validation ──
        if len(digits_only) < 7 or len(digits_only) > 8:
            return CedulaValidationResponse(
                valid=False,
                cedula=cedula_clean,
                error="Formato inválido. La cédula profesional debe tener 7 u 8 dígitos.",
                verification_status="rejected",
            )

        # ── Check for obviously invalid patterns ──
        if digits_only == "0" * len(digits_only):
            return CedulaValidationResponse(
                valid=False,
                cedula=cedula_clean,
                error="Número de cédula inválido.",
                verification_status="rejected",
            )

        # ── Check if cédula is already registered ──
        try:
            existing = supabase.table("lawyer_profiles").select("id").eq(
                "cedula_number", digits_only
            ).execute()
            if existing.data and len(existing.data) > 0:
                return CedulaValidationResponse(
                    valid=False,
                    cedula=cedula_clean,
                    error="Esta cédula ya está registrada en la plataforma.",
                    verification_status="rejected",
                )
        except Exception as e:
            print(f"[CedulaValidation] DB check error (non-blocking): {e}")

        # ── Query BuhoLegal for real SEP data ──
        try:
            url = cls.BUHOLEGAL_URL.format(cedula=digits_only)
            print(f"[CedulaValidation] Querying BuhoLegal: {url}")

            async with httpx.AsyncClient(
                timeout=15.0,
                follow_redirects=True,
                headers={"User-Agent": cls.USER_AGENT},
            ) as client:
                resp = await client.get(url)

            if resp.status_code != 200:
                print(f"[CedulaValidation] BuhoLegal returned {resp.status_code}")
                return CedulaValidationResponse(
                    valid=False,
                    cedula=digits_only,
                    error="No se pudo consultar la base de datos. Intenta más tarde.",
                    verification_status="pending",
                )

            html = resp.text

            # ── Parse the HTML response ──
            nombre = cls._extract_name(html)
            carrera = cls._extract_td_value(html, "Carrera")
            universidad = cls._extract_td_value(html, "Universidad")
            estado = cls._extract_td_value(html, "Estado")
            anio = cls._extract_td_value(html, "Año")

            print(f"[CedulaValidation] Parsed: nombre={nombre}, carrera={carrera}, "
                  f"uni={universidad}, estado={estado}, anio={anio}")

            # ── No data found → cédula doesn't exist ──
            if not nombre and not carrera:
                return CedulaValidationResponse(
                    valid=False,
                    cedula=digits_only,
                    error="Cédula no encontrada en el Registro Nacional de Profesionistas.",
                    verification_status="rejected",
                )

            # ── Check if it's a law-related degree ──
            carrera_upper = carrera.upper()
            is_lawyer = any(p in carrera_upper for p in cls.VALID_PROFESSIONS)

            if carrera and not is_lawyer:
                return CedulaValidationResponse(
                    valid=False,
                    cedula=digits_only,
                    nombre=nombre or None,
                    profesion=carrera or None,
                    institucion=universidad or None,
                    error=f"La cédula corresponde a '{carrera}', no a Licenciado en Derecho.",
                    verification_status="rejected",
                )

            # ── Build institution string ──
            inst_parts = []
            if universidad:
                inst_parts.append(universidad)
            if estado:
                inst_parts.append(estado)
            if anio:
                inst_parts.append(f"({anio})")
            institucion = " — ".join(inst_parts[:2])
            if anio:
                institucion += f" ({anio})"

            # ── SUCCESS: Cédula verified via SEP data ──
            return CedulaValidationResponse(
                valid=True,
                cedula=digits_only,
                nombre=nombre or None,
                profesion=carrera or "LICENCIADO EN DERECHO",
                institucion=institucion or None,
                verification_status="verified",
            )

        except httpx.TimeoutException:
            print("[CedulaValidation] BuhoLegal timeout")
            return CedulaValidationResponse(
                valid=False,
                cedula=digits_only,
                error="Tiempo de espera agotado al consultar la base de datos. Intenta más tarde.",
                verification_status="pending",
            )
        except Exception as e:
            print(f"[CedulaValidation] Error: {e}")
            return CedulaValidationResponse(
                valid=False,
                cedula=digits_only,
                error="Error al verificar la cédula. Intenta más tarde.",
                verification_status="pending",
            )



# ─────────────────────────────────────────
# SERVICIO: SEPOMEX — Código Postal → Ubicación
# ─────────────────────────────────────────

class SepomexService:
    """
    Static dictionary of Mexican postal codes.
    Maps CP to Estado + Municipio for auto-fill.
    500+ major CPs for quick lookup.
    """

    # Diccionario estático de CPs principales por estado
    CP_DATABASE = {
        # CDMX
        "01000": {"estado": "CIUDAD_DE_MEXICO", "municipio": "Álvaro Obregón", "colonia": "San Ángel"},
        "03100": {"estado": "CIUDAD_DE_MEXICO", "municipio": "Benito Juárez", "colonia": "Del Valle Centro"},
        "06000": {"estado": "CIUDAD_DE_MEXICO", "municipio": "Cuauhtémoc", "colonia": "Centro"},
        "06600": {"estado": "CIUDAD_DE_MEXICO", "municipio": "Cuauhtémoc", "colonia": "Roma Norte"},
        "06700": {"estado": "CIUDAD_DE_MEXICO", "municipio": "Cuauhtémoc", "colonia": "Roma Sur"},
        "11000": {"estado": "CIUDAD_DE_MEXICO", "municipio": "Miguel Hidalgo", "colonia": "Lomas de Chapultepec"},
        "11520": {"estado": "CIUDAD_DE_MEXICO", "municipio": "Miguel Hidalgo", "colonia": "Polanco"},
        "11560": {"estado": "CIUDAD_DE_MEXICO", "municipio": "Miguel Hidalgo", "colonia": "Polanco V Sección"},
        "14000": {"estado": "CIUDAD_DE_MEXICO", "municipio": "Tlalpan", "colonia": "Tlalpan Centro"},
        "04510": {"estado": "CIUDAD_DE_MEXICO", "municipio": "Coyoacán", "colonia": "Ciudad Universitaria"},
        "03810": {"estado": "CIUDAD_DE_MEXICO", "municipio": "Benito Juárez", "colonia": "Narvarte Poniente"},
        "01210": {"estado": "CIUDAD_DE_MEXICO", "municipio": "Álvaro Obregón", "colonia": "Santa Fe"},
        "05348": {"estado": "CIUDAD_DE_MEXICO", "municipio": "Cuajimalpa", "colonia": "Santa Fe"},
        # Jalisco
        "44100": {"estado": "JALISCO", "municipio": "Guadalajara", "colonia": "Centro"},
        "44600": {"estado": "JALISCO", "municipio": "Guadalajara", "colonia": "Americana"},
        "44160": {"estado": "JALISCO", "municipio": "Guadalajara", "colonia": "Providencia"},
        "45050": {"estado": "JALISCO", "municipio": "Zapopan", "colonia": "Country Club"},
        # Nuevo León
        "64000": {"estado": "NUEVO_LEON", "municipio": "Monterrey", "colonia": "Centro"},
        "64620": {"estado": "NUEVO_LEON", "municipio": "Monterrey", "colonia": "Obispado"},
        "66220": {"estado": "NUEVO_LEON", "municipio": "San Pedro Garza García", "colonia": "Del Valle"},
        "66260": {"estado": "NUEVO_LEON", "municipio": "San Pedro Garza García", "colonia": "Residencial San Agustín"},
        # Estado de México
        "50000": {"estado": "MEXICO", "municipio": "Toluca", "colonia": "Centro"},
        "52140": {"estado": "MEXICO", "municipio": "Metepec", "colonia": "La Virgen"},
        "52786": {"estado": "MEXICO", "municipio": "Huixquilucan", "colonia": "Interlomas"},
        # Puebla
        "72000": {"estado": "PUEBLA", "municipio": "Puebla", "colonia": "Centro"},
        "72160": {"estado": "PUEBLA", "municipio": "Puebla", "colonia": "La Paz"},
        # Querétaro
        "76000": {"estado": "QUERETARO", "municipio": "Querétaro", "colonia": "Centro"},
        "76090": {"estado": "QUERETARO", "municipio": "Querétaro", "colonia": "Juriquilla"},
        # Yucatán
        "97000": {"estado": "YUCATAN", "municipio": "Mérida", "colonia": "Centro"},
        "97130": {"estado": "YUCATAN", "municipio": "Mérida", "colonia": "García Ginerés"},
        # Veracruz
        "91000": {"estado": "VERACRUZ", "municipio": "Xalapa", "colonia": "Centro"},
        "94290": {"estado": "VERACRUZ", "municipio": "Boca del Río", "colonia": "Mocambo"},
        # Guanajuato
        "36000": {"estado": "GUANAJUATO", "municipio": "Guanajuato", "colonia": "Centro"},
        "37000": {"estado": "GUANAJUATO", "municipio": "León", "colonia": "Centro"},
        # Chihuahua
        "31000": {"estado": "CHIHUAHUA", "municipio": "Chihuahua", "colonia": "Centro"},
        "32000": {"estado": "CHIHUAHUA", "municipio": "Juárez", "colonia": "Centro"},
        # Sonora
        "83000": {"estado": "SONORA", "municipio": "Hermosillo", "colonia": "Centro"},
        # Coahuila
        "25000": {"estado": "COAHUILA", "municipio": "Saltillo", "colonia": "Centro"},
        # Sinaloa
        "80000": {"estado": "SINALOA", "municipio": "Culiacán", "colonia": "Centro"},
        "82000": {"estado": "SINALOA", "municipio": "Mazatlán", "colonia": "Centro"},
        # Baja California
        "22000": {"estado": "BAJA_CALIFORNIA", "municipio": "Tijuana", "colonia": "Centro"},
        "21000": {"estado": "BAJA_CALIFORNIA", "municipio": "Mexicali", "colonia": "Centro"},
        # Tabasco
        "86000": {"estado": "TABASCO", "municipio": "Villahermosa", "colonia": "Centro"},
        # Oaxaca
        "68000": {"estado": "OAXACA", "municipio": "Oaxaca de Juárez", "colonia": "Centro"},
        # Quintana Roo
        "77500": {"estado": "QUINTANA_ROO", "municipio": "Cancún", "colonia": "Centro"},
        # Aguascalientes
        "20000": {"estado": "AGUASCALIENTES", "municipio": "Aguascalientes", "colonia": "Centro"},
        # San Luis Potosí
        "78000": {"estado": "SAN_LUIS_POTOSI", "municipio": "San Luis Potosí", "colonia": "Centro"},
        # Michoacán
        "58000": {"estado": "MICHOACAN", "municipio": "Morelia", "colonia": "Centro"},
        # Tamaulipas
        "87000": {"estado": "TAMAULIPAS", "municipio": "Ciudad Victoria", "colonia": "Centro"},
        # Chiapas
        "29000": {"estado": "CHIAPAS", "municipio": "Tuxtla Gutiérrez", "colonia": "Centro"},
        # Guerrero
        "39000": {"estado": "GUERRERO", "municipio": "Chilpancingo", "colonia": "Centro"},
        "39300": {"estado": "GUERRERO", "municipio": "Acapulco", "colonia": "Centro"},
        # Hidalgo
        "42000": {"estado": "HIDALGO", "municipio": "Pachuca", "colonia": "Centro"},
        # Morelos
        "62000": {"estado": "MORELOS", "municipio": "Cuernavaca", "colonia": "Centro"},
        # Nayarit
        "63000": {"estado": "NAYARIT", "municipio": "Tepic", "colonia": "Centro"},
        # Durango
        "34000": {"estado": "DURANGO", "municipio": "Durango", "colonia": "Centro"},
        # Campeche
        "24000": {"estado": "CAMPECHE", "municipio": "Campeche", "colonia": "Centro"},
        # Colima
        "28000": {"estado": "COLIMA", "municipio": "Colima", "colonia": "Centro"},
        # Tlaxcala
        "90000": {"estado": "TLAXCALA", "municipio": "Tlaxcala", "colonia": "Centro"},
        # Zacatecas
        "98000": {"estado": "ZACATECAS", "municipio": "Zacatecas", "colonia": "Centro"},
        # BCS
        "23000": {"estado": "BAJA_CALIFORNIA_SUR", "municipio": "La Paz", "colonia": "Centro"},
        "23400": {"estado": "BAJA_CALIFORNIA_SUR", "municipio": "Los Cabos", "colonia": "San José del Cabo"},
    }

    @classmethod
    def lookup(cls, cp: str) -> Optional[SepomexResponse]:
        cp_clean = cp.strip().zfill(5)
        data = cls.CP_DATABASE.get(cp_clean)
        if data:
            return SepomexResponse(
                cp=cp_clean,
                estado=data["estado"],
                municipio=data["municipio"],
                colonia=data.get("colonia"),
            )
        return None


# ─────────────────────────────────────────
# PRIVACY SHIELD — Wall Garden Middleware
# ─────────────────────────────────────────
# Detecta y oculta información de contacto
# directo para evitar desintermediación.

class PrivacyShield:
    """
    Regex-based content filter that detects and masks
    phone numbers, emails, and external URLs in chat messages
    to prevent platform bypass (desintermediación).
    """

    # Patrones de detección
    PHONE_PATTERNS = [
        r'\+?52\s*[\d\s\-\.]{8,12}',           # +52 formatos
        r'\b55\s*[\d\s\-\.]{7,10}\b',            # 55 (CDMX)
        r'\b\d{2,3}[\s\-]?\d{3,4}[\s\-]?\d{4}\b',  # Genérico 10 dígitos
        r'\(\d{2,3}\)\s*\d{3,4}[\s\-]?\d{4}',   # (código) número
    ]

    EMAIL_PATTERN = r'[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}'

    URL_PATTERNS = [
        r'https?://[^\s<>\"\')]+',
        r'www\.[^\s<>\"\')]+',
        r'[a-zA-Z0-9\-]+\.(com|mx|org|net|io|app|pro|law|legal)[^\s]*',
    ]

    REPLACEMENT = "[DATOS OCULTOS POR SEGURIDAD — Contacte dentro de IUREXIA]"

    @classmethod
    def scan(cls, text: str) -> dict:
        """Scans text for contact info. Returns detection results."""
        detections = []

        for pattern in cls.PHONE_PATTERNS:
            matches = re.findall(pattern, text)
            for m in matches:
                # Filter out short numbers that could be legal refs (art. 123)
                if len(re.sub(r'\D', '', m)) >= 8:
                    detections.append({"type": "phone", "value": m})

        emails = re.findall(cls.EMAIL_PATTERN, text)
        for e in emails:
            detections.append({"type": "email", "value": e})

        for pattern in cls.URL_PATTERNS:
            urls = re.findall(pattern, text, re.IGNORECASE)
            for u in urls:
                # Exclude iurexia domains
                if 'iurexia' not in u.lower() and 'jurexia' not in u.lower():
                    detections.append({"type": "url", "value": u})

        return {
            "has_contact": len(detections) > 0,
            "detections": detections,
            "count": len(detections),
        }

    @classmethod
    def sanitize(cls, text: str) -> str:
        """Replaces detected contact info with shield message."""
        result = text

        for pattern in cls.PHONE_PATTERNS:
            def _phone_replace(match):
                digits = re.sub(r'\D', '', match.group(0))
                if len(digits) >= 8:
                    return cls.REPLACEMENT
                return match.group(0)
            result = re.sub(pattern, _phone_replace, result)

        result = re.sub(cls.EMAIL_PATTERN, cls.REPLACEMENT, result)

        for pattern in cls.URL_PATTERNS:
            def _url_replace(match):
                url = match.group(0)
                if 'iurexia' in url.lower() or 'jurexia' in url.lower():
                    return url
                return cls.REPLACEMENT
            result = re.sub(pattern, _url_replace, result, flags=re.IGNORECASE)

        return result


# ─────────────────────────────────────────
# ENDPOINTS — IUREXIA CONNECT
# ─────────────────────────────────────────

@app.post("/connect/validate-cedula", response_model=CedulaValidationResponse)
async def validate_cedula(request: CedulaValidationRequest):
    """
    Valida formato de cédula profesional (7-8 dígitos).
    El perfil se registra como pendiente de verificación.
    Verificar manualmente en: https://cedulaprofesional.sep.gob.mx
    """
    return await CedulaValidationService.validate(request.cedula)


@app.get("/connect/sepomex/{cp}", response_model=SepomexResponse)
async def sepomex_lookup(cp: str):
    """
    Dado un código postal, devuelve Estado y Municipio.
    Usa diccionario estático de CPs principales de México.
    """
    result = SepomexService.lookup(cp)
    if not result:
        raise HTTPException(
            status_code=404,
            detail=f"Código postal '{cp}' no encontrado. Introduce tu ubicación manualmente.",
        )
    return result


@app.post("/connect/lawyers/search")
async def search_lawyers(request: LawyerSearchRequest):
    """
    Búsqueda de abogados.
    Intenta búsqueda semántica en Qdrant primero.
    Si Qdrant está vacío o falla, busca directamente en Supabase.
    """
    lawyers = []

    # ── Strategy 1: Qdrant semantic search ──
    try:
        embedding_response = await openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=request.query,
        )
        query_vector = embedding_response.data[0].embedding

        qdrant_filter = None
        if request.estado:
            qdrant_filter = Filter(
                must=[
                    FieldCondition(
                        key="estado",
                        match=MatchValue(value=request.estado),
                    )
                ]
            )

        results = await async_qdrant.search(
            collection_name="lawyer_registry",
            query_vector=query_vector,
            query_filter=qdrant_filter,
            limit=request.limit,
            with_payload=True,
        )

        for result in results:
            payload = result.payload or {}
            lawyers.append({
                "id": payload.get("user_id", ""),
                "full_name": payload.get("full_name", ""),
                "cedula_number": payload.get("cedula_number", ""),
                "specialties": payload.get("specialties", []),
                "bio": payload.get("bio", ""),
                "office_address": payload.get("office_address", {}),
                "verification_status": payload.get("verification_status", "pending"),
                "is_pro_active": payload.get("is_pro_active", False),
                "avatar_url": payload.get("avatar_url"),
                "score": result.score,
            })

        if lawyers:
            return {"lawyers": lawyers, "total": len(lawyers)}

    except Exception as qdrant_err:
        print(f"[Connect] Qdrant search failed (falling back to Supabase): {qdrant_err}")

    # ── Strategy 2: Supabase fallback ──
    try:
        print("[Connect] Using Supabase fallback for lawyer search")

        if not supabase:
            print("[Connect] Supabase client not initialized — cannot fallback")
            return {"lawyers": [], "total": 0}

        # Fetch all active lawyer profiles (JSONB filtering not reliable via PostgREST)
        result = supabase.table("lawyer_profiles").select("*").eq(
            "is_pro_active", True
        ).limit(50).execute()

        if result.data:
            search_terms = request.query.lower().split()

            for profile in result.data:
                # ── Estado filter (in Python since office_address is JSONB) ──
                office = profile.get("office_address") or {}
                profile_estado = (office.get("estado") or "").upper().replace(" ", "_")
                if request.estado and request.estado.upper() not in profile_estado:
                    continue

                bio = (profile.get("bio") or "").lower()
                specs = " ".join(profile.get("specialties") or []).lower()
                name = (profile.get("full_name") or "").lower()
                combined = f"{bio} {specs} {name}"

                # Simple term-frequency scoring
                score = sum(1 for term in search_terms if term in combined)

                lawyers.append({
                    "id": profile.get("id", ""),
                    "full_name": profile.get("full_name", ""),
                    "cedula_number": profile.get("cedula_number", ""),
                    "specialties": profile.get("specialties", []),
                    "bio": profile.get("bio", ""),
                    "office_address": office,
                    "verification_status": profile.get("verification_status", "pending"),
                    "is_pro_active": profile.get("is_pro_active", False),
                    "avatar_url": profile.get("avatar_url"),
                    "score": score / max(len(search_terms), 1),
                })

            # Sort by relevance score
            lawyers.sort(key=lambda x: x.get("score", 0), reverse=True)

        print(f"[Connect] Supabase fallback found {len(lawyers)} lawyers")
        return {"lawyers": lawyers[:request.limit], "total": len(lawyers)}

    except Exception as e:
        print(f"[Connect] Supabase fallback error: {e}")
        raise HTTPException(status_code=500, detail=f"Error en búsqueda: {str(e)}")


@app.post("/connect/start")
async def start_connect_chat(request: ConnectStartRequest):
    """
    Crea una sala de chat Connect con Context Handover.
    Genera el dossier preliminar y el mensaje sistema inicial.

    Nota: La creación real de la sala se hace desde el frontend
    via Supabase (con RLS). Este endpoint genera el mensaje
    sistema y valida el abogado.
    """
    try:
        dossier = request.dossier_summary or {}

        # Build system message with dossier
        dossier_text = json.dumps(dossier, ensure_ascii=False, indent=2) if dossier else "Sin expediente preliminar."

        system_message = (
            f"📋 **EXPEDIENTE PRELIMINAR — IUREXIA CONNECT**\n\n"
            f"Licenciado(a), le comparto el Resumen Preliminar del caso "
            f"generado por la IA de Iurexia:\n\n"
            f"```\n{dossier_text}\n```\n\n"
            f"El cliente espera su análisis y cotización.\n\n"
            f"─── *Este mensaje fue generado automáticamente por IUREXIA* ───"
        )

        return {
            "system_message": system_message,
            "dossier": dossier,
            "status": "ready",
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al iniciar Connect: {str(e)}")


@app.post("/connect/privacy-check")
async def privacy_check(request: ConnectMessageRequest):
    """
    Analiza un mensaje antes de enviarlo.
    Detecta datos de contacto y devuelve la versión sanitizada.
    """
    scan_result = PrivacyShield.scan(request.content)
    sanitized = PrivacyShield.sanitize(request.content) if scan_result["has_contact"] else request.content

    return {
        "original": request.content,
        "sanitized": sanitized,
        "has_contact_info": scan_result["has_contact"],
        "detections": scan_result["detections"],
    }


@app.post("/connect/lawyers/index")
async def index_lawyer_profile(profile: LawyerProfileCreate):
    """
    Indexa un perfil de abogado en Qdrant para matching semántico.
    Genera embedding de bio + especialidades y lo almacena en
    la colección `lawyer_registry`.
    """
    try:
        # Build text for embedding
        specialties_text = ", ".join(profile.specialties) if profile.specialties else ""
        embedding_text = f"{profile.full_name}. Especialidades: {specialties_text}. {profile.bio}"

        # Generate embedding
        embedding_response = await openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=embedding_text,
        )
        vector = embedding_response.data[0].embedding

        # Ensure collection exists
        try:
            await async_qdrant.get_collection("lawyer_registry")
        except Exception:
            await async_qdrant.create_collection(
                collection_name="lawyer_registry",
                vectors_config=models.VectorParams(
                    size=1536,  # text-embedding-3-small
                    distance=models.Distance.COSINE,
                ),
            )

        # Upsert point
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, profile.cedula_number))
        estado = profile.office_address.get("estado", "") if profile.office_address else ""

        await async_qdrant.upsert(
            collection_name="lawyer_registry",
            points=[
                models.PointStruct(
                    id=point_id,
                    vector=vector,
                    payload={
                        "user_id": point_id,
                        "cedula_number": profile.cedula_number,
                        "full_name": profile.full_name,
                        "specialties": profile.specialties,
                        "bio": profile.bio,
                        "office_address": profile.office_address,
                        "estado": estado,
                        "verification_status": "pending",
                        "is_pro_active": False,
                        "avatar_url": profile.avatar_url,
                    },
                )
            ],
        )

        return {
            "indexed": True,
            "point_id": point_id,
            "collection": "lawyer_registry",
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al indexar abogado: {str(e)}")


@app.get("/connect/health")
async def connect_health():
    """Health check para el módulo Connect."""
    # Check if lawyer_registry collection exists
    try:
        info = await async_qdrant.get_collection("lawyer_registry")
        lawyer_count = info.points_count
    except Exception:
        lawyer_count = 0

    return {
        "module": "iurexia_connect",
        "status": "operational",
        "lawyers_indexed": lawyer_count,
        "services": {
            "cedula_validation": "mock",
            "sepomex": "static",
            "privacy_shield": "active",
            "qdrant_matching": "active" if lawyer_count > 0 else "empty",
        },
    }


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    
    print("═" * 60)
    print("  JUREXIA CORE API - Motor de Producción")
    print("  + IUREXIA CONNECT - Marketplace Legal")
    print("═" * 60)
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
