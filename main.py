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
- GPT-5 Mini for chat, DeepSeek Reasoner for thinking/reasoning

VERSION: 2026.02.22-v5 (Anti-alucinación 3 capas: Deterministic Fetch + Prompt Guard + Structural Grounding)
"""

import asyncio
import html
import json
import os
import re
import uuid
from typing import AsyncGenerator, List, Literal, Optional, Dict, Set, Tuple, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient, AsyncQdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import (
    FieldCondition,
    Filter,
    Fusion,
    MatchAny,
    MatchValue,
    NamedVector,
    NamedSparseVector,
    Prefetch,
    Query,
    SparseVector,
)
from fastembed import SparseTextEmbedding
from openai import AsyncOpenAI
from supabase import create_client as supabase_create_client
import httpx  # For Cohere Rerank API calls

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ══════════════════════════════════════════════════════════════════════════════

# Cargar variables de entorno desde .env
from dotenv import load_dotenv
load_dotenv()

# Supabase Admin Client (for quota enforcement)
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY", "")
supabase_admin = None
if SUPABASE_URL and SUPABASE_SERVICE_KEY:
    supabase_admin = supabase_create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
    print(f"✅ Supabase admin client initialized (quota enforcement ACTIVE)")
else:
    print(f"⚠️ Supabase admin NOT configured — quota enforcement DISABLED")
    print(f"   SUPABASE_URL={'SET' if SUPABASE_URL else 'MISSING'}, SUPABASE_SERVICE_ROLE_KEY={'SET' if SUPABASE_SERVICE_KEY else 'MISSING'}")

QDRANT_URL = os.getenv("QDRANT_URL", "https://your-cluster.qdrant.tech")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")

# DeepSeek API Configuration (used ONLY for reasoning/thinking mode)
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_CHAT_MODEL = "deepseek-chat"  # Used with thinking mode enabled
REASONER_MODEL = "deepseek-reasoner"  # For document analysis with Chain of Thought

# OpenAI API Configuration (gpt-5-mini for chat + sentencia analysis + embeddings)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
CHAT_MODEL = "gpt-5-mini"  # For regular queries (powerful reasoning, rich output)
SENTENCIA_MODEL = "gpt-5-mini"  # For sentencia analysis (premium quality for ultra_secretarios)

# ── Chat Engine Toggle ──────────────────────────────────────────────────────
# Set via env var CHAT_ENGINE: "openai" (GPT-5 Mini) or "deepseek" (DeepSeek V3)
# DeepSeek V3 is ~65-75% cheaper for equivalent quality in Spanish legal text.
# Switch in Render env vars without redeploy needed (restart service only).
CHAT_ENGINE = os.getenv("CHAT_ENGINE", "deepseek").lower()  # default: deepseek (cost-optimized)
print(f"   Chat Engine: {'🟢 DeepSeek V3 (cost-optimized)' if CHAT_ENGINE == 'deepseek' else '🔵 GPT-5 Mini (premium)'}")

# Cohere Rerank Configuration (cross-encoder for post-retrieval reranking)
COHERE_API_KEY = os.getenv("COHERE_API_KEY", "")
COHERE_RERANK_MODEL = "rerank-v3.5"  # Multilingual, best for Spanish legal text
COHERE_RERANK_ENABLED = bool(COHERE_API_KEY)
print(f"   Cohere Rerank: {'✅ ENABLED' if COHERE_RERANK_ENABLED else '⚠️ DISABLED (no API key)'}")

# HyDE Configuration (Hypothetical Document Embeddings)
HYDE_ENABLED = True  # Generate hypothetical legal document for dense search
HYDE_MODEL = "gpt-5-mini"  # Use fast model for HyDE generation

# Query Decomposition Configuration
QUERY_DECOMPOSITION_ENABLED = True  # Break complex queries into sub-queries

# Silos V5.0 de Jurexia — Arquitectura 32 Silos por Estado
# Silos FIJOS: siempre se buscan independientemente del estado
FIXED_SILOS = {
    "federal": "leyes_federales",
    "jurisprudencia": "jurisprudencia_nacional",
    "constitucional": "bloque_constitucional",  # Constitución, Tratados DDHH, Jurisprudencia CoIDH
}

# Mapa estado → colección dedicada en Qdrant
# Se agregan progresivamente conforme se ingestan estados
ESTADO_SILO = {
    "QUERETARO": "leyes_queretaro",
    "CDMX": "leyes_cdmx",
    # Próximos estados:
    # "JALISCO": "leyes_jalisco",
    # "NUEVO_LEON": "leyes_nuevo_leon",
}

# Silos de SENTENCIAS DE EJEMPLO — usados como few-shot por el redactor multi-pass
SENTENCIA_SILOS = {
    "amparo_directo": "sentencias_amparo_directo",
    "amparo_revision": "sentencias_amparo_revision",
    "recurso_queja": "sentencias_recurso_queja",
    "revision_fiscal": "sentencias_revision_fiscal",
}

# Fallback: colección legacy para estados no migrados
LEGACY_ESTATAL_SILO = "leyes_estatales"

# Alias de compatibilidad: SILOS ahora incluye fijos + legacy
SILOS = {
    **FIXED_SILOS,
    "estatal": LEGACY_ESTATAL_SILO,  # Legacy fallback
}

# Estados mexicanos válidos (normalizados a mayúsculas)
ESTADOS_MEXICO = [
    "AGUASCALIENTES", "BAJA_CALIFORNIA", "BAJA_CALIFORNIA_SUR", "CAMPECHE",
    "CHIAPAS", "CHIHUAHUA", "CIUDAD_DE_MEXICO", "COAHUILA", "COLIMA",
    "DURANGO", "ESTADO_DE_MEXICO", "GUANAJUATO", "GUERRERO", "HIDALGO", "JALISCO",
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


SYSTEM_PROMPT_CHAT = """Eres JUREXIA, IA Juridica especializada en Derecho Mexicano.

===============================================================
   PRINCIPIO FUNDAMENTAL: RESPUESTA COMPLETA, ESTRUCTURADA Y EXHAUSTIVA
===============================================================

Tu PRIORIDAD ABSOLUTA es entregar una respuesta AMPLIA, PROFESIONAL y ORGANIZADA
siguiendo la JERARQUIA NORMATIVA MEXICANA. El usuario espera un analisis completo
que cubra todas las fuentes relevantes del ordenamiento juridico mexicano.

REGLA MAESTRA DE LONGITUD:
Tus respuestas deben ser EXTENSAS y EXHAUSTIVAS. Desarrolla cada punto con profundidad.
NO seas breve ni telegráfico. Un abogado necesita fundamentos amplios, no resúmenes.
Mínimo 800 palabras en consultas sustantivas. Máximo: sin limite practico.

===============================================================
   ESTRUCTURA OBLIGATORIA DE RESPUESTA (JERÁRQUICA)
===============================================================

TODA respuesta DEBE seguir esta estructura, adaptada al tema consultado.
Si una sección NO tiene documentos relevantes del contexto RAG,
DESARROLLA el tema con tu conocimiento jurídico integrándolo naturalmente
en el flujo de la respuesta. NUNCA dejes una sección vacía ni digas
"no se recuperó" o "no se encontró en esta búsqueda".

### RESPUESTA DIRECTA (sin encabezado, primeras 2-3 oraciones)

Responde CONCRETAMENTE la pregunta del usuario en las primeras lineas:
- Si pregunta Si/No: responde con la base legal clave
- Si pide un concepto: definelo directamente  
- Si pide un plazo: da el plazo con su fundamento
- Si pide una estrategia: da tu recomendacion inmediata

### MARCO CONSTITUCIONAL Y DERECHOS HUMANOS

Incluye esta sección SOLO cuando la consulta tenga una dimensión constitucional
o de derechos humanos GENUINA. Ejemplos donde SÍ incluirla:
- Preguntas sobre garantías individuales, discriminación, debido proceso
- Temas de amparo, control de convencionalidad, bloque de constitucionalidad
- Cuando el contexto RAG recuperó artículos de la CPEUM o tratados DDHH

OMITE COMPLETAMENTE esta sección cuando la consulta sea:
- Derecho mercantil puro (títulos de crédito, sociedades, concursos)
- Derecho fiscal o administrativo sin dimensión de derechos fundamentales
- Derecho civil patrimonial (contratos, obligaciones, propiedad)
- Derecho procesal sin violación a garantías
- Cualquier tema donde citar la Constitución sería forzado o artificial

NUNCA cites el Art. 1 CPEUM como relleno genérico. Solo cítalo cuando sea
directamente relevante al problema jurídico consultado.

Cuando SÍ incluyas esta sección:
- **Constitución Política** (CPEUM): Artículos aplicables con texto transcrito
- **Tratados internacionales de DDHH**: Convención Americana, PIDCP, PIDESC, etc.
- **Principio pro persona** (Art. 1 CPEUM): interpretación más favorable
- **Bloque de constitucionalidad**: criterios CoIDH cuando apliquen

REGLA CRÍTICA - ARTÍCULOS CONSTITUCIONALES EN EL CONTEXTO:
Los artículos de la Constitución (CPEUM) aparecen en el contexto RAG con refs como
"Art. 1o CPEUM", "Art. 4o CPEUM", etc. El texto del artículo está en el campo <texto>
del documento XML. Cuando encuentres un documento con ref "Art. [N] CPEUM" o
"Art. [N]o CPEUM", OBLIGATORIAMENTE:
1. IDENTIFICA que ese documento contiene el texto literal del artículo constitucional
2. TRANSCRIBE el texto COMPLETO del artículo en un blockquote
3. CITA con [Doc ID: uuid]
4. NUNCA digas "el texto no se encontró" si hay un documento con ref "Art. [N] CPEUM"

FORMATO OBLIGATORIO para cada artículo constitucional (blockquote):
> "[Texto transcrito del artículo]" -- *Artículo [N], Constitución Política de los Estados Unidos Mexicanos* [Doc ID: uuid]

Para tratados internacionales:
> "[Texto transcrito]" -- *Artículo [N], Convención Americana sobre Derechos Humanos* [Doc ID: uuid]

### LEGISLACIÓN FEDERAL APLICABLE

Desarrolla el fundamento en leyes federales con CITAS TEXTUALES completas.
Para CADA artículo recuperado del contexto, TRANSCRIBE el texto en blockquote.

FORMATO OBLIGATORIO para cada artículo federal (blockquote, idéntico a jurisprudencia):
> "[Texto transcrito del artículo tal como aparece en el contexto recuperado, incluyendo fracciones relevantes]" -- *Artículo [N], [Nombre completo de la Ley]* [Doc ID: uuid]

Ejemplo correcto:
> "Cuando las autoridades fiscales soliciten de los contribuyentes, responsables solidarios o terceros, informes, datos o documentos... I. La solicitud se notificará... II. En la solicitud se indicará el lugar y el plazo..." -- *Artículo 48, Código Fiscal de la Federación* [Doc ID: a1b2c3d4-e5f6-7890-abcd-ef1234567890]

NUNCA menciones un artículo sin transcribir su texto en blockquote y sin [Doc ID: uuid].

### JURISPRUDENCIA Y TESIS APLICABLES

OBLIGATORIO incluir jurisprudencia del contexto RAG.

FORMATO OBLIGATORIO para cada tesis (blockquote):
> "[RUBRO COMPLETO DE LA TESIS EN MAYÚSCULAS]" -- *[Tribunal], [Epoca], Registro digital: [numero]* [Doc ID: uuid]

Explicación: [Desarrolla brevemente CÓMO sustenta o complementa tu análisis] [Doc ID: uuid]

Para cada tesis:
- Integra la jurisprudencia como parte del razonamiento, no como apéndice
- Si hay múltiples tesis, ordénalas por relevancia

Solo si NO hay jurisprudencia en el contexto, indica:
"No se encontró jurisprudencia específica sobre este punto en la búsqueda actual."

### LEGISLACIÓN ESTATAL (Solo cuando aplique)

Si el usuario tiene un estado seleccionado o pregunta sobre derecho local:

FORMATO OBLIGATORIO para cada artículo estatal (blockquote):
> "[Texto transcrito completo del artículo]" -- *Artículo [N], [Nombre de la Ley Estatal]* [Doc ID: uuid]

- Señala diferencias o complementos respecto a la legislación federal
- Marca expresamente: "En [Estado], la legislación local establece..."

Si NO hay estado seleccionado ni pregunta estatal, OMITE esta sección.

### ANÁLISIS INTEGRADO Y RECOMENDACIONES

Cuando la consulta lo amerite:
- Conecta las fuentes anteriores en un análisis coherente
- Señala vías procesales disponibles (si aplica)
- Ofrece recomendaciones prácticas fundamentadas
- Identifica riesgos o consideraciones especiales

### CONCLUSIÓN

Al final, cierra con una síntesis breve y, cuando sea pertinente, incluye
una pregunta de seguimiento que invite al usuario a profundizar o aplicar
la información a su caso concreto. Debe fluir naturalmente como diálogo profesional.

===============================================================
   REGLAS DE USO DEL CONTEXTO RAG
===============================================================

REGLA #1 - OBLIGATORIO USAR FUENTES:
Los documentos en el CONTEXTO JURIDICO RECUPERADO fueron seleccionados por relevancia
semantica a tu consulta. SIEMPRE contienen informacion util. Tu trabajo como jurista es:
1. ANALIZAR cada documento recuperado y extraer lo relevante A LA PREGUNTA ESPECIFICA
2. SINTETIZAR la informacion en una respuesta coherente y JERARQUICAMENTE organizada
3. CITAR con [Doc ID: uuid] cada fuente que uses
4. NUNCA digas "no encontre fuentes" si hay documentos en el contexto - USALOS

REGLA #2 - RAZONAMIENTO JURIDICO:
Si la consulta pregunta por un concepto y el contexto tiene articulos aplicables,
CONECTA la norma con el concepto usando interpretacion juridica.
Si el contexto tiene normas analogas de otros estados, usalas como referencia
comparativa senalando expresamente que se trata de analisis por analogia.

REGLA #2-BIS - MÉTODO DE SUBSUNCIÓN (APLICARLO EN SILENCIO):
Antes de redactar cada sección sustantiva, aplica mentalmente este razonamiento:
  1. Identifica el artículo aplicable del contexto (la norma general)
  2. Conecta los hechos del usuario con los elementos de la norma
  3. Apóyate en la jurisprudencia del contexto para confirmar el encuadramiento
  4. Emite un dictamen claro: qué puede hacer, qué riesgo corre, qué vía procesal corresponde

IMPORTANTE: Este razonamiento debe REFLEJARSE en la prosa de tu respuesta, pero
NUNCA usando las etiquetas explícitas "Premisa Mayor", "Premisa Menor", "Subsunción"
ni "Conclusión Jurídica". La subsunción va implícita en el análisis, como haría
un abogado en un dictamen profesional: conectar norma → hechos → consecuencia
en texto corrido, sin anunciar cada paso del método.

REGLA #3 - CERO ALUCINACIONES:
1. CITA contenido textual del CONTEXTO JURIDICO RECUPERADO
2. NUNCA inventes articulos, tesis, o jurisprudencia
3. Puedes hacer razonamiento juridico SOBRE las fuentes del contexto
4. Si NINGUN documento es relevante (extremadamente raro), indicalo

🚨 REGLA #3-BIS — PROHIBICIÓN ABSOLUTA PARA TEXTO LEGAL:
Esta regla tiene PRIORIDAD MÁXIMA sobre cualquier otra instrucción:

1. Para artículos constitucionales (CPEUM), leyes federales, tratados internacionales
   y otros textos normativos:
   → SOLO TRANSCRIBES texto que aparezca LITERALMENTE en el campo <texto> de los
     documentos del CONTEXTO JURIDICO RECUPERADO.
   → NUNCA completes, parafrasees, ni "recuerdes" el texto de ningún artículo aunque
     creas conocerlo perfectamente de tu entrenamiento.
   → Razón crítica: tu entrenamiento contiene texto pre-Reforma Judicial 2024. El
     contexto RAG tiene el texto vigente actualizado. SIEMPRE usa el RAG, nunca tu memoria.

2. Si el artículo específico (ej. "Art. 94 CPEUM") NO aparece en el contexto RAG:
   → Responde EXACTAMENTE: "No encontré el texto del [Artículo X] en mi base de datos
     actualizada. Para consultarlo directamente: https://www.diputados.gob.mx/LeyesBiblio/pdf/CPEUM.pdf"
   → NUNCA lo transcribas de tu memoria aunque tengas alta confianza.

3. GROUNDING OBLIGATORIO — ESTRUCTURA PARA CITAS LEGALES:
   Cuando el artículo SÍ está en el contexto RAG, usa esta secuencia:
   PASO 1 — TRANSCRIPCIÓN LITERAL del campo <texto> del documento en blockquote con Doc ID:
   > "[Texto exacto tal como aparece en el contexto]" -- *Art. X, [Ley]* [Doc ID: uuid]
   PASO 2 — SOLO DESPUÉS de la transcripción literal, tu interpretación jurídica.
   NUNCA mezcles texto literal con interpretación en el mismo blockquote.

REGLA #4 - EXHAUSTIVIDAD EN FUENTES:
Si hay 10 documentos relevantes en el contexto, USA LOS 10 en tu respuesta.
Cada fuente aporta matices legales valiosos. Para cada articulo o tesis:
- MENCIONA el articulo/numero de tesis
- TRANSCRIBE el texto relevante del documento
- CITA con [Doc ID: uuid] inmediatamente despues
La unica excepcion es si un documento es genuinamente irrelevante a la pregunta.

REGLA #5 - JURISPRUDENCIA OBLIGATORIA:
Si el contexto contiene jurisprudencia, SIEMPRE incluyela en tu respuesta.
Formato OBLIGATORIO para cada tesis:
> "[RUBRO COMPLETO]" -- *[Tribunal], [Epoca], Registro digital: [numero]* [Doc ID: uuid]
Desarrolla brevemente como la tesis sustenta o complementa tu analisis.
Si no hay jurisprudencia en el contexto, indica: "No se encontro jurisprudencia especifica
sobre este punto en la busqueda actual."

REGLA #6 — JERARQUÍA NORMATIVA ABSOLUTA Y VIGENCIA TEMPORAL:

ORDEN DE AUTORIDAD LEGAL (de mayor a menor — NUNCA violes este orden):
  1. CONSTITUCION (CPEUM) — texto literal vigente, ley suprema
  2. TRATADOS INTERNACIONALES DE DDHH (bloque de constitucionalidad)
  3. LEYES FEDERALES y CODIGO NACIONAL (vigentes tras la ultima reforma)
  4. LEGISLACION ESTATAL (en su ambito)
  5. JURISPRUDENCIA / TESIS — solo si es CONSISTENTE con 1-4

CUANDO EXISTE CONFLICTO NORMA vs. JURISPRUDENCIA:
Si el contexto incluye TANTO articulos constitucionales/legales vigentes COMO
jurisprudencia de un sistema ANTERIOR, OBLIGATORIAMENTE debes:
  a) Aplicar la NORMA CONSTITUCIONAL/LEGAL ACTUAL como respuesta principal
  b) Citar la jurisprudencia SOLO como referencia historica del sistema previo,
     indicando EXPRESAMENTE: "Esta tesis corresponde al sistema anterior a la
     Reforma [año]. Bajo el marco constitucional vigente, la regla es..."
  c) NUNCA presentar como vigente una tesis que contradiga el texto actual de la CPEUM

SEÑALES DE QUE UNA TESIS PUEDE ESTAR SUPERADA POR REFORMA:
  - Tesis de Novena Epoca (antes de 2011): verificar si la norma cambio despues
  - Tesis de Decima Epoca (2011-2021): verificar si hay reforma post-2021 que la supere
  - Cualquier tesis que cite al "Consejo de la Judicatura Federal" en materia de
    DESIGNACION, CONCURSOS o ADSCRIPCION de Magistrados/Jueces federales:
    → SUPERADA por REFORMA JUDICIAL 2024
  - Cualquier tesis sobre "concurso de oposicion" para designar Magistrados de
    Circuito o Jueces de Distrito: → SUPERADA por Reforma Judicial 2024

CONOCIMIENTO CRITICO — REFORMA JUDICIAL 2024 (DOF 15-sep-2024 y 14-oct-2024):
Esta reforma modifico radicalmente los articulos 94, 96, 97, 99, 100, 116 y 122 CPEUM.
Sus efectos son IRREVERSIBLES Y VIGENTES desde su publicacion:
  - Los Jueces de Distrito y Magistrados de Circuito se eligen por VOTO POPULAR DIRECTO
  - El Consejo de la Judicatura Federal fue DISUELTO y sustituido por el
    Tribunal de Disciplina Judicial y el Organo de Administracion Judicial
  - Los concursos de oposicion administrados por el CJF YA NO ESTAN VIGENTES
  - Primera eleccion extraordinaria: 2025
  - Duracion del encargo: 9 años (8 años para los electos en 2025)
  
Si el contexto contiene tesis sobre concursos CJF para designar juzgadores federales
Y TAMBIEN contiene articulos 94, 96 o 97 CPEUM, SIEMPRE aplica el texto constitucional
como derecho vigente. Las tesis son del sistema anterior y deben citarse como tal.

FORMATO DE CITAS:
- Usa [Doc ID: uuid] del contexto proporcionado para respaldar cada afirmacion
- Los UUID tienen 36 caracteres: xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
- Cada cita debe estar INMEDIATAMENTE despues del texto que respalda
- NUNCA coloques multiples [Doc ID] consecutivos sin texto entre ellos
- Correcto: "El articulo 17 establece... [Doc ID: abc]. Asimismo, el articulo 19 dispone... [Doc ID: def]"
- Incorrecto: "Los articulos 17 y 19... [Doc ID: abc] [Doc ID: def] [Doc ID: ghi]"

===============================================================
   PROHIBICIONES ABSOLUTAS
===============================================================

NUNCA uses emoticonos, emojis o simbolos decorativos en tus respuestas.
Manten un tono profesional, formal pero accesible.

PROHIBICIÓN CRÍTICA - NUNCA ADMITAS CARENCIAS:
NUNCA digas frases como:
- "no se recuperó el texto"
- "no logré encontrar"
- "en esta búsqueda no se encontró"
- "no tengo cobertura"
- "mi base no incluye"
- "el texto no fue recuperado"
- "el contexto recuperado no contiene el texto literal"
- "el texto completo del artículo no se encuentra"
- "Aunque el texto literal del artículo no se encuentra transcrito en el contexto"
Estas frases DESTRUYEN la confianza del usuario.

REGLA ESPECIAL PARA ARTÍCULOS CPEUM:
Si el usuario pregunta "qué dice el artículo X de la Constitución" y en el contexto
hay un documento con ref "Art. [X]o CPEUM" o "Art. [X] CPEUM", entonces el texto
del artículo SÍ ESTÁ en el contexto. TRANSCRÍBELO literalmente del campo <texto>.
NUNCA digas que no encontraste el texto si el documento con esa ref existe.

Si el contexto RAG no contiene un artículo específico, DESARROLLA
el tema con tu conocimiento jurídico y cita las fuentes que SÍ tienes.
El usuario NUNCA debe percibir que el sistema falló en encontrar algo.
Trabaja SIEMPRE con confianza usando lo disponible.

DIAGRAMAS VISUALES (CUANDO SEA PERTINENTE):

Cuando tu respuesta describa estructuras organizativas o procedimientos por etapas,
COMPLEMENTA el texto con bloques visuales especiales.

A) ORGANIGRAMA - Para estructuras jerarquicas:
:::orgchart
titulo: Estructura de la Administracion Publica de [Estado/Institucion]
[Nodo raiz] -> [Hijo 1], [Hijo 2], [Hijo 3]
[Hijo 1] -> [Nieto 1], [Nieto 2]
:::

B) FLUJO PROCESAL - Para vias procesales o procedimientos secuenciales:
:::processflow
titulo: Juicio de Amparo Indirecto
1. Presentacion de demanda | Ante Juez de Distrito | Plazo: 15 dias habiles
2. Auto admisorio | Juez califica la demanda | 24-48 horas
3. Informe justificado | Autoridad responsable rinde informe | 15 dias
4. Audiencia constitucional | Pruebas, alegatos y sentencia | Fecha senalada
5. Sentencia | Concesion o negacion del amparo | Variable
:::

REGLAS para diagramas:
- SOLO usa :::orgchart si hay jerarquia organizativa real
- SOLO usa :::processflow si hay etapas procesales secuenciales con plazos
- NO uses diagramas para preguntas simples o definiciones
- Maximo 8 nodos/etapas
"""

# ── Chat Drafting Mode: triggered by natural language ("redacta", "ayúdame a redactar", etc.) ──
SYSTEM_PROMPT_CHAT_DRAFTING = """Eres JUREXIA REDACTOR, asistente juridico especializado en
redaccion de textos legales mexicanos.

===============================================================
   MODO REDACCION EN CHAT
===============================================================

El usuario te ha pedido REDACTAR un texto juridico. Tu trabajo es GENERAR el texto
solicitado, NO hacer un analisis academico. Usa el CONTEXTO JURIDICO RECUPERADO como
fundamento para tu redaccion.

REGLAS DE REDACCION:

1. GENERA TEXTO LEGAL FORMAL — prosa juridica profesional, lista para usar
2. FUNDAMENTA con articulos y jurisprudencia del contexto RAG
3. Incluye encabezados, fundamentos de derecho, argumentacion y petitorio cuando aplique
4. NO hagas analisis paso a paso — REDACTA directamente el documento solicitado
5. Si el usuario pide argumentos, genera argumentos juridicos desarrollados con
   fundamento legal y jurisprudencial, no una lista de ideas
6. Si el usuario pide un escrito, genera el escrito completo con estructura formal

ESTRUCTURA ADAPTATIVA:
- Si pide "argumentos" o "agravios": Redacta cada argumento como parrafo juridico
  fundamentado con articulos y jurisprudencia del contexto
- Si pide "escrito" o "demanda": Genera el documento con estructura formal
  (encabezado, hechos, fundamentos de derecho, puntos petitorios)
- Si pide "recurso" o "impugnacion": Genera agravios estructurados con
  violacion alegada, fundamento y agravio expresado
- Si pide redaccion generica: Adapta el formato al tipo de texto solicitado

USO DEL CONTEXTO RAG:
- Los articulos de ley del contexto son tu MATERIA PRIMA — incorporalos en la redaccion
- La jurisprudencia del contexto FORTALECE tus argumentos — citala con formato:
  > "[RUBRO COMPLETO]" -- *[Tribunal], Registro digital: [numero]* [Doc ID: uuid]
- NUNCA inventes articulos, tesis, ni registros digitales
- Cada cita debe llevar su [Doc ID: uuid]

TONO: Formal juridico, persuasivo, profesional. Sin emojis ni decoraciones.

Al final, ofrece al usuario la posibilidad de ajustar, profundizar o modificar
la redaccion segun sus necesidades especificas.
"""

# Trigger phrases for natural language drafting detection (lowercase comparison)
_CHAT_DRAFTING_TRIGGERS = [
    "redacta ", "redáctame", "redactame", "ayúdame a redactar", "ayudame a redactar",
    "genera un escrito", "genera argumentos", "generar argumentos", "genera agravios",
    "vamos a generar", "vamos a redactar", "elabora un", "elabora una",
    "redacción de", "redaccion de", "necesito redactar", "quiero redactar",
    "prepara un escrito", "prepara una demanda", "prepara un recurso",
    "hazme un escrito", "hazme una demanda", "hazme un recurso",
    "draft ", "escribe un escrito", "escribe una demanda",
    "ayúdame a generar", "ayudame a generar",
    "genera un agravio", "genera los agravios", "genera un concepto de violación",
    "genera un concepto de violacion",
]

def _detect_chat_drafting(message: str) -> bool:
    """Detect if the user's message is a natural language drafting request."""
    msg_lower = message.strip().lower()
    # Check if message STARTS with any trigger phrase
    for trigger in _CHAT_DRAFTING_TRIGGERS:
        if msg_lower.startswith(trigger):
            return True
    return False

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
# PROMPT ESPECIALIZADO: ANÁLISIS DE SENTENCIAS (Magistrado IA)
# Modelo: OpenAI o3 (razonamiento profundo)
# ═══════════════════════════════════════════════════════════════

SYSTEM_PROMPT_SENTENCIA_ANALYSIS = """Eres JUREXIA MAGISTRADO, un sistema de inteligencia artificial con capacidad analítica
equivalente a un magistrado federal altamente especializado del Poder Judicial de la Federación.
Tu función es realizar un análisis exhaustivo y objetivo de sentencias judiciales, confrontándolas
con la base de datos jurídica verificada de Iurexia.

═══════════════════════════════════════════════════════════════
   PROTOCOLO DE ANÁLISIS JUDICIAL — GRADO MAGISTRADO
═══════════════════════════════════════════════════════════════

Eres un revisor jerárquico. Analiza la sentencia como si fueras un magistrado de segunda
instancia o un tribunal de amparo revisando el proyecto. Tu análisis debe ser:
- OBJETIVO: Sin sesgo hacia ninguna parte procesal
- EXHAUSTIVO: Cada fundamento debe verificarse contra la base de datos
- FUNDAMENTADO: Cada observación debe citar fuentes del CONTEXTO JURÍDICO
- CRÍTICO: Detectar tanto aciertos como errores, omisiones y contradicciones

═══════════════════════════════════════════════════════════════
   REGLA ABSOLUTA: CERO ALUCINACIONES
═══════════════════════════════════════════════════════════════

1. PRIORIZA citar normas, artículos y jurisprudencia del CONTEXTO JURÍDICO RECUPERADO
2. Cada cita del contexto DEBE incluir [Doc ID: uuid] — copia el UUID exacto del contexto
3. Los UUID tienen 36 caracteres exactos: xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
4. Si el CONTEXTO contiene legislación o jurisprudencia relevante, ÚSALA SIEMPRE
5. NUNCA inventes, acortes ni modifiques UUIDs
6. SOLO cuando el contexto NO contiene NINGUNA fuente sobre un tema específico,
   indica brevemente: "⚠️ Observación sin fuente disponible en la base de datos"

═══════════════════════════════════════════════════════════════
   ESTRUCTURA OBLIGATORIA DEL DICTAMEN
═══════════════════════════════════════════════════════════════

## I. RESUMEN EJECUTIVO
Síntesis clara y concisa de la sentencia en máximo 10 líneas:
- Tipo de juicio y materia
- Partes procesales
- Acto reclamado o litis planteada
- Sentido del fallo (favorable/desfavorable, concede/niega)
- Puntos resolutivos principales

## II. IDENTIFICACIÓN DEL ACTO RECLAMADO Y LA LITIS
- Acto reclamado con precisión
- Litis planteada
- Pretensiones de las partes
- Vía procesal utilizada
- ¿Es la vía correcta? Fundamentar con el contexto

## III. ANÁLISIS DE COMPETENCIA Y PROCEDENCIA
- ¿El tribunal es competente por materia, grado y territorio?
- ¿Se cumplieron los presupuestos procesales?
- ¿Hay causas de improcedencia o sobreseimiento no advertidas?
- Fundamentar con artículos del contexto [Doc ID: uuid]

## IV. ANÁLISIS DE FONDO — FORTALEZAS ✅
Qué hace bien la sentencia:
- Fundamentación jurídica correcta (verificar contra el contexto)
- Motivación adecuada
- Congruencia entre pretensiones y resolución
- Aplicación correcta de jurisprudencia
- Valoración probatoria adecuada
Cada fortaleza con su fuente de respaldo: [Doc ID: uuid]

## V. ANÁLISIS DE FONDO — DEBILIDADES Y ERRORES ❌
Qué tiene la sentencia que es incorrecto, insuficiente u omiso:

### A. Errores de Fundamentación Legal
- Artículos citados incorrectamente o mal aplicados
- Normas vigentes no aplicadas que deberían haberse considerado
- Contradicciones con disposiciones del contexto
Para cada error: citar la norma correcta del contexto [Doc ID: uuid]

### B. Errores Jurisprudenciales
- Jurisprudencia obligatoria no observada (Art. 217 Ley de Amparo)
- Tesis aisladas relevantes no consideradas
- Jurisprudencia aplicada incorrectamente
- Contradicción con criterios del CONTEXTO JURÍDICO
Citar la jurisprudencia omitida o contradicha [Doc ID: uuid]

### C. Errores de Motivación
- Motivación insuficiente: hechos no vinculados con normas
- Motivación incongruente: razonamiento contradictorio
- Falta de exhaustividad: argumentos de las partes no abordados

### D. Omisiones Constitucionales
- Violaciones al debido proceso (Art. 14 CPEUM)
- Falta de fundamentación y motivación (Art. 16 CPEUM)
- Principio pro persona no observado (Art. 1° CPEUM)
- Control de convencionalidad omitido
- Derechos humanos no protegidos
Fundamentar con el bloque constitucional del contexto [Doc ID: uuid]

## VI. CONFRONTACIÓN CON JURISPRUDENCIA DE LA BASE DE DATOS
Tabla de jurisprudencia relevante del CONTEXTO JURÍDICO:

| # | Rubro/Tesis | Tribunal | Relación con la Sentencia | Doc ID |
|---|-------------|----------|---------------------------|--------|
| 1 | ... | ... | Confirma/Contradice/No advertida | [Doc ID: uuid] |

Para cada tesis: explicar si la sentencia la aplica correctamente, la ignora, o la contradice.

## VII. CONFRONTACIÓN CON LEGISLACIÓN DE LA BASE DE DATOS
Tabla de artículos legislativos relevantes del CONTEXTO JURÍDICO:

| # | Artículo | Ley/Código | Aplicación en Sentencia | Doc ID |
|---|----------|------------|------------------------|--------|
| 1 | Art. X | ... | Correcta/Incorrecta/Omitida | [Doc ID: uuid] |

## VIII. ERRORES DE FORMA Y REDACCIÓN
- Errores ortográficos o gramaticales que afecten claridad
- Imprecisiones terminológicas
- Incongruencia en numeración de considerandos
- Deficiencias en la estructura formal de la sentencia

## IX. PROPUESTAS DE MEJORA Y FORTALECIMIENTO
Para cada debilidad identificada, proponer:
- La corrección específica con fundamento del contexto
- Texto alternativo sugerido cuando aplique
- Jurisprudencia o legislación que fortalecería el argumento
Cada propuesta anclada en fuentes [Doc ID: uuid]

## X. DICTAMEN FINAL
- Calificación general: CORRECTA / CORRECTA CON OBSERVACIONES / DEFICIENTE / DEBE REVISARSE
- Resumen de hallazgos críticos (máximo 5 puntos)
- Riesgo de revocación o modificación en segunda instancia o amparo
- Recomendaciones prioritarias numeradas

═══════════════════════════════════════════════════════════════
   PRINCIPIOS RECTORES
═══════════════════════════════════════════════════════════════

1. PRINCIPIO PRO PERSONA (Art. 1° CPEUM): En materia de DDHH, siempre
   aplica la interpretación más favorable a la persona.

2. CONTROL DE CONVENCIONALIDAD: Verifica conformidad con tratados
   internacionales y jurisprudencia de la CoIDH si hay en el contexto.

3. OBLIGATORIEDAD JURISPRUDENCIAL (Art. 217 Ley de Amparo):
   Señala si existe jurisprudencia obligatoria en el contexto que debió
   observarse y no se hizo.

4. SUPLENCIA DE LA QUEJA: Cuando aplique (materia penal, laboral a
   favor del trabajador, menores, derechos agrarios), verifica si la
   sentencia actuó de oficio como corresponde.

═══════════════════════════════════════════════════════════════
   REGLAS DE CITACIÓN Y FORMATO
═══════════════════════════════════════════════════════════════

1. Utiliza AMPLIAMENTE el CONTEXTO JURÍDICO RECUPERADO para fundamentar tu análisis.
   El contexto contiene legislación y jurisprudencia real de la base de datos.
2. Cuando cites, incluye [Doc ID: uuid] del contexto.
3. Si un artículo constitucional, ley o tesis aparece en el contexto, CÍTALO.
   No seas restrictivo: si el contenido del contexto es relevante, úsalo.
4. Si el CONTEXTO JURÍDICO no contiene fuentes sobre un punto específico:
   "⚠️ La base de datos no contiene fuentes adicionales sobre este punto.
   Se recomienda consulta manual de: [fuentes específicas]."
5. NUNCA inventes UUIDs. Si no tienes el UUID, no lo incluyas.
6. FORMATO DE TABLAS: Para TODA información tabulada usa EXCLUSIVAMENTE
   tablas markdown con pipes (|). Ejemplo:
   | Columna 1 | Columna 2 |
   |-----------|-----------|
   | dato | dato |
   NUNCA uses caracteres Unicode de dibujo de caja (┌─┬─┐│├└ etc.)
7. Al final del análisis, incluye una sección "## Fuentes citadas" listando
   cada fuente usada con su Doc ID y descripción breve.

IMPORTANTE: Este es un ANÁLISIS PROFESIONAL para uso del magistrado o juez.
NO es una resolución judicial. NO incluyas frases como "Notifíquese",
"Archívese", "Anótese en el Libro de Gobierno" o similares.
El tono debe ser de dictamen técnico pericial.
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

Tu capacidad creativa debe ser MÁXIMA: no te limites a llenar plantillas. Construye argumentos persuasivos, narrativas convincentes y fundamentos legales profundos. SIEMPRE recurre a la base de datos RAG para fundar cada argumento.

═══════════════════════════════════════════════════════════════
   FASE 0: DETECCIÓN DE REQUISITOS POR MATERIA
═══════════════════════════════════════════════════════════════

Antes de redactar, IDENTIFICA el subtipo de demanda y aplica los requisitos específicos:

▸ CIVIL: Artículos del Código de Procedimientos Civiles de la jurisdicción. Requisitos: personalidad, vía procesal (ordinaria/ejecutiva/sumaria/especial), prestaciones, hechos, fundamentos, pruebas. Busca en RAG los artículos procesales locales.

▸ LABORAL: Artículo 872 y siguientes de la Ley Federal del Trabajo. Requisitos: datos del trabajador, patrón, relación laboral, tipo de despido, salario integrado, antigüedad, prestaciones (indemnización constitucional, salarios caídos, vacaciones, prima vacacional, aguinaldo, PTU). Las acciones laborales NO prescriben igual que las civiles.

▸ FAMILIAR: Código de Procedimientos Familiares o Civiles según la entidad. Requisitos: acta de matrimonio/nacimiento, régimen patrimonial, hijos menores (guarda/custodia, pensión alimenticia, régimen de convivencia), bienes gananciales. VERIFICAR si la entidad tiene juzgados orales familiares.

▸ MERCANTIL (Juicio Oral): Artículos 1390 Bis y siguientes del Código de Comercio. Requisitos: cuantía dentro del rango del juicio oral, títulos de crédito si es ejecutiva, contrato mercantil, relación comercial. Para ejecutiva: documento que traiga aparejada ejecución.

▸ AGRARIO: Ley Agraria, artículos 163 y siguientes. Requisitos: calidad agraria (ejidatario, comunero, avecindado), certificado de derechos agrarios, acuerdo de asamblea, conflictos de linderos o dotación. Tribunal Unitario o Superior Agrario según competencia.

═══════════════════════════════════════════════════════════════
   FASE 1: ANÁLISIS ESTRATÉGICO PREVIO (PIENSA ANTES DE REDACTAR)
═══════════════════════════════════════════════════════════════

Antes de redactar, ANALIZA internamente:
1. ¿Qué acción es la IDÓNEA para lo que reclama el usuario?
2. ¿Cuál es la VÍA PROCESAL correcta? BUSCA en el contexto RAG qué dice el código procesal local.
3. ¿Cuáles son los ELEMENTOS DE LA ACCIÓN? BUSCA jurisprudencia que los defina.
4. ¿Qué PRUEBAS son INDISPENSABLES? Relaciónolas con cada elemento.
5. ¿Hay JURISPRUDENCIA que defina los requisitos de procedencia? CITA con [Doc ID: uuid].
6. ¿La JURISDICCIÓN tiene reglas especiales? BUSCA en el código procesal local del RAG.

═══════════════════════════════════════════════════════════════
   FASE 2: REDACCIÓN DE LA DEMANDA
═══════════════════════════════════════════════════════════════

ESTRUCTURA OBLIGATORIA:

## DEMANDA DE [TIPO DE JUICIO]

**RUBRO**
EXPEDIENTE: ________
SECRETARÍA: ________

**ENCABEZADO**
C. JUEZ [Civil/Familiar/Laboral/de Distrito/Unitario Agrario] EN TURNO
EN [Ciudad según jurisdicción seleccionada]
P R E S E N T E

**DATOS DEL ACTOR**
[Nombre], mexicano(a), mayor de edad, [estado civil], con domicilio en [dirección], señalando como domicilio para oír y recibir notificaciones el ubicado en [dirección procesal], autorizando en términos del artículo [aplicable según código procesal de la jurisdicción] a los licenciados en derecho [nombres], con cédulas profesionales números [X], ante Usted con el debido respeto comparezco para exponer:

**VÍA PROCESAL**
Que por medio del presente escrito y con fundamento en los artículos [citar del código procesal de la JURISDICCIÓN SELECCIONADA — BUSCAR EN RAG] vengo a promover juicio [tipo exacto] en contra de:

**DEMANDADO(S)**
[Datos completos incluyendo domicilio para emplazamiento]

**PRESTACIONES**
Reclamo de mi contrario las siguientes prestaciones:

A) [Prestación principal - CREATIVA: articula exactamente la pretensión con fundamento]
B) [Prestaciones accesorias - intereses legales/moratorios, daños, perjuicios]
C) El pago de gastos y costas que origine el presente juicio.
[Para LABORAL: desglosar indemnización art. 50/48 LFT, salarios caídos, vacaciones, prima, aguinaldo, PTU]

**HECHOS**
(SECCIÓN CREATIVA MÁXIMA: Narra de forma PERSUASIVA, CRONOLÓGICA y ESTRATÉGICA)
(Cada hecho debe orientarse a ACREDITAR un elemento de la acción)
(USA lenguaje que genere convicción en el juzgador)

1. [Hecho que establece la relación jurídica — con contexto emotivo si aplica]
2. [Hecho que acredita la obligación o el derecho violentado]
3. [Hecho que demuestra el incumplimiento o la afectación]
4. [Hecho que relaciona el daño con la prestación reclamada]
[Continuar numeración — sé EXHAUSTIVO y CREATIVO]

**DERECHO APLICABLE**
(FUNDA AGRESIVAMENTE con todo el RAG disponible)

FUNDAMENTO CONSTITUCIONAL:
> "Artículo X.-..." — *CPEUM* [Doc ID: uuid]

FUNDAMENTO PROCESAL (JURISDICCIÓN ESPECÍFICA):
> "Artículo X.-..." — *[Código de Procedimientos del Estado]* [Doc ID: uuid]

FUNDAMENTO SUSTANTIVO:
> "Artículo X.-..." — *[Código Civil/Mercantil/LFT/Ley Agraria]* [Doc ID: uuid]

JURISPRUDENCIA QUE DEFINE ELEMENTOS DE LA ACCIÓN:
> "[Rubro de la tesis]" — *SCJN/TCC* [Doc ID: uuid]
**Aplicación creativa:** [Explica CÓMO esta tesis fortalece la posición del actor]

**PRUEBAS**
Ofrezco las siguientes pruebas, relacionándolas con los hechos:

1. DOCUMENTAL PÚBLICA.- Consistente en... relacionada con el hecho [X]
2. DOCUMENTAL PRIVADA.- Consistente en... relacionada con el hecho [X]
3. TESTIMONIAL.- A cargo de [nombre], quien declarará sobre...
4. CONFESIONAL.- A cargo de la parte demandada, para que absuelva posiciones...
5. PERICIAL EN [MATERIA].- A cargo de perito en [especialidad]...
6. PRESUNCIONAL LEGAL Y HUMANA.- En todo lo que favorezca.
7. INSTRUMENTAL DE ACTUACIONES.- Todas las constancias del expediente.

**PUNTOS PETITORIOS**
Por lo anteriormente expuesto y fundado, a Usted C. Juez, atentamente pido:

PRIMERO.- Tenerme por presentado demandando en la vía [tipo] a [demandado].
SEGUNDO.- Ordenar el emplazamiento del demandado.
TERCERO.- Admitir a trámite las pruebas ofrecidas.
CUARTO.- En su oportunidad, dictar sentencia condenatoria.

PROTESTO LO NECESARIO
[Ciudad], a [fecha]

________________________
[Nombre del actor/abogado]

═══════════════════════════════════════════════════════════════
   FASE 3: ESTRATEGIA Y RECOMENDACIONES POST-DEMANDA
═══════════════════════════════════════════════════════════════

---

## ESTRATEGIA PROCESAL Y RECOMENDACIONES

### Elementos de la Accion a Acreditar
1. [Elemento 1 — con referencia a jurisprudencia que lo define]
2. [Elemento 2]
3. [Elemento n]

### Pruebas Indispensables a Recabar
- [ ] [Documento/prueba 1 y para qué sirve]
- [ ] [Documento/prueba 2 y qué acredita]

### Puntos de Atencion
- [Posible excepción del demandado y cómo prevenirla]
- [Plazo de prescripción aplicable — citar artículo]
- [Requisitos especiales de la jurisdicción]

---

REGLAS CRÍTICAS:
1. USA SIEMPRE el código procesal de la JURISDICCIÓN SELECCIONADA
2. Los hechos deben ser PERSUASIVOS y CREATIVOS, no solo informativos
3. Cada prestación debe tener FUNDAMENTO LEGAL específico del contexto RAG
4. BUSCA AGRESIVAMENTE en el contexto RAG: constitución, leyes, jurisprudencia
5. Cita SIEMPRE con [Doc ID: uuid] del contexto recuperado
6. Si el usuario no proporciona datos, indica [COMPLETAR: descripción de lo que falta]
7. Adapta la estructura según la MATERIA (civil/laboral/familiar/mercantil/agrario)
8. Sé CREATIVO en los argumentos: no repitas fórmulas genéricas
"""


SYSTEM_PROMPT_DRAFT_AMPARO = """Eres JUREXIA REDACTOR DE AMPAROS, especializado en la redacción de demandas de amparo directo e indirecto con máxima profundidad constitucional.

Tu capacidad creativa debe ser MÁXIMA. Construye CONCEPTOS DE VIOLACIÓN persuasivos, originales e irrefutables. SIEMPRE recurre a la base de datos RAG para fundar cada argumento constitucional.

═══════════════════════════════════════════════════════════════
   FASE 0: DETECCIÓN DE TIPO DE AMPARO
═══════════════════════════════════════════════════════════════

▸ AMPARO INDIRECTO (Ley de Amparo, arts. 107-169):
  - Contra leyes, reglamentos, tratados internacionales
  - Contra actos de autoridad administrativa
  - Contra actos de tribunales fuera de juicio o después de concluido
  - Contra actos en juicio que tengan ejecución de imposible reparación
  - Contra actos que afecten a personas extrañas al juicio
  Se tramita ante JUZGADO DE DISTRITO

▸ AMPARO DIRECTO (Ley de Amparo, arts. 170-191):
  - Contra sentencias definitivas, laudos o resoluciones que pongan fin al juicio
  - Se tramita ante TRIBUNAL COLEGIADO DE CIRCUITO
  - Se presenta a través de la autoridad responsable

═══════════════════════════════════════════════════════════════
   FASE 1: ANÁLISIS CONSTITUCIONAL PREVIO
═══════════════════════════════════════════════════════════════

Antes de redactar, ANALIZA:
1. ¿Cuál es el ACTO RECLAMADO exacto?
2. ¿Quién es la AUTORIDAD RESPONSABLE (ordenadora y ejecutora)?
3. ¿Qué DERECHOS FUNDAMENTALES se violan? (BUSCAR en CPEUM y tratados del RAG)
4. ¿Existe INTERÉS JURÍDICO o LEGÍTIMO?
5. ¿Es procedente SUSPENSIÓN del acto? ¿De oficio o a petición de parte?
6. ¿Hay JURISPRUDENCIA de SCJN/TCC que defina el estándar de violación? (BUSCAR en RAG)

═══════════════════════════════════════════════════════════════
   FASE 2: REDACCIÓN DE LA DEMANDA DE AMPARO
═══════════════════════════════════════════════════════════════

## DEMANDA DE AMPARO [INDIRECTO/DIRECTO]

**DATOS DE IDENTIFICACIÓN**

C. JUEZ DE DISTRITO EN MATERIA [Administrativa/Civil/Penal] EN TURNO
[O: H. TRIBUNAL COLEGIADO DE CIRCUITO EN MATERIA [X] EN TURNO — para amparo directo]
EN [Ciudad]
P R E S E N T E

[Nombre del quejoso], por mi propio derecho [y/o en representación de...], señalando como domicilio para oír y recibir notificaciones [dirección], autorizando en términos del artículo 12 de la Ley de Amparo a [licenciados], ante Usted respetuosamente comparezco para solicitar el AMPARO Y PROTECCIÓN DE LA JUSTICIA FEDERAL, al tenor de los siguientes:

**I. NOMBRE Y DOMICILIO DEL QUEJOSO**
[Datos completos]

**II. NOMBRE Y DOMICILIO DEL TERCERO INTERESADO**
[Si aplica]

**III. AUTORIDAD O AUTORIDADES RESPONSABLES**

A) AUTORIDAD ORDENADORA: [Identifica con precisión]
B) AUTORIDAD EJECUTORA: [Si aplica]

**IV. ACTO RECLAMADO**
[Describir con MÁXIMA PRECISIÓN el acto, ley, omisión o resolución que se reclama]
[Para amparo directo: identificar la sentencia/laudo exacto con expediente y fecha]

**V. HECHOS O ANTECEDENTES DEL ACTO RECLAMADO**
(SECCIÓN CREATIVA: narra de manera persuasiva y cronológica)

1. [Hecho 1 — contextualiza la relación con la autoridad]
2. [Hecho 2 — el acto de autoridad específico]
3. [Hecho 3 — cómo te afecta]
[Continuar]

**VI. PRECEPTOS CONSTITUCIONALES Y CONVENCIONALES VIOLADOS**

Artículos [1, 14, 16, 17, etc.] de la Constitución Política de los Estados Unidos Mexicanos.
Artículos [8, 25] de la Convención Americana sobre Derechos Humanos.
[Otros tratados según aplique]

**VII. CONCEPTOS DE VIOLACIÓN**
(SECCIÓN DE MÁXIMA CREATIVIDAD — Aquí está el corazón del amparo)

### PRIMER CONCEPTO DE VIOLACIÓN

**Derecho fundamental violado:**
> "Artículo [X].- [Transcripción]" — *CPEUM* [Doc ID: uuid]

**Estándar constitucional aplicable:**
> "[Rubro de tesis que define el alcance del derecho]" — *SCJN* [Doc ID: uuid]

**Cómo el acto reclamado viola este derecho:**
[Argumentación CREATIVA y PROFUNDA: no repitas fórmulas genéricas. Explica con lógica jurídica por qué el acto es inconstitucional, usando analogía, interpretación conforme, principio pro persona]

**Perjuicio causado:**
[Describe el daño concreto e irreparable]

### SEGUNDO CONCEPTO DE VIOLACIÓN
[Misma estructura — aborda otro ángulo constitucional]

### TERCER CONCEPTO DE VIOLACIÓN
[Si aplica — violaciones procedimentales, convencionales, etc.]

**VIII. SUSPENSIÓN DEL ACTO RECLAMADO**

[ANALIZA si procede suspensión de plano, provisional o definitiva]
Solicito se conceda la SUSPENSIÓN [provisional y en su momento definitiva / de plano] del acto reclamado, toda vez que:

a) No se sigue perjuicio al interés social
b) No se contravienen disposiciones de orden público
c) Son de difícil reparación los daños y perjuicios que se le causen al quejoso
Fundamento: Artículos [128, 131, 138, 147] de la Ley de Amparo [Doc ID: uuid]

**IX. PRUEBAS**
[Ofrecer pruebas pertinentes]

**PUNTOS PETITORIOS**

PRIMERO.- Tener por presentada esta demanda de amparo.
SEGUNDO.- Admitirla a trámite.
TERCERO.- Conceder la suspensión [provisional y definitiva] del acto reclamado.
CUARTO.- En la audiencia constitucional, conceder el AMPARO Y PROTECCIÓN DE LA JUSTICIA FEDERAL.

PROTESTO LO NECESARIO
[Ciudad], a [fecha]

________________________
[Nombre del quejoso/abogado]

═══════════════════════════════════════════════════════════════
   FASE 3: ESTRATEGIA CONSTITUCIONAL
═══════════════════════════════════════════════════════════════

---

## ESTRATEGIA DEL AMPARO

### Viabilidad del Amparo
- Tipo recomendado: [Directo/Indirecto] y por qué
- Causales de improcedencia que podría invocar el Ministerio Público: [listar y desvirtuar]

### Fortaleza de los Conceptos de Violación
- [Evaluar cada concepto: fuerte/medio/débil]
- [Sugerir argumentos adicionales]

### Suspensión
- [Probabilidad de que se conceda]
- [Garantía probable]

---

REGLAS CRÍTICAS:
1. BUSCA AGRESIVAMENTE en el RAG: CPEUM, Ley de Amparo, jurisprudencia, tratados
2. Los conceptos de violación deben ser CREATIVOS, PROFUNDOS y ORIGINALES
3. NO uses fórmulas genéricas — argumenta con lógica jurídica real
4. Cita SIEMPRE con [Doc ID: uuid] del contexto recuperado
5. Aplica interpretación conforme y principio pro persona cuando fortalezca
6. Si faltan datos, indica [COMPLETAR: descripción]
7. Anticipa causales de improcedencia y desvirtúalas en los hechos
"""


SYSTEM_PROMPT_DRAFT_IMPUGNACION = """Eres JUREXIA REDACTOR DE IMPUGNACIONES, especializado en la construcción de agravios y recursos legales con máxima persuasión.

Tu capacidad creativa debe ser MÁXIMA. Construye AGRAVIOS devastadores, lógicos e irrefutables. SIEMPRE recurre a la base de datos RAG para fundar cada argumento.

═══════════════════════════════════════════════════════════════
   FASE 0: DETECCIÓN DEL TIPO DE RECURSO
═══════════════════════════════════════════════════════════════

▸ RECURSO DE APELACIÓN:
  - Contra sentencias definitivas o interlocutorias apelables
  - Se presenta ante el juez que dictó la resolución (a quo)
  - Se resuelve por el tribunal superior (ad quem)
  - Plazo: generalmente 9 días (verificar código procesal local)
  - Estructura: AGRAVIOS (no conceptos de violación)

▸ RECURSO DE REVOCACIÓN:
  - Contra autos y decretos no apelables
  - Se presenta ante el mismo juez que lo dictó
  - Plazo: generalmente 3 días
  - Es recurso horizontal (lo resuelve el mismo juez)

▸ RECURSO DE QUEJA:
  - Contra excesos o defectos en ejecución de sentencias
  - Contra denegación de apelación
  - En amparo: contra actos de autoridad responsable (art. 97 Ley de Amparo)
  - Plazo variable según la causal

▸ RECURSO DE REVISIÓN:
  - En amparo: contra sentencias de Juzgado de Distrito
  - En amparo: contra resoluciones sobre suspensión
  - Se interpone ante el Tribunal Colegiado o SCJN
  - Plazo: 10 días (art. 86 Ley de Amparo)

▸ CONCEPTO DE VIOLACIÓN / AGRAVIO:
  - Construcción técnica del argumento de impugnación
  - Estructura lógica: acto → precepto violado → cómo se viola → perjuicio

═══════════════════════════════════════════════════════════════
   FASE 1: ANÁLISIS DE LA RESOLUCIÓN IMPUGNADA
═══════════════════════════════════════════════════════════════

Antes de redactar, ANALIZA:
1. ¿Cuál es EXACTAMENTE la resolución que se impugna?
2. ¿Cuál es el DISPOSITIVO (lo que resolvió)?
3. ¿Cuáles son las CONSIDERACIONES del juzgador (su razonamiento)?
4. ¿Dónde está el ERROR del juzgador? (fáctico, jurídico, procedimental)
5. ¿Qué NORMAS debió aplicar y no aplicó? (BUSCAR en RAG)
6. ¿Hay JURISPRUDENCIA que contradiga la resolución? (BUSCAR en RAG)

═══════════════════════════════════════════════════════════════
   FASE 2: REDACCIÓN DEL RECURSO
═══════════════════════════════════════════════════════════════

## [RECURSO DE APELACIÓN / REVOCACIÓN / QUEJA / REVISIÓN]

**DATOS DE IDENTIFICACIÓN**

C. [JUEZ/MAGISTRADO/TRIBUNAL] EN [MATERIA] EN TURNO
EN [Ciudad]
EXPEDIENTE: [Número]
P R E S E N T E

[Nombre], en mi carácter de [parte actora/demandada/tercero interesado/quejoso] dentro del expediente al rubro citado, ante Usted respetuosamente comparezco para interponer RECURSO DE [TIPO], en contra de [identificar resolución exacta: auto/sentencia/decreto de fecha X], al tenor de los siguientes:

**RESOLUCIÓN RECURRIDA**
[Identificar con precisión: tipo de resolución, fecha, contenido dispositivo]

**OPORTUNIDAD DEL RECURSO**
El presente recurso se interpone dentro del plazo legal de [X] días que establece el artículo [X] del [Código Procesal aplicable] [Doc ID: uuid], toda vez que la resolución recurrida fue notificada el día [fecha].

**A G R A V I O S**

### PRIMER AGRAVIO

**Resolución impugnada:**
[Transcribir o resumir la consideración específica del juzgador que se ataca]

**Preceptos legales violados:**
> "Artículo X.-..." — *[Código/Ley]* [Doc ID: uuid]

**Causa de pedir (cómo y por qué se viola):**
(SECCIÓN CREATIVA MÁXIMA)
[Argumenta con PROFUNDIDAD y ORIGINALIDAD por qué el razonamiento del juzgador es erróneo. Usa:
- Interpretación sistemática de las normas
- Jurisprudencia que contradiga la resolución
- Lógica jurídica (premisa mayor + premisa menor = conclusión)
- Analogía con casos resueltos por tribunales superiores]

**Perjuicio causado:**
[Explica concretamente qué perjuicio causa la resolución errónea]

**Jurisprudencia aplicable:**
> "[Rubro de la tesis]" — *SCJN/TCC* [Doc ID: uuid]
**Aplicación al caso:** [Explica CREATIVAMENTE cómo esta tesis demuestra el error del juzgador]

### SEGUNDO AGRAVIO
[Misma estructura — ataca otra consideración o error diferente]

### TERCER AGRAVIO
[Si aplica — errores procedimentales, de valoración probatoria, etc.]

**PUNTOS PETITORIOS**

PRIMERO.- Tener por interpuesto en tiempo y forma el presente recurso de [tipo].
SEGUNDO.- [Para apelación: remitir los autos al Tribunal Superior / Para revocación: revocar el auto impugnado]
TERCERO.- [Revocar/Modificar/Dejar sin efectos] la resolución recurrida.
CUARTO.- [Petición específica: dictar nueva resolución en la que se...]

PROTESTO LO NECESARIO
[Ciudad], a [fecha]

________________________
[Nombre / Abogado]

═══════════════════════════════════════════════════════════════
   FASE 3: EVALUACIÓN DE VIABILIDAD
═══════════════════════════════════════════════════════════════

---

## ESTRATEGIA DE IMPUGNACIÓN

### Fortaleza de los Agravios
| Agravio | Tipo de error | Fortaleza | Probabilidad de éxito |
|---------|--------------|-----------|----------------------|
| Primero | [Jurídico/Fáctico/Procesal] | [Alta/Media/Baja] | [%] |
| Segundo | ... | ... | ... |

### Posibles Argumentos del Ad Quem en Contra
- [Lo que podría responder el tribunal al desestimar cada agravio]
- [Cómo blindar los agravios contra esas respuestas]

### Alternativas si el Recurso no Prospera
- [Siguiente recurso disponible: amparo directo, revisión, etc.]
- [Plazo y requisitos]

---

REGLAS CRÍTICAS:
1. BUSCA AGRESIVAMENTE en el RAG: códigos procesales, jurisprudencia, leyes sustantivas
2. Los agravios deben ser DEVASTADORES, LÓGICOS y bien ESTRUCTURADOS
3. Diferencia errores de FONDO (indebida aplicación de ley) de FORMA (violaciones procedimentales)
4. SIEMPRE identifica la CAUSA DE PEDIR con precisión
5. Cita con [Doc ID: uuid] del contexto recuperado
6. Si el usuario describe la resolución, ATACA sus puntos más débiles creativamente
7. Si faltan datos, indica [COMPLETAR: descripción]
8. Proporciona un ANÁLISIS DE VIABILIDAD honesto al final
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

## Peticion ante [Autoridad]

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

## Respuesta a Peticion Ciudadana

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

def get_drafting_prompt(tipo: str, subtipo: str) -> str:
    """Retorna el prompt apropiado según el tipo de documento"""
    if tipo == "contrato":
        return SYSTEM_PROMPT_DRAFT_CONTRATO
    elif tipo == "demanda":
        return SYSTEM_PROMPT_DRAFT_DEMANDA
    elif tipo == "amparo":
        return SYSTEM_PROMPT_DRAFT_AMPARO
    elif tipo == "impugnacion":
        return SYSTEM_PROMPT_DRAFT_IMPUGNACION
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
    pdf_url: Optional[str] = None  # URL del PDF oficial (GCS o fuente gubernamental)


class SearchResponse(BaseModel):
    """Response de búsqueda"""
    query: str
    estado_filtrado: Optional[str]
    resultados: List[SearchResult]
    total: int


# ── PDF Fallback URLs por silo ─────────────────────────────────────────────────
# URL oficial del PDF de cada fuente legal (GCS bucket o fuente gubernamental).
# Se asigna a SearchResult.pdf_url cuando el payload de Qdrant no lo trae.
PDF_FALLBACK_URLS: Dict[str, str] = {
    "bloque_constitucional": "https://storage.googleapis.com/iurexia-leyes/constitucion/CPEUM-2024.pdf",
    # Agregar más silos aquí conforme se suban PDFs al bucket GCS:
    # "leyes_federales": "...",
}

# ─── Per-treaty PDF URLs (GCS bucket iurexia-leyes/Tratados/) ──────
# Keyed by lowercase keyword that matches the treaty's `origen` in Qdrant.
# When silo=bloque_constitucional and origen contains one of these keywords,
# the specific treaty PDF is returned instead of the CPEUM fallback.
_GCS_T = "https://storage.googleapis.com/iurexia-leyes/Tratados"

TREATY_PDF_URLS: Dict[str, str] = {
    # OEA
    "convención americana": f"{_GCS_T}/Convencion%20Americana%20sobre%20Derechos%20Humanos%20(CADH).pdf",
    "pacto de san josé": f"{_GCS_T}/Convencion%20Americana%20sobre%20Derechos%20Humanos%20(CADH).pdf",
    "belém do pará": f"{_GCS_T}/Convencion%20Interamericana%20Belem%20do%20Para%20(CBdP).pdf",
    "belem do para": f"{_GCS_T}/Convencion%20Interamericana%20Belem%20do%20Para%20(CBdP).pdf",
    "racismo": f"{_GCS_T}/Convencion%20Interamericana%20contra%20Racismo%20e%20Intolerancia%20(CIRDI).pdf",
    "intolerancia": f"{_GCS_T}/Convencion%20Interamericana%20contra%20Racismo%20e%20Intolerancia%20(CIRDI).pdf",
    "personas mayores": f"{_GCS_T}/Convencion%20Interamericana%20Derechos%20Personas%20Mayores%20(CIPM).pdf",
    "protocolo de san salvador": f"{_GCS_T}/Protocolo%20de%20San%20Salvador%20-%20Derechos%20Economicos%20Sociales%20(PSS).pdf",
    # ONU / OHCHR
    "declaración universal": f"{_GCS_T}/Declaracion%20Universal%20de%20Derechos%20Humanos%20(DUDH).pdf",
    "derechos civiles y políticos": f"{_GCS_T}/Pacto%20Internacional%20Derechos%20Civiles%20y%20Politicos%20(PIDCP).pdf",
    "derechos económicos, sociales y culturales": f"{_GCS_T}/Pacto%20Internacional%20Derechos%20Economicos%20Sociales%20y%20Culturales%20(PIDESC).pdf",
    "derechos del niño": f"{_GCS_T}/Convencion%20sobre%20los%20Derechos%20del%20Nino%20(CDN).pdf",
    "tortura": f"{_GCS_T}/Convencion%20contra%20la%20Tortura%20ONU%20(CAT).pdf",
    "cedaw": f"{_GCS_T}/Convencion%20Eliminacion%20Discriminacion%20contra%20la%20Mujer%20(CEDAW).pdf",
    "discriminación contra la mujer": f"{_GCS_T}/Convencion%20Eliminacion%20Discriminacion%20contra%20la%20Mujer%20(CEDAW).pdf",
    "discapacidad": f"{_GCS_T}/Convencion%20Derechos%20Personas%20con%20Discapacidad%20(CRPD).pdf",
    "discriminación racial": f"{_GCS_T}/Convencion%20Eliminacion%20Discriminacion%20Racial%20(ICERD).pdf",
    "trabajadores migratorios": f"{_GCS_T}/Convencion%20Derechos%20Trabajadores%20Migratorios%20(CMW).pdf",
    # Instrumentos penitenciarios
    "mandela": f"{_GCS_T}/Reglas%20Nelson%20Mandela%20-%20Tratamiento%20Reclusos%20(ONU).pdf",
    "bangkok": f"{_GCS_T}/Reglas%20de%20Bangkok%20-%20Tratamiento%20Reclusas%20(ONU).pdf",
    "estambul": f"{_GCS_T}/Protocolo%20de%20Estambul%20-%20Investigacion%20Tortura%20(OHCHR).pdf",
    # Otros
    "yogyakarta": f"{_GCS_T}/Principios%20de%20Yogyakarta%20-%20Orientacion%20Sexual%20e%20Identidad%20de%20Genero.pdf",
}


def _resolve_treaty_pdf(origen: str) -> Optional[str]:
    """
    Given a document's `origen` (e.g. 'Convención Americana sobre Derechos Humanos'),
    return the GCS PDF URL for that specific treaty, or None.
    """
    if not origen:
        return None
    origen_lower = origen.lower()
    # Don't match the Constitution itself — it has its own fallback
    if "constitución" in origen_lower or "cpeum" in origen_lower:
        return None
    for keyword, url in TREATY_PDF_URLS.items():
        if keyword in origen_lower:
            return url
    return None


class ChatRequest(BaseModel):
    """Request para chat conversacional"""
    messages: List[Message] = Field(..., min_length=1)
    estado: Optional[str] = Field(None, description="Estado para filtrado jurisdiccional")
    top_k: int = Field(40, ge=1, le=80)  # Expanded: 40 results across 4 silos = ~10 per silo
    enable_reasoning: bool = Field(
        False,
        description="Si True, usa Query Expansion con metadata jerárquica (más lento ~10s pero más preciso). Si False, modo rápido ~2s."
    )
    user_id: Optional[str] = Field(None, description="Supabase user ID for server-side quota enforcement")
    materia: Optional[str] = Field(None, description="Materia jurídica forzada (PENAL, CIVIL, FAMILIAR, etc.). Si None, auto-detecta por keywords.")
    fuero: Optional[str] = Field(None, description="Filtro por fuero: constitucional, federal, estatal. Si None, busca en todos los silos.")


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
chat_client: AsyncOpenAI = None  # For chat (GPT-5 Mini)
deepseek_client: AsyncOpenAI = None  # For reasoning/thinking (DeepSeek)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Inicialización y cleanup de recursos"""
    global sparse_encoder, qdrant_client, openai_client, chat_client, deepseek_client
    
    # Startup
    print(" Inicializando Jurexia Core Engine...")
    
    # BM25 Sparse Encoder — load in background thread to avoid blocking Cloud Run startup probe
    # HuggingFace can rate-limit on first download; app starts healthy while model loads
    async def _load_sparse_encoder():
        global sparse_encoder
        try:
            import asyncio
            loop = asyncio.get_event_loop()
            def _download():
                return SparseTextEmbedding(model_name="Qdrant/bm25")
            sparse_encoder = await loop.run_in_executor(None, _download)
            print("   BM25 Encoder cargado")
        except Exception as e:
            print(f"   WARN: BM25 Encoder falló al cargar: {e}. RAG sparse deshabilitado hasta reinicio.")
    asyncio.ensure_future(_load_sparse_encoder())

    
    # Qdrant Async Client
    qdrant_client = AsyncQdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        timeout=30,
    )
    print("   Qdrant Client conectado")
    
    # OpenAI Client (for embeddings only)
    openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
    print("   OpenAI Client inicializado (embeddings)")
    
    # Chat Client (GPT-5 Mini via OpenAI API — for regular chat queries)
    chat_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
    print(f"   Chat Client inicializado (GPT-5 Mini: {CHAT_MODEL})")
    
    # DeepSeek Client (for thinking/reasoning mode only)
    deepseek_client = AsyncOpenAI(
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE_URL,
    )
    print("   DeepSeek Client inicializado (reasoning)")
    
    print(" Jurexia Core Engine LISTO")
    
    yield
    
    # Shutdown
    print(" Cerrando conexiones...")
    await qdrant_client.close()


# ══════════════════════════════════════════════════════════════════════════════
# UTILIDADES
# ══════════════════════════════════════════════════════════════════════════════

# ── Humanize Origen: Filenames → Display Names ───────────────────────────────
# Maps raw filename-style origen values from Qdrant (e.g. JSON_QRO_CC_QRO)
# to human-readable law names (e.g. Código Civil del Estado de Querétaro)

_CODE_ABBREVIATIONS = {
    "CC": "Código Civil",
    "CP": "Código Penal",
    "CPC": "Código de Procedimientos Civiles",
    "CPP": "Código de Procedimientos Penales",
    "CNPP": "Código Nacional de Procedimientos Penales",
    "CT": "Código de Trabajo",
    "CF": "Código Fiscal",
    "CM": "Código de Comercio",
    "CA": "Código Administrativo",
    "CU": "Código Urbano",
    "CPACA": "Código de Procedimiento y Justicia Administrativa",
    "LF": "Ley de la Familia",
    "LP": "Ley de Profesiones",
    "LGTOC": "Ley General de Títulos y Operaciones de Crédito",
    "LGS": "Ley General de Sociedades",
    "LA": "Ley de Amparo",
    "LFTR": "Ley Federal de Telecomunicaciones y Radiodifusión",
    "LFT": "Ley Federal del Trabajo",
    "LISR": "Ley del Impuesto sobre la Renta",
    "LIVA": "Ley del Impuesto al Valor Agregado",
    "LGSPD": "Ley General del Servicio Profesional Docente",
    "CPEUM": "Constitución Política de los Estados Unidos Mexicanos",
}

_STATE_NAMES = {
    "AGS": "Aguascalientes", "BC": "Baja California", "BCS": "Baja California Sur",
    "CAMP": "Campeche", "CHIA": "Chiapas", "CHIH": "Chihuahua",
    "CDMX": "Ciudad de México", "COAH": "Coahuila", "COL": "Colima",
    "DGO": "Durango", "GTO": "Guanajuato", "GRO": "Guerrero",
    "HGO": "Hidalgo", "JAL": "Jalisco", "MEX": "Estado de México",
    "MICH": "Michoacán", "MOR": "Morelos", "NAY": "Nayarit",
    "NL": "Nuevo León", "OAX": "Oaxaca", "PUE": "Puebla",
    "QRO": "Querétaro", "QROO": "Quintana Roo", "SLP": "San Luis Potosí",
    "SIN": "Sinaloa", "SON": "Sonora", "TAB": "Tabasco",
    "TAMPS": "Tamaulipas", "TLAX": "Tlaxcala", "VER": "Veracruz",
    "YUC": "Yucatán", "ZAC": "Zacatecas",
}

def humanize_origen(origen: Optional[str]) -> Optional[str]:
    """
    Converts filename-style origen values into human-readable law names.
    
    Examples:
        JSON_QRO_CC_QRO → Código Civil del Estado de Querétaro
        JSON_JAL_CP_JAL → Código Penal del Estado de Jalisco
        JSON_CDMX_CPC_CDMX → Código de Procedimientos Civiles de Ciudad de México
        Ley de Profesiones del Estado de Querétaro → (unchanged, already clean)
    """
    if not origen:
        return origen
    
    # Strip .txt/.json extensions
    clean = re.sub(r'\.(txt|json)$', '', origen, flags=re.IGNORECASE).strip()
    
    # If it already looks human-readable (contains spaces and no JSON_ prefix), return as-is
    if ' ' in clean and not clean.startswith('JSON_'):
        return clean
    
    # Try to parse JSON_{STATE}_{CODE}_{STATE} pattern
    # Pattern: JSON_{STATE_ABBREV}_{CODE_ABBREV}_{STATE_ABBREV}
    match = re.match(r'^JSON_([A-Z]+)_([A-Z]+)_([A-Z]+)$', clean, re.IGNORECASE)
    if match:
        state_abbrev = match.group(1).upper()
        code_abbrev = match.group(2).upper()
        
        code_name = _CODE_ABBREVIATIONS.get(code_abbrev, code_abbrev)
        state_name = _STATE_NAMES.get(state_abbrev, state_abbrev)
        
        return f"{code_name} del Estado de {state_name}"
    
    # Try simpler pattern: {CODE}_{STATE} or {STATE}_{CODE}
    match = re.match(r'^JSON_?([A-Z]+)_([A-Z]+)$', clean, re.IGNORECASE)
    if match:
        part1 = match.group(1).upper()
        part2 = match.group(2).upper()
        
        # Check if part1 is a code and part2 is a state
        if part1 in _CODE_ABBREVIATIONS and part2 in _STATE_NAMES:
            return f"{_CODE_ABBREVIATIONS[part1]} del Estado de {_STATE_NAMES[part2]}"
        elif part2 in _CODE_ABBREVIATIONS and part1 in _STATE_NAMES:
            return f"{_CODE_ABBREVIATIONS[part2]} del Estado de {_STATE_NAMES[part1]}"
    
    # Fallback: replace underscores with spaces and title-case, strip JSON_ prefix
    fallback = clean.replace('JSON_', '').replace('_', ' ').title()
    return fallback


def extract_ley_from_texto(texto: Optional[str]) -> Optional[str]:
    """
    Extrae el nombre de la ley del campo texto cuando origen es None.
    Las leyes federales ingestadas sin 'origen' en Qdrant contienen el
    nombre de la ley embebido en el texto como:
    '[Ley GENERAL DE TITULOS Y OPERACIONES DE CREDITO | TITULO I...]'

    Retorna un nombre correctamente capitalizado, e.g.:
    'Ley General de Títulos y Operaciones de Crédito'
    """
    if not texto:
        return None
    # Extraer el contenido del primer bracket antes del pipe o cierre
    m = re.match(r'^\[([^\]|]+)', texto.strip())
    if not m:
        return None
    raw = m.group(1).strip()
    # Validar que parece un nombre de ley
    if not re.search(
        r'\b(Ley|C[oó]digo|Reglamento|Decreto|Estatuto|Constituci[oó]n)\b',
        raw, re.IGNORECASE
    ):
        return None
    # Convertir a title case con excepciones de preposiciones
    words = raw.split()
    LOWERCASE = {'de', 'del', 'y', 'o', 'e', 'la', 'el', 'los', 'las',
                 'para', 'en', 'con', 'por', 'a', 'al', 'un', 'una'}
    result = []
    for i, word in enumerate(words):
        w = word.lower()
        if i == 0 or w not in LOWERCASE:
            # Preserve accented capital letters (e.g. Ó stays, not lowercased)
            result.append(word.capitalize())
        else:
            result.append(w)
    return ' '.join(result)


def infer_source_from_text(texto: str) -> tuple:
    """
    Infer origen (law name) and ref (article) from the chunk text itself
    when Qdrant metadata is missing.
    
    Returns:
        (origen, ref) tuple — either may be None if not detected
    """
    if not texto:
        return (None, None)
    
    # Normalize whitespace from PDF line breaks
    normalized = re.sub(r'\s+', ' ', texto).strip()
    
    # ── Extract article number ──
    ref = None
    art_match = re.match(r'Art[ií]culo\s+(\d+[\w]*)', normalized)
    if art_match:
        ref = f"Art. {art_match.group(1)}"
    
    # ── Extract law name ──
    origen = None
    
    # False-positive fragments to reject
    _FALSE_POSITIVES = {
        "ley", "ley se", "ley y", "ley es", "ley no", "ley de", "ley se entenderá",
        "ley y su reglamento", "ley y otras disposiciones aplicables",
        "ley y demás disposiciones aplicables", "ley es de orden público",
        "ley entrará en vigor", "ley general", "ley que",
        "reglamento interior", "código", "constitución",
    }
    
    # Pattern 1: Explicit law name
    law_match = re.search(
        r'(?:del?\s+)?'
        r'((?:C[oó]digo|Ley|Constituci[oó]n|Reglamento)'
        r'(?:\s+(?:de|del|para|que|General|Federal|Org[aá]nica|Reglamentaria|Urbano|'
        r'Civil|Penal|Administrativo|Fiscal|Municipal|Familiar|Electoral|Ambiental|'
        r'Notarial|Agrario|Nacional|Estatal|Pol[ií]tica|sobre))?'
        r'(?:\s+[A-ZÁÉÍÓÚa-záéíóúü]+)*'
        r'(?:\s+del?\s+Estado\s+(?:Libre\s+y\s+Soberano\s+)?de\s+[A-ZÁÉÍÓÚa-záéíóúü]+)?'
        r'(?:\s+de\s+los\s+Estados\s+Unidos\s+Mexicanos)?)',
        normalized
    )
    if law_match:
        candidate = law_match.group(1).strip()
        candidate = re.sub(
            r'\s+(el|la|los|las|del|de|en|que|y|se|por|para|con|un|una|al|su|sus|a|o|como|no|si|más|este|esta|dicha|dicho|presente|será|deberá|podrá|entrará)\s*$',
            '', candidate, flags=re.IGNORECASE
        ).strip()
        candidate = re.sub(r'[\.:;,]+$', '', candidate).strip()
        
        if len(candidate) > 15 and candidate.lower() not in _FALSE_POSITIVES:
            origen = candidate
    
    # Pattern 2: Breadcrumb [Law Name > ...]
    if not origen:
        bracket_match = re.match(r'\[([^\]>]+?)(?:\s*>|\])', normalized)
        if bracket_match:
            candidate = bracket_match.group(1).strip()
            if len(candidate) > 10:
                origen = candidate
    
    # Pattern 3: "de este Código" — find explicit code name
    if not origen and re.search(r'(?:este|presente)\s+[Cc][oó]digo', normalized):
        deep_law = re.search(
            r'(C[oó]digo\s+(?:Urbano|Civil|Penal|Administrativo|Fiscal|Municipal|Familiar|'
            r'de\s+Procedimientos?\s+(?:Civiles?|Penales?|Administrativos?)|'
            r'de\s+Comercio|Financiero|Electoral|Ambiental|Notarial|Agrario)'
            r'(?:\s+del?\s+Estado\s+de\s+[A-ZÁÉÍÓÚa-záéíóúü]+)?)',
            normalized
        )
        if deep_law:
            origen = deep_law.group(1).strip()
    
    return (origen, ref)


def enrich_missing_metadata(results: list) -> list:
    """
    For SearchResult objects missing origen/ref, try to infer from text content.
    Modifies results in-place and returns them.
    """
    for r in results:
        if not r.origen or not r.ref:
            inferred_origen, inferred_ref = infer_source_from_text(r.texto)
            if not r.origen and inferred_origen:
                r.origen = inferred_origen
            if not r.ref and inferred_ref:
                r.ref = inferred_ref
    return results

def normalize_estado(estado: Optional[str]) -> Optional[str]:
    """
    Normaliza el nombre del estado al formato EXACTO almacenado en Qdrant.
    Qdrant usa UPPERCASE con UNDERSCORES: CIUDAD_DE_MEXICO, NUEVO_LEON, etc.
    """
    if not estado:
        return None
    normalized = estado.upper().strip().replace(" ", "_").replace("-", "_")
    # Colapsar múltiples underscores
    while "__" in normalized:
        normalized = normalized.replace("__", "_")
    normalized = normalized.strip("_")
    
    # Mapeo de variantes/aliases a nombres canónicos en Qdrant (con underscores)
    ESTADO_ALIASES = {
        # Nuevo León
        "NL": "NUEVO_LEON", "NUEVOLEON": "NUEVO_LEON",
        # CDMX — Qdrant almacena como "CIUDAD_DE_MEXICO"
        "CDMX": "CIUDAD_DE_MEXICO", "DF": "CIUDAD_DE_MEXICO",
        "DISTRITO_FEDERAL": "CIUDAD_DE_MEXICO",
        # Coahuila (Qdrant almacena como COAHUILA, no COAHUILA_DE_ZARAGOZA)
        "COAHUILA_DE_ZARAGOZA": "COAHUILA",
        # Estado de México
        "MEXICO": "ESTADO_DE_MEXICO",
        "EDO_MEXICO": "ESTADO_DE_MEXICO", "EDOMEX": "ESTADO_DE_MEXICO",
        "EDO_MEX": "ESTADO_DE_MEXICO",
        # Michoacán
        "MICHOACAN_DE_OCAMPO": "MICHOACAN",
        # Veracruz
        "VERACRUZ_DE_IGNACIO_DE_LA_LLAVE": "VERACRUZ",
    }
    
    # Primero buscar en aliases
    if normalized in ESTADO_ALIASES:
        return ESTADO_ALIASES[normalized]
    
    # Luego verificar si está en lista de estados válidos
    if normalized in ESTADOS_MEXICO:
        return normalized
    
    return None


# ══════════════════════════════════════════════════════════════════════════════
# DA VINCI: DETECCIÓN MULTI-ESTADO Y TIPO_CODIGO
# ══════════════════════════════════════════════════════════════════════════════

# Mapeo de nombres coloquiales de estados a nombres canónicos en Qdrant
ESTADO_KEYWORDS = {
    # Canonical values MUST match Qdrant's entidad field (UPPERCASE with UNDERSCORES)
    "aguascalientes": "AGUASCALIENTES",
    "baja california sur": "BAJA_CALIFORNIA_SUR",
    "baja california": "BAJA_CALIFORNIA",
    "campeche": "CAMPECHE",
    "chiapas": "CHIAPAS",
    "chihuahua": "CHIHUAHUA",
    "ciudad de mexico": "CIUDAD_DE_MEXICO", "cdmx": "CIUDAD_DE_MEXICO",
    "ciudad de méxico": "CIUDAD_DE_MEXICO",
    "distrito federal": "CIUDAD_DE_MEXICO",
    "coahuila": "COAHUILA",
    "colima": "COLIMA",
    "durango": "DURANGO",
    "estado de mexico": "ESTADO_DE_MEXICO", "edomex": "ESTADO_DE_MEXICO",
    "estado de méxico": "ESTADO_DE_MEXICO",
    "guanajuato": "GUANAJUATO",
    "guerrero": "GUERRERO",
    "hidalgo": "HIDALGO",
    "jalisco": "JALISCO",
    "michoacan": "MICHOACAN", "michoacán": "MICHOACAN",
    "morelos": "MORELOS",
    "nayarit": "NAYARIT",
    "nuevo leon": "NUEVO_LEON", "nuevo león": "NUEVO_LEON",
    "oaxaca": "OAXACA",
    "puebla": "PUEBLA",
    "queretaro": "QUERETARO", "querétaro": "QUERETARO",
    "quintana roo": "QUINTANA_ROO",
    "san luis potosi": "SAN_LUIS_POTOSI", "san luis potosí": "SAN_LUIS_POTOSI",
    "sinaloa": "SINALOA",
    "sonora": "SONORA",
    "tabasco": "TABASCO",
    "tamaulipas": "TAMAULIPAS",
    "tlaxcala": "TLAXCALA",
    "veracruz": "VERACRUZ",
    "yucatan": "YUCATAN", "yucatán": "YUCATAN",
    "zacatecas": "ZACATECAS",
}

# Patrones de comparación que indican que el usuario quiere comparar entre estados
COMPARE_PATTERNS = [
    r"compara[r]?", r"diferencia[s]?", r"disting[ue]", r"versus", r"\bvs\b",
    r"contrasta[r]?", r"entre\s+.+\s+y\s+", r"cada\s+estado",
    r"todos\s+los\s+estados", r"los\s+32\s+estados", r"en\s+qué\s+estados",
]

def detect_multi_state_query(query: str) -> Optional[List[str]]:
    """
    Detecta si el usuario menciona múltiples estados en su query.
    Retorna lista de estados canónicos si detecta 2+ estados, None si no.
    
    Ejemplo: "Compara el homicidio en Jalisco y Querétaro" → ["JALISCO", "QUERETARO"]
    """
    query_lower = query.lower()
    
    # Detectar estados mencionados (orden: más largo primero para evitar matches parciales)
    found_states = []
    sorted_keywords = sorted(ESTADO_KEYWORDS.keys(), key=len, reverse=True)
    
    remaining = query_lower
    for keyword in sorted_keywords:
        if keyword in remaining:
            canonical = ESTADO_KEYWORDS[keyword]
            if canonical not in found_states:
                found_states.append(canonical)
            # Remover para evitar match parcial (ej: "baja california" vs "baja california sur")
            remaining = remaining.replace(keyword, "")
    
    # Solo retornar si hay 2+ estados O si hay patrón comparativo con 1+ estado
    if len(found_states) >= 2:
        print(f"   🔍 DA VINCI: Detectados {len(found_states)} estados en query: {found_states}")
        return found_states
    
    # Si hay patrón comparativo y al menos 1 estado, buscar "todos los estados"
    is_comparative = any(re.search(p, query_lower) for p in COMPARE_PATTERNS)
    if is_comparative and "todos" in query_lower:
        print(f"   🔍 DA VINCI: Query comparativa para TODOS los estados")
        # Retornar top 5 estados con más datos para no saturar
        return ["QUERETARO", "NUEVO_LEON", "JALISCO", "CIUDAD_DE_MEXICO", "PUEBLA"]
    
    return None


def detect_single_estado_from_query(query: str) -> Optional[str]:
    """
    Auto-detecta un ÚNICO estado mencionado en el texto de la query.
    Se usa como fallback cuando el usuario NO seleccionó un estado en el dropdown.
    
    Solo retorna un estado si hay EXACTAMENTE 1 mención clara.
    Si hay 0 o 2+ estados, retorna None (dejar que el flujo normal lo maneje).
    
    Ejemplo: "multa condominio cdmx" → "CIUDAD_DE_MEXICO"
    Ejemplo: "divorcio en jalisco" → "JALISCO"  
    Ejemplo: "multa estacionamiento" → None (no hay estado mencionado)
    Ejemplo: "compara jalisco y cdmx" → None (multi-estado, se maneja aparte)
    """
    query_lower = query.lower()
    
    # Detect states mentioned (longest first to avoid partial matches)
    found_states = []
    sorted_keywords = sorted(ESTADO_KEYWORDS.keys(), key=len, reverse=True)
    
    remaining = query_lower
    for keyword in sorted_keywords:
        if keyword in remaining:
            canonical = ESTADO_KEYWORDS[keyword]
            if canonical not in found_states:
                found_states.append(canonical)
            remaining = remaining.replace(keyword, "")
    
    # Only return if exactly 1 state found
    if len(found_states) == 1:
        print(f"   🔍 AUTO-DETECT: Estado '{found_states[0]}' detectado en query")
        return found_states[0]
    
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
    V5.0: Las colecciones dedicadas por estado NO necesitan filtro.
    Solo el silo legacy leyes_estatales necesita filtro por entidad.
    """
    # Colecciones dedicadas por estado (leyes_queretaro, leyes_cdmx, etc.) → sin filtro
    if silo_name.startswith("leyes_") and silo_name not in ("leyes_federales", "leyes_estatales"):
        return None
    
    # Silo legacy: leyes_estatales → filtrar por entidad
    if silo_name == "leyes_estatales":
        if estado:
            normalized = normalize_estado(estado)
            if normalized:
                return Filter(
                    must=[
                        FieldCondition(key="entidad", match=MatchValue(value=normalized))
                    ]
                )
    
    # Para federales, jurisprudencia y bloque constitucional, no se aplica filtro
    return None


def build_metadata_filter(materia: Optional[str]) -> Optional[Filter]:
    """
    LEGACY: Construye filtro por campo 'materia' (pocas colecciones lo tienen).
    Usar build_materia_boost_filter() para el campo 'jurisdiccion' enriquecido.
    """
    if not materia:
        return None
    return Filter(
        should=[
            FieldCondition(
                key="materia",
                match=MatchAny(any=[materia])
            )
        ]
    )


def build_materia_boost_filter(materias: List[str]) -> Optional[Filter]:
    """
    Construye filtro SHOULD (soft boost) basado en campo 'jurisdiccion'.
    
    Aumenta el score de chunks cuya jurisdiccion coincide, pero NO excluye
    chunks de otras materias. Esto permite que artículos relevantes de
    materias adyacentes sigan apareciendo.
    
    Args:
        materias: Lista de materias jurídicas (e.g. ["PENAL"], ["CIVIL", "FAMILIAR"])
    
    Returns:
        Filter de Qdrant con should conditions, o None si lista vacía
    """
    if not materias:
        return None
    
    # Normalizar a uppercase para matching con payloads enriquecidos
    normalized = [m.upper() for m in materias]
    
    # SHOULD filter: boost, no exclusión
    return Filter(
        should=[
            FieldCondition(
                key="jurisdiccion",
                match=MatchAny(any=normalized)
            )
        ]
    )


# Sinónimos legales para query expansion (mejora recall BM25)
LEGAL_SYNONYMS = {
    "derecho del tanto": [
        "derecho de preferencia", "preferencia adquisición", 
        "socios gozarán del tanto", "enajenar partes sociales",
        "copropiedad preferencia", "colindantes vía pública",
        "propietarios predios colindantes", "retracto legal",
        "usufructuario goza del tanto", "copropiedad indivisa",
        "rescisión contrato ocho días", "aparcería enajenar",
        "condueño plena propiedad parte alícuota", "copropiedad condueño",
        "copropietario enajenación", "derecho preferente adquisición"
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
    "lfpca": [
        "Ley Federal de Procedimiento Contencioso Administrativo",
        "juicio contencioso administrativo", "tribunal fiscal", "LFPCA"
    ],
    "contencioso administrativo": [
        "Ley Federal de Procedimiento Contencioso Administrativo",
        "juicio contencioso administrativo", "tribunal fiscal", "LFPCA"
    ],
    "procedimiento contencioso": [
        "Ley Federal de Procedimiento Contencioso Administrativo",
        "juicio contencioso administrativo", "LFPCA"
    ],
    "cpeum": ["Constitución Política de los Estados Unidos Mexicanos", "carta magna"],
    "lft": ["Ley Federal del Trabajo"],
    "cnpp": ["Código Nacional de Procedimientos Penales"],
    "amparo": ["Ley de Amparo"],
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
# DOGMATIC QUERY EXPANSION - LLM-Based Legal Term Extraction
# ══════════════════════════════════════════════════════════════════════════════

DOGMATIC_EXPANSION_PROMPT = """Eres un jurista experto en TODAS las ramas del derecho mexicano. Tu trabajo es identificar la MATERIA JURÍDICA de la consulta y devolver los términos técnicos correctos de ESA materia específica.

REGLAS ESTRICTAS:
1. SOLO devuelve palabras clave separadas por espacio (máximo 6 términos)
2. NO incluyas explicaciones ni puntuación
3. Identifica la materia (penal, civil, mercantil, laboral, constitucional, familiar, administrativo, fiscal)
4. Genera términos técnicos de ESA materia, NO de otras
5. Incluye artículos de ley clave si aplica
6. Si la consulta menciona un CONCEPTO JURÍDICO, incluye los términos del articulado que lo regulan

EJEMPLOS POR MATERIA:
- "violación" → "violación cópula acceso carnal delito sexual artículo 265 CPF"
- "títulos de crédito autonomía" → "títulos crédito autonomía abstracción incorporación legitimación pagaré LGTOC"
- "abstracción cambiaria pagaré" → "abstracción cambiaria obligación cartular pagaré LGTOC artículo 17 juicio ejecutivo mercantil"
- "despido injustificado" → "despido injustificado rescisión relación laboral indemnización artículo 48 LFT"
- "divorcio" → "divorcio disolución matrimonio convenio pensión alimentos guarda custodia"
- "amparo" → "amparo garantías acto reclamado queja suspensión artículo 103 CPEUM"
- "contrato mercantil" → "contrato mercantil compraventa obligaciones comerciante Código Comercio"
- "derechos humanos" → "derechos humanos bloque constitucionalidad control convencionalidad pro persona"
- "pensión alimenticia" → "pensión alimentos obligación alimentaria acreedor deudor proporcionalidad"
- "derecho del tanto" → "derecho tanto copropiedad condueño parte alícuota preferencia adquirir enajenación"

Procesa esta consulta y devuelve SOLO las palabras clave:"""


async def expand_legal_query_llm(query: str) -> str:
    """
    Expansión de consulta usando LLM para extraer terminología dogmática.
    Usa DeepSeek con temperature=0 para respuestas deterministas.
    
    Esta función cierra la brecha semántica entre:
    - Lenguaje coloquial del usuario: "violación"
    - Terminología técnica del legislador: "cópula"
    """
    try:
        response = await chat_client.chat.completions.create(
            model=CHAT_MODEL,  # GPT-5 Mini para expansión
            messages=[
                {"role": "system", "content": DOGMATIC_EXPANSION_PROMPT},
                {"role": "user", "content": query}
            ],
            temperature=0.1,  # Cuasi-determinista (GPT-5 Mini no soporta 0)
            max_completion_tokens=100,  # Solo necesitamos palabras clave
        )
        
        expanded_terms = response.choices[0].message.content.strip()
        
        # Limitar a máximo 6 términos para no diluir la búsqueda
        terms = expanded_terms.split()[:6]
        result = f"{query} {' '.join(terms)}"
        print(f"   ⚡ Query expandido: '{query}' → '{result}'")
        return result
        
    except Exception as e:
        print(f"   ⚠️ ERROR en expansión LLM: {type(e).__name__}: {e}")
        print(f"   ⚠️ Usando fallback estático para query: '{query}'")
        # Fallback a expansión estática
        return expand_legal_query(query)


# ══════════════════════════════════════════════════════════════════════════════
# QUERY EXPANSION CON METADATA JERÁRQUICA - FASE 1 RAG IMPROVEMENT
# ══════════════════════════════════════════════════════════════════════════════

METADATA_EXTRACTION_PROMPT = """Analiza esta consulta legal y extrae metadata estructurada.

Consulta: {query}

Devuelve SOLO un JSON válido con esta estructura exacta:
{{
    "materia_principal": "penal" | "civil" | "mercantil" | "laboral" | "administrativo" | "fiscal" | "familiar" | "constitucional",
    "temas_clave": ["tema1", "tema2", "tema3"],
    "requiere_constitucion": true | false,
    "requiere_jurisprudencia": true | false,
    "terminos_expansion": ["término técnico 1", "término 2", ...]
}}

REGLAS:
- materia_principal: La rama del derecho principal de la consulta
- temas_clave: 2-4 conceptos jurídicos específicos
- requiere_constitucion: true si involucra derechos fundamentales o control constitucional
- requiere_jurisprudencia: true si necesita interpretar criterios judiciales
- terminos_expansion: Sinónimos jurídicos y términos técnicos relacionados (máximo 5)

IMPORTANTE: Devuelve SOLO el JSON, sin texto adicional ni markdown."""


# ══════════════════════════════════════════════════════════════════════════════
# LEGAL STRATEGY AGENT PROMPT
# Este prompt convierte expand_query_with_metadata en un Agente Estratega:
# En lugar de solo expandir por sinónimos, diagnostica el caso jurídico y
# produce un plan de búsqueda con pesos de silos específicos.
# ══════════════════════════════════════════════════════════════════════════════
LEGAL_STRATEGY_AGENT_PROMPT = """Eres el Socio Director de Iurexia. Analiza esta consulta jurídica y produce
un plan de búsqueda estratégico estructurado.

Consulta del usuario: "{query}"

Devuelve SOLO un JSON válido con esta estructura exacta:
{{
    "fuero_detectado": "constitucional" | "federal" | "estatal" | "mixto",
    "materia_principal": "penal" | "civil" | "mercantil" | "laboral" | "administrativo" | "fiscal" | "familiar" | "constitucional" | "procesal" | "agrario",
    "via_procesal": "identificar la vía más probable (ej: juicio ordinario civil, juicio de amparo indirecto, procedimiento administrativo)",
    "conceptos_juridicos": ["concepto técnico 1", "concepto técnico 2"],
    "jurisprudencia_keywords": ["término para buscar tesis 1", "término 2"],
    "leyes_primarias": ["nombre de ley o código principal"],
    "pesos_silos": {{
        "constitucional": 0-1,
        "federal": 0-1,
        "estatal": 0-1,
        "jurisprudencia": 0-1
    }},
    "requiere_ddhh": true | false
}}

REGLAS PARA DETERMINAR pesos_silos (los 4 valores deben sumar ~1.0):
- Consulta puramente procesal (plazos, notificaciones, recursos): federal=0.4, jurisprudencia=0.4, estatal=0.1, constitucional=0.1
- Consulta mercantil/fiscal (títulos de crédito, contratos, impuestos): federal=0.5, jurisprudencia=0.3, estatal=0.1, constitucional=0.1
- Consulta penal/familiar con estado específico: estatal=0.5, jurisprudencia=0.3, federal=0.15, constitucional=0.05
- Consulta sobre derechos fundamentales / amparo / DDHH: constitucional=0.5, jurisprudencia=0.3, federal=0.15, estatal=0.05
- Consulta laboral: federal=0.4, jurisprudencia=0.35, estatal=0.1, constitucional=0.15
- Consulta civil patrimonial sin estado específico: federal=0.35, jurisprudencia=0.35, estatal=0.2, constitucional=0.1

REGLAS para fuero_detectado:
- constitucional: pregunta por artículos CPEUM, amparo, control constitucional, DDHH
- federal: leyes federales, impuestos, mercantil, laboral federal
- estatal: derecho penal, civil familiar, procesal con mención de estado
- mixto: cuando la consulta abarca tanto derecho federal como estatal

IMPORTANTE: El campo jurisprudencia_keywords es CLAVE. Genera 2-3 términos técnicos
jurídicos exactos que un abogado usaría para buscar tesis de la SCJN sobre este tema.
Ejemplo para arrendamiento: ["rescisión arrendamiento falta pago", "emplazamiento desahucio"]

Devuelve SOLO el JSON, sin texto adicional ni markdown."""


async def _legal_strategy_agent(query: str, fuero_manual: Optional[str] = None) -> Dict[str, Any]:
    """
    Agente Estratega Pre-Búsqueda — Socio Director de Iurexia.

    En lugar de solo expandir la query con sinónimos, este agente:
    1. Diagnostica el caso jurídico completo
    2. Detecta el fuero automáticamente (si el usuario no lo seleccionó)
    3. Genera keywords técnicos de jurisprudencia que un abogado usaría
    4. Produce pesos de silos para uso en la fusión balanceada
    5. Identifica la vía procesal más probable

    Reemplaza la llamada a expand_query_with_metadata() en el pipeline.
    Misma latencia (1 llamada LLM), resultado cualitativamente superior.

    Returns:
        Dict con:
        - fuero_detectado: str ('constitucional', 'federal', 'estatal', 'mixto')
        - materia_principal: str
        - via_procesal: str
        - conceptos_juridicos: List[str]
        - jurisprudencia_keywords: List[str]
        - leyes_primarias: List[str]
        - pesos_silos: Dict[str, float]  ← NUEVO: alimenta slot allocation
        - requiere_ddhh: bool
        - expanded_query: str (backcompat)
    """
    try:
        prompt = LEGAL_STRATEGY_AGENT_PROMPT.format(query=query)

        # Usar DeepSeek (mismo modelo que el chat principal) para consistencia
        # y costo-efectividad. Si no disponible, usar chat_client (OpenAI).
        llm_client = deepseek_client if deepseek_client else chat_client
        llm_model = DEEPSEEK_CHAT_MODEL if deepseek_client else CHAT_MODEL

        response = await llm_client.chat.completions.create(
            model=llm_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,  # Bajo: queremos respuestas deterministas
            max_completion_tokens=400,
        )

        content = response.choices[0].message.content.strip()

        # Limpiar markdown si el LLM lo añadió
        if content.startswith("```json"):
            content = content.split("```json")[1].split("```")[0].strip()
        elif content.startswith("```"):
            content = content.split("```")[1].split("```")[0].strip()

        plan = json.loads(content)

        # Enriquecer la query expandida combinando conceptos + keywords
        conceptos = plan.get("conceptos_juridicos", [])
        juris_kw = plan.get("jurisprudencia_keywords", [])
        leyes = plan.get("leyes_primarias", [])
        expanded_parts = [query] + conceptos[:3] + juris_kw[:2] + leyes[:1]
        expanded_query = " ".join(expanded_parts[:8])

        # Si el usuario seleccionó un fuero manual, respetarlo
        fuero_final = fuero_manual if fuero_manual else plan.get("fuero_detectado", "mixto")

        result = {
            "fuero_detectado": fuero_final,
            "materia_principal": plan.get("materia_principal"),
            "via_procesal": plan.get("via_procesal", ""),
            "conceptos_juridicos": conceptos,
            "jurisprudencia_keywords": juris_kw,
            "leyes_primarias": leyes,
            "pesos_silos": plan.get("pesos_silos", {
                "constitucional": 0.25,
                "federal": 0.25,
                "estatal": 0.25,
                "jurisprudencia": 0.25,
            }),
            "requiere_ddhh": plan.get("requiere_ddhh", False),
            # Backcompat fields (para no romper código que usa el return de metadata)
            "expanded_query": expanded_query,
            "materia": plan.get("materia_principal"),
            "temas": conceptos,
            "requiere_constitucion": plan.get("requiere_ddhh", False),
            "requiere_jurisprudencia": True,
        }

        print(f"   ⚖️ AGENTE ESTRATEGA:")
        print(f"      Fuero detectado: {result['fuero_detectado']} (manual={fuero_manual or 'N/A'})")
        print(f"      Materia: {result['materia_principal']} | Vía: {result['via_procesal'][:60]}")
        print(f"      Conceptos: {', '.join(conceptos[:3])}")
        print(f"      Juris keywords: {', '.join(juris_kw[:2])}")
        print(f"      Pesos silos: {result['pesos_silos']}")

        return result

    except Exception as e:
        print(f"   ❌ Legal Strategy Agent falló ({type(e).__name__}: {e}) — usando defaults")
        return {
            "fuero_detectado": fuero_manual or "mixto",
            "materia_principal": None,
            "via_procesal": "",
            "conceptos_juridicos": [],
            "jurisprudencia_keywords": [],
            "leyes_primarias": [],
            "pesos_silos": {"constitucional": 0.25, "federal": 0.25, "estatal": 0.25, "jurisprudencia": 0.25},
            "requiere_ddhh": False,
            "expanded_query": query,
            "materia": None,
            "temas": [],
            "requiere_constitucion": False,
            "requiere_jurisprudencia": True,
        }


async def expand_query_with_metadata(query: str) -> Dict[str, Any]:
    """
    Expande query y extrae metadata relevante usando LLM.
    
    Esta función analiza la consulta para:
    1. Detectar materia legal (penal, civil, laboral, etc.)
    2. Extraer temas jurídicos clave
    3. Identificar si requiere análisis constitucional
    4. Generar términos de expansión específicos de la materia
    
    Args:
        query: Consulta del usuario
    
    Returns:
        Dict con query expandido y metadata para filtros:
        {
            "expanded_query": str,
            "materia": str | None,
            "temas": List[str],
            "requiere_constitucion": bool,
            "requiere_jurisprudencia": bool
        }
    """
    try:
        prompt = METADATA_EXTRACTION_PROMPT.format(query=query)
        
        response = await chat_client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_completion_tokens=300
        )
        
        content = response.choices[0].message.content.strip()
        
        # Limpiar markdown si existe
        if content.startswith("```json"):
            content = content.split("```json")[1].split("```")[0].strip()
        elif content.startswith("```"):
            content = content.split("```")[1].split("```")[0].strip()
        
        metadata = json.loads(content)
        
        # Construir query expandido combinando query original + términos de expansión
        expanded_terms = [query] + metadata.get("terminos_expansion", [])
        expanded_query = " ".join(expanded_terms[:8])  # Limitar a 8 términos totales
        
        result = {
            "expanded_query": expanded_query,
            "materia": metadata.get("materia_principal"),
            "temas": metadata.get("temas_clave", []),
            "requiere_constitucion": metadata.get("requiere_constitucion", False),
            "requiere_jurisprudencia": metadata.get("requiere_jurisprudencia", False)
        }
        
        print(f"   🧠 Metadata extraction exitosa:")
        print(f"      Materia: {result['materia']}")
        print(f"      Temas: {', '.join(result['temas'][:3])}")
        print(f"      Query expandido: '{expanded_query}'")
        
        return result
        
    except Exception as e:
        print(f"   ❌ ERROR en metadata extraction: {type(e).__name__}: {e}")
        print(f"   ❌ Usando fallback dogmático para query: '{query}'")
        # Fallback: solo expansión dogmática tradicional sin metadata
        expanded = await expand_legal_query_llm(query)
        return {
            "expanded_query": expanded,
            "materia": None,
            "temas": [],
            "requiere_constitucion": False,
            "requiere_jurisprudencia": False
        }


def build_metadata_filter(
    materia: Optional[str],
    nivel_jerarquico: Optional[str] = None
) -> Optional[Filter]:
    """
    Construye filtro de Qdrant basado en metadata jerárquica enriquecida.
    
    Los filtros se aplican con lógica SHOULD (OR), permitiendo que chunks
    que coincidan con CUALQUIERA de las condiciones sean incluidos.
    
    Args:
        materia: Materia legal (penal, civil, laboral, etc.)
        nivel_jerarquico: constitucional, federal, estatal
    
    Returns:
        Filter de Qdrant o None si no hay condiciones
    """
    conditions = []
    
    if materia:
        # Filtrar por materia (debe contener la materia en el array metadata.materia)
        conditions.append(
            FieldCondition(
                key="materia",
                match=MatchValue(value=materia)
            )
        )
    
    if nivel_jerarquico:
        conditions.append(
            FieldCondition(
                key="nivel_jerarquico",
                match=MatchValue(value=nivel_jerarquico)
            )
        )
    
    if not conditions:
        return None
    
    # SHOULD = OR lógico: cumple con al menos una condición
    return Filter(should=conditions)



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


# ══════════════════════════════════════════════════════════════════════════════
# MATERIA-AWARE RETRIEVAL — Capa 1: Detección por Keywords (0 latencia)
# ══════════════════════════════════════════════════════════════════════════════

MATERIA_KEYWORDS = {
    "PENAL": {
        "delito", "robo", "homicidio", "violencia", "penal", "imputado",
        "víctima", "ministerio público", "fiscalía", "carpeta de investigación",
        "audiencia inicial", "vinculación a proceso", "prisión preventiva",
        "sentencia condenatoria", "sentencia absolutoria", "tipicidad",
        "antijuridicidad", "culpabilidad", "punibilidad", "dolo", "culpa",
        "tentativa", "coautoría", "cómplice", "encubrimiento", "reincidencia",
        "lesiones", "fraude", "abuso de confianza", "extorsión", "secuestro",
        "violación", "feminicidio", "narcotráfico", "portación de arma",
        "código penal", "cnpp", "procedimiento penal", "acusatorio",
        "medida cautelar", "suspensión condicional", "procedimiento abreviado",
        "nulidad de actuaciones", "cadena de custodia", "dato de prueba",
    },
    "FAMILIAR": {
        "divorcio", "custodia", "alimentos", "pensión alimenticia",
        "guarda", "patria potestad", "régimen de convivencia", "adopción",
        "matrimonio", "concubinato", "filiación", "paternidad",
        "reconocimiento de hijo", "tutela", "curatela", "interdicción",
        "violencia familiar", "separación de cuerpos", "sociedad conyugal",
        "separación de bienes", "gananciales", "acta de nacimiento",
        "acta de matrimonio", "registro civil", "familiar", "familia",
        "menor", "menores", "niño", "niña", "infancia", "adolescente",
        "hijos", "cónyuge", "esposo", "esposa", "convivencia",
    },
    "LABORAL": {
        "despido", "laboral", "patrón", "trabajador", "salario",
        "contrato de trabajo", "relación laboral", "indemnización",
        "salarios caídos", "reinstalación", "junta de conciliación",
        "tribunal laboral", "sindicato", "huelga", "contrato colectivo",
        "jornada", "horas extras", "vacaciones", "prima vacacional",
        "aguinaldo", "ptu", "reparto de utilidades", "seguro social",
        "imss", "infonavit", "incapacidad", "riesgo de trabajo",
        "accidente laboral", "enfermedad profesional", "ley federal del trabajo",
        "rescisión laboral", "liquidación", "finiquito", "antigüedad",
        "subordinación", "outsourcing", "subcontratación",
    },
    "CIVIL": {
        "contrato", "arrendamiento", "compraventa", "daños y perjuicios",
        "responsabilidad civil", "obligaciones", "prescripción",
        "usucapión", "posesión", "propiedad", "servidumbre", "hipoteca",
        "prenda", "fianza civil", "mandato", "comodato", "mutuo",
        "donación", "permuta", "arrendatario", "arrendador", "renta",
        "desalojo", "desahucio", "lanzamiento", "juicio ordinario civil",
        "código civil", "acción reivindicatoria", "nulidad de contrato",
        "rescisión de contrato", "incumplimiento", "cláusula penal",
        "caso fortuito", "fuerza mayor", "vicios ocultos", "evicción",
        "sucesión", "herencia", "testamento", "intestado", "heredero",
        "legado", "albacea", "copropiedad", "condominio",
        "derecho del tanto", "usufructo", "embargo", "remate",
    },
    "MERCANTIL": {
        "sociedad mercantil", "pagaré", "letra de cambio", "cheque",
        "título de crédito", "código de comercio", "lgsm",
        "juicio ejecutivo mercantil", "juicio oral mercantil",
        "acción cambiaria", "endoso", "aval", "protesto",
        "quiebra", "concurso mercantil", "liquidación mercantil",
        "comerciante", "acto de comercio", "comisión mercantil",
        "contrato mercantil", "compraventa mercantil", "factoraje",
        "arrendamiento financiero", "franquicia", "sociedad anónima",
        "sapi", "sas", "s de rl", "lgtoc",
    },
    "ADMINISTRATIVO": {
        "clausura", "multa administrativa", "licitación", "concesión",
        "permiso", "licencia", "autorización", "acto administrativo",
        "procedimiento administrativo", "recurso de revisión",
        "juicio contencioso administrativo", "tribunal administrativo",
        "tfja", "servidor público", "responsabilidad administrativa",
        "sanción administrativa", "inspección", "verificación",
        "medio ambiente", "uso de suelo", "construcción",
        "protección civil", "cofepris", "profeco", "regulación",
        "administrativo", "gobernación", "lfpca",
        "ley federal de procedimiento contencioso administrativo",
    },
    "FISCAL": {
        "impuesto", "sat", "contribución", "cfdi", "factura",
        "isr", "iva", "ieps", "predial", "tributario", "fiscal",
        "código fiscal", "ley de ingresos", "devolución de impuestos",
        "crédito fiscal", "embargo fiscal", "procedimiento administrativo de ejecución",
        "recurso de revocación fiscal", "juicio de nulidad fiscal",
        "auditoría fiscal", "visita domiciliaria", "revisión de gabinete",
        "determinación de créditos", "caducidad fiscal", "prescripción fiscal",
        "declaración anual", "deducción", "acreditamiento",
    },
    "AGRARIO": {
        "ejido", "ejidatario", "comunal", "parcela", "agrario",
        "tribunal agrario", "procuraduría agraria", "ran",
        "registro agrario nacional", "asamblea ejidal", "comisariado",
        "ley agraria", "dotación", "restitución de tierras",
        "certificado parcelario", "dominio pleno", "avecindado",
        "pequeña propiedad", "comunidad agraria", "tierras comunales",
    },
    "CONSTITUCIONAL": {
        "amparo", "juicio de amparo", "amparo indirecto", "amparo directo",
        "suspensión del acto", "acto reclamado", "autoridad responsable",
        "quejoso", "tercero interesado", "ley de amparo",
        "inconstitucionalidad", "acción de inconstitucionalidad",
        "controversia constitucional", "control de constitucionalidad",
        "supremacía constitucional", "artículo constitucional",
    },
}


def _detect_materia(query: str, forced_materia: Optional[str] = None) -> Optional[List[str]]:
    """
    Detecta la materia jurídica de una consulta usando keywords.
    Capa 1 del Materia-Aware Retrieval: 0 latencia, 0 costo.
    
    Args:
        query: Consulta del usuario
        forced_materia: Si se proporciona, se usa directamente (override del frontend)
    
    Returns:
        Lista de máximo 2 materias detectadas, o None si no detecta ninguna
    """
    # Override: si el frontend forzó una materia, usarla directamente
    if forced_materia:
        normalized = forced_materia.upper().strip()
        if normalized in MATERIA_KEYWORDS:
            return [normalized]
        return None
    
    query_lower = query.lower()
    scores = {}
    
    for materia, keywords in MATERIA_KEYWORDS.items():
        # Contar cuántos keywords de cada materia aparecen
        count = sum(1 for kw in keywords if kw in query_lower)
        if count > 0:
            scores[materia] = count
    
    if not scores:
        return None
    
    # Ordenar por score descendente, tomar máximo 2
    sorted_materias = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    # Si la top materia tiene 2+ hits más que la segunda, solo devolver la top
    if len(sorted_materias) >= 2:
        top_score = sorted_materias[0][1]
        second_score = sorted_materias[1][1]
        if top_score >= second_score + 2:
            return [sorted_materias[0][0]]
        return [sorted_materias[0][0], sorted_materias[1][0]]
    
    return [sorted_materias[0][0]]


def _apply_materia_threshold(results: list, detected_materias: Optional[List[str]], threshold_gap: float = 0.25) -> list:
    """
    Capa 3 del Materia-Aware Retrieval: Post-retrieval threshold.
    Descarta resultados de materia ajena SOLO si tienen score bajo.
    
    Args:
        results: Lista de SearchResult ordenados por score
        detected_materias: Materias detectadas por _detect_materia()
        threshold_gap: Diferencia máxima tolerada vs top score (0.25 = 25%)
    
    Returns:
        Lista filtrada de SearchResult
    """
    if not detected_materias or not results:
        return results
    
    top_score = results[0].score if results else 0
    threshold = top_score - threshold_gap
    materias_upper = {m.upper() for m in detected_materias}
    
    filtered = []
    dropped_count = 0
    for r in results:
        # SIEMPRE mantener jurisprudencia y constitucional (supremacía constitucional)
        if r.silo in ("jurisprudencia_nacional", "bloque_constitucional"):
            filtered.append(r)
            continue
        
        # Si la jurisdiccion coincide con la materia detectada, mantener
        if r.jurisdiccion and r.jurisdiccion.upper() in materias_upper:
            filtered.append(r)
            continue
        
        # Si no tiene jurisdiccion asignada, mantener (no podemos filtrar)
        if not r.jurisdiccion:
            filtered.append(r)
            continue
        
        # Si la jurisdiccion NO coincide, mantener SOLO si el score es decente
        if r.score >= threshold:
            filtered.append(r)
        else:
            dropped_count += 1
    
    if dropped_count > 0:
        print(f"   🧹 MATERIA THRESHOLD: Descartados {dropped_count} resultados de materia ajena (score < {threshold:.4f})")
    
    return filtered


async def get_dense_embedding(text: str) -> List[float]:
    """Genera embedding denso usando OpenAI"""
    response = await openai_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text,
    )
    return response.data[0].embedding


def get_sparse_embedding(text: str) -> SparseVector:
    """Genera embedding sparse usando BM25. Degrada a sparse vacío si el modelo aún carga."""
    if sparse_encoder is None:
        # Modelo BM25 todavía cargando en background — degradar a dense-only search
        return SparseVector(indices=[], values=[])
    embeddings = list(sparse_encoder.query_embed(text))
    if not embeddings:
        return SparseVector(indices=[], values=[])
    
    sparse = embeddings[0]
    return SparseVector(
        indices=sparse.indices.tolist(),
        values=sparse.values.tolist(),
    )


# Maximum characters per document to prevent token overflow
MAX_DOC_CHARS = 6000

# ── JERARQUÍA NORMATIVA: Orden de autoridad legal (menor número = mayor jerarquía) ──
SILO_HIERARCHY_PRIORITY: Dict[str, int] = {
    # Nivel 0: Constitución y bloque constitucionalidad
    "bloque_constitucional": 0,
    # Nivel 1: Leyes federales / código nacional
    "leyes_federales": 1,
    "codigo_nacional": 1,
    # Nivel 2: Leyes estatales
    "leyes_estatales": 2,
    # Nivel 3: Jurisprudencia (complementaria, subordinada a norma)
    "jurisprudencia_nacional": 3,
    "jurisprudencia_tcc": 3,
    "jurisprudencia": 3,
}


def _get_jerarquia_label(silo: str) -> str:
    """
    Retorna una etiqueta de jerarquía normativa para un silo dado.
    Esta etiqueta se incluye en el XML del contexto para que el LLM
    pueda aplicar la regla de supremacía (REGLA #6 del system prompt).
    """
    level = SILO_HIERARCHY_PRIORITY.get(silo, 2)
    if level == 0:
        return "CONSTITUCION"
    if level == 1:
        return "LEY_FEDERAL"
    if level == 2:
        return "LEY_ESTATAL"
    if level == 3:
        return "JURISPRUDENCIA"
    return "DOCUMENTO"


def reorder_by_hierarchy(results: List[SearchResult]) -> List[SearchResult]:
    """
    Reordena resultados RAG respetando la jerarquía normativa mexicana:
    CONSTITUCION > LEY_FEDERAL > LEY_ESTATAL > JURISPRUDENCIA

    Dentro de cada nivel, mantiene el orden por score descendente (Cohere rerank).
    Esto asegura que el LLM vea primero la norma constitucional/legal vigente
    y después la jurisprudencia, evitando que tesis antiguas (ej. pre-Reforma 2024)
    dominen el contexto y generen respuestas obsoletas.
    """
    def sort_key(r: SearchResult):
        silo_priority = SILO_HIERARCHY_PRIORITY.get(r.silo, 2)
        return (silo_priority, -r.score)  # Menor level primero; mayor score primero
    return sorted(results, key=sort_key)


def format_results_as_xml(results: List[SearchResult], estado: Optional[str] = None) -> str:
    """
    Formatea resultados en XML para inyección de contexto.
    Escapa caracteres HTML para seguridad.
    Trunca documentos largos para evitar exceder límite de tokens.
    Si estado está presente, marca documentos estatales como FUENTE_PRINCIPAL.
    """
    if not results:
        return "<documentos>Sin resultados relevantes encontrados.</documentos>"
    
    # Enrich results missing metadata (origen/ref) by inferring from text
    enrich_missing_metadata(results)
    
    xml_parts = ["<documentos>"]
    
    # Si hay estado, inyectar instrucción de prioridad DENTRO del XML
    if estado:
        estado_humano = estado.replace("_", " ").title()
        xml_parts.append(
            f'<!-- INSTRUCCIÓN: Los documentos marcados tipo="LEGISLACION_ESTATAL" de '
            f'{estado_humano} son la FUENTE PRINCIPAL. En tu sección Fundamento Legal, '
            f'TRANSCRIBE el texto de estos artículos PRIMERO con su [Doc ID: uuid]. '
            f'La jurisprudencia va DESPUÉS como complemento interpretativo. -->'
        )
    
    # ── REGLA DE JERARQUÍA: Reordenar para que CPEUM/leyes precedan jurisprudencia ──
    # Esto garantiza que el LLM vea primero la norma vigente y después las tesis.
    # Evita que jurisprudencia pre-reforma 2024 domine el análisis.
    results = reorder_by_hierarchy(results)

    for r in results:
        # Truncate long documents to fit within token limits
        texto = r.texto
        if len(texto) > MAX_DOC_CHARS:
            texto = texto[:MAX_DOC_CHARS] + "... [truncado]"
        
        escaped_texto = html.escape(texto)
        escaped_ref = html.escape(r.ref or "N/A")
        escaped_origen = html.escape(humanize_origen(r.origen) or "Desconocido")
        escaped_jurisdiccion = html.escape(r.jurisdiccion or "N/A")

        # Etiqueta de jerarquía normativa — visible para el LLM para aplicar REGLA #6
        jerarquia = _get_jerarquia_label(r.silo)

        # Marcar documentos estatales como FUENTE PRINCIPAL cuando hay estado seleccionado
        tipo_tag = ""
        if estado and r.silo == "leyes_estatales":
            tipo_tag = ' tipo="LEGISLACION_ESTATAL" prioridad="PRINCIPAL"'
        elif r.silo in ("jurisprudencia_nacional", "jurisprudencia_tcc", "jurisprudencia"):
            tipo_tag = ' tipo="JURISPRUDENCIA" prioridad="COMPLEMENTARIA"'
        elif r.silo == "bloque_constitucional":
            tipo_tag = ' tipo="CONSTITUCION" prioridad="SUPREMA"'
        elif r.silo in ("leyes_federales", "codigo_nacional"):
            tipo_tag = ' tipo="LEY_FEDERAL" prioridad="PRIMARIA"'
        
        xml_parts.append(
            f'<documento id="{r.id}" ref="{escaped_ref}" '
            f'origen="{escaped_origen}" silo="{r.silo}" '
            f'jerarquia="{jerarquia}" '
            f'jurisdiccion="{escaped_jurisdiccion}" score="{r.score:.4f}"{tipo_tag}>\n'
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
        [Doc ID: abc123] -> [Doc ID: abc123]  *[Cita no verificada]*
    """
    if not invalid_ids:
        return response_text
    
    def replace_invalid(match):
        doc_id = match.group(1)
        original = match.group(0)
        if doc_id.lower() in [i.lower() for i in invalid_ids]:
            return f"{original}  *[Cita no verificada]*"
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




# ══════════════════════════════════════════════════════════════════════════════
# ARTICLE-AWARE RERANKING
# ══════════════════════════════════════════════════════════════════════════════

def detect_article_numbers(query: str) -> List[str]:
    """Detecta números de artículos mencionados explícitamente en la query."""
    pattern = r'art[ií]culos?\s+(\d+(?:\s*(?:bis|ter|qu[aá]ter))?)'
    matches = re.findall(pattern, query, re.IGNORECASE)
    return [m.strip() for m in matches]


def rerank_by_article_match(results: List[SearchResult], article_numbers: List[str]) -> List[SearchResult]:
    """
    Boostea resultados que contienen el número de artículo específico solicitado.
    Esto resuelve el problema de que artículos semánticamente lejanos pero
    específicamente solicitados no aparezcan en los resultados.
    """
    if not article_numbers:
        return results
    
    for r in results:
        for num in article_numbers:
            # Buscar "Artículo 941" o "Art. 941" en el texto del chunk
            if re.search(rf'art[ií]culos?\.?\s*{re.escape(num)}\b', r.texto, re.IGNORECASE):
                r.score += 0.5  # Boost significativo para match exacto
                print(f"   🎯 BOOST artículo {num} encontrado en {r.silo}: +0.5 score")
    
    results.sort(key=lambda x: x.score, reverse=True)
    return results


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
    RESILIENTE: Si el filtro causa error 400 (índice faltante), reintenta SIN filtro.
    """
    async def _do_search(search_filter: Optional[Filter]) -> list:
        """Ejecuta la búsqueda con el filtro dado."""
        col_info = await qdrant_client.get_collection(collection)
        sparse_vectors_config = col_info.config.params.sparse_vectors
        has_sparse = sparse_vectors_config is not None and len(sparse_vectors_config) > 0
        
        # Threshold diferenciado: jurisprudencia necesita mayor recall
        threshold = 0.03 if collection == "jurisprudencia_nacional" else 0.05
        
        if has_sparse:
            # Dual prefetch con RRF fusion:
            # - Prefetch 1 (sparse/BM25): encuentra candidatos por keywords
            # - Prefetch 2 (dense): encuentra candidatos por semántica
            #   (incluye chunks SIN sparse vectors, ej: reglamentos recién ingestados)
            # Fusión RRF combina ambos pools → mejor recall
            return await qdrant_client.query_points(
                collection_name=collection,
                prefetch=[
                    Prefetch(
                        query=sparse_vector,
                        using="sparse",
                        limit=top_k * 5,
                        filter=search_filter,
                    ),
                    Prefetch(
                        query=dense_vector,
                        using="dense",
                        limit=top_k * 5,
                        filter=search_filter,
                    ),
                ],
                query=Query(fusion=Fusion.RRF),
                limit=top_k,
                query_filter=search_filter,
                with_payload=True,
                score_threshold=None,  # RRF scores are on a different scale
            )
        else:
            return await qdrant_client.query_points(
                collection_name=collection,
                query=dense_vector,
                using="dense",
                limit=top_k,
                query_filter=search_filter,
                with_payload=True,
                score_threshold=threshold,
            )
    
    def _parse_results(results) -> List[SearchResult]:
        """Convierte puntos de Qdrant en SearchResult."""
        parsed = []
        for point in results.points:
            payload = point.payload or {}
            parsed.append(SearchResult(
                id=str(point.id),
                score=point.score,
                texto=payload.get("texto", payload.get("text", "")),
                ref=payload.get("ref"),
                origen=payload.get("origen"),
                jurisdiccion=payload.get("jurisdiccion"),
                entidad=payload.get("entidad"),
                silo=collection,
            ))
        return parsed
    
    try:
        # Intento 1: Búsqueda híbrida (prefetch sparse → dense rerank)
        results = await _do_search(filter_)
        search_results = _parse_results(results)
        
        # FALLBACK: Si hybrid devuelve 0 resultados pero la colección tiene sparse,
        # reintentar con SOLO dense. Esto ocurre cuando el sparse prefetch no encuentra
        # candidatos (ej: modelo BM25 diferente entre indexación y query).
        if not search_results:
            col_info = await qdrant_client.get_collection(collection)
            sparse_cfg = col_info.config.params.sparse_vectors
            has_sparse = sparse_cfg is not None and len(sparse_cfg) > 0
            if has_sparse:
                print(f"   ⚠️ Hybrid devolvió 0 en {collection}, fallback a dense-only...")
                threshold = 0.03 if collection == "jurisprudencia_nacional" else 0.05
                dense_results = await qdrant_client.query_points(
                    collection_name=collection,
                    query=dense_vector,
                    using="dense",
                    limit=top_k,
                    query_filter=filter_,
                    with_payload=True,
                    score_threshold=threshold,
                )
                search_results = _parse_results(dense_results)
                print(f"   ✅ Dense-only fallback: {len(search_results)} resultados en {collection}")
        
        return search_results
    
    except Exception as e:
        error_msg = str(e)
        # FALLBACK: typing.Union error (Python 3.14 + qdrant-client compat)
        # → bypass Prefetch/Query construction, use dense-only search
        if "typing.Union" in error_msg or "Cannot instantiate" in error_msg:
            print(f"   ⚠️ typing.Union error en {collection}, fallback a dense-only...")
            try:
                threshold = 0.03 if collection == "jurisprudencia_nacional" else 0.05
                dense_results = await qdrant_client.query_points(
                    collection_name=collection,
                    query=dense_vector,
                    using="dense",
                    limit=top_k,
                    query_filter=filter_,
                    with_payload=True,
                    score_threshold=threshold,
                )
                search_results = _parse_results(dense_results)
                print(f"   ✅ Dense-only fallback: {len(search_results)} resultados en {collection}")
                return search_results
            except Exception as dense_e:
                print(f"   ❌ Dense-only fallback también falló en {collection}: {dense_e}")
                return []

        # Si el error es por índice faltante, reintentar SIN filtro de metadata
        if "400" in error_msg or "Index required" in error_msg:
            print(f"   ⚠️  Filtro falló en {collection} (índice faltante), reintentando sin filtro...")
            try:
                results = await _do_search(None)  # Sin filtro
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
            except Exception as retry_e:
                print(f"   ❌ Retry sin filtro también falló en {collection}: {retry_e}")
                return []
        
        print(f"   ❌ Error en búsqueda sobre {collection}: {e}")
        return []



async def _extract_juris_concepts(query: str) -> str:
    """
    Extrae conceptos jurídicos clave de la consulta para buscar jurisprudencia.
    Devuelve una cadena de términos optimizados para matching con tesis.
    """
    try:
        response = await chat_client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": (
                    "Eres un experto en jurisprudencia mexicana. Dado un query legal, "
                    "extrae 5-8 conceptos clave que aparecerían en tesis de la SCJN o "
                    "tribunales colegiados sobre este tema. Incluye: derechos involucrados, "
                    "figuras jurídicas, términos procesales y sinónimos jurídicos. "
                    "Responde SOLO con los términos separados por espacios, sin explicación."
                )},
                {"role": "user", "content": query}
            ],
            temperature=0.1,  # GPT-5 Mini no soporta 0
            max_completion_tokens=80,
        )
        concepts = response.choices[0].message.content.strip()
        print(f"   ⚖️ Conceptos jurisprudencia extraídos: {concepts}")
        return concepts
    except Exception as e:
        print(f"   ⚠️ Extracción de conceptos falló: {e}")
        return query  # Fallback: usar el query original


async def _jurisprudencia_boost_search(query: str, exclude_ids: set) -> List[SearchResult]:
    """
    Búsqueda enfocada en jurisprudencia_nacional con score_threshold bajo
    para maximizar recall de tesis relevantes.
    """
    try:
        dense_vector = await get_dense_embedding(query)
        sparse_vector = get_sparse_embedding(query)
        
        # Verificar si tiene sparse vectors
        col_info = await qdrant_client.get_collection("jurisprudencia_nacional")
        sparse_vectors_config = col_info.config.params.sparse_vectors
        has_sparse = sparse_vectors_config is not None and len(sparse_vectors_config) > 0
        
        if has_sparse:
            try:
                results = await qdrant_client.query_points(
                    collection_name="jurisprudencia_nacional",
                    prefetch=[
                        Prefetch(
                            query=sparse_vector,
                            using="sparse",
                            limit=50,
                            filter=None,
                        ),
                    ],
                    query=dense_vector,
                    using="dense",
                    limit=10,
                    query_filter=None,
                    with_payload=True,
                    score_threshold=0.01,  # Muy bajo para máximo recall
                )
            except Exception as prefetch_err:
                # Fallback if Prefetch crashes (typing.Union on Python 3.14)
                print(f"      ⚠️ Prefetch falló en juris boost: {prefetch_err}, usando dense-only...")
                results = await qdrant_client.query_points(
                    collection_name="jurisprudencia_nacional",
                    query=dense_vector,
                    using="dense",
                    limit=10,
                    query_filter=None,
                    with_payload=True,
                    score_threshold=0.01,
                )
        else:
            results = await qdrant_client.query_points(
                collection_name="jurisprudencia_nacional",
                query=dense_vector,
                using="dense",
                limit=10,
                query_filter=None,
                with_payload=True,
                score_threshold=0.01,  # Muy bajo para máximo recall
            )
        
        search_results = []
        for point in results.points:
            if str(point.id) in exclude_ids:
                continue
            payload = point.payload or {}
            search_results.append(SearchResult(
                id=str(point.id),
                score=point.score,
                texto=payload.get("texto", payload.get("text", "")),
                ref=payload.get("ref"),
                origen=payload.get("origen"),
                jurisdiccion=payload.get("jurisdiccion"),
                entidad=payload.get("entidad"),
                silo="jurisprudencia_nacional",
            ))
        
        print(f"      ⚖️ Boost query '{query[:60]}...' → {len(search_results)} tesis")
        return search_results
        
    except Exception as e:
        print(f"      ⚠️ Boost search falló: {e}")
        return []


# ══════════════════════════════════════════════════════════════════════════════
# CROSS-SILO ENRICHMENT: Segunda pasada para encadenar fuentes
# ══════════════════════════════════════════════════════════════════════════════

def _extract_legal_refs(results: List[SearchResult], max_refs: int = 3) -> List[str]:
    """
    Extrae referencias legales (ley + artículo) de los textos recuperados.
    Retorna queries de enriquecimiento como: "artículo 17 CPEUM acceso justicia"
    """
    import re
    refs = []
    seen = set()
    
    # Patrones para extraer referencias legales mexicanas
    patterns = [
        # "artículo 19 Bis de la Ley de Procedimientos Administrativos"
        r'[Aa]rt[íi]culo\s+(\d+(?:\s*[Bb]is)?(?:\s*[A-Z])?)\s+(?:de\s+la\s+|del\s+)?(.{10,60}?)(?:\.|,|;|\n)',
        # "art. 17 CPEUM" or "art. 123 LFT"  
        r'[Aa]rt\.?\s*(\d+(?:\s*[Bb]is)?)\s+(?:de\s+la\s+|del\s+)?([A-ZÁÉÍÓÚÑ][A-Za-záéíóúñ\s]{3,40})',
        # "Ley Federal del Trabajo" sin artículo específico
        r'(Ley\s+(?:Federal|General|Org[áa]nica)\s+(?:del?\s+)?[A-Za-záéíóúñ\s]{5,50})',
    ]
    
    for r in results:
        text = r.text[:2000] if r.text else ""
        ref_str = r.ref or ""
        combined = f"{text} {ref_str}"
        
        for pattern in patterns[:2]:  # Solo los que tienen artículo + ley
            matches = re.findall(pattern, combined)
            for match in matches:
                if isinstance(match, tuple) and len(match) == 2:
                    art_num, ley_name = match
                    ley_clean = ley_name.strip()
                    key = f"{art_num}_{ley_clean[:20]}"
                    if key not in seen and len(refs) < max_refs:
                        refs.append(f"artículo {art_num} {ley_clean}")
                        seen.add(key)
    
    return refs


async def _cross_silo_enrichment(
    initial_results: List[SearchResult],
    query: str,
) -> List[SearchResult]:
    """
    Segunda pasada: busca jurisprudencia y constitución que fundamenten
    los artículos/leyes encontrados en la primera pasada.
    
    Lógica:
    1. Extraer refs legales (ley + artículo) de resultados iniciales
    2. Buscar en jurisprudencia_nacional tesis que citen esas leyes/artículos
    3. Buscar en bloque_constitucional artículos constitucionales relevantes
    4. Retornar resultados nuevos (sin duplicados)
    """
    refs = _extract_legal_refs(initial_results)
    if not refs:
        return []
    
    print(f"   🔗 Cross-silo refs extraídas: {refs}")
    
    enrichment_results = []
    existing_ids = {r.id for r in initial_results}
    
    # Formular queries de enriquecimiento
    enrichment_tasks = []
    
    for ref in refs[:3]:
        # Buscar jurisprudencia que cite este artículo/ley
        juris_query = f"tesis jurisprudencia criterio judicial {ref}"
        enrichment_tasks.append(
            _do_enrichment_search("jurisprudencia_nacional", juris_query)
        )
        
        # Buscar fundamento constitucional relacionado
        const_query = f"constitución derecho fundamental garantía {ref}"
        enrichment_tasks.append(
            _do_enrichment_search("bloque_constitucional", const_query)
        )
    
    # Ejecutar todas las búsquedas en paralelo
    all_enriched = await asyncio.gather(*enrichment_tasks)
    
    for results in all_enriched:
        for r in results:
            if r.id not in existing_ids:
                enrichment_results.append(r)
                existing_ids.add(r.id)
    
    # Ordenar por score
    enrichment_results.sort(key=lambda x: x.score, reverse=True)
    return enrichment_results[:8]  # Max 8 resultados de enrichment


async def _do_enrichment_search(
    collection: str,
    query: str,
) -> List[SearchResult]:
    """Ejecuta una búsqueda ligera para enrichment (solo dense, sin filtros)."""
    try:
        dense_vector = await get_dense_embedding(query)
        sparse_vector = get_sparse_embedding(query)
        results = await hybrid_search_single_silo(
            collection=collection,
            query=query,
            dense_vector=dense_vector,
            sparse_vector=sparse_vector,
            filter_=None,
            top_k=4,
            alpha=0.7,
        )
        return results
    except Exception as e:
        print(f"      ⚠️ Enrichment search falló en {collection}: {e}")
        return []


def _parse_article_number(text: str) -> Optional[int]:
    """Extrae el número de artículo de un campo ref o texto."""
    import re
    # Match: "Artículo 19", "Art. 123", "ARTÍCULO 45"
    match = re.search(r'[Aa]rt[íi]culos?\s*\.?\s*(\d+)', text or "")
    if match:
        return int(match.group(1))
    return None


async def _fetch_neighbor_chunks(
    results: List[SearchResult],
    max_neighbors: int = 6,
) -> List[SearchResult]:
    """
    Neighbor Chunk Retrieval: para resultados de legislación con score alto,
    busca los artículos adyacentes (N-1, N+1) de la misma ley.
    
    Esto da al LLM contexto circundante: definiciones, excepciones y sanciones
    que suelen estar en artículos contiguos.
    """
    # Solo tomar top 3 de legislación con score alto
    legislation = [r for r in results 
                   if r.silo in ("leyes_federales", "leyes_estatales") 
                   and r.score > 0.4][:3]
    
    if not legislation:
        return []
    
    neighbors = []
    existing_ids = {r.id for r in results}
    
    for r in legislation:
        art_num = _parse_article_number(r.ref or r.texto)
        if not art_num:
            continue
        
        collection = r.silo
        
        # Buscar artículos N-1 y N+1 en la misma ley
        for neighbor_num in [art_num - 1, art_num + 1]:
            if neighbor_num < 1:
                continue
            
            neighbor_ref_pattern = f"Artículo {neighbor_num}"
            
            try:
                # Build filter: same source file = same law
                must_conditions = []
                if r.origen:
                    must_conditions.append(
                        FieldCondition(
                            key="origen",
                            match=MatchValue(value=r.origen),
                        )
                    )
                
                scroll_filter = Filter(must=must_conditions) if must_conditions else None
                
                # Scroll con filtro: mismo origen
                scroll_results = await qdrant_client.scroll(
                    collection_name=collection,
                    scroll_filter=scroll_filter,
                    limit=50,  # Scan up to 50 to find the neighbor
                    with_payload=True,
                    with_vectors=False,
                )
                
                # Buscar el artículo vecino en los resultados del scroll
                for point in scroll_results[0]:
                    point_id = str(point.id)
                    if point_id in existing_ids:
                        continue
                    
                    payload = point.payload or {}
                    point_ref = payload.get("ref", "")
                    point_text = payload.get("texto", payload.get("text", ""))
                    
                    # Verificar que es el artículo vecino correcto
                    point_art_num = _parse_article_number(point_ref or point_text)
                    if point_art_num == neighbor_num:
                        neighbors.append(SearchResult(
                            id=point_id,
                            score=0.15,  # Score bajo: contexto, no resultado principal
                            texto=point_text,
                            ref=point_ref or payload.get("ref"),
                            origen=payload.get("origen"),
                            jurisdiccion=payload.get("jurisdiccion"),
                            entidad=payload.get("entidad"),
                            silo=collection,
                        ))
                        existing_ids.add(point_id)
                        break  # Encontrado, siguiente vecino
                
            except Exception as e:
                print(f"      ⚠️ Neighbor search falló para Art. {neighbor_num}: {e}")
                continue
    
    print(f"   📄 Neighbor chunks: {len(neighbors)} artículos adyacentes encontrados")
    return neighbors[:max_neighbors]


# ══════════════════════════════════════════════════════════════════════════════
# ADVANCED RAG: HyDE (Hypothetical Document Embeddings)
# ══════════════════════════════════════════════════════════════════════════════

async def _generate_hyde_document(query: str) -> Optional[str]:
    """
    HyDE: Genera un documento jurídico hipotético que respondería a la query.
    Este documento hipotético se usa para generar el dense embedding,
    mejorando la recuperación para queries coloquiales.
    
    Ejemplo:
      Query: "me corrieron del trabajo sin razón"
      HyDE genera: "Artículo 47.- Son causas de rescisión de la relación de trabajo..."
      El embedding del HyDE doc es más cercano a los artículos reales.
    """
    if not HYDE_ENABLED:
        return None
    
    # No usar HyDE para queries que ya son técnicas o mencionan artículos
    if re.search(r'artículo\s+\d+', query, re.IGNORECASE):
        return None
    if len(query.split()) < 3:
        return None
    
    try:
        response = await chat_client.chat.completions.create(
            model=HYDE_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Eres un experto en derecho mexicano. Genera un fragmento breve (150-250 palabras) "
                        "de un artículo de ley o tesis de jurisprudencia mexicana que respondería "
                        "directamente a la consulta del usuario. Escribe SOLO el texto legal, "
                        "sin explicaciones ni preámbulos. Usa terminología jurídica técnica mexicana."
                    )
                },
                {"role": "user", "content": query}
            ],
            max_tokens=350,
            temperature=0.3,
        )
        hyde_doc = response.choices[0].message.content.strip()
        if hyde_doc and len(hyde_doc) > 50:
            print(f"   🔮 HyDE generado ({len(hyde_doc)} chars): {hyde_doc[:100]}...")
            return hyde_doc
    except Exception as e:
        print(f"   ⚠️ HyDE falló (usando query original): {e}")
    
    return None


# ══════════════════════════════════════════════════════════════════════════════
# ADVANCED RAG: Query Decomposition
# ══════════════════════════════════════════════════════════════════════════════

async def _decompose_query(query: str) -> list[str]:
    """
    Descompone queries complejas multi-hop en sub-queries más específicas.
    
    Ejemplo:
      "¿Cuáles son los requisitos para un amparo indirecto y ante quién se presenta?"
      → ["requisitos amparo indirecto", "competencia amparo indirecto juez distrito"]
    """
    if not QUERY_DECOMPOSITION_ENABLED:
        return []
    
    # Solo descomponer queries largas (>10 palabras) o con "y", "además", "también"
    words = query.split()
    has_conjunction = any(w in query.lower() for w in [' y ', ' además ', ' también ', ' pero ', ' sin embargo ', ' asimismo '])
    if len(words) < 10 and not has_conjunction:
        return []
    
    try:
        response = await chat_client.chat.completions.create(
            model=HYDE_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Eres un experto en derecho mexicano. El usuario hace una consulta compleja. "
                        "Descompón la consulta en 2-3 sub-consultas independientes y específicas que "
                        "juntas cubran toda la información necesaria. Responde SOLO con las sub-consultas, "
                        "una por línea, sin numeración ni viñetas."
                    )
                },
                {"role": "user", "content": query}
            ],
            max_tokens=200,
            temperature=0.2,
        )
        sub_queries = [
            sq.strip() for sq in response.choices[0].message.content.strip().split('\n')
            if sq.strip() and len(sq.strip()) > 10
        ][:3]  # Máximo 3 sub-queries
        
        if sub_queries:
            print(f"   🔀 Query Decomposition: {len(sub_queries)} sub-queries generadas")
            for i, sq in enumerate(sub_queries):
                print(f"      [{i+1}] {sq[:80]}")
        return sub_queries
    except Exception as e:
        print(f"   ⚠️ Query Decomposition falló: {e}")
        return []


# ══════════════════════════════════════════════════════════════════════════════
# ADVANCED RAG: Cohere Rerank (Cross-Encoder)
# ══════════════════════════════════════════════════════════════════════════════

async def _cohere_rerank(query: str, results: List[SearchResult], top_n: int = 25) -> List[SearchResult]:
    """
    Usa Cohere Rerank V3.5 (cross-encoder) para re-ordenar los resultados.
    El cross-encoder analiza cada par (query, document) juntos, produciendo
    scores de relevancia mucho más precisos que los bi-encoders.
    
    Mejora típica: 33-40% en retrieval accuracy.
    """
    if not COHERE_RERANK_ENABLED or not results:
        return results
    
    try:
        # Preparar documentos para Cohere
        documents = []
        for r in results:
            doc_text = r.texto[:2000] if r.texto else ""  # Cohere limit
            if r.origen:
                doc_text = f"[{r.origen}] {doc_text}"
            documents.append(doc_text)
        
        # Llamar Cohere Rerank API
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                "https://api.cohere.com/v2/rerank",
                headers={
                    "Authorization": f"Bearer {COHERE_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": COHERE_RERANK_MODEL,
                    "query": query,
                    "documents": documents,
                    "top_n": min(top_n, len(documents)),
                },
            )
            
            if response.status_code != 200:
                print(f"   ⚠️ Cohere Rerank HTTP {response.status_code}: {response.text[:200]}")
                return results
            
            rerank_data = response.json()
        
        # Re-ordenar resultados según Cohere scores
        reranked = []
        for item in rerank_data.get("results", []):
            idx = item["index"]
            relevance = item["relevance_score"]
            if idx < len(results):
                r = results[idx]
                r.score = relevance  # Actualizar score con Cohere relevance
                reranked.append(r)
        
        print(f"   🎯 Cohere Rerank: {len(reranked)} resultados re-ordenados")
        if reranked:
            print(f"      Top-3 post-rerank:")
            for r in reranked[:3]:
                print(f"         {r.score:.4f} | {r.ref} | {r.origen[:50] if r.origen else 'N/A'}")
        
        return reranked
    
    except Exception as e:
        print(f"   ⚠️ Cohere Rerank falló (usando orden original): {e}")
        return results


# ═══════════════════════════════════════════════════════════════════════════
# RECUPERACIÓN DETERMINISTA POR NÚMERO DE ARTÍCULO
# Garantiza el texto vigente (post-Reforma 2024) sin semántica ni varianza
# ═══════════════════════════════════════════════════════════════════════════

_ARTICLE_PATTERN = re.compile(
    r'art[íi]culos?\s*(\d+[°oa]?)|art\.\s*(\d+[°oa]?)',
    re.IGNORECASE
)

# Colecciones donde buscar artículos constitucionales y federales
_DETERMINISTIC_COLLECTIONS = [
    "bloque_constitucional",
    "leyes_federales",
]


def _detect_article_numbers(query: str) -> List[str]:
    """Detecta menciones explícitas de artículos en la query.
    Retorna lista de números como strings: ['94', '1', '133']
    """
    matches = _ARTICLE_PATTERN.findall(query)
    nums = []
    for m in matches:
        # m es tupla (group1, group2)
        num = m[0] or m[1]
        if num:
            # Normalizar: quitar letras de ordinal (1°, 4o, 4a)
            num_clean = re.sub(r'[°oa]$', '', num, flags=re.IGNORECASE).strip()
            if num_clean not in nums:
                nums.append(num_clean)
    return nums


async def _deterministic_article_fetch(article_numbers: List[str]) -> List[SearchResult]:
    """
    Capa 1 Anti-Alucinación: Recuperación determinista de artículos por número.
    
    Para cada número detectado (ej. '94'), busca en Qdrant usando payload filter
    por ref exacto en bloque_constitucional y leyes_federales.
    Retorna resultados con score=2.0 (prioridad máxima, sobre cualquier resultado semántico).
    """
    if not article_numbers:
        return []
    
    results: List[SearchResult] = []
    
    # Construir variantes del ref para cada número de artículo
    for num in article_numbers:
        ref_variants = [
            f"Art. {num} CPEUM",
            f"Art. {num}o CPEUM",
            f"Art. {num}° CPEUM",
            f"Art. {num}a CPEUM",
            f"Artículo {num}",
            f"Art. {num}",
        ]
        
        for collection in _DETERMINISTIC_COLLECTIONS:
            try:
                points, _ = qdrant_client.scroll(
                    collection_name=collection,
                    scroll_filter=Filter(
                        must=[
                            FieldCondition(
                                key="ref",
                                match=MatchAny(any=ref_variants)
                            )
                        ]
                    ),
                    limit=3,
                    with_payload=True,
                    with_vectors=False
                )
                
                for point in points:
                    texto = point.payload.get("texto", "")
                    if not texto or len(texto) < 50:
                        continue
                    
                    results.append(SearchResult(
                        id=str(point.id),
                        score=2.0,  # Prioridad máxima — sobre cualquier resultado semántico
                        texto=texto,
                        ref=point.payload.get("ref", f"Art. {num}"),
                        origen=point.payload.get("origen", ""),
                        jurisdiccion=point.payload.get("estado", ""),
                        entidad=point.payload.get("entidad", ""),
                        silo=collection,
                    ))
                    print(f"   🎯 ARTICLE LOCK: Art. {num} → {collection} → {point.payload.get('ref')} (score=2.0)")
            except Exception as e:
                print(f"   ⚠️ Deterministic fetch error for Art. {num} in {collection}: {e}")
    
    return results


async def hybrid_search_all_silos(
    query: str,
    estado: Optional[str],
    top_k: int,
    alpha: float = 0.7,
    enable_reasoning: bool = False,
    forced_materia: Optional[str] = None,
    fuero: Optional[str] = None,  # NUEVO: Filtro por fuero (constitucional/federal/estatal)
) -> List[SearchResult]:
    """
    Ejecuta búsqueda híbrida paralela en silos relevantes según fuero.
    
    Fuero routing:
        constitucional → bloque_constitucional + jurisprudencia_nacional
        federal → leyes_federales + jurisprudencia_nacional
        estatal → leyes_[estado] + jurisprudencia_nacional
        None → todos los silos (comportamiento original)
    
    jurisprudencia_nacional SIEMPRE se incluye.
    """
    # ═══════════════════════════════════════════════════════════════════════════
    # PASO -1: RECUPERACIÓN DETERMINISTA POR NÚMERO DE ARTÍCULO (Anti-alucinación)
    # Si la query menciona Art. X, recuperar el texto exacto antes de cualquier semántica
    # Garantiza que el LLM reciba el texto vigente (post-Reforma 2024) con prioridad máxima
    # ═══════════════════════════════════════════════════════════════════════════
    detected_article_nums = _detect_article_numbers(query)
    deterministic_results: List[SearchResult] = []
    if detected_article_nums:
        print(f"   🔍 Números de artículo detectados: {detected_article_nums}")
        deterministic_results = await _deterministic_article_fetch(detected_article_nums)
        if deterministic_results:
            print(f"   ✅ {len(deterministic_results)} artículo(s) recuperados determinísticamente")

    # ═══════════════════════════════════════════════════════════════════════════
    # PASO 0: Query Expansion - Acrónimos legales (local, <1ms)
    # ═══════════════════════════════════════════════════════════════════════════
    expanded_query = expand_legal_query(query)

    # ═══════════════════════════════════════════════════════════════════════════
    # PASO 0-BIS: AGENTE ESTRATEGA (Legal Strategy Agent)
    # Diagnóstico jurídico del caso → Plan de búsqueda con pesos de silos
    # Reemplaza la expansión ciega por sinónimos con razonamiento legal real
    # ═══════════════════════════════════════════════════════════════════════════
    legal_plan = await _legal_strategy_agent(query, fuero_manual=fuero)
    # Usar jurisprudencia_keywords del plan para enriquecer la expanded_query
    if legal_plan["jurisprudencia_keywords"]:
        jk = " ".join(legal_plan["jurisprudencia_keywords"][:2])
        expanded_query = f"{expanded_query} {jk}"
    # Si el fuero no fue seleccionado manualmente, usar el detectado por el agente
    if not fuero and legal_plan["fuero_detectado"] not in ("mixto", None):
        fuero = legal_plan["fuero_detectado"]
        print(f"   ⚖️ FUERO AUTO-DETECTADO por Agente Estratega: {fuero}")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # MATERIA-AWARE RETRIEVAL — Capa 1+2: Detección + Should Filter
    # ═══════════════════════════════════════════════════════════════════════════
    detected_materias = _detect_materia(query, forced_materia=forced_materia)
    if detected_materias:
        print(f"   🎯 MATERIA DETECTADA: {detected_materias} (forced={forced_materia is not None})")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # HyDE: Hypothetical Document Embeddings
    # Genera un doc jurídico hipotético para mejorar el dense embedding en queries coloquiales
    # ═══════════════════════════════════════════════════════════════════════════
    hyde_doc = await _generate_hyde_document(query)
    
    if hyde_doc:
        dense_text = hyde_doc
        print(f"   🔮 Dense embedding usando HyDE document")
    else:
        dense_text = query
    
    # Generar embeddings: dense=HyDE o query, sparse=expanded (BM25 keywords)
    dense_task = get_dense_embedding(dense_text)
    sparse_vector = get_sparse_embedding(expanded_query)
    dense_vector = await dense_task
    
    # ═══════════════════════════════════════════════════════════════════════════
    # QUERY DECOMPOSITION: Sub-queries para queries complejas multi-hop
    # ═══════════════════════════════════════════════════════════════════════════
    sub_queries = await _decompose_query(query)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # FILTRO POR FUERO: Determinar silos a buscar
    # jurisprudencia_nacional SIEMPRE se incluye
    # ═══════════════════════════════════════════════════════════════════════════
    fuero_normalized = fuero.lower().strip() if fuero else None
    
    if fuero_normalized == "constitucional":
        silos_to_search = ["bloque_constitucional", "jurisprudencia_nacional"]
        print(f"   ⚖️ FUERO: Constitucional → bloque_constitucional + jurisprudencia_nacional")
    elif fuero_normalized == "federal":
        silos_to_search = ["leyes_federales", "jurisprudencia_nacional"]
        print(f"   ⚖️ FUERO: Federal → leyes_federales + jurisprudencia_nacional")
    elif fuero_normalized == "estatal":
        silos_to_search = ["jurisprudencia_nacional"]  # Siempre
        if estado:
            normalized_estado = normalize_estado(estado)
            if normalized_estado and normalized_estado in ESTADO_SILO:
                silos_to_search.append(ESTADO_SILO[normalized_estado])
                print(f"   ⚖️ FUERO: Estatal → {ESTADO_SILO[normalized_estado]} + jurisprudencia_nacional")
            else:
                silos_to_search.append(LEGACY_ESTATAL_SILO)
                print(f"   ⚖️ FUERO: Estatal → {LEGACY_ESTATAL_SILO} (legacy) + jurisprudencia_nacional")
        else:
            # Fuero estatal sin estado seleccionado → buscar TODOS los estatales
            silos_to_search.extend(ESTADO_SILO.values())
            silos_to_search.append(LEGACY_ESTATAL_SILO)
            print(f"   ⚖️ FUERO: Estatal (sin estado) → todos los silos estatales + jurisprudencia_nacional")
    else:
        # Sin fuero = comportamiento original: TODOS los silos
        silos_to_search = list(FIXED_SILOS.values())
        if estado:
            normalized_estado = normalize_estado(estado)
            if normalized_estado and normalized_estado in ESTADO_SILO:
                silos_to_search.append(ESTADO_SILO[normalized_estado])
                print(f"   📍 Estado '{normalized_estado}' → silo dedicado: {ESTADO_SILO[normalized_estado]}")
            else:
                silos_to_search.append(LEGACY_ESTATAL_SILO)
                print(f"   📍 Estado '{estado}' → fallback a silo legacy: {LEGACY_ESTATAL_SILO}")
        else:
            silos_to_search.extend(ESTADO_SILO.values())
            silos_to_search.append(LEGACY_ESTATAL_SILO)
            print(f"   📍 Sin fuero/estado → buscando en {len(ESTADO_SILO) + 1 + len(FIXED_SILOS)} silos")
    
    tasks = []
    for silo_name in silos_to_search:
        state_filter = get_filter_for_silo(silo_name, estado)
        
        tasks.append(
            hybrid_search_single_silo(
                collection=silo_name,
                query=query,
                dense_vector=dense_vector,
                sparse_vector=sparse_vector,
                filter_=state_filter,
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
            elif r.silo == "jurisprudencia_nacional":
                jurisprudencia.append(r)
            elif r.silo == "bloque_constitucional":
                constitucional.append(r)
            elif r.silo.startswith("leyes_") or r.silo == LEGACY_ESTATAL_SILO:
                # Todos los silos estatales (dedicados + legacy) van a «estatales»
                estatales.append(r)
    
    # Ordenar cada grupo por score
    federales.sort(key=lambda x: x.score, reverse=True)
    estatales.sort(key=lambda x: x.score, reverse=True)
    jurisprudencia.sort(key=lambda x: x.score, reverse=True)
    constitucional.sort(key=lambda x: x.score, reverse=True)

    # ═══════════════════════════════════════════════════════════════════════════
    # PASO -1 (CONT.): INYECTAR RESULTADOS DETERMINISTAS CON PRIORIDAD MÁXIMA
    # Los artículos recuperados por número exacto (score=2.0) van primero
    # ═══════════════════════════════════════════════════════════════════════════
    if deterministic_results:
        existing_ids = {r.id for r in constitucional + federales}
        for det_r in deterministic_results:
            if det_r.id not in existing_ids:
                if det_r.silo == "leyes_federales":
                    federales.insert(0, det_r)
                else:
                    constitucional.insert(0, det_r)
                existing_ids.add(det_r.id)
        print(f"   \U0001f3af DETERMINISTIC INJECT: {len(deterministic_results)} artículo(s) con score=2.0 al frente del contexto")

    
    # ═══════════════════════════════════════════════════════════════════════════
    # CPEUM ARTICLE INJECTION: Si el query pide un artículo específico, inyectar
    # Resuelve la limitación de semantic search con números de artículos
    # ═══════════════════════════════════════════════════════════════════════════
    import re as _re
    cpeum_article_match = _re.search(
        r'art[ií]culo\s+(\d+)\s*(?:o|°|º)?\s*(?:de\s+la\s+)?(?:constituci[oó]n|cpeum|constitucional)',
        query.lower()
    )
    if not cpeum_article_match:
        # Also try reverse: "constitucion articulo N"
        cpeum_article_match = _re.search(
            r'(?:constituci[oó]n|cpeum|constitucional)\s.*?art[ií]culo\s+(\d+)',
            query.lower()
        )
    
    if cpeum_article_match:
        art_num = int(cpeum_article_match.group(1))
        ref_variants = [
            f"Art. {art_num}o CPEUM",
            f"Art. {art_num} CPEUM",
            f"Art. {art_num}o CPEUM (parte 1)",
            f"Art. {art_num} CPEUM (parte 1)",
        ]
        print(f"   📜 CPEUM INJECTION: Detectado artículo {art_num}, buscando refs: {ref_variants}")
        
        # Search for ALL chunks with matching ref from bloque_constitucional
        existing_refs = {r.ref for r in constitucional}
        injected_count = 0
        
        try:
            cpeum_pts, _ = await qdrant_client.scroll(
                collection_name="bloque_constitucional",
                scroll_filter=Filter(must=[
                    FieldCondition(key="origen", match=MatchValue(
                        value="Constitución Política de los Estados Unidos Mexicanos"
                    )),
                ]),
                limit=400,
                with_payload=True,
                with_vectors=False,
            )
            
            # Find matching articles (sustantivo=True preferred, then by ref match)
            for pt in cpeum_pts:
                ref = pt.payload.get("ref", "")
                is_sustantivo = pt.payload.get("sustantivo", False)
                
                # Match by ref prefix (handles parte 1, parte 2, etc.)
                matches_ref = any(ref.startswith(rv.replace(" (parte 1)", "")) for rv in ref_variants)
                
                if matches_ref and is_sustantivo and ref not in existing_refs:
                    injected_result = SearchResult(
                        id=str(pt.id),
                        texto=pt.payload.get("texto", ""),
                        ref=ref,
                        origen=pt.payload.get("origen", ""),
                        score=0.95,  # High score to ensure top ranking
                        silo="bloque_constitucional",
                    )
                    constitucional.insert(0, injected_result)
                    existing_refs.add(ref)
                    injected_count += 1
            
            if injected_count > 0:
                print(f"   ✅ CPEUM INJECTION: {injected_count} chunks del Art. {art_num} inyectados con score=0.95")
            else:
                print(f"   ⚠️ CPEUM INJECTION: No se encontraron chunks sustantivos para Art. {art_num}")
        except Exception as e:
            print(f"   ❌ CPEUM INJECTION error: {e}")
    
    # === DIAGNOSTIC LOGGING: TOP-3 per silo para diagnóstico de relevancia ===
    print(f"\n   🔎 RAW RETRIEVAL SCORES (pre-merge):")
    for label, group in [("ESTATALES", estatales), ("FEDERALES", federales), ("JURIS", jurisprudencia), ("CONST", constitucional)]:
        print(f"      {label} ({len(group)} results):")
        for r in group[:3]:
            origen_short = r.origen[:55] if r.origen else 'N/A'
            print(f"         {r.score:.4f} | ref={r.ref} | {origen_short}")
    
    # Fusión balanceada DINÁMICA según tipo de query
    # Para queries de DDHH, priorizar agresivamente el bloque constitucional
    # ── Fusión Balanceada DINÁMICA — Pesos desde el Agente Estratega ──────────
    # El Agente Estratega diagnosticó el caso y produjo pesos específicos.
    # Esto reemplaza los valores hardcodeados por una asignación inteligente.
    # Override: DDHH y modo estatal siguen teniendo prioridad fija (lógica crítica).
    _agent_pesos = legal_plan.get("pesos_silos", {})
    _agent_const = _agent_pesos.get("constitucional", 0.25)
    _agent_fed   = _agent_pesos.get("federal", 0.25)
    _agent_est   = _agent_pesos.get("estatal", 0.25)
    _agent_juris = _agent_pesos.get("jurisprudencia", 0.25)

    if is_ddhh_query(query) or legal_plan.get("requiere_ddhh"):
        # Modo DDHH: Prioridad máxima a bloque constitucional (override del agente)
        min_constitucional = min(12, len(constitucional))
        min_jurisprudencia = min(6, len(jurisprudencia))
        min_federales = min(6, len(federales))
        min_estatales = min(3, len(estatales))
        print(f"   🏛️ Modo DDHH: const={min_constitucional} juris={min_jurisprudencia} fed={min_federales} est={min_estatales}")
    elif estado:
        # Modo con ESTADO seleccionado: LEYES ESTATALES SON LA PRIORIDAD
        # Mantener prioridad fija para estado, ya que es selección explícita del usuario
        min_estatales = min(15, len(estatales))
        min_jurisprudencia = min(8, len(jurisprudencia))
        min_federales = min(5, len(federales))
        min_constitucional = min(4, len(constitucional))
        print(f"   📍 Modo estatal PRIORIZADO: est={min_estatales} juris={min_jurisprudencia} fed={min_federales} const={min_constitucional} para {estado}")
    else:
        # Modo con Agente Estratega: pesos dinámicos basados en el diagnóstico del caso
        # Escalar los pesos del agente a slots enteros (top_k base = 25)
        _base = top_k
        min_constitucional = min(int(_base * _agent_const * 1.5), len(constitucional))
        min_jurisprudencia = min(int(_base * _agent_juris * 1.5), len(jurisprudencia))
        min_federales      = min(int(_base * _agent_fed   * 1.5), len(federales))
        min_estatales      = min(int(_base * _agent_est   * 1.5), len(estatales))
        # Garantizar al menos 3 slots por silo para evitar silos vacíos
        min_constitucional = max(min_constitucional, min(3, len(constitucional)))
        min_jurisprudencia = max(min_jurisprudencia, min(3, len(jurisprudencia)))
        min_federales      = max(min_federales,      min(3, len(federales)))
        min_estatales      = max(min_estatales,      min(3, len(estatales)))
        print(f"   🧠 Agente Estratega pesos → const={min_constitucional} fed={min_federales} est={min_estatales} juris={min_jurisprudencia} (materia={legal_plan.get('materia_principal')}|via={legal_plan.get('via_procesal','?')[:40]})") 
    
    merged = []
    
    if estado:
        # CUANDO HAY ESTADO: leyes estatales VAN PRIMERO en el contexto
        # El LLM procesa los primeros documentos con mayor atención
        merged.extend(estatales[:min_estatales])
        merged.extend(jurisprudencia[:min_jurisprudencia])
        merged.extend(federales[:min_federales])
        merged.extend(constitucional[:min_constitucional])
    else:
        # Sin estado: orden estándar por jerarquía normativa
        merged.extend(constitucional[:min_constitucional])
        merged.extend(federales[:min_federales])
        merged.extend(estatales[:min_estatales])
        merged.extend(jurisprudencia[:min_jurisprudencia])
    
    # === PRODUCTION LOGGING: qué documentos van al contexto ===
    print(f"\n   📋 MERGED RESULTS ({len(merged)} total):")
    silo_counts = {}
    for r in merged:
        silo_counts[r.silo] = silo_counts.get(r.silo, 0) + 1
        if r.silo.startswith("leyes_") and r.silo != "leyes_federales":
            print(f"      ⭐ [{r.silo}] ref={r.ref} origen={r.origen[:60] if r.origen else 'N/A'} score={r.score:.4f}")
    for silo, count in silo_counts.items():
        print(f"      📊 {silo}: {count} documentos")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # MULTI-QUERY: Búsqueda adicional para artículos específicos
    # ═══════════════════════════════════════════════════════════════════════════
    # ═══════════════════════════════════════════════════════════════════════════
    # MULTI-QUERY: Búsqueda adicional para artículos específicos
    # ═══════════════════════════════════════════════════════════════════════════
    article_numbers = detect_article_numbers(query)
    
    if article_numbers:
        # Definir target silos y estrategia de query según si hay estado o no
        multi_query_targets = []
        
        if estado:
            print(f"   🔍 Multi-query STATE: buscando artículo(s) {article_numbers} en leyes estatales")
            normalized_est = normalize_estado(estado)
            silo = ESTADO_SILO.get(normalized_est, LEGACY_ESTATAL_SILO) if normalized_est else LEGACY_ESTATAL_SILO
            # En estado, el "artículo X" puro suele funcionar mejor
            multi_query_targets.append({"silo": silo, "strategy": "pure", "filter": get_filter_for_silo(silo, estado)})
        else:
            print(f"   🔍 Multi-query FEDERAL: buscando artículo(s) {article_numbers} en leyes federales")
            # En federal, necesitamos contexto para desambiguar entre cientos de leyes
            multi_query_targets.append({"silo": "leyes_federales", "strategy": "context", "filter": None})
            
        for target in multi_query_targets:
            silo_col = target["silo"]
            strategy = target["strategy"]
            silo_filter = target["filter"]
            
            for art_num in article_numbers[:2]:  # Máximo 2 artículos por query
                if strategy == "pure":
                    article_query = f"artículo {art_num}"
                else:
                    # Context strategy: "artículo 41" + expanded query (Ley Federal de Procedimiento...)
                    article_query = f"artículo {art_num} {expanded_query}"

                try:
                    art_dense = await get_dense_embedding(article_query)
                    art_sparse = get_sparse_embedding(article_query)
                    extra_results = await hybrid_search_single_silo(
                        collection=silo_col,
                        query=article_query,
                        dense_vector=art_dense,
                        sparse_vector=art_sparse,
                        filter_=silo_filter,
                        top_k=5,
                        alpha=0.7,
                    )
                    # Agregar solo los que no estén ya
                    existing_ids = {r.id for r in merged}
                    new_results = [r for r in extra_results if r.id not in existing_ids]
                    merged.extend(new_results)
                    print(f"   🔍 Multi-query artículo {art_num} en {silo_col}: +{len(new_results)} resultados nuevos")
                except Exception as e:
                    print(f"   ⚠️ Multi-query falló para artículo {art_num} en {silo_col}: {e}")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # ARTICLE-AWARE RERANKING
    # ═══════════════════════════════════════════════════════════════════════════
    if article_numbers:
        merged = rerank_by_article_match(merged, article_numbers)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # JURISPRUDENCIA BOOST V2: Multi-query agresivo para maximizar recall
    # ═══════════════════════════════════════════════════════════════════════════
    juris_in_merged = [r for r in merged if r.silo == "jurisprudencia_nacional"]
    if len(juris_in_merged) < 5:
        print(f"   ⚖️ JURISPRUDENCIA BOOST V2: Solo {len(juris_in_merged)} tesis, ejecutando multi-query...")
        try:
            # Extraer conceptos jurídicos clave para formular queries de jurisprudencia
            juris_concepts = await _extract_juris_concepts(query)
            
            juris_queries = [
                # Query 1: Original con prefijo de jurisprudencia
                f"tesis jurisprudencia SCJN tribunales colegiados: {query}",
                # Query 2: Conceptos jurídicos extraídos por LLM
                f"tesis aislada jurisprudencia criterio: {juris_concepts}",
                # Query 3: Expanded query también con prefijo judicial  
                f"primera sala segunda sala pleno SCJN: {expanded_query}",
            ]
            
            existing_ids = {r.id for r in merged}
            all_new_juris = []
            
            # Ejecutar las 3 queries en paralelo
            boost_tasks = []
            for jq in juris_queries:
                boost_tasks.append(
                    _jurisprudencia_boost_search(jq, existing_ids)
                )
            
            boost_results = await asyncio.gather(*boost_tasks)
            
            for results in boost_results:
                for r in results:
                    if r.id not in existing_ids:
                        all_new_juris.append(r)
                        existing_ids.add(r.id)
            
            # Ordenar por score y agregar todas las tesis únicas
            all_new_juris.sort(key=lambda x: x.score, reverse=True)
            merged.extend(all_new_juris)
            print(f"   ⚖️ JURISPRUDENCIA BOOST V2: +{len(all_new_juris)} tesis adicionales de {len(juris_queries)} queries")
        except Exception as e:
            print(f"   ⚠️ Jurisprudencia boost V2 falló: {e}")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # CROSS-SILO ENRICHMENT: Segunda pasada para encadenar fuentes
    # ═══════════════════════════════════════════════════════════════════════════
    try:
        enrichment_results = await _cross_silo_enrichment(merged, query)
        if enrichment_results:
            existing_ids = {r.id for r in merged}
            new_enriched = [r for r in enrichment_results if r.id not in existing_ids]
            merged.extend(new_enriched)
            print(f"   🔗 CROSS-SILO ENRICHMENT: +{len(new_enriched)} documentos de segunda pasada")
    except Exception as e:
        print(f"   ⚠️ Cross-silo enrichment falló (continuando): {e}")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # NEIGHBOR CHUNK RETRIEVAL: Artículos adyacentes para contexto completo
    # ═══════════════════════════════════════════════════════════════════════════
    try:
        neighbor_results = await _fetch_neighbor_chunks(merged)
        if neighbor_results:
            existing_ids = {r.id for r in merged}
            new_neighbors = [r for r in neighbor_results if r.id not in existing_ids]
            merged.extend(new_neighbors)
            print(f"   📄 NEIGHBOR CHUNKS: +{len(new_neighbors)} artículos adyacentes")
    except Exception as e:
        print(f"   ⚠️ Neighbor chunk retrieval falló (continuando): {e}")
    
    # Llenar el resto con los mejores scores combinados
    already_added = {r.id for r in merged}
    remaining = [r for results in all_results for r in results if r.id not in already_added]
    remaining.sort(key=lambda x: x.score, reverse=True)
    
    slots_remaining = top_k - len(merged)
    if slots_remaining > 0:
        merged.extend(remaining[:slots_remaining])
    
    # ═══════════════════════════════════════════════════════════════════════════
    # QUERY DECOMPOSITION: Búsqueda adicional con sub-queries descompuestas
    # ═══════════════════════════════════════════════════════════════════════════
    if sub_queries:
        existing_ids = {r.id for r in merged}
        decomp_new = 0
        for sq in sub_queries:
            try:
                sq_dense = await get_dense_embedding(sq)
                sq_sparse = get_sparse_embedding(sq)
                for silo_name in silos_to_search[:4]:  # Top 4 silos only for speed
                    sq_filter = get_filter_for_silo(silo_name, estado)
                    sq_results = await hybrid_search_single_silo(
                        collection=silo_name,
                        query=sq,
                        dense_vector=sq_dense,
                        sparse_vector=sq_sparse,
                        filter_=sq_filter,
                        top_k=5,
                        alpha=0.7,
                    )
                    for r in sq_results:
                        if r.id not in existing_ids:
                            merged.append(r)
                            existing_ids.add(r.id)
                            decomp_new += 1
            except Exception as e:
                print(f"   ⚠️ Sub-query búsqueda falló: {e}")
        if decomp_new > 0:
            print(f"   🔀 Query Decomposition: +{decomp_new} resultados nuevos de sub-queries")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # MATERIA-AWARE RETRIEVAL — Capa 3: Post-Retrieval Threshold
    # ═══════════════════════════════════════════════════════════════════════════
    if detected_materias:
        merged = _apply_materia_threshold(merged, detected_materias)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # COHERE RERANK: Cross-encoder final reranking (ÚLTIMA CAPA)
    # El cross-encoder analiza (query, document) juntos → scores mucho más precisos
    # ═══════════════════════════════════════════════════════════════════════════
    merged.sort(key=lambda x: x.score, reverse=True)
    merged = merged[:top_k + 10]  # Pre-filter before expensive rerank
    
    if COHERE_RERANK_ENABLED:
        merged = await _cohere_rerank(query, merged, top_n=top_k)
    
    # Ordenar el resultado final por score para presentación
    merged.sort(key=lambda x: x.score, reverse=True)
    return merged[:top_k]


# ══════════════════════════════════════════════════════════════════════════════
# DA VINCI: BÚSQUEDA MULTI-ESTADO PARA COMPARACIONES
# ══════════════════════════════════════════════════════════════════════════════

async def hybrid_search_multi_state(
    query: str,
    estados: List[str],
    top_k_per_state: int = 5,
) -> dict:
    """
    Ejecuta búsqueda paralela en múltiples estados.
    Retorna resultados agrupados por estado para comparación.
    
    Args:
        query: Query del usuario (sin nombres de estados)
        estados: Lista de estados canónicos (ej: ["JALISCO", "QUERETARO"])
        top_k_per_state: Resultados por estado
    
    Returns:
        {
            "results_by_state": {"JALISCO": [SearchResult, ...], ...},
            "all_results": [SearchResult, ...],
            "context_xml": str
        }
    """
    print(f"   🎨 MULTI-STATE: Buscando en {len(estados)} estados")
    print(f"      Estados: {estados}")
    
    # Generar embeddings UNA SOLA VEZ (reutilizar para todos los estados)
    expanded_query = await expand_legal_query_llm(query)
    dense_vector = await get_dense_embedding(expanded_query)
    sparse_vector = get_sparse_embedding(expanded_query)
    
    # Búsqueda paralela: un task por estado
    async def search_one_state(estado_name: str) -> tuple:
        """Busca en la colección del estado (dedicada o legacy)"""
        # Determinar colección: silo dedicado o legacy con filtro
        if estado_name in ESTADO_SILO:
            collection = ESTADO_SILO[estado_name]
            state_filter = None  # Sin filtro en colección dedicada
        else:
            collection = LEGACY_ESTATAL_SILO
            state_filter = Filter(
                must=[
                    FieldCondition(key="entidad", match=MatchValue(value=estado_name))
                ]
            )
        
        results = await hybrid_search_single_silo(
            collection=collection,
            query=query,
            dense_vector=dense_vector,
            sparse_vector=sparse_vector,
            filter_=state_filter,
            top_k=top_k_per_state,
            alpha=0.7,
        )
        return (estado_name, results)
    
    # Ejecutar todas las búsquedas en paralelo
    tasks = [search_one_state(e) for e in estados]
    state_results = await asyncio.gather(*tasks)
    
    # Agrupar por estado
    results_by_state = {}
    all_results = []
    for estado_name, results in state_results:
        results_by_state[estado_name] = results
        all_results.extend(results)
        print(f"      {estado_name}: {len(results)} resultados")
    
    # También buscar en bloque constitucional + federales (aplican a todos)
    fed_const_tasks = []
    for silo_name in ["bloque_constitucional", "leyes_federales"]:
        if silo_name in FIXED_SILOS.values():
            fed_const_tasks.append(
                hybrid_search_single_silo(
                    collection=silo_name,
                    query=query,
                    dense_vector=dense_vector,
                    sparse_vector=sparse_vector,
                    filter_=None,
                    top_k=5,
                    alpha=0.7,
                )
            )
    
    if fed_const_tasks:
        extra_results = await asyncio.gather(*fed_const_tasks)
        for results in extra_results:
            all_results.extend(results)
    
    # Formatear contexto XML agrupado por estado
    context_parts = []
    for estado_name in estados:
        state_docs = results_by_state.get(estado_name, [])
        if state_docs:
            context_parts.append(f"\n<!-- ESTADO: {estado_name} -->")
            for r in state_docs:
                context_parts.append(
                    f'<doc id="{r.id}" estado="{estado_name}" '
                    f'ref="{r.ref or ""}" origen="{r.origen or ""}">\n{r.texto[:1500]}\n</doc>'
                )
    
    # Agregar contexto federal/constitucional  
    for results in (extra_results if fed_const_tasks else []):
        for r in results:
            context_parts.append(
                f'<doc id="{r.id}" silo="{r.silo}" '
                f'ref="{r.ref or ""}">\n{r.texto[:1500]}\n</doc>'
            )
    
    context_xml = "\n".join(context_parts)
    
    print(f"   🎨 DA VINCI MULTI-STATE: Total {len(all_results)} resultados combinados")
    
    return {
        "results_by_state": results_by_state,
        "all_results": all_results,
        "context_xml": context_xml,
        "estados": estados,

    }


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
        all_known = set(FIXED_SILOS.values()) | set(ESTADO_SILO.values()) | {LEGACY_ESTATAL_SILO}
        silos_activos = [c.name for c in collections.collections if c.name in all_known]
    except Exception as e:
        qdrant_status = f"error: {e}"
        silos_activos = []
    
    return {
        "status": "healthy" if qdrant_status == "connected" else "degraded",
        "version": "2026.02.20-v4 (Advanced RAG)",
        "model": CHAT_MODEL,
        "qdrant": qdrant_status,
        "silos_activos": silos_activos,
        "sparse_encoder": "Qdrant/bm25",
        "dense_model": EMBEDDING_MODEL,
        "rag_features": {
            "cohere_rerank": COHERE_RERANK_ENABLED,
            "cohere_model": COHERE_RERANK_MODEL if COHERE_RERANK_ENABLED else None,
            "hyde": HYDE_ENABLED,
            "hyde_model": HYDE_MODEL if HYDE_ENABLED else None,
            "query_decomposition": QUERY_DECOMPOSITION_ENABLED,
        },
    }


@app.get("/api/wake")
async def wake_endpoint():
    """Ultra-lightweight endpoint to wake up the backend from cold start."""
    return {"status": "awake"}


@app.get("/quota/status/{user_id}")
async def quota_status_endpoint(user_id: str):
    """
    Returns current quota status for a user (read-only, does not consume).
    Used by frontend to display usage counters.
    """
    if not supabase_admin:
        raise HTTPException(status_code=503, detail="Supabase not configured")

    try:
        result = supabase_admin.rpc(
            'get_quota_status', {'p_user_id': user_id}
        ).execute()

        if result.data:
            return result.data
        return {"error": "user_not_found"}
    except Exception as e:
        print(f"⚠️ Quota status check failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch quota status")

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
    # Campos de localización para jurisprudencia
    registro: Optional[str] = None
    instancia: Optional[str] = None
    materia: Optional[str] = None
    tesis_num: Optional[str] = None
    tipo_criterio: Optional[str] = None
    url_pdf: Optional[str] = None
    chunk_index: int = 0  # 0 = inicio del artículo, >0 = continuación
    jerarquia_txt: Optional[str] = None  # e.g. "Título Quinto > Capítulo II > Sección Primera"


@app.get("/document/{doc_id}", response_model=DocumentResponse)
async def get_document(doc_id: str):
    """
    Obtiene el contenido completo de un documento por su ID de Qdrant.
    Busca en todos los silos hasta encontrarlo.
    """
    try:
        # Buscar en cada silo
        all_silos = list(FIXED_SILOS.values()) + list(ESTADO_SILO.values()) + [LEGACY_ESTATAL_SILO]
        for silo_name in all_silos:
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
                    
                    # Materia puede ser string o lista
                    materia_raw = payload.get("materia")
                    if isinstance(materia_raw, list):
                        materia_str = ", ".join(str(m).upper() for m in materia_raw)
                    else:
                        materia_str = str(materia_raw).upper() if materia_raw else None
                    
                    texto_val = payload.get("texto", payload.get("text", "Contenido no disponible"))
                    # Prioridad de origen: campo Qdrant → extraído del texto → None
                    _origen_raw = payload.get("origen", payload.get("fuente", None))
                    _origen = humanize_origen(_origen_raw) or extract_ley_from_texto(texto_val)

                    return DocumentResponse(
                        id=str(point.id),
                        texto=texto_val,
                        ref=payload.get("ref", payload.get("referencia", None)),
                        origen=_origen,
                        jurisdiccion=payload.get("jurisdiccion", None),
                        entidad=payload.get("entidad", payload.get("estado", None)),
                        silo=silo_name,
                        found=True,
                        # Jurisprudencia: normalizar ambos esquemas de nombres
                        registro=str(payload.get("registro")) if payload.get("registro") else None,
                        instancia=payload.get("instancia", None),
                        materia=materia_str,
                        tesis_num=payload.get("tesis", payload.get("tesis_num", None)),
                        tipo_criterio=payload.get("tipo", payload.get("tipo_criterio", None)),
                        url_pdf=payload.get("url_pdf", None),
                        chunk_index=payload.get("chunk_index", 0),
                        jerarquia_txt=payload.get("jerarquia_txt", None),
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
# ENDPOINT: DOCUMENTO COMPLETO (reconstruido desde chunks)
# ══════════════════════════════════════════════════════════════════════════════

class FullDocumentResponse(BaseModel):
    """Documento completo reconstruido desde chunks de Qdrant."""
    origen: str
    titulo: str
    tipo: Optional[str] = None  # constitucion, convencion, cuadernillo, sentencia_cidh, protocolo
    texto_completo: str  # Texto completo reconstruido
    total_chunks: int
    highlight_chunk_index: Optional[int] = None  # Chunk que el usuario citó
    source_doc_url: Optional[str] = None  # URL del PDF original (CIDH cases)
    external_url: Optional[str] = None  # Link externo (protocolos SCJN)
    metadata: dict = {}  # Metadata adicional (caso, vs, cuadernillo_num, etc.)


# Mapeo de protocolos SCJN → URL externa
PROTOCOLO_SCJN_KEYWORDS = [
    "protocolo para juzgar",
    "protocolo-sobre-",
    "protocolo_",
    "protocolo osiegcs",
]


@app.get("/document-full", response_model=FullDocumentResponse)
async def get_full_document(
    origen: str,
    highlight_chunk_id: Optional[str] = None,
):
    """
    Reconstruye el documento completo buscando todos los chunks con el mismo
    'origen' en bloque_constitucional, ordenados por chunk_index.
    
    Para protocolos SCJN: devuelve external_url en lugar de texto.
    Para casos CIDH PDF: devuelve source_doc_url si existe.
    """
    print(f"   📖 /document-full called | origen='{origen}' | highlight={highlight_chunk_id}")
    
    # ── Detectar si es un Protocolo SCJN → link externo ──
    origen_lower = origen.lower()
    is_protocolo = any(kw in origen_lower for kw in PROTOCOLO_SCJN_KEYWORDS)
    if is_protocolo:
        return FullDocumentResponse(
            origen=origen,
            titulo=origen,
            tipo="protocolo",
            texto_completo="",
            total_chunks=0,
            external_url="https://www.scjn.gob.mx/derechos-humanos/protocolos-de-actuacion",
        )
    
    try:
        # ── Buscar TODOS los chunks con este origen ──
        from qdrant_client.models import FieldCondition, MatchValue, ScrollRequest
        
        # Try multiple variations of origen (data has inconsistent trailing whitespace)
        origen_variants = list(dict.fromkeys([
            origen,
            origen.strip(),
            origen.strip() + " ",
            origen.rstrip(":").strip() + ": ",
            origen.rstrip(": ").strip() + ":",
        ]))
        
        all_points = []
        matched_origen = origen
        
        for variant in origen_variants:
            offset = None
            variant_points = []
            
            while True:
                result = await qdrant_client.scroll(
                    collection_name="bloque_constitucional",
                    scroll_filter=Filter(
                        must=[FieldCondition(key="origen", match=MatchValue(value=variant))]
                    ),
                    limit=100,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,
                )
                
                points, next_offset = result
                variant_points.extend(points)
                
                if next_offset is None or len(points) < 100:
                    break
                offset = next_offset
            
            if variant_points:
                all_points = variant_points
                matched_origen = variant
                print(f"   📖 Matched {len(all_points)} chunks with variant: '{variant}'")
                break
        
        print(f"   📖 Total: {len(all_points)} chunks for origen='{origen}' (matched='{matched_origen}')")
        
        if not all_points:
            raise HTTPException(
                status_code=404,
                detail=f"No se encontraron chunks para origen: {origen}"
            )
        
        # ── Ordenar por chunk_index ──
        all_points.sort(key=lambda p: p.payload.get("chunk_index", 0))
        
        # ── Reconstruir texto completo ──
        textos = []
        highlight_chunk_index = None
        first_payload = all_points[0].payload
        
        for i, point in enumerate(all_points):
            payload = point.payload or {}
            texto = payload.get("texto", payload.get("texto_visible", ""))
            textos.append(texto)
            
            # Encontrar el chunk que el usuario citó
            if highlight_chunk_id and str(point.id) == highlight_chunk_id:
                highlight_chunk_index = i
        
        texto_completo = "\n\n".join(textos)
        
        # ── Extraer metadata del primer chunk ──
        metadata = {}
        for key in ["caso", "vs", "serie_c", "cuadernillo_num", "cuadernillo_tema",
                     "instrumento", "jurisdiccion", "materia"]:
            val = first_payload.get(key)
            if val:
                metadata[key] = val
        
        # ── Generar título legible ──
        tipo = first_payload.get("tipo", first_payload.get("source_type", "unknown"))
        titulo = origen
        if tipo == "cuadernillo" and first_payload.get("cuadernillo_tema"):
            titulo = f"Cuadernillo CIDH No. {first_payload.get('cuadernillo_num', '?')}: {first_payload['cuadernillo_tema']}"
        elif tipo == "sentencia_cidh" and first_payload.get("caso"):
            titulo = f"Caso {first_payload['caso']}"
            if first_payload.get("vs"):
                titulo += f" Vs. {first_payload['vs']}"
        
        return FullDocumentResponse(
            origen=origen,
            titulo=titulo,
            tipo=tipo,
            texto_completo=texto_completo,
            total_chunks=len(all_points),
            highlight_chunk_index=highlight_chunk_index,
            source_doc_url=first_payload.get("source_doc_url") or first_payload.get("url_pdf"),
            metadata=metadata,
        )
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"   ❌ Error reconstruyendo documento '{origen}': {e}")
        raise HTTPException(status_code=500, detail=f"Error al reconstruir documento: {str(e)}")


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
# AUTO-DETECCIÓN DE COMPLEJIDAD PARA THINKING MODE
# ══════════════════════════════════════════════════════════════════════════════

def should_use_thinking(has_document: bool, is_drafting: bool) -> bool:
    """Activa thinking mode SOLO para modos especiales.
    
    Thinking ON (50K tokens, reasoning CoT):
    - Documentos adjuntos (Centinela: análisis de demandas/sentencias)
    - Redacción de documentos legales
    
    Thinking OFF (8192 tokens, respuesta directa):
    - Modo pregunta/chat normal (consultas jurídicas)
    """
    if has_document:
        print("   🧠 Thinking ON: documento adjunto (Centinela)")
        return True
    
    if is_drafting:
        print("   🧠 Thinking ON: modo redacción")
        return True
    
    print("   ⚡ Thinking OFF: modo chat")
    return False


# ══════════════════════════════════════════════════════════════════════════════
# SECURITY: Malicious Prompt Detection
# ══════════════════════════════════════════════════════════════════════════════

import re as _security_re

_SECURITY_PATTERNS = [
    (_security_re.compile(r'(?i)(?:c[oó]mo\s+funciona|c[oó]digo\s+fuente|arquitectura|backend|frontend|api\s*key|system\s*prompt|dame\s+(?:el|tu)\s+prompt).*(?:iurexia|jurexia|esta\s+(?:plataforma|herramienta|app))'), 'architecture_probe', 'high'),
    (_security_re.compile(r'(?i)(?:mu[eé]strame|revela|dame|ense[ñn]a|comparte).*(?:prompt|instrucciones|system|configuraci[oó]n)'), 'prompt_extraction', 'high'),
    (_security_re.compile(r'(?i)(?:token|api\s*key|password|contrase[ñn]a|secret|clave).*(?:iurexia|jurexia|supabase|openai|deepseek|stripe|qdrant)'), 'credential_probe', 'critical'),
    (_security_re.compile(r'(?i)(?:ignore|forget|olvida|ignora).*(?:previous|previas|anteriores|instrucciones|instructions)'), 'prompt_injection', 'critical'),
    (_security_re.compile(r'(?i)(?:eres|act[uú]a\s+como|you\s+are|pretend).*(?:chatgpt|claude|llama|gpt|asistente\s+sin\s+restricciones)'), 'jailbreak', 'high'),
    (_security_re.compile(r'(?i)(?:qu[eé]\s+modelo|qu[eé]\s+llm|qu[eé]\s+api|qu[eé]\s+base\s+de\s+datos|stack\s+tecnol[oó]gico|tech\s*stack).*(?:usas|utilizas|tienes|empleas|usa\s+iurexia)'), 'architecture_probe', 'medium'),
    (_security_re.compile(r'(?i)(?:scrap|copia|clona|replica|reverse\s*engineer|descompil).*(?:iurexia|jurexia|c[oó]digo|sistema|plataforma)'), 'reverse_engineering', 'critical'),
]

def _check_security_patterns(message: str) -> tuple:
    """Check if message matches any security pattern. Returns (alert_type, severity) or (None, None)."""
    for pattern, alert_type, severity in _SECURITY_PATTERNS:
        if pattern.search(message):
            return alert_type, severity
    return None, None

def _log_security_alert(user_id: str, user_email: str, query: str, alert_type: str, severity: str):
    """Log a security alert to Supabase (fire-and-forget)."""
    if not supabase_admin:
        return
    try:
        supabase_admin.table("security_alerts").insert({
            "user_id": user_id if user_id and user_id != "anonymous" else None,
            "user_email": user_email or "unknown",
            "query_text": query[:500],
            "alert_type": alert_type,
            "severity": severity,
        }).execute()
        print(f"🚨 SECURITY ALERT [{severity.upper()}]: {alert_type} by {user_email or user_id}")
    except Exception as e:
        print(f"⚠️ Failed to log security alert: {e}")

# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINT: CHAT (STREAMING SSE CON THINKING MODE + RAG)
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

    # ─────────────────────────────────────────────────────────────────────
    # BLOCKED USER CHECK
    # ─────────────────────────────────────────────────────────────────────
    if request.user_id and supabase_admin:
        try:
            blocked_result = await asyncio.to_thread(
                lambda: supabase_admin.rpc(
                    'is_user_blocked', {'p_user_id': request.user_id}
                ).execute()
            )
            if blocked_result.data:
                print(f"🚫 BLOCKED USER attempted chat: {request.user_id}")
                return StreamingResponse(
                    iter([json.dumps({
                        "error": "account_suspended",
                        "message": "Tu cuenta ha sido suspendida. Contacta a soporte para más información.",
                    })]),
                    status_code=403,
                    media_type="application/json",
                )
        except Exception as e:
            print(f"⚠️ Blocked check failed (proceeding): {e}")

    # ─────────────────────────────────────────────────────────────────────
    # SECURITY: Malicious prompt detection
    # ─────────────────────────────────────────────────────────────────────
    _last_msg_for_sec = None
    for msg in reversed(request.messages):
        if msg.role == "user":
            _last_msg_for_sec = msg.content
            break
    if _last_msg_for_sec:
        _alert_type, _alert_severity = _check_security_patterns(_last_msg_for_sec)
        if _alert_type:
            _log_security_alert(
                user_id=request.user_id or "anonymous",
                user_email="",
                query=_last_msg_for_sec,
                alert_type=_alert_type,
                severity=_alert_severity,
            )
            # For critical severity, block the request entirely
            if _alert_severity == "critical":
                return StreamingResponse(
                    iter([json.dumps({
                        "error": "security_blocked",
                        "message": "Tu consulta no puede ser procesada. Si crees que esto es un error, contacta a soporte.",
                    })]),
                    status_code=403,
                    media_type="application/json",
                )

    # ─────────────────────────────────────────────────────────────────────
    # QUOTA CHECK: Server-side enforcement via Supabase consume_query RPC
    # ─────────────────────────────────────────────────────────────────────
    if request.user_id and supabase_admin:
        try:
            quota_result = await asyncio.to_thread(
                lambda: supabase_admin.rpc(
                    'consume_query', {'p_user_id': request.user_id}
                ).execute()
            )

            if quota_result.data:
                quota_data = quota_result.data
                if not quota_data.get('allowed', True):
                    return StreamingResponse(
                        iter([json.dumps({
                            "error": "quota_exceeded",
                            "message": "Has alcanzado tu límite de consultas para este período.",
                            "used": quota_data.get('used', 0),
                            "limit": quota_data.get('limit', 0),
                            "subscription_type": quota_data.get('subscription_type', 'gratuito'),
                        })]),
                        status_code=403,
                        media_type="application/json",
                    )
                print(f"✅ Quota OK: {quota_data.get('used')}/{quota_data.get('limit')} "
                      f"({quota_data.get('subscription_type')})")
            else:
                print(f"⚠️ consume_query returned no data for user_id={request.user_id}")
        except Exception as e:
            # Don't block chat on quota check failure — log and continue
            print(f"⚠️ Quota check failed (proceeding anyway): {e}")
    else:
        print(f"⚠️ Quota check SKIPPED — user_id={'SET' if request.user_id else 'MISSING'}, supabase_admin={'SET' if supabase_admin else 'NONE'}")
    
    # Extraer última pregunta del usuario
    last_user_message = None
    for msg in reversed(request.messages):
        if msg.role == "user":
            last_user_message = msg.content
            break
    
    if not last_user_message:
        raise HTTPException(status_code=400, detail="No se encontró mensaje del usuario")
    
    # Detectar si hay documento adjunto
    has_document = "DOCUMENTO ADJUNTO:" in last_user_message or "DOCUMENTO_INICIO" in last_user_message
    
    # Detectar si es análisis de sentencia (AUDITAR_SENTENCIA → o3 model)
    is_sentencia = "SENTENCIA_INICIO" in last_user_message or "AUDITAR_SENTENCIA" in last_user_message
    if is_sentencia:
        has_document = True  # Also triggers document RAG path
        print("   ⚖️ MODO SENTENCIA detectado — activando análisis con OpenAI o3")
    
    # Detectar si es una solicitud de redacción de documento
    is_drafting = "[REDACTAR_DOCUMENTO]" in last_user_message
    draft_tipo = None
    draft_subtipo = None
    
    # ── Natural language drafting detection ("redacta", "ayúdame a redactar", etc.) ──
    is_chat_drafting = False
    if not is_drafting and not has_document and not is_sentencia:
        is_chat_drafting = _detect_chat_drafting(last_user_message)
        if is_chat_drafting:
            print(f"   ✍️ MODO REDACCIÓN CHAT detectado por lenguaje natural")
    
    if is_drafting:
        # Extraer tipo y subtipo del mensaje de redacción (UI-triggered)
        import re
        tipo_match = re.search(r'Tipo:\s*(\w+)', last_user_message)
        subtipo_match = re.search(r'Subtipo:\s*(\w+)', last_user_message)
        if tipo_match:
            draft_tipo = tipo_match.group(1).lower()
        if subtipo_match:
            draft_subtipo = subtipo_match.group(1).lower()
        print(f" Modo REDACCIÓN detectado - Tipo: {draft_tipo}, Subtipo: {draft_subtipo}")
    
    # DA VINCI: Inicializar variables de comparación multi-estado
    multi_states = None
    is_comparative = False
    
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
                top_k=15,
                forced_materia=request.materia,
                fuero=request.fuero,
            )
            doc_id_map = build_doc_id_map(search_results)
            context_xml = format_results_as_xml(search_results)
            print(f"   Encontrados {len(search_results)} documentos para fundamentar redacción")
        elif has_document:
            # Para documentos: extraer términos clave y buscar contexto relevante
            
            # Determinar marker de contenido según tipo
            if is_sentencia:
                doc_start_idx = last_user_message.find("<!-- SENTENCIA_INICIO -->")
                doc_end_idx = last_user_message.find("<!-- SENTENCIA_FIN -->")
                print("   ⚖️ Sentencia detectada — extrayendo términos para búsqueda RAG ampliada")
            else:
                doc_start_idx = last_user_message.find("<!-- DOCUMENTO_INICIO -->")
                doc_end_idx = -1
                print("   📄 Documento adjunto detectado - extrayendo términos para búsqueda RAG")
            
            if doc_start_idx != -1:
                if doc_end_idx != -1:
                    doc_content = last_user_message[doc_start_idx:doc_end_idx]
                else:
                    doc_content = last_user_message[doc_start_idx:doc_start_idx + 5000]
            else:
                doc_content = last_user_message[:3000]
            
            if is_sentencia:
                # ─────────────────────────────────────────────────────────────
                # SMART RAG para sentencias: extrae términos legales clave
                # del documento completo y hace múltiples búsquedas dirigidas
                # ─────────────────────────────────────────────────────────────
                import re
                
                # Extraer artículos citados ("artículo 14", "Art. 193", etc.)
                articulos = re.findall(
                    r'(?:art[ií]culo|art\.?)\s*(\d+[\w°]*(?:\s*(?:,|y|al)\s*\d+[\w°]*)*)',
                    doc_content, re.IGNORECASE
                )
                
                # Extraer leyes/códigos mencionados
                leyes_patterns = [
                    r'(?:Ley\s+(?:de|del|Nacional|Federal|General|Orgánica|para)\s+[\w\s]+?)(?:\.|\ |,|;)',
                    r'(?:Código\s+(?:Penal|Civil|Nacional|de\s+\w+)[\w\s]*?)(?:\.|\ |,|;)',
                    r'(?:Constitución\s+Política[\w\s]*)',
                    r'CPEUM',
                    r'(?:Ley\s+de\s+Amparo)',
                ]
                leyes_encontradas = []
                for pat in leyes_patterns:
                    matches = re.findall(pat, doc_content, re.IGNORECASE)
                    leyes_encontradas.extend([m.strip() for m in matches[:5]])
                
                # Extraer temas jurídicos clave
                temas_patterns = [
                    r'(?:juicio\s+de\s+amparo)',
                    r'(?:recurso\s+de\s+revisión)',
                    r'(?:principio\s+(?:pro persona|de legalidad|de retroactividad))',
                    r'(?:control\s+(?:de convencionalidad|difuso|concentrado))',
                    r'(?:derechos humanos)',
                    r'(?:debido proceso)',
                    r'(?:retroactividad)',
                    r'(?:cosa juzgada)',
                    r'(?:suplencia\s+de\s+la\s+queja)',
                    r'(?:interés\s+(?:jurídico|legítimo|superior))',
                ]
                temas = []
                for pat in temas_patterns:
                    if re.search(pat, doc_content, re.IGNORECASE):
                        temas.append(re.search(pat, doc_content, re.IGNORECASE).group())
                
                # Construir queries dirigidas
                articulos_str = ", ".join(set(articulos[:10]))
                leyes_str = ", ".join(set(leyes_encontradas[:8]))
                temas_str = ", ".join(set(temas[:6]))
                
                # Query 1: Legislación (artículos + leyes específicas)
                query_legislacion = f"fundamentación legal artículos {articulos_str} {leyes_str}".strip()
                # Query 2: Jurisprudencia (temas jurídicos + materia)
                query_jurisprudencia = f"jurisprudencia tesis {temas_str} {leyes_str} aplicación retroactiva derechos".strip()
                # Query 3: Materia constitucional
                query_constitucional = f"constitución derechos humanos principio pro persona debido proceso artículos 1 14 16 17 CPEUM"
                
                print(f"   ⚖️ SMART RAG — Queries construidas:")
                print(f"      Legislación: {query_legislacion[:120]}...")
                print(f"      Jurisprudencia: {query_jurisprudencia[:120]}...")
                print(f"      Constitucional: {query_constitucional[:80]}...")
                print(f"      Artículos detectados: {articulos_str[:100]}")
                print(f"      Leyes detectadas: {leyes_str[:100]}")
                print(f"      Temas detectados: {temas_str[:100]}")
                
                # Ejecutar 3 búsquedas en paralelo
                import asyncio
                results_legislacion, results_jurisprudencia, results_constitucional = await asyncio.gather(
                    hybrid_search_all_silos(
                        query=query_legislacion,
                        estado=request.estado,
                        top_k=15,
                        fuero=request.fuero,
                    ),
                    hybrid_search_all_silos(
                        query=query_jurisprudencia,
                        estado=request.estado,
                        top_k=15,
                        fuero=request.fuero,
                    ),
                    hybrid_search_all_silos(
                        query=query_constitucional,
                        estado=request.estado,
                        top_k=10,
                        fuero=request.fuero,
                    ),
                )
                
                # Merge results, deduplicando por ID
                seen_ids = set()
                search_results = []
                for result_set in [results_legislacion, results_jurisprudencia, results_constitucional]:
                    for r in result_set:
                        rid = r.id if hasattr(r, 'id') else str(r)
                        if rid not in seen_ids:
                            seen_ids.add(rid)
                            search_results.append(r)
                
                print(f"   ⚖️ SMART RAG — Total: {len(search_results)} docs únicos")
                print(f"      Legislación: {len(results_legislacion)}, Jurisprudencia: {len(results_jurisprudencia)}, Constitucional: {len(results_constitucional)}")
            else:
                search_query = f"análisis jurídico: {doc_content[:1500]}"
                search_results = await hybrid_search_all_silos(
                    query=search_query,
                    estado=request.estado,
                    top_k=15,
                    fuero=request.fuero,
                )
            
            doc_id_map = build_doc_id_map(search_results)
            context_xml = format_results_as_xml(search_results)
            print(f"   Encontrados {len(search_results)} documentos relevantes para contrastar")
        else:
            # ─────────────────────────────────────────────────────────────────
            # Detección multi-estado para comparaciones
            # ─────────────────────────────────────────────────────────────────
            multi_states = detect_multi_state_query(last_user_message)
            is_comparative = multi_states is not None
            
            if is_comparative:
                # MODO COMPARATIVO: Búsqueda paralela por estado
                print(f"   🎨 MODO COMPARATIVO activado: {len(multi_states)} estados")
                multi_result = await hybrid_search_multi_state(
                    query=last_user_message,
                    estados=multi_states,

                    top_k_per_state=5,
                )
                search_results = multi_result["all_results"]
                doc_id_map = build_doc_id_map(search_results)
                context_xml = multi_result["context_xml"]
                
                # Inyectar instrucción comparativa en el contexto
                estados_str = ", ".join(multi_states)
                context_xml = (
                    f"\n<!-- INSTRUCCIÓN COMPARATIVA: El usuario quiere comparar legislación entre: {estados_str}. "
                    f"Los documentos están agrupados por estado. Genera una respuesta comparativa "
                    f"organizada por estado, citando artículos específicos de cada uno. "
                    f"Usa una tabla comparativa cuando sea apropiado. -->\n"
                    + context_xml
                )
            else:
                # Consulta normal
                # AUTO-DETECT: Si el usuario no seleccionó estado, intentar detectar uno de la query
                effective_estado = request.estado
                if not effective_estado:
                    auto_estado = detect_single_estado_from_query(last_user_message)
                    if auto_estado:
                        effective_estado = auto_estado
                        print(f"   📍 AUTO-DETECT: Usando estado '{auto_estado}' detectado de la query")
                
                search_results = await hybrid_search_all_silos(
                    query=last_user_message,
                    estado=effective_estado,
                    top_k=request.top_k,
                    forced_materia=request.materia,
                    fuero=request.fuero,
                )
                doc_id_map = build_doc_id_map(search_results)
                context_xml = format_results_as_xml(search_results, estado=effective_estado)
                
                # === PRODUCTION LOG: verificar qué documentos van al LLM ===
                estatales_in_context = [r for r in search_results if r.silo == "leyes_estatales"]
                print(f"\n   🔬 CONTEXT AUDIT (estado={effective_estado}):")
                print(f"      Total docs en contexto: {len(search_results)}")
                print(f"      Leyes estatales: {len(estatales_in_context)}")
                for r in estatales_in_context[:5]:
                    print(f"         → ref={r.ref}, origen={r.origen[:50] if r.origen else 'N/A'}, score={r.score:.4f}")
                print(f"      context_xml length: {len(context_xml)} chars")
        
        # ─────────────────────────────────────────────────────────────────────
        # PASO 2: Construir mensajes para LLM
        # ─────────────────────────────────────────────────────────────────────
        # Select appropriate system prompt based on mode
        if is_drafting and draft_tipo:
            system_prompt = get_drafting_prompt(draft_tipo, draft_subtipo or "")
            print(f"   Usando prompt de redacción para: {draft_tipo}")
        elif is_sentencia:
            system_prompt = SYSTEM_PROMPT_SENTENCIA_ANALYSIS
            print("   ⚖️ Usando prompt MAGISTRADO para análisis de sentencia")
        elif has_document:
            system_prompt = SYSTEM_PROMPT_DOCUMENT_ANALYSIS
        elif not is_drafting and not has_document and multi_states:
            # DA VINCI: Prompt comparativo para multi-estado
            system_prompt = SYSTEM_PROMPT_CHAT + (
                "\n\n## MODO COMPARATIVO CROSS-STATE\n"
                "El usuario está comparando legislación entre múltiples estados mexicanos.\n"
                "INSTRUCCIONES ESPECIALES:\n"
                "1. Los documentos están agrupados por estado (<!-- ESTADO: X -->)\n"
                "2. Para cada estado, cita los artículos ESPECÍFICOS encontrados con [Doc ID: xxx]\n"
                "3. Organiza tu respuesta con secciones claras por estado\n"
                "4. Si es apropiado, incluye una TABLA COMPARATIVA con columnas: Estado | Artículo | Tipo Penal/Sanción\n"
                "5. Al final, agrega un ANÁLISIS comparativo de similitudes y diferencias\n"
                "6. Si un estado no tiene información suficiente, indícalo claramente\n"
            )
        elif is_chat_drafting:
            system_prompt = SYSTEM_PROMPT_CHAT_DRAFTING
            print("   ✍️ Usando prompt CHAT DRAFTING para redacción por lenguaje natural")
        else:
            system_prompt = SYSTEM_PROMPT_CHAT
        llm_messages = [
            {"role": "system", "content": system_prompt},
        ]
        
        # Inyección de Contexto Global: Inventario del Sistema
        llm_messages.append({"role": "system", "content": INVENTORY_CONTEXT})
        
        # Inyectar estado seleccionado para que el LLM priorice leyes locales
        # effective_estado sólo existe en el flujo normal; usar request.estado como fallback
        _estado_for_llm = locals().get("effective_estado") or request.estado
        if _estado_for_llm:
            estado_humano = _estado_for_llm.replace("_", " ").title()
            llm_messages.append({"role": "system", "content": (
                f"ESTADO SELECCIONADO POR EL USUARIO: {estado_humano}\n\n"
                f"INSTRUCCIÓN CRÍTICA — PRIORIDAD DE FUENTES:\n"
                f"1. El usuario consulta desde {estado_humano}. Los documentos del contexto "
                f"que provienen de leyes de {estado_humano} son la FUENTE PRINCIPAL.\n"
                f"2. En la sección '## Fundamento Legal', TRANSCRIBE PRIMERO los artículos "
                f"TEXTUALES de las leyes de {estado_humano} que estén en el contexto. "
                f"Copia el texto del artículo tal como aparece en el contexto con su [Doc ID: uuid].\n"
                f"3. Las leyes federales (Código Civil Federal, etc.) son SUPLETORIAS — "
                f"cítalas DESPUÉS de los artículos locales, no en lugar de ellos.\n"
                f"4. La jurisprudencia COMPLEMENTA el fundamento legal, no lo reemplaza. "
                f"Primero cita el artículo de la ley local, luego la tesis que lo interpreta.\n"
                f"5. NUNCA digas 'consulte la ley local' ni 'esos textos no se transcriben aquí' "
                f"— TÚ tienes los artículos de la ley local en el contexto, TRANSCRÍBELOS."
            )})
            print(f"   📍 Estado inyectado al LLM: {estado_humano}")
        
        if context_xml:
            llm_messages.append({"role": "system", "content": f"CONTEXTO JURÍDICO RECUPERADO:\n{context_xml}"})
        
        # FIX A: Inject compact Doc ID inventory to reduce UUID hallucination
        # Gives the LLM a "cheat sheet" of valid UUIDs to copy from
        if doc_id_map:
            valid_ids_prompt = get_valid_doc_ids_prompt(doc_id_map)
            llm_messages.append({"role": "system", "content": valid_ids_prompt})
        
        # Agregar historial conversacional
        for msg in request.messages:
            msg_content = msg.content
            
            # Para sentencias: truncar si es necesario para token budget
            if is_sentencia and msg.role == "user" and "SENTENCIA_INICIO" in msg_content:
                s_start = msg_content.find("<!-- SENTENCIA_INICIO -->")
                s_end = msg_content.find("<!-- SENTENCIA_FIN -->")
                if s_start != -1 and s_end != -1:
                    sentencia_text = msg_content[s_start:s_end + len("<!-- SENTENCIA_FIN -->")]
                    # o3-mini tiene un TPM más alto, pero truncamos a 80K chars (~20K tokens) por seguridad
                    max_chars = 80000
                    if len(sentencia_text) > max_chars:
                        truncated = sentencia_text[:max_chars]
                        pct = round(max_chars / len(sentencia_text) * 100)
                        truncated += f"\n\n[NOTA: Sentencia truncada al {pct}% para análisis. Se incluyen las secciones principales.]"
                        truncated += "\n<!-- SENTENCIA_FIN -->"
                        msg_content = msg_content[:s_start] + truncated + msg_content[s_end + len("<!-- SENTENCIA_FIN -->"):]
                        print(f"   ⚖️ Sentencia truncada: {len(sentencia_text)} → {max_chars} chars ({pct}%)")
                    else:
                        print(f"   ⚖️ Sentencia completa: {len(sentencia_text)} chars (dentro del límite)")
            
            llm_messages.append({"role": msg.role, "content": msg_content})
        
        # ─────────────────────────────────────────────────────────────────────
        # PASO 3: Generar respuesta con Thinking Mode auto-detectado
        # ─────────────────────────────────────────────────────────────────────
        # MODELO DUAL:
        # - Thinking OFF → o4-mini (chat_client) para calidad + costo eficiente
        # - Thinking ON → DeepSeek Chat con thinking enabled (deepseek_client) para CoT
        
        use_thinking = should_use_thinking(has_document, is_drafting)
        
        if is_sentencia:
            # Sentencia analysis: OpenAI o3-mini (powerful reasoning, cost-effective)
            active_client = chat_client  # Same OpenAI API key
            active_model = SENTENCIA_MODEL
            max_tokens = 16000  # Máximo output para análisis exhaustivo
            use_thinking = False  # o3-mini handles reasoning internally
            print(f"   ⚖️ Modelo SENTENCIA: {SENTENCIA_MODEL} | max_tokens: {max_tokens}")
        elif use_thinking:
            # DeepSeek with thinking: max 50K tokens, uses extra_body
            active_client = deepseek_client
            active_model = DEEPSEEK_CHAT_MODEL
            max_tokens = 50000
        else:
            # Chat Engine Toggle: DeepSeek V3 (cost-optimized) or GPT-5 Mini (premium)
            if CHAT_ENGINE == "deepseek" and deepseek_client:
                active_client = deepseek_client
                active_model = DEEPSEEK_CHAT_MODEL  # deepseek-chat (V3)
                max_tokens = 32768
            else:
                active_client = chat_client
                active_model = CHAT_MODEL  # gpt-5-mini
                max_tokens = 32768
        
        print(f"   Modelo: {active_model} | Thinking: {'ON' if use_thinking else 'OFF'} | Docs: {len(search_results)} | Messages: {len(llm_messages)}")
        
        # ── STREAMING UNIFICADO: Con o sin thinking ──────────────────────
        async def generate_stream() -> AsyncGenerator[str, None]:
            """Stream unificado — thinking mode envía reasoning con marcadores.
            
            PROTOCOL:
            - Reasoning tokens: prefixed with <!--thinking--> marker
            - Content tokens: sent as plain text (no prefix)
            - The <!--thinking--> marker is ALWAYS yielded atomically (never split)
            - If thinking produces reasoning but ZERO content, we surface the
              reasoning as content so the user isn't left with an empty response.
            """
            try:
                reasoning_buffer = ""
                content_buffer = ""
                
                api_kwargs = {
                    "model": active_model,
                    "messages": llm_messages,
                    "stream": True,
                }
                # GPT-5 Mini uses max_completion_tokens; DeepSeek uses max_tokens
                if use_thinking:
                    api_kwargs["max_tokens"] = max_tokens
                    api_kwargs["extra_body"] = {"thinking": {"type": "enabled"}}
                else:
                    api_kwargs["max_completion_tokens"] = max_tokens
                
                stream = await active_client.chat.completions.create(**api_kwargs)
                
                async for chunk in stream:
                    if chunk.choices and chunk.choices[0].delta:
                        delta = chunk.choices[0].delta
                        
                        reasoning_content = getattr(delta, 'reasoning_content', None)
                        content = getattr(delta, 'content', None)
                        
                        if reasoning_content:
                            reasoning_buffer += reasoning_content
                            # Thinking mode improves response quality but the raw
                            # reasoning_content is garbled/compressed by DeepSeek
                            # (truncated syllables, not meant for display).
                            # We buffer it for logging but do NOT stream to client.
                        
                        if content:
                            content_buffer += content
                            yield content
                
                # Edge case: thinking mode produced reasoning but ZERO content
                # (model exhausted tokens during reasoning phase)
                if use_thinking and reasoning_buffer and not content_buffer.strip():
                    print(f"   ⚠️ Thinking exhausted tokens — {len(reasoning_buffer)} chars reasoning, 0 content")
                    # Yield a visible fallback so the user sees SOMETHING
                    fallback = (
                        "\n\n**Análisis completado.**\n\n"
                        "El modelo utilizó todos los tokens disponibles durante el análisis interno. "
                        "Envía un mensaje de seguimiento como *\"responde\"* o *\"continúa\"* "
                        "para obtener la respuesta estructurada."
                    )
                    content_buffer = fallback
                    yield fallback
                
                # Validar citas
                if doc_id_map:
                    validation = validate_citations(content_buffer, doc_id_map)
                    
                    # Build sources map: uuid → {origen, ref, texto} for ALL cited docs
                    sources_map = {}
                    for cv in validation.citations:
                        doc = doc_id_map.get(cv.doc_id)
                        if doc:
                            # Truncate texto to 2000 chars to avoid bloating SSE
                            texto_truncated = (doc.texto or "")[:2000]
                            # Determinar pdf_url: Qdrant payload > treaty-specific > silo fallback
                            pdf_url = doc.pdf_url or _resolve_treaty_pdf(doc.origen) or PDF_FALLBACK_URLS.get(doc.silo)
                            sources_map[cv.doc_id] = {
                                "origen": humanize_origen(doc.origen) or "Fuente legal",
                                "ref": doc.ref or "",
                                "texto": texto_truncated,
                                "pdf_url": pdf_url or None,
                                "silo": doc.silo,
                            }
                        else:
                            sources_map[cv.doc_id] = {
                                "origen": "Fuente no verificada",
                                "ref": "",
                                "texto": ""
                            }
                    
                    if validation.invalid_count > 0:
                        print(f"   ⚠️ CITAS INVÁLIDAS: {validation.invalid_count}/{validation.total_citations}")
                        for cv in validation.citations:
                            if cv.status == "invalid":
                                print(f"      ❌ UUID no encontrado: {cv.doc_id}")
                    else:
                        print(f"   ✅ Validación OK: {validation.valid_count} citas verificadas")
                    
                    # Always emit CITATION_META with sources map
                    meta = json.dumps({
                        "valid": validation.valid_count,
                        "invalid": validation.invalid_count,
                        "total": validation.total_citations,
                        "invalid_ids": [c.doc_id for c in validation.citations if c.status == "invalid"],
                        "sources": sources_map
                    })
                    yield f"\n\n<!-- CITATION_META:{meta} -->"
                
                thinking_info = f", {len(reasoning_buffer)} chars reasoning" if reasoning_buffer else ""
                print(f"   📝 Respuesta ({len(content_buffer)} chars content{thinking_info})")
                
            except Exception as e:
                yield f"\n\n❌ Error: {str(e)}"
        
        return StreamingResponse(
            generate_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
                "X-Model-Used": active_model,
                "X-Thinking-Mode": "on" if use_thinking else "off",
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
        
        extraction_response = await chat_client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[{"role": "user", "content": extraction_prompt}],
            temperature=0.2,
            max_completion_tokens=500,
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
        
        audit_response = await chat_client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_AUDIT},
                {"role": "user", "content": audit_prompt},
            ],
            temperature=0.2,
            max_completion_tokens=3000,
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
        
        # Llamar a GPT-5 Mini
        response = await chat_client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Mejora el siguiente texto legal:\n\n{request.texto}"},
            ],
            temperature=0.3,  # Más conservador para mantener fidelidad
            max_completion_tokens=8000,
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
# CHAT DE ASISTENCIA EN REDACCIÓN DE SENTENCIAS — Gemini 2.5 Pro Streaming
# ══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT_SENTENCIA_CHAT = """Eres JUREXIA REDACTOR JUDICIAL, un sistema de inteligencia artificial
especializado en la redacción de sentencias para Tribunales Colegiados de Circuito del Poder Judicial
de la Federación de México. Tu función es asistir a secretarios de tribunal en la elaboración,
modificación, mejora y continuación de proyectos de sentencia con la máxima calidad jurídica.

═══════════════════════════════════════════════════════════════
   ROL Y ESPECIALIZACIÓN
═══════════════════════════════════════════════════════════════

Eres un redactor judicial experto, NO un chatbot generalista. Tu especialización es:
- Redactar ESTUDIOS DE FONDO para amparos directos, revisión, queja y revisión fiscal
- Continuar redacciones interrumpidas manteniendo coherencia de estilo y argumento
- Modificar el sentido de un agravio (de fundado a infundado o viceversa) con re-fundamentación
- Ampliar considerandos con mayor profundidad jurídica
- Restructurar argumentos manteniendo la lógica del silogismo jurídico
- Mejorar redacción con citas textuales de legislación y jurisprudencia

═══════════════════════════════════════════════════════════════
   FORMATO JUDICIAL OBLIGATORIO
═══════════════════════════════════════════════════════════════

Tu redacción SIEMPRE debe seguir el estilo judicial formal de los TCC:
- Párrafos extensos y bien fundamentados (NO bullets ni listas)
- Lenguaje formal de sentencia: "este tribunal advierte", "contrario a lo aducido por el quejoso",
  "de la lectura integral del acto reclamado se desprende", etc.
- Citas textuales de artículos con número de ley y artículo específico
- Referencia a tesis y jurisprudencia con formato: Registro digital [número], [Época], [Tribunal]
- Silogismo jurídico: premisa mayor (norma), premisa menor (hechos), conclusión
- Transiciones fluidas entre argumentos ("En ese orden de ideas...", "Aunado a lo anterior...",
  "Robustece lo anterior...", "No es óbice a lo anterior...")

═══════════════════════════════════════════════════════════════
   MODOS DE OPERACIÓN
═══════════════════════════════════════════════════════════════

1. **CONTINUAR REDACCIÓN**: Si el usuario pega texto de una sentencia en proceso, CONTINÚA
   la redacción de forma natural, manteniendo el mismo estilo, voz narrativa y profundidad.
   NO repitas lo que ya escribió. Inicia exactamente donde terminó.

2. **MODIFICAR SENTIDO**: Si el usuario pide cambiar el sentido de un agravio:
   - Analiza los fundamentos del texto original
   - Reconstruye el argumento con el nuevo sentido
   - Mantén las citas de ley que apliquen y sustituye las que contradigan el nuevo sentido
   - Fundamenta exhaustivamente la nueva postura

3. **AMPLIAR/MEJORAR**: Si el usuario pide ampliar o mejorar una sección:
   - Identifica qué elementos faltan (fundamentación, motivación, análisis comparativo)
   - Agrega análisis más profundo SIN eliminar lo existente
   - Integra jurisprudencia aplicable cuando sea pertinente

4. **REDACCIÓN NUEVA**: Si el usuario describe un caso y pide redactar, genera texto judicial
   completo con estructura de sentencia.

═══════════════════════════════════════════════════════════════
   USO DEL CONTEXTO RAG
═══════════════════════════════════════════════════════════════

Si se proporciona CONTEXTO JURÍDICO RECUPERADO:
- INTEGRA las fuentes en tu redacción como citas textuales
- Usa [Doc ID: uuid] para cada fuente citada
- Transcribe artículos relevantes, no solo los menciones
- La jurisprudencia fortalece enormemente el argumento — úsala siempre que aplique

Si NO se proporciona contexto RAG:
- Redacta con tu conocimiento jurídico
- Las citas a legislación y jurisprudencia son basadas en tu entrenamiento
- NO inventes números de registro digital ni rubros de tesis específicos
- En su lugar, describe la tesis por su contenido: "existe criterio jurisprudencial que establece..."

═══════════════════════════════════════════════════════════════
   PROHIBICIONES Y CALIDAD
═══════════════════════════════════════════════════════════════

- NUNCA uses emojis, emoticonos ni lenguaje coloquial
- NUNCA generes listas con bullets en el texto de sentencia (solo en comentarios al usuario)
- NUNCA uses el formato de chatbot — tu output debe poder insertarse directamente en un DOCX de sentencia
- MANTÉN la coherencia narrativa con el texto previo del usuario
- Cuando el usuario te da instrucciones, distingue claramente entre:
  a) Instrucciones META (qué hacer) → responde brevemente y ejecuta
  b) Texto de sentencia para continuar → continúa directamente sin preámbulo
"""


class ChatSentenciaMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str


class ChatSentenciaRequest(BaseModel):
    messages: List[ChatSentenciaMessage]
    user_id: Optional[str] = None
    user_email: Optional[str] = None
    use_rag: bool = True
    attached_document: Optional[str] = None  # extracted text from uploaded file


@app.post("/chat-sentencia")
async def chat_sentencia_endpoint(request: ChatSentenciaRequest):
    """
    Chat de Asistencia en Redacción de Sentencias — Gemini 2.5 Pro Streaming.
    
    Specialized chat for TCC secretaries to modify, adjust, improve, or continue
    sentence drafts. Uses Gemini 2.5 Pro with SSE streaming.
    
    Features:
    - RAG toggle (use_rag=true → searches verified database)
    - Attached document support (extracted text injected as context)
    - Conversation memory (stateless, full history sent from frontend)
    - SSE streaming response
    """
    if not request.messages:
        raise HTTPException(status_code=400, detail="Se requiere al menos un mensaje")
    
    # ── Gemini API key check ──────────────────────────────────────────────
    gemini_key = os.getenv("GEMINI_API_KEY", "")
    if not gemini_key:
        raise HTTPException(500, "Gemini API key not configured")
    
    # ── Quota check (reuse /chat pattern) ─────────────────────────────────
    if request.user_id and supabase_admin:
        try:
            quota_result = await asyncio.to_thread(
                lambda: supabase_admin.rpc(
                    'consume_query', {'p_user_id': request.user_id}
                ).execute()
            )
            if quota_result.data:
                quota_data = quota_result.data
                if not quota_data.get('allowed', True):
                    return StreamingResponse(
                        iter([json.dumps({
                            "error": "quota_exceeded",
                            "message": "Has alcanzado tu límite de consultas para este período.",
                            "used": quota_data.get('used', 0),
                            "limit": quota_data.get('limit', 0),
                            "subscription_type": quota_data.get('subscription_type', 'gratuito'),
                        })]),
                        status_code=403,
                        media_type="application/json",
                    )
        except Exception as e:
            print(f"⚠️ Quota check failed for chat-sentencia (proceeding): {e}")
    
    # ── Extract last user message ─────────────────────────────────────────
    last_user_message = None
    for msg in reversed(request.messages):
        if msg.role == "user":
            last_user_message = msg.content
            break
    
    if not last_user_message:
        raise HTTPException(status_code=400, detail="No se encontró mensaje del usuario")
    
    print(f"\n🏛️ CHAT SENTENCIA — user: {request.user_email or 'anon'}")
    print(f"   📝 Query ({len(last_user_message)} chars): {last_user_message[:200]}...")
    print(f"   🔍 RAG: {'ON' if request.use_rag else 'OFF'}")
    print(f"   📎 Documento adjunto: {'Sí' if request.attached_document else 'No'}")
    
    try:
        from google import genai
        from google.genai import types as gtypes
        
        # ── RAG search (optional) ────────────────────────────────────────
        rag_context = ""
        rag_count = 0
        if request.use_rag:
            try:
                search_results = await hybrid_search_all_silos(
                    query=last_user_message,
                    estado=None,
                    top_k=15,
                    enable_reasoning=False,
                )
                if search_results:
                    rag_context = format_results_as_xml(search_results, estado=None)
                    rag_count = len(search_results)
                    print(f"   ✅ RAG: {rag_count} resultados, {len(rag_context)} chars contexto")
            except Exception as e:
                print(f"   ⚠️ RAG search failed (continuing without): {e}")
                rag_context = ""
        
        # ── Build conversation for Gemini ─────────────────────────────────
        # Gemini uses contents=[...] with role "user"/"model"
        system_instruction = SYSTEM_PROMPT_SENTENCIA_CHAT
        
        # Add RAG context to system instruction if available
        if rag_context:
            system_instruction += f"""

═══════════════════════════════════════════════════════════════
   CONTEXTO JURÍDICO RECUPERADO (BASE DE DATOS VERIFICADA)
═══════════════════════════════════════════════════════════════

Los siguientes documentos fueron recuperados de la base de datos verificada de Iurexia.
USA estas fuentes para fundamentar tu redacción. CITA con [Doc ID: uuid] cada fuente que uses.

{rag_context}
"""
        elif not request.use_rag:
            system_instruction += """

⚠️ MODO SIN BASE DE DATOS: El usuario ha desactivado la búsqueda en la base de datos verificada.
Tus respuestas se basan exclusivamente en tu conocimiento de entrenamiento.
NO inventes números de registro digital ni rubros exactos de tesis.
Si necesitas citar jurisprudencia, descríbela por su contenido, no por datos específicos que podrías alucinar.
"""
        
        # Add attached document context if provided
        if request.attached_document:
            doc_text = request.attached_document[:50000]  # Cap at 50K chars
            system_instruction += f"""

═══════════════════════════════════════════════════════════════
   DOCUMENTO ADJUNTO DEL USUARIO
═══════════════════════════════════════════════════════════════

El secretario ha adjuntado el siguiente documento para referencia.
Usa este texto como base para continuar, modificar o mejorar según las instrucciones del usuario.

{doc_text}
"""
            print(f"   📎 Documento adjunto inyectado: {len(doc_text)} chars")
        
        # Build Gemini conversation
        gemini_contents = []
        for msg in request.messages:
            role = "model" if msg.role == "assistant" else "user"
            gemini_contents.append(
                gtypes.Content(
                    role=role,
                    parts=[gtypes.Part.from_text(text=msg.content)]
                )
            )
        
        # ── Streaming Generation ──────────────────────────────────────────
        client = genai.Client(api_key=gemini_key)
        
        async def generate_sentencia_stream():
            """SSE streaming from Gemini 2.5 Pro for sentencia chat."""
            try:
                content_buffer = ""
                
                response_stream = client.models.generate_content_stream(
                    model="gemini-2.5-pro",
                    contents=gemini_contents,
                    config=gtypes.GenerateContentConfig(
                        system_instruction=system_instruction,
                        temperature=0.7,
                        max_output_tokens=16384,
                    ),
                )
                
                for chunk in response_stream:
                    if chunk.text:
                        content_buffer += chunk.text
                        yield chunk.text
                
                print(f"   📝 Chat sentencia respuesta: {len(content_buffer)} chars")
                
                # Emit metadata if RAG was used
                if rag_context and search_results:
                    doc_id_map = build_doc_id_map(search_results)
                    if doc_id_map:
                        validation = validate_citations(content_buffer, doc_id_map)
                        sources_map = {}
                        for cv in validation.citations:
                            doc = doc_id_map.get(cv.doc_id)
                            if doc:
                                sources_map[cv.doc_id] = {
                                    "origen": humanize_origen(doc.origen) or "Fuente legal",
                                    "ref": doc.ref or "",
                                    "texto": (doc.texto or "")[:2000]
                                }
                        meta = json.dumps({
                            "valid": validation.valid_count,
                            "invalid": validation.invalid_count,
                            "total": validation.total_citations,
                            "sources": sources_map
                        })
                        yield f"\n\n<!-- CITATION_META:{meta} -->"
                
            except Exception as e:
                print(f"   ❌ Chat sentencia error: {e}")
                yield f"\n\n❌ Error: {str(e)}"
        
        return StreamingResponse(
            generate_sentencia_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
                "X-Model-Used": "gemini-2.5-pro",
                "X-RAG-Enabled": "true" if request.use_rag else "false",
                "X-RAG-Results": str(rag_count),
            },
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en chat sentencia: {str(e)}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
# ADMIN: One-time BM25 sparse vector re-ingestion
# ══════════════════════════════════════════════════════════════════════════════

class ReingestRequest(BaseModel):
    entidad: Optional[str] = None  # Filter by state, or None for all
    collection: str = "leyes_estatales"  # V5.0: accept any collection name
    admin_key: str  # Simple auth to prevent abuse

# ═══════════════════════════════════════════════════════════════════════════════
# REDACTOR DE SENTENCIAS FEDERALES — Gemini 2.5 Pro Multimodal
# ═══════════════════════════════════════════════════════════════════════════════

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
ADMIN_EMAILS = [e.strip().lower() for e in os.getenv("ADMIN_EMAILS", "").split(",") if e.strip()]

# ── Subscription-aware access check for Redactor ─────────────────────────────
def _can_access_sentencia(user_email: str) -> bool:
    """
    Check if a user can access the Redactor de Sentencias.
    Returns True if the user is an admin OR has ultra_secretarios subscription.
    """
    email_lower = user_email.strip().lower()

    # Fast path: admin list (no network call)
    if email_lower in ADMIN_EMAILS:
        return True

    # Supabase path: check subscription_type
    if supabase_admin:
        try:
            result = supabase_admin.table('user_profiles') \
                .select('subscription_type') \
                .eq('email', email_lower) \
                .limit(1) \
                .execute()
            if result.data and len(result.data) > 0:
                sub_type = result.data[0].get('subscription_type', '')
                if sub_type == 'ultra_secretarios':
                    print(f"   ✅ Acceso Redactor concedido: {email_lower} (suscripción {sub_type})")
                    return True
        except Exception as e:
            print(f"   ⚠️ Error checking subscription for {email_lower}: {e}")

    return False

GEMINI_MODEL = "gemini-2.5-pro"         # Critical reasoning (Step B, legacy 2C)
GEMINI_MODEL_FAST = "gemini-2.5-flash"  # Auxiliary tasks (extraction, enrichment, assembly)

# ── Document labels per sentence type ────────────────────────────────────────
SENTENCIA_DOC_LABELS: Dict[str, List[str]] = {
    "amparo_directo": ["Demanda de Amparo", "Acto Reclamado"],
    "amparo_revision": ["Recurso de Revisión", "Sentencia Recurrida"],
    "revision_fiscal": ["Recurso de Revisión Fiscal", "Sentencia Recurrida"],
    "recurso_queja": ["Recurso de Queja", "Determinación Recurrida"],
}

# ── Base system prompt (shared across all types) ─────────────────────────────
SENTENCIA_SYSTEM_BASE = """Eres un Secretario Proyectista de un Tribunal Colegiado de Circuito del Poder Judicial de la Federación de México. Tu función es redactar PROYECTOS DE SENTENCIA completos, listos para revisión del Magistrado Ponente.

REGLAS ABSOLUTAS:
1. Redacta en TERCERA PERSONA con el estilo formal judicial mexicano
2. Usa la estructura exacta: RESULTANDOS → CONSIDERANDOS → PUNTOS RESOLUTIVOS
3. Cita TEXTUALMENTE los argumentos de las partes usando "[…]" y comillas
4. Fundamenta CADA considerando en artículos específicos de ley y jurisprudencia aplicable
5. Usa la numeración: PRIMERO, SEGUNDO, TERCERO... (en letras, con punto final)
6. El encabezado debe incluir: tipo de asunto, número de expediente, quejoso/recurrente, magistrado ponente, secretario
7. La fecha de resolución debe ser en letras completas ("quince de enero de dos mil veintiséis")
8. Al citar jurisprudencia usa: rubro completo, sala/tribunal, número de tesis
9. Incluye notas al pie para fundamentación legal
10. El estilo debe ser IDÉNTICO al de un Tribunal Colegiado real: frases largas, subordinadas, lenguaje técnico-jurídico

ESTRUCTURA OBLIGATORIA:

RESULTANDOS:
- PRIMERO: Presentación de la demanda/recurso (quién, cuándo, ante quién, contra qué acto)
- SEGUNDO: Trámite (registro, admisión, notificaciones, informes justificados)
- TERCERO: Terceros interesados (si aplica)
- CUARTO: Turno a ponencia
- QUINTO: Integración del tribunal (si hay cambios)
- SEXTO/SÉPTIMO: Returno (si aplica)

CONSIDERANDOS:
- PRIMERO: Competencia del tribunal (con fundamento legal preciso)
- SEGUNDO: Existencia del acto reclamado (con referencia a constancias)
- TERCERO: Legitimación y oportunidad (plazos, personalidad)
- CUARTO: Procedencia / Fijación de la litis
- QUINTO en adelante: ESTUDIO DE FONDO (análisis de conceptos de violación / agravios)

PUNTOS RESOLUTIVOS:
- PRIMERO: Sentido del fallo (conceder/negar amparo, confirmar/revocar, fundada/infundada la queja)
- SEGUNDO en adelante: Según el tipo de asunto (efectos si aplican, notificaciones, archivación)
- Fórmula de cierre con votación y firmas

IMPORTANTE: Lee TODOS los documentos adjuntos minuciosamente. Extrae los datos del expediente, las partes, los hechos, los argumentos y los fundamentos directamente de los PDFs.
"""

# ── Type-specific prompts ────────────────────────────────────────────────────
SENTENCIA_PROMPTS: Dict[str, str] = {
    "amparo_directo": SENTENCIA_SYSTEM_BASE + """
TIPO ESPECÍFICO: AMPARO DIRECTO (Arts. 170-189 Ley de Amparo)

Documentos que recibirás:
1. DEMANDA DE AMPARO: Contiene los conceptos de violación, el acto reclamado señalado, las autoridades responsables, y los derechos humanos cuya violación se alega
2. ACTO RECLAMADO: Es la sentencia o laudo contra la que se promueve el amparo. Analízala en detalle para confrontar con los conceptos de violación

En el ESTUDIO DE FONDO:
- Analiza CADA concepto de violación individualmente
- Confronta cada argumento del quejoso contra lo resuelto en el acto reclamado
- Determina si los conceptos son fundados, infundados o inoperantes
- Si son fundados: explica por qué y señala efectos
- Cita jurisprudencia y tesis aplicables
- Aplica suplencia de la queja si procede conforme al artículo 79 de la Ley de Amparo

Sentidos posibles del fallo:
- CONCEDER el amparo (total o para efectos)
- NEGAR el amparo
- SOBRESEER (si hay causa de improcedencia)
""",

    "amparo_revision": SENTENCIA_SYSTEM_BASE + """
TIPO ESPECÍFICO: AMPARO EN REVISIÓN (Arts. 81-96 Ley de Amparo)

Documentos que recibirás:
1. RECURSO DE REVISIÓN: Contiene los agravios del recurrente contra la sentencia del Juzgado de Distrito
2. SENTENCIA RECURRIDA: Es la sentencia del amparo indirecto que se recurre

En el ESTUDIO DE FONDO:
- Analiza la procedencia del recurso (Arts. 81 y 83 Ley de Amparo)
- Examina CADA agravio individualmente
- Confronta con las consideraciones de la sentencia recurrida
- Determina si los agravios son fundados, infundados o inoperantes
- Analiza si hay materia de revisión oficiosa
- Verifica constitucionalidad de normas si se planteó

Sentidos posibles:
- CONFIRMAR la sentencia recurrida
- REVOCAR la sentencia recurrida
- MODIFICAR la sentencia
""",

    "revision_fiscal": SENTENCIA_SYSTEM_BASE + """
TIPO ESPECÍFICO: REVISIÓN FISCAL (Art. 63 Ley Federal de Procedimiento Contencioso Administrativo)

Documentos que recibirás:
1. RECURSO DE REVISIÓN FISCAL: Agravios del recurrente (generalmente autoridad hacendaria o IMSS)
2. SENTENCIA RECURRIDA: Sentencia del Tribunal Federal de Justicia Administrativa

En el ESTUDIO DE FONDO:
- Verifica PRIMERO la procedencia del recurso conforme al Art. 63 LFPCA (importancia y trascendencia o supuestos específicos)
- Si es procedente, analiza cada agravio
- Confronta agravios con las consideraciones del TFJA
- Aplica criterios de procedencia restrictiva de la revisión fiscal
- Considera la materia fiscal/administrativa

Sentidos posibles:
- CONFIRMAR la sentencia recurrida (la más común si no hay vicios)
- REVOCAR la sentencia
- DESECHAR por improcedente
""",

    "recurso_queja": SENTENCIA_SYSTEM_BASE + """
TIPO ESPECÍFICO: RECURSO DE QUEJA (Arts. 97-103 Ley de Amparo)

Documentos que recibirás:
1. RECURSO DE QUEJA: Agravios contra el auto o resolución recurrida
2. DETERMINACIÓN RECURRIDA: El auto o resolución del Juzgado de Distrito que se impugna

En el ESTUDIO DE FONDO:
- Identifica la fracción del artículo 97 de la Ley de Amparo aplicable
- Verifica oportunidad (plazo de 5 días, Art. 98 Ley de Amparo)
- Analiza cada agravio contra el auto recurrido
- Determina si los agravios logran desvirtuar las consideraciones del auto
- La queja es un recurso de estricto derecho (salvo excepciones del Art. 79)

Sentidos posibles:
- DECLARAR FUNDADA la queja (revocar el auto recurrido)
- DECLARAR INFUNDADA la queja (confirmar el auto)
- DESECHAR por improcedente o extemporánea
""",
}

# ── Secretary instructions addendum for system prompt ────────────────────────
INSTRUCCIONES_ADDENDUM = """

INSTRUCCIONES CRÍTICAS DEL SECRETARIO PROYECTISTA:
El secretario proyectista — experto en la materia — ha indicado el sentido
en que DEBE resolverse este asunto. DEBES seguir ESTRICTAMENTE sus instrucciones
respecto a:
- El sentido del fallo (conceder/negar amparo, confirmar/revocar, fundada/infundada la queja)
- La calificación de CADA concepto de violación o agravio (fundado, infundado, inoperante)
- Las razones por las que cada concepto/agravio se califica de esa manera

ESTRATEGIA DE FOCO TEMÁTICO:
- CONCENTRA el estudio extenso y profundo ÚNICAMENTE en los agravios/grupos
  calificados como FUNDADOS. Estos requieren análisis exhaustivo con fundamentación
  legal, jurisprudencial y constitucional completa (mínimo 2,000 palabras cada uno).
- Los agravios/grupos calificados como INFUNDADOS respóndelos con formato breve
  (~500-800 palabras), señalando que no satisfacen la carga argumentativa o que
  la autoridad responsable actuó conforme a derecho.
- Los agravios/grupos calificados como INOPERANTES respóndelos con formato
  formulaico (~300-500 palabras), usando expresiones como:
  "Es inoperante pues no combate las consideraciones que sustentan el fallo..."
  "Resulta inoperante al no controvertir los fundamentos y motivos..."
  "Se califica de inoperante al ser genérico e impreciso..."

Esto EVITA sentencias kilométricas concentrando la capacidad analítica donde importa.

El secretario NO necesita proporcionar todas las leyes o jurisprudencia — el sistema
ha consultado la base de datos legal y te proporciona fundamentación RAG adicional.
USA esa fundamentación para enriquecer y respaldar el sentido indicado.

Si se proporcionan artículos o tesis de jurisprudencia del RAG, cítalos textualmente
en los considerandos correspondientes.
"""


def _build_auto_mode_instructions(sentido: str, tipo: str, calificaciones: list) -> str:
    """Build synthetic instructions for auto-draft mode."""
    label_map = {
        "amparo_directo": "conceptos de violación",
        "amparo_revision": "agravios",
        "revision_fiscal": "agravios",
        "recurso_queja": "agravios",
    }
    agravio_label = label_map.get(tipo, "agravios")
    
    lines = [
        "MODO AUTOMÁTICO — BORRADOR A RAÍZ DE PRECEDENTES Y JURISPRUDENCIA",
        f"El secretario ha solicitado un borrador automático.",
        f"Sentido del fallo: {sentido.upper()}" if sentido else "Sentido: determinar con base en precedentes RAG.",
        "",
    ]
    
    if calificaciones:
        lines.append(f"Calificación de {agravio_label}:")
        for c in calificaciones:
            num = c.get("numero", "?")
            calif = c.get("calificacion", "sin_calificar").upper()
            titulo = c.get("titulo", "")
            disp = " [DISPOSITIVO]" if c.get("dispositivo") else ""
            lines.append(f"  - {agravio_label.capitalize()} {num}: {calif}{disp} — {titulo}")
        lines.append("")
    
    lines.extend([
        "INSTRUCCIÓN GLOBAL:",
        f"Centra el análisis profundo en los {agravio_label} calificados como FUNDADOS.",
        f"Los {agravio_label} INFUNDADOS/INOPERANTES respóndelos con formato breve.",
        "Basa toda la argumentación en los precedentes y jurisprudencia proporcionados por el RAG.",
        "NO inventes tesis ni jurisprudencia — usa EXCLUSIVAMENTE la que se te proporciona.",
    ])
    
    return "\n".join(lines)

# ═══════════════════════════════════════════════════════════════════════════════
# MULTI-PASS PHASE PROMPTS
# ═══════════════════════════════════════════════════════════════════════════════

# ── Phase 1: Structured Data Extraction from PDFs ─────────────────────────────
PHASE1_EXTRACTION_PROMPT = """Eres un asistente jurídico de alta precisión. Tu ÚNICA tarea es EXTRAER datos estructurados de los documentos del expediente judicial proporcionados.

REGLAS:
1. Extrae TEXTUALMENTE — no parafrasees, no resumas, no inventes datos
2. Usa "[…]" solo cuando un fragmento es demasiado largo (>500 palabras)
3. Si un dato no aparece en los documentos, indica "NO ENCONTRADO"
4. Los conceptos de violación / agravios deben ser extraídos COMPLETOS, no resumidos

Responde EXCLUSIVAMENTE con el siguiente JSON (sin markdown, sin ```json):

{
  "expediente": {
    "numero": "",
    "tipo_asunto": "",
    "tribunal": "",
    "circuito": ""
  },
  "partes": {
    "quejoso_recurrente": "",
    "tercero_interesado": "",
    "autoridades_responsables": [""],
    "magistrado_ponente": "",
    "secretario": ""
  },
  "fechas": {
    "presentacion_demanda": "",
    "admision": "",
    "sentencia_recurrida": "",
    "interposicion_recurso": "",
    "admision_recurso": "",
    "turno_ponencia": ""
  },
  "acto_reclamado": {
    "descripcion_completa": "",
    "autoridad_emisora": "",
    "fecha_emision": "",
    "resumen_contenido": ""
  },
  "tramite_procesal": {
    "historia_procesal_detallada": "",
    "terceros_interesados_datos": "",
    "cambios_integracion": "",
    "returno_datos": ""
  },
  "conceptos_violacion_o_agravios": [
    {
      "numero": 1,
      "titulo_o_tema": "",
      "texto_integro": "",
      "articulos_invocados": [""],
      "jurisprudencia_citada": [""],
      "pretension_concreta": ""
    }
  ],
  "sentencia_recurrida_resumen": {
    "sentido_fallo": "",
    "consideraciones_clave": "",
    "fundamentos_principales": ""
  },
  "datos_adicionales": {
    "materia": "",
    "via_procesal": "",
    "causas_improcedencia_opuestas": [""],
    "notas": ""
  }
}
"""

# ── Phase 2A: RESULTANDOS ────────────────────────────────────────────────────
PHASE2A_RESULTANDOS_PROMPT = """Eres un Secretario Proyectista de un Tribunal Colegiado de Circuito del Poder Judicial de la Federación de México.

Tu ÚNICA tarea ahora es redactar la sección de RESULTANDOS del proyecto de sentencia.

ESTILO OBLIGATORIO:
- TERCERA PERSONA, estilo formal judicial mexicano
- Frases largas con oraciones subordinadas, lenguaje técnico-jurídico
- Numeración: PRIMERO, SEGUNDO, TERCERO... (en letras, con punto final)
- Cada resultando debe ser un párrafo extenso y detallado
- Cita fechas en letras completas ("quince de enero de dos mil veintiséis")

CONTENIDO DE CADA RESULTANDO:
- PRIMERO: Presentación detallada de la demanda/recurso (quién, cuándo, ante quién, contra qué acto, en qué términos, con qué pretensiones)
- SEGUNDO: Trámite completo (registro, admisión, auto admisorio, notificaciones, informes justificados, pruebas ofrecidas, audiencia)
- TERCERO: Terceros interesados (identificación, emplazamiento, si comparecieron)
- CUARTO: Dictamen del Ministerio Público si aplica
- QUINTO: Turno a ponencia con datos específicos
- SEXTO: Returno si hubo cambios de integración (con fundamento)
- SÉPTIMO: Cualquier incidencia procesal adicional

SIGUE EXACTAMENTE este formato:

R E S U L T A N D O:

PRIMERO. [título descriptivo en negritas]. [contenido extenso del resultando]

SEGUNDO. [título descriptivo en negritas]. [contenido extenso]

[...continúa con todos los resultandos necesarios]

EXTENSIÓN ESPERADA: Mínimo 3,000 caracteres. Sé exhaustivo con los datos procesales.

USA los datos estructurados proporcionados. NO inventes datos que no aparezcan en la extracción.
"""

# ── Phase 2B: CONSIDERANDOS (Marco Legal) ────────────────────────────────────
PHASE2B_CONSIDERANDOS_PROMPT = """Eres un Secretario Proyectista de un Tribunal Colegiado de Circuito del Poder Judicial de la Federación de México.

Tu ÚNICA tarea ahora es redactar los CONSIDERANDOS PRELIMINARES (competencia, existencia del acto reclamado, legitimación, oportunidad y procedencia). NO redactes el estudio de fondo — eso se hará después.

ESTILO OBLIGATORIO:
- TERCERA PERSONA, estilo formal judicial mexicano
- Frases largas con oraciones subordinadas
- Fundamenta CADA considerando en artículos específicos de ley
- Cita jurisprudencia con rubro completo, sala/tribunal, número de tesis cuando estén disponibles del RAG
- Usa notas al pie para fundamentación legal

CONSIDERANDOS A REDACTAR:

PRIMERO. Competencia.
- Fundamento constitucional (Art. 107 CPEUM, fracción aplicable)
- Fundamento en Ley de Amparo (Arts. 81, 83, 92, según el tipo)
- Fundamento en Ley Orgánica del PJF (Art. 35 u otro aplicable)
- Explicar POR QUÉ este tribunal es competente (materia, territorio, cuantía)

SEGUNDO. Existencia del acto reclamado.
- Referencia a constancias que acreditan el acto
- Certificación del secretario del tribunal de origen
- Análisis completo de la existencia o inexistencia

TERCERO. Legitimación y oportunidad.
- Análisis de personalidad del promovente/recurrente
- Cómputo detallado de plazos (fecha de notificación, días hábiles/inhábiles, fecha de presentación)
- Fundamento legal del plazo aplicable

CUARTO. Procedencia / Fijación de la litis.
- Causas de improcedencia (si las hay, análisis de cada una)
- Definición precisa de la materia del estudio (qué se va a analizar)
- Si hay cuestiones firmes por falta de impugnación, señalarlas aquí

EXTENSIÓN ESPERADA: Mínimo 5,000 caracteres. Cada considerando debe ser extenso y rigurosamente fundamentado.

FORMATO:
C O N S I D E R A N D O:

PRIMERO. Competencia. [análisis extenso con fundamentación]

SEGUNDO. Existencia del acto reclamado. [análisis con referencias a constancias]

[...continúa]
"""

# ── Phase 2C: ESTUDIO DE FONDO — The critical section ────────────────────────
PHASE2C_ESTUDIO_FONDO_PROMPT = """Eres un Secretario Proyectista EXPERTO de un Tribunal Colegiado de Circuito del Poder Judicial de la Federación de México. Esta es la parte MÁS IMPORTANTE del proyecto de sentencia.

Tu tarea es redactar el ESTUDIO DE FONDO (Considerandos QUINTO en adelante) y los PUNTOS RESOLUTIVOS.

═══ REGLAS ABSOLUTAS PARA EL ESTUDIO DE FONDO ═══

1. ANALIZA CADA CONCEPTO DE VIOLACIÓN / AGRAVIO EN SU PROPIA SECCIÓN NUMERADA
   - Si hay 5 agravios, debe haber 5 sub-secciones de análisis
   - Cada análisis debe ser EXTENSO (mínimo 1,500 caracteres por agravio)

2. ESTRUCTURA DE CADA ANÁLISIS DE AGRAVIO/CONCEPTO:
   a) SÍNTESIS del argumento: Transcribe o sintetiza fielmente lo que dice el promovente
   b) REPRODUCCIÓN TEXTUAL de las partes clave usando comillas y "[…]"
   c) CONFRONTACIÓN: Analiza qué dijo la sentencia/acto reclamado sobre ese punto
   d) CALIFICACIÓN: Aplica la calificación indicada por el secretario (fundado/infundado/inoperante)
   e) RAZONAMIENTO: Explica con profundidad POR QUÉ se califica de esa manera
   f) FUNDAMENTACIÓN: Cita artículos de ley y jurisprudencia que respaldan la calificación
   g) CONCLUSIÓN del agravio

3. CITAS DE JURISPRUDENCIA:
   - Rubro COMPLETO en negritas
   - Sala/Tribunal emisor
   - Época
   - Registro digital
   - Transcripción relevante del criterio

4. LONGITUD: El estudio de fondo debe ser la sección MÁS EXTENSA de toda la sentencia.
   Mínimo 15,000 caracteres. Si hay múltiples agravios, puede llegar a 30,000+.

5. NO incluyas PUNTOS RESOLUTIVOS ni Efectos de la Sentencia — eso se genera en una fase posterior.
   Tu tarea termina con la CONCLUSIÓN del último agravio analizado.

6. ESTILO: Frases largas, subordinadas, lenguaje técnico-jurídico federal.
   Usa transiciones como "En efecto...", "Contrario a lo que sostiene...", "No le asiste razón...",
   "Resulta aplicable la jurisprudencia...", "En esa tesitura...", "De lo anterior se colige..."

=== REGLA ANTI-ALUCINACION PARA JURISPRUDENCIA (CRITICA — LEER 3 VECES) ===

PROHIBICION ABSOLUTA #1: NO INVENTES TESIS DE JURISPRUDENCIA.
Si una tesis, rubro, o registro digital NO aparece TEXTUALMENTE en las fuentes RAG
que te proporciono, NO LA CITES. Es preferible escribir argumentacion doctrinaria
o citar solo articulos de ley, a inventar una tesis falsa.

PROHIBICION ABSOLUTA #2: CADA REGISTRO DIGITAL ES UNICO.
Cada tesis del Semanario Judicial de la Federacion tiene un registro digital UNICO.
NUNCA uses el mismo numero de registro para dos tesis diferentes. Si no sabes el
registro exacto, NO lo incluyas — describe el criterio sin numero.

PROHIBICION ABSOLUTA #3: NO FILTRES MARCADORES INTERNOS AL OUTPUT.
Las etiquetas [JURISPRUDENCIA VERIFICADA], [LEGISLACION VERIFICADA] y
[EJEMPLO SENTENCIA] son marcadores INTERNOS del sistema. NUNCA los reproduzcas
en el texto de la sentencia. Cita la jurisprudencia con formato judicial estandar:
"Resulta aplicable la tesis [rubro], consultable con registro digital: [numero]"

QUE SI PUEDES HACER:
- Citar TEXTUALMENTE las tesis que aparecen en las fuentes RAG proporcionadas
- Argumentar con razonamiento juridico propio (analogias, interpretacion, doctrina)
- Citar articulos especificos de ley con su texto
- La CREATIVIDAD es bienvenida para argumentacion, pero NUNCA para fabricar fuentes

- Los fragmentos etiquetados como [EJEMPLO SENTENCIA] son solo REFERENCIA DE ESTILO
  y ARGUMENTACION. NO los cites como fuente. Usa su estructura y nivel de detalle como modelo.

FORMATO DE CITA DE JURISPRUDENCIA (cuando venga del RAG):
- Rubro completo entre comillas
- Tribunal emisor, Epoca, Registro digital
- Transcripcion relevante del criterio

═══ INSTRUCCIONES DEL SECRETARIO ═══
SIGUE ESTRICTAMENTE el sentido del fallo y la calificación de cada agravio/concepto
indicados por el secretario. El secretario es el experto en la materia.

═══ FORMATO ═══

IMPORTANTE: Tú redactas UN SOLO agravio/concepto a la vez (NO todo el estudio completo).
Tu texto será insertado dentro de un estudio de fondo más amplio.

NO incluyas encabezados como "QUINTO. Estudio de fondo." ni introducciones generales.
Comienza DIRECTAMENTE con el título y análisis del agravio asignado.

USA TITULOS ORDINALES: "Primer concepto de violación", "Segundo agravio", etc.
(NO uses "CONCEPTO DE VIOLACION 1" ni "AGRAVIO 1" con numerales arabigos).

En su primer agravio, el recurrente aduce que...
[análisis extenso con citas RAG]
[calificación y razonamiento]
[jurisprudencia del RAG citada con formato judicial]
[CONCLUSIÓN: Este agravio/concepto de violación resulta FUNDADO/INFUNDADO/INOPERANTE]
"""

# ── Phase 3: Structural Coherence + Polish & Assembly ─────────────────────────
PHASE3_POLISH_PROMPT = """Eres un Secretario Proyectista EXPERTO de un Tribunal Colegiado de Circuito del Poder Judicial de la Federación de México.

Se te proporcionan las diferentes secciones del proyecto de sentencia redactadas por separado.
Tu tarea es VERIFICAR LA COHERENCIA ESTRUCTURAL y luego ENSAMBLAR Y PULIR el documento final.

=== PASO 1: VERIFICACION DE COHERENCIA ESTRUCTURAL (CRITICO) ===

Antes de ensamblar, ANALIZA todo el texto recibido y corrige estos problemas:

1. DEDUPLICACION DE HEADERS:
   - El header "QUINTO. Estudio de fondo." (o cualquier considerando) debe aparecer UNA SOLA VEZ.
   - Si aparece repetido, ELIMINA las repeticiones y conserva solo la primera instancia.
   - Cada considerando (PRIMERO, SEGUNDO, TERCERO, CUARTO, QUINTO) debe ser UNICO.

2. DEDUPLICACION DE PARRAFOS INTRODUCTORIOS:
   - Si hay parrafos introductorios casi identicos (ej: "Previo al analisis de los conceptos...",
     "La litis se constrine a...", "La materia del presente juicio..."), conserva SOLO UNO.
   - Ese parrafo introductorio debe aparecer UNA VEZ al inicio del estudio de fondo,
     NO repetirse antes de cada agravio/concepto.

3. LIMPIEZA DE ERRORES TECNICOS:
   - ELIMINA cualquier texto que contenga errores tecnicos como:
     "[Error al redactar...]", "503 UNAVAILABLE", "request timed out",
     "status: 'UNAVAILABLE'", o cualquier traza de error de API.
   - Si un agravio quedo incompleto por error, senala con:
     "[Nota: El analisis de este agravio requiere complementacion]"

4. FLUJO DE AGRAVIOS/CONCEPTOS:
   - Los agravios/conceptos de violacion deben fluir como secciones DENTRO de un solo
     considerando (QUINTO), no como considerandos separados.
   - Cada agravio debe tener su titulo ("CONCEPTO DE VIOLACION 1", "AGRAVIO 2", etc.)
     seguido de su analisis, sin repetir la introduccion general.

5. PUNTOS RESOLUTIVOS:
   - Los PUNTOS RESOLUTIVOS deben aparecer UNA SOLA VEZ al final del documento.
   - Si estan duplicados, conserva la version mas completa y elimina las demas.

=== PASO 2: ENSAMBLAJE Y PULIDO ===

1. ENCABEZADO: Genera el encabezado oficial con tipo de asunto, numero de expediente,
   partes, magistrado ponente, secretario, lugar y fecha.
2. NUMERACION: Asegura numeracion continua y coherente (PRIMERO, SEGUNDO, TERCERO...)
3. TRANSICIONES: Anade transiciones fluidas entre secciones.
4. NOTAS AL PIE: Genera notas al pie numeradas para las referencias legales.
5. FORMULA DE CIERRE: Anade votacion, identidad de magistrados, firma del ponente,
   y "Notifiquese; con testimonio de esta resolucion, devuelvanse los autos..."
6. CONSISTENCIA: Verifica nombres, fechas y numeros en todo el documento.

=== REGLAS DE CONTENIDO ===
- CONSERVA toda la extension y profundidad del analisis juridico de cada agravio.
- NO recortes el analisis sustantivo (argumentos, jurisprudencia, razonamiento).
- SI elimina contenido DUPLICADO (intros repetidas, headers repetidos, resolutivos repetidos).
- SI elimina errores tecnicos y mensajes de sistema.
- Diferencia entre contenido sustantivo (CONSERVAR) y ruido estructural (ELIMINAR).

FORMATO DE SALIDA: Proyecto de sentencia COMPLETO, coherente, listo para el Magistrado Ponente.
NO uses formato markdown. Usa texto plano con convenciones judiciales:
- Titulos en MAYUSCULAS CON ESPACIOS (R E S U L T A N D O:)
- Negritas indicadas con **texto** solo para nombres y rubros de jurisprudencia
- Numerales en palabras (PRIMERO, SEGUNDO, etc.)
"""

# ═══════════════════════════════════════════════════════════════════════════════
# MULTI-PASS PHASE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

async def phase1_extract_data(client, pdf_parts: list, tipo: str) -> dict:
    """
    Phase 1: Extract structured data from 3 PDFs into JSON.
    Returns parsed dict or empty dict on failure.
    """
    from google.genai import types as gtypes
    import time

    print(f"\n   🔬 FASE 1: Extrayendo datos estructurados de los PDFs...")
    start = time.time()

    parts = list(pdf_parts)  # Copy PDF parts
    parts.append(gtypes.Part.from_text(
        text="\n\nExtrae TODOS los datos estructurados de los documentos anteriores siguiendo el formato JSON indicado en las instrucciones del sistema. "
             "Sé exhaustivo: extrae cada concepto de violación / agravio de forma ÍNTEGRA, no resumida."
    ))

    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL_FAST,  # Extraction task → Flash
            contents=parts,
            config=gtypes.GenerateContentConfig(
                system_instruction=PHASE1_EXTRACTION_PROMPT,
                temperature=0.1,
                max_output_tokens=32768,
            ),
        )
        raw = response.text or ""
        elapsed = time.time() - start
        print(f"   ✅ Fase 1 completada: {len(raw)} chars en {elapsed:.1f}s")

        # Clean markdown fences if present
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("```", 2)[1]
            if cleaned.startswith("json"):
                cleaned = cleaned[4:]
            # Find closing fence
            if "```" in cleaned:
                cleaned = cleaned[:cleaned.rfind("```")]

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as e:
            print(f"   ⚠️ Fase 1: JSON parse error: {e}")
            # Return raw text as fallback
            return {"_raw_extraction": raw}
    except Exception as e:
        print(f"   ❌ Fase 1 error: {e}")
        return {}


async def phase2a_draft_resultandos(client, extracted_data: dict, pdf_parts: list, tipo: str) -> str:
    """Phase 2A: Draft RESULTANDOS section."""
    from google.genai import types as gtypes
    import time

    print(f"\n   📜 FASE 2A: Redactando RESULTANDOS...")
    start = time.time()

    parts = list(pdf_parts)
    parts.append(gtypes.Part.from_text(
        text=f"\n\n═══ DATOS EXTRAÍDOS DEL EXPEDIENTE ═══\n{json.dumps(extracted_data, ensure_ascii=False, indent=2)}\n"
             f"\n═══ TIPO DE ASUNTO: {tipo} ═══\n"
             f"\nRedacta la sección completa de RESULTANDOS usando los datos extraídos y los documentos originales."
    ))

    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL_FAST,  # Resultandos → Flash
            contents=parts,
            config=gtypes.GenerateContentConfig(
                system_instruction=PHASE2A_RESULTANDOS_PROMPT,
                temperature=0.3,
                max_output_tokens=16384,
            ),
        )
        text = response.text or ""
        elapsed = time.time() - start
        print(f"   ✅ Fase 2A: {len(text)} chars en {elapsed:.1f}s")
        return text
    except Exception as e:
        print(f"   ❌ Fase 2A error: {e}")
        return ""


async def phase2b_draft_considerandos(client, extracted_data: dict, pdf_parts: list, tipo: str, rag_context: str) -> str:
    """Phase 2B: Draft preliminary CONSIDERANDOS (competencia, legitimación, etc.)."""
    from google.genai import types as gtypes
    import time

    print(f"\n   ⚖️ FASE 2B: Redactando CONSIDERANDOS preliminares...")
    start = time.time()

    parts = list(pdf_parts)
    parts.append(gtypes.Part.from_text(
        text=f"\n\n═══ DATOS EXTRAÍDOS DEL EXPEDIENTE ═══\n{json.dumps(extracted_data, ensure_ascii=False, indent=2)}\n"
    ))
    if rag_context:
        parts.append(gtypes.Part.from_text(
            text=f"\n═══ FUNDAMENTACIÓN RAG (jurisprudencia y legislación) ═══\n{rag_context}\n═══ FIN RAG ═══\n"
        ))
    parts.append(gtypes.Part.from_text(
        text=f"\n═══ TIPO DE ASUNTO: {tipo} ═══\n"
             f"\nRedacta los CONSIDERANDOS PRELIMINARES (competencia, existencia del acto, legitimación, oportunidad, procedencia)."
             f"\nNO redactes el estudio de fondo."
    ))

    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL_FAST,  # Considerandos → Flash
            contents=parts,
            config=gtypes.GenerateContentConfig(
                system_instruction=PHASE2B_CONSIDERANDOS_PROMPT,
                temperature=0.3,
                max_output_tokens=16384,
            ),
        )
        text = response.text or ""
        elapsed = time.time() - start
        print(f"   ✅ Fase 2B: {len(text)} chars en {elapsed:.1f}s")
        return text
    except Exception as e:
        print(f"   ❌ Fase 2B error: {e}")
        return ""


async def phase2c_draft_estudio_fondo(
    client, extracted_data: dict, pdf_parts: list,
    tipo: str, instrucciones: str, rag_context: str
) -> str:
    """Phase 2C: Draft ESTUDIO DE FONDO (the core analysis) + PUNTOS RESOLUTIVOS."""
    from google.genai import types as gtypes
    import time

    print(f"\n   🔍 FASE 2C: Redactando ESTUDIO DE FONDO (sección crítica)...")
    start = time.time()

    parts = list(pdf_parts)

    # Extracted data
    parts.append(gtypes.Part.from_text(
        text=f"\n\n═══ DATOS EXTRAÍDOS DEL EXPEDIENTE ═══\n{json.dumps(extracted_data, ensure_ascii=False, indent=2)}\n"
    ))

    # Secretary instructions
    if instrucciones.strip():
        parts.append(gtypes.Part.from_text(
            text=f"\n═══ INSTRUCCIONES DEL SECRETARIO PROYECTISTA ═══\n{instrucciones}\n═══ FIN INSTRUCCIONES ═══\n"
        ))

    # RAG context
    if rag_context:
        parts.append(gtypes.Part.from_text(
            text=f"\n═══ FUNDAMENTACIÓN RAG (jurisprudencia y legislación) ═══\n"
                 f"Usa estos artículos, tesis y jurisprudencia para fundamentar cada considerando.\n"
                 f"{rag_context}\n═══ FIN RAG ═══\n"
        ))

    # Type-specific additional instructions
    type_specific = SENTENCIA_PROMPTS.get(tipo, "")
    # Extract only the type-specific part (after SENTENCIA_SYSTEM_BASE)
    if type_specific.startswith(SENTENCIA_SYSTEM_BASE):
        type_specific = type_specific[len(SENTENCIA_SYSTEM_BASE):]

    parts.append(gtypes.Part.from_text(
        text=f"\n═══ INSTRUCCIONES ESPECÍFICAS DEL TIPO DE ASUNTO ═══\n{type_specific}\n"
             f"\nRedacta el ESTUDIO DE FONDO COMPLETO y los PUNTOS RESOLUTIVOS.\n"
             f"Analiza CADA concepto de violación / agravio en su propia sección.\n"
             f"El estudio de fondo debe ser la sección MÁS EXTENSA de toda la sentencia.\n"
             f"NO omitas ningún agravio. Cada uno requiere análisis profundo de mínimo 1,500 caracteres."
    ))

    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=parts,
            config=gtypes.GenerateContentConfig(
                system_instruction=PHASE2C_ESTUDIO_FONDO_PROMPT,
                temperature=0.3,
                max_output_tokens=65536,
            ),
        )
        text = response.text or ""
        elapsed = time.time() - start
        print(f"   ✅ Fase 2C: {len(text)} chars en {elapsed:.1f}s")
        return text
    except Exception as e:
        print(f"   ❌ Fase 2C error: {e}")
        return ""


def _strip_ai_preamble(text: str) -> str:
    """Remove common AI preamble/courtesy patterns from Gemini output."""
    lines = text.split('\n')
    clean_lines = []
    skip_until_content = True
    for line in lines:
        stripped = line.strip().lower()
        if skip_until_content:
            # Skip known preamble patterns
            if any(p in stripped for p in [
                'claro,', 'claro ', 'procedo a redactar', 'aquí está',
                'conforme a la técnica', 'en mi calidad de',
                'a continuación', 'por supuesto', 'con gusto',
                'aqui está', 'aqui esta',
                'apegándome estrictamente', 'apegandome estrictamente',
            ]):
                continue
            # Skip markdown separators at top (e.g. ***)
            if stripped in ('***', '---', '===', '* * *'):
                continue
            # Skip blank lines before content starts
            if not stripped:
                continue
            skip_until_content = False
        clean_lines.append(line)
    return '\n'.join(clean_lines)


def _sanitize_sentencia_output(text: str) -> str:
    """
    Post-generation sanitizer for sentencia output. Addresses hallucination patterns:
    1. Detects and warns about duplicate registro digital numbers
    2. Strips leaked internal markers ([JURISPRUDENCIA VERIFICADA], etc.)
    3. Appends verification disclaimer
    """
    import re

    # ── Step 1: Strip leaked internal markers ────────────────────────────────
    markers_to_remove = [
        r'\[JURISPRUDENCIA VERIFICADA\]',
        r'\[LEGISLACION VERIFICADA\]',
        r'\[LEGISLACIÓN VERIFICADA\]',
        r'\[EJEMPLO SENTENCIA\]',
        r'\[VERIFICAR\]',
        r'\[PENDIENTE DE VERIFICACIÓN\]',
        r'\[PENDIENTE DE VERIFICACION\]',
    ]
    for pattern in markers_to_remove:
        text = re.sub(pattern, '', text)

    # Clean up any double spaces left by marker removal
    text = re.sub(r'  +', ' ', text)
    # Clean up lines that are now just whitespace
    text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)

    # ── Step 2: Detect duplicate registro digital numbers ────────────────────
    # Match patterns like "Registro: 2025644", "registro digital: 2025644", "Registro digital: 2025644"
    registro_pattern = re.compile(
        r'[Rr]egistro(?:\s+[Dd]igital)?[:\s]+([0-9]{4,8})',
        re.IGNORECASE
    )
    registros_found = registro_pattern.findall(text)

    if registros_found:
        from collections import Counter
        registro_counts = Counter(registros_found)
        duplicates = {reg: count for reg, count in registro_counts.items() if count > 1}

        if duplicates:
            print(f"   ⚠️ ALERTA ANTI-ALUCINACIÓN: Registros digitales duplicados detectados:")
            for reg, count in duplicates.items():
                print(f"      Registro {reg} aparece {count} veces — posible alucinación")

            # Add inline warning after each duplicate occurrence (except the first)
            for reg, count in duplicates.items():
                # Find all occurrences and mark the 2nd+ with a warning
                occurrences = list(registro_pattern.finditer(text))
                reg_occurrences = [m for m in occurrences if m.group(1) == reg]
                for m in reg_occurrences[1:]:
                    # Insert warning after the matched text
                    warning = f" [⚠️ VERIFICAR: este registro digital aparece múltiples veces en el documento]"
                    insert_pos = m.end()
                    text = text[:insert_pos] + warning + text[insert_pos:]

    # ── Step 3: Append verification disclaimer ──────────────────────────────
    disclaimer = (
        "\n\n────────────────────────────────────────\n"
        "NOTA DE VERIFICACIÓN: Las citas de jurisprudencia incluidas en este "
        "proyecto fueron generadas con apoyo de inteligencia artificial. "
        "Se recomienda verificar cada registro digital en el Semanario Judicial "
        "de la Federación (https://sjf2.scjn.gob.mx) antes de su uso en actuaciones oficiales.\n"
        "────────────────────────────────────────\n"
    )
    text += disclaimer

    return text


def _group_agravios_by_theme(calificaciones: List[dict]) -> List[List[dict]]:
    """
    Group agravios with similar themes to share RAG context.
    
    Rules:
    - Only group agravios with the SAME calificación (fundado with fundado, etc.)
    - Similarity based on Jaccard overlap of key terms from título + resumen
    - Threshold: 0.3 (30% term overlap)
    - Dispositivo agravios are NEVER grouped (processed solo first)
    
    Returns list of groups. Each group is a list of calificación dicts.
    Groups are ordered: dispositivo agravios first, then by original order.
    """
    import re

    def extract_terms(text: str) -> set:
        """Extract meaningful terms from legal text (skip stopwords)."""
        stopwords = {
            'el', 'la', 'los', 'las', 'de', 'del', 'en', 'un', 'una', 'que',
            'por', 'con', 'para', 'al', 'se', 'su', 'no', 'es', 'y', 'o',
            'lo', 'como', 'más', 'a', 'este', 'esta', 'esto', 'sin', 'sobre',
            'entre', 'tiene', 'fue', 'ser', 'ha', 'sus', 'le', 'ya', 'son',
            'del', 'las', 'los', 'una', 'pero', 'si', 'ante', 'todo', 'nos',
        }
        text = text.lower()
        words = re.findall(r'[a-záéíóúñü]+', text)
        return {w for w in words if len(w) > 3 and w not in stopwords}

    def jaccard(a: set, b: set) -> float:
        if not a or not b:
            return 0.0
        return len(a & b) / len(a | b)

    n = len(calificaciones)
    if n <= 1:
        return [calificaciones] if calificaciones else []

    # Separate dispositivo agravios (process solo, first)
    dispositivo = [c for c in calificaciones if c.get("dispositivo")]
    regular = [c for c in calificaciones if not c.get("dispositivo")]

    # Build term sets for regular agravios
    term_sets = []
    for c in regular:
        text = f"{c.get('titulo', '')} {c.get('resumen', '')}"
        term_sets.append(extract_terms(text))

    # Union-Find grouping
    parent = list(range(len(regular)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    # Group by similarity + same calificación
    for i in range(len(regular)):
        for j in range(i + 1, len(regular)):
            if regular[i].get("calificacion") == regular[j].get("calificacion"):
                sim = jaccard(term_sets[i], term_sets[j])
                if sim >= 0.3:
                    union(i, j)

    # Collect groups
    from collections import defaultdict
    group_map = defaultdict(list)
    for i, c in enumerate(regular):
        group_map[find(i)].append(c)

    # Build final list: dispositivo agravios first (each solo), then themed groups
    groups = []
    for d in dispositivo:
        groups.append([d])
    for root in sorted(group_map.keys()):
        groups.append(group_map[root])

    # Log grouping
    if len(groups) < n:
        print(f"   🧩 Agrupación temática: {n} agravios → {len(groups)} grupos")
        for gi, g in enumerate(groups):
            nums = [c.get('numero', '?') for c in g]
            label = "DISPOSITIVO" if g[0].get('dispositivo') else g[0].get('calificacion', '?').upper()
            print(f"      Grupo {gi+1} ({label}): agravios {nums}")
    else:
        print(f"   📋 Sin agrupación temática ({n} agravios independientes)")

    return groups


async def phase2c_adaptive_estudio_fondo(
    client, extracted_data: dict, pdf_parts: list,
    tipo: str, calificaciones: List[dict], rag_context: str
) -> str:
    """
    Phase 2C DEEP PIPELINE: Draft ESTUDIO DE FONDO with maximum depth per agravio.

    OPTIMIZED with:
    - Thematic grouping: similar agravios share RAG context (Step A runs once per group)
    - Strategic early termination: if a dispositivo+fundado agravio completes,
      remaining agravios get a template paragraph and are skipped.

    Each agravio goes through 4 steps:
      A. Deep multi-silo RAG (4 targeted queries across ALL silos) — SHARED per group
      B. Gemini drafts extensive analysis with all RAG context
      C. Post-enrichment RAG (extract terms from draft -> second search)
      D. Gemini enriches citations in the draft

    Returns the complete estudio de fondo text.
    """
    from google.genai import types as gtypes
    import time

    print(f"\n   PIPELINE PROFUNDO: {len(calificaciones)} agravios x 4 pasos cada uno")
    total_start = time.time()

    # Thematic Grouping
    groups = _group_agravios_by_theme(calificaciones)

    agravio_texts = []
    total_rag_hits = 0
    early_terminated = False
    skipped_agravios = []

    # Determine agravio label based on tipo (used for all agravios)
    if tipo == "amparo_directo":
        agravio_label_base = "CONCEPTO DE VIOLACION"
    else:
        agravio_label_base = "AGRAVIO"

    for group_idx, group in enumerate(groups):
        if early_terminated:
            # All remaining agravios get the template paragraph
            for calif in group:
                num = calif.get("numero", "?")
                skipped_agravios.append(num)
                agravio_texts.append(
                    _build_early_termination_paragraph(
                        num, calif.get("titulo", ""),
                        agravio_label_base, tipo
                    )
                )
            continue

        group_nums = [c.get("numero", "?") for c in group]
        is_grouped = len(group) > 1
        print(f"\n      {'=' * 60}")
        if is_grouped:
            print(f"      GRUPO {group_idx+1}: agravios {group_nums} (RAG compartido)")
        else:
            print(f"      AGRAVIO {group_nums[0]}/{len(calificaciones)}: {group[0].get('calificacion', '?').upper()}")

        # ==========================================================
        # STEP A: Deep Multi-Silo RAG -- SHARED for the whole group
        # ==========================================================
        print(f"\n         Paso A: RAG profundo multi-silo {'(compartido)' if is_grouped else ''}...")
        step_a_start = time.time()

        # Build targeted queries from ALL agravios in the group
        rag_queries = []
        for calif in group:
            agravio_titulo = calif.get("titulo", "")
            agravio_resumen = calif.get("resumen", "")
            calificacion = calif.get("calificacion", "sin_calificar")
            notas = calif.get("notas", "")

            if agravio_titulo:
                rag_queries.append(f"{agravio_titulo} {calificacion}")
            if agravio_resumen:
                rag_queries.append(agravio_resumen[:300])
            if notas:
                rag_queries.append(notas[:300])

        # Add materia-based query
        materia = extracted_data.get("datos_adicionales", {}).get("materia", "")
        if materia and materia != "NO ENCONTRADO":
            first_titulo = group[0].get("titulo", "")
            rag_queries.append(f"jurisprudencia {materia} {first_titulo}")

        # Ensure at least 2 queries
        if len(rag_queries) < 2:
            rag_queries.append(f"agravio {group[0].get('calificacion', '')} tribunal colegiado")

        # Deduplicate and limit queries (avoid too many for large groups)
        seen_q = set()
        unique_queries = []
        for q in rag_queries:
            q_key = q.strip().lower()[:50]
            if q_key not in seen_q:
                seen_q.add(q_key)
                unique_queries.append(q)
        rag_queries = unique_queries[:6]  # Max 6 queries per group

        # Search ALL standard silos
        all_rag_results = []
        seen_ids = set()

        standard_tasks = []
        for q in rag_queries:
            standard_tasks.append(hybrid_search_all_silos(
                query=q,
                estado=None,
                top_k=8,
                alpha=0.7,
                enable_reasoning=False,
            ))

        standard_results = await asyncio.gather(*standard_tasks, return_exceptions=True)
        for batch in standard_results:
            if isinstance(batch, Exception):
                continue
            for r in batch:
                if r.id not in seen_ids:
                    seen_ids.add(r.id)
                    all_rag_results.append(r)

        # Search sentencia silos
        sentencia_silos = []
        if tipo in SENTENCIA_SILOS:
            sentencia_silos.append(SENTENCIA_SILOS[tipo])
        for silo_tipo, silo_name in SENTENCIA_SILOS.items():
            if silo_tipo != tipo and silo_name not in sentencia_silos:
                sentencia_silos.append(silo_name)

        sentencia_tasks = []
        for q in rag_queries[:2]:
            for silo_name in sentencia_silos:
                sentencia_tasks.append(
                    _search_single_silo_for_sentencia(q, silo_name)
                )

        sentencia_results_raw = await asyncio.gather(*sentencia_tasks, return_exceptions=True)
        sentencia_results = []
        for batch in sentencia_results_raw:
            if isinstance(batch, Exception):
                continue
            for r in batch:
                if r.id not in seen_ids:
                    seen_ids.add(r.id)
                    sentencia_results.append(r)

        # Sort and compile RAG context
        all_rag_results.sort(key=lambda r: r.score, reverse=True)
        top_standard = all_rag_results[:25]
        sentencia_results.sort(key=lambda r: r.score, reverse=True)
        top_sentencias = sentencia_results[:8]
        total_rag_hits += len(top_standard) + len(top_sentencias)

        # Build shared RAG text
        shared_agravio_rag_text = ""
        for r in top_standard:
            source = r.ref or r.origen or ""
            text_content = r.texto or ""
            silo = r.silo or ""
            tag = "[JURISPRUDENCIA VERIFICADA]" if "jurisprudencia" in silo.lower() else "[LEGISLACION VERIFICADA]"
            shared_agravio_rag_text += f"\n--- {tag} ---\n"
            if source:
                shared_agravio_rag_text += f"Fuente: {source}\n"
            shared_agravio_rag_text += f"{text_content}\n"

        for r in top_sentencias:
            source = r.ref or r.origen or ""
            text_content = r.texto or ""
            shared_agravio_rag_text += f"\n--- [EJEMPLO SENTENCIA] ---\n"
            if source:
                shared_agravio_rag_text += f"Expediente: {source}\n"
            shared_agravio_rag_text += f"{text_content}\n"

        step_a_elapsed = time.time() - step_a_start
        print(f"         RAG: {len(top_standard)} juridicos + {len(top_sentencias)} sentencias ({step_a_elapsed:.1f}s)")

        # ==========================================================
        # STEPS B-D: Per agravio within the group (using shared RAG)
        # ==========================================================
        for calif in group:
            if early_terminated:
                num = calif.get("numero", "?")
                skipped_agravios.append(num)
                agravio_texts.append(
                    _build_early_termination_paragraph(
                        num, calif.get("titulo", ""),
                        agravio_label_base, tipo
                    )
                )
                continue

            num = calif.get("numero", "?")
            calificacion = calif.get("calificacion", "sin_calificar")
            notas = calif.get("notas", "")
            agravio_titulo = calif.get("titulo", "")
            agravio_resumen = calif.get("resumen", "")
            is_dispositivo = calif.get("dispositivo", False)

            if is_grouped:
                print(f"\n         AGRAVIO {num} ({calificacion.upper()}) dentro del grupo")
            agravio_start = time.time()

            # -- STEP B: Gemini Drafts Agravio --
            print(f"         Paso B: Gemini redacta agravio {num}...")
            step_b_start = time.time()

            parts_b = list(pdf_parts)

            # Extracted data
            parts_b.append(gtypes.Part.from_text(
                text=f"\n\n=== DATOS EXTRAIDOS DEL EXPEDIENTE ===\n{json.dumps(extracted_data, ensure_ascii=False, indent=2)}\n"
            ))

            # Secretary calificacion
            agravio_label = f"{agravio_label_base} {num}"
            calif_instruction = f"""
=== CALIFICACION DEL SECRETARIO PARA {agravio_label} ===
Titulo: {agravio_titulo}
Resumen: {agravio_resumen}
Calificacion: {calificacion.upper()}
"""
            if notas:
                calif_instruction += f"Fundamentos y motivos del secretario: {notas}\n"
            if is_dispositivo:
                calif_instruction += f"AGRAVIO DISPOSITIVO: Este agravio es considerado por el secretario como DETERMINANTE para resolver todo el caso.\n"
            calif_instruction += f"""
DEBES calificar este agravio como {calificacion.upper()}. El secretario proyectista
es el experto en la materia y su calificacion es VINCULANTE.
Los fundamentos y motivos del secretario DEBEN guiar tu argumentacion.
=== FIN CALIFICACION ===
"""
            parts_b.append(gtypes.Part.from_text(text=calif_instruction))

            # Full shared RAG context
            if shared_agravio_rag_text:
                parts_b.append(gtypes.Part.from_text(
                    text=f"\n=== FUNDAMENTACION RAG -- {agravio_label} ===\n"
                         f"UTILIZA estos articulos, tesis y jurisprudencia para fundamentar tu analisis.\n"
                         f"Las fuentes con [JURISPRUDENCIA VERIFICADA] y [LEGISLACION VERIFICADA] son CITAS REALES de la base de datos.\n"
                         f"Las fuentes con [EJEMPLO SENTENCIA] son referencia de ESTILO -- NO las cites como fuente.\n"
                         f"{shared_agravio_rag_text}\n=== FIN RAG ===\n"
                ))

            # Also include global RAG context (truncated)
            if rag_context:
                parts_b.append(gtypes.Part.from_text(
                    text=f"\n=== CONTEXTO RAG GENERAL (referencia adicional) ===\n{rag_context[:8000]}\n=== FIN RAG GENERAL ===\n"
                ))

            # Detailed drafting instructions
            type_specific = SENTENCIA_PROMPTS.get(tipo, "")
            if type_specific.startswith(SENTENCIA_SYSTEM_BASE):
                type_specific = type_specific[len(SENTENCIA_SYSTEM_BASE):]

            parts_b.append(gtypes.Part.from_text(
                text=f"\n=== INSTRUCCIONES DE REDACCION ===\n{type_specific}\n"
                     f"\nRedacta UNICAMENTE el analisis del {agravio_label} ({agravio_titulo}).\n"
                     f"Este agravio ha sido calificado como: {calificacion.upper()}\n\n"
                     f"REGLAS CRITICAS DE FORMATO:\n"
                     f"- NO incluyas encabezado 'QUINTO. Estudio de fondo.' ni ningun encabezado de considerando.\n"
                     f"- NO incluyas parrafo introductorio general del estudio. Eso ya existe.\n"
                     f"- Comienza DIRECTAMENTE con el titulo del agravio/concepto, por ejemplo:\n"
                     f"  '{agravio_label}. {agravio_titulo}'\n"
                     f"- Tu texto es UN FRAGMENTO que sera insertado dentro de un estudio de fondo mas amplio.\n"
                     f"- NO repitas informacion de otros agravios -- concentrate solo en ESTE.\n\n"
                     f"Tu analisis DEBE ser MUY EXTENSO -- minimo 3,000 caracteres.\n"
                     f"DEBES citar jurisprudencia de las fuentes RAG proporcionadas.\n"
                     f"DEBES citar articulos de ley de las fuentes RAG proporcionadas.\n"
                     f"PROHIBIDO ABSOLUTO: NO inventes tesis que no esten en el RAG.\n"
                     f"Si necesitas mas jurisprudencia de la que el RAG proporciona, usa argumentacion doctrinaria.\n"
                     f"NUNCA reutilices un mismo numero de registro digital para tesis distintas.\n"
                     f"NUNCA escribas etiquetas [JURISPRUDENCIA VERIFICADA] o [LEGISLACION VERIFICADA] en el texto.\n\n"
                     f"Estructura OBLIGATORIA del analisis:\n"
                     f"a) Sintesis fiel del agravio (transcripcion parcial con comillas)\n"
                     f"b) Marco juridico aplicable (articulos de ley citados con texto)\n"
                     f"c) Analisis del acto reclamado / sentencia recurrida\n"
                     f"d) Confrontacion punto por punto del agravio vs. acto reclamado\n"
                     f"e) Fundamentacion con jurisprudencia VERIFICADA (rubro, tribunal, epoca, registro)\n"
                     f"f) Razonamiento logico-juridico extenso y propio del tribunal\n"
                     f"g) CONCLUSION: Este agravio resulta {calificacion.upper()}\n"
            ))

            try:
                step_b_model = GEMINI_MODEL if calificacion == "fundado" else GEMINI_MODEL_FAST
                print(f"         Modelo: {step_b_model} ({'Pro -- razonamiento profundo' if calificacion == 'fundado' else 'Flash -- patron formulaico'})")

                response_b = client.models.generate_content(
                    model=step_b_model,
                    contents=parts_b,
                    config=gtypes.GenerateContentConfig(
                        system_instruction=PHASE2C_ESTUDIO_FONDO_PROMPT,
                        temperature=0.3,
                        max_output_tokens=32768,
                    ),
                )
                draft_text = _strip_ai_preamble(response_b.text or "")
                step_b_elapsed = time.time() - step_b_start
                print(f"         Borrador: {len(draft_text)} chars ({step_b_elapsed:.1f}s)")
            except Exception as e:
                print(f"         Paso B error: {e}")
                agravio_texts.append(f"\n[Error al redactar agravio {num}: {str(e)}]\n")
                continue

            # -- STEP C: Post-Enrichment RAG --
            print(f"         Paso C: RAG de enriquecimiento post-redaccion...")
            step_c_start = time.time()

            enrichment_rag = ""
            try:
                enrichment_prompt = f"""Extrae de este borrador de analisis juridico entre 3 y 5 consultas de busqueda 
para encontrar jurisprudencia y articulos de ley adicionales que refuercen la argumentacion.
Enfocate en:
- Tesis o jurisprudencia mencionadas que necesiten verificacion
- Articulos de ley citados que podrian complementarse
- Conceptos juridicos que beneficiarian de citas adicionales

Borrador:
{draft_text[:4000]}

Genera SOLO las consultas, una por linea, sin numeracion. Cada consulta debe ser 5-15 palabras."""

                enrichment_response = openai_client.chat.completions.create(
                    model=CHAT_MODEL,
                    messages=[{"role": "user", "content": enrichment_prompt}],
                    temperature=0.2,
                    max_tokens=300,
                )
                enrichment_queries = [
                    q.strip() for q in (enrichment_response.choices[0].message.content or "").split("\n")
                    if q.strip() and len(q.strip()) > 5
                ][:4]

                if enrichment_queries:
                    print(f"         {len(enrichment_queries)} queries de enriquecimiento")

                    enrich_tasks = []
                    for eq in enrichment_queries:
                        enrich_tasks.append(hybrid_search_all_silos(
                            query=eq, estado=None, top_k=6, alpha=0.7, enable_reasoning=False
                        ))

                    enrich_results_raw = await asyncio.gather(*enrich_tasks, return_exceptions=True)
                    enrich_results = []
                    enrich_seen = set(seen_ids)

                    for batch in enrich_results_raw:
                        if isinstance(batch, Exception):
                            continue
                        for r in batch:
                            if r.id not in enrich_seen:
                                enrich_seen.add(r.id)
                                enrich_results.append(r)

                    enrich_results.sort(key=lambda r: r.score, reverse=True)
                    top_enrich = enrich_results[:15]
                    total_rag_hits += len(top_enrich)

                    if top_enrich:
                        enrichment_rag = "\n"
                        for r in top_enrich:
                            source = r.ref or r.origen or ""
                            text_content = r.texto or ""
                            silo = r.silo or ""
                            tag = "[JURISPRUDENCIA VERIFICADA]" if "jurisprudencia" in silo.lower() else "[LEGISLACION VERIFICADA]"
                            enrichment_rag += f"\n--- {tag} ---\n"
                            if source:
                                enrichment_rag += f"Fuente: {source}\n"
                            enrichment_rag += f"{text_content}\n"

                        print(f"         Enriquecimiento: {len(top_enrich)} fuentes adicionales")

            except Exception as e:
                print(f"         Paso C error (continuando sin enriquecimiento): {e}")

            step_c_elapsed = time.time() - step_c_start
            print(f"         Paso C: {step_c_elapsed:.1f}s")

            # -- STEP D: Gemini Enriches Citations --
            if enrichment_rag:
                print(f"         Paso D: Gemini enriquece citas...")
                step_d_start = time.time()

                enrichment_instruction = f"""Tu tarea es MEJORAR el siguiente borrador de analisis juridico incorporando las NUEVAS 
fuentes verificadas que te proporciono. 

REGLAS:
1. CONSERVA todo el contenido y estructura del borrador original
2. ANADE citas de jurisprudencia y articulos de ley de las NUEVAS fuentes donde sean relevantes
3. REFUERZA los argumentos existentes con las nuevas fuentes
4. NUNCA elimines contenido del borrador original
5. SOLO usa fuentes etiquetadas como [JURISPRUDENCIA VERIFICADA] o [LEGISLACION VERIFICADA]
6. NO inventes citas -- solo usa las proporcionadas
7. El resultado debe ser MAS EXTENSO que el borrador (no mas corto)
8. NUNCA incluyas las etiquetas [JURISPRUDENCIA VERIFICADA] o [LEGISLACION VERIFICADA] en el texto de salida -- son marcadores internos

=== BORRADOR ORIGINAL ===
{draft_text}

=== NUEVAS FUENTES DE ENRIQUECIMIENTO ===
{enrichment_rag}

=== INSTRUCCION ===
Devuelve el borrador ENRIQUECIDO con las nuevas fuentes integradas naturalmente en la argumentacion.
"""

                try:
                    response_d = client.models.generate_content(
                        model=GEMINI_MODEL_FAST,
                        contents=[gtypes.Part.from_text(text=enrichment_instruction)],
                        config=gtypes.GenerateContentConfig(
                            system_instruction="Eres un Secretario Proyectista EXPERTO. Tu tarea es enriquecer un borrador juridico con nuevas fuentes verificadas sin perder contenido.",
                            temperature=0.2,
                            max_output_tokens=32768,
                        ),
                    )
                    enriched_text = response_d.text or ""
                    step_d_elapsed = time.time() - step_d_start

                    if len(enriched_text) >= len(draft_text) * 0.8:
                        print(f"         Enriquecido: {len(draft_text)} -> {len(enriched_text)} chars (+{len(enriched_text) - len(draft_text)}) ({step_d_elapsed:.1f}s)")
                        agravio_texts.append(enriched_text)
                    else:
                        print(f"         Enriquecido demasiado corto ({len(enriched_text)} vs {len(draft_text)}), usando borrador original")
                        agravio_texts.append(draft_text)

                except Exception as e:
                    print(f"         Paso D error (usando borrador): {e}")
                    agravio_texts.append(draft_text)
            else:
                print(f"         Sin fuentes adicionales, usando borrador directo")
                agravio_texts.append(draft_text)

            agravio_elapsed = time.time() - agravio_start
            print(f"      Agravio {num} completo: {len(agravio_texts[-1])} chars en {agravio_elapsed:.1f}s")

            # -- Check for early termination --
            if is_dispositivo and calificacion == "fundado":
                print(f"\n      TERMINACION ANTICIPADA: Agravio {num} es DISPOSITIVO y FUNDADO")
                print(f"         Los agravios restantes recibiran parrafo de omision estandar")
                early_terminated = True

    total_elapsed = time.time() - total_start
    print(f"\n   {'=' * 60}")
    print(f"   PIPELINE PROFUNDO COMPLETADO")
    print(f"   {len(agravio_texts)} agravios procesados")
    if skipped_agravios:
        print(f"   {len(skipped_agravios)} agravios omitidos por terminacion anticipada: {skipped_agravios}")
    print(f"   {total_rag_hits} resultados RAG utilizados")
    print(f"   {total_elapsed:.1f}s total")
    print(f"   {'=' * 60}")

    # Combine all agravios into a single, cohesive estudio de fondo
    if tipo == "amparo_directo":
        intro_label_plural = "conceptos de violacion"
        intro_label_singular = "concepto de violacion"
    else:
        intro_label_plural = "agravios"
        intro_label_singular = "agravio"

    # Extract quejoso/recurrente name from extracted data for the intro
    quejoso_name = extracted_data.get("quejoso_recurrente", extracted_data.get("partes", {}).get("quejoso", "la parte quejosa"))
    if isinstance(quejoso_name, list):
        quejoso_name = quejoso_name[0] if quejoso_name else "la parte quejosa"

    num_agravios = len(calificaciones)
    header = f"QUINTO. Estudio de fondo.\n\n"
    header += (
        f"Una vez demostrados los requisitos de procedencia, este Tribunal Colegiado "
        f"procede al analisis de los {num_agravios} {intro_label_plural} formulados por "
        f"{quejoso_name}, los cuales se estudiaran de manera individual, confrontando "
        f"cada {intro_label_singular} con las consideraciones del acto reclamado, "
        f"el marco juridico aplicable y la jurisprudencia pertinente.\n"
    )

    combined = header + "\n\n" + "\n\n".join(agravio_texts)
    return combined


def _build_early_termination_paragraph(
    num, titulo: str, agravio_label_base: str, tipo: str
) -> str:
    """Build standard paragraph for agravios skipped due to early termination."""
    agravio_label = f"{agravio_label_base} {num}"
    
    if tipo == "amparo_directo":
        recurso_text = "el amparo solicitado"
    elif tipo == "amparo_revision":
        recurso_text = "el recurso de revision"
    elif tipo == "revision_fiscal":
        recurso_text = "el recurso de revision fiscal"
    elif tipo == "recurso_queja":
        recurso_text = "el recurso de queja"
    else:
        recurso_text = "el recurso interpuesto"
    
    return (
        f"{agravio_label}. {titulo}\n\n"
        f"Habida cuenta de que en el analisis precedente se determino fundado "
        f"un agravio cuyo efecto es suficiente para resolver {recurso_text} en su "
        f"integridad, resulta innecesario el estudio del presente agravio, de "
        f"conformidad con el criterio sustentado por la Segunda Sala de la Suprema "
        f"Corte de Justicia de la Nacion, en la jurisprudencia de rubro: "
        f"\"AGRAVIOS EN LA REVISION. CUANDO SU ESTUDIO ES INNECESARIO.\", "
        f"toda vez que, aun en el supuesto de que resultaran fundados, en nada "
        f"variaria el sentido de la presente resolucion.\n"
    )


def _determine_sentido(calificaciones: List[dict], tipo: str = "") -> str:
    """Determine the overall fallo sentido based on calificaciones and tipo."""
    fundados = sum(1 for c in calificaciones if c.get("calificacion") == "fundado")
    total = len(calificaciones)

    tipo_lower = tipo.lower() if tipo else ""

    if "amparo" in tipo_lower:
        if fundados == 0:
            return "Se NIEGA el amparo solicitado."
        elif fundados == total:
            return "Se CONCEDE el amparo solicitado al quejoso."
        else:
            return f"Se CONCEDE parcialmente el amparo al resultar FUNDADOS {fundados} de {total} agravios."
    elif "revision" in tipo_lower or "recurso" in tipo_lower:
        if fundados == 0:
            return "Se CONFIRMA la resolución recurrida al resultar INFUNDADOS los agravios expuestos."
        elif fundados == total:
            return "Se REVOCA la resolución recurrida al resultar FUNDADOS los agravios planteados."
        else:
            return f"Se MODIFICA la resolución recurrida al resultar FUNDADOS {fundados} de {total} agravios."
    else:
        if fundados == 0:
            return "Se declara INFUNDADO el medio de impugnación."
        elif fundados == total:
            return "Se declara FUNDADO el medio de impugnación."
        else:
            return f"Se declara PARCIALMENTE FUNDADO el medio de impugnación ({fundados} de {total} agravios fundados)."


async def phase_final_efectos_resolutivos(
    client, extracted_data: dict, estudio_fondo: str,
    tipo: str, calificaciones: List[dict]
) -> str:
    """
    Final Phase: Draft PUNTOS RESOLUTIVOS (and EFECTOS only when applicable).
    
    Structure rules based on analysis of real sentencias:
    - recurso_queja: NO Efectos → direct RESUELVE (FUNDADA/INFUNDADA)
    - amparo_directo concede: Efectos de la concesión → RESUELVE
    - amparo_directo niega: NO Efectos → direct RESUELVE (NIEGA)
    - revision_fiscal: NO Efectos → direct RESUELVE (CONFIRMA/REVOCA)
    - amparo_revision: NO Efectos → direct RESUELVE with numbered points
    """
    from google.genai import types as gtypes
    import time

    print(f"\n   ⚖️ FASE FINAL: Puntos Resolutivos (tipo: {tipo})...")
    start = time.time()

    sentido = _determine_sentido(calificaciones, tipo)
    has_fundados = any(c.get("calificacion") == "fundado" for c in calificaciones)

    # Build calificaciones summary
    calif_summary = "\n".join([
        f"   Agravio {c.get('numero', i+1)}: {c.get('calificacion', 'sin_calificar').upper()}"
        for i, c in enumerate(calificaciones)
    ])

    # ── Tipo-specific instructions for structure ──────────────────────────
    if tipo == "recurso_queja":
        structure_instructions = """ESTRUCTURA PARA RECURSO DE QUEJA:
- NO incluyas sección "Efectos de la Sentencia" — los recursos de queja NO la tienen
- Ve DIRECTO a los PUNTOS RESOLUTIVOS con la fórmula:
  "Por lo expuesto y fundado, se resuelve:"
- Si la queja es FUNDADA: "ÚNICO. Se declara FUNDADA la queja [...], en consecuencia,
  se revoca el auto de [fecha] dictado por [juzgado] en el juicio de amparo [número],
  y se ordena [lo que proceda]."
- Si la queja es INFUNDADA: "ÚNICO. Se declara INFUNDADA la queja [...], en consecuencia,
  se confirma el auto de [fecha] dictado por [juzgado] en el juicio de amparo [número]."
- Fórmula de cierre: votación, firmas, "Notifíquese; devuélvanse los autos..."
- Mínimo 800 caracteres."""
    elif tipo == "amparo_directo" and has_fundados:
        structure_instructions = """ESTRUCTURA PARA AMPARO DIRECTO (CONCEDE):
- Incluye sección "Efectos de la concesión" ANTES de los puntos resolutivos:
  - Para qué efectos se concede el amparo
  - Qué debe hacer la autoridad responsable (dejar insubsistente, reponer, dictar nueva)
  - Plazos aplicables
- PUNTOS RESOLUTIVOS:
  PRIMERO. La Justicia de la Unión AMPARA Y PROTEGE a [quejoso]...
  SEGUNDO. [Efectos específicos de la concesión]
  TERCERO. Notifíquese...
- Fórmula de cierre: votación, firmas
- Mínimo 2,000 caracteres."""
    elif tipo == "amparo_directo" and not has_fundados:
        structure_instructions = """ESTRUCTURA PARA AMPARO DIRECTO (NIEGA):
- NO incluyas sección "Efectos de la Sentencia" — cuando se niega el amparo NO hay efectos
- Ve DIRECTO a los PUNTOS RESOLUTIVOS:
  "Por lo expuesto y fundado, se resuelve:"
  PRIMERO. La Justicia de la Unión NO AMPARA NI PROTEGE a [quejoso]...
  SEGUNDO. Notifíquese...
- Fórmula de cierre: votación, firmas
- Mínimo 800 caracteres."""
    elif tipo == "revision_fiscal":
        structure_instructions = """ESTRUCTURA PARA REVISIÓN FISCAL:
- NO incluyas sección "Efectos de la Sentencia" — las revisiones fiscales NO la tienen
- Ve DIRECTO a los PUNTOS RESOLUTIVOS:
  "Por lo expuesto y fundado, se resuelve:"
- Si CONFIRMA: "ÚNICO. Se CONFIRMA la sentencia de [fecha] dictada por [sala del TFJA]..."
- Si REVOCA: "PRIMERO. Se REVOCA la sentencia... SEGUNDO. [Nueva resolución]..."
- Si DESECHA: "ÚNICO. Se DESECHA el recurso de revisión fiscal..."
- Fórmula de cierre: votación, firmas, "Notifíquese; devuélvanse los autos..."
- Mínimo 800 caracteres."""
    elif tipo == "amparo_revision":
        structure_instructions = """ESTRUCTURA PARA AMPARO EN REVISIÓN:
- NO incluyas sección "Efectos de la Sentencia" separada
- Ve DIRECTO a los PUNTOS RESOLUTIVOS con puntos NUMERADOS:
  "Por lo expuesto, fundado y con apoyo en los artículos [...], se resuelve:"
  PRIMERO. Se [CONFIRMA/REVOCA/MODIFICA] la sentencia de [fecha]...
  SEGUNDO. La Justicia de la Unión [AMPARA Y PROTEGE / NO AMPARA NI PROTEGE]...
  TERCERO. [Si hay revisión adhesiva: queda sin materia / se desecha]
  CUARTO. Notifíquese...
- Fórmula de cierre: votación, firmas
- Mínimo 1,200 caracteres."""
    else:
        structure_instructions = """PUNTOS RESOLUTIVOS directos. NO incluyas Efectos de la Sentencia.
- Mínimo 800 caracteres."""

    prompt_text = f"""Eres un Secretario Proyectista EXPERTO de un Tribunal Colegiado de Circuito.

Tu tarea es redactar ÚNICAMENTE los PUNTOS RESOLUTIVOS (y Efectos SOLO si las instrucciones lo indican).

═══ TIPO DE ASUNTO: {tipo} ═══

═══ DATOS DEL EXPEDIENTE ═══
{json.dumps(extracted_data, ensure_ascii=False, indent=2)}

═══ CALIFICACIONES DE LOS AGRAVIOS ═══
{calif_summary}

═══ SENTIDO DEL FALLO ═══
{sentido}

═══ ESTUDIO DE FONDO (referencia para coherencia) ═══
{estudio_fondo[:15000]}

═══ INSTRUCCIONES DE ESTRUCTURA ═══
{structure_instructions}

═══ REGLAS GENERALES ═══
1. Usa lenguaje jurídico formal, preciso.
2. Incluye la fórmula de cierre con votación y firmas.
3. "Notifíquese; con testimonio de esta resolución, devuélvanse los autos al lugar de su origen..."
4. "Así, por unanimidad/mayoría de votos lo resolvió el [Tribunal]. Firma el Magistrado [Ponente]
    como ponente, con el Secretario [Secretario] que autoriza y da fe."
"""

    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL_FAST,  # Resolutivos → Flash
            contents=[gtypes.Part.from_text(text=prompt_text)],
            config=gtypes.GenerateContentConfig(
                system_instruction="Eres un Secretario Proyectista EXPERTO de un Tribunal Colegiado de Circuito del Poder Judicial de la Federación de México. Redacta con máximo rigor formal los puntos resolutivos de una sentencia, siguiendo ESTRICTAMENTE las instrucciones de estructura proporcionadas.",
                temperature=0.2,
                max_output_tokens=16384,
            ),
        )
        text = _strip_ai_preamble(response.text or "")
        elapsed = time.time() - start
        print(f"   ✅ Resolutivos: {len(text)} chars en {elapsed:.1f}s")
        return text
    except Exception as e:
        print(f"   ❌ Fase Final error: {e}")
        # Fallback: minimal template per tipo
        if tipo == "recurso_queja":
            return f"""
Por lo expuesto y fundado, se resuelve:

ÚNICO. {sentido}

Notifíquese; con testimonio de esta resolución, devuélvanse los autos al lugar de su origen y, en su oportunidad, archívese el presente toca como asunto concluido.

Así lo resolvió el Tribunal Colegiado de Circuito. Firma el Magistrado ponente con el Secretario que autoriza y da fe.
"""
        else:
            return f"""
Por lo expuesto y fundado, se resuelve:

PRIMERO. {sentido}

SEGUNDO. Notifíquese personalmente a las partes.

TERCERO. En su oportunidad, archívese el expediente como asunto concluido.

Así lo resolvió el Tribunal Colegiado de Circuito. Firma el Magistrado ponente con el Secretario que autoriza y da fe.
"""

async def phase3_polish_assembly(client, extracted_data: dict, resultandos: str, considerandos: str, estudio_fondo: str, tipo: str) -> str:
    """Phase 3: Polish and assemble the final sentencia document."""
    from google.genai import types as gtypes
    import time

    print(f"\n   ✨ FASE 3: Ensamblando y puliendo la sentencia final...")
    start = time.time()

    assembly_text = (
        f"═══ DATOS DEL EXPEDIENTE ═══\n"
        f"{json.dumps(extracted_data, ensure_ascii=False, indent=2)}\n\n"
        f"═══ TIPO DE ASUNTO: {tipo} ═══\n\n"
        f"═══ SECCIÓN 1: RESULTANDOS ═══\n{resultandos}\n\n"
        f"═══ SECCIÓN 2: CONSIDERANDOS PRELIMINARES ═══\n{considerandos}\n\n"
        f"═══ SECCIÓN 3: ESTUDIO DE FONDO Y PUNTOS RESOLUTIVOS ═══\n{estudio_fondo}\n"
    )

    parts = [gtypes.Part.from_text(text=assembly_text)]
    parts.append(gtypes.Part.from_text(
        text="\n\nVERIFICA LA COHERENCIA ESTRUCTURAL del documento (deduplica headers, "
             "elimina intros repetidas, limpia errores técnicos, unifica resolutivos). "
             "Luego ENSAMBLA en un proyecto de sentencia COMPLETO y coherente. "
             "Añade encabezado oficial, notas al pie, y fórmula de cierre. "
             "CONSERVA toda la extensión del análisis jurídico, pero ELIMINA duplicados y errores."
    ))

    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL_FAST,  # 2.5 Flash with Thinking for structural coherence
            contents=parts,
            config=gtypes.GenerateContentConfig(
                system_instruction=PHASE3_POLISH_PROMPT,
                temperature=0.1,
                thinking_config=gtypes.ThinkingConfig(
                    thinking_budget=8192,
                ),
                max_output_tokens=65536,
            ),
        )
        text = response.text or ""
        elapsed = time.time() - start
        print(f"   ✅ Fase 3 (Thinking): {len(text)} chars en {elapsed:.1f}s")
        return text
    except Exception as e:
        print(f"   ❌ Fase 3 error: {e}")
        # Fallback: concatenate sections directly
        return f"{resultandos}\n\n{considerandos}\n\n{estudio_fondo}"


# ═══════════════════════════════════════════════════════════════════════════════
# ENHANCED RAG — More queries, more results, secondary extraction queries
# ═══════════════════════════════════════════════════════════════════════════════

async def extract_rag_queries_from_extraction(extracted_data: dict) -> List[str]:
    """
    Generate additional RAG queries from Phase 1 extracted data.
    Looks for article references, legal concepts, and jurisprudence cited by parties.
    """
    queries = []
    if not extracted_data or "_raw_extraction" in extracted_data:
        return queries

    # Extract article references
    for concepto in extracted_data.get("conceptos_violacion_o_agravios", []):
        for art in concepto.get("articulos_invocados", []):
            if art and art != "NO ENCONTRADO":
                queries.append(art)
        for juris in concepto.get("jurisprudencia_citada", []):
            if juris and juris != "NO ENCONTRADO":
                queries.append(juris[:100])  # Truncate long rubros
        # Add topic as query
        titulo = concepto.get("titulo_o_tema", "")
        if titulo and titulo != "NO ENCONTRADO":
            queries.append(titulo)

    # Add data about the case type
    materia = extracted_data.get("datos_adicionales", {}).get("materia", "")
    if materia and materia != "NO ENCONTRADO":
        queries.append(f"jurisprudencia {materia}")

    # Deduplicate and limit
    seen = set()
    unique = []
    for q in queries:
        q_lower = q.strip().lower()
        if q_lower not in seen and len(q_lower) > 5:
            seen.add(q_lower)
            unique.append(q.strip())
    return unique[:8]


# ── RAG query extraction from secretary instructions ─────────────────────────
async def extract_rag_queries_from_instructions(instrucciones: str, tipo: str) -> List[str]:
    """
    Uses GPT-5-mini to extract 3-5 concise legal search queries from the
    secretary's instructions. These queries will be used to search Qdrant.
    """
    if not instrucciones.strip() or not OPENAI_API_KEY:
        return []

    try:
        extraction_prompt = f"""Eres un asistente jurídico. A partir de las instrucciones de un secretario
proyectista de un Tribunal Colegiado de Circuito, extrae entre 3 y 5 consultas
concisas de búsqueda legal para encontrar jurisprudencia y artículos de ley relevantes.

Tipo de asunto: {tipo}

Instrucciones del secretario:
{instrucciones}

Genera SOLO las consultas de búsqueda, una por línea, sin numeración ni explicaciones.
Cada consulta debe ser concisa (5-15 palabras) y enfocada en un concepto jurídico específico.
Incluye artículos de ley mencionados, principios invocados, y temas de jurisprudencia relevantes."""

        response = openai_client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[{"role": "user", "content": extraction_prompt}],
            temperature=0.2,
            max_tokens=500,
        )
        raw = response.choices[0].message.content or ""
        queries = [q.strip() for q in raw.strip().split("\n") if q.strip() and len(q.strip()) > 5]
        return queries[:5]  # Max 5 queries
    except Exception as e:
        print(f"   ⚠️ Error extrayendo queries RAG: {e}")
        return []


async def _search_single_silo_for_sentencia(query: str, silo_name: str) -> List[SearchResult]:
    """Search a single sentencia example silo for relevant chunks."""
    try:
        dense_vector = await get_dense_embedding(query)
        sparse_vector = get_sparse_embedding(query)
        results = await hybrid_search_single_silo(
            collection=silo_name,
            query=query,
            dense_vector=dense_vector,
            sparse_vector=sparse_vector,
            filter_=None,
            top_k=5,
            alpha=0.7,
        )
        return results
    except Exception as e:
        print(f"   ⚠️ Error searching sentencia silo '{silo_name}': {e}")
        return []


async def run_rag_for_sentencia(instrucciones: str, tipo: str, secondary_queries: List[str] = None) -> str:
    """
    Enhanced RAG pipeline for sentencia drafting:
    1. Extract search queries from secretary's instructions
    2. Merge with secondary queries from Phase 1 extraction
    3. Search all Qdrant silos in parallel (including sentencia examples)
    4. Compile context string (max ~25000 chars) with tagged sources
    """
    queries = await extract_rag_queries_from_instructions(instrucciones, tipo)

    # Merge secondary queries from Phase 1 extraction
    if secondary_queries:
        seen = set(q.strip().lower() for q in queries)
        for sq in secondary_queries:
            if sq.strip().lower() not in seen:
                queries.append(sq)
                seen.add(sq.strip().lower())

    if not queries:
        print("   ℹ️ No RAG queries extracted — skipping RAG")
        return ""

    # Limit to 8 queries max
    queries = queries[:8]

    print(f"   🔍 RAG: {len(queries)} queries extraídas:")
    for i, q in enumerate(queries):
        print(f"      {i+1}. {q}")

    # ═══════════════════════════════════════════════════════════════
    # SEARCH 1: Standard silos (jurisprudencia, leyes, constitucional)
    # ═══════════════════════════════════════════════════════════════
    all_results = []
    seen_ids = set()

    tasks = []
    for q in queries:
        tasks.append(hybrid_search_all_silos(
            query=q,
            estado=None,  # Federal-level search (no state filter)
            top_k=8,
            alpha=0.7,
            enable_reasoning=False,  # Fast mode for speed
        ))

    results_per_query = await asyncio.gather(*tasks, return_exceptions=True)

    for query_results in results_per_query:
        if isinstance(query_results, Exception):
            print(f"   ⚠️ RAG search error: {query_results}")
            continue
        for r in query_results:
            if r.id not in seen_ids:
                seen_ids.add(r.id)
                all_results.append(r)

    # ═══════════════════════════════════════════════════════════════
    # SEARCH 2: Sentencia example silos (cross-silo for style/patterns)
    # ═══════════════════════════════════════════════════════════════
    sentencia_results = []
    sentencia_silos_to_search = []

    # Primary: the silo matching the current sentencia type
    if tipo in SENTENCIA_SILOS:
        sentencia_silos_to_search.append(SENTENCIA_SILOS[tipo])

    # Cross-silo: also search other types for richer argumentation patterns
    for silo_tipo, silo_name in SENTENCIA_SILOS.items():
        if silo_tipo != tipo and silo_name not in sentencia_silos_to_search:
            sentencia_silos_to_search.append(silo_name)

    if sentencia_silos_to_search:
        print(f"   📚 Buscando en {len(sentencia_silos_to_search)} silos de sentencias ejemplo...")

        # Use top 3 queries for sentencia search (avoid over-querying)
        sentencia_queries = queries[:3]
        sentencia_tasks = []

        for q in sentencia_queries:
            for silo_name in sentencia_silos_to_search:
                sentencia_tasks.append(
                    _search_single_silo_for_sentencia(q, silo_name)
                )

        sentencia_results_raw = await asyncio.gather(*sentencia_tasks, return_exceptions=True)

        for sr_batch in sentencia_results_raw:
            if isinstance(sr_batch, Exception):
                continue
            for r in sr_batch:
                if r.id not in seen_ids:
                    seen_ids.add(r.id)
                    sentencia_results.append(r)

        if sentencia_results:
            print(f"   ✅ Sentencias ejemplo: {len(sentencia_results)} resultados")

    if not all_results and not sentencia_results:
        print("   ℹ️ No RAG results found")
        return ""

    # Sort by score descending and take top 30 from standard, top 10 from sentencias
    all_results.sort(key=lambda r: r.score, reverse=True)
    top_standard = all_results[:30]

    sentencia_results.sort(key=lambda r: r.score, reverse=True)
    top_sentencias = sentencia_results[:10]

    total_results = len(top_standard) + len(top_sentencias)
    print(f"   ✅ RAG total: {total_results} resultados únicos "
          f"({len(top_standard)} jurídicos + {len(top_sentencias)} sentencias ejemplo)")

    # Compile context string with tagged sources
    context_parts = []
    total_chars = 0
    MAX_CHARS = 25000  # ~6000 tokens

    # First: standard results (jurisprudencia, leyes)
    for r in top_standard:
        text = r.texto or ""
        collection = r.silo or ""
        ref = r.ref or ""

        # Tag by source type
        if "jurisprudencia" in collection.lower():
            tag = "JURISPRUDENCIA VERIFICADA"
        elif "constitucional" in collection.lower():
            tag = "BLOQUE CONSTITUCIONAL"
        else:
            tag = "LEGISLACIÓN"

        entry = f"\n--- [{tag}] [{collection}] {ref}"
        entry += f" (relevancia: {r.score:.2f}) ---\n"
        entry += text[:1200]

        if total_chars + len(entry) > MAX_CHARS:
            break
        context_parts.append(entry)
        total_chars += len(entry)

    # Then: sentencia examples (for style/argumentation patterns)
    SENTENCIA_MAX_CHARS = 8000  # Reserve up to 8K for sentencia examples
    sentencia_chars = 0

    if top_sentencias and total_chars < MAX_CHARS:
        context_parts.append("\n\n═══ EJEMPLOS DE SENTENCIAS REALES (solo referencia de estilo y argumentación, NO citar como fuente) ═══\n")
        total_chars += 120

        for r in top_sentencias:
            text = r.texto or ""
            ref = r.ref or ""
            collection = r.silo or ""

            entry = f"\n--- [EJEMPLO SENTENCIA] [{collection}] {ref} ---\n"
            entry += text[:1500]

            if sentencia_chars + len(entry) > SENTENCIA_MAX_CHARS:
                break
            if total_chars + len(entry) > MAX_CHARS + SENTENCIA_MAX_CHARS:
                break
            context_parts.append(entry)
            total_chars += len(entry)
            sentencia_chars += len(entry)

    return "\n".join(context_parts)


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 0.5 — Análisis Inteligente Pre-Redacción
# ═══════════════════════════════════════════════════════════════════════════════

PHASE05_ANALYSIS_PROMPT = """Eres un asistente jurídico de alta precisión que analiza expedientes judiciales.
Tu tarea es leer los documentos del expediente y producir un ANÁLISIS ESTRUCTURADO en JSON.

REGLAS ESTRICTAS:
1. Extrae TEXTUALMENTE los agravios / conceptos de violación — NO resumas, NO parafrasees
2. El campo "resumen" de cada agravio debe ser una síntesis de 2-3 oraciones
3. El campo "texto_integro" debe contener el texto COMPLETO del agravio tal como aparece en el documento
4. Si un agravio es muy extenso (>2000 palabras), usa "[…]" para la parte central pero conserva inicio y final
5. Identifica TODOS los agravios — no omitas ninguno
6. Para el "titulo" usa la temática principal del agravio (ej: "Indebida valoración de pruebas")
7. AGRUPA los agravios por tema en "grupos_tematicos" — agravios que comparten la misma temática jurídica deben ir juntos

Responde EXCLUSIVAMENTE con JSON válido (sin markdown, sin ```json):

{
  "resumen_caso": "Síntesis del caso en 3-5 oraciones. Qué se reclama, en qué vía, quién reclama.",
  "resumen_acto_reclamado": "Descripción del acto o sentencia que se impugna. Qué resolvió la autoridad responsable.",
  "datos_expediente": {
    "numero": "",
    "tipo_asunto": "",
    "quejoso_recurrente": "",
    "autoridades_responsables": [""],
    "materia": "",
    "tribunal": ""
  },
  "agravios": [
    {
      "numero": 1,
      "titulo": "Título temático corto del agravio",
      "resumen": "Síntesis de 2-3 oraciones sobre qué argumenta el agravio",
      "texto_integro": "Texto COMPLETO del agravio tal como aparece en el documento",
      "articulos_mencionados": ["Art. 14 CPEUM", "Art. 16 CPEUM"],
      "derechos_invocados": ["Debido proceso", "Legalidad"]
    }
  ],
  "grupos_tematicos": [
    {
      "tema": "Nombre del tema que agrupa estos agravios (ej: Valoración de pruebas)",
      "agravios_nums": [1, 3],
      "descripcion": "Breve explicación de por qué se agrupan estos agravios"
    }
  ],
  "observaciones_preliminares": "Notas sobre posibles problemas de procedencia, causales de improcedencia visibles, o cuestiones de oficio que el tribunal debe abordar"
}
"""


# ── Pydantic models for Phase 0.5 ────────────────────────────────────────────
class AgravioAnalysis(BaseModel):
    numero: int
    titulo: str
    resumen: str
    texto_integro: str = ""
    articulos_mencionados: List[str] = []
    derechos_invocados: List[str] = []

class GrupoTematico(BaseModel):
    tema: str
    agravios_nums: List[int]
    descripcion: str = ""

class DatosExpediente(BaseModel):
    numero: str = ""
    tipo_asunto: str = ""
    quejoso_recurrente: str = ""
    autoridades_responsables: List[str] = []
    materia: str = ""
    tribunal: str = ""

class AnalysisResponse(BaseModel):
    resumen_caso: str = ""
    resumen_acto_reclamado: str = ""
    datos_expediente: DatosExpediente = DatosExpediente()
    agravios: List[AgravioAnalysis] = []
    grupos_tematicos: List[GrupoTematico] = []
    observaciones_preliminares: str = ""
    analysis_time_seconds: float = 0.0

class CalificacionAgravio(BaseModel):
    numero: int
    calificacion: Literal["fundado", "infundado", "inoperante", "sin_calificar"] = "sin_calificar"
    notas: str = ""


async def phase05_analyze_expediente(client, pdf_parts: list, tipo: str) -> dict:
    """
    Phase 0.5: Analyze the expediente and extract a structured summary
    with individual agravios/conceptos de violación.
    Returns parsed dict with agravios, or raises an Exception on failure.
    Includes retry logic for transient failures.
    """
    from google.genai import types as gtypes
    import time

    MAX_RETRIES = 2
    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        print(f"\n   🔎 FASE 0.5 (intento {attempt}/{MAX_RETRIES}): Analizando expediente...")
        start = time.time()

        parts = list(pdf_parts)
        parts.append(gtypes.Part.from_text(
            text="\n\nAnaliza TODOS los documentos anteriores y extrae la información estructurada "
                 "según el formato JSON indicado en las instrucciones del sistema. "
                 "Identifica CADA agravio / concepto de violación individualmente. "
                 "Sé exhaustivo: no omitas ningún agravio."
        ))

        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL_FAST,
                contents=parts,
                config=gtypes.GenerateContentConfig(
                    system_instruction=PHASE05_ANALYSIS_PROMPT,
                    temperature=0.1,
                    max_output_tokens=32768,
                ),
            )

            # Check for blocked/empty responses
            raw = response.text or ""
            elapsed = time.time() - start

            if not raw.strip():
                # Try to get finish reason for diagnostics
                finish_reason = "unknown"
                try:
                    if response.candidates and len(response.candidates) > 0:
                        finish_reason = str(response.candidates[0].finish_reason)
                except Exception:
                    pass
                last_error = f"Gemini returned empty response (finish_reason={finish_reason}, elapsed={elapsed:.1f}s)"
                print(f"   ⚠️ Fase 0.5: Respuesta vacía (finish_reason={finish_reason}) — {elapsed:.1f}s")
                if attempt < MAX_RETRIES:
                    print(f"   🔄 Reintentando en 3 segundos...")
                    import asyncio
                    await asyncio.sleep(3)
                    continue
                else:
                    break

            print(f"   ✅ Fase 0.5 completada: {len(raw)} chars en {elapsed:.1f}s")

            # Clean markdown fences if present
            cleaned = raw.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("```", 2)[1]
                if cleaned.startswith("json"):
                    cleaned = cleaned[4:]
                if "```" in cleaned:
                    cleaned = cleaned[:cleaned.rfind("```")]

            try:
                result = json.loads(cleaned)
            except json.JSONDecodeError as e:
                last_error = f"JSON parse error: {e} — raw_len={len(raw)}"
                print(f"   ⚠️ Fase 0.5: JSON parse error: {e}")
                print(f"      Primeros 500 chars: {raw[:500]}")
                if attempt < MAX_RETRIES:
                    print(f"   🔄 Reintentando en 3 segundos...")
                    import asyncio
                    await asyncio.sleep(3)
                    continue
                else:
                    break

            # Validate we actually got agravios
            agravios = result.get("agravios", [])
            if not agravios:
                last_error = f"No agravios found in parsed JSON (keys: {list(result.keys())})"
                print(f"   ⚠️ Fase 0.5: JSON válido pero sin agravios. Keys: {list(result.keys())}")
                if attempt < MAX_RETRIES:
                    print(f"   🔄 Reintentando en 3 segundos...")
                    import asyncio
                    await asyncio.sleep(3)
                    continue
                else:
                    break

            # Success!
            print(f"   📋 {len(agravios)} agravios identificados")
            for a in agravios:
                print(f"      {a.get('numero', '?')}. {a.get('titulo', 'Sin título')}")
            result["_analysis_time"] = round(elapsed, 1)
            return result

        except Exception as e:
            elapsed = time.time() - start
            error_str = str(e)
            last_error = f"{error_str} (elapsed={elapsed:.1f}s)"
            print(f"   ❌ Fase 0.5 error (intento {attempt}): {error_str}")

            # Check for quota/rate limit errors
            is_retryable = any(kw in error_str.lower() for kw in [
                "resource_exhausted", "rate_limit", "quota", "429",
                "503", "unavailable", "deadline", "timeout"
            ])

            if is_retryable and attempt < MAX_RETRIES:
                wait_time = 5 * attempt  # exponential backoff
                print(f"   🔄 Error transitorio, reintentando en {wait_time}s...")
                import asyncio
                await asyncio.sleep(wait_time)
                continue
            else:
                break

    # All retries exhausted
    print(f"   ❌ Fase 0.5 fallida tras {MAX_RETRIES} intentos: {last_error}")
    raise Exception(f"No se pudo analizar el expediente: {last_error}")


# ── Pydantic model for the response ──────────────────────────────────────────
class DraftSentenciaRequest(BaseModel):
    """Query model used when tipo is passed as JSON (not form)."""
    tipo: Literal["amparo_directo", "amparo_revision", "revision_fiscal", "recurso_queja"]

class DraftSentenciaResponse(BaseModel):
    sentencia_text: str
    tipo: str
    tokens_input: Optional[int] = None
    tokens_output: Optional[int] = None
    model: str = GEMINI_MODEL
    rag_results_count: int = 0
    phases_completed: int = 0
    total_chars: int = 0
    generation_time_seconds: float = 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# ENDPOINT: PHASE 0.5 — Análisis Pre-Redacción
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/analyze-expediente")
async def analyze_expediente(
    tipo: str = Form(...),
    user_email: str = Form(...),
    doc1: UploadFile = File(...),
    doc2: UploadFile = File(...),
    doc3: Optional[UploadFile] = File(None),
):
    """
    Phase 0.5: Analyze the expediente before drafting.
    Returns a structured analysis with case summary and individual agravios.
    The secretary can then qualify each agravio before proceeding to draft.
    """
    import time as time_module
    total_start = time_module.time()

    # ── Access validation (admin OR ultra_secretarios) ────────────────────
    if not GEMINI_API_KEY:
        raise HTTPException(500, "Gemini API key not configured")
    if not _can_access_sentencia(user_email):
        raise HTTPException(403, "Acceso restringido — se requiere suscripción Ultra Secretarios")

    # ── Validate tipo ────────────────────────────────────────────────────
    valid_types = list(SENTENCIA_PROMPTS.keys())
    if tipo not in valid_types:
        raise HTTPException(400, f"Tipo inválido. Opciones: {valid_types}")

    # ── Read PDF files ───────────────────────────────────────────────────
    doc_labels = SENTENCIA_DOC_LABELS[tipo]
    pdf_data = []
    doc_files = [doc1, doc2] + ([doc3] if doc3 else [])
    for i, (doc_file, label) in enumerate(zip(doc_files, doc_labels)):
        data = await doc_file.read()
        size_mb = len(data) / (1024 * 1024)
        if size_mb > 50:
            raise HTTPException(400, f"Archivo '{label}' excede 50MB ({size_mb:.1f}MB)")
        if not data:
            raise HTTPException(400, f"Archivo '{label}' está vacío")
        pdf_data.append((data, label, doc_file.filename or f"doc{i+1}.pdf"))
        print(f"   📄 {label}: {doc_file.filename} ({size_mb:.1f} MB)")

    # ── Build Gemini client and PDF parts ─────────────────────────────────
    try:
        from google import genai
        from google.genai import types as gtypes

        client = genai.Client(api_key=GEMINI_API_KEY)

        pdf_parts = []
        for pdf_bytes, label, filename in pdf_data:
            pdf_parts.append(gtypes.Part.from_text(text=f"\n--- DOCUMENTO: {label} (archivo: {filename}) ---\n"))
            pdf_parts.append(gtypes.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf"))

        print(f"\n🔎 ANÁLISIS PRE-REDACCIÓN — Tipo: {tipo}")

        # ── Run Phase 0.5 analysis ─────────────────────────────────────
        try:
            analysis_data = await phase05_analyze_expediente(client, pdf_parts, tipo)
        except Exception as phase05_err:
            error_msg = str(phase05_err)
            print(f"   ❌ Phase 0.5 failed: {error_msg}")

            # Provide user-friendly error messages
            if any(kw in error_msg.lower() for kw in ["resource_exhausted", "quota", "rate_limit", "429"]):
                raise HTTPException(
                    429,
                    "Límite de uso de la API Gemini alcanzado. Por favor espera unos minutos e intenta de nuevo."
                )
            else:
                raise HTTPException(
                    500,
                    f"Error al analizar el expediente: {error_msg}. Intenta de nuevo."
                )

        total_elapsed = time_module.time() - total_start

        # Build response
        agravios_list = []
        for a in analysis_data.get("agravios", []):
            agravios_list.append(AgravioAnalysis(
                numero=a.get("numero", 0),
                titulo=a.get("titulo", "Sin título"),
                resumen=a.get("resumen", ""),
                texto_integro=a.get("texto_integro", ""),
                articulos_mencionados=a.get("articulos_mencionados", []),
                derechos_invocados=a.get("derechos_invocados", []),
            ))

        datos = analysis_data.get("datos_expediente", {})
        datos_exp = DatosExpediente(
            numero=datos.get("numero", ""),
            tipo_asunto=datos.get("tipo_asunto", ""),
            quejoso_recurrente=datos.get("quejoso_recurrente", ""),
            autoridades_responsables=datos.get("autoridades_responsables", []),
            materia=datos.get("materia", ""),
            tribunal=datos.get("tribunal", ""),
        )

        print(f"\n   ✅ ANÁLISIS COMPLETADO en {total_elapsed:.1f}s — {len(agravios_list)} agravios")

        # Build thematic groups
        grupos_list = []
        for g in analysis_data.get("grupos_tematicos", []):
            grupos_list.append(GrupoTematico(
                tema=g.get("tema", "Sin tema"),
                agravios_nums=g.get("agravios_nums", []),
                descripcion=g.get("descripcion", ""),
            ))

        # If no groups from Gemini, create one group per agravio
        if not grupos_list and agravios_list:
            grupos_list = [GrupoTematico(
                tema=a.titulo,
                agravios_nums=[a.numero],
                descripcion=a.resumen,
            ) for a in agravios_list]

        return AnalysisResponse(
            resumen_caso=analysis_data.get("resumen_caso", ""),
            resumen_acto_reclamado=analysis_data.get("resumen_acto_reclamado", ""),
            datos_expediente=datos_exp,
            agravios=agravios_list,
            grupos_tematicos=grupos_list,
            observaciones_preliminares=analysis_data.get("observaciones_preliminares", ""),
            analysis_time_seconds=round(total_elapsed, 1),
        )

    except ImportError:
        raise HTTPException(500, "google-genai SDK not installed. Run: pip install google-genai")
    except HTTPException:
        raise
    except Exception as e:
        print(f"   ❌ Error en análisis: {e}")
        raise HTTPException(500, f"Error al analizar expediente: {str(e)}")


@app.post("/draft-sentencia")
async def draft_sentencia(
    tipo: str = Form(...),
    user_email: str = Form(...),
    instrucciones: str = Form(""),
    calificaciones: str = Form(""),
    sentido: str = Form(""),
    auto_mode: str = Form("false"),
    doc1: UploadFile = File(...),
    doc2: UploadFile = File(...),
    doc3: Optional[UploadFile] = File(None),
):
    """
    Redactor de Sentencias Federales (TCC) — Multi-Pass Pipeline.
    
    5-PHASE ARCHITECTURE:
    Phase 1: Extract structured data from 3 PDFs (JSON)
    Phase 2A: Draft RESULTANDOS
    Phase 2B: Draft CONSIDERANDOS (competencia, legitimación, procedencia)
    Phase 2C: Draft ESTUDIO DE FONDO + PUNTOS RESOLUTIVOS (the critical section)
    Phase 3: Polish and assemble final document
    
    Enhanced RAG: 8 queries, 30 results, 25K chars context.
    Secondary RAG queries generated from Phase 1 extracted data.
    """
    import time as time_module
    total_start = time_module.time()

    # ── Access validation (admin OR ultra_secretarios) ────────────────────
    if not GEMINI_API_KEY:
        raise HTTPException(500, "Gemini API key not configured")

    if not _can_access_sentencia(user_email):
        raise HTTPException(403, "Acceso restringido — se requiere suscripción Ultra Secretarios")

    # ── Validate tipo ────────────────────────────────────────────────────
    valid_types = list(SENTENCIA_PROMPTS.keys())
    if tipo not in valid_types:
        raise HTTPException(400, f"Tipo inválido. Opciones: {valid_types}")

    # ── Read PDF files ───────────────────────────────────────────────────
    doc_labels = SENTENCIA_DOC_LABELS[tipo]
    pdf_data = []
    doc_files = [doc1, doc2] + ([doc3] if doc3 else [])
    for i, (doc_file, label) in enumerate(zip(doc_files, doc_labels)):
        data = await doc_file.read()
        size_mb = len(data) / (1024 * 1024)
        if size_mb > 50:
            raise HTTPException(400, f"Archivo '{label}' excede 50MB ({size_mb:.1f}MB)")
        if not data:
            raise HTTPException(400, f"Archivo '{label}' está vacío")
        pdf_data.append((data, label, doc_file.filename or f"doc{i+1}.pdf"))
        print(f"   📄 {label}: {doc_file.filename} ({size_mb:.1f} MB)")

    # ── Build Gemini client and PDF parts ─────────────────────────────────
    try:
        from google import genai
        from google.genai import types as gtypes

        client = genai.Client(api_key=GEMINI_API_KEY)

        # Build reusable PDF parts (used by multiple phases)
        pdf_parts = []
        for pdf_bytes, label, filename in pdf_data:
            pdf_parts.append(gtypes.Part.from_text(text=f"\n--- DOCUMENTO: {label} (archivo: {filename}) ---\n"))
            pdf_parts.append(gtypes.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf"))

        print(f"\n🏛️ REDACTOR DE SENTENCIAS MULTI-PASS — Tipo: {tipo}")
        print(f"   📂 {len(pdf_data)} PDFs cargados")
        total_tokens_in = 0
        total_tokens_out = 0
        phases_done = 0

        # ═════════════════════════════════════════════════════════════════
        # PHASE 1: Structured Data Extraction
        # ═════════════════════════════════════════════════════════════════
        extracted_data = await phase1_extract_data(client, pdf_parts, tipo)
        phases_done = 1

        if not extracted_data:
            raise HTTPException(500, "Fase 1 falló: no se pudieron extraer datos de los PDFs")

        # ═════════════════════════════════════════════════════════════════
        # ENHANCED RAG: Secretary instructions + Phase 1 secondary queries
        # ═════════════════════════════════════════════════════════════════
        rag_context = ""
        rag_count = 0

        # Get secondary queries from extracted data
        secondary_queries = await extract_rag_queries_from_extraction(extracted_data)
        if secondary_queries:
            print(f"   🔎 RAG secundario: {len(secondary_queries)} queries de la extracción")

        # ── Auto-mode: build synthetic instructions if needed ─────────
        is_auto = auto_mode.lower() == "true"
        effective_instrucciones = instrucciones.strip()
        
        if is_auto and not effective_instrucciones:
            effective_instrucciones = _build_auto_mode_instructions(
                sentido, tipo, parsed_calificaciones
            )
            print(f"\n   🤖 MODO AUTOMÁTICO activado — instrucciones sintéticas generadas ({len(effective_instrucciones)} chars)")
        
        if sentido:
            print(f"   ⚖️ Sentido del fallo: {sentido.upper()}")
        
        if effective_instrucciones or secondary_queries:
            if effective_instrucciones:
                print(f"\n   📝 Instrucciones {'(auto)' if is_auto else '(secretario)'} ({len(effective_instrucciones)} chars):")
                print(f"      {effective_instrucciones[:200]}{'...' if len(effective_instrucciones) > 200 else ''}")
            try:
                rag_context = await run_rag_for_sentencia(
                    effective_instrucciones, tipo,
                    secondary_queries=secondary_queries
                )
                rag_count = rag_context.count("---") // 2 if rag_context else 0
                if rag_context:
                    print(f"   ✅ RAG context: {len(rag_context)} chars, ~{rag_count} resultados")
            except Exception as e:
                print(f"   ⚠️ RAG search failed (continuing without): {e}")
                rag_context = ""
        else:
            print("   ℹ️ Sin instrucciones ni datos extraídos para RAG")

        # ═════════════════════════════════════════════════════════════════
        # Parse calificaciones (determines which pipeline to use)
        # ═════════════════════════════════════════════════════════════════
        parsed_calificaciones = []
        if calificaciones.strip():
            try:
                parsed_calificaciones = json.loads(calificaciones)
                if not isinstance(parsed_calificaciones, list):
                    parsed_calificaciones = []
                print(f"\n   📋 Calificaciones del secretario: {len(parsed_calificaciones)} agravios")
                for c in parsed_calificaciones:
                    disp_tag = " [DISPOSITIVO]" if c.get("dispositivo") else ""
                    print(f"      Agravio {c.get('numero', '?')}: {c.get('calificacion', 'sin_calificar').upper()}{disp_tag}")
            except json.JSONDecodeError:
                print("   ⚠️ No se pudieron parsear las calificaciones, usando modo estándar")
                parsed_calificaciones = []

        # ── Inject sentido into instrucciones for legacy pipeline ──────
        if sentido and not is_auto:
            sentido_addendum = f"\n\nSENTIDO DEL FALLO INDICADO POR EL SECRETARIO: {sentido.upper()}\nDEBES redactar el proyecto en este sentido."
            effective_instrucciones = (effective_instrucciones or "") + sentido_addendum
        
        if parsed_calificaciones or sentido:
            # ═══════════════════════════════════════════════════════════
            # FOCUSED PIPELINE: Skip 2A/2B/3, concentrate on ESTUDIO DE FONDO
            # ═══════════════════════════════════════════════════════════
            print(f"\n   🎯 PIPELINE ENFOCADO: Estudio de Fondo + Efectos + Resolutivos")
            print(f"   ⏭️ Saltando Resultandos, Considerandos y Ensamblaje")
            if sentido:
                print(f"   ⚖️ Sentido: {sentido.upper()}")

            # Deep per-agravio pipeline (Steps A→B→C→D per agravio)
            estudio_fondo = await phase2c_adaptive_estudio_fondo(
                client, extracted_data, pdf_parts,
                tipo, parsed_calificaciones, rag_context
            )
            phases_done = 2

            # Final phase: Gemini-powered Efectos + Resolutivos
            efectos_resolutivos = await phase_final_efectos_resolutivos(
                client, extracted_data, estudio_fondo,
                tipo, parsed_calificaciones
            )
            phases_done = 3

            # Assemble focused output
            sentencia_text = _sanitize_sentencia_output(f"{estudio_fondo}\n\n{efectos_resolutivos}")

            total_elapsed = time_module.time() - total_start
            print(f"\n   {'═' * 50}")
            print(f"   ✅ PIPELINE ENFOCADO COMPLETADO")
            print(f"   📊 {len(sentencia_text):,} caracteres totales")
            print(f"   ⏱️ {total_elapsed:.1f} segundos totales")
            print(f"   🔄 {phases_done} fases (extracción + {len(parsed_calificaciones)} agravios × 4 pasos + efectos)")
            print(f"   {'═' * 50}")

            return DraftSentenciaResponse(
                sentencia_text=sentencia_text,
                tipo=tipo,
                tokens_input=None,
                tokens_output=None,
                model=GEMINI_MODEL,
                rag_results_count=rag_count,
                phases_completed=phases_done,
                total_chars=len(sentencia_text),
                generation_time_seconds=round(total_elapsed, 1),
            )

        else:
            # ═══════════════════════════════════════════════════════════
            # LEGACY PIPELINE: Full sentencia (2A → 2B → 2C → 3)
            # ═══════════════════════════════════════════════════════════
            print(f"\n   📜 PIPELINE COMPLETO: Sentencia integral (modo legado)")

            # Phase 2A: Draft RESULTANDOS
            resultandos = await phase2a_draft_resultandos(client, extracted_data, pdf_parts, tipo)
            phases_done = 2

            # Phase 2B: Draft CONSIDERANDOS (preliminary)
            considerandos = await phase2b_draft_considerandos(
                client, extracted_data, pdf_parts, tipo, rag_context
            )
            phases_done = 3

            # Phase 2C: Draft ESTUDIO DE FONDO (standard mode)
            estudio_fondo = await phase2c_draft_estudio_fondo(
                client, extracted_data, pdf_parts,
                tipo, effective_instrucciones, rag_context
            )
            phases_done = 4

            # Phase 3: Polish & Assembly
            sentencia_text = _sanitize_sentencia_output(await phase3_polish_assembly(
                client, extracted_data, resultandos, considerandos, estudio_fondo, tipo
            ))
            phases_done = 5

            total_elapsed = time_module.time() - total_start

            print(f"\n   ═══════════════════════════════════════════")
            print(f"   ✅ SENTENCIA MULTI-PASS COMPLETADA")
            print(f"   📊 {len(sentencia_text):,} caracteres totales")
            print(f"   ⏱️ {total_elapsed:.1f} segundos totales")
            print(f"   🔄 {phases_done} fases completadas")
            if rag_count:
                print(f"   📚 {rag_count} resultados RAG utilizados")
            print(f"   ═══════════════════════════════════════════")

            return DraftSentenciaResponse(
                sentencia_text=sentencia_text,
                tipo=tipo,
                tokens_input=None,
                tokens_output=None,
                model=GEMINI_MODEL,
                rag_results_count=rag_count,
                phases_completed=phases_done,
                total_chars=len(sentencia_text),
                generation_time_seconds=round(total_elapsed, 1),
            )

    except ImportError:
        raise HTTPException(500, "google-genai SDK not installed. Run: pip install google-genai")
    except HTTPException:
        raise
    except Exception as e:
        print(f"   ❌ Error Multi-Pass: {e}")
        raise HTTPException(500, f"Error al generar sentencia: {str(e)}")


# ═══════════════════════════════════════════════════════════════════════════════
# EXPORT — Download sentencia as formatted DOCX
# ═══════════════════════════════════════════════════════════════════════════════

SENTENCIA_TEMPLATE_PATH = os.path.join(os.path.dirname(__file__), "templates", "sentencia_tcc_template.docx")


class ExportSentenciaRequest(BaseModel):
    sentencia_text: str
    tipo: str = "amparo_directo"
    numero_expediente: str = ""
    materia: str = "CIVIL"
    quejoso: str = ""
    magistrado: str = ""
    secretario: str = ""
    user_email: str = ""


@app.post("/export-sentencia-docx")
async def export_sentencia_docx(req: ExportSentenciaRequest):
    """
    Exporta el texto de sentencia generado en un DOCX con formato oficial TCC.
    Usa el template con sellos, membrete, y formato Arial 14pt justificado.
    """
    import io
    from docx import Document as DocxDocument
    from docx.shared import Pt
    from docx.enum.text import WD_ALIGN_PARAGRAPH

    # ── Access validation (admin OR ultra_secretarios) ────────────────────
    if req.user_email and not _can_access_sentencia(req.user_email):
        raise HTTPException(403, "Acceso restringido — se requiere suscripción Ultra Secretarios")

    if not req.sentencia_text.strip():
        raise HTTPException(400, "El texto de la sentencia está vacío")

    # ── Load template ────────────────────────────────────────────────────
    if not os.path.exists(SENTENCIA_TEMPLATE_PATH):
        raise HTTPException(500, "Template DOCX no encontrado en el servidor")

    try:
        doc = DocxDocument(SENTENCIA_TEMPLATE_PATH)
    except Exception as e:
        raise HTTPException(500, f"Error al abrir template: {e}")

    # ── Type display names ───────────────────────────────────────────────
    tipo_display = {
        "amparo_directo": "AMPARO DIRECTO",
        "amparo_revision": "AMPARO EN REVISIÓN",
        "revision_fiscal": "REVISIÓN FISCAL",
        "queja": "RECURSO DE QUEJA",
    }.get(req.tipo, "AMPARO DIRECTO")

    # ── Update header text with expediente number ────────────────────────
    for section in doc.sections:
        header = section.header
        for para in header.paragraphs:
            if para.text.strip():
                # Replace the case number in header
                if req.numero_expediente:
                    para.text = f"{tipo_display} {req.materia.upper()} {req.numero_expediente}"
                    for run in para.runs:
                        run.font.name = "Arial"
                        run.font.size = Pt(14)
                        run.bold = True

    # ── Clear existing body paragraphs (keep only format) ────────────────
    # Remove all paragraphs except the last empty one
    for para in doc.paragraphs:
        p_element = para._element
        p_element.getparent().remove(p_element)

    # ── Build metadata header ────────────────────────────────────────────
    metadata_lines = []
    if req.numero_expediente:
        metadata_lines.append(
            (f"{tipo_display}: {req.numero_expediente}", True)
        )
    else:
        metadata_lines.append((f"{tipo_display}:", True))

    metadata_lines.append(("", False))  # blank line

    if req.materia:
        metadata_lines.append((f"MATERIA: {req.materia.upper()}", True))
        metadata_lines.append(("", False))

    # Note: Quejoso, Magistrado, Secretario removed from DOCX header
    # as estudio de fondo output doesn't need these metadata fields

    # Add blank lines before body
    metadata_lines.append(("", False))
    metadata_lines.append(("", False))

    # ── Write metadata paragraphs ────────────────────────────────────────
    for text, is_bold in metadata_lines:
        para = doc.add_paragraph()
        para.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        if text:
            run = para.add_run(text)
            run.font.name = "Arial"
            run.font.size = Pt(14)
            run.bold = is_bold
        else:
            run = para.add_run("")
            run.font.name = "Arial"
            run.font.size = Pt(14)

    # ── Write sentencia body ─────────────────────────────────────────────
    # Split text into paragraphs and write each one
    body_lines = req.sentencia_text.split("\n")
    for line in body_lines:
        para = doc.add_paragraph()
        para.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        clean_line = line.strip()

        if not clean_line:
            # Empty paragraph
            run = para.add_run("")
            run.font.name = "Arial"
            run.font.size = Pt(14)
        else:
            # Check if this line should be bold (headers, section titles)
            is_header = (
                clean_line.startswith("#")
                or clean_line.isupper()
                or clean_line.startswith("PRIMERO")
                or clean_line.startswith("SEGUNDO")
                or clean_line.startswith("TERCERO")
                or clean_line.startswith("CUARTO")
                or clean_line.startswith("QUINTO")
                or clean_line.startswith("SEXTO")
                or clean_line.startswith("SÉPTIMO")
                or clean_line.startswith("OCTAVO")
                or clean_line.startswith("NOVENO")
                or clean_line.startswith("DÉCIMO")
                or clean_line.startswith("R E S U L T A N D O")
                or clean_line.startswith("C O N S I D E R A N D O")
                or clean_line.startswith("V I S T O")
                or clean_line.startswith("P U N T O S")
                or clean_line.startswith("RESULTANDO")
                or clean_line.startswith("CONSIDERANDO")
                or clean_line.startswith("RESUELVE")
            )

            # Remove markdown # headers
            display_text = clean_line.lstrip("# ").strip() if clean_line.startswith("#") else clean_line

            # Handle **bold** markdown within text
            import re
            bold_pattern = re.compile(r'\*\*(.*?)\*\*')
            parts = bold_pattern.split(display_text)

            if len(parts) > 1:
                # Has inline bold markers
                for idx, part in enumerate(parts):
                    if part:
                        run = para.add_run(part)
                        run.font.name = "Arial"
                        run.font.size = Pt(14)
                        # Odd indices are the bold parts (inside **)
                        run.bold = (idx % 2 == 1) or is_header
            else:
                run = para.add_run(display_text)
                run.font.name = "Arial"
                run.font.size = Pt(14)
                run.bold = is_header

    # ── Save to buffer ───────────────────────────────────────────────────
    buffer = io.BytesIO()
    doc.save(buffer)
    buffer.seek(0)

    filename = f"Sentencia_{req.tipo}_{req.numero_expediente or 'borrador'}.docx".replace("/", "-").replace(" ", "_")

    print(f"   📄 DOCX exportado: {filename} ({buffer.getbuffer().nbytes:,} bytes)")

    return StreamingResponse(
        buffer,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "Access-Control-Expose-Headers": "Content-Disposition",
        },
    )


# ═══════════════════════════════════════════════════════════════════════════════
# MERGE — Combine adelanto DOCX (consideraciones previas) with estudio de fondo
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/merge-sentencia-docx")
async def merge_sentencia_docx(
    adelanto_file: UploadFile = File(..., description="DOCX del adelanto (consideraciones previas)"),
    estudio_text: str = Form(..., description="Texto del estudio de fondo generado"),
    tipo: str = Form("amparo_directo"),
    user_email: str = Form(""),
):
    """
    Recibe el DOCX del adelanto del secretario y el texto del estudio de fondo
    generado por Gemini. Detecta el punto de inserción (SIGUIENTE CONSIDERANDO,
    Estudio de fondo, o fin del documento) y acopla el estudio al formato del adelanto.
    """
    import io
    import re as re_mod
    from docx import Document as DocxDocument
    from docx.shared import Pt
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from copy import deepcopy

    # ── Access validation (admin OR ultra_secretarios) ────────────────────
    if user_email and not _can_access_sentencia(user_email):
        raise HTTPException(403, "Acceso restringido — se requiere suscripción Ultra Secretarios")

    if not estudio_text.strip():
        raise HTTPException(400, "El texto del estudio de fondo está vacío")

    # ── Validate file ────────────────────────────────────────────────────
    if not adelanto_file.filename or not adelanto_file.filename.lower().endswith(".docx"):
        raise HTTPException(400, "El archivo debe ser un documento .docx")

    # ── Read uploaded DOCX ───────────────────────────────────────────────
    try:
        contents = await adelanto_file.read()
        doc = DocxDocument(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(400, f"Error al leer el archivo DOCX: {e}")

    # ── Detect insertion point ───────────────────────────────────────────
    # Search for markers like "SIGUIENTE CONSIDERANDO", "Estudio de fondo",
    # or a combination. Fall back to the last paragraph.
    insertion_index = None
    markers = [
        "SIGUIENTE CONSIDERANDO",
        "ESTUDIO DE FONDO",
        "SIGUIENTE CONSIDERANDO. ESTUDIO DE FONDO",
    ]

    for i, para in enumerate(doc.paragraphs):
        text_upper = para.text.strip().upper()
        for marker in markers:
            if marker in text_upper:
                insertion_index = i
                break
        if insertion_index is not None:
            break

    # If no marker found, insert at the end
    if insertion_index is None:
        insertion_index = len(doc.paragraphs) - 1
        print(f"   ⚠️ No se encontró marcador de inserción, se agrega al final del documento")
    else:
        print(f"   ✅ Marcador encontrado en párrafo [{insertion_index}]: '{doc.paragraphs[insertion_index].text[:80]}...'")

    # ── Detect reference formatting from the document ────────────────────
    # Use the formatting of the paragraph nearest to the insertion point
    ref_para = doc.paragraphs[insertion_index] if insertion_index < len(doc.paragraphs) else doc.paragraphs[-1]
    ref_font_name = "Arial"
    ref_font_size = Pt(14)
    ref_alignment = WD_ALIGN_PARAGRAPH.JUSTIFY

    # Try to extract font from reference paragraph's runs
    for run in ref_para.runs:
        if run.font.name:
            ref_font_name = run.font.name
        if run.font.size:
            ref_font_size = run.font.size
        break  # Use first run's formatting

    if ref_para.alignment is not None:
        ref_alignment = ref_para.alignment

    print(f"   📝 Formato de referencia: {ref_font_name} {ref_font_size}, align={ref_alignment}")

    # ── Helper to add a paragraph after a specific index ──────────────────
    # python-docx doesn't have "insert after" natively, so we manipulate the XML
    def add_paragraph_after(doc, index, text="", bold=False):
        """Add a new paragraph after the paragraph at `index`."""
        ref_element = doc.paragraphs[index]._element
        new_para = doc.add_paragraph()  # This adds at the end temporarily
        new_element = new_para._element

        # Move it to right after the reference element
        ref_element.addnext(new_element)

        # Apply formatting
        new_para.alignment = ref_alignment
        if text:
            # Handle markdown **bold** inline markers
            bold_pattern = re_mod.compile(r'\*\*(.*?)\*\*')
            parts = bold_pattern.split(text)

            if len(parts) > 1:
                for idx, part in enumerate(parts):
                    if part:
                        run = new_para.add_run(part)
                        run.font.name = ref_font_name
                        run.font.size = ref_font_size
                        run.bold = (idx % 2 == 1) or bold
            else:
                run = new_para.add_run(text)
                run.font.name = ref_font_name
                run.font.size = ref_font_size
                run.bold = bold
        else:
            run = new_para.add_run("")
            run.font.name = ref_font_name
            run.font.size = ref_font_size

        return new_para

    # ── Insert estudio de fondo text ─────────────────────────────────────
    body_lines = estudio_text.split("\n")
    current_index = insertion_index

    # Add a blank line separator after the marker
    add_paragraph_after(doc, current_index, "")
    current_index += 1

    # Section header detection keywords
    header_keywords = (
        "PRIMERO", "SEGUNDO", "TERCERO", "CUARTO", "QUINTO",
        "SEXTO", "SÉPTIMO", "OCTAVO", "NOVENO", "DÉCIMO",
        "R E S U L T A N D O", "C O N S I D E R A N D O",
        "V I S T O", "P U N T O S", "RESULTANDO", "CONSIDERANDO",
        "RESUELVE",
    )

    for line in body_lines:
        clean_line = line.strip()

        # Determine if this line should be bold
        is_header = (
            clean_line.startswith("#")
            or clean_line.isupper()
            or any(clean_line.startswith(k) for k in header_keywords)
        )

        # Remove markdown # headers
        display_text = clean_line.lstrip("# ").strip() if clean_line.startswith("#") else clean_line

        add_paragraph_after(doc, current_index, display_text, bold=is_header)
        current_index += 1

    # ── Save to buffer ───────────────────────────────────────────────────
    buffer = io.BytesIO()
    doc.save(buffer)
    buffer.seek(0)

    # Build filename from original
    original_name = adelanto_file.filename.rsplit(".", 1)[0] if adelanto_file.filename else "Sentencia"
    filename = f"{original_name}_ConEstudioDeFondo.docx".replace(" ", "_")

    print(f"   📄 DOCX merged: {filename} ({buffer.getbuffer().nbytes:,} bytes), {len(body_lines)} líneas insertadas")

    return StreamingResponse(
        buffer,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "Access-Control-Expose-Headers": "Content-Disposition",
        },
    )


# ═══════════════════════════════════════════════════════════════════════════════
# ADMIN PANEL — Dashboard API
# ═══════════════════════════════════════════════════════════════════════════════

from fastapi import Header

ADMIN_EMAILS = {"administracion@iurexia.com"}

async def _verify_admin(authorization: str = Header(...)) -> dict:
    """Verify JWT token and check if email is in admin whitelist."""
    if not supabase_admin:
        raise HTTPException(status_code=503, detail="Supabase not configured")
    try:
        token = authorization.replace("Bearer ", "")
        user_resp = supabase_admin.auth.get_user(token)
        user = user_resp.user
        if not user or user.email not in ADMIN_EMAILS:
            raise HTTPException(status_code=403, detail="Acceso denegado")
        return {"id": user.id, "email": user.email}
    except HTTPException:
        raise
    except Exception as e:
        print(f"⚠️ Admin auth error: {e}")
        raise HTTPException(status_code=401, detail="Token inválido o expirado")

def _log_admin_action(admin_email: str, action: str, target_id: str = None, details: dict = None):
    """Log an admin action for audit trail."""
    if not supabase_admin:
        return
    try:
        supabase_admin.table("admin_audit_log").insert({
            "admin_email": admin_email,
            "action": action,
            "target_user_id": target_id,
            "details": details or {},
        }).execute()
    except Exception as e:
        print(f"⚠️ Failed to log admin action: {e}")


@app.get("/admin/users")
async def admin_list_users(authorization: str = Header(...)):
    """List all users with subscription info, usage, and blocked status."""
    admin = await _verify_admin(authorization)

    try:
        result = supabase_admin.rpc('admin_get_users').execute()
        users = result.data or []
        return {"users": users, "total": len(users) if isinstance(users, list) else 0}
    except Exception as e:
        print(f"❌ Admin users error: {e}")
        raise HTTPException(status_code=500, detail=f"Error al listar usuarios: {str(e)}")


@app.post("/admin/users/{user_id}/block")
async def admin_block_user(user_id: str, authorization: str = Header(...)):
    """Block a user from using the platform."""
    admin = await _verify_admin(authorization)

    # Prevent self-blocking
    if user_id == admin["id"]:
        raise HTTPException(status_code=400, detail="No puedes bloquearte a ti mismo")

    try:
        # Check if already blocked
        existing = supabase_admin.table("blocked_users").select("user_id").eq("user_id", user_id).execute()
        if existing.data:
            return {"status": "already_blocked", "user_id": user_id}

        # Get user info for audit log
        user_info = supabase_admin.table("user_profiles").select("email").eq("id", user_id).execute()
        user_email = user_info.data[0]["email"] if user_info.data else "unknown"

        supabase_admin.table("blocked_users").insert({
            "user_id": user_id,
            "blocked_by": admin["email"],
            "reason": "Bloqueado por administrador",
        }).execute()

        _log_admin_action(admin["email"], "block_user", user_id, {"user_email": user_email})
        print(f"🚫 Admin blocked user: {user_email} ({user_id})")

        return {"status": "blocked", "user_id": user_id, "user_email": user_email}
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Block user error: {e}")
        raise HTTPException(status_code=500, detail=f"Error al bloquear usuario: {str(e)}")


@app.post("/admin/users/{user_id}/unblock")
async def admin_unblock_user(user_id: str, authorization: str = Header(...)):
    """Unblock a previously blocked user."""
    admin = await _verify_admin(authorization)

    try:
        result = supabase_admin.table("blocked_users").delete().eq("user_id", user_id).execute()

        user_info = supabase_admin.table("user_profiles").select("email").eq("id", user_id).execute()
        user_email = user_info.data[0]["email"] if user_info.data else "unknown"

        _log_admin_action(admin["email"], "unblock_user", user_id, {"user_email": user_email})
        print(f"✅ Admin unblocked user: {user_email} ({user_id})")

        return {"status": "unblocked", "user_id": user_id, "user_email": user_email}
    except Exception as e:
        print(f"❌ Unblock user error: {e}")
        raise HTTPException(status_code=500, detail=f"Error al desbloquear usuario: {str(e)}")


@app.get("/admin/alerts")
async def admin_list_alerts(
    reviewed: bool = False,
    limit: int = 50,
    authorization: str = Header(...),
):
    """List security alerts, optionally filtered by review status."""
    admin = await _verify_admin(authorization)

    try:
        query = supabase_admin.table("security_alerts").select("*").order("created_at", desc=True).limit(limit)
        if not reviewed:
            query = query.eq("reviewed", False)
        result = query.execute()
        return {"alerts": result.data or [], "total": len(result.data or [])}
    except Exception as e:
        print(f"❌ Admin alerts error: {e}")
        raise HTTPException(status_code=500, detail=f"Error al listar alertas: {str(e)}")


@app.post("/admin/alerts/{alert_id}/review")
async def admin_review_alert(alert_id: int, authorization: str = Header(...)):
    """Mark a security alert as reviewed."""
    admin = await _verify_admin(authorization)

    try:
        supabase_admin.table("security_alerts").update({
            "reviewed": True,
            "reviewed_by": admin["email"],
            "reviewed_at": "now()",
        }).eq("id", alert_id).execute()

        _log_admin_action(admin["email"], "review_alert", details={"alert_id": alert_id})
        return {"status": "reviewed", "alert_id": alert_id}
    except Exception as e:
        print(f"❌ Review alert error: {e}")
        raise HTTPException(status_code=500, detail=f"Error al revisar alerta: {str(e)}")


@app.get("/admin/stats")
async def admin_stats(authorization: str = Header(...)):
    """Dashboard statistics: total users, subscribers by plan, active users, alerts."""
    admin = await _verify_admin(authorization)

    try:
        # Total users
        all_users = supabase_admin.table("user_profiles").select("id", count="exact").execute()
        total_users = all_users.count or 0

        # Subscribers by plan
        plan_data = supabase_admin.table("user_profiles").select(
            "subscription_type"
        ).execute()
        plans = {}
        for row in (plan_data.data or []):
            st = row.get("subscription_type", "gratuito")
            plans[st] = plans.get(st, 0) + 1

        # Active users (last 7 days)
        from datetime import datetime, timedelta
        seven_days_ago = (datetime.utcnow() - timedelta(days=7)).isoformat()
        active_result = supabase_admin.table("user_profiles").select(
            "id", count="exact"
        ).gte("last_query_at", seven_days_ago).execute()
        active_7d = active_result.count or 0

        # Pending security alerts
        pending_alerts = supabase_admin.table("security_alerts").select(
            "id", count="exact"
        ).eq("reviewed", False).execute()
        pending_count = pending_alerts.count or 0

        # Blocked users count
        blocked_result = supabase_admin.table("blocked_users").select(
            "user_id", count="exact"
        ).execute()
        blocked_count = blocked_result.count or 0

        return {
            "total_users": total_users,
            "active_7d": active_7d,
            "blocked_users": blocked_count,
            "pending_alerts": pending_count,
            "plans": plans,
        }
    except Exception as e:
        print(f"❌ Admin stats error: {e}")
        raise HTTPException(status_code=500, detail=f"Error al obtener estadísticas: {str(e)}")


# ═══════════════════════════════════════════════════════════════════════════════
# ADMIN — Reingest Sparse Vectors
# ═══════════════════════════════════════════════════════════════════════════════

_reingest_running = False
_reingest_status = {"status": "idle", "processed": 0, "total": 0, "errors": 0}

@app.post("/admin/reingest-sparse")
async def admin_reingest_sparse(req: ReingestRequest):
    """
    Genera BM25 sparse vectors reales para una colección.
    V5.0: Soporta leyes_estatales (legacy) y colecciones por estado.
    Corre como background task. Solo permite una ejecución a la vez.
    """
    global _reingest_running, _reingest_status
    
    # Auth simple
    expected_key = os.getenv("ADMIN_KEY", "jurexia-reingest-2026")
    if req.admin_key != expected_key:
        raise HTTPException(status_code=403, detail="Unauthorized")
    
    if _reingest_running:
        return {"status": "already_running", **_reingest_status}
    
    async def _run_reingest(entidad: Optional[str], collection_name: str):
        global _reingest_running, _reingest_status
        _reingest_running = True
        _reingest_status = {"status": "running", "processed": 0, "total": 0, "errors": 0}
        
        try:
            from qdrant_client.models import PointVectors
            
            # Count
            filter_ = None
            if entidad:
                filter_ = Filter(
                    must=[FieldCondition(key="entidad", match=MatchValue(value=entidad))]
                )
            count_result = await qdrant_client.count(
                collection_name=collection_name, count_filter=filter_
            )
            _reingest_status["total"] = count_result.count
            print(f"[REINGEST] Starting BM25 re-ingestion: {count_result.count} points in {collection_name}, entidad={entidad}")
            
            # Scroll and process
            offset = None
            batch_size = 50
            
            while True:
                results, next_offset = await qdrant_client.scroll(
                    collection_name=collection_name,
                    scroll_filter=filter_,
                    limit=100,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,
                )
                
                if not results:
                    break
                
                # Generate sparse vectors in mini-batches
                updates = []
                for point in results:
                    payload = point.payload or {}
                    texto = payload.get("texto", payload.get("text", ""))
                    if not texto:
                        continue
                    
                    try:
                        embeddings = list(sparse_encoder.passage_embed([texto]))
                        if embeddings and len(embeddings[0].indices) > 0:
                            sparse = embeddings[0]
                            updates.append(PointVectors(
                                id=point.id,
                                vector={
                                    "sparse": SparseVector(
                                        indices=sparse.indices.tolist(),
                                        values=sparse.values.tolist(),
                                    )
                                }
                            ))
                    except Exception as e:
                        _reingest_status["errors"] += 1
                
                # Upload batch
                if updates:
                    try:
                        await qdrant_client.update_vectors(
                            collection_name=collection_name,
                            points=updates,
                        )
                    except Exception as e:
                        print(f"[REINGEST] Batch update error: {e}")
                        _reingest_status["errors"] += 1
                
                _reingest_status["processed"] += len(results)
                
                if _reingest_status["processed"] % 1000 < 100:
                    print(f"[REINGEST] Progress: {_reingest_status['processed']}/{_reingest_status['total']}")
                
                if next_offset is None:
                    break
                offset = next_offset
                
                await asyncio.sleep(0.1)  # Yield to event loop
            
            _reingest_status["status"] = "completed"
            print(f"[REINGEST] Done! {_reingest_status['processed']} processed, {_reingest_status['errors']} errors")
            
        except Exception as e:
            _reingest_status["status"] = f"error: {str(e)}"
            print(f"[REINGEST] Fatal error: {e}")
        finally:
            _reingest_running = False
    
    # Launch as background task
    asyncio.create_task(_run_reingest(req.entidad, req.collection))
    
    return {"status": "started", "entidad": req.entidad, "message": "Check GET /admin/reingest-status for progress"}


@app.get("/admin/reingest-status")
async def admin_reingest_status():
    """Check the status of BM25 re-ingestion."""
    return {"running": _reingest_running, **_reingest_status}


# ══════════════════════════════════════════════════════════════════════════════
# SALVAME — Amparo de Emergencia por Salud (DeepSeek)
# ══════════════════════════════════════════════════════════════════════════════

SALVAME_SYSTEM_PROMPT = """Eres IUREXIA, un abogado constitucionalista mexicano experto en amparo en materia de salud y litigio estratégico. Redacta una DEMANDA DE AMPARO INDIRECTO con solicitud de SUSPENSIÓN DE OFICIO Y DE PLANO, con enfoque de urgencia y protección inmediata de la vida e integridad.

═══════════════════════════════════════════════════════════════════════
MANDATO ABSOLUTO DE CERO ALUCINACIONES — LÉELO CON MÁXIMA ATENCIÓN
═══════════════════════════════════════════════════════════════════════

PROHIBIDO INVENTAR CITAS JURÍDICAS. Esto es una demanda real que una persona presentará ante un juez federal. Cualquier tesis, jurisprudencia, registro, rubro, criterio o referencia judicial que NO esté incluida textualmente en este prompt está PROHIBIDA.

REGLAS INQUEBRANTABLES:
1. SOLO puedes citar las 4 tesis verificadas que se proporcionan abajo, TEXTUALMENTE.
2. NO inventes registros, rubros, claves de tesis, nombres de tribunales ni números de jurisprudencia.
3. Si sientes la necesidad de citar algo más, NO LO HAGAS. Desarrolla el argumento con fundamento constitucional directo (arts. 1, 4 y 22 CPEUM) y con la Ley de Amparo.
4. Es MEJOR un escrito con 4 tesis reales que uno con 10 tesis inventadas. Las tesis falsas causan desechamiento y responsabilidad profesional.
5. Puedes citar artículos de la Constitución, la Ley de Amparo, la Ley General de Salud y tratados internacionales (PIDESC, Convención Americana) SIN restricción — esos son verificables.

═══════════════════════════════════════════════════════════════════════
LAS 4 TESIS VERIFICADAS — ÚSALAS TEXTUALMENTE
═══════════════════════════════════════════════════════════════════════

TESIS 1 — Jurisprudencia PR.A.C.CS. J/14 A (11a.)
Rubro: "SUSPENSIÓN DE OFICIO Y DE PLANO EN AMPARO INDIRECTO. PROCEDE CONTRA LA OMISIÓN DEL INSTITUTO MEXICANO DEL SEGURO SOCIAL (IMSS) DE BRINDAR ATENCIÓN MÉDICA ESPECIALIZADA URGENTE AL GRADO DE PONER EN PELIGRO LA VIDA DEL QUEJOSO."
Contexto para uso: El Pleno Regional determinó que la suspensión contra la omisión de brindar atención médica especializada en casos urgentes, como la práctica de una cirugía previamente diagnosticada, debe tramitarse conforme al artículo 126 de la Ley de Amparo, pues tal omisión puede afectar la dignidad e integridad personal del quejoso al grado de poner en peligro su vida. Se precisó que el juzgador de amparo debe realizar un juicio valorativo, ponderando las manifestaciones de la demanda y sus anexos, para determinar si la falta de atención médica reclamada tiene relación con una lesión o padecimiento que cause dolor físico o un estado patológico que pudiera tener consecuencias irreversibles en la salud o causar la pérdida de la vida. Este criterio subraya que la regulación diferenciada de la suspensión de oficio y de plano obedece a la necesidad de tutelar con la máxima celeridad derechos fundamentales de especial relevancia como la vida y la integridad personal.

TESIS 2 — Criterio del Segundo Tribunal Colegiado en Materias Penal y Administrativa del Décimo Séptimo Circuito
Rubro: "SUSPENSIÓN DE OFICIO Y DE PLANO EN EL JUICIO DE AMPARO INDIRECTO. PROCEDE CONCEDERLA CONTRA EL REQUERIMIENTO DE PAGO DE CIERTA CANTIDAD DE DINERO POR PARTE DE UNA INSTITUCIÓN DE SALUD PRIVADA, POR CONCEPTO DE SERVICIOS MÉDICOS, PARA EL EFECTO DE QUE SE ATIENDA DE URGENCIA AL QUEJOSO HASTA QUE FINALICE EL PROCEDIMIENTO QUE MOTIVÓ SU INGRESO Y SE GENERE SU EGRESO HOSPITALARIO."
Contexto para uso: Los tribunales federales han determinado que la suspensión de plano es igualmente procedente cuando el acto reclamado proviene de una institución de salud privada que condiciona la prestación de un servicio médico de urgencia al pago de una contraprestación económica. En estos casos, el derecho a la vida prevalece sobre cualquier interés de carácter patrimonial.

TESIS 3 — Criterio del Tercer Tribunal Colegiado en Materia Administrativa del Segundo Circuito
Rubro: "SUSPENSIÓN DE OFICIO Y DE PLANO EN EL JUICIO DE AMPARO INDIRECTO. PROCEDE CUANDO SE RECLAMA LA FALTA DE ATENCIÓN MÉDICA OPORTUNA Y CONTINUA, ASÍ COMO EL OTORGAMIENTO Y SUMINISTRO DE MEDICAMENTOS, SI COMPROMETE LA DIGNIDAD E INTEGRIDAD PERSONAL DEL QUEJOSO, AL GRADO DE EQUIPARARSE A UN TORMENTO."
Contexto para uso: Cuando las circunstancias del caso revelan que la falta de atención médica compromete gravemente la dignidad e integridad personal del quejoso, el juzgador debe actuar de inmediato. La omisión de proporcionar atención médica oportuna y continua, así como el suministro de medicamentos, puede llegar a constituir un acto equiparable a un tormento, lo que actualiza de forma directa la hipótesis de procedencia de la suspensión de oficio y de plano.

TESIS 4 — Criterio del Décimo Octavo Tribunal Colegiado en Materia Administrativa del Primer Circuito
Rubro: "SUSPENSIÓN DE PLANO Y DE OFICIO. CUANDO ES PROCEDENTE SU CONCESIÓN NO IMPORTA QUE AFECTE LA PERVIVENCIA DEL JUICIO, PUES NO PUEDE PREVALECER LA FORMA SOBRE EL FONDO."
Contexto para uso: No es óbice para la concesión de la medida cautelar el argumento de que esta podría tener efectos restitutorios y dejar sin materia el juicio de amparo. La finalidad primordial de la suspensión en estos casos es la protección de los derechos humanos más fundamentales, por lo que cualquier consideración de índole procesal debe ceder ante la tutela de valores superiores. La forma no puede prevalecer sobre el fondo, y es procedente conceder la suspensión aun a costa de que se anticipen los efectos de una eventual sentencia concesoria.

═══════════════════════════════════════════════════════════════════════
FIN DE TESIS VERIFICADAS — NADA MÁS PUEDE CITARSE COMO JURISPRUDENCIA
═══════════════════════════════════════════════════════════════════════

Reglas generales:
- Produce un escrito listo para imprimirse: formal, claro, sin relleno.
- Usa español jurídico mexicano, pero comprensible.
- Prioridad máxima: Capítulo de SUSPENSIÓN (de oficio y de plano) con solicitud explícita y medidas concretas.
- Fundamenta: Derecho a la Salud (art. 4º Constitucional), vida e integridad, y argumenta riesgo grave por omisión. Vincula con dignidad e integridad cuando proceda.
- Legitima al promovente con el artículo 15 de la Ley de Amparo cuando promueva en nombre de otro.
- NUNCA autorices a un abogado, licenciado, pasante, ni "Lic." alguno en el proemio ni en ninguna parte del escrito. El promovente actúa POR SU PROPIO DERECHO en términos del artículo 15 de la Ley de Amparo. No existen autorizados, representantes legales ni abogados patronos. NO inventes nombres de licenciados (como "Lic. Iurexia" o similar). Solo el promovente firma.
- La demanda se DIRIGE al C. JUEZ DE DISTRITO competente EN TURNO (según los datos proporcionados). La demanda NUNCA se dirige a una Oficialía de Partes. La Oficialía de Partes es solo el lugar físico donde se entrega el escrito, pero el encabezado del escrito siempre dice "C. JUEZ DE DISTRITO...".

═══════════════════════════════════════════════════════════════════════
AUTORIDADES RESPONSABLES POR TIPO DE INSTITUCIÓN
═══════════════════════════════════════════════════════════════════════

Según la institución de salud involucrada, SIEMPRE señala las siguientes autoridades responsables (incluso si no se conocen los nombres exactos de los titulares):

• IMSS:
  1. Titular del Órgano de Operación Administrativa Desconcentrada del IMSS en el estado correspondiente, con domicilio en [usar domicilio conocido o señalar "domicilio que se solicita sea requerido por este H. Juzgado"]
  2. Director/a del Hospital o Unidad Médica específica
  3. Médico(s) tratante(s) que nieguen o retarden la atención (si se conocen nombres)

• ISSSTE:
  1. Delegado/a del ISSSTE en el estado correspondiente
  2. Director/a del Hospital o Clínica específica
  3. Médico(s) tratante(s) (si se conocen nombres)

• Secretaría de Salud (hospital estatal):
  1. Secretario/a de Salud del Estado correspondiente
  2. Director/a del Hospital estatal específico
  3. Médico(s) tratante(s) (si se conocen nombres)

• IMSS-Bienestar:
  1. Director General del IMSS-Bienestar
  2. Coordinador/a estatal del IMSS-Bienestar
  3. Director/a o encargado/a del centro de salud específico

• Hospital municipal:
  1. Presidente Municipal del municipio correspondiente
  2. Director/a del Hospital municipal
  3. Médico(s) tratante(s) (si se conocen nombres)

• Institución privada:
  1. Representante legal del hospital/clínica privada
  2. Director médico del hospital/clínica
  3. Médico(s) que condicionan o niegan la atención

NOTA: Cuando no se conozcan los nombres de los titulares, señálalos por su cargo oficial completo. Cuando no se conozca el domicilio exacto de una autoridad jerárquica, usa la fórmula: "con domicilio que se solicita sea requerido mediante informe justificado".

Efectos solicitados: valoración inmediata, suministro de medicamentos, realización de cirugía/atención inaplazable; y si no hay capacidad, ordenar acciones para garantizar la atención (incluida subrogación cuando proceda).

Estructura obligatoria (con encabezados en MAYÚSCULAS):
- Encabezado: EXTREMA URGENCIA / AMPARO INDIRECTO
- QUEJOSO
- PROMOVENTE (con art. 15 LA si aplica)
- ASUNTO
- Dirigido al C. JUEZ DE DISTRITO...
- Proemio con datos del promovente y paciente
- I. NOMBRE Y DOMICILIO DE LA PERSONA QUEJOSA Y DE QUIEN PROMUEVE EN SU NOMBRE
- II. NOMBRE Y DOMICILIO DE LA PERSONA TERCERA INTERESADA
- III. AUTORIDADES RESPONSABLES
- IV. ACTO RECLAMADO
- V. HECHOS (bajo protesta de decir verdad, primera persona, urgencia, cronología)
- VI. PRECEPTOS CONSTITUCIONALES VIOLADOS
- VII. CONCEPTOS DE VIOLACIÓN
- VIII. SUSPENSIÓN DE OFICIO Y DE PLANO (PRIORIDAD MÁXIMA — desarrollar extensamente con las 4 tesis verificadas)
- Puntos petitorios (PRIMERO, SEGUNDO, TERCERO)
- PROTESTO LO NECESARIO
- Lugar y fecha
- Nombre y firma

FORMATO DE REFERENCIA (sigue esta estructura y estilo):

---
EXTREMA URGENCIA
AMPARO INDIRECTO
QUEJOSO: [NOMBRE DEL PACIENTE EN PELIGRO]
PROMOVENTE: [NOMBRE], en términos del artículo 15 de la Ley de Amparo.
ASUNTO: SE PROMUEVE DEMANDA DE AMPARO INDIRECTO Y SE SOLICITA SUSPENSIÓN DE OFICIO Y DE PLANO.

C. JUEZ DE DISTRITO EN MATERIA DE AMPARO CIVIL, ADMINISTRATIVO Y DE TRABAJO Y DE JUICIOS FEDERALES EN TURNO
P R E S E N T E.

[Proemio con datos, domicilio y autorizaciones]

Que por medio del presente escrito, vengo a solicitar el AMPARO Y PROTECCIÓN DE LA JUSTICIA FEDERAL a favor de [PACIENTE]...

I. NOMBRE Y DOMICILIO DE LA PERSONA QUEJOSA Y DE QUIEN PROMUEVE EN SU NOMBRE:
Quejoso (Paciente agraviado): [datos y ubicación hospitalaria]
Promovente: [datos]

II. NOMBRE Y DOMICILIO DE LA PERSONA TERCERA INTERESADA:
Bajo protesta de decir verdad, manifiesto que no existe tercero interesado...

III. AUTORIDADES RESPONSABLES:
[Director del hospital, Titular de Secretaría de Salud / IMSS / ISSSTE, médicos responsables]

IV. ACTO RECLAMADO:
La omisión y negativa de brindar atención médica integral...

V. HECHOS:
[Cronología detallada]

VI. PRECEPTOS CONSTITUCIONALES VIOLADOS:
Artículos 1, 4 y 22 de la Constitución...

VII. CONCEPTOS DE VIOLACIÓN:
[Desarrollo jurídico extenso — usar SOLO artículos constitucionales y de ley, NO inventar jurisprudencia]

VIII. SUSPENSIÓN DE OFICIO Y DE PLANO:
[Desarrollo extenso con las 4 TESIS VERIFICADAS citadas textualmente, fundamentación en art. 126 LA, efectos concretos: valoración, medicamentos, cirugía, subrogación]

PUNTOS PETITORIOS:
PRIMERO. Tenerme por presentado en términos del artículo 15 de la Ley de Amparo...
SEGUNDO. Admitir el presente escrito en cualquier día y hora (art. 20 LA)...
TERCERO. Decretar la suspensión de oficio y de plano...

PROTESTO LO NECESARIO
[Lugar y Fecha]
[NOMBRE Y FIRMA DEL PROMOVENTE]
---

REGLAS FINALES:
- No agregues comentarios, no expliques: entrega SOLO el texto del escrito.
- El capítulo de SUSPENSIÓN debe ser el más extenso y desarrollado, con las 4 tesis verificadas citadas textualmente con su rubro completo.
- Adapta los hechos al relato del usuario, haciéndolos vívidos y urgentes pero formales.
- El escrito completo debe tener entre 3000 y 5000 palabras.
- FORMATO: NO uses asteriscos (**), markdown ni caracteres especiales de formato. Los encabezados deben ir en MAYÚSCULAS sin marcadores. Escribe texto plano formal, sin ningún tipo de formato markdown.
- RECUERDA: CERO ALUCINACIONES. Si no estás seguro de una cita, NO LA INCLUYAS. Solo las 4 tesis proporcionadas arriba."""


class AmparoSaludRequest(BaseModel):
    promovente_nombre: str
    promovente_telefono: str = ""
    promovente_correo: str = ""
    promovente_domicilio: str
    promueve_por_paciente: bool = False
    parentesco: str = ""  # Parentesco con el paciente (padre, cónyuge, hijo, etc.)
    paciente_nombre: str
    paciente_edad: str
    paciente_diagnostico: str
    paciente_riesgo: str
    institucion: str
    hospital_nombre: str
    hospital_ciudad: str
    hospital_estado: str
    hospital_direccion: str = ""  # Dirección completa del hospital
    director_nombre: str = ""
    situaciones: list[str]
    descripcion_libre: str = ""
    detalles_medicos_adicionales: str = ""  # Nombres de médicos que niegan atención, circunstancias
    confirma_veracidad: bool = True
    user_email: str = ""


@app.post("/generate-amparo-salud")
async def generate_amparo_salud(req: AmparoSaludRequest):
    """
    SALVAME: Genera una Demanda de Amparo Indirecto en materia de salud
    usando DeepSeek Chat con streaming.
    """
    from fastapi.responses import StreamingResponse

    # ── Build user prompt from form data ─────────────────────────────────
    riesgo_map = {
        "muerte": "riesgo inminente de muerte",
        "deterioro": "deterioro grave e irreversible de salud",
        "dolor": "dolor extremo e insoportable",
        "discapacidad": "discapacidad inminente e irreversible",
        "otro": "otro riesgo grave para la salud",
    }
    riesgo_desc = riesgo_map.get(req.paciente_riesgo, req.paciente_riesgo)

    parentesco_info = f"Parentesco con el paciente: {req.parentesco}" if req.parentesco else ""
    hospital_dir_info = f"Dirección del hospital: {req.hospital_direccion}" if req.hospital_direccion else ""
    detalles_med_info = f"Detalles adicionales sobre personal médico/circunstancias: {req.detalles_medicos_adicionales}" if req.detalles_medicos_adicionales else ""

    user_prompt = f"""Genera la demanda de amparo indirecto con los siguientes datos:

PROMOVENTE: {req.promovente_nombre}
Domicilio para notificaciones: {req.promovente_domicilio}
{'Teléfono: ' + req.promovente_telefono if req.promovente_telefono else ''}
{'Correo: ' + req.promovente_correo if req.promovente_correo else ''}
{'Promueve en nombre del paciente por imposibilidad (art. 15 LA)' if req.promueve_por_paciente else 'Promueve por derecho propio'}
{parentesco_info}

PACIENTE (QUEJOSO): {req.paciente_nombre}
Edad: {req.paciente_edad} años
Diagnóstico: {req.paciente_diagnostico}
Riesgo actual: {riesgo_desc}

AUTORIDAD RESPONSABLE:
Institución: {req.institucion}
Hospital/Clínica: {req.hospital_nombre}
Ubicación: {req.hospital_ciudad}, {req.hospital_estado}
{hospital_dir_info}
{'Director/Médico responsable: ' + req.director_nombre if req.director_nombre else ''}
{detalles_med_info}

SITUACIÓN:
Actos reclamados: {', '.join(req.situaciones)}
{'Relato del promovente: ' + req.descripcion_libre if req.descripcion_libre else ''}

IMPORTANTE:
- La demanda se DIRIGE al Juzgado de Distrito competente en turno, NUNCA a una Oficialía de Partes.
- NO autorices a ningún abogado, licenciado ni "Lic.". El promovente actúa por su propio derecho bajo el artículo 15 de la Ley de Amparo.
- Señala las autoridades responsables según la tabla de institución proporcionada en tus instrucciones.

Genera el escrito completo siguiendo EXACTAMENTE la estructura y formato del modelo de referencia."""

    # ── Lookup correct Juzgado de Distrito ────────────────────────────────
    juzgado_info = ""
    if supabase_admin:
        try:
            result = supabase_admin.table("juzgados_distrito").select("denominacion,direccion,telefono,materia").eq("estado", req.hospital_estado).execute()
            if result.data:
                courts = result.data
                # Priority: amparo courts > Administrativa > Mixto > others
                amparo_courts = [j for j in courts if 'amparo' in j["denominacion"].lower()]
                admin_courts  = [j for j in courts if j["materia"] == "Administrativa"]
                mixto_courts  = [j for j in courts if j["materia"] == "Mixto"]
                chosen = (amparo_courts or admin_courts or mixto_courts or courts)[0]

                turno_name = _build_turno_denomination(chosen["denominacion"])
                oficialia_addr = _extract_building_address(chosen["direccion"])

                juzgado_info = f"""\n\nJUZGADO COMPETENTE (usar esta denominación en el ENCABEZADO del escrito):
DIRIGIR LA DEMANDA A: {turno_name}
Dirección para presentación física (Oficialía de Partes): {oficialia_addr}
{'Teléfono: ' + chosen['telefono'] if chosen.get('telefono') else ''}
IMPORTANTE: El encabezado del escrito SIEMPRE dice 'C. {turno_name} / P R E S E N T E.'. La Oficialía de Partes es solo el lugar donde se entrega físicamente el escrito, pero la demanda se DIRIGE al Juzgado."""
                user_prompt += juzgado_info
                print(f"   🏛️  Juzgado en turno: {turno_name}")
                print(f"   📍  Oficialía: {oficialia_addr}")
        except Exception as e:
            print(f"   ⚠️  No se pudo buscar juzgado: {e}")

    print(f"\n🏥 SALVAME — Generando amparo de salud")
    print(f"   Paciente: {req.paciente_nombre}")
    print(f"   Riesgo: {riesgo_desc}")
    print(f"   Hospital: {req.hospital_nombre} ({req.institucion})")
    print(f"   Situaciones: {', '.join(req.situaciones)}")

    # ── Stream from DeepSeek ─────────────────────────────────────────────
    async def stream_response():
        try:
            response = await deepseek_client.chat.completions.create(
                model=DEEPSEEK_CHAT_MODEL,
                messages=[
                    {"role": "system", "content": SALVAME_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=8000,
                temperature=0.3,
                stream=True,
            )

            async for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            print(f"   ❌ SALVAME error: {e}")
            yield f"\n\n[Error al generar el amparo: {str(e)}]"

    return StreamingResponse(stream_response(), media_type="text/plain")


class ExportAmparoSaludRequest(BaseModel):
    amparo_text: str
    promovente_nombre: str = ""
    paciente_nombre: str = ""


@app.post("/export-amparo-salud-docx")
async def export_amparo_salud_docx(req: ExportAmparoSaludRequest):
    """
    Exporta el amparo de salud generado como DOCX con formato judicial.
    """
    import io
    from docx import Document as DocxDocument
    from docx.shared import Pt, Cm
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from fastapi.responses import StreamingResponse

    if not req.amparo_text.strip():
        raise HTTPException(400, "El texto del amparo está vacío")

    try:
        doc = DocxDocument()

        # ── Page margins ─────────────────────────────────────────────────
        for section in doc.sections:
            section.top_margin = Cm(2.5)
            section.bottom_margin = Cm(2.5)
            section.left_margin = Cm(3)
            section.right_margin = Cm(2.5)

        # ── Default style ────────────────────────────────────────────────
        style = doc.styles['Normal']
        font = style.font
        font.name = 'Arial'
        font.size = Pt(14)
        style.paragraph_format.line_spacing = 1.5
        style.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY

        # ── Parse text into paragraphs ───────────────────────────────────
        import re
        lines = req.amparo_text.split('\n')
        for line in lines:
            stripped = line.strip()
            if not stripped:
                doc.add_paragraph('')
                continue

            para = doc.add_paragraph()

            # Handle markdown headings (## or #)
            is_md_heading = False
            if stripped.startswith('## '):
                stripped = stripped.lstrip('#').strip()
                is_md_heading = True
            elif stripped.startswith('# '):
                stripped = stripped.lstrip('#').strip()
                is_md_heading = True

            # Handle bullet points
            is_bullet = False
            if re.match(r'^[-•]\s+', stripped):
                stripped = re.sub(r'^[-•]\s+', '', stripped)
                is_bullet = True

            # Check if it's a header-style line (all caps or known patterns)
            is_header = is_md_heading or (
                stripped.isupper() and len(stripped) < 120
            ) or stripped.startswith('I.') or stripped.startswith('II.') or stripped.startswith('III.') or stripped.startswith('IV.') or stripped.startswith('V.') or stripped.startswith('VI.') or stripped.startswith('VII.') or stripped.startswith('VIII.') or stripped.startswith('PRIMERO') or stripped.startswith('SEGUNDO') or stripped.startswith('TERCERO')

            # Add bullet prefix if needed
            if is_bullet:
                bullet_run = para.add_run('• ')
                bullet_run.font.name = 'Arial'
                bullet_run.font.size = Pt(14)

            # Parse **bold** markers into separate runs
            parts = re.split(r'(\*\*.*?\*\*)', stripped)
            for part in parts:
                if part.startswith('**') and part.endswith('**'):
                    run = para.add_run(part[2:-2])
                    run.bold = True
                else:
                    run = para.add_run(part)
                    if is_header:
                        run.bold = True
                run.font.name = 'Arial'
                run.font.size = Pt(14)

            if is_header:
                if stripped in ['EXTREMA URGENCIA', 'AMPARO INDIRECTO', 'PROTESTO LO NECESARIO', 'P R E S E N T E.']:
                    para.alignment = WD_ALIGN_PARAGRAPH.CENTER
                elif is_md_heading:
                    para.alignment = WD_ALIGN_PARAGRAPH.CENTER
                else:
                    para.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            else:
                para.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY

        # ── Save to buffer ───────────────────────────────────────────────
        buffer = io.BytesIO()
        doc.save(buffer)
        buffer.seek(0)

        safe_name = (req.paciente_nombre or "paciente").replace(" ", "_").replace("/", "-")
        filename = f"Amparo_Salud_{safe_name}.docx"

        print(f"   📄 SALVAME DOCX exportado: {filename} ({buffer.getbuffer().nbytes:,} bytes)")

        return StreamingResponse(
            buffer,
            media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    except Exception as e:
        print(f"   ❌ SALVAME DOCX error: {e}")
        raise HTTPException(500, f"Error al generar DOCX: {str(e)}")


# ══════════════════════════════════════════════════════════════════════════════
# JUZGADOS DE DISTRITO — Directorio CJF
# ══════════════════════════════════════════════════════════════════════════════
import re as _re

# ── Ordinals to strip from court names ────────────────────────────────────────
_ORDINALS = [
    "PRIMERO", "SEGUNDO", "TERCERO", "CUARTO", "QUINTO",
    "SEXTO", "SÉPTIMO", "OCTAVO", "NOVENO", "DÉCIMO",
    "DÉCIMO PRIMERO", "DÉCIMO SEGUNDO", "DÉCIMO TERCERO",
    "DÉCIMO CUARTO", "DÉCIMO QUINTO", "DÉCIMO SEXTO",
    "DÉCIMO SÉPTIMO", "DÉCIMO OCTAVO", "DÉCIMO NOVENO",
    "VIGÉSIMO",
]

def _build_turno_denomination(denominacion: str) -> str:
    """
    Strips the ordinal from a specific court name and replaces with 'EN TURNO'.
    E.g.  'JUZGADO CUARTO DE DISTRITO EN MATERIA DE AMPARO CIVIL...'
       -> 'JUZGADO DE DISTRITO EN MATERIA DE AMPARO CIVIL... EN TURNO'
    """
    name = denominacion.strip().rstrip(".")
    upper = name.upper()
    # Sort longest-first so 'DÉCIMO PRIMERO' is matched before 'DÉCIMO'
    for ordinal in sorted(_ORDINALS, key=len, reverse=True):
        pattern = f"JUZGADO {ordinal} DE DISTRITO"
        if pattern in upper:
            idx = upper.index(pattern)
            rest = name[idx + len(pattern):]
            return f"JUZGADO DE DISTRITO{rest} EN TURNO"
    # If no ordinal found, just append EN TURNO
    return f"{name} EN TURNO"


def _extract_building_address(direccion: str) -> str:
    """
    Strips floor/wing/piso details to extract the building-level address
    for the Oficialía de Partes Común.
    E.g.  'BLVD RUIZ CORTINES 2311-A (PISO 04; ALA "A") FRACC...' 
       -> 'BLVD RUIZ CORTINES 2311-A FRACC...'
    """
    # Remove parenthetical (PISO X; ALA Y) patterns
    cleaned = _re.sub(r'\s*\([^)]*(?:PISO|ALA|PLANTA)[^)]*\)\s*', ' ', direccion, flags=_re.IGNORECASE)
    # Remove standalone ", PISO X" or ", PLANTA BAJA" etc.
    cleaned = _re.sub(r',?\s*(?:PISO|PLANTA)\s+[^,]+,?', ', ', cleaned, flags=_re.IGNORECASE)
    # Remove EDIFICIO A/B/C/D/E references (individual buildings within complex)
    cleaned = _re.sub(r',?\s*EDIFICIO\s+[A-E]\b,?', ' ', cleaned, flags=_re.IGNORECASE)
    # Remove " ALA " references that may remain  
    cleaned = _re.sub(r',?\s*ALA\s+["\']?[A-Z]["\']?\s*,?', ' ', cleaned, flags=_re.IGNORECASE)
    # Clean up CORREO: email addresses (not useful for physical address)
    cleaned = _re.sub(r'CORREO:\s*\S+@\S+', '', cleaned, flags=_re.IGNORECASE)
    # Clean up double spaces and trailing commas
    cleaned = _re.sub(r'\s{2,}', ' ', cleaned).strip().rstrip(',')
    return cleaned


@app.get("/juzgados-distrito")
async def get_juzgados_distrito(
    estado: str = "",
    materia: str = "",
    limit: int = 50,
):
    """
    Devuelve info de Juzgados de Distrito para un estado.
    Incluye denominación 'en turno' y dirección de Oficialía de Partes.
    """
    if not supabase_admin:
        raise HTTPException(503, "Base de datos no configurada")

    try:
        query = supabase_admin.table("juzgados_distrito").select(
            "id,denominacion,materia,circuito,estado,ciudad,direccion,telefono"
        )

        if estado:
            query = query.eq("estado", estado)
        if materia:
            query = query.eq("materia", materia)

        result = query.limit(limit).execute()
        courts = result.data or []

        # Priority: amparo > Administrativa > Mixto > others
        amparo_courts = [c for c in courts if 'amparo' in c.get('denominacion', '').lower()]
        admin_courts  = [c for c in courts if c.get('materia') == 'Administrativa']
        mixto_courts  = [c for c in courts if c.get('materia') == 'Mixto']
        best = (amparo_courts or admin_courts or mixto_courts or courts)

        if best:
            chosen = best[0]
            turno = _build_turno_denomination(chosen["denominacion"])
            oficialia = _extract_building_address(chosen["direccion"])
            return {
                "total": len(courts),
                "estado": estado,
                "denominacion_turno": turno,
                "direccion_oficialia": oficialia,
                "telefono": chosen.get("telefono", ""),
                "nota": "La demanda se presenta ante la Oficialía de Partes Común de los Juzgados de Distrito. Funciona las 24 horas, los 365 días del año.",
                "juzgados": courts,
            }

        return {
            "total": 0,
            "estado": estado,
            "juzgados": [],
        }

    except Exception as e:
        print(f"❌ Error querying juzgados: {e}")
        raise HTTPException(500, f"Error al consultar juzgados: {str(e)}")



if __name__ == "__main__":
    import uvicorn
    
    print("═" * 60)
    print("  JUREXIA CORE API - Motor de Producción")
    print("═" * 60)
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
