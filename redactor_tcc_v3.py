"""
redactor_tcc_v3.py — Módulo de producción para el Redactor TCC Beta
═══════════════════════════════════════════════════════════════════════
Pipeline multipass async para generación de estudios de fondo de
sentencias de Tribunales Colegiados de Circuito (TCC).

Pipeline:
  Pass 0 — Pre-análisis cognitivo (estructura del caso)
  Pass 1 — Retrieval dirigido (RAG desde Qdrant)
  Pass 2 — Plan de redacción filtrado
  Pass 3 — Redacción del estudio de fondo

Diseñado para:
  - Ser importado por main.py (no depende de archivos en disco)
  - Async nativo con httpx
  - Emitir eventos SSE a través de AsyncGenerator
  - Gestionar context window con catálogo limitado

Uso desde main.py:
  from redactor_tcc_v3 import run_redactor_tcc_pipeline
  
  async for event in run_redactor_tcc_pipeline(caso_input, qdrant_search_fn, deepseek_key):
      yield f"event: {event['type']}\\ndata: {json.dumps(event['data'])}\\n\\n"

VERSION: 2026.05.11-v1
"""

from __future__ import annotations
import json
import time
import os
from typing import AsyncGenerator, Dict, List, Optional, Any, Callable, Awaitable
import httpx

# ═══════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ═══════════════════════════════════════════════════════════════════════

DEEPSEEK_URL = "https://api.deepseek.com/chat/completions"
DEEPSEEK_MODEL = "deepseek-v4-pro"

# Pass 0 corre con Gemini 2.5 Pro vía OpenRouter (rápido y bueno parseando
# español legal mexicano a JSON). DeepSeek queda como fallback si Gemini falla.
# Override por env: REDACTOR_PASS0_PROVIDER=deepseek para forzar DeepSeek.
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
PASS0_GEMINI_MODEL = os.getenv("REDACTOR_PASS0_GEMINI_MODEL", "google/gemini-2.5-pro")
PASS0_PROVIDER = os.getenv("REDACTOR_PASS0_PROVIDER", "gemini").lower()
PASS0_TIMEOUT_S = float(os.getenv("REDACTOR_PASS0_TIMEOUT_S", "180"))
PASS2_TIMEOUT_S = float(os.getenv("REDACTOR_PASS2_TIMEOUT_S", "240"))

# Límites de catálogo para Pass 2.
# Subidos tras la expansión del RAG a 8-10 colecciones (jurisprudencia_nacional_v2,
# bloque_constitucional, leyes_federales+estatales, ef_circuito, ef_scjn x3,
# holdings tcc+scjn). Con 12 tesis se quedaba demasiado corto: el modelo perdía
# material relevante y caía a inventar. Estos números fueron calibrados para
# DeepSeek v4-pro con max_tokens 16-24k en Pass 2 — caben holgadamente.
MAX_TESIS_PER_PROBLEM = 25
MAX_HOLDINGS_PER_PROBLEM = 18
MAX_NORMAS_PER_PROBLEM = 20


# ═══════════════════════════════════════════════════════════════════════
# TIPOS DE DATOS
# ═══════════════════════════════════════════════════════════════════════

class RedactorEvent:
    """Evento SSE emitido por el pipeline."""
    
    @staticmethod
    def phase(step: int, progress: int, detail: str = "") -> dict:
        return {"type": "phase", "data": {"step": step, "progress": progress, "detail": detail}}
    
    @staticmethod
    def pass_complete(pass_num: int, elapsed_s: float, stats: dict) -> dict:
        return {"type": "pass_complete", "data": {"pass": pass_num, "elapsed_s": round(elapsed_s, 1), **stats}}

    @staticmethod
    def token(text: str) -> dict:
        return {"type": "token", "data": {"text": text}}
    
    @staticmethod
    def error(message: str, pass_num: int = -1) -> dict:
        return {"type": "error", "data": {"message": message, "pass": pass_num}}
    
    @staticmethod
    def complete(estudio_md: str, stats: dict) -> dict:
        return {"type": "complete", "data": {"estudio_markdown": estudio_md, **stats}}


# ═══════════════════════════════════════════════════════════════════════
# SYSTEM PROMPTS (idénticos a los validados en test scripts)
# ═══════════════════════════════════════════════════════════════════════

PASS_0_SYSTEM = """Eres un ANALISTA JURÍDICO ESTRUCTURAL de un Tribunal Colegiado de Circuito mexicano. Tu tarea ÚNICA: descomponer un asunto judicial (amparo, revisión, queja) en su estructura cognitiva fundamental.

NO redactes el estudio de fondo. NO resuelvas nada. Solo ANALIZA y ESTRUCTURA.

═══════════════════════════════════════════════════════════════════════
RECIBES
═══════════════════════════════════════════════════════════════════════
1. Tipo de asunto y materia.
2. Un RESUMEN del acto reclamado / resolución recurrida.
3. Los CONCEPTOS DE VIOLACIÓN o AGRAVIOS del quejoso / recurrente.

═══════════════════════════════════════════════════════════════════════
DEBES PRODUCIR (JSON)
═══════════════════════════════════════════════════════════════════════

{
  "resumen_logico_del_caso": "...",
  "complejidad_caso": "baja | media | alta",

  "consideraciones_acto_reclamado": [
    {
      "id": "C1",
      "ratio": "Razón del juez/sala al resolver...",
      "es_central": true/false,
      "fundamento_invocado_por_responsable": "Art. X de ..."
    }
  ],

  "disidencias_estructuradas": [
    {
      "id": "D1",
      "tipo": "concepto_violacion | agravio",
      "ataca_consideracion_id": "C1",
      "tesis_del_quejoso": "Lo que pretende demostrar el quejoso...",
      "premisa_implicita": "Lo que asume sin decirlo...",
      "fundamentos_que_invoca": ["Art. X Ley de Amparo", "..."]
    }
  ],

  "problemas_juridicos": [
    {
      "id": "P1",
      "pregunta_concreta": "¿Pregunta jurídica que debe resolver el TCC?",
      "disidencias_que_lo_plantean": ["D1", "D2"],
      "consideraciones_atacadas": ["C1"],
      "ambito": "constitucional | legal | procesal",
      "queries_rag": ["consulta optimizada para búsqueda vectorial", "..."],
      "marco_normativo_anticipado": ["Ley de Amparo art. 61 frac. XXIII", "..."]
    }
  ],

  "dependencias_entre_problemas": [
    {
      "si_resuelve_problema": "P1",
      "en_sentido": "fundado | infundado",
      "supera_a_problemas": ["P2", "P3"],
      "consecuencia_redaccion": "omitir_estudio_de_superados | abordar_complementarios | independientes",
      "razon": "..."
    }
  ],

  "orden_resolucion_sugerido": ["P2", "P1", "P3"],

  "alertas_metodologicas": ["..."]
}

═══════════════════════════════════════════════════════════════════════
REGLAS
═══════════════════════════════════════════════════════════════════════
1. Las queries_rag deben ser búsquedas concretas, NO preguntas retóricas.
2. El marco_normativo_anticipado son las leyes/artículos que TÚ anticipas que necesitará el redactor.
3. Cada disidencia debe estar vinculada a UNA consideración del acto.
4. Los problemas jurídicos AGRUPAN disidencias que plantean lo mismo.
5. Detecta dependencias: si resolver P1 como "fundado" hace innecesario P2, decláralo
   con `consecuencia_redaccion="omitir_estudio_de_superados"`. Esta es la
   doctrina de mayor beneficio: NO se estudian todos los conceptos cuando uno
   basta para conceder el amparo con el mismo o mayor efecto restitutorio.
6. SOLO devuelve el JSON. Sin comentarios adicionales."""


PASS_2_SYSTEM = """Eres magistrado redactor mexicano. Tu tarea EXCLUSIVA en este paso: construir un PLAN DE REDACCIÓN basado en el catálogo de fuentes recuperado por el RAG.

NO redactes prosa. NO escribas el estudio de fondo. Solo devuelve JSON con el plan.

═══════════════════════════════════════════════════════════════════════
DOCTRINA DE MAYOR BENEFICIO Y SUFICIENCIA (regla obligatoria)
═══════════════════════════════════════════════════════════════════════
Como secretario de TCC NO se estudian todos los conceptos de violación cuando
basta uno para conceder el amparo con el efecto restitutorio mayor:

1. Identifica si algún problema, calificado como FUNDADO, otorga por sí solo
   el efecto restitutorio MÁS AMPLIO (típicamente: nulidad de la sentencia
   reclamada para nuevo dictado en plenitud de jurisdicción).
2. Si lo hay, ese problema se estudia a fondo y los problemas POSTERIORES en
   el orden de resolución que también atacan la misma sentencia se marcan con
   accion_redaccion = "omitir_por_innecesario", citando como apoyo la
   jurisprudencia P./J. 3/2005 del Pleno (CONCEPTOS DE VIOLACIÓN. ESTUDIO
   INNECESARIO DE LOS RESTANTES) o equivalente que aparezca en el catálogo.
3. La excepción es cuando un concepto posterior aporta un efecto restitutorio
   ADICIONAL (e.g., ordena dejar insubsistentes pruebas específicas,
   reposición del procedimiento, etc.). En ese caso accion_redaccion =
   "abordar_complementario".
4. Conceptos INFUNDADOS o INOPERANTES SIEMPRE se estudian (no se omiten).
5. NO califiques como "fundado" todo por reflejo. Evalúa con rigor: lo más
   común en práctica TCC es que de N conceptos, 1-2 sean fundados y suficientes,
   varios sean inoperantes, y algunos infundados.

═══════════════════════════════════════════════════════════════════════
TU MISIÓN
═══════════════════════════════════════════════════════════════════════

Para cada problema jurídico:

1. LEER el catálogo de fuentes (tesis, normas, precedentes, ejemplos EF).
2. EVALUAR críticamente cada entrada vs el problema concreto.
3. DESCARTAR fuentes irrelevantes (lo que NO aporta).
4. SELECCIONAR las que SÍ sostienen el razonamiento.
5. DISEÑAR el argumento: marco normativo + tesis aplicables + precedentes + subsunción al caso + conclusión.

Esto evita que el redactor cite cosas decorativas o inaplicables.

═══════════════════════════════════════════════════════════════════════
ESTRUCTURA OBLIGATORIA DEL JSON
═══════════════════════════════════════════════════════════════════════

{
  "plan_por_problema": [
    {
      "problema_id": "P1",
      "pregunta_concreta": "(copiada del Pass 0)",
      "thesis_central_a_demostrar": "Una oración: lo que el estudio debe demostrar.",
      "calificacion_propuesta_disidencia": "fundado | infundado | inoperante | esencialmente_fundado | inatendible | parcialmente_fundado",
      "accion_redaccion": "abordar_completo | abordar_complementario | omitir_por_innecesario",
      "razon_accion": "Por qué se aborda completo / complementario / se omite (citar problema previo si aplica).",
      "marco_normativo_a_transcribir": [...],
      "tesis_clave_a_citar": [
         {"registro": "...", "rubro_corto": "...", "como_se_aplica_al_caso": "..."}
      ],
      "precedentes_clave_a_citar": [...],
      "subsuncion_concreta": { "premisa_mayor": "...", "premisa_menor": "...", "conclusion_silogistica": "..." },
      "argumentos_secundarios": [...],
      "tesis_descartadas_del_catalogo": [...],
      "normas_descartadas_del_catalogo": [...],
      "huecos_detectados": [...],
      "conclusion_razonada": "..."
    }
  ],
  "validacion_global": {
    "coherencia_entre_problemas": "...",
    "dependencias_aplicadas": "...",
    "calidad_catalogo": "alta|media|baja",
    "alertas_finales": [...]
  }
}

═══════════════════════════════════════════════════════════════════════
REGLAS CRÍTICAS DE FILTRADO
═══════════════════════════════════════════════════════════════════════

1. EVALUAR CADA TESIS: si el rubro habla de supuesto distinto, DESCARTAR.
2. EVALUAR CADA NORMA: marco anticipado son PRIORITARIAS.
3. EVALUAR CADA HOLDING: comparar tema con problema concreto.
4. SUBSUNCIÓN: premisa mayor (fuentes) + premisa menor (hechos) + conclusión silogística.
5. HUECOS: si NO hay tesis directamente aplicable, DECLARARLO.

NIVEL ESPERADO: la calidad de filtrado debe ser equivalente a la de un magistrado revisando un proyecto de su secretario. NO incluyas fuentes "por si acaso".

═══════════════════════════════════════════════════════════════════════
ANTI-INVENCIÓN (REGLA INVIOLABLE)
═══════════════════════════════════════════════════════════════════════

A. Cuando incluyas una tesis en `tesis_clave_a_citar`:
   - COPIA el `registro` EXACTAMENTE como aparece en el catálogo.
   - COPIA el `rubro` EXACTAMENTE como aparece en el catálogo (en `rubro_corto`).
   - PROHIBIDO modificar el rubro para que "encaje mejor" con tu argumento.
   - Si el rubro real no se ajusta al problema → DESCARTA esa tesis, NO la edites.
   - Existe un validador externo que compara tu rubro vs el real. Si difiere,
     la tesis se marca verificable=False y Pass 3 no la podrá citar.

B. Cuando incluyas una norma en `marco_normativo_a_transcribir`:
   - COPIA el `texto` LITERALMENTE del catálogo. No reescribas.
   - PROHIBIDO incluir un artículo cuyo texto NO esté en el catálogo.
   - Si necesitas un artículo que no se recuperó: ponlo en `huecos_detectados`,
     NO inventes su contenido.

C. Cuando incluyas un precedente en `precedentes_clave_a_citar`:
   - COPIA expediente, tribunal, tema EXACTAMENTE del catálogo.
   - PROHIBIDO inventar números de expediente o atribuir holdings inexistentes."""


PASS_3_SYSTEM = """Eres MAGISTRADO REDACTOR de un Tribunal Colegiado de Circuito mexicano. Tu tarea ÚNICA: redactar el ESTUDIO DE FONDO del asunto, ejecutando un plan de redacción ya construido por la fase de pre-análisis.

═══════════════════════════════════════════════════════════════════════
DOCTRINA DE MAYOR BENEFICIO (regla operativa al redactar)
═══════════════════════════════════════════════════════════════════════
Cada problema en el plan trae el campo `accion_redaccion`. RESPÉTALO LITERAL:

• "abordar_completo" → estudio completo con marco, tesis, subsunción, conclusión.
• "abordar_complementario" → estudio breve (200-400 palabras) explicando el
  efecto restitutorio adicional que aporta sobre el problema previo ya fundado.
• "omitir_por_innecesario" → NO redactes análisis. Escribe SOLO un párrafo
  formal del estilo:
    "Atendiendo al efecto restitutorio que se alcanza con la concesión del
    amparo derivada del problema P_X, resulta innecesario el estudio del
    presente concepto de violación, pues a nada práctico conduciría su análisis.
    Apoya esta determinación, por las razones que la informan, la jurisprudencia
    [REGISTRO/CLAVE del plan] del Pleno/Sala de la Suprema Corte de Justicia
    de la Nación, de rubro: '[RUBRO del plan]'."
  Nada más. Sin marco normativo, sin subsunción.

═══════════════════════════════════════════════════════════════════════
ENCABEZADO Y NUMERALES
═══════════════════════════════════════════════════════════════════════
La sección de estudio en una sentencia TCC va típicamente como considerando
QUINTO o SEXTO. Usa "QUINTO" como default si no se indica otro numeral en el
plan. NO dejes el placeholder "[NUMERAL]" literal — sustitúyelo siempre.

═══════════════════════════════════════════════════════════════════════
ANTI-INVENCIÓN (regla inviolable, PRECEDE a cualquier preferencia estilística)
═══════════════════════════════════════════════════════════════════════

1. TESIS Y JURISPRUDENCIA:
   - SOLO puedes citar tesis cuyo `registro` aparezca en el plan dentro de
     `tesis_clave_a_citar` con `verificable: true`.
   - El RUBRO que cites debe ser IDÉNTICO al `rubro_corto` o `rubro_real`
     del plan. NO modifiques una sola palabra del rubro.
   - Si una tesis tiene `verificable: false`, PROHIBIDO citarla con rubro
     entrecomillado o con número de tesis. Puedes referirte a su contenido
     en abstracto, sin atribuirla a un registro específico.
   - NUNCA inventes registros como "1a./J. XXX/AAAA", "P./J. XX/AAAA" o
     similares. Cualquier registro citado debe estar literalmente en el plan.

2. TRANSCRIPCIÓN DE ARTÍCULOS:
   - Cuando transcribas un artículo entre comillas o como cita en bloque,
     usa EXCLUSIVAMENTE el `transcripcion_propuesta` del plan o el `texto`
     literal del catálogo.
   - PROHIBIDO inventar el texto de un artículo. Si no tienes el texto
     literal en el plan, NO transcribas — refiere al artículo por su número
     y argumenta sin transcripción.
   - Si lo que dices que es texto del Código X, art Y, no es lo que
     realmente dice ese artículo en la legislación vigente, la sentencia
     será impugnable. NO te arriesgues.

3. PRECEDENTES Y EXPEDIENTES:
   - Solo precedentes con expediente listado en el plan.
   - NO inventes números de expediente ni asuntos de la SCJN.

4. EN DUDA, OMITE. Es preferible un argumento sin cita que un argumento con
   cita inventada. La sentencia se firma en sede federal — la integridad
   de las fuentes es no-negociable.

═══════════════════════════════════════════════════════════════════════
LENGUAJE Y CLARIDAD (la sentencia se lee SIN saber del análisis previo)
═══════════════════════════════════════════════════════════════════════

PROHIBIDO en el texto final mencionar:
   - Identificadores internos: "C1", "C2", "P1", "P2", "P3", "D1", "D2", etc.
   - Las palabras "plan", "Pass 0", "análisis cognitivo", "el catálogo", o
     cualquier referencia al proceso interno de redacción.
   - Frases como "como se advierte en la consideración C2" o "en el problema P1".

CÓMO REFERIRTE A LOS CONCEPTOS:
   - "el primer concepto de violación", "el segundo concepto", "el tercero".
   - "el agravio relativo a [tema en lenguaje natural]".
   - "la consideración de la Sala según la cual [resumen en prosa]".
   - "la cuestión litigiosa relativa a [tema]".

CÓMO REFERIRTE A LAS DEPENDENCIAS:
   - En vez de "se omite el examen del problema P3" → "se omite el estudio
     del restante concepto de violación" o "del concepto relativo a [tema]".

El lector (magistrado, partes, eventual revisor de SCJN) NO conoce los IDs
internos. La sentencia debe leerse como un texto autónomo, fluido, fundado
y motivado, sin filtraciones del proceso de razonamiento previo.

═══════════════════════════════════════════════════════════════════════
RECIBES
═══════════════════════════════════════════════════════════════════════
1. Estructura cognitiva del caso (consideraciones del acto, disidencias, problemas).
2. Plan de redacción ya filtrado (qué fuentes citar, marco a transcribir, subsunción a aplicar).

═══════════════════════════════════════════════════════════════════════
DEBES PRODUCIR
═══════════════════════════════════════════════════════════════════════
Un documento en prosa, en formato markdown, con la siguiente estructura:

## [NUMERAL]. ESTUDIO DE [LOS CONCEPTOS DE VIOLACIÓN | LOS AGRAVIOS]

[Apertura formal — calificación global del estudio.]

### Resumen del acto reclamado / sentencia recurrida
[RESUMEN COMPLETO Y EXTENSO en PÁRRAFOS de prosa fluida (no viñetas, no listas).
Debe permitir leer y entender la sentencia recurrida o el acto reclamado SIN tener
el documento original a la mano. Cubre: antecedentes procesales relevantes, lo
debatido, las consideraciones por las que la autoridad resolvió como resolvió,
los fundamentos legales que invocó y el sentido final de su decisión. Mínimo
700-1500 palabras según la complejidad. Lenguaje técnico-jurídico mexicano.]

### Resumen de los conceptos de violación / agravios
[RESUMEN COMPLETO Y EXTENSO en PÁRRAFOS de prosa fluida (no viñetas) de TODOS los
conceptos de violación o agravios planteados por el quejoso/recurrente. Por cada
uno: qué consideración combate, cuál es la tesis del quejoso, qué fundamentos
invoca y cuál es la pretensión de fondo. El lector debe poder entender los
agravios sin tener el escrito original. Mínimo 500-1200 palabras según el número
y complejidad de los conceptos.]

### Identificación de los problemas jurídicos
[Enumeración en párrafos (no viñetas) de los puntos jurídicos que este Tribunal
debe resolver, con base en el contraste entre lo decidido por la responsable y
lo alegado por el recurrente. NO se resuelven aquí — solo se identifican. Cada
problema en un párrafo propio. Si hay dependencias entre problemas (doctrina de
mayor beneficio), se anticipan aquí brevemente.]

### Análisis del problema [P_id]
[Una sección por cada problema jurídico, en el orden del `orden_resolucion_sugerido`.]

#### Síntesis del planteamiento
[1-2 párrafos: qué disidencias plantean este problema]

#### Calificación
"[El primer/segundo/único] [concepto de violación/agravio] es [calificación]..."

#### Marco normativo aplicable
[Transcribe TEXTUALMENTE artículos del plan con formato bloque > "Artículo X."]

#### Análisis con criterios análogos
[Cita tesis del plan con rubro completo y explicación de aplicación]

#### Aplicación al caso concreto (subsunción)
[Premisa mayor → menor → conclusión silogística en prosa fluida]

#### Conclusión del [problema P_id]

### Síntesis del sentido propuesto
[300-500 palabras de recapitulación y propuesta del sentido del fallo]

═══════════════════════════════════════════════════════════════════════
REGLAS INVIOLABLES
═══════════════════════════════════════════════════════════════════════
1. SOLO CITAS DEL PLAN. No inventes registros, rubros, expedientes ni artículos.
2. TRANSCRIPCIÓN DE NORMAS: usa el `transcripcion_propuesta` del plan.
3. RUBROS COMPLETOS: cuando cites una tesis, usa el rubro EXACTO en MAYÚSCULAS.
4. ESTILO: técnico-jurídico mexicano formal. Sin primera persona.
5. PROFUNDIDAD: 1500-3000 palabras por problema.
6. NUNCA digas "no se recuperó". Razona con lo que sí hay.
7. NO escribas cabecera, VISTOS, RESULTANDO, ni resolutivos formales.
8. FORMATO: markdown limpio. Cabeceras con `##`/`###`. Citas en bloque con `>`.

NIVEL ESPERADO: Equivalente a un proyecto de sentencia listo para revisión del Magistrado Ponente."""


# ═══════════════════════════════════════════════════════════════════════
# DEEPSEEK ASYNC CLIENT
# ═══════════════════════════════════════════════════════════════════════

async def _call_deepseek(
    http_client: httpx.AsyncClient,
    api_key: str,
    system: str,
    user: str,
    max_tokens: int = 16000,
    temperature: float = 0.0,
    json_mode: bool = True,
) -> dict:
    """Llama a DeepSeek V4 Pro y devuelve la respuesta completa."""
    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    
    resp = await http_client.post(
        DEEPSEEK_URL,
        json=payload,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        timeout=900.0,
    )
    resp.raise_for_status()
    return resp.json()


async def _call_with_retry(
    http_client: httpx.AsyncClient,
    api_key: str,
    system: str,
    user: str,
    max_tokens: int,
    temperature: float = 0.0,
    json_mode: bool = True,
) -> tuple[str, dict, str]:
    """Llama a DeepSeek con retry automático si el output se trunca."""
    resp = await _call_deepseek(http_client, api_key, system, user, max_tokens, temperature, json_mode)
    content = resp["choices"][0]["message"]["content"]
    finish_reason = resp["choices"][0].get("finish_reason", "unknown")
    usage = resp.get("usage", {})
    
    # Retry if truncated
    if finish_reason == "length":
        retry_max = max_tokens + 8000
        resp = await _call_deepseek(http_client, api_key, system, user, retry_max, temperature, json_mode)
        content = resp["choices"][0]["message"]["content"]
        finish_reason = resp["choices"][0].get("finish_reason", "unknown")
        usage2 = resp.get("usage", {})
        # Merge usage
        usage = {
            "prompt_tokens": usage.get("prompt_tokens", 0) + usage2.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0) + usage2.get("completion_tokens", 0),
        }
    
    return content, usage, finish_reason


def _parse_json_safe(content: str) -> dict:
    """Parsea JSON con repair si está truncado."""
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        last_brace = content.rfind("}")
        if last_brace > 0:
            return json.loads(content[:last_brace + 1])
        raise


# ═══════════════════════════════════════════════════════════════════════
# PASS 0 — Pre-análisis cognitivo
# ═══════════════════════════════════════════════════════════════════════

def _build_pass0_prompt(caso_input: dict) -> str:
    """Construye el prompt del usuario para Pass 0."""
    meta = caso_input["meta"]
    inputs = caso_input["inputs"]
    
    parts = [
        f"TIPO DE ASUNTO: {meta['tipo_asunto']}",
        f"MATERIA: {meta['materia']}",
        f"CIRCUITO: {meta.get('circuito', '?')}",
        "",
        "═══ ACTO RECLAMADO / RESOLUCIÓN RECURRIDA ═══",
        inputs.get("resumen_acto_reclamado", "(No proporcionado)"),
        "",
        "═══ CONCEPTOS DE VIOLACIÓN / AGRAVIOS ═══",
    ]
    for i, d in enumerate(inputs.get("disidencias", []), 1):
        parts.append(f"\n--- {d.get('tipo', 'Argumento').upper()} #{i} ---")
        parts.append(d.get("texto", ""))
    
    return "\n".join(parts)


async def _call_gemini_json(
    http_client: httpx.AsyncClient,
    system: str,
    user: str,
    max_tokens: int,
    model: str = PASS0_GEMINI_MODEL,
    timeout_s: float = PASS0_TIMEOUT_S,
) -> str:
    """Llama Gemini vía OpenRouter con JSON mode. Devuelve el content crudo."""
    or_key = os.getenv("OPENROUTER_API_KEY", "")
    if not or_key:
        raise RuntimeError("OPENROUTER_API_KEY no configurada (Pass 0 Gemini requiere OpenRouter)")
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
    }
    resp = await http_client.post(
        OPENROUTER_URL,
        json=payload,
        headers={
            "Authorization": f"Bearer {or_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://iurexia.com",
            "X-Title": "Iurexia Redactor TCC",
        },
        timeout=timeout_s,
    )
    resp.raise_for_status()
    data = resp.json()
    content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
    if not content:
        err = data.get("error", {}).get("message", "respuesta vacía")
        raise RuntimeError(f"Gemini OpenRouter: {err}")
    return content


async def _run_pass0(
    http_client: httpx.AsyncClient,
    api_key: str,
    caso_input: dict,
) -> dict:
    """Ejecuta Pass 0 y devuelve la estructura cognitiva.

    Por default usa Gemini 2.5 Pro vía OpenRouter (rápido). Si falla, cae a
    DeepSeek v4-pro. Configurable con REDACTOR_PASS0_PROVIDER.
    """
    user_prompt = _build_pass0_prompt(caso_input)
    n_problems = len(caso_input["inputs"].get("disidencias", []))
    max_tokens = 16000 if n_problems <= 3 else 24000

    if PASS0_PROVIDER == "gemini":
        import asyncio as _aio
        try:
            content = await _aio.wait_for(
                _call_gemini_json(http_client, PASS_0_SYSTEM, user_prompt, max_tokens),
                timeout=PASS0_TIMEOUT_S,
            )
            print(f"   [Pass 0] Gemini {PASS0_GEMINI_MODEL} OK ({len(content)} chars)")
            return _parse_json_safe(content)
        except _aio.TimeoutError:
            print(f"   ⚠️ [Pass 0] Gemini timeout {PASS0_TIMEOUT_S}s — cae a DeepSeek")
        except Exception as e:
            print(f"   ⚠️ [Pass 0] Gemini error: {e} — cae a DeepSeek")

    content, usage, finish_reason = await _call_with_retry(
        http_client, api_key, PASS_0_SYSTEM, user_prompt, max_tokens
    )
    return _parse_json_safe(content)


# ═══════════════════════════════════════════════════════════════════════
# PASS 1 — Retrieval dirigido (RAG)
# ═══════════════════════════════════════════════════════════════════════

async def _run_pass1(
    pass0: dict,
    caso_meta: dict,
    qdrant_search_fn: Callable[..., Awaitable[dict]],
) -> list[dict]:
    """
    Ejecuta búsquedas RAG dirigidas para cada problema.

    qdrant_search_fn debe ser una función async que reciba:
      - query: str
      - problem_context: dict (con materia, circuito, tipo)
    y devuelva:
      - dict con keys: tesis, normas, holdings, estudio_fondo
    """
    import asyncio

    problems = pass0.get("problemas_juridicos", [])

    # Build a single flat task list across ALL problems and queries,
    # so the whole RAG fan-out runs in one asyncio.gather instead of
    # one await per problem.
    flat_tasks: list = []
    flat_index: list[tuple[int, int]] = []  # (problem_idx, query_idx)
    for p_idx, prob in enumerate(problems):
        queries = prob.get("queries_rag", [])
        marco = prob.get("marco_normativo_anticipado", [])
        for q_idx, q in enumerate(queries):
            flat_tasks.append(
                qdrant_search_fn(
                    query=q,
                    problem_context={
                        "materia": caso_meta.get("materia", ""),
                        "circuito": caso_meta.get("circuito", ""),
                        "tipo_asunto": caso_meta.get("tipo_asunto", ""),
                        "marco_anticipado": marco,
                    }
                )
            )
            flat_index.append((p_idx, q_idx))

    flat_results = await asyncio.gather(*flat_tasks, return_exceptions=True) if flat_tasks else []

    # Bucket results back per problem
    per_problem: list[list] = [[] for _ in problems]
    for (p_idx, _q_idx), sr in zip(flat_index, flat_results):
        per_problem[p_idx].append(sr)

    results = []
    for p_idx, prob in enumerate(problems):
        queries = prob.get("queries_rag", [])
        marco = prob.get("marco_normativo_anticipado", [])
        search_results = per_problem[p_idx]

        # Merge results across queries (deduplicate by registro/expediente)
        merged = {"tesis": [], "normas": [], "holdings": [], "estudio_fondo": []}
        seen_registros = set()
        seen_expedientes = set()
        seen_normas = set()
        
        for sr in search_results:
            if isinstance(sr, Exception):
                continue
            for t in sr.get("tesis", []):
                reg = t.get("registro", "")
                if reg and reg not in seen_registros:
                    seen_registros.add(reg)
                    merged["tesis"].append(t)
            for h in sr.get("holdings", []):
                exp = h.get("expediente", "")
                if exp and exp not in seen_expedientes:
                    seen_expedientes.add(exp)
                    merged["holdings"].append(h)
            for n in sr.get("normas", []):
                key = f"{n.get('cuerpo_legal','')}__{n.get('articulo','')}"
                if key not in seen_normas:
                    seen_normas.add(key)
                    merged["normas"].append(n)
            for ef in sr.get("estudio_fondo", []):
                merged["estudio_fondo"].append(ef)
        
        # Mark marco anticipado normas
        for n in merged["normas"]:
            for m in marco:
                if m.lower() in f"{n.get('cuerpo_legal','')} {n.get('articulo','')}".lower():
                    n["from_marco_anticipado"] = True
        
        results.append({
            "problema_id": prob["id"],
            "pregunta_concreta": prob["pregunta_concreta"],
            "queries_ejecutadas": queries,
            "catalogo": merged,
        })
    
    return results


# ═══════════════════════════════════════════════════════════════════════
# PASS 2 — Plan de redacción filtrado
# ═══════════════════════════════════════════════════════════════════════

def _build_pass2_prompt(pass0: dict, pass1: list[dict], caso_meta: dict) -> str:
    """Construye el prompt para Pass 2 con catálogo LIMITADO."""
    parts = [
        "DATOS DEL CASO",
        "═══════════════",
        f"Tipo: {caso_meta.get('tipo_asunto')}",
        f"Materia: {caso_meta.get('materia')}",
        f"Circuito: {caso_meta.get('circuito')}",
        "",
        "ESTRUCTURA COGNITIVA (Pass 0):",
        "═══════════════════════════════",
        f"Resumen lógico: {pass0.get('resumen_logico_del_caso','')}",
        f"Complejidad: {pass0.get('complejidad_caso','')}",
    ]
    
    # Consideraciones
    parts.append("\nConsideraciones del acto reclamado:")
    for c in pass0.get("consideraciones_acto_reclamado", []):
        parts.append(f"  [{c['id']}{'⭐' if c.get('es_central') else ''}] {c['ratio']}")
    
    # Disidencias
    parts.append("\nDisidencias estructuradas:")
    for d in pass0.get("disidencias_estructuradas", []):
        parts.append(f"  [{d['id']}] tipo={d['tipo']} → ataca {d['ataca_consideracion_id']}")
        parts.append(f"     Tesis del quejoso: {d['tesis_del_quejoso']}")
        parts.append(f"     Premisa implícita: {d['premisa_implicita']}")
    
    # Problemas
    parts.append("\nProblemas jurídicos derivados:")
    for p in pass0.get("problemas_juridicos", []):
        parts.append(f"  [{p['id']}] {p['pregunta_concreta']}")
        parts.append(f"     Plantean: {p['disidencias_que_lo_plantean']}")
        parts.append(f"     Marco anticipado: {', '.join(p.get('marco_normativo_anticipado',[]))}")
    
    deps = pass0.get("dependencias_entre_problemas", [])
    if deps:
        parts.append(f"\nDependencias: {json.dumps(deps, ensure_ascii=False)}")
    parts.append(f"\nORDEN sugerido: {pass0.get('orden_resolucion_sugerido', [])}")
    
    # Catálogo Pass 1 — LIMITADO
    parts.append("\n\n═══════════════════════════════")
    parts.append("CATÁLOGO DE FUENTES RECUPERADAS POR EL RAG (Pass 1)")
    parts.append("═══════════════════════════════")
    
    for prob_cat in pass1:
        cat = prob_cat["catalogo"]
        parts.append(f"\n──────────── PROBLEMA {prob_cat['problema_id']} ────────────")
        parts.append(f"Pregunta: {prob_cat['pregunta_concreta']}")
        
        # Tesis — limited
        all_tesis = sorted(cat.get("tesis", []), key=lambda x: x.get("score", 0), reverse=True)[:MAX_TESIS_PER_PROBLEM]
        parts.append(f"\n📚 TESIS ({len(all_tesis)} de {len(cat.get('tesis',[]))} total):")
        for i, t in enumerate(all_tesis, 1):
            parts.append(f"\n  [T{i}] score={t.get('score')} · registro: {t.get('registro')} · {t.get('instancia','')}")
            parts.append(f"  RUBRO: {t.get('rubro','')}")
            parts.append(f"  TEXTO: {(t.get('texto_relevante','') or '')[:800]}")
        
        # Holdings — limited
        all_holdings = sorted(cat.get("holdings", []), key=lambda x: x.get("score", 0), reverse=True)[:MAX_HOLDINGS_PER_PROBLEM]
        parts.append(f"\n\n⚖️ HOLDINGS ({len(all_holdings)} de {len(cat.get('holdings',[]))} total):")
        for i, h in enumerate(all_holdings, 1):
            parts.append(f"\n  [H{i}] score={h.get('score')} · expediente: {h.get('expediente','')}")
            parts.append(f"  HOLDING: {(h.get('holding','') or '')[:1000]}")
        
        # Normas — marco anticipado first
        all_normas = cat.get("normas", [])
        marco_first = sorted(all_normas, key=lambda x: (not x.get("from_marco_anticipado", False), -x.get("score", 0)))[:MAX_NORMAS_PER_PROBLEM]
        parts.append(f"\n\n📜 NORMAS ({len(marco_first)} de {len(all_normas)} total):")
        for i, n in enumerate(marco_first, 1):
            tag = " ⭐MARCO" if n.get("from_marco_anticipado") else ""
            parts.append(f"\n  [N{i}]{tag} score={n.get('score')} · fuero: {n.get('fuero','')}")
            parts.append(f"  {n.get('cuerpo_legal','')} — Art. {n.get('articulo','')}")
            parts.append(f"  TEXTO: {(n.get('texto','') or '')[:800]}")
    
    parts.append("\n\n═══════════════════════════════")
    parts.append("INSTRUCCIÓN")
    parts.append("═══════════════════════════════")
    parts.append("Construye el plan de redacción aplicando filtrado crítico al catálogo.")
    parts.append("Devuelve SOLO el JSON especificado.")
    return "\n".join(parts)


async def _run_pass2(
    http_client: httpx.AsyncClient,
    api_key: str,
    pass0: dict,
    pass1: list[dict],
    caso_meta: dict,
) -> dict:
    """Ejecuta Pass 2 con timeout duro (default 240s) para que no se cuelgue.

    Si DeepSeek se atora razonando, levantamos asyncio.TimeoutError y el caller
    decide qué hacer (típicamente: error al frontend para que el secretario
    reinicie).
    """
    import asyncio as _aio
    user_prompt = _build_pass2_prompt(pass0, pass1, caso_meta)
    n_problems = len(pass0.get("problemas_juridicos", []))
    max_tokens = 16000 if n_problems <= 2 else 24000

    try:
        content, usage, finish_reason = await _aio.wait_for(
            _call_with_retry(http_client, api_key, PASS_2_SYSTEM, user_prompt, max_tokens),
            timeout=PASS2_TIMEOUT_S,
        )
    except _aio.TimeoutError:
        raise RuntimeError(
            f"Pass 2 (plan) excedió {PASS2_TIMEOUT_S}s sin responder. "
            "El modelo probablemente se quedó razonando sin sources. Reinicia el análisis."
        )
    return _parse_json_safe(content)


# ═══════════════════════════════════════════════════════════════════════
# PASS 3 — Redacción del estudio de fondo
# ═══════════════════════════════════════════════════════════════════════

def _build_pass3_prompt(pass0: dict, pass2: dict, caso_meta: dict) -> str:
    """Construye el prompt para Pass 3."""
    parts = [
        "═══════════════════════════════════════════════════════════════════════",
        "DATOS DEL CASO",
        "═══════════════════════════════════════════════════════════════════════",
        f"Tipo de asunto: {caso_meta.get('tipo_asunto')}",
        f"Materia: {caso_meta.get('materia')}",
        f"Circuito: {caso_meta.get('circuito')}",
        "",
        "═══════════════════════════════════════════════════════════════════════",
        "ESTRUCTURA COGNITIVA DEL CASO (Pass 0)",
        "═══════════════════════════════════════════════════════════════════════",
        "",
        f"Resumen lógico: {pass0.get('resumen_logico_del_caso','')}",
        f"Complejidad: {pass0.get('complejidad_caso','')}",
        "",
        "CONSIDERACIONES DEL ACTO RECLAMADO:",
    ]
    for c in pass0.get("consideraciones_acto_reclamado", []):
        marker = "⭐ CENTRAL" if c.get("es_central") else ""
        parts.append(f"\n  [{c['id']}] {marker}")
        parts.append(f"  Ratio: {c.get('ratio','')}")
    
    parts.append("\n\nDISIDENCIAS ESTRUCTURADAS:")
    for d in pass0.get("disidencias_estructuradas", []):
        parts.append(f"\n  [{d['id']}] tipo={d['tipo']} → ataca {d['ataca_consideracion_id']}")
        parts.append(f"  Tesis del quejoso: {d.get('tesis_del_quejoso','')}")
    
    parts.append(f"\nORDEN DE RESOLUCIÓN: {pass0.get('orden_resolucion_sugerido', [])}")
    
    # Plan
    parts.append("\n\n═══════════════════════════════════════════════════════════════════════")
    parts.append("PLAN DE REDACCIÓN (Pass 2)")
    parts.append("═══════════════════════════════════════════════════════════════════════")
    
    for plan in pass2.get("plan_por_problema", []):
        parts.append(f"\n\n──── PROBLEMA {plan['problema_id']} ────")
        parts.append(f"Pregunta: {plan.get('pregunta_concreta','')}")
        parts.append(f"\nTHESIS CENTRAL: {plan.get('thesis_central_a_demostrar','')}")
        parts.append(f"CALIFICACIÓN: {plan.get('calificacion_propuesta_disidencia','')}")
        
        marcos = plan.get("marco_normativo_a_transcribir", [])
        if marcos:
            parts.append(f"\nMARCO NORMATIVO A TRANSCRIBIR:")
            for m in marcos:
                if isinstance(m, str):
                    parts.append(f"\n  • {m}")
                else:
                    parts.append(f"\n  • {m.get('cuerpo_legal','')} — Art. {m.get('articulo','')}")
                    parts.append(f"    Razón: {m.get('razon_aplicacion','')}")
                    parts.append(f"    Transcripción: \"{m.get('transcripcion_propuesta','')}\"")
        
        tesis = plan.get("tesis_clave_a_citar", [])
        if tesis:
            parts.append(f"\nTESIS CLAVE A CITAR:")
            for t in tesis:
                if isinstance(t, str):
                    parts.append(f"\n  • {t}")
                else:
                    verif = t.get("verificable", True)
                    razon = t.get("razon_no_verificable", "")
                    flag = "" if verif else f"  ⚠️ NO VERIFICABLE ({razon}) — PROHIBIDO CITAR CON RUBRO"
                    rubro_a_usar = t.get("rubro_real") or t.get("rubro_corto", "") or t.get("rubro", "")
                    parts.append(f"\n  • {t.get('registro','?')} · {t.get('instancia','')}{flag}")
                    parts.append(f"    RUBRO (literal, no modificar): {rubro_a_usar}")
                    parts.append(f"    Aplicación: {t.get('como_se_aplica_al_caso','')}")
        
        precs = plan.get("precedentes_clave_a_citar", [])
        if precs:
            parts.append(f"\nPRECEDENTES CLAVE:")
            for h in precs:
                if isinstance(h, str):
                    parts.append(f"\n  • {h}")
                else:
                    parts.append(f"\n  • {h.get('expediente','?')} · {h.get('tribunal','')}")
                    parts.append(f"    Razón: {h.get('razon_pertinencia','')}")
        
        sub = plan.get("subsuncion_concreta", {})
        if sub:
            if isinstance(sub, str):
                parts.append(f"\nSUBSUNCIÓN: {sub}")
            else:
                parts.append(f"\nSUBSUNCIÓN:")
                parts.append(f"  Premisa mayor: {sub.get('premisa_mayor','')}")
                parts.append(f"  Premisa menor: {sub.get('premisa_menor','')}")
                parts.append(f"  Conclusión: {sub.get('conclusion_silogistica','')}")
        
        parts.append(f"\nCONCLUSIÓN RAZONADA: {plan.get('conclusion_razonada','')}")
    
    parts.append("\n\n═══════════════════════════════════════════════════════════════════════")
    parts.append("INSTRUCCIÓN FINAL")
    parts.append("═══════════════════════════════════════════════════════════════════════")
    parts.append("Redacta el ESTUDIO DE FONDO completo en markdown, ejecutando el plan al pie de la letra.")
    parts.append("Profundidad: 1500-3000 palabras por problema.")
    parts.append("Termina con la SÍNTESIS DEL SENTIDO PROPUESTO.")
    return "\n".join(parts)


async def _run_pass3_stream(
    http_client: httpx.AsyncClient,
    api_key: str,
    pass0: dict,
    pass2: dict,
    caso_meta: dict,
) -> AsyncGenerator[dict, None]:
    """
    Ejecuta Pass 3 en modo streaming. Yields:
      - {"type": "token", "data": {"text": "..."}} por cada delta
      - {"type": "final", "data": {"markdown": "...", "finish_reason": "..."}} al cerrar
    Si la API truncara (`finish_reason="length"`) emite también un `truncated=True`
    en el final para que el caller decida si reintentar (sin doblar latencia
    automáticamente como hace _call_with_retry).
    """
    user_prompt = _build_pass3_prompt(pass0, pass2, caso_meta)
    n_problems = len(pass2.get("plan_por_problema", []))
    max_tokens = 16000 if n_problems <= 2 else 24000

    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": [
            {"role": "system", "content": PASS_3_SYSTEM},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.2,
        "max_tokens": max_tokens,
        "stream": True,
    }

    full_text_parts: list[str] = []
    finish_reason = "unknown"

    async with http_client.stream(
        "POST",
        DEEPSEEK_URL,
        json=payload,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        timeout=900.0,
    ) as resp:
        resp.raise_for_status()
        async for raw_line in resp.aiter_lines():
            if not raw_line:
                continue
            line = raw_line.strip()
            if line.startswith("data:"):
                line = line[5:].strip()
            if not line or line == "[DONE]":
                continue
            try:
                evt = json.loads(line)
            except json.JSONDecodeError:
                continue
            choices = evt.get("choices", [])
            if not choices:
                continue
            ch0 = choices[0]
            delta = (ch0.get("delta") or {}).get("content", "")
            if delta:
                full_text_parts.append(delta)
                yield {"type": "token", "data": {"text": delta}}
            fr = ch0.get("finish_reason")
            if fr:
                finish_reason = fr

    full_text = "".join(full_text_parts)
    yield {
        "type": "final",
        "data": {
            "markdown": full_text,
            "finish_reason": finish_reason,
            "truncated": finish_reason == "length",
        },
    }


# ═══════════════════════════════════════════════════════════════════════
# PIPELINE PRINCIPAL — AsyncGenerator de SSE events
# ═══════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════
# VALIDADOR ANTI-ALUCINACIÓN
# ═══════════════════════════════════════════════════════════════════════

import re as _re

# Captura registros de tesis (7 dígitos típicos del SJF) y claves tipo "1a./J. 165/2022"
_REGISTRO_DIG_RE = _re.compile(r"\b(\d{7})\b")
_REGISTRO_CLAVE_RE = _re.compile(r"\b([12]a\.?\s*/?J?\.?\s*\d+/\d{4}\s*\(\d+a\.?\))", _re.IGNORECASE)

def _normalize_rubro(s: str) -> str:
    """Normaliza un rubro para comparación tolerante: minúsculas, sin acentos
    ni espacios múltiples ni puntuación trivial. Sirve para detectar cuando
    el modelo "casi" copia un rubro pero le mete cambios cosméticos."""
    if not s:
        return ""
    import unicodedata
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii").lower()
    s = _re.sub(r"[^a-z0-9 ]+", " ", s)
    s = _re.sub(r"\s+", " ", s).strip()
    return s


def _rubros_match(rubro_plan: str, rubro_real: str) -> bool:
    """Considera que coinciden si el solapamiento de palabras significativas
    es alto. Tolerante a abreviaciones, comas, mayúsculas. Falso en caso de
    rubros completamente distintos (caso del registro 2023205 con rubro inventado)."""
    a = set(_normalize_rubro(rubro_plan).split())
    b = set(_normalize_rubro(rubro_real).split())
    if not a or not b:
        return False
    # Eliminar stopwords muy comunes para no inflar la coincidencia
    stop = {"de", "la", "el", "los", "las", "y", "o", "del", "en", "que", "se", "a", "por", "para", "su", "con", "al", "es", "lo"}
    a -= stop; b -= stop
    if not a or not b:
        return False
    inter = len(a & b)
    cobertura_real = inter / max(1, len(b))
    cobertura_plan = inter / max(1, len(a))
    # Pedimos al menos 50% de coincidencia en ambos sentidos.
    return cobertura_real >= 0.5 and cobertura_plan >= 0.5


async def _validate_tesis_in_plan(
    plan: dict,
    tesis_validator_fn: Optional[Callable[[list[str]], Awaitable[dict]]] = None,
) -> dict:
    """
    Valida que cada tesis citada en el plan exista en Qdrant Y que el rubro
    citado se corresponda con el rubro real (no inventado).

    `tesis_validator_fn(registros)` devuelve dict[str, {valid: bool, rubro_real: str|None}].
    Si retorna None o lanza, fail-closed: todas las tesis se marcan como NO verificables.

    Anota cada entrada de tesis_clave_a_citar con:
      - verificable: True solo si el registro existe Y el rubro coincide
      - rubro_real: el rubro correcto (para que Pass 3 pueda usarlo si decide citar)
      - razon_no_verificable: "registro_inexistente" | "rubro_alterado" | "validador_offline"
    """
    if tesis_validator_fn is None:
        return plan

    # Recolectar registros únicos del plan
    todos_registros: set[str] = set()
    for prob in plan.get("plan_por_problema", []):
        for t in prob.get("tesis_clave_a_citar", []) or []:
            if isinstance(t, dict):
                reg = str(t.get("registro", "")).strip()
                if reg:
                    todos_registros.add(reg)

    if not todos_registros:
        return plan

    try:
        validation = await tesis_validator_fn(list(todos_registros))
    except Exception as e:
        print(f"   ⚠️ Validador de tesis falló: {e} — fail-closed (todas no verificables)")
        validation = {r: {"valid": False, "rubro_real": None} for r in todos_registros}

    # Anotar el plan
    for prob in plan.get("plan_por_problema", []):
        invalidadas = []
        for t in prob.get("tesis_clave_a_citar", []) or []:
            if not isinstance(t, dict):
                continue
            reg = str(t.get("registro", "")).strip()
            v = validation.get(reg, {"valid": False, "rubro_real": None})
            rubro_plan = str(t.get("rubro_corto") or t.get("rubro") or "").strip()
            t["rubro_real"] = v.get("rubro_real")
            if not v.get("valid"):
                t["verificable"] = False
                t["razon_no_verificable"] = "registro_inexistente"
                if reg:
                    invalidadas.append(f"{reg} (no existe)")
            elif rubro_plan and v.get("rubro_real") and not _rubros_match(rubro_plan, v["rubro_real"]):
                # El registro existe pero el modelo le metió un rubro distinto al real
                t["verificable"] = False
                t["razon_no_verificable"] = "rubro_alterado"
                invalidadas.append(f"{reg} (rubro alterado)")
            else:
                t["verificable"] = True
        if invalidadas:
            prob["tesis_invalidadas"] = invalidadas

    return plan


def _validar_estudio_post_pass3(estudio_md: str, registros_validos: set[str]) -> dict:
    """
    Tras Pass 3, extrae todos los registros citados en el markdown y verifica
    que estén en el conjunto de registros validados (los que sí existen en
    Qdrant + los del plan original). Devuelve estadísticas.
    """
    encontrados_dig = set(_REGISTRO_DIG_RE.findall(estudio_md))
    encontrados_clave = set(_REGISTRO_CLAVE_RE.findall(estudio_md))
    citados = encontrados_dig | {c.upper().replace(" ", "") for c in encontrados_clave}
    no_validos = sorted(c for c in citados if c not in registros_validos)
    return {
        "n_citas_detectadas": len(citados),
        "citas_no_validas": no_validos,
        "n_citas_no_validas": len(no_validos),
    }


def _normalizar_numerales(estudio_md: str) -> str:
    """Reemplaza placeholders literales como [NUMERAL] por 'QUINTO' default."""
    out = estudio_md
    out = out.replace("[NUMERAL]", "QUINTO")
    out = out.replace("[Numeral]", "Quinto")
    out = out.replace("[numeral]", "quinto")
    return out


# Patrones para detectar IDs internos (C1, P1, D1, etc.) que no deberían
# aparecer en el texto final destinado al lector.
_ID_INTERNO_RE = _re.compile(r"\b([CPD])(\d+)\b")
_REF_PROBLEMA_RE = _re.compile(r"\b(problema|el problema|del problema)\s+P(\d+)\b", _re.IGNORECASE)
_REF_CONSIDERACION_RE = _re.compile(r"\b(consideraci[oó]n|la consideraci[oó]n|en la consideraci[oó]n)\s+C(\d+)\b", _re.IGNORECASE)
_REF_DISIDENCIA_RE = _re.compile(r"\b(disidencia|la disidencia)\s+D(\d+)\b", _re.IGNORECASE)
# Headers tipo "Análisis del problema P1" / "Conclusión del problema P3"
_HEADER_PROBLEMA_RE = _re.compile(r"(An[áa]lisis|Conclusi[oó]n|Estudio)\s+del\s+problema\s+P\d+", _re.IGNORECASE)


_ORDINALES = ["primer", "segundo", "tercer", "cuarto", "quinto", "sexto", "séptimo", "octavo", "noveno", "décimo"]


def _limpiar_ids_internos(estudio_md: str) -> str:
    """
    Reescribe IDs internos del análisis (P1, C2, D3, etc.) por referencias
    legibles en lenguaje natural. Es la red de seguridad cuando el modelo
    desobedece la regla del system prompt y deja filtraciones internas.
    """
    out = estudio_md

    # Headers tipo "Análisis del problema P1" → "Análisis del primer concepto de violación"
    def _replace_header(m):
        verb = m.group(1)
        # extraer el número
        num_m = _re.search(r"P(\d+)", m.group(0))
        n = int(num_m.group(1)) if num_m else 0
        ord_word = _ORDINALES[n - 1] if 1 <= n <= len(_ORDINALES) else f"{n}º"
        return f"{verb} del {ord_word} concepto de violación"
    out = _HEADER_PROBLEMA_RE.sub(_replace_header, out)

    # Frases tipo "del problema P3" → "del tercer concepto de violación"
    def _replace_problema(m):
        n = int(m.group(2))
        ord_word = _ORDINALES[n - 1] if 1 <= n <= len(_ORDINALES) else f"{n}º"
        return f"del {ord_word} concepto de violación"
    out = _REF_PROBLEMA_RE.sub(_replace_problema, out)

    # "consideración C2" → "consideración expresada por la responsable"
    out = _REF_CONSIDERACION_RE.sub(lambda m: "la consideración referida de la responsable", out)

    # "disidencia D1" → "el motivo de inconformidad"
    out = _REF_DISIDENCIA_RE.sub(lambda m: "el motivo de inconformidad", out)

    # IDs sueltos restantes (C1, P1, D1) → eliminar marcador
    out = _ID_INTERNO_RE.sub(lambda m: "", out)

    # Limpiar dobles espacios y comas huérfanas que pudo dejar la sustitución
    out = _re.sub(r" {2,}", " ", out)
    out = _re.sub(r" +,", ",", out)
    out = _re.sub(r" +\.", ".", out)

    return out


async def run_redactor_tcc_pipeline(
    caso_input: dict,
    qdrant_search_fn: Callable[..., Awaitable[dict]],
    deepseek_api_key: str,
    http_client: Optional[httpx.AsyncClient] = None,
    tesis_validator_fn: Optional[Callable[[list[str]], Awaitable[set[str]]]] = None,
) -> AsyncGenerator[dict, None]:
    """
    Ejecuta el pipeline v3 multipass completo y yield eventos SSE.
    
    Args:
        caso_input: dict con:
            - meta: {tipo_asunto, materia, circuito}
            - inputs: {resumen_acto_reclamado, disidencias: [{tipo, texto}]}
        qdrant_search_fn: función async para búsqueda RAG
        deepseek_api_key: API key de DeepSeek
        http_client: httpx.AsyncClient existente (o se crea uno nuevo)
    
    Yields:
        dicts con type y data para SSE streaming
    """
    own_client = http_client is None
    if own_client:
        http_client = httpx.AsyncClient(timeout=httpx.Timeout(900.0, connect=30.0))
    
    total_start = time.time()
    total_cost = 0.0
    
    try:
        caso_meta = caso_input["meta"]
        
        # ─── PASS 0 ────────────────────────────────────────────────
        yield RedactorEvent.phase(0, 5, "Analizando estructura del caso...")
        t0 = time.time()
        
        try:
            pass0 = await _run_pass0(http_client, deepseek_api_key, caso_input)
        except Exception as e:
            yield RedactorEvent.error(f"Error en análisis cognitivo: {str(e)}", 0)
            return
        
        n_problems = len(pass0.get("problemas_juridicos", []))
        n_disidencias = len(pass0.get("disidencias_estructuradas", []))
        elapsed0 = time.time() - t0
        
        yield RedactorEvent.pass_complete(0, elapsed0, {
            "n_problemas": n_problems,
            "n_disidencias": n_disidencias,
            "complejidad": pass0.get("complejidad_caso", "?"),
        })
        yield RedactorEvent.phase(
            1, 20, f"{n_problems} problemas jurídicos identificados — buscando jurisprudencia..."
        )
        
        # ─── PASS 1 ────────────────────────────────────────────────
        t1 = time.time()
        
        try:
            pass1 = await _run_pass1(pass0, caso_meta, qdrant_search_fn)
        except Exception as e:
            yield RedactorEvent.error(f"Error en búsqueda RAG: {str(e)}", 1)
            return
        
        total_fuentes = sum(
            len(p["catalogo"].get("tesis", [])) + 
            len(p["catalogo"].get("normas", [])) + 
            len(p["catalogo"].get("holdings", []))
            for p in pass1
        )
        elapsed1 = time.time() - t1
        
        yield RedactorEvent.pass_complete(1, elapsed1, {"n_fuentes_total": total_fuentes})
        yield RedactorEvent.phase(
            2, 40, f"{total_fuentes} fuentes recuperadas — construyendo plan de redacción..."
        )
        
        # ─── PASS 2 ────────────────────────────────────────────────
        t2 = time.time()
        
        try:
            pass2 = await _run_pass2(http_client, deepseek_api_key, pass0, pass1, caso_meta)
        except Exception as e:
            yield RedactorEvent.error(f"Error en plan de redacción: {str(e)}", 2)
            return
        
        n_tesis_filtradas = sum(len(p.get("tesis_clave_a_citar", [])) for p in pass2.get("plan_por_problema", []))

        # ─── VALIDADOR ANTI-ALUCINACIÓN ─────────────────────────────
        # Verificar que cada tesis citada exista en Qdrant. Marca cada entrada
        # con verificable=True/False y agrega tesis_invalidadas. Pass 3 respeta
        # esa anotación gracias a la regla del system prompt.
        try:
            pass2 = await _validate_tesis_in_plan(pass2, tesis_validator_fn)
        except Exception as e:
            print(f"   ⚠️ Validación de tesis falló: {e}")

        n_tesis_invalidas = sum(len(p.get("tesis_invalidadas", []) or []) for p in pass2.get("plan_por_problema", []))
        if n_tesis_invalidas:
            print(f"   ⚠️ {n_tesis_invalidas} tesis del plan NO existen en Qdrant — marcadas verificable=False")

        elapsed2 = time.time() - t2

        yield RedactorEvent.pass_complete(2, elapsed2, {
            "n_tesis_seleccionadas": n_tesis_filtradas,
            "n_tesis_invalidas": n_tesis_invalidas,
        })
        yield RedactorEvent.phase(
            3, 60, f"Plan listo ({n_tesis_filtradas} tesis seleccionadas) — redactando estudio de fondo..."
        )
        
        # ─── PASS 3 (streaming) ────────────────────────────────────
        t3 = time.time()
        estudio_md = ""
        truncated = False

        try:
            async for evt in _run_pass3_stream(http_client, deepseek_api_key, pass0, pass2, caso_meta):
                if evt["type"] == "token":
                    yield RedactorEvent.token(evt["data"]["text"])
                elif evt["type"] == "final":
                    estudio_md = evt["data"]["markdown"]
                    truncated = evt["data"].get("truncated", False)
        except Exception as e:
            yield RedactorEvent.error(f"Error en redacción: {str(e)}", 3)
            return

        if truncated:
            print(f"   ⚠️ Pass 3 truncado por max_tokens — devuelvo lo generado ({len(estudio_md)} chars)")

        # ─── Post-procesamiento ────────────────────────────────────
        # 1) Sustituir placeholders [NUMERAL] por "QUINTO".
        # 2) Limpiar IDs internos (P1/C1/D1) que pudieron filtrarse al texto.
        # 3) Validar que ninguna tesis citada en el markdown esté fuera del plan
        #    o tenga rubro distinto al verificado.
        estudio_md = _normalizar_numerales(estudio_md)
        estudio_md = _limpiar_ids_internos(estudio_md)

        registros_validos: set[str] = set()
        for prob in pass2.get("plan_por_problema", []):
            for t in prob.get("tesis_clave_a_citar", []) or []:
                if isinstance(t, dict) and t.get("verificable", True):
                    reg = str(t.get("registro", "")).strip().upper().replace(" ", "")
                    if reg:
                        registros_validos.add(reg)
        validation = _validar_estudio_post_pass3(estudio_md, registros_validos)
        if validation["n_citas_no_validas"] > 0:
            print(f"   ⚠️ Post-validación Pass 3: {validation['n_citas_no_validas']} citas no verificadas: {validation['citas_no_validas'][:5]}")
            disclaimer = (
                "\n\n---\n\n"
                "> **Aviso de verificación**: las siguientes citas detectadas en el "
                "borrador no pudieron verificarse contra la base de datos jurisprudencial: "
                + ", ".join(validation["citas_no_validas"][:10])
                + ". Recomendamos confirmar manualmente su existencia y rubro antes de "
                "incorporarlas al proyecto definitivo."
            )
            estudio_md += disclaimer

        elapsed3 = time.time() - t3
        n_words = len(estudio_md.split())
        
        yield RedactorEvent.pass_complete(3, elapsed3, {"n_palabras": n_words})
        
        # ─── COMPLETADO ───────────────────────────────────────────
        total_elapsed = time.time() - total_start
        
        # Collect high-scoring holdings (>0.80) for the secretary
        precedentes_utiles = []
        seen_exp = set()
        for prob_cat in pass1:
            for h in prob_cat["catalogo"].get("holdings", []):
                exp = h.get("expediente", "")
                if h.get("score", 0) >= 0.80 and exp and exp not in seen_exp:
                    seen_exp.add(exp)
                    precedentes_utiles.append({
                        "expediente": exp,
                        "tribunal": h.get("tribunal", ""),
                        "tema": h.get("tema", ""),
                        "sentido": h.get("sentido", ""),
                        "score": h.get("score", 0),
                        "pdf_url": h.get("pdf_url", ""),
                    })
        # Sort by score descending
        precedentes_utiles.sort(key=lambda x: x["score"], reverse=True)
        
        yield RedactorEvent.complete(estudio_md, {
            "n_palabras": n_words,
            "n_chars": len(estudio_md),
            "n_problemas": n_problems,
            "total_elapsed_s": round(total_elapsed, 1),
            "precedentes_utiles": precedentes_utiles[:15],  # max 15
            "pass_times": {
                "pass0": round(elapsed0, 1),
                "pass1": round(elapsed1, 1),
                "pass2": round(elapsed2, 1),
                "pass3": round(elapsed3, 1),
            },
        })
    
    finally:
        if own_client:
            await http_client.aclose()
