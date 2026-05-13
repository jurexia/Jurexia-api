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

# Límites de catálogo para Pass 2 (evita saturar contexto)
MAX_TESIS_PER_PROBLEM = 12
MAX_HOLDINGS_PER_PROBLEM = 10
MAX_NORMAS_PER_PROBLEM = 12


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

NIVEL ESPERADO: la calidad de filtrado debe ser equivalente a la de un magistrado revisando un proyecto de su secretario. NO incluyas fuentes "por si acaso"."""


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
ANTI-INVENCIÓN (regla inviolable)
═══════════════════════════════════════════════════════════════════════
SOLO puedes citar tesis cuyo `registro` o clave aparezca en el plan dentro de
`tesis_clave_a_citar`. NUNCA inventes registros como "1a./J. XXX/AAAA",
expedientes, rubros, ni atribuyas criterios a la SCJN sin sustento del plan.
Si una tesis del plan está marcada `verificable: false`, refiérete a ella por
su contenido sin citarla con rubro entrecomillado ni con número de tesis. En
duda, omite la cita y argumenta con el marco normativo y el bloque
constitucional que sí estén en el plan.

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

### Resumen del acto reclamado
[Reconstrucción objetiva, 200-400 palabras]

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


async def _run_pass0(
    http_client: httpx.AsyncClient,
    api_key: str,
    caso_input: dict,
) -> dict:
    """Ejecuta Pass 0 y devuelve la estructura cognitiva."""
    user_prompt = _build_pass0_prompt(caso_input)
    n_problems = len(caso_input["inputs"].get("disidencias", []))
    max_tokens = 16000 if n_problems <= 3 else 24000
    
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
    """Ejecuta Pass 2 y devuelve el plan de redacción."""
    user_prompt = _build_pass2_prompt(pass0, pass1, caso_meta)
    n_problems = len(pass0.get("problemas_juridicos", []))
    max_tokens = 16000 if n_problems <= 2 else 24000
    
    content, usage, finish_reason = await _call_with_retry(
        http_client, api_key, PASS_2_SYSTEM, user_prompt, max_tokens
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
                    parts.append(f"\n  • {t.get('registro','?')} · {t.get('instancia','')}")
                    parts.append(f"    RUBRO: {t.get('rubro_corto','')}")
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

async def _validate_tesis_in_plan(
    plan: dict,
    tesis_validator_fn: Optional[Callable[[list[str]], Awaitable[set[str]]]] = None,
) -> dict:
    """
    Valida que cada tesis citada en el plan exista en Qdrant.
    Marca cada entrada con `verificable: True/False` y agrega `tesis_invalidadas`
    a cada problema. NO elimina entradas — Pass 3 decide cómo tratarlas.

    `tesis_validator_fn` recibe una lista de registros y devuelve el set de
    los que SÍ existen. Si es None, todas se marcan verificable=True (no-op).
    """
    if tesis_validator_fn is None:
        return plan

    # Recolectar todos los registros mencionados en el plan
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
        registros_validos = await tesis_validator_fn(list(todos_registros))
    except Exception as e:
        print(f"   ⚠️ Validador de tesis falló: {e} — marcando todas como verificables (fallback seguro)")
        registros_validos = todos_registros

    # Anotar el plan
    for prob in plan.get("plan_por_problema", []):
        invalidadas = []
        for t in prob.get("tesis_clave_a_citar", []) or []:
            if isinstance(t, dict):
                reg = str(t.get("registro", "")).strip()
                if reg and reg in registros_validos:
                    t["verificable"] = True
                else:
                    t["verificable"] = False
                    if reg:
                        invalidadas.append(reg)
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
        # 1) Sustituir placeholders [NUMERAL] por "QUINTO" default.
        # 2) Validar que ninguna tesis citada en el markdown esté fuera del plan.
        estudio_md = _normalizar_numerales(estudio_md)

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
