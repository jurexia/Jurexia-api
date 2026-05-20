"""
redactor_tcc_v4.py — Redactor TCC con humano en el loop.

Diferencia clave vs v3:
  v3: full-auto Pass 0 → Pass 1 → Pass 2 → Pass 3 (sin pausa).
  v4: ANALYZE = Pass 0+1+2. Pausa. El secretario revisa el plan en la UI,
      filtra tesis, ajusta calificación, agrega instrucción. Luego FINALIZE
      ejecuta Pass 3 sobre el plan ya editado.

Endpoints expuestos en main.py:
  POST /redactor/tcc-v4/analyze    (acepta texto, devuelve plan + job_id)
  POST /redactor/tcc-v4/finalize   (acepta job_id + edits, streamea Pass 3)

Estado entre fases: in-memory dict con TTL de 1 hora. Iteración 2 → Supabase.

VERSION: 2026-05-19-v1
"""

from __future__ import annotations
import time
import uuid
from typing import Any, AsyncGenerator, Awaitable, Callable, Optional

import httpx

from redactor_tcc_v3 import (
    RedactorEvent,
    _run_pass0,
    _run_pass1,
    _run_pass2,
    _run_pass3_stream,
    _validate_tesis_in_plan,
    _validar_estudio_post_pass3,
    _normalizar_numerales,
    _limpiar_ids_internos,
)

# ════════════════════════════════════════════════════════════════════════
# IN-MEMORY JOB STORE (TTL 1 h)
# ════════════════════════════════════════════════════════════════════════

_JOB_TTL_SECONDS = 60 * 60
_jobs: dict[str, dict] = {}


def _gc_expired_jobs() -> None:
    now = time.time()
    expired = [jid for jid, st in _jobs.items() if st.get("expires_at", 0) < now]
    for jid in expired:
        _jobs.pop(jid, None)


def store_job(job_id: str, payload: dict) -> None:
    _gc_expired_jobs()
    payload["expires_at"] = time.time() + _JOB_TTL_SECONDS
    _jobs[job_id] = payload


def get_job(job_id: str) -> Optional[dict]:
    _gc_expired_jobs()
    return _jobs.get(job_id)


def drop_job(job_id: str) -> None:
    _jobs.pop(job_id, None)


# ════════════════════════════════════════════════════════════════════════
# ANALYZE — Pass 0 + 1 + 2 (sin Pass 3)
# ════════════════════════════════════════════════════════════════════════

async def run_analyze_phase(
    caso_input: dict,
    qdrant_search_fn: Callable[..., Awaitable[dict]],
    deepseek_api_key: str,
    http_client: httpx.AsyncClient,
    tesis_validator_fn: Optional[Callable[[list[str]], Awaitable[dict]]] = None,
) -> AsyncGenerator[dict, None]:
    """
    Corre Pass 0+1+2 y emite eventos SSE.
    Termina con un evento `plan_ready` que contiene job_id + plan editable.
    """
    caso_meta = caso_input["meta"]
    total_start = time.time()

    # ─── PASS 0 ──────────────────────────────────────────────
    yield RedactorEvent.phase(0, 5, "Analizando estructura del caso...")
    t0 = time.time()
    try:
        pass0 = await _run_pass0(http_client, deepseek_api_key, caso_input)
    except Exception as e:
        yield RedactorEvent.error(f"Error en análisis cognitivo: {e}", 0)
        return
    n_problems = len(pass0.get("problemas_juridicos", []))
    n_dis = len(pass0.get("disidencias_estructuradas", []))

    # ─── FAIL-FAST ───────────────────────────────────────────
    # Si Pass 0 no detectó problemas jurídicos, NO seguimos con Pass 1/2
    # (se quedan razonando vacío 20+ min). Mejor cortar y avisar.
    if n_problems == 0:
        yield RedactorEvent.error(
            "El análisis no detectó problemas jurídicos en los documentos. "
            "Probablemente el texto del acto reclamado o de los conceptos/agravios "
            "no es claro o quedó corrupto. Revísalos y vuelve a intentar.",
            0,
        )
        return

    yield RedactorEvent.pass_complete(0, time.time() - t0, {
        "n_problemas": n_problems,
        "n_disidencias": n_dis,
        "complejidad": pass0.get("complejidad_caso", "?"),
    })

    # ─── PASS 1 (RAG) ────────────────────────────────────────
    yield RedactorEvent.phase(1, 25, f"{n_problems} problemas identificados — recuperando jurisprudencia...")
    t1 = time.time()
    try:
        pass1 = await _run_pass1(pass0, caso_meta, qdrant_search_fn)
    except Exception as e:
        yield RedactorEvent.error(f"Error en búsqueda RAG: {e}", 1)
        return
    total_fuentes = sum(
        len(p["catalogo"].get("tesis", [])) +
        len(p["catalogo"].get("normas", [])) +
        len(p["catalogo"].get("holdings", []))
        for p in pass1
    )
    yield RedactorEvent.pass_complete(1, time.time() - t1, {"n_fuentes_total": total_fuentes})

    # ─── PASS 2 (plan + validador) ───────────────────────────
    # Si Pass 1 no recuperó nada Y Pass 0 sí detectó problemas, algo está mal
    # en Qdrant o en las queries. Avisamos pero seguimos (el modelo armará un
    # plan con marco normativo solamente; el secretario podrá completar a mano).
    if total_fuentes == 0:
        print(f"   ⚠️ Pass 1 recuperó 0 fuentes — Pass 2 trabajará sin catálogo")
        yield RedactorEvent.phase(2, 55, "Sin fuentes recuperadas — construyendo plan a partir de la estructura cognitiva...")
    else:
        yield RedactorEvent.phase(2, 55, f"{total_fuentes} fuentes recuperadas — construyendo plan...")

    t2 = time.time()
    try:
        pass2 = await _run_pass2(http_client, deepseek_api_key, pass0, pass1, caso_meta)
    except Exception as e:
        yield RedactorEvent.error(f"Error en plan de redacción: {e}", 2)
        return

    try:
        pass2 = await _validate_tesis_in_plan(pass2, tesis_validator_fn)
    except Exception as e:
        print(f"   ⚠️ Validación de tesis falló: {e}")

    n_tesis = sum(len(p.get("tesis_clave_a_citar", [])) for p in pass2.get("plan_por_problema", []))
    n_inv = sum(len(p.get("tesis_invalidadas", []) or []) for p in pass2.get("plan_por_problema", []))
    yield RedactorEvent.pass_complete(2, time.time() - t2, {
        "n_tesis_seleccionadas": n_tesis,
        "n_tesis_invalidas": n_inv,
    })

    # ─── PRECEDENTES ÚTILES (snapshot para la UI del secretario) ─
    precedentes: list[dict] = []
    seen_exp: set[str] = set()
    for prob_cat in pass1:
        for h in prob_cat["catalogo"].get("holdings", []):
            exp = h.get("expediente", "")
            if h.get("score", 0) >= 0.70 and exp and exp not in seen_exp:
                seen_exp.add(exp)
                precedentes.append({
                    "expediente": exp,
                    "tribunal": h.get("tribunal", ""),
                    "tema": h.get("tema", ""),
                    "sentido": h.get("sentido", ""),
                    "score": h.get("score", 0),
                    "pdf_url": h.get("pdf_url", ""),
                })
    precedentes.sort(key=lambda x: x["score"], reverse=True)

    # ─── CREATE JOB ─────────────────────────────────────────
    job_id = uuid.uuid4().hex
    store_job(job_id, {
        "pass0": pass0,
        "pass2": pass2,
        "caso_meta": caso_meta,
        "created_at": time.time(),
    })

    yield {
        "type": "plan_ready",
        "data": {
            "job_id": job_id,
            "plan": pass2,
            "pass0": pass0,
            "precedentes_utiles": precedentes[:30],
            "total_elapsed_s": round(time.time() - total_start, 1),
        },
    }


# ════════════════════════════════════════════════════════════════════════
# APPLY SECRETARY EDITS — validador determinista (filtro de tesis)
# ════════════════════════════════════════════════════════════════════════

def apply_secretary_edits(plan: dict, edits: Optional[dict]) -> dict:
    """
    Aplica las ediciones del secretario al plan.

    edits = {
      "problemas": [
        {
          "problema_id": "P1",
          "calificacion_override": "fundado|infundado|inoperante|...",   # opcional
          "accion_redaccion_override": "abordar_completo|...",            # opcional
          "tesis_aprobadas": ["registro1","registro2"],                   # solo estas sobreviven
          "tesis_manuales": [{"registro":"...", "rubro_corto":"...", ...}],
          "instruccion_secretario": "texto libre del secretario"
        }
      ]
    }

    Reglas duras (validador determinista):
      • Si tesis_aprobadas se provee, las tesis NO incluidas se descartan.
      • Las tesis_manuales se agregan con verificable=True (responsabilidad del secretario).
      • La instrucción del secretario se inyecta en conclusion_razonada para que
        Pass 3 la consuma sin cambiar el prompt builder.
    """
    if not edits or not edits.get("problemas"):
        return plan

    edits_by_id = {e["problema_id"]: e for e in edits["problemas"] if e.get("problema_id")}

    for prob in plan.get("plan_por_problema", []):
        pid = prob.get("problema_id")
        e = edits_by_id.get(pid)
        if not e:
            continue

        if e.get("calificacion_override"):
            prob["calificacion_propuesta_disidencia"] = e["calificacion_override"]
            prob["_secretary_override_calificacion"] = True

        if e.get("accion_redaccion_override"):
            prob["accion_redaccion"] = e["accion_redaccion_override"]
            prob["_secretary_override_accion"] = True

        # FILTRO DETERMINISTA DE TESIS
        if e.get("tesis_aprobadas") is not None:
            aprobadas = {str(r).strip() for r in e["tesis_aprobadas"]}
            original = prob.get("tesis_clave_a_citar", []) or []
            kept = []
            for t in original:
                if isinstance(t, dict):
                    reg = str(t.get("registro", "")).strip()
                    if reg in aprobadas:
                        t["_secretary_approved"] = True
                        kept.append(t)
            for mt in e.get("tesis_manuales", []) or []:
                if isinstance(mt, dict):
                    mt["_secretary_manual"] = True
                    mt["verificable"] = True
                    kept.append(mt)
            prob["tesis_clave_a_citar"] = kept

        instr = (e.get("instruccion_secretario") or "").strip()
        if instr:
            prob["instruccion_secretario"] = instr

    return plan


# ════════════════════════════════════════════════════════════════════════
# FINALIZE — corre Pass 3 streaming sobre el plan editado
# ════════════════════════════════════════════════════════════════════════

async def run_finalize_phase(
    job_id: str,
    edits: Optional[dict],
    deepseek_api_key: str,
    http_client: httpx.AsyncClient,
) -> AsyncGenerator[dict, None]:
    """Yields eventos SSE de Pass 3 streaming + evento complete final."""
    state = get_job(job_id)
    if not state:
        yield RedactorEvent.error(
            f"Sesión expirada (job_id {job_id[:8]}...). Reinicia el análisis.",
            3,
        )
        return

    pass0 = state["pass0"]
    pass2 = state["pass2"]
    caso_meta = state["caso_meta"]

    # Aplicar edits del secretario
    pass2 = apply_secretary_edits(pass2, edits)

    # Inyectar instruccion_secretario al final de conclusion_razonada
    # para que el prompt builder de Pass 3 la consuma sin cambios.
    for prob in pass2.get("plan_por_problema", []):
        instr = prob.get("instruccion_secretario", "").strip()
        if instr:
            existing = prob.get("conclusion_razonada", "")
            prob["conclusion_razonada"] = (
                f"{existing}\n\n"
                "[INSTRUCCIÓN EXPRESA DEL SECRETARIO PROYECTISTA — RESPETAR LITERALMENTE]:\n"
                f"{instr}"
            ).strip()

    n_problems = len(pass2.get("plan_por_problema", []))
    yield RedactorEvent.phase(3, 60, f"Redactando estudio de fondo ({n_problems} problemas)...")

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
        yield RedactorEvent.error(f"Error en redacción: {e}", 3)
        return

    # Post-procesamiento (mismo de v3)
    estudio_md = _normalizar_numerales(estudio_md)
    estudio_md = _limpiar_ids_internos(estudio_md)

    # Validador determinista de citas: solo se aceptan registros aprobados por el secretario
    registros_validos: set[str] = set()
    for prob in pass2.get("plan_por_problema", []):
        for t in prob.get("tesis_clave_a_citar", []) or []:
            if isinstance(t, dict) and t.get("verificable", True):
                reg = str(t.get("registro", "")).strip().upper().replace(" ", "")
                if reg:
                    registros_validos.add(reg)
    validation = _validar_estudio_post_pass3(estudio_md, registros_validos)
    if validation["n_citas_no_validas"] > 0:
        disclaimer = (
            "\n\n---\n\n"
            "> **Aviso de verificación**: las siguientes citas detectadas en el "
            "borrador no figuran en el plan aprobado por el secretario: "
            + ", ".join(validation["citas_no_validas"][:10])
            + ". Confirma manualmente antes de incorporarlas."
        )
        estudio_md += disclaimer

    n_words = len(estudio_md.split())
    yield RedactorEvent.pass_complete(3, time.time() - t3, {"n_palabras": n_words})

    yield RedactorEvent.complete(estudio_md, {
        "n_palabras": n_words,
        "n_chars": len(estudio_md),
        "n_problemas": n_problems,
        "total_elapsed_s": round(time.time() - t3, 1),
        "truncated": truncated,
    })

    drop_job(job_id)
