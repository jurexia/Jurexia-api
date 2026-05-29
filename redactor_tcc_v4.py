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
    _run_pass_minus1,
    _run_pass_minus1_regenerate,
    _validate_tesis_in_plan,
    _validar_estudio_post_pass3,
    _normalizar_numerales,
    _limpiar_ids_internos,
)

# ════════════════════════════════════════════════════════════════════════
# JOB STORE — Supabase persistente + fallback in-memory
# ════════════════════════════════════════════════════════════════════════
#
# Historia: hasta 2026-05-28 el job store era un dict in-memory. Cada vez
# que Render reiniciaba el proceso (deploy nuevo, restart, sleep) entre
# /analyze y /finalize, el secretario veía "Sesión expirada" y tenía que
# repetir todo. Pasó en producción con el deploy del Stage -1.
#
# Ahora: persistencia en Supabase tabla `redactor_tcc_jobs`. Si Supabase no
# está configurado o falla, cae a in-memory (no rompe dev/test local).
#
# Tabla creada por: _tools/migration_redactor_tcc_jobs.sql

import asyncio
import datetime as _dt

_JOB_TTL_SECONDS = 60 * 60

# Fallback in-memory (se usa si Supabase no responde o no está configurado).
_jobs_fallback: dict[str, dict] = {}


def _get_supabase_admin():
    """Lazy import — evita ciclo con main.py."""
    try:
        from main import supabase_admin  # type: ignore
        return supabase_admin
    except Exception:
        return None


def _gc_expired_fallback() -> None:
    now = time.time()
    expired = [jid for jid, st in _jobs_fallback.items() if st.get("expires_at_ts", 0) < now]
    for jid in expired:
        _jobs_fallback.pop(jid, None)


async def store_job(job_id: str, payload: dict) -> None:
    """Persiste el job en Supabase con TTL 1h. Si falla, usa fallback in-memory.

    payload esperado tras run_analyze_phase:
      { "pass0": dict, "pass2": dict, "caso_meta": dict, "created_at": float }
    """
    expires_at_ts = time.time() + _JOB_TTL_SECONDS
    expires_at_iso = _dt.datetime.utcfromtimestamp(expires_at_ts).isoformat() + "Z"

    sb = _get_supabase_admin()
    if sb is not None:
        try:
            row = {
                "job_id": job_id,
                "pass0": payload.get("pass0", {}),
                "pass2": payload.get("pass2", {}),
                "caso_meta": payload.get("caso_meta", {}),
                "expires_at": expires_at_iso,
            }
            # upsert para idempotencia (si el secretario reintenta)
            await asyncio.to_thread(
                lambda: sb.table("redactor_tcc_jobs").upsert(row).execute()
            )
            print(f"   💾 [job store] {job_id[:8]}... persistido en Supabase")
            return
        except Exception as e:
            print(f"   ⚠️ [job store] Supabase upsert falló ({e}); usando fallback in-memory")

    # Fallback: in-memory
    _gc_expired_fallback()
    payload_with_ttl = dict(payload)
    payload_with_ttl["expires_at_ts"] = expires_at_ts
    _jobs_fallback[job_id] = payload_with_ttl
    print(f"   💾 [job store] {job_id[:8]}... persistido en in-memory (fallback)")


async def get_job(job_id: str) -> Optional[dict]:
    """Lee el job. Prueba Supabase primero, después fallback. Devuelve None si expiró."""
    sb = _get_supabase_admin()
    if sb is not None:
        try:
            resp = await asyncio.to_thread(
                lambda: sb.table("redactor_tcc_jobs")
                .select("pass0, pass2, caso_meta, expires_at")
                .eq("job_id", job_id)
                .limit(1)
                .execute()
            )
            rows = resp.data or []
            if rows:
                row = rows[0]
                # Verificar expiración (defensivo; en teoría el GC ya lo borró)
                exp_str = row.get("expires_at", "")
                try:
                    exp_dt = _dt.datetime.fromisoformat(exp_str.replace("Z", "+00:00"))
                    if exp_dt.timestamp() < time.time():
                        # Expirado — borra y devuelve None
                        await asyncio.to_thread(
                            lambda: sb.table("redactor_tcc_jobs")
                            .delete().eq("job_id", job_id).execute()
                        )
                        return None
                except Exception:
                    pass
                return {
                    "pass0": row.get("pass0") or {},
                    "pass2": row.get("pass2") or {},
                    "caso_meta": row.get("caso_meta") or {},
                }
            # No está en Supabase — chequear fallback (por si el analyze se hizo offline)
        except Exception as e:
            print(f"   ⚠️ [job store] Supabase select falló ({e}); revisando fallback")

    # Fallback in-memory
    _gc_expired_fallback()
    return _jobs_fallback.get(job_id)


async def drop_job(job_id: str) -> None:
    """Borra el job de Supabase y del fallback (idempotente)."""
    sb = _get_supabase_admin()
    if sb is not None:
        try:
            await asyncio.to_thread(
                lambda: sb.table("redactor_tcc_jobs")
                .delete().eq("job_id", job_id).execute()
            )
        except Exception as e:
            print(f"   ⚠️ [job store] Supabase delete falló ({e})")
    _jobs_fallback.pop(job_id, None)


# ════════════════════════════════════════════════════════════════════════
# SUMMARIZE — Stage -1 (resúmenes jurídicos editables, método manual)
# ════════════════════════════════════════════════════════════════════════
#
# Replica el flujo manual del secretario: primero generar resúmenes con
# Gemini Pro (acto reclamado + conceptos/agravios en paralelo), permitir
# edición humana en la UI, y solo entonces alimentar Pass 0+1+2+3 con esos
# resúmenes pulidos en vez de con el texto OCR crudo.
#
# Este stage NO usa el job store: los resúmenes editados se envían como
# `texto_acto_reclamado` y `texto_conceptos_agravios` al endpoint /analyze
# existente, sin requerir state server-side.

async def run_summarize_phase(
    texto_acto: str,
    texto_cv: str,
    http_client: httpx.AsyncClient,
) -> AsyncGenerator[dict, None]:
    """
    Stage -1: corre Pass -1 y emite eventos SSE.

    Termina con un evento `summaries_ready` con los 2 resúmenes en prosa
    para que el secretario los edite en la UI antes de /analyze.
    """
    yield RedactorEvent.phase(
        -1, 10,
        "Generando resumen jurídico del acto reclamado y de los conceptos/agravios (paralelo)...",
    )
    t0 = time.time()
    try:
        result = await _run_pass_minus1(http_client, texto_acto, texto_cv)
    except Exception as e:
        yield RedactorEvent.error(f"Error generando resúmenes: {e}", -1)
        return

    n_acto = len(result["resumen_acto"].split())
    n_cv = len(result["resumen_cv"].split())
    yield RedactorEvent.pass_complete(-1, time.time() - t0, {
        "palabras_acto": n_acto,
        "palabras_cv": n_cv,
    })

    yield {
        "type": "summaries_ready",
        "data": {
            "resumen_acto": result["resumen_acto"],
            "resumen_cv": result["resumen_cv"],
            "total_elapsed_s": round(time.time() - t0, 1),
        },
    }


async def run_regenerate_summary_phase(
    kind: str,
    resumen_actual: str,
    instruccion: str,
    texto_original: str,
    http_client: httpx.AsyncClient,
) -> AsyncGenerator[dict, None]:
    """Stage -1bis: regenera UN resumen con la instrucción libre del secretario."""
    if kind not in ("acto", "cv"):
        yield RedactorEvent.error(f"kind inválido: {kind!r} (esperado 'acto' o 'cv')", -1)
        return

    label = "del acto reclamado" if kind == "acto" else "de los conceptos/agravios"
    yield RedactorEvent.phase(
        -1, 30, f"Regenerando resumen {label} con la instrucción del secretario...",
    )
    t0 = time.time()
    try:
        nuevo = await _run_pass_minus1_regenerate(
            http_client, kind, resumen_actual, instruccion, texto_original,
        )
    except Exception as e:
        yield RedactorEvent.error(f"Error regenerando resumen: {e}", -1)
        return

    yield {
        "type": "summary_regenerated",
        "data": {
            "kind": kind,
            "resumen": nuevo,
            "elapsed_s": round(time.time() - t0, 1),
        },
    }


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
    await store_job(job_id, {
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
    state = await get_job(job_id)
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

    await drop_job(job_id)
