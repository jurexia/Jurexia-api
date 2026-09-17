"""LLAMAR AL MODELO SIN QUE UN PARÁMETRO TUMBE LA FASE ENTERA.

Los modelos de razonamiento no aceptan `temperature`. Cuando el taller cambió
de motor, las dos fases que la fijaban —la lectura del expediente y la
propuesta de sentido— empezaron a recibir:

    Unsupported value: 'temperature' does not support 0 with this model.
    Only the default (1) value is supported.

Y ahí se vio lo caro que sale capturar una excepción sin mirarla. En
`/taller/proponer` el error salía a la cara como un 500, que al menos se ve.
Pero en la fase 3 estaba dentro de un `try` que apunta el aviso y sigue:

    except Exception as e:
        f.avisos.append(f"No se pudieron derivar los problemas jurídicos: {e}")

De modo que el adelanto devolvía 200, el documento salía bien de forma, y los
PROBLEMAS JURÍDICOS iban vacíos. Sin ellos no hay consulta al acervo que
apuntar, ni propuesta que calificar, ni estudio que ordenar: el corazón del
taller llevaba semanas apagado y el único síntoma era un aviso entre catorce.

LA REGLA: se quita EL PARÁMETRO que el servidor rechaza, no la llamada. Es la
diferencia entre perder el determinismo —que es un lujo— y perder la fase, que
es el producto.
"""

from __future__ import annotations

import re

# «Unsupported value: 'temperature' does not support 0», «Unsupported
# parameter: 'seed'», «'top_p' is not supported with this model».
_RX_PARAM = re.compile(
    r"[Uu]nsupported (?:value|parameter)[^']*'([a-z_]+)'|"
    r"'([a-z_]+)'\s+is not supported", re.I)

# Sólo se quitan los de MUESTREO. Si el servidor rechaza `messages` o `model`,
# eso no es un lujo prescindible: es la llamada, y tiene que reventar.
_PRESCINDIBLES = {"temperature", "seed", "top_p", "frequency_penalty",
                  "presence_penalty", "logprobs"}


def parametro_rechazado(exc: Exception) -> str:
    m = _RX_PARAM.search(str(exc) or "")
    if not m:
        return ""
    p = (m.group(1) or m.group(2) or "").strip().lower()
    return p if p in _PRESCINDIBLES else ""


# ═══════════════════════════════════════════════════════════════════════════
# LA COLA DE LATENCIA: UNA PETICIÓN DE RESPALDO, NO UN CORTE
# ═══════════════════════════════════════════════════════════════════════════
# Medido el 17-sep-2026 en la revisión fiscal 2/2026: la propuesta del motor,
# que tarda unos 150 s, tardó 525. No era el código: era el proveedor, y como el
# cliente espera hasta 600 s, el secretario también esperaba.
#
# Cortar la llamada lenta sería peor —si iba progresando, se tira el trabajo y
# se empieza de cero—. Lo que se hace es lo estándar para la cola de latencia:
# si una llamada pasa de lo NORMAL para su tamaño, se lanza OTRA IDÉNTICA en
# paralelo y se toma la primera que termine; la otra se cancela. Mismo modelo,
# misma instrucción, mismos parámetros: la calidad no cambia en nada. Sólo
# cuesta tokens en el caso lento, que es justo donde vale la pena.
#
# LO NORMAL depende de cuánto se le pide escribir: 120 s más un segundo por
# cada 50 tokens de tope, entre 180 y 300. Configurable sin desplegar.
import asyncio as _asyncio
import os as _os

RESPALDO_MIN = float(_os.getenv("MODELO_RESPALDO_MIN_S", "180"))
RESPALDO_MAX = float(_os.getenv("MODELO_RESPALDO_MAX_S", "300"))
RESPALDO_ACTIVO = _os.getenv("MODELO_RESPALDO", "1") != "0"


def espera_normal(kw: dict) -> float:
    tope = kw.get("max_completion_tokens") or kw.get("max_tokens") or 4000
    try:
        tope = float(tope)
    except (TypeError, ValueError):
        tope = 4000.0
    return max(RESPALDO_MIN, min(RESPALDO_MAX, 120.0 + tope / 50.0))


async def crear(cliente, **kw):
    """`chat.completions.create`, con respaldo si el proveedor se atasca."""
    if not RESPALDO_ACTIVO or kw.get("stream"):
        return await _crear_una(cliente, **kw)
    limite = espera_normal(kw)
    primera = _asyncio.ensure_future(_crear_una(cliente, **dict(kw)))
    hechas, _ = await _asyncio.wait({primera}, timeout=limite)
    if hechas:
        return primera.result()
    print(f"   ⏳ el modelo lleva {limite:.0f} s sin contestar (lo normal para "
          f"este tamaño): se lanza una petición de respaldo idéntica")
    respaldo = _asyncio.ensure_future(_crear_una(cliente, **dict(kw)))
    pendientes = {primera, respaldo}
    ultimo_error = None
    while pendientes:
        hechas, pendientes = await _asyncio.wait(
            pendientes, return_when=_asyncio.FIRST_COMPLETED)
        for t in hechas:
            if t.exception() is None:
                for otra in pendientes:
                    otra.cancel()
                print(f"   ⏳ contestó {'la de respaldo' if t is respaldo else 'la original'}")
                return t.result()
            ultimo_error = t.exception()
    raise ultimo_error


def _anotar_uso(kw: dict, r, segundos: float) -> None:
    """Una línea por llamada: cuánto tardó y en qué se fue.

    POR QUÉ. La propuesta de la revisión fiscal 2/2026 tardó 542 s en
    producción y 206 s en local, con el mismo material, el mismo modelo, el
    mismo esfuerzo y LA MISMA clave. Sin saber cuántos tokens razonó cada una no
    hay manera de distinguir «producción piensa más» de «producción tarda más
    en lo mismo». Sólo números: ni prompt ni respuesta llegan al registro.
    """
    try:
        u = getattr(r, "usage", None)
        det = getattr(u, "completion_tokens_details", None)
        sal = int(getattr(u, "completion_tokens", 0) or 0)
        raz = int(getattr(det, "reasoning_tokens", 0) or 0)
        print(f"   ⏱️ modelo {kw.get('model')} · {segundos:.1f} s · entrada "
              f"{getattr(u, 'prompt_tokens', '?')} · razonamiento {raz} · visible "
              f"{sal - raz} · esfuerzo {kw.get('reasoning_effort') or '—'} · "
              f"{(sal / segundos) if segundos else 0:.0f} tok/s")
    except Exception:
        pass


async def _crear_una(cliente, **kw):
    """`chat.completions.create`, quitando lo que el modelo no admita."""
    import time as _t_uso
    _t0_uso = _t_uso.perf_counter()
    r = await _crear_una_sin_anotar(cliente, **kw)
    _anotar_uso(kw, r, _t_uso.perf_counter() - _t0_uso)
    return r


async def _crear_una_sin_anotar(cliente, **kw):
    quitados = []
    for _ in range(4):
        try:
            r = await cliente.chat.completions.create(**kw)
            if quitados:
                print(f"   ℹ️ el modelo no admite {', '.join(quitados)}: se "
                      f"llamó sin ellos (se pierde determinismo, no la fase)")
            return r
        except Exception as exc:
            p = parametro_rechazado(exc)
            if not p or p not in kw:
                raise
            kw.pop(p, None)
            quitados.append(p)
    raise RuntimeError("El modelo rechazó demasiados parámetros seguidos.")
