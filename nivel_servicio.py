# -*- coding: utf-8 -*-
"""VELOCIDAD SIN TOCAR EL RAZONAMIENTO, Y AGUANTE CON MUCHOS A LA VEZ
(David, 3-oct-2026: «¿no es posible acelerar el proceso de entrega del
proyecto? sólo si es posible sin perder calidad ni causar conflicto si unos 20
secretarios usan el taller al mismo tiempo. Planeo en el futuro dejarlo correr
cada noche generando entre 100 y 200 proyectos en automático»).

LA REGLA DE SIEMPRE (taller-velocidad, 17-sep): la velocidad se busca en el
pipeline, NUNCA bajando el esfuerzo del modelo. Esto no lo baja: pide al
proveedor el MISMO modelo, con el MISMO razonamiento, en otro carril.

 1. EL CARRIL RÁPIDO DE OPENAI («priority», que OpenAI llama Fast mode desde el
    30-jul-2026). Medido el 3-oct-2026 con gpt-5.6-luna y razonamiento alto,
    mismo prompt, dos veces cada uno: 125-127 tok/s en el carril normal y
    182-187 tok/s en el rápido (+47%). Las llamadas largas del taller (análisis,
    propuesta, plan, estudio) son de escritura, así que tardan ~30% menos.
    Cuesta el doble por token (precio publicado para la familia 5.6). Se
    enciende con TALLER_NIVEL_SERVICIO=priority, sólo en peticiones del taller
    (las que pasan por `contexto_taller.poner`), nunca en el chat.
    OpenAI desaconseja el carril rápido para trabajos masivos en lote: para la
    corrida nocturna está el carril «flex» (más lento, la mitad de precio), que
    se pide por petición con `contexto_taller` (`nivel_servicio`).
    Gemini 3.8 Flash NO gana con su «priority» (184 → 190 tok/s, medido el
    mismo día): no se usa.

 2. LOS LÍMITES DEL PROVEEDOR. OpenAI da a la cuenta 10,000 peticiones y 10
    millones de tokens por minuto (cabeceras x-ratelimit del 3-oct): veinte
    secretarios a la vez usan ~0.5 M por minuto. Gemini es lo frágil: el mismo
    día, el carril «flex» de 3.8 Flash contestó 503 «high demand». Por eso las
    llamadas a Gemini del taller (examinador, supervisor) pasan por un
    SEMÁFORO por proceso y REINTENTAN 429/503 con espera creciente, dentro de
    su tope. Con eso, un pico de 20 asuntos a la vez espera turno en vez de
    caer al motor sin examen o al estudio sin supervisar.
"""
from __future__ import annotations

import asyncio
import os
import random
import re

_TIERS_OPENAI = {"priority", "fast", "flex", "default", "auto"}


def nivel_openai() -> str:
    """El carril para las llamadas de OpenAI del taller en ESTA petición: el que
    pida la petición (`contexto_taller`, p. ej. «flex» en la corrida nocturna)
    o el del entorno; «» = el normal. Fuera del taller, siempre «»."""
    try:
        import contexto_taller as _ct
        c = _ct._CTX.get()
    except Exception:
        return ""
    if c is None:
        return ""
    pedido = str((c or {}).get("nivel_servicio") or "").strip().lower()
    if not pedido:
        pedido = (os.getenv("TALLER_NIVEL_SERVICIO", "") or "").strip().lower()
    if pedido == "fast":
        pedido = "priority"
    return pedido if pedido in _TIERS_OPENAI and pedido not in ("default", "auto") else ""


def aplicar_openai(kw: dict) -> dict:
    """Pone `service_tier` en una llamada de chat del taller si corresponde y
    si quien llama no lo fijó. Devuelve el mismo diccionario."""
    if "service_tier" in kw:
        return kw
    t = nivel_openai()
    if t:
        kw["service_tier"] = t
    return kw


# ═══ GEMINI: TURNO Y REINTENTOS ══════════════════════════════════════════════

_SEM: asyncio.Semaphore | None = None
_SEM_LAZO = None


def semaforo_gemini() -> asyncio.Semaphore:
    """Cuántas llamadas a Gemini del taller corren a la vez en este proceso
    (GEMINI_TALLER_CONCURRENCIA, 8 por omisión). Uno por lazo de eventos."""
    global _SEM, _SEM_LAZO
    try:
        lazo = asyncio.get_running_loop()
    except RuntimeError:
        lazo = None
    if _SEM is None or _SEM_LAZO is not lazo:
        try:
            n = max(1, int(os.getenv("GEMINI_TALLER_CONCURRENCIA", "8") or 8))
        except ValueError:
            n = 8
        _SEM, _SEM_LAZO = asyncio.Semaphore(n), lazo
    return _SEM


_RX_PASAJERO = re.compile(r"\b(429|500|502|503|504)\b|RESOURCE_EXHAUSTED|UNAVAILABLE|high demand|overloaded"
                          r"|rate limit|deadline|temporar", re.I)


def es_pasajero(exc: BaseException) -> bool:
    """¿Es un fallo del proveedor que se cura esperando (cupo, saturación)?"""
    if isinstance(exc, (asyncio.TimeoutError, ConnectionError)):
        return True
    codigo = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    if codigo in (429, 500, 502, 503, 504):
        return True
    return bool(_RX_PASAJERO.search(f"{type(exc).__name__} {exc}"))


async def llamar_gemini(fabrica, tope_s: float, etiqueta: str = "gemini"):
    """Corre `await fabrica()` con turno (semáforo) y reintentos ante fallos
    pasajeros, sin pasar de `tope_s` en total (la espera del turno cuenta).
    Lanza asyncio.TimeoutError si se acaba el tiempo, o el error si no es
    pasajero. La espera crece: 2, 5, 10, 20 s, con algo de azar para que veinte
    asuntos no reintenten a la vez."""
    lazo = asyncio.get_running_loop()
    fin = lazo.time() + float(tope_s)
    esperas = (2.0, 5.0, 10.0, 20.0)
    intento = 0
    async with semaforo_gemini():
        while True:
            restante = fin - lazo.time()
            if restante <= 1:
                raise asyncio.TimeoutError(f"{etiqueta}: sin tiempo para llamar")
            try:
                return await asyncio.wait_for(fabrica(), timeout=restante)
            except asyncio.TimeoutError:
                raise
            except Exception as exc:
                if not es_pasajero(exc) or intento >= len(esperas):
                    raise
                pausa = esperas[intento] * (0.8 + 0.4 * random.random())
                intento += 1
                if fin - lazo.time() - pausa < 15:
                    raise
                print(f"   ⏳ {etiqueta}: el proveedor no atiende ({str(exc)[:90]}); "
                      f"reintento {intento} en {pausa:.0f} s")
                await asyncio.sleep(pausa)
