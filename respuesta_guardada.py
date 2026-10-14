"""LA RESPUESTA QUE EL SERVIDOR YA ESCRIBIÓ NO SE PIERDE.   (14-oct-2026)

Hasta hoy el historial del chat sólo lo guardaba el navegador, y sólo AL
TERMINAR el stream. Roberto (13/14-oct) perdió tres respuestas que el servidor
terminó sin error —23:07, 23:14 y 05:05 UTC—: su navegador leyó cada flujo
hasta el final (pidió el contador de consultas en el mismo segundo en que
Render cerró la respuesta) y luego no guardó nada. Las conversaciones quedaron
vacías y la siguiente consulta las borró por vacías.

Aquí la API guarda su propia copia, ligada al usuario DEL TOKEN y a la
conversación que el navegador dice, en `respuestas_chat`:

  · al empezar, una fila «en_curso» con la pregunta;
  · al terminar, la respuesta tal como la guardaría el navegador
    (`texto_para_historial`), «terminada»; si el navegador cortó a medias,
    «interrumpida» con lo que hubiera.

QUIÉN LA PASA A `messages`, SIN DUPLICAR. La fila se RECLAMA: el navegador que
guarda pone `guardada_at` con un UPDATE condicionado a que siga en null (sólo
uno gana), y sólo quien gana inserta los mensajes. La pestaña que terminó bien
reclama justo antes de guardar; la recuperación (al abrir el chat, o al volver
a la pestaña) reclama lo terminado que nadie reclamó. La fila nace al EMPEZAR
para que el reclamo del navegador —que llega después del último byte— siempre
la encuentre.

Sólo se activa si el front manda `peticion_id`: la app móvil, los flujos que
guardan en otra parte (los alegatos de carpeta van a `expediente_documentos`)
y los bundles viejos siguen exactamente como antes.

Nada de esto puede tumbar una consulta: cada escritura es en segundo plano y
sus fallos sólo se anotan («💾 RESPALDO»).
"""
from __future__ import annotations

import asyncio
import json
import re
import uuid
from datetime import datetime, timezone
from typing import AsyncIterator, Optional

from fastapi import APIRouter, Request

router = APIRouter()

_sb = None

# Lo que se lanza en segundo plano se guarda aquí: una tarea sin referencia
# puede recogerla el recolector de basura antes de terminar.
_tareas: set = set()

TOPE_PREGUNTA = 200_000
TOPE_RESPUESTA = 600_000


def configurar(supabase_admin) -> None:
    """main.py presta su cliente de servicio (uno por hilo)."""
    global _sb
    _sb = supabase_admin


def _uuid_o_none(v: Optional[str]) -> Optional[str]:
    if not v:
        return None
    try:
        return str(uuid.UUID(str(v)))
    except (ValueError, AttributeError, TypeError):
        return None


# ── El mismo texto que guardaría el navegador ─────────────────────────────
#
# Espejo de `useChat.sendMessage` (jurexia-frontend-git, src/hooks/useChat.ts):
# fuera los marcadores de control del stream, el razonamiento en vivo
# reescrito como bloque <!--THINKING_START-->…<!--THINKING_END-->, y fuera el
# mapa previo de fuentes (`FUENTES_PREVIAS`), que en la base sólo pesaría.
# `CITATION_META` y `PRECEDENTES_META` se QUEDAN: sin ellos, al reabrir la
# conversación las citas se quedan sin panel.

_CONTROL = re.compile(
    r"<!--PING-->"
    r"|<!--SOURCES:\d+-->"
    r"|<!--PASO:[^>]*-->"
    r"|<!--CACHE:ACTIVE-->"
    r"|<!--MODE:(?:PLATINUM|PRO|PROFESIONAL)-->"
    r"|<!--REGISTROS_FUERA:[^>]*-->"
    r"|<!--RETRY:\d+:\d+(?::\w+)?-->"
)
_FUENTES_PREVIAS = re.compile(r"\n*<!-- FUENTES_PREVIAS:[\s\S]*?-->\n*")
_ABRE = "<!--thinking-->"
_CIERRA = "<!--/thinking-->"
_SIN_RESPUESTA = ("\n\n*El modelo agotó los tokens durante el razonamiento. "
                  "El análisis completo se encuentra en la pestaña \"Ver razonamiento jurídico\" arriba. "
                  "Puedes enviar un mensaje de seguimiento para obtener la respuesta final.*")


def _separar_razonamiento(t: str) -> tuple[str, str]:
    """(razonamiento, respuesta) con las reglas de `ThinkingParser`: antes del
    primer <!--thinking--> todo es respuesta; entre ése y <!--/thinking--> todo
    es razonamiento (los <!--thinking--> intermedios se descartan); después,
    respuesta otra vez."""
    razon, resp = [], []
    i, en_razon = 0, False
    while i < len(t):
        if not en_razon:
            j = t.find(_ABRE, i)
            if j == -1:
                resp.append(t[i:])
                break
            resp.append(t[i:j])
            i, en_razon = j + len(_ABRE), True
            continue
        fin = t.find(_CIERRA, i)
        sig = t.find(_ABRE, i)
        if fin != -1 and (sig == -1 or fin <= sig):
            razon.append(t[i:fin])
            i, en_razon = fin + len(_CIERRA), False
            continue
        if sig != -1:
            razon.append(t[i:sig])
            i = sig + len(_ABRE)
            continue
        razon.append(t[i:])
        break
    return "".join(razon), "".join(resp)


def texto_para_historial(crudo: str) -> str:
    t = _CONTROL.sub("", crudo or "")
    razon, resp = _separar_razonamiento(t)
    texto = (f"<!--THINKING_START-->{razon}<!--THINKING_END-->" if razon else "") + resp
    if razon and not resp.strip():
        texto += _SIN_RESPUESTA
    if "<!-- FUENTES_PREVIAS:" in texto:
        texto = _FUENTES_PREVIAS.sub("\n\n", texto).strip()
    return texto


# ── Escrituras ─────────────────────────────────────────────────────────────

async def _usuario_del_token(authorization: Optional[str]) -> Optional[str]:
    """El dueño sale del token, nunca del cuerpo: si no, cualquiera podría
    sembrar «respuestas recuperadas» en el historial de otro."""
    token = (authorization or "").removeprefix("Bearer ").strip()
    if not token or _sb is None:
        return None
    try:
        r = await asyncio.to_thread(_sb.auth.get_user, token)
        return str(r.user.id) if r and r.user else None
    except Exception:
        return None


async def _abrir(fila: dict) -> bool:
    try:
        await asyncio.to_thread(
            lambda: _sb.table("respuestas_chat").upsert(fila, on_conflict="peticion_id").execute())
        return True
    except Exception as e:
        print(f"   💾 RESPALDO: no pude abrir la fila {fila.get('peticion_id')}: {type(e).__name__}: {str(e)[:160]}")
        return False


async def _cerrar(apertura: "asyncio.Task", peticion_id: str, user_id_task: "asyncio.Task",
                  crudo: str, terminada: bool) -> None:
    try:
        if not await apertura:
            return
        respuesta = texto_para_historial(crudo)[:TOPE_RESPUESTA]
        estado = "terminada" if terminada else "interrumpida"
        await asyncio.to_thread(
            lambda: _sb.table("respuestas_chat").update({
                "respuesta": respuesta,
                "estado": estado,
                "terminada_at": datetime.now(timezone.utc).isoformat(),
            }).eq("peticion_id", peticion_id).eq("user_id", user_id_task.result()).execute())
        print(f"   💾 RESPALDO {estado}: {peticion_id} ({len(respuesta):,} car.)")
    except Exception as e:
        print(f"   💾 RESPALDO: no pude cerrar {peticion_id}: {type(e).__name__}: {str(e)[:160]}")


def _en_segundo_plano(coro) -> "asyncio.Task":
    t = asyncio.create_task(coro)
    _tareas.add(t)
    t.add_done_callback(_tareas.discard)
    return t


async def con_respaldo(
    flujo: AsyncIterator[str],
    *,
    authorization: Optional[str],
    peticion_id: Optional[str],
    conversation_id: Optional[str],
    expediente_id: Optional[str],
    pregunta: str,
    estado: Optional[str],
) -> AsyncIterator[str]:
    """Deja pasar el stream tal cual y guarda una copia de lo que salió.

    Sin `peticion_id` válido (o sin token, o sin Supabase) es un paso directo:
    ni una línea de comportamiento distinto."""
    peticion = _uuid_o_none(peticion_id)
    if not peticion or _sb is None or not authorization:
        async for trozo in flujo:
            yield trozo
        return

    async def _abrir_fila() -> bool:
        uid = await quien
        if not uid:
            print(f"   💾 RESPALDO: token sin usuario, {peticion} no se respalda")
            return False
        return await _abrir({
            "peticion_id": peticion,
            "user_id": uid,
            "conversation_id": _uuid_o_none(conversation_id),
            "expediente_id": _uuid_o_none(expediente_id),
            "estado_mx": (estado or None),
            "pregunta": (pregunta or "")[:TOPE_PREGUNTA],
            "respuesta": None,
            "estado": "en_curso",
            "terminada_at": None,
            "guardada_at": None,
        })

    quien = _en_segundo_plano(_usuario_del_token(authorization))
    apertura = _en_segundo_plano(_abrir_fila())

    partes: list[str] = []
    terminada = False
    try:
        async for trozo in flujo:
            partes.append(trozo)
            yield trozo
        terminada = True
    finally:
        # Aquí puede llegarse por cancelación (el navegador se fue): no se
        # espera nada en este punto, la escritura va en su propia tarea.
        if partes or terminada:
            _en_segundo_plano(_cerrar(apertura, peticion, quien, "".join(partes), terminada))
        # Si a ESTE generador lo cerraron a medias (aclose), el de dentro se
        # cierra ya, como se cerraba antes de existir la envoltura, y no
        # cuando el recolector de basura lo encuentre.
        if not terminada:
            try:
                await flujo.aclose()
            except Exception:
                pass


# ── Dónde se atora el envío ───────────────────────────────────────────────
#
# El 14-oct, a las 05:10 y a las 05:16, el navegador de Roberto creó la
# conversación y se quedó en «Leyendo la consulta…» sin mandar NUNCA el /chat
# (Render no recibió nada suyo en esos nueve minutos). Entre el clic y el fetch
# hay varias esperas —la sesión, la cuota, la carpeta— y desde aquí no se ve
# cuál se quedó colgada. El navegador avisa por este camino cuando un envío
# lleva demasiado sin salir, o sin leerse, y deja una línea que se cuenta en
# Render: «⏳ ENVIO_TRABADO etapa=… ms=…».
#
# Sin token ni base: es sólo un renglón de registro. Se acepta texto plano
# para que `navigator.sendBeacon` funcione sin preflight, y todo se recorta.

_ETAPAS = {"sesion", "cuota", "contexto", "fetch", "primer_trozo", "lectura", "guardado"}


@router.post("/chat/traba")
async def chat_traba(request: Request):
    try:
        crudo = (await request.body())[:2000]
        d = json.loads(crudo.decode("utf-8", errors="replace") or "{}")
    except Exception:
        return {"ok": False}
    etapa = str(d.get("etapa") or "")[:20]
    if etapa not in _ETAPAS:
        return {"ok": False}
    try:
        ms = int(d.get("ms") or 0)
    except (TypeError, ValueError):
        ms = 0
    peticion = _uuid_o_none(d.get("peticion_id")) or "-"
    usuario = _uuid_o_none(d.get("user_id")) or "-"
    oculta = "1" if d.get("oculta") else "0"
    pestanas = str(d.get("pestanas") or "")[:3]
    print(f"   ⏳ ENVIO_TRABADO etapa={etapa} ms={ms} oculta={oculta} pestanas={pestanas or '-'} "
          f"peticion={peticion} usuario={usuario}")
    return {"ok": True}
