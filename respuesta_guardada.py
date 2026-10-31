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
    (`texto_para_historial`), «terminada» —también si el navegador se fue a
    la mitad: desde el 30-oct el flujo sigue sin él, ver «El trabajo no se
    va con la pantalla»—; si se detuvo, «interrumpida» con lo que hubiera.

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

TAMBIÉN EL ANÁLISIS DE DOCUMENTOS (28-oct-2026). Un Platinum del Tribunal
Colegiado de Tlaxcala (folio 3640-01) trabajaba tras Cisco Umbrella, que
retiene el streaming y lo entrega entero al terminar. De sus cuatro análisis,
la pantalla abortó tres a los 75 s de silencio aunque el servidor los terminó
(144, 110 y 101 s), y /analyze-document no guardaba copia: se perdieron el
análisis y el texto leído del documento. Ahora la ruta pasa por aquí con
`formato="documento"`: su flujo son eventos SSE, y la copia se arma como la
armaría la pantalla —la pregunta con el documento oculto dentro, la respuesta
con los `token`— para que la pantalla la espere cuando la red retiene el flujo
y la recuperación la pase al historial si nadie lo hizo.
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


# ── El análisis de un documento, como lo guardaría la pantalla ────────────
#
# /analyze-document no manda texto plano sino eventos SSE (`data: {…}`), y
# `handleDocumentSubmit` (src/app/chat/page.tsx) arma con ellos dos mensajes:
#
#   · la PREGUNTA: lo que el abogado ve («📄 **Documento adjunto:** …») y,
#     oculto entre DOCUMENTO_INICIO y DOCUMENTO_FIN, el texto que se leyó del
#     documento (evento `documento`), para que los turnos siguientes lo tengan;
#   · la RESPUESTA: los `token` uno tras otro, tal cual llegan; un `error` la
#     sustituye por «❌ Error al analizar documento: …».
#
# `progreso`, `limite_plan`, `done` y los latidos no se guardan.

_ERROR_DOCUMENTO = "❌ Error al analizar documento: "


def encabezado_de_adjunto(nombre: str, pregunta: str) -> str:
    """Espejo de `encabezadoDeAdjunto` (src/lib/memoria-conversacion.ts): lo
    que ve el abogado cuando la pantalla no mandó su `mensaje_visible`."""
    return f"📄 **Documento adjunto:** {nombre}\n\n{pregunta}"


def mensaje_con_documento(visible: str, nombre: str, texto: str, recortado: bool = False) -> str:
    """Espejo de `mensajeConDocumento` (src/lib/memoria-conversacion.ts)."""
    aviso = "; es la primera parte, el documento es más largo" if recortado else ""
    return (f"{visible}\n\n<!-- DOCUMENTO_INICIO -->\n"
            f"CONTENIDO DEL DOCUMENTO ADJUNTO «{nombre}» (texto leído por Iurexia{aviso}):\n\n"
            f"{texto}\n<!-- DOCUMENTO_FIN -->")


def _pregunta_con_documento(visible: str, nombre: str, texto: str, recortado: bool) -> str:
    """Con el tope de la columna: si no cabe se recorta el DOCUMENTO, no el
    mensaje, para que el cierre <!-- DOCUMENTO_FIN --> no se pierda."""
    m = mensaje_con_documento(visible, nombre, texto, recortado)
    if len(m) <= TOPE_PREGUNTA:
        return m
    sobra = len(m) - TOPE_PREGUNTA + 100   # el aviso de «primera parte» también ocupa
    return mensaje_con_documento(visible, nombre, texto[:max(0, len(texto) - sobra)], True)


def _eventos_sse(crudo: str):
    for bloque in crudo.split("\n\n"):
        for linea in bloque.split("\n"):
            if not linea.startswith("data: "):
                continue
            try:
                d = json.loads(linea[6:])
            except ValueError:
                continue
            if isinstance(d, dict):
                yield d


def armar_documento(crudo: str, visible: str) -> tuple[str, str]:
    """(pregunta, respuesta) de un /analyze-document, con el mismo orden de
    preguntas que la pantalla hace a cada evento."""
    pregunta, respuesta = visible, ""
    for d in _eventos_sse(crudo):
        if d.get("token"):
            respuesta += str(d["token"])
        elif d.get("progreso"):
            continue
        elif isinstance(d.get("documento"), dict) and d["documento"].get("texto"):
            doc = d["documento"]
            pregunta = _pregunta_con_documento(visible, str(doc.get("nombre") or ""),
                                               str(doc["texto"]), bool(doc.get("recortado")))
        elif d.get("limite_plan"):
            continue
        elif d.get("error"):
            respuesta = _ERROR_DOCUMENTO + str(d["error"])
    return pregunta, respuesta


def _armar_chat(crudo: str, pregunta: str) -> tuple[str, str]:
    return pregunta, texto_para_historial(crudo)


# Cómo se convierte lo que salió por el flujo en (pregunta, respuesta).
_ARMADORES = {"chat": _armar_chat, "documento": armar_documento}


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
                  crudo: str, terminada: bool, formato: str = "chat", pregunta: str = "") -> None:
    try:
        if not await apertura:
            return
        armar = _ARMADORES.get(formato, _armar_chat)
        pregunta_final, respuesta = armar(crudo, pregunta)
        respuesta = respuesta[:TOPE_RESPUESTA]
        estado = "terminada" if terminada else "interrumpida"
        cambios = {
            "respuesta": respuesta,
            "estado": estado,
            "terminada_at": datetime.now(timezone.utc).isoformat(),
        }
        # El documento leído entra a la pregunta al cerrar: al abrir la fila
        # todavía no se había leído.
        if pregunta_final != pregunta:
            cambios["pregunta"] = pregunta_final[:TOPE_PREGUNTA]
        await asyncio.to_thread(
            lambda: _sb.table("respuestas_chat").update(cambios)
            .eq("peticion_id", peticion_id).eq("user_id", user_id_task.result()).execute())
        print(f"   💾 RESPALDO {estado}: {peticion_id} ({len(respuesta):,} car.)"
              + (f" · {formato}" if formato != "chat" else ""))
    except Exception as e:
        print(f"   💾 RESPALDO: no pude cerrar {peticion_id}: {type(e).__name__}: {str(e)[:160]}")


def _en_segundo_plano(coro) -> "asyncio.Task":
    t = asyncio.create_task(coro)
    _tareas.add(t)
    t.add_done_callback(_tareas.discard)
    return t


# ── El trabajo no se va con la pantalla ───────────────────────────────────
#
# EL TELÉFONO QUE SE VA A WHATSAPP (30-oct-2026). Rubén Ávila (Platinum,
# Android) pidió el 29-oct dos análisis seguidos:
#
#   · el primero lo terminó el servidor y Render le entregó los 170 KB en
#     103 s (10:49:32 CDMX), pero su pantalla ya no estaba para pintarlos y
#     a las 10:52:29 guardó «❌ Error al analizar documento. network error»;
#   · el segundo lo cortó su propio teléfono a los 1.8 s —la página se
#     recargó al volver de WhatsApp, de donde sacó el PDF—, y entonces el
#     servidor DEJÓ de escribir: cerrar la conexión cancelaba el flujo, la
#     copia quedaba «interrumpida» y vacía, y no hubo nada que recuperar.
#
# En los catorce días anteriores, 53 análisis acabaron en «Error al analizar
# documento» y cada día entre 4 y 15 respuestas del chat quedaron
# «interrumpidas». La pantalla ya sabe esperar la copia y pasarla al
# historial (src/lib/analisis-documento.ts y respuestas-servidor.ts en el
# front); lo que faltaba era que hubiera algo que esperar.
#
# Ahora, con copia, el flujo corre en SU PROPIA TAREA y la respuesta HTTP
# sólo lo va leyendo. Si la pantalla se va —se cayó la red, el teléfono
# congeló la pestaña, se cerró el navegador—, la tarea sigue hasta el final
# y la copia queda «terminada»: la pantalla que espera la recibe, y si ya
# no hay pantalla, la recuperación la pone en su conversación.
#
# Sólo se corta en dos casos:
#   · el abogado pulsó «Detener»: la pantalla pone su copia en
#     «interrumpida» mientras sigue «en_curso» (la política de la tabla le
#     deja actualizar sus propias filas) y aquí se mira cada pocos segundos;
#   · pasó TOPE_SIN_PANTALLA_S sin pantalla, que ningún análisis necesita.
# En los dos el flujo se cancela como antes —y devuelve la consulta si no
# había escrito nada— y la copia queda «interrumpida» con lo que hubiera.

TOPE_SIN_PANTALLA_S = 20 * 60
MIRAR_CADA_S = 3.0

# Las tareas que escriben respuestas con copia, para que el apagado las espere.
_productores: set = set()

_FIN = object()


class _Fallo:
    """Una excepción del flujo que viaja por la cola hasta la pantalla."""
    __slots__ = ("error",)

    def __init__(self, error: BaseException):
        self.error = error


async def _la_detuvo_la_pantalla(peticion: str) -> bool:
    """¿La pantalla pidió parar? Ante la duda, no: seguir escribiendo sólo
    cuesta tokens; parar por un fallo de lectura tiraría el trabajo."""
    try:
        r = await asyncio.to_thread(
            lambda: _sb.table("respuestas_chat").select("estado")
            .eq("peticion_id", peticion).limit(1).execute())
        filas = r.data or []
        return bool(filas) and filas[0].get("estado") == "interrumpida"
    except Exception:
        return False


async def _sin_pantalla(productor: "asyncio.Task", peticion: str,
                        tope_s: float, cada_s: float) -> None:
    """Acompaña a un flujo que se quedó sin pantalla hasta que termine, y lo
    corta sólo si la pantalla lo pide o si pasa del tope."""
    print(f"   📵 SIN PANTALLA: {peticion} se sigue escribiendo en el servidor")
    loop = asyncio.get_running_loop()
    limite = loop.time() + tope_s
    motivo = None
    while not productor.done():
        await asyncio.wait({productor}, timeout=cada_s)
        if productor.done():
            break
        if loop.time() >= limite:
            motivo = f"pasó de {tope_s:.0f} s sin pantalla"
        elif await _la_detuvo_la_pantalla(peticion):
            motivo = "la pantalla pidió detenerla"
        if motivo:
            productor.cancel()
            break
    if motivo:
        print(f"   📵 SIN PANTALLA: {peticion} se corta: {motivo}")


async def con_respaldo(
    flujo: AsyncIterator[str],
    *,
    authorization: Optional[str],
    peticion_id: Optional[str],
    conversation_id: Optional[str],
    expediente_id: Optional[str],
    pregunta: str,
    estado: Optional[str],
    formato: str = "chat",
    tope_sin_pantalla_s: float = TOPE_SIN_PANTALLA_S,
    mirar_cada_s: float = MIRAR_CADA_S,
) -> AsyncIterator[str]:
    """Deja pasar el stream tal cual y guarda una copia de lo que salió. Con
    copia, el stream se escribe aunque la pantalla se vaya (ver arriba).

    `formato` dice cómo leer lo que salió: "chat" (texto con marcadores) o
    "documento" (los eventos SSE de /analyze-document; `pregunta` es entonces
    lo que ve el abogado, sin el documento).

    Sin `peticion_id` válido (o sin token, o sin Supabase) es un paso directo:
    ni una línea de comportamiento distinto —incluido que cerrar la conexión
    corta el flujo, porque sin copia no habría dónde dejar lo escrito—."""
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

    cola: asyncio.Queue = asyncio.Queue()
    vivo = {"pantalla": True, "acabo": False}

    async def producir() -> None:
        partes: list[str] = []
        terminada = False
        try:
            async for trozo in flujo:
                partes.append(trozo)
                if vivo["pantalla"]:
                    cola.put_nowait(trozo)
            terminada = True
        except Exception as e:
            # Un error del flujo llega a la pantalla igual que antes.
            if vivo["pantalla"]:
                cola.put_nowait(_Fallo(e))
        finally:
            vivo["acabo"] = True
            # La escritura va en su propia tarea: aquí se puede llegar por
            # cancelación (Detener, tope o apagado) y no se espera nada. Se
            # cierra SIEMPRE, también sin una letra: una copia que se queda
            # «en_curso» la pantalla la esperaría hasta su tope.
            _en_segundo_plano(_cerrar(apertura, peticion, quien, "".join(partes), terminada,
                                      formato, pregunta or ""))
            # Cortado a medias: el de dentro se cierra ya, para que su propio
            # final (la devolución de la consulta sin una letra) corra ahora.
            if not terminada:
                try:
                    await flujo.aclose()
                except BaseException:
                    pass
            cola.put_nowait(_FIN)

    productor = asyncio.create_task(producir())
    _productores.add(productor)
    productor.add_done_callback(_productores.discard)

    try:
        while True:
            item = await cola.get()
            if item is _FIN:
                return
            if isinstance(item, _Fallo):
                raise item.error
            yield item
    finally:
        # Aquí se llega también cuando la pantalla se fue: Starlette cancela
        # esta lectura o la cierra. El productor NO se cancela: se le deja de
        # pasar lo que escribe a una cola que ya nadie lee y se le acompaña
        # hasta el final. Nada se espera aquí.
        if not vivo["acabo"]:
            vivo["pantalla"] = False
            _en_segundo_plano(_sin_pantalla(productor, peticion, tope_sin_pantalla_s, mirar_cada_s))


async def esperar_productores(segundos: float) -> None:
    """Al apagar el servidor (despliegue o reciclaje del worker): lo que se
    sigue escribiendo sin pantalla tiene unos segundos para acabar; lo demás
    se corta y su copia queda «interrumpida» con lo escrito, en vez de
    quedarse «en_curso» para siempre."""
    vivos = [t for t in _productores if not t.done()]
    if not vivos:
        return
    print(f"   📵 Apagado: {len(vivos)} respuesta(s) en curso; se esperan hasta {segundos:.0f} s")
    _, pendientes = await asyncio.wait(vivos, timeout=segundos)
    for t in pendientes:
        t.cancel()
    if pendientes:
        print(f"   📵 Apagado: {len(pendientes)} respuesta(s) cortadas; su copia queda interrumpida")
        await asyncio.wait(pendientes, timeout=5)
    cierres = [t for t in _tareas if not t.done()]
    if cierres:
        await asyncio.wait(cierres, timeout=5)


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
    # EL DOCUMENTO QUE NO SALIÓ (29-oct-2026). 7 de 16 «se cortó la conexión»
    # de la última semana nunca llegaron a Render: desde aquí no se ve por qué.
    # La pantalla manda ahora el error, el peso, el tipo y el intento; sólo
    # caracteres simples y recortado, que es un renglón de registro.
    detalle = re.sub(r"[^\w .:/|=+%-]", "", str(d.get("detalle") or ""))[:160]
    print(f"   ⏳ ENVIO_TRABADO etapa={etapa} ms={ms} oculta={oculta} pestanas={pestanas or '-'} "
          f"peticion={peticion} usuario={usuario}" + (f" detalle={detalle}" if detalle else ""))
    return {"ok": True}
