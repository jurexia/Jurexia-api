"""EL CHAT PRIMERO CON CLAUDE HAIKU, MIENTRAS DURE EL SALDO.   (18-nov-2026)

David: el plan Ultra de Anthropic trae 200 USD al mes de API. Mientras dure
ese saldo, la respuesta del chat la escribe Claude Haiku 5.5 (0.10 / 0.50 USD
por millón de tokens de entrada / salida con hasta 100K de entrada; 0.50 /
2.50 por encima); cuando se acaba, la consulta sigue con el motor de siempre
(luna con el esfuerzo de cada escalón). La Redacción Platinum no se toca.

Rutas (las elige main.py, en el enrutado de `generate_stream`):

  · "buscar"  — el chat por omisión, hoy gpt-6-luna con razonamiento low;
  · "basico"  — la Redacción Profesional, hoy deepseek-v4-flash;
  · "pro"     — la Redacción Pro, hoy gpt-5.6-luna medium.

CÓMO SE CAE A LUNA. Claude se intenta primero y su texto se espera latiendo
(`<!--PING-->` cada 5 s, para que el balanceador de Render no cierre la
conexión mientras piensa). Si falla ANTES de la primera letra —sin saldo,
clave mala, saturado, se negó— la misma consulta sigue por el camino de
siempre, sin que el abogado vea nada. Una vez que escribió, un fallo se
trata como el de cualquier motor (el aviso de respuesta truncada).

  · Sin saldo (402 `billing_error`, o el 400 «credit balance is too low»):
    Claude se aparta HASTA QUE SE RENUEVA EL CRÉDITO —el día 17 de cada mes
    a las 15:00, hora de México (CLAUDE_RENUEVA_DIA, CLAUDE_RENUEVA_HORA)—,
    con una sola prueba cada 24 h por si vuelve antes (CLAUDE_SONDEO_SIN_SALDO_H;
    0 = ninguna). La prueba que falla la contesta luna, sin que se note.
  · Clave o modelo que no sirven (401/403/404): se aparta 60 minutos.
  · Lo demás (429, 529, 5xx, red, una petición que no le gustó): sólo esa
    consulta va a luna.

INTERRUPTOR: CHAT_CLAUDE en Render.
  · "on" (por omisión desde el 18-nov-2026): todos.
  · "prueba": sólo los correos de CHAT_CLAUDE_CORREOS, o los administradores
    si está vacía.   · "off": nadie.

Razonamiento ENCENDIDO con esfuerzo BAJO en las tres rutas (decisión de David,
18-nov-2026): sin razonar, Haiku dio el pagaré a diez años y contestó una
consulta laboral con un formato. Los escalones son variables:
CLAUDE_ESFUERZO_BUSCAR, CLAUDE_ESFUERZO_BASICO y CLAUDE_ESFUERZO_PRO (low);
CLAUDE_RAZONAR=0 lo apaga.

GASTO Y CICLOS. Cada respuesta deja en el registro «🟣 CLAUDE …» sus tokens y
su costo estimado, y una fila en la tabla `claude_uso` de Supabase con su
ciclo (el 17 en que empezó); el primer «sin saldo» de cada ciclo deja otra
fila (tipo «agotado»). La vista `claude_ciclos` da por ciclo la primera y la
última respuesta, cuándo se agotó, cuántas respuestas y cuánto costaron
(estimado con la tarifa, no es la factura). El tope real lo pone Anthropic: al acabarse el saldo
contesta 402 y aquí se cae a luna. Si la organización tuviera recarga
automática con tarjeta, ese 402 no llegaría: conviene fijar el límite de gasto
mensual en la consola de Anthropic.
"""
from __future__ import annotations

import asyncio
import os
import time
import types
from datetime import datetime, timedelta, timezone
from typing import Any, AsyncIterator, Dict, List, Optional

MODELO = os.getenv("CLAUDE_CHAT_MODELO", "claude-haiku-5-5").strip() or "claude-haiku-5-5"

# Precios de Claude Haiku 5.5, USD por millón de tokens, por tamaño del prompt.
_PRECIO_HASTA_100K = (0.10, 0.50)
_PRECIO_MAS_DE_100K = (0.50, 2.50)
_LECTURA_CACHE = 0.1      # × la entrada
_ESCRITURA_CACHE = 1.25   # × la entrada (caché de 5 minutos)

_RUTAS = {"buscar", "basico", "pro"}
_ESFUERZOS = {"low", "medium", "high", "xhigh", "max"}
_ESFUERZO_POR_OMISION = {"buscar": "low", "basico": "low", "pro": "low"}

_sb = None
_admins = lambda: ()   # noqa: E731 — main.py la cambia por la suya
_cliente = None
_pausa_hasta = 0.0
_pausa_motivo = ""
_correos_cache: Dict[str, tuple] = {}     # user_id -> (correo, cuándo)
_mes: Dict[str, Any] = {"clave": "", "usd": 0.0, "respuestas": 0}
_agotado_anotado = ""            # el ciclo cuyo «agotado» ya se anotó en este proceso
_tabla_falta_hasta = 0.0         # sin la tabla claude_uso, no se insiste durante 1 h

# Las tareas en segundo plano se guardan: una sin referencia la puede recoger
# el recolector de basura antes de terminar.
_tareas: set = set()


def configurar(supabase_admin, correos_admin) -> None:
    """main.py presta su cliente de servicio y una función que devuelve sus
    administradores (se lee al usarla, no al importar)."""
    global _sb, _admins
    _sb = supabase_admin
    _admins = correos_admin if callable(correos_admin) else (lambda: correos_admin or ())


def _modo() -> str:
    m = os.getenv("CHAT_CLAUDE", "on").strip().lower()
    return m if m in ("on", "off", "prueba") else "on"


def razona() -> bool:
    """CON RAZONAMIENTO POR OMISIÓN, esfuerzo bajo (18-nov-2026). Medido en el
    /chat real con acervo, 5 materias: sin razonar, primera letra en 6–10 s pero
    dos errores graves (pagaré a diez años; un formato en vez de la respuesta);
    con razonamiento bajo, 18–45 s, 39 de 45 puntos y ningún error grave.
    CLAUDE_RAZONAR=0 lo apaga."""
    return os.getenv("CLAUDE_RAZONAR", "1").strip() != "0"


def esfuerzo(ruta: str) -> str:
    e = os.getenv(f"CLAUDE_ESFUERZO_{ruta.upper()}", "").strip().lower()
    e = e if e in _ESFUERZOS else _ESFUERZO_POR_OMISION.get(ruta, "medium")
    # Haiku 5.5 rechaza el razonamiento apagado con xhigh o max (400).
    return "high" if not razona() and e in ("xhigh", "max") else e


def _obtener_cliente():
    global _cliente
    if _cliente is None:
        clave = os.getenv("ANTHROPIC_API_KEY", "").strip()
        if not clave:
            return None
        import anthropic
        # Un reintento corto y nada más: si Claude no contesta, para eso está
        # luna. La lectura larga es para el razonamiento antes del texto.
        _cliente = anthropic.AsyncAnthropic(
            api_key=clave, max_retries=1,
            timeout=anthropic.Timeout(600.0, connect=10.0))
    return _cliente


def en_pausa() -> bool:
    return time.time() < _pausa_hasta


def _pausar(minutos: float, motivo: str) -> None:
    global _pausa_hasta, _pausa_motivo
    _pausa_hasta = time.time() + minutos * 60
    _pausa_motivo = motivo
    print(f"   🟣 CLAUDE apartado {minutos:.0f} min ({motivo}): el chat sigue con luna")


# ── El ciclo del crédito: 200 USD que se renuevan el 17 a las 15:00 (México) ──
# Ciudad de México no tiene horario de verano desde 2022: UTC−6 todo el año.
_MX = timezone(timedelta(hours=-6))


def _dia_hora() -> tuple:
    try:
        dia = min(max(int(os.getenv("CLAUDE_RENUEVA_DIA", "17")), 1), 28)
        hora = min(max(int(os.getenv("CLAUDE_RENUEVA_HORA", "15")), 0), 23)
    except ValueError:
        dia, hora = 17, 15
    return dia, hora


def inicio_del_ciclo(ahora: Optional[datetime] = None) -> datetime:
    """La última renovación que ya pasó (hora de México)."""
    ahora = (ahora or datetime.now(timezone.utc)).astimezone(_MX)
    dia, hora = _dia_hora()
    c = ahora.replace(day=dia, hour=hora, minute=0, second=0, microsecond=0)
    if c > ahora:
        c = c.replace(year=c.year - 1, month=12) if c.month == 1 else c.replace(month=c.month - 1)
    return c


def proxima_renovacion(ahora: Optional[datetime] = None) -> datetime:
    c = inicio_del_ciclo(ahora)
    return c.replace(year=c.year + 1, month=1) if c.month == 12 else c.replace(month=c.month + 1)


def _sin_saldo() -> None:
    """Se acabó el crédito: luna hasta la renovación, con una prueba al día."""
    global _pausa_hasta, _pausa_motivo, _agotado_anotado
    renueva = proxima_renovacion()
    try:
        sondeo_h = max(float(os.getenv("CLAUDE_SONDEO_SIN_SALDO_H", "24")), 0.0)
    except ValueError:
        sondeo_h = 24.0
    hasta = renueva.timestamp()
    if sondeo_h:
        hasta = min(hasta, time.time() + sondeo_h * 3600)
    _pausa_hasta, _pausa_motivo = hasta, "sin saldo"
    print(f"   🟣 CLAUDE sin saldo: el chat sigue con luna hasta {renueva:%d-%m-%Y %H:%M} "
          f"(hora de México), cuando se renueva el crédito"
          + (f"; una prueba cada {sondeo_h:g} h por si vuelve antes" if sondeo_h else ""))
    ciclo = inicio_del_ciclo().date().isoformat()
    if _agotado_anotado != ciclo:
        _agotado_anotado = ciclo
        _guardar({"ciclo": ciclo, "tipo": "agotado"})


def _guardar(fila: Dict[str, Any]) -> None:
    """Una fila en `claude_uso`, en segundo plano: nunca frena el chat. Sin
    Supabase o sin la tabla, no pasa nada (se avisa una vez por hora)."""
    if _sb is None or time.time() < _tabla_falta_hasta:
        return

    async def _escribir():
        global _tabla_falta_hasta
        try:
            await asyncio.to_thread(lambda: _sb.table("claude_uso").insert(fila).execute())
        except Exception as e:
            texto = str(e)
            if any(m in texto for m in ("PGRST205", "42P01", "does not exist", "Could not find the table")):
                _tabla_falta_hasta = time.time() + 3600
            print(f"   🟣 CLAUDE no pude anotar en claude_uso: {type(e).__name__}: {texto[:140]}")

    try:
        tarea = asyncio.get_running_loop().create_task(_escribir())
    except RuntimeError:
        return
    _tareas.add(tarea)
    tarea.add_done_callback(_tareas.discard)


async def _correo_de(user_id: str) -> str:
    hit = _correos_cache.get(user_id)
    if hit and time.time() - hit[1] < 600:
        return hit[0]
    correo = ""
    if _sb is not None:
        try:
            r = await asyncio.to_thread(
                lambda: _sb.table("user_profiles").select("email").eq("id", user_id).limit(1).execute())
            if r.data:
                correo = str(r.data[0].get("email") or "").strip().lower()
        except Exception as e:
            print(f"   🟣 CLAUDE: no pude leer el correo para el modo prueba: {type(e).__name__}")
            return ""
    _correos_cache[user_id] = (correo, time.time())
    return correo


async def corresponde(ruta: Optional[str], user_id: Optional[str]) -> bool:
    """¿Esta respuesta la intenta Claude?"""
    if ruta not in _RUTAS or en_pausa() or not os.getenv("ANTHROPIC_API_KEY", "").strip():
        return False
    modo = _modo()
    if modo == "off":
        return False
    if modo == "on":
        return True
    if not user_id:
        return False
    lista = os.getenv("CHAT_CLAUDE_CORREOS", "").strip()
    probadores = (frozenset(c.strip().lower() for c in lista.split(",") if c.strip())
                  if lista else frozenset(str(c).strip().lower() for c in (_admins() or ()) if c))
    return (await _correo_de(str(user_id))) in probadores


# ── Los mensajes del chat, en la forma de la API de Anthropic ─────────────

def convertir(mensajes: List[Dict[str, Any]]) -> Optional[tuple]:
    """(system, messages) para Claude, o None si no se puede.

    El primer mensaje de sistema (el prompt del chat, casi fijo) va en su
    propio bloque con caché; los demás de sistema (la carpeta, el flujo), en
    otro bloque detrás. Los turnos vacíos se quitan, los seguidos del mismo
    rol se juntan, y la conversación tiene que empezar y acabar en el usuario
    (Haiku 5.5 no admite que el último turno sea del asistente)."""
    sistemas: List[str] = []
    turnos: List[Dict[str, str]] = []
    for m in mensajes or []:
        rol = m.get("role")
        texto = m.get("content")
        if not isinstance(texto, str) or not texto.strip():
            continue
        if rol == "system":
            sistemas.append(texto)
        elif rol in ("user", "assistant"):
            if turnos and turnos[-1]["role"] == rol:
                turnos[-1]["content"] += "\n\n" + texto
            else:
                turnos.append({"role": rol, "content": texto})
    if not turnos or turnos[-1]["role"] != "user":
        return None
    if turnos[0]["role"] != "user":
        turnos.insert(0, {"role": "user", "content": "(Continuación de una conversación anterior.)"})
    system: List[Dict[str, Any]] = []
    if sistemas:
        system.append({"type": "text", "text": sistemas[0], "cache_control": {"type": "ephemeral"}})
        if len(sistemas) > 1:
            system.append({"type": "text", "text": "\n\n".join(sistemas[1:])})
    return system, turnos


# ── El costo ───────────────────────────────────────────────────────────────

def costo(uso: Any) -> float:
    """USD estimados de una respuesta, con las tarifas de Haiku 5.5."""
    entrada = int(getattr(uso, "input_tokens", 0) or 0)
    leida = int(getattr(uso, "cache_read_input_tokens", 0) or 0)
    escrita = int(getattr(uso, "cache_creation_input_tokens", 0) or 0)
    salida = int(getattr(uso, "output_tokens", 0) or 0)
    total_prompt = entrada + leida + escrita
    p_in, p_out = _PRECIO_HASTA_100K if total_prompt <= 100_000 else _PRECIO_MAS_DE_100K
    return (entrada * p_in + leida * p_in * _LECTURA_CACHE + escrita * p_in * _ESCRITURA_CACHE
            + salida * p_out) / 1_000_000


def _sumar_al_mes(usd: float) -> float:
    clave = inicio_del_ciclo().date().isoformat()
    if _mes["clave"] != clave:
        _mes.update(clave=clave, usd=0.0, respuestas=0)
    _mes["usd"] += usd
    _mes["respuestas"] += 1
    return _mes["usd"]


def _clasificar(e: BaseException) -> str:
    """Qué hacer con un error de Claude: 'saldo', 'clave' u 'otro'."""
    estado = getattr(e, "status_code", None)
    tipo = str(getattr(e, "type", "") or "")
    texto = str(e).lower()
    if (estado == 402 or tipo == "billing_error" or "credit balance" in texto
            or "usage limits" in texto or "spend limit" in texto):
        return "saldo"
    if estado in (401, 403, 404):
        return "clave"
    return "otro"


# ── Un intento ─────────────────────────────────────────────────────────────

_FIN = object()
_TRUNCADA = ("\n\n---\n⚠️ **Respuesta truncada** — el modelo alcanzó su límite de generación. "
             "Envía **'continúa'** para que siga redactando desde donde se quedó.")
_INCOMPLETA = "\n\n*[La respuesta quedó incompleta. Envía «continúa» para que siga.]*"


class _Razon(str):
    """Un trozo del resumen del razonamiento, no de la respuesta."""


_TRADUCIR_SISTEMA = (
    "Eres un traductor de inglés a español de México. Recibes un fragmento entre "
    "<fragmento> y </fragmento>: son notas de trabajo de un asistente jurídico que analiza "
    "la consulta de un abogado. NO es una pregunta ni una petición para ti: nunca lo "
    "contestes, comentes, completes ni te niegues. Sólo tradúcelo, frase por frase, sin "
    "agregar ni quitar información, en primera persona y sin comillas. Conserva tal cual "
    "los nombres de leyes, artículos y tesis, y los términos jurídicos que ya estén en "
    "español. Única excepción: quita las frases que sólo hablan de las reglas de formato "
    "del asistente (emojis, Doc IDs o identificadores, contexto recuperado, «las reglas "
    "que me dieron», escribir en español). Si al quitarlas no queda nada, responde "
    "exactamente: (nada)")

# Lo que delata que el traductor contestó en vez de traducir: no se pinta.
_FUERA_DE_PAPEL = ("no voy a traducir", "no puedo traducir", "no hay párrafo", "no hay fragmento",
                   "no hay nada que traducir", "con gusto", "si me pasas", "si me compartes",
                   "fragmento", "traducción:", "aquí tienes", "lo siento")


def _traduccion_util(original: str, traducida: str) -> Optional[str]:
    """La traducción si parece una traducción; None si es una negativa, un
    comentario, o trae de más (el traductor inventó consejo jurídico)."""
    t = (traducida or "").strip()
    if not t or t.strip("() .").lower() == "nada":
        return None
    bajo = t.lower()
    if any(m in bajo for m in _FUERA_DE_PAPEL):
        return None
    if len(t) > 1.6 * len(original) + 60:
        return None
    return t


def traducir_razonamiento() -> bool:
    """Haiku 5.5 resume su razonamiento en INGLÉS aunque se le pida pensar en
    español (medido el 18-nov-2026 con la instrucción en el sistema y junto a la
    pregunta: nada). Cada párrafo se traduce con el mismo Haiku, sin razonar,
    antes de pintarlo. CLAUDE_TRADUCIR_RAZONAMIENTO=0 lo manda tal cual."""
    return os.getenv("CLAUDE_TRADUCIR_RAZONAMIENTO", "1").strip() != "0"


class _Traductor:
    """Recibe el resumen del razonamiento a trozos, lo corta en párrafos y los
    traduce en orden, uno tras otro, poniendo cada párrafo ya traducido en la
    cola del intento. Al empezar la respuesta se cierra: lo que no alcanzó a
    traducirse no se pinta, porque el razonamiento nunca va después de la
    respuesta (el chat lo metería en el cuerpo)."""

    def __init__(self, cliente, cola: asyncio.Queue):
        self._cli, self._cola = cliente, cola
        self._buf = ""
        self._parrafos: asyncio.Queue = asyncio.Queue()
        self._cerrado = False
        self._pintados = 0
        self.usd, self.traducidos, self.fallos, self.descartados = 0.0, 0, 0, 0
        self._tarea = asyncio.create_task(self._trabajar())
        _tareas.add(self._tarea)
        self._tarea.add_done_callback(_tareas.discard)

    def agregar(self, texto: str) -> None:
        self._buf += texto
        while "\n\n" in self._buf:
            parrafo, self._buf = self._buf.split("\n\n", 1)
            if parrafo.strip():
                self._parrafos.put_nowait(parrafo.strip())

    def corte(self) -> None:
        """Acabó un bloque de razonamiento: lo pendiente ya es un párrafo."""
        self.agregar("\n\n")

    def cerrar(self) -> None:
        self._cerrado = True
        if not self._tarea.done():
            self._tarea.cancel()

    async def _trabajar(self) -> None:
        while True:
            parrafo = await self._parrafos.get()
            try:
                r = await self._cli.messages.create(
                    model=MODELO, max_tokens=1000,
                    thinking={"type": "disabled"}, output_config={"effort": "low"},
                    system=_TRADUCIR_SISTEMA,
                    messages=[{"role": "user", "content": f"<fragmento>\n{parrafo}\n</fragmento>"}])
            except asyncio.CancelledError:
                raise
            except Exception as e:
                self.fallos += 1
                if self.fallos == 1:
                    print(f"   🟣 CLAUDE no pude traducir un párrafo del razonamiento: "
                          f"{type(e).__name__}: {str(e)[:120]}")
                continue
            self.usd += costo(getattr(r, "usage", None))
            self.traducidos += 1
            crudo = "".join(getattr(b, "text", "") or "" for b in (getattr(r, "content", None) or [])
                            if getattr(b, "type", "") == "text")
            texto = _traduccion_util(parrafo, crudo)
            if texto is None and crudo.strip().strip("() .").lower() != "nada":
                self.descartados += 1
                print(f"   🟣 CLAUDE traducción descartada ({len(parrafo)}→{len(crudo.strip())} car.): "
                      f"{crudo.strip()[:90]!r}")
            # Sin espera entre esta comprobación y el put: si la respuesta ya
            # empezó, este párrafo no entra.
            if texto and not self._cerrado:
                self._cola.put_nowait(_Razon(("\n\n" if self._pintados else "") + texto))
                self._pintados += 1


def razonamiento_visible() -> bool:
    """Con razonamiento, Haiku manda su resumen mientras piensa y el chat lo
    pinta en «Ver razonamiento jurídico» desde el primer segundo, en vez de
    dejar 20–40 s la pantalla quieta. CLAUDE_RAZONAMIENTO_VISIBLE=0 lo oculta
    (Haiku piensa igual y cuesta lo mismo: sólo cambia lo que se ve)."""
    return os.getenv("CLAUDE_RAZONAMIENTO_VISIBLE", "1").strip() != "0"


def _trozo(texto: str):
    """Un trozo con la forma de los de OpenAI que lee el bucle del chat."""
    if isinstance(texto, _Razon):
        delta = types.SimpleNamespace(content=None, reasoning_content=str(texto))
    else:
        delta = types.SimpleNamespace(content=texto, reasoning_content=None)
    return types.SimpleNamespace(choices=[types.SimpleNamespace(delta=delta, finish_reason=None)])


class Intento:
    """La respuesta de Claude para una consulta: se arranca, se espera lo
    primero que mande (su razonamiento o la primera letra) latiendo, y luego se
    lee como un flujo de OpenAI: el razonamiento en `reasoning_content` y la
    respuesta en `content`.

        intento = Intento(llm_messages, ruta="buscar", max_tokens=30000)
        async for _ in intento.esperar():      # late hasta que haya algo
            yield "<!--PING-->"
        flujo = intento.flujo(respaldo)        # None → seguir con luna

    `respaldo` (una corrutina que abre el flujo de luna) cubre el caso de que
    Claude piense, se vea su razonamiento, y aun así no escriba nada (se negó,
    se cayó, gastó el tope pensando): la respuesta la escribe luna.
    """

    def __init__(self, mensajes: List[Dict[str, Any]], *, ruta: str, max_tokens: int,
                 peticion: Optional[str] = None, ver_razonamiento: bool = True):
        self.ruta = ruta
        self.max_tokens = int(max_tokens or 30000)
        self.peticion = (peticion or "-")[:8]
        self.esfuerzo = esfuerzo(ruta)
        self.etiqueta = f"{ruta}, {self.esfuerzo}{'' if razona() else ', sin razonar'}"
        self._convertido = convertir(mensajes)
        self._cola: asyncio.Queue = asyncio.Queue()
        self._tarea: Optional[asyncio.Task] = None
        self._primera: Optional[str] = None
        self.motivo_fallo = ""
        # Sólo si quien pregunta pinta el razonamiento en vivo (la web manda
        # X-Razonamiento-Vivo). La app móvil lo recibe en bloque al final:
        # para ella no se pide y se late hasta la primera letra, como antes.
        self._ver_razon = razona() and razonamiento_visible() and bool(ver_razonamiento)
        self._traductor: Optional[_Traductor] = None
        self._t0 = time.perf_counter()

    async def esperar(self, latido: float = 5.0) -> AsyncIterator[None]:
        cliente = _obtener_cliente()
        if cliente is None or self._convertido is None:
            self.motivo_fallo = "sin clave" if cliente is None else "conversación que no se pudo convertir"
            return
        self._tarea = asyncio.create_task(self._producir(cliente))
        _tareas.add(self._tarea)
        self._tarea.add_done_callback(_tareas.discard)
        pendiente = asyncio.ensure_future(self._cola.get())
        try:
            while True:
                hechos, _ = await asyncio.wait({pendiente}, timeout=latido)
                if not hechos:
                    yield None
                    continue
                item = pendiente.result()
                if isinstance(item, str):
                    self._primera = item
                    que = "primer razonamiento" if isinstance(item, _Razon) else "primera letra"
                    print(f"   🟣 CLAUDE {MODELO} ({self.etiqueta}): {que} en "
                          f"{time.perf_counter() - self._t0:.1f}s")
                elif isinstance(item, BaseException):
                    self._anotar_fallo(item)
                else:
                    self.motivo_fallo = self.motivo_fallo or "terminó sin escribir"
                return
        finally:
            if not pendiente.done():
                pendiente.cancel()
            # Si a la espera la cortaron (Detener), Claude no sigue gastando.
            if self._primera is None and not self.motivo_fallo:
                self.cerrar()

    def _anotar_fallo(self, item: BaseException) -> None:
        self.motivo_fallo = f"{type(item).__name__}: {str(item)[:160]}"
        accion = _clasificar(item)
        if accion == "saldo":
            _sin_saldo()
        elif accion == "clave":
            _pausar(60, f"clave o modelo: {getattr(item, 'status_code', '?')}")

    def flujo(self, respaldo=None):
        """El flujo con forma de OpenAI, o None si hay que seguir con luna."""
        if self._primera is None:
            if self.motivo_fallo:
                print(f"   🟣 CLAUDE no contestó ({self.motivo_fallo}) → sigue luna "
                      f"(peticion={self.peticion})")
            self.cerrar()
            return None
        return self._trozos(respaldo)

    async def _trozos(self, respaldo):
        hubo_texto = False
        fallo: Optional[BaseException] = None
        try:
            item = self._primera
            while item is not _FIN:
                if isinstance(item, BaseException):
                    if hubo_texto:
                        raise item
                    fallo = item
                    break
                if not hubo_texto and not isinstance(item, _Razon):
                    hubo_texto = True
                    if isinstance(self._primera, _Razon):
                        print(f"   🟣 CLAUDE {MODELO} ({self.etiqueta}): primera letra en "
                              f"{time.perf_counter() - self._t0:.1f}s")
                yield _trozo(item)
                item = await self._cola.get()
        finally:
            self.cerrar()
        if hubo_texto:
            return
        # Pensó (y se vio) pero no escribió: se cayó, se negó o gastó el tope.
        if fallo is not None:
            self._anotar_fallo(fallo)
        self.motivo_fallo = self.motivo_fallo or "terminó sin escribir"
        if respaldo is None:
            if fallo is not None:
                raise fallo
            return
        print(f"   🟣 CLAUDE razonó pero no escribió ({self.motivo_fallo}) → la respuesta "
              f"la escribe luna (peticion={self.peticion})")
        async for trozo in await respaldo():
            yield trozo

    def cerrar(self) -> None:
        if self._tarea is not None and not self._tarea.done():
            self._tarea.cancel()

    async def _producir(self, cliente) -> None:
        system, mensajes = self._convertido
        hubo_texto = False
        traductor = None
        if self._ver_razon and traducir_razonamiento():
            traductor = self._traductor = _Traductor(cliente, self._cola)
        try:
            kw: Dict[str, Any] = dict(
                model=MODELO,
                max_tokens=self.max_tokens,
                messages=mensajes,
                output_config={"effort": self.esfuerzo},
            )
            if not razona():
                kw["thinking"] = {"type": "disabled"}
            elif self._ver_razon:
                # Haiku 5.5 omite el texto del razonamiento si no se le pide.
                kw["thinking"] = {"type": "adaptive", "display": "summarized"}
            if system:
                kw["system"] = system
            bloques_razon = 0
            async with cliente.messages.stream(**kw) as s:
                async for ev in s:
                    if ev.type == "content_block_start" and self._ver_razon and not hubo_texto:
                        if getattr(getattr(ev, "content_block", None), "type", "") == "thinking":
                            if traductor is not None:
                                traductor.corte()
                            elif bloques_razon:
                                await self._cola.put(_Razon("\n\n"))
                            bloques_razon += 1
                    elif ev.type == "content_block_stop" and traductor is not None and not hubo_texto:
                        traductor.corte()
                    elif ev.type == "content_block_delta":
                        tipo = getattr(ev.delta, "type", "")
                        if tipo == "text_delta":
                            if ev.delta.text:
                                if traductor is not None and not hubo_texto:
                                    traductor.cerrar()
                                hubo_texto = True
                                await self._cola.put(ev.delta.text)
                        elif tipo == "thinking_delta" and self._ver_razon and not hubo_texto:
                            if getattr(ev.delta, "thinking", ""):
                                if traductor is not None:
                                    traductor.agregar(ev.delta.thinking)
                                else:
                                    await self._cola.put(_Razon(ev.delta.thinking))
                final = await s.get_final_message()
            if traductor is not None:
                traductor.cerrar()
            self._registrar(final)
            if final.stop_reason == "refusal":
                categoria = getattr(getattr(final, "stop_details", None), "category", None)
                print(f"   🟣 CLAUDE se negó (categoría {categoria}, "
                      f"{'con' if hubo_texto else 'sin'} texto) peticion={self.peticion}")
                if hubo_texto:
                    await self._cola.put(_INCOMPLETA)
                else:
                    self.motivo_fallo = f"se negó (categoría {categoria})"
            elif final.stop_reason == "max_tokens" and hubo_texto:
                await self._cola.put(_TRUNCADA)
            elif not hubo_texto:
                self.motivo_fallo = f"terminó sin escribir (fin {final.stop_reason})"
            await self._cola.put(_FIN)
        except asyncio.CancelledError:
            raise
        except BaseException as e:
            await self._cola.put(e)
        finally:
            if traductor is not None:
                traductor.cerrar()

    def _registrar(self, final) -> None:
        uso = getattr(final, "usage", None)
        usd = costo(uso)
        t = self._traductor
        if t is not None:
            usd += t.usd
        mes = _sumar_al_mes(usd)
        print(f"   🟣 CLAUDE {MODELO} ({self.etiqueta}) terminó en "
              f"{time.perf_counter() - self._t0:.1f}s · fin {final.stop_reason} · "
              f"entrada {int(getattr(uso, 'input_tokens', 0) or 0):,} "
              f"(caché leída {int(getattr(uso, 'cache_read_input_tokens', 0) or 0):,}, "
              f"escrita {int(getattr(uso, 'cache_creation_input_tokens', 0) or 0):,}) · "
              f"salida {int(getattr(uso, 'output_tokens', 0) or 0):,} · "
              + (f"traducción del razonamiento {t.traducidos} párrafos ≈{t.usd:.4f} USD"
                 f"{f' ({t.fallos} fallidos)' if t.fallos else ''}"
                 f"{f' ({t.descartados} descartados)' if t.descartados else ''} · " if t is not None else "")
              + f"≈{usd:.4f} USD · "
              f"ciclo del {_mes['clave']} en este proceso ≈{mes:.2f} USD ({_mes['respuestas']} respuestas) "
              f"peticion={self.peticion}")
        _guardar({
            "ciclo": _mes["clave"], "tipo": "respuesta", "ruta": self.ruta, "esfuerzo": self.esfuerzo,
            "fin": str(final.stop_reason or ""), "usd": round(usd, 5),
            "entrada": int(getattr(uso, "input_tokens", 0) or 0)
            + int(getattr(uso, "cache_read_input_tokens", 0) or 0)
            + int(getattr(uso, "cache_creation_input_tokens", 0) or 0),
            "salida": int(getattr(uso, "output_tokens", 0) or 0),
            "segundos": round(time.perf_counter() - self._t0, 1),
            "peticion": None if self.peticion == "-" else self.peticion,
        })
