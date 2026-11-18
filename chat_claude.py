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
    Claude se aparta CLAUDE_PAUSA_SIN_SALDO_MIN minutos (360) y luego se
    vuelve a probar: el saldo se renueva cada mes.
  · Clave o modelo que no sirven (401/403/404): se aparta 60 minutos.
  · Lo demás (429, 529, 5xx, red, una petición que no le gustó): sólo esa
    consulta va a luna.

INTERRUPTOR: CHAT_CLAUDE en Render.
  · "prueba" (por omisión): sólo los correos de CHAT_CLAUDE_CORREOS, o los
    administradores si está vacía. Para medir la calidad antes de abrirlo.
  · "on": todos.   · "off": nadie.

Los escalones de esfuerzo de Haiku (low/medium/high) son variables:
CLAUDE_ESFUERZO_BUSCAR (medium), CLAUDE_ESFUERZO_BASICO (medium) y
CLAUDE_ESFUERZO_PRO (high).

GASTO. Cada respuesta deja en el registro «🟣 CLAUDE …» sus tokens y su costo
estimado, y la cuenta del mes en este proceso (no es la factura: hay varios
procesos y se reinician). El tope real lo pone Anthropic: al acabarse el saldo
contesta 402 y aquí se cae a luna. Si la organización tuviera recarga
automática con tarjeta, ese 402 no llegaría: conviene fijar el límite de gasto
mensual en la consola de Anthropic.
"""
from __future__ import annotations

import asyncio
import os
import time
import types
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Dict, List, Optional

MODELO = os.getenv("CLAUDE_CHAT_MODELO", "claude-haiku-5-5").strip() or "claude-haiku-5-5"

# Precios de Claude Haiku 5.5, USD por millón de tokens, por tamaño del prompt.
_PRECIO_HASTA_100K = (0.10, 0.50)
_PRECIO_MAS_DE_100K = (0.50, 2.50)
_LECTURA_CACHE = 0.1      # × la entrada
_ESCRITURA_CACHE = 1.25   # × la entrada (caché de 5 minutos)

_RUTAS = {"buscar", "basico", "pro"}
_ESFUERZOS = {"low", "medium", "high", "xhigh", "max"}
_ESFUERZO_POR_OMISION = {"buscar": "medium", "basico": "medium", "pro": "high"}

_sb = None
_admins = lambda: ()   # noqa: E731 — main.py la cambia por la suya
_cliente = None
_pausa_hasta = 0.0
_pausa_motivo = ""
_correos_cache: Dict[str, tuple] = {}     # user_id -> (correo, cuándo)
_mes: Dict[str, Any] = {"clave": "", "usd": 0.0, "respuestas": 0}

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
    m = os.getenv("CHAT_CLAUDE", "prueba").strip().lower()
    return m if m in ("on", "off", "prueba") else "prueba"


def esfuerzo(ruta: str) -> str:
    e = os.getenv(f"CLAUDE_ESFUERZO_{ruta.upper()}", "").strip().lower()
    return e if e in _ESFUERZOS else _ESFUERZO_POR_OMISION.get(ruta, "medium")


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
    clave = datetime.now(timezone.utc).strftime("%Y-%m")
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


def _trozo(texto: str):
    """Un trozo con la forma de los de OpenAI que lee el bucle del chat."""
    delta = types.SimpleNamespace(content=texto, reasoning_content=None)
    return types.SimpleNamespace(choices=[types.SimpleNamespace(delta=delta, finish_reason=None)])


class Intento:
    """La respuesta de Claude para una consulta: se arranca, se espera la
    primera letra latiendo, y luego se lee como un flujo de OpenAI.

        intento = Intento(llm_messages, ruta="buscar", max_tokens=30000)
        async for _ in intento.esperar():      # late mientras piensa
            yield "<!--PING-->"
        flujo = intento.flujo()                # None → seguir con luna
    """

    def __init__(self, mensajes: List[Dict[str, Any]], *, ruta: str, max_tokens: int,
                 peticion: Optional[str] = None):
        self.ruta = ruta
        self.max_tokens = int(max_tokens or 30000)
        self.peticion = (peticion or "-")[:8]
        self.esfuerzo = esfuerzo(ruta)
        self._convertido = convertir(mensajes)
        self._cola: asyncio.Queue = asyncio.Queue()
        self._tarea: Optional[asyncio.Task] = None
        self._primera: Optional[str] = None
        self.motivo_fallo = ""
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
                    print(f"   🟣 CLAUDE {MODELO} ({self.ruta}, {self.esfuerzo}): primera letra en "
                          f"{time.perf_counter() - self._t0:.1f}s")
                elif isinstance(item, BaseException):
                    self.motivo_fallo = f"{type(item).__name__}: {str(item)[:160]}"
                    accion = _clasificar(item)
                    if accion == "saldo":
                        _pausar(float(os.getenv("CLAUDE_PAUSA_SIN_SALDO_MIN", "360")), "sin saldo")
                    elif accion == "clave":
                        _pausar(60, f"clave o modelo: {getattr(item, 'status_code', '?')}")
                else:
                    self.motivo_fallo = self.motivo_fallo or "terminó sin escribir"
                return
        finally:
            if not pendiente.done():
                pendiente.cancel()
            # Si a la espera la cortaron (Detener), Claude no sigue gastando.
            if self._primera is None and not self.motivo_fallo:
                self.cerrar()

    def flujo(self):
        """El flujo con forma de OpenAI, o None si hay que seguir con luna."""
        if self._primera is None:
            if self.motivo_fallo:
                print(f"   🟣 CLAUDE no contestó ({self.motivo_fallo}) → sigue luna "
                      f"(peticion={self.peticion})")
            self.cerrar()
            return None
        return self._trozos()

    async def _trozos(self):
        try:
            yield _trozo(self._primera)
            while True:
                item = await self._cola.get()
                if item is _FIN:
                    return
                if isinstance(item, BaseException):
                    raise item
                yield _trozo(item)
        finally:
            self.cerrar()

    def cerrar(self) -> None:
        if self._tarea is not None and not self._tarea.done():
            self._tarea.cancel()

    async def _producir(self, cliente) -> None:
        system, mensajes = self._convertido
        hubo_texto = False
        try:
            kw: Dict[str, Any] = dict(
                model=MODELO,
                max_tokens=self.max_tokens,
                messages=mensajes,
                output_config={"effort": self.esfuerzo},
            )
            if system:
                kw["system"] = system
            async with cliente.messages.stream(**kw) as s:
                async for ev in s:
                    if ev.type == "content_block_delta" and getattr(ev.delta, "type", "") == "text_delta":
                        if ev.delta.text:
                            hubo_texto = True
                            await self._cola.put(ev.delta.text)
                final = await s.get_final_message()
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

    def _registrar(self, final) -> None:
        uso = getattr(final, "usage", None)
        usd = costo(uso)
        mes = _sumar_al_mes(usd)
        print(f"   🟣 CLAUDE {MODELO} ({self.ruta}, {self.esfuerzo}) terminó en "
              f"{time.perf_counter() - self._t0:.1f}s · fin {final.stop_reason} · "
              f"entrada {int(getattr(uso, 'input_tokens', 0) or 0):,} "
              f"(caché leída {int(getattr(uso, 'cache_read_input_tokens', 0) or 0):,}, "
              f"escrita {int(getattr(uso, 'cache_creation_input_tokens', 0) or 0):,}) · "
              f"salida {int(getattr(uso, 'output_tokens', 0) or 0):,} · ≈{usd:.4f} USD · "
              f"mes en este proceso ≈{mes:.2f} USD ({_mes['respuestas']} respuestas) "
              f"peticion={self.peticion}")
