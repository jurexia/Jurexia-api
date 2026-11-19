#!/usr/bin/env python3
"""Pruebas sin red de chat_claude.py: un cliente de Anthropic de mentira, sin
importar main (nada de arranque ni de credenciales).

    .venv/bin/python test_chat_claude.py
"""
import asyncio
import os
import sys
import types

import chat_claude as cc

FALLOS = []


def ok(cond, que):
    print(f"   {'PASA ' if cond else 'FALLA'}  {que}")
    if not cond:
        FALLOS.append(que)


# ── Un Anthropic de mentira ────────────────────────────────────────────────

class ErrorApi(Exception):
    def __init__(self, status_code, tipo, mensaje="error"):
        super().__init__(mensaje)
        self.status_code, self.type = status_code, tipo


def _texto(t):
    return types.SimpleNamespace(type="content_block_delta", delta=types.SimpleNamespace(type="text_delta", text=t))


def _pensando():
    return types.SimpleNamespace(type="content_block_delta", delta=types.SimpleNamespace(type="signature_delta"))


def _inicio_razon():
    return types.SimpleNamespace(type="content_block_start", content_block=types.SimpleNamespace(type="thinking"))


def _fin_bloque():
    return types.SimpleNamespace(type="content_block_stop")


def _razon(t):
    return types.SimpleNamespace(type="content_block_delta", delta=types.SimpleNamespace(type="thinking_delta", thinking=t))


class _Flujo:
    def __init__(self, guion, final, registro):
        self.guion, self.final, self.registro = guion, final, registro

    async def __aenter__(self):
        if isinstance(self.guion, BaseException):
            raise self.guion
        return self

    async def __aexit__(self, *a):
        return False

    def __aiter__(self):
        return self._eventos()

    async def _eventos(self):
        try:
            for paso in self.guion:
                if isinstance(paso, (int, float)):
                    await asyncio.sleep(paso)
                elif isinstance(paso, BaseException):
                    raise paso
                else:
                    yield paso
        except asyncio.CancelledError:
            self.registro.append("cancelado")
            raise

    async def get_final_message(self):
        return self.final


class _Cliente:
    def __init__(self, guion, final=None, demora_traduccion=0.0, falla_traduccion=False):
        self.llamadas, self.registro, self.traducciones = [], [], []
        self.guion = guion
        self.demora, self.falla, self.respuesta = demora_traduccion, falla_traduccion, None
        self.final = final or types.SimpleNamespace(
            stop_reason="end_turn", stop_details=None,
            usage=types.SimpleNamespace(input_tokens=40_000, cache_read_input_tokens=9_000,
                                        cache_creation_input_tokens=0, output_tokens=5_000))
        self.messages = types.SimpleNamespace(stream=self._stream, create=self._create)

    def _stream(self, **kw):
        self.llamadas.append(kw)
        return _Flujo(self.guion, self.final, self.registro)

    async def _create(self, **kw):          # la traducción de un párrafo
        self.traducciones.append(kw)
        if self.demora:
            await asyncio.sleep(self.demora)
        if self.falla:
            raise ErrorApi(500, "api_error", "no traduzco")
        return types.SimpleNamespace(
            content=[types.SimpleNamespace(type="text", text=self.respuesta or "ES[" + kw["messages"][0]["content"]
                                           .replace("<fragmento>\n", "").replace("\n</fragmento>", "") + "]")],
            usage=types.SimpleNamespace(input_tokens=300, cache_read_input_tokens=0,
                                        cache_creation_input_tokens=0, output_tokens=200))


def _con(cliente):
    cc._cliente = cliente
    cc._pausa_hasta = 0.0
    os.environ["ANTHROPIC_API_KEY"] = "falsa-para-pruebas"


MENSAJES = [
    {"role": "system", "content": "Eres Iurexia."},
    {"role": "system", "content": "CARPETA: amparo 123/2026"},
    {"role": "user", "content": "¿Plazo de la revisión?"},
    {"role": "assistant", "content": "Diez días."},
    {"role": "user", "content": "¿Y la queja?"},
]


async def _correr(intento, latido=0.05, respaldo=None, razon=None):
    latidos = 0
    async for _ in intento.esperar(latido=latido):
        latidos += 1
    flujo = intento.flujo(respaldo)
    if flujo is None:
        return latidos, None, None
    texto, error = "", None
    try:
        async for trozo in flujo:          # como el bucle de main.py
            if trozo.choices and trozo.choices[0].delta:
                texto += getattr(trozo.choices[0].delta, "content", None) or ""
                if razon is not None:
                    razon.append(getattr(trozo.choices[0].delta, "reasoning_content", None) or "")
    except Exception as e:
        error = e
    return latidos, texto, error


def _luna(texto):
    async def abrir():
        async def gen():
            yield types.SimpleNamespace(choices=[types.SimpleNamespace(
                delta=types.SimpleNamespace(content=texto, reasoning_content=None), finish_reason=None)])
        return gen()
    return abrir


print("\n1 · los mensajes del chat, en la forma de Anthropic")
system, msgs = cc.convertir(MENSAJES)
ok(system[0] == {"type": "text", "text": "Eres Iurexia.", "cache_control": {"type": "ephemeral"}}
   and system[1]["text"] == "CARPETA: amparo 123/2026", "el prompt con caché y la carpeta en otro bloque")
ok([m["role"] for m in msgs] == ["user", "assistant", "user"], "los turnos, sin los de sistema")
_, m2 = cc.convertir([{"role": "assistant", "content": "Bienvenido"}, {"role": "user", "content": "a"},
                      {"role": "user", "content": "b"}, {"role": "assistant", "content": "  "}])
ok(m2[0]["role"] == "user" and m2[-1] == {"role": "user", "content": "a\n\nb"},
   "empieza en el usuario, junta los seguidos y quita los vacíos")
ok(cc.convertir([{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]) is None,
   "si acaba en el asistente no se intenta (Haiku no admite prellenado)")

print("\n2 · el costo con las dos tarifas de Haiku 5.5")
u = types.SimpleNamespace(input_tokens=40_000, cache_read_input_tokens=10_000, cache_creation_input_tokens=0, output_tokens=5_000)
ok(abs(cc.costo(u) - (40_000 * 0.10 + 10_000 * 0.01 + 5_000 * 0.50) / 1e6) < 1e-12, f"≤100K: {cc.costo(u):.5f} USD")
u2 = types.SimpleNamespace(input_tokens=120_000, cache_read_input_tokens=0, cache_creation_input_tokens=0, output_tokens=5_000)
ok(abs(cc.costo(u2) - (120_000 * 0.50 + 5_000 * 2.50) / 1e6) < 1e-12, f">100K: {cc.costo(u2):.5f} USD")

print("\n3 · quién va con Claude (CHAT_CLAUDE)")


FILAS = []        # lo que se anota en claude_uso


class _Perfiles:
    def table(self, nombre):
        self.tabla = nombre
        return self

    def insert(self, fila):
        FILAS.append((self.tabla, fila))
        return self

    def select(self, *_):
        return self

    def eq(self, _, uid):
        self.uid = uid
        return self

    def limit(self, _):
        return self

    def execute(self):
        return types.SimpleNamespace(data=[{"email": "admin@prueba.mx" if self.uid == "u-admin" else "abogado@prueba.mx"}])


cc.configurar(_Perfiles(), lambda: ["admin@prueba.mx"])
_con(_Cliente([]))
cc._correos_cache.clear()
os.environ.pop("CHAT_CLAUDE", None)
os.environ.pop("CHAT_CLAUDE_CORREOS", None)
ok(asyncio.run(cc.corresponde("pro", "u-otro")), "por omisión (on): todos")
os.environ["CHAT_CLAUDE"] = "prueba"
ok(asyncio.run(cc.corresponde("buscar", "u-admin")) and not asyncio.run(cc.corresponde("buscar", "u-otro")),
   "prueba: sólo los administradores")
os.environ["CHAT_CLAUDE"] = "on"
ok(not asyncio.run(cc.corresponde("platinum", "u-otro")) and not asyncio.run(cc.corresponde(None, "u-otro")),
   "una ruta que no es suya (Platinum, Precedentes…) nunca")
os.environ["CHAT_CLAUDE"] = "off"
ok(not asyncio.run(cc.corresponde("buscar", "u-admin")), "off: nadie")
os.environ["CHAT_CLAUDE"] = "on"
del os.environ["ANTHROPIC_API_KEY"]
ok(not asyncio.run(cc.corresponde("buscar", "u-otro")), "sin ANTHROPIC_API_KEY, nadie")

print("\n4 · Claude contesta: se late mientras piensa y el texto llega con forma de OpenAI")
cli = _Cliente([_pensando(), 0.2, _texto("## Queja\n\n"), _texto("Procede contra…")])
_con(cli)
lat, texto, err = asyncio.run(_correr(cc.Intento(MENSAJES, ruta="buscar", max_tokens=30000, peticion="abc")))
ok(lat >= 2 and texto == "## Queja\n\nProcede contra…" and err is None, f"{lat} latidos y el texto entero")
kw = cli.llamadas[0]
ok(kw["model"] == "claude-haiku-5-5" and kw["max_tokens"] == 30000
   and kw["output_config"] == {"effort": "low"} and "temperature" not in kw,
   "Haiku 5.5, esfuerzo low, sin temperatura (Haiku la rechaza)")
ok(kw.get("thinking") == {"type": "adaptive", "display": "summarized"},
   "con razonamiento por omisión, y pidiendo su resumen (Haiku 5.5 lo omite si no)")
ok(all(cc.esfuerzo(r) == "low" for r in ("buscar", "basico", "pro")), "esfuerzo bajo en las tres rutas")
os.environ["CLAUDE_RAZONAR"] = "0"
cli = _Cliente([_texto("Sin razonamiento")])
_con(cli)
asyncio.run(_correr(cc.Intento(MENSAJES, ruta="buscar", max_tokens=30000)))
ok(cli.llamadas[0].get("thinking") == {"type": "disabled"}, "CLAUDE_RAZONAR=0: razonamiento apagado")
os.environ["CLAUDE_ESFUERZO_BUSCAR"] = "xhigh"
ok(cc.esfuerzo("buscar") == "high", "sin razonamiento, xhigh baja a high (Haiku daría 400)")
del os.environ["CLAUDE_RAZONAR"]
ok(cc.esfuerzo("buscar") == "xhigh", "con razonamiento, xhigh se respeta")
del os.environ["CLAUDE_ESFUERZO_BUSCAR"]
os.environ["CLAUDE_ESFUERZO_PRO"] = "high"
cli = _Cliente([_texto("Escrito")])
_con(cli)
asyncio.run(_correr(cc.Intento(MENSAJES, ruta="pro", max_tokens=32000)))
ok(cli.llamadas[0]["output_config"] == {"effort": "high"}, "Redacción Pro con esfuerzo high")

print("\n5 · sin saldo: no escribe, la consulta sigue con luna y Claude se aparta")
_con(_Cliente(ErrorApi(402, "billing_error", "Your credit balance is too low")))
lat, texto, err = asyncio.run(_correr(cc.Intento(MENSAJES, ruta="buscar", max_tokens=30000)))
ok(texto is None, "flujo() → None: sigue luna")
ok(cc.en_pausa() and not asyncio.run(cc.corresponde("buscar", "u-otro")), "apartado: las siguientes van directo a luna")
_con(_Cliente(ErrorApi(400, "invalid_request_error", "Your credit balance is too low to access the Anthropic API")))
asyncio.run(_correr(cc.Intento(MENSAJES, ruta="buscar", max_tokens=30000)))
ok(cc.en_pausa(), "también con el 400 «credit balance is too low»")
_con(_Cliente(ErrorApi(529, "overloaded_error")))
lat, texto, err = asyncio.run(_correr(cc.Intento(MENSAJES, ruta="buscar", max_tokens=30000)))
ok(texto is None and not cc.en_pausa(), "saturado: sólo esa consulta va a luna, sin apartarlo")

print("\n6 · se negó antes de escribir → luna; se cortó después → el error de siempre")
_con(_Cliente([_pensando()], types.SimpleNamespace(
    stop_reason="refusal", stop_details=types.SimpleNamespace(category="general_harms"),
    usage=types.SimpleNamespace(input_tokens=1000, output_tokens=10))))
it = cc.Intento(MENSAJES, ruta="buscar", max_tokens=30000)
lat, texto, err = asyncio.run(_correr(it))
ok(texto is None and "se negó" in it.motivo_fallo, f"negativa sin texto → luna ({it.motivo_fallo})")
_con(_Cliente([_texto("Primera parte. "), ErrorApi(500, "api_error", "se cayó")]))
lat, texto, err = asyncio.run(_correr(cc.Intento(MENSAJES, ruta="buscar", max_tokens=30000)))
ok(texto == "Primera parte. " and isinstance(err, ErrorApi),
   "con texto ya enviado, el error sube al manejador del chat (respuesta truncada)")
_con(_Cliente([_texto("Muy largo")], types.SimpleNamespace(
    stop_reason="max_tokens", stop_details=None, usage=types.SimpleNamespace(input_tokens=1, output_tokens=1))))
lat, texto, err = asyncio.run(_correr(cc.Intento(MENSAJES, ruta="buscar", max_tokens=30000)))
ok(texto.startswith("Muy largo") and "Respuesta truncada" in texto, "tope de salida: el aviso de truncada")

print("\n7 · Detener mientras piensa: Claude deja de gastar")


async def _detener():
    cli = _Cliente([_pensando(), 5.0, _texto("nunca")])
    _con(cli)
    it = cc.Intento(MENSAJES, ruta="buscar", max_tokens=30000)
    gen = it.esperar(latido=0.05)
    await gen.__anext__()          # un latido
    await gen.aclose()             # la consulta se cancela
    await asyncio.sleep(0.05)
    return cli.registro


ok(asyncio.run(_detener()) == ["cancelado"], "la petición a Anthropic se cancela")

print("\n8 · el razonamiento se ve desde el primer momento (sin traducir)")
os.environ["CLAUDE_TRADUCIR_RAZONAMIENTO"] = "0"
cli = _Cliente([_inicio_razon(), _razon("Analizo el art. 165 "), _razon("de la LGTOC."), _inicio_razon(),
                _razon("Plazo: tres años."), 0.2, _texto("## Respuesta\n\n"), _texto("Tres años.")])
_con(cli)
razon = []
it = cc.Intento(MENSAJES, ruta="buscar", max_tokens=30000)
lat, texto, err = asyncio.run(_correr(it, latido=0.05, razon=razon))
ok(lat == 0, "sin latidos: el razonamiento llega antes que el primer latido")
ok("".join(razon) == "Analizo el art. 165 de la LGTOC.\n\nPlazo: tres años.",
   "el resumen del razonamiento va en reasoning_content, con los bloques separados")
ok(texto == "## Respuesta\n\nTres años." and err is None, "la respuesta, entera y aparte")

_con(_Cliente([_inicio_razon(), _razon("Pienso…")], types.SimpleNamespace(
    stop_reason="refusal", stop_details=types.SimpleNamespace(category="general_harms"),
    usage=types.SimpleNamespace(input_tokens=1000, output_tokens=10))))
razon = []
it = cc.Intento(MENSAJES, ruta="buscar", max_tokens=30000)
lat, texto, err = asyncio.run(_correr(it, respaldo=_luna("Respuesta de luna."), razon=razon))
ok("".join(razon) == "Pienso…" and texto == "Respuesta de luna." and err is None,
   "razonó y se negó sin escribir → la respuesta la escribe luna")

_con(_Cliente([_inicio_razon(), _razon("Pienso…"), ErrorApi(500, "api_error", "se cayó")]))
lat, texto, err = asyncio.run(_correr(cc.Intento(MENSAJES, ruta="buscar", max_tokens=30000),
                                      respaldo=_luna("Respuesta de luna.")))
ok(texto == "Respuesta de luna." and err is None, "razonó y se cayó sin escribir → luna")
_con(_Cliente([_inicio_razon(), _razon("Pienso…"), ErrorApi(500, "api_error", "se cayó")]))
lat, texto, err = asyncio.run(_correr(cc.Intento(MENSAJES, ruta="buscar", max_tokens=30000)))
ok(texto == "" and isinstance(err, ErrorApi), "sin respaldo, el error sube al chat")
_con(_Cliente([_inicio_razon(), _razon("Pienso…"), _texto("Parte "), ErrorApi(500, "api_error", "se cayó")]))
lat, texto, err = asyncio.run(_correr(cc.Intento(MENSAJES, ruta="buscar", max_tokens=30000),
                                      respaldo=_luna("NO")))
ok(texto == "Parte " and isinstance(err, ErrorApi), "con texto ya escrito no se mezcla luna: el error de siempre")

os.environ["CLAUDE_RAZONAMIENTO_VISIBLE"] = "0"
cli = _Cliente([_razon("oculto"), _texto("Visible")])
_con(cli)
razon = []
lat, texto, err = asyncio.run(_correr(cc.Intento(MENSAJES, ruta="buscar", max_tokens=30000), razon=razon))
ok("thinking" not in cli.llamadas[0] and "".join(razon) == "" and texto == "Visible",
   "CLAUDE_RAZONAMIENTO_VISIBLE=0: piensa igual pero no se pide ni se manda el resumen")
del os.environ["CLAUDE_RAZONAMIENTO_VISIBLE"]

cli = _Cliente([_inicio_razon(), _razon("oculto"), 0.2, _texto("Visible")])
_con(cli)
razon = []
lat, texto, err = asyncio.run(_correr(cc.Intento(MENSAJES, ruta="buscar", max_tokens=30000,
                                                 ver_razonamiento=False), razon=razon))
ok("thinking" not in cli.llamadas[0] and "".join(razon) == "" and texto == "Visible" and lat >= 2,
   "la app móvil (sin razonamiento en vivo): no se pide el resumen y se late hasta la primera letra")

del os.environ["CLAUDE_TRADUCIR_RAZONAMIENTO"]


async def _en_orden(intento):
    """Qué llega al bucle del chat y en qué orden: ('r', razón) o ('t', texto)."""
    async for _ in intento.esperar(latido=0.05):
        pass
    orden = []
    async for trozo in intento.flujo():
        d = trozo.choices[0].delta
        if d.reasoning_content:
            orden.append(("r", d.reasoning_content))
        if d.content:
            orden.append(("t", d.content))
    return orden


print("\n9 · el razonamiento llega traducido al español, párrafo por párrafo")
cli = _Cliente([_inicio_razon(), _razon("I'm checking "), _razon("art. 165.\n\nThree"), _razon(" years apply."),
                _fin_bloque(), 0.2, _texto("Tres "), _texto("años.")])
_con(cli)
orden = asyncio.run(_en_orden(cc.Intento(MENSAJES, ruta="buscar", max_tokens=30000)))
ok(orden == [("r", "ES[I'm checking art. 165.]"), ("r", "\n\nES[Three years apply.]"), ("t", "Tres "), ("t", "años.")],
   "cada párrafo, entero y traducido, antes de la respuesta")
tr = cli.traducciones[0]
ok(tr["model"] == "claude-haiku-5-5" and tr["thinking"] == {"type": "disabled"} and "español de México" in tr["system"],
   "traduce el mismo Haiku, sin razonar")

cli = _Cliente([_inicio_razon(), _razon("First idea."), _fin_bloque(), _texto("Respuesta")], demora_traduccion=0.3)
_con(cli)
orden = asyncio.run(_en_orden(cc.Intento(MENSAJES, ruta="buscar", max_tokens=30000)))
ok(orden == [("t", "Respuesta")], "si la respuesta empieza antes que la traducción, el párrafo se descarta")

cli = _Cliente([_inicio_razon(), _razon("Idea.\n\n"), 0.1, _texto("Respuesta")], falla_traduccion=True)
_con(cli)
orden = asyncio.run(_en_orden(cc.Intento(MENSAJES, ruta="buscar", max_tokens=30000)))
ok(orden == [("t", "Respuesta")], "si la traducción falla, no se pinta en inglés y la respuesta sigue")

_con(_Cliente([_inicio_razon(), _razon("Thinking…"), _fin_bloque(), 0.1], types.SimpleNamespace(
    stop_reason="refusal", stop_details=types.SimpleNamespace(category="general_harms"),
    usage=types.SimpleNamespace(input_tokens=1000, output_tokens=10))))
razon = []
lat, texto, err = asyncio.run(_correr(cc.Intento(MENSAJES, ruta="buscar", max_tokens=30000),
                                      respaldo=_luna("Respuesta de luna."), razon=razon))
ok("".join(razon) == "ES[Thinking…]" and texto == "Respuesta de luna.", "razonó, se negó → luna, con el razonamiento traducido")

print("\n10 · el crédito se renueva el 17 a las 15:00 (México); sin saldo, luna hasta entonces")
import time  # noqa: E402
from datetime import datetime, timezone  # noqa: E402


def _u(*a):
    return datetime(*a, tzinfo=timezone.utc)


ok(cc.inicio_del_ciclo(_u(2026, 11, 18, 16, 0)).isoformat() == "2026-11-17T15:00:00-06:00",
   "18-nov: el ciclo empezó el 17-nov a las 15:00")
ok(cc.proxima_renovacion(_u(2026, 11, 18, 16, 0)).isoformat() == "2026-12-17T15:00:00-06:00",
   "y se renueva el 17-dic a las 15:00")
ok(cc.inicio_del_ciclo(_u(2026, 12, 17, 20, 59)).date().isoformat() == "2026-11-17",
   "17-dic a las 14:59 (México): todavía el ciclo de noviembre")
ok(cc.inicio_del_ciclo(_u(2026, 12, 17, 21, 0)).date().isoformat() == "2026-12-17",
   "17-dic a las 15:00 (México): ciclo nuevo")
ok(cc.proxima_renovacion(_u(2026, 12, 20)).isoformat() == "2027-01-17T15:00:00-06:00"
   and cc.inicio_del_ciclo(_u(2027, 1, 5)).date().isoformat() == "2026-12-17", "de diciembre a enero")


async def _y_anotar(coro):
    r = await coro
    pendientes = [t for t in list(cc._tareas) if not t.done()]
    if pendientes:
        await asyncio.wait(pendientes, timeout=2)
    return r


ciclo = cc.inicio_del_ciclo().date().isoformat()
FILAS.clear()
cc._agotado_anotado = ""
_con(_Cliente(ErrorApi(402, "billing_error", "Your credit balance is too low")))
lat, texto, err = asyncio.run(_y_anotar(_correr(cc.Intento(MENSAJES, ruta="buscar", max_tokens=30000))))
ok(texto is None and abs(cc._pausa_hasta - (time.time() + 24 * 3600)) < 60,
   "sin saldo: la consulta va a luna y Claude sólo se vuelve a probar en 24 h")
ok(FILAS == [("claude_uso", {"ciclo": ciclo, "tipo": "agotado"})], "se anota cuándo se agotó el ciclo")
_con(_Cliente(ErrorApi(402, "billing_error", "Your credit balance is too low")))
asyncio.run(_y_anotar(_correr(cc.Intento(MENSAJES, ruta="buscar", max_tokens=30000))))
ok(len(FILAS) == 1, "la prueba del día siguiente que falla no vuelve a anotar el agotado")
os.environ["CLAUDE_SONDEO_SIN_SALDO_H"] = "0"
_con(_Cliente(ErrorApi(402, "billing_error", "Your credit balance is too low")))
asyncio.run(_y_anotar(_correr(cc.Intento(MENSAJES, ruta="buscar", max_tokens=30000))))
ok(abs(cc._pausa_hasta - cc.proxima_renovacion().timestamp()) < 1,
   "CLAUDE_SONDEO_SIN_SALDO_H=0: luna hasta el 17 a las 15:00, sin pruebas")
del os.environ["CLAUDE_SONDEO_SIN_SALDO_H"]

FILAS.clear()
_con(_Cliente([_texto("Respuesta")]))
asyncio.run(_y_anotar(_correr(cc.Intento(MENSAJES, ruta="pro", max_tokens=30000, peticion="abc12345"))))
f = FILAS[0][1] if FILAS else {}
ok(FILAS and FILAS[0][0] == "claude_uso" and f["tipo"] == "respuesta" and f["ciclo"] == ciclo
   and f["ruta"] == "pro" and f["usd"] > 0 and f["entrada"] == 49_000 and f["salida"] == 5_000
   and f["peticion"] == "abc12345", "cada respuesta se anota con su ciclo, ruta, tokens y costo")

cc._cliente = None
print("   ·", end="")
for crudo, util in [("Estoy revisando el art. 165.", True), ("(nada)", False),
                    ("No voy a traducir ese párrafo: no reproduzco mi razonamiento interno.", False),
                    ("No hay párrafo de razonamiento que traducir aquí.", False),
                    ("Estoy revisando el art. 165. Mientras tanto, te comento que el artículo 362 del Código de "
                     "Comercio regula los intereses moratorios y el 1084 establece los requisitos de la demanda "
                     "ejecutiva, así que conviene revisar ambos en el texto vigente antes de citarlos.", False)]:
    ok((cc._traduccion_util("I'm checking art. 165.", crudo) is not None) == util,
       f"traducción {'útil' if util else 'descartada'}: {crudo[:50]!r}")
cli = _Cliente([_inicio_razon(), _razon("Idea one.\n\n"), 0.1, _texto("Respuesta")])
cli.respuesta = "No voy a traducir ese párrafo: no reproduzco mi razonamiento."
_con(cli)
orden = asyncio.run(_en_orden(cc.Intento(MENSAJES, ruta="buscar", max_tokens=30000)))
ok(orden == [("t", "Respuesta")] and "<fragmento>" in cli.traducciones[0]["messages"][0]["content"],
   "el párrafo va entre delimitadores, y una negativa del traductor no se pinta")

print()
if FALLOS:
    print("FALLOS:\n  · " + "\n  · ".join(FALLOS))
    sys.exit(1)
print("El chat intenta Claude y, sin él, sigue con luna sin perder la consulta.")
