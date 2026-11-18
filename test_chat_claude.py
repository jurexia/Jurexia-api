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
    def __init__(self, guion, final=None):
        self.llamadas, self.registro = [], []
        self.guion = guion
        self.final = final or types.SimpleNamespace(
            stop_reason="end_turn", stop_details=None,
            usage=types.SimpleNamespace(input_tokens=40_000, cache_read_input_tokens=9_000,
                                        cache_creation_input_tokens=0, output_tokens=5_000))
        self.messages = types.SimpleNamespace(stream=self._stream)

    def _stream(self, **kw):
        self.llamadas.append(kw)
        return _Flujo(self.guion, self.final, self.registro)


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


async def _correr(intento, latido=0.05):
    latidos = 0
    async for _ in intento.esperar(latido=latido):
        latidos += 1
    flujo = intento.flujo()
    if flujo is None:
        return latidos, None, None
    texto, error = "", None
    try:
        async for trozo in flujo:          # como el bucle de main.py
            if trozo.choices and trozo.choices[0].delta:
                texto += getattr(trozo.choices[0].delta, "content", None) or ""
    except Exception as e:
        error = e
    return latidos, texto, error


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


class _Perfiles:
    def table(self, _):
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
ok("thinking" not in kw, "con razonamiento por omisión (adaptativo, el del modelo)")
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

print()
if FALLOS:
    print("FALLOS:\n  · " + "\n  · ".join(FALLOS))
    sys.exit(1)
print("El chat intenta Claude y, sin él, sigue con luna sin perder la consulta.")
