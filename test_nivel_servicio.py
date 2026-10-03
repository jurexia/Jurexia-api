# -*- coding: utf-8 -*-
"""VELOCIDAD Y AGUANTE (3-oct-2026), sin red.

    .venv/bin/python test_nivel_servicio.py

El carril rápido de OpenAI sólo en el taller y sólo si se configura; Gemini con
turno y reintentos ante 429/503 dentro de su tope; y un modelo que no admite el
carril no tumba la llamada.
"""
import asyncio
import os
import sys
import time
import types

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import contexto_taller as ct
import nivel_servicio as ns
import llamada_modelo as lm

FALLOS = []


def ok(c, q):
    print(f"   {'PASA ' if c else 'FALLA'}  {q}")
    if not c:
        FALLOS.append(q)


os.environ.pop("TALLER_NIVEL_SERVICIO", None)

print("\n1 · EL CARRIL, SÓLO EN EL TALLER Y SÓLO SI SE PIDE")
ct._CTX.set(None)
os.environ["TALLER_NIVEL_SERVICIO"] = "priority"
ok(ns.aplicar_openai({"model": "x"}) == {"model": "x"}, "fuera del taller (sin contexto) no se pide nada, aunque el entorno lo diga")
ct.poner(False)
ok(ns.aplicar_openai({"model": "x"}).get("service_tier") == "priority", "en el taller, el del entorno")
ok(ns.aplicar_openai({"model": "x", "service_tier": "default"})["service_tier"] == "default", "lo que fija quien llama manda")
os.environ["TALLER_NIVEL_SERVICIO"] = "fast"
ok(ns.nivel_openai() == "priority", "«fast» se pide como «priority» (el nombre que admite el SDK)")
_c = dict(ct._CTX.get()); _c["nivel_servicio"] = "flex"; ct._CTX.set(_c)
ok(ns.nivel_openai() == "flex", "una petición puede pedir «flex» (la corrida nocturna)")
ct.poner(False)
os.environ.pop("TALLER_NIVEL_SERVICIO")
ok(ns.nivel_openai() == "", "sin entorno, el carril normal")
os.environ["TALLER_NIVEL_SERVICIO"] = "turbo"
ok(ns.nivel_openai() == "", "un valor desconocido no se manda")
os.environ.pop("TALLER_NIVEL_SERVICIO")

print("\n2 · UN MODELO QUE NO ADMITE EL CARRIL NO TUMBA LA LLAMADA")


class _Comp:
    def __init__(self):
        self.vistos = []

    async def create(self, **kw):
        self.vistos.append(dict(kw))
        if "service_tier" in kw:
            raise RuntimeError("Invalid value for 'service_tier': priority is not available for this model")
        return types.SimpleNamespace(usage=None, choices=[])


_cli = types.SimpleNamespace(chat=types.SimpleNamespace(completions=_Comp()))
os.environ["TALLER_NIVEL_SERVICIO"] = "priority"
ct.poner(False)
asyncio.run(lm._crear_una_sin_anotar(_cli, model="gpt-5.6-luna", messages=[]))
_v = _cli.chat.completions.vistos
ok(len(_v) == 2 and _v[0].get("service_tier") == "priority" and "service_tier" not in _v[1],
   "se pidió el carril, el modelo lo rechazó y se llamó sin él")
os.environ.pop("TALLER_NIVEL_SERVICIO")

print("\n3 · GEMINI: REINTENTA LO PASAJERO, NO LO DEMÁS")
ok(ns.es_pasajero(RuntimeError("503 UNAVAILABLE. This model is currently experiencing high demand")), "503 es pasajero")
ok(ns.es_pasajero(RuntimeError("429 RESOURCE_EXHAUSTED")), "429 es pasajero")
ok(not ns.es_pasajero(ValueError("400 INVALID_ARGUMENT: bad request")), "400 no lo es")
ns.random.random = lambda: 0.0  # esperas exactas en la prueba
_esperas = []
_dormir = asyncio.sleep


async def _sleep(s):
    _esperas.append(round(s, 1))
    await _dormir(0)

ns.asyncio.sleep = _sleep


def _fabrica_que_falla(n_fallos, exc):
    estado = {"n": 0}

    async def f():
        estado["n"] += 1
        if estado["n"] <= n_fallos:
            raise exc
        return "bien"
    return f, estado

f, st = _fabrica_que_falla(2, RuntimeError("503 UNAVAILABLE"))
r = asyncio.run(ns.llamar_gemini(f, 120, "prueba"))
ok(r == "bien" and st["n"] == 3 and _esperas == [1.6, 4.0], f"dos 503 y a la tercera; esperas {_esperas}")
_esperas.clear()
f, st = _fabrica_que_falla(5, ValueError("400 INVALID_ARGUMENT"))
try:
    asyncio.run(ns.llamar_gemini(f, 120, "prueba"))
    ok(False, "un 400 debía subir")
except ValueError:
    ok(st["n"] == 1 and not _esperas, "un 400 sube a la primera, sin reintentar")
f, st = _fabrica_que_falla(9, RuntimeError("503 UNAVAILABLE"))
try:
    asyncio.run(ns.llamar_gemini(f, 16, "prueba"))
    ok(False, "con tope corto debía rendirse")
except (RuntimeError, asyncio.TimeoutError):
    ok(st["n"] == 1, f"si tras la espera no quedarían 15 s, no reintenta ({st['n']} llamada)")
ns.asyncio.sleep = _dormir

print("\n4 · EL TURNO: NO MÁS DE N A LA VEZ POR PROCESO")
os.environ["GEMINI_TALLER_CONCURRENCIA"] = "3"
ns._SEM = None


async def _carga():
    vivos, pico = {"n": 0}, {"n": 0}

    async def f():
        vivos["n"] += 1
        pico["n"] = max(pico["n"], vivos["n"])
        await _dormir(0.05)
        vivos["n"] -= 1
        return 1
    rs = await asyncio.gather(*(ns.llamar_gemini(f, 60, "carga") for _ in range(20)))
    return sum(rs), pico["n"]

t0 = time.time()
total, pico = asyncio.run(_carga())
ok(total == 20 and pico == 3, f"20 llamadas, como mucho {pico} a la vez")
os.environ.pop("GEMINI_TALLER_CONCURRENCIA")
ns._SEM = None

print()
if FALLOS:
    print(f"FALLAN {len(FALLOS)}: " + " · ".join(FALLOS))
    raise SystemExit(1)
print("RESULTADO: TODAS LAS COMPROBACIONES PASAN")
