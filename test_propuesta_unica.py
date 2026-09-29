# -*- coding: utf-8 -*-
"""Una sola propuesta viva por adelanto (rediseño, etapa 3, paso 1; bandera «propuesta_unica»).

Medido en el banco Kingston: el botón forzado entraba ~2 s después de que
arrancara la propuesta de fondo y cada una dejaba su mitad del estado.

    .venv/bin/python test_propuesta_unica.py
"""
import ast, asyncio, sys, time, types, uuid, json
import contexto_taller as ct
import taller_estado as te

FALLOS = []


def ok(c, q):
    print(f"   {'PASA ' if c else 'FALLA'}  {q}")
    if not c:
        FALLOS.append(q)


SRC = open("main.py", encoding="utf-8").read()
FN = {n.name: n for n in ast.walk(ast.parse(SRC)) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}


def entorno(nucleo):
    """Las funciones reales de main.py sobre una marca de mentira."""
    marcas = {}
    escritos = []

    def leer(email, numero, clave):
        v = marcas.get(clave)
        return json.loads(json.dumps(v)) if v is not None else None

    def guardar(email, numero, clave, doc, huella=""):
        escritos.append((clave, dict(doc)))
        marcas[clave] = dict(doc)
        return True

    lanzadas = []
    ns = {"asyncio": asyncio, "time": time, "uuid": uuid, "json": json, "_te": te,
          "_taller_leer_marca": leer, "_taller_guardar_marca": guardar,
          "_taller_proponer_nucleo": nucleo,
          "_taller_sesion_en_memoria": lambda *a, **k: None,
          "_taller_plan_adelantar": lambda e: False,
          "_taller_lanzar_deliberacion": lambda *a, **k: lanzadas.append(a[1]),
          "_TALLER_EN_MARCHA": set(), "err": lambda e: str(e)}
    for f in ("_taller_con_latido", "_taller_propuesta_unica", "_taller_propuesta_reclamada",
              "_taller_marca_es_mia", "_taller_preproponer"):
        exec(compile(ast.Module(body=[FN[f]], type_ignores=[]), "main.py", "exec"), ns)
    return ns, marcas, escritos, lanzadas


R = types.SimpleNamespace(fases=types.SimpleNamespace(), encargo=None)
te_huella = te.huella_contraste
te.huella_contraste = lambda r: "H1"

print("\n1 · SIN LA BANDERA, COMO SIEMPRE")


async def _sin():
    async def nucleo(email, numero, ses, contexto):
        await asyncio.sleep(0.01)
        return {"global": {"sentido": "infundado"}, "propuestas": []}
    ns, marcas, escritos, lanz = entorno(nucleo)
    ct.poner(False, {})
    marcas["propuesta"] = {"huella": "H1", "estado": "en_curso", "origen": "boton", "ficha": "boton-x",
                           "desde": time.time()}
    await ns["_taller_preproponer"]("a@b.c", "1/2026", R, object())
    return marcas, escritos, lanz


m, e, l = asyncio.run(_sin())
ok(m["propuesta"]["estado"] == "listo" and "ficha" not in m["propuesta"] and l == ["1/2026"],
   "la de fondo arranca, escribe «listo» sin ficha y lanza la deliberación, aunque haya otra en curso (paridad)")

print("\n2 · CON LA BANDERA: EL BOTÓN YA LA RECLAMÓ")


async def _reclamada():
    llamadas = []

    async def nucleo(email, numero, ses, contexto):
        llamadas.append(1)
        return {"global": {}}
    ns, marcas, escritos, lanz = entorno(nucleo)
    ct.poner(True, {}, pruebas=True)
    marcas["propuesta"] = {"huella": "H1", "estado": "en_curso", "origen": "boton", "ficha": "boton-x",
                           "desde": time.time(), "latido": time.time()}
    await ns["_taller_preproponer"]("a@b.c", "1/2026", R, object())
    return marcas, llamadas, lanz


m, llam, l = asyncio.run(_reclamada())
ok(not llam and m["propuesta"]["ficha"] == "boton-x" and not l,
   "la de fondo no arranca: ni calcula, ni pisa la marca, ni lanza deliberación")

print("\n3 · CON LA BANDERA: EL BOTÓN ENTRA MIENTRAS CORRE LA DE FONDO (la carrera del banco)")


async def _carrera():
    puerta = asyncio.Event()

    async def nucleo(email, numero, ses, contexto):
        await puerta.wait()
        return {"global": {"sentido": "infundado"}, "propuestas": []}
    ns, marcas, escritos, lanz = entorno(nucleo)
    ct.poner(True, {}, pruebas=True)
    tarea = asyncio.ensure_future(ns["_taller_preproponer"]("a@b.c", "1/2026", R, object()))
    await asyncio.sleep(0.01)
    ficha_fondo = marcas["propuesta"].get("ficha")
    # el botón reclama (lo que hace /taller/proponer con la bandera)
    marcas["propuesta"] = {"huella": "H1", "estado": "en_curso", "origen": "boton", "ficha": "boton-y",
                           "desde": time.time()}
    puerta.set()
    await tarea
    return marcas, ficha_fondo, lanz


m, ff, l = asyncio.run(_carrera())
ok(ff and ff.startswith("fondo-"), "la de fondo reclama la marca con su ficha")
ok(m["propuesta"]["ficha"] == "boton-y" and m["propuesta"]["estado"] == "en_curso" and not l,
   "al terminar ve que ya no es suya: no escribe «listo» encima de la del botón ni lanza deliberación")

print("\n4 · CON LA BANDERA: NADIE LA TOCA")


async def _sola():
    async def nucleo(email, numero, ses, contexto):
        return {"global": {"sentido": "fundado"}, "propuestas": []}
    ns, marcas, escritos, lanz = entorno(nucleo)
    ct.poner(True, {}, pruebas=True)
    await ns["_taller_preproponer"]("a@b.c", "1/2026", R, object())
    return marcas, lanz


m, l = asyncio.run(_sola())
ok(m["propuesta"]["estado"] == "listo" and m["propuesta"]["origen"] == "fondo" and l == ["1/2026"],
   "se guarda con su ficha y sigue como siempre (deliberación incluida)")


async def _vieja():
    async def nucleo(email, numero, ses, contexto):
        return {"global": {}}
    ns, marcas, escritos, lanz = entorno(nucleo)
    ct.poner(True, {}, pruebas=True)
    marcas["propuesta"] = {"huella": "H1", "estado": "listo", "origen": "fondo", "ficha": "fondo-viejo"}
    await ns["_taller_preproponer"]("a@b.c", "1/2026", R, object())
    return marcas


ok(asyncio.run(_vieja())["propuesta"]["ficha"] != "fondo-viejo",
   "una lista de OTRA corrida de fondo no la detiene: volver a subir el adelanto recalcula como siempre")

print("\n5 · EL LATIDO CONSERVA LA FICHA")


async def _latido():
    async def nucleo(email, numero, ses, contexto):
        return {}
    ns, marcas, escritos, lanz = entorno(nucleo)
    orig = asyncio.wait

    async def wait_rapido(fs, timeout=None):
        hechas, pend = await orig(fs, timeout=0.001)
        return hechas, pend
    ns["asyncio"] = types.SimpleNamespace(**{k: getattr(asyncio, k) for k in dir(asyncio) if not k.startswith("_")})
    ns["asyncio"].wait = wait_rapido

    async def lenta():
        await asyncio.sleep(0.02)
        return 1
    await ns["_taller_con_latido"]("a", "n", "propuesta", "H1", 1.0, lenta(), extra={"ficha": "f1", "origen": "boton"})
    await ns["_taller_con_latido"]("a", "n", "otra", "H1", 1.0, lenta())
    return escritos


esc = asyncio.run(_latido())
lat_p = [d for c, d in esc if c == "propuesta"]
lat_o = [d for c, d in esc if c == "otra"]
ok(lat_p and all(d.get("ficha") == "f1" for d in lat_p), "el latido de la marca reclamada lleva la ficha")
ok(lat_o and all(set(d) == {"huella", "estado", "desde", "latido"} for d in lat_o),
   "sin `extra`, el latido escribe exactamente lo de siempre")

print("\n6 · EL CABLEADO DEL BOTÓN")
_fn_prop = next(n for n in ast.walk(ast.parse(SRC)) if isinstance(n, ast.AsyncFunctionDef) and n.name == "taller_proponer")
_seg = ast.get_source_segment(SRC, _fn_prop)
ok("_unica_b = _taller_propuesta_unica()" in _seg and '"origen": "boton"' in _seg
   and "_taller_marca_es_mia(user_email, numero, _ficha_b)" in _seg,
   "el botón forzado o con contexto reclama la marca, y sólo escribe «listo» si sigue siendo suya")
ok("_resp = await _taller_proponer_nucleo(user_email, numero, ses, contexto)" in _seg,
   "sin la bandera, el botón calcula como siempre")
ok("propuesta_unica" in ct.BANDERAS_REDISENO, "la bandera está en el rediseño (apagada para los de fuera)")

te.huella_contraste = te_huella
print()
if FALLOS:
    print("FALLAS:\n  " + "\n  ".join(FALLOS))
    sys.exit(1)
print("RESULTADO: TODAS LAS COMPROBACIONES PASAN")
