# -*- coding: utf-8 -*-
"""La compuerta de las preguntas en la propuesta y /taller/responder (2-oct-2026,
bandera «preguntas_al_secretario»), con las funciones REALES de main.py sobre
una fila de mentira: sin red, sin Supabase, sin modelo.

    .venv/bin/python test_compuerta_preguntas.py
"""
import ast, asyncio, hashlib, json, subprocess, sys, time, types, uuid
sys.path.insert(0, ".")
from fastapi import HTTPException
import contexto_taller as ct
import taller_estado as te
import preguntas_secretario as ps

FALLOS = []


def ok(c, q):
    print(f"   {'PASA ' if c else 'FALLA'}  {q}")
    if not c:
        FALLOS.append(q)


SRC = open("main.py", encoding="utf-8").read()
FN = {n.name: n for n in ast.walk(ast.parse(SRC)) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
te.huella_contraste = lambda r: "H1"


class _App:
    def post(self, *a, **k):
        return lambda f: f

    get = post


def entorno(marcas=None, nucleo=None, lanzadas=None):
    marcas = {} if marcas is None else marcas
    escritos = []

    def leer(email, numero, clave):
        v = marcas.get(clave)
        return json.loads(json.dumps(v)) if v is not None else None

    def leer_varias(email, numero, claves):
        return {k: leer(email, numero, k) for k in claves if k in marcas}

    def guardar(email, numero, clave, doc, huella=""):
        escritos.append((clave, json.loads(json.dumps(doc))))
        marcas[clave] = json.loads(json.dumps(doc))
        return True

    async def preproponer_falsa(email, numero, r, material, forzar=False, contexto=""):
        (lanzadas if lanzadas is not None else []).append({"forzar": forzar, "material": material,
                                                           "contexto": contexto})

    ns = {"asyncio": asyncio, "time": time, "uuid": uuid, "json": json, "hashlib": hashlib, "_te": te,
          "HTTPException": HTTPException, "Form": lambda *a, **k: None, "app": _App(),
          "_taller_puerta": lambda *a, **k: None,
          "_taller_leer_marca": leer, "_taller_leer_marcas": leer_varias, "_taller_guardar_marca": guardar,
          "_taller_proponer_nucleo": nucleo, "_taller_firma_origen": lambda: "",
          "_taller_sesion_en_memoria": lambda *a, **k: None, "_taller_plan_adelantar": lambda e: False,
          "_taller_lanzar_deliberacion": lambda *a, **k: (lanzadas if lanzadas is not None else []).append("delib"),
          "_TALLER_EN_MARCHA": set(), "err": lambda e: str(e)}
    for f in ("_taller_compuerta_preguntas", "_taller_analisis_guardado", "_taller_respuestas_guardadas",
              "_taller_bloque_respuestas", "_con_autos", "taller_responder", "_taller_con_latido",
              "_taller_propuesta_unica", "_taller_propuesta_reclamada", "_taller_marca_es_mia",
              "_taller_preproponer", "_taller_hash_contexto"):
        exec(compile(ast.Module(body=[FN[f]], type_ignores=[]), "main.py", "exec"), ns)
    # /taller/responder relanza la propuesta: aquí, la de mentira.
    ns["_taller_preproponer_real"] = ns["_taller_preproponer"]
    ns["_taller_preproponer"] = preproponer_falsa
    return ns, marcas, escritos


def prem(i, problema=1, si_si="fundado", si_no="infundado"):
    return {"id": f"F{i}", "segmento": f"C1.{chr(96 + i)}", "problema": problema,
            "afirma_la_parte": f"hecho {i}", "el_acto": "no_se_pronuncia", "carga_de": "la_parte",
            "pregunta": f"¿Ocurrió el hecho {i}?", "tipo": "si_no", "si_si": si_si, "si_no": si_no,
            "para_que": "x"}


ANALISIS = {"razones": [], "hechos": [], "argumentos": [], "premisas": [prem(1), prem(2, problema=2)]}
PROBS = [{"pregunta": "¿Principal?", "jerarquia": "principal"}, {"pregunta": "¿Accesorio?"}]


def resultado(respuestas=None):
    r = types.SimpleNamespace(
        fases=types.SimpleNamespace(fuentes=["ACTO", "ESCRITO"], autos="AUTOS DEL JUICIO", problemas=PROBS),
        encargo=types.SimpleNamespace(es_recurso=False))
    if respuestas is not None:
        r.respuestas_secretario = respuestas
    return r


def con_bandera(si=True):
    ct.poner(True, {"banderas": {"preguntas_al_secretario": si}}, pruebas=True)


print("\n1 · LA COMPUERTA DEL NÚCLEO")
con_bandera(False)
ns, _, _ = entorno()
d, q, a = ns["_taller_compuerta_preguntas"]("1/2026", resultado(), PROBS, ANALISIS, "m")
ok(d is None and q is None and a is ANALISIS, "sin la bandera no hace nada: ni preguntas ni análisis tocado")
con_bandera(True)
d, q, a = ns["_taller_compuerta_preguntas"]("1/2026", resultado(), PROBS, ANALISIS, "m")
ok(isinstance(d, dict) and d["estado"] == "preguntas" and d["propuestas"] == [] and d["global"] is None
   and d["pendientes"] == 1 and d["modelo"] == "m" and d["formato"] == 3,
   "con una indispensable sin contestar: la respuesta «preguntas» (contrato A), sin modelo")
_id = next(x["id"] for x in q if x["indispensable"])
d2, q2, a2 = ns["_taller_compuerta_preguntas"]("1/2026", resultado([{"id": _id, "respuesta": "no"}]),
                                               PROBS, ANALISIS, "m")
ok(d2 is None and len(q2) == 2 and next(p for p in a2["premisas"] if p["id"] == "F1")["respuesta"] == "no",
   "contestada: se propone, y la respuesta va junto a su premisa en el análisis")
d3, q3, a3 = ns["_taller_compuerta_preguntas"]("1/2026", resultado(), PROBS, None, "m")
ok(d3 is None and q3 == [] and a3 is None, "sin análisis (falló o tardó) se propone como siempre")
d4, q4, a4 = ns["_taller_compuerta_preguntas"]("1/2026", None, PROBS, {"premisas": 7}, "m")
ok(d4 is None, "y un análisis mal formado nunca detiene la propuesta")

print("\n2 · EL CABLEADO DEL NÚCLEO")
nuc = ast.get_source_segment(SRC, FN["_taller_proponer_nucleo"])
ok(nuc.index("_taller_esperar_analisis(") < nuc.index("_taller_compuerta_preguntas(")
   < nuc.index("return _detenida") < nuc.index("_f5.proponer("),
   "la compuerta va después de esperar el análisis y ANTES de llamar al modelo de la propuesta")
ok(nuc.index("_taller_respuestas_guardadas(user_email, numero, r)") < nuc.index("contexto = _con_autos(r, contexto)"),
   "las respuestas se leen de la fila antes de armar el contexto")
ok('"preguntas": _preguntas_p' in nuc and "if _preguntas_p is not None else {}" in nuc,
   "la propuesta lista lleva sus preguntas (sólo con la bandera)")
pr = ast.get_source_segment(SRC, FN["taller_proponer"])
ok("respuestas_firma" in pr and "es de antes de las respuestas" in pr,
   "/taller/proponer no sirve la guardada calculada con otras respuestas")
rec = ast.get_source_segment(SRC, FN["_taller_recuperar_sesion"])
ok("_taller_respuestas_guardadas(email, numero" in rec, "cada petición lee las respuestas de la fila (-w 2)")
pa = ast.get_source_segment(SRC, FN["_taller_preanalizar"])
ok('rediseno("preguntas_al_secretario")' in pa, "el análisis corre también con la bandera de las preguntas")

print("\n3 · LAS RESPUESTAS DELANTE DE LAS CONSTANCIAS, SIN RECORTE")
con_bandera(False)
ns, _, _ = entorno()
try:
    _base = subprocess.run(["git", "show", "b89e049:main.py"], capture_output=True, text=True, check=True).stdout
    _fb = {n.name: n for n in ast.walk(ast.parse(_base)) if isinstance(n, ast.FunctionDef) and n.name == "_con_autos"}
    _nsb = {}
    exec(compile(ast.Module(body=[_fb["_con_autos"]], type_ignores=[]), "base", "exec"), _nsb)
except Exception as _e:
    _nsb = None
    print(f"   (sin la base: {_e})")
_r = resultado([{"id": "P1", "pregunta": "¿Hubo emplazamiento?", "respuesta": "si"}])
if _nsb is not None:
    ok(ns["_con_autos"](_r, "lo del secretario") == _nsb["_con_autos"](_r, "lo del secretario")
       and ns["_con_autos"](resultado(), "") == _nsb["_con_autos"](resultado(), ""),
       "sin la bandera, `_con_autos` idéntico a la base aunque haya respuestas guardadas")
con_bandera(True)
_c = ns["_con_autos"](_r, "lo del secretario")
ok(_c.startswith(ps.CABECERA_BLOQUE) and _c.index(ps.FIN_BLOQUE) < _c.index("CONSTANCIAS QUE OBRAN EN AUTOS")
   and "Respuesta: SÍ" in _c, "con la bandera, el bloque de respuestas va DELANTE de los autos")
_r2 = resultado([{"id": "P1", "pregunta": "¿Hubo emplazamiento?", "respuesta": "si"}])
_r2.fases.autos = ""
ok(ns["_con_autos"](_r2, "x").startswith(ps.CABECERA_BLOQUE), "y también sin autos")
import fase6_estudio as f6
_ap = f6._bloque_aportado(("A" * 30000) + "\n" + ps.bloque_respuestas(_r.respuestas_secretario))
ok(ps.CABECERA_BLOQUE in _ap and _ap.lstrip().startswith(ps.CABECERA_BLOQUE),
   "el estudio las recibe enteras y primero aunque lo aportado pase del tope")

print("\n4 · /taller/responder (contrato C)")
con_bandera(True)
_hu_an = __import__("analisis_litis").huella(resultado())


def fila_base():
    return {"analisis": {"huella": "H1", "huella_analisis": _hu_an, "estado": "listo", "doc": ANALISIS},
            "propuesta": {"huella": "H1", "estado": "listo",
                          "respuesta": ps.respuesta_preguntas("1/2026", ps.preguntas_de(ANALISIS, PROBS))}}


async def responder(ns, cuerpo, ses):
    ns["_taller_recuperar_sesion"] = lambda e, n: ses
    return await ns["taller_responder"](numero="1/2026", user_email="a@b.c", respuestas_json=json.dumps(cuerpo))


_q = ps.preguntas_de(ANALISIS, PROBS)
_ind = next(x["id"] for x in _q if x["indispensable"])
_otra = next(x["id"] for x in _q if not x["indispensable"])

lanz = []
ns, marcas, escritos = entorno(fila_base(), lanzadas=lanz)
_ses = {"resultado": resultado(), "material": object()}
out = asyncio.run(responder(ns, [{"id": _otra, "respuesta": "sí"}], _ses))
ok(out == {"ok": True, "pendientes": 1, "propuesta": "preguntas"} and not lanz,
   "contestar una no indispensable: sigue en «preguntas» y no se relanza nada")
ok(marcas["respuestas"]["huella"] == "H1" and marcas["respuestas"]["items"][0]["respuesta"] == "si",
   "la respuesta se guarda en la fila con la huella del adelanto")
ok(next(p for p in marcas["propuesta"]["respuesta"]["preguntas"] if p["id"] == _otra)["respuesta"] == "si",
   "y la respuesta «preguntas» guardada se rehace, por código, con lo ya contestado")
out = asyncio.run(responder(ns, [{"id": _ind, "respuesta": "No consta"}], _ses))
ok(out == {"ok": True, "pendientes": 0, "propuesta": "en_curso"},
   "contestada la indispensable: la propuesta se relanza")
ok(marcas["propuesta"]["estado"] == "en_curso" and marcas["propuesta"]["respuestas_firma"] == marcas["respuestas"]["firma"],
   "la guardada se invalida EN LA MISMA PETICIÓN (un /taller/proponer inmediato no sirve la de antes)")
ok(lanz and lanz[-1]["forzar"] is True, "y se relanza forzada (la del botón, si la había, es de antes)")
ok(len(marcas["respuestas"]["items"]) == 2, "las dos respuestas, acumuladas")
_n_lanz = len(lanz)
_e = None
try:
    asyncio.run(responder(ns, "no es lista", _ses))
except HTTPException as ex:
    _e = ex.status_code
ok(_e == 422, "un cuerpo que no es lista: 422")
ns2, marcas2, _ = entorno({"propuesta": {"huella": "H1", "estado": "listo"}}, lanzadas=lanz)
_e = None
try:
    asyncio.run(responder(ns2, [{"id": _ind, "respuesta": "si"}], _ses))
except HTTPException as ex:
    _e = ex.status_code
ok(_e == 409 and "respuestas" not in marcas2, "sin análisis de este adelanto: 409 y no se guarda nada")
con_bandera(False)
_e = None
try:
    asyncio.run(responder(ns, [{"id": _ind, "respuesta": "si"}], _ses))
except HTTPException as ex:
    _e = ex.status_code
ok(_e == 409 and len(lanz) == _n_lanz, "sin la bandera: 409")
con_bandera(True)
ns3, marcas3, _ = entorno(fila_base(), lanzadas=lanz)
out = asyncio.run(responder(ns3, [{"id": _ind, "respuesta": "si"}], {"resultado": resultado(), "material": None}))
ok(out["propuesta"] == "en_curso" and marcas3["propuesta"]["estado"] == "fallo" and len(lanz) == _n_lanz,
   "sin acervo en la sesión: no se relanza aquí y la guardada de antes se tira (la encadenada a la consulta la hará)")

print("\n5 · LA PROPUESTA DE FONDO CON PREGUNTAS")


async def _fondo(nucleo, marcas_ini, forzar=False, cambiar_respuestas=False):
    lz = []
    ns, marcas, escritos = entorno(marcas_ini, nucleo=nucleo, lanzadas=lz)

    async def nuc(email, numero, ses, contexto):
        if cambiar_respuestas:
            marcas["respuestas"] = {"huella": "H1", "items": [{"id": "P9", "respuesta": "si"}]}
        return await nucleo(email, numero, ses, contexto)
    ns["_taller_proponer_nucleo"] = nuc
    await ns["_taller_preproponer_real"]("a@b.c", "1/2026", resultado(), object(), forzar=forzar)
    return marcas, lz


async def nucleo_preguntas(*a):
    return ps.respuesta_preguntas("1/2026", ps.preguntas_de(ANALISIS, PROBS))


async def nucleo_lista(*a):
    return {"global": {"sentido": "infundado"}, "propuestas": []}


con_bandera(True)
m, lz = asyncio.run(_fondo(nucleo_preguntas, {}))
ok(m["propuesta"]["estado"] == "listo" and m["propuesta"]["respuesta"]["estado"] == "preguntas" and not lz,
   "detenida por preguntas: la marca queda «listo» con la respuesta «preguntas», sin plan ni deliberación")
m, lz = asyncio.run(_fondo(nucleo_lista, {}, cambiar_respuestas=True))
ok(m.get("propuesta", {}).get("estado") == "en_curso" and not lz,
   "si el secretario contesta mientras corre, la que sale ya no es la suya: no se guarda")
_recl = {"propuesta": {"huella": "H1", "estado": "listo", "origen": "boton", "ficha": "boton-x"}}
m, lz = asyncio.run(_fondo(nucleo_lista, dict(_recl)))
ok(m["propuesta"]["ficha"] == "boton-x", "sin forzar, la reclamada por el botón no se toca (como siempre)")
m, lz = asyncio.run(_fondo(nucleo_lista, dict(_recl), forzar=True))
ok(m["propuesta"]["estado"] == "listo" and m["propuesta"]["ficha"] != "boton-x"
   and "respuestas_firma" in m["propuesta"], "forzada (tras responder), se calcula y lleva la firma de las respuestas")
con_bandera(False)
m, lz = asyncio.run(_fondo(nucleo_lista, {}))
ok("respuestas_firma" not in m["propuesta"] and "respuestas_firma" not in m["propuesta"]["respuesta"],
   "sin la bandera, la marca de siempre")

print("\n6 · EL AVANCE")


class _Q:
    def __init__(self, data):
        self.data = data

    def __getattr__(self, k):
        return lambda *a, **kw: self

    def execute(self):
        return types.SimpleNamespace(data=self.data)


_nsa = {"_te": te, "err": str}
exec(compile(ast.Module(body=[FN["_taller_avance"]], type_ignores=[]), "main.py", "exec"), _nsa)
con_bandera(True)
_nsa["supabase_admin"] = types.SimpleNamespace(table=lambda *a: _Q([{
    "propuesta": {"huella": "H1", "estado": "listo", "respuesta": {"estado": "preguntas", "pendientes": 2}},
    "consulta": {"estado": "listo", "segundos": 3}}]))
_av = _nsa["_taller_avance"]("a@b.c", "1/2026")
ok(_av["propuesta"] == {"estado": "preguntas", "segundos": None, "pendientes": 2}
   and _av["consulta"]["estado"] == "listo", "avance.propuesta vale «preguntas», con cuántas faltan")
_nsa["supabase_admin"] = types.SimpleNamespace(table=lambda *a: _Q([{
    "propuesta": {"huella": "H1", "estado": "listo", "segundos": 9, "respuesta": {"global": {}}}}]))
ok(_nsa["_taller_avance"]("a@b.c", "1/2026")["propuesta"] == {"estado": "listo", "segundos": 9},
   "una propuesta lista, como siempre")
# ROLLBACK (3-oct-2026): sin la bandera, la marca «preguntas» no es «preguntas».
con_bandera(False)
_nsa["supabase_admin"] = types.SimpleNamespace(table=lambda *a: _Q([{
    "propuesta": {"huella": "H1", "estado": "listo", "respuesta": {"estado": "preguntas", "pendientes": 2}}}]))
ok(_nsa["_taller_avance"]("a@b.c", "1/2026")["propuesta"] == {"estado": "fallo", "segundos": None},
   "sin la bandera, una guardada «preguntas» se informa «fallo»: la pantalla ofrece el botón y se recalcula")

ct.poner(False)
print()
if FALLOS:
    print(f"FALLAN {len(FALLOS)}: " + " · ".join(FALLOS))
    sys.exit(1)
print("RESULTADO: TODAS LAS COMPROBACIONES PASAN")
