# -*- coding: utf-8 -*-
"""El flujo de las respuestas del secretario tras la revisión adversarial del
3-oct-2026: contestar no tira lo aportado, corregir el problema no tira las
respuestas, y con la bandera apagada (reversión) nada se queda en «preguntas».

Con las funciones REALES de main.py (por AST) sobre una fila de mentira: sin
red, sin Supabase, sin modelo.

    .venv/bin/python test_flujo_respuestas.py
"""
import ast, asyncio, hashlib, json, sys, time, types, uuid
sys.path.insert(0, ".")
from fastapi import HTTPException
import contexto_taller as ct
import taller_estado as te
import preguntas_secretario as ps
import analisis_litis as al

FALLOS = []


def ok(c, q):
    print(f"   {'PASA ' if c else 'FALLA'}  {q}")
    if not c:
        FALLOS.append(q)


SRC = open("main.py", encoding="utf-8").read()
FN = {n.name: n for n in ast.walk(ast.parse(SRC)) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}


class _App:
    def post(self, *a, **k):
        return lambda f: f

    get = post


def con_bandera(si=True, **otras):
    ct.poner(True, {"banderas": {"preguntas_al_secretario": si, **otras}}, pruebas=True)


class Fases:
    def __init__(self, problemas):
        self.problemas = problemas
        self.fuentes = ["ACTO", "ESCRITO"]
        self.autos = ""

    def parrafos_acto(self):
        return ["la Sala resolvió"]

    def parrafos_conceptos(self):
        return ["la parte alega"]


def resultado(problemas=None):
    return types.SimpleNamespace(
        fases=Fases(problemas if problemas is not None else [
            {"pregunta": "¿Principal?", "jerarquia": "principal"}, {"pregunta": "¿Accesorio?"}]),
        encargo=types.SimpleNamespace(es_recurso=False))


def entorno(marcas, nucleo=None, registro=None):
    registro = registro if registro is not None else {}

    def leer(email, numero, clave):
        v = marcas.get(clave)
        return json.loads(json.dumps(v)) if v is not None else None

    def leer_varias(email, numero, claves):
        return {k: leer(email, numero, k) for k in claves if k in marcas}

    def guardar(email, numero, clave, doc, huella=""):
        marcas[clave] = json.loads(json.dumps(doc))
        return True

    async def nucleo_defecto(email, numero, ses, contexto):
        registro.setdefault("nucleo", []).append(contexto)
        return {"estado": "lista", "formato": 3, "global": {"sentido": "fundado"}, "propuestas": []}

    async def _noop(*a, **k):
        return None

    ns = {"asyncio": asyncio, "time": time, "uuid": uuid, "json": json, "hashlib": hashlib, "_te": te,
          "HTTPException": HTTPException, "Form": lambda *a, **k: None, "app": _App(),
          "_taller_puerta": lambda *a, **k: None, "_taller_purgar": lambda *a, **k: None,
          "_taller_leer_marca": leer, "_taller_leer_marcas": leer_varias, "_taller_guardar_marca": guardar,
          "_taller_proponer_nucleo": nucleo or nucleo_defecto, "_taller_firma_origen": lambda: "",
          "_taller_sesion_en_memoria": lambda *a, **k: None,
          "_taller_plan_adelantar": lambda e: True,
          "_taller_plan_desde_propuesta": lambda *a, **k: registro.setdefault("plan", []).append(1) or _noop(),
          "_taller_lanzar_deliberacion": lambda *a, **k: registro.setdefault("delib", []).append(a[-1] if len(a) > 5 else ""),
          "_taller_redecidir_si_corrigio": _noop, "_taller_es_casa": lambda e: False,
          "_taller_cuenta_de_pruebas": lambda e: True, "_taller_registrar_uso": lambda *a, **k: None,
          "_taller_guardar_global": lambda *a, **k: None,
          "_taller_con_aplicado": lambda resp, *a, **k: resp,
          "_TALLER_EN_MARCHA": set(), "err": lambda e: str(e)}
    for f in ("taller_responder", "_taller_preproponer", "_taller_hash_contexto", "_taller_con_latido",
              "_taller_propuesta_unica", "_taller_propuesta_reclamada", "_taller_marca_es_mia",
              "_taller_analisis_guardado", "taller_proponer", "_taller_formato_propuesta"):
        exec(compile(ast.Module(body=[FN[f]], type_ignores=[]), "main.py", "exec"), ns)
    return ns, registro


def prem(i, problema=1):
    return {"id": f"F{i}", "segmento": f"C1.{chr(96 + i)}", "problema": problema,
            "afirma_la_parte": f"hecho {i}", "el_acto": "no_se_pronuncia", "carga_de": "la_parte",
            "pregunta": f"¿Ocurrió el hecho {i}?", "tipo": "si_no", "si_si": "fundado", "si_no": "infundado",
            "para_que": "x"}


ANALISIS = {"razones": [], "hechos": [], "argumentos": [], "premisas": [prem(1), prem(2, problema=2)]}

print("\n1 · CONTESTAR NO TIRA LO APORTADO (/taller/responder con «contexto»)")
con_bandera(True)
R = resultado()
HU = te.huella_contraste(R)
_q = ps.preguntas_de(ANALISIS, te.problemas_de(R))
_ind = next(x["id"] for x in _q if x["indispensable"])
marcas = {"analisis": {"huella": HU, "huella_analisis": al.huella(R), "estado": "listo", "doc": ANALISIS},
          "propuesta": {"huella": HU, "estado": "listo",
                        "respuesta": ps.respuesta_preguntas("1/2026", _q)}}
lanzadas = []
ns, reg = entorno(marcas)


async def _pre_falsa(email, numero, r, material, forzar=False, contexto=""):
    lanzadas.append({"forzar": forzar, "contexto": contexto})


ns["_taller_preproponer"] = _pre_falsa
ns["_taller_recuperar_sesion"] = lambda e, n: {"resultado": R, "material": object()}
APORTADO = "Cláusula sexta del contrato: el arrendatario renuncia al derecho del tanto."
out = asyncio.run(ns["taller_responder"](numero="1/2026", user_email="a@b.c",
                                          respuestas_json=json.dumps([{"id": _ind, "respuesta": "si"}]),
                                          contexto="  " + APORTADO + " "))
ok(out["propuesta"] == "en_curso" and lanzadas and lanzadas[-1]["contexto"] == APORTADO,
   "la propuesta que se relanza al contestar recibe lo aportado (antes: «»)")
ok(marcas["propuesta"].get("contexto_hash") == ns["_taller_hash_contexto"](APORTADO),
   "y la marca en curso lleva la huella de ese contexto")
lanzadas.clear()
marcas["respuestas"] = {"huella": HU, "items": []}
asyncio.run(ns["taller_responder"](numero="1/2026", user_email="a@b.c",
                                   respuestas_json=json.dumps([{"id": _ind, "respuesta": "no"}])))
ok(lanzadas[-1]["contexto"] == "" and "contexto_hash" not in marcas["propuesta"],
   "sin contexto, como hasta hoy")

print("\n2 · LA RELANZADA LO LLEVA HASTA EL NÚCLEO Y LO SELLA")
con_bandera(True)
marcas = {}
ns, reg = entorno(marcas)
asyncio.run(ns["_taller_preproponer"]("a@b.c", "1/2026", R, object(), forzar=True, contexto=APORTADO))
ok(reg.get("nucleo") == [APORTADO], "el núcleo recibe el contexto aportado")
ok(marcas["propuesta"]["estado"] == "listo"
   and marcas["propuesta"]["contexto_hash"] == ns["_taller_hash_contexto"](APORTADO)
   and marcas["propuesta"]["respuesta"]["contexto_hash"] == marcas["propuesta"]["contexto_hash"],
   "la guardada se sella con la huella del contexto")
ok(not reg.get("plan") and reg.get("delib") == [APORTADO],
   "con contexto no se adelanta el plan (su clave no casaría) y la deliberación lo recibe")
marcas = {}
ns, reg = entorno(marcas)
con_bandera(False)
asyncio.run(ns["_taller_preproponer"]("a@b.c", "1/2026", R, object()))
ok(reg.get("nucleo") == [""] and "contexto_hash" not in marcas["propuesta"] and reg.get("plan"),
   "la de fondo sin contexto: la marca de siempre, con su plan")

print("\n3 · /taller/proponer SIRVE LA GUARDADA CON ESE CONTEXTO, Y NO LA DE OTRO")


def proponer(marcas, contexto="", bandera=True):
    con_bandera(bandera)
    ns, reg = entorno(marcas)
    ns["_taller_recuperar_sesion"] = lambda e, n: {"resultado": resultado(), "material": object(), "consultado": True}
    out = asyncio.run(ns["taller_proponer"](numero="1/2026", user_email="a@b.c", contexto=contexto,
                                            recalcular="", banderas=""))
    return out, reg


_h = None
ns0, _ = entorno({})
_h = ns0["_taller_hash_contexto"](APORTADO)
_guardada = {"estado": "lista", "formato": 3, "global": {"sentido": "infundado"}, "propuestas": [],
             "origen_firma": "", "respuestas_firma": "", "contexto_hash": _h}
out, reg = proponer({"propuesta": {"huella": HU, "estado": "listo", "contexto_hash": _h,
                                   "respuesta": dict(_guardada)}}, contexto=APORTADO)
ok(out["global"]["sentido"] == "infundado" and not reg.get("nucleo"),
   "pedida con el mismo contexto: se sirve la guardada (no se paga dos veces)")
out, reg = proponer({"propuesta": {"huella": HU, "estado": "listo", "contexto_hash": _h,
                                   "respuesta": dict(_guardada)}}, contexto="otro documento distinto")
ok(reg.get("nucleo") == ["otro documento distinto"], "con OTRO contexto se calcula con él")
out, reg = proponer({"propuesta": {"huella": HU, "estado": "listo",
                                   "respuesta": dict(_guardada, contexto_hash="")}}, contexto=APORTADO)
ok(reg.get("nucleo") == [APORTADO], "una guardada sin huella de contexto no se sirve a quien manda contexto (hoy)")
out, reg = proponer({"propuesta": {"huella": HU, "estado": "listo", "contexto_hash": _h,
                                   "respuesta": dict(_guardada)}}, contexto="")
ok(out["global"]["sentido"] == "infundado" and not reg.get("nucleo"),
   "sin contexto (la pantalla recoge tras responder): se sirve la hecha con lo aportado")

print("\n4 · REVERSIÓN: SIN LA BANDERA, NADA SE QUEDA EN «PREGUNTAS»")
_preg = ps.respuesta_preguntas("1/2026", _q)
out, reg = proponer({"propuesta": {"huella": HU, "estado": "listo", "respuesta": _preg}}, bandera=True)
ok(out["estado"] == "preguntas" and not reg.get("nucleo"), "con la bandera, la guardada «preguntas» se sirve")
out, reg = proponer({"propuesta": {"huella": HU, "estado": "listo", "respuesta": _preg}}, bandera=False)
ok(out.get("estado") != "preguntas" and reg.get("nucleo") == [""],
   "sin la bandera, se recalcula (antes: se servía y el asunto quedaba atascado)")
import tarjeta_decision as td
_fila = {"propuesta": {"huella": HU, "estado": "listo", "respuesta": _preg},
         "global_propuesta": {"huella": HU, "global": {"sentido": "fundado", "alcanza": True}}}
con_bandera(True)
ok(td.elegir_marcas(_fila, HU)["estado_calculo"] == "sin_propuesta",
   "la tarjeta, con la bandera, espera las respuestas")
con_bandera(False)
_em = td.elegir_marcas(_fila, HU)
ok(_em["estado_calculo"] != "sin_propuesta" or _em["propuesta"] is not None,
   "sin la bandera, la marca «preguntas» no bloquea la tarjeta (cae a la global de siempre)")
ok(td.elegir_marcas({"propuesta": _fila["propuesta"]}, HU)["estado_calculo"] == "sin_propuesta",
   "y sin otra global, «sin propuesta», como cualquier marca inválida")

print("\n5 · CORREGIR EL PROBLEMA NO TIRA LAS RESPUESTAS (/taller/problema)")


class _Tabla:
    def __init__(self, fila):
        self.fila = fila
        self._upd = None

    def select(self, *a):
        self._upd = None
        return self

    def update(self, d):
        self._upd = d
        return self

    def eq(self, *a):
        return self

    def limit(self, *a):
        return self

    def execute(self):
        if self._upd is not None:
            self.fila.update(json.loads(json.dumps(self._upd)))
            return types.SimpleNamespace(data=[{"actualizado_en": "t"}])
        return types.SimpleNamespace(data=[json.loads(json.dumps(self.fila))])


def corregir(indice, pregunta="", jerarquia="", bandera=True):
    con_bandera(bandera)
    r = resultado()
    hu0 = te.huella_contraste(r)
    items = [{"id": _ind, "pregunta": "¿Ocurrió el hecho 1?", "respuesta": "si", "afirma_la_parte": "hecho 1"}]
    fila = {"estado": {"huella": hu0, "fases": {},
                       "respuestas": {"huella": hu0, "items": items, "firma": ps.firma(items)},
                       "analisis": {"huella": hu0, "huella_analisis": al.huella(r), "estado": "listo",
                                    "doc": ANALISIS}}}
    tabla = _Tabla(fila)
    ns = {"HTTPException": HTTPException, "Form": lambda *a, **k: None, "app": _App(), "_te": te,
          "_taller_puerta": lambda *a, **k: None, "err": str,
          "_taller_recuperar_sesion": lambda e, n: {"resultado": r},
          "supabase_admin": types.SimpleNamespace(table=lambda *a: tabla)}
    exec(compile(ast.Module(body=[FN["taller_problema"]], type_ignores=[]), "main.py", "exec"), ns)
    asyncio.run(ns["taller_problema"](numero="1/2026", user_email="a@b.c", indice=indice,
                                      pregunta=pregunta, jerarquia=jerarquia))
    return r, fila["estado"], hu0


r5, est5, hu0 = corregir(1, jerarquia="principal")
hu1 = te.huella_contraste(r5)
ok(hu1 != hu0 and est5["huella"] == hu1, "hacer principal cambia la huella del adelanto (como siempre)")
ok(est5["respuestas"]["huella"] == hu1 and ps.items_de_marca(est5["respuestas"], hu1),
   "las respuestas pasan a la huella nueva: se siguen leyendo (antes: [])")
ok(est5["analisis"]["huella"] == hu1,
   "y el análisis también: sólo cambió la jerarquía, que no lee (no se vuelve a pagar)")
_doc = est5["analisis"]["doc"] if est5["analisis"]["huella_analisis"] == al.huella(r5) else None
_q5 = ps.preguntas_de(_doc, te.problemas_de(r5), ps.respuestas_de(ps.items_de_marca(est5["respuestas"], hu1)))
ok(next(p for p in _q5 if p["id"] == _ind)["respuesta"] == "si",
   "la pregunta ya contestada sigue contestada (no vuelve)")
r6, est6, hu0 = corregir(0, pregunta="¿El emplazamiento practicado con la hermana de la demandada es válido?")
hu6 = te.huella_contraste(r6)
ok(est6["respuestas"]["huella"] == hu6 and est6["analisis"]["huella"] == hu0,
   "reescribir la pregunta: las respuestas siguen; el análisis (que sí la lee) se rehace")
r7, est7, hu0 = corregir(1, jerarquia="principal", bandera=False)
ok(est7["respuestas"]["huella"] == te.huella_contraste(r7),
   "el resellado no depende de la bandera: sólo toca lo que ya está guardado")
ok(ps.resellar_tras_corregir({"huella": "x"}, "a", "b") == [] and ps.resellar_tras_corregir(None, "a", "b") == [],
   "sin respuestas ni análisis guardados (banderas apagadas), no toca nada")

print("\n6 · SI EL ANÁLISIS SE REHACE, LA RESPUESTA SE REENGANCHA POR SU PREGUNTA")
_items = [{"id": "Pviejo", "pregunta": "¿Ocurrió el hecho 1?", "respuesta": "si", "afirma_la_parte": "hecho 1"}]
_rd = ps.respuestas_de(_items)
ok(ps.firma(_items) == ps.firma([{"id": "Pviejo", "respuesta": "si"}]),
   "los alias no entran en la firma (las propuestas guardadas siguen valiendo)")
_qn = ps.preguntas_de(ANALISIS, te.problemas_de(R), _rd)
ok(next(p for p in _qn if p["premisa"] == "F1")["respuesta"] == "si" and not ps.pendientes(_qn),
   "otro id, misma pregunta: contestada, y la propuesta no se detiene otra vez")
_m, _ = ps.marca_respuestas(HU, _items, [{"id": _qn[0]["id"], "respuesta": "no"}], _qn)
ok([x["id"] for x in _m["items"]] == [_qn[0]["id"]] and _m["items"][0]["respuesta"] == "no",
   "si la cambia, la vieja se sustituye: el bloque no dice «sí» y «no» a la misma pregunta")
ok(ps.bloque_respuestas(_m["items"]).count("Respuesta:") == 1, "una sola respuesta en el bloque")

print("\n7 · EL AVISO DE «VOZ DE HERRAMIENTA», CALIBRADO")
BUENAS = ["De lo aportado al juicio no se advierte que el actor acreditara el despido.",
          "Con lo aportado en autos se acredita la relación laboral.",
          "La información proporcionada en respuesta al requerimiento no desvirtúa la presunción.",
          "La documentación proporcionada a la autoridad fiscalizadora resultó insuficiente.",
          "El expediente proporcionado al perito no incluía los estados de cuenta.",
          "Las constancias proporcionadas por la responsable acreditan el emplazamiento."]
_fp = [x for x in BUENAS if ps.frases_de_herramienta(x)]
ok(not _fp, f"no acusa giros normales con destino o fuente procesal ({_fp})")
MALAS = ["La información proporcionada no permite advertir el pago.",
         "Con el material proporcionado no es posible verificarlo.",
         "Del texto proporcionado se advierte la firma.",
         "Lo aportado no basta para tenerlo por acreditado.",
         "Las constancias proporcionadas no incluyen el acuse."]
ok(all(ps.frases_de_herramienta(x) for x in MALAS), "y sigue cazando la voz de la herramienta")

ct.poner(False)
print()
if FALLOS:
    print(f"FALLAN {len(FALLOS)}: " + " · ".join(FALLOS))
    sys.exit(1)
print("RESULTADO: TODAS LAS COMPROBACIONES PASAN")
