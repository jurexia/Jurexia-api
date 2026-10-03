# -*- coding: utf-8 -*-
"""LA PROPUESTA POR PROBABILIDAD (2-oct-2026), sin red y sin modelo.

    .venv/bin/python test_probabilidad_sentido.py

David: «si hay un 50.01% de probabilidad hacia un lado sea esa la propuesta de
resolución». Aquí se comprueba lo puro de `probabilidad_sentido`: el cálculo,
el volteo con la vía contraria que el motor ya escribió, la explicación (sin
faltas), los tipos sin tasa y los tres tipos nuevos; y que el prompt de la
fase 5 queda IDÉNTICO al de la base con las banderas apagadas.
"""
import copy
import importlib.util
import os
import re
import subprocess
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
FALLOS = []


def ok(c, q):
    print(f"   {'PASA ' if c else 'FALLA'}  {q}")
    if not c:
        FALLOS.append(q)


for _v in ("PROPUESTA_POR_PROBABILIDAD", "PREGUNTAS_AL_SECRETARIO"):
    os.environ.pop(_v, None)

import probabilidad_sentido as ps
import fase5_propuesta as f5

print("\n1 · EL CÁLCULO")
r = ps.calcular("amparo_directo", [], "fundado")
# Medido: 29.1% de los amparos directos prosperan; el voto «concede» del motor
# pesa 1.17 → 0.324. El lado es «no prospera».
ok(r["p_prospera"] == 0.324 and r["lado"] == "no_prospera" and r["fuente"] == "jurimetria",
   f"AD con el motor a favor: p={r['p_prospera']} → {r['lado']}")
ok(ps.calcular("amparo_directo", [], None)["p_prospera"] == 0.291, "sin voto del motor: la tasa sola")
_fil = [{"similitud": 95, "calificacion": "fundado", "neun": 1, "expediente": "1/2024"},
        {"similitud": 90, "calificacion": "esencialmente fundado", "neun": 2, "expediente": "2/2024"}]
r2 = ps.calcular("amparo_directo", _fil, "fundado")
ok(r2["lado"] == "prospera" and r2["p_prospera"] > 0.5 and r2["precedentes"]["n"] == 2,
   f"dos precedentes del mismo problema que prosperaron voltean la tasa: p={r2['p_prospera']}")
ok(ps.calcular("amparo_directo", [{"similitud": 40, "calificacion": "fundado", "neun": 3}], None)
   ["precedentes"]["n"] == 0, "un parecido por debajo del 50% no cuenta")
ok(ps.calcular("amparo_directo", [{"similitud": 99, "calificacion": "no se estudió", "neun": 4}], None)
   ["precedentes"]["n"] == 0, "un planteamiento que no se estudió no vota")
_ex = ps.calcular("queja", [{"similitud": 90, "calificacion": "fundado pero insuficiente", "neun": 5}], None)
ok(_ex["precedentes"]["filas"][0]["prospero"] is False, "«fundado pero insuficiente» no prospera")
ok(ps.calcular("amparo_directo", [], "inoperante")["motor"] == 0
   and ps.calcular("amparo_directo", [], "innecesario")["motor"] is None,
   "el voto del motor: inoperante = no prospera; innecesario = no vota")
# EL 50.01%: con p exactamente 0.5 no se pasa del 50, y el lado es «no prospera».
_cal = {"tasas": {"amparo_directo": {"prospera": 0.5, "n": 10}}, "kappa": 3, "motor": {}}
ok(ps.calcular("amparo_directo", [], None, _cal)["lado"] == "no_prospera", "0.50 no pasa del 50%")
_cal2 = {"tasas": {"amparo_directo": {"prospera": 0.5001, "n": 10}}, "kappa": 3, "motor": {}}
ok(ps.calcular("amparo_directo", [], None, _cal2)["lado"] == "prospera", "0.5001 sí: ése es el lado")

print("\n2 · LOS TIPOS")
for t, k in (("amparo directo", "amparo_directo"), ("Amparo en revisión", "amparo_revision"),
             ("RF", "revision_fiscal"), ("Recurso de reclamación", "reclamacion"),
             ("recurso de inconformidad", "inconformidad"), ("Impedimento", "impedimento")):
    ok(ps.clave_tipo(t) == k and ps.tasa(t)[0] is not None, f"«{t}» → {k} con tasa {ps.tasa(t)}")
ok(ps.tasa("reclamacion") == (0.177, 265) and ps.tasa("inconformidad") == (0.353, 153)
   and ps.tasa("impedimento") == (0.714, 35), "las tres tasas nuevas, con su n")
ok(ps.calcular("impedimento", [], None)["lado"] == "prospera", "el impedimento prospera el 71%: ése es el lado")
_sin = ps.calcular("juicio_raro", [], "fundado")
ok(_sin["p_prospera"] is None and _sin["lado"] == "prospera" and _sin["fuente"] == "motor",
   "un tipo sin tasa: decide el motor, sin número inventado")
_nada = ps.calcular("juicio_raro", [], None)
ok(_nada["lado"] is None and "no hay número" in _nada["explicacion"], "sin tasa ni motor: no hay lado")

print("\n3 · LA EXPLICACIÓN SE LEE")
_ORTO = [r"\bprobabilidad de \d", r"\bningún precedente\b", r"\s,", r"\.\.", r"  ", r"\bel el\b",
         r"\bde de\b", r"\(\s", r"\s\)", r"\bun asuntos\b", r"\b1 precedentes\b", r"\b1 asuntos\b"]
for t in ("amparo_directo", "amparo_revision", "queja", "revision_fiscal", "reclamacion",
          "inconformidad", "impedimento"):
    for filas in ([], _fil[:1], _fil):
        for v in (None, "fundado", "infundado"):
            for propio in (True, False):
                e = ps.calcular(t, filas, v, propio=propio)["explicacion"]
                malas = [x for x in _ORTO if re.search(x, e)]
                if malas or not e.startswith("Se propone ") or not e.endswith("."):
                    ok(False, f"{t}/{len(filas)}/{v}/{propio}: {malas} · {e}")
ok(True, "ninguna explicación con dobles espacios, comas sueltas, «1 precedentes» ni «ningún precedente» como fuente")
e1 = ps.calcular("amparo_directo", _fil[:1], "fundado")["explicacion"]
ok("1 precedente del tribunal sobre el mismo problema (1 en que prosperó y 0 en que no)" in e1,
   f"singular bien dicho: {e1}")
ok("No hay precedentes del tribunal" in ps.calcular("queja", [], None)["explicacion"],
   "lo que no hay se dice aparte")
ok("referencia" in ps.calcular("queja", [], None, propio=False)["explicacion"],
   "si el tribunal no es el de la tasa, se dice que es una referencia")
ok("%" in e1 and "probabilidad del" in e1, "dice el porcentaje del lado propuesto")

print("\n4 · EL VOLTEO, CON LA VÍA QUE EL MOTOR YA ESCRIBIÓ")
PROBS = [{"pregunta": "¿El emplazamiento fue legal?", "jerarquia": "principal",
          "combate": "dice que no hubo cercioramiento", "resolvio": "lo tuvo por legal"},
         {"pregunta": "¿Procede la condena en costas?", "jerarquia": "accesorio"}]


def caso():
    props = [f5.Propuesta(problema=PROBS[0]["pregunta"], sentido="fundado", razon="razón del motor",
                          apoyos=["2001111"], confianza="alta"),
             f5.Propuesta(problema=PROBS[1]["pregunta"], sentido="innecesario", razon="sin materia")]
    g = f5.Global(sentido="fundado", razon="el asunto se concede", efecto="las costas quedan sin materia",
                  apoyos=["2001111"], confianza="alta", en_contra="la objeción del motor",
                  alternativa={"sentido": "infundado", "razon": "el emplazamiento se cercioró",
                               "efecto": "las costas se estudian", "apoyos": ["2002222"]},
                  checklist=[{"numero": 2, "con_propuesta": "innecesario", "con_alternativa": "infundado",
                              "si_prospera": {"sentido": "innecesario", "razon": "x"},
                              "si_no_prospera": {"sentido": "infundado", "razon": "y"}}])
    return props, g


props, g = caso()
info = ps.aplicar(g, props, PROBS, "amparo_directo", [])
ok(info["volteada"] and info["probabilidad"]["volteada"] and not info["necesita_razon"],
   "el motor concedía y la probabilidad niega: se voltea sin pedir otra razón")
ok(g.sentido == "infundado" and g.razon == "el emplazamiento se cercioró" and g.apoyos == ["2002222"]
   and g.efecto == "las costas se estudian", "la global pasa a ser la vía contraria que el motor escribió")
ok(g.alternativa == {"sentido": "fundado", "razon": "el asunto se concede",
                     "efecto": "las costas quedan sin materia", "apoyos": ["2001111"]},
   "y lo que propuso el motor queda como alternativa, entero")
ok(g.en_contra == "el asunto se concede", "su razón es la objeción a la propuesta")
ok(g.checklist[0]["con_propuesta"] == "infundado" and g.checklist[0]["con_alternativa"] == "innecesario"
   and g.checklist[0]["si_prospera"]["sentido"] == "innecesario",
   "la lista se intercambia (con_propuesta ↔ con_alternativa) y la suerte condicional no se toca")
ok(props[0].sentido == "infundado" and props[0].razon == "el emplazamiento se cercioró"
   and props[0].origen == "probabilidad" and props[0].sentido_motor == "fundado"
   and not getattr(props[0], "sentido_propio", ""),
   "el principal toma el lado y la razón; lo del motor va a sentido_motor, NO a sentido_propio")
ok(props[1].sentido == "innecesario", "los accesorios no se tocan aquí: los recalcula el árbol en main")
ok(g.confianza == "media" and g.confianza_motor == "alta",
   f"la confianza sale de la probabilidad (0.68 → media) y la del motor se guarda: {g.confianza}")
ok(g.alcanza is True and info["probabilidad"]["sentido_motor"] == "fundado",
   "alcanza = hay sentido; la probabilidad recuerda qué dijo el motor")
ok(any("SE VOLTEÓ" in a for a in info["avisos"]) and "lo contrario" in info["probabilidad"]["explicacion"],
   "se avisa y la explicación lo dice")

print("\n5 · EL MOTOR YA ESTABA DEL LADO QUE GANA")
props, g = caso()
g.sentido, props[0].sentido = "infundado", "infundado"
g.alternativa = {"sentido": "fundado", "razon": "r", "efecto": "", "apoyos": []}
info = ps.aplicar(g, props, PROBS, "amparo_directo", [])
ok(not info["volteada"] and g.sentido == "infundado" and g.razon == "el asunto se concede"
   and g.alternativa["sentido"] == "fundado", "no se toca nada más que la confianza")
ok(g.confianza == "alta" and g.confianza_motor == "alta", "niega con el motor: 0.81 → alta")

print("\n6 · SIN ALTERNATIVA ESCRITA: SE PIDE UNA RAZÓN")
props, g = caso()
g.alternativa = {"sentido": "", "razon": "", "efecto": "", "apoyos": []}
info = ps.aplicar(g, props, PROBS, "amparo_directo", [])
ok(info["volteada"] and info["necesita_razon"] and g.sentido == "infundado" and g.razon == ""
   and props[0].sentido == "infundado" and g.alternativa["sentido"] == "fundado",
   "lado nuevo sin razón: main la pide con _f5.razonar; lo del motor queda como alternativa")
props, g = caso()
g.alternativa = {"sentido": "fundado", "razon": "misma vía", "efecto": "", "apoyos": []}
info = ps.aplicar(g, props, PROBS, "amparo_directo", [])
ok(info["necesita_razon"], "una «alternativa» del mismo lado no es la vía contraria: se pide razón")

print("\n7 · EL MOTOR NO DIJO NADA")
props = [f5.Propuesta(problema=PROBS[0]["pregunta"], alcanza=False),
         f5.Propuesta(problema=PROBS[1]["pregunta"], alcanza=False)]
g = f5.Global(alcanza=False)
info = ps.aplicar(g, props, PROBS, "queja", [])
ok(not info["volteada"] and info["necesita_razon"] and g.sentido == "infundado" and g.alcanza
   and props[0].sentido == "infundado" and props[0].origen == "probabilidad",
   "sin nada del motor, el lado de la tasa (queja: 66% infundada)")
ok(any("no propuso ningún sentido" in a for a in info["avisos"]), "y se dice")
# Sólo el principal, sin global: la global es la del principal.
props = [f5.Propuesta(problema=PROBS[0]["pregunta"], sentido="infundado", razon="rp", apoyos=["1"])]
g = f5.Global(alcanza=False)
info = ps.aplicar(g, props, PROBS[:1], "amparo_directo", [])
ok(g.sentido == "infundado" and g.razon == "rp" and g.problema_que_decide == PROBS[0]["pregunta"]
   and not info["volteada"], "sin global, la del asunto es la del principal (del que cuelga)")
# El principal es el de la jerarquía, no el primero.
_pj = [{"pregunta": "¿Costas?"}, {"pregunta": "¿Fondo?", "jerarquia": "principal"}]
ok(ps.indice_principal(_pj, [object(), object()]) == 1, "el principal por jerarquía")
ok(ps.indice_principal([{"pregunta": "a"}, {"pregunta": "b"}], [1, 2]) == 0, "sin jerarquía, el primero")
ok(ps.indice_principal([], []) == -1, "sin problemas, ninguno")
# Un dict también sirve (la global guardada).
gd = {"sentido": "fundado", "razon": "x", "alternativa": {"sentido": "infundado", "razon": "y"},
      "checklist": []}
ps.aplicar(gd, [{"problema": "¿P?", "sentido": "fundado", "razon": "x"}], [{"pregunta": "¿P?"}],
           "revision_fiscal", [])
ok(gd["sentido"] == "infundado" and gd["alternativa"]["sentido"] == "fundado", "con dicts, igual")
ok(ps.confianza_de(0.8) == "alta" and ps.confianza_de(0.65) == "media" and ps.confianza_de(0.6) == "baja",
   "la confianza: alta ≥0.80, media ≥0.65, baja por debajo")

print("\n8 · CON LAS BANDERAS APAGADAS, EL PROMPT DE LA BASE LETRA POR LETRA")
_aqui = os.path.dirname(os.path.abspath(__file__))
try:
    _base_src = subprocess.run(["git", "show", "b89e049:fase5_propuesta.py"], cwd=_aqui,
                               capture_output=True, text=True, check=True).stdout
except Exception as _e:
    _base_src = ""
    print(f"   (sin la base en git: {type(_e).__name__}; se salta la comparación)")
if _base_src:
    _m = types.ModuleType("f5_base")
    sys.modules["f5_base"] = _m
    exec(compile(_base_src, "f5_base.py", "exec"), _m.__dict__)

    class _M:
        tipo_asunto = "amparo_directo"; sondeo = None; tesis = []; normas = []; espejo = None

    for _t in ("amparo_directo", "amparo_revision", "queja", "revision_fiscal"):
        _M.tipo_asunto = _t
        _a = _m.prompt_propuesta(PROBS, _M(), "acto", "conceptos", _t != "amparo_directo", "", "")
        _b = f5.prompt_propuesta(PROBS, _M(), "acto", "conceptos", _t != "amparo_directo", "", "")
        ok(_a == _b, f"{_t}: prompt idéntico con las banderas apagadas")
    for _s in ("fundado", "infundado"):
        ok(_m.bloque_direccion(_s, "amparo_directo", False, "c", "r")
           == f5.bloque_direccion(_s, "amparo_directo", False, "c", "r"),
           f"bloque_direccion «{_s}» idéntico")
    os.environ["PROPUESTA_POR_PROBABILIDAD"] = "todos"
    _M.tipo_asunto = "amparo_directo"
    _c = f5.prompt_propuesta(PROBS, _M(), "acto", "conceptos", False, "", "")
    ok("Pon alcanza=false" not in _c and "SIEMPRE DECIDES UN SENTIDO" in _c
       and "«el acervo no ofrece criterio para" not in _c and '"sostenida": true|false' in _c
       and '"alcanza": true' not in _c and "deja «apoyos» vacío" in _c,
       "con la probabilidad: decide siempre, «sostenida» en vez de «alcanza», y no inventa citas")
    ok("13. LAS CONSTANCIAS" in _c, "la regla 13 sigue sin la bandera de las preguntas")
    os.environ["PREGUNTAS_AL_SECRETARIO"] = "todos"
    _d = f5.prompt_propuesta(PROBS, _M(), "acto", "conceptos", False, "", "")
    ok("13. LAS CONSTANCIAS" not in _d and '"constancias": [' not in _d
       and "LO QUE SOSTIENE LA PARTE NO ES UN HECHO" in _d and "según la resolución y los autos" in _d,
       "con las preguntas: sin regla 13 ni constancias en el JSON; la parte se verifica")
    ok("con lo que consta en la resolución o en autos, que lo que ellos sostienen es cierto"
       in f5.bloque_direccion("fundado", "amparo_directo", False, "c", "r"),
       "bloque_direccion exige que conste lo que la parte sostiene")
    os.environ.pop("PROPUESTA_POR_PROBABILIDAD")
    os.environ.pop("PREGUNTAS_AL_SECRETARIO")

print("\n9 · EL PARSEO DE LA PROPUESTA")


class _Cli:
    def __init__(self, txt):
        self.txt = txt


async def _crear(cliente, **kw):
    return types.SimpleNamespace(choices=[types.SimpleNamespace(
        message=types.SimpleNamespace(content=cliente.txt))])


import asyncio
import json
import llamada_modelo as _lm
_lm_orig = _lm.crear
_lm.crear = _crear
_RESP = {"propuestas": [{"problema": PROBS[0]["pregunta"], "sentido": "fundado", "razon": "r",
                         "apoyos": [], "confianza": "baja", "alcanza": False}],
         "global": {"sentido": "fundado", "razon": "g", "alcanza": False, "confianza": "baja",
                    "constancias": [{"que": "el acta", "indispensable": True, "problema": 1}],
                    "checklist": []}}
_SOLO_GLOBAL = {"propuestas": [], "global": {"sentido": "infundado", "razon": "g", "sostenida": False}}


class _Mat:
    tipo_asunto = "amparo_directo"; sondeo = None; tesis = []; normas = []; espejo = None


def _prop(resp):
    return asyncio.run(f5.proponer(_Cli(json.dumps(resp)), PROBS, _Mat(), "a", "c",
                                   contraste_previo=[]))


try:
    p, gl, av = _prop(_RESP)
    ok(not p[0].alcanza and not gl.alcanza and gl.constancias,
       "apagada: alcanza=false suprime el sentido y las constancias se leen, como ayer")
    ok(_prop(_SOLO_GLOBAL)[0] == [] and not _prop(_SOLO_GLOBAL)[1].alcanza,
       "apagada: sin propuestas por problema se tira también la global, como ayer")
    os.environ["PROPUESTA_POR_PROBABILIDAD"] = "todos"
    p, gl, av = _prop(_RESP)
    ok(p[0].alcanza and p[0].sostenida is False and gl.alcanza and gl.sostenida is False
       and gl.sentido == "fundado",
       "encendida: con sentido alcanza; lo que el modelo dijo del acervo va a «sostenida»")
    p2, gl2, _ = _prop(_SOLO_GLOBAL)
    ok(p2 == [] and gl2.alcanza and gl2.sentido == "infundado" and gl2.sostenida is False,
       "encendida: la global se lee aunque no vengan propuestas por problema")
    os.environ["PREGUNTAS_AL_SECRETARIO"] = "todos"
    ok(_prop(_RESP)[1].constancias == [], "con las preguntas: la global ya no trae constancias")
finally:
    _lm.crear = _lm_orig
    os.environ.pop("PROPUESTA_POR_PROBABILIDAD", None)
    os.environ.pop("PREGUNTAS_AL_SECRETARIO", None)

print()
if FALLOS:
    print(f"FALLAN {len(FALLOS)}: " + " · ".join(FALLOS))
    raise SystemExit(1)
print("RESULTADO: TODAS LAS COMPROBACIONES PASAN")
