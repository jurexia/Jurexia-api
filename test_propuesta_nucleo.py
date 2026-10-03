# -*- coding: utf-8 -*-
"""EL VOLTEO EN EL NÚCLEO DE LA PROPUESTA (2-oct-2026), con objetos falsos.

    .venv/bin/python test_propuesta_nucleo.py

Corre `main._taller_proponer_nucleo` entero —emparejar, la capa de
probabilidad, reconciliar, el árbol y la respuesta— con el modelo, internet, la
base y Qdrant sustituidos por falsos: ninguna llamada sale de la máquina. Las
llaves de abajo son de mentira y sólo existen para que `main` se importe en un
árbol de trabajo sin `.env` (el arranque, que borra cachés, NO corre:
importar no lo dispara).

Comprueba el contrato A con la bandera «propuesta_por_probabilidad»: formato 3,
estado «lista», global.probabilidad, el volteo (la global pasa a ser la vía
contraria que el motor escribió), el principal con el lado nuevo, los
accesorios recalculados por el árbol, el hueco rellenado por el árbol (origen
«arbol»), criterios_json con todos; y, apagada, la respuesta de formato 2.
"""
import asyncio
import json
import os
import sys
import types

for _k, _v in {"OPENAI_API_KEY": "sk-falsa-de-prueba", "OPENROUTER_API_KEY": "sk-falsa-de-prueba",
               "DEEPSEEK_API_KEY": "sk-falsa-de-prueba", "MISTRAL_API_KEY": "falsa",
               "GEMINI_API_KEY": "falsa", "QDRANT_URL": "http://127.0.0.1:9"}.items():
    os.environ.setdefault(_k, _v)
for _v in ("PROPUESTA_POR_PROBABILIDAD", "PREGUNTAS_AL_SECRETARIO"):
    os.environ.pop(_v, None)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
FALLOS = []


def ok(c, q):
    print(f"   {'PASA ' if c else 'FALLA'}  {q}")
    if not c:
        FALLOS.append(q)


import main
import fase5_propuesta as f5
import fase_internet

main.supabase_admin = None
main.qdrant_client = None


async def _nada(*a, **k):
    return None


async def _param(*a, **k):
    return "", ""


async def _web(*a, **k):
    return {"buscado": False, "tesis": [], "pistas": []}


main._decidir_oportunidad = lambda *a, **k: None
main._taller_clasificar_cumplimiento = _nada
main._con_autos = lambda r, c: c or ""
main._taller_esperar_contraste = _nada
main._taller_esperar_analisis = _nada
main._taller_requisitos = _nada
main._taller_parametro = _param
main._taller_recurrente = lambda r: ""
main._taller_papel_recurrente = lambda r: ""
main._taller_ficha_bloque = lambda r, *a, **k: ""
main._taller_firma_origen = lambda: ""
fase_internet.precedentes_verificados = _web

P1 = "¿El emplazamiento al quejoso se practicó con el cercioramiento que exige la ley?"
P2 = "¿Procede la condena en costas impuesta en la sentencia reclamada?"
P3 = "¿La responsable valoró la confesional ficta ofrecida por la actora?"
PROBLEMAS = [{"pregunta": P1, "jerarquia": "principal", "combate": "no hubo cercioramiento",
              "resolvio": "tuvo por legal el emplazamiento"},
             {"pregunta": P2, "jerarquia": "accesorio", "combate": "las costas no proceden",
              "resolvio": "condenó en costas"},
             {"pregunta": P3, "jerarquia": "accesorio", "combate": "no se valoró la confesional",
              "resolvio": "valoró la prueba"}]


def _r():
    fases = types.SimpleNamespace(
        problemas=[dict(p) for p in PROBLEMAS], problema_global="", avisos=[], fuentes=["", ""],
        parrafos_acto=lambda: ["La Sala tuvo por legal el emplazamiento."],
        parrafos_conceptos=lambda: ["El quejoso dice que no hubo cercioramiento."])
    enc = types.SimpleNamespace(tipo_asunto="amparo_directo", es_recurso=False, tribunal="",
                                ciudad="", numero="123/2026", materia="civil",
                                coleccion_estatal="")
    return types.SimpleNamespace(fases=fases, encargo=enc)


def _material(espejo=None):
    return types.SimpleNamespace(tipo_asunto="amparo_directo", sondeo=None, tesis=[], normas=[],
                                 espejo=espejo or [], decisiva=None)


def _propuesta_del_motor(p2=True):
    """El motor concede (fundado el principal), escribe la vía contraria y la
    suerte de los accesorios en las dos vías; deja el tercero sin propuesta."""
    props = [f5.Propuesta(problema=P1, sentido="fundado", razon="no hubo cercioramiento",
                          apoyos=["2001111"], confianza="alta")]
    if p2:
        props.append(f5.Propuesta(problema=P2, sentido="innecesario", razon="queda sin materia",
                                  confianza="media"))
    g = f5.Global(sentido="fundado", razon="el emplazamiento es ilegal y se concede",
                  problema_que_decide=P1, efecto="los demás quedan sin materia",
                  apoyos=["2001111"], confianza="alta", en_contra="la diligencia sí se cercioró",
                  alternativa={"sentido": "infundado", "razon": "el actuario se cercioró del domicilio",
                               "efecto": "las costas y la confesional se estudian",
                               "apoyos": ["2002222"]},
                  checklist=[{"numero": 2, "tema": "costas", "papel": "accesorio",
                              "con_propuesta": "innecesario", "con_alternativa": "infundado",
                              "relacion": "depende",
                              "si_prospera": {"sentido": "innecesario", "razon": "sin materia"},
                              "si_no_prospera": {"sentido": "infundado",
                                                 "razon": "la condena sigue al resultado"}},
                             {"numero": 3, "tema": "confesional", "papel": "accesorio",
                              "con_propuesta": "innecesario", "con_alternativa": "inoperante",
                              "relacion": "depende",
                              "si_prospera": {"sentido": "innecesario", "razon": "sin materia"},
                              "si_no_prospera": {"sentido": "inoperante",
                                                 "razon": "no combate la valoración"}}])
    return props, g


_RAZONES = []


async def _razonar(cliente, problema, sentido, material, *a, **k):
    _RAZONES.append((problema, sentido))
    return f"Razón escrita para «{sentido}»."


f5.razonar = _razonar


def correr(props_glob, espejo=None, banderas=True):
    async def _proponer(*a, **k):
        p, g = props_glob
        return p, g, []
    f5.proponer = _proponer
    if banderas:
        os.environ["PROPUESTA_POR_PROBABILIDAD"] = "todos"
    else:
        os.environ.pop("PROPUESTA_POR_PROBABILIDAD", None)
    ses = {"resultado": _r(), "material": _material(espejo)}
    try:
        return asyncio.run(main._taller_proponer_nucleo("prueba@iurexia.com", "123/2026", ses, "")), ses
    finally:
        os.environ.pop("PROPUESTA_POR_PROBABILIDAD", None)


print("\n1 · APAGADA: LA RESPUESTA DE AYER")
resp, _ = correr(_propuesta_del_motor(), banderas=False)
ok(resp["formato"] == 2 and "estado" not in resp and "probabilidad" not in resp["global"],
   "formato 2, sin estado ni probabilidad")
ok(resp["global"]["sentido"] == "fundado" and resp["propuestas"][0]["sentido"] == "fundado",
   "el sentido del motor, tal cual")
ok(resp["propuestas"][2]["alcanza"] is False and resp["propuestas"][2]["sentido"] == ""
   and "sostenida" not in resp["propuestas"][0],
   "el problema sin propuesta queda en blanco, como ayer")
ok(len(json.loads(resp["criterios_json"])) == 2, "criterios_json sólo con los que alcanzan")

print("\n2 · ENCENDIDA: SE VOLTEA CON LA VÍA QUE EL MOTOR ESCRIBIÓ")
_RAZONES.clear()
resp, ses = correr(_propuesta_del_motor())
g = resp["global"]
ok(resp["formato"] == 3 and resp["estado"] == "lista" and resp["preguntas"] == [],
   "formato 3, estado «lista», preguntas (vacías)")
pr = g.get("probabilidad") or {}
ok(pr.get("lado") == "no_prospera" and pr.get("volteada") is True and pr.get("fuente") == "jurimetria"
   and pr.get("sentido_motor") == "fundado" and pr.get("p_prospera") == 0.324,
   f"la probabilidad: {pr.get('p_prospera')} → {pr.get('lado')}, volteada")
ok(g["sentido"] == "infundado" and g["razon"] == "el actuario se cercioró del domicilio"
   and g["alternativa"]["sentido"] == "fundado" and g["en_contra"].startswith("el emplazamiento es ilegal"),
   "la global es la vía contraria del motor y lo del motor queda como alternativa")
ok(g["alcanza"] is True and g["sostenida"] is True and g["confianza"] == "media"
   and g["confianza_motor"] == "alta", "alcanza, sostenida, confianza de la probabilidad")
ok(_RAZONES == [], "no hizo falta pedir ninguna razón: la vía contraria ya estaba escrita")
p = resp["propuestas"]
ok(p[0]["sentido"] == "infundado" and p[0]["origen"] == "probabilidad"
   and p[0]["razon"] == "el actuario se cercioró del domicilio",
   "el principal va con el lado nuevo y su razón")
ok(p[1]["sentido"] == "infundado", f"costas: el árbol le da su suerte en la vía nueva ({p[1]['sentido']})")
ok(p[2]["sentido"] == "inoperante" and p[2]["origen"] == "arbol" and p[2]["alcanza"] is True,
   f"la confesional, sin propuesta del motor, la rellena el árbol ({p[2]['sentido']}, {p[2]['origen']})")
ok(all(x["alcanza"] == bool(x["sentido"]) for x in p), "alcanza = hay sentido")
_cj = json.loads(resp["criterios_json"])
ok(len(_cj) == 3 and [c["sentido"] for c in _cj] == ["infundado", "infundado", "inoperante"],
   "criterios_json con TODOS los problemas con sentido")
ok(any("SE VOLTEÓ" in a for a in resp["avisos"]), "se avisa el volteo")
ok(not getattr(ses["propuestas"][0], "sentido_propio", ""),
   "el principal no guarda lo del motor en sentido_propio (el árbol lo leería como «su vía»)")
ok(g["checklist"][0]["con_propuesta"] == "infundado", "la lista de comprobación, intercambiada")

print("\n3 · CON PRECEDENTES DEL PROPIO TRIBUNAL QUE PROSPERARON, GANA EL MOTOR")
_esp = [{"problema": P1, "filas": [
    {"similitud": 95, "calificacion": "fundado", "neun": "a", "expediente": "10/2024", "nivel": "mismo_problema"},
    {"similitud": 91, "calificacion": "fundado", "neun": "b", "expediente": "11/2024", "nivel": "mismo_problema"}]}]
resp, _ = correr(_propuesta_del_motor(), espejo=_esp)
pr = resp["global"]["probabilidad"]
ok(pr["lado"] == "prospera" and pr["volteada"] is False and pr["precedentes"]["n"] == 2
   and resp["global"]["sentido"] == "fundado",
   f"dos precedentes del mismo problema: p={pr['p_prospera']}, se queda la del motor")
ok(resp["propuestas"][1]["sentido"] == "innecesario" and resp["propuestas"][2]["sentido"] == "innecesario",
   "y los accesorios quedan sin materia (la suerte del principal que prospera)")

print("\n4 · SIN VÍA CONTRARIA ESCRITA: UNA LLAMADA PARA LA RAZÓN")
_RAZONES.clear()
pg = _propuesta_del_motor()
pg[1].alternativa = {"sentido": "", "razon": "", "efecto": "", "apoyos": []}
resp, _ = correr(pg)
ok(_RAZONES == [(P1, "infundado")] and resp["global"]["razon"] == "Razón escrita para «infundado»."
   and resp["propuestas"][0]["razon"] == "Razón escrita para «infundado».",
   "se pide UNA razón para el lado nuevo y la llevan la global y el principal")


async def _razonar_falla(*a, **k):
    raise RuntimeError("sin red")


f5.razonar = _razonar_falla
pg = _propuesta_del_motor()
pg[1].alternativa = {}
resp, _ = correr(pg)
ok(resp["global"]["sentido"] == "infundado" and "está por escribir" in resp["global"]["razon"]
   and any("no trae razón escrita" in a for a in resp["avisos"]),
   "si la razón no llega, se dice de dónde salió el sentido y que la razón está por escribir")
f5.razonar = _razonar

print("\n5 · EL ÚLTIMO PROBLEMA YA NO SE CAE EN SILENCIO (MODO ACERVO)")
import modos_decision as md
os.environ["PROPUESTA_POR_PROBABILIDAD"] = "todos"
_rep, _av = md.repartir(PROBLEMAS, md.ACERVO, "", [{"problema": P1, "sentido": "infundado"}], {})
ok(_rep[2]["sentido"] == "" and any(P3[:40] in a for a in _av),
   "el problema sin sentido no se inventa, pero se dice")
os.environ.pop("PROPUESTA_POR_PROBABILIDAD")
_rep, _av = md.repartir(PROBLEMAS, md.ACERVO, "", [{"problema": P1, "sentido": "infundado"}], {})
ok(not _av, "apagada, como ayer: sin aviso")
_src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "main.py"), encoding="utf-8").read()
_ac = _src.split('elif _modo == "acervo" or usar_propuesta:', 1)[1].split("    else:\n        if not sentido:", 1)[0]
ok("prediccion=(_pred_a.get(" in _ac and "or _rellenar_a]" in _ac,
   "el modo acervo pasa la predicción y deja entrar los vacíos al árbol (con la bandera)")
_nu = _src.split("async def _taller_proponer_nucleo", 1)[1].split("\n@app.", 1)[0]
ok(_nu.index("_taller_probabilidad_sentido(") < _nu.index("_dz.reconciliar(") < _nu.index("_ad.aplicar("),
   "en el núcleo: probabilidad, después reconciliar, después el árbol")

print()
if FALLOS:
    print(f"FALLAN {len(FALLOS)}: " + " · ".join(FALLOS))
    raise SystemExit(1)
print("RESULTADO: TODAS LAS COMPROBACIONES PASAN")
