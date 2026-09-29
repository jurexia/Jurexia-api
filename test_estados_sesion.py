"""Los estados de la sesión — rediseño, etapa 4, pieza 6 (29-sep-2026).

Cada marca de la fila guardaba sólo la huella del adelanto y había marcas que
leían más de lo que esa huella cubre: el contraste con la ficha, el análisis
con la ficha y los segmentos, el plan con la ficha y el contraste. El caso del
AR 631/2025: el secretario corrige quién recurre, el adelanto no cambia y todo
lo hecho con la ficha equivocada seguía pasando por bueno. `taller_estado`
gana dos piezas puras —`manifiesto()` (lo que cada marca «consume») y
`estado_sesion()` (lo que eso dice hoy)— que se cablearán detrás de la bandera
«estados_sesion». Aquí se comprueba, sobre filas de mentira:

  · el grafo: sin ciclos, cada insumo declarado, y cada cambio de insumo
    invalida EXACTAMENTE lo que dice el manifiesto (con su arrastre);
  · el commit por sí solo no invalida nada;
  · una fila sin «consume» (la de hoy) da estado vacío: nada que avisar;
  · sin reloj no envejece nada;
  · borrador_sin_plan se deriva de la ficha del plan, sin escritura nueva;
  · y que lo que ya existía en taller_estado no cambió ni un byte de salida.

    .venv/bin/python test_estados_sesion.py
"""
import ast
import copy
import json
import re
import types

import contexto_taller as ctx
import taller_estado as te

FALLOS = []


def ok(cond, que):
    print(f"   {'PASA ' if cond else 'FALLA'}  {que}")
    if not cond:
        FALLOS.append(que)


def desactualizadas(r):
    return {m for m, v in r["por_marca"].items() if v["estado"] == "desactualizada"}


# ═══ 1 · EL GRAFO ════════════════════════════════════════════════════════════
print("\n1 · EL GRAFO: SIN CICLOS, TODO DECLARADO, EL ARRASTRE EXPLÍCITO")
orden = te.orden()
ok(sorted(orden) == sorted(te.MARCAS), "el orden topológico tiene todas las marcas (no hay ciclos)")
ok(all(orden.index(d) < orden.index(m) for m, s in te.MARCAS.items() for d in s["depende_de"]),
   "cada marca va después de las que la arrastran")
ok(all(k in te.INSUMOS for s in te.MARCAS.values() for k in s["consume"]),
   "cada insumo consumido está declarado en INSUMOS")
ok(all(d in te.MARCAS for s in te.MARCAS.values() for d in s["depende_de"]),
   "cada dependencia es una marca que existe")
ok(all(any(k in s["consume"] for s in te.MARCAS.values()) for k in te.INSUMOS),
   "ningún insumo declarado queda sin quien lo consuma")
ok(set(te.INSUMOS) == set(te._HUELLA) | {"material.registros", "material.normas", "material.espejo"},
   "cada insumo tiene su regla de huella")
ok(te.invalida_por("adelanto") == orden, "rehacer el adelanto lo invalida TODO")
ok(te.invalida_por("commit") == [] and "commit" in te.MARCAS["proyecto"]["consume"],
   "el commit se registra en el proyecto y no invalida nada")
# La cadena que pidió el cruce: adelanto → análisis/consulta → propuesta →
# deliberación; propuesta → plan → proyecto.
ok("propuesta" in te.dependientes("analisis") and "propuesta" in te.dependientes("consulta"),
   "análisis y consulta arrastran la propuesta")
ok("deliberacion" in te.dependientes("propuesta") and "plan" in te.dependientes("propuesta"),
   "la propuesta arrastra la deliberación y el plan")
ok(te.dependientes("plan") == ["proyecto"], "el plan arrastra sólo el proyecto")
ok("plan" not in te.dependientes("deliberacion") and te.invalida_por("material.espejo") == ["deliberacion"],
   "la deliberación no la lee el plan: recalibrar la tabla OAJ sólo la toca a ella")
ok(te.invalida_por("inventario") == ["plan", "proyecto"],
   "la lectura del escrito no invalida el análisis (llega siempre después)")
ok("inventario" not in te.invalida_por("ficha") and "decisiva" in te.invalida_por("ficha"),
   "la ficha invalida la decisiva, el contraste y el análisis, no la lectura")
ok("criterio" not in te.MARCAS["recalificacion"]["consume"] and te.invalida_por("criterio") == ["plan", "proyecto"],
   "la recalificación no consume el criterio que ella misma produce")
ok("material.registros" not in te.MARCAS["requisitos"]["consume"],
   "los requisitos no consumen el material que ellos mismos amplían")
ok(te.arrastran_a("proyecto")[-1] == "plan" and "decisiva" in te.arrastran_a("proyecto"),
   "lo que puede arrastrar al proyecto llega hasta la decisiva")
g = te.grafo()
ok(json.loads(json.dumps(g)) == g and g["version"] == te.MANIFIESTO_VERSION and g["no_invalida"] == ["commit"],
   "el grafo se sirve como JSON, con su versión")

# ═══ 2 · EL MANIFIESTO ═══════════════════════════════════════════════════════
print("\n2 · EL MANIFIESTO: LA HUELLA DE CADA INSUMO")
m0 = te.manifiesto({})
ok(m0 == {"v": te.MANIFIESTO_VERSION}, "sin entradas, sólo la versión: nada se afirma")
ok(te.manifiesto({"analisis": None}) == {"v": te.MANIFIESTO_VERSION, "analisis": ""}
   and "analisis" not in te.manifiesto({"contraste": []}),
   "vacío es «» (se sabe que faltaba); ausente es no decir nada")
ok(te.manifiesto({"analisis": {"r": [1], "segundos": 3.2, "uso": {"coste_usd": 0.1}}})
   == te.manifiesto({"analisis": {"r": [1], "segundos": 40, "uso": {"coste_usd": 0.9}}}),
   "tiempos y costes no entran: lo mismo leído otra vez no «cambia»")
ok(te.manifiesto({"analisis": {"hecho": "el quejoso adquirió el inmueble"}})
   != te.manifiesto({"analisis": {"hecho": "el quejoso vendió el inmueble"}}),
   "«hecho» con texto es un hecho del asunto y sí cuenta")
ok(te.manifiesto({"adelanto": "ae83104d805f34c1"})["adelanto"] == "ae83104d805f34c1",
   "la huella del adelanto se copia tal cual (es ya una identidad)")
ok(te.manifiesto({"ficha": "Quejosa:  Ana\n Recurrente: Luis"})["ficha"]
   == te.manifiesto({"ficha": "Quejosa: Ana Recurrente: Luis"})["ficha"] != "",
   "la ficha es su texto, sin importar los espacios")
ok(te.manifiesto({"ficha": "Recurrente: Luis"})["ficha"] != te.manifiesto({"ficha": "Recurrente: Ana"})["ficha"],
   "corregir quién recurre cambia la ficha (el caso del AR 631/2025)")
ok(te.manifiesto({"contexto": "  "})["contexto"] == "", "sin contexto escrito, «»")

import fase6_estudio as f6
_c1 = [f6.Criterio("¿Procede?", "Infundado", "porque sí", "principal", "", {"porcentaje": 60})]
_c2 = [{"problema": "¿Procede?", "sentido": "infundado", "razonamiento": "porque  sí",
        "jerarquia": "PRINCIPAL", "grupo": "", "tocado": True}]
ok(te.manifiesto({"criterio": _c1})["criterio"] == te.manifiesto({"criterio": _c2})["criterio"],
   "el criterio es el mismo venga como Criterio o como el dict de la pantalla")
ok(te.manifiesto({"criterio": _c1})["criterio"]
   != te.manifiesto({"criterio": [dict(_c2[0], razonamiento="porque no")]})["criterio"],
   "otra razón del secretario es otro criterio")
ok(te.manifiesto({"criterio": _c1})["criterio"]
   == te.manifiesto({"criterio": [f6.Criterio("¿Procede?", "infundado", "porque sí", "principal", "", {})]})["criterio"],
   "la predicción del acervo no es criterio del secretario")

DEC = {"formulada": True, "pregunta_decisiva": "¿Puede el tercero adquirente sustituirse en la ejecución?",
       "figura": "sustitución procesal", "hechos_que_deciden": ["adquirió el inmueble litigioso"],
       "uso": {"coste_usd": 0.02}, "segundos": 11.0, "version": None}
ok(te.manifiesto({"decisiva": DEC})["decisiva"] == te.manifiesto({"decisiva": dict(DEC, segundos=99)})["decisiva"] != "",
   "la decisiva es su cuestión, su figura y sus hechos")
ok(te.manifiesto({"decisiva": dict(DEC, formulada=False)})["decisiva"] == "",
   "una decisiva no formulada es como no tenerla")

_segs = [{"id": "C1.a", "concepto": 1, "anclas": ["b", "a"]}, {"id": "C1.e", "concepto": 1, "anclas": []}]
ok(te.manifiesto({"inventario": _segs})["inventario"]
   == te.manifiesto({"inventario": [dict(_segs[0], anclas=["a", "b"]), _segs[1]]})["inventario"],
   "el inventario no depende del orden de las anclas")
ok(te.manifiesto({"inventario": _segs})["inventario"] != te.manifiesto({"inventario": _segs[::-1]})["inventario"],
   "pero sí del orden de los segmentos (es el de la exposición)")

MAT = {"tesis": [{"registro": "2020001", "rubro": "A"}, {"registro": "2019999", "cupo_figura": True}],
       "normas": [{"cuerpo_legal": "Ley de Amparo", "articulo": "17"}],
       "espejo": [{"problema": "¿Procede?", "filas": [{"neun": 38118729, "nivel": "mismo_problema",
                                                       "similitud": 86, "cota_inferior": False}]}],
       "consulta_estado": {"estado": "completa", "faltan": []}, "requisitos": {}, "completo": True}
_obj = types.SimpleNamespace(**{k: copy.deepcopy(v) for k, v in MAT.items() if k != "completo"})
hm = te.huella_material(MAT)
ok(hm == te.huella_material(_obj) and set(hm) == {"material.registros", "material.normas"},
   "el Material en memoria y su dict de la fila dan la misma huella")
_nueve = types.SimpleNamespace(tesis=MAT["tesis"], normas=MAT["normas"], espejo=[
    {"problema": f"¿P{i}?", "filas": [{"neun": 1000 + i, "nivel": "posible", "similitud": 57}]} for i in range(9)])
ok(te.huella_material(_nueve, "t28") == te.huella_material(te.material_ligero(_nueve), "t28")
   != te.huella_material(dict(vars(_nueve)), "t28"),
   "se mira el material COMO VA A LA FILA: nueve grupos de espejo en memoria son los ocho guardados")
ok(te.huella_material(dict(MAT, tesis=MAT["tesis"][::-1])) == hm,
   "reordenar las tesis (el rerank) no cambia el material")
_mas = te.huella_material(dict(MAT, normas=MAT["normas"] + [{"cuerpo_legal": "CFPC", "articulo": "1"}]))
ok(_mas["material.normas"] != hm["material.normas"] and _mas["material.registros"] == hm["material.registros"],
   "una norma más cambia las normas y no los registros")
ok("material.espejo" not in te.manifiesto({"material": MAT}),
   "sin la versión de la tabla OAJ el espejo no se afirma")
e28 = te.manifiesto({"material": MAT, "version_oaj": "t28"})
e29 = te.manifiesto({"material": MAT, "version_oaj": "t29"})
ok(e28["material.espejo"] != e29["material.espejo"] and e28["material.registros"] == e29["material.registros"],
   "recalibrar la tabla cambia sólo el espejo")
ok(te.huella_material(None, "t28") == {"material.registros": "", "material.normas": "", "material.espejo": ""},
   "sin material, todo «»")
ok(te.version_tabla_oaj({"planteamiento": {"x": 1}}) == te.version_tabla_oaj({"planteamiento": {"x": 1}}) != ""
   and te.version_tabla_oaj({}) == "" and te.version_tabla_oaj(None) == "",
   "la versión de la tabla es su contenido; sin tabla, «»")
_todo = {"adelanto": "a1", "ficha": "F", "contexto": "", "decisiva": DEC, "material": MAT, "commit": "abc"}
ok(set(te.manifiesto(_todo, "contraste")) == {"v", "adelanto", "ficha"},
   "con marca, sólo lo que esa marca consume")
ok(te.consume_de("proyecto", te.manifiesto(_todo)) == te.manifiesto(_todo, "proyecto"),
   "recortar las huellas ya hechas da lo mismo que el manifiesto de la marca")
ok(te.consume_de("no_existe", te.manifiesto(_todo)) == {}, "una marca desconocida no inventa consumo")
ok(list(te.manifiesto(_todo)) == ["v"] + [k for k in te.INSUMOS if k in te.manifiesto(_todo)],
   "las claves salen en el orden del manifiesto (JSON estable)")

# ═══ 3 · CADA CAMBIO INVALIDA EXACTAMENTE LO QUE DICE EL MANIFIESTO ══════════
print("\n3 · CADA CAMBIO DE INSUMO INVALIDA EXACTAMENTE LO QUE DICE EL MANIFIESTO")
BASE = {"v": te.MANIFIESTO_VERSION, **{k: f"h-{k}" for k in te.INSUMOS}}


def fila_completa(base=BASE):
    """Todas las marcas listas, cada una con su consume de `base`."""
    def c(m):
        return te.consume_de(m, base)
    hu = base["adelanto"]
    est = {
        "huella": hu,
        "consulta": {"huella": hu, "estado": "listo", "segundos": 30.0, "consume": c("consulta")},
        "material": dict(copy.deepcopy(MAT), requisitos={
            "huella": f"{hu}|a1|d1", "requisitos": [{"id": "R1"}], "consume": c("requisitos")}),
        "decisiva": {"huella": f"{hu}:abc", "estado": "listo", "doc": DEC, "consume": c("decisiva")},
        "contraste": {"huella": hu, "estado": "listo", "items": [{"p": 1}], "consume": c("contraste")},
        "analisis": {"huella": hu, "huella_analisis": "x", "estado": "listo",
                     "doc": {"razones": [], "faltantes": []}, "consume": c("analisis")},
        "propuesta": {"huella": hu, "estado": "listo",
                      "respuesta": {"propuestas": [], "global": {"sentido": "niega"}}, "consume": c("propuesta")},
        "global_propuesta": {"huella": hu, "global": {"sentido": "niega"}, "desde": 1.0,
                             "consume": c("global_propuesta")},
        "deliberacion": {"huella": hu, "clave": "k", "estado": "listo", "deliberacion": {},
                         "consume": c("deliberacion")},
        "proyecto": {"version": 1, "palabras": 4000, "estado_salida": "",
                     "plan": {"estado": "usado", "clave": base["plan"], "segmentos": []},
                     "consume": c("proyecto")},
    }
    plan = {"rev": 3, "huella": hu, "pedido_clave": base["plan"], "corridas": 1,
            "planes": {base["plan"]: {"estado": "listo", "plan": {"segmentos": []}, "consume": c("plan")}},
            "inventario": {"huella": "li", "estado": "listo", "argumentos": [], "consume": c("inventario")},
            "recalificaciones": {base["recalificacion"]: {"estado": "listo", "resultados": {"2": "x"},
                                                          "consume": c("recalificacion")}}}
    return {"estado": est, "plan": plan}


F = fila_completa()
r0 = te.estado_sesion(F, BASE)
ok(r0["estado"] == "proyecto_verificado" and not r0["desactualizado"] and not desactualizadas(r0),
   "todo al día: proyecto verificado, nada desactualizado")
ok(all(v["estado"] == "vigente" for v in r0["por_marca"].values()), "las doce marcas vigentes")
for x in te.INSUMOS:
    act = dict(BASE, **{x: f"otra-{x}"})
    r = te.estado_sesion(F, act)
    esperado = set(te.invalida_por(x))
    ok(desactualizadas(r) == esperado and r["desactualizado"] == ("proyecto" in esperado),
       f"cambia «{x}»: caen {len(esperado)} marca(s), las del manifiesto")
    directas = {mo["marca"] for v in r["por_marca"].values() for mo in v["motivos"]
                if mo.get("insumo") == x and mo["tipo"] == "cambio"}
    ok(directas == {m for m in esperado if x in te.MARCAS[m]["consume"]}
       and all(any(mo["tipo"] == "arrastre" for mo in r["por_marca"][m]["motivos"]) for m in esperado - directas),
       f"   · «{x}»: la causa en quien lo consume, el arrastre en las demás")
    if x != "commit":
        ok(r["estado"] == "proyecto_verificado", f"   · «{x}»: el estado no se mueve (desactualizado es aparte)")
r_c = te.estado_sesion(F, dict(BASE, commit="otro-commit"))
ok(not desactualizadas(r_c) and r_c["motivos"] == [] and not r_c["desactualizado"],
   "el commit por sí solo NO invalida ni se reporta como motivo")
r_ad = te.estado_sesion(F, dict(BASE, adelanto="nuevo"))
ok(r_ad["motivos"][0] == {"marca": "inventario", "insumo": "adelanto", "tipo": "cambio",
                          "antes": "h-adelanto", "ahora": "nuevo", "invalida": True},
   "los motivos empiezan por la causa, en orden topológico")
ok(all(mo["marca"] in set(te.arrastran_a("proyecto")) | {"proyecto"} for mo in r_ad["motivos"]),
   "los motivos del proyecto sólo nombran lo que puede arrastrarlo (no la deliberación)")
ok(te.estado_sesion(fila_completa(), BASE) == r0, "determinista")
# Un proyecto escrito SIN plan (consumió plan «») no se cuelga de la casilla
# que otro criterio pidió después, aunque ésa esté desactualizada.
FX = fila_completa(dict(BASE, plan=""))
FX["plan"]["pedido_clave"] = "k-otro"
FX["plan"]["planes"] = {"k-otro": {"estado": "listo", "plan": {},
                                   "consume": te.consume_de("plan", dict(BASE, adelanto="viejo"))}}
r = te.estado_sesion(FX, {k: v for k, v in BASE.items() if k != "plan"})
ok(r["por_marca"]["plan"]["estado"] == "ausente" and not r["desactualizado"],
   "un proyecto sin plan no hereda la caída de la casilla de otro criterio")

# ═══ 4 · LLEGADA, RETIRADA Y LA DECISIÓN PENDIENTE ═══════════════════════════
print("\n4 · LLEGADA Y RETIRADA")
b_sin = dict(BASE, contraste="")
F2 = fila_completa(b_sin)
r = te.estado_sesion(F2, BASE)
mo = [m for m in r["por_marca"]["propuesta"]["motivos"] if m.get("insumo") == "contraste"]
ok(mo and mo[0]["tipo"] == "llegada" and r["por_marca"]["propuesta"]["estado"] == "desactualizada",
   "la propuesta hecha sin contraste queda desactualizada cuando el contraste llega")
r = te.estado_sesion(F2, BASE, llegada_desactualiza=False)
ok(r["por_marca"]["propuesta"]["estado"] == "vigente"
   and any(m.get("tipo") == "llegada" and m["invalida"] is False for m in r["por_marca"]["propuesta"]["motivos"]),
   "con llegada_desactualiza=False, se avisa y no se invalida (decisión de David)")
r = te.estado_sesion(F, dict(BASE, analisis=""))
ok(any(m.get("tipo") == "retirada" for m in r["por_marca"]["propuesta"]["motivos"])
   and "propuesta" in desactualizadas(r), "consumió un análisis que ya no está: retirada")

# ═══ 5 · LA FILA SIN «CONSUME»: NADA QUE AVISAR ══════════════════════════════
print("\n5 · LA FILA DE HOY (SIN «CONSUME») DA ESTADO VACÍO")
HOY = {"estado": {
    "huella": "h1", "fases": {"problemas": [{"pregunta": "¿Procede?"}]}, "avisos": [],
    "consulta": {"huella": "h1", "estado": "listo", "segundos": 31.2},
    "material": copy.deepcopy(MAT),
    "decisiva": {"huella": "h1:ab12", "estado": "listo", "doc": DEC},
    "contraste": {"huella": "h1", "estado": "listo", "items": [{"p": 1}]},
    "analisis": {"huella": "h1", "huella_analisis": "x", "estado": "listo", "doc": {"faltantes": []}},
    "propuesta": {"huella": "h1", "estado": "listo", "respuesta": {"propuestas": []}},
    "global_propuesta": {"huella": "h1", "global": {"sentido": "niega"}, "desde": 1.0},
    "deliberacion": {"huella": "h1", "clave": "k", "estado": "en_curso", "desde": 1.0},
    "proyecto": {"version": 2, "plan": {"estado": "sin_plan", "clave": "", "avisos": []},
                 "fuentes_tardias": [{"fuente": "registro 1", "sustantiva": True, "unidades": ["U1"]}],
                 "estado_salida": "justificacion_pendiente"},
    "proyectos": []},
    "plan": {"huella": "h1", "pedido_clave": "k1", "planes": {"k1": {"estado": "listo", "plan": {}}}}}
VACIA = {"version": te.MANIFIESTO_VERSION, "estado": "", "desactualizado": False, "motivos": [],
         "por_marca": {}, "pendientes": [], "faltan": [], "en_curso": []}
ok(te.estado_sesion(HOY) == VACIA, "una fila como las de hoy: estado «», nada desactualizado, ningún motivo")
ok(te.estado_sesion(HOY, dict(BASE), ahora=10 ** 10) == VACIA,
   "tampoco con actuales ni reloj: sin «consume» no se afirma nada")
ok(te.estado_sesion(HOY["estado"], plan=HOY["plan"]) == VACIA, "igual con el estado suelto")
ok(te.estado_sesion({}) == VACIA and te.estado_sesion(None) == VACIA and te.estado_sesion("x") == VACIA,
   "una fila vacía o rara no lanza")
_rara = fila_completa()
_rara["estado"]["material"].update(tesis=True, normas=1.5, espejo="x", consulta_estado={"estado": "provisional",
                                                                                         "faltan": 3})
_rara["estado"]["propuesta"]["respuesta"] = 1.5
_rara["estado"]["proyecto"].update(fuentes_tardias=[7, {"sustantiva": True, "unidades": 3}],
                                   cambios_sin_justificar=True, plan={"estado": "usado", "segmentos": "x"})
try:
    _r_rara = te.estado_sesion(_rara, BASE, version_oaj="t")
    _lanza = False
except Exception:
    _lanza = True
ok(not _lanza and _r_rara["estado"] == "justificacion_pendiente",
   "listas que no son listas (un true, un número) no tumban la insignia")
_viejo = fila_completa()
del _viejo["estado"]["proyecto"]["consume"]
r = te.estado_sesion(_viejo, dict(BASE, adelanto="otro"))
ok(r["estado"] == "" and not r["desactualizado"]
   and r["motivos"] == [{"marca": "proyecto", "tipo": "anterior_a_estados", "invalida": False}],
   "un proyecto anterior a los estados: ni verificado ni desactualizado, y se dice por qué")
_otra = fila_completa()
_otra["estado"]["contraste"]["consume"] = dict(_otra["estado"]["contraste"]["consume"], v="estados-0")
r = te.estado_sesion(_otra, dict(BASE, ficha="otra"))
ok(r["por_marca"]["contraste"]["estado"] == "sin_registro"
   and r["por_marca"]["contraste"]["motivos"][0]["tipo"] == "otra_version",
   "un consume de otra versión del manifiesto no se compara")

# ═══ 6 · SIN RELOJ NO ENVEJECE NADA ══════════════════════════════════════════
print("\n6 · SIN RELOJ NO ENVEJECE NADA")
FC = fila_completa()
del FC["estado"]["proyecto"]
FC["estado"]["consulta"] = {"huella": "h-adelanto", "estado": "en_curso", "desde": 1000.0, "latido": 1000.0,
                            "consume": te.consume_de("consulta", BASE)}
FC["estado"]["material"]["consulta_estado"] = {"estado": "provisional", "faltan": ["decisiva"]}
r = te.estado_sesion(FC, BASE)
ok(r["por_marca"]["consulta"]["estado"] == "en_curso" and r["estado"] == "en_analisis"
   and r["en_curso"] == ["consulta"],
   "sin `ahora`, un latido de hace días sigue «en curso»: en análisis")
r = te.estado_sesion(FC, BASE, ahora=1000.0 + 200)
ok(r["por_marca"]["consulta"]["estado"] == "fallo" and r["estado"] == "faltan_insumos"
   and r["faltan"] == [{"origen": "consulta", "que": "decisiva"}],
   "con reloj, sin latido en 150 s es un fallo; y la consulta provisional deja «faltan insumos»")
ok(te.estado_sesion(FC, BASE, ahora=lambda: 1100.0)["por_marca"]["consulta"]["estado"] == "en_curso",
   "el reloj también puede ser una función; 100 s sin latido todavía es vida")
FP = fila_completa()
del FP["estado"]["proyecto"]
FP["plan"]["planes"]["h-plan"] = {"estado": "en_curso", "desde": 1000.0, "latido": 1000.0,
                                  "consume": te.consume_de("plan", BASE)}
r = te.estado_sesion(FP, BASE, ahora=1100.0)
ok(r["por_marca"]["plan"]["estado"] == "fallo" and r["por_marca"]["consulta"]["estado"] == "vigente",
   "cada rama con su umbral: el plan se da por muerto a los 60 s, las marcas a los 150")
ok(te.estado_sesion(FP, BASE)["por_marca"]["plan"]["estado"] == "en_curso", "y sin reloj, tampoco el plan")

# ═══ 7 · EL ESTADO DE LA INSIGNIA ════════════════════════════════════════════
print("\n7 · EL ESTADO: BORRADOR > JUSTIFICACIÓN > FALTAN > VERIFICADO")


def con_ficha(**cambios):
    f = fila_completa()
    f["estado"]["proyecto"].update(cambios)
    return f


r = te.estado_sesion(con_ficha(plan={"estado": "sin_plan", "clave": "", "avisos": ["no llegó"]},
                               fuentes_tardias=[{"fuente": "registro 1", "sustantiva": True, "unidades": ["U2"]}]),
                     BASE)
ok(r["estado"] == "borrador_sin_plan", "sin plan (la v3) es borrador, aunque haya otras cosas pendientes")
ok(te.estado_sesion(con_ficha(plan={}), BASE)["estado"] == "proyecto_verificado",
   "una ficha sin plan ({}) es de la v1/v2, que no planean: no es borrador")
_ft = te.estado_sesion(con_ficha(fuentes_tardias=[
    {"fuente": "registro 1", "sustantiva": True, "unidades": ["U2"]},
    {"fuente": "art. 5 — CFPC", "sustantiva": False, "unidades": []}],
    estado_salida="justificacion_pendiente"), BASE)
ok(_ft["estado"] == "justificacion_pendiente"
   and _ft["pendientes"] == [{"tipo": "fuente_tardia", "fuente": "registro 1", "unidades": ["U2"]}],
   "una fuente tardía en una unidad que decide: justificación pendiente (la otra no cuenta)")
ok(te.estado_sesion(con_ficha(fuentes_tardias=[{"fuente": "x", "sustantiva": False}]), BASE)["estado"]
   == "proyecto_verificado", "una fuente tardía que no decide no la pide")
ok(te.estado_sesion(con_ficha(estado_salida="justificacion_pendiente"), BASE)["pendientes"]
   == [{"tipo": "estado_salida", "valor": "justificacion_pendiente"}],
   "el estado de salida del redactor, solo, también la pide")
# La forma que escribe `plan_estudio._cambios` (etapa 4) y copia `para_ficha`.
_cs = [{"segmento": "C1.a", "campo": "etiqueta", "antes": "fundado", "despues": "infundado",
        "regla": "portador", "cuenta": True},
       {"unidad": "U2", "campo": "orden", "antes": 1, "despues": 2, "regla": "organizacion", "cuenta": False},
       {"segmento": "C1.c", "campo": "razon", "antes": "", "despues": "z", "regla": "razon_por_etiqueta",
        "cuenta": True, "aceptado": True}]
r = te.estado_sesion(con_ficha(plan={"estado": "usado", "clave": "h-plan", "cambios_sin_justificar": _cs}), BASE)
ok(r["estado"] == "justificacion_pendiente" and [p.get("segmento") for p in r["pendientes"]] == ["C1.a"],
   "un cambio del código que cuenta y nadie aceptó: pendiente (la organización y lo aceptado, no)")
ok(te.estado_sesion(con_ficha(cambios_sin_justificar=[{"seg": "C9", "campo": "etiqueta"}]), BASE)["estado"]
   == "justificacion_pendiente", "sin «cuenta» se tiene por que cuenta; «seg» del mapa también se lee")
# La forma de `exclusiones.nueva`: estado con_prueba | sin_prueba.
r = te.estado_sesion(con_ficha(plan={"estado": "usado", "clave": "h-plan", "exclusiones": [
    {"id": "E1", "tipo": "suficiencia", "estado": "sin_prueba", "segmento_o_problema": {"id": "2"}},
    {"id": "E2", "tipo": "caida", "estado": "con_prueba"}]}), BASE)
ok([(p.get("id"), p.get("clase")) for p in r["pendientes"]] == [("E1", "suficiencia")],
   "una exclusión sin prueba queda pendiente; la probada no")
ok(te.estado_sesion(con_ficha(exclusiones=[{"id": "E3", "estado": "justificacion_pendiente"}]), BASE)["estado"]
   == "justificacion_pendiente", "el nombre del mapa («justificacion_pendiente») se acepta igual")
r = te.estado_sesion(con_ficha(plan={"estado": "usado", "clave": "h-plan",
                                     "proposiciones": [{"id": "P1", "relacion": "sin_clasificar"},
                                                       {"id": "P2", "relacion": "suficiente"}],
                                     "segmentos": [{"id": "C2.a", "etiqueta": "fundado_insuficiente"},
                                                   {"id": "C3.a", "etiqueta": "fundado_insuficiente", "con_p": ""}]}),
                     BASE)
ok([(p["tipo"], p.get("proposicion") or p.get("seg")) for p in r["pendientes"]]
   == [("sin_clasificar", "P1")],
   "la relación sin clasificar queda pendiente (el «insuficiente sin base» se retiró: no podía saltar con una ficha real)")
FF = fila_completa()
FF["estado"]["analisis"]["doc"]["faltantes"] = [{"que": "la constancia de notificación", "por_que_importa": "x"}]
r = te.estado_sesion(FF, BASE)
ok(r["estado"] == "faltan_insumos" and r["faltan"] == [{"origen": "analisis", "que": "la constancia de notificación"}],
   "con proyecto y sin pendientes, lo que el análisis echa en falta: faltan insumos")
FG = fila_completa()
FG["estado"]["global_propuesta"]["global"]["constancias"] = [
    {"que": "el acuerdo de radicación", "indispensable": True}, {"que": "otra", "indispensable": False}]
ok(te.estado_sesion(FG, BASE)["faltan"] == [{"origen": "constancias", "que": "el acuerdo de radicación"}],
   "de las constancias de la global, sólo las indispensables")
FS = fila_completa()
del FS["estado"]["proyecto"]
ok(te.estado_sesion(FS, BASE)["estado"] == "en_analisis" and te.estado_sesion(FS, BASE)["desactualizado"] is False,
   "sin proyecto, nada en curso y nada faltante: el secretario está decidiendo")
ok(te.estado_sesion(FS, dict(BASE, ficha="otra"))["desactualizado"] is False,
   "sin proyecto no hay «desactualizado» de sesión (cada marca lo dice en por_marca)")

# ═══ 8 · CON LA FILA SOLA (/taller/en-curso, sin rehidratar) ═════════════════
print("\n8 · CON LA FILA SOLA: LO QUE SE DEDUCE Y LO QUE NO")
HU, FICHA = "ae83104d805f34c1", "Quejosa: Ana. Recurrente: Luis, tercero extraño."
RESP = {"propuestas": [{"problema": "¿Procede?", "sentido": "infundado"}], "segundos": 44.0,
        "global": {"sentido": "niega", "checklist": [], "constancias": []}}
CONTR = [{"problema": 1, "razon_toral": "la sustitución"}]
AN = {"razones": [{"id": "R1"}], "faltantes": [], "segundos": 50.0}
REQ = {"huella": f"{HU}|a1|d1", "requisitos": [{"id": "R1", "tesis": ["2021000"]}], "huecos": []}
MAT_FINAL = dict(copy.deepcopy(MAT), requisitos=REQ,
                 tesis=MAT["tesis"] + [{"registro": "2021000", "para_requisito": "R1"}])
CRUDO = {"adelanto": HU, "ficha": FICHA, "decisiva": DEC, "analisis": AN, "contraste": CONTR,
         "requisitos": REQ, "material": MAT_FINAL, "version_oaj": "t28", "propuesta": RESP,
         "global": RESP["global"], "contexto": "", "commit": "c1"}


def real(**cambios_crudo):
    """La fila como la dejarían las escritoras con la bandera: cada una con el
    consume de lo que leyó (el mismo material final para todas, como ocurre:
    los requisitos suman antes de proponer)."""
    cr = dict(CRUDO, **cambios_crudo)

    def c(m):
        return te.manifiesto(cr, m)
    est = {"huella": HU,
           "consulta": {"huella": HU, "estado": "listo", "consume": c("consulta")},
           "material": dict(copy.deepcopy(MAT_FINAL), requisitos=dict(REQ, consume=c("requisitos"))),
           "decisiva": {"huella": f"{HU}:ab", "estado": "listo", "doc": DEC, "consume": c("decisiva")},
           "contraste": {"huella": HU, "estado": "listo", "items": CONTR, "consume": c("contraste")},
           "analisis": {"huella": HU, "huella_analisis": "x", "estado": "listo", "doc": AN, "consume": c("analisis")},
           "propuesta": {"huella": HU, "estado": "listo", "respuesta": RESP, "consume": c("propuesta")},
           "global_propuesta": {"huella": HU, "global": RESP["global"], "consume": c("global_propuesta")},
           "deliberacion": {"huella": HU, "estado": "listo", "deliberacion": {}, "consume": c("deliberacion")},
           "proyecto": {"version": 1, "plan": {}, "consume": c("proyecto")}}
    return {"estado": est, "plan": {}}


FR = real()
r = te.estado_sesion(FR, version_oaj="t28")
ok(r["estado"] == "proyecto_verificado" and not desactualizadas(r),
   "lo escrito y lo leído de la fila casan: nada desactualizado")
ok(r["por_marca"]["requisitos"]["estado"] == "vigente",
   "los requisitos no se invalidan con lo que ellos mismos sumaron al material")
_cur = te.actuales_de_fila(FR, version_oaj="t28")
ok(not ({"ficha", "criterio", "contexto", "inventario", "plan", "commit"} & set(_cur)),
   "de la fila no sale ni la ficha, ni el criterio, ni el contexto, ni la lectura: no se comparan")
ok(te.actuales_de_fila({"estado": {}}) == {"v": te.MANIFIESTO_VERSION}, "sin adelanto no se deduce nada")

P = real()
P["estado"]["huella"] = "otra-huella"             # /taller/problema reescribe la huella y deja las marcas
r = te.estado_sesion(P)
ok(r["desactualizado"] and desactualizadas(r) == {m for m in te.MARCAS if P["estado"].get(m) or m == "requisitos"}
   and {"marca": "proyecto", "insumo": "adelanto", "tipo": "cambio", "antes": HU, "ahora": "otra-huella",
        "invalida": True} in r["motivos"],
   "corregir un planteamiento después de generar: todo lo del adelanto viejo cae, y se dice por qué")

T = real(decisiva=None)                            # la decisiva llegó tarde: la consulta siguió sin ella
r = te.estado_sesion(T)
ok(r["por_marca"]["consulta"]["motivos"][0]["tipo"] == "llegada" and r["desactualizado"]
   and {"consulta", "propuesta", "proyecto"} <= desactualizadas(r) and "contraste" not in desactualizadas(r),
   "la decisiva tardía deja atrás la consulta y lo que cuelga de ella, no el contraste")

r = te.estado_sesion(real(), version_oaj="t29")
ok(desactualizadas(r) == {"deliberacion"} and not r["desactualizado"] and r["estado"] == "proyecto_verificado",
   "recalibrar la tabla de la OAJ: cae la deliberación, el proyecto sigue verificado")
ok(not desactualizadas(te.estado_sesion(real())), "sin decir la versión de la tabla, el espejo no se compara")

A = real()
r = te.estado_sesion(A, {"ficha": te.manifiesto({"ficha": "Quejosa: Ana. Recurrente: Ana."})["ficha"]})
ok(desactualizadas(r) == {"decisiva", "contraste", "analisis", "consulta", "requisitos", "propuesta",
                          "global_propuesta", "deliberacion", "proyecto"} and r["desactualizado"],
   "AR 631/2025: la ficha corregida deja atrás todo lo que la leyó (lo que main sabe manda)")

# ═══ 9 · PUREZA Y PARIDAD ════════════════════════════════════════════════════
print("\n9 · PURA, SIN BANDERA DENTRO, Y LO DE ANTES INTACTO")
_antes = copy.deepcopy(F)
te.estado_sesion(F, dict(BASE, ficha="z"), ahora=10 ** 10)
te.actuales_de_fila(F, version_oaj="t")
te.manifiesto({"material": F["estado"]["material"], "version_oaj": "t"})
ok(F == _antes, "no toca la fila que recibe (nunca se guarda desde un GET)")
try:
    ctx.poner(True, {}, pruebas=True)
    en_prueba, r_on = ctx.rediseno("estados_sesion"), te.estado_sesion(F, dict(BASE, ficha="z"))
    ctx.poner(False, {})
    fuera_, r_off = ctx.rediseno("estados_sesion"), te.estado_sesion(F, dict(BASE, ficha="z"))
finally:
    ctx.poner(False, {})
ok(en_prueba is True and fuera_ is False, "la bandera «estados_sesion» existe: prueba sí, los de fuera no")
ok(r_on == r_off, "las funciones no la leen: la aplica quien las llame (main), no ellas")

_src = open("taller_estado.py", encoding="utf-8").read()
_imps = set()
for n in ast.walk(ast.parse(_src)):
    if isinstance(n, ast.Import):
        _imps |= {a.name.split(".")[0] for a in n.names}
    elif isinstance(n, ast.ImportFrom) and n.module:
        _imps.add(n.module.split(".")[0])
ok(not (_imps & {"plan_estudio", "pregunta_decisiva", "contexto_taller", "main", "analisis_litis",
                 "arbol_decision", "fase_oaj"}),
   "no importa módulos que lean banderas o el entorno (sólo compara textos)")


class _Fases:
    problemas = [{"pregunta": "¿Procede la sustitución procesal?", "jerarquia": "principal"},
                 "¿Hubo emplazamiento?"]
    problema_global = ""

    def parrafos_acto(self):
        return ["El juez tuvo por sustituida a la tercera."]

    def parrafos_conceptos(self):
        return ["La recurrente alega violación al 17 constitucional."]


_r = types.SimpleNamespace(fases=_Fases(), encargo=types.SimpleNamespace(es_recurso=True))
ok(te.huella_contraste(_r) == "ae83104d805f34c1",
   "la huella del adelanto es la misma de antes, byte a byte (medida antes de este cambio)")
ok(sorted(te.material_ligero(types.SimpleNamespace(tesis=[{"registro": "1"}]))) == sorted(
    ["completo", "consulta_estado", "convencional", "cuaderno", "decisiva", "entidad", "espejo", "materia",
     "normas", "preceptos_de_internet", "principios", "requisitos", "sede_del_acto", "sondeo", "tesis",
     "tipo_asunto", "tribunal"]),
   "material_ligero no gana campos (sin «huella» ni «consume»): la fila de hoy es la de siempre")

# ═══ 10 · LO DOCUMENTADO CASA CON EL CÓDIGO ══════════════════════════════════
print("\n10 · LAS MARCAS QUE ESCRIBE main.py ESTÁN EN EL MANIFIESTO")
_main = open("main.py", encoding="utf-8").read()
_claves_mod = {}
for _mod in ("analisis_litis", "pregunta_decisiva"):
    _mm = re.search(r'^CLAVE_MARCA\s*=\s*"([a-z_]+)"', open(f"{_mod}.py", encoding="utf-8").read(), re.M)
    _claves_mod[_mod] = _mm.group(1) if _mm else None
_alias = {"_al": _claves_mod["analisis_litis"], "_pd": _claves_mod["pregunta_decisiva"]}
escritas = set(re.findall(r'_taller_guardar_marca\(\s*\w+,\s*\w+,\s*"([a-z_]+)"', _main))
escritas |= {_alias[a] for a in re.findall(r'_taller_guardar_marca\(\s*\w+,\s*\w+,\s*(_al|_pd)\.CLAVE_MARCA', _main)}
escritas |= set(re.findall(r'otras=\{"([a-z_]+)"', _main))
escritas |= {"proyecto"} if 'est["proyecto"] = ficha' in _main else set()
ok(len(escritas) >= 8 and escritas <= set(te.MARCAS),
   f"toda marca que main.py escribe está en MARCAS ({', '.join(sorted(escritas))})")
ok({"consulta", "decisiva", "contraste", "analisis", "propuesta", "global_propuesta", "deliberacion"} <= escritas,
   "y la lectura de main.py encuentra las siete de la cabecera")
_cab = _src[_src.find("# ═══ LOS ESTADOS DE LA SESIÓN"):_src.find("MANIFIESTO_VERSION =")]
ok(all(f"estado.{m}" in _cab for m in escritas), "la cabecera documenta cada una con su huella de hoy")
ok('"inventario"' in open("inventario_escrito.py", encoding="utf-8").read()
   and 'd["recalificaciones"]' in open("recalificar.py", encoding="utf-8").read()
   and '"planes": {}' in open("plan_estudio.py", encoding="utf-8").read(),
   "las tres ramas de la columna plan siguen llamándose como en MARCAS")
import inventario_escrito as _ie
import plan_estudio as _pe
import recalificar as _rc
ok(te._ABANDONO_S == {"plan": _pe.PLAN_ABANDONADO_S, "inventario": _ie.ABANDONADA_S,
                      "recalificacion": _rc.ABANDONADA_S}
   and te.LATIDO_ABANDONADO_S == 150.0,
   "los umbrales de abandono casan con los de cada módulo")

print("\n· LO QUE LA REVISIÓN ADVERSARIAL VIO SIN PROBAR (29-sep): marcas de otro adelanto y fallos que retiran")
_viejo = {"huella": "H1",
          "contraste": {"huella": "H0", "estado": "listo", "items": [{"x": 1}]},
          "analisis": {"huella": "H0", "estado": "listo", "doc": {"razones": []}},
          "propuesta": {"huella": "H0", "estado": "listo", "respuesta": {"global": {"sentido": "fundado"}}},
          "decisiva": {"huella": "H0:ficha", "estado": "listo", "doc": {"pregunta_decisiva": "x"}}}
_a = te.actuales_de_fila(_viejo)
ok(not any(k in _a for k in ("contraste", "analisis", "propuesta", "decisiva", "global")),
   "las marcas de OTRO adelanto no cuentan como lo actual (mata M1: _del_adelanto siempre True)")
_fallo = {"huella": "H1",
          "contraste": {"huella": "H1", "estado": "fallo"},
          "analisis": {"huella": "H1", "estado": "error"},
          "propuesta": {"huella": "H1", "estado": "fallo"},
          "decisiva": {"huella": "H1:ficha", "estado": "fallo"},
          "consulta": {"huella": "H1", "estado": "fallo"}}
_b = te.actuales_de_fila(_fallo)
ok(all(_b.get(k) == "" for k in ("contraste", "analisis", "propuesta", "decisiva")),
   "una marca de ESTE adelanto que falló retira lo que había (\"\"), no se ignora (mata M2, M18, M19)")
ok(any(k != "v" and k != "adelanto" for k in _b if k not in ("contraste", "analisis", "propuesta", "decisiva")),
   "la consulta que falló sin material también deja su rastro (el material vacío)")

print()
if FALLOS:
    print(f"FALLAN {len(FALLOS)}: " + " · ".join(FALLOS))
    raise SystemExit(1)
print("RESULTADO: TODAS LAS COMPROBACIONES PASAN")
