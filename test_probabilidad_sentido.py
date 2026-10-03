# -*- coding: utf-8 -*-
"""EL LADO DE LA PROPUESTA (2-oct-2026): la mecánica pura del volteo, el aviso
de los precedentes y la lectura del examinador, sin red.

    .venv/bin/python test_probabilidad_sentido.py

David: «tú partes de que es con la tasa del tribunal y no es así (…) lo que
resuelve es la inteligencia en el razonamiento jurídico». Aquí se fija que:
la probabilidad la da el examen de las dos vías; la tasa del tribunal no
existe en el cálculo; los precedentes del tribunal sólo avisan; sin examen
manda el motor, sin número inventado.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import probabilidad_sentido as ps
import examinador as ex

FALLOS = []


def ok(c, q):
    print(f"   {'PASA ' if c else 'FALLA'}  {q}")
    if not c:
        FALLOS.append(q)


print("\n1 · QUÉ PROSPERA")
for c, v in (("fundado", 1), ("esencialmente_fundado", 1), ("parcialmente fundado", 1),
             ("infundado", 0), ("inoperante", 0), ("fundado pero inoperante", 0),
             ("fundado_insuficiente", 0), ("ineficaz", 0), ("no se estudió", None), ("sin_materia", None), ("", None)):
    ok(ps.prospera(c) == v, f"{c or '(vacía)'} → {v}")
ok(not hasattr(ps, "calcular") and not hasattr(ps, "tasa") and not os.path.exists(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "probabilidad_sentido.json")),
   "la tasa del tribunal ya no existe: ni función ni JSON")


def _glob():
    return {"sentido": "fundado", "razon": "razón del motor", "efecto": "e", "apoyos": ["1"],
            "confianza": "alta", "alternativa": {"sentido": "infundado", "razon": "razón contraria",
                                                 "efecto": "ec", "apoyos": ["2"]},
            "checklist": [{"con_propuesta": "a", "con_alternativa": "b"}]}


def _props():
    return [{"problema": "P1", "sentido": "fundado", "razon": "razón del motor", "apoyos": ["1"]}]


print("\n2 · EL VOLTEO CON LA PROBABILIDAD RAZONADA")
g, pr = _glob(), _props()
info = ps.aplicar(g, pr, [{"pregunta": "P1", "jerarquia": "principal"}],
                  {"p_prospera": 0.2, "lado": "no_prospera", "fuente": "examinador"})
ok(info["volteada"] and g["sentido"] == "infundado" and g["razon"] == "razón contraria"
   and g["alternativa"]["sentido"] == "fundado" and g["en_contra"] == "razón del motor",
   "gana la contraria: se intercambian las vías")
ok(pr[0]["sentido"] == "infundado" and pr[0]["origen"] == "probabilidad" and pr[0]["sentido_motor"] == "fundado",
   "el principal va con el lado nuevo; lo del motor en sentido_motor")
ok(g["confianza"] == "alta" and g["confianza_motor"] == "alta" and g["checklist"][0]["con_propuesta"] == "b",
   "confianza del lado (0.8 → alta) y la lista intercambiada")
g, pr = _glob(), _props()
info = ps.aplicar(g, pr, [{"pregunta": "P1"}], {"p_prospera": 0.7, "lado": "prospera", "fuente": "examinador"})
ok(not info["volteada"] and g["sentido"] == "fundado" and g["confianza"] == "media",
   "gana la propuesta: se queda, con confianza media (0.7)")
g, pr = _glob(), _props()
info = ps.aplicar(g, pr, [{"pregunta": "P1"}], None)
ok(not info["volteada"] and g["sentido"] == "fundado" and info["probabilidad"]["fuente"] == "motor"
   and info["probabilidad"]["p_prospera"] is None, "sin examen manda el motor, sin número")
g = _glob(); g["alternativa"] = {}
pr = _props()
info = ps.aplicar(g, pr, [{"pregunta": "P1"}], {"p_prospera": 0.1, "lado": "no_prospera"})
ok(info["necesita_razon"] and g["sentido"] == "infundado", "sin vía contraria escrita, se pide la razón")

print("\n3 · LOS PRECEDENTES SÓLO AVISAN")
filas = [{"nivel": "mismo_problema", "calificacion": "fundado", "expediente": "AD 10/2024", "similitud": 95, "neun": 1},
         {"nivel": "posible", "calificacion": "infundado", "expediente": "AD 11/2024", "similitud": 60, "neun": 2}]
a = ps.aviso_precedentes(filas, "infundado")
ok("al revés" in a and "10/2024" in a and "11/2024" not in a, "sólo los de «mismo problema», y dice si van al revés")
ok("en el mismo sentido" in ps.aviso_precedentes(filas, "fundado"), "o en el mismo sentido")
ok(ps.aviso_precedentes([], "fundado") == "", "sin precedentes, sin aviso")

print("\n4 · LA LECTURA DEL EXAMINADOR")
vp = {"sentido": "fundado", "razon": "r"}
va = {"sentido": "infundado", "razon": "c"}
d = {"lado": "A", "p_prospera": 0.2, "razon_decisiva": "porque sí",
     "examen": {"A": {"fallas": [{"gravedad": "fatal"}], "solidez": 3}, "B": {"fallas": [], "solidez": 8}}}
r = ex.decidir({**d, "lado": "B"}, True, vp, va)
ok(r["via"] == "alternativa" and r["lado"] == "no_prospera" and r["p_prospera"] == 0.2 and r["coincide_con_su_lado"]
   and not r["incoherente"], "número y letra van juntos: la probabilidad decide (regla del 50.01%)")
ok(r["fallas"]["propuesta"] == [{"gravedad": "fatal"}] and r["solidez"]["alternativa"] == 8,
   "las fallas de cada vía se devuelven por su nombre, no por la letra")
# NÚMERO CONTRA LETRA (revisión adversarial, 3-oct-2026): antes mandaba el
# número; ahora manda la letra y el número se descarta.
r = ex.decidir(d, True, vp, va)
ok(r["via"] == "propuesta" and r["lado"] == "prospera" and r["p_prospera"] is None and r["incoherente"],
   "si el número contradice la letra, manda la letra y no hay número")
r = ex.decidir({"lado": "B", "p_prospera": 0.9}, False, vp, va)
ok(r["via"] == "propuesta" and r["lado"] == "prospera", "con la propuesta como B")
r = ex.decidir({"lado": "B"}, True, vp, va)
ok(r["via"] == "alternativa" and r["p_prospera"] is None, "sin probabilidad, manda la letra; no se inventa número")
ok(ex.decidir({}, True, vp, va) is None and ex.decidir(None, True, vp, va) is None, "sin nada, None")
ok(ex.leer('texto {"lado": "A", "p_prospera": 0.3} más') == {"lado": "A", "p_prospera": 0.3}
   and ex.leer("sin json") is None, "lee el JSON aunque venga envuelto")
ok(ex.propuesta_es_a("631/2025") == ex.propuesta_es_a("631/2025"), "el orden A/B es fijo para el mismo asunto")
_ords = {ex.propuesta_es_a(f"{n}/2025") for n in range(1, 40)}
ok(_ords == {True, False}, "y se reparte entre asuntos")
e = ex.explicacion({"p_prospera": 0.2, "lado": "no_prospera", "razon_decisiva": "La razón."}, "infundado", True, "fundado")
ok("80%" in e and "infundado" in e and "se inclinaba por «fundado»" in e and e.endswith("La razón."),
   f"la explicación: {e[:120]}…")
_p = ex.prompt("amparo_directo", [{"pregunta": "¿P?"}], vp, va, {}, [], [], "ACTO", "ESCRITO",
               {"fraccion": "II", "rotulo": "menores", "a_favor_de": "los menores", "confirmada": True})
ok("ACTO" in _p and "ESCRITO" in _p and "OPERA la suplencia" in _p and "tasa" not in _p.lower()
   and "estadística" in _p, "el prompt: los dos documentos, la suplencia y ninguna tasa")

print("\n5 · P_PROSPERA EN ESCALA DE 100 (revisión adversarial, 3-oct-2026)")
# El modelo devolvía 35 (por 35 %) eligiendo la vía que NO prospera: se
# recortaba a 1.0 y salía la que prospera, al 100 %.
for _pa in (True, False):
    _letra_no = "B" if _pa else "A"      # la alternativa (infundado) es la otra letra
    r = ex.decidir({"lado": _letra_no, "p_prospera": 35}, _pa, vp, va)
    ok(r["via"] == "alternativa" and r["lado"] == "no_prospera" and r["p_prospera"] == 0.35
       and not r["incoherente"], f"p=35 con la letra de la que no prospera → 0.35, no prospera (A es propuesta: {_pa})")
    _letra_si = "A" if _pa else "B"
    r = ex.decidir({"lado": _letra_si, "p_prospera": 70}, _pa, vp, va)
    ok(r["via"] == "propuesta" and r["p_prospera"] == 0.7, f"p=70 con su letra → 0.70, no 1.0 (A es propuesta: {_pa})")
    r = ex.decidir({"lado": _letra_si, "p_prospera": 35}, _pa, vp, va)
    ok(r["via"] == "propuesta" and r["p_prospera"] is None and r["incoherente"],
       "p=35 (no prospera) contra la letra de la que prospera: manda la letra, sin número")
r = ex.decidir({"lado": "B", "p_prospera": 250}, True, vp, va)
ok(r["via"] == "alternativa" and r["p_prospera"] is None, "p fuera de escala (250): sin número, decide la letra")
ok(ex.normalizar_p(1) == 1.0 and ex.normalizar_p(0.4) == 0.4 and ex.normalizar_p(-1) is None
   and ex.normalizar_p("x") is None, "1 es 100 %; negativo o ilegible, sin número")
e = ex.explicacion(ex.decidir({"lado": "B", "p_prospera": 35}, True, vp, va), "infundado", True, "fundado")
ok("65%" in e and "100%" not in e, f"la explicación da el 65 % del lado propuesto: {e[:70]}")

print("\n6 · LA SUPLENCIA, COMO HECHO SÓLO SI LA CONFIRMÓ EL SECRETARIO")
def _pr(sup, tipo="amparo_directo"):
    return ex.prompt(tipo, [{"pregunta": "¿P?"}], vp, va, {}, [], [], "ACTO", "ESCRITO", sup)
_auto_vii = {"fraccion": "VII", "rotulo": "fracción VII", "a_favor_de": "la quejosa", "porque": "dice ser pobre",
             "alternativas": [], "pedida": ""}
_x = _pr(_auto_vii)
ok("OPERA la suplencia" not in _x and "TODAVÍA NO HA DECIDIDO" in _x and "no la presumas ni la excluyas" in _x
   and "fracción VII" in _x, "la VII automática (lo que la parte afirma) es indicio, no «OPERA»")
_auto_ninguna = {"fraccion": "ninguna", "alternativas": [{"fraccion": "II", "rotulo": "fracción II",
                                                           "porque": "habla de su hijo"}], "pedida": "II"}
_x = _pr(_auto_ninguna)
ok("rige el estricto derecho" not in _x and "NO opera" not in _x and "fracción II" in _x
   and "la pide expresamente" in _x, "«ninguna» automática no afirma el estricto derecho; alternativas y lo pedido, como dato")
_x = _pr({"fraccion": "ninguna", "confirmada": True})
ok("CONFIRMÓ" in _x and "rige el estricto derecho" in _x, "«ninguna» confirmada por el secretario: estricto derecho")
_x = _pr({"fraccion": "V", "a_favor_de": "la trabajadora", "confirmada": True})
ok("CONFIRMÓ que en este asunto OPERA" in _x and "la trabajadora" in _x, "la confirmada se dice como hecho")
_x = _pr({"fraccion": "ninguna"}, "revision_fiscal")
ok("no es un juicio de amparo" in _x, "revisión fiscal: sin artículo 79, por ley")
ok("8. La suplencia" not in _pr(None), "sin suplencia, sin regla 8")

print("\n7 · LO QUE VIO EL MOTOR LLEGA AL EXAMINADOR (constancias, aportado, ficha)")
_x = ex.prompt("amparo_directo", [{"pregunta": "¿P?"}], vp, va, {}, [], [], "ACTO", "ESCRITO", None,
               autos="CONSTANCIAS QUE OBRAN EN AUTOS: cédula de notificación del 3 de marzo",
               ficha="Recurre: la quejosa")
ok("cédula de notificación del 3 de marzo" in _x and "Recurre: la quejosa" in _x,
   "el prompt lleva las constancias y la ficha")
ok(_x.index("cédula de notificación") < _x.index("RESOLUCIÓN RECLAMADA O RECURRIDA, ÍNTEGRA"),
   "en un bloque ANTES de la resolución")
ok("contra eso también se verifica" in _x, "la regla 3 verifica las premisas también contra ese bloque")
_sin = ex.prompt("amparo_directo", [{"pregunta": "¿P?"}], vp, va, {}, [], [], "ACTO", "ESCRITO", None)
ok("CONSTANCIAS DE AUTOS" not in _sin and "contra eso también se verifica" not in _sin,
   "sin autos ni ficha, ni bloque ni regla extra")
_largo = "RESPUESTAS DEL SECRETARIO al inicio. " + ("x" * (ex.MAX_AUTOS * 2)) + " LA EJECUTORIA al final."
_x = ex.prompt("amparo_directo", [], vp, va, {}, [], [], "ACTO", "ESCRITO", None, autos=_largo)
ok("RESPUESTAS DEL SECRETARIO" in _x and "LA EJECUTORIA al final" in _x and len(_x) < ex.MAX_AUTOS + 30_000,
   "tope propio, recortado por EN MEDIO: la cabeza y la cola sobreviven")

# EXAMINAR, ENTERO, CON UN CLIENTE FALSO: el prompt que sale lleva los autos.
import asyncio
import types as _types
_ENVIADO = {}


class _Modelos:
    async def generate_content(self, model, contents, config):
        _ENVIADO["texto"] = contents
        return _types.SimpleNamespace(
            text='{"lado": "B", "p_prospera": 35, "razon_decisiva": "La vía B se sostiene porque la vía A '
                 'no combate la razón autónoma.", "que_lo_cambiaria": "que la A acreditara la notificación"}',
            usage_metadata=None)


ex._gemini = lambda: _types.SimpleNamespace(aio=_types.SimpleNamespace(models=_Modelos()))
_num = next(f"{n}/2026" for n in range(1, 50) if ex.propuesta_es_a(f"{n}/2026"))
dec = asyncio.run(ex.examinar(_num, "amparo_directo", [], vp, va, {}, [], [], "ACTO", "ESCRITO", None,
                              autos="CONSTANCIA APORTADA: acuse de recibo", ficha="FICHA: quejosa recurre"))
ok("CONSTANCIA APORTADA: acuse de recibo" in _ENVIADO.get("texto", "") and "FICHA: quejosa recurre" in _ENVIADO["texto"],
   "examinar() pasa los autos y la ficha al prompt")
ok(dec and dec["via"] == "alternativa" and dec["p_prospera"] == 0.35, "y lee el 35 como 0.35")

print("\n8 · LA RAZÓN DECISIVA NO HABLA DE «VÍA A / VÍA B»")
ok(dec and "vía A" not in dec["razon_decisiva"] and "vía B" not in dec["razon_decisiva"]
   and dec["razon_decisiva"].startswith("La vía que se propone se sostiene porque la vía contraria"),
   f"las letras, por la vía que se propone o la contraria: {dec and dec['razon_decisiva'][:80]}")
ok(dec and "la vía contraria acreditara" in dec["que_lo_cambiaria"], "también en «qué lo cambiaría»")
ok("NUNCA por su letra" in _ENVIADO["texto"], "y el prompt pide nombrarlas por su sentido")
ok(ex.limpiar_letras("El A quo resolvió; A juicio de la Sala, el apartado B del 123 rige.", True, "propuesta")
   == "El A quo resolvió; A juicio de la Sala, el apartado B del 123 rige.",
   "la preposición «A», el «A quo» y el «apartado B» no se tocan")
ok(ex.limpiar_letras("Las vías A y B difieren; del lado B nada.", False, "propuesta")
   == "Las dos vías difieren; de la vía que se propone nada.", "«las vías A y B» y «del lado B»")

print("\n9 · LO QUE FIJÓ LA EJECUTORIA NO SE VOLTEA (revisión adversarial, 3-oct-2026)")
def _glob_neg():
    return {"sentido": "infundado", "razon": "razón del motor", "efecto": "e", "apoyos": ["1"], "confianza": "media",
            "alternativa": {"sentido": "fundado", "razon": "razón contraria", "efecto": "ec", "apoyos": ["2"]},
            "checklist": [{"con_propuesta": "a", "con_alternativa": "b"}]}
g = _glob_neg()
pr = [{"problema": "P1", "sentido": "inoperante", "razon": "vinculado por la ejecutoria", "apoyos": ["3"],
       "origen": "ejecutoria", "sentido_propio": "infundado"}]
info = ps.aplicar(g, pr, [{"pregunta": "P1", "jerarquia": "principal"}],
                  {"p_prospera": 0.6, "lado": "prospera", "fuente": "examinador"})
ok(not info["volteada"] and info["fija_ejecutoria"] and g["sentido"] == "infundado" and g["razon"] == "razón del motor",
   "el examen pedía conceder: la global se queda con el lado del motor")
ok(pr[0]["sentido"] == "inoperante" and pr[0]["origen"] == "ejecutoria" and pr[0]["razon"] == "vinculado por la ejecutoria",
   "el principal que fijó la ejecutoria no se toca")
_pb = info["probabilidad"]
ok(_pb["lado"] == "no_prospera" and _pb["p_prospera"] is None and _pb["fijada_por"] == "ejecutoria"
   and _pb["examen_lado"] == "prospera" and _pb["examen_p"] == 0.6 and g["confianza"] == "media",
   "la probabilidad dice el lado del motor, sin número; lo del examen queda aparte")
ok(any(a.startswith("EL PRINCIPAL LO FIJA LA EJECUTORIA") and "60%" in a for a in info["avisos"]),
   "y se avisa")
g = _glob(); g["alternativa"] = {"sentido": "infundado", "razon": "razón contraria", "efecto": "", "apoyos": ["2"]}
pr = [{"problema": "P1", "sentido": "inoperante", "razon": "vinculado", "apoyos": ["3"], "origen": "ejecutoria"}]
info = ps.aplicar(g, pr, [{"pregunta": "P1", "jerarquia": "principal"}], {"p_prospera": 0.3, "lado": "no_prospera"})
ok(info["volteada"] and not info["fija_ejecutoria"] and g["sentido"] == "infundado" and pr[0]["sentido"] == "inoperante",
   "si el examen va al lado del principal fijado, el volteo de la global corre y el principal no se toca")

print("\n10 · EL VOLTEO DESDE «SIN MATERIA» SE AVISA")
g = {"sentido": "sin_materia", "razon": "queda sin materia", "apoyos": [],
     "alternativa": {"sentido": "fundado", "razon": "se concede", "apoyos": ["2"]}}
pr = [{"problema": "P1", "sentido": "sin_materia", "razon": "x", "apoyos": []}]
info = ps.aplicar(g, pr, [{"pregunta": "P1"}], {"p_prospera": None, "lado": "prospera", "explicacion": "E"})
ok(info["volteada"] and g["sentido"] == "fundado" and any("VOLTEÓ" in a for a in info["avisos"]),
   "de «sin materia» a «fundado»: es volteo y se avisa")

print("\n11 · LOS AVISOS DE REGISTROS SE REHACEN TRAS EL VOLTEO")
import fase5_propuesta as f5
_mat = _types.SimpleNamespace(tesis=[{"registro": "2001111"}])
g = f5.Global(sentido="fundado", razon="r", apoyos=["2001111"], alcanza=True,
              alternativa={"sentido": "infundado", "razon": "c", "efecto": "", "apoyos": ["2009999"]})
props = [f5.Propuesta(problema="P1", sentido="fundado", razon="r", apoyos=["2001111"])]
avisos = (f5.revisar(props, _mat) + f5.revisar_global(g, _mat) + ["otro aviso"])
ok(any(a.startswith("La vía alternativa se apoya") and "2009999" in a for a in avisos),
   "antes del volteo, el inventado es de «la vía alternativa»")
_ap0 = list(props[0].apoyos)
ps.aplicar(g, props, [{"pregunta": "P1", "jerarquia": "principal"}], {"p_prospera": 0.2, "lado": "no_prospera"})
ps.rehacer_avisos_registros(avisos, g, props, _mat, 0, _ap0)
ok(any(a.startswith("La propuesta del asunto se apoya") and "2009999" in a for a in avisos)
   and not any(a.startswith("La vía alternativa se apoya") and "2009999" in a for a in avisos),
   "después, lo lleva «la propuesta del asunto», que es la que se acepta")
ok(any(a.startswith("La propuesta se apoya en registros") and "2009999" in a for a in avisos)
   and "otro aviso" in avisos, "el del principal también, y lo demás se queda")

print()
if FALLOS:
    print(f"FALLAN {len(FALLOS)}: " + " · ".join(FALLOS))
    raise SystemExit(1)
print("RESULTADO: TODAS LAS COMPROBACIONES PASAN")
