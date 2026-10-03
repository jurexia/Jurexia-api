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
r = ex.decidir(d, True, vp, va)
ok(r["via"] == "alternativa" and r["lado"] == "no_prospera" and r["p_prospera"] == 0.2 and not r["coincide_con_su_lado"],
   "la probabilidad manda sobre la letra que eligió (regla del 50.01%)")
ok(r["fallas"]["propuesta"] == [{"gravedad": "fatal"}] and r["solidez"]["alternativa"] == 8,
   "las fallas de cada vía se devuelven por su nombre, no por la letra")
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
               {"fraccion": "II", "rotulo": "menores", "a_favor_de": "los menores"})
ok("ACTO" in _p and "ESCRITO" in _p and "OPERA la suplencia" in _p and "tasa" not in _p.lower()
   and "estadística" in _p, "el prompt: los dos documentos, la suplencia y ninguna tasa")

print()
if FALLOS:
    print(f"FALLAN {len(FALLOS)}: " + " · ".join(FALLOS))
    raise SystemExit(1)
print("RESULTADO: TODAS LAS COMPROBACIONES PASAN")
