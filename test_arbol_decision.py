"""La suerte de los accesorios la dicta el principal — ADC 93/2026, 22-sep-2026.

El caso real va literal: dos problemas, el principal (¿debió admitirse la
ampliación de demanda?) y el accesorio (¿debía pronunciarse sobre los
alegatos?). El motor marcó el accesorio como tema DISTINTO y a la vez escribió
que, si el principal no prospera, «al no formar parte de la litis el crédito
fiscal, los alegatos no podían ampliarla». David marcó el principal infundado
y el accesorio siguió tratándose como el motor lo había generado.

    .venv/bin/python test_arbol_decision.py
"""
import inspect

import arbol_decision as ad

FALLOS = []


def ok(cond, que):
    print(f"   {'PASA ' if cond else 'FALLA'}  {que}")
    if not cond:
        FALLOS.append(que)


P1 = "¿La Sala debió admitir la ampliación de demanda y estudiar los argumentos dirigidos contra el crédito fiscal?"
P2 = "¿La Sala debía pronunciarse sobre los argumentos formulados por la actora en sus alegatos?"
PROBLEMAS = [
    {"pregunta": P1, "jerarquia": "principal", "depende_de": None},
    {"pregunta": P2, "jerarquia": "accesorio", "depende_de": None},
]
# La lista de comprobación TAL COMO SALIÓ en la sesión 448 (sin los campos nuevos).
CHECKLIST_VIEJA = [
    {"tema": "Admisión de la ampliación y estudio de los argumentos contra el crédito fiscal.",
     "papel": "principal", "numero": 1, "tema_distinto": False,
     "con_propuesta": "Fundado: debe dejarse insubsistente la decisión de preclusión…",
     "con_alternativa": "Infundado: la ampliación fue precluida y el crédito fiscal permaneció fuera de la litis."},
    {"tema": "Omisión de pronunciamiento sobre los seis argumentos formulados en alegatos.",
     "papel": "accesorio", "numero": 2, "tema_distinto": True,
     "con_propuesta": "Fundado en lo conducente: la Sala debe examinar los argumentos de alegatos que controviertan pruebas, la contestación o la competencia, sin que ello implique declarar directamente la nulidad del crédito.",
     "con_alternativa": "Infundado: al no formar parte de la litis el crédito fiscal, los alegatos no podían ampliarla y la nulidad previamente decretada hacía innecesario su estudio."},
]
PROPUESTAS = [{"problema": P1, "sentido": "fundado", "alcanza": True},
              {"problema": P2, "sentido": "fundado", "alcanza": True}]


def crit(s1, s2, tocado2=False):
    return [{"problema": P1, "sentido": s1, "razonamiento": "", "jerarquia": "principal", "tocado": True},
            {"problema": P2, "sentido": s2, "razonamiento": "razón vieja del motor", "jerarquia": "accesorio", "tocado": tocado2}]


print("\n1 · EL CASO REAL: el principal cae y el accesorio cae con él")
c = crit("infundado", "fundado")
av, det = ad.aplicar(PROBLEMAS, c, CHECKLIST_VIEJA, PROPUESTAS, sentido_motor="fundado")
ok(c[1]["sentido"] == "infundado", f"el accesorio pasa de fundado a {c[1]['sentido']}")
ok("no formar parte de la litis" in c[1]["razonamiento"], "con la razón que el motor había escrito para esa vía")
ok("razón vieja del motor" not in c[1]["razonamiento"], "y la razón del sentido viejo se va")
ok(det[P2]["de"] == "principal", "la pantalla sabrá que sigue al principal")
ok(any("CAEN CON EL PRINCIPAL" in a for a in av), "con su aviso")

print("\n2 · EL PRINCIPAL PROSPERA: el accesorio sigue la suerte escrita para esa vía")
c = crit("fundado", "infundado")
av, det = ad.aplicar(PROBLEMAS, c, CHECKLIST_VIEJA, PROPUESTAS, sentido_motor="fundado")
ok(c[1]["sentido"] == "fundado", f"«fundado en lo conducente» se lee como fundado: {c[1]['sentido']}")
ok("controviertan pruebas" in c[1]["razonamiento"], "con su razón")

print("\n3 · LO QUE EL SECRETARIO MARCÓ NO SE TOCA")
c = crit("infundado", "fundado", tocado2=True)
av, det = ad.aplicar(PROBLEMAS, c, CHECKLIST_VIEJA, PROPUESTAS, sentido_motor="fundado")
ok(c[1]["sentido"] == "fundado" and det[P2]["de"] == "tuya", "su marca manda")
ok(any("Se respeta tu marca" in a for a in av), "y se le dice que el motor había escrito otra suerte")

print("\n4 · LA LISTA NUEVA, ESTRUCTURADA")
CHECKLIST_NUEVA = [
    dict(CHECKLIST_VIEJA[0]),
    {"tema": "alegatos", "papel": "accesorio", "numero": 2, "relacion": "depende",
     "si_prospera": {"sentido": "innecesario", "razon": "la nueva sentencia los atenderá al integrar la litis"},
     "si_no_prospera": {"sentido": "inoperante", "razon": "presupone que el crédito estaba en la litis"}},
]
c = crit("fundado", "")
av, det = ad.aplicar(PROBLEMAS, c, CHECKLIST_NUEVA, PROPUESTAS)
ok(c[1]["sentido"] == "innecesario" and "integrar la litis" in c[1]["razonamiento"],
   "principal fundado → sin materia, con la razón")
ok(any("SUSTRACCIÓN DE MATERIA" in a for a in av), "con el aviso de sustracción")
c = crit("infundado", "")
av, det = ad.aplicar(PROBLEMAS, c, CHECKLIST_NUEVA, PROPUESTAS)
ok(c[1]["sentido"] == "inoperante" and "presupone que el crédito" in c[1]["razonamiento"],
   "principal infundado → inoperante, con la razón")

print("\n5 · LOS ESCAPES")
c = crit("fundado", "infundado")
av, det = ad.aplicar(PROBLEMAS, c, CHECKLIST_NUEVA, [{"problema": P1, "sentido": "fundado", "alcanza": False}])
ok(c[1]["sentido"] == "infundado" and any("NO SE APLICÓ" in a for a in av),
   "si lo fundado no alcanza, no hay sustracción")
c = crit("fundado", "infundado")
lista_distinta = [dict(CHECKLIST_NUEVA[0]), {"numero": 2, "tema": "alegatos", "papel": "accesorio", "tema_distinto": True}]
av, det = ad.aplicar(PROBLEMAS, c, lista_distinta, PROPUESTAS)
ok(c[1]["sentido"] == "infundado" and det[P2]["de"] == "distinto", "tema distinto sin suerte escrita: se estudia aparte")
P3 = "¿Procede declarar la nulidad lisa y llana del crédito por caducidad?"
probs3 = PROBLEMAS + [{"pregunta": P3, "jerarquia": "accesorio", "depende_de": 1}]
c3 = crit("fundado", "") + [{"problema": P3, "sentido": "infundado", "razonamiento": "", "jerarquia": "accesorio"}]
av, det = ad.aplicar(probs3, c3, CHECKLIST_NUEVA, PROPUESTAS)
ok(c3[2]["sentido"] == "infundado" and det[P3]["de"] == "mayor_beneficio", "el de mayor beneficio se estudia")

print("\n6 · SIN LISTA: `depende_de` de la fase 3 basta")
probs = [{"pregunta": P1, "jerarquia": "principal"}, {"pregunta": P2, "jerarquia": "accesorio", "depende_de": 1}]
c = crit("infundado", "fundado")
av, det = ad.aplicar(probs, c, [], [])
ok(c[1]["sentido"] == "inoperante", "depende del 1 y el 1 cae: inoperante")
c = crit("fundado", "fundado")
av, det = ad.aplicar(probs, c, [], [])
ok(c[1]["sentido"] == "innecesario", "depende del 1 y el 1 prospera: innecesario")
probs_ind = [{"pregunta": P1, "jerarquia": "principal"}, {"pregunta": P2, "jerarquia": "accesorio", "depende_de": None}]
c = crit("infundado", "fundado")
av, det = ad.aplicar(probs_ind, c, [], [])
ok(c[1]["sentido"] == "fundado" and det[P2]["de"] == "propio", "sin dependencia declarada: se estudia por su cuenta")

print("\n7 · SOBRE OBJETOS Criterio Y PARA LA PANTALLA")
import fase6_estudio as f6
cc = [f6.Criterio(problema=P1, sentido="infundado", jerarquia="principal"),
      f6.Criterio(problema=P2, sentido="fundado", jerarquia="accesorio")]
ad.aplicar(PROBLEMAS, cc, CHECKLIST_NUEVA, PROPUESTAS)
ok(cc[1].sentido == "inoperante", "también sobre Criterio")
r = ad.reparto_para_pantalla(PROBLEMAS, crit("infundado", "fundado"), CHECKLIST_VIEJA, PROPUESTAS, "fundado")
ok(r["criterios"][1]["sentido"] == "infundado" and r["criterios"][1]["de"] == "principal"
   and r["criterios"][1]["por_que"], "el reparto para la pantalla trae sentido, de quién y por qué")

print("\n8 · LA LECTURA DE LA PROSA")
ok(ad._leer_suerte("Fundado en lo conducente: la Sala debe…")[0] == "fundado", "«Fundado en lo conducente»")
ok(ad._leer_suerte("Inoperante, porque descansa en…") == ("inoperante", "porque descansa en…"), "«Inoperante, porque…»")
ok(ad._leer_suerte("Queda sin materia al prosperar el principal")[0] == "innecesario", "«Queda sin materia»")
ok(ad._leer_suerte("Fundado pero insuficiente: no alcanza")[0] == "fundado_insuficiente", "«Fundado pero insuficiente»")
ok(ad._leer_suerte("SIN DETERMINAR — el motor no lo incluyó.") == ("", ""), "lo indeterminado no se lee como sentido")
ok(ad._leer_suerte("La Sala debe examinar los alegatos") == ("", ""), "prosa sin calificación al frente: nada")

print("\n9 · LAS PUERTAS ESTÁN CONECTADAS")
src = open("main.py", encoding="utf-8").read()
ok(src.count("_ad.aplicar(") >= 3, f"los dos gemelos del resolver y la propuesta aplican el árbol ({src.count('_ad.aplicar(')})")
ok('@app.post("/taller/reparto")' in src, "la pantalla tiene su puerta: /taller/reparto")
ok('@app.post("/taller/problema")' in src, "y el problema jurídico se puede corregir: /taller/problema")
ok('"tocado"' in src, "los criterios distinguen lo que el secretario marcó")
import fase5_propuesta as f5
ok("si_prospera" in inspect.getsource(f5.prompt_propuesta), "la fase 5 escribe la suerte condicional")

print()
if FALLOS:
    print(f"FALLAN {len(FALLOS)}: " + " · ".join(FALLOS))
    raise SystemExit(1)
print("RESULTADO: TODAS LAS COMPROBACIONES PASAN")
