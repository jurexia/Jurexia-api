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

# ═══════════════════════════════════════════════════════════════════════════
# LA GUARDA PROCESAL — decisión 1 de David, 26-sep-2026
# ═══════════════════════════════════════════════════════════════════════════
# Los artículos 74, fracción V, y 174 de la Ley de Amparo mandan decidir TODAS
# las violaciones procesales; la única excepción es que un problema de FONDO
# prospere con mayor beneficio que la reposición (artículo 189). El árbol
# declaraba innecesaria la segunda procesal igual que cualquier accesorio.
import modos_decision as md

VP1 = "¿La Sala debió admitir la ampliación de demanda presentada el catorce de agosto?"
VP2 = "¿La Sala debió admitir la prueba pericial contable ofrecida por la actora?"
FON = "¿Es legal la determinación del crédito fiscal por omisión de ingresos?"
FON2 = "¿Procede la condena en costas impuesta en la sentencia?"


def probs(principal, *acc, dep=1):
    """El principal primero; cada accesorio depende del principal (fase 3)."""
    out = [dict(principal, jerarquia="principal", depende_de=None)]
    for a in acc:
        out.append(dict(a, jerarquia="accesorio", depende_de=dep))
    return out


def cr(p, s, jer="accesorio", razon="", tocado=False):
    return {"problema": p, "sentido": s, "razonamiento": razon, "jerarquia": jer, "tocado": tocado}


PV1 = {"pregunta": VP1, "clase": "procesal"}
PV2 = {"pregunta": VP2, "clase": "procesal"}
PF = {"pregunta": FON, "clase": "fondo"}
PF2 = {"pregunta": FON2, "clase": "fondo"}

print("\n10 · DOS PROCESALES Y EL PRINCIPAL (PROCESAL) FUNDADO")
pp = probs(PV1, PV2, PF)
cc = [cr(VP1, "fundado", "principal", tocado=True), cr(VP2, "infundado", razon="razón propia"),
      cr(FON, "fundado")]
av, det = ad.aplicar(pp, cc, [], [])
ok(cc[1]["sentido"] == "infundado", f"la segunda procesal NO queda sin materia: {cc[1]['sentido']}")
ok(cc[1]["razonamiento"] == "razón propia", "y conserva su razón")
ok(det[VP2].get("guarda") == "procesal" and det[VP2]["de"] == "propio", "la pantalla sabrá que se estudia")
ok(any("VIOLACIÓN PROCESAL y NO se declaró sin materia" in a and "74, fracción V, y 174" in a
       and "también procesal" in a for a in av), "con su aviso y sus artículos")
ok(cc[2]["sentido"] == "innecesario", "el fondo, que depende, sí queda sin materia con la reposición")
ok(any("SUSTRACCIÓN DE MATERIA aplicada a 1" in a for a in av), "y la sustracción cuenta sólo el fondo")
# La suerte que el motor escribió para esta vía («innecesario») tampoco la saca.
lista_inn = [{"numero": 1, "tema": "ampliación", "papel": "principal"},
             {"numero": 2, "tema": "pericial", "papel": "accesorio", "relacion": "depende",
              "si_prospera": {"sentido": "innecesario", "razon": "la reposición lo absorbe"}}]
cc = [cr(VP1, "fundado", "principal", tocado=True), cr(VP2, "fundado")]
av, det = ad.aplicar(probs(PV1, PV2), cc, lista_inn, [])
ok(cc[1]["sentido"] == "fundado", "ni aunque el motor le haya escrito «innecesario» para esa vía")
# Si LLEGA sacada (de otra pasada), se le devuelve la calificación del motor.
cc = [cr(VP1, "fundado", "principal", tocado=True),
      cr(VP2, "innecesario", razon="Dado el sentido del estudio del problema principal, queda sin materia…")]
av, det = ad.aplicar(probs(PV1, PV2, dep=None), cc, [],
                     [{"problema": VP2, "sentido": "infundado", "alcanza": True}])
ok(cc[1]["sentido"] == "infundado" and "sin materia" not in cc[1]["razonamiento"],
   f"una procesal que llega «innecesario» recupera lo que el motor propuso: {cc[1]['sentido']}")
cc = [cr(VP1, "fundado", "principal", tocado=True), cr(VP2, "innecesario")]
av, det = ad.aplicar(probs(PV1, PV2, dep=None), cc, [], [])
ok(any("ESTÁ SIN CALIFICAR" in a for a in av), "y si no hay qué devolverle, se le pide al secretario")

print("\n11 · EL FONDO PROSPERA CON MAYOR BENEFICIO: la única excepción (art. 189)")
cc = [cr(FON, "fundado", "principal", tocado=True), cr(VP1, "infundado"), cr(VP2, "fundado")]
av, det = ad.aplicar(probs(PF, PV1, PV2), cc, [], [])
ok(cc[1]["sentido"] == "innecesario" and cc[2]["sentido"] == "innecesario",
   "las procesales quedan innecesarias por mayor beneficio")
ok(all("189" in x["razonamiento"] and "mayor" in x["razonamiento"] and "beneficio" in x["razonamiento"]
       for x in cc[1:]), "y su razón lo dice con el artículo 189")
ok(det[VP1].get("guarda") == "mayor_beneficio_189", "la pantalla sabe por qué")
ok(any("POR MAYOR BENEFICIO" in a and "márcala tú" in a for a in av),
   "con el aviso de que, si la concesión no da más que reponer, la marque")
ok(not any("SUSTRACCIÓN DE MATERIA aplicada" in a for a in av),
   "y no se cuenta como sustracción de materia: es otra razón")
cc = [cr(FON, "parcialmente_fundado", "principal", tocado=True), cr(VP1, "infundado")]
av, det = ad.aplicar(probs(PF, PV1), cc, [], [])
ok(cc[1]["sentido"] == "infundado",
   "si el fondo prospera sólo en parte no se puede afirmar que dé más que reponer: se decide")
cc = [cr("¿Fue legal el sobreseimiento del juicio de nulidad?", "fundado", "principal", tocado=True),
      cr(VP1, "infundado")]
av, det = ad.aplicar([{"pregunta": cc[0]["problema"], "jerarquia": "principal"},
                      {"pregunta": VP1, "jerarquia": "accesorio", "depende_de": 1}], cc, [], [])
ok(cc[1]["sentido"] == "infundado",
   "una concesión contra el sobreseimiento (procedencia) no es de fondo: la procesal se decide")
cc = [cr(FON, "fundado", "principal", tocado=True), cr(VP1, "infundado")]
av, det = ad.aplicar(probs(PF, PV1), cc, [], [{"problema": FON, "sentido": "fundado", "alcanza": False}])
ok(cc[1]["sentido"] == "infundado", "si lo fundado no alcanza, nada queda sin estudiar")

print("\n12 · EL PRINCIPAL NO PROSPERA: la procesal no cae con él")
cc = [cr(VP1, "infundado", "principal", tocado=True), cr(VP2, "fundado", razon="la pericial era idónea"),
      cr(FON, "fundado")]
lista_no = [{"numero": 1, "tema": "ampliación", "papel": "principal"},
            {"numero": 2, "tema": "pericial", "papel": "accesorio", "relacion": "depende",
             "si_no_prospera": {"sentido": "inoperante", "razon": "la pericial se ofreció en la ampliación"}},
            {"numero": 3, "tema": "crédito", "papel": "accesorio", "relacion": "depende",
             "si_no_prospera": {"sentido": "inoperante", "razon": "el crédito no entró a la litis"}}]
av, det = ad.aplicar(probs(PV1, PV2, PF), cc, lista_no, [])
# Revisión del 26-sep-2026. Antes conservaba el «fundado» de la vía CONTRARIA
# (el de la pantalla, con el principal fundado) y sólo enseñaba en un aviso lo
# que el motor escribió para ésta: la pericial ofrecida en una ampliación bien
# precluida salía fundada. Es el fallo del 93/2026 que dio origen al árbol.
ok(cc[1]["sentido"] == "inoperante" and cc[1]["razonamiento"] == "la pericial se ofreció en la ampliación",
   f"la procesal se decide con lo que el motor le escribió para ESTA vía: {cc[1]['sentido']}")
ok(not cc[1]["razonamiento"].startswith(ad.CAE_CON_PRINCIPAL),
   "como calificación razonada, no con la fórmula de la caída (el estudio no la despacha)")
ok(det[VP2].get("guarda") == "procesal", "la pantalla sabe que es la guarda procesal")
ok(any("NO se declaró caída con el principal" in a and "la pericial se ofreció en la ampliación" in a
       for a in av), "y el aviso se lo dice al secretario")
ok(cc[2]["sentido"] == "inoperante" and "el crédito no entró" in cc[2]["razonamiento"],
   "el fondo que dependía sí cae con el principal")
# Sin suerte escrita para esta vía —sólo la dependencia de la fase 3—, no hay
# calificación que aplicar: conserva la suya y se avisa.
cc = [cr(VP1, "infundado", "principal", tocado=True), cr(VP2, "fundado", razon="la pericial era idónea")]
av, det = ad.aplicar(probs(PV1, PV2), cc, [], [])
ok(cc[1]["sentido"] == "fundado" and cc[1]["razonamiento"] == "la pericial era idónea",
   "sin suerte escrita para la vía, la procesal conserva su calificación y su razón")
# Una procesal que VUELVE de la pantalla caída (reparto anterior al 26-sep): el
# estudio la reconoce por el arranque de la razón y la escribiría sin decidir.
cc = [cr(VP1, "infundado", "principal", tocado=True),
      cr(VP2, "inoperante", razon="Descansa en la premisa que se desestimó al resolver el "
                                  "problema principal, de modo que su estudio no produciría "
                                  "ningún fin práctico.")]
av, det = ad.aplicar(probs(PV1, PV2, dep=None), cc, [],
                     [{"problema": VP2, "sentido": "fundado", "alcanza": True}])
ok(cc[1]["sentido"] == "fundado" and not cc[1]["razonamiento"].startswith("Descansa"),
   f"la que llega caída se decide por sí misma, con lo que el motor le propuso: {cc[1]['sentido']}")
ok(any("NO se declaró caída con el principal" in a for a in av), "y se avisa")

print("\n13 · LO QUE EL SECRETARIO MARCÓ SE RESPETA, PERO SE LE DICE")
cc = [cr(VP1, "fundado", "principal", tocado=True), cr(VP2, "innecesario", tocado=True)]
av, det = ad.aplicar(probs(PV1, PV2), cc, [], [])
ok(cc[1]["sentido"] == "innecesario" and det[VP2]["de"] == "tuya", "su marca manda")
ok(any("la marcaste innecesario" in a and "189" in a for a in av), "con el aviso de los arts. 74-V, 174 y 189")

print("\n14 · EL FONDO QUE QUEDA VIVO CON LA REPOSICIÓN, SEÑALADO")
cc = [cr(VP1, "fundado", "principal", tocado=True), cr(FON2, "fundado")]
av, det = ad.aplicar(probs(PV1, PF2, dep=None), cc, [], [])
ok(cc[1]["sentido"] == "fundado", "no se inventa una dependencia que nadie escribió")
ok(any("EL PRINCIPAL ES UNA VIOLACIÓN PROCESAL QUE PROSPERA" in a for a in av),
   "pero se le dice que la reposición deja sin materia el fondo")

print("\n15 · SÓLO EN AMPARO DIRECTO (arts. 74-V y 174)")
cc = [cr(VP1, "fundado", "principal", tocado=True), cr(VP2, "infundado")]
av, det = ad.aplicar(probs(PV1, PV2), cc, [], [], tipo_asunto="queja")
ok(cc[1]["sentido"] == "innecesario", "en una queja el árbol sigue como estaba")
cc = [cr(VP1, "fundado", "principal", tocado=True), cr(VP2, "infundado")]
av, det = ad.aplicar(probs(PV1, PV2), cc, [], [], tipo_asunto="amparo directo")
ok(cc[1]["sentido"] == "infundado", "en amparo directo, con su grafía de siempre, rige")
# Sin `clase` manda el vocabulario de `violacion_procesal` (el mismo que decide
# si el estudio recibe la técnica procesal); no se amplía aquí sin calibrarlo.
VP3 = "¿Fue ilegal que la Sala desechara la prueba pericial ofrecida por la actora?"
cc = [cr(VP1, "fundado", "principal", tocado=True), cr(VP3, "infundado")]
av, det = ad.aplicar([{"pregunta": VP1, "jerarquia": "principal"},
                      {"pregunta": VP3, "jerarquia": "accesorio", "depende_de": 1}], cc, [], [])
ok(cc[1]["sentido"] == "infundado", "sin `clase` (sesiones viejas), la procesal se reconoce por lo que combate")
r = ad.reparto_para_pantalla(probs(PV1, PV2), [cr(VP1, "fundado", "principal", tocado=True),
                                               cr(VP2, "infundado")], [], [])
ok(r["criterios"][1]["sentido"] == "infundado" and "74-V" in r["criterios"][1]["por_que"],
   "/taller/reparto enseña por qué se estudia")

print("\n16 · EL REPARTO GLOBAL, CON LA MISMA REGLA")
fu, avm = md.repartir(probs(PV1, PV2, PF), md.GLOBAL, "fundado",
                      [{"problema": VP2, "sentido": "infundado", "razon": "razón del motor"}],
                      {}, global_dictado=True)
_d = {x["problema"]: x for x in fu}
ok(_d[VP2]["sentido"] == "infundado", f"la procesal no se declara innecesaria y toma lo que el motor le propuso: {_d[VP2]['sentido']}")
ok(_d[FON]["sentido"] == "innecesario", "el fondo sí, con la reposición")
ok(any("VIOLACIÓN PROCESAL y NO se declaró sin materia" in a for a in avm), "con su aviso")
cc2 = [cr(x["problema"], x["sentido"], x["jerarquia"], x["razonamiento"]) for x in fu]
av2, _ = ad.aplicar(probs(PV1, PV2, PF), cc2, [], [])
ok(len({a for a in avm + av2 if "VIOLACIÓN PROCESAL y NO se declaró sin materia" in a}) == 1,
   "y el árbol, que corre después, dice lo mismo con las mismas palabras (un aviso, no dos)")
fu, avm = md.repartir(probs(PF, PV1), md.GLOBAL, "fundado", [], {}, global_dictado=True)
_d = {x["problema"]: x for x in fu}
ok(_d[VP1]["sentido"] == "innecesario" and "189" in _d[VP1]["razonamiento"],
   "fondo que prospera: la procesal innecesaria por mayor beneficio, con el 189")
ok(any("POR MAYOR BENEFICIO" in a for a in avm), "y su aviso")
fu, avm = md.repartir(probs(PV1, PV2), md.GLOBAL, "fundado", [], {}, global_dictado=True,
                      tipo_asunto="amparo_revision")
ok({x["problema"]: x for x in fu}[VP2]["sentido"] == "innecesario", "en revisión, como estaba")

print("\n17 · LAS PUERTAS: el tipo llega a los dos gemelos, la propuesta y la pantalla")
_ti = 'tipo_asunto=str(getattr(getattr(r, "encargo", None), "tipo_asunto", "") or ""))'
ok(src.count(_ti) == 6, f"cuatro árboles y dos repartos globales reciben el tipo ({src.count(_ti)})")
i_st = src.find("async def taller_resolver_stream(")
i_pl = src.find("async def taller_resolver(")
ok(src[i_st:i_pl].count(_ti) == 2 and src[i_pl:i_pl + 40000].count(_ti) == 2,
   "los dos gemelos del resolver, igual: reparto global y árbol")

# ═══════════════════════════════════════════════════════════════════════════
# REVISIÓN ADVERSARIAL DEL 26-SEP-2026
# ═══════════════════════════════════════════════════════════════════════════
print("\n18 · EL CAMINO REAL: la propuesta sobrescribe y el secretario cambia el principal")
# main pasa la propuesta por el árbol y SOBRESCRIBE la propuesta guardada con lo
# que el árbol decide. Con el fondo fundado, la procesal se guardaba
# «innecesario» (189) y, al pasar el principal a infundado, ya no quedaba
# calificación que devolverle: salía «ESTÁ SIN CALIFICAR» y el estudio la
# escribía «sin materia». Se reproduce el bloque de main tal cual.
import types
import fase5_propuesta as _f5p


def _propuesta_como_main(pp, props):
    """El bloque «LA SUERTE DE LOS ACCESORIOS, YA EN LA PROPUESTA» de main,
    y lo que se guarda y se repone con la sesión (dos workers)."""
    crit = [{"problema": p.problema, "sentido": p.sentido, "razonamiento": p.razon,
             "jerarquia": "principal" if i == 0 else "accesorio"} for i, p in enumerate(props)]
    ad.aplicar(pp, crit, [], [{"problema": p.problema, "sentido": p.sentido, "alcanza": p.alcanza}
                              for p in props])
    for c, p in zip(crit, props):
        if p.alcanza and p.sentido and c["sentido"] != p.sentido:
            if not getattr(p, "sentido_propio", ""):
                p.sentido_propio = p.sentido
                p.razon_propia = p.razon
            p.sentido = c["sentido"]
            p.razon = c.get("razonamiento") or p.razon
    guardado = [{"problema": p.problema, "sentido": p.sentido, "razon": p.razon,
                 "alcanza": p.alcanza, "sentido_propio": getattr(p, "sentido_propio", "") or "",
                 "razon_propia": getattr(p, "razon_propia", "") or ""} for p in props]
    repuesto = [types.SimpleNamespace(**x) for x in guardado]
    return crit, [{"problema": p.problema, "sentido": p.sentido, "razon": p.razon,
                   "alcanza": p.alcanza, "sentido_propio": p.sentido_propio,
                   "razon_propia": p.razon_propia} for p in repuesto]


pp = probs(PF, PV1)
props = [_f5p.Propuesta(problema=FON, sentido="fundado", razon="el crédito carece de sustento"),
         _f5p.Propuesta(problema=VP1, sentido="infundado", razon="la ampliación fue extemporánea")]
crit, para_arbol = _propuesta_como_main(pp, props)
ok(crit[1]["sentido"] == "innecesario" and props[1].sentido == "innecesario",
   "en la propuesta la procesal queda innecesaria por mayor beneficio (189)")
ok(para_arbol[1]["sentido_propio"] == "infundado",
   "y lo que el motor le propuso se guarda aparte, con la sesión")
pant = [dict(crit[0], sentido="infundado", tocado=True), dict(crit[1])]
r = ad.reparto_para_pantalla(pp, pant, [], para_arbol)
ok(r["criterios"][1]["sentido"] == "infundado"
   and r["criterios"][1]["razonamiento"] == "la ampliación fue extemporánea",
   f"al pasar el principal a infundado, la procesal recupera la calificación del motor y su razón: "
   f"{r['criterios'][1]['sentido']}")
ok(not any("ESTÁ SIN CALIFICAR" in a for a in r["avisos"]), "y no queda sin calificar")
ok(r["criterios"][1]["guarda"] == "procesal", "la pantalla recibe también por qué se estudia (`guarda`)")
# Lo que el secretario tecleó antes de elegir sentido no se borra al devolverle
# la calificación del motor.
pant = [dict(crit[0], sentido="infundado", tocado=True),
        dict(crit[1], sentido="", razonamiento="la actora sí ofreció la pericial en tiempo")]
r = ad.reparto_para_pantalla(pp, pant, [], para_arbol)
ok(r["criterios"][1]["sentido"] == "infundado"
   and r["criterios"][1]["razonamiento"] == "la actora sí ofreció la pericial en tiempo",
   "la razón que tecleó el secretario se conserva")
# Una propuesta guardada ANTES de hoy puede traer la caída que el árbol le
# escribió: eso no se devuelve como si fuera una calificación.
cc = [cr(VP1, "infundado", "principal", tocado=True), cr(VP2, "innecesario")]
av, det = ad.aplicar(probs(PV1, PV2, dep=None), cc, [],
                     [{"problema": VP2, "sentido": "inoperante",
                       "razon": ad.CAE_CON_PRINCIPAL + ", de modo que su estudio no produciría ningún fin práctico."}])
ok(cc[1]["sentido"] == "innecesario" and not cc[1]["razonamiento"].startswith(ad.CAE_CON_PRINCIPAL),
   "la caída guardada de una sesión vieja no se le devuelve a la procesal como calificación")
ok(any("ESTÁ SIN CALIFICAR" in a for a in av), "y se le pide al secretario")

print("\n19 · EL AVISO NO DICE QUE PROSPERA UN PRINCIPAL QUE NO PROSPERA")
cc = [cr(VP1, "infundado", "principal", tocado=True), cr(VP2, "innecesario", razon="queda sin materia")]
av, det = ad.aplicar(probs(PV1, PV2, dep=None), cc, [], [])
_av = [a for a in av if "VIOLACIÓN PROCESAL y NO se declaró sin materia" in a]
ok(_av and not any("prospere y se reponga" in a for a in _av),
   f"con el principal procesal infundado el aviso no habla de reponer: {(_av or [''])[0][120:260]}")
cc = [cr(VP1, "fundado", "principal", tocado=True), cr(VP2, "innecesario")]
av, det = ad.aplicar(probs(PV1, PV2, dep=None), cc, [], [{"problema": VP2, "sentido": "fundado"}])
ok(any("prospere y se reponga" in a for a in av), "con el principal procesal fundado, sí")

print("\n20 · LA EXCEPCIÓN DEL 189 ESCRIBE «innecesario», QUE ES LO QUE EL ESTUDIO LEE")
cc = [cr(FON, "fundado", "principal", tocado=True), cr(VP1, "sin_materia")]
av, det = ad.aplicar(probs(PF, PV1, dep=None), cc, [], [])
ok(cc[1]["sentido"] == "innecesario" and "189" in cc[1]["razonamiento"],
   f"una procesal que llega «sin materia» con el fondo fundado queda «innecesario» con el 189: {cc[1]['sentido']}")
import violacion_procesal as _vp_t
ok("única razón" not in _vp_t.RAZON_MAYOR_BENEFICIO and "74" not in _vp_t.RAZON_MAYOR_BENEFICIO,
   "la razón del 189 es prosa de sentencia: la lección para el secretario va en el aviso")
# En modo global corren el reparto y el árbol seguidos: cada uno escribía su
# aviso del 189 con otras palabras y el secretario leía dos.
for _pp in (probs(PF, PV1), probs(PF, PV1, dep=None), probs(PF, PV1, PV2)):
    fu, avm = md.repartir(_pp, md.GLOBAL, "fundado", [], {}, global_dictado=True)
    av2, _ = ad.aplicar(_pp, [dict(x) for x in fu], [], [])
    _t = {a for a in avm + av2 if "POR MAYOR BENEFICIO" in a}
    ok(len(_t) == 1, f"modo global: un solo aviso del 189, no {len(_t)} "
                     f"({len(_pp) - 1} procesal(es), depende_de={_pp[1].get('depende_de')})")

print("\n21 · IDA Y VUELTA DEL PRINCIPAL: la procesal no se queda con la calificación de la otra vía")
# El motor propone el principal fundado y la pericial fundada; su lista dice
# que, si el principal no prospera, la pericial es inoperante (se ofreció en la
# ampliación). El secretario pasa el principal a infundado y lo devuelve a
# fundado: la pericial tiene que volver a la calificación del motor, no
# quedarse «inoperante por haberse ofrecido en la ampliación» con la ampliación
# admitida.
_lista_iv = [{"numero": 1, "papel": "principal"},
             {"numero": 2, "papel": "accesorio", "relacion": "depende",
              "si_prospera": {"sentido": "innecesario", "razon": "la reposición lo absorbe"},
              "si_no_prospera": {"sentido": "inoperante",
                                 "razon": "la pericial se ofreció en la ampliación, bien tenida por precluida"}}]
_props_iv = [{"problema": VP1, "sentido": "fundado", "razon": "r1", "alcanza": True},
             {"problema": VP2, "sentido": "fundado", "razon": "la pericial era idónea", "alcanza": True}]


def _rep_iv(c):
    return ad.reparto_para_pantalla(probs(PV1, PV2), c, _lista_iv, _props_iv,
                                    sentido_motor="fundado")["criterios"]


c = _rep_iv([cr(VP1, "fundado", "principal", "r1"), cr(VP2, "fundado", razon="la pericial era idónea")])
ok(c[1]["sentido"] == "fundado", "en la vía del motor, la pericial con su calificación")
c[0] = dict(c[0], sentido="infundado", tocado=True)
c = _rep_iv(c)
ok(c[1]["sentido"] == "inoperante" and "se ofreció en la ampliación" in c[1]["razonamiento"],
   "principal a infundado: la pericial, con lo que el motor escribió para esa vía")
c[0] = dict(c[0], sentido="fundado", tocado=True)
c = _rep_iv(c)
ok(c[1]["sentido"] == "fundado" and c[1]["razonamiento"] == "la pericial era idónea",
   f"de vuelta a fundado: la calificación del motor y su razón, no la de la otra vía: "
   f"{c[1]['sentido']} · {c[1]['razonamiento'][:40]}")

print("\n22 · main GUARDA, REPONE Y PASA LO QUE EL MOTOR PROPUSO (gemelos iguales)")
i_prop = src.find("LA SUERTE DE LOS ACCESORIOS, YA EN LA PROPUESTA")
ok(i_prop > 0 and "_p.sentido_propio = _p.sentido" in src[i_prop:i_prop + 4000],
   "la propuesta guarda lo del motor antes de sobrescribirlo")
ok(src.count('"sentido_propio": getattr(_p, "sentido_propio", "") or ""') == 4,
   "se persiste con la sesión y llega al árbol en /taller/reparto y en los dos gemelos")
ok('sentido_propio=str(x.get("sentido_propio") or "")' in src,
   "y el worker que no la calculó la repone")
ok(src[i_st:i_pl].count('"sentido_propio": getattr(_p, "sentido_propio", "") or ""') == 1
   and src[i_pl:i_pl + 40000].count('"sentido_propio": getattr(_p, "sentido_propio", "") or ""') == 1,
   "igual en /taller/resolver/stream y en /taller/resolver")

print()
if FALLOS:
    print(f"FALLAN {len(FALLOS)}: " + " · ".join(FALLOS))
    raise SystemExit(1)
print("RESULTADO: TODAS LAS COMPROBACIONES PASAN")
