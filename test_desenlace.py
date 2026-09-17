"""El desenlace lo dicta la tarjeta final — revisión fiscal 2/2026, 17-sep-2026.

El proyecto confirmaba en el estudio y en la síntesis y revocaba en el cierre y
en los resolutivos. La tarjeta final decía infundado; el tercer planteamiento
viajó «fundado» cuando la propia tarjeta lo llamaba «fundado, pero
insuficiente».

    .venv/bin/python test_desenlace.py
"""
import inspect

import desenlace as dz
import tipos_asunto as ta

FALLOS = []


def ok(cond, que):
    print(f"   {'PASA ' if cond else 'FALLA'}  {que}")
    if not cond:
        FALLOS.append(que)


print("\n1 · LA TARJETA DICE QUE NO PROSPERA: el fundado es insuficiente")
props = [
    {"problema": "¿Los preceptos invocados facultaban a la Subdelegación?", "sentido": "infundado", "jerarquia": "principal"},
    {"problema": "¿La sentencia vulneró congruencia y exhaustividad?", "sentido": "infundado", "jerarquia": "accesorio"},
    {"problema": "¿La Sala podía invocar las resoluciones dictadas en otros juicios?", "sentido": "fundado", "jerarquia": "accesorio"},
]
av = dz.reconciliar(props, "infundado")
ok(props[2]["sentido"] == "fundado_insuficiente", "el accesorio fundado pasa a fundado pero insuficiente")
ok([p["sentido"] for p in props[:2]] == ["infundado", "infundado"], "los demás no se tocan")
ok(len(av) == 1 and "FUNDADO PERO INSUFICIENTE" in av[0], "y queda dicho en un aviso")
ok(not any(ta.prospera(p["sentido"]) for p in props),
   "ninguna calificación prospera: el cierre y los resolutivos confirmarán")
ok("confirmar" in ta.parrafo_cierre("revision_fiscal", any(ta.prospera(p["sentido"]) for p in props)),
   "el cierre dice «lo procedente es confirmar»")

print("\n2 · LA TARJETA DICE QUE PROSPERA y nadie prospera: el principal lleva el global")
props2 = [{"problema": "principal", "sentido": "infundado", "jerarquia": "principal"},
          {"problema": "accesorio", "sentido": "inoperante", "jerarquia": "accesorio"}]
av2 = dz.reconciliar(props2, "fundado")
ok(props2[0]["sentido"] == "fundado" and props2[1]["sentido"] == "inoperante",
   "el principal pasa a fundado; el accesorio queda como estaba")
ok(av2 and "PROSPERA" in av2[0], "con su aviso")

print("\n3 · LO QUE NO SE TOCA")
props3 = [{"problema": "a", "sentido": "infundado"}, {"problema": "b", "sentido": "inoperante"}]
ok(dz.reconciliar(props3, "infundado") == [] and props3[0]["sentido"] == "infundado",
   "tarjeta y calificaciones ya coherentes: nada que hacer")
props4 = [{"problema": "a", "sentido": "fundado"}, {"problema": "b", "sentido": "infundado"}]
ok(dz.reconciliar(props4, "") == [] and props4[0]["sentido"] == "fundado",
   "sin tarjeta global (vía por problema): las marcas del secretario mandan")
props5 = [{"problema": "a", "sentido": "fundado"}, {"problema": "b", "sentido": "infundado"}]
ok(dz.reconciliar(props5, "fundado") == [] and props5[0]["sentido"] == "fundado",
   "tarjeta que prospera con un fundado: nada que hacer")


class _Crit:
    def __init__(self, problema, sentido):
        self.problema, self.sentido = problema, sentido


crits = [_Crit("principal", "infundado"), _Crit("accesorio", "esencialmente_fundado")]
dz.reconciliar(crits, "infundado", {"principal": "principal", "accesorio": "accesorio"})
ok(crits[1].sentido == "fundado_insuficiente", "también sobre objetos Criterio, no sólo dicts")

print("\n4 · LA COMPROBACIÓN SOBRE EL DOCUMENTO")
V5 = (
    "SEXTO. Estudio de los agravios. Los agravios son en parte infundados y en parte fundados.\n"
    "Por ello, se confirma la sentencia de siete de octubre de dos mil veinticinco, sin que "
    "proceda ordenar a la Sala responsable el dictado de una nueva resolución.\n"
    "En ese sentido, al resultar en parte infundados y en parte fundados lo planteado, lo "
    "procedente es revocar la sentencia recurrida.\n"
    "R E S U E L V E\n"
    "PRIMERO. Se revoca la sentencia de siete de octubre de dos mil veinticinco.\n"
    "SEGUNDO. Se ordena a la SALA REGIONAL EN QUERÉTARO dictar otra sentencia.\n"
    "SÍNTESIS\n"
    "Justificación: una referencia imprecisa a antecedentes jurisdiccionales sólo constituye "
    "un argumento accesorio y no justifica revocarla.\n")
av5 = dz.contradicciones(V5, "revision_fiscal")
ok(bool(av5) and av5[0].startswith("NO FIRMABLE"), "el proyecto del 2/2026 se acusa como no firmable")
z = dz._zonas(V5)
ok([p for p, _ in dz._dichos(z["estudio"])] == [False, True],
   "en el estudio ve el «se confirma» Y el «lo procedente es revocar» del cierre")
ok([p for p, _ in dz._dichos(z["sintesis"])] == [False],
   "y en la síntesis ve el «no justifica revocarla» (un «argumento accesorio» no es una parte)")

CONFIRMA = ("SEXTO. Estudio de los agravios. Son infundados.\n"
            "La autoridad recurrente solicita que se revoque la sentencia y sostiene que procede "
            "revocar el fallo; no le asiste razón. Por tanto, lo procedente es confirmar la "
            "sentencia recurrida.\nR E S U E L V E\nÚNICO. Se confirma la sentencia recurrida.\n")
ok(dz.contradicciones(CONFIRMA, "revision_fiscal") == [],
   "un proyecto que confirma de punta a punta, con la recurrente pidiendo revocar: limpio")
REVOCA = ("SEXTO. Estudio de los agravios. Son fundados.\nPor tanto, lo procedente es revocar la "
          "sentencia recurrida.\nR E S U E L V E\nPRIMERO. Se revoca la sentencia recurrida.\n"
          "SÍNTESIS\nPropuesta de resolución: procede revocar la sentencia.\n")
ok(dz.contradicciones(REVOCA, "revision_fiscal") == [], "uno que revoca de punta a punta: limpio")
ok(dz.contradicciones(V5, "amparo_directo") == [],
   "el amparo directo no pasa por aquí: tiene sus fórmulas en revisar_congruencia")

print("\n5 · LAS PUERTAS ESTÁN CONECTADAS")
src = open("main.py", encoding="utf-8").read()
ok("_dz.reconciliar(propuestas, glob.sentido" in src, "la fase que propone reconcilia antes de guardar")
ok(src.count("_dz.reconciliar(crit, _tarjeta") == 2, "los dos gemelos del resolver reconcilian")
ok(src.count('if criterios_json.strip() and not (modo_decision or "").strip().lower() == "global":') == 2,
   "y los dos deciden igual cuando hay sentido global")
import redactor_adelanto as ra
ok("_dz_f.contradicciones(" in inspect.getsource(ra._terminar),
   "el documento final se comprueba en _terminar")

print()
if FALLOS:
    print(f"FALLAN {len(FALLOS)}: " + " · ".join(FALLOS))
    raise SystemExit(1)
print("RESULTADO: TODAS LAS COMPROBACIONES PASAN")
