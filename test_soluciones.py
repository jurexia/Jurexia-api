# -*- coding: utf-8 -*-
"""Las soluciones posibles de un asunto, por código (rediseño, etapa 3; decisión 6).

    .venv/bin/python test_soluciones.py
"""
import sys
import soluciones as so

FALLOS = []


def ok(c, q):
    print(f"   {'PASA ' if c else 'FALLA'}  {q}")
    if not c:
        FALLOS.append(q)


def tipos(r):
    return [s["tipo_efecto"] for s in r["soluciones"]]


print("\n1 · AMPARO DIRECTO")
r = so.posibles("amparo_directo", [{"pregunta": "¿La Sala valoró la prueba?", "clase": "fondo"}])
ok(tipos(r) == ["niega", "para_efectos"] and [s["id"] for s in r["soluciones"]] == ["S1", "S2"],
   "sólo fondo: niega y concede para efectos, en orden del código (la que no prospera primero)")
ok(not r["soluciones"][0]["prospera"] and r["soluciones"][1]["prospera"]
   and r["soluciones"][1]["desenlace"], "cada una con su lado y sus puntos resolutivos calculados por código")
r = so.posibles("amparo_directo", [{"pregunta": "¿Se violó el procedimiento al no admitir la prueba?", "clase": "procesal"},
                                   {"pregunta": "¿Prescribió la acción?", "clase": "fondo",
                                    "combate": "la prescripción de la acción cambiaria"}])
ok(tipos(r) == ["niega", "reposicion", "para_efectos", "liso_y_llano"],
   "procesal + fondo que pide mayor beneficio: las cuatro (reposición, para efectos, lisa y llana)")
ok(r["soluciones"][1]["problemas"] == [0] and r["soluciones"][3]["problemas"] == [1],
   "cada solución dice qué problemas la sostienen")
r = so.posibles("amparo_directo", [{"pregunta": "¿Procede el sobreseimiento del juicio de origen?", "clase": "fondo",
                                    "combate": "debió decretarse el sobreseimiento"}])
ok("liso_y_llano" not in tipos(r),
   "«sobreseimiento» o «improcedencia» en un civil no bastan para proponer una concesión lisa y llana")

print("\n2 · AMPARO EN REVISIÓN")
r = so.posibles("amparo_revision",
                [{"pregunta": "¿Es improcedente el juicio?", "clase": "procedencia", "combate": "x"},
                 {"pregunta": "¿Los efectos?", "clase": "fondo", "combate": "los efectos de la concesión son excesivos"}],
                resolvio_a_quo="concede", quien_recurre="autoridad")
ok(tipos(r)[:2] == ["confirma", "revoca"] and "revoca_sobresee" in tipos(r) and "modifica_efectos" in tipos(r),
   "confirma, revoca, revoca y sobresee (procedencia) y modifica efectos (sólo los efectos)")
ok(all(s["rama"] for s in r["soluciones"]), "cada una con su rama del art. 93")
r2 = so.posibles("amparo_revision", [{"pregunta": "¿Es improcedente?", "clase": "procedencia", "combate": "x"}],
                 resolvio_a_quo="concede", quien_recurre="quejoso")
ok("revoca_sobresee" not in tipos(r2), "la quejosa no pide sobreseer su propio juicio")
r3 = so.posibles("amparo_revision", [{"pregunta": "¿?", "clase": "fondo"}], resolvio_a_quo="")
ok(any("qué resolvió el juzgado" in a for a in r3["avisos"]), "sin saber qué resolvió el juzgado, lo avisa")

print("\n3 · QUEJA, REVISIÓN FISCAL Y TIPOS NO RECONOCIDOS")
ok(tipos(so.posibles("queja", [{"pregunta": "¿?"}])) == ["no_prospera", "prospera"], "queja: prospera o no")
ok(tipos(so.posibles("revision_fiscal", [{"pregunta": "¿?"}])) == ["confirma", "revoca"], "revisión fiscal")
r = so.posibles("reclamacion", [{"pregunta": "¿?"}])
ok(r["soluciones"] == [] and r["avisos"], "la reclamación no hereda el binario del amparo directo: nada y un aviso")

print("\n4 · TOPE Y DUPLICADOS")
r = so.posibles("amparo_directo", [{"pregunta": "a", "clase": "procesal"},
                                   {"pregunta": "prescripción", "clase": "fondo"}], tope=2)
ok(len(r["soluciones"]) == 2 and any("tope" in a for a in r["avisos"]),
   "con tope, se corta y se avisa qué quedó fuera (nunca en silencio)")
ok(so.posibles("amparo_directo", None)["soluciones"], "sin problemas no se rompe")

print("\n5 · LA SEGUNDA CASILLA DE LA TARJETA")
r = so.posibles("amparo_directo", [{"pregunta": "a", "clase": "procesal"}, {"pregunta": "b", "clase": "fondo"}])
S = r["soluciones"]
ok(so.lado_opuesto(S, S[1])["id"] == "S1" and so.lado_opuesto(S, S[0])["id"] == "S2",
   "la contraria es la primera del OTRO lado, no la segunda de la lista")

print()
if FALLOS:
    print("FALLAS:\n  " + "\n  ".join(FALLOS))
    sys.exit(1)
print("RESULTADO: TODAS LAS COMPROBACIONES PASAN")
