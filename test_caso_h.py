# -*- coding: utf-8 -*-
"""El caso (h): la sustracción de materia en modo global declaraba innecesario
todo accesorio no marcado, dependiera o no del principal (rediseño, etapa 4;
bandera «exclusiones_con_prueba»).

    .venv/bin/python test_caso_h.py
"""
import sys
import contexto_taller as ct
import modos_decision as md

FALLOS = []


def ok(c, q):
    print(f"   {'PASA ' if c else 'FALLA'}  {q}")
    if not c:
        FALLOS.append(q)


P = [{"pregunta": "¿El principal?", "jerarquia": "principal", "clase": "fondo"},
     {"pregunta": "¿Un tema independiente?", "jerarquia": "accesorio", "clase": "fondo"},
     {"pregunta": "¿Un tema que depende?", "jerarquia": "accesorio", "clase": "fondo", "depende_de": 1},
     {"pregunta": "¿Un tema sin relación dicha, con propuesta?", "jerarquia": "accesorio", "clase": "fondo"},
     {"pregunta": "¿Un tema sin relación ni propuesta?", "jerarquia": "accesorio", "clase": "fondo"}]
PROPS = [{"problema": "¿El principal?", "sentido": "fundado", "razon": "r", "alcanza": True},
         {"problema": "¿Un tema independiente?", "sentido": "infundado", "razon": "no prospera por su cuenta"},
         {"problema": "¿Un tema sin relación dicha, con propuesta?", "sentido": "inoperante", "razon": "no combate"}]
LISTA = [{"numero": 2, "relacion": "independiente"}]          # sin «tema_distinto»: la marca que se leía


def correr():
    rep, av = md.repartir(P, md.GLOBAL, "fundado", PROPS, {}, temas_distintos=set(),
                          tipo_asunto="amparo_directo", checklist=LISTA)
    return {x["problema"]: x for x in rep}, av


print("\n1 · HOY (sin la bandera): el caso (h) reproducido")
ct.poner(False, {})
R, av = correr()
ok(R["¿Un tema independiente?"]["sentido"] == md.INNECESARIO,
   "el tema que la lista del motor dice INDEPENDIENTE queda innecesario (el defecto)")
ok(all(R[k]["sentido"] == md.INNECESARIO for k in R if k != "¿El principal?"),
   "todo accesorio no marcado queda sin materia, dependa o no (paridad: así sigue sin la bandera)")

print("\n2 · CON LA BANDERA")
ct.poner(True, {}, pruebas=True)
R, av = correr()
ok(R["¿Un tema independiente?"]["sentido"] == "infundado",
   "el independiente se ESTUDIA con la calificación que el motor le propuso (no se sustrae)")
ok(R["¿Un tema que depende?"]["sentido"] == md.INNECESARIO, "el que depende (depende_de) se sustrae como siempre")
ok(R["¿Un tema sin relación dicha, con propuesta?"]["sentido"] == "inoperante",
   "sin relación dicha, conserva la calificación que el motor le propuso (no la brocha ni la sustracción)")
_s = R["¿Un tema sin relación ni propuesta?"]
ok(_s["sentido"] == md.INNECESARIO and _s.get("justificacion_pendiente") is True,
   "sin relación ni propuesta: se sustrae como hoy, pero marcado para justificar")
ok(any("SIN que conste que dependa" in a for a in av) and any("independiente del principal" in a for a in av),
   "y lo dice en los avisos")
ok(all(str(x.get("sentido")).strip() for x in R.values()),
   "ningún problema se queda sin sentido (el criterio descarta los vacíos)")
_rep_d, _av_d = md.repartir(P, md.GLOBAL, "fundado", PROPS, {}, global_dictado=True, temas_distintos=set(),
                            tipo_asunto="amparo_directo", checklist=LISTA)
_Rd = {x["problema"]: x for x in _rep_d}
_d = _Rd["¿Un tema sin relación dicha, con propuesta?"]
ok(_d["sentido"] == md.INNECESARIO and _d.get("justificacion_pendiente") is True
   and _Rd["¿Un tema independiente?"]["sentido"] != md.INNECESARIO,
   "con el global DICTADO, lo que el motor propuso no pasa por delante de su dictado: se sustrae como hoy, "
   "marcado para justificar (el independiente se sigue estudiando)")
ct.poner(False, {})

print()
if FALLOS:
    print("FALLAS:\n  " + "\n  ".join(FALLOS))
    sys.exit(1)
print("RESULTADO: TODAS LAS COMPROBACIONES PASAN")
