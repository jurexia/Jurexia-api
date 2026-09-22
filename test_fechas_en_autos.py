"""Una fecha que no consta en autos no se afirma — ADC 93/2026 v3, 22-sep-2026.

CALIBRADO sobre las tres versiones reales del 93/2026 contra sus dos
documentos y la interlocutoria aportada: cero acusaciones falsas. La fecha
que parecía inventada («cuatro de agosto») estaba en la demanda.

    .venv/bin/python test_fechas_en_autos.py
"""
import inspect

import fechas_en_autos as fe

FALLOS = []


def ok(cond, que):
    print(f"   {'PASA ' if cond else 'FALLA'}  {que}")
    if not cond:
        FALLOS.append(que)


print("\n1 · LEER FECHAS EN LETRAS Y EN CIFRAS")
f = fe.fechas_de("el acuerdo de primero de julio de dos mil veinticinco, notificado el 10 de julio de 2025, "
                 "surtió efectos el cuatro de agosto de dos mil veinticinco; el treinta y uno de enero de "
                 "mil novecientos noventa y nueve; y el 04/08/2025.")
ok((2025, 7, 1) in f, "«primero de julio de dos mil veinticinco» (no se traga «acuerdo de»)")
ok((2025, 7, 10) in f, "«10 de julio de 2025»")
ok((2025, 8, 4) in f and (1999, 1, 31) in f, "«cuatro de agosto…» y «treinta y uno de enero de mil novecientos noventa y nueve»")
ok(len(f) == 4, f"cuatro fechas distintas, no {len(f)}")
ok(not fe.fechas_de("la interlocutoria de dieciocho de septiembre siguiente"), "sin año no es fecha completa")
ok((2025, 9, 18) in fe.fechas_de("de dieciocho de septiembre de dos mil veinticinco se apoyó en cuatro razones"),
   "el año se lee del prefijo aunque sigan más palabras")

print("\n2 · LO QUE NO CONSTA")
FUENTES = ["El acuerdo de primero de julio de dos mil veinticinco. Notificado el 10 de julio de 2025.",
           "Plazo del 5 al 12 de agosto de 2025."]
malas = fe.sin_respaldo("La notificación surtió efectos el cuatro de agosto de dos mil veinticinco y el acuerdo "
                        "es de primero de julio de dos mil veinticinco.", FUENTES)
ok(malas == ["cuatro de agosto de dos mil veinticinco"], f"acusa la deducida y respeta la que consta: {malas}")
ok(fe.sin_respaldo("el cuatro de agosto de dos mil veinticinco", []) == [], "sin fuentes no se acusa a nadie")
ok("FECHA(S) QUE NO CONSTAN" in fe.aviso("el cuatro de agosto de dos mil veinticinco", FUENTES), "con su aviso")
ok(fe.aviso("el primero de julio de dos mil veinticinco", FUENTES) == "", "y nada que decir cuando todo consta")

print("\n3 · LA PUERTA ESTÁ CONECTADA")
import redactor_adelanto as ra
src = inspect.getsource(ra._terminar)
ok("_fe.aviso(" in src and 'str(contexto or "")' in src.split("_fe.aviso(")[0][-900:],
   "_terminar comprueba las fechas con lo aportado entre las fuentes")
ok("contexto" in inspect.signature(ra._terminar).parameters, "y recibe el contexto")
ok(inspect.getsource(ra).count("qdrant, marco,\n") >= 2 or inspect.getsource(ra).count("marco,\n                           contexto)") + inspect.getsource(ra).count("marco,\n                          contexto)") == 2,
   "los dos redactores se lo pasan")

print()
if FALLOS:
    print(f"FALLAN {len(FALLOS)}: " + " · ".join(FALLOS))
    raise SystemExit(1)
print("RESULTADO: TODAS LAS COMPROBACIONES PASAN")
