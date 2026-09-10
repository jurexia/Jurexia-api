"""El resolutivo no puede contradecir a su propio documento.

Medido el 10-sep-2026 conduciendo la revisión fiscal 91/2025 en pantalla: el
proyecto salió diciendo «ÚNICO. Se confirma la sentencia de TRES DE NOVIEMBRE de
dos mil veinticinco» mientras sus resultandos y considerandos fechaban esa misma
sentencia el VEINTIDÓS DE SEPTIEMBRE, cuatro veces.

La causa no estaba en el lector. `fase_origen.datos_del_documento` busca el
proemio en los primeros 4.000 caracteres, y con la depuración el segmento que
llega como «acto reclamado» ya no empieza en el proemio de la sentencia —el OCR
de esa corrida dio 33 páginas donde la sentencia de la Sala tiene 18—. Leyó bien
el proemio que tenía delante; lo que tenía delante era otro documento.

Por eso no se arregla eligiendo lector: se usa la DISCREPANCIA como señal.

Se corre sola:

    python3 test_resolutivo_congruencia.py
"""
import sys

sys.path.insert(0, ".")

import fase_origen as fo

FALLOS = []


def ok(cond, que):
    print(("  OK   " if cond else "  FALLA ") + que)
    if not cond:
        FALLOS.append(que)


# El cuerpo del proyecto tal como lo escribió el modelo en la corrida real.
PROSA_91 = (
    "PRIMERO. Trámite del juicio contencioso administrativo. La actora demandó "
    "la nulidad de las resoluciones. SEGUNDO. Interposición del recurso de "
    "revisión fiscal. El veintinueve de octubre de dos mil veinticinco, la "
    "autoridad recurrente interpuso recurso de revisión fiscal contra la "
    "sentencia recurrida de veintidós de septiembre de dos mil veinticinco, "
    "dictada por la Sala Regional en Querétaro.")

# Lo que el lector determinista sacó del PDF depurado, verificado en los
# registros del servidor: «📑 origen leído del PDF: … fecha tres de noviembre».
PDF_91 = "tres de noviembre de dos mil veinticinco"

print("── la discrepancia se convierte en hueco, no en una fecha inventada ──")
_f, _av = fo.fecha_del_recurrido(PDF_91, PROSA_91)
ok(_f == "", "91/2025 · dos lectores que discrepan dejan la fecha en hueco")
ok("tres de noviembre" in _av and "veintidós de septiembre" in _av,
   "91/2025 · el aviso escribe las DOS fechas, para elegir en cinco segundos")

print("── y cuando coinciden, entra con doble apoyo ──")
_f, _av = fo.fecha_del_recurrido(
    "veintidós de septiembre de dos mil veinticinco", PROSA_91)
ok(_f == "veintidós de septiembre de dos mil veinticinco",
   "coinciden · la fecha se escribe en el resolutivo")
ok(_av == "", "coinciden · sin aviso que distraiga")

print("── una tilde no es una discrepancia ──")
# El OCR y la prosa no siempre acentúan igual. Abrir un hueco por una tilde
# sería la comprobación acusando al trabajo correcto.
_f, _av = fo.fecha_del_recurrido(
    "veintidos de septiembre de dos mil veinticinco", PROSA_91)
ok(_f != "" and _av == "", "«veintidos» y «veintidós» son la misma fecha")

print("── con un solo lector se sigue como antes ──")
ok(fo.fecha_del_recurrido(PDF_91, "sin fechas")[0] == PDF_91,
   "sólo el PDF · se usa el PDF")
ok(fo.fecha_del_recurrido("", PROSA_91)[0]
   == "veintidós de septiembre de dos mil veinticinco",
   "sólo la prosa · se usa la prosa")
ok(fo.fecha_del_recurrido("", "nada") == ("", ""),
   "ninguno · hueco silencioso, sin aviso nuevo")

print("── y el lector del PDF no se tocó ──")
_d = fo.datos_del_documento(
    "EXPEDIENTE: 695/25-09-01-7-OT Santiago de Querétaro, a veintidós de "
    "septiembre de dos mil veinticinco. VISTOS los autos del expediente "
    "695/25-09-01-7-OT para resolver.")
ok(_d["fecha"].startswith("veintidós de septiembre"),
   "datos_del_documento sigue leyendo el proemio")
ok(_d["expediente"] == "695/25-09-01-7-OT",
   "datos_del_documento sigue leyendo el expediente del TFJA")

print()
if FALLOS:
    print(f"FALLOS: {len(FALLOS)}")
    for f in FALLOS:
        print("  ·", f)
    sys.exit(1)
print("Todo en orden.")
