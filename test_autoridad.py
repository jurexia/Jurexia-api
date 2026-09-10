"""La autoridad responsable, contra el banco de nombres reales.

Nació del resolutivo del 91/2025 del 10-sep-2026, que decía «dictada por la
Sala Regional en Querétaro Infrazione Administrat Administración Desconce».
Esa cadena no era un encabezado: era el campo «DEPENDENCIA:» de una carátula de
notificación donde el OCR entrelazó dos columnas de un formulario.

La causa medida fue el tope en caracteres del patrón, que hacía dos daños
opuestos a la vez —tragaba prosa Y cortaba el nombre bueno a 73 caracteres—, y
con eso fabricaba un empate que `max(cands, key=len)` resolvía por posición: la
basura de la página 5 ganaba al nombre bueno de la página 7.

Se corre sola:

    python3 test_autoridad.py
"""
import sys
import time
import unicodedata
from collections import Counter

sys.path.insert(0, ".")

import banco_autoridad as banco
import fase_autoridad as fa

FALLOS = []

# La carátula con la que se envuelve cada nombre: es la forma en que un acto
# reclamado se identifica a sí mismo, y la escritura en VERSALES importa porque
# los encabezados van así y el cuerpo no.
CARATULA = ("{n}\nEXPEDIENTE: 695/25-09-01-7-OT\n"
            "ACTOR: JUAN PÉREZ LÓPEZ\n"
            "Santiago de Querétaro, a veintidós de septiembre de dos mil "
            "veinticinco.\nVistos para resolver los autos.\n{n} resuelve.\n")


def plano(x):
    """Sin tildes ni comas: comparar por ellas sería acusar por una tilde."""
    x = unicodedata.normalize("NFD", x or "")
    x = "".join(c for c in x if unicodedata.category(c) != "Mn")
    return " ".join(x.lower().replace(",", " ").split())


def veredicto(salida, esperado):
    s, e = plano(salida), plano(esperado)
    if not s and not e:
        return "OK-HUECO"
    if not s:
        return "HUECO"        # se perdió un nombre bueno
    if not e:
        return "NOMBRE-MALO"  # debía ser hueco y dio nombre
    if s == e:
        return "OK"
    if e.startswith(s):
        return "AMPUTADO"     # el hueco que NO se ve
    if s.startswith(e):
        return "SOBRA"
    return "OTRO"


def ok(cond, que):
    print(("  OK   " if cond else "  FALLA ") + que)
    if not cond:
        FALLOS.append(que)


print("── 1 · LOS NOMBRES LEGÍTIMOS, EN VERSALES Y EN MINÚSCULA ──")
legit = Counter()
rotos = []
for n, tipo in banco.LEGITIMOS:
    for como in ("VERSALES", "normal"):
        txt = CARATULA.format(n=n.upper() if como == "VERSALES" else n)
        out = fa.de_texto(txt, tipo)
        v = veredicto(out, n)
        legit[v] += 1
        if v != "OK":
            rotos.append(f"[{como}] {n!r} → {out!r} ({v})")
_tot = sum(legit.values())
print(f"     exactos {legit['OK']}/{_tot} · amputados {legit['AMPUTADO']} · "
      f"huecos {legit['HUECO']}")
for r in rotos[:6]:
    print("       ·", r)
ok(legit["OK"] == _tot, f"los {_tot} nombres legítimos salen enteros")
# EL AMPUTADO ES EL PEOR DE LOS FALLOS y merece su propia comprobación: un
# nombre al que le falta el final es creíble y falso, y nadie lo relee.
ok(legit["AMPUTADO"] == 0, "ninguno sale cortado a mitad de palabra")

print("── 2 · EL ADVERSARIO: prefiere el hueco al nombre equivocado ──")
adv = Counter()
malos = []
t0 = time.time()
for cid, txt, tipo, esp in banco.ADV:
    if esp is None:
        continue
    out = fa.de_texto(txt, tipo)
    v = veredicto(out, esp)
    adv[v] += 1
    if v in ("NOMBRE-MALO", "AMPUTADO", "SOBRA", "OTRO"):
        malos.append(f"{cid}: esperado {esp!r} → {out!r} ({v})")
_ms = (time.time() - t0) * 1000
print(f"     aciertos {adv['OK']} · huecos correctos {adv['OK-HUECO']} · "
      f"nombres equivocados {len(malos)} · {_ms:.0f} ms")
for m in malos[:6]:
    print("       ·", m)
ok(not malos, "CERO nombres equivocados en el adversario")

print("── 3 · EL 91/2025, SOBRE SU OCR REAL SI ESTÁ A MANO ──")
# El OCR de origen son 72.647 caracteres y no se versiona; cuando no está, esta
# comprobación se salta en vez de mentir diciendo que pasó.
import os
_ruta = os.path.join(os.path.dirname(__file__), "banco_acto91.txt")
if os.path.exists(_ruta):
    with open(_ruta, encoding="utf-8") as fh:
        _out = fa.de_texto(fh.read(), "revision_fiscal")
    ok(_out == "Sala Regional en Querétaro del Tribunal Federal de Justicia "
               "Administrativa",
       f"el OCR real devuelve el nombre entero (dio {_out!r})")
else:
    print("     (saltada: no está banco_acto91.txt)")

print("── 4 · EL CAMINO DEL FORMULARIO: se avisa, nunca se borra ──")
for sello in ("JUZGADO", "juzgado", "juez", "JUNTA", "Junta federal"):
    ok(fa.sello_sin_identidad(sello), f"«{sello}» se caza como sello sin identidad")
for bueno in ("Legislatura del Estado de Querétaro", "Agencia de Movilidad",
              "Junta Especial Número 50 de la Federal de Conciliación y Arbitraje",
              "Sala Regional en Querétaro del Tribunal Federal de Justicia Administrativa"):
    ok(not fa.sello_sin_identidad(bueno),
       f"«{bueno[:42]}…» NO se acusa")
ok(fa._nunca_responsable("la Segunda Sala de la Suprema Corte de Justicia de la Nación"),
   "la Suprema Corte no puede ser la responsable")
ok(not fa._nunca_responsable("Tribunal Colegiado de Apelación del Vigésimo Segundo Circuito"),
   "pero el Colegiado DE APELACIÓN sí lo es en amparo penal")

print()
if FALLOS:
    print(f"FALLOS: {len(FALLOS)}")
    for f in FALLOS:
        print("  ·", f)
    sys.exit(1)
print("Todo en orden.")
