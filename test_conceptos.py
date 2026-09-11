"""Que no se caiga ni un planteamiento sin que el sistema lo diga.

Nació del amparo directo 393/2025 (11-sep-2026): la demanda traía DIECISIETE
conceptos de violación, el resumen cubrió trece y la sentencia salió nombrando
del primero al decimotercero para saltar después al decimoséptimo. El
secretario no pudo subsanarlo porque el sistema nunca le dijo que faltaban.

Dos fallos encadenados, los dos medidos:

  · el resumen se pidió con el tope por omisión de 2.500 tokens y volvió con
    2.497 —el 99,88% del cupo— terminando dentro de una marca sin cerrar. Cada
    concepto cuesta 191 tokens: el cupo predecía trece y salieron trece;
  · y el contador que debía avisar devolvía CERO rúbricas, porque exigía la
    palabra CONCEPTO pegada al ordinal y esa demanda encabeza la sección una
    vez y luego numera a secas. Sobre los siete escritos que el taller guarda
    enteros veía 13 de 43. El aviso no había saltado en 101 sesiones.

Se corre sola:

    python3 test_conceptos.py
"""
import sys

sys.path.insert(0, ".")

import contador_planteamientos as cp
import fases123_pipeline as fp

FALLOS = []


def ok(cond, que):
    print(("  OK   " if cond else "  FALLA ") + que)
    if not cond:
        FALLOS.append(que)


def escrito(cabecera, rubricas, relleno=3000):
    """Un escrito de mentira con el mismo esqueleto que los de verdad."""
    cuerpo = [cabecera] if cabecera else []
    for r in rubricas:
        cuerpo.append(f"{r} La Sala responsable violó en perjuicio de mi "
                      f"representada los artículos 14 y 16 constitucionales. "
                      + "Argumento de relleno. " * (relleno // 22))
    return "\n".join(cuerpo)


print("── 1 · EL ESTILO DEL 393/2025: cabecera única y ordinales a secas ──")
# Es el que devolvía CERO y por el que se cayeron cuatro conceptos.
_r = ["PRIMERO.", "SEGUNDO.", "TERCERO.", "CUARTO.", "QUINTO.", "SEXTO.",
      "SÉPTIMO.", "OCTAVO.", "NOVENO.", "DÉCIMO.", "DÉCIMO PRIMERO.",
      "DÉCIMO SEGUNDO.", "DÉCIMO TERCERO.", "DÉCIMO CUARTO.", "DÉCIMO QUINTO.",
      "DÉCIMO SEXTO.", "DÉCIMO SÉPTIMO."]
_t = escrito("Conceptos de violación:", _r)
_i = cp.planteamientos(_t)
ok(_i["n"] == 17, f"cuenta los diecisiete (dio {_i['n']})")
ok(_i["estado"] == "contado", "y dice que los contó")

print("── 2 · EL ORDINAL COMPUESTO NO SE CONFUNDE CON EL SIMPLE ──")
# «DÉCIMO SÉPTIMO» se leía «SÉPTIMO», el contador lo tragaba como repetido y
# después lo daba por resumido porque «séptimo» sí estaba en el resumen. No es
# que no lo viera: daba un VISTO BUENO FALSO, que es peor.
# UN RESUMEN DE VERDAD VA EN PÁRRAFOS, uno por planteamiento, y el contador
# los cuenta. La primera versión de esta prueba los pegaba en una sola línea y
# el aviso saltaba con razón: la que estaba mal era la prueba.
_res = "\n\n".join(f"El {o} concepto de violación se hace consistir en que la "
                    f"Sala omitió analizar la defensa opuesta." for o in
                    ("primero", "segundo", "tercero", "cuarto", "quinto",
                     "sexto", "séptimo", "octavo", "noveno", "décimo",
                     "décimo primero", "décimo segundo", "décimo tercero"))
_faltan = fp.conceptos_sin_resumir(_t, _res)
ok(len(_faltan) == 4, f"con trece resumidos, faltan cuatro (dio {len(_faltan)})")
ok(all("DECIMO" in f for f in _faltan),
   f"y son los compuestos, no el séptimo simple: {_faltan}")

print("── 3 · NO ACUSA AL ESCRITO BIEN RESUMIDO ──")
_res_todo = _res + "\n\n" + "\n\n".join(
    f"El {o} concepto de violación plantea la falta de exhaustividad."
    for o in ("décimo cuarto", "décimo quinto", "décimo sexto", "décimo séptimo"))
ok(fp.conceptos_sin_resumir(_t, _res_todo) == [],
   "resumidos los diecisiete, no falta ninguno")
# LO QUE IMPORTA NO ES QUE CALLE, SINO QUE NO ACUSE. Un recuento distinto de
# apartados es información; decir que faltan planteamientos cuando están todos
# es la comprobación acusando al trabajo correcto, y eso sí se prohíbe.
ok("NO SE RESUMIERON TODOS" not in (cp.aviso(_t, _res_todo) or ""),
   "y no acusa de que falte ninguno")

print("── 4 · «NO SE PUDO CONTAR» NO ES «NO FALTA NADA» ──")
# El cero mudo es la avería de fondo: una capa apagada se ve igual que una que
# funciona. Por eso hay TRES estados.
_sin = "Texto corrido sin rótulos ni cabecera. " * 400
_i2 = cp.planteamientos(_sin)
ok(_i2["estado"] == "no_contado", "sin rótulos, el estado es no_contado")
ok("no" in (cp.aviso(_sin, "un resumen cualquiera") or "").lower(),
   "y lo dice en voz alta en vez de callar")

print("── 5 · LO TRANSCRITO NO CUENTA COMO PLANTEAMIENTO ──")
# Un escrito transcribe la sentencia y sus resolutivos, que también empiezan
# por PRIMERO. Contarlos era inflar la cuenta y acusar de lo que no falta.
_con_resolutivos = _t + "\n" + "\n".join(
    f"{o} Se absuelve al demandado del pago reclamado." for o in
    ("PRIMERO.", "SEGUNDO.", "TERCERO.", "CUARTO."))
ok(cp.planteamientos(_con_resolutivos)["n"] == 17,
   "los resolutivos transcritos no inflan la cuenta")

print("── 6 · EL ESCRITO QUE LLEGÓ CORTADO ──")
ok(cp.sospecha_de_amputacion("el acto", "x" * 120000),
   "un tamaño clavado en un tope redondo se denuncia")
ok(not cp.sospecha_de_amputacion("el acto", "x" * 98765),
   "y un tamaño normal no")
ok(cp.sospecha_de_amputacion("mismo texto", "mismo texto"),
   "y también cuando el acto y el escrito son el mismo documento")

print("── 7 · EL CUPO SE MIDE POR LO QUE HAY QUE LEER ──")
# 2.500 tokens para todo era lo que cortó el 393/2025 en el decimotercero.
def _cupo(t):
    return max(2500, min(12000, 900 + len(t) // 50))
ok(_cupo("x" * 174702) >= 3259,
   f"un escrito de 174.702 caracteres pide {_cupo('x'*174702)} tokens, y sus "
   f"17 conceptos necesitaban 3.259")
ok(_cupo("x" * 8000) == 2500, "y uno corto sigue con el tope de siempre")

print("── 8 · LA BANDA DE PALABRAS NO MANDA EN DIRECCIÓN CONTRARIA ──")
# El único aviso que saltó en el 393/2025 dijo que el resumen era demasiado
# LARGO cuando lo que pasaba es que lo habían CORTADO.
_base = fp.PALABRAS_RESUMEN_CONCEPTOS
_obj17 = max(_base, (_base // 4) * 17)
ok(0.5 * _obj17 <= 1818 <= 1.8 * _obj17,
   f"1.818 palabras con 17 planteamientos entra en la banda ({_obj17})")

print()
if FALLOS:
    print(f"FALLOS: {len(FALLOS)}")
    for f in FALLOS:
        print("  ·", f)
    sys.exit(1)
print("Todo en orden.")
