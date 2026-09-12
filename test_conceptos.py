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

# ═══════════════════════════════════════════════════════════════════════════
# EL 93/2026: LA CABECERA CON ARTÍCULO Y LA RÚBRICA ABREVIADA
# ═══════════════════════════════════════════════════════════════════════════
# Medido sobre el escrito real (63,785 caracteres de OCR). Dos rasgos, los dos
# corrientes y los dos ciegos para el contador:
#
#   · la sección se rotula «IX. Los conceptos de violación:» —con artículo—, y
#     el detector de cabeceras exigía la frase pegada al ordinal;
#   · y ya rotulada la sección, las rúbricas se abrevian a «PRIMER CONCEPTO. -»
#     sin repetir «de violación», así que ni siquiera llegaban a candidatas.
#
# Resultado: 26 candidatas en todo el escrito, ninguna un concepto, y un
# «no_contado» sobre una demanda que rotula sus dos conceptos con claridad.
print("\n── el 93/2026: cabecera con artículo y rúbrica abreviada ──")

_c93 = ("IX. Los conceptos de violación:\n"
        "PRIMER CONCEPTO. - La autoridad responsable, en su considerando "
        "cuarto, violó en perjuicio de esta parte actora lo dispuesto por el "
        "artículo 17 de la Ley Federal de Procedimiento Contencioso "
        "Administrativo, al tener por precluido el derecho a ampliar. " +
        ("Relleno del desarrollo argumentativo del planteamiento. " * 90) +
        "\nSEGUNDO CONCEPTO. - La sentencia que en esta ocasión se reclama es "
        "inconstitucional al transgredir lo dispuesto por los artículos 14 y "
        "16 constitucionales, pues omitió pronunciarse sobre los alegatos. " +
        ("Relleno del desarrollo argumentativo del planteamiento. " * 90))

_r93 = cp.planteamientos(_c93)
ok(_r93["estado"] == "contado", "el escrito con artículo en la cabecera se cuenta")
ok(_r93["n"] == 2, "cuenta DOS conceptos (salió %d)" % _r93["n"])
ok(_r93["valores"] == [1, 2], "y en orden: %s" % _r93["valores"])

# Con el proemio delante, que es como llega de verdad: la cabecera está a
# veinte mil caracteres del principio y aun así manda.
_proemio = ("EXPEDIENTE: 765/25-09-01-8-ST. QUEJOSO: UNA SOCIEDAD. " * 300)
_r93b = cp.planteamientos(_proemio + "\n" + _c93)
ok(_r93b["n"] == 2, "con proemio delante sigue contando dos (salió %d)" % _r93b["n"])

# ── CALIBRACIÓN: lo abreviado NO abre la vía explícita ──────────────────
# «PRIMER CONCEPTO» suelto aparece dentro de las tesis que la propia demanda
# transcribe. Si esa forma abreviada se autoidentificara, contaría como
# planteamiento propio sin cabecera que lo respalde — el fallo del 91/2025.
_transcrito = ("La Segunda Sala ha sostenido lo siguiente. " +
               ("Texto de la tesis transcrita por la parte. " * 40) +
               "\nPRIMER CONCEPTO. - dice la ejecutoria que se copia. " +
               ("Mas texto de la tesis transcrita. " * 40) +
               "\nSEGUNDO CONCEPTO. - continua la copia de la ejecutoria. " +
               ("Mas texto de la tesis transcrita. " * 40))
_rt = cp.planteamientos(_transcrito)
ok(_rt["estado"] == "no_contado",
   "sin cabecera, la rúbrica abreviada NO cuenta sola (salió %s, n=%d)"
   % (_rt["estado"], _rt["n"]))

# ── CALIBRACIÓN: el artículo no convierte una frase corrida en cabecera ──
_corrido = ("Situación que será corroborada por este H. Tribunal Colegiado de "
            "la lectura realizada a los conceptos de violación que se "
            "desahogan enseguida, y por ello procede. " * 3)
ok(len(cp._RX_CABECERA.findall(cp._pelar(_corrido))) == 0,
   "«los conceptos de violación» en mitad de un párrafo no es cabecera")


if FALLOS:
    print(f"FALLOS: {len(FALLOS)}")
    for f in FALLOS:
        print("  ·", f)
    sys.exit(1)
print("Todo en orden.")
