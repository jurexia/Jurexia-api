"""Las pruebas de los hallazgos verificados el 1-sep-2026.

Cada una FALLA contra el árbol de antes del arreglo y pasa contra el de ahora.
Se corre sola:

    .venv/bin/python test_taller_hallazgos.py

Los dos hallazgos RETIRADOS —el rótulo compacto «RESUELVE» del linter— también
tienen prueba aquí, pero al revés: comprueban que la comprobación NO acusa a los
documentos correctos, que es lo que se pedía de ella.
"""
import datetime as dt
import re
import sys

sys.path.insert(0, ".")

import documento_generado as dg
import ensamblar_adelanto as ens
import fase0_oportunidad as f0
import fase_rama as fr
import linter_juridico as lj
import normas_estaticas as ne
import tipos_asunto as ta
from docx import Document

FALLOS = []


def ok(cond, que):
    print(("  ✓ " if cond else "  ✗ ") + que)
    if not cond:
        FALLOS.append(que)


def resolutivo_de(ruta):
    ps = [p.text for p in Document(ruta).paragraphs]
    k = [i for i, t in enumerate(ps) if "R E S U E L V E" in t][0]
    return ps[k + 1:k + 4]


def componer_revision(estudio, base, calif=("fundado",), ruta="/tmp/_t.docx"):
    c = f0.computar(dt.date(2025, 3, 13), dt.date(2025, 3, 25), plazo=10)
    est = dg.Estructura(apertura="V.", visto="para resolver.",
                        resultandos=[{"titulo": "Sentencia recurrida",
                                      "texto": base}],
                        competencia="", existencia="", procedencia="")
    dg.componer({"tipo_asunto": "amparo_revision", "numero": "1/2026",
                 "encabezado": "AMPARO EN REVISIÓN 1/2026",
                 "quejoso": "Juan Pérez",
                 "responsable": "el Juzgado Segundo de Distrito",
                 "magistrado": "M", "secretario": "S",
                 "tribunal": "Tercer Tribunal Colegiado", "ciudad": "Q"},
                est, c, f0.fecha_en_letra, ruta, estudio=estudio,
                calificaciones=list(calif), tipo_asunto="amparo_revision")
    return resolutivo_de(ruta), est.avisos


# ── 1. `c.oportuna = True` sobre una @property de sólo lectura ────────────
print("1. SIN PLAZO NO SE ESCRIBE SOBRE `oportuna`")
c = f0.computar(dt.date(2025, 3, 13), dt.date(2025, 4, 8), plazo=15)
try:
    c.oportuna = True
    ok(False, "la propiedad sigue sin setter (se pudo asignar: eso no debe pasar)")
except AttributeError:
    ok(True, "`oportuna` sigue siendo de sólo lectura, como debe")
for tipo, exc in (("queja", "omision_tramite"), ("amparo_directo", "vida_libertad")):
    pl = ta.plazo_de(tipo, exc)
    cc = f0.computar(dt.date(2025, 3, 13), dt.date(2025, 4, 8),
                     plazo=0 if pl["en_cualquier_tiempo"] else pl["dias"])
    ok(cc.oportuna is True and cc.en_cualquier_tiempo,
       f"{tipo}/{exc}: en cualquier tiempo → oportuna, sin AttributeError")

# ── 2. plazo 0 y plazo None ───────────────────────────────────────────────
print("2. EL CÓMPUTO NO REVIENTA CON PLAZO 0 NI None")
for p in (0, None):
    try:
        cc = f0.computar(dt.date(2025, 3, 13), dt.date(2025, 4, 8), plazo=p)
        ok(cc.sin_plazo and cc.dias == [] and cc.oportuna is True,
           f"plazo={p!r}: sin_plazo, sin días y sin extemporaneidad")
    except Exception as e:
        ok(False, f"plazo={p!r} revienta con {type(e).__name__}: {e}")
# y el cómputo de verdad sigue reproduciendo el engrose real ADA 240/2026
cc = f0.computar(dt.date(2026, 2, 23), None, "tja_qro_boletin", 15)
ok(cc.surtio == dt.date(2026, 2, 26) and cc.inicio == dt.date(2026, 2, 27)
   and cc.vencimiento == dt.date(2026, 3, 20)
   and cc.inhabiles_en_medio == [dt.date(2026, 3, 16)],
   "el ADA 240/2026 se sigue reproduciendo al día (sin regresión)")

# ── 3. el SEGUNDO resolutivo no ampara contra el órgano recurrido ─────────
print("3. NUNCA SE AMPARA CONTRA EL ÓRGANO RECURRIDO")
puntos, avisos = componer_revision(
    "El agravio es fundado.",
    "El Juez Segundo de Distrito sobreseyó en el juicio de amparo 742/2023, "
    "promovido por Juan Pérez contra la resolución de veinte de mayo.")
ok("Juzgado" not in puntos[1] and "Distrito" not in puntos[1],
   "sin originaria legible, el punto NO nombra al Juzgado de Distrito")
ok(dg.HUECO in puntos[1], "va con comodín, que se ve y el linter cuenta")
ok(any("ORIGINARIA" in a for a in avisos), "y con su aviso")

# ── 4. la originaria pasa por `_con_articulo` ─────────────────────────────
print("4. LA ORIGINARIA SALE CON SU ARTÍCULO")
puntos, _ = componer_revision(
    "El agravio es fundado.",
    "El Juez Segundo de Distrito sobreseyó en el amparo 742/2023, en el que se "
    "señaló como acto reclamado a Director de Ingresos del Municipio de "
    "Querétaro, consistente en la orden de clausura.")
ok("reclamado al Director de Ingresos" in puntos[1],
   "«a Director» → «al Director» (artículo puesto y contraído)")
puntos, _ = componer_revision(
    "El agravio es fundado.",
    "El Juez Segundo de Distrito sobreseyó en el amparo 742/2023, en el que se "
    "reclamó a Sala Regional del Centro III, consistente en la sentencia.")
ok("reclamado a la Sala Regional" in puntos[1], "y el femenino, «a la Sala»")

# ── 5. `revoca_sobreseimiento_niega` es alcanzable ────────────────────────
print("5. LA RAMA QUE REVOCA Y NIEGA EXISTE DE VERDAD")
alcanzables = {ta.rama_revision(a, s, solo_efectos=se, violacion_procesal=vp,
                                sentido_amparo=sa)
               for a in ("sobresee", "niega", "concede", "")
               for s in ("fundado", "infundado")
               for se in (False, True) for vp in (False, True)
               for sa in ("", "concede", "niega")}
ok("revoca_sobreseimiento_niega" in alcanzables, "la rama se puede alcanzar")
base = ("El Juez Segundo de Distrito sobreseyó en el amparo 742/2023, en el que "
        "se señaló como acto reclamado a Director de Ingresos, consistente en la "
        "orden de clausura.")
puntos, _ = componer_revision(
    "El agravio es fundado: el sobreseimiento fue indebido. Asumida "
    "jurisdicción, ante la ineficacia de los conceptos de violación, lo "
    "procedente es negar el amparo solicitado.", base)
ok("no ampara ni protege" in puntos[1],
   "estudio que niega → SEGUNDO que niega")
puntos, _ = componer_revision(
    "El agravio es fundado. Asumida jurisdicción, ante lo fundado de los "
    "conceptos de violación, lo procedente es conceder el amparo.", base)
ok("ampara y protege" in puntos[1] and "no ampara" not in puntos[1],
   "estudio que concede → SEGUNDO que concede")
puntos, avisos = componer_revision("El agravio es fundado.", base)
ok("ampara y protege" in puntos[1]
   and any("por omisión" in a for a in avisos),
   "estudio que calla → se concede POR OMISIÓN y se dice que lo es")

# ── 6. el JSON de la Ley de Amparo ────────────────────────────────────────
print("6. LA LEY DE AMPARO, ENTERA Y SIN MARKDOWN")
a = ne.LEY_DE_AMPARO
ok(not [k for k, v in a.items() if "###" in v],
   "ningún artículo lleva pegado el encabezado del capítulo siguiente")
ok(not [k for k, v in a.items() if len(v) == 6000],
   "ningún artículo cortado en seco a 6,000 caracteres")
ok("TRANSITORIO" not in a["271"].upper() and len(a["271"]) < 400,
   "el 271 ya no arrastra los transitorios del decreto")
for f in ("XIX", "XX", "XXI", "XXII", "XXIII"):
    ok(ne.fraccion(61, f).startswith(f + "."),
       f"artículo 61, fracción {f}: recuperable")
ok(ne.fraccion(61, "XX").lower().count("recurso") > 0,
   "la fracción XX —falta de definitividad— dice lo que debe")

# ── 7. el desenlace se ancla en el rótulo, no en el verbo ─────────────────
print("7. EL DESENLACE NO ARRANCA EN EL VERBO «resuelve»")
prosa = ("A" * 6000 + " El acto reclamado resuelve de forma terminal la "
         "controversia planteada y por eso no procede conceder el amparo. "
         + "B" * 500 + " Por lo expuesto y fundado, se: R E S U E L V E ÚNICO. "
         "Es fundado el recurso de queja.")
cola = ens._cola_de(prosa)
ok(cola.startswith("Por lo expuesto"),
   "el desenlace empieza en «Por lo expuesto», no en el «resuelve» de la prosa")
ok("controversia planteada" not in cola,
   "y no se traga el cuerpo del estudio")
# RETIRADO: el rótulo compacto. Medido sobre el propio RQC 233/2025, que lo
# escribe ESPACIADO; lo compacto en ese asunto es la prosa en versales del
# escrito de queja («AUTO … QUE RESUELVE SOBRE LA SUSPENSIÓN»).
ok(lj._RX_RESOLUTIVO.search("R E S U E L V E ÚNICO. Es fundado el recurso.")
   is not None, "el linter sigue viendo el rótulo espaciado")
ok(lj._RX_RESOLUTIVO.search(
    "AUTO DE 1 DE JULIO DE 2025, QUE RESUELVE SOBRE LA SUSPENSIÓN SOLICITADA, "
    "promovido por la parte quejosa.") is None,
   "y NO caza la prosa en versales del escrito de queja")

# ── 8. la preposición amputada no acusa a los apartados ───────────────────
print("8. UNA LETRA EN VERSALES NO ES UNA PREPOSICIÓN MUTILADA")
for correcto in ("En el apartado B las reglas del sistema acusatorio se aplican.",
                 "Según el tomo V las tesis relativas fueron superadas.",
                 "siendo el Condominio B el mismo que el Condominio Pirul.",
                 "como los fólios V el número del expediente, cuaderno y foja."):
    ok(not [q for q, _ in lj.revisar(correcto) if "amputada" in q],
       f"no acusa: «{correcto[:46]}…»")
for malo in ("Lo dispuesto n el artículo 17 de la Ley de Amparo.",
             "se advierte d la sentencia reclamada.",
             "conforme n la resolución impugnada."):
    ok([q for q, _ in lj.revisar(malo) if "amputada" in q],
       f"sí caza: «{malo[:46]}…»")

print()
if FALLOS:
    print(f"✗ {len(FALLOS)} COMPROBACIÓN(ES) FALLIDA(S):")
    for f in FALLOS:
        print("   -", f)
    sys.exit(1)
print("✓ TODAS LAS COMPROBACIONES PASAN.")
