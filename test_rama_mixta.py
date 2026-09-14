# -*- coding: utf-8 -*-
"""LA SENTENCIA RECURRIDA MIXTA — sobreseyó un acto y resolvió el fondo por los demás.

Amparo en revisión 322/2025: el proyecto confirmaba un sobreseimiento total y
callaba la negativa. Aquí se garantiza que la lectura la reconozca, que la rama
exista, que los puntos digan las dos cosas y que las sentencias de un solo
verbo sigan leyéndose como antes.
"""
import sys
import fase_rama as fr, tipos_asunto as ta

fallos = []
def ok(c, nota):
    print(f"  {'OK ' if c else 'MAL'} {nota}")
    if not c: fallos.append(nota)

ANT = ("En la sentencia dictada el treinta de abril de dos mil veinticinco, el Juzgado Sexto de "
       "Distrito sobreseyó en el juicio respecto de la orden de restitución de Alan, con fundamento "
       "en el artículo 63, fracción IV, de la Ley de Amparo, y negó el amparo respecto del auto de "
       "diecinueve de febrero de dos mil veinticinco y sus actos de ejecución.")
ok(fr.resolvio_a_quo("", ANT) == "sobresee_niega", "antecedentes del 322/2025 → sobresee_niega")
ok(fr.resolvio_a_quo("", "El juzgado negó el amparo respecto de los actos reclamados.") == "niega", "sólo negó → niega")
ok(fr.resolvio_a_quo("", "El juzgado sobreseyó en el juicio.") == "sobresee", "sólo sobreseyó → sobresee")
ok(fr.resolvio_a_quo("", "El juzgado sobreseyó respecto del actuario y concedió el amparo contra la orden.") == "sobresee_concede", "sobreseyó + concedió → sobresee_concede")
ok(fr.resolvio_a_quo("", "", declarado="Sobreseyó respecto de un acto y negó el amparo por los demás") == "sobresee_niega", "lo declarado también admite el mixto")
# el sustantivo dentro de una tesis no fabrica un mixto
ok(fr._mixto("SOBRESEIMIENTO. PROCEDE CUANDO… El juzgado negó el amparo.") == "", "«sobreseimiento» en un rubro no fabrica un mixto")

ok(ta.rama_revision("sobresee_niega", "infundado") == "confirma_sobresee_niega", "no prospera → confirma la mixta")
ok(ta.rama_revision("sobresee_concede", "inoperante") == "confirma_sobresee_concede", "inoperante → confirma la mixta (concede)")
ok(ta.rama_revision("sobresee_niega", "fundado") == "revoca_fondo_concede", "prospera → se revoca el fondo, no el sobreseimiento")
ok(ta.rama_revision("sobresee", "infundado") == "confirma_sobresee", "un solo verbo sigue igual")
p = ta.RAMAS_REVISION["confirma_sobresee_niega"]["puntos"]
ok(len(p) == 3 and p[1].startswith("SEGUNDO. Se sobresee") and "no ampara ni protege" in p[2], "tres puntos: confirma, sobresee, no ampara")
ok(all(k in ta.RAMAS_REVISION for k in ("confirma_sobresee_niega", "confirma_sobresee_concede")), "las dos ramas existen")

import documento_generado as dg
ok("sobreseyó respecto de un acto" in dg._VERBO_A_QUO["sobresee_niega"], "el resultando sabe decirlo")

print()
if fallos: print(f"FALLAN {len(fallos)}: " + " · ".join(fallos)); sys.exit(1)
print("TODAS LAS COMPROBACIONES PASAN")
