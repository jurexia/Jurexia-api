# -*- coding: utf-8 -*-
"""LA SÍNTESIS A FONDO — el objetivo sale del escrito y las tesis invocadas se cuentan.

Revisión fiscal 61/2025: 68.776 caracteres en un agravio único con ocho
jurisprudencias; el resumen salió de 147 palabras y sin una sola tesis. Aquí
se garantiza la medida, la detección de citas y que el prompt de la segunda
pasada pida lo que David pidió: cada argumento con su párrafo y cada tesis
nombrada.
"""
import sys
import fases123_pipeline as fp
from fases123_resumenes import PALABRAS_RESUMEN_CONCEPTOS, instrucciones_resumen_conceptos

fallos = []
def ok(c, nota):
    print(f"  {'OK ' if c else 'MAL'} {nota}")
    if not c: fallos.append(nota)

ok(fp.objetivo_conceptos("x" * 3000) == PALABRAS_RESUMEN_CONCEPTOS, "un escrito breve pide la mediana del corpus")
ok(1200 <= fp.objetivo_conceptos("x" * 68776) <= 1300, f"68.776 caracteres piden ~1.250 palabras ({fp.objetivo_conceptos('x' * 68776)})")
ok(fp.objetivo_conceptos("x" * 900000) == fp.PALABRAS_CONCEPTOS_TOPE, "y hay tope")

ESCRITO = ("…inaplicó la jurisprudencia 2a./J. 13/2015 (10a.), de registro digital 2008474, "
           "y la diversa 2a./J. 60/2007 ( ** ) citada en la contestación; Registro: 172239. "
           "También la tesis P./J. 44/2014 (10a.) y la I.4o.A. J/12, registro digital 2012284.")
todas = fp.citas_invocadas(ESCRITO)
ok("2A./J. 13/2015" in todas and "2A./J. 60/2007" in todas, f"lee las claves de jurisprudencia: {sorted(todas)}")
ok("2008474" in todas and "172239" in todas and "2012284" in todas, "y los registros")
ok("P./J. 44/2014" in todas, "y las del Pleno")

RES_MUDO = "En el único agravio la autoridad sostiene que procede revocar. Ofrece pruebas."
_, sin = fp.citas_sin_nombrar(ESCRITO, RES_MUDO)
ok(len(sin) == len(todas), "un resumen que no nombra ninguna las debe todas")
RES_OK = ("En el único agravio la autoridad sostiene que la Sala inaplicó la jurisprudencia 2a./J. 13/2015 "
          "(registro 2008474) y la 2a./J. 60/2007 (registro 172239); invoca además la P./J. 44/2014 y la "
          "I.4o.A. J/12, registro 2012284.")
_, sin2 = fp.citas_sin_nombrar(ESCRITO, RES_OK)
ok(not sin2, f"un resumen que las nombra no debe ninguna ({sorted(sin2)})")

pr = fp.prompt_conceptos_a_fondo(ESCRITO, RES_MUDO, sorted(sin), 1250, es_recurso=True, tipo_asunto="revision_fiscal")
ok("SE QUEDÓ CORTO" in pr and "1250" in pr, "la segunda pasada dice cuánto falta")
ok("2A./J. 13/2015" in pr, "y nombra las tesis pendientes")
ok("Reescribe el resumen" in pr and "ENTERO" in pr, "y pide el resumen entero, no un apéndice")
ins = instrucciones_resumen_conceptos(True, "revision_fiscal", 1250)
ok("UN PÁRRAFO POR CADA ARGUMENTO DISTINTO" in ins, "las instrucciones piden un párrafo por argumento")
ok("SE NOMBRAN" in ins and "NO SE RESUME" in ins, "nombrar las tesis; no resumir pruebas ni delegados")
ok("1250 palabras" in ins, "y la extensión viene del escrito")
ok("472 palabras" in instrucciones_resumen_conceptos(False, "amparo_directo"), "sin medida, la mediana de siempre")

# ── y la sentencia, igual ──
ok(fp.objetivo_acto("x" * 2000) == 438, "un acto breve pide la mediana del corpus (438)")
ACTO = "CONSIDERANDO. " + ("La Sala consideró que la notificación fue ilegal con apoyo en la jurisprudencia 2a./J. 13/2015, registro 2008474. " * 400)
ok(fp.objetivo_acto(ACTO) > 438, f"un acto largo pide más ({fp.objetivo_acto(ACTO)})")
pa = fp.prompt_acto_a_fondo(ACTO, "La Sala declaró la nulidad.", ["2A./J. 13/2015", "2008474"], 700, es_recurso=True, tipo_asunto="revision_fiscal")
ok("SE QUEDÓ CORTO" in pa and "2A./J. 13/2015" in pa and "Reescribe el resumen ENTERO" in pa, "la segunda pasada del acto pide el resumen entero con las tesis")
from fases123_resumenes import instrucciones_resumen_acto
ia = instrucciones_resumen_acto("revision_fiscal", 700)
ok("COMPLETO, NO ESCOGIDO" in ia and "SE NOMBRAN" in ia and "700 palabras" in ia, "las instrucciones del acto piden todas las consideraciones y las tesis")

print()
if fallos: print(f"FALLAN {len(fallos)}: " + " · ".join(fallos)); sys.exit(1)
print("TODAS LAS COMPROBACIONES PASAN")
