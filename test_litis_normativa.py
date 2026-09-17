"""La ley local sólo entra si está en la litis — revisión fiscal 2/2026.

El caso que lo motivó va literal: el marco jurídico de un asunto FEDERAL citó
«el artículo 57 de la Ley de Procedimiento Contencioso Administrativo del Estado
de Querétaro» y el 55 «de la propia ley», que son copia del 51 y el 50 de la
LFPCA. La primera versión de la guarda dio CERO detecciones sobre ese texto —el
recorte del nombre cortaba «del Estado» por un límite de palabra que faltaba—,
así que la prueba empieza por ahí: si no atrapa el caso real, no sirve.

    .venv/bin/python test_litis_normativa.py
"""
import inspect

import litis_normativa as ln

FALLOS = []


def ok(cond, que):
    print(f"   {'PASA ' if cond else 'FALLA'}  {que}")
    if not cond:
        FALLOS.append(que)


# El pasaje del marco jurídico del proyecto 2/2026, tal como salió.
MARCO_2_2026 = (
    "La legalidad de una resolución administrativa se examina, en primer término, a partir "
    "de la competencia de la autoridad que la emite. El artículo 16 de la Constitución "
    "Política de los Estados Unidos Mexicanos exige que todo acto de molestia conste en "
    "mandamiento escrito de autoridad competente.\n"
    "En el ámbito local, el artículo 57 de la Ley de Procedimiento Contencioso "
    "Administrativo del Estado de Querétaro establece:\n"
    "“Artículo 57. Se declarará que una resolución administrativa es ilegal cuando se "
    "demuestre alguna de las siguientes causales:\n"
    "I. Incompetencia del funcionario que la haya dictado, ordenado o tramitado el "
    "procedimiento del que deriva dicha resolución;\n"
    "II. Omisión de los requisitos formales exigidos por las leyes, siempre que afecte las "
    "defensas del particular y trascienda al sentido de la resolución impugnada, inclusive "
    "la ausencia de fundamentación o motivación, en su caso;”.\n"
    "De esta disposición se desprende que la competencia material de la emisora constituye "
    "un presupuesto de validez del acto administrativo.\n"
    "La facultad del órgano jurisdiccional para examinar esa cuestión se relaciona con el "
    "artículo 55 de la propia ley, cuyo texto dispone:\n"
    "“Artículo 55. Las sentencias del Tribunal se fundarán en derecho y resolverán sobre la "
    "pretensión del actor que se deduzca de su demanda, en relación con una resolución "
    "impugnada, teniendo la facultad de invocar hechos notorios. Cuando se hagan valer "
    "diversas causales de ilegalidad, la sentencia del Tribunal deberá examinar primero "
    "aquéllas que puedan llevar a declarar la nulidad lisa y llana.”\n"
    "Así, la Sala puede analizar la competencia material de la autoridad emisora.")

# La litis del 2/2026: sólo leyes federales, ninguna local.
LITIS_2_2026 = ["Ley Federal de Procedimiento Contencioso Administrativo",
                "Ley del Seguro Social", "Código Fiscal de la Federación",
                "Reglamento Interior del Instituto Mexicano del Seguro Social",
                "Constitución Política de los Estados Unidos Mexicanos"]

# Lo que el material trae de la LFPCA: los espejos federales.
NORMAS = [
    {"cuerpo_legal": "Ley Federal de Procedimiento Contencioso Administrativo",
     "articulo": "51",
     "texto": "ARTÍCULO 51.- Se declarará que una resolución administrativa es ilegal cuando "
              "se demuestre alguna de las siguientes causales: I. Incompetencia del "
              "funcionario que la haya dictado, ordenado o tramitado el procedimiento del "
              "que deriva dicha resolución; II. Omisión de los requisitos formales exigidos "
              "por las leyes, siempre que afecte las defensas del particular y trascienda al "
              "sentido de la resolución impugnada, inclusive la ausencia de fundamentación o "
              "motivación, en su caso;"},
    {"cuerpo_legal": "Ley Federal de Procedimiento Contencioso Administrativo",
     "articulo": "50",
     "texto": "ARTÍCULO 50.- Las sentencias del Tribunal se fundarán en derecho y resolverán "
              "sobre la pretensión del actor que se deduzca de su demanda, en relación con "
              "una resolución impugnada, teniendo la facultad de invocar hechos notorios. "
              "Cuando se hagan valer diversas causales de ilegalidad, la sentencia del "
              "Tribunal deberá examinar primero aquéllas que puedan llevar a declarar la "
              "nulidad lisa y llana."},
]

print("\n1 · EL CASO REAL SE ATRAPA")
malas = ln.inadmisibles(MARCO_2_2026, LITIS_2_2026)
ok(len(malas) == 2, f"dos citas inadmisibles, no cero: {[(h['nums'], h['anafora']) for h in malas]}")
ok(any(h["anafora"] and "55" in h["nums"] for h in malas),
   "el 55 «de la propia ley» se resuelve a la ley de Querétaro")
ok("Estado de Querétaro" in ln._limpia_nombre(
    "Ley de Procedimiento Contencioso Administrativo del Estado de Querétaro establece"),
   "el nombre no se corta en «del» (límite de palabra de «es»)")

print("\n2 · ESPEJO: se reescribe al precepto federal con el mismo texto")
nuevo, avisos = ln.sanear(MARCO_2_2026, LITIS_2_2026, NORMAS, "el marco jurídico")
ok("artículo 51 de la Ley Federal de Procedimiento Contencioso Administrativo" in nuevo,
   "el 57 de Querétaro pasa a ser el 51 de la LFPCA")
ok("artículo 50 de la propia ley" in nuevo, "el 55 «de la propia ley» pasa a ser el 50")
ok("Artículo 51." in nuevo and "Artículo 57." not in nuevo, "y el rótulo de la transcripción también")
ok("ámbito local" not in nuevo, "sin «En el ámbito local» delante de una ley federal")
ok("presupuesto de validez del acto" in nuevo, "el razonamiento que seguía se conserva")
ok(not ln.inadmisibles(nuevo, LITIS_2_2026), "no queda ninguna cita inadmisible")
ok(len(avisos) == 2 and all("FUERA DE LA LITIS" in a for a in avisos),
   "y cada corrección queda dicha en un aviso")

print("\n3 · RETIRO: sin espejo, se quita la cita, su texto y la remisión que colgaba")
t = ("La competencia se funda en diversas normas. Asimismo, conforme al artículo 12 de la "
     "Ley de Hacienda del Estado de Querétaro, que dispone: “Artículo 12. Son contribuyentes "
     "del impuesto sobre nóminas las personas físicas y morales que realicen pagos por "
     "concepto de remuneraciones al trabajo personal”. De dicho precepto se sigue que el "
     "pago genera la obligación. Por tanto, la Sala actuó conforme a derecho.")
n3, a3 = ln.sanear(t, LITIS_2_2026, [], "el estudio")
ok("Hacienda" not in n3 and "nóminas" not in n3 and "De dicho precepto" not in n3,
   f"retirado: «{n3}»")
ok("Por tanto, la Sala actuó conforme a derecho." in n3, "y lo que no dependía de ella sigue")
ok(a3 and "RETIRADA" in a3[0], "con su aviso")

print("\n4 · LO QUE NO SE TOCA")
litis_coordinada = LITIS_2_2026 + [
    "Reglamento Interior de la Secretaría de Finanzas del Poder Ejecutivo del Estado de Querétaro"]
t4 = ("La competencia de la recurrente se sustenta en el artículo 5 del Reglamento Interior "
      "de la Secretaría de Finanzas del Poder Ejecutivo del Estado de Querétaro.")
ok(ln.sanear(t4, litis_coordinada, [], "el estudio") == (t4, []),
   "ley local que SÍ está en la litis (autoridad estatal coordinada): intacta")
t5 = ("Resulta ilustrativa la tesis de rubro «PRUEBAS…», que interpretó el artículo 940 del "
      "Código de Procedimientos Civiles para el Distrito Federal, cuyo criterio se comparte.")
ok(ln.sanear(t5, LITIS_2_2026, [], "el estudio") == (t5, []),
   "ley ajena nombrada por un criterio: intacta (jurisprudencia ajena permitida)")
t6 = "Conforme al artículo 42 del Código Fiscal de la Federación, la autoridad puede comprobar."
ok(ln.sanear(t6, [], [], "el estudio") == (t6, []), "ley federal: intacta")

print("\n5 · EL FUERO")
ok(ln.es_local("Código Civil para el Distrito Federal"),
   "el Código Civil para el DISTRITO FEDERAL es local, aunque diga «Federal»")
ok(not ln.es_local("Ley Federal de Procedimiento Contencioso Administrativo"), "la LFPCA no")
ok(not ln.admisible("Ley de Procedimiento Contencioso Administrativo del Estado de Querétaro",
                    ["Ley Federal de Procedimiento Contencioso Administrativo"]),
   "la ley federal en la litis NO admite a su espejo estatal")

ok(ln.admisible("Código de Procedimientos Civiles para el Estado de Querétaro en materia civil",
                ["Código de Procedimientos Civiles para el Estado de Querétaro"]),
   "una cola con palabra distintiva («en materia civil») no rompe la identidad")

print("\n6 · LAS PUERTAS ESTÁN CONECTADAS")
import marco_juridico as mj
import redactor_adelanto as ra
src_mj = inspect.getsource(mj)
ok("no se busca en {coleccion_estatal}" in src_mj,
   "el marco no busca una ley federal en el acervo de un estado")
ok("elif not _acto_federal:" in src_mj, "y no avisa «no se pudo leer» cuando sí se leyó")
src_ra = inspect.getsource(ra._terminar)
ok("_ln.sanear(estudio" in src_ra and "_ln.sanear(\n" in src_ra or "marco_escrito, _litis" in src_ra,
   "_terminar sanea el estudio y el marco")
ok("NO FIRMABLE TAL COMO ESTÁ" in src_ra, "y revisa el documento final")
ok(src_ra.find("_ln.sanear(estudio") < src_ra.find("relleno = ens.Relleno("),
   "el estudio se sanea ANTES de copiarse al relleno")
src_all = inspect.getsource(ra)
ok(src_all.count("    _litis_y_material(r, material, ") == 2
   and "def _litis_y_material(r, material" in src_all,
   "los dos redactores del estudio reciben el material acotado")

print()
if FALLOS:
    print(f"FALLAN {len(FALLOS)}: " + " · ".join(FALLOS))
    raise SystemExit(1)
print("RESULTADO: TODAS LAS COMPROBACIONES PASAN")
