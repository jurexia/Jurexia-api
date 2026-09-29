# -*- coding: utf-8 -*-
"""LA CARÁTULA DEL AMPARO EN REVISIÓN, COMO LA ESCRIBE EL CORPUS (28-sep-2026,
AR 631/2025), SIN LLAMAR A NINGÚN MODELO.

Medido sobre la primera página de 500 amparos en revisión del Tercer Tribunal
Colegiado en Materias Administrativa y Civil del Vigésimo Segundo Circuito
(478 carátulas legibles):
  · ningún rubro lleva el órgano recurrido ni el juzgado (0 de 478);
  · si recurre la quejosa: «QUEJOSA Y RECURRENTE» (162 de 226);
  · si recurre otra parte: el recurrente con su carácter («TERCERA INTERESADA
    Y RECURRENTE», «AUTORIDAD RESPONSABLE Y RECURRENTE») y, en la forma que se
    adopta (AR 60/2020), la quejosa en su propio renglón encima.

Nombres sintéticos.

    .venv/bin/python test_caratula_revision.py
"""
import sys

from docx import Document

import documento_generado as dg
import fase_partes as fp
import tipos_asunto as ta

FALLOS = []


def ok(cond, que):
    print(f"   {'PASA ' if cond else 'FALLA'}  {que}")
    if not cond:
        FALLOS.append(que)


QUEJOSA = "Unión Ejemplo, A.C."
TERCERA = "Inmobiliaria Ejemplo, S.A. de C.V."

print("1 · EL CATÁLOGO: SIN ÓRGANO RECURRIDO EN EL RUBRO")
_cat = ta.caratula_de("amparo_revision")
ok(not any("RECURRIDO" in e or c == "responsable" for e, c, _o in _cat),
   f"la revisión no rotula al juzgado ni a la responsable: {_cat}")
ok(ta.caratula_de("queja")[-1][0] == "ÓRGANO QUE DICTÓ EL AUTO RECURRIDO"
   and ta.caratula_de("revision_fiscal")[-1][0] == "SALA RESPONSABLE"
   and ta.caratula_de("amparo_directo")[-1][0] == "AUTORIDAD RESPONSABLE",
   "los otros tipos conservan su renglón del órgano")
ok(ta.etiqueta_de_figura("amparo_revision", "responsable").startswith("ÓRGANO RECURRIDO")
   and ta.etiqueta_de_figura("amparo_directo", "responsable") == "AUTORIDAD RESPONSABLE",
   "la ficha de partes sigue sabiendo cómo se llama el juzgado, aunque no vaya en el rubro")
_ficha = fp.Partes(quejoso=QUEJOSA, tercero_interesado=TERCERA,
                   autoridad_responsable="Juzgado Séptimo de Distrito",
                   tipo_asunto="amparo_revision").bloque()
ok("ÓRGANO RECURRIDO (Juzgado de Distrito que dictó la sentencia recurrida): Juzgado Séptimo" in _ficha and "AUTORIDAD RESPONSABLE: Juzgado" not in _ficha,
   "la ficha no llama «autoridad responsable» al juzgado de la recurrida")

print("\n2 · QUIÉN RECURRE DECIDE LOS RENGLONES")
_f = ta.filas_caratula("amparo_revision", {"quejoso": QUEJOSA})
ok(_f == [("QUEJOSA Y RECURRENTE", "quejoso", QUEJOSA)],
   f"recurre la quejosa: una figura, concordada (A.C. es femenino): {_f}")
_f = ta.filas_caratula("amparo_revision", {
    "quejoso": QUEJOSA, "recurrente": TERCERA, "papel_recurrente": "tercero",
    "quejoso_moral": True, "organo_recurrido": "Juzgado Séptimo de Distrito",
    "responsable": "Magistrada de la Sala Civil"})
ok(_f == [("QUEJOSA", "quejoso", QUEJOSA),
          ("TERCERA INTERESADA Y RECURRENTE", "recurrente", TERCERA)],
   f"el 631: quejosa y tercera recurrente en renglones distintos, sin juzgado ni Magistrada: {_f}")
_f = ta.filas_caratula("amparo_revision", {
    "quejoso": "Juan Pérez", "recurrente": "Gobernador del Estado",
    "papel_recurrente": "autoridad", "quejoso_moral": True})
ok(_f == [("PARTE QUEJOSA", "quejoso", "Juan Pérez"),
          ("AUTORIDAD RESPONSABLE Y RECURRENTE", "recurrente", "Gobernador del Estado")],
   f"recurre la autoridad; «quejoso_moral» del formulario no vale para otra persona: {_f}")
_f = ta.filas_caratula("amparo_revision", {"quejoso": "Juan Pérez", "recurrente": "Pedro Gómez"})
ok(_f[-1] == ("RECURRENTE", "recurrente", "Pedro Gómez"),
   f"sin carácter conocido, «RECURRENTE» a secas: {_f}")
_f = ta.filas_caratula("amparo_revision", {"quejoso": QUEJOSA, "adherente": "Gobernador"})
ok(_f[-1] == ("RECURRENTE ADHESIVO", "adherente", "Gobernador"), f"el adhesivo cuando consta: {_f}")
_f = ta.filas_caratula("queja", {"quejoso": "X", "responsable": "Sala", "organo_recurrido": "Juzgado Segundo"})
ok(_f[-1] == ("ÓRGANO QUE DICTÓ EL AUTO RECURRIDO", "responsable", "Juzgado Segundo"),
   "en la queja el órgano leído sigue mandando en su renglón")

print("\n3 · EL .docx Y LA HOJA DEL PROMPT DICEN LO MISMO")
_datos = {"encabezado": "AMPARO EN REVISIÓN CIVIL: 631/2025", "tipo_asunto": "amparo_revision",
          "quejoso": QUEJOSA, "recurrente": TERCERA, "papel_recurrente": "tercero",
          "organo_recurrido": "Juzgado Séptimo de Distrito en el Estado de Ejemplo",
          "responsable": "Magistrada de la Sala Civil", "magistrado": "M", "secretario": "S"}
_doc = Document()
dg._caratula(_doc, _datos, "amparo_revision")
_cab = " ".join(p.text for p in _doc.paragraphs).upper()
ok("QUEJOSA: UNIÓN EJEMPLO, A.C." in _cab and "TERCERA INTERESADA Y RECURRENTE: INMOBILIARIA EJEMPLO" in _cab
   and "QUEJOSA Y RECURRENTE" not in _cab and "RECURRIDO" not in _cab
   and "JUZGADO" not in _cab and "MAGISTRADA" not in _cab,
   f"la carátula del .docx: {_cab[:260]}")
_p = dg.prompt_estructura(_datos)
ok("TERCERA INTERESADA Y RECURRENTE (quien interpuso el recurso): Inmobiliaria Ejemplo" in _p
   and "ÓRGANO RECURRIDO:" not in _p
   and "sentencia recurrida: Juzgado Séptimo" in _p,
   "la hoja del prompt: el recurrente con su carácter y el juzgado como dato, no como rótulo")

print()
if FALLOS:
    print(f"FALLAN {len(FALLOS)}: " + " · ".join(FALLOS))
    sys.exit(1)
print("RESULTADO: TODAS LAS COMPROBACIONES PASAN")
