# -*- coding: utf-8 -*-
"""EL FONDO NO SE TIRA, Y EL PRECEPTO QUE FALTA SE BUSCA.

David, 16-sep-2026, sobre la revisión fiscal 2/2026:
  · «a pesar de la extemporaneidad, si el secretario sigue trabajando en el
     proyecto el pipeline le debe dar el proyecto» (tercera vez que lo pide);
  · «salió "el motor no se atrevió con un sentido para todo el asunto" porque
     tuvo duda en dos por falta de acervo. Que un motor económico de búsqueda
     en internet traiga el artículo faltante (y así lo diga)».
"""
import asyncio
import datetime as dt
import sys

import busqueda_web as bw
import fase0_oportunidad as f0
import fase6_estudio as fe
import fase6_rag as fr

fallos = []
def ok(c, nota):
    print(f"  {'OK ' if c else 'MAL'} {nota}")
    if not c: fallos.append(nota)

# ══ A · con criterio y extemporaneidad, el fondo SALE ══
def _computo_extemporaneo():
    return f0.computar(dt.date(2025, 12, 4), dt.date(2026, 2, 20), "personal", 15)

c = _computo_extemporaneo()
ok(c.oportuna is False, "el cómputo de prueba es extemporáneo")
ok(c.cierra_por_extemporaneidad and not c.fondo_en_reserva,
   "sin decisión, hoy cerraría por improcedencia y sin fondo")

# LA REGLA NUEVA, tal como la aplica main._decidir_oportunidad: con criterio
# del secretario y cómputo extemporáneo sin decisión, se estampa «reserva».
avisos = f0.aplicar_decision(c, "reserva", "Se elabora el estudio de fondo a "
                             "petición de quien proyecta, no obstante el cómputo "
                             "de extemporaneidad, para el caso de que no se "
                             "comparta esa conclusión al resolver.")
ok(c.decision == "reserva", f"la reserva se estampa: decision={c.decision!r}")
ok(c.fondo_en_reserva is True, "→ fondo_en_reserva: el estudio SÍ se escribe")
ok(c.cierra_por_extemporaneidad is True,
   "y la ejecutoria sigue resolviendo la improcedencia (congruencia del 74-VI)")
ok(any("anexo" in a.lower() for a in avisos), "y se dice dónde va el estudio")

# el motivo corto se cae en silencio: por eso el automático es largo
c2 = _computo_extemporaneo()
f0.aplicar_decision(c2, "reserva", "ok")
ok(c2.decision == "" and not c2.fondo_en_reserva,
   "un motivo corto NO aplica la decisión — el automático debe pasar de "
   f"{f0.MOTIVO_MINIMO} caracteres")
import main as _m  # el motivo automático de verdad
ok(len(_m.MOTIVO_RESERVA_AUTO) >= f0.MOTIVO_MINIMO,
   f"el motivo automático mide {len(_m.MOTIVO_RESERVA_AUTO)} caracteres")
c3 = _computo_extemporaneo()
f0.aplicar_decision(c3, "reserva", _m.MOTIVO_RESERVA_AUTO)
ok(c3.fondo_en_reserva is True, "y con él la reserva SÍ se aplica")

# ══ B · el aviso deja de acusar a los traídos ══
mat = fe.Material()
mat.normas = [{"cuerpo_legal": "Ley del Seguro Social", "articulo": "17", "texto": "…"},
              {"cuerpo_legal": "Ley del Seguro Social", "articulo": "251", "texto": "…"}]
est = ("El artículo 17 de la Ley del Seguro Social obliga; el artículo 251 de la "
       "Ley del Seguro Social faculta; y el artículo 150 del Reglamento Interior "
       "del Instituto Mexicano del Seguro Social confiere competencia.")
_, quedan = fe.preceptos_fuera(est, mat)
ok(len(quedan) == 1 and list(quedan)[0][1] == "150",
   f"tras completar, sólo se acusa el que de verdad falta: {sorted(quedan)}")

# ══ C · el precepto ausente se busca en la web, marcado ══
class _Q:
    def scroll(s, collection_name, scroll_filter, limit, offset=None,
               with_payload=True, with_vectors=False):
        return ([], None)          # el acervo no lo tiene

async def _falso_texto(cuerpo_legal, numero, estado=None):
    return {"texto": "Artículo 150. Corresponde a los titulares…",
            "url": "https://www.diputados.gob.mx/x.pdf",
            "dominio": "diputados.gob.mx", "titulo": "RIIMSS"}

_orig = bw.texto_de_articulo
bw.texto_de_articulo = _falso_texto
try:
    m2 = fe.Material(); m2.normas = []
    trajo = asyncio.run(fr.completar_preceptos(
        _Q(), m2, [("reglamento interior del instituto mexicano del seguro social", "150")],
        None, materia="administrativa", tipo_asunto="revision_fiscal"))
finally:
    bw.texto_de_articulo = _orig

ok(len(m2.normas) == 1, "el precepto que el acervo no tiene entra al material")
n = m2.normas[0] if m2.normas else {}
ok(n.get("de_internet") is True, "y viaja MARCADO como traído de internet")
ok(n.get("dominio") == "diputados.gob.mx", f"con su dominio: {n.get('dominio')}")
ok("fuente" not in n,
   "y NO usa la clave «fuente», que el compositor lee como nombre de la ley")
ok(any("diputados.gob.mx" in x for x in trajo), f"y se reporta como tal: {trajo}")

# la nota al pie lo dice
import docx, documento_generado as dg
d = docx.Document(); notas = []
p_ = d.add_paragraph("Conforme al artículo 150 del Reglamento Interior del "
                     "Instituto Mexicano del Seguro Social, la autoridad…")
dg.notas_de_articulos(d, p_, p_.text, m2.normas, notas)
ok(notas and "no del acervo verificado" in notas[0],
   f"la nota al pie avisa del origen: …{notas[0][-90:] if notas else '(sin nota)'}")

# ══ D · sin fuente oficial, no hay precepto ══
async def _sin_fuente(cuerpo_legal, numero, estado=None):
    return {}
bw.texto_de_articulo = _sin_fuente
try:
    m3 = fe.Material(); m3.normas = []
    asyncio.run(fr.completar_preceptos(
        _Q(), m3, [("ley inventada", "9")], None, materia="administrativa"))
finally:
    bw.texto_de_articulo = _orig
ok(not m3.normas, "si la web no da fuente oficial, NO se inventa el precepto")

# ══ E · el filtro de dominios ══
ok(bw._es_oficial("diputados.gob.mx", bw.AGENTE_ARTICULO["cotos"]), "diputados.gob.mx entra")
ok(bw._es_oficial("ordenjuridico.gob.mx", bw.AGENTE_ARTICULO["cotos"]), "ordenjuridico.gob.mx entra")
ok(not bw._es_oficial("leyes-mx.com", bw.AGENTE_ARTICULO["cotos"]), "leyes-mx.com NO entra")
ok(not bw._es_oficial("justia.com", bw.AGENTE_ARTICULO["cotos"]), "justia.com NO entra")

# ══ F2 · LOS TRES CAMINOS del precepto llevan la marca ══
# El artículo 150 salió al pie SIN marca en la prueba real de la 2/2026: la
# nota la compuso el marco jurídico, y sólo se había marcado el otro camino.
_web = {"cuerpo_legal": "Reglamento Interior del IMSS", "articulo": "150",
        "texto": "Artículo 150. Son atribuciones de las subdelegaciones: [1] I. Vigilar.",
        "de_internet": True, "dominio": "imss.gob.mx", "url": "https://imss.gob.mx/x"}
_acervo = {"cuerpo_legal": "Ley de Amparo", "articulo": "76", "texto": "El órgano corregirá."}
ok(dg.marca_de_origen(_web).strip().startswith("· TEXTO TOMADO DE imss.gob.mx"),
   "marca_de_origen habla para lo traído de internet")
ok(dg.marca_de_origen(_acervo) == "", "y calla para lo que sí es del acervo")
ok("[1]" not in dg.limpiar_texto_web(_web["texto"]),
   "las referencias [1] del buscador no entran en el precepto")
ok(dg.limpiar_texto_web("Artículo 5. Nadie [12] podrá.") == "Artículo 5. Nadie podrá.",
   "y se limpian aunque vengan de dos cifras")
# el marco jurídico —el camino por el que se coló sin marca— ya la pone
import inspect as _insp
_fuente_marco = _insp.getsource(dg)
_i = _fuente_marco.index('_pie = (f"«Artículo {num}')
ok("marca_de_origen(n_)" in _fuente_marco[_i:_i + 260],
   "el marco jurídico compone la nota CON la marca")

# ══ F · los guardianes del texto traído (sin red) ══
# Lo que de verdad devolvió el buscador con una ley inventada, citando un
# dominio oficial: una NEGATIVA. Pasaba los filtros de largo y de dos puntos.
_NEG = ("No puedo citar textualmente ni completo el artículo 77 de una ley "
        "inexistente, porque «Ley Inventada de la Nada Absoluta» no corresponde "
        "a una norma identificable en el orden jurídico mexicano, y por eso no "
        "es posible transcribir su contenido literal de fuente oficial alguna.")
ok(bool(bw._NEGATIVAS.search(_NEG)), "una negativa del buscador se reconoce como tal")
ok(not bw._NEGATIVAS.search(
    "ARTÍCULO 150. Son atribuciones de las subdelegaciones, dentro de su "
    "circunscripción territorial: I. Registrar a los patrones y sujetos obligados."),
   "y un artículo de verdad NO se confunde con una negativa")

# ══ G · lo que encontró la auditoría adversarial ══
import main as _mm

# G1 · la reserva automática se marca como automática
c4 = _computo_extemporaneo()
ok(hasattr(c4, "decision_automatica"), "Computo sabe si la decisión se aplicó sola")

# G2 · el anexo no atribuye una petición que nadie hizo
import inspect as _i2, documento_generado as _dg2
_src = _i2.getsource(_dg2)
ok("decision_automatica" in _src and "no ha declarado razón" in _src,
   "el anexo distingue la reserva pedida de la automática")
ok("Razón declarada por quien proyecta" in _src and "not _auto_r" in _src,
   "y sólo imprime «razón declarada» cuando de verdad la declaró")

# G3 · la hoja que circula no propone el fondo si el resolutivo desecha
# El pie de firmas se retiró el 16-sep-2026 —David: «los nombres de magistrado
# y secretario no tienen que ir hasta abajo»—, así que lo que se comprueba es
# lo que siempre importó: que con un cómputo extemporáneo NO salga la hoja de
# síntesis proponiendo un fondo que el resolutivo no resuelve.
ok("if not _extemp:\n        _bloque_sintesis" in _src,
   "con extemporaneidad, la SÍNTESIS no se escribe")
ok("_bloque_firmas" not in _src, "y el proyecto ya no lleva firmas al pie")

# G4 · el candado del acervo estatal no se salta por la web
_srcr = _i2.getsource(fr)
ok("es ley estatal en una revisión FISCAL" in _srcr,
   "la web no trae ley estatal en una revisión fiscal (regla de la ley ajena)")
ok("if not cols:" in _srcr, "ni busca cuando la colección se vació por candado")
ok("TOPE_WEB" in _srcr, "y hay tope de búsquedas por asunto")

# G5 · el nombre de la ley no arrastra la cola de la pregunta
_, _pares_q = fe.preceptos_fuera(
    "¿El artículo 150, fracción XIX, del Reglamento Interior del Instituto "
    "Mexicano del Seguro Social faculta a la autoridad?", fe.Material())
_nombres = [c for c, a in _pares_q]
ok(_nombres and not any("?" in n_ or n_.endswith(" la") for n_ in _nombres),
   f"el nombre sale limpio de una pregunta: {_nombres}")

# G6 · la cita web tiene que nombrar la norma
_srcw = _i2.getsource(bw)
ok("_nombra_norma" in _srcw, "la cita oficial tiene que nombrar el ordenamiento")
# Desde el 23-sep-2026 el corte llega por `cortada`, que vale para los dos
# motores: finish_reason «length» en sonar, max_output_tokens en OpenAI.
ok('if c["cortada"]:' in _srcw and '== "length"' in _srcw,
   "y una respuesta cortada por longitud se descarta")

print()
if fallos:
    print(f"FALLAN {len(fallos)}: " + " · ".join(fallos)); sys.exit(1)
print("TODAS LAS COMPROBACIONES PASAN")
