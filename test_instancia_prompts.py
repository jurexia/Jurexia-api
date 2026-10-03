# -*- coding: utf-8 -*-
"""Amparo directo sin alzada y en cumplimiento: los prompts dejan de suponer una
Sala (30-sep-2026).

David, AD 323/2025 (juicio oral mercantil): «Siempre, en amparo directo,
partimos de la base de que hay una sala. Es decir, una segunda instancia. Pero
no siempre es así […] la sentencia había sido dictada en cumplimiento». Esto
comprueba:

  1 · con el origen puesto (un juzgado de oralidad mercantil, única instancia),
      los prompts y textos de amparo directo no suponen «la Sala», «toca»,
      «apelación» ni «agravio» —salvo en la orden que dice que NO se narren—;
  2 · con cumplimiento, los antecedentes y el relato piden contarlo;
  3 · la técnica de la violación procesal, el ejemplo de la refutación y los
      efectos de la v1 del estudio, sin «recurso ordinario» que no hubo;
  4 · la coletilla del 1390 Bis y la autoridad del resolutivo de la plantilla;
  5 · LA REGLA DE ORO: sin contexto, o con las banderas apagadas, los textos son
      los de antes, letra por letra.

    .venv/bin/python test_instancia_prompts.py
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
for _k in ("INSTANCIA_ORIGEN", "CUMPLIMIENTO_EJECUTORIA", "ESTUDIO_PROMPT"):
    os.environ.pop(_k, None)

import contexto_taller as ct
import origen_acto as oa
import tipos_asunto as ta
import fases123_pipeline as fp
import fases123_resumenes as fr
import documento_generado as dg
import ensamblar_adelanto as ea
import fase6_estudio as f6

FALLOS = []


def ok(c, q):
    print(f"   {'PASA ' if c else 'FALLA'}  {q}")
    if not c:
        FALLOS.append(q)


AD = "amparo_directo"
JUZGADO = ("Juzgado Segundo de Primera Instancia Especializado en Oralidad Mercantil del "
           "Distrito Judicial de Querétaro")
SALA = "Primera Sala Civil del Tribunal Superior de Justicia del Estado de Querétaro"
# El acto de prueba no trae ninguna de las palabras vigiladas: lo que aparezca
# en el prompt lo puso el prompt, no el documento.
ACTO = ("Querétaro, Querétaro, a cinco de mayo de dos mil veinticinco. VISTOS para resolver los "
        "autos del expediente 123/2024 relativo al juicio oral mercantil promovido por Fulano "
        "contra Mengano. CONSIDERANDO. Se estima que la actora acreditó su acción. RESUELVE.")
CUMPL = (" La presente se dicta en cumplimiento de la ejecutoria dictada en el juicio de amparo "
         "directo civil 590/2023, que concedió el amparo para el efecto de que la responsable deje "
         "insubsistente la sentencia y dicte otra en la que valore la pericial.")

# Una orden de NO narrar algo nombra ese algo; se quita antes de buscar.
_RX_NEGATIVA = re.compile(r"\bno\s+(?:narres|cuentes|inventes)\b[^.]*\.", re.I)
_VIGILADAS = (("la Sala", re.compile(r"\bla\s+Sala\b")),
              ("toca", re.compile(r"\btoca\b", re.I)),
              ("apelación", re.compile(r"apelaci[óo]n", re.I)),
              ("agravio", re.compile(r"\bagravios?\b", re.I)))


def suposiciones(texto: str) -> list:
    t = _RX_NEGATIVA.sub(" ", texto or "")
    return [n for n, rx in _VIGILADAS if rx.search(t)]


C_FUNDADO = [f6.Criterio(problema="¿Debió admitirse la prueba pericial?", sentido="fundado",
                         razonamiento="Porque era pertinente.", jerarquia="principal")]


def material(variante="v1"):
    return f6.Material(tipo_asunto=AD, materia="mercantil", formato="estandar",
                       problemas=[{"pregunta": C_FUNDADO[0].problema, "cubre": [1]}],
                       n_planteamientos=1, variante=variante)


def datos_estructura(acto=ACTO):
    return {"tipo_asunto": AD, "acto": acto, "antecedentes": "ANTECEDENTES",
            "tribunal": "Tribunal Colegiado", "encabezado": "AMPARO DIRECTO CIVIL 1/2025",
            "quejoso": "Fulano", "responsable": JUZGADO}


def textos():
    """Todo lo que se vigila, por nombre."""
    return {
        "resumen del acto": fp.prompt_resumen_acto(ACTO, tipo_asunto=AD),
        "resumen del acto a fondo": fp.prompt_acto_a_fondo(ACTO, "corto", [], 500, tipo_asunto=AD),
        "relato": fp.prompt_relato("ANT", "ACTO", "CONC", tipo_asunto=AD, quejoso="Q",
                                   responsable=JUZGADO, lo_resuelto="ese recurso de apelación"),
        "problemas": fp.prompt_problemas("ACTO", "CONC", tipo_asunto=AD, n_planteamientos=2),
        "antecedentes": fp.prompt_antecedentes(ACTO, AD),
        "instrucciones de antecedentes": fr.instrucciones_antecedentes(AD),
        "instrucciones del resumen": fr.instrucciones_resumen_acto(AD, 400),
        "instrucciones de problemas": fr.instrucciones_problemas(True, AD),
        "estructura": dg.prompt_estructura(datos_estructura()),
        "verbos del recurrido": ta.verbos_del_recurrido(AD),
        "resultandos": " ".join(q for _, q in ta.resultandos_de(AD)),
        "forma moderna del estudio": __import__("formato_sentencia").forma_del_estudio(
            "moderna", "conceptos de violación", "concepto de violación", "quejosa", "fundados", 2000),
        "técnica de la violación procesal": " ".join(
            r["cuando"] + " " + " ".join(r["tecnica"]) for r in ta.tecnica_de(AD, "", True)),
    }


def estudio_v1():
    return f6.prompt_estudio("ACTO", "CONC", C_FUNDADO, material("v1"))


def poner_unica(con_cumplimiento=False):
    ct.poner(True, pruebas=True)
    ct.poner_origen(oa.origen(JUZGADO, "…el juicio oral mercantil…", "",
                              ACTO + (CUMPL if con_cumplimiento else "")))


def sin_contexto():
    ct.poner(False)
    ct.poner_origen(None)


# ═══════════════════════════════════════════════════════════════════════════
print("\n0 · LOS ACCESORES")
sin_contexto()
ok(not ta.unica_instancia(AD) and ta.cumplimiento_de_amparo(AD) == {},
   "sin contexto: ni única instancia ni cumplimiento")
poner_unica()
ok(ta.instancia_actual() == "unica" and ta.unica_instancia(AD) and ta.unica_instancia(""),
   "un juzgado de oralidad mercantil: única instancia (sin tipo se entiende amparo directo)")
ok(not ta.unica_instancia("queja") and not ta.unica_instancia("amparo_revision")
   and ta.cumplimiento_de_amparo("queja") == {},
   "los recursos no toman la variante aunque el contexto la traiga")
ok(ta.cumplimiento_de_amparo(AD) == {}, "sin la mención de la ejecutoria no hay cumplimiento")
poner_unica(con_cumplimiento=True)
_c = ta.cumplimiento_de_amparo(AD)
ok(_c.get("consta") and "590/2023" in _c.get("ejecutoria", ""),
   f"con la mención, cumplimiento de la ejecutoria del {_c.get('ejecutoria')}")


# ═══════════════════════════════════════════════════════════════════════════
print("\n1 · ÚNICA INSTANCIA: NINGÚN TEXTO SUPONE LA ALZADA")
poner_unica()
_u = textos()
for nombre, t in _u.items():
    s = suposiciones(t)
    ok(not s, f"{nombre}: sin {', '.join(s) if s else 'la Sala, toca, apelación ni agravio'} como supuesto")
ok("inferior" not in _u["resumen del acto"] and "consideró fundado" not in _u["resumen del acto"],
   "el ejemplo del resumen ya no es el de una Sala que revoca al inferior")
ok("«El Juez consideró…»" in _u["resumen del acto"],
   "el ejemplo describe la forma con el órgano por su nombre, sin datos de otro asunto")
_ant = _u["instrucciones de antecedentes"]
ok(not re.search(r"\b(interpuso|confirmó|revocó)\b", _ant) and "Inconforme con esa resolución" not in _ant,
   "antecedentes: sin los verbos ni el arranque de la segunda instancia")
ok("ÚNICA INSTANCIA" in _ant and "la que dictó\n  el Juez" in _ant,
   "antecedentes: el último párrafo es la sentencia del juez en única instancia")
ok("una Sala o un tribunal ordinario" not in _ant and "aquí, el Juez" in _ant,
   "antecedentes: la responsable es quien dictó lo reclamado, no «una Sala»")
ok("¿Cómo resolvió el Juez ese juicio?" in _u["relato"] and "cada instancia" not in _u["relato"],
   "relato: una sola instancia y la pregunta sin «ese recurso de apelación»")
ok("«¿Debía el Juez estudiar" in _u["problemas"], "problemas: la pregunta de ejemplo nombra al Juez")
ok("recurso ordinario que la\nconfirmó" not in _u["problemas"] and "medio de defensa" in _u["problemas"],
   "problemas: la violación procesal no se mide contra una resolución de alzada")
ok("IDENTIFÍCALO por fecha, órgano que la dictó y número del expediente de origen" in _u["estructura"]
   and "un número de expediente o un nombre" in _u["estructura"],
   "estructura: el acto se identifica sin sala ni toca, y no se inventa el expediente")
ok(not _u["verbos del recurrido"].startswith("confirmó")
   and "procedente o improcedente la acción" in _u["verbos del recurrido"],
   "los verbos de la responsable son los de quien resolvió el juicio")
_tec = ta.tecnica_de(AD, "", True)
ok(_tec[0] is ta.TECNICA_RESOLUCION["directo_violacion_procesal_unica"]
   and _tec[1] is ta.TECNICA_RESOLUCION["directo_orden_de_estudio"],
   "la técnica de la violación procesal es la de única instancia (y el orden de estudio, el de siempre)")
_tu = " ".join(_tec[0]["tecnica"])
ok("recurso ordinario que la confirmó" not in _tu and "resolución del recurso ordinario" not in _tu
   and "DURANTE el juicio" in _tu and "no se exige haberla recurrido" in _tu and "PASO A PASO" in _tu,
   "técnica: la preparación se mide durante el juicio, no se exige apelar, la reposición paso a paso")
ok(_tec[0]["fuente"].startswith("artículos 171, 172"), "técnica: el mismo fundamento")
_base_tec = ta.TECNICA_RESOLUCION["directo_violacion_procesal"]["tecnica"]
_iguales = [a for a, b in zip(_base_tec, _tec[0]["tecnica"]) if a == b]
ok(len(_tec[0]["tecnica"]) == len(_base_tec) and len(_iguales) == len(_base_tec) - 3,
   "técnica: cambian sólo el objeto, la preparación y los efectos; lo demás es la misma regla")
_e1 = estudio_v1()
ok("«el Juez afirmó X;" in _e1 and "«la Sala afirmó X;" not in _e1,
   "estudio v1: el ejemplo de la refutación nombra al Juez")
ok("resolución del recurso ordinario" not in _e1 and "efectos la actuación viciada y, si se combatió" in _e1,
   "estudio v1: los efectos de la reposición sin la resolución del recurso ordinario")
ok(fr.sujetos_responsable(AD)[0] == "el Juez" and "la Sala" not in fr.sujetos_responsable(AD),
   "el respaldo de sujetos nombra al Juez, no a la Sala")


# ═══════════════════════════════════════════════════════════════════════════
print("\n2 · EN CUMPLIMIENTO: SE CUENTA")
poner_unica(con_cumplimiento=True)
_k = textos()
_ant_c = _k["instrucciones de antecedentes"]
ok("EN CUMPLIMIENTO de la ejecutoria del\n  amparo directo civil 590/2023" in _ant_c
   and "qué ordenó la ejecutoria" in _ant_c and "la sentencia nueva" in _ant_c,
   "antecedentes: el amparo anterior —número, tribunal, qué ordenó— y la sentencia nueva")
ok("ÚNICA INSTANCIA" in _ant_c, "antecedentes: y sigue siendo única instancia")
ok("EN CUMPLIMIENTO" in _k["relato"] and "590/2023" in _k["relato"]
   and "qué ordenó esa ejecutoria" in _k["relato"],
   "relato: pide contar que se dictó en cumplimiento y qué ordenó la ejecutoria")
ok("valore la pericial" in _k["relato"], "relato: los efectos transcritos van como dato del asunto")
for nombre in ("instrucciones de antecedentes", "relato"):
    s = suposiciones(_k[nombre])
    ok(not s, f"{nombre} (en cumplimiento): sin {', '.join(s) if s else 'suponer alzada'}")
# EN CUMPLIMIENTO CON ALZADA: se cuenta el cumplimiento y la alzada sigue ahí.
ct.poner(True, pruebas=True)
ct.poner_origen(oa.origen(SALA, "", "", ACTO + CUMPL))
_rel_a = fp.prompt_relato("ANT", "ACTO", "CONC", tipo_asunto=AD, lo_resuelto="ese recurso de apelación")
ok(ta.instancia_actual() == "alzada" and "cada instancia" in _rel_a and "EN CUMPLIMIENTO" in _rel_a
   and "¿Cómo resolvió la Sala ese recurso de apelación?" in _rel_a,
   "con alzada: la segunda instancia se cuenta y el cumplimiento también")
ok("EN CUMPLIMIENTO" in fr.instrucciones_antecedentes(AD)
   and "ÚNICA INSTANCIA" not in fr.instrucciones_antecedentes(AD)
   and "interpuso, resolvió, confirmó" in fr.instrucciones_antecedentes(AD),
   "con alzada: los antecedentes piden el cumplimiento sin tocar los verbos de la apelación")


# ═══════════════════════════════════════════════════════════════════════════
print("\n3 · LA COLETILLA DEL 1390 BIS Y EL RESOLUTIVO DE LA PLANTILLA")
poner_unica()
_col = dg.coletilla_oral_mercantil(AD, {"acto": ACTO})
ok("1390 bis" in _col and "no procederá recurso ordinario alguno" in _col,
   "oral mercantil en única instancia: la coletilla del banco")
ok(dg.coletilla_oral_mercantil(AD, {"acto": "juicio ejecutivo mercantil oral 5/2024",
                                    "antecedentes": ""}) == "",
   "el ejecutivo mercantil oral es otra vía: sin coletilla")
ok(dg.coletilla_oral_mercantil("queja", {"acto": ACTO}) == "", "una queja: sin coletilla")
_res_juez = ("ÚNICO. La Justicia de la Unión no ampara ni protege a Fulano, contra la sentencia dictada "
             "el cinco de mayo de dos mil veinticuatro, por el Juez Primero de Oralidad Mercantil del "
             "Distrito, en el expediente 5/2024, de su índice.")
_t, _st = ea.autoridad_en_resolutivo(_res_juez, JUZGADO)
ok(f", por el {JUZGADO}, en el expediente 5/2024" in _t and not _st,
   "plantilla de oralidad: «por el Juez…, en el expediente» se sustituye")
_res_sala = ("ÚNICO. La Justicia de la Unión ampara y protege a Fulano, contra la sentencia dictada el "
             "cinco de mayo de dos mil veinticuatro, por la Primera Sala Civil del Tribunal Superior de "
             "Justicia, en el toca civil 12/2024, de su índice.")
_t2, _st2 = ea.autoridad_en_resolutivo(_res_sala, "el " + JUZGADO)
ok(f", por el {JUZGADO}, en el expediente 12/2024" in _t2 and _st2 and "el el" not in _t2
   and "toca" not in _t2,
   "plantilla de alzada en un juicio de única instancia: órgano con su artículo y el toca pasa a expediente")


def competencia_compuesta() -> str:
    """El párrafo de competencia del .docx de un amparo directo mercantil."""
    import datetime as _dt
    import tempfile
    import fase0_oportunidad as _f0
    from docx import Document
    _est = dg.Estructura(apertura="V.", visto="para resolver.",
                         resultandos=[{"titulo": "Presentación de la demanda",
                                       "texto": "Se presentó la demanda."}],
                         competencia="", existencia="", procedencia="")
    _c = _f0.computar(_dt.date(2026, 3, 2), _dt.date(2026, 3, 9), plazo=15)
    _fd, _salida = tempfile.mkstemp(suffix=".docx")
    os.close(_fd)
    try:
        dg.componer({"tipo_asunto": AD, "numero": "1/2025", "acto": ACTO,
                     "encabezado": "AMPARO DIRECTO MERCANTIL 1/2025", "quejoso": "Fulano",
                     "responsable": JUZGADO, "magistrado": "M", "secretario": "S",
                     "tribunal": "Tribunal Colegiado", "ciudad": "Querétaro"},
                    _est, _c, _f0.fecha_en_letra, _salida,
                    estudio=["Es infundado el concepto de violación."],
                    calificaciones=["infundado"], tipo_asunto=AD)
        return next((q.text for q in Document(_salida).paragraphs
                     if "es competente" in q.text), "")
    finally:
        os.remove(_salida)


try:
    _comp_u = competencia_compuesta()
    ok(_comp_u.rstrip().endswith("no procederá recurso ordinario alguno.")
       and "Consejo de la Judicatura Federal. Además se advierte" in _comp_u,
       "el .docx: la coletilla va pegada al final del párrafo de competencia, sin apartado nuevo")
    sin_contexto()
    _comp_s = competencia_compuesta()
    ok(_comp_s and "1390" not in _comp_s, "el .docx sin contexto: la competencia de siempre, sin coletilla")
    poner_unica()
except Exception as _e_doc:
    ok(False, f"no se pudo componer el .docx de prueba: {type(_e_doc).__name__}: {_e_doc}")
    poner_unica()
_t3, _ = ea.autoridad_en_resolutivo(_res_sala, "Órgano sin clase reconocible del Distrito")
ok(_t3 == _res_sala.replace("Primera Sala Civil del Tribunal Superior de Justicia",
                            "Órgano sin clase reconocible del Distrito"),
   "si no se sabe qué órgano es, no se concuerda: la sustitución de siempre")


# ═══════════════════════════════════════════════════════════════════════════
poner_unica()
_fm = textos()["forma moderna del estudio"]
ok("¿La autoridad responsable estaba obligada" in _fm and "La Sala responsable" not in _fm,
   "única instancia: el ejemplo del formato moderno no pregunta por «la Sala»")


print("\n4 · LA REGLA DE ORO: SIN CONTEXTO, COMO ANTES")
sin_contexto()
_s = textos()
ok("¿La Sala responsable estaba obligada" in _s["forma moderna del estudio"],
   "forma moderna: el ejemplo de siempre")
ok("«La Sala consideró fundado\nel agravio respecto a la carga de la prueba. Determinó que, contrario a lo\n"
   "resuelto por el inferior," in _s["resumen del acto"], "resumen del acto: el ejemplo de siempre")
ok("contestar los\n  agravios contra las otras tres" in _s["instrucciones del resumen"]
   and "porque los agravios\n  suelen ir contra ellas" in _s["instrucciones del resumen"],
   "instrucciones del resumen: «agravios», como estaban")
ok("dictó, admitió, interpuso, resolvió, confirmó, turnó." in _s["instrucciones de antecedentes"]
   and "«Inconforme con esa resolución…»." in _s["instrucciones de antecedentes"]
   and "responsable es una Sala o un tribunal ordinario: resuelve el juicio de" in _s["instrucciones de antecedentes"]
   and _s["instrucciones de antecedentes"].endswith(
       "donde debía ir el verbo.\n- NO opines, NO califiques y NO adelantes el estudio."),
   "antecedentes: verbos, arranques, «una Sala o un tribunal ordinario» y el cierre de siempre")
ok("2. AQUÍ EMPIEZA EL PROBLEMA. Qué se discutió y qué resolvió cada instancia\n   hasta llegar a lo que "
   "se reclama; lo principal, en una o dos frases.\n3. CÓMO RESOLVIÓ LA SALA." in _s["relato"]
   and "¿Cómo resolvió la Sala ese recurso de apelación?" in _s["relato"],
   "relato: el hilo y la pregunta de siempre")
ok("«¿Debía la Sala estudiar" in _s["problemas"] and
   "si la hubo, la resolución del recurso ordinario que la\nconfirmó; no lo que la sentencia definitiva dijo de "
   "pasada." in _s["problemas"], "problemas: el ejemplo y la violación procesal de siempre")
ok("IDENTIFÍCALO por fecha, sala, toca y expediente de origen, y qué confirmó, modificó o revocó—"
   in _s["estructura"] and "un número de toca o un nombre" in _s["estructura"]
   and "NÚMERO DE EXPEDIENTE o toca de origen" in _s["estructura"],
   "estructura: la identificación, la regla de no inventar y el resultando de siempre")
ok(_s["verbos del recurrido"] == ta.VERBOS_DEL_RECURRIDO[AD], "los verbos de siempre")
ok(ta.tecnica_de(AD, "", True)[0] is ta.TECNICA_RESOLUCION["directo_violacion_procesal"],
   "la técnica de siempre")
_e0 = estudio_v1()
ok("«la Sala afirmó X;" in _e0 and "efectos la resolución del recurso ordinario y la actuación viciada;" in _e0,
   "estudio v1: el ejemplo y los efectos congelados")
ok(fr.sujetos_responsable(AD) is fr.SUJETOS_RESPONSABLE, "el respaldo de sujetos, el de siempre")
ok(dg.coletilla_oral_mercantil(AD, {"acto": ACTO}) == "", "sin contexto no hay coletilla")
_t0, _ = ea.autoridad_en_resolutivo(_res_juez, JUZGADO)
ok(_t0 == _res_juez, "sin contexto, «por el Juez…, en el expediente» no se toca (como antes)")
_t0b, _ = ea.autoridad_en_resolutivo(_res_sala, "Segunda Sala Civil del Tribunal Superior")
ok(_t0b == _res_sala.replace("Primera Sala Civil del Tribunal Superior de Justicia",
                             "Segunda Sala Civil del Tribunal Superior"),
   "sin contexto, «por la Sala…, en el toca» se sustituye como antes")

# CON EL ORIGEN PUESTO PERO LAS BANDERAS APAGADAS: idéntico a sin contexto, en
# todo. Es lo que ven las cuentas de fuera hasta que se mida.
# (El estudio se compara contra la misma cuenta de pruebas sin origen: las
# otras banderas del rediseño, «casa» por omisión, también lo mueven.)
# (2-oct-2026) Las banderas de la mejora final del redactor también mueven
# estos textos («casa» por omisión): antecedentes en prosa, inoperancia por
# vicio, preguntas al secretario y propuesta por probabilidad. Apagadas aquí,
# que esta prueba mide el origen.
ct.poner(True, {"banderas": {"instancia_origen": False, "cumplimiento_ejecutoria": False,
                             "antecedentes_en_prosa": False, "inoperancia_por_vicio": False,
                             "preguntas_al_secretario": False, "propuesta_por_probabilidad": False,
                             "supervisor_proyecto": False}}, pruebas=True)
ct.poner_origen(None)
_e_casa = estudio_v1()
ct.poner_origen(oa.origen(JUZGADO, "…el juicio oral mercantil…", "", ACTO + CUMPL))
_apag = textos()
_dist = [n for n in _s if _s[n] != _apag[n]]
ok(not _dist, f"banderas apagadas: todos los textos idénticos a sin contexto{(' — difieren: ' + ', '.join(_dist)) if _dist else ''}")
ok(estudio_v1() == _e_casa, "banderas apagadas: el estudio v1 idéntico al de la misma cuenta sin origen")
ok(dg.coletilla_oral_mercantil(AD, {"acto": ACTO}) == "", "banderas apagadas: sin coletilla")

print()
if FALLOS:
    print("FALLAS:\n  " + "\n  ".join(FALLOS))
    sys.exit(1)
print("RESULTADO: TODAS LAS COMPROBACIONES PASAN")
