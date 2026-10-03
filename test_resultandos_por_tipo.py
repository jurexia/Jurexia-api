# -*- coding: utf-8 -*-
"""EL COMPOSITOR POR TIPO (3-oct-2026), sin red.

    .venv/bin/python test_resultandos_por_tipo.py

Y para la revisión humana, un ejemplo completo de cada tipo y variante:

    MUESTRAS_COMPOSITOR=/ruta/muestras.txt .venv/bin/python test_resultandos_por_tipo.py

Los cuatro tipos por cada variante (adhesivo, returno, Ministerio Público sólo
si consta, ejecutora, única instancia, incidente de suspensión, auto de
sobreseimiento del 81-I-d, mixto, queja de las fracciones I y II con el informe
del 101, revisión fiscal por correo, CDMX): con los datos completos, CERO
«*********» y CERO frases evasivas, el mismo número, la misma responsable y la
misma fecha del acto en el V I S T O y en los resultandos, y todas las fechas
con su año. Con un dato faltante, el hueco y el aviso que lo NOMBRA.
"""
import copy
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import resultandos_por_tipo as rpt
import documento_generado as dg
import meta_lenguaje as ml
import tipos_asunto as ta

FALLOS = []


def ok(c, q):
    print(f"   {'PASA ' if c else 'FALLA'}  {q}")
    if not c:
        FALLOS.append(q)


H = rpt.HUECO
_MESES = ("enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|octubre|"
          "noviembre|diciembre")
_RX_SIN_ANIO = re.compile(rf"\b(?:{_MESES})\b(?! de dos mil)")
# LAS EVASIVAS MEDIDAS (errores.txt y SPEC §3.3 b) y las afirmaciones por
# omisión que el catálogo viejo obligaba a escribir.
_PROHIBIDAS = [
    r"se\s+advierte\s+de\s+las\s+constancias", r"que\s+obran?\s+en\s+autos",
    r"la\s+persona\s+a\s+quien\s+resulta", r"no\s+consta\s+en\s+los\s+datos",
    r"oficial[íi]a\s+de\s+partes\s+de\s+este\s+tribunal", r"omiti[óo]\s+formular\s+pedimento",
    r"oficial[íi]a\s+correspondiente", r"\bde\s+el\b", r"\ba\s+el\b", r"\s{2,}",
    r"por\s+el\s+jueza", r"\bel\s+la\b",
]


def _todo(o):
    return "\n".join([o["visto"]] + [f"{r['titulo']} {r['texto']}" for r in o["resultandos"]])


def _limpio(texto):
    """Las evasivas y descuidos que encuentra, en una lista (vacía si está limpio)."""
    malas = list(ml.perifrasis(texto))
    malas += [rx.pattern for rx in ta._EVASIVAS if rx.search(texto)]
    malas += [p for p in _PROHIBIDAS if re.search(p, texto, re.I)]
    malas += [m.group(0) for m in _RX_SIN_ANIO.finditer(texto)]
    return malas


def _mezclar(base, cambios):
    """Copia profunda de `base` con `cambios` encima; None borra la clave."""
    out = copy.deepcopy(base)
    for k, v in cambios.items():
        if v is None:
            out.pop(k, None)
        elif isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _mezclar(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _titulos(o):
    return [r["titulo"] for r in o["resultandos"]]


def _res(o, titulo):
    for r in o["resultandos"]:
        if r["titulo"].startswith(titulo):
            return r["texto"]
    return ""


# ═══════════════════════════════════════════════════════════════════════════
# LAS FICHAS DE PRUEBA (fechas reales del corpus, coherentes entre sí)
# ═══════════════════════════════════════════════════════════════════════════
DATOS = {"tribunal": "Tercer Tribunal Colegiado en Materias Administrativa y Civil del "
                     "Vigésimo Segundo Circuito", "ciudad": "Querétaro, Querétaro",
         "magistrado": "Luis Armando Pérez Topete"}

AD = {
    "formato": 1, "tipo": "amparo_directo", "numero": "174/2026", "materia": "civil",
    "sede": {"tribunal": DATOS["tribunal"], "ciudad": "Querétaro", "circuito": "XXII",
             "cdmx": False},
    "admision": {"fecha": "2026-02-20"},
    "turno": {"fecha": "2026-03-23", "ponente": "Luis Armando Pérez Topete"},
    "ministerio_publico": "pedimento",
    "promovente": "Ruth Gabriela Larrañaga Silva", "caracter": "quejoso",
    "representante": "Pedro Gómez Ruiz", "figura_representante": "apoderado legal",
    "presentacion": "2026-02-12", "via_presentacion": "responsable",
    "notificacion": "2026-01-26",
    "acto": {"clase": "sentencia", "fecha": "2026-01-22", "toca": "toca familiar 4357/2025",
             "expediente": "515/2022",
             "organo": "Segunda Sala Civil del Tribunal Superior de Justicia del Estado de Querétaro"},
    "responsable": "SEGUNDA SALA CIVIL DEL TRIBUNAL SUPERIOR DE JUSTICIA DEL ESTADO DE QUERÉTARO",
    "terceros": ["Jorge Francisco Vargas Garduza", "Metlife México, S.A. de C.V."],
    "derechos": ["1", "14º", "16", "17"],
    "avisos": [], "fuentes": {}, "fechas_imposibles": [],
}
AD_ADHESIVO = {"quien": "Jorge Francisco Vargas Garduza", "notificacion": "2026-02-24",
               "presentacion": "2026-03-10", "admision": "2026-03-12"}

AR = {
    "formato": 1, "tipo": "amparo_revision", "numero": "513/2025", "materia": "civil",
    "admision": {"fecha": "2025-09-19"},
    "turno": {"fecha": "2025-11-11", "ponente": "Luis Armando Pérez Topete"},
    "promovente": "Impulsora de Desarrollos Inmobiliarios VV, S.A. de C.V.",
    "caracter": "quejoso",
    "representante": "Ana María Ruiz Soto", "figura_representante": "apoderada legal",
    "presentacion": "2025-08-29",
    "acto": {"clase": "sentencia", "fecha": "2025-07-24",
             "organo": "Juez Cuarto de Distrito en Materia de Amparo Civil, Administrativo y "
                       "de Trabajo y de Juicios Federales en el Estado de Querétaro",
             "expediente": "950/2024", "incidente": False, "resolvio": "niega"},
    "demanda": {"fecha": "2025-03-31",
                "autoridades": ["Magistrado Presidente de la Sala Familiar del Tribunal "
                                "Superior de Justicia del Estado de Querétaro"],
                "actos": ["la resolución dictada el tres de marzo de dos mil veinticinco en "
                          "el toca familiar 112/2025"]},
    "audiencia": "2025-04-25", "clase_recurrida": "sentencia",
}
AR_ADHESIVO = {"quien": "Director de Ingresos del Municipio de Querétaro",
               "notificacion": "2025-09-23", "presentacion": "2025-09-29",
               "admision": "2025-10-01"}

Q = {
    "formato": 1, "tipo": "queja", "numero": "150/2026", "materia": "civil",
    "admision": {"fecha": "2026-03-13"},
    "turno": {"fecha": "2026-03-20", "ponente": "Luis Armando Pérez Topete"},
    "promovente": "José Eduardo Aguilar Durán", "caracter": "quejoso",
    "presentacion": "2026-03-10",
    "acto": {"clase": "auto", "fecha": "2026-03-05",
             "organo": "Juez Séptimo de Distrito en el Estado de Querétaro",
             "expediente": "312/2026", "incidente": False,
             "sentido": "desechó la demanda de amparo"},
    "fraccion_97": "I", "inciso_97": "a",
}

RF = {
    "formato": 1, "tipo": "revision_fiscal", "numero": "62/2025", "materia": "administrativa",
    "admision": {"fecha": "2025-11-03"},
    "turno": {"fecha": "2025-12-01", "ponente": "Luis Armando Pérez Topete"},
    # CUARTA RONDA (E8): el papel dice «el Titular»; «Titular» a secas va con
    # «el» y con el aviso de género por omisión (sección 32).
    "promovente": "el Titular de la Jefatura de Servicios Jurídicos del Órgano de Operación "
                  "Administrativa Desconcentrada en Querétaro del Instituto Mexicano del "
                  "Seguro Social",
    "caracter": "autoridad", "presentacion": "2025-10-08", "via_presentacion": "responsable",
    "acto": {"clase": "sentencia", "fecha": "2025-09-18",
             "sentido": "declaró la nulidad lisa y llana de la resolución impugnada"},
    "actora": "María Hernández Ruiz",
    "resolucion_impugnada": {"oficio": "SUB/123/2025", "fecha": "2025-01-15",
                             "autoridad": "Titular de la Subdelegación Querétaro del "
                                          "Instituto Mexicano del Seguro Social"},
    # QUINTA RONDA (F5): la demandada con el artículo del papel; sin él va «el
    # Titular» con el aviso de género por omisión (sección 41).
    "autoridad_demandada": "el Titular de la Subdelegación Querétaro del Instituto Mexicano "
                           "del Seguro Social",
    "sala": "SALA REGIONAL DEL CENTRO II", "expediente_tfja": "1234/25-22-01-1",
}
RF_ADHESIVO = {"quien": "María Hernández Ruiz", "notificacion": "2025-11-06",
               "presentacion": "2025-11-21", "admision": "2025-11-24"}

RETURNO = {"returno": {"fecha": "2026-05-04", "ponente": "Magistrada Ana Laura Campos Juárez"}}

# (nombre, tipo, ficha, datos)
CASOS = [
    ("AD · base (toca y expediente, dos terceros, MP con pedimento, apoderado)", "amparo_directo", AD, DATOS),
    ("AD · ordenadora y ejecutora", "amparo_directo",
     _mezclar(AD, {"ejecutora": "Juzgado Tercero de Primera Instancia Familiar del Distrito "
                                "Judicial de Querétaro"}), DATOS),
    ("AD · amparo adhesivo", "amparo_directo", _mezclar(AD, {"adhesivo": AD_ADHESIVO}), DATOS),
    ("AD · returno", "amparo_directo", _mezclar(AD, RETURNO), DATOS),
    ("AD · MP sin pedimento", "amparo_directo", _mezclar(AD, {"ministerio_publico": "sin_pedimento"}), DATOS),
    ("AD · MP no consta (no se dice nada)", "amparo_directo", _mezclar(AD, {"ministerio_publico": ""}), DATOS),
    ("AD · única instancia (TFJA, sólo expediente)", "amparo_directo",
     _mezclar(AD, {"materia": "administrativa", "unica_instancia": True,
                   "responsable": "Sala Regional en Querétaro del Tribunal Federal de Justicia "
                                  "Administrativa",
                   "acto": {"toca": "", "expediente": "2210/24-22-01-3", "organo": ""},
                   "terceros": ["el Titular de la Administración Desconcentrada de Auditoría "
                                "Fiscal de Querétaro \"1\""]}), DATOS),
    ("AD · laudo (laboral) y vía electrónica", "amparo_directo",
     _mezclar(AD, {"materia": "laboral", "via_presentacion": "electronica",
                   "responsable": "Tribunal Laboral de la Ciudad de Querétaro",
                   "acto": {"clase": "laudo", "toca": "", "expediente": "J-145/2024",
                            "organo": ""}}), DATOS),
    ("AD · CDMX (supletorio del CNPCF)", "amparo_directo",
     _mezclar(AD, {"sede": {"tribunal": "Décimo Tribunal Colegiado en Materia Civil del "
                                        "Primer Circuito", "ciudad": "Ciudad de México",
                            "circuito": "I", "cdmx": True}}),
     _mezclar(DATOS, {"tribunal": "Décimo Tribunal Colegiado en Materia Civil del Primer "
                                  "Circuito", "ciudad": "Ciudad de México"})),
    ("AD · el número con la etiqueta del encabezado", "amparo_directo",
     _mezclar(AD, {"numero": "ADC 625-2024 ORAL MERCANTIL", "materia": "mercantil",
                   "presentacion": "2024-11-12", "admision": {"fecha": "2024-11-20"},
                   "acto": {"fecha": "2024-08-29"}}), DATOS),
    ("AR · sentencia que niega, audiencia en otra fecha", "amparo_revision", AR, DATOS),
    ("AR · concede, audiencia y sentencia el mismo día", "amparo_revision",
     _mezclar(AR, {"audiencia": "2025-07-24", "acto": {"resolvio": "concede"}}), DATOS),
    ("AR · sobresee", "amparo_revision", _mezclar(AR, {"acto": {"resolvio": "sobresee"}}), DATOS),
    ("AR · mixto (sobresee y niega)", "amparo_revision",
     _mezclar(AR, {"acto": {"resolvio": "sobresee_niega"}}), DATOS),
    ("AR · incidente de suspensión (concede la definitiva)", "amparo_revision",
     _mezclar(AR, {"clase_recurrida": "interlocutoria_suspension", "audiencia": None,
                   "acto": {"clase": "interlocutoria", "incidente": True, "resolvio": "concede"}}), DATOS),
    ("AR · auto de sobreseimiento fuera de audiencia (81-I-d)", "amparo_revision",
     _mezclar(AR, {"clase_recurrida": "auto_sobreseimiento", "audiencia": None,
                   "acto": {"clase": "auto", "resolvio": "sobresee"}}), DATOS),
    ("AR · revisión adhesiva y MP", "amparo_revision",
     _mezclar(AR, {"adhesivo": AR_ADHESIVO, "ministerio_publico": "sin_pedimento"}), DATOS),
    ("AR · returno", "amparo_revision", _mezclar(AR, RETURNO), DATOS),
    ("AR · recurre la autoridad responsable", "amparo_revision",
     _mezclar(AR, {"promovente": "DIRECTOR DE INGRESOS DEL MUNICIPIO DE QUERÉTARO",
                   "caracter": "autoridad", "representante": None, "figura_representante": None,
                   "acto": {"resolvio": "concede"}}),
     _mezclar(DATOS, {"quejoso": "Impulsora de Desarrollos Inmobiliarios VV, S.A. de C.V."})),
    ("AR · recurre la tercera interesada", "amparo_revision",
     _mezclar(AR, {"promovente": "Unión de Trabajadores de la Construcción, C.T.M.",
                   "caracter": "tercero", "representante": None, "figura_representante": None,
                   "acto": {"resolvio": "concede"}}),
     _mezclar(DATOS, {"quejoso": "Impulsora de Desarrollos Inmobiliarios VV, S.A. de C.V."})),
    ("Q · fr. I, inciso a) (desechó la demanda)", "queja", Q, DATOS),
    ("Q · fr. I, inciso b) en el incidente de suspensión (MP con pedimento)", "queja",
     _mezclar(Q, {"inciso_97": "b", "materia": "administrativa", "ministerio_publico": "pedimento",
                  "acto": {"incidente": True, "sentido": "negó la suspensión provisional",
                           "organo": "JUEZA TERCERA DE DISTRITO EN EL ESTADO DE QUERÉTARO"}}), DATOS),
    ("Q · fr. I, inciso e) (anclado)", "queja",
     _mezclar(Q, {"inciso_97": "e",
                  "acto": {"sentido": "desechó el incidente de nulidad de notificaciones "
                                      "planteado por la parte quejosa"}}), DATOS),
    ("Q · fr. II, inciso b) con el informe del 101", "queja",
     _mezclar(Q, {"fraccion_97": "II", "inciso_97": "b", "informe_101": {"fecha": "2026-03-18"},
                  # CUARTA RONDA (E7): el auto que registra y pide el informe es
                  # otro que el que lo tiene por rendido y admite.
                  "registro": {"fecha": "2026-03-13"}, "admision": {"fecha": "2026-03-18"},
                  "turno": {"fecha": "2026-03-24"},
                  "caracter": "tercero", "promovente": "Banco Nacional de México, S.A.",
                  "acto": {"organo": "Primera Sala Civil del Tribunal Superior de Justicia "
                                     "del Estado de Querétaro", "expediente": "174/2026",
                           "sentido": "concedió la suspensión del acto reclamado"}}), DATOS),
    ("Q · returno", "queja", _mezclar(Q, RETURNO), DATOS),
    ("RF · oficio presentado ante la Sala", "revision_fiscal", RF, DATOS),
    ("RF · por correo (depósito y recepción)", "revision_fiscal",
     _mezclar(RF, {"via_presentacion": "postal", "deposito_postal": "2025-10-08",
                   "presentacion": "2025-10-14"}), DATOS),
    ("RF · revisión adhesiva", "revision_fiscal", _mezclar(RF, {"adhesivo": RF_ADHESIVO}), DATOS),
    ("RF · returno", "revision_fiscal", _mezclar(RF, RETURNO), DATOS),
    ("RF · vía electrónica", "revision_fiscal", _mezclar(RF, {"via_presentacion": "electronica"}), DATOS),
    # SEGUNDA RONDA (3-oct-2026). Al final, para no mover los índices de arriba.
    ("Q · fr. II marcada «en el incidente» (en el directo no hay incidente)", "queja",
     _mezclar(Q, {"fraccion_97": "II", "inciso_97": "b", "informe_101": {"fecha": "2026-03-18"},
                  "registro": {"fecha": "2026-03-13"}, "admision": {"fecha": "2026-03-18"},
                  "turno": {"fecha": "2026-03-24"},
                  "acto": {"organo": "Primera Sala Civil del Tribunal Superior de Justicia "
                                     "del Estado de Querétaro", "expediente": "174/2026",
                           "incidente": True,
                           "sentido": "negó la suspensión del acto reclamado"}}), DATOS),
    ("AD · razones sociales que cierran con punto (sin «S.A..»)", "amparo_directo",
     _mezclar(AD, {"promovente": "Banco del Centro, S.A.", "representante": None,
                   "figura_representante": None,
                   "terceros": ["Metlife México, S.A. de C.V.", "Unión de Crédito del Bajío, S.A. de C.V."],
                   "adhesivo": {"quien": "Metlife México, S.A. de C.V.",
                                "notificacion": "2026-02-24", "presentacion": "2026-03-10",
                                "admision": "2026-03-12"}}), DATOS),
    ("AR · autoridad con razón social y actos que cierran con punto", "amparo_revision",
     _mezclar(AR, {"promovente": "Impulsora de Desarrollos Inmobiliarios VV, S.A. de C.V.",
                   "demanda": {"autoridades": ["Banco Nacional de Obras y Servicios Públicos, S.N.C.",
                                               "Director de Ingresos del Municipio de Querétaro"],
                               "actos": ["La cancelación del crédito otorgado a Constructora Alfa, "
                                         "S.A. de C.V.", "Su ejecución."]}}), DATOS),
    ("RF · con fracción del 63, cuantía y fundamento del surtimiento", "revision_fiscal",
     _mezclar(RF, {"fraccion_63": "I", "cuantia": "$847,738.77",
                   "fundamento_surtimiento": "el artículo 70 de la Ley Federal de Procedimiento "
                                             "Contencioso Administrativo",
                   "actora": "Comercializadora Ejemplo, S.A. de C.V."}), DATOS),
    # SEXTA RONDA (3-oct-2026, C3 y C6). Al final, para no mover los índices: así
    # pasan también por la sección 1 (cero huecos, cero avisos, el mismo número)
    # y salen en las muestras para la revisión humana.
    ("Q · fr. I con la demanda de amparo (C3: abre con «Demanda de amparo.»)", "queja",
     _mezclar(Q, {"quejoso": "José Eduardo Aguilar Durán",
                  "demanda": {"fecha": "2026-02-27",
                              "autoridades": ["Sala Familiar del Tribunal Superior de Justicia del Estado "
                                              "de Querétaro",
                                              "Juez Noveno de Primera Instancia Familiar del Distrito "
                                              "Judicial de Querétaro"],
                              "actos": ["La resolución de veinte de enero de dos mil veintiséis, en el "
                                        "toca familiar 88/2026.", "Su ejecución."]}}), DATOS),
    ("Q · relacionada con el amparo en revisión de la misma sesión (C6, Q 24/2026)", "queja",
     _mezclar(Q, {"relacionados": [{"tipo": "amparo_revision", "numero": "298/2025",
                                    "estado": "misma_sesion"}]}), DATOS),
    ("AD · relacionado con una revisión fiscal y con un amparo ya resuelto (C6, AD 469/2024)",
     "amparo_directo",
     _mezclar(AD, {"relacionados": [{"tipo": "revision_fiscal", "numero": "33/2024", "estado": "misma_sesion"},
                                    {"tipo": "amparo_directo", "numero": "12/2024", "estado": "resuelto"}]}),
     DATOS),
]
N_RONDA1 = 30         # los casos de la primera ronda (índices fijos más abajo)


print("\n0 · EL HUECO ES EL DE SIEMPRE Y LA PUERTA NO LANZA")
ok(rpt.HUECO == dg.HUECO, "el hueco es el mismo «*********» de documento_generado")
_o = rpt.componer("amparo_lunar", AD, DATOS)
ok(_o["visto"] == "" and _o["resultandos"] == [] and "DESCONOCIDO" in _o["avisos"][0],
   "un tipo desconocido no compone nada y lo dice (el cableado cae al camino viejo)")
for _t in rpt.TIPOS:
    _o = rpt.componer(_t, None, None)
    ok(isinstance(_o["resultandos"], list) and _o["resultandos"] and H in _o["visto"],
       f"{_t} sin ficha ni datos: no lanza, compone con huecos")
_f0 = copy.deepcopy(AD)
rpt.componer("amparo_directo", _f0, DATOS)
ok(_f0 == AD, "es puro: no toca la ficha que recibe")
ok(rpt.componer("AD", AD, DATOS)["visto"] == rpt.componer("amparo_directo", AD, DATOS)["visto"],
   "acepta los alias del tipo («AD»)")
ok(rpt.activo() in (True, False), "activo() responde sin lanzar")


print("\n1 · CON LOS DATOS COMPLETOS: CERO HUECOS, CERO EVASIVAS, CERO AVISOS")
for nombre, tipo, ficha, datos in CASOS:
    o = rpt.componer(tipo, ficha, datos)
    todo = _todo(o)
    ok(H not in todo, f"{nombre}: sin «*********»")
    malas = _limpio(todo)
    ok(not malas, f"{nombre}: sin evasivas ni fechas sin año {malas[:3] if malas else ''}")
    ok(not o["avisos"], f"{nombre}: sin avisos {o['avisos'][:2] if o['avisos'] else ''}")
    ok(o["visto"].startswith("para resolver ") and o["visto"].endswith("; y,"),
       f"{nombre}: el V I S T O es el cuerpo, en minúscula, y cierra «; y,»")
    ex = o["datos_extra"]
    ok(ex.get("numero") and ex["numero"] in o["visto"] and
       sum(ex["numero"] in r["texto"] for r in o["resultandos"]) >= 1,
       f"{nombre}: el mismo número en el V I S T O y en el trámite ({ex.get('numero')})")
    ok(ex.get("fecha_acto") and ex["fecha_acto"] in o["visto"] and
       any(ex["fecha_acto"] in r["texto"] for r in o["resultandos"]),
       f"{nombre}: la misma fecha del acto en el V I S T O y en los resultandos")
    ok(ex.get("organo_acto") and ex["organo_acto"] in o["visto"] and
       any(ex["organo_acto"] in r["texto"] for r in o["resultandos"]),
       f"{nombre}: el mismo órgano del acto en el V I S T O y en los resultandos")
    ok(not any(re.search(r"art[íi]culo\s+101", r["texto"]) for r in o["resultandos"]
               if r["titulo"].startswith(("Turno", "Returno"))),
       f"{nombre}: ni el turno ni el returno citan el artículo 101")


print("\n2 · AMPARO DIRECTO: LAS FÓRMULAS")
o = rpt.componer("amparo_directo", AD, DATOS)
ok(o["visto"] == ("para resolver el juicio de amparo directo civil 174/2026, promovido por "
                  "Ruth Gabriela Larrañaga Silva, contra la sentencia dictada el veintidós de "
                  "enero de dos mil veintiséis, por la Segunda Sala Civil del Tribunal Superior "
                  "de Justicia del Estado de Querétaro, en el toca familiar 4357/2025, derivado "
                  "del expediente 515/2022; y,"),
   "V I S T O completo, con la responsable en prosa (no en versales) y su artículo")
ok(_titulos(o) == ["Presentación de la demanda de amparo.", "Derechos humanos que se estiman "
                   "vulnerados.", "Tercero interesado.", "Trámite del juicio de amparo.", "Turno."],
   "los cinco rótulos del corpus, en su orden")
t1 = _res(o, "Presentación")
ok("ante la Segunda Sala Civil del Tribunal Superior" in t1 and "oficial" not in t1.lower(),
   "la demanda, ante la responsable (art. 176), sin oficialía inventada")
ok(", por conducto de su apoderado legal Pedro Gómez Ruiz, promovió" in t1,
   "con su apoderado, entre comas")
ok("AUTORIDAD RESPONSABLE:\nSegunda Sala Civil del Tribunal Superior de Justicia del Estado "
   "de Querétaro\nACTO RECLAMADO:\nLa sentencia dictada el veintidós de enero de dos mil "
   "veintiséis, en el toca familiar 4357/2025, derivado del expediente 515/2022." in t1,
   "los bloques AUTORIDAD RESPONSABLE / ACTO RECLAMADO, uno por renglón")
ok("los artículos 1o., 14, 16 y 17 de la Constitución" in _res(o, "Derechos"),
   "los derechos con «1o.» y la lista en español")
ok(_res(o, "Tercero") == "Tienen ese carácter Jorge Francisco Vargas Garduza y Metlife México, "
   "S.A. de C.V.", "dos terceros: «Tienen», en plural")
t4 = _res(o, "Trámite")
ok("registró la demanda con el número 174/2026, la admitió a trámite" in t4 and
   "artículo 181 de la Ley de Amparo" in t4, "trámite: auto de Presidencia, número y art. 181")
ok("formuló pedimento." in t4 and "no formuló" not in t4, "el MP, como consta: formuló pedimento")
ok(_res(o, "Turno") == ("Por acuerdo de veintitrés de marzo de dos mil veintiséis, se turnaron "
                        "los autos a la ponencia de Luis Armando Pérez Topete, para la "
                        "elaboración del proyecto de resolución, en términos del artículo 183 "
                        "de la Ley de Amparo."),
   "turno con el 183 y el ponente sin inferir su género")
ok(o["datos_extra"]["toca"] == "toca familiar 4357/2025" and
   o["datos_extra"]["expediente"] == "515/2022" and
   o["datos_extra"]["fecha_acto_iso"] == "2026-01-22" and
   o["datos_extra"]["responsable"] == "Segunda Sala Civil del Tribunal Superior de Justicia "
                                      "del Estado de Querétaro",
   "datos_extra: toca y expediente separados, fecha ISO y responsable normalizada")
ok(o["datos_extra"]["descripcion_acto"] == "de la sentencia reclamada" and
   o["datos_extra"]["adhesivo"] is None, "datos_extra: descripción del acto y sin adhesivo")

o = rpt.componer("amparo_directo", _mezclar(AD, {"ministerio_publico": "sin_pedimento"}), DATOS)
ok("no formuló pedimento." in _res(o, "Trámite"), "MP: «no formuló pedimento», sólo porque consta")
o = rpt.componer("amparo_directo", _mezclar(AD, {"ministerio_publico": ""}), DATOS)
ok("Ministerio Público" not in _todo(o), "MP que no consta: no se dice NADA (ni omitió ni formuló)")

o = rpt.componer("amparo_directo", CASOS[1][2], DATOS)
ok(o["visto"].endswith("derivado del expediente 515/2022, y su ejecución; y,") and
   "AUTORIDADES RESPONSABLES:\nSegunda Sala Civil del Tribunal Superior de Justicia del Estado "
   "de Querétaro (ordenadora) y Juzgado Tercero de Primera Instancia Familiar del Distrito "
   "Judicial de Querétaro (ejecutora)" in _res(o, "Presentación") and
   "en contra de las autoridades y del acto" in _res(o, "Presentación"),
   "ejecutora: «y su ejecución» y las dos autoridades con su papel")

o = rpt.componer("amparo_directo", CASOS[2][2], DATOS)
ok("Por auto de doce de marzo de dos mil veintiséis, se admitió el amparo adhesivo promovido "
   "por Jorge Francisco Vargas Garduza." in _res(o, "Trámite"),
   "adhesivo: su admisión en el trámite, con quién y cuándo")
ok(o["datos_extra"]["adhesivo"] == AD_ADHESIVO, "datos_extra lleva el adhesivo")

o = rpt.componer("amparo_directo", CASOS[3][2], DATOS)
ok(_titulos(o)[-1] == "Returno." and _res(o, "Returno") == (
    "Por acuerdo de cuatro de mayo de dos mil veintiséis, se returnaron los autos a la ponencia "
    "de la Magistrada Ana Laura Campos Juárez, para la elaboración del proyecto de resolución."),
   "returno: el cargo de la fuente se respeta («de la Magistrada …») y sin artículo 101")

o = rpt.componer("amparo_directo", CASOS[6][2], DATOS)
ok("juicio de amparo directo administrativo 174/2026" in o["visto"] and
   "en el expediente 2210/24-22-01-3; y," in o["visto"] and "toca" not in o["visto"],
   "única instancia: «administrativo» y sólo el expediente")
ok(_res(o, "Tercero").startswith("Tiene ese carácter el Titular de la Administración"),
   "una autoridad tercera, con su artículo y el cargo en genérico («el Titular»)")
# TERCERA RONDA (3-oct-2026, punta a punta AD 274/2025): el toca es un HECHO de
# la ficha y la marca de única instancia una inferencia. Con toca no hay única
# instancia; antes se tiraba el toca y se escribía sólo el expediente.
o = rpt.componer("amparo_directo", _mezclar(AD, {"acto": {"instancia": "unica",
                                                         "toca": "toca 9/2025"}}), DATOS)
ok("en el toca 9/2025, derivado del expediente 515/2022; y," in o["visto"] and
   not o["datos_extra"]["unica_instancia"] and
   any("ÚNICA INSTANCIA NO CUADRA" in a and "(acto.instancia)" in a for a in o["avisos"]),
   "marca de única instancia (acto.instancia) con toca: manda el toca, y se avisa del choque")

o = rpt.componer("amparo_directo", CASOS[7][2], DATOS)
ok("contra el laudo dictado el" in o["visto"] and "\nEl laudo dictado el" in _res(o, "Presentación")
   and "amparo directo laboral" in o["visto"], "laudo: «el laudo dictado», en la materia laboral")
ok("ante el Tribunal Laboral de la Ciudad de Querétaro, por vía electrónica," in _res(o, "Presentación"),
   "vía electrónica: se dice, y ante quién")

o = rpt.componer("amparo_directo", CASOS[8][2], CASOS[8][3])
_sup = o["datos_extra"]["supletorio"]
if hasattr(ta, "supletorio"):
    ok(_sup == ta.supletorio(CASOS[8][3]["tribunal"], CASOS[8][3]["ciudad"]) and
       "Nacional" in str(_sup.get("documentales", "")),
       "CDMX: el supletorio lo decide tipos_asunto.supletorio y es el CNPCF")
    _sup_q = rpt.componer("amparo_directo", AD, DATOS)["datos_extra"]["supletorio"]
    ok("Federal de Procedimientos Civiles" in str(_sup_q.get("documentales", "")),
       "fuera de CDMX: el CFPC")
else:
    ok(_sup == {}, "sin tipos_asunto.supletorio todavía: el supletorio va vacío (no se inventa)")

o = rpt.componer("amparo_directo", CASOS[9][2], DATOS)
ok("amparo directo civil 625/2024," in o["visto"] and "ADC" not in _todo(o) and
   "ORAL" not in _todo(o), "la etiqueta del encabezado no se cuela: «625/2024», materia «civil»")

o = rpt.componer("amparo_directo", _mezclar(AD, {"terceros": []}), DATOS)
ok("Tercero interesado." not in _titulos(o) and
   any("NO CONSTA TERCERO INTERESADO" in a for a in o["avisos"]),
   "sin tercero: se omite el resultando y se avisa (no se sustituye con perífrasis)")
o = rpt.componer("amparo_directo", _mezclar(AD, {"derechos": ["14"]}), DATOS)
ok("en el artículo 14 de la Constitución" in _res(o, "Derechos"), "un solo artículo, en singular")


print("\n3 · AMPARO EN REVISIÓN: EL VERBO, LA CLASE Y QUIÉN RECURRE")
o = rpt.componer("amparo_revision", AR, DATOS)
ok(o["visto"] == ("para resolver el recurso de revisión 513/2025, interpuesto por Impulsora de "
                  "Desarrollos Inmobiliarios VV, S.A. de C.V., contra la sentencia dictada el "
                  "veinticuatro de julio de dos mil veinticinco, por el Juzgado Cuarto de "
                  "Distrito en Materia de Amparo Civil, Administrativo y de Trabajo y de "
                  "Juicios Federales en el Estado de Querétaro, en el juicio de amparo "
                  "indirecto 950/2024; y,"),
   "V I S T O del AR completo, con el juzgado en forma de órgano (cuarta ronda, E10)")
ok(_titulos(o) == ["Presentación de la demanda de amparo indirecto.",
                   "Trámite del juicio de amparo indirecto.",
                   "Interposición y trámite del recurso de revisión.", "Turno."],
   "los rótulos del AR, sin «Derechos humanos» (0 de 58 en la ponencia de David)")
t1 = _res(o, "Presentación")
ok(t1.startswith("Por escrito presentado el treinta y uno de marzo de dos mil veinticinco, "
                 "Impulsora de Desarrollos Inmobiliarios VV, S.A. de C.V. promovió juicio de "
                 "amparo indirecto en contra de la autoridad y el acto") and
   "\nAUTORIDAD RESPONSABLE:\nMagistrado Presidente de la Sala Familiar" in t1 and
   "\nACTO RECLAMADO:\nLa resolución dictada el tres de marzo de dos mil veinticinco en el toca "
   "familiar 112/2025." in t1, "la demanda con su autoridad y su acto anclados, en singular")
ok("el veinticinco de abril de dos mil veinticinco se celebró la audiencia constitucional y el "
   "veinticuatro de julio de dos mil veinticinco se dictó sentencia, en la que se negó el amparo."
   in _res(o, "Trámite del juicio"), "niega: audiencia y sentencia en fechas distintas")
t3 = _res(o, "Interposición")
ok(t3.startswith("Inconforme, Impulsora de Desarrollos Inmobiliarios VV, S.A. de C.V., en su "
                 "carácter de parte quejosa, por conducto de su apoderada legal Ana María Ruiz "
                 "Soto, interpuso recurso de revisión por escrito presentado el veintinueve de "
                 "agosto de dos mil veinticinco.") and "lo registró con el número 513/2025" in t3,
   "interposición: carácter sin género, apoderada de la fuente, número")
ok("artículo 92 de la Ley de Amparo" in _res(o, "Turno"), "turno con el 92")
ok(o["datos_extra"]["juicio_amparo"] == "950/2024" and o["datos_extra"]["resolvio"] == "niega"
   and o["datos_extra"]["clase_recurrida"] == "sentencia", "datos_extra: juicio, verbo y clase")

o = rpt.componer("amparo_revision", CASOS[11][2], DATOS)
ok("el veinticuatro de julio de dos mil veinticinco se celebró la audiencia constitucional y se "
   "dictó sentencia, en la que se concedió el amparo." in _res(o, "Trámite del juicio"),
   "audiencia y sentencia el mismo día: una sola fecha")
o = rpt.componer("amparo_revision", CASOS[12][2], DATOS)
ok("en la que se sobreseyó en el juicio." in _res(o, "Trámite del juicio"), "sobresee")
o = rpt.componer("amparo_revision", CASOS[13][2], DATOS)
ok("en la que en una parte se sobreseyó en el juicio y en otra se negó el amparo." in
   _res(o, "Trámite del juicio"), "mixto: sobreseyó en una parte y negó en otra")
o = rpt.componer("amparo_revision", _mezclar(AR, {"acto": {"resolvio": "mixto",
                                                          "resolvio_mixto": "sobresee_concede"}}), DATOS)
ok("en la que en una parte se sobreseyó en el juicio y en otra se concedió el amparo." in
   _res(o, "Trámite del juicio") and not o["avisos"],
   "mixto como lo guarda ficha_tramite («mixto» + resolvio_mixto)")
o = rpt.componer("amparo_revision", CASOS[14][2], DATOS)
ok("contra la sentencia interlocutoria dictada el" in o["visto"] and
   "en el incidente de suspensión relativo al juicio de amparo indirecto 950/2024; y," in o["visto"]
   and "Trámite del incidente de suspensión." in _titulos(o) and
   "se celebró la audiencia incidental, en la que se concedió la suspensión definitiva." in
   _res(o, "Trámite del incidente"), "incidente en revisión: interlocutoria, audiencia incidental")
o = rpt.componer("amparo_revision", CASOS[15][2], DATOS)
ok("contra el auto dictado el" in o["visto"] and
   "por auto de veinticuatro de julio de dos mil veinticinco se sobreseyó en el juicio fuera de "
   "la audiencia constitucional." in _res(o, "Trámite del juicio"),
   "auto de sobreseimiento (81-I-d): «contra el auto» y fuera de la audiencia")
o = rpt.componer("amparo_revision", CASOS[16][2], DATOS)
t3 = _res(o, "Interposición")
ok("Por auto de uno de octubre de dos mil veinticinco, se tuvo al Director de Ingresos del "
   "Municipio de Querétaro interponiendo revisión adhesiva." in t3 and
   t3.endswith("no formuló pedimento."), "revisión adhesiva y el MP como consta")
o = rpt.componer("amparo_revision", CASOS[18][2], CASOS[18][3])
ok("interpuesto por el Director de Ingresos del Municipio de Querétaro, contra" in o["visto"] and
   "Inconforme, el Director de Ingresos del Municipio de Querétaro, en su carácter de autoridad "
   "responsable, interpuso" in _res(o, "Interposición") and "DIRECTOR" not in _todo(o),
   "recurre la autoridad: en prosa, con su artículo y su carácter")
o = rpt.componer("amparo_revision", CASOS[19][2], CASOS[19][3])
ok("en su carácter de parte tercera interesada, interpuso" in _res(o, "Interposición") and
   "Unión de Trabajadores de la Construcción, C.T.M." in o["visto"],
   "recurre la tercera interesada: «parte tercera interesada», sin concordar por el nombre")


print("\n4 · QUEJA: LA FRACCIÓN, EL INCISO Y SU COLA, UNA VEZ")
o = rpt.componer("queja", Q, DATOS)
ok(o["visto"] == ("para resolver el recurso de queja civil 150/2026, interpuesto por José "
                  "Eduardo Aguilar Durán, en contra del auto de cinco de marzo de dos mil "
                  "veintiséis, dictado por el Juzgado Séptimo de Distrito en el Estado de "
                  "Querétaro, en el juicio de amparo indirecto 312/2026; y,"),
   "V I S T O de la queja, con la materia concordada con «queja» y el órgano en forma de "
   "órgano («Juzgado», no «Juez»: tercera ronda)")
ok(_titulos(o) == ["Interposición del recurso de queja.", "Trámite del recurso.",
                   "Turno del asunto."], "abre con la interposición (ponencia de David)")
ok(_res(o, "Interposición") == (
    "Por escrito presentado el diez de marzo de dos mil veintiséis, José Eduardo Aguilar Durán, "
    "parte quejosa en el juicio de amparo indirecto 312/2026, interpuso recurso de queja en "
    "contra del auto de cinco de marzo de dos mil veintiséis, dictado por el Juzgado Séptimo de "
    "Distrito en el Estado de Querétaro, en el que desechó la demanda de amparo."),
   "interposición: carácter, juicio, auto, órgano y sentido del catálogo")
ok("artículo" not in _res(o, "Turno del asunto"), "el turno de la queja va sin artículo")
ex = o["datos_extra"]
ok(ex["fraccion_97"] == "I" and ex["inciso_97"] == "a" and
   ex["cola_97"] == "por el cual se desechó la demanda de amparo" and
   ex["descripcion_acto"] == "del auto que desechó la demanda de amparo" and
   ex["via_amparo"] == "indirecto", "datos_extra: fracción, inciso, cola y descripción")
o = rpt.componer("queja", _mezclar(Q, {"acto": {"sentido": "admitió la demanda de amparo"}}), DATOS)
ok(o["datos_extra"]["cola_97"] == "por el cual se admitió la demanda de amparo",
   "inciso a) que ADMITE: la cola dice lo que proveyó este auto, no «desechó»")
o = rpt.componer("queja", CASOS[21][2], DATOS)
ok("queja administrativa 150/2026" in o["visto"] and
   "en el incidente de suspensión relativo al juicio de amparo indirecto 312/2026; y," in o["visto"]
   and "dictado por el Juzgado Tercero de Distrito en el Estado de Querétaro, en el incidente de "
   "suspensión, en el que negó la suspensión provisional." in _res(o, "Interposición") and
   o["datos_extra"]["cola_97"] == "en el que se negó la suspensión provisional" and
   o["datos_extra"]["juzgado"] == "Juzgado Tercero de Distrito en el Estado de Querétaro" and
   "formuló pedimento" in _res(o, "Trámite"),
   "inciso b) en el incidente: «JUEZA TERCERA» pasa a «Juzgado Tercero» (forma de órgano), "
   "su cola y el MP que consta")
o = rpt.componer("queja", CASOS[22][2], DATOS)
ok(o["datos_extra"]["cola_97"] == ("en el que se desechó el incidente de nulidad de "
                                   "notificaciones planteado por la parte quejosa"),
   "inciso e): la cola con el sentido anclado")
o = rpt.componer("queja", CASOS[23][2], DATOS)
ok("en el juicio de amparo directo 174/2026; y," in o["visto"] and
   "parte tercera interesada en el juicio de amparo directo 174/2026" in _res(o, "Interposición"),
   "fracción II: juicio de amparo DIRECTO")
ok(_res(o, "Trámite") == (
    "Por auto de Presidencia de trece de marzo de dos mil veintiséis, este Tribunal Colegiado "
    "registró el recurso con el número 150/2026 y requirió a la autoridad responsable su informe "
    "con justificación sobre la materia de la queja (artículo 101 de la Ley de Amparo); por auto "
    "de dieciocho de marzo de dos mil veintiséis lo tuvo por rendido y admitió el recurso."),
   "fracción II: requiere el informe del 101 y luego admite, en dos autos")
ok(o["datos_extra"]["fraccion_97"] == "II" and o["datos_extra"]["via_amparo"] == "directo",
   "datos_extra: fracción II, vía directa")
o = rpt.componer("queja", _mezclar(Q, {"fraccion_97": "", "acto": {"via": "directo",
                                                                  "organo": "Autoridad X"}}), DATOS)
ok(o["datos_extra"]["fraccion_97"] == "II" and "juicio de amparo directo 312/2026" in o["visto"],
   "sin fracción, la vía de la ficha (acto.via) la decide")
# SEGUNDA RONDA: en el amparo directo la suspensión la provee la responsable SIN
# incidente formal (arts. 190-191 LA). Aunque la ficha traiga la marca, la
# fracción II nunca dice «incidente de suspensión».
_Q2 = [c for c in CASOS if c[0].startswith("Q · fr. II marcada")][0]
o = rpt.componer("queja", _Q2[2], DATOS)
ok(o["visto"] == ("para resolver el recurso de queja civil 150/2026, interpuesto por José Eduardo "
                  "Aguilar Durán, en contra del auto de cinco de marzo de dos mil veintiséis, dictado "
                  "por la Primera Sala Civil del Tribunal Superior de Justicia del Estado de "
                  "Querétaro, en el juicio de amparo directo 174/2026; y,"),
   "fr. II con la marca de incidente: el V I S T O dice «en el juicio de amparo directo», sin incidente")
ok(_res(o, "Interposición") == (
    "Por escrito presentado el diez de marzo de dos mil veintiséis, José Eduardo Aguilar Durán, parte "
    "quejosa en el juicio de amparo directo 174/2026, interpuso recurso de queja en contra del auto de "
    "cinco de marzo de dos mil veintiséis, dictado por la Primera Sala Civil del Tribunal Superior de "
    "Justicia del Estado de Querétaro, en el que negó la suspensión del acto reclamado."),
   "fr. II: el resultando primero tampoco dice «en el incidente de suspensión»")
ok("incidente" not in _todo(o).lower() and o["datos_extra"]["incidente"] is False,
   "fr. II: ni una mención del incidente, y datos_extra.incidente es False")
o = rpt.componer("queja", _mezclar(_Q2[2], {"caracter": ""}), DATOS)
ok("dictado por la Primera Sala Civil del Tribunal Superior de Justicia del Estado de Querétaro, "
   "relativo al juicio de amparo directo 174/2026, en el que negó la suspensión" in _res(o, "Interposición")
   and "incidente" not in _todo(o).lower(),
   "fr. II sin carácter del recurrente: «…dictado por la responsable, relativo al juicio de amparo directo…»")
o = rpt.componer("queja", CASOS[21][2], DATOS)
ok("en el incidente de suspensión relativo al juicio de amparo indirecto 312/2026; y," in o["visto"]
   and o["datos_extra"]["incidente"] is True,
   "fr. I: el incidente de suspensión del amparo indirecto sí se dice")


print("\n5 · REVISIÓN FISCAL")
o = rpt.componer("revision_fiscal", RF, DATOS)
ok(o["visto"] == ("para resolver el recurso de revisión fiscal número 62/2025, interpuesto por "
                  "la parte citada al rubro, contra la sentencia de dieciocho de septiembre de "
                  "dos mil veinticinco, dictada por la Sala Regional del Centro II del Tribunal "
                  "Federal de Justicia Administrativa, en el juicio contencioso administrativo "
                  "1234/25-22-01-1; y,"), "V I S T O de la RF, con el romano de la Sala intacto")
ok(_titulos(o) == ["Trámite del juicio contencioso administrativo.",
                   "Interposición del recurso de revisión fiscal.",
                   "Trámite del recurso de revisión fiscal.", "Turno."], "los rótulos de la RF")
ok(_res(o, "Trámite del juicio") == (
    "María Hernández Ruiz demandó la nulidad de la resolución contenida en el oficio "
    "SUB/123/2025, de quince de enero de dos mil veinticinco, emitida por el Titular de la "
    "Subdelegación Querétaro del Instituto Mexicano del Seguro Social. El conocimiento "
    "correspondió a la Sala Regional del Centro II del Tribunal Federal de Justicia "
    "Administrativa, que lo registró con el número 1234/25-22-01-1; seguido el juicio, el "
    "dieciocho de septiembre de dos mil veinticinco dictó sentencia, en la que declaró la "
    "nulidad lisa y llana de la resolución impugnada."), "el juicio de nulidad, con su sentido")
ok(_res(o, "Interposición").startswith(
    "Inconforme, el Titular de la Jefatura de Servicios Jurídicos del Órgano de Operación "
    "Administrativa Desconcentrada en Querétaro del Instituto Mexicano del Seguro Social, en "
    "representación del Titular de la Subdelegación Querétaro") and
   _res(o, "Interposición").endswith("mediante oficio presentado el ocho de octubre de dos mil "
                                     "veinticinco ante la Sala Regional del Centro II."),
   "interposición: la unidad, a quién representa y ante la Sala")
ok(_res(o, "Turno").endswith("artículo 92 de la Ley de Amparo, aplicable conforme al artículo "
                             "63, último párrafo, de la Ley Federal de Procedimiento "
                             "Contencioso Administrativo."), "turno: 92 LA por el 63 LFPCA")
ex = o["datos_extra"]
ok(ex["sala"] == "Sala Regional del Centro II" and ex["expediente_tfja"] == "1234/25-22-01-1" and
   ex["actora"] == "María Hernández Ruiz" and ex["fecha_acto_iso"] == "2025-09-18",
   "datos_extra: sala, expediente, actora y fecha ISO")
o = rpt.componer("revision_fiscal", CASOS[26][2], DATOS)
ok("mediante oficio depositado en el Servicio Postal Mexicano el ocho de octubre de dos mil "
   "veinticinco y recibido el catorce de octubre de dos mil veinticinco." in
   _res(o, "Interposición") and o["datos_extra"]["deposito_postal_iso"] == "2025-10-08",
   "por correo: las dos fechas, depósito y recepción")
o = rpt.componer("revision_fiscal", CASOS[27][2], DATOS)
ok("Por auto de veinticuatro de noviembre de dos mil veinticinco, se tuvo a María Hernández "
   "Ruiz adhiriéndose al recurso." in _res(o, "Trámite del recurso"), "la adhesión, en el trámite")


print("\n6 · CON UN DATO FALTANTE: HUECO + AVISO QUE LO NOMBRA")
FALTANTES = [
    ("amparo_directo", AD, {"admision": {"fecha": ""}}, "admision.fecha", "Trámite"),
    ("amparo_directo", AD, {"turno": {"fecha": None}}, "turno.fecha", "Turno"),
    ("amparo_directo", AD, {"derechos": []}, "derechos", "Derechos"),
    ("amparo_directo", AD, {"presentacion": None}, "presentacion", "Presentación"),
    ("amparo_directo", AD, {"acto": {"fecha": ""}}, "acto.fecha", "V I S T O"),
    ("amparo_directo", AD, {"responsable": "", "acto": {"organo": ""}}, "responsable", "V I S T O"),
    ("amparo_directo", AD, {"numero": ""}, "numero", "V I S T O"),
    ("amparo_directo", AD, {"materia": "administrativa y civil"}, "materia", "V I S T O"),
    ("amparo_directo", AD, {"acto": {"toca": "", "expediente": ""}}, "acto.toca / acto.expediente", "V I S T O"),
    ("amparo_directo", _mezclar(AD, {"adhesivo": AD_ADHESIVO}), {"adhesivo": {"admision": ""}},
     "adhesivo.admision", "Trámite"),
    ("amparo_directo", _mezclar(AD, RETURNO), {"returno": {"ponente": ""}}, "returno.ponente", "Returno"),
    ("amparo_revision", AR, {"demanda": {"actos": []}}, "demanda.actos", "Presentación"),
    ("amparo_revision", AR, {"acto": {"resolvio": ""}}, "acto.resolvio", "Trámite del juicio"),
    ("amparo_revision", AR, {"acto": {"resolvio": "mixto"}}, "acto.resolvio", "Trámite del juicio"),
    ("amparo_revision", AR, {"acto": {"expediente": ""}}, "acto.expediente", "V I S T O"),
    ("amparo_revision", AR, {"acto": {"organo": ""}}, "acto.organo", "V I S T O"),
    ("amparo_revision", AR, {"presentacion": ""}, "presentacion", "Interposición"),
    ("queja", Q, {"acto": {"sentido": ""}}, "acto.sentido", "Interposición"),
    ("queja", _mezclar(Q, CASOS[23][2]), {"informe_101": {"fecha": ""}, "admision": {"fecha": ""}},
     "informe_101.fecha", "Trámite"),
    # CUARTA RONDA (E7): en la fracción II el primer auto es el registro.
    ("queja", _mezclar(Q, CASOS[23][2]), {"registro": {"fecha": ""}}, "registro.fecha", "Trámite"),
    ("queja", Q, {"fraccion_97": "", "acto": {"organo": "Autoridad Desconocida"}}, "fraccion_97", "V I S T O"),
    ("revision_fiscal", RF, {"acto": {"sentido": ""}}, "acto.sentido", "Trámite del juicio"),
    ("revision_fiscal", RF, {"actora": ""}, "actora", "Trámite del juicio"),
    ("revision_fiscal", RF, {"via_presentacion": "postal", "deposito_postal": ""},
     "deposito_postal", "Interposición"),
    ("revision_fiscal", RF, {"sala": "", "acto": {"organo": ""}}, "sala", "V I S T O"),
]
for tipo, base, cambio, clave, donde in FALTANTES:
    f = _mezclar(base, cambio)
    o = rpt.componer(tipo, f, {"magistrado": DATOS["magistrado"]})
    sitio = o["visto"] if donde == "V I S T O" else _res(o, donde)
    nombra = [a for a in o["avisos"] if f"({clave})" in a]
    ok(H in sitio and nombra, f"{tipo} sin {clave}: hueco en «{donde}» y aviso que lo nombra")
    ok(not [m for m in _limpio(_todo(o)) if "obra" in m or "advierte" in m],
       f"{tipo} sin {clave}: el hueco no se tapa con una perífrasis")

# Lo que la SPEC manda OMITIR con aviso, sin hueco.
o = rpt.componer("amparo_revision", _mezclar(AR, {"demanda": {"fecha": ""}}), DATOS)
ok(H not in _res(o, "Presentación") and
   _res(o, "Presentación").startswith("Impulsora de Desarrollos Inmobiliarios VV, S.A. de C.V. "
                                      "promovió juicio de amparo indirecto") and
   any("(demanda.fecha)" in a for a in o["avisos"]),
   "AR sin fecha de demanda: menos detalle (no perífrasis), con aviso")
o = rpt.componer("revision_fiscal", _mezclar(RF, {"resolucion_impugnada": {"oficio": "", "fecha": ""}}), DATOS)
ok(H not in _res(o, "Trámite del juicio") and
   "demandó la nulidad de la resolución, emitida por el Titular" in _res(o, "Trámite del juicio") and
   any("(resolucion_impugnada.oficio)" in a for a in o["avisos"]) and
   any("(resolucion_impugnada.fecha)" in a for a in o["avisos"]),
   "RF sin oficio ni fecha de la resolución: se omiten de la frase, con aviso")
o = rpt.componer("revision_fiscal", _mezclar(RF, {"autoridad_demandada": ""}), DATOS)
ok("en representación del Titular de la Subdelegación" in _res(o, "Interposición") and
   any("SE TOMÓ COMO AUTORIDAD DEMANDADA" in a for a in o["avisos"]),
   "RF sin autoridad demandada: la emisora, y se avisa que se tomó")
o = rpt.componer("amparo_directo", _mezclar(AD, {"turno": {"ponente": ""}}), DATOS)
ok("ponencia de Luis Armando Pérez Topete" in _res(o, "Turno") and not o["avisos"],
   "turno sin ponente y sin returno: el de la carátula (mismo dato)")
o = rpt.componer("amparo_directo", _mezclar(AD, {"turno": {"ponente": ""}, **RETURNO}), DATOS)
ok(H in _res(o, "Turno") and any("(turno.ponente)" in a for a in o["avisos"]),
   "turno sin ponente CON returno: no se adivina, hueco")
o = rpt.componer("amparo_directo", _mezclar(AD, {"presentacion": "14/03/2025x"}), DATOS)
ok(any("FECHA ILEGIBLE" in a and "(presentacion" in a for a in o["avisos"]),
   "una fecha ilegible se avisa como ilegible")
o = rpt.componer("amparo_directo", _mezclar(AD, {"presentacion": ""}),
                 _mezclar(DATOS, {"presentacion": "catorce de marzo de dos mil veinticinco"}))
ok("presentado el catorce de marzo de dos mil veinticinco ante" in _res(o, "Presentación"),
   "sin la fecha ISO, la del encargo (ya en letra) sirve de respaldo")
o = rpt.componer("queja", _mezclar(Q, {"acto": {"sentido": ""}, "inciso_97": "a"}), DATOS)
ok(o["datos_extra"]["cola_97"] == "", "queja sin sentido: la cola va vacía (documento_generado usa su tabla)")


print("\n7 · EL CONSIDERANDO DEL ADHESIVO (legitimación y oportunidad contada)")
f = _mezclar(AD, {"adhesivo": AD_ADHESIVO})
rot, txt, avs = rpt.considerando_adhesivo("amparo_directo", f, DATOS)
ok(rot == "Legitimación y oportunidad del amparo adhesivo." and H not in txt and not avs,
   "AD: rótulo del corpus, sin huecos ni avisos")
ok("artículo 182, primer párrafo, de la Ley de Amparo" in txt and
   "artículo 181 de la Ley de Amparo" in txt and "artículo 31, fracción II" in txt and
   "es claro que se promovió oportunamente." in txt, "AD: 182 y 181, surtimiento 31-II, oportuno")
ok("transcurrió del veintiséis de febrero de dos mil veintiséis al diecinueve de marzo de dos mil "
   "veintiséis" in txt and "ni el dieciséis de marzo de dos mil veintiséis" in txt,
   "AD: el cómputo nombra el inhábil del tercer lunes de marzo dentro del plazo")
ok(not _limpio(txt), "AD: sin evasivas ni fechas sin año")
rot, txt, avs = rpt.considerando_adhesivo(
    "amparo_directo", _mezclar(f, {"adhesivo": {"presentacion": "2026-03-25"}}), DATOS)
ok("resulta extemporáneo." in txt and any("EXTEMPORÁNEO" in a for a in avs),
   "AD: fuera de plazo se dice y se avisa")
rot, txt, avs = rpt.considerando_adhesivo(
    "amparo_directo", _mezclar(f, {"adhesivo": {"notificacion": ""}}), DATOS)
ok(H in txt and any("(adhesivo.notificacion)" in a for a in avs) and "oportun" not in
   txt.split("Por lo que hace")[1].replace("oportunidad", ""),
   "AD sin notificación: el veredicto en hueco, sin afirmar la oportunidad")
f = _mezclar(AR, {"adhesivo": AR_ADHESIVO})
rot, txt, avs = rpt.considerando_adhesivo("amparo_revision", f, DATOS)
ok(rot == "Legitimación y oportunidad de la revisión adhesiva." and
   txt.startswith("El Director de Ingresos del Municipio de Querétaro tiene legitimación para "
                  "adherirse a la revisión interpuesta por Impulsora") and
   "artículo 82 de la Ley de Amparo" in txt and "el mismo día, conforme al artículo 31, "
   "fracción I" in txt and "es claro que se interpuso oportunamente." in txt and not avs,
   "AR: el adherente autoridad surte el mismo día (31-I); cinco días del 82")
f = _mezclar(RF, {"adhesivo": RF_ADHESIVO})
rot, txt, avs = rpt.considerando_adhesivo("revision_fiscal", f, DATOS)
ok("artículo 63, penúltimo párrafo, de la Ley Federal de Procedimiento Contencioso "
   "Administrativo" in txt and "quince días" in txt and "oportunamente" in txt and not avs,
   "RF: la adhesión es el PENÚLTIMO párrafo del 63 (el último es la tramitación por la LA)")
_ven = None
import fase0_oportunidad as f0
import datetime as _dt
_c = f0.computar(_dt.date(2025, 11, 6), None, regla="lista", plazo=15, tipo_asunto="amparo_revision")
_ven = _c.vencimiento.isoformat()
rot, txt, avs = rpt.considerando_adhesivo(
    "revision_fiscal", _mezclar(f, {"adhesivo": {"presentacion": _ven}}), DATOS)
# TERCERA RONDA (FIXES_R3 D, RF 4/2025): la adhesiva de la revisión fiscal se
# cuenta con la regla LITERAL del 63, penúltimo párrafo («a partir de la fecha en
# la que se le notifique»), sin surtimiento. El día en que vencería con el 31-II
# de la LA ya es extemporánea por la literal, y se avisa de la otra lectura.
ok("resulta extemporánea." in txt and any("EN EL LÍMITE" in a and "31, fracción II" in a
                                           for a in avs) and
   not any("artículo 86" in a for a in avs),
   "RF el último día según la LA: extemporánea por la regla literal del 63, y se avisa que "
   "con el surtimiento de la LA estaría en tiempo (sin el aviso del art. 86, que no le toca)")
ok(rpt.considerando_adhesivo("queja", _mezclar(Q, {"adhesivo": AD_ADHESIVO}), DATOS)[:2] == ("", "")
   and rpt.considerando_adhesivo("amparo_directo", AD, DATOS) == ("", "", []),
   "queja (no hay adhesión) y asunto sin adhesivo: nada")


print("\n8 · EL PUNTO RESOLUTIVO DEL ADHESIVO")
fa = _mezclar(AD, {"adhesivo": AD_ADHESIVO})
ok(rpt.resolutivo_adhesivo("amparo_directo", False, fa) ==
   ("Se declara sin materia el amparo adhesivo promovido por Jorge Francisco Vargas Garduza.", ""),
   "AD, principal negado: sin materia")
t, a = rpt.resolutivo_adhesivo("amparo_directo", True, fa)
ok(t.startswith(H) and "SE CONCEDE" in a, "AD, principal concedido: hueco y aviso (hay que estudiarlo)")
fr_ = _mezclar(AR, {"adhesivo": AR_ADHESIVO})
ok(rpt.resolutivo_adhesivo("amparo_revision", "confirma", fr_)[0] ==
   "Se declara sin materia la revisión adhesiva interpuesta por el Director de Ingresos del "
   "Municipio de Querétaro.", "AR, se confirma: sin materia (y la autoridad con su artículo)")
t, a = rpt.resolutivo_adhesivo("amparo_revision", "fundado", fr_)
ok(t.startswith(H) and "FUNDADO" in a, "AR, principal fundado: hueco y aviso")
ff = _mezclar(RF, {"adhesivo": RF_ADHESIVO})
ok(rpt.resolutivo_adhesivo("revision_fiscal", "desecha", ff)[0] ==
   "Se desecha la revisión adhesiva interpuesta por María Hernández Ruiz.",
   "RF, principal improcedente: se desecha (2a./J. 145/2019)")
ok(rpt.resolutivo_adhesivo("revision_fiscal", False, ff)[0] ==
   "Se declara sin materia la revisión adhesiva interpuesta por María Hernández Ruiz.",
   "RF, principal infundado: sin materia")
ok(rpt.resolutivo_adhesivo("revision_fiscal", True, ff)[0].startswith(H),
   "RF, principal fundado: hueco")
ok(rpt.resolutivo_adhesivo("amparo_directo", None, fa)[0].startswith(H),
   "sin saber el principal: hueco")
ok(rpt.resolutivo_adhesivo("amparo_directo", False, AD) == ("", ""), "sin adhesivo: nada")
t, _a = rpt.resolutivo_adhesivo("revision_fiscal", "desecha",
                                _mezclar(RF, {"adhesivo": {"quien": "Grupo Lala, S.A.B. de C.V.",
                                                           "admision": "2025-11-24"}}))
ok(t == "Se desecha la revisión adhesiva interpuesta por Grupo Lala, S.A.B. de C.V.",
   "sin «..» cuando la razón social cierra con punto")


print("\n9 · LOS NOMBRES EN PROSA Y LAS CLAVES DE datos_extra")
ok(rpt._organo("SALA REGIONAL DEL CENTRO II") == "Sala Regional del Centro II",
   "el romano de la Sala no se vuelve «Ii»")
ok(rpt._organo("JEFATURA DE SERVICIOS JURÍDICOS DEL IMSS") ==
   "Jefatura de Servicios Jurídicos del IMSS", "las siglas no se vuelven «Imss»")
ok(rpt._organo("PRIMERA SALA CIVIL") == "Primera Sala Civil", "«Civil» no se toma por romano")
ok(rpt._organo("el Juzgado Segundo de Distrito") == "Juzgado Segundo de Distrito",
   "sin el artículo que teclea el secretario")
ok(rpt._parte("INMOBILIARIA ADMINISTRADORA DEL BAJÍO, S.A. DE C.V.") ==
   "Inmobiliaria Administradora del Bajío, S.A. de C.V.",
   "una sociedad no es autoridad aunque diga «Administradora»: sin artículo, y en la prosa "
   "sin versales (tercera ronda, E2)")
ok(rpt._parte("Director de Ingresos del Municipio de Querétaro") ==
   "el Director de Ingresos del Municipio de Querétaro", "la autoridad, con su artículo")
ok(rpt._parte("María Sala Pérez") == "María Sala Pérez" and
   rpt._parte("Juan Fiscal Ortega") == "Juan Fiscal Ortega",
   "una persona de apellido Sala o Fiscal no se toma por órgano (se mira la cabeza)")
ok(rpt._materia_clave("Materias Administrativa y Civil") == "" and
   rpt._materia_clave("MERCANTIL") == "mercantil" and rpt._materia_clave("agrario") == "agraria",
   "la especialidad del tribunal no es la materia del asunto; las demás se reconocen")
ok(rpt._numero_asunto("ADC 625-2024 ORAL MERCANTIL") == "625/2024" and
   rpt._numero_asunto("R.Q.C. 150/2026") == "150/2026", "el número sin la etiqueta")
ok(rpt._derechos("1º, 14 y 16") == ["1o.", "14", "16"], "los derechos desde una cadena")
ok(rpt._ponente_en_prosa("MAGISTRADO JUAN PÉREZ") == "del Magistrado Juan Pérez" and
   rpt._ponente_en_prosa("la Secretaria en funciones de Magistrada Bertha Martínez Vega")
   == "de Bertha Martínez Vega, secretaria en funciones de Magistrada",
   "el cargo que trae la fuente, con su artículo contraído (el «en funciones», en aposición: "
   "quinta ronda, sección 40)")
_claves = ("expediente", "toca", "fecha_acto", "fecha_acto_iso", "organo_acto", "inciso_97",
           "cola_97", "fraccion_97", "materia", "responsable", "juicio_amparo", "juzgado",
           "sala", "expediente_tfja", "actora", "descripcion_acto", "adhesivo", "supletorio",
           "fraccion_63", "cuantia", "fundamento_surtimiento")
for _t, _f in (("amparo_directo", AD), ("amparo_revision", AR), ("queja", Q),
               ("revision_fiscal", RF)):
    _ex = rpt.componer(_t, _f, DATOS)["datos_extra"]
    ok(all(k in _ex for k in _claves), f"{_t}: datos_extra trae todas las claves de la SPEC "
                                       f"(y las tres de la segunda ronda)")


print("\n10 · NI «S.A..» NI «C.V..» NI ESPACIOS DOBLES, EN NINGÚN CAMINO")
ok(rpt._pulir("Tiene ese carácter Banco del Centro, S.A..") ==
   "Tiene ese carácter Banco del Centro, S.A." and
   rpt._pulir("promovido por Metlife México, S.A. de C.V. .") ==
   "promovido por Metlife México, S.A. de C.V." and
   rpt._pulir("AUTORIDAD RESPONSABLE:\nBanco X, S.N.C. \nACTO RECLAMADO:\n  Su ejecución..") ==
   "AUTORIDAD RESPONSABLE:\nBanco X, S.N.C.\nACTO RECLAMADO:\nSu ejecución.",
   "_pulir: un solo punto tras la razón social, y renglones sin espacios en los bordes")
ok(rpt._pulir("dijo… y siguió... así") == "dijo… y siguió... así" and
   rpt._pulir("Juan  Pérez , interpuso ;") == "Juan Pérez, interpuso;",
   "_pulir: los puntos suspensivos se respetan; sin espacio doble ni antes de la puntuación")
ok(rpt._organo("Banco Nacional de Obras y Servicios Públicos, S.N.C.") ==
   "Banco Nacional de Obras y Servicios Públicos, S.N.C." and
   rpt._organo("Sala Regional del Centro I.") == "Sala Regional del Centro I",
   "_organo: conserva el punto de la abreviatura («S.N.C.»), quita el punto suelto")
ok(rpt._organo("AEROPUERTOS Y SERVICIOS AUXILIARES, S.A. DE C.V.") ==
   "Aeropuertos y Servicios Auxiliares, S.A. de C.V." and
   rpt._organo("BANCO NACIONAL DE OBRAS Y SERVICIOS PÚBLICOS, S.N.C.") ==
   "Banco Nacional de Obras y Servicios Públicos, S.N.C.",
   "_organo en versales: la abreviatura sale entera («S.A. de C.V.», no «S.a. de C.v»)")
_RX_DOBLE = re.compile(r"(?<!\.)\.[ \t]*\.(?!\.)")
_RX_ESPACIOS = re.compile(r"[ \t]{2,}|[ \t][,.;:]|[ \t]\n|\n[ \t]|^[ \t]|[ \t]$")


def _sucio(texto):
    return [m.group(0) for rx in (_RX_DOBLE, _RX_ESPACIOS) for m in rx.finditer(texto)]


for nombre, tipo, ficha, datos in CASOS:
    o = rpt.componer(tipo, ficha, datos)
    piezas = [o["visto"]] + [r["texto"] for r in o["resultandos"]]
    if ficha.get("adhesivo"):
        piezas.append(rpt.considerando_adhesivo(tipo, ficha, datos)[1])
        piezas += [rpt.resolutivo_adhesivo(tipo, p, ficha)[0] for p in (False, True, "desecha")]
    malos = [m for p in piezas for m in _sucio(p)]
    ok(not malos, f"{nombre}: sin dobles puntos ni espacios de más {malos[:3] if malos else ''}")
o = rpt.componer("amparo_directo", [c for c in CASOS if c[0].startswith("AD · razones")][0][2], DATOS)
ok("promovido por Banco del Centro, S.A., contra la sentencia" in o["visto"] and
   _res(o, "Tercero") == "Tienen ese carácter Metlife México, S.A. de C.V. y Unión de Crédito del "
   "Bajío, S.A. de C.V." and
   "Querétaro, Banco del Centro, S.A. promovió juicio de amparo directo" in _res(o, "Presentación"),
   "AD: la razón social con su punto, una sola vez, en el V I S T O, la presentación y el tercero")
_f_sa = [c for c in CASOS if c[0].startswith("AD · razones")][0][2]
ok(rpt.resolutivo_adhesivo("amparo_directo", False, _f_sa)[0] ==
   "Se declara sin materia el amparo adhesivo promovido por Metlife México, S.A. de C.V.",
   "el resolutivo del adhesivo con la razón social: un punto")
o = rpt.componer("amparo_revision", [c for c in CASOS if c[0].startswith("AR · autoridad con")][0][2], DATOS)
ok("\nAUTORIDADES RESPONSABLES:\nBanco Nacional de Obras y Servicios Públicos, S.N.C.\nDirector de "
   "Ingresos del Municipio de Querétaro\nACTOS RECLAMADOS:\nLa cancelación del crédito otorgado a "
   "Constructora Alfa, S.A. de C.V.\nSu ejecución." in _res(o, "Presentación"),
   "AR: el bloque AUTORIDAD RESPONSABLE conserva «S.N.C.» y los actos cierran con UN punto")


print("\n11 · LOS TRES DATOS DEL FORMULARIO VAN A datos_extra (segunda ronda)")
_FUND = ("el artículo 70 de la Ley Federal de Procedimiento Contencioso Administrativo")
o = rpt.componer("revision_fiscal", [c for c in CASOS if c[0].startswith("RF · con fracción")][0][2], DATOS)
ex = o["datos_extra"]
ok(ex["fraccion_63"] == "I" and ex["cuantia"] == "$847,738.77" and ex["fundamento_surtimiento"] == _FUND
   and not o["avisos"], "RF: fraccion_63, cuantia y fundamento_surtimiento, con el mismo nombre")
ok("847,738.77" not in _todo(o) and "artículo 70" not in _todo(o),
   "…y el compositor no los escribe en los resultandos (son de los considerandos)")
o = rpt.componer("amparo_directo", _mezclar(AD, {"fundamento_surtimiento": "el artículo 126 del Código de "
                                                 "Procedimientos Civiles del Estado de Querétaro"}), DATOS)
ok(o["datos_extra"]["fundamento_surtimiento"] == "el artículo 126 del Código de Procedimientos Civiles "
   "del Estado de Querétaro" and o["datos_extra"]["fraccion_63"] == "" and not o["avisos"],
   "AD: el fundamento del surtimiento pasa; la fracción del 63 va vacía")
o = rpt.componer("amparo_directo", _mezclar(AD, {"fraccion_63": "I", "cuantia": "$1.00"}), DATOS)
ok(o["datos_extra"]["fraccion_63"] == "" and o["datos_extra"]["cuantia"] == "" and
   any("(fraccion_63, cuantia)" in a for a in o["avisos"]),
   "la fracción del 63 y la cuantía en un amparo directo no aplican: vacías y avisadas")
o = rpt.componer("revision_fiscal", _mezclar(RF, {"fraccion_63": "XI"}), DATOS)
ok(o["datos_extra"]["fraccion_63"] == "" and any("(fraccion_63)" in a for a in o["avisos"]),
   "RF con una fracción que no es de la I a la X: no pasa, y se avisa")
o = rpt.componer("revision_fiscal", RF, DATOS)
ok(o["datos_extra"]["fraccion_63"] == "" and o["datos_extra"]["cuantia"] == "" and not o["avisos"],
   "RF sin ellos: vacíos y sin aviso del compositor (la procedencia avisa su hueco)")


# ═══════════════════════════════════════════════════════════════════════════
# TERCERA RONDA (3-oct-2026, FIXES_R3 · D). Cada comprobación falla con el
# compositor de la segunda ronda (scratchpad/procedencia/r3_D/).
# ═══════════════════════════════════════════════════════════════════════════
print("\n12 · E1 · LA ÚNICA INSTANCIA SALE DE LA FICHA (punta a punta, AD 274/2025)")
# La ficha del 9274/2025 tal como la guardó la sesión: toca familiar 4520/2024 y
# expediente 479/2024 ante la Segunda Sala Civil; el contexto venía marcado
# «unica» porque origen_acto leyó «la Jueza» en el relato del juicio de origen.
AD274 = {
    "formato": 1, "tipo": "amparo_directo", "numero": "9274/2025", "materia": "civil",
    "admision": {"fecha": "2025-04-01"},
    "turno": {"fecha": "2025-05-23", "ponente": "J. Guadalupe Tafoya Hernández"},
    "returno": {"fecha": "2025-11-25", "ponente": "Luis Armando Pérez Topete"},
    "ministerio_publico": "pedimento", "promovente": "GABRIEL REYES ALAMO",
    "caracter": "quejoso", "presentacion": "2025-03-13", "notificacion": "2025-02-24",
    "forma_notificacion": "personal",
    "acto": {"clase": "sentencia", "fecha": "2025-02-20", "instancia": "unica",
             "organo": "Segunda Sala Civil del Tribunal Superior de Justicia del Estado de "
                       "Querétaro", "toca": "4520/2024", "expediente": "479/2024"},
    "responsable": "SEGUNDA SALA CIVIL DEL TRIBUNAL SUPERIOR DE JUSTICIA DEL ESTADO DE QUERÉTARO",
    "terceros": ["MA. DEL REFUGIO TREJO RAMOS"], "derechos": ["14", "16"],
}
o = rpt.componer("amparo_directo", AD274, _mezclar(DATOS, {"instancia_origen": "unica"}))
ok(o["visto"] == ("para resolver el juicio de amparo directo civil 9274/2025, promovido por "
                  "Gabriel Reyes Alamo, contra la sentencia dictada el veinte de febrero de dos "
                  "mil veinticinco, por la Segunda Sala Civil del Tribunal Superior de Justicia "
                  "del Estado de Querétaro, en el toca 4520/2024, derivado del expediente "
                  "479/2024; y,"),
   "AD 274: el toca y el expediente aunque el contexto diga «unica», y el quejoso sin versales")
ok(o["datos_extra"].get("unica_instancia") is False and o["datos_extra"].get("toca") == "4520/2024" and
   any("ÚNICA INSTANCIA NO CUADRA" in a and "4520/2024" in a for a in o["avisos"]),
   "AD 274: datos_extra sin única instancia y el aviso nombra el toca que la desmiente")
ok(_res(o, "Tercero") == "Tiene ese carácter Ma. del Refugio Trejo Ramos." and
   ", Gabriel Reyes Alamo promovió juicio de amparo directo" in _res(o, "Presentación") and
   "GABRIEL" not in _todo(o) and "TREJO" not in _todo(o),
   "AD 274 (E2): quejoso y tercero en mayúsculas y minúsculas en toda la prosa («Ma. del»)")
o = rpt.componer("amparo_directo", _mezclar(AD, {"acto": {"toca": ""}}),
                 _mezclar(DATOS, {"instancia_origen": "unica"}))
ok(o["datos_extra"].get("unica_instancia") is False and
   any("ÚNICA INSTANCIA NO CUADRA" in a and "órgano de apelación" in a for a in o["avisos"]) and
   any("(acto.toca)" in a for a in o["avisos"]),
   "sin toca pero con una Sala de apelación: tampoco es única instancia, y se pide el toca")
o = rpt.componer("amparo_directo", _mezclar(AD, {"acto": {"toca": "", "instancia": "alzada"},
                                                 "responsable": "Juzgado Segundo Civil de "
                                                                "Querétaro"}),
                 _mezclar(DATOS, {"instancia_origen": "unica"}))
ok(o["datos_extra"].get("unica_instancia") is False and
   any("ÚNICA INSTANCIA NO CUADRA" in a and "alzada" in a for a in o["avisos"]),
   "la ficha dice alzada (acto.instancia): la marca del contexto no manda")
o = rpt.componer("amparo_directo", CASOS[6][2], DATOS)
ok(o["datos_extra"].get("unica_instancia") is True and not o["avisos"],
   "TFJA sin toca y marcado: sigue siendo única instancia, sin aviso")

print("\n13 · E2 · LOS NOMBRES EN VERSALES PASAN A LA PROSA SIN VERSALES")
for crudo, prosa in (
        ("GABRIEL REYES ALAMO", "Gabriel Reyes Alamo"),
        ("MA. DEL REFUGIO TREJO RAMOS", "Ma. del Refugio Trejo Ramos"),
        ("IMPULSORA DE DESARROLLOS INMOBILIARIOS VV, S.A. DE C.V.",
         "Impulsora de Desarrollos Inmobiliarios VV, S.A. de C.V."),
        ("UNIÓN DE TRABAJADORES DE LA CONSTRUCCIÓN, C.T.M.",
         "Unión de Trabajadores de la Construcción, C.T.M."),
        ("CONSTRUCTORA SIGLO XXI, S. DE R.L. DE C.V.", "Constructora Siglo XXI, S. de R.L. de C.V."),
        ("GRUPO LA MODERNA, S.A.B. DE C.V.", "Grupo La Moderna, S.A.B. de C.V."),
        ("HSBC MÉXICO, S.A.", "HSBC México, S.A."),
        ("J. GUADALUPE GARCÍA-LÓPEZ", "J. Guadalupe García-López"),
        ("JUAN V V", "Juan V V"),
        ("María de la Luz McAllister", "María de la Luz McAllister"),
        ("MARÍA AMPARO LEY PÉREZ", "María Amparo Ley Pérez"),
        ("SUCESIÓN TESTAMENTARIA A BIENES DE PEDRO RUIZ, A TRAVÉS DE SU ALBACEA LAURA RUIZ",
         "la Sucesión Testamentaria a Bienes de Pedro Ruiz, a través de su albacea Laura Ruiz"),
        ("JUAN PÉREZ, POR PROPIO DERECHO Y EN REPRESENTACIÓN DE SU HIJO MENOR DE EDAD",
         "Juan Pérez, por propio derecho y en representación de su hijo menor de edad")):
    ok(rpt._parte(crudo) == prosa, f"«{crudo}» → «{prosa}»")
o = rpt.componer("amparo_revision", _mezclar(AR, {
    "promovente": "IMPULSORA DE DESARROLLOS INMOBILIARIOS VV, S.A. DE C.V."}), DATOS)
ok("interpuesto por Impulsora de Desarrollos Inmobiliarios VV, S.A. de C.V., contra" in o["visto"]
   and "S.A. DE C.V." not in _todo(o), "AR: la sociedad recurrente sin versales en la prosa")

print("\n14 · LA FICHA SE VALIDA AL COMPONER (rev_3: AR 307/2024, 448 y 60/2025)")
_f_imp = _mezclar(AR, {"audiencia": "2025-08-15"})
_copia = copy.deepcopy(_f_imp)
o = rpt.componer("amparo_revision", _f_imp, DATOS)
ok(any(a.startswith("FECHA IMPOSIBLE") and "audiencia" in a for a in o["avisos"]),
   "audiencia posterior a la sentencia recurrida: el aviso de ficha_tramite.validar sale aquí")
ok(_f_imp == _copia, "y la ficha que recibe no se toca (se valida una copia)")

print("\n15 · EL NÚMERO, EL INCISO, EL ARTÍCULO Y LA MAYÚSCULA DEL SUJETO")
o = rpt.componer("amparo_directo", _mezclar(AD, {
    "promovente": "Ana López Ruiz, por conducto de su apoderado legal Pedro Gómez Ruiz, y "
                  "Juan Pérez Gómez, por propio derecho",
    "representante": None, "figura_representante": None}), DATOS)
ok("Juan Pérez Gómez, por propio derecho, promovieron juicio de amparo directo" in
   _res(o, "Presentación"),
   "AD 552/2024: dos quejosos en un campo → «promovieron», y la coma que cierra el inciso")
o = rpt.componer("amparo_revision", _mezclar(AR, {
    "promovente": "Ana Pérez Ruiz y Luis Pérez Ruiz, ambos de apellidos Pérez Ruiz",
    "representante": None, "figura_representante": None}), DATOS)
ok("Ana Pérez Ruiz y Luis Pérez Ruiz, ambos de apellidos Pérez Ruiz, promovieron juicio de "
   "amparo indirecto" in _res(o, "Presentación") and
   "Pérez Ruiz, en su carácter de parte quejosa, interpusieron recurso de revisión" in
   _res(o, "Interposición"), "AR 208/2025: «ambos de apellidos…,» y los verbos en plural")
o = rpt.componer("queja", _mezclar(Q, {"promovente": "Ana López Ruiz y Juan Pérez Gómez"}), DATOS)
ok("interpusieron recurso de queja" in _res(o, "Interposición"), "queja: «interpusieron»")
o = rpt.componer("revision_fiscal", _mezclar(RF, {
    "actora": "Ana López Ruiz, Juan Pérez Gómez y Luis Díaz Mora"}), DATOS)
ok(_res(o, "Trámite del juicio").startswith("Ana López Ruiz, Juan Pérez Gómez y Luis Díaz Mora "
                                            "demandaron la nulidad"),
   "RF 7/2025: varias actoras → «demandaron»")
o = rpt.componer("revision_fiscal", _mezclar(RF, {
    "actora": "Ana López Ruiz, Juan Pérez Gómez, y Luis Díaz Mora"}), DATOS)
ok("y Luis Díaz Mora demandaron la nulidad" in _res(o, "Trámite del juicio"),
   "RF 6/2025: la lista que cierra «, y Nombre» no lleva coma ante el verbo (no es un inciso)")
_CTM = ("UNIÓN DE TRABAJADORES DE LA CONSTRUCCIÓN, TRANSPORTISTAS, MATERIALISTAS Y SIMILARES Y "
        "ANEXOS DEL ESTADO DE QUERÉTARO, C.T.M.")
o = rpt.componer("amparo_revision", _mezclar(AR, {"promovente": _CTM, "representante": None,
                                                 "figura_representante": None}), DATOS)
ok("Querétaro, C.T.M. promovió juicio de amparo indirecto" in _res(o, "Presentación") and
   "en su carácter de parte quejosa, interpuso recurso" in _res(o, "Interposición") and
   "Unión de Trabajadores de la Construcción, Transportistas, Materialistas y Similares y Anexos "
   "del Estado de Querétaro, C.T.M." in o["visto"],
   "AR 631/2025 (punta a punta): una organización que enumera en su nombre sigue siendo UNA")
o = rpt.componer("amparo_directo", _mezclar(AD, {"promovente": "Juan Ortega y Gasset"}), DATOS)
ok("Juan Ortega y Gasset, por conducto de su apoderado legal Pedro Gómez Ruiz, promovió" in
   _res(o, "Presentación"), "un apellido compuesto con «y» no es plural")
o = rpt.componer("amparo_directo", _mezclar(AD, {
    "promovente": "SUCESIÓN A BIENES DE JUAN PÉREZ LÓPEZ", "representante": None,
    "figura_representante": None}), DATOS)
ok("promovido por la Sucesión a Bienes de Juan Pérez López, contra" in o["visto"],
   "AD 335/2025: «la Sucesión a Bienes de…», con su artículo y sin versales")
o = rpt.componer("amparo_revision", _mezclar(AR, {
    "promovente": "EJIDO SAN JUAN DEL RÍO", "representante": None, "figura_representante": None,
    "demanda": {"fecha": ""}}), DATOS)
ok(_res(o, "Presentación").startswith("El Ejido San Juan del Río promovió juicio de amparo "
                                      "indirecto") and "interpuesto por el Ejido" in o["visto"],
   "AR 60/2025: el ejido con su artículo, y en mayúscula cuando abre la oración")

print("\n16 · LAS ETIQUETAS DE ROL Y EL REPRESENTANTE QUE ES LA MISMA PARTE")
o = rpt.componer("amparo_revision", _mezclar(AR, {
    "promovente": "GOBERNADOR DEL ESTADO DE QUERÉTARO (AUTORIDAD RESPONSABLE)", "caracter": "",
    "representante": None, "figura_representante": None, "acto": {"resolvio": "concede"}}),
    _mezclar(DATOS, {"quejoso": "Impulsora de Desarrollos Inmobiliarios VV, S.A. de C.V."}))
ok("interpuesto por el Gobernador del Estado de Querétaro, contra" in o["visto"] and
   "Inconforme, el Gobernador del Estado de Querétaro, en su carácter de autoridad responsable, "
   "interpuso" in _res(o, "Interposición") and "(autoridad" not in _todo(o).lower(),
   "AR 208/2025: sin «(AUTORIDAD RESPONSABLE)» en la prosa, y el carácter sale de la etiqueta")
o = rpt.componer("queja", _mezclar(Q, {"promovente": "Ana Pérez Gómez",
                                      "representante": "ANA PÉREZ GÓMEZ",
                                      "figura_representante": "representante legal"}), DATOS)
ok("por conducto de" not in _res(o, "Interposición") and
   any("(representante = " in a for a in o["avisos"]),
   "Q 337/2025: el representante que es la misma persona no se escribe, y se avisa")
o = rpt.componer("queja", _mezclar(Q, {
    "promovente": "SUCESIÓN A BIENES DE PEDRO RUIZ, A TRAVÉS DE SU ALBACEA LAURA RUIZ SOTO",
    "representante": "Laura Ruiz Soto", "figura_representante": "albacea"}), DATOS)
ok(_res(o, "Interposición").count("Laura Ruiz Soto") == 1,
   "Q 342/2025: el albacea que ya va en el nombre de la sucesión no se repite")
o = rpt.componer("amparo_revision", _mezclar(AR, {
    "figura_representante": "autorizado en términos amplios del artículo 12 de la Ley de Amparo",
    "representante": "LUIS GÓMEZ DÍAZ"}), DATOS)
ok("por conducto de su autorizado en términos amplios del artículo 12 de la Ley de Amparo, "
   "Luis Gómez Díaz, interpuso" in _res(o, "Interposición"),
   "AR 239/2025: la figura larga lleva coma antes del nombre, y el nombre sin versales")

print("\n17 · AR · EL JUZGADO NO PUEDE SER UNA AUTORIDAD RESPONSABLE (AR 307, 448, 60 y 222)")
_SALA = "Sala Familiar del Tribunal Superior de Justicia del Estado de Querétaro"
o = rpt.componer("amparo_revision", _mezclar(AR, {"acto": {"organo": _SALA}}), DATOS)
ok(H in o["visto"] and "Sala Familiar" not in o["visto"] and
   any(a.startswith("FALTA EL JUZGADO") and "(acto.organo)" in a and "autoridad responsable" in a
       for a in o["avisos"]),
   "AR 307/2024: el órgano de la recurrida es la responsable → hueco y un aviso que dice por qué")
o = rpt.componer("amparo_revision", _mezclar(AR, {"acto": {"organo": _SALA}}),
                 _mezclar(DATOS, {"organo_recurrido": "Juzgado Primero de Distrito en el "
                                                      "Estado de Querétaro"}))
ok("por el Juzgado Primero de Distrito en el Estado de Querétaro, en el juicio" in o["visto"]
   and any("SE DESCARTÓ" in a and "(acto.organo" in a for a in o["avisos"]),
   "…y si otra fuente trae el juzgado, se usa ésa, con aviso")
_JUZ = AR["acto"]["organo"]
o = rpt.componer("amparo_revision", _mezclar(AR, {"demanda": {"autoridades": [
    _JUZ, "Juez Sexto de Primera Instancia Civil del Distrito Judicial de Querétaro"]}}), DATOS)
ok("AUTORIDAD RESPONSABLE:\nJuez Sexto de Primera Instancia Civil" in _res(o, "Presentación")
   and "Cuarto de Distrito" not in _res(o, "Presentación") and
   o["datos_extra"].get("autoridades") == ["Juez Sexto de Primera Instancia Civil del Distrito "
                                       "Judicial de Querétaro"] and
   any("ESTABA ENTRE LAS AUTORIDADES" in a for a in o["avisos"]),
   "AR 222/2025: el juzgado que resolvió sale de la lista de autoridades, con aviso")
o = rpt.componer("amparo_revision", _mezclar(AR, {"acto": {
    "organo": "Juez Octavo de Primera Instancia Familiar del Distrito Judicial de Querétaro"}}),
    DATOS)
ok("por el Juzgado Octavo de Primera Instancia Familiar" in o["visto"] and
   any("NO PARECE UN JUZGADO DE DISTRITO" in a for a in o["avisos"]),
   "AR 448/2025: un juez de primera instancia se escribe (puede haber jurisdicción "
   "concurrente), pero se avisa")
o = rpt.componer("amparo_revision", _mezclar(AR, {"acto": {"fecha": "2025-03-03"},
                                                 "audiencia": "", "demanda": {"fecha": ""}}), DATOS)
ok(any("APARECE EN EL ACTO RECLAMADO" in a for a in o["avisos"]),
   "la fecha de la recurrida escrita dentro del acto reclamado: aviso")
o = rpt.componer("amparo_revision", _mezclar(AR, {"acto": {
    "organo": "la titular del Juzgado Primero de Distrito en el Estado de Querétaro"}}), DATOS)
ok("por el Juzgado Primero de Distrito en el Estado de Querétaro, en el juicio" in
   o["visto"] and "Titular" not in o["visto"],
   "«la titular del Juzgado» pasa al órgano, como en la queja (cuarta ronda, E10)")

print("\n18 · LA VÍA ELECTRÓNICA EN LA INTERPOSICIÓN (AR 201 y 60/2025)")
o = rpt.componer("amparo_revision", _mezclar(AR, {"via_presentacion": "electronica"}), DATOS)
ok("por escrito presentado el veintinueve de agosto de dos mil veinticinco a través del Portal de "
   "Servicios en Línea del Poder Judicial de la Federación." in _res(o, "Interposición"),
   "AR: el Portal de Servicios en Línea del Poder Judicial de la Federación")
o = rpt.componer("queja", _mezclar(Q, {"via_presentacion": "electronica"}), DATOS)
ok(_res(o, "Interposición").startswith(
    "Por escrito presentado el diez de marzo de dos mil veintiséis a través del Portal de "
    "Servicios en Línea del Poder Judicial de la Federación, José Eduardo"), "queja: también")
o = rpt.componer("amparo_revision", _mezclar(AR, {"via_presentacion": "juzgado"}), DATOS)
ok("presentado el veintinueve de agosto de dos mil veinticinco ante el Juzgado Cuarto de "
   "Distrito" in _res(o, "Interposición"), "AR por conducto del juzgado (art. 88 LA): ante él")

print("\n19 · QUEJA · EL ÓRGANO, EN FORMA DE ÓRGANO Y EL MÁS COMPLETO (Q 261 y 300/2025)")
_INCOMPLETO = "Juez Sexto de Distrito de Amparo y Juicios Federales en el Estado de Querétaro"
_OFICIAL = ("Juzgado Sexto de Distrito en Materia de Amparo Civil, Administrativo y de Trabajo y "
            "de Juicios Federales en el Estado de Querétaro")
o = rpt.componer("queja", _mezclar(Q, {"acto": {"organo": _INCOMPLETO}, "responsable": _OFICIAL}),
                 DATOS)
ok(f"dictado por el {_OFICIAL}, en el juicio" in o["visto"] and
   o["datos_extra"].get("juzgado") == _OFICIAL == o["datos_extra"].get("responsable") and
   any("DOS FORMAS" in a and "Juzgado Sexto de Distrito de Amparo y Juicios" in a
       for a in o["avisos"]),
   "Q 261: el nombre oficial y completo, no el incompleto del acto; con aviso de los dos")
o = rpt.componer("queja", _mezclar(Q, {"acto": {
    "organo": "la titular del Juzgado Cuarto de Distrito en el Estado de Querétaro"}}), DATOS)
ok("dictado por el Juzgado Cuarto de Distrito en el Estado de Querétaro, en el juicio" in
   o["visto"] and "Titular" not in _todo(o) and not o["avisos"],
   "Q 300: «la titular del Juzgado Cuarto» → «el Juzgado Cuarto» (el órgano, no la persona)")
o = rpt.componer("queja", _mezclar(Q, {"responsable": "Juzgado Séptimo de Distrito en el Estado "
                                                      "de Querétaro"}), DATOS)
ok(not o["avisos"], "«Juez Séptimo…» y «Juzgado Séptimo…» son el mismo nombre: sin aviso")

print("\n20 · RF · QUIÉN RECURRE: LA UNIDAD Y A QUIÉN REPRESENTA (RF 2, 7, 21 y 6/2025, 6/2026)")
_SUBD = ("Subdelegado de Prestaciones Económicas de la Oficina de Representación del ISSSTE en "
         "Querétaro")
o = rpt.componer("revision_fiscal", _mezclar(RF, {
    "promovente": "JEFA DE LA UNIDAD JURÍDICA DE LA OFICINA DE REPRESENTACIÓN DEL ISSSTE EN "
                  "QUERÉTARO, EN REPRESENTACIÓN DEL SUBDELEGADO DE PRESTACIONES ECONÓMICAS DE "
                  "LA OFICINA DE REPRESENTACIÓN DEL ISSSTE EN QUERÉTARO",
    "autoridad_demandada": _SUBD}), DATOS)
t2 = _res(o, "Interposición")
ok(t2.startswith("Inconforme, la Jefa de la Unidad Jurídica de la Oficina de Representación del "
                 "ISSSTE en Querétaro, en representación del Subdelegado de Prestaciones") and
   t2.lower().count("en representación") == 1,
   "RF 2/2025: «la Jefa» (no «el Jefa») y UNA sola «en representación de»")
ok(o["datos_extra"].get("recurrente_unidad") == "Jefa de la Unidad Jurídica de la Oficina de "
   "Representación del ISSSTE en Querétaro" and o["datos_extra"].get("autoridad_demandada") == _SUBD,
   "datos_extra: recurrente_unidad y autoridad_demandada normalizadas, sin artículo")
_JEFE = ("Jefe del Departamento de Pensiones, Seguridad e Higiene de la Oficina de Representación "
         "del ISSSTE en Querétaro")
o = rpt.componer("revision_fiscal", _mezclar(RF, {
    "promovente": _JEFE.upper(), "autoridad_demandada": _JEFE, "representante": "",
    "figura_representante": "Titular de la Unidad de Asuntos Jurídicos"}), DATOS)
ok(_res(o, "Interposición").startswith(
    f"Inconforme, el {_JEFE}, por conducto del Titular de la Unidad de Asuntos Jurídicos, "
    "interpuso recurso de revisión fiscal") and o["datos_extra"].get("recurrente_unidad") ==
   "Titular de la Unidad de Asuntos Jurídicos",
   "RF 6/2026: recurre la propia demandada por conducto de su unidad, con la figura aunque falte "
   "el nombre")
o = rpt.componer("revision_fiscal", _mezclar(RF, {
    "promovente": _JEFE, "autoridad_demandada": _JEFE, "representante": None,
    "figura_representante": None}), DATOS)
ok(f"Inconforme, el {_JEFE}, por conducto de {H}, interpuso" in _res(o, "Interposición") and
   any("(figura_representante)" in a for a in o["avisos"]) and
   "en representación del Jefe" not in _res(o, "Interposición"),
   "RF 21/2025: la misma autoridad sin la figura → hueco y aviso, nunca «X, en representación de X»")
_SUBD_E = ("Subdelegado de Prestaciones Económicas, Delegación Estatal Querétaro, del Instituto de "
           "Seguridad y Servicios Sociales de los Trabajadores del Estado")
_SUBD_C = ("Subdelegado de Prestaciones, Delegación Estatal Querétaro, del Instituto de Seguridad y "
           "Servicios Sociales de los Trabajadores del Estado")
o = rpt.componer("revision_fiscal", _mezclar(RF, {
    "promovente": _SUBD_E, "autoridad_demandada": _SUBD_C, "representante": "",
    "figura_representante": "Titular de la Unidad de Asuntos Jurídicos"}), DATOS)
ok(_res(o, "Interposición").startswith(f"Inconforme, el {_SUBD_C}, por conducto del Titular de "
                                      "la Unidad de Asuntos Jurídicos, interpuso") and
   "en representación" not in _res(o, "Interposición"),
   "RF 6/2026 (ficha real): la misma autoridad escrita con una palabra de más sigue siendo ella")
o = rpt.componer("revision_fiscal", _mezclar(RF, {
    "promovente": _JEFE.upper(), "autoridad_demandada": _JEFE, "representante": "",
    "figura_representante": "representante legal"}), DATOS)
ok(f"Inconforme, el {_JEFE}, por conducto de su representante legal, interpuso" in
   _res(o, "Interposición"),
   "RF 21/2025 (ficha real): una figura genérica va con «su», no como órgano («del Representante»)")
o = rpt.componer("revision_fiscal", _mezclar(RF, {
    "actora": "ANA LÓPEZ RUIZ, JUAN PÉREZ GÓMEZ y LUIS DÍAZ MORA"}), DATOS)
ok(_res(o, "Trámite del juicio").startswith("Ana López Ruiz, Juan Pérez Gómez y Luis Díaz Mora "
                                            "demandaron la nulidad"),
   "RF 6/2026: actoras en versales unidas por una «y» en minúscula: sin versales y en plural")
o = rpt.componer("revision_fiscal", _mezclar(RF, {
    "promovente": f"{_SUBD}, por conducto de la Titular de la Unidad Jurídica",
    "autoridad_demandada": ""}), DATOS)
ok(_res(o, "Interposición").startswith(
    f"Inconforme, el {_SUBD}, por conducto de la Titular de la Unidad Jurídica, interpuso"),
   "«X, por conducto de la Titular…»: X es la representada, y «la titular» del papel se respeta")

print("\n21 · RF · NEGATIVA FICTA, SENTIDO DEL CATÁLOGO, CORREO Y PREFIJO TEMPORAL")
o = rpt.componer("revision_fiscal", _mezclar(RF, {"resolucion_impugnada": {
    "clase": "negativa_ficta", "oficio": "", "fecha": "", "materia": "pago de la pensión por "
                                                                   "jubilación"}}), DATOS)
ok("demandó la nulidad de la resolución negativa ficta recaída a su solicitud de pago de la "
   "pensión por jubilación, atribuida al Titular de la Subdelegación" in
   _res(o, "Trámite del juicio") and not any("resolucion_impugnada.oficio" in a or
                                             "resolucion_impugnada.fecha" in a for a in o["avisos"]),
   "RF 21/2025: la negativa ficta, sin pedir oficio ni fecha que no existen")
o = rpt.componer("revision_fiscal", _mezclar(RF, {"resolucion_impugnada": {
    "oficio": "", "fecha": "", "autoridad": ""}}), DATOS)
ok(f"demandó la nulidad de la resolución {H}." in _res(o, "Trámite del juicio") and
   sum("resolucion_impugnada" in a for a in o["avisos"]) == 1,
   "expresa sin ningún dato: un hueco y UN aviso (no tres y una frase coja)")
for sentido, frase in (
        ("declaró la nulidad para efectos", "declaró la nulidad de la resolución impugnada, para "
                                            "determinados efectos"),
        ("Declaró la NULIDAD LISA Y LLANA", "declaró la nulidad lisa y llana de la resolución "
                                            "impugnada"),
        ("se declaró la nulidad de la resolución", "declaró la nulidad de la resolución impugnada"),
        ("sobreseyó respecto de un acto y reconoció la validez del resto",
         "sobreseyó parcialmente en el juicio y reconoció la validez de la resolución impugnada")):
    o = rpt.componer("revision_fiscal", _mezclar(RF, {"acto": {"sentido": sentido}}), DATOS)
    ok(_res(o, "Trámite del juicio").endswith(f"en la que {frase}.") and not o["avisos"],
       f"sentido «{sentido}» → la frase del catálogo")
o = rpt.componer("revision_fiscal", _mezclar(RF, {"acto": {"sentido": "resolvió lo conducente"}}),
                 DATOS)
ok(any("NO ESTÁ EN EL CATÁLOGO" in a for a in o["avisos"]), "un sentido fuera del catálogo: aviso")
try:
    import ficha_tramite as _ft
    _cat = {k: v[1] for k, v in _ft.CATALOGO_SENTIDO["revision_fiscal"].items()}
    # QUINTA RONDA: el catálogo puede crecer («nulidad_derecho», C); las
    # frases que usa el compositor son las del catálogo para TODAS sus claves,
    # y las cuatro de siempre siguen iguales en los dos lados.
    _fr = getattr(rpt, "_frases_sentido_rf", lambda: rpt._SENTIDO_RF)()
    ok(all(_fr.get(k) == v for k, v in _cat.items()) and
       all(_cat.get(k) == v for k, v in rpt._SENTIDO_RF.items()) and
       _ft._SOBRESEE_PARCIAL == rpt._SOBRESEE_PARCIAL,
       "las frases del compositor son las del catálogo de ficha_tramite")
except Exception as _ex:
    ok(False, f"no se pudo leer ficha_tramite.CATALOGO_SENTIDO ({_ex})")
o = rpt.componer("revision_fiscal", _mezclar(RF, {"via_presentacion": "postal",
                                                 "deposito_postal": "2025-10-08",
                                                 "presentacion": "2025-10-08"}), DATOS)
ok(_res(o, "Interposición").endswith("mediante oficio depositado en el Servicio Postal Mexicano el "
                                     "ocho de octubre de dos mil veinticinco.") and
   any("COINCIDE CON EL DEPÓSITO" in a for a in o["avisos"]),
   "RF 26/2025: depósito y recepción el mismo día → sólo el depósito, y se avisa")
o = rpt.componer("revision_fiscal", _mezclar(RF, {"sala": "ACTUAL SALA REGIONAL EN QUERÉTARO"}),
                 DATOS)
ok("dictada por la actual Sala Regional en Querétaro del Tribunal Federal de Justicia "
   "Administrativa, en el juicio" in o["visto"] and "Actual" not in _todo(o),
   "RF 49/2025: «la actual Sala…», no «el Actual Sala»")

print("\n22 · RF · LA ADHESIVA CON LA REGLA LITERAL DEL 63 Y EL ADHESIVO CON SU CLÁUSULA DE INHÁBILES")
f = _mezclar(RF, {"adhesivo": RF_ADHESIVO})
rot, txt, avs = rpt.considerando_adhesivo("revision_fiscal", f, DATOS)
ok("contado a partir de la fecha en la que se le notificó la admisión del recurso" in txt and
   "surtió efectos" not in txt and "artículo 31" not in txt,
   "RF 4/2025: la regla literal del penúltimo párrafo, sin atribuirle el surtimiento")
rot, txt, avs = rpt.considerando_adhesivo(
    "revision_fiscal", _mezclar(f, {"adhesivo": {"notificacion": ""}}), DATOS)
ok("contado a partir de la fecha en la que se le notificara la admisión del recurso" in txt and
   "surtiera efectos" not in txt, "RF sin fechas: la misma regla literal, con el veredicto en hueco")
_c = f0.computar(_dt.date(2025, 7, 10), _dt.date(2025, 8, 6), regla="lista", plazo=15,
                 tipo_asunto="amparo_revision")
rot, txt, avs = rpt.considerando_adhesivo("amparo_directo", _mezclar(AD, {"adhesivo": {
    "quien": "Jorge Francisco Vargas Garduza", "notificacion": "2025-07-10",
    "presentacion": "2025-08-06", "admision": "2025-08-08"}}), DATOS)
ok(f0.clausula_inhabiles(_c) in txt and "periodo vacacional" in txt.lower(),
   "AD con vacaciones: la cláusula de inhábiles es la de fase0_oportunidad, cada grupo con su "
   "fundamento")

print("\n23 · LO QUE NO ES TEXTO Y LOS CARACTERES DE CONTROL (rev_6)")
o = rpt.componer("amparo_directo", _mezclar(AD, {"turno": {"ponente": {"nombre": "Ana"}},
                                                 **RETURNO}), DATOS)
ok("{" not in _todo(o) and H in _res(o, "Turno") and
   any("(turno.ponente)" in a for a in o["avisos"]),
   "un dict en lugar del ponente es un dato que falta: hueco, nunca «{'nombre': 'Ana'}»")
o = rpt.componer("amparo_directo", _mezclar(AD, {"promovente": "Ruth\x00 Gabriela\x07 Larrañaga"}),
                 DATOS)
ok(not re.search(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", _todo(o)) and "Ruth Gabriela Larrañaga" in o["visto"],
   "los caracteres de control no llegan al .docx")

o = rpt.componer("amparo_directo", _mezclar(AD, {"terceros": 12.5, "derechos": 7}), DATOS)
ok(o["resultandos"] and not any("FALLÓ" in a for a in o["avisos"]) and
   any("(derechos)" in a for a in o["avisos"]),
   "terceros o derechos que no son lista: dato que falta, no un compositor caído (fuzz)")
try:
    _r9 = rpt.considerando_adhesivo("revision_fiscal", _mezclar(RF, {"adhesivo": {
        "quien": "María Hernández Ruiz", "notificacion": "9999-12-30",
        "presentacion": "9999-12-31", "admision": "2025-11-24"}}), DATOS)
    ok(H in _r9[1] and any("adhesivo.notificacion" in a or "FECHA" in a for a in _r9[2]) or
       any("(adhesivo.notificacion)" in a for a in _r9[2]),
       "fechas del año 9999 en el adhesivo: hueco con aviso, no OverflowError (rev_6)")
except Exception as _ex:
    ok(False, f"fechas del año 9999 en el adhesivo: lanzó {type(_ex).__name__}")

print("\n24 · LOS DATOS NUEVOS DE datos_extra")
for _cl, _ar in (("sentencia", "la sentencia reclamada"), ("resolucion", "la resolución reclamada"),
                 ("laudo", "el laudo reclamado")):
    _ex = rpt.componer("amparo_directo", _mezclar(AD, {"acto": {"clase": _cl}}), DATOS)["datos_extra"]
    ok(_ex.get("clase") == _cl and _ex.get("acto_reclamado") == _ar, f"AD {_cl}: clase y «{_ar}»")
ok(rpt.componer("amparo_revision", CASOS[14][2], DATOS)["datos_extra"].get("clase") == "interlocutoria"
   and rpt.componer("queja", Q, DATOS)["datos_extra"].get("clase") == "auto",
   "AR del incidente: «interlocutoria»; queja: «auto»")
ok(rpt.componer("amparo_directo", _mezclar(AD, RETURNO), DATOS)["datos_extra"].get("ponente") ==
   "Magistrada Ana Laura Campos Juárez" and
   rpt.componer("amparo_directo", AD, DATOS)["datos_extra"].get("ponente") ==
   "Luis Armando Pérez Topete" and
   rpt.componer("amparo_directo", _mezclar(AD, {"returno": {"fecha": "2026-05-04"}}),
                DATOS)["datos_extra"].get("ponente") == "",
   "ponente: el del returno; sin returno, el del turno; returno sin ponente, «» (no se adivina)")
ok(rpt.componer("revision_fiscal", _mezclar(RF, {"adhesivo": RF_ADHESIVO}),
                DATOS)["datos_extra"].get("adherente") == "María Hernández Ruiz" and
   rpt.componer("revision_fiscal", RF, DATOS)["datos_extra"].get("adherente") == "",
   "adherente: quien se adhirió, o «»")
for _t, _f in (("amparo_directo", AD), ("amparo_revision", AR), ("queja", Q),
               ("revision_fiscal", RF)):
    _ex = rpt.componer(_t, _f, DATOS)["datos_extra"]
    ok(all(k in _ex for k in ("clase", "acto_reclamado", "ponente", "adherente",
                               "recurrente_unidad", "autoridad_demandada",
                               "promovente_en_prosa", "quejoso_en_prosa")),
       f"{_t}: datos_extra trae las claves de la tercera ronda")
_ex = rpt.componer("amparo_directo", AD274, DATOS)["datos_extra"]
ok(_ex.get("promovente_en_prosa") == "Gabriel Reyes Alamo" and
   getattr(rpt, "nombre_en_prosa", lambda *a, **k: "")("GOBERNADOR DEL ESTADO (AUTORIDAD "
                                                       "RESPONSABLE)", autoridad=True) ==
   "el Gobernador del Estado",
   "el nombre en prosa, para que la legitimación diga lo mismo que el V I S T O (AD 274)")

print("\n25 · AD · «ANTE ESTE TRIBUNAL COLEGIADO» SÓLO SI LO DECLARÓ EL SECRETARIO (rev_1)")
o = rpt.componer("amparo_directo", _mezclar(AD, {"via_presentacion": "tribunal"}), DATOS)
ok("ante este Tribunal Colegiado" not in _res(o, "Presentación") and
   _res(o, "Presentación").startswith("Por escrito presentado el doce de febrero de dos mil "
                                      "veintiséis, Ruth") and
   any("(via_presentacion)" in a for a in o["avisos"]),
   "AD 128/274/335/456: «tribunal» leído del papel no se afirma; se avisa")
o = rpt.componer("amparo_directo", _mezclar(AD, {"via_presentacion": "tribunal_colegiado",
                                                 "fuentes": {"via_presentacion": "secretario"}}),
                 DATOS)
ok("ante este Tribunal Colegiado" in _res(o, "Presentación"),
   "declarado por el secretario: «ante este Tribunal Colegiado»")

print("\n26 · LOS CARGOS FEMENINOS Y EL PREFIJO TEMPORAL CON SU ARTÍCULO")
for crudo, prosa in (("Directora de Ingresos del Municipio de Querétaro",
                      "la Directora de Ingresos del Municipio de Querétaro"),
                     ("COORDINADORA DE RECURSOS HUMANOS DEL MUNICIPIO DE CADEREYTA",
                      "la Coordinadora de Recursos Humanos del Municipio de Cadereyta"),
                     ("Subdelegada de Prestaciones del ISSSTE", "la Subdelegada de Prestaciones del ISSSTE"),
                     ("Legislatura del Estado de Querétaro", "la Legislatura del Estado de Querétaro"),
                     ("Cámara de Diputados", "la Cámara de Diputados"),
                     ("Jefatura de Gobierno", "la Jefatura de Gobierno"),
                     ("entonces Juez Segundo de Distrito", "el entonces Juez Segundo de Distrito"),
                     ("Titular de la Unidad Jurídica", "el Titular de la Unidad Jurídica"),
                     ("la titular de la Unidad Jurídica", "la Titular de la Unidad Jurídica")):
    ok(rpt._parte(crudo, autoridad=True) == prosa, f"«{crudo}» → «{prosa}»")


# ═══════════════════════════════════════════════════════════════════════════
# CUARTA RONDA (3-oct-2026, FIXES_R4 · D), contra rev/verif_final.txt. Cada
# comprobación falla con el compositor de la tercera ronda
# (scratchpad/procedencia/r4_D/viejo/).
# ═══════════════════════════════════════════════════════════════════════════
_np = getattr(rpt, "nombre_en_prosa")


def _np_kw(nombre, **kw):
    """`nombre_en_prosa` con la firma de la cuarta ronda; con la de antes,
    que no conocía `con_articulo`, devuelve «» (y la comprobación falla)."""
    try:
        return _np(nombre, **kw)
    except TypeError:
        return ""


print("\n27 · E1 · UN SOLO CONVERTIDOR DE NOMBRES A PROSA (nombre_en_prosa)")
for crudo, kw, prosa, caso in (
        ("Sucesión a Bienes de JOSÉ GARCÍA RUIZ", {"con_articulo": True},
         "la Sucesión a Bienes de José García Ruiz", "AD 335: la racha en versales de un texto mezclado"),
        ("JUAN y PEDRO, ambos de apellidos PÉREZ LÓPEZ", {},
         "Juan y Pedro, ambos de apellidos Pérez López", "AR 208: «ambos de apellidos» con versales"),
        ("sucesión intestamentaria a bienes de JUAN PÉREZ", {"con_articulo": True},
         "la sucesión intestamentaria a bienes de Juan Pérez", "AR 60: la sucesión en minúscula, con su artículo"),
        ("SOCIEDAD DE PRODUCCIÓN RURAL DE RESPONSABILIDAD ILIMITADA, POR CONDUCTO DE SU CONSEJO DE "
         "ADMINISTRACIÓN, Y PARTE QUEJOSA, POR PROPIO DERECHO", {},
         "Sociedad de Producción Rural de Responsabilidad Ilimitada, por conducto de su consejo de "
         "administración, y Parte Quejosa, por propio derecho",
         "AD 552: «por conducto de su consejo de administración»"),
        ("MARÍA PÉREZ LÓPEZ, POR PROPIO DERECHO Y EN REPRESENTACIÓN DE LA MENOR DE INICIALES A.B.C.", {},
         "María Pérez López, por propio derecho y en representación de la menor de iniciales A.B.C.",
         "Q 342: la representación de la menor en minúscula, con sus iniciales"),
        ("JUAN RUIZ, POR CONDUCTO DE SU APODERADO LEGAL PEDRO GÓMEZ", {},
         "Juan Ruiz, por conducto de su apoderado legal Pedro Gómez", "la figura en minúscula, el nombre no"),
        ("SUCESIÓN TESTAMENTARIA A BIENES DE PARTE RECURRENTE, A TRAVÉS DE SU ALBACEA PARTE RECURRENTE",
         {"con_articulo": True},
         "la Sucesión Testamentaria a Bienes de Parte Recurrente, a través de su albacea Parte Recurrente",
         "Q 342: «a través de su albacea»"),
        ("1) PARTE ACTORA y otros (ya mencionados con anterioridad)", {}, "Parte Actora y otros",
         "RF 7: sin la numeración ni la remisión de la carátula"),
        ("1) PARTE ACTORA Y OTROS (YA MENCIONADOS CON ANTERIORIDAD)", {}, "Parte Actora y otros",
         "RF 7 en versales: «y otros» en minúscula"),
        ("PARTE ACTORA Y OTRA", {}, "Parte Actora y otra", "Q 229: «y Otra» → «y otra»"),
        ("ANA LÓPEZ RUIZ (ACTORA)", {}, "Ana López Ruiz", "sin la etiqueta de rol"),
        ("Inmobiliaria del Bajío, S.A. DE C.V.", {}, "Inmobiliaria del Bajío, S.A. de C.V.",
         "la razón social de un nombre mezclado"),
        ("Ma. del Refugio TREJO RAMOS", {}, "Ma. del Refugio Trejo Ramos", "«Ma.» y los apellidos"),
        ("Delegación del ISSSTE en Querétaro", {}, "Delegación del ISSSTE en Querétaro",
         "una sigla sola en un texto mezclado se queda"),
        ("JUAN PÉREZ, POR CONDUCTO DE SU AUTORIZADO EN TÉRMINOS AMPLIOS DEL ARTÍCULO 12 DE LA LEY DE "
         "AMPARO, LUIS RUIZ", {},
         "Juan Pérez, por conducto de su autorizado en términos amplios del artículo 12 de la Ley de "
         "Amparo, Luis Ruiz", "el artículo en minúscula y la Ley de Amparo con su mayúscula"),
        ("SUCESIÓN A BIENES DE JUAN PÉREZ", {}, "Sucesión a Bienes de Juan Pérez",
         "sin `con_articulo`, el colectivo va sin artículo (lo pone quien llama)"),
        ("SUCESIÓN A BIENES DE JUAN PÉREZ", {"con_articulo": True}, "la Sucesión a Bienes de Juan Pérez",
         "con `con_articulo`, «la Sucesión»"),
        ("GOBERNADOR DEL ESTADO (AUTORIDAD RESPONSABLE)", {"autoridad": True}, "el Gobernador del Estado",
         "la autoridad, siempre con su artículo"),
        ("Director de Ingresos del Municipio de Querétaro", {}, "Director de Ingresos del Municipio de "
         "Querétaro", "el órgano reconocido por su cabeza, sin artículo si no se pide")):
    ok(_np_kw(crudo, **kw) == prosa, f"{caso}: «{crudo[:60]}» → «{prosa[:70]}»")
ok(_np_kw(None) == "" and _np_kw({"a": 1}) == "" and _np_kw(12.5) == "",
   "nombre_en_prosa nunca lanza: lo que no es texto vale «»")
_AGR = "AGRÍCOLA LOS PINOS, S.P.R. DE R.L."
o = rpt.componer("amparo_directo", _mezclar(AD, {"promovente": "Sucesión a Bienes de JOSÉ GARCÍA RUIZ",
                                                 "representante": None, "figura_representante": None}),
                 DATOS)
ok("promovido por la Sucesión a Bienes de José García Ruiz, contra" in o["visto"] and
   "JOSÉ" not in _todo(o) and
   o["datos_extra"]["quejoso_en_prosa"] == _np_kw("Sucesión a Bienes de JOSÉ GARCÍA RUIZ",
                                                   con_articulo=True),
   "AD 335: el V I S T O, el resultando y datos_extra dicen el nombre igual que nombre_en_prosa")
o = rpt.componer("amparo_directo", _mezclar(AD, {"promovente": _AGR, "representante": None,
                                                 "figura_representante": None}), DATOS)
ok(_np_kw(_AGR, con_articulo=True) in o["visto"] and _np_kw(_AGR, con_articulo=True) != "",
   "«Agrícola Los Pinos»: el resultando y la legitimación con el mismo convertidor (AD 552)")

print("\n28 · E2 · UN SOLO DETECTOR DE PLURAL, Y VIAJA EN datos_extra['plural']")
_AD552 = ("SOCIEDAD DE PRODUCCIÓN RURAL DE RESPONSABILIDAD ILIMITADA, POR CONDUCTO DE SU CONSEJO DE "
          "ADMINISTRACIÓN, Y PARTE QUEJOSA, POR PROPIO DERECHO")
o = rpt.componer("amparo_directo", _mezclar(AD, {"promovente": _AD552, "representante": None,
                                                 "figura_representante": None}), DATOS)
ok("por propio derecho, promovieron juicio de amparo directo" in _res(o, "Presentación") and
   o["datos_extra"].get("plural") is True and o["datos_extra"].get("plural_quejoso") is True,
   "AD 552 (sociedad y persona en un campo): «promovieron» y plural=True para la legitimación "
   "y la carátula (con tipos_asunto.es_plural_de_partes)")
if hasattr(ta, "es_plural_de_partes"):
    _mal = []
    for _t, _f in (("amparo_directo", _mezclar(AD, {"promovente": _AD552, "representante": None,
                                                     "figura_representante": None})),
                   ("amparo_directo", AD),
                   ("queja", _mezclar(Q, {"promovente": "Ana López Ruiz y Juan Pérez Gómez"})),
                   ("amparo_revision", _mezclar(AR, {"promovente": "Ana Pérez Ruiz y Luis Pérez Ruiz, "
                                                                   "ambos de apellidos Pérez Ruiz",
                                                     "representante": None,
                                                     "figura_representante": None}))):
        _o = rpt.componer(_t, _f, DATOS)
        _v = _o["datos_extra"].get("plural")
        _det = bool(ta.es_plural_de_partes(rpt._prosa_nombre(_f["promovente"])))
        _verbo_pl = bool(re.search(r"\b(?:promovieron|interpusieron)\b", _todo(_o)))
        if not (_v is _det and _verbo_pl == _det):
            _mal.append((_t, _f["promovente"][:30], _v, _det, _verbo_pl))
    ok(not _mal, f"el verbo del resultando, datos_extra['plural'] y tipos_asunto.es_plural_de_partes "
                 f"dicen lo mismo {_mal}")
_RF7 = "1) PARTE ACTORA y otros (ya mencionados con anterioridad)"
o = rpt.componer("revision_fiscal", _mezclar(RF, {"actora": _RF7}), DATOS)
ok(_res(o, "Trámite del juicio").startswith("Parte Actora y otros demandaron la nulidad") and
   o["datos_extra"].get("plural_actora") is True and
   any(a.startswith("EL NOMBRE VIENE ABREVIADO") and "(actora = " in a for a in o["avisos"]),
   "RF 7: «Parte Actora y otros demandaron», sin «1)» ni la remisión, y el aviso del nombre abreviado")
o = rpt.componer("amparo_directo", AD, DATOS)
ok(o["datos_extra"].get("plural") is False and o["datos_extra"].get("plural_quejoso") is False and
   not any("ABREVIADO" in a for a in o["avisos"]), "una sola quejosa: plural=False, sin aviso")
o = rpt.componer("revision_fiscal", RF, DATOS)
ok(o["datos_extra"].get("plural") is False and o["datos_extra"].get("plural_actora") is False,
   "RF: recurre una autoridad (singular) y la actora es una")

print("\n29 · E3 · EL PONENTE CON SU TRATAMIENTO Y CON EL CARGO DEL AUTO")
for crudo, tit, prosa in (
        ("licenciada Bertha Martínez Vega, secretaria en funciones de Magistrada", "",
         "de la licenciada Bertha Martínez Vega, secretaria en funciones de Magistrada"),
        ("LICENCIADO JUAN PÉREZ", "", "del licenciado Juan Pérez"),
        ("Mtra. Ana Ruiz Soto", "", "de la maestra Ana Ruiz Soto"),
        ("Dr. Luis Gómez Díaz", "", "del doctor Luis Gómez Díaz"),
        ("Lic. Bertha Martínez Vega", "", "de Bertha Martínez Vega"),
        ("Jenica Campos Juárez", "Magistrada", "de la Magistrada Jenica Campos Juárez"),
        ("Jenica Campos Juárez", "", "de Jenica Campos Juárez")):
    try:
        _got = rpt._ponente_en_prosa(crudo, tit) if tit else rpt._ponente_en_prosa(crudo)
    except TypeError:
        _got = ""
    ok(_got == prosa, f"«{crudo}»{' + titulo «' + tit + '»' if tit else ''} → «{prosa}»")
o = rpt.componer("amparo_directo", _mezclar(AD, {"turno": {"ponente": "Lic."}}), DATOS)
ok(f"ponencia de {H}," in _res(o, "Turno") and any("(turno.ponente)" in a for a in o["avisos"]),
   "un ponente que sólo es un tratamiento («Lic.») es un ponente que falta: hueco y aviso")
o = rpt.componer("amparo_revision", _mezclar(AR, {"turno": {"ponente": "Bertha Martínez Vega",
                                                            "titulo": ""},
                                                  "returno": {"fecha": "2025-11-20",
                                                              "ponente": "Jenica Campos Juárez",
                                                              "titulo": "Magistrada"}}), DATOS)
ok("se returnaron los autos a la ponencia de la Magistrada Jenica Campos Juárez," in _res(o, "Returno")
   and "ponencia de Bertha Martínez Vega," in _res(o, "Turno") and
   o["datos_extra"].get("ponente") == "Jenica Campos Juárez" and
   o["datos_extra"].get("ponente_titulo") == "Magistrada",
   "Q 342/AR: el cargo que leyó el auto de returno se escribe y viaja en datos_extra['ponente_titulo']; "
   "el turno sin cargo, neutro")
o = rpt.componer("amparo_directo", _mezclar(AD, {"turno": {"titulo": "Magistrado"},
                                                 "returno": {"fecha": "2026-05-04",
                                                             "ponente": "Ana Laura Campos Juárez"}}),
                 DATOS)
ok(o["datos_extra"].get("ponente_titulo") == "" and
   "ponencia de Ana Laura Campos Juárez," in _res(o, "Returno"),
   "el cargo del turno no se le pone al ponente del returno (no se adivina)")

print("\n30 · E5 · EL ADHESIVO SIN NINGÚN AUTO (AD 274/2025)")
for _t, _base, _sec, _preg in (("amparo_directo", AD, "Trámite", "¿HUBO AMPARO ADHESIVO?"),
                               ("amparo_revision", AR, "Interposición", "¿HUBO REVISIÓN ADHESIVA?"),
                               ("revision_fiscal", RF, "Trámite del recurso", "¿HUBO REVISIÓN ADHESIVA?")):
    _f = _mezclar(_base, {"adhesivo": {"quien": "Ana Pérez Ruiz"}})
    o = rpt.componer(_t, _f, DATOS)
    _avs = [a for a in o["avisos"] if "(adhesivo.admision)" in a]
    ok(f"Por auto de {H}, se" in _res(o, _sec) and "Ana Pérez Ruiz" in _res(o, _sec) and
       len(_avs) == 1 and _avs[0].startswith(_preg),
       f"{_t}: el resultando lo menciona con la fecha en hueco y UN aviso «{_preg}»")
    ok(rpt.considerando_adhesivo(_t, _f, DATOS)[:2] == ("", "") and
       rpt.resolutivo_adhesivo(_t, False, _f)[0] == "" and
       rpt.resolutivo_adhesivo(_t, "desecha", _f)[0] == "",
       f"{_t}: sin considerando ni punto resolutivo del adhesivo")
    ok(o["datos_extra"].get("adhesivo") is None and o["datos_extra"].get("adherente") == "" and
       o["datos_extra"].get("adhesivo_sin_auto") is True,
       f"{_t}: datos_extra no lo da por existente (sin renglón en la carátula) y lo marca")
_f = _mezclar(AD, {"adhesivo": {"quien": "Ana Pérez Ruiz", "presentacion": "2026-03-10"}})
ok(bool(rpt.considerando_adhesivo("amparo_directo", _f, DATOS)[0]) and
   rpt.componer("amparo_directo", _f, DATOS)["datos_extra"].get("adhesivo_sin_auto") is False,
   "con el escrito presentado sí hay adhesivo: su considerando se escribe (con su hueco)")

print("\n31 · E6 · EL CORREO CON UNA SOLA REGLA (RF 2/2025)")
o = rpt.componer("revision_fiscal", _mezclar(RF, {"via_presentacion": "responsable",
                                                 "deposito_postal": "2025-10-08",
                                                 "presentacion": "2025-10-14"}), DATOS)
ok("depositado en el Servicio Postal Mexicano el ocho de octubre de dos mil veinticinco y recibido el "
   "catorce de octubre" in _res(o, "Interposición") and o["datos_extra"].get("via_postal") is True and
   o["datos_extra"].get("deposito_postal_iso") == "2025-10-08",
   "vía «responsable» con depósito anterior: por correo, y via_postal=True para el cómputo")
o = rpt.componer("revision_fiscal", _mezclar(RF, {"via_presentacion": "responsable",
                                                 "deposito_postal": "2025-10-20",
                                                 "presentacion": "2025-10-14"}), DATOS)
ok("Servicio Postal" not in _res(o, "Interposición") and
   "mediante oficio presentado el catorce de octubre" in _res(o, "Interposición") and
   o["datos_extra"].get("via_postal") is False and o["datos_extra"].get("deposito_postal_iso") == "",
   "depósito POSTERIOR a la recepción: no cuenta como correo (y validar acusa la fecha imposible)")
try:
    import ficha_tramite as _ftp
    if hasattr(_ftp, "via_postal"):
        _mal = []
        for _dp in ("2025-10-08", "2025-10-14", "2025-10-20", ""):
            _fd = _mezclar(RF, {"via_presentacion": "responsable", "deposito_postal": _dp,
                                "presentacion": "2025-10-14"})
            if rpt.componer("revision_fiscal", _fd, DATOS)["datos_extra"].get("via_postal") is not \
                    bool(_ftp.via_postal(_fd)):
                _mal.append(_dp)
        ok(not _mal, f"el compositor decide el correo igual que ficha_tramite.via_postal {_mal}")
except Exception as _ex:
    ok(False, f"no se pudo comparar con ficha_tramite.via_postal ({_ex})")

print("\n32 · E8 · LA UNIDAD QUE RECURRE Y SU TITULAR (RF 7, 2, 26, 4/2025 y 6/2026)")
_BCS = "Delegación Estatal del ISSSTE en Baja California Sur"
_SUBD_BCS = f"Subdelegado del Departamento de Pensiones de la {_BCS}"
o = rpt.componer("revision_fiscal", _mezclar(RF, {
    "promovente": f"JEFA DE LA UNIDAD JURÍDICA DE LA {_BCS.upper()}, EN REPRESENTACIÓN DEL "
                  f"{_SUBD_BCS.upper()}",
    "figura_representante": "Jefa de la Unidad Jurídica", "representante": "MARÍA LÓPEZ RUIZ",
    "autoridad_demandada": _SUBD_BCS}), DATOS)
t2 = _res(o, "Interposición")
ok(t2.startswith(f"Inconforme, la Jefa de la Unidad Jurídica de la {_BCS}, en representación del "
                 f"Subdelegado del Departamento de Pensiones") and "por conducto" not in t2,
   "RF 7: la titular de la propia unidad no es «su» representante: sin «por conducto de su Jefa…»")
ok(o["datos_extra"].get("recurrente_nombre") == "María López Ruiz" and
   o["datos_extra"].get("recurrente_unidad_con_articulo") == f"la Jefa de la Unidad Jurídica de la {_BCS}",
   "RF 7: datos_extra lleva el nombre de la titular y la unidad con su artículo, para «lo hizo valer "
   "{nombre}, {unidad}»")
_ISSSTE = "Instituto de Seguridad y Servicios Sociales de los Trabajadores del Estado"
o = rpt.componer("revision_fiscal", _mezclar(RF, {
    "promovente": f"Subdirector de lo Contencioso del {_ISSSTE}",
    "figura_representante": "Subdirector de lo Contencioso", "representante": "Juan Pérez Gómez",
    "autoridad_demandada": f"Subdirectora de Afiliación y Vigencia del {_ISSSTE}"}), DATOS)
ok(_res(o, "Interposición").startswith(f"Inconforme, el Subdirector de lo Contencioso del {_ISSSTE}, "
                                       f"en representación de la Subdirectora de Afiliación") and
   "por conducto" not in _res(o, "Interposición") and
   o["datos_extra"].get("recurrente_nombre") == "Juan Pérez Gómez",
   "RF 26: «el Subdirector de lo Contencioso…, en representación de la Subdirectora…», como el engrose")
o = rpt.componer("revision_fiscal", _mezclar(RF, {
    "promovente": "Subdelegado de Prestaciones Económicas, Delegación Estatal Querétaro, del ISSSTE",
    "figura_representante": "Titular de la Unidad Jurídica…",
    "representante": "Titular de la Unidad de Asuntos Jurídicos de la citada delegación",
    "autoridad_demandada": "Subdelegación de Prestaciones Económicas del ISSSTE en Querétaro"}), DATOS)
t2 = _res(o, "Interposición")
ok("por conducto del Titular de la Unidad de Asuntos Jurídicos de la citada delegación, interpuso" in t2
   and "en representación" not in t2 and "…" not in t2,
   "RF 6/2026: Subdelegado y Subdelegación son la misma autoridad; la figura truncada fuera y el "
   "representante-cargo como figura")
ok(any("(figura_representante = «Titular de la Unidad de Asuntos Jurídicos" in a and "POR OMISIÓN" in a
       for a in o["avisos"]) and any(("CORTAD" in a or "TRUNCAD" in a) for a in o["avisos"]) and
   any("CARGO" in a for a in o["avisos"]),
   "RF 6/2026: avisos del dato truncado, del representante-cargo y de «el Titular» por omisión")
ok(sum(("CORTAD" in a or "TRUNCAD" in a) for a in o["avisos"]) == 1 and
   sum(("ES UN CARGO" in a or "ES EL CARGO" in a) for a in o["avisos"]) == 1,
   "E11: el dato truncado y el representante-cargo, una sola vez cada uno (validar o compositor)")
o = rpt.componer("revision_fiscal", _mezclar(RF, {
    "promovente": f"Subdelegado de Prestaciones Económicas, Delegación Estatal Querétaro, del {_ISSSTE}",
    "figura_representante": "", "representante": "Titular de la Unidad de Asuntos Jurídicos de la "
                                                 "citada delegación",
    "autoridad_demandada": "Subdelegación de Prestaciones Económicas del ISSSTE en Querétaro"}), DATOS)
ok("en representación" not in _res(o, "Interposición") and
   "por conducto del Titular de la Unidad de Asuntos Jurídicos de la citada delegación, interpuso" in
   _res(o, "Interposición"),
   "RF 6/2026 (fichas2): el ISSSTE con su nombre largo y con su sigla es el mismo; nada de «el "
   "Subdelegado…, en representación de la Subdelegación…»")
for a, b, igual in (("Subdelegado de Prestaciones Económicas del ISSSTE",
                     "Subdelegación de Prestaciones Económicas del ISSSTE", True),
                    ("Director de Ingresos", "Dirección de Ingresos", True),
                    ("Jefe de Servicios Jurídicos", "Jefatura de Servicios Jurídicos", True),
                    ("Titular de la Jefatura de Servicios Jurídicos", "Jefatura de Servicios Jurídicos",
                     True),
                    ("Unidad Jurídica de la Delegación Querétaro", "Delegación Querétaro", False)):
    ok(rpt._misma_autoridad(a, b) is igual, f"misma autoridad «{a}» ~ «{b}»: {igual}")
o = rpt.componer("revision_fiscal", _mezclar(RF, {
    "figura_representante": "autorizado en términos del artículo 5o. de la Ley Federal de "
                            "Procedimiento Contencioso Administrativo", "representante": "Luis Ruiz"}),
    DATOS)
ok(sum(bool(re.search(r"(?i)autorizad", a)) for a in o["avisos"]) == 1,
   "RF 4: el autorizado que interpone por la autoridad lleva UN aviso (el 63 exige la unidad jurídica)")
_LA_TIT = ("LA TITULAR DE LA UNIDAD JURÍDICA DE LA OFICINA DE REPRESENTACIÓN DEL ISSSTE EN QUERÉTARO, "
           "EN REPRESENTACIÓN DEL SUBDELEGADO DE PRESTACIONES ECONÓMICAS")
o = rpt.componer("revision_fiscal", _mezclar(RF, {"promovente": _LA_TIT,
                                                 "autoridad_demandada": "Subdelegado de Prestaciones "
                                                                        "Económicas"}), DATOS)
ok(_res(o, "Interposición").startswith("Inconforme, la Titular de la Unidad Jurídica") and
   o["datos_extra"].get("recurrente_unidad_con_articulo", "").startswith("la Titular de la Unidad") and
   o["datos_extra"].get("recurrente_la_titular") is True and
   not any("POR OMISIÓN" in a for a in o["avisos"]),
   "RF 4/49: «la Titular» del papel viaja con su artículo a datos_extra (la legitimación la conserva)")
o = rpt.componer("revision_fiscal", _mezclar(RF, {"promovente": RF["promovente"][3:]}), DATOS)
ok(_res(o, "Interposición").startswith("Inconforme, el Titular de la Jefatura") and
   any(a.startswith("«EL TITULAR» VA POR OMISIÓN (promovente") for a in o["avisos"]),
   "«Titular» sin artículo: «el Titular» y el aviso de género por omisión")
o = rpt.componer("amparo_revision", _mezclar(AR, {
    "promovente": "Director de Ingresos del Municipio de Querétaro", "caracter": "autoridad",
    "figura_representante": "Director de Ingresos", "representante": "JUAN PÉREZ GÓMEZ",
    "acto": {"resolvio": "concede"}}),
    _mezclar(DATOS, {"quejoso": "Impulsora de Desarrollos Inmobiliarios VV, S.A. de C.V."}))
ok("por conducto" not in _res(o, "Interposición") and
   o["datos_extra"].get("recurrente_nombre") == "Juan Pérez Gómez",
   "AR: la autoridad que recurre por su propio titular, sin «por conducto de su Director…»")

print("\n33 · E9 · EL ARTÍCULO 12 (Y EL 9o.) CON SU LEY, EN EL RESULTANDO")
for fig, prosa in (("autorizado en términos amplios del artículo 12",
                    "autorizado en términos amplios del artículo 12 de la Ley de Amparo"),
                   ("delegado en términos del artículo 9o.",
                    "delegado en términos del artículo 9o. de la Ley de Amparo"),
                   ("autorizado en términos amplios del artículo 12 de la Ley de Amparo",
                    "autorizado en términos amplios del artículo 12 de la Ley de Amparo"),
                   ("autorizado en términos del artículo 5o. de la Ley Federal de Procedimiento "
                    "Contencioso Administrativo",
                    "autorizado en términos del artículo 5o. de la Ley Federal de Procedimiento "
                    "Contencioso Administrativo"),
                   ("autorizado del artículo 120", "autorizado del artículo 120")):
    ok(rpt._figura_en_prosa(fig) == prosa, f"«{fig}» → «{prosa}»")
o = rpt.componer("amparo_revision", _mezclar(AR, {
    "figura_representante": "autorizado en términos amplios del artículo 12",
    "representante": "LUIS GÓMEZ DÍAZ"}), DATOS)
ok("por conducto de su autorizado en términos amplios del artículo 12 de la Ley de Amparo, "
   "Luis Gómez Díaz, interpuso" in _res(o, "Interposición"),
   "AR 239 y 60: el resultando no cita un artículo sin su ley")

print("\n34 · E7 · EL REGISTRO Y LA ADMISIÓN, DOS AUTOS CUANDO SON DOS")
o = rpt.componer("revision_fiscal", _mezclar(RF, {"registro": {"fecha": "2025-10-20"}}), DATOS)
ok(_res(o, "Trámite del recurso") == (
    "Por auto de Presidencia de veinte de octubre de dos mil veinticinco, este Tribunal Colegiado "
    "registró el recurso con el número 62/2025; y por auto de tres de noviembre de dos mil "
    "veinticinco lo admitió a trámite."), "RF 7: el registro y la admisión, cada uno con su auto")
o = rpt.componer("amparo_directo", _mezclar(AD, {"registro": {"fecha": "2026-02-16"}}), DATOS)
ok(_res(o, "Trámite").startswith(
    "Por auto de Presidencia de dieciséis de febrero de dos mil veintiséis, este Tribunal Colegiado "
    "registró la demanda con el número 174/2026; y por auto de veinte de febrero de dos mil "
    "veintiséis la admitió a trámite y concedió a las partes el plazo de quince días"),
   "AD: dos autos, y el 181 con la admisión")
o = rpt.componer("amparo_revision", _mezclar(AR, {"registro": {"fecha": "2025-09-05"}}), DATOS)
ok("Por auto de Presidencia de cinco de septiembre de dos mil veinticinco, este Tribunal Colegiado lo "
   "registró con el número 513/2025; y por auto de diecinueve de septiembre de dos mil veinticinco "
   "lo admitió a trámite." in _res(o, "Interposición"), "AR: dos autos")
o = rpt.componer("queja", _mezclar(Q, {"registro": {"fecha": "2026-03-11"}}), DATOS)
ok(_res(o, "Trámite").startswith("Por auto de Presidencia de once de marzo de dos mil veintiséis, este "
                                 "Tribunal Colegiado registró el recurso con el número 150/2026; y por "
                                 "auto de trece de marzo de dos mil veintiséis lo admitió a trámite."),
   "queja fr. I: dos autos")
o = rpt.componer("revision_fiscal", _mezclar(RF, {"registro": {"fecha": "2025-11-03"}}), DATOS)
ok(_res(o, "Trámite del recurso").startswith("Por auto de Presidencia de tres de noviembre de dos mil "
                                             "veinticinco, este Tribunal Colegiado registró el "
                                             "recurso con el número 62/2025 y lo admitió a trámite."),
   "el registro con la fecha de la admisión es el mismo auto: una sola frase")
_Q335 = _mezclar(Q, {"fraccion_97": "II", "inciso_97": "b", "turno": {"fecha": "2025-11-10"},
                     "presentacion": "2025-10-10", "numero": "335/2025",
                     "registro": {"fecha": "2025-10-14"}, "admision": {"fecha": "2025-11-03"},
                     "acto": {"fecha": "2025-10-06",
                              "organo": "Primera Sala Civil del Tribunal Superior de Justicia del "
                                        "Estado de Querétaro", "expediente": "174/2025",
                              "sentido": "concedió la suspensión del acto reclamado"}})
o = rpt.componer("queja", _Q335, DATOS)
ok(_res(o, "Trámite").startswith(
    "Por auto de Presidencia de catorce de octubre de dos mil veinticinco, este Tribunal Colegiado "
    "registró el recurso con el número 335/2025 y requirió a la autoridad responsable su informe con "
    "justificación sobre la materia de la queja (artículo 101 de la Ley de Amparo); por auto de tres "
    "de noviembre de dos mil veinticinco lo tuvo por rendido y admitió el recurso.") and not o["avisos"],
   f"Q 335: el registro (14-oct) y el auto que tiene por rendido el informe y admite (3-nov) "
   f"{o['avisos'][:1]}")
o = rpt.componer("queja", _mezclar(_Q335, {"registro": {"fecha": ""}}), DATOS)
ok(_res(o, "Trámite").startswith(f"Por auto de Presidencia de {H}, este Tribunal Colegiado registró") and
   _res(o, "Trámite").count("tres de noviembre") == 1 and
   any("(registro.fecha)" in a for a in o["avisos"]),
   "Q 335 sin registro: hueco con aviso, y la fecha de la admisión no se escribe en los dos autos")
o = rpt.componer("queja", _mezclar(_Q335, {"registro": {"fecha": ""},
                                           "admision": {"fecha": "2025-10-14"},
                                           "informe_101": {"fecha": "2025-11-03"}}), DATOS)
ok(any("(registro.fecha)" in a and "2025-10-14" in a and "forma y registra" in a for a in o["avisos"]),
   "ficha vieja (la «admisión» era el registro): el aviso propone pasarla al campo del registro")
o = rpt.componer("revision_fiscal", _mezclar(RF, {"registro": {"fecha": "20/13/2025"}}), DATOS)
ok(any("FECHA ILEGIBLE" in a and "(registro.fecha" in a for a in o["avisos"]) and
   "registró el recurso con el número 62/2025 y lo admitió" in _res(o, "Trámite del recurso"),
   "un registro ilegible se avisa y queda un solo auto")

print("\n35 · E10 · AR: EL ÓRGANO EN FORMA DE ÓRGANO, Y EL DESCARTADO EN HUECO HASTA LOS CONSIDERANDOS")
o = rpt.componer("amparo_revision", _mezclar(AR, {"acto": {
    "organo": "Jueza Tercera de Distrito en el Estado de Querétaro"}}), DATOS)
ok("por el Juzgado Tercero de Distrito en el Estado de Querétaro, en el juicio" in o["visto"] and
   "correspondió al Juzgado Tercero de Distrito" in _res(o, "Trámite del juicio") and
   o["datos_extra"]["juzgado"] == o["datos_extra"].get("organo_recurrido") ==
   "Juzgado Tercero de Distrito en el Estado de Querétaro",
   "AR 239: «la Jueza Tercera» pasa a «el Juzgado Tercero» (no se adivina el género)")
o = rpt.componer("amparo_revision", _mezclar(AR, {"acto": {"organo": _SALA}}), DATOS)
_fj = [a for a in o["avisos"] if a.startswith("FALTA EL JUZGADO")]
# C1 (3-oct-2026): el AR ya no lleva existencia; el aviso nombra sólo la
# competencia (antes: «…y también en la competencia y en la existencia»).
ok(o["datos_extra"].get("organo_recurrido") == H and o["datos_extra"].get("juzgado_descartado") == _SALA
   and H in _res(o, "Trámite del juicio") and bool(_fj) and
   "«V I S T O» y «Trámite del juicio de amparo indirecto», y también en la competencia" in _fj[0]
   and "existencia" not in _fj[0],
   "AR 307: el órgano descartado llega en hueco a la competencia (organo_recurrido = hueco) y el aviso "
   "nombra todos los apartados, sin la existencia que el AR ya no lleva (C1)")
o = rpt.componer("amparo_revision", _mezclar(AR, {"acto": {"organo": _SALA}}),
                 _mezclar(DATOS, {"organo_recurrido": "Juzgado Primero de Distrito en el Estado de "
                                                      "Querétaro"}))
ok(o["datos_extra"].get("organo_recurrido") == "Juzgado Primero de Distrito en el Estado de Querétaro"
   and o["datos_extra"].get("juzgado_descartado") == "",
   "con otra fuente, el órgano recurrido es ése y no hay hueco")
o = rpt.componer("queja", _mezclar(Q, {"acto": {"organo": ""}}), {})
_fo = [a for a in o["avisos"] if "(acto.organo)" in a]
ok(bool(_fo) and "«V I S T O» y «Interposición del recurso de queja», y también en la competencia y "
   "en la procedencia" in _fo[0], "Q 335: el aviso del órgano que falta nombra todos los apartados")

print("\n36 · E11 · UN AVISO POR DATO (validar y el compositor no repiten el mismo hecho)")
_f307 = _mezclar(AR, {"acto": {"organo": _SALA}})
o = rpt.componer("amparo_revision", _f307, DATOS)
_val307 = rpt._validada(_f307, "amparo_revision")[1]
ok(not any("ES TAMBIÉN UNA DE LAS AUTORIDADES" in a for a in o["avisos"]) and
   any(a.startswith("FALTA EL JUZGADO") for a in o["avisos"]) and
   all(a in o.get("avisos_repetidos", []) for a in _val307 if "ES TAMBIÉN UNA DE LAS" in a),
   "AR 307: el órgano descartado se avisa una vez (el del compositor, que dice dónde quedó el hueco), "
   "y el de validar va en avisos_repetidos para quitarlo también de ficha['avisos']")
o = rpt.componer("amparo_revision", _mezclar(AR, {"acto": {"fecha": "2025-03-03"}, "audiencia": "",
                                                 "demanda": {"fecha": ""}}), DATOS)
ok(sum(("ACTO RECLAMADO" in a and "FECHA" in a) for a in o["avisos"]) == 1,
   "AR 307: la fecha de la recurrida dentro del acto reclamado, un solo aviso")
o = rpt.componer("amparo_revision", _mezclar(AR, {"acto": {
    "organo": "Juez Octavo de Primera Instancia Familiar del Distrito Judicial de Querétaro"}}), DATOS)
ok(sum(("NO PARECE UN JUZGADO DE DISTRITO" in a or "NO TIENE FORMA DE ÓRGANO DE AMPARO" in a)
       for a in o["avisos"]) == 1, "AR 448: el órgano sin forma de juzgado de amparo, un solo aviso")
o = rpt.componer("amparo_revision", _mezclar(AR, {"audiencia": "2025-08-15"}), DATOS)
ok(any(a.startswith("FECHA IMPOSIBLE") for a in o["avisos"]) and o.get("avisos_repetidos") == [],
   "lo que sólo dice validar (la fecha imposible) se queda, y no hay repetidos que quitar")

print("\n37 · E12 · CON SURTIMIENTO EL MISMO DÍA, SIN «es decir» (adhesivo)")
rot, txt, avs = rpt.considerando_adhesivo("amparo_revision", _mezclar(AR, {"adhesivo": AR_ADHESIVO}),
                                          DATOS)
ok("surtió efectos el mismo día, conforme al artículo 31, fracción I, de la Ley de Amparo; por tanto"
   in txt and "es decir" not in txt, "AR adherente autoridad: «el mismo día» sin repetir la fecha")
rot, txt, avs = rpt.considerando_adhesivo("amparo_directo", _mezclar(AD, {"adhesivo": AD_ADHESIVO}),
                                          DATOS)
ok("al día hábil siguiente, conforme al artículo 31, fracción II, de la Ley de Amparo, es decir, el "
   in txt, "AD adherente particular: el surtimiento al día siguiente sí dice la fecha")


# ═══════════════════════════════════════════════════════════════════════════
# QUINTA RONDA (3-oct-2026, FIXES_R5 · D), contra rev/verif_final2.txt. Cada
# comprobación falla con el compositor de la cuarta ronda
# (scratchpad/procedencia/r5_D/viejo/).
# ═══════════════════════════════════════════════════════════════════════════
print("\n38 · F3 · LA FRACCIÓN DEL 97 SÓLO CON CERTEZA (Q 335/2025)")
for _org, _fr in (("Juez Segundo de lo Civil del Distrito Judicial de Querétaro", "II"),
                  ("Juzgado Mixto de Primera Instancia del Distrito Judicial de Jalpan", "II"),
                  ("Juez Cuarto de Primera Instancia Familiar del Distrito Judicial de Querétaro", "II"),
                  ("Juzgado Tercero de Oralidad Mercantil del Distrito Judicial de Querétaro", "II"),
                  ("Juzgado Sexto de Distrito en Materia de Amparo Civil, Administrativo y de Trabajo y "
                   "de Juicios Federales en el Estado de Querétaro", "I"),
                  ("Juez Séptimo de Distrito en el Estado de Querétaro", "I"),
                  ("Comisión Estatal de Aguas", "")):
    ok(rpt._fraccion_97({}, _org) == _fr, f"«{_org[:60]}» → fracción «{_fr}»")
_q335 = _mezclar(Q, {"fraccion_97": None, "inciso_97": None,
                     "acto": {"organo": "Juez Segundo de lo Civil del Distrito Judicial de Querétaro"}})
o = rpt.componer("queja", _q335, DATOS)
ok(o["datos_extra"]["fraccion_97"] == "II" and o["datos_extra"]["via_amparo"] == "directo" and
   "en el juicio de amparo directo 312/2026; y," in o["visto"] and "indirecto" not in _todo(o),
   "Q 335: el juez «de lo Civil del Distrito Judicial» es la responsable de un amparo directo "
   "(fracción II), no un Juzgado de Distrito")
o = rpt.componer("queja", _mezclar(Q, {"fraccion_97": None, "acto": {"organo": "Comisión Estatal de "
                                                                              "Aguas"}}), DATOS)
_a97 = [a for a in o["avisos"] if "(fraccion_97)" in a]
ok(o["datos_extra"]["fraccion_97"] == "" and o["datos_extra"]["via_amparo"] == H and
   f"en el juicio de amparo {H} 312/2026" in o["visto"] and bool(_a97) and
   "también en la competencia y en la procedencia" in _a97[0],
   "sin certeza: fracción y vía en hueco (sin suponer la I) y el aviso nombra la competencia y la "
   "procedencia")

print("\n39 · F4 · UNA SOLA COMPARACIÓN DE AUTORIDADES (RF 7/2025, Representación/Delegación)")
_ISSSTE7 = "Instituto de Seguridad y Servicios Sociales de los Trabajadores del Estado en Baja California Sur"
o = rpt.componer("revision_fiscal", _mezclar(RF, {
    "promovente": f"la Jefa de la Unidad Jurídica de la Delegación Estatal del {_ISSSTE7}",
    "figura_representante": f"Jefa de la Unidad Jurídica de la Representación Estatal del {_ISSSTE7}",
    "representante": "María Luisa Gómez Ríos",
    "autoridad_demandada": "Subdelegado de Pensiones, Seguridad e Higiene de la Delegación Estatal "
                           "Baja California Sur del Instituto de Seguridad y Servicios Sociales de "
                           "los Trabajadores del Estado"}), DATOS)
t2 = _res(o, "Interposición")
ok(t2.startswith(f"Inconforme, la Jefa de la Unidad Jurídica de la Delegación Estatal del {_ISSSTE7}, "
                 "en representación del Subdelegado de Pensiones") and "por conducto" not in t2 and
   "María Luisa" not in t2 and o["datos_extra"].get("recurrente_nombre") == "María Luisa Gómez Ríos",
   "RF 7: la misma unidad con su nombre viejo y el nuevo no es «X, por conducto de su X»; el nombre "
   "va a datos_extra para la aposición de la legitimación")
ok(rpt._misma_autoridad(f"Jefa de la Unidad Jurídica de la Representación Estatal del {_ISSSTE7}",
                        f"la Jefa de la Unidad Jurídica de la Delegación Estatal del {_ISSSTE7}") is
   bool(ta.misma_autoridad(f"Jefa de la Unidad Jurídica de la Representación Estatal del {_ISSSTE7}",
                           f"la Jefa de la Unidad Jurídica de la Delegación Estatal del {_ISSSTE7}")) is True,
   "el compositor compara con tipos_asunto.misma_autoridad (la de la legitimación y la verja)")

print("\n40 · F2 · LA FIGURA SIN NOMBRE NO SE BORRA (AR 72/2025, Q 342/2025)")
o = rpt.componer("amparo_revision", _mezclar(AR, {
    "promovente": "Gobernador del Estado de Querétaro", "caracter": "autoridad", "representante": "",
    "figura_representante": "delegado", "acto": {"resolvio": "concede"}}),
    _mezclar(DATOS, {"quejoso": "Juan Pérez Gómez"}))
_ar = [a for a in o["avisos"] if "(representante)" in a]
ok("Inconforme, el Gobernador del Estado de Querétaro, en su carácter de autoridad responsable, por "
   f"conducto de su delegado {H}, interpuso recurso de revisión" in _res(o, "Interposición") and
   len(_ar) == 1 and "también en la legitimación" in _ar[0] and "delegado" in _ar[0],
   "AR 72: «por conducto de su delegado *********» y UN aviso con la clave «representante»")
o = rpt.componer("queja", _mezclar(Q, {
    "promovente": "SUCESIÓN TESTAMENTARIA A BIENES DE JOSÉ RUIZ, A TRAVÉS DE SU ALBACEA ANA RUIZ",
    "representante": "", "figura_representante": "autorizado en términos amplios del artículo 12"}),
    DATOS)
ok("parte quejosa en el juicio de amparo indirecto 312/2026, por conducto de su autorizado en términos "
   f"amplios del artículo 12 de la Ley de Amparo, {H}, interpuso recurso de queja" in _res(o, "Interposición")
   and any("(representante)" in a for a in o["avisos"]),
   "Q 342: el autorizado sin nombre se escribe (con su ley y el separador de la figura larga)")
o = rpt.componer("amparo_directo", _mezclar(AD, {"representante": "",
                                                 "figura_representante": "apoderado legal"}),
                 _mezclar(DATOS, {"representante": "", "figura_representante": ""}))
ok(f"Ruth Gabriela Larrañaga Silva, por conducto de su apoderado legal {H}, promovió" in
   _res(o, "Presentación"),
   "AD: la figura de la ficha no la borra un encargo sin representante")
o = rpt.componer("amparo_revision", _mezclar(AR, {"representante": "", "figura_representante": ""}), DATOS)
ok("por conducto" not in _res(o, "Interposición") and not any("(representante)" in a for a in o["avisos"]),
   "sin figura ni nombre, nada (como antes)")

print("\n41 · F5 · EL ARTÍCULO DEL PAPEL EN LOS CARGOS DE DOBLE GÉNERO (AD 128/2025)")
_OM = "Oficial Mayor y Coordinadora de Recursos Humanos del Municipio de Cadereyta de Montes, Querétaro"
o = rpt.componer("amparo_directo", _mezclar(AD, {"terceros": ["la " + _OM]}), DATOS)
ok(_res(o, "Tercero") == f"Tiene ese carácter la {_OM}." and
   not any("GÉNERO DEL CARGO" in a for a in o["avisos"]),
   "AD 128: «la Oficial Mayor…» del papel se conserva (antes «el Oficial Mayor»)")
o = rpt.componer("amparo_directo", _mezclar(AD, {"terceros": [_OM]}), DATOS)
ok(_res(o, "Tercero") == f"Tiene ese carácter el {_OM}." and
   any(a.startswith("EL GÉNERO DEL CARGO NO CONSTA (terceros") for a in o["avisos"]),
   "sin artículo en el papel: «el» (masculino genérico) y el aviso «EL GÉNERO DEL CARGO NO CONSTA»")
for crudo, prosa in (("la Fiscal Especializada en Delitos Patrimoniales", "la Fiscal Especializada en "
                                                                         "Delitos Patrimoniales"),
                     ("la Agente del Ministerio Público", "la Agente del Ministerio Público"),
                     ("la Titular de la Unidad Jurídica", "la Titular de la Unidad Jurídica"),
                     ("Agente del Ministerio Público", "el Agente del Ministerio Público")):
    ok(rpt._parte(crudo, autoridad=True) == prosa, f"«{crudo}» → «{prosa}»")
o = rpt.componer("amparo_directo", _mezclar(AD, {"terceros": [
    "Oficial Mayor del Municipio de Cadereyta de Montes, Querétaro",
    "la Coordinadora de Recursos Humanos del Municipio"]}), DATOS)
ok("el Oficial Mayor del Municipio de Cadereyta de Montes, Querétaro, y la Coordinadora" in
   _res(o, "Tercero"), "AD 128: la coma antes de la «y» final cuando el elemento anterior acaba en un inciso")
ok(rpt._y(["Metlife México, S.A. de C.V.", "Unión de Crédito del Bajío, S.A. de C.V."]) ==
   "Metlife México, S.A. de C.V. y Unión de Crédito del Bajío, S.A. de C.V.",
   "la razón social que acaba en su abreviatura no la necesita")
o = rpt.componer("revision_fiscal", _mezclar(RF, {"autoridad_demandada": "Titular de la Subdelegación "
                                                                         "Querétaro del IMSS"}), DATOS)
ok(any(a.startswith("«EL TITULAR» VA POR OMISIÓN (autoridad_demandada") for a in o["avisos"]),
   "RF: la demandada «Titular» sin artículo lleva su aviso de género por omisión")

print("\n42 · F6 · UNA SOLA PUERTA PARA LA FIGURA (AD 456 y 274, Q 172 y 300)")
_sepf = getattr(rpt, "_sep_figura", None)
for _fig in ("Apoderado Legal", "APODERADO LEGAL", "autorizado en términos amplios del artículo 12",
             "Jefa de la Unidad Jurídica", "autorizada en términos amplios", "delegado"):
    ok(callable(_sepf) and rpt._figura_en_prosa(_fig) == ta.figura_en_prosa(_fig) and
       _sepf(rpt._figura_en_prosa(_fig)) == ta._sep_figura(ta.figura_en_prosa(_fig)),
       f"«{_fig}»: la misma figura y el mismo separador que tipos_asunto")
o = rpt.componer("amparo_revision", _mezclar(AR, {"figura_representante": "Apoderado Legal",
                                                 "representante": "Juan Ruiz Gómez"}), DATOS)
ok("por conducto de su apoderado legal Juan Ruiz Gómez, interpuso" in _res(o, "Interposición"),
   "AD 456: «su apoderado legal», no «su Apoderado Legal» (como la legitimación y el resolutivo)")

print("\n43 · E1 · MÁS FÓRMULAS DEL NOMBRE EN MINÚSCULA (verif_q2 t4, AD 274, 335 y 552)")
for crudo, kw, prosa, caso in (
        ("INSTITUTO MEXICANO DEL SEGURO SOCIAL, POR CONDUCTO DE SU APODERADO LEGAL JUAN RUIZ", {},
         "Instituto Mexicano del Seguro Social, por conducto de su apoderado legal Juan Ruiz",
         "el organismo quejoso, por el camino de órgano"),
        ("MENOR DE EDAD DE INICIALES J.L.M., POR CONDUCTO DE SU MADRE MARÍA LÓPEZ", {},
         "menor de edad de iniciales J.L.M., por conducto de su madre María López", "la madre"),
        ("JUAN PÉREZ, EN SU CARÁCTER DE ALBACEA DE LA SUCESIÓN A BIENES DE PEDRO PÉREZ", {},
         "Juan Pérez, en su carácter de albacea de la sucesión a bienes de Pedro Pérez",
         "el cargo tras «en su carácter de»"),
        ("JUAN PÉREZ, QUIEN TAMBIÉN SE OSTENTA COMO JUAN PÉREZ GARCÍA", {},
         "Juan Pérez, quien también se ostenta como Juan Pérez García", "«quien también se ostenta» y «SE»"),
        ("ANA LUISA GÓMEZ PÉREZ APODERADA DE ROSA Y ELENA, TODAS DE APELLIDO RUIZ, ALBACEAS DE LA "
         "SUCESIÓN A BIENES DE JOSÉ RUIZ", {},
         "Ana Luisa Gómez Pérez apoderada de Rosa y Elena, todas de apellido Ruiz, albaceas de la sucesión "
         "a bienes de José Ruiz", "Q 229: «apoderada de» y «albaceas de la sucesión a bienes de»"),
        ("MARÍA LÓPEZ, HEREDERA ÚNICA Y ALBACEA DE LA SUCESIÓN A BIENES DE JUAN LÓPEZ", {},
         "María López, heredera única y albacea de la sucesión a bienes de Juan López", "«heredera única»"),
        ("JUAN PÉREZ Y/O JUAN PÉREZ GARCÍA", {}, "Juan Pérez y/o Juan Pérez García", "«y/o»"),
        ("JUAN PÉREZ GÓMEZ Y MARÍA LÓPEZ DÍAZ, AMBOS POR PROPIO DERECHO", {},
         "Juan Pérez Gómez y María López Díaz, ambos por propio derecho", "AD 274: «ambos por propio derecho»"),
        ("SUCESIÓN A BIENES DE JOSÉ GARCÍA RUIZ A TRAVÉS DE SU ALBACEA MARÍA LÓPEZ DÍAZ", {"con_articulo": True},
         "la Sucesión a Bienes de José García Ruiz, a través de su albacea María López Díaz",
         "AD 335: la coma que falta antes de «a través de su albacea»"),
        ("JUAN PÉREZ GÓMEZ, POR PROPIO DERECHO (DEMANDADOS PRINCIPALES Y APELANTES)", {},
         "Juan Pérez Gómez, por propio derecho", "AD 552: la etiqueta de rol compuesta"),
        ("ANA RUIZ SOTO (ACTORA EN EL JUICIO NATURAL)", {}, "Ana Ruiz Soto", "«(ACTORA EN EL JUICIO NATURAL)»"),
        ("FINANCIERA ÁGIL DEL CENTRO, SOCIEDAD ANÓNIMA DE CAPITAL VARIABLE, SOFOM, ENTIDAD NO REGULADA", {},
         "Financiera Ágil del Centro, Sociedad Anónima de Capital Variable, SOFOM, Entidad no Regulada",
         "AD 552: SOFOM en siglas y «no» como partícula"),
        ("JUAN PÉREZ, EN SU CARÁCTER DE PRESIDENTE MUNICIPAL DE QUERÉTARO", {},
         "Juan Pérez, en su carácter de Presidente Municipal de Querétaro",
         "un cargo público tras «en su carácter de» conserva su mayúscula")):
    ok(_np_kw(crudo, **kw) == prosa, f"{caso}: «{crudo[:55]}» → «{prosa[:70]}»")
ok(rpt._sin_rol("ANA (QUEJOSA Y RECURRENTE)") == ("ANA", "quejoso") and rpt._sin_rol("ANA (LA)")[0] == "ANA (LA)",
   "la etiqueta compuesta da su carácter; un paréntesis sin rol de verdad se queda")
o = rpt.componer("amparo_directo", _mezclar(AD, {
    "promovente": "SUCESIÓN A BIENES DE JOSÉ GARCÍA RUIZ A TRAVÉS DE SU ALBACEA MARÍA LÓPEZ DÍAZ",
    "representante": None, "figura_representante": None}), DATOS)
ok("la Sucesión a Bienes de José García Ruiz, a través de su albacea María López Díaz, promovió juicio" in
   _res(o, "Presentación"), "AD 335: el inciso de la representación se cierra antes del verbo")
o = rpt.componer("amparo_directo", _mezclar(AD, {
    "promovente": "JUAN PÉREZ GÓMEZ Y MARÍA LÓPEZ DÍAZ, AMBOS POR PROPIO DERECHO",
    "representante": None, "figura_representante": None}), DATOS)
ok("María López Díaz, ambos por propio derecho, promovieron juicio" in _res(o, "Presentación"),
   "AD 274 en plural: «ambos por propio derecho,» en minúscula y con su coma de cierre")

print("\n44 · _cierra_inciso: LA RAZÓN SOCIAL AL FINAL Y EL PUNTO Y COMA (AR 222, RF 2/2025)")
for suj, fin in (("María de la Luz Pérez, por sí y como representante de la moral Constructora del "
                  "Bajío, S.A. de C.V.", "S.A. de C.V.,"),
                 ("Juan Pérez López y Ana Ruiz Soto; por propio derecho", "por propio derecho,"),
                 ("Juan Pérez Gómez y María López Díaz, Ambos por propio derecho", "propio derecho,"),
                 ("Banco del Centro, S.A.", "Banco del Centro, S.A."),
                 ("Juan Pérez, por propio derecho, y María López", "y María López")):
    ok(rpt._cierra_inciso(suj).endswith(fin) and (fin.endswith(",") or not rpt._cierra_inciso(suj).endswith(",")),
       f"«{suj[:60]}» → «…{fin}»")
o = rpt.componer("amparo_revision", _mezclar(AR, {
    "quejoso": "María de la Luz Pérez, por sí y como representante de la moral Constructora del Bajío, "
               "S.A. de C.V."}), DATOS)
ok("Constructora del Bajío, S.A. de C.V., promovió juicio de amparo indirecto" in
   _res(o, "Presentación de la demanda"), "AR 222: la coma de cierre tras la razón social del inciso")

print("\n45 · EL PONENTE «EN FUNCIONES», EN APOSICIÓN (AR 239 y 222/2025)")
for crudo, tit, prosa in (
        ("Bertha Martínez Vega", "Secretaria en funciones de Magistrada",
         "de Bertha Martínez Vega, secretaria en funciones de Magistrada"),
        ("Secretaria en funciones de Magistrada Bertha Martínez Vega", "",
         "de Bertha Martínez Vega, secretaria en funciones de Magistrada"),
        ("Bertha Martínez Vega, Secretaria en funciones de Magistrada", "",
         "de Bertha Martínez Vega, secretaria en funciones de Magistrada"),
        ("magistrado Enrique Villanueva Ruiz", "", "del Magistrado Enrique Villanueva Ruiz"),
        ("licenciada Bertha Martínez Vega", "Secretaria en funciones de Magistrada",
         "de la licenciada Bertha Martínez Vega, secretaria en funciones de Magistrada"),
        ("Secretaria en Funciones de Magistrada Bertha Martínez Vega", "",
         "de Bertha Martínez Vega, secretaria en funciones de Magistrada"),
        ("Bertha Martínez Vega", "secretaria en funciones de magistrada",
         "de Bertha Martínez Vega, secretaria en funciones de Magistrada")):
    try:
        _got = rpt._ponente_en_prosa(crudo, tit)
    except TypeError:
        _got = ""
    ok(_got == prosa, f"«{crudo}»{' + «' + tit + '»' if tit else ''} → «{prosa}»")

print("\n46 · QUEJA: EL SENTIDO CON MINÚSCULA A MEDIA FRASE (Q 335/2025)")
o = rpt.componer("queja", _mezclar(Q, {"acto": {"sentido": "Dejó sin efectos la suspensión del acto "
                                                          "reclamado"}}), DATOS)
ok("en el que dejó sin efectos la suspensión del acto reclamado." in _res(o, "Interposición") and
   "Dejó" not in _todo(o), "«en el que dejó sin efectos…», no «en el que Dejó…»")

print("\n47 · RF: NEGATIVA FICTA SIN «SOLICITUD DE SOLICITUD» (RF 4/2025)")
o = rpt.componer("revision_fiscal", _mezclar(RF, {"resolucion_impugnada": {
    "clase": "negativa_ficta", "oficio": "", "fecha": "",
    "materia": "solicitud de incorporación al sistema de jubilación previsto en el artículo Décimo "
               "Transitorio de la Ley del ISSSTE",
    "autoridad": "Subdelegado de Prestaciones Económicas"}}), DATOS)
ok("la resolución negativa ficta recaída a su solicitud de incorporación al sistema de jubilación" in
   _res(o, "Trámite del juicio") and "solicitud de solicitud" not in _todo(o),
   "RF 4: la materia que ya dice «solicitud de…» no se duplica")

print("\n48 · RF: EL SENTIDO QUE TRAE MÁS QUE EL CATÁLOGO (RF 6/2026)")
o = rpt.componer("revision_fiscal", _mezclar(RF, {"acto": {
    "sentido": "declaró la nulidad de la resolución impugnada para efectos y reconoció el derecho "
               "subjetivo al incremento de la cuota pensionaria y al pago retroactivo de las diferencias"}}),
    DATOS)
ok("en la que declaró la nulidad de la resolución impugnada, para determinados efectos, y reconoció el "
   "derecho subjetivo al incremento de la cuota pensionaria y al pago retroactivo de las diferencias." in
   _res(o, "Trámite del juicio") and not any(a.startswith("EL SENTIDO TRAE MÁS") for a in o["avisos"]),
   "RF 6/2026: el reconocimiento del derecho subjetivo (cola copiada del papel) se conserva sin aviso (integración)")
o = rpt.componer("revision_fiscal", _mezclar(RF, {"acto": {
    "sentido": "declaró la nulidad de la resolución impugnada para efectos y condenó a la autoridad al pago de daños"}}),
    DATOS)
ok(any(a.startswith("EL SENTIDO TRAE MÁS") for a in o["avisos"]),
   "…y lo que sobra y no es el reconocimiento de un derecho sigue con aviso")
o = rpt.componer("revision_fiscal", _mezclar(RF, {"acto": {"sentido": "", "sentido_clave": "nulidad_derecho"}}),
                 DATOS)
ok("para determinados efectos, y reconoció" in _res(o, "Trámite del juicio"),
   "la clave «nulidad_derecho» tiene su frase")

print("\n49 · EL NOMBRE ABREVIADO SÓLO CON REMISIÓN (RF 7/2025)")
o = rpt.componer("revision_fiscal", _mezclar(RF, {"actora": "1) JUAN PÉREZ LÓPEZ, 2) ANA RUIZ SOTO Y "
                                                            "3) LUIS GIL MORA"}), DATOS)
ok(not any("ABREVIADO" in a for a in o["avisos"]) and
   _res(o, "Trámite del juicio").startswith("Juan Pérez López, Ana Ruiz Soto y Luis Gil Mora demandaron"),
   "la lista numerada COMPLETA no está abreviada: sin aviso")
_larga = ", ".join(f"{i}) NOMBRE{i} APELLIDO PRUEBA" for i in range(1, 66)) + " Y OTROS"
o = rpt.componer("revision_fiscal", _mezclar(RF, {"actora": _larga}), DATOS)
_ab = [a for a in o["avisos"] if a.startswith("EL NOMBRE VIENE ABREVIADO")]
ok(len(_ab) == 1 and len(_ab[0]) < 800, "con «y otros» sí avisa, con la cita recortada")

print("\n50 · QUEJA FR. II: EL REGISTRO CON LA FECHA DEL INFORME VA EN HUECO (Q 335/2025)")
_qii = _mezclar(Q, {"fraccion_97": "II", "inciso_97": "b",
                    "acto": {"organo": "Segunda Sala Civil del Tribunal Superior de Justicia del Estado de "
                                       "Querétaro", "sentido": "negó la suspensión"},
                    "registro": {"fecha": "2026-03-13"}, "informe_101": {"fecha": "2026-03-13"},
                    "admision": {"fecha": "2026-03-13"}})
o = rpt.componer("queja", _qii, DATOS)
_t = _res(o, "Trámite del recurso")
ok(_t.startswith(f"Por auto de Presidencia de {H}, este Tribunal Colegiado registró") and
   _t.count("trece de marzo") == 1 and
   any("(registro.fecha)" in a and "misma fecha" in a for a in o["avisos"]),
   "registro = informe = admisión: el registro en hueco, la fecha una sola vez y el aviso dice por qué")
o = rpt.componer("queja", _mezclar(_qii, {"informe_101": {"fecha": "2026-03-20"},
                                          "admision": {"fecha": "2026-03-13"}}), DATOS)
ok(_res(o, "Trámite del recurso").startswith(f"Por auto de Presidencia de {H},"),
   "registro = admisión (con otro informe): también en hueco")

print("\n51 · RF: EL AUTORIZADO DE LA AUTORIDAD, FUERA DEL RESULTANDO COMO DE LA LEGITIMACIÓN (RF 4)")
o = rpt.componer("revision_fiscal", _mezclar(RF, {
    "figura_representante": "autorizado en términos del artículo 5o. de la Ley Federal de "
                            "Procedimiento Contencioso Administrativo", "representante": "Pedro Gil Mora"}),
    DATOS)
ok("autorizado" not in _res(o, "Interposición") and "Pedro Gil Mora" not in _res(o, "Interposición") and
   sum(bool(re.search(r"(?i)autorizad", a)) for a in o["avisos"]) == 1 and
   any("Se omitió del resultando y de la legitimación" in a for a in o["avisos"]),
   "RF 4: un solo trato para el autorizado (fuera de los dos, con UN aviso)")

print("\n52 · RF: LA SALA QUE CAMBIÓ DE NOMBRE (RF 21 y 49/2025)")
o = rpt.componer("revision_fiscal", _mezclar(RF, {
    "sala": "Sala Regional en Querétaro del Tribunal Federal de Justicia Administrativa",
    "acto": {"organo": "SALA REGIONAL DEL CENTRO II, AHORA SALA REGIONAL EN QUERÉTARO DEL TRIBUNAL "
                       "FEDERAL DE JUSTICIA ADMINISTRATIVA"}}), DATOS)
ok("dictada por la Sala Regional del Centro II, ahora Sala Regional en Querétaro del Tribunal Federal de "
   "Justicia Administrativa, en el juicio" in o["visto"],
   "RF 21: el órgano del acto con «ahora» manda sobre ficha.sala")
o = rpt.componer("revision_fiscal", _mezclar(RF, {
    "sala": "Sala Regional en Querétaro del Tribunal Federal de Justicia Administrativa",
    "acto": {"organo": "Sala Regional del Centro II, del Tribunal Federal de Justicia Administrativa, ahora "
                       "Sala Regional en Querétaro del Tribunal Federal de Justicia Administrativa, con sede "
                       "en esta ciudad"}}), DATOS)
ok(o["datos_extra"]["sala"] == "Sala Regional del Centro II, ahora Sala Regional en Querétaro del Tribunal "
   "Federal de Justicia Administrativa" and "esta ciudad" not in _todo(o),
   "RF 21 (fichas3): el Tribunal una vez y sin «con sede en esta ciudad», que en la carátula no remite a nada")


print("\n53 · LAS EXPRESIONES NUEVAS NO SE ATASCAN (backtracking)")
import time as _time
_t0 = _time.time()
for _txt in ("JUAN (" + "LA " * 300 + "ZZZ)", "JUAN (" + "LA, " * 300 + "ZZZ)",
             "JUAN (" + ", Y " * 300 + "ZZZ)"):
    rpt._sin_rol(_txt)
    _np(_txt)
ok(_time.time() - _t0 < 1.0, "un paréntesis de 300 palabras que no es de rol se resuelve en menos de un "
                             "segundo (con «\\s*(?:,|y)?\\s*» tardaba 3,5 s con 22)")


# ═══════════════════════════════════════════════════════════════════════════
# SEXTA RONDA (3-oct-2026): lo que David respondió y aprobó para todos
# ═══════════════════════════════════════════════════════════════════════════
print("\n54 · C3 · LA QUEJA FR. I ABRE CON LA DEMANDA CUANDO LA FICHA LA TRAE (Q 172, 24, 261 y 300)")
# David: «lo más práctico… que el secretario no tenga que modificar». 6 de 8
# quejas del banco abren con la demanda («Demanda de amparo.»); la mecánica es
# la del primer resultando de la revisión.
_DEM54 = {"fecha": "2026-02-27",
          "autoridades": ["Juez Noveno de Primera Instancia Familiar del Distrito Judicial de Querétaro"],
          "actos": ["la determinación dictada el dos de febrero de dos mil veintiséis en el juicio "
                    "ordinario civil 515/2022"]}
_Q54 = _mezclar(Q, {"quejoso": "José Eduardo Aguilar Durán", "demanda": _DEM54})
_f54 = copy.deepcopy(_Q54)
o = rpt.componer("queja", _f54, DATOS)
ok(_f54 == _Q54, "C3: sigue siendo puro (la ficha con la demanda no se toca)")
ok(_titulos(o) == ["Demanda de amparo.", "Interposición del recurso de queja.", "Trámite del recurso.",
                   "Turno del asunto."],
   "C3: «Demanda de amparo.» primero; después, como siempre (interposición, trámite, turno)")
ok(o["resultandos"][0]["texto"] == (
    "Por escrito presentado el veintisiete de febrero de dos mil veintiséis, José Eduardo Aguilar Durán "
    "promovió juicio de amparo indirecto en contra de la autoridad y el acto que a continuación se "
    "señalan:\nAUTORIDAD RESPONSABLE:\nJuez Noveno de Primera Instancia Familiar del Distrito Judicial de "
    "Querétaro\nACTO RECLAMADO:\nLa determinación dictada el dos de febrero de dos mil veintiséis en el "
    "juicio ordinario civil 515/2022."),
   "C3: la fórmula del contrato, con la autoridad y el acto en renglones (singular)")
ok(H not in _todo(o) and not o["avisos"] and not _limpio(_todo(o)),
   f"C3: con la demanda completa, sin huecos, sin avisos y sin evasivas {o['avisos'][:1]}")
ok(o["datos_extra"]["actos"] == ["la determinación dictada el dos de febrero de dos mil veintiséis en el "
                                 "juicio ordinario civil 515/2022"]
   and o["datos_extra"]["resultando_demanda"] == "primero",
   "C3: datos_extra lleva los actos de la demanda y el resultando que la copia")
_base54 = rpt.componer("queja", Q, DATOS)
ok(o["visto"] == _base54["visto"] and o["resultandos"][1:] == _base54["resultandos"],
   "C3: el V I S T O y los demás resultandos no cambian con la demanda")
# LA MISMA MECÁNICA QUE LA REVISIÓN: con los mismos datos, el mismo texto.
_ar54 = rpt.componer("amparo_revision", _mezclar(AR, {"promovente": "José Eduardo Aguilar Durán",
                                                      "representante": None, "figura_representante": None,
                                                      "demanda": _DEM54}), DATOS)
ok(_ar54["resultandos"][0]["texto"] == o["resultandos"][0]["texto"],
   "C3: el texto de la demanda es EL MISMO que el primer resultando del amparo en revisión (una sola función)")

# Sin fecha: «{Quejoso} promovió…», sin hueco ni aviso (contexto, no requisito).
o = rpt.componer("queja", _mezclar(_Q54, {"demanda": {"fecha": ""}}), DATOS)
_t54 = _res(o, "Demanda de amparo")
ok(_t54.startswith("José Eduardo Aguilar Durán promovió juicio de amparo indirecto en contra de la "
                   "autoridad y el acto") and H not in _t54,
   "C3 sin fecha: abre con quien promovió, sin hueco")
ok(not any("(demanda.fecha)" in a for a in o["avisos"]) and not o["avisos"],
   "C3 sin fecha: ningún aviso (en el AR sí lo hay; en la queja la demanda es contexto)")

# Varias autoridades y varios actos; quejosos en plural.
o = rpt.componer("queja", _mezclar(_Q54, {
    "quejoso": "María López Ruiz y Juan Pérez Soto", "caracter": "tercero",
    "promovente": "Banco Nacional de México, S.A.",
    "demanda": {"autoridades": ["Sala Familiar del Tribunal Superior de Justicia del Estado de Querétaro",
                                "JUEZ CUARTO DE PRIMERA INSTANCIA FAMILIAR DEL DISTRITO JUDICIAL DE QUERÉTARO"],
                "actos": ["La sentencia de nueve de junio de dos mil veinticinco.", "Su ejecución"]}}), DATOS)
_t54 = _res(o, "Demanda de amparo")
ok(_t54.startswith("Por escrito presentado el veintisiete de febrero de dos mil veintiséis, María López "
                   "Ruiz y Juan Pérez Soto promovieron juicio de amparo indirecto en contra de las "
                   "autoridades y los actos que a continuación se señalan:"),
   "C3: quejosos en plural («promovieron») y «las autoridades y los actos»")
ok("\nAUTORIDADES RESPONSABLES:\nSala Familiar del Tribunal Superior de Justicia del Estado de Querétaro\n"
   "Juez Cuarto de Primera Instancia Familiar del Distrito Judicial de Querétaro\nACTOS RECLAMADOS:\n"
   "La sentencia de nueve de junio de dos mil veinticinco.\nSu ejecución." in _t54,
   "C3: una autoridad y un acto por renglón; las versales, en prosa; cada acto con su punto")
ok("Banco Nacional de México" not in _t54 and "Banco Nacional de México" in _res(o, "Interposición"),
   "C3: si recurre otra parte, la demanda nombra a la quejosa (ficha.quejoso), no a quien recurre")
ok(o["datos_extra"]["actos"] == ["La sentencia de nueve de junio de dos mil veinticinco", "Su ejecución"],
   "C3: los actos de datos_extra, sin el punto final")

# Sólo las autoridades: el acto en hueco, con su aviso (la otra mitad está a la vista).
o = rpt.componer("queja", _mezclar(_Q54, {"demanda": {"actos": []}}), DATOS)
ok("ACTO RECLAMADO:\n" + H in _res(o, "Demanda de amparo")
   and any("(demanda.actos)" in a and "«Demanda de amparo»" in a and "auto recurrido" in a
           for a in o["avisos"]),
   "C3 sólo con autoridades: el acto en hueco y el aviso que dice dónde buscarlo")
# Sólo los actos: la autoridad en hueco, con su aviso.
o = rpt.componer("queja", _mezclar(_Q54, {"demanda": {"autoridades": []}}), DATOS)
ok("AUTORIDAD RESPONSABLE:\n" + H in _res(o, "Demanda de amparo")
   and any("(demanda.autoridades)" in a for a in o["avisos"]),
   "C3 sólo con actos: la autoridad en hueco y su aviso")

# Sin demanda: la queja abre como hoy, y sin aviso nuevo.
for _nom54, _cam54 in (("sin la clave", {}), ("demanda vacía", {"demanda": {}}),
                       ("demanda sólo con huecos", {"demanda": {"fecha": "", "autoridades": [H],
                                                                "actos": ["*********"]}}),
                       ("sólo la fecha", {"demanda": {"fecha": "2026-02-27"}})):
    o = rpt.componer("queja", _mezclar(_mezclar(Q, {"quejoso": "José Eduardo Aguilar Durán"}), _cam54), DATOS)
    ok(_titulos(o)[0] == "Interposición del recurso de queja." and o["avisos"] == _base54["avisos"]
       and o["resultandos"] == _base54["resultandos"] and o["datos_extra"]["resultando_demanda"] == ""
       and o["datos_extra"]["actos"] == [],
       f"C3 {_nom54}: abre con la interposición, igual que hoy y sin aviso nuevo")

# Sin quejoso legible: el resultando no se escribe (ni hueco ni aviso).
_Q54t = _mezclar(Q, {"caracter": "tercero", "promovente": "Banco Nacional de México, S.A."})
_sin54 = rpt.componer("queja", _Q54t, DATOS)
for _qj in ("", H, "**** ******* ******"):
    o = rpt.componer("queja", _mezclar(_Q54t, {"quejoso": _qj, "demanda": _DEM54}), DATOS)
    ok(_titulos(o) == _titulos(_sin54) and o["avisos"] == _sin54["avisos"],
       f"C3 sin quejoso legible («{_qj}»): no hay «Demanda de amparo.» ni aviso nuevo")

# La fracción II (y la que no consta) no lo lleva.
o = rpt.componer("queja", _mezclar(_Q54, {"fraccion_97": "II", "inciso_97": "b",
                                          "informe_101": {"fecha": "2026-03-18"},
                                          "registro": {"fecha": "2026-03-13"},
                                          "admision": {"fecha": "2026-03-18"}, "turno": {"fecha": "2026-03-24"},
                                          "acto": {"organo": "Primera Sala Civil del Tribunal Superior de "
                                                             "Justicia del Estado de Querétaro"}}), DATOS)
ok("Demanda de amparo." not in _titulos(o) and o["datos_extra"]["resultando_demanda"] == "",
   "C3: la queja de la fracción II no lleva el resultando de la demanda")
o = rpt.componer("queja", _mezclar(_Q54, {"fraccion_97": "", "acto": {"organo": "Juzgado Mixto de Cadereyta"}}),
                 DATOS)
ok("Demanda de amparo." not in _titulos(o), "C3: sin la fracción I cierta, tampoco (no se supone la vía)")

# El propio juzgado del auto, fuera de las autoridades (respaldo de la ficha).
o = rpt.componer("queja", _mezclar(_Q54, {"demanda": {"autoridades": [
    "Juez Séptimo de Distrito en el Estado de Querétaro",
    "Juez Noveno de Primera Instancia Familiar del Distrito Judicial de Querétaro"]}}), DATOS)
_t54 = _res(o, "Demanda de amparo")
ok("Séptimo de Distrito" not in _t54 and "AUTORIDAD RESPONSABLE:\nJuez Noveno" in _t54
   and any("ESTABA ENTRE LAS AUTORIDADES" in a and "(demanda.autoridades" in a for a in o["avisos"]),
   "C3: el juzgado que dictó el auto recurrido sale de las autoridades, con aviso")
o = rpt.componer("queja", _mezclar(_Q54, {"demanda": {"autoridades": [
    "Juez Séptimo de Distrito en el Estado de Querétaro"], "actos": []}}), DATOS)
ok("Demanda de amparo." not in _titulos(o) and o["avisos"] == _base54["avisos"],
   "C3: si la única «autoridad» era el propio juzgado y no hay actos, no queda demanda que contar "
   "(ni aviso de un resultando que no se escribió)")
o = rpt.componer("queja", _mezclar(_Q54t, {"quejoso": "", "demanda": {"autoridades": [
    "Juez Séptimo de Distrito en el Estado de Querétaro", "Juez Noveno Familiar"]}}), DATOS)
ok(o["avisos"] == _sin54["avisos"], "C3: sin quejoso legible tampoco se avisa del juzgado quitado")

# La sucesión y el quejoso que recurre: el nombre como en el resto del documento.
o = rpt.componer("queja", _mezclar(Q, {"promovente": "SUCESIÓN TESTAMENTARIA A BIENES DE JOSÉ GARCÍA RUIZ",
                                       "demanda": _DEM54}), DATOS)
ok(_res(o, "Demanda de amparo").startswith("Por escrito presentado el veintisiete de febrero de dos mil "
                                           "veintiséis, la Sucesión Testamentaria a Bienes de José García "
                                           "Ruiz promovió"),
   "C3: el quejoso que recurre es el de la demanda; en prosa y con su artículo")


print("\n55 · C5 · EL AR PASA LOS ACTOS Y EL RESULTANDO QUE COPIA LA DEMANDA")
o = rpt.componer("amparo_revision", AR, DATOS)
_ex55 = o["datos_extra"]
ok(_ex55["actos"] == ["la resolución dictada el tres de marzo de dos mil veinticinco en el toca familiar "
                      "112/2025"] and _ex55["resultando_demanda"] == "primero"
   and _titulos(o)[0] == "Presentación de la demanda de amparo indirecto.",
   "C5: datos_extra[actos] limpios y resultando_demanda «primero» (el que copia la demanda)")
_p55 = ta.con_actos_del_amparo(ta.RAMAS_REVISION["confirma_niega"]["puntos"][1], _ex55["actos"],
                               _ex55["autoridades"], _ex55["resultando_demanda"], _ex55["plural_quejoso"],
                               rama="confirma_niega")
ok("respecto del acto que reclamó del Magistrado Presidente de la Sala Familiar del Tribunal Superior de "
   "Justicia del Estado de Querétaro, consistente en la resolución dictada el tres de marzo de dos mil "
   "veinticinco en el toca familiar 112/2025" in _p55,
   "C5: con esos datos, el punto que confirma nombra acto y autoridad (tipos_asunto.punto_del_amparo)")
o = rpt.componer("amparo_revision", _mezclar(AR, {"demanda": {"actos": [
    "La cancelación del crédito.", "*********", "  Su ejecución; ", "La cancelación del crédito"]}}), DATOS)
ok(o["datos_extra"]["actos"] == ["La cancelación del crédito", "Su ejecución"],
   "C5: sin huecos, sin punto final y sin repetir")
o = rpt.componer("amparo_revision", _mezclar(AR, {"demanda": {"actos": []}}), DATOS)
ok(o["datos_extra"]["actos"] == [] and o["datos_extra"]["resultando_demanda"] == "primero"
   and "ACTO RECLAMADO:\n" + H in _res(o, "Presentación"),
   "C5: sin actos, lista vacía (el punto usa la remisión genérica) y el resultando con su hueco")
_ex55b = rpt.componer("amparo_revision", _mezclar(AR, {"promovente": "María López Ruiz y Juan Pérez Soto",
                                                       "representante": None, "figura_representante": None,
                                                       "demanda": {"actos": ["El embargo.", "Su ejecución."]}}),
                      DATOS)["datos_extra"]
ok("los actos que reclamaron del Magistrado Presidente" in ta.punto_del_amparo(
    _ex55b["actos"], _ex55b["autoridades"], _ex55b["resultando_demanda"], _ex55b["plural_quejoso"])
   and "resultando primero de esta ejecutoria" in ta.punto_del_amparo(
       _ex55b["actos"], _ex55b["autoridades"], _ex55b["resultando_demanda"], _ex55b["plural_quejoso"]),
   "C5: dos actos y quejosos en plural → «reclamaron…, precisados en el resultando primero»")
for _t55, _f55 in (("amparo_directo", AD), ("revision_fiscal", RF), ("queja", Q)):
    _e55 = rpt.componer(_t55, _f55, DATOS)["datos_extra"]
    ok(_e55["actos"] == [] and _e55["resultando_demanda"] == "",
       f"C5: {_t55} sin demanda que copiar: actos [] y resultando_demanda «»")
ok(rpt.componer("queja", None, None)["datos_extra"]["actos"] == [] and
   isinstance(rpt.componer("amparo_revision", None, None)["datos_extra"]["actos"], list),
   "C5: sin ficha, las listas siguen siendo listas")


print("\n56 · C6 · LOS RELACIONADOS QUE MARCÓ EL SECRETARIO, EN EL V I S T O Y EN datos_extra")
_R1 = [{"tipo": "amparo_directo", "numero": "452/2025", "estado": "misma_sesion"}]
_R2 = [{"tipo": "amparo_directo", "numero": "452/2025", "estado": "misma_sesion"},
       {"tipo": "revision_fiscal", "numero": "33/2024", "estado": "resuelto"}]
# (tipo, ficha, lo que el V I S T O dice antes y después de la inserción)
_CASOS56 = (
    ("amparo_directo", _mezclar(AD, {"relacionados": [{"tipo": "revision_fiscal", "numero": "33/2024",
                                                       "estado": "misma_sesion"}]}),
     "para resolver el juicio de amparo directo civil 174/2026, relacionado con el recurso de revisión "
     "fiscal 33/2024, promovido por Ruth Gabriela"),
    ("amparo_revision", _mezclar(AR, {"relacionados": _R1}),
     "para resolver el recurso de revisión 513/2025, relacionado con el amparo directo civil 452/2025, "
     "interpuesto por Impulsora"),
    ("queja", _mezclar(Q, {"relacionados": [{"tipo": "amparo_revision", "numero": "298/2025"}]}),
     "para resolver el recurso de queja civil 150/2026, relacionado con el amparo en revisión civil "
     "298/2025, interpuesto por José Eduardo"),
    ("revision_fiscal", _mezclar(RF, {"relacionados": [{"tipo": "amparo_directo", "numero": "456/2025",
                                                        "estado": "misma_sesion"}]}),
     "para resolver el recurso de revisión fiscal número 62/2025, relacionado con el amparo directo "
     "administrativo 456/2025, interpuesto por la parte citada al rubro"),
)
for _t56, _f56, _esp in _CASOS56:
    _copia = copy.deepcopy(_f56)
    o = rpt.componer(_t56, _copia, DATOS)
    ok(o["visto"].startswith(_esp), f"C6 · {_t56} con 1 relacionado: «, relacionado con …» tras el número")
    ok(len(o["datos_extra"]["relacionados"]) == 1 and o["datos_extra"]["relacionados"][0]["numero"]
       in o["visto"] and o["datos_extra"]["relacionados"][0]["estado"] == "misma_sesion",
       f"C6 · {_t56}: datos_extra[relacionados] con la lista normalizada (estado por omisión: misma sesión)")
    ok(_copia == _f56 and not o["avisos"] and H not in _todo(o),
       f"C6 · {_t56}: puro, sin avisos y sin huecos")
    _sin = rpt.componer(_t56, {k: v for k, v in _f56.items() if k != "relacionados"}, DATOS)
    _ins = ", relacionado con " + ta.relacionados_en_prosa(o["datos_extra"]["relacionados"],
                                                          o["datos_extra"]["materia"])
    ok(_sin["datos_extra"]["relacionados"] == [] and "relacionado" not in _sin["visto"]
       and o["visto"].replace(_ins, "") == _sin["visto"] and o["resultandos"] == _sin["resultandos"],
       f"C6 · {_t56} sin relacionados: datos_extra [] y el V I S T O de siempre (sólo cambia la inserción)")
    o2 = rpt.componer(_t56, _mezclar(_f56, {"relacionados": _R2 if _t56 not in ("amparo_directo",
                                                                                 "revision_fiscal") else
                                            [{"tipo": "amparo_revision", "numero": "12/2026"},
                                             {"tipo": "queja", "numero": "24/2026", "estado": "resuelto"}]}),
                      DATOS)
    ok(len(o2["datos_extra"]["relacionados"]) == 2 and " y con el " in o2["visto"]
       and o2["visto"].count("relacionado con") == 1,
       f"C6 · {_t56} con 2 relacionados: «relacionado con el X y con el Y», una sola vez")
o = rpt.componer("amparo_revision", _mezclar(AR, {"relacionados": _R2}), DATOS)
ok(o["visto"].startswith("para resolver el recurso de revisión 513/2025, relacionado con el amparo directo "
                         "civil 452/2025 y con el recurso de revisión fiscal 33/2024, interpuesto por"),
   "C6: dos relacionados en el AR, la revisión fiscal sin materia")
o = rpt.componer("queja", _mezclar(Q, {"materia": "administrativa", "relacionados": [
    {"tipo": "amparo_directo", "numero": "452/2025"}, {"tipo": "queja", "numero": "24/2026"}]}), DATOS)
ok(o["visto"].startswith("para resolver el recurso de queja administrativa 150/2026, relacionado con el "
                         "amparo directo administrativo 452/2025 y con el recurso de queja administrativa "
                         "24/2026, interpuesto por"),
   "C6: la materia concuerda con cada nombre («amparo… administrativo», «queja administrativa»)")
o = rpt.componer("amparo_directo", _mezclar(AD, {"materia": "mercantil", "relacionados": [
    {"tipo": "amparo_revision", "numero": "7/2026"}]}), DATOS)
ok("civil 174/2026, relacionado con el amparo en revisión civil 7/2026, promovido" in o["visto"],
   "C6: lo mercantil va como civil, igual que el asunto")
# Lo que no se escribe: el propio asunto, lo inválido, lo que no marcó el secretario.
o = rpt.componer("queja", _mezclar(Q, {"relacionados": [{"tipo": "queja", "numero": "150/2026"}]}), DATOS)
ok("relacionado" not in o["visto"] and o["datos_extra"]["relacionados"] == [],
   "C6: el propio asunto no se relaciona consigo mismo")
o = rpt.componer("queja", _mezclar(Q, {"relacionados": [{"tipo": "amparo_lunar", "numero": "1/2025"},
                                                        {"tipo": "queja", "numero": "veinte"}, "texto",
                                                        {"tipo": "amparo_directo", "numero": "9/2025"},
                                                        {"tipo": "amparo_directo", "numero": "9/2025"}]}),
                 DATOS)
ok(o["datos_extra"]["relacionados"] == [{"tipo": "amparo_directo", "numero": "9/2025",
                                         "estado": "misma_sesion"}]
   and "relacionado con el amparo directo civil 9/2025, interpuesto" in o["visto"],
   "C6: tipo desconocido, número ilegible o repetido, fuera; queda el válido")
o = rpt.componer("queja", _mezclar(Q, {"relacionados": _R1, "fuentes": {"relacionados": "modelo"}}), DATOS)
ok("relacionado" not in o["visto"] and o["datos_extra"]["relacionados"] == [],
   "C6: nunca conexidad en automático: lo que no marcó el secretario no se escribe")
o = rpt.componer("queja", _mezclar(Q, {"relacionados": _R1, "fuentes": {"relacionados": "secretario"}}), DATOS)
ok("relacionado con el amparo directo civil 452/2025" in o["visto"],
   "C6: con la fuente «secretario», sí")
o = rpt.componer("queja", Q, _mezclar(DATOS, {"relacionados": _R1}))
ok("relacionado" not in o["visto"] and o["datos_extra"]["relacionados"] == [],
   "C6: los datos del encargo no traen relacionados: sólo la ficha")
o = rpt.componer("queja", _mezclar(Q, {"relacionados": [{"tipo": "amparo_directo", "numero": f"{i}/2025"}
                                                        for i in range(1, 7)]}), DATOS)
ok(len(o["datos_extra"]["relacionados"]) == 4, "C6: como mucho 4")
ok(o["visto"].count("relacionado con") == 1 and _limpio(o["visto"]) == [],
   "C6: con 4 relacionados el V I S T O sigue limpio")
for _t56 in rpt.TIPOS:
    _e56 = rpt.componer(_t56, None, None)["datos_extra"]
    ok(_e56.get("relacionados") == [], f"C6 · {_t56} sin ficha: datos_extra[relacionados] = []")
# LA MATERIA DE LOS RELACIONADOS, IGUAL QUE EL ENCABEZADO DE ESE TIPO (integración,
# 3-oct-2026): el amparo en revisión en femenino, como «AMPARO EN REVISIÓN
# ADMINISTRATIVA»; la revisión fiscal sin materia en la ficha, la administrativa.
o = rpt.componer("amparo_directo", _mezclar(AD, {"materia": "administrativa", "relacionados": [
    {"tipo": "amparo_revision", "numero": "12/2026"}]}), DATOS)
ok("relacionado con el amparo en revisión administrativa 12/2026, promovido" in o["visto"],
   "integración: «amparo en revisión administrativa», como su encabezado")
o = rpt.componer("revision_fiscal", _mezclar(RF, {"materia": "", "relacionados": [
    {"tipo": "amparo_directo", "numero": "456/2025"}, {"tipo": "amparo_revision", "numero": "12/2026"}]}), DATOS)
ok("relacionado con el amparo directo administrativo 456/2025 y con el amparo en revisión administrativa "
   "12/2026, interpuesto" in o["visto"],
   "integración: la RF sin materia en la ficha nombra a sus relacionados con la administrativa")


print("\n57 · LA RESPONSABLE DEL FORMULARIO NO LE GANA AL ÓRGANO EN LA QUEJA NI EN LA RF (integración)")
# page.tsx manda en `responsable` la ordenadora leída del auto de admisión
# también en la queja y la revisión fiscal, que desde C3 y C4 ya no tienen el
# renglón del órgano en la carátula (`tipos_asunto.responsable_es_el_organo`).
_ORDEN57 = "Segunda Sala Civil del Tribunal Superior de Justicia del Estado de Querétaro"
o = rpt.componer("queja", Q, _mezclar(DATOS, {"responsable": _ORDEN57}))
ok("dictado por el Juzgado Séptimo de Distrito en el Estado de Querétaro" in o["visto"]
   and "Sala" not in o["visto"] and not any("DOS ÓRGANOS" in a for a in o["avisos"]),
   "Q fr. I con el órgano en la ficha y la ordenadora en el formulario: manda el órgano, sin el aviso de «dos "
   "órganos distintos»" + (f" — {o['avisos']}" if o["avisos"] else ""))
o = rpt.componer("queja", _mezclar(Q, {"acto": {"organo": ""}, "fraccion_97": ""}),
                 _mezclar(DATOS, {"responsable": _ORDEN57}))
ok(H in o["visto"] and "Sala" not in o["visto"] and o["datos_extra"]["organo_acto"] == ""
   and o["datos_extra"]["fraccion_97"] == ""
   and any("(acto.organo)" in a for a in o["avisos"]),
   "Q sin órgano en la ficha: la ordenadora NO ocupa su sitio (hueco con aviso) ni decide la fracción II"
   + (f" — {o['visto'][:200]} · {o['datos_extra'].get('fraccion_97')}" if "Sala" in o["visto"] else ""))
o = rpt.componer("queja", _mezclar(Q, {"acto": {"organo": ""}}),
                 _mezclar(DATOS, {"responsable": "Juez Séptimo de Distrito en el Estado de Querétaro"}))
ok("dictado por el Juzgado Séptimo de Distrito en el Estado de Querétaro" in o["visto"],
   "…pero si lo que llega en `responsable` es un juzgado de distrito, sí es el órgano de la queja")
o = rpt.componer("queja", _mezclar(Q, {"acto": {"organo": ""}, "fraccion_97": "II"}),
                 _mezclar(DATOS, {"responsable": _ORDEN57}))
ok("Segunda Sala Civil" in o["visto"] and H not in o["visto"].split("dictado por")[-1][:60],
   "…y en la queja de la fracción II la responsable del amparo directo ES quien dictó el auto")
o = rpt.componer("revision_fiscal", _mezclar(RF, {"sala": "", "acto": {"organo": ""}}),
                 _mezclar(DATOS, {"responsable": "Administración Desconcentrada Jurídica de Querétaro \"1\""}))
ok(H in o["visto"] and "Administración Desconcentrada" not in o["visto"],
   "RF sin Sala en la ficha: la autoridad que llega en `responsable` no se vuelve «… del TFJA»")
o = rpt.componer("revision_fiscal", _mezclar(RF, {"sala": "", "acto": {"organo": ""}}),
                 _mezclar(DATOS, {"responsable": "Sala Regional en Querétaro"}))
ok("la Sala Regional en Querétaro del Tribunal Federal de Justicia Administrativa" in o["visto"],
   "…pero una Sala en `responsable` sí es la Sala de la revisión fiscal")
ok(ta.responsable_es_el_organo("amparo_directo", _ORDEN57)
   and not ta.responsable_es_el_organo("amparo_revision", "Juzgado Primero de Distrito en el Estado de Querétaro")
   and not ta.responsable_es_el_organo("queja", _ORDEN57)
   and ta.responsable_es_el_organo("queja", _ORDEN57, "II")
   and ta.responsable_es_el_organo("queja", "Jueza Primera de Distrito de Amparo en Materia Penal")
   and ta.responsable_es_el_organo("revision_fiscal", "Sala Regional del Centro II")
   and not ta.responsable_es_el_organo("revision_fiscal", "Titular de la Subdelegación del IMSS")
   and not ta.responsable_es_el_organo("queja", ""),
   "tipos_asunto.responsable_es_el_organo: AD sí; AR no; Q por la forma o la fracción II; RF sólo una Sala")


print("\n58 · REVISIÓN ADVERSARIAL (3-oct-2026): LA AMPLIACIÓN EN LA QUEJA Y LA REVISIÓN FISCAL")
# Q 300/2025: el auto que provee sobre la AMPLIACIÓN la describe; lo que el
# modelo copie de ella pasa el anclaje. El resultando se escribe con aviso.
o = rpt.componer("queja", _mezclar(_Q54, {"acto": {"sentido_clave": "desecha_ampliacion"}}), DATOS)
ok(_titulos(o)[0] == "Demanda de amparo."
   and any(a.startswith("EL AUTO RECURRIDO PROVEE SOBRE UNA AMPLIACIÓN DE DEMANDA") for a in o["avisos"]),
   "Q que desecha la ampliación: «Demanda de amparo.» con el aviso de comprobar que no sea la ampliación")
o = rpt.componer("queja", _Q54, DATOS)
ok(not any("AMPLIACIÓN" in a for a in o["avisos"]), "…y sin ampliación de por medio, sin ese aviso")
# VARIAS AUTORIDADES QUE RECURREN ELLAS MISMAS (RF 33/2024): «Inconformes… interpusieron».
_SAT58 = ("Secretario de Hacienda y Crédito Público, Jefe del Servicio de Administración Tributaria y Titular de "
          "la Administración de Operación de Padrones “2” del Servicio de Administración Tributaria")
o = rpt.componer("revision_fiscal", _mezclar(RF, {"promovente": _SAT58, "autoridad_demandada": _SAT58,
                                                  "figura_representante": "Administradora Desconcentrada Jurídica "
                                                                          "de Querétaro “1”",
                                                  "representante": ""}), DATOS)
_i58 = _res(o, "Interposición del recurso de revisión fiscal")
ok(_i58.startswith("Inconformes, el Secretario de Hacienda") and "interpusieron recurso de revisión fiscal" in _i58,
   f"RF de varias autoridades: «Inconformes, …, interpusieron»: {_i58[:160]}")
o = rpt.componer("revision_fiscal", RF, DATOS)
_i58b = _res(o, "Interposición del recurso de revisión fiscal")
ok(_i58b.startswith("Inconforme, ") and "interpuso recurso" in _i58b, "una sola, en singular, como antes")
# LA SALA QUE FALTA: el aviso nombra los cuatro sitios del hueco.
o = rpt.componer("revision_fiscal", _mezclar(RF, {"sala": "", "acto": {"organo": ""}}), DATOS)
_av58 = [a for a in o["avisos"] if a.startswith("FALTA LA SALA DEL TFJA QUE DICTÓ LA SENTENCIA RECURRIDA")]
ok(len(_av58) == 1 and "«V I S T O»" in _av58[0] and "«Trámite del juicio contencioso administrativo»" in _av58[0]
   and "la competencia y el punto resolutivo" in _av58[0],
   f"RF sin Sala: el aviso nombra el V I S T O, el resultando del juicio, la competencia y el resolutivo: {_av58}")


# ═══════════════════════════════════════════════════════════════════════════
# LAS MUESTRAS PARA LA REVISIÓN HUMANA (sólo si se piden)
# ═══════════════════════════════════════════════════════════════════════════
_ORD = ["PRIMERO", "SEGUNDO", "TERCERO", "CUARTO", "QUINTO", "SEXTO", "SÉPTIMO"]
_ruta = os.environ.get("MUESTRAS_COMPOSITOR", "").strip()
if _ruta:
    partes = ["MUESTRAS DEL COMPOSITOR POR TIPO (resultandos_por_tipo.componer), "
              "3-oct-2026. Sin modelo: todo sale de la ficha de trámite de prueba de "
              "test_resultandos_por_tipo.py. El resultando de la sesión vía remota lo "
              "añade documento_generado.componer después de éstos.\n"]
    for nombre, tipo, ficha, datos in CASOS:
        o = rpt.componer(tipo, ficha, datos)
        partes.append("=" * 78)
        partes.append(f"{nombre}   [{tipo}]")
        partes.append("=" * 78)
        partes.append(ta.proemio_de(tipo)["rotulo"] + o["visto"])
        partes.append("R E S U L T A N D O:")
        for i, r in enumerate(o["resultandos"]):
            partes.append(f"{_ORD[i]}. {r['titulo']} {r['texto']}")
        if o["avisos"]:
            partes.append("AVISOS: " + " | ".join(o["avisos"]))
        if ficha.get("adhesivo"):
            rot, txt, avs = rpt.considerando_adhesivo(tipo, ficha, datos)
            partes.append(f"[CONSIDERANDO DEL ADHESIVO] {rot} {txt}")
            if avs:
                partes.append("AVISOS DEL CONSIDERANDO: " + " | ".join(avs))
            for p in (False, True, "desecha"):
                t, a = rpt.resolutivo_adhesivo(tipo, p, ficha)
                partes.append(f"[RESOLUTIVO DEL ADHESIVO · principal {p}] {t}"
                              + (f"   (aviso: {a})" if a else ""))
        partes.append("datos_extra: " + ", ".join(
            f"{k}={v!r}" for k, v in o["datos_extra"].items() if k != "supletorio"))
        partes.append("supletorio.documentales: " +
                      repr((o["datos_extra"].get("supletorio") or {}).get("documentales", "")))
        partes.append("")
    partes.append("=" * 78)
    partes.append("CON DATOS FALTANTES (hueco + aviso que nombra el dato)")
    partes.append("=" * 78)
    for tipo, base, cambio, clave, donde in FALTANTES:
        o = rpt.componer(tipo, _mezclar(base, cambio), {"magistrado": DATOS["magistrado"]})
        sitio = o["visto"] if donde == "V I S T O" else _res(o, donde)
        partes.append(f"· {tipo} sin {clave} → «{donde}»: {sitio}")
        partes.append("  aviso: " + " | ".join(a for a in o["avisos"] if f"({clave})" in a))
    os.makedirs(os.path.dirname(_ruta) or ".", exist_ok=True)
    with open(_ruta, "w", encoding="utf-8") as fh:
        fh.write("\n".join(partes) + "\n")
    print(f"\n   (muestras escritas en {_ruta})")

print()
if FALLOS:
    print(f"FALLAN {len(FALLOS)}: " + " · ".join(FALLOS))
    raise SystemExit(1)
print("RESULTADO: TODAS LAS COMPROBACIONES PASAN")
