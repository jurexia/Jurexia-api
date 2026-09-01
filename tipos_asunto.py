"""EL CATÁLOGO DE TIPOS DE ASUNTO — una sola tabla, y todo lo demás la lee.

David, 31-ago-2026: «los campos establecidos para generar una sentencia están
predispuestos o pensados para un amparo directo. Sólo hay un botoncito que me
permite señalarle al sistema que se trata de una revisión. Me parece que la
lógica está mal pensada. Lo primero sería preguntarle al usuario qué tipo de
asunto va a proyectar».

Tiene razón, y el precio de no haberlo hecho así está medido: el resolutivo
salía cableado al amparo directo y una QUEJA decía «La Justicia de la Unión
ampara y protege», que es una resolución que no existe en derecho. Y el plazo
por omisión era quince días para todo, cuando la queja tiene CINCO.

La causa no fue el descuido: era que el conocimiento de cada tipo estaba
repartido en ocho sitios —`ESQUELETO`, `RESOLUTIVO`, `_IDENTIFICA_ACTO`,
`_NOMBRE_ASUNTO`, `_GENERICO_ACTO`, `banco._LLAVE`, el plazo del endpoint y el
prompt de estructura— y cada vez que se añadía un tipo había que acordarse de
los ocho. Se desincronizaron, como era inevitable.

LO QUE ESTÁ MEDIDO Y LO QUE ESTÁ LEÍDO:

  · La ESTRUCTURA sale de 167 adelantos reales del corpus —77 de amparo
    directo, 57 de revisión, 29 de revisión fiscal y 21 de queja—, contando qué
    apartados aparecen y en cuántos. No son variantes del mismo documento: la
    «Existencia del acto reclamado» está en 57 de 60 amparos directos y en
    NINGUNA revisión; la «Procedencia» está en 20 de 21 quejas y en 11 de 57
    revisiones.
  · Los PLAZOS salen de la Ley de Amparo, artículo por artículo, no de memoria.

LO QUE NO ENTRA, por decisión de David: la reclamación y el impedimento. La
reclamación se podría construir desde la ley —tres días, artículo 104— pero no
hay adelantos suyos en el corpus contra los que comprobarla, y un tipo sin
patrón es un tipo que nadie ha verificado.
"""

from __future__ import annotations

# ── Los plazos, leídos de la ley ─────────────────────────────────────────────
# El plazo NO es un campo que el secretario deba teclear: lo dice la ley y
# depende del tipo. Lo que sí hay que preguntarle es si cae en una excepción,
# porque eso no se deduce del expediente.
PLAZOS = {
    "amparo_directo": {
        "dias": 15,
        "fundamento": "artículo 17 de la Ley de Amparo",
        "excepciones": [
            {"clave": "autoaplicativa", "dias": 30,
             "cuando": "Se reclama una norma general autoaplicativa o el procedimiento de extradición",
             "fundamento": "artículo 17, fracción I, de la Ley de Amparo"},
            {"clave": "penal_prision", "dias": 2920,
             "cuando": "Se reclama sentencia definitiva condenatoria que impone pena de prisión",
             "fundamento": "artículo 17, fracción II, de la Ley de Amparo (hasta ocho años)"},
            {"clave": "agrario_nucleo", "dias": 2555,
             "cuando": "El acto priva de derechos agrarios a un núcleo de población ejidal o comunal",
             "fundamento": "artículo 17, fracción III, de la Ley de Amparo (siete años)"},
            {"clave": "vida_libertad", "dias": None,
             "cuando": "El acto implica peligro de privación de la vida, ataques a la libertad "
                       "personal fuera de procedimiento, incomunicación, deportación o "
                       "desaparición forzada",
             "fundamento": "artículo 17, fracción IV, de la Ley de Amparo (en cualquier tiempo)"},
        ],
    },
    "amparo_revision": {
        "dias": 10,
        "fundamento": "artículo 86 de la Ley de Amparo",
        "excepciones": [],
    },
    "queja": {
        "dias": 5,
        "fundamento": "artículo 98 de la Ley de Amparo",
        "excepciones": [
            {"clave": "suspension", "dias": 2,
             "cuando": "Se trata de suspensión de plano o provisional",
             "fundamento": "artículo 98, fracción I, de la Ley de Amparo"},
            {"clave": "omision_tramite", "dias": None,
             "cuando": "Se omitió tramitar la demanda de amparo",
             "fundamento": "artículo 98, fracción II, de la Ley de Amparo (en cualquier tiempo)"},
        ],
    },
    "revision_fiscal": {
        "dias": 15,
        "fundamento": "artículo 63 de la Ley Federal de Procedimiento "
                      "Contencioso Administrativo",
        "excepciones": [],
    },
}


# ── Cómo se llama cada cosa en cada asunto ───────────────────────────────────
# Un secretario no escribe «quejoso» en una revisión fiscal ni «conceptos de
# violación» en una queja. El vocabulario es del tipo, no del documento.
VOCABULARIO = {
    "amparo_directo": {
        "nombre": "amparo directo",
        "promovente": "quejoso",
        "combate": "conceptos de violación",
        "recurrido": "la sentencia reclamada",
        "sub_recurrido": "Sentencia reclamada",
        "escrito": "demanda de amparo",
        "verbo_promover": "promovió",
        # Cómo se identifica el acto en el resultando primero.
        "identifica": "fecha, sala, toca y expediente de origen, y qué "
                      "confirmó, modificó o revocó",
    },
    "amparo_revision": {
        "nombre": "amparo en revisión",
        "promovente": "recurrente",
        "combate": "agravios",
        "recurrido": "la sentencia recurrida",
        "sub_recurrido": "Resolución recurrida",
        "escrito": "recurso de revisión",
        "verbo_promover": "interpuso",
        "identifica": "fecha, juzgado de distrito y número del juicio de "
                      "amparo indirecto en que se dictó",
    },
    "queja": {
        "nombre": "recurso de queja",
        "promovente": "recurrente",
        "combate": "agravios",
        "recurrido": "el auto recurrido",
        "sub_recurrido": "Auto recurrido",
        "escrito": "recurso de queja",
        "verbo_promover": "interpuso",
        "identifica": "fecha, juzgado de distrito y número del juicio de "
                      "amparo en que se dictó, y qué proveyó",
    },
    "revision_fiscal": {
        "nombre": "revisión fiscal",
        "promovente": "recurrente",
        "combate": "agravios",
        "recurrido": "la sentencia impugnada",
        "sub_recurrido": "Sentencia impugnada",
        "escrito": "recurso de revisión fiscal",
        "verbo_promover": "interpuso",
        "identifica": "fecha, sala del Tribunal Federal de Justicia "
                      "Administrativa y número del juicio de nulidad",
    },
}


# ── La estructura, contada sobre 167 adelantos reales ────────────────────────
# El número es en cuántos de ese tipo aparece el apartado. Se conservan los que
# están en más de la mitad: por debajo no es la estructura, es una variante.
ESTRUCTURA = {
    "amparo_directo": {
        "muestra": 60,
        "resultandos": [("Presentación de la demanda de amparo", 52),
                        ("Trámite del juicio de amparo", 54),
                        ("Turno del asunto", 28)],
        "considerandos": [("Competencia", 59),
                          ("Existencia del acto reclamado", 57),
                          ("Legitimación y oportunidad", 44),
                          ("Acto reclamado y conceptos de violación", 38),
                          ("Estudio", 27)],
    },
    "amparo_revision": {
        "muestra": 57,
        "resultandos": [("Presentación de la demanda de amparo indirecto", 34),
                        ("Trámite del juicio de amparo indirecto", 38),
                        ("Interposición y trámite del recurso de revisión", 26),
                        ("Turno", 32)],
        "considerandos": [("Competencia", 50),
                          ("Legitimación y oportunidad", 33),
                          ("Resolución recurrida y agravios", 27),
                          ("Estudio", 31)],
    },
    "revision_fiscal": {
        "muestra": 29,
        "resultandos": [("Trámite del juicio contencioso administrativo", 12),
                        ("Interposición del recurso de revisión fiscal", 13),
                        ("Trámite del recurso de revisión fiscal", 29),
                        ("Turno", 22)],
        "considerandos": [("Competencia", 29),
                          ("Legitimación y oportunidad", 26),
                          ("Procedencia", 20),
                          ("Consideraciones de la sentencia impugnada y agravios", 20),
                          ("Estudio de los agravios", 17)],
    },
    "queja": {
        "muestra": 21,
        "resultandos": [("Interposición del recurso de queja", 18),
                        ("Trámite del recurso", 11),
                        ("Turno del asunto", 7)],
        "considerandos": [("Competencia", 21),
                          ("Procedencia", 20),
                          ("Legitimación y oportunidad", 16),
                          ("Trascripción innecesaria del auto recurrido y agravios", 16),
                          ("Estudio", 14)],
    },
}


TIPOS = tuple(VOCABULARIO)


def normalizar(tipo: str) -> str:
    """La grafía que entre, la clave que sale. Vacío si no se reconoce."""
    import unicodedata
    x = unicodedata.normalize("NFKD", (tipo or "").strip().lower())
    x = "".join(c for c in x if not unicodedata.combining(c))
    t = x.replace(" ", "_").replace("-", "_")
    if t in VOCABULARIO:
        return t
    alias = {
        "amparo_directo_civil": "amparo_directo",
        "amparo_directo_administrativo": "amparo_directo",
        "amparo_directo_laboral": "amparo_directo",
        "directo": "amparo_directo", "ad": "amparo_directo",
        "revision": "amparo_revision", "amparo_en_revision": "amparo_revision",
        "ar": "amparo_revision", "recurso_de_revision": "amparo_revision",
        "recurso_de_queja": "queja", "queja_urgente": "queja",
        "rq": "queja", "qa": "queja", "qc": "queja",
        "revision_fiscal_": "revision_fiscal", "rf": "revision_fiscal",
        "fiscal": "revision_fiscal",
    }
    return alias.get(t, "")


def plazo_de(tipo: str, excepcion: str = "") -> dict:
    """{dias, fundamento, en_cualquier_tiempo}. El plazo NO se teclea: se sabe.

    `dias=None` significa que el recurso procede EN CUALQUIER TIEMPO, y eso no
    es un plazo largo: es la ausencia de plazo, y el cómputo no debe declarar
    extemporaneidad ninguna.
    """
    t = normalizar(tipo)
    base = PLAZOS.get(t)
    if not base:
        return {"dias": 15, "fundamento": "", "en_cualquier_tiempo": False,
                "aviso": f"Tipo de asunto «{tipo}» no reconocido: se contó con "
                         f"quince días, que es el plazo del amparo. Compruébalo."}
    if excepcion:
        for e in base["excepciones"]:
            if e["clave"] == excepcion:
                return {"dias": e["dias"], "fundamento": e["fundamento"],
                        "en_cualquier_tiempo": e["dias"] is None}
    return {"dias": base["dias"], "fundamento": base["fundamento"],
            "en_cualquier_tiempo": False}


def excepciones_de(tipo: str) -> list:
    """Lo que hay que PREGUNTARLE al secretario, porque no se deduce del acto."""
    return list(PLAZOS.get(normalizar(tipo), {}).get("excepciones", []))


def vocabulario_de(tipo: str) -> dict:
    return dict(VOCABULARIO.get(normalizar(tipo) or "amparo_directo",
                                VOCABULARIO["amparo_directo"]))


def estructura_de(tipo: str) -> dict:
    return dict(ESTRUCTURA.get(normalizar(tipo) or "amparo_directo",
                               ESTRUCTURA["amparo_directo"]))


# ═══════════════════════════════════════════════════════════════════════════
# QUÉ VA DENTRO DE CADA RESULTANDO, POR TIPO
# ═══════════════════════════════════════════════════════════════════════════
# Los rótulos de arriba dicen CUÁLES son; esto dice QUÉ se escribe en cada uno.
#
# Hacía falta porque el prompt de la estructura llevaba los cuatro resultandos
# del amparo directo escritos a mano, y los emitía igual en los cuatro tipos:
# una queja abría con «Presentación de la demanda de amparo» y seguía con
# «Derechos humanos cuya violación se alega» y «Tercero interesado», que en un
# recurso contra un auto no vienen a cuento. Comparado con los adelantos
# reales, era el apartado que más se alejaba: los CONSIDERANDOS ya salían bien
# —de aquí—, y los RESULTANDOS no, porque no salían de aquí.
#
# El contenido está tomado de lo que consignan los adelantos del corpus, no de
# lo que parece razonable: en la queja el turno se rotula «Turno del asunto» y
# en la revisión, «Turno» a secas.
RESULTANDOS = {
    "amparo_directo": [
        ("Presentación de la demanda de amparo",
         "fecha, oficialía, promovente y su carácter. Después IDENTIFICA el "
         "acto reclamado"),
        ("Derechos humanos cuya violación se alega",
         "UNA sola frase con la lista de artículos constitucionales. No argumenta"),
        ("Tercero interesado",
         "una frase: le resulta tal carácter a X, quien fue emplazado al "
         "presente juicio, según las constancias"),
        ("Trámite del juicio de amparo",
         "auto de Presidencia, registro, admisión, vista del artículo 181 de "
         "la Ley de Amparo, y que el agente del Ministerio Público adscrito "
         "omitió formular pedimento"),
        # 28 de 60 en el corpus, y presente en el adelanto real del ADA
        # 448/2025 como QUINTO. Faltaba.
        ("Turno del asunto",
         "fecha del acuerdo, a quién se turnaron los autos para la elaboración "
         "del proyecto, y el artículo 183 de la Ley de Amparo"),
    ],
    "amparo_revision": [
        ("Presentación de la demanda de amparo indirecto",
         "fecha, oficialía y promovente de la DEMANDA DE AMPARO INDIRECTO —no "
         "del recurso—, y contra qué actos se enderezó"),
        ("Trámite del juicio de amparo indirecto",
         "qué juzgado de distrito conoció, con qué número, y en qué paró: "
         "sentencia, desechamiento o sobreseimiento, con su fecha"),
        ("Interposición y trámite del recurso de revisión",
         "fecha y promovente del RECURSO, auto de Presidencia que lo admitió "
         "con su fecha y número de toca"),
        ("Turno",
         "fecha en que se turnó y a qué magistrado, para la elaboración del "
         "proyecto"),
    ],
    "revision_fiscal": [
        ("Trámite del juicio contencioso administrativo",
         "fecha y oficialía de la demanda de nulidad, quién la promovió, qué "
         "resolución impugnó, y la sentencia de la Sala con su fecha y sentido"),
        ("Interposición del recurso de revisión fiscal",
         "fecha, oficialía y autoridad que interpuso el recurso"),
        ("Trámite del recurso de revisión fiscal",
         "auto de Presidencia que lo admitió, con su fecha y número"),
        ("Turno",
         "fecha en que se turnó y a qué magistrado, para la elaboración del "
         "proyecto"),
    ],
    "queja": [
        ("Interposición del recurso de queja",
         "fecha, oficialía, promovente y su carácter en el juicio de amparo. "
         "Después IDENTIFICA el auto recurrido: fecha, juzgado, número de "
         "juicio y qué proveyó"),
        ("Trámite del recurso",
         "auto de Presidencia que lo admitió, con su fecha y número, y el "
         "informe de la autoridad si consta"),
        ("Turno del asunto",
         "fecha en que se turnó y a qué magistrado, para la elaboración del "
         "proyecto"),
    ],
}


def resultandos_de(tipo: str) -> list:
    """[(rótulo, qué va dentro)] del tipo."""
    return RESULTANDOS.get(normalizar(tipo), RESULTANDOS["amparo_directo"])


def rotulo_estudio_de(tipo: str) -> str:
    """Cómo rotula el corpus el considerando de fondo, en este tipo.

    Estaba fijo como «Estudio.» en el ensamblador, y en revisión fiscal el
    corpus lo llama «Estudio de los agravios» —17 de 29—, que es también como
    lo rotula el adelanto real de la RF 44/2025.
    """
    cs = estructura_de(tipo).get("considerandos") or []
    return (cs[-1][0] if cs else "Estudio")
