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

import re

# ── Los plazos, leídos de la ley ─────────────────────────────────────────────
# El plazo NO es un campo que el secretario deba teclear: lo dice la ley y
# depende del tipo. Lo que sí hay que preguntarle es si cae en una excepción,
# porque eso no se deduce del expediente.
PLAZOS = {
    "amparo_directo": {
        "dias": 15,
        "fundamento": "artículo 17 de la Ley de Amparo",
        "excepciones": [
            # LA AUTOAPLICATIVA NO ES DEL AMPARO DIRECTO (verificación de
            # normas, 27-sep-2026): los treinta días de la fracción I operan en
            # el indirecto; en el directo la norma no es acto reclamado
            # (artículo 175, fracción IV). Se quitó de aquí.
            {"clave": "penal_prision", "dias": 2920, "anios": 8,
             "cuando": "Se reclama sentencia definitiva condenatoria que impone pena de prisión",
             "fundamento": "artículo 17, fracción II, de la Ley de Amparo (hasta ocho años)"},
            {"clave": "agrario_nucleo", "dias": 2555, "anios": 7,
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
        # EL SINGULAR NO SE OBTIENE QUITANDO LA ÚLTIMA LETRA. El prompt del
        # estudio hacía `combate[:-1]` y en el amparo directo salía «En el
        # primer conceptos de violació…». Con «agravios» colaba; con esto, no.
        "combate_singular": "concepto de violación",
        # CÓMO SE LA NOMBRA EN LA PROSA. «la parte quejosa» salía en los cuatro
        # tipos porque estaba escrita en un EJEMPLO del prompt del estudio —«En
        # el primer agravio la parte quejosa sostiene que…»— y el modelo la
        # copiaba. El SAT nunca fue quejoso; y en segunda instancia, aunque lo
        # haya sido abajo, al sintetizar SUS agravios es la recurrente.
        "parte": "la parte quejosa",
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
        # EL SINGULAR NO SE OBTIENE QUITANDO LA ÚLTIMA LETRA. El prompt del
        # estudio hacía `combate[:-1]` y en el amparo directo salía «En el
        # primer conceptos de violació…». Con «agravios» colaba; con esto, no.
        "combate_singular": "agravio",
        # CÓMO SE LA NOMBRA EN LA PROSA. «la parte quejosa» salía en los cuatro
        # tipos porque estaba escrita en un EJEMPLO del prompt del estudio —«En
        # el primer agravio la parte quejosa sostiene que…»— y el modelo la
        # copiaba. El SAT nunca fue quejoso; y en segunda instancia, aunque lo
        # haya sido abajo, al sintetizar SUS agravios es la recurrente.
        "parte": "la parte recurrente",
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
        # EL SINGULAR NO SE OBTIENE QUITANDO LA ÚLTIMA LETRA. El prompt del
        # estudio hacía `combate[:-1]` y en el amparo directo salía «En el
        # primer conceptos de violació…». Con «agravios» colaba; con esto, no.
        "combate_singular": "agravio",
        # CÓMO SE LA NOMBRA EN LA PROSA. «la parte quejosa» salía en los cuatro
        # tipos porque estaba escrita en un EJEMPLO del prompt del estudio —«En
        # el primer agravio la parte quejosa sostiene que…»— y el modelo la
        # copiaba. El SAT nunca fue quejoso; y en segunda instancia, aunque lo
        # haya sido abajo, al sintetizar SUS agravios es la recurrente.
        "parte": "la parte recurrente",
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
        # EL SINGULAR NO SE OBTIENE QUITANDO LA ÚLTIMA LETRA. El prompt del
        # estudio hacía `combate[:-1]` y en el amparo directo salía «En el
        # primer conceptos de violació…». Con «agravios» colaba; con esto, no.
        "combate_singular": "agravio",
        # CÓMO SE LA NOMBRA EN LA PROSA. «la parte quejosa» salía en los cuatro
        # tipos porque estaba escrita en un EJEMPLO del prompt del estudio —«En
        # el primer agravio la parte quejosa sostiene que…»— y el modelo la
        # copiaba. El SAT nunca fue quejoso; y en segunda instancia, aunque lo
        # haya sido abajo, al sintetizar SUS agravios es la recurrente.
        "parte": "la autoridad recurrente",
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
        # LA REVISIÓN NO REPRODUCE LA EXISTENCIA; LLEVA LA PROCEDENCIA
        # (3-oct-2026). David: «el considerando de existencia ya no es
        # necesario porque ya viene en la sentencia recurrida; no es usual ni
        # necesario que lo reproduzcamos». Lo había añadido un adelanto suyo
        # ajustado a mano; los 5 engroses del banco que tienen «Procedencia»
        # la ponen entre la competencia y la legitimación (PRIMERO
        # Competencia, SEGUNDO Procedencia, TERCERO Legitimación y
        # oportunidad). Este orden es el que lee `documento_generado` para
        # decidir si la procedencia va antes que la legitimación. El texto lo
        # da `procedencia_revision`. [siempre]
        "considerandos": [("Competencia", 50),
                          ("Procedencia", 5),
                          ("Legitimación y oportunidad", 33),
                          ("Resolución recurrida y agravios", 27),
                          ("Antecedentes", 0),
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


def anios_de(tipo: str, excepcion: str = "") -> int:
    """Los años del plazo si la excepción los fija (art. 17, fr. II y III), o 0.
    Los «dias» de esas excepciones (2920, 2555) se conservan por compatibilidad
    con encargos guardados, pero NO se cuentan: un plazo de años va de fecha a
    fecha (verificación de normas, 27-sep-2026)."""
    for e in PLAZOS.get(normalizar(tipo), {}).get("excepciones", []):
        if excepcion and e["clave"] == excepcion:
            return int(e.get("anios") or 0)
    return 0


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
        # SIN «OFICIALÍA» (3-oct-2026): la demanda de amparo directo se
        # presenta por conducto de la responsable (art. 176 LA), y pedir la
        # oficialía sin dársela invitaba a inventarla: «Oficialía de Partes de
        # este Tribunal» salió en proyectos de octubre. Ni se nombra: un
        # ejemplo en el prompt, aun en negativo, se copia.
        ("Presentación de la demanda de amparo",
         "fecha de presentación por conducto de la autoridad responsable "
         "(artículo 176 de la Ley de Amparo), promovente y su carácter. "
         "Después INDIVIDUALIZA la "
         "sentencia reclamada: su FECHA, el órgano que la dictó, el NÚMERO DE "
         "EXPEDIENTE o toca de origen, y qué resolvió. PROHIBIDO escribir «el "
         "acto reclamado precisado en los antecedentes» o cualquier otra "
         "perífrasis que remita a otro apartado: aquí se nombra"),
        ("Derechos humanos cuya violación se alega",
         "UNA sola frase con la lista de artículos constitucionales. No argumenta"),
        ("Tercero interesado",
         "una frase, CON SU NOMBRE: «Le resulta tal carácter a Fulano de Tal, "
         "quien fue emplazado al presente juicio». Si son varios, se enumeran "
         "todos. PROHIBIDO «la persona a quien resulta tal carácter»: si el "
         "nombre no consta en la ficha de partes ni en el acto, NO ESCRIBAS "
         "este resultando —se omite entero y ya—, pero no lo sustituyas por "
         "una perífrasis, que es afirmar sin decir quién"),
        # EL MINISTERIO PÚBLICO, SÓLO SI CONSTA (3-oct-2026). Decía «y que el
        # agente del Ministerio Público adscrito omitió formular pedimento»: una
        # afirmación por defecto que el modelo escribía sin haber visto la
        # vista ni el pedimento (71 de 72 proyectos de octubre). Si lo hubo, el
        # resultando mentía.
        ("Trámite del juicio de amparo",
         "auto de Presidencia, registro, admisión y vista del artículo 181 de "
         "la Ley de Amparo. Lo que hizo el agente del Ministerio Público de la "
         "Federación (si formuló pedimento o no) SÓLO si consta en el material; "
         "si no consta, no lo menciones"),
        # 28 de 60 en el corpus, y presente en el adelanto real del ADA
        # 448/2025 como QUINTO. Faltaba.
        ("Turno del asunto",
         "fecha del acuerdo, a quién se turnaron los autos para la elaboración "
         "del proyecto, y el artículo 183 de la Ley de Amparo"),
    ],
    "amparo_revision": [
        # EL RESULTANDO PRIMERO LLEVA DOS SUB-BLOQUES ROTULADOS, no prosa
        # corrida. Así lo escribió David al ajustar el adelanto, y tiene
        # sentido: la autoridad responsable y el acto reclamado son los dos
        # datos que se buscan de un vistazo cuando se toma el asunto.
        ("Presentación de la demanda de amparo indirecto",
         "fecha y promovente de la DEMANDA DE AMPARO INDIRECTO —no "
         "del recurso—, cerrando con «en contra de las autoridades y los actos "
         "que a continuación se señalan:». Y ACTO SEGUIDO, en párrafos aparte "
         "y con su rótulo en versales:\n"
         "    AUTORIDAD RESPONSABLE:\n"
         "    <quién, en un párrafo>\n"
         "    ACTOS RECLAMADOS:\n"
         "    <qué, en un párrafo>\n"
         "Si son varias autoridades o varios actos, se enumeran dentro de su "
         "bloque. NO los metas en la prosa del primer párrafo: el rótulo es lo "
         "que permite encontrarlos sin leer"),
        ("Trámite del juicio de amparo indirecto",
         "auto de radicación con su fecha, qué juzgado conoció, bajo qué "
         "número lo registró, y que admitió la demanda, pidió los informes "
         "justificados y dio vista al Ministerio Público. Y CIERRA CON UN "
         "PÁRRAFO APARTE que diga EN QUÉ PARÓ: «Seguido el juicio por sus "
         "etapas legales correspondientes, se celebró audiencia constitucional "
         "el [fecha], en la que se [sobreseyó / concedió el amparo / se negó]». "
         "Esto "
         "último es OBLIGATORIO y con el verbo: «CONCEDIÓ el amparo», «lo "
         "NEGÓ», «SOBRESEYÓ». No basta «dictó sentencia» ni «resolvió el "
         "juicio»: de ese dato depende el punto resolutivo de esta ejecutoria "
         "—si se confirma, se revoca o se modifica— y si no consta, el "
         "resolutivo sale con un hueco. Si de verdad no consta en lo que "
         "tienes, escribe «no consta el sentido de la sentencia recurrida» y "
         "sigue; lo que no puede es callarse"),
        ("Interposición y trámite del recurso de revisión",
         "fecha y promovente del RECURSO, auto de Presidencia que lo admitió "
         "con su fecha y número de toca"),
        ("Turno",
         "LA FECHA en que se turnó y a qué magistrado, para la elaboración del "
         "proyecto. La fecha se busca en el auto de turno y se escribe. "
         # SI NO ESTÁ, SE REMITE AL EXPEDIENTE; NO SE LAMENTA.
         #
         # El proyecto salía con «No consta en los datos proporcionados la "
         # fecha en que el asunto fue turnado», que es lo peor de las tres "
         # opciones: habla de «los datos proporcionados» —el material del "
         # prompt— en vez del expediente, y deja un resultando que no dice "
         # nada. David: «hay que tratar de producir un proyecto completo en la "
         # medida de lo posible, y si no hay datos, remitirnos al expediente».
         "SI NO LA ENCUENTRAS, escribe la fórmula que remite a las "
         "constancias: «el asunto fue turnado al magistrado ponente, en la "
         "fecha que se advierte de las constancias del expediente». NUNCA "
         "escribas que el dato no consta, ni menciones «los datos "
         "proporcionados»: el lector tiene el expediente delante y esa frase "
         "sólo delata de dónde salió el texto"),
    ],
    "revision_fiscal": [
        ("Trámite del juicio contencioso administrativo",
         "fecha de la demanda de nulidad, quién la promovió, qué "
         "resolución impugnó, y LA SENTENCIA DE LA SALA CON SU FECHA Y SU "
         "SENTIDO. El sentido es OBLIGATORIO y con el verbo: «DECLARÓ LA "
         "NULIDAD», «RECONOCIÓ LA VALIDEZ», «SOBRESEYÓ». Ya lo pedía este "
         "apartado y aun así el proyecto salió diciendo quién promovió y contra "
         "qué, sin llegar nunca a decir qué resolvió la Sala —que es el acto "
         "que se está revisando—. Sin ese dato el considerando de procedencia "
         "no puede fundarse y quien lee no sabe qué se revisa"),
        ("Interposición del recurso de revisión fiscal",
         "fecha y autoridad que interpuso el recurso"),
        ("Trámite del recurso de revisión fiscal",
         "auto de Presidencia que lo admitió, con su fecha y número"),
        ("Turno",
         "fecha en que se turnó y a qué magistrado, para la elaboración del "
         "proyecto"),
    ],
    "queja": [
        ("Interposición del recurso de queja",
         "fecha, promovente y su carácter en el juicio de amparo. "
         "Después IDENTIFICA el auto recurrido: fecha, juzgado, número de "
         "juicio y qué proveyó"),
        ("Trámite del recurso",
         "auto de Presidencia que lo admitió, con su fecha y número, y el "
         "informe de la autoridad si consta. PROHIBIDO «se tramitó conforme a "
         "las constancias que integran el expediente» y cualquier otra frase "
         "que diga que hubo trámite sin decir cuál: si no consta la fecha del "
         "auto de Presidencia, NO ESCRIBAS este resultando —se omite entero—, "
         "pero no lo sustituyas por una fórmula que no informa de nada"),
        ("Turno del asunto",
         "fecha en que se turnó y a qué magistrado, para la elaboración del "
         "proyecto"),
    ],
}


def resultandos_de(tipo: str) -> list:
    """[(rótulo, qué va dentro)] del tipo.

    AMPARO DIRECTO SIN ALZADA (30-sep-2026): el primero pedía individualizar la
    sentencia por «el NÚMERO DE EXPEDIENTE o toca de origen»; en única
    instancia no hay toca, y ofrecérselo es invitar a buscarlo (o a inventarlo).
    Sólo si la instancia consta como única; si no, como siempre."""
    base = RESULTANDOS.get(normalizar(tipo), RESULTANDOS["amparo_directo"])
    if unica_instancia(tipo):
        return [(r, q.replace("NÚMERO DE EXPEDIENTE o toca de origen",
                              "NÚMERO DE EXPEDIENTE de origen"))
                for r, q in base]
    return base


def rotulo_estudio_de(tipo: str) -> str:
    """Cómo rotula el corpus el considerando de fondo, en este tipo.

    Estaba fijo como «Estudio.» en el ensamblador, y en revisión fiscal el
    corpus lo llama «Estudio de los agravios» —17 de 29—, que es también como
    lo rotula el adelanto real de la RF 44/2025.
    """
    cs = estructura_de(tipo).get("considerandos") or []
    return (cs[-1][0] if cs else "Estudio")


# ═══════════════════════════════════════════════════════════════════════════
# LA CADENA DE COMPETENCIA
# ═══════════════════════════════════════════════════════════════════════════
# «El error más grave de todos», con esas palabras, en la auditoría de David:
# la revisión fiscal fundaba la competencia del Tribunal Colegiado en el
# artículo 107, fracción VIII, de la Constitución y en los artículos 81,
# fracción I, inciso e), y 84 de la Ley de Amparo. Una revisión fiscal no es un
# amparo ni un amparo en revisión, y citar ahí la Ley de Amparo no es un
# desliz de redacción: es fundar la jurisdicción en una ley que no gobierna el
# recurso, en el considerando PRIMERO, que es el primero que lee quien firma.
#
# LA CAUSA: la revisión fiscal no tiene banco propio y toma prestado el de la
# revisión de amparo. El préstamo se pensó para las fórmulas de TRÁMITE, pero
# la búsqueda del banco no distingue apartados, así que prestaba también la
# competencia —y con ella, la ley equivocada—.
#
# ESTO NO ESTÁ INVENTADO. Sale del adelanto real de la RF 44/2025, palabra por
# palabra, cambiando por marcadores lo que es de aquel tribunal y de aquel
# asunto. Dos cosas que conviene que David confirme:
#
#   · su engrose cita el artículo 35, fracción VI, de la Ley Orgánica del Poder
#     Judicial de la Federación, y su auditoría dice fracción V. Se respeta lo
#     que dice el engrose, porque es lo que se firmó;
#   · su engrose nombra el Consejo de la Judicatura Federal sin el «otrora» que
#     sí llevan sus adelantos de amparo y de queja.
COMPETENCIA = {
    "revision_fiscal": {
        "plantilla":
            "Este {tribunal} ejerce jurisdicción y es competente para conocer "
            "y resolver el presente recurso de revisión fiscal, de conformidad "
            "con lo dispuesto en el precepto 104, fracción III, de la "
            "Constitución Política de los Estados Unidos Mexicanos; artículo "
            "35, fracción VI, de la Ley Orgánica del Poder Judicial de la "
            "Federación vigente, así como en el diverso 63 de la Ley Federal "
            "de Procedimiento Contencioso Administrativo, y en los Acuerdos "
            "Generales 3/2013 y 28/2017, ambos del Pleno del Consejo de la "
            "Judicatura Federal, publicados en el Diario Oficial de la "
            "Federación el quince de febrero de dos mil trece y trece de "
            "noviembre de dos mil diecisiete, respectivamente; en atención a "
            "que fue interpuesto contra una sentencia dictada por "
            "{responsable}, con residencia dentro de la jurisdicción de este "
            "órgano colegiado.",
        "fuente": "adelanto real RF 44/2025",
        # Lo que NO puede aparecer en la competencia de este tipo. Una revisión
        # fiscal fundada en la Ley de Amparo es la firma de que se coló la
        # plantilla del amparo, y es barato comprobarlo antes de entregar.
        "prohibido": [r"Ley\s+de\s+Amparo",
                      r"art[íi]culos?\s+8[14]\b",
                      r"107,?\s*fracci[óo]n\s+VIII"],
    },
}


# LA CADENA DE LA COMPETENCIA, POR TIPO, CONTRA LA LEY VIGENTE (27-sep-2026).
# El prompt de la estructura daba a los cuatro tipos la del amparo directo y
# cerraba con «37 de la Ley Orgánica», que era la competencia de los colegiados
# en la ley de 1995; en la de 2024 el 37 es la oficina de correspondencia común
# y la competencia es el 35. Las fracciones, verificadas en el texto vigente
# (DOF 20-12-2024, última reforma 28-11-2025): I directo (inciso de la materia),
# II revisión en los casos del 81 (la de suspensión), III queja, V revisión de
# la sentencia de audiencia y la remitida por la SCJN, VI revisión fiscal. El
# 210 es el que deja al OAJ fijar circuitos y jurisdicción. Es el respaldo: la
# fórmula que se escribe sale del banco (`banco.texto_de`).
# LA FRACCIÓN DEL 35 SALE DE LO QUE SE RESUELVE, Y NADIE EMITE OTRA (C2,
# 3-oct-2026; LOPJF del DOF 20-12-2024, última reforma 28-11-2025, cotejada en
# el texto descargado): I directo, con el inciso de su materia (a penal; b
# administrativa, también la agraria; c civil, mercantil y familiar; d
# laboral); II la revisión «en los casos a que se refiere el artículo 81» que no
# es la sentencia de audiencia (suspensión, sobreseimiento fuera de audiencia,
# reposición de constancias); III queja; V la sentencia de audiencia («en los
# casos a que se refiere el artículo 84») y lo remitido por la SCJN; VI
# revisión fiscal. La cadena de la revisión callaba el auto de sobreseimiento
# y la reposición, que también son de la II.
CADENA_COMPETENCIA = {
    "amparo_directo": (
        "los artículos 103, fracción I, y 107, fracción V, de la Constitución; "
        "33, fracción II, 34 y 170, fracción I, de la Ley de Amparo; y 35, "
        "fracción I, con el inciso de la materia (a penal; b administrativa, "
        "incluida la agraria; c civil, mercantil y familiar; d laboral), y 210 "
        "de la Ley Orgánica del Poder Judicial de la Federación"),
    "amparo_revision": (
        "los artículos 107, fracción VIII, último párrafo, de la Constitución; "
        "81, fracción I, inciso e), y 84 de la Ley de Amparo; y 35, fracción V, "
        "y 210 de la Ley Orgánica del Poder Judicial de la Federación —si lo "
        "recurrido no es la sentencia de audiencia, el inciso del 81, fracción "
        "I, que le toca (a) o b) la interlocutoria de suspensión, c) la "
        "reposición de constancias, d) el auto que sobreseyó fuera de la "
        "audiencia) y 35, fracción II—"),
    "queja": (
        "los artículos 103 y 107 de la Constitución; 97, fracción I, de la Ley "
        "de Amparo, con el inciso del supuesto; y 35, fracción III, y 210 de la "
        "Ley Orgánica del Poder Judicial de la Federación"),
    "revision_fiscal": (
        "los artículos 104, fracción III, de la Constitución; 35, fracción VI, y "
        "210 de la Ley Orgánica del Poder Judicial de la Federación; y 63, "
        "párrafo primero, de la Ley Federal de Procedimiento Contencioso "
        "Administrativo —nunca la Ley de Amparo—"),
}


def cadena_competencia(tipo: str) -> str:
    return CADENA_COMPETENCIA.get(normalizar(tipo) or "amparo_directo",
                                  CADENA_COMPETENCIA["amparo_directo"])


# LA COMPETENCIA DE LA REVISIÓN DEPENDE DE LO QUE SE RECURRE (27-sep-2026).
# El banco emite una sola fórmula —la de la sentencia de audiencia: 81, fr. I,
# inciso e), y 84 de la Ley de Amparo; 35, fr. V, de la Ley Orgánica— y salía
# también cuando lo recurrido era la interlocutoria de la suspensión
# definitiva, que es el inciso a) (o el b), si se recurre lo que modificó o
# revocó esa suspensión) y la fracción II del 35. Se decide por el PROEMIO de
# la resolución recurrida —«VISTOS para resolver el incidente de suspensión…»—
# y no por el expediente entero: una sentencia de audiencia menciona el
# incidente de suspensión de pasada.
# SÓLO EL PROEMIO, y sólo su fórmula: «VISTOS para resolver el incidente de
# suspensión…» o «INTERLOCUTORIA». Una sentencia de audiencia dice en su
# trámite «ordenó tramitar por cuerda separada el incidente de suspensión
# relativo» y con la fórmula suelta pasaba por interlocutoria (revisión del
# código, 27-sep-2026).
_RX_PROEMIO_SUSPENSION = re.compile(
    r"(?:V\s*I\s*S\s*T\s*O\s*S?|RESOLUCI[ÓO]N\s+INTERLOCUTORIA|INTERLOCUTORIA)"
    r"[^.]{0,220}?(?:para\s+)?resolver[^.]{0,80}?(?:el\s+|los\s+autos\s+del\s+)?"
    r"incidente\s+de\s+suspensi[óo]n", re.I)
_RX_AUDIENCIA = re.compile(r"audiencia\s+constitucional", re.I)
_RX_MODIFICA_SUSP = re.compile(
    r"(?:modific|revoc)\w*\s+(?:el\s+|la\s+)?(?:acuerdo|auto|interlocutoria|resoluci[óo]n)"
    r"[^.]{0,120}suspensi[óo]n\s+definitiva", re.I)
_RX_CONSTITUCIONALIDAD = re.compile(
    r"inconstitucionalidad\s+de\s+(?:la|el|los|las)\s+(?:ley|art[íi]culo|norma|decreto|reglamento|c[óo]digo)|"
    r"(?:expedici[óo]n|promulgaci[óo]n|aprobaci[óo]n|refrendo)[^.]{0,80}"
    r"(?:ley|decreto|reglamento|c[óo]digo)", re.I)
# LA NORMA IMPUGNADA SE NOMBRA COMO ACTO (3-oct-2026, AR 208/2025 y 72/2025).
# Concedido el amparo contra una ley local y recurrido por el Gobernador, el
# acto reclamado decía «La Ley de Hacienda del Estado de Querétaro, artículos 90
# y 97» o «el artículo 90 de la Ley…», sin «inconstitucionalidad» ni
# «expedición», y la competencia salió con la cadena ordinaria y sin aviso de la
# delegación que los dos engroses citan. Esto se mira en cada ACTO reclamado
# (la lista de la ficha), al principio: en los antecedentes, «el artículo 16 de
# la Ley…» es una cita y no un acto. Y el Poder Legislativo entre las
# autoridades responsables sólo está ahí cuando se reclama una norma.
_RX_ACTO_ES_NORMA = re.compile(
    r"^\s*(?:\d+[.)]\s*|[-•·]\s*)?(?:(?:la|el|los|las)\s+)?(?:"
    r"(?:ley|c[óo]digo|reglamento|decreto|norma\s+general)\b|"
    r"art[íi]culos?\s+\d+[^.;]{0,80}?\b(?:de|del)\s+(?:la\s+|el\s+)?(?:ley|c[óo]digo|reglamento|decreto)\b)",
    re.I)
_RX_LEGISLATIVO = re.compile(
    r"\bcongreso\b|\blegislatura\b|\bc[áa]mara\s+de\s+(?:diputad[oa]s|senador[ae]s)\b|"
    r"\bpoder\s+legislativo\b", re.I)


# EL AUTO QUE SOBRESEE FUERA DE LA AUDIENCIA (81, fr. I, inciso d) Y LA
# REPOSICIÓN DE CONSTANCIAS (inciso c) (3-oct-2026). La competencia sólo
# distinguía la suspensión; esos dos casos se fundaban en el inciso e) —el de
# la sentencia de audiencia— y en la fracción V del 35, cuando su cadena es la
# del inciso propio y la fracción II del 35 (oro_AR: «AUTO DE SOBRESEIMIENTO
# fuera de audiencia: 81, fr. I, inciso d), y 84 LA; 35, fr. II, y 210»). Se
# reconocen por lo que el auto dice de sí mismo, nunca por una mención de pasada.
_RX_AUTO_SOBRESEE = re.compile(
    r"sobrese\w*[^.]{0,160}fuera\s+de\s+(?:la\s+)?audiencia(?:\s+constitucional)?|"
    r"fuera\s+de\s+(?:la\s+)?audiencia(?:\s+constitucional)?[^.]{0,160}sobrese\w*", re.I)
_RX_REPOSICION = re.compile(r"incidente\s+de\s+reposici[óo]n\s+de\s+(?:constancias|autos)", re.I)

# Lo que dice el considerando de cada clase de resolución recurrida.
_CLASE_RECURRIDA = {
    "interlocutoria_suspension": "una resolución dictada en el incidente de suspensión de un juicio de amparo indirecto",
    "auto_sobreseimiento": ("un auto que sobreseyó fuera de la audiencia constitucional en un "
                            "juicio de amparo indirecto"),
    "reposicion_constancias": ("una resolución dictada en el incidente de reposición de constancias "
                               "de un juicio de amparo indirecto"),
}

# LA FRACCIÓN DEL 35 Y EL INCISO DEL 81, EN CUALQUIER GRAFÍA DE LA FÓRMULA (C2,
# 3-oct-2026). Sólo el 35 de la LEY ORGÁNICA (lo que sigue en la misma cita lo
# dice), nunca un 135 ni el 35 de otra ley.
_RX_35_FRACCION_V = re.compile(
    r"((?<!\d)35,\s*fracci[óo]n\s+)V\b(?=[^;]{0,80}Ley\s+Org[áa]nica)")
_RX_35_FRACCION_II = re.compile(
    r"((?<!\d)35,\s*fracci[óo]n\s+)II\b(?=[^;]{0,80}Ley\s+Org[áa]nica)")
_RX_81_INCISO_E = re.compile(r"((?<!\d)81,\s*fracci[óo]n\s+I,\s*inciso\s+)e\)")
_RX_81_INCISO_A_D = re.compile(r"(?<!\d)81,\s*fracci[óo]n\s+I,\s*inciso\s+[a-d]\)")


# ═══ LA PROCEDENCIA DE LA REVISIÓN (C1, 3-oct-2026) ═══════════════════════
# David: «el considerando de existencia ya no es necesario porque ya viene en la
# sentencia recurrida; no es usual ni necesario que lo reproduzcamos». En su
# lugar, entre la competencia y la legitimación, el apartado que tienen los 5
# engroses del banco que lo traen (AR 60, 201 y 208/2025 y 307/2024; el 222/2025,
# en párrafos numerados): «El presente recurso de revisión es procedente, de conformidad con el
# artículo 81, fracción I, inciso e), de la Ley de Amparo, en razón de que se
# impugna una sentencia dictada en la audiencia constitucional». El inciso lo
# decide lo recurrido (`clase_recurrida_de`), cotejado con el texto local del
# 81, fracción I: a) las que concedan o nieguen la suspensión definitiva; b) las
# que modifiquen o revoquen el acuerdo en que se conceda o niegue la suspensión
# definitiva; c) las que decidan el incidente de reposición de constancias de
# autos; d) las que declaren el sobreseimiento fuera de la audiencia
# constitucional; e) las sentencias dictadas en la audiencia constitucional.
_PROCEDENCIA_REVISION = {
    "a": "la interlocutoria que resolvió sobre la suspensión definitiva",
    # LOS DOS SUPUESTOS DEL INCISO b) (revisión de normas, 3-oct-2026). El 81-I-b
    # cubre lo que modifica o revoca el acuerdo de la suspensión definitiva «o
    # las que nieguen la revocación o modificación de esos autos», y
    # `_RX_MODIFICA_SUSP` lleva al b) cualquier «modific…/revoc…», también la
    # que NEGÓ modificar: decir «la resolución que modificó o revocó…» de ésa era
    # falso. La frase que cubre los dos: «se pronunció sobre la modificación o
    # revocación…».
    "b": ("la resolución que se pronunció sobre la modificación o revocación del "
          "acuerdo en que se concedió o negó la suspensión definitiva"),
    "c": "la resolución que decidió el incidente de reposición de constancias de autos",
    "d": "el auto que sobreseyó en el juicio fuera de la audiencia constitucional",
    "e": "una sentencia dictada en la audiencia constitucional",
}
_INCISO_DE_CLASE = {"sentencia": "e", "auto_sobreseimiento": "d",
                    "reposicion_constancias": "c", "interlocutoria_suspension": "a"}


def procedencia_revision(clase: str = "", inciso: str = "") -> str:
    """El considerando de procedencia del amparo en revisión.

    `clase` como la de `clase_recurrida_de` («» o «sentencia» |
    «interlocutoria_suspension» | «auto_sobreseimiento» |
    «reposicion_constancias»); sin clase, la sentencia de audiencia, que es lo
    que se recurre casi siempre. `inciso` («a» | «b») sólo cuenta en la
    suspensión: «b» si lo recurrido se pronunció sobre la modificación o la
    revocación de lo resuelto sobre ella (la concediera o la negara)."""
    c = (clase or "").strip().lower()
    i = _INCISO_DE_CLASE.get(c, "e")
    if c == "interlocutoria_suspension":
        _i = (inciso or "").strip().lower().strip("()")
        i = _i if _i in ("a", "b") else "a"
    return ("El presente recurso de revisión es procedente, de conformidad con el "
            f"artículo 81, fracción I, inciso {i}), de la Ley de Amparo, en razón de "
            f"que se impugna {_PROCEDENCIA_REVISION[i]}.")


def clase_recurrida_de(proemio: str, clase: str = "") -> str:
    """«interlocutoria_suspension» | «auto_sobreseimiento» |
    «reposicion_constancias» | «sentencia» | «» (no se pudo decidir).

    `clase`, si viene (la ficha de trámite la trae del papel o del
    secretario), manda; si no, el proemio de la recurrida."""
    c = (clase or "").strip().lower()
    if c in ("interlocutoria_suspension", "auto_sobreseimiento",
             "reposicion_constancias", "sentencia"):
        return c
    pr = proemio or ""
    if (pr and _RX_PROEMIO_SUSPENSION.search(pr[:700])
            and not _RX_AUDIENCIA.search(pr)):
        return "interlocutoria_suspension"
    if pr and _RX_REPOSICION.search(pr[:900]):
        return "reposicion_constancias"
    # LA SENTENCIA DE AUDIENCIA TAMBIÉN SOBRESEE, y su proemio dice
    # «audiencia constitucional» sin el «fuera de»: ésa es la del inciso e).
    _sin_fuera = re.sub(r"fuera\s+de\s+(?:la\s+)?audiencia(?:\s+constitucional)?", " ",
                        pr[:1500], flags=re.I)
    if (pr and _RX_AUTO_SOBRESEE.search(pr[:1500])
            and not _RX_AUDIENCIA.search(_sin_fuera[:900])):
        return "auto_sobreseimiento"
    return "sentencia" if pr.strip() else ""


def norma_impugnada(fuente: str = "", actos=None, autoridades=None) -> str:
    """Por qué se tiene por impugnada una norma general, o «»: la fórmula en el
    texto («inconstitucionalidad de la ley», «expedición y promulgación…»), un
    acto reclamado que ES una norma («La Ley de Hacienda…, artículos 90 y 97»,
    «el artículo 90 de la Ley…») o el Poder Legislativo entre las responsables."""
    if fuente and _RX_CONSTITUCIONALIDAD.search(fuente):
        return "la demanda o la sentencia la impugnan"
    _actos = [actos] if isinstance(actos, str) else list(actos or [])
    for a in _actos:
        for linea in re.split(r"[\n;]", str(a or "")):
            if _RX_ACTO_ES_NORMA.search(linea):
                return "un acto reclamado es una norma general"
    _auts = [autoridades] if isinstance(autoridades, str) else list(autoridades or [])
    if any(_RX_LEGISLATIVO.search(str(x or "")) for x in _auts):
        return "el Poder Legislativo es autoridad responsable"
    return ""


def competencia_revision(texto_comp: str, proemio: str, fuente: str = "",
                         clase: str = "", actos=None, autoridades=None,
                         resolvio: str = "", papel: str = "") -> tuple:
    """(competencia, avisos) para la revisión, según lo recurrido.

    `actos` y `autoridades` (de la ficha: `demanda.actos`, `demanda.autoridades`
    o `procesal.autoridades`) dejan ver la norma impugnada aunque nadie escriba
    «inconstitucionalidad» (`norma_impugnada`); `resolvio` (concede|niega|
    sobresee|mixto) y `papel` (quien recurre) afinan el aviso de la delegación.
    Se miran SÓLO si llegan por estos parámetros: en `fuente` (prosa de los
    antecedentes) «el artículo 16 de la Ley…» o «Congreso» son citas, no actos."""
    avisos = []
    t = texto_comp or ""
    _clase = clase_recurrida_de(proemio, clase)
    if _clase in ("interlocutoria_suspension", "auto_sobreseimiento", "reposicion_constancias"):
        if _clase == "interlocutoria_suspension":
            inciso = "b" if _RX_MODIFICA_SUSP.search(proemio or "") else "a"
        else:
            inciso = "d" if _clase == "auto_sobreseimiento" else "c"
        nuevo = t
        nuevo = re.sub(r"una\s+sentencia\s+dictada\s+en\s+un\s+juicio\s+de\s+amparo\s+indirecto",
                       _CLASE_RECURRIDA[_clase], nuevo, count=1)
        nuevo = nuevo.replace("81, fracción I, inciso e) y 84",
                              f"81, fracción I, inciso {inciso}), y 84")
        nuevo = _RX_81_INCISO_E.sub(lambda m: m.group(1) + inciso + ")", nuevo)
        nuevo = nuevo.replace("35, fracción V y 210", "35, fracción II y 210")
        # LA FRACCIÓN DEL 35 EN CUALQUIER GRAFÍA (C2, 3-oct-2026): la del
        # banco escribe «35, fracción V y 210», pero otras fórmulas de la mesa
        # escriben «35, fracción V, de la Ley Orgánica…».
        nuevo = _RX_35_FRACCION_V.sub(lambda m: m.group(1) + "II", nuevo)
        if nuevo != t:
            _que = {"interlocutoria_suspension": "LA INTERLOCUTORIA DE SUSPENSIÓN",
                    "auto_sobreseimiento": "EL AUTO QUE SOBRESEYÓ FUERA DE LA AUDIENCIA CONSTITUCIONAL",
                    "reposicion_constancias": "LA RESOLUCIÓN DEL INCIDENTE DE REPOSICIÓN DE CONSTANCIAS"}[_clase]
            avisos.append(
                f"LO RECURRIDO ES {_que}: la competencia "
                f"se fundó en el artículo 81, fracción I, inciso {inciso}), de la "
                "Ley de Amparo y en la fracción II del 35 de la Ley Orgánica, no "
                "en los de la sentencia de audiencia. Compruébalo.")
            t = nuevo
    else:
        # LA SENTENCIA DE AUDIENCIA ES LA FRACCIÓN V, TAMBIÉN LA REMITIDA POR LA
        # SCJN (C2, 3-oct-2026). La variante del banco para la norma impugnada
        # (delegación) y la fórmula antigua de la mesa escriben «35, fracción
        # II»; la V del texto vigente cubre la sentencia de audiencia «en los
        # casos a que se refiere el artículo 84» y lo «remitido por la Suprema
        # Corte». Sólo si consta que lo recurrido es la sentencia, o si la
        # propia fórmula cita el inciso e) del 81.
        if _clase == "sentencia" or (not _clase and _RX_81_INCISO_E.search(t)):
            _v = _RX_35_FRACCION_II.sub(lambda m: m.group(1) + "V", t)
            if _v != t:
                avisos.append(
                    "LA COMPETENCIA CITABA LA FRACCIÓN II DEL ARTÍCULO 35 DE LA LEY ORGÁNICA "
                    "para una sentencia dictada en la audiencia constitucional: se cambió a la "
                    "fracción V, que es la de esa sentencia (y la de lo remitido por la SCJN); la "
                    "II es la de lo demás que prevé el 81 (suspensión, sobreseimiento fuera de "
                    "audiencia, reposición de constancias). Compruébalo.")
                t = _v
        elif not _clase and _RX_81_INCISO_A_D.search(t):
            # SIN PROEMIO, LA FÓRMULA MANDA: si ella misma cita el inciso a), b),
            # c) o d) del 81, la fracción del 35 es la II (la fórmula antigua
            # de la mesa trae «inciso {inciso})» y la fracción fija).
            _ii = _RX_35_FRACCION_V.sub(lambda m: m.group(1) + "II", t)
            if _ii != t:
                avisos.append(
                    "LA COMPETENCIA CITABA LA FRACCIÓN V DEL ARTÍCULO 35 DE LA LEY ORGÁNICA con un "
                    "inciso del 81 que no es el de la sentencia de audiencia: se cambió a la fracción "
                    "II. Compruébalo.")
                t = _ii
        _porque = norma_impugnada(fuente, actos, autoridades)
        if _porque:
            _res = (resolvio or "").strip().lower()
            _pap = (papel or "").strip().lower()
            # CONCEDIDO Y RECURRIDO POR LA AUTORIDAD, la delegación es lo
            # esperable (AR 208/2025 y 72/2025: los dos engroses la citan). Y SI
            # SE RESUELVE SOBRE LA CONSTITUCIONALIDAD DE LA NORMA, EL PROYECTO SE
            # PUBLICA ANTES (art. 73, segundo párrafo, LA, texto local: «deberán
            # hacer públicos los proyectos… cuando menos con tres días de
            # anticipación a la publicación de las listas»; el AR 201/2025 trae su
            # apartado de publicidad del proyecto).
            _probable = (" EL JUZGADO CONCEDIÓ EL AMPARO Y RECURRE LA AUTORIDAD: que el "
                         "problema de constitucionalidad subsista es lo esperable, así que "
                         "la delegación muy probablemente aplica."
                         if _res in ("concede", "mixto") and _pap == "autoridad" else "")
            avisos.append(
                f"SE IMPUGNÓ LA CONSTITUCIONALIDAD DE UNA NORMA ({_porque}). Si el problema "
                "subsiste en la revisión, este tribunal conoce por DELEGACIÓN de la "
                "SCJN y la competencia cambia: artículo 83 de la Ley de Amparo en "
                "relación con el punto cuarto, fracción I, inciso que corresponda, "
                "del Acuerdo General 2/2025 (12a.) y el punto segundo del 11/2025 "
                "(12a.), ambos del Pleno de la SCJN (DOF 19 y 22-09-2025), y la "
                "fracción V del 35 de la Ley Orgánica («remitidos por la Suprema "
                "Corte»). Si la norma es federal y no hay jurisprudencia, no hay "
                "delegación: se remite a la SCJN." + _probable
                + " Y SI LA SENTENCIA SE PRONUNCIA SOBRE LA CONSTITUCIONALIDAD DE LA NORMA, "
                  "el proyecto debe hacerse público cuando menos con tres días de "
                  "anticipación a la publicación de la lista (artículo 73, segundo "
                  "párrafo, de la Ley de Amparo).")
    return t, avisos


def competencia_de(tipo: str) -> dict:
    """La cadena propia del tipo, o {} si la toma del banco."""
    return COMPETENCIA.get(normalizar(tipo), {})


# EL ALCANCE ES EL PÁRRAFO, NO EL DOCUMENTO. Correr esto sobre la sentencia
# entera hacía saltar la alarma por el «artículo 19 de la Ley de Amparo» que el
# párrafo del CÓMPUTO cita para los días inhábiles —otra discusión, y no
# medida—. Una alarma que salta siempre deja de leerse, así que se acota al
# considerando de competencia, que es donde la ley ajena sí es un vicio.
_RX_COMPETENCIA = re.compile(
    r"(?:PRIMERO\.\s*)?Competencia\.(.{0,2500}?)(?=\n\s*(?:SEGUNDO|TERCERO)\.|\Z)",
    re.S | re.I)


def parrafo_competencia(texto: str) -> str:
    """El considerando de competencia, aislado del resto."""
    m = _RX_COMPETENCIA.search(texto or "")
    return m.group(1) if m else ""


def prohibido_en_competencia(tipo: str, texto: str, acotar: bool = True) -> list:
    """Qué se coló de otra plantilla. Determinista, sin modelo.

    `acotar` recorta al considerando de competencia; pásalo en False cuando ya
    se le entrega ese párrafo solo.
    """
    c = competencia_de(tipo)
    if not c.get("prohibido"):
        return []
    t = parrafo_competencia(texto) if acotar else (texto or "")
    if acotar and not t:
        return []
    return [pat for pat in c["prohibido"] if re.search(pat, t, re.I)]


# ═══════════════════════════════════════════════════════════════════════════
# EL CIERRE DEL ESTUDIO
# ═══════════════════════════════════════════════════════════════════════════
# LOS CUATRO PROYECTOS CERRABAN IGUAL, incluido el que resolvía una queja:
#
#     «Por lo expuesto, dado lo infundado de los agravios, lo procedente es
#      negar el amparo solicitado.»
#
# En un amparo directo es correcto. En una queja es una resolución que no
# existe —no hay amparo que negar, hay un recurso que declarar infundado— y,
# peor, el documento llevaba LAS DOS frases: ésta en el estudio y «ÚNICO. Es
# infundado el recurso de queja» treinta líneas más abajo. Se contradecía a sí
# mismo, y la comprobación de congruencia que corre sobre el .docx no dijo nada
# porque sólo conocía las fórmulas del amparo.
#
# LA CAUSA: el punto resolutivo SÍ salía de una tabla por tipo; el párrafo de
# cierre del estudio y el apartado de Efectos estaban escritos a mano treinta
# líneas más allá, en la misma función, sin mirarla. El saber de qué decreta
# cada tipo vivía en un módulo de composición, así que sólo lo consultaba el
# trozo de código que tenía al lado.
#
# LA FORMA ESTÁ MEDIDA y es la misma en los cuatro; lo único que cambia es el
# desenlace:
#
#   amparo directo   «En ese sentido, ante la ineficacia de los conceptos de
#                     violación planteados, lo procedente es negar el amparo
#                     solicitado.»
#   revisión fiscal  «En relatadas circunstancias, ante la ineficacia de los
#                     agravios planteados, lo procedente es confirmar la
#                     sentencia recurrida.»
#
# Y SÓLO EL AMPARO DIRECTO LLEVA APARTADO DE EFECTOS. «Procede conceder el
# amparo y protección de la Justicia Federal para el efecto de que la
# responsable deje insubsistente la sentencia reclamada» no se escribe en una
# queja fundada: ahí se revoca el auto y se ordena proveer de nuevo.
CIERRE = {
    "amparo_directo": {
        "negativo": "negar el amparo solicitado",
        "positivo": "conceder el amparo y protección de la Justicia Federal",
        "efectos": True,
    },
    "amparo_revision": {
        "negativo": "confirmar la sentencia recurrida",
        "positivo": "revocar la sentencia recurrida",
        "efectos": False,
    },
    "queja": {
        "negativo": "declarar infundado el recurso de queja",
        "positivo": "declarar fundado el recurso de queja",
        "efectos": False,
    },
    "revision_fiscal": {
        "negativo": "confirmar la sentencia recurrida",
        "positivo": "revocar la sentencia recurrida",
        "efectos": False,
    },
}


# ═══════════════════════════════════════════════════════════════════════════
# EL SUPLETORIO DE LA LEY DE AMPARO DEPENDE DE LA SEDE (3-oct-2026)
# ═══════════════════════════════════════════════════════════════════════════
# El 27-sep-2026 se pasó a toda la república al Código Nacional de
# Procedimientos Civiles y Familiares —el artículo 2o. de la Ley de Amparo lo
# nombra desde la reforma del DOF 13-03-2025— y la existencia del acto salió
# fundada en el CNPCF en 18 de 18 proyectos de octubre. David, 3-oct-2026:
# «aplicar, por ahora en toda la república, el código federal de procedimientos
# civiles de manera supletoria a la ley de amparo, salvo en los proyectos de
# ciudad de mexico donde ya opera el CNPCF». Es lo que firma el corpus: la
# fórmula del CFPC está en 61 de 86 existencias (banco_formulas_medidas.json:64)
# y la combinación 129 y 202 es la más estable.
#
# SIN «CONFORME A SU ARTÍCULO 2o.» FUERA DE LA CIUDAD DE MÉXICO: el 2o. vigente
# nombra al CNPCF, así que citarlo para aplicar el CFPC haría que la frase se
# contradijera a sí misma. En la Ciudad de México, la del Nacional con su 2o.
# (312, fracciones II y VIII: informes de funcionarios y actuaciones judiciales;
# 344: prueba plena; 269: hecho notorio; 348: información electrónica —todos
# verificados contra el texto del CNPCF—).
_CFPC = "Código Federal de Procedimientos Civiles"
_CNPCF = "Código Nacional de Procedimientos Civiles y Familiares"
_COLA_CFPC = f"del {_CFPC}, de aplicación supletoria a la Ley de Amparo"
_COLA_CNPCF = (f"del {_CNPCF}, de aplicación supletoria a la Ley de Amparo "
               f"conforme a su artículo 2o.")

# Alias de compatibilidad: la de fuera de la Ciudad de México, que es la regla.
SUPLETORIO_DOCUMENTALES = f"los artículos 129 y 202 {_COLA_CFPC}"

# UNA SOLA REGLA PARA LA SEDE (3-oct-2026, rev_2). Había dos: ésta dejaba que la
# ciudad decidiera sola aunque el tribunal fuera del Primer Circuito y no leía
# «Cd. de México»; `ficha_tramite.sede_de` sí la leía y aplicaba el circuito.
# Con «Décimo Tribunal Colegiado en Materia Civil del Primer Circuito» y «Cd. de
# México», la existencia salía con el CFPC y la verja acusaba esa misma fórmula
# como ajena a la sede: dos sitios que decidían el código por su lado. Ahora es
# ésta la que deciden todos (`supletorio`, `ficha_tramite.sede_de`, la verja):
#   · el PRIMER CIRCUITO es la Ciudad de México, escriba lo que escriba la
#     ciudad (no tiene otra residencia);
#   · un CENTRO AUXILIAR no es de un circuito: decide su ciudad de residencia;
#   · sin circuito legible, la ciudad, con todas sus grafías («Ciudad de
#     México», «CDMX», «Cd. de México», «Cd. México», «Cd. Mx.», «México,
#     D.F.», «Distrito Federal»);
#   · un circuito que no es el Primero, fuera de la Ciudad de México.
_RX_CDMX = re.compile(
    r"ciudad\s+de\s+m[ée]xico|\bCDMX\b|\bcd\.?\s*(?:de\s+)?m[ée]x(?:ico)?\b(?!\w)|"
    r"\bcd\.?\s*mx\b\.?|\bM[ée]xico,?\s+D\.?\s*F\.?|distrito\s+federal", re.I)
_RX_AUXILIAR_SEDE = re.compile(r"centro\s+auxiliar|auxiliar\s+de\s+la\s+\w+\s+regi[óo]n", re.I)


def es_cdmx(tribunal: str = "", ciudad: str = "") -> bool | None:
    """¿Sede en la Ciudad de México? True | False | None (no se puede decidir).

    El Primer Circuito lo es siempre (salvo un Centro Auxiliar, que decide por
    su ciudad de residencia); sin circuito legible, la ciudad; un circuito
    distinto del Primero, no."""
    c = " ".join(str(ciudad or "").split())
    t = " ".join(str(tribunal or "").split())
    _aux = bool(_RX_AUXILIAR_SEDE.search(t))
    n = 0
    if t and not _aux:
        try:
            import banco as _bk_s
            n = _bk_s.numero_de_circuito(t)
        except Exception:
            n = 0
    if n:
        return n == 1
    if c:
        return bool(_RX_CDMX.search(c))
    if t and _RX_CDMX.search(t):
        return True
    return None


def supletorio(tribunal: str = "", ciudad: str = "") -> dict:
    """Las cadenas del código supletorio de la Ley de Amparo para esta sede.

    {cdmx, codigo, documentales, hecho_notorio, electronica, aviso}. Sin
    ciudad ni circuito legibles se usa la regla general (CFPC) y `aviso` lo
    dice: la excepción es la Ciudad de México, y no se afirma sin saberlo."""
    d = es_cdmx(tribunal, ciudad)
    aviso = ("" if d is not None else
             "NO SE PUDO SABER SI LA SEDE ES LA CIUDAD DE MÉXICO (no hay ciudad ni "
             "circuito legibles): el supletorio de la Ley de Amparo se fundó en el "
             "Código Federal de Procedimientos Civiles, que es la regla; si el "
             "tribunal reside en la Ciudad de México, va el Código Nacional de "
             "Procedimientos Civiles y Familiares.")
    if d:
        return {"cdmx": True, "codigo": _CNPCF,
                "documentales": f"los artículos 312, fracciones II y VIII, y 344 {_COLA_CNPCF}",
                "hecho_notorio": f"el artículo 269 {_COLA_CNPCF}",
                "electronica": f"el artículo 348 {_COLA_CNPCF}",
                "aviso": aviso}
    return {"cdmx": False, "codigo": _CFPC,
            "documentales": SUPLETORIO_DOCUMENTALES,
            "hecho_notorio": f"el artículo 88 {_COLA_CFPC}",
            "electronica": f"el artículo 210-A {_COLA_CFPC}",
            "aviso": aviso}


# LOS EFECTOS ABREN CON SU FUNDAMENTO Y CUELGAN DE «DEBERÁ:» (27-sep-2026).
# David: «inicia con "Efectos. 1. Deje insubsistente…" cuando debería decir:
# "Efectos. Con fundamento en el artículo … de la Ley de Amparo, la autoridad
# responsable deberá: …", con efectos más reducidos pero que comprendan el
# objetivo de la concesión. Regularmente los efectos son 3 a máximo 5 en
# función de la complejidad del caso».
#
# EL PRECEPTO ES EL 77. Es el que manda que «en el último considerando de la
# sentencia que conceda el amparo» se determinen con precisión los efectos; el
# 93 son las reglas con que el colegiado resuelve la REVISIÓN. En su propio
# banco, la apertura que funda los efectos lo hace con el 77 («Efectos del fallo
# protector. De conformidad con el artículo 77 de la Ley de Amparo…») y los
# encabezados de resolutivo que los apoyan citan el 77.
#
# Y LAS ÓRDENES VAN EN INFINITIVO: cuelgan de «deberá:». «Deberá: 1. Deje
# insubsistente…» no concuerda; «deberá: 1. Dejar insubsistente…» sí, y es como
# lo escribe el esqueleto medido del banco (`ad-c6-efectos`).
APERTURA_EFECTOS = ("Con fundamento en el artículo 77 de la Ley de Amparo, la "
                    "autoridad responsable deberá:")
EFECTOS_MIN, EFECTOS_MAX = 3, 5


def cierre_de(tipo: str) -> dict:
    return CIERRE.get(normalizar(tipo), CIERRE["amparo_directo"])


def parrafo_cierre(tipo: str, concede: bool, calificacion: str = "") -> str:
    """El cierre del estudio, con la forma del corpus y el desenlace del tipo."""
    v = vocabulario_de(tipo)
    c = cierre_de(tipo)
    desenlace = c["positivo"] if concede else c["negativo"]
    if concede:
        # «AL RESULTAR FUNDADOS LO PLANTEADO» no concuerda: la calificación
        # llega en plural («fundados», «esencialmente fundados») y el sujeto
        # era singular. El sujeto es el del tipo, como en la rama negativa.
        _cal = (calificacion or "fundados").strip()
        # «sin materia» no se pluraliza: «unos fundados y otros sin materia».
        if not _cal.split()[-1].endswith("s") and not _cal.endswith("sin materia"):
            _cal = _cal + "s"
        return (f"En ese sentido, al resultar {_cal} los {v['combate']} "
                f"planteados, lo procedente es {desenlace}.")
    return (f"En ese sentido, ante la ineficacia de los {v['combate']} "
            f"planteados, lo procedente es {desenlace}.")


# LO QUE NINGÚN TIPO PUEDE DECIR. Determinista, sobre el .docx terminado: si
# una queja habla de negar el amparo, se coló la plantilla del amparo directo.
_AJENO = {
    "amparo_directo": [],
    "amparo_revision": [r"negar el amparo", r"conceder el amparo"],
    "queja": [r"negar el amparo", r"conceder el amparo",
              r"confirmar la sentencia recurrida"],
    # OJO CON EL ALCANCE: «Ley de Amparo» a secas NO va aquí. El párrafo del
    # cómputo cita el artículo 19 de esa ley para los días inhábiles, y eso es
    # otra discusión —cuál es el calendario de una revisión fiscal ante un
    # Colegiado— que no está medida en el corpus: los adelantos reales no
    # desglosan el cómputo, así que no dicen qué precepto invocan. Prohibirla
    # en todo el documento haría saltar la alarma en cada revisión fiscal, y
    # una alarma que salta siempre deja de leerse. Donde sí está prohibida es
    # en la COMPETENCIA, y de eso se ocupa `prohibido_en_competencia`.
    "revision_fiscal": [r"negar el amparo", r"conceder el amparo",
                        r"la Justicia de la Uni[óo]n"],
}


# ═══════════════════════════════════════════════════════════════════════════
# EL RESULTANDO QUE NO DICE QUÉ RESOLVIÓ EL ACTO RECURRIDO
# ═══════════════════════════════════════════════════════════════════════════
# El catálogo YA lo pide —«IDENTIFICA el auto recurrido: fecha, juzgado, número
# de juicio y qué proveyó»— y aun así salió esto:
#
#   «SEGUNDO. Trámite del recurso. El recurso se tramitó conforme a las
#    constancias que integran el expediente.»
#
# frente a lo que escribe el secretario:
#
#   «…contra el auto de seis de marzo de dos mil veintiséis, dictado por el
#    Juzgado Tercero de Distrito…, en el juicio de amparo 371/2026, mediante el
#    cual tuvo por cumplida la prevención y DESECHÓ DE PLANO LA DEMANDA.»
#
# Contra un modelo que incumple una instrucción no vale insistir en la
# instrucción: hace falta mirar el resultado. Y no es cosmética —tres piezas
# dependen de ese dato—: el inciso del artículo 97 se deduce de QUÉ proveyó el
# auto, la rama del artículo 93 de QUÉ resolvió el juzgado, y quien lee el
# proyecto no sabe de qué va el asunto si el resultando no lo dice.
_EVASIVAS = [
    r"se\s+tramit[óo]\s+conforme\s+a\s+(?:las\s+)?constancias",
    r"conforme\s+a\s+(?:las\s+)?constancias\s+que\s+integran",
    r"se\s+siguieron\s+los\s+tr[áa]mites\s+de\s+ley",
    r"previos?\s+los\s+tr[áa]mites\s+(?:de\s+ley|correspondientes)",
    r"en\s+los\s+t[ée]rminos\s+que\s+obran\s+en\s+autos",
    # LA EVASIVA QUE INVENTÓ HOY, cuando el catálogo le exigió la fecha del
    # turno: «La fecha del turno CONSTA EN AUTOS; el asunto fue turnado al
    # magistrado…». Sustituir el dato por la promesa de que el dato existe es
    # la misma figura de siempre con otra ropa, y así van tres.
    # «cuya fecha Y NÚMERO DE TOCA constan en autos» —plural, y con dos datos
    # en medio— se escapaba del patrón en singular. Es la misma evasiva con más
    # cosas escondidas dentro, así que se admite el plural y una lista.
    r"(?:cuy[ao]s?\s+)?(?:la\s+)?fecha[^.;]{0,60}const(?:a|an)\s+en\s+autos",
    r"consta\s+en\s+autos\s+que\s+el\s+asunto\s+fue\s+turnado",
    r"seg[úu]n\s+consta\s+en\s+(?:autos|el\s+expediente)",
    # LA TERCERA FORMA, inventada en la misma tarde que las dos anteriores:
    # «Consta en autos que, EN LA FECHA ASENTADA EN EL AUTO RESPECTIVO, el
    # asunto fue turnado…». Ya no dice «consta en autos» pegado a «fecha»: dice
    # que la fecha está asentada en un auto, que es la misma promesa con dos
    # rodeos más. Van seis formas medidas de la misma figura, y todas hacen lo
    # mismo: ocupar el sitio del dato con la garantía de que el dato existe.
    r"(?:en\s+)?la\s+fecha\s+asentada\s+en\s+el\s+(?:auto|acuerdo|proveido|prove[íi]do)",
    r"en\s+la\s+fecha\s+que\s+(?:obra|consta|aparece)\s+en",
]
_EVASIVAS = [__import__("re").compile(p, __import__("re").I) for p in _EVASIVAS]

# Qué proveyó: los verbos con que un auto o una sentencia deciden algo.
# LOS VERBOS CON QUE SE DECIDE, POR VÍA. En el contencioso administrativo la
# Sala no ampara ni desecha: DECLARA LA NULIDAD, RECONOCE LA VALIDEZ o
# SOBRESEE. Sin esos tres, la comprobación no podía distinguir un resultando
# completo de uno mudo en la revisión fiscal.
_QUE_PROVEYO = __import__("re").compile(
    r"declar[óo]\s+la\s+nulidad|reconoci[óo]\s+la\s+validez|"
    r"desech[óa]|admit[ií][óo]?|previn[oe]|tuvo\s+por|sobresey[óo]|"
    r"conced[ií][óo]|neg[óo]|requiri[óo]|orden[óo]|declar[óo]|resolvi[óo]|"
    r"no\s+admit|ampar[óo]|reserv[óo]|dej[óo]\s+sin\s+efecto", __import__("re").I)


def resultando_evasivo(texto: str, tipo: str = "") -> list:
    """Los resultandos que dicen que hubo trámite sin decir cuál.

    Devuelve [(qué, la frase)] para los avisos. Sólo mira el bloque de
    resultandos: en el estudio, «previos los trámites de ley» puede aparecer
    citando la sentencia de otro, y ahí es una transcripción, no una evasiva.
    """
    import re as _re
    t = texto or ""
    i = _re.search(r"R\s*E\s*S\s*U\s*L\s*T\s*A\s*N\s*D\s*O", t)
    j = _re.search(r"C\s*O\s*N\s*S\s*I\s*D\s*E\s*R\s*A\s*N\s*D\s*O", t)
    if not i:
        return []
    bloque = t[i.end():j.start()] if j and j.start() > i.end() else t[i.end():i.end() + 9000]
    # EL TURNO TIENE PERMISO DE REMITIRSE AL EXPEDIENTE, y el resto no.
    #
    # Estos patrones nacieron para cazar al modelo cuando sustituía un dato por
    # la promesa de que el dato existe. Siguen valiendo para el trámite: decir
    # «se siguieron los trámites de ley» en vez de contar qué se proveyó es
    # esconder trabajo sin hacer.
    #
    # Pero David acotó la regla: «hay que producir un proyecto completo en la
    # medida de lo posible, y si no hay datos, remitirnos al expediente». La
    # fecha del turno es el caso: está en el auto de turno, que no siempre
    # viaja con lo que se sube, y la alternativa —«no consta»— deja un
    # resultando mudo. Ahí la remisión es lo correcto y no se acusa.
    #
    # Se distingue por dónde aparece: si la frase está en el resultando del
    # TURNO, pasa; en cualquier otro, se sigue avisando.
    _rx_turno = _re.compile(r"turn", _re.I)
    def _es_del_turno(pos: int) -> bool:
        ini = bloque.rfind(".", 0, max(0, pos - 240))
        return bool(_rx_turno.search(bloque[max(0, ini):pos + 120]))

    fuera = []
    for rx in _EVASIVAS:
        m = rx.search(bloque)
        if m and _es_del_turno(m.start()):
            continue
        if m:
            fuera.append(("un resultando dice que hubo trámite sin decir cuál",
                          " ".join(bloque[max(0, m.start() - 90):m.end() + 70].split())))
    # Y EL DATO QUE MÁS SE ECHA EN FALTA: qué proveyó el acto recurrido.
    if normalizar(tipo) in ("queja", "amparo_revision", "revision_fiscal"):
        if not _QUE_PROVEYO.search(bloque):
            fuera.append(("NINGÚN RESULTANDO DICE QUÉ RESOLVIÓ EL ACTO RECURRIDO. "
                          "Sin ese dato no se puede fundar la procedencia ni "
                          "elegir el resolutivo, y quien lea el proyecto no "
                          "sabe de qué va el asunto", ""))
    return fuera


def cierre_ajeno(tipo: str, texto: str) -> list:
    """Las fórmulas de otro tipo que aparecen en este documento."""
    import re as _re
    return [p for p in _AJENO.get(normalizar(tipo), [])
            if _re.search(p, texto or "", _re.I)]


# ═══════════════════════════════════════════════════════════════════════════
# LA CARÁTULA: QUIÉN ES QUIÉN EN CADA TIPO
# ═══════════════════════════════════════════════════════════════════════════
# «Rubro indebido: asienta como Autoridad Responsable al Juez de Distrito»,
# «no hay quejoso; es Recurrente y Actora». Las dos son de la auditoría y las
# dos salían de la misma línea: una lista de etiquetas escrita a mano con las
# tres figuras del amparo directo, que se imprimía igual en los cuatro tipos
# aunque la función tuviera el tipo en la mano.
#
# En un recurso NO HAY autoridad responsable. Hay un órgano cuya resolución se
# recurre, y llamarlo «responsable» no es un matiz: en el amparo la autoridad
# responsable es la que emitió el acto reclamado y es PARTE; el Juez de
# Distrito que dictó la sentencia recurrida no es parte de nada, es el órgano
# de control cuya decisión se revisa.
#
# Y LA PRIMERA LÍNEA NO SE ROTULA «EXPEDIENTE». El corpus escribe la clase del
# asunto como etiqueta —«AMPARO DIRECTO CIVIL: 125/2026», «REVISIÓN FISCAL:
# 87/2025»—, así que anteponerle «EXPEDIENTE: » la rotula dos veces.
#
# Cada fila: (etiqueta, clave de los datos, obligatoria).
CARATULA = {
    # EL ADHERENTE TAMBIÉN VA EN EL RUBRO (3-oct-2026, AD 279/2025: «QUEJOSO
    # PRINCIPAL: … / QUEJOSO ADHERENTE: …»). La ficha traía `adhesivo.quien`, el
    # resultando, el considerando y el resolutivo segundo lo usaban, y la
    # carátula lo callaba. Concuerda como la quejosa: «PARTE QUEJOSA ADHERENTE»
    # cuando el nombre no dice el género. Lo llena `datos["adherente"]`.
    "amparo_directo": [
        ("QUEJOSO", "quejoso", True),
        ("{QUEJOSO_A} ADHERENTE", "adherente", False),
        ("TERCERO INTERESADO", "tercero", False),
        ("AUTORIDAD RESPONSABLE", "responsable", True),
    ],
    # LA FORMA LA MANDA EL CORPUS (28-sep-2026, AR 631/2025). Medido sobre la
    # primera página de 500 amparos en revisión del propio Tercer Tribunal
    # Colegiado en Materias Administrativa y Civil del Vigésimo Segundo
    # Circuito (oaj/lectura, semilla 631; 478 carátulas legibles; conteos y
    # ejemplos en SCR/caratulas_revision.md):
    #
    #   · el órgano recurrido o el juzgado NO aparece en NINGUNA (0 de 478).
    #     Una sesión anterior añadió «ÓRGANO RECURRIDO» citando engroses que
    #     no eran de revisión, y el 631 salió con «ÓRGANO RECURRIDO:
    #     MAGISTRADA…». El juzgado ya se nombra donde el corpus lo nombra: en
    #     el V I S T O y en la competencia (`organo_recurrido` sigue vivo ahí);
    #     en el rubro lo ascendía a parte;
    #   · cuando recurre la quejosa, «QUEJOSA Y RECURRENTE» (162 de 226; el
    #     resto «RECURRENTE (PARTE QUEJOSA)»);
    #   · cuando recurre OTRA parte, el recurrente lleva su CARÁCTER en el
    #     rótulo —«TERCERA INTERESADA Y RECURRENTE», «AUTORIDAD RESPONSABLE Y
    #     RECURRENTE»— y en 51 de 146 va además un renglón «QUEJOSA:» encima
    #     (AR 60/2020: «QUEJOSO: … / TERCERA INTERESADA Y RECURRENTE: …», la
    #     forma del 631). Se usa ésa porque David pidió los dos renglones y
    #     porque la carátula de una sola figura callaba quién pidió el amparo;
    #     el partido lo hace `filas_caratula`, no esta tabla;
    #   · «RECURRENTE ADHESIVO» o «ADHERENTE», también con su carácter cuando
    #     consta.
    #
    # LA ETIQUETA CONCUERDA EN GÉNERO con quien promueve, porque «QUEJOSO Y
    # RECURRENTE» sobre una sociedad rural no lo firma nadie.
    "amparo_revision": [
        ("{QUEJOSO_A} Y RECURRENTE", "quejoso", True),
        ("RECURRENTE ADHESIVO", "adherente", False),
    ],
    # LA QUEJA DICE EL CARÁCTER DE QUIEN RECURRE (3-oct-2026, Q 300/2025). El
    # rótulo era «RECURRENTE» sobre `quejoso`, y cuando recurrió la tercera
    # interesada la carátula puso al quejoso como recurrente. Los 8 engroses del
    # banco dicen el carácter: «QUEJOSO Y RECURRENTE» (4), «RECURRENTE
    # (QUEJOSA/O)» (3), «TERCERA INTERESADA Y RECURRENTE» (1). Con el mismo
    # reparto que la revisión (`filas_caratula`): si recurre otra parte, dos
    # renglones.
    # Y SIN EL RENGLÓN DEL ÓRGANO (C3, 3-oct-2026). Lo dejaba pendiente la D8;
    # David: «lo más práctico… que el secretario no tenga que modificar». Ningún
    # engrose de queja del banco lo lleva (0 de 8), y el órgano ya se nombra en
    # el V I S T O y en la competencia. La ficha de partes lo sigue rotulando
    # (`FIGURAS_FUERA_DEL_RUBRO`).
    "queja": [
        ("{QUEJOSO_A} Y RECURRENTE", "quejoso", True),
    ],
    # «RECURRENTE ADHESIVO» (RF 4/2025: «RECURRENTE ADHESIVO: … (ACTORA)»), como
    # en la revisión.
    # «RECURRENTE», SIN «AUTORIDAD» NI LA SALA (C4, 3-oct-2026). David: «hay que
    # quitar autoridades recurrentes y sala»; el banco del tribunal escribe
    # «REVISIÓN FISCAL: {toca} / RECURRENTE: {recurrente}» en 28 de 28. La Sala
    # se nombra en el V I S T O y en la competencia.
    "revision_fiscal": [
        ("RECURRENTE", "quejoso", True),
        ("RECURRENTE ADHESIVO", "adherente", False),
        ("PARTE ACTORA", "tercero", False),
    ],
}

# EL RUBRO DE ANTES, PARA EL CAMINO SIN LA BANDERA (3-oct-2026). C3 y C4 no son
# [siempre]: `PROCEDENCIA_POR_TIPO=0` es la palanca para volver atrás, y
# volver atrás es volver a estas filas (la queja con su órgano; la revisión
# fiscal con «AUTORIDAD RECURRENTE» y «SALA RESPONSABLE»).
_CARATULA_SIN_BANDERA = {
    "queja": [
        ("{QUEJOSO_A} Y RECURRENTE", "quejoso", True),
        ("ÓRGANO QUE DICTÓ EL AUTO RECURRIDO", "responsable", True),
    ],
    "revision_fiscal": [
        ("AUTORIDAD RECURRENTE", "quejoso", True),
        ("RECURRENTE ADHESIVO", "adherente", False),
        ("PARTE ACTORA", "tercero", False),
        ("SALA RESPONSABLE", "responsable", True),
    ],
}


# ═══════════════════════════════════════════════════════════════════════════
# EL GÉNERO DE LA ETIQUETA
# ═══════════════════════════════════════════════════════════════════════════
# «QUEJOSA Y RECURRENTE» sobre una sociedad, «QUEJOSO Y RECURRENTE» sobre un
# hombre. No se adivina del nombre de pila —eso es exactamente lo que este
# proyecto no hace— sino de la FORMA JURÍDICA cuando consta, y en la duda se
# usa el neutro, que existe y es correcto: «PARTE QUEJOSA Y RECURRENTE».
# «A.C.», «S.C.» e «I.A.P.» son asociación, sociedad e institución: femeninas.
# Sin ellas «Unión Ejemplo, A.C.» (la quejosa del AR 631/2025) salía «PARTE
# QUEJOSA» en la carátula.
_RX_FEMENINO = re.compile(
    r"\bA\.\s*C\.|\bS\.\s*C\.|\bI\.\s*A\.\s*P\.|\bsociedad\b|\bS\.?\s*A\.?\b|\bS\.?\s*de\s*R\.?\s*L\.?\b|"
    r"\basociaci[óo]n\b|\bcooperativa\b|\bempresa\b|\binstituci[óo]n\b|"
    r"\bcomisi[óo]n\b|\bsecretar[íi]a\b|\bdirecci[óo]n\b|\bsala\b|"
    r"\bjunta\b|\bunidad\b|\buniversidad\b|\bfundaci[óo]n\b", re.I)
_RX_MASCULINO = re.compile(
    r"\binstituto\b|\bayuntamiento\b|\bmunicipio\b|\borganismo\b|"
    r"\bconsejo\b|\bbanco\b|\btribunal\b|\bjuzgado\b", re.I)


# EL SUSTANTIVO QUE ENCABEZA EL NOMBRE MANDA (revisión adversarial de la fase
# E, 28-sep-2026). Con el nombre REAL de la quejosa del AR 631/2025 —«la Unión
# de Trabajadores de la Construcción, Transportistas, Materialistas y Similares
# y Anexos del Estado de Querétaro, C.T.M.»— la carátula salía «PARTE
# QUEJOSA»: el arreglo de «A.C.» sólo cubría el nombre sintético de la prueba.
# Y con «Impulsora de Desarrollos Inmobiliarios VV», sin «S.A.», el renglón
# salía «PARTE TERCERA INTERESADA». El género de una persona moral es el de
# su sustantivo de cabeza (la unión, la impulsora, el sindicato); si no hay
# uno conocido, lo de siempre: la forma jurídica, y en la duda el neutro.
#
# LA SUCESIÓN Y LA COMUNIDAD SON FEMENINAS; EL EJIDO, MASCULINO (3-oct-2026, AD
# 335/2025): la carátula decía «PARTE QUEJOSA» de «Sucesión a Bienes de…», y el
# engrose, «QUEJOSA». Y LA CABEZA EN -DORA NO ES UN NOMBRE DE PILA: «Isadora»,
# «Teodora» o «Isidora» acaban como «Comercializadora», y una persona física no
# recibe género por su nombre. Se excluyen los nombres de pila conocidos con
# esa terminación; la cabeza de empresa (impulsora, constructora…) sigue.
_RX_CABEZA_FEM = re.compile(
    r"^(?:(?:la|las)\s+)?(?:uni[óo]n|federaci[óo]n|confederaci[óo]n|liga|central|alianza|"
    r"agrupaci[óo]n|c[áa]mara|coalici[óo]n|organizaci[óo]n|asociaci[óo]n|sociedad|cooperativa|"
    r"empresa|instituci[óo]n|comisi[óo]n|secretar[íi]a|direcci[óo]n|sala|junta|unidad|"
    r"universidad|fundaci[óo]n|inmobiliaria|sucesi[óo]n|comunidad|"
    r"(?!(?:is[ai]|t[h]?eo|eu|helio|leo|ama|nido|sal|ten)dora\b)\w+(?:dora|tora|sora))\b", re.I)
_RX_CABEZA_MASC = re.compile(
    r"^(?:(?:el|los)\s+)?(?:sindicato|instituto|ayuntamiento|municipio|organismo|consejo|banco|"
    r"tribunal|juzgado|frente|grupo|colegio|comit[ée]|patronato|fideicomiso|ejido|"
    r"comisariado|n[úu]cleo)\b", re.I)
# «C.T.M.» (Confederación de Trabajadores de México) y las demás centrales
# obreras al final del nombre: la afiliación es femenina.
_RX_CENTRAL_OBRERA = re.compile(r"\bC\.\s*T\.\s*M\.|\bC\.\s*R\.\s*O\.\s*C\.|\bC\.\s*R\.\s*O\.\s*M\.")


def genero_de(nombre: str) -> str:
    """«a», «o» o «" "» —neutro— para concordar la etiqueta de la carátula."""
    n = (nombre or "").strip()
    if not n:
        return ""
    if _RX_CABEZA_FEM.search(n):
        return "a"
    if _RX_CABEZA_MASC.search(n):
        return "o"
    if _RX_FEMENINO.search(n) or _RX_CENTRAL_OBRERA.search(n):
        return "a"
    if _RX_MASCULINO.search(n):
        return "o"
    return ""


# EL CARGO EN MINÚSCULA TAMBIÉN PIERDE EL ARTÍCULO DE LA PROSA (quinta ronda,
# 3-oct-2026; RF 49/2025 del banco: «AUTORIDAD RECURRENTE: LA TITULAR DE LA
# UNIDAD JURÍDICA…»). El V I S T O del engrose la escribe «la titular de la
# Unidad…», con el cargo en minúscula, y el artículo sólo se quitaba ante una
# mayúscula: «la Jefa…» (RF 7) y «la Titular…» (RF 6/2025) lo perdían y «la
# titular…» no. Ante un cargo se quita igual y el cargo sube su inicial.
_RX_CARGO_EN_MINUSCULA = (
    r"(?:titular|jef[ea]|director|subdirector|delegad|subdelegad|encargad|oficial|fiscal|"
    r"agente|representante|coordinador|administrador|secretari|subsecretari|gobernador|"
    r"president|procurador|subprocurador|tesorer|contralor|gerente|juez|magistrad)\w*\b")


def sin_articulo_de_prosa(nombre: str) -> str:
    """«la Unión …» → «Unión …» para un renglón de la carátula. Sólo el
    artículo en MINÚSCULA, que es el de la prosa («la quejosa, la Unión…»);
    el que va en mayúscula es parte del nombre («La Costeña, S.A.»). Ante un
    CARGO en minúscula también, y el cargo sube su inicial: «la titular de la
    Unidad…» → «Titular de la Unidad…» (quinta ronda)."""
    n = str(nombre or "")
    m = re.match(r"^\s*(?:el|la|los|las)\s+(?=" + _RX_CARGO_EN_MINUSCULA + ")", n)
    if m:
        resto = n[m.end():]
        return resto[:1].upper() + resto[1:]
    return re.sub(r"^\s*(?:el|la|los|las)\s+(?=[A-ZÁÉÍÓÚÑ])", "", n)


# EL GÉNERO DE UNA PERSONA FÍSICA SE LEE DEL PAPEL, NUNCA DEL NOMBRE (3-oct-2026).
# «QUEJOSO:» salía en la carátula de 19 quejosas (errores reales): el rótulo del
# amparo directo era fijo. El nombre de pila no dice el género —y adivinarlo es
# firmar un error—, pero el expediente sí lo dice cuando nombra a la persona con
# su artículo: «la actora Ana López…», «el quejoso Juan…», «la señora…». Sólo
# eso cuenta; si el papel no lo dice, el rótulo va neutro («PARTE QUEJOSA»).
_RX_ROL_CON_ARTICULO = (r"\b(la|el)\s+(?:parte\s+)?(?:quejos[oa]|actor(?:a)?|demandad[oa]|"
                        r"recurrente|tercer[oa]\s+interesad[oa]|promovente|apelante|"
                        r"ciudadan[oa]|reconvencionista|ocursante|accionante)\s*,?\s*")


def genero_en_el_papel(nombre: str, texto: str) -> str:
    """«a», «o» o «» según cómo NOMBRA EL PAPEL a esa persona («la actora X»,
    «el quejoso X», «la señora X»). Si el papel la nombra de las dos formas, «»."""
    n = " ".join(str(nombre or "").split())
    t = " ".join(str(texto or "").split())
    if len(n) < 5 or not t:
        return ""
    nom = re.escape(n[:40]).replace(r"\ ", r"\s+")
    vistos = set()
    for m in re.finditer(_RX_ROL_CON_ARTICULO + nom, t, re.I):
        vistos.add("a" if m.group(1).lower() == "la" else "o")
    for m in re.finditer(r"\b(se[ñn]ora|se[ñn]or)\s+" + nom, t, re.I):
        vistos.add("a" if m.group(1).lower().endswith("a") else "o")
    return vistos.pop() if len(vistos) == 1 else ""


def etiqueta_concordada(etiqueta: str, nombre: str) -> str:
    """Sustituye {QUEJOSO_A} por la forma que concuerda con quien promueve."""
    if "{QUEJOSO_A}" not in (etiqueta or ""):
        return etiqueta
    g = genero_de(nombre)
    forma = {"a": "QUEJOSA", "o": "QUEJOSO"}.get(g, "PARTE QUEJOSA")
    return etiqueta.replace("{QUEJOSO_A}", forma)


def _rige_procedencia() -> bool:
    """¿Rige `procedencia_por_tipo`? False si falta el contexto o la bandera."""
    try:
        import contexto_taller as _ct_r
        _f = getattr(_ct_r, "rige", None)
        return bool(_f("procedencia_por_tipo")) if callable(_f) else False
    except Exception:
        return False


# LO QUE LA TERCERA RONDA AÑADIÓ AL RUBRO VA CON LA BANDERA (FIXES_R3: no es
# [siempre]). Sin ella el catálogo es el de antes para todos sus usos —la
# carátula del .docx, la hoja del prompt, el formulario (/taller/tipos) y las
# figuras que se le piden al lector del auto (`fase_admision.prompt`)—: la queja
# rotula «RECURRENTE» y el amparo directo y la revisión fiscal no traen el
# renglón del adherente.
_RUBRO_SIN_BANDERA = {("queja", "{QUEJOSO_A} Y RECURRENTE"): "RECURRENTE"}


def caratula_de(tipo: str) -> list:
    t = normalizar(tipo)
    t = t if t in CARATULA else "amparo_directo"
    filas = CARATULA[t]
    if _rige_procedencia():
        return filas
    filas = _CARATULA_SIN_BANDERA.get(t, filas)
    return [(_RUBRO_SIN_BANDERA.get((t, et), et), cl, ob) for et, cl, ob in filas
            if not (cl == "adherente" and t in ("amparo_directo", "revision_fiscal"))]


# LAS FIGURAS QUE EXISTEN EN EL TIPO PERO NO VAN EN EL RUBRO. En la revisión el
# juzgado que dictó la sentencia recurrida es una figura del expediente —la
# ficha de partes lo lee y la competencia lo nombra— aunque la carátula no lo
# lleve (0 de 478 en el corpus). Sin esta tabla, quien rotulaba la ficha con
# `caratula_de` caía al «AUTORIDAD RESPONSABLE» del amparo directo y le ponía
# ese nombre al juzgado (AR 631/2025).
# LO MISMO EN LA QUEJA Y LA REVISIÓN FISCAL desde que su rubro no lleva el
# órgano ni la Sala (C3 y C4, 3-oct-2026): sin estas filas, `fase_partes` le
# decía al lector del auto que la figura «no existe en este tipo de asunto» y
# el órgano se quedaba sin leer.
FIGURAS_FUERA_DEL_RUBRO = {
    "amparo_revision": {
        "responsable": "ÓRGANO RECURRIDO (Juzgado de Distrito que dictó la sentencia recurrida)",
    },
    "queja": {
        "responsable": "ÓRGANO QUE DICTÓ EL AUTO RECURRIDO",
    },
    "revision_fiscal": {
        "responsable": ("SALA QUE DICTÓ LA SENTENCIA RECURRIDA (Tribunal Federal de Justicia "
                        "Administrativa)"),
    },
}


def etiqueta_de_figura(tipo: str, clave: str, omision: str = "") -> str:
    """El rótulo de esa figura en este tipo: el de la carátula o, si no va en
    el rubro, el de `FIGURAS_FUERA_DEL_RUBRO`."""
    for et, cl, _ob in caratula_de(tipo):
        if cl == clave:
            return et
    return FIGURAS_FUERA_DEL_RUBRO.get(normalizar(tipo), {}).get(clave, omision)


# ── LA «RESPONSABLE» DEL FORMULARIO NO ES EL ÓRGANO DE LA QUEJA NI LA SALA ──
# (integración, 3-oct-2026, consecuencia de C3 y C4). Mientras la carátula de
# la queja y la de la revisión fiscal llevaban el renglón del órgano o de la
# Sala (clave «responsable»), el campo `responsable` del formulario ERA ese
# órgano: lo tecleaba o lo confirmaba el secretario en ese renglón. Sin el
# renglón, la pantalla sigue mandando en ese campo la «responsable ordenadora»
# que `fase_admision` lee del auto de admisión —«la que dictó el acto
# reclamado»—, que en la queja de la fracción I es la autoridad del amparo
# indirecto y no el juzgado que dictó el auto recurrido. Entonces le ganaba al
# órgano cuando la ficha no lo traía (el V I S T O y la competencia nombraban a
# la responsable del acto como quien dictó el auto, y la fracción del 97 se
# deducía de ella) y, cuando sí lo traía, daba el aviso falso «LA FICHA TRAE
# DOS ÓRGANOS DISTINTOS». Ahora ese campo sólo cuenta como órgano si TIENE la
# forma del órgano de ese tipo: en la queja, un órgano de amparo (Juzgado de
# Distrito, Tribunal Colegiado de Apelación, Tribunal Unitario de Circuito,
# Centro de Justicia Penal Federal) o cualquiera si la queja es de la fracción
# II (ahí quien dicta el auto ES la responsable del amparo directo); en la
# revisión fiscal, una Sala.
_RX_ORGANO_DE_AMPARO = re.compile(
    r"(?i)\bju(?:ez|eza|zgado)\b[^;]*?\bde\s+distrito\b(?!\s+judicial)|"
    r"tribunal\s+colegiado\s+de\s+apelaci|tribunal\s+unitario\s+de\s+circuito|"
    r"centro\s+de\s+justicia\s+penal\s+federal|\bju(?:ez|eza|zgado)\s+federal")


def responsable_es_el_organo(tipo: str, responsable: str, fraccion_97: str = "") -> bool:
    """¿El campo `responsable` del formulario nombra el órgano que dictó lo
    recurrido en este tipo? Amparo directo: sí (es la ordenadora). Amparo en
    revisión: no (el órgano es el juzgado; `responsable` es la del acto). Queja
    y revisión fiscal: sí si su carátula aún lleva el renglón «responsable»
    (sin la bandera); si no, sólo con la forma del órgano de ese tipo (ver
    arriba). `fraccion_97` es la de la ficha («I», «II» o «»)."""
    t = normalizar(tipo)
    r = " ".join(str(responsable or "").split())
    if not r:
        return False
    if t == "amparo_directo":
        return True
    if t not in ("queja", "revision_fiscal"):
        return False
    if any(cl == "responsable" for _e, cl, _o in caratula_de(t)):
        return True
    if t == "queja":
        _fr = str(fraccion_97 or "").upper().replace("FRACCIÓN", "").replace("FR.", "").strip()
        return _fr == "II" or bool(_RX_ORGANO_DE_AMPARO.search(r))
    return bool(re.search(r"(?i)\bsala\b", r))


# EL CARÁCTER DE QUIEN RECURRE, EN EL RÓTULO (medido: SCR/caratulas_revision.md).
# Concuerda con el nombre como el de la quejosa; sin género legible, la forma
# con «PARTE», que es la neutra del propio corpus.
_CARACTER_RECURRENTE = {
    "tercero": {"a": "TERCERA INTERESADA", "o": "TERCERO INTERESADO",
                "": "PARTE TERCERA INTERESADA"},
    "autoridad": {"a": "AUTORIDAD RESPONSABLE", "o": "AUTORIDAD RESPONSABLE",
                  "": "AUTORIDAD RESPONSABLE"},
}


def filas_caratula(tipo: str, datos: dict) -> list:
    """[(etiqueta, clave, valor)] del rubro de este asunto, ya concordadas.

    Una sola fuente para la hoja de datos del prompt y para el .docx, que
    partían la etiqueta cada una a su manera. En la revisión, cuando recurre
    otro que la quejosa (`datos["recurrente"]` no vacío, con
    `datos["papel_recurrente"]` = «tercero» | «autoridad»), son dos renglones:
    «QUEJOSA: …» y «{CARÁCTER} Y RECURRENTE: …» (AR 60/2020; el 631 lo
    recurrió la tercera interesada). La queja y la revisión fiscal ya no
    llevan el renglón del órgano (C3 y C4); sin la bandera, que vuelve a su
    rubro de antes, ese renglón lleva `organo_recurrido` cuando se leyó. La clave del segundo renglón es «recurrente». Las filas vacías y
    no obligatorias no salen."""
    d = dict(datos or {})
    _t = normalizar(tipo)
    # SIN EL ARTÍCULO DE LA PROSA (fase E): la ficha da «la Unión …»; en el
    # renglón de la carátula va el nombre.
    for _k in ("quejoso", "recurrente"):
        if isinstance(d.get(_k), str):
            d[_k] = sin_articulo_de_prosa(d[_k])
    _rec = str(d.get("recurrente") or "").strip()
    _q = str(d.get("quejoso") or "")
    # `quejoso_moral` se calcula sobre lo TECLEADO en el formulario, que en un
    # recurso de otra parte puede ser la recurrente (el 631 guardó a la
    # tercera en «quejoso»): con recurrente aparte sólo cuenta el nombre.
    _fem_q = (bool(d.get("quejoso_moral")) and not _rec) or genero_de(_q) == "a"
    # CON LA BANDERA, EL RÓTULO DEL QUEJOSO DEL AMPARO DIRECTO CONCUERDA O VA
    # NEUTRO (3-oct-2026): el género lo da el papel (`genero_quejoso`, ver
    # `genero_en_el_papel`) o la forma de una persona moral; si nada lo dice,
    # «PARTE QUEJOSA», nunca «QUEJOSO» por omisión.
    _rige_pt = False
    try:
        import contexto_taller as _ct_c
        _rige_pt = bool(getattr(_ct_c, "rige", None) and _ct_c.rige("procedencia_por_tipo"))
    except Exception:
        _rige_pt = False
    _g_decl = str(d.get("genero_quejoso") or "").strip().lower()[:1]
    if _g_decl in ("a", "f"):
        _fem_q = True
    # DOS RENGLONES SÓLO SI RECURRE OTRA PARTE (3-oct-2026, Q 300/2025). Antes
    # bastaba que `recurrente` viniera lleno: si recurría la propia quejosa
    # salían «QUEJOSA: X» y «RECURRENTE: X», la misma persona dos veces. Otra
    # parte es la que tiene papel de tercero, autoridad o Ministerio Público,
    # o —sin papel— la que no es la misma que la quejosa.
    _papel_rec = str(d.get("papel_recurrente") or "").strip().lower()
    # LO QUE LA CARÁTULA NO DICE (cuarta ronda, 3-oct-2026, RF 7 y 49/2025): la
    # numeración y la coletilla de una lista copiada del cuerpo («1) X Y OTROS
    # (YA MENCIONADOS CON ANTERIORIDAD)», que en el rubro no remiten a nada), y
    # el prefijo temporal de la Sala («ACTUAL SALA REGIONAL…»).
    for _k in list(d):
        if isinstance(d.get(_k), str) and _k in ("quejoso", "recurrente", "tercero", "adherente",
                                                  "responsable", "organo_recurrido"):
            d[_k] = sin_coletilla_de_caratula(d[_k], temporal=_k in ("responsable", "organo_recurrido"))
    _rec = str(d.get("recurrente") or "").strip()
    _q = str(d.get("quejoso") or "")
    # VARIAS PERSONAS EN EL RUBRO (E2, cuarta ronda; AD 552/2024: «QUEJOSOS:»).
    # Con la bandera: «QUEJOSOS»/«QUEJOSAS» si el texto dice el género
    # (`genero_de_plural`: «ambos», «todas», o el declarado); si no, «PARTE
    # QUEJOSA», que es neutra. El detector es el único (`es_plural_de_partes`);
    # lo que el compositor ya decidió y expuso en `datos_extra` manda
    # («plural_quejoso»; en el amparo directo también «plural», que es el del
    # promovente; en la revisión y la queja «plural» es el de quien recurre y se
    # lee SÓLO de `datos["procesal"]`, quinta ronda), en `datos` o en
    # `datos["procesal"]`.
    _proc = d.get("procesal") if isinstance(d.get("procesal"), dict) else {}

    def _decidido(*claves):
        """Lo que decidió el compositor (`datos_extra`), si lo expuso."""
        for _c in claves:
            for _src in (d, _proc):
                if isinstance(_src.get(_c), bool):
                    return _src.get(_c)
        return None

    _plural_q = False
    _g_pl = ""
    if _rige_pt and _q:
        _pl_d = _decidido("plural_quejoso", "plural") if _t == "amparo_directo" else _decidido("plural_quejoso")
        _plural_q = bool(_pl_d) if _pl_d is not None else es_plural_de_partes(_q)
        _g_pl = genero_de_plural(_q, _g_decl) if _plural_q else ""
    _plural_rec = False
    if _rige_pt and _rec:
        # EL PLURAL DE QUIEN RECURRE ES EL DEL COMPOSITOR, NO EL DEL RUBRO (quinta
        # ronda, 3-oct-2026; AR 208/2025 del banco: «AUTORIDADES RESPONSABLES Y
        # RECURRENTE: GOBERNADOR DEL ESTADO DE QUERÉTARO»). El documento copiaba
        # en `datos["plural"]` el de la QUEJOSA (dos quejosos) y aquí se leía
        # como el del recurrente: un solo Gobernador salía en plural, y al revés
        # una quejosa con dos terceros recurrentes salía en singular. El del
        # recurrente se lee SÓLO de `procesal["plural"]` (los `datos_extra` del
        # compositor); el de la quejosa, de `plural_quejoso`.
        _pl_r = _proc.get("plural") if _t in ("amparo_revision", "queja") else None
        if not isinstance(_pl_r, bool):
            _pl_r = None
        _plural_rec = bool(_pl_r) if _pl_r is not None else es_plural_de_partes(_rec)
    _et_q_plural = {"a": "QUEJOSAS", "o": "QUEJOSOS"}.get(_g_pl, "PARTE QUEJOSA")
    _otra_parte = False
    if _rec and not _rige_pt:
        # SIN LA BANDERA, COMO ANTES (FIXES_R3: no es [siempre]).
        _otra_parte = True
    elif _rec:
        if _papel_rec in ("tercero", "autoridad", "ministerio_publico"):
            _otra_parte = True
        elif _papel_rec != "quejoso":
            try:
                import promovente as _pv_c
                _otra_parte = not _pv_c.misma_parte(_q, _rec)
            except Exception:
                _otra_parte = " ".join(_q.lower().split()) != " ".join(_rec.lower().split())
    filas = []
    for et, clave, ob in caratula_de(tipo):
        valor = d.get(clave, "")
        # Sólo en el camino nuevo: bandera Y ficha de trámite (una sesión vieja
        # sigue como estaba).
        if _rige_pt and d.get("tramite") and _t == "amparo_directo" and clave == "quejoso" and et == "QUEJOSO":
            if _plural_q:
                filas.append((_et_q_plural, clave, valor))
                continue
            _g = "a" if _fem_q else ("o" if _g_decl in ("o", "m") or genero_de(str(valor or "")) == "o" else "")
            filas.append(({"a": "QUEJOSA", "o": "QUEJOSO"}.get(_g, "PARTE QUEJOSA"), clave, valor))
            continue
        if _otra_parte and clave == "quejoso" and "Y RECURRENTE" in et:
            _et_q = et.replace(" Y RECURRENTE", "")
            if _plural_q and "{QUEJOSO_A}" in _et_q:
                _et_q = _et_q_plural
            else:
                _et_q = ("QUEJOSA" if _fem_q else etiqueta_concordada(_et_q, _q)) \
                    if "{QUEJOSO_A}" in _et_q else _et_q
            filas.append((_et_q, "quejoso", valor))
            _papel = str(d.get("papel_recurrente") or "").strip()
            if _plural_rec:
                _g_r = genero_de_plural(_rec)
                _car = _CARACTER_RECURRENTE_PLURAL.get(_papel, {}).get(_g_r, "")
                filas.append(((f"{_car} Y RECURRENTES" if _g_r else f"{_car} Y RECURRENTE")
                              if _car else ("RECURRENTES" if _g_r else "PARTE RECURRENTE"),
                              "recurrente", _rec))
                continue
            _car = _CARACTER_RECURRENTE.get(_papel, {}).get(genero_de(_rec), "")
            filas.append((f"{_car} Y RECURRENTE" if _car else "RECURRENTE",
                          "recurrente", _rec))
            continue
        if (clave == "responsable" and _t != "amparo_revision"
                and str(d.get("organo_recurrido") or "").strip()):
            filas.append((etiqueta_concordada(et, ""), clave, d.get("organo_recurrido")))
            continue
        if not (valor or ob):
            continue
        # EL GÉNERO DE LA QUEJOSA ES SUYO: el renglón del adherente («{QUEJOSO_A}
        # ADHERENTE») concuerda con el nombre del adherente, no con el de ella.
        if "{QUEJOSO_A}" in et and _plural_q and clave == "quejoso":
            # «QUEJOSOS Y RECURRENTES» / «PARTE QUEJOSA Y RECURRENTE».
            _et_p = et.replace("{QUEJOSO_A}", _et_q_plural)
            if _g_pl:
                _et_p = _et_p.replace(" Y RECURRENTE", " Y RECURRENTES")
            filas.append((_et_p, clave, valor))
            continue
        if "{QUEJOSO_A}" in et and _fem_q and clave == "quejoso":
            filas.append((et.replace("{QUEJOSO_A}", "QUEJOSA"), clave, valor))
            continue
        filas.append((etiqueta_concordada(et, str(valor or "")), clave, valor))
    if _rige_pt and _t == "revision_fiscal":
        filas = _filas_rf_concordadas(filas)
    return filas


# ═══ LA CARÁTULA DE LA REVISIÓN FISCAL, CONCORDADA (revisión RF, 3-oct-2026) ═══
# Tres cosas que el tribunal no escribe y el rubro de C4 sí:
#   · VARIAS AUTORIDADES RECURREN («RECURRENTES:», RF 33/2024, pareja del AD
#     469/2024 del banco; 10 de 137 carátulas del corpus): el rótulo iba fijo en
#     singular.
#   · LA ACTORA ES LA RECURRENTE ADHESIVA (la adhesiva sólo la interpone quien
#     obtuvo sentencia favorable): el mismo nombre salía en «RECURRENTE
#     ADHESIVO:» y en «PARTE ACTORA:». El corpus lo escribe una vez (17 de 17):
#     el RF 4/2025 del banco, «RECURRENTE ADHESIVO: ***** (ACTORA)».
#   · MUCHOS ACTORES: «PARTE ACTORA:» copiaba la lista entera (65 nombres en el
#     RF 7/2025). Con más de dos, el primero «Y OTROS», como el corpus (RF
#     42/2023); la lista completa va en el resultando del juicio.
_RX_CABEZA_DE_CARGO = re.compile(r"^(?:(?:el|la|los|las)\s+)?" + _RX_CARGO_EN_MINUSCULA, re.I)
_RX_CORTE_DE_REPRESENTACION = re.compile(
    r",?\s+(?:por\s+conducto\s+de|en\s+representaci[óo]n\s+de|representad[oa]s?\s+por)\b", re.I)


def varias_autoridades(texto: str) -> bool:
    """¿El texto nombra a VARIAS autoridades por su cargo («Secretario de
    Hacienda…, Jefe del Servicio… y Titular de la Administración…»)? Cuenta los
    tramos que abren con un cargo, antes de «por conducto de» o «en
    representación de». `es_plural_de_partes` no sirve aquí: un órgano público
    es una sola parte para él."""
    n = " ".join(str(texto or "").split())
    if not n:
        return False
    n = _RX_CORTE_DE_REPRESENTACION.split(n, maxsplit=1)[0]
    tramos = [x.strip() for x in re.split(r",\s*|\s+[yYeE]\s+", n) if x.strip()]
    return sum(1 for x in tramos if _RX_CABEZA_DE_CARGO.match(x)) >= 2


# La coma de «Empresa A, S.A. de C.V.» no separa a dos personas.
_RX_TRAMO_DE_LISTA = re.compile(
    r",\s*(?!(?:S\.?\s*A\.?|S\.?\s*C\.?|A\.?\s*C\.?|S\.?\s*de\s|SAPI|SOCIEDAD|Sociedad)\b)|"
    r"\s+[yYeE]\s+(?=[A-ZÁÉÍÓÚÑ0-9*])")
_MAS_DE_N_ACTORES = 2


def primero_y_otros(lista: str) -> str:
    """«A, B, C y D» → «A Y OTROS» si nombra a más de dos personas; la lista tal
    cual si no (o si no se puede partir con seguridad)."""
    n = " ".join(str(lista or "").split()).strip(" .;,")
    if not n or not es_plural_de_partes(n):
        return lista
    tramos = [x.strip(" .;,") for x in _RX_TRAMO_DE_LISTA.split(n) if x.strip(" .;,")]
    if len(tramos) <= _MAS_DE_N_ACTORES:
        return lista
    return f"{tramos[0]} Y OTROS"


def _filas_rf_concordadas(filas: list) -> list:
    """Las filas de la revisión fiscal con el plural de quien recurre, la actora
    adhesiva en un solo renglón y la lista de actores recortada (ver arriba)."""
    adh = next((str(v) for _e, c, v in filas if c == "adherente" and str(v or "").strip()), "")
    act = next((str(v) for _e, c, v in filas if c == "tercero" and str(v or "").strip()), "")
    misma = False
    if adh and act:
        try:
            import promovente as _pv_rf
            misma = _pv_rf.misma_parte(adh, act)
        except Exception:
            misma = " ".join(adh.lower().split()) == " ".join(act.lower().split())
    out = []
    for et, c, v in filas:
        if c == "quejoso" and et == "RECURRENTE" and varias_autoridades(str(v or "")):
            out.append(("RECURRENTES", c, v))
            continue
        if c == "adherente" and misma:
            _g = genero_de(adh)
            out.append(("RECURRENTE ADHESIVA" if _g == "a" else "RECURRENTE ADHESIVO", c,
                        f"{str(v).strip().rstrip('.')} ({'ACTOR' if _g == 'o' else 'ACTORA'})"))
            continue
        if c == "tercero" and misma:
            continue
        if c == "tercero":
            out.append((et, c, primero_y_otros(str(v or ""))))
            continue
        out.append((et, c, v))
    return out


_CARACTER_RECURRENTE_PLURAL = {
    "tercero": {"a": "TERCERAS INTERESADAS", "o": "TERCEROS INTERESADOS",
                "": "PARTE TERCERA INTERESADA"},
    "autoridad": {"a": "AUTORIDADES RESPONSABLES", "o": "AUTORIDADES RESPONSABLES",
                  "": "AUTORIDADES RESPONSABLES"},
}

# TODAS LAS NUMERACIONES DE LA LISTA, NO SÓLO LA PRIMERA (quinta ronda,
# 3-oct-2026; RF 7/2025 del banco: «PARTE ACTORA: PARTE ACTORA, 2) PARTE ACTORA,
# 3) PARTE ACTORA, … 66) PARTE ACTORA.»). La regex estaba anclada al principio y
# el renglón quedaba con una lista que empezaba en «2)». La prosa
# (`resultandos_por_tipo.nombre_en_prosa`) ya las quitaba todas.
_RX_NUMERACION_INICIAL = re.compile(r"^\s*\d{1,3}\)\s*")
_RX_NUMERACION_DE_LISTA = re.compile(r"(?<![\w.])\d{1,3}\)\s*")
_RX_COLETILLA_CARATULA = re.compile(
    r"\s*\(\s*(?:ya\s+|antes\s+)?(?:mencionad|citad|señalad|precisad|referid)[oa]s?"
    r"(?:\s+(?:con\s+anterioridad|anteriormente|al\s+rubro|en\s+el\s+rubro))?\s*\)", re.I)
_RX_PREFIJO_TEMPORAL = re.compile(r"^\s*(?:(?:la|el)\s+)?(?:actual|entonces|otrora|ahora|antes)\s+(?=\S)", re.I)
# «; POR PROPIO DERECHO» TAMPOCO ES EL NOMBRE (E1; RF 2/2025 del banco: «PARTE
# ACTORA: … ; POR PROPIO DERECHO.»). Se quita la fórmula, también la de varios
# («AMBOS POR PROPIO DERECHO»), y la coma que dejaba ante la «Y» del siguiente
# («X, POR PROPIO DERECHO, Y Z» → «X Y Z»). NO se quita si sigue la
# representación de otro («X, POR PROPIO DERECHO Y EN REPRESENTACIÓN DE SU MENOR
# HIJA…»): ahí la fórmula dice quién más es parte.
_RX_POR_PROPIO_DERECHO = re.compile(
    r"\s*[,;]\s*(?:(?:amb[oa]s|tod[oa]s)\s+)?por\s+(?:su\s+)?propio\s+derecho\b"
    r"(?!\s*,?\s+(?:y|e)\s+(?:en\s+|como\s+|por\s+)?(?:representaci|nombre|apoderad|albacea|tutor|"
    r"su\s|sus\s|de\s))"
    r"(?:\s*,(?=\s*(?:y|e)\s))?(?:\s*[.;]\s*$)?", re.I)


def sin_coletilla_de_caratula(valor: str, temporal: bool = False) -> str:
    """«1) X, 2) Y Y 3) Z (YA MENCIONADOS CON ANTERIORIDAD); POR PROPIO DERECHO»
    → «X, Y Y Z»; con `temporal`, «ACTUAL SALA REGIONAL…» → «SALA REGIONAL…».
    Lo demás, igual."""
    v = str(valor or "")
    if not v.strip():
        return v
    v2 = _RX_COLETILLA_CARATULA.sub("", _RX_NUMERACION_DE_LISTA.sub("", v))
    v2 = _RX_POR_PROPIO_DERECHO.sub("", v2)
    if temporal:
        v2 = _RX_PREFIJO_TEMPORAL.sub("", v2)
    v2 = " ".join(v2.split())
    return v2 or v


# ═══════════════════════════════════════════════════════════════════════════
# EL PROEMIO
# ═══════════════════════════════════════════════════════════════════════════
# «Visto apócrifo: dice VISTO, para resolver el juicio de amparo directo».
# Salía en la queja y en la revisión fiscal, y la causa es la de siempre en
# este proyecto: el prompt no DESCRIBÍA la fórmula, la ESCRIBÍA. El campo
# `visto` del JSON de ejemplo decía «para resolver el juicio de amparo
# directo…», y un modelo con un ejemplo concreto delante lo copia y le cambia
# los datos. Es la tercera vez que un ejemplo del prompt se firma literal.
#
# Y EL RÓTULO ES PLURAL EN LA REVISIÓN FISCAL: el corpus abre «V I S T O S,
# para resolver el recurso de revisión fiscal número…», 28 de 28. El
# compositor lo imprimía en singular a máquina y además lo limpiaba con una
# regex que admite la S, así que aunque el modelo acertara, se le borraba: la
# revisión fiscal era incorregible por prompt.
PROEMIO = {
    "amparo_directo": {"rotulo": "V I S T O, ",
                       "molde": "para resolver el juicio de amparo directo {materia} {numero}, promovido por {promovente}"},
    "amparo_revision": {"rotulo": "V I S T O, ",
                        "molde": "para resolver el recurso de revisión {numero}, interpuesto por {promovente}"},
    "queja": {"rotulo": "V I S T O, ",
              "molde": "para resolver el recurso de queja {materia} {numero}, interpuesto por {promovente}"},
    "revision_fiscal": {"rotulo": "V I S T O S, ",
                        "molde": "para resolver el recurso de revisión fiscal número {numero}, interpuesto por la parte citada al rubro"},
}


def proemio_de(tipo: str) -> dict:
    return PROEMIO.get(normalizar(tipo), PROEMIO["amparo_directo"])


# ═══════════════════════════════════════════════════════════════════════════
# EL INCISO DEL ARTÍCULO 97: UN MARCADOR CON DOS SIGNIFICADOS
# ═══════════════════════════════════════════════════════════════════════════
# La queja salía fundada en «el artículo 97, fracción I, inciso c)» y el
# adelanto real de la QC 259/2025 dice inciso e). No es que el valor fuera
# malo: es que el marcador `{inciso}` sirve a DOS plantillas que quieren decir
# cosas incompatibles.
#
#   · en el amparo directo abre el 107, fracción V, constitucional y el 35,
#     fracción I, de la Ley Orgánica, que SÍ se reparten por materia —b) en
#     administrativa y agraria, c) en civil y mercantil—;
#   · en la queja abre el 97, fracción I, de la Ley de Amparo, cuyos incisos se
#     reparten por QUÉ SE RECURRE: el desechamiento de la demanda, la
#     suspensión, el carácter de tercero interesado…
#
# Darle el valor de la materia al segundo es fundar la procedencia del recurso
# en un supuesto que no es el suyo, y eso se caza en sesión.
#
# SÓLO SE MAPEA LO QUE ALGUIEN PUEDE RESPALDAR: el corpus —«queja de
# desechamiento, inciso a), 16 de 30»— o el secretario que firma, y en ese caso
# se anota que fue él. Lo que no se puede afirmar sale en HUECO VISIBLE con su
# aviso: un inciso equivocado se firma, un hueco se rellena. No es pereza —es
# que inventar el inciso de un precepto de procedencia es justo lo que este
# trabajo existe para evitar.
_INCISO_97 = [
    (r"desech[óo]?\s+(?:de\s+plano\s+)?(?:total\s+o\s+parcialmente\s+)?la\s+demanda"
     r"|desech[óo]?\s+la\s+demanda|tuvo\s+por\s+no\s+presentada\s+la\s+demanda"
     r"|admit[ií][óo]?\s+.{0,30}\s+demanda\s+de\s+amparo", "a"),
    (r"suspensi[óo]n\s+(?:de\s+plano|provisional)|conced[ií][óo]?\s+la\s+suspensi[óo]n"
     r"|neg[óo]?\s+la\s+suspensi[óo]n", "b"),
    (r"car[áa]cter\s+de\s+tercero\s+interesado", "d"),
    # EL INCISO e) LO DIO DAVID, y con su razón: «tratándose de un auto que
    # desecha un incidente de nulidad de notificaciones dictado TRAS la
    # sentencia de amparo indirecto, el fundamento correcto es el artículo 97,
    # fracción I, inciso e), de la Ley de Amparo». Es el supuesto de las
    # resoluciones dictadas después de la sentencia que, por su naturaleza
    # trascendental y grave, pueden causar un perjuicio no reparable.
    #
    # No estaba en el corpus del banco —ahí sólo se contó el desechamiento de
    # la demanda, inciso a), 16 de 30— y por eso salía en hueco. Lo aporta él,
    # que es quien firma, y se anota de dónde viene para que se pueda discutir.
    # «nulidad de notificación» en singular, y «de actuaciones»: el resultando
    # real decía «el incidente de nulidad de notificación de emplazamiento» y
    # la exigencia del plural lo dejaba fuera.
    # LA REPOSICIÓN DE AUTOS NO VA AQUÍ (verificación de normas, 27-sep-2026):
    # la resolución que decide el incidente de reposición de constancias se
    # recurre en REVISIÓN, artículo 81, fracción I, inciso c); el e) del 97 es
    # sólo para lo que no admite expresamente la revisión.
    (r"incidente\s+de\s+nulidad\s+de\s+(?:notificaci[óo]n(?:es)?|actuaciones)"
     r"|dictad[ao]s?\s+despu[ée]s\s+de\s+(?:la\s+)?sentencia", "e"),
]


def inciso_97(descripcion_acto: str) -> str:
    """El inciso del 97, fracción I, o cadena vacía si no se puede afirmar.

    Se lee de la DESCRIPCIÓN DEL ACTO —una frase—, nunca del OCR entero: una
    heurística de una palabra dentro de cien mil caracteres casa siempre, y con
    lo primero de la lista.
    """
    t = " ".join((descripcion_acto or "").split())
    # El tope sube de 400 a 1,600: ahora se le pasa la descripción MÁS los
    # resultandos, que son cuatro párrafos. Sigue siendo un texto acotado y
    # escrito para este asunto, no el expediente pegado —que es donde una
    # heurística de una palabra casa siempre y con lo primero de la lista—.
    if not t or len(t) > 1600:
        return ""
    for patron, inciso in _INCISO_97:
        if re.search(patron, t, re.I):
            return inciso
    return ""


# ═══════════════════════════════════════════════════════════════════════════
# CÓMO SE NOMBRA A LOS SUJETOS EN LA PROSA
# ═══════════════════════════════════════════════════════════════════════════
# Las listas de sujetos vivían escritas a mano en `fases123_resumenes.py`,
# medidas sobre engroses de AMPARO DIRECTO, y se entregaban a los cuatro tipos.
# Peor: el prompt entregaba `SUJETOS_PARTE[:3]`, que son las tres variantes de
# «quejoso» y ninguna de «recurrente». De ahí salieron, palabra por palabra:
#
#     «En el primer agravio la quejosa aduce…»          (amparo en revisión)
#     «En el primer agravio la quejosa sostiene…»       (revisión fiscal: el SAT)
#     «En el único agravio, la quejosa se duele de…»    (queja)
#     «la responsable recibió…»                          (queja: el Juzgado)
#
# EL EJE ERA UN BOOLEANO. Los prompts recibían `es_recurso: bool`, que sólo
# abre dos caminos donde hacen falta cuatro: los tres recursos entraban por la
# misma rama y esa rama sólo cambiaba «conceptos de violación» por «agravios».
#
# EL MATIZ QUE NO SE PUEDE PERDER: en un amparo EN REVISIÓN, al narrar el juicio
# de amparo indirecto DE ORIGEN, «la quejosa» sí es correcto —lo fue allí— y el
# adelanto real de David escribe «promovido por la quejosa Bertha Alicia Luna
# Flores». Lo que no es correcto es llamarla quejosa al sintetizar SUS AGRAVIOS.
# Por eso esto gobierna la prosa del RECURSO, y la narración del origen puede
# seguir usando su vocabulario. En revisión fiscal, en cambio, «quejosa» no es
# correcto en ningún sitio: nunca hubo amparo.
SUJETOS = {
    "amparo_directo": {
        "parte": ("el quejoso", "la quejosa", "la parte quejosa"),
        "organo": ("la Sala", "la Sala responsable", "la autoridad responsable",
                   "la responsable"),
        "bisagra": ("En contra de esas consideraciones, la parte quejosa "
                    "plantea los siguientes conceptos de violación:"),
    },
    "amparo_revision": {
        "parte": ("la parte recurrente", "el recurrente", "la recurrente"),
        # El Juez de Distrito NO es autoridad responsable en el recurso: es el
        # órgano de control cuya sentencia se revisa. Llamarlo «responsable»
        # convierte en parte a quien no lo es.
        "organo": ("el Juzgado de Distrito", "el juez de amparo",
                   "el órgano de control constitucional", "el a quo"),
        "bisagra": ("En contra de las anteriores consideraciones, la parte "
                    "recurrente formula los agravios siguientes:"),
    },
    "queja": {
        "parte": ("la parte recurrente", "el recurrente", "la recurrente"),
        "organo": ("el Juzgado de Distrito", "el juez de amparo",
                   "el órgano que dictó el auto recurrido", "el a quo"),
        "bisagra": ("En contra de las anteriores consideraciones, la parte "
                    "recurrente formula los agravios siguientes:"),
    },
    "revision_fiscal": {
        # «la autoridad recurrente», no «la quejosa»: el SAT nunca fue quejoso.
        "parte": ("la autoridad recurrente", "la recurrente",
                  "la autoridad demandada en el juicio de nulidad"),
        # Aquí «la Sala responsable» SÍ es correcta: así la nombra el corpus de
        # revisión fiscal, sobre 28 expedientes.
        "organo": ("la Sala", "la Sala responsable", "la Sala regional",
                   "la responsable"),
        "bisagra": ("En contra de las anteriores consideraciones, la autoridad "
                    "recurrente formula los agravios siguientes:"),
    },
}


def sujetos_de(tipo: str) -> dict:
    """Cómo se nombra a la parte y al órgano en la prosa del tipo.

    AMPARO DIRECTO, 30-sep-2026: el órgano ya no es «la Sala» por decreto. Si
    en esta petición rige `instancia_origen` y el contexto trae el origen del
    acto (`origen_acto`), se le nombra por lo que es —«el Juez responsable»,
    «la Junta», «la Sala»— o con una fórmula neutra si no consta. Una docena
    de prompts beben de aquí; ése era el origen de «la Sala responsable» dicho
    de un Juzgado de oralidad mercantil (AD 323/2025)."""
    t = normalizar(tipo)
    base = SUJETOS.get(t, SUJETOS["amparo_directo"])
    if t == "amparo_directo" or t not in SUJETOS:
        suj = _sujetos_del_origen()
        if suj:
            return {**base, "organo": suj}
    return base


def _sujetos_del_origen() -> tuple | None:
    try:
        import contexto_taller as _ct
        o = _ct.origen()
        if not o or not _ct.rediseno("instancia_origen"):
            return None
        s = tuple(o.get("sujetos") or ())
        return s if len(s) == 4 and all(isinstance(x, str) and x for x in s) else None
    except Exception:
        return None


def cumplimiento_actual() -> dict:
    """{consta, ejecutoria, efectos} si lo reclamado se dictó en cumplimiento de
    una ejecutoria de amparo y rige `cumplimiento_ejecutoria`; {} si no. Los
    prompts de narración (antecedentes, relato) lo consultan para CONTARLO:
    que un amparo anterior mandó y ésta es la sentencia que lo acató."""
    try:
        import contexto_taller as _ct
        o = _ct.origen()
        if not o or not _ct.rediseno("cumplimiento_ejecutoria"):
            return {}
        c = o.get("cumplimiento") or {}
        return dict(c) if c.get("consta") else {}
    except Exception:
        return {}


def instancia_actual() -> str:
    """«unica», «alzada» o «» del acto de esta petición (sólo si rige
    `instancia_origen`). Los prompts de amparo directo la consultan para no
    narrar una apelación que no hubo."""
    try:
        import contexto_taller as _ct
        o = _ct.origen()
        if not o or not _ct.rediseno("instancia_origen"):
            return ""
        return str(o.get("instancia") or "")
    except Exception:
        return ""


# ═══════════════════════════════════════════════════════════════════════════
# QUÉ SE RECURRIÓ, SEGÚN EL INCISO DEL 97
# ═══════════════════════════════════════════════════════════════════════════
# La plantilla de procedencia de la queja terminaba, en duro, «por el cual se
# desechó la demanda de amparo». Es cierto en el caso mayoritario —el inciso
# a), 16 de 30— y se emitía con CUALQUIER inciso, así que con el e) la frase se
# contradecía a sí misma: invocaba el supuesto de las resoluciones dictadas
# después de la sentencia y a renglón seguido afirmaba que se había desechado
# la demanda. Lo cazó David en la QC 259/2025, donde la demanda se admitió en
# 2023 y ya había sentencia firme confirmada en revisión.
#
# LA REDACCIÓN DEL INCISO e) ES SUYA, palabra por palabra: «por tratarse de una
# resolución dictada por un Juzgado de Distrito con posterioridad a la
# sentencia definitiva en el juicio de amparo indirecto, que no admite recurso
# de revisión». Se anota de quién viene, como el propio inciso.
COLA_97 = {
    "a": "por el cual se desechó la demanda de amparo",
    "b": "en el que se resolvió sobre la suspensión del acto reclamado",
    # LOS INCISOS QUE EL LECTOR DE PROSA NO DEDUCE (3-oct-2026) llegan ahora
    # de la ficha de trámite; su cola dice lo que dice el inciso, sin más.
    "c": "en el que se resolvió sobre la admisión de fianzas o contrafianzas",
    "d": "en el que se resolvió sobre el carácter de tercero interesado",
    "e": ("por tratarse de una resolución dictada con posterioridad a la "
          "sentencia definitiva en el juicio de amparo indirecto, que no "
          "admite recurso de revisión"),
    "f": "por el que se decidió el incidente de reclamación de daños y perjuicios",
    "g": ("por el que se resolvió el incidente por exceso o defecto en la "
          "ejecución de la suspensión del acto reclamado"),
    "h": "dictado en el incidente de cumplimiento sustituto de la sentencia de amparo",
}

# LA FRACCIÓN II: ACTOS DE LA AUTORIDAD RESPONSABLE EN EL AMPARO DIRECTO
# (3-oct-2026). Sólo se conocía la fracción I, así que una queja contra la Sala
# que proveyó sobre la suspensión en un amparo directo se fundaba en el inciso
# b) de la fracción I —la suspensión del amparo INDIRECTO—, con su plazo de dos
# días cuando el suyo es de cinco (oro_Q: 41 contra 4 en el corpus).
COLA_97_II = {
    "a": "por el que se omitió tramitar la demanda de amparo o se tramitó indebidamente",
    "b": "en el que la autoridad responsable proveyó sobre la suspensión del acto reclamado",
    "c": "por el que se resolvió el incidente de reclamación de daños y perjuicios",
    "d": "en el que se resolvió sobre la libertad caucional del quejoso",
}


def cola_97(inciso: str, fraccion: str = "I") -> str:
    """Qué se recurrió, en la redacción que corresponde al inciso (y a la
    fracción del 97: «I», la de siempre, o «II», amparo directo)."""
    tabla = COLA_97_II if (fraccion or "").strip().upper() == "II" else COLA_97
    return tabla.get((inciso or "").strip().lower().rstrip(")"), "")


# ═══════════════════════════════════════════════════════════════════════════
# LA MATERIA QUE INVOCA LA PARTE Y NO ES LA DEL ASUNTO
# ═══════════════════════════════════════════════════════════════════════════
# Esto no nace de un defecto del proyecto sino de una duda que el proyecto no
# supo despejar. En el ARC 25/2026 la síntesis decía que la recurrente pide
# suplencia «al afirmar que se encuentran involucrados derechos de una persona
# adulta mayor y que la controversia corresponde a un proceso penal», en un
# juicio SUCESORIO. David preguntó lo único que había que preguntar: ¿lo citó
# ella o lo alucinó la máquina?
#
# Lo citó ella. Su escrito dice, literal: «el diverso 79, fracción II, de la
# Ley de Amparo… que en materia penal opera aun ante la ausencia de agravios» y
# «además de darme el carácter de imputado, es que se patentiza que la litis se
# refiere a un proceso penal». La síntesis fue fiel.
#
# PERO EL PROYECTO NO LO DIJO, y ahí está el hueco: quien lee no puede
# distinguir el disparate de la parte del disparate de la máquina, y ante la
# duda tiene que ir al escrito. Peor: una suplencia invocada por una materia
# que no es la del asunto NO es una curiosidad, es una petición que hay que
# contestar —normalmente declarándola improcedente— y el proyecto pasaba de
# largo. El aviso hace las dos cosas: confirma de dónde viene y recuerda que
# pide respuesta.
_MATERIAS_SUPLENCIA = {
    "penal": r"materia\s+penal|proceso\s+penal|car[áa]cter\s+de\s+imputad",
    "laboral": r"materia\s+(?:laboral|de\s+trabajo)|en\s+favor\s+del\s+trabajador",
    "agraria": r"materia\s+agraria|n[úu]cleo\s+de\s+poblaci[óo]n\s+ejidal",
}


def suplencia_de_otra_materia(texto: str, materia: str) -> list:
    """Materias de suplencia invocadas que no son la del asunto."""
    t = texto or ""
    if not re.search(r"suplencia", t, re.I):
        return []
    m = (materia or "").strip().lower()
    return [nombre for nombre, pat in _MATERIAS_SUPLENCIA.items()
            if nombre != m and re.search(pat, t, re.I)]


# ═══════════════════════════════════════════════════════════════════════════
# LAS TRES RAMAS DEL AMPARO EN REVISIÓN
# ═══════════════════════════════════════════════════════════════════════════
# David las dicta, y el corpus del propio tribunal las tiene medidas sobre 62
# expedientes —estaban en `banco_plantillas.json` sin que nadie las leyera—.
# Hasta hoy el resolutivo de la revisión era UNA sola frase, «Se confirma la
# sentencia recurrida», y con ella el proyecto nunca amparaba, nunca modificaba
# efectos y nunca ordenaba reponer.
#
#   RAMA 1 — LEVANTAMIENTO DE SOBRESEIMIENTO (art. 93, frs. I y V). El a quo
#   sobreseyó indebidamente; se destruye la causal, se levanta el
#   sobreseimiento y, en plenitud de jurisdicción, se estudian por primera vez
#   los conceptos de violación que aquél omitió. Resolutivo DOBLE. (Decía «fr.
#   I» a secas: la I manda examinar los agravios contra el sobreseimiento; el
#   estudio de fondo que sigue lo manda la V —«revocará la sentencia recurrida
#   y dictará la que corresponda»—. Cotejado con el texto vigente de
#   normas_ley_de_amparo.json el 28-sep-2026, AR 631/2025.)
#
#   RAMA 2 — FONDO (art. 93, frs. V y VI). Tres supuestos: revocar la negativa
#   —se concede—, revocar la concesión y modificar los EFECTOS, que es el único
#   de los tres con resolutivo ÚNICO. REVOCAR LA CONCESIÓN NO ES NEGAR SIN MÁS
#   (fr. VI): si recurre la autoridad o la tercera interesada y sus agravios de
#   fondo son fundados, el tribunal «analizará los conceptos de violación no
#   estudiados y concederá o negará el amparo». Ver `reasuncion`.
#
#   RAMA 3 — VIOLACIÓN PROCESAL DENTRO DEL AMPARO (art. 93, fr. IV). El único
#   supuesto en que se devuelven los autos al Juzgado para reponer el
#   procedimiento del amparo indirecto. Resolutivo ÚNICO.
#
# LA REGLA MEDIDA QUE NO SE PUEDE PERDER: «en revisión el resolutivo tiene DOS
# puntos, no uno: PRIMERO decide sobre la sentencia recurrida —confirma,
# modifica, revoca— y SEGUNDO reproduce el sentido del amparo —ampara, no
# ampara, sobresee—. Sólo hay ÚNICO cuando se desecha el recurso.» Está escrita
# en el banco, con sus frecuencias: «SEGUNDO. La Justicia de la Unión» 20/62.
#
# Y LA OTRA, que evita un error de renumeración: el SEGUNDO remite al
# considerando DE LA RESOLUCIÓN RECURRIDA —«el considerando segundo de la
# resolución que se revisa»—, no a los de esta ejecutoria. El compositor no lo
# renumera.
RAMAS_REVISION = {
    # ── El recurso no prospera ────────────────────────────────────────────
    "confirma_niega": {
        "fundamento": "artículo 93, fracciones V y VI, de la Ley de Amparo",
        "puntos": [
            "PRIMERO. Se confirma la sentencia recurrida.",
            # EL ACTO Y LA AUTORIDAD, NOMBRADOS (C5, 3-oct-2026). David: «sí
            # está bien que se precise… información valiosa para el lector».
            # {actos_del_amparo} lo llena `punto_del_amparo` con los actos y
            # las autoridades de la demanda (la ficha); sin ellos, la remisión
            # de siempre (`ACTOS_DEL_AMPARO_GENERICO`). Y «de la resolución
            # recurrida» en vez de «de la misma», que con el acto en medio ya
            # no se sabe a qué remite.
            "SEGUNDO. La Justicia de la Unión no ampara ni protege a {quejoso}, "
            "{actos_del_amparo}, por las razones expuestas en el último "
            "considerando de la resolución recurrida."],
        "frecuencia": "8/62 confirma · 8/62 no ampara ni protege",
    },
    "confirma_concede": {
        "fundamento": "artículo 93, fracciones V y VI, de la Ley de Amparo",
        "puntos": [
            "PRIMERO. Se confirma la sentencia recurrida.",
            # C5: el acto y la autoridad, como en confirma_niega.
            "SEGUNDO. La Justicia de la Unión ampara y protege a {quejoso}, "
            "{actos_del_amparo}, en términos del último considerando de la "
            "resolución recurrida."],
        "frecuencia": "18/62 ampara y protege",
    },
    "confirma_sobresee": {
        "fundamento": "artículo 93, fracciones V y VI, de la Ley de Amparo",
        "puntos": [
            # «RECURRIDA», COMO SUS DOS HERMANAS. Esta rama decía «impugnada» y
            # las otras dos del mismo bloque —confirma_niega y confirma_concede,
            # con el mismo verbo «Se confirma»— dicen «recurrida». El estudio
            # también escribe «recurrida». Así que el proyecto de la revisión
            # 410/2026 salió con la misma sentencia llamada de dos maneras: el
            # estudio cerraba «se confirma la sentencia recurrida» y el
            # resolutivo decía «Se confirma la sentencia impugnada».
            #
            # No es sinónimo inocuo en un resolutivo: es el punto que se
            # ejecuta, y nombrar dos veces distinto el mismo acto es lo que un
            # revisor marca a la primera lectura.
            "PRIMERO. Se confirma la sentencia recurrida.",
            # SIN ORDINALES DE UNA RESOLUCIÓN AJENA (3-oct-2026, AR 239/2025).
            # Decía «los actos que quedaron precisados en el considerando
            # segundo de la resolución que se revisa y por las razones
            # expuestas en el último considerando de la misma»: dos ordinales de
            # la sentencia del juzgado que la ficha no conoce. Valen en
            # Querétaro (el AR 448/2025 los usa), no en toda la república. La
            # forma de sus hermanas de rama, que remite sin numerar.
            # C5: el acto y la autoridad, con «contra» (la preposición que
            # esta rama ya usaba).
            "SEGUNDO. Se sobresee en el presente juicio de amparo, promovido "
            "por {quejoso}, {actos_del_amparo}, por las razones expuestas en "
            "la resolución recurrida."],
        "frecuencia": "10/62 sobresee",
    },
    # ── EL JUZGADO HIZO LAS DOS COSAS: sobreseyó un acto y resolvió el fondo
    # por los demás. Amparo en revisión 322/2025: «sobreseyó … respecto de la
    # orden de restitución … y negó el amparo respecto del auto de diecinueve
    # de febrero». Con un solo verbo el resolutivo confirmaba un sobreseimiento
    # total y callaba la negativa, que era lo recurrido. Tres puntos, como los
    # escribe el circuito cuando confirma una sentencia mixta.
    "confirma_sobresee_niega": {
        "fundamento": "artículo 93, fracciones V y VI, de la Ley de Amparo",
        "puntos": [
            "PRIMERO. Se confirma la sentencia recurrida.",
            "SEGUNDO. Se sobresee en el juicio de amparo promovido por "
            "{quejoso}, respecto del acto precisado en la resolución recurrida "
            "y por las razones expuestas en la misma.",
            "TERCERO. La Justicia de la Unión no ampara ni protege a {quejoso}, "
            "respecto de los demás actos precisados en la resolución recurrida, "
            "por las razones expuestas en el último considerando de la misma."],
        "aviso": ("LA SENTENCIA RECURRIDA ES MIXTA: sobreseyó respecto de un "
                  "acto y negó el amparo por los demás, y el resolutivo lo "
                  "reproduce en dos puntos. Comprueba contra el resolutivo del "
                  "juzgado qué acto quedó sobreseído y cuáles negados."),
    },
    "confirma_sobresee_concede": {
        "fundamento": "artículo 93, fracciones V y VI, de la Ley de Amparo",
        "puntos": [
            "PRIMERO. Se confirma la sentencia recurrida.",
            "SEGUNDO. Se sobresee en el juicio de amparo promovido por "
            "{quejoso}, respecto del acto precisado en la resolución recurrida "
            "y por las razones expuestas en la misma.",
            "TERCERO. La Justicia de la Unión ampara y protege a {quejoso}, "
            "respecto de los demás actos precisados en la resolución recurrida, "
            "en términos del último considerando de la misma."],
        "aviso": ("LA SENTENCIA RECURRIDA ES MIXTA: sobreseyó respecto de un "
                  "acto y concedió el amparo por los demás, y el resolutivo lo "
                  "reproduce en dos puntos. Comprueba contra el resolutivo del "
                  "juzgado qué acto quedó sobreseído y cuáles amparados."),
    },
    # ── RAMA 1: se levanta el sobreseimiento y se asume jurisdicción ──────
    "revoca_sobreseimiento_concede": {
        "fundamento": "artículo 93, fracciones I y V, de la Ley de Amparo",
        "puntos": [
            "PRIMERO. Se revoca la sentencia recurrida.",
            "SEGUNDO. La Justicia de la Unión ampara y protege a {quejoso}, "
            "contra el acto reclamado a {responsable_originaria}, por los "
            "motivos y fundamentos expuestos en el último considerando de esta "
            "ejecutoria."],
        # El estudio tiene que decir que asume jurisdicción y por qué puede.
        "plenitud": True,
    },
    "revoca_sobreseimiento_niega": {
        "fundamento": "artículo 93, fracciones I y V, de la Ley de Amparo",
        "puntos": [
            "PRIMERO. Se revoca la sentencia recurrida.",
            "SEGUNDO. La Justicia de la Unión no ampara ni protege a {quejoso}, "
            "contra el acto reclamado a {responsable_originaria}, por los "
            "motivos y fundamentos expuestos en el último considerando de esta "
            "ejecutoria."],
        "plenitud": True,
    },
    # ── RAMA 2 A y B: se revoca el fondo ─────────────────────────────────
    "revoca_fondo_concede": {
        "fundamento": "artículo 93, fracciones V y VI, de la Ley de Amparo",
        "puntos": [
            "PRIMERO. Se revoca la sentencia recurrida.",
            "SEGUNDO. La Justicia de la Unión ampara y protege a {quejoso}, "
            "contra el acto reclamado a {responsable_originaria}, por los "
            "motivos y fundamentos expuestos en el último considerando de esta "
            "ejecutoria."],
        "plenitud": True,
    },
    "revoca_fondo_niega": {
        "fundamento": "artículo 93, fracciones V y VI, de la Ley de Amparo",
        "puntos": [
            "PRIMERO. Se revoca la sentencia recurrida.",
            "SEGUNDO. La Justicia de la Unión no ampara ni protege a {quejoso}, "
            "contra el acto reclamado a {responsable_originaria}, por los "
            "motivos y fundamentos expuestos en el último considerando de esta "
            "ejecutoria."],
        "plenitud": True,
    },
    # ── RAMA 2 D: prospera la IMPROCEDENCIA que alegó quien recurre ────────
    # (revisión del 28-sep-2026, AR 631/2025). La fracción VI del artículo 93
    # es para los agravios DE FONDO; si lo que prospera es el agravio de la
    # autoridad o de la tercera interesada contra la omisión o la negativa a
    # sobreseer (fr. II), se revoca y se sobresee: no hay conceptos de
    # violación que reasumir, y negar supondría haber estudiado el fondo.
    "revoca_sobresee": {
        "fundamento": "artículos 93, fracciones II y III, y 63, fracción V, de la Ley de Amparo",
        "puntos": [
            "PRIMERO. Se revoca la sentencia recurrida.",
            "SEGUNDO. Se sobresee en el juicio de amparo promovido por {quejoso}, "
            "respecto del acto reclamado a {responsable_originaria}, por los "
            "motivos y fundamentos expuestos en el último considerando de esta "
            "ejecutoria."],
    },
    # ── RAMA 2 C: sólo los efectos. ÚNICO, y es el matiz que lo distingue:
    # el amparo estuvo BIEN concedido; lo que falla es la restitución.
    "modifica_efectos": {
        "fundamento": "artículo 93, fracciones V y VI, de la Ley de Amparo",
        "puntos": [
            "ÚNICO. Se modifica la sentencia recurrida, únicamente para los "
            "efectos precisados en el último considerando de esta ejecutoria."],
        "frecuencia": "4/62 modifica",
    },
    # ── RAMA 3: reposición del procedimiento DEL AMPARO ──────────────────
    "repone_procedimiento": {
        "fundamento": "artículo 93, fracción IV, de la Ley de Amparo",
        "puntos": [
            "ÚNICO. Se revoca la sentencia recurrida y se ordena la reposición "
            "del procedimiento en el juicio de amparo indirecto {expediente}, "
            "para los efectos precisados en el último considerando de esta "
            "resolución."],
    },
    # ── El recurso se desecha ────────────────────────────────────────────
    # ── EL RECURSO SE QUEDÓ SIN OBJETO ───────────────────────────────────
    #
    # Medido en el circuito: 644 expedientes se resuelven así —434 quejas y 267
    # revisiones—, y no estaba en este mapa. El caso arquetípico de la queja es
    # el de la suspensión: se recurre la negativa de la PROVISIONAL y, antes de
    # resolver la queja, se dicta la DEFINITIVA; la provisional sólo rige hasta
    # entonces, así que el recurso se queda sin objeto.
    #
    # En revisión: se recurre la interlocutoria que negó la suspensión
    # definitiva y la sentencia del amparo causa ejecutoria antes de resolver.
    #
    # NO ES DESECHAR NI CONFIRMAR. Desechar es rechazar por improcedente desde
    # el origen; aquí el recurso era procedente y algo posterior le quitó la
    # materia. Y confirmar exigiría estudiar unos agravios que ya no tienen
    # objeto.
    "sin_materia": {
        "fundamento": "artículo 93 de la Ley de Amparo; cesación de la materia "
                      "del recurso por hecho superveniente",
        "puntos": [
            "ÚNICO. Se declara sin materia el recurso de revisión interpuesto "
            "por {quejoso}, por las razones expuestas en el último "
            "considerando de esta resolución."],
        "frecuencia": "267/5,092 revisiones del circuito · 434/2,955 quejas",
    },
    "desecha": {
        "fundamento": "artículo 86 de la Ley de Amparo",
        "puntos": [
            "ÚNICO. Se desecha el recurso de revisión, al resultar improcedente "
            "por extemporáneo, por los motivos y fundamentos expuestos en el "
            "considerando último de la presente ejecutoria."],
    },
    # ── Y EL HUECO DELIBERADO, que es como lo deja David en sus adelantos.
    # Medido: «Se ********** la sentencia impugnada». Cuando no consta qué
    # resolvió el a quo, inventarlo sería peor que dejarlo a la vista.
    "sin_determinar": {
        "fundamento": "artículo 93 de la Ley de Amparo",
        "puntos": [
            "PRIMERO. Se {HUECO} la sentencia recurrida.",
            "SEGUNDO. {HUECO} a {quejoso}, respecto de los actos precisados en "
            "la resolución recurrida."],
        "aviso": ("NO CONSTA QUÉ RESOLVIÓ EL JUZGADO DE DISTRITO —si sobreseyó, "
                  "negó o concedió—, así que el resolutivo sale con el hueco a "
                  "la vista, como en tus adelantos. Sin ese dato no se puede "
                  "saber si procede confirmar, revocar o modificar."),
    },
}


# ═══ EL PUNTO DEL AMPARO NOMBRA ACTO Y AUTORIDAD (C5, 3-oct-2026) ═══════════
# David: «sí está bien que se precise… información valiosa para el lector». Las
# tres ramas que confirman (`confirma_niega`, `confirma_concede`,
# `confirma_sobresee`) remitían a «los actos precisados en la resolución
# recurrida», y quien lee el resolutivo se quedaba sin saber contra qué se
# amparó o se negó. La ficha de trámite ya trae los actos y las autoridades de
# la demanda de amparo indirecto, anclados al papel, y el primer resultando de
# la ejecutoria los copia en renglones: el punto los nombra o remite a ESE
# resultando, que es de esta ejecutoria y sí se puede numerar. Sin ficha (o sin
# nada útil en ella), la remisión de siempre. Las ramas mixtas y
# `sin_determinar` no cambian: ahí el reparto de actos lo decide el juzgado.
ACTOS_DEL_AMPARO_GENERICO = {
    "respecto de": "respecto de los actos precisados en la resolución recurrida",
    "contra": "contra los actos precisados en la resolución recurrida",
}

# Un acto más largo que esto no cabe en un punto resolutivo: se remite al
# resultando que lo copia.
_PALABRAS_ACTO_EN_PUNTO = 30
_RX_SOLO_HUECO = re.compile(r"^[\s*_.…\-–—()\[\]]*$")
_RX_VINETA_ACTO = re.compile(r"^\s*(?:\d{1,3}[.)]|[a-zA-Z][.)](?=\s)|[-•·])\s*")


def _contraer_articulo(t: str) -> str:
    """«de el» → «del», «a el» → «al» (no ante «el que»). Sólo el artículo en
    minúscula: «de El Universal» es un nombre propio y se queda."""
    t = re.sub(r"\ba\s+el\b(?!\s+que\b)", "al", t or "")
    return re.sub(r"\bde\s+el\b(?!\s+que\b)", "del", t)


def _lista_de_textos(x) -> list:
    if isinstance(x, str):
        return [x]
    try:
        return [v for v in list(x or []) if isinstance(v, str)]
    except TypeError:
        return []


def _acto_en_punto(a: str) -> str:
    """El acto como entra en el punto: sin la viñeta, sin el punto final y
    sin espacios de más; «» si sólo es hueco («*********»)."""
    t = " ".join(str(a or "").split())
    t = _RX_VINETA_ACTO.sub("", t, count=1)
    t = t.strip().rstrip(" .;,:").strip()
    if not t or _RX_SOLO_HUECO.match(t):
        return ""
    if not re.search(r"[^\W\d_]", re.sub(r"\*+", "", t)):
        return ""
    return t


def _inicial_minuscula(t: str) -> str:
    """«La resolución…» → «la resolución…». Una sigla o una palabra en
    versales al principio («SAT», «LA SENTENCIA…») se dejan como vienen, y
    también el nombre propio de una norma («Ley de Hacienda…»)."""
    m = re.match(r"\S+", t or "")
    if not m:
        return t
    letras = [c for c in m.group(0) if c.isalpha()]
    if len(letras) >= 2 and all(c.isupper() for c in letras):
        return t
    if _es_nombre_de_norma(t):
        return t
    return t[:1].lower() + t[1:]


# EL ACTO SIN ARTÍCULO NO SE CITA ASÍ (revisión AR, 3-oct-2026). «consistente en
# artículo 90 de la Ley…», «consistente en ley de Hacienda…» o «consistente en
# sentencia de diez de mayo…»: la ficha copia el acto como lo escribe la demanda,
# a veces sin artículo y con la mayúscula del renglón. Si la primera palabra no
# es un determinante, se le pone el artículo que le toca; el nombre propio de
# una norma («Ley», «Código», «Reglamento», «Decreto», «Constitución», «Acuerdo
# General…») conserva su mayúscula. Si no se sabe qué artículo lleva, se remite
# al resultando que copia la demanda, que es lo que ya se hacía con los largos.
_DETERMINANTES_ACTO = {
    "el", "la", "los", "las", "un", "una", "unos", "unas", "lo", "su", "sus", "este", "esta",
    "estos", "estas", "ese", "esa", "esos", "esas", "dicho", "dicha", "dichos", "dichas", "al",
    "del", "todo", "toda", "todos", "todas", "aquel", "aquella", "cualquier", "diversos", "diversas"}
_NORMAS_CON_NOMBRE = {
    "ley": "la", "leyes": "las", "código": "el", "codigo": "el", "reglamento": "el",
    "decreto": "el", "constitución": "la", "constitucion": "la", "acuerdo": "el",
    "lineamientos": "los", "norma": "la", "reglas": "las", "estatuto": "el"}
_ARTICULO_DEL_SUSTANTIVO = {
    "artículo": "el", "articulo": "el", "artículos": "los", "articulos": "los",
    "sentencia": "la", "resolución": "la", "resolucion": "la", "auto": "el", "autos": "los",
    "orden": "la", "embargo": "el", "cobro": "el", "pago": "el", "multa": "la", "oficio": "el",
    "acta": "el", "requerimiento": "el", "negativa": "la", "clausura": "la", "laudo": "el",
    "fallo": "el", "proveído": "el", "proveido": "el", "providencia": "la", "interlocutoria": "la",
    "juicio": "el", "procedimiento": "el", "crédito": "el", "credito": "el", "boleta": "la",
    "remate": "el", "secuestro": "el", "citatorio": "el", "baja": "la", "cese": "el", "retiro": "el",
    "descuento": "el", "acuerdo": "el", "acuerdos": "los", "actos": "los", "acto": "el",
    "omisiones": "las", "decreto": "el"}
_RX_ACUERDO_COMUN = re.compile(r"^acuerdos?\s+(?:de\s+(?:fecha|\d|[a-záéíóú]+\s+de\s+[a-záéíóú]+)|dictad|emitid|pronunciad|que\b)",
                               re.I)


def _es_nombre_de_norma(t: str) -> bool:
    """¿El texto abre con el nombre propio de una norma («Ley de Hacienda…»,
    «Código Fiscal…», «Acuerdo General 5/2013…»)? «acuerdo de diez de mayo» es
    un acto del juzgado, no un nombre."""
    m = re.match(r"([^\W\d_]+)", t or "")
    if not m or not m.group(1)[:1].isupper():
        return False
    w = m.group(1).lower()
    if w not in _NORMAS_CON_NOMBRE:
        return False
    return not (w.startswith("acuerdo") and _RX_ACUERDO_COMUN.match(t))


def _acto_con_articulo(t: str) -> str:
    """El acto como entra tras «consistente en», con su artículo; «» si no se
    sabe cuál lleva (quien llama remite al resultando)."""
    m = re.match(r"([^\W\d_]+)", t or "")
    if not m:
        return ""
    w = m.group(1)
    wl = w.lower()
    if wl in _DETERMINANTES_ACTO:
        return _inicial_minuscula(t)
    letras = [c for c in w if c.isalpha()]
    if len(letras) >= 2 and all(c.isupper() for c in letras):
        return ""                                   # una sigla: no se adivina
    if _es_nombre_de_norma(t):
        return f"{_NORMAS_CON_NOMBRE[wl]} {t}"
    art = _ARTICULO_DEL_SUSTANTIVO.get(wl)
    if not art:
        if re.search(r"(?:ci[óo]n|si[óo]n|dad|tad|tud|ez)$", wl):
            art = "las" if wl.endswith("es") else "la"
        elif re.search(r"(?:ciones|siones|dades)$", wl):
            art = "las"
        elif wl.endswith("miento"):
            art = "el"
        elif wl.endswith("mientos"):
            art = "los"
    if not art:
        return ""
    return f"{art} {t[:1].lower() + t[1:]}"


# «DICTADA POR LA AUTORIDAD RESPONSABLE» SOBRA EN EL PUNTO (revisión AR, 3-oct-2026,
# AR 60/2025): el punto ya nombra a cada autoridad, y con dos la remisión es
# ambigua. Se quita al citar el acto.
_RX_POR_LA_RESPONSABLE = re.compile(
    r"\s+por\s+(?:la\s+(?:propia\s+)?(?:autoridad\s+)?(?:se[ñn]alada\s+como\s+)?responsable|"
    r"las\s+(?:propias\s+)?(?:autoridades\s+)?responsables|dicha\s+autoridad)(?=[\s,.;:]|$)", re.I)


def norma_sin_acto(actos=None, autoridades=None) -> bool:
    """¿La demanda reclama una norma que la ficha no trae entre los actos? El
    Poder Legislativo está entre las responsables —sólo está ahí cuando se
    reclama una norma— y ningún acto es una norma (revisión AR, 3-oct-2026, AR
    72/2025: el único acto de la ficha era el de aplicación y el punto se lo
    atribuía a la Legislatura y al Gobernador, cuando el engrose concede contra
    el artículo 90 de la Ley de Hacienda y su acto de aplicación)."""
    _auts = _lista_de_textos(autoridades)
    if not any(_RX_LEGISLATIVO.search(str(a or "")) for a in _auts):
        return False
    for a in _lista_de_textos(actos):
        for linea in re.split(r"[\n;]", str(a or "")):
            # «La aprobación, expedición y promulgación de la Ley…» (AR 201/2025)
            # también es la norma, dicha por sus actos legislativos.
            if _RX_ACTO_ES_NORMA.search(linea) or _RX_CONSTITUCIONALIDAD.search(linea):
                return False
    return True


def _en_versales(t: str) -> bool:
    """¿El texto viene entero en mayúsculas (más que una sigla)?"""
    letras = [c for c in (t or "") if c.isalpha()]
    return len(letras) >= 8 and all(c.isupper() for c in letras)


def _autoridad_en_punto(a: str) -> str:
    """La autoridad con su artículo y en prosa (`con_articulo_de_organo`);
    «» si sólo es hueco."""
    t = " ".join(str(a or "").split()).strip(" .;,:")
    if not t or _RX_SOLO_HUECO.match(t):
        return ""
    try:
        r = con_articulo_de_organo(t)
    except Exception:
        r = t
    r = " ".join(str(r or t).split())
    return r if not _RX_SOLO_HUECO.match(r) else ""


def _de_cada_autoridad(auts: list) -> str:
    """«de la Segunda Sala… y del Juzgado Mixto…»: cada una tras su «de», con
    la contracción. Si alguna trae coma, las separa el punto y coma."""
    xs = [_contraer_articulo("de " + a) for a in auts]
    if len(xs) == 1:
        return xs[0]
    sep = "; " if any("," in a for a in auts) else ", "
    return sep.join(xs[:-1]) + " y " + xs[-1]


def punto_del_amparo(actos: list, autoridades: list, ordinal: str = "primero",
                     plural_quejoso: bool = False, preposicion: str = "respecto de") -> str:
    """Lo que va en {actos_del_amparo} del punto que confirma (C5).

      · un solo acto de hasta 30 palabras: «respecto del acto que reclamó de
        la Segunda Sala… y del Juzgado Mixto…, consistente en la sentencia…»;
      · varios actos, o uno más largo: «respecto de los actos que reclamó de
        …, precisados en el resultando primero de esta ejecutoria»;
      · sin autoridades: «respecto de los actos precisados en el resultando
        primero de esta ejecutoria»;
      · sin actos útiles (vacío o sólo huecos): «», y quien llama pone
        `ACTOS_DEL_AMPARO_GENERICO[preposicion]`.

    `ordinal` es el del resultando que copia la demanda («primero»);
    `plural_quejoso`, «reclamaron»; `preposicion`, «respecto de» o «contra»
    (la del sobreseimiento)."""
    prep = " ".join(str(preposicion or "").split()) or "respecto de"
    xs = []
    for a in _lista_de_textos(actos):
        x = _acto_en_punto(a)
        if x and x not in xs:
            xs.append(x)
    if not xs:
        return ""
    # LA NORMA QUE LA FICHA NO TRAE: nombrar sólo el acto de aplicación y
    # atribuírselo al Legislativo firma un punto falso. La remisión genérica es
    # vaga pero cierta; quien llama avisa (`norma_sin_acto`).
    if norma_sin_acto(xs, autoridades):
        return ""
    auts, vistas = [], set()
    for a in _lista_de_textos(autoridades):
        x = _autoridad_en_punto(a)
        k = x.lower()
        if x and k not in vistas:
            vistas.add(k)
            auts.append(x)
    uno = len(xs) == 1
    el = "el acto" if uno else "los actos"
    precisado = "precisado" if uno else "precisados"
    ordn = " ".join(str(ordinal or "").split()).lower() or "primero"
    remision = f"{precisado} en el resultando {ordn} de esta ejecutoria"
    if auts:
        de = _de_cada_autoridad(auts)
        verbo = "reclamaron" if plural_quejoso else "reclamó"
        # UN ACTO EN VERSALES NO SE CITA: «consistente en LA SENTENCIA…» no se
        # firma, y pasarlo a minúsculas perdería los nombres propios. Se remite
        # al resultando, que lo copia como viene.
        _acto_c = ""
        if (uno and len(xs[0].split()) <= _PALABRAS_ACTO_EN_PUNTO
                and not _en_versales(xs[0])):
            _acto_c = _acto_con_articulo(_RX_POR_LA_RESPONSABLE.sub("", xs[0]).strip(" ,;"))
        if _acto_c:
            t = f"{prep} el acto que {verbo} {de}, consistente en {_acto_c}"
        else:
            t = f"{prep} {el} que {verbo} {de}, {remision}"
    else:
        t = f"{prep} {el} {remision}"
    return _contraer_articulo(t)


def preposicion_del_amparo(rama: str = "", punto: str = "") -> str:
    """«contra» en el sobreseimiento (`confirma_sobresee`, o un punto que dice
    «Se sobresee»); «respecto de» en las demás."""
    if (rama or "").strip() == "confirma_sobresee" or re.search(r"\bSe\s+sobresee\b", punto or ""):
        return "contra"
    return "respecto de"


def con_generico_del_amparo(punto: str, preposicion: str = "") -> str:
    """El punto con la remisión genérica en lugar de {actos_del_amparo}, y la
    cola como era antes de C5 (revisión AR, 3-oct-2026). Con el acto nombrado
    la cola dice «de la resolución recurrida», porque «de la misma» ya no se
    sabe a qué remite; con la remisión genérica la oración decía dos veces
    «resolución recurrida» («…respecto de los actos precisados en la
    resolución recurrida, por las razones expuestas en el último considerando
    de la resolución recurrida»), y en la que concede aparecía una remisión a
    los actos que nunca llevó. Sin actos, el punto de siempre:
      · confirma_niega: «…respecto de los actos precisados en la resolución
        recurrida, por las razones expuestas en el último considerando de la
        misma»;
      · confirma_concede: «…ampara y protege a X, en términos del último
        considerando de la resolución recurrida»;
      · confirma_sobresee: «…contra los actos precisados en la resolución
        recurrida y por las razones expuestas en la misma»."""
    p = punto or ""
    if "{actos_del_amparo}" not in p:
        return p
    prep = preposicion or preposicion_del_amparo("", p)
    g = ACTOS_DEL_AMPARO_GENERICO.get(prep) or ACTOS_DEL_AMPARO_GENERICO["respecto de"]
    p2 = re.sub(r",\s*\{actos_del_amparo\},(\s+en\s+t[ée]rminos\s+del\s+[úu]ltimo\s+considerando)", r",\1", p)
    if p2 != p:
        return p2
    p = p.replace("{actos_del_amparo}, por las razones expuestas en el último considerando de la "
                  "resolución recurrida", g + ", por las razones expuestas en el último considerando "
                  "de la misma")
    p = p.replace("{actos_del_amparo}, por las razones expuestas en la resolución recurrida",
                  g + " y por las razones expuestas en la misma")
    return p.replace("{actos_del_amparo}", g)


def con_actos_del_amparo(punto: str, actos=None, autoridades=None, ordinal: str = "primero",
                         plural_quejoso: bool = False, rama: str = "") -> str:
    """El punto con {actos_del_amparo} ya lleno: `punto_del_amparo` si hay
    actos útiles; si no, la remisión genérica con la cola de siempre
    (`con_generico_del_amparo`). Para quien no tiene ficha (la tarjeta de
    decisión) basta `con_actos_del_amparo(punto)`."""
    p = punto or ""
    if "{actos_del_amparo}" not in p:
        return p
    prep = preposicion_del_amparo(rama, p)
    x = punto_del_amparo(actos or [], autoridades or [], ordinal, plural_quejoso, prep)
    return p.replace("{actos_del_amparo}", x) if x else con_generico_del_amparo(p, prep)


# ═══════════════════════════════════════════════════════════════════════════
# LAS CALIFICACIONES, Y UN SOLO SITIO DONDE SE PREGUNTA SI PROSPERAN
# ═══════════════════════════════════════════════════════════════════════════
# Medido sobre 65,282 agravios calificados del Vigésimo Segundo Circuito. Las
# cuatro de siempre cubren el 92%; el 8% restante son estas otras, y la que más
# pesa —«esencialmente fundado»— es el 23% de los agravios en las revisiones
# que REVOCAN, o sea justo donde más caro sale equivocarse.
#
# POR QUÉ UN PREDICADO Y NO UNA CADENA. Antes, en siete sitios distintos, se
# preguntaba `sentido.startswith("fundad")`. Con «fundado» funciona; con
# «esencialmente_fundado» —que empieza por «esencialmente»— devuelve False en
# todos, y uno de esos sitios es el que decide si el amparo SE CONCEDE. Añadir
# la calificación sin cambiar esto habría convertido concesiones en negativas
# sin que nada fallara.
#
# Ahora se pregunta aquí y sólo aquí. Añadir la siguiente calificación es
# tocar una línea.
CALIFICACIONES = {
    # clave                    plural                  prospera  medido
    "fundado":                 ("fundados",             True,   9022),
    "infundado":               ("infundados",           False, 27839),
    "inoperante":              ("inoperantes",          False, 16927),
    "ineficaz":                ("ineficaces",           False,  5965),
    "esencialmente_fundado":   ("esencialmente fundados", True,  2899),
    # NO PROSPERA NI SE DESESTIMA: se queda sin objeto. Un agravio sin materia
    # no se contesta en el fondo porque ya no hay nada que contestar —lo que
    # atacaba dejó de existir—. Medido: 259 agravios, y 186 de ellos en asuntos
    # cuyo SENTIDO entero es «sin materia».
    "sin_materia":             ("sin materia",            False,   259),

    # ── LAS CUATRO QUE FALTABAN. Si prosperan o no NO se decidió de oído: se
    #    contó en qué sentidos aparece cada una. Las bases, para leerlo:
    #    «fundado» sale en asuntos a favor el 87% de las veces, «infundado» el
    #    11% y «inoperante» el 8%.
    #
    #    INATENDIBLE (24%). Cerca del grupo que no prospera. Es el
    #    planteamiento que no puede atenderse —no por lo que dice, sino por
    #    cómo o cuándo se dice—.
    "inatendible":             ("inatendibles",           False,   713),

    #    FUNDADO PERO INSUFICIENTE (12%). AQUÍ ME HABRÍA EQUIVOCADO. Lleva
    #    «fundado» en el nombre y el respaldo del predicado lo daba por bueno,
    #    pero se comporta EXACTAMENTE como el infundado —12% frente a su 11%—:
    #    el planteamiento tiene razón y aun así no alcanza para mover el
    #    sentido, porque subsisten otras consideraciones que lo sostienen.
    #    Tratarlo como fundado habría revocado sentencias que se confirman.
    "fundado_insuficiente":    ("fundados pero insuficientes", False, 554),

    #    PARCIALMENTE FUNDADO (80%). Prospera, y de forma característica: 141
    #    de sus 365 apariciones están en asuntos que conceden PARCIALMENTE.
    "parcialmente_fundado":    ("parcialmente fundados",  True,    365),

    #    SUSTANCIALMENTE FUNDADO (97%). La que más claramente prospera de
    #    todas, por encima del propio «fundado».
    "sustancialmente_fundado": ("sustancialmente fundados", True,   124),
}

# Lo que el secretario puede elegir hoy en pantalla. Se separa del diccionario
# porque añadir una calificación al catálogo y ofrecerla en la interfaz son dos
# decisiones distintas: la primera es reconocerla, la segunda es pedirla.
# LAS NUEVE, POR ORDEN DE LO QUE HACEN: primero las que sacan adelante el
# recurso, luego las que no, y al final la que lo deja sin objeto.
SENTIDOS_OFRECIDOS = ("fundado", "esencialmente_fundado",
                      "sustancialmente_fundado", "parcialmente_fundado",
                      "fundado_insuficiente", "infundado", "inoperante",
                      "inatendible", "ineficaz", "sin_materia")

# Cómo se llama cada una en pantalla. Va aquí para que la interfaz no tenga que
# saber castellano jurídico ni adivinar el plural.
ETIQUETA_SENTIDO = {
    "fundado": "Fundado",
    "esencialmente_fundado": "Esencialmente fundado",
    "infundado": "Infundado",
    "inoperante": "Inoperante",
    "ineficaz": "Ineficaz",
    "sin_materia": "Sin materia",
    "sustancialmente_fundado": "Sustancialmente fundado",
    "parcialmente_fundado": "Parcialmente fundado",
    "fundado_insuficiente": "Fundado pero insuficiente",
    "inatendible": "Inatendible",
}


def prospera(sentido: str) -> bool:
    """¿Este planteamiento saca adelante el recurso o el amparo?

    EL ÚNICO SITIO DONDE SE DECIDE. De aquí cuelgan el sentido del fallo, la
    rama del resolutivo, la suerte de los accesorios y si el amparo se concede.
    Un `startswith` esparcido por siete ficheros es una bomba de relojería:
    basta una calificación nueva que no empiece por «fundad» para que todos
    devuelvan False a la vez y en silencio.
    """
    s = (sentido or "").strip().lower().replace(" ", "_")
    if s in CALIFICACIONES:
        return CALIFICACIONES[s][1]
    # EL RESPALDO, PARA LO QUE AÚN NO ESTÁ EN EL CATÁLOGO. Lleva una excepción
    # que costó medir: «fundado_insuficiente» tiene «fundad» dentro y NO
    # prospera —12% de apariciones a favor, contra el 11% del infundado—. La
    # regla ingenua lo habría dado por bueno y habría revocado sentencias que
    # se confirman.
    if "insuficien" in s or "inoperan" in s or s.startswith("in"):
        return False
    return "fundad" in s


def plural_de(sentido: str) -> str:
    s = (sentido or "").strip().lower().replace(" ", "_")
    return CALIFICACIONES.get(s, (s, False, 0))[0]


# ═══════════════════════════════════════════════════════════════════════════
# LA TÉCNICA DE RESOLUCIÓN, POR ESCENARIO
# ═══════════════════════════════════════════════════════════════════════════
# David: «hay múltiples escenarios técnicos en revisión… dependiendo del asunto
# y la resolución, la técnica de resolución cambia conforme a las reglas de la
# ley de amparo. Por eso es importante contar con un cuadro de "if"».
#
# QUÉ ES ESTO Y QUÉ NO ES. `RAMAS_REVISION` dice cómo quedan los RESOLUTIVOS.
# Esto dice qué hay que ESTUDIAR para llegar ahí, que es distinto y hasta ahora
# no estaba en ninguna parte: el prompt del estudio no mencionaba ni una vez
# «levantar el sobreseimiento», ni «plenitud de jurisdicción», ni «mayor
# beneficio».
#
# CADA REGLA LLEVA SU FUENTE. Lo que viene de la Ley de Amparo se cita con su
# artículo; lo que viene de la práctica medida de este tribunal se marca como
# tal. Ninguna regla se escribe «de memoria»: en esto una imprecisión no es una
# errata, es un vicio del proyecto.
TECNICA_RESOLUCION = {

    # ── REVISIÓN: se levanta el sobreseimiento ────────────────────────────
    "revision_levanta_sobreseimiento": {
        "cuando": "El recurso es FUNDADO y el Juzgado de Distrito había "
                  "SOBRESEÍDO.",
        # I y V (28-sep-2026): la I es la de los agravios contra el
        # sobreseimiento; el estudio de fondo que sigue lo manda la V.
        "fuente": "artículo 93, fracciones I y V, de la Ley de Amparo",
        "tecnica": [
            "NO HAY REENVÍO. El tribunal colegiado no devuelve el asunto al "
            "Juzgado de Distrito: levanta el sobreseimiento y asume "
            "jurisdicción para resolver lo que aquél no resolvió.",
            "SE ABRE UN CONSIDERANDO NUEVO para el estudio de los CONCEPTOS DE "
            "VIOLACIÓN, que hasta ahora nadie había examinado. Es un análisis "
            "de primera vez, no una revisión de lo que dijo el juzgado.",
            "Ese estudio tiene su propio desenlace: los conceptos pueden "
            "resultar fundados, infundados o inoperantes, y de ahí sale si se "
            "ampara o no se ampara. Que el AGRAVIO sea fundado sólo prueba que "
            "el juzgado no debió sobreseer; no dice nada sobre el fondo.",
        ],
        "necesita": "conceptos_de_violacion",
        "aviso_si_falta":
            "Se va a levantar el sobreseimiento, así que hay que estudiar los "
            "conceptos de violación — y no constan. Sólo aparecen en el "
            "expediente si la sentencia recurrida los relató. Súbelos o "
            "escríbelos antes de generar: sin ellos el proyecto levanta el "
            "sobreseimiento y no resuelve nada.",
    },

    # ── REVISIÓN: se revoca una CONCESIÓN y se reasume jurisdicción ──────
    # AR 631/2025 (28-sep-2026): el juzgado concedió con el segundo concepto y
    # declaró innecesarios los demás; recurrió la tercera interesada y el
    # recurso prosperó. El proyecto pasó de «procede revocarla» a «no ampara ni
    # protege» sin estudiar un solo concepto de los que el juzgado no estudió:
    # el «no ampara» quedó sin considerando que lo sostuviera. Es la fracción
    # VI del artículo 93, y la Segunda Sala ya lo decía de la ley anterior
    # (2a./J. 113/2007, registro 171925, en el acervo: «…debe analizar los
    # conceptos de violación cuyo estudio omitió el juez de distrito, sin
    # importar quién interponga el recurso»).
    #
    # LOS APOYOS SE COMPROBARON EN EL ACERVO el 28-sep-2026, con su rubro y sin
    # marca de pérdida de vigencia: 171925 (2a./J., la reasunción), 178784
    # (J. de colegiado: inoperantes los conceptos que descansan en lo ya
    # desestimado), 182039 (aislada, lo mismo de los agravios) y 174177 (1a./J.:
    # lo no impugnado de la sentencia se declara firme). La 3a./J. 20/91
    # (registro 207016) NO está en el acervo y no se cita. Se traen por su
    # registro al resolver y el prompt sólo nombra los que llegaron
    # (`solo_si_estan`).
    "revision_reasume_concesion": {
        "cuando": "El Juzgado de Distrito CONCEDIÓ el amparo (o sobreseyó "
                  "respecto de un acto y concedió por otro), recurre la "
                  "autoridad responsable o la parte tercera interesada, y sus "
                  "agravios de fondo PROSPERAN.",
        "fuente": "artículo 93, fracción VI, de la Ley de Amparo",
        "tecnica": [
            "REVOCAR NO ES NEGAR. Que los agravios prosperen prueba que la "
            "concesión no se sostiene por la razón que dio el juzgado; no dice "
            "si el amparo procede por otra. El tribunal reasume jurisdicción, "
            "sin reenvío, y estudia los conceptos de violación cuyo estudio "
            "omitió el juzgado —los que declaró innecesarios al conceder con "
            "uno—, y de ese estudio sale si concede o niega.",
            "LO QUE NADIE IMPUGNÓ QUEDA FIRME. Lo que la sentencia recurrida "
            "resolvió y ningún agravio combate —por ejemplo, un sobreseimiento "
            "respecto de otra autoridad— no es materia de la revisión: se dice "
            "que queda firme y la revocación se acota a la materia de la "
            "revisión.",
            "UN CONSIDERANDO PROPIO para los conceptos no estudiados, después "
            "del de los agravios, con la misma lógica de dependencia: los que "
            "descansan en la tesis que el estudio de los agravios ya desestimó "
            "caen por las mismas razones, o son inoperantes por derivar de ella, "
            "en grupo; los de contenido propio —otra prueba, otro vicio, otra "
            "consecuencia— se contestan en lo suyo.",
            "SI ALGUNO PROSPERA, SE CONCEDE POR UNA RAZÓN DISTINTA de la del "
            "juzgado, y entonces sí hay efectos que fijar. Si todos caen, se "
            "niega. El punto resolutivo del amparo sale de ahí, no de la "
            "revocación.",
        ],
        "apoyos": ["171925", "178784", "182039", "174177"],
        # El prompt sólo nombra los apoyos que de verdad llegaron al material:
        # se traen al resolver (la rama no se conoce en el adelanto).
        "solo_si_estan": True,
        "necesita": "conceptos_de_violacion",
        "aviso_si_falta":
            "Se revoca una concesión, así que el tribunal reasume jurisdicción "
            "y tiene que estudiar los conceptos de violación que el juzgado no "
            "estudió (artículo 93, fracción VI, de la Ley de Amparo) — y no "
            "constan. Pégalos o sube la demanda de amparo antes de generar: sin "
            "ellos el proyecto revoca pero no puede decir si se concede o se "
            "niega, y el punto resolutivo del amparo sale con hueco.",
    },

    # ── REVISIÓN: lo que queda fuera, y son DOS figuras distintas ─────────
    #
    # Las tenía yo mezcladas en una sola regla. David las deslindó, y la
    # diferencia está en si la parte perjudicada ACUDIÓ o no al recurso:
    #
    #   · no acudió ................ NO ES MATERIA DEL RECURSO
    #   · acudió y no lo impugnó ... QUEDA FIRME
    #
    # No son sinónimos ni se dicen igual en el proyecto, y confundirlas es de
    # las cosas que un revisor marca a la primera.
    "revision_no_es_materia": {
        "cuando": "Hay consideraciones —y el resolutivo ligado a ellas— que "
                  "perjudican a una parte que NO ACUDIÓ a la revisión. El caso "
                  "típico: se concedió el amparo a una persona y la autoridad "
                  "responsable, a quien esa concesión perjudica, no recurre.",
        "fuente": "artículo 93 de la Ley de Amparo; la materia del recurso la "
                  "fija quien recurre",
        "tecnica": [
            "ESAS CONSIDERACIONES NO SON MATERIA DEL RECURSO, y así se dice, "
            "con esas palabras. No se estudian, no se confirman y no se "
            "revocan: quedan fuera porque nadie las trajo.",
            "SE DICE AUNQUE PERJUDIQUEN A QUIEN NO VINO. Que la concesión "
            "afecte a la autoridad responsable no autoriza a revisarla si ella "
            "se conformó: el tribunal no puede empeorar la situación de quien "
            "ganó, en perjuicio de quien no recurrió.",
            "Y SE HACE CONSTAR EXPRESAMENTE, señalando qué consideraciones y "
            "qué resolutivo quedan fuera. Callarlo se lee como omisión de "
            "estudio.",
        ],
    },
    "revision_firmeza": {
        "cuando": "La parte recurrente SÍ acudió a la revisión, pero no "
                  "impugna algunas de las consideraciones que le afectan y "
                  "sólo combate otras que también le afectan.",
        "fuente": "principio de estricto derecho del recurso; artículo 93 de "
                  "la Ley de Amparo",
        "tecnica": [
            "LAS CONSIDERACIONES NO IMPUGNADAS QUEDAN FIRMES, y rigen el "
            "sentido aunque el tribunal las crea equivocadas. Quien pudo "
            "combatirlas vino al recurso y eligió no hacerlo.",
            "SE DICE CUÁLES. El proyecto identifica las consideraciones que "
            "quedan firmes y explica que por eso no se examinan. No basta con "
            "no mencionarlas: eso se lee como omisión de estudio.",
            "NO SE CONFUNDE CON LO QUE NO ES MATERIA DEL RECURSO. Aquí la "
            "parte SÍ acudió; allá no acudió. Son dos figuras y se nombran "
            "distinto.",
        ],
    },

    # ── EL RECURSO SE QUEDÓ SIN OBJETO ───────────────────────────────────
    "recurso_sin_materia": {
        "cuando": "Un hecho POSTERIOR a la interposición del recurso le quitó "
                  "su objeto. El caso más frecuente con diferencia: se recurre "
                  "la negativa de la suspensión PROVISIONAL y, antes de "
                  "resolver, se dicta la DEFINITIVA.",
        "fuente": "artículo 93 de la Ley de Amparo; cesación de la materia del "
                  "recurso",
        "tecnica": [
            "SE DECLARA SIN MATERIA, y no se estudian los agravios: no hay nada "
            "que contestar porque lo que atacaban dejó de existir.",
            "SE EXPLICA EL HECHO SUPERVENIENTE, con su constancia: qué ocurrió, "
            "cuándo, y por qué eso deja al recurso sin objeto. Es todo el "
            "considerando y basta con él.",
            "NO SE CONFUNDE CON DESECHAR. Desechar es rechazar por improcedente "
            "desde el origen —extemporáneo, contra resolución irrecurrible—; "
            "aquí el recurso era procedente y algo posterior le quitó la "
            "materia. Tampoco se confirma: confirmar exigiría estudiar unos "
            "agravios que ya no tienen objeto.",
            "EL PROVISIONAL VIVE HASTA EL DEFINITIVO. Ésa es la razón en el "
            "caso de la suspensión, y conviene decirla: la provisional surte "
            "efectos únicamente hasta que se resuelve sobre la definitiva.",
        ],
    },

    # ── AMPARO DIRECTO: el orden del estudio ──────────────────────────────
    # ═══ LA VIOLACIÓN PROCESAL SE EXAMINA SOBRE LA RESOLUCIÓN QUE LA DECIDIÓ ═══
    # ADC 93/2026 (22-sep-2026): el estudio citó la jurisprudencia correcta
    # y razonó sobre la regla general; la interlocutoria de la reclamación
    # —que decidió la violación— se despachó en un párrafo. La técnica no es
    # de método: los artículos 171 y 172 dicen sobre qué se examina.
    "directo_violacion_procesal": {
        "cuando": "Amparo directo en que se reclama una violación procesal: "
                  "una actuación del procedimiento (ampliación desechada o "
                  "precluida, prueba no admitida, emplazamiento…) que, si la "
                  "hubo, un recurso ordinario confirmó.",
        "fuente": "artículos 171, 172 y 174 de la Ley de Amparo",
        "tecnica": [
            "EL OBJETO DEL EXAMEN ES LA ACTUACIÓN PROCESAL Y LA RESOLUCIÓN DEL "
            "RECURSO ORDINARIO QUE LA CONFIRMÓ, no la sentencia definitiva. La "
            "razón toral del planteamiento son las razones de ESA resolución; "
            "lo que la sentencia dijo de pasada («la actora no ejerció su "
            "derecho») sólo recoge el resultado.",
            "PRIMERO LA PREPARACIÓN (artículo 171): si la ley ordinaria daba "
            "recurso contra la actuación, se dice que se agotó —y con qué "
            "resultado— o que no era exigible. Sin eso, el planteamiento es "
            "inoperante y se dice por qué.",
            # ── LA CONFRONTACIÓN, REESCRITA EL 26-SEP-2026 (decisión 2) ──
            # Decía «se enuncian, una por una» y «PROHIBIDO despacharla en un
            # párrafo». El estudio lo obedeció al pie de la letra: en el 93/2026
            # (estandar2, l. 104-109) salieron cuatro párrafos, uno por razón,
            # contestados con la MISMA respuesta. La orden fabricaba la
            # repetición. David autorizó la regla nueva: cada razón se
            # identifica y se confronta; las que caen por la misma respuesta se
            # contestan juntas nombrándolas; la dependiente cae con la otra
            # salvo que un efecto la presuponga; la mixta se parte. Van como
            # descripción de la función, sin frases que copiar.
            "LUEGO LA CONFRONTACIÓN: se identifican las razones de la "
            "resolución que decidió la violación (qué plazo aplicó, desde "
            "cuándo lo contó, con qué fundamento, por qué tuvo por no exigible "
            "lo reclamado) y cada una se confronta con la ley de la vía y con "
            "la jurisprudencia obligatoria. Ninguna se queda sin respuesta y "
            "se dice cuál resiste y cuál cae. No basta razonar sobre la regla "
            "general sin bajar a lo que esa resolución sostuvo.",
            "LAS RAZONES QUE CAEN POR LA MISMA RESPUESTA SE CONTESTAN JUNTAS, "
            "nombrándolas a todas: una razón no pide desarrollo propio si lo "
            "que la derriba es lo mismo que derriba a otra.",
            "LA RAZÓN QUE DEPENDE DE OTRA CAE CON ELLA y basta decirlo, "
            "nombrándola, SALVO QUE UN EFECTO DE LA CONCESIÓN LA PRESUPONGA "
            "—la oportunidad, el cómputo o la cuantía con que la responsable "
            "tendrá que actuar al reponer—: entonces se desarrolla con el dato "
            "del expediente, porque no puede ordenarse un efecto sobre una "
            "premisa que nadie examinó. Si el material no trae ese dato, se "
            "dice en ADVERTENCIAS.",
            "LA RAZÓN MIXTA SE PARTE: si una misma razón une afirmaciones que "
            "se contestan de modo distinto —una regla de derecho y un hecho "
            "del expediente, por ejemplo—, cada parte recibe su respuesta.",
            "DESPUÉS LA TRASCENDENCIA (artículo 172): la violación sólo "
            "concede si privó a la parte de una defensa que podía cambiar el "
            "resultado del juicio. Se dice qué defensa y por qué podía cambiarlo.",
            # ── LA SUERTE DE LOS DEMÁS, ALINEADA CON LOS ARTÍCULOS 74 Y 174 ──
            # Decía que, si prospera, «los accesorios… quedan sin materia», sin
            # distinguir: una segunda violación procesal habría quedado sin
            # decidir, y los artículos 74, fracción V, y 174 mandan decidirlas
            # TODAS. David, 26-sep-2026: la única excepción es una concesión de
            # fondo con mayor beneficio (artículo 189). La misma regla vive,
            # determinista, en `arbol_decision.aplicar`.
            # Revisión del mismo día: el fondo «sin materia» iba sin la salvedad
            # del 189 que `directo_orden_de_estudio` sí dice, y las dos llegan
            # juntas al estudio: se contradecían en el mismo prompt.
            "SI PROSPERA, EL EFECTO ES LA REPOSICIÓN desde la actuación viciada. "
            "Los planteamientos de fondo quedan sin materia, porque la "
            "sentencia reclamada se deja insubsistente, y se dice, salvo el que "
            "daría a la parte quejosa un beneficio mayor que la reposición, "
            "que se estudia antes (artículo 189); las demás "
            "violaciones procesales NO quedan sin materia: se deciden todas "
            "(artículos 74, fracción V, y 174). SI NO PROSPERA, los "
            "planteamientos que presuponían la reposición caen con él y se "
            "declaran inoperantes con esa razón, salvo que sean otra violación "
            "procesal: ésa se decide por lo que ella misma plantea.",
            "LOS EFECTOS DE LA REPOSICIÓN SE ORDENAN PASO A PASO, porque la "
            "responsable NO puede dictar otra sentencia de inmediato: 1. deje "
            "insubsistente la sentencia reclamada; 2. deje sin efectos la "
            "resolución del recurso ordinario y la actuación viciada (el "
            "acuerdo que desechó, tuvo por precluido o no admitió); 3. reponga "
            "el procedimiento desde esa actuación —admita la ampliación, la "
            "prueba o el escrito de que se trate—; 4. corra traslado a la "
            "contraparte y desahogue lo que proceda, con alegatos; 5. cerrada "
            "de nuevo la instrucción, dicte la sentencia de fondo con plenitud "
            "de jurisdicción. Un «dicte otra en la que atienda los "
            "lineamientos» provoca requerimientos de cumplimiento defectuoso "
            "(artículos 192 a 196 de la Ley de Amparo).",
        ],
    },
    # ═══ EL ORDEN ES EL DEL ARTÍCULO 189 VIGENTE: EL FONDO PRIMERO ═══════════
    # Decía «LAS VIOLACIONES PROCESALES SON DE ESTUDIO PREFERENTE», que es la
    # regla de ejecutorias anteriores a la redacción vigente. El artículo 189,
    # segunda oración (reforma DOF 13-03-2025), dice lo contrario: «se
    # privilegiará el estudio de los conceptos de violación de fondo por encima
    # de los de procedimiento y forma, a menos que invertir el orden redunde en
    # un mayor beneficio para la persona quejosa». Leído en
    # `normas_ley_de_amparo.json`, no de memoria.
    #
    # David, 26-sep-2026: «Sí» a alinear la técnica con el 189, y «también
    # alinea a cómo debe resolverse (mayor beneficio art 189)». Por eso la
    # regla trae también la otra mitad: los artículos 74, fracción V, y 174
    # mandan DECIDIR todas las violaciones procesales, y la única excepción es
    # que un concepto de fondo prospere con mayor beneficio que la reposición.
    # Lo mismo, sin modelo de por medio, en `arbol_decision.aplicar` y en
    # `modos_decision.repartir`.
    "directo_orden_de_estudio": {
        "cuando": "Amparo directo en que se plantean violaciones procesales "
                  "junto con cuestiones de fondo.",
        "fuente": "artículos 74, fracción V, 174 y 189 de la Ley de Amparo",
        "tecnica": [
            "EL FONDO VA PRIMERO. El artículo 189 manda privilegiar el estudio "
            "de los conceptos de fondo por encima de los de procedimiento y "
            "forma. Una violación procesal no es de estudio preferente por el "
            "solo hecho de serlo.",
            "EL ORDEN SE INVIERTE SÓLO POR MAYOR BENEFICIO: la violación "
            "procesal va antes que el fondo únicamente si, de resultar fundada, "
            "daría a la parte quejosa más de lo que podría darle el fondo. Se "
            "justifica comparando, con los datos de este asunto, qué obtendría "
            "la parte con la reposición y qué con la concesión de fondo.",
            "TODAS LAS VIOLACIONES PROCESALES SE DECIDEN (artículos 74, "
            "fracción V, y 174): cada una recibe su calificación. La única que "
            "puede dejar de estudiarse es la que se vuelve innecesaria porque "
            "un concepto de fondo prospera y da a la parte quejosa un beneficio "
            "mayor que la reposición; entonces el proyecto lo dice así, con el "
            "artículo 189. Ninguna otra razón —sin materia, innecesaria, caída "
            "con otro planteamiento— deja sin decidir una violación procesal.",
            "SI LO QUE PROSPERA ES UNA VIOLACIÓN PROCESAL, las demás "
            "violaciones procesales se deciden igual (artículo 174) y los "
            "conceptos de fondo quedan sin materia, salvo el que daría un "
            "beneficio mayor que la reposición, que se estudia antes.",
            "EL ORDEN ELEGIDO SE JUSTIFICA en el proyecto, en una frase, como "
            "aplicación del artículo 189 a este caso concreto, no como fórmula "
            "de método.",
        ],
    },
}


# ═══════════════════════════════════════════════════════════════════════════
# EN LA REVISIÓN FISCAL SÍ HAY REENVÍO
# ═══════════════════════════════════════════════════════════════════════════
# David: «a diferencia de la revisión en amparo indirecto, sí hay reenvío
# porque la jurisdicción para el análisis del fondo corresponde a la Sala
# Regional».
#
# Es la simetría contraria de `revision_levanta_sobreseimiento`, y por eso hay
# que decirlo con todas sus letras: en el amparo en revisión el colegiado
# ASUME jurisdicción y resuelve lo que el juzgado no resolvió; en la revisión
# fiscal NO puede, porque el estudio de los conceptos de anulación corresponde
# en primera instancia a la Sala. Escribir aquí la regla del amparo produce una
# sentencia que invade la competencia de otro tribunal.
#
# ── EL APARATO, VERIFICADO CONTRA EL ACERVO ──
# La tesis que aportó David está en la colección con registro 188742 (0.977 de
# coincidencia con su texto). Buscando alrededor aparecieron tres más, y dos de
# ellas son JURISPRUDENCIA, o sea de mayor fuerza que la suya:
#
#   188742  aislada  · Procede el reenvío si la sentencia no examina todos y
#                      cada uno de los conceptos de nulidad.
#   193181  JURISPR. · Procede la revocación ante la falta de estudio integral
#                      de los conceptos de anulación.
#   196875  JURISPR. · Las sentencias del tribunal fiscal deben analizar todos
#                      los conceptos de anulación.
#   2000895 aislada  · El colegiado ordena al órgano emisor que subsane, ante
#                      la imposibilidad de pronunciarse sobre la legalidad.
#
# Y el propio circuito lo practica: en el acervo de sentencias, 39 de las 506
# revisiones fiscales que revocan llevan lenguaje de reenvío. La R.R.F.
# 94/2023 lo dice entero: «el tribunal colegiado debe revocar la sentencia y
# ordenar el reenvío para que se subsane la omisión, SIN QUE PUEDA SUSTITUIR A
# LA SALA, conforme a la jurisprudencia 2a./J. 6/91, que establece la
# inaplicabilidad de la regla de sustitución de facultades propia del amparo
# directo».
#
# ── EL DESLINDE, QUE IMPORTA TANTO COMO LA REGLA ──
# El reenvío NO procede siempre que se revoque. La tesis 185493 marca el
# límite: cuando el error de la Sala es de forma y NO TRASCIENDE al sentido del
# fallo, el colegiado lo corrige él mismo. Reenviar entonces devuelve un asunto
# resuelto y alarga el juicio para nada.
# ═══════════════════════════════════════════════════════════════════════════
# UN AGRAVIO FUNDADO LLEVA A REVOCAR O A MODIFICAR, Y NO ES LO MISMO
# ═══════════════════════════════════════════════════════════════════════════
# David: «cuando en un recurso es fundado el agravio, el resultado es revocar
# (o modificar) la sentencia recurrida. La revocación es por un vicio absoluto
# que impide conservar aspectos de la sentencia recurrida; la modificación
# obedece a una ilegalidad que, a pesar de advertirse, permite que subsistan
# otras consideraciones».
#
# El proyecto venía escogiendo entre las dos sin decir por qué, y ésa es
# justamente la parte que sostiene el desenlace: la elección NO se sigue de que
# el agravio sea fundado —eso sólo abre las dos puertas—, sino del ALCANCE del
# vicio. Decirlo es lo que separa un resolutivo razonado de uno afirmado.
TECNICA_RESOLUCION["recurso_revoca_o_modifica"] = {
    "cuando": "El agravio es FUNDADO en un recurso.",
    "fuente": "el alcance del vicio, no la sola procedencia del agravio",
    "tecnica": [
        "UN AGRAVIO FUNDADO NO DICE TODAVÍA QUÉ SE HACE CON LA SENTENCIA. "
        "Abre dos puertas —revocar o modificar— y hay que ELEGIR, y decir por "
        "qué. Ésa es la parte que sostiene el resolutivo.",
        "SE REVOCA cuando el vicio es ABSOLUTO: alcanza a la razón que sostiene "
        "el fallo y no permite conservar nada de lo resuelto. Si la premisa que "
        "lo sujetaba se cae, todo lo que colgaba de ella se cae con ella.",
        "SE MODIFICA cuando la ilegalidad, advertida y todo, DEJA EN PIE otras "
        "consideraciones: se corrige lo ilegal y subsiste el resto. Aquí hay "
        "que decir QUÉ SUBSISTE, porque el resolutivo lo conserva.",
        # SIN LA FRASE HECHA (revisión adversarial de la fase E, 28-sep-2026):
        # el prompt v4 del AR 631/2025 traía las dos frases enteras entre
        # comillas y el estudio las copiaba; se describe lo que la frase dice.
        "DILO ANTES DE CERRAR EL ESTUDIO, con tus palabras y en una frase: hasta "
        "dónde llega el vicio (si alcanza la razón que sostiene el fallo o sólo "
        "una parte), qué consideraciones subsisten si alguna subsiste, y por eso "
        "si procede REVOCAR o MODIFICAR. Sin esa frase, el resolutivo queda "
        "afirmado y no razonado.",
        "Y NO CONFUNDAS REVOCAR CON DEJAR INSUBSISTENTE: revocar es lo que hace "
        "este tribunal con la sentencia recurrida; dejar insubsistente es lo "
        "que se le ordena a la responsable en un AMPARO concedido.",
    ],
    "apoyos": [],
    "necesita": "",
    "aviso_si_falta": "",
}


# PROSPERA LA IMPROCEDENCIA QUE ALEGÓ QUIEN RECURRE (revisión del 28-sep-2026,
# AR 631/2025). Toda concesión revocada se trataba como reasunción de la
# fracción VI, también cuando lo que prosperaba era un agravio de procedencia:
# la pantalla pedía la demanda, el prompt ordenaba no concluir y el resolutivo
# dejaba el amparo en hueco, cuando lo que correspondía era revocar y sobreseer.
TECNICA_RESOLUCION["revision_sobresee_por_improcedencia"] = {
    "cuando": ("Recurre la autoridad responsable o la parte tercera interesada y "
               "prospera su agravio contra la omisión o la negativa del juzgado a "
               "sobreseer."),
    "fuente": ("artículos 93, fracciones II y III, y 63, fracción V, de la Ley "
               "de Amparo"),
    "tecnica": [
        "LA PROCEDENCIA VA PRIMERO Y AQUÍ DECIDE. Si se actualiza la causa de "
        "improcedencia que el juzgado desestimó o no examinó, el juicio no debió "
        "llegar al fondo: se revoca la sentencia recurrida y se sobresee en el "
        "juicio.",
        "NO HAY REASUNCIÓN DE JURISDICCIÓN. La fracción VI del artículo 93 rige "
        "para los agravios de fondo: con el juicio improcedente no se estudian "
        "los conceptos de violación, ni los que el juzgado examinó ni los que "
        "declaró innecesarios.",
        "EL PUNTO RESOLUTIVO SOBRESEE, NO NIEGA: negar el amparo supone haber "
        "estudiado el fondo.",
    ],
    "apoyos": [],
    "necesita": "",
    "aviso_si_falta": "",
}


TECNICA_RESOLUCION["revision_fiscal_reenvio"] = {
    "cuando": ("La REVISIÓN FISCAL es fundada y se revoca la sentencia de la "
               "Sala, quedando conceptos de anulación sin estudiar."),
    # 104, FRACCIÓN III: la I-B no existe desde la reforma del 06-06-2011, y
    # el 103 y el 107 son el régimen del amparo, que no gobierna este recurso
    # (verificación de normas, 27-sep-2026).
    "fuente": ("artículos 104, fracción III, de la Constitución, y 50 de la Ley "
               "Federal de Procedimiento Contencioso Administrativo (antes 237 "
               "del Código Fiscal de la Federación)"),
    "tecnica": [
        "SÍ HAY REENVÍO, y ésta es la diferencia con el amparo en revisión. "
        "Ahí el colegiado asume jurisdicción; aquí NO puede: el estudio de los "
        "conceptos de anulación corresponde en primera instancia a la Sala "
        "Regional, y sustituirla dejaría al particular sin el juicio de amparo "
        "contra ese estudio, porque contra lo que resuelve un tribunal "
        "colegiado no procede recurso alguno.",
        "EL RESOLUTIVO LLEVA DOS PUNTOS: uno que REVOCA la sentencia de la "
        "Sala, identificándola por su fecha y su expediente, y otro que le "
        "ordena DICTAR OTRA en la que, siguiendo los lineamientos de esta "
        "ejecutoria, se ocupe de lo que omitió. NO se le ordena dejarla "
        "insubsistente: revocada, ya no existe. Dejar insubsistente es la "
        "fórmula del AMPARO, donde el tribunal no revoca el acto reclamado "
        "sino que ordena a la responsable que lo retire.",
        "SE DICE QUÉ QUEDÓ SIN ESTUDIAR. «Que se ocupe de los conceptos de "
        "anulación» a secas no le dice a la Sala qué tiene que hacer: se "
        "enumeran los que quedaron pendientes.",
        "NO SE ANTICIPA EL FONDO. El proyecto no dice si esos conceptos son "
        "fundados: eso es justamente lo que le toca resolver a la Sala.",
        "CUÁNDO NO SE REENVÍA: si el vicio de la sentencia es de forma y NO "
        "TRASCIENDE al sentido del fallo, el colegiado lo corrige él mismo y "
        "no devuelve nada (registro 185493).",
    ],
    "apoyos": ["188742", "193181", "196875", "2000895"],
    "necesita": "",
    "aviso_si_falta": "",
}


# ═══ LA VIOLACIÓN PROCESAL CUANDO NO HUBO ALZADA (30-sep-2026) ═══════════════
# David, sobre el AD 323/2025 (un juicio oral mercantil): «Siempre, en amparo
# directo, partimos de la base de que hay una sala. Es decir, una segunda
# instancia. Pero no siempre es así». La técnica de arriba lo suponía en tres
# renglones: la violación «que, si la hubo, un recurso ordinario confirmó», el
# objeto del examen «la resolución del recurso ordinario», y el efecto «deje sin
# efectos la resolución del recurso ordinario». En un juicio sin apelación
# —el oral mercantil (artículo 1390 Bis del Código de Comercio), el de cuantía
# menor, el laudo, la nulidad— el motor buscaba esa resolución, no la
# encontraba, y o la inventaba o reprochaba a la quejosa no haber apelado.
#
# El artículo 171 de la Ley de Amparo no pide eso: la preparación se mide con
# el recurso o medio de defensa que la ley ordinaria prevea DURANTE el juicio
# contra la actuación; contra la sentencia no había recurso, así que no se
# exige haberla recurrido. Y la reposición deja insubsistente la sentencia
# reclamada y repone desde la violación, sin una resolución de alzada que dejar
# sin efectos. Se arma DE la de arriba, cambiando sólo esos tres renglones:
# la confrontación, la trascendencia y la suerte de los demás son las mismas y
# así no se separan con el tiempo. La elige `tecnica_de` cuando la instancia
# es única (`instancia_actual`); si no consta, rige la de siempre.
def _violacion_procesal_unica(base: dict) -> dict:
    _cambio = {
        "EL OBJETO DEL EXAMEN":
            "EL OBJETO DEL EXAMEN ES LA ACTUACIÓN PROCESAL y, si contra ella se "
            "hizo valer durante el juicio el medio de defensa que la ley de la "
            "vía prevé, la resolución que lo decidió; no la sentencia "
            "definitiva. La razón toral del planteamiento son las razones de "
            "ESA actuación o de ESA resolución; lo que la sentencia dijo de "
            "pasada sólo recoge el resultado.",
        "PRIMERO LA PREPARACIÓN":
            "PRIMERO LA PREPARACIÓN (artículo 171), medida con la ley de ESTA "
            "vía: se examina respecto del recurso o medio de defensa que la ley "
            "ordinaria prevea DURANTE el juicio contra esa actuación. Si lo "
            "preveía, se dice si se agotó —y con qué resultado— o por qué no era "
            "exigible; si no preveía ninguno, se dice así y la preparación no se "
            "exige. Contra la sentencia no procedía recurso ordinario: no se "
            "exige haberla recurrido ni se reprocha no haberlo hecho. Si había "
            "un medio exigible y no se agotó, el planteamiento es inoperante y "
            "se dice por qué.",
        "LOS EFECTOS DE LA REPOSICIÓN":
            "LOS EFECTOS DE LA REPOSICIÓN SE ORDENAN PASO A PASO, porque la "
            "responsable NO puede dictar otra sentencia de inmediato: 1. deje "
            "insubsistente la sentencia reclamada; 2. deje sin efectos la "
            "actuación viciada (el acuerdo que desechó, tuvo por precluido o no "
            "admitió) y, si se combatió durante el juicio, la resolución que la "
            "confirmó; 3. reponga el procedimiento desde esa actuación —admita "
            "la ampliación, la prueba o el escrito de que se trate—; 4. corra "
            "traslado a la contraparte y desahogue lo que proceda, con "
            "alegatos; 5. cerrada de nuevo la instrucción, dicte la sentencia de "
            "fondo con plenitud de jurisdicción. Un «dicte otra en la que "
            "atienda los lineamientos» provoca requerimientos de cumplimiento "
            "defectuoso (artículos 192 a 196 de la Ley de Amparo).",
    }

    def _uno(t: str) -> str:
        return next((v for k, v in _cambio.items() if t.startswith(k)), t)
    return dict(base,
                cuando="Amparo directo contra una sentencia dictada en ÚNICA "
                       "instancia —sin recurso ordinario que la combata— en que "
                       "se reclama una violación procesal: una actuación del "
                       "procedimiento (ampliación desechada o precluida, prueba "
                       "no admitida, emplazamiento…) y, si la ley de la vía daba "
                       "contra ella un medio de defensa durante el juicio y se "
                       "hizo valer, la resolución que lo decidió.",
                tecnica=[_uno(t) for t in base["tecnica"]])


TECNICA_RESOLUCION["directo_violacion_procesal_unica"] = _violacion_procesal_unica(
    TECNICA_RESOLUCION["directo_violacion_procesal"])


def tecnica_de(tipo: str, rama: str = "", con_violacion_procesal: bool = False) -> list:
    """Las reglas de técnica que aplican a ESTE asunto y ESTE desenlace.

    Se devuelven sólo las pertinentes: darle al modelo el cuadro entero es
    darle instrucciones para escenarios que no son el suyo, y ya sabemos qué
    pasa con las instrucciones que no vienen a cuento.
    """
    t = normalizar(tipo)
    fuera = []
    if t == "amparo_revision":
        # LAS DOS SE DAN SIEMPRE en revisión: cuál aplica depende de quién
        # acudió al recurso, y eso lo sabe el modelo leyendo el expediente, no
        # el catálogo. Se le dan las dos con su deslinde para que elija.
        fuera.append(TECNICA_RESOLUCION["revision_no_es_materia"])
        fuera.append(TECNICA_RESOLUCION["revision_firmeza"])
        if rama == "sin_materia":
            fuera = [TECNICA_RESOLUCION["recurso_sin_materia"]]
        if rama.startswith("revoca_sobreseimiento"):
            fuera.append(TECNICA_RESOLUCION["revision_levanta_sobreseimiento"])
        # ANTES DE ESCRIBIR, «revoca_fondo_niega» SÓLO SALE de un juzgado que
        # concedió y un recurso que prospera (`rama_revision` sin el sentido del
        # estudio): es exactamente el supuesto de la fracción VI. Lo que el
        # estudio concluya después —conceder por razón distinta o negar— no
        # cambia la técnica que había que darle (AR 631/2025, 28-sep-2026).
        if rama == "revoca_fondo_niega":
            fuera.append(TECNICA_RESOLUCION["revision_reasume_concesion"])
        # PROSPERA LA IMPROCEDENCIA (fr. II): se sobresee, no se reasume nada.
        if rama == "revoca_sobresee":
            fuera.append(TECNICA_RESOLUCION["revision_sobresee_por_improcedencia"])
    # La queja tiene el mismo problema y con mas frecuencia: 434 de 2,955.
    # LA REVISIÓN FISCAL, SIEMPRE: el deslinde con el amparo en revisión hay
    # que tenerlo delante tanto para reenviar como para no hacerlo.
    # EL DESLINDE REVOCAR/MODIFICAR ES DE TODOS LOS RECURSOS, no de un tipo.
    if t in ("amparo_revision", "revision_fiscal", "queja"):
        fuera.append(TECNICA_RESOLUCION["recurso_revoca_o_modifica"])
    if t == "revision_fiscal":
        fuera.append(TECNICA_RESOLUCION["revision_fiscal_reenvio"])
    if t == "queja" and rama == "sin_materia":
        fuera.append(TECNICA_RESOLUCION["recurso_sin_materia"])
    if t == "amparo_directo" and con_violacion_procesal:
        # SIN ALZADA, SIN RECURSO ORDINARIO QUE EXAMINAR (30-sep-2026): ver
        # `_violacion_procesal_unica`. Si la instancia no consta, la de siempre.
        fuera.append(TECNICA_RESOLUCION["directo_violacion_procesal_unica"]
                     if instancia_actual() == "unica"
                     else TECNICA_RESOLUCION["directo_violacion_procesal"])
        fuera.append(TECNICA_RESOLUCION["directo_orden_de_estudio"])
    # LA SENTENCIA DICTADA EN CUMPLIMIENTO (30-sep-2026): sólo si consta en el
    # origen del acto de esta petición y rige `cumplimiento_ejecutoria`.
    if t == "amparo_directo" and cumplimiento_actual():
        fuera.append(TECNICA_CUMPLIMIENTO)
    return fuera


# ═══ LA SENTENCIA DICTADA EN CUMPLIMIENTO DE UNA EJECUTORIA (30-sep-2026) ═════
# David: «sólo si se dejó plenitud de jurisdicción o libertad de jurisdicción a
# la responsable, el ulterior amparo puede ser materia de análisis […] no
# podemos entrar en un bucle de amparos». Las reglas y sus tesis están en
# `cumplimiento_ejecutoria` (verificadas en el acervo el 30-sep-2026). Va
# fuera de TECNICA_RESOLUCION porque sólo entra por el origen del acto, no por
# la rama. `solo_si_estan`: el resolver trae sus apoyos también en sesiones
# cuyo adelanto no los trajo, y el prompt sólo nombra los que llegaron.
TECNICA_CUMPLIMIENTO = {
    "cuando": "Amparo directo contra una sentencia dictada en cumplimiento de una ejecutoria de amparo.",
    "fuente": "artículos 61, fracción IX, 63, fracción V, 77, 196 y 201 de la Ley de Amparo",
    "solo_si_estan": True,
    "apoyos": ["2001857", "2018315", "197240", "2007970", "2008199", "2015559", "2024239", "2015151"],
    "tecnica": [
        "SÓLO ES MATERIA DE ESTE AMPARO LO QUE LA RESPONSABLE RESOLVIÓ CON LIBERTAD DE JURISDICCIÓN y lo NOVEDOSO "
        "del nuevo fallo. Lo que la ejecutoria dejó vinculado —su sentido, sus lineamientos, lo que resolvió en "
        "definitiva— es cosa juzgada.",
        "LO SÓLO REITERADO Y NO IMPUGNADO: lo que la ejecutoria no tocó y la responsable repitió tal cual de la "
        "sentencia anterior, si la parte afectada no lo combatió la primera vez que se resolvió en su contra, es "
        "inoperante por consentimiento tácito, SIN que el juicio se vuelva improcedente (2a./J. 113/2012).",
        "SE DISTINGUE POR LO QUE LA EJECUTORIA ORDENÓ: sus efectos y también sus consideraciones y lineamientos. "
        "Antes de los conceptos, un párrafo que diga qué ordenó, qué quedó vinculado y qué resolvió la responsable "
        "con libertad de jurisdicción.",
        "LO VINCULADO SE DECLARA INOPERANTE con esa razón, sin volver a estudiarlo. Lo LIBRE se estudia a fondo y "
        "se concede o se niega. Si una parte de un concepto combate lo vinculado y otra lo libre, se parte.",
        "EL EXCESO O DEFECTO en el cumplimiento no es materia de este amparo: es inoperante y se dice que su vía es "
        "la vista del artículo 196, el recurso de inconformidad (artículo 201) o la denuncia de repetición.",
        "CON LIBERTAD PARCIAL NO SE SOBRESEE. Sólo si la ejecutoria no dejó libertad alguna el juicio es "
        "improcedente (artículo 61, fracción IX) y se sobresee (artículo 63, fracción V); eso lo confirma el "
        "secretario y, si lo confirmó, no hay estudio de fondo. Advertida de oficio la causa, se da vista a la "
        "quejosa por tres días (artículo 64, párrafo segundo) o se razona por qué no hacía falta.",
        "LA INCONSTITUCIONALIDAD DE LA NORMA QUE LA EJECUTORIA MANDÓ APLICAR, planteada por quien fue tercero "
        "interesado en ese amparo: hay criterio dividido (1a. IX/2022 permite estudiarla por primera vez, respetando "
        "la legalidad fijada en la ejecutoria; 2a. CXLVII/2017 exigía combatirla en revisión contra la ejecutoria). "
        "Se estudia o no según lo que decidió el secretario, y se dice por qué.",
    ],
}


def rama_revision(resolvio_a_quo: str, sentido: str,
                  solo_efectos: bool = False,
                  violacion_procesal: bool = False,
                  sentido_amparo: str = "",
                  quien_recurre: str = "",
                  procedencia: bool = False) -> str:
    """La clave de RAMAS_REVISION que corresponde.

    `resolvio_a_quo` es lo que hizo el Juzgado de Distrito —«sobresee»,
    «niega», «concede»— y `sentido`, el del recurso. Las dos ramas de
    resolutivo ÚNICO se piden expresamente porque no se pueden deducir del
    fondo: modificar sólo los efectos y ordenar reponer son decisiones del
    secretario, no consecuencias de que el agravio prospere.

    `sentido_amparo` —«concede» | «niega» | «»— es lo que el ESTUDIO concluye
    cuando, levantado el sobreseimiento, el tribunal asume jurisdicción y mira
    los conceptos de violación por primera vez. Lo lee
    `fase_rama.sentido_en_plenitud` del texto del estudio. Sin él,
    `revoca_sobreseimiento_niega` era inalcanzable: estaba declarada y ninguna
    combinación de argumentos la devolvía nunca.

    `quien_recurre` —«quejoso» | «tercero» | «autoridad» | «»— y `procedencia`
    (el agravio que prospera es de procedencia: improcedencia o
    sobreseimiento) cambian la rama cuando el juzgado CONCEDIÓ (revisión del
    28-sep-2026, AR 631/2025):
      · si recurre la propia quejosa y su recurso prospera, no se le niega el
        amparo que ya tenía: «dictará la que corresponda» (fr. V) y la
        concede con el alcance o los efectos que resulten del estudio
        («revoca_fondo_concede»; «modifica_efectos» si sólo son los efectos);
        si la sentencia era mixta, lo que ella combate es el sobreseimiento
        (frs. I y V: «revoca_sobreseimiento_*»);
      · si recurre la autoridad o la tercera y lo que prospera es su agravio
        de procedencia, se revoca y se sobresee (fr. II: «revoca_sobresee»),
        sin reasumir nada.
    Vacío = no consta quién recurre, y se sigue como si recurriera la
    autoridad o la tercera (la fr. VI se aplica: `reasuncion`).
    """
    a = (resolvio_a_quo or "").strip().lower()
    s = (sentido or "").strip().lower()
    q = (quien_recurre or "").strip().lower()
    _prospera = prospera(s)

    # LAS DOS RAMAS DE RESOLUTIVO ÚNICO SE MIRAN DESPUÉS, Y CON CONDICIONES.
    # Se resolvían ANTES que nada y sin mirar el sentido ni lo que hizo el
    # juzgado, así que un agravio DESESTIMADO en el que se pedía reponer el
    # procedimiento producía «se ordena la reposición», y un estudio que
    # hablaba de efectos sobre una sentencia que NEGÓ el amparo producía «se
    # modifica la sentencia únicamente para los efectos» —efectos de una
    # concesión que no existe—.
    #
    # REPONER Y MODIFICAR SON DECISIONES QUE SÓLO CABEN SI EL RECURSO PROSPERA:
    # si el agravio es infundado no hay nada que reponer ni que modificar. Y
    # modificar los efectos presupone que el amparo se concedió.
    # SIN MATERIA SE MIRA LO PRIMERO, y no depende del sentido ni de lo que
    # hizo el juzgado: si el recurso perdió su objeto, no hay nada que
    # confirmar ni que revocar. Lo declara el secretario —es un hecho del
    # expediente, no una conclusión del estudio— y por eso llega como sentido.
    if s in ("sin_materia", "sin materia"):
        return "sin_materia"
    if violacion_procesal and _prospera:
        return "repone_procedimiento"
    if solo_efectos and _prospera and a == "concede":
        return "modifica_efectos"
    # LA SENTENCIA MIXTA. Si el recurso no prospera, se confirma entera y el
    # resolutivo dice las dos cosas. Si prospera, lo que se revoca es el FONDO
    # —los agravios van contra la negativa o la concesión, no contra el
    # sobreseimiento de un acto que nadie combatió—, y se sigue por la parte
    # de fondo con la mecánica de siempre.
    if a in ("sobresee_niega", "sobresee_concede"):
        if not _prospera:
            return "confirma_" + a
        # LA QUEJOSA NO COMBATE LA CONCESIÓN QUE GANÓ: si recurre una sentencia
        # que sobreseyó un acto y la amparó por los demás, lo que ataca es el
        # sobreseimiento, y al prosperar se levanta (frs. I y V).
        a = "sobresee" if (a == "sobresee_concede" and q == "quejoso") else a.split("_", 1)[1]
    if a not in ("sobresee", "niega", "concede"):
        return "sin_determinar"
    # LA LISTA A MANO SE VA. Enumeraba tres calificaciones y dejaba fuera
    # «esencialmente_fundado», que en este circuito es el 23% de los agravios
    # de las revisiones que revocan: la rama caía en «confirma» y el
    # resolutivo confirmaba una sentencia que el estudio revocaba.
    if not prospera(s):
        return {"sobresee": "confirma_sobresee", "niega": "confirma_niega",
                "concede": "confirma_concede"}[a]
    if a == "sobresee":
        # Levantado el sobreseimiento, el sentido del AMPARO no lo dice el
        # recurso: hay que estudiar los conceptos por primera vez. Este
        # comentario decía «se deja en concede sólo si el estudio lo afirma;
        # por omisión, niega» y el código de debajo devolvía «concede» siempre,
        # sin leer nada y sin recibir nada que leer.
        #
        # NO SE INVIERTE EL VALOR POR OMISIÓN. Negar un amparo que el estudio
        # concedió es tan grave como lo contrario, y ninguno de los dos
        # sentidos se puede suponer: sólo se aparta de «concede» cuando el
        # estudio dice con todas sus letras que procede negar.
        return ("revoca_sobreseimiento_niega"
                if str(sentido_amparo or "").strip().lower() == "niega"
                else "revoca_sobreseimiento_concede")
    if a == "concede":
        # RECURRE LA QUEJOSA Y GANA (revisión del 28-sep-2026): pedía más —otros
        # efectos, otro alcance— y la rama la dejaba sin amparo («revoca y
        # niega»). Sola recurrente, su recurso no puede empeorarle la situación:
        # la sentencia que corresponde la ampara (fr. V).
        if q == "quejoso":
            return "revoca_fondo_concede"
        # PROSPERA LA IMPROCEDENCIA QUE ALEGÓ LA AUTORIDAD O LA TERCERA (fr. II):
        # se revoca y se sobresee; la fr. VI es para los agravios de fondo.
        if procedencia:
            return "revoca_sobresee"
        # REVOCAR LA CONCESIÓN ES REASUMIR JURISDICCIÓN (art. 93, fr. VI; AR
        # 631/2025, 28-sep-2026). El segundo punto lo dice el estudio de los
        # conceptos que el juzgado no estudió: si uno prospera, se concede por
        # razón distinta. Sin ese dato se sigue diciendo «niega» —es la rama
        # que el estudio recibe ANTES de escribir, y la que `tecnica_de`
        # reconoce—; el documento pone el hueco cuando los conceptos no
        # constaron o el estudio calla (`puntos_reasuncion`).
        return ("revoca_fondo_concede"
                if str(sentido_amparo or "").strip().lower() == "concede"
                else "revoca_fondo_niega")
    return "revoca_fondo_concede"


# ═══════════════════════════════════════════════════════════════════════════
# ¿HAY QUE REASUMIR JURISDICCIÓN? (art. 93, frs. I, V y VI; 28-sep-2026)
# ═══════════════════════════════════════════════════════════════════════════
# Dos supuestos, y en los dos el tribunal estudia CONCEPTOS DE VIOLACIÓN que
# nadie examinó, cuyo texto no está en el expediente del recurso salvo que la
# sentencia recurrida los transcriba o se suba la demanda:
#   · «sobreseimiento»: el juzgado sobreseyó y el recurso prospera (frs. I y V);
#   · «concesion»: el juzgado concedió —o sobreseyó un acto y concedió por
#     otro—, recurre la autoridad o la tercera interesada y el recurso
#     prospera (fr. VI). Fue el defecto de fondo del AR 631/2025: revocó y negó
#     sin estudiar los conceptos que el juzgado declaró innecesarios.
# Un solo sitio para las tres puertas: la pantalla (`necesita_conceptos`), el
# estudio (`fase6_estudio._bloque_conceptos`) y el resolutivo.
FUNDAMENTO_REASUNCION = {
    "sobreseimiento": "artículo 93, fracciones I y V, de la Ley de Amparo",
    "concesion": "artículo 93, fracción VI, de la Ley de Amparo",
}


def reasuncion(resolvio_a_quo: str, sentido: str, solo_efectos: bool = False,
               violacion_procesal: bool = False, quien_recurre: str = "",
               procedencia: bool = False) -> str:
    """«sobreseimiento» | «concesion» | «» — si, con lo que hizo el juzgado y
    el sentido del recurso, el tribunal tiene que estudiar conceptos de
    violación por primera vez.

    `quien_recurre` —«quejoso» | «tercero» | «autoridad» | «»— sólo excluye la
    fracción VI cuando recurre la propia quejosa (ahí rige la V); vacío = no
    consta y se aplica, que es lo que dice la Segunda Sala: «sin importar quién
    interponga el recurso» (2a./J. 113/2007, registro 171925). Reponer el
    procedimiento o modificar sólo los efectos no reasumen nada.

    Revisión del 28-sep-2026 (AR 631/2025), las dos puertas que faltaban:
      · la quejosa que recurre una sentencia MIXTA (sobreseyó un acto y la
        amparó por los demás) combate el sobreseimiento; si prospera, se
        levanta y se estudian los conceptos de ese acto (frs. I y V):
        «sobreseimiento»;
      · `procedencia`: lo que prospera es un agravio de procedencia de la
        autoridad o la tercera (fr. II): se sobresee y no hay conceptos que
        reasumir. La fr. VI habla de los agravios DE FONDO."""
    a = (resolvio_a_quo or "").strip().lower()
    q = (quien_recurre or "").strip().lower()
    if not prospera(sentido) or violacion_procesal or solo_efectos:
        return ""
    if a == "sobresee":
        return "sobreseimiento"
    if a == "sobresee_concede" and q == "quejoso":
        return "sobreseimiento"
    if a in ("concede", "sobresee_concede"):
        if q == "quejoso" or procedencia:
            return ""
        return "concesion"
    return ""


# ¿ESTA EJECUTORIA CONCEDE —y le toca fijar los efectos—? Un solo predicado para
# los avisos que suponían concesión con que un agravio prosperara (AR 631/2025:
# «Se concede y los efectos van en prosa», «EFECTOS INCOMPLETOS PARA UNA
# VIOLACIÓN PROCESAL» y «El resolutivo concede porque hay conceptos
# fundados…» en un proyecto que revoca y NIEGA). Confirmar una concesión no
# cuenta: los efectos son los de la recurrida.
_RAMAS_QUE_CONCEDEN = ("revoca_fondo_concede", "revoca_sobreseimiento_concede",
                       "modifica_efectos")


def ejecutoria_concede(rama: str, sentido_amparo: str = "") -> bool | None:
    """True | False | None (no se sabe: rama vacía o sin determinar). Con
    `sentido_amparo` —lo que concluye el estudio, `fase_rama.sentido_en_plenitud`—
    una rama calculada ANTES de escribir se corrige con lo que el estudio
    resolvió al reasumir jurisdicción: «revoca_fondo_niega» concede si el
    estudio de los conceptos concluye conceder; «revoca_sobreseimiento_*»,
    según lo que concluya (sin dato, concede, como `rama_revision`)."""
    k = (rama or "").strip()
    sa = (sentido_amparo or "").strip().lower()
    if not k or k == "sin_determinar" or k not in RAMAS_REVISION:
        return None
    if k == "revoca_fondo_niega":
        return sa == "concede"
    if k.startswith("revoca_sobreseimiento"):
        if sa:
            return sa != "niega"
        return k == "revoca_sobreseimiento_concede"
    return k in _RAMAS_QUE_CONCEDEN


# ═══════════════════════════════════════════════════════════════════════════
# CUANDO EL CÓMPUTO DA EXTEMPORÁNEA
# ═══════════════════════════════════════════════════════════════════════════
# David: «si oportunidad == Extemporánea, el flujo debe abortar el estudio de
# fondo y generar automáticamente el sobreseimiento». La falla que describe es
# declarar en el considerando tercero que la demanda es extemporánea y después
# entrar al fondo y conceder el amparo.
#
# Hasta hoy la puerta existía y hacía lo contrario de lo que él pide: lanzaba
# un 409 diciendo «ese proyecto todavía hay que escribirlo a mano». Negarse a
# trabajar es una forma de no equivocarse, no de servir.
#
# Y EL CORPUS CORRIGE MI SUPUESTO: en los RECURSOS no se sobresee, se DESECHA,
# y las fórmulas están medidas en el banco del propio tribunal:
#
#   revisión  «ÚNICO. Se desecha el recurso de revisión, al resultar
#              improcedente por extemporáneo, por los motivos y fundamentos
#              expuestos en el considerando último de la presente ejecutoria.»
#   queja     «ÚNICO. Se desecha por improcedente el recurso de queja.»
#
# El sobreseimiento con los artículos 61, fracción XIV, y 63, fracción V —los
# que David cita— es del AMPARO, donde sí hay juicio en que sobreseer.
EXTEMPORANEO = {
    "amparo_directo": {
        "rotulo": "Extemporaneidad de la demanda de amparo",
        "fundamento": "artículos 61, fracción XIV, y 63, fracción V, de la Ley "
                      "de Amparo",
        "considerando": (
            "La demanda de amparo se presentó de manera extemporánea, conforme "
            "al cómputo que antecede, de modo que se actualiza la causa de "
            "improcedencia prevista en el artículo 61, fracción XIV, de la Ley "
            "de Amparo y, en consecuencia, procede sobreseer en el juicio con "
            "fundamento en el artículo 63, fracción V, del mismo ordenamiento."),
        "resolutivo": "ÚNICO. Se sobresee en el presente juicio de amparo "
                      "promovido por {quejoso}.",
        # EL REMATE CUANDO EL SECRETARIO DESESTIMA LA EXTEMPORANEIDAD. El
        # análisis de la improcedencia es OFICIOSO (artículo 62 de la Ley de
        # Amparo), así que el considerando no puede saltársela: tiene que
        # decir por qué NO se actualiza. Los dos preceptos están verificados
        # en el texto vigente — 61, fracción XIV: «Contra normas generales o
        # actos consentidos tácitamente, entendiéndose por tales aquéllos
        # contra los que no se promueva el juicio de amparo dentro de los
        # plazos previstos».
        "rectificado": (
            "no se actualiza la causa de improcedencia prevista en el artículo "
            "61, fracción XIV, de la Ley de Amparo —cuyo análisis es oficioso "
            "en términos del artículo 62 del mismo ordenamiento—, por lo que "
            "procede el estudio de los conceptos de violación."),
    },
    "amparo_revision": {
        "rotulo": "Extemporaneidad del recurso de revisión",
        "fundamento": "artículo 86 de la Ley de Amparo",
        "considerando": (
            "El presente medio de impugnación se interpuso de manera "
            "extemporánea, conforme al cómputo que antecede, por lo que resulta "
            "improcedente y debe desecharse."),
        "resolutivo": "ÚNICO. Se desecha el recurso de revisión, al resultar "
                      "improcedente por extemporáneo, por los motivos y "
                      "fundamentos expuestos en el considerando último de la "
                      "presente ejecutoria.",
        "rectificado": (
            "el recurso se interpuso dentro del plazo de diez días que prevé "
            "el artículo 86 de la Ley de Amparo, por lo que no procede "
            "desecharlo por extemporáneo y debe estudiarse el fondo de los "
            "agravios."),
    },
    "queja": {
        "rotulo": "Extemporaneidad del recurso de queja",
        "fundamento": "artículo 98 de la Ley de Amparo",
        "considerando": (
            "El recurso de queja se interpuso de manera extemporánea, conforme "
            "al cómputo que antecede, por lo que resulta improcedente."),
        "resolutivo": "ÚNICO. Se desecha por improcedente el recurso de queja.",
        "rectificado": (
            "el recurso se interpuso dentro del plazo que prevé el artículo 98 "
            "de la Ley de Amparo, por lo que no procede desecharlo por "
            "extemporáneo y debe estudiarse el fondo de los agravios."),
    },
    "revision_fiscal": {
        "rotulo": "Extemporaneidad de la revisión fiscal",
        "fundamento": "artículo 63 de la Ley Federal de Procedimiento "
                      "Contencioso Administrativo",
        "considerando": (
            "El recurso de revisión fiscal se interpuso de manera extemporánea, "
            "conforme al cómputo que antecede, por lo que resulta improcedente "
            "y debe desecharse."),
        "resolutivo": "ÚNICO. Se desecha por extemporáneo el recurso de "
                      "revisión fiscal.",
        "rectificado": (
            "el recurso se interpuso dentro del plazo de quince días que prevé "
            "el artículo 63 de la Ley Federal de Procedimiento Contencioso "
            "Administrativo, por lo que no procede desecharlo por extemporáneo "
            "y debe estudiarse el fondo de los agravios."),
    },
}


def extemporaneo_de(tipo: str) -> dict:
    return EXTEMPORANEO.get(normalizar(tipo), EXTEMPORANEO["amparo_directo"])


# ═══════════════════════════════════════════════════════════════════════════
# EL ENRUTADOR DE MARCOS NORMATIVOS
# ═══════════════════════════════════════════════════════════════════════════
# David enumera tres fallas concretas:
#
#   F1. citar el artículo 172 de la Ley de Amparo —catálogo de violaciones
#       procesales EXCLUSIVO del amparo directo— dentro de un amparo en
#       revisión indirecto;
#   F2. fundar la queja en el 97, fracción I, cuando lo impugnado es la
#       SUSPENSIÓN dictada por la autoridad responsable en un amparo DIRECTO,
#       que es el 97, fracción II, inciso b);
#   F3. citar los artículos 74, 81 o 93 de la Ley de Amparo dentro de una
#       revisión fiscal, que se rige por el 104, fracción III, constitucional y
#       el 63 de la LFPCA.
#
# EL MATIZ QUE HACE PELIGROSA ESTA REGLA, y por el que no se aplica al
# documento entero: un amparo en revisión cita LEGÍTIMAMENTE preceptos fuera
# del 81-96 —el 61 y el 63 para las causales, el 74 para los requisitos de la
# sentencia, el 79 para la suplencia—. Prohibir «todo lo que no esté en el
# rango» produciría una alarma en cada proyecto, y una alarma que salta siempre
# deja de leerse.
#
# Por eso se prohíben preceptos NOMBRADOS, no rangos: los que sólo pueden
# aparecer si se coló la plantilla de otra vía. Y en revisión fiscal, donde la
# Ley de Amparo no gobierna nada, se prohíbe la ley entera pero SÓLO en el
# considerando de competencia —el párrafo del cómputo cita su artículo 19 para
# los inhábiles, y eso es otra discusión—.
PRECEPTOS_AJENOS = {
    "amparo_revision": [
        (r"art[íi]culo\s+172\b(?![^.]{0,40}LFPCA)",
         "el artículo 172 de la Ley de Amparo es el catálogo de violaciones "
         "procesales del AMPARO DIRECTO: en una revisión de amparo indirecto "
         "no viene a cuento"),
        (r"art[íi]culos?\s+1(?:7[0-9]|8[0-9]|90|91)\b",
         "los artículos 170 a 191 rigen el AMPARO DIRECTO, no la revisión"),
    ],
    "amparo_directo": [
        (r"art[íi]culos?\s+(?:8[1-9]|9[0-6])\s*(?:,|\sy\s|\sde\s+la\s+Ley\s+de\s+Amparo)",
         "los artículos 81 a 96 rigen el RECURSO DE REVISIÓN, no el amparo directo"),
    ],
    "queja": [],
    # LA REVISIÓN FISCAL NO ESTÁ VACÍA DE LEY DE AMPARO, Y DECIRLO ASÍ ERA
    # FALSO. La regla que había aquí prohibía la Ley de Amparo entera, y el
    # engrose firmado por el secretario la cita: el artículo 92, para el turno.
    # La frontera la marca la propia ley que sí gobierna la vía, en el último
    # párrafo de su artículo 63:
    #
    #   «Este recurso de revisión deberá tramitarse en los términos previstos
    #    en la Ley de Amparo EN CUANTO A LA REGULACIÓN DEL RECURSO DE REVISIÓN.»
    #
    # Así que no es «sí o no»: es «para qué». La lista está en `_LA_EN_FISCAL`
    # y la comprueba `_amparo_fuera_de_lugar`, que mira los números uno a uno.
    "revision_fiscal": [],
}

# LO QUE LA REMISIÓN DEL 63 SÍ TRAE: los artículos que regulan el recurso de
# revisión (81 a 96), el cómputo de sus días hábiles (19) y la obligatoriedad
# de la jurisprudencia (215 a 230), que es regla general de todo órgano
# jurisdiccional y no del amparo.
_LA_EN_FISCAL = set(range(81, 97)) | {19} | set(range(215, 231))

# ═══════════════════════════════════════════════════════════════════════════
# QUÉ LEY GOBIERNA CADA VÍA, DICHO ANTES DE ESCRIBIR
# ═══════════════════════════════════════════════════════════════════════════
# Detectar la ley equivocada en el documento terminado es la red; esto es la
# barandilla. Al modelo hay que decirle en qué vía está ANTES de que razone,
# porque si no razona con la ley que más ha visto —la de Amparo— y luego hay
# que tacharla.
# ═══ LA REGLA QUE VALE PARA LAS CUATRO VÍAS ══════════════════════════════════
#
# David, sobre el amparo en revisión 322/2025 —una medida provisional de
# restricción dictada por un juez de San Juan del Río por violencia familiar—:
# «el modelo confunde las disposiciones en materia de amparo relativas a las
# medidas cautelares con las normas que rigen las medidas provisionales a la luz
# del Código de Procedimientos Civiles del Estado. Por lógica, LOS ACTOS
# RECLAMADOS NUNCA SE RIGEN POR LA LEY DE AMPARO, sino por las disposiciones que
# aplica la autoridad responsable».
#
# La causa de fondo era de recuperación y está arreglada en `fase6_rag`: al
# estudio le llegaban trece artículos de la Ley de Amparo y ninguno de
# Querétaro, así que citó lo único que se le puso delante. Esto es la
# barandilla: aunque el material venga bien, el modelo tiene que saber para qué
# sirve cada ley.
#
# NO NOMBRA NINGUNA LEY DEL ACTO NI NINGÚN NÚMERO QUE NO SEA DE LA LEY DE
# AMPARO, a propósito: un ejemplo escrito dentro de un prompt acaba copiado
# literal en la sentencia firmada —van cuatro veces medidas—, y aquí un ejemplo
# sería el nombre de un código que quizá no es el del asunto. Un párrafo que no
# afirma un dato no puede afirmar un dato falso.
# LA REGLA GENERAL, cuando el sistema NO ha podido derivar quién dictó el acto.
# Va en negativo porque tiene que cubrir los dos casos a la vez, y por eso se usa
# lo menos posible: ver `_ACTO_ORDINARIO` para cuando sí se sabe.
_ACTO_NO_ES_AMPARO = (
    "\n\nEL ACTO RECLAMADO NO SE RIGE POR LA LEY DE AMPARO. Esa ley gobierna "
    "ESTE juicio o ESTE recurso —procedencia, oportunidad, legitimación, "
    "suplencia, técnica y resolutivo—, nunca el acto de la autoridad "
    "responsable. El acto se juzga con las disposiciones que ELLA aplicó al "
    "dictarlo, y son las que tienes en las NORMAS del material.\n"
    "  · SI EN LAS NORMAS NO HAY NADA del ordenamiento que aplicó la "
    "responsable, DILO con esas palabras y sigue con lo que sí tengas. "
    "Escribir el estudio con la ley del juicio en su lugar es peor que decir "
    "que falta: el hueco se ve y se corrige, la ley equivocada no.\n"
    "  · LA EXCEPCIÓN, Y ES UNA SOLA: cuando la autoridad responsable es un "
    "órgano de amparo —el juez de distrito cuyo auto o cuya interlocutoria se "
    "recurre—, la ley que ella aplicó ES la Ley de Amparo, y entonces sí funda "
    "el estudio. Mira quién dictó el acto antes de decidirlo."
)

# ═══ Y LA QUE SE USA CUANDO EL SISTEMA YA LO SABE, QUE VA EN POSITIVO ════════
#
# Lección cara: la versión en negativo PRODUCE lo que quiere evitar. Le decía al
# modelo «los preceptos de la Ley de Amparo que regulan la suspensión rigen la
# suspensión del amparo y nada más», y el proyecto del 322/2025 salió con dos
# párrafos explicando exactamente eso —«los artículos 128 y 147 de la Ley de
# Amparo regulan la actuación del juez de amparo, no la validez originaria del
# acto»—. Correcto como derecho, y David no lo quiere: «no es necesario que eso
# aparezca».
#
# Una regla que nombra lo que no debe salir invita a contrastarlo. Cuando el
# sistema HA DERIVADO quién dictó el acto, no hace falta el contraste: basta
# decir cuál es la ley y callar la otra. Lo que no se nombra no se escribe.
_ACTO_ORDINARIO = (
    "\n\nLA LEY QUE RIGE EL ACTO RECLAMADO es la que aplicó la autoridad "
    "responsable al dictarlo, y la tienes en las NORMAS del material: con ella "
    "se juzga si el acto estuvo bien o mal. Transcribe entre comillas el "
    "precepto decisivo y razona sobre él.\n"
    "  · SI EN LAS NORMAS NO HAY NADA de ese ordenamiento, DILO con esas "
    "palabras y sigue con lo que sí tengas. El hueco se ve y se corrige; la ley "
    "equivocada, no.\n"
    "  · De la Ley de Amparo, en este asunto, sólo se usan la procedencia, la "
    "oportunidad, la legitimación, la suplencia y el resolutivo. Para el fondo "
    "no la necesitas, y no hace falta explicar por qué: escribe el estudio con "
    "la ley del acto y ya está.\n"
    "  · NI SIQUIERA POR ANALOGÍA. No tomes de la Ley de Amparo un criterio, un "
    "estándar ni una pauta «orientadora» para juzgar el acto: en este asunto esa "
    "ley no rige el fondo, y un estándar prestado de una ley que no rige es una "
    "motivación equivocada aunque suene razonable. Lo que gobierna está en las "
    "NORMAS; si no alcanza, dilo."
)

# Y cuando el acto SÍ lo dictó un órgano de amparo —la interlocutoria de
# suspensión que se recurre, el auto del juez de distrito—, entonces esa ley es
# la del acto y hay que decirlo igual de claro.
_ACTO_DE_AMPARO = (
    "\n\nAQUÍ LA LEY DE AMPARO SÍ RIGE EL ACTO: quien lo dictó es un órgano "
    "de amparo y aplicó esa ley. Funda el estudio en ella, con el precepto "
    "concreto, igual que harías con el código que aplica cualquier otra "
    "responsable."
)

LEY_DE_LA_VIA = {
    "revision_fiscal":
        "ESTA VÍA NO SE RIGE POR LA LEY DE AMPARO. La revisión fiscal es el "
        "recurso del artículo 63 de la Ley Federal de Procedimiento "
        "Contencioso Administrativo, y la sentencia que se revisa es la de un "
        "Tribunal de Justicia Administrativa, no la de un juez de amparo. "
        "Por tanto:\n"
        "  · Los requisitos de la sentencia son los del artículo 50 de la "
        "LFPCA, NO los del 74 de la Ley de Amparo.\n"
        "  · NO hay suplencia de la queja: el artículo 79 de la Ley de Amparo "
        "habla de «la autoridad que conozca del juicio de amparo», y aquí "
        "quien recurre es la AUTORIDAD. No hay parte débil a la que suplir.\n"
        "  · NO cabe corregir la cita de preceptos por el artículo 76 de la "
        "Ley de Amparo.\n"
        "De la Ley de Amparo sólo puedes invocar lo que su artículo 63 de la "
        "LFPCA manda aplicar —«en cuanto a la regulación del recurso de "
        "revisión», o sea los artículos 81 a 96—, el 19 para los días hábiles "
        "y los de jurisprudencia (215 a 230), que son regla general.",
    "queja":
        "ESTA VÍA ES EL RECURSO DE QUEJA del artículo 97 de la Ley de Amparo. "
        "No resuelves un amparo: resuelves si el auto recurrido estuvo bien "
        "dictado. No hay acto reclamado ni autoridad responsable, hay un auto "
        "y el órgano que lo dictó.",
    "amparo_revision":
        "ESTA VÍA ES EL RECURSO DE REVISIÓN, artículos 81 a 96 de la Ley de "
        "Amparo. El artículo 172 y los 170 a 191 son del AMPARO DIRECTO y no "
        "vienen a cuento aquí.",
    "amparo_directo":
        "ESTA VÍA ES EL AMPARO DIRECTO, artículos 170 a 191 de la Ley de "
        "Amparo. Los artículos 81 a 96 regulan el recurso de revisión y no "
        "gobiernan este juicio.",
}


# EL ENCABEZADO SE COMPONE, NO SE TECLEA. Medido sobre los cinco engroses
# reales: NOMBRE DEL TIPO + MATERIA concordada + número reproduce exacto los
# cinco. «RECURSO DE QUEJA ADMINISTRATIVO: 143/2026», «AMPARO EN REVISIÓN
# ADMINISTRATIVA: 17/2025», «REVISIÓN FISCAL: 6/2025», «AMPARO DIRECTO CIVIL:
# 642/2024», «RECURSO DE QUEJA CIVIL: 233/2025». (La queja administrativa se
# corrigió a femenino en la cuarta ronda: ver `_MATERIA_ENCABEZADO`.)
_ENCABEZADO = {
    "amparo_directo": "AMPARO DIRECTO {materia}: {numero}",
    "amparo_revision": "AMPARO EN REVISIÓN {materia}: {numero}",
    "queja": "RECURSO DE QUEJA {materia}: {numero}",
    "revision_fiscal": "REVISIÓN FISCAL: {numero}",
}

# La materia va concordada con el sustantivo que la precede: «AMPARO DIRECTO
# CIVIL» pero «AMPARO EN REVISIÓN ADMINISTRATIVA». Escribirla en masculino
# siempre da «AMPARO EN REVISIÓN ADMINISTRATIVO», que ningún secretario firma.
#
# LA QUEJA ES FEMENINA: «RECURSO DE QUEJA ADMINISTRATIVA» (E10, cuarta ronda,
# 3-oct-2026). El rótulo concuerda con «queja», no con «recurso»: el corpus de
# quejas del tribunal dice «QUEJA ADMINISTRATIVA» 14 de 14 veces y nunca
# «ADMINISTRATIVO», y el V I S T O del mismo documento decía «recurso de queja
# administrativa» bajo un encabezado «RECURSO DE QUEJA ADMINISTRATIVO» (Q
# 172/2026). Y LO FAMILIAR SE REGISTRA COMO CIVIL: el tribunal es de materias
# administrativa y civil, y el expediente de un asunto familiar se rotula
# «CIVIL» (AR 307/2024: «AMPARO EN REVISIÓN CIVIL», y lo mismo los otros
# familiares del banco, 239 y 448). Sólo en el rótulo: la competencia sigue
# diciendo «en materia familiar», como el engrose. La materia que llegue en
# masculino («administrativo», «agrario») también concuerda con el tipo.
_MATERIA_ENCABEZADO = {
    "amparo_revision": {"administrativa": "ADMINISTRATIVA", "civil": "CIVIL",
                        "laboral": "LABORAL", "penal": "PENAL", "familiar": "CIVIL",
                        "agraria": "AGRARIA"},
    "queja": {"administrativa": "ADMINISTRATIVA", "civil": "CIVIL",
              "laboral": "LABORAL", "penal": "PENAL", "familiar": "CIVIL",
              "agraria": "AGRARIA"},
    "amparo_directo": {"administrativa": "ADMINISTRATIVO", "civil": "CIVIL",
                       "laboral": "LABORAL", "penal": "PENAL", "familiar": "CIVIL",
                       "agraria": "AGRARIO"},
}
_MATERIA_EN_FEMENINO = {"administrativo": "administrativa", "agrario": "agraria"}


# EL APARTADO QUE FIJA LA CUESTIÓN, con el nombre que le da cada vía. Lara
# Chagoyán lo llama «Materia de la revisión» en su ejemplo; el corpus del
# tribunal usa ese mismo rótulo en la revisión y «Materia del recurso» en la
# queja. En el amparo directo la cuestión son los conceptos de violación.
_ROTULO_MATERIA = {
    "amparo_revision": "Materia de la revisión.",
    "revision_fiscal": "Materia de la revisión.",
    "queja": "Materia del recurso.",
    "amparo_directo": "Cuestión a resolver.",
}

_MATERIA_A_RESOLVER = {
    "amparo_revision":
        "La materia de la revisión se constriñe a resolver las cuestiones "
        "siguientes:",
    "revision_fiscal":
        "La materia de la revisión se constriñe a resolver las cuestiones "
        "siguientes:",
    "queja":
        "La materia del recurso se constriñe a resolver las cuestiones "
        "siguientes:",
    "amparo_directo":
        "El estudio de los conceptos de violación se constriñe a resolver las "
        "cuestiones siguientes:",
}


# ═══════════════════════════════════════════════════════════════════════════
# LA LEGITIMACIÓN: EL APARTADO PROMETÍA DOS COSAS Y DECÍA UNA
# ═══════════════════════════════════════════════════════════════════════════
# El considerando se rotula «Legitimación y oportunidad» y sólo se escribía la
# oportunidad —el cómputo—. Peor: el párrafo del cómputo empieza por
# «Igualmente,», que es el enlace que sigue a la legitimación, así que el
# apartado abría con un conector que no se refería a nada.
#
# David lo escribió a mano al ajustar el adelanto:
#
#   «El presente medio de impugnación fue interpuesto por [quien], a través de
#    su representante legal [quien], quien se encuentra legitimada para
#    interponer el recurso de revisión conforme al artículo 6º de la misma ley,
#    toda vez que la resolución impugnada le resulta desfavorable.»
#
# EL FUNDAMENTO ES DE LA VÍA, y ése es el motivo de que esto viva aquí y no en
# el compositor: en el amparo y su revisión legitima el artículo 6º de la Ley
# de Amparo; en la queja, el 97; y en la revisión fiscal no es «quien resulta
# afectado» sino la AUTORIDAD, a través de su unidad de defensa jurídica, por
# el 63 de la LFPCA. Escribir el mismo precepto en los cuatro es el vicio de
# plantilla única otra vez.
LEGITIMACION = {
    "amparo_directo": {
        "molde": ("La demanda de amparo fue promovida por {parte}{rep}, quien "
                  "está legitimad{a} para ello conforme al artículo 5o., "
                  "fracción I, de la Ley de Amparo, por resentir el perjuicio "
                  "que le causa la sentencia reclamada."),
        "fundamento": "artículo 5o., fracción I, de la Ley de Amparo"},
    "amparo_revision": {
        "molde": ("El presente medio de impugnación fue interpuesto por "
                  "{parte}{rep}, quien se encuentra legitimad{a} para "
                  "interponer el recurso de revisión conforme al artículo 6o. "
                  "de la Ley de Amparo, toda vez que la resolución impugnada "
                  "le resulta desfavorable."),
        "fundamento": "artículo 6o. de la Ley de Amparo"},
    "queja": {
        # EL 97 ES LA PROCEDENCIA, NO LA LEGITIMACIÓN (verificación de normas,
        # 27-sep-2026): legitima ser PARTE del juicio de amparo, artículo 5o.,
        # que es como lo funda la variante del banco (qj-c3-quejoso).
        # CON SU FRACCIÓN (3-oct-2026): la del papel de quien recurre —I la
        # quejosa, II la autoridad responsable, III la tercera interesada, IV el
        # Ministerio Público—. «El artículo 5o.» a secas no dice de qué parte se
        # trata; la variante del banco ya la escribe («5o., fracción I»).
        "molde": ("El recurso fue interpuesto por {parte}{rep}, quien está "
                  "legitimad{a} para hacerlo en términos del artículo 5o., "
                  "fracción {fr}, de la Ley de Amparo, por ser parte en el juicio "
                  "de amparo en que se dictó el auto recurrido y resultarle "
                  "desfavorable."),
        "fundamento": "artículo 5o. de la Ley de Amparo"},
    "revision_fiscal": {
        # AQUÍ NO ES «QUIEN RESIENTE EL PERJUICIO»: es la autoridad, y sólo
        # por conducto de su unidad de defensa jurídica. Lo dice el propio 63.
        # LA UNIDAD NO ES LA AUTORIDAD DEMANDADA (3-oct-2026). El molde decía
        # «interpuesto por Administración Desconcentrada Jurídica…, autoridad
        # facultada para hacerlo por conducto de la unidad administrativa
        # encargada de su defensa jurídica»: llamaba autoridad demandada a su
        # propia unidad de defensa, y sin artículo. La fórmula del corpus
        # (rf-c2, oro_RF): parte legítima por el 63, párrafo primero, porque
        # recurre la unidad encargada de la defensa jurídica de la autoridad
        # demandada, a la que la sentencia le resultó adversa.
        # (Desde la tercera ronda el párrafo lo arma `_legitimacion_rf`, que
        # distingue la unidad, la propia demandada y lo demás; este molde queda
        # como la forma de la unidad, que es la del corpus.)
        "molde": ("El recurso de revisión fiscal fue interpuesto por parte "
                  "legítima, en términos del artículo 63, párrafo primero, en "
                  "relación con el 5o., cuarto párrafo, de la Ley Federal de "
                  "Procedimiento Contencioso Administrativo, pues lo hizo valer "
                  "{parte}{rep}, en su carácter de unidad administrativa "
                  "encargada de la defensa jurídica de {autoridad}, a la que la "
                  "sentencia recurrida le resultó adversa."),
        "fundamento": ("artículo 63, párrafo primero, en relación con el 5o., "
                       "cuarto párrafo, de la Ley Federal de Procedimiento "
                       "Contencioso Administrativo")},
}


# QUIEN RECURRE NO SIEMPRE ES LA QUEJOSA (28-sep-2026, AR 631/2025): la
# tercera interesada que recurrió quedó legitimada «conforme al artículo 6º»,
# que es el de quien promueve el amparo. Está legitimada por ser PARTE del
# juicio de amparo (art. 5o., fr. III) y su personería es la del 11, que la
# nombra; la autoridad responsable, por el 5o., fr. II, con el límite del 87.
_LEGITIMACION_REVISION_POR_PAPEL = {
    "tercero": {
        "molde": ("El presente medio de impugnación fue interpuesto por "
                  "{parte}{rep}, quien se encuentra legitimad{a} para "
                  "interponer el recurso de revisión en su carácter de parte "
                  "tercera interesada en el juicio de amparo, conforme al "
                  "artículo 5o., fracción III, de la Ley de Amparo, toda vez "
                  "que la resolución impugnada le resulta desfavorable."),
        "rep": "en términos del artículo 11 de la Ley de Amparo"},
    "autoridad": {
        "molde": ("El presente medio de impugnación fue interpuesto por "
                  "{parte}{rep}, quien se encuentra legitimad{a} para "
                  "interponer el recurso de revisión en su carácter de "
                  "autoridad responsable, conforme a los artículos 5o., "
                  "fracción II, y 87 de la Ley de Amparo, toda vez que la "
                  "sentencia recurrida afecta directamente el acto que se le "
                  "reclamó."),
        "rep": "en términos del artículo 9o. de la Ley de Amparo"},
    # LA QUEJOSA QUE RECURRE, CON SU FRACCIÓN (3-oct-2026, AR 60/2025: «acorde
    # con lo dispuesto en los artículos 5, fracción I y 6 párrafo primero»). El
    # 6o. solo regula quién PROMUEVE el amparo; ser parte —y por eso poder
    # recurrir— es el 5o., fracción I. Sólo cuando el papel consta.
    "quejoso": {
        "molde": ("El presente medio de impugnación fue interpuesto por "
                  "{parte}{rep}, quien se encuentra legitimad{a} para "
                  "interponer el recurso de revisión en su carácter de parte "
                  "quejosa en el juicio de amparo, conforme a los artículos 5o., "
                  "fracción I, y 6o. de la Ley de Amparo, toda vez que la "
                  "resolución impugnada le resulta desfavorable."),
        "rep": "en términos de los artículos 6o. y 11 de la Ley de Amparo"},
    # EL PROMULGADOR QUE RECURRE EN EL AMPARO CONTRA NORMAS (E12, cuarta ronda,
    # 3-oct-2026; AR 208 y 72/2025). El 87, primer párrafo, tiene dos hipótesis:
    # la sentencia que afecta directamente el acto reclamado de cada autoridad
    # y, «tratándose de amparo contra normas generales», los titulares de los
    # órganos del Estado a los que se encomiende su emisión o promulgación
    # (texto local de la Ley de Amparo). Con la norma impugnada y el Gobernador
    # o la Legislatura recurriendo, el documento escribía la primera; el engrose
    # del 208 funda en la segunda («en su carácter de promulgadora»). Lo pide
    # `legitimacion_de(..., hay_norma=True)`.
    "autoridad_norma": {
        "molde": ("El presente medio de impugnación fue interpuesto por "
                  "{parte}{rep}, quien se encuentra legitimad{a} para "
                  "interponer el recurso de revisión en su carácter de "
                  "autoridad responsable, conforme a los artículos 5o., "
                  "fracción II, y 87, primer párrafo, de la Ley de Amparo, pues "
                  "tratándose de amparo contra normas generales pueden "
                  "interponerlo los titulares de los órganos del Estado a los "
                  "que se encomiende su emisión o promulgación."),
        "rep": "en términos del artículo 9o. de la Ley de Amparo"},
}

# Los órganos que emiten o promulgan normas generales (art. 87, primer párrafo,
# LA). Al principio del nombre: «Secretario Ejecutivo…» no es el Ejecutivo.
_RX_PROMULGADOR = re.compile(
    r"^(?:(?:el|la|los|las)\s+)?(?:titular\s+del\s+)?(?:"
    r"gobernador(?:a)?|(?:poder\s+)?ejecutivo\s+(?:del\s+estado|federal|estatal|local)|"
    r"poder\s+ejecutivo|presidente\s+de\s+la\s+rep[úu]blica|presidenta\s+de\s+la\s+rep[úu]blica|"
    r"jef[ea]\s+de\s+gobierno|congreso|legislatura|c[áa]mara\s+de\s+(?:diputad[oa]s|senador[ae]s)|"
    r"poder\s+legislativo|ayuntamiento|cabildo|presidente\s+municipal|presidenta\s+municipal)\b",
    re.I)


def es_promulgador(nombre: str) -> bool:
    """¿Es un órgano que emite o promulga normas generales (Gobernador,
    Legislatura, Congreso, Ejecutivo, Ayuntamiento…)?"""
    return bool(_RX_PROMULGADOR.search(" ".join(str(nombre or "").split())))


# LA PERSONERÍA DE QUIEN ACTÚA POR LA PARTE, POR SU FIGURA Y POR EL PAPEL DE LA
# PARTE (3-oct-2026, rev_0, rev_2 y rev_3). Todo representante salía «en
# términos de los artículos 6o. y 11», que son los de quien comparece por la
# persona quejosa: Q 172/2026 y AR 239/2025 fundaban así a la AUTORIZADA en
# términos amplios (art. 12: «podrá interponer los recursos que procedan»); la
# queja del Director de Ingresos, a su DELEGADO (art. 9o.: las autoridades
# «podrán por medio de oficio acreditar personas delegadas»); y la tercera, con
# el 6o., que no es suyo (el 11 la nombra). El Ministerio Público no comparece
# por representante: sin cola.
_RX_AUTORIZADO = re.compile(r"\bautorizad[oa]s?\b", re.I)
_RX_DELEGADO = re.compile(r"\bdelegad[oa]s?\b", re.I)


# UN ARTÍCULO SIN SU LEY NO SE FIRMA (E9, cuarta ronda, 3-oct-2026; Q 172/2026,
# Q 300/2025, AR 239 y 60/2025): la figura llegaba «autorizado en términos
# amplios del artículo 12» y `personeria_de` daba el 12 por citado en cuanto veía
# «artículo 12», sin mirar de qué ley; salía «del artículo 12, Representante;»
# en el resultando y en la legitimación. Ahora la personería sólo calla si la
# figura ya cita la LEY; si cita el 12 o el 9o. sin ella, `figura_en_prosa` la
# completa con «de la Ley de Amparo» y la cola no se repite.
_RX_CITA_UNA_LEY = re.compile(
    r"art[íi]culos?\s+\d+[^;]{0,40}?\b(?:de\s+la\s+(?:Ley|L\.?\s*A\.?\b|LFPCA)|del\s+(?:C[óo]digo|Reglamento))",
    re.I)
_RX_ART_12_O_9 = re.compile(r"\bart[íi]culo\s+(?:12|9o?\.?)(?!\d)", re.I)
# LA FIGURA QUE YA CITA LA LEY DE AMPARO, SEA CUAL SEA (F6, quinta ronda,
# 3-oct-2026; AD 456 y 274/2025 con nombres de prueba): «apoderado en términos
# de los artículos 6o. y 11 de la Ley de Amparo» recibía además la cola «en
# términos de los artículos 6o. y 11…», porque sólo el autorizado y el delegado
# callaban. Ahora calla cualquiera que ya cite la Ley de Amparo; la cita de
# OTRA ley («… del artículo 10 de la Ley General de Sociedades Mercantiles») no
# dice la personería en el amparo y la cola sigue.
_RX_CITA_LA_LEY_DE_AMPARO = re.compile(
    r"art[íi]culos?\s+\d+[^;]{0,40}?\bde\s+la\s+(?:Ley\s+de\s+Amparo\b|L\.?\s*A\.?(?!\w))", re.I)
_RX_YA_DICE_EN_TERMINOS = re.compile(r"\ben\s+t[ée]rminos\b", re.I)


def personeria_de(papel: str = "", figura: str = "") -> str:
    """«en términos del artículo 12 de la Ley de Amparo» (autorizado), «… 9o.…»
    (autoridad o delegado), «… 11…» (tercero) o «… 6o. y 11…» (quejoso o sin
    papel). Vacío si la figura ya cita su artículo CON SU LEY («… del artículo
    12 de la Ley de Amparo») o es el Ministerio Público; «del artículo 12» a
    secas no cuenta como cita (E9). QUINTA RONDA (F6): vacío también si
    CUALQUIER figura ya cita la Ley de Amparo; y si la figura ya dice «en
    términos» sin citar artículo («autorizada en términos amplios», Q
    172/2026), la cola dice «conforme al artículo 12…», sin repetir la
    fórmula."""
    p = (papel or "").strip().lower()
    f = " ".join(str(figura or "").split())
    _cita = bool(_RX_CITA_UNA_LEY.search(f))
    if _RX_CITA_LA_LEY_DE_AMPARO.search(f):
        return ""
    if _RX_AUTORIZADO.search(f):
        cola = "" if _cita else "en términos del artículo 12 de la Ley de Amparo"
    elif p == "autoridad" or _RX_DELEGADO.search(f):
        cola = "" if _cita else "en términos del artículo 9o. de la Ley de Amparo"
    elif p == "ministerio_publico":
        cola = ""
    elif p in ("tercero", "tercera", "tercero_interesado"):
        cola = "en términos del artículo 11 de la Ley de Amparo"
    else:
        cola = "en términos de los artículos 6o. y 11 de la Ley de Amparo"
    if cola and _RX_YA_DICE_EN_TERMINOS.search(f) and not re.search(r"art[íi]culo", f, re.I):
        cola = "conforme " + _conforme_en_vez_de_en_terminos(cola[len("en términos "):])
    return cola


def _conforme_en_vez_de_en_terminos(x: str) -> str:
    """«del artículo 12…» → «al artículo 12…»; «de los artículos…» → «a los
    artículos…» (la contracción de «en términos del» pasada a «conforme al»)."""
    x = (x or "").strip()
    if x.startswith("del "):
        return "al " + x[4:]
    if x.startswith("de los "):
        return "a los " + x[7:]
    if x.startswith("de la "):
        return "a la " + x[6:]
    return x


# LAS FIGURAS GENÉRICAS van en minúscula («su apoderado legal», «su consejo de
# administración», corpus AD 552/2024); LOS CARGOS de una autoridad conservan
# su mayúscula, como en el resultando del mismo documento (rev RF 2, 26 y
# 6/2026: la legitimación decía «su subdirector de lo Contencioso» y el
# resultando «su Subdirector de lo Contencioso»: dos grafías del mismo cargo).
_RX_FIGURA_GENERICA = re.compile(
    r"^(?:su\s+|el\s+|la\s+)?(?:apoderad[oa]s?|representantes?|autorizad[oa]s?|delegad[oa]s?|"
    r"albaceas?|mandatari[oa]s?|tutor(?:a|es)?|curador(?:a|es)?|gestor(?:a)?|abogad[oa]s?|"
    r"asesor(?:a)?|defensor(?:a)?|procurador(?:a)?\s+judicial|administrador(?:a)?\s+[úu]nic[oa]|"
    r"gerente|consejo\s+de\s+administraci[óo]n|[óo]rgano\s+de\s+representaci[óo]n|"
    r"presidente\s+del\s+consejo|socio|socia|interventor(?:a)?|liquidador(?:a)?|s[íi]ndic[oa]|"
    r"madre|padre|progenitor(?:a|es)?|leg[íi]tim[oa]\s+representante)\b",
    re.I)


def figura_en_prosa(figura: str, ley_de_amparo: bool = True) -> str:
    """«Apoderado Legal» → «apoderado legal»; «autorizado en términos amplios
    del artículo 12 de la Ley de Amparo» intacto; «Titular de la Unidad de
    Asuntos Jurídicos» intacto (un cargo conserva su mayúscula).

    LA FIGURA NO SE PASA ENTERA A MINÚSCULAS (rev_3, AR 239/2025: «del artículo
    12 de la ley de amparo»): en la figura genérica, sólo el tramo del cargo,
    hasta la primera preposición; lo que sigue —la ley, la unidad, la
    institución— conserva sus mayúsculas. EL CARGO DE UNA AUTORIDAD NO SE BAJA
    (cuarta ronda): si llegó en versales se pasa a prosa como órgano
    («TITULAR DE LA UNIDAD JURÍDICA» → «Titular de la Unidad Jurídica»).

    CON `ley_de_amparo` (el amparo y sus recursos; la revisión fiscal lo apaga),
    el «artículo 12» o «9o.» sin ley se completa: «… del artículo 12 de la Ley de
    Amparo» (E9); en el autorizado y el delegado siempre, y en cualquier otra
    figura cuando la cita la cierra (quinta ronda, F6).

    UNA SOLA PUERTA PARA LA FIGURA (F6, quinta ronda, 3-oct-2026; AD 456 y
    274/2025 con nombres de prueba): el resultando, la legitimación y el
    resolutivo la escribían con tres convertidores —«su Apoderado Legal» en uno,
    «su apoderado legal» en otro, y «del artículo 12 Pedro Gil Mora» sin ley en
    el resolutivo firmado—. Ésta es la puerta, y `sep_figura` el separador del
    nombre que la sigue."""
    f = " ".join(str(figura or "").split()).strip(" ,;")
    f = re.sub(r"(?i)^su\s+", "", f)
    if not f:
        return ""
    generica = bool(_RX_FIGURA_GENERICA.search(f))
    if f == f.upper() and any(c.isalpha() for c in f):
        if generica:
            f = f.lower()
            f = re.sub(r"\bley de amparo\b", "Ley de Amparo", f)
        else:
            f = sin_articulo_de_prosa(con_articulo_de_organo(f)) or f.lower()
    elif generica:
        m = re.search(r"\s(?:de|del|en|para|a)\s", f)
        f = f.lower() if not m else f[:m.start()].lower() + f[m.start():]
    f = re.sub(r"(?i)\bley\s+de\s+amparo\b", "Ley de Amparo", f)
    if ley_de_amparo:
        m = _RX_ART_12_O_9.search(f)
        if m and not _RX_CITA_UNA_LEY.search(f) and (
                _RX_AUTORIZADO.search(f) or _RX_DELEGADO.search(f) or not f[m.end():].strip(" ,.;")):
            f = f[:m.end()] + " de la Ley de Amparo" + f[m.end():]
    return " ".join(f.split())


# LA ETIQUETA DE ROL QUE VIAJA DENTRO DEL NOMBRE (3-oct-2026, AR 208/2025 y RF
# 4/2025). Las carátulas del circuito la ponen entre paréntesis —«GOBERNADOR DEL
# ESTADO DE QUERÉTARO (AUTORIDAD RESPONSABLE)», «… del ISSSTE (DEMANDADA)»— y
# llegaba a la prosa: «el Gobernador… (autoridad Responsable), en su carácter de
# autoridad responsable». Se quita y se devuelve el rol, por si el carácter no
# constaba.
_RX_ROL_FINAL = re.compile(
    r"\s*\(\s*(?P<rol>autoridad\s+(?:responsable|demandada)|demandad[oa]|quejos[oa]s?|"
    r"tercer[oa]s?\s+interesad[oa]s?|tercer[oa]|recurrentes?|actor(?:a|es)?|"
    r"(?:parte\s+)?(?:quejosa|actora|demandada|recurrente|tercera\s+interesada)|adherente)\s*\)\s*\.?$",
    re.I)


def sin_etiqueta_de_rol(nombre: str) -> tuple:
    """(nombre sin la etiqueta final de rol, rol en minúsculas o «»)."""
    n = " ".join(str(nombre or "").split())
    m = _RX_ROL_FINAL.search(n)
    if not m:
        return n, ""
    return n[:m.start()].rstrip(" ,"), " ".join(m.group("rol").lower().split())


# VARIAS PERSONAS EN UN SOLO CAMPO (3-oct-2026, AD 552/2024 y AR 208/2025: «X y
# Z, ambos de apellidos …» → «promovió», «quien está legitimada»). Se reconoce
# sin adivinar: «ambos», «todos», «y otros», el punto y coma, o dos o más tramos
# que tienen forma de nombre (dos palabras con mayúscula) separados por coma o
# «y». Una persona moral o un órgano nunca es plural por llevar «y» en su
# nombre («Transportistas y Similares»).
#
# UN SOLO DETECTOR, Y LA LISTA SE MIRA ANTES QUE LA SOCIEDAD (E2, cuarta ronda,
# 3-oct-2026). «X, SOCIEDAD DE PRODUCCIÓN RURAL…, POR CONDUCTO DE SU CONSEJO DE
# ADMINISTRACIÓN, Y Z, POR PROPIO DERECHO» (AD 552/2024) daba singular porque el
# campo contenía una sociedad, y el compositor —con su propio detector— decía
# «promovieron»: el resultando en plural y la legitimación y la carátula en
# singular, en el mismo documento. Ahora lo EXPRESO decide primero, aunque haya
# una sociedad: «ambos/todos», «y otros/otras/otra», «coagraviados», el punto y
# coma, la numeración de una lista («1) …, 2) …») y «, y Nombre» (una coma y
# luego otra persona, no «, y en representación de…»). Después, como antes: una
# sociedad, un órgano o una cabeza colectiva son UNA; y dos tramos con forma de
# nombre, varias. El compositor (`resultandos_por_tipo`) pregunta aquí.
# «todos/todas» sólo cuando se refiere a las partes («todos de apellido…»,
# «todas por propio derecho»), no en «en representación de todos sus hijos»; y
# «codemandado» en singular describe a una persona, no a varias.
_RX_PLURAL_EXPRESO = re.compile(
    r"\b(?:ambos|ambas)\b|\b(?:todos|todas)\s+(?:de\s+apellidos?|ellos|ellas|por\s+(?:su\s+)?propio)\b|"
    r"\by\s+otr[oa]s?\b|\bco(?:agraviad[oa]s|demandad[oa]s|quejos[oa]s|actor(?:es|as)|recurrentes)\b|;",
    re.I)
_RX_NUMERACION_LISTA = re.compile(r"(?:^|[\s,;])\d{1,3}\)\s*\S")
_RX_COMA_Y = re.compile(r",\s+(?:y|e)\s+(\S+)", re.I)
# Lo que sigue a «, y» o abre un tramo y NO es otra persona: fórmulas de
# representación, artículos, y los cargos o figuras que se ponen tras el nombre.
_RX_NO_ES_NOMBRE = re.compile(
    r"^(?:por|en|a|como|su|sus|quien|quienes|de|del|con|mediante|el|la|los|las|se|ambos|ambas|"
    r"todos|todas|otr[oa]s?|albaceas?|apoderad[oa]s?|representantes?|autorizad[oa]s?|tutor(?:a|es)?|"
    r"curador(?:a|es)?|gerente|administrador(?:a)?|socio|socia|presidente|presidenta|vocal|"
    r"secretari[oa]|tesorer[oa]|consejo|comisariado|menor(?:es)?|hij[oa]s?)\b", re.I)
_RX_TRAMO_NOMBRE = re.compile(
    r"^(?:[A-ZÁÉÍÓÚÑ][\wÁÉÍÓÚÑáéíóúñ.'-]*)(?:\s+(?:(?:de|del|la|las|los|y|e|van|von|da|do|dos|das)\s+)?"
    r"[A-ZÁÉÍÓÚÑ][\wÁÉÍÓÚÑáéíóúñ.'-]*)+$")


def _otra_persona_tras_coma_y(n: str) -> bool:
    """«…, y Juan Pérez» / «…, Y PARTE QUEJOSA» / «…, y *****»: tras la coma y
    la «y» empieza otra persona (mayúscula o testado, y no una fórmula)."""
    for m in _RX_COMA_Y.finditer(n):
        w = m.group(1).strip(",.;:()«»\"'")
        if not w:
            continue
        if w.startswith("*"):
            return True
        if w[:1].isupper() and not _RX_NO_ES_NOMBRE.match(w):
            return True
    return False


def es_plural_de_partes(nombre: str, moral=None) -> bool:
    """¿El campo nombra a VARIAS personas (quejosos, recurrentes, actores)?

    LA ÚNICA PUERTA (E2): la usan el compositor —que expone el resultado en
    `datos_extra['plural']`—, la legitimación y la carátula."""
    n = " ".join(str(nombre or "").split())
    if not n:
        return False
    # LO EXPRESO, ANTES QUE LA SOCIEDAD (AD 552/2024).
    if _RX_PLURAL_EXPRESO.search(n) or _RX_NUMERACION_LISTA.search(n) or _otra_persona_tras_coma_y(n):
        return True
    if moral or es_organo_publico(n) or _RX_SOCIEDAD.search(n) or _RX_CABEZA_COLECTIVO.search(n) \
            or _RX_CENTRAL_OBRERA.search(n):
        return False
    tramos = [x.strip() for x in re.split(r",|\s+[yYeE]\s+", n) if x.strip()]
    return sum(1 for x in tramos if _RX_TRAMO_NOMBRE.match(x) and not _RX_NO_ES_NOMBRE.match(x)
               and not es_organo_publico(x)) >= 2


# EL GÉNERO DE VARIAS PERSONAS, SÓLO SI EL TEXTO LO DICE (E2). «X y Z, ambos de
# apellidos…» concuerda en masculino porque el papel escribió «ambos»; «todas de
# apellido…», en femenino. Nunca por los nombres de pila. «y otra» no dice nada
# del primero («otra persona»). Lo declarado en la ficha (`genero_quejoso`)
# manda. Sin dato, «» y la fórmula neutra: «la parte quejosa está legitimada»,
# «PARTE QUEJOSA».
_RX_PLURAL_MASC = re.compile(r"\b(?:ambos|todos|los\s+dos|y\s+otros|coagraviados|codemandados|"
                             r"coquejosos|coactores|corecurrentes)\b", re.I)
_RX_PLURAL_FEM = re.compile(r"\b(?:ambas|todas|las\s+dos)\b", re.I)


def genero_de_plural(nombre: str, declarado: str = "") -> str:
    """«o», «a» o «» para concordar a VARIAS personas (ver arriba)."""
    d = (declarado or "").strip().lower()[:1]
    if d in ("a", "f"):
        return "a"
    if d in ("o", "m"):
        return "o"
    n = " ".join(str(nombre or "").split())
    if _RX_PLURAL_MASC.search(n):
        return "o"
    if _RX_PLURAL_FEM.search(n):
        return "a"
    return ""


# EL ARTÍCULO DE LAS CABEZAS COLECTIVAS QUE NO SON SOCIEDADES (3-oct-2026, AD
# 335/2025 y AR 60/2025: «no ampara ni protege a Sucesión a Bienes de…»,
# «interpuesto por sucesión intestamentaria…»). La sucesión, la comunidad y la
# asociación llevan «la»; el ejido, el comisariado y el núcleo de población,
# «el». Las sociedades mercantiles van sin artículo, como siempre.
_COLECTIVO_ARTICULO = (
    (re.compile(r"^(?:sucesi[óo]n|comunidad|asociaci[óo]n)\b", re.I), "la"),
    (re.compile(r"^(?:ejido|comisariado|n[úu]cleo)\b", re.I), "el"),
)


def con_articulo_de_colectivo(nombre: str) -> str:
    """«Sucesión a Bienes de X» → «la Sucesión a Bienes de X»; «Ejido San
    Juan» → «el Ejido San Juan»; lo demás, tal cual."""
    n = " ".join(str(nombre or "").split())
    if not n or re.match(r"^(?:el|la|los|las)\s", n, re.I):
        return n
    for rx, art in _COLECTIVO_ARTICULO:
        if rx.match(n):
            return f"{art} {n[:1].upper()}{n[1:]}"
    return n


# LA FRACCIÓN DEL 5o. POR EL PAPEL DE QUIEN RECURRE LA QUEJA (3-oct-2026).
_FRACCION_5O = {"quejoso": "I", "quejosa": "I", "autoridad": "II", "tercero": "III",
                "tercera": "III", "tercero_interesado": "III", "ministerio_publico": "IV",
                "ministerio público": "IV", "mp": "IV"}


def fraccion_5o_de(papel: str = "", recurre_el_quejoso=None) -> str:
    """«I» | «II» | «III» | «IV» | «» — la fracción del artículo 5o. de quien
    recurre. Sin papel, la del quejoso SÓLO si consta que recurre el quejoso;
    si no, vacío: quien llama pone el hueco y el aviso. No se adivina."""
    f = _FRACCION_5O.get((papel or "").strip().lower(), "")
    if f:
        return f
    return "I" if recurre_el_quejoso else ""


# LA PERSONA FÍSICA NO CONCUERDA POR SU NOMBRE (3-oct-2026). «Juan Pérez López,
# quien está legitimado» / «Ana López Ruiz, quien está legitimado»: el adjetivo
# tomaba el género por omisión, y con una quejosa salía en masculino (19
# proyectos de octubre). Su género no se sabe y NO se infiere del nombre de
# pila; la fórmula neutra ya existía en la rama con representante: «; la parte
# quejosa está legitimada». Lo colectivo se reconoce por la CABEZA del nombre
# o por su forma social, nunca por un sufijo —«Isadora» acaba como
# «Comercializadora»—.
_RX_CABEZA_COLECTIVO = re.compile(
    r"^(?:(?:el|la|los|las)\s+)?(?:uni[óo]n|federaci[óo]n|confederaci[óo]n|liga|central|alianza|"
    r"agrupaci[óo]n|c[áa]mara|coalici[óo]n|organizaci[óo]n|asociaci[óo]n|sociedad|cooperativa|"
    r"empresa|instituci[óo]n|fundaci[óo]n|universidad|inmobiliaria|comercializadora|constructora|"
    r"arrendadora|operadora|administradora|distribuidora|sindicato|instituto|organismo|consejo|"
    r"banco|frente|grupo|colegio|comit[ée]|patronato|fideicomiso|ejido|comunidad|n[úu]cleo|"
    r"sucesi[óo]n|comisariado)\b", re.I)


def es_persona_fisica(nombre: str, moral=None) -> bool:
    """¿Es (o puede ser) una persona física? Entonces su legitimación se dice
    con la fórmula neutra. Ante la duda, sí: la neutra es correcta para todos."""
    n = " ".join(str(nombre or "").split())
    if not n or es_organo_publico(n):
        return False
    if moral is None:
        try:
            import promovente as _pv_f
            moral = _pv_f.es_moral(n)
        except Exception:
            moral = False
    if moral or _RX_SOCIEDAD.search(n) or _RX_CENTRAL_OBRERA.search(n):
        return False
    return not _RX_CABEZA_COLECTIVO.search(n)


def legitimacion_de(tipo: str, parte: str = "", representante: str = "",
                    hueco: str = "*********", figura: str = "",
                    moral=None, papel: str = "", autoridad_demandada: str = "",
                    recurre_el_quejoso=None, avisos=None, plural=None, genero: str = "",
                    hay_norma=None, norma_impugnada=None) -> str:
    """El párrafo de legitimación de esta vía, o cadena vacía si no hay parte.

    SIN NOMBRE NO SE ESCRIBE. Un párrafo que dice «la parte está legitimada»
    sin decir quién no acredita nada, y es justo la perífrasis que el catálogo
    persigue en los resultandos.

    DOS COSAS DISTINTAS, DICHAS APARTE. David (22-sep-2026, ADC 93/2026): la
    persona física que comparece por la moral «no resiente el perjuicio ni
    tiene legitimación activa ad causam (…) únicamente ostenta la
    legitimación procesal activa (representación o personería) en términos
    de los artículos 6 y 11 de la Ley de Amparo». Así que la legitimación se
    predica de LA PARTE y la personería del representante, cada una con su
    fundamento; antes el molde pegaba «a través de su representante legal»
    y seguía diciendo «quien está legitimada… por resentir el perjuicio»
    de la persona física.

    LA AUTORIDAD LLEVA SU ARTÍCULO Y CONCUERDA POR SU CARGO (3-oct-2026). En el
    AR del XXII Circuito salió «interpuesto por Director de Ingresos…, quien se
    encuentra legitimada»: sin artículo, y en femenino porque
    `promovente.separar` marca como persona MORAL al Director (el indicador de
    persona moral es de las sociedades, no de los órganos). Un órgano o un
    servidor público se nombra con su artículo —«el Director», «la
    Administración Desconcentrada»— y el adjetivo concuerda con ese artículo.
    `autoridad_demandada` (sólo revisión fiscal) es la autoridad cuya defensa
    lleva la unidad que recurre; sin ella, «la autoridad demandada en el
    juicio de nulidad».

    LA PERSONA FÍSICA, EN NEUTRO (3-oct-2026): «…promovida por Ana López Ruiz;
    la parte quejosa está legitimada…». Su género no se infiere del nombre.
    Las morales y los órganos, como estaban: concuerdan por su artículo.

    LA QUEJA, CON LA FRACCIÓN DEL 5o. de quien recurre (`fraccion_5o_de`):
    `recurre_el_quejoso` decide cuando el papel no consta; si tampoco, la
    fracción va en hueco y quien llama avisa.

    TERCERA RONDA (3-oct-2026):
      · la PERSONERÍA por la figura y el papel (`personeria_de`): autorizado,
        art. 12; autoridad o delegado, 9o.; tercero, 11; quejoso, 6o. y 11. La
        figura conserva sus mayúsculas fuera del cargo (`figura_en_prosa`) y,
        si es larga («…del artículo 12 de la Ley de Amparo»), el nombre va tras
        coma;
      · el ÓRGANO QUE ES QUEJOSO (IMSS, INFONAVIT, un municipio) es «la persona
        moral oficial quejosa/recurrente», nunca «la autoridad quejosa»: en el
        amparo el quejoso no es autoridad (art. 7o.) y sólo es «la autoridad
        recurrente» quien recurre CON ese papel (rev_4, regresión);
      · VARIAS PERSONAS (`es_plural_de_partes`): «las personas quejosas están
        legitimadas… el perjuicio que les causa»; el adjetivo concuerda con
        «personas», no con los nombres;
      · la SUCESIÓN, la COMUNIDAD, el EJIDO, con su artículo
        (`con_articulo_de_colectivo`);
      · la ETIQUETA DE ROL entre paréntesis y las VERSALES no pasan a la prosa;
      · la REVISIÓN FISCAL, por `_legitimacion_rf` (la unidad, la propia
        demandada o ninguna de las dos, con el 5o., cuarto párrafo).
    CUARTA RONDA (3-oct-2026):
      · E1: el nombre pasa por la puerta única del compositor
        (`resultandos_por_tipo.nombre_en_prosa`): las rachas en versales, las
        fórmulas de representación en minúscula, «la Titular» y el artículo de
        los colectivos, igual que en el V I S T O del mismo documento;
      · E2: VARIAS PERSONAS (`es_plural_de_partes`, que ya mira la lista antes
        que la sociedad; `plural` lo fuerza): si el texto dice su género
        (`genero_de_plural`: «ambos», «todas»; o `genero` declarado), «quienes
        están legitimados… les causa»; si no, «; la parte quejosa está
        legitimada», que es neutra y no infiere nada de los nombres;
      · E12: la representación DENTRO del nombre («…, a través de su albacea
        X», «por propio derecho y en representación de…») cuenta como
        representante: «; la parte recurrente está legitimada» (el «quien» se
        leía como el albacea); y el representante que ya va en el nombre no se
        repite;
      · E12: con `hay_norma` (o `norma_impugnada`, la razón que da la función
        del mismo nombre; cualquier valor verdadero) y un promulgador que recurre
        (Gobernador, Legislatura, Congreso…), la hipótesis del 87, primer
        párrafo, de los titulares de los órganos que emiten o promulgan la norma;
      · E9: la figura con «artículo 12» o «9o.» sin ley se completa con «de la
        Ley de Amparo» (`figura_en_prosa`) y la personería no se repite.
    QUINTA RONDA (3-oct-2026):
      · F2: la FIGURA SIN NOMBRE no se borra: «, por conducto de su {figura}
        *********», con su personería y el aviso «FALTA EL NOMBRE DEL
        REPRESENTANTE (representante)», sólo si se pasa `avisos` (el camino
        de la ficha; el viejo no los pide y queda igual);
      · F6: el separador entre la figura y el nombre es `sep_figura`, el mismo
        del resultando y del resolutivo;
      · la representación dentro del nombre se reconoce también SIN COMA
        («la Sucesión… a través de su albacea X», AD 335/2025).
    `avisos` (opcional): una lista donde dejar lo que el secretario debe
    comprobar o escribir.
    """
    m = LEGITIMACION.get(normalizar(tipo))
    if not m or not (parte or "").strip():
        return ""
    t = normalizar(tipo)
    _papel = (papel or "").strip().lower()
    parte, _rol = sin_etiqueta_de_rol(parte)
    if not _papel and _rol:
        _papel = ("autoridad" if _rol.startswith("autoridad") else
                  "tercero" if _rol.startswith("tercer") else
                  "quejoso" if _rol.startswith(("quejos", "parte quejosa")) else "")
    if t == "revision_fiscal":
        _l_rf = _legitimacion_rf(parte, representante, hueco, figura,
                                 autoridad_demandada, avisos)
        # VARIAS AUTORIDADES DEMANDADAS, EN PLURAL (revisión RF, 3-oct-2026): «…autoridad
        # demandada…, a la que la sentencia recurrida le resultó adversa» de tres
        # autoridades del SAT no concuerda.
        _dem_rf = autoridad_demandada or (_RX_EN_REPRESENTACION.split(parte or "", maxsplit=1)[-1]
                                          if _RX_EN_REPRESENTACION.search(parte or "") else parte)
        return _demandadas_en_plural(_l_rf) if varias_autoridades(_dem_rf) else _l_rf
    _por_papel = (_LEGITIMACION_REVISION_POR_PAPEL.get(_papel)
                  if t == "amparo_revision" else None)
    _hay_norma = hay_norma if hay_norma is not None else norma_impugnada
    if _por_papel and _papel == "autoridad" and _hay_norma and es_promulgador(parte):
        _por_papel = _LEGITIMACION_REVISION_POR_PAPEL["autoridad_norma"]
    if _por_papel:
        m = _por_papel
    _es_autoridad = _papel == "autoridad"
    _organo = _es_autoridad or es_organo_publico(parte)
    if moral is None:
        try:
            import promovente as _pv
            moral = _pv.es_moral(parte)
        except Exception:
            moral = False
    rep = ""
    _rep_n = " ".join(str(representante or "").split())
    # QUIEN ACTÚA POR SÍ NO SE REPRESENTA A SÍ MISMO (3-oct-2026, Q 261/2025 y Q
    # 337/2026: la recurrente «por propio derecho y en representación de» sus
    # menores quedó como su propia representante, y salía «X, por conducto de su
    # representante legal X»). Y EL REPRESENTANTE QUE YA VA DENTRO DEL NOMBRE
    # TAMPOCO (Q 342/2025: «la Sucesión…, a través de su albacea X, por conducto
    # de su representante legal X»).
    _rep_borrado = False
    if _rep_n and (misma_autoridad(_rep_n, parte) or _contenido_en(_rep_n, parte)):
        _rep_n, _rep_borrado = "", True
    _rep_dentro = bool(_RX_REPRESENTACION_EN_EL_NOMBRE.search(parte))
    if _rep_n:
        fig = figura_en_prosa(figura) or "representante legal"
        _cola = personeria_de(_papel, fig)
        # F6 (quinta ronda): el separador es el del resultando (`sep_figura`).
        rep = (f", por conducto de su {fig}{sep_figura(fig)}{_nombre_en_prosa(_rep_n)}"
               + (f", {_cola}" if _cola else ""))
    elif not _rep_borrado and isinstance(avisos, list) and _figura_sin_nombre(figura, parte, _rep_dentro):
        # LA FIGURA SIN NOMBRE NO SE BORRA (F2, quinta ronda, 3-oct-2026; AR
        # 72/2025 y Q 342/2025 del banco). La ficha traía la figura («delegado»,
        # «autorizado») y el nombre testado o perdido, y el párrafo callaba la
        # representación: «fue interpuesto por el Gobernador…» cuando lo
        # interpuso su delegado con personería del 9o., y «por la Sucesión…, a
        # través de su albacea…» cuando firmó el autorizado del 12, que es justo
        # lo que el engrose analiza. Ahora la figura va con el nombre en HUECO, su
        # personería, y un aviso con la clave del dato. Sólo para quien recibe
        # los avisos (`avisos`, el camino de la ficha): un hueco sin su aviso no
        # se escribe, y el camino viejo, que no los pide, queda como estaba.
        fig = figura_en_prosa(figura)
        _cola = personeria_de(_papel, fig)
        rep = (f", por conducto de su {fig}{sep_figura(fig)}{hueco}"
               + (f", {_cola}" if _cola else ""))
        _verbo = "promovió" if t == "amparo_directo" else "recurrió"
        _av_rep = (f"FALTA EL NOMBRE DEL REPRESENTANTE (representante): consta que {_verbo} por "
                   f"conducto de su {fig}, pero no quién; la legitimación lo deja en HUECO. "
                   f"Escríbelo en la ficha o en el hueco (está en el escrito y en el auto que "
                   f"le reconoce la personería).")
        if _av_rep not in avisos:
            avisos.append(_av_rep)
    nombre = parte.strip()
    if plural is None:
        _plural = (not _organo) and es_plural_de_partes(nombre, moral)
    else:
        _plural = bool(plural) and not _organo
    _fisica = (not _organo) and es_persona_fisica(nombre, moral)
    if _organo:
        nombre = con_articulo_de_organo(nombre)
        a = "a" if nombre.lower().startswith(("la ", "las ")) else "o"
    else:
        nombre = _nombre_en_prosa(nombre, con_articulo=True)
        a = "a" if (moral or genero_de(parte) == "a") else "o"
    fr = fraccion_5o_de(_papel, recurre_el_quejoso) or hueco
    texto = m["molde"].format(parte=nombre, rep=rep, a=a, autoridad="", fr=fr)
    _rec = t in ("amparo_revision", "queja")

    def _sustituye(sujeto_y_verbo, plural_gram: bool) -> str:
        """«, quien está legitimada» → lo que se le pase (con «está» o «se
        encuentra» según el molde); en plural, «les causa», «resultarles»."""
        out = texto
        for _v, _vp in (("está", "están"), ("se encuentra", "se encuentran")):
            for _g in ("a", "o"):
                out = out.replace(f", quien {_v} legitimad{_g}",
                                  sujeto_y_verbo(_vp if plural_gram else _v), 1)
        if plural_gram:
            for _s, _p in ((" le causa ", " les causa "), (" le resulta ", " les resulta "),
                           ("resultarle ", "resultarles "), ("por ser parte en", "por ser partes en")):
                out = out.replace(_s, _p)
        return out

    if _plural:
        # VARIAS PERSONAS (E2). El género, sólo si el texto o la ficha lo dicen.
        g = genero_de_plural(parte, genero)
        if g and not (rep or _rep_dentro):
            texto = _sustituye(lambda v: f", quienes {v} legitimad{g}s", True)
        elif g:
            _suj = (("los recurrentes" if g == "o" else "las recurrentes") if _rec
                    else ("los quejosos" if g == "o" else "las quejosas"))
            texto = _sustituye(lambda v: f"; {_suj} {v} legitimad{g}s", True)
        else:
            _suj = "la parte recurrente" if _rec else "la parte quejosa"
            texto = _sustituye(lambda v: f"; {_suj} {v} legitimada", False)
    elif rep or _rep_dentro or _fisica:
        # «…, quien está legitimada» ya no puede referirse al representante: se
        # nombra a la parte por su figura. Y EL ADJETIVO CONCUERDA CON ESA
        # FIGURA, que es femenina —«la parte», «la persona moral», «la
        # autoridad»—, no con el nombre: salía «la parte quejosa está
        # legitimado» cuando la parte era un hombre con apoderado.
        quien = ("la persona moral recurrente" if moral else "la parte recurrente") if _rec \
            else ("la persona moral quejosa" if moral else "la parte quejosa")
        if _organo:
            quien = ("la autoridad recurrente" if _es_autoridad else
                     "la persona moral oficial recurrente" if _rec else
                     "la persona moral oficial quejosa")
        texto = _sustituye(lambda v: f"; {quien} {v} legitimada", False)
    return texto


# LA REPRESENTACIÓN QUE VIAJA DENTRO DEL NOMBRE (E12): «la Sucesión…, a través de
# su albacea X», «Ana López, por propio derecho y en representación de su hija».
# CON COMA O SIN ELLA (quinta ronda, 3-oct-2026; AD 335/2025 del banco): la
# carátula la escribe «LA SUCESIÓN A BIENES DE X A TRAVÉS DE SU ALBACEA Y», sin
# coma, y con la coma obligatoria el molde dejaba «…su albacea Y, quien está
# legitimada», donde el «quien» se lee como el albacea cuando la legitimada es la
# sucesión: justo lo que E12 vino a evitar en este asunto.
_RX_REPRESENTACION_EN_EL_NOMBRE = re.compile(
    r"(?:^|[\s,;])(?:a\s+trav[ée]s|por\s+conducto)\s+de\s+su[s]?\s|"
    r"\ben\s+(?:su\s+)?representaci[óo]n\s+de",
    re.I)
_RX_TRAS_LA_REPRESENTACION = re.compile(
    r"(?:a\s+trav[ee]s|por\s+conducto)\s+de\s+sus?\s+(\w+)|en\s+(?:su\s+)?representacion\s+de")
_RX_NO_ES_FIGURA = re.compile(r"propio\s+derecho|^\s*por\s+s[íi]\b|^\s*(?:el|la)\s+mism[oa]\b", re.I)


def _figura_sin_nombre(figura: str, parte: str, rep_dentro: bool) -> bool:
    """¿Se escribe «, por conducto de su {figura} *********» (F2)? Sí, si hay
    figura y no es la parte misma ni la que ya va dentro del nombre. Con la
    representación dentro del nombre («…, a través de su albacea X», «por propio
    derecho y en representación de…»), sólo el AUTORIZADO o el DELEGADO, que son
    otra persona que actúa en el recurso (arts. 12 y 9o.), y sólo si el nombre no
    los trae ya (Q 342/2025: la sucesión por su albacea, y firmó el autorizado)."""
    f = " ".join(str(figura or "").split()).strip(" ,;")
    f = re.sub(r"(?i)^su\s+", "", f)
    if not f or _RX_NO_ES_FIGURA.search(f) or misma_autoridad(f, parte):
        return False
    if not rep_dentro:
        return True
    if not (_RX_AUTORIZADO.search(f) or _RX_DELEGADO.search(f)):
        return False
    _cab = (_sin_tildes_min(f).split() or [""])[0][:5]
    for m in _RX_TRAS_LA_REPRESENTACION.finditer(_sin_tildes_min(parte)):
        if m.group(1) and m.group(1)[:5] == _cab:
            return False
    return True


def _contenido_en(corto: str, largo: str) -> bool:
    """¿El nombre `corto` (dos palabras o más) va entero dentro de `largo`?
    Sin tildes, mayúsculas ni puntuación: «Juan Ruiz Gómez» dentro de «la
    Sucesión…, a través de su albacea JUAN RUIZ GÓMEZ»."""
    c, l_ = _sin_tildes_min(corto), _sin_tildes_min(largo)
    return len(c.split()) >= 2 and f" {c} " in f" {l_} "


# LA LEGITIMACIÓN DE LA REVISIÓN FISCAL (3-oct-2026, rev_5). El molde afirmaba de
# CUALQUIER promovente que era «unidad administrativa encargada de la defensa
# jurídica» de la autoridad demandada: de la propia autoridad demandada (RF
# 21/2025 y 6/2026: «el Jefe del Departamento de Pensiones…, en su carácter de
# unidad… de la defensa jurídica del Jefe del Departamento de Pensiones») y del
# Instituto (RF 26/2025). El 63 exige que recurra esa unidad, y el texto se la
# atribuía a quien no lo es: una afirmación sin dato, como el viejo «el MP omitió
# formular pedimento». Ahora, tres casos:
#   · recurre la UNIDAD JURÍDICA (se reconoce por su nombre): la fórmula del
#     corpus, en representación de la demandada;
#   · recurre la PROPIA DEMANDADA: «lo hizo valer {la demandada}…, por conducto
#     de {la figura que firmó}» (RF 6/2026: «por conducto de la Titular de la
#     Unidad de Asuntos Jurídicos»), y sin figura, HUECO y aviso;
#   · ninguno de los dos: sin afirmar el carácter (en representación de la
#     demandada si consta; si no, HUECO), con aviso.
# Y SE FUNDA ADEMÁS EN EL 5o., CUARTO PÁRRAFO, DE LA LFPCA («La representación
# de las autoridades corresponderá a las unidades administrativas encargadas de
# su defensa jurídica», verificado en el texto local): lo citan RF 21, 26 y 49.
_RX_UNIDAD_JURIDICA = re.compile(
    r"unidad\s+(?:de\s+)?(?:asuntos\s+)?jur[íi]dic|asuntos\s+jur[íi]dicos|servicios\s+jur[íi]dicos|"
    r"\blo\s+contencioso\b|defensa\s+jur[íi]dica|administraci[óo]n\s+(?:desconcentrada\s+|central\s+)?"
    r"(?:de\s+lo\s+contencioso|jur[íi]dica)|direcci[óo]n\s+(?:general\s+)?(?:de\s+)?(?:asuntos\s+)?"
    r"jur[íi]dic|director[a]?\s+(?:general\s+)?(?:de\s+)?(?:asuntos\s+)?jur[íi]dic|"
    r"coordinaci[óo]n\s+(?:general\s+)?jur[íi]dica|procuradur[íi]a\s+fiscal|subprocuradur[íi]a|"
    r"[áa]rea\s+jur[íi]dica|jefatura\s+(?:de\s+)?(?:servicios\s+)?jur[íi]dic", re.I)
_RX_EN_REPRESENTACION = re.compile(
    r",?\s+(?:en\s+(?:nombre\s+y\s+)?representaci[óo]n|en\s+nombre)\s+(?:de\s+(?:la|las|los)\s+|del\s+|de\s+)",
    re.I)
_FUND_LEGITIMACION_RF = ("en términos del artículo 63, párrafo primero, en relación con el 5o., "
                         "cuarto párrafo, de la Ley Federal de Procedimiento Contencioso "
                         "Administrativo")


def es_unidad_juridica(nombre: str) -> bool:
    """¿El nombre es el de la unidad encargada de la defensa jurídica (o de su
    titular)? Unidad Jurídica, de Asuntos Jurídicos, Subdirección de lo
    Contencioso, Administración Desconcentrada Jurídica, Procuraduría Fiscal…"""
    return bool(_RX_UNIDAD_JURIDICA.search(" ".join(str(nombre or "").split())))


def _sin_tildes_min(x: str) -> str:
    import unicodedata as _ud
    x = "".join(c for c in _ud.normalize("NFD", str(x or "")) if _ud.category(c) != "Mn")
    x = re.sub(r"[^\w\s]", " ", x.lower())
    x = re.sub(r"^\s*(?:el|la|los|las)\s+", "", " ".join(x.split()))
    return x


# EL CARGO Y SU ÓRGANO SON LA MISMA AUTORIDAD (E8, cuarta ronda, 3-oct-2026; RF
# 6/2026: «el Subdelegado de Prestaciones Económicas…, en representación de la
# Subdelegación de Prestaciones Económicas…», X en representación de X). El
# engrose usa las dos formas para la misma autoridad —la carátula
# «SUBDELEGACIÓN», el V I S T O «el Subdelegado»—, y «Titular de la X» es «X».
# Se comparan con la cabeza del órgano: subdelegado → subdelegación, director →
# dirección, jefe → jefatura…, y sin el «Titular de» del principio.
_CARGO_A_ORGANO = (
    (re.compile(r"^subdelegad[oa]s?\b"), "subdelegacion"),
    (re.compile(r"^delegad[oa]s?\b"), "delegacion"),
    (re.compile(r"^subdirector(?:a|es)?\b"), "subdireccion"),
    (re.compile(r"^director(?:a|es)?\b"), "direccion"),
    (re.compile(r"^jef[ea]s?\b"), "jefatura"),
    (re.compile(r"^administrador(?:a|es)?\b"), "administracion"),
    (re.compile(r"^coordinador(?:a|es)?\b"), "coordinacion"),
    (re.compile(r"^subprocurador(?:a|es)?\b"), "subprocuraduria"),
    (re.compile(r"^procurador(?:a|es)?\b"), "procuraduria"),
    (re.compile(r"^subsecretari[oa]s?\b"), "subsecretaria"),
    (re.compile(r"^secretari[oa]s?\b"), "secretaria"),
    (re.compile(r"^tesorer[oa]s?\b"), "tesoreria"),
    (re.compile(r"^contralor(?:a|es)?\b"), "contraloria"),
    (re.compile(r"^gerentes?\b"), "gerencia"),
    (re.compile(r"^(?:presidente|presidenta)\b"), "presidencia"),
)
# La adscripción que se añade tras el núcleo del cargo: «… del ISSSTE en
# Querétaro», «…, Delegación Estatal Querétaro, del Instituto…».
_RX_ADSCRIPCION = re.compile(
    r"\s+(?:del|de\s+la|de\s+los|de)\s+(?:instituto|issste|imss|infonavit|delegacion|representacion|"
    r"oficina|organo\s+de\s+operacion|servicio\s+de\s+administracion\s+tributaria|sat)\b.*$")


def _clave_de_autoridad(x: str) -> str:
    """El nombre sin tildes ni artículo, con «Titular de la X» → «X» y el cargo
    de la cabeza en su órgano («subdelegado» → «subdelegacion»)."""
    k = _sin_tildes_min(x)
    k = re.sub(r"^(?:titular|encargad[oa]\s+del\s+despacho)\s+(?:de\s+la|de\s+los|del|de)\s+", "", k)
    for rx, org in _CARGO_A_ORGANO:
        if rx.match(k):
            k = rx.sub(org, k, count=1)
            break
    return k


def _nucleo_de_autoridad(x: str) -> str:
    """El cargo sin su adscripción: hasta la primera coma y sin «del ISSSTE»,
    «de la Delegación…»; «» si no queda un núcleo de tres palabras o más."""
    import unicodedata as _ud
    t = "".join(c for c in _ud.normalize("NFD", str(x or "")) if _ud.category(c) != "Mn").lower()
    t = re.sub(r"[^\w\s,]", " ", t).split(",")[0]
    k = _clave_de_autoridad(t)
    k = _RX_ADSCRIPCION.sub("", k).strip()
    return k if len(k.split()) >= 3 else ""


def misma_autoridad(a: str, b: str) -> bool:
    """¿Nombran a la misma autoridad? Sin tildes, artículo ni puntuación: igual,
    o el nombre largo EMPIEZA por el corto (con al menos tres palabras) y sólo
    le añade su adscripción («Subdelegado de Prestaciones Económicas» y «…del
    ISSSTE»). Contener no basta: «Titular de la Unidad Jurídica… del Instituto
    de Seguridad…» contiene al Instituto y no es el Instituto (RF 4/2025).

    E8 (cuarta ronda): el cargo y su órgano cuentan como la misma autoridad
    («Subdelegado»/«Subdelegación», «Director»/«Dirección», «Jefe»/«Jefatura»,
    «Titular de la X»/«X»), y también el mismo núcleo con distinta adscripción
    («Subdelegado de Prestaciones Económicas, Delegación Estatal Querétaro, del
    ISSSTE» y «Subdelegación de Prestaciones Económicas del ISSSTE en
    Querétaro»)."""
    x, y = _sin_tildes_min(a), _sin_tildes_min(b)
    if not x or not y:
        return False
    if x == y:
        return True
    corto, largo = (x, y) if len(x) <= len(y) else (y, x)
    if len(corto.split()) >= 3 and largo.startswith(corto + " "):
        return True
    cx, cy = _clave_de_autoridad(a), _clave_de_autoridad(b)
    if cx == cy:
        return True
    corto, largo = (cx, cy) if len(cx) <= len(cy) else (cy, cx)
    if len(corto.split()) >= 3 and largo.startswith(corto + " "):
        return True
    nx, ny = _nucleo_de_autoridad(a), _nucleo_de_autoridad(b)
    if not (nx and ny):
        return False
    corto, largo = (nx, ny) if len(nx) <= len(ny) else (ny, nx)
    return corto == largo or largo.startswith(corto + " ")


def _de_x(x: str) -> str:
    """«de» + lo que sigue, con la contracción: «del Subdelegado», «de la Sala»."""
    x = (x or "").strip()
    if re.match(r"^el\s", x):
        return "del " + x[3:]
    return "de " + x


# CUARTA RONDA (E8, 3-oct-2026), lo que delataba su propio defecto y se firmaba:
#   · LA TITULAR QUE FIRMA POR SU UNIDAD NO ES «OTRA PERSONA» (RF 7/2025: «lo
#     hizo valer la Jefa de la Unidad Jurídica…, por conducto de Representante,
#     en su carácter de unidad…»). Si la figura es el mismo cargo que encabeza
#     al promovente, o está contenida en la unidad, quien firmó ES su titular:
#     «lo hizo valer {nombre}, {cargo}, en representación de {la demandada}»,
#     la aposición del corpus (RF 2 y 7/2025: «ya que lo promovió X, Jefa de la
#     Unidad Jurídica…, en representación de la autoridad demandada»);
#   · un valor TRUNCADO («Titular de la Unidad Jurídica…», la copia del ejemplo
#     del prompt) no se escribe: fuera, con aviso;
#   · un representante que es un CARGO («Titular de la Unidad de Asuntos
#     Jurídicos de la citada delegación», RF 6/2026) es la figura, no un nombre;
#   · un AUTORIZADO no interpone la revisión fiscal por la autoridad (RF
#     4/2025): el 5o., quinto párrafo, LFPCA da el autorizado a los particulares
#     —las autoridades nombran delegados— y el 63 exige a la unidad de defensa
#     jurídica; se omite del párrafo (el aviso lo da `ficha_tramite.validar`);
#   · la figura de un cargo conserva su mayúscula y, si pasa de tres palabras,
#     lleva coma antes del nombre, como en el resultando (`_representacion`).
_RX_TRUNCADO = re.compile(r"(?:…|\.{3})\s*$")
_RX_CARGO_DE_PERSONA = re.compile(
    r"^(?:(?:el|la)\s+)?(?:titular|jef[ea]|director(?:a)?|subdirector(?:a)?|delegad[oa]|"
    r"subdelegad[oa]|encargad[oa]|coordinador(?:a)?|administrador(?:a)?|gerente|procurador(?:a)?|"
    r"subprocurador(?:a)?|secretari[oa]|subsecretari[oa]|tesorer[oa]|contralor(?:a)?|"
    r"consejer[oa]\s+jur[íi]dic[oa])\b", re.I)


def _es_cargo(x: str) -> bool:
    """¿Es un cargo (o una unidad), no el nombre de una persona?"""
    x = " ".join(str(x or "").split())
    return bool(x) and (bool(_RX_CARGO_DE_PERSONA.search(x)) or
                        bool(re.search(r"\bunidad\b|\bdirecci[óo]n\b|\bdepartamento\b", x, re.I)))


def sep_figura(fig: str) -> str:
    """EL SEPARADOR ENTRE LA FIGURA Y EL NOMBRE, UNA SOLA REGLA (F6, quinta
    ronda, 3-oct-2026; Q 172 y 300 del banco: «su autorizada en términos amplios,
    Representante, interpuso» en el resultando y «su autorizada en términos
    amplios Representante, en términos del artículo 12» en la legitimación del
    mismo documento): coma si la figura cita un artículo o pasa de tres palabras;
    si no, un espacio («su apoderado legal Juan Ruiz»). La usan el resultando, la
    legitimación y el resolutivo."""
    return ", " if (re.search(r"art[íi]culo", fig or "", re.I) or len((fig or "").split()) > 3) else " "


_sep_figura = sep_figura


def _demandadas_en_plural(t: str) -> str:
    """La legitimación de la revisión fiscal con las autoridades demandadas en
    plural (`varias_autoridades`)."""
    for a, b in (("autoridad demandada en el juicio de nulidad, a la que la sentencia recurrida le "
                  "resultó adversa", "autoridades demandadas en el juicio de nulidad, a las que la "
                  "sentencia recurrida les resultó adversa"),
                 ("la autoridad demandada en el juicio de nulidad", "las autoridades demandadas en el "
                  "juicio de nulidad"),
                 ("autoridad demandada en el juicio de nulidad", "autoridades demandadas en el juicio "
                  "de nulidad"),
                 (", a la que la sentencia recurrida le resultó adversa",
                  ", a las que la sentencia recurrida les resultó adversa"),
                 ("resultó adversa a la autoridad demandada", "resultó adversa a las autoridades "
                  "demandadas")):
        t = t.replace(a, b)
    # Si las que recurren son ellas mismas (no su unidad en su representación),
    # el verbo también: «pues lo hicieron valer el Secretario…, el Jefe… y el
    # Titular…, autoridades demandadas…».
    m = re.search(r"pues lo hizo valer ((?:(?!representaci[óo]n|defensa jur[íi]dica).)+?), autoridades "
                  r"demandadas en el juicio de nulidad, a las que", t)
    if m:
        t = t[:m.start()] + "pues lo hicieron valer " + t[m.start() + len("pues lo hizo valer "):]
    return t


def _legitimacion_rf(parte: str, representante: str = "", hueco: str = "*********",
                     figura: str = "", autoridad_demandada: str = "", avisos=None) -> str:
    """El párrafo de legitimación de la revisión fiscal (ver arriba)."""
    _av = avisos if isinstance(avisos, list) else []
    unidad = " ".join(str(parte or "").split())
    aut = " ".join(str(autoridad_demandada or "").split())
    aut, _ = sin_etiqueta_de_rol(aut)
    # «X, en representación de Y» en el mismo campo: X recurre, Y es la demandada.
    m = _RX_EN_REPRESENTACION.search(unidad)
    if m:
        _repr = unidad[m.end():].strip(" ,.")
        unidad = unidad[:m.start()].strip(" ,")
        if _repr and not aut:
            aut = sin_etiqueta_de_rol(_repr)[0]
    unidad, _ = sin_etiqueta_de_rol(unidad)
    unidad_txt = con_articulo_de_organo(unidad)
    rep_n = " ".join(str(representante or "").split())
    fig_cr = " ".join(str(figura or "").split()).strip(" ,;")
    # LO TRUNCADO NO SE FIRMA.
    _truncado = False
    for _clave, _valor in (("representante", rep_n), ("figura_representante", fig_cr)):
        if _valor and _RX_TRUNCADO.search(_valor):
            _av.append(
                f"DATO TRUNCADO ({_clave} = «{_valor}»): termina en puntos suspensivos y no se "
                f"escribió en la legitimación. Cópialo completo del oficio de interposición.")
            _truncado = True
            if _clave == "representante":
                rep_n = ""
            else:
                fig_cr = ""
    # EL REPRESENTANTE QUE ES UN CARGO ES LA FIGURA (el cargo que se escribió
    # donde iba el nombre es el de quien firmó: manda sobre la figura).
    if rep_n and _es_cargo(rep_n):
        fig_cr, rep_n = rep_n, ""
    # EL AUTORIZADO NO INTERPONE POR LA AUTORIDAD: no se firma que la unidad
    # actuó «por conducto de su autorizado». EL AVISO ES DE `ficha_tramite.validar`
    # (E8, mismo hecho, E11: un aviso por dato), que ya lo dice con la ficha.
    if fig_cr and _RX_AUTORIZADO.search(fig_cr):
        fig_cr, rep_n = "", ""
    fig = figura_en_prosa(fig_cr, ley_de_amparo=False)
    # LA FIGURA QUE YA ES EL NOMBRE NO SE REPITE (RF 21/2025 del banco: «lo hizo
    # valer el Titular de la Unidad de Asuntos Jurídicos, por conducto de su
    # titular de la Unidad de Asuntos Jurídicos»), ni el representante que es el
    # propio promovente. Y SI HAY NOMBRE, ESA PERSONA ES LA TITULAR (E8).
    _titular_firma = False

    def _dentro(a_: str, b_: str) -> bool:
        return len(a_.split()) >= 2 and f" {a_} " in f" {b_} "

    if fig and (misma_autoridad(fig, unidad) or _dentro(_sin_tildes_min(fig), _sin_tildes_min(unidad))
                or _dentro(_clave_de_autoridad(fig), _clave_de_autoridad(unidad))):
        fig = ""
        _titular_firma = bool(rep_n)
    if rep_n and misma_autoridad(rep_n, unidad):
        rep_n = ""
        _titular_firma = False
    # LA FIGURA QUE SÓLO DICE QUE REPRESENTA A LA DEMANDADA NO ES UN CARGO (banco,
    # RF 49/2025: «por conducto de su representante de la autoridad demandada»):
    # se queda el nombre, si lo hay.
    if fig and re.search(r"autoridad\s+demandada|\brepresentaci[óo]n\b", fig, re.I):
        fig = ""
    # SIN FIGURA, EL NOMBRE QUE FIRMA POR UN CARGO («la Jefa de la Unidad
    # Jurídica…») ES EL DE SU TITULAR: aposición, y aviso si no hubo figura.
    if rep_n and not fig and not _titular_firma and _RX_CARGO_DE_PERSONA.search(unidad):
        _titular_firma = True
        if fig_cr == "" and not _truncado:
            _av.append(
                f"FALTA LA FIGURA DE QUIEN FIRMÓ EL OFICIO (figura_representante): se escribió que "
                f"{rep_n} lo firmó como {sin_articulo_de_prosa(unidad_txt)}. Compruébalo en el oficio "
                f"de interposición.")
    rep_txt = _nombre_en_prosa(rep_n) if rep_n else ""
    _aut_txt = con_articulo_de_organo(aut) if aut else ""
    autoridad = (f"{_aut_txt}, autoridad demandada en el juicio de nulidad" if _aut_txt
                 else "la autoridad demandada en el juicio de nulidad")
    base = f"El recurso de revisión fiscal fue interpuesto por parte legítima, {_FUND_LEGITIMACION_RF}, pues lo hizo valer "
    # LA PROPIA AUTORIDAD DEMANDADA, por conducto de quien firmó por ella.
    if aut and misma_autoridad(unidad, aut):
        if fig:
            _quien = con_articulo_de_organo(fig_cr) if es_unidad_juridica(fig) else f"su {fig}"
            if not es_unidad_juridica(fig):
                _av.append(
                    f"COMPRUEBA LA LEGITIMACIÓN DE LA REVISIÓN FISCAL: recurre la propia autoridad "
                    f"demandada por conducto de «{fig}», que no se reconoce como la unidad encargada "
                    f"de su defensa jurídica (arts. 63, párrafo primero, y 5o., cuarto párrafo, LFPCA).")
            _por = f", por conducto {_de_x(_quien)}" + (f", {rep_txt}" if rep_txt else "")
        else:
            _por = f", por conducto de {hueco}"
            _av.append(
                "FALTA LA UNIDAD JURÍDICA QUE FIRMÓ EL OFICIO (art. 63 LFPCA): recurre la propia "
                "autoridad demandada, y la revisión fiscal sólo la interpone por conducto de la "
                "unidad administrativa encargada de su defensa jurídica. Escribe en la ficha la "
                "figura del representante (p. ej., «Titular de la Unidad de Asuntos Jurídicos») "
                "y su nombre.")
        return (f"{base}{_aut_txt}, autoridad demandada en el juicio de nulidad, a la que la "
                f"sentencia recurrida le resultó adversa{_por}.")
    # LA TITULAR QUE FIRMA POR SU UNIDAD JURÍDICA: la aposición del corpus.
    if _titular_firma and rep_txt and es_unidad_juridica(unidad):
        if _RX_CARGO_DE_PERSONA.search(unidad):
            _cargo = sin_articulo_de_prosa(unidad_txt)
        else:
            # «Unidad Jurídica de la Delegación…» con la figura «Titular de la
            # Unidad Jurídica»: la persona es «Titular de la Unidad Jurídica…».
            _cab = (fig_cr.split()[:1] or ["Titular"])[0]
            _cargo = f"{_cab[:1].upper()}{_cab[1:].lower()} {_de_x(unidad_txt)}"
        return (f"{base}{rep_txt}, {_cargo}, en representación {_de_x(autoridad)}, a la que la "
                f"sentencia recurrida le resultó adversa.")
    # Sin nombre, a la unidad no se le añade su propio titular: no dice nada que
    # el nombre de la unidad no diga.
    rep = ""
    if rep_txt:
        rep = (f", por conducto de su {fig}{_sep_figura(fig)}{rep_txt}" if fig
               else f", por conducto de {rep_txt}")
    elif fig and not es_unidad_juridica(unidad):
        rep = f", por conducto de su {fig}"
    # LA UNIDAD JURÍDICA: la fórmula del corpus.
    if es_unidad_juridica(unidad):
        return (f"{base}{unidad_txt}{rep}, en su carácter de unidad administrativa encargada "
                f"de la defensa jurídica {_de_x(autoridad)}, a la que la sentencia recurrida "
                f"le resultó adversa.")
    if _aut_txt:
        _av.append(
            f"COMPRUEBA LA LEGITIMACIÓN DE LA REVISIÓN FISCAL: «{unidad}» no se reconoce como la "
            f"unidad encargada de la defensa jurídica {_de_x(_aut_txt)} (arts. 63, párrafo primero, y "
            f"5o., cuarto párrafo, LFPCA). El considerando dice que recurrió en su representación, "
            f"sin afirmar que sea esa unidad.")
        return (f"{base}{unidad_txt}{rep}, en representación {_de_x(autoridad)}, a la que la "
                f"sentencia recurrida le resultó adversa.")
    _av.append(
        f"EL CARÁCTER CON QUE RECURRE «{unidad}» VA EN HUECO EN LA LEGITIMACIÓN: la revisión "
        f"fiscal sólo la interpone la unidad administrativa encargada de la defensa jurídica de "
        f"la autoridad demandada (arts. 63, párrafo primero, y 5o., cuarto párrafo, LFPCA). "
        f"Escribe en la ficha la autoridad demandada y, si recurre ella misma, la unidad que "
        f"firmó el oficio.")
    return (f"{base}{unidad_txt}{rep}, en su carácter de {hueco}, en el juicio de nulidad en el "
            f"que la sentencia recurrida resultó adversa a la autoridad demandada.")


# LOS ÓRGANOS Y LOS SERVIDORES PÚBLICOS SE NOMBRAN CON SU ARTÍCULO (3-oct-2026).
# «interpuesto por Director de Ingresos» no es español; «interpuesto por Juan
# Pérez» sí. Se distingue por la forma del nombre: un cargo o un órgano al
# frente, y nunca la forma de una sociedad (S.A., S.C., A.C.), que es persona
# moral privada y va sin artículo.
_RX_ORGANO_PUBLICO = re.compile(
    r"^(?:(?:el|la|los|las)\s+)?(?:"
    r"administraci[óo]n|administrador[a]?|director[a]?|direcci[óo]n|subdirector[a]?|"
    r"subdirecci[óo]n|titular|jef[ea]|secretar[íi]a|secretari[oa]|subsecretar\w*|"
    r"procurador[a]?|procuradur[íi]a|fiscal[íi]a|fiscal|juez[a]?|juzgado|tribunal|sala|"
    r"junta|magistrad[oa]|presidente|presidenta|ayuntamiento|municipio|instituto|"
    r"comisi[óo]n|unidad|delegad[oa]|delegaci[óo]n|subdelegaci[óo]n|coordinaci[óo]n|"
    r"coordinador[a]?|gerencia|gerente|tesorer[íi]a|tesorer[oa]|recaudaci[óo]n|"
    r"recaudador[a]?|oficina|registr[oa]dor[a]?|congreso|gobernador[a]?|gobierno|"
    r"consejo|servicio\s+de\s+administraci[óo]n|organismo|autoridad|agente|"
    r"encargad[oa]|actuari[oa]|notari[oa]|polic[íi]a|guardia|ejecutor[a]?|"
    r"verificador[a]?|inspector[a]?|visitador[a]?|contralor[íi]a|contralor[a]?)\b", re.I)
_RX_SOCIEDAD = re.compile(r"\bS\.?\s*A\.?\b|\bS\.?\s*C\.?\b|\bA\.?\s*C\.?\b|"
                          r"\bS\.?\s*de\s*R\.?\s*L\.?\b|\bsociedad\b", re.I)


def es_organo_publico(nombre: str) -> bool:
    """¿El nombre es el de un órgano o un servidor público, no una persona?"""
    n = " ".join(str(nombre or "").split())
    return bool(n and _RX_ORGANO_PUBLICO.search(n) and not _RX_SOCIEDAD.search(n))


# UN SOLO CONVERTIDOR DE NOMBRES A PROSA (E1, cuarta ronda, 3-oct-2026). El
# considerando de legitimación pasaba el nombre por sus propias reglas —las
# versales enteras y nada más— y el V I S T O del mismo documento por las del
# compositor: «a Través de Su Albacea» en una sección y «a través de su albacea»
# en otra (Q 342/2025), «el gobernador» y «el Gobernador» (AR 208/2025), «la
# Titular» que se volvía «el Titular» (RF 4/2025). La puerta es
# `resultandos_por_tipo.nombre_en_prosa`; si aún no acepta `con_articulo` o no
# se puede importar, lo de antes (el artículo de los colectivos lo pone
# `con_articulo_de_colectivo`).
def _nombre_en_prosa(nombre: str, autoridad: bool = False, con_articulo: bool = False) -> str:
    n = " ".join(str(nombre or "").split())
    if not n:
        return ""
    try:
        import resultandos_por_tipo as _rpt_n
        _f = getattr(_rpt_n, "nombre_en_prosa", None)
        if callable(_f):
            try:
                r = _f(n, autoridad=autoridad, con_articulo=con_articulo)
            except TypeError:
                r = _f(n, autoridad=autoridad)
            r = " ".join(str(r or "").split()) or n
            return con_articulo_de_colectivo(r) if (con_articulo and not autoridad) else r
    except Exception:
        pass
    return con_articulo_de_colectivo(n) if (con_articulo and not autoridad) else n


def con_articulo_de_organo(nombre: str) -> str:
    """«Director de Ingresos…» → «el Director de Ingresos…». Usa la puerta
    única del compositor (`resultandos_por_tipo.nombre_en_prosa`, E1), que
    conserva «la Titular» del papel y pasa las versales a prosa; si no se puede
    importar, la regla de artículos de `documento_generado._con_articulo`; y si
    tampoco, el nombre tal cual."""
    n = " ".join(str(nombre or "").split()).strip(" .,;")
    n = sin_etiqueta_de_rol(n)[0]
    if not n:
        return ""
    _art = articulo_del_papel(n)
    _p = _nombre_en_prosa(n, autoridad=True)
    if re.match(r"(?i)^(?:el|la|los|las)\s", _p or ""):
        return _con_el_articulo_del_papel(_p, _art)
    try:
        import documento_generado as _dg_a
        # LAS VERSALES DE LA CARÁTULA NO PASAN A LA PROSA (3-oct-2026, AR
        # 208/2025: «el GOBERNADOR DEL ESTADO DE QUERÉTARO»; RF 2/2025: «lo hizo
        # valer el JEFA DE LA UNIDAD JURÍDICA…»). `_nombre_de_organo` sólo rehace
        # lo que viene entero en mayúsculas y respeta romanos y siglas.
        if n == n.upper() and any(c.isalpha() for c in n):
            n = _dg_a._nombre_de_organo(n) or n
        return _con_el_articulo_del_papel(_dg_a._con_articulo(n) or n, _art)
    except Exception:
        return n


# EL ARTÍCULO DEL PAPEL SE CONSERVA EN LOS CARGOS DE DOBLE GÉNERO (F5, quinta
# ronda, 3-oct-2026; AD 128/2025 del banco: la ficha trae «la Oficial Mayor y
# Coordinadora de Recursos Humanos del Municipio de Cadereyta» y el documento
# decía «el Oficial Mayor»). Titular, Oficial, Fiscal, Agente, Representante,
# Juez y Encargado no dicen el género por su forma: lo dice el artículo con que
# el papel los nombra, y sólo ése se firma. Sin artículo en el papel, el
# masculino genérico de siempre (y el aviso «EL GÉNERO DEL CARGO NO CONSTA»
# lo da el compositor). Antes sólo «la Titular» se respetaba.
_CARGO_DOBLE_GENERO = r"(?:titular|oficial|fiscal|agente|representante|juez|encargad[oa])(?:es|s)?\b"
_RX_CARGO_DOBLE_GENERO = re.compile(r"^\s*(el|la|los|las)\s+(?=" + _CARGO_DOBLE_GENERO + ")", re.I)


def articulo_del_papel(crudo: str) -> str:
    """«la», «el», «las» o «los» si el valor del papel nombra un cargo de doble
    género con su artículo («la Oficial Mayor», «LA FISCAL GENERAL»); si no,
    «»."""
    m = _RX_CARGO_DOBLE_GENERO.match(" ".join(str(crudo or "").split()))
    return m.group(1).lower() if m else ""


def _con_el_articulo_del_papel(prosa: str, art: str) -> str:
    """La prosa con el artículo del papel (`articulo_del_papel`) cuando el
    convertidor puso otro ante el mismo cargo."""
    if not art or not prosa:
        return prosa
    m = _RX_CARGO_DOBLE_GENERO.match(prosa)
    if m and m.group(1).lower() != art:
        return art + prosa[m.end(1):]
    return prosa


def rotulo_materia_de(tipo: str) -> str:
    return _ROTULO_MATERIA.get(normalizar(tipo), "Cuestión a resolver.")


def materia_a_resolver(tipo: str) -> str:
    return _MATERIA_A_RESOLVER.get(
        normalizar(tipo), _MATERIA_A_RESOLVER["amparo_directo"])


def encabezado_de(tipo: str, materia: str = "", numero: str = "") -> str:
    """El encabezado del asunto, compuesto de lo que ya se eligió."""
    t = normalizar(tipo)
    plantilla = _ENCABEZADO.get(t)
    if not plantilla:
        return (numero or "").strip()
    _m = (materia or "").strip().lower()
    _m = _MATERIA_EN_FEMENINO.get(_m, _m)
    mat = _MATERIA_ENCABEZADO.get(t, {}).get(_m, (materia or "").strip().upper())
    return " ".join(plantilla.format(materia=mat, numero=(numero or "").strip()).split())


# ═══ LOS ASUNTOS RELACIONADOS, SÓLO SI EL SECRETARIO LOS MARCA (C6, 3-oct-2026) ═══
# David: «siempre y cuando haya asuntos relacionados. No vamos a meter conexidad
# en automático. Hay que habilitar en el taller la opción de con un clic precisar
# si existen asuntos relacionados y con ello se genera el considerando». Nada de
# esto se lee de los papeles: la lista la da el secretario en el taller (la
# ficha la guarda en `relacionados`, fuente «secretario», hasta 4) y de aquí
# salen tres piezas, todas puras:
#   · el rubro, «RELACIONADO CON EL AMPARO DIRECTO CIVIL 452/2025.» (AD
#     456/2025 y 552/2024, Q 24/2026, AD 469/2024 del banco, que lo ponen bajo
#     el número);
#   · lo que el V I S T O inserta tras el número («…, relacionado con el amparo
#     en revisión civil 298/2025, interpuesto por…»);
#   · el considerando: conexidad si se resuelven en la misma sesión (con el 64
#     de la LFPCA sólo para el par amparo directo ↔ revisión fiscal, AD
#     469/2024: «Si el particular interpuso amparo directo contra la misma
#     resolución o sentencia impugnada mediante el recurso de revisión, el
#     Tribunal Colegiado de Circuito que conozca del amparo resolverá el citado
#     recurso, lo cual tendrá lugar en la misma sesión en que decida el
#     amparo»); hecho notorio si ya se resolvieron (CFPC 88 o CNPCF 269, según
#     la sede, `supletorio`).
# LA MATERIA CONCUERDA IGUAL QUE EN EL ENCABEZADO DE ESE TIPO (integración,
# 3-oct-2026; A1 dudó y el contrato decía «amparo en revisión administrativo»):
# el renglón «RELACIONADO CON EL AMPARO EN REVISIÓN ADMINISTRATIVA 12/2026» dice
# lo mismo que diría el encabezado de ese amparo en revisión («AMPARO EN
# REVISIÓN ADMINISTRATIVA: 12/2026»), y la prosa, lo mismo en minúsculas. Sale
# de la misma tabla (`_MATERIA_ENCABEZADO`): «amparo directo administrativo»,
# «amparo en revisión administrativa», «recurso de queja administrativa»; la
# revisión fiscal no lleva materia. Lo familiar se registra como civil.
TIPOS_RELACIONADOS = ("amparo_directo", "amparo_revision", "queja", "revision_fiscal")
ESTADOS_RELACIONADO = ("misma_sesion", "resuelto")
MAX_RELACIONADOS = 4
_NOMBRE_RELACIONADO = {
    "amparo_directo": "amparo directo",
    "amparo_revision": "amparo en revisión",
    "queja": "recurso de queja",
    "revision_fiscal": "recurso de revisión fiscal",
}
_RX_NUMERO_RELACIONADO = re.compile(r"^\s*(\d{1,6})\s*/\s*(\d{4})\s*$")
_RX_NUMERO_EN_TEXTO = re.compile(r"(\d{1,6})\s*/\s*(\d{4})\b")


def _materia_en_prosa(tipo: str, materia: str = "") -> str:
    """La materia concordada con el nombre del asunto como en su encabezado
    (`encabezado_de`), en minúsculas; «» en la revisión fiscal o sin materia."""
    t = normalizar(tipo)
    m = " ".join(str(materia or "").split()).lower()
    if t == "revision_fiscal" or not m:
        return ""
    m = _MATERIA_EN_FEMENINO.get(m, m)
    if m == "familiar":
        m = "civil"
    return _MATERIA_ENCABEZADO.get(t, {}).get(m, m.upper()).lower()


def _numero_normalizado(numero) -> str:
    """«452/2025» si el valor es exactamente un número de asunto; «» si no."""
    m = _RX_NUMERO_RELACIONADO.match(str(numero or ""))
    return f"{int(m.group(1))}/{m.group(2)}" if m else ""


def relacionados_validos(lista) -> list:
    """La lista tal como la usan las piezas: tipo conocido, número N/AAAA,
    estado «misma_sesion» (por omisión) o «resuelto», sin duplicados y hasta
    `MAX_RELACIONADOS`. Lo que no cumple se queda fuera en silencio (los
    avisos los da la ficha, `ficha_tramite.de_formulario`)."""
    out, vistos = [], set()
    for r in (lista if isinstance(lista, (list, tuple)) else []):
        if not isinstance(r, dict):
            continue
        t = normalizar(str(r.get("tipo") or ""))
        num = _numero_normalizado(r.get("numero"))
        if t not in TIPOS_RELACIONADOS or not num or (t, num) in vistos:
            continue
        est = str(r.get("estado") or "").strip().lower()
        vistos.add((t, num))
        out.append({"tipo": t, "numero": num,
                    "estado": "resuelto" if est == "resuelto" else "misma_sesion"})
    return out[:MAX_RELACIONADOS]


def relacionado_en_prosa(r: dict, materia: str = "") -> str:
    """«amparo directo civil 452/2025», «amparo en revisión administrativa
    12/2026», «recurso de queja administrativa 24/2026», «recurso de revisión
    fiscal 33/2024». «» si el relacionado no es válido."""
    v = relacionados_validos([r])
    if not v:
        return ""
    t, num = v[0]["tipo"], v[0]["numero"]
    return " ".join(f"{_NOMBRE_RELACIONADO[t]} {_materia_en_prosa(t, materia)} {num}".split())


def relacionados_en_prosa(lista, materia: str = "") -> str:
    """«el amparo directo civil 452/2025» / «el amparo directo civil 452/2025
    y con el recurso de revisión fiscal 33/2024» (tres: «el X, con el Y y con
    el Z»). Va tras «relacionado con» en el V I S T O."""
    xs = ["el " + relacionado_en_prosa(r, materia) for r in relacionados_validos(lista)]
    if not xs:
        return ""
    if len(xs) == 1:
        return xs[0]
    return ", con ".join(xs[:-1]) + " y con " + xs[-1]


def rotulo_relacionados(lista, materia: str = "") -> str:
    """El renglón del rubro, en versales: «RELACIONADO CON EL AMPARO DIRECTO
    CIVIL 452/2025.»; «» sin relacionados."""
    p = relacionados_en_prosa(lista, materia)
    return f"RELACIONADO CON {p.upper()}." if p else ""


def _este_asunto_en_prosa(tipo: str, numero: str, materia: str) -> str:
    """Cómo se nombra «el presente»: «juicio de amparo directo civil 456/2025»,
    «amparo en revisión civil 298/2025», «recurso de queja civil 24/2026»,
    «recurso de revisión fiscal 33/2024»."""
    t = normalizar(tipo) or "amparo_directo"
    m = _RX_NUMERO_EN_TEXTO.search(str(numero or ""))
    num = f"{int(m.group(1))}/{m.group(2)}" if m else ""
    nombre = {"amparo_directo": "juicio de amparo directo"}.get(t, _NOMBRE_RELACIONADO[t])
    return " ".join(f"{nombre} {_materia_en_prosa(t, materia)} {num}".split())


def _par_del_64(este: str, otro: str) -> bool:
    """¿Amparo directo y revisión fiscal? Es el único par que funda la misma
    sesión en el artículo 64 de la LFPCA."""
    return {este, otro} == {"amparo_directo", "revision_fiscal"}


def considerando_relacionados(tipo: str, numero: str, materia: str, lista: list,
                              hecho_notorio: str) -> tuple:
    """(rótulo, texto) del considerando de los asuntos relacionados; ("", "")
    sin lista. Rótulo «Conexidad.» si todos se resuelven en la misma sesión,
    «Hecho notorio.» si todos ya se resolvieron, «Asuntos relacionados.» si hay
    de los dos (un párrafo por estado, separados por "\\n"). `hecho_notorio` es
    `supletorio(tribunal, ciudad)["hecho_notorio"]`; vacío, el de la regla
    general (CFPC)."""
    t = normalizar(tipo) or "amparo_directo"
    m = _RX_NUMERO_EN_TEXTO.search(str(numero or ""))
    propio = f"{int(m.group(1))}/{m.group(2)}" if m else ""
    xs = [r for r in relacionados_validos(lista)
          if not (r["tipo"] == t and r["numero"] == propio)]
    if not xs:
        return ("", "")
    misma = [r for r in xs if r["estado"] == "misma_sesion"]
    hechos = [r for r in xs if r["estado"] == "resuelto"]
    rotulo = ("Asuntos relacionados." if (misma and hechos)
              else "Conexidad." if misma else "Hecho notorio.")
    este = _este_asunto_en_prosa(t, numero, materia)
    indice = "del índice de este Tribunal Colegiado de Circuito"
    parrafos = []
    if misma:
        par = [r for r in misma if _par_del_64(t, r["tipo"])]
        otros = [r for r in misma if r not in par]
        if par:
            _ambos = "ambos" if len(par) == 1 else "todos"
            p = (f"Con vista en la conexión que guarda el presente {este} con "
                 f"{relacionados_en_prosa(par, materia)}, {_ambos} {indice}, se resuelven "
                 f"en la misma sesión, con fundamento en el artículo 64 de la Ley Federal de "
                 f"Procedimiento Contencioso Administrativo, porque en {_ambos} se impugna "
                 f"la misma sentencia.")
            if otros:
                # LOS DEMÁS, CON LA FÓRMULA GENERAL Y EN SU ORACIÓN: el 64 no se
                # afirma de un par que no es amparo directo ↔ revisión fiscal.
                p += (f" Asimismo, con vista en la conexión que guarda el presente asunto "
                      f"con {relacionados_en_prosa(otros, materia)}, también {indice}, se "
                      f"resuelven en la misma sesión, a fin de evitar el dictado de "
                      f"resoluciones contradictorias.")
        else:
            _ambos = "ambos" if len(otros) == 1 else "todos"
            p = (f"Con vista en la conexión que guarda el presente {este} con "
                 f"{relacionados_en_prosa(otros, materia)}, {_ambos} {indice}, se resuelven "
                 f"en la misma sesión, a fin de evitar el dictado de resoluciones "
                 f"contradictorias.")
        parrafos.append(p)
    if hechos:
        hn = " ".join(str(hecho_notorio or "").split()) or supletorio()["hecho_notorio"]
        rels = [relacionado_en_prosa(r, materia) for r in hechos]
        if len(rels) == 1:
            p = (f"Se invoca como hecho notorio, en términos de {hn}, la ejecutoria dictada "
                 f"por este Tribunal Colegiado de Circuito en el {rels[0]}, relacionado con "
                 f"el presente asunto.")
        else:
            _en = ", en el ".join(rels[:-1]) + " y en el " + rels[-1]
            p = (f"Se invocan como hecho notorio, en términos de {hn}, las ejecutorias "
                 f"dictadas por este Tribunal Colegiado de Circuito en el {_en}, "
                 f"relacionados con el presente asunto.")
        parrafos.append(p)
    return (rotulo, "\n".join(_contraer_articulo(p) for p in parrafos))


def ley_de_la_via(tipo: str, sede_del_acto: str = "",
                  cuaderno: str = "") -> str:
    """El marco normativo de la vía, para decírselo al modelo antes de escribir.

    La regla del acto reclamado va en LAS CUATRO vías —la confusión no es de un
    tipo de asunto, es de para qué sirve cada ley— pero CAMBIA DE FORMA según lo
    que el sistema haya podido derivar del expediente:

      · sede ordinaria y cuaderno principal → se dice, en positivo, cuál es la
        ley del acto y no se nombra la otra. Ver `_ACTO_ORDINARIO`.
      · el acto lo dictó un órgano de amparo, o se recurre lo resuelto en el
        incidente de suspensión → esa ley SÍ funda. Ver `_ACTO_DE_AMPARO`.
      · no se pudo derivar → la regla general, que va en negativo porque tiene
        que cubrir los dos casos.
    """
    t = LEY_DE_LA_VIA.get(normalizar(tipo), "")
    if not t:
        return ""
    _s = (sede_del_acto or "").strip().lower()
    _c = (cuaderno or "").strip().lower()
    if _s == "amparo" or _c == "incidental":
        return t + _ACTO_DE_AMPARO
    if _s == "ordinaria" and _c == "principal":
        return t + _ACTO_ORDINARIO
    return t + _ACTO_NO_ES_AMPARO

# LA VENTANA NO PUEDE SALTAR A LA CITA SIGUIENTE. La primera versión permitía
# 110 caracteres cualesquiera entre el número y «Ley de Amparo», y con eso
# acusó al proyecto por escribir lo correcto:
#
#   «los requisitos de la sentencia se examinan conforme al artículo 50 de la
#    Ley Federal de Procedimiento Contencioso Administrativo Y NO con base en
#    el artículo 74 de la Ley de Amparo»
#
# El 50 es de la LFPCA y la frase lo dice; la ventana se lo saltó y lo enganchó
# a la ley de la cita de al lado. Ahora el hueco se corta ante otro «artículo»
# y ante cualquier otro nombre de ley o código: la ley de una cita es la que
# viene ANTES de que empiece otra.
_RX_CITA_LA = re.compile(
    r"art[íi]culos?\s+((?:\d{1,3}(?:\s*(?:,|y|e)\s*)?)+)"
    # LA CONSTITUCIÓN TAMBIÉN CORTA LA VENTANA, y faltaba. El documento dice
    # «el artículo 14 de la CONSTITUCIÓN POLÍTICA de los Estados Unidos
    # Mexicanos» y la ventana saltaba por encima de ese nombre hasta encontrar
    # «Ley de Amparo» más abajo, acusando de citarla un artículo que es de la
    # Carta Magna. Duodécima retirada, y la misma causa que las anteriores: la
    # ley de una cita es la que viene ANTES de que empiece otra.
    r"(?:(?!art[íi]culos?\s+\d|\bLey\s+(?!de\s+Amparo)|\bC[óo]digo\s|"
    r"\bConstituci[óo]n\s|\bReglamento\s|\bAcuerdo\s)[^.;]){0,110}?"
    r"\bLey\s+de\s+Amparo", re.I)


# CITAR PARA DESCARTAR ES RAZONAR BIEN, Y HAY QUE PREMIARLO. El proyecto
# regenerado escribió esto, que es exactamente lo que se le pidió:
#
#   «los requisitos de la sentencia se examinan conforme al artículo 50 de la
#    Ley Federal de Procedimiento Contencioso Administrativo Y NO CON BASE EN
#    el artículo 74 de la Ley de Amparo»
#
# Acusarlo por nombrar el 74 castiga al que distingue y premia al que calla. Lo
# que se persigue es APLICAR la ley ajena, no mencionarla para apartarla. Es la
# octava vez hoy que una comprobación mía acusa al documento correcto, y la
# regla no cambia: si acusa al que hace bien, la que está mal es la regla.
# Y LOS GIROS QUE FALTABAN, vistos en el proyecto de hoy. El texto dice, y dice
# BIEN: «tratándose de una revisión fiscal promovida por una autoridad conforme
# al artículo 63 de la LFPCA, NO OPERA LA SUPLENCIA de la queja prevista en el
# artículo 79 de la Ley de Amparo». Es exactamente la regla que se le enseñó,
# aplicada y explicada, y la comprobación lo acusaba. Décima retirada.
_RX_DESCARTA = re.compile(
    r"(?:\bno\b\s+(?:con\s+base\s+en|conforme\s+a|en\s+t[ée]rminos\s+de|"
    r"por|se\s+rige|resulta\s+aplicable|es\s+aplicable|aplica|"
    r"opera|cabe|hay|procede|existe|resulta|rige|gobierna|se\s+invoca|"
    r"se\s+funda|se\s+surte)|"
    r"sin\s+que\s+(?:sea|resulte)\s+aplicable|tampoco\s+\w+|"
    r"a\s+diferencia\s+de|\bno\s+as[íi]\b|en\s+lugar\s+de|"
    r"y\s+no\b|ni\s+(?:por|conforme|con\s+base)|"
    r"(?:no|nunca)\s+(?:rige|gobierna|se\s+invoca))"
    # LA COLA ERA CORTA. Entre la negación y la cita cabe una subordinada
    # entera: «NO OPERA la suplencia de la queja prevista en el ARTÍCULO 79»
    # son 48 caracteres y el tope estaba en 40, así que el control sintético
    # pasaba y el documento real no. Se mide sobre la frase de verdad, no sobre
    # la que uno se inventa para probar.
    r"[^.;]{0,90}$", re.I)


def _amparo_fuera_de_lugar(texto: str) -> list:
    """En la revisión fiscal: los artículos de la Ley de Amparo que no toca.

    MEDIDO CONTRA EL ENGROSE FIRMADO: el secretario cita el 92 y sólo el 92.
    El proyecto generado citaba cinco —19, 74, 76, 79 y 217— de los cuales el
    74 (requisitos de la sentencia de AMPARO, cuando aquí rige el 50 de la
    LFPCA), el 76 (corrección de la cita de preceptos en el amparo) y el 79
    (suplencia de la queja, que no existe en este recurso porque quien recurre
    es la autoridad) son la ley equivocada aplicada a un asunto que no es suyo.
    """
    fuera, vistos = [], set()
    for m in _RX_CITA_LA.finditer(texto or ""):
        # Lo que va justo ANTES de la cita dice si se aplica o se aparta.
        antes = (texto or "")[max(0, m.start() - 90):m.start()]
        if _RX_DESCARTA.search(antes):
            continue
        for num in re.findall(r"\d{1,3}", m.group(1)):
            n_ = int(num)
            if n_ in _LA_EN_FISCAL or n_ in vistos:
                continue
            vistos.add(n_)
            fuera.append((
                f"artículo {n_} de la Ley de Amparo",
                f"el artículo {n_} de la Ley de Amparo no gobierna la revisión "
                f"fiscal. El artículo 63, último párrafo, de la LFPCA remite a "
                f"esa ley SÓLO «en cuanto a la regulación del recurso de "
                f"revisión» —los artículos 81 a 96—; fuera de ahí rige la "
                f"propia LFPCA (el 50 para los requisitos de la sentencia) y "
                f"la Ley Orgánica del Poder Judicial de la Federación"))
    return fuera

# Dónde se aplica la regla dura. Fuera del marco competencial y de
# procedencia, una cita de otra ley puede ser legítima —un criterio análogo,
# una remisión— y prohibirla sería empobrecer el estudio.
_RX_MARCO = re.compile(
    r"(?:PRIMERO\.\s*)?Competencia\.(.{0,2500}?)"
    r"(?=\n\s*(?:SEGUNDO|TERCERO|CUARTO)\.|\Z)|"
    r"Procedencia\.(.{0,1800}?)(?=\n\s*[A-ZÁÉÍÓÚ]{5,}\.|\Z)", re.S | re.I)


def marco_de(texto: str) -> str:
    """El marco competencial y de procedencia, aislado del resto."""
    return " ".join(m.group(1) or m.group(2) or ""
                    for m in _RX_MARCO.finditer(texto or ""))


def preceptos_ajenos(tipo: str, texto: str, solo_marco: bool = True) -> list:
    """Los preceptos de otra vía que se colaron. [(patrón, por qué)]"""
    # EN LA REVISIÓN FISCAL SE MIRA EL DOCUMENTO ENTERO, y ésa es la lección de
    # este caso. La regla estaba acotada al marco competencial porque una cita
    # de otra ley en el estudio puede ser legítima —un criterio análogo, una
    # remisión—. Pero aquí no era analogía: eran los requisitos de la sentencia
    # de AMPARO y la suplencia de la queja aplicados como derecho vigente al
    # fondo de un recurso que no se rige por esa ley. Acotar la comprobación al
    # primer considerando dejó pasar las tres.
    if normalizar(tipo) == "revision_fiscal":
        return _amparo_fuera_de_lugar(texto)

    reglas = PRECEPTOS_AJENOS.get(normalizar(tipo), [])
    if not reglas:
        return []
    t = marco_de(texto) if solo_marco else (texto or "")
    if solo_marco and not t.strip():
        return []
    return [(p, por) for p, por in reglas if re.search(p, t, re.I)]


# ═══════════════════════════════════════════════════════════════════════════
# CONFIRMAR UNA SENTENCIA QUE CONCEDIÓ EL AMPARO
# ═══════════════════════════════════════════════════════════════════════════
# David, sobre la revisión 650/2025: «Hay muchas sentencias en las que se
# concede el amparo y, al calificar como infundados los agravios, el efecto
# siempre será confirmar la sentencia en sus términos. En este caso basta con
# reproducir el resolutivo de la sentencia recurrida y en la parte final
# modificar "para los efectos precisados en la sentencia recurrida" (porque
# somos órgano revisor cuando se trata de amparo en revisión)».
#
# LO QUE SE ESCRIBÍA: «La Justicia de la Unión ampara y protege a {quejoso}, en
# términos del último considerando de la resolución recurrida». Dice a quién se
# ampara y se calla contra qué acto, con qué alcance y para qué efectos. El
# único precedente de la carpeta que confirma amparando —ARA 361/2025— sí lo
# dice: «…ampara y protege a [los quejosos], CONTRA LOS ARTÍCULOS 90 Y 99 de la
# Ley de Hacienda del Estado de Querétaro, por las razones y efectos
# especificados en el considerando sexto DE LA SENTENCIA QUE SE REVISA».
#
# ── «EN LA MATERIA DE LA REVISIÓN» NO VA SIEMPRE, Y ESTÁ MEDIDO ──
# David: «en el resolutivo primero puse "En la materia de la revisión, se
# confirma la sentencia recurrida". Esto porque había aspectos que no fueron
# combatidos por la recurrente como la concesión del amparo en sus términos,
# solo se dolió de las convivencias».
#
# La fórmula aparece en 0 de los 381 resolutivos legibles de la carpeta del
# tribunal: no es la fórmula de la casa. Y el 27% de los engroses SÍ habla de
# aspectos no combatidos sin usarla. Así que no se pone por costumbre ni por
# rama: se pone cuando el estudio de este proyecto dice que algo quedó fuera de
# la revisión, que es la condición que él describe. Si el estudio no lo dice,
# el punto es el de siempre.
CONFIRMA_PARCIAL = "PRIMERO. En la materia de la revisión, se confirma la sentencia recurrida."
CONFIRMA_ENTERA = "PRIMERO. Se confirma la sentencia recurrida."


def puntos_confirma_concede(resolutivo_reproducido: str = "",
                            parcial: bool = False) -> list:
    """Los dos puntos de una revisión que confirma una sentencia que concedió.

    `resolutivo_reproducido` es el del juzgado, ya sin su ordinal y con la cola
    apuntando a la sentencia recurrida, tal como lo devuelve
    `fase_rama.resolutivo_recurrida`. Si viene vacío —porque el resolutivo del
    juzgado no se pudo leer con seguridad, o porque tenía más de un punto— se
    escribe la fórmula genérica de siempre, que dice menos pero no dice de más.
    """
    primero = CONFIRMA_PARCIAL if parcial else CONFIRMA_ENTERA
    if (resolutivo_reproducido or "").strip():
        return [primero, "SEGUNDO. " + resolutivo_reproducido.strip()]
    return [primero, RAMAS_REVISION["confirma_concede"]["puntos"][1]]


# ═══ REVOCAR UNA CONCESIÓN ES NEGAR LO QUE ELLA CONCEDIÓ (28-sep-2026) ═══════
# AR 631/2025: la fórmula genérica —«no ampara ni protege a {quejoso}, contra
# el acto reclamado a {responsable_originaria}»— se llena con lo tecleado en el
# formulario (en un recurso, QUIEN RECURRE) y con la primera autoridad que
# nombran los antecedentes; así salió amparada la recurrente, que era la
# tercera interesada, contra un juzgado cuyo acto se había sobreseído. El
# resolutivo del juzgado ya dice a quién amparó y contra qué acto: revocar esa
# concesión es negar ESO, con sus palabras, y la cola apunta a esta ejecutoria.
_RX_AMPARA_PROTEGE = re.compile(r"\bampara\s+y\s+protege\b", re.I)
_RX_COLA_DEL_JUZGADO = re.compile(
    r",?\s*(?:por\s+(?:los|las)\s+(?:motivos|razones|consideraciones|fundamentos)\b"
    r"|para\s+(?:los|el)\s+efectos?\b|en\s+t[ée]rminos\s+del?\b).*$", re.I)


_COLA_ESTA_EJECUTORIA = ("por los motivos y fundamentos expuestos en el último "
                         "considerando de esta ejecutoria.")


def _cuerpo_con_verbo(resolutivo_reproducido: str, verbo: str,
                      cola: str = _COLA_ESTA_EJECUTORIA) -> str:
    """El punto del juzgado —su sujeto y su acto— con otro verbo y la cola hacia
    esta ejecutoria; «» si no es un «ampara y protege» limpio."""
    t = " ".join((resolutivo_reproducido or "").split())
    if (not t or not _RX_AMPARA_PROTEGE.search(t)
            or re.search(r"\bno\s+ampara\b|\bsobrese", t, re.I)):
        return ""
    cuerpo = _RX_COLA_DEL_JUZGADO.sub("", t).rstrip(" .,;:")
    cuerpo = _RX_AMPARA_PROTEGE.sub(verbo, cuerpo, count=1)
    if len(cuerpo.split()) < 8:
        return ""
    return f"{cuerpo}, {cola}"


def puntos_revoca_concesion(resolutivo_reproducido: str = "") -> list:
    """Los dos puntos de una revisión que revoca una concesión y niega, con el
    sujeto y el acto del resolutivo del juzgado; [] si ese resolutivo no es un
    «ampara y protege» limpio (entonces va la fórmula genérica de la rama)."""
    cuerpo = _cuerpo_con_verbo(resolutivo_reproducido, "no ampara ni protege")
    if not cuerpo:
        return []
    return [RAMAS_REVISION["revoca_fondo_niega"]["puntos"][0], f"SEGUNDO. {cuerpo}"]


# ═══ LA QUEJOSA RECURRE SU CONCESIÓN Y GANA (revisión del 28-sep-2026) ═══════
# AR 631/2025, revisión adversarial: con la quejosa como recurrente —pedía otros
# efectos, otro alcance— y el recurso fundado, la rama era «revoca y niega» y la
# tarjeta, la deliberación y el documento la dejaban SIN el amparo que ya tenía.
# Sola recurrente, su recurso no puede empeorarle la situación: se revoca o se
# modifica —según el alcance del vicio, `recurso_revoca_o_modifica`— y la
# sentencia que corresponde (art. 93, fr. V) la ampara con el alcance o los
# efectos que resulten del estudio. Antes de escribir no se sabe si se revoca o
# se modifica: ese verbo va en {HUECO}.
_COLA_EFECTOS_EJECUTORIA = ("en los términos y para los efectos precisados en el "
                            "último considerando de esta ejecutoria.")


def puntos_quejosa_mejora(resolutivo_reproducido: str = "", verbo_sentencia: str = "") -> list:
    """Los dos puntos cuando la quejosa recurre una concesión y su recurso
    prospera. `verbo_sentencia` —«revoca» | «modifica» | «»— decide el PRIMERO;
    vacío lo deja en {HUECO}. El SEGUNDO la ampara: con el sujeto y el acto del
    resolutivo del juzgado si es un «ampara y protege» limpio; si no, con la
    fórmula genérica. Con {quejoso}, {responsable_originaria} y {HUECO} para que
    el compositor los llene."""
    v = (verbo_sentencia or "").strip().lower()
    primero = {"revoca": "PRIMERO. Se revoca la sentencia recurrida.",
               "modifica": "PRIMERO. Se modifica la sentencia recurrida."}.get(
        v, "PRIMERO. Se {HUECO} la sentencia recurrida.")
    cuerpo = (_cuerpo_con_verbo(resolutivo_reproducido, "ampara y protege",
                                _COLA_EFECTOS_EJECUTORIA)
              or ("La Justicia de la Unión ampara y protege a {quejoso}, contra el acto "
                  "reclamado a {responsable_originaria}, " + _COLA_EFECTOS_EJECUTORIA))
    return [primero, f"SEGUNDO. {cuerpo}"]


# ═══ REVOCAR UNA CONCESIÓN Y REASUMIR JURISDICCIÓN (art. 93, fr. VI) ════════
# AR 631/2025 (28-sep-2026). Tres cosas que el resolutivo del proyecto arreglado
# no hacía:
#   · «Se revoca la sentencia recurrida» a secas arrastraba también el
#     sobreseimiento respecto del Juez Quinto, que nadie impugnó: la revocación
#     se acota a la MATERIA DE LA REVISIÓN y lo no impugnado queda firme
#     (1a./J., registro 174177, en el acervo);
#   · el punto del amparo salía «no ampara» sin que nadie estudiara los
#     conceptos que el juzgado no estudió: sale de ESE estudio —concede por
#     razón distinta o niega—;
#   · y si los conceptos no constaron o el estudio no concluye, el verbo va en
#     HUECO, a la vista, como en los adelantos de David: «no ampara» sin
#     considerando que lo sostenga es peor que un hueco.
# LA FORMA Y EL ORDEN SON LOS DEL CIRCUITO, medidos en los 3,526 amparos en
# revisión de oaj/lectura (28-sep-2026): «Queda firme el sobreseimiento
# decretado en …» va PRIMERO (AR 105/2019, 112/2021, 168/2025: firme, luego «En
# la materia de la revisión, se revoca…», luego el amparo); «En la materia de
# la revisión, se revoca la sentencia recurrida» en más de 40 resolutivos.
REVOCA_PARCIAL = "PRIMERO. En la materia de la revisión, se revoca la sentencia recurrida."
FIRME_SOBRESEIMIENTO = "Queda firme el sobreseimiento decretado en la sentencia recurrida."
_GENERICO_REASUNCION = ("La Justicia de la Unión {verbo} a {quejoso}, contra el "
                        "acto reclamado a {responsable_originaria}, por los motivos "
                        "y fundamentos expuestos en el último considerando de esta "
                        "ejecutoria.")


def sobreseimiento_firme(sobresee_ademas: bool, quien_recurre: str = "",
                         adhesiva: bool = False) -> bool:
    """¿El sobreseimiento de una sentencia mixta quedó firme, por código?

    AR 631/2025 (28-sep-2026, al generar en pantalla): el juzgado sobreseyó
    respecto de la interlocutoria del Juez Quinto —en sus considerandos, no en
    sus puntos— y concedió contra la Sala; recurrió la tercera interesada. El
    documento sólo ponía «Queda firme el sobreseimiento…» si el ESTUDIO lo
    declaraba (`fase_rama.declara_firme_el_sobreseimiento`), el modelo no lo
    escribió y los resolutivos salieron sin él, aunque el registro dijera «la
    recurrida también sobreseyó». Pero eso no depende de la prosa: el
    sobreseimiento sólo perjudica a la quejosa (`ficha_procesal._perjudica`);
    si quien recurre es la autoridad o la tercera interesada, nadie lo impugnó
    —salvo que la quejosa se adhiriera (art. 82), y entonces se comprueba a
    mano—. Sin saber quién recurre, tampoco se afirma."""
    q = (quien_recurre or "").strip().lower()
    return bool(sobresee_ademas) and q in ("tercero", "autoridad") and not adhesiva


def puntos_reasuncion(sentido_amparo: str, resolutivo_reproducido: str = "",
                      firme: bool = False, parcial: bool = False) -> list:
    """Los puntos de una revisión que revoca una concesión y reasume
    jurisdicción. `sentido_amparo` es lo que concluye el estudio de los
    conceptos no estudiados —«concede» | «niega» | «»—; vacío deja el verbo en
    {HUECO}. `firme`: el estudio declara firme un sobreseimiento no impugnado
    (va como punto propio, el PRIMERO, como en el circuito). `parcial`: algo de la recurrida quedó fuera de la
    revisión (PRIMERO «en la materia de la revisión»). Con {quejoso},
    {responsable_originaria} y {HUECO} para que el compositor los llene, como
    los de `RAMAS_REVISION`."""
    sa = (sentido_amparo or "").strip().lower()
    verbo = {"niega": "no ampara ni protege", "concede": "ampara y protege"}.get(sa, "{HUECO}")
    revoca = (REVOCA_PARCIAL if (parcial or firme)
              else RAMAS_REVISION["revoca_fondo_niega"]["puntos"][0]).split(". ", 1)[1]
    cuerpo = (_cuerpo_con_verbo(resolutivo_reproducido, verbo)
              or _GENERICO_REASUNCION.replace("{verbo}", verbo))
    cuerpos = ([FIRME_SOBRESEIMIENTO] if firme else []) + [revoca, cuerpo]
    return [f"{o}. {c}" for o, c in zip(("PRIMERO", "SEGUNDO", "TERCERO"), cuerpos)]


# ═══════════════════════════════════════════════════════════════════════════
# CON QUÉ VERBOS DECIDE EL ÓRGANO DE ORIGEN
#
# David, leyendo el contexto de un amparo directo: «erróneamente dice "la
# responsable negó el amparo". Eso es un error. La responsable declara
# procedente o improcedente la acción o la excepción. La responsable, en amparo
# directo, NO resuelve amparos, por lo que no puede negar o conceder.»
#
# Tenía razón, y el hueco era de catálogo. Este módulo tiene vocabulario por
# tipo para casi todo —CIERRE, SUJETOS, VOCABULARIO, RAMAS_REVISION— pero NO
# para los verbos con que decide el órgano RECURRIDO o la RESPONSABLE. Ese
# hueco lo rellenaban dos prompts con lo primero que tenían a mano, que era el
# vocabulario del amparo en revisión: «sobreseyó, negó el amparo, concedió el
# amparo». En un amparo directo eso es un disparate, y en uno de los dos sitios
# —los antecedentes— se firma.
#
# Quién decide qué, sin confundirlo:
#   · amparo DIRECTO      la responsable es una Sala o tribunal ordinario. No
#                         resuelve amparos: resuelve el juicio de origen.
#   · amparo en REVISIÓN  el a quo es un juez de distrito, y ése SÍ concedió,
#                         negó o sobreseyó en un amparo.
#   · revisión FISCAL     la Sala del Tribunal Federal de Justicia
#                         Administrativa declara la nulidad o reconoce la
#                         validez.
#   · QUEJA               lo recurrido es un auto o una resolución de trámite.
#
# ADVERTENCIA HONESTA: esto es doctrina, no medición. Se comprobó que el banco
# de fórmulas medidas NO contiene «declaró procedente», «absolvió» ni «condenó»,
# así que estos verbos no salen del acervo de engroses reales como los demás de
# este módulo. Cuando haya corpus suficiente, hay que medirlos.
VERBOS_DEL_RECURRIDO = {
    "amparo_directo": (
        "confirmó, modificó o revocó la sentencia de primera instancia; "
        "declaró procedente o improcedente la acción o la excepción; condenó "
        "o absolvió; declaró la nulidad; decretó la caducidad; dejó a salvo "
        "los derechos"),
    "amparo_revision": (
        "sobreseyó, negó el amparo, concedió el amparo o desechó la demanda"),
    "revision_fiscal": (
        "declaró la nulidad de la resolución impugnada, reconoció su validez "
        "o sobreseyó en el juicio"),
    "queja": (
        "admitió, desechó o tuvo por no interpuesta la demanda; concedió o "
        "negó la suspensión; o proveyó sobre el trámite"),
}


def verbos_del_recurrido(tipo: str) -> str:
    """Los verbos con que decidió el órgano de origen de ESTE tipo de asunto.

    Nunca los del amparo cuando el asunto no es un amparo en revisión: la
    autoridad responsable de un amparo directo no concede ni niega amparos.

    Y en un amparo directo SIN ALZADA (30-sep-2026), nunca los de la apelación:
    ver `VERBOS_DEL_RECURRIDO_UNICA`. Los dos consumidores —los antecedentes y
    el contexto de la propuesta— ya pasan por aquí, así que la variante les
    llega sin tocarlos.
    """
    t = normalizar(tipo) or "amparo_directo"
    if t == "amparo_directo" and instancia_actual() == "unica":
        return VERBOS_DEL_RECURRIDO_UNICA
    return VERBOS_DEL_RECURRIDO.get(t, VERBOS_DEL_RECURRIDO["amparo_directo"])


# ═══════════════════════════════════════════════════════════════════════════
# AMPARO DIRECTO SIN SEGUNDA INSTANCIA (30-sep-2026)
#
# David (AD 323/2025, juicio oral mercantil): «Siempre, en amparo directo,
# partimos de la base de que hay una sala. Es decir, una segunda instancia.
# Pero no siempre es así. La segunda instancia ocurre sólo si hubo recurso de
# apelación». El menú de arriba empezaba por «confirmó, modificó o revocó la
# sentencia de primera instancia», que es lo que hace una Sala de apelación:
# dado a los antecedentes de un juicio oral mercantil —art. 1390 Bis del Código
# de Comercio: contra sus resoluciones no procede recurso ordinario alguno—, el
# motor narraba una alzada que no hubo. Sin alzada, quien dictó lo reclamado
# resolvió el juicio mismo: la acción, la condena, la nulidad o el laudo.
#
# Estos dos accesores son los que consultan los prompts de amparo directo. Los
# dos devuelven «no» si el tipo no es amparo directo, si no rige la bandera o
# si la instancia no consta: entonces cada prompt queda exactamente como antes
# (las cuentas de fuera no ven el cambio hasta medirlo).
VERBOS_DEL_RECURRIDO_UNICA = (
    "declaró procedente o improcedente la acción o la excepción; condenó o "
    "absolvió; declaró la nulidad o reconoció la validez de la resolución "
    "impugnada; decretó la caducidad; dejó a salvo los derechos")


def unica_instancia(tipo: str = "") -> bool:
    """¿Es un amparo directo contra lo dictado en ÚNICA instancia (sin
    apelación) y rige `instancia_origen` en esta petición? Sin tipo se
    entiende amparo directo, como en `sujetos_de`."""
    t = normalizar(tipo) or "amparo_directo"
    return t == "amparo_directo" and instancia_actual() == "unica"


def cumplimiento_de_amparo(tipo: str = "") -> dict:
    """{consta, ejecutoria, efectos} si ESTE amparo directo reclama una
    sentencia dictada en cumplimiento de una ejecutoria (ver
    `cumplimiento_actual`); {} en cualquier otro caso."""
    t = normalizar(tipo) or "amparo_directo"
    return cumplimiento_actual() if t == "amparo_directo" else {}
