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
        # EL ORDEN Y LA EXISTENCIA, del adelanto que David ajustó a mano.
        #
        # Faltaba «Existencia del acto reclamado», que en la revisión NO es el
        # acto del amparo indirecto sino la SENTENCIA RECURRIDA: se acredita
        # con el informe justificado del Juzgado de Distrito y con los autos
        # que lo acompañan. Sin ese considerando el proyecto revisa una
        # sentencia sin haber dicho antes que consta.
        #
        # Y el orden cambia: la existencia va DESPUÉS de la competencia y ANTES
        # de la legitimación, porque primero se establece que el acto existe y
        # sólo entonces tiene sentido preguntar si quien recurre está
        # legitimado y si lo hizo a tiempo.
        "considerandos": [("Competencia", 50),
                          ("Existencia del acto reclamado", 0),
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
         "fecha, oficialía, promovente y su carácter. Después INDIVIDUALIZA la "
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
        # EL RESULTANDO PRIMERO LLEVA DOS SUB-BLOQUES ROTULADOS, no prosa
        # corrida. Así lo escribió David al ajustar el adelanto, y tiene
        # sentido: la autoridad responsable y el acto reclamado son los dos
        # datos que se buscan de un vistazo cuando se toma el asunto.
        ("Presentación de la demanda de amparo indirecto",
         "fecha, oficialía y promovente de la DEMANDA DE AMPARO INDIRECTO —no "
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
         "proyecto. La fecha es obligatoria: «Consta en autos que el asunto "
         "fue turnado» no dice cuándo, y ese dato lo tiene el auto de turno"),
    ],
    "revision_fiscal": [
        ("Trámite del juicio contencioso administrativo",
         "fecha y oficialía de la demanda de nulidad, quién la promovió, qué "
         "resolución impugnó, y LA SENTENCIA DE LA SALA CON SU FECHA Y SU "
         "SENTIDO. El sentido es OBLIGATORIO y con el verbo: «DECLARÓ LA "
         "NULIDAD», «RECONOCIÓ LA VALIDEZ», «SOBRESEYÓ». Ya lo pedía este "
         "apartado y aun así el proyecto salió diciendo quién promovió y contra "
         "qué, sin llegar nunca a decir qué resolvió la Sala —que es el acto "
         "que se está revisando—. Sin ese dato el considerando de procedencia "
         "no puede fundarse y quien lee no sabe qué se revisa"),
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


def cierre_de(tipo: str) -> dict:
    return CIERRE.get(normalizar(tipo), CIERRE["amparo_directo"])


def parrafo_cierre(tipo: str, concede: bool, calificacion: str = "") -> str:
    """El cierre del estudio, con la forma del corpus y el desenlace del tipo."""
    v = vocabulario_de(tipo)
    c = cierre_de(tipo)
    desenlace = c["positivo"] if concede else c["negativo"]
    if concede:
        return (f"En ese sentido, al resultar {calificacion or 'fundado'} "
                f"lo planteado, lo procedente es {desenlace}.")
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
    fuera = []
    for rx in _EVASIVAS:
        m = rx.search(bloque)
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
    "amparo_directo": [
        ("QUEJOSO", "quejoso", True),
        ("TERCERO INTERESADO", "tercero", False),
        ("AUTORIDAD RESPONSABLE", "responsable", True),
    ],
    # EL ADELANTO QUE DAVID AJUSTÓ ESCRIBE UNA SOLA FIGURA: «QUEJOSA Y
    # RECURRENTE». No dos renglones, y sin el del órgano recurrido.
    #
    # Y tiene su lógica: en un amparo EN REVISIÓN quien recurre es —casi
    # siempre— quien fue quejoso en el amparo indirecto, así que separarlo en
    # dos etiquetas repite a la misma persona; y el Juzgado de Distrito no es
    # parte del recurso, es el órgano cuya sentencia se revisa, y ya se nombra
    # en el V I S T O y en la competencia. Ponerlo en el rubro lo asciende a
    # parte, que es el error que este catálogo existía para evitar… y se corregía
    # en la dirección contraria.
    #
    # LA ETIQUETA CONCUERDA EN GÉNERO con quien promueve, porque «QUEJOSO Y
    # RECURRENTE» sobre una sociedad rural no lo firma nadie.
    "amparo_revision": [
        ("{QUEJOSO_A} Y RECURRENTE", "quejoso", True),
        ("RECURRENTE ADHESIVO", "adherente", False),
        # EL ÓRGANO RECURRIDO FALTABA, y era el único tipo sin él: el amparo
        # directo lleva «AUTORIDAD RESPONSABLE», la queja «ÓRGANO QUE DICTÓ EL
        # AUTO RECURRIDO» y la revisión fiscal «SALA RESPONSABLE». La revisión
        # se quedó con quejoso y adherente, así que el proyecto salía con una
        # carátula que no dice contra quién va el asunto.
        #
        # Los engroses reales de David SÍ lo llevan: «ÓRGANO RECURRIDO: SALA
        # REGIONAL EN QUERÉTARO DEL TRIBUNAL FEDERAL DE JUSTICIA ADMINISTRATIVA
        # Y DE LA DIRECCIÓN LOCAL QUERÉTARO DE LA COMISIÓN NACIONAL DEL AGUA».
        #
        # Se rotula «ÓRGANO RECURRIDO» y no «AUTORIDAD RESPONSABLE»: en un
        # recurso lo que se recurre es la resolución de un órgano, y la
        # responsable es otra figura del amparo de origen.
        ("ÓRGANO RECURRIDO", "responsable", True),
    ],
    "queja": [
        ("RECURRENTE", "quejoso", True),
        ("ÓRGANO QUE DICTÓ EL AUTO RECURRIDO", "responsable", True),
    ],
    "revision_fiscal": [
        ("AUTORIDAD RECURRENTE", "quejoso", True),
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
_RX_FEMENINO = re.compile(
    r"\bsociedad\b|\bS\.?\s*A\.?\b|\bS\.?\s*de\s*R\.?\s*L\.?\b|"
    r"\basociaci[óo]n\b|\bcooperativa\b|\bempresa\b|\binstituci[óo]n\b|"
    r"\bcomisi[óo]n\b|\bsecretar[íi]a\b|\bdirecci[óo]n\b|\bsala\b|"
    r"\bjunta\b|\bunidad\b|\buniversidad\b|\bfundaci[óo]n\b", re.I)
_RX_MASCULINO = re.compile(
    r"\binstituto\b|\bayuntamiento\b|\bmunicipio\b|\borganismo\b|"
    r"\bconsejo\b|\bbanco\b|\btribunal\b|\bjuzgado\b", re.I)


def genero_de(nombre: str) -> str:
    """«a», «o» o «" "» —neutro— para concordar la etiqueta de la carátula."""
    n = (nombre or "").strip()
    if not n:
        return ""
    if _RX_FEMENINO.search(n):
        return "a"
    if _RX_MASCULINO.search(n):
        return "o"
    return ""


def etiqueta_concordada(etiqueta: str, nombre: str) -> str:
    """Sustituye {QUEJOSO_A} por la forma que concuerda con quien promueve."""
    if "{QUEJOSO_A}" not in (etiqueta or ""):
        return etiqueta
    g = genero_de(nombre)
    forma = {"a": "QUEJOSA", "o": "QUEJOSO"}.get(g, "PARTE QUEJOSA")
    return etiqueta.replace("{QUEJOSO_A}", forma)


def caratula_de(tipo: str) -> list:
    return CARATULA.get(normalizar(tipo), CARATULA["amparo_directo"])


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
    (r"incidente\s+de\s+nulidad\s+de\s+(?:notificaci[óo]n(?:es)?|actuaciones)"
     r"|incidente\s+de\s+reposici[óo]n\s+de\s+autos"
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
    return SUJETOS.get(normalizar(tipo), SUJETOS["amparo_directo"])


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
    "d": "en el que se resolvió sobre el carácter de tercero interesado",
    "e": ("por tratarse de una resolución dictada con posterioridad a la "
          "sentencia definitiva en el juicio de amparo indirecto, que no "
          "admite recurso de revisión"),
}


def cola_97(inciso: str) -> str:
    """Qué se recurrió, en la redacción que corresponde al inciso."""
    return COLA_97.get((inciso or "").strip().lower(), "")


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
#   RAMA 1 — LEVANTAMIENTO DE SOBRESEIMIENTO (art. 93, fr. I). El a quo
#   sobreseyó indebidamente; se destruye la causal, se levanta el
#   sobreseimiento y, en plenitud de jurisdicción, se estudian por primera vez
#   los conceptos de violación que aquél omitió. Resolutivo DOBLE.
#
#   RAMA 2 — FONDO (art. 93, frs. V y VI). Tres supuestos: revocar la negativa
#   —se concede—, revocar la concesión —se niega— y modificar los EFECTOS, que
#   es el único de los tres con resolutivo ÚNICO.
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
            "SEGUNDO. La Justicia de la Unión no ampara ni protege a {quejoso}, "
            "respecto de los actos precisados en la resolución recurrida, por "
            "las razones expuestas en el último considerando de la misma."],
        "frecuencia": "8/62 confirma · 8/62 no ampara ni protege",
    },
    "confirma_concede": {
        "fundamento": "artículo 93, fracciones V y VI, de la Ley de Amparo",
        "puntos": [
            "PRIMERO. Se confirma la sentencia recurrida.",
            "SEGUNDO. La Justicia de la Unión ampara y protege a {quejoso}, en "
            "términos del último considerando de la resolución recurrida."],
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
            "SEGUNDO. Se sobresee en el presente juicio de amparo, promovido "
            "por {quejoso}, contra los actos que quedaron precisados en el "
            "considerando segundo de la resolución que se revisa y por las "
            "razones expuestas en el último considerando de la misma."],
        "frecuencia": "10/62 sobresee",
    },
    # ── RAMA 1: se levanta el sobreseimiento y se asume jurisdicción ──────
    "revoca_sobreseimiento_concede": {
        "fundamento": "artículo 93, fracción I, de la Ley de Amparo",
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
        "fundamento": "artículo 93, fracción I, de la Ley de Amparo",
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
}

# Lo que el secretario puede elegir hoy en pantalla. Se separa del diccionario
# porque añadir una calificación al catálogo y ofrecerla en la interfaz son dos
# decisiones distintas: la primera es reconocerla, la segunda es pedirla.
SENTIDOS_OFRECIDOS = ("fundado", "esencialmente_fundado", "infundado",
                      "inoperante", "ineficaz")


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
    # Las que el catálogo aún no conoce pero llevan «fundad» dentro —
    # «parcialmente_fundado», «sustancialmente_fundado», «fundado_suplido»—
    # prosperan: es más seguro reconocerlas que tratarlas como desestimadas.
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
        "fuente": "artículo 93, fracción I, de la Ley de Amparo",
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

    # ── AMPARO DIRECTO: el orden del estudio ──────────────────────────────
    "directo_orden_de_estudio": {
        "cuando": "Amparo directo en que se plantean violaciones procesales "
                  "junto con cuestiones de fondo.",
        "fuente": "artículos 174 y 189 de la Ley de Amparo",
        "tecnica": [
            "LAS VIOLACIONES PROCESALES SON DE ESTUDIO PREFERENTE, porque si "
            "prosperan obligan a reponer el procedimiento y lo demás quedaría "
            "sin materia.",
            "PERO MANDA EL MAYOR BENEFICIO. El artículo 189 obliga a estudiar "
            "primero aquello que, de resultar fundado, otorgue a la parte "
            "quejosa un beneficio MAYOR. Si el fondo puede darle más que la "
            "reposición —una concesión lisa y llana frente a repetir el "
            "procedimiento—, el fondo va primero y se dice por qué.",
            "EL ORDEN ELEGIDO SE JUSTIFICA en el proyecto, en una frase. No se "
            "presenta como método: se presenta como la aplicación del artículo "
            "189 a este caso concreto.",
        ],
    },
}


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
        if rama.startswith("revoca_sobreseimiento"):
            fuera.append(TECNICA_RESOLUCION["revision_levanta_sobreseimiento"])
    if t == "amparo_directo" and con_violacion_procesal:
        fuera.append(TECNICA_RESOLUCION["directo_orden_de_estudio"])
    return fuera


def rama_revision(resolvio_a_quo: str, sentido: str,
                  solo_efectos: bool = False,
                  violacion_procesal: bool = False,
                  sentido_amparo: str = "") -> str:
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
    """
    a = (resolvio_a_quo or "").strip().lower()
    s = (sentido or "").strip().lower()
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
    if violacion_procesal and _prospera:
        return "repone_procedimiento"
    if solo_efectos and _prospera and a == "concede":
        return "modifica_efectos"
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
    return ("revoca_fondo_niega" if a == "concede"
            else "revoca_fondo_concede")


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
    },
    "queja": {
        "rotulo": "Extemporaneidad del recurso de queja",
        "fundamento": "artículo 98 de la Ley de Amparo",
        "considerando": (
            "El recurso de queja se interpuso de manera extemporánea, conforme "
            "al cómputo que antecede, por lo que resulta improcedente."),
        "resolutivo": "ÚNICO. Se desecha por improcedente el recurso de queja.",
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
# 642/2024», «RECURSO DE QUEJA CIVIL: 233/2025».
_ENCABEZADO = {
    "amparo_directo": "AMPARO DIRECTO {materia}: {numero}",
    "amparo_revision": "AMPARO EN REVISIÓN {materia}: {numero}",
    "queja": "RECURSO DE QUEJA {materia}: {numero}",
    "revision_fiscal": "REVISIÓN FISCAL: {numero}",
}

# La materia va concordada con el sustantivo que la precede: «AMPARO DIRECTO
# CIVIL» pero «AMPARO EN REVISIÓN ADMINISTRATIVA». Escribirla en masculino
# siempre da «AMPARO EN REVISIÓN ADMINISTRATIVO», que ningún secretario firma.
_MATERIA_ENCABEZADO = {
    "amparo_revision": {"administrativa": "ADMINISTRATIVA", "civil": "CIVIL",
                        "laboral": "LABORAL", "penal": "PENAL"},
    "queja": {"administrativa": "ADMINISTRATIVO", "civil": "CIVIL",
              "laboral": "LABORAL", "penal": "PENAL"},
    "amparo_directo": {"administrativa": "ADMINISTRATIVO", "civil": "CIVIL",
                       "laboral": "LABORAL", "penal": "PENAL"},
}


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
                  "está legitimad{a} para ello conforme al artículo 5º, "
                  "fracción I, de la Ley de Amparo, por resentir el perjuicio "
                  "que le causa la sentencia reclamada."),
        "fundamento": "artículo 5º, fracción I, de la Ley de Amparo"},
    "amparo_revision": {
        "molde": ("El presente medio de impugnación fue interpuesto por "
                  "{parte}{rep}, quien se encuentra legitimad{a} para "
                  "interponer el recurso de revisión conforme al artículo 6º "
                  "de la Ley de Amparo, toda vez que la resolución impugnada "
                  "le resulta desfavorable."),
        "fundamento": "artículo 6º de la Ley de Amparo"},
    "queja": {
        "molde": ("El recurso fue interpuesto por {parte}{rep}, quien está "
                  "legitimad{a} para hacerlo conforme al artículo 97 de la Ley "
                  "de Amparo, por ser parte en el juicio de amparo en que se "
                  "dictó el auto recurrido y resultarle desfavorable."),
        "fundamento": "artículo 97 de la Ley de Amparo"},
    "revision_fiscal": {
        # AQUÍ NO ES «QUIEN RESIENTE EL PERJUICIO»: es la autoridad, y sólo
        # por conducto de su unidad de defensa jurídica. Lo dice el propio 63.
        "molde": ("El recurso fue interpuesto por {parte}{rep}, autoridad "
                  "facultada para hacerlo por conducto de la unidad "
                  "administrativa encargada de su defensa jurídica, en "
                  "términos del artículo 63 de la Ley Federal de Procedimiento "
                  "Contencioso Administrativo."),
        "fundamento": ("artículo 63 de la Ley Federal de Procedimiento "
                       "Contencioso Administrativo")},
}


def legitimacion_de(tipo: str, parte: str = "", representante: str = "",
                    hueco: str = "*********") -> str:
    """El párrafo de legitimación de esta vía, o cadena vacía si no hay parte.

    SIN NOMBRE NO SE ESCRIBE. Un párrafo que dice «la parte está legitimada»
    sin decir quién no acredita nada, y es justo la perífrasis que el catálogo
    persigue en los resultandos.
    """
    m = LEGITIMACION.get(normalizar(tipo))
    if not m or not (parte or "").strip():
        return ""
    rep = ""
    if (representante or "").strip():
        rep = f", a través de su representante legal {representante.strip()}"
    elif normalizar(tipo) == "revision_fiscal":
        rep = ""
    return m["molde"].format(parte=parte.strip(), rep=rep,
                             a="a" if genero_de(parte) == "a" else "o")


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
    mat = _MATERIA_ENCABEZADO.get(t, {}).get(
        (materia or "").strip().lower(), (materia or "").strip().upper())
    return " ".join(plantilla.format(materia=mat, numero=(numero or "").strip()).split())


def ley_de_la_via(tipo: str) -> str:
    """El marco normativo de la vía, para decírselo al modelo antes de escribir."""
    return LEY_DE_LA_VIA.get(normalizar(tipo), "")

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
