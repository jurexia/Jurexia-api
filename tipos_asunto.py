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
    "amparo_revision": [
        ("RECURRENTE", "quejoso", True),
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
