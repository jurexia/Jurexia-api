"""QUÉ RESOLVIÓ EL JUZGADO DE DISTRITO, Y CONTRA QUIÉN SE AMPARA.

Sin estos dos datos el amparo en revisión no puede tener resolutivo: el
PRIMERO decide sobre la sentencia recurrida —confirmar, revocar, modificar— y
eso depende de qué hizo el a quo; y el SEGUNDO nombra a la AUTORIDAD
RESPONSABLE ORIGINARIA, que es la del acto reclamado del amparo indirecto y no
el Juzgado, que es el órgano recurrido.

David lo dice sin rodeos en su encomienda: «SEGUNDO. La Justicia de la Unión
ampara y protege a [Quejoso] contra el acto reclamado a [Autoridad Responsable
ORIGINARIA]».

SE LEE, NO SE PREGUNTA. Los dos datos están en el resumen del acto que las
fases 1-3 ya escribieron con la sentencia recurrida delante, y en los
resultandos. Pedírselos otra vez al modelo sería una llamada más y una ocasión
más de inventarlos.

Y CUANDO NO CONSTA, SE DEJA EL HUECO. El propio corpus lo hace: los adelantos
de este tribunal escriben «Se ********** la sentencia impugnada» cuando el
sentido aún no está decidido. Un resolutivo que confirma lo que en realidad se
sobreseyó es un error que no se ve leyendo por encima.
"""

from __future__ import annotations

import re

# Qué hizo el Juzgado de Distrito. El orden importa: «sobreseyó» gana sobre
# «negó» porque una sentencia que sobresee suele decir además que «se niega el
# amparo respecto de los demás actos», y quedarse con lo segundo cambiaría la
# rama entera.
_QUE_HIZO = [
    ("sobresee", r"\bsobresey[óo]\b|\bsobrese[ei]miento\b|"
                 r"decret[óo]\s+el\s+sobreseimiento|"
                 r"tuvo\s+por\s+no\s+presentada|\bdesech[óo]\s+de\s+plano\b"),
    ("concede",  r"\bconcedi[óo]\s+(?:el\s+)?(?:amparo|la\s+protecci[óo]n)\b|"
                 r"\bampar[óo]\s+y\s+protegi[óo]\b|otorg[óo]\s+el\s+amparo"),
    ("niega",    r"\bneg[óo]\s+(?:el\s+)?(?:amparo|la\s+protecci[óo]n)\b|"
                 r"\bno\s+ampar[óo]\s+ni\s+protegi[óo]\b"),
]


# LOS MISMOS SUPUESTOS, PERO SÓLO EN VERBO. Sin el sustantivo
# «sobreseimiento», que es el que aparece en los rubros de las tesis que el
# estudio transcribe y el que hacía confirmar sobreseimientos inexistentes.
_QUE_HIZO_VERBOS = [
    ("sobresee", r"\bsobresey[óo]\b|decret[óo]\s+el\s+sobreseimiento|"
                 r"tuvo\s+por\s+no\s+presentada|\bdesech[óo]\s+de\s+plano\b"),
    ("concede",  r"\bconcedi[óo]\s+(?:el\s+)?(?:amparo|la\s+protecci[óo]n)\b|"
                 r"\bampar[óo]\s+y\s+protegi[óo]\b|otorg[óo]\s+el\s+amparo"),
    ("niega",    r"\bneg[óo]\s+(?:el\s+)?(?:amparo|la\s+protecci[óo]n)\b|"
                 r"\bno\s+ampar[óo]\s+ni\s+protegi[óo]\b"),
]


def _algo_dice(t: str) -> bool:
    """¿Este texto dice, de alguna forma, en qué paró el juicio?"""
    return any(re.search(rx, t, re.I) for _, rx in _QUE_HIZO)


def _resolvio_de(fuente: str, solo_verbos: bool = False) -> str:
    """El recuento, aislado para poder correrlo sobre más de una fuente."""
    cuenta = {}
    for clave, rx in (_QUE_HIZO_VERBOS if solo_verbos else _QUE_HIZO):
        n_ = 0
        for m in re.finditer(rx, fuente, re.I):
            if _afirmado(fuente, m):
                n_ += 1
        if n_:
            cuenta[clave] = n_
    if not cuenta:
        return ""
    # A igualdad, el orden de `_QUE_HIZO`: sobreseer es lo más grave y lo que
    # se decide primero.
    orden = {c: i for i, (c, _) in enumerate(_QUE_HIZO)}
    return max(cuenta, key=lambda c: (cuenta[c], -orden[c]))


def resolvio_a_quo(texto: str, antecedentes: str = "",
                   declarado: str = "") -> str:
    """«sobresee» | «concede» | «niega», o cadena vacía si no consta.

    DOS CAMBIOS, Y LOS DOS SALIERON DEL MISMO CASO REAL.

    (a) SI HAY ANTECEDENTES, MANDAN ELLOS. La versión anterior recibía el
        estudio entero —con las tesis TRANSCRITAS dentro— y devolvía
        «sobresee» en cuanto la palabra «sobreseimiento» aparecía en cualquier
        sitio. En el ARA 17/2025 aparecía dentro de una tesis citada, y el
        proyecto confirmaba un sobreseimiento que nadie había decretado.
        Medido: con el estudio entero da «sobresee»; sin las tesis, «concede».

    (b) GANA EL QUE MÁS VECES SE DICE, no el primero de la lista. Devolver al
        primer patrón que casa hacía que el orden de `_QUE_HIZO` decidiera el
        resolutivo, que es tanto como echarlo a suertes.
    """
    # (c) LO DECLARADO MANDA SOBRE EL BARRIDO. `declarado` es la frase con la
    #     que el motor resumió qué resolvió el órgano al preparar la propuesta
    #     —«Sobreseyó con fundamento en el artículo 63, fracción IV, de la Ley
    #     de Amparo»—. No es una fuente más que sumar al recuento: es la
    #     respuesta a esta misma pregunta, dada por quien leyó el expediente
    #     entero.
    #
    #     Salió de la revisión 410/2026: los antecedentes narraban el juicio de
    #     nulidad paso a paso y nunca decían en qué paró el amparo, así que el
    #     resolutivo salió «Se ********* la sentencia recurrida» mientras el
    #     estudio, tres párrafos antes, decía «se confirma la sentencia
    #     recurrida» y «debe mantener el sobreseimiento decretado». El dato
    #     estaba a un cable de distancia.
    #
    #     Se pasa por el mismo barrido —no se cree a ciegas—: si la frase no
    #     dice claramente qué se hizo, se sigue como antes.
    dec = " ".join((declarado or "").split())
    if dec:
        _d = _resolvio_de(dec, solo_verbos=False)
        if _d:
            return _d

    ant = " ".join((antecedentes or "").split())
    todo = " ".join((texto or "").split())
    fuente = ant or todo
    if not fuente:
        return ""
    _solo_verbos = False
    if ant and not _algo_dice(ant):
        # LOS ANTECEDENTES NO LO DICEN. Pasa cuando el resultando se queda en
        # «conoció de la demanda y la radicó bajo el expediente 795/2023» sin
        # llegar a decir en qué paró. Se mira el resto del documento, pero
        # SÓLO CON LOS VERBOS —«concedió», «negó», «sobreseyó»— y no con el
        # sustantivo «sobreseimiento», que es justo lo que aparece dentro de
        # las tesis transcritas y lo que contaminaba la lectura.
        fuente, _solo_verbos = todo, True
    return _resolvio_de(fuente, solo_verbos=_solo_verbos)


# LA VIOLACIÓN PROCESAL DEL AMPARO —la que obliga a reponer— no es cualquier
# irregularidad: es la cometida DENTRO del juicio de amparo que dejó sin
# defensa a una parte. El artículo 93, fracción IV, lo acota, y David da los
# ejemplos: la audiencia constitucional celebrada sin emplazar al tercero
# interesado, o una prueba desechada ilegalmente EN EL AMPARO.
#
# Se exige que la frase nombre el juicio de amparo o al juez de distrito: una
# violación procesal del juicio de ORIGEN —la laboral, la civil— no lleva a
# reponer el amparo, lleva a concederlo. Confundirlas devolvería el expediente
# al Juzgado por algo que no le toca.
_RX_PROCESAL = re.compile(
    r"(?:reposici[óo]n\s+del\s+procedimiento|viola(?:ci[óo]n|torio)\s+"
    r"(?:al\s+)?procedimiento|sin\s+emplazar|falta\s+de\s+emplazamiento|"
    r"indebida\s+notificaci[óo]n)"
    r"[^.]{0,160}(?:juicio\s+de\s+amparo|amparo\s+indirecto|"
    r"audiencia\s+constitucional|ju(?:ez|zgado)\s+de\s+distrito)|"
    r"(?:juicio\s+de\s+amparo|amparo\s+indirecto|audiencia\s+constitucional)"
    r"[^.]{0,160}(?:reposici[óo]n\s+del\s+procedimiento|sin\s+emplazar|"
    r"falta\s+de\s+emplazamiento)", re.I)


def hay_violacion_procesal(texto: str) -> bool:
    t = " ".join((texto or "").split())
    return any(_afirmado(t, m) for m in _RX_PROCESAL.finditer(t))


# SÓLO LOS EFECTOS. El amparo estuvo bien concedido y lo que falla es la
# restitución: lineamientos deficientes, incongruentes o excesivos. Es el único
# supuesto de resolutivo ÚNICO que no revoca nada.
_RX_EFECTOS = re.compile(
    # SE PUEDE ENSANCHAR PORQUE AHORA HAY GUARDA. Mientras el patrón cazaba
    # también las frases negadas, cada palabra nueva era un riesgo; con
    # `_afirmado` delante, ampliarlo sólo gana cobertura. Faltaban las dos
    # formas más naturales de decirlo: «incompleta» y «procede modificarlos».
    r"(?:efectos?\s+de\s+la\s+concesi[óo]n|lineamientos?)[^.]{0,140}"
    r"(?:deficiente|incongruente|excesiv|insuficiente|imprecis|incomplet|"
    r"deben?\s+(?:precisarse|modificarse|ajustarse)|"
    r"procede\s+(?:modificar|precisar|ajustar)|"
    r"omiti[óo]\s+precisar)|"
    r"(?:modificar|precisar|ajustar)[^.]{0,60}(?:los\s+)?efectos?"
    r"[^.]{0,60}(?:de\s+la\s+)?concesi[óo]n", re.I)


# ═══════════════════════════════════════════════════════════════════════════
# AFIRMAR NO ES DESCARTAR, Y AQUÍ SE CONFUNDÍAN
# ═══════════════════════════════════════════════════════════════════════════
# El engrose del ARA 17/2025 que David firmó ABRE su estudio así:
#
#   «los agravios resultan INOPERANTES PARA MODIFICAR LOS EFECTOS de la
#    concesión»
#
# y `solo_los_efectos` casaba «modificar los efectos de la concesión» sin mirar
# lo que iba delante. Resultado: sobre su propio asunto, el proyecto elegía la
# rama `modifica_efectos` y salía «ÚNICO. Se modifica la sentencia recurrida,
# únicamente para los efectos precisados», que es lo contrario de lo que él
# resolvió. Lo mismo con la reposición: un agravio DESESTIMADO en el que se
# pedía reponer activaba la rama de reposición.
#
# Es la misma lección que ya costó ocho retiradas hoy —«conceder el amparo»
# citado para descartarlo, el artículo 74 apartado diciéndolo—, sólo que aquí
# no producía un aviso falso sino un RESOLUTIVO falso.
_RX_NIEGA_ANTES = re.compile(
    r"(?:inoperantes?|infundados?|ineficaces?|inatendibles?|"
    r"insuficientes?|no\s+prosper\w+|se\s+desestim\w+|desestimad\w+|"
    r"no|sin\s+que|tampoco)\s*(?:\w+\s+){0,4}$", re.I)


def _afirmado(texto: str, m) -> bool:
    """¿La frase que casó AFIRMA el supuesto, o lo descarta?"""
    antes = (texto or "")[max(0, m.start() - 90):m.start()]
    return not _RX_NIEGA_ANTES.search(antes)


def solo_los_efectos(texto: str) -> bool:
    t = " ".join((texto or "").split())
    return any(_afirmado(t, m) for m in _RX_EFECTOS.finditer(t))


# LA AUTORIDAD RESPONSABLE ORIGINARIA. La del acto reclamado del amparo
# indirecto, que NO es el Juzgado de Distrito. Se lee de donde el resumen la
# nombra como emisora del acto reclamado.
_RX_ORIGINARIA = re.compile(
    r"actos?\s+(?:reclamados?\s+)?(?:atribuidos?\s+)?a(?:l)?\s+"
    r"((?:la\s+|el\s+)?[A-ZÁÉÍÓÚÑ][\w\sáéíóúñ,\.]{6,90}?)"
    r"(?=[,;\.]|\s+consistente|\s+por\s+|\s+en\s+el\s+que)|"
    r"reclam[óo]\s+(?:de|a)\s+"
    r"((?:la\s+|el\s+)?[A-ZÁÉÍÓÚÑ][\w\sáéíóúñ,\.]{6,90}?)(?=[,;\.]|\s+el\s+)",
    re.I)

# El órgano de control NUNCA es la responsable originaria: es el recurrido.
# Y EL FORMATO ROTULADO, que es como lo escribe el corpus de verdad:
#
#   «en contra de la autoridad y actos que a continuación se señalan:
#    Autoridad responsable:
#    Tribunal Unitario Agrario del Distrito 42»
#
# `_RX_ORIGINARIA` sólo leía la prosa —«actos atribuidos a X»— y con el
# resultando real del 17/2025 devolvía cadena vacía. El segundo punto del
# resolutivo salía entonces con el respaldo, que en un recurso es el ÓRGANO
# RECURRIDO: se habría amparado contra el acto del Juzgado de Distrito.
# LOS DOS PUNTOS SON OBLIGATORIOS, y costó un resolutivo descubrirlo. Con
# `:?` el patrón disparaba sobre la PROSA: en el amparo en revisión 322/2025 la
# recurrida dice «admitió a trámite la demanda de amparo, solicitó a las
# autoridades responsables su informe justificado», y `responsable_originaria`
# devolvía «su informe justificado». Eso viajaba al SEGUNDO punto resolutivo,
# donde va el nombre de la autoridad contra cuyo acto se ampara.
#
# Y EL ANCLA DE MAYÚSCULA NO ANCLABA NADA: `[A-ZÁÉÍÓÚÑ]` bajo `re.I` acepta
# minúsculas, que es justo lo que esa clase venía a impedir. Quitar la bandera
# rompería «ACTOS RECLAMADOS» en versales, así que la mayúscula se comprueba
# aparte, sobre el texto capturado.
_RX_ORIGINARIA_ROTULO = re.compile(
    r"autoridad(?:es)?\s+responsable(?:s)?\s*:\s*\n?\s*"
    r"((?:la\s+|el\s+)?[A-ZÁÉÍÓÚÑ][\w\sáéíóúñ,\.]{6,90}?)"
    r"(?=\n|acto\s+reclamado|[;\.]|$)", re.I)

# «CONTRA ACTOS DEL Juez Primero de Primera Instancia Civil de San Juan del
# Río». Es como lo escribe el corpus de verdad —lo dice cinco veces en la
# recurrida del 322/2025— y `_RX_ORIGINARIA` no lo veía: esperaba «a» o «al» y
# aquí va «del». Sin esta forma, el amparo en revisión cuyo acto viene de un
# juzgado común se quedaba sin responsable originaria.
# LAS RANURAS QUE NOMBRAN AL AUTOR DEL ACTO. Medido sobre el corpus: los
# patrones de arriba viven en el TEXTO CRUDO de la recurrida, que sólo se guarda
# en 14 de 108 sesiones; lo que hay en las 108 es la narrativa —`antecedentes` y
# `resumen_acto`—, y ahí el acto se atribuye con otras formas: «dictado por»,
# «emitido por», «atribuidos a». Sin estas ranuras la sede se quedaba en hueco
# en 36 de 50 revisiones.
_RX_ORIGINARIA_AUTOR = re.compile(
    r"(?:dictad[oa]s?|emitid[oa]s?|pronunciad[oa]s?|practicad[oa]s?|"
    r"decretad[oa]s?|ordenad[oa]s?)\s+por\s+"
    r"((?:la\s+|el\s+)?[A-ZÁÉÍÓÚÑ][\w\sáéíóúñ,\.]{6,90}?)"
    r"(?=\s+(?:en|dentro|el|la|los|las|al|con|que|y)\s|[;\.]|$)"
    r"|atribuid[oa]s?\s+a(?:l)?\s+"
    r"((?:la\s+|el\s+)?[A-ZÁÉÍÓÚÑ][\w\sáéíóúñ,\.]{6,90}?)"
    r"(?=\s+(?:en|dentro|que|y|consistente)\s|[;\.]|$)", re.I)

_RX_ORIGINARIA_CONTRA = re.compile(
    r"(?:contra|combate)\s+(?:los\s+)?actos?\s+(?:reclamados?\s+)?"
    r"(?:de\s+l[ao]s?|del|de|a\s+l[ao]s?|al|a)\s+"
    r"((?:la\s+|el\s+)?[A-ZÁÉÍÓÚÑ][\w\sáéíóúñ,\.]{6,90}?)"
    r"(?=\s+y\s+del?\s|\s+que\s+hizo|\s+consistente|[;\.]|$)", re.I)

_RX_NO_ES = re.compile(
    r"ju(?:ez|zgado)\s+.{0,30}de\s+distrito|tribunal\s+colegiado", re.I)


# LOS CARGOS QUE TAMBIÉN SON AUTORIDAD. `fase_autoridad` nombra los órganos que
# juzgan; aquí hace falta además quien ADMINISTRA, porque el acto reclamado de
# un amparo indirecto lo dicta tan a menudo un director de ingresos como un juez
# de primera instancia.
_RX_ROL_DE_AUTORIDAD = re.compile(
    r"\b(?:ju(?:ez|eza|zgado)|sala|tribunal|magistrad\w*|junta|pleno|"
    r"direc(?:tor|tora|ci[óo]n)|secretar\w*|presiden\w*|titular|delegad\w*|"
    r"subdelegad\w*|subdirec\w*|administrador\w*|instituto|comisi[óo]n|"
    r"consejo|coordinad\w*|jefe|jefa|tesorer\w*|oficial|notari\w*|"
    r"registrador\w*|fiscal|agente|procurador\w*|ayuntamiento|municipio|"
    r"encargad\w*|superintenden\w*|contralor\w*|recaudad\w*|"
    r"comisionad\w*|inspector\w*|auditor\w*)\b", re.I)


def _sirve(n: str, excluir_amparo: bool = True) -> bool:
    """¿Es un nombre de autoridad y no un trozo de prosa?

    LA MAYÚSCULA SE COMPRUEBA AQUÍ porque en el patrón no sirve: `re.I` anula
    la clase `[A-ZÁÉÍÓÚÑ]` y deja pasar «su informe justificado».

    `excluir_amparo` distingue los dos usos, y son opuestos. Para el RESOLUTIVO
    hay que descartar a los órganos de amparo: el juzgado de distrito es el
    recurrido, no la responsable, y ampararse contra su acto sería el error que
    `_RX_NO_ES` viene a impedir. Para saber la SEDE DEL ACTO hay que admitirlos:
    cuando lo que se reclamó fue el auto de un juez de distrito, la respuesta
    correcta es precisamente «un órgano de amparo».
    """
    if len(n) < 8 or not n[:1].isupper():
        return False
    if excluir_amparo and _RX_NO_ES.search(n):
        return False
    # Y QUE SEA UNA AUTORIDAD, NO UNA FRASE QUE EMPIEZA EN MAYÚSCULA. Medido
    # sobre el corpus: el amparo en revisión 888/2026 devolvía «Usted y otras
    # autoridades», que pasa el largo y la mayúscula y no nombra a nadie.
    #
    # NO BASTA `fase_autoridad.identificable`, y la prueba lo cazó: ese
    # comprobador tiene vocabulario de órganos JURISDICCIONALES y rechaza
    # «Director de Ingresos del Municipio de Querétaro», que es una responsable
    # de las más corrientes en amparo administrativo. Usarlo a secas habría
    # tirado la materia entera. Vale como señal positiva, y al lado va el
    # vocabulario de los cargos que también son autoridad.
    if _RX_ROL_DE_AUTORIDAD.search(n):
        return True
    try:
        import fase_autoridad as _fa
        return bool(_fa.identificable(n))
    except Exception:
        return True


def responsable_originaria(texto: str, excluir_amparo: bool = True) -> str:
    """La autoridad del acto reclamado, o cadena vacía.

    Por omisión descarta a los órganos de amparo, que es lo que necesita el
    resolutivo. `sede_del_acto` la llama con `excluir_amparo=False`.
    """
    # EL FORMATO ROTULADO PRIMERO, sobre el texto SIN aplanar: la etiqueta y el
    # nombre van en líneas distintas y aplanar los saltos borra la frontera.
    for m in _RX_ORIGINARIA_ROTULO.finditer(texto or ""):
        n_ = " ".join((m.group(1) or "").split()).strip(" ,.")
        if _sirve(n_, excluir_amparo):
            return n_
    t = " ".join((texto or "").split())
    # «contra actos del …», que es como lo escribe el corpus.
    for m in _RX_ORIGINARIA_CONTRA.finditer(t):
        n = " ".join((m.group(1) or "").split()).strip(" ,.")
        if _sirve(n, excluir_amparo):
            return n
    # «dictado por …», «atribuidos a …»: las ranuras de la narrativa.
    for m in _RX_ORIGINARIA_AUTOR.finditer(t):
        n = " ".join((m.group(1) or m.group(2) or "").split()).strip(" ,.")
        if _sirve(n, excluir_amparo):
            return n
    for m in _RX_ORIGINARIA.finditer(t):
        n = " ".join((m.group(1) or m.group(2) or "").split()).strip(" ,.")
        if _sirve(n, excluir_amparo):
            return n
    return ""


# ═══════════════════════════════════════════════════════════════════════════
# LA SEDE DEL ACTO Y EL CUADERNO DEL QUE VIENE LA RECURRIDA
# ═══════════════════════════════════════════════════════════════════════════
#
# David, 13-sep-2026: «cuando se trata de amparo en revisión, si el tema no se
# refiere a la suspensión definitiva —que sería revisión contra la sentencia
# dictada en el CUADERNO INCIDENTAL—, las medidas cautelares que dicta la
# responsable no se rigen por la Ley de Amparo, sino por la ley que rige el acto
# reclamado. No quiero que le impongas al modelo que invoque esa ley, sino que
# modifiques la ARQUITECTURA para que lo entienda».
#
# Eso son dos datos, y los dos SE LEEN del expediente. Preguntárselos al
# secretario sería una casilla más; pedírselos al modelo, una ocasión más de
# inventarlos.
#
#   SEDE DEL ACTO — quién dictó el acto reclamado del amparo indirecto:
#     · «amparo»    un órgano de amparo. Su auto o su interlocutoria SÍ se rigen
#                   por la Ley de Amparo, porque es la ley que él aplicó.
#     · «ordinaria» una autoridad del fuero común o administrativo. Entonces
#                   rige LA LEY QUE ELLA APLICÓ, y los preceptos de la
#                   suspensión del amparo no vienen a cuento.
#
#   CUADERNO — de dónde sale la sentencia recurrida:
#     · «principal»  de la audiencia constitucional.
#     · «incidental» del incidente de suspensión. Ahí el objeto del recurso ES
#                    la suspensión del juicio de amparo, y la Ley de Amparo
#                    gobierna el fondo con todas las letras.
#
# LA ASIMETRÍA ES LO QUE HACE FIABLE ESTO: los órganos de amparo son un conjunto
# CERRADO Y CORTO —juzgados de distrito, tribunales colegiados, unitarios de
# circuito, plenos regionales—, mientras que las autoridades ordinarias son
# incontables. Así que se reconoce lo poco y todo lo demás es lo otro, en vez de
# intentar enumerar el mundo.
#
# OJO CON EL UNITARIO: el «Tribunal Unitario de Circuito» es órgano de amparo;
# el «Tribunal Unitario Agrario» es autoridad ordinaria y sus sentencias se
# reclaman en amparo. Por eso se exige «de circuito».
_ES_ORGANO_DE_AMPARO = re.compile(
    r"ju(?:ez|eza|zgado)\s+[^,;.]{0,40}\bde\s+distrito\b"
    r"|tribunal(?:es)?\s+colegiado"
    r"|tribunal(?:es)?\s+unitario\s+de\s+circuito"
    r"|pleno\s+regional",
    re.I)


# ═══ LA SEDE ES EL CARÁCTER, NO LA CLASE DE ÓRGANO ═══════════════════════════
#
# La primera versión decía: si quien dictó el acto es un juzgado de distrito, la
# sede es de amparo. El corpus trae el contraejemplo y lo trae ya cargado: el
# amparo en revisión 888/2026. Allí el acto reclamado es un auto del JUZGADO
# QUINTO DE DISTRITO dictado dentro de unas MEDIDAS DE ASEGURAMIENTO en
# jurisdicción ordinaria concurrente —lo fundó en los artículos 227, 228, 231,
# 240 y 241 del Código Federal de Procedimientos Civiles, no en la Ley de
# Amparo—, y el recurrido es otro juzgado de distrito distinto.
#
# O sea: un juez federal puede actuar en jurisdicción ordinaria, y entonces su
# acto se juzga con la ley que ÉL aplicó, que es exactamente la regla de David.
# Así que primero se mira LA LEY, que es el dato que la regla nombra, y sólo si
# no consta se cae a la clase de órgano.
_RX_LEY_DEL_ACTO_AMPARO = re.compile(
    r"(?:con\s+)?fundamento\s+en[^.;]{0,120}?\bley\s+de\s+amparo\b|"
    r"art[íi]culos?\s+[\d,\s(?:y)]{1,40}\s+de\s+la\s+ley\s+de\s+amparo", re.I)
_RX_LEY_DEL_ACTO_OTRA = re.compile(
    r"(?:con\s+)?fundamento\s+en[^.;]{0,140}?\b"
    r"(c[óo]digo[^.;,]{0,60}|ley\s+(?!de\s+amparo)[^.;,]{0,60})", re.I)


def sede_del_acto(texto: str) -> tuple:
    """(«amparo»|«ordinaria»|«», por qué). Del acto reclamado del indirecto."""
    t = texto or ""
    quien = responsable_originaria(t, excluir_amparo=False)
    de_amparo = bool(_ES_ORGANO_DE_AMPARO.search(quien)) if quien else False

    # LA LEY QUE ELLA APLICÓ manda sobre quién es ella. Si el acto se fundó en
    # un código y no en la Ley de Amparo, la sede es ordinaria aunque lo haya
    # dictado un juez federal.
    otra = _RX_LEY_DEL_ACTO_OTRA.search(t)
    if de_amparo and otra and not _RX_LEY_DEL_ACTO_AMPARO.search(t):
        return "ordinaria", (
            f"{quien} — pero el acto se fundó en «{otra.group(1).strip()[:60]}», "
            f"no en la Ley de Amparo: actuó en jurisdicción ordinaria")
    if not quien:
        return "", ("No se pudo leer del expediente qué autoridad dictó el acto "
                    "reclamado del amparo indirecto.")
    if de_amparo:
        return "amparo", quien
    return "ordinaria", quien


# EL CUADERNO. La audiencia constitucional resuelve el juicio; el incidente de
# suspensión resuelve la suspensión. Los marcadores son de los que no se
# prestan: una sentencia de amparo dice cuál celebró.
# LOS SEIS MARCADORES DE «INCIDENTAL» ERAN FALSOS POSITIVOS. LOS SEIS. Medido
# sobre las 28 filas de amparo en revisión del corpus:
#   · «cuaderno incidental» sale 4 veces en 2 filas y las 2 son un cuaderno
#     incidental DEL JUICIO ORDINARIO —una remoción de albacea, una entrega de
#     posesión—, no del amparo;
#   · «incidente de suspensión» y «suspensión definitiva/provisional» salen
#     narrando el trámite de la suspensión como ANTECEDENTE de una revisión que
#     es de cuaderno principal;
#   · «interlocutoria» sale 14 veces y ninguna es interlocutoria de suspensión;
#   · «cuaderno principal» no aparece NUNCA: 0 de 108 filas.
#
# El clasificador ingenuo acertaba 21 de 28 y —esto es lo grave— 6 de sus 7
# fallos eran FALSO INCIDENTAL: declaraba aplicable la Ley de Amparo justo donde
# no rige, que es el defecto que todo esto viene a cerrar. Con dos correcciones
# sube a 27 de 28 y el único fallo que queda es un HUECO, no una respuesta
# falsa:
#   (1) «audiencia constitucional» tiene PRIORIDAD ABSOLUTA. Su especificidad es
#       perfecta: 58 apariciones en 26 de 28 revisiones y cero en los 57 asuntos
#       de las otras vías.
#   (2) la suspensión sólo cuenta si va pegada a un VERBO DE RESOLUCIÓN O DE
#       RECURSO —se recurre, se resuelve, se dicta la interlocutoria—, no a la
#       mera mención del trámite.
_RX_PRINCIPAL = re.compile(r"audiencia\s+constitucional", re.I)
_RX_INCIDENTAL = re.compile(
    r"(?:recurre|recurri[óo]|impugna|impugn[óo]|resuelve|resolvi[óo]|"
    r"dict[óo]|confirm[óo]|revoc[óo]|modific[óo])"
    r"[^.;]{0,80}?(?:interlocutori[ao]|incidente\s+de\s+suspensi[óo]n|"
    r"cuaderno\s+incidental)"
    r"|(?:interlocutori[ao]|sentencia)[^.;]{0,40}?"
    r"(?:incidente\s+de\s+suspensi[óo]n|cuaderno\s+incidental)", re.I)


def cuaderno_recurrido(texto: str) -> tuple:
    """(«principal»|«incidental»|«», por qué) del que viene la recurrida.

    SE CALLA CUANDO LOS DOS APARECEN. Una sentencia del cuaderno principal puede
    mencionar de pasada el incidente, y al revés; cuando las dos familias de
    marcadores están presentes, quien decide es el secretario y no una cuenta de
    palabras. Callar aquí no rompe nada: el resto del sistema trata el «no lo sé»
    como el caso general.
    """
    t = texto or ""
    pri = len(_RX_PRINCIPAL.findall(t))
    # PRIORIDAD ABSOLUTA, no recuento. Si el a quo celebró audiencia
    # constitucional, la sentencia recurrida es la del cuaderno principal aunque
    # el expediente narre además el trámite de la suspensión — que es lo
    # normal, y es lo que hacía fallar al recuento.
    if pri:
        return "principal", f"el a quo celebró audiencia constitucional ({pri})"
    inc = len(_RX_INCIDENTAL.findall(t))
    if inc:
        return "incidental", (f"se recurre lo resuelto en el incidente de "
                              f"suspensión ({inc})")
    return "", "el expediente no dice de qué cuaderno viene la recurrida"


# ═══════════════════════════════════════════════════════════════════════════
# QUÉ CONCLUYE EL ESTUDIO CUANDO SE ASUME JURISDICCIÓN
# ═══════════════════════════════════════════════════════════════════════════
# Levantado el sobreseimiento, el tribunal estudia los conceptos de violación
# POR PRIMERA VEZ, y el sentido de ese estudio no lo dice el recurso: que el
# agravio sea fundado prueba que el juez no debió sobreseer, no que el quejoso
# tenga razón en el fondo. `tipos_asunto.rama_revision` devolvía siempre
# «revoca_sobreseimiento_concede», de modo que la rama gemela —revocar y
# NEGAR— estaba declarada y era inalcanzable: un proyecto que levanta el
# sobreseimiento y niega el amparo no se podía escribir.
#
# Se lee del propio estudio, y sólo cuando lo dice con todas las letras.
# MEDIDO sobre los engroses reales del corpus (17 con resolutivo legible):
# acierta 13, se equivoca 0, calla 4. Callar es la respuesta correcta cuando no
# consta: quien decide el sentido es el secretario, y el aviso se lo recuerda.
_RX_NIEGA_FONDO = re.compile(
    r"(?:procede|procedente\s+es|debe|deber[áa]|ha\s+lugar\s+a|se\s+impone)\s+"
    r"(?:\w+\s+){0,2}?neg(?:ar|arse|ada)\s+(?:el\s+)?(?:amparo|la\s+protecci[óo]n)|"
    r"neg(?:ar|arse)\s+(?:el\s+)?amparo\s+(?:y\s+la\s+)?protecci[óo]n", re.I)
_RX_CONCEDE_FONDO = re.compile(
    r"(?:procede|procedente\s+es|debe|deber[áa]|ha\s+lugar\s+a|se\s+impone)\s+"
    r"(?:\w+\s+){0,2}?conced(?:er|erse|ida)\s+(?:el\s+)?(?:amparo|la\s+protecci[óo]n)|"
    r"conced(?:er|erse)\s+(?:el\s+)?amparo\s+y\s+(?:la\s+)?protecci[óo]n", re.I)


def sentido_en_plenitud(texto: str) -> str:
    """«concede» | «niega», o cadena vacía si el estudio no lo dice.

    Gana la ÚLTIMA mención, que es la conclusión: un estudio menciona la
    concesión al resumir lo que pide el quejoso y la niega al final.
    """
    t = " ".join((texto or "").split())
    if not t:
        return ""
    n = [m.start() for m in _RX_NIEGA_FONDO.finditer(t)]
    c = [m.start() for m in _RX_CONCEDE_FONDO.finditer(t)]
    if not n and not c:
        return ""
    return "niega" if (n and (not c or n[-1] > c[-1])) else "concede"


# ═══════════════════════════════════════════════════════════════════════════
# EL RESOLUTIVO DEL JUZGADO, REPRODUCIDO
# ═══════════════════════════════════════════════════════════════════════════
# David, sobre la revisión 650/2025: «Hay muchas sentencias en las que se
# concede el amparo y, al calificar como infundados los agravios, el efecto
# siempre será confirmar la sentencia en sus términos. En este caso basta con
# reproducir el resolutivo de la sentencia recurrida y en la parte final
# modificar "para los efectos precisados en la sentencia recurrida" (porque
# somos órgano revisor cuando se trata de amparo en revisión)».
#
# QUÉ SE ESCRIBÍA ANTES: «La Justicia de la Unión ampara y protege a {quejoso},
# en términos del último considerando de la resolución recurrida». Dice a quién
# se ampara y nada más: ni contra qué acto, ni con qué alcance, ni para qué
# efectos. El precedente del propio tribunal —ARA 361/2025, la única revisión
# de la carpeta que confirma amparando— sí lo dice: «…ampara y protege a María
# de la Luz Aguilera Vázquez y José Francisco Rubio Bayón, CONTRA LOS ARTÍCULOS
# 90 Y 99 de la Ley de Hacienda del Estado de Querétaro, por las razones y
# efectos especificados en el considerando sexto DE LA SENTENCIA QUE SE
# REVISA».
#
# LA COLA HAY QUE REESCRIBIRLA, y es el detalle que se escapa al copiar y
# pegar: el juzgado escribió «para los efectos precisados en el diverso séptimo
# DE ESTA SENTENCIA», y en el engrose del tribunal «esta sentencia» ya es otra.
# En el proyecto que David corrigió a mano la cola quedó sin cambiar —por eso
# lo pidió explícitamente—.
_RX_RESUELVE = re.compile(
    r"R\s*E\s*S\s*U\s*E\s*L\s*V\s*E\s*(?:N)?\s*:?", re.I)
_RX_FIN_RESOLUTIVO = re.compile(
    r"\b(?:Notif[íi]quese|As[íi]\s+lo\s+resolvi|As[íi],?\s+(?:por|lo)\s|"
    r"Publ[íi]quese)", re.I)
_RX_ORDINAL_PUNTO = re.compile(
    r"^\s*(?:PRIMERO|SEGUNDO|TERCERO|CUARTO|QUINTO|[ÚU]NICO)\s*\.\s*", re.I)
# «de esta sentencia», «del presente fallo»… todo lo que en la recurrida
# apuntaba a sí misma y en el engrose apuntaría al engrose.
_RX_COLA_PROPIA = re.compile(
    r"\bde\s+(?:est[ae]|l[ao]\s+presente)\s+"
    r"(?:sentencia|resoluci[óo]n|fallo|ejecutoria)\b", re.I)


def resolutivo_recurrida(texto: str) -> str:
    """El punto resolutivo del juzgado, listo para reproducirse. '' si no cuadra.

    Devuelve el TEXTO SIN SU ORDINAL —el ordinal lo calcula el compositor, como
    todos— y con la cola apuntando a la sentencia recurrida.

    NO REPRODUCE SI HAY MÁS DE UN PUNTO. Un resolutivo con «PRIMERO. Se
    sobresee… SEGUNDO. La Justicia de la Unión ampara…» no cabe en un punto
    solo, y encajarlo a la fuerza produciría un resolutivo que dice menos que
    el del juzgado. En ese caso se devuelve vacío y el documento escribe la
    fórmula genérica de siempre, que es lo que ya hacía.
    """
    t = " ".join((texto or "").split())
    if not t:
        return ""
    m = None
    for m in _RX_RESUELVE.finditer(t):
        pass                       # el ÚLTIMO: los anteriores son citas
    if m is None:
        return ""
    resto = t[m.end():].lstrip(" :")
    fin = _RX_FIN_RESOLUTIVO.search(resto)
    cuerpo = (resto[:fin.start()] if fin else resto[:1500]).strip()
    if not cuerpo:
        return ""
    # ¿UN SOLO PUNTO? Se cuentan los ordinales que abren punto.
    ordinales = re.findall(
        r"\b(?:PRIMERO|SEGUNDO|TERCERO|CUARTO|QUINTO|[ÚU]NICO)\s*\.", cuerpo, re.I)
    if len(ordinales) > 1:
        return ""
    cuerpo = _RX_ORDINAL_PUNTO.sub("", cuerpo).strip()
    # Un resolutivo de dos palabras no es un resolutivo: es un corte mal hecho.
    if len(cuerpo.split()) < 8 or len(cuerpo) > 1200:
        return ""
    cuerpo = _RX_COLA_PROPIA.sub("de la sentencia recurrida", cuerpo)
    return cuerpo.rstrip(" .") + "."


# QUÉ QUEDÓ FUERA DE LA REVISIÓN. David: «en el resolutivo primero puse "En la
# materia de la revisión, se confirma la sentencia recurrida". Esto porque
# había aspectos que no fueron combatidos por la recurrente como la concesión
# del amparo en sus términos, solo se dolió de las convivencias».
#
# NO SE PONE SIEMPRE, y esto está medido: la fórmula aparece en 0 de los 381
# resolutivos legibles de la carpeta del tribunal. No es la fórmula de la casa;
# es la que corresponde cuando la revisión fue parcial. Por eso sólo se escribe
# cuando el propio estudio dice que algo quedó fuera.
_RX_NO_COMBATIDO = re.compile(
    r"\bno\s+(?:fue(?:ron)?|ha\s+sido|resulta)?\s*"
    r"(?:combatid|controvertid|impugnad|recurrid)[oa]s?\b"
    r"|\bno\s+(?:combati[óo]|controvirti[óo]|impugn[óo]|recurri[óo])\b"
    r"|\bno\s+es\s+materia\s+de(?:l|\s+la)\s+(?:recurso|revisi[óo]n)\b"
    r"|\bqued[óo]\s+firme\b|\bquedaron\s+firmes\b"
    r"|\bintangib(?:le|ilidad)\b", re.I)


def hay_aspectos_no_combatidos(estudio: str) -> bool:
    """¿El estudio dice que algo de la recurrida quedó fuera de la revisión?"""
    return bool(_RX_NO_COMBATIDO.search(" ".join((estudio or "").split())))


# ═══════════════════════════════════════════════════════════════════════════
# ¿QUEDÓ ALGO SIN ESTUDIAR EN LA SENTENCIA DE LA SALA?
# ═══════════════════════════════════════════════════════════════════════════
# Es la condición del reenvío en la revisión fiscal: el colegiado revoca, pero
# no puede sustituir a la Sala en el estudio de lo que ésta omitió.
#
# NO BASTA CON QUE SE REVOQUE. La tesis 185493 marca el límite: si el vicio es
# de forma y no trasciende al sentido del fallo, el colegiado lo corrige y no
# devuelve nada. Por eso esto busca la OMISIÓN DE ESTUDIO, no la revocación.
#
# Se exige que el verbo de omisión vaya cerca de aquello que se omitió —los
# conceptos, los agravios, los argumentos, las pruebas—: «omitió» a secas
# aparece en cualquier estudio hablando de otra cosa.
_RX_SIN_ESTUDIAR = re.compile(
    r"\b(?:omiti[óo]|omisi[óo]n\s+de|dej[óo]\s+de|no\s+se\s+ocup[óo]\s+de|"
    r"no\s+analiz[óo]|no\s+estudi[óo]|se\s+abstuvo\s+de)\s+"
    r"(?:\w+\s+){0,6}?"
    r"(?:concepto|agravio|argumento|planteamiento|prueba|cuesti[óo]n|"
    r"anulaci[óo]n|impugnaci[óo]n)", re.I)


def hay_conceptos_sin_estudiar(estudio: str) -> bool:
    """¿El estudio afirma que la Sala dejó algo sin resolver?"""
    return bool(_RX_SIN_ESTUDIAR.search(" ".join((estudio or "").split())))
