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


def resolvio_a_quo(texto: str, antecedentes: str = "") -> str:
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
    cuenta = {}
    for clave, rx in (_QUE_HIZO_VERBOS if _solo_verbos else _QUE_HIZO):
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
_RX_ORIGINARIA_ROTULO = re.compile(
    r"autoridad(?:es)?\s+responsable(?:s)?\s*:?\s*\n?\s*"
    r"((?:la\s+|el\s+)?[A-ZÁÉÍÓÚÑ][\w\sáéíóúñ,\.]{6,90}?)"
    r"(?=\n|acto\s+reclamado|[;\.]|$)", re.I)

_RX_NO_ES = re.compile(
    r"ju(?:ez|zgado)\s+.{0,30}de\s+distrito|tribunal\s+colegiado", re.I)


def responsable_originaria(texto: str) -> str:
    """La autoridad del acto reclamado, o cadena vacía."""
    # EL FORMATO ROTULADO PRIMERO, sobre el texto SIN aplanar: la etiqueta y el
    # nombre van en líneas distintas y aplanar los saltos borra la frontera.
    for m in _RX_ORIGINARIA_ROTULO.finditer(texto or ""):
        n_ = " ".join((m.group(1) or "").split()).strip(" ,.")
        if len(n_) >= 8 and not _RX_NO_ES.search(n_):
            return n_
    t = " ".join((texto or "").split())
    for m in _RX_ORIGINARIA.finditer(t):
        n = (m.group(1) or m.group(2) or "").strip(" ,.")
        if len(n) < 8 or _RX_NO_ES.search(n):
            continue
        return n
    return ""


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
