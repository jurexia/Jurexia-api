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


def resolvio_a_quo(texto: str) -> str:
    """«sobresee» | «concede» | «niega», o cadena vacía si no consta."""
    t = " ".join((texto or "").split())
    if not t:
        return ""
    for clave, rx in _QUE_HIZO:
        if re.search(rx, t, re.I):
            return clave
    return ""


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
    return bool(_RX_PROCESAL.search(" ".join((texto or "").split())))


# SÓLO LOS EFECTOS. El amparo estuvo bien concedido y lo que falla es la
# restitución: lineamientos deficientes, incongruentes o excesivos. Es el único
# supuesto de resolutivo ÚNICO que no revoca nada.
_RX_EFECTOS = re.compile(
    r"(?:efectos?\s+de\s+la\s+concesi[óo]n|lineamientos?)[^.]{0,140}"
    r"(?:deficiente|incongruente|excesiv|insuficiente|imprecis|"
    r"deben?\s+(?:precisarse|modificarse|ajustarse))|"
    r"(?:modificar|precisar|ajustar)[^.]{0,60}(?:los\s+)?efectos?"
    r"[^.]{0,60}(?:de\s+la\s+)?concesi[óo]n", re.I)


def solo_los_efectos(texto: str) -> bool:
    return bool(_RX_EFECTOS.search(" ".join((texto or "").split())))


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
_RX_NO_ES = re.compile(
    r"ju(?:ez|zgado)\s+.{0,30}de\s+distrito|tribunal\s+colegiado", re.I)


def responsable_originaria(texto: str) -> str:
    """La autoridad del acto reclamado, o cadena vacía."""
    t = " ".join((texto or "").split())
    for m in _RX_ORIGINARIA.finditer(t):
        n = (m.group(1) or m.group(2) or "").strip(" ,.")
        if len(n) < 8 or _RX_NO_ES.search(n):
            continue
        return n
    return ""
