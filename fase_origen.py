"""EL EXPEDIENTE DE ORIGEN Y LA FECHA DEL FALLO, LEÍDOS DE LOS AUTOS.

David: «Obligar a la IA a extraer el expediente de origen y la fecha del fallo
desde el formulario o el OCR para no generar comodines con asteriscos».

Tenía razón en que era evitable. La plantilla de «Existencia del acto
reclamado» pide `{expediente}` —«se corrobora con los autos del expediente
905/2017, que acompañó al referido informe»— y nadie alimentaba ese marcador,
así que salía «los autos del expediente *********» en el considerando SEGUNDO.

DE DÓNDE SE SACA, Y DE DÓNDE NO. No se le pregunta otra vez al modelo: se lee
de lo que YA escribió en los resultandos, que salió del OCR del expediente y ya
pasó por su lectura. Pedirlo aparte sería una llamada más y una ocasión más de
inventarlo.

LA TRAMPA, y es la que se ha repetido en este proyecto: una heurística de una
palabra explota dentro de un expediente pegado. Buscar «el primer número con
forma de expediente» en cien mil caracteres de OCR devuelve siempre algo —una
tesis «2a./J. 58/2010», un acuerdo «3/2013», la foja «1620»—. Por eso:

  · se busca sólo en los RESULTANDOS, que son cuatro párrafos, no en el OCR;
  · se exige que el número vaya PRECEDIDO de una palabra que lo declare
    expediente («expediente», «juicio», «toca», «juicio de amparo»);
  · se descartan las claves de tesis —llevan letras y barras antes— y los
    acuerdos generales;
  · y se descarta el número del PROPIO asunto: el toca de este recurso no es
    el expediente de origen.

Si nada de eso se cumple, devuelve vacío y el hueco se queda: un hueco se ve y
se rellena; un expediente equivocado se firma.
"""

from __future__ import annotations

import re

# «expediente 905/2017», «juicio de amparo indirecto 742/2023-II», «toca civil
# 374/2019», «juicio de nulidad 1409/24-09-01-5-OT».
_RX = re.compile(
    r"\b(?:expediente|juicio|toca|cuaderno|amparo(?:\s+(?:indirecto|directo))?"
    r"|nulidad)\s+(?:de\s+\w+\s+)?"
    r"(?:n[úu]mero\s+)?"
    r"(\d{1,5}\s*/\s*\d{2,4}(?:\s*-\s*[\w-]{1,14})?)",
    re.I)

# Lo que TIENE forma de expediente y no lo es.
_NO_ES = re.compile(
    r"(?:[A-Za-z]{1,4}\.?\s*/\s*J\.?|tesis|jurisprudencia|acuerdo\s+general|"
    r"registro\s+digital|p[áa]gina|foja)", re.I)


def _mismo(numero: str, propio: str) -> bool:
    """¿Es el número de ESTE asunto y no el de origen?"""
    n = re.sub(r"\s", "", numero or "")
    p = re.sub(r"[\s.]", "", (propio or "")).replace("-", "/")
    return bool(n and p and (n in p or n.split("-")[0] in p))


def numero_de(resultandos: str, propio: str = "") -> str:
    """El expediente de origen, o cadena vacía."""
    t = " ".join((resultandos or "").split())
    if not t:
        return ""
    for m in _RX.finditer(t):
        antes = t[max(0, m.start() - 60):m.start()]
        if _NO_ES.search(antes):
            continue
        num = re.sub(r"\s*", "", m.group(1))
        if _mismo(num, propio):
            continue
        return num
    return ""


# ═══════════════════════════════════════════════════════════════════════════
# LA FECHA DE LO RECURRIDO
# ═══════════════════════════════════════════════════════════════════════════
# David: «obligar a la IA a extraer el expediente de origen Y LA FECHA DEL
# FALLO». Es el otro marcador que salía en asteriscos: la plantilla de
# procedencia de la queja dice «se impugna el auto de {fecha_acto}» y nadie lo
# alimentaba, así que el considerando que funda la procedencia decía «el auto de
# *********» —justo al lado del inciso que sí se dedujo—.
#
# Se lee del mismo sitio y con la misma disciplina: de los resultandos ya
# escritos, en letra (que es como los escribe el corpus), y sólo cuando la
# fecha va PEGADA a la palabra que la declara —auto, acuerdo, sentencia,
# resolución—. Un «primer día que aparezca» se llevaría la de la notificación,
# la de la presentación o la del emplazamiento, que están en el mismo párrafo.
_DIAS = (r"(?:uno|dos|tres|cuatro|cinco|seis|siete|ocho|nueve|diez|once|doce|"
         r"trece|catorce|quince|diecis[éeí]is|diecisiete|dieciocho|diecinueve|"
         r"veinte|veinti\w+|treinta(?:\s+y\s+uno)?)")
_MESES = (r"enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|"
          r"octubre|noviembre|diciembre")

_RX_FECHA = re.compile(
    r"\b(?:auto|acuerdo|sentencia|resoluci[óo]n|prove[íi]do|interlocutoria)\b"
    r"[^.]{0,40}?\bde\s+(" + _DIAS + r"\s+de\s+(?:" + _MESES + r")\s+de\s+"
    r"(?:dos\s+mil\s+\w+(?:\s+\w+)?|\d{4}))",
    re.I)


def fecha_de(resultandos: str) -> str:
    """La fecha de lo recurrido, en letra, o cadena vacía."""
    t = " ".join((resultandos or "").split())
    m = _RX_FECHA.search(t)
    return m.group(1).strip() if m else ""
