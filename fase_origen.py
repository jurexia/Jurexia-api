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
    r"\b(?:expediente|juicio|toca|cuaderno|amparo|nulidad)\s+"
    # LA PALABRA CLAVE Y EL NÚMERO NO VAN PEGADOS. «toca civil 374/2019»,
    # «juicio agrario 905/2017», «juicio de amparo indirecto 742/2023-II»,
    # «amparo directo administrativo 448/2025»: entre una y otra caben hasta
    # tres palabras de materia o de vía. Exigiéndolos pegados, mi propio
    # ejemplo documentado —«toca civil»— devolvía vacío.
    r"(?:(?:de\s+)?(?:amparo|nulidad|sucesorio|ejecutivo|ordinario|oral|"
    r"indirecto|directo|civil|mercantil|penal|laboral|agrario|"
    r"administrativo|familiar)\s+){0,3}"
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
    # EL AÑO NO SE COME LA PALABRA SIGUIENTE. `(?:\s+\w+)?` capturaba
    # «veinticinco SE» en «…de dos mil veinticinco se registró». El único año
    # de cuatro palabras es «dos mil treinta y uno», así que la continuación
    # sólo se admite tras «y».
    r"(?:dos\s+mil\s+\w+(?:\s+y\s+\w+)?|\d{4}))",
    re.I)


# LA PRIMERA CANDIDATA NO ES LA BUENA, Y ESO PONÍA UNA FECHA FALSA EN EL
# DOCUMENTO. Medido sobre los cinco engroses reales: acertaba 3 de 5 y en las
# otras dos devolvía una fecha EQUIVOCADA —nunca vacía—. En el QA 143/2026
# ponía el auto de Presidencia (31 de marzo) donde va el auto recurrido (6 de
# marzo); en el ARA 17/2025 ponía «veintidós de junio de dos mil veintidós».
#
# El resultando primero nombra varias resoluciones en el mismo párrafo —la
# recurrida, la de Presidencia que admite, la de turno— y quedarse con la
# primera que casa es una moneda al aire. Su hermana `numero_de` acierta 5 de 5
# porque exige que el número vaya pegado a la palabra que lo declara; aquí
# faltaba la otra mitad de esa disciplina: si hay DOS candidatas, no se sabe.
#
# Y un hueco se ve. Una fecha equivocada se firma.
# LA MARCA DE «ESTO ES LO RECURRIDO». La primera versión sólo miraba los
# adjetivos —«recurrido», «impugnado»— y se dejaba fuera la forma en que el
# V I S T O lo dice de verdad, que es la preposición: «contra del auto de seis
# de marzo…, dictado por la Jueza Tercero de Distrito».
_RX_RECURRIDO = re.compile(
    r"\b(?:recurrid[oa]|impugnad[oa]|reclamad[oa]|combatid[oa]|"
    r"que\s+se\s+revisa|materia\s+del\s+recurso)\b|"
    r"\bcontra\s+(?:d?el\s+|la\s+)?"
    r"(?:auto|acuerdo|sentencia|resoluci[óo]n|prove[íi]do|interlocutoria)\b",
    re.I)


def fecha_de(resultandos: str) -> str:
    """La fecha de lo recurrido, en letra, o cadena vacía si hay duda."""
    t = " ".join((resultandos or "").split())
    cand = list(_RX_FECHA.finditer(t))
    if not cand:
        return ""
    if len(cand) == 1:
        return cand[0].group(1).strip()

    # HAY VARIAS. Sólo vale la que su propia frase declara recurrida; si
    # ninguna lo dice, o lo dicen dos, se calla.
    con_marca = []
    for m in cand:
        ini = t.rfind(". ", 0, m.start()) + 1
        fin = t.find(". ", m.end())
        frase = t[ini:fin if fin > 0 else len(t)]
        if _RX_RECURRIDO.search(frase):
            con_marca.append(m.group(1).strip())
    # QUE COINCIDAN ES MÁS PRUEBA, NO MENOS. La primera versión exigía
    # EXACTAMENTE una marcada y callaba cuando había dos —y en el engrose real
    # las dos decían la MISMA fecha, una en el V I S T O y otra en el
    # resultando primero—. Se agrupan y basta con que no se contradigan.
    distintas = {" ".join(x.lower().split()) for x in con_marca}
    if len(distintas) == 1:
        return con_marca[0]
    return ""
