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


# ═══════════════════════════════════════════════════════════════════════════
# LEER EL DOCUMENTO FUENTE, NO LA PROSA QUE ESCRIBIMOS SOBRE ÉL
# ═══════════════════════════════════════════════════════════════════════════
# `numero_de` y `fecha_de` leen los RESULTANDOS del proyecto —la prosa que el
# modelo escribe— y están calibrados para eso. Sobre la sentencia original no
# aciertan, y no es un defecto suyo: el documento fuente escribe esos datos de
# otra manera.
#
# Medido sobre la sentencia recurrida de la revisión fiscal 91/2025 de David,
# pasada por Azure —18 páginas, 64,568 caracteres—: los dos devuelven cadena
# vacía, y el resolutivo sale con dos huecos.
#
#   · el expediente viene como «EXPEDIENTE: 695/25-09-01-7-OT», con DOS PUNTOS,
#     y `numero_de` exige un espacio detrás de la palabra.
#   · la fecha está en el proemio —«Santiago de Querétaro, …, a veintidós de
#     septiembre de dos mil veinticinco»— y `fecha_de` exige que delante vaya
#     «sentencia», «auto» o «resolución».
#
# NO SE TOCAN LOS OTROS DOS. Aciertan donde se les midió, y aflojar sus
# patrones para que además sirvan aquí es la manera de romper lo que funciona.
# Esto es un lector aparte, para una fuente distinta.

# El expediente del Tribunal Federal de Justicia Administrativa tiene una forma
# propia y muy poco confundible: 695/25-09-01-7-OT. Cinco grupos separados por
# guiones detrás de la barra. No hace falta anclarlo a ninguna palabra.
_RX_EXPTE_TFJA = re.compile(r"\b(\d{1,6}/\d{2}-\d{2}-\d{2}-\d[\w-]*)")
# Y la forma general, ya con dos puntos admitidos.
_RX_EXPTE_ROTULO = re.compile(
    r"\b(?:EXPEDIENTE|EXP)\s*[:.]?\s*([0-9][\w/.-]{4,30})", re.I)

_DIA_MES_ANO = (
    r"(?:uno|dos|tres|cuatro|cinco|seis|siete|ocho|nueve|diez|once|doce|trece|"
    r"catorce|quince|diecis[éeí]is|diecisiete|dieciocho|diecinueve|veinte|"
    r"veinti\w+|treinta(?:\s+y\s+uno)?)\s+de\s+(?:enero|febrero|marzo|abril|"
    r"mayo|junio|julio|agosto|septiembre|octubre|noviembre|diciembre)\s+de\s+"
    r"(?:dos\s+mil\s+\w+(?:\s+y\s+\w+)?|\d{4})")
# EL PROEMIO: «…Querétaro, a veintidós de septiembre de dos mil veinticinco».
_RX_FECHA_PROEMIO = re.compile(rf",\s*a\s+(?:los\s+)?({_DIA_MES_ANO})", re.I)


# ═══════════════════════════════════════════════════════════════════════════
# DOS LECTORES QUE DISCREPAN NO SE RESUELVEN ELIGIENDO UNO
# ═══════════════════════════════════════════════════════════════════════════
# Medido en el 91/2025 del 10 de septiembre: el resolutivo salió diciendo «Se
# confirma la sentencia de TRES DE NOVIEMBRE de dos mil veinticinco» mientras
# el cuerpo del mismo proyecto fechaba esa sentencia el VEINTIDÓS DE SEPTIEMBRE
# cuatro veces —en los resultandos y en los considerandos—. La única línea que
# resuelve contradecía a su propio documento.
#
# La causa no está en el lector: `datos_del_documento` busca el proemio en los
# primeros 4.000 caracteres, y con la depuración el segmento que llega como
# «acto reclamado» ya no empieza en el proemio de la sentencia —el OCR de hoy
# da 33 páginas donde la sentencia de la Sala tiene 18—. El lector leyó bien el
# proemio que tenía delante; lo que tenía delante era otro documento.
#
# Por eso no se arregla eligiendo lector. Se arregla usando la DISCREPANCIA
# como lo que es: la señal de que uno de los dos está mirando el papel
# equivocado. Cuando los dos coinciden, la fecha entra con doble apoyo. Cuando
# discrepan, va a hueco con los dos candidatos escritos, y el secretario tarda
# cinco segundos en elegir. La doctrina de la casa ya estaba escrita para el
# nombre de la autoridad y vale igual aquí: en el resolutivo, un dato
# equivocado es peor que un hueco, porque el hueco se ve.
def _norm_fecha(x: str) -> str:
    x = " ".join((x or "").lower().split())
    # El OCR y la prosa no siempre acentúan igual: «veintidos» y «veintidós»
    # son la misma fecha y discrepar por una tilde sería un hueco inventado.
    for a, b in (("á", "a"), ("é", "e"), ("í", "i"), ("ó", "o"), ("ú", "u")):
        x = x.replace(a, b)
    return x


def fecha_del_recurrido(fecha_pdf: str, prosa: str) -> tuple:
    """(fecha, aviso). Vacía cuando los dos lectores no dicen lo mismo."""
    _pdf = (fecha_pdf or "").strip()
    _prosa = (fecha_de(prosa) or "").strip()
    if _pdf and _prosa:
        if _norm_fecha(_pdf) == _norm_fecha(_prosa):
            return _pdf, ""
        return "", (
            f"LA FECHA DE LA SENTENCIA RECURRIDA NO CUADRA y por eso sale en "
            f"hueco: leída del PDF dice «{_pdf}» y el cuerpo de este proyecto "
            f"la fecha «{_prosa}». Una de las dos está mirando el documento "
            f"equivocado —con la depuración, el segmento del acto puede "
            f"empezar antes del proemio de la sentencia—. Comprueba cuál es la "
            f"buena en la carátula y escríbela: el resolutivo no puede "
            f"contradecir a su propio documento.")
    return (_pdf or _prosa), ""


def datos_del_documento(texto: str) -> dict:
    """{'expediente', 'fecha'} leídos de la sentencia original. '' si no consta.

    Se prefiere lo que MÁS SE REPITE, no lo primero: en una sentencia el número
    de expediente aparece en el encabezado de cada página, y cualquier otro
    número que se mencione de paso lo hace una vez.
    """
    t = " ".join((texto or "").split())
    if not t:
        return {"expediente": "", "fecha": ""}

    from collections import Counter
    cuenta = Counter(m.group(1) for m in _RX_EXPTE_TFJA.finditer(t))
    if not cuenta:
        cuenta = Counter(m.group(1).rstrip(".,;")
                         for m in _RX_EXPTE_ROTULO.finditer(t))
    expte = ""
    if cuenta:
        mejor, n = cuenta.most_common(1)[0]
        # UNA SOLA APARICIÓN NO IDENTIFICA UNA SENTENCIA. El expediente propio
        # se repite; un número citado de pasada, no.
        # DOS APARICIONES, SIEMPRE. Con `or len(cuenta) == 1` bastaba una, y
        # entonces un número citado de pasada —«el juicio 123/20-09-01-4-OT,
        # invocado como precedente»— se colaba al resolutivo como si fuera la
        # sentencia que se revisa. El expediente propio va en el encabezado de
        # cada página: si sólo aparece una vez, no es el propio. Y quedarse sin
        # el dato es un hueco visible con su aviso; ponerlo mal es una
        # sentencia mal identificada que nadie relee.
        if n >= 2:
            expte = mejor

    # LA FECHA, DEL PRINCIPIO DEL DOCUMENTO. Más allá del proemio empiezan las
    # fechas de los antecedentes —la demanda, el emplazamiento, las pruebas— y
    # cualquiera de ellas leída aquí pondría en el resolutivo una sentencia que
    # no es la que se revisa.
    fecha = ""
    m = _RX_FECHA_PROEMIO.search(t[:4000])
    if m:
        fecha = " ".join(m.group(1).split())
    return {"expediente": expte, "fecha": fecha}
