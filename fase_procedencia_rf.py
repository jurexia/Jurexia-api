"""LA PROCEDENCIA DE LA REVISIÓN FISCAL, MOTIVADA DE OFICIO.

David: «En la revisión fiscal es estrictamente obligatorio que el Tribunal
Colegiado motive de oficio por qué se surte la procedencia —por ejemplo,
verificando si el crédito fiscal de $847,738.77 supera el umbral de las 3,500
UMAs conforme al art. 63, fr. I, LFPCA—. La autoridad dedicó sus tres agravios
a justificar la procedencia, por lo que el Colegiado no puede limitarse a una
fórmula vacía de dos líneas».

Lo que salía era esa fórmula vacía, y encima llamaba JUICIO al recurso:

    «El juicio es procedente y no se advierte causa de improcedencia que impida
     el análisis de la controversia.»

═══════════════════════════════════════════════════════════════════════════
LO QUE ESTE MÓDULO AFIRMA Y LO QUE NO
═══════════════════════════════════════════════════════════════════════════
La aritmética es determinista y comprobable: cuantía contra 3,500 veces el
valor diario de la UMA del año en que se emitió la resolución recurrida. Eso se
calcula y se escribe.

Lo que NO se hace es decidir por el secretario cuando falta una pieza. Si no se
puede leer la cuantía, o no se conoce la UMA de ese año, NO se escribe un
razonamiento a medias: se deja la fórmula del corpus y se avisa. Un considerando
de procedencia mal motivado es peor que uno escueto, porque el escueto se ve.

LOS VALORES DE LA UMA son los publicados por el INEGI, vigentes del 1 de febrero
de cada año. Se anotan aquí porque son cifra pública y estable, pero cada una
lleva su año: si el asunto es de un año que no está en la tabla, no se inventa
—se avisa y se deja el hueco—. El propio proyecto ya usó 113.14 para 2025 al
sintetizar los agravios de la RF 44/2025, lo que da un punto de contraste.
"""

from __future__ import annotations

import re

# Valor DIARIO de la UMA, por año (INEGI, vigente desde el 1 de febrero).
UMA_DIARIA = {
    2016: 73.04, 2017: 75.49, 2018: 80.60, 2019: 84.49, 2020: 86.88,
    2021: 89.62, 2022: 96.22, 2023: 103.74, 2024: 108.57, 2025: 113.14,
    2026: 117.31,     # INEGI, DOF 09-01-2026, vigente desde el 1-feb-2026
}

# EL UMBRAL CAMBIÓ EL 10 DE JUNIO DE 2026. La reforma a la LFPCA publicada en
# el DOF el 09-06-2026 (en vigor al día siguiente, transitorio primero) llevó la
# fracción I del artículo 63 de «tres mil quinientas veces» a «veintisiete mil
# veces la Unidad de Medida y Actualización, vigente al momento de la emisión
# de la resolución o sentencia». El código seguía comparando contra 3,500: una
# cuantía de un millón de pesos se declaraba procedente cuando hoy no alcanza
# (27,000 × 117.31 = 3,167,370). Verificación de normas, 27-sep-2026.
import datetime as _dt

VECES_ANTES = 3500
VECES_DESDE_2026 = 27000
REFORMA_63 = _dt.date(2026, 6, 10)
VECES = VECES_ANTES   # compatibilidad: el umbral de antes de la reforma
_LETRA_VECES = {VECES_ANTES: "tres mil quinientas", VECES_DESDE_2026: "veintisiete mil"}


def veces_de(fecha_sentencia) -> int:
    """Cuántas UMAs pide la fracción I según la sentencia recurrida.

    QUÉ FECHA DECIDE. No hay transitorio para la cuantía ni jurisprudencia de
    la SCJN sobre el umbral nuevo. Se toma la fecha de EMISIÓN de la sentencia
    recurrida, con el criterio orientador PC.I.A. J/147 A (10a.), registro
    2020101 —la procedencia del recurso se rige por la norma vigente al
    dictarse la sentencia recurrida—, que coincide con la lectura del propio
    precepto («vigente al momento de la emisión de la resolución o
    sentencia»). El caso de frontera —sentencia anterior, recurso posterior—
    sale con aviso (`parrafo`). Sin fecha, el texto vigente."""
    if fecha_sentencia is None:
        return VECES_DESDE_2026
    if isinstance(fecha_sentencia, _dt.datetime):
        fecha_sentencia = fecha_sentencia.date()
    return VECES_DESDE_2026 if fecha_sentencia >= REFORMA_63 else VECES_ANTES


def uma_de(fecha) -> tuple:
    """(valor diario, año de la UMA) VIGENTE en esa fecha: la de cada año rige
    del 1 de febrero al 31 de enero siguiente, así que en enero manda la del
    año anterior. (0.0, año) si no consta."""
    anio = fecha.year if (fecha.month, fecha.day) >= (2, 1) else fecha.year - 1
    return UMA_DIARIA.get(anio, 0.0), anio


_UNIDADES = {
    "un": 1, "uno": 1, "primero": 1, "dos": 2, "tres": 3, "cuatro": 4, "cinco": 5,
    "seis": 6, "siete": 7, "ocho": 8, "nueve": 9, "diez": 10, "once": 11,
    "doce": 12, "trece": 13, "catorce": 14, "quince": 15, "dieciseis": 16,
    "diecisiete": 17, "dieciocho": 18, "diecinueve": 19, "veinte": 20,
    "veintiuno": 21, "veintiun": 21, "veintidos": 22, "veintitres": 23,
    "veinticuatro": 24, "veinticinco": 25, "veintiseis": 26, "veintisiete": 27,
    "veintiocho": 28, "veintinueve": 29, "treinta": 30, "treinta y uno": 31,
}
_MESES = {m: i for i, m in enumerate(
    ("enero febrero marzo abril mayo junio julio agosto septiembre octubre "
     "noviembre diciembre").split(), 1)}


def fecha_de_letra(texto: str):
    """«veintidós de septiembre de dos mil veinticinco» → date, o None.
    También acepta la ISO del formulario. Si no se puede leer con seguridad,
    None: una fecha supuesta decidiría el umbral."""
    t = (texto or "").strip()
    if not t:
        return None
    try:
        return _dt.date.fromisoformat(t[:10])
    except Exception:
        pass
    x = " ".join(t.lower().split())
    for a, b_ in (("á", "a"), ("é", "e"), ("í", "i"), ("ó", "o"), ("ú", "u")):
        x = x.replace(a, b_)
    m = re.search(r"\b([a-z]+(?: y uno)?|\d{1,2})\s+de\s+([a-z]+)\s+(?:del?\s+)?(?:año\s+)?"
                  r"(dos\s+mil(?:\s+[a-z]+(?:\s+y\s+[a-z]+)?)?|\d{4})\b", x)
    if not m:
        return None
    d = int(m.group(1)) if m.group(1).isdigit() else _UNIDADES.get(m.group(1))
    mes = _MESES.get(m.group(2))
    y = m.group(3)
    if y.isdigit():
        anio = int(y)
    else:
        resto = y.replace("dos mil", "").strip()
        anio = 2000 + (_UNIDADES.get(resto, 0) if resto else 0)
        if resto and not _UNIDADES.get(resto):
            return None
    try:
        return _dt.date(anio, mes, d) if (d and mes) else None
    except ValueError:
        return None


# LA FRACCIÓN VI, SÓLO CUANDO EL ASUNTO ES DE APORTACIONES DE SEGURIDAD SOCIAL.
# La plantilla del banco la fijaba para TODAS las revisiones fiscales —«El
# recurso es procedente conforme al artículo 63, fracción VI»—, y el propio
# banco la mide en 14 de 28: la otra mitad procede por otra fracción. Y con la
# UMA de 2026 ausente de la tabla, era la que salía en cada revisión fiscal
# notificada este año. Aquí se reconoce el supuesto por su vocabulario y se
# motiva con la porción que corresponde; si no se reconoce, no se afirma.
_RX_SEG_SOCIAL = re.compile(
    r"instituto\s+mexicano\s+del\s+seguro\s+social|\bIMSS\b|\bISSSTE\b|"
    r"instituto\s+de\s+seguridad\s+y\s+servicios\s+sociales|aportaciones\s+de\s+seguridad"
    r"|cuotas\s+obrero[\s-]*patronales|infonavit", re.I)
_SUPUESTOS_VI = (
    (re.compile(r"prima\s+(?:de|del)\s+(?:seguro\s+de\s+)?riesgos?|grado\s+de\s+riesgo|"
                r"siniestralidad|riesgos?\s+de\s+trabajo", re.I),
     "el grado de riesgo de la empresa para los efectos del seguro de riesgos "
     "del trabajo"),
    (re.compile(r"base\s+de\s+cotizaci[óo]n|salario\s+base\s+de\s+cotizaci[óo]n|"
                r"integraci[óo]n\s+del\s+salario", re.I),
     "los conceptos que integran la base de cotización"),
    (re.compile(r"sujetos?\s+obligad[oa]s?|patr[óo]n\s+sustituto|sustituci[óo]n\s+patronal|"
                r"relaci[óo]n\s+(?:laboral|de\s+trabajo)\s+inexistente", re.I),
     "la determinación de sujetos obligados"),
    (re.compile(r"pensi[óo]n", re.I),
     "un aspecto relacionado con pensiones que otorga el Instituto de Seguridad "
     "y Servicios Sociales de los Trabajadores del Estado"),
)


def supuesto_fraccion_vi(texto: str) -> str:
    """La porción de la fracción VI que se surte, o «» si no se reconoce."""
    t = texto or ""
    if not _RX_SEG_SOCIAL.search(t):
        return ""
    for rx, porcion in _SUPUESTOS_VI:
        if rx.search(t):
            if "pensiones" in porcion and not re.search(r"ISSSTE|Seguridad\s+y\s+Servicios", t, re.I):
                continue
            return porcion
    return ""


# «un crédito fiscal por $847,738.77», «crédito fiscal de $ 1,234,567.00».
_RX_CUANTIA = re.compile(
    r"cr[ée]dito\s+fiscal\s+(?:determinado\s+)?(?:por|de|en)\s+(?:la\s+"
    r"cantidad\s+de\s+)?\$?\s*([\d][\d,\. ]{3,20})", re.I)


def cuantia_de(texto: str) -> float:
    """El monto del crédito fiscal, o 0.0.

    Se busca sólo donde el texto lo DECLARA crédito fiscal. Un «primer número
    con forma de dinero» dentro de cien mil caracteres de expediente casa
    siempre —con una foja, con un número de oficio, con un año—.
    """
    for m in _RX_CUANTIA.finditer(texto or ""):
        crudo = m.group(1).strip().rstrip(".,")
        # 847,738.77 → 847738.77 ; 1.234.567,00 no se usa en México.
        n = crudo.replace(" ", "").replace(",", "")
        try:
            v = float(n)
        except ValueError:
            continue
        if v > 0:
            return v
    return 0.0


def umbral(anio: int, veces: int = VECES) -> float:
    """`veces` la UMA diaria de ese año, o 0.0 si no consta."""
    u = UMA_DIARIA.get(int(anio or 0))
    return round(veces * u, 2) if u else 0.0


def _pesos(x: float) -> str:
    return f"${x:,.2f}"


def _parrafo_vi(porcion: str) -> str:
    return (f"El recurso es procedente en términos del artículo 63, fracción "
            f"VI, de la Ley Federal de Procedimiento Contencioso Administrativo, "
            f"toda vez que la sentencia recurrida versa sobre una resolución en "
            f"materia de aportaciones de seguridad social relativa "
            f"{'al ' + porcion[3:] if porcion.startswith('el ') else 'a ' + porcion}.")


def parrafo(texto_fuente: str, anio_resolucion: int = 0, fecha_sentencia=None,
            fecha_interposicion=None, fecha_notificacion=None) -> tuple:
    """(párrafo de procedencia, avisos). Cadena vacía si no se puede motivar.

    Primero la fracción I (la cuantía contra el umbral de la sentencia
    recurrida); si no alcanza o no se lee, la VI cuando el asunto es de
    aportaciones de seguridad social; si ninguna, vacío y aviso: la fracción
    que funda la procedencia la decide quien firma, no una fórmula fija."""
    avisos = []
    c = cuantia_de(texto_fuente)
    porcion_vi = supuesto_fraccion_vi(texto_fuente)
    # LA FECHA QUE DECIDE: la de la sentencia; si no se leyó, la de su
    # notificación, que es posterior —si ésta es anterior a la reforma, aquélla
    # también—.
    if isinstance(fecha_sentencia, _dt.datetime):
        fecha_sentencia = fecha_sentencia.date()
    if isinstance(fecha_notificacion, _dt.datetime):
        fecha_notificacion = fecha_notificacion.date()
    if isinstance(fecha_interposicion, _dt.datetime):
        fecha_interposicion = fecha_interposicion.date()
    ref = fecha_sentencia or fecha_notificacion
    veces = veces_de(ref)
    if fecha_sentencia is None and fecha_notificacion is not None \
            and REFORMA_63 <= fecha_notificacion <= REFORMA_63 + _dt.timedelta(days=45):
        avisos.append(
            "NO SE LEYÓ LA FECHA DE LA SENTENCIA RECURRIDA y se notificó poco "
            "después del 10 de junio de 2026: si se dictó antes, el umbral de la "
            "fracción I es el de tres mil quinientas veces, no el de veintisiete "
            "mil. Compruébalo en la carátula.")
    if (fecha_sentencia is not None and fecha_sentencia < REFORMA_63
            and fecha_interposicion is not None and fecha_interposicion >= REFORMA_63):
        avisos.append(
            "SENTENCIA ANTERIOR A LA REFORMA, RECURSO POSTERIOR: la procedencia "
            "se midió con el umbral de antes (tres mil quinientas veces la UMA), "
            "siguiendo el criterio orientador PC.I.A. J/147 A (10a.) —rige la "
            "norma vigente al dictarse la sentencia recurrida—. Si su tribunal "
            "aplica la vigente al interponerse el recurso, el umbral es de "
            "veintisiete mil veces (reforma DOF 09-06-2026).")
    letra = _LETRA_VECES[veces]
    cifra = f"{veces:,}"
    if ref is not None:
        u, anio_uma = uma_de(ref)
    else:
        anio_uma = int(anio_resolucion or 0)
        u = UMA_DIARIA.get(anio_uma, 0.0)
    if not c:
        if porcion_vi:
            avisos.append(
                "LA PROCEDENCIA SE FUNDÓ EN LA FRACCIÓN VI DEL ARTÍCULO 63 porque "
                "el asunto es de aportaciones de seguridad social "
                f"({porcion_vi}). Compruébalo antes de firmar.")
            return _parrafo_vi(porcion_vi), avisos
        avisos.append(
            "NO SE PUDO LEER LA CUANTÍA DEL CRÉDITO FISCAL ni se reconoció otro "
            "supuesto del artículo 63, así que la procedencia sale con la "
            "fracción en hueco. En revisión fiscal esa motivación es oficiosa: "
            "escribe la fracción que se surte y por qué antes de firmar.")
        return "", avisos
    if not u:
        avisos.append(
            f"NO CONSTA EL VALOR DE LA UMA PARA {anio_uma}: no se pudo "
            f"comparar la cuantía de {_pesos(c)} contra las {cifra} UMAs del "
            f"artículo 63, fracción I. Compruébalo y escríbelo.")
        return "", avisos

    t = round(veces * u, 2)
    if c > t:
        p = (f"El recurso es procedente en términos del artículo 63, fracción "
             f"I, de la Ley Federal de Procedimiento Contencioso "
             f"Administrativo, toda vez que el asunto versa sobre una "
             f"resolución en la que se determinó un crédito fiscal por "
             f"{_pesos(c)}, cantidad que excede de {letra} veces "
             f"el valor diario de la Unidad de Medida y Actualización vigente "
             f"al momento de la emisión de la sentencia recurrida "
             f"—{_pesos(u)} en {anio_uma}, esto es, {_pesos(t)}—, sin "
             f"que sea necesario que el asunto revista, además, importancia y "
             f"trascendencia.")
        return p, avisos
    if porcion_vi:
        avisos.append(
            f"LA CUANTÍA NO ALCANZA LA FRACCIÓN I ({_pesos(c)} contra "
            f"{_pesos(t)}) y la procedencia se fundó en la fracción VI, porque el "
            f"asunto es de aportaciones de seguridad social ({porcion_vi}). "
            f"Compruébalo antes de firmar.")
        return _parrafo_vi(porcion_vi), avisos

    p = (f"El crédito fiscal determinado asciende a {_pesos(c)}, cantidad que "
         f"NO excede de {letra} veces el valor diario de la Unidad "
         f"de Medida y Actualización vigente al momento de la emisión de la "
         f"sentencia recurrida —{_pesos(u)} en {anio_uma}, esto es, "
         f"{_pesos(t)}—, por lo que la procedencia no puede sustentarse en la "
         f"fracción I del artículo 63 de la Ley Federal de Procedimiento "
         f"Contencioso Administrativo.")
    avisos.append(
        f"LA CUANTÍA NO ALCANZA EL UMBRAL: {_pesos(c)} contra {_pesos(t)} "
        f"({cifra} UMAs de {anio_uma}). El recurso sólo procede si se "
        f"surte OTRA fracción del artículo 63 —o el supuesto de importancia y "
        f"trascendencia—: decídela y escríbela, porque de esto depende que el "
        f"recurso se estudie o se deseche.")
    return p, avisos
