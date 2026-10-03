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
    # LAS PENSIONES DEL ISSSTE NO SIEMPRE SE LLAMAN «PENSIÓN» (quinta ronda,
    # 3-oct-2026; RF 4/2025 del banco: «solicitud de incorporación al sistema de
    # jubilación previsto en el artículo Décimo Transitorio de la Ley del
    # ISSSTE…» salía con el supuesto en hueco). La jubilación, el régimen del
    # Décimo Transitorio, el retiro por edad, la cesantía y la cuota pensionaria
    # son el mismo supuesto; la guarda del ISSSTE (abajo) sigue. Con la bandera
    # (`_RX_PENSION_AMPLIADA`); sin ella, «pensión» como antes.
    (re.compile(r"pensi[óo]n", re.I),
     "un aspecto relacionado con pensiones que otorga el Instituto de Seguridad "
     "y Servicios Sociales de los Trabajadores del Estado"),
)


_RX_PENSION_AMPLIADA = re.compile(
    r"pensi[óo]n|jubilaci[óo]n|d[ée]cimo\s+transitorio|retiro\s+por\s+edad|"
    r"cesant[íi]a|cuota\s+(?:diaria\s+)?pensionaria", re.I)


def supuesto_fraccion_vi(texto: str, ampliado=None) -> str:
    """La porción de la fracción VI que se surte, o «» si no se reconoce.
    `ampliado` (por omisión, lo que diga la bandera `procedencia_por_tipo`):
    las pensiones también por jubilación, Décimo Transitorio, retiro, cesantía
    o cuota pensionaria (quinta ronda)."""
    t = texto or ""
    if not _RX_SEG_SOCIAL.search(t):
        return ""
    if ampliado is None:
        ampliado = _rige_procedencia()
    for rx, porcion in _SUPUESTOS_VI:
        if ampliado and "pensiones" in porcion:
            rx = _RX_PENSION_AMPLIADA
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


def _rige_procedencia() -> bool:
    """¿Rige `procedencia_por_tipo`? False si falta el contexto o la bandera."""
    try:
        import contexto_taller as _ct_pf
        _f = getattr(_ct_pf, "rige", None)
        return bool(_f("procedencia_por_tipo")) if callable(_f) else False
    except Exception:
        return False


def _parrafo_vi(porcion: str, nuevo: bool = True) -> str:
    # LAS PENSIONES DEL ISSSTE NO SON «APORTACIONES» (3-oct-2026, rev_5). La
    # fracción VI las enumera dentro de la misma oración, pero los 6 engroses del
    # banco que la usan presentan las pensiones como hipótesis propia (RF
    # 28/2025: «esa hipótesis normativa prevé la procedencia de la revisión
    # fiscal en contra de las resoluciones relacionadas con pensiones otorgadas
    # por el Instituto…»), y llamar «resolución en materia de aportaciones de
    # seguridad social» a la que niega o calcula una pensión es describir mal lo
    # resuelto. `nuevo`: con la bandera (en el camino viejo, que reconoce la
    # fracción por el vocabulario de la sentencia, queda la fórmula de antes).
    if nuevo and "pensiones" in porcion:
        return ("El recurso es procedente en términos del artículo 63, fracción VI, "
                "de la Ley Federal de Procedimiento Contencioso Administrativo, que "
                "prevé su procedencia contra las resoluciones que versen sobre "
                "cualquier aspecto relacionado con pensiones que otorga el Instituto "
                "de Seguridad y Servicios Sociales de los Trabajadores del Estado, "
                "toda vez que la sentencia recurrida versa sobre un aspecto de esa "
                "naturaleza.")
    return (f"El recurso es procedente en términos del artículo 63, fracción "
            f"VI, de la Ley Federal de Procedimiento Contencioso Administrativo, "
            f"toda vez que la sentencia recurrida versa sobre una resolución en "
            f"materia de aportaciones de seguridad social relativa "
            f"{'al ' + porcion[3:] if porcion.startswith('el ') else 'a ' + porcion}.")


# LA NULIDAD FORMAL Y LA DE FONDO (3-oct-2026, rev_5). En 3 de 8 engroses la
# procedencia por la fracción VI razona además que la Sala no anuló por vicios
# formales sino de fondo (RF 28/2025: «de ahí que la nulidad recurrida no es de
# carácter formal, sino de fondo»), y en 5 de 8 la Sala anuló «para efectos» por
# falta de fundamentación y motivación, que es justo el caso de riesgo. No se
# afirma ni se niega: se avisa, porque sólo quien leyó la sentencia lo sabe.
# En el SENTIDO de la ficha basta «nulidad»; en el TEXTO de la sentencia no,
# porque todo él dice «juicio de nulidad»: ahí tiene que DECLARARSE.
_RX_NULIDAD = re.compile(r"\bnulidad\b|\banul[óo]\b", re.I)
_RX_NULIDAD_DECLARADA = re.compile(
    r"declar\w*\s+(?:la\s+)?nulidad|nulidad\s+lisa|\banul[óo]\b|"
    r"nulidad\s+de\s+la\s+resoluci[óo]n\s+impugnada", re.I)
_RX_PARA_EFECTOS = re.compile(
    r"nulidad[^.]{0,120}?para\s+(?:determinados\s+|los\s+)?efectos|"
    r"falta\s+de\s+(?:debida\s+)?(?:fundamentaci[óo]n|motivaci[óo]n)|"
    r"vicios?\s+(?:formales?|de\s+procedimiento)", re.I)


# CON UN DERECHO SUBJETIVO RECONOCIDO, LA NULIDAD ES DE FONDO (quinta ronda,
# 3-oct-2026; RF 6/2026 del banco: la Sala «declaró la nulidad… para efectos y
# reconoció el derecho subjetivo al incremento de la cuota pensionaria y al pago
# retroactivo de las diferencias»). El aviso preguntaba si la nulidad era formal
# o de fondo cuando el sentido ya lo dice: reconocer al actor la existencia de un
# derecho subjetivo es la nulidad del artículo 52, fracción V, inciso a), LFPCA
# (verificado en el texto local), que no es por vicios formales. Sólo se mira el
# SENTIDO de la ficha: en el texto de la sentencia «derecho subjetivo» aparece
# también en lo que pidió la actora o en los agravios. EL CONSIDERANDO NO SE
# TOCA: el engrose de RF 6/2026 no lo razona (3 de 8 sí); el aviso da el
# fundamento por si el secretario quiere decirlo.
_RX_DERECHO_SUBJETIVO = re.compile(
    r"reconoc\w*[^.;]{0,80}?derecho\s+subjetivo|derecho\s+subjetivo[^.;]{0,40}?reconoc\w*", re.I)


def reconoce_derecho_subjetivo(sentido: str = "") -> bool:
    """¿El sentido de la sentencia de la Sala (el de la ficha, o su clave
    «nulidad_derecho») dice que reconoció un derecho subjetivo?"""
    s = " ".join(str(sentido or "").split())
    return s.lower() == "nulidad_derecho" or bool(_RX_DERECHO_SUBJETIVO.search(s))


def aviso_nulidad_vi(sentido: str = "", texto: str = "") -> str:
    """El aviso de nulidad formal o de fondo para la fracción VI, o «». Mira el
    sentido de la ficha si lo hay; si no, el texto de la sentencia. Con un
    derecho subjetivo reconocido en el sentido (quinta ronda), el aviso ya no
    pregunta: dice que la nulidad es de fondo y con qué fundamento."""
    s = " ".join(str(sentido or "").split())
    if s and reconoce_derecho_subjetivo(s):
        return ("PROCEDENCIA POR LA FRACCIÓN VI Y LA SALA RECONOCIÓ UN DERECHO SUBJETIVO: la "
                "nulidad es de fondo, no por vicios formales —es la del artículo 52, fracción V, "
                "inciso a), de la Ley Federal de Procedimiento Contencioso Administrativo—. Si "
                "el considerando debe razonarlo, como algunos engroses («la nulidad decretada no "
                "es de carácter formal, sino de fondo»), ése es el fundamento. Compruébalo en los "
                "puntos resolutivos de la sentencia recurrida.")
    t = s or " ".join(str(texto or "").split())
    if not t or not (_RX_NULIDAD if s else _RX_NULIDAD_DECLARADA).search(t):
        return ""
    _efectos = bool(_RX_PARA_EFECTOS.search(t))
    return ("PROCEDENCIA POR LA FRACCIÓN VI Y LA SALA DECLARÓ LA NULIDAD"
            + (" PARA EFECTOS" if _efectos else "") + ": comprueba si la nulidad es de "
            "fondo o por vicios formales (falta de fundamentación o motivación, vicios del "
            "procedimiento). Los engroses que fundan la procedencia en esta fracción lo "
            "razonan («la nulidad decretada no es de carácter formal, sino de fondo»); si "
            "la nulidad es formal, la procedencia del recurso se discute y conviene "
            "decirlo en el considerando.")


def es_indeterminada(x) -> bool:
    """¿La cuantía de la ficha es «indeterminada»? (art. 63, fr. II: «o de
    cuantía indeterminada»)."""
    return isinstance(x, str) and bool(re.search(r"\bindeterminad[ao]\b", x, re.I))


HUECO = "*********"
_ROMANOS = ("I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X")


def fraccion_de(x) -> str:
    """«I».. «X» de lo que traiga la ficha: «II», «fr. III», «fracción vi», «6»."""
    t = " ".join(str(x or "").split()).upper()
    t = re.sub(r"^(?:FR(?:ACCI[ÓO]N)?\.?\s*)", "", t).strip(" .,)")
    if t.isdigit() and 1 <= int(t) <= 10:
        return _ROMANOS[int(t) - 1]
    return t if t in _ROMANOS else ""


def cuantia_de_cadena(x) -> float:
    """«$847,738.77», «847738.77 pesos», «1,234,567» → float; 0.0 si no hay cifra.
    Es la cuantía que el secretario escribe en la ficha («cuantia»)."""
    if isinstance(x, (int, float)):
        return float(x) if x > 0 else 0.0
    m = re.search(r"\d[\d,]*(?:\.\d+)?", str(x or "").replace(" ", ""))
    if not m:
        return 0.0
    try:
        v = float(m.group(0).replace(",", ""))
    except ValueError:
        return 0.0
    return v if v > 0 else 0.0


# LA FRACCIÓN QUE DICE LA FICHA (3-oct-2026, bandera `procedencia_por_tipo`).
# El secretario declara en «Trámite en este tribunal» por qué fracción del 63
# procede —es su decisión, la oficiosa del colegiado— y la cuantía en pesos. La
# I se motiva con la aritmética de siempre; la II, la III y la VI con su
# supuesto; las demás, con la fórmula del artículo y el PORQUÉ en hueco, porque
# ése sólo lo sabe quien leyó la sentencia. Textos de los supuestos, del 63
# local (leyes/LEYES_FEDERALES).
SUPUESTO_63 = {
    "IV": ("se trata de una resolución dictada en materia de la Ley Federal de "
           "Responsabilidades Administrativas de los Servidores Públicos"),
    "V": "se trata de una resolución dictada en materia de comercio exterior",
    "VII": ("se trata de una resolución en la que se declaró el derecho a la "
            "indemnización, o se condenó al Servicio de Administración Tributaria, en "
            "términos del artículo 34 de la Ley del Servicio de Administración Tributaria"),
    "VIII": ("se resuelve sobre la condenación en costas o la indemnización previstas en "
             "el artículo 6o. de la Ley Federal de Procedimiento Contencioso Administrativo"),
    "IX": ("se trata de una resolución dictada con motivo de las reclamaciones previstas "
           "en la Ley Federal de Responsabilidad Patrimonial del Estado"),
    "X": ("en la sentencia se declaró la nulidad con motivo de la inaplicación de una "
          "norma general, en ejercicio del control difuso de la constitucionalidad y de "
          "la convencionalidad"),
}
_INCISOS_III = ("a) interpretación de leyes o reglamentos; b) alcance de los elementos "
                "esenciales de las contribuciones; c) competencia de la autoridad o "
                "ejercicio de las facultades de comprobación; d) violaciones procesales que "
                "trasciendan al sentido del fallo; e) violaciones cometidas en las propias "
                "resoluciones o sentencias; f) las que afecten el interés fiscal de la Federación")
_RX_SAT = re.compile(r"servicio\s+de\s+administraci[óo]n\s+tributaria|\bSAT\b|"
                     r"administraci[óo]n\s+(?:desconcentrada|central|general)|"
                     r"administrador[a]?\s+(?:desconcentrad|central|general)", re.I)
_RX_SHCP = re.compile(r"secretar[íi]a\s+de\s+hacienda\s+y\s+cr[ée]dito\s+p[úu]blico|\bSHCP\b|"
                      r"procuradur[íi]a\s+fiscal\s+de\s+la\s+federaci[óo]n", re.I)
_RX_ENTIDAD = re.compile(r"secretar[íi]a\s+de\s+(?:planeaci[óo]n\s+y\s+)?finanzas|"
                         r"secretar[íi]a\s+de\s+(?:administraci[óo]n\s+y\s+finanzas|hacienda\s+del\s+estado)|"
                         r"tesorer[íi]a\s+del\s+estado|direcci[óo]n\s+(?:general\s+)?de\s+ingresos\s+del\s+estado", re.I)


def _quien_dicto_iii(autoridad: str) -> str:
    a = autoridad or ""
    if _RX_SHCP.search(a):
        return "la Secretaría de Hacienda y Crédito Público"
    if _RX_SAT.search(a):
        return "una autoridad del Servicio de Administración Tributaria"
    if _RX_ENTIDAD.search(a):
        return ("una autoridad fiscal de una entidad federativa coordinada en "
                "ingresos federales")
    return ""


def _parrafo_por_fraccion(fr: str, c: float, porcion_vi: str, autoridad: str,
                          veces: int, u: float, anio_uma: int, avisos: list,
                          indeterminada: bool = False, sentido: str = "",
                          texto: str = "") -> str:
    """El párrafo de las fracciones II, III, VI y IV-X (la I la escribe `parrafo`).
    `indeterminada`: la ficha dice «cuantía indeterminada»; `sentido`/`texto`:
    para el aviso de nulidad formal o de fondo de la fracción VI."""
    base = (f"El recurso es procedente en términos del artículo 63, fracción {fr}, de la "
            f"Ley Federal de Procedimiento Contencioso Administrativo")
    letra = _LETRA_VECES[veces]
    if fr == "II":
        t = round(veces * u, 2) if u else 0.0
        if indeterminada:
            # «O DE CUANTÍA INDETERMINADA» (art. 63, fr. II): lo que el secretario
            # declaró se escribe; antes se perdía y se tomaba una cifra de la
            # prosa o salía el hueco con un aviso que pedía justo ese dato.
            cuantia = "el asunto es de cuantía indeterminada"
        elif c and t:
            cuantia = (f"el asunto es de una cuantía de {_pesos(c)}, inferior a la señalada "
                       f"en la fracción I de ese precepto —{letra} veces el valor diario de "
                       f"la Unidad de Medida y Actualización, {_pesos(u)} en {anio_uma}, esto "
                       f"es, {_pesos(t)}—")
        else:
            cuantia = f"el asunto es de cuantía {HUECO}"
            avisos.append(
                "PROCEDENCIA POR LA FRACCIÓN II DEL ARTÍCULO 63: la ficha no trae la "
                "cuantía, y esa fracción exige que sea inferior a la de la fracción I o "
                "indeterminada. Escribe cuál (la cuantía en «Trámite en este tribunal», o "
                "«indeterminada» en el hueco).")
        avisos.append(
            "PROCEDENCIA POR IMPORTANCIA Y TRASCENDENCIA (art. 63, fr. II, LFPCA): el "
            "considerando dice que la autoridad recurrente la razonó. COMPRUEBA en el "
            "escrito de agravios que lo hizo y que el razonamiento basta: si no, el "
            "recurso se desecha.")
        return (f"{base}, toda vez que {cuantia} y la autoridad recurrente razonó su "
                f"importancia y trascendencia, como lo exige esa fracción.")
    if fr == "III":
        quien = _quien_dicto_iii(autoridad)
        if not quien:
            quien = HUECO
            avisos.append(
                "PROCEDENCIA POR LA FRACCIÓN III DEL ARTÍCULO 63: no se reconoció en la "
                "autoridad que dictó la resolución impugnada a la Secretaría de Hacienda, "
                "al SAT ni a una autoridad fiscal de una entidad coordinada. Escríbelo.")
        avisos.append(
            "PROCEDENCIA POR LA FRACCIÓN III DEL ARTÍCULO 63: el supuesto y su inciso van "
            f"en HUECO —{_INCISOS_III}—. Escribe cuál se surte y por qué.")
        return (f"{base}, toda vez que la resolución impugnada en el juicio de nulidad "
                f"la dictó {quien} y el asunto se refiere a {HUECO}, supuesto del inciso "
                f"{HUECO}) de esa fracción.")
    if fr == "VI":
        _av_nul = aviso_nulidad_vi(sentido, texto)
        if porcion_vi:
            avisos.append(
                "LA PROCEDENCIA SE FUNDÓ EN LA FRACCIÓN VI DEL ARTÍCULO 63, como dice la "
                f"ficha, en el supuesto que se reconoció en el asunto ({porcion_vi}). "
                "Compruébalo antes de firmar.")
            if _av_nul:
                avisos.append(_av_nul)
            return _parrafo_vi(porcion_vi)
        avisos.append(
            "PROCEDENCIA POR LA FRACCIÓN VI DEL ARTÍCULO 63: no se reconoció cuál de sus "
            "supuestos se surte —determinación de sujetos obligados, conceptos que "
            "integran la base de cotización, grado de riesgo para el seguro de riesgos "
            "del trabajo o pensiones del ISSSTE—. Va en hueco: escríbelo.")
        if _av_nul:
            avisos.append(_av_nul)
        # CON EL SUPUESTO EN HUECO NO SE PRESUPONEN «APORTACIONES» (rev_5): si
        # lo que se surte son las pensiones del ISSSTE, la frase ya habría
        # comprometido otra hipótesis antes de que el secretario la escriba.
        return f"{base}, toda vez que la sentencia recurrida versa sobre {HUECO}."
    sup = SUPUESTO_63.get(fr)
    if not sup:
        return ""
    avisos.append(
        f"PROCEDENCIA POR LA FRACCIÓN {fr} DEL ARTÍCULO 63, como dice la ficha: el "
        f"porqué va en HUECO. Escribe por qué {sup}.")
    return (f"{base}, que lo admite cuando {sup}, supuesto que se actualiza porque "
            f"{HUECO}.")


def parrafo(texto_fuente: str, anio_resolucion: int = 0, fecha_sentencia=None,
            fecha_interposicion=None, fecha_notificacion=None,
            fraccion: str = "", cuantia=None, autoridad: str = "",
            sentido: str = "") -> tuple:
    """(párrafo de procedencia, avisos). Cadena vacía si no se puede motivar.

    Primero la fracción I (la cuantía contra el umbral de la sentencia
    recurrida); si no alcanza o no se lee, la VI cuando el asunto es de
    aportaciones de seguridad social; si ninguna, vacío y aviso: la fracción
    que funda la procedencia la decide quien firma, no una fórmula fija.

    CON LA FICHA (3-oct-2026, bandera): `fraccion` («I».. «X», la que declaró el
    secretario) manda y no se cambia por otra en silencio; `cuantia` (la cadena
    en pesos de la ficha) manda sobre la que se lea de la prosa; `autoridad` es
    la que dictó la resolución impugnada (fracción III).

    TERCERA RONDA (3-oct-2026, rev_2 y rev_5): `cuantia` «indeterminada» no se
    sustituye por una cifra de la prosa: la fracción II la escribe y la I avisa
    que exige cuantía. `sentido` (el de la ficha: «declaró la nulidad…, para
    determinados efectos») alimenta el aviso de nulidad formal o de fondo de la
    fracción VI; sin él se mira el texto de la sentencia."""
    avisos = []
    fr = fraccion_de(fraccion)
    _indet = es_indeterminada(cuantia)
    c_ficha = (cuantia_de_cadena(cuantia) if cuantia not in (None, "") and not _indet else 0.0)
    c = c_ficha or (0.0 if _indet else cuantia_de(texto_fuente))
    # EL SENTIDO DE LA FICHA TAMBIÉN DICE EL SUPUESTO (quinta ronda; RF 6/2026: «…y
    # reconoció el derecho subjetivo al incremento de la cuota pensionaria»). Sólo
    # llega en el camino nuevo; en el viejo `sentido` es «» y nada cambia.
    porcion_vi = supuesto_fraccion_vi(" ".join(x for x in (texto_fuente, sentido) if x))
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
    # LA FRACCIÓN QUE DECLARÓ EL SECRETARIO, salvo la II con una cuantía que
    # alcanza la I: esa fracción exige cuantía inferior, y entonces procede por
    # la I (se dice en el aviso).
    if fr and fr != "I":
        _t2 = round(veces * u, 2) if u else 0.0
        if fr == "II" and c and _t2 and c > _t2:
            avisos.append(
                f"LA FICHA DICE FRACCIÓN II, PERO LA CUANTÍA ({_pesos(c)}) EXCEDE EL UMBRAL "
                f"DE LA FRACCIÓN I ({_pesos(_t2)}): la II sólo procede con cuantía inferior "
                f"o indeterminada, así que la procedencia se motivó por la I. Compruébalo.")
            fr = "I"
        else:
            return _parrafo_por_fraccion(fr, c, porcion_vi, autoridad, veces, u,
                                         anio_uma, avisos, indeterminada=_indet,
                                         sentido=sentido, texto=texto_fuente), avisos
    if not c:
        if fr == "I" and _indet:
            avisos.append(
                "LA FICHA DICE FRACCIÓN I DEL ARTÍCULO 63 Y CUANTÍA INDETERMINADA: la "
                "fracción I exige una cuantía que exceda el umbral, así que no puede "
                "motivarse con una indeterminada; con cuantía indeterminada el supuesto es "
                "el de la fracción II (importancia y trascendencia, razonadas por la "
                "autoridad). Corrige la fracción o escribe la cuantía, y vuelve a generar.")
            return (f"El recurso es procedente en términos del artículo 63, fracción I, de "
                    f"la Ley Federal de Procedimiento Contencioso Administrativo, toda vez "
                    f"que el asunto es de una cuantía de {HUECO}, que excede de {letra} "
                    f"veces el valor diario de la Unidad de Medida y Actualización vigente "
                    f"al momento de la emisión de la sentencia recurrida."), avisos
        if fr == "I":
            avisos.append(
                "LA FICHA DICE FRACCIÓN I DEL ARTÍCULO 63 Y NO TRAE LA CUANTÍA (ni se "
                "pudo leer): la comparación contra el umbral va en hueco. Escribe la "
                "cuantía en «Trámite en este tribunal» y vuelve a generar.")
            return (f"El recurso es procedente en términos del artículo 63, fracción I, de "
                    f"la Ley Federal de Procedimiento Contencioso Administrativo, toda vez "
                    f"que el asunto es de una cuantía de {HUECO}, que excede de {letra} "
                    f"veces el valor diario de la Unidad de Medida y Actualización vigente "
                    f"al momento de la emisión de la sentencia recurrida."), avisos
        if porcion_vi:
            avisos.append(
                "LA PROCEDENCIA SE FUNDÓ EN LA FRACCIÓN VI DEL ARTÍCULO 63 porque "
                + (f"se reconoció en el asunto uno de sus supuestos ({porcion_vi}). "
                   if _rige_procedencia() else
                   f"el asunto es de aportaciones de seguridad social ({porcion_vi}). ")
                + "Compruébalo antes de firmar.")
            _av_nul = aviso_nulidad_vi(sentido, texto_fuente) if _rige_procedencia() else ""
            if _av_nul:
                avisos.append(_av_nul)
            return _parrafo_vi(porcion_vi, _rige_procedencia()), avisos
        if _indet:
            avisos.append(
                "LA FICHA DICE CUANTÍA INDETERMINADA Y NO DICE LA FRACCIÓN DEL ARTÍCULO 63: "
                "con cuantía indeterminada el supuesto es el de la fracción II (importancia "
                "y trascendencia, razonadas por la autoridad recurrente) o el de otra "
                "fracción que no dependa de la cuantía. Escríbela en «Trámite en este "
                "tribunal» y vuelve a generar.")
            return "", avisos
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
        # LA CUANTÍA DE LA FICHA NO SIEMPRE ES UN CRÉDITO FISCAL (una devolución,
        # una multa, una negativa): se dice con la palabra de la ley, «cuantía».
        _sobre = (f"el asunto es de una cuantía de {_pesos(c)}, cantidad que"
                  if c_ficha else
                  f"el asunto versa sobre una resolución en la que se determinó un "
                  f"crédito fiscal por {_pesos(c)}, cantidad que")
        p = (f"El recurso es procedente en términos del artículo 63, fracción "
             f"I, de la Ley Federal de Procedimiento Contencioso "
             f"Administrativo, toda vez que {_sobre} excede de {letra} veces "
             f"el valor diario de la Unidad de Medida y Actualización vigente "
             f"al momento de la emisión de la sentencia recurrida "
             f"—{_pesos(u)} en {anio_uma}, esto es, {_pesos(t)}—, sin "
             f"que sea necesario que el asunto revista, además, importancia y "
             f"trascendencia.")
        return p, avisos
    if porcion_vi and fr != "I":
        avisos.append(
            f"LA CUANTÍA NO ALCANZA LA FRACCIÓN I ({_pesos(c)} contra "
            f"{_pesos(t)}) y la procedencia se fundó en la fracción VI, porque "
            + (f"se reconoció en el asunto uno de sus supuestos ({porcion_vi}). "
               if _rige_procedencia() else
               f"el asunto es de aportaciones de seguridad social ({porcion_vi}). ")
            + "Compruébalo antes de firmar.")
        _av_nul = aviso_nulidad_vi(sentido, texto_fuente) if _rige_procedencia() else ""
        if _av_nul:
            avisos.append(_av_nul)
        return _parrafo_vi(porcion_vi, _rige_procedencia()), avisos

    p = (f"{'La cuantía del asunto' if c_ficha else 'El crédito fiscal determinado'} "
         f"asciende a {_pesos(c)}, cantidad que "
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
