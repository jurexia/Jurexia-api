"""UNA FECHA QUE NO CONSTA EN AUTOS NO SE AFIRMA.

POR QUÉ EXISTE. ADC 93/2026, v3 (22-sep-2026). El estudio escribió: «la
notificación del acuerdo de primero de julio surtió efectos el cuatro de
agosto de dos mil veinticinco, por lo que el plazo de cinco días hábiles
comenzó el cinco siguiente». Ningún documento del asunto dice «cuatro de
agosto»: la interlocutoria aportada decía que el plazo corrió del cinco al
doce, y el modelo DEDUJO la fecha en que surtió efectos y la afirmó como
constancia («conforme a las constancias consideradas por la Sala»).

Una fecha inventada en un proyecto de sentencia es un hecho inventado. La
regla del prompt —«nunca supongas lo que consta»— no la atrapa porque no va
en condicional: va afirmada. Esto la atrapa por aritmética: toda fecha
completa (día, mes y año) que el estudio afirme tiene que aparecer en alguna
fuente del asunto —los documentos leídos, los resúmenes, los antecedentes,
lo que el secretario aportó—, en letras o en cifras.

CONSERVADOR A PROPÓSITO: sólo fechas con los tres elementos. «El cinco
siguiente», «el doce del mismo mes» no se comprueban; una fecha con día, mes
y año que no está en ninguna fuente, sí.
"""
from __future__ import annotations

import re

_MESES = {"enero": 1, "febrero": 2, "marzo": 3, "abril": 4, "mayo": 5, "junio": 6,
          "julio": 7, "agosto": 8, "septiembre": 9, "setiembre": 9, "octubre": 10,
          "noviembre": 11, "diciembre": 12}
_UNIDADES = {"uno": 1, "un": 1, "primero": 1, "dos": 2, "tres": 3, "cuatro": 4,
             "cinco": 5, "seis": 6, "siete": 7, "ocho": 8, "nueve": 9, "diez": 10,
             "once": 11, "doce": 12, "trece": 13, "catorce": 14, "quince": 15,
             "dieciséis": 16, "dieciseis": 16, "diecisiete": 17, "dieciocho": 18,
             "diecinueve": 19, "veinte": 20, "veintiuno": 21, "veintiún": 21,
             "veintiun": 21, "veintidós": 22, "veintidos": 22, "veintitrés": 23,
             "veintitres": 23, "veinticuatro": 24, "veinticinco": 25,
             "veintiséis": 26, "veintiseis": 26, "veintisiete": 27,
             "veintiocho": 28, "veintinueve": 29, "treinta": 30}
_DECENAS = {"veinte": 20, "treinta": 30, "cuarenta": 40, "cincuenta": 50,
            "sesenta": 60, "setenta": 70, "ochenta": 80, "noventa": 90}
_CENTENAS = {"cien": 100, "ciento": 100, "doscientos": 200, "trescientos": 300,
             "cuatrocientos": 400, "quinientos": 500, "seiscientos": 600,
             "setecientos": 700, "ochocientos": 800, "novecientos": 900}

_PAL = r"[a-záéíóúñü]+"
# EL DÍA Y EL AÑO SON PALABRAS DE NÚMERO, no cualquier palabra. Con `_PAL` el
# patrón se tragaba «acuerdo de primero de julio…» como (acuerdo)(primero) y
# la fecha real, que empezaba una palabra después, se perdía: medido sobre la
# v3 del 93/2026, 1 fecha leída de 6.
_DIA_RX = (r"(?:" + "|".join(sorted(_UNIDADES, key=len, reverse=True))
           + r"|(?:veinte|treinta)\s+y\s+(?:un|uno|dos|tres|cuatro|cinco|seis|siete|ocho|nueve))")
_NUM_RX = (r"(?:mil|dos|" + "|".join(sorted(set(_UNIDADES) | set(_DECENAS) | set(_CENTENAS),
                                            key=len, reverse=True)) + r"|y)")
# «cuatro de agosto de dos mil veinticinco» · «treinta y uno de enero de mil
# novecientos noventa y nueve» · «4 de agosto de 2025» · «04/08/2025»
_RX_LETRAS = re.compile(
    rf"\b({_DIA_RX})\s+de\s+({_PAL})\s+de\s+"
    rf"((?:{_NUM_RX})(?:\s+{_NUM_RX}){{0,5}})\b", re.I)
_RX_CIFRAS = re.compile(
    rf"\b(\d{{1,2}})\s+de\s+({_PAL})\s+(?:de\s+|del\s+)?(\d{{4}})\b|"
    r"\b(\d{1,2})[/-](\d{1,2})[/-](\d{4})\b", re.I)


def _dia(palabras: str):
    p = palabras.lower().strip()
    if p in _UNIDADES:
        return _UNIDADES[p]
    m = re.match(r"^(veinte|treinta)\s+y\s+(\w+)$", p)
    if m and m.group(2) in _UNIDADES and _UNIDADES[m.group(2)] < 10:
        d = _DECENAS[m.group(1)] + _UNIDADES[m.group(2)]
        return d if d <= 31 else None
    return None


def _anio(palabras: str):
    toks = [t for t in re.split(r"\s+", palabras.lower().strip()) if t and t != "y"]
    if not toks:
        return None
    total, i = 0, 0
    if toks[0] == "mil":
        total, i = 1000, 1
    elif toks[0] == "dos" and len(toks) > 1 and toks[1] == "mil":
        total, i = 2000, 2
    else:
        return None
    for t in toks[i:]:
        if t in _CENTENAS:
            total += _CENTENAS[t]
        elif t in _DECENAS:
            total += _DECENAS[t]
        elif t in _UNIDADES and t != "primero":
            total += _UNIDADES[t]
        else:
            return None
    return total if 1900 <= total <= 2100 else None


def fechas_de(texto: str) -> dict:
    """{(año, mes, día): 'cómo apareció'} en un texto, en letras o en cifras."""
    t = " ".join(str(texto or "").split())
    fuera: dict = {}
    for m in _RX_LETRAS.finditer(t):
        d, mes = _dia(m.group(1)), _MESES.get(m.group(2).lower())
        # EL AÑO SE LEE DEL PREFIJO MÁS LARGO QUE SEA UN AÑO. «dos mil
        # veinticinco se apoyó en cuatro» trae cuatro palabras de más; el
        # prefijo «dos mil veinticinco» es el año.
        toks = m.group(3).split()
        a, usado = None, ""
        for n in range(len(toks), 0, -1):
            if toks[n - 1].lower() == "y":
                continue                 # «dos mil veinticinco y» no es el año
            a = _anio(" ".join(toks[:n]))
            if a:
                usado = " ".join(toks[:n])
                break
        if d and mes and a:
            fuera.setdefault((a, mes, d),
                             f"{m.group(1)} de {m.group(2)} de {usado}")
    for m in _RX_CIFRAS.finditer(t):
        if m.group(1):
            d, mes, a = int(m.group(1)), _MESES.get(m.group(2).lower()), int(m.group(3))
        else:
            d, mes, a = int(m.group(4)), int(m.group(5)), int(m.group(6))
        if d and mes and 1 <= d <= 31 and 1 <= mes <= 12 and 1900 <= a <= 2100:
            fuera.setdefault((a, mes, d), m.group(0))
    return fuera


def sin_respaldo(estudio: str, fuentes) -> list:
    """Las fechas completas del estudio que ninguna fuente contiene.

    `fuentes` es una lista de textos (o uno solo). Devuelve las apariciones
    literales, en el orden del estudio, sin repetir.
    """
    if isinstance(fuentes, str):
        fuentes = [fuentes]
    en_autos: set = set()
    for f in (fuentes or []):
        en_autos |= set(fechas_de(f).keys())
    if not en_autos:
        # Sin fuentes no hay contra qué comprobar: no se acusa a nadie.
        return []
    return [aparicion for clave, aparicion in fechas_de(estudio).items()
            if clave not in en_autos]


def aviso(estudio: str, fuentes) -> str:
    malas = sin_respaldo(estudio, fuentes)
    if not malas:
        return ""
    return (f"{len(malas)} FECHA(S) QUE NO CONSTAN EN NINGÚN DOCUMENTO DEL ASUNTO y "
            f"el estudio afirma como hecho: {' · '.join('«' + x + '»' for x in malas[:5])}. "
            f"O están en autos y hay que citarlas de la constancia, o son una "
            f"deducción del redactor y no se afirman: verifícalas antes de firmar.")
