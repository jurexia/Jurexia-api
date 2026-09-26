# -*- coding: utf-8 -*-
"""EL ESTUDIO DE FONDO, CONTADO SIN MODELO — las métricas del banco del estudio.

POR QUÉ EXISTE. Propuesta aprobada por David el 26-sep-2026 («Si adelante
autorizo», «Si acepto»): antes de tocar el prompt del estudio hay que poder
decir, con números que se repiten, si una variante repite menos SIN contestar
menos. El diagnóstico midió a mano la repetición en cuatro salidas; ninguna
cifra sobrevivía a la corrida siguiente. Esto es el Paso 0: medidas
DETERMINISTAS sobre el texto del proyecto, las mismas para lo generado y para
los engroses reales.

QUÉ MIDE, Y SOBRE QUÉ. Sobre la «Solución» del considerando de estudio —el
tramo que contesta—, no sobre los resúmenes, que repiten el escrito por
diseño. El corte se hace por los subtítulos que ya usa el documento
(«Sentencia reclamada», «Conceptos de violación», «Solución»), el considerando
de Efectos y la «SÍNTESIS». Cuando el engrose no rotula la Solución —siete de
los 24 Kingston— se mide el considerando entero y la fila lo dice
(`solucion_rotulada=False`), para no comparar peras con manzanas.

LAS MEDIDAS, cada una con el defecto que la motivó (w2_final §1 y §4.4):
  · palabras de la Solución — F1: se pedían 3,733 para la Solución contando
    dos veces los resúmenes;
  · apartados y palabras de apertura — C1 y fila 4: 59-84 palabras para
    rotular frente a 27 del engrose;
  · párrafos (media, p90) y frases de más de 60 palabras — claridad;
  · citas por registro y citas repetidas — C3 y V4(a);
  · el mismo precepto expuesto como premisa en más de un apartado — C2, V4(b);
  · calificación del mismo concepto más de dos veces — V8;
  · marcadores de objeción — C4;
  · recapitulación final — Decisión 3 de David: sin cierre por defecto;
  · el molde «Por cuestión de método» — C8, un ejemplo del prompt copiado;
  · órdenes a la responsable fuera de Efectos — V6;
  · remisiones con contenido frente a remisiones en blanco — V3;
  · cobertura por ordinal (`formato_sentencia.sin_contestar`, con la n del
    contador) y COBERTURA CONTRA LA DEMANDA aproximada: las «anclas duras»
    del escrito —artículo con su ley, registros, expedientes, cifras,
    fechas— que aparecen en la Solución.

LO QUE NO MIDE. No sabe si una respuesta CONTESTA: sólo si nombra. La
cobertura por anclas es una aproximación y sirve para una cosa —ver si una
variante pierde lo que la otra sí decía—, no para certificar exhaustividad.
Eso lo hacen los localizadores del Paso 0 y la lectura de David.

LA REGLA DE LA CASA, APLICADA A SÍ MISMA: toda medida se corre primero sobre
los engroses buenos (`--calibrar`). La que acusa a más del 10 % de ellos
(el umbral de w2_final §6.1, paso 6) queda marcada como candidata a no usarse
como alarma; sigue valiendo para comparar variantes entre sí.

Uso:
    .venv/bin/python metricas_estudio.py proyecto.txt [--escrito demanda.txt]
    .venv/bin/python metricas_estudio.py --calibrar [--salida bancos/estudio]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import contador_planteamientos as cp
import formato_sentencia as fs

AQUI = Path(__file__).parent
SALIDA = AQUI / "bancos" / "estudio"

# Dónde vive lo que se calibra. Los `diag/` y `93/` son del diagnóstico del
# 25-sep y viven en el scratchpad de aquella sesión: si ya no están, la
# calibración sigue con el corpus Kingston, que es la referencia de verdad.
CASOS_KINGSTON = Path("/Users/josedavidalcantarmendoza/Documents/IUREXIA-MAC/"
                      "redactor-sentencias/corpus/casos")
_SCRATCH = Path("/private/tmp/claude-501/-Users-josedavidalcantarmendoza-Documents-"
                "Viaje-a-Europa/5f71a5c8-bc09-427e-90e2-108495d0f272/scratchpad")
DIAG = _SCRATCH / "diag"
DIR93 = _SCRATCH / "93"


# ═══════════════════════════════════════════════════════════════════════════
# TEXTO PLANO QUE CONSERVA LAS POSICIONES
# ═══════════════════════════════════════════════════════════════════════════
# Minúsculas y sin acentos, CARÁCTER POR CARÁCTER: así una posición en el
# plano es la misma en el original. `unicodedata.normalize` sobre la cadena
# entera no lo garantiza —«½» se vuelve tres caracteres— y las distancias
# entre una cita y su registro dejarían de cuadrar.
def plano(t: str) -> str:
    fuera = []
    for c in (t or ""):
        d = unicodedata.normalize("NFKD", c)
        base = [x for x in d if not unicodedata.combining(x)]
        fuera.append(base[0].lower() if len(base) == 1 else c.lower())
    return "".join(fuera)


def palabras(t: str) -> int:
    # Por espacios, como `palabras` del evento «listo» (`len(estudio.split())`):
    # las cifras de la Solución se comparan con las que ya dio el taller.
    return len((t or "").split())


def _pct(xs, q: float):
    """Percentil con interpolación lineal (el de numpy por omisión)."""
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    if len(xs) == 1:
        return float(xs[0])
    k = (len(xs) - 1) * q
    i = int(k)
    j = min(i + 1, len(xs) - 1)
    return float(xs[i] + (xs[j] - xs[i]) * (k - i))


def mediana(xs):
    xs = [x for x in xs if x is not None]
    return float(statistics.median(xs)) if xs else None


# ═══════════════════════════════════════════════════════════════════════════
# EL CORTE POR SUBTÍTULOS
# ═══════════════════════════════════════════════════════════════════════════
_ORD_CONS = (r"PRIMERO|SEGUNDO|TERCERO|CUARTO|QUINTO|SEXTO|S[ÉE]PTIMO|OCTAVO|"
             r"NOVENO|D[ÉE]CIMO(?:\s+PRIMERO|\s+SEGUNDO)?")
_RX_CONSIDERANDO = re.compile(r"^\s*(?:" + _ORD_CONS + r")\.\s*(.*)$")
_RX_TITULO_ESTUDIO = re.compile(
    r"^(?:Estudio|Soluci[óo]n|Cuestiones\s+a\s+resolver|An[áa]lisis|Fondo)", re.I)
_RX_TITULO_EFECTOS = re.compile(r"^Efectos", re.I)
_RX_FIN_ESTUDIO = re.compile(
    r"^\s*(?:Por\s+lo\s+expuesto|R\s*E\s*S\s*U\s*E\s*L\s*V\s*E|S[ÍI]NTESIS\s*$)", re.I)
_RX_SUB_SR = re.compile(
    r"^\s*(?:Sentencia\s+reclamada|Resoluci[óo]n\s+reclamada|Acto\s+reclamado|"
    r"Sentencia\s+recurrida|Resoluci[óo]n\s+recurrida|"
    r"Consideraciones(?:\s+relevantes)?\s+de\s+la\s+(?:sentencia|resoluci[óo]n)"
    r"(?:\s+(?:reclamada|recurrida))?)\s*[.:]?\s*$", re.I)
_RX_SUB_CV = re.compile(
    r"^\s*(?:Conceptos\s+de\s+violaci[óo]n|Agravios|"
    r"S[íi]ntesis\s+de\s+(?:los\s+)?(?:conceptos\s+de\s+violaci[óo]n|agravios))"
    r"\s*[.:]?\s*$", re.I)
_RX_SUB_SOL = re.compile(r"^\s*Soluci[óo]n(?:\s+del\s+asunto)?\s*[.:]?\s*$", re.I)
_RX_SINTESIS = re.compile(r"^\s*S[ÍI]NTESIS\s*$")


def secciones(texto: str) -> dict:
    """El considerando de estudio y sus partes, cortado por sus subtítulos.

    Sirve igual para el documento entero que genera el taller (del .docx) y
    para el `oro` de un caso Kingston, que empieza ya en «SEXTO. Estudio.» o
    —el 722/2025— directamente en la calificación general.
    """
    lineas = (texto or "").split("\n")
    n = len(lineas)
    ini = None
    for i, ln in enumerate(lineas):
        m = _RX_CONSIDERANDO.match(ln)
        if m and _RX_TITULO_ESTUDIO.match(m.group(1).strip()):
            ini = i
            break
    # SIN RÓTULO DE CONSIDERANDO el texto ya es el estudio (el oro del
    # 722/2025 empieza en «Los conceptos de violación son ineficaces»): se mide
    # desde la primera línea, que es la calificación y no un encabezado.
    rotulado = ini is not None
    ini = ini if rotulado else 0
    desde = ini + 1 if rotulado else 0
    fin, ef_ini = n, None
    for i in range(desde, n):
        ln = lineas[i]
        if _RX_FIN_ESTUDIO.match(ln):
            fin = i
            break
        m = _RX_CONSIDERANDO.match(ln)
        if m and _RX_TITULO_EFECTOS.match(m.group(1).strip()):
            ef_ini = i
            fin = i
            break
    ef_fin = n
    if ef_ini is not None:
        for i in range(ef_ini + 1, n):
            if _RX_FIN_ESTUDIO.match(lineas[i]) or (
                    _RX_CONSIDERANDO.match(lineas[i])
                    and not _RX_TITULO_EFECTOS.match(
                        _RX_CONSIDERANDO.match(lineas[i]).group(1).strip())):
                ef_fin = i
                break
    sin_ini = next((i for i in range(n) if _RX_SINTESIS.match(lineas[i])), None)

    i_sr = i_cv = i_sol = None
    for i in range(desde, fin):
        s = lineas[i]
        if len(s.strip()) > 90:
            continue
        if i_sr is None and _RX_SUB_SR.match(s):
            i_sr = i
        elif i_cv is None and _RX_SUB_CV.match(s):
            i_cv = i
        elif i_sol is None and _RX_SUB_SOL.match(s):
            i_sol = i

    def tramo(a, b):
        return "\n".join(lineas[a:b]).strip() if a is not None and a < b else ""

    marcas = sorted(x for x in (i_sr, i_cv, i_sol) if x is not None)

    def hasta(i):
        sig = [x for x in marcas if x > i]
        return sig[0] if sig else fin

    estudio = tramo(ini, fin)
    if i_sol is not None:
        solucion = tramo(i_sol + 1, fin)
    else:
        # SIN «SOLUCIÓN» ROTULADA no hay forma honrada de separar la
        # respuesta del resumen: se toma el considerando entero y se dice.
        solucion = tramo(desde, fin)
    primera = lineas[ini] if (lineas and rotulado) else ""
    return {
        "encabezado": primera.strip() if _RX_CONSIDERANDO.match(primera or "") else "",
        "estudio": estudio,
        "sentencia_reclamada": tramo(i_sr + 1, hasta(i_sr)) if i_sr is not None else "",
        "conceptos": tramo(i_cv + 1, hasta(i_cv)) if i_cv is not None else "",
        "solucion": solucion,
        "solucion_rotulada": i_sol is not None,
        "efectos": tramo(ef_ini, ef_fin) if ef_ini is not None else "",
        "sintesis": tramo(sin_ini + 1, n) if sin_ini is not None else "",
    }


# ═══════════════════════════════════════════════════════════════════════════
# PÁRRAFOS Y FRASES
# ═══════════════════════════════════════════════════════════════════════════
_RX_CITA_LARGA = re.compile(r"[«“\"]([^»”\"]{40,})[»”\"]")
_RX_VERSALES = re.compile(r"[A-ZÁÉÍÓÚÑ][A-ZÁÉÍÓÚÑ\s,.;:«»\"“”()0-9º°/-]{40,}")


def parrafos(t: str) -> list:
    """Un párrafo por línea no vacía: así sale el texto del .docx y así están
    los engroses del corpus (a veces con una línea en blanco entre medias)."""
    return [p.strip() for p in (t or "").split("\n") if p.strip()]


def es_transcripcion(p: str) -> bool:
    """Un párrafo que es sobre todo cita —la tesis, el precepto transcrito—
    mide la fuente, no al redactor: se deja fuera de las medidas de párrafo."""
    total = palabras(p)
    if not total:
        return False
    s = p.strip()
    if s[:1] in "«“\"" and s[-1:] in "»”\".":
        return True
    resto = _RX_VERSALES.sub(" ", _RX_CITA_LARGA.sub(" ", p))
    return palabras(resto) < 0.4 * total


# Abreviaturas que llevan punto y no cierran frase. La lista es corta a
# propósito: una frase partida de más acorta la cifra; una de menos la alarga,
# y lo que se compara es la misma medida en las dos variantes.
_ABREV = re.compile(
    r"\b(arts?|fracc?|frs?|n[úu]ms?|no|lic|ing|dra?|mtr[oa]|sra?|srta|ma|mag|"
    r"arq|profr?a?|gral|cd|edo|col|mpio|av|inc|p[áa]gs?|p[áa]rrs?|reg|exp|vol|"
    r"cfr|op|cit|ob|ed|s\.a|c\.v)\.", re.I)
_RX_CORTE_FRASE = re.compile(r"(?<=[.!?])\s+(?=[A-ZÁÉÍÓÚÑ¿«“\"])")


def frases(t: str) -> list:
    fuera = []
    for p in parrafos(t):
        # Lo transcrito entre comillas no es frase del redactor.
        q = _RX_CITA_LARGA.sub(" «cita» ", p)
        q = _ABREV.sub(lambda m: m.group(0).replace(".", "∯"), q)
        for f in _RX_CORTE_FRASE.split(q):
            f = f.replace("∯", ".").strip()
            if f:
                fuera.append(f)
    return fuera


# ═══════════════════════════════════════════════════════════════════════════
# NÚMEROS ESCRITOS CON LETRA
# ═══════════════════════════════════════════════════════════════════════════
# La sentencia escribe «veintiocho de noviembre de dos mil veinticinco» y
# «doscientos setenta y nueve mil doscientos diez pesos» donde el escrito dice
# «28 de noviembre de 2025» y «$279,210.00». Sin esto, la cobertura contra la
# demanda acusaría como no contestadas todas las fechas y todas las cifras.
_VALOR = {
    "cero": 0, "un": 1, "uno": 1, "una": 1, "primero": 1, "primer": 1,
    "dos": 2, "tres": 3, "cuatro": 4, "cinco": 5, "seis": 6, "siete": 7,
    "ocho": 8, "nueve": 9, "diez": 10, "once": 11, "doce": 12, "trece": 13,
    "catorce": 14, "quince": 15, "dieciseis": 16, "diecisiete": 17,
    "dieciocho": 18, "diecinueve": 19, "veinte": 20, "veintiun": 21,
    "veintiuno": 21, "veintiuna": 21, "veintidos": 22, "veintitres": 23,
    "veinticuatro": 24, "veinticinco": 25, "veintiseis": 26,
    "veintisiete": 27, "veintiocho": 28, "veintinueve": 29, "treinta": 30,
    "cuarenta": 40, "cincuenta": 50, "sesenta": 60, "setenta": 70,
    "ochenta": 80, "noventa": 90, "cien": 100, "ciento": 100,
    "doscientos": 200, "doscientas": 200, "trescientos": 300,
    "trescientas": 300, "cuatrocientos": 400, "cuatrocientas": 400,
    "quinientos": 500, "quinientas": 500, "seiscientos": 600,
    "seiscientas": 600, "setecientos": 700, "setecientas": 700,
    "ochocientos": 800, "ochocientas": 800, "novecientos": 900,
    "novecientas": 900,
}
_MULT = {"mil": 1000, "millon": 10 ** 6, "millones": 10 ** 6}
_ALT_NUM = "|".join(sorted(list(_VALOR) + list(_MULT), key=len, reverse=True))
NUMPAL = r"(?:(?:%s)(?:\s+(?:y\s+)?(?:%s))*)" % (_ALT_NUM, _ALT_NUM)


def numero_de_palabras(s: str):
    """«doscientos setenta y nueve mil doscientos diez» → 279210."""
    total, actual, visto = 0, 0, False
    for t in plano(s).split():
        if t == "y":
            continue
        if t in _VALOR:
            actual += _VALOR[t]
            visto = True
        elif t == "mil":
            total += (actual or 1) * 1000
            actual = 0
            visto = True
        elif t in ("millon", "millones"):
            total = (total + (actual or 1)) * 10 ** 6
            actual = 0
            visto = True
        else:
            return None
    return total + actual if visto else None


# ═══════════════════════════════════════════════════════════════════════════
# LAS ANCLAS DURAS
# ═══════════════════════════════════════════════════════════════════════════
# w2_final §3.4-5 y §4.4 (V1b): lo que un argumento trae de propio y un
# resumen no puede inventar —el artículo con su ley, el registro de la tesis,
# el número de un expediente, una cifra, una fecha—. Si el escrito lo dice y
# la Solución no lo nombra nunca, es un indicio (no una prueba) de que lo que
# se apoyaba en ello no se contestó.
_MESES = {"enero": 1, "febrero": 2, "marzo": 3, "abril": 4, "mayo": 5,
          "junio": 6, "julio": 7, "agosto": 8, "septiembre": 9, "setiembre": 9,
          "octubre": 10, "noviembre": 11, "diciembre": 12}
_RX_FECHA = re.compile(
    r"\b(?P<d>\d{1,2}|" + NUMPAL + r")\s+(?:(?:dias?\s+)?del?\s+mes\s+)?de\s+"
    r"(?P<m>" + "|".join(_MESES) + r")\s+(?:de|del)\s+(?:(?:ano|año)\s+)?"
    r"(?P<a>\d{4}|" + NUMPAL + r")\b")
_RX_FECHA_NUM = re.compile(r"(?<![\d/])(\d{1,2})[/.-](\d{1,2})[/.-](\d{4})(?!\d)")


def _dia(s):
    return int(s) if s.isdigit() else numero_de_palabras(s)


def fechas(texto: str, pl: str = None) -> list:
    """[(pos, fin, 'fecha 2025-11-28')]"""
    pl = plano(texto) if pl is None else pl
    fuera = []
    for m in _RX_FECHA.finditer(pl):
        d, a = _dia(m.group("d")), _dia(m.group("a"))
        # «el uno de julio de dos mil veinticinco» empieza en «uno», pero el
        # NUMPAL también come palabras de antes que son números («los dos
        # primero…»): el día se queda con la cola que da 1-31.
        if d is None or not 1 <= d <= 31:
            toks = m.group("d").split()
            d = next((v for k in range(len(toks)) for v in [numero_de_palabras(
                " ".join(toks[k:]))] if v is not None and 1 <= v <= 31), None)
        if d is None or a is None or not 1900 <= a <= 2100:
            continue
        fuera.append((m.start(), m.end(),
                      f"fecha {a:04d}-{_MESES[m.group('m')]:02d}-{d:02d}"))
    for m in _RX_FECHA_NUM.finditer(pl):
        d, mes, a = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if 1 <= d <= 31 and 1 <= mes <= 12 and 1900 <= a <= 2100:
            fuera.append((m.start(), m.end(), f"fecha {a:04d}-{mes:02d}-{d:02d}"))
    return fuera


_RX_PESOS_NUM = re.compile(
    r"\$\s*(\d{1,3}(?:[,' ]\d{3})+(?:\.\d{1,2})?|\d+(?:\.\d{1,2})?)|"
    r"(?<![\d.,/])(\d{1,3}(?:[,']\d{3})+(?:\.\d{1,2})?|\d+(?:\.\d{1,2})?)\s*"
    r"(?:pesos|m\.\s?n\.)")
_RX_PESOS_PAL = re.compile(r"\b(" + NUMPAL + r")\s+pesos\b")


def cifras(texto: str, pl: str = None) -> list:
    pl = plano(texto) if pl is None else pl
    fuera = []
    for m in _RX_PESOS_NUM.finditer(pl):
        s = (m.group(1) or m.group(2) or "").replace(",", "").replace("'", "").replace(" ", "")
        try:
            v = int(float(s))
        except ValueError:
            continue
        if v >= 1:
            fuera.append((m.start(), m.end(), f"cifra {v}"))
    for m in _RX_PESOS_PAL.finditer(pl):
        v = numero_de_palabras(m.group(1))
        if v:
            fuera.append((m.start(), m.end(), f"cifra {v}"))
    return fuera


_RX_REGISTRO = re.compile(
    r"registro(?:\s+digital)?\s*(?:(?:numero|num\.|no\.|n\.)\s*)?[:.]?\s*"
    r"(\d{1,3}(?:[,.]\d{3}){1,2}|\d{6,7})(?!\d)")


def registros(texto: str, pl: str = None) -> list:
    pl = plano(texto) if pl is None else pl
    fuera = []
    for m in _RX_REGISTRO.finditer(pl):
        num = re.sub(r"\D", "", m.group(1))
        if 6 <= len(num) <= 7:
            fuera.append((m.start(), m.end(), f"reg {num}"))
    return fuera


# UN EXPEDIENTE ES NÚMERO/AÑO, PERO NO TODO NÚMERO/AÑO ES UN EXPEDIENTE. La
# clave de una jurisprudencia («1a./J. 104/2008») y el número de un acuerdo
# general («Acuerdo General 6/2026») tienen la misma forma; se descartan por
# lo que llevan delante.
_RX_EXPEDIENTE = re.compile(
    r"(?<![\w./])(\d{1,6}/(?:19|20)?\d{2}(?:-[0-9a-z]+)*)(?![\w/])")
_RX_NO_EXPEDIENTE = re.compile(
    r"(?:/\s*j\.?|\bj\.|\btesis|\bjurisprudencias?|\bacuerdos?(?:\s+generale?s?)?"
    r"(?:\s+(?:plenario|conjunto))?|\bregistro|\bepoca|\(\d+a\.\)|\bpag(?:ina)?s?\.?|"
    r"\bp\.|\bfojas?|\btomo)\s*(?:numero|num\.|no\.)?\s*$")


def expedientes(texto: str, pl: str = None, tapar: list = None) -> list:
    pl = plano(texto) if pl is None else pl
    if tapar:
        # Las fechas «28/11/2025» ya se contaron como fecha: se tapan antes de
        # buscar expedientes para que «28/11» no cuente dos veces.
        l = list(pl)
        for a, b, _ in tapar:
            for i in range(a, b):
                l[i] = " "
        pl = "".join(l)
    fuera = []
    for m in _RX_EXPEDIENTE.finditer(pl):
        antes = pl[max(0, m.start() - 30):m.start()]
        if _RX_NO_EXPEDIENTE.search(antes):
            continue
        if m.group(1).startswith("00/"):
            continue
        fuera.append((m.start(), m.end(), f"exp {m.group(1)}"))
    return fuera


# ── ARTÍCULO CON SU LEY ───────────────────────────────────────────────────
# Un artículo sin ley no es ancla: «el artículo 17» está en la Constitución,
# en la Ley de Amparo y en el código procesal. La ley se reconoce por su
# nombre, por su sigla o —«del mismo ordenamiento», «de la citada ley»— por la
# última que se nombró.
_LEYES = [
    (r"ley de amparo,?\s+reglamentaria de los articulos 103 y 107(?:\s+de la "
     r"constitucion politica de los estados unidos mexicanos|\s+constitucionales)?",
     "ley_amparo"),
    (r"ley reglamentaria de los articulos 103 y 107(?:\s+de la constitucion "
     r"politica de los estados unidos mexicanos|\s+constitucionales)?", "ley_amparo"),
    (r"constitucion politica de los estados unidos mexicanos", "cpeum"),
    (r"constitucion politica del estado[a-z ]{0,60}", "const_estatal"),
    (r"constitucion (?:federal|general)|carta magna|constitucionale?s?\b|\bcpeum\b|"
     r"constitucion\b", "cpeum"),
    (r"ley de amparo", "ley_amparo"),
    (r"ley agraria", "ley_agraria"),
    (r"codigo fiscal de la federacion|\bcff\b", "cff"),
    (r"ley federal del? procedimiento contencioso administrativo|\blfpca\b", "lfpca"),
    (r"ley federal del? procedimiento administrativo|\blfpa\b", "lfpa"),
    (r"codigo federal de procedimientos civiles|\bcfpc\b", "cfpc"),
    (r"codigo nacional de procedimientos civiles y familiares", "cnpcf"),
    (r"codigo nacional de procedimientos penales|\bcnpp\b", "cnpp"),
    (r"codigo del? procedimientos civiles|codigo procesal civil|\bcpc\b", "cpc"),
    (r"codigo civil federal|codigo civil para el distrito federal", "ccf"),
    (r"codigo civil", "cc"),
    (r"codigo de comercio", "ccom"),
    (r"ley general de titulos y operaciones de credito", "lgtoc"),
    (r"ley federal del trabajo|\blft\b", "lft"),
    (r"ley organica del poder judicial(?: de la federacion)?", "lopjf"),
    (r"convencion americana(?: sobre derechos humanos)?", "cadh"),
    (r"declaracion universal(?: de (?:los )?derechos humanos)?", "dudh"),
    (r"pacto internacional de derechos civiles y politicos", "pidcp"),
    (r"codigo penal", "cp"),
    (r"ley del seguro social", "lss"),
    (r"ley del impuesto sobre la renta", "lisr"),
    (r"ley del impuesto al valor agregado", "liva"),
]
# Los espacios del nombre toleran dobles espacios y saltos del PDF.
_LEYES = [(re.compile(p.replace(" ", r"\s+")), k) for p, k in _LEYES]
# EL NOMBRE DE UNA LEY QUE NO ESTÁ EN LA LISTA se reduce a sus tres primeras
# palabras con peso («ley ingresos federacion»). Se corta en la primera
# palabra que ya no puede ser del nombre: si no, «de la Ley de Ingresos y el
# artículo 20…» se comía el artículo siguiente.
_STOP_LEY = {"de", "del", "la", "las", "los", "el", "para", "sobre"}
_CORTE_LEY = {"y", "e", "o", "u", "que", "articulo", "articulos", "art", "arts",
              "numeral", "numerales", "en", "por", "con", "conforme", "segun",
              "cuando", "donde", "pues", "porque", "lo", "se", "al", "a", "como",
              "ni", "sino", "mediante", "cuyo", "cuya", "asi"}
_RX_LEY_GENERICA = re.compile(r"\b(ley|codigo|reglamento|convencion|estatuto)\b")
_RX_PALABRA_LEY = re.compile(r"\s+([a-z]+)\b")
# «del mismo ordenamiento», «de dicha ley», «del código en cita»: la ley es la
# última que se nombró. Hace falta el calificativo —antes o después—: «de la
# ley agraria» es una ley nueva, no una anáfora.
_CUERPO = r"(?:ordenamiento|ley|codigo|cuerpo\s+(?:legal|normativo)|norma|legislacion)"
_RX_ANAFORA = re.compile(
    r"(?:del|de\s+la|de\s+los|de\s+las)\s+(?:mism[oa]|citad[oa]|referid[oa]|aludid[oa]|"
    r"invocad[oa])\s+" + _CUERPO + r"|"
    r"de\s+(?:dich[oa]|es[ea]|este|esta|aquel|aquella)\s+(?:mism[oa]\s+)?" + _CUERPO + r"|"
    r"(?:del|de\s+la)\s+" + _CUERPO + r"\s+(?:en\s+cita|citad[oa]|referid[oa]|aludid[oa]|"
    r"invocad[oa]|en\s+comento)")
# «LA LEY DE LA MATERIA» NO ES LA ÚLTIMA NOMBRADA: es la que rige el asunto
# —en un amparo, la de Amparo—, y el escrito del 93/2026 la usa justo después
# de citar la Constitución. Sin saber cuál es, no se atribuye.
_RX_LEY_ANAFORICA = re.compile(
    r"^(?:ley|codigo|ordenamiento)\s+(?:en\s+cita|citad[oa]|referid[oa]|aludid[oa]|"
    r"invocad[oa]|en\s+comento)")
_RX_LEY_INDETERMINADA = re.compile(
    r"^(?:ley|codigo|ordenamiento)\s+(?:de\s+la\s+materia|aplicable|relativ[oa]|"
    r"respectiv[oa]|correspondiente)")
_RX_ART_CLAVE = re.compile(r"\b(?:articulos?|arts?\.|numeral(?:es)?|preceptos?|ordinal(?:es)?)\s+(?=\d)")
# «210-A», «58-2», «17 bis»: el sufijo es parte del número del artículo.
_RX_ART_NUM = re.compile(
    r"(?<![\w/.,])(\d{1,4}(?:-(?:\d{1,2}|[a-z])(?![\w-]))?)(?:\s*[oº°](?![a-z]))?"
    r"(?:\s*-?\s*(bis|ter|quater)\b)?(?![\w/])")
_RX_NO_ART = re.compile(
    r"(?:fraccion(?:es)?|parrafos?|incisos?|numeral(?:es)?|apartados?|puntos?|bases?|"
    r"tomo|pagina|fojas?|p\.)\s*$")


def _ley_en(s: str):
    """La ley que empieza en `s` (plano): (clave, fin) o (None, 0)."""
    for rx, k in _LEYES:
        m = rx.match(s)
        if m:
            return k, m.end()
    m = _RX_LEY_GENERICA.match(s)
    if m:
        a = _RX_LEY_ANAFORICA.match(s)
        if a:
            return "ANAFORA", a.end()
        a = _RX_LEY_INDETERMINADA.match(s)
        if a:
            return "INDETERMINADA", a.end()
        sig, pos, fin = [], m.end(), m.end()
        while len(sig) < 3:
            w = _RX_PALABRA_LEY.match(s, pos)
            if not w or w.group(1) in _CORTE_LEY:
                break
            pos = w.end()
            if w.group(1) not in _STOP_LEY:
                sig.append(w.group(1))
                fin = pos
        if sig:
            return f"{m.group(1)} {' '.join(sig)}", fin
    return None, 0


# Una ley que llega a más de esto del último número ya no es la de la lista:
# es la de otra frase («el artículo 17 establece que… en términos de la Ley
# Orgánica…»).
DISTANCIA_LEY = 90
_RX_CORTE_VENTANA = re.compile(r"\.\s+[A-ZÁÉÍÓÚÑ¿«\"“]|;|\n")


def preceptos(texto: str, pl: str = None) -> list:
    """[(pos, fin, 'art 16 cpeum')] — cada artículo de la lista con su ley."""
    texto = texto or ""
    pl = plano(texto) if pl is None else pl
    fuera, ultima, hasta = [], None, -1
    for m in _RX_ART_CLAVE.finditer(pl):
        if m.start() < hasta:
            continue
        # El corte de frase se mira en el ORIGINAL: en el plano todo es
        # minúscula y «1o. y 17» parecería un punto y seguido.
        orig = texto[m.end():m.end() + 260]
        corte = _RX_CORTE_VENTANA.search(orig)
        ventana = pl[m.end():m.end() + (corte.start() if corte else 260)]
        nums, ley, fin_ley, fin_num = [], None, 0, 0
        pos = 0
        while pos < len(ventana):
            if nums and pos - fin_num > DISTANCIA_LEY:
                break
            limite = pos == 0 or not ventana[pos - 1].isalnum()
            mn = _RX_ART_NUM.match(ventana, pos) if limite else None
            if mn and not _RX_NO_ART.search(ventana[max(0, pos - 16):pos]):
                nums.append(mn.group(1) + (f" {mn.group(2)}" if mn.group(2) else ""))
                pos = fin_num = mn.end()
                continue
            if limite and ventana[pos].isalpha():
                if nums:
                    a = _RX_ANAFORA.match(ventana, pos)
                    if a:
                        ley, fin_ley = "ANAFORA", a.end()
                        break
                k, e = _ley_en(ventana[pos:])
                if k:
                    ley, fin_ley = k, pos + e
                    break
            pos += 1
        if not nums or not ley:
            continue
        if ley == "INDETERMINADA":
            hasta = m.end() + fin_ley
            continue
        if ley == "ANAFORA":
            if not ultima:
                continue
            ley = ultima
        ultima = ley
        hasta = m.end() + fin_ley
        for n in nums:
            fuera.append((m.start(), m.end() + fin_ley, f"art {n} {ley}"))
    return fuera


def anclas(texto: str) -> list:
    """Todas las anclas duras con su posición: [(pos, fin, clave)].

    LOS SALTOS DE LÍNEA SE VUELVEN ESPACIOS (misma longitud, mismas
    posiciones). El escrito llega del PDF con un salto a media frase —«Ley
    Federal de⏎Procedimiento Contencioso…»— y la ley se partía en tres leyes
    distintas; medido en la demanda del 93/2026.
    """
    texto = (texto or "").replace("\n", " ")
    pl = plano(texto)
    fs_ = fechas(texto, pl)
    return (preceptos(texto, pl) + registros(texto, pl) + fs_ + cifras(texto, pl)
            + expedientes(texto, pl, tapar=fs_))


# LOS DATOS DEL PROPIO ASUNTO NO SON ANCLAS. El encabezado del escrito
# —expediente, toca, fecha de la sentencia reclamada— aparece en cualquier
# sentencia del asunto, conteste o no; contarlo inflaría la cobertura de todas
# las variantes por igual y taparía justo la diferencia que se busca.
ENCABEZADO_ESCRITO = 1500


def anclas_del_escrito(escrito: str) -> set:
    todas = anclas(escrito)
    propias = {k for a, _, k in todas if a < ENCABEZADO_ESCRITO}
    return {k for a, _, k in todas if a >= ENCABEZADO_ESCRITO} - propias


def tipo_de_ancla(k: str) -> str:
    return k.split(" ", 1)[0]


# ═══════════════════════════════════════════════════════════════════════════
# CITAS DE TESIS
# ═══════════════════════════════════════════════════════════════════════════
_RX_CLAVE_TESIS = re.compile(
    r"\b(?:p|1a|2a)\.\s*/\s*j\.\s*\d+/\d{4}|"               # 1a./J. 104/2008
    r"\b(?:p|1a|2a)\.\s*[clxvi]+/\d{4}|"                     # 1a. CCXLI/2014
    r"\b[ivxl]+\.\s*\d+o\.\s*[a-z](?:\.[a-z])*\.\s*(?:j/\d+|\d+\s*[a-z]\b)|"  # I.3o.C. 12 K
    r"\bpc\.[ivxl]+\.[a-z.]*\s*j/\d+\s*[a-z]*")              # PC.XXII. J/3 A
_RX_RUBRO = re.compile(r"[«“\"]\s*([A-ZÁÉÍÓÚÑ][A-ZÁÉÍÓÚÑ0-9 ,;:()/-]{30,})")
CERCA_CITA = 260


def _clave_rubro(s: str) -> str:
    return re.sub(r"[^a-z]", "", plano(s))[:60]


# CITAR NO ES MENCIONAR. Calibrado sobre los Kingston (26-sep-2026): el ADC
# 43/2025 nombra cuatro veces la jurisprudencia 1a./J. 61/2020 porque la
# DISCUTE —es la que aplicó la responsable—, y contarlo como «la misma tesis
# citada cuatro veces» acusaba al engrose de un defecto que no tiene. Es cita
# de autoridad la que lleva registro o rubro, o la que introduce una fórmula
# de apoyo; la clave suelta dentro de la prosa es una mención.
_RX_INTRO_CITA = re.compile(
    r"(?:sirve[n]?\s+de\s+apoyo|(?:resulta|es|son)\s+aplicables?|tiene[n]?\s+aplicaci[óo]n|"
    r"cobra[n]?\s+aplicaci[óo]n|apoya[n]?\s+(?:lo\s+anterior|esta|la)|sustenta[n]?\s+"
    r"(?:lo\s+anterior|esta|la)|en\s+(?:igual|similar|el\s+mismo)\s+sentido|ilustra|"
    r"orienta|robustece|de\s+rubro|cuyo\s+rubro|rubro\s+y\s+texto)[^.]{0,200}$")
# LA TRANSCRIPCIÓN NO ES OTRA CITA. En el ADC 810/2025 el registro 2024088 se
# anuncia antes del rubro y vuelve a aparecer en la ficha que cierra la tesis
# transcrita; entre una y otra no hay prosa del tribunal. Dos apariciones de la
# misma tesis con menos de esto de prosa propia en medio son una sola cita.
PROSA_ENTRE_CITAS = 40


def _prosa_propia(t: str) -> int:
    """Palabras del redactor en `t`: sin comillas, sin versales, sin fichas."""
    t = _RX_VERSALES.sub(" ", _RX_CITA_LARGA.sub(" ", t or ""))
    return sum(palabras(ln) for ln in t.split("\n") if palabras(ln) > 12)


def citas(texto: str) -> dict:
    """Cada tesis citada, reconocida por su registro, su rubro o su clave.

    UNA TESIS SE CITA A VECES SÓLO POR EL RUBRO la segunda vez. gen_642 cita la
    176546 con registro en la línea 79 y la vuelve a traer por el rubro, sin
    registro, veinte párrafos después (w2_final §1.1): contar sólo registros
    no la vería repetida. El rubro que aparece junto a un registro queda
    asociado a él, y la mención suelta del mismo rubro cuenta como la misma
    tesis.
    """
    texto = texto or ""
    pl = plano(texto)
    items = [(p, "reg", k.split()[1]) for p, _, k in registros(texto, pl)]
    items += [(m.start(), "clave", re.sub(r"\s+", "", m.group(0)))
              for m in _RX_CLAVE_TESIS.finditer(pl)]
    items += [(m.start(), "rubro", _clave_rubro(m.group(1)))
              for m in _RX_RUBRO.finditer(texto or "")]
    items.sort()
    grupos, actual = [], []
    for it in items:
        if actual and it[0] - actual[-1][0] > CERCA_CITA:
            grupos.append(actual)
            actual = []
        actual.append(it)
    if actual:
        grupos.append(actual)
    # Primera pasada: qué rubro y qué clave van con qué registro.
    a_reg = {}
    for g in grupos:
        regs = [v for _, t, v in g if t == "reg"]
        if len(regs) == 1:
            for _, t, v in g:
                if t in ("rubro", "clave"):
                    a_reg.setdefault((t, v), regs[0])
    apariciones, menciones = [], 0     # (pos, fin, identidad)
    for g in grupos:
        ini, fin = g[0][0], g[-1][0]
        tipos = {t for _, t, _ in g}
        autoridad = ("reg" in tipos or "rubro" in tipos
                     or bool(_RX_INTRO_CITA.search(pl[max(0, ini - 250):ini])))
        if not autoridad:
            menciones += 1
            continue
        regs = [v for _, t, v in g if t == "reg"]
        if regs:
            for r in dict.fromkeys(regs):
                apariciones.append((ini, fin, f"reg {r}"))
            continue
        otros = [(t, v) for _, t, v in g if t in ("rubro", "clave")]
        mapeados = [a_reg[x] for x in otros if x in a_reg]
        if mapeados:
            apariciones.append((ini, fin, f"reg {mapeados[0]}"))
        elif otros:
            t, v = next(((t, v) for t, v in otros if t == "clave"), otros[0])
            apariciones.append((ini, fin, f"{t} {v}"))
    ids, ultima = [], {}
    for ini, fin, ident in apariciones:
        if ident in ultima and _prosa_propia(texto[ultima[ident]:ini]) < PROSA_ENTRE_CITAS:
            ultima[ident] = fin          # la ficha de la misma transcripción
            continue
        ids.append(ident)
        ultima[ident] = fin
    cuenta = Counter(ids)
    return {"citas": len(ids), "distintas": len(cuenta), "menciones": menciones,
            "con_registro": sum(1 for i in ids if i.startswith("reg ")),
            "repetidas": sum(c - 1 for c in cuenta.values() if c > 1),
            "lista_repetidas": sorted(k for k, c in cuenta.items() if c > 1)}


# ═══════════════════════════════════════════════════════════════════════════
# CONCEPTOS NOMBRADOS, CALIFICACIONES Y APARTADOS
# ═══════════════════════════════════════════════════════════════════════════
_RX_UNICO = re.compile(
    r"\b[úu]nico\s+(?:concepto|agravio)|\b(?:concepto|agravio)(?:\s+de\s+violaci[óo]n)?"
    r"\s+[úu]nico", re.I)
_RX_CALIF = re.compile(
    r"\b(?:fundad[oa]s?|infundad[oa]s?|inoperantes?|ineficaz|ineficaces|"
    r"inatendibles?|innecesari[oa]s?|insuficientes?|sin\s+materia|"
    r"desestim[a-z]*)\b", re.I)
# «debidamente fundada y motivada» no califica ningún concepto.
_RX_NO_CALIF = re.compile(
    r"(?:debida|indebida|suficiente)mente\s+fundad\w*|fundad\w*\s+y\s+motivad\w*|"
    r"motivad\w*\s+y\s+fundad\w*", re.I)


# «TERCERO INTERESADO» Y «POR CONCEPTO DE» NO NOMBRAN NINGÚN CONCEPTO. En el
# ADC 526/2024 «depósitos realizados por el tercero interesado por concepto de
# alimentos» abría un apartado del «tercer concepto» que no existe. Se limpian
# antes de buscar ordinales. (`formato_sentencia.nombrados` —producción— tiene
# el mismo punto ciego; aquí no se toca, sólo se anota.)
_RX_FALSO_ORDINAL = re.compile(
    r"\btercer[oa]s?\s+(?:interesad|extra[nñ]|perjudicad|llamad)\w*|"
    r"\b(?:por|en|bajo\s+el)\s+concepto\s+de\b|"
    r"\ben\s+(?:primer|segundo|tercer|cuarto)\s+(?:lugar|t[ée]rmino)\b", re.I)


def ordinales(frase: str) -> set:
    frase = _RX_FALSO_ORDINAL.sub(" ", frase or "")
    s = fs.nombrados(frase)
    if _RX_UNICO.search(frase):
        s = s | {1}
    return s


def califica(frase: str) -> bool:
    return bool(_RX_CALIF.search(_RX_NO_CALIF.sub(" ", frase or "")))


def _primera_frase(p: str) -> str:
    fr = frases(p)
    return fr[0] if fr else p


_RX_PREGUNTA = re.compile(r"^\s*\d+\s*\.\s*¿")
_RX_METODO_ANUNCIO = re.compile(
    r"por\s+cuesti[óo]n\s+de\s+m[ée]todo|\bse\s+(?:analizar|estudiar|examinar|abordar)"
    r"[áa]n?\b|\ben\s+el\s+orden\s+(?:en\s+que|propuesto)|art[íi]culo\s+76\s+de\s+la\s+ley",
    re.I)
_RX_REMITE_A = re.compile(
    r"\bal\s+(?:examinar|analizar|estudiar|contestar|resolver|dar\s+respuesta)\b|"
    r"\bcomo\s+(?:ya\s+)?se\s+(?:dijo|precis|expus|se[nñ]al|mencion|advirti|indic|"
    r"estableci|determin|vio)\w*", re.I)
TOPE_APERTURA = 150


def apartados(solucion: str) -> list:
    """Dónde empieza cada apartado de la Solución y cuánto mide su apertura.

    Un apartado empieza donde una pregunta numerada abre (moderna) o donde la
    PRIMERA frase de un párrafo nombra un concepto que ningún apartado
    anterior había abierto (estándar y engroses: «Sobre el primer concepto…»,
    «Respecto al Primer y Segundo Conceptos…», «Finalmente, respecto al
    cuarto…»). No abre apartado:
      · el anuncio de método («se analizarán conjuntamente…»);
      · la remisión («como se precisó al examinar el primero…»);
      · volver a nombrar un concepto ya abierto («En cuanto al tercer
        concepto…, la conclusión es la misma», gen_642:85), que sigue dentro
        del apartado que lo agrupó.

    La APERTURA mide desde el inicio del apartado hasta la primera
    calificación (fila 4 de w2_final §4.6: 59-84 palabras frente a 27 del
    engrose). Si la calificación no llega en `TOPE_APERTURA` palabras, se toma
    la primera frase.
    """
    ps = parrafos(solucion)
    fuera, vistos = [], set()
    tras_pregunta = False
    for i, p in enumerate(ps):
        if _RX_PREGUNTA.match(p):
            fuera.append({"i": i, "tipo": "pregunta", "ordinales": set()})
            tras_pregunta = True
            continue
        f1 = _primera_frase(p)
        o = ordinales(f1)
        if tras_pregunta:
            # «Sí. El primer concepto de violación es fundado…»: la respuesta
            # abre con el monosílabo y nombra el concepto en la frase siguiente.
            o = ordinales(" ".join(frases(p)[:2]))
            # La respuesta a la pregunta es parte de su apartado.
            if fuera:
                fuera[-1]["ordinales"] |= o
                vistos |= o
            tras_pregunta = False
            continue
        if not o or _RX_METODO_ANUNCIO.search(f1) or _RX_REMITE_A.search(f1):
            continue
        if not (o - vistos):
            continue
        # Un rótulo suelto («Primer concepto de violación») y el párrafo que lo
        # desarrolla son un solo apartado.
        if fuera and fuera[-1]["i"] == i - 1 and fuera[-1].get("rotulo"):
            fuera[-1]["ordinales"] |= o
            vistos |= o
            continue
        fuera.append({"i": i, "tipo": "concepto", "ordinales": set(o),
                      "rotulo": palabras(p) <= 12 and not califica(p)})
        vistos |= o
    for k, a in enumerate(fuera):
        fin = fuera[k + 1]["i"] if k + 1 < len(fuera) else len(ps)
        a["fin"] = fin
        cuerpo = " ".join(ps[a["i"]:min(fin, a["i"] + 3)])
        m = None
        for mm in _RX_CALIF.finditer(cuerpo):
            if not _RX_NO_CALIF.search(cuerpo[max(0, mm.start() - 25):mm.end() + 25]):
                m = mm
                break
        # EL RESULTADO DEL APARTADO: su primera calificación, por familia. Lo
        # necesita la Decisión 3 (b) —el cierre sólo con resultados distintos—.
        a["resultado"] = familia_calificacion(m.group(0)) if m else None
        n_hasta = palabras(cuerpo[:m.end()]) if m else None
        if n_hasta is None or n_hasta > TOPE_APERTURA:
            if a["tipo"] == "pregunta" and a["i"] + 1 < len(ps):
                n_hasta = palabras(ps[a["i"]]) + palabras(_primera_frase(ps[a["i"] + 1]))
            else:
                n_hasta = palabras(_primera_frase(ps[a["i"]]))
        a["apertura"] = n_hasta
        a["texto"] = "\n".join(ps[a["i"]:fin])
    return fuera


_FAMILIAS_CALIF = (("infundad", "infundado"), ("fundad", "fundado"),
                   ("inoperant", "inoperante"), ("ineficac", "ineficaz"),
                   ("ineficaz", "ineficaz"), ("inatendibl", "inatendible"),
                   ("innecesari", "innecesario"), ("sin materia", "innecesario"),
                   ("insuficient", "insuficiente"), ("desestim", "desestimado"))


def familia_calificacion(palabra: str):
    """«infundados» → «infundado»; «sin materia» → «innecesario»."""
    t = re.sub(r"\s+", " ", plano(palabra or ""))
    return next((f for raiz, f in _FAMILIAS_CALIF if t.startswith(raiz)), None)


def resultados_distintos(aps: list):
    """¿Los apartados llegan a resultados distintos? (Decisión 3 b)

    True si hay al menos dos familias de calificación entre los apartados, o
    si alguno no deja leer la suya: ante la duda no se acusa —el ADA 400/2024
    abre dos de sus tres apartados sin calificar en las primeras líneas, y es
    un engrose bueno—. False sólo cuando TODOS dicen lo mismo."""
    if not aps:
        return None
    res = [a.get("resultado") for a in aps]
    if any(r is None for r in res):
        return True
    return len(set(res)) >= 2


def calificaciones_por_concepto(solucion: str) -> dict:
    """Cuántas frases califican a cada concepto nombrándolo (V8)."""
    c = Counter()
    for f in frases(solucion):
        if califica(f):
            for o in ordinales(f):
                c[o] += 1
    return dict(sorted(c.items()))


# ═══════════════════════════════════════════════════════════════════════════
# PRECEPTOS EXPUESTOS EN MÁS DE UN APARTADO
# ═══════════════════════════════════════════════════════════════════════════
_RX_EXPONE = re.compile(
    r"\b(?:establecen?|disponen?|preven?|precept[uú]a|estatuye|se[nñ]alan?|"
    r"ordena|contempla|consagra|reconocen?|regula|exigen?|impone|obliga|faculta|"
    r"permite|determina|prescribe|deriva|se\s+desprende|se\s+advierte|"
    r"de\s+(?:dich[oa]s?|es[oa]s?|tal(?:es)?)\s+(?:numeral|precepto|disposici[óo]n|"
    r"art[íi]culo)s?)\b", re.I)


def preceptos_por_apartado(solucion: str, aps: list) -> dict:
    """{'art 16 cpeum': {'expuesto': {0, 2}, 'mencion': {0, 1, 2}}}

    El segmento 0 es lo que va antes del primer apartado (la calificación
    general y el anuncio de método); los apartados cuentan desde 1.
    """
    ps = parrafos(solucion)
    cortes = [0] + [a["i"] for a in aps] + [len(ps)]
    segs = []
    for k in range(len(cortes) - 1):
        segs.append("\n".join(ps[cortes[k]:cortes[k + 1]]))
    if aps and aps[0]["i"] == 0:
        segs = segs[1:]
        base = 1
    else:
        base = 0
    fuera = defaultdict(lambda: {"expuesto": set(), "mencion": set()})
    for k, seg in enumerate(segs):
        for f in frases(seg):
            ks = {x[2] for x in preceptos(f)}
            if not ks:
                continue
            expone = bool(_RX_EXPONE.search(f))
            for x in ks:
                fuera[x]["mencion"].add(k + base)
                if expone:
                    fuera[x]["expuesto"].add(k + base)
    return dict(fuera)


# ═══════════════════════════════════════════════════════════════════════════
# MARCADORES DE FORMA
# ═══════════════════════════════════════════════════════════════════════════
_RX_OBJECION = re.compile(
    r"\bno\s+pasa\s+inadvertid[oa]|\bno\s+se\s+desconoce\b|\bno\s+obsta\b|"
    r"\bno\s+(?:es|resulta|constituye)\s+[óo]bice\b|\bno\s+se\s+pierde\s+de\s+vista\b|"
    r"\bno\s+escapa\b|\bsin\s+que\s+obste\b|(?:^|(?<=[.;:]\s))tampoco\b", re.I | re.M)
_RX_RECAPITULA = re.compile(
    r"(?:^|(?<=[.;:]\s))(?:en\s+suma|en\s+s[íi]ntesis|por\s+todo\s+lo\s+anterior|"
    r"en\s+consecuencia,?\s+al\s+resultar|en\s+conclusi[óo]n|como\s+conclusi[óo]n|"
    r"recapitulando)\b", re.I | re.M)
_RX_METODO = re.compile(r"por\s+cuesti[óo]n\s+de\s+m[ée]todo", re.I)
# UNA ORDEN, NO UN SUBJUNTIVO. El ADA 702/2022 dice que la sentencia de nulidad
# «impide que una decisión… deje insubsistente uno de los efectos»: el verbo
# está, la orden a la responsable no. Cuenta sólo lo que manda: «para que la
# Sala deje insubsistente», «deberá dictar otra», «la responsable emita una
# nueva».
_VERBO_ORDEN = (r"(?:dej(?:e|ar)\s+(?:insubsistente|sin\s+efectos?)|"
                r"dict(?:e|ar)\s+(?:otra|una\s+nueva)|emit(?:a|ir)\s+(?:otra|una\s+nueva))")
_RX_ORDEN_RESPONSABLE = re.compile(
    r"\bpara\s+(?:el\s+(?:solo\s+)?efecto\s+de\s+)?que\s+(?:[\wáéíóúñ]+\s+){0,6}?" + _VERBO_ORDEN +
    r"|\bdeber[áa]n?\s+(?:[\wáéíóúñ]+\s+){0,2}?" + _VERBO_ORDEN +
    r"|\b(?:la|el)\s+(?:responsable|sala|autoridad(?:\s+responsable)?|juez[a]?|tribunal)\s+"
    + _VERBO_ORDEN, re.I)

# ── REMISIONES ─────────────────────────────────────────────────────────────
# PD AD 19/2024 ¶107 y JRC 7/2014 ¶160-164 remiten con destino y con la
# proposición que decide; GOM 71-2014 ¶371 («han quedado respondidos…») es la
# remisión en blanco (w2_final §2.2). Aquí: destino = se nombra a dónde
# (concepto por su ordinal, apartado, considerando, «al examinar…»);
# en blanco = sin destino y sin proposición propia en la frase.
_RX_REMISION = re.compile(
    r"como\s+(?:ya\s+)?se\s+(?:dijo|precis[óo]|expuso|se[nñ]al[óo]|mencion[óo]|"
    r"advirti[óo]|indic[óo]|estableci[óo]|determin[óo]|vio|analiz[óo]|razon[óo]|"
    r"ha\s+(?:dicho|expuesto|precisado|visto|se[nñ]alado))|"
    r"seg[úu]n\s+se\s+(?:ha\s+)?(?:expuesto|dicho|precisado)|"
    r"por\s+(?:las|los)\s+(?:mismas?\s+|mismos\s+)?(?:razones|motivos|consideraciones)"
    r"(?:\s+(?:antes|ya))?\s+(?:expuest|precisad|anotad|apuntad)[oa]s|"
    r"por\s+(?:las|los)\s+mism[oa]s\s+(?:razones|motivos|consideraciones)|"
    r"en\s+las\s+relatadas\s+consideraciones|\bse\s+reitera\b|"
    r"en\s+los\s+t[ée]rminos\s+(?:antes\s+)?(?:precisados|expuestos)|"
    r"(?:ya\s+)?qued[óo]\s+(?:precisado|demostrado|establecido|evidenciado|dicho)|"
    r"l[íi]neas\s+arriba|p[áa]rrafos\s+(?:precedentes|anteriores)|"
    r"corre[n]?\s+la\s+misma\s+suerte|la\s+conclusi[óo]n\s+es\s+la\s+misma|"
    r"(?:recibe|merece)[n]?\s+(?:la\s+misma|id[ée]ntica)\s+respuesta|"
    r"la\s+respuesta\s+(?:ya\s+)?(?:est[áa]|qued[óo])\s+dada|"
    r"\bal\s+(?:examinar|analizar|estudiar|contestar)\s+(?:el|la|los|las)\s+\w+", re.I)
# EL DESTINO ES A DÓNDE SE REMITE, NO EL CONCEPTO QUE SE CALIFICA. «Por las
# razones expuestas, el tercer concepto es inoperante» nombra un ordinal, pero
# es el del concepto que se despacha: la remisión sigue en blanco. El destino
# se busca pegado al marcador —«al examinar el primero», «en el apartado
# anterior», «del considerando quinto»— y no en cualquier parte de la frase.
_RX_DESTINO = re.compile(
    r"\bal\s+(?:examinar|analizar|estudiar|contestar|resolver|dar\s+respuesta\s+a)\s+"
    r"(?:el|la|los|las)\s+\w+|"
    r"\b(?:en|de|del)\s+(?:el\s+|los\s+)?(?:apartado|considerando|p[áa]rrafo|punto)s?\s+"
    r"(?:anterior|precedente|que\s+antecede|[ivxl\d]+|primero|segundo|tercero|cuarto|"
    r"quinto|sexto|s[ée]ptimo|octavo)\b|"
    r"\b(?:respecto\s+(?:al|del)|en\s+relaci[óo]n\s+con\s+el|en\s+cuanto\s+al|"
    r"sobre\s+el)\s+(?:primer|segund|tercer|cuart|quint|sext|s[ée]ptim|octav|noven|"
    r"d[ée]cim)[oa]?s?\b", re.I)
VENTANA_DESTINO = 70
MIN_CONTENIDO = 15


def remisiones(solucion: str) -> dict:
    total = destino = blanco = 0
    for f in frases(solucion):
        m = _RX_REMISION.search(f)
        if not m:
            continue
        total += 1
        cerca = f[max(0, m.start() - 10):m.end() + VENTANA_DESTINO]
        if _RX_DESTINO.search(m.group(0)) or _RX_DESTINO.search(cerca):
            destino += 1
        elif palabras(f) - palabras(m.group(0)) < MIN_CONTENIDO:
            blanco += 1
    return {"remisiones": total, "con_destino": destino, "en_blanco": blanco}


# ═══════════════════════════════════════════════════════════════════════════
# LA MEDIDA ENTERA
# ═══════════════════════════════════════════════════════════════════════════
def medir(texto: str, escrito: str = None, n: int = None) -> dict:
    """Todas las medidas sobre un proyecto (o un engrose).

    `escrito` es la demanda o el recurso: con él se cuentan los planteamientos
    (contador de producción) y se miden las anclas. `n` fuerza la cuenta.
    Devuelve las cifras planas y, bajo `detalle`, los conjuntos que el
    comparador necesita para ver QUÉ se perdió, no sólo cuánto.
    """
    sec = secciones(texto)
    sol = sec["solucion"]
    pal_sol = palabras(sol)
    aps = apartados(sol)
    ps = [p for p in parrafos(sol) if palabras(p) >= 8 and not es_transcripcion(p)]
    lp = [palabras(p) for p in ps]
    fr = frases(sol)
    lf = [palabras(f) for f in fr]
    cit = citas(sol)
    pre = preceptos_por_apartado(sol, aps)
    reexp = sorted(k for k, v in pre.items() if len(v["expuesto"]) >= 2)
    rep = sorted(k for k, v in pre.items() if len(v["mencion"]) >= 2)
    calif = calificaciones_por_concepto(sol)
    rem = remisiones(sol)
    ps_todos = parrafos(sol)
    cola = "\n".join([p for p in ps_todos if palabras(p) >= 8][-2:])
    x1000 = (lambda v: round(1000 * v / pal_sol, 2) if pal_sol else None)

    if n is None and escrito:
        try:
            n = int(cp.planteamientos(escrito).get("n") or 0)
        except Exception:
            n = None
    faltan = fs.sin_contestar(sol, n) if n else []
    # Para ver QUÉ concepto deja de nombrarse se usa el reconocedor limpio
    # (sin «tercero interesado» ni «por concepto de»); la cobertura por ordinal
    # de arriba es la de producción tal cual, para que la cifra sea la misma
    # que el aviso que ve el secretario.
    nombr = sorted(x for x in ordinales(sol) if not n or x <= n)

    a_esc = anclas_del_escrito(escrito) if escrito else set()
    a_sol = {k for _, _, k in anclas(sol)}
    cubiertas = a_esc & a_sol
    por_tipo = {}
    for t in ("art", "reg", "exp", "cifra", "fecha"):
        e = {k for k in a_esc if tipo_de_ancla(k) == t}
        if e:
            por_tipo[t] = round(len(e & a_sol) / len(e), 3)

    aperturas = [a["apertura"] for a in aps]
    return {
        "solucion_rotulada": sec["solucion_rotulada"],
        "palabras_solucion": pal_sol,
        "palabras_estudio": palabras(sec["estudio"]),
        "palabras_resumenes": palabras(sec["sentencia_reclamada"]) + palabras(sec["conceptos"]),
        "apartados": len(aps),
        "apertura_mediana": mediana(aperturas),
        "apertura_max": max(aperturas) if aperturas else None,
        "parrafos": len(ps),
        "parrafo_media": round(statistics.mean(lp), 1) if lp else None,
        "parrafo_p90": round(_pct(lp, 0.9), 1) if lp else None,
        "frases": len(fr),
        "frases_largas": sum(1 for x in lf if x > 60),
        "frases_largas_pct": round(sum(1 for x in lf if x > 60) / len(lf), 3) if lf else None,
        "citas": cit["citas"],
        "citas_distintas": cit["distintas"],
        "citas_repetidas": cit["repetidas"],
        "citas_con_registro": cit["con_registro"],
        "menciones_de_tesis": cit["menciones"],
        "preceptos_reexpuestos": len(reexp),
        "preceptos_repetidos": len(rep),
        "calif_max_concepto": max(calif.values()) if calif else 0,
        "conceptos_calif_mas_2": sum(1 for v in calif.values() if v > 2),
        "objeciones": len(_RX_OBJECION.findall(sol)),
        "objeciones_x1000": x1000(len(_RX_OBJECION.findall(sol))),
        "recapitulaciones": len(_RX_RECAPITULA.findall(sol)),
        "recapitulacion_final": int(bool(_RX_RECAPITULA.search(cola))),
        "resultados_distintos": resultados_distintos(aps),
        "metodo_molde": len(_RX_METODO.findall(sol)),
        "ordenes_fuera_efectos": len(_RX_ORDEN_RESPONSABLE.findall(sol)),
        "remisiones": rem["remisiones"],
        "remisiones_x1000": x1000(rem["remisiones"]),
        "remisiones_con_destino": rem["con_destino"],
        "remisiones_en_blanco": rem["en_blanco"],
        "remisiones_con_contenido_pct": (
            round((rem["remisiones"] - rem["en_blanco"]) / rem["remisiones"], 3)
            if rem["remisiones"] else None),
        "n_planteamientos": n,
        "ordinales_sin_nombrar": len(faltan),
        "cobertura_ordinal": (round((n - len(faltan)) / n, 3) if n and n >= 2 else None),
        "anclas_escrito": len(a_esc),
        "anclas_cubiertas": len(cubiertas),
        "cobertura_demanda": round(len(cubiertas) / len(a_esc), 3) if a_esc else None,
        "detalle": {
            "aperturas": aperturas,
            "resultados_por_apartado": [a.get("resultado") for a in aps],
            "ordinales_por_apartado": [sorted(a["ordinales"]) for a in aps],
            "calificaciones_por_concepto": calif,
            "citas_repetidas": cit["lista_repetidas"],
            "preceptos_reexpuestos": reexp,
            "ordinales_nombrados": nombr,
            "ordinales_faltan": faltan,
            "anclas_cubiertas": sorted(cubiertas),
            "anclas_faltan": sorted(a_esc - a_sol),
            "cobertura_por_tipo": por_tipo,
        },
    }


# ═══════════════════════════════════════════════════════════════════════════
# QUÉ ES MEJOR EN CADA MEDIDA, Y CUÁNDO ACUSA
# ═══════════════════════════════════════════════════════════════════════════
# `mejor`: «menos» o «mas» (dirección buena) o None (informativa).
# `acusa`: el umbral FIJO que la propuesta aprobada pone a esa medida como
# alarma (w2_final §4.4 y §6.3). Las que no lo tienen se comparan contra la
# banda del oro y no se juzgan solas.
METRICAS = [
    # clave, nombre, mejor, umbral (texto), acusa(v, fila)
    ("palabras_solucion", "Palabras de la Solución", "menos", None, None),
    ("apartados", "Apartados", None, None, None),
    ("apertura_mediana", "Apertura por apartado (mediana, palabras)", "menos",
     "> 45 (§6.3 claridad)", lambda v, f: v is not None and v > 45),
    ("apertura_max", "Apertura más larga (palabras)", "menos", None, None),
    ("parrafo_media", "Palabras por párrafo (media)", "menos", None, None),
    ("parrafo_p90", "Palabras por párrafo (p90)", "menos", None, None),
    ("frases_largas_pct", "Frases de más de 60 palabras (proporción)", "menos", None, None),
    ("citas", "Citas de tesis", None, None, None),
    ("citas_repetidas", "Tesis citadas más de una vez", "menos",
     "> 0 (V4 a)", lambda v, f: bool(v)),
    ("preceptos_reexpuestos", "Precepto expuesto como premisa en ≥2 apartados", "menos",
     "> 0 (V4 b)", lambda v, f: bool(v)),
    ("preceptos_repetidos", "Precepto mencionado en ≥2 apartados", None, None, None),
    ("calif_max_concepto", "Calificaciones del mismo concepto (máx.)", "menos",
     "> 2 (V8, M2d)", lambda v, f: v is not None and v > 2),
    ("objeciones_x1000", "Marcadores de objeción por 1000 palabras", "menos", None, None),
    # Decisión 3 (b) de David, 26-sep-2026: «sin cierre por defecto; un cierre
    # breve sólo cuando hay tres o más apartados con resultados distintos».
    # LAS DOS CONDICIONES, no sólo la cuenta (revisión, 26-sep-2026): tres
    # apartados que concluyen todos «infundado» no justifican el cierre. Con el
    # resultado ilegible no se acusa (ver `resultados_distintos`). Calibrado: el
    # oro sigue acusado en 1 de 24 (ADC 529/2024, un solo apartado); el ADA
    # 103/2025 cierra con infundado/inoperante/ineficaz y no se acusa. «Breve»
    # no se mide: no hay número que lo defina.
    ("recapitulacion_final", "Recapitulación al final", "menos",
     "presente sin 3+ apartados de resultado distinto (Decisión 3 b)",
     lambda v, f: bool(v) and not ((f.get("apartados") or 0) >= 3
                                   and f.get("resultados_distintos"))),
    ("metodo_molde", "Molde «Por cuestión de método»", "menos",
     "> 0 (C8)", lambda v, f: bool(v)),
    ("ordenes_fuera_efectos", "Órdenes a la responsable fuera de Efectos", "menos",
     "> 0 (V6)", lambda v, f: bool(v)),
    ("remisiones_x1000", "Remisiones por 1000 palabras", None, None, None),
    ("remisiones_en_blanco", "Remisiones en blanco", "menos",
     "> 0 (§6.3 claridad)", lambda v, f: bool(v)),
    ("remisiones_con_contenido_pct", "Remisiones con contenido (proporción)", "mas",
     "< 0.90 (§6.3 claridad)", lambda v, f: v is not None and v < 0.9),
    ("cobertura_ordinal", "Cobertura por ordinal", "mas",
     "< 1 (sin_contestar)", lambda v, f: v is not None and v < 1),
    ("cobertura_demanda", "Cobertura de anclas del escrito", "mas", None, None),
]
CLAVES = [m[0] for m in METRICAS]
MEJOR = {m[0]: m[2] for m in METRICAS}
NOMBRE = {m[0]: m[1] for m in METRICAS}
# Más del 10 % de engroses buenos acusados = la alarma está mal (w2 §6.1, 6).
TOPE_ACUSADOS = 0.10


# ═══════════════════════════════════════════════════════════════════════════
# CALIBRACIÓN CONTRA LOS ENGROSES BUENOS
# ═══════════════════════════════════════════════════════════════════════════
def _huella_texto(t: str) -> str:
    return hashlib.sha1(re.sub(r"\s+", " ", t or "").strip().encode()).hexdigest()[:12]


def filas_oro() -> list:
    """Los 24 Kingston de oro sólido (la regla de `banco_kingston.banco()`)."""
    if not CASOS_KINGSTON.exists():
        return []
    import banco_kingston as bk
    fuera = []
    for c in bk.banco():
        dem = ((c.get("piezas") or {}).get("demanda") or {}).get("texto", "")
        fuera.append({"grupo": "oro", "caso": c["asunto"], "texto": c["oro"],
                      "escrito": dem, "huella": _huella_texto(c["oro"])})
    return fuera


def filas_diag(diag: Path = DIAG, dir93: Path = DIR93) -> list:
    """Los oros sueltos del diagnóstico y las salidas «v1» ya generadas."""
    fuera = []
    for f in sorted(diag.glob("oro_*.txt")) if diag.exists() else []:
        num = f.stem[4:]
        esc = diag / f"escrito_{num}.txt"
        t = f.read_text(encoding="utf-8")
        fuera.append({"grupo": "oro_diag", "caso": num, "texto": t,
                      "escrito": esc.read_text(encoding="utf-8") if esc.exists() else "",
                      "huella": _huella_texto(t)})
    for f in sorted(diag.glob("gen_*.txt")) if diag.exists() else []:
        if f.stem.endswith(("_estudio_crudo", "_stream")):
            continue
        num = f.stem[4:]
        esc = diag / f"escrito_{num}.txt"
        fuera.append({"grupo": "v1", "caso": f"gen {num}",
                      "texto": f.read_text(encoding="utf-8"),
                      "escrito": esc.read_text(encoding="utf-8") if esc.exists() else ""})
    esc93 = dir93 / "fuente1.txt"
    for nombre in ("estandar2.txt", "moderna4.txt"):
        f = dir93 / nombre
        if f.exists():
            fuera.append({"grupo": "v1", "caso": f"93/2026 {f.stem}",
                          "texto": f.read_text(encoding="utf-8"),
                          "escrito": esc93.read_text(encoding="utf-8") if esc93.exists() else ""})
    return fuera


def _fmt(v, nd=2):
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.{nd}f}".rstrip("0").rstrip(".") if nd else f"{v:.0f}"
    return str(v)


def referencia(filas_medidas: list) -> dict:
    """Mediana, p10, p90, mín. y máx. de cada medida en los oros (sin duplicados)."""
    oros = [f for f in filas_medidas if f["grupo"] == "oro"]
    ref = {}
    for k in CLAVES:
        xs = [f["m"][k] for f in oros if f["m"].get(k) is not None]
        xs = [float(x) for x in xs]
        ref[k] = {"n": len(xs), "mediana": mediana(xs), "p10": _pct(xs, 0.1),
                  "p90": _pct(xs, 0.9), "min": min(xs) if xs else None,
                  "max": max(xs) if xs else None}
    rot = [f["m"]["palabras_solucion"] for f in oros if f["m"]["solucion_rotulada"]]
    ref["_palabras_solucion_rotulada"] = {
        "n": len(rot), "mediana": mediana(rot), "p75": _pct(rot, 0.75),
        "p90": _pct(rot, 0.9)}
    return ref


def calibrar(salida: Path = SALIDA, diag: Path = DIAG, dir93: Path = DIR93) -> dict:
    filas = filas_oro()
    vistas = {f["huella"] for f in filas}
    for f in filas_diag(diag, dir93):
        if f["grupo"] == "oro_diag" and f["huella"] in vistas:
            f["duplicado"] = True
        filas.append(f)
    for f in filas:
        f["m"] = medir(f["texto"], f.get("escrito") or None)
    ref = referencia(filas)
    salida.mkdir(parents=True, exist_ok=True)
    (salida / "calibracion.json").write_text(json.dumps(
        {"referencia": ref,
         "filas": [{"grupo": f["grupo"], "caso": f["caso"],
                    "duplicado": f.get("duplicado", False), "m": f["m"]} for f in filas]},
        ensure_ascii=False, indent=1, default=list), encoding="utf-8")
    md = informe_calibracion(filas, ref)
    (salida / "calibracion.md").write_text(md, encoding="utf-8")
    return {"filas": filas, "referencia": ref, "informe": md}


def _acusados(filas: list, clave: str, acusa) -> tuple:
    ev = [f for f in filas if f["m"].get(clave) is not None]
    ac = [f for f in ev if acusa(f["m"][clave], f["m"])]
    frac = len(ac) / len(ev) if ev else 0.0
    return frac, (f"{len(ac)}/{len(ev)} ({frac:.0%})" if ev else "—")


def informe_calibracion(filas: list, ref: dict) -> str:
    oros = [f for f in filas if f["grupo"] == "oro"]
    v1 = [f for f in filas if f["grupo"] == "v1"]
    dup = [f for f in filas if f.get("duplicado")]
    L = ["# Calibración de las métricas del estudio contra los engroses buenos", "",
         f"Referencia: **{len(oros)} engroses Kingston de oro sólido** (regla de "
         "`banco_kingston.banco()`: más de 3 calificaciones). Ejemplos «v1» (salidas "
         f"del taller ya generadas): **{len(v1)}**.", ""]
    if dup:
        L += [f"Los oros sueltos del diagnóstico ({', '.join(f['caso'] for f in dup)}) son "
              "idénticos a los Kingston del mismo asunto y no se cuentan dos veces.", ""]
    L += ["Regla (w2_final §6.1, paso 6): una medida con umbral fijo que acusa a más del "
          f"{int(TOPE_ACUSADOS * 100)} % de los engroses buenos es **candidata a no usarse "
          "como alarma**. Sigue sirviendo para comparar variantes entre sí.", "",
          "## Distribución de referencia y veredicto", "",
          "| Métrica | Mejor | Oro mediana | Oro p10–p90 | Oro mín–máx | v1 mediana | Umbral "
          "| Oros acusados | v1 acusados | Veredicto |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for clave, nombre, mejor, umbral, acusa in METRICAS:
        r = ref[clave]
        v1m = mediana([f["m"][clave] for f in v1 if f["m"].get(clave) is not None])
        txt_v1 = "—"
        if acusa:
            frac, txt_acus = _acusados(oros, clave, acusa)
            frac_v1, txt_v1 = _acusados(v1, clave, acusa)
            veredicto = ("**CANDIDATA A NO USARSE** como alarma" if frac > TOPE_ACUSADOS
                         else "usable como alarma")
            # ¿SEPARA? En una alarma se compara cuántos documentos acusa: la
            # mediana de un recuento casi siempre en cero no dice nada.
            if v1 and frac_v1 <= frac:
                veredicto += f"; no acusa más a v1 que al oro (en {len(v1)} ejemplos)"
        elif clave.startswith("cobertura"):
            txt_acus = "—"
            veredicto = "sólo como regresión entre variantes"
        else:
            txt_acus = "—"
            veredicto = "comparativa (sin umbral fijo)"
            # Si lo generado no queda peor que el oro en la dirección que la
            # propuesta dice mala, la medida no ve el defecto que se busca.
            if mejor and r["mediana"] is not None and v1m is not None:
                peor = v1m > r["mediana"] if mejor == "menos" else v1m < r["mediana"]
                if not peor:
                    veredicto += "; **no separa v1 del oro**"
        L.append(f"| {nombre} | {mejor or '—'} | {_fmt(r['mediana'])} | "
                 f"{_fmt(r['p10'])}–{_fmt(r['p90'])} | {_fmt(r['min'])}–{_fmt(r['max'])} | "
                 f"{_fmt(v1m)} | {umbral or '—'} | {txt_acus} | {txt_v1} | {veredicto} |")
    r = ref["_palabras_solucion_rotulada"]
    L += ["", "## Techo de extensión de la Solución (peldaño B2)", "",
          f"Sólo engroses con la Solución rotulada ({r['n']}): mediana "
          f"**{_fmt(r['mediana'], 0)}** palabras, p75 {_fmt(r['p75'], 0)}, p90 "
          f"{_fmt(r['p90'], 0)}. Hoy el prompt pide 3,733 para la Solución "
          "(`fase6_estudio.PALABRAS_ESTUDIO`), que es la mediana del considerando entero "
          "con sus resúmenes (F1).", ""]
    # LA COBERTURA POR ANCLAS ES BAJA EN LOS BUENOS, y hay que decirlo antes de
    # que alguien la lea como «el engrose contesta el 11 %». El escrito cuenta
    # su historia procesal con fechas y precedentes que la Solución no tiene
    # por qué repetir; por eso esta medida sólo vale como REGRESIÓN entre
    # variantes (un ancla que A nombraba y B ya no), nunca como nivel.
    L += ["## Cobertura de anclas del escrito, por tipo", "",
          "Mediana de la proporción de anclas del escrito que la Solución nombra. Es "
          "baja también en los engroses: el escrito narra fechas y precedentes que la "
          "Solución no necesita repetir. **Sólo sirve como regresión entre variantes** "
          "(un ancla que una nombraba y la otra ya no), nunca como nivel absoluto.", "",
          "| Tipo | Oro mediana | v1 mediana | Oros con anclas de ese tipo |", "|---|---|---|---|"]
    for t, nom in (("art", "artículo con su ley"), ("reg", "registro de tesis"),
                   ("exp", "expediente"), ("cifra", "cifra"), ("fecha", "fecha")):
        xo = [f["m"]["detalle"]["cobertura_por_tipo"].get(t) for f in oros]
        xv = [f["m"]["detalle"]["cobertura_por_tipo"].get(t) for f in v1]
        L.append(f"| {nom} | {_fmt(mediana(xo))} | {_fmt(mediana(xv))} | "
                 f"{sum(1 for x in xo if x is not None)}/{len(oros)} |")
    L.append("")
    cols = ["palabras_solucion", "apartados", "apertura_mediana", "parrafo_p90",
            "frases_largas_pct", "citas_repetidas", "preceptos_reexpuestos",
            "calif_max_concepto", "recapitulacion_final", "metodo_molde",
            "ordenes_fuera_efectos", "remisiones_en_blanco", "cobertura_ordinal",
            "cobertura_demanda"]
    cab = ["Caso", "Sol. rotulada", "n"] + cols
    for titulo, grupo in (("Engroses (oro)", oros), ("Salidas v1 ya generadas", v1)):
        L += ["", f"## {titulo}", "", "| " + " | ".join(cab) + " |",
              "|" + "---|" * len(cab)]
        for f in grupo:
            m = f["m"]
            L.append("| " + " | ".join(
                [f["caso"][:34], "sí" if m["solucion_rotulada"] else "no",
                 _fmt(m["n_planteamientos"])] + [_fmt(m.get(c)) for c in cols]) + " |")
    L += ["", "## Qué acusa a los buenos, caso por caso", ""]
    for clave, nombre, mejor, umbral, acusa in METRICAS:
        if not acusa:
            continue
        acus = [f for f in oros if f["m"].get(clave) is not None and acusa(f["m"][clave], f["m"])]
        if acus:
            L.append(f"- **{nombre}** ({umbral}): " + "; ".join(
                f"{f['caso'][:30]} = {_fmt(f['m'][clave])}" for f in acus))
    L.append("")
    return "\n".join(L)


# ═══════════════════════════════════════════════════════════════════════════
def _imprimir(m: dict) -> None:
    for k, v in m.items():
        if k == "detalle":
            continue
        print(f"  {k:<32} {_fmt(v)}")
    d = m["detalle"]
    print(f"  {'aperturas':<32} {d['aperturas']}")
    print(f"  {'calificaciones por concepto':<32} {d['calificaciones_por_concepto']}")
    if d["anclas_faltan"]:
        print(f"  {'anclas que faltan':<32} {', '.join(d['anclas_faltan'][:25])}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("archivo", nargs="?")
    ap.add_argument("--escrito")
    ap.add_argument("--n", type=int)
    ap.add_argument("--calibrar", action="store_true")
    ap.add_argument("--salida", default=str(SALIDA))
    ap.add_argument("--diag", default=str(DIAG))
    ap.add_argument("--dir93", default=str(DIR93))
    a = ap.parse_args()
    if a.calibrar:
        r = calibrar(Path(a.salida), Path(a.diag), Path(a.dir93))
        print(r["informe"])
        print(f"\n→ {Path(a.salida) / 'calibracion.md'}")
    elif a.archivo:
        t = Path(a.archivo).read_text(encoding="utf-8")
        e = Path(a.escrito).read_text(encoding="utf-8") if a.escrito else None
        _imprimir(medir(t, e, a.n))
    else:
        ap.print_help()
        sys.exit(2)
