# -*- coding: utf-8 -*-
"""LA FICHA DE TRÁMITE: LOS HECHOS PROCESALES PRIMERO, LA PROSA DESPUÉS.

David, 3-oct-2026: «necesitamos pulcritud en los resultandos y considerandos…
disminuyendo el margen de error si el secretario introduce el auto de admisión
y los datos correctos (fechas). La idea es que estos resultandos funcionen en
toda la república sin que el secretario deba modificar más».

LO QUE HABÍA, MEDIDO. Una sola llamada de modelo (`redactar_estructura`)
escribía el V I S T O y TODOS los resultandos a ciegas —sin el auto de
admisión, sin el de turno, sin la demanda— y después el código leía esa prosa
con expresiones regulares para la existencia, la procedencia y el resolutivo.
En los 10 proyectos de octubre: la responsable mal nombrada en 9 de 9 amparos
directos («TRIBUNAL EL SEIS», el juzgado de primera instancia en lugar de la
Sala), fechas imposibles declaradas «oportunas» (una demanda de marzo de 2025
contra una sentencia de enero de 2026), el Ministerio Público «omitió formular
pedimento» afirmado sin fuente en 71 de 72, la «Oficialía de Partes de este
Tribunal» inventada.

LA REGLA DE ESTE MÓDULO: cada hecho procesal es un CAMPO con su FUENTE, y la
prosa se escribe después, con fórmulas fijas (`resultandos_por_tipo`). Las
fuentes, por precedencia:

    secretario  — lo que tecleó o confirmó en el formulario (`tramite_json`)
    auto        — leído del auto de admisión / turno / returno
    escrito     — leído del escrito del recurso: quién recurre, con qué
                  carácter y por conducto de quién (`leer_escrito`, tercera
                  ronda; AR 631/2025)
    acto        — leído de la resolución reclamada o recurrida (y la demanda)
    ficha_procesal — lo que ya resolvía `ficha_procesal.armar`
    formulario  — lo que el formulario principal SUPONE y no dice: la
                  responsable prellenada (AD) y, en un recurso con el
                  recurrente vacío, la quejosa como respaldo
    omision     — QUINTA RONDA (3-oct-2026, F1): la forma de notificación que
                  sólo sale del valor POR OMISIÓN del formulario principal
                  («personal»); «omision_autoridad», el «oficio» con que se
                  computa a la autoridad que recurre (D2) cuando ningún papel
                  dice cómo se le notificó. Van detrás de todo papel y el
                  considerando no afirma esa forma (`forma_por_omision`).

EL MODELO SÓLO LEE, Y LO QUE DEVUELVE SE ANCLA AL PAPEL. Una fecha se acepta
si su forma en letra o en dígitos aparece en el documento (`fecha_en_papel`);
un nombre, si aparece literal —normalizando espacios, comillas, mayúsculas y
tildes— o si hay un tramo del papel que se le parece en 0.85 o más, y entonces
se usa EL DEL PAPEL (`anclar`). Lo que no está, se tira y se dice. El sentido de
la resolución no se copia en prosa libre: se elige de un CATÁLOGO por tipo, con
palabras clave que tienen que estar en la parte resolutiva del papel (en la
revisión fiscal, el derecho subjetivo que la Sala además reconoció se copia del
resolutivo tras la frase del catálogo: `cola_derecho_subjetivo`, quinta ronda).

Y UN DATO QUE FALTA NO SE RELLENA: se queda vacío, el compositor pone el hueco
«*********» y el aviso nombra el dato. Una fecha equivocada se firma; un hueco
se ve.

FORMA DE LA FICHA (formato 1). Un dict plano y serializable, fechas ISO. Las
secciones OPCIONALES (`registro`, `returno`, `adhesivo`, `informe_101`,
`demanda`, `resolucion_impugnada`) son `{}` cuando no consta nada —así
`if ficha["adhesivo"]` dice la verdad— y, cuando consta algo, traen todas sus
claves. `admision`, `turno` y `acto` traen siempre sus claves. Las claves que no
aplican al tipo van vacías. `turno.titulo` y `returno.titulo` (el cargo del
ponente como lo dice el papel: «Magistrada», «Secretaria en funciones de
Magistrada») sólo aparecen si constan.

`registro` (CUARTA RONDA, 3-oct-2026) es el auto de Presidencia que forma y
registra el expediente cuando NO es el que lo admite (declinatoria, prevención;
en la queja de la fracción II, el que además requiere el informe del 101).
`admision` es siempre el auto que ADMITE. Se lee con
`(ficha.get("registro") or {}).get("fecha", "")`.

`relacionados` (SEXTA RONDA, 3-oct-2026) es la lista de los asuntos que el
SECRETARIO marcó como relacionados —[{"tipo", "numero", "estado":
"misma_sesion" | "resuelto"}], cuatro como mucho, [] si ninguno— y su fuente es
siempre «secretario»: la conexidad nunca se lee de los papeles. Y en la queja
la ficha lee también la demanda de amparo (`quejoso`, `demanda.fecha`,
`demanda.autoridades`, `demanda.actos`) del auto recurrido y del escrito, con
las guardas del amparo en revisión.
"""
from __future__ import annotations

import asyncio
import copy
import datetime as _dt
import difflib
import json
import os
import re
import unicodedata
from array import array
from collections import Counter

FORMATO = 1
HUECO = "*********"
TIPOS = ("amparo_directo", "amparo_revision", "queja", "revision_fiscal")
# «escrito»: el escrito con el que se interpone el recurso (TERCERA RONDA,
# 3-oct-2026): dice quién recurre y con qué carácter. Va detrás del auto y
# delante del acto y de la ficha procesal (`armar`).
FUENTES = ("secretario", "auto", "escrito", "acto", "ficha_procesal")
# Las fuentes que NO son un papel ni un dato del secretario (QUINTA RONDA,
# 3-oct-2026, F1): la forma de notificación por omisión. Ver `forma_por_omision`.
FUENTES_OMISION = ("omision", "omision_autoridad")

# El mismo modelo barato que lee el auto de admisión (`fase_admision`). Lee, no
# escribe: una llamada JSON por tipo con tope de salida corto.
MODELO_ACTO = os.getenv("MODELO_FICHA_TRAMITE", "gpt-5.6-luna")
ESFUERZO_ACTO = os.getenv("FICHA_TRAMITE_ESFUERZO", "low")
TOPE_SALIDA = 2500
UMBRAL_PARECIDO = 0.85

# ── LOS TOPES (TERCERA RONDA, 3-oct-2026: robustez, rev_6) ─────────────────
# TOPE_SEGUNDOS_MODELO: la lectura con modelo es RELLENO (lo determinista
# manda) y corre en el `gather` del adelanto con las fases 1-3. Con un proveedor
# colgado, `leer_acto` no volvía en 8 s, el respaldo de `llamada_modelo` entraba
# a los 180 s y después valía el tope del cliente (600 s): todo el adelanto
# esperando un dato prescindible. A los 60 s se sigue sin él, y se dice.
# TOPE_AUTO: un auto de admisión cabe de sobra en 30 000 caracteres. El lector
# acepta cualquier PDF de 400 o más y el secretario puede subir el expediente
# entero en «Tengo el auto de admisión»: medido, 498 000 caracteres (unas 166
# páginas) tardaban 14.9 s en el bucle de eventos.
TOPE_SEGUNDOS_MODELO = 60
TOPE_AUTO = 30000
# Las anclas del parecido (`anclar`): sólo las palabras RARAS del valor, y no
# más de éstas. Ver `anclar`.
TOPE_ANCLAS = 200

# Las claves planas del formulario «Trámite en este tribunal» (SPEC §1.1). El
# front las manda en `tramite_json` y `/taller/desde-admision` las devuelve en
# `ficha["tramite"]`: ida y vuelta con `de_formulario` / `a_formulario`.
#
# SEGUNDA RONDA (3-oct-2026): VEINTE CLAVES. Se añaden al final, con el mismo
# nombre en el formulario y en la ficha (primer nivel), tres datos que hasta hoy
# los considerandos sacaban de la prosa o dejaban en hueco:
#   «fraccion_63»  — la fracción del artículo 63 de la LFPCA por la que procede
#                    la revisión fiscal, en romano (I a X);
#   «cuantia»      — el monto en pesos de la revisión fiscal (fracción I; también
#                    sirve a la II, que es la de cuantía inferior o
#                    indeterminada), en la forma «$847,738.77» o «indeterminada»;
#   «fundamento_surtimiento» — el precepto que rige el surtimiento de la
#                    notificación, tal como va en la prosa: «el artículo 126 del
#                    Código de Procedimientos Civiles del Estado de Querétaro».
# El último era el hueco «conforme al *********» de la oportunidad en el amparo
# directo cuando la ley del acto no está en el catálogo (considerandos.txt).
#
# CUARTA RONDA (3-oct-2026): VEINTIUNA. «fecha_registro» —el auto de
# Presidencia que forma y registra, SÓLO si es distinto del que admite— va al
# final, con su ruta `registro.fecha`. RF 7/2025: el Tribunal radicó el recurso
# el 10-feb-2025, declinó competencia y lo admitió el 15-may; la fórmula decía
# «Por auto de Presidencia de quince de mayo… registró el recurso… y lo
# admitió», y el registro fue en febrero. Q 335/2025 (queja, fracción II): el
# auto que registró y requirió el informe es del 14-oct y el que lo tuvo por
# rendido y admitió, del 3-nov; salía el 3-nov en los dos.
#
# SEXTA RONDA (3-oct-2026): VEINTIDÓS. «relacionados» —los asuntos que el
# secretario marca como relacionados con el presente, con un clic en «Trámite
# en este tribunal»—. David: «siempre y cuando haya asuntos relacionados. No
# vamos a meter conexidad en automático. Hay que habilitar en el taller la
# opción de con un clic precisar si existen asuntos relacionados y con ello se
# genera el considerando». Es TEXTO PLANO como las demás claves: «tipo|numero|
# estado» separados por «;» («amparo_directo|452/2025|misma_sesion;
# revision_fiscal|33/2024|resuelto»); en la ficha, la lista de
# {"tipo", "numero", "estado"} (`relacionados_de`). Fuente SIEMPRE
# «secretario»: nunca se lee de los papeles (ni del auto ni del acto).
CLAVES_FORMULARIO = (
    "fecha_admision", "fecha_turno", "ponente_turno", "fecha_returno",
    "ponente_returno", "ministerio_publico", "adhesivo_quien",
    "adhesivo_presentacion", "adhesivo_admision", "adhesivo_notificacion",
    "fecha_acto", "organo_acto", "toca", "expediente_origen",
    "fecha_informe_101", "deposito_postal", "forma_notificacion",
    "fraccion_63", "cuantia", "fundamento_surtimiento",
    "fecha_registro", "relacionados",
)
_MAPA_FORMULARIO = {
    "relacionados": "relacionados",
    "fecha_registro": "registro.fecha",
    "fecha_admision": "admision.fecha", "fecha_turno": "turno.fecha",
    "ponente_turno": "turno.ponente", "fecha_returno": "returno.fecha",
    "ponente_returno": "returno.ponente", "ministerio_publico": "ministerio_publico",
    "adhesivo_quien": "adhesivo.quien", "adhesivo_presentacion": "adhesivo.presentacion",
    "adhesivo_admision": "adhesivo.admision", "adhesivo_notificacion": "adhesivo.notificacion",
    "fecha_acto": "acto.fecha", "organo_acto": "acto.organo", "toca": "acto.toca",
    "expediente_origen": "acto.expediente", "fecha_informe_101": "informe_101.fecha",
    "deposito_postal": "deposito_postal", "forma_notificacion": "forma_notificacion",
    "fraccion_63": "fraccion_63", "cuantia": "cuantia",
    "fundamento_surtimiento": "fundamento_surtimiento",
}
# UN CAMPO DEL FORMULARIO, UN DATO… QUE CADA TIPO GUARDA EN SU SITIO (front.txt
# #48). La tarjeta pide «organo_acto» y «expediente_origen» una sola vez y los
# rotula por tipo, pero el compositor y los considerandos los leen de campos
# distintos según el tipo:
#   AD — organo_acto es la responsable ORDENADORA (`responsable` y
#        `acto.organo`: un solo nombre en todo el documento); expediente_origen
#        es el juicio natural (`acto.expediente`; el toca va en «toca»).
#   AR — organo_acto es el JUZGADO que dictó lo recurrido (`acto.organo`, que el
#        compositor escribe como `juzgado`); expediente_origen es el juicio de
#        amparo indirecto (`acto.expediente`, que es su `juicio_amparo`).
#   Q  — organo_acto es el órgano que dictó el auto (`acto.organo`);
#        expediente_origen, el juicio de amparo (`acto.expediente`, también su
#        `juicio_amparo`).
#   RF — organo_acto es la SALA del TFJA (`sala` y `acto.organo`);
#        expediente_origen, el juicio contencioso (`expediente_tfja` y
#        `acto.expediente`).
# SIN ESTO la Sala que confirmó el secretario se quedaba en `acto.organo` y la
# leída del papel —en `sala`— seguía mandando en la revisión fiscal: lo
# confirmado perdía contra lo leído. La primera ruta de cada tupla es la que
# `a_formulario` lee primero al volver.
_POR_TIPO_FORMULARIO = {
    "amparo_directo": {"organo_acto": ("responsable", "acto.organo"),
                       "expediente_origen": ("acto.expediente",)},
    "amparo_revision": {"organo_acto": ("acto.organo",),
                        "expediente_origen": ("acto.expediente",)},
    "queja": {"organo_acto": ("acto.organo",),
              "expediente_origen": ("acto.expediente",)},
    "revision_fiscal": {"organo_acto": ("sala", "acto.organo"),
                        "expediente_origen": ("expediente_tfja", "acto.expediente")},
}
# Claves que el formulario PUEDE mandar además (no son del contrato mínimo con
# el front, pero si llegan mandan igual que las demás).
_MAPA_FORMULARIO_EXTRA = {
    "presentacion": "presentacion", "notificacion": "notificacion",
    "responsable": "responsable", "ejecutora": "ejecutora",
    "fecha_audiencia": "audiencia", "fecha_demanda": "demanda.fecha",
    "fraccion_97": "fraccion_97", "inciso_97": "inciso_97",
    "via_presentacion": "via_presentacion", "sala": "sala",
    "expediente_tfja": "expediente_tfja", "actora": "actora",
    "autoridad_demandada": "autoridad_demandada", "sentido": "acto.sentido",
}

# ── LOS ASUNTOS RELACIONADOS (SEXTA RONDA, 3-oct-2026) ─────────────────────
# SÓLO LOS QUE MARCA EL SECRETARIO. La conexidad no se deduce ni se lee de los
# papeles: con un relacionado marcado, el proyecto lleva el considerando de
# conexidad (se resuelven en la misma sesión) o de hecho notorio (ya se
# resolvió) y el rubro «RELACIONADO CON…» (`tipos_asunto`). Cuatro como mucho.
TIPOS_RELACIONADO = TIPOS
ESTADOS_RELACIONADO = ("misma_sesion", "resuelto")
TOPE_RELACIONADOS = 4
# El texto del formulario: cuatro «revision_fiscal|123456/2025|misma_sesion»
# caben de sobra; lo que excede es otra cosa y se ignora con aviso (`_tope_de`).
TOPE_TEXTO_RELACIONADOS = 400
_RX_NUMERO_RELACIONADO = re.compile(r"^\d{1,6}/\d{4}$")
_ESTADOS_ALIAS = {"misma_sesion": "misma_sesion", "en_la_misma_sesion": "misma_sesion",
                  "se_resuelve_en_la_misma_sesion": "misma_sesion", "conexidad": "misma_sesion",
                  "resuelto": "resuelto", "resuelta": "resuelto", "ya_resuelto": "resuelto",
                  "ya_se_resolvio": "resuelto", "hecho_notorio": "resuelto"}
_TIPO_EN_AVISO = {"amparo_directo": "amparo directo", "amparo_revision": "amparo en revisión",
                  "queja": "recurso de queja", "revision_fiscal": "revisión fiscal"}


# ═══════════════════════════════════════════════════════════════════════════
# LA BANDERA
# ═══════════════════════════════════════════════════════════════════════════
def rige() -> bool:
    """¿Rige `procedencia_por_tipo` en esta petición? Sin contexto o sin la
    bandera declarada, False: el camino viejo queda idéntico."""
    try:
        import contexto_taller as _ct
        f = getattr(_ct, "rige", None)
        if f is not None:
            return bool(f("procedencia_por_tipo"))
        if "procedencia_por_tipo" in getattr(_ct, "BANDERAS_REDISENO", {}):
            return bool(_ct.rediseno("procedencia_por_tipo"))
    except Exception:
        pass
    return False


# ═══════════════════════════════════════════════════════════════════════════
# NORMALIZAR
# ═══════════════════════════════════════════════════════════════════════════
_COMILLAS = str.maketrans({"“": '"', "”": '"', "«": '"', "»": '"', "„": '"',
                           "‘": "'", "’": "'", "´": "'", "`": "'"})


def _sin_tildes(x: str) -> str:
    x = unicodedata.normalize("NFD", str(x or ""))
    return "".join(c for c in x if unicodedata.category(c) != "Mn")


def _plano(x: str) -> str:
    """Minúsculas, sin tildes, comillas unificadas y un solo espacio."""
    return " ".join(_sin_tildes(str(x or "").translate(_COMILLAS)).lower().split())


# LOS CARACTERES DE CONTROL, FUERA (TERCERA RONDA, 3-oct-2026, rev_6). Uno solo
# —«Ana\x00Pérez» en `tramite_json`, o el «\x00» que trae el texto fitz de 5-6 %
# de los PDF reales del OAJ— tumbaba el .docx con «All strings must be XML
# compatible» DESPUÉS de pagar el OCR y el modelo: un 500 sin CORS, «Failed to
# fetch». Todo lo que entra a la ficha pasa por aquí (también el tramo anclado
# del papel). Los de espacio vertical (\x0b, \x0c: saltos de página) y los
# separadores \x1c-\x1f los convierte en espacio el `split`; los demás se quitan.
_RX_CONTROL = re.compile(r"[\x00-\x08\x0e-\x1b\x7f]")


def _limpio(x) -> str:
    return " ".join(_RX_CONTROL.sub("", str(x or "")).split()).strip()


def _tipo(tipo: str) -> str:
    try:
        import tipos_asunto as _ta
        t = _ta.normalizar(tipo or "")
    except Exception:
        t = (tipo or "").strip().lower()
    return t if t in TIPOS else ""


# ═══════════════════════════════════════════════════════════════════════════
# LAS FECHAS DEL PAPEL
# ═══════════════════════════════════════════════════════════════════════════
# CÓMO SE ESCRIBEN DE VERDAD. Medido en los 24 actos del banco Kingston: los
# tribunales de Querétaro escriben «a 20 veinte de febrero de 2025 dos mil
# veinticinco» —dígito, palabra y año doble—; la Primera Sala se come el «de»
# («19 diecinueve agosto de 2024»); la demanda dice «17 de enero del 2021»; la
# portada de la Oficina de Correspondencia, «13/11/2025». Un lector que sólo
# entiende «veinte de febrero de dos mil veinticinco» no ve ninguna de esas.
_MESES = {"enero": 1, "febrero": 2, "marzo": 3, "abril": 4, "mayo": 5, "junio": 6,
          "julio": 7, "agosto": 8, "septiembre": 9, "setiembre": 9, "octubre": 10,
          "noviembre": 11, "diciembre": 12}
_UNIDADES = {
    "cero": 0, "un": 1, "uno": 1, "primero": 1, "dos": 2, "tres": 3, "cuatro": 4,
    "cinco": 5, "seis": 6, "siete": 7, "ocho": 8, "nueve": 9, "diez": 10,
    "once": 11, "doce": 12, "trece": 13, "catorce": 14, "quince": 15,
    "dieciseis": 16, "diecisiete": 17, "dieciocho": 18, "diecinueve": 19,
    "veinte": 20, "veintiun": 21, "veintiuno": 21, "veintidos": 22, "veintitres": 23,
    "veinticuatro": 24, "veinticinco": 25, "veintiseis": 26, "veintisiete": 27,
    "veintiocho": 28, "veintinueve": 29, "treinta": 30, "treintaiuno": 31,
}
_RX_FECHA_LETRA = re.compile(
    r"\b(?P<dia>(?:treinta|veinte)\s+y\s+[a-z]+|[a-z]{2,}|\d{1,2})(?:\s*(?:o|º|°)\.?)?\)?"
    r"\s+(?:de\s+)?(?P<mes>" + "|".join(_MESES) + r")"
    r"\s+(?:del?\s+)?(?:(?:ano|año)\s+)?\(?(?P<anio>\d{4}|dos\s+mill?(?:\s+[a-z]+){0,3})")
_RX_FECHA_DIGITOS = re.compile(r"\b(\d{1,2})\s*([/.\-])\s*(\d{1,2})\s*\2\s*(\d{4})\b")
_RX_FECHA_ISO = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")


def _numero_de(palabras: str, ocr: bool = False):
    p = _sin_tildes(palabras or "").lower().strip()
    if p.isdigit():
        return int(p)
    if p in _UNIDADES:
        return _UNIDADES[p]
    m = re.fullmatch(r"(treinta|veinte)\s+y\s+([a-z]+)", p)
    if m and m.group(2) in _UNIDADES and _UNIDADES[m.group(2)] < 10:
        return _UNIDADES[m.group(1)] + _UNIDADES[m.group(2)]
    # LA ERRATA DEL OCR, SÓLO EN EL DÍA Y SÓLO SI ES INEQUÍVOCA. «vertinueve de
    # agosto de dos mil veinticuatro» (ADC 625/2024) es el veintinueve: una
    # palabra larga a 0.85 o más de un solo número. Una corta no se adivina.
    if ocr and len(p) >= 6 and " " not in p:
        cerca = difflib.get_close_matches(p, [k for k in _UNIDADES if len(k) >= 5], n=2, cutoff=0.85)
        if len(cerca) == 1 or (len(cerca) == 2 and _UNIDADES[cerca[0]] == _UNIDADES[cerca[1]]):
            return _UNIDADES[cerca[0]]
    return None


def _anio_de(cadena: str):
    c = (cadena or "").strip()
    if c.isdigit():
        return int(c)
    pal = c.split()
    if len(pal) < 2 or pal[0] != "dos" or pal[1] not in ("mil", "mill"):
        return None
    resto = pal[2:]
    # DE LA MÁS LARGA A LA MÁS CORTA: «dos mil veinte y cinco» es 2025, pero en
    # «dos mil veinticinco y su ejecución» la cola no es del año.
    for n in (3, 2, 1):
        if len(resto) >= n:
            v = _numero_de(" ".join(resto[:n]))
            if v is not None and v < 100:
                return 2000 + v
    # «dos mil» a secas sería el año 2000: en un expediente vivo es casi
    # siempre un año cortado por el OCR («dos mill veintidós»). No se adivina.
    return None


def _iso(a: int, m: int, d: int) -> str:
    try:
        return _dt.date(a, m, d).isoformat()
    except (ValueError, TypeError):
        return ""


def _plano_fechas(texto: str) -> str:
    """El plano sobre el que se leen las fechas (y en el que caen sus
    posiciones): sin tildes, en minúsculas, un solo espacio."""
    return " ".join(_sin_tildes(str(texto or "")).lower().split())


def _fechas_de_plano(plano: str) -> tuple:
    """Las fechas de un texto YA PLANO (`_plano_fechas`). SIN CACHÉ (tercera
    ronda): quien lee el mismo papel muchas veces lo hace con un `Papel`, que
    las calcula una vez y muere con la llamada."""
    salida = []
    for m in _RX_FECHA_LETRA.finditer(plano):
        d = _numero_de(m.group("dia"), ocr=True)
        a = _anio_de(m.group("anio"))
        iso = _iso(a or 0, _MESES[m.group("mes")], d or 0) if d and a and 1900 < a < 2100 else ""
        if iso:
            salida.append((iso, m.group(0), m.start()))
    for m in _RX_FECHA_DIGITOS.finditer(plano):
        iso = _iso(int(m.group(4)), int(m.group(3)), int(m.group(1)))
        if iso:
            salida.append((iso, m.group(0), m.start()))
    for m in _RX_FECHA_ISO.finditer(plano):
        iso = _iso(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        if iso:
            salida.append((iso, m.group(0), m.start()))
    salida.sort(key=lambda x: x[2])
    return tuple(salida)


def fechas_del_texto(texto) -> list:
    """[(iso, literal, posición)] de todas las fechas del papel, en orden.
    `texto` puede ser un `Papel` (las calcula una sola vez)."""
    if isinstance(texto, Papel):
        return list(texto.fechas)
    return list(_fechas_de_plano(_plano_fechas(texto)))


def a_iso(valor) -> str:
    """Una fecha en cualquiera de sus formas → «AAAA-MM-DD», o «» si no es fecha."""
    if isinstance(valor, _dt.datetime):
        return valor.date().isoformat()
    if isinstance(valor, _dt.date):
        return valor.isoformat()
    s = _limpio(valor)
    if not s:
        return ""
    m = re.fullmatch(r"(\d{4})-(\d{1,2})-(\d{1,2})(?:[T ].*)?", s)
    if m:
        return _iso(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    f = fechas_del_texto(s)
    return f[0][0] if f else ""


def fecha_en_papel(iso, texto) -> bool:
    """¿Está esta fecha escrita en el papel, en letra o en dígitos? `texto`
    puede ser un `Papel`: así se lee el papel una vez para todas las fechas."""
    i = a_iso(iso)
    pp = _papel(texto)
    if not i or not pp.texto.strip():
        return False
    return i in pp.isos


def _fecha_obj(iso):
    i = a_iso(iso)
    try:
        return _dt.date.fromisoformat(i) if i else None
    except ValueError:
        return None


def _dmy(iso) -> str:
    f = _fecha_obj(iso)
    return f.strftime("%d/%m/%Y") if f else str(iso or "")


# LA FECHA DE LA RESOLUCIÓN ES LA DE SU PROEMIO: «Santiago de Querétaro,
# Querétaro, a 20 veinte de febrero de 2025…». Se busca la primera fecha que
# va tras «Ciudad, Estado, a» en el principio del papel; si no la hay, la
# primera fecha de la cabecera que no sea la de otra resolución citada.
# Se mira sobre el texto SIN TILDES PERO CON SUS MAYÚSCULAS: tras la coma del
# proemio van el estado o la abreviatura («Querétaro, Qro.;»), con mayúscula;
# «…Administrativo en Querétaro, el 3 (tres) de febrero…» es otra cosa (ADA
# 702/2022: daba el día en que se recibió un oficio).
_RX_ANTES_PROEMIO = re.compile(
    r"[A-Z][A-Za-z.]{2,}\s*[,.:;]+\s*(?:[A-Z][A-Za-z.]{1,}\s*[,.;]*\s*){0,3}"
    r"(?:[aoAO]\s+|[aA]\s+los\s+|el\s+d[ií]a\s+)?(?:\d{1,2}\s*\(?\s*)?$")
_RX_CITA_ANTES = re.compile(
    r"(?:(?:contra|de\s+fecha|con\s+fecha|en\s+fecha|sentencia|resoluci[oó]n|auto|acuerdo|recibid[oa]|"
    r"presentad[oa]|notificad[oa]|el\s+d[ií]a)\s+(?:de\s+|del?\s+|la\s+|el\s+)?(?:\d{1,2}\s+)?$)|"
    # La boleta de la Oficina de Correspondencia: «Fecha de presentación/Fecha
    # de depósito: 20/09/2024» no es el proemio de nada (ADC 590/2024).
    r"fecha[^,;]{0,45}[:.]?\s*$", re.I)


def _fecha_proemio(texto: str, cabeza: int = 8000) -> str:
    t = " ".join(str(texto or "").split())[:cabeza]
    fs = fechas_del_texto(t)
    if not fs:
        return ""
    vistas = _vistas_proemio(t)
    for iso, _lit, pos in fs:
        if _es_proemio(t, pos, vistas):
            return iso
    return ""


def _vistas_proemio(t: str) -> tuple:
    """(con_mayus, plano) de `t`: las dos vistas en las que `_es_proemio` mira
    lo que va antes de una fecha. SE CALCULAN UNA VEZ POR TEXTO (tercera ronda,
    rev_6): antes se recalculaban del texto ENTERO por cada fecha y `leer_auto`
    crecía al cuadrado (15 s con un documento de 166 páginas)."""
    con_mayus = " ".join(_sin_tildes(t).split())
    plano = con_mayus.lower()
    if len(con_mayus) != len(plano):       # un carácter que cambia de largo: sin mayúsculas
        con_mayus = plano
    return con_mayus, plano


def _es_proemio(t: str, pos: int, vistas: tuple = None) -> bool:
    """¿La fecha que empieza en `pos` (posición en el plano de `t`) es la de
    un encabezado «Ciudad, Estado, a…»? `vistas`: lo de `_vistas_proemio(t)`,
    para no recalcularlo por cada fecha."""
    con_mayus, plano = vistas or _vistas_proemio(t)
    antes_m = con_mayus[max(0, pos - 90):pos]
    antes = plano[max(0, pos - 90):pos]
    return bool(_RX_ANTES_PROEMIO.search(antes_m)) and not _RX_CITA_ANTES.search(antes)


# ═══════════════════════════════════════════════════════════════════════════
# LA FORMA DE ÓRGANO
# ═══════════════════════════════════════════════════════════════════════════
# «AUTORIDAD RESPONSABLE: TRIBUNAL EL SEIS» (AD 103, nueve proyectos), «PRIMERA
# SALA CIVIL JUICIO» (AD 642), «Sala Familiar del Tribunal Superior de Justicia
# en el Estado» (640, 296). Ninguno de los tres es el nombre de un órgano y los
# tres llegaron a la carátula, la competencia, la existencia y el resolutivo.
_CABEZA_JURISDICCIONAL = (r"sala|tribunal|juzgado|juez|jueza|junta|magistrad[oa]|pleno|"
                          r"seccion|ponencia|presidencia")
_CABEZA_AUTORIDAD = (_CABEZA_JURISDICCIONAL + r"|presidente|presidenta|director|directora|"
                     r"direccion|secretari[oa]|secretaria|administracion|administrador[a]?|"
                     r"instituto|procuraduria|procurador[a]?|comision|fiscalia|fiscal|agente|"
                     r"unidad|delegacion|delegad[oa]|titular|gobernador[a]?|congreso|"
                     r"legislatura|ayuntamiento|cabildo|municipio|consejo|oficial|registr\w+|"
                     r"actuari[oa]|coordinacion|coordinador[a]?|jefe|jefa|subdirector[a]?|"
                     r"tesorer\w+|comisionad[oa]|servicio|sistema|organismo|policia|camara|"
                     r"autoridad|centro|oficina|subsecretari[oa]|visitador[a]?|inspector[a]?")
_RX_CABEZA_J = re.compile(r"\b(?:" + _CABEZA_JURISDICCIONAL + r")\b")
_RX_CABEZA_A = re.compile(r"\b(?:" + _CABEZA_AUTORIDAD + r")\b")
_RX_BASURA_ORGANO = re.compile(
    r"^(?:el\s+|la\s+)?(?:tribunal|sala|juzgado|junta)\s+(?:el|la|los|las|un|una)\b")
_ULTIMA_NO = {"juicio", "toca", "expediente", "sentencia", "amparo", "resolucion", "demanda",
              "de", "del", "la", "el", "en", "y", "los", "las", "e", "con", "por", "a", "al"}
_RX_SIN_ENTIDAD = re.compile(r"\b(?:en\s+el|del)\s+estado\s*\.?$")


def es_organo(nombre: str, jurisdiccional: bool = True) -> tuple:
    """(True, «») si `nombre` tiene forma de órgano; (False, motivo) si no.
    El motivo «sin_entidad» es un aviso, no un rechazo: el nombre es de órgano
    pero le falta el estado («…en el Estado»)."""
    n = _limpio(nombre).strip(" ,.;:-")
    p = _plano(n)
    if not p:
        return False, "vacío"
    if HUECO in n or "*" in n:
        return False, "trae un hueco"
    pal = p.split()
    if len(pal) < 2:
        return False, "es una sola palabra"
    if len(pal) > 45:
        return False, "es demasiado largo para un nombre"
    if not (_RX_CABEZA_J if jurisdiccional else _RX_CABEZA_A).search(p):
        return False, "no nombra un órgano (sala, tribunal, juzgado, junta…)"
    if _RX_BASURA_ORGANO.search(p):
        return False, "es un fragmento («tribunal el…»), no un nombre"
    if pal[-1].strip(".,") in _ULTIMA_NO:
        return False, f"está cortado (termina en «{pal[-1]}»)"
    # Los dígitos sólo se admiten donde los nombres los llevan: «Distrito 42»,
    # «Tercera Ponencia», «Región», «número».
    for m in re.finditer(r"\d+", p):
        antes = p[max(0, m.start() - 14):m.start()]
        if not re.search(r"(?:distrito|numero|no\.|ponencia|region|seccion|sala|juzgado)\s*$", antes):
            return False, "trae números que no son de un nombre"
    if _RX_SIN_ENTIDAD.search(p):
        return True, "sin_entidad"
    return True, ""


_MINUSCULAS = {"en", "de", "del", "y", "la", "el", "los", "las", "e", "con", "a", "al", "por", "lo"}


def sin_versales(nombre: str) -> str:
    """«SALA REGIONAL DEL CENTRO II DEL TRIBUNAL…» → «Sala Regional del Centro
    II del Tribunal…». Sólo si viene TODO en versales; los romanos y las siglas
    cortas se respetan. El nombre no cambia: cambia la caja."""
    n = _limpio(nombre)
    letras = [c for c in n if c.isalpha()]
    if not letras or not all(c.isupper() for c in letras):
        return n
    fuera = []
    for i, w in enumerate(n.split()):
        nucleo = w.strip(",.;:()\"'«»")
        # Los romanos, las siglas con punto («S.A.», «C.T.M.») y lo que va entre
        # paréntesis («(INFONAVIT)») se quedan como están.
        if re.fullmatch(r"[IVXL]{1,6}", nucleo) or re.fullmatch(r"(?:[A-ZÁÉÍÓÚÑ]\.){1,4}[A-ZÁÉÍÓÚÑ]?", nucleo) \
                or (w.startswith("(") and w.rstrip(",.;:").endswith(")")) or re.fullmatch(r"(?:[A-ZÁÉÍÓÚÑ]\.){2,}\w*\.?", nucleo):
            fuera.append(w)
        elif w.lower() in _MINUSCULAS and i:
            fuera.append(w.lower())
        else:
            fuera.append(w[:1] + w[1:].lower())
    return " ".join(fuera)


_RX_TRATAMIENTO = re.compile(
    r"^(?:(?:el|la|los|las)\s+)?(?:c\.\s*['’]?\s*|ciudadan[oa]\s+)?"
    r"(?:magistrad[oa]s\s+que\s+integran\s+(?:la|el)\s+|integrantes\s+de\s+la\s+)?(?:h\.\s*)?", re.I)


def _sin_tratamiento(n: str) -> str:
    """«El C. Juez…», «Los Magistrados que integran la Primera Sala…», «H. Sala…»
    → el nombre del órgano. Y sin la abreviatura del estado colgando al final
    («…con sede en Querétaro, Qro.»)."""
    n = _limpio(_RX_TRATAMIENTO.sub("", _limpio(n)))
    return re.sub(r",\s*[A-Z][a-z]{1,3}\.$", "", n).strip()


def _nucleo(n: str) -> str:
    """Las tres primeras palabras con contenido: «segunda sala civil»,
    «juzg segundo administrativo». Juez, jueza y juzgado son el mismo órgano."""
    p = _plano(_sin_tratamiento(n))
    p = re.sub(r"\b(?:juez|jueza|juzgado)\b", "juzg", p)
    # «Jueza Segunda… Especializada» y «Juez Segundo… Especializado» (590/2024)
    # son el mismo juzgado.
    p = re.sub(r"\b(primer|segund|tercer|cuart|quint|sext|septim|octav|noven|decim)(?:a|o)\b", r"\1o", p)
    pal = [w for w in re.findall(r"[a-z0-9]+", p) if w not in _MINUSCULAS]
    return " ".join(pal[:3])


def _puntaje_organo(n: str, papel) -> float:
    """Cuánto se parece a un nombre oficial completo y bien escrito: forma de
    órgano, con su estado, y con palabras que el papel repite (una errata del
    OCR —«Queretato», «Superio»— aparece una sola vez). `papel`: texto o
    `Papel` (las frecuencias se cuentan una vez por papel)."""
    ok_, mot = es_organo(n)
    if not ok_:
        return -1.0
    p = _plano(n)
    frec = _papel(papel).frecuencias
    pal = [w for w in re.findall(r"[a-z]{5,}", p)]
    apoyo = sum(1 for w in pal if frec.get(w, 0) >= 2) / max(1, len(pal))
    entidad = bool(re.search(r"\bestado\s+de\s+[a-z]{4,}|federal\s+de\s+justicia|distrito\s+\d", p))
    return (2.0 if mot == "" else 0.5) + (1.5 if entidad else 0.0) + 2.0 * apoyo + min(len(pal), 14) / 14.0


def mejor_organo(candidatos: list, papel="") -> str:
    """De varios nombres del MISMO órgano leídos en fuentes distintas, el más
    completo y mejor escrito; si nombran órganos distintos, gana el que más
    fuentes nombran (y, a igualdad, el primero de la lista). Sin tratamientos
    («El C.», «H.») y sin versales. `papel`: texto o `Papel`."""
    cands = [sin_versales(_sin_tratamiento(c)) for c in candidatos if _limpio(c)]
    cands = [c for c in cands if es_organo(c)[0]]
    if not cands:
        return ""
    pp = _papel(papel)
    grupos: dict = {}
    for i, c in enumerate(cands):
        grupos.setdefault(_nucleo(c), []).append((i, c))
    nucleo = max(grupos, key=lambda k: (len(grupos[k]), -grupos[k][0][0]))
    return max((c for _i, c in grupos[nucleo]), key=lambda c: _puntaje_organo(c, pp))


def _misma_base(a: str, b: str) -> bool:
    """¿Nombran al mismo órgano, uno más completo que el otro?"""
    pa = re.sub(r"\bh\.\s*", "", _plano(a)).strip(" .,")
    pb = re.sub(r"\bh\.\s*", "", _plano(b)).strip(" .,")
    pa = re.sub(r"\ben\s+el\s+estado\b", "del estado", pa)
    pb = re.sub(r"\ben\s+el\s+estado\b", "del estado", pb)
    return bool(pa and pb and (pa in pb or pb in pa))


# ═══════════════════════════════════════════════════════════════════════════
# EL ANCLAJE AL PAPEL
# ═══════════════════════════════════════════════════════════════════════════
def _mapa_plano(texto: str) -> tuple:
    """(plano, mapa): el texto normalizado y, por cada carácter suyo, su índice
    en el original. Así un tramo hallado en el plano se devuelve TAL COMO ESTÁ
    ESCRITO en el papel.

    SIN CACHÉ Y CON EL MAPA EN `array('I')` (tercera ronda, rev_6): la tupla de
    enteros costaba unos 40 bytes por carácter y una `lru_cache` la retenía con
    el texto entero como llave. Quien lo necesita varias veces usa un `Papel`."""
    plano: list = []
    mapa = array("I")
    espacio = True
    for i, c in enumerate(texto.translate(_COMILLAS)):
        if c.isspace():
            if not espacio:
                plano.append(" ")
                mapa.append(i)
            espacio = True
            continue
        if c < "\x80":                     # ASCII: sin tildes que quitar
            plano.append(c.lower())
            mapa.append(i)
            espacio = False
            continue
        for d in unicodedata.normalize("NFD", c):
            if unicodedata.category(d) == "Mn":
                continue
            plano.append(d.lower())
            mapa.append(i)
        espacio = False
    return "".join(plano), mapa


_PUNTA = ".,;:()[]{}\"'«»¡!¿?-–—/*"


class Papel:
    """UN PAPEL, NORMALIZADO UNA SOLA VEZ POR LLAMADA (tercera ronda,
    3-oct-2026, rev_6).

    LO QUE HABÍA: tres `functools.lru_cache` con el texto ENTERO como llave
    (`_fechas_cache`, `_frecuencias`, `_mapa_plano`). Medido con tracemalloc:
    tras 10 lecturas de amparos directos (unos 210 000 caracteres de acto más
    demanda) quedaban 72 MB retenidos por worker —con -w 2, unos 145 MB fijos
    en una instancia de 2 GB que ya tuvo un OOM el 25-sep— y el texto de los
    expedientes de otros secretarios vivía en el proceso entre peticiones.

    AHORA quien lee crea un `Papel`, se lo pasa a quien lo necesita (`anclar`,
    `fecha_en_papel`, `mejor_organo`, `_numero_en_papel`) y el papel muere con
    la llamada. Cada vista se calcula la primera vez que se pide. Sin estado
    global: se puede usar dentro de `asyncio.to_thread`."""
    __slots__ = ("texto", "_plano", "_mapa", "_fechas", "_isos", "_toks", "_indice",
                 "_frec", "_compacto")

    def __init__(self, texto=""):
        self.texto = _RX_CONTROL.sub("", str(texto or ""))
        self._plano = self._mapa = self._fechas = self._isos = None
        self._toks = self._indice = self._frec = self._compacto = None

    def _pm(self):
        if self._plano is None:
            self._plano, self._mapa = _mapa_plano(self.texto)

    @property
    def plano(self) -> str:
        self._pm()
        return self._plano

    @property
    def mapa(self):
        self._pm()
        return self._mapa

    @property
    def fechas(self) -> tuple:
        if self._fechas is None:
            self._fechas = _fechas_de_plano(_plano_fechas(self.texto))
        return self._fechas

    @property
    def isos(self) -> frozenset:
        if self._isos is None:
            self._isos = frozenset(x[0] for x in self.fechas)
        return self._isos

    @property
    def tokens(self) -> list:
        """[(palabra sin puntuación, inicio, fin)] sobre `plano`; inicio y fin
        son los de la palabra CON su puntuación, para devolver el tramo tal
        como está escrito."""
        if self._toks is None:
            self._toks = [(m.group(0).strip(_PUNTA), m.start(), m.end())
                          for m in re.finditer(r"\S+", self.plano)]
        return self._toks

    @property
    def indice(self) -> dict:
        """palabra → [posiciones en `tokens`]: dónde está cada palabra."""
        if self._indice is None:
            ind: dict = {}
            for i, (w, _a, _b) in enumerate(self.tokens):
                if w:
                    ind.setdefault(w, []).append(i)
            self._indice = ind
        return self._indice

    @property
    def frecuencias(self) -> Counter:
        if self._frec is None:
            base = self._plano if self._plano is not None else _plano(self.texto)
            self._frec = Counter(re.findall(r"[a-z]{4,}", base))
        return self._frec

    @property
    def compacto(self) -> str:
        if self._compacto is None:
            self._compacto = re.sub(r"\s+", "", self.texto).lower()
        return self._compacto


def _papel(x) -> Papel:
    return x if isinstance(x, Papel) else Papel(x)


def _tramo_original(texto: str, mapa, ini: int, fin: int) -> str:
    if ini >= fin or fin > len(mapa):
        return ""
    t = _limpio(texto[mapa[ini]:mapa[fin - 1] + 1]).strip(" ,;:")
    # «…del Estado de Querétaro..» (la demanda del 274/2025): dos puntos
    # seguidos nunca son de un nombre; uno solo sí («S.A. de C.V.»).
    return re.sub(r"\s*\.{2,}$", "", t).strip(" ,;:")


def anclar(valor: str, texto, umbral: float = UMBRAL_PARECIDO) -> tuple:
    """(valor, modo). `modo` es «literal» (está en el papel tal cual, salvo
    espacios, comillas, mayúsculas y tildes: se devuelve el valor), «parecido»
    (se devuelve EL TRAMO DEL PAPEL que más se le parece, con ratio ≥ umbral)
    o «» (no está: se devuelve «»). `texto`: el papel, o un `Papel` ya hecho.

    CÓMO SE BUSCA EL PARECIDO (tercera ronda, 3-oct-2026, rev_6). Antes se
    anclaba en CADA palabra útil del valor —hasta 4 000 anclas— y en cada una
    se probaban varias ventanas con `SequenceMatcher.ratio` sobre caracteres;
    `quick_ratio` no poda entre dos tramos de prosa jurídica de largo parecido.
    Medido en un AR real de 317 316 caracteres: 5 actos reclamados de 60
    palabras con dos distintas (la errata del OCR que el modelo corrige)
    tardaban 3.9, 8.7, 3.6, 2.8 y 12.9 s, síncronos en el bucle de eventos. Es
    el caso esperado, no el raro. Ahora:
      1. se ancla sólo en las palabras RARAS del valor (las que menos se
         repiten en el papel): cada aparición vota por dónde empezaría el
         tramo, y se miran las `TOPE_ANCLAS` regiones más votadas;
      2. en cada región, un alineamiento POR PALABRAS (barato) dice dónde
         empieza y acaba el tramo, y poda las regiones que comparten menos
         del 40 % de las palabras;
      3. sólo en lo que sobrevive se mide el parecido por caracteres, moviendo
         cada borde hasta dos palabras."""
    v = _limpio(valor)
    pp = _papel(texto)
    pv = _plano(v)
    if len(pv) < 3 or not pp.texto.strip():
        return "", ""
    plano = pp.plano
    if pv in plano:
        return v, "literal"
    toks_v = [w.strip(_PUNTA) for w in pv.split()]
    n = len(toks_v)
    toks = pp.tokens
    indice = pp.indice
    if not toks or not n:
        return "", ""
    # 1 · LAS ANCLAS: las palabras del valor que están en el papel, de la más
    # rara a la más común (y, a igualdad, la primera del valor). Cada aparición
    # de una de ellas VOTA por el sitio donde empezaría el tramo; donde votan
    # varias palabras está el tramo. Así, aunque todas sean comunes, no se mira
    # sólo el principio del papel.
    primera: dict = {}
    for k, w in enumerate(toks_v):
        if w and w in indice and w not in primera:
            primera[w] = k
    if not primera:
        return "", ""
    utiles = [w for w in primera if len(w) >= 4] or list(primera)
    utiles.sort(key=lambda w: (len(indice[w]), primera[w]))
    votos: Counter = Counter()
    vistas = 0
    for w in utiles[:8]:
        k = primera[w]
        for i in indice[w]:
            votos[i - k] += 1
        vistas += len(indice[w])
        if vistas > 50000:
            break
    # 2 · LAS REGIONES: las más votadas, hasta `TOPE_ANCLAS`, sin repetir una
    # que ya cubre otra (±3 palabras). Un tramo largo (un acto reclamado) puede
    # venir abreviado: la región crece con él, para que el tramo del papel quepa.
    holgura = n + 4 if n < 12 else int(n * 1.4) + 4
    regiones, cubiertos = [], set()
    for ini, _c in votos.most_common():
        if len(regiones) >= TOPE_ANCLAS:
            break
        if ini in cubiertos:
            continue
        regiones.append(ini)
        cubiertos.update(range(ini - 3, ini + 4))
    palabras = [x[0] for x in toks]
    tramos = set()
    for ini in regiones:
        a = max(0, ini - 4)
        b = min(len(toks), max(a + 1, ini + holgura))
        sm = difflib.SequenceMatcher(None, palabras[a:b], toks_v, autojunk=False)
        bloques = [bl for bl in sm.get_matching_blocks() if bl.size]
        if not bloques or sum(bl.size for bl in bloques) < 0.4 * n:
            continue
        tramos.add((a + bloques[0].a, a + bloques[-1].a + bloques[-1].size))
    # 3 · EL PARECIDO POR CARACTERES, sólo en lo que sobrevivió.
    mejor = (0.0, 0, 0)
    vistos = set()
    for s, e in tramos:
        for s2 in range(s - 2, s + 3):
            for e2 in range(e - 2, e + 3):
                if s2 < 0 or e2 > len(toks) or e2 <= s2 or (s2, e2) in vistos:
                    continue
                vistos.add((s2, e2))
                cand = plano[toks[s2][1]:toks[e2 - 1][2]]
                sm = difflib.SequenceMatcher(None, cand, pv, autojunk=False)
                if sm.real_quick_ratio() < umbral or sm.quick_ratio() < umbral:
                    continue
                r = sm.ratio()
                if r > mejor[0]:
                    mejor = (r, toks[s2][1], toks[e2 - 1][2])
        if mejor[0] >= 0.995:
            break
    if mejor[0] >= umbral:
        tramo = _tramo_original(pp.texto, pp.mapa, mejor[1], mejor[2])
        if tramo:
            return tramo, "parecido"
    return "", ""


def _anclar_lista(valores, papel, que: str, avisos: list) -> list:
    pp = _papel(papel)
    fuera = []
    for v in valores or []:
        v = _limpio(v) if isinstance(v, str) else ""
        if not v:
            continue
        a, modo = anclar(v, pp)
        if a:
            a = sin_versales(a)
            if not any(_plano(a) == _plano(x) for x in fuera):
                fuera.append(a)
        else:
            avisos.append(f"SE DESCARTÓ «{v[:90]}» como {que}: no aparece en el papel. "
                          f"Escríbelo tú si es correcto.")
    return fuera


# UN ACTO RECLAMADO CORTADO A MEDIA FRASE NO SE FIRMA (3-oct-2026, integración).
# AR 631/2025 de punta a punta: el segundo acto salió «…relativo al juicio sumario
# civil sobre rescisión de contrato de arrendamiento promovido por Unión de.» —el
# lector pedía «sus primeras 60 palabras» y el tramo anclado terminaba donde el
# modelo cortó—. Si el tramo no cierra con su puntuación, se sigue EN EL PAPEL
# hasta el punto o el punto y coma siguiente (tope de 120 palabras más); si aun así
# no cierra, va lo que hay con aviso.
_RX_CIERRE_ORACION = re.compile(r"[.;:](?=\s|$)")
_TOPE_COMPLETAR = 900


def _completar_oracion(valor: str, papel, avisos: list = None, que: str = "acto reclamado") -> str:
    v = (valor or "").rstrip()
    if not v or re.search(r"[.;:]\s*[»”\"']?$", v):
        return valor
    texto = _papel(papel).texto or ""
    pal = [re.escape(w) for w in re.findall(r"\w+", v)[-6:]]
    if len(pal) < 3:
        return valor
    m = None
    for m in re.finditer(r"\W+".join(pal), texto, flags=re.I):
        pass
    if m is None:
        if avisos is not None:
            avisos.append(f"EL {que.upper()} «…{v[-60:]}» PARECE CORTADO A MEDIA FRASE y no se pudo completar "
                          f"con el papel: compruébalo antes de firmar.")
        return valor
    cola = texto[m.end(): m.end() + _TOPE_COMPLETAR]
    c = None
    for c_ in _RX_CIERRE_ORACION.finditer(cola):
        if c_.group(0) != ".":
            c = c_
            break
        # EL PUNTO DE UNA ABREVIATURA NO CIERRA («S.A. de C.V.», «Lic.», «núm.», «C.»).
        tok = re.search(r"(\S+)$", cola[:c_.start()])
        tok = tok.group(1) if tok else ""
        if re.fullmatch(r"(?:[A-Za-zÁÉÍÓÚÑáéíóúñ]\.)*[A-Za-zÁÉÍÓÚÑáéíóúñ]", tok) or \
                tok.lower().rstrip(",") in ("lic", "licda", "mtro", "mtra", "dr", "dra", "núm", "num", "art",
                                            "fr", "frac", "no", "pág", "p", "c", "sr", "sra", "ing", "arq"):
            continue
        c = c_
        break
    if not c:
        if avisos is not None:
            avisos.append(f"EL {que.upper()} «…{v[-60:]}» PARECE CORTADO A MEDIA FRASE: el papel no lo cierra "
                          f"cerca; compruébalo antes de firmar.")
        return valor
    extra = " ".join(cola[:c.start()].split())
    return (v + (" " if extra and not extra.startswith((",", ";")) else "") + extra + cola[c.start()]).strip()


def _numero_en_papel(num: str, papel) -> bool:
    n = re.sub(r"\s+", "", str(num or ""))
    if len(n) < 3:
        return False
    return n.lower() in _papel(papel).compacto


# ═══════════════════════════════════════════════════════════════════════════
# LA SEDE: CIRCUITO Y CIUDAD DE MÉXICO
# ═══════════════════════════════════════════════════════════════════════════
# Lo decide la supletoriedad (David, 3-oct-2026): el Código Federal de
# Procedimientos Civiles en toda la república, salvo en Ciudad de México, donde
# ya opera el Código Nacional de Procedimientos Civiles y Familiares. Y la
# fracción del Acuerdo General 3/2013 es el número del circuito en romanos.
_ROMANO_A_INT = {"I": 1, "V": 5, "X": 10, "L": 50}
_RX_CDMX = re.compile(
    r"ciudad\s+de\s+mexico|\bcdmx\b|\bcd\.?\s*(?:de\s+)?mex(?:ico)?\b|mexico,?\s+d\.?\s*f\.?|"
    r"distrito\s+federal|\bcd\.?\s*mx\b")
_ORDINALES = {"primer": 1, "primero": 1, "segundo": 2, "tercer": 3, "tercero": 3, "cuarto": 4,
              "quinto": 5, "sexto": 6, "septimo": 7, "octavo": 8, "noveno": 9, "decimo": 10,
              "undecimo": 11, "duodecimo": 12, "vigesimo": 20, "trigesimo": 30}


def _romano_int(r: str) -> int:
    r = (r or "").upper()
    if not r or not re.fullmatch(r"[IVXL]+", r):
        return 0
    total, prev = 0, 0
    for c in reversed(r):
        v = _ROMANO_A_INT[c]
        total = total - v if v < prev else total + v
        prev = max(prev, v)
    return total


def _circuito_num(tribunal: str) -> int:
    t = str(tribunal or "")
    p = _plano(t)
    if "centro auxiliar" in p or "region" in p and "circuito" not in p:
        return 0
    try:
        import banco as _bk
        r = _bk.fraccion_del_acuerdo(t)
        if r:
            return _romano_int(r)
    except Exception:
        pass
    m = re.search(r"\b(\d{1,2})\s*(?:o|º|°|er|vo|no|to)?\.?\s+circuito\b", p)
    if m:
        return int(m.group(1))
    m = re.search(r"\b([ivxl]{1,6})\s+circuito\b|\bcircuito\s+([ivxl]{1,6})\b", p)
    if m:
        return _romano_int(m.group(1) or m.group(2))
    m = re.search(r"((?:[a-z]+\s+){1,2}?)circuito\b", p)
    if m:
        total = sum(_ORDINALES.get(w, 0) for w in m.group(1).split())
        if total:
            return total
    return 0


def sede_de(tribunal: str = "", ciudad: str = "") -> dict:
    """{tribunal, ciudad, circuito (romano o «»), cdmx}. Los Centros Auxiliares
    no tienen circuito propio: su sede la dice la ciudad de residencia."""
    try:
        import banco as _bk
        romano = _bk._romano
    except Exception:  # pragma: no cover
        romano = None
    n = _circuito_num(tribunal)
    circ = (romano(n) if romano else "") if 1 <= n <= 40 else ""
    return {"tribunal": _limpio(tribunal), "ciudad": _limpio(ciudad), "circuito": circ,
            "cdmx": _es_cdmx(tribunal, ciudad, circ)}


def _es_cdmx(tribunal: str, ciudad: str, circ: str) -> bool:
    """UNA SOLA REGLA PARA LA CIUDAD DE MÉXICO (tercera ronda, 3-oct-2026,
    rev_2). `sede_de` y `tipos_asunto.es_cdmx` la decidían cada uno a su modo:
    con «Décimo Tribunal Colegiado en Materia Civil del Primer Circuito» y la
    ciudad «Cd. de México», aquí salía Ciudad de México y en `supletorio` no
    (su expresión no leía «Cd. de México»). La existencia se fundaba en el CFPC
    y la verja, que lee `ficha.sede.cdmx`, acusaba esa misma fórmula: dos sitios
    que deciden el código por su lado acaban citando dos códigos. Manda
    `tipos_asunto.es_cdmx` (Primer Circuito ⇒ Ciudad de México salvo los
    Centros Auxiliares; «Cd. de México», «Cd. Mx.», «México, D.F.»). Lo de
    aquí sólo vale si esa pieza no se puede importar; su «no se puede decidir»
    (None) es False, que es la regla general (CFPC)."""
    try:
        import tipos_asunto as _ta
        r = _ta.es_cdmx(str(tribunal or ""), str(ciudad or ""))
        return bool(r)
    except Exception:
        pass
    return bool(_RX_CDMX.search(_plano(ciudad))) or (
        circ == "I" and "centro auxiliar" not in _plano(tribunal))


# ═══════════════════════════════════════════════════════════════════════════
# EL AUTO DE ADMISIÓN, EL DE TURNO Y EL DE RETURNO — SIN MODELO
# ═══════════════════════════════════════════════════════════════════════════
# Las fórmulas son fijas en todos los colegiados: «fórmese el expediente
# número…», «túrnense los autos al Magistrado…», «retúrnense…», «se tiene al
# agente del Ministerio Público formulando pedimento». Pagar un modelo por eso
# es pagar por la ocasión de que lo reformule.
_NOMBRE = (r"(?:[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+|[A-ZÁÉÍÓÚÑ]{2,}|[A-ZÁÉÍÓÚÑ]\.)"
           r"(?:\s+(?:(?:de|del|de\s+la|de\s+los|y)\s+)?(?:[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+|[A-ZÁÉÍÓÚÑ]{2,}|[A-ZÁÉÍÓÚÑ]\.)){1,6}")
_RX_TURNO = re.compile(r"\bt[uú]rne(?:se|nse)\b|\bse\s+turn(?:a|an|aron|ó)\s+(?:los\s+autos|el\s+asunto|el\s+expediente|el\s+presente)", re.I)
_RX_RETURNO = re.compile(r"\bret[uú]rne(?:se|nse)\b|\bse\s+return(?:a|an|aron|ó)\b|\bordena\s+returnar\b|\breturnar\s+los\s+autos\b", re.I)
# EL AUTO QUE REGISTRA Y EL QUE ADMITE SON DOS COSAS (CUARTA RONDA,
# 3-oct-2026). Antes una sola expresión juntaba «fórmese el expediente»,
# «regístrese» y «se admite»: el auto de una declinatoria o de una prevención
# (RF 7/2025: «radíquese… se declara legalmente incompetente») y el de la queja
# de la fracción II que sólo registra y pide el informe (Q 335/2025) pasaban
# por autos de ADMISIÓN, y su fecha salía en «lo admitió a trámite». Ahora:
#   _RX_REGISTRO — forma, registra o radica el expediente, o requiere el informe
#                  con justificación (art. 101): `registro.fecha`, SÓLO si el
#                  mismo auto no admite.
#   _RX_ADMITE   — admite el asunto o el recurso: `admision.fecha`.
# LA ADMISIÓN DEL ASUNTO, NO LA DEL ADHESIVO: «se admite el amparo adhesivo»
# es otro auto y otra fecha. Y «no se admite» no es admitir.
_RX_REGISTRO = re.compile(
    r"f[oó]rmese\s+(?:el\s+)?(?:expediente|cuaderno|toca)|reg[ií]str(?:ese|ense)|"
    r"se\s+registr[aó]\s+(?:con|bajo)|rad[ií]qu(?:ese|ense)|se\s+radica\b|"
    r"(?:requi[eé]rase|se\s+requiere|p[ií]dase|solic[ií]tese)\s+[^.;]{0,220}?"
    r"informe\s+(?:con\s+)?justific(?:ado|ada|aci[oó]n)", re.I)
_RX_ADMITE = re.compile(
    r"(?<!no\s)(?:\bse\s+admite|\badm[ií]te(?:se|nse)|\badm[ií]tase|\bse\s+tiene\s+por\s+admitid[oa])\s+"
    r"(?:a\s+tr[aá]mite\s+)?(?:(?:la|el)\s+(?:presente\s+)?(?:demanda|recurso|revisi[oó]n|amparo|queja)\b)"
    r"(?![^.;]{0,40}?adhesiv)|"
    r"(?:recurso|demanda)\s+que\s+se\s+admite|"
    r"(?<!no\s)(?:\bse\s+admite|\badm[ií]te(?:se|nse)|\badm[ií]tase)\s+a\s+tr[aá]mite(?![^.;]{0,40}?adhesiv)|"
    r"(?<!no\s)\bse\s+admite\s*[.;]|"
    r"(?<!no\s)\b(?:es\s+procedente|procede)\s+admitir|\bse\s+ordena\s+(?:su\s+admisi[oó]n|admitir)", re.I)
# Lo que dice, tras «por auto de Presidencia de <fecha>», qué hizo ese auto.
_RX_TRAS_PRESIDENCIA_ADMITE = re.compile(r"\badmiti[oó]|\bse\s+admite|\badmitid[oa]|\badmitiendo", re.I)
_RX_TRAS_PRESIDENCIA_REGISTRO = re.compile(
    r"\bform[oó]|\bse\s+form[oó]|\bregistr[oóa]|\bradic[oóa]|\brequiri[oó]|\bse\s+requiere|\bsolicit[oó]", re.I)
_RX_TRIBUNAL_CABECERA = re.compile(
    r"((?:Primer|Segundo|Tercer|Cuarto|Quinto|Sexto|S[ée]ptimo|Octavo|Noveno|D[ée]cimo)[\w\s]{0,30}?\s+"
    r"Tribunal\s+Colegiado[\w\s,áéíóúñÁÉÍÓÚÑ]{5,160}?Circuito)", re.I)
# EL CARGO DEL PONENTE, COMO LO ESCRIBE EL AUTO (CUARTA RONDA, 3-oct-2026). La
# carátula decía «MAGISTRADO PONENTE: JENICA CAMPOS JUÁREZ» (AD 128, 279 y 552;
# Q 342; AR 307, 201, 208, 60 y 72) aunque el auto de returno dice «a la
# ponencia a cargo de la magistrada Jenica Campos Juárez»: el cargo sólo se leía
# con mayúscula inicial y sin «a cargo de». El género no se adivina por el
# nombre de pila: se copia del cargo escrito. «secretaria en funciones de
# Magistrada» (AR 239/2025) también es cargo, y va DETRÁS del nombre.
_RX_PONENTE = re.compile(
    r"(?i:\bt[uú]rne(?:se|nse)|\bret[uú]rne(?:se|nse)|\bse\s+(?:re)?turn(?:a|an|aron|ó)|\breturnar)"
    r"[^.;]{0,160}?\b(?i:al?|a\s+la)\s+(?i:ponencia\s+(?:a\s+cargo\s+)?(?:del?|de\s+la)\s+)?"
    r"(?P<titulo>(?i:magistrad[oa]))?(?:\s+(?i:de\s+circuito))?\s*"
    r"(?:(?i:licenciad[oa]|lic\.|mtr[oa]\.|maestr[oa]|dr[a]?\.|doctor[a]?)\s+)?"
    r"(?:en\s+Derecho\s+)?(?P<nombre>" + _NOMBRE + r")")
_RX_EN_FUNCIONES_TRAS = re.compile(
    r"^\s*,?\s*(?:quien\s+(?:funge|act[uú]a)\s+como\s+)?(?P<s>secretari[oa])\s+"
    r"(?:de\s+(?:tribunal|estudio\s+y\s+cuenta)\s+)?en\s+funciones\s+de\s+(?P<m>magistrad[oa])", re.I)
_RX_MP_SIN = re.compile(
    r"ministerio\s+p[uú]blico[^.;]{0,200}?(?:no\s+formul[oó]|omiti[oó]\s+formular|fue\s+omis[oa]\s+en\s+formular|"
    r"se\s+abstuvo\s+de\s+formular|sin\s+que\s+(?:haya|hubiera|hubiese)\s+formulado|no\s+form(?:ul)?ó)"
    r"\s+(?:el\s+|su\s+)?pedimento|"
    r"sin\s+que\s+el\s+(?:agente\s+del\s+)?ministerio\s+p[uú]blico[^.;]{0,120}?(?:haya|hubiera|hubiese)\s+formulado",
    re.I)
_RX_MP_CON = re.compile(
    r"(?:se\s+tiene\s+al?|t[eé]ngase\s+al?|se\s+tuvo\s+al?)\s+(?:agente\s+del\s+)?ministerio\s+p[uú]blico"
    r"[^.;]{0,160}?(?:formulando|rindiendo|emitiendo)\s+(?:su\s+|el\s+)?pedimento|"
    r"ministerio\s+p[uú]blico[^.;]{0,160}?(?<!no\s)(?:formul[oó]|rindi[oó]|emiti[oó])\s+(?:su\s+|el\s+)?pedimento|"
    r"pedimento\s+(?:n[uú]mero|n[uú]m\.|no\.)\s*[\w/.-]+", re.I)
_TERMINA_NOMBRE = (r"(?=,?\s+(?:en\s+su\s+car[aá]cter|tercer[oa]s?\s+interesad|quejos[oa]|autoridad\s+responsable|"
                   r"por\s+conducto|por\s+(?:su\s+)?propio\s+derecho|interponiendo|promoviendo|formulando|"
                   r"adhiri[eé]ndose|contra\b|en\s+contra\b|quien\b|mediante\b)|\s*[;:]|\.\s+(?-i:[A-ZÁÉÍÓÚÑ])|$)")
_RX_ADHESIVO = [
    re.compile(r"(?:se\s+admite|adm[ií]tese|se\s+tiene\s+por\s+(?:interpuest|promovid|presentad)[oa])\s+(?:el\s+|la\s+)?"
               r"(?:amparo|revisi[oó]n|recurso\s+de\s+revisi[oó]n(?:\s+fiscal)?)\s+adhesiv[oa]\s+"
               r"(?:promovid[oa]|interpuest[oa]|hech[oa]\s+valer|formulad[oa])\s+por\s+(?P<n>.{3,160}?)" + _TERMINA_NOMBRE, re.I | re.S),
    re.compile(r"se\s+tiene\s+a\s+(?P<n>.{3,160}?)\s*,?\s+(?:[^.;]{0,90}?\s)?(?:promoviendo|interponiendo|formulando)\s+"
               r"(?:el\s+)?(?:amparo|revisi[oó]n|recurso\s+de\s+revisi[oó]n(?:\s+fiscal)?)\s+adhesiv", re.I | re.S),
    re.compile(r"se\s+tiene\s+a\s+(?P<n>.{3,160}?)\s*,?\s+(?:[^.;]{0,90}?\s)?adhiri[eé]ndose\s+al\s+recurso", re.I | re.S),
]
_RX_INFORME_RENDIDO = re.compile(
    r"(?:se\s+tiene|t[eé]ngase|se\s+tuvo)\s+(?:a\s+[^.;]{0,140}?)?(?:por\s+)?(?:rendid[oa]|rindiendo)\s+"
    r"(?:su\s+|el\s+)?informe\s+(?:con\s+)?justific|por\s+rendido\s+(?:su\s+|el\s+)?informe\s+(?:con\s+)?justific",
    re.I)
_RX_REPRESENTANTE = re.compile(
    r"por\s+conducto\s+de\s+su\s+(?P<f>apoderad[oa](?:\s+legal)?|representante\s+legal|mandatari[oa]\s+judicial|"
    r"autorizad[oa](?:\s+en\s+t[eé]rminos\s+amplios)?|delegad[oa])\s+(?P<n>" + _NOMBRE + r")")
_RX_RECURRENTE_AUTO = [
    re.compile(r"se\s+tiene\s+a\s+(?P<n>.{3,160}?)\s*,\s*(?P<c>tercer[oa]\s+interesad[oa]|quejos[oa]|"
               r"parte\s+quejosa|autoridad\s+responsable)\s*,?\s+(?:[^.;]{0,60}?\s)?interponiendo\s+(?:el\s+)?recurso", re.I | re.S),
    re.compile(r"recurso\s+de\s+(?:revisi[oó]n(?:\s+fiscal)?|queja)\s+interpuesto\s+por\s+(?P<n>.{3,160}?)" + _TERMINA_NOMBRE, re.I | re.S),
    re.compile(r"demanda\s+de\s+amparo\s+(?:directo\s+)?(?:promovida|presentada)\s+por\s+(?P<n>.{3,160}?)" + _TERMINA_NOMBRE, re.I | re.S),
]
_RX_PRESIDENCIA_DE = re.compile(r"auto\s+de\s+presidencia\s+de\s+(?:fecha\s+)?", re.I)
_RX_CLASE_ACTO = re.compile(
    r"contra\s+(?:de\s+)?(?:la|el)\s+(?P<c>sentencia|resoluci[oó]n|laudo|auto|interlocutoria|acuerdo)"
    r"(?:\s+(?:definitiva|interlocutoria|incidental))?\s+(?:de\s+fecha\s+|dictad[oa]\s+el\s+|emitid[oa]\s+el\s+|"
    r"pronunciad[oa]\s+el\s+|de\s+|del\s+d[ií]a\s+)?", re.I)
_RX_ORGANO_DICTO = re.compile(
    r"(?:dictad[oa]|emitid[oa]|pronunciad[oa])\s+por\s+(?:el|la|los|las)\s+(?P<o>[A-ZÁÉÍÓÚÑ][^;]{8,220}?)"
    r"(?=,\s|\s+en\s+(?:el|los)\s+(?:toca|expediente|juicio|autos|incidente|cuaderno)|\s+dentro\s+de|\.\s|;|$)")
_RX_TOCA = re.compile(r"\btoca\s+(?:(?:civil|familiar|penal|mercantil|administrativo|de\s+apelaci[oó]n)\s+)?"
                      r"(?:n[uú]mero\s+)?(?P<n>\d{1,6}\s*/\s*\d{4})", re.I)
_RX_EXPTE = re.compile(
    r"\b(?:expediente|juicio\s+(?:de\s+amparo\s+(?:indirecto\s+)?|oral\s+mercantil\s+|ordinario\s+\w+\s+|"
    r"contencioso\s+administrativo\s+(?:federal\s+)?|de\s+nulidad\s+|agrario\s+|sumario\s+\w+\s+)?|"
    r"amparo\s+indirecto)\s*(?:n[uú]mero\s+|n[uú]m\.\s*|no\.\s*)?(?P<n>\d{1,6}\s*/\s*\d{2,4}(?:-[\w-]{1,20})?)", re.I)


def _encabezados_de_auto(texto: str) -> list:
    """Posiciones donde empieza un auto: «Ciudad, Estado, a <fecha>». LINEAL
    (tercera ronda): las vistas del texto se calculan una vez, no por fecha."""
    t = str(texto or "")
    vistas = _vistas_proemio(t)
    fuera = []
    for iso, _lit, pos in fechas_del_texto(t):
        if _es_proemio(t, pos, vistas):
            fuera.append((pos, iso))
    return fuera


def _fecha_del_auto(texto: str) -> str:
    """La fecha del auto: la de su encabezado «Ciudad, Estado, a…» (la que
    sigue a «Conste.» si hay razón de cuenta); si no hay encabezado, la primera."""
    t = str(texto or "")
    plano = " ".join(_sin_tildes(t).lower().split())
    enc = _encabezados_de_auto(t)
    if enc:
        i_conste = plano.find("conste")
        if i_conste >= 0:
            for pos, iso in enc:
                if pos > i_conste:
                    return iso
        return enc[0][1]
    fs = fechas_del_texto(t)
    return fs[0][0] if fs else ""


def _trozos(texto: str) -> list:
    """Un papel con varios autos pegados se parte por sus encabezados."""
    t = " ".join(str(texto or "").split())
    plano = " ".join(_sin_tildes(t).lower().split())
    enc = _encabezados_de_auto(t)
    if len(enc) < 2 or len(plano) != len(t):
        return [t]
    cortes = []
    for pos, _iso in enc:
        # Se corta al principio de la frase del encabezado (la ciudad), no en la fecha.
        ini = max(t.rfind(". ", 0, pos), t.rfind("Conste", 0, pos))
        cortes.append(max(0, ini + 1) if ini >= 0 else 0)
    cortes = sorted(set(cortes))
    if cortes[0] != 0:
        cortes = [0] + cortes
    return [t[a:b].strip() for a, b in zip(cortes, cortes[1:] + [len(t)]) if t[a:b].strip()]


def _nombre_limpio(n: str) -> str:
    n = _limpio(n).strip(" ,;:")
    if n.endswith(".") and not re.search(r"(?:\b\w\.){2,}$|\b(?:C|V|L|R|S|A)\.$", n):
        n = n[:-1].rstrip()
    # El artículo en minúscula es de la frase, no del nombre («la Unión de
    # Trabajadores…»); «La Romita» con mayúscula sí es nombre.
    # SALVO ANTE UN CARGO DE DOBLE GÉNERO (CUARTA RONDA, 3-oct-2026): en
    # «interpuesto por la Titular de la Unidad Jurídica…» el artículo es lo
    # único que dice el género, y el compositor lo conserva («la Titular») si
    # llega. Quitarlo dejaba «Titular…», que la prosa escribe «el Titular» (RF 4,
    # 6 y 49/2025 y RF 21: el engrose dice «la Titular»).
    if not _RX_ART_CARGO_DOBLE.match(n):
        n = re.sub(r"^(?:el|la|los|las|a|al)\s+", "", n)
    return n if 3 <= len(n) <= 160 else ""


# Cargos que sirven igual para hombre y mujer: su artículo es su género.
_RX_ART_CARGO_DOBLE = re.compile(
    r"^(?:el|la)\s+(?:titular|oficial\s+mayor|fiscal|juez|representante|agente)\b", re.I)


def _ponente_limpio(n: str) -> str:
    n = _limpio(n).strip(" ,.;:")
    try:
        import fase_autos as _fa
        if _fa._CARGOS.match(n):
            return ""
    except Exception:
        pass
    return n if len(n.split()) >= 2 else ""


# ── EL CARGO DEL PONENTE: SE COPIA, NO SE ADIVINA (cuarta ronda, 3-oct-2026) ──
# Tres formas, todas del papel o de lo que tecleó el secretario: «Magistrada»,
# «Magistrado» y «Secretaria (o Secretario) en funciones de Magistrada (o
# Magistrado)». `documento_generado` las lee para el rótulo de la carátula
# («MAGISTRADA PONENTE», «MAGISTRADO PONENTE», «PONENTE») y el compositor para
# la prosa del turno. Sin cargo escrito, «» —y el rótulo va por omisión con
# su aviso—: el nombre de pila no dice el género.
_RX_CARGO_DELANTE = re.compile(r"^\s*(?:(?:el|la)\s+)?(?P<m>magistrad[oa])(?:\s+de\s+circuito)?\b\.?\s*", re.I)
_RX_FUNCIONES_EN_VALOR = re.compile(
    r",?\s*\(?\s*(?:(?:el|la)\s+)?(?P<s>secretari[oa])\s+(?:de\s+(?:tribunal|estudio\s+y\s+cuenta)\s+)?"
    r"en\s+funciones\s+de\s+(?P<m>magistrad[oa])(?:\s+de\s+circuito)?\s*\)?", re.I)


def _titulo_de(valor) -> str:
    """El cargo en su forma del esquema, o «» si no es un cargo de ponente."""
    v = _limpio(valor if isinstance(valor, str) else "")
    m = _RX_FUNCIONES_EN_VALOR.search(v)
    if m:
        return f"{m.group('s').capitalize()} en funciones de {m.group('m').capitalize()}"
    m = _RX_CARGO_DELANTE.match(v)
    if m and not v[m.end():].strip(" .,;"):
        return m.group("m").capitalize()
    return ""


def partir_ponente(valor) -> tuple:
    """(nombre, cargo) de lo que se teclea como ponente: «Magistrada Jenica
    Campos Juárez» → («Jenica Campos Juárez», «Magistrada»); «Bertha Martínez
    Vega, secretaria en funciones de Magistrada» → («Bertha Martínez Vega»,
    «Secretaria en funciones de Magistrada»). Sin cargo escrito, («…», «»)."""
    v = _limpio(valor if isinstance(valor, str) else "")
    m = _RX_FUNCIONES_EN_VALOR.search(v)
    if m:
        nombre = (v[:m.start()] + " " + v[m.end():]).strip(" ,.;")
        return " ".join(nombre.split()), _titulo_de(m.group(0))
    m = _RX_CARGO_DELANTE.match(v)
    if m and v[m.end():].strip(" .,;"):
        return v[m.end():].strip(" ,;"), m.group("m").capitalize()
    return v, ""


def con_titulo(ponente: str, titulo: str) -> str:
    """Lo inverso de `partir_ponente`, para el formulario: el cargo delante
    («Magistrada X») o, el de la secretaria en funciones, detrás («X, secretaria
    en funciones de Magistrada»). Si el valor ya trae un cargo, no se toca."""
    p, t = _limpio(ponente), _titulo_de(titulo)
    if not p or not t or partir_ponente(p)[1]:
        return p
    if "funciones" in t:
        return f"{p}, {t[:1].lower()}{t[1:]}"
    return f"{t} {p}"


def _titulo_en_papel(titulo: str, ponente: str, papel) -> bool:
    """¿El papel escribe ESE cargo junto al nombre del ponente? «magistrada»
    no vale por «magistrado» (el parecido de 0.9 cambiaría el género)."""
    t = _titulo_de(titulo)
    nombre = _plano(ponente)
    if not t or not nombre:
        return False
    plano = _papel(papel).plano
    clave = r"\bsecretari%s\s+[^.;]{0,40}?en\s+funciones\s+de\s+magistrad%s\b" % (
        _plano(t).split()[0][-1], _plano(t).split()[-1][-1]) if "funciones" in t else \
        r"\bmagistrad%s\b" % _plano(t)[-1]
    i = plano.find(nombre)
    while i >= 0:
        if re.search(clave, plano[max(0, i - 90):i + len(nombre) + 120]):
            return True
        i = plano.find(nombre, i + 1)
    return False


def _poner(d: dict, ruta: str, valor) -> None:
    partes = ruta.split(".")
    for p in partes[:-1]:
        d = d.setdefault(p, {})
    d[partes[-1]] = valor


def _tomar(d, ruta: str):
    for p in ruta.split("."):
        if not isinstance(d, dict):
            return None
        d = d.get(p)
    return d


def _leer_un_auto(t: str, tipo: str, avisos: list) -> dict:
    import fase_autos as _fa
    import fase_admision as _fad
    out: dict = {}
    plano = " ".join(t.split())
    base = _fa.leer(plano)
    det = _fad.deterministas(plano)
    if det.get("aviso_numero"):
        avisos.append(det["aviso_numero"])
    fecha = _fecha_del_auto(plano)
    es_adm = bool(_RX_ADMITE.search(plano))
    es_reg = bool(_RX_REGISTRO.search(plano))
    es_ret = bool(_RX_RETURNO.search(plano))
    # «retúrnense» no casa con «túrnense» (no hay límite de palabra entre la
    # «e» y la «t»), así que las dos marcas se miran por separado.
    es_tur = bool(_RX_TURNO.search(plano))
    out["lectura"] = {"fecha_auto": fecha, "es_admision": es_adm, "es_registro": es_reg,
                      "es_turno": es_tur, "es_returno": es_ret}
    if det.get("numero") or base.get("numero"):
        out["numero"] = det.get("numero") or base.get("numero")
    if det.get("tipo_asunto"):
        out["tipo"] = det["tipo_asunto"]
    if det.get("tribunal"):
        _poner(out, "sede.tribunal", det["tribunal"])
    else:
        m = _RX_TRIBUNAL_CABECERA.search(plano[:800])
        if m:
            trib = _limpio(m.group(1))
            if trib.isupper():
                _min = {"en", "de", "del", "y", "la", "el", "los", "las", "e"}
                trib = " ".join(w.lower() if w.lower() in _min and i else w.capitalize()
                                for i, w in enumerate(trib.split()))
            _poner(out, "sede.tribunal", trib)
    if det.get("ciudad"):
        _poner(out, "sede.ciudad", det["ciudad"])
    if base.get("presentacion"):
        out["presentacion"] = base["presentacion"]

    # ── REGISTRO Y ADMISIÓN: el propio auto, o «por auto de Presidencia de…» ──
    # CUARTA RONDA (3-oct-2026): `admision` es el auto que ADMITE; el que sólo
    # forma, registra o radica el expediente —o, en la queja de la fracción II,
    # el que además requiere el informe del 101— va a `registro`. Si el mismo
    # auto registra y admite (lo normal en el amparo directo), es UNO: sólo
    # `admision`. El auto que tiene por rendido el informe va además a
    # `informe_101` (SPEC §2.2, queja), y si admite, también es la admisión.
    informe = bool(_RX_INFORME_RENDIDO.search(plano))
    if es_adm and fecha:
        _poner(out, "admision.fecha", fecha)
    elif es_reg and fecha and not informe and not es_ret:
        _poner(out, "registro.fecha", fecha)
    for m in _RX_PRESIDENCIA_DE.finditer(plano):
        f = fechas_del_texto(plano[m.end():m.end() + 90])
        if not f or f[0][2] > 12:            # la fecha va pegada a «auto de Presidencia de»
            continue
        # Desde la fecha misma: el literal de la fecha puede tragarse la palabra
        # que sigue al año («…veinticinco se requirió»), y en la fecha no hay
        # verbos.
        tras = plano[m.end() + f[0][2]:m.end() + f[0][2] + 220]
        # Lo que hizo ESE auto es el primer verbo tras su fecha: «…de catorce
        # de octubre se requirió el informe…; se admite el recurso» (Q 335) dice
        # que el de octubre REQUIRIÓ; el «se admite» es del auto que se lee.
        m_adm = _RX_TRAS_PRESIDENCIA_ADMITE.search(tras)
        m_reg = _RX_TRAS_PRESIDENCIA_REGISTRO.search(tras)
        if m_reg and (not m_adm or m_reg.start() < m_adm.start()):
            ruta = "registro.fecha"
        else:
            # Admitió, o ningún verbo lo dice: lo de siempre, el auto de
            # Presidencia es la admisión.
            ruta = "admision.fecha"
        if not _tomar(out, ruta) and f[0][0] != _tomar(out, "admision.fecha"):
            _poner(out, ruta, f[0][0])

    # ── TURNO Y RETURNO, con su ponente ──
    # «TÚRNESE a la ponencia A MI CARGO»: el ponente es quien firma como
    # Presidente, y eso ya lo sabe leer `fase_autos`.
    mp = None if re.search(r"ponencia\s+a\s+mi\s+cargo", plano, re.I) else _RX_PONENTE.search(plano)
    ponente = _ponente_limpio(mp.group("nombre")) if mp else ""
    titulo = (mp.group("titulo") or "").capitalize() if mp else ""
    if mp and not titulo:
        mf = _RX_EN_FUNCIONES_TRAS.match(plano[mp.end():mp.end() + 120])
        if mf:
            titulo = (f"{mf.group('s').capitalize()} en funciones de "
                      f"{mf.group('m').capitalize()}")
    if not ponente and (es_tur or es_ret):
        ponente = _ponente_limpio(base.get("magistrado") or "")
    if es_ret and fecha:
        _poner(out, "returno.fecha", fecha)
        if ponente:
            _poner(out, "returno.ponente", ponente)
            if titulo:
                _poner(out, "returno.titulo", titulo)
    elif es_tur and fecha:
        _poner(out, "turno.fecha", fecha)
        if ponente:
            _poner(out, "turno.ponente", ponente)
            if titulo:
                _poner(out, "turno.titulo", titulo)

    # ── MINISTERIO PÚBLICO: SÓLO SI CONSTA ──
    # «Dese vista al Ministerio Público para que formule pedimento» NO dice
    # que lo formuló ni que no: no se escribe nada. Lo negativo se mira antes
    # porque «no formuló pedimento» contiene «formuló pedimento».
    if _RX_MP_SIN.search(plano):
        out["ministerio_publico"] = "sin_pedimento"
    elif _RX_MP_CON.search(plano):
        out["ministerio_publico"] = "pedimento"

    # ── EL ADHESIVO ──
    for rx in _RX_ADHESIVO:
        m = rx.search(plano)
        if not m:
            continue
        quien = _nombre_limpio(m.group("n"))
        if not quien:
            continue
        _poner(out, "adhesivo.quien", quien)
        if fecha and re.search(r"se\s+admite|adm[ií]tese|se\s+tiene", m.group(0), re.I):
            _poner(out, "adhesivo.admision", fecha)
        frase = plano[max(0, m.start() - 200):m.end() + 200]
        mp_ = re.search(r"presentad[oa]\s+(?:el\s+|en\s+fecha\s+)?", frase, re.I)
        if mp_:
            f = fechas_del_texto(frase[mp_.end():mp_.end() + 80])
            if f:
                _poner(out, "adhesivo.presentacion", f[0][0])
        break

    # ── EL INFORME CON JUSTIFICACIÓN (queja, fracción II) ──
    if fecha and informe:
        _poner(out, "informe_101.fecha", fecha)

    # ── QUIÉN PROMUEVE O RECURRE, Y POR CONDUCTO DE QUIÉN ──
    for rx in _RX_RECURRENTE_AUTO:
        m = rx.search(plano)
        if not m:
            continue
        n = _nombre_limpio(m.group("n"))
        if not n:
            continue
        out["promovente"] = n
        c = _plano(m.groupdict().get("c") or "")
        if c:
            out["caracter"] = ("tercero" if c.startswith("tercer") else "autoridad"
                               if c.startswith("autoridad") else "quejoso")
        break
    if not out.get("promovente") and base.get("recurrente"):
        out["promovente"] = base["recurrente"]
    m = _RX_REPRESENTANTE.search(plano)
    if m:
        out["figura_representante"] = _limpio(m.group("f")).lower()
        out["representante"] = _limpio(m.group("n"))

    # ── LO RECLAMADO O RECURRIDO, SI EL AUTO LO DICE ──
    m = _RX_CLASE_ACTO.search(plano)
    if m:
        clase = _plano(m.group("c")).replace("resolucion", "resolucion")
        _poner(out, "acto.clase", {"acuerdo": "auto"}.get(clase, clase))
        f = fechas_del_texto(plano[m.end():m.end() + 80])
        if f and f[0][2] <= 14:              # «contra la sentencia de <fecha>», pegada
            _poner(out, "acto.fecha", f[0][0])
        tramo = plano[m.start():m.start() + 600]
        mo = _RX_ORGANO_DICTO.search(tramo)
        if mo:
            org = _limpio(mo.group("o"))
            # La coma corta «Juzgado… de Amparo Civil, Administrativo y de
            # Trabajo…»: el nombre entero, si el lector de órganos lo ve.
            enteros = _organos_en(tramo[mo.start("o"):mo.start("o") + 400])
            if enteros and _misma_base(org, enteros[0]) and len(enteros[0]) > len(org):
                org = enteros[0]
            ok_, _mot = es_organo(org, jurisdiccional=False)
            if ok_:
                _poner(out, "acto.organo", org)
    propio = re.sub(r"\s+", "", str(out.get("numero") or ""))
    mt = _RX_TOCA.search(plano)
    if mt:
        _poner(out, "acto.toca", re.sub(r"\s+", "", mt.group("n")))
    for me in _RX_EXPTE.finditer(plano):
        n = re.sub(r"\s+", "", me.group("n"))
        antes = plano[max(0, me.start() - 25):me.start()].lower()
        if n == propio or re.search(r"f[oó]rmese\s+(?:el\s+)?$", antes):
            continue
        if mt and n == re.sub(r"\s+", "", mt.group("n")):
            continue
        _poner(out, "acto.expediente", n)
        break
    if base.get("expediente_origen") and not _tomar(out, "acto.expediente"):
        _poner(out, "acto.expediente", base["expediente_origen"])
    if re.search(r"incidente\s+de\s+suspensi[oó]n|cuaderno\s+incidental", plano, re.I):
        _poner(out, "acto.incidente", True)

    # ── LA VÍA DE PRESENTACIÓN ──
    if re.search(r"portal\s+de\s+servicios\s+en\s+l[ií]nea|firma\s+electr[oó]nica|v[ií]a\s+electr[oó]nica", plano, re.I):
        out["via_presentacion"] = "electronica"
    elif re.search(r"servicio\s+postal|correo\s+certificado|correos\s+de\s+m[eé]xico", plano, re.I):
        out["via_presentacion"] = "postal"
    return out


def fundir_lecturas(*lecturas, rotulo: str = "auto") -> dict:
    """Varias lecturas en una: gana la primera que trae el dato; si otra lo
    trae DISTINTO (una fecha, un número, el ponente), se conserva el primero y
    se avisa. Una contradicción entre constancias es cosa del secretario."""
    fin: dict = {"avisos": []}
    for lec in lecturas:
        if not isinstance(lec, dict):
            continue
        for a in lec.get("avisos") or []:
            if a not in fin["avisos"]:
                fin["avisos"].append(a)
        for ruta, v in _hojas(lec):
            if ruta.startswith(("avisos", "fuentes", "lectura")):
                continue
            ya = _tomar(fin, ruta)
            if _vacio(ya):
                if not _vacio(v):
                    _poner(fin, ruta, v)
            # LO QUE RESOLVIÓ EL JUZGADO TAMBIÉN (4-oct-2026, AR 380/2025): si el
            # lector de los puntos y el modelo no dicen lo mismo, de eso depende
            # confirmar o revocar, sobreseer o negar; se avisa, no se elige en
            # silencio.
            elif not _vacio(v) and ya != v and (_es_ruta_fecha(ruta) or ruta in (
                    "numero", "turno.ponente", "returno.ponente", "acto.toca", "acto.expediente",
                    "responsable", "acto.resolvio", "acto.resolvio_mixto")) \
                    and not (ruta == "responsable" and _misma_base(ya, v)):
                av = (f"DOS CONSTANCIAS NO DICEN LO MISMO sobre {ruta.replace('.', ' ')}: "
                      f"«{_dmy(ya) if _es_ruta_fecha(ruta) else ya}» y "
                      f"«{_dmy(v) if _es_ruta_fecha(ruta) else v}». Se tomó el primero; "
                      f"compruébalo en el {rotulo}.")
                if av not in fin["avisos"]:
                    fin["avisos"].append(av)
    lects = [lec.get("lectura") for lec in lecturas if isinstance(lec, dict) and lec.get("lectura")]
    if lects:
        fin["lectura"] = lects[0] if len(lects) == 1 else lects
    return fin


def leer_auto(texto: str, tipo: str = "") -> dict:
    """Lo que el auto de admisión, el de turno o el de returno dicen, SIN
    MODELO. Devuelve una ficha parcial (sólo lo que se leyó) con «avisos». Un
    papel con varios autos pegados se lee auto por auto.

    SÍNCRONA, PURA Y ACOTADA (tercera ronda, 3-oct-2026, rev_6): no toca estado
    global, así que quien la llama desde una ruta async la manda a un hilo
    (`asyncio.to_thread`); lee los primeros `TOPE_AUTO` caracteres y avisa si
    recortó."""
    t = _RX_CONTROL.sub("", str(texto or ""))
    if not t.strip():
        return {"avisos": []}
    tp = _tipo(tipo)
    recorte = []
    if len(t) > TOPE_AUTO:
        _mil = lambda x: f"{x:,}".replace(",", " ")      # «498 000»
        recorte.append(
            f"EL AUTO QUE SUBISTE ES MUY LARGO ({_mil(len(t))} caracteres): se leyeron los primeros "
            f"{_mil(TOPE_AUTO)}, donde va el auto de admisión. Si subiste el expediente entero, la "
            f"fecha del turno o del returno pudo quedar fuera: compruébalas en «Trámite en este "
            f"tribunal».")
        t = t[:TOPE_AUTO]
    lecturas = []
    for trozo in _trozos(t):
        av: list = []
        try:
            lec = _leer_un_auto(trozo, tp, av)
        except Exception as ex:  # una lectura rota no tumba la ficha
            lec = {}
            av.append(f"No se pudo leer un auto: {type(ex).__name__}.")
        lec["avisos"] = av
        lecturas.append(lec)
    fin = fundir_lecturas(*lecturas)
    # El auto que registra con la fecha del que admite es el mismo auto.
    if _tomar(fin, "registro.fecha") and _tomar(fin, "registro.fecha") == _tomar(fin, "admision.fecha"):
        fin.pop("registro", None)
    if recorte:
        fin["avisos"] = recorte + [a for a in fin.get("avisos") or [] if a not in recorte]
    if tp and not fin.get("tipo"):
        fin["tipo"] = tp
    fin["fuentes"] = {r: "auto" for r, v in _hojas(fin)
                      if not r.startswith(("avisos", "fuentes", "lectura", "tipo"))}
    return fin


def _prompt_auto(texto: str, tipo: str) -> str:
    return f"""Eres el secretario de un Tribunal Colegiado leyendo los AUTOS de un
asunto ({tipo or "tipo por determinar"}): el auto de Presidencia que lo registra
y admite, el de turno, el de returno o el que admite un adhesivo.

Reglas que no se negocian:
- COPIA LITERAL de nombres y números, tal como están escritos.
- FECHAS en AAAA-MM-DD, y sólo si están escritas en el papel.
- SI NO CONSTA, CADENA VACÍA. Esto se firma.
- MINISTERIO PÚBLICO: «pedimento» sólo si el auto dice que lo FORMULÓ;
  «sin_pedimento» sólo si dice que NO lo formuló; si sólo le da vista, vacío.

LOS AUTOS:
──────────────────────────────────────────
{texto[:20000]}
──────────────────────────────────────────

Devuelve JSON y nada más:
{{"numero": "el que ordena formar el expediente, o vacío",
 "fecha_admision": "fecha del auto que ADMITE el asunto o el recurso",
 "fecha_registro": "fecha del auto de Presidencia que forma y registra el expediente (o lo radica, o requiere el informe con justificación) SÓLO si es un auto DISTINTO del que lo admite; si es el mismo, vacío",
 "fecha_turno": "fecha del auto que turna a ponencia", "ponente_turno": "nombre literal, sin el cargo",
 "titulo_turno": "el cargo del ponente tal como lo escribe el auto (Magistrada, Magistrado, Secretaria en funciones de Magistrada…), o vacío si no lo dice",
 "fecha_returno": "", "ponente_returno": "", "titulo_returno": "",
 "ministerio_publico": "pedimento | sin_pedimento | vacío",
 "adhesivo_quien": "quien promueve el amparo o la revisión adhesiva, literal",
 "adhesivo_presentacion": "", "adhesivo_admision": "",
 "fecha_informe_101": "fecha del auto que tiene por rendido el informe con justificación",
 "fecha_acto": "fecha de la resolución reclamada o recurrida", "organo_acto": "quien la dictó, literal",
 "toca": "", "expediente_origen": "", "presentacion": "fecha de presentación del escrito"}}"""


async def leer_auto_modelo(cliente, texto: str, tipo: str = "") -> dict:
    """Los mismos campos que `leer_auto`, leídos con el modelo de la admisión
    (`fase_admision.MODELO_ADMISION`) y ANCLADOS al papel: una fecha que no está
    escrita, o un nombre que no aparece, se tira con aviso."""
    out: dict = {"avisos": []}
    t = _RX_CONTROL.sub("", str(texto or ""))[:TOPE_AUTO]
    if not cliente or not t.strip():
        return out
    try:
        import fase_admision as _fad
        import llamada_modelo as _lm
        kw = dict(model=_fad.MODELO_ADMISION,
                  messages=[{"role": "user", "content": _prompt_auto(t, _tipo(tipo))}],
                  max_completion_tokens=1500, response_format={"type": "json_object"})
        if ESFUERZO_ACTO:
            kw["reasoning_effort"] = ESFUERZO_ACTO
        r = await asyncio.wait_for(_lm.crear(cliente, **kw), timeout=TOPE_SEGUNDOS_MODELO)
        d = _json_de(r)
    except asyncio.TimeoutError:
        out["avisos"].append(f"LA LECTURA DE LOS AUTOS CON MODELO NO CONTESTÓ EN {TOPE_SEGUNDOS_MODELO} s: "
                             f"vale lo que se leyó sin modelo.")
        return out
    except Exception as ex:
        out["avisos"].append(f"No se pudieron leer los autos con el modelo: {type(ex).__name__}. "
                             f"Vale lo que se leyó sin modelo.")
        return out
    pp = Papel(t)
    # Sólo texto: lo que el modelo devuelve en otra forma (una lista, un
    # número) no es un dato del auto. Y los asuntos relacionados NUNCA salen
    # de un papel (sexta ronda: sólo los marca el secretario).
    capa = de_formulario({k: v for k, v in d.items()
                          if k in _MAPA_FORMULARIO and k != "relacionados" and isinstance(v, str)})
    # EL CARGO DEL PONENTE (cuarta ronda): en su campo, o dentro del nombre (que
    # `de_formulario` ya separó). Se cree sólo si está escrito junto al nombre.
    for sec in ("turno", "returno"):
        crudo = d.get(f"titulo_{sec}")
        tit = _titulo_de(crudo) if isinstance(crudo, str) else ""
        if tit and not _tomar(capa, f"{sec}.titulo"):
            _poner(capa, f"{sec}.titulo", tit)
    for ruta, v in _hojas(capa):
        if ruta.startswith(("avisos", "fuentes")):
            continue
        if ruta in ("turno.titulo", "returno.titulo"):
            pon = _tomar(capa, ruta.replace("titulo", "ponente"))
            if _titulo_en_papel(v, pon, pp):
                _poner(out, ruta, v)
            else:
                out["avisos"].append(f"SE DESCARTÓ el cargo «{v}» del ponente ({ruta.split('.')[0]}): no está "
                                     f"escrito junto a su nombre en los autos.")
            continue
        if _es_ruta_fecha(ruta):
            if fecha_en_papel(v, pp):
                _poner(out, ruta, v)
            else:
                out["avisos"].append(f"SE DESCARTÓ la fecha {_dmy(v)} como {ruta.replace('.', ' ')}: "
                                     f"no está escrita en los autos.")
        elif ruta in ("acto.toca", "acto.expediente", "numero"):
            if _numero_en_papel(v, pp):
                _poner(out, ruta, v)
        elif ruta == "ministerio_publico":
            # El MP se cree sólo si el papel lo dice con su fórmula.
            if (v == "sin_pedimento" and _RX_MP_SIN.search(t)) or (v == "pedimento" and _RX_MP_CON.search(t)):
                _poner(out, ruta, v)
        elif isinstance(v, str):
            a, _m = anclar(v, pp)
            if a:
                _poner(out, ruta, a)
            else:
                out["avisos"].append(f"SE DESCARTÓ «{v[:80]}» como {ruta.replace('.', ' ')}: no aparece "
                                     f"en los autos.")
    for k in ("numero", "presentacion"):
        v = _limpio(d.get(k)) if isinstance(d.get(k), str) else ""
        if k == "numero" and v and _numero_en_papel(v, pp):
            out["numero"] = re.sub(r"\s+", "", v)
        if k == "presentacion" and v and fecha_en_papel(v, pp):
            out["presentacion"] = a_iso(v)
    # Un cargo sin su ponente no es de nadie; un «registro» con la fecha de la
    # admisión es el mismo auto.
    for sec in ("turno", "returno"):
        if _tomar(out, f"{sec}.titulo") and not _tomar(out, f"{sec}.ponente"):
            out[sec].pop("titulo", None)
    if _tomar(out, "registro.fecha") and _tomar(out, "registro.fecha") == _tomar(out, "admision.fecha"):
        out.pop("registro", None)
    out["fuentes"] = {r: "auto" for r, v in _hojas(out) if not r.startswith(("avisos", "fuentes"))}
    return out


async def leer_auto_completo(cliente, texto: str, tipo: str = "") -> dict:
    """`leer_auto` y, si hay cliente, `leer_auto_modelo` para lo que faltó.
    Lo leído sin modelo manda: es fórmula fija, no interpretación. La lectura
    sin modelo corre en un hilo: no para el bucle de eventos."""
    sin = await asyncio.to_thread(leer_auto, texto, tipo)
    con = await leer_auto_modelo(cliente, texto, tipo) if cliente else {}
    fin = fundir_lecturas(sin, con)
    fin["fuentes"] = {r: "auto" for r, v in _hojas(fin)
                      if not r.startswith(("avisos", "fuentes", "lectura", "tipo"))}
    return fin


# ═══════════════════════════════════════════════════════════════════════════
# LOS DERECHOS QUE LA DEMANDA DICE VIOLADOS
# ═══════════════════════════════════════════════════════════════════════════
# El resultando «Derechos humanos que se estiman vulnerados» está en 92-97% de
# los amparos directos del tribunal y sale del capítulo de la demanda (art.
# 175, fr. VI, de la Ley de Amparo). El modelo no tenía la demanda y lo
# inventaba u omitía. Medido en las 24 demandas del banco Kingston, el capítulo
# se rotula de nueve maneras —«PRECEPTOS CONSTITUCIONALES VIOLADOS», «GARANTÍAS
# VIOLADAS», «Los preceptos que… contengan los derechos humanos cuya violación
# se reclame», «se violaron en mi perjuicio los artículos…»— y a veces mezcla la
# Ley de Amparo o la Convención: sólo cuentan los de la Constitución.
_RX_CAPITULO_DERECHOS = re.compile(
    r"(?:preceptos?|garantias?|derechos?)[^.:]{0,120}?(?:violad|vulnerad|violentad|transgredid|infringid|"
    r"cuya\s+violacion\s+se\s+reclam)\w*|"
    r"se\s+(?:viola|violan|violaron|violo|vulnera|vulneran|vulneraron|transgrede|transgreden)\s+"
    r"(?:en\s+(?:mi|su|nuestro|nuestra)\s+perjuicio|en\s+perjuicio\s+de)")
_RX_FIN_CAPITULO = re.compile(
    r"\b(?:conceptos?\s+de\s+violacion|antecedentes|leyes?\s+(?:secundarias|aplicadas)|"
    r"derechos\s+sustantivos|preceptos\s+no\s+aplicados|bajo\s+protesta|hechos\b|"
    r"(?:vii|viii|ix|x)\s*[.-])")
_RX_LISTA_ARTICULOS = re.compile(
    r"\b(?:articulos?|preceptos?|numerales?|arts?\.)\s+(?P<l>(?:\d{1,3}\s*(?:o|º|°)?\.?(?:\s*(?:[a-z]{0,3}\s*parrafo|"
    r"(?:primer|segundo|tercer|cuarto|quinto)\s+parrafo|parrafo\s+\w+|fraccion(?:es)?\s+[ivxl]+(?:\s*(?:,|y)\s*[ivxl]+)*))?"
    r"(?:\s*(?:,|\by\b|\be\b|/))*\s*)+)")
_RX_DESPUES_CONSTITUCION = re.compile(
    r"^\W{0,4}(?:\w+\W+){0,6}?(?:de\s+la\s+)?(?:constitucion|constitucional|carta\s+magna|ley\s+fundamental|"
    r"pacto\s+federal)")
_RX_DESPUES_OTRA_LEY = re.compile(
    r"^\W{0,4}(?:\w+\W+){0,4}?(?:de\s+la\s+|del\s+|de\s+los\s+)(?:ley|codigo|convencion|declaracion|"
    r"pacto(?!\s+federal)|tratado|reglamento|acuerdo)")


def _arts_de_lista(lista: str) -> list:
    fuera = []
    for m in re.finditer(r"\d{1,3}", lista):
        despues = lista[m.end():m.end() + 14]
        if re.match(r"\s*(?:er|ro|do|to|vo|no)\b|\s*(?:er\s+)?parrafo", despues):
            continue
        n = int(m.group(0))
        if 1 <= n <= 136:
            fuera.append(f"{n}o." if n < 10 else str(n))
    return fuera


def derechos_de(texto_demanda: str) -> list:
    """Los artículos constitucionales que la demanda de amparo directo dice
    violados, en el orden en que los dice («1o.», «14», «16», «17»). [] si la
    demanda no tiene ese capítulo: el compositor deja el hueco y lo avisa."""
    p = " ".join(_sin_tildes(texto_demanda or "").lower().split())
    if not p:
        return []
    fuera: list = []
    for m in _RX_CAPITULO_DERECHOS.finditer(p):
        cab = m.group(0)
        tramo = p[m.end():m.end() + 700]
        # El rótulo del art. 175, fr. VI, nombra «el artículo 1o. de la Ley de
        # Amparo»: eso es el rótulo, no la lista.
        tramo = re.sub(r"^[^:.]{0,140}?ley\s+de\s+amparo[^:.]{0,80}?[:.]", " ", tramo) \
            if "ley de amparo" in tramo[:200] and ":" in tramo[:260] else tramo
        f = _RX_FIN_CAPITULO.search(tramo, 20)
        if f:
            tramo = tramo[:f.start()]
        rotulo_constitucional = bool(re.search(r"constitucion", cab))
        listas = []
        for ml in _RX_LISTA_ARTICULOS.finditer(tramo):
            despues = tramo[ml.end():ml.end() + 120]
            otra = bool(_RX_DESPUES_OTRA_LEY.search(despues)) and not _RX_DESPUES_CONSTITUCION.search(despues[:40])
            const = (not otra) and bool(_RX_DESPUES_CONSTITUCION.search(despues) or rotulo_constitucional)
            listas.append([ml, const, otra])
        # LAS LISTAS ENCADENADAS: «los artículos 1o., 14, 16 y 17 en relación
        # con los artículos 39 y 133 de la Constitución» (ADC 529/2024): la
        # Constitución nombrada al final vale para las dos.
        for i in range(len(listas) - 2, -1, -1):
            ml, const, otra = listas[i]
            hueco = tramo[ml.end():listas[i + 1][0].start()]
            if not const and not otra and listas[i + 1][1] and re.fullmatch(
                    r"\W*(?:en\s+relacion\s+con|y|asi\s+como|ademas\s+de)?\s*(?:los|el|las|la)?\s*"
                    r"(?:diversos?\s+)?", hueco):
                listas[i][1] = True
        for ml, const, _otra in listas:
            if not const:
                continue
            arts = _arts_de_lista(ml.group("l"))
            if any(a in ("103", "107") for a in arts):
                continue          # el fundamento de la demanda, no lo violado
            for a in arts:
                if a not in fuera:
                    fuera.append(a)
        if fuera:
            return fuera
    return fuera


# ═══════════════════════════════════════════════════════════════════════════
# EL SENTIDO, DE UN CATÁLOGO (SPEC §6)
# ═══════════════════════════════════════════════════════════════════════════
# NO UN TRAMO LIBRE. Una frase copiada del papel trae su sujeto, su tiempo y su
# cola («se desecha de plano la demanda de amparo promovida por…») y no cabe
# en «…en el que {sentido}». Cada clave tiene su frase en pretérito, con el
# sujeto implícito, y la expresión que TIENE que estar en la parte resolutiva.
CATALOGO_SENTIDO = {
    "queja": {
        "desecha_demanda": ("a", "desechó la demanda de amparo",
                            r"\bdesech\w*\s+(?:de\s+plano\s+)?(?:la\s+)?demanda\b|\bdemanda\b[^.]{0,80}\bse\s+desecha"),
        "no_presentada": ("a", "tuvo por no presentada la demanda de amparo",
                          r"\bpor\s+no\s+presentada\b"),
        "admite_demanda": ("a", "admitió la demanda de amparo",
                           r"\bse\s+admite\s+(?:a\s+tramite\s+)?(?:la\s+)?demanda\b|\badmitese\s+(?:a\s+tramite\s+)?la\s+demanda"),
        "desecha_ampliacion": ("a", "desechó la ampliación de la demanda",
                               r"\bdesech\w*\s+(?:de\s+plano\s+)?(?:la\s+)?ampliacion\b|\bampliacion\b[^.]{0,80}\bse\s+desecha"),
        "concede_provisional": ("b", "concedió la suspensión provisional",
                                r"\bconced\w*\s+(?:[a-z]+\s+){0,4}?la\s+suspension\s+provisional"),
        "niega_provisional": ("b", "negó la suspensión provisional",
                              r"\bnieg\w*\s+(?:[a-z]+\s+){0,4}?la\s+suspension\s+provisional|\bse\s+niega\s+(?:[a-z]+\s+){0,4}?la\s+suspension\s+provisional"),
        "concede_plano": ("b", "concedió la suspensión de plano",
                          r"\bconced\w*\s+(?:de\s+plano\s+)?(?:[a-z]+\s+){0,3}?la\s+suspension\s+de\s+plano|\bse\s+concede\s+de\s+plano\s+la\s+suspension"),
        "niega_plano": ("b", "negó la suspensión de plano",
                        r"\bnieg\w*\s+(?:de\s+plano\s+)?(?:[a-z]+\s+){0,3}?la\s+suspension\s+de\s+plano|\bse\s+niega\s+de\s+plano\s+la\s+suspension"),
        "reconoce_tercero": ("d", "reconoció el carácter de tercero interesado",
                             r"\breconoc\w*\s+(?:el\s+)?caracter\s+de\s+tercer[oa]s?\s+interesad"),
        "niega_tercero": ("d", "negó el carácter de tercero interesado",
                          r"(?:no\s+ha\s+lugar\s+a\s+reconocer|se\s+niega|no\s+se\s+reconoce)[^.]{0,60}caracter\s+de\s+tercer[oa]s?\s+interesad"),
    },
    "revision_fiscal": {
        "nulidad_lisa": ("", "declaró la nulidad lisa y llana de la resolución impugnada",
                         r"\bnulidad\s+lisa\s+y\s+llana\b"),
        "nulidad_efectos": ("", "declaró la nulidad de la resolución impugnada, para determinados efectos",
                            r"\bnulidad\b[^.]{0,220}?\bpara\s+(?:los\s+|el\s+|determinados\s+)?efectos?\b"),
        "validez": ("", "reconoció la validez de la resolución impugnada",
                    r"\breconoce\s+la\s+validez\b|\breconocio\s+la\s+validez\b|\bvalidez\s+de\s+la\s+resolucion\s+impugnada"),
        "sobresee": ("", "sobreseyó en el juicio", r"\bse\s+sobresee\b|\bsobreseyo\b"),
    },
}
# Más de una clave en el mismo resolutivo de la Sala: sobreseyó en parte y
# resolvió el fondo del resto. Si hay fondo, el sobreseimiento fue parcial.
_SOBRESEE_PARCIAL = "sobreseyó parcialmente en el juicio"


def _parte_resolutiva(texto: str, tipo: str) -> str:
    try:
        import fase_rama as _fr
        s = _fr.seccion_resolutiva(texto)
    except Exception:
        s = ""
    if s:
        return s
    t = " ".join(str(texto or "").split())
    # Un auto no tiene «RESUELVE»: lo que acuerda va tras «se acuerda» o en todo
    # el auto, que es corto.
    m = None
    for m in re.finditer(r"\b(?:se\s+acuerda|acuerda|se\s+provee|provey[eé]ndose)\s*:", t, re.I):
        pass
    if m:
        return t[m.end():]
    return t if (tipo == "queja" and len(t) < 40000) else t[-6000:]


def sentido_de(tipo: str, texto: str) -> tuple:
    """(clave, frase, claves_halladas) del catálogo, por palabras clave en la
    parte resolutiva. («», «», [...]) si no hay ninguna o hay más de una que no
    se puede componer."""
    tp = _tipo(tipo)
    cat = CATALOGO_SENTIDO.get(tp) or {}
    if not cat:
        return "", "", []
    parte = " ".join(_sin_tildes(_parte_resolutiva(texto, tp)).lower().split())
    halladas = [k for k, (_i, _f, rx) in cat.items() if re.search(rx, parte)]
    if tp == "queja":
        # La suspensión de plano y la provisional se excluyen; «desecha la
        # ampliación» contiene «desecha … demanda» a veces.
        if "desecha_ampliacion" in halladas and "desecha_demanda" in halladas:
            halladas.remove("desecha_demanda")
    if len(halladas) == 1:
        return halladas[0], cat[halladas[0]][1], halladas
    if tp == "revision_fiscal" and len(halladas) == 2 and "sobresee" in halladas:
        otra = [k for k in halladas if k != "sobresee"][0]
        return "sobresee+" + otra, f"{_SOBRESEE_PARCIAL} y {cat[otra][1]}", halladas
    return "", "", halladas


def _frase_del_catalogo(tipo: str, clave: str) -> str:
    cat = CATALOGO_SENTIDO.get(_tipo(tipo)) or {}
    if clave in cat:
        return cat[clave][1]
    if "+" in clave:
        a, b = clave.split("+", 1)
        if a == "sobresee" and b in cat:
            return f"{_SOBRESEE_PARCIAL} y {cat[b][1]}"
    return ""


def _clave_verificada(tipo: str, clave: str, texto: str) -> bool:
    cat = CATALOGO_SENTIDO.get(_tipo(tipo)) or {}
    parte = " ".join(_sin_tildes(_parte_resolutiva(texto, _tipo(tipo))).lower().split())
    claves = clave.split("+") if "+" in clave else [clave]
    return all(k in cat and re.search(cat[k][2], parte) for k in claves)


# ── EL SENTIDO VA A MEDIA FRASE (QUINTA RONDA, 3-oct-2026; Q 335/2025) ──────
# «…dictado por el Juzgado…, en el que Dejó sin efectos la suspensión del acto
# reclamado.» La mayúscula venía del papel (el auto empieza la oración así) o
# del formulario, y el resultando la copiaba; el mismo documento decía
# después «del auto que dejó sin efectos…». Una sigla («IMSS», «SAT») o un
# sentido entero en versales se dejan como vienen.
_RX_PRIMERA_PALABRA = re.compile(r"[^\wÁÉÍÓÚÑÜáéíóúñü]*([A-Za-zÁÉÍÓÚÑÜáéíóúñü]+)")


def sentido_en_minuscula(sentido) -> str:
    """El sentido con su primera letra en minúscula, salvo que la primera
    palabra sea una sigla o vaya en versales («Dejó sin efectos…» → «dejó sin
    efectos…»; «IMSS…» se queda)."""
    s = _limpio(sentido)
    m = _RX_PRIMERA_PALABRA.match(s)
    if not m:
        return s
    w, i = m.group(1), m.start(1)
    if len(w) >= 2 and w.isupper():
        return s
    return s[:i] + s[i].lower() + s[i + 1:]


# ── LA SALA QUE ADEMÁS RECONOCIÓ UN DERECHO (QUINTA RONDA, 3-oct-2026; RF 6/2026) ──
# La Sala declaró la nulidad para efectos «y reconoció el derecho subjetivo al
# incremento de la cuota pensionaria y al pago retroactivo de las diferencias»
# (art. 52, fr. V, inciso a), LFPCA). El catálogo sólo tenía «declaró la
# nulidad… para determinados efectos» y la cola se perdía sin aviso: el
# resultando callaba que la sentencia es de FONDO, que es lo que decide la
# procedencia por la fracción VI del 63. La clave sigue siendo la del catálogo
# (la prueba del compositor exige que sus frases y las de aquí sean las mismas)
# y la cola del papel se CONSERVA en `acto.sentido`: «declaró la nulidad de la
# resolución impugnada, para determinados efectos, y reconoció el derecho
# subjetivo de la actora al incremento…». Se copia del resolutivo (está en el
# papel); si viene en versales o es muy larga, la fórmula del artículo 52.
# «Se reconoce [a quien, sin pasar de la oración: «a la parte actora», «a la
# C. ANA RUIZ»] [la existencia de] el/un derecho…».
_RX_RECONOCE_DERECHO = re.compile(
    r"\b(?:se\s+)?reconoce(?:n)?\s+(?P<obj>(?:a\s+(?:[^.;:]|(?<=\b\w)\.){1,160}?\s+)?"
    r"(?:la\s+existencia\s+del?\s+)?(?:el\s+|un\s+|los\s+)?derechos?\b)", re.I)
# El tramo termina en el punto que cierra la oración (no en el de «art.», «fr.»
# ni en el de «S.A.»), en «;» o «:», en «y se condena…» o en el siguiente
# resolutivo.
_RX_FIN_DERECHO = re.compile(
    r"(?<!\bart)(?<!\barts)(?<!\bfr)(?<!\bfracc)(?<!\bnum)(?<!\bnúm)(?<!\bno)(?<!\b\w)\.(?:\s|$)"
    r"|[;:](?:\s|$)|\s+y\s+(?:se\s+)?conden|,\s+(?:por\s+lo\s+que|en\s+los?\s+t[eé]rminos)"
    r"|\s+(?:SEGUND|TERCER|CUART|QUINT|SEXT|S[EÉ]PTIM)[OA]\s*[.\-:]", re.I)
_RX_NEGADO = re.compile(r"\b(?:no|ni|sin\s+que)\s+(?:se\s+)?$", re.I)
# Un nombre en versales dentro del tramo («…de JUAN PÉREZ LÓPEZ al incremento…»):
# dos o más palabras seguidas en versales, a mayúsculas y minúsculas
# (`sin_versales`); una sigla sola («ISSSTE») se queda.
_RX_RACHA_VERSALES = re.compile(
    r"\b[A-ZÁÉÍÓÚÑÜ]{2,}(?:\s+(?:[A-ZÁÉÍÓÚÑÜ]{2,}|DE|DEL|LA|LAS|LOS|Y|E)\b)*\s+[A-ZÁÉÍÓÚÑÜ]{2,}\b")
_TOPE_COLA_DERECHO = 45            # palabras
_COLA_DERECHO_GENERICA = "y reconoció a la parte actora la existencia de un derecho subjetivo"


def cola_derecho_subjetivo(texto) -> str:
    """«y reconoció el derecho subjetivo…» si los puntos resolutivos de la
    sentencia de la Sala reconocen un derecho (art. 52, fr. V, a), LFPCA); «» si
    no («no se reconoce el derecho…» no cuenta). Copiada del papel; la fórmula
    genérica si el tramo viene en versales o pasa de `_TOPE_COLA_DERECHO`
    palabras."""
    parte = " ".join(_parte_resolutiva(str(texto or ""), "revision_fiscal").split())
    m = next((x for x in _RX_RECONOCE_DERECHO.finditer(parte)
              if not _RX_NEGADO.search(parte[max(0, x.start() - 16):x.start()])), None)
    if not m:
        return ""
    resto = parte[m.start("obj"):]
    fin = _RX_FIN_DERECHO.search(resto, len(m.group("obj")))
    tramo = resto[:fin.start() if fin else len(resto)].strip(" ,;:.")
    if (not re.search(r"[a-záéíóúñ]", tramo) or len(tramo.split()) > _TOPE_COLA_DERECHO
            or len(tramo.split()) < 2):
        return _COLA_DERECHO_GENERICA
    return "y reconoció " + _RX_RACHA_VERSALES.sub(lambda r: sin_versales(r.group(0)), tramo)


def _con_cola_derecho(sentido: str, clave: str, texto) -> str:
    """El sentido de la revisión fiscal con la cola del derecho reconocido, si
    la clave es una nulidad y el papel lo reconoce (y la cola no está ya)."""
    s = _limpio(sentido)
    if not s or "nulidad" not in (clave or "") or re.search(r"\breconoci\w*\b[^.]{0,80}\bderecho", _plano(s)):
        return s
    cola = cola_derecho_subjetivo(texto)
    if not cola:
        return s
    return f"{s.rstrip(' .')}{',' if ',' in s else ''} {cola}"


# ═══════════════════════════════════════════════════════════════════════════
# LO RECLAMADO O RECURRIDO — LO QUE SE LEE SIN MODELO
# ═══════════════════════════════════════════════════════════════════════════
def _clase_de(texto: str) -> str:
    cab = _plano(" ".join(str(texto or "").split())[:2500])
    cab_junta = re.sub(r"(?<=\b\w) (?=\w\b)", "", cab)   # «s e n t e n c i a»
    if re.search(r"\blaudo\b", cab):
        return "laudo"
    if re.search(r"\binterlocutoria\b", cab):
        return "interlocutoria"
    if re.search(r"\bsentencia\b", cab) or "sentencia" in cab_junta.replace(" ", "")[:400]:
        return "sentencia"
    if re.search(r"\bresolucion\b", cab):
        return "resolucion"
    if re.search(r"\b(?:auto|acuerdo|proveido)\b", cab[:600]):
        return "auto"
    return ""


def _mas_frecuente(rx, texto: str, grupo: str = "n", cabeza: int = 0) -> str:
    from collections import Counter
    t = " ".join(str(texto or "").split())
    if cabeza:
        t = t[:cabeza]
    c = Counter(re.sub(r"\s+", "", m.group(grupo)) for m in rx.finditer(t))
    return c.most_common(1)[0][0] if c else ""


_RX_ORGANO_GENERICO = re.compile(
    r"\b(?:(?:H\.\s*)?(?:Primera|Segunda|Tercera|Cuarta|Quinta|Sexta|S[ée]ptima|Octava|Novena|D[ée]cima|"
    r"PRIMERA|SEGUNDA|TERCERA|CUARTA|QUINTA|SEXTA)\s+)?"
    r"(?:Sala|SALA|Tribunal|TRIBUNAL|Juzgado|JUZGADO|Juez|JUEZ|Jueza|JUEZA|Junta|JUNTA)"
    r"(?:[ \t]+(?:(?i:del|de|las|la|los|el|en|y|e|con|sede)\b|(?<=Distrito )\d{1,3}|(?<=DISTRITO )\d{1,3}|"
    r"(?<=Centro )[IVXL]{1,5}\b|(?<=CENTRO )[IVXL]{1,5}\b|(?<=Juzgado )\d{1,2}\s?[°º]|(?<=JUZGADO )\d{1,2}\s?[°º]|(?<=Región )[IVXL]{1,5}\b|(?<=Region )[IVXL]{1,5}\b|"
    # LO QUE NO ES PARTE DE UN NOMBRE DE ÓRGANO y en una cabecera va pegado a él
    # («SALA FAMILIAR DEL TRIBUNAL SUPERIOR DE JUSTICIA RECURSO DE APELACIÓN
    # TOCA FAMILIAR 4022/2024»). «Amparo» y «Juicios» sí lo son («…de Amparo
    # Civil… y de Juicios Federales»), «Juicio» en singular no.
    r"(?!(?i:toca|recurso|expediente|sentencia|juicio|resoluci[oó]n|vistos?|actor|actora|demandad[oa]|"
    r"acto|actos|apelaci[oó]n|ponente|secretari[oa]|ordenadora|ejecutora|asunto|presentes?|quejos[oa]|"
    r"tercer[oa]s?\s+interesad[oa]s?|domicilio|[ivxl]+)\b)"
    # EL PUNTO CIERRA EL NOMBRE, salvo en una abreviatura («H.», «Qro.»): sin
    # esto «…DEL ESTADO DE QUERETARO. H. TRIBUNAL COLEGIADO…» era un solo órgano.
    r"(?:[A-ZÁÉÍÓÚÑ][\wáéíóúñÁÉÍÓÚÑ]{0,2}\.(?=\s)|[A-ZÁÉÍÓÚÑ][\wáéíóúñÁÉÍÓÚÑ]+(?!\w)))|"
    r",(?=[ \t]+(?i:Distrito|con\s+sede|Administrativ[oa]|Civil|Penal|Familiar|Mercantil|y\s+de\s+Trabajo)))+")


def _organos_en(texto: str) -> list:
    t = re.sub(r"[ \t]+", " ", str(texto or ""))
    fuera = []
    for m in _RX_ORGANO_GENERICO.finditer(t):
        o = re.sub(r"(?i)(?:[ ,]+(?:de|del|la|las|el|los|en|y|e|con|sede))+$", "", m.group(0).strip()).strip(" ,")
        if 4 <= len(o.split()) <= 24 and es_organo(o)[0]:
            fuera.append(o)
    return fuera


# «…demandó la nulidad de la resolución negativa ficta recaída a su solicitud
# de devolución del impuesto al valor agregado de marzo de 2024…»: la clase y,
# si va pegado, lo que se pidió.
_RX_NEGATIVA_FICTA = re.compile(
    r"negativa\s+ficta(?:[^.;]{0,80}?\b(?:reca[ií]da|configurada|recay[oó]|respecto)\s+"
    r"(?:a|al|a\s+la|de|del|de\s+la|en)\s+"
    r"(?P<m>(?:su\s+|la\s+|el\s+|sus\s+)?(?:solicitud|petici[oó]n|escrito|instancia|"
    r"recurso\s+de\s+revocaci[oó]n|promoci[oó]n|consulta)[^.;]{3,240}?)"
    r"(?=[.;]|,\s+(?:emitid|atribuid|que\s+se|por\s+la\s+que)|$))?", re.I)
_RX_CAP_AUTORIDADES = re.compile(r"autoridad(?:es)?\s+responsables?\b\s*[:.\-]*", re.I)
_RX_CAP_ACTO = re.compile(r"\bactos?\s+reclamados?\s*[:.\-]", re.I)


def _ordenadora_de_la_demanda(demanda: str) -> tuple:
    """(ordenadora, ejecutora) del capítulo «AUTORIDAD RESPONSABLE» de la
    demanda de amparo directo, tal como la escribe el quejoso. Lo que el
    tribunal copia en el resultando primero."""
    t = " ".join(str(demanda or "").split())
    # EL RÓTULO DEL CAPÍTULO, NO LA FRASE: «…los actos atribuidos a las
    # autoridades responsables. SEGUNDO. Se me conceda…» (el petitorio del ADC
    # 810/2025) no abre ningún capítulo. Un rótulo va en versales, numerado o
    # seguido de dos puntos.
    m = None
    for c in _RX_CAP_AUTORIDADES.finditer(t[:40000]):
        txt = c.group(0)
        antes = t[max(0, c.start() - 14):c.start()]
        if (re.sub(r"[^A-Za-zÁÉÍÓÚÑáéíóúñ]", "", txt).isupper() or ":" in txt
                or re.search(r"(?:\b[IVX]{1,4}|\b\d{1,2}|\b[a-dA-D])\s*[.\-)]*\s*$", antes)):
            m = c
            break
    if not m:
        return "", ""
    fin = _RX_CAP_ACTO.search(t, m.end())
    cap = t[m.end():fin.start() if fin and fin.start() - m.end() < 1500 else m.end() + 900]
    orden = re.search(r"(?:como\s+)?ordenadora\s*[:.\-]?\s*(?:se[ñn]alo\s+(?:a\s+)?)?(?:la|el|al|a\s+la)?\s*", cap, re.I)
    ejec = re.search(r"(?:como\s+)?ejecutora\s*[:.\-]?\s*(?:se[ñn]alo\s+(?:a\s+)?)?(?:la|el|al|a\s+la)?\s*", cap, re.I)
    if orden:
        o = [x for x in _organos_en(cap[orden.end():orden.end() + 300]) if "tribunal colegiado" not in _plano(x)]
        e = [x for x in _organos_en(cap[ejec.end():ejec.end() + 300]) if "tribunal colegiado" not in _plano(x)] \
            if ejec else []
        e = [x for x in e if re.match(r"(?i)(?:el\s+|la\s+)?(?:juez|jueza|juzgado|actuari)", x)]
        o_, e_ = (sin_versales(o[0]) if o else ""), (sin_versales(e[0]) if e else "")
        # «ORDENADORA Y EJECUTORA: el Tribunal Unitario…» nombra a una sola.
        return o_, ("" if o_ and e_ and _misma_base(o_, e_) else e_)
    # SIN RÓTULO DE ORDENADORA Y EJECUTORA, por lo que es cada uno: la Sala, el
    # Tribunal o la Junta dicta la sentencia que se reclama; el juez de primera
    # instancia la ejecuta. «A) JUEZ SEXTO… B) SEGUNDA SALA CIVIL…» (ADC
    # 810/2025) pone primero al ejecutor. Y el Tribunal Colegiado al que se
    # dirige la demanda nunca es responsable.
    orgs = [sin_versales(x) for x in _organos_en(cap) if "tribunal colegiado" not in _plano(x)]
    if not orgs:
        return "", ""
    juez = [x for x in orgs if re.match(r"(?i)(?:el\s+|la\s+)?(?:juez|jueza|juzgado)\b", x)]
    otros = [x for x in orgs if x not in juez]
    if otros:
        o_ = otros[0]
        e_ = juez[0] if juez and not _misma_base(o_, juez[0]) else ""
        return o_, e_
    return orgs[0], ""


def _notificacion_dicha(demanda: str) -> str:
    t = " ".join(str(demanda or "").split())[:40000]
    m = re.search(r"fecha\s+(?:en\s+que\s+(?:\w+\s+){0,4}?notific\w+|de\s+(?:la\s+)?notificaci[oó]n)"
                  r"[^.:]{0,120}[:.\-]\s*", t, re.I)
    if not m:
        return ""
    f = fechas_del_texto(t[m.end():m.end() + 120])
    return f[0][0] if f else ""


# «…el escrito de demanda presentado el…», «la demanda de amparo indirecto
# recibida en la Oficina de Correspondencia Común el…» (sexta ronda, C3).
_RX_DEMANDA_PRESENTADA = re.compile(
    r"\bdemanda(?:\s+de\s+amparo(?:\s+indirecto)?)?\s*,?\s+(?:que\s+(?:fue\s+)?)?"
    r"(?:presentad[oa]|recibid[oa]|depositad[oa])\b", re.I)
_RX_AMPLIACION_ANTES = re.compile(r"ampliacion(?:\s+de)?(?:\s+la)?\s*$")


def _fecha_demanda_presentada(t: str) -> str:
    """La fecha de presentación de la demanda de amparo, sólo si el papel la
    dice con su fórmula y dentro de la misma oración; «» si no."""
    for m in _RX_DEMANDA_PRESENTADA.finditer(t or ""):
        if _RX_AMPLIACION_ANTES.search(_plano(t[max(0, m.start() - 40):m.start()])):
            continue
        cola = t[m.end():m.end() + 200]
        fin = re.search(r";|\.\s+[A-ZÁÉÍÓÚÑ]", cola)
        f = fechas_del_texto(cola[:fin.start()] if fin else cola)
        if f:
            return f[0][0]
    return ""


def _lectura_determinista(tipo: str, acto: str, demanda: str, numero: str) -> dict:
    import fase_origen as _fo
    out: dict = {"acto": {}}
    a = out["acto"]
    a["clase"] = _clase_de(acto)
    # LA FECHA, SÓLO DEL PROEMIO. `fase_origen.datos_del_documento` también la
    # busca, pero sin descartar las fechas citadas: en el ADC 43/2025, cuyo
    # papel empieza en los resultandos, daba el 19 de enero de 2021 —un cargo
    # bancario— por la sentencia del 3 de julio de 2024. Mejor vacía.
    a["fecha"] = _fecha_proemio(acto)
    dd = _fo.datos_del_documento(acto)
    propio = re.sub(r"\s+", "", numero or "")
    cab = " ".join(str(acto or "").split())[:6000]
    if tipo == "amparo_directo":
        a["toca"] = _mas_frecuente(_RX_TOCA, acto, cabeza=6000)
        # El expediente de origen: el rótulo («EXPEDIENTE: 784/2022») o el
        # «dictada en el expediente N» de la cabecera; nunca el toca.
        # EL MÁS REPETIDO, NO EL PRIMERO. La Sala cita en su proemio otros
        # juicios (el amparo anterior, la ejecutoria que cumple) y el primer
        # «expediente N» de la cabecera era otro asunto en 642/2024 (1114/2017
        # por 1414/2020) y en 590/2024; el expediente de origen se nombra una y
        # otra vez.
        exp = dd.get("expediente") or ""
        if not exp:
            from collections import Counter
            cuenta = Counter()
            primero = {}
            for i, m in enumerate(_RX_EXPTE.finditer(" ".join(str(acto).split()))):
                n = re.sub(r"\s+", "", m.group("n"))
                if n in (a["toca"], propio):
                    continue
                cuenta[n] += 1
                primero.setdefault(n, i)
            if cuenta:
                exp = max(cuenta, key=lambda k: (cuenta[k], -primero[k]))
        a["expediente"] = exp
        o, e = _ordenadora_de_la_demanda(demanda)
        if o:
            out["responsable"] = o
        if e:
            out["ejecutora"] = e
        # EL ÓRGANO NO SE TOMA DE LA CABECERA DEL ACTO: la Sala nombra en su
        # proemio al juez cuya sentencia revisa, y el primer órgano que aparece
        # suele ser ése (274/2025: «…dictada… por la Jueza Tercero de Primera
        # Instancia…»). La ordenadora, de la demanda; o del modelo.
        if demanda:
            out["derechos"] = derechos_de(demanda)
            n = _notificacion_dicha(demanda)
            if n:
                out["notificacion"] = n
        try:
            import origen_acto as _oa
            o_ = _oa.origen(out.get("responsable") or a.get("organo") or "", acto[:20000],
                            tipo_asunto="amparo_directo", numero=numero)
            if o_.get("instancia"):
                a["instancia"] = o_["instancia"]
            # «ese recurso de apelación» / «ese juicio oral mercantil»
            # (`fase_origen.lo_resuelto`, por `origen_acto`).
            if o_.get("lo_resuelto"):
                a["lo_resuelto"] = o_["lo_resuelto"]
        except Exception:
            pass
    elif tipo == "amparo_revision":
        import fase_rama as _fr
        a["organo"] = _fr.juzgado_de_la_recurrida(acto)
        # `juzgado_de_la_recurrida` no junta renglones (por la cabecera en
        # versales) y en el 631 se quedaba en «Juzgado Séptimo de Distrito»: el
        # mismo juzgado, escrito entero en el cuerpo, completa el nombre.
        if a["organo"]:
            enteros = [x for x in _organos_en(" ".join(str(acto).split())[:30000])
                       if _misma_base(a["organo"], x) and len(x) > len(a["organo"])]
            if enteros:
                a["organo"] = max(enteros, key=len)
        a["expediente"] = _fr.numero_del_amparo(acto, numero) or ""
        if not a["expediente"]:
            m = re.search(r"\b(?:juicio\s+de\s+)?amparo(?:\s+indirecto)?\s*:?\s*(\d{1,6}/\d{4})", cab, re.I)
            if m and re.sub(r"\s+", "", m.group(1)) != propio:
                a["expediente"] = m.group(1)
        inc = bool(re.search(r"incidente\s+de\s+suspensi[oó]n|cuaderno\s+incidental|suspensi[oó]n\s+definitiva",
                             cab, re.I))
        a["incidente"] = inc
        if inc:
            a["clase"] = "interlocutoria"
        sec = _fr.seccion_resolutiva(acto)
        que = _fr.que_dice_el_resolutivo(sec) if sec else ""
        if inc:
            ps = _plano(sec or acto[-6000:])
            if re.search(r"\bse\s+concede\s+(?:[a-z]+\s+){0,4}?la\s+suspension\s+definitiva", ps):
                que = "concede"
            elif re.search(r"\bse\s+niega\s+(?:[a-z]+\s+){0,4}?la\s+suspension\s+definitiva", ps):
                que = "niega"
        if not que:
            que = _fr.resolvio_a_quo(acto, resolutivo=sec) if not inc else ""
        if que:
            a["resolvio"] = "mixto" if "_" in que else que
            if "_" in que:
                a["resolvio_mixto"] = que
        out["clase_recurrida"] = ("interlocutoria_suspension" if inc else
                                  "auto_sobreseimiento" if a["clase"] == "auto" and que == "sobresee"
                                  else "sentencia" if a["clase"] in ("sentencia", "") else "")
        q = _nombre_limpio(_fr.quejoso_del_resolutivo(sec or ""))
        if q:
            out["quejoso"] = q
        # La demanda de amparo indirecto, según el resultando primero de la recurrida.
        m = re.search(r"presentaci[oó]n\s+de\s+la\s+demanda|demanda\s+de\s+amparo", cab, re.I)
        if m:
            f = fechas_del_texto(cab[m.end():m.end() + 400])
            if f:
                out.setdefault("demanda", {})["fecha"] = f[0][0]
        m = re.search(r"(?:celebr[oó]|tuvo\s+verificativo|se\s+celebr[oó])\s+la\s+audiencia\s+constitucional", acto or "", re.I)
        if m:
            f = fechas_del_texto(" ".join(acto[max(0, m.start() - 160):m.end() + 120].split()))
            if len(f) == 1:
                out["audiencia"] = f[0][0]
    elif tipo == "queja":
        import fase_rama as _fr
        a["clase"] = a["clase"] if a["clase"] in ("auto", "interlocutoria", "resolucion") else "auto"
        jz = _fr.juzgado_de_la_recurrida(acto)
        orgs = _organos_en(cab)
        a["organo"] = sin_versales(jz or (orgs[0] if orgs else ""))
        # La cabecera en versales parte el nombre en dos renglones: el mismo
        # juzgado, entero, si el papel lo trae.
        if a["organo"]:
            enteros = [sin_versales(x) for x in _organos_en(" ".join(str(acto).split())[:30000])
                       if _misma_base(a["organo"], x) and len(x) > len(a["organo"])]
            if enteros:
                a["organo"] = max(enteros, key=len)
        m = re.search(r"\b(?:juicio\s+de\s+)?amparo(?:\s+(?:indirecto|directo))?\s*(?:n[uú]mero\s+)?:?\s*(\d{1,6}/\d{4}(?:-[\w-]{1,8})?)",
                      cab, re.I)
        if m and re.sub(r"\s+", "", m.group(1)) != propio:
            a["expediente"] = re.sub(r"\s+", "", m.group(1))
        inc = bool(re.search(r"incidente\s+de\s+suspensi[oó]n|cuaderno\s+incidental|suspensi[oó]n\s+provisional",
                             " ".join(str(acto or "").split()), re.I))
        clave, frase, _h = sentido_de("queja", acto)
        if clave:
            a["sentido"] = frase
            a["sentido_clave"] = clave
            if clave in ("concede_plano", "niega_plano"):
                inc = False
        a["incidente"] = inc
        via = ("indirecto" if jz or re.search(r"amparo\s+indirecto", cab, re.I) else
               "directo" if re.search(r"amparo\s+directo", cab, re.I) else "")
        a["via"] = via
        if via:
            out["fraccion_97"] = "I" if via == "indirecto" else "II"
        if clave:
            out["inciso_97"] = CATALOGO_SENTIDO["queja"][clave][0]
        # LA FECHA DE LA DEMANDA, SÓLO CON SU FÓRMULA (sexta ronda, 3-oct-2026,
        # C3): «el escrito de demanda presentado el cuatro de marzo…». Nunca la
        # primera fecha que siga a «demanda» (la del propio auto, la de una
        # ampliación); y no posterior al auto recurrido.
        fd = _fecha_demanda_presentada(" ".join(str(acto or "").split())[:20000])
        if fd and (not a["fecha"] or fd <= a["fecha"]):
            out.setdefault("demanda", {})["fecha"] = fd
    elif tipo == "revision_fiscal":
        a["clase"] = "sentencia"
        m = re.search(r"(?:(?:Primera|Segunda|Tercera|Cuarta|Quinta|Sexta)\s+)?Sala\s+(?:Regional|Especializada|Superior)"
                      r"[^.;]{0,160}?Tribunal\s+Federal\s+de\s+Justicia\s+(?:Administrativa|Fiscal)",
                      " ".join(str(acto or "").split())[:20000], re.I)
        if m:
            out["sala"] = sin_versales(m.group(0))
            a["organo"] = out["sala"]
        if dd.get("expediente"):
            out["expediente_tfja"] = dd["expediente"]
            a["expediente"] = dd["expediente"]
        clave, frase, _h = sentido_de("revision_fiscal", acto)
        if clave:
            a["sentido"] = frase
            a["sentido_clave"] = clave
        m = re.search(r"promovid[oa]\s+por\s+(?P<n>.{3,160}?)" + _TERMINA_NOMBRE, cab, re.I)
        if m:
            out["actora"] = _nombre_limpio(m.group("n"))
        # LA NEGATIVA FICTA NO TIENE OFICIO NI FECHA (tercera ronda, 3-oct-2026,
        # rev_5). En dos revisiones fiscales del banco lo impugnado era una
        # negativa ficta y el esquema sólo sabía de «oficio, fecha, autoridad»:
        # con los tres vacíos salía «la nulidad de la resolución.» y tres avisos
        # FALTA EL OFICIO / LA FECHA / LA AUTORIDAD que engañan. La clase y lo
        # que se pidió (`materia`) se leen del papel; la materia, anclada.
        mn = _RX_NEGATIVA_FICTA.search(cab)
        if mn:
            ri = out.setdefault("resolucion_impugnada", {})
            ri["clase"] = "negativa_ficta"
            mat = _limpio(mn.group("m") or "").strip(" ,;:.")
            if 3 <= len(mat.split()) <= 40:
                ri["materia"] = mat
        m = re.search(r"oficio\s+(?:n[uú]mero\s+|no\.\s*)?(?P<o>[A-Z0-9][\w./-]{3,60}\d)", cab)
        if m and not mn:
            ri = out.setdefault("resolucion_impugnada", {})
            ri["clase"] = "expresa"
            ri["oficio"] = m.group("o").rstrip(".,")
            # «…en el oficio X, de quince de enero de dos mil veinticinco,
            # emitida por la Administración…»: la fecha y la autoridad, si van
            # pegadas al oficio.
            cola = cab[m.end():m.end() + 400]
            f = fechas_del_texto(cola[:90])
            if f:
                ri["fecha"] = f[0][0]
            me = re.search(r"emitid[oa]\s+por\s+(?:el|la|los|las)\s+(?P<a>[A-ZÁÉÍÓÚÑ][^;]{5,200}?)"
                           r"(?=,\s|\s+mediante\b|\s+por\s+la\s+que\b|;|\.\s+[A-ZÁÉÍÓÚÑ]|$)", cola)
            if me:
                ri["autoridad"] = _limpio(me.group("a"))
    return {k: v for k, v in out.items() if not _vacio(v)}


# ═══════════════════════════════════════════════════════════════════════════
# LO RECLAMADO O RECURRIDO — CON MODELO, ANCLADO
# ═══════════════════════════════════════════════════════════════════════════
def _extracto(texto: str, cabeza: int, cola: int) -> str:
    t = str(texto or "")
    if len(t) <= cabeza + cola + 200:
        return t
    return t[:cabeza] + "\n[…]\n" + t[-cola:]


_CAMPOS_MODELO = {
    "amparo_directo": """{"clase": "sentencia | resolucion | laudo",
 "fecha": "AAAA-MM-DD de la sentencia reclamada (su proemio)",
 "organo": "el órgano que la DICTÓ, nombre completo literal",
 "toca": "número del toca de apelación, o vacío si no hubo apelación",
 "expediente": "número del expediente o juicio de origen",
 "ordenadora": "la autoridad ORDENADORA que señala la demanda, literal",
 "ejecutora": "la EJECUTORA que señala la demanda, literal, o vacío",
 "terceros": ["cada tercero interesado (la contraparte en el juicio de origen), literal"],
 "quejoso": "quien promueve el amparo, literal",
 "notificacion": "AAAA-MM-DD en que, según la demanda, se le notificó la sentencia, o vacío"}""",
    "amparo_revision": """{"clase": "sentencia | interlocutoria | auto",
 "incidente": "true si se dictó en el incidente de suspensión",
 "fecha": "AAAA-MM-DD de la resolución recurrida",
 "juzgado": "el Juzgado de Distrito que la dictó, nombre completo literal",
 "juicio": "número del juicio de amparo indirecto",
 "quejoso": "quien promovió el amparo, literal",
 "demanda_fecha": "AAAA-MM-DD en que se presentó la demanda de amparo",
 "autoridades": ["cada autoridad responsable, literal"],
 "actos": ["cada acto reclamado COPIADO TAL CUAL del papel, sin resumir ni saltar palabras, COMPLETO hasta su punto final; si pasa de 120 palabras, sus primeras 120 seguidas"],
 "audiencia": "AAAA-MM-DD de la audiencia constitucional, si el papel la fecha",
 "resolvio": "concede | niega | sobresee | sobresee_concede | sobresee_niega, según sus PUNTOS RESOLUTIVOS"}""",
    "queja": """{"clase": "auto | acuerdo | resolucion",
 "fecha": "AAAA-MM-DD del auto recurrido",
 "organo": "el órgano que lo dictó, nombre completo literal",
 "expediente": "número del juicio de amparo",
 "via": "indirecto | directo",
 "incidente": "true si se dictó en el incidente de suspensión",
 "sentido_clave": "UNA de: CLAVES, o «otro»",
 "sentido_libre": "SÓLO si la clave es «otro»: lo que decidió el auto, copiado del papel, máximo 25 palabras",
 "quejoso": "quien promovió el amparo, literal",
 "demanda_fecha": "AAAA-MM-DD en que se PRESENTÓ la demanda de amparo, sólo si el auto la escribe; si no, vacío",
 "autoridades": ["cada autoridad responsable que SEÑALÓ LA DEMANDA de amparo, literal (nunca el juzgado que dictó el auto)"],
 "actos": ["cada acto que RECLAMÓ LA DEMANDA de amparo, COPIADO TAL CUAL del papel, sin resumir ni saltar palabras, COMPLETO hasta su punto final; si pasa de 120 palabras, sus primeras 120 seguidas"]}""",
    "revision_fiscal": """{"fecha": "AAAA-MM-DD de la sentencia de la Sala",
 "sala": "la Sala que la dictó, nombre completo literal",
 "expediente_tfja": "número del juicio contencioso administrativo",
 "actora": "la parte actora, literal",
 "resolucion_clase": "expresa | negativa_ficta (si lo impugnado es una negativa ficta)",
 "resolucion_materia": "SÓLO si es negativa ficta: lo que se pidió (la solicitud), literal",
 "oficio": "número del oficio de la resolución impugnada (vacío si es negativa ficta)",
 "fecha_resolucion_impugnada": "AAAA-MM-DD de la resolución impugnada",
 "autoridad_emisora": "quien emitió la resolución impugnada, literal",
 "autoridad_demandada": "la autoridad demandada en el juicio, literal",
 "sentido_clave": "UNA de: CLAVES"}""",
}


def _prompt_acto(tipo: str, acto: str, demanda: str, numero: str) -> str:
    campos = _CAMPOS_MODELO[tipo]
    cat = CATALOGO_SENTIDO.get(tipo) or {}
    if cat:
        campos = campos.replace("CLAVES", ", ".join(f"«{k}» ({v[1]})" for k, v in cat.items()))
    nombre = {"amparo_directo": "la SENTENCIA RECLAMADA en un amparo directo",
              "amparo_revision": "la RESOLUCIÓN RECURRIDA del Juzgado de Distrito en un amparo en revisión",
              "queja": "el AUTO RECURRIDO en un recurso de queja",
              "revision_fiscal": "la SENTENCIA de la Sala del Tribunal Federal de Justicia Administrativa "
                                 "recurrida en revisión fiscal"}[tipo]
    bloque_demanda = ""
    if demanda.strip():
        bloque_demanda = f"""
LA DEMANDA DE AMPARO (sus primeras páginas):
──────────────────────────────────────────
{_extracto(demanda, 14000, 0)}
──────────────────────────────────────────
"""
    # LA QUEJA: LA DEMANDA, COMO LA DESCRIBE EL AUTO (sexta ronda, 3-oct-2026,
    # C3). El auto que desecha, admite o provee sobre la suspensión suele
    # describir la demanda; el resultando «Demanda de amparo.» sale de ahí.
    # Las tres confusiones que hay que cerrar: la demanda no es el escrito de
    # queja, ni lo que decidió el auto, ni el juzgado una de sus autoridades.
    regla_queja = ""
    if tipo == "queja":
        regla_queja = """- LA DEMANDA DE AMPARO («quejoso», «demanda_fecha», «autoridades»,
  «actos») es la que promovió el quejoso en el juicio de amparo, tal como la
  DESCRIBE ESTE AUTO («Visto el escrito de demanda presentado el… por…,
  contra actos de…, consistentes en…»). NO es el escrito del recurso de queja
  ni lo que alega el recurrente; NO es lo que decidió el auto (eso va en
  «sentido_clave»); y el juzgado que dictó el auto NUNCA es una de sus
  autoridades responsables. Si el auto no describe la demanda, esos campos
  van vacíos.
- LA AMPLIACIÓN DE DEMANDA NO ES LA DEMANDA: si el auto provee sobre una
  ampliación (la admite, la desecha, la tiene por no presentada), la fecha,
  las autoridades y los actos son los de la demanda INICIAL; no copies los de
  la ampliación. Si el auto sólo describe la ampliación, esos campos van vacíos.
"""
    return f"""Eres el secretario de un Tribunal Colegiado fichando el asunto {numero or ''}.
Tienes delante {nombre}. Extrae los datos procesales.

Reglas que no se negocian:
- COPIA LITERAL de nombres, órganos y números, como están escritos (el papel
  puede venir de un OCR con erratas: copia el nombre, no la errata evidente).
- FECHAS en AAAA-MM-DD y sólo si están escritas en el papel.
- SI NO CONSTA, CADENA VACÍA (o lista vacía). Esto se firma: un dato inventado
  aquí acaba en un resultando.
{regla_queja}
EL DOCUMENTO:
──────────────────────────────────────────
{_extracto(acto, 16000, 8000)}
──────────────────────────────────────────
{bloque_demanda}
Devuelve JSON y nada más:
{campos}"""


def _json_de(r) -> dict:
    crudo = (r.choices[0].message.content or "").strip()
    m = re.search(r"\{.*\}", crudo, re.S)
    d = json.loads(m.group(0) if m else crudo)
    return d if isinstance(d, dict) else {}


def _bool(x):
    if isinstance(x, bool):
        return x
    s = _plano(x)
    if s in ("true", "si", "1", "verdadero"):
        return True
    if s in ("false", "no", "0", "falso"):
        return False
    return None


def _fecha_de_ampliacion(iso, papel) -> bool:
    """¿Esa fecha sólo aparece en el papel como la de una AMPLIACIÓN de demanda
    («el escrito de ampliación de demanda presentado el veinte de agosto…»)?
    (revisión Q, 3-oct-2026; Q 300/2025: el 97-I-a cubre «su ampliación», y el
    auto que provee sobre ella la describe; el modelo podía copiar su fecha
    como la de la demanda y pasaba el anclaje, porque está en el papel). Se
    mira la oración que lleva a cada aparición de la fecha, en sus 90
    caracteres de antes: basta una aparición sin «ampliación» para conservarla."""
    pp = _papel(papel)
    i = a_iso(iso)
    if not i:
        return False
    plano = _plano_fechas(pp.texto)
    hits = [pos for (x, _lit, pos) in pp.fechas if x == i]
    if not hits:
        return False
    for pos in hits:
        antes = plano[max(0, pos - 90):pos]
        antes = re.split(r"[.;]\s", antes)[-1]
        if "ampliaci" not in antes:
            return False
    return True


def _demanda_anclada(d: dict, papel, avisos: list, out: dict) -> dict:
    """LA DEMANDA DE AMPARO QUE DIJO EL MODELO, PASADA POR EL PAPEL: la fecha
    de presentación (`demanda_fecha`), las autoridades responsables y los
    actos reclamados, en `out["demanda"]`. Las guardas son las del amparo en
    revisión (tercera ronda e integración, 3-oct-2026) y desde la sexta ronda
    valen también para la queja (C3) y para el escrito del recurso:
      · la fecha, sólo si está escrita en el papel;
      · cada autoridad y cada acto, anclados al papel (y del papel su forma);
      · UN ACTO LARGO NO SE TIRA EN SILENCIO: sus primeras 120 palabras y
        `_completar_oracion` lo cierra con el papel.
    Pura; sólo añade a `avisos` y a `out`."""
    pp = _papel(papel)
    v = d.get("demanda_fecha")
    v = a_iso(v) if isinstance(v, (str, _dt.date)) else ""
    if v:
        if not fecha_en_papel(v, pp):
            avisos.append(f"SE DESCARTÓ la fecha {_dmy(v)} como presentación de la demanda de amparo: no "
                          f"está escrita en el papel.")
        elif _fecha_de_ampliacion(v, pp):
            avisos.append(f"SE DESCARTÓ la fecha {_dmy(v)} como presentación de la demanda de amparo: en el "
                          f"papel es la de una AMPLIACIÓN de demanda, no la de la demanda inicial "
                          f"(demanda.fecha).")
        else:
            _poner(out, "demanda.fecha", v)
    aut = _anclar_lista(d.get("autoridades") if isinstance(d.get("autoridades"), list) else [],
                        pp, "autoridad responsable", avisos)
    if aut:
        _poner(out, "demanda.autoridades", aut)
    actos = []
    for a in (d.get("actos") if isinstance(d.get("actos"), list) else []):
        if not isinstance(a, str) or not _limpio(a):
            continue
        pal = _limpio(a).split()
        actos.append(" ".join(pal[:120]) if len(pal) > 130 else _limpio(a))
    act = [_completar_oracion(a, pp, avisos, "acto reclamado")
           for a in _anclar_lista(actos, pp, "acto reclamado", avisos)]
    if act:
        _poner(out, "demanda.actos", act)
    return out


def _modelo_anclado(tipo: str, d: dict, papel, avisos: list) -> dict:
    """Lo que dijo el modelo, pasado por el papel. Se devuelve con la forma de
    la ficha (acto.*, responsable, demanda.*…) y sólo con lo que se sostuvo.

    `papel`: el texto o un `Papel`. PURA Y SIN ESTADO GLOBAL (tercera ronda,
    rev_6): `leer_acto` la manda a un hilo, porque anclar los actos reclamados
    de un amparo en revisión es CPU de segundos. Sólo añade a `avisos`, la
    lista que le pasan."""
    out: dict = {}
    papel = _papel(papel)
    d = d if isinstance(d, dict) else {}

    def fecha(ruta, v, que):
        v = a_iso(v) if isinstance(v, (str, _dt.date)) else ""
        if not v:
            return
        if fecha_en_papel(v, papel):
            _poner(out, ruta, v)
        else:
            avisos.append(f"SE DESCARTÓ la fecha {_dmy(v)} como {que}: no está escrita en el papel.")

    def nombre(ruta, v, que):
        # Sólo texto: una lista o un dict del modelo no es un nombre (antes
        # se anclaba su repr y salía un aviso con «{'x': 1}»).
        v = _limpio(v) if isinstance(v, str) else ""
        if not v:
            return
        a, _m = anclar(v, papel)
        if a:
            # El papel en versales se copia sin versales: el nombre es el mismo.
            _poner(out, ruta, sin_versales(a))
        else:
            avisos.append(f"SE DESCARTÓ «{v[:90]}» como {que}: no aparece en el papel. "
                          f"Escríbelo tú si es correcto.")

    def numero(ruta, v, que):
        # «4520/2024 FM» (274/2025): el número, sin la clave de la mesa.
        m_ = re.search(r"\d{1,6}\s*/\s*\d{2,4}(?:-[\w-]{1,20})?|[A-Z0-9][\w./-]{3,60}\d",
                       v if isinstance(v, str) else "")
        v = re.sub(r"\s+", "", m_.group(0) if m_ else "")
        if not v:
            return
        if _numero_en_papel(v, papel):
            _poner(out, ruta, v)
        else:
            avisos.append(f"SE DESCARTÓ «{v}» como {que}: no aparece en el papel.")

    clase = _plano(d.get("clase"))
    if clase in ("sentencia", "resolucion", "laudo", "auto", "interlocutoria", "acuerdo"):
        _poner(out, "acto.clase", "auto" if clase == "acuerdo" else clase)
    fecha("acto.fecha", d.get("fecha"), "fecha de lo reclamado o recurrido")
    inc = _bool(d.get("incidente"))
    if inc is not None and tipo in ("amparo_revision", "queja"):
        _poner(out, "acto.incidente", inc)
    if tipo == "amparo_directo":
        nombre("acto.organo", d.get("organo"), "órgano que dictó la sentencia reclamada")
        numero("acto.toca", d.get("toca"), "toca")
        numero("acto.expediente", d.get("expediente"), "expediente de origen")
        nombre("responsable", d.get("ordenadora"), "autoridad responsable ordenadora")
        nombre("ejecutora", d.get("ejecutora"), "autoridad ejecutora")
        ter = _anclar_lista(d.get("terceros") if isinstance(d.get("terceros"), list) else
                            [d.get("terceros")], papel, "tercero interesado", avisos)
        if ter:
            out["terceros"] = ter
        nombre("quejoso", d.get("quejoso"), "quejoso")
        fecha("notificacion", d.get("notificacion"), "notificación según la demanda")
    elif tipo == "amparo_revision":
        nombre("acto.organo", d.get("juzgado"), "juzgado que dictó la resolución recurrida")
        numero("acto.expediente", d.get("juicio"), "juicio de amparo")
        nombre("quejoso", d.get("quejoso"), "quejoso")
        _demanda_anclada(d, papel, avisos, out)
        fecha("audiencia", d.get("audiencia"), "audiencia constitucional")
        r = _plano(d.get("resolvio")).replace(" ", "_")
        if r in ("concede", "niega", "sobresee", "sobresee_concede", "sobresee_niega"):
            _poner(out, "acto.resolvio", "mixto" if "_" in r else r)
            if "_" in r:
                _poner(out, "acto.resolvio_mixto", r)
    elif tipo == "queja":
        nombre("acto.organo", d.get("organo"), "órgano que dictó el auto recurrido")
        numero("acto.expediente", d.get("expediente"), "juicio de amparo")
        via = _plano(d.get("via"))
        if via in ("indirecto", "directo"):
            _poner(out, "acto.via", via)
        nombre("quejoso", d.get("quejoso"), "quejoso")
        # LA DEMANDA DE AMPARO, COMO LA DESCRIBE EL AUTO RECURRIDO (sexta ronda,
        # 3-oct-2026, C3): con las mismas guardas que el amparo en revisión.
        _demanda_anclada(d, papel, avisos, out)
        clave = _limpio(d.get("sentido_clave")) if isinstance(d.get("sentido_clave"), str) else ""
        if clave in CATALOGO_SENTIDO["queja"]:
            if _clave_verificada("queja", clave, papel.texto):
                _poner(out, "acto.sentido_clave", clave)
                _poner(out, "acto.sentido", CATALOGO_SENTIDO["queja"][clave][1])
            else:
                avisos.append(f"EL MODELO DIJO QUE EL AUTO «{CATALOGO_SENTIDO['queja'][clave][1]}» y la "
                              f"parte que acuerda no lo dice con esas palabras: el sentido queda en hueco.")
        elif clave == "otro":
            libre = _limpio(d.get("sentido_libre")) if isinstance(d.get("sentido_libre"), str) else ""
            if libre and len(libre.split()) <= 25:
                a, _m = anclar(libre, papel)
                if a and len(a.split()) <= 25:
                    _poner(out, "acto.sentido", a)
                    _poner(out, "acto.sentido_clave", "otro")
                else:
                    avisos.append(f"SE DESCARTÓ «{libre[:90]}» como lo que decidió el auto: no "
                                  f"está en el papel.")
    elif tipo == "revision_fiscal":
        nombre("sala", d.get("sala"), "Sala que dictó la sentencia")
        if out.get("sala"):
            _poner(out, "acto.organo", out["sala"])
        numero("expediente_tfja", d.get("expediente_tfja"), "expediente del juicio contencioso")
        if out.get("expediente_tfja"):
            _poner(out, "acto.expediente", out["expediente_tfja"])
        nombre("actora", d.get("actora"), "parte actora")
        numero("resolucion_impugnada.oficio", d.get("oficio"), "oficio de la resolución impugnada")
        fecha("resolucion_impugnada.fecha", d.get("fecha_resolucion_impugnada"),
              "fecha de la resolución impugnada")
        nombre("resolucion_impugnada.autoridad", d.get("autoridad_emisora"),
               "autoridad que emitió la resolución impugnada")
        nombre("autoridad_demandada", d.get("autoridad_demandada"), "autoridad demandada")
        # La clase de lo impugnado: la negativa ficta, sólo si el papel lo dice.
        rc = _plano(d.get("resolucion_clase") if isinstance(d.get("resolucion_clase"), str) else "")
        rc = rc.replace(" ", "_")
        if rc == "negativa_ficta":
            if re.search(r"negativa\s+ficta", papel.plano):
                _poner(out, "resolucion_impugnada.clase", "negativa_ficta")
                nombre("resolucion_impugnada.materia", d.get("resolucion_materia"),
                       "lo que se pidió (la negativa ficta)")
            else:
                avisos.append("EL MODELO DIJO QUE LO IMPUGNADO ES UNA NEGATIVA FICTA y la sentencia "
                              "no lo dice con esas palabras: no se tomó.")
        elif rc == "expresa" and (_tomar(out, "resolucion_impugnada.oficio")
                                  or _tomar(out, "resolucion_impugnada.fecha")):
            _poner(out, "resolucion_impugnada.clase", "expresa")
        clave = _limpio(d.get("sentido_clave")) if isinstance(d.get("sentido_clave"), str) else ""
        if clave and _frase_del_catalogo("revision_fiscal", clave):
            if _clave_verificada("revision_fiscal", clave, papel.texto):
                _poner(out, "acto.sentido_clave", clave)
                _poner(out, "acto.sentido", _frase_del_catalogo("revision_fiscal", clave))
            else:
                avisos.append(f"EL MODELO DIJO QUE LA SALA «{_frase_del_catalogo('revision_fiscal', clave)}» "
                              f"y sus puntos resolutivos no lo dicen: el sentido queda en hueco.")
    return out


async def leer_acto(cliente, texto_acto: str, tipo: str, numero: str = "", *,
                    demanda: str = "") -> dict:
    """Lo reclamado o recurrido, leído del papel. Primero lo determinista que ya
    existe (`fase_origen`, `fase_rama`, `origen_acto`, el catálogo del sentido);
    después, si hay `cliente`, UNA llamada JSON por tipo (`MODELO_ACTO`, tope
    `TOPE_SALIDA`) que sólo rellena lo que faltó, y todo lo que devuelve se ancla
    al papel. `demanda`: el texto de la demanda de amparo directo (de ahí salen
    la ordenadora tal como la señala el quejoso, los terceros y los derechos).

    Devuelve una ficha parcial: {acto:{…}, responsable, ejecutora, terceros,
    derechos, demanda:{…}, audiencia, clase_recurrida, fraccion_97, inciso_97,
    actora, resolucion_impugnada:{…}, sala, expediente_tfja, quejoso, avisos,
    fuentes}. Lo que no se leyó no aparece.

    LA QUEJA LEE LA DEMANDA (sexta ronda, 3-oct-2026, C3): el auto recurrido
    —el que desecha, admite o provee sobre la suspensión— suele describirla;
    quejoso, `demanda.fecha`, `demanda.autoridades` y `demanda.actos` se piden
    y se anclan como en el amparo en revisión, y el juzgado que dictó el auto
    sale de las autoridades. Sin ellos no se avisa: es contexto, no requisito."""
    tp = _tipo(tipo)
    avisos: list = []
    acto = _RX_CONTROL.sub("", str(texto_acto or ""))
    dem = _RX_CONTROL.sub("", str(demanda or ""))
    if not tp:
        return {"avisos": ["NO SE SABE EL TIPO DE ASUNTO: no se puede leer lo reclamado o recurrido."]}
    if not acto.strip():
        return {"avisos": ["NO HAY TEXTO DE LA RESOLUCIÓN reclamada o recurrida: los datos del acto "
                           "salen del formulario o quedan en hueco."]}
    # FUERA DEL BUCLE DE EVENTOS (tercera ronda, 3-oct-2026, rev_6). La lectura
    # sin modelo, el anclaje de lo que dice el modelo y el voto del órgano son
    # CPU sobre papeles de cientos de miles de caracteres: en el bucle paraban
    # el worker entero —latidos del stream, /taller/estado— decenas de
    # segundos. Van a un hilo, con UN `Papel` para toda la lectura.
    pp = Papel(acto + "\n" + dem)
    try:
        det = await asyncio.to_thread(_lectura_determinista, tp, acto, dem, numero)
    except Exception as ex:
        det = {}
        avisos.append(f"La lectura sin modelo del acto falló ({type(ex).__name__}); se sigue con el modelo.")
    mod: dict = {}
    if cliente:
        try:
            import llamada_modelo as _lm
            kw = dict(model=MODELO_ACTO,
                      messages=[{"role": "user", "content": _prompt_acto(tp, acto, dem, numero)}],
                      max_completion_tokens=TOPE_SALIDA, response_format={"type": "json_object"})
            if ESFUERZO_ACTO:
                kw["reasoning_effort"] = ESFUERZO_ACTO
            # CON TOPE PROPIO: lo del modelo es relleno; si no contesta, se
            # sigue con lo determinista (ver TOPE_SEGUNDOS_MODELO).
            r = await asyncio.wait_for(_lm.crear(cliente, **kw), timeout=TOPE_SEGUNDOS_MODELO)
            d = _json_de(r)
            mod = await asyncio.to_thread(_modelo_anclado, tp, d, pp, avisos)
        except asyncio.TimeoutError:
            avisos.append(f"LA LECTURA CON MODELO DEL ACTO NO CONTESTÓ EN {TOPE_SEGUNDOS_MODELO} s: vale lo "
                          f"que se leyó sin modelo.")
        except Exception as ex:
            avisos.append(f"No se pudo leer el acto con el modelo ({type(ex).__name__}): vale lo que "
                          f"se leyó sin modelo.")
    return await asyncio.to_thread(_cerrar_acto, tp, det, mod, avisos, pp)


def _cerrar_acto(tp: str, det: dict, mod: dict, avisos: list, pp: Papel) -> dict:
    """La lectura del acto, fundida: lo determinista manda, el modelo rellena,
    la ordenadora del AD por voto, lo derivado por tipo y lo que falta dicho.
    Síncrona y pura (corre en un hilo)."""
    # LO DETERMINISTA MANDA; el modelo rellena. Y si discrepan en una fecha o un
    # número, se dice: uno de los dos está mirando otra parte del papel.
    fin = fundir_lecturas(det, mod, rotulo="papel")
    fin["avisos"] = avisos + [a for a in fin.get("avisos") or [] if a not in avisos]
    a = fin.setdefault("acto", {})
    # El AD: la ordenadora dictó la sentencia. Si la cabecera trae sólo un
    # pedazo («Segunda Sala Civil») y la demanda o el modelo el nombre entero,
    # vale el entero.
    if tp == "amparo_directo":
        # LA ORDENADORA, POR VOTO Y POR FORMA. Medido en 10 AD del banco
        # Kingston contra su sentencia real: el modelo lee bien QUIÉN DICTÓ la
        # sentencia (su nombre oficial, de la propia sentencia) y la demanda lo
        # escribe a su manera —«Segunda Sala Civil de Queretato», «Los
        # Magistrados que integran la Primera Sala Civil del Poder Judicial»—.
        # Las tres lecturas votan qué órgano es; de ése, se toma la forma más
        # completa y mejor escrita que esté en el papel.
        cands = [(mod.get("acto") or {}).get("organo"), mod.get("responsable"),
                 det.get("responsable"), (det.get("acto") or {}).get("organo")]
        elegido = mejor_organo([c for c in cands if c], pp)
        if elegido:
            otros = {_nucleo(c) for c in cands if c} - {_nucleo(elegido)}
            if otros:
                fin["avisos"].append(
                    f"LAS LECTURAS NO NOMBRAN AL MISMO ÓRGANO COMO RESPONSABLE: se tomó «{elegido}»; "
                    f"compruébalo en la demanda y en la sentencia reclamada.")
            fin["responsable"] = elegido
            a["organo"] = elegido
        fin["avisos"] = [x for x in fin["avisos"] if not x.startswith("DOS CONSTANCIAS NO DICEN LO MISMO sobre responsable")]
    if tp == "queja":
        if not fin.get("fraccion_97") and a.get("via"):
            fin["fraccion_97"] = "I" if a["via"] == "indirecto" else "II"
        if not fin.get("inciso_97") and a.get("sentido"):
            ck = a.get("sentido_clave", "")
            if ck in CATALOGO_SENTIDO["queja"]:
                fin["inciso_97"] = CATALOGO_SENTIDO["queja"][ck][0]
            else:
                try:
                    import tipos_asunto as _ta
                    fin["inciso_97"] = _ta.inciso_97(a["sentido"])
                except Exception:
                    pass
        if a.get("sentido_clave") in ("concede_plano", "niega_plano"):
            a["incidente"] = False
        # EL PROPIO JUZGADO FUERA DE LAS AUTORIDADES DE LA DEMANDA (sexta ronda,
        # C3): el auto recurrido lo dictó él; en su propio juicio no es
        # responsable. Sólo si tiene forma de órgano de amparo (en la fracción
        # II quien dicta es la responsable del amparo directo).
        dem = fin.get("demanda") if isinstance(fin.get("demanda"), dict) else {}
        org = a.get("organo") or ""
        if org and dem.get("autoridades") and es_organo_de_amparo(org):
            quedan = [x for x in dem["autoridades"] if not _mismo_organo(org, x)]
            if len(quedan) != len(dem["autoridades"]):
                fin["avisos"].append(
                    f"EL JUZGADO QUE DICTÓ EL AUTO RECURRIDO («{_cita(org)}») SE LEYÓ ENTRE LAS AUTORIDADES "
                    f"RESPONSABLES DE LA DEMANDA: se quitó de esa lista (un juzgado de amparo no es responsable "
                    f"en su propio juicio).")
                if quedan:
                    dem["autoridades"] = quedan
                else:
                    dem.pop("autoridades", None)
                if not dem:
                    fin.pop("demanda", None)
    if tp == "amparo_revision" and not fin.get("clase_recurrida"):
        fin["clase_recurrida"] = ("interlocutoria_suspension" if a.get("incidente") else
                                  "auto_sobreseimiento" if a.get("clase") == "auto" else "sentencia")
    if tp == "revision_fiscal":
        if fin.get("sala") and not a.get("organo"):
            a["organo"] = fin["sala"]
        if fin.get("expediente_tfja") and not a.get("expediente"):
            a["expediente"] = fin["expediente_tfja"]
        # EL DERECHO RECONOCIDO NO SE PIERDE (quinta ronda, RF 6/2026).
        if a.get("sentido"):
            a["sentido"] = _con_cola_derecho(a["sentido"], a.get("sentido_clave", ""), pp.texto)
    if a.get("sentido"):
        a["sentido"] = sentido_en_minuscula(a["sentido"])
    # Los avisos de lo que falta, por tipo. Nombran el dato y dónde está.
    falta = {"acto.fecha": "la fecha de la resolución (está en su proemio)",
             "acto.organo": "el órgano que la dictó (está en su encabezado y en la firma)"}
    if tp == "amparo_revision":
        falta["acto.expediente"] = "el número del juicio de amparo indirecto (encabezado de la sentencia)"
        falta["acto.resolvio"] = "qué resolvió el juzgado (sus puntos resolutivos)"
    if tp in ("queja", "revision_fiscal"):
        falta["acto.sentido"] = ("qué decidió el auto recurrido (su parte que acuerda)" if tp == "queja"
                                 else "qué resolvió la Sala (sus puntos resolutivos)")
    for ruta, que in falta.items():
        if _vacio(_tomar(fin, ruta)):
            fin["avisos"].append(f"NO SE LEYÓ {que.split(' (')[0].upper()}: falta {que}.")
    fin["fuentes"] = {r: "acto" for r, v in _hojas(fin)
                      if not r.startswith(("avisos", "fuentes", "lectura"))}
    return fin


# ═══════════════════════════════════════════════════════════════════════════
# EL ESCRITO DEL RECURSO: QUIÉN RECURRE Y CON QUÉ CARÁCTER (tercera ronda)
# ═══════════════════════════════════════════════════════════════════════════
# AR 631/2025, PRUEBA DE PUNTA A PUNTA (3-oct-2026). El formulario no decía
# quién recurría; el escrito de agravios sí, en su primer párrafo: «LIC.
# NAZARIO TORRES RAMÍREZ, con la personalidad de Apoderado legal, de la persona
# moral demandada IMPULSORA DE DESARROLLOS INMOBILIARIOS V V S.A. DE C.V., hoy
# TERCERO INTERESADO en el presente Juicio Federal…». El documento salió con
# «QUEJOSA Y RECURRENTE: UNIÓN DE TRABAJADORES…», la legitimación por el
# artículo 6o. y la interposición a nombre de la quejosa: la contraparte. La
# comparecencia tiene fórmulas fijas en todo el foro —«X, con la personalidad
# de / en mi carácter de / como {figura} de Y», «X, por mi propio derecho»,
# «X, {cargo}, en representación de Y»— y se lee SIN MODELO; el modelo sólo
# entra si no sale el carácter, y lo que diga se ancla al papel.
TOPE_ESCRITO = 12000          # la comparecencia va en la primera página
_FIGURAS_ESCRITO = (
    r"apoderad[oa]s?(?:\s+(?:legal|general|especial)(?:es)?)?(?:\s+para\s+pleitos\s+y\s+cobranzas)?|"
    r"representantes?\s+legal(?:es)?|"
    r"autorizad[oa]s?(?:\s+en\s+t[eé]rminos(?:\s+amplios)?)?"
    r"(?:\s+del?\s+(?:art[ií]culo|art\.|numeral)\s*12(?:\s*,?\s*(?:primer\s+)?p[aá]rrafo(?:\s+primero)?)?"
    r"(?:\s+de\s+la\s+(?:ley\s+de\s+amparo|ley\s+de\s+la\s+materia))?)?|"
    r"delegad[oa]s?|mandatari[oa]s?(?:\s+judicial(?:es)?)?|albacea|tutor[a]?|"
    r"administrador[a]?\s+[uú]nic[oa]|gerente\s+general|director[a]?\s+general|"
    r"representante(?:\s+com[uú]n)?")
_RX_COMPARECE = re.compile(
    r"(?<![\wÁÉÍÓÚÑáéíóúñ.])(?P<n>" + _NOMBRE + r")\s*,?\s+(?i:"
    r"(?P<propio>por\s+(?:mi|su|nuestro|nuestra|sus)?\s*propio\s+derecho)|"
    r"(?:con\s+la\s+personalidad\s+de|en\s+(?:mi|su|nuestro)\s+(?:car[aá]cter|calidad)\s+de|como)\s+"
    r"(?P<fig>" + _FIGURAS_ESCRITO + r")\s*,?\s+(?:de|del|de\s+la|de\s+los|de\s+las)\s+(?P<de>[^;]{3,300})|"
    r"en\s+(?:nombre\s+y\s+)?representaci[oó]n\s+(?:de|del|de\s+la|de\s+los|de\s+las)\s+(?P<de2>[^;]{3,300}))")
# «X, Administradora Desconcentrada Jurídica de Querétaro "1", del Servicio de
# Administración Tributaria, en representación del Secretario…»: el cargo va
# entre el nombre y «en representación».
_RX_COMPARECE_CARGO = re.compile(
    r"(?<![\wÁÉÍÓÚÑáéíóúñ.])(?P<n>" + _NOMBRE + r")\s*,\s+(?P<cargo>[A-ZÁÉÍÓÚÑ][^;]{5,220}?)\s*,?\s+(?i:"
    r"en\s+(?:nombre\s+y\s+)?representaci[oó]n\s+(?:legal\s+)?(?:de|del|de\s+la|de\s+los|de\s+las)\s+)"
    r"(?P<de>[^;]{3,300})")
# «COMERCIALIZADORA DEL BAJÍO, S.A. DE C.V., por conducto de su apoderado legal
# Pedro Ruiz Gómez, quejosa en el juicio…»: la parte va primero.
_RX_POR_CONDUCTO_ESCRITO = re.compile(
    r",\s+(?i:por\s+conducto\s+de\s+su\s+(?P<fig>" + _FIGURAS_ESCRITO + r"))\s*,?\s+"
    r"(?<![\wÁÉÍÓÚÑáéíóúñ.])(?P<n>" + _NOMBRE + r")")
_RX_ABREVIATURA_FINAL = re.compile(
    r"(?:\b(?:[A-Za-zÁÉÍÓÚÑ]\.){1,3}[A-Za-z]?\.?|\b(?i:lic|licda|sr|sra|dr|dra|ing|arq|mtro|mtra|no|n[uú]m|c))\.?$")


def _parte_antes(texto: str, fin: int) -> str:
    """La parte que va ANTES de «, por conducto de su…»: desde el último corte
    («PRESENTE», o un punto que no es de abreviatura) hasta la coma."""
    ini0 = max(0, fin - 260)
    ventana = texto[ini0:fin]
    cortes = [m.end() for m in re.finditer(r"P\s?R\s?E\s?S\s?E\s?N\s?T\s?E\s*[.:,]?\s*|(?i:presente)\s*[.:,]?\s+",
                                           ventana)]
    for m in re.finditer(r"[.:;]\s+(?=[A-ZÁÉÍÓÚÑ\"«])", ventana):
        if ventana[m.start()] == "." and _RX_ABREVIATURA_FINAL.search(ventana[:m.start() + 1]):
            continue
        cortes.append(m.end())
    parte = ventana[max(cortes) if cortes else 0:].strip(" ,")
    return parte if 2 <= len(parte.split()) and re.match(r"[A-ZÁÉÍÓÚÑ0-9\"«]", parte) else ""


_RX_ANTES_DEL_NOMBRE = re.compile(
    r"^(?:(?:presente|p\s*r\s*e\s*s\s*e\s*n\s*t\s*e|el|la|suscrit[oa]|c|lic|licda|licenciad[oa]|mtr[oa]|"
    r"maestr[oa]|dr|dra|doctor[a]?|ing|arq)\.?[:,]?\s+)+", re.I)
_RX_PREFIJO_PARTE = re.compile(
    r"^(?:(?:la|el|los|las|dicha|dicho|mi|mis|su|sus)\s+)?(?P<pre>(?:(?:persona\s+(?:moral|jur[ií]dica)|"
    r"empresa|sociedad(?:\s+mercantil)?|moral|parte|autoridad(?:es)?\s+responsables?|"
    r"tercer[oa]s?\s+interesad[oa]s?|quejos[oa]s?|demandad[oa]s?|actor(?:a|es)?|codemandad[oa]s?|"
    r"recurrentes?|incidental(?:es)?|denominad[oa])\s*,?\s*)*)", re.I)
_RX_FIN_PARTE = re.compile(
    r",?\s+(?:hoy|quien(?:es)?\b|que\s+tiene|en\s+su\s+car[aá]cter|con\s+el\s+car[aá]cter|"
    r"en\s+el\s+(?:presente\s+)?juicio|parte\s+(?:quejosa|tercera)|tercer[oa]s?\s+interesad|"
    r"autoridad(?:es)?\s+(?:responsables?|demandadas?)|personalidad|se[ñn]alando|y\s+autorizando|"
    r"en\s+raz[oó]n|ante\s+usted|comparezco|ocurro|vengo)|;|(?-i:\.\s+[A-ZÁÉÍÓÚÑ][a-záéíóúñ])", re.I)
# «autoridad demandada» (cuarta ronda): en el escrito de la revisión fiscal es
# el rol de la autoridad representada, no parte de su nombre.
_RX_ROL_TRAS_PARTE = re.compile(
    r"^\s*,?\s*(?:(?:hoy|en\s+(?:su|mi)\s+car[aá]cter\s+de|con\s+(?:el\s+)?car[aá]cter\s+de|como|"
    r"quien(?:es)?\s+tienen?\s+el\s+car[aá]cter\s+de|en\s+su\s+calidad\s+de)\s+)?(?:la\s+|el\s+)?"
    r"(?:parte\s+)?(?P<r>tercer[oa]s?\s+interesad[oa]s?|quejos[oa]s?|agraviad[oa]s?|"
    r"autoridad(?:es)?\s+(?:responsables?|demandadas?)|ministerio\s+p[uú]blico)", re.I)


def _rol_de(texto: str) -> str:
    p = _plano(texto)
    if "tercer" in p:
        return "tercero"
    if "quejos" in p or "agraviad" in p:
        return "quejoso"
    if "autoridad" in p:
        return "autoridad"
    if "ministerio" in p:
        return "ministerio_publico"
    return ""


def _parte_de(texto: str) -> tuple:
    """(nombre de la parte, carácter) de lo que sigue a «… de» en la
    comparecencia: «la persona moral demandada IMPULSORA… S.A. DE C.V., hoy
    TERCERO INTERESADO en…» → («Impulsora… S.A. de C.V.», «tercero»)."""
    t = _limpio(texto)
    # «en representación de LA TITULAR de la Administración…»: el artículo de
    # un cargo de doble género es su género (cuarta ronda, E8). Se aparta, se
    # lee el nombre y se le devuelve.
    art = _RX_ART_CARGO_DOBLE.match(t)
    if art:
        nombre, car = _parte_de(t[t.index(" ") + 1:])
        return (f"{t.split()[0].lower()} {nombre}" if nombre else ""), car
    m = _RX_PREFIJO_PARTE.match(t)
    pre = m.group("pre") if m else ""
    resto = t[m.end():] if m else t
    car = _rol_de(pre)
    # Con un espacio delante: «…de la parte quejosa, comparezco…» deja, tras el
    # prefijo, «comparezco…», y eso es el fin, no un nombre.
    fin = _RX_FIN_PARTE.search(" " + resto)
    if fin and fin.start() <= 220:
        nombre, cola = resto[:max(0, fin.start() - 1)], resto[max(0, fin.start() - 1):]
    else:
        c = resto.find(",")
        nombre, cola = (resto[:c], resto[c:]) if 0 <= c <= 220 else ("", "")
    nombre = nombre.strip(" ,.;:")
    if nombre.endswith(("S.A. DE C.V", "S.A. de C.V", "C.V")):
        nombre += "."
    mr = _RX_ROL_TRAS_PARTE.match(cola)
    if mr and not car:
        car = _rol_de(mr.group("r"))
    # UN NOMBRE EMPIEZA CON MAYÚSCULA y no lleva dos puntos: lo demás es la
    # frase que sigue («comparezco para exponer: que…»).
    if len(nombre.split()) < 2 or not re.match(r"[A-ZÁÉÍÓÚÑ0-9\"«]", nombre) or ":" in nombre:
        nombre = ""
    return sin_versales(nombre), car


def _persona(n: str) -> str:
    """El nombre de quien comparece, sin «LIC.», «C.», «el suscrito» ni versales."""
    n = _RX_ANTES_DEL_NOMBRE.sub("", _limpio(n)).strip(" ,.;:")
    return sin_versales(n) if len(n.split()) >= 2 else ""


def _figura(fig: str) -> str:
    """La figura como va en la prosa: «Apoderado legal» → «apoderado legal»;
    «AUTORIZADO… DE LA LEY DE AMPARO» → «autorizado… de la Ley de Amparo» (el
    nombre de la ley conserva sus mayúsculas, rev_3)."""
    f = _limpio(fig)
    if not f:
        return ""
    f = f.lower() if f == f.upper() else f[:1].lower() + f[1:]
    return re.sub(r"\bley de amparo\b", "Ley de Amparo", f)


def _escrito_sin_modelo(tipo: str, cabeza: str) -> dict:
    out: dict = {}
    m = _RX_COMPARECE.search(cabeza)
    mc = _RX_COMPARECE_CARGO.search(cabeza)
    mp = _RX_POR_CONDUCTO_ESCRITO.search(cabeza)
    primero = min((x.start() for x in (m, mc) if x), default=len(cabeza) + 1)
    if mp and mp.start() < primero and _parte_antes(cabeza, mp.start()):
        parte = sin_versales(_parte_antes(cabeza, mp.start()))
        out.update({"promovente": parte, "representante": _persona(mp.group("n")),
                    "figura_representante": _figura(mp.group("fig"))})
        mr = _RX_ROL_TRAS_PARTE.match(cabeza[mp.end():mp.end() + 160])
        if mr:
            out["caracter"] = _rol_de(mr.group("r"))
        if tipo == "revision_fiscal":
            out["caracter"] = "autoridad"
        out["_frase"] = cabeza[max(0, mp.start() - len(parte) - 2):mp.end() + 60]
        return {k: v for k, v in out.items() if not _vacio(v)}
    # La que aparezca primero es la comparecencia; la otra puede ser una cita.
    if mc and (not m or mc.start() < m.start()):
        rep, cargo = _persona(mc.group("n")), sin_versales(_limpio(mc.group("cargo")).strip(" ,"))
        parte, car = _parte_de(mc.group("de"))
        if tipo == "revision_fiscal":
            # LA UNIDAD QUE RECURRE ES QUIEN LO INTERPONE (art. 63 LFPCA): el
            # cargo es la parte, no la figura de un representante.
            out.update({"promovente": cargo, "caracter": "autoridad"})
        else:
            # El cargo, como lo escribe el papel («Directora Jurídica de la
            # Secretaría de Gobierno»): bajarlo a minúscula a medias no es prosa.
            out.update({"promovente": parte, "representante": rep,
                        "figura_representante": cargo, "caracter": car})
        out["_frase"] = mc.group(0)[:300]
        return {k: v for k, v in out.items() if not _vacio(v)}
    if not m:
        return out
    quien = _persona(m.group("n"))
    if m.group("propio"):
        out["promovente"] = quien
        mr = _RX_ROL_TRAS_PARTE.match(cabeza[m.end():m.end() + 160])
        if mr:
            out["caracter"] = _rol_de(mr.group("r"))
    else:
        de = m.group("de") or m.group("de2") or ""
        parte, car = _parte_de(de)
        # «X, en representación de Y» dice que X representa a Y: su figura es
        # «representante», la palabra del propio papel.
        out.update({"promovente": parte, "representante": quien, "caracter": car,
                    "figura_representante": _figura(m.group("fig") or "") or
                    ("representante" if m.group("de2") else "")})
    if tipo == "revision_fiscal":
        out["caracter"] = "autoridad"
    out["_frase"] = m.group(0)[:300]
    return {k: v for k, v in out.items() if not _vacio(v)}


# CÓMO SE LE NOTIFICÓ LA RECURRIDA, SEGÚN QUIEN RECURRE: «…la Sentencia
# Definitiva de fecha 22 veintidós de abril de 2025…, y misma que me fuera
# notificada en listas, el día 30 treinta de abril de 2025» (AR 631). Sólo la
# fórmula en primera persona sobre lo que se recurre («me/nos/le fue(ra)
# notificada»): una notificación de otra cosa no cuenta. La forma alimenta el
# cómputo de la autoridad (D2, `redactor_adelanto.forma_de_notificacion`); la
# fecha entra detrás de la del secretario y, si no cuadra, se dice.
_RX_NOTIFICADA = re.compile(
    r"\b(?:me|nos|le|les)\s+(?:fue(?:ra|se)?|fueron|ha\s+sido|haya\s+sido)\s+notificad[oa]s?\s*,?\s+"
    r"(?:(?:en|por|mediante|a\s+trav[eé]s\s+de|v[ií]a|de\s+manera|en\s+forma)\s+)?(?:las?\s+|el\s+)?"
    r"(?P<f>listas?(?:\s+de\s+acuerdos)?|boletin\s+jurisdiccional|oficio|electr[oó]nic\w*|portal\s+de\s+servicios"
    r"|correo\s+electr[oó]nico|personal(?:mente)?)", re.I)
_FORMA_DEL_ESCRITO = (("lista", r"^lista"), ("boletin", r"^boletin"), ("oficio", r"^oficio"),
                      ("electronica", r"electr|portal|correo"), ("personal", r"^personal"))


def _notificacion_del_escrito(cabeza: str) -> dict:
    m = _RX_NOTIFICADA.search(cabeza)
    if not m:
        return {}
    f = _plano(m.group("f"))
    out = {}
    for clave, rx in _FORMA_DEL_ESCRITO:
        if re.search(rx, f):
            out["forma_notificacion"] = clave
            break
    fs = fechas_del_texto(cabeza[m.end():m.end() + 90])
    if fs:
        out["notificacion"] = fs[0][0]
    return out


_RX_ROL_EN_PAPEL = {
    "tercero": re.compile(r"tercer[oa]s?\s+interesad", re.I),
    "quejoso": re.compile(r"quejos[oa]|agraviad[oa]", re.I),
    "autoridad": re.compile(r"autoridad(?:es)?\s+responsables?|autoridad\s+demandada", re.I),
    "ministerio_publico": re.compile(r"ministerio\s+p[uú]blico", re.I),
}


def _prompt_escrito(tipo: str, cabeza: str) -> str:
    rec = {"amparo_revision": "un recurso de revisión", "queja": "un recurso de queja",
           "revision_fiscal": "un recurso de revisión fiscal"}.get(tipo, "un recurso")
    # LA QUEJA: ADEMÁS, LA DEMANDA DE AMPARO SI EL ESCRITO LA DESCRIBE (sexta
    # ronda, 3-oct-2026, C3). Separada de quién recurre: no es este escrito.
    regla_queja = campos_queja = ""
    if tipo == "queja":
        regla_queja = """- LA DEMANDA DE AMPARO («quejoso», «demanda_fecha», «autoridades»,
  «actos») NO ES ESTE ESCRITO: es la que promovió el quejoso en el juicio de
  amparo, y sólo si este escrito la describe (quién la promovió, cuándo se
  presentó, contra qué autoridades y qué actos reclamó). No copies lo que
  decidió el auto recurrido ni los agravios; el juzgado que dictó el auto NUNCA
  es una autoridad responsable. Si no la describe, vacío.
- LA AMPLIACIÓN DE DEMANDA NO ES LA DEMANDA: si el escrito habla de una
  ampliación, la fecha, las autoridades y los actos son los de la demanda
  INICIAL; no copies los de la ampliación. Si sólo describe la ampliación,
  vacío.
"""
        campos_queja = """,
 "quejoso": "quien promovió el juicio de amparo, literal, o vacío",
 "demanda_fecha": "AAAA-MM-DD en que se presentó la demanda de amparo, sólo si el escrito la escribe",
 "autoridades": ["cada autoridad responsable que señaló la demanda de amparo, literal"],
 "actos": ["cada acto que reclamó la demanda de amparo, COPIADO TAL CUAL del escrito, COMPLETO hasta su punto final; si pasa de 120 palabras, sus primeras 120 seguidas"]"""
    return f"""Eres el secretario de un Tribunal Colegiado. Tienes delante la primera parte
del ESCRITO con el que se interpone {rec}. Di QUIÉN lo interpone y con qué carácter.

Reglas que no se negocian:
- COPIA LITERAL de los nombres, como están escritos.
- SI NO CONSTA, CADENA VACÍA. Esto se firma.
- «promovente» es la PARTE que recurre (la persona física o moral, o la
  autoridad); «representante», la persona que firma POR ella (su apoderado,
  autorizado o delegado), o vacío si firma por su propio derecho.
{regla_queja}
EL ESCRITO:
──────────────────────────────────────────
{cabeza[:8000]}
──────────────────────────────────────────

Devuelve JSON y nada más:
{{"promovente": "la parte que recurre, literal",
 "caracter": "quejoso | tercero | autoridad | ministerio_publico | vacío",
 "representante": "quien firma por ella, literal, o vacío",
 "figura_representante": "apoderado legal | autorizado | delegado | representante legal…, literal, o vacío"{campos_queja}}}"""


async def leer_escrito(cliente, texto_escrito: str, tipo: str) -> dict:
    """QUIÉN RECURRE, LEÍDO DEL ESCRITO DEL RECURSO (AR, queja y revisión
    fiscal). Ficha parcial: {promovente, caracter (quejoso|tercero|autoridad|
    ministerio_publico), representante, figura_representante, y si el escrito
    lo dice en primera persona, forma_notificacion (lista|boletin|oficio|
    electronica|personal) y notificacion (ISO); avisos, fuentes (todas
    «escrito»), lectura: {modo, frase}}. Lo que no se leyó no aparece.

    Primero SIN MODELO (las fórmulas de la comparecencia); si el carácter no
    sale y hay `cliente`, UNA llamada JSON corta (`MODELO_ACTO`, con tope de
    `TOPE_SEGUNDOS_MODELO`) cuyos nombres se anclan a la cabeza del escrito y
    cuyo carácter sólo se cree si el escrito usa esa palabra. Lo leído sin
    modelo manda. En el amparo directo el escrito es la demanda: no hay nada
    que leer aquí ({"avisos": []}).

    EN LA QUEJA, ADEMÁS LA DEMANDA DE AMPARO (sexta ronda, 3-oct-2026, C3): si
    hay `cliente`, la llamada se hace siempre y pide también quejoso,
    demanda_fecha, autoridades y actos, anclados al escrito con las guardas
    del amparo en revisión (`_demanda_anclada`). Salen como `quejoso` y
    `demanda.{fecha, autoridades, actos}`; en `armar` van DETRÁS de lo que dice
    el auto recurrido (la descripción del juzgado manda sobre la de quien
    recurre)."""
    tp = _tipo(tipo)
    if tp not in ("amparo_revision", "queja", "revision_fiscal"):
        return {"avisos": []}
    cabeza = " ".join(_RX_CONTROL.sub("", str(texto_escrito or ""))[:TOPE_ESCRITO].split())
    if len(cabeza) < 80:
        return {"avisos": ["DEL ESCRITO DEL RECURSO NO SALIÓ TEXTO LEGIBLE: quién recurre y con qué "
                           "carácter salen del formulario o del auto."]}
    avisos: list = []
    try:
        det = await asyncio.to_thread(_escrito_sin_modelo, tp, cabeza)
    except Exception as ex:
        det = {}
        avisos.append(f"La lectura sin modelo del escrito falló ({type(ex).__name__}).")
    frase = det.pop("_frase", "")
    modo = "sin_modelo" if det else ""
    try:
        for k, v in _notificacion_del_escrito(cabeza).items():
            det.setdefault(k, v)
    except Exception:
        pass
    if cliente and (not det.get("caracter") or tp == "queja"):
        try:
            import llamada_modelo as _lm
            # En la queja la respuesta trae además los actos reclamados (hasta
            # 120 palabras cada uno): más tope de salida.
            kw = dict(model=MODELO_ACTO,
                      messages=[{"role": "user", "content": _prompt_escrito(tp, cabeza)}],
                      max_completion_tokens=1500 if tp == "queja" else 600,
                      response_format={"type": "json_object"})
            if ESFUERZO_ACTO:
                kw["reasoning_effort"] = ESFUERZO_ACTO
            r = await asyncio.wait_for(_lm.crear(cliente, **kw), timeout=TOPE_SEGUNDOS_MODELO)
            d = _json_de(r)
            pp = Papel(cabeza)
            # LA DEMANDA (queja): aparte de quién recurre, con sus propias guardas.
            mod_dem: dict = {}
            if tp == "queja":
                q_ = _limpio(d.get("quejoso")) if isinstance(d.get("quejoso"), str) else ""
                if q_:
                    a_q, _m = anclar(q_, pp)
                    if a_q:
                        mod_dem["quejoso"] = sin_versales(a_q)
                    else:
                        avisos.append(f"SE DESCARTÓ «{q_[:90]}» como quejoso (escrito): no aparece en el papel. "
                                      f"Escríbelo tú si es correcto.")
                _demanda_anclada(d, pp, avisos, mod_dem)
            for k, v in mod_dem.items():
                det.setdefault(k, v)
            mod: dict = {}
            for k in ("promovente", "representante", "figura_representante"):
                v = d.get(k)
                v = _limpio(v) if isinstance(v, str) else ""
                if not v:
                    continue
                a_, _m = anclar(v, pp)
                if a_:
                    mod[k] = sin_versales(a_) if k != "figura_representante" else a_.lower()
                else:
                    avisos.append(f"SE DESCARTÓ «{v[:90]}» como {k.replace('_', ' ')} del escrito: no "
                                  f"aparece en el papel.")
            car = _plano(d.get("caracter") if isinstance(d.get("caracter"), str) else "").replace(" ", "_")
            if car in _RX_ROL_EN_PAPEL:
                if _RX_ROL_EN_PAPEL[car].search(cabeza):
                    mod["caracter"] = car
                else:
                    avisos.append(f"EL MODELO DIJO QUE SE RECURRE COMO «{car.replace('_', ' ')}» y el escrito "
                                  f"no usa esa palabra: no se tomó.")
            # Lo del modelo sólo completa a la MISMA parte que se leyó sin modelo.
            if det.get("promovente") and mod.get("promovente") and \
                    not _misma_parte(det["promovente"], mod["promovente"]):
                mod = {}
            puso = [k for k in mod if k not in det]
            for k, v in mod.items():
                det.setdefault(k, v)
            if (mod if tp != "queja" else puso) or mod_dem:
                modo = modo + "+modelo" if modo else "modelo"
        except asyncio.TimeoutError:
            avisos.append(f"LA LECTURA CON MODELO DEL ESCRITO NO CONTESTÓ EN {TOPE_SEGUNDOS_MODELO} s: vale "
                          f"lo que se leyó sin modelo.")
        except Exception as ex:
            avisos.append(f"No se pudo leer el escrito con el modelo ({type(ex).__name__}): vale lo que se "
                          f"leyó sin modelo.")
    if tp == "revision_fiscal":
        det.setdefault("caracter", "autoridad")
    if not det.get("caracter"):
        avisos.append("NO SE LEYÓ DEL ESCRITO CON QUÉ CARÁCTER SE RECURRE (quejoso, tercero interesado o "
                      "autoridad responsable): dilo en el formulario; está en el primer párrafo del escrito.")
    out = {k: v for k, v in det.items() if not _vacio(v)}
    out["avisos"] = avisos
    # Por ruta (la demanda va anidada: «demanda.fecha», «demanda.actos»…).
    out["fuentes"] = {r: "escrito" for r, _v in _hojas(out) if not r.startswith("avisos")}
    if modo:
        out["lectura"] = {"modo": modo, "frase": _limpio(frase).lstrip(" .,;:")}
    return out


# ═══════════════════════════════════════════════════════════════════════════
# EL FORMULARIO: IDA Y VUELTA
# ═══════════════════════════════════════════════════════════════════════════
_MP_ALIAS = {"pedimento": "pedimento", "si": "pedimento", "sí": "pedimento", "formulo": "pedimento",
             "con_pedimento": "pedimento", "sin_pedimento": "sin_pedimento", "no": "sin_pedimento",
             "no_formulo": "sin_pedimento", "": "", "no_consta": "", "no consta": ""}
_FORMAS_NOTIF = ("personal", "lista", "oficio", "electronica", "boletin")

# ── LOS TRES CAMPOS NUEVOS (segunda ronda, 3-oct-2026) ─────────────────────
_ROMANOS_63 = ("I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X")
_RX_PREFIJO_FRACCION = re.compile(
    r"^(?:(?:del?\s+)?(?:art[ií]culo|art\.)\s*63\s*,?\s*)?"
    r"(?:fracci[oó]n|fracc?\.|fr\.)?\s*", re.I)


def fraccion_63_de(valor) -> str:
    """«3», «iii», «fracción III», «fr. III.» → «III». «» si no es una
    fracción del artículo 63 de la LFPCA (I a X).

    ESTRICTO A PROPÓSITO: «III, inciso a)» o «la tercera» no se adivinan. La
    fracción va a la procedencia, que en revisión fiscal es OFICIOSA; una
    fracción mal leída se firma, un hueco con aviso se ve."""
    s = _limpio(valor)
    if not s:
        return ""
    s = _RX_PREFIJO_FRACCION.sub("", s).strip(" .,;)").upper()
    if s.isdigit():
        n = int(s)
        return _ROMANOS_63[n - 1] if 1 <= n <= len(_ROMANOS_63) else ""
    return s if s in _ROMANOS_63 else ""


_RX_PESOS_RUIDO = re.compile(
    r"(?i)\$|\bm\.?\s?n\.?(?=\s|$)|\bmxn\b|\bpesos?\b|\bmoneda\s+nacional\b|\(|\)")
_RX_MONTO_MX = re.compile(r"^(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d{1,2})?$")


def cuantia_de(valor) -> str:
    """El monto en pesos en una sola forma, «$847,738.77»; «indeterminada» si
    así lo dice el secretario (la fracción II del 63 la admite); «» si no se
    puede leer como monto.

    UNA SOLA FORMA porque la procedencia la compara contra el umbral de UMAs
    (`fase_procedencia_rf`) y la escribe en la prosa: «1.234.567,00» (formato
    europeo) o «un millón» no se interpretan, se avisan. Es idempotente: lo
    que devuelve vuelve a dar lo mismo, y la ida y vuelta del formulario no
    cambia el dato."""
    s = _limpio(valor)
    if not s:
        return ""
    if re.search(r"indetermin", _plano(s)):
        return "indeterminada"
    n = re.sub(r"\s+", "", _RX_PESOS_RUIDO.sub(" ", s)).rstrip(".")
    if not _RX_MONTO_MX.match(n):
        return ""
    try:
        v = float(n.replace(",", ""))
    except ValueError:
        return ""
    return f"${v:,.2f}" if v > 0 else ""


_RX_CONFORME = re.compile(
    r"^(?:conforme\s+(?:a\s+lo\s+dispuesto\s+(?:en|por)\s+|al?\s+)|en\s+t[ée]rminos\s+del?\s+|"
    r"de\s+conformidad\s+con\s+|seg[uú]n\s+|con\s+fundamento\s+en\s+)", re.I)


# EL PRECEPTO DEL SURTIMIENTO QUE NO LO ES (tercera ronda, 3-oct-2026, rev_1).
# AD 274/2025: el extractor dio «artículo 17 del mencionado ordenamiento» —el
# PLAZO de la Ley de Amparo y, además, una remisión a una ley que no nombra— y
# se aceptaba; si hubiera llegado, el considerando habría fundado el surtimiento
# en el precepto del plazo. En el amparo directo la notificación del acto
# reclamado surte efectos conforme a la LEY QUE RIGE EL ACTO: los artículos 17
# (plazo), 18 (cómputo), 19 (días hábiles) y 31 (surtimiento de las
# notificaciones DEL JUICIO DE AMPARO) de la Ley de Amparo no lo fundan. Y una
# remisión sin ley («del mencionado ordenamiento», «de la citada ley», «de la
# ley de la materia») suelta en un considerando no dice nada.
_RX_REMISION_SIN_LEY = re.compile(
    r"\b(?:mencionad[oa]|citad[oa]|precitad[oa]|invocad[oa]|referid[oa]|aludid[oa]|indicad[oa]|"
    r"propi[oa]|mism[oa]|dich[oa])s?\s+(?:ordenamiento|ley|c[oó]digo|legislaci[oó]n|"
    r"cuerpo\s+(?:legal|normativo)|norma|reglamento)\b|"
    r"\bordenamiento\s+(?:legal\s+)?(?:en\s+cita|invocado|citado|aludido)\b|"
    r"\bley\s+de\s+la\s+materia\b", re.I)
_RX_NOMBRA_LEY = re.compile(
    r"\b(?:ley|c[oó]digo|reglamento|constituci[oó]n|acuerdo|estatuto|lineamientos?|decreto)\b", re.I)
# Las siglas con que se escribe una ley («del CPC», «de la LFPCA»). En un texto
# todo en versales cualquier palabra parecería sigla: ahí sólo valen éstas.
_SIGLAS_LEY = ("CPC", "CPCE", "CFPC", "CNPCF", "CNPP", "LFPCA", "LFPA", "LFT", "CCOM", "CFF",
               "LGTOC", "LOPJF", "CPEUM", "LSS", "LISSSTE", "LA")
_RX_SIGLA = re.compile(r"\b[A-ZÁÉÍÓÚÑ]{2,6}\b")
_ARTS_LA_NO_SURTEN = ("17", "18", "19", "31")


def motivo_fundamento(valor, tipo: str = "") -> str:
    """«» si el precepto del surtimiento puede ir a la prosa; si no, POR QUÉ
    (para el aviso). Las remisiones sin ley y los artículos sin ley se
    rechazan siempre; los artículos 17, 18, 19 y 31 de la Ley de Amparo, sólo
    en el amparo directo (en los recursos el surtimiento SÍ lo rige el 31)."""
    s = _limpio(valor)
    if not s:
        return ""
    if _RX_REMISION_SIN_LEY.search(s):
        return ("remite a una ley que no nombra («mencionado ordenamiento», «citada ley»): "
                "escríbelo con el nombre de la ley")
    versales = s == s.upper()
    siglas = [x for x in _RX_SIGLA.findall(s) if (x in _SIGLAS_LEY if versales else True)]
    if not (_RX_NOMBRA_LEY.search(s) or siglas):
        return "no dice de qué ley es el artículo"
    if _tipo(tipo) == "amparo_directo" and re.search(r"ley\s+de\s+amparo", _plano(s)):
        nums = re.findall(r"(?<![\d/.\-])(\d{1,3})(?![\d/])", s)
        malos = [n for n in _ARTS_LA_NO_SURTEN if n in nums]
        if malos:
            return (f"cita el artículo {', '.join(malos)} de la Ley de Amparo, que no rige el "
                    f"surtimiento de la notificación del acto reclamado: en el amparo directo lo "
                    f"rige la ley del acto")
    return ""


def fundamento_surtimiento_de(valor, tipo: str = "") -> str:
    """El precepto del surtimiento tal como entra en la prosa, con su artículo:
    «el artículo 126 del Código de Procedimientos Civiles del Estado de
    Querétaro». Los considerandos lo escriben tras «conforme a» (y contraen
    «a el» → «al»), así que:
      · se quita el conector que el secretario haya tecleado delante
        («conforme al…», «en términos del…»), para no escribirlo dos veces;
      · «artículo 126…» / «art. 126…» gana su «el» y «artículos…» su «los»;
      · sin punto final (la oración sigue).
    Lo demás se respeta: es el dato del secretario. «» si no es un precepto que
    pueda fundar el surtimiento (`motivo_fundamento`)."""
    s = _limpio(valor).rstrip(" .;,")
    if not s or motivo_fundamento(s, tipo):
        return ""
    if s == s.upper() and re.search(r"[A-ZÁÉÍÓÚÑ]{4,}", s):
        s = sin_versales(s)          # «ARTÍCULO 126 DEL CÓDIGO…» → prosa
    s = _RX_CONFORME.sub("", s).strip()
    s = re.sub(r"(?i)^arts\.?\s*(?=\d)", "artículos ", s)
    s = re.sub(r"(?i)^art\.?\s*(?=\d)", "artículo ", s)
    if re.match(r"(?i)^art[ií]culos\b", s):
        s = "los a" + s[1:]
    elif re.match(r"(?i)^art[ií]culo\b", s):
        s = "el a" + s[1:]
    return s


# ── LA VÍA DE PRESENTACIÓN (tercera ronda, 3-oct-2026, rev_1) ───────────────
# Cuatro fichas del banco traían «tribunal» cuando la demanda entró por la
# Oficialía del Tribunal Superior de Justicia del Estado —por conducto de la
# responsable, art. 176— y el compositor afirmó en falso «ante este Tribunal
# Colegiado». El valor era ambiguo: un secretario que elija «Tribunal» pensando
# en el Tribunal Superior provoca lo mismo. Ahora se llama «tribunal_colegiado»
# (ESTE Tribunal Colegiado, art. 176, segundo párrafo); un «tribunal» suelto se
# lee así, y lo que no se reconoce no entra.
VIAS = ("responsable", "juzgado", "tribunal_colegiado", "electronica", "postal")


def via_de(valor) -> str:
    """La vía de presentación en la forma del esquema (`VIAS`), o «» si no se
    reconoce. «tribunal» → «tribunal_colegiado»; «Tribunal Superior de
    Justicia» no es ninguna (es la responsable, pero eso no se adivina)."""
    p = _plano(valor if isinstance(valor, str) else "").replace("_", " ")
    if not p:
        return ""
    if re.search(r"electr|portal|en\s+linea", p):
        return "electronica"
    if re.search(r"postal|correo|sepomex", p):
        return "postal"
    if re.search(r"colegiado|^tribunal$|^este\s+tribunal$|partes\s+de\s+este\s+tribunal", p):
        return "tribunal_colegiado"
    if re.search(r"^(?:la\s+)?(?:autoridad\s+)?responsable$|por\s+conducto\s+de\s+la\s+responsable", p):
        return "responsable"
    if re.search(r"^(?:el\s+)?(?:juzgado|juez)(?:\s+de\s+distrito)?$", p):
        return "juzgado"
    return ""


# ── EL CORREO: UNA SOLA REGLA (CUARTA RONDA, 3-oct-2026, E6) ────────────────
# RF 2/2025: la ficha traía la vía «responsable» —el oficio lo recibe la Sala,
# así que es plausible— y un depósito en el correo del 24-oct-2024. El
# compositor narró el depósito en cuanto vio la fecha («mediante oficio
# depositado en el Servicio Postal Mexicano…»), y el cómputo
# (`redactor_adelanto.deposito_que_cuenta`) lo descartó porque la vía no decía
# «postal»: el resultando dijo que se depositó y el considerando DESECHÓ por
# extemporáneo. Dos reglas para un mismo hecho. Ahora hay una: un depósito que
# consta y no es posterior a la presentación dice que el recurso fue por
# correo, diga lo que diga la vía (y `validar` avisa del choque).
def via_postal(ficha, presentacion=None) -> bool:
    """¿El recurso se mandó por correo? True si consta `deposito_postal` y no
    es posterior a la presentación (o no hay presentación con que
    compararlo), aunque la vía diga otra cosa; o si `via_presentacion` es
    «postal». `presentacion` (fecha o ISO) sustituye a la de la ficha cuando
    quien llama la tiene aparte (el `Encargo`). Lo usan el compositor,
    `redactor_adelanto.deposito_que_cuenta` y main: la misma respuesta en las
    tres piezas. Nunca lanza."""
    try:
        f = ficha if isinstance(ficha, dict) else {}
        dep = _fecha_obj(f.get("deposito_postal"))
        pres = _fecha_obj(presentacion if presentacion else f.get("presentacion"))
        if dep and (pres is None or dep <= pres):
            return True
        return via_de(f.get("via_presentacion")) == "postal"
    except Exception:
        return False


def _por_tipo(parcial: dict, tipo: str) -> dict:
    """Lleva «organo_acto» y «expediente_origen» —que `de_formulario` guarda
    en `acto.organo` / `acto.expediente`— también al campo que ESE tipo usa
    (`_POR_TIPO_FORMULARIO`): la responsable en AD, la Sala y el expediente del
    TFJA en RF. NO PISA lo que el formulario mandó con su propio nombre
    («responsable», «sala», «expediente_tfja» explícitos mandan) y sólo copia lo
    que es del secretario. Idempotente; trabaja sobre el dict que recibe."""
    mapa = _POR_TIPO_FORMULARIO.get(tipo or "")
    if not mapa or not isinstance(parcial, dict):
        return parcial
    fuentes = parcial.setdefault("fuentes", {})
    for clave, rutas in mapa.items():
        origen = _MAPA_FORMULARIO[clave]          # acto.organo | acto.expediente
        v = _tomar(parcial, origen)
        if _vacio(v) or fuentes.get(origen) != "secretario":
            continue
        for ruta in rutas:
            if ruta == origen or not _vacio(_tomar(parcial, ruta)):
                continue
            _poner(parcial, ruta, v)
            fuentes[ruta] = "secretario"
    return parcial


# ── LO QUE ENTRA POR EL FORMULARIO (tercera ronda, 3-oct-2026, rev_6) ──────
# TOPES POR CLASE DE CAMPO. Starlette admite hasta 1 MB por campo y aquí no se
# recortaba nada: el valor llegaba a los resultandos y a
# `meta_lenguaje.frases`, cuadrática sin puntos. Medido con «ponente_turno»:
# 5 000 caracteres → 0.84 s; 20 000 → 11.5 s; 40 000 → 45.1 s, síncronos en el
# bucle de eventos. LO QUE EXCEDE NO SE RECORTA —un nombre cortado se firma—:
# se ignora y se dice.
TOPE_NOMBRE = 300
TOPE_NUMERO = 40
TOPE_FUNDAMENTO = 400
_TOPE_FECHA = 120
_TOPE_CORTO = 40
_RUTAS_NUMERO = {"acto.toca", "acto.expediente", "expediente_tfja", "fraccion_63", "cuantia",
                 "fraccion_97", "inciso_97", "numero"}
_RUTAS_CORTAS = {"ministerio_publico", "forma_notificacion", "via_presentacion"}


def _tope_de(ruta: str) -> int:
    if _es_ruta_fecha(ruta):
        return _TOPE_FECHA
    if ruta in _RUTAS_NUMERO:
        return TOPE_NUMERO
    if ruta in _RUTAS_CORTAS:
        return _TOPE_CORTO
    if ruta == "fundamento_surtimiento":
        return TOPE_FUNDAMENTO
    if ruta == "relacionados":
        return TOPE_TEXTO_RELACIONADOS
    return TOPE_NOMBRE


# ── LOS ASUNTOS RELACIONADOS: DEL TEXTO A LA LISTA Y DE VUELTA (sexta ronda) ─
def _numero_relacionado(x) -> str:
    """«452 / 2025» → «452/2025»; lo demás, tal cual (y `relacionados_de`
    decide si tiene la forma)."""
    return re.sub(r"\s+", "", _limpio(x))


def _estado_relacionado(x) -> str:
    """«misma_sesion» | «resuelto»; vacío → «misma_sesion» (lo que la tarjeta
    marca por omisión); lo que no se reconoce → «»."""
    s = _plano(x).replace(" ", "_").replace("-", "_").strip("_")
    if not s:
        return "misma_sesion"
    return _ESTADOS_ALIAS.get(s, "")


def _prosa_relacionado_aviso(tipo: str, numero: str) -> str:
    return f"{_TIPO_EN_AVISO.get(tipo, tipo.replace('_', ' '))} {numero}".strip()


def relacionados_de(valor, numero: str = "", tipo: str = "") -> tuple:
    """Los asuntos relacionados que marcó el secretario → (lista, avisos).

    `valor`: el texto del formulario («tipo|numero|estado» separados por «;»)
    o una lista ya hecha (de dicts {"tipo", "numero", "estado"} o de cadenas
    «tipo|numero|estado»). La lista sale normalizada: cada elemento
    {"tipo": uno de TIPOS_RELACIONADO, "numero": «452/2025», "estado":
    «misma_sesion» | «resuelto»}, sin duplicados (mismo tipo y número: vale el
    primero) y cuatro como mucho.

    FUERA, CON AVISO: el tipo que no es de los cuatro, el número que no tiene
    la forma «452/2025», el estado que no se reconoce (sin estado, «misma
    sesión») y EL PROPIO ASUNTO —`numero` (y `tipo`, si se sabe) del asunto que
    se resuelve—: un asunto no se relaciona consigo mismo. Con el mismo número
    pero OTRO tipo es otro expediente (el amparo directo 33/2024 y la revisión
    fiscal 33/2024 tienen cada uno su registro) y se queda.

    Pura; nunca lanza."""
    avisos: list = []
    if valor is None:
        return [], avisos
    if isinstance(valor, str):
        crudos = [x for x in (y.strip() for y in valor.split(";")) if x]
    elif isinstance(valor, (list, tuple)):
        crudos = list(valor)
    else:
        return [], [f"LOS ASUNTOS RELACIONADOS NO SE PUDIERON LEER (llegó {_que_es(valor)}) (relacionados): "
                    f"se ignoraron. Márcalos otra vez en «Trámite en este tribunal»."]
    propio_n = _numero_relacionado(numero)
    propio_t = _tipo(tipo) if tipo else ""
    lista: list = []
    vistos = set()
    sobran = 0
    for x in crudos:
        if isinstance(x, dict):
            t_, n_, e_ = x.get("tipo"), x.get("numero"), x.get("estado")
            t_ = t_ if isinstance(t_, str) else ""
            n_ = n_ if isinstance(n_, str) else ""
            e_ = e_ if isinstance(e_, str) else ""
            dicho = "|".join(_limpio(v) for v in (t_, n_, e_)).strip("|")
        elif isinstance(x, str):
            partes = [p.strip() for p in x.split("|")]
            t_, n_, e_ = (partes + ["", "", ""])[:3]
            dicho = _limpio(x)
        else:
            avisos.append(f"UN ASUNTO RELACIONADO NO SE PUDO LEER (llegó {_que_es(x)}) (relacionados): se quitó.")
            continue
        if not _limpio(dicho):
            continue
        tp_ = _tipo(t_)
        if not tp_:
            avisos.append(f"«{dicho[:80]}» NO ES UN ASUNTO RELACIONADO VÁLIDO (relacionados): el tipo "
                          f"«{_limpio(t_)[:40]}» no es amparo directo, amparo en revisión, queja ni revisión "
                          f"fiscal. Se quitó.")
            continue
        num_ = _numero_relacionado(n_)
        if not _RX_NUMERO_RELACIONADO.match(num_):
            avisos.append(f"«{dicho[:80]}» NO ES UN ASUNTO RELACIONADO VÁLIDO (relacionados): el número "
                          f"«{_limpio(n_)[:40]}» no tiene la forma «452/2025». Se quitó.")
            continue
        est_ = _estado_relacionado(e_)
        if not est_:
            avisos.append(f"«{dicho[:80]}» NO ES UN ASUNTO RELACIONADO VÁLIDO (relacionados): el estado "
                          f"«{_limpio(e_)[:40]}» no es «misma_sesion» (se resuelve en la misma sesión) ni "
                          f"«resuelto» (ya se resolvió). Se quitó.")
            continue
        if propio_n and num_ == propio_n and (not propio_t or tp_ == propio_t):
            avisos.append(f"EL ASUNTO RELACIONADO «{_prosa_relacionado_aviso(tp_, num_)}» ES ESTE MISMO ASUNTO "
                          f"(relacionados): se quitó. Un asunto no se relaciona consigo mismo; si es otro "
                          f"expediente, revisa el tipo o el número.")
            continue
        if (tp_, num_) in vistos:
            continue
        if len(lista) >= TOPE_RELACIONADOS:
            sobran += 1
            continue
        vistos.add((tp_, num_))
        lista.append({"tipo": tp_, "numero": num_, "estado": est_})
    if sobran:
        avisos.append(f"SE MARCARON {len(lista) + sobran} ASUNTOS RELACIONADOS Y EL TOPE ES {TOPE_RELACIONADOS} "
                      f"(relacionados): se tomaron los {TOPE_RELACIONADOS} primeros.")
    return lista, avisos


def relacionados_a_texto(lista) -> str:
    """La lista de la ficha → el texto del formulario: «amparo_directo|452/2025|
    misma_sesion;revision_fiscal|33/2024|resuelto». «» sin lista. Lo que no
    es un relacionado válido no se escribe (`relacionados_de`)."""
    if isinstance(lista, str):
        lista = relacionados_de(lista)[0]
    if not isinstance(lista, (list, tuple)):
        return ""
    buenos, _av = relacionados_de(list(lista))
    return ";".join(f"{r['tipo']}|{r['numero']}|{r['estado']}" for r in buenos)


def _que_es(v) -> str:
    if isinstance(v, bool):
        return "un sí/no"
    if isinstance(v, (int, float)):
        return "un número"
    if isinstance(v, (list, tuple, set)):
        return "una lista"
    if isinstance(v, dict):
        return "un objeto"
    return type(v).__name__


def de_formulario(form, tipo: str = "", *, extra: bool = False, numero: str = "") -> dict:
    """El `tramite_json` del front → ficha parcial, todo con fuente
    «secretario». Lo que viene mal escrito (una fecha que no es fecha, una
    fracción que no es del 63, un monto que no es monto) no entra y se avisa.

    `tipo` (o «tipo_asunto»/«tipo» dentro del formulario) decide a qué campo va
    lo que la tarjeta llama «organo_acto» y «expediente_origen»
    (`_POR_TIPO_FORMULARIO`). Sin tipo van a `acto.organo` / `acto.expediente`
    y `armar`, que sí conoce el tipo, completa el resto con `_por_tipo`.

    «relacionados» (SEXTA RONDA, 3-oct-2026) llega como texto «tipo|numero|
    estado;…» y sale como la lista de `relacionados_de`, con sus avisos. El
    PROPIO NÚMERO del asunto, que no puede ser relacionado suyo, sale de
    `numero` o de «numero» dentro del formulario (contexto, como «tipo»: no
    entra a la ficha); por HTTP no viene, y `armar` lo quita con el número de
    la ficha.

    SÓLO LAS CLAVES DEL CONTRATO (tercera ronda, 3-oct-2026, rev_6; 22 desde la sexta). Por
    HTTP (`redactor_adelanto.tramite_de_formulario`) entra lo que la tarjeta
    manda y nada más. Antes `presentacion`, `notificacion` o `sentido` metidos
    a mano en el JSON entraban como del secretario: el resultando decía
    «presentado el cinco de febrero» y la oportunidad «si se presentó el tres»,
    y `sentido` se saltaba el catálogo. Las claves de `_MAPA_FORMULARIO_EXTRA`
    y las listas `derechos`/`terceros` entran SÓLO con `extra=True`: usos
    internos explícitos (`armar` con un dict plano, los bancos).

    SÓLO TEXTO. Un valor que no es cadena —{"nombre": "Ana"}, ["374/2025"]—
    se escribía con su repr de Python («a la ponencia de {'nombre': 'Ana'}»).
    Ahora se ignora y se dice (las fechas admiten además `date`; con
    `extra`, `derechos` y `terceros` admiten una lista de cadenas). Y cada
    campo con su tope (`_tope_de`)."""
    if isinstance(form, str):
        try:
            form = json.loads(form or "{}")
        except (ValueError, TypeError):
            return {"avisos": ["EL TRÁMITE DEL FORMULARIO NO SE PUDO LEER (JSON mal formado): "
                               "se ignoró."], "fuentes": {}}
    if not isinstance(form, dict):
        return {"avisos": [], "fuentes": {}}
    _tf = form.get("tipo_asunto") or form.get("tipo")
    tp = _tipo(tipo or (_limpio(_tf) if isinstance(_tf, str) else ""))
    _nf = form.get("numero")
    propio = _limpio(numero) or (_limpio(_nf) if isinstance(_nf, str) else "")
    out: dict = {"avisos": [], "fuentes": {}}
    mapa = dict(_MAPA_FORMULARIO, **_MAPA_FORMULARIO_EXTRA) if extra else dict(_MAPA_FORMULARIO)
    listas = ("derechos", "terceros") if extra else ()
    ajenas = [str(k)[:40] for k in form
              if k not in mapa and k not in listas and k not in ("tipo_asunto", "tipo", "numero")
              and not _vacio(form.get(k))]
    if ajenas:
        out["avisos"].append(
            f"SE IGNORÓ LO QUE NO ES DE «TRÁMITE EN ESTE TRIBUNAL» ({', '.join(ajenas[:5])}"
            f"{'…' if len(ajenas) > 5 else ''}): esos datos van en el formulario principal o se "
            f"leen de los papeles.")
    for k, ruta in mapa.items():
        if k not in form:
            continue
        v = form.get(k)
        if v is None:
            continue
        if isinstance(v, _dt.date) and _es_ruta_fecha(ruta):
            v = v.date().isoformat() if isinstance(v, _dt.datetime) else v.isoformat()
        # LOS RELACIONADOS YA EN LISTA (sexta ronda): sólo en usos internos
        # (`extra`: `armar` con un dict plano, los bancos). Por HTTP, texto.
        if ruta == "relacionados" and extra and isinstance(v, (list, tuple)):
            lista, avs = relacionados_de(list(v), propio, tp)
            out["avisos"].extend(avs)
            if lista:
                out["relacionados"] = lista
                out["fuentes"]["relacionados"] = "secretario"
            continue
        if not isinstance(v, str):
            out["avisos"].append(f"«{k}» NO TRAE TEXTO (llegó {_que_es(v)}): se ignoró.")
            continue
        v = _limpio(v)
        if len(v) > _tope_de(ruta):
            out["avisos"].append(
                f"«{k}» ES DEMASIADO LARGO ({len(v)} caracteres; el tope es {_tope_de(ruta)}): se "
                f"ignoró. Escríbelo sin texto de más.")
            continue
        if ruta == "relacionados":
            # «» = sin relacionados (el interruptor apagado): nada, sin aviso.
            lista, avs = relacionados_de(v, propio, tp)
            out["avisos"].extend(avs)
            if lista:
                out["relacionados"] = lista
                out["fuentes"]["relacionados"] = "secretario"
            continue
        if ruta == "ministerio_publico":
            s = _plano(v).replace(" ", "_")
            if s not in _MP_ALIAS:
                out["avisos"].append(f"«{v[:80]}» no es un valor del Ministerio Público (pedimento / "
                                     f"sin_pedimento / no consta): se ignoró.")
                continue
            if _MP_ALIAS[s]:
                _poner(out, ruta, _MP_ALIAS[s])
                out["fuentes"][ruta] = "secretario"
            continue
        if _es_ruta_fecha(ruta):
            if _vacio(v):
                continue
            i = a_iso(v)
            if not i:
                out["avisos"].append(f"«{v[:80]}» no es una fecha ({k}): se ignoró.")
                continue
            _poner(out, ruta, i)
            out["fuentes"][ruta] = "secretario"
            continue
        if ruta == "forma_notificacion":
            s = _plano(v)
            if s and s in _FORMAS_NOTIF:
                _poner(out, ruta, s)
                out["fuentes"][ruta] = "secretario"
            continue
        if ruta == "via_presentacion":
            if _vacio(v):
                continue
            s = via_de(v)
            if not s:
                out["avisos"].append(f"«{v[:80]}» no es una vía de presentación ({', '.join(VIAS)}): "
                                     f"se ignoró.")
                continue
            _poner(out, ruta, s)
            out["fuentes"][ruta] = "secretario"
            continue
        if ruta in ("fraccion_63", "cuantia", "fundamento_surtimiento"):
            if _vacio(v):
                continue
            if ruta == "fundamento_surtimiento" and motivo_fundamento(v, tp):
                out["avisos"].append(f"EL PRECEPTO DEL SURTIMIENTO «{v[:120]}» {motivo_fundamento(v, tp)} "
                                     f"(fundamento_surtimiento): se ignoró.")
                continue
            s = {"fraccion_63": fraccion_63_de, "cuantia": cuantia_de,
                 "fundamento_surtimiento": lambda x: fundamento_surtimiento_de(x, tp)}[ruta](v)
            if not s:
                out["avisos"].append(
                    {"fraccion_63": f"«{v[:80]}» no es una fracción del artículo 63 de la LFPCA (I a "
                                    f"X) (fraccion_63): se ignoró y la procedencia la pedirá.",
                     "cuantia": f"«{v[:80]}» no se lee como un monto en pesos (cuantia; p. ej. "
                                f"«$847,738.77» o «indeterminada»): se ignoró.",
                     "fundamento_surtimiento": f"«{v[:120]}» no dice un precepto (fundamento_"
                                               f"surtimiento): se ignoró."}[ruta])
                continue
            _poner(out, ruta, s)
            out["fuentes"][ruta] = "secretario"
            continue
        s = v
        if s:
            if ruta in ("acto.toca", "acto.expediente", "expediente_tfja"):
                # SÓLO LOS ESPACIOS JUNTO A «/» Y «-» (« 4520 / 2024» → «4520/2024»).
                # Quitarlos todos convertía «toca civil 374/2024» en
                # «tocacivil374/2024», y así salía en el V I S T O.
                s = re.sub(r"\s*([/\-])\s*", r"\1", s)
            if ruta == "fraccion_97":
                s = s.upper()
            if ruta == "inciso_97":
                s = s.lower().strip(")")
            if ruta in ("turno.ponente", "returno.ponente"):
                # EL CARGO, EN SU CAMPO (cuarta ronda): «Magistrada Jenica
                # Campos Juárez» → ponente «Jenica Campos Juárez» y titulo
                # «Magistrada», la misma forma que da la lectura del auto.
                # `a_formulario` lo vuelve a juntar: ida y vuelta sin pérdida.
                s, tit = partir_ponente(s)
                if tit:
                    _poner(out, ruta.replace("ponente", "titulo"), tit)
                    out["fuentes"][ruta.replace("ponente", "titulo")] = "secretario"
                if not s:
                    continue
            _poner(out, ruta, s)
            out["fuentes"][ruta] = "secretario"
    for k in listas:
        v = form.get(k)
        if v is None:
            continue
        if isinstance(v, str):
            v = [x.strip() for x in re.split(r"\s*(?:,|;|\by\b)\s*", v) if x.strip()]
        if not isinstance(v, list):
            out["avisos"].append(f"«{k}» NO ES UNA LISTA DE NOMBRES (llegó {_que_es(v)}): se ignoró.")
            continue
        malos = [x for x in v if not isinstance(x, str) or len(_limpio(x)) > TOPE_NOMBRE]
        if malos:
            out["avisos"].append(f"«{k}» TRAE {len(malos)} ELEMENTO(S) QUE NO SON UN NOMBRE (no son "
                                 f"texto o pasan de {TOPE_NOMBRE} caracteres): se ignoraron.")
        buenos = [_limpio(x) for x in v if isinstance(x, str) and 0 < len(_limpio(x)) <= TOPE_NOMBRE]
        if buenos:
            out[k] = buenos
            out["fuentes"][k] = "secretario"
    if tp:
        _por_tipo(out, tp)
        out["tipo"] = tp
    return out


def a_formulario(ficha: dict, tipo: str = "") -> dict:
    """Ficha (completa o parcial) → las claves planas del formulario, siempre
    las 22, en cadena («» si no consta). Los ponentes, con el cargo que
    conste (`con_titulo`): «Magistrada Jenica Campos Juárez». Los asuntos
    relacionados, en el texto del formulario (`relacionados_a_texto`, sexta
    ronda): «amparo_directo|452/2025|misma_sesion;…».

    POR TIPO, como `de_formulario` (`_POR_TIPO_FORMULARIO`): «organo_acto» es
    la responsable en AD y la Sala en RF; «expediente_origen», el expediente
    del TFJA en RF. El tipo sale de `tipo` o de `ficha["tipo"]`; sin él, el
    campo genérico (`acto.organo`, `acto.expediente` y, si falta, el del
    TFJA)."""
    f = ficha if isinstance(ficha, dict) else {}
    tp = _tipo(tipo or _limpio(f.get("tipo")))
    rutas_tipo = _POR_TIPO_FORMULARIO.get(tp, {})
    # LO QUE SÓLO ES LA OMISIÓN NO SE PROPONE COMO DATO (quinta ronda, F1): la
    # forma «personal» que nadie dijo saldría en la tarjeta como leída.
    fuentes = f.get("fuentes") if isinstance(f.get("fuentes"), dict) else {}
    out = {}
    for k in CLAVES_FORMULARIO:
        rutas = rutas_tipo.get(k) or (_MAPA_FORMULARIO[k],)
        if not tp and k == "expediente_origen":
            rutas = ("acto.expediente", "expediente_tfja")
        v = None
        for ruta in rutas:
            v = None if fuentes.get(ruta) in FUENTES_OMISION else _tomar(f, ruta)
            if not _vacio(v):
                break
        if k == "relacionados":
            # La lista de la ficha vuelve como el texto que la tarjeta sabe
            # reconstruir (sexta ronda); nunca su repr de Python.
            out[k] = relacionados_a_texto(v) if not _vacio(v) else ""
            continue
        out[k] = "" if _vacio(v) else str(v)
        # El cargo del ponente viaja con su nombre (cuarta ronda): el
        # formulario no tiene campo para él y, sin esto, el «Magistrada» que
        # leyó el auto se perdía al volver de la pantalla.
        if k in ("ponente_turno", "ponente_returno") and out[k]:
            sec = "turno" if k == "ponente_turno" else "returno"
            out[k] = con_titulo(out[k], _tomar(f, f"{sec}.titulo") or "")
    return out


# ═══════════════════════════════════════════════════════════════════════════
# ARMAR: LA PRECEDENCIA, LAS FUENTES Y LA VALIDACIÓN
# ═══════════════════════════════════════════════════════════════════════════
_OPCIONALES = {
    # «registro»: el auto que forma y registra cuando NO es el que admite
    # (cuarta ronda, 3-oct-2026; RF 7/2025 y Q 335/2025).
    "registro": ("fecha",),
    "returno": ("fecha", "ponente"),
    "adhesivo": ("quien", "presentacion", "admision", "notificacion"),
    "informe_101": ("fecha",),
    "demanda": ("fecha", "autoridades", "actos"),
    # «clase» (expresa | negativa_ficta) y «materia» (lo que se pidió, en la
    # negativa ficta): tercera ronda, rev_5.
    "resolucion_impugnada": ("clase", "oficio", "fecha", "autoridad", "materia"),
}
_RUTAS_FECHA = {"presentacion", "notificacion", "deposito_postal", "audiencia",
                "adhesivo.presentacion", "adhesivo.admision", "adhesivo.notificacion"}
_RUTAS = (
    "numero", "materia", "sede.tribunal", "sede.ciudad",
    "registro.fecha", "admision.fecha", "turno.fecha", "turno.ponente", "turno.titulo",
    "returno.fecha", "returno.ponente", "returno.titulo",
    "ministerio_publico", "adhesivo.quien", "adhesivo.presentacion", "adhesivo.admision",
    "adhesivo.notificacion", "informe_101.fecha",
    "promovente", "caracter", "representante", "figura_representante",
    "presentacion", "via_presentacion", "deposito_postal", "notificacion", "forma_notificacion",
    "acto.clase", "acto.fecha", "acto.organo", "acto.toca", "acto.expediente", "acto.incidente",
    "acto.sentido", "acto.resolvio", "acto.resolvio_mixto", "acto.sentido_clave", "acto.instancia",
    "acto.via", "responsable", "ejecutora", "terceros", "derechos",
    "demanda.fecha", "demanda.autoridades", "demanda.actos", "audiencia", "clase_recurrida",
    "fraccion_97", "inciso_97", "actora", "resolucion_impugnada.clase", "resolucion_impugnada.oficio",
    "resolucion_impugnada.fecha", "resolucion_impugnada.autoridad", "resolucion_impugnada.materia",
    "autoridad_demandada",
    "sala", "expediente_tfja", "quejoso",
    "fraccion_63", "cuantia", "fundamento_surtimiento",
)
_DISCREPANCIA = {"numero", "acto.fecha", "acto.toca", "acto.expediente", "presentacion",
                 "registro.fecha", "admision.fecha", "turno.fecha", "returno.fecha", "notificacion",
                 "deposito_postal", "audiencia", "demanda.fecha", "acto.organo"}
# «acto.organo» (tercera ronda, rev_6): al retomar, lo leído del papel volvía
# como dato del secretario y el juzgado CAMBIABA SIN AVISO al subir el acto
# correcto («Juzgado Primero…» contra «Juzgado Segundo…»), porque esta lista
# no lo tenía. Se compara como órgano (`_mismo_organo`: «Juez» y «Juzgado»,
# versales y nombre más o menos completo son el mismo); en el AD lo cubre ya
# el cotejo de la responsable.


def _es_ruta_fecha(ruta: str) -> bool:
    return ruta.endswith(".fecha") or ruta in _RUTAS_FECHA


def _vacio(v) -> bool:
    if v is None:
        return True
    if isinstance(v, bool):
        return False
    if isinstance(v, (list, tuple, dict, set)):
        return not v
    return not str(v).strip()


def _hojas(d: dict, pre: str = ""):
    for k, v in (d or {}).items():
        r = f"{pre}.{k}" if pre else k
        if isinstance(v, dict) and k not in ("fuentes",):
            yield from _hojas(v, r)
        else:
            yield r, v


def _vacia(tipo: str) -> dict:
    return {"formato": FORMATO, "tipo": tipo, "numero": "", "materia": "",
            "sede": {"tribunal": "", "ciudad": "", "circuito": "", "cdmx": False},
            "registro": {}, "admision": {"fecha": ""}, "turno": {"fecha": "", "ponente": ""}, "returno": {},
            "ministerio_publico": "", "adhesivo": {}, "informe_101": {},
            "promovente": "", "caracter": "", "representante": "", "figura_representante": "",
            "presentacion": "", "via_presentacion": "", "deposito_postal": "",
            "notificacion": "", "forma_notificacion": "",
            "acto": {"clase": "", "fecha": "", "organo": "", "toca": "", "expediente": "",
                     "incidente": False, "sentido": "", "resolvio": ""},
            "responsable": "", "ejecutora": "", "terceros": [], "derechos": [],
            "quejoso": "",
            "demanda": {}, "audiencia": "", "clase_recurrida": "",
            "fraccion_97": "", "inciso_97": "",
            "actora": "", "resolucion_impugnada": {}, "autoridad_demandada": "",
            "sala": "", "expediente_tfja": "",
            "fraccion_63": "", "cuantia": "", "fundamento_surtimiento": "",
            # Los asuntos que el secretario marcó como relacionados (sexta
            # ronda): [{"tipo", "numero", "estado"}], fuente siempre «secretario».
            "relacionados": [],
            "avisos": [], "fuentes": {}, "fechas_imposibles": []}


def _attr(o, k, defecto=None):
    if isinstance(o, dict):
        return o.get(k, defecto)
    return getattr(o, k, defecto)


_MATERIAS = ("civil", "administrativa", "penal", "laboral", "agraria", "familiar", "mercantil")


def _materia(encargo) -> str:
    m = _plano(_attr(encargo, "materia", ""))
    m = {"administrativo": "administrativa", "agrario": "agraria", "trabajo": "laboral"}.get(m, m)
    if m in _MATERIAS:
        return m
    try:
        import fase_precedente as _fp
        m = _fp.materia_de(_attr(encargo, "encabezado", "") or "", _attr(encargo, "tribunal", "") or "",
                           _attr(encargo, "tipo_asunto", "") or "")
    except Exception:
        m = ""
    return m if m in _MATERIAS else ""


def _forma_de_la_regla(e) -> str:
    """La forma de notificación que dice la regla de surtimiento del encargo.
    Las reglas de `fase0_oportunidad.REGLAS_SURTE` («personal», «lista»,
    «oficio», «electronica», «tja_qro_boletin», «lfpca_boletin»…) la dicen;
    «otra» y «lfpca» no («» entonces)."""
    regla = _plano(_attr(e, "regla_surtimiento", "")).replace(" ", "_")
    # LAS DEL CÓDIGO NACIONAL Y LA DEL FEDERAL EN LO AGRARIO (C7, 3-oct-2026)
    # dicen la misma forma que su gemela: «cnpcf_personal» es la «personal» que
    # el guardián de materia pone en la Ciudad de México cuando llega por
    # omisión, y «cfpc_personal» la que el desplegable propone fuera de ella;
    # las dos van a la capa «omision», como la personal (F1).
    for _pref in ("cnpcf_", "cfpc_"):
        if regla.startswith(_pref):
            regla = regla[len(_pref):]
    return ("boletin" if "boletin" in regla else
            {"personal": "personal", "lista": "lista", "oficio": "oficio",
             "electronica": "electronica"}.get(regla.split("_")[0] if regla else "", ""))


# ── LA FORMA DE NOTIFICACIÓN QUE NADIE DIJO (QUINTA RONDA, 3-oct-2026, F1) ──
# Q 335/2025, Q 229, Q 261 y Q 342/2025, AD 274 y AD 335/2025 del banco: la
# ficha no traía la forma y el considerando escribía «se notificó… de manera
# personal y surtió efectos al día hábil siguiente», sin aviso. En producción
# es peor: `main` recibe `regla_surtimiento: str = Form("personal")` y `armar`
# metía ese «personal» como forma DECLARADA por el secretario, por delante del
# escrito y del acto; la verja (m) no podía distinguir lo declarado de lo
# supuesto. Ahora:
#   · «personal» de la regla va en la capa «omision», la última: si el
#     escrito, el auto, el acto o el formulario «Trámite» dicen la forma, manda
#     lo que dicen;
#   · en la revisión y en la queja, el «oficio» con que se computa a la
#     autoridad que recurre (D2: `redactor_adelanto._recomputar_para_la_
#     autoridad` cambia la regla ANTES de armar) tampoco es un dato: si ningún
#     papel dice cómo se le notificó, su fuente es «omision_autoridad» y el
#     considerando escribe «surtió efectos el mismo día» (art. 31, fr. I) sin
#     «por oficio».
# Quien quiera que conste, la escribe en «Trámite en este tribunal»
# (`forma_notificacion`), que es dato del secretario.
_RECURSOS_DE_AMPARO = ("amparo_revision", "queja")


def _omision_encargo(e) -> dict:
    """La forma de notificación POR OMISIÓN del formulario principal: {} si la
    regla elegida es otra que «personal» (ésa sí la eligió alguien)."""
    return {"forma_notificacion": "personal"} if _forma_de_la_regla(e) == "personal" else {}


def forma_por_omision(ficha) -> str:
    """«omision» u «omision_autoridad» si la forma de notificación de la ficha
    sólo sale del valor por omisión (QUINTA RONDA, F1); «» si la dijo un papel
    o el secretario, o si no hay forma. Con «omision…» el considerando NO
    afirma la forma: «se notificó a la parte X el {fecha} y surtió efectos…»."""
    f = ficha if isinstance(ficha, dict) else {}
    fu = f.get("fuentes") if isinstance(f.get("fuentes"), dict) else {}
    x = str(fu.get("forma_notificacion") or "")
    return x if (x in FUENTES_OMISION and _limpio(f.get("forma_notificacion"))) else ""


def _aviso_forma_por_omision(tipo: str) -> str:
    """El aviso de la forma que no consta (F1), con la regla con que se contó."""
    regla = ("artículo 31, fracción II, de la Ley de Amparo" if tipo in _RECURSOS_DE_AMPARO else
             "la regla general; la ley que rige el acto reclamado puede decir otra cosa"
             if tipo == "amparo_directo" else
             "la regla general; la ley que rige la notificación de la sentencia de la Sala puede decir "
             "otra cosa")
    que = {"amparo_directo": "la sentencia reclamada", "amparo_revision": "la resolución recurrida",
           "queja": "el auto recurrido", "revision_fiscal": "la sentencia recurrida"}.get(tipo, "lo recurrido")
    return (f"LA FORMA DE NOTIFICACIÓN NO CONSTA (forma_notificacion): ningún papel ni el formulario "
            f"«Trámite en este tribunal» dicen cómo se notificó {que}; se contó como personal, que surte "
            f"efectos al día hábil siguiente ({regla}). Compruébala en la constancia de notificación; si fue "
            f"otra (por lista, electrónica, por oficio), escríbela en «Trámite en este tribunal» y vuelve a "
            f"generar.")


def _capa_encargo(e, tipo: str) -> dict:
    """Lo que el secretario tecleó en el formulario principal."""
    c: dict = {}
    for k, ruta in (("numero", "numero"), ("tribunal", "sede.tribunal"), ("ciudad", "sede.ciudad")):
        v = _limpio(_attr(e, k, ""))
        if v:
            _poner(c, ruta, re.sub(r"\s+", "", v) if k == "numero" else v)
    m = _materia(e)
    if m:
        c["materia"] = m
    for k in ("presentacion", "notificacion"):
        v = a_iso(_attr(e, k, ""))
        if v:
            c[k] = v
    # LA REGLA ELEGIDA DICE LA FORMA… SALVO LA DE OMISIÓN (QUINTA RONDA, F1).
    # «personal» es lo que el formulario manda cuando nadie eligió nada (main:
    # `regla_surtimiento: str = Form("personal")`) y no un dato del
    # secretario: va en `_omision_encargo`, detrás de todo papel.
    forma = _forma_de_la_regla(e)
    if forma and forma != "personal":
        c["forma_notificacion"] = forma
    q = _limpio(_attr(e, "quejoso", ""))
    rec = _limpio(_attr(e, "recurrente", ""))
    if tipo == "amparo_directo":
        if q:
            c["promovente"] = q
        c["caracter"] = "quejoso"
    elif rec:
        c["promovente"] = rec
    return c


def _respaldo_encargo(e, tipo: str) -> dict:
    """EL «RECURRENTE» VACÍO NO ES UN DATO (tercera ronda, 3-oct-2026; AR
    631/2025, prueba de punta a punta). En un recurso, el formulario con el
    recurrente vacío se leía como «recurre la propia quejosa» y esa suposición
    entraba como dato del SECRETARIO, por delante de todo: en el 631 la
    revisión la interpuso la tercera interesada (Impulsora de Desarrollos
    Inmobiliarios V V, S.A. de C.V., por conducto de su apoderado) y la carátula
    salió «QUEJOSA Y RECURRENTE: UNIÓN DE TRABAJADORES…». La quejosa queda como
    RESPALDO, detrás del auto, del escrito y de la ficha procesal, con la
    fuente «formulario»: vale sólo si ningún papel dice quién recurre."""
    if tipo == "amparo_directo":
        return {}
    q = _limpio(_attr(e, "quejoso", ""))
    rec = _limpio(_attr(e, "recurrente", ""))
    return {"promovente": q} if (q and not rec) else {}


def _capa_ficha_procesal(fp: dict, tipo: str) -> dict:
    c: dict = {}
    if not isinstance(fp, dict) or not fp:
        return c
    rec = fp.get("recurrente") or {}
    if tipo != "amparo_directo" and _limpio(rec.get("nombre")):
        c["promovente"] = _limpio(rec["nombre"])
    papel = _plano(rec.get("papel"))
    if papel in ("quejoso", "tercero", "autoridad"):
        c["caracter"] = papel
    q = _limpio((fp.get("quejosa") or {}).get("nombre"))
    if q:
        c["quejoso"] = q
        if tipo == "amparo_directo":
            c["promovente"] = q
    ter = [_limpio(t.get("nombre")) for t in fp.get("terceros") or [] if isinstance(t, dict)]
    ter = [t for t in ter if t]
    if ter:
        c["terceros"] = ter
    resp = [r for r in fp.get("responsables") or [] if isinstance(r, dict) and _limpio(r.get("autoridad"))]
    if tipo == "amparo_directo" and resp:
        c["responsable"] = _limpio(resp[0]["autoridad"])
    if tipo == "amparo_revision":
        aut = [_limpio(r["autoridad"]) for r in resp]
        if aut:
            _poner(c, "demanda.autoridades", aut)
        que = _plano((fp.get("resolvio") or {}).get("que"))
        if que in ("concede", "niega", "sobresee"):
            _poner(c, "acto.resolvio", que)
        elif que in ("sobresee_concede", "sobresee_niega"):
            _poner(c, "acto.resolvio", "mixto")
            _poner(c, "acto.resolvio_mixto", que)
    org = _limpio((fp.get("organo_recurrido") or {}).get("nombre"))
    if org and tipo in ("amparo_revision", "queja"):
        _poner(c, "acto.organo", org)
    return c


def _capa_partes(partes, tipo: str) -> dict:
    c: dict = {}
    if partes is None:
        return c
    t = _limpio(_attr(partes, "tercero_interesado", ""))
    if t:
        c["terceros"] = [x for x in re.split(r"\s+y\s+(?=[A-ZÁÉÍÓÚÑ])", t) if x.strip()]
    a = _limpio(_attr(partes, "autoridad_responsable", ""))
    if a and tipo == "amparo_directo":
        c["responsable"] = a
    q = _limpio(_attr(partes, "quejoso", ""))
    if q:
        c["quejoso"] = q
    return c


def _elegir_responsable(candidatos: list, avisos: list) -> tuple:
    """(nombre, fuente) de la ordenadora: la primera con forma de órgano, por
    precedencia; si le falta el estado y otra fuente trae el mismo órgano
    completo, la completa."""
    validos = []
    for nombre, fuente in candidatos:
        n = _limpio(nombre)
        if not n:
            continue
        ok_, mot = es_organo(n)
        if not ok_:
            avisos.append(f"SE DESCARTÓ «{n[:90]}» COMO AUTORIDAD RESPONSABLE ({fuente}): {mot}. "
                          f"Se tomó la siguiente fuente.")
            continue
        validos.append((n, fuente, mot))
    if not validos:
        return "", ""
    n, fuente, mot = validos[0]
    if mot == "sin_entidad":
        for n2, f2, m2 in validos[1:]:
            if m2 != "sin_entidad" and _misma_base(n, n2) and len(n2) > len(n):
                avisos.append(f"LA AUTORIDAD RESPONSABLE SE COMPLETÓ: «{n}» ({fuente}) no dice de qué "
                              f"estado; se tomó «{n2}» ({f2}).")
                return n2, f2
        avisos.append(f"LA AUTORIDAD RESPONSABLE «{n}» NO DICE DE QUÉ ESTADO: escríbela completa.")
        return n, fuente
    # El mismo órgano más completo en una fuente posterior (la cabecera dice
    # «Segunda Sala Civil» y la demanda el nombre entero).
    for n2, f2, m2 in validos[1:]:
        if m2 == "" and _misma_base(n, n2) and len(n2) > len(n) + 8 and _plano(n) in _plano(n2):
            return n2, f2
    return n, fuente


# ── EL ÓRGANO DE AMPARO Y LA PARTE, COMPARADOS (tercera ronda, 3-oct-2026) ──
# Quien dicta una sentencia recurrible en amparo en revisión: un Juzgado de
# Distrito, un Tribunal Colegiado de Apelación (antes Unitario de Circuito) o
# el Centro de Justicia Penal Federal. «Juzgado Primero Civil del Distrito
# Judicial de Querétaro» NO lo es (es un juzgado del Estado).
_RX_ORGANO_DE_AMPARO = re.compile(
    r"\b(?:juzgado|juez|jueza)\s+(?:[a-z]+\s+){0,4}?de\s+distrito\b(?!\s+judicial)|"
    r"\btribunal\s+colegiado\s+de\s+apelacion\b|"
    r"\btribunal\s+unitario\s+(?:[a-z]+\s+){0,3}?de(?:l)?\s+(?:[a-z]+\s+){0,3}?circuito\b|"
    r"\bcentro\s+de\s+justicia\s+penal\s+federal\b")


def es_organo_de_amparo(nombre: str) -> bool:
    """¿Tiene forma de órgano que conoce del amparo indirecto?"""
    return bool(_RX_ORGANO_DE_AMPARO.search(_plano(nombre)))


def _mismo_organo(a: str, b: str) -> bool:
    """Dos nombres del MISMO órgano: uno contiene al otro, o el mismo núcleo
    («Juez Séptimo de Distrito» y «Juzgado Séptimo de Distrito…»)."""
    a, b = _limpio(a), _limpio(b)
    if not a or not b:
        return False
    na, nb = _nucleo(a), _nucleo(b)
    return _misma_base(a, b) or bool(na and na == nb and len(na.split()) >= 3)


def _misma_parte(a: str, b: str) -> bool:
    """`promovente.misma_parte` (puntuación, forma social y fórmula de
    representación aparte); sin esa pieza, igualdad sin tildes ni versales."""
    if not _limpio(a) or not _limpio(b):
        return False
    try:
        import promovente as _pv
        return bool(_pv.misma_parte(a, b))
    except Exception:
        return _plano(a) == _plano(b)


# «GOBERNADOR DEL ESTADO DE QUERÉTARO (AUTORIDAD RESPONSABLE)» (AR 208 y 72):
# la etiqueta de rol de la carátula viajaba dentro del nombre y salía en la
# prosa («…(autoridad Responsable), en su carácter de autoridad responsable»).
_RX_ETIQUETA_ROL = re.compile(
    r"\s*\(\s*(?P<r>autoridad(?:es)?\s+responsables?|quejos[oa]s?|tercer[oa]s?\s+interesad[oa]s?|"
    r"recurrentes?|parte\s+(?:quejosa|recurrente|tercera\s+interesada))\s*\)\s*$", re.I)


def _sin_etiqueta_rol(nombre: str) -> tuple:
    """(nombre sin la etiqueta final de rol, carácter que decía la etiqueta)."""
    n = _limpio(nombre)
    m = _RX_ETIQUETA_ROL.search(n)
    if not m:
        return n, ""
    r = _plano(m.group("r"))
    car = ("autoridad" if r.startswith("autoridad") else "tercero" if "tercer" in r else
           "quejoso" if "quejos" in r else "")
    return n[:m.start()].strip(" ,"), car


def armar(encargo, *, auto=None, acto=None, ficha_procesal=None, partes=None,
          declarado: dict = None, escrito=None) -> dict:
    """La ficha de trámite (§1), por precedencia: secretario > auto > escrito >
    acto > ficha_procesal. `encargo` es el `redactor_adelanto.Encargo` (o un
    dict con sus claves): lo que tecleó en el formulario principal cuenta como
    «secretario», salvo la responsable, que va detrás de lo leído en el papel
    (el extractor que la prellenaba la nombró mal en 64 de 72 proyectos), y la
    suposición «recurrente vacío = recurre la quejosa», que va al final
    (`_respaldo_encargo`).
    `auto`: lo de `leer_auto` (o una lista de lecturas). `acto`: lo de
    `leer_acto`. `escrito`: lo de `leer_escrito` (quién recurre y con qué
    carácter; si no se pasa, se toma `encargo.tramite_escrito` si existe).
    `declarado`: el `tramite_json` del formulario —su JSON (se lee como por
    HTTP: sólo las 22 claves del contrato), un dict plano (uso interno: admite
    además las claves extra) o una ficha parcial ya hecha por `de_formulario`.

    CUARTA RONDA (3-oct-2026): la ficha sale además con `avisos_claves`
    ({aviso: [rutas del dato]}) para que el compositor, la verja y el cableado
    no repitan el mismo hecho con otro texto (E11), sin nombres cortados ni
    cargos como representante (`normalizar`, E8) y con el cargo del ponente
    sólo si es de ESE ponente.

    QUINTA RONDA (3-oct-2026): la forma de notificación que sólo sale de la
    regla por omisión («personal») o del oficio de D2 lleva la fuente «omision»
    u «omision_autoridad» (F1, `forma_por_omision`), detrás de todo papel; el
    sentido, con inicial minúscula; en la queja de la fracción II, el registro
    leído con la fecha del informe es el mismo auto.

    SEXTA RONDA (3-oct-2026): `relacionados` sale SÓLO de `declarado` (C6),
    normalizado y sin el propio asunto; en la queja, la demanda de amparo
    (C3) se toma primero del auto recurrido y después del escrito, y el juzgado
    que dictó el auto sale de sus autoridades."""
    tipo = _tipo(_attr(encargo, "tipo_asunto", "")) or "amparo_directo"
    f = _vacia(tipo)
    avisos: list = []
    claves_av: dict = {}

    # ¿FORMULARIO PLANO O FICHA PARCIAL YA LEÍDA? La ficha parcial de
    # `de_formulario` trae «fuentes» (un dict); el JSON del front, nunca. Antes
    # se decidía además por «no trae ninguna clave del formulario», y eso falla
    # con las claves que se llaman IGUAL en las dos formas (`deposito_postal`,
    # `forma_notificacion` y, desde la segunda ronda, `fraccion_63`, `cuantia`,
    # `fundamento_surtimiento`): la ficha parcial de una revisión fiscal se
    # releía como plana y perdía la admisión, el turno y la Sala (3-oct-2026).
    if isinstance(declarado, str) and declarado:
        dec = de_formulario(declarado, tipo)
    elif isinstance(declarado, dict) and declarado:
        dec = declarado if isinstance(declarado.get("fuentes"), dict) \
            else de_formulario(declarado, tipo, extra=True)
    else:
        dec = {}
    # LO DEL FORMULARIO VA AL CAMPO DE ESTE TIPO. `redactor_adelanto` lee el
    # `tramite_json` con `de_formulario` ANTES de saber el tipo (para contestar
    # 422 antes del OCR): aquí, que ya se sabe, la Sala y el expediente del TFJA
    # (RF) o la responsable (AD) que confirmó el secretario llegan a su campo y
    # mandan sobre lo leído. Sobre una copia: el dict del llamador no se toca.
    if dec:
        dec = _por_tipo(copy.deepcopy(dec), tipo)
        # UNA FICHA YA ARMADA QUE VUELVE COMO «DECLARADA» NO CONVIERTE LA
        # OMISIÓN EN DATO (quinta ronda, F1): lo que traía fuente «omision…» se
        # vuelve a decidir aquí, como la primera vez.
        for ruta, fu in list((dec.get("fuentes") or {}).items()):
            if fu in FUENTES_OMISION:
                _poner(dec, ruta, "")
                dec["fuentes"].pop(ruta, None)
    avisos.extend(dec.get("avisos") or [])
    if isinstance(auto, (list, tuple)):
        aut = fundir_lecturas(*auto)
    else:
        aut = auto if isinstance(auto, dict) else {}
    avisos.extend(a for a in aut.get("avisos") or [] if a not in avisos)
    if escrito is None:
        escrito = _attr(encargo, "tramite_escrito", None)
    esc = escrito if (isinstance(escrito, dict) and tipo != "amparo_directo") else {}
    avisos.extend(a for a in esc.get("avisos") or [] if a not in avisos)
    act = acto if isinstance(acto, dict) else {}
    avisos.extend(a for a in act.get("avisos") or [] if a not in avisos)
    enc = _capa_encargo(encargo, tipo)
    fp = _capa_ficha_procesal(ficha_procesal, tipo)
    pa = _capa_partes(partes, tipo)
    resp = _respaldo_encargo(encargo, tipo)
    # F1 (quinta ronda): el «oficio» de la regla en la revisión o en la queja
    # puede ser el de la autoridad que recurre (D2), que nadie declaró. Se
    # decide cuando ya se sabe quién recurre (más abajo, tras la etiqueta de rol).
    oficio_de_la_regla = tipo in _RECURSOS_DE_AMPARO and enc.get("forma_notificacion") == "oficio"
    if oficio_de_la_regla:
        enc.pop("forma_notificacion")

    # El formulario «Trámite» y el principal son los dos del secretario; el
    # principal sólo trae lo básico (número, sede, fechas del cómputo). La
    # forma de notificación por omisión, la última (F1).
    capas = [("secretario", dec), ("secretario", enc), ("auto", aut), ("escrito", esc),
             ("acto", act), ("ficha_procesal", fp), ("ficha_procesal", pa), ("formulario", resp),
             ("omision", _omision_encargo(encargo))]
    # LA DEMANDA DE AMPARO, PRIMERO COMO LA DESCRIBE EL JUZGADO (sexta ronda,
    # 3-oct-2026, C3). En la queja la leen el auto recurrido (`leer_acto`) y el
    # escrito del recurso (`leer_escrito`): el auto es la descripción del
    # juzgado; el escrito, la de quien recurre. Para la demanda y la quejosa,
    # el acto va delante del escrito (si no cuadran, se dice).
    capas_demanda = [c for c in capas if c[0] != "escrito"]
    capas_demanda.insert(next(i for i, c in enumerate(capas_demanda) if c[0] == "acto") + 1, ("escrito", esc))
    _ignorar = {"responsable"}           # se elige aparte, con su forma de órgano
    _DEL_RECURRENTE = ("caracter", "representante", "figura_representante")
    for ruta in _RUTAS:
        if ruta in _ignorar:
            continue
        elegido, fuente = None, ""
        for nombre, capa in (capas_demanda if ruta.startswith("demanda.") or ruta == "quejoso" else capas):
            v = _tomar(capa, ruta)
            if _vacio(v):
                continue
            if _es_ruta_fecha(ruta):
                v = a_iso(v)
                if not v:
                    continue
            # EL CARÁCTER Y EL REPRESENTANTE SON DE SU PARTE (tercera ronda).
            # Una fuente que nombra a OTRA parte como recurrente no da el
            # carácter ni el representante de la elegida: el «tercero» del
            # escrito no se le pega a la quejosa que tecleó el secretario.
            if ruta in _DEL_RECURRENTE and f["promovente"] and _limpio(_tomar(capa, "promovente")) \
                    and not _misma_parte(_tomar(capa, "promovente"), f["promovente"]):
                continue
            if elegido is None:
                elegido, fuente = v, nombre
            elif ruta in _DISCREPANCIA and v != elegido and not (
                    ruta == "acto.organo" and (tipo == "amparo_directo" or _mismo_organo(v, elegido))):
                txt_a = _dmy(elegido) if _es_ruta_fecha(ruta) else elegido
                txt_b = _dmy(v) if _es_ruta_fecha(ruta) else v
                av = (f"{_rotulo_ruta(ruta).upper()} NO CUADRA ENTRE FUENTES: {fuente} dice "
                      f"«{txt_a}» y {nombre} «{txt_b}». Vale la de {fuente}; compruébala.")
                if av not in avisos:
                    avisos.append(av)
                    claves_av[av] = [ruta]
            elif tipo != "amparo_directo" and ruta in ("promovente", "caracter") \
                    and nombre in ("secretario", "auto", "escrito") \
                    and fuente in ("secretario", "auto", "escrito") and v != elegido \
                    and not (ruta == "promovente" and _misma_parte(v, elegido)):
                # QUIÉN RECURRE, SEGÚN DOS PAPELES (tercera ronda, AR 631): si el
                # auto o el escrito dicen otra parte u otro carácter, se dice.
                que = "QUIEN RECURRE" if ruta == "promovente" else "EL CARÁCTER DE QUIEN RECURRE"
                av = (f"{que} NO CUADRA ENTRE FUENTES: {fuente} dice «{elegido}» y {nombre} «{v}». "
                      f"Vale la de {fuente}; compruébalo en el escrito del recurso y en el auto "
                      f"que lo admitió.")
                if av not in avisos:
                    avisos.append(av)
                    claves_av[av] = [ruta]
        if elegido is None:
            continue
        _poner(f, ruta, elegido)
        f["fuentes"][ruta] = fuente

    # ── LOS ASUNTOS RELACIONADOS: SÓLO LOS QUE MARCÓ EL SECRETARIO (sexta ronda) ──
    # David, 3-oct-2026: «siempre y cuando haya asuntos relacionados. No vamos
    # a meter conexidad en automático». Salen SÓLO de lo declarado en «Trámite
    # en este tribunal» (nunca del auto, del acto ni del escrito) y se vuelven
    # a normalizar aquí, que ya se sabe el número y el tipo del asunto: el
    # propio asunto no es relacionado suyo (por HTTP `de_formulario` no tenía
    # el número).
    for av in dec.get("avisos") or []:
        if "(relacionados)" in av:
            claves_av.setdefault(av, ["relacionados"])
    _rel = dec.get("relacionados")
    if not _vacio(_rel):
        _lista_rel, _av_rel = relacionados_de(_rel, f["numero"], tipo)
        for av in _av_rel:
            if av not in avisos:
                avisos.append(av)
                claves_av[av] = ["relacionados"]
        if _lista_rel:
            f["relacionados"] = _lista_rel
            f["fuentes"]["relacionados"] = "secretario"

    # ── EL CARGO DEL PONENTE ES DE SU PONENTE (cuarta ronda, 3-oct-2026) ──
    # El «Magistrada» que leyó el auto se queda sólo si el auto habla de la
    # MISMA persona que quedó como ponente: si el secretario tecleó a otra,
    # ese cargo no es el suyo (y el rótulo va por omisión, con su aviso).
    for sec in ("turno", "returno"):
        r_t, r_p = f"{sec}.titulo", f"{sec}.ponente"
        tit_f = f["fuentes"].get(r_t)
        if not tit_f:
            continue
        pon = _tomar(f, r_p)
        suyo = next((_tomar(c, r_p) for n_, c in capas
                     if n_ == tit_f and not _vacio(_tomar(c, r_t))), "")
        if not pon or (tit_f != f["fuentes"].get(r_p) and not (suyo and _misma_persona(suyo, pon))):
            f[sec].pop("titulo", None)
            f["fuentes"].pop(r_t, None)

    # EL REGISTRO CON LA FECHA DE LA ADMISIÓN ES EL MISMO AUTO (cuarta ronda):
    # dos lecturas pueden traerlo así; lo que tecleó el secretario se respeta.
    # En la queja de la fracción II, también con la fecha del auto que tuvo
    # por rendido el informe (quinta ronda): el que registra lo REQUIERE, es
    # otro auto y es anterior. Lo del secretario se queda y `validar` lo dice.
    _reg_f = _tomar(f, "registro.fecha")
    _fr2 = tipo == "queja" and (_limpio(f["fraccion_97"]).upper() == "II"
                                or _plano(_tomar(f, "acto.via")) == "directo")
    if _reg_f and (_reg_f == _tomar(f, "admision.fecha") or (_fr2 and _reg_f == _tomar(f, "informe_101.fecha"))) \
            and f["fuentes"].get("registro.fecha") != "secretario":
        f["registro"] = {}
        f["fuentes"].pop("registro.fecha", None)

    # ── LA RESPONSABLE (AD: la ORDENADORA) ──
    if tipo == "amparo_directo":
        cands = [(_tomar(dec, "responsable"), "secretario"), (_tomar(dec, "acto.organo"), "secretario"),
                 (_tomar(aut, "responsable"), "auto"), (_tomar(aut, "acto.organo"), "auto"),
                 (_tomar(act, "responsable"), "acto"), (_tomar(act, "acto.organo"), "acto"),
                 (_attr(encargo, "responsable", ""), "formulario"),
                 (_tomar(fp, "responsable"), "ficha_procesal"), (_tomar(pa, "responsable"), "ficha_procesal")]
        n, fuente = _elegir_responsable(cands, avisos)
        if n:
            f["responsable"] = n
            f["fuentes"]["responsable"] = fuente
            # LA DEL SECRETARIO MANDA, PERO SE COTEJA CON EL PAPEL. El campo de la
            # carátula que el front manda como «organo_acto» lo prellena el
            # extractor que nombró mal la responsable en 64 de 72 proyectos; si el
            # auto o el acto nombran OTRO órgano, se dice (no se corrige).
            if fuente == "secretario":
                for c, fc in cands[2:6]:
                    c = _limpio(c)
                    if c and es_organo(c)[0] and not _misma_base(c, n) and _nucleo(c) != _nucleo(n):
                        avisos.append(f"LA AUTORIDAD RESPONSABLE NO CUADRA ENTRE FUENTES: el secretario "
                                      f"dice «{n[:90]}» y el {fc} «{c[:90]}». Vale la del secretario; "
                                      f"compruébala en el proemio de la sentencia reclamada.")
                        break
            # La ordenadora dictó la sentencia reclamada: un solo nombre en todo.
            if not f["acto"]["organo"] or (_misma_base(f["acto"]["organo"], n)
                                           and f["fuentes"].get("acto.organo") != "secretario"):
                f["acto"]["organo"] = n
                f["fuentes"]["acto.organo"] = fuente
        elif not f["acto"]["organo"]:
            avisos.append("NO CONSTA LA AUTORIDAD RESPONSABLE ORDENADORA: escríbela en el formulario "
                          "(está en el capítulo «Autoridad responsable» de la demanda).")

    # ── LA ETIQUETA DE ROL FUERA DEL NOMBRE (tercera ronda, rev_3) ──
    # Si el carácter no constaba, la etiqueta lo dice (es el papel el que lo
    # escribe, no una suposición).
    for ruta in ("promovente", "quejoso"):
        n_, car = _sin_etiqueta_rol(f.get(ruta) or "")
        if n_ != (f.get(ruta) or ""):
            f[ruta] = n_
            if ruta == "promovente" and car and not f["caracter"]:
                f["caracter"] = car
                f["fuentes"]["caracter"] = f["fuentes"].get("promovente", "")
    if f["terceros"]:
        f["terceros"] = [_sin_etiqueta_rol(t)[0] for t in f["terceros"] if _sin_etiqueta_rol(t)[0]]

    # ── EL OFICIO DE LA REGLA, YA SABIDO QUIÉN RECURRE (quinta ronda, F1) ──
    # Si recurre una autoridad (o no se sabe quién), ese «oficio» es el de D2:
    # vale sólo si ningún papel dijo otra forma, y con la fuente
    # «omision_autoridad» (el considerando no escribe «por oficio»). Si recurre
    # un particular, D2 no lo puso: lo eligió el secretario y manda sobre lo
    # leído, como cualquier otro dato suyo (salvo lo de «Trámite»).
    if oficio_de_la_regla:
        fu_forma = f["fuentes"].get("forma_notificacion", "")
        if _plano(f["caracter"]).startswith(("quejos", "tercer")):
            if fu_forma != "secretario":
                f["forma_notificacion"], f["fuentes"]["forma_notificacion"] = "oficio", "secretario"
        elif not f["forma_notificacion"] or fu_forma in FUENTES_OMISION:
            f["forma_notificacion"], f["fuentes"]["forma_notificacion"] = "oficio", "omision_autoridad"
    if f["fuentes"].get("forma_notificacion") == "omision":
        av = _aviso_forma_por_omision(tipo)
        if av not in avisos:
            avisos.append(av)
            claves_av[av] = ["forma_notificacion"]

    # ── EL SENTIDO A MEDIA FRASE, CON INICIAL MINÚSCULA (quinta ronda) ──
    # Q 335/2025: «…dictado por el Juzgado…, en el que Dejó sin efectos la
    # suspensión». El sentido va siempre tras «en el que» / «en la que».
    if f["acto"].get("sentido"):
        f["acto"]["sentido"] = sentido_en_minuscula(f["acto"]["sentido"])

    # ── LOS NOMBRES CORTADOS, FUERA (cuarta ronda, E8) ──
    for av, rutas in _quitar_truncados(f):
        if av not in avisos:
            avisos.append(av)
            claves_av[av] = rutas

    # ── EL REPRESENTANTE QUE ES LA MISMA PARTE (tercera ronda, rev_0/rev_3) ──
    # AR 307/2024: «por propio derecho y en representación de su hijo» se leyó
    # como un representante distinto de la recurrente, y el resultando escribía
    # «X, por conducto de su representante X». Si el representante es la misma
    # parte que promueve (o ya va dentro de su nombre), no es un representante.
    rep, prom = f["representante"], f["promovente"]
    if rep and prom:
        pr, pp_ = _plano(rep), _plano(prom)
        if pr == pp_ or _misma_parte(rep, prom) or (len(pr.split()) >= 2 and pr in pp_):
            av = (f"EL REPRESENTANTE «{_cita(rep)}» ES LA MISMA PARTE QUE PROMUEVE O RECURRE "
                  f"(«{_cita(prom)}»): se quitó, y no se escribe «por conducto de». Si actúa "
                  f"también por otra persona, escríbelo en el nombre de la parte.")
            avisos.append(av)
            claves_av[av] = ["representante", "promovente"]
            f["representante"] = f["figura_representante"] = ""
            f["fuentes"].pop("representante", None)
            f["fuentes"].pop("figura_representante", None)

    # ── EL REPRESENTANTE QUE ES UN CARGO, A LA FIGURA (cuarta ronda, E8) ──
    for av, rutas in _cargo_a_figura(f):
        if av not in avisos:
            avisos.append(av)
            claves_av[av] = rutas

    # ── LA VÍA, EN LA FORMA DEL ESQUEMA (tercera ronda, rev_1) ──
    if f["via_presentacion"]:
        via = via_de(f["via_presentacion"])
        if not via:
            avisos.append(f"LA VÍA DE PRESENTACIÓN «{f['via_presentacion']}» NO SE RECONOCE "
                          f"({', '.join(VIAS)}): se quitó.")
            f["fuentes"].pop("via_presentacion", None)
        f["via_presentacion"] = via

    # ── EL PRECEPTO DEL SURTIMIENTO, SÓLO SI LO FUNDA (tercera ronda, rev_1) ──
    # Llega del formulario sin tipo (`tramite_de_formulario` lo lee antes del
    # OCR): aquí, que se sabe que es amparo directo, se le aplica la regla.
    if f["fundamento_surtimiento"]:
        mot = motivo_fundamento(f["fundamento_surtimiento"], tipo)
        if mot:
            avisos.append(f"EL PRECEPTO DEL SURTIMIENTO «{f['fundamento_surtimiento'][:120]}» {mot}: "
                          f"se quitó (fundamento_surtimiento).")
            f["fundamento_surtimiento"] = ""
            f["fuentes"].pop("fundamento_surtimiento", None)

    # ── DERIVADOS POR TIPO ──
    a = f["acto"]
    if tipo == "revision_fiscal":
        if not f["sala"] and a["organo"]:
            f["sala"], f["fuentes"]["sala"] = a["organo"], f["fuentes"].get("acto.organo", "")
        if not f["expediente_tfja"] and a["expediente"]:
            f["expediente_tfja"] = a["expediente"]
            f["fuentes"]["expediente_tfja"] = f["fuentes"].get("acto.expediente", "")
        if not f["caracter"]:
            f["caracter"] = "autoridad"
    if tipo == "amparo_revision" and not f["clase_recurrida"]:
        f["clase_recurrida"] = ("interlocutoria_suspension" if a.get("incidente") else
                                "auto_sobreseimiento" if a.get("clase") == "auto" and a.get("resolvio") == "sobresee"
                                else "sentencia" if (a.get("clase") or a.get("fecha")) else "")
    if tipo == "queja":
        if not f["fraccion_97"] and a.get("via"):
            f["fraccion_97"] = "I" if a["via"] == "indirecto" else "II"
        if not f["inciso_97"] and a.get("sentido"):
            ck = a.get("sentido_clave", "")
            try:
                import tipos_asunto as _ta
                f["inciso_97"] = (CATALOGO_SENTIDO["queja"][ck][0] if ck in CATALOGO_SENTIDO["queja"]
                                  else _ta.inciso_97(a["sentido"]))
            except Exception:
                pass
            if f["inciso_97"]:
                f["fuentes"]["inciso_97"] = f["fuentes"].get("acto.sentido", "acto")
    # Las secciones opcionales: {} si no consta nada; si consta algo, todas sus claves.
    for sec, claves in _OPCIONALES.items():
        d = f.get(sec) or {}
        if any(not _vacio(d.get(k)) for k in claves):
            for k in claves:
                d.setdefault(k, [] if k in ("autoridades", "actos") else "")
            f[sec] = d
        else:
            f[sec] = {}

    # ── AR: EL JUZGADO NO ES UNA AUTORIDAD RESPONSABLE (tercera ronda, rev_3) ──
    # En el 307, el 448 y el 60 la lectura confundió la sentencia recurrida con
    # el acto reclamado: tomó la Sala Familiar, el Juez Octavo Familiar o la
    # Segunda Sala Civil —autoridades de la demanda— como si hubieran dictado la
    # sentencia de amparo, y el documento afirmó que una Sala local conoció de un
    # amparo indirecto. En el 222, al revés: el Juzgado Séptimo de Distrito
    # aparecía entre las responsables de su propio juicio. Si el órgano tiene
    # forma de órgano de amparo, sale de la lista de autoridades; si no, es el
    # acto reclamado y sale de `acto.organo` (hueco con aviso). Lo que tecleó
    # el secretario no se quita: se avisa (`validar`).
    if tipo == "amparo_revision" and a.get("organo") and (f["demanda"] or {}).get("autoridades"):
        org = a["organo"]
        auts = f["demanda"]["autoridades"]
        iguales = [x for x in auts if _mismo_organo(org, x)]
        if iguales and es_organo_de_amparo(org):
            f["demanda"]["autoridades"] = [x for x in auts if x not in iguales]
            av = (f"EL JUZGADO QUE DICTÓ LA RECURRIDA («{_cita(org)}») FIGURABA ENTRE LAS "
                  f"AUTORIDADES RESPONSABLES DE LA DEMANDA: se quitó de esa lista (un juzgado de "
                  f"amparo no es responsable en su propio juicio). Comprueba las autoridades en "
                  f"la demanda de amparo.")
            avisos.append(av)
            claves_av[av] = ["demanda.autoridades", "acto.organo"]
        elif iguales and f["fuentes"].get("acto.organo") != "secretario":
            fuente_o = f["fuentes"].pop("acto.organo", "")
            a["organo"] = ""
            av = (f"EL ÓRGANO QUE DICTÓ LA RESOLUCIÓN RECURRIDA NO PUEDE SER «{_cita(org)}»: es una "
                  f"de las autoridades responsables de la demanda, es decir, quien emitió el ACTO "
                  f"RECLAMADO, no la sentencia de amparo (lo dijo: {fuente_o or 'una lectura'}). "
                  f"Se dejó en hueco: escribe el juzgado de distrito en «Trámite en este tribunal».")
            avisos.append(av)
            claves_av[av] = ["acto.organo"]
    # ── QUEJA: EL PROPIO JUZGADO TAMPOCO (sexta ronda, 3-oct-2026, C3) ──
    # La demanda de amparo indirecto que describe el auto recurrido trae sus
    # autoridades; el juzgado que dictó el auto no es una de ellas. Sólo se
    # quita si tiene forma de órgano de amparo: en la queja de la fracción II
    # quien dictó el auto es la responsable del amparo directo, y ahí sí lo es.
    if tipo == "queja" and a.get("organo") and (f["demanda"] or {}).get("autoridades") \
            and es_organo_de_amparo(a["organo"]):
        org = a["organo"]
        auts = f["demanda"]["autoridades"]
        iguales = [x for x in auts if _mismo_organo(org, x)]
        if iguales:
            f["demanda"]["autoridades"] = [x for x in auts if x not in iguales]
            av = (f"EL JUZGADO QUE DICTÓ EL AUTO RECURRIDO («{_cita(org)}») FIGURABA ENTRE LAS "
                  f"AUTORIDADES RESPONSABLES DE LA DEMANDA: se quitó de esa lista (un juzgado de "
                  f"amparo no es responsable en su propio juicio). Comprueba las autoridades en "
                  f"la demanda de amparo.")
            avisos.append(av)
            claves_av[av] = ["demanda.autoridades", "acto.organo"]
    if isinstance(f.get("demanda"), dict) and f["demanda"] \
            and all(_vacio(f["demanda"].get(k)) for k in _OPCIONALES["demanda"]):
        f["demanda"] = {}
        for _r in [r for r in f["fuentes"] if r.startswith("demanda.")]:
            f["fuentes"].pop(_r, None)

    f["sede"] = sede_de(f["sede"].get("tribunal", ""), f["sede"].get("ciudad", ""))
    if isinstance(ficha_procesal, dict) and (ficha_procesal.get("adhesivo") or {}).get("consta") \
            and not f["adhesivo"]:
        avisos.append("LA FICHA PROCESAL DICE QUE HAY ADHESIVO, PERO NO CONSTA QUIÉN NI CUÁNDO: escribe "
                      "en el formulario quién lo promovió y la fecha del auto que lo admitió.")
    f["avisos"] = avisos
    for x in validar_detalle(f):
        if x["texto"] not in f["avisos"]:
            f["avisos"].append(x["texto"])
    # LA CLAVE DEL DATO DE CADA AVISO (E11): lo de `validar_detalle` ya está;
    # se suman los de `armar` que la tienen.
    f["avisos_claves"] = {**claves_av, **(f.get("avisos_claves") or {})}
    return f


_ROTULOS = {"numero": "el número del asunto", "acto.fecha": "la fecha de lo reclamado o recurrido",
            "acto.toca": "el toca", "acto.expediente": "el expediente de origen",
            "presentacion": "la fecha de presentación", "admision.fecha": "la fecha del auto de admisión",
            "registro.fecha": "la fecha del auto de Presidencia que formó y registró el asunto",
            "turno.fecha": "la fecha del turno", "returno.fecha": "la fecha del returno",
            "notificacion": "la fecha de notificación", "deposito_postal": "la fecha del depósito postal",
            "audiencia": "la fecha de la audiencia constitucional",
            "demanda.fecha": "la fecha de la demanda de amparo",
            "acto.organo": "el órgano que dictó lo reclamado o recurrido",
            # Los nombres (cuarta ronda: el aviso del dato cortado dice cuál es).
            "promovente": "quien promueve o recurre", "representante": "el representante",
            "figura_representante": "la figura del representante",
            "quejoso": "la parte quejosa", "actora": "la parte actora",
            "autoridad_demandada": "la autoridad demandada", "responsable": "la autoridad responsable",
            "ejecutora": "la autoridad ejecutora", "sala": "la Sala del Tribunal Federal de Justicia "
                                                          "Administrativa",
            "adhesivo.quien": "quien promovió el adhesivo", "turno.ponente": "el ponente del turno",
            "returno.ponente": "el ponente del returno", "terceros": "los terceros interesados",
            "demanda.autoridades": "las autoridades responsables de la demanda",
            "resolucion_impugnada.autoridad": "la autoridad que emitió la resolución impugnada",
            "relacionados": "los asuntos relacionados"}


def _rotulo_ruta(ruta: str) -> str:
    return _ROTULOS.get(ruta, ruta.replace(".", " ").replace("_", " "))


def _a(x: str) -> str:
    """«a» + el rótulo, con la contracción: «al auto», «a la presentación»."""
    return "al " + x[3:] if x.startswith("el ") else "a " + x


def _cita(x, tope: int = TOPE_NOMBRE) -> str:
    """Un nombre dentro de un aviso, ENTERO (cuarta ronda, 3-oct-2026: «…Juicios
    Federales en »)» salía cortado a 110 caracteres en el AR 222). Sólo lo que
    pasa del tope de un nombre se corta, en palabra y con «…»."""
    v = _limpio(x)
    if len(v) <= tope:
        return v
    return v[:tope].rsplit(" ", 1)[0].rstrip(" ,;") + "…"


# ═══════════════════════════════════════════════════════════════════════════
# LO QUE NO SE PUEDE FIRMAR TAL COMO LLEGA (CUARTA RONDA, 3-oct-2026, E8)
# ═══════════════════════════════════════════════════════════════════════════
# LOS DATOS CORTADOS. La revisión fiscal del banco traía la figura «Titular de
# la Unidad Jurídica…» (RF 2/2025 y 6/2026: la extracción copió el ejemplo, con
# sus puntos suspensivos) y el compositor la escribió como hecho firmado: «por
# conducto de su Titular de la Unidad Jurídica…, Representante». Un nombre que
# termina —o se interrumpe— en «…» o «...» es un nombre a medias: se quita y se
# dice (hueco antes que medio nombre). «S.A. DE C.V...» no está cortado: es el
# punto de la sigla pegado al de la frase.
_RUTAS_NOMBRE = ("promovente", "representante", "figura_representante", "quejoso", "actora",
                 "autoridad_demandada", "responsable", "ejecutora", "acto.organo", "sala",
                 "adhesivo.quien", "turno.ponente", "returno.ponente", "resolucion_impugnada.autoridad")
_LISTAS_NOMBRE = ("terceros", "demanda.autoridades")
_RX_ELIPSIS = re.compile(r"…|\.{3,}")
_RX_SIGLA_PUNTOS_FINAL = re.compile(r"\b(?:[A-ZÁÉÍÓÚÑ]\.){1,4}[A-ZÁÉÍÓÚÑ]?\.{2,}\s*$")


def truncado(valor) -> bool:
    """¿El valor viene cortado («…» o «...» al final o en medio)?"""
    v = _limpio(valor) if isinstance(valor, str) else ""
    if not v:
        return False
    return bool(_RX_ELIPSIS.search(_RX_SIGLA_PUNTOS_FINAL.sub("", v)))


# EL REPRESENTANTE QUE ES UN CARGO. RF 6/2026: representante «Titular de la
# Unidad de Asuntos Jurídicos de la citada delegación» —un cargo, no una
# persona— y la prosa decía «por conducto de su Titular de la Unidad
# Jurídica…, Titular de la Unidad de Asuntos Jurídicos…». El engrose: «por
# conducto de la Titular de la Unidad de Asuntos Jurídicos de la citada
# delegación, interpuso…», sin nombre. Un cargo va a la FIGURA; si el valor es
# «Persona, Cargo», la persona se queda y el cargo pasa a la figura.
_CARGOS_REPRESENTANTE = (
    r"titular|jef[ea]|subjef[ea]|director[a]?|subdirector[a]?|delegad[oa]|subdelegad[oa]|encargad[oa]|"
    r"coordinador[a]?|administrador[a]?|gerente|secretari[oa]|subsecretari[oa]|procurador[a]?|"
    r"subprocurador[a]?|president[ea]|sindic[oa]|oficial\s+mayor|consejer[oa]|abogad[oa]\s+general|fiscal|"
    r"visitador[a]?|tesorer[oa]|contralor[a]?|comisionad[oa]|vocal|unidad|direccion|subdireccion|jefatura|"
    r"delegacion|subdelegacion|coordinacion|gerencia|oficina|departamento|procuraduria|secretaria|"
    r"consejeria|area")
_RX_CARGO_REPRESENTANTE = re.compile(r"^(?:(?:el|la)\s+)?(?:" + _CARGOS_REPRESENTANTE + r")\b")
# Palabras que no lleva el nombre de una persona y sí el de un cargo
# («Director Jurídico», «Jefe Jurídico Regional»).
_PALABRAS_DE_CARGO = {
    "juridico", "juridica", "juridicos", "regional", "general", "contencioso", "contenciosa", "estatal",
    "federal", "ejecutivo", "ejecutiva", "adjunto", "adjunta", "unidad", "direccion", "delegacion",
    "subdelegacion", "oficina", "asuntos", "instituto", "servicio", "administracion", "jefatura",
    "coordinacion", "area", "departamento", "hacienda", "secretaria", "municipio", "municipal", "estado",
    "nacional", "fiscal", "legal", "local", "desconcentrada", "desconcentrado", "representacion",
}


def _parece_persona(t: str) -> bool:
    pal = _limpio(t).strip(" ,.;:").split()
    if not 2 <= len(pal) <= 6 or not re.match(r"[A-ZÁÉÍÓÚÑ]", pal[0]):
        return False
    for w in pal:
        p = _plano(w).strip(".,;:")
        if p in _PALABRAS_DE_CARGO:
            return False
        if p not in ("de", "del", "la", "las", "los", "y") and not re.match(r"[A-ZÁÉÍÓÚÑ]", w):
            return False
    return True


def es_cargo(valor) -> bool:
    """¿El valor es un CARGO o una unidad («Titular de la Unidad Jurídica»,
    «Subdirector de lo Contencioso», «Director Jurídico») y no el nombre de una
    persona? «Delegado Juan Pérez Ruiz» no: es una figura con su nombre."""
    v = _limpio(valor) if isinstance(valor, str) else ""
    m = _RX_CARGO_REPRESENTANTE.match(_plano(v))
    if not m:
        return False
    # El resto, en el original (la mayúscula dice si es un nombre propio).
    pal_cargo = len(_plano(v)[:m.end()].split())
    resto = " ".join(v.split()[pal_cargo:])
    return not (resto and _parece_persona(resto))


def partir_representante(valor) -> tuple:
    """(persona, cargo) de lo que llegó como representante: «Titular de la
    Unidad Jurídica» → («», «Titular de la Unidad Jurídica»); «Ana Ruiz Pérez,
    Jefa de la Unidad Jurídica» → («Ana Ruiz Pérez», «Jefa de la Unidad
    Jurídica»); una persona sola → (persona, «»)."""
    v = _limpio(valor) if isinstance(valor, str) else ""
    if not v:
        return "", ""
    if es_cargo(v):
        return "", v
    if "," in v:
        antes, despues = (x.strip(" ,") for x in v.split(",", 1))
        if antes and despues and es_cargo(despues) and not es_cargo(antes):
            return antes, despues
    return v, ""


def _quitar_truncados(f: dict) -> list:
    """Los nombres cortados («…», «...»), fuera: [(aviso, [ruta])]."""
    fuentes = f.get("fuentes") if isinstance(f.get("fuentes"), dict) else {}
    out: list = []
    tocadas: set = set()
    for ruta in _RUTAS_NOMBRE:
        v = _tomar(f, ruta)
        if isinstance(v, str) and truncado(v):
            _poner(f, ruta, "")
            fuentes.pop(ruta, None)
            tocadas.add(ruta.split(".")[0])
            # La forma compartida («DATO TRUNCADO (clave = …)», la de
            # `tipos_asunto`, el compositor y `documento_generado`): así el
            # mismo dato cortado se dice UNA vez aunque lo vean varias piezas.
            out.append((f"DATO TRUNCADO ({ruta} = «{_cita(v)}»): viene cortado con puntos suspensivos "
                        f"y se quitó de la ficha ({_rotulo_ruta(ruta)}), porque un nombre a medias se "
                        f"firmaría así. Escríbelo completo, como lo dice el papel.", [ruta]))
    for ruta in _LISTAS_NOMBRE:
        lista = _tomar(f, ruta)
        if isinstance(lista, list) and any(isinstance(x, str) and truncado(x) for x in lista):
            cortados = [x for x in lista if isinstance(x, str) and truncado(x)]
            _poner(f, ruta, [x for x in lista if not (isinstance(x, str) and truncado(x))])
            tocadas.add(ruta.split(".")[0])
            out.append((f"DATO TRUNCADO ({ruta} = «{_cita(cortados[0], 120)}»"
                        f"{f' y {len(cortados) - 1} más' if len(cortados) > 1 else ''}): viene cortado con "
                        f"puntos suspensivos y se quitó de {_rotulo_ruta(ruta)}, porque un nombre a medias se "
                        f"firmaría así. Escríbelo completo.", [ruta]))
    # La sección opcional que se quedó sin nada vuelve a {} (`if ficha["adhesivo"]`).
    for sec in _OPCIONALES:
        d = f.get(sec)
        if sec in tocadas and isinstance(d, dict) and d and all(_vacio(x) for x in d.values()):
            f[sec] = {}
    return out


def _cargo_a_figura(f: dict) -> list:
    """El representante que es un cargo, a la figura: [(aviso, [rutas])]."""
    fuentes = f.get("fuentes") if isinstance(f.get("fuentes"), dict) else {}
    rep = f.get("representante") if isinstance(f.get("representante"), str) else ""
    fig = _limpio(f.get("figura_representante")) if isinstance(f.get("figura_representante"), str) else ""
    persona, cargo = partir_representante(rep)
    if not cargo:
        return []
    rutas = ["representante", "figura_representante"]
    pc, pf = _plano(cargo), _plano(fig)
    pp_ = _plano(f.get("promovente") if isinstance(f.get("promovente"), str) else "")
    sustituye = (not fig or pf in ("representante", "representante legal", "su representante")
                 or pf in pc or pc in pf)
    if sustituye:
        f["figura_representante"] = cargo if len(cargo) >= len(fig) else fig
        if fuentes.get("representante"):
            fuentes["figura_representante"] = fuentes["representante"]
        if pp_ and len(pc.split()) >= 2 and pc in pp_:
            # LA TITULAR DE LA PROPIA UNIDAD QUE RECURRE (E8, RF 7/2025): no es
            # «X, por conducto de su X»; el compositor la escribe en aposición
            # o sola (la figura contenida en la unidad se lo dice).
            av = (f"EL REPRESENTANTE ES EL CARGO DE QUIEN RECURRE (representante = «{_cita(rep)}»; recurre "
                  f"«{_cita(f['promovente'])}»)"
                  + (", NO UNA PERSONA: no se escribe «por conducto de su…». Si consta quién lo ocupa, "
                     "escríbelo como representante." if not persona else
                     f": quien lo ocupa es «{persona}», y no se escribe «por conducto de su…»."))
        elif not persona:
            av = (f"EL REPRESENTANTE ES UN CARGO, NO UNA PERSONA (representante = «{_cita(rep)}»): se escribe "
                  f"como su figura («por conducto de su {_cita(f['figura_representante'])}») y sin nombre. Si "
                  f"consta quién firmó, escríbelo como representante.")
        else:
            av = (f"EL REPRESENTANTE TRAÍA SU CARGO PEGADO (representante = «{_cita(rep)}»): el nombre es "
                  f"«{persona}» y el cargo «{_cita(cargo)}» pasó a la figura.")
    else:
        av = ("EL REPRESENTANTE "
              + ("ES UN CARGO, NO UNA PERSONA" if not persona else "TRAÍA SU CARGO PEGADO")
              + f" (representante = «{_cita(rep)}»)"
              + (f": el nombre es «{persona}»" if persona else "")
              + f", Y LA FIGURA YA DICE «{fig}»: el cargo «{_cita(cargo)}» no se escribe. Compruébalo en "
                f"el escrito del recurso.")
    f["representante"] = persona
    if not persona:
        fuentes.pop("representante", None)
    return [(av, rutas)]


def normalizar(ficha: dict) -> list:
    """Quita de la ficha lo que no se puede firmar tal como llega y dice qué
    quitó: [(aviso, [rutas del dato])]. IDEMPOTENTE: `armar` la hace por
    partes y `validar` entera (el compositor y la verja validan fichas que no
    pasaron por `armar`, como las del banco); la segunda vez ya no hay nada.

      · Un nombre cortado («…», «...») se quita: hueco, no medio nombre.
      · Un representante que es un cargo pasa a la figura; «Persona, Cargo» se
        parte. Si la figura ya dice otra cosa (un delegado, un autorizado), se
        queda la figura y el cargo no se escribe.
      · Un asunto relacionado que no lo es (sexta ronda): el propio asunto, el
        de tipo o número ilegible, el que no marcó el secretario
        (`_relacionados_firmables`)."""
    f = ficha if isinstance(ficha, dict) else {}
    return _quitar_truncados(f) + _cargo_a_figura(f) + _relacionados_firmables(f)


def _relacionados_firmables(f: dict) -> list:
    """LOS RELACIONADOS, COMO LOS DEJA `relacionados_de` (sexta ronda,
    3-oct-2026). Una ficha que no pasó por `armar` —la del banco, la que llega
    de otra pieza— puede traerlos en otra forma, con el propio asunto o con
    una fuente que no es el secretario: la conexidad nunca sale de los papeles
    (David: «no vamos a meter conexidad en automático»). Idempotente."""
    rel = f.get("relacionados")
    if _vacio(rel):
        return []
    fuentes = f.get("fuentes") if isinstance(f.get("fuentes"), dict) else {}
    fu = fuentes.get("relacionados")
    if fu and fu != "secretario":
        f["relacionados"] = []
        fuentes.pop("relacionados", None)
        return [(f"LOS ASUNTOS RELACIONADOS VENÍAN DE «{fu}» Y SÓLO LOS MARCA EL SECRETARIO (relacionados): "
                 f"se quitaron. Si existen, márcalos en «Trámite en este tribunal».", ["relacionados"])]
    lista, avs = relacionados_de(rel, _limpio(f.get("numero")), _limpio(f.get("tipo")))
    if lista != rel:
        f["relacionados"] = lista
        if not lista:
            fuentes.pop("relacionados", None)
    return [(av, ["relacionados"]) for av in avs]


def _coherencia_revision(f: dict, aud, acto_f, avisa) -> None:
    """Lo que en el amparo en revisión no es imposible por fecha pero no puede
    ser (tercera ronda, 3-oct-2026, rev_3). Sólo avisa: corregir es del
    secretario, que tiene el expediente. `avisa(texto, *rutas)`.

      · AUTO DE SOBRESEIMIENTO CON AUDIENCIA. El 81, fr. I, inciso d), es el
        sobreseimiento FUERA de la audiencia constitucional. Si consta una
        audiencia anterior a lo recurrido (AR 239/2025: audiencia el
        29-nov-2024, resolución el 17-feb-2025), lo recurrido parece la
        sentencia de audiencia (inciso e) y el documento diría «por auto…
        fuera de la audiencia» y «Se confirma el auto recurrido».
      · INCIDENTE SIN INTERLOCUTORIA. Lo dictado en el incidente de suspensión
        es la interlocutoria de la suspensión definitiva (inciso a/b).
      · EL ÓRGANO DE LA RECURRIDA ES UNA AUTORIDAD RESPONSABLE, o no tiene
        forma de órgano de amparo (AR 307, 448, 60): es el acto reclamado.
      · LA FECHA DE LA RECURRIDA ES LA DEL ACTO RECLAMADO: si está escrita
        dentro de los actos reclamados de la demanda, se tomó del acto."""
    a = f.get("acto") or {}
    clase = f.get("clase_recurrida") or ""
    if clase == "auto_sobreseimiento" and aud and acto_f and aud <= acto_f:
        avisa(f"CONSTA AUDIENCIA CONSTITUCIONAL ({_dmy(aud)}) ANTES DE LO RECURRIDO ({_dmy(acto_f)}) "
              f"y la ficha dice «auto de sobreseimiento» (81, fr. I, inciso d, que es FUERA de la "
              f"audiencia): lo recurrido parece la sentencia de audiencia (inciso e). Compruébalo en "
              f"el encabezado de la resolución recurrida.", "clase_recurrida", "audiencia")
    if a.get("incidente") is True and ((clase and clase != "interlocutoria_suspension")
                                       or (not clase and a.get("clase") not in ("", "interlocutoria"))):
        avisa(f"LA RESOLUCIÓN RECURRIDA ES DEL INCIDENTE DE SUSPENSIÓN, PERO LA FICHA LA LLAMA "
              f"«{(clase or a.get('clase') or '').replace('_', ' ')}»: lo que se dicta en el incidente "
              f"es la interlocutoria de la suspensión definitiva. Compruébalo antes de firmar la "
              f"competencia y la procedencia.", "clase_recurrida", "acto.incidente")
        # «Y LA PROCEDENCIA», NO «Y LA EXISTENCIA» (revisión AR, 3-oct-2026): C1
        # quitó la existencia del amparo en revisión, y la clase de lo recurrido
        # es la que decide el inciso del 81 en el considerando de procedencia.
    org = _limpio(a.get("organo"))
    auts = (f.get("demanda") or {}).get("autoridades") or []
    if org:
        iguales = [x for x in auts if isinstance(x, str) and _mismo_organo(org, x)]
        if iguales:
            avisa(f"EL ÓRGANO DE LA RESOLUCIÓN RECURRIDA («{_cita(org)}») ES TAMBIÉN UNA DE LAS "
                  f"AUTORIDADES RESPONSABLES DE LA DEMANDA: o la recurrida se confundió con el acto "
                  f"reclamado, o el juzgado se coló entre las autoridades. Compruébalo en la "
                  f"sentencia recurrida y en la demanda.", "acto.organo", "demanda.autoridades")
        elif not es_organo_de_amparo(org):
            avisa(f"EL ÓRGANO DE LA RESOLUCIÓN RECURRIDA («{_cita(org)}») NO TIENE FORMA DE ÓRGANO DE "
                  f"AMPARO (Juzgado de Distrito, Tribunal Colegiado de Apelación, Centro de Justicia "
                  f"Penal Federal): compruébalo en el encabezado de la sentencia recurrida; si es la "
                  f"autoridad responsable, se leyó el acto reclamado.", "acto.organo")
    actos = (f.get("demanda") or {}).get("actos") or []
    iso_acto = a_iso(a.get("fecha")) if a.get("fecha") else ""
    if iso_acto and actos:
        en_actos = {x[0] for t in actos if isinstance(t, str) for x in fechas_del_texto(t)}
        if iso_acto in en_actos:
            avisa(f"LA FECHA DE LA RESOLUCIÓN RECURRIDA ({_dmy(iso_acto)}) ESTÁ ESCRITA EN EL ACTO "
                  f"RECLAMADO de la demanda: parece la fecha del acto reclamado, no la de la "
                  f"sentencia de amparo. Compruébala en el proemio de la sentencia recurrida.",
                  "acto.fecha", "demanda.actos")


# EL MISMO NOMBRE, CON O SIN TRATAMIENTO: «Lic. Bertha Martínez Vega» y «Bertha
# Martínez Vega» son la misma ponente.
_RX_TRATAMIENTO_PERSONA = re.compile(
    r"^(?:(?:el|la)\s+)?(?:licenciad[oa]|lic\.?|licda\.?|maestr[oa]|mtr[oa]\.?|doctor[a]?|dr[a]?\.?)\s+", re.I)


def _misma_persona(a: str, b: str) -> bool:
    pa = _plano(_RX_TRATAMIENTO_PERSONA.sub("", partir_ponente(a)[0]))
    pb = _plano(_RX_TRATAMIENTO_PERSONA.sub("", partir_ponente(b)[0]))
    if not pa or not pb:
        return False
    return pa == pb or (len(pa.split()) >= 2 and len(pb.split()) >= 2 and (pa in pb or pb in pa))


_RX_ROTULO_PARTE = re.compile(
    r"^\s*(?:(?:la|el)\s+)?(?:parte\s+)?(?:quejos[oa]s?|recurrentes?|tercer[oa]s?(?:\s+interesad[oa]s?)?|"
    r"actor(?:a|es)?|agraviad[oa]s?)(?:\s+\d+)?\s*$", re.I)


# Las palabras de rótulo no distinguen a nadie («y otra», el marcador «PARTE
# RECURRENTE» de lo testado dentro de un nombre más largo).
_PALABRAS_DE_ROTULO = {"parte", "recurrente", "recurrentes", "quejosa", "quejoso", "quejosas", "quejosos",
                       "tercero", "tercera", "terceros", "terceras", "interesado", "interesada", "otra", "otro",
                       "otras", "otros", "testado"}


def _palabras_de_nombre(n: str) -> set:
    return {w for w in re.findall(r"[a-z0-9]+", _plano(n))
            if len(w) > 2 and w not in _MINUSCULAS and w not in _PALABRAS_DE_ROTULO}


def validar_detalle(ficha: dict) -> list:
    """Lo mismo que `validar`, con la CLAVE DEL DATO de cada aviso (CUARTA
    RONDA, 3-oct-2026, E11): [{"texto", "rutas": [rutas de la ficha], "clave":
    la primera}]. Un aviso por dato: el compositor y la verja ya no repiten lo
    que esto dijo si su aviso es del mismo dato (`acto.organo`, `presentacion`,
    `acto.fecha`…). Además deja `ficha["avisos_claves"]` = {texto: rutas}."""
    f = ficha if isinstance(ficha, dict) else {}
    tipo = f.get("tipo") or ""
    det: list = []
    imposibles: list = list(f.get("fechas_imposibles") or [])

    def avisa(texto, *rutas):
        if texto and all(x["texto"] != texto for x in det):
            det.append({"texto": texto, "rutas": list(rutas), "clave": rutas[0] if rutas else ""})

    def d(ruta):
        return _fecha_obj(_tomar(f, ruta))

    def choca(cond, clave, texto, *rutas):
        if cond:
            if clave not in imposibles:
                imposibles.append(clave)
            avisa(texto, *rutas)

    # LO QUE NO SE PUEDE FIRMAR, FUERA (E8): idempotente; tras `armar`, nada.
    for texto, rutas in normalizar(f):
        avisa(texto, *rutas)

    acto_f, notif, pres = d("acto.fecha"), d("notificacion"), d("presentacion")
    aud = d("audiencia")
    que = {"amparo_directo": "la sentencia reclamada", "amparo_revision": "la resolución recurrida",
           "queja": "el auto recurrido", "revision_fiscal": "la sentencia recurrida"}.get(tipo, "el acto")
    escrito = "la demanda" if tipo == "amparo_directo" else "el recurso"
    choca(acto_f and notif and notif < acto_f, "notificacion < acto.fecha",
          f"FECHA IMPOSIBLE: la notificación ({_dmy(notif)}) es anterior a {que} ({_dmy(acto_f)}). "
          f"Nada se notifica antes de dictarse; revisa las dos en el formulario.", "notificacion", "acto.fecha")
    # UN HECHO, UN AVISO (E11, AR 448): si la audiencia es del mismo día que lo
    # recurrido, «el recurso es anterior a la recurrida» y «el recurso es
    # anterior a la audiencia» son lo mismo.
    misma_aud = bool(tipo == "amparo_revision" and aud and acto_f and aud == acto_f)
    choca(acto_f and pres and pres < acto_f, "presentacion < acto.fecha",
          f"FECHA IMPOSIBLE: {escrito} se presentó el {_dmy(pres)} y {que} es del {_dmy(acto_f)}"
          f"{' (el mismo día de la audiencia constitucional)' if misma_aud else ''}. "
          f"Nadie combate una resolución que aún no existe; revisa las dos fechas.",
          *(("presentacion", "acto.fecha", "audiencia") if misma_aud else ("presentacion", "acto.fecha")))
    cadena = [("presentacion", "la presentación"),
              ("registro.fecha", "el auto que formó y registró el asunto"),
              ("admision.fecha", "el auto de admisión"),
              ("turno.fecha", "el turno"), ("returno.fecha", "el returno")]
    previo = None
    for ruta, nombre in cadena:
        v = d(ruta)
        if not v:
            continue
        if previo and v < previo[1]:
            choca(True, f"{ruta} < {previo[0]}",
                  f"FECHA IMPOSIBLE: {nombre} ({_dmy(v)}) es anterior {_a(previo[2])} "
                  f"({_dmy(previo[1])}). El orden del trámite es presentación, registro y admisión, turno "
                  f"y returno; revisa esas fechas.", ruta, previo[0])
        previo = (ruta, v, nombre)
    m = re.search(r"/\s*(\d{4})\b", str(f.get("numero") or ""))
    if m and pres:
        anio = int(m.group(1))
        choca(anio < pres.year or anio > pres.year + 1, "año(numero) vs presentacion",
              f"FECHA IMPOSIBLE: el asunto {f.get('numero')} es de {anio} y {escrito} se presentó el "
              f"{_dmy(pres)}. Un expediente no se registra antes de presentarse el escrito ni más de "
              f"un año después; revisa la fecha de presentación.", "presentacion", "numero")
    choca(tipo == "amparo_revision" and aud and acto_f and aud > acto_f, "audiencia > acto.fecha",
          f"FECHA IMPOSIBLE: la audiencia constitucional ({_dmy(aud)}) es posterior a la sentencia "
          f"recurrida ({_dmy(acto_f)}). La sentencia se dicta en la audiencia o después.",
          "audiencia", "acto.fecha")
    dem = d("demanda.fecha")
    # LA QUEJA TAMBIÉN (sexta ronda, 3-oct-2026, C3): desde que su ficha lee la
    # demanda, la demanda va antes del auto recurrido y antes del recurso.
    _con_demanda = tipo in ("amparo_revision", "queja")
    choca(_con_demanda and dem and acto_f and dem > acto_f, "demanda.fecha > acto.fecha",
          f"FECHA IMPOSIBLE: la demanda de amparo ({_dmy(dem)}) es posterior {_a(que)} "
          f"({_dmy(acto_f)}).", "demanda.fecha", "acto.fecha")
    # LA CADENA ENTERA DEL AMPARO EN REVISIÓN (tercera ronda, 3-oct-2026, rev_3):
    # demanda ≤ audiencia ≤ sentencia ≤ recurso, y la demanda ANTES del recurso.
    # En el AR 448 la ficha dio como presentación del recurso la fecha de la
    # demanda (13-may-2025), anterior a la audiencia (15-jul-2025), y el
    # documento afirmó un hecho falso con fecha completa; en el 307 y el 60 la
    # audiencia salió después de la sentencia. Ninguna lo acusaba.
    choca(tipo == "amparo_revision" and dem and aud and dem > aud, "demanda.fecha > audiencia",
          f"FECHA IMPOSIBLE: la demanda de amparo ({_dmy(dem)}) es posterior a la audiencia "
          f"constitucional ({_dmy(aud)}).", "demanda.fecha", "audiencia")
    # Con la audiencia el mismo día que lo recurrido, ya lo dijo el aviso de
    # la presentación anterior a la recurrida (E11).
    choca(tipo == "amparo_revision" and aud and pres and aud > pres
          and not (misma_aud and acto_f and pres < acto_f), "audiencia > presentacion",
          f"FECHA IMPOSIBLE: el recurso de revisión se presentó el {_dmy(pres)}, antes de la audiencia "
          f"constitucional ({_dmy(aud)}). La revisión se interpone contra lo resuelto en la audiencia "
          f"o después; revisa la fecha de presentación del recurso.", "presentacion", "audiencia")
    choca(_con_demanda and dem and pres and dem >= pres, "demanda.fecha >= presentacion",
          f"FECHA IMPOSIBLE: el recurso de {'queja' if tipo == 'queja' else 'revisión'} ({_dmy(pres)}) no "
          f"puede presentarse {'el mismo día que' if dem == pres else 'antes de'} la demanda de amparo "
          f"({_dmy(dem)}); ¿se tomó la fecha de la demanda como la del recurso?", "presentacion", "demanda.fecha")
    if tipo == "amparo_revision":
        _coherencia_revision(f, aud, acto_f, avisa)
    adm = d("admision.fecha")
    for ruta, nombre in (("adhesivo.admision", "la admisión del adhesivo"),
                         ("adhesivo.presentacion", "la presentación del adhesivo"),
                         ("adhesivo.notificacion", "la notificación del auto admisorio al adherente")):
        v = d(ruta)
        choca(v and adm and v < adm, f"{ruta} < admision.fecha",
              f"FECHA IMPOSIBLE: {nombre} ({_dmy(v)}) es anterior al auto de Presidencia que admitió "
              f"el asunto ({_dmy(adm)}).", ruta, "admision.fecha")
    # EL INFORME DEL 101 SE RINDE DESPUÉS DEL AUTO QUE LO PIDE (cuarta ronda):
    # el auto que lo requiere es el del registro. Con la admisión ya no se
    # compara: en la fracción II el auto que tiene por rendido el informe
    # suele ser el mismo que admite (antes, «admisión» era el registro).
    reg, inf = d("registro.fecha"), d("informe_101.fecha")
    choca(reg and inf and inf < reg, "informe_101.fecha < registro.fecha",
          f"FECHA IMPOSIBLE: el auto que tuvo por rendido el informe con justificación ({_dmy(inf)}) es "
          f"anterior al que lo requirió al registrar el recurso ({_dmy(reg)}).",
          "informe_101.fecha", "registro.fecha")
    # LA QUEJA DE LA FRACCIÓN II, EN SUS DOS AUTOS (quinta ronda, 3-oct-2026;
    # Q 335/2025). El de Presidencia registra y REQUIERE el informe del 101; el
    # que lo tiene por rendido es OTRO, posterior, y admite el recurso (el mismo
    # día o después). Con el registro escrito con la fecha del informe —el
    # secretario puso en «fecha_registro» la del auto que admite, o se leyó dos
    # veces el mismo auto— el resultando decía la misma fecha en los dos autos;
    # con la admisión anterior al informe, se tomó como admisión el registro.
    fr97 = _limpio(f.get("fraccion_97")).upper() or \
        ("II" if _plano(_tomar(f, "acto.via")) == "directo" else "")
    if tipo == "queja" and fr97 == "II":
        choca(reg and inf and reg == inf, "registro.fecha = informe_101.fecha",
              f"FECHA IMPOSIBLE: en la queja de la fracción II el auto de Presidencia que registró el recurso "
              f"y requirió el informe con justificación y el que lo tuvo por rendido son dos autos distintos, y "
              f"los dos dicen {_dmy(reg)}. ¿Se escribió como registro la fecha del auto que admite, o se leyó "
              f"dos veces el mismo auto? Corrige «Auto de Presidencia que forma y registra».",
              "registro.fecha", "informe_101.fecha")
        adm_ = d("admision.fecha")
        choca(adm_ and inf and adm_ < inf, "admision.fecha < informe_101.fecha",
              f"FECHA IMPOSIBLE: en la queja de la fracción II el recurso se admite al tener por rendido el "
              f"informe con justificación o después, y la admisión ({_dmy(adm_)}) es anterior al auto que lo "
              f"tuvo por rendido ({_dmy(inf)}). Si esa fecha es la del auto que registró el recurso y requirió "
              f"el informe, va en «Auto de Presidencia que forma y registra».",
              "admision.fecha", "informe_101.fecha")
    dep = d("deposito_postal")
    choca(dep and pres and dep > pres, "deposito_postal > presentacion",
          f"FECHA IMPOSIBLE: el depósito en el correo ({_dmy(dep)}) es posterior a la recepción del "
          f"oficio ({_dmy(pres)}).", "deposito_postal", "presentacion")
    choca(dep and acto_f and dep < acto_f, "deposito_postal < acto.fecha",
          f"FECHA IMPOSIBLE: el recurso se depositó en el correo ({_dmy(dep)}) antes de la sentencia "
          f"recurrida ({_dmy(acto_f)}).", "deposito_postal", "acto.fecha")
    # EL CORREO Y LA VÍA (E6): una sola regla (`via_postal`); si la vía dice
    # otra cosa, el choque se dice.
    via = via_de(f.get("via_presentacion"))
    if dep and via and via != "postal" and via_postal(f):
        avisa(f"HAY FECHA DE DEPÓSITO POSTAL ({_dmy(dep)}) Y LA VÍA DE PRESENTACIÓN DICE «{via}»: se tomó "
              f"como presentado por correo"
              f"{', y la oportunidad se mide con el depósito' if tipo == 'revision_fiscal' else ''}. "
              f"Si no se mandó por correo, borra la fecha del depósito; si sí, pon la vía «postal».",
              "via_presentacion", "deposito_postal")
    ri = d("resolucion_impugnada.fecha")
    choca(tipo == "revision_fiscal" and ri and acto_f and ri > acto_f, "resolucion_impugnada > acto.fecha",
          f"FECHA IMPOSIBLE: la resolución impugnada ({_dmy(ri)}) es posterior a la sentencia de la "
          f"Sala que la juzgó ({_dmy(acto_f)}).", "resolucion_impugnada.fecha", "acto.fecha")
    hoy = _dt.date.today()
    for ruta, v in _hojas(f):
        if ruta.startswith(("fuentes", "avisos", "fechas_imposibles")) or not _es_ruta_fecha(ruta):
            continue
        o = _fecha_obj(v)
        choca(o and o > hoy, f"{ruta} futura",
              f"FECHA IMPOSIBLE: {_rotulo_ruta(ruta)} ({_dmy(o)}) es posterior a hoy.", ruta)
    # La forma de la responsable (AD).
    if tipo == "amparo_directo" and f.get("responsable"):
        ok_, mot = es_organo(f["responsable"])
        if not ok_:
            avisa(f"LA AUTORIDAD RESPONSABLE «{_cita(f['responsable'])}» NO TIENE FORMA DE ÓRGANO ({mot}): "
                  f"escríbela como la nombra la demanda.", "responsable")
    # UN AUTORIZADO NO INTERPONE LA REVISIÓN FISCAL (E8). El artículo 63 de la
    # LFPCA da el recurso a la unidad administrativa encargada de la defensa
    # jurídica de la autoridad; los autorizados son de los particulares (RF
    # 4/2025: la ficha le puso a la autoridad el autorizado de la actora).
    fig = _plano(f.get("figura_representante"))
    if tipo == "revision_fiscal" and (f.get("caracter") or "autoridad") == "autoridad" \
            and re.search(r"\bautorizad[oa]s?\b", fig):
        # EL MISMO TEXTO QUE EL DEL COMPOSITOR (`resultandos_por_tipo`, E8):
        # un aviso por dato, también cuando lo ven los dos.
        avisa(f"UN AUTORIZADO NO INTERPONE LA REVISIÓN FISCAL POR LA AUTORIDAD "
              f"(figura_representante = «{_limpio(f.get('figura_representante'))}»): el artículo 63 de la "
              f"LFPCA exige que la interponga la unidad administrativa encargada de su defensa jurídica. "
              f"Compruébalo en el oficio; puede ser el autorizado de la actora, que va en la adhesiva.",
              "figura_representante", "representante")
    # QUIEN RECURRE COMO QUEJOSA Y NO ES LA QUEJOSA (E12). AR 60/2025: la
    # sucesión era la quejosa y recurrió su albacea, una persona física, con el
    # carácter de «quejoso»: el V I S T O decía «interpuesto por» la albacea y la
    # legitimación la llamaba «parte quejosa». Si todas las palabras del nombre
    # de una están en el de la otra («Sucesión… a través de su albacea X» y «la
    # Sucesión…»; «Juan Pérez» y «Juan Pérez y otros»), es la misma parte.
    # Un RÓTULO («parte quejosa», «PARTE RECURRENTE», el marcador de lo testado)
    # no es un nombre: no se compara.
    prom, q = _limpio(f.get("promovente")), _limpio(f.get("quejoso"))
    if tipo in ("amparo_revision", "queja") and f.get("caracter") == "quejoso" and prom and q \
            and not (_RX_ROTULO_PARTE.match(prom) or _RX_ROTULO_PARTE.match(q)) \
            and not _misma_parte(prom, q):
        wp, wq = _palabras_de_nombre(prom), _palabras_de_nombre(q)
        if wp and wq and not (wp <= wq or wq <= wp):
            avisa(f"QUIEN RECURRE COMO QUEJOSA («{_cita(prom)}») NO ES LA PARTE QUEJOSA («{_cita(q)}»): ¿es su "
                  f"albacea, su representante o su autorizado? Si lo es, escríbelo como representante, con "
                  f"su figura («{_cita(q)}, por conducto de su albacea…»); si recurre otra parte, corrige el "
                  f"carácter.", "promovente", "quejoso", "caracter")
    f["fechas_imposibles"] = imposibles
    claves = f.get("avisos_claves") if isinstance(f.get("avisos_claves"), dict) else {}
    for x in det:
        claves[x["texto"]] = list(x["rutas"])
    f["avisos_claves"] = claves
    return det


def validar(ficha: dict) -> list:
    """La CRONOLOGÍA, la forma de la responsable y la coherencia de quién
    recurre. Cada fecha imposible va a `ficha["fechas_imposibles"]` y a un aviso
    en mayúsculas. No corrige nada que se pueda firmar: dice qué no puede ser,
    para que lo corrija quien tiene el expediente; sólo QUITA lo que no se
    puede firmar como llega (un nombre cortado, un cargo como representante:
    `normalizar`). Con la clave de cada aviso: `validar_detalle`."""
    return [x["texto"] for x in validar_detalle(ficha)]


def clave_de_aviso(ficha: dict, texto: str) -> list:
    """Las rutas del dato de un aviso de `validar`/`armar` («acto.organo»,
    «presentacion»…), o [] si no se sabe. Para no repetir el mismo hecho con
    otro texto (E11)."""
    c = (ficha or {}).get("avisos_claves") if isinstance(ficha, dict) else None
    return list((c or {}).get(texto) or [])
