# -*- coding: utf-8 -*-
"""EL COMPOSITOR POR TIPO: el V I S T O y los resultandos, sin modelo (3-oct-2026).

    componer(tipo, ficha, datos) -> {"visto", "resultandos", "avisos",
                                     "avisos_repetidos", "datos_extra"}
    considerando_adhesivo(tipo, ficha, datos) -> (rótulo, texto, avisos)
    resolutivo_adhesivo(tipo, prospera_principal, ficha) -> (texto, aviso)
    nombre_en_prosa(nombre, autoridad=False, con_articulo=False) -> str

═══════════════════════════════════════════════════════════════════════════
POR QUÉ EXISTE
═══════════════════════════════════════════════════════════════════════════
Hasta hoy UNA llamada de modelo (`documento_generado.redactar_estructura`)
escribía el V I S T O y TODOS los resultandos a ciegas —sin el auto de
admisión, sin el de turno, sin la demanda— y después el código leía esa prosa
con expresiones regulares para llenar la existencia, la procedencia y los
resolutivos. Medido en los 10 proyectos de octubre (SPEC §0): la fecha del auto
de Presidencia «que se advierte de las constancias» en 76 de 81, «el Ministerio
Público omitió formular pedimento» afirmado sin fuente en 71 de 72, la
«Oficialía de Partes de este Tribunal» inventada en 16, 43 V I S T O distintos
en 81 proyectos y la etiqueta del encabezado colada en el número («ADC
625-2024 ORAL MERCANTIL»).

LA REGLA DE ORO: LOS DATOS PRIMERO, LA PROSA DESPUÉS. Este módulo no lee papel
ni llama a ningún modelo: recibe la FICHA DE TRÁMITE (`ficha_tramite.armar`,
con cada dato y su fuente) y escribe con fórmulas fijas, medidas en el corpus
del tribunal (scratchpad/procedencia/res/oro_*.txt). Un mismo valor alimenta el
V I S T O, los resultandos y `datos_extra`, que es lo que los considerandos y
el resolutivo leen en lugar de la prosa: así la fecha del acto no puede salir
distinta en el proemio y en el resolutivo (30 de 529 AD del corpus lo hacían).

DATO OBLIGATORIO QUE FALTA → HUECO VISIBLE «*********» + AVISO QUE NOMBRA EL
DATO Y DÓNDE ESTÁ. Nunca perífrasis («se advierte de las constancias», «en los
términos que obran en autos») y nunca afirmación por omisión: el Ministerio
Público sólo se nombra si la ficha dice qué hizo; la oficina receptora, sólo si
consta la vía.

ES PURO Y NO LANZA. Los helpers de `documento_generado`, `tipos_asunto` y
`fase0_oportunidad` se importan DENTRO de las funciones: `documento_generado`
llegará a importar este módulo y un import circular en el arranque tumbaría el
taller entero.

LA BANDERA no vive aquí: este módulo sólo se llama cuando
`contexto_taller.rige("procedencia_por_tipo")` está puesta (lo decide el
cableado). `activo()` la consulta sin lanzar, para quien la necesite.

TERCERA RONDA (3-oct-2026, FIXES_R3 · D), medida contra los 32 engroses del
banco de oráculo y la prueba de punta a punta (AD 274/2025 y AR 631/2025):
  · la única instancia la dice la FICHA: con toca o con un órgano de alzada no
    la hay, aunque la marca del contexto diga otra cosa (y se avisa);
  · los nombres que llegan en versales pasan a la prosa sin versales; la
    carátula los sigue escribiendo en versales (`documento_generado`);
  · la ficha se valida al componer (`ficha_tramite.validar`, sobre una copia);
  · número gramatical del sujeto, coma que cierra el inciso, artículo de la
    sucesión, la comunidad y el ejido, mayúscula al abrir la oración, sin las
    etiquetas de rol de la carátula, sin «por conducto de» a sí mismo;
  · AR: el juzgado no puede ser una autoridad responsable (ni al revés);
  · la vía electrónica se dice en la interposición (AR y queja);
  · queja: el órgano en forma de órgano y el nombre más completo;
  · RF: la unidad que recurre y a quién representa, sin duplicar; la negativa
    ficta; el sentido del catálogo; depósito y recepción del mismo día;
  · el adhesivo de la RF con la regla literal del 63, penúltimo párrafo, y la
    cláusula de inhábiles de `fase0_oportunidad`;
  · `datos_extra` nuevos: clase, acto_reclamado, ponente, adherente,
    recurrente_unidad (y autoridad_demandada siempre presente).

CUARTA RONDA (3-oct-2026, FIXES_R4 · D), contra la verificación final de los
32 engroses (rev/verif_final.txt):
  · E1 · `nombre_en_prosa` es LA puerta de los nombres para todo el documento:
    pasa a prosa las rachas en versales aunque el texto venga mezclado («la
    Sucesión a Bienes de JOSÉ GARCÍA RUIZ»), deja en minúscula las fórmulas de
    representación («por conducto de su consejo de administración», «y otros»)
    y quita la numeración y la coletilla de la carátula («1) …», «(ya
    mencionados…)»);
  · E2 · el plural lo decide UN detector, `tipos_asunto.es_plural_de_partes`, y
    viaja en `datos_extra["plural"]` para la legitimación y la carátula;
  · E3 · el ponente con su tratamiento y su artículo («de la licenciada…»), y
    con el cargo que leyó el auto de turno o de returno (`titulo`);
  · E5 · el adhesivo sin ningún auto: hueco y pregunta, sin considerando ni
    resolutivo;
  · E6 · el correo de la revisión fiscal con la regla única
    `ficha_tramite.via_postal`;
  · E7 · el registro y la admisión, dos autos cuando son dos;
  · E8 · la unidad que recurre y su titular, sin «X, por conducto de su X»;
  · E10 · AR: el órgano de la recurrida en forma de órgano, y el que se
    descartó llega en hueco también a la competencia y la existencia;
  · E11 · un aviso por dato: lo que ya dijo el compositor no se repite con el
    aviso de `validar`.

QUINTA RONDA (3-oct-2026, FIXES_R5 · D), contra la segunda verificación de
los 32 engroses (rev/verif_final2.txt):
  · F3 · la fracción del 97 sólo con certeza: «del Distrito Judicial» es un
    juzgado local (fracción II), no un Juzgado de Distrito; sin certeza,
    fracción y vía en hueco, sin suponer la I (Q 335/2025);
  · F4 · una sola comparación de autoridades, `tipos_asunto.misma_autoridad`
    (RF 7/2025: «Representación» y «Delegación» Estatal, la misma unidad);
  · F2 · la figura sin nombre no se borra: «por conducto de su delegado
    *********» con el aviso de la clave «representante» (AR 72, Q 342);
  · F5 · el artículo del papel en los cargos de doble género («la Oficial
    Mayor…», AD 128/2025); sin él, «el» y el aviso del género por omisión;
  · F6 · una sola puerta para la figura y su separador (`tipos_asunto.
    figura_en_prosa` y `_sep_figura`);
  · `nombre_en_prosa`: madre, padre, «en su carácter de albacea…», «quien
    también se ostenta», «y/o», «ambos por propio derecho», la coma antes de
    la fórmula de representación, el camino de órgano, las etiquetas de rol
    compuestas, SOFOM y «no»; `_cierra_inciso` con «;» y la razón social al
    final; la «y» final con su coma; el ponente «en funciones» en aposición;
    el sentido de la queja con minúscula; la negativa ficta sin «solicitud de
    solicitud»; el sentido de la Sala que trae más que el catálogo; el
    nombre abreviado sólo con remisión; el registro de la queja fr. II con la
    fecha de otro auto, en hueco; el autorizado de la autoridad fuera de los
    dos lados; la Sala que cambió de nombre.

SEXTA RONDA (3-oct-2026), lo que David respondió y aprobó para todos:
  · C3 · la queja de la fracción I abre con «Demanda de amparo.» cuando la
    ficha la trae, con la misma mecánica del primer resultando de la revisión
    (`_texto_de_la_demanda`, una sola para los dos); sin ella, como siempre y
    sin aviso: es contexto, no requisito;
  · C5 · `datos_extra` de la revisión lleva `actos` (limpios) y
    `resultando_demanda` («primero»): el punto que confirma nombra acto y
    autoridad (`tipos_asunto.punto_del_amparo`);
  · C6 · los asuntos relacionados que marcó el secretario —nunca leídos de
    los papeles— van en el V I S T O tras el número («…, relacionado con el
    amparo directo civil 452/2025, …») y en `datos_extra["relacionados"]`,
    de donde salen el rubro y el considerando.
"""
from __future__ import annotations

import datetime as _dt
import re
import unicodedata

# El mismo hueco que `documento_generado.HUECO`: la verja procesal y el
# compositor del .docx lo buscan por su forma exacta (la prueba lo comprueba).
HUECO = "*********"
BANDERA = "procedencia_por_tipo"

TIPOS = ("amparo_directo", "amparo_revision", "queja", "revision_fiscal")


def activo() -> bool:
    """¿Rige la procedencia por tipo en esta petición? Nunca lanza: sin el
    contexto o sin la bandera declarada, el camino viejo (False)."""
    try:
        import contexto_taller as _ct
        return bool(_ct.rige(BANDERA))
    except Exception:
        return False


# ═══════════════════════════════════════════════════════════════════════════
# UTILIDADES — sin estado, sin red
# ═══════════════════════════════════════════════════════════════════════════
def _d(x) -> dict:
    return x if isinstance(x, dict) else {}


def _lista(x) -> list:
    """Una lista para recorrer: la lista o la tupla tal cual, un texto como
    lista de uno y cualquier otra cosa (un número, un dict) como vacía. Un
    `terceros: 12.5` tumbaba el compositor (TypeError) y el documento caía al
    camino viejo sin decir por qué (fuzz de la tercera ronda, 3-oct-2026)."""
    if isinstance(x, (list, tuple)):
        return list(x)
    if isinstance(x, str):
        return [x] if x.strip() else []
    return []


# LOS CARACTERES DE CONTROL NO LLEGAN AL .docx (3-oct-2026, rev_6): uno solo
# —«Ana\x00Pérez» en el ponente, o el \x00 que trae el texto de fitz en el 5-6 %
# de los PDF del OAJ— tumbaba el adelanto con «All strings must be XML
# compatible» al escribir el documento, DESPUÉS de pagar el OCR y el modelo.
_RX_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def _s(x) -> str:
    """El texto con los espacios cerrados. NO quita el punto final: «S.A. de
    C.V.» lo necesita, y quien lo quite se lo come a la razón social.

    UN VALOR QUE NO ES TEXTO ES UN DATO QUE FALTA (3-oct-2026, rev_6): con
    `{"ponente_turno": {"nombre": "Ana"}}` salía «a la ponencia de {'nombre':
    'Ana'}». Un dict no es un nombre ni una fecha: vale «» y el hueco con su
    aviso lo dice. En una lista sólo cuentan los textos y los números."""
    if x is None or isinstance(x, (dict, set, frozenset, bool)):
        return ""
    if isinstance(x, (list, tuple)):
        x = " ".join(str(v) for v in x
                     if isinstance(v, (str, int, float)) and not isinstance(v, bool) and v)
    return " ".join(_RX_CONTROL.sub("", str(x)).split()).strip(" ,;")


def _sin_tildes(x: str) -> str:
    t = unicodedata.normalize("NFKD", x or "")
    return "".join(c for c in t if not unicodedata.combining(c)).lower()


def _normalizar_tipo(tipo: str) -> str:
    try:
        import tipos_asunto as _ta
        t = _ta.normalizar(tipo)
        if t:
            return t
    except Exception:
        pass
    t = _sin_tildes(_s(tipo)).replace(" ", "_").replace("-", "_")
    return t if t in TIPOS else ""


def _fecha(x):
    """ISO «AAAA-MM-DD» (o un `date`, o «dd/mm/aaaa») → `date`; None si no se
    puede leer. Una fecha que no se lee se trata como fecha que falta: una
    fecha aproximada en un resultando es peor que un hueco, que se ve."""
    f = _fecha_cruda(x)
    # UNA FECHA DEL AÑO 1 O DEL 9999 NO ES DE UN EXPEDIENTE (rev_6): con ella el
    # cómputo del adhesivo lanzaba OverflowError. Fuera de 1900-2100 es ilegible.
    return f if (f is not None and 1900 <= f.year <= 2100) else None


def _fecha_cruda(x):
    if isinstance(x, _dt.datetime):
        return x.date()
    if isinstance(x, _dt.date):
        return x
    t = _s(x)
    if not t:
        return None
    try:
        return _dt.date.fromisoformat(t[:10])
    except ValueError:
        pass
    m = re.match(r"^(\d{1,2})[/.-](\d{1,2})[/.-](\d{4})$", t)
    if m:
        try:
            return _dt.date(int(m.group(3)), int(m.group(2)), int(m.group(1)))
        except ValueError:
            return None
    return None


def _letra(f) -> str:
    """Día, mes y AÑO completos, siempre (SPEC §2): «diez de diciembre» sin año
    vuelve ambigua la fecha y abunda en el corpus (oro_Q, errores)."""
    if f is None:
        return ""
    import fase0_oportunidad as _f0
    return _f0.fecha_en_letra(f)


_RX_ROMANO = re.compile(r"^(?=[IVXL])(XL|L?X{0,3})(IX|IV|V?I{0,3})$")
# Dos o más «letra.» al cierre: la abreviatura de una razón social («S.A.»,
# «C.V.», «C.T.M.», «S. de R.L.»). Un «Centro I.» con punto suelto no lo es.
_RX_ABREVIATURA_FINAL = re.compile(r"(?:\b[A-Za-zÁÉÍÓÚÑáéíóúñ]\.\s?){2,}$")

# LAS SIGLAS DE LAS AUTORIDADES FEDERALES, que el paso de versales a prosa
# convierte en «Imss» o «Issste». Sólo las que salen en el corpus de RF y AR
# (oro_RF: SAT, IMSS, ISSSTE, CONAGUA, INFONAVIT, ANAM, SFP, STPS…).
_SIGLAS = {"IMSS", "ISSSTE", "SAT", "SHCP", "TFJA", "CONAGUA", "INFONAVIT", "ANAM",
           "CDMX", "SCJN", "CJF", "OAJ", "PJF", "UIF", "CFE", "PEMEX", "FOVISSSTE",
           "CONDUSEF", "PROFECO", "SEP", "SFP", "STPS", "SEMARNAT", "COFEPRIS", "IFT",
           "CRE", "CNH", "ASF", "COMAR", "INM", "IMPI", "SADER", "SEDATU", "RAN",
           "SEDENA", "SICT", "SE", "CNBV", "CONSAR", "SRE"}
# «SE» ES SIGLA SÓLO EN EL NOMBRE DE UNA AUTORIDAD (quinta ronda, 3-oct-2026, Q
# 335 con nombres de prueba): en el de una persona es el pronombre, y «JUAN
# PÉREZ, QUIEN TAMBIÉN SE OSTENTA COMO…» salía «Quien También SE Ostenta». El
# camino de órgano ya exige más de dos letras para conservar una sigla.
_SIGLAS_QUE_SON_PALABRA = {"SE"}


def _palabra_de_organo(w: str) -> str:
    # «S.n.c.», «C.v»: la abreviatura que el paso a prosa dejó a medias (el
    # punto final se lo quitó `_nombre_de_organo`; `_organo` lo repone).
    if re.fullmatch(r"(?:[A-Za-zÁÉÍÓÚÑáéíóúñ]\.)+[A-Za-zÁÉÍÓÚÑáéíóúñ]\.?[,;:]?", w):
        return w.upper()
    nucleo = w.strip(".,;:()«»\"'")
    if nucleo and len(nucleo) <= 6 and _RX_ROMANO.match(nucleo.upper()):
        return w.replace(nucleo, nucleo.upper())
    if nucleo and nucleo.upper() in _SIGLAS and len(nucleo) > 2:
        return w.replace(nucleo, nucleo.upper())
    return w


def _organo(x) -> str:
    """El nombre de un órgano en prosa: sin versales, con los romanos y las
    siglas en mayúscula y sin artículo.

    LOS ROMANOS SE REPONEN AQUÍ. `documento_generado._nombre_de_organo`
    convierte «SALA REGIONAL DEL CENTRO II» en «Sala Regional del Centro Ii»
    (comprobado el 3-oct-2026): su comentario dice que los ordinales romanos y
    las siglas se quedan como están, pero el código los capitaliza como
    palabras. Sólo se tocan las palabras hechas de I, V, X y L que formen un
    romano válido —el circuito y la región nunca pasan de XL—, para no
    convertir «Civil» en «CIVIL», y las siglas de la lista.

    EL PUNTO FINAL SE QUITA, SALVO EL DE UNA ABREVIATURA («S.A.», «C.V.»,
    «A.C.», «S. de R.L.»): quitárselo deja «…, S.A. de C.V» en el renglón de
    AUTORIDAD RESPONSABLE, y el doble punto que pudiera quedar en la prosa lo
    quita `_pulir`."""
    t, _rol = _sin_rol(_s(x))
    abrev = bool(_RX_ABREVIATURA_FINAL.search(t))
    t = t.rstrip(" .")
    if not t:
        return ""
    import documento_generado as _dg
    versales = t == t.upper()
    t = _dg._nombre_de_organo(t)
    if versales:
        t = " ".join(_palabra_de_organo(w) for w in t.split(" "))
    t = _dg._sin_articulo(t)
    # `_nombre_de_organo` también le quita el punto a «S.N.C.»: se repone.
    if abrev and t and not t.endswith("."):
        t += "."
    if not t:
        return ""
    # EL PREFIJO TEMPORAL VA EN MINÚSCULA (3-oct-2026, RF 49/2025): «ACTUAL
    # SALA REGIONAL EN QUERÉTARO» salía «el Actual Sala…, localizado» en el
    # V I S T O, los resultandos, la competencia y el resolutivo. «actual»
    # califica al órgano, no es parte de su nombre; el artículo lo elige
    # `_con_art` por el sustantivo que sigue («la actual Sala…»).
    # Y A MEDIA FRASE TAMBIÉN (quinta ronda, RF 21/2025): «SALA REGIONAL DEL
    # CENTRO II, AHORA SALA REGIONAL EN QUERÉTARO» → «…, ahora Sala…».
    t = re.sub(r"(?<=, )(Ahora|Hoy|Actualmente|Antes|Entonces|Otrora)\b",
               lambda x: x.group(1).lower(), t)
    m = _RX_PREFIJO_TEMPORAL.match(t)
    if m:
        resto = t[m.end():]
        return f"{m.group(1).lower()} {resto[:1].upper()}{resto[1:]}"
    return t[:1].upper() + t[1:]


# ── LAS ETIQUETAS DE ROL DE LA CARÁTULA (3-oct-2026, AR 208/2025 y RF 4/2025) ──
# Las carátulas del circuito ponen el papel entre paréntesis tras el nombre
# —«GOBERNADOR DEL ESTADO DE QUERÉTARO (AUTORIDAD RESPONSABLE)», «… DEL ISSSTE
# (DEMANDADA)»— y la etiqueta viajaba dentro del nombre: «interpuesto por el
# Gobernador del Estado de Querétaro (autoridad Responsable), en su carácter de
# autoridad responsable». La etiqueta se quita; si el carácter no consta, lo da.
#
# LAS ETIQUETAS COMPUESTAS (quinta ronda, 3-oct-2026, AD 552/2024 con el
# quejoso como lo escribe su carátula): «(DEMANDADOS PRINCIPALES Y APELANTES)»
# salía «(DEMANDADOS Principales y Apelantes)» en el V I S T O, el resultando,
# la legitimación y el resolutivo, y «(ACTORA EN EL JUICIO NATURAL)» como
# «(ACTORA en el Juicio Natural)». La etiqueta es una SERIE de palabras de rol
# unidas por «y» o coma, con «en el juicio natural/de origen/principal» al
# final; un paréntesis final hecho SÓLO de esas palabras se quita, siempre que
# traiga al menos un rol de verdad (no basta «(LA)»).
_PALABRA_DE_ROL = (
    r"(?:autoridad(?:es)?|responsables?|demandad[oa]s?|codemandad[oa]s?|"
    r"actor(?:a|as|es)?|coactor(?:a|as|es)?|quejos[oa]s?|tercer[oa]s?|interesad[oa]s?|"
    r"recurrentes?|adherentes?|apelantes?|apelad[oa]s?|principal(?:es)?|"
    r"reconvencionistas?|reconvenid[oa]s?|parte|la|el|los|las)")
# EL SEPARADOR ES UNO Y SIN AMBIGÜEDAD («, », «, y », « y », « e » o un espacio): con
# «\s*(?:,|y)?\s*» el espacio entre dos palabras se podía repartir de dos
# maneras y un paréntesis de 22 palabras que no cerraba en rol tardaba 3,5 s
# (backtracking exponencial, medido el 3-oct-2026).
_RX_ROL = re.compile(
    r"\s*\(\s*(?P<rol>" + _PALABRA_DE_ROL + r"(?:(?:\s*,\s*(?:[ye]\s+)?|\s+[ye]\s+|\s+)" + _PALABRA_DE_ROL +
    r"(?![\w]))*"
    r"(?:\s+en\s+el\s+(?:juicio|expediente)\s+(?:natural|de\s+origen|principal|de\s+amparo))?)"
    r"\s*\)\s*\.?\s*$", re.I)
_RX_ROL_DE_VERDAD = re.compile(
    r"autoridad|demandad|actor|quejos|tercer|recurrent|adherent|apelant|apelad|reconven")


def _sin_rol(nombre: str) -> tuple:
    """(nombre sin la etiqueta de rol final, carácter que la etiqueta dice):
    «autoridad», «quejoso», «tercero» o «» (demandada, actora, recurrente,
    adherente y apelante no son caracteres del amparo)."""
    n = nombre or ""
    m = _RX_ROL.search(n)
    if not m or not _RX_ROL_DE_VERDAD.search(_sin_tildes(m.group("rol"))):
        return n, ""
    rol = re.sub(r"^(?:(?:la|el|los|las)\s+)?(?:parte\s+)?", "", _sin_tildes(m.group("rol")))
    car = ("autoridad" if rol.startswith("autoridad") and "responsable" in rol else
           "quejoso" if rol.startswith("quejos") else
           "tercero" if rol.startswith("tercer") else "")
    return n[:m.start()].rstrip(" ,;"), car


_RX_PREFIJO_TEMPORAL = re.compile(
    r"(?i)^(actual|entonces|hoy|ahora|otrora|extint[oa]|antes)\s+(?=\S)")

# LOS CARGOS Y ÓRGANOS FEMENINOS que `documento_generado._con_articulo` no
# conocía (3-oct-2026, rev_1/rev_3/rev_5): salían «el Jefa de la Unidad
# Jurídica», «el Subdirectora de Afiliación», «el Coordinadora de Recursos
# Humanos», «el Legislatura del Estado». La misma lista que pide FIXES_R3 para
# `_con_articulo`; aquí va como guarda para que lo que compone este módulo
# salga bien aunque aquella todavía no la tenga. Los cargos en -dora, -tora y
# -sora (directora, procuradora, auditora, asesora) los cubre la regla.
_FEMENINOS_CARGO = {
    "jefa", "subjefa", "delegada", "subdelegada", "encargada", "presidenta",
    "comisionada", "consejera", "contralora", "tesorera", "legislatura", "camara",
    "jefatura", "subjefatura", "oficina", "sindicatura", "gubernatura", "regiduria",
    "alcaldia", "gerenta", "subsecretaria", "secretaria", "jueza", "magistrada",
}


def _la_titular(crudo) -> bool:
    """¿El papel dice «la titular»? (Q 300/2025: «por la titular del Juzgado
    Cuarto de Distrito»). Entonces el artículo es el del papel, no el genérico."""
    return bool(re.match(r"(?i)^\s*la\s+titular\b", _s(crudo)))


# ── EL ARTÍCULO DEL PAPEL, PARA LOS CARGOS DE DOBLE GÉNERO (quinta ronda, 3-oct-2026, F5) ──
# AD 128/2025: la extracción ya traía «la Oficial Mayor y Coordinadora de
# Recursos Humanos del Municipio de Cadereyta de Montes» y el resultando del
# tercero decía «Tiene ese carácter el Oficial Mayor…»: `_organo` quita el
# artículo y `_con_art` sólo respetaba el del papel para «Titular»; para
# «Oficial» caía a `documento_generado._con_articulo`, que pone «el». Pasa con
# cualquier cargo que no cambia de forma con el género (Titular, Oficial,
# Fiscal, Agente, Representante, Juez, Encargado/a). El género lo dice el
# PAPEL —su artículo—, nunca el nombre de quien ocupa el cargo; sin artículo
# en el papel, «el» (masculino genérico) y el aviso «EL GÉNERO DEL CARGO NO
# CONSTA» (`_aviso_genero_del_cargo`).
_CARGOS_DOBLE_GENERO = {"titular", "oficial", "fiscal", "agente", "representante", "juez",
                        "encargado", "encargada", "gerente", "comandante", "intendente",
                        "superintendente", "edil"}


def _articulo_del_papel(crudo) -> str:
    """El artículo con que el papel abre el nombre: «la», «el», «las», «los»
    o «» si no trae ninguno («la Oficial Mayor…» → «la»)."""
    m = re.match(r"(?i)^\s*(el|la|los|las)\s+(?=\S)", _s(crudo))
    return m.group(1).lower() if m else ""


def _primera_del_cargo(nombre: str) -> str:
    """La primera palabra del cargo, sin tildes, sin artículo y sin el prefijo
    temporal («la entonces Oficial Mayor» → «oficial»)."""
    n = re.sub(r"(?i)^(?:el|la|los|las)\s+", "", _s(nombre))
    n = _RX_PREFIJO_TEMPORAL.sub("", n)
    return _sin_tildes(n.split(" ", 1)[0]).strip(".,;:") if n else ""


def _con_art(nombre: str, la_titular: bool = False, art_papel: str = "") -> str:
    """«la Primera Sala Civil…», «el Juzgado Cuarto…».

    «TITULAR» ES EL CARGO, NO LA PERSONA. `documento_generado._con_articulo` lo
    trata como femenino («la Titular de la Jefatura…») y eso es adivinar quién
    ocupa el cargo. El corpus nombra el cargo en masculino genérico: «el
    Titular de» 59 veces contra 12 «la Titular de» (RF y AR, 3-oct-2026).
    SALVO QUE EL PAPEL DIGA «la titular» (`la_titular`): entonces no se adivina
    nada, se copia (Q 300/2025, tercera ronda).

    QUINTA RONDA (F5): lo mismo para TODO cargo de doble género
    (`_CARGOS_DOBLE_GENERO`): `art_papel` es el artículo con que el papel abría
    el nombre («la Oficial Mayor…», AD 128/2025) y se conserva; sin él, «el».

    El prefijo temporal («actual», «entonces», «hoy»…) se salta para elegir el
    artículo: «la actual Sala Regional…», no «el Actual Sala»."""
    if not nombre:
        return ""
    art_p = (art_papel or "").strip().lower()
    if art_p not in ("el", "la", "los", "las"):
        art_p = "la" if la_titular else ""
    n = nombre.strip()
    m = _RX_PREFIJO_TEMPORAL.match(n)
    if m:
        resto = n[m.end():]
        con = _con_art(resto[:1].upper() + resto[1:], art_papel=art_p)
        art, _, cuerpo = con.partition(" ")
        if art.lower() in ("el", "la", "los", "las") and cuerpo:
            return f"{art.lower()} {m.group(1).lower()} {cuerpo}"
        return n
    if re.match(r"(?i)^(?:el|la|los|las)\s", n):
        return n
    primera = _sin_tildes(n.split()[0]).strip(".,;:")
    if primera in _CARGOS_DOBLE_GENERO:
        return f"{art_p or 'el'} {n}"
    if primera in _FEMENINOS_CARGO or re.search(r"(?:dora|tora|sora)$", primera):
        return f"la {n}"
    import documento_generado as _dg
    return _dg._con_articulo(n)


def _organo_art(crudo) -> str:
    """El órgano del papel, en prosa y con su artículo («la Titular…», «la
    Oficial Mayor…» si el papel lo dice así). «» si no hay nombre."""
    o = _organo(crudo)
    return _con_art(o, art_papel=_articulo_del_papel(crudo)) if o else ""


def _mayus(t: str) -> str:
    """Mayúscula inicial para el sujeto que ABRE la oración: «la Sucesión…
    promovió» al principio del resultando salía con minúscula (AR 60/2025)."""
    return t[:1].upper() + t[1:] if t else t


def _contraer(t: str) -> str:
    import documento_generado as _dg
    return _dg._contraer(t)


# UNA SOCIEDAD NO ES AUTORIDAD aunque su nombre diga «Administradora» o
# «Instituto»: `_rec_es_autoridad` mira palabras sueltas, y pasar «INMOBILIARIA
# ADMINISTRADORA DEL BAJÍO, S.A. DE C.V.» por el tratamiento de órgano la
# dejaba en «la Inmobiliaria … S.a. de C.v.».
_RX_SOCIEDAD = re.compile(r"(?i)\bS\.?\s?A\.?(?:\s|,|$)|\bS\.?\s?A\.?\s?P\.?\s?I\b|"
                          r"S\.?\s?de\s?R\.?\s?L\b|\bA\.?\s?C\.?(?:\s|,|$)|\bS\.?\s?C\.?(?:\s|,|$)|"
                          r"\bsociedad\b|\basociaci[óo]n\s+civil\b|\bS\.?\s?A\.?\s?B\b")


# Y SE MIRA LA CABEZA DEL NOMBRE, NO CUALQUIER PALABRA. `_rec_es_autoridad`
# busca «sala », «fiscal», «titular»… en todo el nombre, y una persona de
# apellido Sala salía «la María Sala Pérez». Un órgano EMPIEZA por su cargo o
# su clase; una persona, por su nombre de pila.
_CABEZAS_AUTORIDAD = {
    "titular", "director", "directora", "subdirector", "subdirectora", "jefe", "jefa",
    "juez", "jueza", "juzgado", "tribunal", "sala", "magistrado", "magistrada",
    "presidente", "presidenta", "presidencia", "secretario", "secretaria", "secretaria",
    "administracion", "administrador", "administradora", "delegado", "delegada",
    "delegacion", "subdelegacion", "coordinador", "coordinadora", "coordinacion", "unidad",
    "instituto", "comision", "comisionado", "procurador", "procuradora", "procuraduria",
    "fiscal", "fiscalia", "ayuntamiento", "gobernador", "gobernadora", "congreso",
    "servicio", "junta", "agente", "registrador", "registradora", "registro", "tesorero",
    "tesorera", "tesoreria", "recaudador", "recaudadora", "direccion", "gerencia",
    "gerente", "contralor", "contralora", "contraloria", "organo", "pleno", "consejo",
    "legislatura", "cabildo", "municipio", "gobierno", "poder", "actuario", "actuaria",
    "oficial", "oficialia", "subprocurador", "subprocuradora", "visitador", "visitadora",
    "encargado", "encargada", "auditor", "auditora", "auditoria", "superintendente",
    "comisaria", "comisario", "ministerio", "defensoria", "policia", "sistema",
    "organismo", "consejeria", "subsecretario", "subsecretaria", "oficina", "area",
    "departamento", "jefatura", "subjefatura", "comite", "corporacion", "autoridad",
    # Tercera ronda (3-oct-2026): cargos y órganos que faltaban.
    "camara", "senado", "subjefa", "subdelegado", "subdelegada", "comisionada",
    "consejero", "consejera", "gerenta", "inspector", "inspectora", "sindicatura",
    "sindico", "regidor", "regidora", "alcaldia", "alcalde", "alcaldesa", "gubernatura",
    "prefectura", "diputacion", "centro",
}


def _es_autoridad(nombre: str) -> bool:
    if not nombre or _RX_SOCIEDAD.search(nombre):
        return False
    n = _RX_PREFIJO_TEMPORAL.sub("", re.sub(r"^(?:el|la|los|las)\s+", "",
                                            _sin_tildes(_sin_rol(_s(nombre))[0])))
    palabras = n.split()
    return bool(palabras) and palabras[0].strip(".,;:") in _CABEZAS_AUTORIDAD


# ── LOS NOMBRES DE LAS PARTES, EN PROSA (3-oct-2026, punta a punta AD 274/2025) ──
# El V I S T O del 9274/2025 salió «promovido por GABRIEL REYES ALAMO» y el
# tercero «Ma. del Refugio…» sólo porque así lo tecleó el secretario: lo que
# viene de la carátula viene en VERSALES, y en la prosa del proyecto el nombre
# va en mayúsculas y minúsculas (los 32 engroses del banco). La carátula sigue
# en versales: eso lo escribe `documento_generado`, no este módulo.
#
# SÓLO SE TOCA LO QUE LLEGA ENTERO EN VERSALES —el secretario que escribe
# «McAllister» o «de la Garza» sabe cómo se escribe—, y dentro de las versales
# se quedan como están: las abreviaturas de letras sueltas («S.A.», «C.V.»,
# «S. de R.L.», «J.»), las siglas sin vocal o conocidas («VV», «HSBC», «IMSS»),
# los romanos de dos letras o más («XXI», «II»), lo que tiene dígitos y lo que
# va entre paréntesis. Las partículas van en minúscula («de», «del», «y», «a»)
# y el artículo sólo detrás de una de ellas: «María de la Luz», pero «Grupo La
# Moderna», donde «La» es parte de la marca.
_PARTICULAS_NOMBRE = {"de", "del", "y", "e", "a", "en", "al", "por", "con", "para",
                      "sin", "o", "u",
                      # Quinta ronda (3-oct-2026, AD 552/2024 con su tercero en
                      # versales): «Entidad No Regulada» → «Entidad no Regulada».
                      "no"}
_ARTICULOS_NOMBRE = {"el", "la", "los", "las"}
_ANTES_DE_ARTICULO = {"de", "a", "en", "por", "con", "para", "sin"}
# Las de las sociedades financieras (quinta ronda, AD 552/2024: «…, Sofom,
# Entidad No Regulada»): SOFOM, ENR, SOFIPO, SOFOL, SOCAP y SOFINCO.
_SIGLAS_SOCIEDAD = {"SA", "SAB", "CV", "SC", "AC", "SNC", "SAPI", "RL", "SRL", "SPR",
                    "IAP", "ABP", "SCL", "SOFOM", "ENR", "SOFIPO", "SOFOL", "SOCAP",
                    "SOFINCO"}
_RX_ABREV_LETRAS = re.compile(r"(?:[A-ZÁÉÍÓÚÑ]\.)+[A-ZÁÉÍÓÚÑ]?\.?")


def _palabra_de_nombre(w: str, i: int, anterior: str) -> str:
    m = re.match(r"^([(\[\"'«“]*)(.*?)([)\]\"'»”,;:]*)$", w)
    pre, nuc, post = m.groups() if m else ("", w, "")
    if not nuc or pre.startswith(("(", "[")):
        return w
    if _RX_ABREV_LETRAS.fullmatch(nuc) or any(ch.isdigit() for ch in nuc):
        return w
    base = nuc.rstrip(".")
    puntos = nuc[len(base):]
    if not base:
        return w
    sin = _sin_tildes(base)
    if len(base) >= 2 and ((base in _SIGLAS and base not in _SIGLAS_QUE_SON_PALABRA) or
                           base in _SIGLAS_SOCIEDAD or
                           (base.isalpha() and not re.search(r"[aeiouy]", sin)) or
                           _RX_ROMANO.match(base)):
        return w
    low = base.lower()
    if i and low in _PARTICULAS_NOMBRE:
        nuevo = low
    elif i and low in _ARTICULOS_NOMBRE and anterior in _ANTES_DE_ARTICULO:
        nuevo = low
    elif len(base) == 1:
        nuevo = base
    else:
        # «GARCÍA-LÓPEZ» → «García-López»; «D'ANGELO» → «D'Angelo».
        nuevo = re.sub(r"[^\W\d_]+", lambda x: x.group(0)[:1] + x.group(0)[1:].lower(), base)
    return pre + nuevo + puntos + post


def _prosa_nombre(nombre: str) -> str:
    """«GABRIEL REYES ALAMO» → «Gabriel Reyes Alamo»; «MA. DEL REFUGIO TREJO
    RAMOS» → «Ma. del Refugio Trejo Ramos»; «IMPULSORA … VV, S.A. DE C.V.» →
    «Impulsora … VV, S.A. de C.V.».

    CUARTA RONDA (3-oct-2026, E1; AD 335/2025, AR 208 y 60/2025): lo que viene
    MEZCLADO también se toca, pero sólo en sus rachas en versales. «Sucesión a
    Bienes de JOSÉ GARCÍA RUIZ» (la forma del auto y de la ficha del AD 335) y
    «JUAN y PEDRO, ambos de apellidos PÉREZ LÓPEZ» (AR 208) salían en
    mayúsculas en el V I S T O, los resultandos y el resolutivo, porque sólo
    se convertía el nombre que llegaba ENTERO en versales. Lo que el secretario
    escribió en mayúsculas y minúsculas («McAllister», «de la Garza») sigue sin
    tocarse."""
    t = _s(nombre)
    # EN VERSALES = toda palabra con letras va en mayúsculas, salvo las
    # partículas que el secretario teclea en minúscula entre dos nombres en
    # versales («ANA LÓPEZ RUIZ y JUAN PÉREZ GÓMEZ», RF 6/2026).
    palabras = [w for w in t.split() if any(c.isalpha() for c in w)]
    if not palabras or not any(c.isupper() for c in t):
        return t
    mezclado = any(any(c.islower() for c in w) and
                   w.strip(",;:.").lower() not in (_PARTICULAS_NOMBRE | _ARTICULOS_NOMBRE)
                   for w in palabras)
    if mezclado:
        fuera = _prosa_rachas(t)
    else:
        fuera, anterior = [], ""
        for i, w in enumerate(t.split(" ")):
            nuevo = _palabra_de_nombre(w, i, anterior)
            fuera.append(nuevo)
            anterior = _sin_tildes(nuevo.strip(",;:.()«»\"'"))
        fuera = " ".join(fuera)
    # LAS FÓRMULAS QUE VIAJAN DENTRO DEL NOMBRE van en minúscula (Q 342/2025:
    # «SUCESIÓN … A TRAVÉS DE SU ALBACEA …» salía «a Través de Su Albacea»;
    # AD 552/2024: «por Conducto de Su Consejo de Administración»), y con la
    # coma que las separa del nombre si el papel no la traía (quinta ronda).
    fuera = _formulas_en_minuscula(fuera)
    # «S.A. DE C.V.» dentro de un nombre mezclado: la partícula entre dos
    # abreviaturas va en minúscula, como en la razón social entera.
    return re.sub(r"(?<=\.)\s+(DE|DEL)\s+(?=[A-ZÁÉÍÓÚÑ]\.)",
                  lambda m: f" {m.group(1).lower()} ", fuera)


def _es_versal(w: str) -> bool:
    """¿La palabra tiene letras y todas van en mayúsculas?"""
    letras = [c for c in w if c.isalpha()]
    return bool(letras) and all(c.isupper() for c in letras)


def _palabra_de_nombre_en_versales(w: str) -> bool:
    """¿Una palabra de NOMBRE en versales («JOSÉ», «GARCÍA»)? No lo son una
    sigla («IMSS», «VV»), un romano («XXI»), una abreviatura («S.A.», «J.»), lo
    que lleva dígitos, lo que va entre paréntesis ni las palabras de menos de
    tres letras («DE», «LA», «Y»)."""
    m = re.match(r"^([(\[\"'«“]*)(.*?)([)\]\"'»”,;:]*)$", w)
    pre, nuc, _post = m.groups() if m else ("", w, "")
    if not nuc or pre.startswith(("(", "[")) or any(ch.isdigit() for ch in nuc):
        return False
    if _RX_ABREV_LETRAS.fullmatch(nuc):
        return False
    base = nuc.rstrip(".")
    if len(base) < 3 or not _es_versal(base) or not re.fullmatch(r"[^\W\d_]+(?:[-'][^\W\d_]+)*", base):
        return False
    if base in _SIGLAS or base in _SIGLAS_SOCIEDAD or _RX_ROMANO.match(base):
        return False
    return bool(re.search(r"[aeiouy]", _sin_tildes(base)))


def _prosa_rachas(t: str) -> str:
    """Las RACHAS en versales de un texto mezclado, en mayúsculas y minúsculas.

    Una racha es una serie de palabras en versales —con las partículas en
    minúscula que el secretario teclea entre ellas («JUAN y PEDRO»)— que trae
    al menos DOS palabras de nombre (`_palabra_de_nombre_en_versales`). Una
    sola palabra en mayúsculas dentro de un texto mezclado se queda: casi
    siempre es una sigla («del ISSSTE», «UNAM»). Dentro de la racha valen las
    reglas de siempre: siglas, romanos, abreviaturas («S.A.», «Ma.») y «V V»
    se quedan; las partículas, en minúscula."""
    toks = t.split(" ")
    out = list(toks)
    particulas = _PARTICULAS_NOMBRE | _ARTICULOS_NOMBRE
    i, n = 0, len(toks)
    while i < n:
        if not _es_versal(toks[i]):
            i += 1
            continue
        k = ultimo = i
        while k + 1 < n:
            sig = toks[k + 1]
            if _es_versal(sig):
                k += 1
                ultimo = k
            elif sig == sig.lower() and sig.strip(",;:.") in particulas:
                k += 1
            else:
                break
        if sum(_palabra_de_nombre_en_versales(toks[x]) for x in range(i, ultimo + 1)) >= 2:
            anterior = _sin_tildes(toks[i - 1].strip(",;:.()«»\"'")) if i else ""
            for x in range(i, ultimo + 1):
                if _es_versal(toks[x]):
                    out[x] = _palabra_de_nombre(toks[x], x, anterior)
                anterior = _sin_tildes(out[x].strip(",;:.()«»\"'"))
        i = ultimo + 1
    return " ".join(out)


# LAS FIGURAS DE REPRESENTACIÓN QUE VIAJAN DENTRO DEL NOMBRE (cuarta ronda,
# 3-oct-2026, E1): «por conducto de su consejo de administración» (AD
# 552/2024), «a través de su albacea» (Q 342/2025), «por conducto de su
# apoderado legal». La lista es cerrada para no bajar a minúscula el nombre de
# la persona que sigue a la figura («…, POR CONDUCTO DE SU APODERADO LEGAL
# JUAN RUIZ» → «…, por conducto de su apoderado legal Juan Ruiz»).
#
# QUINTA RONDA (3-oct-2026, Q 229/2026 y las seis pruebas de verif_q2): faltaban
# la madre, el padre y el progenitor («MENOR DE EDAD DE INICIALES J.L.M., POR
# CONDUCTO DE SU MADRE MARÍA LÓPEZ» salía «por Conducto de Su Madre»), el
# albacea «de la sucesión a bienes de» y el heredero único.
_SUCESION_A_BIENES = (r"de\s+la\s+sucesi[oó]n(?:\s+(?:testamentaria|intestamentaria|"
                      r"leg[íi]tima|intestada))?\s+a\s+bienes\s+de")
_FIGURA_EN_EL_NOMBRE = (
    r"(?:albaceas?(?:\s+(?:provisional(?:es)?|definitiv[oa]s?))?(?:\s+" + _SUCESION_A_BIENES + r")?|"
    r"apoderad[oa]s?(?:\s+(?:legal(?:es)?|general(?:es)?|especial(?:es)?))?"
    r"(?:\s+para\s+pleitos\s+y\s+cobranzas)?|"
    r"representantes?(?:\s+(?:legal(?:es)?|com[uú]n))?|"
    r"autorizad[oa]s?(?:\s+en\s+t[eé]rminos\s+amplios)?|"
    r"(?:presidente\s+del\s+)?consejo\s+de\s+administraci[oó]n|"
    r"administrador(?:a)?(?:\s+[uú]nic[oa]|\s+general)?|"
    r"president[ea]|gerente(?:\s+general)?|director(?:a)?\s+general|"
    r"[oó]rgano\s+de\s+representaci[oó]n|mandatari[oa]s?|liquidador(?:a)?|"
    r"s[ií]ndic[oa]|tutor(?:a)?|curador(?:a)?|comisariado\s+ejidal|"
    r"delegad[oa]s?|abogad[oa]s?\s+patron[oa]s?|"
    r"madre|padre|progenitor(?:a|es|as)?|abuel[oa]s?|"
    r"herede(?:ro|ra)s?(?:\s+[uú]nic[oa]s?)?)")
# Tras «en su carácter de» sólo las figuras de REPRESENTACIÓN PRIVADA: un cargo
# público («en su carácter de Presidente Municipal de…») conserva su mayúscula,
# y bajar sólo «presidente» dejaba «presidente Municipal».
_FIGURA_TRAS_CARACTER = (
    r"(?:albaceas?(?:\s+(?:provisional(?:es)?|definitiv[oa]s?))?(?:\s+" + _SUCESION_A_BIENES + r")?|"
    r"apoderad[oa]s?(?:\s+(?:legal(?:es)?|general(?:es)?|especial(?:es)?))?"
    r"(?:\s+para\s+pleitos\s+y\s+cobranzas)?|"
    r"representantes?(?:\s+(?:legal(?:es)?|com[uú]n))?|"
    r"autorizad[oa]s?(?:\s+en\s+t[eé]rminos\s+amplios)?|"
    r"(?:presidente\s+del\s+)?consejo\s+de\s+administraci[oó]n|"
    r"administrador(?:a)?\s+[uú]nic[oa]|mandatari[oa]s?|liquidador(?:a)?|"
    r"tutor(?:a)?|curador(?:a)?|madre|padre|progenitor(?:a|es|as)?|abuel[oa]s?|"
    r"herede(?:ro|ra)s?(?:\s+[uú]nic[oa]s?)?)")

# QUINTA RONDA (3-oct-2026), además de las figuras: el cargo que sigue a «en
# su carácter de» («en su carácter de Albacea de la Sucesión a Bienes de»
# salía con mayúsculas), «apoderada de», «albaceas de la sucesión a bienes
# de», «heredera única», «quien también se ostenta como», «y/o», «ambos por
# propio derecho» (AD 274 en plural: «…María López Díaz, Ambos por propio
# derecho promovieron», sin la coma de cierre porque el inciso empezaba en
# mayúscula) y «menor de edad de iniciales» (Q con la madre que promueve).
_RX_FRASES_DEL_NOMBRE = re.compile(
    r"(?i)\b(?:(?:amb[oa]s|tod[oa]s)\s+por\s+(?:su\s+)?propio\s+derecho|"
    r"por\s+(?:su\s+)?propio\s+derecho|por\s+s[ií](?=[\s,])|"
    r"(?:a\s+trav[eé]s|por\s+conducto)\s+de\s+sus?\s+" + _FIGURA_EN_EL_NOMBRE + r"|"
    r"(?:del|en\s+t[eé]rminos\s+(?:amplios\s+)?del)\s+art[íi]culo(?=\s+\d)|"
    r"en\s+(?:representaci[oó]n|nombre)\s+(?:del|de)(?:\s+(?:la|el|los|las|sus?))?"
    r"(?:\s+(?:menor(?:es)?|hij[oa]s?|niet[oa]s?|niñ[oa]s?|adolescentes?))?"
    r"(?:\s+(?:menor(?:es)?|hij[oa]s?|niet[oa]s?))?(?:\s+de\s+edad)?(?:\s+de\s+iniciales)?|"
    r"en\s+su\s+(?:car[aá]cter|calidad)\s+de(?:\s+" + _FIGURA_TRAS_CARACTER + r"(?![\w/]))?|"
    r"albaceas?(?:\s+(?:provisional(?:es)?|definitiv[oa]s?))?\s+" + _SUCESION_A_BIENES + r"|"
    r"herede(?:ro|ra)s?\s+[uú]nic[oa]s?|"
    r"apoderad[oa]s?(?:\s+(?:legal(?:es)?|general(?:es)?|especial(?:es)?))?\s+(?:de|del)(?=\s)|"
    r"quien(?:es)?\s+(?:tambi[eé]n\s+)?se\s+ostentan?(?:\s+como)?|"
    r"y/o|"
    r"(?:(?:el|la|los|las)\s+)?(?:menor(?:es)?\s+de\s+edad(?:\s+de\s+iniciales)?|"
    r"(?:menor(?:es)?|niñ[oa]s?|adolescentes?)\s+de\s+iniciales)|"
    r"amb[oa]s\s+de\s+apellidos?|tod[oa]s\s+de\s+apellidos?|"
    r"y\s+otr[oa]s?|y\s+coagraviad[oa]s|y\s+codemandad[oa]s)(?![\w/])")

# LA COMA QUE FALTA ANTES DE LA FÓRMULA DE REPRESENTACIÓN (quinta ronda,
# 3-oct-2026, AD 335/2025): la carátula escribe «SUCESIÓN A BIENES DE X A
# TRAVÉS DE SU ALBACEA Y», sin coma, y salía «…a través de su albacea Y
# promovió juicio» —sin la coma de cierre— y en la legitimación «…su albacea
# Y, quien está legitimada», que atribuye la legitimación al albacea. Con la
# coma, el inciso se reconoce y se cierra. No se pone tras «y», «ambos» ni
# tras otra coma.
_RX_ANTES_DE_FORMULA = re.compile(
    r"(?i)\s+(?=(?:a\s+trav[eé]s|por\s+conducto)\s+de\s+sus?\s+" + _FIGURA_EN_EL_NOMBRE +
    r"|por\s+(?:su\s+)?propio\s+derecho\b|en\s+su\s+(?:car[aá]cter|calidad)\s+de\b|"
    r"quien(?:es)?\s+(?:tambi[eé]n\s+)?se\s+ostentan?\b)")
_NO_COMA_TRAS = {"y", "e", "o", "u", "ambos", "ambas", "todos", "todas", "de", "del", "su", "sus",
                 "a", "como", "y/o"}


def _coma_ante_formula(t: str) -> str:
    """«Sucesión a Bienes de X a través de su albacea Y» → «…de X, a través de
    su albacea Y»; «Juan Pérez por propio derecho» → «Juan Pérez, por propio
    derecho». Lo que ya trae su coma no se toca."""
    def _pon(m):
        antes = t[:m.start()]
        if not antes.strip() or antes.rstrip()[-1:] in (",", ";", "(", ":"):
            return m.group(0)
        if _sin_tildes(antes.split()[-1]).strip(".") in _NO_COMA_TRAS:
            return m.group(0)
        return ", "
    return _RX_ANTES_DE_FORMULA.sub(_pon, t or "")


def _formulas_en_minuscula(t: str) -> str:
    """Las fórmulas de `_RX_FRASES_DEL_NOMBRE` en minúscula, con su coma."""
    return _RX_FRASES_DEL_NOMBRE.sub(lambda m: m.group(0).lower(), _coma_ante_formula(t))


# ── LO QUE LA CARÁTULA AÑADE AL NOMBRE (cuarta ronda, 3-oct-2026, RF 7/2025) ──
# «1) PARTE ACTORA y otros (ya mencionados con anterioridad)» pasaba tal cual
# al resultando: la numeración y la remisión son de la lista de la carátula y
# en la prosa no remiten a nada. Se quitan; si el nombre queda incompleto
# («y otros»), quien compone lo avisa (`_aviso_abreviado`).
_RX_NUMERACION = re.compile(r"(?:^|(?<=[\s,;]))\(?\d{1,3}\s?(?:\)|\.-)\s*")
_RX_COLETILLA = re.compile(
    r"(?i)\s*\(\s*(?:ya|antes|arriba|previamente|anteriormente)\s+(?:mencionad|citad|"
    r"señalad|senalad|precisad|referid|identificad|nombrad|indicad)[^)]*\)")
_RX_Y_OTROS = re.compile(r"(?i)\by\s+(?:otr[oa]s?|coagraviad[oa]s|codemandad[oa]s)\b")


def _sin_caratula(nombre: str) -> tuple:
    """(nombre sin numeración ni coletilla de la carátula, ¿venía abreviado?).
    Abreviado = traía numeración, remisión o «y otros»: la carátula resume y
    los nombres completos están en otra parte."""
    t = _s(nombre)
    sin = _RX_COLETILLA.sub("", t)
    sin = _RX_NUMERACION.sub("", sin)
    sin = _s(sin)
    return sin, (sin != t or bool(_RX_Y_OTROS.search(sin)))


# LAS ENTIDADES COLECTIVAS QUE NO SON SOCIEDADES llevan artículo (3-oct-2026,
# AD 335/2025 y AR 60/2025): salía «no ampara ni protege a Sucesión a Bienes
# de…», «promovido por Sucesión…» en cuatro secciones; el engrose dice «la
# Sucesión a Bienes de». Una persona física no lleva artículo; una sucesión,
# una comunidad o un ejido sí. La asociación, con «la», como en
# `tipos_asunto.con_articulo_de_colectivo` (cuarta ronda: una sola regla).
_ARTICULO_COLECTIVO = {"sucesion": "la", "comunidad": "la", "ejido": "el", "nucleo": "el",
                       "comisariado": "el", "asociacion": "la"}


def _articulo_colectivo(n: str) -> str:
    """«la Sucesión a Bienes de…», «el Ejido San Juan». Si el nombre ya trae
    su artículo («LA SUCESIÓN…» pasado a prosa), se deja en minúscula."""
    m = re.match(r"(?i)^(el|la|los|las)\s+(\S+)", n)
    if m and _sin_tildes(m.group(2)).strip(".,") in _ARTICULO_COLECTIVO:
        return m.group(1).lower() + n[len(m.group(1)):]
    pal = _sin_tildes(n.split(" ", 1)[0]).strip(".,;:") if n else ""
    art = _ARTICULO_COLECTIVO.get(pal)
    return f"{art} {n}" if art else n


def _nombre(nombre: str, autoridad: bool = False, con_articulo: bool = True) -> str:
    """El nombre de una parte en la prosa (ver `nombre_en_prosa`). Puede
    lanzar: dentro del compositor lo recoge su red de seguridad."""
    n = _sin_rol(_s(nombre))[0]
    n = _sin_rol(_sin_caratula(n)[0])[0]
    if not n:
        return ""
    # EL CAMINO DE ÓRGANO TAMBIÉN BAJA LAS FÓRMULAS (quinta ronda, 3-oct-2026,
    # Q con un organismo quejoso): «INSTITUTO MEXICANO DEL SEGURO SOCIAL, POR
    # CONDUCTO DE SU APODERADO LEGAL JUAN RUIZ» iba por `_organo` y salía «…,
    # Por Conducto de Su Apoderado Legal Juan Ruiz». Y el artículo del papel se
    # conserva en los cargos de doble género (F5, «la Oficial Mayor…»).
    if autoridad:
        o = _formulas_en_minuscula(_organo(n))
        return _con_art(o, art_papel=_articulo_del_papel(n))
    if _es_autoridad(n):
        o = _formulas_en_minuscula(_organo(n))
        return _con_art(o, art_papel=_articulo_del_papel(n)) if con_articulo else o
    p = _prosa_nombre(n)
    return _articulo_colectivo(p) if con_articulo else p


def _parte(nombre: str, autoridad: bool = False) -> str:
    """Una PARTE en la prosa. La autoridad, como órgano y con su artículo: «el
    Director de Ingresos…» («la Titular…» si el papel lo dice así). La persona
    o la sociedad, con su nombre en mayúsculas y minúsculas si llegó en
    versales (TERCERA RONDA, 3-oct-2026: antes se copiaba tal cual y el
    V I S T O decía «promovido por GABRIEL REYES ALAMO»); la sucesión, la
    comunidad y el ejido, con su artículo. Sin la etiqueta de rol de la
    carátula («(AUTORIDAD RESPONSABLE)», «(DEMANDADA)») ni su numeración o su
    remisión («1) …», «(ya mencionados…)», cuarta ronda)."""
    return _nombre(nombre, autoridad=autoridad, con_articulo=True)


# ── EL NÚMERO GRAMATICAL DEL SUJETO (3-oct-2026, AD 552/2024, AR 208/2025, RF 7/2025) ──
# Dos quejosos en un solo campo salían con el verbo en singular: «X, por
# conducto de…, y Z, por propio derecho promovió», «X y Z, ambos de apellidos
# … promovió», «A, B, … y C demandó la nulidad». El plural se reconoce por lo
# que lo dice sin duda: «ambos/todos», el punto y coma de la lista, «, y» tras
# un inciso, una lista con comas cerrada por «y Nombre», o dos nombres de dos
# palabras o más unidos por «y». Una sociedad («Aeropuertos y Servicios
# Auxiliares, S.A. de C.V.») o un apellido compuesto («Ortega y Gasset») no.
_RX_PLURAL_EXPRESO = re.compile(r"(?i)\b(?:ambos|ambas|todos|todas|los\s+dos|las\s+dos)\b")
_RX_PLURAL_COMA_Y = re.compile(r",\s+(?:y|e)\s+\S")
_RX_Y_NOMBRE = re.compile(r"\s(?:y|e)\s+[A-ZÁÉÍÓÚÑ]")
# UNA ORGANIZACIÓN ES UNA, aunque su nombre enumere: «Unión de Trabajadores de
# la Construcción, Transportistas, Materialistas y Similares y Anexos del Estado
# de Querétaro, C.T.M.» (AR 631/2025, punta a punta) salía «promovieron».
_CABEZAS_COLECTIVAS = {
    "union", "sindicato", "asociacion", "confederacion", "federacion", "grupo", "banco",
    "comercializadora", "inmobiliaria", "constructora", "productos", "servicios",
    "industrias", "colegio", "fundacion", "sociedad", "cooperativa", "compania", "empresa",
    "club", "consorcio", "corporativo", "desarrolladora", "promotora", "transportes",
    "distribuidora", "universidad", "patronato", "camara", "frente", "coalicion", "liga",
    "central", "alianza", "agrupacion", "organizacion", "red", "consejo", "comite",
}


def _lista_de_nombres(n: str) -> bool:
    """«Ana López Ruiz, Juan Pérez Gómez y Luis Díaz Mora»: cada elemento de la
    lista es un nombre de dos palabras o más que empieza en mayúscula. «Unión de
    Trabajadores…, Transportistas, Materialistas y Similares…» no lo es."""
    if not _RX_Y_NOMBRE.search(n):
        return False
    # Lo testado de la versión pública («** PARTE ACTORA») no cuenta.
    partes = [p.strip(" *") for p in re.split(r",\s*|\s+(?:y|e)\s+", n) if p.strip(" *")]
    return len(partes) >= 2 and all(len(p.split()) >= 2 and p[:1].isupper() for p in partes)


def _es_plural(nombre: str) -> bool:
    """¿El campo nombra a VARIAS partes?

    UN SOLO DETECTOR (cuarta ronda, 3-oct-2026, E2; AD 552/2024): el resultando
    decía «promovieron» con el detector de aquí y la legitimación y la carátula
    «quien está legitimada» y «QUEJOSA:» con el de `tipos_asunto`; el mismo
    documento se contradecía. Decide `tipos_asunto.es_plural_de_partes`, sobre
    el nombre ya en prosa (sus reglas leen la «y» en minúscula), y lo decidido
    viaja en `datos_extra["plural"]` para que la legitimación y la carátula no
    vuelvan a decidirlo. El detector local sólo responde si aquél no se puede
    llamar. Una autoridad nunca es plural."""
    n = _prosa_nombre(_sin_rol(_s(nombre))[0])
    if not n or _es_autoridad(n):
        return False
    try:
        import tipos_asunto as _ta
        _f = getattr(_ta, "es_plural_de_partes", None)
        if callable(_f):
            return bool(_f(n))
    except Exception:
        pass
    return _es_plural_local(n)


_RX_NUMERADO = re.compile(r"(?:^|[\s,;])\(?\d{1,3}\s?(?:\)|\.-)")


def _es_plural_local(n: str) -> bool:
    """El detector de respaldo, sobre el nombre ya en prosa."""
    if _RX_PLURAL_EXPRESO.search(n) or ";" in n or _RX_PLURAL_COMA_Y.search(n):
        return True
    # «y otros», «y coagraviados» (RF 7/2025) y la lista numerada de la carátula.
    if _RX_Y_OTROS.search(n) or len(_RX_NUMERADO.findall(n)) >= 2:
        return True
    # LO QUE SIGUE YA NO ES EXPRESO: una sociedad, una abreviatura final
    # («C.T.M.», «S.A.») o la cabeza de una organización dicen que es UNA.
    cabeza = _sin_tildes(n.split(" ", 1)[0]).strip(".,;:")
    if _RX_SOCIEDAD.search(n) or _RX_ABREVIATURA_FINAL.search(n) or \
            cabeza in _CABEZAS_COLECTIVAS or cabeza in _ARTICULO_COLECTIVO:
        return False
    # «…, por propio derecho y en representación de su hijo» NO es plural: tras
    # la «y» no viene un nombre (minúscula). Dos nombres de dos palabras o más
    # unidos por «y» sí; «Juan Ortega y Gasset», no.
    return _lista_de_nombres(n)


_RX_NUEVO_NOMBRE = re.compile(r"(?:y|e)\s+[A-ZÁÉÍÓÚÑ*]")


def _cierra_inciso(sujeto: str) -> str:
    """La coma que CIERRA el inciso antes del verbo (3-oct-2026, AD 552/2024):
    «…y Z, por propio derecho promovió» → «…y Z, por propio derecho,
    promovieron». Hay inciso si un tramo tras una coma empieza en minúscula
    («por propio derecho», «ambos de apellidos…») o es una fórmula del nombre;
    «Banco del Centro, S.A.» no lo tiene y no lleva coma ante el verbo.

    QUINTA RONDA (3-oct-2026): no sólo el ÚLTIMO tramo. «X, por sí y como
    representante de la moral Y, S.A. de C.V.» (AR 222 con nombres de prueba)
    acaba en la razón social, que empieza en mayúscula, y el inciso seguía
    abierto: «…, S.A. de C.V. promovió». Manda el ÚLTIMO tramo que abre inciso,
    salvo que después venga otro nombre de la lista («…, y María López»). El
    punto y coma también separa («…; por propio derecho», RF 2/2025), y una
    fórmula en mayúscula («Ambos por propio derecho») también es inciso."""
    t = (sujeto or "").rstrip()
    if not t or t.endswith(","):
        return t
    tramos = re.split(r"[,;]\s+", t)
    abierto = False
    for tr in tramos[1:]:
        if _RX_NUEVO_NOMBRE.match(tr):
            abierto = False             # «…, y Parte Actora»: otro nombre, no un inciso
        elif (tr[:1].islower() or _RX_FRASES_DEL_NOMBRE.match(tr)):
            abierto = True
    return t + "," if abierto else t


def nombre_en_prosa(nombre: str, autoridad: bool = False, con_articulo: bool = False) -> str:
    """EL ÚNICO CONVERTIDOR DE NOMBRES A PROSA (cuarta ronda, 3-oct-2026, E1).

    Lo llaman el V I S T O y los resultandos (por `_parte`) y, con import local,
    la legitimación, los resolutivos y los efectos (`documento_generado.
    _en_prosa`, `tipos_asunto.con_articulo_de_organo`): el mismo nombre se
    escribe igual en todo el documento. Antes había dos convertidores y el
    mismo proyecto decía «a través de su albacea» en el V I S T O y «a Través
    de Su Albacea» en la legitimación (Q 342/2025), «Agrícola Los Pinos» y
    «Agrícola los Pinos», «la Sucesión» y «Sucesión» (AD 335/2025).

      · sin la etiqueta de rol, la numeración ni la remisión de la carátula
        («(ACTORA)», «1) …», «(ya mencionados con anterioridad)»);
      · las versales, en mayúsculas y minúsculas, también las rachas de un
        texto mezclado («Sucesión a Bienes de JOSÉ GARCÍA RUIZ»); siglas,
        romanos, «S.A. de C.V.», «C.T.M.», «V V» y «Ma.» se respetan;
      · las fórmulas de representación en minúscula («por conducto de su
        consejo de administración», «a través de su albacea», «por propio
        derecho y en representación de», «ambos de apellidos», «y otros»).

    `autoridad=True`: el órgano en prosa y SIEMPRE con su artículo («el
    Director de Ingresos…», «la Titular…» si el papel lo dice así) —es lo que
    pide `con_articulo_de_organo`—. Sin ella, la persona, la sociedad o el
    órgano reconocido por su cabeza van sin artículo, salvo `con_articulo=True`,
    que pone el de los colectivos («la Sucesión…», «la Comunidad…», «el
    Ejido…») y el del órgano. Nunca lanza; lo que no es texto (un número, un
    dict) no es un nombre y vale «»."""
    if not isinstance(nombre, str):
        return ""
    try:
        return _nombre(nombre, autoridad=autoridad, con_articulo=con_articulo)
    except Exception:
        return _s(nombre)


def _recorte(x, tope: int = 200) -> str:
    """La cita de un valor en un aviso, recortada: con 66 nombres el aviso del
    nombre abreviado ocupaba miles de caracteres (RF 7/2025)."""
    t = _s(x)
    return t if len(t) <= tope else t[:tope].rstrip(" ,;") + "…"


def _aviso_abreviado(crudo, clave: str, av: "_Avisos",
                     donde: str = "el proemio de la sentencia recurrida") -> None:
    """Si el nombre venía resumido por la carátula («1) PARTE ACTORA y otros
    (ya mencionados con anterioridad)», RF 7/2025), se dice: la prosa ya no
    remite a la lista y el nombre queda incompleto.

    SÓLO CON UNA REMISIÓN DE VERDAD (quinta ronda, 3-oct-2026, RF 7/2025): la
    numeración sola de una lista COMPLETA («1) A, 2) B y 3) C») no abrevia
    nada, y el aviso decía que el nombre estaba abreviado cuando no lo estaba.
    Abrevian «y otros», «y coagraviados», «y codemandados» y la coletilla «(ya
    mencionados…)». La cita va recortada a 200 caracteres."""
    t = _sin_rol(_s(crudo))[0]
    if not (_RX_COLETILLA.search(t) or _RX_Y_OTROS.search(t)):
        return
    av.add(f"EL NOMBRE VIENE ABREVIADO DE LA CARÁTULA ({clave} = «{_recorte(crudo)}»): se "
           f"escribió «{_recorte(nombre_en_prosa(crudo))}», sin la numeración ni la remisión de "
           f"la carátula. Si son varias personas, sus nombres están en {donde}; escríbelos en la "
           f"ficha si deben ir completos.")


def _y(lista) -> str:
    """«A», «A y B», «A, B y C». Si algún nombre trae coma («S.A. de C.V.»),
    se separa con punto y coma para que la lista se lea sin ambigüedad.

    LA «Y» FINAL TRAS UN ELEMENTO CON COMA LLEVA SU SIGNO (quinta ronda,
    3-oct-2026, AD 128/2025): «el Oficial Mayor del Municipio de Cadereyta de
    Montes, Querétaro y la Coordinadora…» se lee como si Querétaro fuera otra
    parte. Si un elemento ANTES de la «y» trae coma: «A, y B» con dos y «A; B;
    y C» con más. La coma del último («y Metlife México, S.A. de C.V.») no
    confunde y no cambia nada, ni la de una razón social que acaba en su
    abreviatura («Metlife México, S.A. de C.V. y Unión…»)."""
    xs = [x for x in lista if x]
    if not xs:
        return ""
    if len(xs) == 1:
        return xs[0]
    sep = "; " if any("," in x for x in xs) else ", "
    # Una razón social que cierra en su abreviatura («…, S.A. de C.V.») ya marca
    # dónde acaba: sólo confunde el inciso que acaba en una palabra.
    antes_con_coma = any("," in x and not _RX_ABREVIATURA_FINAL.search(x) for x in xs[:-1])
    if not antes_con_coma:
        return sep.join(xs[:-1]) + " y " + xs[-1]
    final = ", y " if len(xs) == 2 else "; y "
    return sep.join(xs[:-1]) + final + xs[-1]


_RX_NUM = re.compile(r"(\d{1,6})\s*[/-]\s*((?:19|20)\d{2})\b")


def _numero_asunto(x) -> str:
    """El número del propio asunto, «N/AAAA». LA ETIQUETA DEL ENCABEZADO NO
    ENTRA: «ADC 625-2024 ORAL MERCANTIL» se coló en 27 proyectos (errores.txt)
    como si fuera el número; aquí sale «625/2024»."""
    t = _s(x)
    m = _RX_NUM.search(t)
    return f"{m.group(1)}/{m.group(2)}" if m else t


def _rotulado(valor: str, rotulos: tuple, rotulo: str) -> str:
    """«toca civil 374/2024» se queda; «374/2024» se rotula «toca 374/2024». Sin
    esto salía «el toca toca civil…» cuando la fuente ya traía el rótulo."""
    v = _s(valor)
    if not v:
        return ""
    return v if _sin_tildes(v).startswith(rotulos) else f"{rotulo} {v}"


_RX_NUM_JUICIO = re.compile(r"\d{1,6}\s*/\s*\d{2,4}(?:[-/][\w.]+)*")


def _solo_numero(x) -> str:
    """«juicio de amparo indirecto 950/2024» → «950/2024»: la fórmula ya dice
    «juicio de amparo indirecto» y no se repite."""
    v = _s(x)
    if re.match(r"(?i)^\s*(?:juicio|amparo|expediente|n[úu]mero)", v):
        m = _RX_NUM_JUICIO.search(v)
        if m:
            return m.group(0).replace(" ", "")
    return v


# ── LA MATERIA, CONCORDADA CON EL SUSTANTIVO ───────────────────────────────
# «amparo directo civil / administrativo», «queja civil / administrativa»
# (oro_Q: la materia concuerda con «queja» en 485 de 564). Lo mercantil y lo
# familiar son civil; lo agrario va como administrativo, igual que su inciso
# b) en la competencia (oro_AD, SUBTIPOS).
_MATERIA_FORMA = {
    "civil": ("civil", "civil"), "mercantil": ("civil", "civil"),
    "familiar": ("civil", "civil"),
    "administrativa": ("administrativo", "administrativa"),
    "agraria": ("administrativo", "administrativa"),
    "penal": ("penal", "penal"), "laboral": ("laboral", "laboral"),
}


def _materia_clave(x) -> str:
    """La clave de la materia, o «» si no se puede afirmar. «Materias
    administrativa y civil» es la especialidad DEL TRIBUNAL, no la del asunto
    (errores.txt: «amparo directo en materias administrativa y civil 93/2026»),
    y no se elige una de las dos al azar."""
    t = _sin_tildes(_s(x))
    if not t:
        return ""
    hallas = []
    for clave, pats in (("administrativa", ("administrativ", "fiscal")),
                        ("agraria", ("agrari",)), ("civil", ("civil",)),
                        ("familiar", ("familiar",)), ("mercantil", ("mercantil",)),
                        ("penal", ("penal",)), ("laboral", ("laboral", "trabajo"))):
        if any(p in t for p in pats):
            hallas.append(clave)
    if len(hallas) == 1:
        return hallas[0]
    # «civil» y «mercantil» o «familiar» juntas son la misma materia.
    if hallas and set(hallas) <= {"civil", "mercantil", "familiar"}:
        return "civil"
    return ""


def _materia_concordada(clave: str, genero: str) -> str:
    par = _MATERIA_FORMA.get(clave or "")
    if not par:
        return ""
    return par[0] if genero == "m" else par[1]


# ═══════════════════════════════════════════════════════════════════════════
# LOS AVISOS: el dato, dónde está y dónde quedó el hueco
# ═══════════════════════════════════════════════════════════════════════════
class _Avisos:
    def __init__(self):
        self.lista: list = []

    def add(self, texto: str):
        if texto and texto not in self.lista:
            self.lista.append(texto)

    def falta(self, que: str, clave: str, donde: str, apartado, tambien: str = "") -> str:
        """Registra el aviso y devuelve el hueco, para escribirlo en su sitio.

        `apartado` puede ser una lista: el aviso nombra TODOS los sitios donde
        queda el hueco (cuarta ronda, 3-oct-2026, Q 335/2025 y AR 307/2024: el
        aviso decía «Va en hueco en «V I S T O»» y el hueco también estaba en
        la interposición, la competencia y la procedencia). `tambien` nombra
        los considerandos que lo heredan («la competencia y la existencia»)."""
        verbo = "FALTAN" if re.match(r"(?i)^(?:los|las)\s", que) else "FALTA"
        sitios = [apartado] if isinstance(apartado, str) else [a for a in apartado if a]
        en = _y([f"«{a}»" for a in sitios])
        cola = f", y también en {tambien}" if tambien else ""
        self.add(f"{verbo} {que.upper()} ({clave}): {donde}. Va en hueco en {en}{cola}.")
        return HUECO


def _o(valor, av: _Avisos, que, clave, donde, apartado) -> str:
    v = _s(valor)
    return v if v else av.falta(que, clave, donde, apartado)


def _fecha_o(valor, av: _Avisos, que, clave, donde, apartado, respaldo="") -> str:
    """La fecha en letra; o lo que ya venía en letra del encargo; o el hueco
    con su aviso. Una fecha ilegible se avisa como ilegible, no como ausente."""
    f = _fecha(valor)
    if f is not None:
        return _letra(f)
    r = _s(respaldo)
    if r and HUECO not in r:
        return r
    if _s(valor):
        av.add(f"FECHA ILEGIBLE EN LA FICHA ({clave} = «{_s(valor)}»): se esperaba "
               f"AAAA-MM-DD. Va en hueco en «{apartado}».")
        return HUECO
    return av.falta(que, clave, donde, apartado)


# ═══════════════════════════════════════════════════════════════════════════
# LO COMÚN A LOS CUATRO TIPOS
# ═══════════════════════════════════════════════════════════════════════════
# LOS TRATAMIENTOS DEL PONENTE (cuarta ronda, 3-oct-2026, E3; AR 239/2025): «se
# turnaron los autos a la ponencia de licenciada Bertha Martínez Vega» salía sin
# artículo, y `documento_generado._con_articulo` le ponía «el licenciada». El
# engrose: «a la Ponencia a cargo de la licenciada Bertha Martínez Vega». El
# género lo dice la PALABRA del papel («licenciada», «Mtra.»), nunca el nombre
# de pila; «Lic.» no lo dice y se queda la fórmula neutra, sin el tratamiento.
_TRATAMIENTOS = {
    "licenciada": ("la", "licenciada"), "licenciado": ("el", "licenciado"),
    "licda.": ("la", "licenciada"), "licdo.": ("el", "licenciado"),
    "maestra": ("la", "maestra"), "maestro": ("el", "maestro"),
    "mtra.": ("la", "maestra"), "mtro.": ("el", "maestro"),
    "doctora": ("la", "doctora"), "doctor": ("el", "doctor"),
    "dra.": ("la", "doctora"), "dr.": ("el", "doctor"),
    "lic.": ("", ""),
}
_RX_TRATAMIENTO = re.compile(
    r"(?i)^(licenciad[oa]|licd[oa]\.|lic\.|maestr[oa]|mtr[oa]\.|doctora?|dra?\.)"
    r"(?:\s+(?:en\s+derecho\s+)?|$)")
_RX_CARGO_PONENTE = re.compile(
    r"(?i)^(?:magistrad[oa]|secretari[oa]|juez|jueza|presidente|presidenta)\b")


def _ponente_en_prosa(nombre: str, titulo: str = "") -> str:
    """«a la ponencia de Luis Armando Pérez Topete» — SIN INFERIR EL GÉNERO POR
    EL NOMBRE (fórmula neutra del corpus). Si la fuente trae el cargo
    («Magistrada …», «Secretaria en funciones de Magistrada …») se respeta tal
    cual y se le pone su artículo: «de la Magistrada …», «del Magistrado …».
    El tratamiento, en minúscula y con su artículo: «de la licenciada …».

    `titulo` es el cargo que leyó el auto de turno o de returno
    (`turno.titulo`, `returno.titulo` de `ficha_tramite.leer_auto`): si el
    nombre llega sin cargo y el auto lo dice, se escribe («de la Magistrada
    Jenica Campos Juárez», Q 342/2025). Es lo escrito en el auto, no una
    inferencia.

    EL CARGO «EN FUNCIONES» VA EN APOSICIÓN (quinta ronda, 3-oct-2026, AR 239
    y 222/2025): «a la ponencia de la secretaria en funciones de Magistrada
    Bertha Martínez Vega» se lee como si Bertha fuera la magistrada a la que
    se suple. Los dos engroses ponen el nombre primero: «de Bertha Martínez
    Vega, secretaria en funciones de Magistrada», con el cargo en minúscula
    inicial. EL CARGO SIMPLE, CON SU MAYÚSCULA: «del Magistrado…» y «de la
    Magistrada…» (el corpus, «ponencia del Magistrado Luis Armando…»), aunque
    el auto lo escriba en minúscula, para que un mismo documento no mezcle
    «del magistrado Enrique…» con «de la Magistrada Jenica…» (AR 60/2025)."""
    n = _s(nombre).strip()
    if not n or _RX_TRATAMIENTO.fullmatch(n):
        return ""                       # sólo un tratamiento: el ponente falta
    n = n.rstrip(" .")
    import documento_generado as _dg
    n = _dg._nombre_de_organo(n)
    tit = _s(titulo).rstrip(" .")
    en_funciones = _en_funciones(n, tit)
    if en_funciones:
        nom, cargo = en_funciones
        base = _ponente_en_prosa(nom) or f"de {nom}"
        return f"{base}, {_cargo_en_funciones(cargo)}"
    if re.match(r"(?i)^(?:el|la)\s+", n):
        n = re.sub(r"(?i)^(el|la)\s+(magistrad[oa]|juez[a]?|president[ea]|secretari[oa])\b",
                   lambda x: f"{x.group(1)} {x.group(2)[:1].upper()}{x.group(2)[1:]}", n)
        return _contraer(f"de {n[:1].lower()}{n[1:]}")
    m = _RX_TRATAMIENTO.match(n)
    if m:
        resto = n[m.end():].strip()
        art, palabra = _TRATAMIENTOS.get(_sin_tildes(m.group(1)), ("", ""))
        if not resto:
            return ""
        return _contraer(f"de {art} {palabra} {resto}") if palabra else f"de {resto}"
    if _RX_CARGO_PONENTE.match(n):
        n = n[:1].upper() + n[1:]
        return _contraer(f"de {_dg._con_articulo(n)}")
    if tit and _RX_CARGO_PONENTE.match(tit):
        return _ponente_en_prosa(f"{tit} {n}")
    return f"de {n}"


_RX_EN_FUNCIONES = re.compile(r"(?i)\ben\s+funciones\b")


def _cargo_en_funciones(cargo: str) -> str:
    """«Secretaria en Funciones de Magistrada» → «secretaria en funciones de
    Magistrada»: el cargo en aposición con minúscula inicial, «en funciones
    de» en minúscula y el cargo suplido con la mayúscula del corpus (AR 239 y
    AD 456/2025 traían tres grafías del mismo cargo)."""
    c = re.sub(r"(?i)\ben\s+funciones\s+de(l|\s+la|\s+el)?\b",
               lambda m: "en funciones de" + (m.group(1) or "").lower(), _s(cargo))
    c = re.sub(r"(?i)\b(magistrad[oa]|juez[a]?)\b", lambda m: m.group(1)[:1].upper() + m.group(1)[1:].lower(), c)
    return c[:1].lower() + c[1:]
# «[la] Secretaria en funciones de Magistrada Bertha Martínez Vega»: el cargo
# compuesto delante y el nombre detrás (la palabra tras «de» cierra el cargo).
_RX_CARGO_EN_FUNCIONES_DELANTE = re.compile(
    r"(?i)^(?:(?:el|la)\s+)?(?P<cargo>\S+(?:\s+\S+){0,3}?\s+en\s+funciones\s+de\s+\S+)\s+(?P<nom>\S.*)$")


def _en_funciones(n: str, tit: str):
    """(nombre, cargo «en funciones») si el ponente es un secretario en
    funciones de magistrado; None si no. El cargo puede venir en el título
    del auto, delante del nombre o detrás, tras una coma."""
    if tit and _RX_EN_FUNCIONES.search(tit) and not _RX_EN_FUNCIONES.search(n):
        nom = re.sub(r"(?i)^(?:el|la)\s+", "", n).strip(" ,")
        return (nom, re.sub(r"(?i)^(?:el|la)\s+", "", tit)) if nom else None
    if not _RX_EN_FUNCIONES.search(n):
        return None
    m = re.match(r"^(?P<nom>[^,]+),\s*(?P<cargo>.*\ben\s+funciones\b.*)$", n, re.I)
    if m and not _RX_EN_FUNCIONES.search(m.group("nom")):
        return (m.group("nom").strip(),
                re.sub(r"(?i)^(?:el|la)\s+", "", m.group("cargo").strip()))
    m = _RX_CARGO_EN_FUNCIONES_DELANTE.match(n)
    if m:
        return m.group("nom").strip(), m.group("cargo").strip()
    return None


def _ministerio_publico(valor: str) -> str:
    """SÓLO SI CONSTA. «El agente del Ministerio Público omitió formular
    pedimento» se afirmaba en 71 de 72 AD porque el catálogo lo exigía
    (tipos_asunto.py:357-360), conste o no. Vacío = no se dice nada."""
    v = _sin_tildes(_s(valor))
    if v in ("pedimento", "formulo", "formulo_pedimento", "si"):
        return ("El agente del Ministerio Público de la Federación adscrito "
                "formuló pedimento.")
    if v in ("sin_pedimento", "no", "omiso", "no_formulo"):
        return ("El agente del Ministerio Público de la Federación adscrito no "
                "formuló pedimento.")
    return ""


def _turno_y_returno(ficha: dict, datos: dict, av: _Avisos, cola_turno: str,
                     rotulo_turno: str = "Turno.") -> list:
    """El resultando de turno y, si consta, el de returno.

    EL ARTÍCULO DEL TURNO ES EL DE CADA VÍA: 183 en el amparo directo, 92 en la
    revisión (y en la fiscal, por el 63 de la LFPCA) y NINGUNO en la queja. El
    «artículo 101» en el turno o el returno es error medido (42 AR, 34 AD y 9
    RF): el 101 regula el TRÁMITE de la queja, no el turno.

    EL PONENTE DEL TURNO. Si el auto de turno no trae el nombre y NO hubo
    returno, el ponente es el de la carátula (`datos["magistrado"]`): es el
    mismo dato. Con returno, el del turno fue otro y no se adivina."""
    turno = _d(ficha.get("turno"))
    returno = _d(ficha.get("returno"))
    hay_returno = bool(_s(returno.get("fecha")) or _s(returno.get("ponente")))
    rot = rotulo_turno.rstrip(".")
    f_t = _fecha_o(turno.get("fecha"), av, "la fecha del auto de turno",
                   "turno.fecha", "está en el auto de turno (a veces es el mismo "
                   "auto de Presidencia que admite)", rot)
    pon = _s(turno.get("ponente"))
    if not pon and not hay_returno:
        pon = _s(datos.get("magistrado"))
    # Un ponente que sólo es un tratamiento («Licenciada») es un ponente que falta.
    pon_txt = (_ponente_en_prosa(pon, turno.get("titulo")) if pon else "") or "de " + av.falta(
        "el ponente al que se turnó", "turno.ponente",
        "está en el auto de turno", rot)
    out = [{"titulo": rotulo_turno,
            "texto": (f"Por acuerdo de {f_t}, se turnaron los autos a la ponencia "
                      f"{pon_txt}, para la elaboración del proyecto de resolución"
                      f"{cola_turno}.")}]
    if hay_returno:
        f_r = _fecha_o(returno.get("fecha"), av, "la fecha del auto de returno",
                       "returno.fecha", "está en el auto de returno", "Returno")
        pon_r = _s(returno.get("ponente"))
        pon_r_txt = (_ponente_en_prosa(pon_r, returno.get("titulo")) if pon_r else "") or "de " + av.falta(
            "el ponente del returno", "returno.ponente",
            "está en el auto de returno", "Returno")
        out.append({"titulo": "Returno.",
                    "texto": (f"Por acuerdo de {f_r}, se returnaron los autos a la "
                              f"ponencia {pon_r_txt}, para la elaboración del "
                              f"proyecto de resolución.")})
    return out


def _caracter_en_prosa(caracter: str) -> str:
    """El carácter procesal SIN GÉNERO: «parte quejosa», «autoridad
    responsable», «parte tercera interesada». No se concuerda por el nombre."""
    c = _sin_tildes(_s(caracter))
    return {"quejoso": "parte quejosa", "quejosa": "parte quejosa",
            "autoridad": "autoridad responsable",
            "autoridad_responsable": "autoridad responsable",
            "tercero": "parte tercera interesada",
            "tercero_interesado": "parte tercera interesada",
            "ministerio_publico": "agente del Ministerio Público de la Federación",
            }.get(c, "")


def _clave_nombre(x) -> str:
    """Para comparar dos nombres: sin tildes, sin artículos, sin puntuación,
    sin la etiqueta de rol y con juez/jueza/juzgado iguales."""
    t = _sin_tildes(_sin_rol(_s(x))[0])
    t = re.sub(r"\b(?:juez|jueza|juzgado)\b", "juzg", t)
    t = re.sub(r"[^\w\s]", " ", t)
    return " ".join(w for w in t.split() if w not in ("el", "la", "los", "las"))


def _contiene(a: str, b: str) -> bool:
    """¿Un nombre contiene al otro (comparados con `_clave_nombre`)?"""
    ka, kb = _clave_nombre(a), _clave_nombre(b)
    return bool(ka and kb) and (ka == kb or f" {kb} " in f" {ka} " or f" {ka} " in f" {kb} ")


def _figura_en_prosa(fig: str, ley_de_amparo: bool = True) -> str:
    """La figura del representante en prosa: la genérica en minúscula («su
    apoderado legal»), el cargo de una autoridad con su mayúscula, sin perder
    la de la ley que cita («autorizado en términos amplios del artículo 12 de
    la Ley de Amparo»).

    UNA SOLA PUERTA PARA LA FIGURA (quinta ronda, 3-oct-2026, F6; AD 456 y 274
    con nombres de prueba): el resultando decía «por conducto de su Apoderado
    Legal Juan Ruiz Gómez» y la legitimación y el resolutivo del mismo
    documento «su apoderado legal Juan Ruiz Gómez»: había tres convertidores.
    Decide `tipos_asunto.figura_en_prosa` (import local); el cuerpo de abajo es
    el respaldo si esa pieza no se puede llamar. `ley_de_amparo=False` en la
    revisión fiscal: ahí un «artículo 12» no es el de la Ley de Amparo."""
    f = re.sub(r"(?i)^(?:su|el|la)\s+", "", _s(fig))
    if not f:
        return ""
    try:
        import tipos_asunto as _ta
        _f = getattr(_ta, "figura_en_prosa", None)
        if callable(_f):
            r = _s(_f(f, ley_de_amparo=ley_de_amparo))
            if r:
                return r
    except Exception:
        pass
    if f == f.upper():
        f = f.lower()
    f = re.sub(r"(?i)\bley\s+de\s+amparo\b", "Ley de Amparo", f)
    # EL ARTÍCULO 12 SIN SU LEY (cuarta ronda, 3-oct-2026, E9; AR 239 y 60/2025,
    # Q 172 y 300): «por conducto de su autorizado en términos amplios del
    # artículo 12, Fulano» citaba un artículo sin ordenamiento en el resultando
    # firmado. El 12 (autorizado) y el 9o. (delegado de la autoridad) son de la
    # Ley de Amparo: se completa, si no cita ya otra ley.
    if not ley_de_amparo:
        return f
    return _RX_ART_SIN_LEY.sub(lambda m: m.group(0) + " de la Ley de Amparo", f)


def _sep_figura(fig: str) -> str:
    """El separador entre la figura y el nombre: coma si la figura cita un
    artículo o pasa de tres palabras («…del artículo 12 de la Ley de Amparo,
    Fulano»). LA MISMA REGLA en el resultando y en la legitimación (quinta
    ronda, F6; Q 172 y 300: «autorizada en términos amplios, Representante» en
    uno y «…amplios Representante» en el otro): `tipos_asunto._sep_figura`."""
    try:
        import tipos_asunto as _ta
        _f = getattr(_ta, "_sep_figura", None)
        if callable(_f):
            r = _f(fig)
            if r in (", ", " "):
                return r
    except Exception:
        pass
    return ", " if (re.search(r"(?i)\bart[íi]culo\b", fig or "") or len((fig or "").split()) > 3) else " "


_RX_ART_SIN_LEY = re.compile(
    r"(?i)\bart[íi]culo\s+(?:12|9(?:o\.?|º|°)?)(?![\wº°])"
    r"(?!\s*,?\s*(?:de\s+la\s+(?:ley|propia)|de\s+esta\s+ley|del\s+c[óo]digo|de\s+la\s+lfpca))")


def _representacion(ficha: dict, datos: dict = None, usar_datos: bool = False,
                    promovente: str = "", av: "_Avisos" = None, rep_fig: tuple = None,
                    info: dict = None, apartado: str = "", sin_nombre: bool = True,
                    ley_de_amparo: bool = True) -> str:
    """«, por conducto de su apoderado legal Fulano». Vacío si no consta.

    EL REPRESENTANTE QUE ES LA MISMA PARTE NO SE ESCRIBE (3-oct-2026, Q 261,
    Q 337 y Q 342/2025, AR 307/2024): quien actúa «por propio derecho y en
    representación de» sus hijos llegaba como su propio representante, y el
    albacea que ya va en el nombre de la sucesión se repetía: «SUCESIÓN … A
    TRAVÉS DE SU ALBACEA X, …, por conducto de su albacea X». Si el
    representante está contenido en quien promueve, no hay «por conducto de».

    LA FIGURA LARGA LLEVA COMA ANTES DEL NOMBRE (AR 239/2025): «por conducto
    de su autorizado en términos amplios del artículo 12 de la Ley de Amparo,
    Fulano», no «…de la Ley de Amparo Fulano».

    LA TITULAR DE LA UNIDAD QUE RECURRE NO ES «SU» REPRESENTANTE (cuarta
    ronda, 3-oct-2026, E8; RF 7, 2 y 26/2025): «la Jefa de la Unidad Jurídica…,
    por conducto de su Jefa de la Unidad Jurídica, Fulana» decía que la unidad
    actuó por conducto de sí misma. Si la figura es el cargo que encabeza a
    quien promueve, el representante ES su titular: no hay «por conducto de»
    y su nombre sale en `info["titular"]` para la legitimación («lo hizo valer
    Fulana, Jefa de la Unidad Jurídica…», como el corpus).

    `rep_fig` = (representante, figura) ya depurados por quien llama (la
    revisión fiscal descarta los truncados y pasa a figura el cargo).
    `apartado` es el resultando donde queda el hueco del nombre si sólo consta
    la figura; `sin_nombre=False` calla la figura sin nombre (la revisión
    fiscal, cuya legitimación tiene su propia regla para la unidad)."""
    if rep_fig is not None:
        rep, fig = _s(rep_fig[0]), _s(rep_fig[1])
    else:
        rep = _s(ficha.get("representante"))
        fig = _s(ficha.get("figura_representante"))
        # Del encargo sólo si la ficha no trae el NOMBRE; y su figura sólo si
        # la del encargo existe (la de la ficha no se pierde, F2).
        if not rep and usar_datos and datos and _s(datos.get("representante")):
            rep = _s(datos.get("representante"))
            fig = _s(datos.get("figura_representante")) or fig
    if not rep and not fig:
        return ""
    if fig and promovente and _es_autoridad(promovente) and _figura_es_la_unidad(fig, promovente):
        if info is not None:
            info["titular"] = _prosa_nombre(rep) if rep else ""
        return ""
    if not rep:
        # LA FIGURA SIN NOMBRE NO SE BORRA (quinta ronda, 3-oct-2026, F2; AR
        # 72/2025 y Q 342/2025): la ficha traía «delegado» (y «autorizado») sin
        # el nombre, testado, y el documento decía que el Gobernador recurrió
        # en persona —y la sucesión «por conducto de su albacea»—, cuando firmó
        # su delegado (art. 9o.) o el autorizado (art. 12). Se escribe la figura
        # con el nombre en hueco y se avisa con la clave «representante»; la
        # legitimación (`tipos_asunto.legitimacion_de`) escribe lo mismo.
        if not sin_nombre:
            return ""
        fig_p = _figura_en_prosa(fig, ley_de_amparo=ley_de_amparo)
        hueco = (av.falta("el nombre del representante", "representante",
                          f"consta que actuó por conducto de su {fig_p}, pero la ficha no trae su "
                          "nombre; está en el escrito y en el auto que le reconoce la personería",
                          apartado or "el resultando", tambien="la legitimación")
                 if av is not None else HUECO)
        return f", por conducto de su {fig_p}{_sep_figura(fig_p)}{hueco}"
    if promovente and _contiene(promovente, rep) and \
            len(_clave_nombre(rep)) <= len(_clave_nombre(promovente)):
        if av is not None:
            av.add("EL REPRESENTANTE ES LA MISMA PARTE QUE PROMUEVE (representante = "
                   f"«{rep}»): no se escribió «por conducto de». Si actuó por propio "
                   "derecho y en representación de otra persona, o recurrió un "
                   "autorizado, corrígelo en la ficha.")
        return ""
    rep = _prosa_nombre(rep)
    fig = _figura_en_prosa(fig, ley_de_amparo=ley_de_amparo)
    if not fig:
        if av is not None:
            av.add("FALTA LA FIGURA DEL REPRESENTANTE (figura_representante): está en el "
                   "escrito (apoderado, representante legal, autorizado…). Se escribió "
                   f"«por conducto de {rep}» sin decir con qué carácter.")
        return f", por conducto de {rep}"
    return f", por conducto de su {fig}{_sep_figura(fig)}{rep}"


def _supletorio(ficha: dict, datos: dict) -> dict:
    """La supletoriedad de la sede (SPEC §3.1). La decide UNA función,
    `tipos_asunto.supletorio`, y aquí sólo se le pregunta: dos sitios que
    decidieran el código por su lado acabarían citando dos códigos."""
    sede = _d(ficha.get("sede"))
    try:
        import tipos_asunto as _ta
        return dict(_ta.supletorio(_s(sede.get("tribunal")) or _s(datos.get("tribunal")),
                                   _s(sede.get("ciudad")) or _s(datos.get("ciudad"))))
    except Exception:
        return {}


# ── LOS ASUNTOS RELACIONADOS, SÓLO LOS QUE MARCÓ EL SECRETARIO (sexta ronda, C6) ──
# David, 3-oct-2026: «siempre y cuando haya asuntos relacionados. No vamos a
# meter conexidad en automático». La lista sale SÓLO de la ficha (`relacionados`,
# que llena el interruptor de «Trámite en este tribunal», fuente «secretario»);
# nunca de los papeles ni de `datos`. El V I S T O la nombra justo tras el
# número («…queja civil 24/2026, relacionado con el amparo en revisión civil
# 298/2025, interpuesto por…», como Q 24/2026 y AD 469/2024 del banco) y
# `datos_extra["relacionados"]` la pasa a `documento_generado`, que pone el
# rubro «RELACIONADO CON…» y el considerando de conexidad o de hecho notorio.
# La prosa y la normalización son de `tipos_asunto` (una sola puerta).
def _relacionados(ficha: dict, tipo: str, num: str) -> list:
    """La lista normalizada (`tipos_asunto.relacionados_validos`): tipo
    conocido, número N/AAAA, estado, sin duplicados, hasta 4. Sin el propio
    asunto ni lo que no marcó el secretario: `ficha_tramite.validar` ya los
    quita con su aviso (el compositor compone sobre la copia validada); esto
    es el respaldo para cuando esa pieza no responde. [] si no hay."""
    fuentes = _d(ficha.get("fuentes"))
    if _s(fuentes.get("relacionados")) not in ("", "secretario"):
        return []
    try:
        import tipos_asunto as _ta
        lista = _ta.relacionados_validos(ficha.get("relacionados"))
    except Exception:
        return []
    m = _RX_NUM.search(_s(num))
    propio = f"{int(m.group(1))}/{m.group(2)}" if m else ""
    return [dict(r) for r in lista if not (r.get("tipo") == tipo and r.get("numero") == propio)]


def _relacionado_con(lista: list, mat: str) -> str:
    """«, relacionado con el amparo directo civil 452/2025» (y «… y con el
    recurso de revisión fiscal 33/2024»), para el V I S T O justo tras el
    número; «» sin relacionados. La materia es la del asunto, concordada por
    `tipos_asunto` con cada nombre («amparo directo administrativo», «recurso
    de queja administrativa»; la revisión fiscal, sin materia)."""
    if not lista:
        return ""
    try:
        import tipos_asunto as _ta
        p = _ta.relacionados_en_prosa(lista, _materia_concordada(mat, "f"))
    except Exception:
        return ""
    return f", relacionado con {p}" if p else ""


# ═══════════════════════════════════════════════════════════════════════════
# AMPARO DIRECTO
# ═══════════════════════════════════════════════════════════════════════════
_CLASE_AD = {"laudo": ("el laudo", "dictado", "El laudo"),
             "resolucion": ("la resolución", "dictada", "La resolución"),
             "sentencia": ("la sentencia", "dictada", "La sentencia")}

_RX_SALA_ALZADA = re.compile(r"(?i)\bsala\b.*\btribunal\s+superior\b|\bsala\b.*\b(civil|familiar|penal)\b|"
                             r"\btribunal\s+(?:de\s+alzada|de\s+apelaci[oó]n)\b|"
                             r"\btribunal\s+unitario\s+de\s+circuito\b")


def _unica_instancia(ficha: dict, datos: dict, acto: dict, toca: str, resp: str,
                     av: _Avisos) -> bool:
    """¿Se dictó el acto en ÚNICA instancia? LO DICE LA FICHA, no la marca.

    TERCERA RONDA (3-oct-2026, punta a punta AD 274/2025 → 9274/2025): la marca
    de única instancia llegó del contexto (`origen_acto` leyó «la Jueza» en el
    relato del juicio de origen) y el V I S T O se escribió «en el expediente
    479/2024», aunque la ficha traía el toca familiar 4520/2024 y la
    responsable era la Segunda Sala Civil del Tribunal Superior de Justicia: el
    toca se TIRÓ por obedecer a la marca. Un toca o una Sala de apelación son
    HECHOS del papel; la marca es una inferencia. Con cualquiera de los dos no
    hay única instancia, diga lo que diga la marca, y si la marca decía otra
    cosa se avisa. Sin toca y sin órgano de alzada, deciden la marca de la
    ficha (`unica_instancia`, `acto.instancia`, que `ficha_tramite` toma de
    `origen_acto.origen`), la del encargo y la del contexto."""
    marcas = []
    if ficha.get("unica_instancia") is True:
        marcas.append("unica_instancia")
    if _sin_tildes(_s(acto.get("instancia"))) == "unica":
        marcas.append("acto.instancia")
    if _sin_tildes(_s(datos.get("instancia_origen"))) == "unica":
        marcas.append("instancia_origen")
    if not marcas:
        try:
            import tipos_asunto as _ta
            if _ta.unica_instancia("amparo_directo"):
                marcas.append("instancia_origen")
        except Exception:
            pass
    alzada = bool(resp and _RX_SALA_ALZADA.search(resp))
    alzada_ficha = _sin_tildes(_s(acto.get("instancia"))) == "alzada"
    if toca or alzada or alzada_ficha:
        if marcas:
            hechos = []
            if toca:
                hechos.append(f"trae el toca «{toca}»")
            if alzada:
                hechos.append("la responsable es un órgano de apelación")
            if alzada_ficha:
                hechos.append("dice que el acto es de alzada (acto.instancia)")
            av.add("LA MARCA DE ÚNICA INSTANCIA NO CUADRA CON LA FICHA ("
                   + ", ".join(marcas) + "): la ficha " + " y ".join(hechos)
                   + ", así que hubo apelación. Se escribieron el toca y el expediente "
                   "de origen. Si de verdad se dictó en única instancia, quita el toca "
                   "de la ficha y corrige la autoridad responsable.")
        return False
    return bool(marcas)


def _via_tribunal_confirmada(ficha: dict) -> bool:
    """«Ante este Tribunal Colegiado» sólo si lo DECLARÓ el secretario (rev_1,
    AD 128, 274, 335 y 456/2025): cuatro fichas leídas traían «tribunal» porque
    la demanda entró por la Oficialía del Tribunal Superior de Justicia del
    Estado, que es la responsable, y el resultando afirmaba en falso que se
    presentó ante este Tribunal Colegiado."""
    return _sin_tildes(_s(_d(ficha.get("fuentes")).get("via_presentacion"))) == "secretario"


def _amparo_directo(ficha, datos, av: _Avisos):
    acto = _d(ficha.get("acto"))
    num = _numero_asunto(ficha.get("numero") or datos.get("numero"))
    num_t = num or av.falta("el número del amparo directo", "numero",
                            "está en el auto de Presidencia que registra la demanda",
                            "V I S T O")
    mat = _materia_clave(ficha.get("materia") or datos.get("materia"))
    mat_t = _materia_concordada(mat, "m") or av.falta(
        "la materia del amparo directo", "materia",
        "está en el auto de admisión (civil, administrativa, penal o laboral)",
        "V I S T O")
    quejoso = _s(ficha.get("promovente")) or _s(datos.get("quejoso")) or \
        _s(_d(_d(datos.get("ficha_procesal")).get("quejosa")).get("nombre"))
    quejoso_t = _parte(quejoso) if quejoso else av.falta(
        "el nombre de la parte quejosa", "promovente", "está en la demanda de amparo",
        "V I S T O")
    _aviso_abreviado(quejoso, "promovente", av, "la demanda de amparo")
    plural = _es_plural(quejoso)
    resp_crudo = ficha.get("responsable") or acto.get("organo") or datos.get("responsable")
    resp = _organo(resp_crudo)
    resp_t = resp or av.falta("la autoridad responsable que dictó el acto",
                              "responsable", "está en el proemio de la sentencia "
                              "reclamada", "V I S T O")
    ejec = _organo(ficha.get("ejecutora"))
    clase = _sin_tildes(_s(acto.get("clase")))
    clase = clase if clase in _CLASE_AD else "sentencia"
    art_cl, dictad, Cl = _CLASE_AD[clase]
    f_acto_d = _fecha(acto.get("fecha"))
    f_acto = _fecha_o(acto.get("fecha"), av, "la fecha del acto reclamado",
                      "acto.fecha", "está en la sentencia reclamada", "V I S T O")
    toca = _s(acto.get("toca"))
    exp = _s(acto.get("expediente"))
    unica = _unica_instancia(ficha, datos, acto, toca, resp, av)
    toca_r = _rotulado(toca, ("toca",), "toca")
    exp_r = _rotulado(exp, ("expediente", "juicio", "exp."), "expediente")
    if toca_r and exp_r:
        donde = f"{toca_r}, derivado del {exp_r}"
    elif toca_r:
        donde = toca_r
        if not unica:
            av.add("FALTA EL EXPEDIENTE DE ORIGEN (acto.expediente): está en la "
                   "sentencia reclamada, junto al toca. El acto se identificó sólo "
                   "con el toca; la existencia lo necesita aparte.")
    elif exp_r:
        donde = exp_r
        if not unica and resp and _RX_SALA_ALZADA.search(resp):
            av.add("FALTA EL TOCA (acto.toca): la responsable es una Sala de "
                   "apelación y el acto se identificó sólo con el expediente de "
                   "origen. Está en el proemio de la sentencia reclamada.")
    else:
        rot = "toca" if (resp and _RX_SALA_ALZADA.search(resp) and not unica) else "expediente"
        donde = f"{rot} " + av.falta(
            "el toca o el expediente en que se dictó el acto",
            "acto.toca / acto.expediente", "está en el proemio de la sentencia "
            "reclamada", "V I S T O")
    y_ejec = ", y su ejecución" if ejec else ""
    resp_art = _con_art(resp, art_papel=_articulo_del_papel(resp_crudo)) if resp else resp_t
    if resp:
        _aviso_genero_del_cargo(resp_crudo, "responsable", av)

    # C6 (3-oct-2026): los relacionados que marcó el secretario, tras el número.
    rel = _relacionados(ficha, "amparo_directo", num)
    visto = (f"para resolver el juicio de amparo directo {mat_t} {num_t}"
             f"{_relacionado_con(rel, mat)}, promovido "
             f"por {quejoso_t}, contra {art_cl} {dictad} el {f_acto}, por "
             f"{resp_art}, en el {donde}{y_ejec}; y,")
    visto = _contraer(visto)

    res = []
    # 1 · PRESENTACIÓN — art. 176 LA: por conducto de la responsable. NADA de
    # «Oficialía de Partes de este Tribunal» (16 de 72 AD la inventaban).
    ap1 = "Presentación de la demanda de amparo"
    pres = _fecha_o(ficha.get("presentacion"), av, "la fecha de presentación de la "
                    "demanda", "presentacion", "está en el sello de recepción o en "
                    "la certificación de la responsable al pie de la demanda (art. "
                    "178, fr. I, LA)", ap1, respaldo=datos.get("presentacion"))
    via = _sin_tildes(_s(ficha.get("via_presentacion")))
    if via == "electronica":
        ante = f" ante {resp_art}, por vía electrónica"
    elif via in ("tribunal", "tribunal_colegiado") and _via_tribunal_confirmada(ficha):
        ante = " ante este Tribunal Colegiado"
        av.add("LA DEMANDA SE PRESENTÓ ANTE ESTE TRIBUNAL COLEGIADO y no por conducto "
               "de la responsable (art. 176 LA): el resultando lo dice así porque la "
               "ficha lo declara; revisa la oportunidad con el segundo párrafo del "
               "artículo 176.")
    elif via in ("tribunal", "tribunal_colegiado"):
        # MENOS DETALLE, NO UNA AFIRMACIÓN SIN FUENTE: la ficha leída dice
        # «tribunal» y el secretario no lo confirmó. No se escribe ante quién.
        ante = ""
        av.add("LA FICHA LEÍDA DICE QUE LA DEMANDA SE PRESENTÓ ANTE «TRIBUNAL» "
               "(via_presentacion) y no lo confirmaste: no se escribió ante quién. Si "
               "entró por la Oficialía de la responsable (art. 176 LA), elige "
               "«responsable»; si se presentó ante este Tribunal Colegiado, elige esa "
               "vía en «Trámite en este tribunal».")
    elif via == "juzgado":
        ante = " ante el Juzgado de Distrito " + av.falta(
            "el Juzgado de Distrito ante el que se presentó la demanda",
            "via_presentacion", "está en el acuerdo de remisión del juzgado", ap1)
    else:
        ante = f" ante {resp_art}"
    rep = _representacion(ficha, datos, usar_datos=True, promovente=quejoso, av=av, apartado=ap1)
    sujeto = f"{quejoso_t}{rep}," if rep else _cierra_inciso(quejoso_t)
    promovio = "promovieron" if plural else "promovió"
    # EL RENGLÓN DE LA AUTORIDAD VA SIN ARTÍCULO Y SIN PUNTO, como en el corpus
    # («Autoridad responsable: • Magistrado Presidente de la Sala Familiar…»).
    if ejec:
        bloque_aut = (f"AUTORIDADES RESPONSABLES:\n{_mayus(resp or resp_t)} (ordenadora) y "
                      f"{ejec} (ejecutora)")
        contra = "de las autoridades y del acto"
    else:
        bloque_aut = f"AUTORIDAD RESPONSABLE:\n{_mayus(resp or resp_t)}"
        contra = "de la autoridad y del acto"
    t1 = (f"Por escrito presentado el {pres}{ante}, {sujeto} {promovio} juicio de "
          f"amparo directo en contra {contra} que a continuación se precisan:\n"
          f"{bloque_aut}\nACTO RECLAMADO:\n{Cl} {dictad} el {f_acto}, en el "
          f"{donde}{y_ejec}.")
    res.append({"titulo": ap1 + ".", "texto": _contraer(t1)})

    # 2 · DERECHOS HUMANOS — la lista del capítulo de la demanda; sin ella,
    # hueco. La perífrasis «los artículos que precisó en su demanda» salía en
    # 57 de 72 AD.
    ap2 = "Derechos humanos que se estiman vulnerados"
    ders = _derechos(ficha.get("derechos"))
    if ders:
        art = "los artículos" if len(ders) > 1 else "el artículo"
        lista = _y(ders)
    else:
        art = "los artículos"
        lista = av.falta("los artículos constitucionales que la demanda dice violados",
                         "derechos", "están en el capítulo de preceptos violados "
                         "de la demanda de amparo", ap2)
    res.append({"titulo": ap2 + ".",
                "texto": (f"La parte quejosa señaló como tales los contenidos en {art} "
                          f"{lista} de la Constitución Política de los Estados Unidos "
                          f"Mexicanos.")})

    # 3 · TERCERO INTERESADO — sólo si lo hay. Sin él se omite el resultando
    # (no se sustituye por una perífrasis) y se avisa.
    terceros = _terceros(ficha, datos)
    if terceros:
        verbo = "Tienen" if (len(terceros) > 1 or any(_es_plural(x) for x in terceros)) \
            else "Tiene"
        res.append({"titulo": "Tercero interesado.",
                    "texto": f"{verbo} ese carácter {_y([_parte(x) for x in terceros])}."})
        for x in terceros:
            _aviso_genero_del_cargo(x, "terceros", av)
    else:
        av.add("NO CONSTA TERCERO INTERESADO (terceros): en un amparo directo casi "
               "siempre lo es la contraparte del juicio de origen. Está en el auto "
               "de admisión o en la constancia de emplazamiento de la responsable "
               "(art. 178, fr. II, LA). Se omitió el resultando.")

    # 4 · TRÁMITE — auto de Presidencia, art. 181; MP sólo si consta; adhesivo.
    ap4 = "Trámite del juicio de amparo"
    f_adm = _fecha_o(_d(ficha.get("admision")).get("fecha"), av,
                     "la fecha del auto de Presidencia que admitió la demanda",
                     "admision.fecha", "está en el auto de Presidencia de este "
                     "Tribunal Colegiado", ap4)
    f_reg = _registro_distinto(ficha, av, ap4)
    if f_reg:
        t4 = (f"Por auto de Presidencia de {f_reg}, este Tribunal Colegiado registró la "
              f"demanda con el número {num_t}; y por auto de {f_adm} la admitió a trámite y "
              f"concedió a las partes el plazo de quince días para formular alegatos o "
              f"promover amparo adhesivo, en términos del artículo 181 de la Ley de Amparo.")
    else:
        t4 = (f"Por auto de Presidencia de {f_adm}, este Tribunal Colegiado registró la "
              f"demanda con el número {num_t}, la admitió a trámite y concedió a las "
              f"partes el plazo de quince días para formular alegatos o promover amparo "
              f"adhesivo, en términos del artículo 181 de la Ley de Amparo.")
    mp = _ministerio_publico(ficha.get("ministerio_publico"))
    if mp:
        t4 += "\n" + mp
    adh = _d(ficha.get("adhesivo"))
    if _hay_adhesivo(adh):
        f_ad = _fecha_adhesivo("amparo_directo", adh, av, "la fecha del auto que admitió el "
                               "amparo adhesivo", "está en el auto que lo admite", ap4)
        quien = _o(adh.get("quien"), av, "quién promovió el amparo adhesivo",
                   "adhesivo.quien", "está en el escrito de amparo adhesivo y en el "
                   "auto que lo admite", ap4)
        t4 += (f"\nPor auto de {f_ad}, se admitió el amparo adhesivo promovido por "
               f"{_parte(quien) if quien != HUECO else quien}.")
    res.append({"titulo": ap4 + ".", "texto": t4})

    # 5 · TURNO (art. 183) y 6 · RETURNO.
    res += _turno_y_returno(ficha, datos, av,
                            ", en términos del artículo 183 de la Ley de Amparo")

    extra = {
        # `toca` y `expediente` TAL COMO VIENEN en la ficha; con su rótulo
        # («toca familiar 4357/2025», «expediente 515/2022») en `*_en_prosa`,
        # para que la existencia no escriba «del toca toca familiar…».
        "expediente": exp, "toca": toca,
        "toca_en_prosa": toca_r, "expediente_en_prosa": exp_r,
        "fecha_acto": f_acto if f_acto != HUECO else "",
        "fecha_acto_iso": f_acto_d.isoformat() if f_acto_d else "",
        "organo_acto": resp, "responsable": resp, "ejecutora": ejec,
        "materia": mat, "numero": num, "unica_instancia": unica,
        "descripcion_acto": _contraer(f"de {art_cl} reclamad"
                                      f"{'o' if art_cl.startswith('el ') else 'a'}"),
        # TERCERA RONDA (3-oct-2026, AD 349/2025 y AD 552/2024): la ficha decía
        # «resolución» y la competencia, la existencia, la legitimación y la
        # oportunidad seguían diciendo «sentencia». Los considerandos leen la
        # clase y el nombre del acto de aquí.
        "clase": clase,
        "acto_reclamado": f"{art_cl} reclamad{'o' if art_cl.startswith('el ') else 'a'}",
        "terceros": terceros,
        # EL NOMBRE COMO LO ESCRIBEN LOS RESULTANDOS, para que la legitimación y
        # los resolutivos digan lo mismo que el V I S T O (y no «GABRIEL REYES
        # ALAMO» en versales, punta a punta AD 274/2025). «» si fue hueco.
        "promovente_en_prosa": quejoso_t if quejoso_t != HUECO else "",
        "quejoso_en_prosa": quejoso_t if quejoso_t != HUECO else "",
        # E2: el número que decidió el resultando, para la legitimación y la
        # carátula («están legitimados», «QUEJOSOS:»).
        "plural": plural, "plural_quejoso": plural,
        # C6: el rubro y el considerando de conexidad o de hecho notorio.
        "relacionados": rel,
    }
    return visto, res, extra


def _derechos(x) -> list:
    """«1», «1º», «1o.» → «1o.»; los del 10 en adelante, el número solo."""
    if isinstance(x, str):
        x = re.split(r"\s*(?:,|;|\by\b)\s*", x)
    out = []
    for a in _lista(x):
        t = _s(a).replace("º", "o").replace("°", "o")
        t = re.sub(r"(?i)^(?:art[íi]culos?|art\.)\s*", "", t).strip(" .")
        if not t:
            continue
        m = re.match(r"^(\d{1,3})\s*o?\.?$", t, re.I)
        if m:
            n = int(m.group(1))
            t = f"{n}o." if n < 10 else str(n)
        if t not in out:
            out.append(t)
    return out


def _terceros(ficha: dict, datos: dict) -> list:
    ts = ficha.get("terceros")
    if isinstance(ts, str):
        ts = [ts]
    out = [_s(t.get("nombre") if isinstance(t, dict) else t) for t in _lista(ts)]
    out = [t for t in out if t]
    if not out:
        fp = [_s(_d(t).get("nombre")) for t in
              _lista(_d(datos.get("ficha_procesal")).get("terceros"))]
        out = [t for t in fp if t]
    if not out and _s(datos.get("tercero")):
        out = [t.strip() for t in str(datos.get("tercero")).split(";") if t.strip()]
    vistos, uno = set(), []
    for t in out:
        if t.lower() not in vistos:
            vistos.add(t.lower())
            uno.append(t)
    return uno


def _hay_adhesivo(adh: dict) -> bool:
    return any(_s(adh.get(k)) for k in ("quien", "presentacion", "admision", "notificacion"))


# ── EL ADHESIVO SIN NINGÚN AUTO (cuarta ronda, 3-oct-2026, E5; AD 274/2025) ──
# La ficha traía como adherente a la tercera interesada, que sólo presentó
# alegatos, y el documento le escribió un renglón en la carátula, un párrafo en
# el trámite, un considerando con tres huecos y un segundo punto resolutivo:
# un resolutivo sobre un adhesivo sin fecha de admisión, de presentación ni de
# notificación. Sin el auto que lo admite NI el escrito, no se escriben su
# considerando ni su resolutivo; el resultando lo menciona con la fecha en
# hueco y el aviso pregunta si lo hubo.
_PREGUNTA_ADHESIVO = {
    "amparo_directo": ("¿HUBO AMPARO ADHESIVO? No consta el auto que lo admite",
                       "lo promovió", "lo menciona"),
    "amparo_revision": ("¿HUBO REVISIÓN ADHESIVA? No consta el auto que la admite",
                        "la interpuso", "la menciona"),
    "revision_fiscal": ("¿HUBO REVISIÓN ADHESIVA? No consta el auto que la admite",
                        "se adhirió", "la menciona"),
}


def _adhesivo_sin_auto(adh: dict) -> bool:
    """¿Consta quién, pero ni la admisión ni la presentación del adhesivo?"""
    return bool(_s(adh.get("quien"))) and not _s(adh.get("admision")) and \
        not _s(adh.get("presentacion"))


def _aviso_adhesivo_sin_auto(t: str, adh: dict) -> str:
    """El MISMO texto en el resultando, el considerando y el resolutivo: la
    lista de avisos lo deja una sola vez."""
    preg, verbo, menciona = _PREGUNTA_ADHESIVO.get(t, _PREGUNTA_ADHESIVO["amparo_directo"])
    return (f"{preg} (adhesivo.admision): la ficha dice que {verbo} «{_s(adh.get('quien'))}», "
            f"pero no trae la fecha de su admisión ni la de su presentación. El resultando "
            f"{menciona} con la fecha en hueco y no se escribieron su considerando ni su punto "
            f"resolutivo. Si sólo presentó alegatos, quítalo de la ficha.")


def _fecha_adhesivo(t: str, adh: dict, av: "_Avisos", que: str, donde: str,
                    apartado: str) -> str:
    """La fecha del auto que admitió el adhesivo, o el hueco con su aviso (el
    de la pregunta si no consta ningún auto)."""
    if _adhesivo_sin_auto(adh):
        av.add(_aviso_adhesivo_sin_auto(t, adh))
        return HUECO
    return _fecha_o(adh.get("admision"), av, que, "adhesivo.admision", donde, apartado)


# ── EL REGISTRO Y LA ADMISIÓN, DOS AUTOS CUANDO SON DOS (cuarta ronda, 3-oct-2026, E7) ──
# RF 7/2025: el Tribunal radicó el recurso el 10-feb-2025, declinó competencia y
# lo admitió el 15-may-2025; el resultando decía «Por auto de Presidencia de
# quince de mayo…, este Tribunal Colegiado registró el recurso… y lo admitió»,
# que atribuye el registro al auto de admisión. Q 335/2025 (fracción II): el
# auto que forma, registra y pide el informe fue el 14-oct y el que lo tiene
# por rendido y admite, el 3-nov; salía el 3-nov en los dos. La ficha trae
# ahora el auto que forma y registra aparte (`registro.fecha`).
def _registro_distinto(ficha: dict, av: "_Avisos", apartado: str) -> str:
    """La fecha en letra del auto que FORMA Y REGISTRA si consta y es OTRO auto
    (otra fecha) que el que admite; «» si no consta o es el mismo."""
    reg_v = _d(ficha.get("registro")).get("fecha")
    if not _s(reg_v):
        return ""
    reg = _fecha(reg_v)
    if reg is None:
        av.add(f"FECHA ILEGIBLE EN LA FICHA (registro.fecha = «{_s(reg_v)}»): se esperaba "
               f"AAAA-MM-DD. «{apartado}» se escribió con un solo auto de Presidencia, el que "
               f"admite.")
        return ""
    return "" if _fecha(_d(ficha.get("admision")).get("fecha")) == reg else _letra(reg)


# ═══════════════════════════════════════════════════════════════════════════
# AMPARO EN REVISIÓN
# ═══════════════════════════════════════════════════════════════════════════
def _clase_recurrida(ficha: dict) -> str:
    acto = _d(ficha.get("acto"))
    c = _sin_tildes(_s(ficha.get("clase_recurrida")))
    if c in ("sentencia", "interlocutoria_suspension", "auto_sobreseimiento"):
        return c
    if acto.get("incidente") or _sin_tildes(_s(acto.get("clase"))) == "interlocutoria":
        return "interlocutoria_suspension"
    if _sin_tildes(_s(acto.get("clase"))) == "auto":
        return "auto_sobreseimiento"
    return "sentencia"


_VERBO_AR = {
    "sobresee": "se sobreseyó en el juicio",
    "concede": "se concedió el amparo",
    "niega": "se negó el amparo",
    "sobresee_niega": "en una parte se sobreseyó en el juicio y en otra se negó el amparo",
    "sobresee_concede": "en una parte se sobreseyó en el juicio y en otra se concedió el amparo",
    "concede_niega": "en una parte se concedió el amparo y en otra se negó",
    "niega_concede": "en una parte se concedió el amparo y en otra se negó",
}


def _resolvio(acto: dict, datos: dict) -> str:
    """La clave de lo que resolvió el juzgado. «mixto» a secas no dice si la
    otra parte concedió o negó: se completa con `acto.resolvio_mixto` (donde
    `ficha_tramite` guarda la clave de `fase_rama`, «sobresee_niega» o
    «sobresee_concede») o con `resolvio_a_quo` de los datos; si ninguno lo
    dice, queda sin completar (hueco)."""
    r = _sin_tildes(_s(acto.get("resolvio"))).replace(" ", "_")
    if r == "mixto":
        for otro in (acto.get("resolvio_mixto"), datos.get("resolvio_a_quo")):
            otro = _sin_tildes(_s(otro)).replace(" ", "_")
            if otro in ("sobresee_niega", "sobresee_concede"):
                return otro
        return "mixto"
    if r in _VERBO_AR:
        return r
    otro = _sin_tildes(_s(datos.get("resolvio_a_quo")))
    return otro if otro in _VERBO_AR else ""


_PORTAL_PJF = "a través del Portal de Servicios en Línea del Poder Judicial de la Federación"


def _via_del_escrito(ficha: dict, organo_con_art: str = "") -> str:
    """Lo que sigue a «presentado el {fecha}» en la revisión y en la queja.

    LA VÍA ELECTRÓNICA SE DICE (3-oct-2026, AR 201/2025 y AR 60/2025): la
    ficha traía `via_presentacion: electronica` y el resultando la callaba; los
    dos engroses dicen que el escrito entró por el Portal de Servicios en Línea
    del Poder Judicial de la Federación. «juzgado» con el juzgado conocido: ante
    él (art. 88 LA, el recurso se interpone por su conducto). Lo demás, nada:
    no se afirma una oficina receptora que no consta."""
    via = _sin_tildes(_s(ficha.get("via_presentacion")))
    if via == "electronica":
        return " " + _PORTAL_PJF
    if via == "juzgado" and organo_con_art and HUECO not in organo_con_art:
        return f" ante {organo_con_art}"
    return ""


# LA FORMA DEL ÓRGANO QUE RESUELVE UN AMPARO INDIRECTO: Juzgado (o Juez) «de
# Distrito» —no «del Distrito Judicial», que es el de primera instancia—, el
# Tribunal Colegiado de Apelación, el Tribunal Unitario de Circuito o el Centro
# de Justicia Penal Federal.
_RX_ORGANO_AMPARO = re.compile(
    r"(?i)\bju(?:ez|eza|zgado)\b[^,;]*?\bde\s+distrito\b(?!\s+judicial)|"
    r"tribunal\s+colegiado\s+de\s+apelaci|tribunal\s+unitario\s+de\s+circuito|"
    r"centro\s+de\s+justicia\s+penal\s+federal|\bju(?:ez|eza|zgado)\s+federal")


def _juzgado_del_amparo(acto: dict, datos: dict, fp: dict, autoridades_crudas: list,
                        av: _Avisos) -> tuple:
    """(juzgado en prosa sin artículo, su valor crudo, autoridades sin él).

    TERCERA RONDA (3-oct-2026, AR 307/2024, 448/2025 y 60/2025): la ficha
    confundió la sentencia recurrida con el acto reclamado y el V I S T O, el
    trámite, la competencia y la existencia afirmaron que la Sala Familiar o
    el Juez Octavo Familiar «conoció» del amparo indirecto; en el 222/2025, el
    Juzgado Séptimo de Distrito salió entre las autoridades responsables de su
    propio juicio. Ningún aviso. Ahora:
      · el candidato que COINCIDE con una autoridad responsable y no tiene forma
        de órgano de amparo es el acto reclamado, no el juzgado: se descarta
        (si no queda otro, HUECO) y se avisa;
      · si coincide y SÍ tiene forma de órgano de amparo, el error está en la
        lista de autoridades: se quita de ella y se avisa;
      · un juzgado sin forma de órgano de amparo se escribe (hay jurisdicción
        concurrente y auxiliar), pero se avisa."""
    candidatos = [(acto.get("organo"), "acto.organo"),
                  (datos.get("organo_recurrido"), "organo_recurrido"),
                  (_d(fp.get("organo_recurrido")).get("nombre"), "ficha_procesal")]
    autoridades = [a for a in autoridades_crudas if _s(a)]
    descartados = []
    elegido = ("", "")
    for crudo, fuente in candidatos:
        if not _s(crudo):
            continue
        choca = [a for a in autoridades if _contiene(a, crudo)]
        if choca and not _RX_ORGANO_AMPARO.search(_s(crudo)):
            descartados.append((_s(crudo), fuente))
            continue
        elegido = (_s(crudo), fuente)
        break
    juz_crudo = elegido[0]
    if descartados and juz_crudo:
        # Con otra fuente que sí trae el juzgado basta un aviso; sin ella, el
        # aviso del hueco (lo pone quien llama) dice por qué se descartó.
        av.add("SE DESCARTÓ COMO ÓRGANO DE LA RESOLUCIÓN RECURRIDA UNA AUTORIDAD "
               f"RESPONSABLE (acto.organo = «{descartados[0][0]}»): es el acto reclamado, no "
               f"la sentencia de amparo. Se escribió «{juz_crudo}» ({elegido[1]}); revisa "
               "también la fecha de la recurrida (acto.fecha).")
    if juz_crudo:
        fuera = [a for a in autoridades if _contiene(a, juz_crudo)]
        if fuera:
            autoridades = [a for a in autoridades if a not in fuera]
            av.add("EL JUZGADO QUE RESOLVIÓ EL AMPARO ESTABA ENTRE LAS AUTORIDADES "
                   f"RESPONSABLES (demanda.autoridades: «{_s(fuera[0])}»): se quitó de la "
                   "lista. Comprueba las autoridades en la demanda; puede faltar la que "
                   "se perdió.")
        if not _RX_ORGANO_AMPARO.search(juz_crudo):
            av.add(f"EL ÓRGANO QUE DICTÓ LA RESOLUCIÓN RECURRIDA NO PARECE UN JUZGADO DE "
                   f"DISTRITO (acto.organo = «{juz_crudo}»): el amparo indirecto lo "
                   "resuelve un Juzgado de Distrito o un Tribunal Colegiado de Apelación "
                   "(o el superior de la responsable, en jurisdicción concurrente). "
                   "Compruébalo en el proemio de la resolución recurrida.")
    # EN FORMA DE ÓRGANO, COMO EN LA QUEJA (cuarta ronda, 3-oct-2026, E10; AR
    # 239/2025): «por el Juez Tercero de Distrito…» para una jueza («por la Juez
    # Tercero», dice el engrose). El juzgado no tiene género que adivinar: «el
    # Juzgado Tercero de Distrito…» en el V I S T O, el trámite, la competencia
    # y la existencia. «la titular del Juzgado…» también pasa al órgano.
    juz = _forma_de_organo(_organo(juz_crudo)) if juz_crudo else ""
    return juz, juz_crudo, autoridades, [c for c, _f in descartados]


# ── LA DEMANDA DE AMPARO INDIRECTO, UNA SOLA MECÁNICA (AR y queja, sexta ronda) ──
# El primer resultando del amparo en revisión copia la demanda —autoridades y
# actos anclados al papel por `ficha_tramite`, uno por renglón— y desde el
# 3-oct-2026 (C3) la queja de la fracción I abre igual cuando la ficha la trae
# (Q 172/2026 del banco: «Demanda de amparo. … solicitó el amparo… en contra de
# las autoridades y actos siguientes»). David: «lo más práctico… que el
# secretario no tenga que modificar». Una sola función para los dos: los
# nombres, los plurales, la mayúscula del acto y su punto no pueden salir
# distintos en un tipo y en otro.
def _actos_de_la_demanda(dem: dict) -> list:
    """Los actos reclamados de la ficha, en texto y sin vacíos."""
    return [_s(a) for a in _lista(dem.get("actos")) if _s(a)]


def _actos_limpios(actos) -> list:
    """Los actos para `datos_extra["actos"]` (C5, el punto que confirma los
    nombra): sin los que sólo son hueco, sin el punto final, sin repetir."""
    out = []
    for a in _lista(actos):
        t = _s(a).rstrip(" .;,:").strip()
        if t and re.search(r"[^\W\d_]", re.sub(r"\*+", "", t)) and t not in out:
            out.append(t)
    return out


def _texto_de_la_demanda(q_txt: str, plural_q: bool, aut_crudas: list, actos: list, f_dem,
                         av: _Avisos, ap: str, donde: str, avisar_fecha: bool = True) -> str:
    """«Por escrito presentado el {fecha}, {quejoso} promovió juicio de amparo
    indirecto en contra de la autoridad y el acto que a continuación se
    señalan:» y los bloques AUTORIDAD(ES) RESPONSABLE(S) / ACTO(S)
    RECLAMADO(S), uno por renglón.

    `donde` es el otro papel en que constan («el primer resultando de la
    sentencia recurrida» en la revisión; «el auto recurrido» en la queja). Lo
    que falta de autoridades o de actos va en hueco con su aviso. Sin la fecha,
    la oración abre con quien promovió (menos detalle, no perífrasis); el aviso
    de la fecha sólo si `avisar_fecha` (en la queja la demanda es contexto, no
    requisito: sin aviso)."""
    autoridades = [_mayus(_organo(a)) for a in aut_crudas]
    autoridades = [a for a in autoridades if a]
    if not autoridades:
        autoridades = [av.falta("las autoridades responsables del amparo indirecto",
                                "demanda.autoridades", "están en la demanda de "
                                f"amparo o en {donde}", ap)]
    if not actos:
        actos = [av.falta("los actos reclamados en el amparo indirecto",
                          "demanda.actos", f"están en la demanda de amparo o en {donde}", ap)]
    actos = [(a[:1].upper() + a[1:]).rstrip(" ;,") for a in actos]
    actos = [a if a.endswith((".", HUECO)) else a + "." for a in actos]
    pl_a, pl_c = len(autoridades) > 1, len(actos) > 1
    de_quien = (f"{'las autoridades' if pl_a else 'la autoridad'} y "
                f"{'los actos' if pl_c else 'el acto'}")
    promovio = "promovieron" if plural_q else "promovió"
    if f_dem:
        abre = f"Por escrito presentado el {_letra(f_dem)}, {_cierra_inciso(q_txt)} {promovio}"
    else:
        # MENOS DETALLE, NO PERÍFRASIS (SPEC §2.2): sin la fecha la frase dice
        # quién promovió y contra qué.
        abre = f"{_mayus(_cierra_inciso(q_txt))} {promovio}"
        if avisar_fecha:
            av.add("FALTA LA FECHA DE PRESENTACIÓN DE LA DEMANDA DE AMPARO INDIRECTO "
                   "(demanda.fecha): está en el sello de la demanda o en "
                   f"{donde}. «{ap}» se escribió sin ella.")
    return (f"{abre} juicio de amparo indirecto en contra de {de_quien} que a "
            f"continuación se señalan:\n"
            f"{'AUTORIDADES RESPONSABLES' if pl_a else 'AUTORIDAD RESPONSABLE'}:\n"
            + "\n".join(autoridades) + "\n"
            + f"{'ACTOS RECLAMADOS' if pl_c else 'ACTO RECLAMADO'}:\n"
            + "\n".join(actos))


def _amparo_revision(ficha, datos, av: _Avisos):
    acto = _d(ficha.get("acto"))
    dem = _d(ficha.get("demanda"))
    fp = _d(datos.get("ficha_procesal"))
    num = _numero_asunto(ficha.get("numero") or datos.get("numero"))
    num_t = num or av.falta("el número del amparo en revisión", "numero",
                            "está en el auto de Presidencia que registra el recurso",
                            "V I S T O")
    caracter = _sin_tildes(_s(ficha.get("caracter") or datos.get("papel_recurrente")))
    rec_d = _s(datos.get("recurrente"))
    quejoso = _s(ficha.get("quejoso")) or \
        (_s(ficha.get("promovente")) if caracter in ("quejoso", "quejosa") else "") or \
        _s(datos.get("quejoso")) or _s(_d(fp.get("quejosa")).get("nombre"))
    recurrente = _s(ficha.get("promovente")) or rec_d or \
        _s(_d(fp.get("recurrente")).get("nombre")) or (quejoso if not rec_d else "")
    # LA ETIQUETA DE LA CARÁTULA DICE EL CARÁCTER cuando la ficha no lo trae
    # («GOBERNADOR… (AUTORIDAD RESPONSABLE)», AR 208/2025).
    if not caracter:
        caracter = _sin_rol(recurrente)[1]
    if not caracter and recurrente and quejoso and recurrente == quejoso and not rec_d:
        caracter = "quejoso"
    es_aut = caracter == "autoridad"
    rec_txt = _parte(recurrente, autoridad=es_aut) if recurrente else av.falta(
        "quién interpuso el recurso de revisión", "promovente",
        "está en el escrito de agravios y en el auto de admisión", "V I S T O")
    _aviso_abreviado(recurrente, "promovente", av, "el escrito de agravios")
    _aviso_genero_del_cargo(recurrente, "promovente", av)
    plural = _es_plural(recurrente) if not es_aut else False
    aut_crudas = [a for a in _lista(dem.get("autoridades")) if _s(a)]
    if not aut_crudas and _s(datos.get("responsable")):
        aut_crudas = [datos.get("responsable")]
    juz, juz_crudo, aut_crudas, descartados = _juzgado_del_amparo(acto, datos, fp,
                                                                  aut_crudas, av)
    clase = _clase_recurrida(ficha)
    ap2 = ("Trámite del incidente de suspensión" if clase == "interlocutoria_suspension"
           else "Trámite del juicio de amparo indirecto")
    donde_juz = "está en el proemio de la resolución recurrida"
    if descartados:
        donde_juz += (f"; el que traía la ficha («{descartados[0]}») es una autoridad "
                      "responsable de la demanda —el acto reclamado, no la sentencia de "
                      "amparo—, así que revisa también la fecha de la recurrida (acto.fecha)")
    # EL HUECO LLEGA A TODOS LOS SITIOS (cuarta ronda, E10; AR 307/2024, 448 y
    # 60/2025): el aviso decía «Va en hueco en «V I S T O»» y la competencia y
    # la existencia seguían nombrando el órgano descartado. `organo_recurrido`
    # va en hueco en `datos_extra` para que esos considerandos también lo pongan.
    juz_t = _con_art(juz) if juz else av.falta(
        "el juzgado que dictó la resolución recurrida", "acto.organo", donde_juz,
        # C1 (3-oct-2026): la revisión ya no lleva existencia; el hueco sólo
        # llega a la competencia.
        ["V I S T O", ap2], tambien="la competencia")
    juicio = _solo_numero(acto.get("expediente"))
    juicio_t = juicio or av.falta("el número del juicio de amparo indirecto",
                                  "acto.expediente", "está en el proemio de la "
                                  "resolución recurrida", "V I S T O")
    f_acto_d = _fecha(acto.get("fecha"))
    f_acto = _fecha_o(acto.get("fecha"), av, "la fecha de la resolución recurrida",
                      "acto.fecha", "está en la resolución recurrida", "V I S T O")
    if clase == "interlocutoria_suspension":
        contra = (f"contra la sentencia interlocutoria dictada el {f_acto}, por {juz_t}, "
                  f"en el incidente de suspensión relativo al juicio de amparo "
                  f"indirecto {juicio_t}")
    elif clase == "auto_sobreseimiento":
        contra = (f"contra el auto dictado el {f_acto}, por {juz_t}, en el juicio de "
                  f"amparo indirecto {juicio_t}")
    else:
        contra = (f"contra la sentencia dictada el {f_acto}, por {juz_t}, en el juicio "
                  f"de amparo indirecto {juicio_t}")
    # C6 (3-oct-2026): los relacionados que marcó el secretario, tras el número.
    mat = _materia_clave(ficha.get("materia") or datos.get("materia"))
    rel = _relacionados(ficha, "amparo_revision", num)
    visto = _contraer(f"para resolver el recurso de revisión {num_t}{_relacionado_con(rel, mat)}, "
                      f"interpuesto por {rec_txt}, {contra}; y,")

    res = []
    # 1 · LA DEMANDA DE AMPARO INDIRECTO — las autoridades y los actos, ANCLADOS
    # al papel por `ficha_tramite`; una por renglón.
    ap1 = "Presentación de la demanda de amparo indirecto"
    q_txt = _parte(quejoso) if quejoso else av.falta(
        "el nombre de la parte quejosa del amparo indirecto", "quejoso",
        "está en la demanda de amparo y en el proemio de la sentencia recurrida", ap1)
    if quejoso != recurrente:
        _aviso_abreviado(quejoso, "quejoso", av, "la demanda de amparo")
    plural_q = _es_plural(quejoso)
    actos = _actos_de_la_demanda(dem)
    # LA FECHA DE LA RECURRIDA ESCRITA DENTRO DEL ACTO RECLAMADO es la señal
    # de que se tomó la fecha del acto (AR 307/2024: «La resolución de seis de
    # diciembre de dos mil veintitrés» y la «sentencia» del 6-dic-2023).
    if f_acto_d and actos and \
            _sin_tildes(_letra(f_acto_d)) in _sin_tildes(" ".join(actos)):
        av.add(f"LA FECHA DE LA RESOLUCIÓN RECURRIDA APARECE EN EL ACTO RECLAMADO "
               f"(acto.fecha = {f_acto_d.isoformat()}): puede ser la del acto reclamado y "
               "no la de la sentencia de amparo. Compruébala en la resolución recurrida.")
    t1 = _texto_de_la_demanda(q_txt, plural_q, aut_crudas, actos, _fecha(dem.get("fecha")),
                              av, ap1, "el primer resultando de la sentencia recurrida")
    res.append({"titulo": ap1 + ".", "texto": t1})
    # Las autoridades del renglón, para `responsable` en `datos_extra`.
    autoridades = [a for a in (_mayus(_organo(a)) for a in aut_crudas) if a]

    # 2 · TRÁMITE DEL JUICIO (o del incidente) Y SENTENCIA — el verbo, del enum.
    r = _resolvio(acto, datos)
    if clase == "interlocutoria_suspension":
        ap2 = "Trámite del incidente de suspensión"
        sentido = _s(acto.get("sentido"))
        if r in ("concede", "niega"):
            que = ("se concedió" if r == "concede" else "se negó") + " la suspensión definitiva"
        elif sentido and re.search(r"(?i)suspensi", sentido):
            que = sentido.rstrip(" .")
        else:
            que = av.falta("si se concedió o se negó la suspensión definitiva",
                           "acto.resolvio", "está en los puntos resolutivos de la "
                           "interlocutoria recurrida", ap2)
        t2 = (f"El conocimiento del asunto correspondió a {juz_t}, que lo registró con "
              f"el número {juicio_t} y formó el incidente de suspensión. Seguido el "
              f"incidente por sus etapas, el {f_acto} se celebró la audiencia "
              f"incidental, en la que {que}.")
    elif clase == "auto_sobreseimiento":
        ap2 = "Trámite del juicio de amparo indirecto"
        t2 = (f"El conocimiento del asunto correspondió a {juz_t}, que lo registró con "
              f"el número {juicio_t} y admitió la demanda. Seguido el juicio por sus "
              f"etapas, por auto de {f_acto} se sobreseyó en el juicio fuera de la "
              f"audiencia constitucional.")
    else:
        ap2 = "Trámite del juicio de amparo indirecto"
        if r in _VERBO_AR:
            verbo = _VERBO_AR[r]
        elif r == "mixto":
            verbo = ("en una parte se sobreseyó en el juicio y en otra " +
                     av.falta("si en la otra parte se concedió o se negó el amparo",
                              "acto.resolvio", "está en los puntos resolutivos de "
                              "la sentencia recurrida", ap2) + " el amparo")
        else:
            verbo = av.falta("qué resolvió el juzgado (sobreseyó, concedió o negó)",
                             "acto.resolvio", "está en los puntos resolutivos de la "
                             "sentencia recurrida", ap2)
        aud = _fecha(ficha.get("audiencia"))
        if aud and f_acto_d and aud == f_acto_d:
            etapa = (f"el {f_acto} se celebró la audiencia constitucional y se dictó "
                     f"sentencia")
        elif aud:
            etapa = (f"el {_letra(aud)} se celebró la audiencia constitucional y el "
                     f"{f_acto} se dictó sentencia")
        else:
            etapa = f"el {f_acto} se dictó sentencia"
        t2 = (f"El conocimiento del asunto correspondió a {juz_t}, que lo registró con "
              f"el número {juicio_t} y admitió la demanda. Seguido el juicio por sus "
              f"etapas, {etapa}, en la que {verbo}.")
    res.append({"titulo": ap2 + ".", "texto": _contraer(t2)})

    # 3 · INTERPOSICIÓN Y TRÁMITE DEL RECURSO.
    ap3 = "Interposición y trámite del recurso de revisión"
    pres = _fecha_o(ficha.get("presentacion"), av, "la fecha de presentación del "
                    "recurso de revisión", "presentacion", "está en el acuse o "
                    "sello del escrito de agravios", ap3,
                    respaldo=datos.get("presentacion"))
    car = _caracter_en_prosa(caracter)
    car_txt = f", en su carácter de {car}" if car else ""
    if not car:
        av.add("FALTA EL CARÁCTER PROCESAL DE QUIEN RECURRE (caracter): está en el "
               "escrito de agravios y en el auto de admisión (quejoso, autoridad "
               f"responsable o tercero interesado). «{ap3}» lo omite.")
    info_rep: dict = {}
    rep = _representacion(ficha, promovente=recurrente, av=av, info=info_rep, apartado=ap3)
    f_adm = _fecha_o(_d(ficha.get("admision")).get("fecha"), av,
                     "la fecha del auto de Presidencia que admitió el recurso",
                     "admision.fecha", "está en el auto de Presidencia de este "
                     "Tribunal Colegiado", ap3)
    interpuso = "interpusieron" if plural else "interpuso"
    f_reg = _registro_distinto(ficha, av, ap3)
    tramite = (f"Por auto de Presidencia de {f_reg}, este Tribunal Colegiado lo registró con "
               f"el número {num_t}; y por auto de {f_adm} lo admitió a trámite." if f_reg else
               f"Por auto de Presidencia de {f_adm}, este Tribunal Colegiado lo registró con "
               f"el número {num_t} y lo admitió a trámite.")
    t3 = (f"Inconforme, {rec_txt}{car_txt}{rep}, {interpuso} recurso de revisión por "
          f"escrito presentado el {pres}{_via_del_escrito(ficha, juz_t if juz else '')}. "
          f"{tramite}")
    adh = _d(ficha.get("adhesivo"))
    if _hay_adhesivo(adh):
        f_ad = _fecha_adhesivo("amparo_revision", adh, av, "la fecha del auto que tuvo por "
                               "interpuesta la revisión adhesiva",
                               "está en el auto que la provee", ap3)
        quien = _o(adh.get("quien"), av, "quién interpuso la revisión adhesiva",
                   "adhesivo.quien", "está en el escrito de adhesión", ap3)
        t3 += (f"\nPor auto de {f_ad}, se tuvo a "
               f"{_parte(quien) if quien != HUECO else quien} interponiendo "
               f"revisión adhesiva.")
    mp = _ministerio_publico(ficha.get("ministerio_publico"))
    if mp:
        t3 += "\n" + mp
    res.append({"titulo": ap3 + ".", "texto": _contraer(t3)})

    # 4 · TURNO (art. 92) y 5 · RETURNO.
    res += _turno_y_returno(ficha, datos, av,
                            ", en términos del artículo 92 de la Ley de Amparo")

    extra = {
        "expediente": juicio, "toca": "", "juicio_amparo": juicio, "juzgado": juz,
        "fecha_acto": f_acto if f_acto != HUECO else "",
        "fecha_acto_iso": f_acto_d.isoformat() if f_acto_d else "",
        # EN LA REVISIÓN «responsable» ES LA DEL ACTO RECLAMADO (la primera de
        # la demanda); todas van en `autoridades`. El órgano recurrido es el
        # juzgado (`juzgado`).
        "organo_acto": juz, "responsable": autoridades[0] if autoridades and
        autoridades[0] != HUECO else "",
        "autoridades": [o for o in (_organo(a) for a in aut_crudas) if o],
        # C5 (3-oct-2026, David: «sí está bien que se precise… información
        # valiosa para el lector»): el punto que confirma nombra el acto y la
        # autoridad (`tipos_asunto.punto_del_amparo`) o remite al resultando que
        # copia la demanda, que en la revisión es siempre el primero.
        "actos": _actos_limpios(actos), "resultando_demanda": "primero",
        "materia": mat,
        "numero": num, "clase_recurrida": clase, "resolvio": r,
        "promovente_en_prosa": rec_txt if rec_txt != HUECO else "",
        "quejoso_en_prosa": q_txt if q_txt != HUECO else "",
        # E10: la competencia y la existencia nombran el MISMO órgano que el
        # V I S T O; si se descartó (era una autoridad responsable) y no hubo
        # otra fuente, va en hueco también ahí. `juzgado_descartado` dice cuál.
        "organo_recurrido": juz or HUECO,
        "juzgado_descartado": descartados[0] if (descartados and not juz) else "",
        # E2: quien recurre y la quejosa, cada uno con su número.
        "plural": plural, "plural_quejoso": plural_q,
        # E8: la autoridad que recurre por conducto de su propio titular.
        "recurrente_nombre": info_rep.get("titular", ""),
        "clase": {"interlocutoria_suspension": "interlocutoria",
                  "auto_sobreseimiento": "auto"}.get(clase, "sentencia"),
        "descripcion_acto": {"interlocutoria_suspension":
                             "de la sentencia interlocutoria dictada en el incidente "
                             "de suspensión",
                             "auto_sobreseimiento":
                             "del auto que sobreseyó en el juicio fuera de la "
                             "audiencia constitucional"}.get(clase, "de la sentencia recurrida"),
        # C6: el rubro y el considerando de conexidad o de hecho notorio.
        "relacionados": rel,
    }
    return visto, res, extra


# ═══════════════════════════════════════════════════════════════════════════
# QUEJA
# ═══════════════════════════════════════════════════════════════════════════
# LA FRACCIÓN DEL 97 SÓLO CON CERTEZA (quinta ronda, 3-oct-2026, F3; Q 335/2025
# reproducido): `_RX_DISTRITO` buscaba «juez… distrito» y casaba con «Juez
# Segundo de lo Civil del DISTRITO JUDICIAL de Querétaro», la responsable típica
# de la queja de la fracción II; salían la vía, la fracción y el fundamento
# equivocados en cuatro apartados, y el aviso del plazo empujaba a recontar con
# dos días un recurso que tiene cinco. Ahora: «de Distrito» —no «del Distrito
# Judicial»— (o un Tribunal Colegiado de Apelación, un Unitario de Circuito, un
# juez federal) es la I; un juzgado local (primera instancia, de lo civil,
# familiar o mercantil, «Distrito Judicial», oralidad, menor, mixto) o una Sala
# es la II; si no se sabe, «» y el hueco con su aviso, sin suponer la I.
_RX_DISTRITO = re.compile(r"(?i)\bju(?:ez|eza|zgado)\b[^;]*?\bde\s+distrito\b(?!\s+judicial)|"
                          r"tribunal\s+colegiado\s+de\s+apelaci|tribunal\s+unitario\s+de\s+circuito|"
                          r"centro\s+de\s+justicia\s+penal\s+federal|\bju(?:ez|eza|zgado)\s+federal")
_RX_RESP_DIRECTO = re.compile(r"(?i)\bsala\b|tribunal\s+superior|tribunal\s+federal\s+de\s+justicia|"
                              r"\bjunta\b|tribunal\s+de\s+justicia\s+administrativa|"
                              r"tribunal\s+unitario\s+agrario|tribunal\s+laboral|"
                              r"primera\s+instancia|distrito\s+judicial|"
                              r"\bde\s+lo\s+(?:civil|familiar|mercantil|penal)\b|oralidad|"
                              r"\bju(?:ez|eza|zgado)\s+(?:\S+\s+){0,2}?(?:civil|familiar|mercantil|mixto|"
                              r"menor|de\s+paz)\b")

# EL VERBO DEL CATÁLOGO, EN IMPERSONAL. El sentido llega del catálogo de la
# ficha en pretérito y con su sujeto implícito («desechó la demanda de
# amparo»); la cola de la procedencia lo dice en impersonal («por el cual se
# desechó…»), que es la forma del corpus (tipos_asunto.COLA_97).
_PRETERITOS = ("desechó", "tuvo", "admitió", "concedió", "negó", "declaró", "reconoció",
               "sobreseyó", "impuso", "requirió", "ordenó", "resolvió", "dejó", "decretó",
               "determinó", "revocó", "modificó", "confirmó", "fijó", "omitió", "rehusó")


def _inicial_minuscula(t: str) -> str:
    """«Dejó sin efectos…» → «dejó sin efectos…»; «IMSS…» se queda."""
    if not t:
        return t
    primera = t.split(" ", 1)[0].strip(".,;:")
    if len(primera) > 1 and primera.isupper():
        return t
    return t[:1].lower() + t[1:]


def _impersonal(sentido: str) -> str:
    s = _s(sentido).rstrip(" .")
    if not s:
        return ""
    primera = s.split(" ", 1)[0].lower()
    return f"se {s[:1].lower()}{s[1:]}" if primera in _PRETERITOS else s


def _fraccion_97(ficha, organo) -> str:
    f = _s(ficha.get("fraccion_97")).upper().replace("FRACCIÓN", "").replace("FR.", "").strip()
    if f in ("I", "II"):
        return f
    via = _sin_tildes(_s(_d(ficha.get("acto")).get("via")))
    if via in ("indirecto", "directo"):
        return "I" if via == "indirecto" else "II"
    if organo and _RX_DISTRITO.search(organo):
        return "I"
    if organo and _RX_RESP_DIRECTO.search(organo):
        return "II"
    return ""


# ── EL ÓRGANO DE LA QUEJA, EN FORMA DE ÓRGANO (3-oct-2026, Q 261, 300, 337 y 342/2025) ──
# Los 8 engroses de queja nombran en la competencia al JUZGADO («dictado por el
# Juzgado Séptimo de Distrito…, localizado en la circunscripción»), porque se
# predica su ubicación; nosotros copiábamos la forma de persona —«el Juez
# Sexto…», «el Titular del Juzgado Cuarto…»— en el V I S T O, el resultando, la
# competencia, la procedencia y la carátula. Y cuando la ficha traía dos
# nombres, se tomaba el del acto aunque fuera el incompleto («Juez Sexto de
# Distrito de Amparo y Juicios Federales» frente al oficial «Juzgado Sexto de
# Distrito en Materia de Amparo Civil, Administrativo y de Trabajo y de Juicios
# Federales en el Estado de Querétaro»). Ahora: forma de órgano y, entre dos
# nombres del mismo órgano, el más completo, con aviso si difieren.
_ORDINAL_MASCULINO = {
    "primera": "primero", "segunda": "segundo", "tercera": "tercero", "cuarta": "cuarto",
    "quinta": "quinto", "sexta": "sexto", "septima": "séptimo", "octava": "octavo",
    "novena": "noveno", "decima": "décimo", "undecima": "undécimo",
    "duodecima": "duodécimo", "vigesima": "vigésimo", "trigesima": "trigésimo",
    "especializada": "especializado",
}
_RX_FORMA_ORGANO = re.compile(r"(?i)^(?:\S+\s+){0,2}?(?:juzgado|tribunal|sala|junta|centro|pleno)\b")


def _forma_de_organo(t: str) -> str:
    """«Juez Séptimo de Distrito…» → «Juzgado Séptimo de Distrito…»; «Jueza
    Tercera de Distrito…» → «Juzgado Tercero de Distrito…»; «Titular del
    Juzgado Cuarto…» → «Juzgado Cuarto…»; «Magistrada Presidenta de la Primera
    Sala…» → «Primera Sala…». Lo demás, como viene."""
    n = _s(t)
    m = re.match(r"(?i)^titular\s+del\s+(?=juzgado|tribunal)", n)
    if m:
        n = n[m.end():]
        return n[:1].upper() + n[1:]
    m = re.match(r"(?i)^(?:magistrad[oa]s?|presidente|presidenta)(?:\s+(?:presidente|presidenta))?"
                 r"\s+de\s+(?:la\s+|el\s+)?(?=(?:\S+\s+){0,2}?(?:sala|tribunal|junta)\b)", n)
    if m:
        n = n[m.end():]
        return n[:1].upper() + n[1:]
    m = re.match(r"(?i)^juez(a)?\s+", n)
    if not m:
        return n
    palabras = n[m.end():].split(" ")
    al_principio = True     # el ordinal va pegado al cargo: «Jueza Décima Primera…»
    for i, w in enumerate(palabras):
        clave = _sin_tildes(w).strip(".,;:")
        if clave in _ORDINAL_MASCULINO and (al_principio or clave == "especializada"):
            nuevo = _ORDINAL_MASCULINO[clave]
            palabras[i] = (nuevo[:1].upper() + nuevo[1:]) if w[:1].isupper() else nuevo
        else:
            al_principio = False
    return "Juzgado " + " ".join(palabras)


def _nucleo_organo(t: str) -> str:
    pal = [w for w in _clave_nombre(t).split()
           if w not in ("de", "del", "en", "y", "e", "materia", "materias")]
    return " ".join(pal[:3])


def _organo_de_la_queja(acto: dict, ficha: dict, datos: dict, av: _Avisos) -> tuple:
    """(órgano en forma de órgano y en prosa, su valor crudo)."""
    # LA RESPONSABLE DEL FORMULARIO YA NO ES EL ÓRGANO (integración, 3-oct-2026,
    # consecuencia de C3): sin el renglón del órgano en la carátula, la pantalla
    # manda en `responsable` la ordenadora leída del auto de admisión. Sólo
    # cuenta si tiene la forma del órgano de la queja
    # (`tipos_asunto.responsable_es_el_organo`); si no, ni compite ni da el
    # aviso de «dos órganos distintos».
    _resp_enc = datos.get("responsable")
    try:
        import tipos_asunto as _ta_q
        if not _ta_q.responsable_es_el_organo("queja", _s(_resp_enc), _fraccion_97(ficha, "")):
            _resp_enc = ""
    except Exception:
        pass
    vistos = []
    for crudo, fuente in ((acto.get("organo"), "acto.organo"),
                          (ficha.get("responsable"), "responsable"),
                          (datos.get("organo_recurrido"), "organo_recurrido"),
                          (_resp_enc, "responsable del encargo")):
        o = _forma_de_organo(_organo(crudo)) if _s(crudo) else ""
        if o and not any(_clave_nombre(o) == _clave_nombre(v[0]) for v in vistos):
            vistos.append((o, fuente, _s(crudo)))
    if not vistos:
        return "", ""
    pool = [v for v in vistos if _RX_FORMA_ORGANO.search(v[0])] or vistos
    elegido = pool[0]
    for v in pool[1:]:
        mas_largo = len(_clave_nombre(v[0])) > len(_clave_nombre(elegido[0]))
        if mas_largo and (_contiene(v[0], elegido[0]) or
                          _nucleo_organo(v[0]) == _nucleo_organo(elegido[0])):
            elegido = v
    if len(vistos) > 1:
        lista = "; ".join(f"{f}: «{o}»" for o, f, _c in vistos)
        if all(_nucleo_organo(o) == _nucleo_organo(elegido[0]) or _contiene(o, elegido[0])
               for o, _f, _c in vistos):
            av.add(f"EL ÓRGANO QUE DICTÓ EL AUTO RECURRIDO VIENE ESCRITO DE DOS FORMAS "
                   f"({lista}): se escribió «{elegido[0]}», el nombre completo del órgano. "
                   "Compruébalo en el auto recurrido.")
        else:
            av.add(f"LA FICHA TRAE DOS ÓRGANOS DISTINTOS PARA EL AUTO RECURRIDO ({lista}): "
                   f"se escribió «{elegido[0]}». Comprueba en el auto recurrido quién lo "
                   "dictó.")
    return elegido[0], elegido[2]


def _queja(ficha, datos, av: _Avisos):
    acto = _d(ficha.get("acto"))
    num = _numero_asunto(ficha.get("numero") or datos.get("numero"))
    num_t = num or av.falta("el número del recurso de queja", "numero",
                            "está en el auto de Presidencia que registra el recurso",
                            "V I S T O")
    mat = _materia_clave(ficha.get("materia") or datos.get("materia"))
    mat_t = _materia_concordada(mat, "f") or av.falta(
        "la materia de la queja", "materia", "está en el auto de Presidencia (la "
        "materia del juicio de amparo)", "V I S T O")
    caracter = _sin_tildes(_s(ficha.get("caracter") or datos.get("papel_recurrente")))
    recurrente = _s(ficha.get("promovente")) or _s(datos.get("recurrente")) or \
        _s(datos.get("quejoso"))
    if not caracter:
        caracter = _sin_rol(recurrente)[1]
    es_aut = caracter == "autoridad"
    rec_txt = _parte(recurrente, autoridad=es_aut) if recurrente else av.falta(
        "quién interpuso la queja", "promovente", "está en el escrito de queja",
        "V I S T O")
    _aviso_abreviado(recurrente, "promovente", av, "el escrito de queja")
    _aviso_genero_del_cargo(recurrente, "promovente", av)
    plural = _es_plural(recurrente) if not es_aut else False
    organo, _org_crudo = _organo_de_la_queja(acto, ficha, datos, av)
    # El hueco del órgano va en el V I S T O, en la interposición y en los dos
    # considerandos que lo nombran (cuarta ronda, Q 335/2025: el aviso sólo
    # decía «V I S T O»).
    org_t = _con_art(organo) if organo else av.falta(
        "el órgano que dictó el auto recurrido", "acto.organo",
        "está en el auto recurrido", ["V I S T O", "Interposición del recurso de queja"],
        tambien="la competencia y en la procedencia")
    fr = _fraccion_97(ficha, organo)
    if fr:
        via = "directo" if fr == "II" else "indirecto"
    else:
        # NI «INDIRECTO» POR OMISIÓN: decirlo sería afirmar la vía sin dato.
        via = HUECO
        av.add("FALTA LA FRACCIÓN DEL ARTÍCULO 97 (fraccion_97): sale del auto "
               "recurrido y de la vía —I si lo dictó un Juzgado de Distrito en "
               "amparo indirecto, II si lo dictó la responsable en amparo directo—; "
               f"el órgano («{organo or 'no consta'}») no lo dice con certeza. La vía "
               "del juicio («indirecto» o «directo») va en hueco en el V I S T O y en "
               "«Interposición del recurso de queja», y la fracción y la vía también "
               "en la competencia y en la procedencia. El plazo del recurso depende "
               "de la fracción (art. 98 LA): fíjala antes de contar.")
    exp = _solo_numero(acto.get("expediente"))
    exp_t = exp or av.falta("el número del juicio de amparo", "acto.expediente",
                            "está en el auto recurrido", "V I S T O")
    # EN EL AMPARO DIRECTO NO HAY INCIDENTE DE SUSPENSIÓN. La suspensión la
    # provee la propia responsable (arts. 190 y 191 de la Ley de Amparo), sin
    # incidente formal, y la queja de la fracción II se interpone contra ese
    # auto «en el juicio de amparo directo». Escribir «en el incidente de
    # suspensión relativo al juicio de amparo directo» era inventar un
    # cuaderno que no existe (considerandos.txt, Q fr. II integrado). La marca
    # `acto.incidente` sólo cuenta en la fracción I (amparo indirecto).
    inc = bool(acto.get("incidente")) and fr != "II"
    f_acto_d = _fecha(acto.get("fecha"))
    f_acto = _fecha_o(acto.get("fecha"), av, "la fecha del auto recurrido",
                      "acto.fecha", "está en el auto recurrido", "V I S T O")
    rel = "incidente de suspensión relativo al juicio" if inc else "juicio"
    # C6 (3-oct-2026): los relacionados que marcó el secretario, tras el número.
    relacionados = _relacionados(ficha, "queja", num)
    visto = _contraer(
        f"para resolver el recurso de queja {mat_t} {num_t}"
        f"{_relacionado_con(relacionados, mat)}, interpuesto por "
        f"{rec_txt}, en contra del auto de {f_acto}, dictado por {org_t}, en el "
        f"{rel} de amparo {via} {exp_t}; y,")

    res = []
    # 0 · LA DEMANDA DE AMPARO INDIRECTO (C3, 3-oct-2026) — sólo en la fracción
    # I y sólo si la ficha la trae; si no, la queja abre como siempre, sin aviso.
    t_dem, actos_dem = _demanda_de_la_queja(ficha, datos, fr, caracter, recurrente,
                                            organo, _org_crudo, av)
    if t_dem:
        res.append({"titulo": _AP_DEMANDA_QUEJA + ".", "texto": t_dem})

    # 1 · INTERPOSICIÓN — estilo de la ponencia de David: abre con ella cuando
    # no hay demanda que contar.
    ap1 = "Interposición del recurso de queja"
    pres = _fecha_o(ficha.get("presentacion"), av, "la fecha de presentación de la "
                    "queja", "presentacion", "está en el acuse o sello del escrito "
                    "de queja", ap1, respaldo=datos.get("presentacion"))
    # EL SENTIDO A MEDIA FRASE VA CON MINÚSCULA (quinta ronda, 3-oct-2026, Q
    # 335/2025): la extracción traía «Dejó sin efectos…» y salía «en el que
    # Dejó sin efectos la suspensión», cuando el mismo documento decía «del
    # auto que dejó sin efectos…». Una sigla al principio se respeta.
    sentido = _inicial_minuscula(_s(acto.get("sentido")).rstrip(" ."))
    sentido_t = sentido or av.falta("qué proveyó el auto recurrido", "acto.sentido",
                                    "está en la parte final del auto recurrido", ap1)
    car = _caracter_en_prosa(caracter)
    info_rep: dict = {}
    rep = _representacion(ficha, promovente=recurrente, av=av, info=info_rep, apartado=ap1)
    juicio = f"juicio de amparo {via} {exp_t}"
    interpuso = "interpusieron" if plural else "interpuso"
    por_escrito = f"Por escrito presentado el {pres}{_via_del_escrito(ficha)}"
    if car:
        t1 = (f"{por_escrito}, {rec_txt}, {car} en el {juicio}{rep}, "
              f"{interpuso} recurso de queja en contra del auto de {f_acto}, dictado por "
              f"{org_t}{', en el incidente de suspensión' if inc else ''}, en el que "
              f"{sentido_t}.")
    else:
        av.add("FALTA EL CARÁCTER PROCESAL DE QUIEN RECURRE (caracter): está en el "
               "escrito de queja y en el auto admisorio de la demanda (quejoso, "
               f"autoridad responsable o tercero interesado). «{ap1}» lo omite.")
        sujeto = f"{rec_txt}{rep}," if rep else _cierra_inciso(rec_txt)
        if fr == "II":
            # La responsable dicta el auto en el juicio de amparo directo que
            # está en trámite ante este Tribunal: «relativo al juicio…».
            donde_q = f", relativo al {juicio}"
        else:
            donde_q = (f" en el {'incidente de suspensión relativo al ' if inc else ''}"
                       f"{juicio}")
        t1 = (f"{por_escrito}, {sujeto} {interpuso} recurso de queja en "
              f"contra del auto de {f_acto}, dictado por {org_t}{donde_q}, en el "
              f"que {sentido_t}.")
    res.append({"titulo": ap1 + ".", "texto": _contraer(t1)})

    # 2 · TRÁMITE — fr. I: registra y admite; fr. II: registra y requiere el
    # informe con justificación (art. 101) y después lo tiene por rendido y
    # admite, en dos autos (cuarta ronda, E7: el primero es el REGISTRO).
    ap2 = "Trámite del recurso"
    if fr == "II":
        t2 = _tramite_queja_ii(ficha, av, num_t, ap2)
    else:
        f_adm = _fecha_o(_d(ficha.get("admision")).get("fecha"), av,
                         "la fecha del auto de Presidencia que admitió el recurso",
                         "admision.fecha", "está en el auto de Presidencia de este "
                         "Tribunal Colegiado", ap2)
        f_reg = _registro_distinto(ficha, av, ap2)
        if f_reg:
            t2 = (f"Por auto de Presidencia de {f_reg}, este Tribunal Colegiado registró el "
                  f"recurso con el número {num_t}; y por auto de {f_adm} lo admitió a "
                  f"trámite.")
        else:
            t2 = (f"Por auto de Presidencia de {f_adm}, este Tribunal Colegiado registró el "
                  f"recurso con el número {num_t} y lo admitió a trámite.")
    mp = _ministerio_publico(ficha.get("ministerio_publico"))
    if mp:
        t2 += "\n" + mp
    res.append({"titulo": ap2 + ".", "texto": t2})
    if _hay_adhesivo(_d(ficha.get("adhesivo"))):
        av.add("LA QUEJA NO ADMITE ADHESIÓN: la ficha trae un adhesivo y se ignoró. "
               "Revisa de qué asunto viene ese dato.")

    # 3 · TURNO DEL ASUNTO — sin artículo: el 101 regula el trámite, no el turno.
    res += _turno_y_returno(ficha, datos, av, "", rotulo_turno="Turno del asunto.")

    # LA FRACCIÓN, EL INCISO Y SU COLA, UNA SOLA VEZ (oro_Q: ~30 de 404 sentencias
    # traen incisos distintos en la competencia y en la procedencia).
    inciso = _s(ficha.get("inciso_97")).lower().strip(" )")
    if not inciso and fr == "I" and sentido:
        try:
            import tipos_asunto as _ta
            inciso = _ta.inciso_97(sentido)
        except Exception:
            inciso = ""
    if not inciso:
        av.add("FALTA EL INCISO DEL ARTÍCULO 97 (inciso_97): sale de qué proveyó el "
               "auto recurrido. La competencia y la procedencia lo llevarán en hueco.")
    # LA COLA SALE DEL SENTIDO, NO DE LA TABLA. `tipos_asunto.COLA_97["a"]` dice
    # «se desechó la demanda» para todo el inciso a), que también es admitir o
    # tener por no presentada; con el sentido del catálogo la cola dice lo que
    # proveyó ESTE auto. Sin sentido va vacía y `documento_generado` usa la
    # tabla con su fracción (`cola_97(inciso, fraccion)`): en `datos_extra` van
    # datos, no huecos.
    cola = ""
    if sentido and inciso:
        imp = _impersonal(sentido)
        cola = (f"por el cual {imp}" if (fr == "I" and inciso == "a")
                else f"en el que {imp}")
    extra = {
        "expediente": exp, "toca": "", "juicio_amparo": exp, "juzgado": organo,
        "fecha_acto": f_acto if f_acto != HUECO else "",
        "fecha_acto_iso": f_acto_d.isoformat() if f_acto_d else "",
        "organo_acto": organo, "responsable": organo,
        "materia": mat, "numero": num,
        "fraccion_97": fr, "inciso_97": inciso, "cola_97": cola,
        "via_amparo": via, "incidente": inc, "clase": "auto",
        "promovente_en_prosa": rec_txt if rec_txt != HUECO else "",
        "plural": plural,
        # EL NÚMERO REAL DE LA QUEJOSA aunque recurra otra parte (3-oct-2026,
        # integración): con varios quejosos y un tercero recurrente la carátula
        # decía «QUEJOSA» en singular.
        "plural_quejoso": (plural if caracter in ("quejoso", "quejosa")
                           else _es_plural(_s(ficha.get("quejoso")) or _s(datos.get("quejoso")))),
        "recurrente_nombre": info_rep.get("titular", ""),
        "descripcion_acto": f"del auto que {sentido[:1].lower() + sentido[1:]}" if sentido else "",
        # C3: los actos de la demanda y el resultando que la copia, si se escribió.
        "actos": actos_dem, "resultando_demanda": "primero" if t_dem else "",
        # C6: el rubro y el considerando de conexidad o de hecho notorio.
        "relacionados": relacionados,
    }
    return visto, res, extra


_AP_DEMANDA_QUEJA = "Demanda de amparo"


def _legible(nombre: str) -> bool:
    """¿Hay un nombre que escribir? Ni vacío ni sólo huecos o testado."""
    return bool(re.search(r"[^\W\d_]", re.sub(r"\*+", "", _s(nombre))))


def _demanda_de_la_queja(ficha: dict, datos: dict, fr: str, caracter: str, recurrente: str,
                         organo: str, org_crudo: str, av: _Avisos) -> tuple:
    """(texto del resultando «Demanda de amparo.», actos limpios) o ("", []).

    C3, 3-oct-2026 (David: «lo más práctico… que el secretario no tenga que
    modificar; que nuestro estilo sea bastante bueno»). La queja de la fracción
    I nace en un amparo indirecto y el auto recurrido —el desechamiento, la
    suspensión— suele describir la demanda; `ficha_tramite` la lee (`leer_acto`
    y `leer_escrito` con tipo «queja») y la ancla al papel con las mismas
    guardas que el amparo en revisión. Si la trae —al menos las autoridades o
    los actos—, la queja abre con ella, con la mecánica del primer resultando
    de la revisión (`_texto_de_la_demanda`):
      · sin la fecha, «{Quejoso} promovió…», sin hueco ni aviso: la demanda es
        contexto, no requisito;
      · lo que falte de autoridades o de actos, en hueco con su aviso: con la
        otra mitad a la vista, el secretario sabe dónde buscarla;
      · sin quejoso legible el resultando no se escribe (no se abre con un
        hueco un párrafo que es sólo contexto);
      · la fracción II no lo lleva (el amparo es directo y la demanda ya está
        en este tribunal), ni la fracción que no consta;
      · el propio juzgado que dictó el auto, fuera de las autoridades
        (`ficha_tramite` ya lo quita al leer; esto es el respaldo)."""
    if fr != "I":
        return "", []
    dem = _d(ficha.get("demanda"))
    aut_crudas = [a for a in _lista(dem.get("autoridades")) if _legible(a)]
    actos = [a for a in _actos_de_la_demanda(dem) if _legible(a)]
    juz = _s(org_crudo) or _s(organo)
    fuera = []
    if aut_crudas and juz and (_RX_ORGANO_AMPARO.search(juz) or _RX_ORGANO_AMPARO.search(organo or "")):
        fuera = [a for a in aut_crudas if _contiene(a, juz) or (organo and _contiene(a, organo))]
        aut_crudas = [a for a in aut_crudas if a not in fuera]
    if not aut_crudas and not actos:
        return "", []
    es_quejoso = caracter in ("quejoso", "quejosa")
    quejoso = _s(ficha.get("quejoso")) or (recurrente if es_quejoso else "") or \
        _s(datos.get("quejoso"))
    q_txt = _parte(quejoso) if _legible(quejoso) else ""
    if not _legible(q_txt):
        return "", []
    # Los avisos, sólo si el resultando se escribe (si no, no hay de qué avisar).
    if fuera:
        av.add("EL JUZGADO QUE DICTÓ EL AUTO RECURRIDO ESTABA ENTRE LAS AUTORIDADES "
               f"RESPONSABLES DE LA DEMANDA (demanda.autoridades: «{_s(fuera[0])}»): se quitó "
               f"de «{_AP_DEMANDA_QUEJA}». Comprueba las autoridades en la demanda.")
    if quejoso != recurrente:
        _aviso_abreviado(quejoso, "quejoso", av, "la demanda de amparo")
    # EL AUTO QUE PROVEE SOBRE LA AMPLIACIÓN (revisión Q, 3-oct-2026; Q 300/2025
    # del banco: «la Jueza de Distrito admitió a trámite la ampliación de
    # demanda»). El auto describe la ampliación y lo que el modelo copie de ella
    # pasa el anclaje —está en el papel— y la cronología —es anterior al auto—:
    # el resultando diría que la demanda se presentó en la fecha de la
    # ampliación y contra su acto, sin un solo aviso. La ficha ya descarta la
    # fecha de la ampliación (`ficha_tramite._fecha_de_ampliacion`); las
    # autoridades y los actos no se pueden separar sin el modelo, así que se
    # escribe y se avisa.
    _acto_q = _d(ficha.get("acto"))
    _sent_q = " ".join(str(_acto_q.get(k) or "") for k in ("sentido_clave", "sentido"))
    if re.search(r"ampliaci", _sin_tildes(_sent_q).lower()):
        av.add("EL AUTO RECURRIDO PROVEE SOBRE UNA AMPLIACIÓN DE DEMANDA (acto.sentido_clave): "
               f"comprueba que «{_AP_DEMANDA_QUEJA}» describa la demanda INICIAL —su fecha, sus "
               "autoridades y sus actos— y no la ampliación, que el auto también describe.")
    texto = _texto_de_la_demanda(q_txt, _es_plural(quejoso), aut_crudas, actos,
                                 _fecha(dem.get("fecha")), av, _AP_DEMANDA_QUEJA,
                                 "el auto recurrido", avisar_fecha=False)
    return texto, _actos_limpios(actos)


def _tramite_queja_ii(ficha: dict, av: _Avisos, num_t: str, ap2: str) -> str:
    """El trámite de la queja de la fracción II, en sus dos autos.

    CUARTA RONDA (3-oct-2026, E7; Q 335/2025): el primer auto —el de
    Presidencia que forma y registra el expediente y requiere el informe con
    justificación (art. 101 LA)— es el REGISTRO (`registro.fecha`), no la
    admisión. Antes se escribía `admision.fecha` en los dos autos y salía el 3
    de noviembre como fecha del registro, del requerimiento y de la admisión,
    cuando el registro fue el 14 de octubre. Sin el registro, hueco y aviso;
    la fecha de la admisión nunca se escribe en los dos autos. El segundo auto
    es el que tiene por rendido el informe (`informe_101.fecha`) y admite (o
    `admision.fecha`, si es el único que consta)."""
    reg_v = _d(ficha.get("registro")).get("fecha")
    inf_v = _d(ficha.get("informe_101")).get("fecha")
    reg, inf = _fecha(reg_v), _fecha(inf_v)
    adm = _fecha(_d(ficha.get("admision")).get("fecha"))
    # EL REGISTRO CON LA FECHA DEL INFORME O DE LA ADMISIÓN NO ES EL REGISTRO
    # (quinta ronda, 3-oct-2026; Q 335/2025 reproducido): con registro =
    # informe = admisión salía «Por auto de Presidencia de tres de noviembre…
    # registró… y requirió…; por auto de tres de noviembre… lo tuvo por
    # rendido», la misma fecha en los dos autos, que es lo que E7 prohíbe. En
    # la fracción II son dos autos distintos y el que registra es ANTERIOR: si
    # la ficha le da la fecha de otro, va en hueco con su aviso y su pista.
    repetida = ""
    if reg and ((inf and reg == inf) or (adm and reg == adm)):
        repetida = reg.isoformat()
        reg = None
    if reg:
        f_reg = _letra(reg)
    elif _s(reg_v) and not repetida:
        f_reg = _fecha_o(reg_v, av, "", "registro.fecha", "", ap2)
    else:
        pista = ""
        if repetida:
            pista = (f" La ficha le daba al registro la misma fecha que al "
                     f"{'informe' if (inf and inf.isoformat() == repetida) else 'auto que admite'} "
                     f"({repetida}): son dos autos, y el que registra y pide el informe es "
                     "anterior. Corrige «Auto de Presidencia que forma y registra».")
        elif adm and (not inf or adm < inf):
            pista = (f" Si la fecha que la ficha trae como admisión ({adm.isoformat()}) es la "
                     "de ese auto, pásala a «Auto de Presidencia que forma y registra».")
        av.add("FALTA LA FECHA DEL AUTO DE PRESIDENCIA QUE REGISTRÓ EL RECURSO Y REQUIRIÓ EL "
               "INFORME CON JUSTIFICACIÓN (registro.fecha): en la queja de la fracción II es "
               "otro auto, anterior al que tiene por rendido el informe y admite el recurso "
               f"(art. 101 LA). Va en hueco en «{ap2}».{pista}")
        f_reg = HUECO
    if inf and adm and adm > inf:
        segundo = (f"por auto de {_letra(inf)} lo tuvo por rendido y por auto de "
                   f"{_letra(adm)} admitió el recurso")
    elif inf or adm:
        segundo = f"por auto de {_letra(inf or adm)} lo tuvo por rendido y admitió el recurso"
    else:
        f2 = _fecha_o(inf_v, av, "la fecha del auto que tuvo por rendido el informe con "
                      "justificación y admitió el recurso", "informe_101.fecha",
                      "está en el auto que tiene por rendido el informe y admite el recurso",
                      ap2)
        segundo = f"por auto de {f2} lo tuvo por rendido y admitió el recurso"
    return (f"Por auto de Presidencia de {f_reg}, este Tribunal Colegiado registró el "
            f"recurso con el número {num_t} y requirió a la autoridad responsable su "
            f"informe con justificación sobre la materia de la queja (artículo 101 de la "
            f"Ley de Amparo); {segundo}.")


# ═══════════════════════════════════════════════════════════════════════════
# REVISIÓN FISCAL
# ═══════════════════════════════════════════════════════════════════════════
_RX_TFJA = re.compile(r"(?i)tribunal\s+federal\s+de\s+justicia\s+administrativa")


def _sala_renombrada(sala: str) -> str:
    """«Sala Regional del Centro II, del Tribunal Federal de Justicia
    Administrativa, ahora Sala Regional en Querétaro del Tribunal Federal de
    Justicia Administrativa, con sede en esta ciudad» (RF 21/2025) → «Sala
    Regional del Centro II, ahora Sala Regional en Querétaro del Tribunal
    Federal de Justicia Administrativa»: el Tribunal una vez, y sin «con sede
    en esta ciudad», que en la carátula y en el resolutivo no remite a nada."""
    t = re.sub(r"(?i),?\s+con\s+(?:sede|residencia)\s+en\s+esta\s+ciudad\b", "", _s(sala))
    m = re.search(r"(?i),?\s+(?:ahora|hoy|actualmente)\b", t)
    if m and _RX_TFJA.search(t[m.end():]):
        t = re.sub(r"(?i),?\s+del\s+tribunal\s+federal\s+de\s+justicia\s+administrativa(?=,?\s+(?:ahora|hoy|actualmente)\b)",
                   "", t)
    return _s(t).rstrip(" ,")


def _sala_del_tfja(sala: str) -> str:
    """«la Sala Regional en Querétaro del Tribunal Federal de Justicia
    Administrativa», sin repetir el Tribunal si la fuente ya lo trae."""
    if not sala:
        return ""
    s = _con_art(sala)
    return s if _RX_TFJA.search(s) else f"{s} del Tribunal Federal de Justicia Administrativa"


# ── EL SENTIDO DE LA SENTENCIA DE LA SALA, DEL CATÁLOGO (SPEC §6; 3-oct-2026, rev_5) ──
# El compositor escribía `acto.sentido` tal cual: «en la que declaró la nulidad
# para efectos.» (RF 2, 7, 21 y 49/2025). Las frases son las de
# `ficha_tramite.CATALOGO_SENTIDO["revision_fiscal"]` (la prueba comprueba que
# no se separen); la clave se reconoce por las palabras que la definen.
_SENTIDO_RF = {
    "nulidad_lisa": "declaró la nulidad lisa y llana de la resolución impugnada",
    "nulidad_efectos": "declaró la nulidad de la resolución impugnada, para determinados efectos",
    "validez": "reconoció la validez de la resolución impugnada",
    "sobresee": "sobreseyó en el juicio",
}
_SOBRESEE_PARCIAL = "sobreseyó parcialmente en el juicio"
_NULIDAD_SIN_CALIFICAR = "declaró la nulidad de la resolución impugnada"
# LA NULIDAD PARA EFECTOS QUE ADEMÁS RECONOCE UN DERECHO (art. 52, fr. V, a),
# LFPCA; quinta ronda, 3-oct-2026, RF 6/2026): la clave que `ficha_tramite`
# añade a su catálogo. Si el catálogo trae otra frase para ella, manda la del
# catálogo (`_frases_sentido_rf`).
_NULIDAD_DERECHO = ("declaró la nulidad de la resolución impugnada, para determinados efectos, "
                    "y reconoció a la parte actora un derecho subjetivo")
# LO QUE EL SENTIDO TRAE DE MÁS QUE EL CATÁLOGO NO SE TIRA (RF 6/2026: «…para
# efectos y reconoció el derecho subjetivo al incremento de la cuota
# pensionaria y al pago retroactivo de las diferencias» salía «…para
# determinados efectos.», sin el reconocimiento, que además decide la
# procedencia por la fracción VI). La cola «y reconoció … derecho…» o «y
# condenó…» se conserva tras la frase del catálogo, con aviso.
_RX_COLA_SENTIDO_RF = re.compile(
    r"(?i),?\s+(?:y|e)\s+(?:(?:se\s+)?reconoci[óo]\s+(?:a\s+(?:la\s+)?(?:parte\s+)?actora\s+)?"
    r"(?:el|un|los|su|sus)\s+derechos?\b|conden[óo]\b).*$")


def _frases_sentido_rf() -> dict:
    """Las frases del sentido: las de `ficha_tramite.CATALOGO_SENTIDO
    ["revision_fiscal"]` (la fuente) sobre las de aquí (el respaldo)."""
    frases = dict(_SENTIDO_RF, nulidad_derecho=_NULIDAD_DERECHO)
    try:
        import ficha_tramite as _ft
        cat = _d(getattr(_ft, "CATALOGO_SENTIDO", {})).get("revision_fiscal")
        for k, v in _d(cat).items():
            if isinstance(v, (list, tuple)) and len(v) > 1 and isinstance(v[1], str) and v[1]:
                frases[k] = v[1]
    except Exception:
        pass
    return frases


def _sentido_rf(acto: dict, av: _Avisos, apartado: str) -> str:
    sentido = _s(acto.get("sentido")).rstrip(" .")
    m_cola = _RX_COLA_SENTIDO_RF.search(sentido)
    cola = m_cola.group(0).strip(" ,") if m_cola else ""
    frase = _sentido_rf_catalogo(acto, sentido, av, apartado)
    if not cola or frase == sentido or HUECO in frase or _sin_tildes(cola) in _sin_tildes(frase):
        return frase
    # LA COLA «y reconoció…» VIENE DEL PAPEL (ficha_tramite.cola_derecho_subjetivo,
    # copiada del resolutivo de la Sala): se conserva sin aviso (3-oct-2026).
    if re.match(r"(?i)y\s+reconoci", cola) and not re.search(r"(?i)\breconoci[óo]\b", frase):
        return f"{frase}, {cola}"
    if re.search(r"(?i)\breconoci[óo]\b.*\bderecho", frase) and \
            re.search(r"(?i)\breconoci[óo]\b.*\bderecho", cola):
        # El catálogo ya dice que reconoció un derecho: se queda el del papel.
        frase = re.sub(r",?\s+y\s+reconoci[óo]\b.*$", "", frase)
    av.add("EL SENTIDO TRAE MÁS DE LO QUE DICE EL CATÁLOGO (acto.sentido = "
           f"«{_recorte(sentido)}»): se escribió la frase del catálogo y se conservó "
           f"«{_recorte(cola, 120)}». Compruébalo en los puntos resolutivos de la sentencia de la "
           "Sala; si reconoció un derecho subjetivo, la nulidad es de fondo.")
    return f"{frase}, {cola}"


def _sentido_rf_catalogo(acto: dict, sentido: str, av: _Avisos, apartado: str) -> str:
    """La frase del catálogo para el sentido (la regla de la tercera ronda)."""
    frases = _frases_sentido_rf()
    clave = _sin_tildes(_s(acto.get("sentido_clave")))
    claves = clave.split("+") if clave else []
    if claves and all(k in frases for k in claves):
        fr = [frases[k] for k in claves]
        if len(fr) == 1:
            return fr[0]
        if len(fr) == 2 and claves[0] == "sobresee":
            return f"{_SOBRESEE_PARCIAL} y {fr[1]}"
    if not (claves and all(k in _SENTIDO_RF for k in claves)):
        t = _sin_tildes(sentido)
        claves = []
        if re.search(r"\bsobrese", t):
            claves.append("sobresee")
        lisa = bool(re.search(r"\blisa\s+y\s+llana\b", t))
        efectos = bool(re.search(r"\bpara\s+(?:los\s+|el\s+|determinados\s+|ciertos\s+)?efectos?\b", t))
        if lisa and not efectos:
            claves.append("nulidad_lisa")
        elif efectos and not lisa:
            claves.append("nulidad_efectos")
        elif not lisa and re.search(r"\bnulidad\b", t):
            claves.append("nulidad")
        if re.search(r"\bvalidez\b", t):
            claves.append("validez")
        if lisa and efectos:
            claves = ["?"]
    frases = [(_SENTIDO_RF.get(k) or (_NULIDAD_SIN_CALIFICAR if k == "nulidad" else ""))
              for k in claves]
    if len(frases) == 1 and frases[0]:
        return frases[0]
    if len(frases) == 2 and claves[0] == "sobresee" and frases[1]:
        return f"{_SOBRESEE_PARCIAL} y {frases[1]}"
    if sentido:
        av.add("EL SENTIDO DE LA SENTENCIA RECURRIDA NO ESTÁ EN EL CATÁLOGO (acto.sentido = "
               f"«{sentido}»): se escribió como vino. Debe ser nulidad lisa y llana, "
               "nulidad para efectos, validez o sobreseimiento; compruébalo en los puntos "
               "resolutivos de la sentencia de la Sala.")
        return sentido
    return av.falta("el sentido de la sentencia recurrida (nulidad, validez o "
                    "sobreseimiento)", "acto.sentido", "está en los puntos resolutivos de "
                    "la sentencia recurrida", apartado)


# ── QUIÉN INTERPONE LA REVISIÓN FISCAL (3-oct-2026, RF 2, 7, 21, 6/2025 y 6/2026) ──
# El art. 63 LFPCA da el recurso a la unidad encargada de la defensa jurídica
# de la autoridad demandada. La ficha lo trae de tres maneras y el resultando
# salía mal en las tres:
#   · el promovente ya dice a quién representa («JEFA DE LA UNIDAD JURÍDICA…,
#     EN REPRESENTACIÓN DEL SUBDELEGADO…»), y se le añadía otra vez «, en
#     representación del Subdelegado…» (3 de 8);
#   · recurre la propia autoridad demandada por conducto de su unidad, y salía
#     «el Jefe del Departamento…, en representación del Jefe del Departamento…»
#     (2 de 8), sin la figura «Titular de la Unidad de Asuntos Jurídicos» que
#     la ficha sí traía (RF 6/2026);
#   · «X, por conducto de la Titular de la Unidad Jurídica»: X es la autoridad
#     representada y la unidad es quien firma.
_RX_CONECTOR_RF = re.compile(
    r"(?i),?\s+(?P<modo>en\s+representaci[óo]n|en\s+nombre|por\s+conducto)\s+de(?:l)?\s+"
    r"(?:(?P<art>la|el|los|las)\s+)?")


def _art_del_papel(x) -> str:
    """El artículo con que el papel abre el nombre: «la», «el» o «»."""
    m = re.match(r"(?i)^\s*(el|la)\s+", _s(x))
    return m.group(1).lower() if m else ""


def _partir_promovente_rf(promovente: str) -> tuple:
    """(unidad, representada, artículo del papel para la unidad, modo) del texto
    del promovente; el artículo es «la», «el» o «» (cuarta ronda: «la Titular»
    y «el Titular» se distinguen de «Titular» a secas, que va con «el» y con
    aviso); modo es «representacion», «conducto» o «» (sin conector)."""
    p, _rol = _sin_rol(_s(promovente))
    m = _RX_CONECTOR_RF.search(p)
    if not m or not p[:m.start()].strip() or not p[m.end():].strip():
        return p, "", _art_del_papel(p), ""
    izq, der = p[:m.start()].strip(" ,"), p[m.end():].strip(" ,")
    if _sin_tildes(m.group("modo")).startswith("por"):
        # «X, por conducto de [la] Y»: X es la representada, Y la unidad.
        art = _sin_tildes(m.group("art") or "")
        return der, izq, art if art in ("el", "la") else "", "conducto"
    return izq, der, _art_del_papel(izq), "representacion"


# ── EL CARGO Y SU ÓRGANO SON LA MISMA AUTORIDAD (cuarta ronda, 3-oct-2026, E8) ──
# RF 6/2026: el promovente era «el Subdelegado de Prestaciones Económicas…» y la
# autoridad demandada «la Subdelegación de Prestaciones Económicas…» —el engrose
# usa las dos formas—, y salía «el Subdelegado…, en representación de la
# Subdelegación…»: X en representación de X. Para comparar, el cargo se
# escribe como su órgano y «Titular de la X» como «X».
_CARGO_A_ORGANO = {
    "subdelegado": "subdelegacion", "subdelegada": "subdelegacion",
    "delegado": "delegacion", "delegada": "delegacion",
    "director": "direccion", "directora": "direccion",
    "subdirector": "subdireccion", "subdirectora": "subdireccion",
    "jefe": "jefatura", "jefa": "jefatura", "subjefe": "subjefatura", "subjefa": "subjefatura",
    "coordinador": "coordinacion", "coordinadora": "coordinacion",
    "administrador": "administracion", "administradora": "administracion",
    "tesorero": "tesoreria", "tesorera": "tesoreria",
    "procurador": "procuraduria", "procuradora": "procuraduria",
    "contralor": "contraloria", "contralora": "contraloria",
    "presidente": "presidencia", "presidenta": "presidencia",
    "gerente": "gerencia", "gerenta": "gerencia",
    "secretario": "secretaria", "subsecretario": "subsecretaria",
    "recaudador": "recaudacion", "recaudadora": "recaudacion",
    "oficial": "oficialia", "sindico": "sindicatura",
}


# EL NOMBRE LARGO Y LA SIGLA SON EL MISMO INSTITUTO (RF 6/2026 del banco: el
# promovente decía «…del Instituto de Seguridad y Servicios Sociales de los
# Trabajadores del Estado» y la autoridad demandada «…del ISSSTE en Querétaro»,
# y salía otra vez «el Subdelegado…, en representación de la Subdelegación…»).
_SIGLAS_DE_INSTITUTOS = (
    (re.compile(r"\binstituto de seguridad y servicios sociales de (?:los )?trabajadores del estado\b"),
     "issste"),
    (re.compile(r"\binstituto mexicano del seguro social\b"), "imss"),
    (re.compile(r"\bservicio de administracion tributaria\b"), "sat"),
    (re.compile(r"\binstituto del fondo nacional de (?:la )?vivienda para (?:los )?trabajadores\b"),
     "infonavit"),
    (re.compile(r"\bcomision nacional del agua\b"), "conagua"),
    (re.compile(r"\bcomision federal de electricidad\b"), "cfe"),
)


def _clave_autoridad(x) -> str:
    """`_clave_nombre` con el cargo como su órgano, sin «Titular de» y con el
    nombre largo de los institutos como su sigla."""
    k = re.sub(r"^titular\s+del?\s+", "", _clave_nombre(x))
    for rx, sigla in _SIGLAS_DE_INSTITUTOS:
        k = rx.sub(sigla, k)
    ws = k.split()
    if ws and ws[0] in _CARGO_A_ORGANO:
        ws[0] = _CARGO_A_ORGANO[ws[0]]
    return " ".join(ws)


def _figura_es_la_unidad(fig: str, unidad: str) -> bool:
    """¿La figura nombra el MISMO cargo que encabeza a quien promueve?
    «Jefa de la Unidad Jurídica» frente a «Jefa de la Unidad Jurídica de la
    Delegación Estatal…» (RF 7/2025), «Subdirector de lo Contencioso» frente a
    «Subdirector de lo Contencioso del ISSSTE» (RF 26/2025), «Titular» frente
    a «Titular de la Unidad Jurídica». Entonces quien firma ES su titular.

    QUINTA RONDA (3-oct-2026, F4; RF 7/2025 reproducido con nombre real): la
    figura decía «…de la REPRESENTACIÓN Estatal del Instituto…» y el
    promovente «…de la DELEGACIÓN Estatal…» —la misma unidad con su nombre
    viejo y el nuevo—; aquí salía False y la legitimación (que compara por
    núcleo) True, y el documento decía «la Jefa…, por conducto de su Jefa…»
    en el resultando y «lo hizo valer Fulana, Jefa…» en la legitimación.
    Decide `_misma_autoridad`, que es `tipos_asunto.misma_autoridad`."""
    if not (_s(fig) and _s(unidad)):
        return False
    if _misma_autoridad(fig, unidad):
        return True
    for clave in (_clave_autoridad, _clave_nombre):
        kf, ku = clave(fig), clave(unidad)
        if kf and (ku == kf or ku.startswith(kf + " ")):
            return True
    return False


_RX_TRUNCADO = re.compile(r"…|\.\.\.")


def _es_cargo(x) -> bool:
    """¿El texto nombra un CARGO o una unidad, y no a una persona?"""
    return _es_autoridad(x) or bool(re.search(r"(?i)\bunidad\b", _s(x)))


def _depurar_representacion_rf(ficha: dict, av: _Avisos) -> tuple:
    """(representante, figura) de la ficha de la revisión fiscal, sin los
    datos que delatan su propio defecto (cuarta ronda, 3-oct-2026, E8; RF 2,
    4/2025 y 6/2026, que el documento escribía como hechos firmados):
      · un valor TRUNCADO con «…» («Titular de la Unidad Jurídica…») se
        descarta, con aviso;
      · un representante que no es una persona sino un CARGO («Titular de la
        Unidad de Asuntos Jurídicos de la citada delegación») pasa a figura;
      · un «autorizado» que interpone por la autoridad se avisa: el 63 de la
        LFPCA exige la unidad encargada de su defensa jurídica, y los
        autorizados del 5o. son de los particulares."""
    rep = _s(ficha.get("representante"))
    fig = _s(ficha.get("figura_representante"))
    for clave, valor in (("figura_representante", fig), ("representante", rep)):
        if valor and _RX_TRUNCADO.search(valor):
            av.add(f"DATO TRUNCADO EN LA FICHA ({clave} = «{valor}»): viene cortado con «…» "
                   "y se descartó. Escríbelo completo como lo dice el oficio de interposición.")
            if clave == "figura_representante":
                fig = ""
            else:
                rep = ""
    if rep and _es_cargo(rep):
        av.add(f"EL REPRESENTANTE ES UN CARGO, NO UNA PERSONA (representante = «{rep}»): se "
               "escribió como la figura de quien firmó"
               + (f", en lugar de «{fig}»" if fig else "")
               + ". Si el oficio da el nombre de quien firmó, escríbelo en la ficha.")
        fig, rep = rep, ""
    if fig and re.match(r"(?i)^\s*(?:su\s+)?autorizad[oa]s?\b", fig):
        # UN SOLO TRATO PARA EL AUTORIZADO (quinta ronda, 3-oct-2026; RF 4/2025
        # con nombres de prueba): el resultando lo escribía («…, por conducto de
        # su autorizado…, Pedro Gil Mora, en representación…») y la legitimación
        # (`tipos_asunto._legitimacion_rf`) lo omitía: el mismo documento daba
        # dos versiones de quién interpuso. Fuera de los dos, con el aviso.
        av.add(f"UN AUTORIZADO NO INTERPONE LA REVISIÓN FISCAL POR LA AUTORIDAD "
               f"(figura_representante = «{fig}»): el artículo 63 de la LFPCA exige que la "
               "interponga la unidad administrativa encargada de su defensa jurídica. Se omitió "
               "del resultando y de la legitimación"
               + (f" (representante = «{rep}»)" if rep else "")
               + ". Compruébalo en el oficio; puede ser el autorizado de la actora, que va en la "
               "adhesiva.")
        rep, fig = "", ""
    return rep, fig


def _aviso_titular_generico(nombre_org: str, crudo: str, clave: str, av: _Avisos) -> None:
    """«Titular» sin artículo en el papel: va «el Titular» (masculino genérico
    del corpus) y se avisa, porque quien firma puede ser una mujer (RF 4,
    6/2025, 21 y 49/2025: «la Titular de la Unidad Jurídica»).

    QUINTA RONDA (F5): igual con cualquier cargo de doble género («Oficial»,
    «Fiscal», «Agente»…), con el aviso «EL GÉNERO DEL CARGO NO CONSTA». El de
    «Titular» conserva su texto: `documento_generado` deduplica por él."""
    if _articulo_del_papel(crudo):
        return
    primera = _primera_del_cargo(nombre_org)
    if primera == "titular":
        av.add(f"«EL TITULAR» VA POR OMISIÓN ({clave} = «{_recorte(crudo)}»): el papel no dice si "
               "firma una mujer o un hombre y el cargo se escribió en masculino genérico. Si "
               "firma una mujer, corrígelo en la ficha («la Titular…»).")
    elif primera in _CARGOS_DOBLE_GENERO:
        av.add(f"EL GÉNERO DEL CARGO NO CONSTA ({clave} = «{_recorte(crudo)}»): el papel no dice "
               f"«el» ni «la» y se escribió «{_con_art(_s(nombre_org))}», en masculino genérico. "
               "El género lo dice el papel, nunca el nombre de quien ocupa el cargo; si es una "
               "mujer, escribe «la …» en la ficha.")


def _aviso_genero_del_cargo(crudo, clave: str, av: _Avisos) -> None:
    """El aviso de `_aviso_titular_generico` para el nombre de una PARTE que el
    documento escribe como autoridad (con su artículo): la responsable, el
    tercero, la autoridad que recurre, la demandada (F5, quinta ronda). La
    autoridad que emitió la resolución impugnada no es parte y no se avisa."""
    n = _sin_rol(_s(crudo))[0]
    if not n or not _es_autoridad(n):
        return
    _aviso_titular_generico(_organo(n), n, clave, av)


def _misma_autoridad(a: str, b: str) -> bool:
    """¿La unidad que recurre ES la autoridad demandada? Iguales, o una dentro
    de la otra con la misma cabeza («Jefe del Departamento de Pensiones» y «Jefe
    del Departamento de Pensiones… de la Delegación…»). «Unidad Jurídica de la
    Delegación X» contiene a «Delegación X» y NO es ella: la cabeza cambia.
    El cargo y su órgano cuentan como la misma («Subdelegado…» y
    «Subdelegación…», «Titular de la X» y «X»: cuarta ronda, E8).

    UNA SOLA COMPARACIÓN (quinta ronda, 3-oct-2026, F4): la decide
    `tipos_asunto.misma_autoridad` —por núcleo, sin la adscripción—, la misma
    que usan la legitimación y la verja; el resultando y la legitimación no
    pueden volver a contradecirse sobre quién firmó (RF 7/2025). La regla de
    abajo sólo responde si esa pieza no se puede llamar."""
    try:
        import tipos_asunto as _ta
        _f = getattr(_ta, "misma_autoridad", None)
        if callable(_f):
            return bool(_f(_s(a), _s(b)))
    except Exception:
        pass
    return _misma_autoridad_local(a, b)


def _misma_autoridad_local(a: str, b: str) -> bool:
    """El respaldo de `_misma_autoridad` (la regla de la cuarta ronda)."""
    ka, kb = _clave_autoridad(a), _clave_autoridad(b)
    if not (ka and kb):
        return False
    if ka == kb:
        return True
    if (f" {kb} " in f" {ka} " or f" {ka} " in f" {kb} ") and ka.split()[:3] == kb.split()[:3]:
        return True
    # LA MISMA AUTORIDAD ESCRITA CON UNA PALABRA DE MÁS (RF 6/2026): «Subdelegado
    # de Prestaciones Económicas, Delegación Estatal Querétaro, del ISSSTE» y
    # «Subdelegado de Prestaciones, Delegación Estatal Querétaro, del ISSSTE».
    # Misma cabeza y las palabras de una dentro de las de la otra.
    vacias = {"de", "del", "y", "e", "en", "a"}
    ta = [w for w in ka.split() if w not in vacias]
    tb = [w for w in kb.split() if w not in vacias]
    if len(ta) < 3 or len(tb) < 3 or ta[:2] != tb[:2]:
        return False
    return set(ta) <= set(tb) or set(tb) <= set(ta)


def _quien_recurre_rf(ficha: dict, recurrente: str, aut_ri: str, apartado: str,
                      av: _Avisos) -> tuple:
    """(sujeto de «Inconforme, {sujeto}, interpuso…», unidad normalizada,
    autoridad demandada normalizada, info) — `info` lleva lo que la
    legitimación necesita para nombrar a quien firmó igual que el resultando:
    «unidad_con_articulo» («la Titular de la Unidad Jurídica…», «el Subdelegado…»),
    «nombre» (la persona que firmó como titular, en prosa, o «») y
    «la_titular» (el papel dice «la Titular»).

    CUARTA RONDA (3-oct-2026, E8; RF 7, 2 y 26/2025): si quien firma es la
    titular del mismo cargo que encabeza al promovente («Jefa de la Unidad
    Jurídica…» con figura «Jefa de la Unidad Jurídica»), no se escribe «por
    conducto de su Jefa de la Unidad Jurídica, Fulana»: el resultando nombra la
    unidad sola, como el corpus, y el nombre de la titular va en `info` para la
    legitimación («lo hizo valer Fulana, Jefa de la Unidad Jurídica…»)."""
    info = {"unidad_con_articulo": "", "nombre": "", "la_titular": False}
    unidad_cr, repr_cr, art_u, modo = _partir_promovente_rf(recurrente)
    dem_cr = _s(ficha.get("autoridad_demandada"))
    if dem_cr and _RX_TRUNCADO.search(dem_cr):
        av.add(f"DATO TRUNCADO EN LA FICHA (autoridad_demandada = «{dem_cr}»): viene cortado "
               "con «…» y se descartó. Escríbelo completo como lo dice la sentencia recurrida.")
        dem_cr = ""
    if repr_cr and dem_cr and not _misma_autoridad(_organo(repr_cr), _organo(dem_cr)):
        av.add("LA AUTORIDAD REPRESENTADA NO CUADRA ENTRE FUENTES (autoridad_demandada = "
               f"«{dem_cr}»; el oficio dice «{repr_cr}»): se escribió la de la ficha. "
               "Compruébalo en el oficio de interposición.")
    if not dem_cr:
        dem_cr = repr_cr
    dem = _organo(dem_cr)
    if not dem and aut_ri:
        dem, dem_cr = aut_ri, aut_ri
        av.add("SE TOMÓ COMO AUTORIDAD DEMANDADA LA QUE EMITIÓ LA RESOLUCIÓN IMPUGNADA "
               "(autoridad_demandada vacía): compruébalo en el oficio de interposición, "
               "que dice a quién representa la unidad jurídica.")
    dem_t = _con_art(dem, art_papel=_articulo_del_papel(dem_cr)) if dem else ""
    if dem:
        _aviso_genero_del_cargo(dem_cr, "autoridad_demandada", av)
    rep_cr, fig_cr = _depurar_representacion_rf(ficha, av)
    # «X, POR CONDUCTO DE LA TITULAR DE LA UNIDAD…»: la unidad es la figura.
    por_conducto = modo == "conducto"
    if por_conducto and not fig_cr:
        fig_cr = (f"{art_u} " if art_u else "") + unidad_cr
        unidad_cr = ""
    misma = bool(dem) and (por_conducto or (unidad_cr and _misma_autoridad(unidad_cr, dem)))
    # VARIAS AUTORIDADES QUE RECURREN ELLAS MISMAS (revisión RF, 3-oct-2026): el
    # sujeto es plural («Inconformes, el Secretario…, el Jefe… y el Titular…,
    # por conducto de…, interpusieron…»). Si recurre su unidad en su
    # representación, el sujeto es la unidad y sigue en singular.
    try:
        import tipos_asunto as _ta_pl
        info["sujeto_plural"] = bool(misma) and _ta_pl.varias_autoridades(dem_cr or dem)
    except Exception:
        info["sujeto_plural"] = False
    if not unidad_cr and not misma:
        return (av.falta("la unidad jurídica que interpuso el recurso", "promovente",
                         "está en el oficio de interposición", apartado),
                "", dem, info)
    if misma:
        if fig_cr and _figura_es_la_unidad(fig_cr, dem_cr or dem):
            # LA FIGURA ES LA PROPIA DEMANDADA: firmó su titular. Nada de «X,
            # por conducto de su X»; el 63 pide la unidad de defensa jurídica, y
            # eso se avisa para que lo compruebe quien firma.
            av.add(f"LA FIGURA DEL REPRESENTANTE ES LA PROPIA AUTORIDAD DEMANDADA "
                   f"(figura_representante = «{fig_cr}»): se escribió «{dem_t}» sin «por "
                   "conducto de». El artículo 63 de la LFPCA pide que interponga la unidad "
                   "encargada de su defensa jurídica; compruébalo en el oficio.")
            info.update(unidad_con_articulo=dem_t, la_titular=_la_titular(dem_cr),
                        nombre=_prosa_nombre(rep_cr) if rep_cr else "")
            return _contraer(dem_t), dem, dem, info
        # RECURRE LA PROPIA DEMANDADA POR CONDUCTO DE SU UNIDAD JURÍDICA: el
        # sujeto es la demandada y la unidad va «por conducto de». Sin la figura,
        # hueco: el art. 63 exige que firme la unidad de defensa jurídica.
        fig = _organo(fig_cr) if fig_cr else ""
        generica = bool(fig) and not _es_autoridad(fig)
        if generica:
            # UNA FIGURA GENÉRICA («representante legal», «apoderado», «delegado»,
            # RF 21/2025) no es un órgano: «por conducto de su representante legal».
            fig = ""
            fig_t = f"su {_figura_en_prosa(fig_cr)}"
        elif fig:
            fig_t = _con_art(fig, art_papel=_articulo_del_papel(fig_cr))
            _aviso_titular_generico(fig, fig_cr, "figura_representante", av)
            info.update(unidad_con_articulo=fig_t, la_titular=_la_titular(fig_cr))
        else:
            fig_t = av.falta("la unidad jurídica que firmó el oficio por la autoridad "
                             "demandada (art. 63 LFPCA)", "figura_representante",
                             "está en el oficio de interposición, junto a la firma",
                             apartado)
        rep = _prosa_nombre(rep_cr) if rep_cr and not _contiene(dem, rep_cr) else ""
        sujeto = f"{dem_t}, por conducto de {fig_t}" + (
            (f" {rep}" if generica else f", {rep}") if rep else "")
        info["nombre"] = rep
        return _contraer(sujeto), fig, dem, info
    unidad = _organo(unidad_cr)
    la_tit = art_u == "la" or _la_titular(unidad_cr)
    unidad_t = _con_art(unidad, art_papel=art_u or _articulo_del_papel(unidad_cr))
    if not art_u:
        _aviso_titular_generico(unidad, unidad_cr, "promovente", av)
    info_rep: dict = {}
    rep = _representacion(ficha, promovente=unidad_cr, av=av, rep_fig=(rep_cr, fig_cr),
                          info=info_rep, sin_nombre=False, ley_de_amparo=False)
    info.update(unidad_con_articulo=unidad_t, la_titular=la_tit,
                nombre=info_rep.get("titular", ""))
    dem_txt = dem_t or av.falta("la autoridad demandada a la que representa la recurrente",
                                "autoridad_demandada", "está en el oficio de interposición",
                                apartado)
    return _contraer(f"{unidad_t}{rep}, en representación de {dem_txt}"), unidad, dem, info


def _revision_fiscal(ficha, datos, av: _Avisos):
    acto = _d(ficha.get("acto"))
    num = _numero_asunto(ficha.get("numero") or datos.get("numero"))
    num_t = num or av.falta("el número de la revisión fiscal", "numero",
                            "está en el auto de Presidencia que registra el recurso",
                            "V I S T O")
    # LA SALA QUE CAMBIÓ DE NOMBRE (quinta ronda, 3-oct-2026; RF 21 y 49/2025):
    # las sentencias de 2024 y principios de 2025 salían atribuidas a la «Sala
    # Regional en Querétaro» porque `ficha.sala` trae el nombre de hoy, cuando
    # el acto dice «la Sala Regional del Centro II, ahora Sala Regional en
    # Querétaro…». Si el órgano del acto dice «ahora», «hoy» o «actualmente»,
    # manda él: nombra a la que dictó la sentencia y a la de hoy.
    _org_acto = _s(acto.get("organo"))
    if _org_acto and re.search(r"(?i)\b(?:ahora|hoy|actualmente)\b", _org_acto):
        sala = _sala_renombrada(_organo(_org_acto))
    else:
        # LA RESPONSABLE DEL FORMULARIO SÓLO SI ES UNA SALA (integración,
        # 3-oct-2026, consecuencia de C4): sin el renglón «SALA RESPONSABLE»
        # la pantalla manda ahí la ordenadora leída del auto de admisión, que
        # puede ser la autoridad demandada; con «del Tribunal Federal de
        # Justicia Administrativa» añadido por `_sala_del_tfja` salía una
        # autoridad hacendaria como si fuera la Sala.
        _resp_rf = _s(datos.get("responsable"))
        try:
            import tipos_asunto as _ta_rf
            if not _ta_rf.responsable_es_el_organo("revision_fiscal", _resp_rf):
                _resp_rf = ""
        except Exception:
            pass
        sala = _organo(ficha.get("sala") or acto.get("organo") or
                       datos.get("organo_recurrido") or _resp_rf)
    # LOS CUATRO SITIOS DEL HUECO (revisión RF, 3-oct-2026): el aviso decía sólo
    # «Va en hueco en «V I S T O»» y la Sala falta también en el resultando del
    # juicio, en la competencia y en el resolutivo; la verja, con este aviso ya
    # dado, callaba los demás, y quien corregía un sitio dejaba tres huecos.
    sala_t = _sala_del_tfja(sala) or av.falta(
        "la Sala del TFJA que dictó la sentencia recurrida", "sala",
        "está en el proemio de la sentencia recurrida",
        ["V I S T O", "Trámite del juicio contencioso administrativo"],
        tambien="la competencia y el punto resolutivo")
    exp = _s(ficha.get("expediente_tfja") or acto.get("expediente"))
    exp_t = exp or av.falta("el expediente del juicio contencioso administrativo",
                            "expediente_tfja", "está en el proemio de la sentencia "
                            "recurrida", "V I S T O")
    f_acto_d = _fecha(acto.get("fecha"))
    f_acto = _fecha_o(acto.get("fecha"), av, "la fecha de la sentencia recurrida",
                      "acto.fecha", "está en la sentencia recurrida", "V I S T O")
    # C6 (3-oct-2026): los relacionados que marcó el secretario, tras el número.
    # La materia de la revisión fiscal es la administrativa (la de su amparo
    # directo relacionado, por el 64 de la LFPCA): la de la ficha si la trae
    # y, si no, «administrativa» (integración, 3-oct-2026: la misma que toman
    # el rubro y el considerando de `documento_generado`).
    relacionados = _relacionados(ficha, "revision_fiscal", num)
    _mat_rf = _materia_clave(ficha.get("materia") or datos.get("materia")) or "administrativa"
    visto = _contraer(
        f"para resolver el recurso de revisión fiscal número {num_t}"
        f"{_relacionado_con(relacionados, _mat_rf)}, interpuesto por "
        f"la parte citada al rubro, contra la sentencia de {f_acto}, dictada por "
        f"{sala_t}, en el juicio contencioso administrativo {exp_t}; y,")

    res = []
    # 1 · EL JUICIO DE NULIDAD — oficio o fecha que falten se OMITEN de la frase
    # con aviso; la sentencia y su sentido son obligatorios.
    ap1 = "Trámite del juicio contencioso administrativo"
    actora = _s(ficha.get("actora")) or _s(datos.get("tercero"))
    act_t = _mayus(_cierra_inciso(_parte(actora))) if actora else av.falta(
        "la parte actora del juicio de nulidad", "actora",
        "está en el proemio de la sentencia recurrida", ap1)
    _aviso_abreviado(actora, "actora", av)
    plural_act = _es_plural(actora)
    ri = _d(ficha.get("resolucion_impugnada"))
    oficio = _s(ri.get("oficio"))
    f_ri = _fecha(ri.get("fecha"))
    aut_ri_cr = _s(ri.get("autoridad"))
    aut_ri = _organo(aut_ri_cr)
    aut_ri_art = _con_art(aut_ri, art_papel=_articulo_del_papel(aut_ri_cr)) if aut_ri else ""
    negativa = _sin_tildes(_s(ri.get("clase"))).replace(" ", "_").replace("-", "_") in (
        "negativa_ficta", "negativa")
    if negativa:
        # LA NEGATIVA FICTA NO TIENE OFICIO NI FECHA (3-oct-2026, RF 21/2025 y RF
        # 4/2025): con el esquema de la resolución expresa salía «demandó la
        # nulidad de la resolución.» y tres avisos que pedían un oficio y una
        # fecha que no existen.
        materia = _s(ri.get("materia")).rstrip(" .")
        trozos = "la resolución negativa ficta recaída a su solicitud"
        # SIN «SOLICITUD DE SOLICITUD» (quinta ronda, 3-oct-2026, RF 4/2025): la
        # materia llega como la teclea el secretario —«solicitud de
        # incorporación al sistema de jubilación…»— y salía «recaída a su
        # solicitud de solicitud de incorporación…». Si la materia ya nombra el
        # escrito («solicitud», «escrito», «petición», «instancia»), se escribe
        # «recaída a su {materia}».
        m_esc = re.match(r"(?i)^(?:su\s+)?(?=(?:solicitud|escrito|petici[óo]n|instancia)\b)", materia)
        if materia and m_esc:
            trozos = "la resolución negativa ficta recaída a su " + materia[m_esc.end():]
        elif materia:
            trozos += (" " if re.match(r"(?i)^(?:de|del|sobre|relativa|para|por)\b", materia)
                       else " de ") + materia
        else:
            av.add("FALTA QUÉ SE PIDIÓ EN LA SOLICITUD NO CONTESTADA (resolucion_impugnada."
                   "materia): está en el primer resultando de la sentencia recurrida. Se "
                   f"omitió de «{ap1}».")
        if aut_ri:
            trozos += f", atribuida a {aut_ri_art}"
        else:
            av.add("FALTA LA AUTORIDAD A LA QUE SE ATRIBUYE LA NEGATIVA FICTA "
                   "(resolucion_impugnada.autoridad): está en el primer resultando de la "
                   f"sentencia recurrida. Se omitió de «{ap1}».")
    elif not (oficio or f_ri or aut_ri):
        # SIN NINGÚN DATO, UN HUECO Y UN AVISO (no tres avisos y una frase coja).
        trozos = "la resolución " + av.falta(
            "los datos de la resolución impugnada (oficio, fecha y autoridad que la "
            "emitió; o «negativa_ficta» si no la hubo)", "resolucion_impugnada",
            "están en el primer resultando de la sentencia recurrida", ap1)
    else:
        trozos = "la resolución"
        if oficio:
            trozos += f" contenida en el oficio {oficio}"
        else:
            av.add("FALTA EL OFICIO DE LA RESOLUCIÓN IMPUGNADA (resolucion_impugnada.oficio): "
                   "está en el primer resultando de la sentencia recurrida. Se omitió de "
                   f"«{ap1}».")
        if f_ri:
            trozos += f", de {_letra(f_ri)}"
        else:
            av.add("FALTA LA FECHA DE LA RESOLUCIÓN IMPUGNADA (resolucion_impugnada.fecha): "
                   "está en el primer resultando de la sentencia recurrida. Se omitió de "
                   f"«{ap1}».")
        if aut_ri:
            trozos += f", emitida por {aut_ri_art}"
        else:
            av.add("FALTA LA AUTORIDAD QUE EMITIÓ LA RESOLUCIÓN IMPUGNADA "
                   "(resolucion_impugnada.autoridad): está en el primer resultando de la "
                   f"sentencia recurrida. Se omitió de «{ap1}».")
    sent_t = _sentido_rf(acto, av, ap1)
    demando = "demandaron" if plural_act else "demandó"
    t1 = (f"{act_t} {demando} la nulidad de {trozos}. El conocimiento correspondió a "
          f"{sala_t}, que lo registró con el número {exp_t}; seguido el juicio, el "
          f"{f_acto} dictó sentencia, en la que {sent_t}.")
    res.append({"titulo": ap1 + ".", "texto": _contraer(t1)})

    # 2 · INTERPOSICIÓN — ante la Sala (art. 63 LFPCA); por correo, las dos
    # fechas: depósito (la del cómputo, D3 de FIXES_R3) y recepción.
    ap2 = "Interposición del recurso de revisión fiscal"
    recurrente = _s(ficha.get("promovente")) or _s(datos.get("recurrente")) or \
        _s(datos.get("quejoso"))
    sujeto, unidad, dem, info_rf = _quien_recurre_rf(ficha, recurrente, aut_ri, ap2, av)
    pres = _fecha_o(ficha.get("presentacion"), av, "la fecha de presentación (o de "
                    "recepción) del oficio de revisión", "presentacion", "está en el "
                    "sello de la Sala sobre el oficio", ap2,
                    respaldo=datos.get("presentacion"))
    via = _sin_tildes(_s(ficha.get("via_presentacion")))
    dep = _fecha(ficha.get("deposito_postal"))
    # EL CORREO LO DECIDE UNA SOLA REGLA (cuarta ronda, 3-oct-2026, E6; RF
    # 2/2025): la ficha decía vía «responsable» y traía el depósito del 24-oct;
    # el resultando narraba el depósito «en plazo» y el cómputo
    # (`redactor_adelanto.deposito_que_cuenta`) lo descartaba por la vía, y el
    # considerando desechaba por extemporáneo: el documento se contradecía. Un
    # oficio por correo también lo recibe la Sala. Ahora deciden el compositor,
    # el cómputo y main con `ficha_tramite.via_postal`; la vía «postal» sin
    # fecha de depósito sigue narrándose con el hueco del depósito.
    postal = via == "postal" or _via_postal(ficha)
    if postal:
        f_dep = _letra(dep) if dep else av.falta(
            "la fecha de depósito en el Servicio Postal Mexicano", "deposito_postal",
            "está en el sobre o la guía postal (es la fecha que cuenta para la "
            "oportunidad)", ap2)
        if dep and dep == _fecha(ficha.get("presentacion")):
            # DEPÓSITO Y RECEPCIÓN EL MISMO DÍA (RF 26/2025): casi seguro una de
            # las dos fechas está mal (la extracción copió una en la otra). Se
            # escribe sólo el depósito, que es la que cuenta, y se avisa.
            como = f"mediante oficio depositado en el Servicio Postal Mexicano el {f_dep}"
            av.add("LA RECEPCIÓN EN LA SALA COINCIDE CON EL DEPÓSITO EN EL CORREO "
                   f"(deposito_postal = presentacion = {dep.isoformat()}): se escribió sólo "
                   "el depósito. Comprueba el sello de recepción de la Sala; casi siempre "
                   "una de las dos fechas está mal.")
        else:
            como = (f"mediante oficio depositado en el Servicio Postal Mexicano el {f_dep} "
                    f"y recibido el {pres}")
    elif via == "electronica":
        como = f"mediante oficio presentado por vía electrónica el {pres}"
    else:
        como = f"mediante oficio presentado el {pres}" + (f" ante {_con_art(sala)}" if sala else "")
    _inc, _int = (("Inconformes", "interpusieron") if info_rf.get("sujeto_plural")
                  else ("Inconforme", "interpuso"))
    t2 = f"{_inc}, {sujeto}, {_int} recurso de revisión fiscal {como}."
    res.append({"titulo": ap2 + ".", "texto": _contraer(t2)})

    # 3 · TRÁMITE + adhesiva.
    ap3 = "Trámite del recurso de revisión fiscal"
    f_adm = _fecha_o(_d(ficha.get("admision")).get("fecha"), av,
                     "la fecha del auto de Presidencia que admitió el recurso",
                     "admision.fecha", "está en el auto de Presidencia de este "
                     "Tribunal Colegiado", ap3)
    f_reg = _registro_distinto(ficha, av, ap3)
    if f_reg:
        t3 = (f"Por auto de Presidencia de {f_reg}, este Tribunal Colegiado registró el "
              f"recurso con el número {num_t}; y por auto de {f_adm} lo admitió a trámite.")
    else:
        t3 = (f"Por auto de Presidencia de {f_adm}, este Tribunal Colegiado registró el "
              f"recurso con el número {num_t} y lo admitió a trámite.")
    adh = _d(ficha.get("adhesivo"))
    if _hay_adhesivo(adh):
        f_ad = _fecha_adhesivo("revision_fiscal", adh, av, "la fecha del auto que tuvo por "
                               "interpuesta la adhesión", "está en el auto que la provee", ap3)
        quien = _o(adh.get("quien"), av, "quién se adhirió al recurso",
                   "adhesivo.quien", "está en el escrito de adhesión", ap3)
        t3 += (f"\nPor auto de {f_ad}, se tuvo a "
               f"{_parte(quien) if quien != HUECO else quien} adhiriéndose al "
               f"recurso.")
    res.append({"titulo": ap3 + ".", "texto": _contraer(t3)})

    # 4 · TURNO (art. 92 LA por el 63, último párrafo, LFPCA) y 5 · RETURNO.
    res += _turno_y_returno(ficha, datos, av,
                            ", en términos del artículo 92 de la Ley de Amparo, "
                            "aplicable conforme al artículo 63, último párrafo, de la "
                            "Ley Federal de Procedimiento Contencioso Administrativo")

    extra = {
        "expediente": exp, "toca": "", "expediente_tfja": exp, "sala": sala,
        "actora": actora,
        # NORMALIZADOS (en prosa y sin artículo), para la legitimación
        # (`documento_generado`): la unidad que firmó el oficio y la autoridad
        # demandada a la que representa. RF 2, 7 y 6/2025: la legitimación
        # copiaba el nombre crudo, en versales y con «EN REPRESENTACIÓN DEL…».
        "recurrente_unidad": unidad, "autoridad_demandada": dem,
        "clase": "sentencia", "negativa_ficta": negativa,
        "fecha_acto": f_acto if f_acto != HUECO else "",
        "fecha_acto_iso": f_acto_d.isoformat() if f_acto_d else "",
        "organo_acto": sala, "responsable": sala,
        "materia": "administrativa", "numero": num,
        # E6: el depósito sólo cuenta si la regla única dice que fue por correo.
        "deposito_postal_iso": dep.isoformat() if (dep and postal) else "",
        "via_postal": postal,
        # E8: quien firmó, como lo nombra el resultando («la Titular…» se
        # conserva en todo el documento) y, si firmó la titular de la propia
        # unidad, su nombre para «lo hizo valer {nombre}, {unidad}».
        "recurrente_unidad_con_articulo": info_rf.get("unidad_con_articulo", ""),
        "recurrente_nombre": info_rf.get("nombre", ""),
        "recurrente_la_titular": bool(info_rf.get("la_titular")),
        # E2: recurre una autoridad (singular); la actora, con su número.
        "plural": False, "plural_actora": plural_act,
        "descripcion_acto": "de la sentencia recurrida",
        # C6: el rubro y el considerando de conexidad o de hecho notorio.
        "relacionados": relacionados,
    }
    return visto, res, extra


def _via_postal(ficha: dict) -> bool:
    """¿El recurso se interpuso por correo? La regla es UNA, la de
    `ficha_tramite.via_postal` (E6): depósito presente y anterior o igual a la
    presentación ⇒ por correo, aunque la vía diga «responsable». Si esa pieza
    no se puede llamar, la misma regla aquí."""
    try:
        import ficha_tramite as _ft
        _f = getattr(_ft, "via_postal", None)
        if callable(_f):
            return bool(_f(ficha))
    except Exception:
        pass
    dep = _fecha(ficha.get("deposito_postal"))
    if not dep:
        return False
    pres = _fecha(ficha.get("presentacion"))
    return pres is None or dep <= pres


# ═══════════════════════════════════════════════════════════════════════════
# LA PUERTA
# ═══════════════════════════════════════════════════════════════════════════
_COMPONEDORES = {
    "amparo_directo": _amparo_directo,
    "amparo_revision": _amparo_revision,
    "queja": _queja,
    "revision_fiscal": _revision_fiscal,
}


_CLAVES_EXTRA = ("expediente", "toca", "fecha_acto", "fecha_acto_iso", "organo_acto",
                 "inciso_97", "cola_97", "fraccion_97", "materia", "responsable",
                 "juicio_amparo", "juzgado", "sala", "expediente_tfja", "actora",
                 "descripcion_acto", "numero",
                 # Segunda ronda (3-oct-2026): del formulario, con el mismo nombre.
                 "fraccion_63", "cuantia", "fundamento_surtimiento",
                 # Tercera ronda (3-oct-2026, FIXES_R3 D): la clase del acto y su
                 # nombre (AD), el ponente vigente, el adherente y la unidad que
                 # recurre en la revisión fiscal.
                 "clase", "acto_reclamado", "ponente", "adherente", "recurrente_unidad",
                 "promovente_en_prosa", "quejoso_en_prosa",
                 "autoridad_demandada",
                 # Cuarta ronda (3-oct-2026, FIXES_R4 D): el cargo del ponente
                 # (E3), el órgano recurrido del AR y el descartado (E10), y quien
                 # firmó la revisión fiscal con su artículo y su nombre (E8).
                 "ponente_titulo", "organo_recurrido", "juzgado_descartado",
                 "recurrente_unidad_con_articulo", "recurrente_nombre",
                 # Sexta ronda (3-oct-2026, C5): el resultando que copia la
                 # demanda de amparo indirecto («primero»; «» si no la hay).
                 "resultando_demanda")
# LAS LISTAS DE LA SEXTA RONDA, SIEMPRE LISTAS (3-oct-2026): los actos de la
# demanda (C5, el punto que confirma los nombra) y los asuntos relacionados
# (C6). Vacías si el tipo no las usa o la ficha no las trae.
_LISTAS_EXTRA = ("actos", "relacionados")
# Las marcas de la cuarta ronda, siempre presentes y booleanas: el número de
# quien promueve o recurre, de la quejosa y de la actora (E2), el adhesivo sin
# ningún auto (E5), el correo (E6) y «la Titular» del papel (E8).
_MARCAS_EXTRA = ("plural", "plural_quejoso", "plural_actora", "adhesivo_sin_auto",
                 "via_postal", "recurrente_la_titular")

_ROMANOS_63 = ("I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X")


def _validada(ficha: dict, tipo: str) -> tuple:
    """(copia de la ficha tras `ficha_tramite.validar`, sus avisos). Sin el
    módulo, o si falla, la ficha tal cual y ningún aviso: la validación ayuda,
    no decide si se compone."""
    import copy
    try:
        f = copy.deepcopy(ficha)
    except Exception:
        return ficha, []
    if not f.get("tipo"):
        f["tipo"] = tipo
    try:
        import ficha_tramite as _ft
        avisos = _ft.validar(f)
    except Exception:
        return ficha, []
    return f, [str(a) for a in (avisos or []) if a]


def _ponente_y_titulo(ficha: dict) -> tuple:
    """(ponente vigente, el cargo que el auto le dio: «Magistrada», «Magistrado»
    o «»). El ponente que firma: el del returno si lo hubo, el del turno si no.

    LA CARÁTULA NOMBRABA AL DEL TURNO (3-oct-2026: 8 de 8 AD, 8 de 8 AR y 6 de
    8 RF del banco tuvieron returno) mientras el resultando de returno decía
    que los autos pasaron a otro. Con returno pero sin su ponente se devuelve
    «»: el del turno ya no es, y no se adivina. El cargo es el del MISMO auto
    del que sale el nombre (E3, cuarta ronda): el del returno no se le pone al
    ponente del turno."""
    returno = _d(ficha.get("returno"))
    if _s(returno.get("ponente")):
        return _s(returno.get("ponente")), _s(returno.get("titulo"))
    if _s(returno.get("fecha")):
        return "", ""
    turno = _d(ficha.get("turno"))
    if _s(turno.get("ponente")):
        return _s(turno.get("ponente")), _s(turno.get("titulo"))
    return "", ""


# ── UN AVISO POR DATO (cuarta ronda, 3-oct-2026, E11; AR 448/2025 y rev_3) ──
# `validar` y el compositor decían el mismo hecho con dos textos: «EL ÓRGANO DE
# LA RESOLUCIÓN RECURRIDA («…») ES TAMBIÉN UNA DE LAS AUTORIDADES» junto a
# «SE DESCARTÓ COMO ÓRGANO…» o «FALTA EL JUZGADO…»; «LA FECHA … ESTÁ ESCRITA
# EN EL ACTO RECLAMADO» junto a «… APARECE EN EL ACTO RECLAMADO». Cuando el
# compositor ya dijo el hecho —y además dónde quedó el hueco o qué fuente usó—,
# el de `validar` sobra. Cada par: (lo que dice validar, lo que dijo el compositor).
_MISMO_HECHO = (
    (re.compile(r"ES TAMBIÉN UNA DE LAS AUTORIDADES"),
     re.compile(r"SE DESCARTÓ COMO ÓRGANO|^FALTA EL JUZGADO|ESTABA ENTRE LAS AUTORIDADES")),
    (re.compile(r"NO TIENE FORMA DE ÓRGANO DE AMPARO"),
     re.compile(r"NO PARECE UN JUZGADO DE DISTRITO|SE DESCARTÓ COMO ÓRGANO|^FALTA EL JUZGADO")),
    (re.compile(r"ESTÁ ESCRITA EN EL ACTO RECLAMADO"),
     re.compile(r"APARECE EN EL ACTO RECLAMADO")),
    # Quinta ronda: el del compositor dice además que el autorizado se omitió
    # del resultando y de la legitimación (y quién era); ése se queda.
    (re.compile(r"^UN AUTORIZADO NO INTERPONE"),
     re.compile(r"^UN AUTORIZADO NO INTERPONE.*Se omitió del resultando")),
    # El registro igual al informe (queja fr. II, Q 335): el de validar y el del
    # compositor son el mismo dato (registro.fecha); se queda el del compositor.
    (re.compile(r"^FECHA IMPOSIBLE: en la queja de la fracción II"),
     re.compile(r"^FALTA LA FECHA DEL AUTO DE PRESIDENCIA QUE REGISTRÓ")),
)


def _repite(aviso_validar: str, del_compositor: list) -> bool:
    """¿El aviso de `validar` dice un hecho que el compositor ya avisó?"""
    for rx_v, rx_c in _MISMO_HECHO:
        if rx_v.search(aviso_validar) and any(rx_c.search(a) for a in del_compositor):
            return True
    return False


# Y AL REVÉS: lo que el compositor sólo repetiría. El dato truncado y el
# representante que es un cargo los acusa `validar` (que además limpia la copia
# que aquí se compone); si ya lo dijo, el aviso del compositor no añade nada y
# se calla. Así el hecho sale una vez aunque los avisos de `validar` lleguen
# también por `ficha["avisos"]`. (El autorizado que interpone por la autoridad
# pasó a `_MISMO_HECHO` en la quinta ronda: el del compositor dice más.)
_YA_LO_DIJO_VALIDAR = (
    (re.compile(r"^DATO TRUNCADO"), re.compile(r"CORTAD|TRUNCAD")),
    (re.compile(r"^EL REPRESENTANTE ES UN CARGO"), re.compile(r"ES (?:UN|EL) CARGO")),
)
_RX_VALOR_DEL_AVISO = re.compile(r"=\s*«(.+?)»\)")


def _ya_lo_dijo(aviso_mio: str, de_validar: list) -> bool:
    for rx_m, rx_v in _YA_LO_DIJO_VALIDAR:
        if not rx_m.search(aviso_mio):
            continue
        m = _RX_VALOR_DEL_AVISO.search(aviso_mio)
        valor = m.group(1)[:25] if m else ""
        return any(rx_v.search(v) and (not valor or valor in v) for v in de_validar)
    return False


def _datos_del_formulario(t: str, ficha: dict, av: _Avisos) -> dict:
    """Los tres datos que el secretario confirma en el formulario y que los
    considerandos leen tal cual (SEGUNDA RONDA, 3-oct-2026):

      fraccion_63            — romano I..X; sólo revisión fiscal (procedencia).
      cuantia                — «$847,738.77» o «indeterminada»; sólo revisión
                               fiscal (fracciones I y II del 63 de la LFPCA).
      fundamento_surtimiento — «el artículo 126 del Código de Procedimientos
                               Civiles del Estado de Querétaro»; todos los tipos
                               (la oportunidad lo escribe tras «conforme a»).

    El compositor NO los escribe en los resultandos: sólo los pasa. Una
    fracción que no es romano de la I a la X no pasa (la procedencia la pide
    en hueco); la fracción y la cuantía en otro tipo no aplican y se avisa."""
    fr63 = _s(ficha.get("fraccion_63")).upper().strip(" .")
    cuantia = _s(ficha.get("cuantia"))
    fund = _s(ficha.get("fundamento_surtimiento")).rstrip(" .")
    if t != "revision_fiscal":
        if fr63 or cuantia:
            av.add("LA FRACCIÓN DEL ARTÍCULO 63 DE LA LFPCA Y LA CUANTÍA (fraccion_63, "
                   "cuantia) SÓLO APLICAN A LA REVISIÓN FISCAL: la ficha las trae en "
                   "este asunto y se ignoraron. Revisa de qué asunto vienen.")
        fr63 = cuantia = ""
    elif fr63 and fr63 not in _ROMANOS_63:
        av.add(f"LA FRACCIÓN DEL ARTÍCULO 63 «{fr63}» NO ES UNA FRACCIÓN DE LA I A LA X "
               f"(fraccion_63): se ignoró; escríbela en romano en el formulario.")
        fr63 = ""
    return {"fraccion_63": fr63, "cuantia": cuantia, "fundamento_surtimiento": fund}


def componer(tipo: str, ficha: dict, datos: dict) -> dict:
    """El V I S T O (cuerpo, en minúscula, sin rótulo) y los resultandos del
    tipo, con sus avisos y los `datos_extra` para los considerandos.

    → {"visto": str, "resultandos": [{"titulo", "texto"}], "avisos": [str],
       "avisos_repetidos": [str], "datos_extra": {...}}

    `avisos_repetidos` (cuarta ronda, E11): los avisos de `ficha_tramite.
    validar` que se callaron porque el compositor dijo el mismo hecho mejor.
    Quien junte además `ficha["avisos"]` (que `armar` llenó con `validar`)
    debe quitarlos también de ahí, o el hecho saldrá dos veces.

    El resultando de la sesión NO va aquí: lo pone `documento_generado.componer`
    después de éstos, como hoy. NUNCA LANZA: un dato con forma rara se trata
    como dato que falta (hueco + aviso), y un fallo interno devuelve la
    estructura vacía con el aviso, para que el cableado caiga al camino viejo.
    """
    av = _Avisos()
    ficha = _d(ficha)
    datos = _d(datos)
    t = _normalizar_tipo(tipo)
    if t not in _COMPONEDORES:
        av.add(f"TIPO DE ASUNTO DESCONOCIDO («{_s(tipo)}»): el compositor por tipo "
               f"sólo conoce {', '.join(TIPOS)}. No se compuso nada.")
        return {"visto": "", "resultandos": [], "avisos": av.lista, "avisos_repetidos": [],
                "datos_extra": {}}
    # LA FICHA SE VALIDA AQUÍ TAMBIÉN (3-oct-2026, rev_3: AR 307/2024, 448 y
    # 60/2025): «el veinticinco de marzo de 2024 se celebró la audiencia… y el
    # seis de diciembre de 2023 se dictó sentencia» salió sin un aviso, porque
    # el compositor no llamaba a `ficha_tramite.validar` y una ficha que no pasó
    # por `armar`, o que se editó después, entregaba fechas imposibles como
    # buenas. Se valida una COPIA (validar escribe `fechas_imposibles` y este
    # módulo es puro) y se compone con lo que la validación deja.
    ficha, av_validar = _validada(ficha, t)
    try:
        visto, res, extra = _COMPONEDORES[t](ficha, datos, av)
    except Exception as ex:  # pragma: no cover — la red de seguridad
        return {"visto": "", "resultandos": [], "datos_extra": {}, "avisos_repetidos": [],
                "avisos": _unir(av_validar, av.lista) + [
                    f"EL COMPOSITOR POR TIPO FALLÓ ({type(ex).__name__}: {ex}); se compone "
                    f"por el camino anterior."]}
    adh = _d(ficha.get("adhesivo"))
    # E5: el adhesivo sin ningún auto no llega a la carátula ni a los
    # considerandos como si existiera; el resultando lo menciona con su hueco.
    sin_auto = t != "queja" and _adhesivo_sin_auto(adh)
    hay = _hay_adhesivo(adh) and t != "queja" and not sin_auto
    extra["adhesivo"] = dict(adh) if hay else None
    extra["adherente"] = _sin_rol(_s(adh.get("quien")))[0] if hay else ""
    extra["adhesivo_sin_auto"] = sin_auto
    extra["ponente"], extra["ponente_titulo"] = _ponente_y_titulo(ficha)
    extra["supletorio"] = _supletorio(ficha, datos)
    extra["tipo"] = t
    extra.update(_datos_del_formulario(t, ficha, av))
    # LAS CLAVES DE LA SPEC (§2) ESTÁN SIEMPRE, vacías si el tipo no las usa:
    # quien lee `datos_extra` no tiene que saber qué tipo trae cuál.
    for k in _CLAVES_EXTRA:
        extra.setdefault(k, "")
    for k in _MARCAS_EXTRA:
        extra[k] = bool(extra.get(k))
    for k in _LISTAS_EXTRA:
        v = extra.get(k)
        extra[k] = list(v) if isinstance(v, (list, tuple)) else []
    for r in res:
        r["texto"] = _pulir(r["texto"])
    # UN AVISO POR DATO (E11). Los de `validar` van primero, salvo los que
    # repiten un hecho que el compositor dijo mejor (dónde quedó el hueco, qué
    # fuente usó); y del compositor se calla lo que `validar` ya dijo igual.
    # `avisos_repetidos` son los de `validar` que se callaron: quien junte
    # además `ficha["avisos"]` (que `armar` llenó con `validar`) debe
    # quitarlos también de ahí.
    propios = [a for a in av.lista if not _ya_lo_dijo(a, av_validar)]
    repetidos = [a for a in av_validar if _repite(a, propios)]
    return {"visto": _pulir(visto), "resultandos": res,
            "avisos": _unir([a for a in av_validar if a not in repetidos], propios),
            "avisos_repetidos": repetidos, "datos_extra": extra}


def _unir(*listas) -> list:
    """Las listas de avisos en orden, sin repetir ninguno."""
    fuera = []
    for lista in listas:
        for a in (lista or []):
            if a and a not in fuera:
                fuera.append(a)
    return fuera


_RX_DOBLE_PUNTO = re.compile(r"(?<!\.)\.[ \t]*\.(?!\.)")


def _pulir(t: str) -> str:
    """Tipografía segura, la última pasada de TODO lo que sale de aquí (V I S T O,
    cada resultando con sus bloques AUTORIDAD RESPONSABLE / ACTO RECLAMADO, el
    considerando y el resolutivo del adhesivo). Las palabras no se tocan.

      · «S.A..», «C.V..», «S.A. .» → un punto: el de la fórmula sobre el de la
        razón social (cableado.txt: «Tiene ese carácter Banco del Centro,
        S.A..»). Los puntos suspensivos («…», «...») se respetan.
      · espacios dobles → uno; sin espacio antes de «,», «.», «;», «:»;
      · sin espacios al principio ni al final de cada renglón (los bloques de
        autoridades y actos van uno por renglón)."""
    t = _RX_DOBLE_PUNTO.sub(".", _RX_CONTROL.sub("", t or ""))
    t = re.sub(r"[ \t]{2,}", " ", t)
    t = re.sub(r"[ \t]+([,.;:])", r"\1", t)
    t = re.sub(r"[ \t]+\n", "\n", t)
    t = re.sub(r"\n[ \t]+", "\n", t)
    return t.strip(" \t")


# ═══════════════════════════════════════════════════════════════════════════
# EL ADHESIVO: CONSIDERANDO DE LEGITIMACIÓN Y OPORTUNIDAD, Y SU RESOLUTIVO
# ═══════════════════════════════════════════════════════════════════════════
# El 722/2025 (8 de 8 corridas) no tenía ni resultando, ni considerando, ni
# punto resolutivo del amparo adhesivo, y el V I S T O se lo atribuía al
# quejoso. ESQUELETO declaraba el rótulo y nadie lo emitía (codigo.txt).
#
# LA ADHESIÓN EN REVISIÓN FISCAL ES EL PENÚLTIMO PÁRRAFO DEL ARTÍCULO 63 de la
# LFPCA, no el último: el último manda tramitar el recurso conforme a la Ley de
# Amparo (texto local de la LFPCA, art. 63; el corpus lo cita «penúltimo» 6
# veces y «último» ninguna para la adhesión).
_ADH = {
    "amparo_directo": {
        "rotulo": "Legitimación y oportunidad del amparo adhesivo.",
        "nombre": "el amparo adhesivo", "f": False, "plazo": 15,
        "plazo_letra": "quince días", "notificado": "el auto admisorio de la demanda",
        "fund_plazo": "artículo 181 de la Ley de Amparo",
        "legit": ("{quien} tiene legitimación para promover el amparo adhesivo, en "
                  "términos del artículo 182, primer párrafo, de la Ley de Amparo, pues "
                  "es parte en el juicio del que emana el acto reclamado y la sentencia "
                  "reclamada le resultó favorable, de modo que tiene interés jurídico "
                  "en que subsista."),
    },
    "amparo_revision": {
        "rotulo": "Legitimación y oportunidad de la revisión adhesiva.",
        "nombre": "la revisión adhesiva", "f": True, "plazo": 5,
        "plazo_letra": "cinco días",
        "notificado": "el auto de Presidencia que admitió el recurso",
        "fund_plazo": "artículo 82 de la Ley de Amparo",
        "legit": ("{quien} tiene legitimación para adherirse a la revisión interpuesta "
                  "por {recurrente}, en términos del artículo 82 de la Ley de Amparo, "
                  "pues obtuvo resolución favorable en el juicio de amparo."),
    },
    "revision_fiscal": {
        "rotulo": "Legitimación y oportunidad de la revisión adhesiva.",
        "nombre": "la revisión adhesiva", "f": True, "plazo": 15,
        "plazo_letra": "quince días",
        "notificado": "el auto de Presidencia que admitió el recurso",
        "fund_plazo": ("artículo 63, penúltimo párrafo, de la Ley Federal de "
                       "Procedimiento Contencioso Administrativo"),
        "legit": ("{quien} tiene legitimación para adherirse al recurso de revisión "
                  "fiscal interpuesto por {recurrente}, en términos del artículo 63, "
                  "penúltimo párrafo, de la Ley Federal de Procedimiento Contencioso "
                  "Administrativo, pues la sentencia recurrida le resultó favorable."),
    },
}


def _recurrente_principal(t: str, ficha: dict, datos: dict) -> str:
    caracter = _sin_tildes(_s(ficha.get("caracter") or datos.get("papel_recurrente")))
    r = _s(ficha.get("promovente")) or _s(datos.get("recurrente")) or _s(datos.get("quejoso"))
    if not r:
        return ""
    return _parte(r, autoridad=(t == "revision_fiscal" or caracter == "autoridad"))


def _clausula_inhabiles(_f0, c) -> str:
    """«sin contar sábados y domingos, ni …, por ser inhábiles en términos del
    artículo 19 de la Ley de Amparo, ni …, por ser inhábil conforme a…».

    LA MISMA FÓRMULA QUE EL CONSIDERANDO DE OPORTUNIDAD (3-oct-2026, rev_2): el
    adhesivo armaba la suya y cerraba TODO con el artículo 19, también las
    vacaciones del artículo 226 de la LOPJF o una circular. Ahora la pone
    `fase0_oportunidad.clausula_inhabiles`, cada grupo con su fundamento; si
    no está, la de antes."""
    try:
        return _f0.clausula_inhabiles(c)
    except Exception:
        tr = _f0.tramos_inhabiles(c.inhabiles_en_medio, c.cal_amparo)
        ni = f", ni {_f0.tramos_en_letra(tr)}," if tr else ""
        return (f"sin contar sábados y domingos{ni} por ser inhábiles en términos del "
                f"{c.cal_amparo.fundamento}")


def considerando_adhesivo(tipo: str, ficha: dict, datos: dict) -> tuple:
    """(rótulo, texto, avisos) del considerando del adhesivo, o ("", "", avisos)
    si no hay adhesivo. Va ANTES de la dispensa (lo emite `componer`).

    LA OPORTUNIDAD SE CUENTA, NO SE AFIRMA: con las dos fechas (notificación de
    la admisión al adherente y presentación del escrito) se cuenta con
    `fase0_oportunidad.computar`, calendario federal (el escrito se presenta
    ante este Tribunal Colegiado) y surtimiento del artículo 31 de la Ley de
    Amparo —fracción I si el adherente es autoridad, fracción II si es
    particular—. Sin alguna de las dos fechas, el veredicto va en hueco."""
    av = _Avisos()
    t = _normalizar_tipo(tipo)
    ficha = _d(ficha)
    datos = _d(datos)
    adh = _d(ficha.get("adhesivo"))
    if not _hay_adhesivo(adh):
        return "", "", []
    if t not in _ADH:
        av.add("LA QUEJA NO ADMITE ADHESIÓN: la ficha trae un adhesivo y no se "
               "compuso considerando para él.")
        return "", "", av.lista
    if _adhesivo_sin_auto(adh):
        # E5 (cuarta ronda): sin el auto que lo admite ni el escrito no hay
        # considerando; el resultando lo menciona con la pregunta.
        return "", "", [_aviso_adhesivo_sin_auto(t, adh)]
    cfg = _ADH[t]
    rot = cfg["rotulo"]
    quien_raw = _s(adh.get("quien"))
    forma = _sin_tildes(_s(adh.get("forma_notificacion")))
    autoridad = forma == "oficio" or (not forma and _es_autoridad(quien_raw))
    quien = _parte(quien_raw, autoridad=autoridad) if quien_raw else av.falta(
        "quién promovió el adhesivo", "adhesivo.quien",
        "está en el escrito de adhesión y en el auto que lo admite", rot.rstrip("."))
    quien_may = quien[:1].upper() + quien[1:]
    recurrente = ""
    if "{recurrente}" in cfg["legit"]:
        recurrente = _recurrente_principal(t, ficha, datos) or av.falta(
            "quién interpuso el recurso principal", "promovente",
            "está en el escrito de agravios", rot.rstrip("."))
    legit = cfg["legit"].format(quien=quien_may, recurrente=recurrente)

    notif = _fecha(adh.get("notificacion"))
    pres = _fecha(adh.get("presentacion"))
    nombre = cfg["nombre"]
    ext_ = "extemporánea" if cfg["f"] else "extemporáneo"
    rf = t == "revision_fiscal"
    if notif and pres:
        import fase0_oportunidad as _f0
        surte = ""
        if forma == "electronica":
            c = None
        elif rf:
            # LA REGLA LITERAL DEL 63, PENÚLTIMO PÁRRAFO, DE LA LFPCA (3-oct-2026,
            # FIXES_R3 D; RF 4/2025): «dentro del plazo de quince días contados a
            # partir de la fecha en la que se le notifique la admisión del
            # recurso». El considerando le atribuía a ese párrafo un cómputo
            # desde el SURTIMIENTO que el párrafo no dice. Se cuenta como lo
            # dice: desde la notificación, sin surtimiento.
            c = _f0.computar(notif, pres, regla="otra", plazo=cfg["plazo"],
                             tipo_asunto="amparo_revision", surtio_manual=notif)
        elif autoridad:
            c = _f0.computar(notif, pres, regla="otra", plazo=cfg["plazo"],
                             tipo_asunto="amparo_revision", surtio_manual=notif)
            surte = "el mismo día, conforme al artículo 31, fracción I, de la Ley de Amparo"
        else:
            c = _f0.computar(notif, pres, regla="lista", plazo=cfg["plazo"],
                             tipo_asunto="amparo_revision")
            surte = ("al día hábil siguiente, conforme al artículo 31, fracción II, de "
                     "la Ley de Amparo")
        if c is None:
            av.add("LA NOTIFICACIÓN AL ADHERENTE FUE ELECTRÓNICA (adhesivo.forma_"
                   "notificacion): surte cuando se genera la constancia de consulta "
                   "(art. 31, fr. III, LA), fecha que la ficha no trae. El veredicto "
                   "de oportunidad va en hueco.")
            opo = (f"Por lo que hace a la oportunidad, {nombre} debía presentarse dentro "
                   f"del plazo de {cfg['plazo_letra']} previsto en el {cfg['fund_plazo']}; "
                   f"{cfg['notificado']} se notificó a {quien} por vía electrónica el "
                   f"{_letra(notif)} y el escrito se presentó el {_letra(pres)}, por lo "
                   f"que {HUECO}.")
        else:
            for a in c.avisos:
                # EL ESCRITO DEL ADHESIVO SE PRESENTA ANTE ESTE TRIBUNAL COLEGIADO:
                # el aviso del cómputo de la revisión que pide los días en que no
                # laboró el juzgado «ante quien se presenta el escrito» (art. 86
                # LA) no le toca, y mandaba al secretario a buscar días ajenos.
                if "artículo 86" in str(a):
                    continue
                av.add(f"CÓMPUTO {_contraer('de ' + nombre).upper()}: {a}")
            rango = _f0.tramos_en_letra([(c.inicio, c.vencimiento)])
            sin_contar = _clausula_inhabiles(_f0, c)
            if c.anticipada:
                veredicto = ("se presentó con anterioridad al inicio del plazo, lo que "
                             "no le resta oportunidad")
            elif c.oportuna:
                veredicto = ("es claro que se interpuso oportunamente" if cfg["f"]
                             else "es claro que se promovió oportunamente")
            else:
                veredicto = f"resulta {ext_}"
                av.add(f"{nombre.upper()} SALE {ext_.upper()} según el cómputo "
                       f"(venció el {c.vencimiento.isoformat()} y se presentó el "
                       f"{pres.isoformat()}): el considerando lo dice así y el "
                       f"resolutivo debe {'desecharla' if cfg['f'] else 'desecharlo'}. "
                       f"Comprueba las dos fechas.")
            ultimo = ", último día del plazo," if pres == c.vencimiento else ","
            if rf:
                opo = (f"Por lo que hace a la oportunidad, {cfg['notificado']} se notificó "
                       f"a {quien} el {_letra(notif)}; por tanto, el plazo de "
                       f"{cfg['plazo_letra']} previsto en el {cfg['fund_plazo']}, contado "
                       f"a partir de la fecha en la que se le notificó la admisión del "
                       f"recurso, transcurrió {rango}, {sin_contar}; entonces, si el "
                       f"escrito se presentó el {_letra(pres)}{ultimo} {veredicto}.")
            else:
                # CON SURTIMIENTO EL MISMO DÍA, SIN «es decir, el {la misma fecha}»
                # (cuarta ronda, E12; Q 172 y 24/2026): repetía la fecha de la
                # notificación.
                es_decir = ("" if c.surtio == notif else f", es decir, el {_letra(c.surtio)}")
                opo = (f"Por lo que hace a la oportunidad, {cfg['notificado']} se notificó a "
                       f"{quien} el {_letra(notif)} y esa notificación surtió efectos "
                       f"{surte}{es_decir}; por tanto, el plazo de "
                       f"{cfg['plazo_letra']} previsto en el {cfg['fund_plazo']}, que corre "
                       f"a partir del día siguiente al en que surtió efectos la "
                       f"notificación, transcurrió {rango}, {sin_contar}; entonces, si el "
                       f"escrito se presentó el {_letra(pres)}{ultimo} {veredicto}.")
            # LA OTRA LECTURA, PARA QUIEN FIRMA. Si con el surtimiento del artículo
            # 31, fracción II, de la Ley de Amparo (aplicable por el último
            # párrafo del 63) el escrito quedara en tiempo, el veredicto depende de
            # la lectura que se adopte, y eso lo decide quien firma.
            if rf and c.oportuna is False and not c.anticipada:
                c2 = _f0.computar(notif, pres, regla="lista", plazo=cfg["plazo"],
                                  tipo_asunto="amparo_revision")
                if c2.oportuna:
                    av.add("OPORTUNIDAD DE LA ADHESIVA EN EL LÍMITE: con la regla literal "
                           "del penúltimo párrafo del artículo 63 de la LFPCA («a partir "
                           "de la fecha en la que se le notifique») venció el "
                           f"{c.vencimiento.isoformat()}; con el surtimiento del artículo "
                           "31, fracción II, de la Ley de Amparo vencería el "
                           f"{c2.vencimiento.isoformat()} y estaría en tiempo. Decide cuál "
                           "aplica antes de firmar.")
    else:
        # `_fecha_o` distingue la fecha que FALTA de la que vino ilegible o
        # fuera de rango («9999-12-30»): el aviso dice cuál de las dos.
        f_n = _letra(notif) if notif else _fecha_o(
            adh.get("notificacion"), av,
            f"la fecha en que se notificó al adherente {cfg['notificado']}",
            "adhesivo.notificacion", "está en la constancia de notificación del auto "
            "de admisión", rot.rstrip("."))
        f_p = _letra(pres) if pres else _fecha_o(
            adh.get("presentacion"), av, "la fecha de presentación del escrito del adhesivo",
            "adhesivo.presentacion", "está en el sello o acuse del escrito", rot.rstrip("."))
        desde = ("contado a partir de la fecha en la que se le notificara la admisión del "
                 "recurso" if rf else
                 "contado a partir del día siguiente al en que surtiera efectos la "
                 f"notificación de {cfg['notificado']}")
        opo = (f"Por lo que hace a la oportunidad, {nombre} debía presentarse dentro del "
               f"plazo de {cfg['plazo_letra']} previsto en el {cfg['fund_plazo']}, "
               f"{desde}; {'la admisión se notificó' if rf else 'esa notificación se hizo'} "
               f"a {quien} el {f_n} y el escrito se presentó el {f_p}, por lo que {HUECO}.")
        av.add(f"EL VEREDICTO DE OPORTUNIDAD {_contraer('de ' + nombre).upper()} VA EN "
               f"HUECO: sin las dos fechas no se cuenta, y no se afirma.")
    texto = _pulir(_contraer(legit + "\n" + opo))
    return rot, texto, av.lista


def _prospera(p) -> object:
    """True (prospera el principal) | False (no prospera) | "desecha" | None."""
    if p is None or isinstance(p, bool):
        return p
    v = _sin_tildes(_s(p))
    if v in ("desecha", "improcedente", "desechado", "extemporaneo"):
        return "desecha"
    if v in ("fundado", "concede", "ampara", "revoca", "modifica", "prospera", "si", "true"):
        return True
    if v in ("infundado", "niega", "no_ampara", "confirma", "inoperante", "no", "false",
             "sobresee"):
        return False
    return None


def resolutivo_adhesivo(tipo: str, prospera_principal, ficha: dict) -> tuple:
    """Ver `_resolutivo_adhesivo`; aquí sólo se pule la tipografía."""
    texto, aviso = _resolutivo_adhesivo(tipo, prospera_principal, ficha)
    return _pulir(texto), aviso


def _resolutivo_adhesivo(tipo: str, prospera_principal, ficha: dict) -> tuple:
    """(texto, aviso) del punto resolutivo del adhesivo; ("", "") si no lo hay.

    `prospera_principal`: True si el principal prospera (se concede / es
    fundado), False si no (se niega / infundado, se confirma), "desecha" si el
    principal es improcedente (sólo revisión fiscal: 2a./J. 145/2019), None si
    aún no se sabe. También admite las claves «fundado», «infundado»,
    «concede», «niega», «confirma», «revoca», «modifica».

    SÓLO SE ESCRIBE LO QUE SE SIGUE DEL PRINCIPAL: con el principal negado o
    infundado el adhesivo queda sin materia (35 de 35 AD del corpus); con el
    principal concedido o fundado hay que ESTUDIARLO, y el verbo va en hueco
    con aviso."""
    t = _normalizar_tipo(tipo)
    adh = _d(_d(ficha).get("adhesivo"))
    if not _hay_adhesivo(adh) or t not in _ADH:
        return "", ""
    if _adhesivo_sin_auto(adh):
        # E5 (cuarta ronda, AD 274/2025): ningún punto resolutivo sobre un
        # adhesivo sin auto que lo admita ni escrito que lo presente.
        return "", _aviso_adhesivo_sin_auto(t, adh)
    quien_raw = _s(adh.get("quien"))
    quien = _parte(quien_raw) if quien_raw else HUECO
    aviso_q = ("" if quien_raw else
               "FALTA QUIÉN PROMOVIÓ EL ADHESIVO (adhesivo.quien): está en el escrito "
               "de adhesión. Va en hueco en el punto resolutivo. ")
    p = _prospera(prospera_principal)
    if t == "amparo_directo":
        cola = f"el amparo adhesivo promovido por {quien}."
        if p is False:
            return f"Se declara sin materia {cola}", aviso_q.strip()
        if p is True:
            return (f"{HUECO} {cola}", (aviso_q + "EL AMPARO PRINCIPAL SE CONCEDE: el "
                    "adhesivo hay que estudiarlo y decidir su punto resolutivo (se niega "
                    "o queda sin materia, según sus conceptos). Va en hueco.").strip())
        if p == "desecha":
            return (f"{HUECO} {cola}", (aviso_q + "EL AMPARO PRINCIPAL NO SE ESTUDIA: "
                    "decide la suerte del adhesivo, que sigue la del principal (art. 182 "
                    "LA). Va en hueco.").strip())
        return (f"{HUECO} {cola}", (aviso_q + "NO SE SABE TODAVÍA SI PROSPERA EL AMPARO "
                "PRINCIPAL: el punto del adhesivo va en hueco.").strip())
    cola = f"la revisión adhesiva interpuesta por {quien}."
    if t == "revision_fiscal" and p == "desecha":
        return f"Se desecha {cola}", aviso_q.strip()
    if p is False:
        return f"Se declara sin materia {cola}", aviso_q.strip()
    if p is True:
        return (f"{HUECO} {cola}", (aviso_q + "EL RECURSO PRINCIPAL ES FUNDADO: la "
                "revisión adhesiva hay que estudiarla y decidir su punto resolutivo. Va "
                "en hueco.").strip())
    if p == "desecha":
        return (f"{HUECO} {cola}", (aviso_q + "EL RECURSO PRINCIPAL SE DESECHA: decide "
                "la suerte de la adhesiva, que sigue la del principal (art. 82 LA). Va "
                "en hueco.").strip())
    return (f"{HUECO} {cola}", (aviso_q + "NO SE SABE TODAVÍA SI PROSPERA EL RECURSO "
            "PRINCIPAL: el punto de la adhesiva va en hueco.").strip())
