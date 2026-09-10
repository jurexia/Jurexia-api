"""LA AUTORIDAD RESPONSABLE SE LEE, NO SE PREGUNTA.

David, 30-ago-2026: «no deberías de preguntarme eso pues viene en la propia
sentencia reclamada».

Y NO SE INVENTA: si no se reconoce ninguna, se devuelve vacío y el formulario
la pide. Un nombre de autoridad equivocado en el resolutivo es peor que un
hueco, porque el hueco se ve.

────────────────────────────────────────────────────────────────────────────
CUARTO INTENTO, Y LOS TRES ANTERIORES FALLARON POR LO MISMO: MEDIR EL NOMBRE
EN CARACTERES.

El 10-sep-2026 el resolutivo del 91/2025 salió diciendo «dictada por la Sala
Regional en Querétaro Infrazione Administrat Administración Desconce». Se
recuperó el OCR de origen y la causa no era la que parecía. El patrón era
`Sala\\s+Regional[\\w\\sáéíóúñ]{0,60}`, y ese {0,60} hacía DOS daños opuestos a
la vez:

  · TRAGABA DE MÁS: la clase acepta cualquier cosa que parezca palabra, así que
    se llevó el campo «DEPENDENCIA:» de una carátula de notificación donde el
    OCR entrelazó dos columnas de un formulario;
  · Y CORTABA DE MENOS: «Sala Regional en Querétaro del Tribunal Federal de
    Justicia Administrativa» mide 74 caracteres y el patrón sólo alcanza 73.
    Ni con un escáner impecable podía devolver el nombre entero.

Las dos cosas juntas fabricaban un EMPATE: la basura recortada a 73 y el
nombre bueno recortado a 73 medían lo mismo, y `max(cands, key=len)` con empate
se queda con el primero del documento, que era la basura de la página 5. La
basura ganó por posición, no por mérito. Y subir el tope lo empeora: medido
sobre el texto real, con 70, 80, 90, 110 o 130 la basura sigue ganando, porque
lo que le sigue en la carátula también son caracteres de palabra.

EL NOMBRE DE UN ÓRGANO NO SE MIDE EN CARACTERES, SE LEE EN PALABRAS. Y las
palabras con las que se nombra un órgano jurisdiccional mexicano son un
vocabulario CERRADO: se midió sobre los 45 nombres correctos que hay en el
acervo del taller y son 85 palabras distintas, más los topónimos, que también
son una lista pública y cerrada. Así que ya no se cuenta: se avanza token a
token mientras el vocabulario reconozca lo que viene, y se para en la primera
palabra que no sea de este idioma. «Infrazione» no lo es. «Administrat»
tampoco. «del Tribunal Federal de Justicia Administrativa» sí, entero, sin
tope que lo corte.

LO QUE ESTO ARREGLA, MEDIDO:
  · el 91/2025 sobre su OCR real devuelve el nombre correcto y completo;
  · de 45 nombres correctos del acervo, el módulo anterior no devolvía entero
    21; éste no devuelve enteros 2, y los dos son autoridades ADMINISTRATIVAS
    (una Agencia, una Legislatura), que nunca estuvieron en su alcance;
  · los 20 falsos negativos medidos en la auditoría —«Sala Especializada en
    Juicio en Línea», «Tribunal Federal de Justicia Administrativa», las salas
    con guion como «Norte-Centro I»— salen los 20.

Y LO QUE NO RESUELVE, ESCRITO PARA QUE NADIE LO DESCUBRA POR SORPRESA:
un vocabulario cerrado convierte lo desconocido en HUECO, no en basura. Un
órgano cuyo nombre lleve una palabra que aquí no esté saldrá recortado o
vacío. Eso es lo que la doctrina de arriba pide, pero significa que este
fichero hay que ampliarlo cuando aparezca un órgano nuevo: la lista es la
especificación, y está toda a la vista.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter


def _n(w: str) -> str:
    """Sin tildes, en minúscula y sin la puntuación que pega el OCR."""
    w = unicodedata.normalize("NFD", w or "")
    w = "".join(c for c in w if unicodedata.category(c) != "Mn")
    return w.lower().strip(' ,;.:"“”()[]«»/\\')


def _S(txt: str) -> frozenset:
    return frozenset(_n(w) for w in txt.split())


# ═══════════════════════════════════════════════════════════════════════════
# EL VOCABULARIO. Es la especificación: si un órgano no se puede escribir con
# estas palabras, este módulo no lo lee y devuelve hueco. Ampliarlo es el
# mantenimiento previsto, y se hace aquí, a la vista, no tocando un regex.
# ═══════════════════════════════════════════════════════════════════════════

# Con qué palabra EMPIEZA el nombre de un órgano.
_NUCLEOS = _S("""
    sala salas juzgado juzgados juez jueza junta juntas tribunal tribunales
    pleno ponencia centro comision comite consejo
""")

_ORDINALES = _S("""
    primero primera segundo segunda tercero tercera cuarto cuarta quinto quinta
    sexto sexta septimo septima setimo setima octavo octava noveno novena
    decimo decima undecimo undecima duodecimo duodecima decimoprimero
    decimoprimera decimosegundo decimosegunda decimotercero decimotercera
    decimocuarto decimocuarta decimoquinto decimoquinta unico unica
    vigesimo vigesima trigesimo trigesima cuadragesimo cuadragesima
    quincuagesimo quincuagesima
""")

_CARDINALES = _S("""
    uno dos tres cuatro cinco seis siete ocho nueve diez once doce trece
    catorce quince dieciseis diecisiete dieciocho diecinueve veinte treinta
    cuarenta cincuenta sesenta setenta ochenta noventa cien ciento
""")

_CONECTORES = _S("de del la las los el al y e en")

_MATERIAS = _S("""
    civil civiles familiar familiares penal penales mercantil mercantiles
    laboral laborales administrativa administrativo administrativas
    administrativos agrario agraria fiscal fiscales constitucional
    amparo amparos trabajo trabajos materia materias juicio juicios
    especializada especializado especial especiales mixto mixta oral orales
    ejecucion sanciones adolescentes extincion dominio tutelar
    responsabilidades anticorrupcion burocratico burocratica
""")

_ORGANICO = _S("""
    regional regionales distrito distritos judicial judiciales circuito
    circuitos federal federales federacion local locales estatal estatales
    municipal justicia superior instancia unitario unitarios
    colegiado colegiados apelacion revision conciliacion arbitraje registro
    numero num no procedimiento contencioso union nacion republica estado
    estados ciudad municipio partido zona region sede residencia con sin
    linea electronico electronica comercio exterior recursos individuales
    colectivos asuntos hacendaria hacendario auxiliar auxiliares itinerante
""")

_LEXICO = (_NUCLEOS | _ORDINALES | _CARDINALES | _CONECTORES
           | _MATERIAS | _ORGANICO)

# Los topónimos también son lista cerrada y pública: las 32 entidades, las
# regiones con nombre propio del TFJA y las cabeceras del acervo.
_TOPONIMOS = _S("""
    aguascalientes baja california sur campeche coahuila zaragoza colima
    chiapas chihuahua durango guanajuato guerrero hidalgo jalisco mexico
    michoacan ocampo morelos nayarit nuevo leon oaxaca puebla queretaro
    quintana roo potosi luis sinaloa sonora tabasco tamaulipas tlaxcala
    veracruz llave yucatan zacatecas cdmx
    norte centro sur oriente occidente noreste noroeste sureste suroeste
    golfo pacifico peninsular caribe metropolitana
    san santa santo villa valle puerto cerro real altos bravo
    santiago juan rio blanco monterrey guadalajara torreon celaya
    acapulco cancun merida tijuana culiacan hermosillo tuxtla gutierrez
    morelia toluca pachuca xalapa villahermosa tepic saltillo chilpancingo
    cuernavaca irapuato salamanca corregidora marcos cadereyta amealco
    tequisquiapan pedro escobedo toliman jalpan huimilpan pinal victoria
""")

# Lo que NUNCA es parte del nombre, por mucho que venga detrás de «de» o «en»:
# etiquetas de formulario, encabezados de sello y deícticos. Sin esta lista,
# «CON SEDE EN ESTA CIUDAD» en versales entraba como si «ESTA» fuera un lugar.
_NUNCA_TOPONIMO = _S("""
    oficio oficios expediente expedientes dependencia asunto asuntos folio
    citatorio acta actas acuerdo acuerdos anexos fojas hojas foja hoja
    notificacion notificaciones documentacion revision autorizacion recibe
    remito copia copias derecho propio nombre firma fecha hora horas dia dias
    mes meses año años nov dic ene feb mar abr may jun jul ago sep oct
    interior casa colonia calle avenida cp tel telefono correo
    esta este esto estos estas dicho dicha dichos dichas mismo misma
    referido referida citado citada aquel aquella suscrita suscrito
    licenciado licenciada lic ing arq cc atentamente sufragio efectivo
    presente presenta unidos mexicanos teja tfja sat imss issste
    infonavit sello recibido cumplase notifiquese doy fe
""")

# La palabra que ANUNCIA un lugar. Una palabra desconocida sólo entra en el
# nombre dentro de este marco: sin él, «Sala Regional en Querétaro» seguía con
# «Administración Desconcentrada», que es la PARTE, no el órgano.
_ABRE_TOPO = _S("""
    estado ciudad distrito judicial sede residencia regional municipio
    partido circunscripcion zona region delegacion
""")
_CABEZA_TOPO = _S("san santa santo villa ciudad valle nuevo nueva puerto cerro real")

# Un número es parte del nombre cuando lo pide la palabra anterior. Si no,
# «da cuenta ... de esta Sala el 29 de octubre» devolvía «Sala el 29».
_PIDE_NUMERO = _S("""
    numero num no distrito especial sala juzgado junta circuito ponencia
    zona region unitario colegiado tribunal
""")

# Una ponencia está DENTRO de una sala; un juez ES el titular de un juzgado.
# «Centro» queda fuera de la escala a propósito: es núcleo («Centro Federal de
# Conciliación») pero también región («Sala Regional del Centro II»).
_RANGO = {"juez": 1, "jueza": 1, "juzgado": 1, "juzgados": 1, "ponencia": 1,
          "junta": 2, "juntas": 2, "sala": 2, "salas": 2,
          "comision": 2, "comite": 2,
          "tribunal": 3, "tribunales": 3, "pleno": 3, "consejo": 3}

# Abreviaturas que llevan punto y NO cierran el nombre («Junta Especial No. 50»).
_ABREV = _S("no num nro art lic c cc dr sr sra av")

# EL ROMANO CANONICO. «^[IVXLCDM]{1,7}$» toma MIL, LIC, LID y DIVIDID por
# numeros. Un romano de verdad tiene forma canonica.
_ROMANO = re.compile(
    r"^M{0,3}(?:CM|CD|D?C{0,3})(?:XC|XL|L?X{0,3})(?:IX|IV|V?I{0,3})$")
_NUMERO = re.compile(r"^\d{1,4}$")
_PALABRA = re.compile("^[A-Za-zÀ-ÖØ-öø-ÿ]+(?:-[A-Za-zÀ-ÖØ-öø-ÿ]+)*$")

MAX_TOKENS = 32            # el nombre más largo del acervo tiene 22 palabras
MAX_TOPONIMO_LIBRE = 3
VECES_TOPONIMO_LIBRE = 2   # medido: «Infrazione» sale 1 vez en 72.647 caracteres
                           # y «Querétaro» 45. Lo desconocido que no se repite
                           # es ruido del escáner, no un lugar.
VENTANA = 6000


def _guion_conocido(nrm: str) -> bool:
    """«Hidalgo-México», «Norte-Centro»: el TFJA nombra así sus salas."""
    if "-" not in nrm:
        return False
    partes = [p for p in nrm.split("-") if p]
    return len(partes) > 1 and all(p in _TOPONIMOS or p in _LEXICO
                                   for p in partes)


def _es_romano(nrm: str, crudo: str) -> bool:
    """«CIVIL» son todas letras de numeración romana, y por eso el resolutivo
    salía gritando «Primera Sala CIVIL». También caen MIL, LIC y LID. Una
    palabra que ESTÁ en el vocabulario es una palabra, no un número."""
    return bool(_ROMANO.match(crudo.strip(".,;:")) and nrm not in _LEXICO
                and nrm not in _TOPONIMOS)


# ═══════════════════════════════════════════════════════════════════════════
# LA MARCHA: token a token, y se para en la primera palabra ajena
# ═══════════════════════════════════════════════════════════════════════════
class _Cand:
    __slots__ = ("texto", "ini", "corte", "antes")

    def __init__(self, texto, ini, corte, antes=""):
        self.texto = texto     # el nombre tal como está escrito en el acto
        self.ini = ini
        self.corte = corte     # la palabra que detuvo la marcha
        self.antes = antes     # la palabra anterior al corte (¿ranura de lugar?)


def _arranca(i, toks, nrm) -> bool:
    if nrm[i] in _NUCLEOS:
        return True
    return (nrm[i] in _ORDINALES and i + 1 < len(toks)
            and nrm[i + 1] in _NUCLEOS)


def _marcha(i, toks, nrm, crudos, veces_doc):
    j = i
    rango = _RANGO.get(nrm[i], 2)
    if nrm[i] in _ORDINALES and i + 1 < len(toks):
        rango = _RANGO.get(nrm[i + 1], 2)
    libre = 0
    seguidos = 0
    corte = ""
    while j + 1 < len(toks) and (j - i) < MAX_TOKENS:
        k = j + 1
        w, c = nrm[k], crudos[k]
        if not w:
            break
        # EL PUNTO CIERRA EL ENCABEZADO. «…JUSTICIA ADMINISTRATIVA. EXPEDIENTE»
        # y «…Tuxtepec, Oaxaca. En la ciudad de» son dos frases, no un nombre.
        if crudos[j].endswith(".") and nrm[j] not in _ABREV and j > i:
            corte = c
            break
        # UN ORDINAL SEGUIDO DE NÚCLEO ABRE NOMBRE NUEVO.
        if (k > i + 1 and w in _ORDINALES and k + 1 < len(toks)
                and nrm[k + 1] in _NUCLEOS):
            corte = c
            break
        # NI DOS ÓRGANOS DEL MISMO RANGO EN UN SOLO NOMBRE: «Jueza de Distrito
        # del Juzgado Quinto de Distrito» es el mismo órgano nombrado dos
        # veces, y el bueno es el segundo. Pero «Sala Regional … del Tribunal
        # Federal» sí, porque ahí el segundo es la institución que la contiene.
        if k > i + 1 and w in _NUCLEOS and w in _RANGO:
            if nrm[j] not in _CONECTORES or _RANGO[w] <= rango:
                corte = c
                break
        if w in _CONECTORES:
            seguidos += 1
            if seguidos > 3:
                corte = c
                break
            j = k
            continue
        seguidos = 0
        if _NUMERO.match(w) or _es_romano(w, c):
            if (nrm[j] in _PIDE_NUMERO or nrm[j] in _ORDINALES
                    or nrm[j] in _TOPONIMOS or _guion_conocido(nrm[j])):
                j, libre = k, 0
                continue
            corte = c
            break
        if w in _LEXICO or w in _TOPONIMOS or _guion_conocido(w):
            j, libre = k, 0    # el vocabulario cierra la ranura del topónimo
            continue
        # ── PALABRA DESCONOCIDA. Sólo pasa como topónimo, y con condiciones ──
        abre = ((nrm[j] in ("de", "del", "en") and j - 1 >= 0
                 and nrm[j - 1] in _ABRE_TOPO)
                or nrm[j] in _ABRE_TOPO
                or nrm[j] in _CABEZA_TOPO
                or (libre and nrm[j] not in _CONECTORES))
        if (abre and libre < MAX_TOPONIMO_LIBRE
                and w not in _NUNCA_TOPONIMO
                and _PALABRA.match(c.strip('.,;:«»"\''))
                and len(w) >= 4
                and c[:1].isupper()
                and veces_doc.get(w, 0) >= VECES_TOPONIMO_LIBRE):
            j, libre = k, libre + 1
            continue
        corte = c
        break
    return j, corte, (nrm[j] if corte else '')


def _recorta(txt: str) -> str:
    txt = " ".join((txt or "").split()).strip(" ,;.:")
    partes = txt.split()
    while partes and _n(partes[-1]) in (_CONECTORES
                                        | _S("con sin sede residencia")):
        partes.pop()
    return " ".join(partes).strip(" ,;.:")


def _versales(x: str) -> str:
    """Los encabezados van en versales; el resolutivo no."""
    if not (x.isupper() and len(x) > 12):
        return x
    menores = _CONECTORES | _S("con sin sede residencia ante por para")
    fuera = []
    for w in x.split():
        nw = _n(w)
        if _es_romano(nw, w.strip(".,;:")):
            fuera.append(w)
        elif nw in menores:
            fuera.append(w.lower())
        else:
            # EL GUION LLEVA MAYUSCULA DETRAS. «HIDALGO-MÉXICO».capitalize()
            # da «Hidalgo-méxico», y el TFJA nombra asi tres de sus salas.
            fuera.append("-".join(t.capitalize() for t in w.split("-")))
    return " ".join(fuera)


def _candidatos(t: str, veces_doc):
    toks = list(re.finditer(r"\S+", t))
    crudos = [m.group(0) for m in toks]
    nrm = [_n(c) for c in crudos]
    fuera, i = [], 0
    while i < len(toks):
        if _arranca(i, toks, nrm):
            j, corte, antes = _marcha(i, toks, nrm, crudos, veces_doc)
            nombre = _versales(_recorta(t[toks[i].start():toks[j].end()]))
            if nombre:
                fuera.append(_Cand(nombre, toks[i].start(), corte, antes))
            i = j + 1
            continue
        i += 1
    return fuera


# ═══════════════════════════════════════════════════════════════════════════
# QUÉ NOMBRE IDENTIFICA A ALGUIEN Y CUÁL ES UN SELLO
# ═══════════════════════════════════════════════════════════════════════════
def identificable(nombre: str) -> bool:
    """¿Este nombre señala a un órgano concreto, o es un sello sin identidad?

    Pública a propósito: por el formulario entraron «JUZGADO», «juez» y «JUNTA»
    —siete de las treinta y siete que tecleó el secretario—, y ninguna
    corrección del extractor los arregla, porque no pasan por él.
    """
    crudos = (nombre or "").split()
    ws = [_n(w) for w in crudos]
    if len(ws) < 2 or not any(w in _NUCLEOS for w in ws):
        return False
    # Hace falta ALGO que lo individualice: ordinal, número, romano, topónimo
    # o materia. «JUZGADO» y «Juzgado de Distrito» no lo tienen.
    for w, c in zip(ws, crudos):
        if (w in _ORDINALES or w in _CARDINALES or w in _TOPONIMOS
                or w in _MATERIAS or _NUMERO.match(w) or _es_romano(w, c)
                or _guion_conocido(w)):
            return True
    return False


def _corte_sospechoso(c: _Cand) -> bool:
    """La palabra que paró la marcha, ¿es una que el OCR partió por la mitad?

    «…DE JUSTICIA ADMINISTRA TIVA» deja un nombre creíble al que le falta el
    final, y eso es justo el hueco que no se ve. Si nadie más en el documento
    lo escribe entero, vale más devolver hueco.
    """
    w = _n(c.corte)
    if len(w) < 4 or w in _NUNCA_TOPONIMO or w in _LEXICO or w in _TOPONIMOS:
        return False
    if any(v.startswith(w) and len(v) > len(w) for v in _LEXICO):
        return True
    # INJERTO · EL CORTE EN LA RANURA DEL LUGAR. Si la marcha se paró en una
    # palabra que PARECE nombre propio y venía justo detrás de «de/en/del» o de
    # una palabra que abre lugar, el nombre se quedó sin su topónimo: «Juzgado
    # Primero de Primera Instancia Mixto de Zacapu» -> «…Mixto». Eso es un
    # nombre creíble y falso, que es lo que la doctrina declara peor que el
    # hueco. Un topónimo que este módulo no conoce se trata como hueco.
    if (c.corte[:1].isupper() and c.corte.strip('.,;:').isalpha()
            and (c.antes in ("de", "del", "en") or c.antes in _ABRE_TOPO
                 or c.antes in _CABEZA_TOPO)):
        return 2   # 2 = sospecha que NO se redime repitiendo el mismo recorte
    return False


# ═══════════════════════════════════════════════════════════════════════════
# QUIÉN NO PUEDE SER LA RESPONSABLE
# ═══════════════════════════════════════════════════════════════════════════
# La Suprema Corte aparece en las cabeceras porque los autos citan sus tesis.
# UN TRIBUNAL COLEGIADO tampoco: en cuatro de cinco asuntos reales la
# «responsable» salía de una cita de tesis, y en uno era ESTE mismo tribunal.
# Pero el Tribunal Colegiado DE APELACIÓN sí es responsable en amparo penal, y
# el veto en bloque se lo comía: por eso ahora se le exceptúa.
_RX_NUNCA = re.compile(
    r"Suprema\s+Corte|Alto\s+Tribunal|Tribunal\s+Pleno|Pleno\s+Regional|"
    r"Tribunal(?:es)?\s+Colegiado(?!\s+de\s+Apelaci)", re.I)


def _nunca_responsable(nombre: str) -> bool:
    return bool(_RX_NUNCA.search(nombre or ""))


_SOLO_FEDERAL = re.compile(
    r"de\s+Distrito|Tribunal\s+Colegiado|Tribunal\s+Unitario\s+de\s+Circuito",
    re.I)

_ADMITE = {
    "queja": lambda n: bool(_SOLO_FEDERAL.search(n)),
    "amparo_revision": lambda n: bool(_SOLO_FEDERAL.search(n)),
    # En la revisión fiscal es una Sala del Tribunal Federal de Justicia
    # Administrativa. Se añadió «Sala Especializada» y el Tribunal a secas:
    # sin ellos, la Sala Especializada en Juicio en Línea salía hueco.
    "revision_fiscal": lambda n: bool(re.search(
        r"Sala\s+Regional|Sala\s+Especializada|Sala\s+Superior|"
        r"Justicia\s+Administrativa", n, re.I)),
    "amparo_directo": lambda n: not _SOLO_FEDERAL.search(n),
}


def _admisible(nombre: str, tipo: str) -> bool:
    f = _ADMITE.get((tipo or "").strip().lower())
    return True if f is None else bool(f(nombre))


# BASTA UNA MENCIÓN SÓLO PARA EL ÓRGANO NUMERADO. El auto que se recurre en una
# queja nombra una sola vez a la «Jueza Cuarto de Distrito», y eso no es una
# cita de tesis. Pero «Sala Regional del Golfo Norte» a secas sí puede serlo:
# con una sola mención y la cabecera muda, la doctrina de arriba pide hueco.
_CONCRETO = re.compile(
    r"(?:Juez|Jueza|Juzgado)\s+(?:Primer[oa]|Segund[oa]|Tercer[oa]|Cuart[oa]|"
    r"Quint[oa]|Sext[oa]|S[ée]ptim[oa]|Octav[oa]|Noven[oa]|D[ée]cim[oa]|\d+)"
    r"\s+de\s+Distrito", re.I)


def _clave(n: str) -> str:
    return " ".join(_n(w) for w in (n or "").split())


def _elegir(cands, tipo, exigir_dos):
    vivos = [c for c in cands
             if identificable(c.texto) and not _nunca_responsable(c.texto)
             and _admisible(c.texto, tipo)]
    if not vivos:
        return "", 0, False
    cuenta = Counter(_clave(c.texto) for c in vivos)
    primero = {}
    for c in vivos:
        primero.setdefault(_clave(c.texto), c)
    claves = list(cuenta)
    # UN NOMBRE QUE CONTIENE A OTRO NO COMPITE CON ÉL: LO ABSORBE. Ya no hace
    # falta el desempate por longitud que premiaba lo que el patrón tragaba de
    # más; ahora la contención es literal, porque nada viene recortado.
    absorbidos = {k for k in claves
                  if any(k2 != k and k2.startswith(k + " ") for k2 in claves)}
    finales = [k for k in claves if k not in absorbidos] or claves
    # y hereda sus menciones: una lectura corta del mismo nombre lo CONFIRMA.
    total = {k: sum(v for k2, v in cuenta.items()
                    if k2 == k or k.startswith(k2 + " ")) for k in finales}
    mejor = max(finales, key=lambda k: (bool(_CONCRETO.search(primero[k].texto)),
                                        total[k], len(k)))
    c = primero[mejor]
    if exigir_dos and total[mejor] < 2 and not _CONCRETO.search(c.texto):
        return "", 0, False
    _s = _corte_sospechoso(c)
    return c.texto, total[mejor], (_s == 2) or (_s and total[mejor] < 2)


def de_texto(acto: str, tipo: str = "") -> str:
    """La autoridad que dictó el acto, leída del propio documento.

    LA CABECERA MANDA; EL CUERPO SÓLO SI LA CABECERA CALLA. La responsable de
    un acto se identifica en su encabezado, así que ahí se busca; y sólo si el
    encabezado no da ninguna —porque venía en un sello que el OCR rompió— se
    mira el documento entero y se exige insistencia para no confundir una cita
    con el emisor.
    """
    t = " ".join((acto or "").split())
    if not t:
        return ""
    veces_doc = Counter(_n(w) for w in t.split())

    nombre, _, sosp = _elegir(_candidatos(t[:VENTANA], veces_doc), tipo,
                              exigir_dos=False)
    if nombre:
        # UNA EXTENSIÓN DEL MISMO NOMBRE, DONDE ESTÉ, SIEMPRE ES MEJOR: lo
        # contiene y le añade lo que le falta. Pero sólo si el documento la
        # repite: vista una sola vez es tan probablemente un renglón de prosa.
        todos = _candidatos(t, veces_doc)
        cta = Counter(_clave(c.texto) for c in todos)
        k = _clave(nombre)
        for c in todos:
            k2 = _clave(c.texto)
            if (k2.startswith(k + " ") and cta[k2] >= 2
                    and not _nunca_responsable(c.texto)
                    and _admisible(c.texto, tipo)):
                nombre, sosp, k = c.texto, _corte_sospechoso(c), k2
    else:
        nombre, _, sosp = _elegir(_candidatos(t, veces_doc), tipo,
                                  exigir_dos=True)
    # Y SI EL NOMBRE PUEDE ESTAR CORTADO Y NADIE LO CONFIRMA, HUECO. Un nombre
    # al que le falta el final no se ve; un hueco sí.
    return "" if sosp else nombre


def sello_sin_identidad(nombre: str) -> bool:
    """«JUZGADO», «juez», «JUNTA»: el sello de un órgano no es su nombre.

    Para el camino del FORMULARIO, que no valida nada: siete de los treinta y
    siete nombres que tecleó el secretario son sellos así, y el cómputo los
    acepta y los arrastra hasta el resolutivo. Es a propósito más estrecha que
    `identificable`: aquí manda lo que escriba el secretario, y una autoridad
    ADMINISTRATIVA —«Legislatura del Estado de Querétaro»— es legítima aunque
    este módulo no sepa leerla. Sólo se avisa de lo que no nombra a nadie.
    """
    ws = [_n(w) for w in (nombre or "").split()]
    if not ws or len(ws) > 3:
        return False
    if not all(w in _NUCLEOS or w in _CONECTORES or w in _ORGANICO
               for w in ws):
        return False
    return not any(w in _ORDINALES or w in _CARDINALES or w in _TOPONIMOS
                   or w in _MATERIAS or _NUMERO.match(w) for w in ws)
