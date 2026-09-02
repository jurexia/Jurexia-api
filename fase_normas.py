"""LOS ARTÍCULOS QUE EL ESTUDIO CITA, TRAÍDOS POR NÚMERO.

David: «Los artículos sí están en qdrant, es un tema de recuperación… deben
estar todos, tanto los de la constitución, convencionales y código civil y
procesal civil de Querétaro citados. Esto es uno de los pasos más grandes de
calidad en el producto».

Tenía razón en las dos cosas. El acervo tiene los 15,858 artículos de Querétaro
y `articulo_num` está INDEXADO: se piden por número, exacto y completo. Lo que
fallaba era el momento y el método.

EL MÉTODO ANTERIOR: antes de escribir, se buscaban por parecido semántico
cuatro artículos por problema. Si el estudio acababa citando el 296 y el 568,
que no estaban entre esos cuatro, se quedaban sin texto —y sin nota al pie—.

EL MÉTODO BUENO: DESPUÉS de escribir, se lee qué artículos citó de verdad y se
piden ESOS por número. No es adivinar lo que hará falta: es traer lo que hizo
falta. Cuesta un scroll por artículo y devuelve el texto íntegro para la nota
al pie, que es lo que permite a quien firma comprobar de un vistazo si el
precepto dice lo que se le atribuye.

LA TRAMPA, la misma de siempre: el artículo 296 existe en el Código Civil, en
el de Procedimientos Civiles, en el Penal y en el Ambiental de Querétaro. Traer
el equivocado es peor que no traer nada, así que se elige el cuerpo legal que
más palabras comparte con el que el estudio nombró, y si el estudio no nombra
ninguno, NO se trae: un artículo sin cuerpo legal identificado no se cita.
"""

from __future__ import annotations

import asyncio
import re

COLECCION_BLOQUE = "bloque_constitucional"
COLECCION_FEDERAL = "leyes_federales"

# Cuántos artículos se persiguen por sentencia. Los engroses invocan entre
# cuatro y diez preceptos; más allá se está trayendo el código entero.
MAX_ARTICULOS = 12

# «artículo 296 del Código Civil del Estado de Querétaro», «artículos 74,
# fracción IV y 76 de la Ley de Amparo», «el 1º constitucional».
# LAS CITAS SE ENUMERAN. «los artículos 84, 86, 276 y 277 del Código de
# Procedimientos Civiles del Estado de Querétaro» son CUATRO artículos y UNA
# ley, y la ley va al final de la enumeración. Capturar sólo dos números —como
# hacía la primera versión— perdía la mitad de las citas y, peor, dejaba el
# nombre de la ley fuera de la ventana, así que ni siquiera se sabía dónde
# buscarlos.
_RX_ENUM = re.compile(
    r"art[íi]culos?\s+"                       # el rótulo
    r"((?:\d{1,4}(?:\s*(?:bis|ter|qu[áa]ter))?"
    r"(?:\s*,?\s*fracci[óo]n(?:es)?\s+[IVXLC]+(?:\s*(?:,|y)\s*[IVXLC]+)*)?"
    r"(?:\s*(?:,|y|e)\s*)?)+)"                # uno o varios números
    r"([^.;:]{0,140})",                       # y la ley, que va al final
    re.I)
_RX_NUMERO = re.compile(r"\b(\d{1,4})\b")

_RX_CONSTITUCION = re.compile(r"constituci[óo]n|constitucional", re.I)
_RX_CONVENCIONAL = re.compile(
    r"convenci[óo]n|pacto|tratado|protocolo|declaraci[óo]n americana", re.I)
# LO QUE FALTABA EN ESTE PATRÓN COSTABA CITAS. «Código Fiscal de la
# FEDERACIÓN» no dice «federal»; la Ley del Seguro Social y la del ISSSTE no
# dicen ni lo uno ni lo otro. Todas caían al final de `_donde` y se buscaban en
# la colección del ESTADO, donde no están, así que se quedaban sin texto al pie
# —el mismo fallo mudo de la Ley de Amparo, sólo que con más leyes—.
_RX_FEDERAL = re.compile(
    r"ley\s+de\s+amparo|c[óo]digo\s+federal|ley\s+federal|ley\s+org[áa]nica|"
    r"c[óo]digo\s+nacional|ley\s+general|de\s+la\s+federaci[óo]n|"
    r"ley\s+del\s+seguro\s+social|ISSSTE|INFONAVIT|"
    r"ley\s+del\s+instituto\s+de\s+seguridad|ley\s+aduanera|"
    r"ley\s+del\s+impuesto|ley\s+de\s+instituciones\s+de\s+cr[ée]dito", re.I)

# ── EL FUERO DE LA AUTORIDAD ───────────────────────────────────────────────
# David: «si la autoridad es federal, bloquear legislación local». Es una regla
# de competencia, no de búsqueda: el SAT no aplica el Código Civil de Querétaro
# y una cita así no es un error de recuperación sino un disparate jurídico.
#
# LA REGLA SÓLO CORRE EN UN SENTIDO. Una autoridad federal puede tener que
# aplicar derecho local —el juez de distrito que conoce de un juicio mercantil
# por jurisdicción concurrente, el que resuelve sobre bienes regidos por el
# código civil de la entidad—, pero el caso frecuente y dañino es el contrario:
# el asunto es puramente federal y la colección del estado le mete un código
# que no le toca. Así que se bloquea el fondo estatal cuando el asunto es
# federal Y la cita no nombra ninguna ley local.
_RX_AUTORIDAD_FEDERAL = re.compile(
    r"\bSAT\b|servicio\s+de\s+administraci[óo]n\s+tributaria|"
    r"\bIMSS\b|instituto\s+mexicano\s+del\s+seguro\s+social|"
    r"\bISSSTE\b|\bINFONAVIT\b|\bFOVISSSTE\b|\bPROFECO\b|"
    # ESCRITOS ENTEROS, que es como aparecen en la carátula de un asunto real:
    # el recurrente de la revisión fiscal de hoy se llama «…del Instituto de
    # Seguridad y Servicios Sociales de los Trabajadores del Estado», sin sigla
    # por ninguna parte, y el filtro de fuero no lo reconocía.
    r"servicios\s+sociales\s+de\s+los\s+trabajadores|"
    r"fondo\s+nacional\s+de\s+la\s+vivienda|"
    r"junta\s+federal\s+de\s+conciliaci[óo]n|tribunal\s+federal\s+de\s+"
    r"justicia\s+administrativa|administraci[óo]n\s+desconcentrada|"
    # «Secretaría de Hacienda» A SECAS la tienen casi todos los estados. Se
    # exige el nombre completo de la federal.
    r"secretar[íi]a\s+de\s+hacienda\s+y\s+cr[ée]dito\s+p[úu]blico|"
    r"comisi[óo]n\s+nacional|"
    r"instituto\s+nacional\b|procuradur[íi]a\s+federal", re.I)

# «del Estado de Querétaro», «local», «municipal»: si la cita lo dice, va al
# acervo estatal aunque la autoridad sea federal.
_RX_LEY_LOCAL = re.compile(
    r"del\s+estado\b|estatal|municipal|del\s+municipio|de\s+la\s+entidad|"
    r"local\b|ayuntamiento", re.I)


def autoridad_es_federal(texto: str) -> bool:
    """¿El acto viene de una autoridad del fuero federal?"""
    return bool(_RX_AUTORIDAD_FEDERAL.search(texto or ""))

_VACIAS = {"de", "del", "la", "el", "los", "las", "y", "en", "para", "por",
           "estado", "que", "artículo", "articulo", "artículos", "articulos"}


def _palabras(x: str) -> set:
    import unicodedata
    x = unicodedata.normalize("NFKD", (x or "").lower())
    x = "".join(c for c in x if not unicodedata.combining(c))
    return {w for w in re.findall(r"[a-z]{4,}", x) if w not in _VACIAS}


def citados(estudio: str) -> list:
    """[(número, cómo lo nombró el estudio)] sin repetir."""
    fuera, vistos = [], set()
    for m in _RX_ENUM.finditer(estudio or ""):
        # SE MIRA A LOS DOS LADOS. La ley va detrás en «artículo 296 DEL Código
        # Civil» pero DELANTE en «la Convención sobre los Derechos del Niño, en
        # su artículo 3»: mirando sólo hacia atrás, ese artículo se buscaba en
        # el código de Querétaro.
        # LA VENTANA SE CORTA EN LA FRASE. Sin ese corte se llevaba la
        # «Constitución» de la oración anterior y mandaba los artículos de la
        # Ley de Amparo al bloque constitucional. El contexto de una cita es su
        # frase, no el párrafo entero.
        antes = (estudio or "")[max(0, m.start() - 140):m.start()]
        antes = re.split(r"(?<=[.;:])\s+", antes)[-1]
        cola = (antes + " " + (m.group(2) or "")).strip()
        # Todos los números de la enumeración comparten la MISMA ley.
        for num in _RX_NUMERO.findall(m.group(1) or ""):
            clave = (num, " ".join(sorted(_palabras(cola))[:3]))
            if clave in vistos:
                continue
            vistos.add(clave)
            fuera.append((num, cola))
    return fuera[: MAX_ARTICULOS * 2]


def _donde(cola: str, coleccion_estatal: str,
           fuero_federal: bool = False) -> tuple:
    """(colección, filtro extra) según qué ley nombró el estudio."""
    if _RX_CONSTITUCION.search(cola):
        return COLECCION_BLOQUE, "constitucion"
    if _RX_CONVENCIONAL.search(cola):
        return COLECCION_BLOQUE, "convencion"
    # LA LEY LOCAL SE MIRA PRIMERO, y el orden no es un detalle: `_RX_FEDERAL`
    # incluye «ley orgánica» y «ley general», así que «de la Ley Orgánica del
    # Poder Judicial DEL ESTADO de Querétaro» se iba al acervo federal, donde
    # no está, y la cita se quedaba sin texto. Si la cita dice «del Estado»,
    # «municipal» o «local», ya ha dicho dónde vive.
    if _RX_LEY_LOCAL.search(cola or "") and coleccion_estatal:
        return coleccion_estatal, ""
    if _RX_FEDERAL.search(cola):
        return COLECCION_FEDERAL, ""
    # El asunto es federal y la cita no nombra ninguna ley local: el acervo del
    # estado no le toca. Antes que traer el código civil de Querétaro a un
    # asunto del SAT, se prefiere no traer nada.
    if fuero_federal and not _RX_LEY_LOCAL.search(cola or ""):
        return COLECCION_FEDERAL, ""
    return (coleccion_estatal or ""), ""


async def _traer(qdrant, coleccion: str, num: str, tipo: str) -> list:
    import inspect
    from qdrant_client.models import FieldCondition, Filter, MatchValue
    if not coleccion:
        return []
    debe = [FieldCondition(key="articulo_num", match=MatchValue(value=int(num)))]
    if tipo:
        debe.append(FieldCondition(key="tipo", match=MatchValue(value=tipo)))
    # UNA VENTANA CIEGA DE 40 ERA UNA LOTERÍA, y perdía justo lo que se pedía.
    # El mismo número de artículo existe en decenas de cuerpos legales de la
    # misma colección: el 79 tiene 48 trozos en leyes_federales y el de la Ley
    # de Amparo NO entraba en los primeros 40, mientras que el 61 y el 93 sí.
    # Es decir, el precepto se recuperaba o no según dónde hubiera caído en el
    # orden interno de Qdrant. Peor todavía: como `_elegir` no encontraba el
    # cuerpo legal pedido, devolvía vacío EN SILENCIO, y la cita se quedaba sin
    # texto al pie sin que nada lo dijera.
    #
    # Ahí nació mi diagnóstico falso de esta mañana —«la Ley de Amparo no está
    # en el acervo»—: está, con 290 trozos y 270 artículos, pero por esta
    # ventana no se alcanzaba. El diccionario estático sigue siendo la decisión
    # correcta para esa ley, porque quita la lotería del todo, pero el motivo
    # que apunté era otro y el fallo era GENERAL, no de una ley.
    #
    # Se pagina hasta agotar. Un artículo repartido en cien trozos es raro; el
    # tope existe sólo para que un `articulo_num` corrupto no traiga la
    # colección entera.
    puntos, desde, MAX = [], None, 400
    try:
        while len(puntos) < MAX:
            r = qdrant.scroll(collection_name=coleccion,
                              scroll_filter=Filter(must=debe),
                              limit=200, offset=desde, with_payload=True)
            if inspect.isawaitable(r):
                r = await r
            lote, desde = (r if isinstance(r, tuple) else (r, None))
            puntos += list(lote or [])
            if not desde:
                break
        return [p.payload for p in puntos]
    except Exception:
        return [p.payload for p in puntos]


def _elegir(pl: list, cola: str) -> list:
    """Los fragmentos del cuerpo legal que el estudio nombró, y sólo de ése.

    El artículo 296 existe en cuatro códigos de Querétaro. Sin nombre de ley en
    la cita no se puede saber cuál, y traer el equivocado es peor que no traer.
    """
    pedidas = _palabras(cola)
    if not pedidas:
        return []
    # NO BASTA CONTAR COINCIDENCIAS: HAY QUE PENALIZAR LO QUE SOBRA. «Ley
    # Federal del Trabajo» y «Ley Federal de los Trabajadores al Servicio del
    # Estado» comparten «ley», «federal» y la raíz de «trabajo»; contando sólo
    # aciertos ganaba la segunda y el proyecto transcribió como artículo 123
    # constitucional un precepto burocrático sobre la huelga. Lo vio el
    # dictamen de un colega, y con razón: atribuir a la Carta Magna el texto de
    # otra ley es de los errores que no se perdonan.
    #
    # Se resta lo que la ley candidata trae y la cita NO nombra. Así «Servicio»
    # y «Estado» hunden a la burocrática cuando se pidió la del Trabajo.
    mejor, puntos = None, -99
    for p in pl:
        ley = str(p.get("cuerpo_legal_oficial") or p.get("origen")
                  or p.get("ref") or "")
        suyas = _palabras(ley)
        acierta = len(pedidas & suyas)
        sobra = len(suyas - pedidas)
        n = acierta - sobra
        if n > puntos:
            mejor, puntos = ley, n
    # UNA PALABRA DISTINTIVA BASTA cuando es la que separa un código de otro:
    # «procedimientos» distingue el procesal del civil, «penal» del ambiental.
    # Exigir dos descartaba citas correctas como «del código procesal civil».
    _distintivas = {"procedimientos", "penal", "civil", "familiar", "ambiental",
                    "amparo", "fiscal", "administrativo", "mercantil",
                    "hacienda", "trabajo"}
    if not mejor or puntos < 0:
        return []
    _ac = len(pedidas & _palabras(mejor))
    if _ac < 2 and not (_palabras(mejor) & pedidas & _distintivas):
        return []
    return [p for p in pl
            if str(p.get("cuerpo_legal_oficial") or p.get("origen")
                   or p.get("ref") or "") == mejor]


# EL TRANSITORIO DE 1917. Los artículos transitorios del decreto constitucional
# se ingestaron con el MISMO `articulo_num` que los permanentes, y ordenan
# primero. Se reconocen por la jerarquía: cuelgan del «TITULO NOVENO. DE LA
# INVIOLABILIDAD DE LA CONSTITUCION» porque en el PDF van detrás del 136, que
# es el único artículo permanente de ese título. Medido sobre los 355 trozos de
# la CPEUM: 17 artículos afectados, del 1º al 17 —el 1º, el 14, el 16 y el 17,
# los cuatro que se citan en todo amparo—.
def _es_transitorio(p: dict) -> bool:
    j = str(p.get("jerarquia") or "")
    return "INVIOLABILIDAD" in j and int(p.get("articulo_num") or 0) != 136


# DOS INGESTAS EN LA MISMA COLECCIÓN. La CPEUM se cargó dos veces con troceados
# distintos, y unirlas producía notas al pie con la palabra partida por la
# mitad: «…idad judicial federal, a petición de…» era el final de «autoridad»
# de un troceado pegado al otro. Se elige UNA rendición —la que más trozos
# tenga, que es la más completa— y no se mezclan.
def _una_sola_rendicion(fr: list) -> list:
    if len(fr) < 2:
        return fr
    grupos: dict = {}
    for p in fr:
        grupos.setdefault(str(p.get("jerarquia") or ""), []).append(p)
    if len(grupos) < 2:
        return fr
    def _abre(g):
        primero = min(g, key=lambda p: int(p.get("chunk_index") or 0))
        return bool(re.match(r"\s*Art[íi]culo?\.?\s*\d",
                             str(primero.get("texto") or "")))
    # Que EMPIECE por su propio número manda sobre que sea la más larga: una
    # rendición que arranca a mitad de frase no es el artículo.
    return max(grupos.values(), key=lambda g: (_abre(g), len(g)))


def _reunir(fr: list) -> str:
    """El artículo entero: viene troceado y un trozo no es el artículo."""
    fr = [p for p in fr if not _es_transitorio(p)]
    fr = _una_sola_rendicion(fr)
    fr = sorted(fr, key=lambda p: int(p.get("chunk_index") or 0))
    partes, visto = [], set()
    for p in fr:
        t = " ".join(str(p.get("texto") or "").split())
        if t and t not in visto:
            visto.add(t)
            partes.append(t)
    return " ".join(partes)[:2000]


async def recuperar(qdrant, estudio: str, coleccion_estatal: str = "",
                    fuero_federal: bool = False) -> list:
    """Las normas que el estudio cita, con su texto. Listas para la nota al pie."""
    pares = citados(estudio)
    if not pares:
        return []
    # LA LEY DE AMPARO NO SE BUSCA: SE SABE. No está en ninguna colección del
    # acervo —medido: cero trozos en leyes_federales, en el bloque y en las de
    # materia—, así que todas sus citas venían sin texto al pie desde siempre,
    # y en silencio.
    # EL TOPE NO APLICA A LO QUE NO SE BUSCA. `MAX_ARTICULOS` existe para no
    # hacer veinte viajes a Qdrant por sentencia; una consulta al diccionario
    # estático es un acceso a un dict. Aplicándoselo, el proyecto de la cuota
    # pensionaria se quedó sin el 19 y sin el 217 de la Ley de Amparo sólo por
    # llegar los trece y catorce de la lista.
    import normas_estaticas as _ne
    estaticos, pendientes = [], []
    for num, cola in pares:
        fijo = _ne.articulo(num, cola)
        if fijo:
            estaticos.append(fijo)
    _fijos = {(d["articulo"], d["citado_como"]) for d in estaticos}
    for num, cola in pares[:MAX_ARTICULOS]:
        if (str(num), (cola or "").strip()[:90]) in _fijos:
            continue
        pendientes.append((num, cola))

    tareas, meta = [], []
    for num, cola in pendientes:
        col, tipo = _donde(cola, coleccion_estatal, fuero_federal)
        if not col:
            continue
        tareas.append(_traer(qdrant, col, num, tipo))
        meta.append((num, cola))
    _vis, _uno = set(), []
    for d in estaticos:
        k = (d["articulo"], d["cuerpo_legal"])
        if k in _vis:
            continue
        _vis.add(k); _uno.append(d)
    estaticos = _uno

    if not tareas:
        return estaticos
    res = await asyncio.gather(*tareas)

    fuera, vistos = list(estaticos), {(d["articulo"], d["cuerpo_legal"])
                                      for d in estaticos}
    for (num, cola), pl in zip(meta, res):
        elegidos = _elegir(pl, cola)
        if not elegidos:
            continue
        texto = _reunir(elegidos)
        if not texto:
            continue
        ley = str(elegidos[0].get("cuerpo_legal_oficial")
                  or elegidos[0].get("origen") or "")
        clave = (num, ley)
        if clave in vistos:
            continue
        vistos.add(clave)
        fuera.append({"articulo": str(num), "cuerpo_legal": ley,
                      "texto": texto, "citado_como": cola.strip()[:90]})
    return fuera
