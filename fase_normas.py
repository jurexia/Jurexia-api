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
_RX_FEDERAL = re.compile(
    r"ley\s+de\s+amparo|c[óo]digo\s+federal|ley\s+federal|ley\s+org[áa]nica|"
    r"c[óo]digo\s+nacional|ley\s+general", re.I)

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


def _donde(cola: str, coleccion_estatal: str) -> tuple:
    """(colección, filtro extra) según qué ley nombró el estudio."""
    if _RX_CONSTITUCION.search(cola):
        return COLECCION_BLOQUE, "constitucion"
    if _RX_CONVENCIONAL.search(cola):
        return COLECCION_BLOQUE, "convencion"
    if _RX_FEDERAL.search(cola):
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
    try:
        r = qdrant.scroll(collection_name=coleccion,
                          scroll_filter=Filter(must=debe),
                          limit=40, with_payload=True)
        if inspect.isawaitable(r):
            r = await r
        puntos = r[0] if isinstance(r, tuple) else r
        return [p.payload for p in puntos]
    except Exception:
        return []


def _elegir(pl: list, cola: str) -> list:
    """Los fragmentos del cuerpo legal que el estudio nombró, y sólo de ése.

    El artículo 296 existe en cuatro códigos de Querétaro. Sin nombre de ley en
    la cita no se puede saber cuál, y traer el equivocado es peor que no traer.
    """
    pedidas = _palabras(cola)
    if not pedidas:
        return []
    mejor, puntos = None, 0
    for p in pl:
        ley = str(p.get("cuerpo_legal_oficial") or p.get("origen")
                  or p.get("ref") or "")
        n = len(pedidas & _palabras(ley))
        if n > puntos:
            mejor, puntos = ley, n
    # UNA PALABRA DISTINTIVA BASTA cuando es la que separa un código de otro:
    # «procedimientos» distingue el procesal del civil, «penal» del ambiental.
    # Exigir dos descartaba citas correctas como «del código procesal civil».
    _distintivas = {"procedimientos", "penal", "civil", "familiar", "ambiental",
                    "amparo", "fiscal", "administrativo", "mercantil",
                    "hacienda", "trabajo"}
    if not mejor:
        return []
    if puntos < 2 and not (_palabras(mejor) & pedidas & _distintivas):
        return []
    return [p for p in pl
            if str(p.get("cuerpo_legal_oficial") or p.get("origen")
                   or p.get("ref") or "") == mejor]


def _reunir(fr: list) -> str:
    """El artículo entero: viene troceado y un trozo no es el artículo."""
    fr = sorted(fr, key=lambda p: int(p.get("chunk_index") or 0))
    partes, visto = [], set()
    for p in fr:
        t = " ".join(str(p.get("texto") or "").split())
        if t and t not in visto:
            visto.add(t)
            partes.append(t)
    return " ".join(partes)[:2000]


async def recuperar(qdrant, estudio: str, coleccion_estatal: str = "") -> list:
    """Las normas que el estudio cita, con su texto. Listas para la nota al pie."""
    pares = citados(estudio)
    if not pares:
        return []
    tareas, meta = [], []
    for num, cola in pares[:MAX_ARTICULOS]:
        col, tipo = _donde(cola, coleccion_estatal)
        if not col:
            continue
        tareas.append(_traer(qdrant, col, num, tipo))
        meta.append((num, cola))
    if not tareas:
        return []
    res = await asyncio.gather(*tareas)

    fuera, vistos = [], set()
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
