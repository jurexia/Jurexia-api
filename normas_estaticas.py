"""LAS DOS LEYES QUE NUNCA DEBEN VENIR DE UNA BÚSQUEDA.

David: «CPEUM y Ley de Amparo estáticos». Tenía razón por dos motivos
distintos, y sólo se ven midiendo el acervo.

LA LEY DE AMPARO SÍ ESTÁ EN QDRANT, y mi primera versión de este texto decía
lo contrario. Lo escribo aquí porque el error de método importa más que el
dato: comprobé su ausencia con un filtro `match: {text: …}` sobre un campo sin
índice de texto, Qdrant devolvió un 400, y mi código leyó `.get("result",{})
.get("points",[])` sobre la respuesta de error y contó CERO. Convertí un fallo
de la consulta en un hecho sobre el acervo. Medido después como se debía:
`leyes_federales` tiene 290 trozos de la Ley de Amparo y 270 artículos, entre
ellos los diez que estos asuntos citan.

LO QUE SÍ ERA CIERTO ES EL SÍNTOMA, y su causa resultó peor: `_traer` pedía 40
trozos A CIEGAS. El artículo 79 tiene 48 trozos con ese número repartidos entre
muchas leyes federales, y el de la Ley de Amparo no entraba en los primeros 40;
el 61 y el 93 sí entraban. O sea que el precepto se recuperaba o no según dónde
hubiera caído en el orden interno de la base, y cuando no, `_elegir` devolvía
vacío EN SILENCIO. No era una ley ausente: era una ventana ciega, y afectaba a
todas las leyes por igual. Eso está arreglado en `fase_normas._traer`, que
ahora pagina hasta agotar.

ENTONCES, ¿PARA QUÉ SIRVE ESTE MÓDULO? Para quitar la lotería del todo en la
ley que se cita en cada considerando de competencia y de procedencia de los
cuatro tipos de asunto. Una búsqueda que acierta casi siempre no basta cuando
el precepto es el que funda la jurisdicción: aquí el texto viene del archivo
oficial y no depende de cómo esté troceada una colección. Para las demás leyes
sigue mandando el acervo, que es donde están.

LA CONSTITUCIÓN SÍ ESTÁ, PERO NO SE PODÍA CITAR. Los artículos TRANSITORIOS del
decreto de 1917 se ingestaron con el mismo `articulo_num` que los permanentes y
ordenan primero. Medido sobre los 355 trozos: 17 artículos afectados, del 1º al
17, es decir EXACTAMENTE los que se citan en amparo. La nota al pie del
«artículo 16 constitucional» abría con «El Congreso Constitucional en el período
ordinario de sus sesiones, que comenzará el 1o. de septiembre de este año…».

Esa parte se arregla en `fase_normas`, no aquí: el texto bueno está en el
acervo y sólo hay que dejar de traer el transitorio. Aquí va lo que el acervo
no tiene.

DE DÓNDE SALE EL TEXTO: del archivo oficial de la Cámara de Diputados que ya
estaba en disco, no de mí. Un precepto escrito de memoria es exactamente la
alucinación que este módulo existe para impedir, así que el JSON se genera
parseando el archivo y se versiona junto al código.
"""

from __future__ import annotations

import json
import os
import re

_AQUI = os.path.dirname(os.path.abspath(__file__))

# La cita nombra la ley de muchas maneras: «la Ley de Amparo», «la Ley
# Reglamentaria de los artículos 103 y 107 constitucionales», «la ley de la
# materia». La última NO se acepta: es ambigua fuera de contexto.
_RX_AMPARO = re.compile(
    r"ley\s+de\s+amparo|ley\s+reglamentaria\s+de\s+los\s+art[íi]culos?\s+103", re.I)


def _cargar() -> dict:
    try:
        with open(os.path.join(_AQUI, "normas_ley_de_amparo.json"),
                  encoding="utf8") as f:
            return json.load(f)
    except Exception as e:          # nunca debe tumbar el pipeline
        print(f"   ⚠️ normas estáticas no disponibles: {e}")
        return {"cuerpo_legal": "", "articulos": {}}


_LA = _cargar()
LEY_DE_AMPARO = _LA.get("articulos", {})
CUERPO_AMPARO = _LA.get("cuerpo_legal", "Ley de Amparo")


def es_ley_de_amparo(cola: str) -> bool:
    """¿La cita nombra la Ley de Amparo?"""
    return bool(_RX_AMPARO.search(cola or ""))


def articulo(num: str | int, cola: str = "") -> dict | None:
    """El artículo de la Ley de Amparo, con su texto oficial. None si no toca.

    Devuelve el mismo diccionario que `fase_normas.recuperar`, para que el
    compositor no tenga que distinguir de dónde vino cada precepto.
    """
    if cola and not es_ley_de_amparo(cola):
        return None
    t = LEY_DE_AMPARO.get(str(num))
    if not t:
        return None
    return {"articulo": str(num), "cuerpo_legal": CUERPO_AMPARO,
            "texto": t[:2000], "citado_como": (cola or "").strip()[:90],
            "fuente": "texto oficial"}


# ── LA PORCIÓN NORMATIVA EXACTA ────────────────────────────────────────────
# David: «citar la porción normativa exacta». Un artículo de la Ley de Amparo
# tiene hasta veinte fracciones y transcribir las veinte para justificar una es
# ruido: quien firma tiene que poder cotejar de un vistazo.
_RX_FRACCION = re.compile(r"(?:^|\s)([IVXLC]{1,7})\.\s")


def fraccion(num: str | int, romano: str) -> str:
    """El texto de UNA fracción. Vacío si no se puede aislar con seguridad.

    Se devuelve vacío —y no una aproximación— cuando la fracción no aparece:
    media fracción atribuida a un artículo es peor que ninguna.
    """
    t = LEY_DE_AMPARO.get(str(num)) or ""
    if not t or not romano:
        return ""
    r = str(romano).strip().upper().rstrip(".")
    marcas = [(m.start(), m.end(), m.group(1)) for m in _RX_FRACCION.finditer(t)]
    for i, (ini, fin, rom) in enumerate(marcas):
        if rom != r:
            continue
        corte = marcas[i + 1][0] if i + 1 < len(marcas) else len(t)
        return f"{r}. " + t[fin:corte].strip().rstrip(",;")
    return ""
