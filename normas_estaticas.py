"""LAS DOS LEYES QUE NUNCA DEBEN VENIR DE UNA BÚSQUEDA.

David: «CPEUM y Ley de Amparo estáticos». Tenía razón por dos motivos
distintos, y sólo se ven midiendo el acervo.

LA LEY DE AMPARO NO ESTÁ EN QDRANT. Ni en `leyes_federales`, ni en el bloque
constitucional, ni en las colecciones por materia: cero. Es la ley más citada
de estos cuatro tipos de asunto —el 61, el 63, el 74, el 79, el 93, el 97— y
NINGUNA de sus citas ha llevado nunca su texto a la nota al pie. El fallo era
mudo: `_donde()` la mandaba a `leyes_federales`, allí no había nada, y
`recuperar()` se saltaba el artículo sin decirlo. Ninguna alarma, ninguna nota,
sólo un pie de página que no existía.

Y menos mal que era mudo. Si `_elegir()` hubiera sido menos estricto, «artículo
79 de la Ley de Amparo» habría cogido el 79 de la Ley del ISR o el de la Ley
Federal de Procedimiento Contencioso Administrativo, que sí están. La única
razón de que no ocurriera es que ninguna de esas leyes comparte la palabra
«amparo» y el marcador de sobrantes las hundía a todas.

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
