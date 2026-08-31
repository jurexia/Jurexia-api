"""El RAG del estudio de fondo: de un problema jurídico al material para fundarlo.

POR QUÉ ESTE MÓDULO BUSCA DISTINTO QUE EL CHAT
══════════════════════════════════════════════

La medición sobre 404 tesis realmente citadas en 828 engroses del disco KINGSTON
dejó dos conclusiones que aquí se aprovechan enteras:

    v3 conceptual → vector RUBRO ..... 50% en 1ª posición · 75% en top-10  ← ésta
    v3 conceptual → vector texto ..... 44% / 69%
    v3 prosa      → vector RUBRO .....  3% / 15%          ← lo peor de todo

Es decir: **cómo se pregunta importa diez veces más que con qué se busca**, y el
vector del rubro sólo rinde cuando la consulta es una pregunta conceptual.

Y aquí se da la coincidencia que hace barato todo esto: la Fase 3 ya redacta los
problemas jurídicos **como preguntas** —«¿Puede condenarse a pensión
compensatoria invirtiendo la carga de la prueba?»— porque así es como se
resuelven en una sentencia. O sea que la entrada del RAG ya viene en el formato
que mejor recupera, sin reformular nada ni pagar una llamada extra al modelo.

LA OBLIGATORIEDAD MANDA EN EL ORDEN
═══════════════════════════════════
`vincula` separa 17.930 tesis obligatorias de 53.998 orientadoras. En un chat da
igual el orden; en una sentencia no: una jurisprudencia obligatoria en contra
cambia el sentido, y una aislada sólo ilustra. Se buscan las dos, pero las
obligatorias van primero y se marcan como tales para que el estudio las trate
como lo que son.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import re
from typing import Awaitable, Callable, Optional

log = logging.getLogger("fase6_rag")

import fase6_estudio as f6

COLECCION_JURIS = "jurisprudencia_nacional_v3"
VECTOR_RUBRO = "rubro"          # el ganador medido, con pregunta conceptual
COLECCION_FEDERAL = "leyes_federales"

TESIS_POR_PROBLEMA = 6          # 6 entran en el prompt sin ahogar el material
NORMAS_POR_PROBLEMA = 4


# El rubro de una tesis local lo dice sin ambigüedad: «(LEGISLACIÓN DEL ESTADO
# DE PUEBLA)». Un criterio sobre el código de otra entidad no rige aquí, por
# obligatorio que sea en su circuito.
_RX_LEGISLACION = re.compile(r"LEGISLACI[ÓO]N\s+(?:DEL?\s+)?(?:ESTADO\s+DE\s+)?"
                             r"([A-ZÁÉÍÓÚÑ ]{4,40}?)\s*\)", re.I)

_ESTADO_DE_COLECCION = {
    "leyes_queretaro": "QUERETARO", "leyes_jalisco": "JALISCO",
    "leyes_cdmx": "CIUDAD DE MEXICO", "leyes_nuevo_leon": "NUEVO LEON",
    "leyes_guanajuato": "GUANAJUATO", "leyes_puebla": "PUEBLA",
}


def _sin_acentos(x: str) -> str:
    import unicodedata
    return "".join(c for c in unicodedata.normalize("NFKD", (x or "").upper())
                   if not unicodedata.combining(c))


def _de_otro_estado(t: dict, coleccion: Optional[str]) -> bool:
    """True si la tesis interpreta la legislación de OTRA entidad."""
    m = _RX_LEGISLACION.search(t.get("rubro", "") or "")
    if not m:
        return False
    mia = _ESTADO_DE_COLECCION.get((coleccion or "").lower(), "")
    suya = _sin_acentos(m.group(1)).strip()
    return bool(mia) and mia not in suya and suya not in mia


def _es_scjn(t: dict) -> bool:
    """Pleno o Salas. David: «preferentemente jurisprudencia de la Suprema Corte»."""
    inst = _sin_acentos(t.get("instancia", ""))
    return any(x in inst for x in ("PRIMERA SALA", "SEGUNDA SALA", "PLENO",
                                   "SUPREMA CORTE"))


def _tesis_de(p: dict) -> dict:
    return {
        "registro": str(p.get("registro") or ""),
        "instancia": p.get("instancia") or "",
        "rubro": p.get("rubro") or "",
        "texto": p.get("texto") or "",
        "tipo": p.get("tipo") or "",
        "obligatoria": bool(p.get("vincula")),
        "localizacion": p.get("localizacion") or "",
    }


def _norma_de(p: dict) -> dict:
    return {
        "cuerpo_legal": p.get("cuerpo_legal_oficial") or p.get("ref") or "",
        "articulo": p.get("articulo_num") or "",
        "texto": p.get("texto") or p.get("contenido") or "",
        "entidad": p.get("entidad") or "",
    }


async def _buscar(qdrant, coleccion: str, vector: str, v: list[float],
                  limite: int, filtro=None) -> list[dict]:
    """Una consulta a Qdrant, tolerante con la colección que falta y RUIDOSA con
    el resto.

    La primera versión devolvía `[]` ante cualquier excepción y el estudio salía
    sin una sola tesis, indistinguible de «no hay nada que citar». El error real
    era que el cliente ya no tiene `.search` —hoy es `query_points`— y quedó
    escondido tras el `except`. Una búsqueda que falla se REGISTRA; sólo el
    estado sin ingestar es un vacío legítimo.
    """
    try:
        r = qdrant.query_points(collection_name=coleccion, query=v,
                                using=vector, limit=limite,
                                query_filter=filtro, with_payload=True)
        if inspect.isawaitable(r):          # AsyncQdrantClient, el de main.py
            r = await r
        return [p.payload or {} for p in r.points]
    except Exception as e:
        texto = str(e).lower()
        if "not found" in texto or "doesn" in texto:
            log.info("colección %s sin ingestar; se sigue sin ella", coleccion)
        else:
            log.error("búsqueda fallida en %s/%s: %s: %s",
                      coleccion, vector, type(e).__name__, e)
        return []


# ═══════════════════════════════════════════════════════════════════════════
# EL ARTÍCULO ENTERO, NO EL TROZO QUE CASÓ
# ═══════════════════════════════════════════════════════════════════════════
# En el ADL 382/2024 —un trabajador despedido por faltas— el motor propuso
# INFUNDADO cinco veces razonando sobre si las incapacidades se habían entregado
# a tiempo. Nunca discutió lo único que decide el asunto: que el artículo 47,
# fracción X, de la Ley Federal del Trabajo exige que las faltas sean «SIN CAUSA
# JUSTIFICADA», y una incapacidad médica real es causa justificada, se entregue
# el papel cuando se entregue.
#
# No fue un fallo de razonamiento. ES QUE NUNCA LO LEYÓ. El artículo 47 sí
# llegó al material… con 600 caracteres: el encabezado y la fracción I. El
# artículo tiene quince fracciones y está troceado; la búsqueda por parecido
# devuelve EL TROZO que casó, y casó el primero. La fracción X se quedó fuera.
#
# Un trozo no es un artículo. Cuando la ley se trae para fundar, se trae entera:
# `fase_normas` ya lo hacía para las notas al pie —«el artículo entero: viene
# troceado y un trozo no es el artículo»— y el material que sostiene el
# RAZONAMIENTO iba sin ese cuidado, que es donde más falta hace.

async def _completar(qdrant, coleccion: str, norma: dict) -> dict:
    """Los demás trozos del mismo artículo y la misma ley, en orden."""
    from qdrant_client.models import FieldCondition, Filter, MatchValue
    num, ley = norma.get("articulo"), str(norma.get("cuerpo_legal") or "")
    if not num or not ley:
        return norma
    try:
        r = qdrant.scroll(
            collection_name=coleccion,
            scroll_filter=Filter(must=[FieldCondition(
                key="articulo_num", match=MatchValue(value=int(num)))]),
            limit=60, with_payload=True)
        if inspect.isawaitable(r):
            r = await r
        pts = r[0] if isinstance(r, tuple) else r
    except Exception as e:
        log.error("no se pudo completar el artículo %s: %s", num, e)
        return norma
    # SÓLO LOS DE LA MISMA LEY. El artículo 47 existe en decenas de códigos y
    # juntarlos daría un texto que no es de ninguno.
    suyos = [x.payload for x in (pts or [])
             if str((x.payload or {}).get("cuerpo_legal_oficial") or "") == ley]
    if len(suyos) < 2:
        return norma
    partes, visto = [], set()
    for pl in sorted(suyos, key=lambda z: int(z.get("chunk_index") or 0)):
        # La migaja de cabecera «[Ley … | CAPITULO IV …]» se repite en cada
        # trozo; una vez basta y en los demás estorba.
        txt = re.sub(r"^\s*\[[^\]]{0,250}\]\s*", "",
                     " ".join(str(pl.get("texto") or "").split()))
        if txt and txt not in visto:
            visto.add(txt)
            partes.append(txt)
    entero = " ".join(partes)
    if len(entero) > len(str(norma.get("texto") or "")):
        norma = dict(norma)
        norma["texto"] = entero[:6000]
        norma["completo"] = True
    return norma


async def material_para(qdrant, embed_juris, embed_leyes,
                        problema: str, coleccion_estatal: Optional[str] = None,
                        ) -> f6.Material:
    """El material verificado para UN problema jurídico.

    `embed_juris` vectoriza con el modelo de la v3 (3072 dim) y `embed_leyes`
    con el de las colecciones de leyes: son modelos distintos y cruzarlos
    devuelve ruido con buena puntuación, que es la peor clase de error.
    """
    v_juris, v_leyes = await asyncio.gather(embed_juris(problema),
                                            embed_leyes(problema))

    colecciones = [c for c in (coleccion_estatal, COLECCION_FEDERAL) if c]
    tareas = [_buscar(qdrant, COLECCION_JURIS, VECTOR_RUBRO, v_juris,
                      TESIS_POR_PROBLEMA * 2)]
    tareas += [_buscar(qdrant, c, "dense", v_leyes, NORMAS_POR_PROBLEMA)
               for c in colecciones]
    res = await asyncio.gather(*tareas)

    # EL ORDEN, CORREGIDO. La primera versión de esto penalizaba la tesis por
    # venir de otra entidad y la mandaba al fondo. Estaba mal, y el barrido de
    # 139 documentos de este tribunal lo demuestra: hay decenas de criterios
    # sobre legislación ajena invocados con toda naturalidad —la XVI.1o.A. J/54,
    # sobre la Ley de Hacienda de GUANAJUATO, sostiene el interés jurídico en
    # SEIS amparos queretanos— y CERO aplicaciones de ley de otra entidad.
    #
    # Lo que no entra es la LEY ajena, y eso no se arregla ordenando tesis: se
    # arregla en el prompt y en el verificador. Aquí sólo se ordena por peso.
    #
    #   1. Que venga de la Suprema Corte (Pleno o Salas): es obligatoria por el
    #      artículo 217 y la legislación que interpretó es irrelevante.
    #   2. Que sea obligatoria.
    #   3. A igualdad de lo anterior, antes la de Querétaro que la de fuera.
    tesis = [_tesis_de(p) for p in res[0]]
    tesis.sort(key=lambda t: (not _es_scjn(t),
                              not t["obligatoria"],
                              _de_otro_estado(t, coleccion_estatal)))
    vistos: set[str] = set()
    unicas: list[dict] = []
    for t in tesis:
        if t["registro"] and t["registro"] not in vistos:
            vistos.add(t["registro"])
            unicas.append(t)

    normas = [_norma_de(p) for grupo in res[1:] for p in grupo]
    # Cada norma, completada con el resto de su articulado. Van en paralelo y
    # es un scroll por artículo: el mismo precio que ya paga la nota al pie.
    if normas:
        pares = []
        for grupo, col in zip(res[1:], colecciones):
            pares += [(col, _norma_de(p)) for p in grupo]
        normas = list(await asyncio.gather(
            *[_completar(qdrant, c, n) for c, n in pares]))

    return f6.Material(tesis=unicas[:TESIS_POR_PROBLEMA],
                       normas=normas[:NORMAS_POR_PROBLEMA * 2])


async def material_del_caso(qdrant, embed_juris, embed_leyes,
                            problemas: list[str],
                            coleccion_estatal: Optional[str] = None,
                            ) -> f6.Material:
    """Un solo Material con lo de TODOS los problemas, sin repetir tesis.

    El estudio se escribe de una vez —es una sola pieza de prosa— así que el
    material también se le entrega de una vez, deduplicado por registro y por
    artículo. Si se le mandara por problema, citaría la misma tesis tres veces.
    """
    # Se acepta la pregunta suelta o el problema entero de la Fase 3: que el
    # módulo aguante las dos formas cuesta tres líneas y evita repetir el fallo
    # en cada sitio que lo llame.
    preguntas = []
    for p in (problemas or []):
        q = p.get("pregunta", "") if isinstance(p, dict) else str(p or "")
        if q.strip():
            preguntas.append(q.strip())

    partes = await asyncio.gather(*[
        material_para(qdrant, embed_juris, embed_leyes, p, coleccion_estatal)
        for p in preguntas])

    tesis, normas = [], []
    r_vistos, n_vistos = set(), set()
    for m in partes:
        for t in m.tesis:
            if t["registro"] not in r_vistos:
                r_vistos.add(t["registro"])
                tesis.append(t)
        for n in m.normas:
            clave = (n["cuerpo_legal"], str(n["articulo"]))
            if clave not in n_vistos:
                n_vistos.add(clave)
                normas.append(n)

    tesis.sort(key=lambda t: (not _es_scjn(t), not t["obligatoria"],
                              _de_otro_estado(t, coleccion_estatal)))
    return f6.Material(tesis=tesis, normas=normas)
