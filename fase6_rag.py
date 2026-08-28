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
from typing import Awaitable, Callable, Optional

log = logging.getLogger("fase6_rag")

import fase6_estudio as f6

COLECCION_JURIS = "jurisprudencia_nacional_v3"
VECTOR_RUBRO = "rubro"          # el ganador medido, con pregunta conceptual
COLECCION_FEDERAL = "leyes_federales"

TESIS_POR_PROBLEMA = 6          # 6 entran en el prompt sin ahogar el material
NORMAS_POR_PROBLEMA = 4


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

    # Las obligatorias primero: en una sentencia la jerarquía de la fuente es
    # parte del argumento, no un detalle de presentación.
    tesis = [_tesis_de(p) for p in res[0]]
    tesis.sort(key=lambda t: not t["obligatoria"])
    vistos: set[str] = set()
    unicas: list[dict] = []
    for t in tesis:
        if t["registro"] and t["registro"] not in vistos:
            vistos.add(t["registro"])
            unicas.append(t)

    normas = [_norma_de(p) for grupo in res[1:] for p in grupo]

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
    partes = await asyncio.gather(*[
        material_para(qdrant, embed_juris, embed_leyes, p, coleccion_estatal)
        for p in problemas if (p or "").strip()])

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

    tesis.sort(key=lambda t: not t["obligatoria"])
    return f6.Material(tesis=tesis, normas=normas)
