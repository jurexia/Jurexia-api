"""LA BÓVEDA · EL ÍNDICE EN QDRANT.   (oct-2026)

Una sola colección para todos (`boveda`), partida por inquilino: cada punto
lleva `user_id`, `carpeta_id`, `doc_id` y `pagina`, y TODA lectura filtra por
`user_id` y `carpeta_id`. No hay búsqueda sin dueño: las funciones de aquí lo
exigen como argumento, no como opción.

POR QUÉ ASÍ, CON 1 CPU Y 3.4 GB DE RAM (2.1 GB ya ocupados por el acervo):
  · Vectores, índice HNSW y payload en DISCO; la cuantización int8 también
    en disco (always_ram=False). La Bóveda no le quita memoria al acervo.
  · HNSW por inquilino (m=0, payload_m=16) con `user_id` como índice de
    inquilino: Qdrant no construye un grafo global que nadie usaría, y la
    búsqueda de un usuario sólo lee lo suyo.
  · Híbrida: densa (text-embedding-3-small, la misma del acervo) + BM25 con
    IDF calculado por Qdrant, fusionadas por RRF. En contratos manda lo
    literal: nombres de partes, montos, números de cláusula.

Cliente propio y no el del chat: el del chat lleva el veto del selector de
«Fuentes» (fuentes_elegidas.py), que no conoce esta colección.
"""
from __future__ import annotations

import asyncio
import os
import uuid
from dataclasses import dataclass
from typing import Optional

COLECCION = os.getenv("BOVEDA_COLECCION", "boveda")
MODELO_DENSO = "text-embedding-3-small"
DIM = 1536
PRECIO_EMBEDDING = 0.02 / 1_000_000     # USD por token
LOTE_EMBEDDING = 96

_qdrant = None
_openai = None
_bm25 = None
_lista = False
_cerrojo = asyncio.Lock()


def configurar(openai_client=None) -> None:
    """main.py presta su cliente de OpenAI. El BM25 NO se comparte: el del
    acervo es el de fábrica (inglés) y aquí se quiere español —sin «de», «la»,
    «que» y con «tribunales» y «tribunal» en la misma raíz—."""
    global _openai
    if openai_client is not None:
        _openai = openai_client


def qdrant():
    global _qdrant
    if _qdrant is None:
        from qdrant_client import AsyncQdrantClient
        _qdrant = AsyncQdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"), timeout=60)
    return _qdrant


def _cliente_openai():
    global _openai
    if _openai is None:
        from openai import AsyncOpenAI
        _openai = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _openai


def _encoder_bm25():
    global _bm25
    if _bm25 is None:
        from fastembed import SparseTextEmbedding
        _bm25 = SparseTextEmbedding(model_name="Qdrant/bm25", language="spanish")
    return _bm25


async def asegurar_coleccion() -> None:
    """Crea la colección y sus índices si no existen. Idempotente."""
    global _lista
    if _lista:
        return
    async with _cerrojo:
        if _lista:
            return
        from qdrant_client import models as m
        q = qdrant()
        if not await q.collection_exists(COLECCION):
            await q.create_collection(
                collection_name=COLECCION,
                vectors_config={"dense": m.VectorParams(size=DIM, distance=m.Distance.COSINE, on_disk=True)},
                sparse_vectors_config={"bm25": m.SparseVectorParams(
                    modifier=m.Modifier.IDF, index=m.SparseIndexParams(on_disk=True))},
                hnsw_config=m.HnswConfigDiff(m=0, payload_m=16, on_disk=True),
                quantization_config=m.ScalarQuantization(scalar=m.ScalarQuantizationConfig(
                    type=m.ScalarType.INT8, quantile=0.99, always_ram=False)),
                on_disk_payload=True,
            )
            await q.create_payload_index(COLECCION, "user_id", field_schema=m.KeywordIndexParams(
                type=m.KeywordIndexType.KEYWORD, is_tenant=True))
            for campo in ("carpeta_id", "doc_id"):
                await q.create_payload_index(COLECCION, campo, field_schema=m.PayloadSchemaType.KEYWORD)
            await q.create_payload_index(COLECCION, "pagina", field_schema=m.PayloadSchemaType.INTEGER)
            print(f"   🗄️ Bóveda: colección «{COLECCION}» creada")
        _lista = True


# ── Vectores ───────────────────────────────────────────────────────────────
@dataclass
class Uso:
    tokens: int = 0

    @property
    def usd(self) -> float:
        return self.tokens * PRECIO_EMBEDDING


async def densos(textos: list[str], uso: Optional[Uso] = None) -> list[list[float]]:
    """Embeddings por lotes. Un fragmento de la Bóveda mide ~1,400 caracteres:
    lejos del tope de 8,192 tokens, pero se acota igual por si acaso."""
    out: list[list[float]] = []
    cli = _cliente_openai()
    for i in range(0, len(textos), LOTE_EMBEDDING):
        lote = [t[:14_000] or " " for t in textos[i:i + LOTE_EMBEDDING]]
        for intento in range(4):
            try:
                r = await cli.embeddings.create(model=MODELO_DENSO, input=lote)
                break
            except Exception:
                if intento == 3:
                    raise
                await asyncio.sleep(1.5 * (intento + 1))
        out.extend(d.embedding for d in r.data)
        if uso is not None and getattr(r, "usage", None):
            uso.tokens += int(r.usage.total_tokens or 0)
    return out


def _dispersos(textos: list[str], consulta: bool = False):
    from qdrant_client import models as m
    enc = _encoder_bm25()
    vecs = list(enc.query_embed(textos[0])) if consulta else list(enc.embed(textos))
    return [m.SparseVector(indices=v.indices.tolist(), values=v.values.tolist()) for v in vecs]


def _texto_indexado(nombre: str, pagina: int, texto: str) -> str:
    # El nombre del archivo va dentro: «el contrato con Textiles Cuauhtémoc»
    # suele estar en el nombre antes que en la primera página.
    return f"{nombre} — página {pagina}\n{texto}"


def id_punto(doc_id: str, pagina: int, orden: int) -> str:
    """Determinista: reingestar un documento reescribe sus puntos, no los duplica."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"boveda:{doc_id}:{pagina}:{orden}"))


async def indexar_documento(*, user_id: str, carpeta_id: str, doc_id: str, nombre: str,
                            fragmentos, uso: Optional[Uso] = None) -> int:
    """Sube los fragmentos (boveda_texto.Fragmento) de un documento. Devuelve cuántos."""
    from qdrant_client import models as m
    if not fragmentos:
        return 0
    await asegurar_coleccion()
    textos = [_texto_indexado(nombre, f.pagina, f.texto) for f in fragmentos]
    vd = await densos(textos, uso)
    vs = await asyncio.to_thread(_dispersos, textos)
    puntos = [
        m.PointStruct(
            id=id_punto(doc_id, f.pagina, f.orden),
            vector={"dense": d, "bm25": s},
            payload={"user_id": user_id, "carpeta_id": carpeta_id, "doc_id": doc_id,
                     "doc_nombre": nombre, "pagina": f.pagina, "orden": f.orden, "texto": f.texto},
        )
        for f, d, s in zip(fragmentos, vd, vs)
    ]
    q = qdrant()
    for i in range(0, len(puntos), 128):
        await q.upsert(COLECCION, points=puntos[i:i + 128], wait=True)
    return len(puntos)


def _filtro(user_id: str, carpeta_id: Optional[str] = None, doc_ids: Optional[list[str]] = None):
    from qdrant_client import models as m
    if not user_id:
        raise ValueError("La Bóveda no se lee sin dueño.")
    must = [m.FieldCondition(key="user_id", match=m.MatchValue(value=user_id))]
    if carpeta_id:
        must.append(m.FieldCondition(key="carpeta_id", match=m.MatchValue(value=carpeta_id)))
    if doc_ids:
        must.append(m.FieldCondition(key="doc_id", match=m.MatchAny(any=list(doc_ids))))
    return m.Filter(must=must)


@dataclass
class Hallazgo:
    id: str            # el del punto: es el [Doc ID] que cita el chat
    doc_id: str
    doc_nombre: str
    pagina: int
    texto: str
    puntaje: float


async def buscar(consulta: str, *, user_id: str, carpeta_id: str, limite: int = 12,
                 doc_ids: Optional[list[str]] = None) -> list[Hallazgo]:
    """Búsqueda híbrida (densa + BM25, RRF) dentro de la carpeta del usuario."""
    from qdrant_client import models as m
    await asegurar_coleccion()
    filtro = _filtro(user_id, carpeta_id, doc_ids)
    vd = (await densos([consulta]))[0]
    vs = (await asyncio.to_thread(_dispersos, [consulta], True))[0]
    r = await qdrant().query_points(
        COLECCION,
        prefetch=[
            m.Prefetch(query=vd, using="dense", filter=filtro, limit=limite * 3),
            m.Prefetch(query=vs, using="bm25", filter=filtro, limit=limite * 3),
        ],
        query=m.FusionQuery(fusion=m.Fusion.RRF),
        limit=limite,
        with_payload=True,
    )
    return [Hallazgo(id=str(p.id), doc_id=p.payload["doc_id"], doc_nombre=p.payload.get("doc_nombre", ""),
                     pagina=int(p.payload.get("pagina") or 1), texto=p.payload.get("texto", ""),
                     puntaje=float(p.score or 0)) for p in r.points]


async def contar(user_id: str, carpeta_id: Optional[str] = None) -> int:
    r = await qdrant().count(COLECCION, count_filter=_filtro(user_id, carpeta_id), exact=True)
    return int(r.count)


async def borrar(*, user_id: str, carpeta_id: Optional[str] = None, doc_id: Optional[str] = None) -> None:
    """Borra los vectores de un documento, de una carpeta o de TODO el usuario."""
    from qdrant_client import models as m
    if not await qdrant().collection_exists(COLECCION):
        return
    await qdrant().delete(COLECCION, points_selector=m.FilterSelector(
        filter=_filtro(user_id, carpeta_id, [doc_id] if doc_id else None)), wait=True)
