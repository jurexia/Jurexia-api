# -*- coding: utf-8 -*-
"""CUÁNTO LE PIDE A QDRANT UNA CONSULTA DEL CHAT (29-oct-2026).

Reproduce la ráfaga de búsquedas que lanza /chat —la búsqueda principal (Q1),
la constitucional (Q3) cuando la pregunta la pide y los «precedentes
relacionados» que corren en paralelo— contra el Qdrant REAL, en sólo lectura,
y cuenta: cuántas llamadas, cuántas a la vez, cuántos bytes volvieron, cuánto
tardó la fase de búsqueda y cuántas fallaron.

Sirve para comparar dos versiones del código: se le pasa la carpeta y se
importa ESE main.py.

    python scripts/medir_abanico_qdrant.py --codigo <carpeta> --escenario sin_estado
    python scripts/medir_abanico_qdrant.py --codigo <carpeta> --escenario con_estado
    python scripts/medir_abanico_qdrant.py --codigo <carpeta> --preguntas frias.json

OJO CON LA CACHÉ: los vectores de las leyes v4 viven en disco y la RAM del
clúster está llena; una pregunta ya buscada vuelve a costar milisegundos y una
nueva, segundos. Para comparar dos versiones, preguntas NUEVAS en cada brazo
(--preguntas, JSON [["pregunta", "ESTADO" | null], ...]); repetir las mismas
favorece al segundo que corre.

Sin modelos de lenguaje (chat_client=None: el re-ranking de leyes y el resto
de ayudantes se apagan solos). Sólo embeddings: unas 12 llamadas por
consulta a text-embedding-3-small/large con textos de ~60 tokens, menos de un
centavo de dólar por cien consultas. No usa el arranque del servidor (el
lifespan borra las cachés de Gemini de producción): los clientes se arman a
mano.
"""
import argparse
import asyncio
import contextlib
import inspect
import io
import json
import os
import re
import sys
import time
from pathlib import Path

ENV = "/Users/josedavidalcantarmendoza/Documents/IUREXIA-MAC/jurexia-api-git/.env"

PREGUNTAS_SIN_ESTADO = [
    # Sin estado nombrado: la ráfaga grande del 28-oct.
    "¿Cuál es el plazo para promover el juicio de amparo indirecto contra una orden de aprehensión?",
    "Requisitos de la demanda de divorcio incausado y cómo se fijan los alimentos provisionales",
    "¿Cuándo prescribe la acción cambiaria directa derivada de un pagaré?",
]
PREGUNTAS_CON_ESTADO = [
    ("¿Qué plazo tengo para contestar la demanda en la vía ordinaria civil?", "QUERETARO"),
    ("Pensión alimenticia: cómo se calcula y qué pasa si el deudor deja de pagar", "JALISCO"),
    ("Requisitos del arrendamiento y causas de rescisión del contrato", "CIUDAD_DE_MEXICO"),
]


def _cargar_env():
    for l in Path(ENV).read_text().splitlines():
        if "=" in l and not l.lstrip().startswith("#"):
            k, v = l.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"'))


class Contador:
    """Envuelve los métodos del cliente de Qdrant que usa la búsqueda."""

    METODOS = ("query_points", "get_collection", "scroll", "facet", "retrieve",
               "query_batch_points", "count", "search")

    def __init__(self):
        self.reiniciar()

    def reiniciar(self):
        self.llamadas = {}
        self.en_vuelo = 0
        self.pico = 0
        self.errores = 0
        self.con_vectores = 0
        self.bytes = 0
        self.duraciones = []

    def envolver(self, cliente):
        for nombre in self.METODOS:
            orig = getattr(cliente, nombre, None)
            if orig is None:
                continue

            def hacer(nombre=nombre, orig=orig):
                async def envuelto(*a, **kw):
                    self.llamadas[nombre] = self.llamadas.get(nombre, 0) + 1
                    wv = kw.get("with_vectors")
                    if wv not in (None, False):
                        self.con_vectores += 1
                    self.en_vuelo += 1
                    self.pico = max(self.pico, self.en_vuelo)
                    t0 = time.perf_counter()
                    try:
                        return await orig(*a, **kw)
                    except Exception:
                        self.errores += 1
                        raise
                    finally:
                        self.en_vuelo -= 1
                        self.duraciones.append(time.perf_counter() - t0)
                return envuelto
            setattr(cliente, nombre, hacer())

    async def al_responder(self, r):
        await r.aread()
        self.bytes += len(r.content or b"")

    def resumen(self):
        d = sorted(self.duraciones)
        p = lambda q: round(d[min(len(d) - 1, int(len(d) * q))], 2) if d else None
        return {"llamadas": sum(self.llamadas.values()), "por_metodo": self.llamadas,
                "pico_simultaneas": self.pico, "errores": self.errores,
                "con_vectores": self.con_vectores, "MB": round(self.bytes / 1e6, 2),
                "llamada_p50_s": p(0.5), "llamada_p95_s": p(0.95)}


def preparar(codigo: str):
    _cargar_env()
    sys.path.insert(0, codigo)
    os.chdir(codigo)
    with contextlib.redirect_stdout(io.StringIO()):
        import main
    from qdrant_client import AsyncQdrantClient
    from openai import AsyncOpenAI
    from fastembed import SparseTextEmbedding
    cont = Contador()
    cliente = AsyncQdrantClient(url=os.environ["QDRANT_URL"], api_key=os.environ["QDRANT_API_KEY"],
                                timeout=30, event_hooks={"response": [cont.al_responder]})
    cont.envolver(cliente)
    main.qdrant_client = cliente
    main.openai_client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
    main.sparse_encoder = SparseTextEmbedding(model_name="Qdrant/bm25", language="spanish")
    main.chat_client = None
    return main, cont


async def una_consulta(main, cont, pregunta: str, estado):
    """La ráfaga de /chat para una pregunta: Q1 + Q3 (si aplica) + precedentes."""
    plan = {"materia_principal": "general", "fuero_detectado": "mixto",
            "jurisprudencia_keywords": [], "conceptos_clave": [],
            "pesos_silos": {"constitucional": 0.25, "federal": 0.25, "estatal": 0.25, "jurisprudencia": 0.25}}
    comunes = dict(skip_llm_presearch=True, precomputed_plan=plan, precomputed_hyde=None,
                   precomputed_juris_concepts=None, materia_por_palabras=False)
    tareas = [main.hybrid_search_all_silos(query=pregunta, estado=estado, top_k=65, forced_materia=None,
                                           fuero=None, include_sentencias=False, **comunes)]
    low = pregunta.lower()
    if any(k in low for k in ["derechos", "derecho human", "constitución", "cpeum", "amparo",
                              "debido proceso", "garantía", "discriminación", "libertad", "dignidad"]):
        tareas.append(main.hybrid_search_all_silos(
            query=f"derechos humanos constitución CPEUM bloque constitucionalidad artículos 1 14 16 17 20 "
                  f"CPEUM tratados internacionales DDHH principio pro persona {pregunta[:300]}",
            estado=None, top_k=15, forced_materia=None, fuero="constitucional", **comunes))
    kw_prec = {}
    firma = inspect.signature(main.search_precedentes_unified).parameters
    if "una_busqueda_tcc" in firma:
        # Lo mismo que decide /chat (ver _precedentes_con_hilo): sin estado
        # elegido ni nombrado, los siete circuitos en una consulta.
        kw_prec["una_busqueda_tcc"] = main._PRECEDENTES_UNA_BUSQUEDA and not (
            estado or main.detect_single_estado_from_query(pregunta, avisar=False))

    async def precedentes():
        # Como /chat: los precedentes en su propio carril de Qdrant, si existe.
        if hasattr(main, "_CARRIL_QDRANT"):
            main._CARRIL_QDRANT.set(main.QDRANT_SEM_PRECEDENTES)
        return await main.search_precedentes_unified(query=pregunta, limit_scjn=12, limit_tcc=24, **kw_prec)
    tareas.append(precedentes())
    # El vector de la pregunta se pide antes, como en producción (el embedding
    # no es Qdrant y no debe contar en la fase de búsqueda).
    await main.get_dense_embedding(pregunta)
    cont.reiniciar()
    salida = io.StringIO()
    t0 = time.perf_counter()
    with contextlib.redirect_stdout(salida):
        res = await asyncio.gather(*tareas, return_exceptions=True)
    total = time.perf_counter() - t0
    log = salida.getvalue()
    fases = [(int(n), float(s)) for n, s in re.findall(r"Búsqueda en (\d+) silos: ([\d.]+)s", log)]
    r = cont.resumen()
    r.update({
        "pregunta": pregunta[:60], "estado": estado,
        "fase_busqueda_s": [s for _, s in fases], "silos": [n for n, _ in fases],
        "total_s": round(total, 2),
        "resultados": [len(x) if isinstance(x, list) else type(x).__name__ for x in res],
        "errores_en_log": len(re.findall(r"❌ Error en búsqueda", log)),
        "tcc": sum(1 for x in (res[-1] if isinstance(res[-1], list) else []) if "circuito" in (x.silo or "")),
    })
    return r, res


async def correr(main, cont, escenario: str, pausa: float, casos=None):
    if casos is None:
        casos = ([(p, None) for p in PREGUNTAS_SIN_ESTADO] if escenario == "sin_estado"
                 else PREGUNTAS_CON_ESTADO)
    filas = []
    for i, (p, est) in enumerate(casos):
        if i:
            await asyncio.sleep(pausa)   # que el clúster respire entre ráfagas
        r, _ = await una_consulta(main, cont, p, est)
        filas.append(r)
        print(json.dumps(r, ensure_ascii=False), flush=True)
    return filas


def main_cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--codigo", required=True)
    ap.add_argument("--escenario", choices=["sin_estado", "con_estado"], default="sin_estado")
    ap.add_argument("--pausa", type=float, default=20.0)
    # Preguntas propias, en JSON: [["pregunta", "ESTADO" | null], ...]. Para
    # medir con la caché fría hay que usar preguntas que nadie haya buscado.
    ap.add_argument("--preguntas", default=None)
    a = ap.parse_args()
    casos = json.loads(Path(a.preguntas).read_text()) if a.preguntas else None
    main, cont = preparar(a.codigo)
    print(f"# código: {a.codigo} · QDRANT_SEM={getattr(main.QDRANT_SEM, '_value', '?')} · "
          f"escenario {a.escenario}", flush=True)
    asyncio.new_event_loop().run_until_complete(correr(main, cont, a.escenario, a.pausa, casos))


if __name__ == "__main__":
    main_cli()
