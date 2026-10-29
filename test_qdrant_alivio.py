# -*- coding: utf-8 -*-
"""Alivio de Qdrant y respaldo ante el filtro de OpenAI (29-oct-2026).

Revisión de capacidad del 28-oct: un nodo de 0.5 vCPU y 4 GiB con la RAM
llena; una consulta del chat sin estado lanzaba 41-44 búsquedas a la vez y
unas 240 al día morían a los 30 s («ResponseHandlingException»).

LO MEDIDO (scripts/medir_abanico_qdrant.py contra el Qdrant real, en sólo
lectura, desde fuera de Render; «antes» = origin/main 151cc05):

  Consulta SIN estado (3 preguntas, la ráfaga de /chat: Q1 + Q3 + precedentes)
                          antes                  después
    silos de Q1           41                     5
    llamadas a Qdrant     104-152                23-61
    a la vez (pico)       50                     8 (6 + 2 de precedentes)
    caídas a los 30 s     29-40 por consulta     0
    MB recibidos          7.5-20.2               1.9-5.7
    fase de búsqueda      31.2-32.3 s            1.0-1.5 s
    total de la ráfaga    42.8-48.2 s            3.7-4.7 s

  Consulta CON estado, 6 preguntas nuevas por brazo (caché fría), medianas
    fase de búsqueda      7.7 s (4.5-10.3)       3.6 s (3.0-4.3)
    total                 9.3 s (6.3-12.5)       5.6 s (4.6-6.4)
    MB recibidos          8.0                    1.9
  El semáforo, aislado (código nuevo, un solo usuario, otras 6 preguntas
  frías por brazo): fase 3.2 s sin tope, 4.3 s con 6 para todo —los siete
  circuitos de precedentes ocupaban los seis lugares y la ley esperaba— y
  3.6 s con los precedentes en su carril de 2. Lo que queda es variación de
  pregunta a pregunta: en la línea de tiempo la búsqueda de la ley arranca
  sin esperar (ver QDRANT_SEM_PRECEDENTES).

  Una llamada híbrida (12 preguntas × 4 colecciones): con vectores 2.7 MB,
  servidor 450 ms y reloj 2.4 s (p50); sin vectores + coseno por id 262 KB,
  33 ms y 0.34 s. Cosenos idénticos en 4,680 puntos (diferencia máxima
  0.00027).

  Prefetch con caché fría (20 preguntas nuevas): colección del estado 520→130,
  servidor 8.0 → 2.8 s (p50); federal 260→100, 356 → 163 ms.

Parte 1, sin red: qué colecciones se buscan, la ficha guardada, el semáforo,
los cosenos sin vectores, el prefetch, los circuitos en una consulta y el
respaldo ante el filtro de contenido, con un Qdrant y un OpenAI falsos.

Parte 2, Qdrant REAL en sólo lectura (sólo si hay credenciales en el .env):
la misma llamada híbrida a la manera de antes y a la de ahora —bytes, tiempo
y cosenos punto por punto— y la consulta sin estado contada. Embeddings: tres
llamadas de ~20 tokens (menos de un centavo). Sin modelos de lenguaje y sin el
arranque del servidor (el lifespan borra las cachés de Gemini de producción).

    python test_qdrant_alivio.py
"""
import asyncio
import contextlib
import io
import os
import re
import sys
import time
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
FALLOS = []


def ok(cond, que):
    print(("   PASA   " if cond else "   FALLA  ") + que)
    if not cond:
        FALLOS.append(que)


_BUCLE = asyncio.new_event_loop()


def correr(co):
    # Un solo bucle: el semáforo de main.py queda atado al primero que lo usa.
    return _BUCLE.run_until_complete(co)


def callado(co):
    with contextlib.redirect_stdout(io.StringIO()) as f:
        r = correr(co)
    return r, f.getvalue()


# main.py lee el .env del directorio actual; en un worktree no hay, y sin
# claves ni siquiera se importa (los clientes de OpenAI nacen al importar).
_ENV = Path("/Users/josedavidalcantarmendoza/Documents/IUREXIA-MAC/jurexia-api-git/.env")
if not Path(".env").exists() and _ENV.exists():
    from dotenv import load_dotenv
    load_dotenv(_ENV)

with contextlib.redirect_stdout(io.StringIO()):
    import main
from qdrant_client.http import models

FUENTE = Path(main.__file__).read_text(encoding="utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# Un Qdrant falso: anota cada llamada y cuántas van a la vez.
# ─────────────────────────────────────────────────────────────────────────────
class QdrantFalso:
    def __init__(self, espera=0.0, puntos=None, cosenos=None):
        self.llamadas = []
        self.en_vuelo = 0
        self.pico = 0
        self.espera = espera
        self.puntos = puntos or {}      # colección → lista de ScoredPoint
        self.cosenos = cosenos or {}    # id → coseno (para la consulta has_id)

    async def _anota(self, metodo, coleccion, kw):
        self.llamadas.append((metodo, coleccion, kw))
        self.en_vuelo += 1
        self.pico = max(self.pico, self.en_vuelo)
        try:
            if self.espera:
                await asyncio.sleep(self.espera)
        finally:
            self.en_vuelo -= 1

    async def query_points(self, collection_name, **kw):
        await self._anota("query_points", collection_name, kw)
        flt = kw.get("query_filter")
        ids = [i for c in (getattr(flt, "must", None) or []) if isinstance(c, models.HasIdCondition)
               for i in c.has_id]
        if ids:
            return models.QueryResponse(points=[
                models.ScoredPoint(id=i, version=1, score=self.cosenos.get(i, 0.0)) for i in ids])
        return models.QueryResponse(points=list(self.puntos.get(collection_name, [])))

    async def get_collection(self, collection_name, **kw):
        await self._anota("get_collection", collection_name, kw)
        sparse = ({"sparse": types.SimpleNamespace(modifier="idf")}
                  if collection_name.startswith("leyes_") else None)
        return types.SimpleNamespace(config=types.SimpleNamespace(params=types.SimpleNamespace(
            sparse_vectors=sparse)), payload_schema={})

    async def facet(self, collection_name, **kw):
        await self._anota("facet", collection_name, kw)
        return types.SimpleNamespace(hits=[])

    async def scroll(self, collection_name, **kw):
        await self._anota("scroll", collection_name, kw)
        return ([], None)

    async def retrieve(self, collection_name, **kw):
        await self._anota("retrieve", collection_name, kw)
        return []

    def colecciones(self, metodo="query_points"):
        return [c for m, c, _ in self.llamadas if m == metodo]


async def _vector_falso(texto, modelo=None):
    return [0.01] * (3072 if modelo == main.EMBEDDING_MODEL_JURIS else 1536)

main.get_dense_embedding = _vector_falso
main.chat_client = None          # sin re-ranking por modelo ni ayudantes
main.sparse_encoder = None       # sparse vacío: la v4 va densa (la híbrida se prueba aparte)

ESTATALES = set(main.ESTADO_SILO.values())
PLAN = {"materia_principal": "general", "fuero_detectado": "mixto", "jurisprudencia_keywords": [],
        "conceptos_clave": [], "pesos_silos": {"constitucional": 0.25, "federal": 0.25,
                                               "estatal": 0.25, "jurisprudencia": 0.25}}


def buscar(q, estado=None, fuero=None, top_k=65):
    main._INFO_COLECCION.clear()
    main.qdrant_client = QdrantFalso()
    # skip_post_search=False, como el chat: con True la función revienta
    # más abajo (`existing_ids` sin definir; nadie la llama así).
    _, log = callado(main.hybrid_search_all_silos(
        query=q, estado=estado, top_k=top_k, fuero=fuero, skip_llm_presearch=True,
        precomputed_plan=PLAN, materia_por_palabras=False))
    m = re.search(r"Búsqueda en (\d+) silos", log)
    return main.qdrant_client, (int(m.group(1)) if m else None), log


print("\n1 · SIN ESTADO NO SE ABREN LAS 32 COLECCIONES ESTATALES")
Q_SIN = "¿Qué requisitos debe cumplir el contrato de arrendamiento y cuándo procede su rescisión?"
cli, n, log = buscar(Q_SIN)
buscadas = set(cli.colecciones())
ok(not (buscadas & ESTATALES), f"ninguna colección estatal ({sorted(buscadas & ESTATALES)})")
ok({"leyes_federales", "bloque_constitucional", "jurisprudencia_nacional_v3"} <= buscadas,
   "federal, bloque constitucional y jurisprudencia sí")
ok(n == 5, f"«Búsqueda en 5 silos» (antes 41) → {n}")

cli, n, log = buscar("¿Qué plazo hay para contestar la demanda civil en Jalisco?")
ok(set(cli.colecciones()) & ESTATALES == {"leyes_jalisco"},
   f"si la pregunta nombra un estado, sólo ése → {sorted(set(cli.colecciones()) & ESTATALES)}")

cli, n, log = buscar(Q_SIN, fuero="estatal")
b = set(cli.colecciones())
ok(not (b & ESTATALES) and {"leyes_federales", "bloque_constitucional"} <= b,
   "«FUERO MULTI estatal» sin estado: ni una estatal, y lo que rige en toda la República ocupa su lugar")
ok("FUERO MULTI: ['estatal'] → 3 silos" in log, "el registro dice 3 silos (antes 33)")

cli, n, log = buscar(Q_SIN, estado="QUERETARO")
q_est = [kw for m, c, kw in cli.llamadas if m == "query_points" and c == "leyes_queretaro"]
ok(q_est and q_est[0]["limit"] == 130, "con estado elegido: su colección, con top_k × 2 como siempre")
ok(not (set(cli.colecciones()) & (ESTATALES - {"leyes_queretaro"})), "y ninguna otra estatal")

main._ABANICO_ESTATAL_COMPLETO = True
cli, n, log = buscar(Q_SIN)
import collections
veces = collections.Counter(c for c in cli.colecciones() if c in ESTATALES)
ok(len(veces) == 32 and n == 37 and len(set(veces.values())) == 1,
   f"ABANICO_ESTATAL=todos: las 32, cada una las mismas veces (antes Edomex iba triple) → {n} silos")
main._ABANICO_ESTATAL_COMPLETO = False

main.fuentes_sel.fijar(frozenset({"estatal"}))
cli, n, log = buscar(Q_SIN, fuero="estatal")
ok(len(set(cli.colecciones()) & ESTATALES) == 32,
   "sólo lo estatal encendido en el selector y sin decir dónde: se abren como antes")
main.fuentes_sel.fijar(None)

ok(len(main.ESTADO_SILO) == 36 and len(ESTATALES) == 32,
   "ESTADO_SILO trae 36 llaves para 32 colecciones: por eso leyes_edomex se buscaba tres veces")


print("\n2 · LA FICHA DE LA COLECCIÓN SE PIDE UNA VEZ")
main._INFO_COLECCION.clear()
main.qdrant_client = QdrantFalso()
for _ in range(10):
    callado(main.hybrid_search_single_silo("leyes_cdmx", "renta", [0.1] * 1536,
                                           models.SparseVector(indices=[], values=[]), None, 5, 0.7))
ok(main.qdrant_client.colecciones("get_collection") == ["leyes_cdmx"],
   f"10 búsquedas, 1 get_collection (antes 10, más uno por cada respaldo) → "
   f"{len(main.qdrant_client.colecciones('get_collection'))}")
main._INFO_COLECCION["leyes_cdmx"] = (time.time() - main._INFO_COLECCION_TTL - 1,
                                      main._INFO_COLECCION["leyes_cdmx"][1])
callado(main.hybrid_search_single_silo("leyes_cdmx", "renta", [0.1] * 1536,
                                       models.SparseVector(indices=[], values=[]), None, 5, 0.7))
ok(len(main.qdrant_client.colecciones("get_collection")) == 2, "caducada la ficha, se vuelve a pedir")
ok(FUENTE.count("qdrant_client.get_collection(") == 2,
   "sólo quedan dos get_collection: _info_coleccion y el índice de payload (también guardado)")


print("\n3 · EL SEMÁFORO DE QDRANT: 6 A LA VEZ, Y ENVUELVE LA BÚSQUEDA")
ok(main.QDRANT_SEM._value == 6, f"QDRANT_SEM = {main.QDRANT_SEM._value} (antes 100)")
main._INFO_COLECCION.clear()
main.qdrant_client = QdrantFalso(espera=0.02)


async def _rafaga():
    await asyncio.gather(*[main.hybrid_search_single_silo(
        c, "renta", [0.1] * 1536, models.SparseVector(indices=[], values=[]), None, 5, 0.7)
        for c in sorted(ESTATALES)])
callado(_rafaga())
ok(main.qdrant_client.pico <= 6, f"32 búsquedas juntas: nunca más de 6 en Qdrant (pico {main.qdrant_client.pico})")
ok(main.qdrant_client.pico >= 4, "y sí corren en paralelo hasta el tope")

ok(main.QDRANT_SEM_PRECEDENTES._value == 2, "los precedentes del chat tienen su carril de 2")


class QdrantPorCarril(QdrantFalso):
    """Cuenta por separado lo que va a sentencias_holdings y lo demás."""
    def __init__(self, **k):
        super().__init__(**k)
        self.vuelo = {"prec": 0, "ley": 0}
        self.picos = {"prec": 0, "ley": 0}

    async def query_points(self, collection_name, **kw):
        c = "prec" if collection_name.startswith("sentencias_") else "ley"
        self.vuelo[c] += 1
        self.picos[c] = max(self.picos[c], self.vuelo[c])
        try:
            if c == "prec":
                await asyncio.sleep(0.3)   # las de precedentes, lentas como en frío
            return await super().query_points(collection_name, **kw)
        finally:
            self.vuelo[c] -= 1


main._INFO_COLECCION.clear()
main.qdrant_client = QdrantPorCarril(espera=0.02)


async def _chat_con_precedentes():
    async def precedentes():          # como _precedentes_con_hilo
        main._CARRIL_QDRANT.set(main.QDRANT_SEM_PRECEDENTES)
        return await main.search_precedentes_unified("renta", limit_scjn=12, limit_tcc=24)
    t = asyncio.create_task(precedentes())
    await asyncio.sleep(0.01)         # que los precedentes salgan primero, como en producción
    t0 = time.perf_counter()
    await asyncio.gather(*[main.hybrid_search_single_silo(
        c, "renta", [0.1] * 1536, models.SparseVector(indices=[], values=[]), None, 5, 0.7)
        for c in ("leyes_federales", "leyes_queretaro", "bloque_constitucional")])
    ley = time.perf_counter() - t0
    await t
    return ley
ley, _ = callado(_chat_con_precedentes())
ok(main.qdrant_client.picos["prec"] <= 2, f"los ocho de precedentes, de dos en dos ({main.qdrant_client.picos['prec']})")
ok(ley < 0.2, f"y la ley no espera detrás de ellos ({ley:.2f} s; con un solo semáforo, ≥ 0.3 s)")
ok(main._CARRIL_QDRANT.get() is None, "el carril no se sale de la tarea de precedentes")


print("\n4 · LA HÍBRIDA YA NO TRAE VECTORES: EL COSENO SE PIDE POR ID")
pts = [models.ScoredPoint(id=i, version=1, score=s, payload={"texto": f"Art. {i}", "ref": f"Art. {i}"})
       for i, s in ((11, 0.033), (12, 0.031), (13, 0.016))]
cos = {11: 0.61, 12: 0.74, 13: 0.52}
main._INFO_COLECCION.clear()
main.qdrant_client = QdrantFalso(puntos={"leyes_queretaro": pts}, cosenos=cos)
r, _ = callado(main.hybrid_search_single_silo(
    "leyes_queretaro", "plazo para contestar", [0.1] * 1536,
    models.SparseVector(indices=[3, 9], values=[1.0, 0.5]), None, 130, 0.7))
qp = [kw for m, c, kw in main.qdrant_client.llamadas if m == "query_points"]
ok(len(qp) == 2, f"dos llamadas: la fusión y la de cosenos ({len(qp)})")
ok(all(kw.get("with_vectors") in (False, None) for kw in qp), "ninguna pide vectores (antes with_vectors=['dense'])")
ok(qp[1].get("with_payload") is False and qp[1]["limit"] == 3, "la de cosenos va sin payload y con límite = ids")
ok([x.id for x in r] == ["11", "12", "13"], "se conserva el ORDEN de la fusión")
ok([round(x.score, 2) for x in r] == [0.74, 0.61, 0.52], "con la escala del coseno, repartida en ese orden")
ok(not re.search(r'^\s+with_vectors=\["dense"\]', FUENTE, re.M),
   "no queda ninguna búsqueda que traiga los vectores densos")


print("\n5 · EL PREFETCH DE LAS LEYES, CON TOPE")
ok(main._prefetch_hibrido("leyes_federales", 65) == 100, "federal top_k 65: 260 → 100")
ok(main._prefetch_hibrido("leyes_queretaro", 130) == 130, "estatal top_k 130: 520 → 130 (nunca menos que top_k)")
ok(main._prefetch_hibrido("leyes_cdmx", 5) == 40, "top_k chico: el piso de 40, como antes")
ok(main._prefetch_hibrido("bloque_constitucional", 65) == 260, "el bloque constitucional no cambia (ahí sí perdía)")
ok(qp[0]["prefetch"][0].limit == 130 and qp[0]["prefetch"][1].limit == 130,
   "la llamada real a la colección del estado lleva 130 por lado")


print("\n6 · LOS SIETE CIRCUITOS EN UNA SOLA CONSULTA CUANDO NO HAY ESTADO")


def holding(i, circ, s):
    return models.ScoredPoint(id=i, version=1, score=s, payload={
        "tribunal": "1TCC", "expediente": f"AD-{i}/2024", "fecha_sentencia": "2024-01-01",
        "sentido": "concede", "materia": "civil", "holding": f"criterio {i}", "circuito": circ})


todos = [holding(100 + k, c, 0.9 - 0.001 * k) for k, c in enumerate(
    ["1"] * 30 + ["2", "3", "4", "6", "16", "22"] * 3)]
main.qdrant_client = QdrantFalso(puntos={"sentencias_holdings": todos, "sentencias_scjn_holdings": []})
main.sparse_encoder = None
r, _ = callado(main.search_precedentes_unified("arrendamiento", limit_scjn=12, limit_tcc=24, una_busqueda_tcc=True))
hold = [kw for m, c, kw in main.qdrant_client.llamadas if m == "query_points" and c == "sentencias_holdings"]
ok(len(hold) == 1, f"una sola consulta a sentencias_holdings (antes 7) → {len(hold)}")
cond = hold[0]["query_filter"].must[0]
ok(isinstance(cond.match, models.MatchAny) and set(cond.match.any) == set(main._ACTIVE_CIRCUITS),
   "filtrada a los siete circuitos con MatchAny")
ok(hold[0]["limit"] == 24 * 7, f"24 por circuito (168): el mismo acierto que las siete, medido → {hold[0]['limit']}")
ok(hold[0]["with_payload"] == main._CAMPOS_HOLDING, "y sólo con los campos que se leen")
silos = [x.silo for x in r if "circuito" in x.silo]
ok(len(silos) == 24 and len(set(silos)) == 7,
   f"24 sentencias de los 7 circuitos aunque uno domine la puntuación ({len(set(silos))} circuitos)")
ok(silos.count("sentencias_1_circuito") <= 24 - 3 * 6, "el mínimo por circuito se respeta")
main.qdrant_client = QdrantFalso(puntos={"sentencias_holdings": todos})
callado(main.search_precedentes_unified("arrendamiento", limit_scjn=12, limit_tcc=24))
ok(main.qdrant_client.colecciones().count("sentencias_holdings") == 7,
   "sin la bandera (modo Precedentes, o con estado): una por circuito, como siempre")
ok("una_busqueda_tcc=_PRECEDENTES_UNA_BUSQUEDA and _sin_estado_prec" in FUENTE
   and "_sin_estado_prec = not (request.estado or detect_single_estado_from_query(" in FUENTE,
   "el chat la usa sólo sin estado elegido ni nombrado")


print("\n7 · EL FILTRO DE CONTENIDO DE OPENAI: UN REINTENTO CON OTRO MODELO")


class Trozo:
    def __init__(self, texto):
        self.choices = [types.SimpleNamespace(delta=types.SimpleNamespace(content=texto, reasoning_content=None))]


class Rechazo(Exception):
    pass


FILTRO = ("Invalid prompt: your prompt was flagged as potentially violating our usage policy. "
          "Please try again with a different prompt")


class OpenAIFalso:
    """`chat.completions.create` en flujo; cada modelo se porta como se le diga."""
    def __init__(self, conducta):
        self.conducta = conducta     # modelo → "rechaza_al_abrir" | "rechaza_en_flujo" | "corta_tras_texto" | "otro_error" | "bien"
        self.pedidos = []
        self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self._create))

    async def _create(self, **kw):
        self.pedidos.append(kw)
        c = self.conducta.get(kw["model"], "bien")
        if c == "rechaza_al_abrir":
            raise Rechazo(f"Error code: 400 - {{'error': {{'message': '{FILTRO}', 'code': 'invalid_prompt'}}}}")

        async def flujo():
            if c == "rechaza_en_flujo":
                raise Rechazo(FILTRO)
            if c == "otro_error":
                raise Rechazo("Connection reset by peer")
            yield Trozo(f"respuesta de {kw['model']}")
            if c == "corta_tras_texto":
                raise Rechazo(FILTRO)
            yield Trozo(" · fin")
        return flujo()


async def leer(cli, kw):
    uso = {}
    texto = ""
    try:
        async for t in await main._flujo_con_respaldo_filtro(cli, kw, uso):
            texto += t.choices[0].delta.content or ""
        return texto, uso, None
    except Exception as e:
        return texto, uso, e

KW = {"model": "gpt-6-luna", "messages": [{"role": "user", "content": "x"}], "stream": True,
      "max_completion_tokens": 30000, "reasoning_effort": "low"}
for conducta in ("rechaza_al_abrir", "rechaza_en_flujo"):
    cli = OpenAIFalso({"gpt-6-luna": conducta})
    (texto, uso, e), log = callado(leer(cli, dict(KW)))
    ok(texto == "respuesta de gpt-5.6-terra · fin" and e is None,
       f"{conducta}: contesta gpt-5.6-terra con el mismo contexto")
    ok("🛡️ filtro de OpenAI → respaldo con gpt-5.6-terra" in log, "y queda en el registro con el emoji")
    ok(uso.get("modelo") == "gpt-5.6-terra" and cli.pedidos[1]["messages"] == KW["messages"]
       and cli.pedidos[1]["max_completion_tokens"] == 30000, "mismos mensajes y mismo tope; se avisa qué modelo fue")
cli = OpenAIFalso({"gpt-6-luna": "corta_tras_texto"})
(texto, uso, e), _ = callado(leer(cli, dict(KW)))
ok(texto == "respuesta de gpt-6-luna" and e is not None and len(cli.pedidos) == 1,
   "si ya había texto no se reintenta: lo entregado se queda y manda el aviso de siempre")
cli = OpenAIFalso({"gpt-6-luna": "otro_error"})
(texto, uso, e), _ = callado(leer(cli, dict(KW)))
ok(e is not None and len(cli.pedidos) == 1, "otro error (red, 500…) no dispara el respaldo")
cli = OpenAIFalso({"gpt-6-luna": "rechaza_al_abrir", "gpt-5.6-terra": "rechaza_al_abrir"})
(texto, uso, e), _ = callado(leer(cli, dict(KW)))
ok(e is not None and len(cli.pedidos) == 2, "UN solo reintento: si el respaldo también se niega, falla como antes")
cli = OpenAIFalso({"gpt-5.6-terra": "rechaza_al_abrir"})
(texto, uso, e), _ = callado(leer(cli, dict(KW, model="gpt-5.6-terra")))
ok(e is not None and len(cli.pedidos) == 1, "si el que se niega ya es el de respaldo, no hay a dónde ir")
k2 = main._kwargs_respaldo_filtro(dict(KW, reasoning_effort="minimal", extra_body={"thinking": {}}))
ok(k2["reasoning_effort"] == "low" and "extra_body" not in k2,
   "«minimal» (que la 5.6 rechaza) pasa a «low» y los parámetros de DeepSeek se quitan")
ok(main._kwargs_respaldo_filtro(dict(KW, model="deepseek-v4-flash")) is None, "DeepSeek no pasa por aquí")
ok("stream = await _flujo_con_respaldo_filtro(active_client, api_kwargs, _uso_flujo)" in FUENTE,
   "el chat abre su flujo por aquí")


# ─────────────────────────────────────────────────────────────────────────────
print("\n8 · QDRANT REAL, SÓLO LECTURA: ANTES Y DESPUÉS DE UNA LLAMADA HÍBRIDA")
try:
    import httpx
    from openai import AsyncOpenAI
    from fastembed import SparseTextEmbedding
    from qdrant_client import AsyncQdrantClient
    e = {}
    for l in Path("/Users/josedavidalcantarmendoza/Documents/IUREXIA-MAC/jurexia-api-git/.env").read_text().splitlines():
        if "=" in l and not l.startswith("#"):
            k, v = l.split("=", 1)
            e[k.strip()] = v.strip().strip('"')
    URL, KEY = e["QDRANT_URL"].rstrip("/"), e["QDRANT_API_KEY"]
    vivo = True
except Exception as ex:
    vivo = False
    print(f"   (sin Qdrant: {type(ex).__name__}; se salta la parte real)")

if vivo:
    oa = AsyncOpenAI(api_key=e["OPENAI_API_KEY"])
    bm25 = SparseTextEmbedding(model_name="Qdrant/bm25", language="spanish")
    PREG = "¿Qué plazo tengo para contestar la demanda en la vía ordinaria civil?"
    d = correr(oa.embeddings.create(model=main.EMBEDDING_MODEL, input=PREG)).data[0].embedding
    s = list(bm25.query_embed(PREG))[0]
    s = {"indices": s.indices.tolist(), "values": s.values.tolist()}
    flt = {"must_not": [{"key": "es_transitorio", "match": {"value": True}}]}

    async def comparar():
        async with httpx.AsyncClient(timeout=60, headers={"api-key": KEY}) as c:
            async def q(col, cuerpo):
                t0 = time.perf_counter()
                r = await c.post(f"{URL}/collections/{col}/points/query", json=cuerpo)
                return r.json(), time.perf_counter() - t0, len(r.content)

            def fusion(pf, con_vectores):
                return {"prefetch": [{"query": d, "using": "dense", "limit": pf, "filter": flt},
                                     {"query": s, "using": "sparse", "limit": pf, "filter": flt}],
                        "query": {"fusion": "rrf"}, "limit": 130, "with_payload": True,
                        "with_vector": ["dense"] if con_vectores else False}
            # ANTES: prefetch 520 y vectores de vuelta; el coseno, en Python.
            ja, ta, ba = await q("leyes_queretaro", fusion(520, True))
            loc = {p["id"]: sum(x * y for x, y in zip(d, p["vector"]["dense"])) for p in ja["result"]["points"]}
            # AHORA: prefetch 130, sin vectores; el coseno, por id.
            jn, tn, bn = await q("leyes_queretaro", fusion(main._prefetch_hibrido("leyes_queretaro", 130), False))
            ids = list(loc)
            ji, ti, bi = await q("leyes_queretaro", {"query": d, "using": "dense", "filter": {"must": [{"has_id": ids}]},
                                                     "limit": len(ids), "with_payload": False, "with_vector": False})
            qd = {p["id"]: p["score"] for p in ji["result"]["points"]}
            return (ja, ta, ba), (jn, tn + ti, bn + bi), loc, qd

    (ja, ta, ba), (jn, tn, bn), loc, qd = correr(comparar())
    print(f"   antes : {ba / 1e3:7.0f} KB · servidor {ja.get('time', 0) * 1000:6.0f} ms · reloj {ta:5.2f} s")
    print(f"   ahora : {bn / 1e3:7.0f} KB · servidor {(jn.get('time', 0)) * 1000:6.0f} ms · reloj {tn:5.2f} s (fusión + cosenos)")
    ok(bn < ba * 0.25, f"menos de la cuarta parte de bytes ({bn / max(1, ba):.0%})")
    dif = max(abs(loc[i] - qd.get(i, 9)) for i in loc)
    ok(len(qd) == len(loc) and dif < 1e-3, f"el coseno por id es el mismo número en los {len(loc)} puntos (dif. máx. {dif:.5f})")

    print("\n9 · QDRANT REAL: LA CONSULTA SIN ESTADO, CONTADA")
    main.qdrant_client = AsyncQdrantClient(url=URL, api_key=KEY, timeout=30)
    main.openai_client = oa
    main.sparse_encoder = bm25
    cuenta = {"n": 0, "vec": 0, "colecciones": set(), "pico": 0, "vuelo": 0}
    _qp = main.qdrant_client.query_points

    async def qp_contado(*a, **kw):
        cuenta["n"] += 1
        cuenta["vec"] += kw.get("with_vectors") not in (None, False)
        cuenta["colecciones"].add(kw.get("collection_name"))
        cuenta["vuelo"] += 1
        cuenta["pico"] = max(cuenta["pico"], cuenta["vuelo"])
        try:
            return await _qp(*a, **kw)
        finally:
            cuenta["vuelo"] -= 1
    main.qdrant_client.query_points = qp_contado

    async def _emb(texto, modelo=None):
        r = await oa.embeddings.create(model=modelo or main.EMBEDDING_MODEL, input=texto[:2000])
        return r.data[0].embedding
    main.get_dense_embedding = _emb
    main._INFO_COLECCION.clear()
    t0 = time.perf_counter()
    r, log = callado(main.hybrid_search_all_silos(
        query=Q_SIN, estado=None, top_k=65, skip_llm_presearch=True, precomputed_plan=PLAN,
        materia_por_palabras=False))
    t = time.perf_counter() - t0
    fase = re.search(r"Búsqueda en (\d+) silos: ([\d.]+)s", log)
    print(f"   {fase.group(0) if fase else '?'} · {cuenta['n']} llamadas · pico {cuenta['pico']} · total {t:.1f} s")
    ok(fase and int(fase.group(1)) == 5, "5 silos (antes 41)")
    ok(not (cuenta["colecciones"] & ESTATALES), "ni una colección estatal")
    ok(cuenta["vec"] == 0, "ninguna llamada trae vectores")
    ok(cuenta["pico"] <= 6, f"nunca más de 6 a la vez ({cuenta['pico']})")
    ok("❌ Error en búsqueda" not in log, "ninguna búsqueda cae")
    ok(len(r) > 30 and all(not x.silo.startswith("leyes_") or x.silo == "leyes_federales" for x in r),
       f"{len(r)} resultados, ninguno de una ley estatal ajena")


print("\n" + ("TODO PASA" if not FALLOS else f"{len(FALLOS)} FALLAS:\n   - " + "\n   - ".join(FALLOS)))
sys.exit(1 if FALLOS else 0)
