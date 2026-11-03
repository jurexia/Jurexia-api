# -*- coding: utf-8 -*-
"""LA COLA DE LA BÚSQUEDA, EN PARALELO — 2-nov-2026.

    .venv/bin/python test_velocidad_busqueda.py

Sin red, sin modelos, sin embeddings de verdad y sin el arranque del servidor
(el lifespan borra las cachés de Gemini de producción). Qdrant, OpenAI y Azure
son falsos y deterministas: la misma llamada con lo mismo devuelve lo mismo,
llegue en el orden que llegue.

Lo que se comprueba:

  1 · EQUIVALENCIA. hybrid_search_all_silos de ahora contra la de antes del
      cambio (`git show 312cecd:main.py`, o MAIN_ANTERIOR=<ruta>), con el mismo
      Qdrant falso: la lista final —ids, orden, puntajes y todo lo demás de cada
      SearchResult— es idéntica. Los corpus se pisan a propósito (el mismo id
      en varias colecciones y etapas) para que la deduplicación por
      `existing_ids` tenga trabajo, y se exige que cada etapa de la cola haya
      corrido y aportado en las dos versiones: una equivalencia en vacío no
      prueba nada.
  2 · TIEMPO. Con esperas artificiales en cada etapa, la cola pasa de la suma
      a más o menos la etapa más larga; y nunca hay más de QDRANT_SEM llamadas
      a la vez.
  3 · FALLOS. Una etapa de la cola que falla se traga el error y sigue, igual
      que antes (mismo resultado, mismo aviso en el registro); lo que antes
      reventaba revienta igual, y no deja tareas colgando. Cancelada la
      búsqueda (el plazo de 25 s del acervo de /analyze-document), sus tareas
      adelantadas se cancelan.
  4 · AZURE. El sondeo pregunta al segundo y luego cada segundo, con el mismo
      tope de 120 s de espera.
  5 · LO DEMÁS. Los dos vectores de la pasada por concepto a la vez, la
      corrección del abogado en un hilo y el plan para la memoria desde la
      entrada de /chat.

La comparación de 1 y 3 es contra el código de ANTES de este cambio. Si más
adelante otro cambio altera A PROPÓSITO lo que devuelve la búsqueda, fallará
con razón: entonces se corre con VELOCIDAD_BASE=<el commit anterior a ese
cambio> (o MAIN_ANTERIOR=<ruta a un main.py>), o se retira esa parte.
Validada con mutaciones el 2-nov-2026: fundir las NOM contra el `existing_ids`
de cuando se lanzaron, quitar la deduplicación del código local, o fundir las
NOM antes que el código local, la hacen fallar.
"""
import asyncio
import contextlib
import copy
import hashlib
import io
import os
import re
import subprocess
import sys
import threading
import time
import types
from pathlib import Path

import dotenv

# Ni una credencial de producción. main.py busca el .env SUBIENDO directorios,
# y desde un worktree encuentra el del checkout principal: no se lee ninguno, y
# las llaves van falsas a la fuerza, como en test_documento_respaldo.py.
dotenv.load_dotenv = lambda *a, **k: False
for _k in ("OPENAI_API_KEY", "OPENROUTER_API_KEY", "DEEPSEEK_API_KEY", "GEMINI_API_KEY",
           "MISTRAL_API_KEY", "ELEVENLABS_API_KEY", "AZURE_DOCINT_KEY", "AZURE_SPEECH_KEY"):
    os.environ[_k] = "falsa-para-pruebas"
os.environ.update(QDRANT_URL="http://127.0.0.1:9", QDRANT_API_KEY="",
                  SUPABASE_URL="", SUPABASE_SERVICE_KEY="", AZURE_DOCINT_ENDPOINT="")

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
# El código de antes del cambio: origin/main el 2-nov-2026.
BASE = os.getenv("VELOCIDAD_BASE", "312cecd")

FALLOS = []


def ok(cond, que):
    print(f"   {'PASA ' if cond else 'FALLA'}  {que}")
    if not cond:
        FALLOS.append(que)


with contextlib.redirect_stdout(io.StringIO()):
    import main  # noqa: E402
assert main.supabase_admin is None, "main.py cargó un Supabase de verdad: la prueba no sigue"
import procesal_federal as _pf  # noqa: E402
import regimen_procesal as _rp  # noqa: E402


def _cargar_antes():
    """El main.py de antes, como módulo aparte y con el mismo __file__ que el de
    ahora: así lee los mismos datos/ (los plazos, el régimen) y la única
    diferencia es el código."""
    ruta = os.getenv("MAIN_ANTERIOR")
    try:
        if ruta:
            fuente = Path(ruta).read_bytes()
        else:
            fuente = subprocess.run(["git", "-C", str(AQUI), "show", f"{BASE}:main.py"],
                                    capture_output=True, check=True, timeout=60).stdout
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"
    mod = types.ModuleType("main_antes")
    mod.__file__ = str(AQUI / "main.py")
    sys.modules["main_antes"] = mod
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(fuente, f"{AQUI / 'main.py'} @ {BASE}", "exec"), mod.__dict__)
    assert mod.supabase_admin is None
    return mod, None


antes, _por_que = _cargar_antes()

_BUCLE = asyncio.new_event_loop()


def correr(co):
    # Un solo bucle: los semáforos de main.py quedan atados al primero que los usa.
    return _BUCLE.run_until_complete(co)


def callado(co):
    with contextlib.redirect_stdout(io.StringIO()) as f:
        try:
            r = correr(co)
        except BaseException as e:   # noqa: BLE001 — se compara también cómo revienta
            r = e
    return r, f.getvalue()


# ─────────────────────────────────────────────────────────────────────────────
# LOS CORPUS FALSOS
# ─────────────────────────────────────────────────────────────────────────────
CPC_QRO = "Código de Procedimientos Civiles del Estado de Querétaro"
CC_QRO = "Código Civil del Estado de Querétaro"
TRANSITO_QRO = "Ley de Tránsito del Estado de Querétaro"
LFPA = _pf.LEYES["LFPA"][0]
CNPCF = _pf.LEYES["CNPCF"][0]
CNPP = _pf.LEYES["CNPP"][0]
LGMASC = _pf.LEYES["LGMASC"][0]


def P(pid, **payload):
    return (pid, payload)


def corpus_base() -> dict:
    """Ids repartidos para que se pisen: q03/q04 están en la colección de
    Querétaro y en la procesal federal; p02 en la procesal y en las NOM; d15 en
    la jurisprudencia y entre los artículos de la CADH del bloque."""
    c = {}
    c["leyes_federales"] = (
        [P(f"d{i:02d}", texto=f"Artículo {i}. {LFPA}: plazo de quince días para interponer el recurso de revisión {i}.",
           ref=f"Art. {i}", origen=LFPA, ley=LFPA, materia="administrativo", entidad="FEDERAL") for i in range(1, 7)]
        + [P(f"d{i:02d}", texto=f"Artículo {200 + i}. {CNPCF}: la apelación se interpone dentro de nueve días.",
             ref=f"Art. {200 + i}", origen=CNPCF, ley=CNPCF, materia="procesal_civil") for i in (7, 8)]
        # Muchos del CNPP: la búsqueda sin filtro sólo ve algunos y el código
        # nacional (filtrado por materia) encuentra los demás.
        + [P(f"n{i:02d}", texto=f"Artículo {300 + i}. {CNPP}: vinculación a proceso del imputado {i}.",
             ref=f"Art. {300 + i}", origen=CNPP, ley=CNPP, cuerpo_legal_oficial=CNPP,
             materia="procesal_penal") for i in range(1, 17)]
    )
    c["jurisprudencia_nacional_v3"] = [
        P(f"d{i:02d}", texto=f"APELACIÓN. EL PLAZO PARA INTERPONERLA {i}. Criterio sobre garantías judiciales.",
          rubro=f"APELACIÓN. EL PLAZO PARA INTERPONERLA ({i}).", registro=str(2030000 + i),
          tipo="Jurisprudencia", instancia="Tribunales Colegiados de Circuito", materia="Civil")
        for i in range(9, 19)]
    c["bloque_constitucional"] = (
        [P(f"b{i:02d}", texto=f"Artículo 8. Garantías Judiciales, párrafo {i + 1}: toda persona tiene derecho a ser oída.",
           ref=f"Art. 8.{i + 1} CADH", origen="Convención Americana sobre Derechos Humanos", tipo="convencion",
           instrumento="CADH", articulo_num=8, chunk_index=i) for i in range(4)]
        + [P("d15", texto="Artículo 8. Garantías Judiciales (texto consolidado).", ref="Art. 8 CADH",
             origen="Convención Americana sobre Derechos Humanos", tipo="convencion", instrumento="CADH",
             articulo_num=8, chunk_index=4)]
        + [P(f"b{i:02d}", texto=f"Artículo {i}o. Nadie podrá ser privado de sus derechos sin juicio {i}.",
             ref=f"Art. {i}o CPEUM", origen="Constitución Política de los Estados Unidos Mexicanos",
             tipo="constitucion", sustantivo=True) for i in (14, 16, 17)]
        + [P(f"b{i:02d}", texto=f"Cuadernillo {i}: debido proceso en la Corte Interamericana.",
             ref=f"Cuadernillo {i}", origen="Cuadernillo de Jurisprudencia de la Corte Interamericana No. 12",
             tipo="cuadernillo") for i in (20, 21, 22)]
    )
    c["leyes_queretaro"] = (
        [P(f"q{i:02d}", texto=f"Artículo {i}. El término para apelar la sentencia definitiva será de nueve días ({i}).",
           ref=f"Artículo {i}", origen=CPC_QRO, entidad="QUERETARO") for i in range(1, 11)]
        + [P(f"q{i:02d}", texto=f"Artículo {i}. Las obligaciones se extinguen por el pago ({i}).",
             ref=f"Artículo {i}", origen=CC_QRO, entidad="QUERETARO") for i in range(11, 15)]
        + [P(f"q{i:02d}", texto=f"Artículo {i}. Las infracciones de tránsito se sancionan ({i}).",
             ref=f"Artículo {i}", origen=TRANSITO_QRO, entidad="QUERETARO") for i in range(15, 18)]
    )
    c["sentencias_holdings"] = [
        P(f"h{i:02d}", texto=f"Holding {i}: la apelación extemporánea se desecha.", holding=f"criterio {i}",
          tribunal="1TCC", circuito="22", sentido="niega", materia="civil") for i in range(1, 7)]
    c["sentencias_scjn_holdings"] = [
        P(f"s{i:02d}", texto=f"Holding SCJN {i}: plazo razonable.", holding=f"criterio SCJN {i}",
          sentido="concede", materia="civil") for i in range(1, 5)]

    def _proc(pid, abrev, ley, n, figuras, materia):
        return P(pid, abreviatura=abrev, ley=ley, ref=f"Artículo {n}", articulo_num=n, figuras=figuras,
                 materia_procesal=materia, texto_raw=f"Artículo {n}. Plazo de nueve días para apelar ({pid}).",
                 texto=f"Artículo {n}. Plazo de nueve días para apelar ({pid}).", remisiones=[], chunk_index=0)
    c["procesal_federal"] = (
        [_proc(f"q{i:02d}", "LGMASC", LGMASC, 40 + i, ["plazo", "recurso"], "civil_familiar") for i in (3, 4)]
        + [_proc(f"p{i:02d}", "LGMASC", LGMASC, 50 + i, ["plazo"], "civil_familiar") for i in range(1, 6)]
        + [_proc(f"p{i:02d}", "CNPCF", CNPCF, 60 + i, ["plazo", "recurso"], "civil_familiar") for i in range(6, 9)]
        + [_proc(f"p{i:02d}", "CNPP", CNPP, 70 + i, ["plazo", "recurso", "vinculacion"], "penal") for i in range(9, 13)]
    )
    c["normas_oficiales"] = (
        [P(f"o{i:02d}", clave="NOM-001-SCT-2020", titulo="Señalamiento vial en carreteras",
           texto_raw=f"Numeral {i}. Las señales de plazo y tránsito.", texto=f"Numeral {i} de la NOM-001.",
           ref=f"NOM-001-SCT-2020, numeral {i}", origen="NOM-001-SCT-2020") for i in range(1, 5)]
        + [P(f"o{i:02d}", clave="NOM-034-SCT2-2022", titulo="Reductores de velocidad",
             texto_raw=f"Numeral {i}. Reductores y apelación vial.", texto=f"Numeral {i} de la NOM-034.",
             ref=f"NOM-034-SCT2-2022, numeral {i}", origen="NOM-034-SCT2-2022") for i in range(5, 9)]
        + [P("p02", clave="NOM-034-SCT2-2022", titulo="Reductores de velocidad",
             texto_raw="Numeral 9. Repetido con la procesal.", texto="Numeral 9 de la NOM-034.",
             ref="NOM-034-SCT2-2022, numeral 9", origen="NOM-034-SCT2-2022")]
        # Repetidos con el código procesal de Querétaro: quién entra primero —el
        # código local o la NOM— decide qué versión del id se queda.
        + [P(pid, clave=clave, titulo="Repetido con el código local", texto_raw=f"Numeral {pid}. Plazo vial.",
             texto=f"Numeral {pid}.", ref=f"{clave}, numeral {pid}", origen=clave)
           for pid, clave in (("q05", "NOM-001-SCT-2020"), ("q06", "NOM-001-SCT-2020"),
                              ("q07", "NOM-034-SCT2-2022"), ("q08", "NOM-034-SCT2-2022"))]
    )
    return c


# Los escenarios que cambian un corpus lo hacen sobre éste (corpus_base se
# reasigna mientras duran).
CORPUS_ORIGINAL = corpus_base


# ─────────────────────────────────────────────────────────────────────────────
# UN QDRANT FALSO, DETERMINISTA
# ─────────────────────────────────────────────────────────────────────────────
def _h01(*partes) -> float:
    h = hashlib.md5("|".join(map(str, partes)).encode("utf-8")).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF


def _condiciones(flt):
    return ([c for c in (getattr(flt, "must", None) or [])],
            [c for c in (getattr(flt, "should", None) or [])],
            [c for c in (getattr(flt, "must_not", None) or [])])


def _cumple(cond, pid, pl) -> bool:
    if hasattr(cond, "has_id"):
        return pid in {str(i) for i in cond.has_id}
    key, match = getattr(cond, "key", None), getattr(cond, "match", None)
    if key is None or match is None:
        return True
    v = pl.get(key)
    if v is None:
        return False
    valores = set(v) if isinstance(v, list) else {v}
    if hasattr(match, "any"):
        return bool(valores & set(match.any))
    if hasattr(match, "value"):
        return match.value in valores
    return True


class Punto:
    def __init__(self, pid, score, payload):
        self.id, self.score, self.payload, self.version, self.vector = pid, score, payload, 1, None


class QdrantFalso:
    """Puntúa cada punto con un hash de (colección, consulta, filtro, id): lo
    mismo para la misma llamada, en cualquier orden. Sin filtro, la búsqueda
    sólo «ve» los `tope_sin_filtro` mejores —lo que la semántica suele traer—;
    con filtro ve todo lo que cumple, como las búsquedas por nombre exacto."""

    def __init__(self, corpus, demora=None, falla=None, tope_sin_filtro=8, extra=None):
        self.corpus = corpus
        self.demora = demora or (lambda m, c, kw: 0.0)
        self.falla = falla or (lambda m, c, kw: None)
        self.tope = tope_sin_filtro
        self.extra = extra or {}
        self.llamadas = []
        self.en_vuelo = 0
        self.pico = 0
        self.canceladas = 0

    async def _entrar(self, metodo, coleccion, kw):
        self.llamadas.append((metodo, coleccion))
        self.en_vuelo += 1
        self.pico = max(self.pico, self.en_vuelo)
        try:
            d = self.demora(metodo, coleccion, kw)
            if d:
                await asyncio.sleep(d)
            e = self.falla(metodo, coleccion, kw)
            if e is not None:
                raise e
        except asyncio.CancelledError:
            self.canceladas += 1
            raise
        finally:
            self.en_vuelo -= 1

    def _puntos(self, coleccion, firma, flt, limite, umbral=None):
        must, should, must_not = _condiciones(flt)
        cand = []
        for pid, pl in self.corpus.get(coleccion, []):
            if not all(_cumple(c, pid, pl) for c in must):
                continue
            if should and not any(_cumple(c, pid, pl) for c in should):
                continue
            if any(_cumple(c, pid, pl) for c in must_not):
                continue
            s = round(0.30 + 0.65 * _h01(coleccion, firma, pid) + self.extra.get(coleccion, 0.0), 6)
            cand.append(Punto(pid, s, copy.deepcopy(pl)))
        cand.sort(key=lambda p: (-p.score, p.id))
        if not (must or should) and self.tope:
            cand = cand[:self.tope]
        if umbral is not None:
            cand = [p for p in cand if p.score >= umbral]
        return cand[:limite]

    async def query_points(self, collection_name, query=None, using=None, query_filter=None, limit=10,
                           with_payload=True, score_threshold=None, prefetch=None, **kw):
        await self._entrar("query_points", collection_name,
                           dict(query_filter=query_filter, using=using, limit=limit, score_threshold=score_threshold))
        firma = (f"{query[0]:.9f}" if isinstance(query, list) and query else "q") + f"|{query_filter!r}|{using}"
        return types.SimpleNamespace(points=self._puntos(collection_name, firma, query_filter, limit,
                                                         score_threshold))

    async def scroll(self, collection_name, scroll_filter=None, limit=10, with_payload=True, **kw):
        await self._entrar("scroll", collection_name, dict(scroll_filter=scroll_filter, limit=limit))
        return (self._puntos(collection_name, f"scroll|{scroll_filter!r}", scroll_filter, limit), None)

    async def facet(self, collection_name, key=None, limit=10, **kw):
        await self._entrar("facet", collection_name, dict(key=key))
        valores = sorted({pl.get(key) for _, pl in self.corpus.get(collection_name, []) if pl.get(key)})
        return types.SimpleNamespace(hits=[types.SimpleNamespace(value=v, count=1) for v in valores])

    async def get_collection(self, collection_name, **kw):
        await self._entrar("get_collection", collection_name, {})
        return types.SimpleNamespace(
            config=types.SimpleNamespace(params=types.SimpleNamespace(sparse_vectors=None)),
            payload_schema={"materia": 1, "origen": 1, "ley": 1})

    async def retrieve(self, collection_name, **kw):
        await self._entrar("retrieve", collection_name, kw)
        return []


class ChatFalso:
    """`chat.completions.create`: lo que pregunta cada etapa, contestado igual siempre."""

    def __init__(self, demora=None, falla=None):
        self.demora = demora or (lambda tipo: 0.0)
        self.falla = falla or (lambda tipo: None)
        self.pedidos = []
        self.canceladas = 0
        self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self._create))

    async def _create(self, **kw):
        sistema = kw["messages"][0]["content"]
        tipo = ("nom" if sistema.startswith("Eliges normas") else
                "leyes" if "ordenamiento jurídico del Estado de" in sistema else "otro")
        self.pedidos.append(tipo)
        try:
            d = self.demora(tipo)
            if d:
                await asyncio.sleep(d)
        except asyncio.CancelledError:
            self.canceladas += 1
            raise
        e = self.falla(tipo)
        if e is not None:
            raise e
        if tipo == "nom":
            contenido = '{"elegidas": [2, 1]}'
        elif tipo == "leyes":
            contenido = "Reglamento Vial Queretano\nCódigo Civil del Estado de Querétaro"
        else:
            contenido = ""
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=contenido))])


def vector_de(texto, modelo=None):
    v0 = int(hashlib.md5(f"{modelo or ''}|{texto}".encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
    return [round(v0, 9)] + [0.01] * 7


PLAN = {"materia_principal": "civil", "fuero_detectado": "mixto", "jurisprudencia_keywords": [],
        "conceptos_clave": [], "via_procesal": "ordinaria civil", "requiere_ddhh": False,
        "pesos_silos": {"constitucional": 0.25, "federal": 0.25, "estatal": 0.25, "jurisprudencia": 0.25}}
SUBCONSULTAS = ["plazo para apelar la sentencia definitiva civil en Querétaro",
                "garantías judiciales del artículo 8 de la Convención Americana"]


def instalar(mod, qdrant, chat, demora_emb=None, falla_emb=None):
    """Los mismos falsos, con las cachés vacías, en el módulo de antes o en el de
    ahora. Lo que no cambió (Estratega, HyDE, descomposición, reordenamiento por
    modelo, refuerzo de jurisprudencia, enriquecimiento cruzado y vecinos) se
    sustituye por respuestas fijas; lo que cambió corre de verdad."""
    async def _emb(texto, modelo=None):
        d = demora_emb(texto) if demora_emb else 0.0
        if d:
            await asyncio.sleep(d)
        e = falla_emb(texto) if falla_emb else None
        if e is not None:
            raise e
        return vector_de(texto, modelo)

    async def _estratega(query, fuero_manual=None):
        return dict(PLAN)

    async def _hyde(query, estado=None):
        return None

    async def _descomponer(query):
        return list(SUBCONSULTAS)

    async def _reordenar(query, estatales, federales, n_est=20, n_fed=14):
        return estatales, federales

    async def _conceptos(query):
        return "plazo apelación sentencia"

    async def _refuerzo(query, exclude_ids):
        return []

    async def _cruzado(initial_results, query):
        return [mod.SearchResult(id="d03", score=0.61, texto="cruzado d03", silo="leyes_federales",
                                 origen=LFPA, ref="Art. 3"),
                mod.SearchResult(id="x01", score=0.58, texto="cruzado x01", silo="jurisprudencia_nacional_v3",
                                 ref="Tesis x01")]

    async def _vecinos(results, max_neighbors=6, estado=None):
        return []

    mod.qdrant_client = qdrant
    mod.chat_client = chat
    mod.sparse_encoder = None
    mod.get_dense_embedding = _emb
    mod._legal_strategy_agent = _estratega
    mod._generate_hyde_document = _hyde
    mod._decompose_query = _descomponer
    mod._reordenar_leyes_con_modelo = _reordenar
    mod._extract_juris_concepts = _conceptos
    mod._jurisprudencia_boost_search = _refuerzo
    mod._cross_silo_enrichment = _cruzado
    mod._fetch_neighbor_chunks = _vecinos
    mod._INFO_COLECCION.clear()
    mod._INDICES_POR_COLECCION.clear()
    mod._NOMBRES_POR_COLECCION.clear()
    mod._NOMBRES_FEDERALES.update(lista=[], ts=0.0)
    mod._CACHE_EMB_JURIS.clear()
    mod._CACHE_EMB_DENSO.clear()
    mod.fuentes_sel.fijar(None)


def foto(res):
    """Lo que sale de la búsqueda, campo por campo, en su orden."""
    if isinstance(res, BaseException):
        return ("excepción", type(res).__name__)
    return [r.model_dump() for r in res]


def resumen(res):
    if isinstance(res, BaseException):
        return f"{type(res).__name__}: {res}"
    return [(r.id, round(r.score, 4), r.silo) for r in res]


Q_CIVIL = ("¿Qué plazo tengo para apelar la sentencia definitiva en un juicio ordinario civil en Querétaro "
           "y cómo se relaciona con el artículo 8 de la CADH sobre garantías judiciales?")
Q_PENAL = ("¿Cuál es el plazo para resolver la vinculación a proceso del imputado y qué recurso procede "
           "contra ese auto?")


def correr_version(mod, kwargs, qdrant_args=None, chat_args=None, emb_args=None):
    qd = QdrantFalso(corpus_base(), **(qdrant_args or {}))
    ch = ChatFalso(**(chat_args or {}))
    instalar(mod, qd, ch, **(emb_args or {}))
    t0 = time.perf_counter()
    res, log = callado(mod.hybrid_search_all_silos(**kwargs))
    return res, log, time.perf_counter() - t0, qd, ch


def tareas_vivas():
    return [t for t in asyncio.all_tasks(_BUCLE) if not t.done()]


# Lo que deja en el registro cada etapa de la cola cuando APORTA algo (un «+0»
# no cuenta: una etapa que no metió nada no prueba la fusión).
MARCAS = {
    "sub-consultas": r"🔀 Query Decomposition: \+[1-9]",
    "procesal federal": r"🧭 PROCESAL FEDERAL .*: \+[1-9]\d* artículos",
    "procesal local": r"🏛️ PROCESAL LOCAL \(procesal_civil_local\) leyes_queretaro: \+[1-9]",
    "NOM": r"📏 NOM: \+[1-9]",
    "tratado": r"🌐 TRATADO: \+[1-9]",
    "código nacional": r"⚖️ CÓDIGO NACIONAL \(\w+\): \+[1-9]",
    "código nacional, aunque sea +0": r"⚖️ CÓDIGO NACIONAL \(\w+\): \+\d+",
    "law routing": r"📖 Law-Level Routing: 1 leyes FALTANTES",
}


def comparar(nombre, kwargs, etapas=(), **args):
    """Antes y ahora con lo mismo: ¿la misma lista? ¿corrieron las etapas?"""
    r_a, log_a, t_a, *_ = correr_version(antes, kwargs, **args)
    r_n, log_n, t_n, qd_n, ch_n = correr_version(main, kwargs, **args)
    ok(foto(r_a) == foto(r_n),
       f"{nombre}: misma lista antes y ahora ({len(r_n)} documentos)" if isinstance(r_n, list)
       else f"{nombre}: la misma excepción antes y ahora ({type(r_n).__name__})")
    if foto(r_a) != foto(r_n):
        print("      antes:", resumen(r_a)[:12] if isinstance(r_a, list) else resumen(r_a))
        print("      ahora:", resumen(r_n)[:12] if isinstance(r_n, list) else resumen(r_n))
    for etapa in etapas:
        marca = MARCAS[etapa]
        corrio = re.search(marca, log_a) is not None and re.search(marca, log_n) is not None
        ok(corrio, f"{nombre}: «{etapa}» corrió y aportó en las dos versiones")
        if not corrio:
            clave = marca.split(" ")[1].strip("\\")
            for et, log in (("antes", log_a), ("ahora", log_n)):
                print(f"      {et}:", [l.strip()[:150] for l in log.splitlines() if clave in l][:4])
    return r_a, r_n, log_a, log_n, t_a, t_n, qd_n, ch_n


# ═════════════════════════════════════════════════════════════════════════════
print("\n1 · EQUIVALENCIA: LA MISMA LISTA, CAMPO POR CAMPO")
if antes is None:
    print(f"   (sin el código de antes —{_por_que}—: se saltan las comparaciones; "
          f"MAIN_ANTERIOR=<ruta> para dárselo a mano)")
else:
    ok(not hasattr(antes, "_cola_procesal") and hasattr(main, "_cola_procesal"),
       f"de verdad son dos versiones: la de {BASE} no tiene la cola paralela y la de ahora sí")

    # Como el chat: sin pre-búsqueda por modelo (no hay sub-consultas), estado
    # elegido y top_k 65.
    KW_CHAT = dict(query=Q_CIVIL, estado="QUERETARO", top_k=65, skip_llm_presearch=True,
                   precomputed_plan=dict(PLAN), materia_por_palabras=False)
    comparar("chat (Querétaro, top_k 65)", KW_CHAT,
             etapas=("procesal federal", "procesal local", "NOM", "tratado", "law routing"))

    # Como /analyze-document: con pre-búsqueda (sub-consultas), top_k 30 y la
    # materia por palabras (inyección del código de la materia).
    KW_DOC = dict(query=Q_CIVIL, estado="QUERETARO", top_k=30)
    _r = comparar("documento (Querétaro, top_k 30, con sub-consultas)", KW_DOC,
                  etapas=("sub-consultas", "procesal federal", "procesal local", "NOM", "tratado",
                          "law routing"))
    _log_n = _r[3]
    ok("PROCESAL FEDERAL" in _log_n and re.search(r"PROCESAL FEDERAL .*: \+\d+ artículos", _log_n) is not None,
       "la procesal federal entra por su fusión, contra lo ya reunido")

    # Sin estado y sin procesal federal: el código nacional se ADELANTA en la cadena.
    _pf_activo = (main.PROCESAL_FEDERAL_ACTIVO, antes.PROCESAL_FEDERAL_ACTIVO)
    main.PROCESAL_FEDERAL_ACTIVO = antes.PROCESAL_FEDERAL_ACTIVO = False
    KW_PENAL = dict(query=Q_PENAL, estado=None, top_k=30)
    comparar("penal sin estado ni procesal federal (código nacional adelantado)", KW_PENAL,
             etapas=("sub-consultas", "código nacional"))
    main.PROCESAL_FEDERAL_ACTIVO, antes.PROCESAL_FEDERAL_ACTIVO = _pf_activo

    # El caso que no se puede adelantar: la procesal federal trae artículos, sin
    # régimen (el resolvedor falla), y TODOS estaban ya; sólo al fundir se sabe
    # que hace falta el código nacional, y se pide ahí, como antes.
    def _corpus_repetido():
        c = CORPUS_ORIGINAL()
        # Seis del CNPP y nada más: caben enteros en el cupo federal, así que
        # todo lo que traiga la procesal federal (los mismos ids) ya estará.
        c["leyes_federales"] = [x for x in c["leyes_federales"] if x[0] in {f"n{i:02d}" for i in range(1, 7)}]
        c["procesal_federal"] = [(pid, dict(pl, abreviatura="CNPP", ley=CNPP, materia_procesal="penal",
                                            figuras=["plazo", "recurso", "vinculacion"], remisiones=[],
                                            texto_raw=pl["texto"], chunk_index=0))
                                 for pid, pl in c["leyes_federales"]]
        return c
    _resolver = _rp.resolver
    _rp.resolver = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("régimen caído a propósito"))
    corpus_base = _corpus_repetido
    _r = comparar("código nacional al fundir (procesal federal toda repetida)",
                  dict(query=Q_PENAL, estado=None, top_k=30, skip_llm_presearch=True,
                       precomputed_plan=dict(PLAN), materia_por_palabras=False),
                  etapas=("código nacional, aunque sea +0",), qdrant_args=dict(tope_sin_filtro=0))
    corpus_base = CORPUS_ORIGINAL
    _rp.resolver = _resolver
    ok("RÉGIMEN procesal sin decidir" in _r[2] and "RÉGIMEN procesal sin decidir" in _r[3]
       and re.search(r"PROCESAL FEDERAL .*: \+0 artículos", _r[3]) is not None,
       "y de verdad fue ese camino: sin régimen y la procesal federal sin nada nuevo")

    comparar("constitucional sin estado (la Q3 del chat)",
             dict(query=Q_CIVIL, estado=None, top_k=15, fuero="constitucional", skip_llm_presearch=True,
                  precomputed_plan=dict(PLAN), materia_por_palabras=False),
             etapas=("tratado",))

    # skip_post_search=True: nadie la llama así y HOY revienta (`existing_ids`
    # sin definir, ver test_qdrant_alivio.py). Sigue reventando igual.
    _r = comparar("skip_post_search=True (revienta igual que antes)",
                  dict(query="¿Qué dice la ley sobre los contratos?", estado=None, top_k=10,
                       skip_llm_presearch=True, precomputed_plan=dict(PLAN), skip_post_search=True,
                       materia_por_palabras=False))
    ok(isinstance(_r[1], UnboundLocalError), f"…con el mismo UnboundLocalError ({resumen(_r[1])[:60]})")
    correr(asyncio.sleep(0.05))
    ok(not tareas_vivas(), "y no deja tareas colgando")


# ═════════════════════════════════════════════════════════════════════════════
print("\n2 · TIEMPO: DE LA SUMA A LA ETAPA MÁS LARGA, SIN PASAR DE QDRANT_SEM")
D = 0.30


def _demora_qdrant(metodo, coleccion, kw):
    if coleccion in ("procesal_federal", "normas_oficiales"):
        return D
    if coleccion == "bloque_constitucional" and metodo == "scroll":
        return D                                  # el tratado citado por número
    if coleccion == "leyes_queretaro" and metodo == "query_points" and kw.get("score_threshold") is None:
        return D                                  # el código procesal local (sin umbral)
    return 0.005


def _demora_emb(texto):
    return D if texto in SUBCONSULTAS else 0.002


ARGS_T = dict(qdrant_args=dict(demora=_demora_qdrant),
              chat_args=dict(demora=lambda tipo: D if tipo in ("nom", "leyes") else 0.0),
              emb_args=dict(demora_emb=_demora_emb))
KW_T = dict(query=Q_CIVIL, estado="QUERETARO", top_k=30, materia_por_palabras=False)
# La suma de antes: sub-consultas (D) + procesal federal (D) + local (D) + NOM (D + D
# del modelo) + tratado (D), y el modelo de Law-Level Routing (D) en el post-search.
SUMA = 7 * D
# Lo más largo ahora: la cadena procesal federal → local (2D) o las NOM (2D).
MAXIMO = 2 * D
r_n, log_n, t_n, qd_n, ch_n = correr_version(main, KW_T, **ARGS_T)
if antes is not None:
    r_a, log_a, t_a, *_ = correr_version(antes, KW_T, **ARGS_T)
    ok(foto(r_a) == foto(r_n), "con esperas, la misma lista antes y ahora")
    print(f"      antes {t_a:.2f} s · ahora {t_n:.2f} s · suma de las etapas {SUMA:.2f} s · "
          f"la más larga {MAXIMO:.2f} s")
    ok(t_a >= SUMA * 0.95, f"antes iban en fila: {t_a:.2f} s ≥ la suma ({SUMA:.2f} s)")
ok(t_n <= MAXIMO + 0.35, f"ahora, más o menos la más larga: {t_n:.2f} s ≤ {MAXIMO:.2f} s + margen")
ok(re.search(r"Cola en paralelo .*se esperaron (\d+\.\d+)s", log_n) is not None,
   "el registro dice cuánto se esperó al fundir la cola")
ok(ch_n.pedidos.count("leyes") == 1 and ch_n.pedidos.count("nom") == 1,
   f"las mismas llamadas al modelo, una vez cada una ({ch_n.pedidos})")
ok(qd_n.pico <= main.QDRANT_SEM._value,
   f"nunca más de {main.QDRANT_SEM._value} llamadas a Qdrant a la vez (pico {qd_n.pico})")

# El semáforo, apretado: TODAS las llamadas tardan, la principal llena los seis
# lugares y la cola corre a la vez. Sin turno, la cola los rebasaría.
_r, _log, _t, qd_s, _ = correr_version(main, KW_T, qdrant_args=dict(demora=lambda m, c, kw: 0.05))
ok(qd_s.pico <= main.QDRANT_SEM._value,
   f"con todo lento, la cola pide turno: pico {qd_s.pico} ≤ {main.QDRANT_SEM._value}")
ok(qd_s.pico >= 4, f"…y sí corre en paralelo hasta el tope (pico {qd_s.pico})")
_proxy_sin_turno = main._QdrantPorTurno._turno


async def _sin_turno(self, metodo, *a, **k):
    r = getattr(main.qdrant_client, metodo)(*a, **k)
    return await r if hasattr(r, "__await__") else r
main._QdrantPorTurno._turno = _sin_turno
_r, _log, _t, qd_x, _ = correr_version(main, KW_T, qdrant_args=dict(demora=lambda m, c, kw: 0.05))
main._QdrantPorTurno._turno = _proxy_sin_turno
ok(qd_x.pico > main.QDRANT_SEM._value,
   f"(control: sin el turno, la misma consulta pasa de {main.QDRANT_SEM._value}: pico {qd_x.pico})")


# ═════════════════════════════════════════════════════════════════════════════
print("\n3 · FALLOS: LO QUE SE TRAGABA SE TRAGA IGUAL; LO QUE REVENTABA, REVIENTA IGUAL")


class Caida(Exception):
    pass


FALLAS = [
    ("NOM: la búsqueda", dict(qdrant_args=dict(falla=lambda m, c, kw: Caida("nom") if c == "normas_oficiales" else None)),
     "⚠️ NOM: búsqueda falló"),
    ("NOM: la elección por modelo", dict(chat_args=dict(falla=lambda tipo: Caida("modelo") if tipo == "nom" else None)),
     "⚠️ NOM: la elección por modelo falló"),
    # procesal_federal.buscar se traga los fallos de sus consultas
    # (gather con return_exceptions): sale sin artículos y con régimen.
    ("procesal federal: sus consultas a Qdrant",
     dict(qdrant_args=dict(falla=lambda m, c, kw: Caida("pf") if c == "procesal_federal" else None)), None),
    ("procesal local: la búsqueda",
     dict(qdrant_args=dict(falla=lambda m, c, kw: Caida("local") if (c == "leyes_queretaro" and m == "query_points"
                                                                     and kw.get("score_threshold") is None) else None)),
     "⚠️ PROCESAL LOCAL: búsqueda falló"),
    ("tratado", dict(qdrant_args=dict(falla=lambda m, c, kw: Caida("tratado") if (c == "bloque_constitucional"
                                                                                and m == "scroll") else None)),
     "⚠️ TRATADO: CADH art. 8 no se pudo traer"),
    ("sub-consulta: su vector", dict(emb_args=dict(falla_emb=lambda t: Caida("emb") if t == SUBCONSULTAS[0] else None)),
     "⚠️ Sub-query búsqueda falló"),
    ("Law-Level Routing: el modelo", dict(chat_args=dict(falla=lambda tipo: Caida("leyes") if tipo == "leyes" else None)),
     "⚠️ Law-Level Routing falló"),
]
if antes is None:
    print("   (sin el código de antes: se saltan)")
else:
    for nombre, args, aviso in FALLAS:
        r_a, r_n, log_a, log_n, *_ = comparar(f"falla {nombre}", KW_DOC, **args)
        if aviso:
            ok(isinstance(r_n, list) and aviso in log_a and aviso in log_n,
               f"falla {nombre}: se sigue, con el aviso de siempre («{aviso[:40]}…»)")
        else:
            ok(isinstance(r_a, list) and isinstance(r_n, list), f"falla {nombre}: se sigue, como antes")

    # La búsqueda procesal entera revienta: se sigue sin ella, con su aviso, y en
    # una consulta penal el código nacional entra en su lugar (ahora adelantado
    # en la cadena). Con la procesal federal sana, el régimen penal lo impide.
    _buscar_pf = _pf.buscar

    async def _pf_caido(*a, **k):
        raise Caida("procesal federal caída")
    _pf.buscar = _pf_caido
    r_a, r_n, log_a, log_n, *_ = comparar("falla procesal federal entera (penal)",
                                          dict(query=Q_PENAL, estado=None, top_k=30),
                                          etapas=("código nacional",))
    _pf.buscar = _buscar_pf
    ok("⚠️ PROCESAL FEDERAL falló (se sigue con el CNPP de leyes_federales)" in log_a
       and "⚠️ PROCESAL FEDERAL falló (se sigue con el CNPP de leyes_federales)" in log_n,
       "…con el aviso de siempre")

    # El código local que NO se puede leer (un texto que no es texto revienta
    # su puntuación, fuera de su try): lo atrapa el bucle que funde, como antes.
    def _corpus_local_roto():
        c = CORPUS_ORIGINAL()
        c["leyes_queretaro"] = [(pid, dict(pl, texto=12345) if pl["origen"] == CPC_QRO and pid == "q09" else pl)
                                for pid, pl in c["leyes_queretaro"]]
        return c
    corpus_base = _corpus_local_roto
    r_a, r_n, log_a, log_n, *_ = comparar("falla procesal local: su puntuación", KW_DOC)
    corpus_base = CORPUS_ORIGINAL
    ok("⚠️ RÉGIMEN: no se pudo inyectar el código local (procesal_civil_local)" in log_a
       and "⚠️ RÉGIMEN: no se pudo inyectar el código local (procesal_civil_local)" in log_n,
       "…y se avisa en el mismo sitio")

    # La nota del régimen que revienta: antes se calculaba ANTES de buscar el
    # código local; el aviso es el de la nota, no el de la búsqueda.
    _nota = _rp.nota_para_modelo
    _rp.nota_para_modelo = lambda regs: (_ for _ in ()).throw(Caida("nota"))
    r_a, r_n, log_a, log_n, *_ = comparar("falla la nota del régimen", KW_DOC)
    _rp.nota_para_modelo = _nota
    ok(log_a.count("Caida: nota") == log_n.count("Caida: nota") >= 2,
       "…con los mismos avisos, en el mismo número")

    # Lo que antes reventaba —un trozo de NOM cuyo texto no es texto rompe su
    # puntuación, que no está en un try— revienta igual, y ahora además no deja
    # tareas vivas: la envoltura cancela las adelantadas.
    def _corpus_nom_roto():
        c = CORPUS_ORIGINAL()
        c["normas_oficiales"] = [(pid, dict(pl, texto_raw=7, texto=7)) for pid, pl in c["normas_oficiales"]]
        return c
    corpus_base = _corpus_nom_roto
    r_a, r_n, log_a, log_n, *_ = comparar("revienta la puntuación de las NOM", KW_DOC)
    corpus_base = CORPUS_ORIGINAL
    ok(isinstance(r_a, Exception) and type(r_a) is type(r_n),
       f"…las dos con {type(r_n).__name__}")
    correr(asyncio.sleep(0.05))
    ok(not tareas_vivas(), "…y sin tareas colgando")

# Cancelada a medias (el plazo del acervo de /analyze-document): las tareas
# adelantadas no se quedan gastando Qdrant ni modelo.
qd_c = QdrantFalso(corpus_base(), demora=lambda m, c, kw: 2.0 if c in ("procesal_federal", "normas_oficiales") else 0.005)
ch_c = ChatFalso(demora=lambda tipo: 2.0)
instalar(main, qd_c, ch_c)


async def _con_plazo():
    try:
        await asyncio.wait_for(main.hybrid_search_all_silos(**KW_T), timeout=0.3)
        return "terminó"
    except asyncio.TimeoutError:
        await asyncio.sleep(0.05)
        return "plazo"
_res_c, _ = callado(_con_plazo())
ok(_res_c == "plazo", "la búsqueda con plazo de 0.3 s se corta")
ok(not tareas_vivas(), "al cortarse no queda ninguna tarea viva")
ok(qd_c.canceladas >= 2 and qd_c.en_vuelo == 0,
   f"las búsquedas de la cola en vuelo se cancelaron ({qd_c.canceladas}) y Qdrant quedó libre")
ok(ch_c.canceladas >= 1, f"y la llamada adelantada al modelo, también ({ch_c.canceladas})")


# ═════════════════════════════════════════════════════════════════════════════
print("\n4 · AZURE: AL SEGUNDO Y CADA SEGUNDO, CON EL MISMO TOPE")


class RelojFalso:
    def __init__(self):
        self.t = 0.0
        self.pausas = []


class _Resp:
    def __init__(self, headers=None, cuerpo=None):
        self.headers = headers or {}
        self._cuerpo = cuerpo or {}

    def raise_for_status(self):
        return None

    def json(self):
        return self._cuerpo


def cliente_azure(reloj, listo_en):
    class _Cliente:
        consultas = 0

        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, url, content=None, headers=None):
            return _Resp(headers={"Operation-Location": "https://azure.falso/operacion/1"})

        async def get(self, url, headers=None):
            type(self).consultas += 1
            if listo_en is not None and reloj.t >= listo_en:
                return _Resp(cuerpo={"status": "succeeded", "analyzeResult": {"pages": [
                    {"lines": [{"content": "PRIMERA PÁGINA"}, {"content": "texto leído"}]}]}})
            return _Resp(cuerpo={"status": "running"})
    return _Cliente


def ocr_con_reloj(mod, listo_en):
    import httpx
    reloj = RelojFalso()
    cliente = cliente_azure(reloj, listo_en)
    dormir = asyncio.sleep

    async def _dormir_falso(s, *a, **k):
        reloj.pausas.append(s)
        reloj.t += s
        await dormir(0)
    viejo_cliente, mod.AZURE_DOCINT_ENDPOINT, mod.AZURE_DOCINT_KEY = httpx.AsyncClient, "https://azure.falso", "x"
    httpx.AsyncClient, asyncio.sleep = cliente, _dormir_falso
    try:
        r, _ = callado(mod._ocr_azure(b"%PDF-falso", paginas_esperadas=0))
    finally:
        httpx.AsyncClient, asyncio.sleep = viejo_cliente, dormir
    return r, reloj, cliente.consultas


r, reloj, n = ocr_con_reloj(main, listo_en=0.5)
ok(isinstance(r, str) and "texto leído" in r and reloj.pausas == [1.0] and n == 1,
   f"listo enseguida: la primera consulta va al segundo (pausas {reloj.pausas})")
r, reloj, n = ocr_con_reloj(main, listo_en=4.3)
ok(isinstance(r, str) and reloj.t == 5.0 and set(reloj.pausas) == {1.0} and n == 5,
   f"listo a los 4.3 s: se entera a los {reloj.t:.0f} s, preguntando cada segundo ({n} consultas)")
if antes is not None:
    r_a, reloj_a, n_a = ocr_con_reloj(antes, listo_en=4.3)
    ok(reloj_a.t == 6.0 and set(reloj_a.pausas) == {2}, f"(antes se enteraba a los {reloj_a.t:.0f} s, cada 2 s)")
r, reloj, n = ocr_con_reloj(main, listo_en=None)
ok(isinstance(r, RuntimeError) and reloj.t == 120.0 and n == 120 and "120 sondeos" in str(r),
   f"si nunca termina, el mismo tope de 120 s de espera ({n} consultas): «{str(r)[:60]}…»")
if antes is not None:
    r_a, reloj_a, n_a = ocr_con_reloj(antes, listo_en=None)
    ok(isinstance(r_a, RuntimeError) and reloj_a.t == 120.0, f"(antes también 120 s, en {n_a} consultas)")


# ═════════════════════════════════════════════════════════════════════════════
print("\n5 · LOS DOS VECTORES DE LA PASADA POR CONCEPTO, LA CORRECCIÓN Y EL PLAN")


async def _emb_lento(texto, modelo=None):
    await asyncio.sleep(0.2)
    if texto == "falla":
        raise Caida(f"vector de «{texto}»")
    return vector_de(texto, modelo)
main.get_dense_embedding = _emb_lento
t0 = time.perf_counter()
vs, _ = callado(main._vectores_en_paralelo(["concepto limpio", "documento hipotético"]))
dt = time.perf_counter() - t0
ok(vs == [vector_de("concepto limpio"), vector_de("documento hipotético")] and dt < 0.35,
   f"dos vectores en {dt:.2f} s (uno detrás del otro serían 0.40) y en su orden")
vs, _ = callado(main._vectores_en_paralelo(["concepto limpio", "falla"]))
ok(vs[0] == vector_de("concepto limpio") and isinstance(vs[1], Caida),
   "si falla el segundo, el primero llega y el fallo queda en SU lugar (se lanza al llegar a él)")
FUENTE = (AQUI / "main.py").read_text(encoding="utf-8")
ok("_dvs = await _vectores_en_paralelo([_txt_a for _, _txt_a in _anclas])" in FUENTE
   and "if isinstance(_dv, BaseException):\n                                        raise _dv" in FUENTE
   and "_dv = await get_dense_embedding(_txt)" not in FUENTE,
   "la pasada por concepto los pide así, y lanza el fallo en el orden de antes")


class _SupabaseLento:
    def __init__(self, falla=False):
        self.filas, self.falla, self.hilos = [], falla, []

    def table(self, nombre):
        sb = self

        class _T:
            def insert(self, fila):
                class _E:
                    def execute(self_):
                        sb.hilos.append(threading.current_thread() is threading.main_thread())
                        time.sleep(0.3)
                        if sb.falla:
                            raise ConnectionError("base caída")
                        sb.filas.append((nombre, fila))
                return _E()
        return _T()


async def _anotar_y_seguir(sb):
    main.supabase_admin = sb
    t0 = time.perf_counter()
    tarea = asyncio.ensure_future(asyncio.to_thread(main._anotar_correccion, {"texto": "estás equivocado"}))
    seguido = time.perf_counter() - t0
    await tarea
    return seguido


sb = _SupabaseLento()
seguido, _ = callado(_anotar_y_seguir(sb))
ok(seguido < 0.05 and sb.filas == [("correcciones_usuario", {"texto": "estás equivocado"})] and sb.hilos == [False],
   f"la corrección se guarda en un hilo y la consulta sigue sin esperarla ({seguido * 1000:.0f} ms)")
sb = _SupabaseLento(falla=True)
_, log = callado(_anotar_y_seguir(sb))
ok("⚠️ No pude anotar la corrección (no fatal): ConnectionError" in log,
   "si la base falla, la misma línea de siempre en el registro")
main.supabase_admin = None
ok("asyncio.to_thread(_anotar_correccion, _fila_corr)" in FUENTE
   and 'supabase_admin.table("correcciones_usuario").insert({' not in FUENTE,
   "en la ruta ya no queda el insert síncrono")

_i_entrada = FUENTE.find("_nivel_mem_tarea = asyncio.ensure_future(nivel_del_usuario(request.user_id))")
_i_cobro = FUENTE.find("infra_check_task = asyncio.create_task(_run_infra_checks())")
ok(0 < _i_entrada < _i_cobro and "_nivel_mem = await _nivel_mem_tarea" in FUENTE
   and "await nivel_del_usuario(request.user_id)" not in FUENTE,
   "el plan para la memoria sale al entrar en /chat y se espera donde se usaba")
ok(correr(main.nivel_del_usuario(None)) == "gratuito", "y sin usuario sigue siendo «gratuito»")

print()
if FALLOS:
    print(f"✗ {len(FALLOS)} FALLOS")
    for f in FALLOS:
        print("   ·", f)
else:
    print("✓ TODO PASA")
sys.exit(1 if FALLOS else 0)
