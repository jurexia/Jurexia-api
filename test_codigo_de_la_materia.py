# -*- coding: utf-8 -*-
"""El código de la materia y el CNPP llegan al contexto (27-sep-2026).

Parte 1, sin red: el vocabulario del filtro federal, la inyección por nombre
exacto con un Qdrant falso (qué entra, qué no, repetidos, tope, el caso de
la colección que no tiene el código) y el piso del CNPP.

Parte 2, Qdrant REAL en sólo lectura (sólo si hay credenciales en el .env):
qué colecciones indexan `materia`, que el patrón encuentra el código penal
en las 32 entidades, y que el filtro por `origen` devuelve sólo artículos
del Código Penal para el Distrito Federal con un vector cualquiera (el de un
artículo del Código Fiscal: la consulta «equivocada» del caso). Sin
embeddings ni modelos.

    python test_codigo_de_la_materia.py
"""
import asyncio
import contextlib
import io
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
FALLOS = []


def ok(cond, que):
    print(("   PASA   " if cond else "   FALLA  ") + que)
    if not cond:
        FALLOS.append(que)


_BUCLE = asyncio.new_event_loop()


def correr(co):
    # Un solo bucle para toda la prueba: el cliente asíncrono de Qdrant y el
    # semáforo de main.py quedan atados al primero que los usa.
    return _BUCLE.run_until_complete(co)


with contextlib.redirect_stdout(io.StringIO()):
    import main

SR = main.SearchResult

print("\n1 · EL FILTRO FEDERAL INCLUYE EL CÓDIGO PROCESAL DE LA MATERIA")
V = main._MATERIA_QDRANT_VALUES
ok("procesal_penal" in V["PENAL"], "PENAL incluye procesal_penal (el CNPP)")
ok("procesal_civil" in V["CIVIL"] and "procesal_civil" in V["FAMILIAR"], "CIVIL y FAMILIAR incluyen procesal_civil")
ok("amparo" in V["CONSTITUCIONAL"], "CONSTITUCIONAL incluye amparo (la Ley de Amparo)")
f = main.build_metadata_filter("PENAL")
ok({c.match.value for c in f.should} >= {"penal", "procesal_penal"}, "build_metadata_filter('PENAL') lo lleva")
fuente = Path("main.py").read_text(encoding="utf-8")
ok('and await _coleccion_indexa(silo_name, "materia")' in fuente
   and 'not (silo_name == "leyes_federales" and _ley_federal_detectada)' in fuente,
   "la materia sólo se aplica donde está indexada y nunca sobre el filtro por ley")


class _QdrantCaido:
    async def get_collection(self, nombre):
        raise TimeoutError("Qdrant no contesta")


_orig_q = main.qdrant_client
main.qdrant_client = _QdrantCaido()
with contextlib.redirect_stdout(io.StringIO()):
    _sin_esquema = correr(main._coleccion_indexa("leyes_prueba_caida", "materia"))
ok(_sin_esquema is True and "leyes_prueba_caida" not in main._INDICES_POR_COLECCION,
   "si Qdrant no da el esquema se filtra como antes (el reintento sin filtro cubre la falta de índice) y no se guarda")
main.qdrant_client = _orig_q

print("\n2 · LA INYECCIÓN POR NOMBRE EXACTO (Qdrant falso)")
_orig_nombres, _orig_busca = main._nombres_de_coleccion, main.hybrid_search_single_silo
NOMBRES = {
    "leyes_cdmx": ["Código Fiscal de la Ciudad de México", "Código Penal para el Distrito Federal",
                   "CÓDIGO DE PROCEDIMIENTOS CIVILES PARA EL DISTRITO FEDERAL",
                   "Código de Procedimientos Civiles para el Distrito Federal", "Código Civil para el Distrito Federal"],
    "leyes_oaxaca": ["Ley de Ingresos del Estado de Oaxaca"],
    "leyes_federales": ["Ley Federal del Trabajo", "Codigo NACIONAL DE PROCEDIMIENTOS PENALES"],
}
PEDIDOS = []


async def _nombres_falsos(col):
    return sorted(((main._normalizar_nombre_ley(n), n) for n in NOMBRES.get(col, [])), key=lambda x: -len(x[0]))


def _busca_falsa(respuesta):
    async def _f(collection, query, dense_vector, sparse_vector, filter_, top_k, alpha):
        PEDIDOS.append((collection, filter_, top_k))
        return [SR(**{**r.__dict__}) for r in respuesta.get(collection, [])]
    return _f


def _sr(i, origen, ref, silo="leyes_cdmx", score=0.5):
    return SR(id=i, score=score, texto="…", ref=ref, origen=origen, jurisdiccion=None, entidad=None, silo=silo)


main._nombres_de_coleccion = _nombres_falsos
try:
    fiscal = [_sr(f"f{i}", "Código Fiscal de la Ciudad de México", f"Art. {450 + i}", score=0.84) for i in range(5)]
    penal = [_sr(f"p{i}", "Código Penal para el Distrito Federal", f"Art. {90 + i}") for i in range(9)]
    main.hybrid_search_single_silo = _busca_falsa({"leyes_cdmx": penal})
    PEDIDOS.clear()
    out = correr(main._inyectar_codigo_de_la_materia(list(fiscal), "penal", "leyes_cdmx", "CIUDAD_DE_MEXICO",
                                                     "escrito", [0.1], None, 30, 0.7))
    col, filtro, k = PEDIDOS[0]
    ok(col == "leyes_cdmx" and any(c.key == "origen" and list(c.match.any) == ["Código Penal para el Distrito Federal"]
                                   for c in filtro.must),
       "busca en la colección del estado filtrando origen = «Código Penal para el Distrito Federal»")
    ok([r.origen for r in out[:6]] == ["Código Penal para el Distrito Federal"] * 6 and all(r.score >= 0.90 for r in out[:6]),
       "seis artículos del Código Penal encabezan el contexto (top_k 30) con puntuación ≥ 0.90")
    ok([r.id for r in out[6:]] == [r.id for r in fiscal], "lo que ya venía sigue detrás, intacto")

    PEDIDOS.clear()
    correr(main._inyectar_codigo_de_la_materia([], "penal", "leyes_cdmx", "CIUDAD_DE_MEXICO", "q", [0.1], None, 10, 0.7))
    ok(PEDIDOS and PEDIDOS[0][2] == 2 * 3, "con top_k 10 se reservan 2 lugares (pide 6 para escoger)")

    # Versiones repetidas del mismo código: un artículo entra una vez.
    dup = [_sr("c1", "CÓDIGO DE PROCEDIMIENTOS CIVILES PARA EL DISTRITO FEDERAL", "Art. 255"),
           _sr("c2", "Código de Procedimientos Civiles para el Distrito Federal", "Art. 255"),
           _sr("c3", "Código Civil para el Distrito Federal", "Art. 1910"),
           _sr("c4", "Código Civil para el Distrito Federal", "Art. 1910")]
    dup[3].texto = "… segunda parte del artículo 1910 …"   # el mismo artículo, otro trozo
    main.hybrid_search_single_silo = _busca_falsa({"leyes_cdmx": dup})
    PEDIDOS.clear()
    out = correr(main._inyectar_codigo_de_la_materia([], "civil", "leyes_cdmx", "CIUDAD_DE_MEXICO", "q", [0.1], None, 30, 0.7))
    ok([r.id for r in out] == ["c1", "c3", "c4"],
       f"el art. 255 del CPC entra una vez aunque haya dos versiones; las dos partes del 1910 entran ({[r.id for r in out]})")
    ok(len(list(PEDIDOS[0][1].must[-1].match.any)) == 3, "civil busca en el Código Civil y en las dos versiones del CPC")

    # Si ya venían bastantes artículos del código, no se busca.
    ya = [_sr(f"y{i}", "Código Penal para el Distrito Federal", f"Art. {i}") for i in range(6)]
    PEDIDOS.clear()
    out = correr(main._inyectar_codigo_de_la_materia(list(ya), "penal", "leyes_cdmx", "CIUDAD_DE_MEXICO", "q", [0.1], None, 30, 0.7))
    ok(not PEDIDOS and [r.id for r in out] == [r.id for r in ya], "si ya venían 6 artículos del código, no busca ni toca nada")

    # Un resultado de otra ley (reintento sin filtro) no se inyecta.
    main.hybrid_search_single_silo = _busca_falsa({"leyes_cdmx": [_sr("x", "Ley de Cultura Cívica de la Ciudad de México", "Art. 94")]})
    out = correr(main._inyectar_codigo_de_la_materia([], "penal", "leyes_cdmx", "CIUDAD_DE_MEXICO", "q", [0.1], None, 30, 0.7))
    ok(out == [], "si el filtro no se aplicó y vuelve otra ley, no se inyecta")

    # Entidad sin el código: no busca y no rompe.
    PEDIDOS.clear()
    out = correr(main._inyectar_codigo_de_la_materia([], "civil", "leyes_oaxaca", "OAXACA", "q", [0.1], None, 30, 0.7))
    ok(out == [] and not PEDIDOS, "una entidad sin ese código (Código Civil de Oaxaca) no busca y no rompe")

    # Laboral: la LFT, en la federal, filtrando por `ley`; su nombre se rellena.
    lft = [_sr(f"l{i}", None, f"Artículo {47 + i}.", silo="leyes_federales") for i in range(3)]
    main.hybrid_search_single_silo = _busca_falsa({"leyes_federales": lft})
    PEDIDOS.clear()
    out = correr(main._inyectar_codigo_de_la_materia([], "laboral", None, None, "despido", [0.1], None, 30, 0.7))
    ok(PEDIDOS and PEDIDOS[0][0] == "leyes_federales"
       and any(c.key == "ley" and list(c.match.any) == ["Ley Federal del Trabajo"] for c in PEDIDOS[0][1].must),
       "laboral busca la LFT en leyes_federales por `ley`, sin necesidad de entidad")
    ok([r.origen for r in out] == ["Ley Federal del Trabajo"] * 3, "a los artículos de la LFT se les pone su nombre")

    # Materias sin código: nada.
    PEDIDOS.clear()
    out = correr(main._inyectar_codigo_de_la_materia(list(fiscal), "mercantil", "leyes_cdmx", "CIUDAD_DE_MEXICO", "q", [0.1], None, 30, 0.7))
    ok(not PEDIDOS and out == fiscal, "una materia sin código definido no hace nada")

    # Si la búsqueda revienta, se sigue sin inyección.
    async def _revienta(**_k):
        raise RuntimeError("Qdrant caído")
    main.hybrid_search_single_silo = lambda **k: _revienta(**k)
    out = correr(main._inyectar_codigo_de_la_materia(list(fiscal), "penal", "leyes_cdmx", "CIUDAD_DE_MEXICO", "q", [0.1], None, 30, 0.7))
    ok(out == fiscal, "si Qdrant falla, devuelve lo que había")
finally:
    main._nombres_de_coleccion, main.hybrid_search_single_silo = _orig_nombres, _orig_busca

print("\n3 · EL CNPP «GARANTIZADO» LO ESTÁ DE VERDAD")
i = fuente.index("CÓDIGOS NACIONALES CON LUGAR GARANTIZADO")
tramo = fuente[i:i + 2500]
ok("_consulta_procesal_fuerte(query) else 2)" in tramo and "r.score = max(r.score, 0.86)" in tramo
   and 'if _materia_procesal == "procesal_penal" else 0' in tramo,
   "el CNPP: piso de 0.86 (bajo el 0.90 del código local), lugares completos sólo si la consulta es claramente procesal; el CNPCF, sin piso")
ok("and _consulta_procesal_fuerte(query)):" in fuente,
   "el código penal local sólo cede la mitad de lugares ante una consulta claramente procesal")
Q_SUST = "¿Qué pena corresponde al fraude procesal en la Ciudad de México, cuándo prescribe la acción y procede la reparación del daño a la víctima?"
ok(main._detectar_consulta_procesal(Q_SUST) is None, "«fraude procesal» es un delito, no una señal de procedimiento")
ok(main._detectar_consulta_procesal("¿Qué plazo tengo para apelar el auto de vinculación a proceso?") == "procesal_penal"
   and main._consulta_procesal_fuerte("¿Qué plazo tengo para apelar el auto de vinculación a proceso?"),
   "la vinculación a proceso sigue siendo procesal penal, y fuerte")
ok(not main._consulta_procesal_fuerte("Escrito sobre la pena del robo para la audiencia"),
   "una sola señal débil («audiencia») no es consulta claramente procesal")

print("\n4 · QDRANT REAL, SÓLO LECTURA")
try:
    from qdrant_client import QdrantClient, models
    e = {}
    for l in Path("/Users/josedavidalcantarmendoza/Documents/IUREXIA-MAC/jurexia-api-git/.env").read_text().splitlines():
        if "=" in l and not l.startswith("#"):
            k, v = l.split("=", 1)
            e[k.strip()] = v.strip().strip('"')
    q = QdrantClient(url=e["QDRANT_URL"], api_key=e["QDRANT_API_KEY"], timeout=60)
except Exception as ex:
    q = None
    print(f"   (sin Qdrant: {type(ex).__name__}; se salta la parte real)")
if q is not None:
    main.qdrant_client = __import__("qdrant_client").AsyncQdrantClient(url=e["QDRANT_URL"], api_key=e["QDRANT_API_KEY"], timeout=60)
    ok(correr(main._coleccion_indexa("leyes_federales", "materia")) is True, "leyes_federales indexa materia")
    ok(correr(main._coleccion_indexa("leyes_cdmx", "materia")) is False, "leyes_cdmx NO indexa materia (por eso ya no se le manda)")
    estatales = sorted(c.name for c in q.get_collections().collections
                       if c.name.startswith("leyes_") and c.name not in ("leyes_federales", "leyes_estatales",
                                                                         "leyes_administrativa", "leyes_civil",
                                                                         "leyes_laboral", "leyes_penal"))
    pat = main._CODIGOS_DE_LA_MATERIA["penal"][1]
    sin_penal = []
    for c in estatales:
        nombres = [o for n, o in correr(main._nombres_de_coleccion(c)) if any(re.search(p, n) for p in pat)]
        if len(nombres) != 1:
            sin_penal.append((c, nombres))
    ok(len(estatales) == 32 and not sin_penal, f"el patrón encuentra exactamente un código penal en las 32 entidades {sin_penal}")
    # El vector de un artículo del Código Fiscal (la consulta «equivocada») + filtro por origen.
    r, _ = q.scroll("leyes_cdmx", scroll_filter=models.Filter(must=[models.FieldCondition(
        key="origen", match=models.MatchValue(value="Código Fiscal de la Ciudad de México"))]), limit=1, with_vectors=True)
    vec = r[0].vector["dense"] if isinstance(r[0].vector, dict) else r[0].vector
    res = q.query_points("leyes_cdmx", query=vec, using="dense", limit=6, with_payload=["origen", "ref"],
                         query_filter=models.Filter(must=[models.FieldCondition(
                             key="origen", match=models.MatchAny(any=["Código Penal para el Distrito Federal"]))])).points
    ok(len(res) == 6 and all(p.payload.get("origen") == "Código Penal para el Distrito Federal" for p in res),
       f"filtrando por origen, hasta el vector del Código Fiscal devuelve sólo el Código Penal ({[p.payload.get('ref') for p in res]})")
    n = q.count("leyes_federales", exact=True, count_filter=models.Filter(
        must=[models.FieldCondition(key="ley", match=models.MatchValue(value="Codigo NACIONAL DE PROCEDIMIENTOS PENALES"))],
        should=[models.FieldCondition(key="materia", match=models.MatchValue(value=v)) for v in V["PENAL"]])).count
    ok(n > 400, f"«ley = CNPP» con la materia PENAL ampliada ya no da cero ({n} puntos)")

print()
if FALLOS:
    print(f"FALLAN {len(FALLOS)}:")
    for f_ in FALLOS:
        print("  ·", f_)
    sys.exit(1)
print("TODO PASA")
