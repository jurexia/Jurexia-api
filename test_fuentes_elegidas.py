"""El selector de fuentes, sin tocar la red — 23-sep-2026.

    .venv/bin/python test_fuentes_elegidas.py

Lo que importa probar no es el `if`: es que el veto alcance a las tareas que
nacen de la consulta —ahí es donde corre el buscador— y que la respuesta
vacía tenga la forma que el llamador espera, para que nadie reviente.
"""
import asyncio

import fuentes_elegidas as fe

FALLOS = []


def ok(cond, que):
    print(f"   {'PASA ' if cond else 'FALLA'}  {que}")
    if not cond:
        FALLOS.append(que)


print("\n1 · LO QUE MANDA EL CLIENTE")
ok(fe.normalizar(None) is None, "sin selector no hay filtro")
ok(fe.normalizar([]) is None, "lista vacía = sin filtro (nunca «no buscar nada»)")
ok(fe.normalizar(list(fe.FUENTES)) is None, "las cuatro = sin filtro")
ok(fe.normalizar(["federal", "Jurisprudencia "]) == frozenset({"federal", "jurisprudencia"}), "normaliza mayúsculas y espacios")
ok(fe.normalizar("constitucional,federal") == frozenset({"constitucional", "federal"}), "el formulario manda texto con comas")
ok(fe.normalizar(["federal", "inventada"]) == frozenset({"federal"}), "lo desconocido se ignora")
ok(fe.normalizar(["inventada"]) is None, "si no queda nada válido, sin filtro")
ok(fe.normalizar(123) is None, "un tipo raro no rompe")

print("\n2 · A QUÉ FUENTE PERTENECE CADA COLECCIÓN")
ok(fe.categoria("leyes_federales") == "federal", "leyes_federales → federal")
ok(fe.categoria("leyes_hidalgo") == "estatal", "leyes_hidalgo → estatal")
ok(fe.categoria("leyes_estatales") == "estatal", "la vieja leyes_estatales → estatal")
ok(fe.categoria("bloque_constitucional") == "constitucional", "bloque → constitucional")
for c in ("jurisprudencia_nacional_v3", "jurisprudencia_tcc", "sentencias_holdings",
          "sentencias_scjn_holdings", "sentencias_ef_scjn_pleno"):
    ok(fe.categoria(c) == "jurisprudencia", f"{c} → jurisprudencia")
ok(fe.categoria("documentos_usuario") is None, "lo que no es fuente del selector no tiene categoría")

print("\n3 · EL FUERO QUE LE CORRESPONDE AL BUSCADOR")
ok(fe.fuero_equivalente(frozenset({"federal", "jurisprudencia", "constitucional"})) == "constitucional,federal", "todo menos estatal")
ok(fe.fuero_equivalente(frozenset({"estatal"})) == "estatal", "sólo estatal")
ok(fe.fuero_equivalente(frozenset({"jurisprudencia"})) is None, "sólo jurisprudencia no es un fuero")
ok(fe.fuero_equivalente(None) is None, "sin filtro, sin fuero")


class Punto:
    def __init__(self, silo):
        self.silo = silo


class ClienteFalso:
    """Las mismas firmas que AsyncQdrantClient en lo que usa el chat."""
    def __init__(self):
        self.llamadas = []

    async def query_points(self, collection_name, **k):
        from qdrant_client.http import models
        self.llamadas.append(collection_name)
        return models.QueryResponse(points=[models.ScoredPoint(id=1, version=1, score=0.9, payload={})])

    async def scroll(self, collection_name, **k):
        self.llamadas.append(collection_name)
        return ([1, 2], "siguiente")

    async def retrieve(self, collection_name, ids, **k):
        self.llamadas.append(collection_name)
        return [1]

    async def count(self, collection_name, **k):
        from qdrant_client.http import models
        self.llamadas.append(collection_name)
        return models.CountResult(count=7)

    async def query_batch_points(self, collection_name, requests, **k):
        from qdrant_client.http import models
        self.llamadas.append(collection_name)
        return [models.QueryResponse(points=[]) for _ in requests]

    async def upsert(self, collection_name, points, **k):
        self.llamadas.append(("upsert", collection_name))
        return "ok"


print("\n4 · EL VETO EN EL CLIENTE")
cli = ClienteFalso()
n = fe.vetar(cli)
ok(n == 5, f"envuelve las cinco lecturas que tiene el falso ({n})")
ok(fe.vetar(cli) == 0, "envolver dos veces no duplica")


async def consulta(elegidas):
    """Como el chat: fija las fuentes y el buscador corre en tareas hijas."""
    fe.fijar(elegidas)

    async def buscador_estatal():
        return await cli.query_points(collection_name="leyes_hidalgo", limit=5)

    async def buscador_federal():
        return await cli.query_points(collection_name="leyes_federales", limit=5)

    r_est, r_fed = await asyncio.gather(
        asyncio.create_task(buscador_estatal()), buscador_federal())
    scroll = await cli.scroll("leyes_hidalgo", limit=10)
    lote = await cli.query_batch_points(collection_name="leyes_hidalgo", requests=[1, 2, 3])
    cuenta = await cli.count(collection_name="leyes_hidalgo")
    rec = await cli.retrieve(collection_name="leyes_hidalgo", ids=["x"])
    escrito = await cli.upsert(collection_name="leyes_hidalgo", points=[])
    return r_est, r_fed, scroll, lote, cuenta, rec, escrito


antes = len(cli.llamadas)
r_est, r_fed, scroll, lote, cuenta, rec, escrito = asyncio.run(
    consulta(frozenset({"constitucional", "jurisprudencia", "federal"})))
ok(r_est.points == [], "la tarea hija que pide leyes de Hidalgo recibe vacío")
ok(len(r_fed.points) == 1, "la federal responde normal")
ok(scroll == ([], None), "el scroll vetado se desempaqueta en dos")
ok(len(lote) == 3 and all(x.points == [] for x in lote), "el lote vetado trae una respuesta vacía por petición")
ok(cuenta.count == 0, "la cuenta vetada es cero")
ok(rec == [], "retrieve vetado: lista vacía")
ok(escrito == "ok", "las escrituras no se vetan")
ok("leyes_hidalgo" not in cli.llamadas[antes:], "a Hidalgo ni siquiera se le llamó")
ok(fe.informe().get("leyes_hidalgo", 0) >= 5, "y el informe lo cuenta")

antes = len(cli.llamadas)
r_est, *_ = asyncio.run(consulta(None))
ok(len(r_est.points) == 1 and "leyes_hidalgo" in cli.llamadas[antes:], "sin selector, Hidalgo responde como siempre")

# Una consulta no contamina a la siguiente: cada una corre en su contexto.
async def otra():
    return await cli.query_points(collection_name="leyes_hidalgo")
ok(len(asyncio.run(otra()).points) == 1, "fuera de la consulta que lo fijó no hay veto")

print("\n5 · EL SEGUNDO CINTURÓN Y LA INSTRUCCIÓN")
res = [Punto("leyes_hidalgo"), Punto("leyes_federales"), Punto("jurisprudencia_nacional_v3"), Punto("web")]
f = fe.filtrar(res, frozenset({"federal"}))
ok([p.silo for p in f] == ["leyes_federales", "web"], "filtrar deja lo elegido y lo que no es fuente del selector")
ok(len(fe.filtrar(res, None)) == 4, "sin selector no quita nada")
txt = fe.instruccion(frozenset({"federal", "jurisprudencia", "constitucional"}), "Hidalgo")
ok("APAGÓ las leyes del estado de Hidalgo" in txt, "la instrucción nombra lo apagado con la entidad")
ok("de memoria" in txt, "y prohíbe rellenarlo de memoria")
ok(fe.instruccion(None) == "", "sin selector, ninguna instrucción")
ok(fe.excluye("estatal", frozenset({"federal"})) and not fe.excluye("estatal", None), "excluye()")

print("\n6 · INTERNET CRUZADA CON LAS DEMÁS FUENTES (25-sep-2026)")
import busqueda_web as bw
ids = lambda f: [a["id"] for a in bw.agentes_para(f)]
ok(ids(None) == ["vigencia", "criterios", "local"], "sin selector corren los tres agentes")
ok(ids(fe.normalizar(list(fe.FUENTES))) == ["vigencia", "criterios", "local"], "con las cuatro, los tres")
ok(ids(fe.normalizar(["jurisprudencia"])) == ["criterios"], "sólo jurisprudencia → sólo criterios (SCJN, CJF)")
ok(ids(fe.normalizar(["federal"])) == ["vigencia"], "sólo leyes federales → sólo vigencia (DOF, Cámara)")
ok(ids(fe.normalizar(["estatal"])) == ["local"], "sólo leyes estatales → sólo el agente local")
ok(ids(fe.normalizar(["constitucional"])) == ["vigencia", "criterios"], "bloque constitucional → reformas y criterios")
ok(ids(fe.normalizar(["jurisprudencia", "estatal"])) == ["criterios", "local"], "dos rubros → sus dos agentes")
con = fe.instruccion(frozenset({"jurisprudencia"}), None, con_internet=True)
sin = fe.instruccion(frozenset({"jurisprudencia"}), None)
ok("«Internet»" in con and "«Internet»" not in sin, "con Internet encendida, la instrucción deja usar lo de internet")
ok(fe.instruccion(None, None, con_internet=True) == "", "sin selector sigue sin instrucción")

async def _sin_agentes():
    return bw.lanzar_agentes("¿procede el amparo?", None, ())
_web = bw.WEB_ACTIVA
bw.WEB_ACTIVA = True
ok(asyncio.run(_sin_agentes()) == [], "agentes=() no lanza nada (no es «los tres»)")
bw.WEB_ACTIVA = _web

print("\n" + ("TODO PASA" if not FALLOS else f"{len(FALLOS)} FALLAN: {FALLOS}"))
raise SystemExit(1 if FALLOS else 0)
