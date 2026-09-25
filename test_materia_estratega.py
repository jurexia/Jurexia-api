"""La materia del chat la dictamina el Estratega, y no filtra: añade — 25-sep-2026.

    .venv/bin/python test_materia_estratega.py

Las palabras clave filtraban las leyes federales y tapaban leyes enteras —la
Ley de Amparo en todo lo «constitucional», los códigos procesales en lo civil—.
El chat ya no las usa: busca sin filtro y, cuando vuelve el Estratega, pide
hasta cinco federales MÁS de la materia que dictaminó. Sin red: la búsqueda en
Qdrant es falsa y registra qué filtro recibió cada colección.
"""
import asyncio
import os
import sys

sys.path.insert(0, os.getcwd())
import main

FALLOS = []


def ok(cond, que):
    print(f"   {'PASA ' if cond else 'FALLA'}  {que}")
    if not cond:
        FALLOS.append(que)


def condiciones(filtro, llave):
    """Los valores que un filtro exige (must o should) para una llave."""
    if filtro is None:
        return []
    vistas = []
    for grupo in (filtro.must or [], filtro.should or []):
        for c in grupo:
            if getattr(c, "key", None) == llave:
                v = getattr(c.match, "any", None) or [getattr(c.match, "value", None)]
                vistas.append(set(v))
    return vistas


print("\n1 · QUÉ MATERIA DEL DICTAMEN SIRVE PARA BUSCAR")
m = main._materia_del_estratega
ok(m({"materia_principal": "Laboral"}) == "laboral", "la del dictamen, en minúsculas")
ok(m({"materia_principal": " penal "}) == "penal", "sin espacios alrededor")
ok(m({"materia_principal": "procesal"}) is None, "«procesal» no dice qué leyes: no suma nada")
ok(m({"materia_principal": None}) is None and m({}) is None, "sin materia, nada")
ok(m({"materia_principal": "electoral"}) is None, "una que no está en el mapa, nada")
ok(m(None) is None and m("civil") is None, "un dictamen que no es dict (el Estratega falló), nada")
ok("amparo" in main._MATERIA_ESTRATEGA_EN_FEDERAL["constitucional"],
   "lo constitucional incluye la Ley de Amparo, que el filtro viejo dejaba fuera")
ok("procesal_civil" in main._MATERIA_ESTRATEGA_EN_FEDERAL["civil"]
   and "procesal_civil" in main._MATERIA_ESTRATEGA_EN_FEDERAL["familiar"],
   "lo civil y lo familiar incluyen sus códigos procesales")


class Registro:
    def __init__(self, devolver=None):
        self.llamadas = []
        self.devolver = devolver or {}

    async def buscar(self, collection, query, dense_vector, sparse_vector, filter_=None, top_k=10, alpha=0.7, **kw):
        self.llamadas.append((collection, filter_, top_k))
        return list(self.devolver.get(collection, []))


def resultado(i, silo="leyes_federales", materia="laboral"):
    return main.SearchResult(id=f"{silo}-{i}", score=1.0 - i / 100, texto="…", ref=f"Artículo {i}",
                             silo=silo, materia_meta=materia)


async def falso_denso(texto, modelo=None):
    return [0.0] * 8


async def parte_suplemento():
    print("\n2 · EL SUPLEMENTO: AÑADE FEDERALES DE LA MATERIA, NO QUITA NADA")
    main.get_dense_embedding = falso_denso
    reg = Registro({"leyes_federales": [resultado(i) for i in range(12)]})
    main.hybrid_search_single_silo = reg.buscar

    rs = await main._federales_de_la_materia("me despidieron sin causa", "laboral",
                                             ya_presentes={"leyes_federales-0", "leyes_federales-1"})
    ok(len(reg.llamadas) == 1 and reg.llamadas[0][0] == "leyes_federales", "una sola búsqueda, en lo federal")
    ok(condiciones(reg.llamadas[0][1], "materia") == [{"laboral", "seguridad_social"}],
       "filtrada por las materias de lo laboral")
    ok([r.id for r in rs] == [f"leyes_federales-{i}" for i in range(2, 7)],
       "salta lo que ya estaba y se queda en cinco")

    reg.llamadas.clear()
    ok(await main._federales_de_la_materia("x", None, set()) == [] and not reg.llamadas,
       "sin materia no busca")
    ok(await main._federales_de_la_materia("x", "procesal", set()) == [] and not reg.llamadas,
       "con «procesal» tampoco")


async def parte_busqueda():
    print("\n3 · LA BÚSQUEDA DEL CHAT YA NO FILTRA POR PALABRAS CLAVE")
    main._NOMBRES_FEDERALES["lista"] = [(main._normalizar_nombre_ley("Ley Agraria"), "Ley Agraria")]
    main._NOMBRES_FEDERALES["ts"] = 9e18          # que no intente refrescar contra la red
    plan = {"materia_principal": "general", "fuero_detectado": "mixto", "jurisprudencia_keywords": [],
            "conceptos_clave": [], "pesos_silos": {"constitucional": 0.25, "federal": 0.25,
                                                   "estatal": 0.25, "jurisprudencia": 0.25}}
    consulta = "me despidieron y el patrón no me quiere pagar el finiquito ni los salarios caídos"
    ok(main._detect_materia(consulta) == ["LABORAL"], "la consulta de prueba dispara LABORAL por palabras")

    async def filtro_federal(**kw):
        reg = Registro()
        main.hybrid_search_single_silo = reg.buscar
        await main.hybrid_search_all_silos(query=consulta, estado=None, top_k=10, skip_llm_presearch=True,
                                           precomputed_plan=plan, skip_post_search=True, **kw)
        return [f for c, f, _ in reg.llamadas if c == "leyes_federales"]

    antes = await filtro_federal()
    ok(any(condiciones(f, "materia") for f in antes),
       "por omisión (los demás flujos) sigue filtrando por materia: la prueba sí lo detecta")
    chat = await filtro_federal(materia_por_palabras=False)
    ok(chat and not any(condiciones(f, "materia") for f in chat),
       "el chat busca en lo federal sin filtro de materia")
    genio = await filtro_federal(materia_por_palabras=False, forced_materia="LABORAL")
    ok(any(condiciones(f, "materia") for f in genio),
       "la materia de un Genio sigue mandando aunque no haya palabras clave")


asyncio.run(parte_suplemento())
asyncio.run(parte_busqueda())

print("\n" + ("TODO PASA" if not FALLOS else f"{len(FALLOS)} FALLAN: {FALLOS}"))
raise SystemExit(1 if FALLOS else 0)
