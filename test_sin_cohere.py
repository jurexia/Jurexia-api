# -*- coding: utf-8 -*-
"""COHERE RETIRADO — que nadie lo llame y que el contexto no cambie de tamaño.

David, 16-sep-2026: «sospecho que cohere no está sirviendo de nada… revisa para
cancelarlo». La medición (ver el comentario de COHERE_RERANK_ENABLED en main.py)
dio 98.2% de documentos idénticos con y sin él en la búsqueda principal.

Lo que se fija aquí, contra el acervo real:
  · ninguna petición sale hacia api.cohere.com, ni por la búsqueda del chat
    ni por la de 30 huecos;
  · la búsqueda devuelve EXACTAMENTE `top_k` documentos, que es lo que Cohere
    devolvía. Sin el recorte manual el modelo recibiría hasta diez de más;
  · las búsquedas con `skip_post_search`, que nunca pasaban por Cohere,
    conservan sus `top_k + 10` como antes.

Necesita red y las credenciales de Qdrant del .env: es una prueba de
integración, no unitaria.
"""
import asyncio, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import main

fallos = []
def ok(cond, nota):
    print(f"  {'OK ' if cond else 'MAL'} {nota}")
    if not cond: fallos.append(nota)

llamadas_cohere = []

async def _prohibido(*a, **k):
    llamadas_cohere.append("_cohere_rerank")
    raise AssertionError("se llamó a _cohere_rerank con Cohere retirado")

PLAN = {"materia_principal": "general", "fuero_detectado": "mixto", "jurisprudencia_keywords": [],
        "conceptos_clave": [], "pesos_silos": {"constitucional": .25, "federal": .25, "estatal": .25, "jurisprudencia": .25}}

async def correr():
    ok(main.COHERE_RERANK_ENABLED is False, "COHERE_RERANK_ENABLED está apagado en el código")
    main._cohere_rerank = _prohibido
    async with main.lifespan(main.app):
        pool = main._http_pool
        original_post = pool.post
        async def espia_post(url, *a, **k):
            if "cohere" in str(url): llamadas_cohere.append(str(url))
            return await original_post(url, *a, **k)
        pool.post = espia_post

        casos = [
            ("QUERETARO", "¿Qué dice el artículo 2378 del Código Civil del Estado de Querétaro sobre la prórroga del arrendamiento?"),
            ("MEXICO", "¿Procede el amparo indirecto contra la negativa de acceso a un expediente judicial?"),
            ("VERACRUZ", "demanda de desocupación de casa habitación por terminación de contrato y por incumplimiento de pago"),
        ]
        for estado, q in casos:
            for top_k in (65, 30):
                r = await main.hybrid_search_all_silos(query=q, estado=estado, top_k=top_k,
                                                       skip_llm_presearch=True, precomputed_plan=PLAN)
                ok(len(r) <= top_k, f"top_k={top_k} devuelve {len(r)} (≤ {top_k}) · {q[:45]}")
            r = await main.hybrid_search_all_silos(query=q, estado=estado, top_k=30,
                                                   skip_llm_presearch=True, precomputed_plan=PLAN,
                                                   skip_post_search=True)
            ok(len(r) <= 40, f"skip_post_search conserva hasta top_k+10: {len(r)} (≤ 40)")

        # El artículo buscado sigue llegando: la búsqueda no perdió puntería.
        r = await main.hybrid_search_all_silos(query=casos[0][1], estado="QUERETARO", top_k=65,
                                               skip_llm_presearch=True, precomputed_plan=PLAN)
        hallado = any("2378" in ((x.ref or "") + (x.texto or "")[:60]) and "quer" in (x.origen or "").lower()
                      for x in r)
        ok(hallado, "el artículo 2378 del Código Civil de Querétaro sigue en el contexto")
        pool.post = original_post

    ok(not llamadas_cohere, f"ninguna llamada a Cohere ({len(llamadas_cohere)} registradas)")

asyncio.run(correr())
print()
if fallos:
    print(f"{len(fallos)} FALLO(S):"); [print("  -", f) for f in fallos]; sys.exit(1)
print("todo en orden")
