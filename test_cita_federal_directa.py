"""La cita que nombra una ley federal va a esa ley — 25-sep-2026.

    .venv/bin/python test_cita_federal_directa.py

Medido contra producción ese día: «el artículo 29 de la Ley Federal de las
Entidades Paraestatales» inyectaba el 29 de la ley de paraestatales de
Michoacán; el 17 de la Ley Federal del Trabajo, el 17 del Código Penal del
Distrito Federal. Sin red: Qdrant es falso y registra qué se le preguntó, que
es lo que importa — a qué colección y con qué filtro.
"""
import asyncio
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.getcwd())
import main

FALLOS = []


def ok(cond, que):
    print(f"   {'PASA ' if cond else 'FALLA'}  {que}")
    if not cond:
        FALLOS.append(que)


FEDERALES = [
    "Ley Federal de las Entidades Paraestatales",
    "Ley de Adquisiciones, Arrendamientos y Servicios del Sector Público",
    "Ley de Obras Públicas y Servicios Relacionados con las Mismas",
    "Código Fiscal de la Federación", "Ley Federal del Trabajo", "Código Civil Federal",
    "Codigo NACIONAL DE PROCEDIMIENTOS PENALES", "Código Nacional de Procedimientos Civiles y Familiares",
    "Ley de Puertos", "Ley Agraria", "Ley de Amparo",
]
main._NOMBRES_FEDERALES["lista"] = sorted(((main._normalizar_nombre_ley(n), n) for n in FEDERALES),
                                          key=lambda x: -len(x[0]))
main._NOMBRES_FEDERALES["ts"] = 9e18          # que no intente refrescar contra la red

pista = main._ley_federal_de_pista
print("\n1 · QUÉ LEY FEDERAL NOMBRA UNA PISTA")
ok(pista("Ley Federal de las Entidades Paraestatales?") == "Ley Federal de las Entidades Paraestatales",
   "nombre completo, aunque traiga el signo de interrogación pegado")
ok(pista("LEY FEDERAL DE LAS ENTIDADES PARAESTATALES") == "Ley Federal de las Entidades Paraestatales",
   "sin importar mayúsculas")
ok(pista("Código Fiscal de la Federacion") == "Código Fiscal de la Federación", "ni acentos")
ok(pista("Ley de Adquisiciones") is None, "el arranque del nombre no basta por omisión")
ok(pista("Ley de Adquisiciones", permitir_prefijo=True).startswith("Ley de Adquisiciones, Arrendamientos"),
   "con prefijo permitido, «Ley de Adquisiciones» es la federal (única que empieza así)")
ok(pista("Código Civil", permitir_prefijo=True) is None, "«Código Civil» no se lee como el Federal")
ok(pista("Código Nacional de Procedimientos", permitir_prefijo=True) is None,
   "un arranque que casa con dos códigos no elige ninguno")
ok(pista("Ley de Entidades Paraestatales del Estado de Michoacán") is None, "la ley estatal no se hace federal")
ok(pista("Código Civil para el Estado de Jalisco") is None, "el código civil de un estado tampoco")
ok(pista("puertos") is None, "una palabra suelta no es el nombre de la ley")
ok(pista("la LFT") == "Ley Federal del Trabajo", "los alias escritos a mano siguen valiendo")
ok(main._detect_ley_federal_mencionada("¿qué dice la Ley de Puertos sobre la API?") == "Ley de Puertos",
   "la búsqueda principal reconoce una ley que no está en el mapa escrito a mano")


class QdrantFalso:
    def __init__(self):
        self.preguntas = []

    async def scroll(self, collection_name, scroll_filter=None, limit=10, **kw):
        conds = {}
        for c in scroll_filter.must:
            v = getattr(c.match, "value", None)
            if v is None:                       # MatchAny([29]) → 29
                v = getattr(c.match, "any", None)
                v = v[0] if isinstance(v, list) and len(v) == 1 else v
            conds[c.key] = v
        self.preguntas.append((collection_name, conds))
        if collection_name == "leyes_federales" and conds.get("ley") == "Ley Federal de las Entidades Paraestatales" \
                and conds.get("articulo_num") == 29:
            return [SimpleNamespace(id="lfep-29", payload={
                "texto": "[Ley Federal de las Entidades Paraestatales]\nARTICULO 29.- No tienen el carácter…",
                "ref": "Artículo 29.", "ley": "Ley Federal de las Entidades Paraestatales",
                "cuerpo_legal_oficial": "Ley Federal de las Entidades Paraestatales",
                "materia": "administrativo", "entidad": "FEDERAL", "chunk_index": 0,
                "url_pdf": "https://www.diputados.gob.mx/LeyesBiblio/pdf/LFEP.pdf"})], None
        if collection_name == "leyes_michoacan" and conds.get("articulo_num") == 29:
            return [SimpleNamespace(id="mich-29", payload={
                "texto": "ARTICULO 29. El Ejecutivo del Estado…", "ref": "Art. 29",
                "origen": "Ley de Entidades Paraestatales del Estado de Michoacán."})], None
        return [], None


async def parte_directa():
    print("\n2 · LA BÚSQUEDA DIRECTA DEL CHAT")
    main.qdrant_client = QdrantFalso()
    q = main.qdrant_client
    citas = main._extract_legal_citations("¿Qué dice el artículo 29 de la Ley Federal de las Entidades Paraestatales?")
    rs = await main._buscar_articulos_citados(citas, None)
    ok(len(rs) == 1 and rs[0].id == "lfep-29" and rs[0].silo == "leyes_federales",
       "inyecta el 29 de la LFEP y nada más")
    ok(rs and rs[0].origen == "Ley Federal de las Entidades Paraestatales" and rs[0].pdf_url.endswith("LFEP.pdf"),
       "con su nombre de ley y su PDF")
    ok(all(c == "leyes_federales" for c, _ in q.preguntas), "no recorre las colecciones estatales")
    ok(any(f.get("ley") == "Ley Federal de las Entidades Paraestatales" for _, f in q.preguntas),
       "pregunta por ley + número, no por número a secas")

    main.qdrant_client = QdrantFalso()
    citas = main._extract_legal_citations("artículo 30 de la Ley Federal de las Entidades Paraestatales")
    rs = await main._buscar_articulos_citados(citas, None)
    ok(rs == [], "si la ley nombrada no tiene ese artículo, no se inyecta el de otra")

    main.qdrant_client = QdrantFalso()
    citas = main._extract_legal_citations("artículo 29 de la Ley de Entidades Paraestatales del Estado de Michoacán")
    rs = await main._buscar_articulos_citados(citas, None)
    ok(len(rs) == 1 and rs[0].id == "mich-29", "la cita de una ley estatal sigue su camino de siempre")
    ok(not any(f.get("ley") for _, f in main.qdrant_client.preguntas), "y no se le fuerza la federal")


asyncio.run(parte_directa())

print("\n" + ("TODO PASA" if not FALLOS else f"{len(FALLOS)} FALLAN: {FALLOS}"))
raise SystemExit(1 if FALLOS else 0)
