# -*- coding: utf-8 -*-
"""LA LISTA Y EL RERANK DE TESIS — las piezas que no necesitan red.

Lo que se mide de verdad está en redactor-sentencias/rag/medir_v4.py contra 404
tesis reales. Aquí se garantiza que las piezas existan, lean con tolerancia y
no rompan en silencio: la fusión por rango, el prompt del rerank, la lectura
del orden y el marcado de las elegidas. Y que apagar la bandera deje la lista
tal como venía: una capa que se pueda apagar sin ruido se mide, no se supone.
"""
import asyncio, json, sys
import fase6_rag as r

fallos = []
def ok(c, nota):
    print(f"  {'OK ' if c else 'MAL'} {nota}")
    if not c: fallos.append(nota)

# ── fusión por rango ──
f = r._rrf_registros([["a", "b", "c"], ["c", "a", "d"], ["e"]])
ok(f[0] == "a" and f[1] == "c", "RRF: lo que aparece bien situado en dos listas sube")
ok(f.index("e") > f.index("c") and f.index("d") > f.index("c"), "RRF: lo que aparece en una sola lista queda detrás de lo que aparece en dos")
ok(r._rrf_registros([["", "x"], [None, "x"]]) == ["x"], "RRF: ignora registros vacíos")

# ── el prompt ──
pr = r._prompt_rerank("¿La Sala valoró bien la pericial?", "Resolvió que sí. Se combate que no ratificó.",
                      ["PRUEBA PERICIAL. VALORACIÓN", "COSTAS. CONDENA"])
ok("1. PRUEBA PERICIAL" in pr and "2. COSTAS" in pr, "el prompt numera las candidatas")
ok("RESOLVIÓ Y SE COMBATE" in pr, "lleva lo que se resolvió y se combate")
ok("RESOLVIÓ Y SE COMBATE" not in r._prompt_rerank("p", "", ["x"]), "sin hecho, sin ese bloque")

# ── leer el orden ──
ok(r._leer_orden('{"orden": [3, 1, 3, 99, "2"]}', 5) == [2, 0, 1], "lee, quita repetidos y fuera de rango, acepta cadenas")
ok(r._leer_orden("nada", 5) == [], "sin JSON, vacío")
ok(r._leer_orden('{"orden": []}', 5) == [], "lista vacía es vacía, no invento")
ok(len(r._leer_orden(json.dumps({"orden": list(range(1, 40))}), 40)) == 10, "como mucho diez")

# ── el rerank marca y no rompe ──
class _R:
    def __init__(s, t): s.choices = [type("c", (), {"message": type("m", (), {"content": t})()})()]
class Falso:
    def __init__(s, t): s.t = t; s.chat = s
    @property
    def completions(s): return s
    async def create(s, **kw): return _R(s.t)

cands = [{"registro": "1", "rubro": "A"}, {"registro": "2", "rubro": "B"}, {"registro": "3", "rubro": "C"}]
out = asyncio.run(r.rerank_tesis(Falso('{"orden": [2, 3]}'), "p", "h", [dict(c) for c in cands]))
ok(out[1].get("rerank") == 1 and out[2].get("rerank") == 2 and "rerank" not in out[0], "marca las elegidas en su orden")
out2 = asyncio.run(r.rerank_tesis(Falso("sin json"), "p", "h", [dict(c) for c in cands]))
ok(all("rerank" not in c for c in out2), "sin JSON, nadie queda marcado")
ok(asyncio.run(r.rerank_tesis(None, "p", "h", [dict(c) for c in cands]))[0].get("rerank") is None, "sin cliente, no hace nada")

# ── la bandera apaga de verdad ──
r.RAG_RERANK_TESIS = False
out3 = asyncio.run(r.rerank_tesis(Falso('{"orden": [1]}'), "p", "h", [dict(c) for c in cands]))
ok(all("rerank" not in c for c in out3), "con la bandera apagada, la lista se queda como venía")
r.RAG_RERANK_TESIS = True

# ── el orden final respeta al rerank y, sin él, lo de siempre ──
tesis = [{"registro": "1", "rubro": "A", "obligatoria": False, "instancia": "Tribunales Colegiados", "veces": 0},
         {"registro": "2", "rubro": "B", "obligatoria": True,  "instancia": "Primera Sala", "veces": 0, "rerank": 1},
         {"registro": "3", "rubro": "C", "obligatoria": True,  "instancia": "Pleno", "veces": 0}]
tesis.sort(key=lambda t: (t.get("rerank", 10 ** 6), r._rango_instancia(t), not t["obligatoria"],
                          -int(t.get("veces") or 0), False))
ok([t["registro"] for t in tesis][0] == "2", "la elegida va primero aunque no sea del Pleno")
ok([t["registro"] for t in tesis][1] == "3", "y detrás manda lo de siempre: el Pleno obligatorio")

print()
if fallos: print(f"FALLAN {len(fallos)}: " + " · ".join(fallos)); sys.exit(1)
print("TODAS LAS COMPROBACIONES PASAN")
