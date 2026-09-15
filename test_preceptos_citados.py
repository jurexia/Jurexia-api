# -*- coding: utf-8 -*-
"""LOS PRECEPTOS QUE EL ESTUDIO CITA SIN TENERLOS — se traen del acervo si existen.

Revisión 322/2025: el estudio citó bien el artículo 210 del Código de
Procedimientos Civiles de Querétaro; el material traía el 211. El aviso decía
«no está en el material» y el artículo no se transcribía. Aquí: la detección
en datos, la traída con un Qdrant fingido, y que el aviso quede sólo para lo
que no existe.
"""
import asyncio, sys
import fase6_estudio as fe, fase6_rag as fr

fallos = []
def ok(c, nota):
    print(f"  {'OK ' if c else 'MAL'} {nota}")
    if not c: fallos.append(nota)

mat = fe.Material()
mat.normas = [{"cuerpo_legal": "Código de Procedimientos Civiles del Estado de Querétaro", "articulo": "211", "texto": "…"}]
est = ("En el ámbito local, el artículo 210 del Código de Procedimientos Civiles del Estado de Querétaro "
       "permite variar las medidas. El artículo 211 del mismo código regula el incidente.")
fuera, pares = fe.preceptos_fuera(est, mat)
ok(("código de procedimientos civiles del estado de querétaro", "210") in pares, "detecta el 210 como par (cuerpo, artículo)")
ok(not any("211" in x for x in fuera), "el 211 sí está en el material")

class _Pt:
    def __init__(s, pl): s.payload = pl
class QFalso:
    def __init__(s): s.llamadas = []
    def scroll(s, collection_name, scroll_filter, limit, with_payload):
        s.llamadas.append(collection_name)
        if collection_name == "leyes_queretaro":
            return ([_Pt({"cuerpo_legal_oficial": "Código de Procedimientos Civiles del Estado de Querétaro",
                          "articulo_num": 210, "texto": "Artículo 210. El juez podrá variar las medidas…", "chunk_index": 0}),
                     _Pt({"cuerpo_legal_oficial": "Código Civil del Estado de Querétaro",
                          "articulo_num": 210, "texto": "otra cosa", "chunk_index": 0})], None)
        return ([], None)

q = QFalso()
anad = asyncio.run(fr.completar_preceptos(q, mat, sorted(pares), "leyes_queretaro"))
ok(anad == ["art. 210 — Código de Procedimientos Civiles del Estado de Querétaro"], f"trae el 210 del código correcto: {anad}")
ok(any(str(n.get("articulo")) == "210" and "Procedimientos" in n["cuerpo_legal"] for n in mat.normas), "y queda en el material")
ok("Artículo 210. El juez" in [n for n in mat.normas if str(n.get("articulo")) == "210"][0]["texto"], "con su texto")
fuera2, _ = fe.preceptos_fuera(est, mat)
ok(not fuera2, "tras traerlo, la detección ya no acusa")
ok(asyncio.run(fr.completar_preceptos(None, mat, [("x", "1")], None)) == [], "sin Qdrant no hace nada")
ok(asyncio.run(fr.completar_preceptos(QFalso(), mat, [("Ley Inventada de Nada", "999")], "leyes_queretaro")) == [], "lo que no existe no se trae")

print()
if fallos: print(f"FALLAN {len(fallos)}: " + " · ".join(fallos)); sys.exit(1)
print("TODAS LAS COMPROBACIONES PASAN")
