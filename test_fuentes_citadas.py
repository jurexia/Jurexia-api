# -*- coding: utf-8 -*-
"""TODAS LAS FUENTES CITADAS Y VERIFICADAS — la ley correcta, la tesis con su ficha, el diálogo.

Revisión fiscal 61/2025 (David, 15-sep-2026): cuatro tesis de la Segunda Sala en
prosa sin ficha; el «artículo 134 del Código Fiscal de la Federación» traído
del Código Civil de Querétaro; «se obtiene d la jurisprudencia». Aquí se
garantiza: la identidad de la ley por fuero, la detección de leyes que el
material no trae, la traída de tesis por registro y clave, y el diálogo.
"""
import asyncio, sys
import fase6_rag as fr, fase6_estudio as fe, documento_generado as dg

fallos = []
def ok(c, nota):
    print(f"  {'OK ' if c else 'MAL'} {nota}")
    if not c: fallos.append(nota)

# ── la identidad de la ley ──
ok(fr.fuero_de("Código Fiscal de la Federación") == "federal", "CFF es federal")
ok(fr.fuero_de("Código Fiscal del Estado de Querétaro") == "estatal", "CF Querétaro es estatal")
ok(fr.fuero_de("Ley Federal de Procedimiento Contencioso Administrativo") == "federal", "LFPCA es federal")
ok(fr.fuero_de("Constitución Política de los Estados Unidos Mexicanos") == "federal", "la CPEUM es federal")
ok(not fr.misma_ley("Código Fiscal de la Federación", "Código Fiscal del Estado de Querétaro"), "CFF ≠ CF Querétaro")
ok(not fr.misma_ley("Código Fiscal del Estado de Querétaro", "Código Civil del Estado de Querétaro"), "CF Qro ≠ CC Qro")
ok(fr.misma_ley("Código Fiscal de la Federación", "Código Fiscal de la Federación"), "CFF = CFF")
ok(fr.misma_ley("Código de Procedimientos Civiles del Estado de Querétaro", "Código de Procedimientos Civiles del Estado de Querétaro"), "CPC Qro = CPC Qro")
ok(not fr.misma_ley("Código de Procedimientos Civiles del Estado de Querétaro", "Código de Procedimientos Civiles del Estado de Puebla"), "Qro ≠ Puebla")
ok(fr.misma_ley("Ley de Amparo", "Ley de Amparo, Reglamentaria de los artículos 103 y 107 de la Constitución Política de los Estados Unidos Mexicanos"), "el nombre corto casa con el oficial largo")
ok(not fr.misma_ley("Ley Federal de Procedimiento Administrativo", "Ley Federal de Procedimiento Contencioso Administrativo"), "LFPA ≠ LFPCA aunque una quepa dentro de la otra")
ok(not fr.misma_ley("Código Civil Federal", "Código Civil del Estado de Querétaro"), "civil federal ≠ civil de Querétaro")
mat_lfpa = fe.Material(); mat_lfpa.normas = [{"cuerpo_legal": "Ley Federal de Procedimiento Administrativo", "articulo": "92", "texto": "…"}]
_, pares_lfpa = fe.preceptos_fuera("El artículo 63 de la Ley Federal de Procedimiento Contencioso Administrativo regula el recurso.", mat_lfpa)
ok(pares_lfpa == {("ley federal de procedimiento contencioso administrativo", "63")}, f"la LFPCA citada no se da por la LFPA del material: {sorted(pares_lfpa)}")
mat_cff = fe.Material(); mat_cff.normas = [{"cuerpo_legal": "Código Fiscal de la Federación", "articulo": "137", "texto": "…"}]
_, pares_cff = fe.preceptos_fuera("Los artículos 134 y 137 del Código Fiscal de la Federación regulan la notificación.", mat_cff)
ok(pares_cff == {("código fiscal de la federación", "134")}, f"el 137 ya estaba; el 134 se pide con el nombre del material: {sorted(pares_cff)}")

# ── la detección de leyes que el material no trae ──
mat = fe.Material(); mat.normas = [{"cuerpo_legal": "Ley de Amparo", "articulo": "63", "texto": "…"}]
est = ("El artículo 134 del Código Fiscal de la Federación establece que los actos pueden notificarse; "
       "el artículo 63 de la Ley de Amparo regula el recurso; y el artículo 50 de la Ley Federal de "
       "Procedimiento Contencioso Administrativo fija los requisitos de la sentencia.")
fuera, pares = fe.preceptos_fuera(est, mat)
ok(("código fiscal de la federación", "134") in pares, f"detecta el CFF 134 aunque el material no traiga el CFF: {sorted(pares)}")
ok(("ley federal de procedimiento contencioso administrativo", "50") in pares, f"y la LFPCA 50, con el nombre limpio: {sorted(pares)}")
ok(not any(a == "63" for c, a in pares), "el 63 de Amparo sí estaba")
est2 = ("Los artículos 134 y 137 del Código Fiscal de la Federación regulan la notificación; el artículo 6º de la "
        "Ley Federal de Procedimiento Contencioso Administrativo prevé las costas; y el artículo 68 del Código "
        "Fiscal de la Federación y 42 de la Ley Federal de Procedimiento Contencioso Administrativo presumen la legalidad.")
_, pares2 = fe.preceptos_fuera(est2, mat)
ok({("código fiscal de la federación", "134"), ("código fiscal de la federación", "137")} <= pares2, f"«artículos 134 y 137 del CFF» son dos citas: {sorted(pares2)}")
ok(("ley federal de procedimiento contencioso administrativo", "6") in pares2, "«artículo 6º» lleva el ordinal pegado y se lee")
ok(("código fiscal de la federación", "68") in pares2 and ("ley federal de procedimiento contencioso administrativo", "42") in pares2, "«68 del CFF y 42 de la LFPCA» son dos leyes")
normas_x = [{"cuerpo_legal": "Código Fiscal de la Federación", "articulo": "137", "texto": "Artículo 137. Cuando la notificación…"},
            {"cuerpo_legal": "Código Fiscal de la Federación", "articulo": "134", "texto": "Artículo 134. Las notificaciones…"}]
ok([n_ for n_, _ in dg._preceptos_del_parrafo("Los artículos 134 y 137 del Código Fiscal de la Federación regulan…", normas_x)] == ["134", "137"], "el compositor también reparte la lista")

# ── la traída, con la ley correcta y por fuero ──
class _Pt:
    def __init__(s, pl): s.payload = pl
class QFalso:
    def __init__(s): s.llamadas = []
    def scroll(s, collection_name, scroll_filter, limit, offset=None, with_payload=True, with_vectors=False):
        s.llamadas.append(collection_name)
        num = scroll_filter.must[0].match.value
        if collection_name == "leyes_queretaro":
            return ([_Pt({"cuerpo_legal_oficial": "Código Civil del Estado de Querétaro", "articulo_num": num, "texto": "civil qro", "chunk_index": 0}),
                     _Pt({"cuerpo_legal_oficial": "Código Fiscal del Estado de Querétaro", "articulo_num": num, "texto": "fiscal qro", "chunk_index": 0})], None)
        if collection_name in ("leyes_administrativa", "leyes_federales"):
            return ([_Pt({"cuerpo_legal_oficial": "Ley General de Salud", "articulo_num": num, "texto": "salud", "chunk_index": 0}),
                     _Pt({"cuerpo_legal_oficial": "Código Fiscal de la Federación", "articulo_num": num, "texto": f"Artículo {num}. Las notificaciones…", "chunk_index": 0})], None)
        if collection_name == "jurisprudencia_nacional_v3":
            k = scroll_filter.must[0].key; v = scroll_filter.must[0].match.value
            if (k, v) in (("registro", "171707"),):
                return ([_Pt({"registro": "171707", "clave_tesis": "2a./J. 158/2007", "rubro": "NOTIFICACIÓN FISCAL DE CARÁCTER PERSONAL…", "instancia": "Segunda Sala", "tipo": "Jurisprudencia", "texto": "t"})], None)
            if k == "clave_tesis" and v.upper().replace(" ", "") == "2A./J.60/2007":
                return ([_Pt({"registro": "172470", "clave_tesis": "2a./J. 60/2007", "rubro": "NOTIFICACIÓN PERSONAL. EN LA PRACTICADA…", "instancia": "Segunda Sala", "tipo": "Jurisprudencia", "texto": "t"})], None)
            return ([], None)
        return ([], None)

q = QFalso(); mat2 = fe.Material(); mat2.normas = []
anad = asyncio.run(fr.completar_preceptos(q, mat2, [("Código Fiscal de la Federación", "134")], "leyes_queretaro", materia="administrativa", tipo_asunto="revision_fiscal"))
ok(anad == ["art. 134 — Código Fiscal de la Federación"], f"trae el CFF 134 del silo, no el civil de Querétaro: {anad}")
ok("leyes_queretaro" not in q.llamadas, "en revisión fiscal no toca la colección estatal")
mat3 = fe.Material(); mat3.normas = []
q2 = QFalso()
anad2 = asyncio.run(fr.completar_preceptos(q2, mat3, [("Código Fiscal del Estado de Querétaro", "159")], "leyes_queretaro", materia="administrativa"))
ok(anad2 == ["art. 159 — Código Fiscal del Estado de Querétaro"], f"una ley estatal se trae del estado y con su nombre exacto: {anad2}")

# ── las tesis por registro y por clave ──
mat4 = fe.Material(); mat4.tesis = []; q3 = QFalso()
nuevas = asyncio.run(fr.completar_tesis_citadas(q3, mat4, ["171707", "2a./J. 60/2007", "999999"]))
ok(len(nuevas) == 2 and {t["registro"] for t in mat4.tesis} == {"171707", "172470"}, f"trae por registro y por clave; lo que no existe no entra: {nuevas}")
ok(all(t.get("citada_por_la_parte") for t in mat4.tesis), "y quedan marcadas como citadas")

# ── el compositor ──
ok(dg._verbo_de_enlace("La misma conclusión se obtiene de la jurisprudencia de la Primera Sala, de") == "Sirve de apoyo", "un arranque que no es fórmula vuelve a «Sirve de apoyo»")
ok(dg._verbo_de_enlace("Resulta aplicable, en calidad de criterio orientador, la tesis") == "Resulta aplicable", "las fórmulas del oficio se conservan")
tesis = [{"registro": "2007413", "rubro": "NOTIFICACIÓN PERSONAL EN MATERIA FISCAL. PARA CIRCUNSTANCIAR EL ACTA DE LA DILIGENCIA ENTENDIDA CON UN TERCERO, ES INNECESARIO…", "tipo": "Jurisprudencia", "instancia": "Segunda Sala", "obligatoria": True, "texto": "x " * 100}]
h, m = dg.tesis_del_rubro("La ausencia se explica en el criterio de la Segunda Sala, de rubro «NOTIFICACIÓN PERSONAL EN MATERIA FISCAL. PARA CIRCUNSTANCIAR EL ACTA DE LA DILIGENCIA CON UN TERCERO, ES INNECESARIO QUE EL NOTIFICADOR RECABE DOCUMENTOS», registro 2007413.", tesis)
ok(h is not None and h["registro"] == "2007413", "un rubro parafraseado se reconoce por su registro")
ok(dg._desencadenar("y la tesis con registro 2007413, de rubro «NOTIFICACIÓN PERSONAL EN MATERIA FISCAL. PARA CIRCUNSTANCIAR EL ACTA DE LA DILIGENCIA»", tesis).startswith("También sirve de apoyo"), "«y la tesis con registro…» se desencadena por registro")
import docx
d = docx.Document(); notas = []
dg._escribir_estudio(d, ["y la tesis con registro 2007413, de rubro «NOTIFICACIÓN PERSONAL EN MATERIA FISCAL. PARA CIRCUNSTANCIAR EL ACTA DE LA DILIGENCIA ENTENDIDA CON UN TERCERO, ES INNECESARIO…», que resulta aplicable.",
                        "y ello conduce a desestimar el planteamiento de la recurrente."], tesis, notas)
ps = [p.text for p in d.paragraphs if p.text.strip()]
ok(any(p.startswith("También sirve de apoyo") for p in ps), f"un párrafo que arranca con «y la tesis…» sale como anuncio: {ps[:2]}")
ok(not any(p.lstrip().startswith("y ") for p in ps) and any(p.startswith("Asimismo,") for p in ps), "ningún párrafo del estudio arranca con «y …»")
ok(any("registro digital 2007413" in n for n in notas), "y la tesis baja con su ficha al pie")

print()
if fallos: print(f"FALLAN {len(fallos)}: " + " · ".join(fallos)); sys.exit(1)
print("TODAS LAS COMPROBACIONES PASAN")
