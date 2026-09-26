"""Las citas agrupadas — 26-sep-2026.

    .venv/bin/python test_citas_plurales.py

Sin red y sin gastar API. El caso: a «Traza la línea cronológica desde el
nacimiento del control de convencionalidad hasta la postura actual de la
SCJN…» el modelo escribió 31 citas y agrupó varias en PLURAL,
«[Doc IDs: da1de55e-…; 2b2dc535-…]». Los 7 ids que sólo aparecían así eran
REALES y estaban en el contexto (Radilla ¶340 y ¶341, García Rodríguez ¶301,
¶303 y el resolutivo 14, y dos tesis de la v3), pero el validador sólo leía
el singular: faltaron en CITATION_META.sources, el sello las contó como «no
verificadas» y en pantalla quedó «[Doc IDs: [25]; [26]]» sin PDF que abrir.

Se comprueba: que las 7 entran a `sources` (con la conversación real si está
en el scratchpad; si no, con un fragmento del mismo texto), que todas las
formas —«[Doc IDs: a; b]», «[Doc ID: a; b]», «[Doc ID: a, b]»,
«[Doc ID: a; Doc ID: b]», «(Doc ID: a)», cualquier caja— se leen como
singulares, que el singular sigue igual, que un id inexistente sigue
marcándose inválido, que ninguna forma rara rompe el sello, la reparación
de ids dentro de un grupo, el historial y la regla del prompt.

Y que ninguna expresión se cuelga (sección 9): la primera versión tardaba
22 s con «(véase Doc ID: 0123abcd» y veinte grupos «-0123456789abcdef» sin
cierre, y el tiempo se duplicaba con cada grupo. Cada expresión del bloque
de las citas, y cada función que las usa, se mide con cadenas patológicas
de 10,000 caracteres: menos de 50 ms cada una.

La sección 10 es la garantía de tiempo lineal por construcción: ninguna
expresión del bloque tiene un cuantificador sin tope (se recorre su árbol),
y una prueba de propiedad con fuzz de semilla fija —500 cadenas de 20,000
caracteres y 20 de 100,000 con un alfabeto adversario, más las cargas de
las revisiones— exige a cada función pública de las citas menos de 60 ms y
de 300 ms, y que con 100,000 no tarde más de ~6 veces lo que con 20,000.
La sección 11 comprueba que el caso real sale igual que con 8029c97.
"""
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, os.getcwd())
import main  # noqa: E402
import linea_coidh as lc  # noqa: E402
import documento_acervo as da  # noqa: E402

FALLOS = []


def ok(cond, que):
    print(f"   {'PASA ' if cond else 'FALLA'}  {que}")
    if not cond:
        FALLOS.append(que)


# La conversación real, si la sesión que la bajó sigue viva. La prueba no
# depende de ella: el fragmento de abajo es el mismo texto, recortado.
CASO = Path(os.getenv("CITAS26_CONVERSACION") or (
    "/private/tmp/claude-501/-Users-josedavidalcantarmendoza-Documents-IUREXIA-MAC-jurexia-api-git"
    "--claude-worktrees-xenodochial-poincare-fb5468/ef3b257b-8e89-453a-8ee1-4c9ffa7ff848/scratchpad"
    "/citas26/conversacion.json"))

# Los 7 que faltaron en sources, con lo que eran.
FALTAN = {
    "da1de55e-52d9-de76-8001-92f4bd4c0424": ("coidh", "Radilla Pacheco Vs. México", "Párr. 340", None),
    "2b2dc535-08b7-f9e7-6884-42d87f9088b0": ("coidh", "Radilla Pacheco Vs. México", "Párr. 341", None),
    "42e42c82-8bdc-1c9f-da76-da8856c477a5": ("coidh", "García Rodríguez y otro Vs. México", "Párr. 301", None),
    "fcd8d6c8-b56e-2488-6196-5bc587ad9e37": ("coidh", "García Rodríguez y otro Vs. México", "Párr. 303", None),
    "a431aca8-1896-c694-1d7d-4bc75d24a453": ("coidh", "García Rodríguez y otro Vs. México", "Punto resolutivo 14", None),
    "81e7710c-d16f-58cf-9753-7fc0b19c09e7": ("jurisprudencia_nacional_v3", "SCJN", "Tesis", "2005115"),
    "0b9477ff-9b7b-571c-8a57-227f8683013f": ("jurisprudencia_nacional_v3", "SCJN", "Tesis", "2010959"),
}

# El mismo texto del caso, recortado a los párrafos con citas agrupadas (y
# las singulares que los rodean). Es la respuesta del modelo, sin tocar.
FRAGMENTO = """\
En *Radilla Pacheco Vs. México*, la Corte IDH reiteró que el Poder Judicial debía ejercer control de convencionalidad *ex officio*. [Corte IDH, *Radilla Pacheco Vs. México*, párr. 339.] [Doc ID: 06b7f119-2b54-c948-fbe5-ba31b29fceea] [Corte IDH, *Radilla Pacheco Vs. México*, párrs. 340-341.] [Doc IDs: da1de55e-52d9-de76-8001-92f4bd4c0424; 2b2dc535-08b7-f9e7-6884-42d87f9088b0]

*Punto resolutivo 8.* [Doc ID: 64663123-5802-2995-b765-7b0e83779ebd]

En *García Rodríguez y otro Vs. México*, la Corte IDH volvió sobre el problema y ordenó adecuar el derecho interno, incluidas sus disposiciones constitucionales. *Corte IDH. Caso García Rodríguez y otro Vs. México. Sentencia de 25 de enero de 2023. Serie C No. 482, párrs. 301 y 303.* [Doc IDs: 42e42c82-8bdc-1c9f-da76-da8856c477a5; fcd8d6c8-b56e-2488-6196-5bc587ad9e37]

Esos extractos deben tratarse como criterios de orientación. [Doc IDs: c84141ee-2f66-5a10-ad91-7f4f6fd5694a; 312e8360-4826-5c5d-b7ca-f1536d6e870d]

En *Tzompaxtle* y *García Rodríguez*, la Corte ordenó adecuaciones internas; en *García Rodríguez* incluyó expresamente las disposiciones constitucionales. [Doc IDs: 1b1da64c-a350-e480-d1e2-84aa6fd9e85c; 64663123-5802-2995-b765-7b0e83779ebd; 42e42c82-8bdc-1c9f-da76-da8856c477a5; a431aca8-1896-c694-1d7d-4bc75d24a453]

La jurisprudencia de la Primera Sala sobre la prolongación de la prisión preventiva también destaca esos factores. [Doc ID: e63f92ad-a3f0-583e-a7be-001a2b95956d]

La inaplicación no se sigue automáticamente de invocar el principio pro persona. [Doc IDs: 81e7710c-d16f-58cf-9753-7fc0b19c09e7; 0b9477ff-9b7b-571c-8a57-227f8683013f]
"""


def sr(i, silo="leyes_federales", origen="Fuente", ref="", registro=None):
    return main.SearchResult(id=i, score=0.9, texto=f"texto de {ref or i}", ref=ref, origen=origen,
                             silo=silo, registro=registro)


def contexto(ids_extra=()):
    """El doc_id_map del caso: las 7 que faltaron con su silo, y el resto."""
    docs = [sr(i, silo=s, origen=o, ref=r, registro=g) for i, (s, o, r, g) in FALTAN.items()]
    docs += [sr(i) for i in ids_extra if i not in FALTAN]
    return docs


def meta_de(salida):
    for linea in salida:
        m = re.search(r"<!-- CITATION_META:(\{.*\}) -->", linea, re.S)
        if m:
            return json.loads(m.group(1))
    return None


# ═══════════════════════════════════════════════ 1. el caso real (o su fragmento)
print("\n── 1. el caso del 26-sep: las 7 agrupadas entran a sources ──")
texto, fuente_caso, sources_antes = FRAGMENTO, "fragmento incrustado", None
if CASO.exists():
    try:
        _c = json.loads(CASO.read_text(encoding="utf-8"))[1]["content"]
        _m = re.search(r"<!-- CITATION_META:(.*?) -->", _c, re.S)
        sources_antes = set(json.loads(_m.group(1))["sources"]) if _m else None
        texto, fuente_caso = main._limpiar_marcadores(_c), "conversación real"
    except Exception as e:  # la prueba no depende del scratchpad
        print(f"   (no pude leer la conversación real: {type(e).__name__}; sigo con el fragmento)")
print(f"   usando: {fuente_caso} ({len(texto):,} caracteres)")

ids_en_texto = sorted(set(re.findall(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", texto)))
docs = contexto(ids_en_texto)
mapa = main.build_doc_id_map(docs)

antes = set(main.DOC_ID_PATTERN.findall(texto))
ok(not (set(FALTAN) & antes), "ANTES: el patrón singular no veía ninguna de las 7 (así faltaron)")
if sources_antes is not None:
    ok(not (set(FALTAN) & sources_antes) and len(sources_antes) == 60,
       f"y en el CITATION_META guardado faltan justo ésas ({len(sources_antes)} fuentes)")

citados = set(main.extract_doc_ids(texto))
ok(set(FALTAN) <= citados, f"extract_doc_ids ve las 7 ({len(set(FALTAN) & citados)}/7)")
ok(citados == set(ids_en_texto), f"y ve todos los ids del texto, ni uno más ({len(citados)} de {len(ids_en_texto)})")

val = main.validate_citations(texto, mapa)
ok(val.invalid_count == 0 and val.valid_count == val.total_citations == len(ids_en_texto),
   f"validate_citations: {val.valid_count} válidas de {val.total_citations}, {val.invalid_count} inválidas")
ok(set(FALTAN) <= {c.doc_id for c in val.citations if c.status == "valid"}, "las 7 salen válidas")

sello = main._marcadores_del_sello(texto, mapa, docs)
meta = meta_de(sello)
ok(meta is not None and set(FALTAN) <= set(meta["sources"]),
   f"CITATION_META.sources trae las 7 ({len(set(FALTAN) & set((meta or {}).get('sources', {})))}/7)")
_coidh = [meta["sources"][i] for i in FALTAN if FALTAN[i][0] == "coidh"] if meta else []
ok(len(_coidh) == 5 and all(s.get("silo") == "coidh" for s in _coidh),
   "las 5 de la Corte IDH van con su silo (el visor abre su PDF)")
_v3 = [meta["sources"][i] for i in FALTAN if FALTAN[i][0] != "coidh"] if meta else []
ok(sorted(str(s.get("registro")) for s in _v3) == ["2005115", "2010959"], "las 2 de la v3 van con su registro")

canon = main.repair_hallucinated_uuids(texto, mapa)
ok("Doc IDs" not in canon and not re.search(r"\[Doc ID:[^\]]*[;,]", canon),
   "el texto reparado ya no tiene «Doc IDs» ni ids agrupados en unos corchetes")
ok(len(re.findall(r"\[Doc ID: ", canon)) == len(re.findall(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", texto)),
   "cada id del original queda en sus propios corchetes (una cita por id)")
ok(main.expandir_citas_doc_id(canon) == canon, "canonizar dos veces no cambia nada (idempotente)")

# ═══════════════════════════════════════════════ 2. todas las formas
print("\n── 2. todas las formas agrupadas o sueltas se leen como singulares ──")
A, B = "da1de55e-52d9-de76-8001-92f4bd4c0424", "2b2dc535-08b7-f9e7-6884-42d87f9088b0"
C = "42e42c82-8bdc-1c9f-da76-da8856c477a5"
DOS = f"[Doc ID: {A}] [Doc ID: {B}]"
FORMAS = {
    f"[Doc IDs: {A}; {B}]": DOS,
    f"[Doc ID: {A}; {B}]": DOS,
    f"[Doc ID: {A}, {B}]": DOS,
    f"[Doc ID: {A}; Doc ID: {B}]": DOS,
    f"[Doc IDs: {A} y {B}]": DOS,
    f"[doc ids: {A}; {B}]": DOS,
    f"[DOC ID: {A}, {B}]": DOS,
    f"[DocIDs: {A}; {B}]": DOS,
    f"[Doc IDs:\n{A};\n{B}]": DOS,
    f"[Doc IDs: {A}; {B}; {C}]": DOS + f" [Doc ID: {C}]",
    f"[Doc IDs: {A}; {A}]": f"[Doc ID: {A}]",
    f"(Doc ID: {A})": f"[Doc ID: {A}]",
    f"(doc id: {A})": f"[Doc ID: {A}]",
    f"(Doc IDs: {A}; {B})": DOS,
    f"[doc id: {A}]": f"[Doc ID: {A}]",
    f"[Doc ID:{A}]": f"[Doc ID: {A}]",
    f"[Doc ID: **{A}**]": f"[Doc ID: {A}]",
    f"Doc IDs: {A}; {B}.": DOS + ".",
    f"(Tesis 2a./J. 5/2020, Doc ID: {A})": f"(Tesis 2a./J. 5/2020) [Doc ID: {A}]",
    f"[Registro 2005115; Doc IDs: {A}; {B}]": f"[Registro 2005115] {DOS}",
    f"[Doc ID: {A}, párr. 340]": f"[Doc ID: {A}] (párr. 340)",
    # Lo que señaló la revisión del 26-sep: un id recortado con «-…» dentro
    # de un grupo, las conjunciones del resto, más de quince ids, la
    # etiqueta con énfasis y «Doc. ID».
    f"[Doc IDs: {A}; 2b2dc535-…]": f"[Doc ID: {A}] [Doc ID: 2b2dc535-…]",
    f"[Doc IDs: {A}; da1de55e-52d9-…]": f"[Doc ID: {A}] [Doc ID: da1de55e-52d9-…]",
    f"[Doc IDs: {A}; 2b2dc535-...]": f"[Doc ID: {A}] [Doc ID: 2b2dc535-...]",
    f"[Doc ID: {A}, párr. 340 y 341]": f"[Doc ID: {A}] (párr. 340 y 341)",
    f"[Doc ID: {A}, párrs. 340 o 341 e interpretación]": f"[Doc ID: {A}] (párrs. 340 o 341 e interpretación)",
    f"[Doc IDs: {A}, y {B}]": DOS,
    f"[Doc IDs: {A}; párr. 3 y {B}]": f"{DOS} (párr. 3)",
    f"[Doc IDs: {'; '.join([A, B, C] * 6)}]": DOS + f" [Doc ID: {C}]",
    f"[**Doc IDs:** {A}; {B}]": DOS,
    f"[*Doc IDs*: {A}; {B}]": DOS,
    f"[**Doc ID:** {A}]": f"[Doc ID: {A}]",
    f"[Doc. ID: {A}]": f"[Doc ID: {A}]",
    f"(Doc. IDs: {A}; {B})": DOS,
    f"**Doc IDs:** {A}; {B}.": DOS + ".",
    f"(Tesis 2a./J. 5/2020, **Doc ID:** {A})": f"(Tesis 2a./J. 5/2020) [Doc ID: {A}]",
    f"Doc IDs: {A} o {B}.": DOS + ".",
    # El énfasis que es de la cita entera se queda donde estaba: sin un «**»
    # huérfano que ponga en negritas el resto del párrafo.
    f"**Doc ID: {A}**": f"**[Doc ID: {A}]**",
    # La revisión de 8029c97: el énfasis en la etiqueta Y en el id dejaba
    # «[Doc ID: **a]» (no idempotente; el sello lo contaba inválido), y la
    # conjunción junto a un id con adorno dejaba «(y)» o «(y ** **, …)».
    f"[**Doc ID:** **{A}**]": f"[Doc ID: {A}]",
    f"(**Doc ID:** **{A}**)": f"[Doc ID: {A}]",
    f"[**Doc ID:** `{A}`]": f"[Doc ID: {A}]",
    f"(**Doc IDs:** **{A}**; **{B}**)": DOS,
    f"[Doc IDs: **{A}** y **{B}**]": DOS,
    f"[Doc IDs: `{A}` y **{B}**, Tesis 2a./J. 5/2020]": f"{DOS} (Tesis 2a./J. 5/2020)",
    f"[Doc IDs: **{A}**, **{B}**, párr. 3 y 4]": f"{DOS} (párr. 3 y 4)",
}
for crudo, esperado in FORMAS.items():
    sale = main.expandir_citas_doc_id(f"Frase. {crudo} Sigue.")
    ok(sale == f"Frase. {esperado} Sigue.", f"{crudo[:34]!r:40} → {sale[7:-7][:60]!r}")
    ok(set(main.extract_doc_ids(crudo)) == set(re.findall(r"\[Doc ID: ([^\]]+)\]", esperado)),
       f"   y extract_doc_ids lee sus ids")
    ok(main.expandir_citas_doc_id(sale) == sale, "   e idempotente")

# ═══════════════════════════════════════════════ 3. el singular sigue igual
print("\n── 3. el singular de siempre no cambia ──")
SINGULAR = (f"El artículo 1 establece… [Doc ID: {A}]. Asimismo… [Doc ID: {B}]\n"
            f"> \"texto\" -- *Artículo 19, Constitución* [Doc ID: {C}]\n"
            "Un placeholder que el modelo copió: [Doc ID: uuid]. Y la etiqueta en prosa: sin Doc ID no hay cita.\n"
            "Un id cortado: [Doc ID: 779835a0…]. Una referencia numérica [3] y un enlace [x](http://a.b/c).")
ok(main.expandir_citas_doc_id(SINGULAR) == SINGULAR, "un texto ya canónico sale byte por byte igual")
ok(set(main.extract_doc_ids(SINGULAR)) == set(main.DOC_ID_PATTERN.findall(SINGULAR)),
   "y extract_doc_ids lee lo mismo que el patrón de siempre")
ok(main.expandir_citas_doc_id("") == "" and main.extract_doc_ids("") == [] and main.extract_doc_ids(None) == [],
   "vacío y None no truenan")

# ═══════════════════════════════════════════════ 4. lo inexistente sigue inválido
print("\n── 4. un id que no estaba en el contexto sigue marcándose inválido ──")
FALSO = "0f0f0f0f-1234-4abc-9def-00000000beef"
m1 = main.build_doc_id_map([sr(A), sr(B)])
v = main.validate_citations(f"Algo. [Doc IDs: {A}; {FALSO}]", m1)
ok(v.valid_count == 1 and v.invalid_count == 1 and [c.doc_id for c in v.citations if c.status == "invalid"] == [FALSO],
   "en un grupo, el real sale válido y el inventado inválido")
v = main.validate_citations(f"Algo. (Doc ID: {FALSO})", m1)
ok(v.invalid_count == 1 and v.total_citations == 1, "entre paréntesis, también se cuenta y se marca")
rep = main.repair_hallucinated_uuids(f"Algo. [Doc IDs: {A}; {FALSO}]", m1)
ok(rep == f"Algo. [Doc ID: {A}] [Doc ID: {FALSO}]", "la reparación no inventa un dueño para el inventado")
for _adornada in (f"[**Doc ID:** **{A}**]", f"(**Doc ID:** **{A}**)", f"[**Doc ID:** `{A}`]"):
    # /analyze-document y /chat-sentencia validan sin reparar: la cita con
    # énfasis en la etiqueta y en el id tiene que contar como la que es.
    v = main.validate_citations(f"Algo. {_adornada}", m1)
    ok(v.valid_count == 1 and v.invalid_count == 0, f"{_adornada[:16]}… valida sin reparar (sello de /analyze-document)")
meta4 = meta_de(main._marcadores_del_sello(f"Algo. [Doc IDs: {A}; {FALSO}]", m1, [sr(A), sr(B)]))
ok(meta4 and meta4["invalid_ids"] == [FALSO] and meta4["sources"][FALSO]["origen"] == "Fuente no verificada"
   and meta4["sources"][A]["origen"] != "Fuente no verificada", "y el sello lo dice así")

# ═══════════════════════════════════════════════ 5. la reparación dentro de un grupo
print("\n── 5. un id estropeado dentro de un grupo se repara ──")
B_ROTO = B[:10] + B[11:]                      # se le cayó un carácter
rep = main.repair_hallucinated_uuids(f"Algo. [Doc IDs: {A}; {B_ROTO}]", m1)
ok(rep == f"Algo. {DOS}", f"«{B_ROTO[:14]}…» vuelve a ser {B[:14]}…")
v = main.validate_citations(rep, m1)
ok(v.valid_count == 2 and v.invalid_count == 0, "y las dos validan")
rep = main.repair_hallucinated_uuids(f"Algo. [Doc IDs: {A}; 2b2dc535-…]", m1)
ok(rep == f"Algo. {DOS}", "«2b2dc535-…» (recortado tras el guion) dentro de un grupo también se repara")
rep = main.repair_hallucinated_uuids(f"Algo. [Doc IDs: {B}; da1de55e-52d9-…]", m1)
ok(rep == f"Algo. [Doc ID: {B}] [Doc ID: {A}]", "y «da1de55e-52d9-…», sin dejar «(…)» colgando")
_rep_map = {}
rep = main.repair_hallucinated_uuids(f"Algo [Doc ID: {B_ROTO}] y otra vez [Doc ID: {B_ROTO}] [Doc ID: {FALSO}]",
                                     m1, reparados=_rep_map)
ok(rep == f"Algo [Doc ID: {B}] y otra vez [Doc ID: {B}] [Doc ID: {FALSO}]" and _rep_map == {B_ROTO: B},
   "`reparados` anota qué id se reparó a cuál (lo usa /chat para su CITATION_META), una vez por id")
_muchos = " ".join(f"[Doc ID: {i:08x}-0000-0000-0000-000000000000]" for i in range(main._REPARACIONES_MAXIMAS + 10))
_rep_map = {}
main.repair_hallucinated_uuids(_muchos + f" [Doc ID: {B_ROTO}]", m1, reparados=_rep_map)
ok(B_ROTO not in _rep_map, f"pasado el tope de {main._REPARACIONES_MAXIMAS} ids distintos fuera del contexto "
   "no se intenta reparar más (no son dedazos)")

# ═══════════════════════════════════════════════ 6. ninguna forma rara rompe el sello
print("\n── 6. ninguna forma rara rompe el sello ──")
RARAS = [
    "[Doc IDs: ]", "[Doc ID: ;;]", "(Doc ID)", "[Doc IDs]", f"[Doc IDs: {A}; {B}",   # sin cerrar
    f"[Doc IDs: [25]; [26]]", f"[Doc ID: [{A}]]", "[Doc ID: no disponible]", "[Doc IDs: uuid; uuid]",
    f"[[Doc IDs: {A}; {B}]]", f"((Doc ID: {A}))", "Doc ID:", "Doc IDs: ;", f"[Doc ID: {A}) y [Doc ID: {B}]",
    "[" * 500 + "Doc ID: " + "(" * 500, "Doc ID " * 2000, f"[Doc IDs: {'; '.join([A] * 300)}]",
    "<!-- CITATION_META:{} -->", "\x00[Doc ID: \x00]", "[Doc ID: ✓]", "(Doc ID: 𝒶)",
]
rotas = []
for raro in RARAS:
    try:
        t = f"Texto. {raro} Fin."
        main.expandir_citas_doc_id(t)
        main.extract_doc_ids(t)
        main.repair_hallucinated_uuids(t, m1)
        main.validate_citations(t, m1)
        if meta_de(main._marcadores_del_sello(t, m1, [sr(A), sr(B)])) is None:
            rotas.append(f"{raro[:30]!r}: sin CITATION_META")
    except Exception as e:
        rotas.append(f"{raro[:30]!r}: {type(e).__name__}: {e}")
ok(not rotas, f"{len(RARAS)} formas raras pasan por expandir, reparar, validar y sellar sin romperse"
   + (f" — {rotas[:3]}" if rotas else ""))
import time as _t
_t0 = _t.perf_counter()
_grande = (texto + "\n") * 8
main.expandir_citas_doc_id(_grande)
_seg = _t.perf_counter() - _t0
ok(_seg < 1.0, f"canonizar {len(_grande):,} caracteres tarda {_seg * 1000:.0f} ms (< 1 s)")

# ═══════════════════════════════════════════════ 6b. los modelos de estilo
print("\n── 6b. los modelos de estilo no enseñan ids, en ninguna forma ──")
_estilo = main._sanitize_style_example(
    f"El quejoso alega. [Doc IDs: {A}; {B}] Se estima fundado (Doc ID: {A}). "
    f"Así lo sostuvo [**Doc IDs:** {A}; {B}] la Sala (Tesis X, Doc ID: {C}) y "
    f"Doc IDs: {A}; {B}. Consta [Doc ID: {A}, párr. 3] y [Doc. ID: {B}]. "
    f"También [Doc ID: {A} (párr. 3)] y [Doc ID: {A}, véase [nota]]. Fin.")
ok("doc" not in _estilo.lower() and not re.search(r"[0-9a-f]{8}-[0-9a-f]{4}", _estilo)
   and "El quejoso alega" in _estilo and "la Sala" in _estilo and _estilo.endswith("Fin."),
   f"_sanitize_style_example las quita todas y deja el texto: {_estilo[:90]!r}")
ok("párr" not in _estilo and "nota" not in _estilo and "]" not in _estilo and "(" not in _estilo,
   "y se lleva enteras las que traen paréntesis o corchetes dentro: «[Doc ID: a (párr. 3)]», «[Doc ID: a, véase [nota]]»")

# ═══════════════════════════════════════════════ 7. el historial vuelve en singular
print("\n── 7. el historial que vuelve al modelo va en singular ──")
hist = [main.Message(role="user", content=f"¿Y esto? [Doc IDs: {A}; {B}]"),
        main.Message(role="assistant", content=f"Así. [Doc IDs: {A}; {B}]")]
limpio = main._limpiar_historial(hist)
ok(limpio[1].content == f"Así. {DOS}", "la respuesta guardada con «Doc IDs» vuelve canonizada")
ok(limpio[0].content == hist[0].content, "lo que escribió el abogado no se toca")
_ya = [main.Message(role="assistant", content=f"Así. {DOS}")]
ok(main._limpiar_historial(_ya) is _ya, "un historial limpio no paga copia")
_tope_antes = main.HISTORIAL_CITAS_MAX_CHARS
try:
    main.HISTORIAL_CITAS_MAX_CHARS = 5000
    _parrafo = f"Así. [Doc IDs: {A}; {B}]\n"
    _largo = _parrafo * 400                                   # 32,800 caracteres
    _hecho = main._limpiar_historial([main.Message(role="assistant", content=_largo)])[0].content
    _corte = _largo.rfind("\n", 0, 5000)
    ok(_hecho == main.expandir_citas_doc_id(_largo[:_corte]) + _largo[_corte:]
       and _hecho.count("[Doc IDs: ") == _largo[_corte:].count("[Doc IDs: ") > 0,
       "un turno del historial se canoniza hasta HISTORIAL_CITAS_MAX_CHARS y el resto sigue tal cual")
    ok(_largo[_corte] == "\n" and _corte > 5000 - len(_parrafo),
       "y el corte cae en el último salto de línea antes del tope: ninguna cita queda partida")
finally:
    main.HISTORIAL_CITAS_MAX_CHARS = _tope_antes

# ═══════════════════════════════════════════════ 8. el cableado y el prompt
print("\n── 8. el cableado y la regla del prompt ──")
FUENTE = Path("main.py").read_text(encoding="utf-8")
_i = FUENTE.index("content_buffer = expandir_citas_doc_id(content_buffer)")
ok(_i < FUENTE.index("uuid_repair_map: Dict[str, str] = {}", _i)
   < FUENTE.index("validation = validate_citations(content_buffer, doc_id_map)", _i)
   < FUENTE.index("_RE_PAR.finditer(content_buffer", _i),
   "/chat canoniza el búfer antes de la reparación, la validación y el sello de correspondencia")
ok("content_buffer, doc_id_map, reparados=uuid_repair_map)" in FUENTE
   and 'repair_hallucinated_uuids(\n                                f"[Doc ID: {cited_id}]"' not in FUENTE,
   "/chat repara el búfer en una sola pasada que llena uuid_repair_map (antes, id por id y luego otra vez)")
ok("enhanced_text = expandir_citas_doc_id(enhanced_text)" in FUENTE,
   "/enhance devuelve el texto entero ya canónico")
ok("matches = DOC_ID_PATTERN.findall(expandir_citas_doc_id(" in FUENTE,
   "extract_doc_ids (validador, sello, /analyze-document, /chat-sentencia) lee las agrupadas")
_viejos = len(re.findall(r"""re\.compile\(r['"]\\\[Doc ID:""", FUENTE))
ok(_viejos == 1, f"ninguna otra expresión busca «[Doc ID:» con el patrón singular ({_viejos}: DOC_ID_PATTERN)")
_n_directos = len(re.findall(r"DOC_ID_PATTERN\.(?:findall|finditer)\(", FUENTE))
ok(_n_directos == 1, f"nadie más lee citas con el patrón singular a pelo ({_n_directos} lectura: la de extract_doc_ids)")
_pm = main.SYSTEM_PROMPT_CHAT
ok('NUNCA agrupes varios ids en unos corchetes ni escribas "Doc IDs"' in _pm
   and "NUNCA coloques multiples [Doc ID] consecutivos" not in _pm,
   "el prompt maestro pide un [Doc ID] por fuente y ya no prohíbe ponerlos seguidos (eso empujaba a agruparlos)")
ok("un [Doc ID] por párrafo, cada uno en sus corchetes (nunca «Doc IDs»)" in lc._INSTRUCCION_CITA,
   "_INSTRUCCION_CITA (<casos_corte_idh> y la línea de la Corte IDH) lo dice")
ok("nunca «Doc IDs»" in lc._INSTRUCCION_SOLO_MX, "la línea sólo-México también")
ok(FUENTE.count("(uno por fragmento, en sus corchetes; nunca «Doc IDs»)") == 2, "los dos bloques de doctrina también")
ok("nunca «[Doc IDs: a; b]»" in da._CON_ACERVO, "y el análisis de documentos con acervo")
ok('NUNCA agrupes varios ids ni escribas "Doc IDs"' in main.SYSTEM_PROMPT_DOCUMENT_ANALYSIS,
   "y el prompt de análisis de documentos del chat")

# ═══════════════════════════════════════════════ 9. ninguna expresión se cuelga
print("\n── 9. ninguna expresión se cuelga: cadenas patológicas, < 50 ms cada una ──")
# El bloque de las citas, de la nota de LAS CITAS AGRUPADAS a extract_doc_ids.
# Cada expresión compilada ahí se mide sola, y ninguna puede ir en línea
# («re.sub(r"…"», «re.search(r"…"»): así una nueva no se escapa de la medida.
_ini = FUENTE.index("# ── LAS CITAS AGRUPADAS (26-sep-2026)")
_bloque = FUENTE[_ini:FUENTE.index("def extract_doc_ids(", _ini)]
_nombres = re.findall(r"^(_?[A-Z][A-Z0-9_]*) = re\.compile\(", _bloque, re.M)
_en_linea = re.findall(r"\bre\.(?:search|sub|match|fullmatch|findall|finditer|split)\(r?['\"]", _bloque)
ok(len(_nombres) >= 9 and not _en_linea,
   f"{len(_nombres)} expresiones compiladas en el bloque y ninguna en línea ({len(_en_linea)})")
_PATRONES = {n: getattr(main, n) for n in _nombres}
_PATRONES["DOC_ID_PATTERN"] = main.DOC_ID_PATTERN   # la lee extract_doc_ids tras canonizar

N = 10_000
H16 = "0123456789abcdef"


def _relleno(pieza):
    return (pieza * (N // len(pieza) + 1))[:N]


_PATOLOGICOS = {
    # El caso de la revisión: k=60 grupos de 16, sin el cierre que esperaba.
    "k=60 ( … .)": "Respuesta (véase Doc ID: 0123abcd" + ("-" + H16) * 60 + ".)",
    "k=60 [ … .]": "Respuesta [véase Doc ID: 0123abcd" + ("-" + H16) * 60 + ".]",
    "k=60 sin cierre": "Respuesta (véase Doc ID: 0123abcd" + ("-" + H16) * 60 + " x",
    "k=60 en grupo sin cierre": "[Doc IDs: 0123abcd" + ("-" + H16) * 60 + " x",
    "k=60 suelto": "Doc IDs: 0123abcd" + ("-" + H16) * 60 + ".",
    "etiquetas en paréntesis sin cierre": "(x " + _relleno("Doc ID: 0123abcd-0123; ") + " x",
    "etiquetas en corchete sin cierre": "[x " + _relleno("Doc ID: 0123abcd-0123; ") + " x",
    "[Doc ID: repetido": _relleno("[Doc ID:"),
    "(Doc ID repetido": _relleno("(Doc ID "),
    "Doc. ID repetido": _relleno("Doc. ID "),
    "_Doc ID_: repetido": _relleno("_Doc ID_: "),
    "10k aperturas": "(" * N + "Doc ID: 0123abcd-0123",
    "10k corchetes": "[" * N + "Doc ID: 0123abcd-0123",
    "espacios tras la etiqueta": "[Doc ID" + " " * N + "x",
    "espacios a los dos lados de «:»": "[Doc ID" + " " * (N // 2) + ":" + " " * (N // 2) + "x",
    "espacios en el prefijo": "(x" + " " * N + "Doc ID: 0123abcd-0123 x",
    "comas en el prefijo": "(x" + ", " * (N // 2) + "Doc ID: 0123abcd-0123 x",
    "Doc y espacios": "Doc" + " " * N + "x",
    "Doc y guiones bajos": "Doc" + "_-" * (N // 2) + "x",
    "asteriscos en el prefijo": "(x " + "*" * N + "Doc ID: 0123abcd-0123 x",
    "asteriscos alrededor": "*" * N + "Doc IDs:" + "*" * N + " 0123abcd-0123",
    "separadores sin id": "Doc ID: 0123abcd-0123" + ";" * N,
    "conjunciones sin id": "Doc ID: 0123abcd-0123" + " y" * (N // 2),
    "conjunciones en el resto": "[Doc ID: 0123abcd-0123, " + " y" * (N // 2) + "]",
    "ids y conjunciones": "[Doc IDs: " + _relleno("0123abcd-0123 y o e ") + "]",
    "resto largo": "[Doc ID: 0123abcd-0123, " + "párr. 340 y 341, " * 600 + "]",
    "300 ids": f"[Doc IDs: {'; '.join([A] * 300)}]",
    "alfanumérico largo": "Doc ID: 0123abcd-" + "a" * N,
    "puntos": "Doc ID: 0123abcd" + "." * N,
    # La revisión de 8029c97: la etiqueta empezaba con «[*_]*+» sin ancla y
    # una racha de «_» o «*» se recorría desde cada uno de sus caracteres.
    "racha de _": "_" * N,
    "racha de *": "*" * N,
    "grupo, id y racha de _": f"Respuesta [Doc IDs: {A} " + "_" * N + "]",
    "grupo, id y racha de *": f"Respuesta [Doc IDs: {A} " + "*" * N + "]",
    "(Doc ID: a; ___)": f"Texto (Doc ID: {A}; " + "_" * N + ")",
    "[Doc ID: x___y]": "[Doc ID: x" + "_" * N + "y]",
    "_Doc_ repetido": _relleno("_Doc_"),
}
for _et in ("", "Doc ID: ", "(Doc ID: ", "[Doc IDs: ", "(x, Doc ID: ", "[x; Doc IDs: ",
            "[**Doc IDs:** ", "**Doc IDs:** ", "(x, **Doc ID:** "):
    for _n, _cuerpo in (("hex y guiones", _relleno("0123abcd-")), ("grupos de 16", _relleno("-" + H16)),
                        ("a-", _relleno("a-")), ("hex puro", _relleno("abcdef01")),
                        ("hex…", _relleno("0123abcd…")), ("hex-…", _relleno("0123abcd-…")),
                        ("ids;", _relleno("0123abcd-0123; ")), ("ids pegados", _relleno("0123abcd-0123")),
                        ("ids y etiquetas", _relleno("0123abcd-0123; Doc ID: "))):
        _PATOLOGICOS[f"{_et.strip() or 'sin etiqueta'} + {_n}"] = _et + _cuerpo + " x"

import contextlib as _ctx
import io as _io


def _mide(f, t):
    """El mejor de dos: que un tirón del equipo no pase por retroceso."""
    mejor = 9e9
    for _ in range(2):
        with _ctx.redirect_stdout(_io.StringIO()):
            t0 = _t.perf_counter()
            f(t)
            mejor = min(mejor, _t.perf_counter() - t0)
    return mejor


_FUNCIONES = {
    "expandir_citas_doc_id": main.expandir_citas_doc_id,
    "extract_doc_ids": main.extract_doc_ids,
    "repair_hallucinated_uuids": lambda t: main.repair_hallucinated_uuids(t, m1),
    "validate_citations": lambda t: main.validate_citations(t, m1),
    "_limpiar_historial": lambda t: main._limpiar_historial([main.Message(role="assistant", content=t)]),
    "_sanitize_style_example": main._sanitize_style_example,
    "_quitar_citas_doc_id": main._quitar_citas_doc_id,
}
for _n, _p in _PATRONES.items():
    # search recorre cada posición de arranque si no hay coincidencia; sub,
    # cada posición fuera de una coincidencia.
    _FUNCIONES[_n] = (lambda p: lambda t: (p.search(t), p.sub("", t)))(_p)

_lentos, _peor = [], (0.0, "", "")
for _caso, _texto in _PATOLOGICOS.items():
    for _fn, _f in _FUNCIONES.items():
        _seg = _mide(_f, _texto)
        _peor = max(_peor, (_seg, _fn, _caso))
        if _seg >= 0.05:
            _lentos.append(f"{_fn} · {_caso}: {_seg * 1000:.0f} ms")
ok(not _lentos, f"{len(_PATOLOGICOS)} cadenas × {len(_FUNCIONES)} funciones y expresiones, todas < 50 ms "
   f"(la peor: {_peor[1]} con «{_peor[2]}», {_peor[0] * 1000:.1f} ms)" + (f" — {_lentos[:4]}" if _lentos else ""))
_k60 = "Respuesta (véase Doc ID: 0123abcd" + ("-" + H16) * 60 + ".)"
ok(main.expandir_citas_doc_id(_k60) == "Respuesta (véase [Doc ID: 0123abcd" + ("-" + H16) * 60 + "].)",
   "y la carga de la revisión se sigue leyendo como una cita (el id estropeado lo decide la reparación)")

# ═══════════════════════════════════════════════ 10. tiempo lineal por construcción
print("\n── 10. tiempo lineal por construcción: ningún cuantificador sin tope, y fuzz de 20k y 100k ──")
# Dos rondas seguidas, arreglar una expresión de este bloque metió otra que
# no era lineal: primero la lista de ids (exponencial), después la etiqueta
# con «[*_]*+» sin ancla (cuadrática: 10 s con 60,000 «_» en un turno del
# historial). La sección 9 no lo vio porque no tenía una racha pura de «_».
# Esta sección no depende de haber imaginado la cadena mala:
#   a) LA REGLA, LEÍDA DEL PATRÓN. Ninguna expresión del bloque de las citas
#      (ni DOC_ID_PATTERN, ni la de los registros, ni la de los marcadores)
#      tiene un «*», un «+» o un «{n,}»: con topes, lo que cada expresión hace
#      en un punto de arranque está acotado y el total es proporcional al
#      texto. Lo que no se deja escribir así va en código (ver TIEMPO LINEAL
#      POR CONSTRUCCIÓN en main.py).
#   b) LA PROPIEDAD, MEDIDA. Fuzz con semilla fija: 500 cadenas de 20,000
#      caracteres y 20 de 100,000 de un alfabeto adversario, más las cargas
#      de las revisiones. Cada función pública de las citas, < 60 ms con 20k
#      y < 300 ms con 100k (el mejor de dos), y con 100k no más de ~6 veces
#      lo que tarda con 20k (lineal da 5; cuadrático, 25).
import random as _random
import re._constants as _sre_const
import re._parser as _sre_parser


def _sin_tope(patron):
    """Los cuantificadores sin tope («*», «+», «{n,}», también posesivos o
    perezosos) de una expresión compilada, recorriendo su árbol."""
    malos = []

    def recorre(sub):
        for op, av in sub:
            n = str(op)
            if n in ("MAX_REPEAT", "MIN_REPEAT", "POSSESSIVE_REPEAT"):
                if av[1] == _sre_const.MAXREPEAT:
                    malos.append(f"{{{av[0]},}} sobre {str(list(av[2]))[:40]}")
                recorre(av[2])
            elif n == "SUBPATTERN":
                recorre(av[-1])
            elif n == "ATOMIC_GROUP":
                recorre(av)
            elif n in ("ASSERT", "ASSERT_NOT"):
                recorre(av[1])
            elif n == "BRANCH":
                for b in av[1]:
                    recorre(b)
            elif n == "GROUPREF_EXISTS":
                recorre(av[1])
                if av[2]:
                    recorre(av[2])

    recorre(_sre_parser.parse(patron.pattern, patron.flags))
    return malos


ok(all(_sin_tope(re.compile(x)) for x in (r"a*", r"a+", r"(?:ab){2,}", r"(?>a++)", r"x(?=a*?)", r"(a|b+)")),
   "el detector ve «*», «+», «{n,}», posesivos, perezosos, en grupos, alternativas y aserciones")
_LINEALES = dict(_PATRONES)                       # el bloque entero y DOC_ID_PATTERN
for _n in ("_RE_REGISTRO_CITADO", "_RE_MARCADOR_ABRE", "_RE_NO_SALTO"):
    _LINEALES[_n] = getattr(main, _n)
_sin = {n: _sin_tope(p) for n, p in _LINEALES.items() if _sin_tope(p)}
ok(not _sin, f"{len(_LINEALES)} expresiones y ni un cuantificador sin tope" + (f" — {_sin}" if _sin else ""))

_rnd = _random.Random(20260926)
# Un contexto de 60 fuentes, como el de una consulta: la reparación compara
# cada id que no está en él contra todas.
_D60 = contexto() + [sr(f"{_rnd.getrandbits(32):08x}-{_rnd.getrandbits(16):04x}-{_rnd.getrandbits(16):04x}-"
                        f"{_rnd.getrandbits(16):04x}-{_rnd.getrandbits(48):012x}", registro="2005115")
                     for _ in range(53)]
_M60 = main.build_doc_id_map(_D60)
_PUBLICAS = {
    "expandir_citas_doc_id": main.expandir_citas_doc_id,
    "extract_doc_ids": main.extract_doc_ids,
    "repair_hallucinated_uuids": lambda t: main.repair_hallucinated_uuids(t, _M60),
    "validate_citations": lambda t: main.validate_citations(t, _M60),
    "registros_fuera_del_contexto": lambda t: main.registros_fuera_del_contexto(t, _D60),
    "_marcadores_del_sello": lambda t: main._marcadores_del_sello(t, _M60, _D60),
    "_quitar_citas_doc_id": main._quitar_citas_doc_id,
    "_sanitize_style_example": main._sanitize_style_example,
    "_limpiar_marcadores": main._limpiar_marcadores,
    "_limpiar_historial": lambda t: main._limpiar_historial([main.Message(role="assistant", content=t)]),
    "_recortar_historial": lambda t: main._recortar_historial([main.Message(role="assistant", content=t)]),
}
_UUIDS = [A, B, C, "0b9477ff-9b7b-571c-8a57-227f8683013f"]
# El alfabeto adversario: la puntuación de las citas, dígitos hex, las piezas
# de la etiqueta, la conjunción y uuids enteros y recortados.
_ALFABETO = (list("[]()*_`-;,.: \n") + list("0123456789abcdef") + ["Doc", "IDs", "ID", "y", "…"]
             + _UUIDS + [u[:8] for u in _UUIDS] + [u[:13] for u in _UUIDS] + [u[:23] + "…" for u in _UUIDS]
             + [u[:8] + "-…" for u in _UUIDS] + [u.replace("-", "") for u in _UUIDS])
# Y en una de cada cinco, lo que mueven las otras funciones medidas.
_EXTRA = ["<!--", "-->", "<!-- SOURCES", "Registro", "digital", "núm", main._HUECO_ID, "o", "and",
          "Doc ID: ", "[Doc IDs: ", "(Doc ID: ", "**", "\t"]


def _fuzz(rnd, n, fichas, uniforme=False):
    """Una cadena de `n` caracteres: un motivo de 1 a 6 fichas repetido (el
    90% de las veces) con fichas sueltas en medio, que es como se forman las
    rachas y las repeticiones que hacen daño; o fichas al azar."""
    motivo = fichas if uniforme else [rnd.choice(fichas) for _ in range(rnd.randint(1, 6))]
    trozos, largo = [], 0
    while largo < n:
        x = rnd.choice(motivo) if rnd.random() < 0.9 else rnd.choice(fichas)
        trozos.append(x)
        largo += len(x)
    return "".join(trozos)[:n]


def _cargas(n):
    """Las cargas de las revisiones (d3c257c, 8029c97 y ésta), a `n` caracteres."""
    k = max(1, (n - 40) // 17)
    grupos = ("-" + H16) * k

    def rell(pieza):
        return (pieza * (n // len(pieza) + 1))[:n]

    return {
        "(véase Doc ID: 0123abcd + k grupos .)": "Respuesta (véase Doc ID: 0123abcd" + grupos + ".)",
        "[véase Doc ID: 0123abcd + k grupos .]": "Respuesta [véase Doc ID: 0123abcd" + grupos + ".]",
        "k grupos sin cierre": "Respuesta (véase Doc ID: 0123abcd" + grupos + " x",
        "k grupos en un grupo sin cierre": "[Doc IDs: 0123abcd" + grupos + " x",
        "k grupos sueltos": "Doc IDs: 0123abcd" + grupos + ".",
        "k grupos en mayúsculas": ("Respuesta (véase Doc ID: 0123abcd" + grupos + ".)").upper(),
        "[Doc IDs: a + racha de _]": f"Respuesta [Doc IDs: {A} " + "_" * n + "]",
        "[Doc IDs: a + racha de *]": f"Respuesta [Doc IDs: {A} " + "*" * n + "]",
        "(Doc ID: a; ___)": f"Texto (Doc ID: {A}; " + "_" * n + ")",
        "[Doc ID: x___y]": "[Doc ID: x" + "_" * n + "y]",
        "racha de _": "_" * n,
        "racha de *": "*" * n,
        "_*_* sin Doc": rell("_*"),
        # Justo por debajo de _GRUPO_MAXIMO: cada grupo sí se lee entero, así
        # que una etiqueta sin ancla volvería a ser cuadrática dentro de cada
        # uno (7.6 millones de pasos por grupo).
        "grupos de 3,900 «_», repetidos": rell(f"[Doc IDs: {A} " + "_" * 3900 + "] "),
        "grupos de 3,900 «*», repetidos": rell(f"(Doc IDs: {A}; " + "*" * 3900 + ") "),
        "[Doc IDs: **a** y **b**] repetido": rell(f"[Doc IDs: **{A}** y `{B}`, Tesis X] "),
        "[**Doc ID:** **a**] repetido": rell(f"(**Doc ID:** **{A}**) "),
        "etiquetas en paréntesis sin cierre": "(x " + rell("Doc ID: 0123abcd-0123; ") + " x",
        "aperturas cada 900 con id largo": rell("(" + "x" * 370 + "Doc ID: 0123abcd" + "-" + H16 * 1 + ("-" + H16) * 28),
        "DocID:0123abcd… denso": rell("(" + "DocID:0123abcd…" * 60),
        "[Doc ID: sin cerrar, repetido": rell("[Doc ID: "),
        "saltos de línea antes de <!--": "\n" * n + "<!-- x",
        "<!-- SOURCES sin cierre": rell("<!-- SOURCES x"),
        "Registro y espacios": "Registro" + " " * n + "x",
        "Registro y 200 espacios, repetido": rell("Registro" + " " * 200 + "x"),
        "ids inventados, todos distintos": " ".join(
            f"[Doc ID: {_random.Random(i).getrandbits(128):032x}]" for i in range(n // 43)),
    }


def _uno(f, t):
    with _ctx.redirect_stdout(_io.StringIO()):
        t0 = _t.perf_counter()
        f(t)
        return _t.perf_counter() - t0


def _mejor(f, t, veces):
    return min(_uno(f, t) for _ in range(veces))


_LIM = {20_000: 0.060, 100_000: 0.300}
_lentos, _peor, _no_lineales, _n_cadenas = [], {}, [], {20_000: 0, 100_000: 0}


def _exige(fn, f, texto, n, que):
    """< 60 ms con 20k y < 300 ms con 100k, el mejor de dos (el segundo sólo
    hace falta si el primero no cumple de sobra)."""
    s = _uno(f, texto)
    if s >= _LIM[n] / 2:
        s = min(s, _uno(f, texto))
    if s > _peor.get((fn, n), (0.0, ""))[0]:
        _peor[(fn, n)] = (s, que)
    if s >= _LIM[n]:
        _lentos.append(f"{fn} · {que}: {s * 1000:.0f} ms")


def _lineal(fn, f, t20, t100, que):
    """Con 100k no más de ~6 veces que con 20k. Por debajo de 2 ms no hay
    nada que medir; y antes de acusar se vuelve a medir con más vueltas: un
    tirón del equipo no es un retroceso (el cuadrático da 25 veces, siempre)."""
    # 8× con suelo de 10 ms (26-sep-2026): con 6× y 2 ms fallaba 1 de cada 4
    # corridas con la máquina cargada (2.68 → 16.20 ms, 6.1×) midiendo algo
    # lineal (5.1× y 4.0× aislado). Lo cuadrático da 25×: sigue cazándolo.
    a, b = _mejor(f, t20, 3), _mejor(f, t100, 3)
    if b > 8 * a and b > 0.010:
        a, b = _mejor(f, t20, 7), _mejor(f, t100, 7)
    if b > 8 * a and b > 0.010:
        _no_lineales.append(f"{fn} · {que}: {a * 1000:.2f} → {b * 1000:.2f} ms ({b / a:.1f}×)")


_t0 = _t.perf_counter()
for _i in range(500):
    _fichas = _ALFABETO + (_EXTRA if _i % 5 == 0 else [])
    _s = _fuzz(_rnd, 20_000, _fichas, uniforme=_i % 7 == 6)
    _n_cadenas[20_000] += 1
    for _fn, _f in _PUBLICAS.items():
        _exige(_fn, _f, _s, 20_000, f"fuzz 20k #{_i}")
for _i in range(20):
    _fichas = _ALFABETO + (_EXTRA if _i % 5 == 0 else [])
    _s100 = _fuzz(_rnd, 100_000, _fichas, uniforme=_i % 7 == 6)
    _s20 = _fuzz(_rnd, 20_000, _fichas, uniforme=_i % 7 == 6)
    _n_cadenas[100_000] += 2
    for _fn, _f in _PUBLICAS.items():
        _exige(_fn, _f, _s100, 100_000, f"fuzz 100k #{_i}")
        _exige(_fn, _f, _s20 * 5, 100_000, f"fuzz 20k×5 #{_i}")
        # La misma cadena cinco veces: lo que tarde de más no es el contenido.
        _lineal(_fn, _f, _s20, _s20 * 5, f"fuzz 20k #{_i} y cinco veces")
_C20, _C100 = _cargas(20_000), _cargas(100_000)
for _que in _C20:
    for _fn, _f in _PUBLICAS.items():
        _exige(_fn, _f, _C20[_que], 20_000, _que)
        _exige(_fn, _f, _C100[_que], 100_000, _que)
        _lineal(_fn, _f, _C20[_que], _C100[_que], _que)
_seg = _t.perf_counter() - _t0
for _n, _lim in _LIM.items():
    _p = max((v for (fn, n), v in _peor.items() if n == _n), default=(0.0, ""))
    _pf = max(((v[0], fn) for (fn, n), v in _peor.items() if n == _n), default=(0.0, ""))
    print(f"   con {_n:,}: la más lenta, {_pf[1]} ({_p[1]}), {_p[0] * 1000:.1f} ms")
ok(not _lentos, f"{len(_PUBLICAS)} funciones públicas × ({_n_cadenas[20_000]} + {len(_C20)} cadenas de 20k, "
   f"{_n_cadenas[100_000]} + {len(_C100)} de 100k): < 60 ms y < 300 ms ({_seg:.0f} s en total)"
   + (f" — {_lentos[:6]}" if _lentos else ""))
ok(not _no_lineales, "y con 100k ninguna tarda más de ~6 veces lo que con 20k"
   + (f" — {_no_lineales[:6]}" if _no_lineales else ""))

# ═══════════════════════════════════════════════ 11. el caso real, igual que con 8029c97
print("\n── 11. el caso real sale igual que con el commit anterior (8029c97) ──")
# Las huellas de abajo se calcularon con 8029c97 y esta misma función. Todo
# lo que cambió en esta ronda son formas que el caso real no tiene: si una
# huella cambia, cambió lo que el abogado ve.
import hashlib as _hl


def _entradas_del_caso(crudo):
    """El texto de la respuesta (sin CITATION_META), su contexto y la
    conversación como la manda el cliente."""
    cuerpo = crudo.split("\n\n<!-- CITATION_META")[0]
    ids = sorted(set(re.findall(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", cuerpo)))
    docs_ = contexto(ids)
    mensajes = [main.Message(role="user", content="Traza la línea del control de convencionalidad."),
                main.Message(role="assistant", content=crudo)]
    return cuerpo, main.build_doc_id_map(docs_), docs_, mensajes


def _huella(crudo):
    """Lo que cada función de las citas hace con el caso, en una huella
    SHA-256. Lo que depende del orden de un conjunto se ordena."""
    texto, mapa, docs_, mensajes = _entradas_del_caso(crudo)
    with _ctx.redirect_stdout(_io.StringIO()):
        v = main.validate_citations(texto, mapa)
        sello = []
        for x in main._marcadores_del_sello(texto, mapa, docs_):
            m = re.search(r"<!-- CITATION_META:(\{.*\}) -->", x, re.S)
            if m:
                j = json.loads(m.group(1))
                j["invalid_ids"] = sorted(j["invalid_ids"])
                x = j
            sello.append(x)
        salidas = {
            "expandir": main.expandir_citas_doc_id(texto),
            "extract": sorted(main.extract_doc_ids(texto)),
            "reparar": main.repair_hallucinated_uuids(texto, mapa),
            "validar": [v.valid_count, v.invalid_count, v.total_citations,
                        sorted((c.doc_id, c.status) for c in v.citations)],
            "sello": sello,
            "registros": main.registros_fuera_del_contexto(texto, docs_),
            "estilo": main._sanitize_style_example(texto),
            "quitar": main._quitar_citas_doc_id(texto),
            "marcadores": main._limpiar_marcadores(crudo),
            "historial": [m.content for m in main._limpiar_historial(mensajes)],
            "recorte": [m.content for m in main._recortar_historial(mensajes)[0]],
        }
    return _hl.sha256(json.dumps(salidas, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


_HUELLA_FRAGMENTO_8029C97 = "67ce0dd9d39b853e78437e2ff21b5ff3199e400fa71455a265c0d5ac828f7e1c"
_HUELLA_CASO_8029C97 = "bcb098d6c8d51a96bc11232c3f387a1e60ddbcb1e0818db55b972f6646c66ee9"
ok(_huella(FRAGMENTO) == _HUELLA_FRAGMENTO_8029C97,
   "el fragmento del caso: expandir, extraer, reparar, validar, sello, estilo, marcadores e historial, idénticos")
if CASO.exists():
    _crudo = json.loads(CASO.read_text(encoding="utf-8"))[1]["content"]
    ok(_huella(_crudo) == _HUELLA_CASO_8029C97,
       f"la conversación real entera ({len(_crudo):,} caracteres), idéntica")
else:
    print("   (la conversación real no está en el scratchpad: sólo se compara el fragmento)")

print()
if FALLOS:
    print(f"✗ {len(FALLOS)} FALLO(S):")
    for f in FALLOS:
        print("   ·", f)
    sys.exit(1)
print("✓ TODO PASA — las citas agrupadas se leen como singulares y las 7 entran a sources.")
