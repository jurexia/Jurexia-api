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

# ═══════════════════════════════════════════════ 7. el historial vuelve en singular
print("\n── 7. el historial que vuelve al modelo va en singular ──")
hist = [main.Message(role="user", content=f"¿Y esto? [Doc IDs: {A}; {B}]"),
        main.Message(role="assistant", content=f"Así. [Doc IDs: {A}; {B}]")]
limpio = main._limpiar_historial(hist)
ok(limpio[1].content == f"Así. {DOS}", "la respuesta guardada con «Doc IDs» vuelve canonizada")
ok(limpio[0].content == hist[0].content, "lo que escribió el abogado no se toca")
_ya = [main.Message(role="assistant", content=f"Así. {DOS}")]
ok(main._limpiar_historial(_ya) is _ya, "un historial limpio no paga copia")

# ═══════════════════════════════════════════════ 8. el cableado y el prompt
print("\n── 8. el cableado y la regla del prompt ──")
FUENTE = Path("main.py").read_text(encoding="utf-8")
_i = FUENTE.index("content_buffer = expandir_citas_doc_id(content_buffer)")
ok(_i < FUENTE.index("uuid_repair_map: Dict[str, str] = {}", _i)
   < FUENTE.index("validation = validate_citations(content_buffer, doc_id_map)", _i)
   < FUENTE.index("_RE_PAR.finditer(content_buffer", _i),
   "/chat canoniza el búfer antes de la reparación, la validación y el sello de correspondencia")
ok("enhanced_text = expandir_citas_doc_id(enhanced_text)" in FUENTE,
   "/enhance devuelve el texto entero ya canónico")
ok("matches = DOC_ID_PATTERN.findall(expandir_citas_doc_id(" in FUENTE,
   "extract_doc_ids (validador, sello, /analyze-document, /chat-sentencia) lee las agrupadas")
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

print()
if FALLOS:
    print(f"✗ {len(FALLOS)} FALLO(S):")
    for f in FALLOS:
        print("   ·", f)
    sys.exit(1)
print("✓ TODO PASA — las citas agrupadas se leen como singulares y las 7 entran a sources.")
