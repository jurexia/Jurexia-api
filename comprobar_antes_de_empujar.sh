#!/bin/zsh
# LA COMPROBACIÓN QUE FALLÓ EN SILENCIO, HECHA BIEN.
#
# Estaba escrita así:
#     python -c "import main, ..." >/dev/null 2>&1 && echo "  ✓ importan"
#
# Con `&&`, un fallo de importación NO detiene nada: sólo se salta el echo. Yo
# lo miré, no vi el «✓», y empujé igual un fichero con una cadena sin cerrar.
# Toda la API devolvió 500. Es exactamente el fallo que este proyecto lleva
# semanas corrigiendo en otros —capturar un error y seguir— cometido en la
# herramienta que existe para impedirlo.
#
# Ahora: `set -e`, y cada comprobación imprime lo que hizo. Si algo falla, el
# guion muere y el push no ocurre.
set -e
cd "$(dirname "$0")"

echo "── 1. sintaxis de todos los módulos del taller ──"
.venv/bin/python - <<'PY'
import ast, io, sys, glob
malos = []
for f in sorted(glob.glob("*.py")):
    try:
        ast.parse(io.open(f, encoding="utf8").read())
    except SyntaxError as e:
        malos.append(f"{f}:{e.lineno} {e.msg}")
if malos:
    print("   ✗ " + "\n   ✗ ".join(malos)); sys.exit(1)
print(f"   ✓ {len(glob.glob('*.py'))} ficheros sin errores de sintaxis")
PY

echo "── 2. importan de verdad ──"
.venv/bin/python - <<'PY'
import sys
sys.path.insert(0, ".")
mods = ["main", "redactor_adelanto", "documento_generado", "fase6_estudio",
        "fase5_propuesta", "fases123_pipeline", "fases123_resumenes",
        "fase_precedente", "fase_partes", "fase_origen", "fase_autoridad",
        "tipos_asunto", "banco", "ensamblar_adelanto", "modos_decision",
        "calidad_estudio", "meta_lenguaje", "llamada_modelo",
        "fase_procedencia_rf", "contaminacion"]
for m in mods:
    __import__(m)
print(f"   ✓ los {len(mods)} módulos importan")
PY

echo "── 3. las rutas del taller cuelgan de su función ──"
.venv/bin/python - <<'PY'
import ast, sys
a = ast.parse(open("main.py", encoding="utf8").read())
r = []
for n in ast.walk(a):
    if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
        for d in n.decorator_list:
            f = d.func if isinstance(d, ast.Call) else d
            if isinstance(f, ast.Attribute) and f.attr in ("post", "get"):
                r.append((str(d.args[0].value if isinstance(d, ast.Call) and d.args
                              else "?"), n.name))
t = [x for x in r if "/taller" in x[0]]
mal = [x for x in t if not x[1].startswith("taller_")]
if mal:
    print(f"   ✗ mal colgadas: {mal}"); sys.exit(1)
print(f"   ✓ {len(t)} rutas del taller, todas bien colgadas")
PY

# ── 3b. LOS NOMBRES QUE SÓLO EXISTEN A VECES ──────────────────────────────
# Tres veces en un día: una variable usada en una función que no la recibe.
# `criterios=criterios` dentro de `_componer_generado`, que no tenía ese
# parámetro, tumbó el amparo en revisión DESPUÉS de cinco minutos de trabajo y
# con un mensaje que no dice dónde: «name 'criterios' is not defined». La verja
# no lo vio porque compone directo, sin pasar por el módulo que lo rompía.
echo "── 3b. ningún nombre indefinido en los módulos del taller ──"
.venv/bin/python - <<'PYEOF'
import ast, builtins, sys, pathlib

# UNA FUNCIÓN ANIDADA VE LO DE LA DE FUERA, y la primera versión de esta
# comprobación no lo sabía: acusó a `_partes_y_estructura()` por usar
# `cliente`, que es un parámetro de la función que la contiene. Novena vez hoy
# que una comprobación mía señala código correcto, y la regla no cambia: se
# recorre el árbol arrastrando el ámbito, no fichero a fichero.
MODULOS = ["redactor_adelanto.py", "documento_generado.py", "fase6_estudio.py",
           "fase_rama.py", "tipos_asunto.py", "ensamblar_adelanto.py",
           "fase0_oportunidad.py", "fase_origen.py", "calidad_estudio.py",
           "linter_juridico.py", "normas_estaticas.py", "fase_normas.py",
           # main.py ES DONDE VIVEN LAS RUTAS, y estaba fuera. El 9 de
           # septiembre metí un `avisos.append(...)` en dos funciones donde
           # `avisos` no existe: habría reventado la generación entera con un
           # NameError, y esta comprobación no lo vio porque el fichero no
           # estaba en la lista. Es el mismo error de ámbito del `_rama`,
           # entrando por el único módulo que no se miraba.
           "main.py",
           "fase6_rag.py", "fase5_propuesta.py", "fase_partes.py",
           "fases123_pipeline.py", "fase_sintesis.py"]
BUILTIN = set(dir(builtins)) | {
    # Globales de módulo que siempre existen y el barrido no conoce.
    "__file__", "__name__", "__doc__", "__package__", "__spec__",
}

# HALLAZGOS ANTERIORES A QUE main.py ENTRARA AQUÍ, fuera del taller. Se anotan
# para que el guardián quede verde con lo nuevo SIN esconderlos: son fallos
# reales que hay que arreglar, no falsos positivos.
#
#   main.py:7539-7540  _fetch_neighbor_chunks() usa «tesis_num» y «registro»,
#                      que no son parámetros suyos ni se asignan dentro: es un
#                      NameError en el recuperador de vecinos del chat.
#   main.py:22207      qdrant_search_for_redactor() usa «generate_embedding»
#   main.py:25716      phase1_activate() usa «_embed_async»
viejos = set()
CONOCIDOS_VIEJOS = {
    ("main.py", "_fetch_neighbor_chunks", "registro"),
    ("main.py", "_fetch_neighbor_chunks", "tesis_num"),
    ("main.py", "qdrant_search_for_redactor", "generate_embedding"),
    ("main.py", "phase1_activate", "_embed_async"),
}


def propios(nodo):
    """Los nodos de ESTE ámbito, SIN entrar en funciones ni clases de dentro.

    ÉSTE ERA EL AGUJERO. `liga` usaba `ast.walk`, que desciende en todo, así
    que las variables locales de CADA función acababan contadas como nombres
    del MÓDULO y quedaban visibles en todas partes.

    Medido: `_rama` estaba asignado dentro de una función y usado dentro de
    OTRA, que no lo tenía. El guardián dijo «TODO PASA» y cada generación del
    taller murió en producción con «name '_rama' is not defined» —el servidor
    devolvía 200 con su evento de error y la pantalla se quedaba muda—.

    Un guardián que no ve esta clase de fallo es peor que ninguno: da permiso
    para empujar.
    """
    fuera = []
    pila = list(ast.iter_child_nodes(nodo))
    while pila:
        x = pila.pop()
        fuera.append(x)
        # No se entra en el cuerpo de funciones ni clases: sus locales son
        # suyas. Sí se miran sus DECORADORES y sus valores por omisión, que se
        # evalúan en este ámbito.
        if isinstance(x, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            for d in getattr(x, "decorator_list", []):
                pila.extend(ast.walk(d))
            continue
        pila.extend(ast.iter_child_nodes(x))
    return fuera


def liga(nodo, heredado):
    """Los nombres que existen dentro de `nodo`, con lo que hereda."""
    n = set(heredado)
    for x in propios(nodo):
        if isinstance(x, ast.Name) and isinstance(x.ctx, ast.Store):
            n.add(x.id)
        elif isinstance(x, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            n.add(x.name)
        elif isinstance(x, (ast.Import, ast.ImportFrom)):
            for a in x.names:
                n.add((a.asname or a.name).split(".")[0])
        elif isinstance(x, ast.ExceptHandler) and x.name:
            n.add(x.name)
        elif isinstance(x, ast.arg):
            n.add(x.arg)
        elif isinstance(x, ast.Global) or isinstance(x, ast.Nonlocal):
            n.update(x.names)
    return n


malos = []
for f in MODULOS:
    arbol = ast.parse(pathlib.Path(f).read_text(encoding="utf8"))
    modulo = liga(arbol, BUILTIN)

    def revisa(fn, visible):
        dentro = liga(fn, visible)
        # Los usos también se miran SÓLO en este ámbito: lo que hace una
        # función anidada se comprueba cuando le toque su turno, con su propio
        # ámbito heredado.
        for x in propios(fn):
            if isinstance(x, ast.Name) and isinstance(x.ctx, ast.Load) \
                    and x.id not in dentro:
                if (f, fn.name, x.id) in CONOCIDOS_VIEJOS:
                    viejos.add((f, fn.name, x.id))
                    continue
                malos.append(f"{f}:{x.lineno} {fn.name}() usa «{x.id}»")
        for h in ast.iter_child_nodes(fn):
            for y in ast.walk(h):
                if isinstance(y, (ast.FunctionDef, ast.AsyncFunctionDef)) and y is not fn:
                    pass

    # CADA FUNCIÓN CON EL ÁMBITO DE SU PADRE, no con el del módulo. Una
    # anidada ve las locales de quien la contiene; una hermana, no.
    def recorre(nodo, visible):
        for h in ast.iter_child_nodes(nodo):
            if isinstance(h, (ast.FunctionDef, ast.AsyncFunctionDef)):
                revisa(h, visible)
                recorre(h, liga(h, visible))
            elif isinstance(h, ast.ClassDef):
                recorre(h, visible)
            else:
                for y in ast.iter_child_nodes(h):
                    recorre(h, visible)
                    break
    recorre(arbol, modulo)

if malos:
    print("   ✗ NOMBRES QUE PUEDEN NO EXISTIR:")
    for m in malos[:12]:
        print("     ", m)
    sys.exit(1)
if viejos:
    print(f"   ⚠️ {len(viejos)} nombres indefinidos ANTERIORES, fuera del "
          f"taller, pendientes de arreglar:")
    for f_, fn_, n_ in sorted(viejos):
        print(f"      {f_} · {fn_}() usa «{n_}»")
print(f"   ✓ {len(MODULOS)} módulos sin nombres indefinidos nuevos")
PYEOF

echo "── 4. el documento se compone en los cuatro tipos ──"
.venv/bin/python - <<'PY'
import sys, datetime as dt
sys.path.insert(0, ".")
import documento_generado as dg, fase0_oportunidad as f0, tipos_asunto as ta
import ensamblar_adelanto as ens, meta_lenguaje as ml
from docx import Document
c = f0.computar(dt.date(2025, 3, 13), dt.date(2025, 4, 8), plazo=15)
malos = []
for t in ("amparo_directo", "amparo_revision", "queja", "revision_fiscal"):
    for cs in (["infundado"], ["fundado"]):
        est = dg.Estructura(apertura="Q.", visto="para resolver.",
                            resultandos=[{"titulo": "X", "texto":
                                "contra el auto de seis de junio de dos mil "
                                "veinticinco, dictado en el juicio de amparo "
                                "742/2023-II, que desechó el incidente de "
                                "nulidad de notificación."}],
                            competencia="", existencia="", procedencia="")
        r = f"/tmp/_guard_{t}_{cs[0]}.docx"
        dg.componer({"tipo_asunto": t, "numero": "1/2026",
                     "encabezado": "ASUNTO 1/2026", "quejoso": "P",
                     "responsable": "el Juzgado Segundo de Distrito",
                     "magistrado": "M", "secretario": "S",
                     "tribunal": "Tercer Tribunal Colegiado en Materias "
                                 "Administrativa y Civil del Vigésimo Segundo "
                                 "Circuito", "ciudad": "Q"},
                    est, c, f0.fecha_en_letra, r, estudio="El estudio.",
                    calificaciones=cs, tipo_asunto=t)
        txt = "\n".join(p.text for p in Document(r).paragraphs)
        for k, ok in (("fórmula ajena", not ta.cierre_ajeno(t, txt)),
                      ("congruencia", not ens.revisar_congruencia(r, cs, t)),
                      ("competencia", not ta.prohibido_en_competencia(t, txt)),
                      ("meta-lenguaje", not ml.frases(txt)),
                      ("perífrasis", not ml.perifrasis(txt))):
            if not ok:
                malos.append(f"{t}/{cs[0]}: {k}")
if malos:
    print("   ✗ " + "\n   ✗ ".join(malos)); sys.exit(1)
print("   ✓ 8 documentos compuestos, 40 comprobaciones limpias")
PY

echo
echo "  ✓ TODO PASA. Se puede empujar."
