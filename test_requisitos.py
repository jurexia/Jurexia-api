# -*- coding: utf-8 -*-
"""La recuperación por requisitos (rediseño, etapa 2).

    .venv/bin/python test_requisitos.py
"""
import asyncio, ast, sys, types
sys.path.insert(0, ".")
import requisitos as rq
import contexto_taller as ct

FALLOS = []


def ok(c, q):
    print(f"   {'PASA ' if c else 'FALLA'}  {q}")
    if not c:
        FALLOS.append(q)


print("\n1 · LA DEMOSTRACIÓN, NORMALIZADA")
dem = rq._normalizar({"figura": "sustitución procesal", "mecanismo": "cesión", "momento": "durante la ejecución",
                      "requisitos": [{"id": "Q1", "enunciado": "que se transmita el derecho concreto",
                                      "consulta_rubro": "SUSTITUCIÓN PROCESAL. REQUISITOS", "concepto_norma": "cesión de derechos litigiosos",
                                      "preceptos": [{"ley": "Código Civil Federal", "articulo": "2029 bis"}, {"ley": "", "articulo": "1"}]},
                                     {"enunciado": ""}, {"enunciado": "que se notifique"}, {"enunciado": "a"}, {"enunciado": "b"}, {"enunciado": "c"}],
                      "excepciones": ["x"], "consecuencias": ["y"]})
ok(len(dem["requisitos"]) == 4 and dem["requisitos"][0]["preceptos"] == [{"ley": "Código Civil Federal", "articulo": "2029"}],
   "hasta cuatro requisitos con enunciado; preceptos con ley y número (el «bis» no se inventa como otro artículo)")
ok(dem["requisitos"][1]["consulta_rubro"] == "que se notifique", "sin consulta, se busca con el enunciado")
ok(rq._normalizar({"requisitos": []}) is None, "sin requisitos no hay demostración")

print("\n2 · LA BÚSQUEDA POR REQUISITO, SUMADA Y MARCADA")


class Mat:
    def __init__(self):
        self.tesis = [{"registro": "1", "rubro": "YA ESTABA"}]
        self.normas = [{"cuerpo_legal": "Código Civil Federal", "articulo": "1", "texto": "ya"}]


async def _prueba():
    import fase6_rag as f6r
    orig_mp, orig_cp = f6r.material_para, f6r.completar_preceptos

    async def falso_mp(q, ej, el, problema, col, materia="", cliente=None, hecho="", sede_acto="", cuaderno=""):
        if "NOTIFIQUE" in problema.upper():
            return types.SimpleNamespace(tesis=[], normas=[])
        return types.SimpleNamespace(tesis=[{"registro": "1"}, {"registro": "7", "fecha_publicacion": "2020-01-01"}],
                                     normas=[{"cuerpo_legal": "Código Civil Federal", "articulo": "2029", "texto": "La cesión…"}])

    async def falso_cp(q, material, pares, col=None, materia="", tipo_asunto=""):
        material.normas.append({"cuerpo_legal": "Código Civil Federal", "articulo": "2030", "texto": "otro"})
        return ["art. 2030"]
    f6r.material_para, f6r.completar_preceptos = falso_mp, falso_cp
    try:
        m = Mat()
        d = rq._normalizar({"requisitos": [
            {"id": "Q1", "enunciado": "transmitir el derecho", "consulta_rubro": "CESIÓN",
             "preceptos": [{"ley": "Código Civil Federal", "articulo": "2030"}]},
            {"id": "Q2", "enunciado": "que se notifique", "consulta_rubro": "que se notifique"}]})
        ct.poner(False, {})
        d = await rq.recuperar(object(), None, None, m, d)
        return m, d
    finally:
        f6r.material_para, f6r.completar_preceptos = orig_mp, orig_cp

m, d = asyncio.run(_prueba())
nuevas = [t for t in m.tesis if t.get("para_requisito")]
ok([t["registro"] for t in nuevas] == ["7"] and nuevas[0]["origen"] == "requisito",
   "la tesis nueva entra marcada «para_requisito»; la que ya estaba no se duplica")
ok(any(n.get("articulo") == "2029" and n.get("para_requisito") == "Q1" for n in m.normas)
   and any(n.get("articulo") == "2030" and n.get("para_requisito") == "Q1" for n in m.normas),
   "las normas del requisito y el precepto nombrado, sumados y atribuidos al requisito")
ok(d["huecos"] == ["Q2"], "el requisito sin ninguna fuente es un HUECO DECLARADO")
ok(d["fichas"] and d["fichas"][0]["requisito"] == "Q1" and "version" in d["fichas"][0], "y cada norma lleva su ficha")


print("\n2b · LA LEY DEL ESTADO SÓLO SI RIGE (103/2025: catastro y adolescentes en una sucesión agraria)")
_d = lambda *leyes: {"requisitos": [{"preceptos": [{"ley": l, "articulo": "1"} for l in leyes]}]}
ok(not rq.rige_ley_local(_d("Constitución Política de los Estados Unidos Mexicanos", "Ley Agraria"))
   and not rq.rige_ley_local(_d("Ley de Amparo")) and not rq.rige_ley_local(_d("Código Civil Federal"))
   and not rq.rige_ley_local(_d("Código Nacional de Procedimientos Penales")) and not rq.rige_ley_local({}),
   "leyes federales (aunque su nombre no diga fuero): la cesta del estado no se abre")
ok(rq.rige_ley_local(_d("Código de Procedimientos Civiles del Estado de Querétaro"))
   and rq.rige_ley_local(_d("Código Civil")),
   "ley del estado, o código que sin entidad es local: se abre")
_ns = [{"cuerpo_legal": "Ley de Catastro para el Estado de Querétaro", "articulo": "72"},
       {"cuerpo_legal": "Ley Agraria", "articulo": "17"}]
ok([n["articulo"] for n in rq._por_leyes_nombradas(_ns, {"preceptos": [{"ley": "Ley Agraria", "articulo": "189"}]})] == ["17", "72"],
   "las normas de la ley que el requisito nombra van primero en su cupo")


async def _prueba_cesta():
    import fase6_rag as f6r
    orig_mp, orig_cp = f6r.material_para, f6r.completar_preceptos
    vistas = []

    async def falso_mp(q, ej, el, problema, col, materia="", cliente=None, hecho="", sede_acto="", cuaderno=""):
        vistas.append(col)
        return types.SimpleNamespace(tesis=[], normas=[])

    async def falso_cp(q, material, pares, col=None, materia="", tipo_asunto=""):
        vistas.append(("cp", col))
        return []
    f6r.material_para, f6r.completar_preceptos = falso_mp, falso_cp
    try:
        m = Mat()
        m.normas.append({"cuerpo_legal": "Ley Agraria", "articulo": "189", "texto": "ya estaba"})
        d = rq._normalizar({"requisitos": [
            {"id": "Q1", "enunciado": "valorar pruebas", "consulta_rubro": "PRUEBAS AGRARIAS",
             "preceptos": [{"ley": "Ley Agraria", "articulo": "189"}]}]})
        d = await rq.recuperar(object(), None, None, m, d, coleccion_estatal="leyes_queretaro")
        return vistas, d
    finally:
        f6r.material_para, f6r.completar_preceptos = orig_mp, orig_cp

_v, _dd = asyncio.run(_prueba_cesta())
ok(_v and all((x is None) or (isinstance(x, tuple) and x[1] is None) for x in _v),
   "sin ley local nombrada, ni la búsqueda ni los preceptos abren leyes_queretaro")
ok(_dd["requisitos"][0]["normas"][:1] == ["art. 189 — Ley Agraria"] and _dd["huecos"] == []
   and _dd["fichas"] and _dd["fichas"][0]["norma"] == "art. 189 — Ley Agraria",
   "el precepto nombrado que YA estaba en el material sostiene el requisito y tiene ficha")

ok(not rq.rige_ley_local(_d("Ley del Instituto de Seguridad y Servicios Sociales de los Trabajadores del Estado"))
   and not rq.rige_ley_local(_d("Ley Federal de los Trabajadores al Servicio del Estado")),
   "la Ley del ISSSTE dice «del Estado» y es federal: no abre la ley de la entidad")
ok(rq._entidad_por_nombre("Código Civil del Estado de Querétaro") == "QUERÉTARO"
   and rq._entidad_por_nombre("Ley Agraria") == "SIN FUERO EN EL NOMBRE"
   and rq._entidad_por_nombre("Código Fiscal de la Federación") == "FEDERAL",
   "la ficha no llama federal a un código del estado traído de internet (sin `entidad`)")


async def _prueba_por_requisito():
    import fase6_rag as f6r
    orig_mp, orig_cp = f6r.material_para, f6r.completar_preceptos
    vistas = {}

    async def falso_mp(q, ej, el, problema, col, materia="", cliente=None, hecho="", sede_acto="", cuaderno=""):
        vistas[problema] = col
        return types.SimpleNamespace(tesis=[], normas=[])

    async def falso_cp(q, material, pares, col=None, materia="", tipo_asunto=""):
        vistas["pares"] = sorted(pares)
        return []
    f6r.material_para, f6r.completar_preceptos = falso_mp, falso_cp
    try:
        m = Mat()
        m.sede_del_acto, m.cuaderno = "ordinaria", "principal"
        d = rq._normalizar({"requisitos": [
            {"id": "Q1", "enunciado": "a", "consulta_rubro": "LOCAL",
             "preceptos": [{"ley": "Código Civil del Estado de Querétaro", "articulo": "1480"}]},
            {"id": "Q2", "enunciado": "b", "consulta_rubro": "FEDERAL",
             "preceptos": [{"ley": "Ley de Amparo", "articulo": "128"}, {"ley": "Código Civil Federal", "articulo": "1"}]}]})
        await rq.recuperar(object(), None, None, m, d, coleccion_estatal="leyes_queretaro")
        return vistas
    finally:
        f6r.material_para, f6r.completar_preceptos = orig_mp, orig_cp

_vp = asyncio.run(_prueba_por_requisito())
ok(_vp.get("LOCAL") == "leyes_queretaro" and _vp.get("FEDERAL") is None,
   "la ley del estado se abre SÓLO para el requisito que nombra una ley local")
ok(_vp.get("pares") == [("código civil del estado de querétaro", "1480")],
   "los preceptos nombrados pasan las guardas de siempre: ni la suspensión del amparo en sede ordinaria ni lo ya presente")

print("\n3 · LA FICHA Y LA VERSIÓN")
import json as _j
_una = _j.load(open("scripts/leyes_federales_catalogo.json", encoding="utf-8"))[0]["ley"]
ok(rq.version_de(_una).startswith(("texto vigente con la última reforma", "texto original")),
   f"una ley federal del catálogo dice su versión DOF ({_una[:40]}…: {rq.version_de(_una)})")
ok(rq.version_de("Código Civil del Estado de Querétaro") == "versión sin fecha", "una estatal, «versión sin fecha»")

print("\n4 · PARA LA PROPUESTA")
b = rq.bloque_propuesta(d)
ok("LA DEMOSTRACIÓN QUE HAY QUE COMPLETAR" in b and "Q2. que se notifique — SIN FUENTE EN EL ACERVO" in b,
   "cada requisito con sus fuentes o su hueco dicho")
ok(rq.bloque_propuesta(None) == "", "sin demostración, nada")

print("\n5 · LAS NORMAS POR PROCEDENCIA EN LA PROPUESTA")
import fase5_propuesta as f5
mat = types.SimpleNamespace(normas=[{"cuerpo_legal": f"Ley {i}", "articulo": str(i), "texto": "t"} for i in range(12)]
                                   + [{"cuerpo_legal": "Ley CITADA", "articulo": "99", "texto": "t", "origen": "citada"}])
ct.poner(True, {}, pruebas=False)
ok("Ley CITADA" not in f5._bloque_normas(mat), "sin la bandera, como hoy: la citada al final se queda fuera del corte de 10")
ct.poner(True, {}, pruebas=True)
_bn = f5._bloque_normas(mat)
ok(_bn.startswith("· Ley 0") and "Ley 9 —" in _bn and "Ley CITADA" in _bn and "Ley 10 —" not in _bn,
   "con la bandera, las diez de la consulta se quedan y lo traído se SUMA detrás (no las desplaza)")
_mat2 = types.SimpleNamespace(normas=[{"cuerpo_legal": f"Ley {i}", "articulo": str(i), "texto": "t"} for i in range(10)]
                                     + [{"cuerpo_legal": f"Ley R{i}", "articulo": str(i), "texto": "t", "origen": "requisito"}
                                        for i in range(10)])
_bn2 = f5._bloque_normas(_mat2)
ok(_bn2.count("· Ley R") == f5.NORMAS_TRAIDAS_EXTRA and "· Ley 9 —" in _bn2,
   "con diez de requisitos, la ley de la consulta sigue en el prompt (revisión adversarial) y el extra tiene tope")
b_sin = __import__("requisitos").bloque_propuesta(
    {"requisitos": [{"id": "Q1", "enunciado": "e", "tesis": ["123"], "normas": ["art. 1 — Ley X"]}]},
    types.SimpleNamespace(tesis=[], normas=[]))
ok("SIN FUENTE EN EL ACERVO" in b_sin and "123" not in b_sin,
   "la fuente que no llegó al material (tope al guardar, otro worker) no se nombra como presente")

print("\n6 · EL CABLEADO")
src = open("main.py", encoding="utf-8").read()
FN = {n.name: n for n in ast.walk(ast.parse(src)) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
fr = ast.get_source_segment(src, FN["_taller_requisitos"])
ok('rediseno("recuperacion_requisitos")' in fr and "wait_for(_todo(), timeout=REQUISITOS_ESPERA_S)" in fr,
   "sólo con la bandera, y con tope: nunca bloquea la propuesta")
nuc = ast.get_source_segment(src, FN["_taller_proponer_nucleo"])
ok("requisitos=_requisitos_p" in nuc and "leyes_de_la_litis" in nuc, "la propuesta la recibe; y la litis se aplica antes de proponer")
bd = open("busqueda_dirigida.py", encoding="utf-8").read()
ok('[_f6r.COLECCION_FEDERAL]' in bd, "leer_articulo puede leer leyes federales (con la bandera)")

print()
if FALLOS:
    print("FALLAS:\n  " + "\n  ".join(FALLOS))
    sys.exit(1)
print("RESULTADO: TODAS LAS COMPROBACIONES PASAN")
