"""La tesis de la inoperancia, por el vicio concreto — 2-oct-2026.

David: «en la cita de jurisprudencias sobre inoperancia siempre es la misma; el
sistema es muy recurrente con jurisprudencias iguales cuando tenemos un acervo
basto». Esto comprueba, sin red (Qdrant y embeddings falsos):

  1 · el catálogo: cada razón de inoperancia del plan y cada calificativa de
      técnica tiene 2-3 consultas de rubro, y el vicio se deduce de la razón;
  2 · con la bandera APAGADA, todo como antes: el renglón del impedimento, la
      pregunta sintética, la búsqueda por calificativa y los prompts del
      estudio (con la 2a./J. 58/2010 donde estaba);
  3 · con la bandera ENCENDIDA: el prompt del estudio no trae la clave real ni
      manda citar de memoria, y la fase 3 pide el vicio descrito;
  4 · los filtros: fuera las que perdieron vigencia y las de otro recurso; la
      pertinencia al argumento ordena el pozo y, entre las pertinentes, manda
      la jerarquía (la obligatoria de la Corte no cede ante una orientadora);
  5 · antes del plan: una COPIA del material con las tesis del vicio —el de
      la sesión no se toca—, sin vicios de forma con suplencia confirmada;
  6 · /taller/razonar no muta el material de la sesión con la bandera, y los
      dos gemelos del resolver pasan la copia al plan y al estudio (AST).

    .venv/bin/python test_inoperancia_vicio.py
"""
import ast
import asyncio
import os
import sys

os.environ.pop("INOPERANCIA_POR_VICIO", None)
os.environ.pop("ESTUDIO_PROMPT", None)

import fase6_estudio as f6
import fase6_rag as rag
import fases123_pipeline as fp
import plan_estudio as pe
import redactor_adelanto as ra
import vicio_inoperancia as vi

FALLOS = []
AQUI = os.path.dirname(os.path.abspath(__file__))


def ok(cond, que):
    print(f"   {'PASA ' if cond else 'FALLA'}  {que}")
    if not cond:
        FALLOS.append(que)


def encender(si: bool):
    if si:
        os.environ["INOPERANCIA_POR_VICIO"] = "todos"
    else:
        os.environ.pop("INOPERANCIA_POR_VICIO", None)


# ═══ EL ACERVO FALSO ═════════════════════════════════════════════════════════
class _Punto:
    def __init__(self, payload):
        self.payload = payload


class _Resp:
    def __init__(self, pts):
        self.points = pts


def _tesis(reg, rubro, tipo="Tesis Aislada", instancia="Tribunales Colegiados de Circuito",
           epoca="11a."):
    return {"registro": reg, "rubro": rubro, "texto": "texto " + reg, "tipo": tipo,
            "instancia": instancia, "vincula": tipo == "Jurisprudencia",
            "localizacion": f"[{'J' if tipo == 'Jurisprudencia' else 'TA'}]; {epoca} Época"}


POZO = [
    _tesis("900001", "AGRAVIOS INOPERANTES. LO SON LOS QUE NO COMBATEN LAS CONSIDERACIONES "
                     "EN LA REVISIÓN FISCAL"),
    _tesis("900002", "AGRAVIOS INOPERANTES. LO SON AQUELLOS QUE SE REFIEREN A CUESTIONES NO "
                     "ADUCIDAS EN LA DEMANDA"),
    _tesis("900003", "CONCEPTOS DE VIOLACIÓN INOPERANTES. LO SON LOS NOVEDOSOS",
           tipo="Jurisprudencia", instancia="Segunda Sala", epoca="10a."),
    _tesis("900004", "AGRAVIOS INOPERANTES EN LA QUEJA. LOS QUE NO COMBATEN EL AUTO"),
    _tesis("900005", "AGRAVIOS INOPERANTES. SUPERADA", tipo="Jurisprudencia",
           instancia="Segunda Sala"),
    _tesis("900006", "AGRAVIOS NOVEDOSOS. ORIENTADORA DE COLEGIADO VIEJA", epoca="9a."),
]


class QdrantFalso:
    """Sin filtro, el pozo entero; con el filtro por registro (la pertinencia al
    argumento), las pedidas AL REVÉS del pozo: así se ve que el argumento
    reordena."""
    def __init__(self):
        self.llamadas = []

    def query_points(self, collection_name, query, using, limit, query_filter=None,
                     with_payload=True):
        self.llamadas.append(query_filter)
        if query_filter is None:
            return _Resp([_Punto(dict(p)) for p in POZO[:limit]])
        regs = list(query_filter.must[0].match.any)
        pts = [p for p in POZO if p["registro"] in regs]
        return _Resp([_Punto(dict(p)) for p in reversed(pts)][:limit])

    def scroll(self, collection_name, scroll_filter, limit, with_payload=True):
        regs = list(scroll_filter.must[0].match.any)
        return ([_Punto(dict(p)) for p in POZO if p["registro"] in regs], None)


async def embed_falso(texto):
    return [float(len(texto or ""))]


# El sello de vigencia falso: 900005 la perdió.
_de_original = rag._vig.de
rag._vig.de = lambda reg: ({"estado": "superada", "parcial": False}
                           if str(reg) == "900005" else _de_original(reg))


def correr(c):
    return asyncio.get_event_loop().run_until_complete(c)


# ═══════════════════════════════════════════════════════════════════════════
print("\n1 · EL CATÁLOGO Y EL VICIO DE UN TEXTO")
_razones = set(pe._RAZONES_DE_INOPERANTE)
ok(_razones <= set(rag.CONSULTA_POR_VICIO), "toda razón de inoperancia del plan tiene consulta")
ok({"ineficaz", "inatendible", "innecesario"} <= set(rag.CONSULTA_POR_VICIO),
   "y las calificativas de técnica también")
ok(all(2 <= len(v) <= 3 for v in rag.CONSULTA_POR_VICIO.values()),
   "cada vicio con 2 o 3 formulaciones de rubro")
ok(set(vi.VICIOS) == set(rag.CONSULTA_POR_VICIO), "catálogo y consultas con las mismas claves")
ok(vi.VICIOS_DE_FORMA == {k for k, v in pe.RAZONES.items() if v.get("forma")},
   "los vicios de forma son los de plan_estudio.RAZONES")
for texto, esperado in (
        ("El agravio es novedoso porque no se hizo valer en la demanda", "novedoso"),
        ("parte de una premisa falsa", "falsa_premisa"),
        ("se limita a reiterar los conceptos de violación", "reitera_sin_combatir"),
        ("no combate la consideración toral", "no_combate"),
        ("es genérico y dogmático", "generico"),
        ("aun cuando fuera fundado, no trasciende al fallo", "fundado_insuficiente"),
        ("ataca lo dicho a mayor abundamiento", "ataca_accesoria"),
        ("se hace depender de los ya desestimados", "deriva_de_desestimado"),
        ("la violación procesal no se preparó (art. 171)", "procesal_171_172"),
        ("ya se decidió en el amparo anterior: cosa juzgada", "cosa_juzgada_amparo_previo"),
        ("la responsable valoró mal la pericial", "")):
    ok(vi.vicio_de_texto(texto) == esperado, f"«{texto[:40]}» → {esperado or 'ninguno'}")
ok(vi.vicio_de("inoperante") == "no_combate", "inoperante sin más: no_combate")
ok(vi.vicio_de("inoperante", "", "novedoso") == "novedoso", "el vicio declarado manda")
ok(vi.vicio_de("ineficaz", "es novedoso") == "ineficaz", "las otras calificativas son su vicio")
ok(vi.vicio_de("fundado") == "" and vi.vicio_de("infundado") == "", "el fondo no pide tesis de técnica")

# ═══════════════════════════════════════════════════════════════════════════
print("\n2 · APAGADA: TODO COMO ANTES")
encender(False)
ok(not vi.activa(), "fuera de una cuenta de casa, apagada por omisión")
_p3 = fp.prompt_problemas("ACTO", "CONC", False, "amparo_directo", 0)
ok('Si adviertes un impedimento técnico que llevaría a inoperancia, ponlo en\n'
   '"impedimento" como {"motivo": "inoperancia", "explicacion": "..."}.\n'
   'Y si adviertes lo contrario' in _p3, "fase 3: el renglón del impedimento, literal")
ok(vi.pregunta_sintetica("impedimento", "inoperancia",
                         {"motivo": "inoperancia", "vicio": "novedoso", "explicacion": "x"})
   == "¿Inoperancia: x?", "la pregunta sintética de siempre")
ok(vi.pregunta_sintetica("apoyo", "sustento", {"explicacion": "y"}) == "¿Sustento: y?",
   "y la del apoyo")
C = [f6.Criterio(problema="¿Uno?", sentido="fundado", razonamiento="r", jerarquia="principal"),
     f6.Criterio(problema="¿Dos?", sentido="inoperante", razonamiento="no combate",
                 jerarquia="accesorio")]
for v in ("v1", "v4"):
    _m = f6.Material(tipo_asunto="amparo_directo", materia="civil", formato="estandar",
                     problemas=[{"pregunta": "¿Uno?"}, {"pregunta": "¿Dos?"}],
                     n_planteamientos=2, variante=v)
    _pe = f6.prompt_estudio("A", "C", C, _m)
    ok(_pe.count("58/2010") == 2, f"{v}: la 2a./J. 58/2010 sigue donde estaba (2 veces)")
    if v == "v1":
        ok("SE CITA UNA TESIS SOBRE LA INOPERANCIA" in _pe, "v1: la regla de antes")
    else:
        ok("inoperancia se cita sólo si hace falta para sostenerla.\n" in _pe, "v4: la regla de antes")
# La búsqueda por calificativa, la de antes: 4, sin vicio, sin filtros.
_q = QdrantFalso()
_viejas = correr(rag.tesis_de_la_calificativa(_q, embed_falso, "inoperante", "¿algo?"))
ok([t["registro"] for t in _viejas] == ["900001", "900002", "900003", "900004"],
   "la búsqueda de antes: las 4 primeras tal cual (también la de revisión fiscal)")
ok(all(t.get("tecnica") and t.get("de_la_calificativa") == "inoperante" and "vicio" not in t
       for t in _viejas), "marcadas como antes, sin vicio")
ok(correr(rag.tesis_de_la_calificativa(_q, embed_falso, "fundado", "x")) == [],
   "fundado: nada")
_mat0 = f6.Material(tesis=[{"registro": "1", "rubro": "R"}])
ok(correr(ra.material_con_tesis_del_vicio(_q, embed_falso, None, _mat0, C[1:])) is _mat0,
   "antes del plan: el MISMO material")

# ═══════════════════════════════════════════════════════════════════════════
print("\n3 · ENCENDIDA: LOS PROMPTS")
encender(True)
ok(vi.activa(), "la variable de entorno la enciende")
_p3 = fp.prompt_problemas("ACTO", "CONC", False, "amparo_directo", 0)
ok('"vicio"' in _p3 and "no_combate:" in _p3 and "novedoso:" in _p3,
   "fase 3: pide el vicio con las claves descritas")
ok("procesal_171_172" in _p3 and "reitera_sin_combatir" not in _p3,
   "amparo directo: el 171-172 sí, la reiteración de la instancia no")
_p3r = fp.prompt_problemas("ACTO", "CONC", True, "queja", 0)
ok("reitera_sin_combatir" in _p3r and "procesal_171_172" not in _p3r
   and "adhesivo_fuera_182" not in _p3r, "queja: al revés")
ok('"motivo", que vale siempre\n"inoperancia"' in _p3, "el motivo sigue siendo «inoperancia» (la pantalla lo pinta)")
_q_n = vi.pregunta_sintetica("impedimento", "inoperancia",
                             {"motivo": "inoperancia", "vicio": "novedoso", "explicacion": "x"})
ok(_q_n.startswith("¿Inoperancia porque el planteamiento plantea una cuestión que no se hizo")
   and _q_n.endswith(": x?"), "la pregunta sintética nombra el vicio")
ok(vi.pregunta_sintetica("impedimento", "inoperancia",
                         {"explicacion": "parte de una premisa falsa"}).startswith(
    "¿Inoperancia porque el planteamiento parte de una premisa falsa"),
   "sin vicio declarado, el de la explicación")
ok(vi.pregunta_sintetica("impedimento", "inoperancia", {"explicacion": "z"}) == "¿Inoperancia: z?",
   "sin vicio reconocible, la de siempre")
for v in ("v1", "v2", "v3", "v4"):
    for tipo, rec in (("amparo_directo", False), ("revision_fiscal", True)):
        _m = f6.Material(tipo_asunto=tipo, materia="civil", formato="estandar",
                         problemas=[{"pregunta": "¿Uno?"}, {"pregunta": "¿Dos?"}],
                         n_planteamientos=2, variante=v)
        _pe = f6.prompt_estudio("A", "C", C, _m, es_recurso=rec)
        ok("58/2010" not in _pe and "2a./J." not in _pe,
           f"{v}/{tipo}: sin la clave real en el prompt")
        ok("nunca de memoria" in _pe.lower() or "NUNCA se cita de memoria" in _pe,
           f"{v}/{tipo}: la tesis de la inoperancia nunca de memoria")
_pe1 = f6.prompt_estudio("A", "C", C, f6.Material(tipo_asunto="amparo_directo", variante="v1"))
ok("SE CITA UNA TESIS SOBRE LA INOPERANCIA" not in _pe1, "v1: ya no manda citar aunque no haya")

# ═══════════════════════════════════════════════════════════════════════════
print("\n4 · LOS FILTROS Y EL ORDEN")
rag._CACHE_VICIO = None
_q = QdrantFalso()
_ad = correr(rag.tesis_de_la_calificativa(_q, embed_falso, "inoperante", "", 6, vicio="novedoso",
                                          argumento="no se planteó en la demanda",
                                          tipo_asunto="amparo_directo"))
_regs = [t["registro"] for t in _ad]
ok("900005" not in _regs, "la que perdió vigencia, fuera")
ok("900001" not in _regs and "900004" not in _regs,
   "amparo directo: fuera las de revisión fiscal y de queja")
ok(_regs and _regs[0] == "900003", "la jurisprudencia de la Sala, primera entre las pertinentes")
ok(_regs.index("900006") > _regs.index("900002") if "900006" in _regs else True,
   "la 9a. Época después de la 11a. a igual jerarquía")
ok(all(t.get("tecnica") and t.get("vicio") == "novedoso"
       and t.get("de_la_calificativa") == "inoperante" for t in _ad),
   "marcadas: técnica, vicio y calificativa")
ok(any(f is not None for f in _q.llamadas), "la pertinencia al argumento se pidió (filtro por registro)")
rag._CACHE_VICIO = None
_rf = correr(rag.tesis_de_la_calificativa(QdrantFalso(), embed_falso, "inoperante", "", 6,
                                          vicio="no_combate", argumento="x",
                                          tipo_asunto="revision_fiscal"))
ok("900001" in [t["registro"] for t in _rf], "revisión fiscal: la de revisión fiscal sí")
ok(rag.apta_para_el_recurso("AGRAVIOS INOPERANTES EN EL AMPARO DIRECTO EN REVISIÓN", "amparo_revision")
   and not rag.apta_para_el_recurso("AGRAVIOS INOPERANTES EN EL AMPARO DIRECTO EN REVISIÓN",
                                    "amparo_directo"),
   "«amparo directo en revisión» es de la revisión")
ok(rag.apta_para_el_recurso("QUEJOSO. SU INTERÉS", "amparo_directo"), "«quejoso» no es la queja")
ok(rag.apta_para_el_recurso("AGRAVIOS INOPERANTES", ""), "sin tipo, no se filtra")
# La evaluación excluye lo publicado desde el corte: el filtro de siempre.
import contexto_taller as ct
_filtro_original = ct.filtrar_tesis
ct.filtrar_tesis = lambda ts: [t for t in ts if t.get("registro") != "900003"]
rag._CACHE_VICIO = None
_ev = correr(rag.tesis_de_la_calificativa(QdrantFalso(), embed_falso, "inoperante", "", 6,
                                          vicio="novedoso", argumento="x",
                                          tipo_asunto="amparo_directo"))
ok("900003" not in [t["registro"] for t in _ev], "contexto_taller.filtrar_tesis se aplica")
ct.filtrar_tesis = _filtro_original
# Una caída de Qdrant no tumba nada.


class QdrantRoto(QdrantFalso):
    def query_points(self, *a, **k):
        raise RuntimeError("caído")


rag._CACHE_VICIO = None
ok(correr(rag.tesis_de_la_calificativa(QdrantRoto(), embed_falso, "inoperante", "", 4,
                                       vicio="generico", argumento="x")) == [],
   "Qdrant caído: [] sin lanzar")

# ═══════════════════════════════════════════════════════════════════════════
print("\n5 · ANTES DEL PLAN: UNA COPIA")
rag._CACHE_VICIO = None


class _Fases:
    problemas = [{"pregunta": "¿Dos?", "combate": "lo novedoso", "resolvio": "algo",
                  "impedimento": {"motivo": "inoperancia", "vicio": "novedoso", "explicacion": "e"}}]


class _Enc:
    tipo_asunto = "amparo_directo"
    suplencia = None


class _R:
    fases = _Fases()
    encargo = _Enc()


_orig = {"registro": "900002", "rubro": "AGRAVIOS INOPERANTES. LO SON AQUELLOS QUE SE REFIEREN "
                                         "A CUESTIONES NO ADUCIDAS EN LA DEMANDA"}
_mat = f6.Material(tesis=[dict(_orig), {"registro": "7", "rubro": "FONDO"}], tipo_asunto="amparo_directo")
_antes = [dict(t) for t in _mat.tesis]
_crit = [f6.Criterio(problema="¿Dos?", sentido="inoperante", razonamiento="", jerarquia="accesorio")]
_nuevo = correr(ra.material_con_tesis_del_vicio(QdrantFalso(), embed_falso, _R(), _mat, _crit))
ok(_nuevo is not _mat, "devuelve otra instancia")
ok(_mat.tesis == _antes, "el material de la sesión queda INTACTO")
_vic = [t for t in _nuevo.tesis if t.get("vicio")]
ok(1 <= len(_vic) <= ra.CUPO_TESIS_POR_VICIO and all(t.get("vicio") == "novedoso" for t in _vic),
   "1-2 tesis del vicio declarado en la fase 3 (novedoso)")
ok(len({t["registro"] for t in _nuevo.tesis}) == len(_nuevo.tesis), "sin duplicados")
_ya = [t for t in _nuevo.tesis if t["registro"] == "900002"]
ok(len(_ya) == 1, "la que ya estaba se marca en su sitio, no se duplica")
_idx = pe.indice_material(_nuevo)
ok(all(t["registro"] in {x["registro"] for x in _idx["tesis"]} for t in _vic),
   "las del vicio entran al índice del plan")
ok(pe.huella_indice(_idx) != pe.huella_indice(pe.indice_material(_mat)),
   "y cambian la huella (el plan se calcula con ellas)")
# La razón del secretario manda sobre la fase 3.
rag._CACHE_VICIO = None
_crit2 = [f6.Criterio(problema="¿Dos?", sentido="inoperante",
                      razonamiento="es genérico y dogmático", jerarquia="accesorio")]
_n2 = correr(ra.material_con_tesis_del_vicio(QdrantFalso(), embed_falso, _R(), _mat, _crit2))
ok({t.get("vicio") for t in _n2.tesis if t.get("vicio")} == {"generico"},
   "el vicio de la razón del secretario manda")
# Suplencia confirmada: los vicios de forma no traen tesis.
import suplencia as _sp
_conf_orig = _sp.confirmada
_sp.confirmada = lambda s: True
rag._CACHE_VICIO = None
_n3 = correr(ra.material_con_tesis_del_vicio(QdrantFalso(), embed_falso, _R(), _mat, _crit2))
ok(_n3 is _mat, "con suplencia confirmada, el genérico no trae tesis (inoperancia de forma prohibida)")
_sp.confirmada = _conf_orig
# Fundado: nada que añadir.
ok(correr(ra.material_con_tesis_del_vicio(QdrantFalso(), embed_falso, _R(), _mat,
                                          [C[0]])) is _mat, "un criterio fundado no trae nada")
# Lo que la razón cita y falta se trae por registro.
rag._CACHE_VICIO = None
_crit3 = [f6.Criterio(problema="¿Dos?", sentido="inoperante",
                      razonamiento="es novedoso, como dice la tesis (registro 900006)", jerarquia="accesorio")]
_mat_sin = f6.Material(tesis=[{"registro": "7", "rubro": "FONDO"}])


class QdrantSinPozo(QdrantFalso):
    def query_points(self, *a, **k):
        return _Resp([])


_n4 = correr(ra.material_con_tesis_del_vicio(QdrantSinPozo(), embed_falso, _R(), _mat_sin, _crit3))
ok("900006" in [t["registro"] for t in _n4.tesis], "el registro que cita la razón se trae si falta")
_crit5 = [f6.Criterio(problema="¿Dos?", sentido="inoperante",
                      razonamiento="es novedoso (900005)", jerarquia="accesorio")]
_n5 = correr(ra.material_con_tesis_del_vicio(QdrantSinPozo(), embed_falso, _R(), _mat_sin, _crit5))
ok(_n5 is _mat_sin, "salvo que haya perdido vigencia")
_crit7 = [f6.Criterio(problema="¿Dos?", sentido="inoperante",
                      razonamiento="es novedoso; reclamó 900006 pesos", jerarquia="accesorio")]
_n7 = correr(ra.material_con_tesis_del_vicio(QdrantSinPozo(), embed_falso, _R(), _mat_sin, _crit7))
ok(_n7 is _mat_sin, "una cifra suelta no se trae como registro")


# Un acervo lento no detiene el resolver.
class QdrantLento(QdrantFalso):
    async def _lento(self):
        await asyncio.sleep(5)
        return _Resp([])

    def query_points(self, *a, **k):
        return self._lento()


rag._CACHE_VICIO = None
_n6 = correr(ra.material_con_tesis_del_vicio(QdrantLento(), embed_falso, _R(), _mat, _crit,
                                             tope_s=0.2))
ok(_n6 is _mat, "con tope vencido, el mismo material")

# ═══════════════════════════════════════════════════════════════════════════
print("\n6 · RAZONAR NO MUTA; LOS GEMELOS PASAN LA COPIA (AST)")
_src = open(os.path.join(AQUI, "main.py"), encoding="utf-8").read()
_arbol = ast.parse(_src)
_fn = {n.name: n for n in ast.walk(_arbol) if isinstance(n, ast.AsyncFunctionDef)}
_raz = _fn["taller_razonar"]
_raz_src = ast.get_source_segment(_src, _raz)
_i_act = _raz_src.find("if _vi_r.activa():")
_i_else = _raz_src.find("else:", _i_act)
_rama = _raz_src[_i_act:_i_else]
ok(_i_act > 0 and "material = _copy_r.copy(material)" in _rama, "razonar: copia antes de tocar")
ok(_rama.find("material = _copy_r.copy(material)") < _rama.find("material.tesis ="),
   "razonar: la copia va antes de cambiar sus tesis")
ok("ses[" not in _rama, "razonar: la rama nueva no toca la sesión")
ok("vicio=_vic_r" in _rama and "vicio_y_argumento" in _rama,
   "razonar: busca por el vicio, con el mismo cálculo que el resolver")


def _llamadas(fn, nombre):
    return [n for n in ast.walk(fn) if isinstance(n, ast.Call)
            and ((isinstance(n.func, ast.Attribute) and n.func.attr == nombre)
                 or (isinstance(n.func, ast.Name) and n.func.id == nombre))]


# Sólo las del módulo: `_trabajar` vive dentro del gemelo de flujo.
_de_modulo = [n for n in _arbol.body if isinstance(n, ast.AsyncFunctionDef)]
_gemelos = [f for f in _de_modulo
            if _llamadas(f, "_taller_plan_para") and (_llamadas(f, "resolver_en_vivo")
                                                     or _llamadas(f, "resolver"))]
ok(len(_gemelos) >= 1, f"los gemelos del resolver ({len(_gemelos)} funciones)")
_n_vivo = _n_plano = 0
for f in _gemelos:
    _s = ast.get_source_segment(_src, f)
    for llamada, var in (("resolver_en_vivo(", "_mat_v"), ("_ra.resolver(", "_mat_v")):
        if llamada in _s:
            _i_m = _s.find("material_con_tesis_del_vicio(")
            _i_p = _s.find("_taller_plan_para(user_email, numero, r, _ses_v")
            _i_r = _s.find(llamada)
            _ok = 0 < _i_m < _i_p < _i_r and _s[_i_r:_i_r + 120].count(var) == 1
            ok(_ok, f"{f.name}: tesis del vicio → plan con la copia → {llamada[:-1]} con la copia")
            if llamada.startswith("resolver_en_vivo"):
                _n_vivo += 1
            else:
                _n_plano += 1
ok(_n_vivo == 1 and _n_plano == 1, "los dos gemelos, uno de flujo y uno plano")

# ═══════════════════════════════════════════════════════════════════════════
encender(False)
print()
if FALLOS:
    print(f"RESULTADO: {len(FALLOS)} COMPROBACIONES FALLAN")
    for f in FALLOS:
        print("   ·", f)
    sys.exit(1)
print("RESULTADO: TODAS LAS COMPROBACIONES PASAN")
