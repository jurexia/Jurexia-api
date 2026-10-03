"""Los fallos de la revisión adversarial de la inoperancia por vicio — 3-oct-2026.

Sin red (Qdrant y embeddings falsos), con la bandera `inoperancia_por_vicio`
encendida por variable de entorno. Cada bloque falla con el código de antes:

  1 · la clave del plan PEDIDO (y del precálculo) casa con la del resolver
      cuando hay tesis del vicio (`main._taller_plan_pedido`, extraída por AST);
  2 · la caché por worker de `tesis_del_vicio` no guarda lo degradado por una
      caída pasajera de Qdrant o del embedding;
  3 · `vicio_de_texto` no casa números sueltos ni «reproduc» genérico, y el
      vicio que declaró la fase 3 manda sobre el párrafo largo de la razón;
      la directriz corta del secretario sigue mandando; el vicio respeta la vía;
  4 · el filtro de vía: «suplencia de la queja» no es el recurso de queja, y
      la apelación o los conceptos de anulación como instancia previa no
      excluyen la tesis;
  5 · en la v4, un problema infundado con vicio declarado trae su tesis;
  6 · `sin_numeracion` respeta los efectos de una ejecutoria transcritos;
  7 · las tesis traídas por registro desde la razón pasan la vía y la
      evaluación;
  8 · /taller/razonar descarta el eco de la otra vía antes de deducir el vicio.

    .venv/bin/python test_inoperancia_revision.py
"""
import ast
import asyncio
import os
import sys

os.environ["INOPERANCIA_POR_VICIO"] = "todos"
os.environ.pop("ESTUDIO_PROMPT", None)

import contexto_taller as ct
import fase6_estudio as f6
import fase6_rag as rag
import fases123_resumenes as fr
import plan_estudio as pe
import redactor_adelanto as ra
import taller_estado as te
import vicio_inoperancia as vi

FALLOS = []
AQUI = os.path.dirname(os.path.abspath(__file__))


def ok(cond, que):
    print(f"   {'PASA ' if cond else 'FALLA'}  {que}")
    if not cond:
        FALLOS.append(que)


def correr(c):
    return asyncio.get_event_loop().run_until_complete(c)


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
    _tesis("900006", "AGRAVIOS NOVEDOSOS. ORIENTADORA DE COLEGIADO", epoca="10a."),
]


class QdrantFalso:
    """Sin filtro, el pozo entero; con filtro por registro, las pedidas al revés.
    `caido` hace fallar todo; `caida_pertinencia`, sólo la búsqueda filtrada."""
    def __init__(self):
        self.llamadas = 0
        self.caido = False
        self.caida_pertinencia = False

    def query_points(self, collection_name, query, using, limit, query_filter=None,
                     with_payload=True):
        self.llamadas += 1
        if self.caido or (query_filter is not None and self.caida_pertinencia):
            raise TimeoutError("lectura vencida")
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


rag._vig.de = lambda reg: {"estado": "vigente", "parcial": False}


class _Fases:
    def __init__(self, problemas):
        self.problemas = problemas
        self.fuentes = ["acto", "escrito"]
        self.avisos = []

    def parrafos_acto(self):
        return ["acto"]

    def parrafos_conceptos(self):
        return ["conceptos"]


class _Enc:
    def __init__(self, tipo="amparo_directo", variante="v4"):
        self.tipo_asunto = tipo
        self.suplencia = None
        self.es_recurso = tipo != "amparo_directo"
        self.conceptos_violacion = ""
        self.variante_estudio = variante


class _R:
    def __init__(self, problemas, tipo="amparo_directo", variante="v4"):
        self.fases = _Fases(problemas)
        self.encargo = _Enc(tipo, variante)


_PROB_NOV = {"pregunta": "¿Dos?", "combate": "lo que no dijo en la demanda", "resolvio": "algo",
             "impedimento": {"motivo": "inoperancia", "vicio": "novedoso", "explicacion": "e"}}

# ═══════════════════════════════════════════════════════════════════════════
print("\n1 · LA CLAVE DEL PLAN PEDIDO CASA CON LA DEL RESOLVER")
_src = open(os.path.join(AQUI, "main.py"), encoding="utf-8").read()
_arbol = ast.parse(_src)
_de_mod = {n.name: n for n in _arbol.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
_lanzadas = []
_g = {"_te": te, "time": __import__("time"), "qdrant_client": QdrantFalso(),
      "_embedding_juris": embed_falso,
      "_taller_plan_cas": lambda email, numero, cambio: (None, "listo"),
      "_taller_plan_lanzar": lambda *a, **k: _lanzadas.append(a)}
exec(compile(ast.Module(body=[_de_mod["_taller_plan_entradas"], _de_mod["_taller_plan_pedido"]],
                        type_ignores=[]), "main.py", "exec"), _g)
_entradas, _pedido = _g["_taller_plan_entradas"], _g["_taller_plan_pedido"]
_segs_orig = pe.segmentos_de
pe.segmentos_de = lambda fases, esc="", es_rec=False, extraidos=None: [
    {"id": "S1", "concepto": 1, "anclas": []}]


def _claves(crit, problemas, suplencia=None):
    """(clave del pedido, clave del resolver, material del resolver)."""
    rag._CACHE_VICIO = None
    r = _R(problemas)
    r.encargo.suplencia = suplencia
    mat = f6.Material(tesis=[{"registro": "7", "rubro": "FONDO", "texto": "x"}],
                      tipo_asunto="amparo_directo")
    ses = {"material": mat}
    out = correr(_pedido("a@b.c", "1/2026", r, ses, {"crit": crit}, contexto="",
                         suplencia=dict(suplencia or {}), conceptos_violacion="", extraidos=[]))
    # El resolver, como los dos gemelos (main.py): copia → plan con la copia.
    _mat_v = correr(ra.material_con_tesis_del_vicio(QdrantFalso(), embed_falso, r, mat, crit))
    _ses_v = ses if _mat_v is mat else {**ses, "material": _mat_v}
    k_res = _entradas(r, _ses_v, crit, contexto="", suplencia=dict(r.encargo.suplencia or {}),
                      conceptos_violacion="", extraidos=[])["clave"]
    ok(ses["material"] is mat and mat.tesis == [{"registro": "7", "rubro": "FONDO", "texto": "x"}],
       "el pedido no toca el material de la sesión")
    return out.get("clave"), k_res, _mat_v


_crit_inop = [f6.Criterio(problema="¿Dos?", sentido="inoperante", razonamiento="",
                          jerarquia="principal")]
_kp, _kr, _mv = _claves(_crit_inop, [_PROB_NOV])
ok(any(t.get("vicio") for t in _mv.tesis), "el resolver trajo tesis del vicio")
ok(_kp and _kp == _kr, "inoperante con vicio: la clave del pedido CASA con la del resolver")
_crit_inf = [f6.Criterio(problema="¿Dos?", sentido="infundado", razonamiento="",
                         jerarquia="principal")]
_kp2, _kr2, _ = _claves(_crit_inf, [_PROB_NOV])
ok(_kp2 and _kp2 == _kr2, "infundado con vicio declarado (v4): también casan")
_kp3, _kr3, _ = _claves(_crit_inop, [_PROB_NOV], suplencia={"aplica": True})
ok(_kp3 and _kp3 == _kr3, "con la suplencia del formulario: casan")
os.environ.pop("INOPERANCIA_POR_VICIO", None)
_kp4, _kr4, _mv4 = _claves(_crit_inop, [_PROB_NOV])
ok(_kp4 == _kr4 and not any(t.get("vicio") for t in _mv4.tesis),
   "bandera apagada: como siempre")
os.environ["INOPERANCIA_POR_VICIO"] = "todos"
pe.segmentos_de = _segs_orig

# ═══════════════════════════════════════════════════════════════════════════
print("\n2 · LA CACHÉ NO GUARDA LO DEGRADADO")
rag._CACHE_VICIO = None
_q = QdrantFalso()
_q.caido = True
_v1 = correr(rag.tesis_del_vicio(_q, embed_falso, "no_combate", argumento="x",
                                 tipo_asunto="amparo_directo"))
ok(_v1 == [], "Qdrant caído: [] sin lanzar")
ok(not rag._CACHE_VICIO, "y el vacío NO se guarda")
_q.caido = False
_n0 = _q.llamadas
_v2 = correr(rag.tesis_del_vicio(_q, embed_falso, "no_combate", argumento="x",
                                 tipo_asunto="amparo_directo"))
ok(_v2 and _q.llamadas > _n0, "repuesto Qdrant, el mismo worker vuelve a buscar y trae tesis")
ok(len(rag._CACHE_VICIO) == 1, "lo completo sí se guarda")
_n1 = _q.llamadas
correr(rag.tesis_del_vicio(_q, embed_falso, "no_combate", argumento="x",
                           tipo_asunto="amparo_directo"))
ok(_q.llamadas == _n1, "y la segunda vez sale de la caché")
rag._CACHE_VICIO = None
_q2 = QdrantFalso()
_q2.caida_pertinencia = True
_v3 = correr(rag.tesis_del_vicio(_q2, embed_falso, "novedoso", argumento="y",
                                 tipo_asunto="amparo_directo"))
ok(_v3 and not rag._CACHE_VICIO, "sin la pertinencia: se sirven, pero no se guardan")

# ═══════════════════════════════════════════════════════════════════════════
print("\n3 · EL VICIO DE UN TEXTO LARGO")
for texto, esperado in (
        ("no combate la consideración toral, como lo ha reiterado la Suprema Corte (2012073)",
         "no_combate"),
        ("cuyo rubro y texto se reproducen", ""),
        ("la sentencia dictada en el toca 171/2025", ""),
        ("no combate lo relativo al artículo 172 del Código de Procedimientos Civiles",
         "no_combate"),
        ("la documental que obra a foja 182", ""),
        ("así lo sostiene la jurisprudencia 2a./J. 172/2010", ""),
        ("la tercera interesada promovió amparo adhesivo", ""),
        ("innecesario el estudio de la cosa juzgada que alegó la demandada", ""),
        ("la prueba pericial no se preparó en tiempo", ""),
        ("la violación procesal no se preparó (art. 171)", "procesal_171_172"),
        ("no se preparó la violación procesal conforme al artículo 171 de la Ley de Amparo",
         "procesal_171_172"),
        ("en el amparo adhesivo plantea lo que el artículo 182 de la Ley de Amparo no autoriza",
         "adhesivo_fuera_182"),
        ("se limita a reiterar los conceptos de violación", "reitera_sin_combatir"),
        ("reproduce los agravios de la apelación", "reitera_sin_combatir"),
        ("ya se decidió en el amparo anterior: cosa juzgada", "cosa_juzgada_amparo_previo")):
    ok(vi.vicio_de_texto(texto) == esperado, f"«{texto[:55]}» → {esperado or 'ninguno'}")
ok(vi.vicio_de("ineficaz", "", "procesal_171_172") == "ineficaz",
   "la calificativa que no es «inoperante» no la desplaza un vicio")
_largo = ("Es inoperante, porque el quejoso no controvierte la consideración toral de la "
          "responsable, como lo ha reiterado este tribunal; además, la documental que obra a "
          "foja 171 no cambia nada, y la Sala sostuvo que el plazo del artículo 182 del Código "
          "Fiscal había corrido. Por ello, conforme a la jurisprudencia (2012073), cuyo rubro y "
          "texto se reproducen, el planteamiento no puede prosperar y debe desestimarse aquí.")
ok(len(_largo) > ra.RAZON_CORTA_CHARS, "la razón de prueba es un párrafo largo")
_r = _R([_PROB_NOV])
ok(ra.vicio_y_argumento(_r, "¿Dos?", "inoperante", _largo)[0] == "novedoso",
   "razón larga: manda el vicio que declaró la fase 3")
ok(ra.vicio_y_argumento(_r, "¿Dos?", "inoperante", "es genérico y dogmático")[0] == "generico",
   "directriz corta del secretario: manda ella")
_r_sin = _R([{"pregunta": "¿Dos?"}])
ok(ra.vicio_y_argumento(_r_sin, "¿Dos?", "inoperante", _largo)[0] == "no_combate",
   "sin vicio declarado, el de la razón larga (ya sin falsos positivos)")
_r_rf = _R([{"pregunta": "¿Dos?", "impedimento": {"vicio": "adhesivo_fuera_182",
                                                  "explicacion": "x"}}], tipo="revision_fiscal")
ok(ra.vicio_y_argumento(_r_rf, "¿Dos?", "inoperante", "")[0] == "no_combate",
   "revisión fiscal: el adhesivo no cabe en la vía")
_r_ad = _R([{"pregunta": "¿Dos?"}])
ok(ra.vicio_y_argumento(_r_ad, "¿Dos?", "inoperante",
                        "reitera los argumentos de la demanda")[0] == "no_combate",
   "amparo directo: la reiteración de la instancia no cabe")
ok(vi.cabe_en_la_via("procesal_171_172", "amparo_directo")
   and not vi.cabe_en_la_via("procesal_171_172", "queja")
   and vi.cabe_en_la_via("reitera_sin_combatir", "queja")
   and vi.cabe_en_la_via("procesal_171_172", ""), "cabe_en_la_via, la partición de la fase 3")

# ═══════════════════════════════════════════════════════════════════════════
print("\n4 · EL FILTRO DE VÍA")
_TIPOS = ("amparo_directo", "amparo_revision", "queja", "revision_fiscal")


def _aptas(rubro):
    return [t for t in _TIPOS if rag.apta_para_el_recurso(rubro, t)]


ok(_aptas("CONCEPTOS DE VIOLACIÓN INOPERANTES. TIENEN ESTA CALIDAD SI SE REFIEREN A CUESTIONES "
          "NO ADUCIDAS EN LOS AGRAVIOS DEL RECURSO DE APELACIÓN Y NO SE DEJÓ SIN DEFENSA AL "
          "APELANTE") == list(_TIPOS), "1a./J. 12/2008: la apelación como instancia previa no excluye")
ok("amparo_directo" in _aptas("CONCEPTOS DE VIOLACIÓN INOPERANTES. LO SON LOS QUE NO SE HICIERON "
                              "VALER COMO CONCEPTOS DE ANULACIÓN"),
   "los conceptos de anulación no planteados: apta para el directo")
ok(_aptas("AGRAVIOS INOPERANTES EN APELACIÓN. LO SON LOS QUE NO COMBATEN") == [],
   "la apelación como el escrito que se califica sí excluye")
for _rub in ("AGRAVIOS INOPERANTES. LO SON LOS QUE NO COMBATEN, SI NO SE ESTÁ EN EL CASO DE "
             "SUPLIR LA DEFICIENCIA DE LA QUEJA",
             "AGRAVIOS INOPERANTES. A FALTA DE SUPLENCIA DE LA QUEJA DEFICIENTE"):
    ok(_aptas(_rub) == list(_TIPOS), f"la suplencia no es el recurso de queja: «{_rub[-40:]}»")
ok(_aptas("AGRAVIOS INOPERANTES EN EL RECURSO DE QUEJA. X") == ["queja"],
   "el recurso de queja sigue siendo de la queja")
ok(_aptas("AGRAVIOS INOPERANTES EN EL AMPARO DIRECTO EN REVISIÓN. X") == ["amparo_revision"],
   "«amparo directo en revisión», de la revisión")

# ═══════════════════════════════════════════════════════════════════════════
print("\n5 · LA v4: EL PROBLEMA INFUNDADO CON VICIO DECLARADO")
rag._CACHE_VICIO = None
_mat = f6.Material(tesis=[{"registro": "7", "rubro": "FONDO"}], tipo_asunto="amparo_directo")
_n = correr(ra.material_con_tesis_del_vicio(QdrantFalso(), embed_falso, _R([_PROB_NOV]), _mat,
                                            _crit_inf))
_vics = [t for t in _n.tesis if t.get("vicio")]
ok(_n is not _mat and _vics and all(t["vicio"] == "novedoso"
                                    and t.get("de_la_calificativa") == "inoperante" for t in _vics),
   "v4: el infundado con impedimento «novedoso» trae la tesis de ese vicio")
_n_v1 = correr(ra.material_con_tesis_del_vicio(QdrantFalso(), embed_falso,
                                               _R([_PROB_NOV], variante="v1"), _mat, _crit_inf))
ok(_n_v1 is _mat, "v1: como antes, el infundado no trae nada")
_n_sin = correr(ra.material_con_tesis_del_vicio(QdrantFalso(), embed_falso,
                                                _R([{"pregunta": "¿Dos?"}]), _mat, _crit_inf))
ok(_n_sin is _mat, "v4 sin vicio declarado: nada")

# ═══════════════════════════════════════════════════════════════════════════
print("\n6 · LOS EFECTOS DE LA EJECUTORIA CONSERVAN SU NÚMERO")
_ant = ["1. Por escrito presentado el quince de marzo, el actor demandó.",
        "2. Este tribunal concedió el amparo para los efectos siguientes:",
        "1. Deje insubsistente la sentencia reclamada.",
        "2. Dicte otra en la que valore la pericial.",
        "En cumplimiento, la Sala dictó la sentencia reclamada.",
        "3) Inconforme, el quejoso promovió este amparo."]
_s = fr.sin_numeracion(_ant)
ok(_s == ["Por escrito presentado el quince de marzo, el actor demandó.",
          "Este tribunal concedió el amparo para los efectos siguientes:",
          "1. Deje insubsistente la sentencia reclamada.",
          "2. Dicte otra en la que valore la pericial.",
          "En cumplimiento, la Sala dictó la sentencia reclamada.",
          "Inconforme, el quejoso promovió este amparo."],
   "los antecedentes pierden su número; los efectos transcritos, no")
ok(fr.sin_numeracion(_s) == _s, "aplicarla dos veces da lo mismo")
ok(fr.sin_numeracion(["Para contextualizar, se relatan los siguientes antecedentes:",
                      "1. Por auto de…", "2. Seguido el juicio…"])[1:] == ["Por auto de…",
                                                                           "Seguido el juicio…"],
   "tras la fórmula de entrada, la numeración del modelo se quita")

# ═══════════════════════════════════════════════════════════════════════════
print("\n7 · LAS TRAÍDAS POR REGISTRO PASAN LOS FILTROS")


class QdrantSinPozo(QdrantFalso):
    def query_points(self, *a, **k):
        return _Resp([])


rag._CACHE_VICIO = None
_mat_s = f6.Material(tesis=[{"registro": "7", "rubro": "FONDO"}])
_c_q = [f6.Criterio(problema="¿Dos?", sentido="inoperante",
                    razonamiento="es novedoso (900004)", jerarquia="accesorio")]
_n7 = correr(ra.material_con_tesis_del_vicio(QdrantSinPozo(), embed_falso, _R([_PROB_NOV]),
                                             _mat_s, _c_q))
ok("900004" not in [t["registro"] for t in _n7.tesis],
   "amparo directo: la de la queja citada en la razón no entra")
_c_6 = [f6.Criterio(problema="¿Dos?", sentido="inoperante",
                    razonamiento="es novedoso (900006)", jerarquia="accesorio")]
_filtro = ct.filtrar_tesis
ct.filtrar_tesis = lambda ts: [t for t in ts if t.get("registro") != "900006"]
_n8 = correr(ra.material_con_tesis_del_vicio(QdrantSinPozo(), embed_falso, _R([_PROB_NOV]),
                                             _mat_s, _c_6))
ct.filtrar_tesis = _filtro
ok("900006" not in [t["registro"] for t in _n8.tesis], "la que la evaluación excluye tampoco")
_n9 = correr(ra.material_con_tesis_del_vicio(QdrantSinPozo(), embed_falso, _R([_PROB_NOV]),
                                             _mat_s, _c_6))
ok("900006" in [t["registro"] for t in _n9.tesis], "sin exclusión, la apta sí entra")

# ═══════════════════════════════════════════════════════════════════════════
print("\n8 · RAZONAR: EL ECO DE LA OTRA VÍA, ANTES DEL VICIO")
_raz = ast.get_source_segment(_src, _de_mod["taller_razonar"])
_i_eco = _raz.find("razon_de_la_otra_via(")
_i_vic = _raz.find("vicio_y_argumento(")
ok(0 < _i_eco < _i_vic, "razon_de_la_otra_via corre antes de vicio_y_argumento")
ok("vicio_y_argumento(r, problema, sentido, _dir_vic)" in _raz,
   "y el vicio se deduce de la directriz ya limpia")

# ═══════════════════════════════════════════════════════════════════════════
os.environ.pop("INOPERANCIA_POR_VICIO", None)
print()
if FALLOS:
    print(f"RESULTADO: {len(FALLOS)} COMPROBACIONES FALLAN")
    for f in FALLOS:
        print("   ·", f)
    sys.exit(1)
print("RESULTADO: TODAS LAS COMPROBACIONES PASAN")
