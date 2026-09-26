"""El pilar de la línea del control de convencionalidad — 25-sep-2026.

    .venv/bin/python test_linea_pilar.py

POR QUÉ: David preguntó si los Tribunales Colegiados pueden hacer control
difuso sobre las normas del juicio de origen y el chat le dio como vigentes la
P. IX/2015 y la P. X/2015, ABANDONADAS por la P./J. 2/2022 (reg. 2024159).
Esto prueba, sin red y sin gastar API, que ahora:
  1. sus preguntas reales abren la línea (y las que no son de la línea, no);
  2. la selección trae el estado actual y marca lo abandonado con su sustituta;
  3. el XML dice vigencia, postura y fuerza; y con sólo «jurisprudencia» en el
     selector, nada interamericano;
  4. la cronología del pilar está verificada, con fuente y postura coherente;
  5. el guion de Qdrant en seco arma puntos con ids únicos;
  6. la sonda no rompe nada sin su colección;
  7. el bloque cabe en el presupuesto de tokens.

Qdrant es falso: los párrafos de la Corte IDH salen de los puntos en seco de la
ingesta (si no están, esa parte se omite) y las tesis llevan su rubro real de
la cronología verificada, para medir tokens como en producción.
"""
import asyncio
import importlib.util
import json
import os
import re
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, os.getcwd())
os.environ["COIDH_ACTIVO"] = "admins"
import linea_coidh as lc  # noqa: E402
from qdrant_client.models import MatchAny, MatchValue  # noqa: E402

FALLOS = []


def ok(cond, que):
    print(f"   {'PASA ' if cond else 'FALLA'}  {que}")
    if not cond:
        FALLOS.append(que)


def correr(coro):
    return asyncio.run(coro)


FIG = "control_convencionalidad"
F = lc.figuras()[FIG]
CRONO = F["cronologia"]
POR_ID = {e["id"]: e for e in CRONO}
REC = {r["registro"]: r for r in F["recepcion_mx"]}

# Las preguntas reales de David (d1_diagnostico.json), tal como las escribió.
COLEGIADOS = ("Hay tesis de la scjn que habilite a ejercer este control difuso a los tribunales colegiados incluso "
              "respecto de normas que no son de su competencia (normas que rigen el acto reclamado)")
LINEA = "¿Cuál es la línea jurisprudencial del control de convencionalidad desde su origen hasta la postura actual de la SCJN?"
PETREAS = ("A partir de precedentes de la cidh y del sistema jurídico nacional. Sería posible conceder el amparo en contra "
           "de reformas estructurales de la constitución (como la reforma judicial) cuando se trata de reformas que "
           "atenta contra cláusula pétras")
ERRATA = ("que establece la ley de amparo y la constitución sobre la procedencia del amparo contraicnovencionalidades en "
          "la constitución?")


# ═══════════════════════════════════════════════════════════════ Qdrant falso
class ColeccionAusente(Exception):
    pass


class QdrantFalso:
    def __init__(self, cols, sonda=None):
        self.col = cols
        self.sonda = sonda
        self.llamadas = []

    def _pts(self, nombre):
        if nombre not in self.col:
            raise ColeccionAusente(f"Not found: Collection `{nombre}` doesn't exist!")
        return self.col[nombre]

    @staticmethod
    def _cumple(pl, flt):
        for c in (getattr(flt, "must", None) or []):
            v = pl.get(c.key)
            vals = v if isinstance(v, list) else [v]
            if isinstance(c.match, MatchAny) and not any(x in c.match.any for x in vals):
                return False
            if isinstance(c.match, MatchValue) and c.match.value not in vals:
                return False
        return True

    async def scroll(self, collection_name, scroll_filter=None, limit=10, **kw):
        self.llamadas.append(("scroll", collection_name))
        return [p for p in self._pts(collection_name) if self._cumple(p.payload, scroll_filter)][:limit], None

    async def retrieve(self, collection_name, ids, **kw):
        self.llamadas.append(("retrieve", collection_name))
        quiero = {str(i) for i in ids}
        return [p for p in self._pts(collection_name) if str(p.id) in quiero]

    async def count(self, collection_name, **kw):
        self.llamadas.append(("count", collection_name))
        self._pts(collection_name)
        return SimpleNamespace(count=len(self.col[collection_name]))

    async def query_points(self, collection_name, query=None, using=None, limit=10, score_threshold=None, **kw):
        self.llamadas.append(("query_points", collection_name))
        self._pts(collection_name)
        pts = [p for p in (self.sonda or []) if score_threshold is None or p.score >= score_threshold]
        return SimpleNamespace(points=pts[:limit])


F0 = Path(os.getenv("COIDH_F0", "/Users/josedavidalcantarmendoza/Documents/IUREXIA-MAC/reingesta/coidh/f0"))
RUTA_PUNTOS = F0 / "puntos_seco.jsonl"
PUNTOS = []
if RUTA_PUNTOS.exists():
    with RUTA_PUNTOS.open(encoding="utf-8") as fh:
        for linea in fh:
            o = json.loads(linea)
            PUNTOS.append(SimpleNamespace(id=o["id"], payload=o["payload"]))
POR_LLAVE = {}
for p in PUNTOS:
    POR_LLAVE.setdefault(p.payload["llave"], []).append(p)

# Las 59 tesis de la recepción con su rubro REAL (el título verificado de la
# cronología o el abreviado de la ficha) y su id del acervo.
TESIS = []
_rubro = {e.get("registro"): e.get("titulo") for e in CRONO if e.get("registro")}
for r in F["recepcion_mx"]:
    TESIS.append(SimpleNamespace(id=r["qdrant_id"], payload={
        "registro": r["registro"], "clave_tesis": r["clave"], "rubro": _rubro.get(r["registro"]) or r["rubro_abreviado"],
        "texto": "Texto de prueba.", "instancia": r["instancia"], "tipo": r["tipo"]}))
NORMAS = [SimpleNamespace(id=c["qdrant_id"], payload={"texto": c["extracto"], "ref": c["ref_acervo"]})
          for c in F["constitucion_mx"]]
DOCTRINA = [SimpleNamespace(id=e["en_acervo"]["id"], payload={
    "texto": e.get("extracto") or "", "autor": e.get("organo"), "obra": e.get("titulo"), "anio": 2014,
    "url_oficial": e.get("url_oficial"), "pagina_impresa": 1})
    for e in CRONO if (e.get("en_acervo") or {}).get("coleccion") == "doctrina"]


def qdrant(coidh=True, **extra):
    cols = {"jurisprudencia_nacional_v3": TESIS, "bloque_constitucional": NORMAS, "doctrina": DOCTRINA}
    if coidh and PUNTOS:
        cols["coidh"] = PUNTOS
    cols.update(extra)
    return QdrantFalso(cols)


def tesis_a_dict(pid, pl, silo):
    return dict(id=str(pid), score=1.0, texto=pl.get("rubro") or "", registro=pl.get("registro"), silo=silo)


def norma_a_dict(pid, pl, silo):
    return dict(id=str(pid), score=1.0, texto=pl.get("texto") or "", ref=pl.get("ref"), silo=silo)


def linea(pregunta, alcance="completa", q=None, ids=()):
    det = lc.pregunta_por_linea(pregunta)
    if det is None:
        return None
    return correr(lc.traer_linea(q or qdrant(), det, "jurisprudencia_nacional_v3", tesis_a_dict=tesis_a_dict,
                                 norma_a_dict=norma_a_dict, alcance=alcance, pregunta=pregunta, ids=ids))


# ═══════════════════════════════════════════════════════════════ 1 · detección
print("\n1 · LAS PREGUNTAS REALES ABREN; LAS AJENAS, NO")
prueba = F["prueba_disparadores"]["pilar"]
for p in prueba["positivas"]:
    ok(lc.pregunta_por_linea(p) is not None, f"abre: «{p[:70]}»")
for p in prueba["negativas"]:
    ok(lc.pregunta_por_linea(p) is None, f"no abre: «{p[:70]}»")
n_pos = sum(1 for p in prueba["positivas"] if lc.pregunta_por_linea(p) is not None)
n_neg = sum(1 for p in prueba["negativas"] if lc.pregunta_por_linea(p) is None)
ok(prueba["resultado"] == f"{n_pos}/{len(prueba['positivas'])} positivas y {n_neg}/{len(prueba['negativas'])} negativas",
   f"prueba_disparadores de la ficha dice lo medido ({prueba['resultado']})")
det = lc.pregunta_por_linea(COLEGIADOS)
ok(det and "tema:control_ex_officio_mx" in det[1] and "mexico" in det[1],
   f"la de los Colegiados: tema control ex officio y México ({det})")
det = lc.pregunta_por_linea("¿Cuál es la postura actual de la SCJN sobre el control difuso?")
ok(det and "actual" in det[1] and "tema_primero" in det[1], f"«postura actual… control difuso»: tema y fase actual ({det})")
ok(lc.pregunta_por_linea(PETREAS)[1] == ["tema:amparo_reformas_constitucionales"]
   and lc.pregunta_por_linea(ERRATA)[1] == ["tema:amparo_reformas_constitucionales"],
   "cláusulas «pétras» y «contraicnovencionalidades»: amparo contra reformas constitucionales, con sus erratas")
ok(lc.pregunta_por_linea("prisión preventiva oficiosa para extorsión") == (FIG, ["tema:prision_preventiva_oficiosa"]),
   "un tema sin fase no cambia su forma (compatibilidad con test_coidh_chat)")


# ═══════════════════════════════════════════════════════════════ 2 · selección
print("\n2 · LA SELECCIÓN: ESTADO ACTUAL Y PAREJAS ABANDONADA/SUSTITUTA")
det = lc.pregunta_por_linea(LINEA)
sel = lc.seleccionar(det[0], det[1], pregunta=LINEA)
for reg in ("2024159", "2006224", "2006225", "2008148", "2030517"):
    ok(reg in sel["registros"], f"«{LINEA[:50]}…» trae {reg} ({REC[reg]['clave']})")
for reg in ("2009816", "2009817"):
    ok(reg in sel["registros"] and REC[reg]["vigencia"] == "abandonada" and REC[reg]["reemplazo"] == "2024159",
       f"{reg} entra ABANDONADA con su sustituta 2024159")
ok({"C-470|s|118", "C-482|s|176", "C-482|s|301", "C-482|s|303"} <= {h["llave"] for h in sel["hitos"]},
   "la fase «actual» con México trae Tzompaxtle ¶118 y García Rodríguez ¶176, 301 y 303")
ok({"DOF-2024-10-31-inimpugnabilidad", "DOF-2024-12-31-art19-literalidad", "SCJN-AG-2-2024",
    "SCJN-impedimento-60-2025"} <= set(sel["cronologia"]),
   "y las reformas de 2024, el AG 2/2024 y el expediente de recepción 3/2023 (pendiente)")
sel_c = lc.seleccionar(FIG, lc.pregunta_por_linea(COLEGIADOS)[1], pregunta=COLEGIADOS)
ok(sel_c["registros"][0] == "2024159", f"la de los Colegiados trae PRIMERO la P./J. 2/2022 ({sel_c['registros'][:4]})")
ok({"2009816", "2009817"} <= set(sel_c["registros"]), "y las P. IX y X/2015, marcadas abandonadas")
# Al revés: si entra sólo la abandonada, entra su sustituta.
sel_x = lc.seleccionar(FIG, ["tema:control_ex_officio_mx"], pregunta="P. X/2015", ids=["2009817"])
ok("2009817" in sel_x["registros"] and "2024159" in sel_x["registros"], "si entra la abandonada, entra su sustituta")
ok(set(sel_x["registros"]) >= {"2009816", "2009817", "2024159"}, "y la sustituta trae a las dos que abandona")
ori = lc.seleccionar(FIG, ["origen"], pregunta="¿Dónde nace el control de convencionalidad?")
ok("160589" in ori["registros"] and len(ori["registros"]) <= 6,
   f"el origen trae pocas tesis, con la P. LXVII/2011, tesis madre de la recepción ({ori['registros']})")


# ═══════════════════════════════════════════════════════════════ 3 · el XML
print("\n3 · EL XML: VIGENCIA, POSTURA Y FUERZA; «SOLO_MX» SIN NADA INTERAMERICANO")
lin = linea(LINEA)
x = lin["xml"] if lin else ""
t_2009817 = re.search(r'<tesis [^>]*registro="2009817"[^>]*>', x)
ok(t_2009817 and 'vigencia="abandonada"' in t_2009817.group(0) and 'reemplazo="2024159"' in t_2009817.group(0),
   "<tesis> de la P. X/2015 con vigencia=\"abandonada\" y reemplazo=\"2024159\"")
ok("ABANDONADA por la tesis de registro 2024159" in x, "y el aviso en el cuerpo, antes del rubro")
t_2024159 = re.search(r'<tesis [^>]*registro="2024159"[^>]*>', x)
ok(t_2024159 and 'vigencia="vigente"' in t_2024159.group(0) and 'fuerza="jurisprudencia_obligatoria"' in t_2024159.group(0)
   and 'sustituye="2009816,2009817"' in t_2024159.group(0), "la P./J. 2/2022: vigente, obligatoria, sustituye a las dos")
ok('postura="prevalece_constitucion"' in x and re.search(r'<entrada [^>]*vigencia="[a-z_]+"[^>]*fuerza="[a-z_]+"', x),
   "postura y fuerza en tesis y en <cronologia>")
ok(re.search(r'<hito [^>]*postura="ordena_adecuar"', x) is not None, "un hito de García Rodríguez dice que ORDENA")
ok("<cortes>" in x and "2025-09-30" in x and "2026-05-29" in x and "2032443" in x and "2026-09-25" in x,
   "<cortes>: Corte IDH, acervo, API del SJF y DOF, con su fecha")
ok("<pendientes>" in x and "3/2023" in x, "el expediente de recepción 3/2023 como pendiente")
ok("<reglas>" in x and "P./J. 64/2014" in x, "las reglas que tocan (la P./J. 64/2014 frente a quien quiera inaplicar)")
ok(x.index('<cronologia') > x.index("<recepcion_mx>") and re.findall(r'<entrada fecha="([\d-]+)"', x)
   == sorted(re.findall(r'<entrada fecha="([\d-]+)"', x)), "la cronología va en orden de fecha")

sm = linea(LINEA, alcance="solo_mx", q=qdrant(coidh=False))
xs = sm["xml"] if sm else ""
ok(sm is not None and 'modo="solo_mx"' in xs, "con sólo «jurisprudencia» la línea entra en modo solo_mx (sin colección coidh)")
ok(not sm["hitos"] and "<hito " not in xs and "<supervision" not in xs and "<precision_historica>" not in xs
   and "<tensiones>" not in xs and "<constitucion_mx>" not in xs, "sin hitos ni supervisiones de la Corte IDH")
tipos = {POR_ID[i]["tipo"] for i in sm["cronologia"]}
ok(tipos <= {"scjn_tesis", "scjn_resolucion", "acuerdo_general", "pleno_regional_tesis", "tcc_tesis"},
   f"su cronología sólo trae resoluciones y tesis mexicanas ({sorted(tipos)})")
ok(not re.search(r'tipo="(cidh|onu|doctrina|scjn_voto|scjn_proyecto|reforma_constitucional|ley|corte_idh\w*)"', xs),
   "ni CIDH, ni ONU, ni doctrina, ni votos, ni proyectos, ni reformas")
ok(all(d["silo"] == "jurisprudencia_nacional_v3" for d in sm["docs"]), "y sus documentos son sólo tesis del acervo")
ok("2024159" in sm["registros"] and "Corte IDH. Caso" not in xs, "con la P./J. 2/2022 y sin citas de la Corte IDH")
smc = linea(COLEGIADOS, alcance="solo_mx", q=qdrant(coidh=False))
ok(smc and smc["registros"][0] == "2024159", "la pregunta de los Colegiados en solo_mx: la P./J. 2/2022 primero")
ok(correr(lc.traer_linea(qdrant(), lc.pregunta_por_linea(LINEA), "jurisprudencia_nacional_v3", alcance=None)) is None,
   "con los dos rubros apagados (alcance None), nada")
ok(lc.alcance_por_fuentes(frozenset({"jurisprudencia"})) == "solo_mx"
   and lc.alcance_por_fuentes(frozenset({"constitucional"})) == "completa"
   and lc.alcance_por_fuentes(frozenset({"federal", "estatal"})) is None
   and lc.alcance_por_fuentes(None) == "completa", "alcance_por_fuentes: completa / solo_mx / None")
if PUNTOS:
    ok(any(d.get("silo") == "doctrina" for d in lin["docs"]) or not any(
        (POR_ID[i].get("en_acervo") or {}).get("coleccion") == "doctrina" for i in lin["cronologia"]),
        "la doctrina de la cronología entra con su [Doc ID] del acervo")


# ═══════════════════════════════════════════════════════════════ 4 · cronología
print("\n4 · LA CRONOLOGÍA DEL PILAR ESTÁ VERIFICADA Y ES COHERENTE")
ids = [e["id"] for e in CRONO]
ok(len(ids) == len(set(ids)) and len(ids) >= 150, f"{len(ids)} entradas, ids únicos")
ok([e["orden"] for e in CRONO] == list(range(1, len(CRONO) + 1)), "orden 1..N sin huecos")
POSTURAS = {"prevalece_constitucion", "interpreta_conforme_o_inaplica", "ordena_adecuar", "recomienda", "propone",
            "mixta", "no_aplica"}
FUERZAS = {"obligatoria", "orientadora", "recomendacion", "voto", "doctrina", "proyecto", "historica", "norma"}
malos = []
for e in CRONO:
    r = lc.entrada_resuelta(FIG, e)
    t, p = e["tipo"], e["postura"]
    if (e.get("verificacion") or {}).get("estado") not in ("verificado", "parcial"):
        malos.append((e["id"], "verificacion"))
    if not (r.get("url_oficial") or e.get("en_acervo")):
        malos.append((e["id"], "sin fuente"))
    if p not in POSTURAS or e["fuerza"] not in FUERZAS or e["tramo"] not in lc.lineas()["figuras"][FIG]["tramos"]:
        malos.append((e["id"], "valor fuera del esquema"))
    if p == "ordena_adecuar" and t not in ("corte_idh_sentencia", "corte_idh_supervision"):
        malos.append((e["id"], "ordena sin ser de la Corte IDH"))
    if t in ("cidh", "onu") and p != "recomienda":
        malos.append((e["id"], "CIDH/ONU que no recomienda"))
    if t in ("corte_idh_voto", "scjn_voto", "doctrina", "scjn_proyecto") and p != "propone":
        malos.append((e["id"], "voto, doctrina o proyecto que no propone"))
    if len((r.get("extracto") or "").split()) > 40:
        malos.append((e["id"], "extracto de más de 40 palabras"))
    if e.get("ref_hito") and e["ref_hito"] not in lc._hitos_por_llave(FIG):
        malos.append((e["id"], "ref_hito sin hito"))
    if e.get("reemplazo") and e["reemplazo"] not in REC:
        malos.append((e["id"], "reemplazo fuera de la recepción"))
ok(not malos, f"verificación, fuente, esquema y postura según el tipo en las {len(CRONO)} entradas ({malos[:5]})")
ok(not any(e.get("tipo") == "cidh" and e["postura"] == "ordena_adecuar" for e in CRONO)
   and not any("voto" in e["tipo"] and e["postura"] == "ordena_adecuar" for e in CRONO),
   "una CIDH nunca «ordena_adecuar»; un voto tampoco")
tramos = F["tramos"]
ok(list(tramos) == ["t1", "t2", "t3", "t4", "t5", "t6"] and sum(len(t["ids"]) for t in tramos.values()) == len(CRONO)
   and all(t["ids"] == sorted(t["ids"], key=lambda i: POR_ID[i]["orden"]) for t in tramos.values()),
   "seis tramos que reparten la cronología entera, cada uno en orden")
frases = [t.get("frases") for t in tramos.values()]
ok(all(isinstance(n, int) and 8 <= n <= 15 for n in frases) and all(len(t["resumen"].split()) >= 150 for t in tramos.values()),
   f"cada tramo con un resumen de 8-15 frases ({frases})")
ok({"corte_idh", "acervo_sjf", "api_sjf", "dof_y_acuerdos"} <= set(F["cortes"]) and F["cortes"]["api_sjf"]["registro"] == "2032443",
   "los cortes de cada fuente")
ok(len(F["reglas_de_redaccion"]) <= 12, f"{len(F['reglas_de_redaccion'])} reglas de redacción (máx. 12)")
en_acervo_2026 = [r for r in F["recepcion_mx"] if str(r.get("fecha_publicacion") or "") >= "2026"]
ok(all(r.get("en_acervo") for r in en_acervo_2026) and "2032440" not in REC
   and POR_ID["SCJN-2032440"]["en_acervo"] is None,
   "las de 2026 en recepcion_mx están en el acervo; la P./J. 168/2026 sólo en la cronología, sin acervo")
ok(POR_ID["SCJN-2009817"]["vigencia"] == "abandonada" and REC["160526"]["vigencia"] == "superada_en_parte"
   and "2030607" in REC and "J/13 P" in REC["2030607"]["porque"],
   "correcciones de la crítica: P. X/2015 abandonada, P. LXVIII/2011 superada en parte, el «porque» de 2030607")
sup_ppo = F["temas_mx"]["prision_preventiva_oficiosa"]["supervision"][0]["extracto"]
sup_arr = F["temas_mx"]["arraigo"]["supervision"][0]["extracto"]
ok(sup_ppo.endswith("295 a 299 y 301 a 303") and "arraigo" in sup_arr and sup_arr != sup_ppo,
   "la supervisión de García Rodríguez con su remisión entera, y el arraigo con su propio extracto")
c105 = next(c for c in F["constitucion_mx"] if c["ref"] == "art. 105")
ok(c105["fecha_reforma"] == "2024-10-31" and "DOF" in c105["fuente_fecha"], "arts. 105 y 107 con la fecha del DOF")
ok("No lo lee todavía" not in lc.lineas()["estado"], "el «estado» de la ficha ya no dice que nadie la lee")


# ═══════════════════════════════════════════════════════════════ 5 · Qdrant en seco
print("\n5 · scripts/lineas_qdrant.py --seco")
esp = importlib.util.spec_from_file_location("lineas_qdrant", Path("scripts") / "lineas_qdrant.py")
lq = importlib.util.module_from_spec(esp)
esp.loader.exec_module(lq)
pts = lq.armar_puntos()
ok(len(pts) == len(CRONO) + len(F["tramos"]) + 1 and len({p["id"] for p in pts}) == len(pts),
   f"{len(pts)} puntos (cronología + tramos + mapa), ids únicos")
ok(pts == lq.armar_puntos(), "deterministas: armarlos dos veces da los mismos ids y textos")
with tempfile.TemporaryDirectory() as d:
    inf = lq.seco(Path(d))
    lineas_jsonl = (Path(d) / "lineas_puntos_seco.jsonl").read_text().splitlines()
    ok(len(lineas_jsonl) == inf["puntos"] == len(pts) and inf["ids_unicos"] == len(pts) and inf["tokens"] > 0
       and inf["costo_usd"] < 0.01, f"el seco escribe {inf['puntos']} puntos, {inf['tokens']:,} tokens, {inf['costo_usd']} USD")
p0 = next(p for p in pts if p["payload"]["id"] == "SCJN-2024159")["payload"]
ok(p0["tipo_ficha"] == "hito" and p0["registro"] == "2024159" and p0["vigencia"] == "vigente"
   and p0["en_acervo"]["coleccion"] == "jurisprudencia_nacional_v3" and "control_ex_officio_mx" in p0["temas"],
   "el payload lleva linea, tramo, vigencia, registro, en_acervo y temas")
ok(lq.main(["--escribir"]) == 1 and lq.main(["--revertir"]) == 1, "--escribir y --revertir sin --confirmar abortan")


# ═══════════════════════════════════════════════════════════════ 6 · la sonda
print("\n6 · LA SONDA SEMÁNTICA")
v = [0.01] * 1536
ok(correr(lc.sondear(qdrant(), v, 0.5)) is None, "sin la colección `lineas`, sondear devuelve None")
ok(correr(lc.sondear(qdrant(), None, 0.5)) is None, "sin vector, None")
lc._DISPONIBLE.update(t=0.0, v=False)
ok(correr(lc.lineas_disponible(qdrant())) is False, "lineas_disponible: False sin colección")
hit = SimpleNamespace(score=0.71, payload={"linea": FIG, "tipo_ficha": "hito", "tramo": "t4", "id": "SCJN-2024159",
                                            "registro": "2024159"})
bajo = SimpleNamespace(score=0.2, payload={"linea": FIG, "tipo_ficha": "hito", "tramo": "t1", "id": "C-154|s|124"})
qs = qdrant(lineas=[hit])
qs.sonda = [hit, bajo]
son = correr(lc.sondear(qs, v, 0.55))
ok(son and son[0] == FIG and "mexico" in son[1] and "2024159" in son[2] and "C-154|s|124" not in son[2],
   f"con colección: figura, fases del tramo e ids por encima del umbral ({son})")
lc._DISPONIBLE.update(t=0.0, v=False)
ok(correr(lc.lineas_disponible(qs)) is True, "lineas_disponible: True con colección")
lc._DISPONIBLE.update(t=0.0, v=False)
os.environ["LINEAS_UMBRAL"] = "0.8"
ok(lc.umbral() == 0.8 and correr(lc.sondear(qs, v)) is None, "LINEAS_UMBRAL manda (0.8 deja fuera 0.71)")
os.environ.pop("LINEAS_UMBRAL")
ok(lc.umbral() == lc.UMBRAL_LINEAS, f"sin variable, el umbral conservador ({lc.UMBRAL_LINEAS})")
ok(lc.candidata_sonda("¿los jueces pueden dejar de aplicar la Constitución si choca con un tratado?")
   and not lc.candidata_sonda("¿cuánto cuesta un acta de nacimiento en Hidalgo?"),
   "candidata_sonda: sólo preguntas que huelen a la línea")
sl = correr(lc.traer_linea(qdrant(), (son[0], son[1], son[2]), "jurisprudencia_nacional_v3",
                           tesis_a_dict=tesis_a_dict, pregunta="colegiados"))
ok(sl and "2024159" in sl["registros"], "lo que devuelve la sonda entra a traer_linea como detección (3-tupla)")


# ═══════════════════════════════════════════════════════════════ 7 · presupuesto
print("\n7 · EL BLOQUE CABE (tiktoken cl100k, tesis con su rubro real)")
try:
    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")
    casos = [("línea hasta la postura actual", LINEA, "completa"),
             ("estado actual en México", "¿Cuál es el estado actual del control de convencionalidad en México?", "completa"),
             ("obligados hoy", "¿Quiénes están obligados hoy a ejercer el control de convencionalidad?", "completa"),
             ("Colegiados", COLEGIADOS, "completa"), ("Colegiados, solo_mx", COLEGIADOS, "solo_mx"),
             ("línea, solo_mx", LINEA, "solo_mx"), ("prisión preventiva (tema)", "prisión preventiva oficiosa para extorsión", "completa"),
             ("amparo contra reformas (tema)", PETREAS, "completa")]
    for nombre, p, alc in casos:
        if alc == "completa" and not PUNTOS:
            continue
        li = linea(p, alcance=alc, q=qdrant(coidh=(alc == "completa")))
        n = len(enc.encode(li["xml"])) if li else 0
        tope = 5500 if "tema" in nombre else 8000
        ok(0 < n < tope, f"«{nombre}»: {n:,} tokens (< {tope:,})")
except ImportError:
    print("   (sin tiktoken: se omite)")


# ═══════════════════════════════════════════════════════════════ 8 · main.py
print("\n8 · EN EL CHAT: LA PUERTA, EL MODO Y EL EMBEDDING COMPARTIDO")
import main  # noqa: E402


async def _perfil_falso(user_id):
    return (None, True, True) if user_id == "u-admin" else (None, False, True)


_orig = main._plan_para_redaccion
main._plan_para_redaccion = _perfil_falso
try:
    ok(correr(main._coidh_puerta("u-admin", frozenset({"jurisprudencia"}), linea=True)) == (True, "admins:solo_mx"),
       "sólo «jurisprudencia»: la LÍNEA pasa en modo solo_mx")
    ok(correr(main._coidh_puerta("u-admin", frozenset({"jurisprudencia"}))) == (False, "selector"),
       "pero los casos citados de la Corte IDH siguen cerrados")
    ok(correr(main._coidh_puerta("u-admin", frozenset({"federal", "estatal"}), linea=True)) == (False, "selector"),
       "con los dos rubros apagados, cerrada también para la línea")
    ok(correr(main._coidh_puerta("u-normal", frozenset({"jurisprudencia"}), linea=True))[0] is False,
       "y el piloto sigue siendo sólo de administradores")
finally:
    main._plan_para_redaccion = _orig

llamadas = []


async def _emb_falso(texto, modelo=None):
    llamadas.append(texto)
    await asyncio.sleep(0.01)
    return [0.0] * 1536


_orig_emb = main.get_dense_embedding
main.get_dense_embedding = _emb_falso
try:
    async def _dos():
        return await asyncio.gather(main._embedding_denso("la misma pregunta"), main._embedding_denso("la misma pregunta"))
    correr(_dos())
    ok(len(llamadas) == 1, f"doctrina y sonda piden el mismo vector a la vez: una sola llamada ({len(llamadas)})")
    correr(main._embedding_denso("otra pregunta"))
    ok(len(llamadas) == 2, "un texto distinto sí paga su llamada")
finally:
    main.get_dense_embedding = _orig_emb
fuente = Path("main.py").read_text(encoding="utf-8")
ce = fuente[fuente.index("async def chat_endpoint("):]
ce = ce[:ce.index("\n@app.")]
ok("_alcance_coidh = _lc_chat.alcance_por_fuentes(_fuentes_elegidas)" in ce
   and "alcance=_alcance_coidh, pregunta=_texto_det, ids=_ids_sonda" in ce,
   "el chat pasa el alcance del selector, la pregunta y los ids de la sonda a traer_linea")
i_perm = ce.index("or not await _coidh_permitido(linea=True)")
i_emb = ce.index("await _embedding_denso(_pregunta_coidh[:500])")
ok(ce.index("_lc_linea.lineas_disponible(qdrant_client)") < i_perm < i_emb,
   "la sonda: colección que existe y puerta de admins ANTES de pedir el embedding")
ok("v = await _embedding_denso(last_user_message[:500])" in ce, "la doctrina usa el mismo vector compartido")

print()
if FALLOS:
    print(f"FALLAN {len(FALLOS)}:")
    for f in FALLOS:
        print("  ·", f)
    sys.exit(1)
print("TODO PASA")
