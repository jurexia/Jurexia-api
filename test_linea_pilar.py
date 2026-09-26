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
  7. el bloque cabe en el presupuesto de tokens;
  9. lo que encontró la revisión adversarial del 26-sep-2026 (hallazgos 5-7,
     9-17 y 19) no vuelve.

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
# La pregunta nueva de David (revisión adversarial, 26-sep-2026).
TRAZA = ("Traza la línea cronológica desde el nacimiento del control de convencionalidad hasta la postura actual de la "
         "SCJN, y dime qué posturas serias hay para inaplicar restricciones constitucionales como la prisión preventiva "
         "oficiosa")
# Las demás preguntas reales (d1_diagnostico.json), con el 🧵 HILO que dio el
# registro de Render cuando la pregunta sola no abre la línea.
HILO_SCJN = ("Evolución jurisprudencial del control de convencionalidad en México, alcance del artículo 1o. constitucional "
             "y criterios de la Suprema Corte")
LINEA_SCJN = "Y luego cual fue la linea jurisprudencial en la SCJN y su implementación de ese control en mexico"
HILO_MX = ("Evolución jurisprudencial del control de convencionalidad en México tras la reforma constitucional de derechos "
           "humanos de 2011 y el cumplimiento de la sentencia Radilla Pacheco")
MX_EVOL = "Y en México como se introdujo esa figura y cual ha sido su evolución jurisprudencial"
DAVID = [("¿Dónde nace el control de convencionalidad?", None), (MX_EVOL, HILO_MX), (COLEGIADOS, None),
         (LINEA_SCJN, HILO_SCJN), (PETREAS, None), (ERRATA, None),
         ("Que es el control de convencionalidad ex officio y donde surgió?", None)]


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


def linea_hilo(pregunta, hilo=None, alcance="completa"):
    """Como el chat (main.py): la pregunta; si no abre, el 🧵 HILO, y entonces
    la selección ve «pregunta + hilo»."""
    det, texto = lc.pregunta_por_linea(pregunta), pregunta
    if det is None and hilo:
        det, texto = lc.pregunta_por_linea(hilo), f"{pregunta}\n{hilo}"
    if det is None:
        return None
    return correr(lc.traer_linea(qdrant(coidh=(alcance == "completa")), det, "jurisprudencia_nacional_v3",
                                 tesis_a_dict=tesis_a_dict, norma_a_dict=norma_a_dict, alcance=alcance, pregunta=texto))


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
ok(tipos <= {"scjn_tesis", "scjn_resolucion", "acuerdo_general", "pleno_regional_tesis", "pleno_regional_resolucion",
             "tcc_tesis"},
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
FUERZAS = {"obligatoria", "orientadora", "recomendacion", "voto", "doctrina", "proyecto", "historica", "norma",
           "no_publicada"}
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

# Hallazgo 19 (revisión adversarial, 26-sep-2026): reescribir tras quitar una
# entrada dejaba su punto huérfano en lineas_p1 y la sonda lo seguía
# devolviendo. Qdrant EN MEMORIA y OpenAI FALSO: sin red y sin costo.
import hashlib  # noqa: E402
import random  # noqa: E402

import openai  # noqa: E402
import qdrant_client  # noqa: E402


def _vec(t):
    r = random.Random(hashlib.md5(t.encode()).hexdigest())
    v = [r.gauss(0, 1) for _ in range(1536)]
    n = sum(x * x for x in v) ** .5
    return [x / n for x in v]


class _OpenAIFalso:
    def __init__(self, api_key=None):
        pass

    class embeddings:
        @staticmethod
        def create(model, input):
            return SimpleNamespace(data=[SimpleNamespace(embedding=_vec(t)) for t in input],
                                   usage=SimpleNamespace(total_tokens=sum(len(t) // 4 for t in input)))


_MEM = qdrant_client.QdrantClient(location=":memory:")
_orig_qc, _orig_oa, _orig_pausa = qdrant_client.QdrantClient, openai.OpenAI, lq.PAUSA
qdrant_client.QdrantClient, openai.OpenAI, lq.PAUSA = (lambda *a, **k: _MEM), _OpenAIFalso, 0
try:
    with tempfile.TemporaryDirectory() as d:
        env = Path(d) / "falso.env"
        env.write_text('QDRANT_URL="http://falso"\nQDRANT_API_KEY=x\nOPENAI_API_KEY=sk-falso\n')
        r1 = lq.main(["--escribir", "--confirmar", "--env", str(env)])
        n1 = _MEM.count(lq.COLECCION, exact=True).count
        datos = json.loads(lc.RUTA_LINEAS.read_text(encoding="utf-8"))
        fig = datos["figuras"][FIG]
        quitada = next(e for e in fig["cronologia"] if e.get("registro"))
        fig["cronologia"] = [e for e in fig["cronologia"] if e is not quitada]
        mod = Path(d) / "lineas_mod.json"
        mod.write_text(json.dumps(datos, ensure_ascii=False), encoding="utf-8")
        r2 = lq.main(["--escribir", "--confirmar", "--env", str(env), "--lineas", str(mod)])
        n2 = _MEM.count(lq.COLECCION, exact=True).count
        viejo = next(p for p in pts if p["payload"]["id"] == quitada["id"])

        class _Asinc:
            async def query_points(self, **kw):
                return _MEM.query_points(**kw)

        son_q = correr(lc.sondear(_Asinc(), _vec(viejo["texto"]), 0.55))
    ok(r1 == 0 and n1 == len(pts), f"--escribir (Qdrant en memoria): {n1} puntos, los del JSON")
    ok(r2 == 0 and n2 == len(pts) - 1, f"reescribir sin {quitada['id']}: borra su punto huérfano ({n2} puntos)")
    ok(son_q is None or quitada["id"] not in son_q[2], f"y la sonda ya no la devuelve ({son_q})")
finally:
    qdrant_client.QdrantClient, openai.OpenAI, lq.PAUSA = _orig_qc, _orig_oa, _orig_pausa


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
    # Desde el 26-sep-2026 el presupuesto se APLICA en traer_linea (antes sólo
    # se medía aquí): la línea, lc.PRESUPUESTO_LINEA; un tema sin fase,
    # lc.PRESUPUESTO_TEMA. Con las preguntas reales de David (d1_diagnostico)
    # y la nueva «Traza la línea…», que medía 10,285 sin tope.
    casos = [("línea hasta la postura actual", LINEA, None, "completa"),
             ("estado actual en México", "¿Cuál es el estado actual del control de convencionalidad en México?", None,
              "completa"),
             ("obligados hoy", "¿Quiénes están obligados hoy a ejercer el control de convencionalidad?", None, "completa"),
             ("Colegiados", COLEGIADOS, None, "completa"), ("Colegiados, solo_mx", COLEGIADOS, None, "solo_mx"),
             ("línea, solo_mx", LINEA, None, "solo_mx"),
             ("prisión preventiva (tema)", "prisión preventiva oficiosa para extorsión", None, "completa"),
             ("amparo contra reformas (tema)", PETREAS, None, "completa"),
             ("Traza la línea (nueva)", TRAZA, None, "completa"), ("Traza la línea (nueva), solo_mx", TRAZA, None, "solo_mx")]
    casos += [(f"David: {p[:40]}…", p, h, alc) for p, h in DAVID for alc in ("completa", "solo_mx")]
    for nombre, p, h, alc in casos:
        if alc == "completa" and not PUNTOS:
            continue
        li = linea_hilo(p, h, alcance=alc)
        n = len(enc.encode(li["xml"])) if li else 0
        tope = lc.PRESUPUESTO_LINEA if li and li["fases"] else lc.PRESUPUESTO_TEMA
        ok(0 < n < tope, f"«{nombre}»: {n:,} tokens (< {tope:,})")
except ImportError:
    print("   (sin tiktoken: se omite)")


# ═══════════════════════════════════════════════════════════════ 9 · revisión adversarial
print("\n9 · LA REVISIÓN ADVERSARIAL DEL 26-SEP-2026 (hallazgos 5-7, 9-17)")
TRAMOS = F["tramos"]

# 5 · La CC 3/2026 del Pleno Regional Centro-Norte, con la SCJN frenando su
# publicación: sólo consta en prensa y entra así, marcada.
cc = POR_ID.get("PR-PTCN-CC-3-2026") or {}
ok(cc.get("fecha_resolucion") == "2026-04-16" and cc.get("fuerza") == "no_publicada"
   and cc.get("vigencia") == "pendiente" and (cc.get("verificacion") or {}).get("estado") == "parcial"
   and cc.get("fuente_tipo") == "prensa" and len(cc.get("fuentes_prensa") or []) == 3
   and "17-abr-2026" in cc.get("vigencia_nota", "") and "AG 2/2024" in cc.get("vigencia_nota", ""),
   "5 · la CC 3/2026 (16-abr-2026) entra como no publicada, parcial y de prensa, con el oficio del 17-abr-2026")
ok("no se localizó criterio de Pleno Regional" not in TRAMOS["t6"]["resumen"] and "CC 3/2026" in TRAMOS["t6"]["resumen"],
   "5 · el t6 ya no dice que no hay criterio regional sobre el art. 166 LA")
desplazada = [x.get("registro") or x.get("id") for x in F["recepcion_mx"] + CRONO
              if "desplazada en la práctica" in str(x.get("vigencia_nota") or "")]
ok(not desplazada and all("CC 3/2026" in str(x.get("vigencia_nota")) for x in (
    REC["2027280"], REC["2030441"], REC["2030607"], POR_ID["SCJN-2027280"], POR_ID["SCJN-2030441"],
    POR_ID["SCJN-2030607"])), f"5 · «desplazada en la práctica» pasa a «disputada» con la CC 3/2026 ({desplazada})")
ok("CC 3/2026" in F["fase_actual"]["pendientes"] and any("CC 3/2026" in r for r in F["reglas_de_redaccion"])
   and "PR-PTCN-CC-3-2026" in TRAMOS["t6"]["ids"], "5 · en pendientes del estado actual, en la regla del art. 166 y en el t6")
lp = linea("prisión preventiva oficiosa para extorsión")
xp = lp["xml"] if lp else ""
ecc = re.search(r'<entrada [^>]*clave="CC 3/2026"[^>]*>.*?</entrada>', xp)
ok(ecc and 'verificacion="parcial"' in ecc.group(0) and 'fuerza="no_publicada"' in ecc.group(0)
   and "sólo consta en prensa" in ecc.group(0),
   "5 · con el tema de la prisión preventiva el modelo la ve: parcial, no publicada y «sólo consta en prensa»")
t2030607 = re.search(r'<tesis [^>]*registro="2030607"[^>]*>.*?</tesis>', xp)
ok(t2030607 and "CC 3/2026" in t2030607.group(0) and "suspendida por la SCJN" in t2030607.group(0),
   "5 · y la nota de la J/33 P llega entera: quién la tiene por inaplicable y quién sostuvo lo contrario")

# 6 · Un solo Pleno Regional (Penal y de Trabajo, Centro-Norte) en la J/31 y la
# J/33 P; Centro-Norte frente a Centro-Sur.
t6 = TRAMOS["t6"]["resumen"]
ok("dos Plenos Regionales" not in t6 and "Pleno Regional en Materias Penal y de Trabajo de la Región Centro-Norte dijo"
   in t6 and "Centro-Sur" in t6 and "cada uno obliga sólo en su región" in t6,
   "6 · el t6 dice UN Pleno Regional (Centro-Norte) en dos contradicciones y lo separa del Centro-Sur")
ok("Los Plenos Regionales la leyeron" not in POR_ID["DOF-2024-12-31-art19-literalidad"]["detalle"]
   and POR_ID["SCJN-2030607"]["organo"] == POR_ID["SCJN-2030441"]["organo"]
   == "Pleno Regional en Materias Penal y de Trabajo de la Región Centro-Norte"
   and POR_ID["SCJN-2028043"]["organo"] == "Pleno Regional en Materia Penal de la Región Centro-Sur",
   "6 · la J/31 y la J/33 P del mismo órgano; la J/16 P, del Centro-Sur")

# 7 · P./J. 2/2025: la ejecutoria es del 17-jun-2024 (CC 175/2022), no la
# aprobación del número (27-may-2025), y va en su lugar de la cronología.
p25 = POR_ID["SCJN-2030517"]
antes_de = [e for e in CRONO if e["orden"] == p25["orden"] - 1][0]
despues_de = [e for e in CRONO if e["orden"] == p25["orden"] + 1][0]
_f = lambda e: str(e.get("fecha_resolucion") or e.get("fecha_publicacion"))  # noqa: E731
ok(p25["fecha_resolucion"] == "2024-06-17" and p25["fecha_publicacion"] == "2025-06-13"
   and _f(antes_de) <= "2024-06-17" < _f(despues_de) and p25["orden"] < POR_ID["SCJN-AG-2-2024"]["orden"],
   f"7 · la P./J. 2/2025 con su ejecutoria (17-jun-2024), entre {antes_de['id']} y {despues_de['id']}")

# 9 · Un hito «verificado» enlaza el MISMO PDF que se cotejó (Gelman ¶238 y
# Kawas ¶202 enlazaban el _esp1 y se cotejaron en el _esp).
otro_pdf = []
for h in F["hitos"]:
    url = (h.get("url_oficial") or "").split("#")[0]
    ver = str(h.get("verificacion") or "")
    if h.get("verificado") and ver and "corteidh" in url:
        base = url.rsplit("/", 1)[-1]
        if base not in ver or f"{base} (no cotejado)" in ver:
            otro_pdf.append(h["llave"])
g238 = lc._hitos_por_llave(FIG)["C-221|s|238"]
ok(not otro_pdf, f"9 · todo hito verificado enlaza el PDF en que se cotejó ({otro_pdf})")
ok("legal-docs" in (lc.pdf_de(g238["url_oficial"], g238.get("pdf_sha1"))[0] or "")
   and g238.get("url_catalogo", "").endswith("_esp1.pdf"),
   "9 · Gelman ¶238 abre la copia verificada del _esp y guarda el _esp1 como url_catalogo")

# 10 · La IX.P. J/4, J/5 y J/6 P interpretan conforme el art. 19.
ok(all(REC[r]["postura"] == "interpreta_conforme_o_inaplica" for r in ("2027760", "2027761", "2027766", "2027756")),
   "10 · las IX.P. J/2, J/4, J/5 y J/6 P, todas «interpreta_conforme_o_inaplica»")
ok(not [r["registro"] for r in F["recepcion_mx"] if (r.get("rubro_abreviado") or "").startswith("PRISIÓN PREVENTIVA")
        and re.search(r"interpretación conforme|test de proporcionalidad", r.get("porque") or "")
        and r.get("postura") == "no_aplica"],
   "10 · ninguna tesis de prisión preventiva que interpreta conforme (o aplica el test de García Rodríguez) queda "
   "en «no_aplica»")

# 17a · Toda tesis del acervo en la recepción tiene fecha.
sin_fecha = [r["registro"] for r in F["recepcion_mx"] if r.get("en_acervo") and not r.get("fecha_publicacion")]
ok(not sin_fecha and REC["160482"]["fecha_publicacion"] == "2011-12" and REC["2003156"]["fecha_publicacion"] == "2013-03",
   f"17a · todas las tesis de la recepción con fecha ({sin_fecha})")


def hitos_de(li):
    return set(li["hitos"]) if li else set()


# 11 · El núcleo mexicano entra SIEMPRE con la pregunta amplia, antes que la
# relevancia (la pregunta real de David sobre la línea de la SCJN, con su hilo).
NUCLEO_MX = [str(x) for x in F["nucleo_mx"]]
for alc in ("completa", "solo_mx"):
    li = linea_hilo(LINEA_SCJN, HILO_SCJN, alcance=alc)
    ok(li and set(NUCLEO_MX) <= set(li["registros"]) and li["registros"][:len(NUCLEO_MX)] == NUCLEO_MX,
       f"11 · «{LINEA_SCJN[:40]}…» + hilo [{alc}]: el núcleo entero y primero ({li and li['registros'][:8]})")
    ok('registro="2006224"' in li["xml"] and 'registro="160589"' in li["xml"],
       f"11 · y la P./J. 20/2014 y la P. LXVII/2011 llegan al XML [{alc}]")

# 12 · Con México y «actual», hitos interamericanos entre Cesados (2006) y
# Tzompaxtle (2022); y «→ ejes» sólo si entró alguno.
for nombre, p in (("LINEA", LINEA), ("Traza la línea", TRAZA)):
    li = linea(p) if PUNTOS else None
    if not li:
        continue
    medio = [h for h in li["hitos"] if "2006-11-24" < str(lc._hitos_por_llave(FIG)[h].get("fecha")) < "2022-11-07"
             and not h.startswith("C-209|")]
    ok(len(medio) >= lc.MIN_POR_FASE, f"12 · «{nombre}»: {len(medio)} hitos interamericanos de evolución ({medio})")
    fases_eje = {(lc._hitos_por_llave(FIG)[h].get("fase") or "").lower() for h in li["hitos"]} & set(lc._RX_EJES)
    ok(("→ ejes" in li["xml"]) == bool(fases_eje), f"12 · «{nombre}»: «→ ejes» sólo con hitos de eje ({sorted(fases_eje)})")
ok("evolucion" in lc.pregunta_por_linea(TRAZA)[1], "12 · «desde el nacimiento hasta la postura actual» pide también la evolución")
if PUNTOS:
    lo = linea("¿Dónde nace el control de convencionalidad?")
    ok(lo and "→ ejes" not in lo["xml"] and "→ ex officio (Cesados ¶128) → estado actual" in lo["xml"],
       "12 · «¿Dónde nace…?» no trae ejes y la instrucción ya no los pide")

# 13 · Radilla ¶339-341 cuando la pregunta es de México y pide origen, evolución
# o concepto; no sin México.
RAD = set(lc.RADILLA)
for nombre, p, h in (("Y en México como se introdujo… (hilo)", MX_EVOL, HILO_MX),
                     ("¿Cómo se introdujo… en México…?", "¿Cómo se introdujo el control de convencionalidad en México y "
                      "cuál ha sido su evolución jurisprudencial?", None),
                     ("¿Cómo llegó… a México?", "¿Cómo llegó el control de convencionalidad a México?", None),
                     ("Traza la línea", TRAZA, None), ("LINEA", LINEA, None)):
    if PUNTOS:
        li = linea_hilo(p, h)
        ok(RAD <= hitos_de(li), f"13 · «{nombre}»: Radilla ¶339-341 ({sorted(hitos_de(li) & RAD)})")
ok(not RAD & {h["llave"] for h in lc.seleccionar(FIG, ["origen"], pregunta="¿Dónde nace?")["hitos"]},
   "13 · sin México, Radilla no entra")

# 14 · «Traza la línea…»: dentro del presupuesto, con la 2a./J. 163/2017 y la
# doctrina del tema de restricciones; los temas por turnos y en el orden en que
# la pregunta los nombra.
det_t = lc.pregunta_por_linea(TRAZA)
ok([x for x in det_t[1] if x.startswith("tema:")] == ["tema:restricciones_constitucionales",
                                                        "tema:prision_preventiva_oficiosa"],
   "14 · los temas en el orden en que la pregunta los nombra (restricciones, luego prisión preventiva)")
ok(lc._por_turnos([["a", "b", "c"], ["b", "d"], ["e"]]) == [("a", 0, 0), ("b", 0, 1), ("e", 0, 2), ("c", 1, 0), ("d", 1, 1)],
   "14 · _por_turnos: una de cada lista por vuelta, sin repetir")
for alc in ("completa", "solo_mx"):
    if alc == "completa" and not PUNTOS:
        continue
    lt = linea_hilo(TRAZA, None, alcance=alc)
    try:
        import tiktoken
        nt = len(tiktoken.get_encoding("cl100k_base").encode(lt["xml"]))
    except ImportError:
        nt = lc.medida(lt["xml"])
    ok(nt < lc.PRESUPUESTO_LINEA and "2015828" in lt["registros"] and 'registro="2015828"' in lt["xml"],
       f"14 · «Traza la línea…» [{alc}]: {nt:,} tokens (< {lc.PRESUPUESTO_LINEA:,}) y con la 2a./J. 163/2017")
    if alc == "completa":
        ok("DOCTRINA-885ee8c6-7543-5a1b-bbdb-fa0a3e510053" in lt["cronologia"]
           and "SCJN-ct-293-2011-ejecutoria-24985" in lt["cronologia"]
           and {"DOF-2024-10-31-inimpugnabilidad", "DOF-2024-12-31-art19-literalidad", "SCJN-AG-2-2024",
                "SCJN-impedimento-60-2025"} <= set(lt["cronologia"]),
           f"14 · y la regla (CT 293/2011) con la doctrina contraria, sin perder el estado actual ({lt['cronologia']})")
        ok(set(NUCLEO_MX) <= set(lt["registros"]) and set(lc.NUCLEO) <= set(lt["hitos"]) and RAD <= set(lt["hitos"])
           and {"C-470|s|118", "C-482|s|176", "C-482|s|301", "C-482|s|303"} <= set(lt["hitos"])
           and {"2009816", "2009817"} <= set(lt["registros"]),
           "14 · el recorte nunca toca el núcleo, Radilla, el estado actual ni las abandonadas con su sustituta")
        fuera_q4 = lt.get("recortado") or []
        ok(fuera_q4 and all(x.split(":", 1)[0] in ("cronologia", "registro", "hito", "supervision", "tensiones")
                            for x in fuera_q4),
           f"14 · el presupuesto se aplica en traer_linea ({len(fuera_q4)} piezas fuera)")
sel_q4 = lc.seleccionar(FIG, det_t[1], pregunta=TRAZA)
prot = set(NUCLEO_MX) | {"2009816", "2009817"}
rec_q4 = sel_q4.get("recortables")
ok(rec_q4 and not prot & {x for t, x in rec_q4 if t == "registro"}
   and not (set(lc.NUCLEO) | RAD | {"C-470|s|118", "C-482|s|176", "C-482|s|301", "C-482|s|303"})
   & {x for t, x in rec_q4 if t == "hito"},
   "14 · hay un orden de recorte y lo protegido no está en él")

# 15 · En solo_mx no hay fecha de la Corte IDH para el bloque; en completa, la
# fecha dice que es de la Corte IDH.
xs = linea(TRAZA, alcance="solo_mx", q=qdrant(coidh=False))["xml"]
ok("vigente_al=" not in xs and "<vigencia>" not in xs and "<cortes>" in xs,
   "15 · solo_mx: sin vigente_al ni <vigencia>; sólo <cortes>")
if PUNTOS:
    xc = linea(TRAZA)["xml"]
    ok('corte_idh="2025-09-30"' in xc.split("\n")[0] and "vigente_al=" not in xc
       and "Según lo ingerido hasta 2025-09-30 en la Corte IDH" in xc,
       "15 · completa: la fecha de ingesta es de la Corte IDH, no del bloque")

# 16 · El aporte se corta en un fin de frase y no a media: la AI 130/2019 dice
# que la interpretación conforme no tuvo mayoría.
ai = re.search(r'<entrada [^>]*clave="AI 130/2019[^"]*"[^>]*>.*?</entrada>', xs)
ok(ai and "no tuvo mayoría y la P./J. 20/2014 quedó intacta" in ai.group(0),
   "16 · la AI 130/2019 llega con «no tuvo mayoría y la P./J. 20/2014 quedó intacta»")
ok(lc._recorte("Uno dos tres. Cuatro cinco seis siete.", 4) == "Uno dos tres."
   and lc._recorte("Según el art. 19 y el art. 166 fr. I de la ley vigente hoy", 10)
   == "Según el art. 19 y el art. 166 fr. I…"
   and lc._recorte("Lo dice el art. 19 y el art. 166; y algo más largo que no cabe aquí.", 9)
   == "Lo dice el art. 19 y el art. 166;"
   and lc._recorte("la P./J. 20/2014 quedó intacta y sigue", 3).endswith("…")
   and lc._recorte("corto.", 45) == "corto.",
   "16 · _recorte: en «.» o «;», nunca después de «art.» ni de «P./J.»; sin fin de frase, «…»")
largos = [e["id"] for e in CRONO if e.get("aporte") and len(e["aporte"].split()) > lc.MAX_PALABRAS_APORTE
          and not re.search(r"[.;:]$", lc._recorte(e["aporte"], lc.MAX_PALABRAS_APORTE))]
ok(not largos, f"16 · ningún aporte de la cronología queda a media frase ({largos[:4]})")

# 17b · <recepcion_mx> en orden de fecha, como <hitos> y <cronologia>.
for alc in ("completa", "solo_mx"):
    if alc == "completa" and not PUNTOS:
        continue
    for p, h in ((TRAZA, None), (LINEA_SCJN, HILO_SCJN), (LINEA, None)):
        li = linea_hilo(p, h, alcance=alc)
        rec_x = li["xml"][li["xml"].index("<recepcion_mx>"):li["xml"].index("</recepcion_mx>")]
        fechas = [f[:10] for f in re.findall(r'<tesis registro="\d+"[^>]*fecha="([^"]*)"', rec_x)]
        ok(fechas and all(fechas) and fechas == sorted(fechas),
           f"17b · <recepcion_mx> cronológica y sin fecha vacía [{alc}] «{p[:30]}…» ({fechas[:4]}…)")


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
