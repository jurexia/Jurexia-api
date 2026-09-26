"""La Corte IDH en el chat: el resolvedor, la línea y la puerta — 25-sep-2026.

    .venv/bin/python test_coidh_chat.py

Sin red y sin gastar API. Qdrant es falso y lo alimentan los puntos EXACTOS
que escribirá la ingesta (reingesta/coidh/f0/puntos_seco.jsonl, 16,825
puntos de 58 resoluciones): si el payload cambia, esto se entera. Las
preguntas negativas salen de preguntas_oro.json. Cubre lo que pide el plan
(§3.5-3.7) con su revisión escéptica: A.1 (la puerta), A.3 (el bloque propio,
después del contexto normal), B.7 (el seguimiento), B.8 (sin documentos),
B.10 (los temas mexicanos) y que una colección ausente no cambia nada.
"""
import asyncio
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, os.getcwd())
os.environ["COIDH_ACTIVO"] = "admins"
import main  # noqa: E402
import linea_coidh as lc  # noqa: E402
import coidh_catalogo as cc  # noqa: E402
from qdrant_client.models import MatchAny, MatchValue  # noqa: E402

F0 = Path(os.getenv("COIDH_F0", "/Users/josedavidalcantarmendoza/Documents/IUREXIA-MAC/reingesta/coidh/f0"))
RUTA_PUNTOS = F0 / "puntos_seco.jsonl"
RUTA_ORO = F0 / "preguntas_oro.json"

FALLOS = []


def ok(cond, que):
    print(f"   {'PASA ' if cond else 'FALLA'}  {que}")
    if not cond:
        FALLOS.append(que)


if not RUTA_PUNTOS.exists():
    print(f"No están los puntos de la fase en seco ({RUTA_PUNTOS}); se omite la prueba.")
    sys.exit(0)

# Las copias verificadas de legal-docs/CorteIDH (25-sep-2026): lo que abre el
# visor. Se lee el manifiesto que viaja con el código, no una lista escrita aquí.
MANIFIESTO = json.loads(lc.RUTA_COPIAS.read_text(encoding="utf-8"))
COPIA_BASE = "https://ukcuzhwmmfwvcedvhfll.supabase.co/storage/v1/object/public/legal-docs/CorteIDH/"

main._NOMBRES_FEDERALES["lista"] = [(main._normalizar_nombre_ley("Ley de Amparo"), "Ley de Amparo")]
main._NOMBRES_FEDERALES["ts"] = 9e18          # que no intente refrescar contra la red


# ═══════════════════════════════════════════════════════════════ Qdrant falso

class ColeccionAusente(Exception):
    """Lo que hace Qdrant con una colección que no existe (404)."""


class QdrantFalso:
    def __init__(self, colecciones):
        self.col = colecciones
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
            if isinstance(c.match, MatchAny):
                if not any(x in c.match.any for x in vals):
                    return False
            elif isinstance(c.match, MatchValue):
                if c.match.value not in vals:
                    return False
            if c.range is not None:
                if v is None:
                    return False
                r = c.range
                if (r.gte is not None and not v >= r.gte) or (r.gt is not None and not v > r.gt) \
                        or (r.lte is not None and not v <= r.lte) or (r.lt is not None and not v < r.lt):
                    return False
        return True

    async def scroll(self, collection_name, scroll_filter=None, limit=10, offset=None, order_by=None,
                     with_payload=True, with_vectors=False, **kw):
        self.llamadas.append(("scroll", collection_name))
        pts = [p for p in self._pts(collection_name) if self._cumple(p.payload, scroll_filter)]
        if order_by is not None:
            desc = "desc" in str(getattr(order_by, "direction", "")).lower()
            pts.sort(key=lambda p: p.payload.get(order_by.key) or 0, reverse=desc)
            return pts[:limit], None
        pts.sort(key=lambda p: str(p.id))
        ini = 0
        if offset is not None:
            ini = next((i for i, p in enumerate(pts) if str(p.id) >= str(offset)), len(pts))
        pagina = pts[ini:ini + limit]
        return pagina, (str(pts[ini + limit].id) if ini + limit < len(pts) else None)

    async def retrieve(self, collection_name, ids, with_payload=True, **kw):
        self.llamadas.append(("retrieve", collection_name))
        quiero = {str(i).lower() for i in ids}
        return [p for p in self._pts(collection_name) if str(p.id).lower() in quiero]


print("Cargando los puntos de la fase en seco…")
PUNTOS = []
with RUTA_PUNTOS.open(encoding="utf-8") as fh:
    for linea in fh:
        o = json.loads(linea)
        PUNTOS.append(SimpleNamespace(id=o["id"], payload=o["payload"]))
POR_LLAVE = {}
for p in PUNTOS:
    POR_LLAVE.setdefault(p.payload["llave"], []).append(p)
print(f"   {len(PUNTOS):,} puntos, {len({p.payload['doc_id'] for p in PUNTOS})} resoluciones")

# Tres tesis y un artículo, INVENTADOS para la prueba (sólo lo que el código
# lee): la recepción en México se trae de jurisprudencia_nacional_v3 por
# registro y la Constitución por el id de la ficha.
_FICHA = lc.figuras()["control_convencionalidad"]
_ID_ART19 = next(c["qdrant_id"] for c in _FICHA["constitucion_mx"] if c["ref"] == "art. 19 párr. 2")
TESIS = [SimpleNamespace(id=f"00000000-0000-4000-8000-00000000000{k}", payload={
    "registro": reg, "clave_tesis": clave, "rubro": rubro, "texto": rubro + ". Texto de prueba.",
    "instancia": "Pleno", "tipo": "TESIS AISLADA"})
    for k, (reg, clave, rubro) in enumerate([
        ("160589", "P. LXVII/2011(9a.)", "CONTROL DE CONVENCIONALIDAD EX OFFICIO EN UN MODELO DE CONTROL DIFUSO"),
        ("2006224", "P./J. 20/2014 (10a.)", "DERECHOS HUMANOS CONTENIDOS EN LA CONSTITUCIÓN Y EN LOS TRATADOS"),
        ("2006225", "P./J. 21/2014 (10a.)", "JURISPRUDENCIA EMITIDA POR LA CORTE INTERAMERICANA")])]
NORMAS = [SimpleNamespace(id=_ID_ART19, payload={
    "texto": "Art. 19 CPEUM (texto de prueba)", "ref": "Art. 19 CPEUM (parte 2)",
    "origen": "Constitución Política de los Estados Unidos Mexicanos"})]


def qdrant_con(coidh=True):
    cols = {"jurisprudencia_nacional_v3": TESIS, "bloque_constitucional": NORMAS}
    if coidh:
        cols["coidh"] = PUNTOS
    return QdrantFalso(cols)


def correr(coro):
    return asyncio.run(coro)


ORO = json.loads(RUTA_ORO.read_text(encoding="utf-8"))["preguntas"]


# ═══════════════════════════════════════════════════════════════ 1 · puerta
print("\n1 · LA PUERTA SE QUEDA CERRADA EN LAS PREGUNTAS MEXICANAS")
negativas = [q["turnos"][0] for q in ORO if q.get("negativa")]
negativas += ["interés convencional moratorio en un pagaré", "control difuso de constitucionalidad en materia civil",
              "¿Cómo se regula el voto en Hidalgo?", "mi cliente José García Rodríguez fue detenido",
              "el amparo contra la orden de aprehensión", "Ley de Hacienda de Hidalgo, párrafo 4"]
ok(len(negativas) >= 10, f"{len(negativas)} preguntas negativas ({sum(1 for q in ORO if q.get('negativa'))} de preguntas_oro.json)")
for q in negativas:
    cit = main._extract_legal_citations(q, pregunta_coidh=q)
    ok(lc.pregunta_por_linea(q) is None and cit["casos_coidh"] == [],
       f"«{q[:60]}»: ni línea ni caso")


# ═══════════════════════════════════════════════════════════════ 2 · resolvedor
print("\n2 · EL RESOLVEDOR, DESDE _extract_legal_citations, DA LA LLAVE")
ESPERADAS = [
    ("Almonacid Arellano vs. Chile, párrafo 124", "C-154|s|124"),
    ("Voto de García Ramírez en Trabajadores Cesados, párr. 12", "C-158|v:garcia-ramirez|12"),
    ("Tzompaxtle, resolutivo 8", "C-470|r|8"),
    ("OC-24/17 párr. 26", "A-24|s|26"),
]
CASOS = {}
for q, llave in ESPERADAS:
    cit = main._extract_legal_citations(q, pregunta_coidh=q)
    casos = cit["casos_coidh"]
    CASOS[llave] = casos
    ok(len(casos) == 1 and casos[0]["llaves"] == [llave], f"«{q}» → {llave}  (dio {[c['llaves'] for c in casos]})")
ok(set(main._extract_legal_citations("x")) == {"articles", "registros", "tesis_nums", "casos_coidh"},
   "el dict de citas trae «casos_coidh» siempre (vacío si no hay pregunta)")
doc = "DEMANDA DE AMPARO … como resolvió la Corte IDH en el Caso Almonacid Arellano y otros Vs. Chile, párr. 124 …"
ok(main._extract_legal_citations(doc)["casos_coidh"] == [],
   "con un escrito (sin pregunta_coidh) no se resuelve nada: revisión B.8")
os.environ["COIDH_ACTIVO"] = "off"
ok(main._extract_legal_citations(ESPERADAS[0][0], pregunta_coidh=ESPERADAS[0][0])["casos_coidh"] == [],
   "con COIDH_ACTIVO=off ni se llama al resolvedor")
os.environ["COIDH_ACTIVO"] = "admins"
largo = "x " * 3000 + "Almonacid Arellano vs. Chile, párrafo 124"
ok(main._extract_legal_citations(largo, pregunta_coidh=largo)["casos_coidh"] == [],
   "la pregunta se recorta a 4,000 caracteres, como se midió el 0.03 %")

print("\n   · seguimiento (revisión B.7)")
hist = [SimpleNamespace(role="user", content="Almonacid Arellano vs. Chile, párrafo 124"),
        SimpleNamespace(role="assistant", content="El párrafo 124 de Almonacid dice…"),
        SimpleNamespace(role="user", content="¿y el párrafo 125?")]
previo = lc.previo_de_historial(hist, "¿y el párrafo 125?")
ok(previo == "C-154|s|124", f"la última llave citada es la de la pregunta anterior (dio {previo})")
seg = main._extract_legal_citations("¿y el párrafo 125?", pregunta_coidh="¿y el párrafo 125?", previo_coidh=previo)
ok([c["llaves"] for c in seg["casos_coidh"]] == [["C-154|s|125"]], "«¿y el párrafo 125?» → C-154|s|125")
hist2 = [SimpleNamespace(role="user", content="háblame del control de convencionalidad"),
         SimpleNamespace(role="assistant", content='… <!-- CITATION_META:{"sources": {"a": {"llave": "C-158|s|128"}}} -->'),
         SimpleNamespace(role="user", content="¿y el párrafo 129?")]
ok(lc.previo_de_historial(hist2, "¿y el párrafo 129?") == "C-158|s|128",
   "sin caso en la pregunta anterior, la última llave de los marcadores de la respuesta")
ok(lc.previo_de_historial(hist, "¿qué dice el artículo 14 de la Ley de Amparo sobre el plazo de quince días en materia penal y agraria?") is None,
   "una pregunta que no habla de párrafos no busca cita previa")


# ═══════════════════════════════════════════════════════════════ 3 · por llave
print("\n3 · LO CITADO SE TRAE POR LLAVE, CON SCORE 1.0 Y LO PEDIDO PRIMERO")
QD = qdrant_con()
main.qdrant_client = QD


def directos(casos):
    return correr(main._buscar_articulos_citados(
        {"articles": [], "registros": [], "tesis_nums": [], "casos_coidh": casos}))


PAGINAS = {"C-154|s|124": 53, "C-158|v:garcia-ramirez|12": 68, "C-470|r|8": 62, "A-24|s|26": 14}
for llave, casos in CASOS.items():
    res = [r for r in directos(casos) if r.silo == "coidh"]
    pri = res[0] if res else None
    ok(pri is not None and pri.llave == llave and pri.pagina == PAGINAS[llave] and pri.score == 1.0
       and pri.rol_coidh == "pedido",
       f"{llave}: primer documento coidh, pág. {PAGINAS[llave]}, score 1.0")
    # Desde el 25-sep-2026 el visor abre la COPIA verificada de legal-docs
    # (Cloudflare de la Corte reta con 403 al proxy); la cita sigue siendo la
    # URL oficial, y el sha1 es el de la copia (el del manifiesto).
    ok(pri is not None and pri.pdf_url and "#" not in pri.pdf_url
       and pri.pdf_url.startswith(COPIA_BASE)
       and pri.url_oficial.startswith("https://www.corteidh.or.cr/") and "#" not in pri.url_oficial
       and pri.pdf_sha1 == MANIFIESTO[pri.url_oficial]["sha1"]
       and pri.pdf_url == MANIFIESTO[pri.url_oficial]["copia"],
       f"{llave}: pdf_url = la copia de legal-docs, url_oficial = corteidh, pdf_sha1 del manifiesto, sin #page")
alm = [r for r in directos(CASOS["C-154|s|124"]) if r.silo == "coidh"]
ok([r.llave for r in alm] == ["C-154|s|124", "C-154|s|123", "C-154|s|125"],
   f"Almonacid ¶124 con un vecino a cada lado por orden ({[r.llave for r in alm]})")
p124 = alm[0]
ok(p124.id == POR_LLAVE["C-154|s|124"][0].id, "el [Doc ID] es el id del punto que escribirá la ingesta")
ok(p124.parrafo == "124" and p124.seg == "sentencia" and p124.caso == "Almonacid Arellano y otros Vs. Chile"
   and p124.serie == "Serie C No. 154" and p124.fecha == "2006-09-26" and p124.tipo == "sentencia_coidh"
   and p124.cita_canonica.endswith("Serie C No. 154, párr. 124.") and p124.ancla.startswith("La Corte es consciente"),
   "el contrato completo: parrafo «124», caso, serie, fecha, tipo, ancla, cita canónica")
voto = [r for r in directos(CASOS["C-158|v:garcia-ramirez|12"]) if r.silo == "coidh"][0]
ok(voto.voto_autor == "García Ramírez" and voto.tipo == "voto_coidh" and voto.seg == "voto",
   "el voto trae el NOMBRE del juez, no el slug")
oc = [r for r in directos(CASOS["A-24|s|26"]) if r.silo == "coidh"][0]
ok(oc.seg == "sentencia" and oc.tipo == "oc_coidh" and oc.voto_autor is None,
   "OC-24 ¶26 es el de la opinión, no un ¶26 de los votos (págs. 98, 132, 141)")

radilla = cc.resolver_citas_coidh("caso Radilla")
res = [r for r in directos(radilla) if r.silo == "coidh"]
ok(res and all(r.llave.startswith("C-209|") for r in res) and any(r.seg == "resolutivos" for r in res)
   and res[0].rol_coidh == "destacado",
   f"«caso Radilla» sin párrafo: los más citados y los resolutivos ({len(res)} unidades)")
ok(len(res) <= lc.MAX_UNIDADES_CITADAS, "con tope de unidades")
vr = cc.resolver_citas_coidh("Caso Artavia Murillo y otros Vs. Costa Rica, párr. 264")
fic = [r for r in directos(vr) if r.silo == "coidh"]
ok(len(fic) == 1 and fic[0].rol_coidh == "ficha" and fic[0].id == lc.ficha_id(vr[0]["doc_id"])
   and fic[0].url_oficial and "no está ingerido" in fic[0].texto and fic[0].pagina is None,
   "caso fuera del piloto: la ficha del catálogo, con cita y URL, sin texto de la sentencia")
ok(bool(fic) and fic[0].url_oficial not in MANIFIESTO and fic[0].pdf_url == fic[0].url_oficial
   and fic[0].pdf_sha1 is None,
   "sin copia en el manifiesto, la ficha sigue abriendo la URL oficial (como antes), sin sha1")
ok(bool(fic) and lc.ficha_por_id(fic[0].id)["url_oficial"] == fic[0].url_oficial, "y /cita la reconstruye por su id")
vel = cc.resolver_citas_coidh("Caso Velásquez Rodríguez Vs. Honduras, párr. 166")
ok(vel and vel[0]["doc_id"] is None and not [r for r in directos(vel) if r.silo == "coidh"]
   and "pregunta al abogado" in main._coidh_xml_resueltos([], vel),
   "Velásquez Rodríguez ¶166 (fondo, reparaciones, interpretación): no se elige, se pregunta")
malo = cc.resolver_citas_coidh("Almonacid Arellano vs. Chile, párrafo 900")
fm = [r for r in directos(malo) if r.silo == "coidh"]
ok(len(fm) == 1 and fm[0].rol_coidh == "ficha" and "puede no existir" in fm[0].texto,
   "un párrafo que no existe en la resolución ingerida no se inventa")


# ═══════════════════════════════════════════════════════════════ 4 · bloque
print("\n4 · EL BLOQUE DE LA CORTE VA DESPUÉS DEL CONTEXTO NORMAL (revisión A.3)")
normales = [
    main.SearchResult(id="11111111-1111-4111-8111-111111111111", score=0.9, silo="bloque_constitucional",
                      texto="Art. 1o. En los Estados Unidos Mexicanos todas las personas…",
                      ref="Art. 1o CPEUM", origen="Constitución Política de los Estados Unidos Mexicanos"),
    main.SearchResult(id="22222222-2222-4222-8222-222222222222", score=0.8, silo="leyes_federales",
                      texto="Artículo 1o.- El juicio de amparo…", ref="Artículo 1", origen="Ley de Amparo"),
]
resultados = alm + normales               # la búsqueda directa va al frente
ctx = main.format_results_as_xml(resultados) + main._coidh_xml_resueltos(resultados, CASOS["C-154|s|124"])
fin_docs = ctx.index("</documentos>")
ok(all(ctx.index(r.id) > fin_docs for r in alm), "ningún párrafo de la Corte dentro de <documentos>")
ok("<casos_corte_idh>" in ctx and ctx.index("<casos_corte_idh>") > fin_docs, "<casos_corte_idh> después de </documentos>")
bloque = ctx[ctx.index("<casos_corte_idh>"):]
primero = re.search(r'<documento id="([^"]+)"[^>]*rol="(\w+)"', bloque)
ok(primero and primero.group(1) == p124.id and primero.group(2) == "pedido", "dentro, lo pedido primero (¶124)")
ok('pagina="53"' in bloque and 'jerarquia="JURISPRUDENCIA_INTERAMERICANA"' in bloque
   and "[Doc ID: " + p124.id + "]" in bloque, "con página, jerarquía propia y [Doc ID]")
ok("sistema interamericano" in bloque and "UN VOTO NO ES LA CORTE" in bloque and "NUNCA en su lugar" in bloque,
   "con la instrucción de cita: «sistema interamericano», un voto no es la Corte, además de la norma mexicana")
ok(main.format_results_as_xml(alm).count("<documento ") == 0, "format_results_as_xml ya no ordena ni pinta la Corte")
amb = [{"doc_id": None, "evidencia": "caso X", "candidatos": [{"doc_id": "C-1", "caso": "Uno", "fecha": "1988"},
                                                               {"doc_id": "C-2", "caso": "Dos", "fecha": "1989"}]}]
ok("pregunta al abogado" in main._coidh_xml_resueltos([], amb), "una cita ambigua se declara para que el modelo pregunte")
ok(main._coidh_xml_resueltos(normales, []) == "", "sin nada de la Corte, el contexto no cambia ni un carácter")


# ═══════════════════════════════════════════════════════════════ 5 · línea
print("\n5 · LA LÍNEA JURISPRUDENCIAL")
det = lc.pregunta_por_linea("¿Dónde nace el control de convencionalidad?")
ok(det == ("control_convencionalidad", ["origen"]), f"«¿Dónde nace…?» → figura y fase origen ({det})")
lin = correr(lc.traer_linea(QD, det, "jurisprudencia_nacional_v3", tesis_a_dict=main._tesis_a_dict,
                            norma_a_dict=main._norma_a_dict))
xml = lin["xml"] if lin else ""
id124, id27 = POR_LLAVE["C-154|s|124"][0].id, POR_LLAVE["C-101|v:garcia-ramirez|27"][0].id
ok(re.search(r'<hito fecha="2006-09-26"[^>]*parrafo="124" pagina="53"[^>]*>\n<cita>Corte IDH\. Caso Almonacid'
             r'[^<]*párr\. 124\. \[Doc ID: ' + id124 + r'\]', xml) is not None,
   "Almonacid ¶124, pág. 53, 26-sep-2006, con su cita y el [Doc ID] del punto")
ok(re.search(r'<hito fecha="2003-11-25"[^>]*parrafo="27" pagina="165" voto="García Ramírez"[^>]*>\n<cita>'
             r'[^<]*Myrna Mack[^<]*\[Doc ID: ' + id27 + r'\]', xml) is not None,
   "Myrna Mack, voto de García Ramírez ¶27, pág. 165, 25-nov-2003")
ok("sistema interamericano" in xml and "en toda la historia" in xml.split("-->")[0],
   "dice «en el sistema interamericano» y prohíbe «en toda la historia»")
pre = xml[xml.index("<precision_historica>"):xml.index("</precision_historica>")]
ok(pre.index("Myrna Mack") < pre.index("Almonacid") < pre.index("Trabajadores Cesados")
   and "ex officio" in pre and "una especie de" in pre,
   "precisión histórica: votos de García Ramírez → Almonacid ¶124 → ex officio en Cesados ¶128")
fechas = re.findall(r'<hito fecha="([\d-]+)"', xml)
ok(fechas == sorted(fechas) and len(fechas) >= 5, f"hitos en orden cronológico ({len(fechas)} hitos)")
ok("Según lo ingerido hasta 2025-09-30" in xml, "«según lo ingerido hasta [fecha]», nunca «hoy»")
ok("<recepcion_mx>" in xml and 'registro="160589"' in xml and TESIS[0].id in xml,
   "recepción en México: la P. LXVII/2011 traída por registro de jurisprudencia_nacional_v3, con su [Doc ID]")
ok("<tensiones>" not in xml, "si sólo se pregunta el origen, sin tensiones (se ahorra contexto)")

sr = [normales[0]]
dmap = main.build_doc_id_map(sr)
ctx_normal = main.format_results_as_xml(sr)
xml_lin, n_lin = main._coidh_sumar_linea(lin, sr, dmap)
ctx_total = ctx_normal + xml_lin
ok(ctx_total.index("<linea_jurisprudencial") > ctx_total.index("</documentos>"),
   "la línea se AÑADE después del contexto normal")
ok(id124 in dmap and dmap[id124].silo == "coidh" and dmap[id124].pagina == 53 and TESIS[0].id in dmap,
   f"sus hitos y tesis entran al doc_id_map ({n_lin} documentos): [Doc ID] y sello funcionan")
xml2, n2 = main._coidh_sumar_linea(lin, sr, dmap)
ok(n2 == 0 and len(sr) == 1 + n_lin, "lo que ya estaba no se duplica")

det5 = lc.pregunta_por_linea("¿Quiénes están obligados hoy a ejercer el control de convencionalidad?")
lin5 = correr(lc.traer_linea(QD, det5, "jurisprudencia_nacional_v3", tesis_a_dict=main._tesis_a_dict,
                             norma_a_dict=main._norma_a_dict))
x5 = lin5["xml"]
for ll in ("C-154|s|124", "C-218|s|287", "C-220|s|225", "C-221|s|239", "C-259|s|142", "C-450|s|202"):
    ok(POR_LLAVE[ll][0].id in x5, f"pregunta de oro 5: {ll} en la línea")
ok(det5[1][:2] == ["actual", "eje:sujetos"], f"«¿quiénes están obligados hoy?» → eje sujetos y estado actual ({det5[1]})")
fuera = [h for h in lin5["hitos"] if h not in POR_LLAVE]
ok(fuera and 'ingerido="no"' in x5 and all(lc.ficha_id(h) in x5 for h in fuera),
   f"el estado actual trae lo de 2024-2025 fuera del piloto como ficha, ingerido=\"no\" ({fuera})")
ok(lc.ficha_por_id(lc.ficha_id("C-527|s|286"))["pagina"] == 72, "y /cita reconstruye la ficha de un hito")
ok("<tensiones>" in x5 and "<constitucion_mx>" not in x5, "con tensiones; sin México en la pregunta, sin CPEUM")
detm = lc.pregunta_por_linea("¿Cuál es el estado actual del control de convencionalidad en México?")
linm = correr(lc.traer_linea(QD, detm, "jurisprudencia_nacional_v3", tesis_a_dict=main._tesis_a_dict,
                             norma_a_dict=main._norma_a_dict))
xm = linm["xml"]
ok("mexico" in detm[1] and '<resumen parte="mexico">' in xm and _ID_ART19 in xm
   and POR_LLAVE["C-482|s|303"][0].id in xm and "31 de diciembre de 2024" in xm,
   "con México: su resumen (la reforma al art. 19 del 31-dic-2024), García Rodríguez ¶303 y el art. 19")

detp = lc.pregunta_por_linea("prisión preventiva oficiosa para extorsión")
ok(detp == ("control_convencionalidad", ["tema:prision_preventiva_oficiosa"]),
   f"tema sin la palabra «convencionalidad»: prisión preventiva oficiosa ({detp})")
linp = correr(lc.traer_linea(QD, detp, "jurisprudencia_nacional_v3", tesis_a_dict=main._tesis_a_dict,
                             norma_a_dict=main._norma_a_dict))
xp = linp["xml"]
ok(all(POR_LLAVE[ll][0].id in xp for ll in ("C-482|s|301", "C-482|s|303", "C-470|r|8"))
   and id27 not in xp and "hitos del tema" in xp and len(linp["hitos"]) <= 4,
   "trae sus 2-4 hitos (García Rodríguez ¶301 y ¶303, Tzompaxtle res. 8), no la línea entera")
ok("<supervision" in xp and 'fecha="2024-11-26"' in xp and "abierto" in xp,
   "y las supervisiones del 26-nov-2024 que dejaron el punto abierto")
ok('registro="2006224"' in xp and _ID_ART19 in xp, "con la P./J. 20/2014 y el art. 19 vigente")
detf = lc.pregunta_por_linea("Radilla y el fuero militar")
linf = correr(lc.traer_linea(QD, detf, "jurisprudencia_nacional_v3"))
ok(POR_LLAVE["C-209|s|340"][0].id in linf["xml"] and POR_LLAVE["C-209|s|341"][0].id in linf["xml"],
   "«Radilla y el fuero militar» (pregunta de oro 6): Radilla ¶340-341 por el tema fuero militar")

try:
    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")
    tam = {nombre: len(enc.encode(x)) for nombre, x in (
        ("origen", xml), ("obligados hoy", x5), ("estado actual en México", xm), ("prisión preventiva", xp),
        ("fuero militar", linf["xml"]))}
    for nombre, n in tam.items():
        print(f"      tokens del bloque «{nombre}»: {n:,}")
    # El tope del tema subió de 3.5 a 5.5 mil con el pilar (25-sep-2026): el
    # tema ahora trae sus tesis con vigencia, su cronología (reforma del
    # 31-dic-2024, AG 2/2024, art. 166 LA, proyecto de recepción 3/2023) y las
    # reglas de redacción que le tocan. Medido con tesis reales en
    # test_linea_pilar.py: ~5.4 mil; aquí, con tres tesis de prueba, ~4.2 mil.
    ok(max(tam.values()) < 8000 and tam["prisión preventiva"] < 5500,
       "presupuesto: la línea entera < 8 mil tokens; un tema, < 5.5 mil")
except ImportError:
    pass


# ═══════════════════════════════════════════════════════════════ 6 · admins
print("\n6 · COIDH_ACTIVO=admins: EL PILOTO ES SÓLO PARA ADMINISTRADORES")


async def _perfil_falso(user_id):
    return (("platinum_monthly", False, True) if user_id == "u-normal"
            else (None, True, True) if user_id == "u-admin" else (None, False, False))


_original = main._plan_para_redaccion
main._plan_para_redaccion = _perfil_falso
os.environ["COIDH_ACTIVO"] = "admins"
ok(correr(main._coidh_puerta("u-normal"))[0] is False, "un Platinum que no es admin: cerrada")
ok(correr(main._coidh_puerta("u-admin"))[0] is True, "un admin (ADMIN_EMAILS): abierta")
ok(correr(main._coidh_puerta(None))[0] is False, "sin usuario: cerrada")
ok(correr(main._coidh_puerta("u-sin-perfil"))[0] is False, "sin perfil legible: cerrada (no se abre hacia arriba)")
os.environ["COIDH_ACTIVO"] = "on"
ok(correr(main._coidh_puerta("u-normal"))[0] is True, "con «on», abierta para todos")
os.environ["COIDH_ACTIVO"] = "off"
ok(correr(main._coidh_puerta("u-admin"))[0] is False, "con «off», cerrada incluso para admins")
os.environ["COIDH_ACTIVO"] = "cualquier-cosa"
ok(lc.modo() == "admins", "un valor raro cae a «admins», nunca a «on»")
os.environ["COIDH_ACTIVO"] = "admins"


def flujo(user_id, pregunta):
    """Lo que hace el chat, en el mismo orden: resolver sobre la pregunta,
    puerta, búsqueda directa y bloque propio (ver chat_endpoint)."""
    cit = main._extract_legal_citations(pregunta, pregunta_coidh=pregunta)
    if cit["casos_coidh"] and not correr(main._coidh_puerta(user_id))[0]:
        cit["casos_coidh"] = []
    res = correr(main._buscar_articulos_citados(cit))
    return res, main._coidh_xml_resueltos(res, cit["casos_coidh"])


res_n, xml_n = flujo("u-normal", "Almonacid Arellano vs. Chile, párrafo 124")
res_a, xml_a = flujo("u-admin", "Almonacid Arellano vs. Chile, párrafo 124")
ok(not any(r.silo == "coidh" for r in res_n) and xml_n == "", "el no-admin no recibe nada de la Corte")
ok(any(r.silo == "coidh" for r in res_a) and "<casos_corte_idh>" in xml_a, "el admin recibe el ¶124")
main._plan_para_redaccion = _original

# El cableado del chat, leído del código: si alguien mueve una pieza, se ve.
fuente = Path("main.py").read_text(encoding="utf-8")
ce = fuente[fuente.index("async def chat_endpoint("):]
ce = ce[:ce.index("\n@app.")]
i_ext = ce.index("_extract_legal_citations(\n                        last_user_message, pregunta_coidh=_pregunta_coidh")
i_puerta = ce.index("if not await _coidh_permitido():", i_ext)
i_directa = ce.index("_direct_article_lookup(_citations, effective_estado)", i_ext)
i_fmt = ce.index("context_xml = format_results_as_xml(search_results, estado=effective_estado, prose_mode=is_chat_drafting)")
i_bloque = ce.index("_coidh_xml_resueltos(search_results, _citations.get(\"casos_coidh\"))")
ok(i_ext < i_puerta < i_directa, "en el chat, la puerta se consulta ANTES de la búsqueda directa")
ok(i_fmt < i_bloque, "y el bloque de casos se añade DESPUÉS de format_results_as_xml")
i_hilo = ce.index("_hilo_task = asyncio.create_task(")
i_linea = ce.index("_linea_task = asyncio.create_task(_buscar_linea_coidh())")
i_doct = ce.index('yield f"<!--PASO:doctrina|{_autores}-->"')
i_suma = ce.index("_coidh_sumar_linea(_lin, search_results, doc_id_map)")
i_prec = ce.index("# ── Los precedentes ENTRAN al razonamiento")
ok(i_hilo < i_linea and i_doct < i_suma < i_prec,
   "la línea nace tras el hilo y se suma donde la doctrina, antes del prompt")
ok("_has_explicit_citations = bool(_citations)" in ce, "_has_explicit_citations sigue igual (revisión C.18)")
try:
    head = subprocess.run(["git", "show", "HEAD:main.py"], capture_output=True, text=True, check=True).stdout
    tomar = lambda t: t[t.index("DDHH_KEYWORDS = {"):t.index("}", t.index("DDHH_KEYWORDS = {"))]
    ok(tomar(head) == tomar(fuente), "DDHH_KEYWORDS no se tocó")
except Exception as e:
    print(f"      (sin git para comparar DDHH_KEYWORDS: {e})")


# ═══════════════════════════════════════════════════════════════ 7 · ausente
print("\n7 · SIN LA COLECCIÓN (HOY NO EXISTE), LA CONSULTA SIGUE IGUAL")
main.qdrant_client = qdrant_con(coidh=False)
cit = main._extract_legal_citations("Almonacid Arellano vs. Chile, párrafo 124",
                                    pregunta_coidh="Almonacid Arellano vs. Chile, párrafo 124")
cit["registros"] = ["160589"]
res = correr(main._buscar_articulos_citados(cit))
ok([r.silo for r in res] == ["jurisprudencia_nacional_v3"],
   "la búsqueda directa no lanza y sólo trae lo de siempre (ni párrafos ni fichas)")
ok(main._coidh_xml_resueltos(res, cit["casos_coidh"]) == "", "sin bloque de casos")
ok(correr(lc.traer_linea(main.qdrant_client, det, "jurisprudencia_nacional_v3")) is None, "la línea devuelve vacío")
sr7 = list(normales)
dm7 = main.build_doc_id_map(sr7)
antes = main.format_results_as_xml(sr7)
x7, n7 = main._coidh_sumar_linea(None, sr7, dm7)
ok(x7 == "" and n7 == 0 and len(sr7) == 2 and antes == main.format_results_as_xml(sr7),
   "contexto, resultados y doc_id_map idénticos")
ok(correr(lc.documento_completo(main.qdrant_client, highlight_id=id124)) is None, "/document-full cae a su ruta de siempre")


# ═══════════════════════════════════════════════════════════════ 8 · marcadores
print("\n8 · EL CONTRATO LLEGA AL FRONTEND")
main.qdrant_client = QD
marc = main._marcador_fuentes_previas([p124, normales[1]])
dat = json.loads(marc[marc.index(":") + 1:marc.rindex("-->")])
e124, eley = dat[p124.id], dat[normales[1].id]
ALM_OFICIAL = "https://www.corteidh.or.cr/docs/casos/articulos/seriec_154_esp.pdf"
ALM_COPIA = COPIA_BASE + "seriec_154_esp.pdf"
ok(e124["silo"] == "coidh" and e124["pagina"] == 53 and e124["parrafo"] == "124" and e124["llave"] == "C-154|s|124"
   and e124["pdf_url"] == ALM_COPIA and e124["url_oficial"] == ALM_OFICIAL and "#" not in e124["pdf_url"]
   and e124["pdf_sha1"] == MANIFIESTO[ALM_OFICIAL]["sha1"] and e124["ancla"]
   and e124["cita_canonica"] and e124["tipo"] == "sentencia_coidh" and e124["serie"] == "Serie C No. 154"
   and e124["caso"] and e124["fecha"] == "2006-09-26" and "voto_autor" in e124 and e124["seg"] == "sentencia",
   "FUENTES_PREVIAS: los 14 campos del contrato; pdf_url = la copia, url_oficial = corteidh, pdf_sha1 del manifiesto")
ok(not any(k in eley for k in ("pagina", "llave", "ancla", "cita_canonica", "url_oficial", "pdf_sha1",
                               "pagina_impresa", "obra", "autor", "anio")),
   "y ni una clave nueva en la fuente de una ley")
cita = correr(main.resolver_cita(p124.id))
ok(cita["silo"] == "coidh" and cita["pagina"] == 53 and cita["parrafo"] == "124" and cita["pdf_url"] == ALM_COPIA
   and cita["url_oficial"] == ALM_OFICIAL and cita["pdf_sha1"] == MANIFIESTO[ALM_OFICIAL]["sha1"]
   and cita["texto"] and cita["origen"].startswith("Corte IDH"), "/cita/{doc_id} devuelve el contrato (con la copia)")
cita_f = correr(main.resolver_cita(lc.ficha_id("C-527|s|286")))
ok(cita_f["silo"] == "coidh" and cita_f["pagina"] == 72 and "seriec_527" in cita_f["pdf_url"],
   "/cita resuelve también la ficha de un hito sin ingerir")
ok("coidh" in main._COLECCIONES_CITA and main._COLECCIONES_CITA.index("coidh") < main._COLECCIONES_CITA.index("doctrina"),
   "_COLECCIONES_CITA busca en «coidh» antes que en «doctrina»")
ver = correr(main._fuentes_ya_verificadas([p124.id, lc.ficha_id("C-527|s|286")]))
ok([v.silo for v in ver] == ["coidh", "coidh"] and ver[0].pagina == 53 and ver[1].pagina == 72
   and ver[0].pdf_url == ALM_COPIA and ver[0].url_oficial == ALM_OFICIAL,
   "las fuentes ya verificadas vuelven con su contrato (también la ficha), con la copia")
full = correr(main.get_full_document(origen=p124.origen, highlight_chunk_id=p124.id))
trozos = full.texto_completo.split("\n\n")
# El botón «PDF» del modal es un ENLACE a otra pestaña: va a la Corte, como
# antes; la copia es para dibujar y viaja en metadata (revisión 25-sep-2026).
ok(full.source_doc_url == ALM_OFICIAL and full.metadata.get("url_oficial") == ALM_OFICIAL
   and full.metadata.get("pdf_url") == ALM_COPIA and full.metadata.get("pdf_sha1") == MANIFIESTO[ALM_OFICIAL]["sha1"]
   and full.total_chunks == len(trozos)
   and trozos[full.highlight_chunk_index].startswith("124. La Corte es consciente")
   and full.metadata.get("pagina") == 53,
   "/document-full: la resolución por `orden`, el botón PDF a la URL oficial, la copia en metadata y el ¶124 señalado")
ok(main.resolver_pdf(None, "x", "coidh", url_oficial="https://www.corteidh.or.cr/a.pdf#page=3")
   == "https://www.corteidh.or.cr/a.pdf", "resolver_pdf: sin copia, la url_oficial sin el #page")
ok(main.resolver_pdf(None, "x", "coidh", url_oficial=ALM_OFICIAL + "#page=53") == ALM_COPIA
   and main.resolver_pdf(None, "x", "coidh", url_oficial="http://corteidh.or.cr/docs/casos/articulos/seriec_154_esp.pdf")
   == ALM_COPIA,
   "resolver_pdf: con copia, la copia (también desde http:// y sin www.)")


# ═══════════════════════════════════════════════════════════════ 9 · revisión
# Lo que encontró la revisión adversarial del 25-sep-2026 (cada uno fallaba
# antes de su arreglo).
print("\n9 · REVISIÓN ADVERSARIAL: SELECTOR, AMBIGUAS SIN COLECCIÓN, ESCRITOS Y FICHAS")
main._plan_para_redaccion = _perfil_falso
os.environ["COIDH_ACTIVO"] = "admins"
ok(correr(main._coidh_puerta("u-admin", frozenset({"federal", "estatal"}))) == (False, "selector"),
   "«constitucional» apagado en el selector cierra la puerta, también al admin")
ok(correr(main._coidh_puerta("u-admin", frozenset({"constitucional"})))[0] is True,
   "con «constitucional» encendido, el admin pasa")
os.environ["COIDH_ACTIVO"] = "on"
ok(correr(main._coidh_puerta("u-normal", frozenset({"jurisprudencia"})))[0] is False,
   "con «on» el selector también manda")
# El pilar (25-sep-2026): con sólo «jurisprudencia», la LÍNEA pasa en modo
# «solo_mx» (sin nada interamericano); los casos citados siguen cerrados.
ok(correr(main._coidh_puerta("u-normal", frozenset({"jurisprudencia"}), linea=True)) == (True, "on:solo_mx"),
   "con «on» y sólo «jurisprudencia», la línea pasa en modo solo_mx")
os.environ["COIDH_ACTIVO"] = "admins"
ok(correr(main._coidh_puerta("u-admin", frozenset({"jurisprudencia"}), linea=True)) == (True, "admins:solo_mx")
   and correr(main._coidh_puerta("u-admin", frozenset({"federal"}), linea=True)) == (False, "selector"),
   "con «admins»: solo_mx para el admin; con los dos rubros apagados, cerrada")
main._plan_para_redaccion = _original
ok("_coidh_puerta(request.user_id, _fuentes_elegidas)" in ce, "el chat le pasa a la puerta el selector de la consulta")
i_ver = ce.index("_verificadas = await _fuentes_ya_verificadas(request.fuentes_previas)")
i_ver_p = ce.index('if any(r.silo == "coidh" for r in _verificadas) and not await _coidh_permitido():', i_ver)
# Desde el sello de vigencia (25-sep-2026) el XML de las verificadas lo arma
# `_xml_verificadas` (separa las que perdieron vigencia): el orden que se
# comprueba es el mismo.
i_ver_x = ce.index("_xml_verificadas(_nuevas", i_ver)
ok(i_ver < i_ver_p < i_ver_x, "lo que vuelve por FUENTES_PREVIAS pasa por la puerta antes de entrar al contexto")

main.qdrant_client = qdrant_con(coidh=False)
cit_v = {"articles": [], "registros": [], "tesis_nums": [], "casos_coidh": list(vel)}
res_v = correr(main._buscar_articulos_citados(cit_v))
ok(cit_v["casos_coidh"] == [] and main._coidh_xml_resueltos(res_v, cit_v["casos_coidh"]) == "",
   "sin colección, una cita ambigua tampoco mete <casos_corte_idh> (la consulta sigue igual)")
main.qdrant_client = QD
cit_v = {"articles": [], "registros": [], "tesis_nums": [], "casos_coidh": list(vel)}
res_v = correr(main._buscar_articulos_citados(cit_v))
ok("pregunta al abogado" in main._coidh_xml_resueltos(res_v, cit_v["casos_coidh"]),
   "con la colección, la ambigua se sigue declarando para que el modelo pregunte")

_escrito = ("DOCUMENTO ADJUNTO: DEMANDA DE AMPARO … como resolvió la Corte IDH en el Caso Almonacid "
            "Arellano y otros Vs. Chile, párr. 124 …")
hist_doc = [SimpleNamespace(role="user", content=_escrito),
            SimpleNamespace(role="assistant", content="Análisis del escrito…"),
            SimpleNamespace(role="user", content="¿qué dice el párrafo 5?")]
hist_sin = [SimpleNamespace(role="user", content=_escrito.replace("DOCUMENTO ADJUNTO: ", ""))] + hist_doc[1:]
ok(lc.previo_de_historial(hist_sin, "¿qué dice el párrafo 5?") == "C-154|s|124"
   and lc.previo_de_historial(hist_doc, "¿qué dice el párrafo 5?") is None,
   "tras un escrito adjunto, «¿qué dice el párrafo 5?» no hereda el caso que cita el escrito (B.8)")

_por = lc._hitos_por_llave("control_convencionalidad")
fx = lc.ficha_hito(_por["C-220|s|233"])
ok(fx["parrafo"] == "233" and fx["serie"] == "Serie C No. 220"
   and (fx["cita_canonica"] or "").endswith("Serie C No. 220, párr. 233."),
   "el hito extra de un tema (C-220 ¶233) sin ingerir sale con párrafo, serie y cita canónica")


# ═══════════════════════════════════════════════════════════════ 10 · la copia
# 25-sep-2026: «No se pudo abrir el PDF aquí» en todas las sentencias.
# Cloudflare de corteidh.or.cr reta con 403 a todo cliente automático, también
# al proxy; el visor abre la copia verificada de legal-docs/CorteIDH.
print("\n10 · EL VISOR ABRE LA COPIA DE LEGAL-DOCS; LA CITA SIGUE SIENDO LA OFICIAL")
docs_col = {}
for p in PUNTOS:
    docs_col.setdefault(p.payload["doc_id"], p)
ok(len(MANIFIESTO) == 58 and len(lc.copias()) == 58, f"el manifiesto trae 58 copias ({len(MANIFIESTO)})")
malos, por_sha1 = [], []
for doc_id, p in sorted(docs_col.items()):
    c = lc.contrato(p.id, p.payload)
    oficial = p.payload["url_oficial"]
    m = MANIFIESTO.get(oficial)
    if m is None:
        # Los puntos en seco son de antes de --aceptar-alterno: cuatro traen la
        # URL «_esp1/_esp2» del catálogo y se trocearon con el «_esp». Mismo
        # sha1 = mismo archivo: la copia se encuentra por el sha1.
        m = next((x for x in MANIFIESTO.values() if x["sha1"] == p.payload.get("pdf_sha1")), {})
        por_sha1.append(doc_id)
    if not (c["pdf_url"] == m.get("copia") and c["pdf_url"].startswith(COPIA_BASE) and c["url_oficial"] == oficial
            and c["pdf_sha1"] == m.get("sha1") == p.payload.get("pdf_sha1") and m.get("doc_id") == doc_id
            and "#" not in c["pdf_url"]):
        malos.append(doc_id)
ok(len(docs_col) == 58 and not malos,
   f"las {len(docs_col)} resoluciones de la colección abren su copia, con el MISMO sha1 que se troceó ({malos[:5]})")
ok(por_sha1 == ["C-217", "C-218", "C-221", "C-253"],
   f"las cuatro del «_esp» alterno se encuentran por el sha1, no por el número de caso ({por_sha1})")
p217 = docs_col["C-217"]
e217 = json.loads((lambda m: m[m.index(":") + 1:m.rindex("-->")])(
    main._marcador_fuentes_previas([main._sr_de(lc.contrato(p217.id, p217.payload))])))[str(p217.id)]
ok(e217["pdf_url"] == COPIA_BASE + "seriec_217_esp.pdf" and e217["url_oficial"].endswith("seriec_217_esp1.pdf")
   and e217["pdf_sha1"] == p217.payload["pdf_sha1"],
   "y el marcador de fuentes también manda esa copia (por el sha1), con la URL oficial intacta")
ok(lc.pdf_de("https://www.corteidh.or.cr/docs/casos/articulos/seriec_217_esp1.pdf")[0]
   == "https://www.corteidh.or.cr/docs/casos/articulos/seriec_217_esp1.pdf",
   "sin sha1, una URL que no está en el manifiesto NO se cambia por la de otro archivo del mismo caso")
_hito_alm = lc._hitos_por_llave("control_convencionalidad").get("C-154|s|124")
fh = lc.ficha_hito(_hito_alm) if _hito_alm else {}
ok(fh.get("pdf_url") == ALM_COPIA and fh.get("url_oficial") == ALM_OFICIAL
   and fh.get("pdf_sha1") == MANIFIESTO[ALM_OFICIAL]["sha1"],
   "la ficha de un hito con copia (Almonacid ¶124): la copia para el visor, la oficial para la cita")
fc = lc.ficha_catalogo("C-154", parrafo=124) or {}
ok(fc.get("pdf_url") == ALM_COPIA and fc.get("url_oficial") == ALM_OFICIAL and fc.get("pdf_sha1"),
   "y la ficha del catálogo de una resolución con copia, igual")
ok(lc.canon_corteidh("HTTP://CorteIDH.or.cr/docs/casos/articulos/seriec_154_esp.pdf#page=3") == ALM_OFICIAL
   and lc.canon_corteidh("https://archivos.juridicas.unam.mx/www/bjv/libros/8/3632/11.pdf") is None
   and lc.canon_corteidh(None) is None,
   "la URL oficial se canoniza como en el frontend (proxyPdf.canonCorteIDH); otro host no es de la Corte")
_ruta = lc.RUTA_COPIAS
try:
    lc.RUTA_COPIAS = Path("/no/existe/coidh_copias.json")
    lc.copias.cache_clear()
    sin = lc.contrato(p124.id, POR_LLAVE["C-154|s|124"][0].payload)
    ok(sin["pdf_url"] == ALM_OFICIAL and sin["url_oficial"] == ALM_OFICIAL
       and sin["pdf_sha1"] == POR_LLAVE["C-154|s|124"][0].payload.get("pdf_sha1"),
       "sin manifiesto (roto o ausente), todo vuelve a la URL oficial: no se cae nada")
finally:
    lc.RUTA_COPIAS = _ruta
    lc.copias.cache_clear()
ok(lc.pdf_de(ALM_OFICIAL)[0] == ALM_COPIA, "y con el manifiesto de vuelta, otra vez la copia")
# Revisión (25-sep-2026): si el payload trae un sha1 que NO es el de la copia
# de su URL, las páginas se midieron en otro archivo: no se abre esa copia.
ok(lc.pdf_de(ALM_OFICIAL, "0" * 40) == (ALM_OFICIAL, "0" * 40),
   "sha1 del payload distinto del de la copia: la URL oficial, no una copia con otras páginas")
ok(lc.pdf_de(ALM_OFICIAL, MANIFIESTO[ALM_OFICIAL]["sha1"].upper())[0] == ALM_COPIA,
   "y con el mismo sha1 (sin importar mayúsculas), la copia")
_e_mal = main._campos_coidh(main._sr_de(dict(lc.contrato(p124.id, dict(POR_LLAVE["C-154|s|124"][0].payload,
                                                                         pdf_sha1="0" * 40)))))
ok(_e_mal["pdf_url"] == ALM_OFICIAL and _e_mal["url_oficial"] == ALM_OFICIAL,
   "y el marcador tampoco la manda (resolver_pdf no le gana al sha1)")


print()
if FALLOS:
    print(f"FALLAN {len(FALLOS)}:")
    for f in FALLOS:
        print("  ·", f)
    sys.exit(1)
print("TODO PASA")
