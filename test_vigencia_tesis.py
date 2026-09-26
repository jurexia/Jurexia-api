"""El sello de vigencia de las tesis — 25-sep-2026.

    .venv/bin/python test_vigencia_tesis.py

Sin red y sin gastar API: Qdrant es falso. El caso que lo motivó: el chat le
dio a David como vigentes la P. X/2015 (10a.) (2009817) y la P. IX/2015
(2009816), ABANDONADAS por la P./J. 2/2022 (11a.) (2024159). Se comprueba:
el índice y la capa curada, la etiqueta, el XML (atributos y línea visible,
con TESIS_SOLO_RUBRO), que la sustituta se trae (sin duplicar, con tope, en
cadena y con plazo), la búsqueda directa por registro y por clave, la regla del
prompt, que FUENTES_PREVIAS no blinda una abandonada, los marcadores para la
app, el registro de Render, el taller y que sin el índice nada se rompe.
"""
import asyncio
import contextlib
import importlib.util
import io
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, os.getcwd())
import main  # noqa: E402
import vigencia_tesis as vt  # noqa: E402
import fase6_rag as fr  # noqa: E402
import fase6_estudio as fe  # noqa: E402
import toulmin  # noqa: E402
from qdrant_client.models import MatchAny, MatchValue  # noqa: E402

FALLOS = []


def ok(cond, que):
    print(f"   {'PASA ' if cond else 'FALLA'}  {que}")
    if not cond:
        FALLOS.append(que)


def correr(coro):
    return asyncio.run(coro)


V3 = "jurisprudencia_nacional_v3"
main._NOMBRES_FEDERALES["ts"] = 9e18          # que no intente refrescar contra la red


# ═══════════════════════════════════════════════════════════════ Qdrant falso

class ColeccionAusente(Exception):
    pass


class QdrantFalso:
    def __init__(self, colecciones, demora=0.0):
        self.col = colecciones
        self.demora = demora
        self.llamadas = []

    def _pts(self, nombre):
        if nombre not in self.col:
            raise ColeccionAusente(nombre)
        return self.col[nombre]

    @staticmethod
    def _cumple(pl, flt):
        for c in (getattr(flt, "must", None) or []):
            if c.key not in pl:
                # La v3 no indexa «tesis» ni «tesis_num»: Qdrant responde 400.
                raise ValueError(f"Index required but not found for \"{c.key}\"")
            v = pl.get(c.key)
            if isinstance(c.match, MatchAny):
                if v not in c.match.any:
                    return False
            elif isinstance(c.match, MatchValue):
                if v != c.match.value:
                    return False
        return True

    @staticmethod
    def _validar(flt):
        """Lo que la v3 rechaza ANTES de mirar un solo punto (26-sep-2026):
        `registro` y `clave_tesis` sólo tienen índice keyword, así que un
        entero da 400 aunque el registro exista. Antes este doble dejaba pasar
        los enteros —simplemente no casaban— y por eso ninguna prueba vio que
        una sustituta ausente tumbaba a todas las demás."""
        for c in (getattr(flt, "must", None) or []):
            vals = c.match.any if isinstance(c.match, MatchAny) else [getattr(c.match, "value", None)]
            if c.key in ("registro", "clave_tesis") and any(isinstance(v, int) for v in vals):
                raise ValueError(f'Unexpected Response: 400 (Bad Request) Index required but not found '
                                 f'for "{c.key}" of one of the following types: [integer]')

    async def scroll(self, collection_name, scroll_filter=None, limit=10, with_payload=True,
                     with_vectors=False, **kw):
        self.llamadas.append(("scroll", collection_name, scroll_filter))
        self._validar(scroll_filter)
        if self.demora:
            await asyncio.sleep(self.demora)
        pts = [p for p in self._pts(collection_name) if self._cumple(p.payload, scroll_filter)]
        return pts[:limit], None

    async def retrieve(self, collection_name, ids, with_payload=True, **kw):
        quiero = {str(i).lower() for i in ids}
        return [p for p in self._pts(collection_name) if str(p.id).lower() in quiero]


def punto(n, registro, clave, rubro, tipo="TESIS AISLADA", instancia="Pleno", epoca=None):
    """Una tesis de mentira con los campos que el código lee de la v3."""
    pl = {"registro": registro, "clave_tesis": clave, "rubro": rubro,
          "texto": f"Cuerpo de la tesis {registro}, que NO debe llegar al modelo con TESIS_SOLO_RUBRO.",
          "tipo": tipo, "instancia": instancia, "materia": "Común"}
    if epoca:
        pl["epoca"] = epoca
    return SimpleNamespace(id=f"00000000-0000-4000-8000-{n:012d}", payload=pl)


TESIS = [
    punto(1, "2009817", "P. X/2015 (10a.)", "CONTROL CONSTITUCIONAL Y CONVENCIONAL. SU EJERCICIO EN AMPARO DIRECTO."),
    punto(2, "2009816", "P. IX/2015 (10a.)", "CONTROL CONSTITUCIONAL Y CONVENCIONAL. LÍMITES EN AMPARO DIRECTO."),
    punto(3, "2024159", "P./J. 2/2022 (11a.)",
          "CONTROL DE REGULARIDAD CONSTITUCIONAL. CONTENIDO Y ALCANCE DEL DEBER DE LOS ÓRGANOS "
          "JURISDICCIONALES DEL PODER JUDICIAL DE LA FEDERACIÓN [ABANDONO DE LAS TESIS AISLADAS "
          "P. IX/2015 (10a.) Y P. X/2015 (10a.)].", tipo="JURISPRUDENCIA"),
    punto(4, "159870", "I.3o.C.106 K", "UNA TESIS DE COLEGIADO ABANDONADA EN PARTE."),
    punto(5, "2019453", "I.3o.C.106 K (10a.)", "LA TESIS DE COLEGIADO QUE LA REEMPLAZA."),
    punto(6, "2015368", "II.2o.P.1 P (10a.)", "UNA TESIS ACLARADA."),
    punto(7, "2026728", "II.2o.P.32 P (11a.)", "LA VERSIÓN ACLARADA."),
    punto(8, "160584", "P. LXVI/2011 (9a.)", "CRITERIOS EMITIDOS POR LA CORTE INTERAMERICANA … SON ORIENTADORES."),
    punto(9, "2006225", "P./J. 21/2014 (10a.)", "JURISPRUDENCIA EMITIDA POR LA CORTE INTERAMERICANA … ES VINCULANTE.",
          tipo="JURISPRUDENCIA"),
]
POR_REG = {p.payload["registro"]: p for p in TESIS}


def sr(registro, score=0.5):
    """Una tesis como la arma el chat (mismo conversor que la línea y la búsqueda directa)."""
    p = POR_REG[registro]
    r = main._sr_de(main._tesis_a_dict(p.id, p.payload, V3))
    r.score = score
    return r


def fuera_de_llamadas(q):
    return [c for c in q.llamadas if c[0] == "scroll"]


# ═══════════════════════════════════════════════════════════════ 1 · el índice
print("\n1 · EL ÍNDICE Y LA CAPA CURADA")
crudo = json.loads(vt.RUTA_INDICE.read_text(encoding="utf-8"))
# 558 al generarlo el 25-sep-2026; al regenerarlo con la carga semanal del
# Semanario puede crecer, así que se exige un piso y no el número exacto.
ok(set(crudo) >= {"generado", "fuente", "n", "tesis"} and crudo["n"] == len(crudo["tesis"]) >= 550,
   f"datos/vigencia_tesis.json: {{generado, fuente, n, tesis}} con sus tesis (tiene {crudo.get('n')})")
CAMPOS = {"estado", "parcial", "por_registro", "por_clave", "por_rubro", "por_resolucion", "desde", "fuente", "nota"}
# 26-sep-2026: `alcance` y `por_intermedias` sólo en las entradas que los tienen.
OPCIONALES = {"alcance", "por_intermedias"}
ok(all(CAMPOS <= set(v) <= CAMPOS | OPCIONALES for v in crudo["tesis"].values()),
   "cada entrada lleva los campos del contrato (y alcance / por_intermedias sólo si los tiene)")
ok(sum(1 for v in crudo["tesis"].values() if set(v) & OPCIONALES) == 3,
   "sólo 3 entradas llevan los opcionales (164500, 190237, 192096): el resto no cambió ni un byte")
ok(all(len(v["nota"] or "") <= 400 and len(v["por_rubro"] or "") <= 200 and len(v.get("alcance") or "") <= 240
       for v in crudo["tesis"].values()),
   "nota ≤ 400, por_rubro ≤ 200 y alcance ≤ 240 caracteres")

# EL ALCANCE Y LA CADENA (hallazgo 8, 26-sep-2026). La 164500 figuraba como
# interrumpida EN TODO por la 3a./J. 23/91, que también perdió vigencia; el SJF
# dice «en la parte relativa» y que a la 3a./J. 23/91 la interrumpió a su vez
# la P./J. 55/2003 (183349). La 190237 y la 192096, abandonadas «por lo que se
# refiere a los efectos de … la veda electoral», figuraban como totales.
v164 = vt.de("164500")
ok(v164["parcial"] and v164.get("alcance") == "en la parte relativa" and v164["por_clave"] == "P./J. 55/2003"
   and v164["por_registro"] == "183349" and v164.get("por_intermedias") == ["3a./J. 23/91"]
   and v164["desde"] == "1991-05",
   "164500 → EN PARTE («en la parte relativa»), reemplazo el último eslabón: P./J. 55/2003 (183349), "
   "vía la 3a./J. 23/91; desde mayo de 1991 (cuando la perdió ella)")
ok(vt.cadena("164500") == ["183349"], "y la sustituta que se trae es la 183349, no una tesis también interrumpida")
for _r in ("190237", "192096"):
    _v = vt.de(_r)
    ok(_v["parcial"] and (_v.get("alcance") or "").startswith("por lo que se refiere a los efectos")
       and "veda electoral" in _v["alcance"] and _v["alcance"].endswith("Estados Unidos Mexicanos"),
       f"{_r} → abandonada EN PARTE, con su alcance entero («por lo que se refiere a … veda electoral …»)")
ok(not any((v["nota"] or "").endswith(" pu") for v in crudo["tesis"].values())
   and crudo["tesis"]["2009817"]["nota"].endswith("…"),
   "la nota se corta en palabra entera con «…» (el prototipo dejaba «…, pu»)")
ESTADOS = {"abandonada", "interrumpida", "modificada", "sin_efectos", "superada", "sustituida",
           "texto_sustituido", "aclarada"}
ok({v["estado"] for v in crudo["tesis"].values()} <= ESTADOS,
   "sólo pérdidas expresas: ningún «contendió» ni «sin materia» (ver_jurisprudencia queda fuera)")
_cur = json.loads(vt.RUTA_CURADA.read_text(encoding="utf-8"))["tesis"]
ok(len(vt.indice()) == len(set(crudo["tesis"]) | set(_cur)),
   f"el índice en memoria: expresas + curadas ({len(crudo['tesis'])} + {len(_cur)} = {len(vt.indice())})")

for reg, clave in (("2009817", "P. X/2015"), ("2009816", "P. IX/2015")):
    v = vt.de(reg)
    ok(v and v["estado"] == "abandonada" and v["por_registro"] == "2024159"
       and v["por_clave"] == "P./J. 2/2022 (11a.)" and v["desde"] == "2022-02-11"
       and v["fuente"] == "nota_propia" and not v["parcial"],
       f"{reg} ({clave}) → ABANDONADA por 2024159 (P./J. 2/2022), desde 2022-02-11, nota propia")
ok(vt.de("2024159") is None and vt.de("2006225") is None, "las que reemplazan están vigentes: sin sello")
ok(vt.de(2009817)["registro"] == "2009817" and vt.de(" 2009817 ") is not None, "acepta el registro como número o con espacios")
ok(vt.de(None) is None and vt.de("") is None and vt.de("P. X/2015") is None, "lo que no es un registro da None")
_c = vt.de("2009817")
_c["estado"] = "tocada"
ok(vt.de("2009817")["estado"] == "abandonada", "de() devuelve una copia: quien la toca no toca el índice")

v584 = vt.de("160584")
ok(v584 and v584["fuente"] == "curaduria" and v584["por_registro"] == "2006225" and not v584["parcial"],
   "160584 (P. LXVI/2011, «orientadores») → curada: superada por 2006225 (P./J. 21/2014)")
v526 = vt.de("160526")
ok(v526 and v526["fuente"] == "curaduria" and v526["parcial"] and "inciso d)" in v526["alcance"],
   "160526 (P. LXVIII/2011) → curada y PARCIAL: sólo el inciso d)")
cur = json.loads(vt.RUTA_CURADA.read_text(encoding="utf-8"))
ok(all(e.get("justificacion", {}).get("afectada") and e["justificacion"].get("sustituta")
       for e in cur["tesis"].values()), "cada entrada curada trae la cita de las dos tesis")
ok(vt.cadena("2009817") == ["2024159"], "cadena(2009817) = [2024159]")

# La curada nunca pisa a la expresa.
with tempfile.TemporaryDirectory() as d:
    _ri, _rc = vt.RUTA_INDICE, vt.RUTA_CURADA
    Path(d, "e.json").write_text(json.dumps({"tesis": {"111111": {"estado": "abandonada", "por_registro": "2",
                                                                   "fuente": "nota_propia"}}}))
    Path(d, "c.json").write_text(json.dumps({"tesis": {"111111": {"estado": "superada", "por_registro": "3",
                                                                   "fuente": "curaduria"}}}))
    vt.RUTA_INDICE, vt.RUTA_CURADA = Path(d, "e.json"), Path(d, "c.json")
    vt.indice.cache_clear()
    ok(vt.de("111111")["fuente"] == "nota_propia", "si las dos capas hablan de la misma tesis, manda la expresa")
    vt.RUTA_INDICE, vt.RUTA_CURADA = _ri, _rc
    vt.indice.cache_clear()


# ═══════════════════════════════════════════════════════════════ 2 · la etiqueta
print("\n2 · LA ETIQUETA")
ok(vt.etiqueta(vt.de("2009817")) ==
   "ABANDONADA por la P./J. 2/2022 (11a.), registro 2024159, desde el 11 de febrero de 2022",
   f"«{vt.etiqueta(vt.de('2009817'))}»")
e159 = vt.etiqueta(vt.de("159870"))
ok(e159.startswith("ABANDONADA EN PARTE por la I.3o.C.106 K (10a.), registro 2019453"), f"parcial: «{e159}»")
e584 = vt.etiqueta(v584)
ok(e584.startswith("SUPERADA EN LOS HECHOS por la P./J. 21/2014 (10a.), registro 2006225, desde el 28 de abril de 2014")
   and e584.endswith("(el Semanario no lo anota)"), f"curada: «{e584}»")
e526 = vt.etiqueta(v526)
ok("EN PARTE, EN LOS HECHOS" in e526 and "inciso d)" in e526, f"curada parcial con su alcance: «{e526[:90]}…»")
e_res = vt.etiqueta(vt.de("164587"))
ok("al resolverse el amparo en revisión 151/2021" in e_res, f"sin tesis sustituta, el asunto: «{e_res}»")
ok(vt.etiqueta({"estado": "interrumpida", "por_clave": "X", "desde": "2013-07"}).endswith("desde julio de 2013"),
   "«desde» sólo con mes y año")
ok(vt.etiqueta(None) == "" and vt.atributos_xml(None) == "" and vt.linea_visible(None) == ""
   and vt.marcador(None) is None, "sin pérdida, todo vacío")
ok(vt.linea_visible(vt.de("2015368")).startswith("⚠️ TEXTO CORREGIDO: ACLARADA"),
   "una aclaración se avisa como texto corregido, no como «perdió vigencia»")
e164 = vt.etiqueta(vt.de("164500"))
ok(e164 == ("INTERRUMPIDA EN PARTE —en la parte relativa— por la 3a./J. 23/91 y, a su vez, por la "
            "P./J. 55/2003, registro 183349, desde mayo de 1991"), f"la cadena y el alcance, dichos: «{e164}»")
ok(vt.linea_visible(vt.de("164500")).startswith("⚠️ PERDIÓ VIGENCIA EN PARTE:")
   and vt.linea_visible(vt.de("190237")).startswith("⚠️ PERDIÓ VIGENCIA EN PARTE: ABANDONADA EN PARTE —por lo que"),
   "ya no «No la presentes como vigente» a secas: «EN PARTE», con el inciso")


# ═══════════════════════════════════════════════════════════════ 3 · el XML
print("\n3 · EL XML DEL MODELO: ATRIBUTOS Y LÍNEA VISIBLE")
os.environ["TESIS_SOLO_RUBRO"] = "1"
a817, a4159 = sr("2009817"), sr("2024159", score=0.4)
xml = main.format_results_as_xml([a817, a4159])
doc817 = re.search(r'<documento id="%s"[^>]*>(.*?)</documento>' % a817.id, xml, re.S)
tag817 = re.search(r'<documento id="%s"[^>]*>' % a817.id, xml).group(0)
ok(all(x in tag817 for x in ('vigencia="abandonada"', 'reemplazada_por="2024159"',
                             'reemplazo_clave="P./J. 2/2022 (11a.)"', 'vigencia_desde="2022-02-11"')),
   "<documento> de 2009817 lleva vigencia, reemplazada_por, reemplazo_clave y vigencia_desde")
cuerpo817 = doc817.group(1).strip().split("\n")
ok(cuerpo817[0].startswith("[TIPO:") and cuerpo817[1].startswith("⚠️ PERDIÓ VIGENCIA: ABANDONADA por la P./J. 2/2022")
   and cuerpo817[2].startswith("CONTROL CONSTITUCIONAL Y CONVENCIONAL"),
   "la línea visible va entre la cabecera y el rubro")
ok("Nota del Semanario en esta tesis: «La presente tesis fue abandonada" in doc817.group(1),
   "y trae la nota literal del Semanario")
ok("Cuerpo de la tesis 2009817" not in xml, "TESIS_SOLO_RUBRO se respeta: el cuerpo no llega al modelo")
tag4159 = re.search(r'<documento id="%s"[^>]*>' % a4159.id, xml).group(0)
ok("vigencia=" not in tag4159 and "PERDIÓ VIGENCIA" not in xml.split(tag4159)[1].split("</documento>")[0],
   "2024159 (vigente) sin atributos ni aviso")
ok("INSTRUCCIÓN VIGENCIA" in xml, "con una tesis sin vigencia, el XML lleva su instrucción")
ok("INSTRUCCIÓN VIGENCIA" not in main.format_results_as_xml([a4159]), "sin ninguna, no se infla el prompt")
ok('vigencia_parcial="si"' in main.format_results_as_xml([sr("159870")])
   and "⚠️ PERDIÓ VIGENCIA EN PARTE" in main.format_results_as_xml([sr("159870")]), "la parcial se marca como parcial")
ok('vigencia_fuente="curaduria"' in main.format_results_as_xml([sr("160584")]), "la curada se distingue en el tag")

# DE QUIÉN ES LA PÉRDIDA (hallazgo 18, 26-sep-2026): la instrucción decía «según
# el Semanario» también para las curadas, que el Semanario NO anota.
def _instr(xml_):
    return re.search(r"<!-- INSTRUCCIÓN VIGENCIA:.*?-->", xml_, re.S).group(0)


i_cur = _instr(main.format_results_as_xml([sr("160584")]))
i_exp = _instr(main.format_results_as_xml([sr("2009817")]))
i_mix = _instr(main.format_results_as_xml([sr("2009817"), sr("160584")]))
ok("según el Semanario" not in i_cur and "consta en el Semanario" not in i_cur
   and "curaduría de Iurexia" in i_cur and "no se lo atribuyas al Semanario" in i_cur,
   "sólo curadas: la instrucción NO se la atribuye al Semanario y dice que es curaduría de Iurexia")
ok("consta en el Semanario" in i_exp and "curaduría" not in i_exp,
   "sólo expresas: la pérdida consta en el Semanario, y no se habla de curaduría")
ok("consta en el Semanario" in i_mix and "curaduría de Iurexia" in i_mix, "mezcladas: cada origen con el suyo")
ok(all("parte afectada" in x and "parte abandonada" not in x for x in (i_cur, i_exp, i_mix)),
   "«sólo en la parte afectada», no «abandonada» (la parcial curada está superada)")
os.environ["TESIS_SOLO_RUBRO"] = "0"
xml0 = main.format_results_as_xml([sr("2009817")])
ok("Cuerpo de la tesis 2009817" in xml0 and "⚠️ PERDIÓ VIGENCIA" in xml0, "con TESIS_SOLO_RUBRO=0 van la línea y el cuerpo")
os.environ["TESIS_SOLO_RUBRO"] = "1"
# Un SearchResult armado SIN vigencia (otro constructor) la obtiene al consumirse, por registro.
crudo817 = main.SearchResult(id="x-1", score=0.3, texto="[REGISTRO: 2009817]\nRUBRO", silo=V3, registro="2009817")
ok('vigencia="abandonada"' in main.format_results_as_xml([crudo817]) and crudo817.vigencia,
   "una tesis armada sin el campo lo obtiene por registro al consumirse (_vigencia_sr)")
otro = main.SearchResult(id="x-2", score=0.3, texto="Artículo 1", silo="bloque_constitucional", registro="2009817")
ok(main._vigencia_sr(otro) is None, "fuera de la jurisprudencia no se sella aunque un campo coincida")


# ═══════════════════════════════════════════════════════════════ 4 · la sustituta
print("\n4 · LA SUSTITUTA SE TRAE POR SU REGISTRO")
Q = QdrantFalso({V3: TESIS})
res = [sr("2009817", 0.61), sr("2009816", 0.59), sr("2006225", 0.2)]
mapa = main.build_doc_id_map(res)
xml_s, n = correr(main._sumar_sustitutas(res, mapa, qdrant=Q))
regs = [r.registro for r in res]
ok(n == 1 and regs.count("2024159") == 1, f"2009817 y 2009816 apuntan a 2024159: entra UNA vez (orden {regs})")
ok(regs.index("2024159") == regs.index("2009817") + 1, "entra junto a la afectada, justo detrás")
s4159 = next(r for r in res if r.registro == "2024159")
ok(s4159.id in mapa and s4159.tesis_num == "P./J. 2/2022 (11a.)" and s4159.texto.startswith("[TIPO:"),
   "con su [Doc ID] en el mapa y armada como las demás tesis (cabecera, rubro, clave)")
ok(abs(s4159.score - (0.61 + 1e-6)) < 1e-9, "con el puntaje de la afectada, un pelo arriba")
ok("<vigencia_tesis>" in xml_s and f'doc_id_reemplazo="{s4159.id}"' in xml_s
   and xml_s.count("<perdida ") == 2, "el resumen <vigencia_tesis>: una línea por afectada, con el Doc ID de la sustituta")
ok(f'<documento id="{s4159.id}"' in xml_s, "y el XML de la sustituta, para que el modelo la lea y la cite")
ok(len(fuera_de_llamadas(Q)) == 1, "una sola consulta a Qdrant para las dos")

Q2 = QdrantFalso({V3: TESIS})
res2 = [sr("2009817"), sr("2024159")]
xml2, n2 = correr(main._sumar_sustitutas(res2, main.build_doc_id_map(res2), qdrant=Q2))
ok(n2 == 0 and not fuera_de_llamadas(Q2) and len(res2) == 2 and "<vigencia_tesis>" in xml2,
   "si la sustituta ya venía, no se consulta ni se duplica (pero el resumen sí sale)")

Q3 = QdrantFalso({V3: TESIS})
res3 = [sr("2009817", 0.9), sr("159870", 0.8), sr("2015368", 0.7)]
_, n3 = correr(main._sumar_sustitutas(res3, main.build_doc_id_map(res3), qdrant=Q3, tope=2))
ok(n3 == 2 and len(res3) == 5, f"respeta el tope: con tope=2 entran 2 de 3 ({[r.registro for r in res3]})")

res4 = [sr("2006225")]
Q4 = QdrantFalso({V3: TESIS})
ok(correr(main._sumar_sustitutas(res4, main.build_doc_id_map(res4), qdrant=Q4)) == ("", 0)
   and not Q4.llamadas, "sin tesis sin vigencia: nada, y ni una consulta")

Q5 = QdrantFalso({V3: TESIS}, demora=2.0)
res5 = [sr("2009817")]
t0 = time.perf_counter()
xml5, n5 = correr(main._sumar_sustitutas(res5, main.build_doc_id_map(res5), qdrant=Q5, plazo=0.2))
ok(n5 == 0 and time.perf_counter() - t0 < 1.0 and "<vigencia_tesis>" in xml5,
   "con Qdrant lento, a los 0.2 s se sigue sin la sustituta; la afectada conserva su sello")
res6 = [sr("2009817")]
xml6, n6 = correr(main._sumar_sustitutas(res6, main.build_doc_id_map(res6), qdrant=QdrantFalso({})))
ok(n6 == 0 and "<vigencia_tesis>" in xml6, "si la colección no responde, tampoco se cae")

# En cadena: A → B → C → D → E, con tres saltos se llega a D y no a E.
with tempfile.TemporaryDirectory() as d:
    _ri, _rc = vt.RUTA_INDICE, vt.RUTA_CURADA
    cad = {str(9000001 + k): {"estado": "abandonada", "parcial": False, "por_registro": str(9000002 + k),
                              "por_clave": f"P./J. {k + 2}/2030", "desde": "2030-01-01", "fuente": "nota_propia",
                              "nota": "Nota: La presente tesis fue abandonada."} for k in range(4)}
    Path(d, "e.json").write_text(json.dumps({"tesis": cad}))
    vt.RUTA_INDICE, vt.RUTA_CURADA = Path(d, "e.json"), Path(d, "no.json")
    vt.indice.cache_clear()
    with contextlib.redirect_stdout(io.StringIO()):
        CAD = [punto(100 + k, str(9000001 + k), f"P./J. {k + 1}/2030", f"TESIS {k}") for k in range(5)]
        Q7 = QdrantFalso({V3: CAD})
        res7 = [main._sr_de(main._tesis_a_dict(CAD[0].id, CAD[0].payload, V3))]
        xml7, n7 = correr(main._sumar_sustitutas(res7, main.build_doc_id_map(res7), qdrant=Q7))
    ok([r.registro for r in res7] == ["9000001", "9000002", "9000003", "9000004"] and n7 == 3,
       f"en cadena, hasta tres saltos: {[r.registro for r in res7]}")
    ok(vt.cadena("9000001") == ["9000002", "9000003", "9000004"], "cadena() también se detiene en tres")
    vt.RUTA_INDICE, vt.RUTA_CURADA = _ri, _rc
    vt.indice.cache_clear()

# UNA SUSTITUTA AUSENTE NO TUMBA A LAS DEMÁS (hallazgo 1, 26-sep-2026). La
# 2016052 (índice real) remite a la 2031844, que no está en la v3; la 2009817
# a la 2024159, que sí. Antes: la pasada con enteros daba 400 y se perdían las
# dos («no pude traer 2 sustituta(s) (UnexpectedResponse)»).
p2016052 = punto(10, "2016052", "XIV.C.A.3 C (10a.)", "UNA TESIS DE YUCATÁN ABANDONADA POR LA XIV.C.A.1 C (12a.).")
ok((vt.de("2016052") or {}).get("por_registro") == "2031844" and "2031844" not in POR_REG,
   "el caso: 2016052 → 2031844, que no está en el acervo")
Q9 = QdrantFalso({V3: TESIS + [p2016052]})
try:
    with contextlib.redirect_stdout(io.StringIO()):
        hall9 = correr(main._traer_tesis_por_registro(Q9, V3, ["2024159", "2031844"]))
except Exception as _e9:
    hall9 = {f"lanzó {type(_e9).__name__}": None}
ok(set(hall9) == {"2024159"}, f"_traer_tesis_por_registro con una presente y una ausente: {sorted(hall9)} (antes lanzaba)")
res9 = [sr("2009817", 0.7), main._sr_de(main._tesis_a_dict(p2016052.id, p2016052.payload, V3))]
buf9 = io.StringIO()
with contextlib.redirect_stdout(buf9):
    xml9, n9 = correr(main._sumar_sustitutas(res9, main.build_doc_id_map(res9), qdrant=Q9))
ok(n9 == 1 and [r.registro for r in res9] == ["2009817", "2024159", "2016052"] and "no pude traer" not in buf9.getvalue(),
   f"la presente entra y la ausente sólo falta: {[r.registro for r in res9]}")
ok(xml9.count("<perdida ") == 2 and 'reemplazada_por="2031844"' in xml9,
   "la ausente conserva su sello y su reemplazada_por en el resumen")
ok(not any(isinstance(v, int) for c in fuera_de_llamadas(Q9) for v in (c[2].must[0].match.any
           if isinstance(c[2].must[0].match, MatchAny) else [c[2].must[0].match.value])),
   "ninguna consulta por registro va con enteros (el índice es keyword)")


print("\n   · dónde se engancha")
FUENTE = Path("main.py").read_text(encoding="utf-8")
CE = FUENTE[FUENTE.index("async def chat_endpoint("):]
CE = CE[:CE.index("\n@app.")]
i_ret = CE.index("async def _perform_retrieval(")
i_gather = CE.index("_gather_future = asyncio.gather(")
i_lin = CE.index("_coidh_sumar_linea(_lin, search_results, doc_id_map)")
i_prec = CE.index("Precedentes AL CONTEXTO")
i_sus = CE.index("await _sumar_sustitutas(search_results, doc_id_map)")
i_prompt = CE.index("PASO 2: Construir mensajes para LLM")
i_prev = CE.index('"\\n<!-- FUENTES_PREVIAS:"')
ok(i_ret < i_gather < i_lin < i_prec < i_sus < i_prompt < i_prev,
   "en /chat: tras la recopilación (todas las ramas de _perform_retrieval), la línea y los precedentes; "
   "antes del prompt y de FUENTES_PREVIAS")
_ramas = CE[i_ret:i_gather]
ok(all(x in _ramas for x in ("if is_precedentes_mode:", "if is_drafting:", "is_sentencia", "has_document")),
   "por ese punto pasan la consulta, la redacción, el documento, la sentencia y precedentes")
AD = FUENTE[FUENTE.index("async def analyze_document("):]
AD = AD[:AD.index("\n@app.")]
i_fx = AD.index("context_xml = format_results_as_xml(search_results, estado=_entidad_acervo)")
i_sa = AD.index("await _sumar_sustitutas(search_results, doc_id_map)")
i_cx = AD.index('"\\n\\nCONTEXTO JURÍDICO RECUPERADO:\\n" + context_xml')
ok(i_fx < i_sa < i_cx, "en /analyze-document: después del XML del acervo y antes de meterlo al prompt")

# /chat-sentencia y /audit (hallazgo 4, 26-sep-2026): el XML les ordenaba
# «funda en la que los reemplaza» y la sustituta no viajaba.
def _en_orden(texto, *trozos):
    """¿Están todos los trozos en `texto`, y en ese orden?"""
    pos = [texto.find(t) for t in trozos]
    return all(p >= 0 for p in pos) and pos == sorted(pos)


CS = FUENTE[FUENTE.index("async def chat_sentencia_endpoint("):]
CS = CS[:CS.index("\n@app.")]
ok(_en_orden(CS, "search_results: List[SearchResult] = []", "if request.use_rag:",
             "rag_context = format_results_as_xml(search_results, estado=None)",
             "await _sumar_sustitutas(search_results, build_doc_id_map(search_results))",
             "rag_context += _xml_vig", "CONTEXTO JURÍDICO RECUPERADO", "registros_fuera_del_contexto(content_buffer"),
   "en /chat-sentencia: sustitutas tras el XML y antes del prompt; search_results existe aunque use_rag=False")
AU = FUENTE[FUENTE.index("async def audit_endpoint("):]
AU = AU[:AU.index("\n@app.")]
ok(_en_orden(AU, "evidence_xml = format_results_as_xml(consolidated_results)",
             "await _sumar_sustitutas(consolidated_results, build_doc_id_map(consolidated_results))",
             "evidence_xml += _xml_vig", "EVIDENCIA JURÍDICA:"),
   "en /audit: sustitutas tras el XML de la evidencia y antes del prompt")
# Y si la sustituta NO se pudo traer, su registro no es «de memoria»: lo dio el contexto.
_resp = "La P. X/2015 fue abandonada por la P./J. 2/2022 (11a.), registro digital 2024159."
ok(main.registros_fuera_del_contexto(_resp, [sr("2009817")]) == [],
   "REGISTROS_FUERA no marca el reemplazada_por de una tesis del contexto (antes: ['2024159'])")
ok(main.registros_fuera_del_contexto(_resp + " Véase el registro digital 2000004.", [sr("2009817")]) == ["2000004"],
   "y un registro que nadie dio sigue saliendo fuera")


# ═══════════════════════════════════════════════════════════════ 5 · búsqueda directa
print("\n5 · LA BÚSQUEDA DIRECTA POR REGISTRO Y POR CLAVE")
FRASE = ("¿Sigue vigente la tesis P. X/2015 (10a.), registro 2009817? "
         "¿Y la P./J. 2/2022 (11a.), registro digital 2024159?")
with contextlib.redirect_stdout(io.StringIO()):
    cit = main._extract_legal_citations(FRASE)
ok(cit["registros"] == ["2009817", "2024159"], f"la frase del diagnóstico da los dos registros: {cit['registros']}")
ok(cit["tesis_nums"] == ["P./J. 2/2022 (11a.)", "P. X/2015 (10a.)"], f"y las dos claves: {cit['tesis_nums']}")
CLAVES = {
    "1a./J. 84/2022": "1a./J. 84/2022", "2a. CIV/2014": "2a. CIV/2014",
    "PR.P.CN. J/13 P (11a.)": "PR.P.CN. J/13 P (11a.)", "la P./J.2/2022(11a.)": "P./J. 2/2022 (11a.)",
    "P. X/2015 (10a.)": "P. X/2015 (10a.)", "1a. J. 96/2011": "1a./J. 96/2011",
    "VI.2o. J/91": "VI.2o. J/91", "I.3o.C.106 K (10a.)": "I.3o.C.106 K (10a.)",
}
with contextlib.redirect_stdout(io.StringIO()):
    for txt, esperada in CLAVES.items():
        tn = main._extract_legal_citations(f"según la tesis {txt}, …")["tesis_nums"]
        ok(esperada in tn, f"«{txt}» → «{esperada}» ({tn})")
    for txt in ("En 2022 la Corte resolvió y en 2021 también.", "el expediente 1234/2022 del 11-02-2022",
                "tel. 4421234567 y $2,500,000", "desde 20220211"):
        c = main._extract_legal_citations(txt)
        ok(c["registros"] == [], f"«{txt}»: ningún año ni cifra suelta pasa por registro ({c['registros']})")
    for txt, esperado in (("registro 2030517", ["2030517"]), ("reg. digital: 2032440", ["2032440"]),
                          ("la 2024159 y la 2009817", ["2024159", "2009817"]), ("registro 160584", ["160584"]),
                          ("160584", [])):
        c = main._extract_legal_citations(txt)
        ok(c["registros"] == esperado, f"«{txt}» → {esperado} (dio {c['registros']})")
    # UN NÚMERO DE SIETE CIFRAS NO ES UN REGISTRO POR SERLO (hallazgo 3,
    # 26-sep-2026): toda la franja 2020000-2029999 existe en la v3, así que
    # cada importe, folio o expediente metía una tesis ajena al frente.
    for txt in ("Me demandan por 2025000 pesos de daños", "expediente 2024123", "folio 2025001",
                "matrícula 2023456 del IMSS", "Registro Civil acta 2025678", "el crédito 2029999",
                "inscrito en el registro 185000 del RPP", "pagué $2025000 de anticipo",
                "la tesis que cité en el expediente 2024123", "registro 2024159 del Registro Público de la Propiedad",
                "el monto asciende a 2025000 MXN", "¿Qué pasa con 2025000?"):
        c = main._extract_legal_citations(txt)
        ok(c["registros"] == [], f"«{txt}»: no es un registro ({c['registros']})")
    for txt, esperado in (("la tesis 2024159 sigue vigente", ["2024159"]),
                          ("esa jurisprudencia, 2024159, dice", ["2024159"]),
                          ("el criterio 2024159", ["2024159"]),
                          ("IUS 2024159", ["2024159"]),
                          ("la P./J. 2/2022 (11a.) 2024159", ["2024159"]),
                          ("registros digitales 2024159, 2009817 y 2009816", ["2024159", "2009817", "2009816"]),
                          ("el Ministerio Público invocó la tesis con registro 160584", ["160584"]),
                          ("la tesis en materia civil, registro 160584", ["160584"]),
                          ("registro digital 2025678 del Semanario", ["2025678"]),
                          ("[J]; 11a. Época; Pleno; Gaceta S.J.F.; Libro 11, Marzo de 2022; Tomo I; Pág. 5. 2024159",
                           ["2024159"]),
                          ("en esa época cobraba 2025000 al año", [])):
        c = main._extract_legal_citations(txt)
        ok(c["registros"] == esperado, f"«{txt}» → {esperado} (dio {c['registros']})")
ok(main._variantes_de_clave("P./J. 2/2022") [:1] == ["P./J. 2/2022"]
   and "P./J. 2/2022 (11a.)" in main._variantes_de_clave("P./J. 2/2022")
   and main._variantes_de_clave("P. X/2015 (10a.)") == ["P. X/2015 (10a.)", "P. X/2015(10a.)", "P. X/2015  (10a.)"]
   and "III.5o.A.1 A  (12a.)" in main._variantes_de_clave("III.5o.A.1 A"),
   "variantes: sin época, las cuatro épocas; con época, sólo ésa (con 0, 1 o 2 espacios antes del paréntesis)")

main.qdrant_client = QdrantFalso({V3: TESIS})
with contextlib.redirect_stdout(io.StringIO()):
    dir_res = correr(main._buscar_articulos_citados(cit))
por_reg = {r.registro: r for r in dir_res}
ok(set(por_reg) == {"2009817", "2024159"}, f"trae 2009817 y 2024159 (antes, 2024159 se tiraba por «año»): {sorted(por_reg)}")
ok(por_reg["2009817"].vigencia and por_reg["2009817"].vigencia["por_registro"] == "2024159"
   and por_reg["2009817"].tesis_num == "P. X/2015 (10a.)" and por_reg["2009817"].texto.startswith("[TIPO:"),
   "la tesis de la búsqueda directa lleva registro, clave, cabecera y su sello")
ok(all(c[2].must[0].key in ("registro", "clave_tesis") for c in main.qdrant_client.llamadas if c[0] == "scroll"),
   "por clave consulta `clave_tesis`, el campo indexado (no «tesis» ni «tesis_num», que dan 400)")
with contextlib.redirect_stdout(io.StringIO()):
    solo_clave = correr(main._buscar_articulos_citados(
        {"articles": [], "registros": [], "tesis_nums": ["P./J. 2/2022"], "casos_coidh": []}))
ok([r.registro for r in solo_clave] == ["2024159"], "sólo con la clave, sin época: encuentra 2024159")
ok(solo_clave[0].score == 1.0 and not getattr(solo_clave[0], "clave_ambigua", None), "y como es la única, entra como exacta")

# LA CLAVE SIN ÉPOCA QUE NOMBRA A VARIAS TESIS (hallazgo 2, 26-sep-2026). En el
# doble, el orden de la lista hace de orden por UUID de Qdrant: la literal va
# AL FINAL, que es donde el limit=3 de antes la dejaba fuera.
HOMONIMAS = [
    # «I.3o.C.15 K»: la 9a. es la clave literal; la 10a. y la 11a. comparten la base.
    punto(30, "2026057", "I.3o.C.15 K (11a.)", "EMPLAZAMIENTO A JUICIO A LOS CODEMANDADOS."),
    punto(31, "2001572", "I.3o.C.15 K (10a.)", "APARIENCIA DEL BUEN DERECHO. CUESTIONES JURÍDICAS."),
    punto(32, "190791", "I.3o.C.15 K", "VIOLACIONES AL PROCEDIMIENTO. OPORTUNIDAD.", epoca="Novena Época"),
    # «IV.2o.P.3 P»: tres de época delante y la literal (186061) la cuarta.
    punto(33, "2032126", "IV.2o.P.3 P (12a.)", "UNA DE LA DUODÉCIMA."),
    punto(34, "2026779", "IV.2o.P.3 P (11a.)", "UNA DE LA UNDÉCIMA."),
    punto(35, "2002803", "IV.2o.P.3 P (10a.)", "UNA DE LA DÉCIMA."),
    punto(36, "186061", "IV.2o.P.3 P", "LA LITERAL, LA QUE SE CITÓ.", epoca="Novena Época"),
    # «VII.9o.C.9 C»: sin literal y con cuatro homónimas (dos de la 10a.).
    punto(37, "2031999", "VII.9o.C.9 C (12a.)", "HOMÓNIMA DOCE."),
    punto(38, "2026999", "VII.9o.C.9 C (11a.)", "HOMÓNIMA ONCE."),
    punto(39, "2012999", "VII.9o.C.9 C (10a.)", "HOMÓNIMA DIEZ A."),
    punto(40, "2002999", "VII.9o.C.9 C(10a.)", "HOMÓNIMA DIEZ B, SIN ESPACIO ANTES DEL PARÉNTESIS."),
    # «II.4o.P.10 P (10a.)»: CON época y aun así dos tesis.
    punto(41, "2020715", "II.4o.P.10 P (10a.)", "UNA."),
    punto(42, "2012791", "II.4o.P.10 P (10a.)", "OTRA."),
]
main.qdrant_client = QdrantFalso({V3: TESIS + HOMONIMAS})


def por_clave(clave):
    with contextlib.redirect_stdout(io.StringIO()):
        return correr(main._buscar_articulos_citados(
            {"articles": [], "registros": [], "tesis_nums": [main._normalizar_clave(clave)], "casos_coidh": []}))


h1 = por_clave("I.3o.C.15 K")
ok([(r.registro, r.score, getattr(r, "clave_ambigua", None)) for r in h1] == [("190791", 1.0, None)],
   f"«I.3o.C.15 K» → sólo la literal (190791, 9a.), exacta; antes entraban 3 con score 1.0: {[r.registro for r in h1]}")
h2 = por_clave("IV.2o.P.3 P")
ok([r.registro for r in h2] == ["186061"] and h2[0].score == 1.0,
   f"«IV.2o.P.3 P» → la literal aunque venga cuarta (antes limit=3 la dejaba fuera): {[r.registro for r in h2]}")
h3 = por_clave("VII.9o.C.9 C")
ok([r.registro for r in h3] == ["2002999", "2012999", "2026999", "2031999"],
   f"sin literal: las cuatro, ordenadas por época y no por UUID: {[r.registro for r in h3]}")
ok(all(r.score == getattr(main, "CLAVE_AMBIGUA_SCORE", None) and r.score < 1.0 for r in h3), "ninguna como exacta: score < 1.0")
ok(all(getattr(r, "clave_ambigua", None) == "4 tesis con la clave «VII.9o.C.9 C»: 2002999 (10a.), 2012999 (10a.), "
                          "2026999 (11a.), 2031999 (12a.)" for r in h3),
   f"cada una dice cuántas son y de qué época: «{getattr(h3[0], 'clave_ambigua', None) if h3 else None}»")
x3 = main.format_results_as_xml(h3)
ok(x3.count('clave_ambigua="4 tesis con la clave') == 4 and 'score="1.0000"' not in x3
   and "INSTRUCCIÓN CLAVE AMBIGUA" in x3, "en el XML: atributo clave_ambigua=, sin score 1.0 y con su instrucción")
ok("INSTRUCCIÓN CLAVE AMBIGUA" not in main.format_results_as_xml(h1), "sin ambigüedad no se infla el prompt")
h4 = por_clave("II.4o.P.10 P (10a.)")
ok(sorted(r.registro for r in h4) == ["2012791", "2020715"] and all(getattr(r, "clave_ambigua", None) and r.score < 1.0 for r in h4),
   "con época y dos tesis con la misma clave: también ambigua")
ok(all(c[2].must[0].key == "clave_tesis" and c[2].must[0].match.any == main._variantes_de_clave(
           main._normalizar_clave(k))
       for c, k in zip([c for c in main.qdrant_client.llamadas if c[0] == "scroll"][-4:],
                       ("I.3o.C.15 K", "IV.2o.P.3 P", "VII.9o.C.9 C", "II.4o.P.10 P (10a.)"))),
   "una sola consulta por clave, con todas sus variantes")


# ═══════════════════════════════════════════════════════════════ 6 · el prompt
print("\n6 · LA REGLA DEL PROMPT")
P = main.SYSTEM_PROMPT_CHAT
i6 = P.find("REGLA #6")
i_v = P.find("TESIS QUE PERDIERON VIGENCIA", i6)
ok(i6 >= 0 and i_v > i6 and i_v < P.find("REGLA #7"), "la REGLA #6 trae el apartado de tesis que perdieron vigencia")
bloque = P[i_v:P.find("SEÑALES DE QUE UNA TESIS", i_v)].strip().splitlines()
ok(4 <= len(bloque) <= 6, f"sin inflar el prompt: {len(bloque)} líneas")
ok(all(x in P[i_v:i_v + 600] for x in ("NUNCA", "reemplazo_clave", "vigencia_desde", "vigencia_parcial", "[Doc ID]")),
   "dice: nunca vigente, por cuál, desde cuándo, si es parcial, y fundar en la sustituta")


# ═══════════════════════════════════════════════════════════════ 7 · verificadas
print("\n7 · FUENTES_PREVIAS NO BLINDA UNA ABANDONADA")
v_ab, v_ok = sr("2009817"), sr("2006225")
xv = main._xml_verificadas([v_ab, v_ok])
i_bl = xv.find("sin reservas")
i_pv = xv.find("PERDIERON VIGENCIA")
ok(0 <= i_bl < i_pv, "primero las blindadas, después las que perdieron vigencia, cada una con su aviso")
ok(f'id="{v_ok.id}"' in xv[i_bl:i_pv] and f'id="{v_ab.id}"' not in xv[i_bl:i_pv],
   "la abandonada NO va bajo «cítalas sin reservas»")
ok(f'id="{v_ab.id}"' in xv[i_pv:] and 'vigencia="abandonada"' in xv[i_pv:], "va aparte, con su sello")
ok("PERDIERON VIGENCIA" not in main._xml_verificadas([v_ok]), "sin abandonadas, el bloque de siempre")
main.qdrant_client = QdrantFalso({V3: TESIS})
_verif = correr(main._fuentes_ya_verificadas([POR_REG["2009817"].id]))
ok(len(_verif) == 1 and _verif[0].vigencia and _verif[0].texto.startswith("[TIPO:") and _verif[0].score == 2.0,
   "las verificadas vuelven armadas como tesis (cabecera, rubro) y con su sello")


# ═══════════════════════════════════════════════════════════════ 8 · marcadores y registro
print("\n8 · LOS MARCADORES PARA LA APP Y EL REGISTRO DE RENDER")
ok(main._campos_vigencia(v_ok) == {}, "una tesis vigente no cambia ni una clave del marcador")
mv = main._campos_vigencia(v_ab)["vigencia"]
ok(mv["estado"] == "abandonada" and mv["por_registro"] == "2024159" and mv["por_clave"] == "P./J. 2/2022 (11a.)"
   and mv["etiqueta"].startswith("ABANDONADA por la P./J. 2/2022"), "la clave «vigencia»: estado, etiqueta, por_registro, por_clave")
prev = json.loads(main._marcador_fuentes_previas([v_ab, v_ok]).split("FUENTES_PREVIAS:")[1].rsplit("-->", 1)[0])
ok(prev[v_ab.id]["vigencia"]["por_registro"] == "2024159" and "vigencia" not in prev[v_ok.id],
   "FUENTES_PREVIAS lleva la vigencia de la abandonada, y nada en la vigente")
s4 = sr("2024159")
mapa8 = main.build_doc_id_map([v_ab, s4])
resp = f"La P. X/2015, Registro digital: 2009817 [Doc ID: {v_ab.id}], dice…"
with contextlib.redirect_stdout(io.StringIO()):
    marc = main._marcadores_del_sello(resp, mapa8, [v_ab, s4])
meta = json.loads(marc[-1].split("CITATION_META:")[1].rsplit("-->", 1)[0])
ok(meta["sources"][v_ab.id]["vigencia"]["estado"] == "abandonada", "CITATION_META.sources lleva la vigencia")
ok(main._vigencia_citadas(resp, mapa8) == ["   📛 VIGENCIA: citó 2009817 (abandonada; sustituta 2024159 citada: no)"],
   f"registro: citó la abandonada sin su sustituta ({main._vigencia_citadas(resp, mapa8)})")
resp2 = resp + f" Hoy rige la P./J. 2/2022 [Doc ID: {s4.id}]."
ok(main._vigencia_citadas(resp2, mapa8) == ["   📛 VIGENCIA: citó 2009817 (abandonada; sustituta 2024159 citada: sí)"],
   "y cuando cita también la sustituta, «sí»")
ok(main._vigencia_citadas("Nada que ver, correo@dominio.mx", mapa8) == [], "si no la citó, ninguna línea")
cita = correr(main.resolver_cita(POR_REG["2009817"].id))
ok(cita.get("vigencia", {}).get("por_registro") == "2024159" and "vigencia" not in correr(
    main.resolver_cita(POR_REG["2024159"].id)), "/cita/{doc_id} lleva la vigencia de la abandonada y no de la vigente")


# ═══════════════════════════════════════════════════════════════ 9 · el taller
print("\n9 · EL TALLER Y TOULMIN")
t817 = fr._tesis_de(POR_REG["2009817"].payload)
ok(t817["vigencia"]["por_registro"] == "2024159" and fr._tesis_de(POR_REG["2006225"].payload)["vigencia"] is None,
   "fase6_rag._tesis_de lleva «vigencia» (None si vigente)")
m = fe.Material()
m.tesis = [dict(t817, obligatoria=True)]
bloque_mat = fe._bloque_material(m)
ok("[SIN VIGENCIA]" in bloque_mat and "ABANDONADA por la P./J. 2/2022" in bloque_mat,
   "el estudio la imprime SIN VIGENCIA (ya no «OBLIGATORIA») y dice por cuál")
cat, fuentes = toulmin.catalogo(SimpleNamespace(tesis=[t817], normas=[]), SimpleNamespace())
ok("ABANDONADA por la P./J. 2/2022" in cat and fuentes["T1"]["vigencia"]["por_registro"] == "2024159",
   "Toulmin la marca en el catálogo y en su ficha")
rep = toulmin._tesis_repartidas([t817] + [fr._tesis_de(POR_REG["2006225"].payload)], 5)
ok([t["registro"] for t in rep] == ["2006225", "2009817"], "y la manda al final del reparto")


# ═══════════════════════════════════════════════════════════════ 10 · sin índice
print("\n10 · SIN EL ARCHIVO DEL ÍNDICE NADA SE ROMPE")
_ri, _rc = vt.RUTA_INDICE, vt.RUTA_CURADA
vt.RUTA_INDICE, vt.RUTA_CURADA = Path("/no/existe/vigencia_tesis.json"), Path("/no/existe/vigencia_curada.json")
vt.indice.cache_clear()
buf = io.StringIO()
try:
    with contextlib.redirect_stdout(buf):
        sin1, sin2 = vt.de("2009817"), vt.de("2009816")
        limpio = sr("2009817")
        xml_sin = main.format_results_as_xml([limpio])
        sum_sin = correr(main._sumar_sustitutas([limpio], main.build_doc_id_map([limpio]), qdrant=QdrantFalso({V3: TESIS})))
        marc_sin = main._campos_vigencia(limpio)
    ok(sin1 is None and sin2 is None, "de() devuelve None")
    ok(buf.getvalue().count("no está vigencia_tesis.json") == 1, "y lo avisa UNA vez por consola, no en cada consulta")
    ok("vigencia=" not in xml_sin and "PERDIÓ VIGENCIA" not in xml_sin and "<documento" in xml_sin,
       "el XML sale como antes del sello")
    ok(sum_sin == ("", 0) and marc_sin == {}, "sin sustitutas ni marcador")
    ok(fr._tesis_de(POR_REG["2009817"].payload)["vigencia"] is None, "el taller tampoco se cae")
finally:
    vt.RUTA_INDICE, vt.RUTA_CURADA = _ri, _rc
    vt.indice.cache_clear()
with tempfile.TemporaryDirectory() as d:
    Path(d, "roto.json").write_text("{esto no es json")
    vt.RUTA_INDICE = Path(d, "roto.json")
    vt.indice.cache_clear()
    with contextlib.redirect_stdout(io.StringIO()):
        ok(vt.de("2009817") is None and vt.de("160584") is not None,
           "con el índice roto, la curada sigue y la expresa calla")
    vt.RUTA_INDICE = _ri
    vt.indice.cache_clear()


# ═══════════════════════════════════════════════════════════════ 11 · el generador
print("\n11 · EL GENERADOR")
_spec = importlib.util.spec_from_file_location("vigencia_tesis_generar",
                                               Path(os.getcwd(), "scripts", "vigencia_tesis_generar.py"))
gen = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gen)
largo = "palabra " * 80
c = gen.cortar(largo, 400)
ok(len(c) <= 400 and c.endswith("…") and not c.endswith(" …") and gen.cortar("corta", 400) == "corta",
   "cortar(): ≤ tope, en palabra entera y con «…»")
comp = gen.compacto({"2009817": dict(vt.de("2009817"), fuentes=["nota_propia"], patron="tesis_fue")})
ok(set(comp["2009817"]) == CAMPOS, "compacto(): sólo los campos del contrato")

# EL ALCANCE Y LA CADENA, EN EL GENERADOR (hallazgo 8, 26-sep-2026). Las notas
# son las del Semanario, literales; 900001/900002/900009 son de mentira.
NOTA_164500 = (
    'Este criterio fue interrumpido y modificado, en la parte relativa, por la tesis 3a./J. 23/91, de rubro: '
    '"AMPARO INDIRECTO, RESULTA IMPROCEDENTE CONTRA LA RESOLUCIÓN QUE DESECHA LA EXCEPCIÓN DE FALTA DE '
    'COMPETENCIA (INTERRUPCIÓN Y MODIFICACIÓN EN LA PARTE RELATIVA, DE LA TESIS JURISPRUDENCIAL NÚMERO 166, '
    'VISIBLE EN LAS PÁGINAS 297 Y 298, SEGUNDA PARTE, DE LA COMPILACIÓN DE 1917 A 1988).", publicada en el '
    'Semanario Judicial de la Federación, Octava Época, Tomo VII, mayo de 1991, página 47; la que, a su vez, '
    'fue interrumpida y modificada, en la parte relativa, por la tesis P./J. 55/2003, de rubro: "AMPARO '
    'INDIRECTO, RESULTA PROCEDENTE CONTRA LA RESOLUCIÓN QUE DESECHA LA EXCEPCIÓN DE FALTA DE COMPETENCIA.", '
    'publicada en el Semanario Judicial de la Federación y su Gaceta, Novena Época, Tomo XVIII, septiembre de '
    '2003, página 5.')
NOTA_190237 = (
    'Nota: El criterio contenido en la presente tesis fue abandonado por el Pleno de la Suprema Corte de '
    'Justicia de la Nación, por lo que se refiere a los efectos de que se declare la violación a la veda '
    'electoral señalada en el párrafo penúltimo de la fracción II del artículo 105 de la Constitución Política '
    'de los Estados Unidos Mexicanos, al resolver la acción de inconstitucionalidad 145/2017 y su acumulada 146/2017.')
T_GEN = [
    {"registro": "164500", "clave_tesis": "3a. (7a. Época)", "rubro": "AMPARO INDIRECTO. NO PROCEDE…",
     "precedentes": "Amparo en revisión 1/1986.\n\n" + NOTA_164500},
    {"registro": "183349", "clave_tesis": "P./J. 55/2003", "fecha_publicacion": "2003-09-01", "precedentes": "",
     "rubro": "AMPARO INDIRECTO, RESULTA PROCEDENTE CONTRA LA RESOLUCIÓN QUE DESECHA LA EXCEPCIÓN DE FALTA DE COMPETENCIA."},
    {"registro": "190237", "clave_tesis": "P./J. 25/2001", "rubro": "VEDA ELECTORAL…", "precedentes": NOTA_190237},
    {"registro": "900001", "clave_tesis": "P./J. 1/2030", "rubro": "UNA TOTAL.",
     "precedentes": "Nota: La presente tesis fue abandonada por la tesis P./J. 9/2030."},
    {"registro": "900002", "clave_tesis": "P./J. 2/2030", "rubro": "UNA EN LO CONDUCENTE.",
     "precedentes": "Nota: Este criterio fue interrumpido, en lo conducente, por la tesis P./J. 9/2030."},
    {"registro": "900009", "clave_tesis": "P./J. 9/2030", "rubro": "LA NUEVA.", "precedentes": "",
     "fecha_publicacion": "2030-01-10"},
]
AG = gen.Acervo(T_GEN)
EX = gen.Extractor(AG)
for _t in T_GEN:
    EX.procesar(_t)
CG = gen.compacto(gen.consolidar(AG, EX.rel)[0])
c164 = CG["164500"]
ok(c164["por_registro"] == "183349" and c164["por_clave"] == "P./J. 55/2003"
   and c164.get("por_intermedias") == ["3a./J. 23/91"] and c164["desde"] == "1991-05",
   f"«la que, a su vez, fue interrumpida … por la P./J. 55/2003»: el reemplazo es el último eslabón ({c164['por_clave']})")
ok(c164["parcial"] and c164.get("alcance") == "en la parte relativa",
   "«, en la parte relativa,» hace parcial la pérdida (antes sólo «parcialmente»)")
ok(CG["190237"]["parcial"] and (CG["190237"].get("alcance") or "").startswith("por lo que se refiere a los efectos")
   and (CG["190237"].get("alcance") or "").endswith("Estados Unidos Mexicanos")
   and CG["190237"]["por_resolucion"] == "acción de inconstitucionalidad 145/2017",
   "«por lo que se refiere a …»: parcial, con el inciso hasta «, al resolver»")
ok(CG["900002"]["parcial"] and CG["900002"].get("alcance") == "en lo conducente", "«en lo conducente»: parcial")
ok(not CG["900001"]["parcial"] and set(CG["900001"]) == CAMPOS and CG["900001"]["por_registro"] == "900009",
   "una pérdida total sigue igual: sin alcance ni por_intermedias")
ok(hasattr(gen, "alcance_en") and gen.alcance_en(' por la tesis X, de rubro: "… (INTERRUPCIÓN Y MODIFICACIÓN EN LA PARTE RELATIVA …)"') is None,
   "el «EN LA PARTE RELATIVA» del rubro citado es de la OTRA tesis: no cuenta")

# LA CACHÉ DEL SJF, OBLIGATORIA (hallazgo 20, 26-sep-2026). Antes, sin ella, el
# guion avisaba por stderr, salía con 0 y escribía un índice peor (555 en vez
# de 558) sobre datos/vigencia_tesis.json.
GEN = Path(os.getcwd(), "scripts", "vigencia_tesis_generar.py")
ok("/private/tmp" not in GEN.read_text(encoding="utf-8") and "claude-worktrees" not in GEN.read_text(encoding="utf-8"),
   "ninguna carpeta de sesión como caché por omisión")
with tempfile.TemporaryDirectory() as d:
    tc = Path(d, "tesis.jsonl")
    tc.write_text("\n".join(json.dumps(t, ensure_ascii=False) for t in [
        {"registro": "900001", "clave_tesis": "P./J. 1/2030", "rubro": "UNA CORTADA.", "precedentes": "x" * 2600},
        {"registro": "900009", "clave_tesis": "P./J. 9/2030", "rubro": "LA NUEVA.", "precedentes": "",
         "fecha_publicacion": "2030-01-10"}]), encoding="utf-8")
    cache = Path(d, "sjf")
    cache.mkdir()
    (cache / "900001.json").write_text(json.dumps({"precedentes": "<p>" + "x" * 2600 + "</p><p>Nota: La presente "
                                                   "tesis fue abandonada por la tesis P./J. 9/2030.</p>"}))
    Path(d, "vacia").mkdir()
    _n = [0]

    def gen_cli(*args):
        _n[0] += 1
        sal = Path(d, f"salida_{_n[0]}.json")
        r = subprocess.run([sys.executable, str(GEN), "--tesis-cache", str(tc), "--salida", str(sal), *args],
                           capture_output=True, text=True, timeout=120)
        return r, sal

    r, sal = gen_cli()
    ok(r.returncode != 0 and not sal.exists(), f"sin --sjf-cache ni --sin-sjf: error (rc={r.returncode}) y no escribe")
    r, sal = gen_cli("--sjf-cache", str(Path(d, "no_existe")))
    ok(r.returncode != 0 and not sal.exists() and "No se escribió nada" in r.stderr,
       f"con una caché que no existe: error (rc={r.returncode}) y no escribe")
    r, sal = gen_cli("--sjf-cache", str(Path(d, "vacia")))
    ok(r.returncode != 0 and not sal.exists(), "con una caché vacía, tampoco")
    r, sal = gen_cli("--sjf-cache", str(cache))
    ok(r.returncode == 0 and "900001" in json.loads(sal.read_text())["tesis"],
       "con la caché: la nota que el corte de 2,500 se comía aparece")
    r, sal = gen_cli("--sin-sjf")
    _s = json.loads(sal.read_text()) if sal.exists() else {}
    ok(r.returncode == 0 and "900001" not in _s.get("tesis", {}) and "SIN la caché del SJF" in _s.get("fuente", "")
       and "--sin-sjf" in r.stderr, "con --sin-sjf explícito corre, pero lo dice en stderr y en «fuente»")

# EL DESCARGADOR DE LA CACHÉ, EN EL REPO (sólo GET, ≤ 1 petición/s, sin red aquí).
_RUTA_DL = Path(os.getcwd(), "scripts", "sjf_cache_descargar.py")
ok(_RUTA_DL.exists(), "scripts/sjf_cache_descargar.py está en el repo (antes vivía en un scratchpad)")
if _RUTA_DL.exists():
    _spec2 = importlib.util.spec_from_file_location("sjf_cache_descargar", _RUTA_DL)
    dl = importlib.util.module_from_spec(_spec2)
    _spec2.loader.exec_module(dl)
else:
    dl = None


class _Resp:
    def __init__(self, cuerpo):
        self.cuerpo = cuerpo

    def read(self):
        return self.cuerpo

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


pedidas, dormido, reloj = [], [], [100.0]


def _abrir(req, timeout=30):
    pedidas.append(req)
    if "/tesis/999999?" in req.full_url:
        raise OSError("HTTP Error 403: Forbidden")
    return _Resp(json.dumps({"precedentes": "…"}).encode())


def _dormir(s):
    dormido.append(s)
    reloj[0] += s


if dl is not None:
    with tempfile.TemporaryDirectory() as d:
        D = dl.Descargador(Path(d), abrir=_abrir, dormir=_dormir, reloj=lambda: reloj[0])
        a1, a2, _b, e1, _e2 = (D.tesis(x) for x in ("164500", "164500", "183349", "999999", "999999"))
        ok(len(pedidas) == 4 and a1 == a2, "lo ya bajado no se vuelve a pedir; el error sí se reintenta")
        ok(all(p.get_method() == "GET" and "sjf2.scjn.gob.mx" in p.full_url for p in pedidas), "sólo GET, sólo al SJF")
        ok(len(dormido) == 3 and all(s >= 1.0 for s in dormido), f"al menos 1 s entre peticiones ({dormido})")
        ok(Path(d, "164500.json").exists() and not Path(d, "999999.json").exists() and "_error" in e1,
           "un error no se guarda en la caché")
        ok(dl.Descargador(Path(d), pausa=0.2).pausa >= 1.0, "ni pidiéndolo baja de una petición por segundo")
    ok(dl.registros_cortados([{"registro": "1", "precedentes": "x" * 2500},
                              {"registro": "2", "precedentes": "x" * 2499},
                              {"registro": "1", "precedentes": "x" * 2600}], 2500) == ["1"],
       "pide sólo las tesis con `precedentes` cortado (≥ 2,500), sin repetir")


print()
if FALLOS:
    print(f"FALLAN {len(FALLOS)}:")
    for f in FALLOS:
        print("  ·", f)
    sys.exit(1)
print("TODO PASA")
