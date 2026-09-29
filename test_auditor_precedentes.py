# -*- coding: utf-8 -*-
"""El auditor de sentencias con los precedentes del propio tribunal (29-sep-2026).

Sin red: el modelo y Qdrant son dobles que contestan lo que cada prueba pide.

    .venv/bin/python test_auditor_precedentes.py
"""
import asyncio, json, os, sys
from pathlib import Path
sys.path.insert(0, ".")
import auditor_precedentes as ap

FALLOS = []


def ok(c, q):
    print(f"   {'PASA ' if c else 'FALLA'}  {q}")
    if not c:
        FALLOS.append(q)


TERCERO = "Tercer Tribunal Colegiado en Materias Administrativa y Civil del Vigésimo Segundo Circuito"
PRIMERO_22 = "Primer Tribunal Colegiado en Materias Administrativa y Civil del Vigésimo Segundo Circuito"
AJENO = "Primer Tribunal Colegiado en Materia Civil del Tercer Circuito"


def mensaje(texto, favorables=False):
    return ("[AUDITAR_SENTENCIA]\n" + (ap.MARCA_FAVORABLES + "\n" if favorables else "")
            + "Archivo: proyecto.pdf\n\n<!-- SENTENCIA_INICIO -->\n" + texto + "\n<!-- SENTENCIA_FIN -->")


PROYECTO = (f"QUEJA 115/2026\nQUEJOSA: ********\n{TERCERO.upper()}.\n\nVISTOS para resolver el recurso de queja "
            "115/2026 ... el incidente previsto en el artículo 211 del Código de Procedimientos Civiles ...")


# ── Dobles ────────────────────────────────────────────────────────────────

class _Msg:
    def __init__(self, c):
        self.content = c


class _Resp:
    def __init__(self, c):
        self.choices = [type("C", (), {"message": _Msg(c)})()]
        self.usage = None


class Modelo:
    """`responder(prompt, kw)` devuelve el texto o lanza."""
    def __init__(self, responder):
        self.llamadas = []
        dueño = self

        class _Comp:
            async def create(self_, **kw):
                dueño.llamadas.append(kw)
                return _Resp(responder(kw["messages"][-1]["content"], kw))
        self.chat = type("Ch", (), {"completions": _Comp()})()


class _Punto:
    def __init__(self, score, payload):
        self.score, self.payload = score, payload


class Qdrant:
    def __init__(self, planteamientos, asuntos=(), falla_retrieve=False):
        self.planteamientos, self.asuntos, self.falla = list(planteamientos), list(asuntos), falla_retrieve
        self.consultas, self.traidos = [], []

    async def query_points(self, **kw):
        self.consultas.append(kw)
        return type("R", (), {"points": [_Punto(s, p) for s, p in self.planteamientos]})()

    async def retrieve(self, collection_name, ids, with_payload=True):
        self.traidos.append(list(ids))
        if self.falla:
            raise RuntimeError("se cayó")
        return [_Punto(0, a) for a in self.asuntos if ap.id_asunto(a["neun"]) in ids]


async def embed(_t):
    return [0.1] * 8


def plant(neun, alias, fecha, calif="infundado", razon="El 211 es el medio idóneo.", tipo="Queja"):
    return {"clase": "planteamiento", "neun": neun, "alias": alias, "tipo": tipo, "fecha": fecha,
            "organo": "x", "pregunta": "¿Debe agotarse el incidente del 211?", "resolvio": "desechó",
            "combate": "que no hay que agotarlo", "calificacion": calif, "razon": razon}


LECTURA = {"tribunal": TERCERO, "expediente": "recurso de queja 115/2026", "autoridad": "Juez",
           "acto": "desechamiento", "sentido": "infundada",
           "planteamientos": [{"pregunta": "¿Debe agotarse el incidente del artículo 211 antes del amparo?",
                               "resolvio": "desechó la demanda", "combate": "que el 211 no suspende",
                               "calificacion": "infundado", "razon": "El 211 es idóneo aunque no suspenda."}]}


def responder_normal(prompt, kw):
    if "Extrae, SÓLO de lo que está escrito" in prompt:
        return json.dumps(LECTURA, ensure_ascii=False)
    if "Juzga cada precedente" in prompt:
        return json.dumps({"precedentes": [
            {"id": "P1", "mismo_punto": "sí", "relacion": "coincide", "por_que": "Sostuvo lo mismo."},
            {"id": "P2", "mismo_punto": "si", "relacion": "contradice", "por_que": "Sostuvo lo contrario."},
            {"id": "P3", "mismo_punto": "parcial", "relacion": "distingue", "por_que": "Otro supuesto."}]})
    raise AssertionError("prompt inesperado")


print("\n1 · LO QUE VIENE EN EL MENSAJE")
ok(ap.enfoque_de(mensaje("x", True)) == "favorables" and ap.enfoque_de(mensaje("x")) == "completo"
   and ap.enfoque_de(None) == "completo", "el enfoque: completo de entrada, favorables sólo con la marca")
ok(ap.texto_del_mensaje(mensaje("  el proyecto  ")) == "el proyecto", "el proyecto sale de entre los marcadores")
ok(ap.texto_del_mensaje("sin marcadores") == "", "sin marcador de inicio no hay proyecto")
ok(ap.texto_del_mensaje("<!-- SENTENCIA_INICIO -->truncado") == "truncado", "sin marcador de fin, hasta el final")
ok(ap.tipo_del_proyecto("RECURSO DE QUEJA 1/2026, derivado del juicio de amparo indirecto") == "queja",
   "la queja antes que el amparo del que viene")
ok(ap.tipo_del_proyecto("AMPARO EN REVISIÓN 631/2025") == "amparo_revision", "amparo en revisión, con acento")
ok(ap.tipo_del_proyecto("REVISIÓN FISCAL 2/2026 ... recurso de revisión") == "revision_fiscal",
   "revisión fiscal antes que revisión")
ok(ap.tipo_del_proyecto("AMPARO DIRECTO 704/2022") == "amparo_directo", "amparo directo")
ok(ap.tipo_del_proyecto("oficio sin tipo") == "", "lo que no se reconoce queda vacío")
ok(ap.tribunal_del_proyecto("encabezado", {"tribunal": TERCERO}) == TERCERO, "el tribunal que leyó el lector")
ok(ap.tribunal_del_proyecto(PROYECTO, None).lower() == TERCERO.lower(), "sin lectura, el del encabezado")
ok(ap.tribunal_del_proyecto(PROYECTO, {"tribunal": "la autoridad"}).lower() == TERCERO.lower(),
   "una lectura que no es un colegiado no gana al encabezado")
import fase_oaj as _fo
ok(_fo.organo_de(ap.tribunal_del_proyecto(PROYECTO, None))[0] == "3TCC",
   "el encabezado en MAYÚSCULAS también se resuelve al Tercer Tribunal")
ok(ap.expediente_del_proyecto("", {"expediente": "recurso de queja 115/2026"}) == (115, "2026"),
   "el propio expediente, de la lectura")

print("\n2 · LA LECTURA")
n = ap.normalizar_lectura({"tribunal": "null", "planteamientos": [{"pregunta": ""}, "basura"]
                           + [{"pregunta": f"¿{i}?", "resolvió": "algo"} for i in range(9)]})
ok(len(n["planteamientos"]) == ap.MAX_PUNTOS and n["tribunal"] == "", "a lo más seis puntos, sin «null»")
ok(n["planteamientos"][0]["resolvio"] == "algo", "«resolvió» con acento también se lee")
ok(ap._json_de('texto {"a": 1} más') == {"a": 1} and ap._json_de("nada") == {} and ap._json_de("[1]") == {},
   "el JSON, aunque venga con texto alrededor; lo que no es objeto, vacío")
ok(ap.texto_consulta({"pregunta": "p", "combate": "c", "resolvio": "r"}) == "p c r",
   "la consulta en el orden del índice: pregunta, combate, resolvió")

print("\n3 · LOS CANDIDATOS")
q = Qdrant([(0.91, plant(10, "12/2020", "05-03-2020")), (0.80, plant(10, "12/2020", "05-03-2020")),
            (0.85, plant(20, "115/2026", "01-01-2026")),                     # el propio asunto
            (0.84, plant(21, "115/2026", "01-01-2026", tipo="Amparo en revisión")),  # mismo número, otro tipo
            (0.83, plant(30, "7/2019", "09-02-2019")), (0.75, {"clase": "planteamiento", "neun": "x"})],
           asuntos=[{"neun": 10, "tema": "El 211 es idóneo", "sintesis": "s", "sentido": "Infundado"}])
cs = asyncio.run(ap.candidatos(q, embed, LECTURA["planteamientos"][0], "ORGANO", (115, "2026"), "queja"))
ok([c["neun"] for c in cs] == [10, 21, 30], f"uno por NEUN, el mejor, sin el propio asunto (salió {[c['neun'] for c in cs]})")
ok(cs[0]["coseno"] == 0.91 and cs[0]["tema"] == "El 211 es idóneo" and cs[0]["sentido"] == "Infundado",
   "el mejor planteamiento, con el tema y el sentido del asunto")
kw = q.consultas[0]
ok(kw["using"] == "dense" and kw["limit"] == ap.PEDIDOS_POR_PUNTO and kw["score_threshold"] == ap.COSENO_MINIMO
   and {(c.key, c.match.value) for c in kw["query_filter"].must} == {("clase", "planteamiento"), ("organo", "ORGANO")},
   "la consulta: vector denso, sólo planteamientos del órgano, sin filtro de tipo")
cs2 = asyncio.run(ap.candidatos(Qdrant(q.planteamientos, falla_retrieve=True), embed,
                                LECTURA["planteamientos"][0], "ORGANO", None, "", tope=1))
ok(len(cs2) == 1 and cs2[0]["neun"] == 10 and cs2[0]["tema"] == "", "si no llegan los asuntos, se sigue; el tope manda")
ok(asyncio.run(ap.candidatos(q, embed, {"pregunta": ""}, "ORGANO")) == [] and
   asyncio.run(ap.candidatos(q, embed, LECTURA["planteamientos"][0], "")) == [], "sin consulta u órgano, nada")

print("\n4 · LA CLASIFICACIÓN")
cands = [dict(c, fecha="01-01-2020") for c in cs]
m = Modelo(responder_normal)
cl = asyncio.run(ap.clasificar(m, LECTURA["planteamientos"][0], cands))
ok([c["relacion"] for c in cl] == ["coincide", "contradice", "distingue"], "coincide, contradice y distingue")
ok(cl[0]["mismo_punto"] == "si" and cl[0]["por_que"] == "Sostuvo lo mismo.", "«sí» con acento cuenta como sí")
ok(m.llamadas[0]["model"] == ap.modelo_lector() and m.llamadas[0]["reasoning_effort"] == "low",
   "clasifica el lector, con razonamiento bajo")
m2 = Modelo(lambda p, kw: json.dumps({"precedentes": [
    {"id": "p1", "mismo_punto": "no", "relacion": "coincide"}, {"id": "P2", "mismo_punto": "si", "relacion": "otra"},
    {"id": "P3", "mismo_punto": "si", "relacion": "coincide"}]}))
ok([c["neun"] for c in asyncio.run(ap.clasificar(m2, LECTURA["planteamientos"][0], cands))] == [30],
   "fuera lo que no es el mismo punto y lo que trae una relación desconocida")
ok(asyncio.run(ap.clasificar(Modelo(lambda p, kw: "no es JSON"), LECTURA["planteamientos"][0], cands)) == [],
   "si el clasificador no contesta JSON, se calla")

print("\n5 · EL BLOQUE PARA EL REDACTOR")
ok(ap.fecha_legible("9-02-2023") == "9 de febrero de 2023" and ap.fecha_legible("") == "fecha no consta"
   and ap.fecha_legible("31-13-2020") == "31-13-2020", "las fechas legibles; lo raro, tal cual")
ok(sorted(["05-03-2020", "09-02-2019", "01-01-2021"], key=ap._clave_fecha) == ["09-02-2019", "05-03-2020", "01-01-2021"],
   "el orden cronológico por la fecha dd-mm-aaaa")
g = list(range(20))
vistos, mas = ap._muestra(g)
k = max(1, ap.MOSTRADOS_POR_GRUPO // 3)
ok(len(vistos) == ap.MOSTRADOS_POR_GRUPO and vistos[:k] == g[:k] and vistos[-1] == 19
   and mas == 20 - ap.MOSTRADOS_POR_GRUPO, "a lo más ocho por grupo: los más antiguos y los más recientes")
ok(ap._muestra([1, 2]) == ([1, 2], 0), "un grupo corto se enseña entero")
for cob, frase in (("sin_tribunal", "no consta en el proyecto qué tribunal"),
                   ("sin_lectura", "todavía no cuenta con las sentencias leídas"), ("error", "no se pudieron consultar")):
    b = ap.bloque([], PRIMERO_22, "completo", cob)
    ok(frase in b and "no afirmes" in b, f"cobertura «{cob}»: lo dice y prohíbe afirmar la línea")


def prec(neun, rel, fecha, tema="t"):
    return {"neun": neun, "tipo": "Queja", "expediente": f"{neun}/2020", "fecha": fecha, "sentido": "Infundado",
            "tema": tema, "pregunta": "¿?", "calificacion": "infundado", "razon": "r", "relacion": rel,
            "por_que": f"por qué {neun}"}


puntos = [{"punto": LECTURA["planteamientos"][0],
           "precedentes": [prec(3, "contradice", "01-01-2022"), prec(1, "coincide", "01-01-2019", "x" * 400),
                           prec(2, "distingue", "01-01-2020")]},
          {"punto": {"pregunta": "¿Otro punto?", "calificacion": "fundado"}, "precedentes": []}]
b = ap.bloque(puntos, TERCERO, "completo", "leida")
ok(all(t in b for t in ap._TITULOS.values()) and "3/2020" in b and "Sin precedentes del tribunal" in b,
   "completo: los tres grupos, y el punto sin precedentes lo dice")
ok("no cita textual" in b and "puede referirse a otros puntos" in b and "[…]" in b,
   "la razón se marca como síntesis; el tema se recorta y se advierte")
ok("ENFOQUE" not in b, "sin enfoque pedido, no hay instrucción de callar")
bf = ap.bloque(puntos, TERCERO, "favorables", "leida")
ok("ENFOQUE PEDIDO POR EL MAGISTRADO" in bf and "1/2020" in bf and "3/2020" not in bf and "2/2020" not in bf
   and ap._TITULOS["contradice"] not in bf, "favorables: sólo los que sostienen el proyecto llegan al prompt")
muchos = [{"punto": LECTURA["planteamientos"][0],
           "precedentes": [prec(100 + i, "coincide", f"01-01-{2000 + i}") for i in range(12)]}]
bm = ap.bloque(muchos, TERCERO, "completo", "leida")
ok(f"Además, {12 - ap.MOSTRADOS_POR_GRUPO} asunto(s) más" in bm and "catálogo" not in bm.split("\n", 2)[-1].lower(),
   "los que no se enseñan se cuentan, sin la palabra «catálogo» en el cuerpo")

print("\n6 · TODO JUNTO: NUNCA LANZA")
pasos = []
q3 = Qdrant([(0.9, plant(10, "12/2020", "05-03-2020")), (0.88, plant(30, "7/2019", "09-02-2019")),
             (0.87, plant(40, "8/2021", "09-02-2021"))])
out = asyncio.run(ap.preparar(Modelo(responder_normal), q3, embed, mensaje(PROYECTO), paso=pasos.append))
ok(out["cobertura"] == "leida" and out["puntos"] == 1 and (out["coinciden"], out["contradicen"], out["distinguen"]) == (1, 1, 1),
   f"tercer tribunal: se consulta y se cuenta (salió {out['cobertura']}, {out['coinciden']}/{out['contradicen']}/{out['distinguen']})")
ok(pasos == ["proyecto", "tribunal|3TCC", "contrastar|1|1"], f"los pasos que pinta la pantalla (salió {pasos})")
ok(out["organo"] == _fo.ORGANOS_OAJ["3TCC"] and out["bloque"].startswith("PRECEDENTES DEL PROPIO TRIBUNAL ("),
   "el órgano de la OAJ y el bloque con la línea")
pasos_f = []
out_f = asyncio.run(ap.preparar(Modelo(responder_normal), q3, embed, mensaje(PROYECTO, True), paso=pasos_f.append))
ok(out_f["enfoque"] == "favorables" and pasos_f[-1] == "contrastar|1|0",
   "favorables: la pantalla no anuncia los contrarios")

for nombre, trib in (("otro tribunal del circuito", PRIMERO_22), ("otro circuito", AJENO)):
    lect = dict(LECTURA, tribunal=trib)
    qv, mv = Qdrant([]), Modelo(lambda p, kw, l=lect: json.dumps(l))
    o = asyncio.run(ap.preparar(mv, qv, embed, mensaje(PROYECTO.replace(TERCERO.upper(), trib.upper()))))
    ok(o["cobertura"] == "sin_lectura" and "todavía no cuenta" in o["bloque"] and qv.consultas == []
       and mv.llamadas == [] and o["tribunal"].lower() == trib.lower(),
       f"{nombre}: lo dice sin leer el proyecto ni buscar")

# Con el nombre partido en dos renglones (así sale de muchos PDF) y otro
# colegiado mencionado antes, el encabezado no basta: se lee el proyecto.
partido = (f"QUEJA 115/2026\nEl {AJENO} declinó la competencia.\nTERCER TRIBUNAL COLEGIADO EN MATERIAS\n"
           "ADMINISTRATIVA Y CIVIL DEL VIGÉSIMO SEGUNDO CIRCUITO.\nVISTOS ...")
ok(ap.tribunal_sin_lectura(partido) == "", "el nombre del Tercero en dos renglones: hay que leer el proyecto")
ok(ap.tribunal_sin_lectura("un proyecto sin encabezado") == "", "sin tribunal en el encabezado: hay que leerlo")
ok(ap.tribunal_sin_lectura(f"{AJENO}\n...") == AJENO, "sólo un tribunal ajeno: no hace falta leer")
mp = Modelo(responder_normal)
o = asyncio.run(ap.preparar(mp, q3, embed, mensaje(partido)))
ok(o["cobertura"] == "leida" and len(mp.llamadas) >= 1, "y en ese caso la lectura decide que es el Tercero")

o = asyncio.run(ap.preparar(Modelo(lambda p, kw: json.dumps(dict(LECTURA, tribunal=""))), Qdrant([]), embed,
                            mensaje("un proyecto sin encabezado")))
ok(o["cobertura"] == "sin_tribunal", "sin tribunal en el proyecto: lo dice")


def revienta(p, kw):
    raise RuntimeError("el proveedor se cayó")


o = asyncio.run(ap.preparar(Modelo(revienta), Qdrant([]), embed, mensaje(PROYECTO)))
ok(o["cobertura"] == "error" and "no se pudieron consultar" in o["bloque"], "si el lector revienta: error, sin lanzar")
o = asyncio.run(ap.preparar(Modelo(responder_normal), Qdrant([]), embed, "sin proyecto"))
ok(o["cobertura"] == "error" and o["bloque"], "sin proyecto en el mensaje: error, sin lanzar")


def paso_roto(_):
    raise ValueError("pantalla")


o = asyncio.run(ap.preparar(Modelo(responder_normal), q3, embed, mensaje(PROYECTO), paso=paso_roto))
ok(o["cobertura"] == "leida", "un paso que falla no tumba la preparación")


class QdrantRoto(Qdrant):
    async def query_points(self, **kw):
        raise ConnectionError("qdrant")


o = asyncio.run(ap.preparar(Modelo(responder_normal), QdrantRoto([]), embed, mensaje(PROYECTO)))
ok(o["cobertura"] == "leida" and "Sin precedentes del tribunal" in o["bloque"],
   "si Qdrant falla en un punto, ese punto va sin precedentes y la nota sigue")


async def _lento():
    antes = ap.TOPE_PREPARAR_S
    ap.TOPE_PREPARAR_S = 0.05

    async def tarda(*a, **k):
        await asyncio.sleep(1)
    original = ap.preparar
    ap.preparar = tarda
    try:
        return await ap.preparar_con_tope(None, None, embed, mensaje(PROYECTO, True))
    finally:
        ap.preparar, ap.TOPE_PREPARAR_S = original, antes


o = asyncio.run(_lento())
ok(o["cobertura"] == "tiempo" and o["enfoque"] == "favorables" and "no se pudieron consultar" in o["bloque"],
   "si tarda más del tope: se audita sin catálogo y conserva el enfoque")

print("\n7 · EL MOTOR Y EL INTERRUPTOR")
for k in ("AUDITOR_MODELO", "AUDITOR_ESFUERZO", "AUDITOR_PRECEDENTES"):
    os.environ.pop(k, None)
ok(ap.modelo() == "gpt-5.6-terra" and ap.esfuerzo() == "low" and ap.activo(),
   "de entrada: Terra, esfuerzo bajo, encendido (lo pidió David)")
os.environ["AUDITOR_PRECEDENTES"] = "0"
ok(not ap.activo(), "AUDITOR_PRECEDENTES=0 vuelve al auditor anterior sin desplegar")
os.environ.pop("AUDITOR_PRECEDENTES")
ok("Nada de porcentajes" in ap.SYSTEM_PROMPT_AUDITOR and "Se aparta de la línea" in ap.SYSTEM_PROMPT_AUDITOR
   and "# NOTA DE AUDITORÍA DEL PROYECTO" in ap.SYSTEM_PROMPT_AUDITOR, "el prompt: encabezados, dictamen y la regla 8")

print("\n8 · DÓNDE SE ENGANCHA EN /chat")
F = Path("main.py").read_text(encoding="utf-8")
G = F[F.index("async def generate_stream("):]
i_prec = G.index("Precedentes error:")
i_aud = G.index("_auditor = None")
i_prep = G.index("_ap.preparar_con_tope(")
i_sp = G.index("system_prompt = _ap_sp.SYSTEM_PROMPT_AUDITOR")
i_inj = G.index('dynamic_injections.append(_auditor["bloque"])')
i_ctx = G.index('dynamic_injections.append(f"CONTEXTO JURÍDICO RECUPERADO:')
i_mod = G.index("active_model = _ap_m.modelo()")
i_esf = G.index('api_kwargs["reasoning_effort"] = _esfuerzo_redaccion')
ok(i_prec < i_aud < i_prep < i_sp < i_inj < i_ctx < i_mod < i_esf,
   "tras los precedentes; antes del prompt, del contexto jurídico, del modelo y del esfuerzo")
seg = G[i_aud:i_sp]
ok("if is_sentencia:" in seg and "if _ap.activo():" in seg and 'yield "<!--PING-->"' in seg
   and 'yield f"<!--PASO:{_pasos_aud[_vistos_aud]}-->"' in seg, "sólo en la sentencia, con interruptor, latido y pasos")
ok("_esfuerzo_redaccion = _ap_m.esfuerzo()" in G and 'active_model = "gpt-5.2"' in G
   and "system_prompt = SYSTEM_PROMPT_SENTENCIA_ANALYSIS" in G, "y el camino anterior sigue ahí si el auditor no corre")
ok(G.index("_auditor = None") < G.index("if _auditor is not None:"), "`_auditor` existe en todas las rutas antes de leerse")

print()
if FALLOS:
    print("FALLAS:\n  " + "\n  ".join(FALLOS))
    sys.exit(1)
print("RESULTADO: TODAS LAS COMPROBACIONES PASAN")
