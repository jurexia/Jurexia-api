# -*- coding: utf-8 -*-
"""LA PREGUNTA DECISIVA, PARA TODOS (SPEC E3), con MODELOS FALSOS — 28-sep-2026.

Nada de red ni de modelos de pago: el cliente falso contesta según la TAREA
que abre cada prompt y guarda lo que le preguntaron; Qdrant y los embeddings
son falsos y registran qué se buscó. Se comprueba, con el 631 SINTÉTICO:

  · la pregunta decisiva sale de UNA llamada barata (contada: llamadas, tokens
    y USD) con la figura, las consultas sobre la figura, la interpretación
    conforme, la cita literal verificada y la pregunta como la planteó la
    recurrida, aparte;
  · el RAG AÑADE las consultas sobre la figura contra el vector `rubro`, con
    cupo, marcadas `para` el principal y sin lo que perdió vigencia; lo de hoy
    no se toca;
  · internet se apunta a la pregunta decisiva y a la interpretación conforme;
  · el prompt de la propuesta y el guion y el criterio del estudio llevan «LA
    CUESTIÓN DECISIVA»; el `formato` de la propuesta no cambia;
  · la tarjeta: `principal.pregunta` es la decisiva y `pregunta_recurrida` va
    aparte; si el principal cambió, no se usa;
  · la deliberación la reutiliza (no la vuelve a pagar);
  · la bandera apagada = ninguna llamada; y las plantillas no llevan casos.

El caso (AR 631/2025): la recurrida concedió porque la sustitución procesal
alteró la cosa juzgada; lo que decide es si el tercero adquirente del inmueble
objeto de un juicio sobre una acción personal puede sustituirse válidamente en
la ejecución. Son DATOS de la prueba, no de ningún prompt.

    .venv/bin/python test_pregunta_decisiva.py
"""
import ast
import asyncio
import json
import os
import re
import sys
import types

os.environ.pop("PREGUNTA_DECISIVA_ACTIVA", None)
for _v in ("DELIBERACION_ACTIVA", "DELIBERACION_CUENTAS"):
    os.environ.pop(_v, None)
os.environ["MODELO_RESPALDO"] = "0"          # sin petición de respaldo en pruebas

import deliberacion as dl
import fase5_propuesta as f5
import fase6_estudio as f6
import fase6_rag as rag
import plan_estudio as pe
import pregunta_decisiva as pd
import taller_estado as te
import tarjeta_decision as td

FALLOS = []


def ok(cond, que):
    print(f"   {'PASA ' if cond else 'FALLA'}  {que}")
    if not cond:
        FALLOS.append(que)


# ═══ EL CASO (sintético) ═════════════════════════════════════════════════════
ACTO = (
    "RESULTANDOS. " + "Antecedentes del juicio de arrendamiento. " * 400 +
    "CONSIDERANDO QUINTO. Estudio. Este juzgado estima que la sustitución procesal de la "
    "parte actora alteró sustancialmente la cosa juzgada al introducir a un tercero ajeno "
    "al juicio natural en la etapa de ejecución, por lo que procede conceder el amparo. "
    "RESOLUTIVOS. PRIMERO. Se sobresee respecto del acto atribuido al actuario. SEGUNDO. "
    "La Justicia de la Unión ampara y protege a María López Ruiz.")
P1 = {"pregunta": "¿La sustitución de la parte actora en la ejecución alteró la cosa juzgada?",
      "jerarquia": "principal", "clase": "fondo",
      "resolvio": "Concedió el amparo porque la sustitución procesal alteró la cosa juzgada.",
      "combate": "El adquirente del inmueble es causahabiente de la actora y se subroga en sus "
                 "derechos litigiosos, por lo que su intervención no altera la cosa juzgada."}
P2 = {"pregunta": "¿La sentencia recurrida es incongruente por no ocuparse de la escritura?",
      "jerarquia": "accesorio", "clase": "fondo", "depende_de": 1,
      "resolvio": "No se pronunció sobre la escritura de compraventa.",
      "combate": "La sentencia es incongruente porque no se ocupó de la escritura."}
PROBLEMAS = [P1, P2]
DECISIVA = ("¿El tercero adquirente del inmueble objeto de un juicio sobre una acción personal "
            "puede sustituirse válidamente a la actora en la ejecución de la sentencia?")
BUSQUEDAS = ["SUSTITUCIÓN PROCESAL. EL ADQUIRENTE DEL INMUEBLE ARRENDADO EN LA EJECUCIÓN",
             "CAUSAHABIENCIA PROCESAL. ADQUIRENTE DEL BIEN LITIGIOSO",
             "¿SUCESIÓN PROCESAL. ALCANCES?"]


class _R:
    def __init__(self, txt, prompt):
        self.choices = [types.SimpleNamespace(message=types.SimpleNamespace(content=txt))]
        # EL USO, PROPORCIONAL AL PROMPT (≈ 4 caracteres por token): así el
        # contador de coste mide algo parecido a lo real sin llamar a nadie.
        self.usage = types.SimpleNamespace(prompt_tokens=len(prompt) // 4,
                                           completion_tokens=len(txt) // 4 + 600,
                                           completion_tokens_details=None)


class Falso:
    def __init__(self, contestar):
        self.contestar = contestar
        self.llamadas = []
        self.chat = self
        self.completions = self

    async def create(self, **kw):
        p = kw["messages"][0]["content"]
        self.llamadas.append((p.splitlines()[0], p, kw))
        return _R(self.contestar(p, kw), p)

    def de(self, tarea):
        return [x for x in self.llamadas if x[0].startswith("TAREA: " + tarea)]


def contestar(p, kw):
    t = p.splitlines()[0]
    if t.startswith("TAREA: LA PREGUNTA DECISIVA"):
        return json.dumps({
            "figura": "SUSTITUCIÓN PROCESAL DEL ADQUIRENTE EN LA EJECUCIÓN",
            "pregunta_decisiva": DECISIVA,
            "proposicion_toral": {"dice": "La sustitución alteró la cosa juzgada.",
                                  "cita": "la sustitución procesal de la parte actora alteró "
                                          "sustancialmente la cosa juzgada al introducir a un tercero"},
            "hechos_que_deciden": ["La compraventa del inmueble arrendado.",
                                   "El juicio natural versó sobre la rescisión del arrendamiento."],
            "busquedas": BUSQUEDAS + [BUSQUEDAS[0].lower()],            # repetida: se quita
            "interpretacion_conforme": {"precepto": "artículo 2294 del Código Civil del Estado",
                                        "por_que": "admite leer la subrogación del adquirente "
                                                   "como extensiva a la ejecución"}})
    return "{}"


# ═══ 1 · LA PREGUNTA, UNA LLAMADA BARATA ════════════════════════════════════
print("\n1 · la pregunta decisiva (una llamada, contada)")
cli = Falso(contestar)
doc = asyncio.run(pd.formular(cli, problemas=PROBLEMAS, resumen_acto="Concedió el amparo.",
                              resumen_conceptos="La tercera sostiene la causahabiencia.",
                              texto_acto=ACTO, tipo_asunto="amparo_revision", es_recurso=True))
ok(len(cli.llamadas) == 1 and cli.de("LA PREGUNTA DECISIVA"), "una sola llamada, la de la pregunta")
_kw = cli.llamadas[0][2]
ok(_kw.get("model") == dl.MODELO_LECTURA and _kw.get("reasoning_effort") == "low",
   "con el modelo de lectura y esfuerzo bajo (la barata)")
ok(doc["pregunta_decisiva"] == DECISIVA and doc["formulada"] and doc["numero"] == 1,
   "sale la pregunta decisiva del principal (problema 1)")
ok(doc["pregunta_recurrida"] == P1["pregunta"], "la pregunta como la planteó la recurrida, aparte")
ok(doc["figura"].startswith("SUSTITUCIÓN PROCESAL"), "sale la figura")
ok(doc["proposicion_toral"]["verificada"] and "alteró sustancialmente" in doc["proposicion_toral"]["cita"],
   "la cita de la proposición toral, verificada contra el acto")
ok(len(doc["busquedas"]) == 3 and all("?" not in q and "¿" not in q for q in doc["busquedas"]),
   "consultas sobre la figura, sin repetir y sin signos de pregunta (prosa = lo peor medido)")
ok(doc["interpretacion_conforme"] == {"precepto": "artículo 2294 del Código Civil del Estado",
                                      "por_que": "admite leer la subrogación del adquirente como "
                                                 "extensiva a la ejecución"},
   "la interpretación conforme por examinar, con precepto y porqué")
u = doc["uso"]
print(f"      uso medido: {u['llamadas']} llamada · {u['entrada']} + {u['salida']} tokens · "
      f"{u['coste_usd']} USD")
ok(u["llamadas"] == 1 and 0 < u["coste_usd"] <= 0.01,
   f"coste por asunto dentro de lo previsto (≤ 0.01 USD; medido {u['coste_usd']})")
ok("TEXTO LITERAL DE LA RESOLUCIÓN" in cli.llamadas[0][1] and "CONSIDERANDO QUINTO" in cli.llamadas[0][1],
   "el prompt lleva el texto literal donde está la consideración")

# la que falla no se usa
_cf = Falso(lambda p, kw: "sin json")
_df = asyncio.run(pd.formular(_cf, problemas=PROBLEMAS, texto_acto=ACTO))
ok(not _df["formulada"] and not pd.util(_df) and pd.consultas_rag(_df) == [],
   "si el modelo no responde, no hay pregunta decisiva y nadie la usa")


# ═══ 2 · EL RAG AÑADE LA FIGURA ═════════════════════════════════════════════
print("\n2 · el RAG (consultas sobre la figura contra `rubro`)")


def _t(reg, rubro, inst="Primera Sala", tipo="JURISPRUDENCIA", vincula=True):
    return {"registro": reg, "rubro": rubro, "instancia": inst, "tipo": tipo, "texto": "…",
            "vincula": vincula}


class QdrantFalso:
    def __init__(self, textos):
        self.textos = textos
        self.consultas = []

    def query_points(self, collection_name, query, using, limit, query_filter=None, with_payload=True):
        texto = self.textos[int(query[0])]
        self.consultas.append((collection_name, using, limit, texto))
        if "SUSTITUCIÓN" in texto:
            res = [_t("2015688", "CAUSAHABIENTE PROCESAL. EL ADQUIRENTE PUEDE SUSTITUIRSE."),
                   _t("188480", "SUSTITUCIÓN PROCESAL. PROCEDE SIN CESIÓN EXPRESA.",
                      "Tribunales Colegiados de Circuito", "TESIS AISLADA", False),
                   # ABANDONADA (vigencia_tesis): no entra por esta puerta
                   _t("2009817", "CONTROL DIFUSO. LOS COLEGIADOS PUEDEN EJERCERLO.", "Pleno",
                      "TESIS AISLADA", False)]
        elif "CAUSAHABIENCIA" in texto:
            res = [_t("173604", "CAUSAHABIENCIA. SUS EFECTOS PROCESALES."),
                   _t("2015688", "CAUSAHABIENTE PROCESAL. EL ADQUIRENTE PUEDE SUSTITUIRSE.")]
        else:
            res = [_t("168958", "COSA JUZGADA. SUS ELEMENTOS Y LÍMITES OBJETIVOS."),
                   _t("2007402", "SUCESIÓN PROCESAL. SUS ALCANCES.", "Tribunales Colegiados de Circuito",
                      "TESIS AISLADA", False)]
        return types.SimpleNamespace(points=[types.SimpleNamespace(payload=x) for x in res])


_textos = []


async def embed_juris(t):
    _textos.append(t)
    return [float(len(_textos) - 1)]


qd = QdrantFalso(_textos)
_cr = pd.consultas_rag(doc)
tesis_fig = asyncio.run(rag.tesis_de_la_figura(qd, embed_juris, _cr))
ok(_textos == _cr and all(("SUSTITUCIÓN" in q or "CAUSAHABIENCIA" in q or "SUCESIÓN" in q)
                          for q in _textos),
   "lo que se vectoriza son las consultas sobre la figura (no la pregunta de la recurrida)")
ok(all(c[0] == rag.COLECCION_JURIS and c[1] == rag.VECTOR_RUBRO for c in qd.consultas)
   and len(qd.consultas) == len(_cr), "contra la colección de jurisprudencia y el vector `rubro`")
ok(not any("cosa juzgada" in q.lower() for q in _textos),
   "ninguna consulta de la figura repite la cosa juzgada")
_regs = [t["registro"] for t in tesis_fig]
ok("2015688" in _regs and "173604" in _regs and "188480" in _regs,
   "trae el anaquel de la figura (causahabiencia, sustitución procesal)")
ok("2009817" not in _regs, "lo abandonado no entra por la figura")
ok(_regs[0] == "2015688", "la que contestan dos consultas va primero (fusión por rango)")
ok(all(t.get("de_figura") for t in tesis_fig), "cada una marcada «de la figura»")
_pocas = asyncio.run(rag.tesis_de_la_figura(QdrantFalso(_textos := []), embed_juris, _cr, cupo=2))
ok(len(_pocas) == 2, "con su cupo")

# se SUMA al material, marcada `para` el principal; lo de hoy no se toca
T_COSA = {"registro": "168958", "rubro": "COSA JUZGADA. SUS ELEMENTOS Y LÍMITES OBJETIVOS.",
          "instancia": "Primera Sala", "tipo": "JURISPRUDENCIA", "texto": "…", "para": [1],
          "obligatoria": True}
T_OTRO = {"registro": "2000002", "rubro": "CONGRUENCIA. SU ALCANCE.", "instancia": "Primera Sala",
          "tipo": "JURISPRUDENCIA", "texto": "…", "para": [2], "obligatoria": True}
m = f6.Material()
m.tesis = [dict(T_COSA), dict(T_OTRO)]
m.tipo_asunto = "amparo_revision"
m.materia = "civil"
_antes = [dict(x) for x in m.tesis]
# la consulta del GÉNERO trae también la genérica de cosa juzgada, que el
# material ya tenía: se marca, no se repite.
ok("168958" in _regs, "(la consulta del género alcanza también la genérica que ya estaba)")
res = rag.sumar_figura(m, tesis_fig, pd.numero(doc))
_nuevas = [x for x in m.tesis if x["registro"] in _regs and x["registro"] != "168958"]
ok(res["nuevas"] == len(tesis_fig) - 1 and len(_nuevas) == res["nuevas"]
   and all(x["para"] == [1] and x["de_figura"] for x in _nuevas),
   "las de la figura entran marcadas `para` el principal (problema 1)")
ok(m.tesis[0]["registro"] == "2015688", "delante: el reparto de la propuesta las pone primero en el principal")
_cosa = next(x for x in m.tesis if x["registro"] == "168958")
ok(res["marcadas"] == 1 and _cosa["para"] == [1] and _cosa.get("de_figura")
   and sum(1 for x in m.tesis if x["registro"] == "168958") == 1,
   "la que ya estaba no se repite: gana la marca")
ok(next(x for x in m.tesis if x["registro"] == "2000002") == _antes[1],
   "lo del otro problema no se toca")
_rep = f5._tesis_del_material(m)
ok(_rep[0]["registro"] in _regs and any(t["registro"] == "2015688" for t in _rep),
   "la propuesta recibe las de la figura en su reparto por turnos")


# material_del_caso: con la figura ya formulada, en el mismo tiro (botón/rescate)
async def _mp_falso(qdrant, ej, el, problema, *a, **k):
    x = f6.Material()
    x.tesis = [dict(T_COSA)] if "cosa juzgada" in problema else [dict(T_OTRO)]
    return x

_orig_mp = rag.material_para
rag.material_para = _mp_falso
try:
    _textos.clear()
    mc = asyncio.run(rag.material_del_caso(QdrantFalso(_textos), embed_juris, None,
                                           PROBLEMAS, figura=_cr, principal=1))
finally:
    rag.material_para = _orig_mp
_mc_regs = {x["registro"]: x for x in mc.tesis}
ok("2015688" in _mc_regs and _mc_regs["2015688"]["para"] == [1] and "2000002" in _mc_regs,
   "material_del_caso suma la figura (para el principal) sin quitar lo de cada problema")
_textos.clear()
mc0 = asyncio.run(rag.material_del_caso(QdrantFalso(_textos), embed_juris, None, [], figura=None))
ok(_textos == [], "sin figura, material_del_caso no busca nada más")


# ═══ 3 · INTERNET ═══════════════════════════════════════════════════════════
print("\n3 · internet (la pregunta decisiva y la interpretación conforme)")
_qi = pd.pregunta_internet(doc, P1["pregunta"])
ok(_qi.startswith(DECISIVA) and "SUSTITUCIÓN PROCESAL" in _qi and "2294" in _qi
   and "Interpretación conforme" in _qi, "la búsqueda en internet lleva la decisiva, la figura y el precepto")
ok("cosa juzgada" not in _qi.lower(), "y no la pregunta de la recurrida")
ok(pd.pregunta_internet(None, P1["pregunta"]) == P1["pregunta"], "sin pregunta decisiva, la de siempre")


# ═══ 4 · LA PROPUESTA ═══════════════════════════════════════════════════════
print("\n4 · la propuesta (fase 5)")
m.decisiva = doc
cp = Falso(lambda p, kw: json.dumps({"global": {"alcanza": False}, "propuestas": []}))
asyncio.run(f5.proponer(cp, PROBLEMAS, m, "Concedió.", "Causahabiencia.", True,
                        contraste_previo=[]))
_pp = cp.llamadas[-1][1]
ok("LA CUESTIÓN DECISIVA DEL PROBLEMA 1" in _pp and f"LA CUESTIÓN QUE DECIDE: {DECISIVA}" in _pp,
   "el prompt de la propuesta lleva LA CUESTIÓN DECISIVA del principal")
ok(f"ASÍ LO PLANTEÓ LA RECURRIDA: {P1['pregunta']}" in _pp,
   "con «así lo planteó la recurrida» como dato")
ok("LA FIGURA: SUSTITUCIÓN PROCESAL" in _pp and "INTERPRETACIÓN CONFORME POR EXAMINAR" in _pp,
   "la figura y la interpretación conforme")
ok("La propuesta global razona sobre esa cuestión" in _pp, "el global razona sobre ella")
ok(re.search(r"\[registro 2015688\][^\n]*responde al problema 1 · de la figura", _pp) is not None,
   "las tesis de la figura, rotuladas y para el problema 1")
ok(_pp.index("LA CUESTIÓN DECISIVA") < _pp.index("LO QUE RESOLVIÓ"),
   "va junto a los problemas, antes de lo resuelto")
# el principal cambió: no se usa
_otro = [dict(P1, jerarquia="accesorio"), dict(P2, jerarquia="principal")]
cp2 = Falso(lambda p, kw: "{}")
asyncio.run(f5.proponer(cp2, _otro, m, "", "", True, contraste_previo=[]))
ok("LA CUESTIÓN DECISIVA" not in cp2.llamadas[-1][1],
   "si el secretario cambió el principal, la pregunta de otro problema no entra")
m.decisiva = None
cp3 = Falso(lambda p, kw: "{}")
asyncio.run(f5.proponer(cp3, PROBLEMAS, m, "", "", True, contraste_previo=[]))
ok("LA CUESTIÓN DECISIVA" not in cp3.llamadas[-1][1], "sin pregunta decisiva, el prompt de siempre")
_main = open("main.py", encoding="utf8").read()
ok(_main.count('"formato": 2') >= 1 and "!= 2" in _main,
   "el `formato` de la propuesta sigue siendo 2 (no se recalculan las guardadas)")


# ═══ 5 · EL ESTUDIO: CRITERIO Y GUION ═══════════════════════════════════════
print("\n5 · el estudio (criterio del principal y guion)")
crit = [f6.Criterio(problema=P2["pregunta"], sentido="innecesario", jerarquia="accesorio"),
        f6.Criterio(problema=P1["pregunta"], sentido="fundado", jerarquia="principal",
                    razonamiento="El adquirente puede sustituirse.")]
_bc = f6._bloque_criterio(crit, "civil", "", "amparo_revision", "estandar", [], decisiva=doc)
_i1 = _bc.index("[PRINCIPAL]")
_i2 = _bc.index("[ACCESORIO]")
ok(f"LA CUESTIÓN DECISIVA: {DECISIVA}" in _bc and _i1 < _bc.index("LA CUESTIÓN DECISIVA") < _i2,
   "el criterio del principal lleva LA CUESTIÓN DECISIVA, bajo el principal")
ok(f"ASÍ LO PLANTEÓ LA RECURRIDA: {P1['pregunta']}" in _bc and "SE ORGANIZA EN TORNO A LA CUESTIÓN DECISIVA" in _bc,
   "y la de la recurrida como marco; el estudio se organiza en torno a la decisiva")
_bc2 = f6._bloque_criterio(crit, "civil", "", "amparo_revision", "moderna", [], variante="v2",
                           decisiva=doc)
ok("LA CUESTIÓN DECISIVA" in _bc2, "también en la v2 y en la moderna")
_bc0 = f6._bloque_criterio(crit, "civil", "", "amparo_revision", "estandar", [])
ok("LA CUESTIÓN DECISIVA" not in _bc0 and _bc0 == f6._bloque_criterio(
    crit, "civil", "", "amparo_revision", "estandar", [], decisiva=None),
   "sin pregunta decisiva, el criterio de siempre")
_crit_otro = [f6.Criterio(problema="¿Otra pregunta reescrita?", sentido="fundado", jerarquia="principal")]
ok("LA CUESTIÓN DECISIVA" not in f6._bloque_criterio(_crit_otro, "civil", "", "amparo_revision",
                                                     "estandar", [], decisiva=doc),
   "si la pregunta del principal ya es otra, no se cuelga de él")
_g = pe.vista({"problemas": [{"id": "P1", "sentido": "fundado"}]}, "estandar", decisiva=doc)
ok(f"LA CUESTIÓN DECISIVA (problema 1): {DECISIVA}" in _g and "planteada así: " + P1["pregunta"] in _g,
   "el guion lleva LA CUESTIÓN DECISIVA como dato")
ok("LA CUESTIÓN DECISIVA" not in pe.vista({"problemas": [{"id": "P1", "sentido": "fundado"}]}, "estandar"),
   "sin ella, el guion de siempre")
ok("decisiva=_dec_g" in _main and "_pd_g.de_material(ses.get(\"material\"), _te.problemas_de(r))" in _main,
   "main pasa al guion la pregunta vigente del material")
ok("decisiva=getattr(material, \"decisiva\", None)" in open("fase6_estudio.py", encoding="utf8").read(),
   "los dos redactores del estudio pasan la del material")


# ═══ 6 · LA TARJETA ═════════════════════════════════════════════════════════
print("\n6 · la tarjeta (principal.pregunta y pregunta_recurrida)")
_resp = {"formato": 2, "global": {"alcanza": True, "sentido": "fundado", "razon": "Prospera.",
                                   "apoyos": []},
         "propuestas": [{"problema": P1["pregunta"], "sentido": "fundado", "razon": "…", "alcanza": True},
                        {"problema": P2["pregunta"], "sentido": "innecesario", "razon": "…", "alcanza": True}],
         "contraste": []}
_mt = {"tesis": [], "espejo": [], "decisiva": doc}
tj = td.armar(_resp, _mt, PROBLEMAS, rama_info={"tipo_asunto": "amparo_revision"})
ok(tj["principal"]["pregunta"] == DECISIVA and tj["principal"]["pregunta_recurrida"] == P1["pregunta"],
   "principal.pregunta = la decisiva; pregunta_recurrida aparte")
ok(tj["principal"]["figura"].startswith("SUSTITUCIÓN PROCESAL"), "y la figura")
tj0 = td.armar(_resp, {"tesis": [], "espejo": []}, PROBLEMAS, rama_info={"tipo_asunto": "amparo_revision"})
ok(tj0["principal"]["pregunta"] == P1["pregunta"] and tj0["principal"]["pregunta_recurrida"] is None,
   "sin pregunta decisiva, la de la fase 3 y nada aparte")
tj2 = td.armar(_resp, _mt, [dict(P1, jerarquia="accesorio"), dict(P2, jerarquia="principal")],
               rama_info={"tipo_asunto": "amparo_revision"})
ok(tj2["principal"]["pregunta"] == P2["pregunta"] and tj2["principal"]["pregunta_recurrida"] is None,
   "si el principal cambió, la pregunta de otro problema no se enseña")
_P1r = dict(P1, pregunta="¿Pudo el adquirente sustituirse?", pregunta_original=P1["pregunta"])
tj3 = td.armar(_resp, _mt, [_P1r, P2], rama_info={"tipo_asunto": "amparo_revision"})
ok(tj3["principal"]["pregunta"] == DECISIVA,
   "si el secretario sólo reescribió la pregunta (pregunta_original), sigue valiendo")


# ═══ 7 · VIAJA CON EL MATERIAL, ENTRE WORKERS ═══════════════════════════════
print("\n7 · la fila (gunicorn -w 2)")
m.decisiva = doc
_l = te.material_ligero(m)
_m2 = te.material_rehidratado(json.loads(json.dumps(_l, ensure_ascii=False)))
ok(_m2.decisiva == json.loads(json.dumps(doc, ensure_ascii=False)),
   "material_ligero → material_rehidratado conserva la pregunta decisiva")
_mk = pd.marca(doc, "h1")
ok(pd.doc_de_marca(_mk, "h1") is doc and pd.doc_de_marca(_mk, "h2") is None
   and pd.doc_de_marca({"estado": "en_curso", "huella": "h1"}, "h1") is None,
   "la marca «decisiva» sólo vale para su adelanto y lista")


# ═══ 8 · LA DELIBERACIÓN LA REUTILIZA ═══════════════════════════════════════
print("\n8 · la deliberación no la vuelve a pagar")
m.decisiva = doc
cd = Falso(lambda p, kw: "{}")
dd = asyncio.run(dl.deliberar(cd, problemas=PROBLEMAS, material=m, textos={"acto": ACTO},
                              tipo_asunto="amparo_revision", es_recurso=True,
                              decisiva_previa=doc))
ok(not cd.de("LA PREGUNTA DECISIVA") and dd["principal"]["pregunta_decisiva"] == DECISIVA,
   "con la pregunta previa del mismo principal, no hay llamada de la etapa A")
cd2 = Falso(lambda p, kw: "{}")
asyncio.run(dl.deliberar(cd2, problemas=_otro, material=m, textos={"acto": ACTO},
                         tipo_asunto="amparo_revision", es_recurso=True, decisiva_previa=doc))
ok(len(cd2.de("LA PREGUNTA DECISIVA")) == 1, "si el principal es otro, la formula")


# ═══ 9 · LA BANDERA Y EL CABLEADO EN main.py ════════════════════════════════
print("\n9 · la bandera y main.py")
ok(pd.activa(), "encendida por omisión")
os.environ["PREGUNTA_DECISIVA_ACTIVA"] = "0"
ok(not pd.activa(), "«0» la apaga")
os.environ.pop("PREGUNTA_DECISIVA_ACTIVA", None)
_tree = ast.parse(_main)
_fn = {n.name: n for n in ast.walk(_tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
_dec_src = ast.get_source_segment(_main, _fn["_taller_decisiva"])
ok("_pd.activa()" in _dec_src and _dec_src.index("_pd.activa()") < _dec_src.index("formular"),
   "con la bandera apagada no se formula nada")
ok("CLAVE_MARCA" in _dec_src and "huella" in _dec_src, "se guarda como marca con huella")
_pc = ast.get_source_segment(_main, _fn["_taller_preconsultar"])
ok(_pc.index("_taller_decisiva(") < _pc.index("_ra.consultar(") < _pc.index("await _tarea_dec")
   < _pc.index("_taller_figura_al_material(") < _pc.index("_bd.reforzar(")
   < _pc.index("_taller_preproponer("),
   "en paralelo con la consulta; la figura entra antes del refuerzo y de la propuesta")
ok("_espera" in _pc and "PREGUNTA DECISIVA" in _pc, "la espera añadida se mide y se imprime")
_pn = ast.get_source_segment(_main, _fn["_taller_proponer_nucleo"])
ok("pregunta_internet(" in _pn and _pn.index("pregunta_internet(") < _pn.index("precedentes_verificados("),
   "internet se apunta a la pregunta decisiva antes de buscar")
_tc = ast.get_source_segment(_main, _fn["taller_consultar"])
ok("_taller_decisiva(" in _tc and "_taller_figura_al_material(" in _tc,
   "el botón «consultar» con contexto también suma la figura")
ok(_main.count("_taller_decisiva_guardada(user_email, numero,") == 4,
   "los cuatro rescates de la consulta suman la figura de la marca")
_nu = ast.get_source_segment(_main, _fn["_taller_deliberar_nucleo"])
ok("decisiva_previa=" in _nu, "la deliberación recibe la pregunta ya formulada")


# ═══ 10 · SIN FRASES MODELO NI CASOS EN LAS PLANTILLAS ══════════════════════
print("\n10 · las plantillas")
_neutro = {"formulada": True, "numero": 2, "pregunta_decisiva": "¿X?", "figura": "F",
           "pregunta_recurrida": "¿Y?", "es_recurso": True, "busquedas": ["F"],
           "proposicion_toral": {"dice": "D", "cita": ""}, "hechos_que_deciden": ["H"],
           "interpretacion_conforme": {"precepto": "art. N", "por_que": "P"}}
_todo = "\n".join([pd.bloque_propuesta(_neutro), "\n".join(pd.lineas_criterio(_neutro)),
                   "\n".join(pd.lineas_guion(_neutro)), pd.pregunta_internet(_neutro),
                   dl.prompt_pregunta_decisiva({"pregunta": "¿X?"}, None, "", "", "")])
ok(not re.search(r"631|2015688|168958|causahab|sustituci|Quer[eé]taro|cosa juzgada|arrend", _todo, re.I),
   "ningún caso real ni registro de ejemplo en las plantillas")
ok("PROBLEMA 2" in pd.bloque_propuesta(_neutro) and "(problema 2)" in pd.lineas_guion(_neutro)[0],
   "el número del principal sale del documento")
ok("ASÍ LLEGÓ PLANTEADO" in pd.bloque_propuesta(dict(_neutro, es_recurso=False)),
   "en amparo directo no se habla de «la recurrida»")

if FALLOS:
    print(f"\n{len(FALLOS)} FALLA(S)")
    sys.exit(1)
print("\nTODO PASA")
