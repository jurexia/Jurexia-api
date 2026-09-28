# -*- coding: utf-8 -*-
"""LA DELIBERACIÓN DEL PRINCIPAL, con MODELOS FALSOS — 28-sep-2026.

Nada de red ni de modelos de pago: el cliente falso contesta según la TAREA
que abre cada prompt y guarda lo que le preguntaron. Se comprueba:

  · el CONTRATO que proyecta la tarjeta (VIA, APOYO, SUERTE, crux, recomendada,
    estado) y el documento de la marca «deliberacion»;
  · la VERIFICACIÓN E: un identificador que no existe, un registro escrito a
    mano y una tesis abandonada no llegan a la tarjeta; un hecho sin cita
    literal queda «no acreditado»;
  · el JUEZ CIEGO: dos pasadas con el orden invertido, sin nada del motor; si
    no coinciden es «reñido»; si invocan «lo que obliga» con algo que sólo
    orienta, no es «claro»;
  · la FUERZA para un colegiado (la jurisprudencia de otro colegiado orienta);
  · la CONSECUENCIA por código (revocar la concesión niega lo que concedió y
    anuncia los conceptos omitidos, art. 93, fr. VI) y los SECUNDARIOS por el
    árbol en las dos vías;
  · la BANDERA APAGADA = ninguna llamada.

El caso es un 631 SINTÉTICO (AR 631/2025): el juez concedió porque la
sustitución procesal alteró la cosa juzgada; la recurrente sostiene que el
adquirente es causahabiente. Son DATOS de la prueba, no de ningún prompt.

    .venv/bin/python test_deliberacion.py
"""
import ast
import asyncio
import inspect
import json
import os
import re
import sys
import types

for _v in ("DELIBERACION_ACTIVA", "DELIBERACION_CUENTAS", "DELIBERACION_REGIONES"):
    os.environ.pop(_v, None)
os.environ["MODELO_RESPALDO"] = "0"          # sin petición de respaldo en pruebas

import deliberacion as dl

FALLOS = []


def ok(cond, que):
    print(f"   {'PASA ' if cond else 'FALLA'}  {que}")
    if not cond:
        FALLOS.append(que)


# ═══ EL CASO ═════════════════════════════════════════════════════════════════
ACTO = (
    "CONSIDERANDO QUINTO. Estudio. Este juzgado estima que la sustitución procesal de la "
    "parte actora alteró sustancialmente la cosa juzgada al introducir a un tercero ajeno "
    "al juicio natural en la etapa de ejecución, por lo que procede conceder el amparo. "
    "RESOLUTIVOS. ÚNICO. La Justicia de la Unión ampara y protege a María López Ruiz, "
    "contra el acto reclamado al Juez Tercero de lo Civil, consistente en el auto de doce "
    "de marzo de dos mil veinticinco, por los motivos expuestos en el considerando quinto.")
ESCRITO = (
    "AGRAVIOS. PRIMERO. El juez pasó por alto que el adquirente del inmueble arrendado es "
    "causahabiente de la actora y se subroga en sus derechos litigiosos desde la "
    "compraventa, de modo que su intervención en la ejecución no altera la cosa juzgada. "
    "SEGUNDO. La sentencia es incongruente porque no se ocupó de la escritura de compraventa.")
RESOLUTIVO = ("La Justicia de la Unión ampara y protege a María López Ruiz, contra el acto "
              "reclamado al Juez Tercero de lo Civil, consistente en el auto de doce de marzo "
              "de dos mil veinticinco, por los motivos expuestos en el considerando quinto.")
P1 = {"pregunta": "¿La sustitución de la parte actora en la ejecución alteró la cosa juzgada?",
      "jerarquia": "principal", "clase": "fondo",
      "resolvio": "Concedió el amparo porque la sustitución procesal alteró la cosa juzgada.",
      "combate": "El adquirente del inmueble es causahabiente de la actora y se subroga en sus "
                 "derechos litigiosos, por lo que su intervención no altera la cosa juzgada."}
P2 = {"pregunta": "¿La sentencia recurrida es incongruente por no ocuparse de la escritura?",
      "jerarquia": "accesorio", "clase": "fondo", "depende_de": 1,
      "resolvio": "No se pronunció sobre la escritura de compraventa.",
      "combate": "La sentencia es incongruente porque no se ocupó de la escritura de compraventa, "
                 "que acredita que el adquirente es causahabiente de la actora."}
PROBLEMAS = [P1, P2]

T_COSA = {"registro": "168958", "rubro": "COSA JUZGADA. SUS ELEMENTOS Y LÍMITES OBJETIVOS.",
          "instancia": "Primera Sala", "tipo": "JURISPRUDENCIA", "texto": "La cosa juzgada…",
          "para": [1], "obligatoria": True}
T_COLJ = {"registro": "2000001", "rubro": "COSA JUZGADA EN LA EJECUCIÓN. NO LA ALTERA EL CAMBIO DE ACTOR.",
          "instancia": "Tribunales Colegiados de Circuito", "tipo": "JURISPRUDENCIA",
          "texto": "…", "para": [1], "obligatoria": True}
T_ABANDONADA = {"registro": "2009817", "rubro": "CONTROL DIFUSO. LOS COLEGIADOS PUEDEN EJERCERLO.",
                "instancia": "Pleno", "tipo": "TESIS AISLADA", "texto": "…", "para": [1]}
T_OTRO_PROBLEMA = {"registro": "2000002", "rubro": "CONGRUENCIA. SU ALCANCE.",
                   "instancia": "Primera Sala", "tipo": "JURISPRUDENCIA", "texto": "…", "para": [2]}
BUSQUEDA = [
    {"registro": "2015688", "rubro": "CAUSAHABIENTE PROCESAL. EL ADQUIRENTE PUEDE SUSTITUIRSE EN LA EJECUCIÓN.",
     "instancia": "Primera Sala", "tipo": "JURISPRUDENCIA", "texto": "El adquirente…"},
    {"registro": "188480", "rubro": "SUSTITUCIÓN PROCESAL. PROCEDE SIN CESIÓN EXPRESA.",
     "instancia": "Tribunales Colegiados de Circuito", "tipo": "TESIS AISLADA", "texto": "…"},
    {"registro": "2000003", "rubro": "ALIMENTOS ENTRE CÓNYUGES. PROPORCIONALIDAD.",
     "instancia": "Primera Sala", "tipo": "JURISPRUDENCIA", "texto": "…"},
]
NORMAS = [{"cuerpo_legal": "Código Civil del Estado de Querétaro", "articulo": "2294",
           "texto": "El adquirente de la cosa arrendada se subroga en los derechos del arrendador."}]
PROPIOS = [{"expediente": "325/2024", "tipo_asunto": "amparo en revisión", "fecha": "2024-10-10",
            "sentido": "revoca"}]


def material():
    return types.SimpleNamespace(tesis=[dict(T_COSA), dict(T_COLJ), dict(T_ABANDONADA),
                                        dict(T_OTRO_PROBLEMA)],
                                 normas=[dict(n) for n in NORMAS], espejo=[], materia="civil",
                                 tipo_asunto="amparo_revision", entidad="Querétaro", sondeo=None)


# ═══ EL CLIENTE FALSO ═══════════════════════════════════════════════════════
class _R:
    def __init__(self, txt):
        self.choices = [types.SimpleNamespace(message=types.SimpleNamespace(content=txt))]
        self.usage = types.SimpleNamespace(prompt_tokens=1000, completion_tokens=200,
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
        return _R(self.contestar(p, kw))

    def de(self, tarea):
        return [x for x in self.llamadas if x[0].startswith("TAREA: " + tarea)]


def id_de(prompt, palabra):
    for m in re.finditer(r"^\[([TNO]\d+)\](.*?)(?=^\[[TNO]\d+\]|\Z)", prompt, re.S | re.M):
        if palabra.lower() in m.group(2).lower():
            return m.group(1)
    return None


def respuestas(juez="gana_fundado", abogado_a=None, abogado_b=None, escalon=1, cita_juez=None):
    def contestar(p, kw):
        t = p.splitlines()[0]
        if t.startswith("TAREA: LA PREGUNTA DECISIVA"):
            return json.dumps({
                "figura": "CAUSAHABIENCIA PROCESAL",
                "pregunta_decisiva": "¿El adquirente del inmueble arrendado puede sustituirse como "
                                     "actor en la ejecución sin cesión expresa de derechos litigiosos?",
                "proposicion_toral": {"dice": "La sustitución alteró la cosa juzgada.",
                                      # con UNA palabra de más al principio: se recorta
                                      "cita": "Considerando: la sustitución procesal de la parte actora "
                                              "alteró sustancialmente la cosa juzgada al introducir a un tercero"},
                "hechos_que_deciden": ["La compraventa del inmueble arrendado."]})
        if t.startswith("TAREA: LECTURA DE CANDIDATOS"):
            out = []
            for m in re.finditer(r"^(\d+)\. \[[^\]]*\] (.+)$", p, re.M):
                r = m.group(2)
                if "ALIMENTOS" in r:
                    out.append({"n": int(m.group(1)), "es": "ajena", "a_favor": "ninguna"})
                elif "CAUSAHABIENTE" in r or "SUSTITUCIÓN" in r:
                    out.append({"n": int(m.group(1)), "es": "resuelve", "a_favor": "prospera"})
                else:
                    out.append({"n": int(m.group(1)), "es": "distinguible", "a_favor": "no_prospera"})
            return json.dumps({"lectura": out})
        tc, tcos, tcol = id_de(p, "CAUSAHABIENTE PROCESAL"), id_de(p, "SUS ELEMENTOS"), id_de(p, "CAMBIO DE ACTOR")
        n1, o1 = id_de(p, "2294"), id_de(p, "325/2024")
        if t.startswith("TAREA: ABOGADO DE LA VÍA EN QUE EL PLANTEAMIENTO PRINCIPAL PROSPERA"):
            if abogado_a is not None:
                return abogado_a(p, locals())
            return json.dumps({
                "sentido": "fundado",
                "razon": f"El adquirente es causahabiente [{tc}] y así lo dice el registro 2099999.",
                "interpretacion": f"[{n1}] leído con [{tc}].",
                "regla": f"El causahabiente procesal puede sustituirse en la ejecución [{tc}] [T77].",
                "hechos": [{"afirma": "El adquirente se subroga en los derechos litigiosos.",
                            "cita": "el adquirente del inmueble arrendado es causahabiente de la actora "
                                    "y se subroga en sus derechos litigiosos", "fuente": "escrito"},
                           {"afirma": "Hubo cesión expresa ante notario.",
                            "cita": "consta la cesión expresa de derechos litigiosos otorgada ante notario",
                            "fuente": "acto"}],
                "subsuncion": "El adquirente cae en la figura.", "conclusion": "Fundado.",
                "objecion": {"de_la_otra_via": "La cosa juzgada impide cambiar de parte.",
                             "respuesta": f"No, por [{tc}]."},
                "autoridad_contraria": [{"id": tcos, "distincion": "Trata los límites objetivos."}],
                "propongo_aplicar": [tc, "T99", n1, "2099999"],
                "precedente_propio": [{"id": o1, "trato": "sigue", "por_que": "Mismo punto."}],
                "secundarios": [{"numero": 2, "relacion": "depende",
                                 "suerte": {"sentido": "innecesario", "razon": "Lo absorbe el principal."}}],
                "sostenible": True})
        if t.startswith("TAREA: ABOGADO DE LA VÍA EN QUE EL PLANTEAMIENTO PRINCIPAL NO PROSPERA"):
            if abogado_b is not None:
                return abogado_b(p, locals())
            return json.dumps({
                "sentido": "infundado",
                "razon": f"La sustitución alteró la cosa juzgada [{tcos}].",
                "interpretacion": f"[{tcos}].",
                "regla": f"La cosa juzgada no admite cambiar a la parte en ejecución [{tcos}].",
                "hechos": [{"afirma": "El juez lo sostuvo.",
                            "cita": "la sustitución procesal de la parte actora alteró sustancialmente "
                                    "la cosa juzgada", "fuente": "acto"}],
                "subsuncion": "…", "conclusion": "Infundado.",
                "objecion": {"de_la_otra_via": "Es causahabiente.", "respuesta": "No hay cesión."},
                "autoridad_contraria": [{"id": tc, "distincion": "Supone cesión expresa."}],
                "propongo_aplicar": [tcos],
                "precedente_propio": [],
                "secundarios": [{"numero": 2, "relacion": "depende",
                                 "suerte": {"sentido": "inoperante", "razon": "Descansa en la causahabiencia."},
                                 "presupone": {"premisa": "que el adquirente es causahabiente",
                                               "cita": "que acredita que el adquirente es causahabiente de la actora",
                                               "causa_propia": None}}],
                "sostenible": True})
        if t.startswith("TAREA: JUEZ DE LAS DOS VÍAS"):
            v1 = re.search(r"VÍA 1 — sentido: (\w+)", p).group(1)
            if juez == "siempre_1":
                rec = "1"
            else:
                rec = "1" if v1 == "fundado" else "2"
            fuentes = [cita_juez(p) if cita_juez else tc]
            return json.dumps({
                "recomendada": rec, "escalon": escalon, "fuentes": fuentes, "hechos": [f"{rec}.h1"],
                "por_que": ["Lo que obliga contesta la pregunta.", "Y el registro 2088888 lo confirma.",
                            "La otra vía no lo distingue."],
                "crux": {"que": "si el adquirente es causahabiente", "si_cambia": "se sostendría la otra",
                         "constancia": "la escritura de compraventa"},
                "debilidad": "No consta la cesión.", "otra_sostenible": True, "estado": "claro",
                "precedente_propio": {"se_aparta": False, "por_que": "Lo sigue."}})
        return ""
    return contestar


async def _buscar(pregunta, figura):
    return {"tesis": [dict(t) for t in BUSQUEDA], "normas": []}


def correr(cliente, **extra):
    kw = dict(problemas=PROBLEMAS, material=material(), resumen_acto="El juez concedió.",
              resumen_conceptos="La recurrente dice que es causahabiente.",
              textos={"acto": ACTO, "escrito": ESCRITO, "constancia": ""},
              tipo_asunto="amparo_revision", es_recurso=True, recurrente="Inmobiliaria del Centro",
              contraste=[{"numero": 1, "razon_toral": "la sustitución alteró la cosa juzgada",
                          "la_combate": True, "sobrevive": False, "veredicto_previo": "a_examinar",
                          "por_que": "La combate."}],
              constancias_faltantes=[{"que": "escritura de compraventa", "para_que": "causahabiencia",
                                      "indispensable": True}],
              resolvio_a_quo="concede", resolutivo_recurrida=RESOLUTIVO, quejoso="María López Ruiz",
              buscar=_buscar, filas_propias=PROPIOS)
    kw.update(extra)
    return asyncio.run(dl.deliberar(cliente, **kw))


# ═══ 1 · EL CASO QUE SE DECIDE CLARO ════════════════════════════════════════
print("\n1 · el contrato y la verificación, con el juez que coincide en las dos pasadas")
cli = Falso(respuestas())
d = correr(cli)
dump = json.dumps(d, ensure_ascii=False)

ok(d["formato"] == dl.FORMATO and d["origen"] == "deliberacion", "formato y origen")
pr = d["principal"]
ok(pr["pregunta_decisiva"].startswith("¿El adquirente") and pr["figura"] == "CAUSAHABIENCIA PROCESAL",
   "A · la pregunta decisiva y la figura")
ok(pr["proposicion_toral"]["verificada"] and pr["proposicion_toral"]["cita"].startswith("la sustitución procesal"),
   "A · la cita de la proposición toral se VERIFICA contra el acto (y se le recorta la palabra de más)")
cat = d["catalogo"]
regs = {e.get("registro"): e for e in cat.values() if e.get("clase") == "tesis"}
ok("2009817" not in regs, "B · la tesis ABANDONADA no entra al catálogo")
ok("2000003" not in regs, "B · lo que la lectura dice ajeno no entra")
ok("2000002" not in regs, "B · la tesis buscada para otro problema no entra al del principal")
ok(regs.get("2015688", {}).get("fuerza") == "obliga" and regs["2015688"]["escalon"] == 1,
   "B · la jurisprudencia de la Primera Sala OBLIGA (escalón 1)")
ok(regs.get("2000001", {}).get("fuerza") == "orienta" and regs["2000001"]["escalon"] == 2,
   "B · la jurisprudencia de colegiado ORIENTA aunque el acervo diga «obligatoria» (217, párr. 3)")
ok(any(e.get("clase") == "propio" and e.get("expediente") == "325/2024" for e in cat.values()),
   "B · el precedente propio entra como O, no como voto")
ok(any(e.get("clase") == "norma" and e.get("articulo") == "2294" for e in cat.values()),
   "B · la norma entra entera como N")
_ids = [k for k, e in cat.items() if e.get("clase") == "tesis" and e["escalon"] == 1]
ok(_ids and cat[_ids[0]]["registro"] == "2015688",
   "B · dentro del escalón, primero lo que la lectura dice que RESUELVE el punto")

va, vb = d["vias"]["A"], d["vias"]["B"]
ok("2099999" not in json.dumps(d["vias"], ensure_ascii=False)
   and "2088888" not in dump, "E · el registro escrito a mano que no está en el catálogo se QUITA de todo")
ok("2099999" not in " ".join(d["avisos"]) and any("quitó" in a for a in d["avisos"]),
   "E · el aviso dice cuántas se quitaron, no cuáles (no vuelve a enseñar el inventado)")
ok(d["verificacion"]["referencias_quitadas"] >= 4,
   "E · lo quitado queda CONTADO para el banco (T99, T77, 2099999, 2088888), sin guardar cuáles")
ok([a.get("registro") for a in va["apoyos"] if a.get("registro")] == ["2015688"]
   and any(a.get("norma") for a in va["apoyos"]), "E · propongo_aplicar = sólo lo del catálogo (T99 fuera)")
ok("(registro 2015688)" in va["razon"], "E · [T…] se traduce al registro verificado en el texto libre")
_q = []
ok(dl.limpiar_texto("según T1 y T99, con $ 250000.", {"T1": {"id": "T1", "clase": "tesis", "registro": "2015688"}}, _q)
   == "según (registro 2015688), con $ 250000." and _q == ["T99"],
   "E · el identificador sin corchetes también se traduce o se borra; una cantidad en pesos se queda")
h = va["cadena"]["hechos"]
ok(len(h) == 2 and h[0]["verificada"] and h[0]["fuente"] == "escrito"
   and not h[1]["verificada"] and h[1]["cita"] == "",
   "E · el hecho con cita literal queda acreditado; el inventado pierde la cita y NO está acreditado")
ok(d["verificacion"]["hechos_no_acreditados"] == 1, "E · el hecho no acreditado se cuenta")

ok(d["estado"] == "claro" and d["recomendada"] == "A", "D · las dos pasadas coinciden en la vía A: «claro»")
ok(d["juez"]["coinciden"] and [p["orden"] for p in d["juez"]["pasadas"]] == [["A", "B"], ["B", "A"]],
   "D · dos pasadas con el orden INVERTIDO")
jueces = cli.de("JUEZ DE LAS DOS VÍAS")
ok(len(jueces) == 2, "D · exactamente dos llamadas al juez")
ok(re.search(r"VÍA 1 — sentido: fundado", jueces[0][1]) and re.search(r"VÍA 1 — sentido: infundado", jueces[1][1]),
   "D · en la primera pasada la Vía 1 es la que prospera; en la segunda, la que no")
ok(len(cli.de("ABOGADO")) == 2 and sorted(x[0].endswith("NO PROSPERA") for x in cli.de("ABOGADO")) == [False, True],
   "C · dos abogados, uno por vía")
_a, _b = cli.de("ABOGADO DE LA VÍA EN QUE EL PLANTEAMIENTO PRINCIPAL PROSPERA"), \
    cli.de("ABOGADO DE LA VÍA EN QUE EL PLANTEAMIENTO PRINCIPAL NO PROSPERA")
ok(len(_a) == 1 and len(_b) == 1 and _a[0][2].get("reasoning_effort") == "high"
   and _b[0][2].get("reasoning_effort") == "high", "C · los abogados con esfuerzo ALTO (David rechazó el medio)")
ok(all(x[2].get("reasoning_effort") == "high" for x in jueces), "D · el juez con esfuerzo alto")
ok("QUÉ QUIERE DECIR ESA CALIFICACIÓN" in _a[0][1],
   "C · cada abogado sabe que la calificación es del agravio, no de la pregunta (lección del 631)")
ok("presupone" in _b[0][1] and "presupone" not in _a[0][1].split("LO QUE ESCRIBES")[1],
   "C · sólo la vía que no prospera declara `presupone`")
ok(d["crux"] and d["crux"]["que"] and d["crux"]["constancia"] == "la escritura de compraventa", "D · el crux")
ok(len(d["por_que"]) == 3 and not any("2088888" in x for x in d["por_que"]), "D · el porqué, tres renglones, limpio")

# F · consecuencia y secundarios
ok(va["rama"] == "revoca_fondo_niega" and any("no ampara ni protege a María López Ruiz" in x for x in va["desenlace"]),
   "F · vía A: revocar la concesión NIEGA lo que ella concedió, con su sujeto (577c700)")
ok(va["conceptos_omitidos"]["hacen_falta"] and "93" in va["desenlace_nota"] and "VI" in va["desenlace_nota"],
   "F · vía A anuncia el estudio de los conceptos omitidos (art. 93, fr. VI)")
ok(vb["rama"] == "confirma_concede" and any("ampara y protege a María López Ruiz" in x for x in vb["desenlace"]),
   "F · vía B: se confirma la concesión con el resolutivo del juzgado")
s2 = d["secundarios"][0]
ok(s2["numero"] == 2 and s2["en_A"]["sentido"] == "innecesario" and s2["en_A"]["de"] == "principal",
   "F · el árbol: con A, el accesorio queda sin materia")
ok(s2["en_B"]["sentido"] == "inoperante" and s2["en_B"]["relacion"] == "presupone",
   "F · el árbol: con B, cae con lo desestimado porque su cita de «presupone» consta")
ok(not s2["en_A"]["recalificar"] and not s2["en_B"]["recalificar"], "F · nada queda por recalificar")
ok("sin materia" in va["efecto"] and "cae" in vb["efecto"], "F · el efecto de cada vía lo cuenta el código")
ok("%" not in dump, "sin porcentajes en ninguna parte")

# ═══ 2 · LA PROYECCIÓN SOBRE LA TARJETA ══════════════════════════════════════
print("\n2 · la proyección sobre el contrato de la tarjeta")
t = dl.para_tarjeta(d, sentido_motor="infundado")
ok(t["recomendada"] == "propuesta" and t["estado"] == "claro", "recomendada «propuesta» sólo con «claro»")
ok(t["vias"]["propuesta"]["sentido"] == "fundado" and t["vias"]["opuesta"]["sentido"] == "infundado",
   "la propuesta es la del juez, aunque el motor propusiera lo contrario")
VIA = {"sentido", "prospera", "razon", "efecto", "desenlace", "desenlace_nota", "interpretacion",
       "cadena", "objecion", "apoyos", "via_protectora"}
ok(VIA <= set(t["vias"]["propuesta"]) and VIA <= set(t["vias"]["opuesta"]), "VIA con todos sus campos")
ok({"regla", "hechos", "subsuncion", "conclusion"} <= set(t["vias"]["propuesta"]["cadena"])
   and {"afirma", "cita", "fuente"} <= set(t["vias"]["propuesta"]["cadena"]["hechos"][0]), "la cadena")
ok({"de_la_otra_via", "respuesta"} <= set(t["vias"]["propuesta"]["objecion"]), "la objeción")
APOYO = {"registro", "rubro", "instancia", "tipo", "fuerza", "fuerza_texto", "vigencia", "de_internet",
         "en_acervo", "norma"}
_ap = [a for a in t["vias"]["propuesta"]["apoyos"] if a.get("registro")][0]
ok(APOYO <= set(_ap) and _ap["fuerza"] in dl.CODIGOS_FUERZA and _ap["en_acervo"] is True, "APOYO del contrato")
SUERTE = {"sentido", "de", "por_que", "relacion", "guarda", "recalificar", "previsto"}
sc = t["secundarios"][0]
ok(SUERTE <= set(sc["en_propuesta"]) and SUERTE <= set(sc["en_opuesta"])
   and sc["en_propuesta"]["sentido"] == "innecesario" and sc["en_opuesta"]["sentido"] == "inoperante",
   "SUERTE en cada vía, del árbol")
ok(sc["en_propuesta"]["de"] in ("principal", "arbol", "motor", "secretario")
   and sc["en_opuesta"]["relacion"] in ("depende", "presupone", "distinto", "autonoma"), "vocabulario de SUERTE")
ok(set(t["que_la_cambiaria"]["crux"]) == {"que", "si_cambia", "constancia"}, "crux del contrato")
ok(t["deliberacion"]["origen"] == "deliberacion" and t["deliberacion"]["pregunta_decisiva"]
   and set(t["deliberacion"]["proposicion_toral"]) == {"dice", "cita"}, "el bloque «deliberacion»")
ok(t["conceptos_omitidos"]["hacen_falta"], "los conceptos omitidos llegan a la tarjeta")

# ═══ 3 · EL JUEZ QUE NO COINCIDE ═════════════════════════════════════════════
print("\n3 · el juez con sesgo de posición: siempre la Vía 1")
cli3 = Falso(respuestas(juez="siempre_1"))
d3 = correr(cli3)
ok(d3["estado"] == "reñido" and d3["recomendada"] is None and d3["inclinacion"] is None,
   "si las dos pasadas no coinciden: «reñido», sin recomendación")
ok(any("no recomiendan la misma vía" in x for x in d3["estado_por_que"]), "y lo dice")
t3 = dl.para_tarjeta(d3, sentido_motor="infundado")
ok(t3["recomendada"] is None and t3["vias"]["propuesta"]["sentido"] == "infundado",
   "sin inclinación, la columna izquierda sigue el orden de la propuesta del motor; no se rotula recomendada")

# ═══ 4 · «LO QUE OBLIGA» CON ALGO QUE SÓLO ORIENTA ═══════════════════════════
print("\n4 · el juez invoca el escalón 1 con una jurisprudencia de colegiado")


def _a_solo_colegiado(p, v):
    return json.dumps({"sentido": "fundado", "razon": f"[{v['tcol']}]", "regla": f"[{v['tcol']}]",
                       "hechos": [], "propongo_aplicar": [v["tcol"]], "secundarios": [],
                       "sostenible": True})


cli4 = Falso(respuestas(abogado_a=_a_solo_colegiado, cita_juez=lambda p: id_de(p, "CAMBIO DE ACTOR")))
d4 = correr(cli4)
ok(d4["estado"] == "reñido" and d4["inclinacion"] == "A",
   "coinciden, pero «obliga» no se comprueba: no es «claro» (queda la inclinación)")
ok(any("obliga" in x for x in d4["estado_por_que"]), "y dice por qué se rebajó")

# ═══ 5 · NINGUNA VÍA CON APOYO ══════════════════════════════════════════════
print("\n5 · ninguna vía con apoyo verificado")


def _sin_apoyo(sentido):
    return lambda p, v: json.dumps({"sentido": sentido, "razon": "sin fuentes", "regla": "…",
                                    "hechos": [], "propongo_aplicar": ["T99", "2099999"],
                                    "secundarios": [], "sostenible": False})


cli5 = Falso(respuestas(abogado_a=_sin_apoyo("fundado"), abogado_b=_sin_apoyo("infundado")))
d5 = correr(cli5)
ok(d5["estado"] == "no_alcanza" and d5["recomendada"] is None, "«no_alcanza» por código")
ok(len(cli5.de("JUEZ")) == 0, "y no se gasta en el juez")
ok(dl.para_tarjeta(d5)["recomendada"] is None, "la tarjeta no recomienda nada")

# ═══ 6 · UN ABOGADO QUE NO RESPONDE, UNO QUE SE EQUIVOCA DE VÍA ══════════════
print("\n6 · un abogado mudo y otro que devuelve el sentido de la otra vía")
cli6 = Falso(respuestas(abogado_b=lambda p, v: "", abogado_a=lambda p, v: json.dumps(
    {"sentido": "infundado", "razon": f"[{v['tc']}]", "propongo_aplicar": [v["tc"]], "secundarios": []})))
d6 = correr(cli6)
ok(d6["vias"]["A"]["sentido"] == "fundado" and any("no es de su vía" in x for x in d6["avisos"]),
   "el abogado de la vía que prospera no puede devolver «infundado»")
ok(not d6["vias"]["B"]["respondio"] and d6["vias"]["B"]["sentido"] == "infundado"
   and d6["vias"]["B"]["desenlace"], "la vía muda existe: su consecuencia la calcula el código")
_bs = cli6.de("ABOGADO DE LA VÍA EN QUE EL PLANTEAMIENTO PRINCIPAL NO PROSPERA")
ok(len(_bs) == 2 and _bs[1][2]["max_completion_tokens"] == 2 * dl.TOKENS_ABOGADO
   and _bs[1][2].get("reasoning_effort") == "high",
   "vacía = se repite UNA vez con el doble de sitio y el MISMO esfuerzo")

# ═══ 7 · EL JUEZ NO SABE QUÉ PROPUSO EL MOTOR ═══════════════════════════════
print("\n7 · el juez ciego")
_firma = set(inspect.signature(dl.deliberar).parameters)
ok(not (_firma & {"propuesta", "sentido_motor", "global_", "glob", "propuesto"}),
   "deliberar() no recibe la propuesta del motor: no hay cómo revelársela al juez")
_pj = dl.prompt_juez(("A", "B"), {"A": d["vias"]["A"], "B": d["vias"]["B"]},
                     decisiva=d["principal"], cat={}, constancias_faltantes=[])
ok("motor" not in _pj.lower() and "propuesta" not in _pj.lower(), "el prompt del juez no habla del motor ni de su propuesta")
ok(_pj.index("1. LO QUE OBLIGA MANDA") < _pj.index("2. EL HECHO ACREDITADO") < _pj.index("3. PRESUNCIÓN")
   < _pj.index("4. EL PRECEDENTE PROPIO") < _pj.index("5. LA TASA BASE"), "el orden fijo de los cinco escalones")

# ═══ 8 · LA FUERZA PARA UN COLEGIADO ════════════════════════════════════════
print("\n8 · fuerza_para_colegiado")
F = dl.fuerza_para_colegiado
ok(F({"instancia": "Segunda Sala", "tipo": "JURISPRUDENCIA"})["fuerza"] == "obliga", "J de Sala: obliga")
ok(F({"instancia": "Pleno", "tipo": "JURISPRUDENCIA"})["fuerza"] == "obliga", "J del Pleno de la Corte: obliga")
ok(F({"instancia": "Primera Sala", "tipo": "TESIS AISLADA"})["fuerza"] == "orienta", "aislada de la Corte: orienta")
ok(F({"instancia": "Primera Sala", "tipo": "Precedente obligatorio"})["fuerza"] == "obliga",
   "precedente obligatorio de la Corte (arts. 222 y 223): obliga")
_fj = F({"instancia": "Tribunales Colegiados de Circuito", "tipo": "JURISPRUDENCIA", "obligatoria": True})
ok(_fj["fuerza"] == "orienta" and "217" in _fj["fuerza_texto"], "J de colegiado: orienta (217, párr. 3), aunque vincule")
ok(F({"instancia": "Plenos de Circuito", "tipo": "JURISPRUDENCIA"})["fuerza"] == "pleno_circuito",
   "Pleno de Circuito: rótulo propio, sin afirmar si obliga")
_pr = {"instancia": "Plenos Regionales", "tipo": "JURISPRUDENCIA", "numero_tesis": "PR.A.C.CN. J/7 K (12a.)"}
ok(F(_pr)["fuerza"] == "orienta" and F(_pr).get("por_confirmar"), "Pleno Regional sin región confirmada: orienta, por confirmar")
ok(F(_pr, region="CN")["fuerza"] == "obliga" and F(_pr, region="CS")["fuerza"] == "orienta",
   "Pleno Regional: obliga en su región, orienta en otra")
os.environ["DELIBERACION_REGIONES"] = "22:CN"
ok(F(_pr)["fuerza"] == "obliga", "la región del circuito se configura (DELIBERACION_REGIONES)")
os.environ.pop("DELIBERACION_REGIONES", None)
ok(F({"instancia": "Tribunales Colegiados de Circuito", "tipo": "TESIS AISLADA",
      "numero_tesis": "XXII.3o.A.C.12 C (11a.)"}, clave_propia="XXII.3o.A.C.")["fuerza"] == "precedente_propio",
   "la tesis del propio tribunal: precedente propio (228)")
ok(dl.region_de_clave("PR.P.T.CS. J/2 L (11a.)") == "CS" and dl.region_de_clave("1a./J. 5/2020") == "",
   "la región sale de la clave")
ok(dl.clave_de_tesis({"localizacion": "Gaceta S.J.F.; Tesis: PR.A.C.CN. J/7 K (12a.)"}) == "PR.A.C.CN. J/7 K (12a.)"
   and dl.clave_de_tesis({"localizacion": "[J]; 10a. Época; Pleno; Gaceta S.J.F.; Pág. 11"}) == "",
   "la clave se busca en la localización (el payload del taller no guarda `numero_tesis`); si no está, no se inventa")
ok(F({"instancia": "Plenos Regionales", "tipo": "JURISPRUDENCIA"}, region="CN").get("por_confirmar"),
   "sin la clave, ni con la región del circuito se afirma que un Pleno Regional obliga")

# ═══ 9 · LA BANDERA APAGADA = NINGUNA LLAMADA ═══════════════════════════════
print("\n9 · la bandera")
lanzadas = []
ok(not dl.activa_para("david@iurexia.com"), "apagada por omisión")
ok(dl.programar("david@iurexia.com", lambda: lanzadas.append(1)) is False and not lanzadas,
   "apagada: programar() no llama a nada")
os.environ["DELIBERACION_ACTIVA"] = "1"
ok(not dl.activa_para("david@iurexia.com") and not dl.programar("david@iurexia.com", lambda: lanzadas.append(1)),
   "encendida con la lista vacía: tampoco corre para nadie")
os.environ["DELIBERACION_CUENTAS"] = "soporte@iurexia.com, David@Iurexia.com"
ok(dl.activa_para("david@iurexia.com") and not dl.activa_para("otra@correo.com"), "sólo las cuentas de la lista")
ok(dl.programar("david@iurexia.com", lambda: lanzadas.append(1)) and lanzadas == [1], "y entonces sí lanza")
os.environ["DELIBERACION_CUENTAS"] = "*"
ok(dl.activa_para("cualquiera@correo.com"), "«*» la abre a todas (sólo tras las compuertas)")
os.environ["DELIBERACION_ACTIVA"] = "0"
ok(not dl.activa_para("cualquiera@correo.com"), "apagar la bandera manda sobre la lista")
for _v in ("DELIBERACION_ACTIVA", "DELIBERACION_CUENTAS"):
    os.environ.pop(_v, None)

# ═══ 10 · EL GANCHO EN main.py (sin importarlo: el arranque borra cachés) ════
print("\n10 · el gancho en main.py")
_src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "main.py"), encoding="utf8").read()
_arbol = ast.parse(_src)
_fn = {n.name: n for n in ast.walk(_arbol) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
ok(all(k in _fn for k in ("_taller_lanzar_deliberacion", "_taller_predeliberar", "_taller_deliberar_nucleo")),
   "las tres piezas existen")
_seg = ast.get_source_segment(_src, _fn["_taller_lanzar_deliberacion"])
ok("_delib.programar(" in _seg and _seg.index("ensure_future") > _seg.index("def _lanzar"),
   "la tarea sólo se crea DENTRO de lo que `programar` decide lanzar")
ok("_taller_lanzar_deliberacion(" in ast.get_source_segment(_src, _fn["_taller_preproponer"]),
   "se engancha tras la propuesta calculada sola")
ok("_taller_lanzar_deliberacion(" in ast.get_source_segment(_src, _fn["taller_proponer"]),
   "y tras una propuesta calculada en /taller/proponer")
_pd = ast.get_source_segment(_src, _fn["_taller_predeliberar"])
ok('"deliberacion"' in _pd and "huella" in _pd and "_taller_con_latido" in _pd,
   "marca «deliberacion» con huella y latido (gunicorn -w 2)")
_nu = ast.get_source_segment(_src, _fn["_taller_deliberar_nucleo"])
ok("_taller_guardar" not in _nu and "supabase" not in _nu, "el núcleo no escribe nada (lo usa el banco)")
ok("not e.es_recurso" in _nu,
   "en un recurso no se pasa el «quejoso» del formulario (es quien recurre: la lección del 631)")
ok(not re.search(r"\bsentido\b", re.sub(r"#.*", "", _nu)),
   "el núcleo no lee ni pasa el sentido que propuso el motor (la tasa del circuito sí: es dato)")

# ═══ 11 · LOS PROMPTS NO LLEVAN FRASES MODELO NI CASOS ═══════════════════════
print("\n11 · los prompts")
_neutro = {"pregunta": "¿X?", "resolvio": "Y", "combate": "Z"}
_ps = [dl.prompt_pregunta_decisiva(_neutro, None, "", "", ""),
       dl.prompt_lectura({"pregunta_decisiva": "¿X?"}, [{"registro": "1", "rubro": "R", "tipo": "", "instancia": ""}]),
       dl.prompt_abogado("A", pral=_neutro, pi=0, problemas=[_neutro], decisiva={}, contraste=None,
                         resumen_acto="", resumen_conceptos="", textos={}, cat={}),
       dl.prompt_abogado("B", pral=_neutro, pi=0, problemas=[_neutro], decisiva={}, contraste=None,
                         resumen_acto="", resumen_conceptos="", textos={}, cat={}),
       dl.prompt_juez(("A", "B"), {"A": d["vias"]["A"], "B": d["vias"]["B"]}, decisiva={}, cat={},
                      constancias_faltantes=[])]
_todo = "\n".join(_ps[:4])
ok(not re.search(r"631|2015688|168958|causahab|Quer[eé]taro|cosa juzgada", _todo, re.I),
   "ningún caso real ni registro de ejemplo en las plantillas")
ok(all(x.startswith("TAREA: ") for x in _ps), "cada prompt abre con su TAREA")
ok("%" not in _todo and "porcentaje" in _ps[4].lower() and "Sin porcentajes" in _ps[4],
   "el juez tiene prohibidos los porcentajes")

if FALLOS:
    print(f"\n{len(FALLOS)} FALLA(S)")
    sys.exit(1)
print("\nTODO PASA")
