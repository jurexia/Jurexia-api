# -*- coding: utf-8 -*-
"""LA FICHA PROCESAL (E2) Y LA PREGUNTA DECISIVA (E3), JUNTAS — 28-sep-2026.

AR 631/2025. E1 (la carátula), E2 (la ficha) y E3 (la pregunta decisiva) se
escribieron en paralelo y cada una pasa sus pruebas; aquí se prueba lo que sólo
existe al juntarlas, sin modelo de pago y sin red:

  1. la pregunta decisiva se formula CON la ficha delante: la misma ficha, pura
     y por código, que después ven la propuesta, la deliberación y el estudio;
  2. el prompt de la propuesta lleva la ficha y la cuestión decisiva, una vez
     cada una y en ese orden (quién es quién, y luego qué decide);
  3. la tarjeta devuelve `principal.pregunta` (la decisiva),
     `principal.pregunta_recurrida` y `ficha` con la forma del contrato, y
     ninguna contradice a la otra (quién recurre, qué órgano, cuántas
     responsables);
  4. el estudio v4 lleva la ficha en su encabezado de datos y la cuestión
     decisiva en el criterio del principal y en el guion, una vez cada una;
  5. la forma del 631 REAL que la ficha sintética de E2 no cubría: la
     responsable del formulario es «MAGISTRADA …, INTEGRANTE DE LA SALA …» y el
     juzgado la llama «la Sala … en el Estado de …» (una responsable, no dos);
     y el sobreseimiento del otro acto está en los considerandos, no en el
     único punto resolutivo (la ficha lo dice, con su acto).

Todo es SINTÉTICO: nombres, números y fechas de prueba. Los modelos son falsos
y se cuentan las llamadas.

    .venv/bin/python test_integracion_e.py
"""
import asyncio
import datetime as _dt
import json
import os
import re
import sys
import types

os.environ.pop("ESTUDIO_PROMPT", None)
os.environ.pop("PREGUNTA_DECISIVA_ACTIVA", None)
for _v in ("DELIBERACION_ACTIVA", "DELIBERACION_CUENTAS"):
    os.environ.pop(_v, None)
os.environ["MODELO_RESPALDO"] = "0"

import fase5_propuesta as f5
import fase6_estudio as f6
import fases123_pipeline as f123
import ficha_procesal as fp
import plan_estudio as pe
import pregunta_decisiva as pd
import redactor_adelanto as ra
import tarjeta_decision as td

FALLOS = []
AQUI = os.path.dirname(os.path.abspath(__file__))


def ok(cond, que):
    print(f"   {'PASA ' if cond else 'FALLA'}  {que}")
    if not cond:
        FALLOS.append(que)


# ═══ EL 631 SINTÉTICO, CON LA FORMA DEL REAL ═════════════════════════════════
QUEJOSA = "Unión Ejemplo, A.C."
RECURRENTE = "Inmobiliaria Ejemplo, S.A. de C.V."
MAGISTRADA = ("MAGISTRADA NOMBRE DE PRUEBA, INTEGRANTE DE LA SALA CIVIL UNO DEL TRIBUNAL SUPERIOR "
              "DE JUSTICIA DEL ESTADO DE EJEMPLO")
RESOL = ("La Justicia de la Unión ampara y protege a Unión Ejemplo, A.C., en contra del acto "
         "atribuido a la Sala Civil Uno del Tribunal Superior de Justicia en el Estado de Ejemplo, "
         "por los motivos expresados en el considerando séptimo de este fallo y para los efectos "
         "precisados en el último considerando.")
CITA = "lo determinado en la resolución impugnada altera la cosa juzgada de forma sustancial"
RECURRIDA = (
    "JUZGADO SÉPTIMO DE DISTRITO EN EL ESTADO DE EJEMPLO\n"
    "AMPARO INDIRECTO 950/2024\n"
    "VISTOS para resolver los autos del juicio de amparo 950/2024. "
    + "Antecedentes del juicio de arrendamiento. " * 60 +
    "CONSIDERANDO QUINTO. En consecuencia, lo conducente es sobreseer respecto del acto y "
    "autoridad responsable consistentes en la resolución interlocutoria de ocho de abril de dos mil "
    "veinticuatro, dictada por el Juzgado Quinto de Primera Instancia Civil, dentro del incidente. "
    "CONSIDERANDO SÉPTIMO. Debe estimarse que " + CITA + ", pues introduce a quien no fue parte. "
    "Resulta innecesario el estudio de los restantes conceptos de violación.\n"
    "Por lo expuesto, se R E S U E L V E: ÚNICO. " + RESOL + " Notifíquese.")
P1 = {"pregunta": "¿La sustitución de la parte actora para ejecutar la sentencia alteró la cosa juzgada?",
      "cubre": [1], "clase": "fondo", "jerarquia": "principal",
      "resolvio": "La sustitución alteró la cosa juzgada.",
      "combate": "La sustitución sólo cambió quién ejecuta."}
P2 = {"pregunta": "¿La sentencia recurrida fue congruente y exhaustiva?", "cubre": [2],
      "clase": "fondo", "jerarquia": "accesorio", "depende_de": 1,
      "resolvio": "Estimó innecesario estudiar el resto.",
      "combate": "Omitió examinar los preceptos invocados."}
PROBLEMAS = [P1, P2]
DECISIVA = ("¿Puede el tercero adquirente del inmueble objeto de un juicio sobre una acción personal "
            "sustituirse válidamente a la actora en la ejecución de la sentencia?")
FIGURA = "SUSTITUCIÓN PROCESAL DEL ADQUIRENTE EN LA EJECUCIÓN"


def resultado():
    f = f123.Fases123(resumen_acto="El Juzgado de Distrito sobreseyó respecto de la interlocutoria "
                                   "y concedió el amparo contra la resolución de la Sala.",
                      resumen_conceptos="La tercera sostiene la causahabiencia del adquirente.",
                      problemas=[dict(p) for p in PROBLEMAS],
                      fuentes=[RECURRIDA, "AGRAVIOS. PRIMERO. La sustitución no alteró la cosa juzgada."])
    f.resolutivo_recurrida = RESOL
    f.resolvio_a_quo = "concede"
    e = ra.Encargo(numero="631/2025", encabezado="AMPARO EN REVISIÓN 631/2025",
                   quejoso=RECURRENTE, magistrado="M", secretario="S",
                   notificacion=_dt.date(2025, 4, 25), presentacion=_dt.date(2025, 5, 15),
                   tipo_asunto="amparo_revision", responsable=MAGISTRADA)
    e.es_recurso = True
    e.materia = "civil"
    return types.SimpleNamespace(encargo=e, fases=f, partes=None, avisos=[])


class _R:
    def __init__(self, txt, prompt):
        self.choices = [types.SimpleNamespace(message=types.SimpleNamespace(content=txt))]
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
        self.llamadas.append(p)
        return _R(self.contestar(p), p)


def contestar(p):
    if p.startswith("TAREA: LA PREGUNTA DECISIVA"):
        return json.dumps({
            "figura": FIGURA, "pregunta_decisiva": DECISIVA,
            "proposicion_toral": {"dice": "La sustitución alteró la cosa juzgada.", "cita": CITA},
            "hechos_que_deciden": ["El juicio natural versó sobre la rescisión de un arrendamiento."],
            "busquedas": ["SUSTITUCIÓN PROCESAL. ADQUIRENTE DEL INMUEBLE EN LA EJECUCIÓN"],
            "interpretacion_conforme": None})
    return json.dumps({"global": {"alcanza": False}, "propuestas": []})


r = resultado()
FICHA = fp.de_resultado(r)
BLOQUE = fp.bloque(FICHA)

# ═══════════════════════════════════════════════════════════════════════════
print("\n0 · LA FICHA DEL 631 CON LA FORMA DEL REAL")
ok([x["autoridad"] for x in FICHA["responsables"]][:1]
   == ["Sala Civil Uno del Tribunal Superior de Justicia en el Estado de Ejemplo"]
   and not any("MAGISTRADA" in x["autoridad"] for x in FICHA["responsables"]),
   "«MAGISTRADA …, INTEGRANTE DE LA SALA …» y «la Sala … en el Estado …» son UNA responsable")
ok(any(x["autoridad"] == "Juzgado Quinto de Primera Instancia Civil"
       and x["acto"].startswith("la resolución interlocutoria") for x in FICHA["responsables"]),
   "la autoridad del acto sobreseído en los considerandos también figura, con su acto")
ok(FICHA["resolvio"].get("sobresee_en_considerandos", "").startswith("la resolución interlocutoria")
   and "además, en sus considerandos" in BLOQUE,
   "lo que resolvió: el ÚNICO punto (concesión) y, aparte, el sobreseimiento de los considerandos")
ok(FICHA["firme"] and FICHA["firme"][0].startswith("sobreseimiento")
   and "interlocutoria" in FICHA["firme"][0],
   "y lo firme dice de qué acto es el sobreseimiento")
ok(FICHA["recurrente"]["papel"] == "tercero" and FICHA["quejosa"]["nombre"] == QUEJOSA
   and FICHA["organo_recurrido"]["nombre"].startswith("Juzgado Séptimo de Distrito"),
   "quejosa del resolutivo, recurre la tercera, recurrida del Juzgado de Distrito")

# ═══════════════════════════════════════════════════════════════════════════
print("\n1 · LA PREGUNTA DECISIVA SE FORMULA CON LA FICHA DELANTE")
ent = pd.entradas_de(r)
ok(ent.get("ficha") == BLOQUE, "entradas_de trae la MISMA ficha que arma ficha_procesal")
cli = Falso(contestar)
DOC = asyncio.run(pd.formular(cli, **ent))
_pa = [p for p in cli.llamadas if p.startswith("TAREA: LA PREGUNTA DECISIVA")]
ok(len(cli.llamadas) == 1 and len(_pa) == 1, "una sola llamada (la ETAPA A)")
ok(BLOQUE.strip() in _pa[0], "el prompt de la pregunta decisiva lleva la ficha procesal")
ok(pd.util(DOC) and DOC["pregunta_decisiva"] == DECISIVA and DOC["numero"] == 1
   and DOC["pregunta_recurrida"] == P1["pregunta"]
   and DOC["proposicion_toral"]["verificada"] is True,
   "la decisiva del principal, con la de la recurrida aparte y la cita verificada contra el acto")
_sin = dict(ent, ficha="")
cli0 = Falso(contestar)
asyncio.run(pd.formular(cli0, **_sin))
ok("LA FICHA PROCESAL" not in cli0.llamadas[0], "sin ficha, el prompt de la pregunta como antes")

# ═══════════════════════════════════════════════════════════════════════════
print("\n2 · LA PROPUESTA: LA FICHA Y LA CUESTIÓN DECISIVA, JUNTAS")
m = f6.Material(tipo_asunto="amparo_revision", materia="civil")
m.decisiva = DOC
cp = Falso(contestar)
asyncio.run(f5.proponer(cp, PROBLEMAS, m, r.fases.resumen_acto, r.fases.resumen_conceptos, True,
                        contraste_previo=[], ficha=BLOQUE))
_pp = cp.llamadas[-1]
ok(_pp.count("LA FICHA PROCESAL DEL ASUNTO") == 1, "la ficha, una vez")
ok(_pp.count("LA CUESTIÓN DECISIVA DEL PROBLEMA 1") == 1 and f"LA CUESTIÓN QUE DECIDE: {DECISIVA}" in _pp,
   "la cuestión decisiva del principal, una vez")
ok(_pp.index("LA FICHA PROCESAL DEL ASUNTO") < _pp.index("LOS PROBLEMAS JURÍDICOS")
   < _pp.index("LA CUESTIÓN DECISIVA DEL PROBLEMA 1") < _pp.index("LO QUE RESOLVIÓ"),
   "en orden: quién es quién, los problemas, la cuestión que decide, lo resuelto")
ok(f"Recurrente: {RECURRENTE} (parte tercera interesada)" in _pp
   and "Órgano que dictó la resolución recurrida: Juzgado Séptimo de Distrito" in _pp,
   "la ficha de la propuesta dice quién recurre y qué órgano dictó la recurrida")

# ═══════════════════════════════════════════════════════════════════════════
print("\n3 · LA TARJETA: pregunta_recurrida Y ficha, SIN CONTRADECIRSE")
_resp = {"formato": 2, "global": {"alcanza": True, "sentido": "fundado", "razon": "Prospera.",
                                   "apoyos": []},
         "propuestas": [{"problema": P1["pregunta"], "sentido": "fundado", "razon": "r", "alcanza": True},
                        {"problema": P2["pregunta"], "sentido": "innecesario", "razon": "r",
                         "alcanza": True}],
         "contraste": []}
tj = td.armar(_resp, {"tesis": [], "espejo": [], "decisiva": DOC}, PROBLEMAS,
              rama_info={"tipo_asunto": "amparo_revision", "ficha": fp.para_tarjeta(FICHA)})
pr_ = tj["principal"]
ok(pr_["pregunta"] == DECISIVA and pr_["pregunta_recurrida"] == P1["pregunta"]
   and pr_["figura"] == FIGURA,
   "principal.pregunta = la decisiva; pregunta_recurrida y figura aparte")
fi = tj.get("ficha") or {}
ok(set(fi) >= {"tipo", "quejosa", "responsables", "terceros", "recurrida", "recurrente", "materia",
               "avisos", "linea"}, "la ficha con la forma FICHA del contrato (y su línea)")
ok(fi["recurrente"] == {"quien": RECURRENTE, "caracter": "tercera interesada"}
   and fi["quejosa"] == QUEJOSA and fi["quejosa"] != fi["recurrente"]["quien"],
   "quejosa y recurrente son personas distintas, con el carácter de la recurrente")
ok(fi["recurrida"]["organo"].startswith("Juzgado Séptimo de Distrito")
   and {x["sentido"] for x in fi["recurrida"]["resolvio"]} == {"concede", "sobresee"},
   "la recurrida: el Juzgado de Distrito, que concedió y sobreseyó")
ok(len(fi["responsables"]) == 2 and not any("MAGISTRADA" in x["autoridad"] for x in fi["responsables"]),
   "dos responsables (la Sala y el Juzgado Quinto), ninguna repetida por su titular")
ok("materia: concesión" in fi["linea"] and "firme: sobreseimiento" in fi["linea"]
   and "art. 93, fr. VI" in fi["linea"] and "(concesión y sobreseimiento)" in fi["linea"],
   "la línea: lo que resolvió, qué es materia, qué quedó firme y la fracción que rige")
ok(json.loads(json.dumps(tj, ensure_ascii=False)) == tj, "la tarjeta es JSON puro")
tj0 = td.armar(_resp, {"tesis": [], "espejo": []}, PROBLEMAS, rama_info={"tipo_asunto": "amparo_revision"})
ok(tj0.get("ficha") is None and tj0["principal"]["pregunta"] == P1["pregunta"]
   and tj0["principal"]["pregunta_recurrida"] is None,
   "sin ficha ni decisiva, la tarjeta de antes")

# ═══════════════════════════════════════════════════════════════════════════
print("\n4 · EL ESTUDIO v4: LA FICHA EN EL ENCABEZADO, LA DECISIVA EN EL CRITERIO Y EN EL GUION")
me = f6.Material(tipo_asunto="amparo_revision", materia="civil", formato="estandar",
                 n_planteamientos=2, variante="v4",
                 inventario=[{"id": "A1.a", "concepto": 1, "parrafo": 0, "texto": "t", "cita": "",
                              "anclas": []}])
ra._formato_al_material(r, me)
ok(me.ficha_procesal == BLOQUE,
   "_formato_al_material pone en el material la MISMA ficha que vio la pregunta y la propuesta")
me.variante = "v4"
me.decisiva = DOC
crit = [f6.Criterio(problema=P1["pregunta"], sentido="fundado", razonamiento="El adquirente puede.",
                    jerarquia="principal"),
        f6.Criterio(problema=P2["pregunta"], sentido="innecesario", razonamiento="Sin materia.",
                    jerarquia="accesorio")]
guion = pe.vista({"problemas": [{"id": "P1", "sentido": "fundado"}, {"id": "P2", "sentido": "innecesario"}]},
                 "estandar", decisiva=pd.de_material(me, PROBLEMAS))
p4 = f6.prompt_estudio(r.fases.resumen_acto, r.fases.resumen_conceptos, crit, me, es_recurso=True,
                       rama="revoca_fondo_niega", guion=guion)
ok(p4.count("LA FICHA PROCESAL DEL ASUNTO") == 1, "la ficha, una vez")
ok(p4.count(f"   LA CUESTIÓN DECISIVA: {DECISIVA}") == 1, "la decisiva en el criterio, una vez")
ok(p4.count("LA CUESTIÓN DECISIVA (problema 1)") == 1, "y en el guion, una vez")
_ic = p4.index("[PRINCIPAL]")
ok(p4.index("LA FICHA PROCESAL DEL ASUNTO") < _ic < p4.index("   LA CUESTIÓN DECISIVA:")
   < p4.index("[ACCESORIO]"),
   "la ficha antes del criterio; la decisiva bajo el principal y no bajo el accesorio")
ok("QUEJOSA Y RECURRENTE" not in p4.upper()
   and not re.search(r"(?i)órgano recurrido[^\n]{0,40}magistrad", p4)
   and "Autoridad responsable: MAGISTRADA" not in p4,
   "ni «QUEJOSA Y RECURRENTE», ni la Magistrada como órgano recurrido, ni como segunda responsable")

# ═══════════════════════════════════════════════════════════════════════════
print("\n5 · EL SERVIDOR ARMA LAS DOS DE LA MISMA SESIÓN (por el fuente)")
_main = open(os.path.join(AQUI, "main.py"), encoding="utf8").read()
_pdsrc = open(os.path.join(AQUI, "pregunta_decisiva.py"), encoding="utf8").read()
ok("_fp_b.bloque(_taller_ficha(r, declarado))" in _main and "_fp_t.de_resultado(r, declarado=declarado)" in _main,
   "main: la ficha de los prompts es `ficha_procesal.bloque(de_resultado(r))`")
ok("_fp.bloque(_fp.de_resultado(r))" in _pdsrc and "ficha=ficha)" in _pdsrc,
   "pregunta_decisiva: la misma construcción, y la pasa a la ETAPA A")
ok('rama_info["ficha"] = _fp_tj.para_tarjeta(' in _main,
   "la tarjeta recibe la ficha en la forma del contrato")

print()
if FALLOS:
    print(f"FALLAN {len(FALLOS)}: " + " · ".join(FALLOS))
    sys.exit(1)
print("RESULTADO: TODAS LAS COMPROBACIONES PASAN")
