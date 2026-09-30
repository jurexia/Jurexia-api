# -*- coding: utf-8 -*-
"""El origen del acto en amparo directo y la sentencia dictada en cumplimiento (30-sep-2026).

Sin red: el modelo es un doble. Las frases de detección son de versiones públicas
del 3er TCC XXII (anonimizadas) y de la trampa medida en su corpus.

    .venv/bin/python test_cumplimiento_ejecutoria.py
"""
import asyncio, json, sys
from pathlib import Path
sys.path.insert(0, ".")
import contexto_taller as ct
import cumplimiento_ejecutoria as ce
import origen_acto as oa
import tipos_asunto as ta

FALLOS = []


def ok(c, q):
    print(f"   {'PASA ' if c else 'FALLA'}  {q}")
    if not c:
        FALLOS.append(q)


JUZGADO = "Juzgado Segundo de Primera Instancia Especializado en Oralidad Mercantil del Distrito Judicial de Querétaro"
EFECTOS = ("a) Deje insubsistente la resolución reclamada. b) Emita otra en la que, a la luz de la correcta aplicación "
           "del artículo 71 de la Ley Federal de Protección al Consumidor, resuelva por orden prioritario la acción "
           "reconvencional de cumplimiento de contrato. c) Con plenitud de jurisdicción, se pronuncie sobre la totalidad "
           "de las pretensiones y excepciones tanto en lo principal, como en lo reconvencional.")
ACTO = ("En cumplimiento a la ejecutoria dictada en el amparo directo civil 590/2023, en la que se concedió el amparo "
        "para el efecto de que la autoridad responsable " + EFECTOS)

print("\n1 · LA CLASE DEL ÓRGANO Y LA INSTANCIA (medidas en los 4,697 AD del 3er TCC)")
casos = [
    (JUZGADO, "juez", "unica"),
    ("Jueza Primero de Primera Instancia Civil del Distrito Judicial de San Juan del Río, Querétaro", "jueza", "unica"),
    ("Juez Tercero de Distrito en el Estado de Querétaro", "juez", "unica"),
    ("Juzgado Primero Administrativo del Tribunal de Justicia Administrativa del Estado de Querétaro", "juez", "unica"),
    ("Primera Sala Civil del Tribunal Superior de Justicia del Estado de Querétaro", "sala_alzada", "alzada"),
    ("Sala Superior del Tribunal de Justicia Administrativa del Estado de Querétaro", "sala_alzada", "alzada"),
    ("Segunda Sección de la Sala Superior del Tribunal de Justicia Administrativa del Estado de Querétaro", "sala_alzada", "alzada"),
    ("Sala Regional del Centro II del Tribunal Federal de Justicia Administrativa", "sala_tfja", "unica"),
    ("Sala Superior del Tribunal Federal de Justicia Administrativa", "sala_tfja", "unica"),
    ("Primer Tribunal Unitario del Vigésimo Segundo Circuito", "tribunal_alzada", "alzada"),
    ("Tribunal Unitario Agrario del Distrito 42", "tribunal_agrario", "unica"),
    ("Junta Local de Conciliación y Arbitraje del Estado de Querétaro", "junta", "unica"),
    ("", "", ""),
]
for nombre, clase, inst in casos:
    c = oa.clase_de_organo(nombre)
    ok(c == clase and oa.instancia_de(c) == inst, f"«{nombre[:55]}» → {clase or 'hueco'} / {inst or 'no consta'} (dio {c}/{oa.instancia_de(c)})")
ok(oa.instancia_de("", "Inconforme, interpuso recurso de apelación, que se resolvió en el toca civil 12/2020") == "alzada",
   "sin clase, el texto: una apelación resuelta → alzada")
ok(oa.instancia_de("", "sentencia definitiva dictada en el juicio oral mercantil") == "unica", "sin clase: oral mercantil → única")
ok(oa.instancia_de("juez", "Inconforme, interpuso recurso de apelación") == "unica",
   "la clase manda: si lo reclamado lo dictó un juez, no hubo alzada aunque se cuente una apelación de otra etapa")
o = oa.origen(JUZGADO, "", "", "")
ok(o["sujetos"][0] == "el Juez" and o["sujetos"][1] == "el Juez responsable", "al juez se le nombra «el Juez responsable»")
ok(oa.origen("", "", "", "")["sujetos"][0] == "la autoridad responsable", "sin responsable: fórmula neutra, nunca «la Sala»")
ok(oa.origen(JUZGADO, "", "", "", manual={"instancia": "no_consta"})["instancia"] == "",
   "«no consta» del secretario gana a la lectura (antes volvía a «única»)")
_sala_o = "Primera Sala Civil del Tribunal Superior de Justicia del Estado de Querétaro"
_o_m = oa.origen(_sala_o, "", "", "", manual={"instancia": "unica"})
ok(_o_m["instancia"] == "unica" and "apelaci" not in _o_m["lo_resuelto"],
   "si él corrige a única, lo resuelto deja de ser «ese recurso de apelación»")

# LO QUE ENCONTRÓ LA REVISIÓN ADVERSARIAL (30-sep-2026)
for nombre, clase in (
        ("Primera Sala Regional Civil de Toluca del Tribunal Superior de Justicia del Estado de México", "sala_alzada"),
        ("Sala Regional Colegiada Civil Zona 01 Tuxtla del Tribunal Superior de Justicia del Estado de Chiapas", "sala_alzada"),
        ("Primera Sala Regional Civil de Toluca", "sala_alzada"),
        ("Sala Especializada en Materia Familiar del Tribunal Superior de Justicia", "sala_alzada"),
        ("Primera Sección de la Sala Superior del Tribunal Federal de Justicia Administrativa", "sala_tfja"),
        ("Sala Especializada en Materia de Comercio Exterior del Tribunal Federal de Justicia Administrativa", "sala_tfja"),
        ("La Juez Séptimo de Distrito de Amparo y Juicios Federales en el Estado de Querétaro", "jueza"),
        ("Titular del Juzgado Segundo Civil, Jueza María Pérez", "jueza")):
    ok(oa.clase_de_organo(nombre) == clase, f"«{nombre[:60]}» → {clase} (dio {oa.clase_de_organo(nombre)})")
ok(oa.origen("Tribunal de Justicia Administrativa del Estado de Querétaro",
             "La actora promovió juicio contencioso administrativo. El Juez Primero Administrativo dictó sentencia; "
             "inconforme, la autoridad interpuso recurso de revisión, del que conoció la Primera Sección de la Sala "
             "Superior, que en la sentencia reclamada confirmó.")["instancia"] == "alzada",
   "la revisión ante la Sala Superior del tribunal local es alzada, aunque sea juicio contencioso")

print("\n2 · LA SENTENCIA DICTADA EN CUMPLIMIENTO: lo que es y lo que no")
si = [
    "se dejó insubsistente la sentencia definitiva, en cumplimiento de la ejecutoria dictada en el juicio de amparo directo civil 590/2023",
    "37. Sentencia reclamada. En cumplimiento al fallo protector, la Sala responsable dictó la resolución de quince de enero de dos mil veinticinco",
    ("el cual se resolvió en sesión de cinco de noviembre de dos mil veintiuno, en el sentido de conceder el amparo "
     "solicitado. En cumplimiento a esa ejecutoria la Sala responsable dictó la resolución de doce de enero de dos mil "
     "veintidós, la cual constituye el acto reclamado"),                       # AD 106/2022
    "La sentencia reclamada fue dictada en cumplimiento al fallo protector emitido en el amparo directo 12/2021",
]
no = [
    "se reclama cualquier acto tendiente a dar cumplimiento a la ejecutoria de fecha diez de mayo dictada dentro de los autos del toca civil 123/2020",
    "la quejosa aduce que la sentencia reclamada fue dictada en cumplimiento de la ejecutoria de amparo",
    "en cumplimiento al Acuerdo General 12/2020 del Pleno",
    "sin más efecto que obligar a la responsable a no aplicar la norma general relativa en el nuevo acto que emita en cumplimiento a la ejecutoria de amparo.",
    "podrá exponerlos para controvertir la resolución de la sala dictada en cumplimiento de la sentencia de amparo, de estimar que le causa perjuicio",
    "condenó al cumplimiento del contrato en cumplimiento de la sentencia de primera instancia",
]
si += [
    ("VISTOS para resolver en cumplimiento a la ejecutoria dictada en el amparo directo civil 590/2023 del índice del "
     "Tercer Tribunal Colegiado en Materias Administrativa y Civil del Vigésimo Segundo Circuito, los autos del juicio "
     "oral mercantil 800/2021"),
    "6. Cumplimiento de concesión de amparo. En cumplimiento a la sentencia emitida dentro del juicio de amparo 174/2023, el Juez Segundo dejó sin efectos la sentencia",
    "y para dar cumplimiento a la ejecutoria del amparo directo 55/2022, la Sala dictó nueva sentencia",
]
no += [
    ("En cumplimiento a la ejecutoria dictada por la Primera Sección de la Sala Superior en el recurso de revisión "
     "45/2021, el Juez Primero Administrativo dictó nueva sentencia, en la que reiteró la nulidad de la negativa de "
     "renovación de la concesión de taxi."),
]
for s in si:
    ok(oa.cumplimiento_de(s)["consta"], f"SÍ: «{s[:70]}…»")
for s in no:
    ok(not oa.cumplimiento_de(s)["consta"], f"NO: «{s[:70]}…»")
c = oa.cumplimiento_de("", "", ACTO)
ok(c["ejecutoria"] == "amparo directo civil 590/2023" and c["efectos"].startswith("para el efecto de que"),
   f"la ejecutoria y sus efectos transcritos ({c['ejecutoria']})")
_cadena = ("En cumplimiento a la ejecutoria dictada en el amparo directo civil 100/2019, la Sala dictó resolución, en la "
           "que se concedió el amparo para el efecto de que valorara la prueba pericial. Inconforme, la actora promovió el "
           "amparo directo civil 250/2020, que se concedió para el efecto de que dictara otra. En cumplimiento a la "
           "ejecutoria dictada en el amparo directo civil 250/2020, la Sala dictó la sentencia reclamada.")
ok(oa.cumplimiento_de(_cadena)["ejecutoria"] == "amparo directo civil 250/2020",
   "en una cadena de amparos, la ejecutoria más reciente (antes, la primera)")
ok(oa.cumplimiento_de("En cumplimiento a esa ejecutoria, la Sala dictó nueva sentencia. Se promovió el presente "
                      "juicio de amparo directo civil 323/2025 en su contra.", excluir="323/2025")["ejecutoria"] == "",
   "el número del propio amparo no se toma por la ejecutoria")
_tram = oa.cumplimiento_de("SENTENCIA DEFINITIVA en cumplimiento a la ejecutoria dictada en el amparo directo civil "
                           "590/2023, se dicta nueva sentencia. RESULTANDOS. 1. La actora demandó la rescisión, y se "
                           "tuvo por presentada para los efectos legales a que haya lugar.")
ok(_tram["consta"] and _tram["efectos"] == "", "«para los efectos legales a que haya lugar» no son los efectos del amparo")
o = oa.origen(JUZGADO, "", "", ACTO, manual={"instancia": "alzada", "sobreseer": True, "efectos": "EFECTOS PEGADOS"})
ok(o["instancia"] == "alzada" and o["sobreseer_confirmado"] and o["cumplimiento"]["efectos"] == "EFECTOS PEGADOS"
   and o["fuente"] == "secretario", "lo que corrige el secretario manda sobre lo leído")

print("\n3 · LAS BANDERAS Y EL CONTEXTO")
ct.poner(False, pruebas=False)
ct.poner_origen(oa.origen(JUZGADO, "", "", ACTO))
ok(ta.sujetos_de("amparo_directo")["organo"][0] == "la Sala" and ta.cumplimiento_actual() == {}
   and ce.bloque_contexto() == "" and ta.tecnica_de("amparo_directo") == [],
   "fuera de las cuentas de prueba, todo como antes (bandera «casa»)")
ct.poner(True, pruebas=True)
ct.poner_origen(oa.origen(JUZGADO, "", "", ACTO))
ok(ta.sujetos_de("amparo_directo")["organo"][0] == "el Juez" and ta.instancia_actual() == "unica",
   "con la bandera: «el Juez» y única instancia")
ok(ta.cumplimiento_actual().get("ejecutoria") == "amparo directo civil 590/2023", "y el cumplimiento consta")
tec = ta.tecnica_de("amparo_directo")
ok(len(tec) == 1 and tec[0] is ta.TECNICA_CUMPLIMIENTO and tec[0]["solo_si_estan"]
   and "2001857" in tec[0]["apoyos"], "la técnica del cumplimiento entra, con 2a./J. 113/2012 entre sus apoyos")
ok(ta.sujetos_de("queja")["organo"][0] == "el Juzgado de Distrito", "los recursos no cambian")

print("\n4 · LA CLASIFICACIÓN")
PROBLEMAS = [
    {"pregunta": "¿El artículo 71 de la Ley Federal de Protección al Consumidor podía desplazar la cláusula de rescisión?",
     "resolvio": "aplicó el 71", "combate": "que el 71 no desplaza el pacto"},
    {"pregunta": "¿La sentencia debía incluir intereses moratorios en la liquidación del saldo?",
     "resolvio": "redujo los moratorios", "combate": "que debió aplicarlos"},
    {"pregunta": "¿Es inconstitucional el artículo 71 de la Ley Federal de Protección al Consumidor?",
     "resolvio": "lo aplicó", "combate": "viola la libertad contractual"},
    {"pregunta": "¿La responsable se excedió al cumplir la ejecutoria?", "resolvio": "", "combate": "exceso"},
]


class _Msg:
    def __init__(self, c):
        self.content = c


class Modelo:
    def __init__(self, respuesta):
        self.llamadas = []
        yo = self

        class _Comp:
            async def create(self_, **kw):
                yo.llamadas.append(kw)
                return type("R", (), {"choices": [type("C", (), {"message": _Msg(respuesta)})()], "usage": None})()
        self.chat = type("Ch", (), {"completions": _Comp()})()


resp = json.dumps({"problemas": [
    {"n": 1, "vinculacion": "vinculado", "efecto": "inciso b)", "por_que": "La ejecutoria ordenó aplicar el 71."},
    {"n": 2, "vinculacion": "libre", "efecto": "inciso c)", "por_que": "Los intereses quedaron con plenitud."},
    {"n": 3, "vinculacion": "constitucionalidad", "efecto": "inciso b)", "por_que": "Ataca la norma que se mandó aplicar."},
    {"n": 4, "vinculacion": "EXCESO_DEFECTO", "por_que": "Alega exceso."}],
    "hay_libertad": True, "resumen": "Vinculó el 71; dejó libres las demás pretensiones."})
m = Modelo(resp)
cl = asyncio.run(ce.clasificar(m, {"ejecutoria": "amparo directo civil 590/2023", "efectos": EFECTOS}, PROBLEMAS, "resumen"))
ok([p["vinculacion"] for p in cl["problemas"]] == ["vinculado", "libre", "constitucionalidad", "exceso_defecto"],
   "la clasificación, normalizada (mayúsculas incluidas)")
ok(cl["hay_libertad"] and not cl["todo_vinculado"] and cl["huella"] == ce.huella(EFECTOS, PROBLEMAS), "libertad y huella")
ok("590/2023" in m.llamadas[0]["messages"][0]["content"] and "plenitud de jurisdicción" in m.llamadas[0]["messages"][0]["content"],
   "el prompt lleva la ejecutoria y sus efectos")
ok(asyncio.run(ce.clasificar(m, {"ejecutoria": "x", "efectos": ""}, PROBLEMAS)) is None,
   "sin efectos no se clasifica: se piden, no se adivinan")
ok(ce.normalizar({"problemas": [{"n": 1, "vinculacion": "rarísimo"}]}, PROBLEMAS[:2])["problemas"][1]["vinculacion"] == "no_consta",
   "lo que falta o no está en el catálogo, «no_consta»")
ok(ce.normalizar({}, PROBLEMAS[:1])["hay_libertad"] is True, "ante la duda sobre la libertad, NO se propone sobreseer")

print("\n5 · LA CALIFICACIÓN QUE SE SIGUE")
crit = [{"problema": p["pregunta"], "sentido": "fundado", "razonamiento": "x"} for p in PROBLEMAS]
av, det = ce.aplicar(crit, cl, "amparo directo civil 590/2023")
ok([c["sentido"] for c in crit] == ["inoperante", "fundado", "fundado", "inoperante"],
   "vinculado y exceso → inoperantes; libre y constitucionalidad intactos")
ok("2a./J. 113/2012" in crit[0]["razonamiento"] and "inciso b)" in crit[0]["razonamiento"]
   and "artículo 196" in crit[3]["razonamiento"], "cada una con su razón y su tesis")
ok(any("CRITERIO DIVIDIDO" in a and "1a. IX/2022" in a for a in av), "la inconstitucionalidad se le presenta al secretario")
crit2 = [{"problema": PROBLEMAS[0]["pregunta"], "sentido": "fundado", "razonamiento": "suyo"}]
ce.aplicar(crit2, cl, "x", tocados={PROBLEMAS[0]["pregunta"]})
ok(crit2[0]["sentido"] == "fundado", "lo que el secretario tocó no se toca")


class Prop:
    def __init__(self, problema, sentido, apoyos=None):
        self.problema, self.sentido, self.razon = problema, sentido, "r"
        self.apoyos = list(apoyos or [])


props = [Prop(PROBLEMAS[0]["pregunta"], "infundado"), Prop(PROBLEMAS[3]["pregunta"], "fundado", ["999"]),
         Prop(PROBLEMAS[1]["pregunta"], "fundado")]
ce.aplicar(props, cl, "x")
ok(props[0].sentido == "inoperante" and "cosa juzgada" in props[0].razon, "también sobre las propuestas del motor (objetos)")
ok(props[0].apoyos == ["2001857", "2018315"], "lo vinculado lleva de apoyo la 2a./J. 113/2012 y la 1a./J. 57/2018")
_cl_c = ce.normalizar({"problemas": [{"n": 1, "vinculacion": "consentido", "por_que": "Sólo lo reiteró."}],
                       "hay_libertad": True}, PROBLEMAS[:1])
_pc = [Prop(PROBLEMAS[0]["pregunta"], "fundado")]
ce.aplicar(_pc, _cl_c, "amparo directo civil 590/2023")
ok(_pc[0].sentido == "inoperante" and "consentimiento tácito" in _pc[0].razon.replace("consintió tácitamente", "consentimiento tácito")
   and "sin que ello haga improcedente" in _pc[0].razon and _pc[0].apoyos == ["2001857"],
   "lo sólo reiterado y no impugnado: inoperante por consentido, sin improcedencia (2a./J. 113/2012)")
ok(props[1].apoyos == ["999"], "los apoyos que ya traía no se pisan")
ok(props[2].apoyos == [], "lo libre no gana apoyos")
_F5 = Path("main.py").read_text(encoding="utf-8"); MP = _F5[_F5.index("_ce_pp.aplicar(propuestas"):]
ok("se propone SIN APOYO del acervo" in MP[:1600], "el «SIN APOYO» se rehace después de aplicar")

print("\n6 · EL SOBRESEIMIENTO: se propone sin libertad, se escribe si él lo confirma")
o = dict(oa.origen(JUZGADO, "", "", ACTO), clasificacion=cl)
ok(not ce.sobreseer_propuesto(o), "con libertad parcial NO se propone sobreseer (2a./J. 113/2012)")
cl_todo = ce.normalizar({"problemas": [{"n": 1, "vinculacion": "vinculado"}], "hay_libertad": False}, PROBLEMAS[:1])
o2 = dict(oa.origen(JUZGADO, "", "", ACTO), clasificacion=cl_todo)
ok(ce.sobreseer_propuesto(o2) and cl_todo["todo_vinculado"], "sin libertad alguna, se propone")
ok(not ce.sobreseer_confirmado(o2), "pero no se escribe sin su confirmación")
o3 = dict(oa.origen(JUZGADO, "", "", ACTO, manual={"sobreseer": True}), clasificacion=cl_todo)
ok(ce.sobreseer_confirmado(o3), "con su confirmación, sí")
for _v in ("no_consta", "exceso_defecto", "consentido", "inconstitucionalidad"):
    _cl_d = ce.normalizar({"problemas": [{"n": 1, "vinculacion": "vinculado"}, {"n": 2, "vinculacion": _v}],
                           "hay_libertad": False}, PROBLEMAS[:2])
    _o_d = dict(oa.origen(JUZGADO, "", "", ACTO, manual={"sobreseer": True}), clasificacion=_cl_d)
    ok(not ce.sobreseer_propuesto(_o_d) and not ce.sobreseer_confirmado(_o_d),
       f"con un «{_v}» no se propone sobreseer, y una confirmación vieja no vale")
_o_lib = dict(oa.origen(JUZGADO, "", "", ACTO, manual={"sobreseer": True}), clasificacion=cl)
ok(not ce.sobreseer_confirmado(_o_lib), "confirmado sobre una clasificación que ya encontró libertad: no sobresee")
ok("artículo 64, párrafo segundo" in ce.considerando_improcedencia(o3), "el considerando lleva la vista del art. 64")
ok("61, fracción IX" in ce.considerando_improcedencia(o3) and "590/2023" in ce.considerando_improcedencia(o3)
   and ce.IMPROCEDENTE["resolutivo"].startswith("ÚNICO. Se sobresee"), "considerando y resolutivo del sobreseimiento")
import documento_generado as dg
ct.poner_origen(o3)
ok(dg._sobresee_por_cumplimiento("amparo_directo") and not dg._sobresee_por_cumplimiento("amparo_revision"),
   "el documento sobresee sólo en amparo directo y sólo confirmado")
ct.poner_origen(o)
ok(not dg._sobresee_por_cumplimiento("amparo_directo"), "sin confirmación, el documento entra al fondo")

print("\n7 · LO QUE LEEN LA PROPUESTA, EL PLAN Y EL ESTUDIO")
b = ce.bloque_contexto(o)
ok("CUMPLIMIENTO DE LA EJECUTORIA DEL AMPARO DIRECTO CIVIL 590/2023" in b and "plenitud de jurisdicción" in b
   and "VINCULADO por la ejecutoria" in b and "LIBERTAD de jurisdicción" in b and "2a./J. 113/2012" in b,
   "el bloque: la ejecutoria, sus efectos, la clasificación y la regla")
ok("SUS EFECTOS NO CONSTAN" in ce.bloque_contexto(oa.origen(JUZGADO, "en cumplimiento al fallo protector, la Sala dictó la resolución de ayer en el amparo directo 1/2020", "", "")),
   "sin efectos: no supongas qué quedó vinculado")

print("\n8 · DÓNDE SE ENGANCHA")
F = Path("main.py").read_text(encoding="utf-8")
PN = F[F.index("async def _taller_proponer_nucleo("):]
PN = PN[:PN.index("\n@app.")] if "\n@app." in PN else PN
ok(PN.index("_taller_clasificar_cumplimiento(") < PN.index("contexto = _con_autos(r, contexto)"),
   "se clasifica ANTES de armar el contexto de la propuesta")
ok(PN.index("_ce_pp.aplicar(propuestas") < PN.index("_ad.aplicar("), "y se aplica ANTES del árbol en la propuesta")
AC = F[F.index("def _taller_armar_criterio("):]
ok("tocados=_toc_ad" in AC[AC.index("_ce_cc.aplicar(crit"):][:300], "y en el criterio, respetando lo tocado")
CA = F[F.index("def _con_autos("):]
ok("bloque_contexto()" in CA[:2500], "el bloque va en `_con_autos` (propuesta, plan, estudio, recalificación)")
ok('@app.post("/taller/origen")' in F and "_taller_guardar_origen(" in F, "la puerta del secretario para corregirlo")
PO = F[F.index('@app.post("/taller/origen")'):]
ok(PO.index("if not tocados:") < PO.index("_taller_guardar_origen("),
   "un POST sin cambios no escribe (antes dejaba sin clasificación lo ya clasificado)")
PO2 = F[F.index('@app.post("/taller/origen")'):]
ok("_efectivo(antes), _efectivo(manual)" in PO2[:6000] and 'manual.pop("sobreseer", None)' in PO2[:6000],
   "el origen compara lo EFECTIVO y la confirmación caduca con otros efectos")
ok('"origen_firma": _taller_firma_origen()' in F and "se calculó con otro origen del acto" in F,
   "la propuesta guardada se recalcula si cambió el origen («reclasificar»)")
_PN = F[F.index("async def _taller_proponer_nucleo"):]
ok(_PN.index("_ce_pp.aplicar(propuestas") < _PN.index("_dz.reconciliar(propuestas") < _PN.index("_ad.aplicar("),
   "lo vinculado se aplica ANTES de reconciliar y del árbol")
_AC = F[F.index("def _taller_armar_criterio("):]
ok(_AC.index("_ce_cc.aplicar(crit") < _AC.index("_ad.aplicar("), "y en el criterio, antes del árbol")
AD_ = F[F.index('@app.post("/taller/adelanto")'):]
ok(AD_.index("_ctx_g0.poner(") < AD_.index("await _ra.generar("),
   "el adelanto pone el contexto ANTES de generar (si no, las banderas del origen van apagadas en las fases 1-3)")
RS = F[F.index("def _taller_recuperar_sesion("):]
ok("poner_origen(_taller_origen(ses))" in RS[:1500], "cada petición recalcula el origen al recuperar la sesión")
RA = Path("redactor_adelanto.py").read_text(encoding="utf-8")
ok(RA.index("_ct_o.poner_origen(_o_pre)") < RA.index("f123.correr(cliente, texto_acto"), "el adelanto lo pone ANTES de las fases 1-3")

print()
if FALLOS:
    print("FALLAS:\n  " + "\n  ".join(FALLOS))
    sys.exit(1)
print("RESULTADO: TODAS LAS COMPROBACIONES PASAN")
