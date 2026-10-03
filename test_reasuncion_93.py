# -*- coding: utf-8 -*-
"""REVOCAR UNA CONCESIÓN ES REASUMIR JURISDICCIÓN, Y LOS DEFECTOS DE RAMA Y DE
PAPELES DEL AR 631/2025 (28-sep-2026), SIN LLAMAR A NINGÚN MODELO.

1. Artículo 93, fracción VI: el juzgado concedió, recurre la tercera
   interesada, el recurso prospera → el tribunal estudia los conceptos que el
   juzgado no estudió; el resolutivo sale de ese estudio (y va con hueco si no
   constaron); lo no impugnado queda firme y la revocación se acota a la
   materia de la revisión.
2. Ningún aviso de «se concede» en una revisión que revoca y niega.
3. Los avisos de rama del adelanto no sobreviven a los del proyecto.
4. Los papeles: la recurrente no es la quejosa.
5. «El considerando octavo» de la recurrida no es una remisión rota.

Todo es SINTÉTICO: nombres, números y fechas de prueba.

    .venv/bin/python test_reasuncion_93.py
"""
import ast
import datetime as _dt
import os
import sys
import tempfile
import types

os.environ.pop("ESTUDIO_PROMPT", None)

import calidad_estudio as ce
import dialogo_constitucional as dc
import ensamblar_adelanto as ens
import fase6_estudio as f6
import fase_admision as fa
import fase_partes as fp
import fase_rama as fr
import fase_sintesis as fs
import fases123_pipeline as f123
import plan_estudio as pe
import redactor_adelanto as ra
import tipos_asunto as ta

FALLOS = []


def ok(cond, que):
    print(f"   {'PASA ' if cond else 'FALLA'}  {que}")
    if not cond:
        FALLOS.append(que)


AQUI = os.path.dirname(os.path.abspath(__file__))

# ═══ EL ASUNTO DE PRUEBA (sintético, a la manera del 631) ════════════════════
QUEJOSA = "Unión Ejemplo, A.C."
RECURRENTE = "Inmobiliaria Ejemplo, S.A. de C.V."
RESOL = ("La Justicia de la Unión ampara y protege a Unión Ejemplo, A.C., en contra del acto "
         "atribuido a la Sala Civil Uno del Tribunal Superior de Justicia, por los motivos "
         "expresados en el considerando séptimo de esta sentencia y para los efectos precisados en "
         "el último considerando.")
RECURRIDA = (
    "JUZGADO SÉPTIMO DE DISTRITO EN EL ESTADO DE EJEMPLO\n"
    "AMPARO INDIRECTO 950/2024\n"
    "VISTOS para resolver los autos del juicio de amparo 950/2024. "
    "CONSIDERANDO QUINTO. Se sobresee respecto del acto atribuido al Juzgado Quinto Civil. "
    "CONSIDERANDO SÉPTIMO. Es fundado el segundo concepto de violación y suficiente para conceder; "
    "resulta innecesario el estudio de los restantes conceptos de violación.\n"
    "Por lo expuesto, se R E S U E L V E: PRIMERO. Se sobresee en el juicio respecto del acto del "
    "Juzgado Quinto Civil. SEGUNDO. " + RESOL + " Notifíquese.")
DEMANDA = (
    "DEMANDA DE AMPARO INDIRECTO\nQUEJOSA: Unión Ejemplo, A.C.\n"
    "CONCEPTOS DE VIOLACIÓN\n"
    "PRIMERO. La resolución reclamada viola los artículos 14 y 16 constitucionales porque altera la "
    "cosa juzgada al sustituir a la actora en la ejecución. " + "Argumento. " * 60 + "\n"
    "SEGUNDO. La interlocutoria dejó de valorar las pruebas del incidente. " + "Argumento. " * 50 + "\n"
    "TERCERO. La garantía fijada no cubre los daños. " + "Argumento. " * 40 + "\n"
    "SUSPENSIÓN\nSe solicita la suspensión del acto reclamado.")
RESULTANDOS = ("Unión Ejemplo, A.C. reclamó la resolución dictada en el toca civil 2338/2024 por la "
               "Sala Civil Uno. El Juzgado Séptimo de Distrito registró la demanda con el número "
               "950/2024 y, seguido el juicio, concedió el amparo. Inmobiliaria Ejemplo, S.A. de C.V., "
               "tercera interesada, interpuso recurso de revisión.")


def fases(**kw):
    f = f123.Fases123(resumen_acto="El Juzgado de Distrito sobreseyó respecto del Juzgado Quinto y "
                                   "concedió el amparo contra la resolución de la Sala.",
                      problemas=[{"pregunta": "¿La sustitución alteró la cosa juzgada?", "cubre": [1],
                                  "clase": "fondo", "jerarquia": "principal"}],
                      fuentes=[RECURRIDA, "AGRAVIOS. PRIMERO. La sustitución no alteró la cosa juzgada."])
    f.resolutivo_recurrida = RESOL
    f.resolvio_a_quo = "niega"             # lo que guardaba la sesión del 631 (el recuento)
    for k, v in kw.items():
        setattr(f, k, v)
    return f


def encargo(**kw):
    e = ra.Encargo(numero="631/2025", encabezado="AMPARO EN REVISIÓN 631/2025",
                   # Lo que guardó el formulario en el 631: la RECURRENTE como «quejoso».
                   quejoso=RECURRENTE, magistrado="M", secretario="S",
                   notificacion=_dt.date(2025, 4, 25), presentacion=_dt.date(2025, 5, 15),
                   tipo_asunto="amparo_revision", responsable="Magistrada de la Sala Civil Uno")
    e.es_recurso = True
    for k, v in kw.items():
        setattr(e, k, v)
    return e


def resultado(e=None, f=None):
    return types.SimpleNamespace(encargo=e or encargo(), fases=f or fases(), partes=None, avisos=[])


CRIT_F = [f6.Criterio(problema="¿La sustitución alteró la cosa juzgada?", sentido="fundado",
                      razonamiento="La interlocutoria sólo cambió quién ejecuta.", jerarquia="principal")]
CRIT_I = [f6.Criterio(problema="¿La sustitución alteró la cosa juzgada?", sentido="infundado",
                      razonamiento="r", jerarquia="principal")]

# ═══════════════════════════════════════════════════════════════════════════
print("\n1 · EL ARTÍCULO 93: CUÁNDO SE REASUME JURISDICCIÓN")
ok(ta.reasuncion("concede", "fundado") == "concesion", "concedió + prospera → fracción VI")
ok(ta.reasuncion("sobresee_concede", "esencialmente_fundado", quien_recurre="tercero") == "concesion",
   "sobreseyó y concedió + prospera (con el predicado, no con «fundad») → fracción VI")
ok(ta.reasuncion("concede", "infundado") == "" and ta.reasuncion("niega", "fundado") == "",
   "no prospera, o el juzgado negó → no hay conceptos omitidos que estudiar")
ok(ta.reasuncion("concede", "fundado", quien_recurre="quejoso") == "",
   "si recurre la propia quejosa rige la fracción V, no la VI")
ok(ta.reasuncion("concede", "fundado", violacion_procesal=True) == ""
   and ta.reasuncion("concede", "fundado", solo_efectos=True) == "",
   "reponer el procedimiento o modificar los efectos no reasumen nada")
ok(ta.reasuncion("sobresee", "fundado") == "sobreseimiento", "levantar el sobreseimiento también")
ok(ta.FUNDAMENTO_REASUNCION["sobreseimiento"] == "artículo 93, fracciones I y V, de la Ley de Amparo"
   and ta.RAMAS_REVISION["revoca_sobreseimiento_concede"]["fundamento"].endswith("fracciones I y V, de la Ley de Amparo")
   and ta.TECNICA_RESOLUCION["revision_levanta_sobreseimiento"]["fuente"].startswith("artículo 93, fracciones I y V"),
   "levantar el sobreseimiento es la fr. I (agravios) y la V (el fondo), no la I a secas")
ok(ta.rama_revision("concede", "fundado") == "revoca_fondo_niega",
   "antes de escribir, la rama sigue siendo «revoca_fondo_niega» (la que reconoce la técnica)")
ok(ta.rama_revision("concede", "fundado", sentido_amparo="concede") == "revoca_fondo_concede"
   and ta.rama_revision("concede", "fundado", sentido_amparo="niega") == "revoca_fondo_niega",
   "con el estudio escrito, el segundo punto sale de lo que concluye (concede por razón distinta o niega)")
_reglas = [x.get("fuente") for x in ta.tecnica_de("amparo_revision", "revoca_fondo_niega")]
ok("artículo 93, fracción VI, de la Ley de Amparo" in _reglas,
   "la técnica de la fracción VI llega al estudio cuando se revoca una concesión")
ok("artículo 93, fracción VI, de la Ley de Amparo" not in
   [x.get("fuente") for x in ta.tecnica_de("amparo_revision", "revoca_fondo_concede")]
   and "artículo 93, fracción VI, de la Ley de Amparo" not in
   [x.get("fuente") for x in ta.tecnica_de("amparo_revision", "confirma_concede")],
   "y no cuando se revoca una negativa ni cuando se confirma")
_r6 = ta.TECNICA_RESOLUCION["revision_reasume_concesion"]
ok(_r6["apoyos"] == ["171925", "178784", "182039", "174177"] and _r6.get("solo_si_estan")
   and "207016" not in _r6["apoyos"],
   "apoyos comprobados en el acervo (la 3a./J. 20/91, reg. 207016, no está y no se cita)")
ok(not any("«" in t for t in _r6["tecnica"]), "la técnica describe; no trae frases entre comillas para copiar")

print("\n2 · LOS PUNTOS: FIRME PRIMERO, EN LA MATERIA DE LA REVISIÓN, Y EL AMPARO DEL ESTUDIO")
_p = ta.puntos_reasuncion("niega", fr.resolutivo_recurrida("se R E S U E L V E: ÚNICO. " + RESOL + " Notifíquese."),
                          firme=True)
ok(_p[0] == "PRIMERO. Queda firme el sobreseimiento decretado en la sentencia recurrida."
   and _p[1] == "SEGUNDO. En la materia de la revisión, se revoca la sentencia recurrida."
   and _p[2].startswith("TERCERO. La Justicia de la Unión no ampara ni protege a Unión Ejemplo, A.C.,")
   and _p[2].endswith("último considerando de esta ejecutoria."),
   f"el orden del circuito (AR 105/2019, 112/2021, 168/2025): {_p}")
_ph = ta.puntos_reasuncion("", RESOL)
ok(len(_ph) == 2 and "{HUECO}" in _ph[1] and "no ampara" not in _ph[1] and "ampara y protege" not in _ph[1],
   f"sin conclusión del estudio, el verbo va en hueco: {_ph[1][:90]}")
_pc = ta.puntos_reasuncion("concede", RESOL, parcial=True)
ok(_pc[0].startswith("PRIMERO. En la materia de la revisión") and "ampara y protege a Unión Ejemplo" in _pc[1],
   "si un concepto no estudiado prospera, se concede por razón distinta a quien pidió el amparo")
ok(ta.puntos_revoca_concesion(RESOL)[1].startswith("SEGUNDO. La Justicia de la Unión no ampara ni protege a Unión"),
   "`puntos_revoca_concesion` (577c700) sigue igual")

print("\n3 · ¿HAY CONCEPTOS OMITIDOS Y LOS TENEMOS? (la función de la tarjeta)")
ok(len(fr.conceptos_en_texto(DEMANDA)) > 600 and fr.conceptos_en_texto(DEMANDA).startswith("PRIMERO.")
   and "SUSPENSIÓN" not in fr.conceptos_en_texto(DEMANDA),
   "la demanda entre las constancias trae los conceptos (se cortan en el siguiente apartado)")
ok(fr.conceptos_en_texto(RECURRIDA) == "", "la mención suelta en la recurrida no son los conceptos")
ok(fr.conceptos_en_texto("TERCERO. Conceptos de violación.\nLa quejosa expresó los conceptos que se tienen por "
                         "reproducidos en obvio de repeticiones. PRIMERO. " + "x " * 400) == "",
   "darlos por reproducidos no es transcribirlos")
_info = {"tipo_asunto": "amparo_revision", "que_hizo": "concede", "quien_recurre": "tercero"}
_co = fr.conceptos_omitidos(_info, "fundado", fases())
ok(_co and _co["hacen_falta"] is True and _co["tenemos"] is False and "93, fracción VI" in _co["por_que"]
   and _co["reasuncion"] == "concesion",
   f"el 631 sin los conceptos: hacen falta y no los tenemos: {_co}")
_co2 = fr.conceptos_omitidos(_info, "fundado", fases(autos=DEMANDA))
ok(_co2["tenemos"] is True and _co2["donde"] == "constancias"
   and "comprueba que estén completos" in _co2["por_que"],
   "con la demanda entre las constancias, se toman de ahí y se dice que se comprueben")
ok(fr.conceptos_omitidos(dict(_info, conceptos_violacion="Mis conceptos"), "fundado", fases(autos=DEMANDA))["donde"]
   == "secretario", "lo que aporta el secretario manda")
ok(fr.conceptos_omitidos(_info, "infundado", fases()) is None
   and fr.conceptos_omitidos(dict(_info, quien_recurre="quejoso"), "fundado") is None
   and fr.conceptos_omitidos(dict(_info, tipo_asunto="amparo_directo"), "fundado") is None,
   "no aplica si no prospera, si recurre la quejosa o fuera de la revisión")
ok(fr.conceptos_omitidos("sobresee", "fundado")["reasuncion"] == "sobreseimiento",
   "también al levantar un sobreseimiento, con un str por rama_info")
ok(fr.sobreseyo_ademas(fases()) and not fr.sobreseyo_ademas(fases(fuentes=["", ""], resumen_acto="Concedió.")),
   "la recurrida también sobreseyó (su sección resolutiva lo dice), aunque el punto reproducible sea uno")
_r = resultado()
_i = ra.info_de_rama(_r)
ok(_i["que_hizo"] == "concede" and _i["quien_recurre"] == "tercero" and _i["sobresee_ademas"],
   f"`info_de_rama` desde una sesión: el resolutivo manda (no el «niega» guardado) y recurre la tercera: {_i}")

print("\n4 · EL ESTUDIO: EL CONSIDERANDO DE LOS CONCEPTOS NO ESTUDIADOS")
_b = f6._bloque_conceptos("revoca_fondo_niega", "", reasuncion={"reasuncion": "concesion", "sobresee_ademas": True})
ok("FALTAN ESOS CONCEPTOS" in _b and "NO concluyas si se concede o se niega" in _b and "ADVERTENCIAS" in _b,
   "sin conceptos: no concluye el sentido del amparo y lo dice en ADVERTENCIAS")
ok("sobreseyó respecto de algún acto" in _b and "queda firme" in _b, "y sabe que la recurrida también sobreseyó")
_b2 = f6._bloque_conceptos("revoca_fondo_niega", "", reasuncion={"reasuncion": "concesion", "donde": "constancias",
                                                                  "conceptos": "PRIMERO. Concepto."})
ok("ABRE UN CONSIDERANDO PROPIO" in _b2 and "PRIMERO. Concepto." in _b2 and "demanda de amparo que obra" in _b2
   and "por las mismas" in _b2 and "inoperantes por derivar" in _b2.replace("\n", " ").replace("  ", " ")
   and "si procede conceder o\n  negar el amparo" in _b2,
   "con los del material: considerando propio, la misma dependencia y la conclusión expresa")
ok(f6._bloque_conceptos("revoca_fondo_niega", "x", reasuncion={"reasuncion": ""}) == "",
   "si no aplica (recurre la quejosa), nada")
ok("REVOCAR NO ES NEGAR" in f6._bloque_conceptos("revoca_fondo_niega", "PRIMERO. c"),
   "sin cálculo previo, decide la rama")
ok("con la MISMA técnica de los cuatro pasos\nque usaste con los agravios."
   in f6._bloque_conceptos("revoca_sobreseimiento", "Primer concepto: x " * 30)
   and "tal como constan en la demanda" in f6._bloque_conceptos(
       "revoca_sobreseimiento_concede", "", reasuncion={"reasuncion": "sobreseimiento", "donde": "constancias",
                                                        "conceptos": "PRIMERO. c"}),
   "el levantamiento del sobreseimiento sigue igual y también toma los conceptos del material")
_mt = f6.Material(tipo_asunto="amparo_revision", tesis=[{"registro": "171925", "rubro": "R"}])
_bt = f6._bloque_tecnica("amparo_revision", "revoca_fondo_niega", False, _mt)
ok("APOYOS PARA ESTA TÉCNICA: entre las tesis del acervo que tienes abajo están las de registro 171925." in _bt,
   "sólo se anuncian los apoyos que llegaron al material")
ok("171925" not in f6._bloque_tecnica("amparo_revision", "revoca_fondo_niega", False, f6.Material()),
   "si el acervo no los devolvió, no se anuncian")

print("\n5 · NINGÚN «SE CONCEDE» EN UNA REVISIÓN QUE REVOCA Y NIEGA")
EST_NIEGA = ("Es fundado el agravio. Los efectos de la concesión recurrida no subsisten. Revocada la "
             "sentencia, procede revocar la sentencia recurrida y negar el amparo.")
ok(f6._efectos_de_reposicion(EST_NIEGA, CRIT_F, True, rama="revoca_fondo_niega") == ""
   and f6._efectos_de_reposicion(EST_NIEGA, CRIT_F, True) != "",
   "«EFECTOS INCOMPLETOS PARA UNA VIOLACIÓN PROCESAL»: no con la rama que niega (sin rama, como antes)")
ok(f6._cierre_operativo(EST_NIEGA, CRIT_F, "revoca_fondo_niega") == ""
   and f6._cierre_operativo(EST_NIEGA, CRIT_F) != "",
   "«Se concede y los efectos van en prosa»: no con la rama que niega")
EST_CONCEDE = "Es fundado el tercer concepto no estudiado; procede conceder el amparo. Efectos: dejar insubsistente."
ok(f6._cierre_operativo(EST_CONCEDE, CRIT_F, "revoca_fondo_niega") != "",
   "si el estudio de los conceptos concede por razón distinta, el aviso vuelve (hay efectos)")
ok(ens.formula_resolutivo(["fundado", "infundado"], rama="revoca_fondo_niega") == ("no ampara ni protege", "")
   and ens.formula_resolutivo(["fundado", "infundado"])[1].startswith("El resolutivo concede"),
   "«El resolutivo concede porque hay conceptos fundados…»: no con la rama que niega")
ok(ta.ejecutoria_concede("revoca_fondo_niega") is False and ta.ejecutoria_concede("revoca_fondo_niega", "concede")
   and ta.ejecutoria_concede("confirma_concede") is False and ta.ejecutoria_concede("sin_determinar") is None,
   "un solo predicado: ¿esta ejecutoria concede?")
ok(pe.concede_de(resultado(), CRIT_F) is False, "el guion tampoco habla de las unidades de las que salen efectos")
_src_ra = open(os.path.join(AQUI, "redactor_adelanto.py"), encoding="utf-8").read()
ok(_src_ra.count("rama=_rama_de(r, criterios, estudio))") == 2 and "sentido_amparo=__import__(\"fase_rama\")" in _src_ra
   and "rama=_rama_t," in _src_ra,
   "los dos gemelos y `_terminar` pasan la rama que ya leyó el estudio")
ok(ra._rama_de(resultado(), CRIT_F) == "revoca_fondo_niega"
   and ra._rama_de(resultado(), CRIT_F, EST_CONCEDE) == "revoca_fondo_concede"
   and ra._rama_de(resultado(), CRIT_I) == "confirma_concede",
   "una sola fuente de verdad para la rama: el resolutivo del juzgado, el sentido y lo que concluyó el estudio")

print("\n6 · LOS AVISOS DE RAMA DEL ADELANTO NO SOBREVIVEN A LOS DEL PROYECTO")
_viejos = ["RESOLUTIVO DE REVISIÓN, rama «confirma_sobresee» (artículo 93). El a quo sobresee; el recurso "
           "resultó infundado.", "EL SENTIDO DE LA SENTENCIA RECURRIDA NO SE PUDO LEER DEL PDF y se dedujo.",
           "Otro aviso del adelanto que se queda."]
_nuevos = ["RESOLUTIVO DE REVISIÓN, rama «revoca_fondo_niega» (artículo 93). El a quo concede; el recurso "
           "resultó fundado."]
_fin = ra._avisos_de_rama_al_dia(_viejos + _nuevos, _nuevos)
ok(_fin == ["Otro aviso del adelanto que se queda.", _nuevos[0]],
   f"se queda la rama del proyecto y lo que no es de rama: {_fin}")
ok(ra._avisos_de_rama_al_dia(_viejos, ["nada de rama"]) == _viejos,
   "si el proyecto no dijo su rama, no se quita nada")

print("\n7 · LOS PAPELES: LA RECURRENTE NO ES LA QUEJOSA")
_e = encargo()
ok(ra._quejoso_del_amparo(_e, None, RESOL) == QUEJOSA and ra._recurrente_de(_e, None, RESOL) == RECURRENTE
   and ra.papel_del_recurrente(_e, None, RESOL) == "tercero",
   "el resolutivo del juzgado dice a quién amparó: la tecleada es la recurrente, tercera interesada")
ok(ra._quejoso_del_amparo(encargo(quejoso=QUEJOSA), None, RESOL) == QUEJOSA
   and ra.papel_del_recurrente(encargo(quejoso=QUEJOSA), None, RESOL) == "quejoso",
   "si recurre la propia quejosa, nada cambia")
_uif = encargo(quejoso="Titular de la Unidad de Inteligencia Financiera")
ok(ra.papel_del_recurrente(_uif, types.SimpleNamespace(quejoso="Interamericana Ejemplo, S.A.",
                                                      tercero_interesado="", autoridad_responsable=""), "")
   == "autoridad", "la UIF (711/2025) sigue siendo autoridad recurrente")
ok(ra._misma_parte("Impulsora de Desarrollos S.A de C.V.", "Impulsora de Desarrollos VV, sociedad anónima de capital variable")
   and not ra._misma_parte("Juzgado Quinto de Distrito", "Juzgado Séptimo de Distrito"),
   "la misma parte escrita de dos maneras; dos juzgados distintos no")
_d = ra._datos_estructura(_e, "", acto=RECURRIDA, partes=None, fases=fases())
ok(_d["quejoso"] == QUEJOSA and _d["recurrente"] == RECURRENTE and _d["papel_recurrente"] == "tercero"
   and _d["tercero"] == RECURRENTE,
   f"los datos del documento: quejosa, recurrente y tercera: {(_d['quejoso'], _d['recurrente'], _d['tercero'])}")
ok(_d["organo_recurrido"] == "Juzgado Séptimo de Distrito en el Estado de Ejemplo",
   f"el órgano recurrido es el juzgado que dictó la recurrida, no la Magistrada: {_d['organo_recurrido']}")
ok(dc.favorece_a_la_persona("fundado", "amparo_revision", RECURRENTE, True, papel="tercero") is False
   and dc.favorece_a_la_persona("infundado", "amparo_revision", RECURRENTE, True, papel="tercero") is True
   and dc.favorece_a_la_persona("fundado", "amparo_revision", RECURRENTE, True) is True,
   "el diálogo: si prospera la tercera, pierde la quejosa (sin papel, el nombre decía lo contrario)")
ok("PARTE TERCERA INTERESADA" in dc.quien_combate("amparo_revision", RECURRENTE, True, papel="tercero"),
   "la propuesta sabe quién combate")
_leg = ta.legitimacion_de("amparo_revision", RECURRENTE, "", "***", papel="tercero")
ok("artículo 5o., fracción III" in _leg and "6º" not in _leg and "tercera interesada" in _leg,
   f"legitimación de la tercera: art. 5o., fr. III, no el 6o.: {_leg}")
ok("5o., fracción II, y 87" in ta.legitimacion_de("amparo_revision", "Director de Ingresos", "", "***", papel="autoridad")
   and "6o." in ta.legitimacion_de("amparo_revision", QUEJOSA, "", "***"),
   "la autoridad por el 5o., fr. II y el 87; la quejosa, como siempre")
_ps = fs.prompt(tipo_asunto="amparo_revision", expediente="631/2025", quejoso=QUEJOSA, recurrente=RECURRENTE,
                papel_recurrente="tercero", organo="Juzgado Séptimo de Distrito", estudio="x")
ok("Parte quejosa (promovió el amparo): Unión Ejemplo" in _ps
   and "Parte recurrente: Inmobiliaria Ejemplo, S.A. de C.V. (parte tercera interesada)" in _ps
   and "Órgano que dictó la sentencia recurrida: Juzgado Séptimo" in _ps
   and "«la autoridad responsable»" not in _ps,
   "la síntesis recibe los papeles como datos, sin las figuras del amparo directo como ejemplo")
ok("PROMOVIÓ" in fp._figura("amparo_revision", "quejoso") and "TERCERA INTERESADA" in fp._figura("amparo_revision", "tercero")
   and fp._figura("amparo_directo", "quejoso") == "QUEJOSO",
   "la ficha de partes de la revisión pide quién promovió el amparo, no «QUEJOSA Y RECURRENTE»")
AUTO = ("Fórmese el expediente número 631/2025. Se tiene a Inmobiliaria Ejemplo, S.A. de C.V., tercera interesada, "
        "interponiendo recurso de revisión contra la sentencia dictada en el juicio de amparo indirecto 950/2024, "
        "promovido por Unión Ejemplo, A.C., contra actos de la Sala Civil Uno.")
_fi = {"tipo_asunto": "amparo_revision", "quejoso": RECURRENTE, "recurrente": "", "tercero_interesado": ""}
_av = fa.reconciliar_papeles(_fi, AUTO)
ok(_fi["recurrente"] == RECURRENTE and _fi["quejoso"] == QUEJOSA and _av and "TERCERO INTERESADO" in _av[0],
   f"en la raíz: la ficha de la admisión pone a la tercera como recurrente: {_fi}")
_fi2 = {"tipo_asunto": "amparo_revision", "quejoso": QUEJOSA, "recurrente": "", "tercero_interesado": ""}
ok(fa.reconciliar_papeles(_fi2, AUTO) == [] and _fi2["quejoso"] == QUEJOSA,
   "y no toca una ficha que ya estaba bien (el carácter va pegado al nombre)")
ok(fr.numero_del_amparo(RESULTANDOS, "631/2025") == "950/2024",
   "el número del amparo, no el del toca que se nombra antes")

print("\n8 · EL «CONSIDERANDO OCTAVO» DE LA RECURRIDA NO ES UNA REMISIÓN ROTA")
ok(ce.remisiones_rotas("SEXTO. Estudio. La queja contra el considerando octavo de la sentencia recurrida es "
                       "infundada.") == [],
   "rotulado como de la recurrida, no se acusa")
ok(ce.remisiones_rotas("La queja contra el considerando octavo es infundada.") == ["considerando octavo"],
   "sin rótulo, se sigue acusando")
_g = pe._linea_seg({"id": "A2.a", "etiqueta": "innecesario", "dato": {
    "fuente": "escrito", "cita": "para controvertir el considerando OCTAVO de esta resolución"}}, "residual")
ok(_g.endswith("· considerando de la resolución combatida: octavo"),
   f"el guion rotula el considerando del dato como de la resolución combatida: {_g[-70:]}")
ok("«considerando de la resolución combatida»" in pe.bloque("APARTADO 1\n" + _g)
   and "«considerando de la resolución combatida»" not in pe.bloque("APARTADO 1\n  APLICA A1.a"),
   "y el bloque describe el rótulo sólo cuando aparece")

print("\n9 · LAS PUERTAS DEL SERVIDOR (por AST)")
_src = open(os.path.join(AQUI, "main.py"), encoding="utf8").read()
_arbol = ast.parse(_src)
_prop = next(n for n in ast.walk(_arbol) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
             and n.name == "_taller_proponer_nucleo")
_seg = ast.get_source_segment(_src, _prop)
ok("conceptos_omitidos(_info_c, \"fundado\"" in _seg and '"conceptos_omitidos": _conceptos_omitidos' in _seg
   and "_fr_c.resolvio_a_quo(" not in _seg and "info_de_rama(" in _seg,
   "/taller/proponer devuelve los conceptos omitidos para la vía que prospera, con el resolutivo del juzgado")
_fav = next(n for n in ast.walk(_arbol) if isinstance(n, ast.FunctionDef) and n.name == "_taller_favorece")
ok("papel=_taller_papel_recurrente(r)" in ast.get_source_segment(_src, _fav),
   "el diálogo recibe el carácter del recurrente")

_gs = next(n for n in ast.walk(_arbol) if isinstance(n, ast.FunctionDef) and n.name == "_taller_guardar_sesion")
ok('"recurrente": getattr(e, "recurrente", "")' in ast.get_source_segment(_src, _gs),
   "la sesión guarda al recurrente (la recuperación ya lo leía y nadie lo escribía)")
_gen = next(n for n in ast.walk(ast.parse(_src_ra)) if isinstance(n, ast.AsyncFunctionDef) and n.name == "generar")
_gsrc = ast.get_source_segment(_src_ra, _gen)
ok("_quejoso_del_amparo(e, partes, _res_ad)" in _gsrc and "e.recurrente = _q_tecleado" in _gsrc,
   "en la raíz (el adelanto): si el resolutivo del juzgado dice que la quejosa es otra, lo tecleado es la recurrente")

# ═══════════════════════════════════════════════════════════════════════════
print("\n10 · LA REVISIÓN ADVERSARIAL DEL 28-sep-2026")
# (a) LA QUEJOSA QUE GANA SU RECURSO CONTRA UNA CONCESIÓN NO PIERDE EL AMPARO.
ok(ta.rama_revision("concede", "fundado", quien_recurre="quejoso") == "revoca_fondo_concede"
   and ta.rama_revision("concede", "fundado", quien_recurre="quejoso", solo_efectos=True) == "modifica_efectos"
   and ta.rama_revision("concede", "fundado") == "revoca_fondo_niega"
   and ta.rama_revision("concede", "fundado", quien_recurre="tercero") == "revoca_fondo_niega",
   "recurre la quejosa y gana: se concede (fr. V); si no consta o recurre otra parte, como antes")
ok(ta.rama_revision("sobresee_concede", "fundado", quien_recurre="quejoso") == "revoca_sobreseimiento_concede"
   and ta.reasuncion("sobresee_concede", "fundado", quien_recurre="quejoso") == "sobreseimiento"
   and ta.reasuncion("sobresee_concede", "fundado", quien_recurre="tercero") == "concesion",
   "la quejosa que recurre una mixta combate el sobreseimiento (frs. I y V)")
_pq = ta.puntos_quejosa_mejora(RESOL)
ok(_pq[0] == "PRIMERO. Se {HUECO} la sentencia recurrida." and "ampara y protege a Unión Ejemplo, A.C." in _pq[1]
   and "no ampara" not in _pq[1]
   and ta.puntos_quejosa_mejora("", "modifica")[0] == "PRIMERO. Se modifica la sentencia recurrida.",
   "los puntos: revocar o modificar en hueco hasta el estudio, y la ampara con el acto del juzgado")
# (b) LA IMPROCEDENCIA QUE PROSPERA SOBRESEE, SIN REASUMIR (fr. II).
_tec_rs = ta.tecnica_de("amparo_revision", "revoca_sobresee")
ok(ta.rama_revision("concede", "fundado", quien_recurre="autoridad", procedencia=True) == "revoca_sobresee"
   and ta.reasuncion("concede", "fundado", quien_recurre="autoridad", procedencia=True) == ""
   and ta.ejecutoria_concede("revoca_sobresee") is False
   and ta.TECNICA_RESOLUCION["revision_sobresee_por_improcedencia"] in _tec_rs
   and ta.TECNICA_RESOLUCION["revision_reasume_concesion"] not in _tec_rs,
   "prospera la improcedencia de la autoridad: revoca y sobresee, con su técnica y sin la fr. VI")
_f_pro = fases(problemas=[{"pregunta": "¿Se actualiza la falta de interés jurídico?", "cubre": [1],
                           "clase": "procedencia", "jerarquia": "principal"}])
_i_pro = ra.info_de_rama(resultado(f=_f_pro))
ok(_i_pro["clase_principal"] == "procedencia" and fr.conceptos_omitidos(_i_pro, "fundado") is None
   and ra._rama_de(resultado(f=_f_pro), CRIT_F) == "revoca_sobresee"
   and ra._rama_de(resultado(f=_f_pro), CRIT_I) == "confirma_concede",
   "la clase del principal viaja en la rama: no se piden conceptos y la rama del estudio sobresee")
ok(ra._reasuncion_del_asunto(resultado(f=_f_pro), CRIT_F)["reasuncion"] == ""
   and ra._reasuncion_del_asunto(resultado(f=_f_pro), CRIT_F)["procedencia"] is True,
   "el material del estudio sabe que no se reasume y por qué")
# (c) LOS PAPELES SIN PRUEBA: «» y no «quejoso»; la grafía no convierte a la quejosa en tercera.
QUEJOSA_N = "María de la Luz Hernández Ruiz"
RES_N = ("La Justicia de la Unión no ampara ni protege a Ma. de la Luz Hernández Ruiz, contra el acto que "
         "reclamó al Director de Catastro, por los motivos expuestos en el considerando cuarto.")
_eq = encargo(quejoso=QUEJOSA_N)
ok(ra._quejoso_del_amparo(_eq, None, RES_N) == QUEJOSA_N and ra._recurrente_de(_eq, None, RES_N) == ""
   and ra.papel_del_recurrente(_eq, None, RES_N) == "quejoso",
   "la quejosa recurrente con su nombre abreviado en el resolutivo sigue siendo la quejosa")
ok(ra.papel_del_recurrente(encargo(quejoso="Juan Gómez Pérez"), None,
                           "La Justicia de la Unión ampara y protege a Juan Gomes Pérez, contra el acto "
                           "que reclamó al Director de Catastro.") == "quejoso",
   "una errata tampoco la convierte en tercera interesada")
ok(ra._parte_parecida(QUEJOSA_N, "Ma. de la Luz Hernández Ruiz")
   and not ra._parte_parecida("María Hernández Ruiz", "Inmobiliaria Hernández Ruiz, S.A. de C.V.")
   and not ra._parte_parecida(QUEJOSA, RECURRENTE),
   "la comparación tolera abreviaturas y erratas, no apellidos compartidos entre partes distintas")
RES_REM = ("La Justicia de la Unión ampara y protege a la parte quejosa precisada en el resultando primero, "
           "contra el acto que reclamó a la Sala, por los motivos expuestos en el considerando séptimo.")
_sin = types.SimpleNamespace(quejoso="", tercero_interesado="", autoridad_responsable="")
ok(ra.papel_del_recurrente(encargo(), _sin, RES_REM) == "",
   "sin prueba de quién recurre (resolutivo que remite, ficha vacía): «», no «quejoso»")
_f_rem = fases()
_f_rem.resolutivo_recurrida = RES_REM
_i_rem = ra.info_de_rama(resultado(f=_f_rem))
ok(_i_rem["quien_recurre"] == "" and fr.conceptos_omitidos(_i_rem, "fundado")["reasuncion"] == "concesion",
   "…y con «» la fracción VI se aplica: se piden los conceptos (el fallo del 631 no vuelve)")
ok(ra.papel_del_recurrente(encargo(quejoso=QUEJOSA), None,
                           "La Justicia de la Unión no ampara ni protege a la parte quejosa precisada en el "
                           "resultando primero, contra el acto que reclamó a la Sala.") == "quejoso",
   "si el juzgado negó, sólo la quejosa resiente el fallo: recurre ella")
# (d) CONCEPTOS «POR CONFIRMAR»: la recurrida es legible y no dice que dejara ninguno sin estudiar.
_REC_TODOS = ("JUZGADO SÉPTIMO DE DISTRITO\nCONSIDERANDO SÉPTIMO. Es fundado el segundo concepto de "
              "violación. " + "Estudio del concepto. " * 80 + "Son infundados el primero y el tercero. "
              + "Estudio de los otros. " * 40 + "Por lo expuesto, se R E S U E L V E: SEGUNDO. " + RESOL)
_co_pc = fr.conceptos_omitidos({"tipo_asunto": "amparo_revision", "que_hizo": "concede",
                                "quien_recurre": "tercero"}, "fundado", fases(fuentes=[_REC_TODOS, ""]))
ok(_co_pc["hacen_falta"] == "por_confirmar" and "revisión adhesiva" in _co_pc["por_que"],
   "el juzgado estudió todos: «por confirmar», no se bloquea")
ok(fr.dejo_conceptos_sin_estudiar(RECURRIDA) is True and fr.dejo_conceptos_sin_estudiar("") is None
   and fr.dejo_conceptos_sin_estudiar(_REC_TODOS) is False
   and fr.dejo_conceptos_sin_estudiar("Texto. " * 300 + "sin que sea necesario analizar los demás conceptos "
                                      "de violación") is True,
   "la lectura de la recurrida: innecesario el estudio de los restantes, sin que sea necesario…")
_bpc = f6._bloque_conceptos("revoca_fondo_niega", "", reasuncion={"reasuncion": "concesion",
                                                                   "hacen_falta": "por_confirmar",
                                                                   "tenemos": False})
ok("NO DICE QUE EL JUZGADO DEJARA CONCEPTOS SIN ESTUDIAR" in _bpc and "«" not in _bpc
   and "FALTAN ESOS CONCEPTOS" not in _bpc,
   "el estudio sabe que puede no haber nada que reasumir, sin frases para copiar")
# (e) LA TÉCNICA DE LA fr. VI SIN LOS CONCEPTOS NO MANDA CALIFICARLOS (v3, sin guion).
_TEC4 = [{"registro": x, "rubro": f"RUBRO {x}.", "instancia": "Segunda Sala", "tipo": "JURISPRUDENCIA",
          "texto": "texto"} for x in ("171925", "178784", "182039", "174177")]
_m_sin = f6.Material(tipo_asunto="amparo_revision", tesis=[dict(t) for t in _TEC4])
_m_sin.reasuncion = {"reasuncion": "concesion", "hacen_falta": True, "tenemos": False, "conceptos": ""}
_bt_sin = f6._bloque_tecnica("amparo_revision", "revoca_fondo_niega", False, _m_sin)
ok("178784" not in _bt_sin and "182039" not in _bt_sin and "171925" in _bt_sin and "174177" in _bt_sin
   and "UN CONSIDERANDO PROPIO" not in _bt_sin and "SIN LOS CONCEPTOS NO SE CALIFICAN" in _bt_sin,
   "sin conceptos: ni el renglón del considerando que los califica ni sus dos apoyos")
_m_con = f6.Material(tipo_asunto="amparo_revision", tesis=[dict(t) for t in _TEC4])
_m_con.reasuncion = dict(_m_sin.reasuncion, tenemos=True, conceptos="PRIMERO. Concepto.")
_bt_con = f6._bloque_tecnica("amparo_revision", "revoca_fondo_niega", False, _m_con)
ok("178784" in _bt_con and "UN CONSIDERANDO PROPIO" in _bt_con
   and "178784" in f6._bloque_tecnica("amparo_revision", "revoca_fondo_niega", False,
                                      f6.Material(tipo_asunto="amparo_revision",
                                                  tesis=[dict(t) for t in _TEC4])),
   "con los conceptos, o sin la reasunción calculada, como antes")
# (f) LA TESIS QUE INVOCÓ OTRA PARTE NO ES APOYO DE LA VÍA.
_RA = ("La quejosa había alegado que una tercera persona no podía colocarse en el lugar de la actora. "
       "Para apoyar su planteamiento, había citado la tesis II.2o.C.296 C, con registro digital 188480, "
       "de rubro “SUSTITUCIÓN PROCESAL. NO AFECTA A LA RELACIÓN SUSTANCIAL PACTADA”.")
ok(f6.registros_de_otros(_RA, True) == {"188480"} and f6.registros_de_otros(_RA, False) == set()
   and "188480" in f6.registros_de_la_parte("", None, _RA)[0],
   "en la revisión, lo que relata la recurrida también es de otro (1-duodecies lo ve)")
_bg = f6._bloque_global({"sentido": "infundado", "razon": "La razón del motor.",
                         "alternativa": {"sentido": "fundado", "razon": "La vía que se tomó.",
                                         "apoyos": ["170378", "188480"]}}, CRIT_F, {"188480"})
ok("Apoyos del acervo para esta vía: 170378 (" in _bg and "tiene que contestar" in _bg
   and _bg.count("188480") == 1,
   "en la vía que se tomó, la tesis de la quejosa va aparte, para contestarla, no como apoyo")

print()
if FALLOS:
    print(f"FALLAN {len(FALLOS)}: " + " · ".join(FALLOS))
    sys.exit(1)
print("RESULTADO: TODAS LAS COMPROBACIONES PASAN")
