# -*- coding: utf-8 -*-
"""LA TARJETA DEL PROBLEMA PRINCIPAL (AR 631/2025, 28-sep-2026), SIN MODELO.

El asunto de prueba reproduce la forma del 631 —amparo en revisión que la
tercera interesada interpuso contra una sentencia que CONCEDIÓ; el principal
es de fondo (¿la sustitución procesal alteró la cosa juzgada?) y el accesorio
procesal (congruencia y exhaustividad)—, con nombres de prueba. Lo que se
comprueba, por qué importa:

  · las dos vías llevan CADA UNA su razón (fallo 1 del 631: la de la otra vía
    viajó con el sentido elegido);
  · el desenlace sale por código y, al revocar una concesión, dice que antes
    de negar se estudian los conceptos que el juez no estudió (art. 93, fr. VI);
  · el accesorio se resuelve solo en cada vía con el árbol, y como en los dos
    planes de producción de la fila 462: con el principal infundado, P2
    infundado; con el principal fundado, P2 innecesario;
  · ningún registro fuera del acervo se pinta como cita; lo abandonado no se
    propone; la fuerza para un colegiado no se toma de `obligatoria`;
  · el estado es determinista y sin porcentajes, y la propuesta no se rotula
    «recomendada» cuando el contraste no cierra el punto.

Con TARJETA_FILA_462=<propuesta_462.json> (y TARJETA_PLAN_462=<plan>) corre
además sobre la fila real volcada, sólo en lectura.

    .venv/bin/python test_tarjeta_decision.py
"""
import copy
import json
import os
import sys
import time

import tarjeta_decision as td

FALLOS = []


def ok(cond, que):
    print(f"   {'PASA ' if cond else 'FALLA'}  {que}")
    if not cond:
        FALLOS.append(que)


TRIBUNAL = ("Tercer Tribunal Colegiado en Materias Administrativa y Civil del Vigésimo "
            "Segundo Circuito")

# ═══ EL ASUNTO DE PRUEBA ═══════════════════════════════════════════════════
P1 = "¿La sustitución de la parte actora para ejecutar la sentencia alteró la cosa juzgada?"
P2 = ("¿La sentencia recurrida fue congruente, exhaustiva y debidamente fundada y motivada al "
      "resolver sobre la sustitución de la parte actora?")
PROBLEMAS = [
    {"pregunta": P1, "jerarquia": "principal", "clase": "fondo", "depende_de": None,
     "resolvio": "El Juzgado estimó que la sustitución alteró la cosa juzgada porque la "
                 "adquirente no era causahabiente del contrato de arrendamiento.",
     "combate": "La recurrente sostiene que la interlocutoria sólo determinó quién estaba "
                "legitimado para continuar la ejecución."},
    {"pregunta": P2, "jerarquia": "accesorio", "clase": "procesal", "depende_de": 1,
     "resolvio": "El Juzgado consideró innecesario estudiar los restantes conceptos.",
     "combate": "La recurrente sostiene que la sentencia omitió examinar las normas locales y "
                "las constancias de la transmisión del inmueble."},
]


def _tesis(reg, rubro, instancia, tipo, **kw):
    loc = "[J]" if tipo == "JURISPRUDENCIA" else "[TA]"
    base = {"registro": reg, "rubro": rubro, "instancia": instancia, "tipo": tipo,
            "texto": "texto de prueba", "obligatoria": tipo == "JURISPRUDENCIA",
            "localizacion": f"{loc}; 9a. Época; prueba", "vigencia": None}
    base.update(kw)
    return base


MATERIAL = {
    "tesis": [
        _tesis("2026918", "COSA JUZGADA Y SUS EFECTOS DIRECTO Y REFLEJO.", "Primera Sala",
               "JURISPRUDENCIA"),
        _tesis("168958", "COSA JUZGADA. SUS LÍMITES OBJETIVOS Y SUBJETIVOS.", "Pleno",
               "JURISPRUDENCIA"),
        _tesis("170353", "COSA JUZGADA. PRESUPUESTOS PARA SU EXISTENCIA.", "Primera Sala",
               "JURISPRUDENCIA"),
        _tesis("170378", "CESIÓN DE DERECHOS LITIGIOSOS. PROCEDE ANTES DE QUE ADQUIERA "
                         "FIRMEZA EL FALLO.", "Tribunales Colegiados de Circuito",
               "JURISPRUDENCIA"),
        _tesis("188480", "SUSTITUCIÓN PROCESAL. NO AFECTA A LA RELACIÓN SUSTANCIAL PACTADA.",
               "Tribunales Colegiados de Circuito", "TESIS AISLADA"),
        _tesis("2007402", "SUSTITUCIÓN PROCESAL. SE ACTUALIZA RESPECTO DE QUIEN ADQUIERE LA "
                          "PROPIEDAD.", "Tribunales Colegiados de Circuito", "TESIS AISLADA"),
        # La línea de la Corte que confirmó el acervo.
        _tesis("2013721", "SOBRESEIMIENTO POR INEXISTENCIA DEL ACTO.", "Pleno",
               "JURISPRUDENCIA", de_internet=True, tecnica=True),
        # ABANDONADA según el índice de vigencia (la P. X/2015, 2009817): la
        # fila la trae con `vigencia` null, como las anteriores al 25-sep.
        _tesis("2009817", "CONTROL DIFUSO. LOS TRIBUNALES COLEGIADOS PUEDEN EJERCERLO.",
               "Pleno", "TESIS AISLADA"),
    ],
    "espejo": [
        {"problema": P1, "filas": [
            {"expediente": "AR 100/2024", "fecha": "2024-05-02", "sentido": "confirma",
             "score": 0.78, "tipo_asunto": "AR", "tema": "sustitución procesal"},
            {"expediente": "AR 200/2023", "fecha": "2023-03-01", "sentido": "revoca",
             "similitud": 86, "nivel": "posible", "calificacion": "fundado",
             "razon": "la adquirente es causahabiente procesal", "neun": 12345}]},
        {"problema": P2, "filas": [
            {"expediente": "AR 300/2022", "fecha": "2022-01-01", "sentido": "confirma",
             "score": 0.75}]},
    ],
}

CHECKLIST = [
    {"tema": "Efectos de la sustitución sobre la cosa juzgada.", "papel": "principal",
     "numero": 1, "relacion": "distinto", "presupone": None,
     "si_prospera": {"sentido": "fundado", "razon": "la adquisición transmitió el interés"},
     "si_no_prospera": {"sentido": "infundado", "razon": "no hay causahabiencia"},
     "con_propuesta": "infundado", "con_alternativa": "fundado", "tema_distinto": False},
    {"tema": "Congruencia y exhaustividad de la sentencia.", "papel": "accesorio",
     "numero": 2, "relacion": "depende", "presupone": None,
     "si_prospera": {"sentido": "innecesario",
                     "razon": "Si prospera el principal, la nueva decisión hace innecesario "
                              "resolver la omisión."},
     "si_no_prospera": {"sentido": "infundado",
                        "razon": "Si subsiste la razón principal, el juzgado podía omitir lo "
                                 "que no cambiaba el sentido."},
     "con_propuesta": "infundado", "con_alternativa": "innecesario", "tema_distinto": False},
]

RAZON_MOTOR = ("La adquisición del inmueble, por sí sola, no acredita la transmisión de los "
               "derechos litigiosos; la sustitución alteró los límites subjetivos.")
RAZON_ALT = ("La interlocutoria sólo sustituyó al acreedor en la ejecución, sin alterar la cosa "
             "ni las prestaciones; la transmisión trasladó el interés para ejecutar.")

GLOBAL = {
    "sentido": "infundado", "razon": RAZON_MOTOR, "alcanza": True, "confianza": "media",
    "apoyos": ["2026918", "168958", "170353", "188480", "9999991"],
    "efecto": "Se confirma la concesión.",
    "problema_que_decide": P1,
    "en_contra": "La transmisión de la propiedad pudo trasladar el interés para ejecutar.",
    "contexto": {"tema_principal": "Es el punto del que depende la exhaustividad.",
                 "resolvio": "Concedió el amparo."},
    "alternativa": {"sentido": "fundado", "razon": RAZON_ALT,
                    "apoyos": ["170378", "2007402", "188480", "2026918", "2009817"],
                    "efecto": "El principal prosperaría y el segundo quedaría sin materia."},
    "via_protectora": {"sentido": "fundado", "norma": "art. 49 CPC local",
                       "apoyos": ["170378", "2014332"], "limite": "inviable si sólo se "
                       "acredita la propiedad", "lectura": "…", "posible": True},
    "constancias": [{"que": "escritura de prueba", "indispensable": False, "problema": 1}],
    "checklist": CHECKLIST,
}
PROPUESTAS = [
    {"problema": P1, "sentido": "infundado", "razon": RAZON_MOTOR, "alcanza": True,
     "apoyos": ["2026918"], "confianza": "media", "prediccion": {}},
    {"problema": P2, "sentido": "infundado", "razon": "Expuso las razones decisivas.",
     "alcanza": True, "apoyos": ["188480"], "confianza": "media", "prediccion": {}},
]
CONTRASTE = [
    {"numero": 1, "razon_toral": "La adquirente no era causahabiente del contrato.",
     "la_combate": True, "sobrevive": False, "veredicto_previo": "a_examinar"},
    {"numero": 2, "razon_toral": "Era innecesario estudiar lo demás.", "la_combate": True,
     "sobrevive": False, "veredicto_previo": "a_examinar"},
]
RESP = {"formato": 2, "global": GLOBAL, "propuestas": PROPUESTAS, "contraste": CONTRASTE,
        "necesita_conceptos": False}
RAMA = {"tipo_asunto": "amparo_revision", "resolvio_a_quo": "concede", "tribunal": TRIBUNAL,
        "huella": "h631"}
INTERNET = {"pistas": ["2099999"], "resumen": "línea de prueba", "buscado": True}


def armar(resp=None, material=None, problemas=None, contraste=None, rama=None,
          deliberacion=None, internet=INTERNET, **kw):
    return td.armar(copy.deepcopy(resp or RESP), copy.deepcopy(material or MATERIAL),
                    copy.deepcopy(problemas or PROBLEMAS), contraste, None, internet,
                    rama or RAMA, deliberacion, **kw)


print("\n0 · EL MÓDULO ES PURO")
ok("main" not in sys.modules, "importar tarjeta_decision no importa main (ni su arranque)")

print("\n1 · LA FUERZA PARA UN COLEGIADO, POR CÓDIGO (no `obligatoria`)")
F = td.fuerza_para_colegiado
ok(F(_tesis("1", "r", "Primera Sala", "JURISPRUDENCIA"), TRIBUNAL)["fuerza"] == "obliga",
   "jurisprudencia de Sala: obliga")
ok(F(_tesis("1", "r", "Pleno", "JURISPRUDENCIA"), TRIBUNAL)["fuerza"] == "obliga",
   "jurisprudencia del Pleno de la Corte: obliga")
ok(F(_tesis("1", "r", "Segunda Sala", "TESIS AISLADA"), TRIBUNAL)["fuerza"] == "orienta",
   "aislada de la Corte: orienta")
_jc = F(_tesis("1", "r", "Tribunales Colegiados de Circuito", "JURISPRUDENCIA",
               obligatoria=True), TRIBUNAL)
ok(_jc["fuerza"] == "orienta" and "217, párr. tercero" in _jc["fuerza_texto"],
   f"jurisprudencia de colegiado rotulada obligatoria en el acervo: orienta ({_jc['fuerza_texto']})")
ok(F(_tesis("1", "r", "Tribunales Colegiados de Circuito", "TESIS AISLADA"), TRIBUNAL)
   == {"fuerza": "orienta", "fuerza_texto": "orienta"}, "aislada de colegiado: orienta")
_pc = F(_tesis("1", "r", "Plenos de Circuito", "JURISPRUDENCIA", obligatoria=True), TRIBUNAL)
ok(_pc["fuerza"] == "pleno_circuito" and "obliga" not in _pc["fuerza_texto"],
   "Pleno de Circuito: rótulo propio, sin afirmar si obliga")
_prn = F(_tesis("1", "r", "Plenos Regionales", "JURISPRUDENCIA", clave="PR.A.C.CN. J/7 K (11a.)"),
         TRIBUNAL)
ok(_prn["fuerza"] == "obliga" and "Centro-Norte" in _prn["fuerza_texto"],
   "Pleno Regional Centro-Norte ante el XXII Circuito (medido: Centro-Norte): obliga")
ok(F(_tesis("1", "r", "Plenos Regionales", "JURISPRUDENCIA", clave="PR.A.C.CS. J/8 K (12a.)"),
     TRIBUNAL)["fuerza"] == "orienta", "Pleno Regional de otra región: orienta")
_prx = F(_tesis("1", "r", "Plenos Regionales", "JURISPRUDENCIA"), TRIBUNAL)
ok(_prx["fuerza"] == "orienta" and "sin región" in _prx["fuerza_texto"],
   "Pleno Regional sin clave: no se afirma que obligue (se dice por qué)")
ok(F(_tesis("1", "r", "Plenos Regionales", "JURISPRUDENCIA", texto="[TESIS: PR.P.T.CN. J/11 P] …"),
     TRIBUNAL)["fuerza"] == "obliga", "la región se lee también del encabezado del texto")
ok(td.designacion_de(TRIBUNAL) == "XXII.3o.A.C." and td.circuito_de(TRIBUNAL) == 22,
   "la designación y el circuito del tribunal se leen del nombre")
ok(F(_tesis("1", "r", "Tribunales Colegiados de Circuito", "JURISPRUDENCIA",
            clave="XXII.3o.A.C. J/5 K (11a.)"), TRIBUNAL)["fuerza"] == "precedente_propio",
   "jurisprudencia del propio tribunal: precedente propio (art. 228)")
ok(F(_tesis("1", "r", "Tribunales Colegiados de Circuito", "TESIS AISLADA",
            clave="XXII.3o.A.C.12 C (11a.)"), TRIBUNAL)["fuerza"] == "precedente_propio",
   "aislada del propio tribunal: precedente propio")
# EL ART. 228 ES DE LAS PROPIAS JURISPRUDENCIAS (revisión del 28-sep-2026): la
# jurisprudencia del propio tribunal leída de la localización no se reconocía
# («orienta») y la aislada propia llevaba el 228.
_jp = F({"instancia": "Tribunales Colegiados de Circuito", "tipo": "",
         "localizacion": "[J]; 11a. Época; T.C.C.; Gaceta S.J.F.; Tesis: XXII.3o.A.C. J/2 K (11a.); Pág. 5"},
        TRIBUNAL)
ok(_jp["fuerza"] == "precedente_propio" and "228" in _jp["fuerza_texto"]
   and "vincula" in _jp["fuerza_texto"],
   f"jurisprudencia propia leída de la localización: vincula (art. 228): {_jp['fuerza_texto']}")
_ap_p = F({"instancia": "Tribunales Colegiados de Circuito", "tipo": "TESIS AISLADA",
           "localizacion": "[TA]; 11a. Época; T.C.C.; Tesis: XXII.3o.A.C.5 K (11a.)"}, TRIBUNAL)
ok(_ap_p["fuerza"] == "precedente_propio" and "228" not in _ap_p["fuerza_texto"],
   f"la aislada propia es precedente propio sin el art. 228: {_ap_p['fuerza_texto']}")
ok("228" not in td._apoyo_precedente_propio("AR 100/2024")["fuerza_texto"],
   "la sentencia del espejo tampoco lleva el art. 228")
ok(td.clave_de_tesis({"localizacion": "Tesis: I.4o.A. J/12 A (10a.)"}) == "I.4o.A. J/12 A (10a.)",
   "la clave de jurisprudencia de un colegiado se lee de la localización")
ok(F(_tesis("1", "r", "Tribunales Colegiados de Circuito", "TESIS AISLADA",
            clave="XXII.3o.A.C.P.12 C"), TRIBUNAL)["fuerza"] == "orienta",
   "otra designación que empieza igual no es la del tribunal")
ok(F(_tesis("1", "r", "Tribunales Colegiados de Circuito", "TESIS AISLADA",
            clave="XXII.1o.A.C.12 C"), TRIBUNAL)["fuerza"] == "orienta",
   "otro tribunal del mismo circuito no es precedente propio")

print("\n2 · LOS APOYOS, VERIFICADOS CONTRA EL ACERVO")
cat = td.catalogo_de(MATERIAL)
ap, av = td.hidratar_apoyos(["2026918", "el registro 188480", "9999991", "2009817",
                             "art. 2294 del Código Civil local", "2026918"], cat, TRIBUNAL,
                            via="la vía de prueba")
regs = [a["registro"] for a in ap if a["registro"]]
ok(regs == ["2026918", "188480"], f"sólo los del acervo y vigentes, sin repetir: {regs}")
ok(ap[0]["rubro"] == "COSA JUZGADA Y SUS EFECTOS DIRECTO Y REFLEJO." and ap[0]["en_acervo"],
   "con el rubro LITERAL del acervo")
ok(any("9999991" in x and "no está en el acervo" in x for x in av),
   "el registro que no está en el acervo va a los avisos, no a la tarjeta")
ok(any("2009817" in x and "ABANDONADA" in x for x in av),
   "la tesis abandonada (sello del índice, aunque la fila la traiga sin él) no se propone")
ok(any(a["norma"] == "art. 2294 del Código Civil local" for a in ap), "la norma va como norma")
_parcial = _tesis("555555", "PARCIAL", "Pleno", "JURISPRUDENCIA",
                  vigencia={"estado": "abandonada", "parcial": True, "por_clave": "P./J. 1/2020"})
ap2, av2 = td.hidratar_apoyos(["555555"], {"555555": _parcial}, TRIBUNAL)
ok(len(ap2) == 1 and "EN PARTE" in (ap2[0]["vigencia"] or ""),
   "la abandonada EN PARTE se enseña con su sello, no se quita")
ap3, av3 = td.hidratar_apoyos([{"id": "T7"}], cat, TRIBUNAL)
ok(not ap3 and av3, "un id del catálogo sin registro no se pinta")
ok(td.registros_sueltos("como dice la tesis (9999993) y la 2026918, con una condena de $600000 "
                        "y otra de 700000 pesos", cat) == ["9999993"],
   "en la razón, un registro fuera del acervo se señala; una cantidad no es un registro")

print("\n3 · EL DESENLACE, POR CÓDIGO")
D = td.desenlace_de
_c, _n = D("amparo_revision", "concede", "infundado")
ok(_c and _c[0].startswith("PRIMERO. Se confirma") and "ampara y protege" in _c[1] and _n is None,
   "no prospera contra una concesión: se confirma y se ampara")
_r, _nr = D("amparo_revision", "concede", "fundado")
# LOS MISMOS PUNTOS QUE EL DOCUMENTO (integración con SPEC B, 28-sep-2026): el
# verbo del amparo depende del estudio de los conceptos que el juez no estudió,
# así que va en hueco —antes decía «no ampara», que es lo que se firmó mal en
# el 631—.
ok(_r[0] == "PRIMERO. Se revoca la sentencia recurrida." and td.HUECO in _r[1]
   and "no ampara" not in _r[1],
   "prospera contra una concesión: se revoca, y el amparo queda en hueco…")
ok(_nr and "93, fr. VI" in _nr and "reasume jurisdicción" in _nr and "sólo si caen" in _nr,
   "…porque antes se estudian los conceptos que el juez no estudió (art. 93, fr. VI)")
_rq, _nrq = D("amparo_revision", "concede", "fundado", quien_recurre="quejoso")
# REVISIÓN DEL 28-sep-2026: esto afirmaba «se revoca y se niega» como correcto.
# La quejosa que recurre su concesión pide más —otros efectos, otro alcance—;
# si gana, la sentencia que corresponde la ampara (fr. V), no la deja sin nada.
ok(td.HUECO in _rq[0] and "ampara y protege" in _rq[1] and "no ampara" not in _rq[1]
   and not (_nrq and "fr. VI" in _nrq) and "fr. V," in (_nrq or ""),
   "si recurre la propia quejosa y gana: no se le niega el amparo (fr. V); revocar o "
   "modificar queda en hueco hasta el estudio")
_RES_Q = ("La Justicia de la Unión ampara y protege a María López Ruiz, contra el acto que reclamó "
          "al Director de Catastro, para los efectos precisados en el considerando quinto.")
_rq2, _ = D("amparo_revision", "concede", "fundado", quien_recurre="quejoso",
            resolutivo_recurrida=_RES_Q)
ok("ampara y protege a María López Ruiz" in _rq2[1] and "no ampara" not in " ".join(_rq2)
   and "efectos precisados en el último considerando de esta ejecutoria" in _rq2[1],
   "…con el sujeto y el acto del resolutivo del juzgado y los efectos de esta ejecutoria")
_rqm, _nqm = D("amparo_revision", "sobresee_concede", "fundado", quien_recurre="quejoso")
ok(_rqm[0].startswith("PRIMERO. En la materia de la revisión, se revoca") and td.HUECO in _rqm[1]
   and "primera vez" in (_nqm or "") and "fracciones I y V" in (_nqm or "")
   and "fr. VI" not in (_nqm or ""),
   "la quejosa que recurre una sentencia mixta combate el sobreseimiento: se levanta (frs. I y V)")
_rp, _np = D("amparo_revision", "concede", "fundado", quien_recurre="autoridad", procedencia=True)
ok(_rp[0] == "PRIMERO. Se revoca la sentencia recurrida." and "Se sobresee en el juicio" in _rp[1]
   and td.HUECO not in " ".join(_rp) and "fr. VI" not in (_np or "") and "fracciones II" in (_np or ""),
   "prospera la improcedencia que alegó la autoridad: se revoca y se sobresee, sin reasumir (fr. II)")
_rc, _nc = D("amparo_revision", "sobresee_concede", "infundado", quien_recurre="tercero",
             resolutivo_recurrida=_RES_Q)
ok(_rc[0] == "PRIMERO. En la materia de la revisión, se confirma la sentencia recurrida."
   and "María López Ruiz" in _rc[1] and not any("Se sobresee" in x for x in _rc)
   and "queda firme" in (_nc or ""),
   "la vía que confirma una sentencia mixta tampoco redecreta el sobreseimiento que nadie impugnó")
_rcq, _ = D("amparo_revision", "sobresee_concede", "infundado", quien_recurre="quejoso")
ok(len(_rcq) == 3 and "Se sobresee" in _rcq[1],
   "si la recurrente es la quejosa y pierde, se confirma la mixta entera, como antes")
_rs, _ = D("amparo_revision", "sobresee_concede", "fundado", sobresee_ademas=True)
ok(_rs[0] == "PRIMERO. En la materia de la revisión, se revoca la sentencia recurrida.",
   "si la recurrida también sobreseyó, la revocación se acota a la materia de la revisión")
_RES_J = ("La Justicia de la Unión ampara y protege a Unión Ejemplo, A.C., contra el acto que "
          "reclamó a la Sala, por los motivos expuestos en el considerando séptimo de esta sentencia.")
_rn, _ = D("amparo_revision", "concede", "fundado", resolutivo_recurrida=_RES_J)
_cn, _ = D("amparo_revision", "concede", "infundado", resolutivo_recurrida=_RES_J)
ok("Unión Ejemplo, A.C." in _rn[1] and td.HUECO in _rn[1] and "Unión Ejemplo, A.C." in _cn[1],
   "con el resolutivo del juzgado, los puntos nombran a quien él amparó (577c700)")
ok("{" not in json.dumps([_c, _r]) and "parte quejosa" in _c[1],
   "sin nombres propios ni marcadores sin rellenar (el encargo del 631 cruzaba los papeles)")
_s, _ns = D("amparo_revision", "sobresee", "fundado")
ok(_s[0].startswith("PRIMERO. Se revoca") and td.HUECO in _s[1] and "primera vez" in (_ns or ""),
   "levantado el sobreseimiento, el sentido del amparo queda a la vista del estudio")
_x, _nx = D("amparo_revision", "", "fundado")
ok(td.HUECO in " ".join(_x) and _nx, "sin saber qué hizo el juzgado, el hueco a la vista")
ok("ampara y protege" in D("amparo_directo", "", "fundado")[0][0]
   and "no ampara" in D("amparo_directo", "", "infundado")[0][0], "amparo directo: concede / niega")
ok(D("queja", "", "fundado")[0] == ["ÚNICO. Es fundado el recurso de queja."], "queja")

print("\n4 · EL 631 SINTÉTICO: LA TARJETA ENTERA")
t = armar()
ok(t["formato"] == 1 and t["estado_calculo"] == "listo" and t["huella"] == "h631",
   "formato 1, listo, con la huella del adelanto")
pr = t["principal"]
ok(pr["numero"] == 1 and pr["jerarquia_de"] == "fase3" and pr["clase"] == "fondo",
   "el principal es el 1, de fondo, por la fase 3")
ok(pr["por_que_principal"] == GLOBAL["contexto"]["tema_principal"] and pr["discrepa_motor"] is None,
   "con el porqué que escribió el motor y sin discrepancia")
ok(pr["contraste"]["veredicto_previo"] == "a_examinar", "con el contraste del principal")
vp, vo = t["vias"]["propuesta"], t["vias"]["opuesta"]
ok(vp["sentido"] == "infundado" and vp["prospera"] is False
   and vo["sentido"] == "fundado" and vo["prospera"] is True, "las dos vías, en sentidos contrarios")
ok(vp["razon"] == RAZON_MOTOR and vo["razon"] == RAZON_ALT,
   "CADA VÍA CON SU RAZÓN: la contraria no lleva la del motor (fallo 1 del 631)")
ok(vp["desenlace"][0].startswith("PRIMERO. Se confirma") and vo["desenlace"][0].startswith(
    "PRIMERO. Se revoca") and "93, fr. VI" in (vo["desenlace_nota"] or ""),
   "el desenlace de cada vía, con el estudio de los conceptos omitidos en la que revoca")
ok([a["registro"] for a in vp["apoyos"]] == ["2026918", "168958", "170353", "188480"],
   "los apoyos de la propuesta, sin el registro inventado")
ok("2009817" not in [a["registro"] for a in vo["apoyos"]],
   "la contraria no propone la tesis abandonada")
ok(all(a["fuerza"] in ("obliga", "orienta", "pleno_circuito", "precedente_propio")
       for v in (vp, vo) for a in v["apoyos"] if a["registro"]), "cada apoyo con su fuerza")
ok(vp["via_protectora"] is None and vo["via_protectora"] is not None
   and vo["via_protectora"]["apoyos"] == ["170378"],
   "la vía protectora va con la vía que favorece y sólo con sus apoyos comprobados")
ok(any("2014332" in a for a in t["avisos"]) and any("9999991" in a for a in t["avisos"]),
   "lo no comprobado de la vía protectora y de la propuesta, en los avisos")
sec = t["secundarios"]
ok(len(sec) == 1 and sec[0]["numero"] == 2 and sec[0]["relacion"] == "depende",
   "un secundario: el 2, que depende del principal")
ok(sec[0]["en_propuesta"]["sentido"] == "infundado" and sec[0]["en_opuesta"]["sentido"] == "innecesario",
   "P2 se resuelve solo: infundado con el principal infundado, innecesario con el principal "
   "fundado (los dos planes de producción de la fila 462)")
ok(sec[0]["en_opuesta"]["de"] == "principal" and "prospera el principal" in sec[0]["en_opuesta"]["por_que"],
   "y dice por qué, con la suerte que el motor escribió para esa vía")
ok(t["estado"] == "reñido" and t["recomendada"] is None,
   "el contraste no cierra el punto: reñido y la propuesta NO se rotula recomendada")
ok(t["estado_por_que"] and not any("%" in x for x in t["estado_por_que"]),
   "con sus razones y sin porcentajes")
ok("confianza" not in json.dumps(t), "la confianza que el modelo dice de sí mismo no se enseña")
tt = t["tu_tribunal"]
ok([f["expediente"] for f in tt] == ["AR 100/2024", "AR 200/2023"],
   "su tribunal: sólo las filas del principal")
ok(tt[1]["similitud"] == 0.86 and tt[1]["nivel"] == "posible" and tt[1]["calificacion"] == "fundado"
   and tt[0]["similitud"] is None,
   "tolera la fila de la OAJ (similitud calibrada) y la vieja (el coseno no es similitud)")
ok([a["registro"] for a in t["linea_corte"]["confirmadas"]] == ["2013721"]
   and t["linea_corte"]["pistas"] == ["2099999"],
   "la línea de la Corte: lo confirmado por el acervo y las pistas aparte")
ok(t["que_la_cambiaria"]["en_contra"] == GLOBAL["en_contra"]
   and t["que_la_cambiaria"]["constancias_indispensables"] == []
   and t["que_la_cambiaria"]["limite_protector"], "lo que la cambiaría")
ok(t["deliberacion"] is None and t["independientes"] == [], "sin deliberación: null")
# LOS CONCEPTOS OMITIDOS YA NO SON UN GANCHO: con la función de SPEC B, la vía
# contraria del 631 (la que revoca la concesión) los pide.
ok(t["conceptos_omitidos"] and t["conceptos_omitidos"]["hacen_falta"]
   and t["conceptos_omitidos"]["reasuncion"] == "concesion" and not t["conceptos_omitidos"]["tenemos"],
   "la vía que revoca una concesión pide los conceptos que el juez no estudió")
ok(json.loads(json.dumps(t)) == t and td.clave_de(t) == td.clave_de(armar()),
   "JSON puro y estable (la marca no se reescribe si no cambia)")
_t0 = time.perf_counter()
armar()
ok(time.perf_counter() - _t0 < 1.0, "menos de un segundo")

print("\n5 · EL ESTADO, DETERMINISTA")
r_cl = copy.deepcopy(RESP)
r_cl["contraste"][0].update(la_combate=False, veredicto_previo="inoperante")
r_cl["global"]["alternativa"]["apoyos"] = ["170378", "2007402"]
t_cl = armar(r_cl)
ok(t_cl["estado"] == "claro" and t_cl["recomendada"] == "propuesta",
   "contraste que cierra en la dirección de la propuesta + criterio que obliga que la otra no "
   "invoca: claro y recomendada")
r_f = copy.deepcopy(RESP)
r_f["global"].update(sentido="fundado", razon=RAZON_ALT, apoyos=["168958", "170353"])
r_f["global"]["alternativa"] = {"sentido": "infundado", "razon": RAZON_MOTOR, "apoyos": ["188480"]}
ok(armar(r_f)["estado"] == "reñido",
   "una propuesta que hace prosperar el agravio no sale «claro» sin deliberación (sesgo a conceder)")
r_c = copy.deepcopy(r_cl)
r_c["global"].update(sentido="fundado", razon=RAZON_ALT)
r_c["global"]["alternativa"] = {"sentido": "infundado", "razon": RAZON_MOTOR, "apoyos": ["188480"]}
t_c = armar(r_c)
ok(t_c["estado"] == "reñido" and any("hace prosperar" in x for x in t_c["estado_por_que"]),
   "contraste «inoperante» y propuesta que prospera: reñido, y lo dice")
r_i = copy.deepcopy(RESP)
r_i["global"]["constancias"] = [{"que": "la escritura", "indispensable": True, "problema": 1}]
t_i = armar(r_i)
ok(t_i["estado"] == "no_alcanza" and t_i["que_la_cambiaria"]["constancias_indispensables"] == ["la escritura"],
   "falta una constancia indispensable: no alcanza")
r_n = copy.deepcopy(RESP)
r_n["global"]["apoyos"] = ["art. 49 CPC"]
r_n["global"]["alternativa"]["apoyos"] = ["7777777"]
ok(armar(r_n)["estado"] == "no_alcanza", "ninguna vía con un criterio verificado: no alcanza")
# LA LEY ES EL PRIMER FUNDAMENTO (revisión del 28-sep-2026): una vía fundada
# sólo en preceptos del material ya no se tiene por «no alcanza».
_m_n = copy.deepcopy(MATERIAL)
_m_n["normas"] = [{"cuerpo_legal": "Código Procesal Civil para el Estado de Querétaro",
                   "articulo": "49", "texto": "…"},
                  {"cuerpo_legal": "Código Civil del Estado de Querétaro", "articulo": "2294",
                   "texto": "…"}]
r_nn = copy.deepcopy(r_n)
r_nn["global"]["apoyos"] = ["art. 49 CPC Qro"]
r_nn["global"]["alternativa"]["apoyos"] = ["artículo 2294 del Código Civil local"]
t_nn = armar(r_nn, material=_m_n)
ok(t_nn["estado"] != "no_alcanza"
   and all(a["en_acervo"] is True for v in t_nn["vias"].values() for a in v["apoyos"]),
   f"dos vías fundadas en preceptos del material: alcanza ({t_nn['estado']})")
ok(td.norma_en_material("art. 49 CPC Qro", _m_n["normas"])
   and not td.norma_en_material("art. 49 del Código Civil", _m_n["normas"])
   and not td.norma_en_material("art. 49 de la Ley de Amparo", _m_n["normas"])
   and not td.norma_en_material("art. 50 CPC", _m_n["normas"]),
   "la norma casa por artículo y ley; la procesal no se confunde con la sustantiva")
_ap_d, _ = td.hidratar_apoyos([{"norma": "artículo 2294 del Código Civil del Estado de Querétaro",
                                "id": "N1", "en_acervo": True}], {}, TRIBUNAL)
ok(_ap_d[0]["en_acervo"] is True, "la norma del catálogo de la deliberación ya viene verificada")
ok(any("precepto verificado" in x for x in armar(r_n)["estado_por_que"]),
   "y la razón del «no alcanza» habla también de los preceptos")

print("\n6 · SIN PROPUESTA GLOBAL, SIN VÍA CONTRARIA REAL")
r_ng = copy.deepcopy(RESP)
r_ng["global"]["alcanza"] = False
t_ng = armar(r_ng)
ok(t_ng["vias"]["propuesta"]["sentido"] == "infundado" and t_ng["vias"]["opuesta"] is None
   and t_ng["estado"] == "no_alcanza",
   "sin global: el principal con su propuesta, sin columna contraria, no alcanza")
ok(t_ng["secundarios"][0]["en_opuesta"] is None, "y los secundarios sin la otra vía")
r_mismo = copy.deepcopy(RESP)
r_mismo["global"]["alternativa"]["sentido"] = "inoperante"
t_m = armar(r_mismo)
ok(t_m["vias"]["opuesta"] is None and any("mismo sentido" in a for a in t_m["avisos"]),
   "una «alternativa» en el mismo sentido no es vía contraria: columna apagada y aviso")
r_rz = copy.deepcopy(RESP)
r_rz["global"]["alternativa"]["razon"] = RAZON_ALT + " Lo sostiene la tesis (9999993)."
t_rz = armar(r_rz)
ok(t_rz["vias"]["opuesta"]["razon"] == r_rz["global"]["alternativa"]["razon"]
   and any("9999993" in a and "vía contraria" in a for a in t_rz["avisos"]),
   "la razón no se reescribe (viaja tal cual al resolver), pero su registro suelto se avisa")
r_sr = copy.deepcopy(RESP)
r_sr["global"]["alternativa"]["razon"] = ""
t_sr = armar(r_sr)
ok(t_sr["vias"]["opuesta"]["razon"] == "" and any("Redactar el criterio" in a for a in t_sr["avisos"]),
   "la contraria sin razón no hereda la del motor: vacía, y se ofrece redactarla con clic")

print("\n7 · LA JERARQUÍA Y EL MOTOR QUE DISCREPA")
r_d = copy.deepcopy(RESP)
r_d["global"]["checklist"][0]["papel"] = "accesorio"
r_d["global"]["checklist"][1]["papel"] = "principal"
t_d = armar(r_d)
ok(t_d["principal"]["numero"] == 1 and t_d["principal"]["discrepa_motor"]["numero_motor"] == 2,
   "el motor tomó el 2: se dice, no se elige en silencio")
ok(t_d["estado"] != "claro", "y con esa discrepancia no sale «claro»")
_pp = copy.deepcopy(PROBLEMAS)
_pp[0]["jerarquia_por_secretario"] = True
ok(armar(problemas=_pp)["principal"]["jerarquia_de"] == "secretario", "el principal que marcó él")
_pp2 = copy.deepcopy(PROBLEMAS)
for p in _pp2:
    p["jerarquia"] = ""
ok(armar(problemas=_pp2)["principal"]["jerarquia_de"] == "por_omision", "sin marca: el primero")

print("\n8 · LO QUE EL ÁRBOL DEJA POR RECALIFICAR SE ENSEÑA «PREVISTO»")
P3 = "¿Debió valorarse la pericial ofrecida por la quejosa?"
_prob3 = [{"pregunta": "¿Debió admitirse la ampliación de la demanda?", "jerarquia": "principal",
           "clase": "fondo"},
          {"pregunta": P3, "jerarquia": "accesorio", "clase": "fondo", "depende_de": 1}]
_chk3 = [{"numero": 1, "papel": "principal", "tema": "ampliación"},
         {"numero": 2, "papel": "accesorio", "tema": "pericial", "relacion": "depende",
          "si_prospera": {"sentido": "fundado", "razon": "admitida la ampliación, se valora"},
          "si_no_prospera": "Inoperante"}]
_resp3 = {"formato": 2, "contraste": [],
          "global": {"sentido": "fundado", "razon": "debió admitirse", "alcanza": True,
                     "apoyos": ["168958"], "checklist": _chk3,
                     "alternativa": {"sentido": "infundado", "razon": "la ampliación precluyó",
                                     "apoyos": ["170353"]}},
          "propuestas": [{"problema": _prob3[0]["pregunta"], "sentido": "fundado", "razon": "x",
                          "alcanza": True},
                         {"problema": P3, "sentido": "fundado", "razon": "se valora",
                          "alcanza": True}]}
t3 = armar(_resp3, problemas=_prob3, rama={"tipo_asunto": "amparo_directo", "tribunal": TRIBUNAL})
s3 = t3["secundarios"][0]
ok(s3["en_propuesta"]["sentido"] == "fundado" and not s3["en_propuesta"]["recalificar"],
   "en la vía del motor, su calificación")
ok(s3["en_opuesta"]["recalificar"] and s3["en_opuesta"]["previsto"]
   and s3["en_opuesta"]["sentido"] == "inoperante" and s3["en_opuesta"]["de"] == "motor",
   "en la contraria, lo que el motor escribió para esa vía, rotulado «previsto» y por recalificar")

print("\n9 · EL TEMA DISTINTO VA APARTE")
P4 = "¿Procedía la condena en costas?"
_prob4 = copy.deepcopy(PROBLEMAS) + [{"pregunta": P4, "jerarquia": "accesorio", "clase": "fondo"}]
_resp4 = copy.deepcopy(RESP)
_resp4["global"]["checklist"] = CHECKLIST + [{"numero": 3, "papel": "accesorio", "tema": "costas",
                                              "tema_distinto": True, "relacion": "distinto"}]
_resp4["propuestas"] = PROPUESTAS + [{"problema": P4, "sentido": "fundado",
                                      "razon": "no hubo temeridad", "alcanza": True,
                                      "apoyos": ["170378", "8888888"]}]
t4 = armar(_resp4, problemas=_prob4)
ok([s["numero"] for s in t4["secundarios"]] == [2] and len(t4["independientes"]) == 1
   and t4["independientes"][0]["numero"] == 3, "el 3 no es secundario: va en independientes")
ok(t4["independientes"][0]["propuesta"]["sentido"] == "fundado"
   and [a["registro"] for a in t4["independientes"][0]["propuesta"]["apoyos"]] == ["170378"],
   "con su propuesta propia y sus apoyos verificados")

print("\n10 · LOS CONCEPTOS OMITIDOS: LA FUNCIÓN DE SPEC B, LLAMADA DIRECTAMENTE")
# Antes era un gancho perezoso que se probaba con un doble. Con la función real
# el gancho le pasaba el sentido DEL MOTOR: en el 631 (motor «infundado», la vía
# que revoca es la contraria) contestaba que no hacía falta nada.
import types as _types
_om = armar()["conceptos_omitidos"]
ok(_om and _om["hacen_falta"] and _om["fundamento"].startswith("artículo 93, fracción VI"),
   "con el motor en «infundado», se pregunta por la VÍA CONTRARIA, que es la que revoca")
ok(armar(rama=dict(RAMA, quien_recurre="quejoso"))["conceptos_omitidos"] is None,
   "si recurre la propia quejosa, no (rige la fr. V)")
ok(armar(rama=dict(RAMA, resolvio_a_quo="niega"))["conceptos_omitidos"] is None,
   "si el juzgado negó, revocar no reasume nada que falte estudiar")
_om2 = armar(rama=dict(RAMA, conceptos_violacion="PRIMERO. Concepto aportado por el secretario."))
ok(_om2["conceptos_omitidos"]["tenemos"] and _om2["conceptos_omitidos"]["donde"] == "secretario",
   "los que aportó el secretario cuentan (el TEXTO, no un booleano)")
ok(armar(rama=dict(RAMA, conceptos_violacion=True))["conceptos_omitidos"]["tenemos"] is False,
   "un booleano no se lee como conceptos aportados")
_DEMANDA = ("DEMANDA DE AMPARO INDIRECTO\nCONCEPTOS DE VIOLACIÓN\nPRIMERO. La resolución reclamada "
            "altera la cosa juzgada. " + "Argumento. " * 70 + "\nSEGUNDO. No se valoraron las pruebas. "
            + "Argumento. " * 40 + "\nSUSPENSIÓN\nSe pide la suspensión.")
_om3 = armar(fases=_types.SimpleNamespace(fuentes=["", ""], autos=_DEMANDA))["conceptos_omitidos"]
ok(_om3["tenemos"] and _om3["donde"] == "constancias",
   "con las fases, los toma de la demanda que obra entre las constancias")
ok(armar(rama=dict(RAMA, tipo_asunto="amparo_directo"))["conceptos_omitidos"] is None,
   "en amparo directo no aplica")

print("\n11 · LA DELIBERACIÓN, SI EXISTE, SE PROYECTA SOBRE LAS VÍAS")
DELIB = {"huella": "h631", "estado": "listo", "resultado": {
    "principal": {"numero": 1, "pregunta_decisiva": "¿El adquirente puede sustituirse como "
                  "ejecutante sin cesión de los derechos litigiosos?", "figura": "causahabiencia "
                  "procesal", "proposicion_toral": {"dice": "no era causahabiente", "cita": "no "
                                                    "era causahabiente respecto del contrato"}},
    "recomendada": "A", "estado": "claro",
    "crux": {"que": "si la escritura transmitió los derechos del arrendamiento",
             "si_cambia": "la vía B", "constancia": "escritura"},
    "vias": {
        "A": {"sentido": "fundado", "razon": "razón del abogado A",
              "cadena": {"regla": "regla A", "hechos": [], "subsuncion": "s", "conclusion": "c"},
              "propongo_aplicar": [{"id": "T1", "registro": "170378"},
                                   {"id": "T2", "registro": "2026918"},
                                   {"id": "T9", "registro": "9999992"}],
              "objecion_de_la_otra_via": "objeción de B", "respuesta": "respuesta de A"},
        "B": {"sentido": "infundado", "razon": "razón del abogado B",
              "cadena": {"regla": "regla B"}, "propongo_aplicar": [{"id": "T3", "registro": "168958"}]}},
    "avisos": []}}
t_dl = armar(deliberacion=DELIB)
ok(t_dl["vias"]["propuesta"]["sentido"] == "fundado" and t_dl["vias"]["opuesta"]["sentido"] == "infundado",
   "la vía que recomienda el juez va como «propuesta», aunque el motor propusiera la otra")
ok(t_dl["vias"]["propuesta"]["razon"] == "razón del abogado A"
   and t_dl["vias"]["opuesta"]["razon"] == "razón del abogado B", "cada una con la razón de su abogado")
ok(t_dl["vias"]["propuesta"]["cadena"]["regla"] == "regla A"
   and t_dl["vias"]["propuesta"]["objecion"]["respuesta"] == "respuesta de A",
   "con su cadena y la objeción de la otra vía")
ok([a["registro"] for a in t_dl["vias"]["propuesta"]["apoyos"]] == ["170378", "2026918"]
   and any("9999992" in a for a in t_dl["avisos"]),
   "sus apoyos, verificados otra vez: el registro fuera del catálogo se quita")
ok(t_dl["estado"] == "claro" and t_dl["recomendada"] == "propuesta"
   and t_dl["que_la_cambiaria"]["crux"]["constancia"] == "escritura",
   "el estado y el crux del juez")
ok(t_dl["deliberacion"]["origen"] == "deliberacion" and t_dl["deliberacion"]["figura"],
   "la pregunta decisiva y la figura, a la vista")
ok(t_dl["secundarios"][0]["en_propuesta"]["sentido"] == "innecesario",
   "los secundarios siguen saliendo del árbol, en la vía de cada columna")
_d2 = copy.deepcopy(DELIB)
_d2["resultado"]["estado"] = "reñido"
ok(armar(deliberacion=_d2)["recomendada"] is None, "reñido: ninguna se rotula recomendada")
r_ind = copy.deepcopy(RESP)
r_ind["global"]["constancias"] = [{"que": "la escritura", "indispensable": True}]
ok(armar(r_ind, deliberacion=DELIB)["estado"] == "no_alcanza",
   "sin la constancia indispensable no alcanza, diga lo que diga el juez")
_d3 = copy.deepcopy(DELIB)
_d3["estado"] = "en_curso"
ok(armar(deliberacion=_d3)["deliberacion"] is None
   and armar(deliberacion=_d3)["vias"]["propuesta"]["sentido"] == "infundado",
   "una deliberación en curso no se usa: todo sale de la propuesta")

print("\n12 · QUÉ MARCAS SE LEEN")
E = td.elegir_marcas
_prop = {"huella": "h631", "estado": "listo", "respuesta": copy.deepcopy(RESP)}
ok(E({"propuesta": _prop}, "h631")["estado_calculo"] == "listo", "la propuesta de este adelanto")
ok(E({"propuesta": _prop}, "otra")["estado_calculo"] == "sin_propuesta",
   "la de otro adelanto no sirve")
_ahora = time.time()
ok(E({"propuesta": {"huella": "h631", "estado": "en_curso", "desde": _ahora - 10}}, "h631",
     ahora=_ahora)["estado_calculo"] == "calculando", "si corre y late: calculando")
ok(E({"propuesta": {"huella": "h631", "estado": "en_curso", "desde": _ahora - 900}}, "h631",
     ahora=_ahora)["estado_calculo"] == "sin_propuesta", "si murió sin latido: sin propuesta")
_gp = {"huella": "h631", "global": dict(GLOBAL, sentido="fundado")}
_s = E({"propuesta": _prop, "global_propuesta": _gp}, "h631")
ok(_s["propuesta"]["global"]["sentido"] == "fundado",
   "la global que vio la pantalla por última vez manda")
_s2 = E({"global_propuesta": _gp, "contraste": {"huella": "h631", "estado": "listo",
                                                  "items": CONTRASTE}}, "h631",
        propuestas_fila=[{"problema": P1, "sentido": "fundado"}])
ok(_s2["estado_calculo"] == "listo" and _s2["contraste"] == CONTRASTE
   and _s2["propuesta"]["propuestas"][0]["sentido"] == "fundado",
   "sin la marca de la propuesta: la global, las propuestas de la fila y el contraste adelantado")
ok(E({"propuesta": _prop, "deliberacion": dict(DELIB, huella="otra")}, "h631")["deliberacion"] is None,
   "la deliberación de otro adelanto no se usa")
ok(td.vacia("calculando", "h")["vias"] == {"propuesta": None, "opuesta": None},
   "la tarjeta vacía tiene la forma del contrato")

print("\n13 · EL ENDPOINT, SIN IMPORTAR main (se lee el fuente)")
_src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "main.py"), encoding="utf8").read()
ok('@app.get("/taller/tarjeta")\nasync def taller_tarjeta(numero: str, user_email: str):' in _src,
   "GET /taller/tarjeta colgado de taller_tarjeta")
_cuerpo = _src.split("async def taller_tarjeta", 1)[1].split("\n@app.", 1)[0]
# REVISIÓN DEL 28-sep-2026: la marca «tarjeta» reescribía el `estado` entero sin
# control de versión en cada GET y podía pisar la de la deliberación.
ok("_taller_puerta(user_email)" in _cuerpo and "_td.armar(" in _cuerpo
   and "_taller_guardar_marca(" not in _cuerpo,
   "con la puerta de /taller/proponer, arma con el módulo y NO reescribe el estado")
ok("chat_client" not in _cuerpo and "_f5." not in _cuerpo and "proponer(" not in _cuerpo,
   "sin ningún cliente de modelo")
ok('"internet": (ses.get("internet")' in _src, "la propuesta guarda la línea de internet (pistas)")
ok('p["jerarquia_por_secretario"] = True' in _src, "/taller/problema marca quién fijó la jerarquía")

print("\n13 bis · LA REVISIÓN ADVERSARIAL DEL 28-sep-2026")
# (a) Ningún porcentaje junto al principal: la jurimetría no va en la tarjeta.
#     (2-oct-2026) La regla queda SÓLO con la bandera «propuesta_por_probabilidad»
#     apagada; encendida, la tarjeta enseña la probabilidad de la propuesta
#     (contrato E, sección 15).
r_pr = copy.deepcopy(RESP)
r_pr["propuestas"][0]["prediccion"] = {"frase": "Infundado (75% de 12 sentencias del acervo)", "n": 12}
t_pr = armar(r_pr)
ok(t_pr["principal"]["prediccion"] is None and "%" not in json.dumps(t_pr, ensure_ascii=False),
   "la frase de la jurimetría con su porcentaje no se proyecta en la tarjeta")
# (b) La vía protectora sólo a favor de la persona: si recurre la tercera, el
#     «fundado» le quita el amparo a la quejosa (la 462).
t_vp = armar(rama=dict(RAMA, quien_recurre="tercero"))
ok(t_vp["vias"]["opuesta"]["via_protectora"] is None
   and any("lectura protectora" in a and "no favorece" in a for a in t_vp["avisos"]),
   "recurre la tercera: la lectura protectora no se cuelga de la vía que perjudica a la quejosa")
ok(armar(rama=dict(RAMA, quien_recurre="quejoso"))["vias"]["opuesta"]["via_protectora"] is not None,
   "recurre la quejosa: el «fundado» la favorece y la lectura protectora sí va")
# (c) El efecto del motor que es un reenvío choca con la reasunción (fr. VI).
r_ef = copy.deepcopy(RESP)
r_ef["global"]["alternativa"]["efecto"] = ("La revocación exigiría una nueva decisión sobre la "
                                           "legitimación del adquirente.")
t_ef = armar(r_ef)
ok(t_ef["vias"]["opuesta"]["efecto"] == "" and "nueva decisión" in t_ef["vias"]["opuesta"]["efecto_motor"]
   and any("sin reenvío" in a for a in t_ef["avisos"]),
   "el efecto de reenvío no se enseña junto a la nota de la reasunción: se guarda y se avisa")
ok(armar()["vias"]["opuesta"]["efecto"] == GLOBAL["alternativa"]["efecto"],
   "el efecto que no choca se enseña como siempre")
# (d) Con «reñido», las razones hablan de la vía A y la vía B, como la pantalla,
#     y no dicen «la contraria no» cuando las dos comparten un criterio.
_rz = armar()["estado_por_que"]
ok(not any("la propuesta" in x.lower() or "la contraria no" in x.lower() for x in _rz)
   and any("vía A" in x for x in _rz) and any("comparten 2026918" in x for x in _rz),
   f"reñido: vía A / vía B y lo que comparten: {_rz}")
ok(any("la propuesta" in x.lower() for x in armar(r_cl)["estado_por_que"]),
   "con «claro» sí se llama la propuesta, como en pantalla")
# (e) No consta quién recurre: se dice.
ok(any("No consta quién recurre" in a for a in armar(rama=dict(RAMA, quien_recurre=""))["avisos"])
   and not any("No consta quién recurre" in a for a in armar()["avisos"]),
   "si `info_de_rama` no sabe quién recurre, la tarjeta lo dice")
# (f) Conceptos «por confirmar»: la recurrida es legible y no dice que quedaran
#     conceptos sin estudiar.
_REC_TODO = ("SENTENCIA. CONSIDERANDO SEXTO. Es fundado el segundo concepto de violación. "
             + "Estudio del concepto. " * 60 + " Son infundados el primero y el tercero. "
             + "Estudio de los otros dos. " * 30)
_om_pc = armar(fases=_types.SimpleNamespace(fuentes=[_REC_TODO, ""], autos=""))
ok(_om_pc["conceptos_omitidos"]["hacen_falta"] == "por_confirmar"
   and any("no dice que el juzgado dejara conceptos sin estudiar" in a for a in _om_pc["avisos"]),
   "el juzgado estudió todos los conceptos: «por confirmar», sin bloquear, y se avisa")
_REC_INN = _REC_TODO + " Por tanto, resulta innecesario el estudio de los restantes conceptos."
ok(armar(fases=_types.SimpleNamespace(fuentes=[_REC_INN, ""], autos=""))
   ["conceptos_omitidos"]["hacen_falta"] is True,
   "si la recurrida declaró innecesario el estudio de los restantes, hacen falta")
# (g) Si el principal es de procedencia, la vía que lo hace prosperar sobresee.
_pp_pro = copy.deepcopy(PROBLEMAS)
_pp_pro[0]["clase"] = "procedencia"
t_pro = armar(problemas=_pp_pro, rama=dict(RAMA, quien_recurre="tercero"))
ok(any("Se sobresee" in x for x in t_pro["vias"]["opuesta"]["desenlace"])
   and t_pro["conceptos_omitidos"] is None,
   "principal de procedencia: la vía que prospera revoca y sobresee, y no pide conceptos")

# ═══ LA FILA 462 REAL, SI SE PASA (sólo lectura, fuera del repositorio) ══════
_ruta = os.environ.get("TARJETA_FILA_462", "")
if _ruta and os.path.exists(_ruta):
    print("\n14 · LA FILA 462 VOLCADA (AR 631/2025)")
    import types as _types
    import fase_rama as _fr          # faltaba: la sección opcional caía en NameError
    d = json.load(open(_ruta, encoding="utf8"))
    fila = {"propuesta": d["propuesta"], "global_propuesta": d["global_propuesta"],
            "contraste": d["contraste"]}
    sel = td.elegir_marcas(fila, d["huella"], d["propuestas_fila"])
    g = sel["propuesta"]["global"]
    a_quo = _fr.que_hizo_el_juzgado(_types.SimpleNamespace(**d["fases"]),
                                    declarado=(g.get("contexto") or {}).get("resolvio", ""))
    ok(a_quo == "concede", f"el juzgado concedió (su resolutivo manda, 577c700): {a_quo}")
    t462 = td.armar(sel["propuesta"], d["material"], d["fases"]["problemas"], sel["contraste"],
                    None, None, {"tipo_asunto": d["encargo"]["tipo_asunto"],
                                 "resolvio_a_quo": a_quo, "tribunal": d["encargo"]["tribunal"],
                                 "huella": d["huella"]},
                    propuestas_motor=d["propuestas_fila"])
    ok(t462["principal"]["numero"] == 1 and t462["principal"]["discrepa_motor"] is None,
       "el principal es P1 y el motor coincide")
    vo = t462["vias"]["opuesta"]
    ok(vo and vo["sentido"] == "fundado" and "93, fr. VI" in (vo["desenlace_nota"] or ""),
       "la vía contraria revoca y anuncia el estudio de los conceptos omitidos")
    _cat = td.catalogo_de(d["material"])
    ok(all(a["registro"] in _cat for v in t462["vias"].values() if v
           for a in v["apoyos"] if a["registro"]), "0 registros fuera del acervo")
    ok(t462["estado"] != "claro" and t462["recomendada"] is None,
       f"el motor propuso infundado y no se le rotula recomendada ({t462['estado']})")
    _rplan = os.environ.get("TARJETA_PLAN_462", "")
    if _rplan and os.path.exists(_rplan):
        planes = json.load(open(_rplan, encoding="utf8"))["planes"]
        s2 = t462["secundarios"][0]
        for _k, _v in planes.items():
            _pr = {p["id"]: p for p in _v["plan"]["problemas"]}
            _via = "en_propuesta" if _pr[1]["sentido"] == t462["vias"]["propuesta"]["sentido"] \
                else "en_opuesta"
            ok(s2[_via]["sentido"] == _pr[2]["sentido"],
               f"plan {_k[:8]}: con P1 {_pr[1]['sentido']}, P2 {s2[_via]['sentido']} "
               f"(el plan de producción: {_pr[2]['sentido']})")
else:
    print("\n14 · (sin TARJETA_FILA_462: la fila real no se prueba aquí)")

print("\n15 · LA PROPUESTA POR PROBABILIDAD (2-oct-2026, contrato E)")
# David: «el motor nunca se atreve a proponer (…) si hay un 50.01% de
# probabilidad hacia un lado sea esa la propuesta». Con la bandera, la
# recomendación es siempre la propuesta y el estado queda como certeza.
ok("probabilidad" not in armar(), "apagada: la tarjeta no trae probabilidad (como ayer)")
os.environ["PROPUESTA_POR_PROBABILIDAD"] = "todos"
try:
    t15 = armar()
    ok(t15["estado"] == "reñido" and t15["recomendada"] == "propuesta",
       "reñido sigue siendo el grado de certeza, pero la propuesta se recomienda")
    ok(not any("vía A" in x for x in t15["estado_por_que"]),
       "y las razones la llaman «la propuesta», no «la vía A»")
    ok(t15["probabilidad"] is None, "sin probabilidad en la propuesta guardada: null")
    r15 = copy.deepcopy(RESP)
    r15["formato"] = 3
    r15["global"]["probabilidad"] = {"p_prospera": 0.324, "lado": "no_prospera",
                                     "explicacion": "Se propone que no prospere, con una probabilidad del 68%."}
    t15p = armar(r15)
    ok(t15p["probabilidad"] == {"p": 0.676, "lado": "no_prospera",
                                "explicacion": "Se propone que no prospere, con una probabilidad del 68%."},
       f"la probabilidad del lado propuesto: {t15p['probabilidad']}")
    r15i = copy.deepcopy(RESP)
    r15i["global"]["constancias"] = [{"que": "la escritura", "indispensable": True, "problema": 1}]
    t15i = armar(r15i)
    ok(t15i["estado"] != "no_alcanza" and t15i["recomendada"] == "propuesta"
       and t15i["que_la_cambiaria"]["constancias_indispensables"] == ["la escritura"],
       "una constancia indispensable ya no pone «no_alcanza»: se sigue diciendo en lo que la cambiaría")
    r15n = copy.deepcopy(RESP)
    r15n["global"] = {}
    t15n = armar(r15n)
    ok(t15n["recomendada"] == "propuesta" and t15n["vias"]["propuesta"]["sentido"] == "infundado",
       "sin global, la propuesta del principal también se recomienda")
    r15v = copy.deepcopy(RESP)
    r15v["global"] = {}
    r15v["propuestas"] = [dict(PROPUESTAS[0], sentido=""), PROPUESTAS[1]]
    ok(armar(r15v)["recomendada"] is None, "sin ningún sentido no hay qué recomendar")
    # La marca de formato 3 se usa; la de «preguntas» no arma tarjeta.
    _f3 = {"propuesta": {"huella": "h", "estado": "listo", "respuesta": dict(r15)}}
    ok(td.elegir_marcas(_f3, "h")["estado_calculo"] == "listo", "la propuesta de formato 3 se lee")
    _fq = {"propuesta": {"huella": "h", "estado": "listo",
                         "respuesta": {"formato": 3, "estado": "preguntas", "propuestas": [],
                                       "global": None, "preguntas": [{"id": "P1"}]}},
           "global_propuesta": {"huella": "h", "global": GLOBAL}}
    ok(td.elegir_marcas(_fq, "h")["estado_calculo"] == "sin_propuesta",
       "con preguntas pendientes no hay tarjeta, aunque quede una global de antes")
finally:
    os.environ.pop("PROPUESTA_POR_PROBABILIDAD", None)
_f2 = {"propuesta": {"huella": "h", "estado": "listo", "respuesta": dict(RESP)}}
ok(td.elegir_marcas(_f2, "h")["estado_calculo"] == "listo", "la de formato 2, como siempre")

print()
if FALLOS:
    print(f"FALLAN {len(FALLOS)}: " + " · ".join(FALLOS))
    raise SystemExit(1)
print("RESULTADO: TODAS LAS COMPROBACIONES PASAN")
