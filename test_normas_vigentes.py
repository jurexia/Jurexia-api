"""LAS NORMAS Y ACUERDOS DEL ADELANTO, CONTRA EL TEXTO VIGENTE (27-sep-2026).

David: «revisar que los acuerdos del adelanto y normas sean las correctas en los
cuatro tipos de asuntos… investiga si las normas y acuerdos son los correctos y
vigentes». Cinco frentes verificados contra el DOF y la Cámara de Diputados
(Ley de Amparo, última reforma DOF 16-10-2025; LOPJF DOF 20-12-2024, reforma
28-11-2025; LFPCA reforma DOF 09-06-2026; CNPCF; acuerdos de la SCJN y del OAJ).
Cada comprobación de aquí es una cita que estaba mal y ya no puede volver.
"""
import datetime as dt
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import banco
import documento_generado as dg
import fase0_oportunidad as f0
import fase_procedencia_rf as pf
import tipos_asunto as ta

FALLAS = []


def ok(cond, que):
    print(("   PASA   " if cond else "   FALLA  ") + que)
    if not cond:
        FALLAS.append(que)


_B = json.dumps(json.load(open(os.path.join(os.path.dirname(__file__), "banco_plantillas.json"),
                               encoding="utf-8")), ensure_ascii=False)

print("\n1 · LEY ORGÁNICA DEL PODER JUDICIAL DE LA FEDERACIÓN (DOF 20-12-2024)")
ok("41, fracción II" not in _B and "24, fracción II, de la Ley Orgánica" in _B,
   "el turno se funda en el 24, fracción II (el 41 es hoy la integración de los Juzgados de Distrito)")
ok("124 de la Ley Orgánica" not in _B, "ninguna plantilla cita el 124 (circuitos en la ley de 2021, abrogada)")
for t in ta.TIPOS:
    c = ta.cadena_competencia(t)
    ok("37 de la" not in c and "210" in c, f"{t}: la cadena de competencia del prompt no cita el 37 y sí el 210")
ok("35, fracción I," in ta.cadena_competencia("amparo_directo")
   and "35, fracción V" in ta.cadena_competencia("amparo_revision")
   and "35, fracción III" in ta.cadena_competencia("queja")
   and "35, fracción VI" in ta.cadena_competencia("revision_fiscal")
   and "Ley de Amparo" not in ta.cadena_competencia("revision_fiscal").replace("nunca la Ley de Amparo", ""),
   "cada tipo con su fracción del 35 (I, V, III, VI); la revisión fiscal sin la Ley de Amparo")
_p = dg.prompt_estructura({"tipo_asunto": "queja"})
ok("37 de la Ley" not in _p and "35, fracción III" in _p, "el prompt de la estructura de una queja ya no da la cadena del amparo directo")

_c_rv, _ = banco.texto_de("amparo_revision", "competencia", {"materia": "administrativa"})
_n, _a = ta.competencia_revision(_c_rv, "V I S T O S para resolver el incidente de suspensión relativo "
                                        "al juicio de amparo 123/2026")
ok("inciso a), y 84" in _n and "35, fracción II y 210" in _n and "incidente de suspensión" in _n and _a,
   "revisión de la suspensión definitiva: 81-I-a) y 35-II, no los de la sentencia de audiencia")
_n2, _a2 = ta.competencia_revision(_c_rv, "VISTOS para resolver los autos del juicio de amparo; audiencia "
                                          "constitucional", "reclamó la expedición y promulgación de la Ley de "
                                          "Ingresos del Municipio de Querétaro")
ok(_n2 == _c_rv and any("2/2025 (12a.)" in x for x in _a2),
   "constitucionalidad: la fórmula no se toca a ciegas y se avisa de la delegación (AG 2/2025 y 11/2025, 12a.)")

print("\n1b · ACUERDOS GENERALES DEL CJF Y DEL OAJ")
_datos = {"fraccion_acuerdo": "XXII", "materia": "civil", "inciso": "c"}
for _tipo in ta.TIPOS:
    _t3, _ = banco.texto_de(_tipo, "competencia", dict(_datos, tribunal="Tercer Tribunal Colegiado en Materias "
                                                            "Administrativa y Civil del Vigésimo Segundo Circuito"))
    _t1, _ = banco.texto_de(_tipo, "competencia", dict(_datos, tribunal="Primer Tribunal Colegiado en Materias "
                                                            "Administrativa y Civil del Vigésimo Segundo Circuito"))
    ok("28/2017" in _t3 and "28/2017" not in _t1 and "Acuerdo General 3/2013 del Pleno del otrora" in _t1,
       f"{_tipo}: el 28/2017 (creación del Tercer TCC) sólo para el Tercero; el 3/2013 para todos")
ok("artículo 4°" not in _B and "a través del oficio SEADS" not in _B, "ni el art. 4° del 28/2017 (reparto de 2017) ni un oficio de adscripción fijo en la plantilla")
ok("12/2020" in _B and "abrogó el 12/2020" in _B, "el 6/2026 del OAJ abrogó el 12/2020: la cola COVID queda para lo anterior")

print("\n2 · CONSTITUCIÓN")
ok("I-B" not in ta.TECNICA_RESOLUCION["revision_fiscal_reenvio"]["fuente"]
   and "104, fracción III" in ta.TECNICA_RESOLUCION["revision_fiscal_reenvio"]["fuente"],
   "la revisión fiscal se funda en el 104, fracción III (la I-B no existe desde 2011)")

print("\n3 · LEY DE AMPARO: SUPLETORIO, LEGITIMACIÓN, INCISOS Y PLAZOS")
ok("Código Federal de Procedimientos Civiles" not in _B
   and "Código Nacional de Procedimientos Civiles y Familiares" in _B,
   "la existencia del acto se funda en el Código Nacional (art. 2o. LA, reforma DOF 13-03-2025)")
_e, _ = banco.texto_de("amparo_directo", "existencia", {})
ok("312, fracciones II y VIII, y 344 del Código Nacional" in _e, "amparo directo: 312-II, 312-VIII y 344 del CNPCF")
ok("artículo 5o." in ta.LEGITIMACION["queja"]["molde"] and "97" not in ta.LEGITIMACION["queja"]["molde"],
   "queja: legitima ser parte (art. 5o.), no el 97, que es la procedencia")
ok(ta.inciso_97("auto que resolvió el incidente de reposición de autos") != "e",
   "la reposición de autos no es el inciso e) del 97 (se recurre en revisión, 81-I-c)")
ok(ta.inciso_97("auto que desechó el incidente de nulidad de notificaciones") == "e",
   "la nulidad de notificaciones sigue en el inciso e)")
ok("autoaplicativa" not in [x["clave"] for x in ta.excepciones_de("amparo_directo")],
   "la norma autoaplicativa (30 días) no es excepción del amparo directo")
ok(ta.plazo_de("amparo_directo")["dias"] == 15 and ta.plazo_de("amparo_revision")["dias"] == 10
   and ta.plazo_de("queja")["dias"] == 5 and ta.plazo_de("queja", "suspension")["dias"] == 2,
   "los plazos de 15, 10, 5 y 2 días siguen como los dice la ley")
for fecha in (dt.date(2027, 2, 5), dt.date(2027, 3, 21), dt.date(2027, 11, 20)):
    ok(not f0.CALENDARIO_AMPARO.es_habil(fecha),
       f"art. 19: el {fecha.isoformat()} es inhábil aunque la OAJ no haya publicado aún su lista")
ok(ta.APERTURA_EFECTOS.startswith("Con fundamento en el artículo 77 de la Ley de Amparo"),
   "los efectos se fundan en el 77 (el 93 son las reglas de la revisión)")

print("\n4 · REVISIÓN FISCAL (LFPCA, reforma DOF 09-06-2026)")
_pr, _faltan = banco.texto_de("revision_fiscal", "procedencia", {})
ok("fracción VI" not in _pr and "fraccion_63" in _faltan,
   "la procedencia ya no fija la fracción VI para todos: la fracción va en hueco si no se motiva")
ok(pf.UMA_DIARIA.get(2026) == 117.31, "UMA 2026: 117.31 (INEGI, DOF 09-01-2026)")
ok(pf.uma_de(dt.date(2026, 1, 20)) == (113.14, 2025), "en enero rige la UMA del año anterior")
ok(pf.veces_de(dt.date(2026, 6, 10)) == 27000 and pf.veces_de(dt.date(2026, 6, 9)) == 3500,
   "27,000 UMA para sentencias dictadas desde el 10-06-2026; 3,500 antes")
_t = "la resolución determinó un crédito fiscal por $1,500,000.00 a cargo de la actora"
p_ant, _ = pf.parrafo(_t, fecha_sentencia=dt.date(2026, 3, 2))
p_nue, a_nue = pf.parrafo(_t, fecha_sentencia=dt.date(2026, 7, 1))
ok("fracción I" in p_ant and "tres mil quinientas" in p_ant, "1.5 millones en marzo de 2026: procede por la fracción I")
ok("NO excede de veintisiete mil" in p_nue and any("NO ALCANZA" in a for a in a_nue),
   "1.5 millones en julio de 2026: no alcanza (27,000 × 117.31 = 3,167,370) y se avisa")
_, a_fr = pf.parrafo(_t, fecha_sentencia=dt.date(2026, 6, 1), fecha_interposicion=dt.date(2026, 6, 20))
ok(any("SENTENCIA ANTERIOR A LA REFORMA" in a for a in a_fr), "sentencia anterior y recurso posterior: aviso de frontera")
p_vi, _ = pf.parrafo("El Instituto Mexicano del Seguro Social determinó la prima de riesgos de trabajo",
                     fecha_sentencia=dt.date(2026, 7, 1))
ok("fracción VI" in p_vi and "relativa al grado de riesgo" in p_vi,
   "aportaciones de seguridad social: fracción VI con su porción")
p_no, a_no = pf.parrafo("resolución que negó una devolución", fecha_sentencia=dt.date(2026, 7, 1))
ok(p_no == "" and a_no, "sin cuantía ni supuesto reconocible: no se afirma fracción y se avisa")
ok(pf.fecha_de_letra("veintidós de septiembre de dos mil veinticinco") == dt.date(2025, 9, 22)
   and pf.fecha_de_letra("primero de julio de dos mil veintiséis") == dt.date(2026, 7, 1),
   "la fecha de la sentencia se lee en letra")

for _n, _dias, _txt in ((dt.date(2025, 12, 5), 3, ""), (dt.date(2026, 7, 6), 3, "texto anterior"),
                        (dt.date(2027, 2, 8), 2, "")):
    _c = f0.computar(_n, _n + dt.timedelta(days=20), plazo=15, regla="lfpca_boletin")
    ok(_c.regla.dias_habiles == _dias and _txt in _c.regla.fundamento,
       f"Boletín del TFJA notificado el {_n.isoformat()}: surte al {_dias}º día hábil (art. 65 LFPCA y su Tercero transitorio)")

print()
if FALLAS:
    print(f"FALLAN {len(FALLAS)}: " + " · ".join(FALLAS))
    sys.exit(1)
print("RESULTADO: TODAS LAS COMPROBACIONES PASAN")
