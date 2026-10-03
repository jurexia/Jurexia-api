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
for _grafia in ("Tercer Tribunal Colegiado en Materia Administrativa y Civil del Vigésimo Segundo Circuito",
                "Tercer Tribunal Colegiado en Materias Administrativa y Civil del XXII Circuito",
                "Tercer Tribunal Colegiado de Circuito en Materias Administrativa y Civil, Querétaro"):
    _tg, _ = banco.texto_de("amparo_directo", "competencia", dict(_datos, tribunal=_grafia))
    ok("28/2017" in _tg, f"otra grafía del Tercero conserva su 28/2017: «{_grafia[:50]}…»")
_c_aud = ("SENTENCIA. Querétaro. VISTOS para resolver los autos del juicio de amparo 55/2026; RESULTANDO: "
          "... ordenó tramitar por cuerda separada el incidente de suspensión relativo ...")
_na, _aa = ta.competencia_revision(_c_rv, _c_aud)
ok(_na == _c_rv, "una sentencia de audiencia que menciona el incidente de suspensión no pasa por interlocutoria")
ok("artículo 4°" not in _B and "a través del oficio SEADS" not in _B, "ni el art. 4° del 28/2017 (reparto de 2017) ni un oficio de adscripción fijo en la plantilla")
ok("12/2020" in _B and "abrogó el 12/2020" in _B, "el 6/2026 del OAJ abrogó el 12/2020: la cola COVID queda para lo anterior")

print("\n2 · CONSTITUCIÓN")
ok("I-B" not in ta.TECNICA_RESOLUCION["revision_fiscal_reenvio"]["fuente"]
   and "104, fracción III" in ta.TECNICA_RESOLUCION["revision_fiscal_reenvio"]["fuente"],
   "la revisión fiscal se funda en el 104, fracción III (la I-B no existe desde 2011)")

print("\n3 · LEY DE AMPARO: SUPLETORIO, LEGITIMACIÓN, INCISOS Y PLAZOS")
# EL SUPLETORIO DEPENDE DE LA SEDE (David, 3-oct-2026): «aplicar, por ahora en
# toda la república, el código federal de procedimientos civiles de manera
# supletoria a la ley de amparo, salvo en los proyectos de ciudad de mexico
# donde ya opera el CNPCF». Hasta hoy esta prueba EXIGÍA el Nacional en toda la
# república (27-sep-2026) y así salieron 18 de 18 proyectos de octubre.
_ex_ad = banco.apartado("amparo_directo", "existencia")
_civil = banco.variante_de("amparo_directo", "existencia", "ad-c2-civil")
ok("{supletorio_documentales}" in _ex_ad.get("plantilla", "") and "{supletorio_documentales}" in _civil
   and "Código" not in _ex_ad.get("plantilla", "") and "Código" not in _civil,
   "la plantilla de la existencia ya no trae escrito el código: lo pone {supletorio_documentales}")
_fuera = ta.supletorio("Tercer Tribunal Colegiado en Materias Administrativa y Civil del Vigésimo "
                       "Segundo Circuito", "Querétaro, Querétaro")
_cdmx = ta.supletorio("Décimo Tribunal Colegiado en Materia Civil del Primer Circuito", "Ciudad de México")
_e, _ = banco.texto_de("amparo_directo", "existencia", {"supletorio_documentales": _fuera["documentales"]})
ok("artículos 129 y 202 del Código Federal de Procedimientos Civiles, de aplicación supletoria a la Ley "
   "de Amparo" in _e and "Código Nacional" not in _e and "conforme a su artículo 2o." not in _e,
   "fuera de la Ciudad de México: 129 y 202 del CFPC, SIN «conforme a su artículo 2o.» (el 2o. vigente nombra al CNPCF)")
_e, _ = banco.texto_de("amparo_directo", "existencia", {"supletorio_documentales": _cdmx["documentales"]})
ok("312, fracciones II y VIII, y 344 del Código Nacional de Procedimientos Civiles y Familiares, de "
   "aplicación supletoria a la Ley de Amparo conforme a su artículo 2o." in _e,
   "en la Ciudad de México: 312-II, 312-VIII y 344 del CNPCF, conforme al artículo 2o. de la Ley de Amparo")
ok(_fuera["cdmx"] is False and _cdmx["cdmx"] is True
   and "88 del Código Federal" in _fuera["hecho_notorio"] and "269 del Código Nacional" in _cdmx["hecho_notorio"]
   and "210-A del Código Federal" in _fuera["electronica"] and "348 del Código Nacional" in _cdmx["electronica"],
   "hecho notorio (CFPC 88 / CNPCF 269) e información electrónica (CFPC 210-A / CNPCF 348) por sede")
ok(ta.SUPLETORIO_DOCUMENTALES == _fuera["documentales"],
   "SUPLETORIO_DOCUMENTALES queda como alias de la regla general (fuera de la Ciudad de México)")
ok(ta.supletorio("Primer Tribunal Colegiado en Materia Administrativa del Primer Circuito")["cdmx"]
   and not ta.supletorio("Primer Tribunal Colegiado del Centro Auxiliar de la Primera Región",
                         "Naucalpan, Estado de México")["cdmx"]
   and ta.supletorio("Primer Tribunal Colegiado del Centro Auxiliar de la Primera Región", "Ciudad de México")["cdmx"],
   "la sede: Primer Circuito = Ciudad de México; un Centro Auxiliar, por la ciudad donde reside")
_ns = ta.supletorio("", "")
ok(_ns["cdmx"] is False and "Código Federal" in _ns["documentales"] and _ns["aviso"],
   "sin ciudad ni circuito: la regla general (CFPC) y un aviso, no la excepción")
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

print("\n4b · EL CÓMPUTO DE LA REVISIÓN FISCAL, CON EL CALENDARIO DEL TFJA")
_rf = f0.computar(dt.date(2026, 1, 30), dt.date(2026, 2, 27), regla="lfpca_boletin", plazo=15,
                  responsable="Sala Regional en Querétaro del Tribunal Federal de Justicia Administrativa",
                  tipo_asunto="revision_fiscal")
ok(f0.CALENDARIO_TFJA.es_habil(dt.date(2026, 2, 5)) and not f0.CALENDARIO_AMPARO.es_habil(dt.date(2026, 2, 5)),
   "el 5 de febrero de 2026 es inhábil para el PJF (art. 19) pero hábil para el TFJA")
ok(not f0.CALENDARIO_TFJA.es_habil(dt.date(2025, 12, 15)) and f0.CALENDARIO_AMPARO.es_habil(dt.date(2025, 12, 15)),
   "el 15 de diciembre de 2025 es vacación del TFJA y hábil para el PJF")
ok(_rf.surtio == dt.date(2026, 2, 5) and dt.date(2026, 2, 23) in _rf.inhabiles_en_medio,
   "boletín del 30-ene-2026: surte el 5-feb (hábil para el TFJA) y descuenta la suspensión de la Sala de Querétaro (SRQ/01/2026)")
_prf = f0.parrafo_oportunidad(_rf, "", "revision_fiscal")
ok("artículo 74, fracción II, de la Ley Federal de Procedimiento Contencioso Administrativo" in _prf
   and "SS/2/2026" in _prf and "artículo 19 de la Ley de Amparo" not in _prf,
   "el considerando funda los inhábiles en el 74-II de la LFPCA y el acuerdo del TFJA, no en el 19 de la LA")
_ad = f0.computar(dt.date(2026, 1, 23), dt.date(2026, 2, 20), regla="lfpca_boletin", plazo=15,
                  responsable="Sala Regional", tipo_asunto="amparo_directo")
ok(dt.date(2026, 2, 5) in _ad.inhabiles_en_medio, "el amparo directo sigue con el calendario del artículo 19")

print("\n5 · PLAZOS DE AÑOS DEL ARTÍCULO 17 (fracciones II y III)")
ok(ta.anios_de("amparo_directo", "penal_prision") == 8 and ta.anios_de("amparo_directo", "agrario_nucleo") == 7
   and ta.anios_de("amparo_directo", "") == 0, "ocho años (fr. II) y siete (fr. III); el plazo ordinario no es de años")
_c8 = f0.computar(dt.date(2018, 3, 9), dt.date(2026, 3, 5), regla="personal", plazo=2920, plazo_anios=8)
ok(_c8.inicio == dt.date(2018, 3, 13) and _c8.vencimiento == dt.date(2026, 3, 13)
   and _c8.oportuna is True and not _c8.dias,
   "ocho años calendario desde el día siguiente al surtimiento, de fecha a fecha (1a./J. 41/2023), sin sumar 2,920 hábiles")
_c8t = f0.computar(dt.date(2018, 3, 9), dt.date(2026, 6, 5), regla="personal", plazo=2920, plazo_anios=8)
ok(_c8t.oportuna is False and _c8t.cierra_por_extemporaneidad,
   "presentada después de los ocho años: extemporánea (con 2,920 hábiles salía en tiempo)")
_p7 = f0.parrafo_oportunidad(f0.computar(dt.date(2019, 1, 10), dt.date(2026, 1, 5), regla="personal",
                                         plazo=2555, plazo_anios=7), "", "amparo_directo")
ok("siete años" in _p7 and "artículo 17, fracción III" in _p7 and "de fecha a fecha" in _p7
   and "2026377" in _p7,
   "el considerando dice los años, la fracción y que se computan de fecha a fecha")

print("\n6 · LO QUE SE CITA DESDE EL 3-oct-2026, CONTRA EL TEXTO LOCAL DE LAS LEYES")
# Cada precepto nuevo de la segunda ronda se coteja con su texto en
# leyes/LEYES_FEDERALES: si el archivo cambia (una reforma) y la frase deja de
# estar, esta prueba lo dice antes que un considerando firmado. Sin la carpeta
# (otro equipo), se salta con su mensaje.
_LEYES = os.path.expanduser("~/Documents/IUREXIA-MAC/leyes/LEYES_FEDERALES")


def _art(archivo: str, encabezado: str) -> str:
    try:
        t = open(os.path.join(_LEYES, archivo), encoding="utf-8").read()
    except OSError:
        return ""
    i = t.find(encabezado)
    if i < 0:
        return ""
    j = t.find("####", i + len(encabezado))
    return " ".join(t[i:j if j > 0 else len(t)].split())


if not os.path.isdir(_LEYES):
    print("   (sin leyes/LEYES_FEDERALES en este equipo: se omite)")
else:
    _lo226 = _art("LEY Orgánica del Poder Judicial de la Federación.txt", "#### Artículo 226.")
    ok("dos periodos vacacionales de quince días" in _lo226 and "Órgano de Administración Judicial" in _lo226
       and "artículo 226 de la Ley Orgánica" in f0.FUENTES_INHABIL["vac"]["texto"],
       "vacaciones de los tribunales de circuito: artículo 226 de la LOPJF (las fija el OAJ)")
    _lo229 = _art("LEY Orgánica del Poder Judicial de la Federación.txt", "#### Artículo 229.")
    ok("21 de marzo" in _lo229 and "lunes" not in _lo229 and "2 de mayo" not in _lo229,
       "el 229 de la LOPJF no nombra los lunes de la LFT ni los días de circular: por eso no se cita para ellos")
    _lft = _art("LEY Federal del Trabajo.txt", "Artículo 74")
    ok("El tercer lunes de marzo en conmemoración del 21 de marzo" in _lft
       and "El primer lunes de febrero" in _lft and "El tercer lunes de noviembre" in _lft,
       "los lunes trasladados: artículo 74, fracciones II, III y VI, de la LFT")
    _cc = _art("CÓDIGO de Comercio.txt", "#### Artículo 1075.-")
    ok("Las notificaciones personales surten efectos al día siguiente del que se hayan practicado" in _cc,
       "mercantil: el surtimiento de la personal está en el 1075 del Código de Comercio (omisión nacional)")
    _l70 = _art("LEY Federal de Procedimiento Contencioso Administrativo.txt", "#### ARTÍCULO 70.")
    ok("el día hábil siguiente a aquél en que fueren hechas" in _l70,
       "TFJA: las notificaciones surten al día hábil siguiente (art. 70 LFPCA)")
    _la167 = _art("Ley Agraria.txt", "#### Artículo 167.-")
    # C7 (3-oct-2026). David: «si va con el Código Nacional, y no el Federal,
    # entonces hay que adecuar al Código Nacional». El 167 ya remite al CNPCF,
    # pero su transitorio Segundo ata la aplicación a la declaratoria: fuera de
    # la Ciudad de México (o sin sede legible) el surtimiento va con el 321 del
    # CFPC y su aviso; en la Ciudad de México, con las reglas del CNPCF. Antes
    # esta comprobación exigía ("", "") sin mirar la sede.
    _sn_sin_sede = f0.surtimiento_nacional("amparo_directo", "agraria", "", "personal")
    try:
        _sn_cdmx = f0.surtimiento_nacional("amparo_directo", "agraria", "", "personal",
                                           tribunal="Décimo Tribunal Colegiado en Materia Administrativa "
                                                    "del Primer Circuito", ciudad="Ciudad de México")
    except TypeError:
        _sn_cdmx = ("(surtimiento_nacional aún no recibe la sede)", "")
    ok("Código Nacional de Procedimientos Civiles y Familiares" in _la167
       and "321 del Código Federal de Procedimientos Civiles" in _sn_sin_sede[0]
       and "declaratoria" in _sn_sin_sede[1].lower()
       and "321" not in _sn_cdmx[0],
       "agrario (C7): el 167 remite al CNPCF; sin sede (o fuera de la CDMX) el 321 del CFPC con el aviso del "
       "transitorio, y en la Ciudad de México no")
    _lf63 = _art("LEY Federal de Procedimiento Contencioso Administrativo.txt", "#### ARTÍCULO 63.")
    ok("Sea una resolución dictada en materia de comercio exterior" in _lf63
       and "comercio exterior" in pf.SUPUESTO_63["V"]
       and "Ley Federal de Responsabilidad Patrimonial del Estado" in pf.SUPUESTO_63["IX"],
       "los supuestos del 63 que se transcriben (IV, V, VII-X) son los del texto de la ley")
    _la19 = _art("Ley de Amparo.txt", "#### Artículo 19.")
    ok(all(x in _la19 for x in ("uno y cinco de mayo", "doce de octubre", "veinticinco de diciembre"))
       and "dos de mayo" not in _la19,
       "el 19 de la Ley de Amparo no nombra el dos de mayo: ese día va con su circular")

print("\n7 · LO QUE SE CITA DESDE LA TERCERA RONDA (3-oct-2026), CONTRA EL TEXTO LOCAL")
# Mismo método que la sección 6: si una reforma mueve la frase, esta prueba lo
# dice antes que un considerando firmado.
if not os.path.isdir(_LEYES):
    print("   (sin leyes/LEYES_FEDERALES en este equipo: se omite)")
else:
    def _parrafos(archivo: str, encabezado: str) -> list:
        """Los párrafos del artículo, en orden (el primero lleva el encabezado)."""
        t = open(os.path.join(_LEYES, archivo), encoding="utf-8").read()
        i = t.find(encabezado)
        if i < 0:
            return []
        return [" ".join(x.split()) for x in t[i:t.find("####", i + len(encabezado))].split("\n\n")
                if x.strip()]

    _parrafos5 = _parrafos("LEY Federal de Procedimiento Contencioso Administrativo.txt", "#### ARTÍCULO 5o.-")
    ok(len(_parrafos5) >= 4 and _parrafos5[3].startswith("La representación de las autoridades corresponderá "
                                                          "a las unidades administrativas encargadas de su "
                                                          "defensa jurídica")
       and "5o., cuarto párrafo" in ta.legitimacion_de("revision_fiscal", "Unidad Jurídica de X"),
       "RF: la unidad de defensa jurídica es el CUARTO párrafo del 5o. de la LFPCA, y la legitimación lo cita")
    _la73 = _parrafos("Ley de Amparo.txt", "#### Artículo 73.")
    ok(len(_la73) >= 2 and "deberán hacer públicos los proyectos de sentencias" in _la73[1]
       and "tres días de anticipación a la publicación de las listas" in _la73[1],
       "art. 73, segundo párrafo, LA: la publicidad del proyecto que recuerda el aviso de la norma impugnada")
    _la9 = _art("Ley de Amparo.txt", "#### Artículo 9o.")
    _la11 = _art("Ley de Amparo.txt", "#### Artículo 11.")
    _la12 = _art("Ley de Amparo.txt", "#### Artículo 12.")
    ok("acreditar personas delegadas" in _la9 and "interpongan recursos" in _la9
       and "persona tercera interesada" in _la11
       and "quedará facultada para interponer los recursos que procedan" in _la12,
       "la personería: delegados de la autoridad (9o.), quien comparece por la tercera (11), autorizado (12)")
    _la7 = _art("Ley de Amparo.txt", "#### Artículo 7o.")
    ok("Las personas morales oficiales" in _la7,
       "el órgano quejoso es «persona moral oficial» (art. 7o.), no «autoridad quejosa»")
    _l66 = _art("LEY Federal de Procedimiento Contencioso Administrativo.txt", "#### ARTÍCULO 66.")
    _l67 = _art("LEY Federal de Procedimiento Contencioso Administrativo.txt", "#### ARTÍCULO 67.")
    _l65 = _art("LEY Federal de Procedimiento Contencioso Administrativo.txt", "#### ARTÍCULO 65.")
    ok("se publicará en el Boletín Jurisdiccional" in _l66
       and "En los demás casos, las notificaciones deberán realizarse por medio del Boletín Jurisdiccional" in _l67
       and "al tercer día hábil siguiente" in _l65
       and f0.surtimiento_nacional("amparo_directo", "administrativa",
                                   "Sala Regional del Golfo del Tribunal Federal de Justicia Administrativa",
                                   "lista") == ("", ""),
       "ante el TFJA lo que no es personal va por Boletín (66, 67) y surte al tercer día (65): «lista» no es "
       "regla y no se le atribuyen los artículos 65 y 70")
    ok("o de cuantía indeterminada" in _lf63
       and "cualquier aspecto relacionado con pensiones que otorga el Instituto de Seguridad y Servicios "
           "Sociales de los Trabajadores del Estado" in _lf63
       and "cualquier aspecto relacionado con pensiones que otorga el Instituto de Seguridad y Servicios "
           "Sociales de los Trabajadores del Estado" in pf._parrafo_vi(
               "un aspecto relacionado con pensiones que otorga el Instituto de Seguridad y Servicios "
               "Sociales de los Trabajadores del Estado"),
       "63 LFPCA: la fracción II admite la cuantía indeterminada; la VI, las pensiones del ISSSTE (literal)")

print("\n8 · LO QUE SE CITA DESDE LA CUARTA RONDA (3-oct-2026), CONTRA EL TEXTO LOCAL")
# E12: la hipótesis del promulgador en el 87, primer párrafo; E8: el autorizado
# del 5o., quinto párrafo, LFPCA es de los particulares (las autoridades nombran
# delegados); E9: el 12 de la Ley de Amparo es el del autorizado.
if not os.path.isdir(_LEYES):
    print("   (sin leyes/LEYES_FEDERALES en este equipo: se omite)")
else:
    _la87 = _parrafos("Ley de Amparo.txt", "#### Artículo 87.")
    _leg87 = ta.legitimacion_de("amparo_revision", "Gobernador del Estado de Querétaro", papel="autoridad",
                                hay_norma=True)
    ok(len(_la87) >= 1 and "tratándose de amparo contra normas generales podrán hacerlo los titulares de los "
                           "órganos del Estado a los que se encomiende su emisión o promulgación" in _la87[0]
       and "87, primer párrafo, de la Ley de Amparo" in _leg87
       and "los titulares de los órganos del Estado a los que se encomiende su emisión o promulgación" in _leg87,
       "art. 87, PRIMER párrafo, LA: la hipótesis del promulgador que la legitimación del Gobernador cita (AR 208)")
    ok(len(_parrafos5) >= 5
       and _parrafos5[4].startswith("Los particulares o sus representantes podrán autorizar por escrito")
       and "Las autoridades podrán nombrar delegados para los mismos fines" in _parrafos5[4],
       "5o., QUINTO párrafo, LFPCA: el autorizado es de los particulares; las autoridades nombran delegados (RF 4)")
    ok("autorizado" not in ta.legitimacion_de("revision_fiscal", "Titular de la Unidad Jurídica de X", "Juan Pérez",
                                              figura="autorizado en términos del artículo 5",
                                              autoridad_demandada="Subdelegado"),
       "por eso la legitimación de la revisión fiscal no firma que la unidad actuó por su autorizado")

print("\n9 · LO QUE CITA LA SEXTA RONDA (3-oct-2026, C1, C2 y C6), CONTRA EL TEXTO LOCAL")
# C1: la procedencia del AR cita el inciso del 81, fracción I (texto de
# normas_ley_de_amparo.json, que viaja con el código). C2: la fracción del 35
# de la LOPJF por tipo y por lo recurrido. C6: el 64 de la LFPCA de la
# conexidad AD↔RF, y el 88 del CFPC y el 269 del CNPCF del hecho notorio.
_la81 = (json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "normas_ley_de_amparo.json"),
                        encoding="utf-8")).get("articulos", {}).get("81", ""))
for _cl, _inc, _frase in (("sentencia", "e", "e) Las sentencias dictadas en la audiencia constitucional"),
                          ("interlocutoria_suspension", "a", "a) Las que concedan o nieguen la suspensión definitiva"),
                          ("reposicion_constancias", "c", "c) Las que decidan el incidente de reposición de "
                                                          "constancias de autos"),
                          ("auto_sobreseimiento", "d", "d) Las que declaren el sobreseimiento fuera de la "
                                                       "audiencia constitucional")):
    ok(_frase in _la81 and f"81, fracción I, inciso {_inc})," in ta.procedencia_revision(_cl),
       f"C1 · {_cl}: la procedencia cita el inciso {_inc}) del 81, fracción I, que dice «{_frase[3:50]}…»")
# LOS DOS SUPUESTOS DEL INCISO b) (revisión de normas, 3-oct-2026): el 81-I-b
# cubre también «las que nieguen la revocación o modificación de esos autos», y
# la detección (`_RX_MODIFICA_SUSP`) lleva ahí la que negó modificar. Antes sólo
# se cotejaba la primera mitad del inciso.
ok("b) Las que modifiquen o revoquen el acuerdo en que se conceda o niegue la suspensión definitiva" in _la81
   and "o las que nieguen la revocación o modificación de esos autos" in _la81
   and "inciso b)," in ta.procedencia_revision("interlocutoria_suspension", "b")
   and "se pronunció sobre la modificación o revocación" in ta.procedencia_revision(
       "interlocutoria_suspension", "b")
   and ta._RX_MODIFICA_SUSP.search("VISTOS para resolver el incidente de suspensión en el que se niega la "
                                   "petición de modificar la interlocutoria que concedió la suspensión definitiva."),
   "C1 · el inciso b) con sus dos supuestos: lo que modificó o revocó y lo que negó hacerlo")
if not os.path.isdir(_LEYES):
    print("   (sin leyes/LEYES_FEDERALES en este equipo: se omite el cotejo del 35, el 64, el 88 y el 269)")
else:
    _lo35 = _art("LEY Orgánica del Poder Judicial de la Federación.txt", "#### Artículo 35.")
    ok("II. Del recurso de revisión en los casos a que se refiere el artículo 81 de la Ley de Amparo" in _lo35
       and "III. Del recurso de queja" in _lo35
       and "V. Del recurso de revisión contra las sentencias pronunciadas en la audiencia constitucional" in _lo35
       and "en los casos a que se refiere el artículo 84 de la Ley de Amparo" in _lo35
       and "remitidos por la Suprema Corte de Justicia de la Nación" in _lo35
       and "VI. De los recursos de revisión que las leyes establezcan en términos de la fracción III del artículo "
           "104" in _lo35
       and "b) En materia administrativa" in _lo35,
       "C2 · LOPJF 35: I directo (b administrativa), II revisión del 81, III queja, V sentencia de audiencia y "
       "lo remitido por la SCJN, VI revisión fiscal")
    _cv, _ = ta.competencia_revision("81, fracción I, inciso e) y 84 de la Ley de Amparo; 35, fracción V y 210 de "
                                     "la Ley Orgánica del Poder Judicial de la Federación", "",
                                     clase="reposicion_constancias")
    ok("inciso c), y 84" in _cv and "35, fracción II y 210" in _cv,
       "C2 · la reposición de constancias: inciso c) y fracción II (antes caía en la e) y la V)")
    _l64 = _art("LEY Federal de Procedimiento Contencioso Administrativo.txt", "#### ARTÍCULO 64.")
    _c64 = ta.considerando_relacionados("amparo_directo", "469/2024", "administrativa",
                                        [{"tipo": "revision_fiscal", "numero": "33/2024"}], "")[1]
    ok("Si el particular interpuso amparo directo contra la misma resolución o sentencia impugnada mediante el "
       "recurso de revisión" in _l64 and "lo cual tendrá lugar en la misma sesión en que decida el amparo" in _l64
       and "artículo 64 de la Ley Federal de Procedimiento Contencioso Administrativo" in _c64,
       "C6 · LFPCA 64: el amparo directo y la revisión fiscal contra la misma sentencia, en la misma sesión")
    _cf88 = _art("Código Federal de Procedimientos Civiles.txt", "#### ARTICULO 88.-")
    _cn269 = _art("CÓDIGO Nacional de Procedimientos Civiles y Familiares.txt", "#### Artículo 269.")
    ok("Los hechos notorios pueden ser invocados por el tribunal" in _cf88
       and "Los hechos notorios no necesitan ser probados" in _cn269
       and "artículo 88 del Código Federal" in ta.supletorio("", "Querétaro, Querétaro")["hecho_notorio"]
       and "artículo 269 del Código Nacional" in ta.supletorio("", "Ciudad de México")["hecho_notorio"],
       "C6 · el hecho notorio: CFPC 88 fuera de la Ciudad de México, CNPCF 269 en ella")

print()
if FALLAS:
    print(f"FALLAN {len(FALLAS)}: " + " · ".join(FALLAS))
    sys.exit(1)
print("RESULTADO: TODAS LAS COMPROBACIONES PASAN")
