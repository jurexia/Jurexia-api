"""LAS NORMAS DE LA PROCEDENCIA, TERCERA RONDA (3-oct-2026) — dueño B de FIXES_R3.

Cada comprobación es un hallazgo de la revisión contra los engroses del banco de
oráculo (rev_0 queja, rev_1 amparo directo, rev_2 fundamentos, rev_3 amparo en
revisión, rev_4 regresión sin bandera, rev_5 revisión fiscal) y FALLA con el
código de antes. Sin red y sin modelo. Las secciones 1-18 corren CON la bandera
`procedencia_por_tipo` (lo que no es [siempre] va tras ella, FIXES_R3); la 19 la
apaga y comprueba que el camino viejo queda como estaba salvo los [siempre].

CUARTA RONDA (3-oct-2026, FIXES_R4, dueño B): las secciones 20-28 son los
hallazgos de la verificación final (rev/verif_final.txt) con las decisiones
E1, E2, E8, E9, E10 y E12; cada una FALLA con el código de la tercera ronda.

QUINTA RONDA (3-oct-2026, FIXES_R5, dueño B): las secciones 29-39 son los
hallazgos de la segunda verificación (rev/verif_final2.txt) con las decisiones
F1, F2, F4, F5 y F6; cada una FALLA con el código de la cuarta ronda, salvo la
guarda de F4 (sección 36), que ya pasaba y queda para que nadie la rompa.

    .venv/bin/python test_normas_procedencia.py
"""
import datetime as dt
import inspect
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("OPENAI_API_KEY", "sk-falsa")

import contexto_taller as ct
import fase0_oportunidad as f0
import fase_origen as fo
import fase_procedencia_rf as pf
import tipos_asunto as ta

FALLAS = []
D = dt.date
H = "*********"
TFJA = "Sala Regional del Centro II del Tribunal Federal de Justicia Administrativa"


def ok(cond, que):
    print(("   PASA   " if cond else "   FALLA  ") + que)
    if not cond:
        FALLAS.append(que)


def bandera(encendida: bool):
    ct.poner(True, {"banderas": {"procedencia_por_tipo": bool(encendida)}}, pruebas=True)


bandera(True)


# CADA SECCIÓN SE CORRE APARTE: con el código de antes una función o un campo
# nuevo no existe y la sección revienta; eso cuenta como FALLA de la sección y
# las demás siguen, para que se vea qué arreglo falta y no sólo el primero.
def _seccion(n, fn):
    try:
        fn()
    except Exception as e:  # noqa: BLE001 — una sección rota es una FALLA, no un aborto
        ok(False, f"la sección {n} reventó: {type(e).__name__}: {e}")


def _s1():
    print("\n1 · LOS INHÁBILES ENTRE LA NOTIFICACIÓN Y EL INICIO DEL PLAZO (rev_0, rev_1)")
    # AD 335/2025: notificada el miércoles 19-mar-2025, surtió el 20, el viernes 21
    # es inhábil (art. 19) y el plazo arranca el lunes 24. El engrose nombra el 21.
    c = f0.computar(D(2025, 3, 19), D(2025, 4, 9), "personal", 15)
    p = f0.parrafo_oportunidad(c, "", "amparo_directo", con_previos=True)
    ok(c.inicio == D(2025, 3, 24) and c.inhabiles_previos == [D(2025, 3, 21)] and c.inhabiles_en_medio == [],
       "AD 335/2025: el 21 de marzo queda en `inhabiles_previos`, aparte de los del plazo")
    ok("ni el veintiuno de marzo de dos mil veinticinco, por ser inhábiles en términos del artículo 19" in p,
       "…y el considerando lo nombra con su fundamento (antes: sólo «sin contar sábados y domingos»)")
    ok("veintiuno de marzo" not in f0.parrafo_oportunidad(c, "", "amparo_directo"),
       "sin pedirlo (`con_previos`, el camino nuevo = bandera Y ficha) el párrafo queda como estaba")
    # Q 24/2026: electrónica el 18-dic-2025; vacaciones del 19 al 31 y el 1 de enero.
    c = f0.computar(D(2025, 12, 18), D(2026, 1, 8), "electronica", 5, tipo_asunto="queja")
    p = f0.parrafo_oportunidad(c, "", "queja", con_previos=True)
    ok(c.inicio == D(2026, 1, 2) and c.vencimiento == D(2026, 1, 8),
       "Q 24/2026 con la regla electrónica: del dos al ocho de enero, como el engrose")
    ok("ni el uno de enero de dos mil veintiséis, por ser inhábiles en términos del artículo 19 de la Ley de Amparo, "
       "ni del diecinueve al treinta y uno de diciembre de dos mil veinticinco, por corresponder al periodo "
       "vacacional del Poder Judicial de la Federación, conforme al artículo 226 de la Ley Orgánica del Poder "
       "Judicial de la Federación" in p,
       "Q 24/2026: las vacaciones y el 1 de enero que movieron el arranque, cada uno con su fundamento (el 19 una vez)")
    # Q 172/2026: electrónica el 31-mar-2026; Circular 3/2026 del 1 al 3 de abril.
    c = f0.computar(D(2026, 3, 31), D(2026, 4, 10), "electronica", 5, tipo_asunto="queja")
    ok(c.inicio == D(2026, 4, 6) and "ni del uno al tres de abril de dos mil veintiséis, por ser inhábiles "
       "conforme a la Circular 3/2026" in f0.parrafo_oportunidad(c, "", "queja", con_previos=True),
       "Q 172/2026: el 1 al 3 de abril de la Circular 3/2026 se funda aunque caiga antes del plazo")
    # LOS DE LA RESPONSABLE NO SE CUELAN: sólo cuentan dentro del plazo.
    c = f0.computar(D(2026, 3, 2), D(2026, 3, 20), "personal", 15,
                    inhabiles_responsable="2026-03-03", tipo_asunto="amparo_directo")
    ok(D(2026, 3, 3) not in c.inhabiles_previos and D(2026, 3, 3) not in c.resp_en_medio,
       "un día declarado de la responsable antes del plazo no entra a los previos (ni a los del plazo)")

_seccion(1, _s1)

def _s2():
    print("\n2 · LOS INHÁBILES EN ORDEN CRONOLÓGICO (D6, rev_4)")
    c = f0.computar(D(2025, 12, 5), D(2026, 1, 8), "personal", 15)
    cl = f0.clausula_inhabiles(c)
    ok(cl.startswith("sin contar sábados y domingos, ni el uno de enero de dos mil veintiséis, por ser inhábiles en "
                     "términos del artículo 19") and cl.count("artículo 19") == 1
       and "ni del dieciséis al treinta y uno de diciembre" in cl,
       "el grupo del artículo 19 con los sábados y domingos, UNA vez; después los demás (integración)")
    ok([k for k, _ in f0.grupos_inhabiles(f0.tramos_inhabiles(c.inhabiles_en_medio, c.cal_amparo))] == ["vac", "art19"]
       and [k for k, _ in f0.grupos_inhabiles(f0.tramos_inhabiles(c.inhabiles_en_medio, c.cal_amparo),
                                              art19_al_final=True)] == ["vac", "art19"],
       "grupos_inhabiles va por la fecha de su primer tramo")
    cm = f0.computar(D(2025, 4, 10), D(2025, 5, 6), "personal", 15, tipo_asunto="queja")
    ok([k for k, _ in f0.grupos_inhabiles(f0.tramos_inhabiles(cm.inhabiles_en_medio, cm.cal_amparo))]
       == ["c1_2025", "art19"]
       and f0.tramos_en_letra(f0.tramos_inhabiles(cm.inhabiles_en_medio, cm.cal_amparo)).endswith(
           "ni el uno y el cinco de mayo de dos mil veinticinco"),
       "Semana Santa (Circular 1/2025) antes que el 1 de mayo; `tramos_en_letra` sigue con el 19 al final")

_seccion(2, _s2)

def _s3():
    print("\n3 · LA NOTIFICACIÓN O LA PRESENTACIÓN EN DÍA INHÁBIL (rev_1, rev_5)")
    c = f0.computar(D(2024, 6, 30), D(2024, 7, 19), "personal", 15)
    ok(any(a.startswith("LA NOTIFICACIÓN CAE EN DÍA INHÁBIL (domingo treinta de junio de dos mil veinticuatro)")
           for a in c.avisos), "AD 552/2024: notificación personal en domingo, con aviso")
    c = f0.computar(D(2024, 12, 8), D(2025, 1, 20), "lfpca_boletin", 15, TFJA, tipo_asunto="revision_fiscal")
    ok(any("LA NOTIFICACIÓN CAE EN DÍA INHÁBIL (domingo" in a for a in c.avisos),
       "RF 26/2025: Boletín Jurisdiccional en domingo, con aviso")
    c = f0.computar(D(2026, 3, 2), D(2026, 3, 14), "personal", 15)
    ok(any("LA PRESENTACIÓN CAE EN DÍA INHÁBIL (sábado" in a for a in c.avisos),
       "presentación en sábado: aviso (puede ser electrónica; se pide comprobarla)")
    c = f0.computar(D(2025, 12, 18), D(2026, 1, 8), "electronica", 5, tipo_asunto="queja")
    ok(not any("CAE EN DÍA INHÁBIL" in a for a in c.avisos),
       "la notificación electrónica en vacaciones (Q 24/2026) no se acusa: existe")
    c = f0.computar(D(2026, 3, 2), D(2026, 3, 20), "personal", 15)
    ok(not any("CAE EN DÍA INHÁBIL" in a for a in c.avisos), "días hábiles: ningún aviso")

_seccion(3, _s3)

def _s4():
    print("\n4 · EL PRECEPTO DEL SURTIMIENTO EN EL AMPARO DIRECTO (D1)")
    c = f0.computar(D(2025, 2, 24), D(2025, 3, 13), "personal", 15)
    p = f0.parrafo_oportunidad(c, "", "amparo_directo", sin_precepto_en_hueco=True)
    ok("de manera personal y surtió efectos al día hábil siguiente, es decir, el veinticinco de febrero" in p
       and H not in p and "ley del acto" not in p,
       "AD 274/2025: sin el precepto en el catálogo, se OMITE la cláusula (ni hueco ni «ley del acto»)")
    a = f0.aviso_fundamento(c, "amparo_directo", en_hueco=True)
    ok(a.startswith("SE RECOMIENDA CITAR EL PRECEPTO DEL SURTIMIENTO") and H not in a and "HUECO" not in a,
       "…con un aviso que RECOMIENDA el precepto, no que lo pide como hueco")
    p = f0.parrafo_oportunidad(c, "", "amparo_directo", sin_precepto_en_hueco=True,
                               fundamento_surtimiento="el artículo 126 del Código de Procedimientos Civiles "
                                                      "del Estado de Querétaro")
    ok("surtió efectos al día hábil siguiente, conforme al artículo 126 del Código de Procedimientos Civiles del "
       "Estado de Querétaro, es decir" in p, "si el precepto consta en la ficha, se escribe")
    ok("conforme a la ley del acto" in f0.parrafo_oportunidad(c, "", "amparo_directo"),
       "el camino viejo (sin la bandera) queda idéntico")
    ca = f0.computar(D(2025, 3, 20), D(2025, 3, 27), "personal", 10, tipo_asunto="amparo_revision")
    ok("conforme al *********" in f0.parrafo_oportunidad(ca, "", "amparo_revision", papel="autoridad",
                                                          sin_precepto_en_hueco=True),
       "la autoridad notificada como particular en un recurso sigue en hueco (el día sería falso)")

_seccion(4, _s4)

def _s5():
    print("\n5 · LO MERCANTIL QUE SE REGISTRÓ COMO CIVIL, Y EL TFJA SÓLO CON LA PERSONAL (rev_1, rev_2)")
    ok(f0.surtimiento_nacional("amparo_directo", "civil", "Juzgado Primero de Primera Instancia Especializado en "
                               "Oralidad Mercantil del Distrito Judicial de Querétaro", "personal")[0]
       == "el artículo 1075 del Código de Comercio",
       "AD 456/2025: «AMPARO DIRECTO CIVIL» con un juzgado de oralidad mercantil → 1075 del Código de Comercio")
    ok(f0.surtimiento_nacional("amparo_directo", "civil", "Juzgado Menor Civil de Querétaro", "personal",
                               expediente="juicio ejecutivo mercantil 1234/2024")[0]
       == "el artículo 1075 del Código de Comercio",
       "AD 279/2025: el ejecutivo mercantil se reconoce por el expediente")
    ok(f0.surtimiento_nacional("amparo_directo", "civil", "Juzgado Segundo Civil y Mercantil", "personal") == ("", ""),
       "un juzgado «Civil y Mercantil» conoce de las dos: no decide")
    ok(f0.surtimiento_nacional("amparo_directo", "administrativa", TFJA, "lista") == ("", "")
       and "65 y 70" in f0.surtimiento_nacional("amparo_directo", "administrativa", TFJA, "personal")[0],
       "TFJA: los artículos 65 y 70 sólo para la personal; con «lista» no se afirma el día siguiente")
    c = f0.computar(D(2025, 3, 3), D(2025, 3, 20), "lista", 15, TFJA, tipo_asunto="amparo_directo")
    ok(any(a.startswith("LA SALA DEL TFJA NO NOTIFICA «POR LISTA»") and "tercer día hábil" in a for a in c.avisos),
       "…y el cómputo pide la regla del Boletín (tercer día hábil, art. 65)")

_seccion(5, _s5)

def _s6():
    print("\n6 · LA REVISIÓN FISCAL POR CORREO SE MIDE CON EL DEPÓSITO (D3)")
    ok("deposito" in inspect.signature(f0.computar).parameters, "computar recibe `deposito`")
    c = f0.computar(D(2024, 10, 1), D(2024, 11, 6), "lfpca_boletin", 15, TFJA, tipo_asunto="revision_fiscal",
                    deposito=D(2024, 10, 24))
    ok(c.presentacion == D(2024, 10, 24) and c.deposito == D(2024, 10, 24) and c.recepcion == D(2024, 11, 6)
       and c.oportuna is True,
       "RF 2/2025: depósito en plazo y recepción fuera → OPORTUNO (antes se desechaba por extemporáneo)")
    p = f0.parrafo_oportunidad(c, "", "revision_fiscal")
    ok("si el oficio se depositó en el Servicio Postal Mexicano el veinticuatro de octubre de dos mil veinticuatro"
       in p and "XV.4o.1 A (11a.)" in p and "2025728" in p,
       "el párrafo dice la fecha del depósito y la tesis que lo sostiene (RF 28/2025)")
    ok(any("LA OPORTUNIDAD SE MIDIÓ CON LA FECHA DEL DEPÓSITO" in a and "EXTEMPORÁNEO" in a for a in c.avisos),
       "…con aviso: compruébalo, y con la recepción habría sido extemporáneo")
    c = f0.computar(D(2024, 10, 1), D(2024, 10, 24), "lfpca_boletin", 15, TFJA, tipo_asunto="amparo_directo",
                    deposito=D(2024, 10, 20))
    ok(c.presentacion == D(2024, 10, 24) and c.deposito is None
       and any("EL DEPÓSITO POSTAL" in a and "NO SE USÓ" in a for a in c.avisos),
       "fuera de la revisión fiscal el depósito no mide la oportunidad (y se dice)")
    c = f0.computar(D(2024, 10, 1), D(2024, 10, 20), "lfpca_boletin", 15, TFJA, tipo_asunto="revision_fiscal",
                    deposito=D(2024, 10, 24))
    ok(c.deposito is None and c.presentacion == D(2024, 10, 20) and any("FECHA IMPOSIBLE" in a for a in c.avisos),
       "depósito posterior a la recepción: imposible, no se usa")

_seccion(6, _s6)

def _s7():
    print("\n7 · LOS ACUERDOS DEL CALENDARIO DEL TFJA (rev_5)")
    ok(f0.ACUERDO_TFJA[2025] == "los Acuerdos SS/1/2025 y SS/22/2025", "2025: el SS/1/2025 con el SS/22/2025")
    c = f0.computar(D(2025, 12, 5), D(2026, 1, 7), "lfpca_boletin", 15, TFJA, tipo_asunto="revision_fiscal")
    p = f0.parrafo_oportunidad(c, "", "revision_fiscal")
    ok("de los Acuerdos SS/1/2025 y SS/22/2025" in p and "SS/2/2026" not in p,
       "RF 6/2026: el 2 de enero de 2026 por el SS/22/2025, no por el SS/2/2026 (publicado el 12 de enero)")
    c = f0.computar(D(2025, 4, 10), D(2025, 5, 6), "lfpca_boletin", 15, tipo_asunto="revision_fiscal")
    ok("del Acuerdo SS/1/2025" in f0.parrafo_oportunidad(c, "", "revision_fiscal"),
       "un plazo de abril de 2025 no cita el SS/22/2025, que aún no existía")
    ok(f0.acuerdos_tfja_del_dia(D(2027, 1, 1)) == ("SS/2/2026",) and f0.acuerdos_tfja_del_dia(D(2026, 1, 2))
       == ("SS/22/2025",), "el 1 de enero lo fija el calendario del año anterior; el 2-ene-2026, el SS/22/2025")

_seccion(7, _s7)

def _s8():
    print("\n8 · UNA SOLA REGLA PARA LA CIUDAD DE MÉXICO (rev_2)")
    P1 = "Décimo Tribunal Colegiado en Materia Civil del Primer Circuito"
    ok(ta.es_cdmx(P1, "Cd. de México") is True and ta.supletorio(P1, "Cd. de México")["cdmx"] is True
       and "Código Nacional" in ta.supletorio(P1, "México")["documentales"],
       "Primer Circuito con «Cd. de México» o «México»: el CNPCF (antes el CFPC y la verja lo acusaba)")
    ok(all(ta.es_cdmx("", c_) for c_ in ("Cd. Mx.", "Cd. México", "México, D.F.", "Distrito Federal", "CDMX"))
       and ta.es_cdmx("", "Cd. Mexicali") is False and ta.es_cdmx("", "Toluca, Estado de México") is False,
       "las grafías de la Ciudad de México, sin confundir Mexicali ni el Estado de México")
    ok(ta.es_cdmx("Primer Tribunal Colegiado del Centro Auxiliar de la Primera Región", "Naucalpan") is False
       and ta.es_cdmx("Primer Tribunal Colegiado del Centro Auxiliar de la Primera Región", "Ciudad de México") is True
       and ta.es_cdmx("Tercer Tribunal Colegiado del Vigésimo Segundo Circuito", "") is False
       and ta.es_cdmx("", "") is None,
       "el Centro Auxiliar decide por su ciudad; otro circuito, no; sin datos, no se sabe")

_seccion(8, _s8)

def _s9():
    print("\n9 · LA PERSONERÍA POR FIGURA Y PAPEL (rev_0, rev_2, rev_3)")
    _q = ta.legitimacion_de("queja", "Ana López Ruiz", "Juan Pérez López", figura="autorizada en términos amplios",
                            papel="quejoso")
    # QUINTA RONDA (F6): el separador es el del resultando —coma, porque la figura
    # pasa de tres palabras— y la cola no repite «en términos».
    ok("autorizada en términos amplios, Juan Pérez López, conforme al artículo 12 de la Ley de Amparo" in _q
       and "6o. y 11" not in _q, "Q 172/2026: la autorizada en términos amplios, por el artículo 12")
    _q = ta.legitimacion_de("queja", "Director de Ingresos del Municipio de Querétaro", "Juan Pérez López",
                            figura="delegado", papel="autoridad")
    ok("delegado Juan Pérez López, en términos del artículo 9o. de la Ley de Amparo" in _q and "6o. y 11" not in _q,
       "la autoridad que recurre la queja por su delegado: artículo 9o.")
    _q = ta.legitimacion_de("queja", "Inmobiliaria Ejemplo, S.A. de C.V.", "Juan Pérez López", figura="apoderado",
                            papel="tercero")
    ok("en términos del artículo 11 de la Ley de Amparo" in _q and "6o." not in _q,
       "la tercera interesada por su apoderado: artículo 11 (el 6o. es de la quejosa)")
    _q = ta.legitimacion_de("amparo_directo", "Ana López Ruiz", "Juan Pérez López", figura="apoderado")
    ok("en términos de los artículos 6o. y 11 de la Ley de Amparo" in _q, "la quejosa por su apoderado: 6o. y 11")
    _q = ta.legitimacion_de("amparo_revision", "Ana López Ruiz", "Juan Pérez López",
                            figura="autorizado en términos amplios del artículo 12 de la Ley de Amparo", papel="quejoso")
    ok("del artículo 12 de la Ley de Amparo, Juan Pérez López;" in _q and "ley de amparo" not in _q
       and _q.count("artículo 12") == 1,
       "AR 239/2025: la figura conserva «Ley de Amparo», el nombre va tras coma y el 12 no se repite")
    # CUARTA RONDA: el cargo de una autoridad ya no se baja (rev RF 2/26/6-2026:
    # dos grafías del mismo cargo en un documento); la figura genérica, sí.
    ok(ta.figura_en_prosa("Apoderado Legal") == "apoderado legal"
       and ta.figura_en_prosa("Titular de la Unidad de Asuntos Jurídicos") == "Titular de la Unidad de Asuntos Jurídicos",
       "la figura genérica pasa a minúsculas; el cargo de una autoridad conserva su mayúscula")
    ok("5o., fracción I, y 6o. de la Ley de Amparo" in ta.legitimacion_de("amparo_revision", "Ana López Ruiz",
                                                                          papel="quejoso"),
       "AR 60/2025: la quejosa que recurre, con el 5o., fracción I, y el 6o.")
    ok("por conducto" not in ta.legitimacion_de("queja", "Ana López Ruiz", "Ana López Ruiz",
                                                figura="representante legal", papel="quejoso"),
       "Q 261/2025: quien actúa por sí no aparece como su propia representante")

_seccion(9, _s9)

def _s10():
    print("\n10 · EL ÓRGANO QUEJOSO ES PERSONA MORAL OFICIAL, NO AUTORIDAD (rev_4, regresión)")
    _l = ta.legitimacion_de("amparo_directo", "Instituto Mexicano del Seguro Social", "Juan Pérez López",
                            figura="apoderado")
    ok("la persona moral oficial quejosa está legitimada" in _l and "autoridad quejosa" not in _l,
       "AD con el IMSS como quejoso: «la persona moral oficial quejosa»")
    _l = ta.legitimacion_de("amparo_revision", "Municipio de Querétaro", "Juan Pérez López", figura="apoderado",
                            papel="quejoso")
    ok("la persona moral oficial recurrente" in _l and "autoridad recurrente" not in _l,
       "AR con el municipio quejoso que recurre: «la persona moral oficial recurrente»")
    ok("la autoridad recurrente" in ta.legitimacion_de("amparo_revision", "Director de Ingresos", "Juan Pérez",
                                                       figura="delegado", papel="autoridad"),
       "la autoridad responsable que recurre sigue siendo «la autoridad recurrente»")

_seccion(10, _s10)

def _s11():
    print("\n11 · VARIAS PERSONAS EN UN CAMPO (rev_1, rev_3)")
    ok(ta.es_plural_de_partes("Juan Pérez López y María Gómez Ruiz")
       and ta.es_plural_de_partes("PARTE QUEJOSA y PARTE QUEJOSA, ambos de apellidos PARTE QUEJOSA")
       and not ta.es_plural_de_partes("Juan Pérez López")
       and not ta.es_plural_de_partes("Unión de Trabajadores de la Construcción, Transportistas y Similares")
       and not ta.es_plural_de_partes("Juan Pérez López, por propio derecho"),
       "se reconoce el plural sin confundir una persona moral con «y» en su nombre")
    # CUARTA RONDA (E2): sin género en el texto, la fórmula neutra «la parte
    # quejosa está legitimada»; con «ambos», «quienes están legitimados».
    _l = ta.legitimacion_de("amparo_directo", "Juan Pérez López y María Gómez Ruiz")
    ok("; la parte quejosa está legitimada" in _l and "le causa" in _l and "quien" not in _l,
       "AD 552/2024: varias personas sin género que conste → «; la parte quejosa está legitimada»")

_seccion(11, _s11)

def _s12():
    print("\n12 · LA SUCESIÓN, LA COMUNIDAD, EL EJIDO; Y EL NOMBRE DE PILA NO DA GÉNERO (rev_1)")
    ok(ta.genero_de("Sucesión a Bienes de Juan Pérez") == "a" and ta.genero_de("Comunidad Indígena de X") == "a"
       and ta.genero_de("Ejido San Juan") == "o",
       "AD 335/2025: «QUEJOSA» de la Sucesión (antes «PARTE QUEJOSA»)")
    ok(ta.genero_de("Isadora Pérez López") == "" and ta.genero_de("Teodora Ruiz") == ""
       and ta.genero_de("Impulsora de Desarrollos Ejemplo VV") == "a",
       "«Isadora» y «Teodora» no son cabezas en -dora; «Impulsora» sí")
    ok(ta.con_articulo_de_colectivo("Sucesión a Bienes de Juan Pérez") == "la Sucesión a Bienes de Juan Pérez"
       and ta.con_articulo_de_colectivo("Ejido San Juan") == "el Ejido San Juan"
       and ta.con_articulo_de_colectivo("Comercializadora X, S.A.") == "Comercializadora X, S.A.",
       "el artículo de la cabeza colectiva; la sociedad sigue sin él")
    ok(ta.legitimacion_de("amparo_directo", "Sucesión a Bienes de Juan Pérez").startswith(
       "La demanda de amparo fue promovida por la Sucesión a Bienes de Juan Pérez, quien está legitimada"),
       "la legitimación nombra «la Sucesión…» y concuerda con ella")

_seccion(12, _s12)

def _s13():
    print("\n13 · LA CARÁTULA (rev_0, rev_1, rev_5)")
    _f = ta.filas_caratula("queja", {"quejoso": "Juan Pérez López", "recurrente": "Inmobiliaria Ejemplo, S.A. de C.V.",
                                     "papel_recurrente": "tercero", "responsable": "Juzgado Cuarto de Distrito"})
    ok(_f[:2] == [("PARTE QUEJOSA", "quejoso", "Juan Pérez López"),
                  ("TERCERA INTERESADA Y RECURRENTE", "recurrente", "Inmobiliaria Ejemplo, S.A. de C.V.")],
       "Q 300/2025: recurre la tercera → «PARTE QUEJOSA» y «TERCERA INTERESADA Y RECURRENTE»")
    _f = ta.filas_caratula("queja", {"quejoso": "Juan Pérez López", "recurrente": "Juan Pérez López",
                                     "papel_recurrente": "quejoso", "responsable": "Juzgado Cuarto"})
    ok(_f[0] == ("PARTE QUEJOSA Y RECURRENTE", "quejoso", "Juan Pérez López")
       and not any(c_ == "recurrente" for _e, c_, _v in _f),
       "recurre la propia quejosa: UN renglón, no la misma persona dos veces")
    _f = ta.filas_caratula("amparo_revision", {"quejoso": "Unión Ejemplo, A.C.", "recurrente": "Unión Ejemplo, A.C."})
    ok(_f == [("QUEJOSA Y RECURRENTE", "quejoso", "Unión Ejemplo, A.C.")],
       "lo mismo en la revisión (antes se duplicaba)")
    _f = ta.filas_caratula("amparo_directo", {"quejoso": "Comercializadora X, S.A. de C.V.",
                                              "adherente": "Juan Pérez López", "responsable": "Sala Civil"})
    ok(("PARTE QUEJOSA ADHERENTE", "adherente", "Juan Pérez López") in _f,
       "AD 279/2025: el adherente en la carátula, concordado con SU nombre (no con el de la quejosa)")
    _f = ta.filas_caratula("revision_fiscal", {"quejoso": "Jefe de la Unidad Jurídica", "adherente": "Ana Ruiz",
                                               "responsable": "Sala Regional"})
    ok(("RECURRENTE ADHESIVO", "adherente", "Ana Ruiz") in _f, "RF 4/2025: el recurrente adhesivo en la carátula")

_seccion(13, _s13)

def _s14():
    print("\n14 · LA NORMA IMPUGNADA Y LA DELEGACIÓN DE LA SUPREMA CORTE (rev_3)")
    _c, _a = ta.competencia_revision("…", "VISTOS para resolver los autos del juicio de amparo; audiencia constitucional",
                                     "", actos=["La Ley de Hacienda del Estado de Querétaro, artículos 90 y 97"],
                                     autoridades=["Gobernador del Estado de Querétaro"],
                                     resolvio="concede", papel="autoridad")
    ok(any("SE IMPUGNÓ LA CONSTITUCIONALIDAD DE UNA NORMA" in x and "muy probablemente" in x
           and "artículo 73, segundo párrafo" in x for x in _a),
       "AR 208/2025: «La Ley de Hacienda…, artículos 90 y 97», concedido y recurrido por el Gobernador")
    _c, _a = ta.competencia_revision("…", "VISTOS…; audiencia constitucional", "",
                                     actos=["el artículo 90 de la Ley de Hacienda del Estado de Querétaro"])
    ok(any("CONSTITUCIONALIDAD" in x for x in _a), "AR 72/2025: «el artículo 90 de la Ley…»")
    _c, _a = ta.competencia_revision("…", "VISTOS…; audiencia constitucional", "",
                                     autoridades=["Legislatura del Estado de Querétaro"])
    ok(any("Poder Legislativo" in x for x in _a), "la Legislatura entre las responsables también lo delata")
    _c, _a = ta.competencia_revision("…", "VISTOS…; audiencia constitucional",
                                     "en el juicio se citó el artículo 16 de la Ley de Amparo",
                                     actos=["La orden de clausura del establecimiento"])
    ok(not _a, "un acto ordinario y una cita en la prosa no disparan el aviso")

_seccion(14, _s14)

def _s15():
    print("\n15 · EL RESOLUTIVO DE LA REVISIÓN SIN ORDINALES AJENOS (rev_3)")
    # C5 (3-oct-2026): la remisión genérica ya no va escrita en la rama; va el
    # marcador {actos_del_amparo} y, sin ficha, `ACTOS_DEL_AMPARO_GENERICO`
    # («contra» en el sobreseimiento). Antes se buscaba la frase en la rama.
    _pt = " ".join(ta.RAMAS_REVISION["confirma_sobresee"]["puntos"])
    _gen = ta.con_actos_del_amparo(ta.RAMAS_REVISION["confirma_sobresee"]["puntos"][1])
    ok("considerando segundo" not in _pt and "último considerando" not in _pt
       and "{actos_del_amparo}" in _pt
       and "contra los actos precisados en la resolución recurrida" in _gen,
       "AR 239/2025: «Se sobresee… contra los actos precisados en la resolución recurrida» (respaldo genérico)")

_seccion(15, _s15)

def _s16():
    print("\n16 · LA LEGITIMACIÓN DE LA REVISIÓN FISCAL (rev_5)")
    _l = ta.legitimacion_de("revision_fiscal", "JEFA DE LA UNIDAD JURÍDICA DE LA REPRESENTACIÓN ESTATAL QUERÉTARO "
                            "DEL ISSSTE (DEMANDADA)", autoridad_demandada="Subdelegado de Prestaciones Económicas")
    ok("en relación con el 5o., cuarto párrafo, de la Ley Federal de Procedimiento Contencioso Administrativo" in _l
       and "JEFA" not in _l and "(DEMANDADA)" not in _l.upper().replace("AUTORIDAD DEMANDADA", "")
       and "defensa jurídica del Subdelegado" in _l and "de el " not in _l,
       "la unidad jurídica: el 5o., cuarto párrafo, sin versales, sin la etiqueta de rol, «del» y no «de el»")
    _av = []
    _l = ta.legitimacion_de("revision_fiscal", "Jefe del Departamento de Pensiones del ISSSTE",
                            figura="Titular de la Unidad de Asuntos Jurídicos",
                            autoridad_demandada="Jefe del Departamento de Pensiones del ISSSTE", avisos=_av)
    ok("lo hizo valer el Jefe del Departamento de Pensiones del ISSSTE, autoridad demandada" in _l
       and "por conducto del Titular de la Unidad de Asuntos Jurídicos" in _l
       and "en su carácter de unidad administrativa" not in _l and not _av,
       "RF 6/2026: recurre la propia demandada, por conducto de la figura que firmó (aunque falte el nombre)")
    _av = []
    _l = ta.legitimacion_de("revision_fiscal", "Jefe del Departamento de Pensiones del ISSSTE",
                            autoridad_demandada="Jefe del Departamento de Pensiones del ISSSTE", avisos=_av)
    ok(f"por conducto de {H}" in _l and any("FALTA LA UNIDAD JURÍDICA QUE FIRMÓ EL OFICIO" in x for x in _av),
       "RF 21/2025: sin la figura, HUECO y aviso (antes: «X, en su carácter de unidad… de X»)")
    _av = []
    _l = ta.legitimacion_de("revision_fiscal", "SUBDELEGADO DE PRESTACIONES ECONÓMICAS, EN REPRESENTACIÓN DEL "
                            "SUBDELEGADO DE PRESTACIONES ECONÓMICAS", avisos=_av)
    ok(_l.count("Subdelegado de Prestaciones Económicas") == 1 and "en representación" not in _l,
       "RF 2/2025: «X, en representación de X» no se escribe dos veces")
    _av = []
    _l = ta.legitimacion_de("revision_fiscal", "Instituto de Seguridad y Servicios Sociales de los Trabajadores del "
                            "Estado", autoridad_demandada="Subdelegado de Prestaciones Económicas", avisos=_av)
    ok("en su carácter de unidad administrativa" not in _l and "en representación del Subdelegado" in _l
       and any("COMPRUEBA LA LEGITIMACIÓN" in x for x in _av),
       "RF 26/2025: al Instituto no se le atribuye ser la unidad de defensa jurídica")
    _l = ta.legitimacion_de("revision_fiscal", "Titular de la Unidad de Asuntos Jurídicos",
                            figura="Titular de la Unidad de Asuntos Jurídicos",
                            autoridad_demandada="Subdelegado de Prestaciones")
    ok(_l.count("Unidad de Asuntos Jurídicos") == 1,
       "RF 21/2025 (banco): la figura que ya es el nombre no se repite («por conducto de su titular de…»)")
    _l = ta.legitimacion_de("revision_fiscal", "Titular de la Unidad Jurídica de la Oficina de Representación en "
                            "Querétaro del Instituto de Seguridad y Servicios Sociales de los Trabajadores del Estado",
                            autoridad_demandada="Instituto de Seguridad y Servicios Sociales de los Trabajadores "
                                                "del Estado")
    ok("lo hizo valer el Titular de la Unidad Jurídica" in _l and "en su carácter de unidad administrativa" in _l
       and H not in _l,
       "RF 4/2025: la unidad que lleva el nombre del Instituto no es el Instituto (contener no es ser)")

_seccion(16, _s16)

def _s17():
    print("\n17 · LA PROCEDENCIA DE LA REVISIÓN FISCAL (rev_2, rev_5)")
    p, a = pf.parrafo("la resolución determinó un crédito fiscal por $95,000.00", 2026, fraccion="II",
                      cuantia="indeterminada")
    ok("el asunto es de cuantía indeterminada" in p and "95,000" not in p and H not in p,
       "fracción II con cuantía «indeterminada»: se escribe, sin tomar la cifra de la prosa")
    p, a = pf.parrafo("sin cifras", 2026, fraccion="I", cuantia="indeterminada")
    ok(any("CUANTÍA INDETERMINADA" in x and "fracción II" in x for x in a),
       "fracción I con cuantía indeterminada: aviso (la I exige cuantía)")
    p, a = pf.parrafo("El ISSSTE negó la pensión por jubilación", fecha_sentencia=D(2025, 5, 1), fraccion="VI",
                      sentido="declaró la nulidad de la resolución impugnada, para determinados efectos")
    ok("aportaciones" not in p and "pensiones que otorga el Instituto de Seguridad y Servicios Sociales" in p,
       "pensiones del ISSSTE: su propia hipótesis, sin llamarla «aportaciones»")
    ok(any("NULIDAD PARA EFECTOS" in x and "fondo o por vicios formales" in x for x in a),
       "nulidad para efectos en la fracción VI: aviso de nulidad formal o de fondo")
    p, a = pf.parrafo("sin supuesto", fecha_sentencia=D(2025, 5, 1), fraccion="VI")
    ok(p.endswith(f"versa sobre {H}.") and "aportaciones" not in p,
       "fracción VI con el supuesto en hueco: no presupone aportaciones")

_seccion(17, _s17)

def _s18():
    print("\n18 · LA FECHA DEL AUTO RECURRIDO EN LA QUEJA (D5)")
    _r = ("Por auto de trece de marzo de dos mil veinticinco, la Jueza Tercero de Distrito en el Estado de Querétaro "
          "desechó de plano la demanda de amparo 742/2025. Inconforme, Juan Pérez López interpuso recurso de queja. "
          "Por auto de Presidencia de veintidós de abril de dos mil veinticinco, este Tribunal Colegiado registró el "
          "recurso con el número 12/2025 y lo admitió a trámite.")
    ok(fo.fecha_de(_r) == "trece de marzo de dos mil veinticinco",
       "la candidata única con el verbo del auto recurrido («desechó») se toma (main la daba; la ronda 2, hueco)")
    ok(fo.fecha_de("Seguido el juicio, dictó sentencia el trece de marzo de dos mil veinticinco. Por auto de "
                   "Presidencia de veintidós de abril de dos mil veinticinco se admitió el recurso.") == "",
       "RF de Yucatán: el auto de Presidencia («se admitió») sigue sin tomarse")
    ok(fo.fecha_de("Por auto de diez de enero de dos mil veinticinco el Juez admitió la demanda. Por auto de cinco "
                   "de marzo de dos mil veinticinco el Juez negó la suspensión provisional.") == "",
       "dos autos con verbo y fechas distintas: no se adivina")

_seccion(18, _s18)


def _s19():
    print("\n19 · SIN LA BANDERA, EL CAMINO VIEJO (salvo los [siempre])")
    bandera(False)
    c = f0.computar(D(2025, 3, 19), D(2025, 4, 9), "personal", 15)
    p = f0.parrafo_oportunidad(c, "", "amparo_directo")
    ok(c.inhabiles_previos == [D(2025, 3, 21)] and "veintiuno de marzo" not in p,
       "los previos se calculan, pero el considerando viejo no los nombra")
    c = f0.computar(D(2024, 6, 30), D(2024, 7, 13), "lista", 15, TFJA, tipo_asunto="amparo_directo")
    ok(not any("CAE EN DÍA INHÁBIL" in a or "POR LISTA" in a for a in c.avisos),
       "ni los avisos de día inhábil ni el de «lista» ante el TFJA")
    ok(ta.caratula_de("queja")[0][0] == "RECURRENTE"
       and not any(cl == "adherente" for _e, cl, _o in ta.caratula_de("amparo_directo"))
       and not any(cl == "adherente" for _e, cl, _o in ta.caratula_de("revision_fiscal"))
       and any(cl == "adherente" for _e, cl, _o in ta.caratula_de("amparo_revision")),
       "el rubro de siempre: «RECURRENTE» en la queja y sin adherente en AD y RF (el AR ya lo tenía)")
    ok(ta.filas_caratula("amparo_revision", {"quejoso": "Unión Ejemplo, A.C.", "recurrente": "Unión Ejemplo, A.C."})
       == [("QUEJOSA", "quejoso", "Unión Ejemplo, A.C."), ("RECURRENTE", "recurrente", "Unión Ejemplo, A.C.")],
       "y la partición de la revisión, como estaba")
    ok(ta.filas_caratula("amparo_directo", {"quejoso": "Ana Pérez Ruiz y Luis Pérez Ruiz, ambos de apellidos "
                                                       "Pérez Ruiz", "tramite": {"tipo": "amparo_directo"},
                                            "responsable": "Sala Civil"})[0][0] == "QUEJOSO",
       "el rubro plural (cuarta ronda) es del camino nuevo: sin bandera, «QUEJOSO» como antes")
    p, a = pf.parrafo("El ISSSTE negó la pensión por jubilación. Se declara la nulidad lisa y llana",
                      fecha_sentencia=D(2025, 5, 1))
    ok("aportaciones de seguridad social relativa a un aspecto relacionado con pensiones" in p
       and not any("NULIDAD" in x for x in a),
       "la procedencia RF que se deduce de la sentencia, con la fórmula y los avisos de antes")
    # LOS [siempre]: el orden de los inhábiles (D6), la legitimación, la sede, la
    # fecha del auto recurrido (D5).
    c = f0.computar(D(2025, 12, 5), D(2026, 1, 8), "personal", 15)
    _cl = f0.clausula_inhabiles(c)
    ok(_cl.count("artículo 19") == 1 and "por corresponder al periodo vacacional" in _cl,
       "[siempre] cada grupo con su fundamento y el artículo 19 una sola vez")
    ok("artículo 9o." in ta.legitimacion_de("queja", "Director de Ingresos", "Juan Pérez", figura="delegado",
                                            papel="autoridad")
       and "persona moral oficial quejosa" in ta.legitimacion_de("amparo_directo", "Instituto Mexicano del "
                                                                  "Seguro Social", "Juan Pérez", figura="apoderado"),
       "[siempre] la legitimación corregida (art. 9o. del delegado; persona moral oficial)")
    ok(ta.es_cdmx("Décimo Tribunal Colegiado en Materia Civil del Primer Circuito", "Cd. de México") is True,
       "[siempre] la sede de la Ciudad de México")
    bandera(True)


_seccion(19, _s19)


# ══════════════════════════════════════════════════════════════════════════════
# CUARTA RONDA (FIXES_R4, dueño B)
# ══════════════════════════════════════════════════════════════════════════════
_ISS = "Instituto de Seguridad y Servicios Sociales de los Trabajadores del Estado"
_AD552 = ("PARTE QUEJOSA, SOCIEDAD DE PRODUCCIÓN RURAL DE RESPONSABILIDAD ILIMITADA, POR CONDUCTO DE SU "
          "CONSEJO DE ADMINISTRACIÓN, Y PARTE QUEJOSA, POR PROPIO DERECHO")


def _s20():
    print("\n20 · E2: UN SOLO DETECTOR DE PLURAL, Y LA LISTA ANTES QUE LA SOCIEDAD (AD 552, RF 7, Q 229)")
    ok(ta.es_plural_de_partes(_AD552) and ta.es_plural_de_partes(_AD552, moral=True),
       "AD 552/2024: «X, SOCIEDAD…, POR CONDUCTO DE…, Y Z, POR PROPIO DERECHO» es plural aunque haya sociedad")
    ok(ta.es_plural_de_partes("1) PARTE ACTORA y otros (ya mencionados con anterioridad)")
       and ta.es_plural_de_partes("MARÍA PÉREZ Y OTRA")
       and ta.es_plural_de_partes("Comercializadora X, S.A. de C.V.; Juan Pérez López")
       and ta.es_plural_de_partes("*** ********* *****, *** ********* ******, y *** *********")
       and ta.es_plural_de_partes("Juan Pérez López y coagraviados"),
       "RF 7/2025 y Q 229/2026: «y otros», «y otra», la numeración, el punto y coma, «, y» testado, coagraviados")
    ok(not ta.es_plural_de_partes("JUAN PÉREZ LÓPEZ, POR PROPIO DERECHO, Y EN REPRESENTACIÓN DE SU MENOR HIJO")
       and not ta.es_plural_de_partes("Juan Pérez López, Albacea Sustituto")
       and not ta.es_plural_de_partes("Juan Pérez López, Director General")
       and not ta.es_plural_de_partes("SUCESIÓN TESTAMENTARIA A BIENES DE MARÍA PÉREZ, A TRAVÉS DE SU ALBACEA JUAN RUIZ")
       and not ta.es_plural_de_partes("Aeropuertos y Servicios Auxiliares, S.A. de C.V.")
       and not ta.es_plural_de_partes("Juan Ortega y Gasset")
       and not ta.es_plural_de_partes("Juan Pérez López, codemandado")
       and not ta.es_plural_de_partes("Juan Pérez López, por propio derecho y en representación de todos sus hijos"),
       "ni «, y en representación de», ni el cargo tras el nombre, ni la sucesión con su albacea, ni la sociedad, "
       "ni «codemandado» en singular, ni «todos sus hijos»")
    ok(ta.genero_de_plural("X y Z, ambos de apellidos W") == "o" and ta.genero_de_plural("X y Z, todas de apellido W") == "a"
       and ta.genero_de_plural("Juan Pérez López y María Gómez Ruiz") == ""
       and ta.genero_de_plural("MARÍA PÉREZ Y OTRA") == "" and ta.genero_de_plural("Ana y Rosa", "a") == "a",
       "el género de varias personas sólo si el texto o la ficha lo dicen («y otra» no dice el del primero)")
    _l = ta.legitimacion_de("amparo_directo", _AD552, moral=True)
    ok("; la parte quejosa está legitimada" in _l and "quien está" not in _l and "la persona moral" not in _l,
       "AD 552/2024: la legitimación ya no dice «quien está legitimada» de dos quejosos (el resultando dice «promovieron»)")
    _l = ta.legitimacion_de("amparo_directo", "Ana Pérez Ruiz y Luis Pérez Ruiz, ambos de apellidos Pérez Ruiz")
    ok(", quienes están legitimados para ello" in _l and "les causa" in _l,
       "AR 208/2025 (la forma del oro): «ambos» → «quienes están legitimados… les causa»")
    _l = ta.legitimacion_de("queja", "Ana López Ruiz", plural=True, papel="quejoso")
    ok("; la parte recurrente está legitimada" in _l,
       "`plural` (lo que expuso el compositor en datos_extra) manda sobre el detector")
    _tr = {"tipo": "amparo_directo"}
    _f = ta.filas_caratula("amparo_directo", {"quejoso": _AD552, "quejoso_moral": True, "tramite": _tr,
                                              "responsable": "Sala Civil"})
    ok(_f[0][0] == "PARTE QUEJOSA",
       "AD 552/2024: la carátula no dice «QUEJOSA» de dos quejosos (el género de la lista no consta)")
    _f = ta.filas_caratula("amparo_directo", {"quejoso": "Ana Pérez Ruiz y Luis Pérez Ruiz, ambos de apellidos Pérez "
                                                         "Ruiz", "tramite": _tr, "responsable": "Sala Civil"})
    ok(_f[0][0] == "QUEJOSOS", "con «ambos», «QUEJOSOS:» como el oro de AD 552")
    _f = ta.filas_caratula("amparo_revision", {"quejoso": "Ana Pérez Ruiz y Luis Pérez Ruiz, ambos de apellidos Pérez "
                                                          "Ruiz", "recurrente": "Gobernador del Estado de Querétaro",
                                               "papel_recurrente": "autoridad"})
    ok(_f[0] == ("QUEJOSOS", "quejoso", "Ana Pérez Ruiz y Luis Pérez Ruiz, ambos de apellidos Pérez Ruiz")
       and _f[1][0] == "AUTORIDAD RESPONSABLE Y RECURRENTE",
       "AR 208/2025: «QUEJOSOS» y «AUTORIDAD RESPONSABLE Y RECURRENTE»")
    _f = ta.filas_caratula("queja", {"quejoso": "MARÍA PÉREZ Y OTRA", "recurrente": "MARÍA PÉREZ Y OTRA",
                                     "papel_recurrente": "quejoso", "responsable": "Juzgado Primero"})
    ok(_f[0][0] == "PARTE QUEJOSA Y RECURRENTE", "Q 229/2026: «y otra» sin género que conste → «PARTE QUEJOSA»")
    _f = ta.filas_caratula("amparo_directo", {"quejoso": "Ana López Ruiz", "plural": True, "tramite": _tr,
                                              "responsable": "Sala Civil"})
    ok(_f[0][0] == "PARTE QUEJOSA" and ta.filas_caratula(
        "amparo_directo", {"quejoso": "Ana y Rosa López Ruiz, todas de apellido López", "tramite": _tr,
                           "responsable": "Sala"})[0][0] == "QUEJOSAS",
       "en el AD, `datos['plural']` del compositor manda; «todas» → «QUEJOSAS»")

_seccion(20, _s20)


def _s21():
    print("\n21 · E12: LA REPRESENTACIÓN DENTRO DEL NOMBRE (Q 342/2025)")
    _suc = "Sucesión Testamentaria a Bienes de María Pérez López, a través de su albacea Juan Ruiz Gómez"
    _l = ta.legitimacion_de("queja", _suc, papel="quejoso")
    ok("Juan Ruiz Gómez; la parte recurrente está legitimada" in _l and ", quien está legitimad" not in _l,
       "«…, a través de su albacea X» → «; la parte recurrente está legitimada» (el «quien» se leía como el albacea)")
    _l = ta.legitimacion_de("queja", _suc, "Juan Ruiz Gómez", figura="representante legal", papel="quejoso")
    ok("representante legal" not in _l and _l.count("Juan Ruiz Gómez") == 1,
       "el albacea que ya va en el nombre no se repite «por conducto de su representante legal»")
    _l = ta.legitimacion_de("amparo_revision", "Ana López Ruiz, por propio derecho y en representación de su hija",
                            papel="quejoso")
    ok("; la parte recurrente se encuentra legitimada" in _l,
       "«por propio derecho y en representación de…» también: la fórmula de la parte")

_seccion(21, _s21)


def _s22():
    print("\n22 · E9: EL ARTÍCULO 12 (O 9o.) SIN SU LEY (Q 172, Q 300, AR 239, AR 60)")
    ok(ta.personeria_de("quejoso", "autorizado conforme al artículo 12") == "en términos del artículo 12 de la Ley de Amparo"
       and ta.personeria_de("quejoso", "autorizado en términos del artículo 12 de la Ley de Amparo") == ""
       and ta.personeria_de("autoridad", "delegado en términos del artículo 9o.") == "en términos del artículo 9o. de la Ley de Amparo",
       "la personería sólo calla si la figura ya cita la LEY (antes bastaba «artículo 12»)")
    ok(ta.figura_en_prosa("autorizado en términos amplios del artículo 12")
       == "autorizado en términos amplios del artículo 12 de la Ley de Amparo"
       and ta.figura_en_prosa("delegado en términos del artículo 9o.") == "delegado en términos del artículo 9o. de la Ley de Amparo"
       and ta.figura_en_prosa("autorizado en términos del artículo 5", ley_de_amparo=False) == "autorizado en términos del artículo 5",
       "la figura se completa con «de la Ley de Amparo» (en la revisión fiscal, no)")
    _l = ta.legitimacion_de("queja", "Parte Recurrente", "Representante",
                            figura="autorizado en términos amplios del artículo 12", papel="quejoso")
    ok("del artículo 12 de la Ley de Amparo, Representante;" in _l and _l.count("artículo 12") == 1,
       "Q 172/Q 300: «…del artículo 12 de la Ley de Amparo, Representante» y el 12 una sola vez")

_seccion(22, _s22)


def _s23():
    print("\n23 · E8: LA UNIDAD QUE RECURRE Y SU TITULAR (RF 7, 2, 26, 4 y 6/2026)")
    ok(ta.misma_autoridad("Subdelegado de Prestaciones Económicas, Delegación Estatal Querétaro, del " + _ISS,
                          "Subdelegación de Prestaciones Económicas del ISSSTE en Querétaro")
       and ta.misma_autoridad("Director de Ingresos", "Dirección de Ingresos")
       and ta.misma_autoridad("Jefe de Servicios Jurídicos", "Jefatura de Servicios Jurídicos")
       and ta.misma_autoridad("Titular de la Unidad Jurídica de la Oficina de Representación",
                              "Unidad Jurídica de la Oficina de Representación"),
       "RF 6/2026: el cargo y su órgano son la misma autoridad (Subdelegado/Subdelegación, Titular de la X/X)")
    ok(not ta.misma_autoridad("Jefe del Departamento de Pensiones", "Jefe del Departamento de Afiliación")
       and not ta.misma_autoridad("Unidad Jurídica de la Delegación Estatal", "Delegación Estatal")
       and not ta.misma_autoridad("Titular de la Unidad Jurídica de la Oficina de Representación en Querétaro del "
                                  + _ISS, _ISS)
       and not ta.misma_autoridad("Subdirector de lo Contencioso del " + _ISS,
                                  "Subdirectora de Afiliación y Vigencia de Derechos, del " + _ISS),
       "sin confundir dos departamentos, la unidad con su delegación ni la unidad con el Instituto")
    _av = []
    _l = ta.legitimacion_de("revision_fiscal", "JEFA DE LA UNIDAD JURÍDICA DE LA DELEGACIÓN ESTATAL DEL ISSSTE EN BAJA "
                            "CALIFORNIA SUR, EN REPRESENTACIÓN DEL SUBDELEGADO DEL DEPARTAMENTO DE PENSIONES",
                            "María López Ruiz", H, figura="Jefa de la Unidad Jurídica", avisos=_av)
    ok("lo hizo valer María López Ruiz, Jefa de la Unidad Jurídica de la Delegación Estatal del ISSSTE en Baja "
       "California Sur, en representación del Subdelegado del Departamento de Pensiones" in _l
       and "por conducto de María" not in _l and "en su carácter de unidad" not in _l,
       "RF 7/2025: la titular que firma por su unidad va en aposición, como el corpus («lo promovió X, Jefa de…»)")
    _av = []
    _l = ta.legitimacion_de("revision_fiscal", "JEFA DE LA UNIDAD JURÍDICA DE LA REPRESENTACIÓN ESTATAL DEL ISSSTE, EN "
                            "REPRESENTACIÓN DEL SUBDELEGADO DE PRESTACIONES ECONÓMICAS", "María López Ruiz", H,
                            figura="Titular de la Unidad Jurídica…", avisos=_av)
    ok("…" not in _l and "lo hizo valer María López Ruiz, Jefa de la Unidad Jurídica" in _l
       and sum("DATO TRUNCADO (figura_representante" in x for x in _av) == 1 and len(_av) == 1,
       "RF 2/2025: la figura truncada no se firma (un aviso, con la clave del dato)")
    _av = []
    _l = ta.legitimacion_de("revision_fiscal", "Subdelegado de Prestaciones Económicas, Delegación Estatal Querétaro, "
                            "del " + _ISS, "Titular de la Unidad de Asuntos Jurídicos de la citada delegación", H,
                            figura="Titular de la Unidad Jurídica…",
                            autoridad_demandada="Subdelegación de Prestaciones Económicas del ISSSTE en Querétaro",
                            avisos=_av)
    ok("por conducto del Titular de la Unidad de Asuntos Jurídicos de la citada delegación." in _l
       and "en representación de la Subdelegación" not in _l and "…" not in _l and "su Titular" not in _l,
       "RF 6/2026: el representante que es un cargo es la figura, y «X en representación de X» ya no sale")
    _av = []
    _l = ta.legitimacion_de("revision_fiscal", "Titular de la Unidad Jurídica de la Oficina de representación en "
                            "Querétaro del " + _ISS, "Juan Pérez López", H,
                            figura="autorizado en términos del artículo 5 de la Ley Federal de Procedimiento "
                                   "Contencioso Administrativo", autoridad_demandada=_ISS, avisos=_av)
    ok("autorizado" not in _l and "Juan Pérez" not in _l and "en su carácter de unidad administrativa" in _l
       and not any("AUTORIZADO" in x for x in _av),
       "RF 4/2025: el autorizado no interpone por la autoridad: fuera del párrafo (el aviso es de validar, E11)")
    _l = ta.legitimacion_de("revision_fiscal", "INSTITUTO DE SEGURIDAD Y SERVICIOS SOCIALES DE LOS TRABAJADORES DEL "
                            "ESTADO", "Juan Pérez López", H, figura="Subdirector de lo Contencioso",
                            autoridad_demandada="Subdirectora de Afiliación y Vigencia de Derechos")
    ok("por conducto de su Subdirector de lo Contencioso, Juan Pérez López" in _l,
       "RF 26/2025: la figura con su mayúscula y la coma, como el resultando del mismo documento")
    _l = ta.legitimacion_de("revision_fiscal", "la Titular de la Unidad Jurídica de la Oficina de Representación del "
                            + _ISS, autoridad_demandada="Subdelegado de Prestaciones")
    ok("lo hizo valer la Titular de la Unidad Jurídica" in _l, "RF 49/2025: «la Titular» del papel se conserva")

_seccion(23, _s23)


def _s24():
    print("\n24 · E10: EL RÓTULO DEL EXPEDIENTE (Q 172/2026, AR 307/2024)")
    ok(ta.encabezado_de("queja", "administrativa", "172/2026") == "RECURSO DE QUEJA ADMINISTRATIVA: 172/2026",
       "Q 172/2026: «RECURSO DE QUEJA ADMINISTRATIVA» (14 de 14 en el corpus), no «ADMINISTRATIVO»")
    ok(ta.encabezado_de("amparo_revision", "familiar", "307/2024") == "AMPARO EN REVISIÓN CIVIL: 307/2024"
       and ta.encabezado_de("amparo_directo", "familiar", "1/2026") == "AMPARO DIRECTO CIVIL: 1/2026",
       "AR 307/2024: lo familiar se rotula CIVIL en el rótulo del expediente")
    ok(ta.encabezado_de("amparo_directo", "agraria", "1/2026") == "AMPARO DIRECTO AGRARIO: 1/2026"
       and ta.encabezado_de("amparo_revision", "administrativo", "1/2026") == "AMPARO EN REVISIÓN ADMINISTRATIVA: 1/2026"
       and ta.encabezado_de("amparo_directo", "administrativa", "1/2026") == "AMPARO DIRECTO ADMINISTRATIVO: 1/2026",
       "la materia concuerda con el tipo aunque llegue en el otro género")

_seccion(24, _s24)


def _s25():
    print("\n25 · E12: EL PROMULGADOR QUE RECURRE EN EL AMPARO CONTRA NORMAS (AR 208 y 72/2025)")
    _l = ta.legitimacion_de("amparo_revision", "gobernador del Estado de Querétaro", "Juan Ruiz Soto",
                            figura="delegado", papel="autoridad", hay_norma=True)
    ok("87, primer párrafo, de la Ley de Amparo" in _l and "emisión o promulgación" in _l
       and "afecta directamente el acto" not in _l,
       "AR 208/2025: el Gobernador con la norma impugnada, por la hipótesis de los titulares que la promulgan")
    ok("afecta directamente el acto" in ta.legitimacion_de("amparo_revision", "Gobernador del Estado de Querétaro",
                                                           papel="autoridad")
       and "afecta directamente el acto" in ta.legitimacion_de("amparo_revision", "Secretario Ejecutivo del Sistema",
                                                               papel="autoridad", hay_norma=True)
       and ta.es_promulgador("Legislatura del Estado de Querétaro") and not ta.es_promulgador("Secretario Ejecutivo"),
       "sin norma, o sin promulgador, la primera hipótesis de siempre")
    ok("87, primer párrafo" in ta.legitimacion_de("amparo_revision", "Legislatura del Estado de Querétaro",
                                                  papel="autoridad",
                                                  norma_impugnada="un acto reclamado es una norma general")
       and "87, primer párrafo" not in ta.legitimacion_de("amparo_revision", "Legislatura del Estado de Querétaro",
                                                          papel="autoridad", norma_impugnada=""),
       "el compositor pasa la razón de `norma_impugnada` con ese nombre; vacía, no cuenta")

_seccion(25, _s25)


def _s26():
    print("\n26 · E12: EL SURTIMIENTO EL MISMO DÍA NO REPITE LA FECHA (Q 172 y 24/2026)")
    c = f0.computar(D(2026, 3, 31), D(2026, 4, 8), "electronica", 10, tipo_asunto="queja")
    p = f0.parrafo_oportunidad(c, "", "queja", papel="quejoso")
    ok("surtió efectos el mismo día, conforme al artículo 31, fracción III, de la Ley de Amparo, por lo que" in p
       and "es decir, el treinta y uno de marzo" not in p,
       "«surtió efectos el mismo día, conforme al artículo 31, fracción III…», sin «, es decir, el {misma fecha}»")
    c = f0.computar(D(2026, 3, 9), D(2026, 3, 20), "personal", 10, tipo_asunto="queja")
    ok("es decir, el diez de marzo" in f0.parrafo_oportunidad(c, "", "queja", papel="quejoso"),
       "con surtimiento al día siguiente, la fecha sí se dice")

_seccion(26, _s26)


def _s27():
    print("\n27 · E1: LA LEGITIMACIÓN PASA EL NOMBRE POR LA PUERTA DEL COMPOSITOR")
    import resultandos_por_tipo as _rpt
    _orig = _rpt.nombre_en_prosa
    _llamadas = []

    def _puerta(n, autoridad=False, con_articulo=False):
        _llamadas.append((n, autoridad, con_articulo))
        return ("el Órgano De Prueba" if autoridad else "la Sucesión De Prueba")
    try:
        _rpt.nombre_en_prosa = _puerta
        _o = ta.con_articulo_de_organo("DIRECTOR DE PRUEBA")
        _l = ta.legitimacion_de("amparo_directo", "SUCESIÓN A BIENES DE JUAN PÉREZ")
    finally:
        _rpt.nombre_en_prosa = _orig
    ok(_o == "el Órgano De Prueba" and "promovida por la Sucesión De Prueba" in _l
       and any(c_ for _n, a_, c_ in _llamadas),
       "`con_articulo_de_organo` y la legitimación llaman a `resultandos_por_tipo.nombre_en_prosa` (con_articulo)")

    def _vieja(n, autoridad=False):
        return n.title()
    try:
        _rpt.nombre_en_prosa = _vieja
        _l = ta.legitimacion_de("amparo_directo", "SUCESIÓN A BIENES DE JUAN PÉREZ")
    finally:
        _rpt.nombre_en_prosa = _orig
    ok("promovida por la Sucesión A Bienes De Juan Pérez" in _l,
       "si la puerta aún no acepta `con_articulo`, se llama sin él y el artículo lo pone `con_articulo_de_colectivo`")

_seccion(27, _s27)


def _s28():
    print("\n28 · LA CARÁTULA SIN COLETILLAS NI PREFIJO TEMPORAL (RF 7 y 49/2025)")
    # C4 (3-oct-2026): la revisión fiscal ya no lleva el renglón de la Sala; el
    # prefijo temporal se comprueba en el de la autoridad del amparo directo,
    # que sí lo lleva (antes: «SALA RESPONSABLE» de la revisión fiscal).
    _f = ta.filas_caratula("revision_fiscal", {"quejoso": "JEFA DE LA UNIDAD JURÍDICA",
                                               "tercero": "1) PARTE ACTORA Y OTROS (YA MENCIONADOS CON ANTERIORIDAD)",
                                               "responsable": "ACTUAL SALA REGIONAL EN QUERÉTARO DEL TFJA"})
    _fd = ta.filas_caratula("amparo_directo", {"quejoso": "JUAN PÉREZ LÓPEZ",
                                               "responsable": "ACTUAL SALA REGIONAL EN QUERÉTARO DEL TFJA"})
    ok(("PARTE ACTORA", "tercero", "PARTE ACTORA Y OTROS") in _f
       and not any(c_ == "responsable" for _e, c_, _v in _f)
       and ("AUTORIDAD RESPONSABLE", "responsable", "SALA REGIONAL EN QUERÉTARO DEL TFJA") in _fd,
       "ni «1)», ni «(YA MENCIONADOS CON ANTERIORIDAD)», ni «ACTUAL» en el rubro")
    ok(ta.sin_coletilla_de_caratula("Juan Pérez López") == "Juan Pérez López"
       and ta.sin_coletilla_de_caratula("Actual Sala Regional", temporal=False) == "Actual Sala Regional",
       "lo demás no se toca (y el prefijo sólo en el renglón de la Sala)")

_seccion(28, _s28)


# ═══ QUINTA RONDA (FIXES_R5, dueño B) ═══════════════════════════════════════

def _s29():
    print("\n29 · LA REPRESENTACIÓN DENTRO DEL NOMBRE SIN COMA (AD 335/2025)")
    _l = ta.legitimacion_de("amparo_directo",
                            "la Sucesión a Bienes de JOSÉ GARCÍA RUIZ a través de su albacea María López Díaz")
    ok("María López Díaz; la parte quejosa está legitimada" in _l and ", quien está legitimad" not in _l,
       "AD 335/2025: sin la coma de la carátula, el «quien» ya no se lee como el albacea")
    _l = ta.legitimacion_de("amparo_directo", "Comercializadora del Centro, S.A. de C.V.")
    ok(", quien está legitimada" in _l, "sin representación en el nombre, el molde de siempre")

_seccion(29, _s29)


def _s30():
    print("\n30 · F2: LA FIGURA SIN NOMBRE NO SE BORRA (AR 72/2025, Q 342/2025)")
    _av = []
    _l = ta.legitimacion_de("amparo_revision", "Gobernador del Estado de Querétaro", "", H, figura="delegado",
                            papel="autoridad", avisos=_av)
    ok("interpuesto por el Gobernador del Estado de Querétaro, por conducto de su delegado *********, en "
       "términos del artículo 9o. de la Ley de Amparo; la autoridad recurrente se encuentra legitimada" in _l,
       "AR 72/2025: el delegado sin nombre va con HUECO y su personería (9o.), no se calla")
    ok(len(_av) == 1 and _av[0].startswith("FALTA EL NOMBRE DEL REPRESENTANTE (representante)")
       and "delegado" in _av[0],
       "…con UN aviso con la clave «representante» (la que deduplica el compositor, E11)")
    _av = []
    _l = ta.legitimacion_de("queja", "la Sucesión Testamentaria a Bienes de Pedro Ruiz Soto, a través de su albacea "
                            "Juan Ruiz Gómez", "", H, figura="autorizado", papel="quejoso", avisos=_av)
    ok("a través de su albacea Juan Ruiz Gómez, por conducto de su autorizado *********, en términos del artículo "
       "12 de la Ley de Amparo; la parte recurrente está legitimada" in _l and len(_av) == 1,
       "Q 342/2025: la sucesión por su albacea, y firmó el autorizado del 12 (lo que analiza el engrose)")
    _av = []
    _l = ta.legitimacion_de("queja", "Ana López Ruiz, por propio derecho y en representación de su hija", "", H,
                            figura="representante legal", papel="quejoso", avisos=_av)
    ok("*********" not in _l and not _av,
       "quien actúa por sí y por su hija no lleva un «representante legal» en hueco")
    _av = []
    _l = ta.legitimacion_de("queja", "la Sucesión a Bienes de Pedro Ruiz Soto, a través de su albacea Juan Ruiz "
                            "Gómez", "", H, figura="albacea", papel="quejoso", avisos=_av)
    ok("*********" not in _l and not _av, "la figura que ya va con su nombre dentro del de la parte, tampoco")
    _l = ta.legitimacion_de("amparo_revision", "Gobernador del Estado de Querétaro", "", H, figura="delegado",
                            papel="autoridad")
    ok("delegado" not in _l and "*********" not in _l,
       "sin `avisos` (el camino viejo, que no los pide) el párrafo queda como estaba")

_seccion(30, _s30)


def _s31():
    print("\n31 · F6: UNA SOLA REGLA PARA LA FIGURA Y SU SEPARADOR (Q 172 y 300, AD 456 y 274)")
    ok(ta.sep_figura("apoderado legal") == " " and ta.sep_figura("autorizada en términos amplios") == ", "
       and ta.sep_figura("delegado en términos del artículo 9o. de la Ley de Amparo") == ", ",
       "`sep_figura` pública: coma si cita un artículo o pasa de tres palabras")
    _l = ta.legitimacion_de("queja", "Parte Recurrente", "Representante", H, figura="autorizada en términos amplios",
                            papel="quejoso")
    ok("por conducto de su autorizada en términos amplios, Representante, conforme al artículo 12 de la Ley de "
       "Amparo;" in _l and "amplios Representante" not in _l,
       "Q 172/2026: la legitimación pone la coma como el resultando y no repite «en términos»")
    ok(ta.personeria_de("quejoso", "apoderado en términos de los artículos 6o. y 11 de la Ley de Amparo") == ""
       and ta.personeria_de("tercero", "representante conforme al artículo 11 de la Ley de Amparo") == "",
       "personeria_de calla con CUALQUIER figura que ya cite la Ley de Amparo")
    ok(ta.personeria_de("quejoso", "apoderado conforme al artículo 10 de la Ley General de Sociedades Mercantiles")
       == "en términos de los artículos 6o. y 11 de la Ley de Amparo",
       "…pero la cita de otra ley no dice la personería en el amparo")
    ok(ta.figura_en_prosa("Representante Legal conforme al artículo 12")
       == "representante legal conforme al artículo 12 de la Ley de Amparo"
       and ta.figura_en_prosa("MADRE") == "madre" and ta.figura_en_prosa("Progenitora") == "progenitora",
       "«artículo 12» sin ley se completa en cualquier figura que lo cite al final; madre/progenitora en minúscula")

_seccion(31, _s31)


def _s32():
    print("\n32 · LA CARÁTULA SIN NINGUNA NUMERACIÓN NI «POR PROPIO DERECHO» (RF 7 y 2/2025)")
    _lista = ", ".join(f"{i}) PARTE ACTORA" for i in range(1, 66)) + " Y 66) PARTE ACTORA"
    _v = ta.sin_coletilla_de_caratula(_lista)
    ok(")" not in _v and _v.startswith("PARTE ACTORA, PARTE ACTORA") and _v.endswith("PARTE ACTORA Y PARTE ACTORA"),
       "RF 7/2025: se quitan las 66 numeraciones, no sólo el «1)»")
    ok(ta.sin_coletilla_de_caratula("1) JUAN PÉREZ LÓPEZ, 2) ANA RUIZ SOTO Y 3) LUIS GIL MORA; POR PROPIO DERECHO.")
       == "JUAN PÉREZ LÓPEZ, ANA RUIZ SOTO Y LUIS GIL MORA",
       "RF 2/2025: «; POR PROPIO DERECHO» no es parte del nombre")
    ok(ta.sin_coletilla_de_caratula("JUAN PÉREZ, POR PROPIO DERECHO, Y ANA RUIZ, POR PROPIO DERECHO")
       == "JUAN PÉREZ Y ANA RUIZ"
       and ta.sin_coletilla_de_caratula("ANA LÓPEZ, POR PROPIO DERECHO Y EN REPRESENTACIÓN DE SU MENOR HIJA")
       == "ANA LÓPEZ, POR PROPIO DERECHO Y EN REPRESENTACIÓN DE SU MENOR HIJA"
       and ta.sin_coletilla_de_caratula("BANCO DEL CENTRO, S.A. DE C.V.") == "BANCO DEL CENTRO, S.A. DE C.V.",
       "la fórmula de cada uno sale; la que dice a quién más representa, y la razón social, se quedan")

_seccion(32, _s32)


def _s33():
    print("\n33 · EL ARTÍCULO DE LA PROSA ANTE UN CARGO EN MINÚSCULA (RF 49/2025)")
    ok(ta.sin_articulo_de_prosa("la titular de la Unidad Jurídica de la Oficina de Representación")
       == "Titular de la Unidad Jurídica de la Oficina de Representación"
       and ta.sin_articulo_de_prosa("la Jefa de la Unidad Jurídica") == "Jefa de la Unidad Jurídica"
       and ta.sin_articulo_de_prosa("La Costeña, S.A. de C.V.") == "La Costeña, S.A. de C.V."
       and ta.sin_articulo_de_prosa("la quejosa") == "la quejosa",
       "«la titular de la Unidad…» pierde el artículo como «la Jefa…»; «La Costeña» y lo demás, no")
    _f = ta.filas_caratula("revision_fiscal", {"quejoso": "la titular de la Unidad Jurídica de la Oficina de "
                                                          "Representación del " + _ISS,
                                               "responsable": "Sala Regional en Querétaro"})
    ok(_f and _f[0][2].startswith("Titular de la Unidad Jurídica"),
       "RF 49/2025: el renglón de la autoridad recurrente sin «LA»")

_seccion(33, _s33)


def _s34():
    print("\n34 · EL PLURAL DE QUIEN RECURRE ES EL DEL COMPOSITOR (AR 208/2025, regresión)")
    _q = "Juan Pérez López y Pedro Pérez López, ambos de apellidos Pérez López"
    # Lo que llega del documento: `datos["plural"]` copiado del de la quejosa, y
    # en `procesal` los `datos_extra` del compositor.
    _f = ta.filas_caratula("amparo_revision", {
        "quejoso": _q, "recurrente": "Gobernador del Estado de Querétaro", "papel_recurrente": "autoridad",
        "plural": True, "procesal": {"plural": False, "plural_quejoso": True}})
    ok(_f[0][0] == "QUEJOSOS" and _f[1][0] == "AUTORIDAD RESPONSABLE Y RECURRENTE",
       "AR 208/2025: un Gobernador con dos quejosos ya no sale «AUTORIDADES RESPONSABLES»")
    _f = ta.filas_caratula("amparo_revision", {
        "quejoso": "Ana López Ruiz", "recurrente": "Juan Gil Mora y Pedro Gil Mora, ambos de apellidos Gil Mora",
        "papel_recurrente": "tercero", "plural": False,
        "procesal": {"plural": True, "plural_quejoso": False}})
    ok(_f[0][0] == "QUEJOSA" or _f[0][0] == "PARTE QUEJOSA", "una quejosa sigue en singular")
    ok(_f[1][0] == "TERCEROS INTERESADOS Y RECURRENTES",
       "y al revés: dos terceros recurrentes con una quejosa salen en plural")

_seccion(34, _s34)


def _s35():
    print("\n35 · F1: LA FORMA DE NOTIFICACIÓN QUE NO CONSTA NO SE AFIRMA (Q 335, AD 274 y 335)")
    c = f0.computar(D(2025, 10, 6), D(2025, 10, 10), "personal", 5, tipo_asunto="queja")
    _p = f0.parrafo_oportunidad(c, "", "queja", papel="quejoso", fuente_forma="omision")
    ok("el auto recurrido se notificó a la parte recurrente el seis de octubre de dos mil veinticinco y surtió "
       "efectos al día hábil siguiente, conforme al artículo 31, fracción II, de la Ley de Amparo" in _p
       and "de manera personal" not in _p,
       "Q 335/2025: «se notificó… el {fecha} y surtió efectos al día hábil siguiente», sin «de manera personal»")
    ok("de manera personal y surtió" in f0.parrafo_oportunidad(c, "", "queja", papel="quejoso"),
       "con la forma declarada (o sin decir la fuente), como antes")
    _av = f0.aviso_forma_no_consta(c, "queja", "quejoso")
    ok(_av.startswith("LA FORMA DE NOTIFICACIÓN NO CONSTA") and "personal" in _av
       and "artículo 31, fracción II" in _av and "constancia" in _av,
       "el aviso dice con qué regla se contó y que se compruebe en la constancia")
    c = f0.computar(D(2025, 10, 6), D(2025, 10, 10), "oficio", 10, tipo_asunto="amparo_revision")
    _p = f0.parrafo_oportunidad(c, "", "amparo_revision", papel="autoridad", fuente_forma="omision_autoridad")
    ok("a la autoridad recurrente el seis de octubre de dos mil veinticinco y surtió efectos el mismo día, conforme "
       "al artículo 31, fracción I, de la Ley de Amparo" in _p and "por oficio" not in _p,
       "D2 sin constancia: «surtió efectos el mismo día» con el 31-I, sin «por oficio»")
    c = f0.computar(D(2025, 3, 19), D(2025, 4, 9), "personal", 15)
    _p = f0.parrafo_oportunidad(c, "", "amparo_directo", sin_precepto_en_hueco=True, forma_consta=False)
    _a = f0.aviso_fundamento(c, "amparo_directo", en_hueco=True, forma_consta=False)
    ok("el diecinueve de marzo de dos mil veinticinco y surtió efectos al día hábil siguiente" in _p
       and "de manera personal" not in _p and "de manera personal" not in _a,
       "AD 274/335: ni el párrafo ni el aviso del precepto repiten la forma que no consta")

_seccion(35, _s35)


def _s36():
    print("\n36 · F4 (GUARDA): LA MISMA UNIDAD CON OTRA ADSCRIPCIÓN (RF 7/2025)")
    _fig = "Jefa de la Unidad Jurídica de la Representación Estatal del " + _ISS + " en Baja California Sur"
    _pro = ("JEFA DE LA UNIDAD JURÍDICA DE LA DELEGACIÓN ESTATAL DEL ISSSTE EN BAJA CALIFORNIA SUR, EN "
            "REPRESENTACIÓN DEL SUBDELEGADO DEL DEPARTAMENTO DE PENSIONES")
    ok(ta.misma_autoridad(_fig, "la Jefa de la Unidad Jurídica de la Delegación Estatal del " + _ISS
                          + " en Baja California Sur, en representación del Subdelegado"),
       "«Delegación Estatal» y «Representación Estatal» de la misma unidad: la misma autoridad (por núcleo)")
    _l = ta.legitimacion_de("revision_fiscal", _pro, "María Luisa Gómez Ríos", H, figura=_fig, avisos=[])
    ok("lo hizo valer María Luisa Gómez Ríos, Jefa de la Unidad Jurídica de la Delegación Estatal" in _l
       and "por conducto de su Jefa" not in _l,
       "RF 7/2025: la legitimación pone el nombre en aposición, sin «X, por conducto de su X»")

_seccion(36, _s36)


def _s37():
    print("\n37 · F5: EL ARTÍCULO DEL PAPEL EN LOS CARGOS DE DOBLE GÉNERO (AD 128/2025)")
    # CON LA PUERTA DE ANTES (`nombre_en_prosa` cambiaba «la Oficial» por «el
    # Oficial», verif_final2): la legitimación conserva el artículo del papel
    # aunque el convertidor lo cambie.
    import resultandos_por_tipo as _rpt
    _orig = _rpt.nombre_en_prosa

    def _puerta_vieja(n, autoridad=False, con_articulo=False):
        n = " ".join(str(n or "").split())
        n = n[3:] if n.lower().startswith(("la ", "el ")) else n
        return "el " + n if autoridad else n
    try:
        _rpt.nombre_en_prosa = _puerta_vieja
        _o = ta.con_articulo_de_organo("la Oficial Mayor del Municipio de Cadereyta de Montes, Querétaro")
        _l = ta.legitimacion_de("amparo_revision", "la Oficial Mayor del Municipio de Cadereyta", papel="autoridad")
    finally:
        _rpt.nombre_en_prosa = _orig
    ok(_o.startswith("la Oficial Mayor") and "interpuesto por la Oficial Mayor" in _l,
       "AD 128/2025: «la Oficial Mayor» del papel se conserva aunque la puerta diga «el»")
    ok(ta.con_articulo_de_organo("la Oficial Mayor del Municipio de Cadereyta de Montes, Querétaro")
       .startswith("la Oficial Mayor")
       and ta.con_articulo_de_organo("LA FISCAL GENERAL DEL ESTADO").startswith("la Fiscal")
       and ta.con_articulo_de_organo("la Agente del Ministerio Público").startswith("la Agente"),
       "AD 128/2025: «la Oficial Mayor» queda «la Oficial Mayor» (antes «el»), y la Fiscal y la Agente")
    ok(ta.con_articulo_de_organo("Oficial Mayor del Municipio").startswith("el Oficial")
       and ta.articulo_del_papel("Oficial Mayor") == "" and ta.articulo_del_papel("la Titular de la Unidad") == "la",
       "sin artículo en el papel, el masculino genérico de siempre")
    _l = ta.legitimacion_de("amparo_revision", "la Oficial Mayor del Municipio de Cadereyta", papel="autoridad")
    ok("interpuesto por la Oficial Mayor del Municipio de Cadereyta" in _l,
       "la legitimación de la autoridad que recurre también")

_seccion(37, _s37)


def _s38():
    print("\n38 · «CONFORME A LOS ARTÍCULOS 65 Y 70» (regla «Personal ante el TFJA»)")
    c = f0.computar(D(2024, 12, 11), D(2025, 1, 8), "lfpca", 15, tipo_asunto="revision_fiscal")
    _p = f0.parrafo_oportunidad(c, "", "revision_fiscal", papel="autoridad")
    ok("conforme a los artículos 65 y 70 de la Ley Federal de Procedimiento Contencioso Administrativo" in _p
       and "conforme al artículos" not in _p, "RF: la preposición concuerda con el plural del fundamento")
    c = f0.computar(D(2025, 3, 19), D(2025, 4, 9), "lfpca", 15, TFJA)
    _p = f0.parrafo_oportunidad(c, "", "amparo_directo", sin_precepto_en_hueco=True)
    ok("conforme a los artículos 65 y 70" in _p and "conforme al artículos" not in _p,
       "AD contra el TFJA con la misma regla")
    c = f0.computar(D(2025, 10, 6), D(2025, 10, 10), "personal", 5, tipo_asunto="queja")
    ok("conforme al artículo 31, fracción II" in f0.parrafo_oportunidad(c, "", "queja", papel="quejoso"),
       "el singular, como siempre")

_seccion(38, _s38)


def _s39():
    print("\n39 · LA FRACCIÓN VI: JUBILACIÓN Y DERECHO SUBJETIVO RECONOCIDO (RF 4/2025 y 6/2026)")
    _t = ("solicitud de incorporación al sistema de jubilación previsto en el artículo Décimo Transitorio de la "
          "Ley del " + _ISS)
    ok("pensiones" in pf.supuesto_fraccion_vi(_t) and pf.supuesto_fraccion_vi(_t, ampliado=False) == "",
       "RF 4/2025: la jubilación del Décimo Transitorio es el supuesto de pensiones (sin la bandera, como antes)")
    _p, _av = pf.parrafo(_t, 2025, fraccion="VI")
    ok("*********" not in _p and "pensiones que otorga el Instituto" in _p,
       "…y la procedencia ya no sale con el supuesto en hueco")
    _s = ("declaró la nulidad de la resolución impugnada para efectos y reconoció el derecho subjetivo al "
          "incremento de la cuota pensionaria y al pago retroactivo de las diferencias")
    _p, _av = pf.parrafo("pensión del ISSSTE", 2025, fraccion="VI", sentido=_s)
    ok(any("RECONOCIÓ UN DERECHO SUBJETIVO" in a and "la nulidad es de fondo" in a
           and "artículo 52, fracción V, inciso a)" in a for a in _av)
       and not any("comprueba si la nulidad es de fondo o por vicios formales" in a for a in _av),
       "RF 6/2026: con el derecho subjetivo reconocido, el aviso dice que la nulidad es de fondo (52-V-a)")
    ok("de fondo" not in _p and "pensiones que otorga el Instituto" in _p,
       "…y el considerando queda como el engrose de RF 6/2026, que no lo razona")
    ok(pf.reconoce_derecho_subjetivo("nulidad_derecho")
       and not pf.reconoce_derecho_subjetivo("declaró la nulidad lisa y llana"),
       "la clave «nulidad_derecho» del catálogo (C) cuenta; la nulidad lisa, no")
    _p, _av = pf.parrafo("pensión del ISSSTE", 2025, fraccion="VI",
                         sentido="declaró la nulidad de la resolución impugnada para determinados efectos")
    ok("de fondo" not in _p and any("comprueba si la nulidad es de fondo" in a for a in _av),
       "sin derecho subjetivo, el aviso de siempre y nada se afirma")

_seccion(39, _s39)


# ═══ SEXTA RONDA (3-oct-2026, contrato C1-C6, dueño A1 · tipos_asunto) ═══════
# David respondió los pendientes y dio visto bueno para todos los usuarios. Cada
# sección FALLA con el código de la quinta ronda (la función o el texto no
# existían).
import json as _json

_LA81 = (_json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                      "normas_ley_de_amparo.json"), encoding="utf-8"))
         .get("articulos", {}).get("81", ""))


def _s40():
    print("\n40 · C1 · LA REVISIÓN SIN EXISTENCIA Y CON SU PROCEDENCIA (art. 81, fr. I, texto local)")
    _base = ("El presente recurso de revisión es procedente, de conformidad con el artículo 81, fracción I, "
             "inciso {i}), de la Ley de Amparo, en razón de que se impugna {que}.")
    ok(ta.procedencia_revision("sentencia") == _base.format(
        i="e", que="una sentencia dictada en la audiencia constitucional")
       and ta.procedencia_revision() == ta.procedencia_revision("sentencia"),
       "sentencia de audiencia (y sin clase): inciso e), con el texto exacto del contrato")
    ok(ta.procedencia_revision("interlocutoria_suspension") == _base.format(
        i="a", que="la interlocutoria que resolvió sobre la suspensión definitiva")
       and ta.procedencia_revision("interlocutoria_suspension", "a") == ta.procedencia_revision(
           "interlocutoria_suspension"),
       "interlocutoria de suspensión: inciso a)")
    _b = ta.procedencia_revision("interlocutoria_suspension", "b")
    # REVISIÓN DE NORMAS (3-oct-2026): el inciso b) cubre también la que NIEGA
    # modificar o revocar. Antes se comprobaba «modificó o revocó el acuerdo…».
    ok("inciso b), de la Ley de Amparo" in _b
       and "se pronunció sobre la modificación o revocación del acuerdo en que se concedió o negó la "
           "suspensión definitiva" in _b and "que modificó o revocó" not in _b,
       f"lo que se pronunció sobre modificar o revocar la suspensión: inciso b), con los dos supuestos: {_b}")
    ok(ta.procedencia_revision("auto_sobreseimiento") == _base.format(
        i="d", que="el auto que sobreseyó en el juicio fuera de la audiencia constitucional"),
       "auto de sobreseimiento fuera de audiencia: inciso d)")
    ok(ta.procedencia_revision("reposicion_constancias") == _base.format(
        i="c", que="la resolución que decidió el incidente de reposición de constancias de autos"),
       "incidente de reposición de constancias: inciso c)")
    ok("a) Las que concedan o nieguen la suspensión definitiva" in _LA81
       and "b) Las que modifiquen o revoquen el acuerdo en que se conceda o niegue la suspensión definitiva" in _LA81
       and "c) Las que decidan el incidente de reposición de constancias de autos" in _LA81
       and "d) Las que declaren el sobreseimiento fuera de la audiencia constitucional" in _LA81
       and "e) Las sentencias dictadas en la audiencia constitucional" in _LA81,
       "cada inciso cotejado con el artículo 81, fracción I, de normas_ley_de_amparo.json")
    _cons = [c for c, _n in ta.estructura_de("amparo_revision")["considerandos"]]
    ok(not any("existencia" in c.lower() for c in _cons)
       and _cons[:3] == ["Competencia", "Procedencia", "Legitimación y oportunidad"],
       f"el catálogo del AR: sin existencia; Competencia, Procedencia, Legitimación (5/5 engroses): {_cons[:4]}")
    bandera(False)
    try:
        _cons0 = [c for c, _n in ta.estructura_de("amparo_revision")["considerandos"]]
        ok(_cons0 == _cons, "[siempre] también sin la bandera")
    finally:
        bandera(True)

_seccion(40, _s40)


def _s41():
    print("\n41 · C2 · LA FRACCIÓN DEL 35 DE LA LOPJF, POR TIPO Y POR LO RECURRIDO")
    ok("35, fracción I, con el inciso de la materia (a penal; b administrativa, incluida la agraria; "
       "c civil, mercantil y familiar; d laboral)" in ta.cadena_competencia("amparo_directo"),
       "amparo directo: fracción I con el inciso de su materia")
    _ar = ta.cadena_competencia("amparo_revision")
    ok("35, fracción V" in _ar and "35, fracción II" in _ar and "c) la reposición de constancias" in _ar
       and "d) el auto que sobreseyó fuera de la audiencia" in _ar,
       "revisión: V para la sentencia de audiencia; II para la suspensión, la reposición y el sobreseimiento")
    ok("35, fracción III" in ta.cadena_competencia("queja")
       and "35, fracción VI" in ta.cadena_competencia("revision_fiscal")
       and "35, fracción VI, de la Ley Orgánica" in ta.COMPETENCIA["revision_fiscal"]["plantilla"],
       "queja: III; revisión fiscal: VI (cadena y plantilla propia)")
    # La fórmula de la mesa que escribe «35, fracción V, de la Ley Orgánica»
    # (con coma) también se corrige cuando lo recurrido es el auto de
    # sobreseimiento; antes sólo se cambiaba «35, fracción V y 210».
    _vieja = ("… de conformidad con los artículos 107, fracción VIII, último párrafo, de la Constitución, 81, "
              "fracción I, inciso e), y 84, de Ley de Amparo y 35, fracción V, de la Ley Orgánica del Poder "
              "Judicial de la Federación vigente; por haber sido interpuesto contra una sentencia dictada en un "
              "juicio de amparo indirecto por el Juzgado Primero de Distrito.")
    _n, _a = ta.competencia_revision(_vieja, "", clase="auto_sobreseimiento")
    ok("35, fracción II, de la Ley Orgánica" in _n and "inciso d), y 84" in _n and _a,
       f"auto de sobreseimiento con la fórmula antigua: 81-I-d) y 35-II: {_n[60:220]}")
    # La variante del banco para la norma impugnada (delegación de la SCJN)
    # escribía «35, fracción II»: lo remitido por la SCJN es la V.
    _del = ("… con fundamento en los artículos 107, fracción VIII, último párrafo, de la Constitución Federal; "
            "81, fracción I, inciso e) y 83, segundo párrafo de la Ley de Amparo; 35, fracción II y 210 de la "
            "Ley Orgánica del Poder Judicial de la Federación; punto cuarto, fracción I, inciso b), del Acuerdo "
            "General 2/2025 (12a.)…")
    _n, _a = ta.competencia_revision(_del, "VISTOS para resolver los autos del juicio de amparo; audiencia "
                                           "constitucional")
    ok("35, fracción V y 210 de la Ley Orgánica" in _n and "35, fracción II" not in _n
       and any("FRACCIÓN II DEL ARTÍCULO 35" in x for x in _a),
       "sentencia de audiencia con la variante de la delegación: 35-V, con aviso")
    _n, _a = ta.competencia_revision(_del.replace("inciso e)", "inciso a)"),
                                     "VISTOS para resolver el incidente de suspensión relativo al juicio de "
                                     "amparo 12/2026")
    ok("35, fracción II y 210" in _n and "35, fracción V" not in _n,
       "la suspensión se queda en la II")
    _n, _a = ta.competencia_revision(_vieja.replace("inciso e)", "inciso d)"), "")
    ok("35, fracción II, de la Ley Orgánica" in _n and _a,
       "sin proemio, la fórmula que cita el inciso d) del 81 lleva la fracción II del 35")
    _otra = "…artículo 135, fracción II, de la Ley Orgánica del Poder Judicial de la Federación…"
    ok(ta.competencia_revision(_otra, "", clase="sentencia")[0] == _otra,
       "un 135 no es el 35: no se toca")

_seccion(41, _s41)


def _s42():
    print("\n42 · C5 · EL PUNTO DEL AMPARO NOMBRA ACTO Y AUTORIDAD")
    _sala = "Segunda Sala Civil del Tribunal Superior de Justicia del Estado de Querétaro"
    _juz = "Juzgado Mixto de Primera Instancia del Distrito Judicial de Cadereyta"
    _p = ta.punto_del_amparo(["La sentencia de diez de marzo de dos mil veinticinco, dictada en el toca civil "
                              "123/2024."], [_sala, _juz])
    ok(_p == ("respecto del acto que reclamó de la Segunda Sala Civil del Tribunal Superior de Justicia del "
              "Estado de Querétaro y del Juzgado Mixto de Primera Instancia del Distrito Judicial de Cadereyta, "
              "consistente en la sentencia de diez de marzo de dos mil veinticinco, dictada en el toca civil "
              "123/2024"),
       f"un acto corto: «respecto del acto que reclamó de la… y del…, consistente en…» (sin punto final): {_p}")
    _largo = "La resolución " + " ".join(["dictada en el expediente"] * 10)
    _p = ta.punto_del_amparo([_largo], [_sala])
    ok(_p == ("respecto del acto que reclamó de la Segunda Sala Civil del Tribunal Superior de Justicia del "
              "Estado de Querétaro, precisado en el resultando primero de esta ejecutoria"),
       f"un acto de más de 30 palabras: remite al resultando: {_p}")
    _p = ta.punto_del_amparo(["La orden de clausura.", "1) La multa impuesta.", "El embargo"],
                             [_sala, _juz, "Director de Ingresos del Municipio de Querétaro"], "primero")
    ok(_p.startswith("respecto de los actos que reclamó de la Segunda Sala Civil")
       and ", del Juzgado Mixto" in _p and " y del Director de Ingresos" in _p
       and _p.endswith(", precisados en el resultando primero de esta ejecutoria"),
       f"tres actos: «los actos que reclamó de…, precisados en el resultando primero»: {_p}")
    _p = ta.punto_del_amparo(["La orden de clausura"], [], "segundo")
    ok(_p == "respecto del acto precisado en el resultando segundo de esta ejecutoria",
       f"sin autoridades: remite al resultando, con su ordinal: {_p}")
    ok(ta.punto_del_amparo([], [_sala]) == "" and ta.punto_del_amparo(["*********", "  ", "*****."], [_sala]) == ""
       and ta.punto_del_amparo(None, None) == "",
       "sin nada útil (vacío o sólo huecos): «»")
    _p = ta.punto_del_amparo(["La orden de clausura"], ["el Director de Ingresos"], plural_quejoso=True,
                             preposicion="contra")
    ok(_p == "contra el acto que reclamaron del Director de Ingresos, consistente en la orden de clausura",
       f"quejoso plural y «contra» (el sobreseimiento): «contra el» se queda, «de el» → «del»: {_p}")
    ok(ta.punto_del_amparo(["La publicación en El Universal"], ["el Juez Segundo"]).endswith(
        "consistente en la publicación en El Universal"),
       "la «El» de un nombre propio no se contrae ni se baja")
    _p = ta.punto_del_amparo(["LA SENTENCIA DE DIEZ DE MARZO"], ["SEGUNDA SALA CIVIL DEL TRIBUNAL SUPERIOR"])
    ok(_p == "respecto del acto que reclamó de la Segunda Sala Civil del Tribunal Superior, precisado en el "
             "resultando primero de esta ejecutoria",
       f"un acto en versales no se cita (se remite al resultando); la autoridad en versales pasa a prosa: {_p}")
    ok(ta.ACTOS_DEL_AMPARO_GENERICO == {
        "respecto de": "respecto de los actos precisados en la resolución recurrida",
        "contra": "contra los actos precisados en la resolución recurrida"},
       "el respaldo genérico, por preposición")
    _r = ta.RAMAS_REVISION
    ok(_r["confirma_niega"]["puntos"][1] == "SEGUNDO. La Justicia de la Unión no ampara ni protege a {quejoso}, "
       "{actos_del_amparo}, por las razones expuestas en el último considerando de la resolución recurrida."
       and _r["confirma_concede"]["puntos"][1] == "SEGUNDO. La Justicia de la Unión ampara y protege a {quejoso}, "
       "{actos_del_amparo}, en términos del último considerando de la resolución recurrida."
       and _r["confirma_sobresee"]["puntos"][1] == "SEGUNDO. Se sobresee en el presente juicio de amparo, "
       "promovido por {quejoso}, {actos_del_amparo}, por las razones expuestas en la resolución recurrida.",
       "las tres ramas que confirman llevan {actos_del_amparo} con el texto del contrato")
    ok(not any("{actos_del_amparo}" in p for k in ("confirma_sobresee_niega", "confirma_sobresee_concede",
                                                    "sin_determinar") for p in _r[k]["puntos"])
       and "respecto de los actos precisados en" in _r["sin_determinar"]["puntos"][1],
       "las mixtas y sin_determinar no cambian")
    # SIN ACTOS, LA COLA DE SIEMPRE (revisión AR, 3-oct-2026): antes se comprobaba
    # «…en la resolución recurrida, por las razones expuestas en la resolución
    # recurrida».
    ok(ta.con_actos_del_amparo(_r["confirma_sobresee"]["puntos"][1]).endswith(
        "{quejoso}, contra los actos precisados en la resolución recurrida y por las razones expuestas en la "
        "misma.")
       and ta.con_actos_del_amparo(_r["confirma_niega"]["puntos"][1], ["La orden"], ["el Juez Segundo"])
       == ("SEGUNDO. La Justicia de la Unión no ampara ni protege a {quejoso}, respecto del acto que reclamó del "
           "Juez Segundo, consistente en la orden, por las razones expuestas en el último considerando de la "
           "resolución recurrida."),
       "con_actos_del_amparo: el genérico con «contra» en el sobreseimiento; con ficha, el acto y la autoridad")
    ok(ta.puntos_confirma_concede("")[1] == _r["confirma_concede"]["puntos"][1],
       "sin el resolutivo del juzgado, el punto genérico lleva el marcador para quien lo llene")

_seccion(42, _s42)


def _s43():
    print("\n43 · C6 · LOS ASUNTOS RELACIONADOS, SÓLO SI EL SECRETARIO LOS MARCA")
    _ad = {"tipo": "amparo_directo", "numero": "452/2025", "estado": "misma_sesion"}
    _rf = {"tipo": "revision_fiscal", "numero": "33/2024", "estado": "misma_sesion"}
    _ar = {"tipo": "amparo_revision", "numero": "298/2025", "estado": "resuelto"}
    # INTEGRACIÓN (3-oct-2026): la materia concuerda IGUAL que el encabezado de
    # ese tipo; antes se comprobaba «amparo en revisión administrativo 12/2026».
    ok(ta.relacionado_en_prosa(_ad, "civil") == "amparo directo civil 452/2025"
       and ta.relacionado_en_prosa({"tipo": "amparo_revision", "numero": "12/2026"}, "administrativa")
       == "amparo en revisión administrativa 12/2026"
       and ta.relacionado_en_prosa({"tipo": "amparo_directo", "numero": "12/2026"}, "administrativa")
       == "amparo directo administrativo 12/2026"
       and ta.relacionado_en_prosa({"tipo": "queja", "numero": "24/2026"}, "administrativa")
       == "recurso de queja administrativa 24/2026"
       and ta.relacionado_en_prosa({"tipo": "queja", "numero": "24/2026"}, "civil") == "recurso de queja civil 24/2026"
       and ta.relacionado_en_prosa(_rf, "administrativa") == "recurso de revisión fiscal 33/2024"
       and ta.relacionado_en_prosa(_ad, "") == "amparo directo 452/2025",
       "en prosa: el amparo directo en masculino, la revisión y la queja en femenino, la revisión fiscal "
       "sin materia, sin materia sin palabra")
    # EL RENGLÓN DICE LO QUE DIRÍA EL ENCABEZADO DE ESE ASUNTO (integración, 3-oct-2026).
    _enc_igual = []
    for _t in ("amparo_directo", "amparo_revision", "queja"):
        for _m in ("administrativa", "administrativo", "civil", "familiar", "agraria", "penal", "laboral"):
            _rot = ta.rotulo_relacionados([{"tipo": _t, "numero": "12/2026"}], _m)
            _enc = ta.encabezado_de(_t, _m, "12/2026").replace(":", "")
            if _rot != f"RELACIONADO CON EL {_enc}.":
                _enc_igual.append((_t, _m, _rot, _enc))
    ok(not _enc_igual, "«RELACIONADO CON EL …» = el encabezado de ese tipo con esa materia, en los tres "
                       "tipos y siete materias" + (f" — {_enc_igual[:3]}" if _enc_igual else ""))
    ok(ta.relacionados_en_prosa([_ad], "civil") == "el amparo directo civil 452/2025"
       and ta.relacionados_en_prosa([_ad, _rf], "civil")
       == "el amparo directo civil 452/2025 y con el recurso de revisión fiscal 33/2024",
       "lo que va tras «relacionado con» en el V I S T O")
    ok(ta.rotulo_relacionados([_ad], "civil") == "RELACIONADO CON EL AMPARO DIRECTO CIVIL 452/2025."
       and ta.rotulo_relacionados([_ad, _rf], "civil")
       == "RELACIONADO CON EL AMPARO DIRECTO CIVIL 452/2025 Y CON EL RECURSO DE REVISIÓN FISCAL 33/2024."
       and ta.rotulo_relacionados([], "civil") == "",
       "el renglón del rubro, con uno y con dos; vacío sin lista")
    _hn = ta.supletorio("Tercer Tribunal Colegiado en Materias Administrativa y Civil del Vigésimo Segundo "
                        "Circuito", "Querétaro, Querétaro")["hecho_notorio"]
    _r, _t = ta.considerando_relacionados("amparo_directo", "469/2024", "administrativa", [_rf], _hn)
    ok(_r == "Conexidad." and _t == (
        "Con vista en la conexión que guarda el presente juicio de amparo directo administrativo 469/2024 con el "
        "recurso de revisión fiscal 33/2024, ambos del índice de este Tribunal Colegiado de Circuito, se resuelven "
        "en la misma sesión, con fundamento en el artículo 64 de la Ley Federal de Procedimiento Contencioso "
        "Administrativo, porque en ambos se impugna la misma sentencia."),
       "AD 469/2024 con la RF 33/2024 en la misma sesión: el artículo 64 de la LFPCA")
    _r, _t = ta.considerando_relacionados("revision_fiscal", "33/2024", "administrativa",
                                          [{"tipo": "amparo_directo", "numero": "469/2024"}], _hn)
    ok(_r == "Conexidad." and "artículo 64" in _t and "presente recurso de revisión fiscal 33/2024 con el amparo "
       "directo administrativo 469/2024" in _t,
       "y al revés (la RF con su amparo directo; estado ausente = misma sesión)")
    _r, _t = ta.considerando_relacionados("amparo_directo", "456/2025", "civil", [_ad], _hn)
    ok(_r == "Conexidad." and "artículo 64" not in _t and _t.endswith(
        "ambos del índice de este Tribunal Colegiado de Circuito, se resuelven en la misma sesión, a fin de evitar "
        "el dictado de resoluciones contradictorias."),
       "dos amparos directos en la misma sesión: la fórmula general, sin el 64")
    _r, _t = ta.considerando_relacionados("queja", "24/2026", "civil", [_ar], _hn)
    ok(_r == "Hecho notorio." and _t == (
        "Se invoca como hecho notorio, en términos del artículo 88 del Código Federal de Procedimientos Civiles, "
        "de aplicación supletoria a la Ley de Amparo, la ejecutoria dictada por este Tribunal Colegiado de "
        "Circuito en el amparo en revisión civil 298/2025, relacionado con el presente asunto."),
       "Q 24/2026 con el AR 298/2025 ya resuelto: hecho notorio, CFPC 88 fuera de la Ciudad de México")
    _hn_cdmx = ta.supletorio("Décimo Tribunal Colegiado en Materia Civil del Primer Circuito",
                             "Ciudad de México")["hecho_notorio"]
    ok("artículo 269 del Código Nacional" in ta.considerando_relacionados("queja", "24/2026", "civil", [_ar],
                                                                          _hn_cdmx)[1],
       "en la Ciudad de México, el 269 del CNPCF (el que da `supletorio`)")
    _r, _t = ta.considerando_relacionados(
        "amparo_directo", "469/2024", "administrativa",
        [_rf, {"tipo": "amparo_directo", "numero": "470/2024"}, _ar], _hn)
    _ps = _t.split("\n")
    ok(_r == "Asuntos relacionados." and len(_ps) == 2
       and "artículo 64" in _ps[0] and "con el recurso de revisión fiscal 33/2024, ambos" in _ps[0]
       and "con el amparo directo administrativo 470/2024, también del índice" in _ps[0]
       and _ps[0].count("artículo 64") == 1
       and _ps[0].split("Asimismo")[1].find("artículo 64") < 0
       and _ps[1].startswith("Se invoca como hecho notorio"),
       "mezcla: el 64 sólo para el par AD↔RF, el otro AD con la fórmula general, y el resuelto aparte")
    _r, _t = ta.considerando_relacionados(
        "amparo_revision", "298/2025", "civil",
        [{"tipo": "amparo_revision", "numero": "299/2025"}, {"tipo": "queja", "numero": "24/2026"}], _hn)
    ok("con el amparo en revisión civil 299/2025 y con el recurso de queja civil 24/2026, todos del índice" in _t
       and "artículo 64" not in _t,
       "varios en la misma sesión: «con el X y con el Y, todos del índice…»")
    ok(ta.considerando_relacionados("amparo_directo", "469/2024", "civil", [], _hn) == ("", "")
       and ta.considerando_relacionados("amparo_directo", "469/2024", "civil",
                                        [{"tipo": "amparo_directo", "numero": "469/2024"},
                                         {"tipo": "otro", "numero": "1/2025"},
                                         {"tipo": "queja", "numero": "abc"}], _hn) == ("", ""),
       "sin lista, el propio asunto, un tipo desconocido o un número inválido: nada (nunca conexidad automática)")
    ok(len(ta.relacionados_validos([{"tipo": "queja", "numero": f"{i}/2026"} for i in range(1, 8)])) == 4
       and ta.relacionados_validos([_ad, dict(_ad)]) == [_ad],
       "hasta 4 y sin duplicados")

_seccion(43, _s43)


def _s44():
    print("\n44 · REVISIÓN ADVERSARIAL (3-oct-2026): EL PUNTO DEL AMPARO, LA MAYÚSCULA DEL ÓRGANO Y LA RF EN PLURAL")
    # AR 72/2025 (fichas4): la demanda reclama el artículo 90 de la Ley de
    # Hacienda; la ficha sólo trae el acto de aplicación. El punto se lo atribuía
    # a la Legislatura y al Gobernador como si fuera todo lo reclamado.
    _a72 = ["el pago de los derechos registrales correspondientes a la inscripción de una transmisión de "
            "propiedad en ejecución de fideicomiso y extinción parcial de éste sobre un inmueble"]
    _l72 = ["Legislatura y Gobernador, ambos del Estado de Querétaro"]
    ok(ta.norma_sin_acto(_a72, _l72) and ta.punto_del_amparo(_a72, _l72) == ""
       and not ta.norma_sin_acto(["el artículo 90 de la Ley de Hacienda del Estado de Querétaro"] + _a72, _l72)
       and not ta.norma_sin_acto(_a72, ["Director de Ingresos"])
       and not ta.norma_sin_acto(["La aprobación, expedición y promulgación de la Ley de la Agencia de Movilidad y "
                                  "Modalidades de Transporte Público para el Estado de Querétaro, en específico su "
                                  "artículo 210", "La retención del vehículo"], ["Legislatura del Estado de Querétaro"]),
       "AR 72: el Legislativo responsable y ningún acto que sea norma → el punto no cita «consistente en» "
       "(la remisión genérica; el aviso lo da documento_generado); la «aprobación, expedición y promulgación de la "
       "Ley…» del AR 201 sí es la norma")
    _pc = ta.con_actos_del_amparo(ta.RAMAS_REVISION["confirma_concede"]["puntos"][1], _a72, _l72,
                                  rama="confirma_concede")
    ok(_pc == "SEGUNDO. La Justicia de la Unión ampara y protege a {quejoso}, en términos del último considerando "
              "de la resolución recurrida." and "consistente en el pago" not in _pc,
       f"y en la rama que concede, el punto de antes de C5, vago pero cierto: {_pc}")
    # AR 222/2025: «actuario de su adscripción» salía «el Actuario de Su Adscripción».
    _p222 = ta.punto_del_amparo(["la sentencia de diez de mayo de dos mil veinticuatro"],
                                ["Juzgado Sexto de Primera Instancia Civil del Distrito Judicial de Querétaro",
                                 "actuario de su adscripción"])
    ok("y del Actuario de su adscripción, consistente en" in _p222 and "Su Adscripción" not in _p222
       and ta.con_articulo_de_organo("Actuario adscrito") == "el Actuario adscrito"
       and ta.con_articulo_de_organo("secretario ejecutor adscrito a la dirección de ingresos")
       == "el Secretario Ejecutor adscrito a la Dirección de Ingresos",
       f"«su adscripción» y «adscrito» en minúscula; «a la» sin «A»: {_p222[:160]}")
    # AR 60/2025: «dictada por la autoridad responsable» tras nombrar dos autoridades.
    _p60 = ta.punto_del_amparo(["La resolución de cinco de julio de dos mil veinticuatro, dictada por la autoridad "
                                "responsable en el toca civil 12/2024."],
                               ["Segunda Sala Civil del Tribunal Superior de Justicia del Estado de Querétaro",
                                "Juzgado Mixto de Primera Instancia del Distrito Judicial de Cadereyta de Montes, "
                                "Querétaro"])
    ok("consistente en la resolución de cinco de julio de dos mil veinticuatro, dictada en el toca civil 12/2024"
       in _p60 and "autoridad responsable" not in _p60,
       "el acto citado pierde «por la autoridad responsable»: el punto ya nombra a cada una")
    # EL ACTO SIN ARTÍCULO: el artículo que le toca; el nombre de la norma, con su mayúscula.
    _L, _G = ["Legislatura del Estado de Querétaro", "Gobernador del Estado de Querétaro"], ["Juez Primero"]
    ok(ta.punto_del_amparo(["Artículo 90 de la Ley de Hacienda del Estado de Querétaro"], _L).endswith(
           "consistente en el artículo 90 de la Ley de Hacienda del Estado de Querétaro")
       and ta.punto_del_amparo(["Ley de Hacienda del Estado de Querétaro, artículo 90"], _L).endswith(
           "consistente en la Ley de Hacienda del Estado de Querétaro, artículo 90")
       and ta.punto_del_amparo(["Código Fiscal del Estado de Querétaro, artículo 5"], _L).endswith(
           "consistente en el Código Fiscal del Estado de Querétaro, artículo 5")
       and ta.punto_del_amparo(["Sentencia de diez de mayo de dos mil veinticuatro"], _G).endswith(
           "consistente en la sentencia de diez de mayo de dos mil veinticuatro")
       and ta.punto_del_amparo(["Acuerdo de diez de mayo de dos mil veinticuatro"], _G).endswith(
           "consistente en el acuerdo de diez de mayo de dos mil veinticuatro")
       and ta.punto_del_amparo(["Ordenar el desalojo"], _G).endswith("precisado en el resultando primero de "
                                                                     "esta ejecutoria"),
       "«consistente en el artículo 90…», «la Ley de Hacienda…», «el Código Fiscal…», «la sentencia…»; sin saber "
       "el artículo, la remisión al resultando")
    # LA REMISIÓN GENÉRICA, CON LA COLA DE ANTES (la tarjeta y el camino sin actos).
    _r = ta.RAMAS_REVISION
    ok(ta.con_actos_del_amparo(_r["confirma_niega"]["puntos"][1]).endswith(
           "respecto de los actos precisados en la resolución recurrida, por las razones expuestas en el último "
           "considerando de la misma.")
       and ta.con_actos_del_amparo(_r["confirma_concede"]["puntos"][1]) == (
           "SEGUNDO. La Justicia de la Unión ampara y protege a {quejoso}, en términos del último considerando de "
           "la resolución recurrida.")
       and all(ta.con_actos_del_amparo(_r[k]["puntos"][1]).count("resolución recurrida") == 1
               for k in ("confirma_niega", "confirma_concede", "confirma_sobresee")),
       "sin actos, una sola «resolución recurrida» por punto (y la que concede, sin remisión a los actos)")
    import tarjeta_decision as _td
    _pt, _ = _td.desenlace_de("amparo_revision", "niega", "infundado")
    ok(any("por las razones expuestas en el último considerando de la misma." in x for x in _pt)
       and not any("considerando de la resolución recurrida" in x and "precisados en la resolución recurrida" in x
                   for x in _pt),
       f"la tarjeta de decisión, igual: {[x for x in _pt if 'SEGUNDO' in x]}")
    # LA REVISIÓN FISCAL DE VARIAS AUTORIDADES: la legitimación en plural.
    _sat = ("Secretario de Hacienda y Crédito Público, Jefe del Servicio de Administración Tributaria y Titular de "
            "la Administración de Operación de Padrones")
    _lg = ta.legitimacion_de("revision_fiscal", _sat, figura="Administradora Desconcentrada Jurídica de Querétaro 1",
                             autoridad_demandada=_sat)
    ok("pues lo hicieron valer el Secretario de Hacienda" in _lg
       and "autoridades demandadas en el juicio de nulidad, a las que la sentencia recurrida les resultó adversa" in _lg
       and "a la que" not in _lg,
       f"varias autoridades demandadas: «lo hicieron valer…, autoridades demandadas…, a las que… les resultó adversa»")
    _lg1 = ta.legitimacion_de("revision_fiscal", "Titular de la Unidad Jurídica del Instituto de Ejemplo",
                              autoridad_demandada="Director General del Instituto de Ejemplo")
    ok("autoridades demandadas" not in _lg1 and "lo hicieron valer" not in _lg1,
       "una sola autoridad demandada: en singular, como antes")
    ok(ta.varias_autoridades(_sat + ", por conducto de la Administradora Desconcentrada Jurídica")
       and not ta.varias_autoridades("Titular de la Unidad Jurídica de la Oficina de Representación en Querétaro del "
                                     "Instituto de Seguridad y Servicios Sociales de los Trabajadores del Estado")
       and not ta.varias_autoridades("Titular de la Unidad Jurídica, en representación del Director General y del "
                                     "Subdirector"),
       "varias_autoridades cuenta los cargos antes de «por conducto de» o «en representación de»")


_seccion(44, _s44)

print()
if FALLAS:
    print(f"FALLAN {len(FALLAS)}: " + " · ".join(FALLAS))
    sys.exit(1)
print("RESULTADO: TODAS LAS COMPROBACIONES PASAN")
