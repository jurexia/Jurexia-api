# -*- coding: utf-8 -*-
"""LA FICHA DE TRÁMITE (3-oct-2026), sin red.

    .venv/bin/python test_ficha_tramite.py

Los datos primero, la prosa después: cada hecho procesal con su fuente
(secretario > auto > acto > ficha procesal), lo que dice el modelo anclado al
papel, el sentido por catálogo y la cronología que no puede ser. Los autos son
sintéticos pero con la forma de los reales («Conste. Querétaro, Querétaro, a…»,
«fórmese el expediente número…», «túrnense los autos al Magistrado…»); el
modelo se simula con un cliente falso que cuenta sus llamadas.
"""
import asyncio
import json
import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import contexto_taller as ct
import fase_admision as fa
import ficha_tramite as ft

FALLOS = []


def ok(c, q):
    print(f"   {'PASA ' if c else 'FALLA'}  {q}")
    if not c:
        FALLOS.append(q)


# ── material ──────────────────────────────────────────────────────────────
AUTO_ADMISION_AD = """TERCER TRIBUNAL COLEGIADO EN MATERIAS ADMINISTRATIVA Y CIVIL DEL VIGÉSIMO SEGUNDO CIRCUITO
A.D.C. 43/2025
En diecisiete de enero de dos mil veinticinco, el Secretario de Acuerdos da cuenta al Magistrado
Presidente con el oficio 245/2025 y sus anexos. Conste.
Querétaro, Querétaro, a veinte de enero de dos mil veinticinco.
Visto el oficio 245/2025, por el cual la Juez Primero de Distrito en Materia de Amparo Civil,
Administrativo y de Trabajo y de Juicios Federales en el Estado de Querétaro remite la demanda de
amparo directo promovida por Banco Mercantil del Norte, Sociedad Anónima, Institución de Banca
Múltiple, Grupo Financiero Banorte, por conducto de su apoderada legal María Fernanda Ruiz Olvera,
contra la sentencia de tres de julio de dos mil veinticuatro, dictada por la Juez Primero de Distrito
en Materia de Amparo Civil, Administrativo y de Trabajo y de Juicios Federales en el Estado de
Querétaro, en el juicio oral mercantil 83/2024; con fundamento en los artículos 179 y 181 de la Ley
de Amparo, regístrese y fórmese el expediente número 43/2025.
Se admite la demanda de amparo. Se tiene como tercero interesado a José Antonio Juárez Lira.
Notifíquese a las partes para que en el plazo de quince días presenten sus alegatos o promuevan
amparo adhesivo. Dese la intervención que corresponde al agente del Ministerio Público de la
Federación adscrito para que, si lo estima pertinente, formule pedimento.
Notifíquese.
Así lo proveyó y firma el Magistrado Presidente Ismael Camacho Herrera, ante el Secretario de
Acuerdos Omar Alejandro Elizalde Herrera, que autoriza y da fe."""

AUTO_TURNO_AD = """A.D.C. 43/2025
Querétaro, Querétaro, a veintiuno de febrero de dos mil veinticinco.
Visto el estado procesal que guardan los autos, y toda vez que el agente del Ministerio Público de
la Federación adscrito no formuló pedimento dentro del plazo concedido, con fundamento en el
artículo 183 de la Ley de Amparo, TÚRNENSE los autos al Magistrado J. Guadalupe Tafoya Hernández,
para la formulación del proyecto de resolución correspondiente.
Notifíquese."""

AUTO_RETURNO = """A.D.C. 43/2025
Querétaro, Querétaro, a veinticinco de noviembre de dos mil veinticinco.
Visto el oficio SEADS/3771/2025 de la Secretaría Ejecutiva de la Comisión de Adscripción del Órgano
de Administración Judicial, retúrnense los autos a la ponencia del Magistrado Luis Armando Pérez
Topete, a la que previamente se habían turnado, para el dictado de la resolución correspondiente.
Notifíquese."""

AUTO_ADHESIVO = """A.D.C. 722/2025
Querétaro, Querétaro, a diez de febrero de dos mil veintiséis.
Agréguese a los autos el escrito presentado el seis de febrero de dos mil veintiséis por Mariana
Paulina Nieves Ortega, tercera interesada; con fundamento en el artículo 182 de la Ley de Amparo, se
admite el amparo adhesivo promovido por Mariana Paulina Nieves Ortega, en contra de la sentencia
reclamada. Notifíquese."""

AUTO_AR = """AMPARO EN REVISIÓN 631/2025
Querétaro, Querétaro, a doce de agosto de dos mil veinticinco.
Visto el oficio por el que el Juzgado Séptimo de Distrito en Materia de Amparo Civil, Administrativo
y de Trabajo y de Juicios Federales en el Estado de Querétaro remite el escrito de agravios y los
autos originales; fórmese el expediente número 631/2025. Se tiene a Impulsora de Desarrollos
Inmobiliarios VV, Sociedad Anónima de Capital Variable, tercera interesada, interponiendo recurso de
revisión contra la sentencia de veintidós de abril de dos mil veinticinco, dictada por el Juzgado
Séptimo de Distrito en Materia de Amparo Civil, Administrativo y de Trabajo y de Juicios Federales en
el Estado de Querétaro, en el juicio de amparo indirecto 950/2024; recurso que se admite. Se tiene
al agente del Ministerio Público de la Federación adscrito formulando pedimento.
Notifíquese."""

AUTO_Q_FR2 = """RECURSO DE QUEJA 143/2026
Querétaro, Querétaro, a tres de marzo de dos mil veintiséis.
Fórmese el expediente número 143/2026 con el escrito por el que Fraccionadora La Romita, Sociedad
Anónima de Capital Variable, interpone recurso de queja contra el auto de veinte de febrero de dos
mil veintiséis, dictado por la Primera Sala Civil del Tribunal Superior de Justicia del Estado de
Querétaro, en el amparo directo 127/2026. Con fundamento en el artículo 101 de la Ley de Amparo,
requiérase a la autoridad responsable para que rinda su informe con justificación.
Notifíquese.
Querétaro, Querétaro, a nueve de marzo de dos mil veintiséis.
Se tiene a la Primera Sala Civil del Tribunal Superior de Justicia del Estado de Querétaro rindiendo
su informe con justificación; se admite el recurso de queja. Notifíquese."""

AUTO_RF = """REVISIÓN FISCAL 91/2025
Querétaro, Querétaro, a veintiuno de noviembre de dos mil veinticinco.
Visto el oficio de expresión de agravios presentado por Karen Yadira Meza Ruiz, Administradora
Desconcentrada Jurídica de Querétaro "1", del Servicio de Administración Tributaria, depositado en el
Servicio Postal Mexicano, contra la sentencia de veintidós de septiembre de dos mil veinticinco,
dictada por la Sala Regional del Centro II del Tribunal Federal de Justicia Administrativa, en el
juicio contencioso administrativo 695/25-09-01-7-OT; fórmese el expediente número 91/2025 y se
admite el recurso de revisión fiscal.
Fecha de presentación: 13/11/2025
Notifíquese."""

ACTO_AD = """Sentencia Segunda Sala Civil Juicio ordinario civil Toca familiar 4520/2024
Santiago de Querétaro, Querétaro, a 20 veinte de febrero de 2025 dos mil veinticinco.
Vistos para resolver los autos del toca familiar 4520/2024, formado al recurso de apelación
interpuesto por la parte actora contra la sentencia de 10 diez de septiembre de 2024 dos mil
veinticuatro, dictada en el expediente 479/2024, por la Jueza Tercero de Primera Instancia Familiar
del Distrito Judicial de Querétaro, en el juicio ordinario civil que sobre divorcio inició Ma. del
Refugio Trejo Ramos contra Gabriel Reyes Álamo.
R E S U E L V E: PRIMERO. Se confirma la sentencia apelada. Notifíquese.
Así lo resolvieron los Magistrados integrantes de la Segunda Sala Civil del Tribunal Superior de
Justicia del Estado de Querétaro. Toca familiar 4520/2024."""

DEMANDA_AD = """C.C. MAGISTRADOS DEL TRIBUNAL COLEGIADO EN TURNO. Ma. del Refugio Trejo Ramos, por propio
derecho, promuevo juicio de amparo directo con fundamento en los artículos 103 y 107 de la
Constitución Política de los Estados Unidos Mexicanos.
3.- AUTORIDAD RESPONSABLE. - Se señala como autoridad responsable, Segunda Sala Civil del Tribunal
Superior de Justicia del Estado de Querétaro.. 4.- ACTO RECLAMADO.- Reclamo la resolución definitiva
dictada en el toca de apelación número 4520/2024 de fecha 20 de Febrero de 2025.
5.- LA FECHA EN QUE SE HAYA NOTIFICADO EL ACTO RECLAMADO: fue el día 21 de Febrero de 2025.
6.- PRECEPTOS QUE CONTENGAN LOS DERECHOS HUMANOS CUYA VIOLACION SE RECLAMEN. - Artículos 14, 16 y
17 Constitucionales, en relación con el artículo 8 de la Convención Americana sobre Derechos
Humanos. 7.- CONCEPTOS DE VIOLACIÓN. PRIMERO. La responsable viola el artículo 1o. constitucional."""

ACTO_Q = """JUZGADO TERCERO DE DISTRITO EN MATERIA DE AMPARO CIVIL, ADMINISTRATIVO Y DE TRABAJO Y DE JUICIOS
FEDERALES EN EL ESTADO DE QUERÉTARO
JUICIO DE AMPARO INDIRECTO 1234/2025
Querétaro, Querétaro, a seis de marzo de dos mil veintiséis.
Visto el escrito de demanda presentado por Comercializadora del Bajío, Sociedad Anónima de Capital
Variable, contra actos del Director de Ingresos del Municipio de Querétaro. Del análisis de la
demanda se advierte que se actualiza la causa de improcedencia prevista en el artículo 61, fracción
XXIII, de la Ley de Amparo. En consecuencia, con fundamento en el artículo 113 de la Ley de Amparo,
se desecha de plano la demanda de amparo. Notifíquese.
Así lo proveyó y firma la Jueza Tercero de Distrito, ante la Secretaria que autoriza y da fe."""

ACTO_RF = """SALA REGIONAL DEL CENTRO II DEL TRIBUNAL FEDERAL DE JUSTICIA ADMINISTRATIVA
EXPEDIENTE: 695/25-09-01-7-OT
ACTORA: Transportes Ejemplo del Centro, S.A. de C.V.
Querétaro, Querétaro, a veintidós de septiembre de dos mil veinticinco.
VISTOS los autos del juicio contencioso administrativo número 695/25-09-01-7-OT, promovido por
Transportes Ejemplo del Centro, S.A. de C.V., en contra de la resolución contenida en el oficio
500-62-00-03-01-2025-1234, de quince de enero de dos mil veinticinco, emitida por la Administración
Desconcentrada de Auditoría Fiscal de Querétaro "1", se resuelve conforme a lo siguiente.
Por lo expuesto, se R E S U E L V E: I. La parte actora probó su pretensión; II. Se declara la
nulidad lisa y llana de la resolución impugnada. NOTIFÍQUESE.
EXPEDIENTE: 695/25-09-01-7-OT"""


class _Comp:
    def __init__(self, respuestas):
        self.respuestas = list(respuestas)
        self.vistos = []

    async def create(self, **kw):
        self.vistos.append(dict(kw))
        r = self.respuestas.pop(0) if self.respuestas else {}
        msg = types.SimpleNamespace(content=json.dumps(r, ensure_ascii=False))
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg, finish_reason="stop")],
                                     usage=None)


def cliente_falso(*respuestas):
    comp = _Comp(respuestas)
    return types.SimpleNamespace(chat=types.SimpleNamespace(completions=comp)), comp


ct._CTX.set(None)

print("\n1 · LAS FECHAS COMO SE ESCRIBEN DE VERDAD")
f = [x[0] for x in ft.fechas_del_texto("Santiago de Querétaro, Querétaro, a 20 veinte de febrero de 2025 dos mil "
                                        "veinticinco; 19 diecinueve agosto de 2024; el 17 de enero del 2021; "
                                        "Fecha de presentación: 13/11/2025; 3 (tres) de diciembre de 2025 (dos mil "
                                        "veinticinco); veintidós de abril de dos mil veinticinco")]
ok(f == ["2025-02-20", "2024-08-19", "2021-01-17", "2025-11-13", "2025-12-03", "2025-04-22"],
   f"dígito+letra, sin «de», «del 2021», dd/mm/aaaa, paréntesis y letra pura: {f}")
ok(ft.a_iso("treinta y uno de diciembre de dos mil veinticinco y su ejecución") == "2025-12-31",
   "«dos mil veinticinco y su ejecución»: la cola no es del año")
ok(ft.a_iso("veintitrés de agosto de dos mil") == "", "«dos mil» a secas no se adivina como el año 2000")
ok(ft.a_iso("Santiago de Querétaro, Qro., vertinueve de agosto de dos mil veinticuatro") == "2024-08-29",
   "la errata del OCR en el día, sólo si es inequívoca («vertinueve» = veintinueve)")
ok(ft.a_iso("31/02/2025") == "" and ft.a_iso("2025-02-30") == "", "una fecha que no existe no es fecha")
ok(ft.fecha_en_papel("2025-02-20", ACTO_AD) and not ft.fecha_en_papel("2025-02-21", ACTO_AD),
   "fecha_en_papel: la del proemio está, la de la notificación no (está en la demanda)")
ok(ft.fecha_en_papel("2025-02-21", DEMANDA_AD), "…y en la demanda sí")
ok(ft._fecha_proemio(ACTO_AD) == "2025-02-20", "la fecha de la sentencia es la de su proemio")
ok(ft._fecha_proemio("Juzgado Segundo Administrativo en Querétaro, el 3 (tres) de febrero de 2022") == "",
   "«…en Querétaro, el 3 de febrero» no es un proemio (ADA 702/2022)")
ok(ft._fecha_proemio("Fecha de presentación/Fecha de depósito: 20/09/2024 Hora") == "",
   "la boleta de la Oficina de Correspondencia no es el proemio (ADC 590/2024)")

print("\n2 · EL ANCLAJE AL PAPEL")
v, m = ft.anclar("segunda sala civil del tribunal superior de justicia del estado de QUERETARO", DEMANDA_AD)
ok(m == "literal" and v, "literal salvo mayúsculas y tildes: se acepta")
v, m = ft.anclar("Segunda Sala Civil del Tribunal Superior de Justicia del Estado de Queretaro, Qro", DEMANDA_AD)
ok(m == "parecido" and v.startswith("Segunda Sala Civil") and "Qro" not in v,
   f"parecido (≥ 0.85): se usa EL DEL PAPEL: «{v}»")
v, m = ft.anclar("Primera Sala Penal del Tribunal Superior de Justicia de Guanajuato", DEMANDA_AD)
ok(m == "" and v == "", "lo que no está se tira")

print("\n3 · LA FORMA DE ÓRGANO")
ok(not ft.es_organo("TRIBUNAL EL SEIS")[0], "«TRIBUNAL EL SEIS» (AD 103) no es un órgano")
ok(not ft.es_organo("PRIMERA SALA CIVIL JUICIO")[0], "«PRIMERA SALA CIVIL JUICIO» (AD 642) está cortado")
ok(ft.es_organo("Sala Familiar del Tribunal Superior de Justicia en el Estado") == (True, "sin_entidad"),
   "«…en el Estado» sin el estado: órgano, pero se avisa")
ok(ft.es_organo("Tribunal Unitario Agrario del Distrito 42, con sede en Querétaro") == (True, ""),
   "«Distrito 42» sí lleva número")
ok(not ft.es_organo("Banco Mercantil del Norte, Sociedad Anónima")[0], "un banco no es un órgano")

print("\n4 · LA SEDE")
s = ft.sede_de("Tercer Tribunal Colegiado en Materias Administrativa y Civil del Vigésimo Segundo Circuito",
               "Querétaro, Querétaro")
ok(s["circuito"] == "XXII" and not s["cdmx"], f"XXII, fuera de CDMX: {s}")
ok(ft.sede_de("Primer Tribunal Colegiado en Materia Civil del 22o. Circuito")["circuito"] == "XXII", "«22o. Circuito»")
ok(ft.sede_de("Segundo Tribunal Colegiado del XXII Circuito")["circuito"] == "XXII", "«XXII Circuito»")
ok(ft.sede_de("Tribunal Colegiado en Materia Penal del Décimo Primer Circuito")["circuito"] == "XI", "«Décimo Primer»")
s = ft.sede_de("Quinto Tribunal Colegiado en Materia Civil del Primer Circuito", "")
ok(s["circuito"] == "I" and s["cdmx"], "Primer Circuito: Ciudad de México")
ok(ft.sede_de("", "Ciudad de México")["cdmx"] and ft.sede_de("", "México, D.F.")["cdmx"], "la ciudad lo dice")
s = ft.sede_de("Primer Tribunal Colegiado de Circuito del Centro Auxiliar de la Primera Región", "Ciudad de México")
ok(s["circuito"] == "" and s["cdmx"], "Centro Auxiliar: sin circuito; CDMX por su residencia")
s = ft.sede_de("Primer Tribunal Colegiado de Circuito del Centro Auxiliar de la Tercera Región", "Guanajuato, Guanajuato")
ok(s["circuito"] == "" and not s["cdmx"], "Centro Auxiliar fuera de CDMX")

print("\n5 · LOS AUTOS, SIN MODELO")
a = ft.leer_auto(AUTO_ADMISION_AD)
ok(a.get("numero") == "43/2025" and (a.get("admision") or {}).get("fecha") == "2025-01-20",
   f"admisión: número y fecha del auto (la que sigue a «Conste.», no la razón de cuenta): {a.get('admision')}")
ok(not a.get("ministerio_publico"), "«dese la intervención… para que formule pedimento» NO dice nada del pedimento")
ok(a.get("representante") == "María Fernanda Ruiz Olvera" and a.get("figura_representante") == "apoderada legal",
   "por conducto de su apoderada legal")
ok((a.get("acto") or {}).get("fecha") == "2024-07-03" and (a.get("acto") or {}).get("expediente") == "83/2024"
   and "Juicios Federales en el Estado de Querétaro" in (a.get("acto") or {}).get("organo", ""),
   f"lo reclamado según el auto, con el órgano ENTERO: {a.get('acto')}")
ok("Vigésimo Segundo Circuito" in (a.get("sede") or {}).get("tribunal", ""), "el tribunal, de la cabecera")
t = ft.leer_auto(AUTO_TURNO_AD)
ok(t.get("turno") == {"fecha": "2025-02-21", "ponente": "J. Guadalupe Tafoya Hernández", "titulo": "Magistrado"},
   f"turno con ponente (TÚRNENSE en versales): {t.get('turno')}")
ok(t.get("ministerio_publico") == "sin_pedimento" and not t.get("admision"), "«no formuló pedimento» consta; no es admisión")
r = ft.leer_auto(AUTO_RETURNO)
ok((r.get("returno") or {}).get("ponente") == "Luis Armando Pérez Topete" and not r.get("turno"),
   "returno: «retúrnense» no se lee como turno")
h = ft.leer_auto(AUTO_ADHESIVO)
ok(h.get("adhesivo") == {"quien": "Mariana Paulina Nieves Ortega", "admision": "2026-02-10",
                         "presentacion": "2026-02-06"} and not h.get("admision"),
   f"el adhesivo, sin confundir su admisión con la del asunto: {h.get('adhesivo')} {h.get('admision')}")
r = ft.leer_auto(AUTO_AR)
ok(r.get("promovente", "").startswith("Impulsora de Desarrollos Inmobiliarios VV") and r.get("caracter") == "tercero"
   and r.get("ministerio_publico") == "pedimento" and (r.get("acto") or {}).get("expediente") == "950/2024",
   "revisión: quien recurre y su carácter, el pedimento y el juicio de amparo")
q = ft.leer_auto(AUTO_Q_FR2)
# CUARTA RONDA (E7): el auto que registra y requiere el informe es el REGISTRO;
# el que lo tiene por rendido y admite es la admisión (y el informe).
ok((q.get("registro") or {}).get("fecha") == "2026-03-03" and (q.get("admision") or {}).get("fecha") == "2026-03-09"
   and (q.get("informe_101") or {}).get("fecha") == "2026-03-09"
   and not q.get("avisos"), f"queja fr. II: registro y, aparte, el auto que tiene por rendido el informe y admite: "
                            f"{q.get('registro')} {q.get('admision')} {q.get('avisos')}")
f_ = ft.leer_auto(AUTO_RF)
ok(f_.get("presentacion") == "2025-11-13" and f_.get("via_presentacion") == "postal"
   and (f_.get("acto") or {}).get("expediente") == "695/25-09-01-7-OT", "revisión fiscal: portada, correo y expediente del TFJA")
j = ft.leer_auto(AUTO_ADMISION_AD + "\n" + AUTO_TURNO_AD + "\n" + AUTO_RETURNO)
ok((j.get("admision") or {}).get("fecha") == "2025-01-20" and (j.get("turno") or {}).get("fecha") == "2025-02-21"
   and (j.get("returno") or {}).get("fecha") == "2025-11-25", "tres autos pegados se leen auto por auto")
ok(set((j.get("fuentes") or {}).values()) == {"auto"}, "todo con fuente «auto»")

print("\n6 · LOS DERECHOS DE LA DEMANDA")
ok(ft.derechos_de(DEMANDA_AD) == ["14", "16", "17"],
   f"del capítulo, sin la Convención ni el 103/107 del proemio: {ft.derechos_de(DEMANDA_AD)}")
ok(ft.derechos_de("PRECEPTOS VIOLENTADOS: Artículos 172, fracciones XI y XII, de la Ley de Amparo, en relación con "
                  "los artículos 1°, 14, 16 Y 17 de la Constitución Federal. PRECEPTOS NO APLICADOS.- 1") ==
   ["1o.", "14", "16", "17"], "la Ley de Amparo no cuenta; el 1° se escribe «1o.»")
ok(ft.derechos_de("VI.- Los preceptos que, conforme a la fracción I del artículo 1º de la Ley de Amparo, contengan "
                  "los derechos humanos cuya violación se reclame. Los artículos 1o., 14 y 16 de la Constitución "
                  "Política de los Estados Unidos Mexicanos. VII.-") == ["1o.", "14", "16"],
   "el rótulo del art. 175, fr. VI, no se cuenta como lista")
ok(ft.derechos_de("Se viola en perjuicio de la Quejosa las Garantías consagradas en los Artículos 1° , 14, 16 y 17 "
                  "en relación con los artículos 39, y 133 de la Constitución Política") ==
   ["1o.", "14", "16", "17", "39", "133"], "dos listas encadenadas a la Constitución")
ok(ft.derechos_de("los derechos humanos de mis representados y conceptos de violación") == [],
   "sin capítulo, lista vacía (el compositor pone el hueco y lo avisa)")

print("\n7 · EL SENTIDO, DE UN CATÁLOGO")
ok(ft.sentido_de("queja", ACTO_Q)[:2] == ("desecha_demanda", "desechó la demanda de amparo"), "queja: desechó la demanda")
ok(ft.sentido_de("queja", "…se niega a la parte quejosa la suspensión provisional solicitada. Notifíquese.")[0]
   == "niega_provisional", "queja: negó la suspensión provisional")
ok(ft.sentido_de("revision_fiscal", ACTO_RF)[0] == "nulidad_lisa", "revisión fiscal: nulidad lisa y llana")
k, fr, _ = ft.sentido_de("revision_fiscal", "R E S U E L V E: I. Se sobresee respecto de la multa; II. Se reconoce "
                                            "la validez de la resolución impugnada. Notifíquese.")
ok(k == "sobresee+validez" and fr.startswith("sobreseyó parcialmente en el juicio y reconoció la validez"),
   f"sobreseyó en parte y resolvió el fondo del resto: «{fr}»")

print("\n8 · EL ACTO CON MODELO (cliente falso), TODO ANCLADO")
cli, comp = cliente_falso({
    "clase": "sentencia", "fecha": "2025-02-20",
    "organo": "Segunda Sala Civil del Tribunal Superior de Justicia del Estado de Querétaro",
    "toca": "4520/2024", "expediente": "479/2024",
    "ordenadora": "Segunda Sala Civil del Tribunal Superior de Justicia del Estado de Querétaro",
    "ejecutora": "Juez Noveno de lo Civil de Querétaro",            # inventado: no está
    "terceros": ["Gabriel Reyes Álamo", "Pedro Inventado Pérez"],    # el segundo no está
    "quejoso": "Ma. del Refugio Trejo Ramos", "notificacion": "2025-02-22"})   # fecha que no está
x = asyncio.run(ft.leer_acto(cli, ACTO_AD, "amparo_directo", "274/2025", demanda=DEMANDA_AD))
ok(len(comp.vistos) == 1 and comp.vistos[0]["model"] == ft.MODELO_ACTO
   and comp.vistos[0]["max_completion_tokens"] <= 2500
   and comp.vistos[0]["response_format"] == {"type": "json_object"}, "UNA llamada JSON, gpt-5.6-luna, tope ≤ 2500")
ok(x["acto"]["fecha"] == "2025-02-20" and x["acto"]["toca"] == "4520/2024" and x["acto"]["expediente"] == "479/2024",
   f"fecha, toca y expediente: {x['acto']}")
ok(x.get("responsable") == "Segunda Sala Civil del Tribunal Superior de Justicia del Estado de Querétaro"
   and x["acto"]["organo"] == x["responsable"], "la ordenadora, con su nombre entero, es quien dictó la sentencia")
ok(not x.get("ejecutora") and any("Juez Noveno" in a for a in x["avisos"]), "la ejecutora inventada se tira con aviso")
ok(x.get("terceros") == ["Gabriel Reyes Álamo"] and any("Pedro Inventado" in a for a in x["avisos"]),
   "el tercero que no está en el papel se tira")
ok(x.get("notificacion") == "2025-02-21", "la notificación, de la demanda (sin modelo); la del modelo no estaba escrita")
ok(x.get("derechos") == ["14", "16", "17"], "los derechos, de la demanda")
ok(x["acto"].get("instancia") == "alzada", "hubo apelación (toca): alzada")

cli, comp = cliente_falso({"clase": "auto", "fecha": "2026-03-06", "organo": "Jueza Tercero de Distrito",
                           "expediente": "1234/2025", "via": "indirecto", "incidente": False,
                           "sentido_clave": "admite_demanda", "quejoso": "Comercializadora del Bajío"})
q = asyncio.run(ft.leer_acto(cli, ACTO_Q, "queja", "50/2026"))
ok(q["acto"].get("sentido") == "desechó la demanda de amparo" and q.get("inciso_97") == "a" and q.get("fraccion_97") == "I",
   f"queja: el catálogo verificado manda; inciso a), fracción I: {q['acto'].get('sentido')} {q.get('inciso_97')}")
ok(q["acto"].get("expediente") == "1234/2025" and q["acto"].get("fecha") == "2026-03-06", "el auto recurrido y su juicio")
ok(q["acto"].get("organo", "").upper().startswith("JUZGADO TERCERO DE DISTRITO"), "el juzgado, de su cabecera")
cli, comp = cliente_falso({"sentido_clave": "concede_provisional", "fecha": "2026-03-06"})
q2 = asyncio.run(ft.leer_acto(cli, ACTO_Q.replace("se desecha de plano la demanda de amparo", "se acuerda lo conducente"),
                              "queja", "50/2026"))
ok(not q2["acto"].get("sentido") and any("hueco" in a.lower() for a in q2["avisos"]),
   "la clave que el papel no sostiene no entra: el sentido queda en hueco, con aviso")
cli, comp = cliente_falso({"fecha": "2025-09-22", "sala": "Sala Regional del Centro II del Tribunal Federal de "
                                                          "Justicia Administrativa",
                           "expediente_tfja": "695/25-09-01-7-OT", "actora": "Transportes Ejemplo del Centro, S.A. de C.V.",
                           "oficio": "500-62-00-03-01-2025-1234", "fecha_resolucion_impugnada": "2025-01-15",
                           "autoridad_emisora": "Administración Desconcentrada de Auditoría Fiscal de Querétaro \"1\"",
                           "sentido_clave": "nulidad_lisa"})
r = asyncio.run(ft.leer_acto(cli, ACTO_RF, "revision_fiscal", "91/2025"))
ok(r.get("expediente_tfja") == "695/25-09-01-7-OT" and r["acto"].get("fecha") == "2025-09-22"
   and r["acto"].get("sentido") == "declaró la nulidad lisa y llana de la resolución impugnada",
   "revisión fiscal: expediente del TFJA, fecha y sentido del catálogo")
ok(r.get("resolucion_impugnada", {}).get("oficio") == "500-62-00-03-01-2025-1234"
   and r.get("resolucion_impugnada", {}).get("fecha") == "2025-01-15" and r.get("actora", "").startswith("Transportes"),
   f"la resolución impugnada y la actora: {r.get('resolucion_impugnada')}")
s = asyncio.run(ft.leer_acto(None, "", "amparo_directo"))
ok(s["avisos"] and not s.get("acto"), "sin texto del acto: aviso, ningún dato")
ok(r.get("actora") == "Transportes Ejemplo del Centro, S.A. de C.V.", "la actora con su «S.A. de C.V.» entero")
ok(ft.sin_versales("SALA REGIONAL DEL CENTRO II DEL TRIBUNAL FEDERAL DE JUSTICIA ADMINISTRATIVA")
   == "Sala Regional del Centro II del Tribunal Federal de Justicia Administrativa", "sin versales (los romanos, igual)")
ok(q["acto"].get("organo", "") == "Juzgado Tercero de Distrito en Materia de Amparo Civil, Administrativo y de Trabajo "
   "y de Juicios Federales en el Estado de Querétaro", "la cabecera partida en dos renglones se lee entera")

print("\n8b · LA SENTENCIA RECURRIDA (AR), SIN MODELO")
ACTO_AR = """JUICIO DE AMPARO 950/2024.
S E N T E N C I A
En Santiago de Querétaro, Querétaro, a veintidós de abril de dos mil veinticinco.
V I S T O S los autos para resolver el juicio de amparo 950/2024, promovido por la Unión de Trabajadores
de la Construcción del Estado de Querétaro, C.T.M., en contra de los actos de la Primera Sala Civil del
Tribunal Superior de Justicia del Estado de Querétaro, y otra autoridad; y,
R E S U L T A N D O
PRIMERO. Presentación de la demanda. El dieciséis de julio del dos mil veinticuatro, se depositó en el
buzón judicial el escrito por el que la quejosa promovió juicio de amparo.
C O N S I D E R A N D O
PRIMERO. Competencia. Este Juzgado Séptimo de Distrito
en Materia de Amparo Civil, Administrativo y de Trabajo y de Juicios Federales en el Estado de Querétaro,
es competente para conocer y resolver este juicio.
Por lo expuesto, se R E S U E L V E: ÚNICO. La Justicia de la Unión ampara y protege a la Unión de
Trabajadores de la Construcción del Estado de Querétaro, C.T.M., contra el acto reclamado a la Primera Sala
Civil del Tribunal Superior de Justicia del Estado de Querétaro. Notifíquese."""
ar = asyncio.run(ft.leer_acto(None, ACTO_AR, "amparo_revision", "631/2025"))
ok(ar["acto"].get("fecha") == "2025-04-22" and ar["acto"].get("expediente") == "950/2024"
   and ar["acto"].get("resolvio") == "concede" and ar.get("clase_recurrida") == "sentencia",
   f"fecha, juicio, lo resuelto por sus puntos resolutivos y la clase: {ar['acto']}")
ok(ar["acto"].get("organo", "").endswith("Juicios Federales en el Estado de Querétaro"),
   "el juzgado, con el renglón que la cabecera parte")
ok((ar.get("demanda") or {}).get("fecha") == "2024-07-16", "la demanda de amparo indirecto, del resultando primero")
ok(ar.get("quejoso", "").startswith("Unión de Trabajadores"), "la quejosa, del resolutivo (sin el artículo)")
inc = asyncio.run(ft.leer_acto(None, ACTO_AR.replace("S E N T E N C I A", "INCIDENTE DE SUSPENSIÓN. INTERLOCUTORIA")
                                      .replace("La Justicia de la Unión ampara y protege a", "Se concede la suspensión definitiva a"),
                              "amparo_revision", "631/2025"))
ok(inc["acto"].get("incidente") is True and inc["acto"].get("resolvio") == "concede"
   and inc.get("clase_recurrida") == "interlocutoria_suspension", "la interlocutoria de suspensión: concedió la definitiva")
ok(x["acto"].get("lo_resuelto") == "ese recurso de apelación", "lo resuelto (fase_origen, vía origen_acto)")
ok(ft.leer_auto("Querétaro, Querétaro, a veinticinco de noviembre de dos mil veinticinco. Visto que por auto de "
                "Presidencia de veinte de enero de dos mil veinticinco se admitió la demanda, retúrnense los autos a "
                "la ponencia de la Magistrada Jenica Campos Juárez.").get("admision") == {"fecha": "2025-01-20"},
   "«por auto de Presidencia de…», citado en otro auto")


class _Rompe:
    async def create(self, **kw):
        raise RuntimeError("caído")


s = asyncio.run(ft.leer_acto(types.SimpleNamespace(chat=types.SimpleNamespace(completions=_Rompe())),
                             ACTO_RF, "revision_fiscal", "91/2025"))
ok(s["acto"].get("fecha") == "2025-09-22" and any("sin modelo" in a for a in s["avisos"]),
   "si el modelo falla, vale lo leído sin modelo (y se dice)")

print("\n9 · ARMAR: PRECEDENCIA, FUENTES Y CRONOLOGÍA")
E = {"tipo_asunto": "amparo_directo", "numero": "274/2025", "materia": "civil",
     "tribunal": "Tercer Tribunal Colegiado en Materias Administrativa y Civil del Vigésimo Segundo Circuito",
     "ciudad": "Querétaro, Querétaro", "quejoso": "Ma. del Refugio Trejo Ramos",
     "presentacion": "2025-03-13", "notificacion": "2025-02-21", "responsable": "TRIBUNAL EL SEIS"}
auto = ft.leer_auto(AUTO_TURNO_AD.replace("43/2025", "274/2025"))
auto2 = {"admision": {"fecha": "2025-04-01"}, "acto": {"fecha": "2025-02-19"}, "avisos": []}
FP = {"quejosa": {"nombre": "Ma. del Refugio Trejo Ramos"}, "terceros": [{"nombre": "Gabriel Reyes Álamo"}],
      "responsables": [{"autoridad": "Sala Civil"}], "adhesivo": {"consta": False}}
fi = ft.armar(E, auto=[auto2, auto], acto=x, ficha_procesal=FP,
              declarado={"fecha_turno": "2025-05-14", "ponente_turno": "Luis Armando Pérez Topete",
                         "ministerio_publico": "no consta"})
ok(fi["formato"] == 1 and fi["tipo"] == "amparo_directo" and fi["numero"] == "274/2025", "forma y número")
ok(fi["turno"]["fecha"] == "2025-05-14" and fi["fuentes"]["turno.fecha"] == "secretario"
   and fi["turno"]["ponente"] == "Luis Armando Pérez Topete", "lo del formulario manda sobre el auto")
ok(any("NO CUADRA" in a and "turno" in a.lower() for a in fi["avisos"]), "…y se avisa que el auto decía otra fecha")
ok(fi["ministerio_publico"] == "sin_pedimento" and fi["fuentes"]["ministerio_publico"] == "auto",
   "«no consta» del secretario no borra lo que el auto sí dice")
ok(fi["admision"]["fecha"] == "2025-04-01" and fi["fuentes"]["admision.fecha"] == "auto", "la admisión, del auto")
ok(fi["acto"]["fecha"] == "2025-02-19" and fi["fuentes"]["acto.fecha"] == "auto"
   and any("FECHA DE LO RECLAMADO" in a.upper() and "NO CUADRA" in a for a in fi["avisos"]),
   "el auto manda sobre el acto, y la discrepancia se dice")
ok(fi["responsable"] == "Segunda Sala Civil del Tribunal Superior de Justicia del Estado de Querétaro"
   and fi["fuentes"]["responsable"] == "acto" and fi["acto"]["organo"] == fi["responsable"],
   f"la responsable: la del papel, no «TRIBUNAL EL SEIS» del formulario ni «Sala Civil»: {fi['responsable']}")
ok(fi["terceros"] == ["Gabriel Reyes Álamo"] and fi["derechos"] == ["14", "16", "17"], "terceros y derechos")
ok(fi["sede"]["circuito"] == "XXII" and fi["sede"]["cdmx"] is False, "la sede")
ok(fi["adhesivo"] == {} and fi["returno"] == {} and fi["informe_101"] == {} and fi["demanda"] == {},
   "lo opcional que no consta es {} (así `if ficha['adhesivo']` dice la verdad)")
ok(fi["fechas_imposibles"] == [] and fi["presentacion"] == "2025-03-13", "una cronología posible no avisa nada")
json.dumps(fi, ensure_ascii=False)
ok(True, "la ficha es serializable")

f2 = ft.armar(dict(E, presentacion="2025-01-10", numero="274/2027"), acto=x,
              declarado={"fecha_admision": "2025-04-01", "fecha_turno": "2025-03-30",
                         "adhesivo_quien": "Gabriel Reyes Álamo", "adhesivo_admision": "2025-03-01"})
imp = f2["fechas_imposibles"]
ok("presentacion < acto.fecha" in imp and "turno.fecha < admision.fecha" in imp
   and "año(numero) vs presentacion" in imp and "adhesivo.admision < admision.fecha" in imp,
   f"cronología imposible, cada una con su aviso: {imp}")
ok(sum(1 for a in f2["avisos"] if a.startswith("FECHA IMPOSIBLE")) == len(imp), "un aviso por violación")
ok(f2["adhesivo"]["quien"] == "Gabriel Reyes Álamo" and set(f2["adhesivo"]) == {"quien", "presentacion", "admision", "notificacion"},
   "el adhesivo que consta trae todas sus claves")
f3 = ft.armar(dict(E, notificacion="2025-02-10"), acto=x)
ok("notificacion < acto.fecha" in f3["fechas_imposibles"], "notificación antes de la sentencia: imposible")
f4 = ft.armar(dict(E, presentacion="2025-02-20", notificacion="2025-02-21"), acto=x)
ok(f4["fechas_imposibles"] == [], "presentar antes de notificarse vale (anticipada): no es imposible")
f5 = ft.armar(dict(E, responsable="Sala Familiar del Tribunal Superior de Justicia en el Estado"),
              acto={"acto": {}, "responsable": "Sala Familiar del Tribunal Superior de Justicia del Estado de Querétaro"},
              declarado={"organo_acto": "Sala Familiar del Tribunal Superior de Justicia en el Estado"})
ok(f5["responsable"] == "Sala Familiar del Tribunal Superior de Justicia del Estado de Querétaro"
   and any("SE COMPLETÓ" in a for a in f5["avisos"]),
   "«…en el Estado» sin entidad: si otra fuente la trae completa, la completa")
f6 = ft.armar(dict(E, responsable="TRIBUNAL EL SEIS"), acto={})
ok(f6["responsable"] == "" and any("TRIBUNAL EL SEIS" in a for a in f6["avisos"])
   and any("NO CONSTA LA AUTORIDAD RESPONSABLE" in a for a in f6["avisos"]),
   "sin ninguna fuente buena: vacía y dicha (el compositor pone el hueco)")
fx = ft.armar(dict(E, presentacion="2027-01-10"), acto=x)
ok("presentacion futura" in fx["fechas_imposibles"], "una fecha posterior a hoy es imposible")
ok(ft.armar(dict(E, regla_surtimiento="tja_qro_boletin"), acto=x)["forma_notificacion"] == "boletin"
   and ft.armar(dict(E, regla_surtimiento="otra"), acto=x)["forma_notificacion"] == "",
   "la forma de notificación, de la regla del formulario (y «otra» no dice cuál)")
import datetime as _dtt
try:
    import redactor_adelanto as _ra
    _Enc = _ra.Encargo
except Exception as _ex:      # otra pieza puede estar editándolo: se prueba con su forma
    print(f"   (redactor_adelanto no importa ahora: {type(_ex).__name__}; se usa un sustituto)")
    _Enc = lambda **kw: types.SimpleNamespace(**dict({"regla_surtimiento": "personal", "materia": "",
                                                      "recurrente": "", "responsable": None}, **kw))
enc_real = _Enc(numero="274/2025", encabezado="AMPARO DIRECTO CIVIL: 274/2025", quejoso="Gabriel Reyes Álamo",
                       magistrado="", secretario="", notificacion=_dtt.date(2025, 2, 21),
                       presentacion=_dtt.date(2025, 3, 13), tipo_asunto="amparo_directo",
                       tribunal=E["tribunal"], ciudad="Querétaro, Querétaro")
fe = ft.armar(enc_real, acto=x)
ok(fe["presentacion"] == "2025-03-13" and fe["materia"] == "civil" and fe["promovente"] == "Gabriel Reyes Álamo"
   and fe["fuentes"]["presentacion"] == "secretario",
   "con el Encargo de verdad: fechas como date, materia del encabezado, promovente")

AR = {"tipo_asunto": "amparo_revision", "numero": "631/2025", "es_recurso": True, "quejoso": "",
      "recurrente": "Impulsora de Desarrollos Inmobiliarios VV, Sociedad Anónima de Capital Variable",
      "presentacion": "2025-05-20", "notificacion": "2025-05-08"}
FPR = {"recurrente": {"nombre": "Impulsora de Desarrollos Inmobiliarios VV", "papel": "tercero"},
       "resolvio": {"que": "sobresee_concede"},
       "responsables": [{"autoridad": "Primera Sala Civil del Tribunal Superior de Justicia del Estado de Querétaro"}]}
fr_ = ft.armar(AR, auto=ft.leer_auto(AUTO_AR), acto={"acto": {"fecha": "2025-04-22"}, "audiencia": "2025-04-30"},
               ficha_procesal=FPR)
ok(fr_["caracter"] == "tercero" and fr_["acto"]["expediente"] == "950/2024" and fr_["acto"]["resolvio"] == "mixto"
   and fr_["acto"].get("resolvio_mixto") == "sobresee_concede", "AR: carácter, juicio y lo resuelto (mixto) de la ficha procesal")
ok(fr_["demanda"]["autoridades"] == ["Primera Sala Civil del Tribunal Superior de Justicia del Estado de Querétaro"]
   and fr_["clase_recurrida"] == "sentencia", "AR: las autoridades del amparo indirecto; clase de lo recurrido")
ok("audiencia > acto.fecha" in fr_["fechas_imposibles"], "AR: audiencia después de la sentencia, imposible")
fq = ft.armar({"tipo_asunto": "queja", "numero": "143/2026", "presentacion": "2026-02-25"},
              auto=ft.leer_auto(AUTO_Q_FR2), acto={"acto": {"via": "directo"}})
ok(fq["fraccion_97"] == "II" and fq["informe_101"] == {"fecha": "2026-03-09"} and fq.get("registro") == {"fecha": "2026-03-03"}
   and fq["admision"]["fecha"] == "2026-03-09",
   "queja fr. II: la fracción de la vía; el registro (que requiere el informe) aparte de la admisión (cuarta ronda)")
frf = ft.armar({"tipo_asunto": "revision_fiscal", "numero": "91/2025", "presentacion": "2025-11-13"},
               auto=ft.leer_auto(AUTO_RF), acto=r, declarado={"deposito_postal": "2025-11-10"})
ok(frf["sala"].startswith("Sala Regional del Centro II") and frf["expediente_tfja"] == "695/25-09-01-7-OT"
   and frf["deposito_postal"] == "2025-11-10" and frf["caracter"] == "autoridad" and frf["fechas_imposibles"] == [],
   "revisión fiscal: Sala, expediente, depósito postal y carácter de autoridad")
frf2 = ft.armar({"tipo_asunto": "revision_fiscal", "numero": "91/2025", "presentacion": "2025-11-13"},
                acto=r, declarado={"deposito_postal": "2025-11-20"})
ok("deposito_postal > presentacion" in frf2["fechas_imposibles"], "depositar después de recibido: imposible")

print("\n10 · EL FORMULARIO: IDA Y VUELTA")
form = {"fecha_admision": "20/01/2025", "fecha_turno": "veintiuno de febrero de dos mil veinticinco",
        "ponente_turno": "  J. Guadalupe   Tafoya Hernández ", "ministerio_publico": "sí",
        "adhesivo_quien": "", "fecha_acto": "2025-02-20", "toca": " 4520 / 2024",
        "fecha_returno": "no es fecha", "forma_notificacion": "Lista"}
d = ft.de_formulario(json.dumps(form))
ok(d["admision"]["fecha"] == "2025-01-20" and d["turno"]["fecha"] == "2025-02-21"
   and d["turno"]["ponente"] == "J. Guadalupe Tafoya Hernández" and d["ministerio_publico"] == "pedimento"
   and d["acto"]["toca"] == "4520/2024" and d["forma_notificacion"] == "lista", "fechas en cualquier forma, alias y limpieza")
ok("returno" not in d and any("no es una fecha" in a for a in d["avisos"]), "lo que no es fecha no entra y se avisa")
ok(set(d["fuentes"].values()) == {"secretario"}, "todo con fuente «secretario»")
plano = ft.a_formulario(fi)
ok(list(plano) == list(ft.CLAVES_FORMULARIO) and all(isinstance(v, str) for v in plano.values()),
   "a_formulario: las 22 claves, en cadena")
ok(plano["fecha_turno"] == "2025-05-14" and plano["organo_acto"].startswith("Segunda Sala Civil")
   and plano["ministerio_publico"] == "sin_pedimento" and plano["adhesivo_quien"] == "", "…con lo de la ficha")
ida = ft.de_formulario(plano)
ok(ft.a_formulario(ida) == plano, "ida y vuelta sin pérdida")
ok(ft.de_formulario("{no es json")["avisos"], "un JSON roto se ignora con aviso")
ok(ft.a_formulario(frf)["expediente_origen"] == "695/25-09-01-7-OT", "RF: el expediente del TFJA va como expediente de origen")

print("\n11 · fase_admision.leer DEVUELVE ADEMÁS EL TRÁMITE")
cli, comp = cliente_falso({"tipo_asunto": "", "quejoso": "Banco Mercantil del Norte, Sociedad Anónima, Institución "
                                                         "de Banca Múltiple, Grupo Financiero Banorte",
                           "responsable_ordenadora": "", "tercero_interesado": "José Antonio Juárez Lira"})
fic = asyncio.run(fa.leer(cli, AUTO_ADMISION_AD, con_tramite=True))
ok(fic.get("numero") == "43/2025" and fic.get("tercero_interesado") == "José Antonio Juárez Lira"
   and fic.get("ciudad") == "Querétaro, Querétaro", "la ficha de siempre, intacta (y la ciudad «a veinte de…» ya se lee)")
ok(isinstance(fic.get("tramite"), dict) and fic["tramite"]["fecha_admision"] == "2025-01-20"
   and fic["tramite"]["fecha_acto"] == "2024-07-03" and set(fic["tramite"]) == set(ft.CLAVES_FORMULARIO),
   f"…y además «tramite», con las claves del formulario: {fic.get('tramite', {}).get('fecha_admision')}")
ok(len(comp.vistos) == 1, "sin llamada de modelo de más")
# SEXTA RONDA (3-oct-2026, C8): la bandera nace «todos» (David: «empuja para
# todos los usuarios»). Sin contexto RIGE; PROCEDENCIA_POR_TIPO=0 es el freno y
# con él la ficha sale como siempre.
ok(ft.rige() is True, "sin contexto, la bandera rige («todos» por omisión)")
_pt_antes = os.environ.get("PROCEDENCIA_POR_TIPO")
os.environ["PROCEDENCIA_POR_TIPO"] = "0"
try:
    cli, comp = cliente_falso({})
    fic2 = asyncio.run(fa.leer(cli, AUTO_ADMISION_AD))
    ok("tramite" not in fic2, "con la bandera apagada (PROCEDENCIA_POR_TIPO=0), la ficha sale como siempre")
    ok(ft.rige() is False, "PROCEDENCIA_POR_TIPO=0: la bandera no rige")
finally:
    if _pt_antes is None:
        os.environ.pop("PROCEDENCIA_POR_TIPO", None)
    else:
        os.environ["PROCEDENCIA_POR_TIPO"] = _pt_antes
cd = fa.deterministas("Ciudad de México, a quince de marzo de dos mil veintiséis. Fórmese el expediente número 12/2026")
ok(cd.get("ciudad") == "Ciudad de México", "Ciudad de México, a quince…: la ciudad se lee")

print("\n12 · LOS AUTOS CON MODELO, ANCLADOS")
cli, comp = cliente_falso({"numero": "43/2025", "fecha_admision": "2025-01-20", "fecha_turno": "2025-02-21",
                           "ponente_turno": "J. Guadalupe Tafoya Hernández", "ministerio_publico": "pedimento",
                           "fecha_acto": "2024-07-03", "toca": "999/2024", "adhesivo_quien": "Nadie Que Exista"})
m_ = asyncio.run(ft.leer_auto_modelo(cli, AUTO_ADMISION_AD + "\n" + AUTO_TURNO_AD, "amparo_directo"))
ok((m_.get("turno") or {}).get("ponente") == "J. Guadalupe Tafoya Hernández" and (m_.get("admision") or {}).get("fecha") == "2025-01-20",
   "lo que está en el papel entra")
ok(m_.get("ministerio_publico") != "pedimento", "«pedimento» sin la fórmula en el papel no entra (el turno dice que NO lo formuló)")
ok(not (m_.get("acto") or {}).get("toca") and not m_.get("adhesivo") and any("Nadie Que Exista" in a for a in m_["avisos"]),
   "un toca y un adherente que no están se tiran")
ok(comp.vistos and comp.vistos[0]["model"] == fa.MODELO_ADMISION, "con el modelo de la admisión")
cli, comp = cliente_falso({"fecha_turno": "2025-02-21", "ponente_turno": "J. Guadalupe Tafoya Hernández"})
c_ = asyncio.run(ft.leer_auto_completo(cli, AUTO_ADMISION_AD, "amparo_directo"))
ok((c_.get("admision") or {}).get("fecha") == "2025-01-20" and not c_.get("turno"),
   "completo: lo de sin modelo manda; un turno que no está en ESTE auto no entra")

print("\n13 · SEGUNDA RONDA: LOS TRES CAMPOS NUEVOS (fraccion_63, cuantia, fundamento_surtimiento)")
# SEXTA RONDA (3-oct-2026): 22 claves; «relacionados» al final, tras «fecha_registro».
ok(len(ft.CLAVES_FORMULARIO) == 22 and ft.CLAVES_FORMULARIO[-5:-2] == ("fraccion_63", "cuantia",
                                                                         "fundamento_surtimiento"),
   "CLAVES_FORMULARIO: 22 claves (sexta ronda), las tres de la segunda con su mismo nombre")
ok([ft.fraccion_63_de(v) for v in ("3", "iii", "fracción III", "fr. IV.", "artículo 63, fracción II", "10")]
   == ["III", "III", "III", "IV", "II", "X"], "la fracción del 63, en romano, se escriba como se escriba")
ok([ft.fraccion_63_de(v) for v in ("XI", "0", "III, inciso a)", "la tercera", "")] == [""] * 5,
   "lo que no es fracción de la I a la X no se adivina (ni la XI, ni «III, inciso a)»)")
ok([ft.cuantia_de(v) for v in ("$ 847,738.77", "847738.77", "$1,234,567", "847,738.77 pesos M.N.")]
   == ["$847,738.77", "$847,738.77", "$1,234,567.00", "$847,738.77"], "la cuantía, en UNA sola forma")
ok(ft.cuantia_de("cuantía indeterminada") == "indeterminada" and ft.cuantia_de("1.234.567,00") == ""
   and ft.cuantia_de("mucho") == "" and ft.cuantia_de("0") == "",
   "«indeterminada» se respeta; el formato europeo, las palabras y el cero no se adivinan")
ok(ft.cuantia_de(ft.cuantia_de("847738.77")) == "$847,738.77", "la cuantía es idempotente")
ok(ft.fundamento_surtimiento_de("artículo 126 del Código de Procedimientos Civiles del Estado de Querétaro.")
   == "el artículo 126 del Código de Procedimientos Civiles del Estado de Querétaro"
   and ft.fundamento_surtimiento_de("conforme al art. 126 del CPC") == "el artículo 126 del CPC"
   and ft.fundamento_surtimiento_de("en términos de los artículos 1 y 2 de la ley del acto")
   == "los artículos 1 y 2 de la ley del acto",
   "el fundamento del surtimiento con su artículo, sin conector repetido ni punto final")
ok(ft.fundamento_surtimiento_de("ARTÍCULO 126 DEL CÓDIGO DE PROCEDIMIENTOS CIVILES DEL ESTADO DE QUERÉTARO")
   == "el artículo 126 del Código de Procedimientos Civiles del Estado de Querétaro",
   "el fundamento en versales pasa a prosa")
_FUND = "el artículo 126 del Código de Procedimientos Civiles del Estado de Querétaro"
d3 = ft.de_formulario({"fraccion_63": "fracción I", "cuantia": "847738.77", "fundamento_surtimiento":
                       "conforme al artículo 126 del Código de Procedimientos Civiles del Estado de Querétaro"})
ok(d3.get("fraccion_63") == "I" and d3.get("cuantia") == "$847,738.77" and d3.get("fundamento_surtimiento") == _FUND
   and {d3["fuentes"][k] for k in ("fraccion_63", "cuantia", "fundamento_surtimiento")} == {"secretario"},
   "de_formulario: los tres, normalizados y con fuente «secretario»")
d4 = ft.de_formulario({"fraccion_63": "XI", "cuantia": "un millón"})
ok("fraccion_63" not in d4 and "cuantia" not in d4 and any("(fraccion_63)" in a for a in d4["avisos"])
   and any("(cuantia" in a for a in d4["avisos"]), "lo que no se lee no entra, y el aviso nombra la clave")
frf3 = ft.armar({"tipo_asunto": "revision_fiscal", "numero": "91/2025", "presentacion": "2025-11-13"},
                acto=r, declarado={"fraccion_63": "I", "cuantia": "$847,738.77", "fundamento_surtimiento": _FUND})
ok(frf3["fraccion_63"] == "I" and frf3["cuantia"] == "$847,738.77" and frf3["fundamento_surtimiento"] == _FUND
   and all(frf3["fuentes"][k] == "secretario" for k in ("fraccion_63", "cuantia", "fundamento_surtimiento")),
   "armar: los tres en la ficha, en primer nivel, con fuente «secretario»")
ok(all(k in ft.armar({"tipo_asunto": "queja"}) for k in ("fraccion_63", "cuantia", "fundamento_surtimiento")),
   "armar: los tres están siempre (vacíos si no constan)")
_pl3 = ft.a_formulario(frf3)
ok(_pl3["fraccion_63"] == "I" and _pl3["cuantia"] == "$847,738.77" and _pl3["fundamento_surtimiento"] == _FUND,
   "a_formulario los devuelve con el mismo nombre")

print("\n14 · SEGUNDA RONDA: organo_acto Y expediente_origen, POR TIPO (ida y vuelta)")
_COMUNES = {"fecha_admision": "2025-09-19", "fecha_turno": "2025-11-11", "ponente_turno": "Luis Armando Pérez Topete",
            "fecha_returno": "2025-12-01", "ponente_returno": "Magistrada Ana Laura Campos Juárez",
            "ministerio_publico": "pedimento", "fecha_acto": "2025-07-24", "fundamento_surtimiento": _FUND}
_ADH = {"adhesivo_quien": "Gabriel Reyes Álamo", "adhesivo_presentacion": "2025-10-08",
        "adhesivo_admision": "2025-10-10", "adhesivo_notificacion": "2025-09-25"}
_SALA = "Sala Regional del Centro II"
FORMS = {
    "amparo_directo": dict(_COMUNES, **_ADH, organo_acto="Segunda Sala Civil del Tribunal Superior de Justicia "
                                                         "del Estado de Querétaro",
                           toca="4357/2025", expediente_origen="515/2022"),
    "amparo_revision": dict(_COMUNES, **_ADH, organo_acto="Juzgado Cuarto de Distrito en el Estado de Querétaro",
                            expediente_origen="950/2024"),
    # TERCERA RONDA: el órgano en forma de ÓRGANO («Juzgado…»). El compositor de
    # la queja (dueño D) escribe ahora el juzgado y no «Juez…»; esta prueba mira
    # a qué clave llega el dato, no esa normalización.
    "queja": dict(_COMUNES, organo_acto="Juzgado Séptimo de Distrito en el Estado de Querétaro",
                  expediente_origen="312/2026", fecha_informe_101="2025-11-05"),
    "revision_fiscal": dict(_COMUNES, **_ADH, organo_acto=_SALA, expediente_origen="1234/25-22-01-1",
                            deposito_postal="2025-08-20", fraccion_63="I", cuantia="$847,738.77"),
}
# Dónde tiene que quedar cada uno (SPEC §1.1 y front.txt #48).
DESTINO = {
    "amparo_directo": {"responsable": FORMS["amparo_directo"]["organo_acto"],
                       "acto.organo": FORMS["amparo_directo"]["organo_acto"], "acto.expediente": "515/2022",
                       "acto.toca": "4357/2025"},
    "amparo_revision": {"acto.organo": FORMS["amparo_revision"]["organo_acto"], "acto.expediente": "950/2024"},
    "queja": {"acto.organo": FORMS["queja"]["organo_acto"], "acto.expediente": "312/2026"},
    "revision_fiscal": {"sala": _SALA, "acto.organo": _SALA, "expediente_tfja": "1234/25-22-01-1",
                        "acto.expediente": "1234/25-22-01-1"},
}
for _t, _x in FORMS.items():
    _d = ft.de_formulario(_x, _t)
    ok(all(ft._tomar(_d, ruta) == v and _d["fuentes"].get(ruta) == "secretario" for ruta, v in DESTINO[_t].items()),
       f"{_t}: organo_acto y expediente_origen llegan a su campo, como del secretario "
       f"({ {ruta: ft._tomar(_d, ruta) for ruta in DESTINO[_t]} })")
    _vuelta = ft.a_formulario(_d, _t)
    ok({k: _vuelta[k] for k in _x} == _x, f"{_t}: a_formulario(de_formulario(x)) == x en las claves que aplican")
    ok(ft.a_formulario(_d) == _vuelta, f"{_t}: la ficha parcial recuerda su tipo (sin pasarlo de nuevo)")
    ok(ft.a_formulario(ft.de_formulario(json.dumps(_x), _t), _t) == _vuelta, f"{_t}: igual desde el JSON")
# Lo explícito manda sobre lo derivado: «sala» y «responsable» con su propio nombre.
# «sala» no es del contrato HTTP (tercera ronda): con `extra=True` (uso interno) sí entra y manda.
_dx = ft.de_formulario(dict(FORMS["revision_fiscal"], sala="Sala Regional del Golfo"), "revision_fiscal",
                       extra=True)
ok(_dx["sala"] == "Sala Regional del Golfo" and _dx["acto"]["organo"] == _SALA,
   "RF: si el uso interno manda «sala» con su nombre, organo_acto no la pisa")
_dx = ft.de_formulario({"tipo_asunto": "RF", "organo_acto": _SALA}, "")
ok(_dx.get("sala") == _SALA, "el tipo también puede venir dentro del formulario («tipo_asunto»)")
_dx = ft.de_formulario({"organo_acto": _SALA, "expediente_origen": "1/2025"})
ok("sala" not in _dx and _dx["acto"]["organo"] == _SALA and "tipo" not in _dx,
   "sin tipo: el campo genérico (acto.organo), como lo lee hoy el cableado")
_dx = ft.de_formulario({"toca": "toca civil 374 / 2024", "expediente_origen": " expediente 905 - 2023"},
                       "amparo_directo")
ok(_dx["acto"]["toca"] == "toca civil 374/2024" and _dx["acto"]["expediente"] == "expediente 905-2023",
   "el toca y el expediente con su rótulo conservan los espacios entre palabras (no «tocacivil374/2024»)")

# EL CAMINO DEL CABLEADO: redactor_adelanto.tramite_de_formulario llama a
# de_formulario SIN tipo (antes del OCR) y armar, que sí lo sabe, lo completa.
_sin_tipo = ft.de_formulario(json.dumps(FORMS["revision_fiscal"]))
frf4 = ft.armar({"tipo_asunto": "revision_fiscal", "numero": "91/2025", "presentacion": "2025-08-21"},
                auto=ft.leer_auto(AUTO_RF), acto=r, declarado=_sin_tipo)
ok(frf4["sala"] == _SALA and frf4["fuentes"]["sala"] == "secretario"
   and frf4["expediente_tfja"] == "1234/25-22-01-1" and frf4["fuentes"]["expediente_tfja"] == "secretario",
   f"RF por el cableado: la Sala y el expediente del secretario mandan sobre los del papel ({frf4['sala']!r})")
ok("sala" not in _sin_tipo, "…y el dict del llamador no se toca")
ok(frf4["admision"]["fecha"] == "2025-09-19" and frf4["fuentes"]["admision.fecha"] == "secretario"
   and frf4["turno"]["ponente"] == "Luis Armando Pérez Topete" and frf4["deposito_postal"] == "2025-08-20",
   "la ficha parcial con claves de nombre igual al plano (deposito_postal, fraccion_63…) no se relee "
   "como plana: conserva la admisión y el turno")
fad = ft.armar(dict(E, responsable=""), acto=x, declarado=ft.de_formulario(
    json.dumps({"organo_acto": "Primera Sala Penal del Tribunal Superior de Justicia del Estado de Querétaro"})))
ok(fad["responsable"].startswith("Primera Sala Penal") and fad["fuentes"]["responsable"] == "secretario"
   and fad["acto"]["organo"] == fad["responsable"],
   "AD por el cableado: organo_acto es la ordenadora (responsable y acto.organo, un solo nombre)")
ok(any("NO CUADRA ENTRE FUENTES" in a and "Segunda Sala Civil" in a for a in fad["avisos"]),
   "…y si el papel nombra OTRA Sala, se avisa (manda la del secretario, no se corrige)")
fad2 = ft.armar(dict(E, responsable=""), acto=x, declarado=ft.de_formulario(
    json.dumps({"organo_acto": "Segunda Sala Civil del Tribunal Superior de Justicia del Estado de Querétaro"})))
ok(not any("AUTORIDAD RESPONSABLE NO CUADRA" in a for a in fad2["avisos"]),
   "la misma Sala en el formulario y en el papel: sin aviso")

# DE PUNTA A PUNTA: formulario → armar → compositor → datos_extra (lo que leen
# los considerandos). Cada tipo, su órgano y su juicio donde los busca.
import resultandos_por_tipo as rpt
_ENC = {"numero": "150/2025", "materia": "civil", "presentacion": "2025-08-29",
        "tribunal": "Tercer Tribunal Colegiado en Materias Administrativa y Civil del Vigésimo Segundo Circuito",
        "ciudad": "Querétaro, Querétaro", "quejoso": "José Eduardo Aguilar Durán",
        "recurrente": "José Eduardo Aguilar Durán"}
_ESPERA = {
    "amparo_directo": {"responsable": FORMS["amparo_directo"]["organo_acto"],
                       "organo_acto": FORMS["amparo_directo"]["organo_acto"], "expediente": "515/2022",
                       "toca": "4357/2025"},
    "amparo_revision": {"juzgado": FORMS["amparo_revision"]["organo_acto"],
                        "organo_acto": FORMS["amparo_revision"]["organo_acto"], "juicio_amparo": "950/2024"},
    "queja": {"juzgado": FORMS["queja"]["organo_acto"], "organo_acto": FORMS["queja"]["organo_acto"],
              "juicio_amparo": "312/2026"},
    "revision_fiscal": {"sala": _SALA, "organo_acto": _SALA, "expediente_tfja": "1234/25-22-01-1",
                        "fraccion_63": "I", "cuantia": "$847,738.77"},
}
for _t, _x in FORMS.items():
    _fi = ft.armar(dict(_ENC, tipo_asunto=_t), declarado=json.dumps(_x))
    _ex = rpt.componer(_t, _fi, {"magistrado": "Luis Armando Pérez Topete"})["datos_extra"]
    _mal = {k: _ex.get(k) for k, v in _ESPERA[_t].items() if _ex.get(k) != v}
    ok(not _mal and _ex["fundamento_surtimiento"] == _FUND,
       f"{_t}: de la tarjeta a datos_extra, cada dato en su clave {_mal or ''}")
    ok(ft.a_formulario(_fi)["organo_acto"] == _x["organo_acto"]
       and ft.a_formulario(_fi)["expediente_origen"] == _x["expediente_origen"],
       f"{_t}: al reanudar, la tarjeta vuelve con lo que confirmó el secretario")

# ═══════════════════════════════════════════════════════════════════════════
print("\n15 · TERCERA RONDA (3-oct-2026): QUIÉN RECURRE, ROBUSTEZ, CRONOLOGÍA DEL AR Y ESQUEMA")
import random as _rnd
import re as _re
import threading as _th
import time as _time


def _seguro(fn, *a, **k):
    """Lo que devuelve `fn`, o la excepción (para que una API que aún no
    existe cuente como FALLA y no tumbe el resto)."""
    try:
        return fn(*a, **k)
    except Exception as ex:          # noqa: BLE001
        return ex


def _d(x):
    return x if isinstance(x, dict) else {}


# ── 15.1 · E3: EL ESCRITO DEL RECURSO DICE QUIÉN RECURRE ────────────────────
# La comparecencia del escrito real del AR 631/2025 (prueba de punta a punta),
# con su cabecera de OCR: la contraparte (la quejosa) aparece ANTES, en el
# rótulo del escrito, y no es quien recurre.
ESCRITO_631 = (
    "Expediente Principal: 950/2024 Relativo al Juicio de AMPARO CIVIL Deducido la Sentencia emitida dentro del "
    "Toca Civil Número 2338/2024, De la Primera Sala Civil Del Tribunal Superior de Justicia Del Estado de "
    "Querétaro, Que se formo, con motivo de la interposición del Recurso de APELACIÓN, hecho valer por LA "
    "DEMANDADA INCIDENTAL, La Unión de Trabajadores de la Construcción, Transportistas, Materialistas y Similares "
    "y Anexos del Estado de Querétaro, C.T.M. En contra de la resolución interlocutoria de fecha 8 de abril de "
    "2024 H. C. JUEZ SEPTIMO DE DISTRITO EN MATERIA DE AMPARO CIVIL, ADMINISTRATIVO Y DEL TRABAJO DE JUICIOS "
    "FEDERALES EN EL ESTADO DE QUERETARO CO'PRESENTE TRATIVA Y DIVE FIDO C.RCUITO. 3. ORG. LIC. NAZARIO TORRES "
    "RAMÍREZ, con la personalidad de Apoderado legal, de la persona moral demandada IMPULSORA DE DESARROLLOS "
    "INMOBILIARIOS V V S.A. DE C.V., hoy TERCERO INTERESADO en el presente Juicio Federal, en razón de ser PARTE "
    "ACTORA INDIDENTAL como está acreditado, en El incidente de Sustitución de Parte Actora, por lo que, ante "
    "Usted; C. Juez con todo respeto se comparece y expone: Que, se viene a interponer EL RECURSO DE REVISION en "
    "contra de la Sentencia Definitiva de fecha 22 veintidós de abril de 2025 dos mil veinticinco, misma que me "
    "fuera notificada en listas, el día 30 treinta de abril de 2025.")
_e631 = _seguro(asyncio.run, ft.leer_escrito(None, ESCRITO_631, "amparo_revision")) \
    if hasattr(ft, "leer_escrito") else AttributeError("leer_escrito")
ok(_d(_e631).get("promovente") == "Impulsora de Desarrollos Inmobiliarios V V S.A. de C.V."
   and _d(_e631).get("caracter") == "tercero" and _d(_e631).get("representante") == "Nazario Torres Ramírez"
   and _d(_e631).get("figura_representante") == "apoderado legal"
   and set(_d(_d(_e631).get("fuentes")).values()) == {"escrito"},
   f"AR 631: el escrito dice que recurre la TERCERA INTERESADA por conducto de su apoderado, sin modelo: {_e631}")
ok(_d(_e631).get("forma_notificacion") == "lista" and _d(_e631).get("notificacion") == "2025-04-30",
   "…y cómo y cuándo se le notificó la recurrida, porque el escrito lo dice en primera persona («me fuera "
   "notificada en listas, el día 30…»)")
_ESCRITOS = {
    "autorizado": ("amparo_revision", "PRESENTE. Juan Pérez López, en mi carácter de autorizado en términos amplios "
                   "del artículo 12 de la Ley de Amparo de la parte quejosa, comparezco para exponer: que vengo a "
                   "interponer recurso de revisión.",
                   {"caracter": "quejoso", "representante": "Juan Pérez López"}, ("promovente",)),
    "propio_derecho": ("amparo_revision", "PRESENTE. MARÍA LÓPEZ RUIZ, por mi propio derecho, en mi carácter de "
                       "tercera interesada en el juicio de amparo 123/2025, ante usted comparezco y expongo.",
                       {"promovente": "María López Ruiz", "caracter": "tercero"}, ("representante",)),
    "delegado": ("amparo_revision", "PRESENTE. Lic. Roberto Gómez Díaz, en mi carácter de delegado de la autoridad "
                 "responsable Gobernador Constitucional del Estado de Querétaro, personalidad que tengo reconocida, "
                 "interpongo recurso de revisión.",
                 {"promovente": "Gobernador Constitucional del Estado de Querétaro", "caracter": "autoridad",
                  "representante": "Roberto Gómez Díaz", "figura_representante": "delegado"}, ()),
    "cargo": ("amparo_revision", "H. JUEZ PRESENTE. Ana Ruiz Gómez, Directora Jurídica de la Secretaría de Gobierno del "
              "Estado, en representación del Gobernador Constitucional del Estado de Querétaro, autoridad responsable "
              "en el juicio de amparo 55/2025, interpongo recurso de revisión.",
              {"promovente": "Gobernador Constitucional del Estado de Querétaro", "caracter": "autoridad",
               "representante": "Ana Ruiz Gómez",
               "figura_representante": "Directora Jurídica de la Secretaría de Gobierno del Estado"}, ()),
    "por_conducto": ("queja", "C. JUEZ TERCERO DE DISTRITO PRESENTE. COMERCIALIZADORA DEL BAJÍO, S.A. DE C.V., por "
                     "conducto de su apoderado legal Pedro Ruiz Gómez, quejosa en el juicio de amparo 1234/2025, "
                     "interpongo recurso de queja contra el auto que desechó la demanda.",
                     {"promovente": "Comercializadora del Bajío, S.A. de C.V.", "caracter": "quejoso",
                      "representante": "Pedro Ruiz Gómez"}, ()),
    "rf_unidad": ("revision_fiscal", "H. TRIBUNAL COLEGIADO PRESENTE. Karen Yadira Meza Ruiz, Administradora "
                  "Desconcentrada Jurídica de Querétaro \"1\", del Servicio de Administración Tributaria, en "
                  "representación del Secretario de Hacienda y Crédito Público, interpongo recurso de revisión fiscal.",
                  {"caracter": "autoridad"}, ("representante",)),
}
for _k, (_t, _txt, _esp, _ausentes) in _ESCRITOS.items():
    _l = _seguro(asyncio.run, ft.leer_escrito(None, _txt, _t)) if hasattr(ft, "leer_escrito") else None
    ok(all(_d(_l).get(c) == v for c, v in _esp.items()) and not any(_d(_l).get(c) for c in _ausentes),
       f"escrito «{_k}»: {_esp} {('sin ' + ', '.join(_ausentes)) if _ausentes else ''} → "
       f"{ {c: _d(_l).get(c) for c in ('promovente', 'caracter', 'representante', 'figura_representante')} }")
ok(hasattr(ft, "leer_escrito") and asyncio.run(ft.leer_escrito(None, DEMANDA_AD, "amparo_directo")) == {"avisos": []},
   "en el amparo directo el escrito es la demanda: nada que leer")
# Sin fórmula: el modelo, anclado. El nombre que no está se tira; el carácter
# sólo se cree si el escrito usa la palabra.
_SIN_FORMULA = ("H. JUEZ PRESENTE. Fraccionadora La Romita, S.A. de C.V., tercera interesada en el juicio de amparo "
                "88/2025, hace valer recurso de revisión contra la sentencia de diez de marzo de dos mil veintiséis.")
cli, comp = cliente_falso({"promovente": "Fraccionadora La Romita, S.A. de C.V.", "caracter": "tercero",
                           "representante": "Pedro Inventado Pérez", "figura_representante": ""})
_m1 = _seguro(asyncio.run, ft.leer_escrito(cli, _SIN_FORMULA, "amparo_revision")) if hasattr(ft, "leer_escrito") else None
ok(_d(_m1).get("promovente") == "Fraccionadora La Romita, S.A. de C.V." and _d(_m1).get("caracter") == "tercero"
   and not _d(_m1).get("representante") and any("Pedro Inventado" in a for a in _d(_m1).get("avisos", []))
   and len(comp.vistos) == 1 and comp.vistos[0]["max_completion_tokens"] <= 600,
   "sin fórmula: UNA llamada corta; lo que está en el papel entra, el representante inventado no")
cli, comp = cliente_falso({"promovente": "Fraccionadora La Romita, S.A. de C.V.", "caracter": "autoridad"})
_m2 = _seguro(asyncio.run, ft.leer_escrito(cli, _SIN_FORMULA, "amparo_revision")) if hasattr(ft, "leer_escrito") else None
ok(_d(_m2).get("caracter") != "autoridad" and any("no usa esa palabra" in a for a in _d(_m2).get("avisos", [])),
   "un carácter que el escrito no dice no entra (el escrito no habla de autoridad responsable)")

# ARMAR con el escrito: la convención «recurrente vacío = recurre la quejosa»
# ya no entra como dato del secretario.
_Q631 = "Unión de Trabajadores de la Construcción, Transportistas, Materialistas y Similares y Anexos del Estado de Querétaro, C.T.M."
E631 = {"tipo_asunto": "amparo_revision", "numero": "631/2025", "quejoso": _Q631, "recurrente": "",
        "presentacion": "2025-05-15", "notificacion": "2025-04-30"}
FP631 = {"recurrente": {"nombre": _Q631, "papel": "quejoso"}, "quejosa": {"nombre": _Q631}}
_f631 = _seguro(ft.armar, E631, ficha_procesal=FP631, escrito=_e631 if isinstance(_e631, dict) else {})
ok(_d(_f631).get("promovente", "").startswith("Impulsora") and _d(_d(_f631).get("fuentes")).get("promovente") == "escrito"
   and _d(_f631).get("caracter") == "tercero" and _d(_f631).get("representante") == "Nazario Torres Ramírez"
   and _d(_f631).get("quejoso") == _Q631,
   "AR 631 por armar: recurre la tercera (fuente «escrito»), por conducto de su apoderado; la quejosa sigue siendo quejosa")
_fn631 = _seguro(ft.armar, dict(E631, notificacion="2025-05-02"), escrito=_e631 if isinstance(_e631, dict) else {})
ok(_d(_fn631).get("notificacion") == "2025-05-02"
   and any("NOTIFICACIÓN NO CUADRA" in a and "escrito" in a for a in _d(_fn631).get("avisos", [])),
   "la notificación del secretario manda; si el escrito dice otra fecha, se avisa")
_ns631 = types.SimpleNamespace(**E631, tramite_escrito=_e631 if isinstance(_e631, dict) else {})
_f631n = _seguro(ft.armar, _ns631)
ok(_d(_f631n).get("caracter") == "tercero", "…también si la lectura viaja en `encargo.tramite_escrito`")
_f631b = ft.armar(dict(E631))
ok(_f631b["promovente"] == _Q631 and _f631b["fuentes"].get("promovente") == "formulario",
   "sin escrito ni auto: la quejosa como RESPALDO, con fuente «formulario» (no «secretario»)")
_f631c = _seguro(ft.armar, dict(E631, recurrente=_Q631), escrito=_e631 if isinstance(_e631, dict) else {})
ok(_d(_f631c).get("promovente") == _Q631 and _d(_f631c).get("caracter") != "tercero"
   and not _d(_f631c).get("representante")
   and any("QUIEN RECURRE NO CUADRA" in a for a in _d(_f631c).get("avisos", [])),
   "el secretario dice otra parte: manda, se avisa, y el carácter y el apoderado del escrito NO se le pegan")

# ── 15.2 · ROBUSTEZ (rev_6) ────────────────────────────────────────────────
ok(not [n for n in dir(ft) if hasattr(getattr(ft, n), "cache_info")],
   "ninguna lru_cache con el texto entero como llave (72 MB retenidos por worker)")
_pp = _seguro(getattr(ft, "Papel", None) or (lambda x: None), ACTO_AD)
ok(hasattr(ft, "Papel") and ft.fecha_en_papel("2025-02-20", _pp) and not ft.fecha_en_papel("2025-02-21", _pp)
   and ft.fechas_del_texto(_pp) == ft.fechas_del_texto(ACTO_AD),
   "un Papel por llamada: las mismas fechas que el texto, calculadas una vez")
_VOC = ("amparo juicio sentencia autoridad responsable quejosa tercero interesado recurso revision agravio concepto "
        "violacion articulo fraccion constitucion federal juzgado distrito tribunal colegiado circuito estado "
        "queretaro civil administrativa materia resolucion acto reclamado demanda presentacion notificacion termino "
        "plazo dias habiles suspension definitiva provisional audiencia constitucional pruebas alegatos ministerio "
        "publico federacion pedimento expediente toca apelacion primera instancia segunda sala superior justicia "
        "magistrado ponente secretario").split()
_r3 = _rnd.Random(3)
_trozos3, _n3 = [], 0
while _n3 < 3000:
    _fr3 = " ".join(_r3.choice(_VOC) for _ in range(_r3.randint(8, 25))).capitalize() + ". "
    _trozos3.append(_fr3)
    _n3 += len(_fr3)
_PROSA = "".join(_trozos3)
_pal = _PROSA.split()
_span = _pal[150:210]
_span[10], _span[40] = "xxxx", "yyyy"          # dos palabras que el modelo «corrigió»
_t0 = _time.perf_counter()
_v, _modo = ft.anclar(" ".join(_span), _PROSA)
_dt = _time.perf_counter() - _t0
ok(_modo == "parecido" and ft._plano(_v) == ft._plano(" ".join(_pal[150:210])) and _dt < 1.0,
   f"anclar un acto de 60 palabras con dos distintas en prosa repetitiva: el tramo exacto, en {_dt:.2f} s (antes 5 s)")
_RELLENO = ("Visto el estado procesal, el veinte de enero de dos mil veinticinco se tuvo por recibido el oficio, y el "
            "tres de febrero de dos mil veinticinco se acordó lo conducente. ") * 450
_t0 = _time.perf_counter()
_ag = ft.leer_auto(AUTO_ADMISION_AD + "\n" + _RELLENO)
_dt = _time.perf_counter() - _t0
ok(_d(_ag.get("admision")).get("fecha") == "2025-01-20" and any("MUY LARGO" in a for a in _ag["avisos"]) and _dt < 2.0,
   f"un «auto» de {len(AUTO_ADMISION_AD + _RELLENO):,} caracteres: se lee el principio, se avisa, {_dt:.2f} s")
_HILOS: dict = {}
_orig_anc, _orig_det = ft._modelo_anclado, ft._lectura_determinista


def _espia_anc(*a, **k):
    _HILOS["anclado"] = _th.get_ident()
    return _orig_anc(*a, **k)


def _espia_det(*a, **k):
    _HILOS["determinista"] = _th.get_ident()
    return _orig_det(*a, **k)


ft._modelo_anclado, ft._lectura_determinista = _espia_anc, _espia_det
try:
    cli, comp = cliente_falso({"clase": "sentencia", "fecha": "2025-02-20", "toca": "4520/2024"})
    asyncio.run(ft.leer_acto(cli, ACTO_AD, "amparo_directo", "274/2025", demanda=DEMANDA_AD))
finally:
    ft._modelo_anclado, ft._lectura_determinista = _orig_anc, _orig_det
ok(_HILOS.get("anclado") not in (None, _th.get_ident()) and _HILOS.get("determinista") not in (None, _th.get_ident()),
   "leer_acto: la lectura sin modelo y el anclaje corren en un hilo, no en el bucle de eventos")


class _Cuelga:
    async def create(self, **kw):
        await asyncio.sleep(3600)


_cli_c = types.SimpleNamespace(chat=types.SimpleNamespace(completions=_Cuelga()))
_tope0 = getattr(ft, "TOPE_SEGUNDOS_MODELO", None)
ft.TOPE_SEGUNDOS_MODELO = 0.3


async def _con_tope():
    return await asyncio.wait_for(ft.leer_acto(_cli_c, ACTO_RF, "revision_fiscal", "91/2025"), timeout=5)


try:
    _rc = asyncio.run(_con_tope())
except asyncio.TimeoutError:
    _rc = None
finally:
    if _tope0 is None:
        del ft.TOPE_SEGUNDOS_MODELO
    else:
        ft.TOPE_SEGUNDOS_MODELO = _tope0
ok(_rc is not None and _rc["acto"].get("fecha") == "2025-09-22" and any("NO CONTESTÓ" in a for a in _rc["avisos"]),
   "el modelo colgado: leer_acto sigue con lo determinista al vencer su tope (antes, 180-600 s esperando)")
_XML_MALO = _re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
_dc = ft.de_formulario({"fecha_turno": "2026-02-12", "ponente_turno": "Ana\u0000Pérez\u0007 Ruiz"})
ok(_d(_dc.get("turno")).get("ponente") and not _XML_MALO.search(_dc["turno"]["ponente"]),
   f"un carácter de control en tramite_json no llega al .docx: «{_d(_dc.get('turno')).get('ponente')}»")
_va, _ = ft.anclar("Segunda Sala Civil del Tribunal Superior de Justicia del Estado de Queretaro, Qro",
                   DEMANDA_AD.replace("Civil", "Ci\u0000vil"))
ok(_va and not _XML_MALO.search(_va), "…ni el que trae el texto del PDF en el tramo anclado")
_dl = ft.de_formulario({"ponente_turno": "Ana Pérez " * 600, "toca": "1/" + "9" * 60,
                        "fundamento_surtimiento": "el artículo " + "1" * 500 + " del Código Civil"})
ok(not _d(_dl.get("turno")).get("ponente") and not _d(_dl.get("acto")).get("toca")
   and not _dl.get("fundamento_surtimiento") and sum("DEMASIADO LARGO" in a for a in _dl["avisos"]) == 3,
   "topes por campo (nombres 300, números 40, fundamento 400): lo que excede se ignora y se dice")
_dn = ft.de_formulario({"fecha_turno": "2026-02-12", "ponente_turno": {"nombre": "Ana"}, "toca": ["374/2025"]})
ok(not _d(_dn.get("turno")).get("ponente") and not _d(_dn.get("acto")).get("toca")
   and sum("NO TRAE TEXTO" in a for a in _dn["avisos"]) == 2,
   "un valor que no es texto no se escribe con su repr («{'nombre': 'Ana'}»): se ignora y se dice")
_dt3 = _seguro(ft.de_formulario, {"terceros": [{"a": 1}, 5, "Gabriel Reyes Álamo"]}, extra=True)
ok(_d(_dt3).get("terceros") == ["Gabriel Reyes Álamo"], "con `extra`, las listas sólo con sus cadenas")
_dh = ft.de_formulario({"presentacion": "2026-02-05", "sentido": "concedió todo", "fecha_admision": "2026-02-10"})
ok("presentacion" not in _dh and not _d(_dh.get("acto")).get("sentido") and _dh["admision"]["fecha"] == "2026-02-10"
   and any("SE IGNORÓ" in a and "presentacion" in a for a in _dh["avisos"]),
   "por HTTP sólo las claves del contrato: «presentacion» y «sentido» a mano no mandan sobre el formulario principal")
_dx2 = _seguro(ft.de_formulario, {"presentacion": "2026-02-05"}, extra=True)
ok(_d(_dx2).get("presentacion") == "2026-02-05", "…y con `extra=True` (uso interno) sí entran")
# Al retomar: lo leído vuelve como del secretario; si el acto nuevo nombra OTRO juzgado, se dice.
E_RET = {"tipo_asunto": "amparo_revision", "numero": "300/2026", "presentacion": "2026-02-03", "notificacion": "2026-01-27"}
_fr1 = ft.armar(E_RET, acto={"acto": {"fecha": "2026-01-15", "expediente": "905/2025",
                                      "organo": "Juzgado Primero de Distrito en el Estado de Querétaro"}})
_dec_ret = ft.de_formulario({k: v for k, v in ft.a_formulario(_fr1).items() if v})
_fr2 = ft.armar(E_RET, declarado=_dec_ret,
                acto={"acto": {"fecha": "2026-01-20", "expediente": "77/2025",
                               "organo": "Juzgado Segundo de Distrito en el Estado de Querétaro"}})
ok(any("NO CUADRA" in a and "Juzgado Segundo" in a for a in _fr2["avisos"]),
   "acto.organo en la discrepancia: el juzgado ya no cambia en silencio al retomar")
_fr3 = ft.armar(E_RET, declarado=_dec_ret,
                acto={"acto": {"organo": "JUZGADO PRIMERO DE DISTRITO EN EL ESTADO DE QUERÉTARO"}})
ok(not any("NO CUADRA" in a and "ÓRGANO" in a for a in _fr3["avisos"]), "…el mismo juzgado en versales no es discrepancia")

# ── 15.3 · LA CRONOLOGÍA Y LA COHERENCIA DEL AMPARO EN REVISIÓN (rev_3) ────
_JZ = "Juzgado Cuarto de Distrito en el Estado de Querétaro"
_v1 = {"tipo": "amparo_revision", "demanda": {"fecha": "2025-06-01"}, "audiencia": "2025-05-20",
       "acto": {"fecha": "2025-07-15", "organo": _JZ}, "presentacion": "2025-07-30"}
ft.validar(_v1)
ok("demanda.fecha > audiencia" in _v1["fechas_imposibles"], "la demanda después de la audiencia: imposible")
_v2 = {"tipo": "amparo_revision", "audiencia": "2025-07-15", "presentacion": "2025-07-01", "acto": {"organo": _JZ}}
ft.validar(_v2)
ok("audiencia > presentacion" in _v2["fechas_imposibles"], "el recurso antes de la audiencia (AR 448): imposible")
_v3 = {"tipo": "amparo_revision", "demanda": {"fecha": "2025-05-13"}, "presentacion": "2025-05-13", "acto": {"organo": _JZ}}
_a3 = ft.validar(_v3)
ok("demanda.fecha >= presentacion" in _v3["fechas_imposibles"] and any("mismo día" in a for a in _a3),
   "el recurso el mismo día que la demanda: se tomó la fecha de la demanda")
_a4 = ft.validar({"tipo": "amparo_revision", "clase_recurrida": "auto_sobreseimiento", "audiencia": "2024-11-29",
                  "acto": {"fecha": "2025-02-17", "organo": _JZ}})
ok(any("CONSTA AUDIENCIA" in a for a in _a4), "auto de sobreseimiento con audiencia anterior (AR 239): aviso del inciso e)")
_a5 = ft.validar({"tipo": "amparo_revision", "clase_recurrida": "sentencia",
                  "acto": {"incidente": True, "clase": "sentencia", "organo": _JZ}})
ok(any("INCIDENTE DE SUSPENSIÓN" in a for a in _a5), "dictada en el incidente pero llamada «sentencia»: aviso")
_SALA_F = "Sala Familiar del Tribunal Superior de Justicia del Estado de Querétaro"
_fo1 = ft.armar({"tipo_asunto": "amparo_revision", "numero": "307/2024"},
                acto={"acto": {"organo": _SALA_F, "fecha": "2023-12-06"},
                      "demanda": {"autoridades": [_SALA_F, "Juez Octavo de Primera Instancia Familiar de Querétaro"]}})
ok(_fo1["acto"]["organo"] == "" and any("NO PUEDE SER" in a for a in _fo1["avisos"]),
   "AR 307: la Sala responsable no dictó la sentencia de amparo: fuera de acto.organo (hueco), con aviso")
_JZ7 = "Juzgado Séptimo de Distrito en Materia de Amparo Civil, Administrativo y de Trabajo y de Juicios Federales en el Estado de Querétaro"
_fo2 = ft.armar({"tipo_asunto": "amparo_revision", "numero": "222/2025"},
                acto={"acto": {"organo": _JZ7, "fecha": "2025-06-10"},
                      "demanda": {"autoridades": [_JZ7, "Juzgado Sexto de Primera Instancia Civil de Querétaro"]}})
ok(_fo2["acto"]["organo"] == _JZ7 and _JZ7 not in _fo2["demanda"]["autoridades"]
   and any("FIGURABA ENTRE LAS AUTORIDADES" in a for a in _fo2["avisos"]),
   "AR 222: el juzgado de distrito entre las responsables de su propio juicio: sale de esa lista, con aviso")
_fo3 = ft.armar({"tipo_asunto": "amparo_revision", "numero": "307/2024"},
                declarado=ft.de_formulario({"organo_acto": _SALA_F}, "amparo_revision"),
                acto={"demanda": {"autoridades": [_SALA_F]}})
ok(_fo3["acto"]["organo"] == _SALA_F and any("ES TAMBIÉN UNA DE LAS" in a for a in _fo3["avisos"]),
   "lo que tecleó el secretario no se quita: se avisa")
_a6 = ft.validar({"tipo": "amparo_revision", "acto": {"organo": "Juez Octavo de Primera Instancia Familiar de Querétaro"}})
ok(any("NO TIENE FORMA DE ÓRGANO DE AMPARO" in a for a in _a6), "un juez de primera instancia como juzgado del amparo: aviso")
_a7 = ft.validar({"tipo": "amparo_revision", "acto": {"fecha": "2023-12-06", "organo": _JZ},
                  "demanda": {"actos": ["La resolución de seis de diciembre de dos mil veintitrés, dictada en el toca 12/2023."]}})
ok(any("ESTÁ ESCRITA EN EL ACTO" in a for a in _a7), "la fecha de la recurrida es la del acto reclamado: aviso")
_fr_rep = ft.armar({"tipo_asunto": "amparo_revision", "numero": "307/2024"},
                   auto={"promovente": "José Pérez Ruiz, por propio derecho y en representación de su menor hijo",
                         "representante": "José Pérez Ruiz", "figura_representante": "representante"})
ok(_fr_rep["representante"] == "" and any("ES LA MISMA PARTE" in a for a in _fr_rep["avisos"]),
   "el representante que es la misma parte que recurre se quita (no «X, por conducto de X»)")
_fr_rol = ft.armar({"tipo_asunto": "amparo_revision", "numero": "208/2025",
                    "recurrente": "GOBERNADOR DEL ESTADO DE QUERÉTARO (AUTORIDAD RESPONSABLE)"})
ok(_fr_rol["promovente"] == "GOBERNADOR DEL ESTADO DE QUERÉTARO" and _fr_rol["caracter"] == "autoridad",
   "AR 208: la etiqueta de rol sale del nombre y dice el carácter")

# ── 15.4 · EL PRECEPTO DEL SURTIMIENTO (rev_1) ─────────────────────────────
ok(ft.fundamento_surtimiento_de("artículo 17 del mencionado ordenamiento") == ""
   and ft.fundamento_surtimiento_de("el artículo 126") == "",
   "AD 274: una remisión sin ley, o un artículo sin ley, no funda nada")
ok(_seguro(ft.fundamento_surtimiento_de, "el artículo 31, fracción II, de la Ley de Amparo", "amparo_directo") == ""
   and _seguro(ft.fundamento_surtimiento_de, "el artículo 31, fracción II, de la Ley de Amparo", "amparo_revision")
   == "el artículo 31, fracción II, de la Ley de Amparo",
   "los arts. 17, 18, 19 y 31 de la Ley de Amparo no rigen el surtimiento del acto en el AD (en los recursos, sí)")
_df = ft.de_formulario({"fundamento_surtimiento": "artículo 18 de la Ley de Amparo"}, "amparo_directo")
ok(not _df.get("fundamento_surtimiento") and any("Ley de Amparo" in a and "se ignoró" in a for a in _df["avisos"]),
   "el formulario del AD con el art. 18 de la Ley de Amparo: se ignora y se dice por qué")
_fa_ad = ft.armar(E, acto=x, declarado=ft.de_formulario(json.dumps({"fundamento_surtimiento":
                                                                    "el artículo 19 de la Ley de Amparo"})))
ok(_fa_ad["fundamento_surtimiento"] == "" and any("se quitó (fundamento_surtimiento)" in a for a in _fa_ad["avisos"]),
   "por HTTP llega sin tipo; armar, que sabe que es AD, lo quita y avisa")
ok(ft.armar(E, acto=x, declarado={"fundamento_surtimiento": _FUND})["fundamento_surtimiento"] == _FUND,
   "el precepto de la ley del acto sigue entrando")

# ── 15.5 · EL ESQUEMA: NEGATIVA FICTA Y VÍA (rev_5, rev_1) ─────────────────
ok(_d(r.get("resolucion_impugnada")).get("clase") == "expresa", "la resolución con oficio es «expresa»")
ACTO_RF_NF = ("SALA REGIONAL DEL CENTRO II DEL TRIBUNAL FEDERAL DE JUSTICIA ADMINISTRATIVA EXPEDIENTE: 812/25-09-01-3 "
              "Querétaro, Querétaro, a doce de agosto de dos mil veinticinco. VISTOS los autos del juicio contencioso "
              "administrativo número 812/25-09-01-3, promovido por Transportes Ejemplo del Centro, S.A. de C.V., en "
              "contra de la resolución negativa ficta recaída a su solicitud de devolución del impuesto al valor "
              "agregado del mes de marzo de dos mil veinticuatro; se resuelve. Por lo expuesto, se R E S U E L V E: "
              "I. Se declara la nulidad lisa y llana de la resolución impugnada. NOTIFÍQUESE.")
_nf = asyncio.run(ft.leer_acto(None, ACTO_RF_NF, "revision_fiscal", "91/2025"))
_ri = _d(_nf.get("resolucion_impugnada"))
ok(_ri.get("clase") == "negativa_ficta" and _ri.get("materia", "").startswith("su solicitud de devolución")
   and not _ri.get("oficio"), f"la negativa ficta: su clase y lo que se pidió, sin oficio ni fecha: {_ri}")
_fnf = ft.armar({"tipo_asunto": "revision_fiscal", "numero": "91/2025"}, acto=_nf)
ok(set(_fnf["resolucion_impugnada"]) == {"clase", "oficio", "fecha", "autoridad", "materia"}
   and _fnf["resolucion_impugnada"]["clase"] == "negativa_ficta", "armar: la sección con sus cinco claves")
cli, comp = cliente_falso({"resolucion_clase": "negativa_ficta", "resolucion_materia": "su petición de pensión"})
_nf2 = asyncio.run(ft.leer_acto(cli, ACTO_RF, "revision_fiscal", "91/2025"))
ok(_d(_nf2.get("resolucion_impugnada")).get("clase") == "expresa"
   and any("NEGATIVA FICTA" in a for a in _nf2["avisos"]),
   "el modelo dice negativa ficta y la sentencia no: no se toma")
ok(_seguro(lambda: ft.via_de("tribunal")) == "tribunal_colegiado" and ft.via_de("Este Tribunal Colegiado") ==
   "tribunal_colegiado" and ft.via_de("Tribunal Superior de Justicia del Estado") == "" and ft.via_de("Portal") ==
   "electronica", "la vía: «tribunal» es ESTE Tribunal Colegiado; el Tribunal Superior no se adivina")
_fv = ft.armar({"tipo_asunto": "amparo_directo", "numero": "274/2025"}, acto=x,
               declarado={"via_presentacion": "tribunal"})
ok(_fv["via_presentacion"] == "tribunal_colegiado" and _fv["fuentes"].get("via_presentacion") == "secretario",
   "armar: «tribunal» llega como «tribunal_colegiado», con su fuente")
_fv2 = ft.armar({"tipo_asunto": "amparo_directo", "numero": "274/2025"}, acto=x,
                auto={"via_presentacion": "Oficialía del Tribunal Superior"})
ok(_fv2["via_presentacion"] == "" and any("NO SE RECONOCE" in a for a in _fv2["avisos"]),
   "una vía que no es del esquema no entra")

# ── 15.6 · LA SEDE: UNA SOLA REGLA (rev_2) ─────────────────────────────────
import tipos_asunto as _ta_s
_SEDES = [("Décimo Tribunal Colegiado en Materia Civil del Primer Circuito", "Cd. de México"),
          ("Quinto Tribunal Colegiado en Materia Civil del Primer Circuito", ""),
          ("Tercer Tribunal Colegiado en Materias Administrativa y Civil del Vigésimo Segundo Circuito", "Querétaro, Querétaro"),
          ("Primer Tribunal Colegiado de Circuito del Centro Auxiliar de la Primera Región", "Ciudad de México"),
          ("", "México, D.F."), ("Décimo Tribunal Colegiado en Materia Civil del Primer Circuito", "México")]
ok(all(ft.sede_de(t, c)["cdmx"] == bool(_ta_s.es_cdmx(t, c)) for t, c in _SEDES),
   "sede_de y tipos_asunto.es_cdmx dicen lo mismo en todos los casos (una sola regla)")
_es0 = _ta_s.es_cdmx
_ta_s.es_cdmx = lambda *a, **k: True
try:
    _delega = ft.sede_de("Tercer Tribunal Colegiado del Vigésimo Segundo Circuito", "Querétaro, Querétaro")["cdmx"]
finally:
    _ta_s.es_cdmx = _es0
ok(_delega is True, "sede_de pregunta a tipos_asunto.es_cdmx (no decide por su lado)")
ok(ft.sede_de("Décimo Tribunal Colegiado en Materia Civil del Primer Circuito", "Cd. de México")["cdmx"] is True,
   "Primer Circuito con «Cd. de México»: Ciudad de México (regla de tipos_asunto, dueño B)")

# ── 15.7 · fase_admision.leer: EL TRÁMITE SE LEE EN UN HILO ────────────────
_HL: dict = {}
_orig_la = ft.leer_auto


def _espia_la(*a, **k):
    _HL["leer_auto"] = _th.get_ident()
    return _orig_la(*a, **k)


ft.leer_auto = _espia_la
try:
    cli, comp = cliente_falso({"tipo_asunto": "", "quejoso": "", "responsable_ordenadora": ""})
    _fic3 = asyncio.run(fa.leer(cli, AUTO_ADMISION_AD, con_tramite=True))
finally:
    ft.leer_auto = _orig_la
ok(_HL.get("leer_auto") not in (None, _th.get_ident()) and _fic3["tramite"]["fecha_admision"] == "2025-01-20",
   "fase_admision.leer: leer_auto corre en un hilo y el trámite sale igual")

# ═══════════════════════════════════════════════════════════════════════════
print("\n16 · CUARTA RONDA (3-oct-2026): CORREO, REGISTRO, CARGO DEL PONENTE, DATOS CORTADOS, UN AVISO POR DATO")


def _f(nombre):
    """La función pública de la cuarta ronda, o una que lanza (cuenta como FALLA)."""
    g = getattr(ft, nombre, None)
    if callable(g):
        return g

    def _no(*a, **k):
        raise AttributeError(f"ficha_tramite.{nombre} no existe")
    return _no


def _txt(avisos):
    return " ".join(str(a) for a in (avisos or []))


# ── 16.1 · E6: EL CORREO, UNA SOLA REGLA (RF 2/2025) ───────────────────────
# Vía «responsable» —el oficio lo recibe la Sala— y depósito el 24-oct-2024: el
# resultando narraba el depósito y el cómputo lo descartaba.
_via_postal = _f("via_postal")
_RF2 = {"tipo": "revision_fiscal", "via_presentacion": "responsable", "deposito_postal": "2024-10-24",
        "presentacion": "2024-11-06", "acto": {"fecha": "2024-09-30"}}
ok(_seguro(_via_postal, _RF2) is True, "RF 2: depósito anterior a la recepción ⇒ por correo, aunque la vía diga «responsable»")
ok(_seguro(_via_postal, dict(_RF2, deposito_postal="2024-11-06")) is True, "…el depósito el mismo día de la recepción, también")
ok(_seguro(_via_postal, dict(_RF2, deposito_postal="2024-11-08")) is False,
   "…un depósito POSTERIOR a la recepción no hace postal la vía «responsable» (es imposible: lo acusa validar)")
ok(_seguro(_via_postal, {"via_presentacion": "postal"}) is True, "vía «postal» sin fecha de depósito: por correo")
ok(_seguro(_via_postal, {"via_presentacion": "responsable"}) is False and _seguro(_via_postal, None) is False,
   "sin depósito ni vía postal, no; con None, no (y no lanza)")
import datetime as _dtm
ok(_seguro(_via_postal, {"deposito_postal": "2024-10-24"}, _dtm.date(2024, 11, 6)) is True
   and _seguro(_via_postal, {"deposito_postal": "2024-10-24"}, "2024-10-01") is False,
   "la presentación del Encargo (fecha o ISO) manda sobre la de la ficha")
import copy as _cp
_a_rf2 = ft.validar(_cp.deepcopy(_RF2))
ok(sum("DEPÓSITO POSTAL" in a and "responsable" in a for a in _a_rf2) == 1,
   f"validar avisa UNA vez del choque depósito/vía: {[a[:70] for a in _a_rf2]}")
ok(not any("DEPÓSITO POSTAL" in a and "VÍA" in a for a in ft.validar(dict(_cp.deepcopy(_RF2), via_presentacion="postal"))),
   "con la vía «postal», ningún aviso del choque")

# ── 16.2 · E7: EL AUTO QUE REGISTRA NO ES EL QUE ADMITE (RF 7/2025, Q 335/2025) ──
ok(len(ft.CLAVES_FORMULARIO) == 22 and ft.CLAVES_FORMULARIO[-2:] == ("fecha_registro", "relacionados"),
   "CLAVES_FORMULARIO: 22, «fecha_registro» y (sexta ronda) «relacionados» al final")
_dr = ft.de_formulario({"fecha_registro": "10/02/2025", "fecha_admision": "15/05/2025"})
ok(_d(_dr.get("registro")).get("fecha") == "2025-02-10" and _d(_dr.get("admision")).get("fecha") == "2025-05-15"
   and _dr["fuentes"].get("registro.fecha") == "secretario", "de_formulario: «fecha_registro» va a registro.fecha (secretario)")
_fr_ = ft.a_formulario(_dr)
ok(_fr_.get("fecha_registro") == "2025-02-10" and ft.a_formulario(ft.de_formulario(_fr_)) == _fr_,
   "a_formulario: «fecha_registro» de vuelta, ida y vuelta sin pérdida")
_ar7 = ft.armar({"tipo_asunto": "revision_fiscal", "numero": "7/2025"}, declarado=json.dumps(
    {"fecha_registro": "2025-02-10", "fecha_admision": "2025-05-15"}))
ok(_ar7.get("registro") == {"fecha": "2025-02-10"} and _ar7["admision"]["fecha"] == "2025-05-15"
   and _ar7["fuentes"].get("registro.fecha") == "secretario", f"armar: registro y admisión, cada uno el suyo: {_ar7.get('registro')}")
ok(ft.armar({"tipo_asunto": "amparo_directo", "numero": "43/2025"}).get("registro") == {},
   "sin registro, la sección opcional va vacía ({})")
_AUTO_RF7 = ("REVISIÓN FISCAL 7/2025\nQuerétaro, Querétaro, a diez de febrero de dos mil veinticinco.\nVisto el oficio "
             "de agravios de la autoridad demandada, radíquese el recurso y regístrese con el número 7/2025. Ahora "
             "bien, este Tribunal Colegiado se declara legalmente incompetente para conocer del asunto y ordena "
             "remitirlo. Notifíquese.\nQuerétaro, Querétaro, a quince de mayo de dos mil veinticinco.\nVisto lo "
             "resuelto por el Tribunal Colegiado competente, se admite el recurso de revisión fiscal. Notifíquese.")
_l7 = ft.leer_auto(_AUTO_RF7, "revision_fiscal")
ok(_d(_l7.get("registro")).get("fecha") == "2025-02-10" and _d(_l7.get("admision")).get("fecha") == "2025-05-15",
   f"RF 7: el auto que radica (y declina) es el registro; el que admite, la admisión: {_l7.get('registro')} {_l7.get('admision')}")
_l7b = ft.leer_auto(_AUTO_RF7.split("\nQuerétaro, Querétaro, a quince")[0], "revision_fiscal")
ok(_d(_l7b.get("registro")).get("fecha") == "2025-02-10" and not _l7b.get("admision"),
   "sólo el auto que radica: registro, y la admisión NO se rellena con su fecha")
ok(not ft.leer_auto(AUTO_ADMISION_AD).get("registro"), "AD: el mismo auto registra y admite: sólo la admisión")
_lno = ft.leer_auto("Querétaro, Querétaro, a dos de marzo de dos mil veintiséis. Fórmese el expediente número "
                    "10/2026. Por extemporáneo, no se admite el recurso de queja. Notifíquese.", "queja")
ok(not _lno.get("admision") and _d(_lno.get("registro")).get("fecha") == "2026-03-02",
   "«no se admite» no es admitir: el auto sólo registró")
_lpa = ft.leer_auto("Querétaro, Querétaro, a dos de marzo de dos mil veintiséis. Fórmese el expediente número "
                    "10/2026. Es procedente admitir el recurso de queja.", "queja")
_lpn = ft.leer_auto("Querétaro, Querétaro, a dos de marzo de dos mil veintiséis. Fórmese el expediente número "
                    "10/2026. No es procedente admitir el recurso de queja.", "queja")
ok(_d(_lpa.get("admision")).get("fecha") == "2026-03-02" and not _lpa.get("registro")
   and not _lpn.get("admision") and _d(_lpn.get("registro")).get("fecha") == "2026-03-02",
   "«es procedente admitir» admite; «no es procedente admitir», no")
ok(ft.armar({"tipo_asunto": "queja"}, auto=[{"registro": {"fecha": "2026-03-02"}},
                                             {"admision": {"fecha": "2026-03-02"}}]).get("registro") == {},
   "armar: un registro leído con la fecha de la admisión es el mismo auto (fuera)")
_lpr = ft.leer_auto("Querétaro, Querétaro, a tres de noviembre de dos mil veinticinco. Visto que por auto de "
                    "Presidencia de catorce de octubre de dos mil veinticinco se requirió a la autoridad responsable su "
                    "informe con justificación, se tiene a la responsable rindiendo su informe con justificación; se "
                    "admite el recurso de queja.",
                    "queja")
ok(_d(_lpr.get("registro")).get("fecha") == "2025-10-14" and _d(_lpr.get("admision")).get("fecha") == "2025-11-03"
   and _d(_lpr.get("informe_101")).get("fecha") == "2025-11-03",
   f"Q 335: «por auto de Presidencia de… se requirió» es el registro; el que tiene por rendido y admite, la admisión: "
   f"{_lpr.get('registro')} {_lpr.get('admision')}")
_vr = {"tipo": "queja", "registro": {"fecha": "2025-11-05"}, "admision": {"fecha": "2025-11-03"}}
_avr = ft.validar(_vr)
ok("admision.fecha < registro.fecha" in _vr["fechas_imposibles"] and any("anterior al auto que formó" in a for a in _avr),
   f"validar: la admisión antes del registro es imposible (y dice «anterior al»): {[a[:90] for a in _avr]}")
_vi = {"tipo": "queja", "registro": {"fecha": "2025-10-14"}, "informe_101": {"fecha": "2025-10-01"}}
ft.validar(_vi)
ok("informe_101.fecha < registro.fecha" in _vi["fechas_imposibles"], "validar: el informe rendido antes del auto que lo pidió")
_vo = {"tipo": "queja", "registro": {"fecha": "2026-03-03"}, "admision": {"fecha": "2026-03-10"},
       "informe_101": {"fecha": "2026-03-09"}}
ok(not ft.validar(_vo), "un informe rendido después del registro y antes del auto que admite NO es imposible (antes se "
                        "comparaba con la admisión)")
cli, comp = cliente_falso({"fecha_admision": "2026-03-09", "fecha_registro": "2026-03-03"})
_mr = asyncio.run(ft.leer_auto_modelo(cli, AUTO_Q_FR2, "queja"))
ok(_d(_mr.get("registro")).get("fecha") == "2026-03-03" and "fecha_registro" in comp.vistos[0]["messages"][0]["content"],
   "con modelo: el prompt pide «fecha_registro» y, anclada al papel, entra")
cli, comp = cliente_falso({"fecha_admision": "2025-01-20", "fecha_registro": "2025-01-20"})
ok(not asyncio.run(ft.leer_auto_modelo(cli, AUTO_ADMISION_AD, "amparo_directo")).get("registro"),
   "con modelo: un «registro» con la fecha de la admisión es el mismo auto (no entra)")

# ── 16.3 · E3: EL CARGO DEL PONENTE VIAJA EN LA FICHA ───────────────────────
_lr = ft.leer_auto("Querétaro, Querétaro, a dos de junio de dos mil veinticinco. Visto el oficio de la Comisión de "
                   "Adscripción, retúrnense los autos a la ponencia a cargo de la magistrada Jenica Campos Juárez. "
                   "Notifíquese.")
ok(_d(_lr.get("returno")) == {"fecha": "2025-06-02", "ponente": "Jenica Campos Juárez", "titulo": "Magistrada"},
   f"«a la ponencia a cargo de la magistrada…» (minúscula): el cargo se lee: {_lr.get('returno')}")
_lt = ft.leer_auto("Querétaro, Querétaro, a tres de marzo de dos mil veinticinco. Túrnense los autos a la ponencia de "
                   "la licenciada Bertha Martínez Vega, secretaria en funciones de Magistrada, para su resolución.")
ok(_d(_lt.get("turno")).get("ponente") == "Bertha Martínez Vega"
   and _d(_lt.get("turno")).get("titulo") == "Secretaria en funciones de Magistrada",
   f"AR 239: la secretaria en funciones, sin el tratamiento en el nombre y con su cargo: {_lt.get('turno')}")
_pp = _f("partir_ponente")
_ct = _f("con_titulo")
ok(_seguro(_pp, "Magistrada Jenica Campos Juárez") == ("Jenica Campos Juárez", "Magistrada")
   and _seguro(_pp, "Bertha Martínez Vega, secretaria en funciones de Magistrada")
   == ("Bertha Martínez Vega", "Secretaria en funciones de Magistrada")
   and _seguro(_pp, "Jenica Campos Juárez") == ("Jenica Campos Juárez", ""),
   "partir_ponente: el cargo escrito, aparte; sin cargo, nada (el nombre de pila no dice el género)")
ok(_seguro(_ct, "Jenica Campos Juárez", "Magistrada") == "Magistrada Jenica Campos Juárez"
   and _seguro(_ct, "Bertha Martínez Vega", "Secretaria en funciones de Magistrada")
   == "Bertha Martínez Vega, secretaria en funciones de Magistrada",
   "con_titulo: lo inverso, para el formulario")
_dp = ft.de_formulario({"ponente_returno": "Magistrada Jenica Campos Juárez", "fecha_returno": "2025-06-02"})
ok(_d(_dp.get("returno")) == {"ponente": "Jenica Campos Juárez", "titulo": "Magistrada", "fecha": "2025-06-02"}
   and _dp["fuentes"].get("returno.titulo") == "secretario",
   f"de_formulario: «Magistrada X» → ponente y cargo, como los lee el auto: {_dp.get('returno')}")
ok(ft.a_formulario(_dp)["ponente_returno"] == "Magistrada Jenica Campos Juárez"
   and ft.a_formulario(ft.de_formulario(ft.a_formulario(_dp))) == ft.a_formulario(_dp),
   "a_formulario: el cargo vuelve con el nombre (ida y vuelta sin pérdida)")
ok(ft.a_formulario(ft.leer_auto(AUTO_TURNO_AD))["ponente_turno"] == "Magistrado J. Guadalupe Tafoya Hernández",
   "lo leído del auto llega a la pantalla con su cargo")
cli, comp = cliente_falso({"tipo_asunto": "", "quejoso": "", "responsable_ordenadora": ""})
_fic4 = asyncio.run(fa.leer(cli, AUTO_TURNO_AD, con_tramite=True))
ok(_fic4["tramite"]["ponente_turno"] == "Magistrado J. Guadalupe Tafoya Hernández"
   and "fecha_registro" in _fic4["tramite"],
   "fase_admision.leer: el trámite propuesto trae el cargo del ponente y las 22 claves")
_Eret = {"tipo_asunto": "amparo_directo", "numero": "128/2025"}
_mismo = ft.armar(_Eret, auto=_lr, declarado=json.dumps({"ponente_returno": "Jenica Campos Juárez"}))
ok(_d(_mismo.get("returno")).get("titulo") == "Magistrada" and _mismo["fuentes"].get("returno.titulo") == "auto",
   "armar: el secretario teclea el nombre sin cargo y el auto lo dice de ESA persona: el cargo se queda")
_otro = ft.armar(_Eret, auto=_lr, declarado=json.dumps({"ponente_returno": "Luis Armando Pérez Topete"}))
ok(not _d(_otro.get("returno")).get("titulo"),
   "armar: si el secretario tecleó a OTRA persona, el cargo del auto no se le pega")
cli, comp = cliente_falso({"fecha_turno": "2025-02-21", "ponente_turno": "J. Guadalupe Tafoya Hernández",
                           "titulo_turno": "Magistrada"})
_mt = asyncio.run(ft.leer_auto_modelo(cli, AUTO_TURNO_AD, "amparo_directo"))
ok(not _d(_mt.get("turno")).get("titulo") and any("cargo «Magistrada»" in a for a in _mt["avisos"]),
   "con modelo: «Magistrada» cuando el auto dice «Magistrado» no entra (0.9 de parecido cambiaría el género)")

# ── 16.4 · E8: LO QUE NO SE PUEDE FIRMAR COMO LLEGA ─────────────────────────
_tr = _f("truncado")
ok(_seguro(_tr, "Titular de la Unidad Jurídica…") is True and _seguro(_tr, "Jefa de la Unidad... de la Delegación") is True
   and _seguro(_tr, "COMERCIALIZADORA DEL BAJÍO, S.A. DE C.V...") is False and _seguro(_tr, "Juan Pérez") is False,
   "truncado: «…» y «...» (al final o en medio); la sigla con el punto de la frase, no")
_ec = _f("es_cargo")
ok(all(_seguro(_ec, x) is True for x in ("Titular de la Unidad de Asuntos Jurídicos de la citada delegación",
                                           "Subdirector de lo Contencioso", "Director Jurídico",
                                           "la Jefa de la Unidad Jurídica"))
   and all(_seguro(_ec, x) is False for x in ("Juan Pérez Ruiz", "Delegado Juan Pérez Ruiz", "Karen Yadira Meza Ruiz")),
   "es_cargo: el cargo y la unidad sí; una persona (o una figura con su nombre) no")
ok(_seguro(_f("partir_representante"), "Ana Ruiz Pérez, Jefa de la Unidad Jurídica")
   == ("Ana Ruiz Pérez", "Jefa de la Unidad Jurídica"), "«Persona, Cargo»: se parte")
_SUB = "Subdelegado de Prestaciones Económicas, Delegación Estatal Querétaro, del ISSSTE"
_rf6 = ft.armar({"tipo_asunto": "revision_fiscal", "numero": "6/2026"},
                escrito={"promovente": _SUB, "caracter": "autoridad",
                         "figura_representante": "Titular de la Unidad Jurídica…",
                         "representante": "Titular de la Unidad de Asuntos Jurídicos de la citada delegación",
                         "avisos": []})
ok(_rf6["figura_representante"] == "Titular de la Unidad de Asuntos Jurídicos de la citada delegación"
   and _rf6["representante"] == "" and "representante" not in _rf6["fuentes"],
   f"RF 6/2026: la figura cortada fuera; el representante que es un cargo pasa a la figura: "
   f"{_rf6['figura_representante']!r} / {_rf6['representante']!r}")
ok(any(a.startswith("DATO TRUNCADO (figura_representante") for a in _rf6["avisos"])
   and any("ES UN CARGO" in a for a in _rf6["avisos"]),
   "…con un aviso de cada cosa (el del dato cortado, en la forma compartida «DATO TRUNCADO (clave = …)»)")
_rf7 = ft.armar({"tipo_asunto": "revision_fiscal", "numero": "7/2025"},
                escrito={"promovente": "Jefa de la Unidad Jurídica de la Delegación Estatal del ISSSTE en Baja "
                                       "California Sur", "caracter": "autoridad",
                         "representante": "Ana Ruiz Pérez, Jefa de la Unidad Jurídica", "avisos": []})
ok(_rf7["representante"] == "Ana Ruiz Pérez" and _rf7["figura_representante"] == "Jefa de la Unidad Jurídica"
   and any("ES EL CARGO DE QUIEN RECURRE" in a for a in _rf7["avisos"]),
   f"RF 7: la titular de la propia unidad: nombre aparte y figura contenida en la unidad (el compositor la pone en "
   f"aposición): {_rf7['representante']!r} / {_rf7['figura_representante']!r}")
_banco = {"tipo": "revision_fiscal", "promovente": _SUB, "caracter": "autoridad",
          "figura_representante": "Titular de la Unidad Jurídica…", "representante": "Representante Testado",
          "adhesivo": {"quien": "Transportes del Centro…", "presentacion": "", "admision": "", "notificacion": ""},
          "terceros": ["Ana López", "Pedro…"], "fuentes": {}}
_ab = ft.validar(_banco)
ok(_banco["figura_representante"] == "" and _banco["adhesivo"] == {} and _banco["terceros"] == ["Ana López"]
   and _banco["representante"] == "Representante Testado" and sum(a.startswith("DATO TRUNCADO") for a in _ab) == 3,
   f"validar sobre una ficha que no pasó por armar (el banco, el compositor): quita lo cortado y lo dice: {_banco}")
ok(not any("TRUNCADO" in a for a in ft.validar(_banco)), "…y es idempotente: la segunda vez, nada")
_au = ft.validar({"tipo": "revision_fiscal", "caracter": "autoridad", "promovente": _SUB,
                  "figura_representante": "autorizado en términos del artículo 5o. de la LFPCA",
                  "representante": "Luis Gómez Ruiz"})
ok(any(a.startswith("UN AUTORIZADO NO INTERPONE LA REVISIÓN FISCAL") and "63" in a for a in _au),
   "RF 4: un «autorizado» que interpone por la autoridad: aviso (el 63 exige la unidad jurídica)")
ok(not any("AUTORIZADO NO INTERPONE" in a for a in ft.validar(
    {"tipo": "amparo_revision", "caracter": "quejoso", "figura_representante": "autorizado en términos amplios"})),
   "…en el amparo en revisión, el autorizado del quejoso es normal: sin aviso")
_lt2 = ft.leer_auto("Querétaro, Querétaro, a cinco de marzo de dos mil veinticinco. Fórmese el expediente número 4/2025 "
                    "con el recurso de revisión fiscal interpuesto por la Titular de la Unidad Jurídica de la Oficina de "
                    "Representación del Instituto Mexicano del Seguro Social en Querétaro, contra la sentencia de diez "
                    "de enero de dos mil veinticinco; se admite el recurso.", "revision_fiscal")
ok(str(_lt2.get("promovente", "")).startswith("la Titular de la Unidad Jurídica"),
   f"«interpuesto por la Titular…»: el artículo del papel se conserva (es lo único que dice el género): "
   f"{_lt2.get('promovente')!r}")
_esc_rf = ("C. MAGISTRADOS DEL TRIBUNAL COLEGIADO. PRESENTE. Karen Yadira Meza Ruiz, en representación de la Titular "
           "de la Administración Desconcentrada Jurídica de Querétaro \"1\", autoridad demandada en el juicio contencioso "
           "administrativo 695/25-09-01-7-OT, interpongo recurso de revisión fiscal en contra de la sentencia de "
           "veintidós de septiembre de dos mil veinticinco.")
_erf = asyncio.run(ft.leer_escrito(None, _esc_rf, "revision_fiscal"))
ok(_erf.get("promovente") == 'la Titular de la Administración Desconcentrada Jurídica de Querétaro "1"'
   and _erf.get("representante") == "Karen Yadira Meza Ruiz",
   f"escrito RF: «en representación de la Titular…, autoridad demandada»: el artículo se queda y el rol no es "
   f"parte del nombre: {_erf.get('promovente')!r}")
_lt3 = ft.leer_auto("Querétaro, Querétaro, a doce de agosto de dos mil veinticinco. Fórmese el expediente número "
                    "631/2025 con el recurso de revisión interpuesto por la Unión de Trabajadores de la Construcción, "
                    "contra la sentencia de veintidós de abril de dos mil veinticinco; recurso que se admite.")
ok(_lt3.get("promovente") == "Unión de Trabajadores de la Construcción",
   "…pero «la Unión de Trabajadores…» sigue sin el artículo de la frase")

# ── 16.5 · E11: UN AVISO POR DATO, CON SU CLAVE ─────────────────────────────
_vd = _f("validar_detalle")
_det = _seguro(_vd, {"tipo": "queja", "presentacion": "2026-04-10", "acto": {"fecha": "2026-05-08"}})
ok(isinstance(_det, list) and any(x.get("rutas") == ["presentacion", "acto.fecha"] and x.get("clave") == "presentacion"
                                  for x in _det if isinstance(x, dict)),
   f"validar_detalle: cada aviso con las rutas de su dato (Q 229): {_det if not isinstance(_det, list) else [x.get('rutas') for x in _det]}")
_448 = {"tipo": "amparo_revision", "audiencia": "2025-07-15", "acto": {"fecha": "2025-07-15",
        "organo": "Juzgado Cuarto de Distrito en el Estado de Querétaro"}, "presentacion": "2025-05-13"}
_a448 = ft.validar(_448)
ok(sum(a.startswith("FECHA IMPOSIBLE") and "13/05/2025" in a for a in _a448) == 1
   and any("mismo día de la audiencia" in a for a in _a448),
   f"AR 448: el recurso antes de la recurrida y de la audiencia DEL MISMO DÍA es UN aviso: {[a[:80] for a in _a448]}")
_JZL = ("Juzgado Séptimo de Distrito en Materia de Amparo Civil, Administrativo y de Trabajo y de Juicios Federales "
        "en el Estado de Querétaro")
_a222 = ft.validar({"tipo": "amparo_revision", "acto": {"organo": _JZL}, "demanda": {"autoridades": [_JZL]}})
ok(any(f"«{_JZL}»" in a for a in _a222), "AR 222: el nombre del juzgado, ENTERO en el aviso (no «…Juicios Federales en »)")
_fk = ft.armar({"tipo_asunto": "queja", "numero": "229/2026", "presentacion": "2026-04-10"},
               acto={"acto": {"fecha": "2026-05-08"}})
_ka = [a for a in _fk["avisos"] if a.startswith("FECHA IMPOSIBLE") and "10/04/2026" in a]
ok(_ka and _seguro(_f("clave_de_aviso"), _fk, _ka[0]) == ["presentacion", "acto.fecha"]
   and isinstance(_fk.get("avisos_claves"), dict),
   "armar: la ficha trae «avisos_claves» y clave_de_aviso dice de qué dato es cada aviso")
_fk2 = ft.armar({"tipo_asunto": "amparo_directo", "numero": "274/2025"}, auto={"turno": {"fecha": "2025-02-21"}},
                declarado=json.dumps({"fecha_turno": "2025-05-14"}))
_nc = [a for a in _fk2["avisos"] if "NO CUADRA" in a]
ok(_nc and _fk2.get("avisos_claves", {}).get(_nc[0]) == ["turno.fecha"], "…también los «NO CUADRA ENTRE FUENTES» de armar")

# ── 16.6 · E12: QUIEN RECURRE COMO QUEJOSA Y NO ES LA QUEJOSA (AR 60/2025) ──
_SUC = "Sucesión intestamentaria a bienes de José García Ruiz"
_a60 = ft.validar({"tipo": "amparo_revision", "caracter": "quejoso", "promovente": "María Fernanda Pérez López",
                   "quejoso": _SUC})
ok(any("NO ES LA PARTE QUEJOSA" in a and "albacea" in a for a in _a60),
   "AR 60: recurre «como quejosa» la albacea (persona física) y la quejosa es la sucesión: aviso")
ok(not any("NO ES LA PARTE QUEJOSA" in a for a in ft.validar(
    {"tipo": "amparo_revision", "caracter": "quejoso", "quejoso": _SUC,
     "promovente": _SUC + ", a través de su albacea María Fernanda Pérez López"})),
   "…la sucesión que recurre por conducto de su albacea es la quejosa: sin aviso")
ok(not any("NO ES LA PARTE QUEJOSA" in a for a in ft.validar(
    {"tipo": "queja", "caracter": "quejoso", "promovente": "Juan Pérez López",
     "quejoso": "Juan y Pedro, ambos de apellidos Pérez López"})),
   "…uno de varios quejosos recurre: sin aviso")
ok(not any("NO ES LA PARTE QUEJOSA" in a for a in ft.validar(
    {"tipo": "amparo_directo", "caracter": "quejoso", "promovente": "Ana Ruiz", "quejoso": "Otra Persona Distinta"})),
   "…en el amparo directo no aplica")
ok(not any("NO ES LA PARTE QUEJOSA" in a for a in ft.validar(
    {"tipo": "amparo_revision", "caracter": "quejoso", "promovente": "PARTE RECURRENTE", "quejoso": "PARTE QUEJOSA"})),
   "…un rótulo («PARTE RECURRENTE», lo testado del banco) no es un nombre: no se compara")
ok(not any("NO ES LA PARTE QUEJOSA" in a for a in ft.validar(
    {"tipo": "queja", "caracter": "quejoso",
     "promovente": "SUCESIÓN TESTAMENTARIA A BIENES DE PARTE RECURRENTE, A TRAVÉS DE SU ALBACEA PARTE RECURRENTE",
     "quejoso": "SUCESIÓN TESTAMENTARIA A BIENES DE PARTE QUEJOSA, A TRAVÉS DE SU ALBACEA PARTE QUEJOSA"})),
   "…ni dentro de un nombre más largo (Q 342 del banco)")

print("\nINTEGRACIÓN · EL ACTO RECLAMADO CORTADO A MEDIA FRASE SE COMPLETA CON EL PAPEL (AR 631/2025)")
_pap631 = ("ACTOS RECLAMADOS: Resolución interlocutoria de ocho de abril de dos mil veinticuatro, relativo al juicio "
           "sumario civil sobre rescisión de contrato de arrendamiento promovido por Unión de\nTrabajadores de la "
           "Construcción, C.T.M., en contra de Impulsora de Desarrollos S.A. de C.V. y otros. SEGUNDO. Trámite.")
_av = []
_c = ft._completar_oracion("relativo al juicio sumario civil sobre rescisión de contrato de arrendamiento "
                           "promovido por Unión de", _pap631, _av)
ok(_c.endswith("en contra de Impulsora de Desarrollos S.A. de C.V. y otros.") and not _av,
   "se sigue en el papel hasta el punto final, sin cortar en «C.T.M.» ni en «S.A.»")
ok(ft._completar_oracion("La resolución de dos de julio.", _pap631, []) == "La resolución de dos de julio.",
   "lo que ya cierra con su punto no se toca")
_av = []
ok(ft._completar_oracion("algo que no está en el papel de", _pap631, _av) == "algo que no está en el papel de"
   and any("CORTADO A MEDIA FRASE" in a for a in _av), "si no se encuentra, queda como está y se avisa")

print("\n17 · QUINTA RONDA (3-oct-2026): LA FORMA QUE NADIE DIJO, EL DERECHO RECONOCIDO, LA QUEJA DE LA FR. II, "
      "EL SENTIDO A MEDIA FRASE")

# ── 17.1 · F1: LA FORMA DE NOTIFICACIÓN POR OMISIÓN NO ES UN DATO ───────────
# Q 335/2025, AD 274 y AD 335/2025 del banco: la forma no constaba y el
# considerando escribía «de manera personal». En producción, `main` recibe
# `regla_surtimiento: str = Form("personal")` y `armar` lo metía como forma
# DECLARADA por el secretario, por delante del escrito y del acto.
_forma_por_omision = _f("forma_por_omision")
_enc_ad = _Enc(numero="274/2025", encabezado="AMPARO DIRECTO CIVIL: 274/2025", quejoso="Gabriel Reyes Álamo",
               magistrado="", secretario="", notificacion=_dtt.date(2025, 2, 24),
               presentacion=_dtt.date(2025, 3, 13), tipo_asunto="amparo_directo",
               tribunal=E["tribunal"], ciudad="Querétaro, Querétaro")
ok(getattr(_enc_ad, "regla_surtimiento", "") == "personal", "el Encargo llega con la regla de omisión («personal»)")
_fo = ft.armar(_enc_ad, acto=x)
_av_fo = [a for a in _fo["avisos"] if a.startswith("LA FORMA DE NOTIFICACIÓN NO CONSTA")]
ok(_fo["forma_notificacion"] == "personal" and _fo["fuentes"].get("forma_notificacion") == "omision"
   and _seguro(_forma_por_omision, _fo) == "omision",
   f"AD 274: la forma sale de la omisión, con fuente «omision» (no «secretario»): "
   f"{_fo['forma_notificacion']!r} / {_fo['fuentes'].get('forma_notificacion')!r}")
ok(len(_av_fo) == 1 and "personal" in _av_fo[0] and "Trámite en este tribunal" in _av_fo[0]
   and _fo["avisos_claves"].get(_av_fo[0]) == ["forma_notificacion"],
   f"…con UN aviso «LA FORMA DE NOTIFICACIÓN NO CONSTA», con su clave: {[a[:70] for a in _av_fo]}")
ok(ft.a_formulario(_fo)["forma_notificacion"] == "",
   "…y la tarjeta no la propone como dato (a_formulario: «»; al retomar no vuelve como leída)")
_fo_r = ft.armar(_enc_ad, acto=x, declarado=_fo)
ok(_fo_r["fuentes"].get("forma_notificacion") == "omision",
   f"…y una ficha ya armada que vuelve como «declarada» no convierte la omisión en dato del secretario: "
   f"{_fo_r['fuentes'].get('forma_notificacion')!r}")
import copy as _cp5
_enc_lista = _cp5.copy(_enc_ad)
_enc_lista.regla_surtimiento = "lista"
_fo_l = ft.armar(_enc_lista, acto=x)
ok(_fo_l["forma_notificacion"] == "lista" and _fo_l["fuentes"].get("forma_notificacion") == "secretario"
   and not any(a.startswith("LA FORMA DE NOTIFICACIÓN NO CONSTA") for a in _fo_l["avisos"])
   and _seguro(_forma_por_omision, _fo_l) == "",
   "una regla ELEGIDA («lista») sí es dato del secretario: sin aviso")
_fo_d = ft.armar(_enc_ad, acto=x, declarado={"forma_notificacion": "personal"})
ok(_fo_d["fuentes"].get("forma_notificacion") == "secretario"
   and not any(a.startswith("LA FORMA DE NOTIFICACIÓN NO CONSTA") for a in _fo_d["avisos"]),
   "«personal» declarada en «Trámite en este tribunal» es del secretario: sin aviso")
_fo_a = ft.armar(_enc_ad, acto=dict(x, forma_notificacion="electronica"))
ok(_fo_a["forma_notificacion"] == "electronica" and _fo_a["fuentes"].get("forma_notificacion") == "acto",
   f"lo que dice el ACTO manda sobre la omisión: {_fo_a['forma_notificacion']!r} / "
   f"{_fo_a['fuentes'].get('forma_notificacion')!r}")
_Q5 = {"tipo_asunto": "queja", "numero": "335/2025", "recurrente": "Ana López Ruiz", "presentacion": "2025-10-13",
       "notificacion": "2025-10-06", "regla_surtimiento": "personal"}
_fo_e = ft.armar(_Q5, escrito={"promovente": "Ana López Ruiz", "caracter": "quejoso", "forma_notificacion": "lista"})
ok(_fo_e["forma_notificacion"] == "lista" and _fo_e["fuentes"].get("forma_notificacion") == "escrito",
   f"queja: lo que dice el ESCRITO («me fue notificado por lista») manda sobre la omisión: "
   f"{_fo_e['forma_notificacion']!r} / {_fo_e['fuentes'].get('forma_notificacion')!r}")
ok(ft.armar(dict(E, regla_surtimiento="otra"), acto=x)["forma_notificacion"] == ""
   and ft.armar(E, acto=x)["fuentes"].get("forma_notificacion") is None,
   "«otra» y sin regla: no hay forma (ni omisión)")
ok(_seguro(_forma_por_omision, {"forma_notificacion": "personal", "fuentes": {"forma_notificacion": "escrito"}}) == ""
   and _seguro(_forma_por_omision, {"forma_notificacion": "", "fuentes": {"forma_notificacion": "omision"}}) == ""
   and _seguro(_forma_por_omision, None) == "",
   "forma_por_omision: «» si la dijo un papel, si no hay forma o si no hay ficha")

# EL OFICIO DE D2 (la autoridad que recurre): `redactor_adelanto` cambia la
# regla a «oficio» ANTES de armar; ese oficio no lo dijo nadie.
_AR5 = {"tipo_asunto": "amparo_revision", "numero": "208/2025", "es_recurso": True,
        "recurrente": "Gobernador del Estado de Querétaro", "presentacion": "2025-05-20",
        "notificacion": "2025-05-08", "regla_surtimiento": "oficio"}
_FP_AUT = {"recurrente": {"nombre": "Gobernador del Estado de Querétaro", "papel": "autoridad"}}
_fo_o = ft.armar(_AR5, ficha_procesal=_FP_AUT)
ok(_fo_o["caracter"] == "autoridad" and _fo_o["forma_notificacion"] == "oficio"
   and _fo_o["fuentes"].get("forma_notificacion") == "omision_autoridad"
   and _seguro(_forma_por_omision, _fo_o) == "omision_autoridad",
   f"AR: recurre una autoridad y ningún papel dice cómo se le notificó: «oficio» con fuente «omision_autoridad»: "
   f"{_fo_o['fuentes'].get('forma_notificacion')!r}")
ok(not any(a.startswith("LA FORMA DE NOTIFICACIÓN NO CONSTA") for a in _fo_o["avisos"]),
   "…sin el aviso de la personal (el del cómputo de la autoridad ya lo dice, E11)")
_fo_oe = ft.armar(_AR5, ficha_procesal=_FP_AUT,
                  escrito={"promovente": "Gobernador del Estado de Querétaro", "caracter": "autoridad",
                           "forma_notificacion": "electronica"})
ok(_fo_oe["forma_notificacion"] == "electronica" and _fo_oe["fuentes"].get("forma_notificacion") == "escrito",
   "…si el escrito dice «electrónica», manda el escrito")
_fo_oq = ft.armar(dict(_AR5, recurrente="Ana López Ruiz"),
                  ficha_procesal={"recurrente": {"nombre": "Ana López Ruiz", "papel": "quejoso"}})
ok(_fo_oq["forma_notificacion"] == "oficio" and _fo_oq["fuentes"].get("forma_notificacion") == "secretario",
   "…si recurre un particular, D2 no puso ese «oficio»: lo eligió el secretario y es suyo")
ok(ft.armar(dict(_AR5, tipo_asunto="revision_fiscal", numero="7/2025"))["fuentes"].get("forma_notificacion")
   == "secretario", "…y en la revisión fiscal el «oficio» elegido es del secretario (D2 es de la revisión y la queja)")
if "_ra" in globals() and hasattr(_ra, "regla_de_la_autoridad"):
    # LA CADENA DE VERDAD: el Encargo con la regla de omisión, D2 la cambia y se arma.
    _e_d2 = _Enc(numero="208/2025", encabezado="AMPARO EN REVISIÓN: 208/2025", quejoso="Juan Pérez López",
                 magistrado="", secretario="", notificacion=_dtt.date(2025, 5, 8),
                 presentacion=_dtt.date(2025, 5, 20), tipo_asunto="amparo_revision", es_recurso=True,
                 recurrente="Gobernador del Estado de Querétaro")
    _rr_d2 = _seguro(_ra.regla_de_la_autoridad, _e_d2, "autoridad")
    _r_d2 = _rr_d2[0] if isinstance(_rr_d2, tuple) else ""
    _e_d2.regla_surtimiento = _r_d2 or _e_d2.regla_surtimiento
    _fo_d2 = ft.armar(_e_d2, ficha_procesal=_FP_AUT)
    ok(_r_d2 == "oficio" and _fo_d2["fuentes"].get("forma_notificacion") == "omision_autoridad",
       f"con redactor_adelanto.regla_de_la_autoridad (D2) y luego armar: «omision_autoridad» "
       f"({_r_d2!r}, {_fo_d2['fuentes'].get('forma_notificacion')!r})")

# ── 17.2 · LA SALA QUE ADEMÁS RECONOCIÓ UN DERECHO (RF 6/2026) ─────────────
ACTO_RF_DERECHO = ACTO_RF.replace(
    "II. Se declara la\nnulidad lisa y llana de la resolución impugnada. NOTIFÍQUESE.",
    "II. Se declara la\nnulidad de la resolución impugnada, para los efectos precisados en el último considerando "
    "de este fallo. III. Se reconoce el derecho subjetivo de la actora al incremento de la cuota pensionaria y al "
    "pago retroactivo de las diferencias, y se condena a la autoridad demandada a su pago. NOTIFÍQUESE.")
ok(ACTO_RF_DERECHO != ACTO_RF, "(el papel de prueba cambió los resolutivos)")
_rd = asyncio.run(ft.leer_acto(None, ACTO_RF_DERECHO, "revision_fiscal", "6/2026"))
_frase_ef = ft.CATALOGO_SENTIDO["revision_fiscal"]["nulidad_efectos"][1]
_s_rd = _d(_rd.get("acto")).get("sentido", "")
ok(_d(_rd.get("acto")).get("sentido_clave") == "nulidad_efectos"
   and _s_rd == (_frase_ef + ", y reconoció el derecho subjetivo de la actora al incremento de la cuota "
                 "pensionaria y al pago retroactivo de las diferencias"),
   f"RF 6/2026: la clave del catálogo y la cola del derecho reconocido, copiada del resolutivo: «{_s_rd}»")
_fr_rd = ft.armar({"tipo_asunto": "revision_fiscal", "numero": "6/2026"}, acto=_rd)
ok(_fr_rd["acto"]["sentido"] == _s_rd and _fr_rd["acto"]["sentido"].startswith(_frase_ef),
   "…armar la conserva, y empieza por la frase del catálogo (el compositor la reconoce)")
ok(set(ft.CATALOGO_SENTIDO["revision_fiscal"]) == {"nulidad_lisa", "nulidad_efectos", "validez", "sobresee"},
   "el catálogo de la revisión fiscal no cambia de claves (el compositor exige las mismas frases)")
_cola = _f("cola_derecho_subjetivo")
_res_d = ("R E S U E L V E: PRIMERO. Se declara la nulidad de la resolución impugnada. SEGUNDO. Se reconoce a la "
          "parte actora el derecho a percibir la pensión conforme al art. 57 de la Ley del ISSSTE. TERCERO. "
          "Notifíquese.")
ok(_seguro(_cola, _res_d) == "y reconoció a la parte actora el derecho a percibir la pensión conforme al art. 57 de "
                             "la Ley del ISSSTE",
   f"«art. 57» no corta la cola; el punto que cierra, sí: {_seguro(_cola, _res_d)!r}")
ok(_seguro(_cola, _res_d.replace("a la parte actora", "a JUAN PÉREZ LÓPEZ")) == "y reconoció a Juan Pérez López el "
   "derecho a percibir la pensión conforme al art. 57 de la Ley del ISSSTE",
   "el nombre en versales dentro del tramo pasa a mayúsculas y minúsculas; la sigla sola se queda")
ok(_seguro(_cola, _res_d.replace("Se reconoce a la", "No se reconoce a la")) == "",
   "«no se reconoce el derecho» no es reconocerlo")
ok(_seguro(_cola, _res_d.upper()) == "y reconoció a la parte actora la existencia de un derecho subjetivo",
   "en versales no se copia (no se sabe qué va en mayúscula): la fórmula del artículo 52, fr. V, a)")
ok(_seguro(_cola, "R E S U E L V E: PRIMERO. Se reconoce la validez de la resolución impugnada.") == "",
   "reconocer la VALIDEZ no es reconocer un derecho")
_rv = asyncio.run(ft.leer_acto(None, ACTO_RF, "revision_fiscal", "91/2025"))
ok(_d(_rv.get("acto")).get("sentido") == ft.CATALOGO_SENTIDO["revision_fiscal"]["nulidad_lisa"][1],
   "sin derecho reconocido, la frase del catálogo tal cual")

# ── 17.3 · LA QUEJA DE LA FRACCIÓN II EN SUS DOS AUTOS (Q 335/2025, t6) ──────
_q2 = {"tipo": "queja", "fraccion_97": "II", "registro": {"fecha": "2025-11-03"},
       "admision": {"fecha": "2025-11-03"}, "informe_101": {"fecha": "2025-11-03"}}
_aq2 = ft.validar(_q2)
ok("registro.fecha = informe_101.fecha" in _q2["fechas_imposibles"]
   and any("dos autos distintos" in a for a in _aq2),
   f"fr. II: registro con la fecha del informe (el mismo auto dos veces): aviso: {[a[:80] for a in _aq2]}")
_q2b = {"tipo": "queja", "fraccion_97": "II", "admision": {"fecha": "2025-10-14"},
        "informe_101": {"fecha": "2025-11-03"}}
_aq2b = ft.validar(_q2b)
ok("admision.fecha < informe_101.fecha" in _q2b["fechas_imposibles"]
   and any("Auto de Presidencia que forma y registra" in a for a in _aq2b),
   "fr. II: la admisión antes del auto que tuvo por rendido el informe: aviso, con la pista del registro")
ok(not ft.validar({"tipo": "queja", "fraccion_97": "II", "registro": {"fecha": "2025-10-14"},
                   "admision": {"fecha": "2025-11-03"}, "informe_101": {"fecha": "2025-11-03"}}),
   "Q 335 como es (registro 14-oct; informe y admisión 3-nov): ningún aviso")
ok(not ft.validar({"tipo": "queja", "fraccion_97": "I", "admision": {"fecha": "2025-10-14"},
                   "informe_101": {"fecha": "2025-11-03"}}),
   "fr. I: la regla de los dos autos no aplica")
_q2c = {"tipo": "queja", "acto": {"via": "directo"}, "admision": {"fecha": "2025-10-14"},
        "informe_101": {"fecha": "2025-11-03"}}
ft.validar(_q2c)
ok("admision.fecha < informe_101.fecha" in _q2c["fechas_imposibles"],
   "sin la fracción, la vía «directo» dice que es la II")
_fq2 = ft.armar({"tipo_asunto": "queja", "numero": "335/2025"},
                declarado={"fecha_registro": "2025-11-03", "fecha_admision": "2025-11-03",
                           "fecha_informe_101": "2025-11-03", "fraccion_97": "II"})
ok(_fq2.get("registro") == {"fecha": "2025-11-03"} and any("dos autos distintos" in a for a in _fq2["avisos"])
   and _fq2["avisos_claves"].get(next((a for a in _fq2["avisos"] if "dos autos distintos" in a), ""))
   == ["registro.fecha", "informe_101.fecha"],
   "armar: el registro que tecleó el secretario se respeta, y se avisa (con la clave del dato)")
_fq2l = ft.armar({"tipo_asunto": "queja", "numero": "335/2025"},
                 auto=[{"registro": {"fecha": "2025-11-03"}, "informe_101": {"fecha": "2025-11-03"},
                        "admision": {"fecha": "2025-11-10"}}], acto={"acto": {"via": "directo"}})
ok(_fq2l.get("registro") == {} and "registro.fecha" not in _fq2l["fuentes"]
   and not any("dos autos distintos" in a for a in _fq2l["avisos"]),
   f"armar: un registro LEÍDO con la fecha del informe (fr. II) es el mismo auto leído dos veces: fuera "
   f"({_fq2l.get('registro')})")

# ── 17.4 · EL SENTIDO A MEDIA FRASE, CON INICIAL MINÚSCULA (Q 335/2025) ─────
_min = _f("sentido_en_minuscula")
ok(_seguro(_min, "Dejó sin efectos la suspensión del acto reclamado")
   == "dejó sin efectos la suspensión del acto reclamado"
   and _seguro(_min, "IMSS negó la pensión") == "IMSS negó la pensión"
   and _seguro(_min, "DESECHÓ LA DEMANDA") == "DESECHÓ LA DEMANDA"
   and _seguro(_min, "«Se deja sin efectos»") == "«se deja sin efectos»",
   "la inicial a minúscula; la sigla y lo que viene en versales se quedan")
_fs = ft.armar({"tipo_asunto": "queja", "numero": "335/2025"},
               declarado={"sentido": "Dejó sin efectos la suspensión del acto reclamado"})
ok(_fs["acto"]["sentido"] == "dejó sin efectos la suspensión del acto reclamado",
   f"armar: «en el que Dejó sin efectos…» no vuelve a salir: «{_fs['acto']['sentido']}»")
ACTO_Q_LIBRE = ("JUZGADO TERCERO DE DISTRITO EN MATERIA DE AMPARO CIVIL, ADMINISTRATIVO Y DE TRABAJO Y DE JUICIOS "
                "FEDERALES EN EL ESTADO DE QUERÉTARO\nJuicio de amparo indirecto 512/2026\nQuerétaro, Querétaro, a dos "
                "de marzo de dos mil veintiséis.\nVisto el estado de los autos, se acuerda: Se deja sin efectos la "
                "suspensión provisional concedida a la parte quejosa. Notifíquese.")
cli, comp = cliente_falso({"sentido_clave": "otro",
                           "sentido_libre": "Se deja sin efectos la suspensión provisional concedida a la parte quejosa"})
_ql = asyncio.run(ft.leer_acto(cli, ACTO_Q_LIBRE, "queja", "40/2026"))
ok(_d(_ql.get("acto")).get("sentido") == "se deja sin efectos la suspensión provisional concedida a la parte quejosa",
   f"leer_acto: el sentido libre anclado al papel sale con minúscula: «{_d(_ql.get('acto')).get('sentido')}»")

# ═══════════════════════════════════════════════════════════════════════════
print("\n18 · SEXTA RONDA (3-oct-2026): LOS ASUNTOS RELACIONADOS (C6) Y LA DEMANDA EN LA QUEJA (C3)")
# ═══════════════════════════════════════════════════════════════════════════
# ── 18.1 · C6: «relacionados», del formulario a la ficha y de vuelta ────────
# David: «siempre y cuando haya asuntos relacionados. No vamos a meter
# conexidad en automático». Texto plano «tipo|numero|estado;…»; en la ficha,
# la lista; fuente siempre «secretario».
_rel_de = _f("relacionados_de")
_d0 = ft.de_formulario({"relacionados": ""})
ok("relacionados" not in _d0 and not _d0["avisos"] and ft.a_formulario(_d0)["relacionados"] == "",
   "0 relacionados: «» no entra, sin aviso, y vuelve como «»")
_d1 = ft.de_formulario({"relacionados": "amparo_directo|452/2025|misma_sesion"})
ok(_d1.get("relacionados") == [{"tipo": "amparo_directo", "numero": "452/2025", "estado": "misma_sesion"}]
   and _d1["fuentes"].get("relacionados") == "secretario" and not _d1["avisos"],
   f"1 relacionado: la lista con su fuente «secretario»: {_d1.get('relacionados')}")
ok(ft.a_formulario(_d1)["relacionados"] == "amparo_directo|452/2025|misma_sesion"
   and ft.de_formulario(ft.a_formulario(_d1)).get("relacionados") == _d1["relacionados"],
   "1 relacionado: ida y vuelta sin pérdida")
_t3 = "amparo_directo|452/2025|misma_sesion;revision_fiscal|33/2024|resuelto;queja|24/2026"
_d3 = ft.de_formulario({"relacionados": _t3})
ok([r["estado"] for r in _d3.get("relacionados") or []] == ["misma_sesion", "resuelto", "misma_sesion"]
   and [r["tipo"] for r in _d3.get("relacionados") or []] == ["amparo_directo", "revision_fiscal", "queja"]
   and not _d3["avisos"],
   "3 relacionados, en su orden; el estado ausente es «misma_sesion»")
_v3 = ft.a_formulario(_d3)["relacionados"]
ok(_v3 == _t3 + "|misma_sesion" and ft.de_formulario({"relacionados": _v3}).get("relacionados") == _d3["relacionados"],
   f"3 relacionados: a_formulario los serializa con el estado explícito y vuelven iguales: «{_v3}»")
ok(ft.a_formulario(ft.de_formulario({"relacionados": " amparo directo | 452 / 2025 | Ya resuelto ; RF|33/2024"}))
   ["relacionados"] == "amparo_directo|452/2025|resuelto;revision_fiscal|33/2024|misma_sesion",
   "se escriba como se escriba («amparo directo», «452 / 2025», «Ya resuelto», «RF»), sale en la forma del contrato")
_dm = ft.de_formulario({"relacionados": "amparo_indirecto|1/2025;amparo_directo|452-2025;"
                                        "amparo_directo|452/2025|pendiente;queja|24/2026"})
ok(_dm.get("relacionados") == [{"tipo": "queja", "numero": "24/2026", "estado": "misma_sesion"}]
   and len([a for a in _dm["avisos"] if "(relacionados)" in a]) == 3
   and any("el tipo «amparo_indirecto»" in a for a in _dm["avisos"])
   and any("no tiene la forma «452/2025»" in a for a in _dm["avisos"])
   and any("el estado «pendiente»" in a for a in _dm["avisos"]),
   f"inválidos (tipo, número, estado) fuera, cada uno con su aviso: {[a[:70] for a in _dm['avisos']]}")
_dp = ft.de_formulario({"relacionados": "amparo_directo|456/2025;revision_fiscal|456/2025|misma_sesion"},
                       "amparo_directo", numero="456/2025")
ok(_dp.get("relacionados") == [{"tipo": "revision_fiscal", "numero": "456/2025", "estado": "misma_sesion"}]
   and any("ES ESTE MISMO ASUNTO" in a for a in _dp["avisos"]),
   "el propio asunto fuera con aviso (la revisión fiscal con el mismo número es otro expediente y se queda)")
_dp2 = ft.de_formulario({"numero": "456/2025", "tipo_asunto": "amparo_directo",
                         "relacionados": "amparo_directo|456/2025"})
ok(not _dp2.get("relacionados") and any("ES ESTE MISMO ASUNTO" in a for a in _dp2["avisos"])
   and not any("SE IGNORÓ LO QUE NO ES" in a for a in _dp2["avisos"]),
   "el propio número también sale de «numero» dentro del formulario (contexto, no dato ajeno)")
_dd = ft.de_formulario({"relacionados": "amparo_directo|452/2025;amparo_directo|452/2025|resuelto"})
ok(_dd.get("relacionados") == [{"tipo": "amparo_directo", "numero": "452/2025", "estado": "misma_sesion"}]
   and not _dd["avisos"], "duplicados fuera (vale el primero), sin aviso")
_d5 = ft.de_formulario({"relacionados": ";".join(f"queja|{i}/2026" for i in range(1, 6))})
ok(len(_d5.get("relacionados") or []) == 4 and any("EL TOPE ES 4" in a for a in _d5["avisos"]),
   "más de cuatro: los cuatro primeros, con aviso")
_dl = ft.de_formulario({"relacionados": "amparo_directo|452/2025|misma_sesion;" * 20})
ok(not _dl.get("relacionados") and any("DEMASIADO LARGO" in a for a in _dl["avisos"]),
   "un texto que excede el tope se ignora con aviso (no se recorta)")
_dh = ft.de_formulario({"relacionados": [{"tipo": "amparo_directo", "numero": "452/2025"}]})
ok(not _dh.get("relacionados") and any("NO TRAE TEXTO" in a for a in _dh["avisos"]),
   "por HTTP es texto: una lista se ignora con aviso")
_dx = ft.de_formulario({"relacionados": [{"tipo": "AD", "numero": "452 / 2025"}, "RF|33/2024|resuelto"]},
                       extra=True)
ok(_dx.get("relacionados") == [{"tipo": "amparo_directo", "numero": "452/2025", "estado": "misma_sesion"},
                               {"tipo": "revision_fiscal", "numero": "33/2024", "estado": "resuelto"}],
   "uso interno (extra=True): la lista ya hecha entra normalizada")
ok(_seguro(_rel_de, None) == ([], []) and _seguro(_f("relacionados_a_texto"), []) == ""
   and _seguro(_f("relacionados_a_texto"), "x|y") == "",
   "relacionados_de / relacionados_a_texto: vacío y basura no lanzan")
ok(len(ft.a_formulario({})) == 22 and ft.a_formulario({})["relacionados"] == "",
   "a_formulario: «relacionados» siempre presente («» sin lista)")

# armar: sólo lo declarado; el propio número sale con el de la ficha.
_Erel = {"tipo_asunto": "amparo_directo", "numero": "456/2025", "materia": "civil"}
_decl_rel = ft.de_formulario({"relacionados": "amparo_directo|456/2025|misma_sesion;"
                                              "revision_fiscal|33/2024|resuelto"})
_fr = ft.armar(_Erel, declarado=_decl_rel)
_av_propio = next((a for a in _fr["avisos"] if "ES ESTE MISMO ASUNTO" in a), "")
ok(_fr["relacionados"] == [{"tipo": "revision_fiscal", "numero": "33/2024", "estado": "resuelto"}]
   and _fr["fuentes"].get("relacionados") == "secretario"
   and _av_propio and _fr["avisos_claves"].get(_av_propio) == ["relacionados"],
   "armar: por HTTP de_formulario no sabía el número; armar quita el propio asunto, con aviso y su clave")
ok(ft.armar(_Erel)["relacionados"] == [] and "relacionados" not in ft.armar(_Erel)["fuentes"],
   "sin relacionados marcados, la ficha trae [] y ninguna fuente")
_fpap = ft.armar(_Erel, auto={"relacionados": [{"tipo": "queja", "numero": "1/2026", "estado": "misma_sesion"}],
                              "fuentes": {"relacionados": "auto"}},
                 acto={"relacionados": "queja|2/2026", "acto": {"fecha": "2025-06-01"}},
                 escrito={"relacionados": "queja|3/2026"})
ok(_fpap["relacionados"] == [], "NUNCA de los papeles: lo que traigan el auto, el acto o el escrito no entra")
_frj = ft.armar(_Erel, declarado=json.dumps({"relacionados": "revision_fiscal|33/2024|misma_sesion"}))
ok(_frj["relacionados"] == [{"tipo": "revision_fiscal", "numero": "33/2024", "estado": "misma_sesion"}]
   and _frj["fuentes"].get("relacionados") == "secretario",
   "armar con el JSON del formulario (como por HTTP)")
ok(ft.a_formulario(_frj)["relacionados"] == "revision_fiscal|33/2024|misma_sesion"
   and json.loads(json.dumps(_frj, ensure_ascii=False, default=str))["relacionados"] == _frj["relacionados"],
   "la ficha armada vuelve a la tarjeta (retomar) y sobrevive al JSON de la sesión")
_frc = json.loads(json.dumps(_frj, ensure_ascii=False, default=str))
_frc["fuentes"]["relacionados"] = "auto"
_avn = ft.validar(_frc)
ok(_frc["relacionados"] == [] and any("SÓLO LOS MARCA EL SECRETARIO" in a for a in _avn),
   "validar: una ficha con relacionados que no vienen del secretario los pierde, con aviso")
_frn = {"tipo": "queja", "numero": "24/2026", "relacionados": [{"tipo": "queja", "numero": "24/2026"},
                                                                {"tipo": "amparo_revision", "numero": "298/2025"}]}
ft.validar(_frn)
ok(_frn["relacionados"] == [{"tipo": "amparo_revision", "numero": "298/2025", "estado": "misma_sesion"}],
   "validar: la ficha que no pasó por armar (banco) queda con la forma del contrato y sin el propio asunto")
ok(not any("(relacionados)" in a for a in ft.validar(ft.armar(_Erel, declarado=_decl_rel))),
   "idempotente: tras armar, validar ya no tiene nada que decir de los relacionados")
cli, comp = cliente_falso({"fecha_admision": "2025-01-20", "relacionados": "amparo_directo|1/2025|misma_sesion"})
_lam = asyncio.run(ft.leer_auto_modelo(cli, AUTO_ADMISION_AD, "amparo_directo"))
ok("relacionados" not in _lam and _d(_lam.get("admision")).get("fecha") == "2025-01-20",
   "leer_auto_modelo: aunque el modelo los diga, los relacionados no se leen del auto")

# ── 18.2 · C3: LA QUEJA LEE LA DEMANDA DEL AUTO RECURRIDO ───────────────────
# Q 24/2026 del banco: «Demanda de amparo. … promovió juicio de amparo
# indirecto en contra del Juzgado Séptimo Civil… de quien reclamó la ilegalidad
# de todo lo actuado durante el procedimiento de remate…». El auto que la
# desechó la describe.
JUZ_Q = ("Juzgado Primero de Distrito en Materia de Amparo Civil, Administrativo y de Trabajo y de Juicios "
         "Federales en el Estado de Querétaro")
ACTO_Q_DEM = (
    "JUZGADO PRIMERO DE DISTRITO EN MATERIA DE AMPARO CIVIL, ADMINISTRATIVO Y DE TRABAJO Y DE JUICIOS\n"
    "FEDERALES EN EL ESTADO DE QUERÉTARO\nJUICIO DE AMPARO INDIRECTO 1810/2025\n"
    "Querétaro, Querétaro, a once de diciembre de dos mil veinticinco.\n"
    "Visto el escrito de demanda presentado el ocho de diciembre de dos mil veinticinco ante la Oficina de "
    "Correspondencia Común, por José Luis Hernández Ramírez, por propio derecho, mediante el cual promueve juicio "
    "de amparo indirecto contra actos del Juez Séptimo Civil de Primera Instancia del Distrito Judicial de "
    "Querétaro y del Secretario de Acuerdos del Juzgado Séptimo Civil de Primera Instancia del Distrito Judicial "
    "de Querétaro, de quienes reclama la ilegalidad de todo lo actuado durante el procedimiento de remate llevado "
    "a cabo dentro del expediente 1520/2019 de su índice, así como la publicación del edicto de remate de catorce "
    "de noviembre de dos mil veinticuatro.\n"
    "Regístrese con el número 1810/2025. Del análisis de la demanda se advierte que se actualiza la causa de "
    "improcedencia prevista en el artículo 61, fracción XXIII, de la Ley de Amparo. En consecuencia, se desecha "
    "de plano la demanda de amparo. Notifíquese.\n"
    "Así lo proveyó y firma el Juez Primero de Distrito en Materia de Amparo Civil, Administrativo y de Trabajo y "
    "de Juicios Federales en el Estado de Querétaro, ante el Secretario que autoriza y da fe.")
_AUT_Q = ["Juez Séptimo Civil de Primera Instancia del Distrito Judicial de Querétaro",
          "Secretario de Acuerdos del Juzgado Séptimo Civil de Primera Instancia del Distrito Judicial de Querétaro"]
cli, comp = cliente_falso({
    "clase": "auto", "fecha": "2025-12-11", "organo": JUZ_Q, "expediente": "1810/2025", "via": "indirecto",
    "incidente": False, "sentido_clave": "desecha_demanda", "quejoso": "José Luis Hernández Ramírez",
    "demanda_fecha": "2025-12-08",
    "autoridades": _AUT_Q + [JUZ_Q, "Gobernador Constitucional del Estado de Querétaro"],
    "actos": ["la ilegalidad de todo lo actuado durante el procedimiento de remate llevado a cabo dentro del "
              "expediente 1520/2019 de su índice"]})
_qd = asyncio.run(ft.leer_acto(cli, ACTO_Q_DEM, "queja", "24/2026"))
_dq = _d(_qd.get("demanda"))
ok(_dq.get("fecha") == "2025-12-08" and _qd.get("quejoso") == "José Luis Hernández Ramírez",
   f"queja: la fecha de la demanda y el quejoso, del auto recurrido: {_dq.get('fecha')} · {_qd.get('quejoso')}")
ok(_dq.get("autoridades") == _AUT_Q,
   f"las autoridades de la demanda, ancladas; sin el propio juzgado ni la inventada: {_dq.get('autoridades')}")
ok(any("SE DESCARTÓ «Gobernador" in a for a in _qd["avisos"])
   and any("SE LEYÓ ENTRE LAS AUTORIDADES" in a for a in _qd["avisos"]),
   "…y se dice qué se quitó (la que no está en el papel; el juzgado que dictó el auto)")
ok(bool(_dq.get("actos")) and _dq["actos"][0].endswith("de catorce de noviembre de dos mil veinticuatro.")
   and _dq["actos"][0].startswith("la ilegalidad de todo lo actuado"),
   f"el acto cortado por el modelo se completa con el papel hasta su punto: «…{(_dq.get('actos') or [''])[0][-60:]}»")
ok(_qd["acto"].get("sentido") == "desechó la demanda de amparo" and _qd.get("fraccion_97") == "I"
   and _d(_qd.get("fuentes")).get("demanda.actos") == "acto",
   "lo de siempre, intacto (sentido del catálogo, fracción I) y la demanda con fuente «acto»")
_pq = comp.vistos[0]["messages"][0]["content"] if comp.vistos else ""
ok("NO es el escrito del recurso de queja" in _pq and "NUNCA es una de sus" in _pq
   and '"autoridades"' in _pq and '"actos"' in _pq and '"demanda_fecha"' in _pq,
   "el prompt de la queja pide la demanda y la separa del escrito de queja, del auto y del juzgado")
_q0 = asyncio.run(ft.leer_acto(None, ACTO_Q_DEM, "queja", "24/2026"))
ok(_d(_q0.get("demanda")) == {"fecha": "2025-12-08"},
   f"sin modelo, la fecha de la demanda sale con su fórmula («escrito de demanda presentado el…»): {_q0.get('demanda')}")
_qs = asyncio.run(ft.leer_acto(None, ACTO_Q, "queja", "50/2026"))
ok(not _qs.get("demanda") and not any("DEMANDA" in a.upper() and "FALTA" in a.upper() for a in _qs["avisos"]),
   "sin datos de la demanda, nada y SIN aviso (es contexto, no requisito)")
ok(ft._fecha_demanda_presentada("Visto el escrito de ampliación de demanda presentado el ocho de diciembre de "
                                "dos mil veinticinco.") == ""
   and ft._fecha_demanda_presentada("JUICIO 1/2025. Demanda de amparo. Querétaro, a once de diciembre de dos mil "
                                    "veinticinco.") == "",
   "la fecha de una ampliación o la del propio auto no son la de la demanda")
# El escrito de queja también la describe (C3, leer_escrito).
ESCRITO_Q = ("C. JUEZ PRIMERO DE DISTRITO PRESENTE. José Luis Hernández Ramírez, por mi propio derecho, quejoso en "
             "el juicio de amparo indirecto 1810/2025, ante usted comparezco para interponer recurso de queja contra "
             "el auto de once de diciembre de dos mil veinticinco, que desechó la demanda de amparo que presenté el "
             "ocho de diciembre de dos mil veinticinco contra actos del Juez Séptimo Civil de Primera Instancia del "
             "Distrito Judicial de Querétaro, de quien reclamé la ilegalidad de todo lo actuado en el procedimiento "
             "de remate del expediente 1520/2019.")
cli, comp = cliente_falso({"promovente": "José Luis Hernández Ramírez", "caracter": "quejoso",
                           "quejoso": "José Luis Hernández Ramírez", "demanda_fecha": "2025-12-08",
                           "autoridades": [_AUT_Q[0], "Presidente Municipal de Querétaro"],
                           "actos": ["la ilegalidad de todo lo actuado en el procedimiento de remate del "
                                     "expediente 1520/2019"]})
_eq = asyncio.run(ft.leer_escrito(cli, ESCRITO_Q, "queja"))
ok(_d(_eq.get("demanda")).get("fecha") == "2025-12-08" and _d(_eq.get("demanda")).get("autoridades") == [_AUT_Q[0]]
   and _d(_eq.get("demanda")).get("actos") == ["la ilegalidad de todo lo actuado en el procedimiento de remate del "
                                                "expediente 1520/2019."]
   and _eq.get("quejoso") == "José Luis Hernández Ramírez"
   and _d(_eq.get("fuentes")).get("demanda.fecha") == "escrito"
   and any("Presidente Municipal" in a for a in _eq["avisos"]),
   f"leer_escrito (queja): la demanda que el escrito describe, anclada; lo inventado fuera: {_eq.get('demanda')}")
ok(len(comp.vistos) == 1 and comp.vistos[0]["max_completion_tokens"] == 1500
   and "NO ES ESTE ESCRITO" in comp.vistos[0]["messages"][0]["content"]
   and _eq.get("caracter") == "quejoso" and _eq.get("promovente") == "José Luis Hernández Ramírez",
   "en la queja la llamada se hace aunque el carácter salga sin modelo; quién recurre, intacto")
# armar: la demanda del auto recurrido manda sobre la del escrito; el propio juzgado fuera.
_Eq = {"tipo_asunto": "queja", "numero": "24/2026", "materia": "civil"}
_fq = ft.armar(_Eq, acto=_qd, escrito=dict(_eq, demanda=dict(_d(_eq.get("demanda")), fecha="2025-12-05")))
ok(_d(_fq.get("demanda")).get("fecha") == "2025-12-08" and _fq["fuentes"].get("demanda.fecha") == "acto"
   and any("NO CUADRA ENTRE FUENTES" in a and "acto" in a for a in _fq["avisos"]),
   "armar: la fecha de la demanda del auto recurrido manda sobre la del escrito, y se dice")
ok(_d(_fq.get("demanda")).get("autoridades") == _AUT_Q and _fq.get("quejoso") == "José Luis Hernández Ramírez",
   "armar: las autoridades y la quejosa del auto recurrido llegan a la ficha de la queja")
_fq2j = ft.armar(_Eq, acto={"acto": {"organo": JUZ_Q, "fecha": "2025-12-11"},
                            "demanda": {"autoridades": [JUZ_Q] + _AUT_Q}})
ok(_d(_fq2j.get("demanda")).get("autoridades") == _AUT_Q
   and any("EL JUZGADO QUE DICTÓ EL AUTO RECURRIDO" in a for a in _fq2j["avisos"]),
   "armar (queja): el juzgado que dictó el auto sale de las autoridades de la demanda, con aviso")
_SALA_Q = "Primera Sala Civil del Tribunal Superior de Justicia del Estado de Querétaro"
_fq3 = ft.armar(_Eq, acto={"acto": {"organo": _SALA_Q, "via": "directo"}, "demanda": {"autoridades": [_SALA_Q]}})
ok(_d(_fq3.get("demanda")).get("autoridades") == [_SALA_Q],
   "fracción II: quien dictó el auto es la responsable del amparo directo y se queda")
_fq4 = ft.armar(_Eq, acto={"acto": {"organo": JUZ_Q}, "demanda": {"autoridades": [JUZ_Q]}})
ok(_fq4.get("demanda") == {} and not any(r.startswith("demanda.") for r in _fq4["fuentes"]),
   "si la única autoridad era el propio juzgado, la demanda queda vacía ({}), sin fuentes colgando")
_vq = {"tipo": "queja", "acto": {"fecha": "2025-12-11"}, "demanda": {"fecha": "2025-12-15"},
       "presentacion": "2025-12-12"}
_avq = ft.validar(_vq)
ok("demanda.fecha > acto.fecha" in _vq["fechas_imposibles"]
   and "demanda.fecha >= presentacion" in _vq["fechas_imposibles"]
   and any("posterior al auto recurrido" in a for a in _avq)
   and any("el recurso de queja" in a and "demanda de amparo" in a for a in _avq),
   "validar (queja): la demanda después del auto recurrido o del recurso es FECHA IMPOSIBLE")
_var = {"tipo": "amparo_revision", "acto": {"fecha": "2025-12-11"}, "demanda": {"fecha": "2025-12-15"}}
ok(any(a.startswith("FECHA IMPOSIBLE: la demanda de amparo (") and a.endswith("es posterior a la resolución "
                                                                                "recurrida (" + a.split("(")[-1])
       for a in ft.validar(_var)),
   "(el aviso del amparo en revisión, con su texto de siempre)")

print("\n19 · REVISIÓN ADVERSARIAL (3-oct-2026): LA AMPLIACIÓN NO ES LA DEMANDA; SIN EXISTENCIA EN EL AR")
# Q 300/2025: el 97-I-a cubre «su ampliación», y el auto que provee sobre ella la
# describe; su fecha pasaba el anclaje (está en el papel) como la de la demanda.
_amp = ("VISTO el escrito de ampliación de demanda presentado el veinte de agosto de dos mil veinticinco por "
        "Juan Pérez López, contra actos del Director de Obras, se admite. La demanda de amparo se presentó el diez "
        "de julio de dos mil veinticinco.")
ok(ft._fecha_de_ampliacion("2025-08-20", _amp) and not ft._fecha_de_ampliacion("2025-07-10", _amp)
   and not ft._fecha_de_ampliacion("2025-07-11", _amp),
   "la fecha que sólo aparece tras «ampliación» es la de la ampliación; la de la demanda inicial, no")
_av_a, _out_a = [], {}
ft._demanda_anclada({"demanda_fecha": "2025-08-20"}, _amp, _av_a, _out_a)
ok("fecha" not in _d(_out_a.get("demanda")) and any("AMPLIACIÓN de demanda" in a for a in _av_a),
   "la demanda anclada descarta la fecha de la ampliación, con aviso")
_av_b, _out_b = [], {}
ft._demanda_anclada({"demanda_fecha": "2025-07-10"}, _amp, _av_b, _out_b)
ok(_d(_out_b.get("demanda")).get("fecha") == "2025-07-10" and not _av_b, "…y conserva la de la demanda inicial")
ok("LA AMPLIACIÓN DE DEMANDA NO ES LA DEMANDA" in ft._prompt_acto("queja", "auto", "", "300/2025")
   and "LA AMPLIACIÓN DE DEMANDA NO ES LA DEMANDA" in ft._prompt_escrito("queja", "escrito")
   and "LA AMPLIACIÓN DE DEMANDA" not in ft._prompt_acto("amparo_revision", "sentencia", "", "410/2026"),
   "los dos prompts de la queja (auto y escrito) dicen que la ampliación no es la demanda")
# C1 QUITÓ LA EXISTENCIA DEL AR: el aviso ya no manda a firmarla.
_v448 = {"tipo": "amparo_revision", "clase_recurrida": "sentencia",
         "acto": {"fecha": "2025-06-10", "incidente": True, "clase": "sentencia"}}
_a448 = ft.validar(_v448)
ok(any("LA RESOLUCIÓN RECURRIDA ES DEL INCIDENTE DE SUSPENSIÓN" in a and "la competencia y la procedencia" in a
       for a in _a448) and not any("existencia" in a for a in _a448),
   "AR del incidente con clase «sentencia»: el aviso pide comprobar la competencia y la procedencia, no la existencia")

print()
if FALLOS:
    print(f"FALLAN {len(FALLOS)}: " + " · ".join(FALLOS))
    raise SystemExit(1)
print("RESULTADO: TODAS LAS COMPROBACIONES PASAN")
