# -*- coding: utf-8 -*-
"""LA VERJA PROCESAL (3-oct-2026), sin red.

    .venv/bin/python test_verja_procesal.py

Lo que se prueba, en este orden:

1. Los cuatro tipos BIEN compuestos (con las fórmulas del SPEC §2 y los
   considerandos del banco ya arreglados) no llevan NI UN aviso.
2. Cada regla, (a) a (k), acusa su error y nombra el dato y el apartado.
3. Nunca lanza: texto vacío, ficha None, tipo desconocido, texto de una pieza.
4. `arreglar` sólo toca tipografía segura y es idempotente.
5. CALIBRACIÓN CONTRA LAS SENTENCIAS DEL TRIBUNAL (si el corpus está en este
   equipo): 41 versiones públicas recientes (AD, AR, Q y RF de 2025-2026). «Si
   acusa a las sentencias buenas, la que está mal es la verificación»: lo único
   que se admite es la lista exacta de hallazgos que se revisaron uno por uno y
   son errores del propio engrose (LOPJF abrogada, AG 1/2023 abrogado, una fecha
   del acto distinta entre resultando y resolutivo, «localizado» con la Sala).
6. Si `resultandos_por_tipo` ya existe, lo que compone pasa la verja limpio.
7. RONDA 3 (3-oct-2026): lo que el banco de oráculo (32 sentencias reales) y la
   prueba de punta a punta (AD 274, AR 631) encontraron sin aviso: el ponente
   del último returno, los femeninos en -dora/-tora/-sora y con prefijo
   temporal, las versales tras «lo hizo valer», la etiqueta de rol, la carátula
   sin número, «X, en representación de X», la clase del acto (l), la forma de
   notificación (m), el supletorio con «supletoriamente» o el art. 2o., la sede
   que decide `tipos_asunto`, la cronología del amparo indirecto en el AR, el
   renglón RECURRENTE, y el hueco con su clave (agrupado y sin repetir lo que
   ya avisó el compositor).
8. RONDA 4 (3-oct-2026, rev/verif_final.txt y FIXES_R4 E2, E7, E8 y E11): el
   órgano que decidió el compositor manda (Q 261) y el que descartó no puede
   seguir en la competencia ni en la existencia (AR 307, 448, 60); la
   cronología que ya dijo validar no se repite (AR 448); «por Conducto de Su»,
   «por Propio Derecho», «a Través de Su», «y Otra»; la racha en versales
   dentro de la mención; el colectivo sin artículo; «X, por conducto de su X»
   y el cargo igual a su órgano (RF 7, 2, 26 y 6-2026); el plural del
   resultando contra el singular de la legitimación y la carátula (AD 552); la
   clave del auto que registra (registro.fecha); el prefijo temporal de la
   Sala (RF 49 y 21); y el adhesivo sin ningún auto (E5, AD 274).
9. RONDA 5 (3-oct-2026, rev/verif_final2.txt y FIXES_R5 F1, F3, F4 y F5): la
   vía en hueco con el número detrás no es «falta el número» (Q 335 sin
   fracción del 97: la clave es fraccion_97 y se calla si el compositor ya lo
   avisó); «X, por conducto de su X» con la adscripción escrita distinto, por
   el núcleo de `tipos_asunto.misma_autoridad` (RF 7); la forma de notificación
   sin papel (fuente «omision» u «omision_autoridad») que la oportunidad
   afirma (Q 335, AD 274 y 335); el artículo del cargo de doble género que
   contradice el del papel (AD 128); y (n), el par de palabras repetido
   («solicitud de solicitud de», RF 4). Las 41 sentencias: 0 avisos nuevos.
10. SEXTA RONDA (3-oct-2026, las respuestas de David, contrato C1-C8): las
   formas nuevas pasan sin avisos falsos —el AR sin existencia y con
   «Procedencia.», el punto del amparo que nombra acto y autoridad (ni órgano
   recurrido, ni clase de lo recurrido, ni fecha del acto), la queja sin el
   renglón del órgano y con «Demanda de amparo.» primero (y su demanda antes
   del auto, j), la revisión fiscal con «RECURRENTE:», el renglón «RELACIONADO
   CON …» que no es el número del asunto, la supletoriedad agraria (C7)—; el
   marcador «{actos_del_amparo}» sin sustituir (a); y las reglas nuevas de los
   asuntos relacionados: (o) el propio asunto, (p) el 64 de la LFPCA fuera del
   par AD ↔ RF y (q) el rubro, el V I S T O y el considerando contra sí mismos y
   contra la ficha. Las 41 sentencias: 0 avisos nuevos.
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import verja_procesal as vp

FALLOS = []


def ok(c, q):
    print(f"   {'PASA ' if c else 'FALLA'}  {q}")
    if not c:
        FALLOS.append(q)


def reglas(texto, ficha=None, datos=None, tipo=""):
    return [x["regla"] for x in vp.revisar_detalle(texto, ficha, datos, tipo)]


def avisos_de(texto, ficha=None, datos=None, tipo="", regla=None):
    return [x["aviso"] for x in vp.revisar_detalle(texto, ficha, datos, tipo)
            if regla is None or x["regla"] == regla]


TRIB = "Tercer Tribunal Colegiado en Materias Administrativa y Civil del Vigésimo Segundo Circuito"
SESION = ("El presente asunto se listó el *********, para verse en sesión ordinaria de ********* siguiente; "
          "lo anterior, de conformidad con el Acuerdo General 6/2026, del Pleno del Órgano de Administración "
          "Judicial que regula la integración y trámite del expediente electrónico y el uso de videoconferencias "
          "en todos los asuntos competencia de los órganos jurisdiccionales a cargo del Órgano, publicado en el "
          "Diario Oficial de la Federación el diecisiete de abril de dos mil veintiséis; y,")
COLA_COMPETENCIA = ("así como el punto tercero, fracción XXII, del Acuerdo General 3/2013, en relación con el "
                    "artículo 1 del diverso 28/2017, ambos del Pleno del otrora Consejo de la Judicatura Federal.")
DISPENSA = ("Es innecesario transcribir el contenido de la resolución combatida y los argumentos hechos valer en su "
            "contra, pues el deber formal y material de exponer los argumentos legales que sustenten esta "
            "resolución no depende de su reproducción literal, sino de su adecuado análisis. Sustenta esa "
            "consideración la jurisprudencia 2a./J. 58/2010 de la Segunda Sala de la Suprema Corte de Justicia de "
            "la Nación.")

# ═══════════════════════════════════════════════════════════════════════════
# LOS CUATRO TIPOS, BIEN COMPUESTOS
# ═══════════════════════════════════════════════════════════════════════════
SALA = "Segunda Sala Civil del Tribunal Superior de Justicia del Estado de Querétaro"
AD = f"""AMPARO DIRECTO CIVIL: 174/2026.
QUEJOSA: RUTH GABRIELA LARRAÑAGA SILVA.
TERCERO INTERESADO: ALEJANDRO DANIEL BETANCOURT VELARDE.
AUTORIDAD RESPONSABLE: {SALA.upper()}.
MAGISTRADO PONENTE: LUIS ARMANDO PÉREZ TOPETE.
SECRETARIO: JOSÉ DAVID ALCÁNTAR MENDOZA.
Querétaro, Querétaro. Resolución del {TRIB}, correspondiente a la sesión de *********.
V I S T O, para resolver el juicio de amparo directo civil 174/2026, promovido por Ruth Gabriela Larrañaga Silva, contra la sentencia dictada el veintidós de enero de dos mil veintiséis, por la {SALA}, en el toca familiar 4357/2025, derivado del expediente 515/2022; y,
R E S U L T A N D O
PRIMERO. Presentación de la demanda de amparo. Por escrito presentado el doce de febrero de dos mil veintiséis ante la {SALA}, Ruth Gabriela Larrañaga Silva promovió juicio de amparo directo en contra de la autoridad y del acto que a continuación se precisan:
AUTORIDAD RESPONSABLE:
{SALA}
ACTO RECLAMADO:
La sentencia dictada el veintidós de enero de dos mil veintiséis, en el toca familiar 4357/2025, derivado del expediente 515/2022.
SEGUNDO. Derechos humanos que se estiman vulnerados. La parte quejosa señaló como tales los contenidos en los artículos 1o., 4o., 14 y 16 de la Constitución Política de los Estados Unidos Mexicanos.
TERCERO. Tercero interesado. Tiene ese carácter Alejandro Daniel Betancourt Velarde.
CUARTO. Trámite del juicio de amparo. Por auto de Presidencia de dos de marzo de dos mil veintiséis, este Tribunal Colegiado registró la demanda con el número 174/2026, la admitió a trámite y concedió a las partes el plazo de quince días para formular alegatos o promover amparo adhesivo, en términos del artículo 181 de la Ley de Amparo.
QUINTO. Turno. Por acuerdo de veinte de abril de dos mil veintiséis, se turnaron los autos a la ponencia de Luis Armando Pérez Topete, para la elaboración del proyecto de resolución, en términos del artículo 183 de la Ley de Amparo.
SEXTO. Celebración de la sesión vía remota. {SESION}
C O N S I D E R A N D O
PRIMERO. Competencia. Este {TRIB} es competente para conocer y resolver el presente juicio de amparo directo, por haberse promovido en contra de una sentencia definitiva en materia civil, dictada por la {SALA}, localizada en la circunscripción territorial en la que este órgano colegiado ejerce jurisdicción, con fundamento en los artículos 103, fracción I y 107, fracción V, inciso c), de la Constitución Federal; 33, fracción II, 34 y 170, fracción I, párrafo primero, de la Ley de Amparo; 35, fracción I, inciso c) y 210 de la Ley Orgánica del Poder Judicial de la Federación; {COLA_COMPETENCIA}
SEGUNDO. Existencia del acto reclamado. La existencia del acto reclamado está acreditada con el informe justificado rendido por la {SALA}, certeza que se corrobora con los autos del toca familiar 4357/2025 y del expediente 515/2022, que acompañó al referido informe. Documentales que en términos de los artículos 129 y 202 del Código Federal de Procedimientos Civiles, de aplicación supletoria a la Ley de Amparo, merecen eficacia probatoria plena.
TERCERO. Legitimación y oportunidad. La demanda de amparo fue promovida por Ruth Gabriela Larrañaga Silva, quien está legitimada para ello conforme al artículo 5o., fracción I, de la Ley de Amparo, por resentir el perjuicio que le causa la sentencia reclamada.
Por cuanto hace a la oportunidad, la sentencia reclamada se notificó a la quejosa el veintiséis de enero de dos mil veintiséis de manera personal y surtió efectos al día hábil siguiente, conforme al artículo 126 del Código de Procedimientos Civiles del Estado de Querétaro, es decir, el veintisiete de enero de dos mil veintiséis, por lo que el plazo de quince días del artículo 17 de la Ley de Amparo fue del veintiocho de enero al dieciocho de febrero de dos mil veintiséis, sin contar sábados y domingos, ni el dos de febrero de dos mil veintiséis, inhábil en términos del artículo 19 de la Ley de Amparo; si se presentó el doce de febrero de dos mil veintiséis, fue oportuna.
CUARTO. Sentencia reclamada y conceptos de violación. {DISPENSA}
QUINTO. Estudio.
R E S U E L V E
ÚNICO. La Justicia de la Unión no ampara ni protege a Ruth Gabriela Larrañaga Silva, contra la sentencia dictada el veintidós de enero de dos mil veintiséis, por la {SALA}, en el toca familiar 4357/2025."""
FICHA_AD = {
    "formato": 1, "tipo": "amparo_directo", "numero": "174/2026", "materia": "civil",
    "sede": {"tribunal": TRIB, "ciudad": "Querétaro", "circuito": "XXII", "cdmx": False},
    "admision": {"fecha": "2026-03-02"}, "turno": {"fecha": "2026-04-20", "ponente": "Luis Armando Pérez Topete"},
    "ministerio_publico": "", "promovente": "Ruth Gabriela Larrañaga Silva", "caracter": "quejoso",
    "presentacion": "2026-02-12", "notificacion": "2026-01-26", "forma_notificacion": "personal",
    "acto": {"clase": "sentencia", "fecha": "2026-01-22", "organo": SALA, "toca": "4357/2025",
             "expediente": "515/2022"},
    "responsable": SALA, "terceros": ["Alejandro Daniel Betancourt Velarde"], "derechos": ["1o.", "4o.", "14", "16"],
    "avisos": [], "fuentes": {}, "fechas_imposibles": [],
}
DATOS_AD = {"numero": "174/2026", "tribunal": TRIB, "ciudad": "Querétaro, Querétaro",
            "quejoso": "Ruth Gabriela Larrañaga Silva", "responsable": SALA, "tipo_asunto": "amparo_directo"}

JUZ = ("Juzgado Quinto de Distrito en Materia de Amparo Civil, Administrativo y de Trabajo y de Juicios Federales "
       "en el Estado de Querétaro")
COM = "Comercializadora Ejemplo, S.A. de C.V."
AR = f"""AMPARO EN REVISIÓN ADMINISTRATIVA: 410/2026.
QUEJOSA Y RECURRENTE: {COM.upper()}.
MAGISTRADA PONENTE: JENICA CAMPOS JUÁREZ.
SECRETARIO: PABLO SERGIO VARGAS QUIROGA.
Querétaro, Querétaro. Resolución del {TRIB}, correspondiente a la sesión de *********.
V I S T O, para resolver el recurso de revisión 410/2026, interpuesto por {COM}, contra la sentencia dictada el diez de abril de dos mil veintiséis, por el {JUZ}, en el juicio de amparo indirecto 950/2025; y,
R E S U L T A N D O
PRIMERO. Presentación de la demanda de amparo indirecto. Por escrito presentado el tres de noviembre de dos mil veinticinco, {COM}, promovió juicio de amparo indirecto en contra de las autoridades y los actos que a continuación se señalan:
AUTORIDAD RESPONSABLE:
Director de Ingresos del Municipio de Querétaro.
ACTOS RECLAMADOS:
La determinación del crédito fiscal contenida en el oficio DI/123/2025, que le concedió un plazo para pagar.
SEGUNDO. Trámite del juicio de amparo indirecto. El conocimiento del asunto correspondió al {JUZ}, que lo registró con el número 950/2025 y admitió la demanda. Seguido el juicio por sus etapas, el diez de abril de dos mil veintiséis se celebró la audiencia constitucional y se dictó sentencia, en la que se negó el amparo.
TERCERO. Interposición y trámite del recurso de revisión. Inconforme, {COM}, interpuso recurso de revisión por escrito presentado el veintisiete de abril de dos mil veintiséis. Por auto de Presidencia de once de mayo de dos mil veintiséis, este Tribunal Colegiado lo registró con el número 410/2026 y lo admitió a trámite.
CUARTO. Turno. Por acuerdo de uno de junio de dos mil veintiséis, se turnaron los autos a la ponencia de Jenica Campos Juárez, para la elaboración del proyecto de resolución, en términos del artículo 92 de la Ley de Amparo.
QUINTO. Verificación de la sesión vía remota. {SESION}
C O N S I D E R A N D O
PRIMERO. Competencia. Este {TRIB} es competente para conocer y resolver el presente recurso de revisión, interpuesto contra una sentencia dictada en un juicio de amparo indirecto en materia administrativa, por el {JUZ}, localizado en la circunscripción territorial en la que este órgano colegiado ejerce jurisdicción, con fundamento en los artículos 107, fracción VIII, último párrafo, de la Constitución Federal; 81, fracción I, inciso e) y 84 de la Ley de Amparo; 35, fracción V y 210 de la Ley Orgánica del Poder Judicial de la Federación; {COLA_COMPETENCIA}
SEGUNDO. Existencia de la resolución recurrida. La existencia de la sentencia recurrida está acreditada con los autos originales del juicio de amparo indirecto 950/2025, que remitió el {JUZ} en términos del artículo 89 de la Ley de Amparo; documentales que, en términos de los artículos 129 y 202 del Código Federal de Procedimientos Civiles, de aplicación supletoria a la Ley de Amparo, merecen eficacia probatoria plena.
TERCERO. Legitimación y oportunidad. El recurso fue interpuesto por {COM}, parte quejosa, quien está legitimada para ello en términos del artículo 5o., fracción I, de la Ley de Amparo. La sentencia recurrida se notificó a la quejosa el trece de abril de dos mil veintiséis, y surtió efectos el catorce siguiente, conforme al artículo 31, fracción II, de la Ley de Amparo; el plazo de diez días del artículo 86 corrió del quince al veintiocho de abril de dos mil veintiséis; si se presentó el veintisiete de abril de dos mil veintiséis, es oportuno.
CUARTO. Sentencia recurrida y agravios. {DISPENSA}
QUINTO. Estudio.
R E S U E L V E
PRIMERO. En la materia de la revisión, se confirma la sentencia recurrida.
SEGUNDO. La Justicia de la Unión no ampara ni protege a {COM}, contra el acto que reclamó del Director de Ingresos del Municipio de Querétaro, precisado en el resultando primero de esta ejecutoria."""
FICHA_AR = {
    "formato": 1, "tipo": "amparo_revision", "numero": "410/2026", "materia": "administrativa",
    "sede": {"tribunal": TRIB, "ciudad": "Querétaro", "circuito": "XXII", "cdmx": False},
    "admision": {"fecha": "2026-05-11"}, "turno": {"fecha": "2026-06-01", "ponente": "Jenica Campos Juárez"},
    "ministerio_publico": "", "promovente": COM, "caracter": "quejoso",
    "presentacion": "2026-04-27", "notificacion": "2026-04-13",
    "acto": {"clase": "sentencia", "fecha": "2026-04-10", "organo": JUZ, "expediente": "950/2025",
             "incidente": False, "resolvio": "niega"},
    "demanda": {"fecha": "2025-11-03", "autoridades": ["Director de Ingresos del Municipio de Querétaro"],
                "actos": ["La determinación del crédito fiscal contenida en el oficio DI/123/2025"]},
    "audiencia": "2026-04-10", "clase_recurrida": "sentencia", "fechas_imposibles": [],
}
DATOS_AR = {"numero": "410/2026", "tribunal": TRIB, "ciudad": "Querétaro", "quejoso": COM,
            "organo_recurrido": JUZ, "responsable": "Director de Ingresos del Municipio de Querétaro"}

JUZ3 = JUZ.replace("Quinto", "Tercero")
Q = f"""RECURSO DE QUEJA CIVIL: 355/2026.
RECURRENTE: JUAN PÉREZ LÓPEZ.
ÓRGANO QUE DICTÓ EL AUTO RECURRIDO: {JUZ3.upper()}.
MAGISTRADO PONENTE: LUIS ARMANDO PÉREZ TOPETE.
SECRETARIO: JOSÉ DAVID ALCÁNTAR MENDOZA.
Querétaro, Querétaro. Resolución del {TRIB}, correspondiente a la sesión de *********.
V I S T O, para resolver el recurso de queja civil 355/2026, interpuesto por Juan Pérez López, en contra del auto de dos de junio de dos mil veintiséis, dictado por el {JUZ3}, en el incidente de suspensión relativo al juicio de amparo indirecto 1201/2026; y,
R E S U L T A N D O
PRIMERO. Interposición del recurso de queja. Por escrito presentado el cuatro de junio de dos mil veintiséis, Juan Pérez López, quejoso en el juicio de amparo indirecto 1201/2026, interpuso recurso de queja en contra del auto de dos de junio de dos mil veintiséis, dictado por el {JUZ3}, en el incidente de suspensión, en el que negó la suspensión provisional.
SEGUNDO. Trámite del recurso. Por auto de Presidencia de nueve de junio de dos mil veintiséis, este Tribunal Colegiado registró el recurso con el número 355/2026 y lo admitió a trámite.
TERCERO. Turno del asunto. Por acuerdo de once de junio de dos mil veintiséis, se turnaron los autos a la ponencia de Luis Armando Pérez Topete, para la elaboración del proyecto de resolución.
CUARTO. Verificación de la sesión vía remota. {SESION}
C O N S I D E R A N D O
PRIMERO. Competencia. Este {TRIB} es competente para conocer y resolver el presente recurso de queja, por haberse interpuesto en contra de un auto en el que se negó la suspensión provisional, dictado por el {JUZ3}, localizado en la circunscripción territorial en que ejerce jurisdicción este órgano colegiado, con fundamento en los artículos 103 y 107 de la Constitución Federal; 97, fracción I, inciso b), de la Ley de Amparo; 35, fracción III y 210 de la Ley Orgánica del Poder Judicial de la Federación; {COLA_COMPETENCIA}
SEGUNDO. Procedencia. El recurso de queja es procedente en términos del artículo 97, fracción I, inciso b), de la Ley de Amparo, porque se interpone contra el auto que negó la suspensión provisional.
TERCERO. Legitimación y oportunidad. El recurso fue interpuesto por Juan Pérez López, quejoso en el juicio de amparo indirecto, por lo que está legitimado para ello. El auto recurrido se le notificó el tres de junio de dos mil veintiséis y el plazo de dos días del artículo 98, fracción I, de la Ley de Amparo transcurrió el cuatro y cinco de junio; si el recurso se presentó el cuatro de junio de dos mil veintiséis, es oportuno.
CUARTO. Auto recurrido y agravios. {DISPENSA}
QUINTO. Estudio.
R E S U E L V E
ÚNICO. Es infundado el recurso de queja interpuesto por Juan Pérez López, contra el auto de dos de junio de dos mil veintiséis, dictado por el {JUZ3}, en el incidente de suspensión relativo al juicio de amparo indirecto 1201/2026."""
FICHA_Q = {
    "formato": 1, "tipo": "queja", "numero": "355/2026", "materia": "civil",
    "sede": {"tribunal": TRIB, "ciudad": "Querétaro", "circuito": "XXII", "cdmx": False},
    "admision": {"fecha": "2026-06-09"}, "turno": {"fecha": "2026-06-11", "ponente": "Luis Armando Pérez Topete"},
    "promovente": "Juan Pérez López", "caracter": "quejoso", "presentacion": "2026-06-04",
    "notificacion": "2026-06-03",
    "acto": {"clase": "auto", "fecha": "2026-06-02", "organo": JUZ3, "expediente": "1201/2026", "incidente": True,
             "sentido": "negó la suspensión provisional"},
    "fraccion_97": "I", "inciso_97": "b", "fechas_imposibles": [],
}
DATOS_Q = {"numero": "355/2026", "tribunal": TRIB, "ciudad": "Querétaro", "quejoso": "Juan Pérez López",
           "organo_recurrido": JUZ3}

SALA_RF = "Sala Regional en Querétaro"
UJ = "Titular de la Unidad Jurídica de la Delegación Estatal en Querétaro del Instituto de Seguridad y Servicios Sociales de los Trabajadores del Estado"
RF = f"""REVISIÓN FISCAL: 42/2026.
AUTORIDAD RECURRENTE: {UJ.upper()}.
PARTE ACTORA: MARÍA GÓMEZ RUIZ.
SALA RESPONSABLE: {SALA_RF.upper()} DEL TRIBUNAL FEDERAL DE JUSTICIA ADMINISTRATIVA.
MAGISTRADO PONENTE: LUIS ARMANDO PÉREZ TOPETE.
SECRETARIA: NORMA ANGÉLICA GUERRERO SANTILLÁN.
Querétaro, Querétaro. Resolución del {TRIB}, correspondiente a la sesión de *********.
V I S T O S, para resolver el recurso de revisión fiscal número 42/2026, interpuesto por la parte citada al rubro, contra la sentencia de seis de febrero de dos mil veintiséis, dictada por la {SALA_RF} del Tribunal Federal de Justicia Administrativa, en el juicio contencioso administrativo 1234/25-22-01-3; y,
R E S U L T A N D O
PRIMERO. Trámite del juicio contencioso administrativo. María Gómez Ruiz demandó la nulidad de la resolución contenida en el oficio 09-52-07-1234/2025, de quince de julio de dos mil veinticinco, emitida por el Titular de la Delegación Estatal en Querétaro del Instituto de Seguridad y Servicios Sociales de los Trabajadores del Estado. El conocimiento correspondió a la {SALA_RF} del Tribunal Federal de Justicia Administrativa, que lo registró con el número 1234/25-22-01-3; seguido el juicio, el seis de febrero de dos mil veintiséis dictó sentencia, en la que declaró la nulidad de la resolución impugnada, para determinados efectos.
SEGUNDO. Interposición del recurso de revisión fiscal. Inconforme, el {UJ}, en representación de la autoridad demandada, interpuso recurso de revisión fiscal mediante oficio presentado el catorce de abril de dos mil veintiséis ante la {SALA_RF} del Tribunal Federal de Justicia Administrativa.
TERCERO. Trámite del recurso de revisión fiscal. Por auto de Presidencia de seis de mayo de dos mil veintiséis, este Tribunal Colegiado registró el recurso con el número 42/2026 y lo admitió a trámite.
CUARTO. Turno. Por acuerdo de doce de junio de dos mil veintiséis, se turnaron los autos a la ponencia de Luis Armando Pérez Topete, para la elaboración del proyecto de resolución, en términos del artículo 92 de la Ley de Amparo, aplicable conforme al artículo 63, último párrafo, de la Ley Federal de Procedimiento Contencioso Administrativo.
QUINTO. Celebración de la sesión vía remota. {SESION}
C O N S I D E R A N D O
PRIMERO. Competencia. Este {TRIB} es competente para conocer y resolver el presente recurso de revisión fiscal, toda vez que la sentencia recurrida fue dictada en un juicio contencioso administrativo tramitado ante la {SALA_RF} del Tribunal Federal de Justicia Administrativa, localizada en la circunscripción territorial en que ejerce jurisdicción este órgano colegiado, con fundamento en los artículos 104, fracción III, de la Constitución Federal; 35, fracción VI y 210 de la Ley Orgánica del Poder Judicial de la Federación; 63, párrafo primero, de la Ley Federal de Procedimiento Contencioso Administrativo; {COLA_COMPETENCIA}
SEGUNDO. Legitimación y oportunidad. El recurso lo interpone el {UJ}, unidad encargada de la defensa jurídica de la autoridad demandada, en términos del artículo 63 de la Ley Federal de Procedimiento Contencioso Administrativo. La sentencia se le notificó el veinticinco de marzo de dos mil veintiséis; si el recurso se presentó el catorce de abril de dos mil veintiséis, dentro de los quince días, es oportuno.
TERCERO. Procedencia. El recurso es procedente en términos del artículo 63, fracción III, de la Ley Federal de Procedimiento Contencioso Administrativo.
CUARTO. Sentencia recurrida y agravios. {DISPENSA}
QUINTO. Estudio.
R E S U E L V E
ÚNICO. Se confirma la sentencia de seis de febrero de dos mil veintiséis, dictada por la {SALA_RF} del Tribunal Federal de Justicia Administrativa, en el juicio contencioso administrativo 1234/25-22-01-3."""
FICHA_RF = {
    "formato": 1, "tipo": "revision_fiscal", "numero": "42/2026", "materia": "administrativa",
    "sede": {"tribunal": TRIB, "ciudad": "Querétaro", "circuito": "XXII", "cdmx": False},
    "admision": {"fecha": "2026-05-06"}, "turno": {"fecha": "2026-06-12", "ponente": "Luis Armando Pérez Topete"},
    "promovente": UJ, "caracter": "autoridad", "presentacion": "2026-04-14", "notificacion": "2026-03-25",
    "acto": {"clase": "sentencia", "fecha": "2026-02-06", "organo": SALA_RF,
             "sentido": "declaró la nulidad de la resolución impugnada, para determinados efectos"},
    "actora": "María Gómez Ruiz",
    "resolucion_impugnada": {"oficio": "09-52-07-1234/2025", "fecha": "2025-07-15",
                             "autoridad": "Titular de la Delegación Estatal en Querétaro del ISSSTE"},
    "autoridad_demandada": "Titular de la Delegación Estatal en Querétaro del ISSSTE",
    "sala": SALA_RF, "expediente_tfja": "1234/25-22-01-3", "fechas_imposibles": [],
}
DATOS_RF = {"numero": "42/2026", "tribunal": TRIB, "ciudad": "Querétaro", "quejoso": UJ, "organo_recurrido": SALA_RF}

# CDMX, con amparo adhesivo, returno y pedimento del MP que SÍ consta.
TRIB_CDMX = "Décimo Tribunal Colegiado en Materia Civil del Primer Circuito"
SALA_CDMX = "Tercera Sala Civil del Tribunal Superior de Justicia de la Ciudad de México"
AD_CDMX = (AD.replace(TRIB, TRIB_CDMX).replace("Querétaro, Querétaro. Resolución", "Ciudad de México. Resolución")
           .replace(SALA.upper(), SALA_CDMX.upper()).replace(SALA, SALA_CDMX)
           .replace("conforme al artículo 126 del Código de Procedimientos Civiles del Estado de Querétaro",
                    "conforme al artículo 120 del Código de Procedimientos Civiles para el Distrito Federal")
           .replace("los artículos 129 y 202 del Código Federal de Procedimientos Civiles, de aplicación supletoria "
                    "a la Ley de Amparo,",
                    "los artículos 312, fracciones II y VIII, y 344 del Código Nacional de Procedimientos Civiles y "
                    "Familiares, de aplicación supletoria a la Ley de Amparo conforme a su artículo 2o.,")
           .replace("fracción XXII, del Acuerdo", "fracción I, del Acuerdo")
           .replace("en términos del artículo 181 de la Ley de Amparo.",
                    "en términos del artículo 181 de la Ley de Amparo. El agente del Ministerio Público de la "
                    "Federación adscrito formuló pedimento. Por auto de diecisiete de marzo de dos mil veintiséis, se "
                    "admitió el amparo adhesivo promovido por Alejandro Daniel Betancourt Velarde.")
           # RONDA 3: con returno, la carátula lleva al ponente del returno (antes
           # el fixture conservaba al del turno, y la verja nueva lo acusa con razón).
           .replace("MAGISTRADO PONENTE: LUIS ARMANDO PÉREZ TOPETE.", "MAGISTRADO PONENTE: ISMAEL CAMACHO HERRERA.")
           .replace("SEXTO. Celebración", "SEXTO. Returno. Por acuerdo de cuatro de mayo de dos mil veintiséis, se "
                    "returnaron los autos a la ponencia de Ismael Camacho Herrera, para la elaboración del proyecto de "
                    "resolución.\nSÉPTIMO. Celebración")
           .replace("CUARTO. Sentencia reclamada y conceptos de violación.",
                    "CUARTO. Legitimación y oportunidad del amparo adhesivo. Alejandro Daniel Betancourt Velarde, "
                    "parte que obtuvo sentencia favorable, está legitimado para promover amparo adhesivo en términos "
                    "del artículo 182, primer párrafo, de la Ley de Amparo, y lo hizo dentro de los quince días del "
                    "artículo 181.\nQUINTO. Sentencia reclamada y conceptos de violación.")
           .replace("QUINTO. Estudio.", "SEXTO. Estudio.")
           .replace("ÚNICO. La Justicia", "PRIMERO. La Justicia")
           + "\nSEGUNDO. Se declara sin materia el amparo adhesivo promovido por Alejandro Daniel Betancourt Velarde.")
FICHA_CDMX = dict(FICHA_AD, responsable=SALA_CDMX, ministerio_publico="pedimento",
                  sede={"tribunal": TRIB_CDMX, "ciudad": "Ciudad de México", "circuito": "I", "cdmx": True},
                  acto=dict(FICHA_AD["acto"], organo=SALA_CDMX),
                  adhesivo={"quien": "Alejandro Daniel Betancourt Velarde", "admision": "2026-03-17"},
                  returno={"fecha": "2026-05-04", "ponente": "Ismael Camacho Herrera"})
DATOS_CDMX = dict(DATOS_AD, tribunal=TRIB_CDMX, ciudad="Ciudad de México", responsable=SALA_CDMX)

# AR MIXTO: el juzgado sobreseyó en una parte y negó en otra; queda firme.
AR_MIXTO = (AR.replace("en la que se negó el amparo.",
                       "en la que en una parte se sobreseyó en el juicio y en otra se negó el amparo.")
            .replace("PRIMERO. En la materia de la revisión, se confirma la sentencia recurrida.",
                     "PRIMERO. Queda firme el sobreseimiento decretado en la sentencia recurrida.\n"
                     "SEGUNDO. En la materia de la revisión, se confirma la sentencia recurrida.")
            .replace("SEGUNDO. La Justicia", "TERCERO. La Justicia"))
FICHA_AR_MIXTO = dict(FICHA_AR, acto=dict(FICHA_AR["acto"], resolvio="mixto"))

print("\n1 · LOS CUATRO TIPOS BIEN COMPUESTOS NO LLEVAN AVISOS")
for nombre, texto, ficha, datos, tipo in (
        ("amparo directo (XXII)", AD, FICHA_AD, DATOS_AD, "amparo_directo"),
        ("amparo directo (CDMX, adhesivo, returno, pedimento)", AD_CDMX, FICHA_CDMX, DATOS_CDMX, "amparo_directo"),
        ("amparo en revisión (niega)", AR, FICHA_AR, DATOS_AR, "amparo_revision"),
        ("amparo en revisión (mixto, queda firme)", AR_MIXTO, FICHA_AR_MIXTO, DATOS_AR, "amparo_revision"),
        ("queja fr. I, b)", Q, FICHA_Q, DATOS_Q, "queja"),
        ("revisión fiscal", RF, FICHA_RF, DATOS_RF, "revision_fiscal")):
    av = vp.revisar(texto, ficha, datos, tipo)
    ok(av == [], f"{nombre}: cero avisos" + (f" — salió: {av[:2]}" if av else ""))
    av = vp.revisar(texto, None, datos, tipo)
    ok(av == [], f"{nombre}, SIN ficha: tampoco" + (f" — salió: {av[:2]}" if av else ""))
s = vp.secciones(AD)
ok([x["zona"] for x in s][:3] == ["caratula", "proemio", "visto"], "las zonas: carátula, proemio, V I S T O…")
ok(any("competencia" in x["etiquetas"] for x in s) and any("existencia" in x["etiquetas"] for x in s)
   and any(x["zona"] == "resolutivos" for x in s), "…competencia, existencia y resolutivos, por su rótulo")
ok(vp.revisar(AD.replace("\n", " "), FICHA_AD, DATOS_AD, "amparo_directo") == [],
   "el mismo AD en una sola pieza (sin saltos de renglón) tampoco lleva avisos")


def con(texto, viejo, nuevo):
    assert viejo in texto, viejo[:60]
    return texto.replace(viejo, nuevo)


print("\n2 · (a) HUECOS FUERA DE LAS FECHAS DE LA SESIÓN")
t = con(AD, "inciso c), de la Constitución", "inciso *********), de la Constitución")
a = avisos_de(t, FICHA_AD, DATOS_AD, "amparo_directo", "a")
ok(len(a) == 1 and "COMPETENCIA" in a[0] and "el inciso" in a[0], "el inciso en hueco: se nombra el dato y el apartado")
t = con(AD, "Por acuerdo de veinte de abril de dos mil veintiséis, se turnaron",
        "Por acuerdo de *********, se turnaron")
a = avisos_de(t, FICHA_AD, DATOS_AD, "amparo_directo", "a")
ok(len(a) == 1 and "TURNO" in a[0] and "la fecha del auto" in a[0], "la fecha del turno en hueco")
t = con(AD, "AUTORIDAD RESPONSABLE: SEGUNDA", "AUTORIDAD RESPONSABLE: *********.\nX: SEGUNDA")
ok(any("CARÁTULA" in x and "órgano responsable" in x for x in avisos_de(t, FICHA_AD, DATOS_AD, "amparo_directo", "a")),
   "la responsable en hueco en la carátula")
t = con(AD, "La Justicia de la Unión no ampara ni protege", "La Justicia de la Unión ********* a")
ok(any("RESOLUTIVOS" in x and "sentido del fallo" in x for x in avisos_de(t, FICHA_AD, DATOS_AD, "amparo_directo", "a")),
   "«La Justicia de la Unión ********* a…» (AR 631 v2): falta el sentido del fallo")
ok("a" not in reglas(AD, FICHA_AD, DATOS_AD, "amparo_directo"),
   "los tres huecos de la sesión (proemio, lista y sesión) no se acusan")
# Ronda 2 (3-oct-2026): los campos nuevos de la ficha —fundamento_surtimiento,
# fraccion_63, cuantia—. El hueco dice en qué campo va el dato; si la ficha ya
# lo traía, dice que lo perdió una pieza del camino.
t = con(AD, "conforme al artículo 126 del Código de Procedimientos Civiles del Estado de Querétaro",
        "conforme al *********")
a = avisos_de(t, FICHA_AD, DATOS_AD, "amparo_directo", "a")
ok(len(a) == 1 and "surtimiento" in a[0] and "«fundamento_surtimiento»" in a[0] and "SÍ LO TRAE" not in a[0],
   "el surtimiento en hueco, sin el dato en la ficha: se nombra el campo «fundamento_surtimiento»")
a = avisos_de(t, dict(FICHA_AD, fundamento_surtimiento="el artículo 126 del Código de Procedimientos Civiles del "
                                                       "Estado de Querétaro"), DATOS_AD, "amparo_directo", "a")
ok(len(a) == 1 and "SÍ LO TRAE" in a[0] and "artículo 126" in a[0],
   "…y con el dato en la ficha: el aviso dice que se perdió en el camino")
t = con(RF, "artículo 63, fracción III, de la Ley", "artículo 63, fracción *********, de la Ley")
a = avisos_de(t, FICHA_RF, DATOS_RF, "revision_fiscal", "a")
ok(len(a) == 1 and "PROCEDENCIA" in a[0] and "«fraccion_63»" in a[0], "RF: la fracción del 63 en hueco nombra «fraccion_63»")
a = avisos_de(t, dict(FICHA_RF, cuantia="$2,350,000.00"), DATOS_RF, "revision_fiscal", "a")
ok(len(a) == 1 and "cuantía" in a[0] and "SÍ LO TRAE" not in a[0],
   "…con la cuantía y sin la fracción: se dice, sin culpar al camino (la cuantía sola no decide la fracción)")
a = avisos_de(t, dict(FICHA_RF, fraccion_63="III", cuantia="$2,350,000.00"), DATOS_RF, "revision_fiscal", "a")
ok(len(a) == 1 and "SÍ LO TRAE (fraccion_63: «III»)" in a[0], "…con la fracción en la ficha: se perdió en el camino")
ok(not avisos_de(RF, dict(FICHA_RF, fraccion_63="III", cuantia="$2,350,000.00",
                          fundamento_surtimiento="el artículo 65 de la Ley Federal de Procedimiento "
                                                 "Contencioso Administrativo"), DATOS_RF, "revision_fiscal"),
   "los tres campos nuevos en la ficha no fabrican avisos sobre un texto completo")

print("\n3 · (b) FÓRMULAS EVASIVAS Y AFIRMACIONES SIN FUENTE")
t = con(AD, "Por acuerdo de veinte de abril de dos mil veintiséis, se turnaron",
        "En la fecha que se advierte de las constancias, se turnaron")
a = avisos_de(t, FICHA_AD, DATOS_AD, "amparo_directo", "b")
ok(len(a) == 1 and "TURNO" in a[0], "«en la fecha que se advierte de las constancias» (79 de 81): un aviso, no dos")
t = con(AD, "Por acuerdo de veinte de abril", "Por acuerdo de fecha que obra en autos, el veinte de abril")
ok(any("obra en autos" in x for x in avisos_de(t, FICHA_AD, DATOS_AD, "amparo_directo", "b")),
   "«por acuerdo de fecha que obra en autos»")
t = con(AD, "Tiene ese carácter Alejandro Daniel Betancourt Velarde.",
        "Tiene ese carácter la persona a quien resulte, en los términos que obran en autos.")
ok(len(avisos_de(t, FICHA_AD, DATOS_AD, "amparo_directo", "b")) == 2, "«la persona a quien resulta…» y «en los términos que obran en autos»")
mp = con(AD, "en términos del artículo 181 de la Ley de Amparo.",
         "en términos del artículo 181 de la Ley de Amparo; el agente del Ministerio Público adscrito omitió formular pedimento.")
ok(any("AFIRMACIÓN SIN FUENTE" in x for x in avisos_de(mp, FICHA_AD, DATOS_AD, "amparo_directo", "b")),
   "el MP «omitió formular pedimento» sin que conste (71 de 72 AD)")
ok(not avisos_de(mp, dict(FICHA_AD, ministerio_publico="sin_pedimento"), DATOS_AD, "amparo_directo", "b"),
   "si la ficha dice «sin_pedimento», decirlo es correcto")
ok(any("SÍ hubo pedimento" in x for x in avisos_de(mp, dict(FICHA_AD, ministerio_publico="pedimento"), DATOS_AD,
                                                    "amparo_directo", "b")), "y si dice que hubo pedimento, es contradicción")
t = con(AD, "Por escrito presentado el doce de febrero de dos mil veintiséis ante la",
        "Por escrito presentado el doce de febrero de dos mil veintiséis en la Oficialía de Partes de este Tribunal, contra la")
ok(any("OFICIALÍA INVENTADA" in x for x in avisos_de(t, FICHA_AD, DATOS_AD, "amparo_directo", "b")),
   "«Oficialía de Partes de este Tribunal» en el AD (art. 176)")
ok(not any("OFICIALÍA" in x for x in avisos_de(t, dict(FICHA_AD, via_presentacion="tribunal"), DATOS_AD,
                                               "amparo_directo", "b")), "…salvo que el secretario declare que llegó aquí")
t = con(AD, "La parte quejosa señaló como tales los contenidos en los artículos 1o., 4o., 14 y 16 de la Constitución "
            "Política de los Estados Unidos Mexicanos.",
        "La parte quejosa alegó la violación de los artículos constitucionales que precisó en su demanda de amparo.")
ok(any("PERÍFRASIS" in x and "DERECHOS" in x for x in avisos_de(t, FICHA_AD, DATOS_AD, "amparo_directo", "b")),
   "el resultando de derechos sin un solo artículo (57 de 72)")
t = con(AD, "conforme al artículo 126 del Código de Procedimientos Civiles del Estado de Querétaro",
        "conforme a la ley del acto")
ok(any("ley del acto" in x for x in avisos_de(t, FICHA_AD, DATOS_AD, "amparo_directo", "b")),
   "«surtió efectos… conforme a la ley del acto» sin el precepto (64 de 72)")

print("\n4 · (c) FECHAS SIN FUENTE")
t = con(AD, "Por auto de Presidencia de dos de marzo", "Por auto de Presidencia de tres de marzo")
a = avisos_de(t, FICHA_AD, DATOS_AD, "amparo_directo", "c")
ok(len(a) == 1 and "tres de marzo de dos mil veintiséis" in a[0] and "TRÁMITE" in a[0],
   "una fecha del trámite que no está en la ficha ni en el acto")
ok(not avisos_de(t, FICHA_AD, dict(DATOS_AD, acto="…el tres de marzo de dos mil veintiséis se acordó…"),
                 "amparo_directo", "c"), "…pero si está en el texto del acto, tiene fuente")
ok(not avisos_de(t, None, DATOS_AD, "amparo_directo", "c"), "sin ficha no se coteja (no hay contra qué)")
t = con(AD, "ÚNICO. La Justicia de la Unión no ampara ni protege a Ruth Gabriela Larrañaga Silva, contra la sentencia "
            "dictada el veintidós de enero", "ÚNICO. La Justicia de la Unión no ampara ni protege a Ruth Gabriela "
            "Larrañaga Silva, contra la sentencia dictada el dos de marzo")
ok(any("RESOLUTIVOS" in x for x in avisos_de(t, None, DATOS_AD, "amparo_directo", "c")),
   "el resolutivo fechado con el auto de Presidencia (RF Yucatán del mapa), aun sin ficha")

print("\n5 · (d) LA RESPONSABLE Y EL TRIBUNAL, DE UNA SOLA FORMA")
t = con(AD, f"AUTORIDAD RESPONSABLE: {SALA.upper()}.", "AUTORIDAD RESPONSABLE: TRIBUNAL EL SEIS.")
a = avisos_de(t, FICHA_AD, DATOS_AD, "amparo_directo", "d")
ok(len(a) == 1 and "TRIBUNAL EL SEIS" in a[0] and "carátula" in a[0], "«TRIBUNAL EL SEIS» en la carátula (103/2025)")
t = con(AD, f"dictada por la {SALA}, localizada", "dictada por el Juzgado Tercero de Primera Instancia Familiar del "
            "Distrito Judicial de Querétaro, localizado")
ok(any("Competencia" in x and "otro órgano" in x for x in avisos_de(t, FICHA_AD, DATOS_AD, "amparo_directo", "d")),
   "el juzgado de primera instancia en vez de la Sala, en la competencia (174 y 722)")
t = con(AD, f"rendido por la {SALA},", "rendido por la Segunda Sala Civil del Tribunal Superior de Justicia del Estado,")
ok(any("le falta el final" in x for x in avisos_de(t, FICHA_AD, DATOS_AD, "amparo_directo", "d")),
   "«…del Estado» sin la entidad en la existencia (640, 296)")
sin_ent = "Sala Familiar del Tribunal Superior de Justicia en el Estado"
t = AD.replace(SALA.upper(), sin_ent.upper()).replace(SALA, sin_ent)
ok(any("SIN ENTIDAD" in x for x in avisos_de(t, dict(FICHA_AD, responsable=sin_ent), dict(DATOS_AD, responsable=sin_ent),
                                             "amparo_directo", "d")), "la ficha misma sin entidad federativa")
t = con(AD, f"Competencia. Este {TRIB}", f"Competencia. Este {TRIB.upper()}")
ok(any("VERSALES" in x for x in avisos_de(t, FICHA_AD, DATOS_AD, "amparo_directo", "d")),
   "«Este TERCER TRIBUNAL COLEGIADO…» en versales en la competencia")
t = con(AD, f"Competencia. Este {TRIB}", f"Competencia. Este el {TRIB}")
ok(any("ARTÍCULO DUPLICADO" in x for x in avisos_de(t, FICHA_AD, DATOS_AD, "amparo_directo", "d")), "«Este el Tercer…»")
t = con(AD, f"Competencia. Este {TRIB}", "Competencia. Este Primer Tribunal Colegiado en Materias Administrativa y "
            "Civil del Vigésimo Segundo Circuito")
ok(any("TRIBUNAL ESCRITO DE DOS FORMAS" in x for x in avisos_de(t, FICHA_AD, DATOS_AD, "amparo_directo", "d")),
   "el tribunal del proemio y el de la competencia no coinciden")
t = con(AR, "que remitió el", "certeza que se corrobora con el informe justificado rendido por la MAGISTRADA LAURA "
            "ANGÉLICA LÓPEZ DE LA FUENTE, INTEGRANTE DE LA PRIMERA SALA CIVIL, y con los autos que remitió el")
ok(any("VERSALES" in x for x in avisos_de(t, FICHA_AR, DATOS_AR, "amparo_revision", "d")),
   "un nombre en VERSALES en mitad de la prosa (AR 631)")
# Ronda 2: la cola «del Tribunal Federal de Justicia Administrativa» no es otro
# nombre, ni en el texto ni en el dato (el compositor midió 2 avisos falsos por
# RF con la ficha en versales y sin la cola).
ok(not avisos_de(RF, dict(FICHA_RF, sala=SALA_RF.upper(), acto=dict(FICHA_RF["acto"], organo=SALA_RF.upper())),
                 dict(DATOS_RF, organo_recurrido=""), "revision_fiscal", "d"),
   "RF: ficha.sala en versales y sin la cola del TFJA, el texto con ella: sin aviso")
_SALA_TFJA = SALA_RF + " del Tribunal Federal de Justicia Administrativa"
_rf_corto = RF.replace(f"{SALA_RF} del Tribunal Federal de Justicia Administrativa", SALA_RF)
ok(_rf_corto != RF and not avisos_de(_rf_corto, dict(FICHA_RF, sala=_SALA_TFJA), DATOS_RF, "revision_fiscal", "d"),
   "RF: al revés, la ficha con la cola y el texto sin ella: sin aviso")
t = con(RF, f"dictada por la {SALA_RF} del Tribunal", "dictada por la Sala Regional del Centro II del Tribunal")
ok(any("V I S T O" in x and "otro órgano" in x for x in avisos_de(t, FICHA_RF, DATOS_RF, "revision_fiscal", "d")),
   "RF: otra Sala en el V I S T O sigue acusándose")
# Ronda 2: EL PONENTE de la carátula y el del turno (5 de los 13 documentos
# integrados: «MAGISTRADO PONENTE: MAGISTRADO EJEMPLO» y «a la ponencia de la
# Magistrada Ejemplo»). El género sale del cargo escrito, no del nombre.
t = con(AD, "a la ponencia de Luis Armando Pérez Topete", "a la ponencia de Jenica Campos Juárez")
a = avisos_de(t, FICHA_AD, DATOS_AD, "amparo_directo", "d")
ok(len(a) == 1 and "EL PONENTE NO ES EL MISMO" in a[0] and "Jenica Campos Juárez" in a[0],
   "el turno a otra persona que la de la carátula, sin returno: se acusa")
t = con(AD, "a la ponencia de Luis Armando Pérez Topete", "a la ponencia de la Magistrada Luis Armando Pérez Topete")
a = avisos_de(t, FICHA_AD, DATOS_AD, "amparo_directo", "d")
ok(len(a) == 1 and "EL CARGO DEL PONENTE NO CONCUERDA" in a[0],
   "«MAGISTRADO PONENTE» en la carátula y «la Magistrada…» en el turno: el cargo no concuerda")
ok(not avisos_de(con(AD, "a la ponencia de Luis Armando Pérez Topete", "al Magistrado Luis Armando Pérez Topete"),
                 FICHA_AD, DATOS_AD, "amparo_directo", "d"), "«al Magistrado X» con «MAGISTRADO PONENTE: X»: sin aviso")
ok(not avisos_de(con(AD, "a la ponencia de Luis Armando Pérez Topete", "a la ponencia del Magistrado Pérez Topete"),
                 FICHA_AD, DATOS_AD, "amparo_directo", "d"), "el turno con el nombre abreviado: sin aviso")
ok(not avisos_de(con(AD_CDMX, "a la ponencia de Luis Armando Pérez Topete", "a la ponencia de Jenica Campos Juárez"),
                 FICHA_CDMX, DATOS_CDMX, "amparo_directo", "d"),
   "con returno de por medio, el ponente del turno puede ser otro: no se coteja")

print("\n6 · (e) NÚMEROS: el del asunto, el toca y el expediente")
t = con(AD, "amparo directo civil 174/2026, promovido", "amparo directo civil 147/2026, promovido")
ok(any("147/2026" in x for x in avisos_de(t, FICHA_AD, DATOS_AD, "amparo_directo", "e")), "el V I S T O con otro número")
t = con(AD, "amparo directo civil 174/2026, promovido", "amparo directo civil ADC 625-2024 ORAL MERCANTIL, promovido")
ok(any("ETIQUETA" in x and "ADC 625-2024" in x for x in avisos_de(t, FICHA_AD, DATOS_AD, "amparo_directo", "e")),
   "la etiqueta del encabezado colada en el V I S T O (27 de 81)")
t = con(AD, "con el número 174/2026, la admitió", "con el número ADA 174/2026, la admitió")
ok(any("ETIQUETA" in x for x in avisos_de(t, FICHA_AD, DATOS_AD, "amparo_directo", "e")),
   "…y en el trámite («se registró la demanda con el número ADA 103/2025»)")
t = con(AD, "con los autos del toca familiar 4357/2025 y del expediente 515/2022",
        "con los autos del expediente 4357/2025")
ok(any("TOCA" in x for x in avisos_de(t, FICHA_AD, DATOS_AD, "amparo_directo", "e")),
   "la existencia llama «expediente» al toca (174/2026)")
ok(any("TOCA" in x for x in avisos_de(t, None, dict(DATOS_AD, responsable=""), "amparo_directo", "e")),
   "…también sin ficha: el toca lo dice el propio V I S T O")
t = con(AR, "acreditada con los autos originales", "acreditada con el informe justificado rendido por el Director de "
            "Ingresos del Municipio de Querétaro, certeza que se corrobora con los autos originales")
ok(any("EXISTENCIA MAL PLANTEADA" in x for x in avisos_de(t, FICHA_AR, DATOS_AR, "amparo_revision", "e")),
   "AR: la existencia por el informe justificado de la responsable (9 de 9)")

print("\n7 · (f) CONCORDANCIAS")
t = Q.replace(f"dictado por el {JUZ3}, localizado", "dictado por el Jueza Tercero de Distrito en el Estado de Querétaro, localizado")
ok(any("masculino ante un nombre femenino" in x for x in avisos_de(t, FICHA_Q, DATOS_Q, "queja", "f")), "«por el Jueza»")
t = con(AD, f"dictada por la {SALA}, localizada", f"dictada por la {SALA}, localizado")
ok(any("localizado" in x for x in avisos_de(t, FICHA_AD, DATOS_AD, "amparo_directo", "f")), "«por la Sala…, localizado»")
t = con(AD, "se notificó a la quejosa el", "se notificó a la quejoso el")
ok(any("no concuerda con la parte" in x for x in avisos_de(t, FICHA_AD, DATOS_AD, "amparo_directo", "f")), "«la quejoso»")
t = con(AR, "contra el acto que reclamó del Director", "en contra el acto que reclamó del Director")
ok(any("en contra el" in x for x in avisos_de(t, FICHA_AR, DATOS_AR, "amparo_revision", "f")),
   "«en contra el acto» (8 de 9 AR)")
t = con(AD, "dictada por la Segunda Sala Civil del Tribunal Superior de Justicia del Estado de Querétaro, localizada",
        "dictada por el Tribunal Unitario Agrario del Distrito 42, con sede en la Santiago de Querétaro, localizado")
ok(any("nombre de la ciudad" in x for x in avisos_de(t, FICHA_AD, DATOS_AD, "amparo_directo", "f")),
   "«con sede en la Santiago de Querétaro» (263/2025)")
t = con(AD, "quien está legitimada para ello", "quien está legitimado para ello")
ok(any("legitimado" in x for x in avisos_de(t, FICHA_AD, DATOS_AD, "amparo_directo", "f")),
   "«legitimado» con la QUEJOSA de la carátula")
t2 = t.replace("QUEJOSA: RUTH", "PARTE QUEJOSA: RUTH")
ok(not avisos_de(t2, FICHA_AD, DATOS_AD, "amparo_directo", "f"),
   "…pero sin género declarado no se infiere del nombre de pila")
t = con(AR, "quien está legitimada para ello", "quien está legitimado para ello")
ok(any("legitimado" in x for x in avisos_de(t, FICHA_AR, DATOS_AR, "amparo_revision", "f")),
   "«legitimado» con una persona moral de cabeza femenina («Comercializadora…»)")
tr = RF.replace("El recurso lo interpone el {}, unidad encargada".format(UJ),
                "El recurso lo interpone el {}, quien está legitimado como unidad encargada".format(UJ))
ok(not avisos_de(tr, FICHA_RF, DATOS_RF, "revision_fiscal", "f"),
   "«el Titular de la Unidad Jurídica…, quien está legitimado»: «titular» no dice el género y «unidad» no manda")
ok(any("legitimado" in x for x in avisos_de(
    AR.replace(f"interpuesto por {COM}, parte quejosa, quien está legitimada",
               "interpuesto por el Director de Ingresos del Municipio de Querétaro, autoridad responsable, quien está "
               "legitimada"), FICHA_AR, dict(DATOS_AR, recurrente="Director de Ingresos del Municipio de Querétaro"),
    "amparo_revision", "f")), "«Director de Ingresos…, quien está legitimada» (mapa AR XXII): el cargo manda")
ok(not avisos_de(AD.replace("Ruth Gabriela Larrañaga Silva, quien está legitimada",
                            "Banco Nacional de México, S.A., quien está legitimada"), FICHA_AD,
                 dict(DATOS_AD, quejoso="Banco Nacional de México, S.A."), "amparo_directo", "f"),
   "«Banco…, S.A.» admite las dos concordancias: no se acusa")

print("\n8 · (g) SUPLETORIEDAD Y SEDE")
t = con(AD, "los artículos 129 y 202 del Código Federal de Procedimientos Civiles, de aplicación supletoria a la Ley "
            "de Amparo,", "los artículos 312, fracciones II y VIII, y 344 del Código Nacional de Procedimientos "
            "Civiles y Familiares, de aplicación supletoria a la Ley de Amparo conforme a su artículo 2o.,")
a = avisos_de(t, FICHA_AD, DATOS_AD, "amparo_directo", "g")
ok(len(a) == 1 and "EXISTENCIA" in a[0], "el CNPCF en Querétaro (18 de 18 desde el 27-sep)")
ok(len(avisos_de(t, None, dict(DATOS_AD, tribunal="", ciudad=""), "amparo_directo", "g")) == 1,
   "…también sin datos de sede: se lee del proemio")
t = AD_CDMX.replace("los artículos 312, fracciones II y VIII, y 344 del Código Nacional de Procedimientos Civiles y "
                    "Familiares, de aplicación supletoria a la Ley de Amparo conforme a su artículo 2o.,",
                    "los artículos 129 y 202 del Código Federal de Procedimientos Civiles, de aplicación supletoria a "
                    "la Ley de Amparo,")
ok(len(avisos_de(t, FICHA_CDMX, DATOS_CDMX, "amparo_directo", "g")) == 1, "el CFPC en la Ciudad de México")
ok(not avisos_de(RF.replace("63, párrafo primero", "63, párrafo primero, y el Código Federal de Procedimientos Civiles, "
                                                   "de aplicación supletoria a la Ley Federal de Procedimiento "
                                                   "Contencioso Administrativo"),
                 dict(FICHA_RF, sede=dict(FICHA_RF["sede"], cdmx=True)), DATOS_RF, "revision_fiscal", "g"),
   "el CFPC supletorio de la LFPCA no es el de la Ley de Amparo: no se acusa ni en CDMX")

print("\n9 · (h) NORMAS ABROGADAS O FUERA DE LUGAR")
t = con(AD, "35, fracción I, inciso c) y 210 de la Ley Orgánica", "38, fracción III y 124 de la Ley Orgánica")
ok(any("38, 124" in x for x in avisos_de(t, FICHA_AD, DATOS_AD, "amparo_directo", "h")), "LOPJF 38 y 124 en la competencia")
t = con(RF, "R E S U E L V E", "Por lo expuesto, fundado y con apoyo en los artículos 104, fracción III, de la "
            "Constitución Federal, 63 de la Ley Federal de Procedimiento Contencioso Administrativo, y 37, fracción "
            "V, de la Ley Orgánica del Poder Judicial de la Federación; se,\nR E S U E L V E")
ok(any("artículo 37" in x for x in avisos_de(t, FICHA_RF, DATOS_RF, "revision_fiscal", "h")),
   "«37, fracción V, de la LOPJF» en la antesala de los resolutivos (13 RF)")
ok(not avisos_de(t.replace("Ley Orgánica del Poder Judicial de la Federación;",
                           "Ley Orgánica del Poder Judicial de la Federación abrogada;"), FICHA_RF, DATOS_RF,
                 "revision_fiscal", "h"), "…pero si se cita como la abrogada, aplicable, no se acusa")
t = con(AR, COLA_COMPETENCIA, COLA_COMPETENCIA + " Así como en el punto quinto, fracción I, inciso B), del Acuerdo "
            "General 1/2023 del Pleno de la Suprema Corte de Justicia de la Nación.")
ok(any("1/2023" in x for x in avisos_de(t, FICHA_AR, DATOS_AR, "amparo_revision", "h")), "el AG 1/2023, abrogado")
t = con(AD, "ambos del Pleno del otrora Consejo", "ambos del Pleno del Consejo")
ok(any("OTRORA" in x for x in avisos_de(t, FICHA_AD, DATOS_AD, "amparo_directo", "h")), "«Pleno del Consejo» sin «otrora»")
t = con(AD, "lo anterior, de conformidad con el Acuerdo General 6/2026", "lo anterior, de conformidad con el artículo "
            "84 de la Ley de Amparo y el Acuerdo General 6/2026")
ok(any("184" in x for x in avisos_de(t, FICHA_AD, DATOS_AD, "amparo_directo", "h")), "«artículo 84» en la sesión")
t = con(AD, "en términos del artículo 183 de la Ley de Amparo.", "en términos del artículo 101 de la Ley de Amparo.")
ok(any("101" in x and "183" in x for x in avisos_de(t, FICHA_AD, DATOS_AD, "amparo_directo", "h")),
   "el 101 en el turno del AD (es el de la queja)")
tq = con(Q, "para la elaboración del proyecto de resolución.\n", "para la elaboración del proyecto de resolución, "
         "conforme al artículo 101 de la Ley de Amparo.\n")
ok(not avisos_de(tq, FICHA_Q, DATOS_Q, "queja", "h"), "…en la queja el 101 sí es su artículo")

print("\n10 · (i) EN QUÉ PARÓ EL JUICIO (AR)")
t = con(AR, "en la que se negó el amparo.", "en la que se concedió el amparo.")
ok(any("NO CUADRA" in x for x in avisos_de(t, FICHA_AR, DATOS_AR, "amparo_revision", "i")),
   "el resultando dice «concedió» y la sentencia negó")
t = con(AR_MIXTO, "PRIMERO. Queda firme el sobreseimiento decretado en la sentencia recurrida.\n", "")
ok(any("SOBRESEIMIENTO NO SE RESUELVE" in x for x in avisos_de(t, FICHA_AR_MIXTO, DATOS_AR, "amparo_revision", "i")),
   "mixto sin «queda firme el sobreseimiento» (8 de 9 AR)")
ok(not avisos_de(AR, dict(FICHA_AR, acto=dict(FICHA_AR["acto"], incidente=True, resolvio="concede")), DATOS_AR,
                 "amparo_revision", "i"), "en el incidente de suspensión no se coteja el verbo del fondo")
# Ronda 2: las claves compuestas de fase_rama son MIXTOS, no sobreseimientos.
for _clave in ("sobresee_niega", "sobresee niega", "Sobresee y niega"):
    ok(not avisos_de(AR_MIXTO, dict(FICHA_AR, acto=dict(FICHA_AR["acto"], resolvio=_clave)), DATOS_AR,
                     "amparo_revision", "i"), f"acto.resolvio = {_clave!r} en crudo se lee como mixto: sin aviso")
ok(not avisos_de(AR_MIXTO, dict(FICHA_AR, acto=dict(FICHA_AR["acto"], resolvio="mixto", resolvio_mixto="sobresee_niega")),
                 DATOS_AR, "amparo_revision", "i"), "«mixto» + resolvio_mixto (la forma de ficha_tramite): sin aviso")
ok(not avisos_de(AR_MIXTO, {}, dict(DATOS_AR, resolvio_a_quo="sobresee_niega"), "amparo_revision", "i"),
   "sin ficha, resolvio_a_quo = «sobresee_niega»: sin aviso")
a = avisos_de(AR_MIXTO, dict(FICHA_AR, acto=dict(FICHA_AR["acto"], resolvio="sobresee_concede")), DATOS_AR,
              "amparo_revision", "i")
ok(len(a) == 1 and "NO CUADRA" in a[0] and "concedió el amparo" in a[0],
   "mixto que concedió y el resultando dice que negó: se acusa, con el fondo")
a = avisos_de(AR_MIXTO, dict(FICHA_AR, acto=dict(FICHA_AR["acto"], resolvio="mixto", resolvio_mixto="sobresee_concede")),
              DATOS_AR, "amparo_revision", "i")
ok(len(a) == 1 and "NO CUADRA" in a[0], "…también con «mixto» + resolvio_mixto = «sobresee_concede»")
t = con(AR_MIXTO, "PRIMERO. Queda firme el sobreseimiento decretado en la sentencia recurrida.\n", "")
ok(any("SOBRESEIMIENTO NO SE RESUELVE" in x for x in avisos_de(t, dict(FICHA_AR, acto=dict(FICHA_AR["acto"],
                                                                                             resolvio="sobresee_niega")),
                                                                 DATOS_AR, "amparo_revision", "i")),
   "con la clave cruda, el mixto sin «queda firme el sobreseimiento» también se acusa")
ok(any("NO CUADRA" in x for x in avisos_de(AR, dict(FICHA_AR, acto=dict(FICHA_AR["acto"], resolvio="sobresee_niega")),
                                          DATOS_AR, "amparo_revision", "i")),
   "la clave mixta contra un resultando que sólo dice «negó» (falta el sobreseimiento): se acusa")

print("\n11 · (j) FECHAS IMPOSIBLES")
ok(any("acto posterior" in x for x in avisos_de(AD, dict(FICHA_AD, fechas_imposibles=["acto posterior a la demanda"]),
                                                DATOS_AD, "amparo_directo", "j")), "las de la ficha salen tal cual")
t = AD.replace("veintidós de enero de dos mil veintiséis", "veintidós de enero de dos mil veintisiete")
ok(any("antes de que existiera" in x for x in avisos_de(t, None, DATOS_AD, "amparo_directo", "j")),
   "una sentencia posterior a la demanda (174 y 722, 17 proyectos)")
t = con(AD, "Por escrito presentado el doce de febrero de dos mil veintiséis", "Por escrito presentado el doce de "
            "febrero de dos mil veintisiete")
ok(any("174/2026" in x and "año anterior" in x for x in avisos_de(t, None, DATOS_AD, "amparo_directo", "j")),
   "un asunto registrado en un año anterior a su presentación (590/2024 presentado en 2025)")
t = con(AD, "se notificó a la quejosa el veintiséis de enero", "se notificó a la quejosa el tres de enero")
viejo = (AD.split("PRIMERO. Presentación")[0] + "PRIMERO. Presentación de la demanda de amparo. El catorce de marzo "
         "de dos mil veinticinco, Ruth Gabriela Larrañaga Silva promovió juicio de amparo directo contra la sentencia de "
         "veintidós de enero de dos mil veintiséis, dictada por la Segunda Sala Civil en el toca familiar 4357/2025.\n"
         "SEGUNDO. Derechos" + AD.split("SEGUNDO. Derechos")[1])
ok(any("antes de que existiera" in x for x in avisos_de(viejo, None, DATOS_AD, "amparo_directo", "j")),
   "la fórmula del camino viejo: «El {fecha}, X promovió… contra la sentencia de {fecha}»")
ok(any("notificación" in x for x in avisos_de(t, None, DATOS_AD, "amparo_directo", "j")),
   "notificada antes de dictarse")
t = con(Q, "en contra del auto de dos de junio de dos mil veintiséis, dictado por el {}, en el incidente de suspensión, "
           "en el que".format(JUZ3), "en contra del auto de nueve de junio de dos mil veintiséis, dictado por el {}, en "
                                     "el incidente de suspensión, en el que".format(JUZ3))
ok(any("antes de que existiera" in x for x in avisos_de(t, None, DATOS_Q, "queja", "j")),
   "queja contra un auto posterior a su interposición")

print("\n12 · (k) EL ADHESIVO QUE CONSTA Y NO SE TRATA")
f = dict(FICHA_AD, adhesivo={"quien": "Alejandro Daniel Betancourt Velarde", "admision": "2026-03-17"})
a = avisos_de(AD, f, DATOS_AD, "amparo_directo", "k")
ok(len(a) == 1 and "resultandos" in a[0] and "considerandos" in a[0] and "resolutivos" in a[0],
   "AD con adhesivo en la ficha y sin tratarlo: los tres sitios (722/2025)")
t = con(AD, "amparo directo civil 174/2026, promovido", "amparo directo civil 174/2026 y su adhesivo, promovido")
ok(bool(avisos_de(t, None, DATOS_AD, "amparo_directo", "k")), "…y sin ficha, si el V I S T O lo nombra")
ok(not avisos_de(AD_CDMX, FICHA_CDMX, DATOS_CDMX, "amparo_directo", "k"), "tratado en los tres sitios: sin aviso")
# Ronda 2: el verbo conjugado va con «i» («adhiriéndose», «se adhirió»). La
# fórmula del compositor de la RF («se tuvo a X adhiriéndose al recurso») daba
# un aviso falso en integ1/RF_xxii.
_ADH_RF = {"quien": "María Gómez Ruiz", "notificacion": "2026-05-08", "presentacion": "2026-05-20",
           "admision": "2026-05-22"}
RF_ADH = (con(RF, "lo admitió a trámite.\n", "lo admitió a trámite.\nPor auto de veintidós de mayo de dos mil "
                                            "veintiséis, se tuvo a María Gómez Ruiz adhiriéndose al recurso.\n")
          .replace("CUARTO. Sentencia recurrida y agravios.",
                   "CUARTO. Legitimación y oportunidad de la revisión adhesiva. María Gómez Ruiz tiene legitimación "
                   "para adherirse al recurso, en términos del artículo 63, penúltimo párrafo, de la Ley Federal de "
                   "Procedimiento Contencioso Administrativo, y lo hizo en tiempo.\nQUINTO. Sentencia recurrida y "
                   "agravios.")
          .replace("QUINTO. Estudio.", "SEXTO. Estudio.")
          .replace("ÚNICO. Se confirma", "PRIMERO. Se confirma")
          + "\nSEGUNDO. Se declara sin materia la revisión adhesiva interpuesta por María Gómez Ruiz.")
FICHA_RF_ADH = dict(FICHA_RF, adhesivo=_ADH_RF)
ok(not avisos_de(RF_ADH, FICHA_RF_ADH, DATOS_RF, "revision_fiscal"),
   "RF: «se tuvo a X adhiriéndose al recurso» trata el adhesivo en los resultandos (cero avisos de toda la verja)")
ok(not avisos_de(RF_ADH.replace("se tuvo a María Gómez Ruiz adhiriéndose al recurso",
                                "María Gómez Ruiz se adhirió al recurso"), FICHA_RF_ADH, DATOS_RF, "revision_fiscal", "k"),
   "RF: «se adhirió» también")
ok(not avisos_de(RF_ADH.replace("se tuvo a María Gómez Ruiz adhiriéndose al recurso",
                                "se admitió la adhesión de María Gómez Ruiz"), FICHA_RF_ADH, DATOS_RF,
                 "revision_fiscal", "k"), "RF: «la adhesión» también")
a = avisos_de(RF_ADH.replace("Por auto de veintidós de mayo de dos mil veintiséis, se tuvo a María Gómez Ruiz "
                             "adhiriéndose al recurso.\n", ""), FICHA_RF_ADH, DATOS_RF, "revision_fiscal", "k")
ok(len(a) == 1 and "en los resultandos" in a[0] and "considerandos" not in a[0],
   "RF: sin el auto que la tuvo por interpuesta, se sigue acusando en los resultandos")
_ad_adh = con(AD_CDMX, "se admitió el amparo adhesivo promovido por Alejandro Daniel Betancourt Velarde",
              "se tuvo a Alejandro Daniel Betancourt Velarde adhiriéndose al juicio")
ok(not avisos_de(_ad_adh, FICHA_CDMX, DATOS_CDMX, "amparo_directo", "k"),
   "AD: «se tuvo a X adhiriéndose» cuenta como su admisión (no sólo la mención del 181)")

print("\n13 · NUNCA LANZA")
for args in (("", None, None, ""), (None, None, None, None), ("texto sin estructura alguna", {}, {}, "x"),
             (AD, {"acto": "no es un dict"}, {"tramite": "tampoco"}, "amparo_directo"),
             (AD, FICHA_AD, DATOS_AD, "AD"), (RF, FICHA_RF, DATOS_RF, "rf")):
    try:
        r = vp.revisar(*args)
        ok(isinstance(r, list), f"revisar({str(args[0])[:20]!r}…, tipo={args[3]!r}) devuelve lista")
    except Exception as ex:
        ok(False, f"revisar lanzó {type(ex).__name__}: {ex}")
ok(vp.revisar(AD, None, dict(DATOS_AD, tramite=FICHA_AD), "") == [],
   "la ficha también se toma de datos['tramite'] y el tipo de la ficha")
ok(isinstance(vp.activa(), bool), "activa() responde (bandera procedencia_por_tipo)")

print("\n14 · ARREGLAR: SÓLO TIPOGRAFÍA SEGURA")
x, c = vp.arreglar("el escrito de el quejoso, a el que se refiere el artículo 5º, fracción I, y el 2°. de la ley; "
                   "Impulsora, S.A de C.V. y Otra, S.A. de C.V lo firmaron  el  dos ml veinte.")
ok(x == "el escrito del quejoso, al que se refiere el artículo 5o., fracción I, y el 2o. de la ley; Impulsora, S.A. "
        "de C.V. y Otra, S.A. de C.V. lo firmaron el dos mil veinte.", f"todo a la vez: {x}")
ok(len(c) == 8 and all(isinstance(z, str) for z in c), f"cada arreglo se declara ({len(c)} cambios)")
ok(vp.arreglar(x) == (x, []), "idempotente: lo arreglado no se vuelve a tocar")
intacto = ("Juzgado Mixto de El Marqués, ubicado en la Calle de El Oro; se lo dijo a él. Sesión de *********. "
           "artículo 5o., fracción I; Inmobiliaria, S. A. de C. V.; «DE EL MARQUÉS»; una 1a. Sala.")
ok(vp.arreglar(intacto) == (intacto, []), "no toca «de El Marqués», «a él», el hueco, «5o.» ni las versales")
ok(vp.arreglar("") == ("", []) and vp.arreglar(None) == ("", []), "vacío y None")

# ═══════════════════════════════════════════════════════════════════════════
# 15 · CALIBRACIÓN CONTRA LAS SENTENCIAS DEL TRIBUNAL
# ═══════════════════════════════════════════════════════════════════════════
# Las versiones públicas más recientes de cada tipo que son sentencia con
# resultandos y considerandos (3-oct-2026). Los asteriscos de la versión
# pública NO son huecos del taller —suprimen nombres— y se sustituyen antes de
# revisar; la regla (a) se mide en los proyectos, no aquí.
LECTURA = os.path.expanduser("~/Documents/IUREXIA-MAC/redactor-sentencias/oaj/lectura")
CALIBRACION = """AD_386-2026_41872006 AD_48-2026_40904128 AD_621-2025_39689978 AD_657-2025_39906309
AD_659-2025_39908387 AD_661-2025_39910735 AD_742-2025_40350643 AD_750-2025_40363206 AD_771-2025_40429341
AD_835-2025_40626383 AD_837-2025_40636454 AR_165-2026_41305438 AR_174-2026_41314407 AR_180-2026_41335770
AR_197-2026_41380695 AR_214-2026_41424284 AR_251-2026_41547528 AR_257-2026_41569359 AR_266-2026_41600204
AR_281-2026_41645409 AR_295-2026_41674024 Q_347-2026_42225905 Q_348-2026_42228899 Q_350-2026_42229359
Q_354-2026_42236753 Q_355-2026_42237777 Q_356-2026_42238267 Q_357-2026_42238839 Q_359-2026_42239300
Q_416-2026_42567764 Q_417-2026_42578969 RF_1-2026_40854675 RF_11-2026_41005568 RF_12-2026_41026924
RF_19-2026_41109599 RF_23-2026_41163340 RF_4-2026_40897736 RF_42-2026_41626549 RF_73-2025_39685663
RF_77-2025_39885434 RF_83-2025_40232836""".split()
# LO QUE LA VERJA ENCUENTRA EN ELLAS Y SE REVISÓ A MANO: son errores del propio
# engrose, medidos también por los lectores del oro (res/oro_*.txt):
#   AR 180 y 214: AG 1/2023 de la SCJN, abrogado el 4-sep-2025 (oro AR: 9 casos);
#   Q 417: LOPJF 38 y 124 (ley de 2021) en la competencia (oro Q: 29 residuos);
#   RF 4, 11 y 73: «37, fracción V, de la LOPJF» en la antesala (oro RF: ~13);
#   AD 837: el resultando dice «sentencia de cuatro de noviembre» y el
#     resolutivo «dictada el cuatro de septiembre» (la demanda es del 24-nov);
#   AD 621: la demanda «presentada el veinte de noviembre de dos mil
#     veinticuatro» contra una sentencia de once de julio de dos mil
#     veinticinco (el asunto es el 621/2025: una de las dos fechas es errata);
#   AD 771 y AR 174: «por la Primera Sala… / por la Jueza…, localizado» —la
#     concordancia que el SPEC §3.2 manda arreglar en la plantilla—.
ESPERADOS = {("AR_180-2026_41335770", "h"), ("AR_214-2026_41424284", "h"), ("Q_417-2026_42578969", "h"),
             ("RF_4-2026_40897736", "h"), ("RF_11-2026_41005568", "h"), ("RF_73-2025_39685663", "h"),
             ("AD_837-2025_40636454", "c"), ("AD_621-2025_39689978", "j"),
             ("AD_771-2025_40429341", "f"), ("AR_174-2026_41314407", "f")}
_ORD_C = (r"(?:PRIMERO|SEGUNDO|TERCERO|CUARTO|QUINTO|SEXTO|S[ÉE]PTIMO|OCTAVO|NOVENO|D[ÉE]CIMO(?:\s+\w+)?|[ÚU]NICO)")
_PROC_C = (r"(?:Competencia|Existencia|Certeza|Legitimaci|Oportunidad|Procedencia|Transcripci|Innecesar|"
           r"Sentencia reclamada y|Resoluci[óo]n reclamada y|Acto reclamado y|Laudo reclamado y|Sentencia recurrida y|"
           r"Resoluci[óo]n recurrida y|Auto recurrido y|Relaci[óo]n|Conexidad|Agravios y|Conceptos de violaci[óo]n y|"
           r"Resoluci[óo]n impugnada y|Requisitos)")


def _version_publica(ruta):
    """De la carátula al rótulo del Estudio + los resolutivos, por párrafos."""
    import fitz
    paginas = []
    for i, p in enumerate(fitz.open(ruta)):
        ls = [l.rstrip() for l in p.get_text().split("\n")]
        while ls and not ls[-1].strip():
            ls.pop()
        if ls and "Versión Pública" in ls[-1]:
            ls.pop()
            while ls and re.match(r"^\s*(?:\d\d/\d\d/\d\d \d\d:\d\d:\d\d|[0-9a-f]{30,})\s*$", ls[-1]):
                ls.pop()
            if ls and re.match(r"^[A-ZÁÉÍÓÚÑ .]+$", ls[-1].strip()):
                ls.pop()
        for k in range(max(0, len(ls) - 25), len(ls)):        # notas al pie
            if re.match(r"^\s*\d{1,2}\s+[A-ZÁÉÍÓÚ“\"«]", ls[k]) and k > len(ls) * 0.4:
                ls = ls[:k]
                break
        if i:
            while ls and len(ls) and (not ls[0].strip() or re.match(r"^\s*\d{1,3}\s*$", ls[0]) or re.match(
                    r"^\s*(?:AMPARO|RECURSO|QUEJA|REVISI|INCIDENTE|TOCA).{0,60}\d+/\d{4}\.?\s*$", ls[0].strip(), re.I)
                    or re.match(r"^\s*[A-Z](?:\.?[A-Z]){1,4}\.?\s*\d+/\d{4}\s*\d{0,3}\s*$", ls[0].strip())):
                ls = ls[1:]
        paginas.append("\n".join(ls))
    t = re.sub(r"\s+", " ", " \n".join(paginas))
    t = re.sub(r"\s*(R\s*E\s*S\s*U\s*L\s*T\s*A\s*N\s*D\s*O\s*S?\s*:?|C\s*O\s*N\s*S\s*I\s*D\s*E\s*R\s*A\s*N\s*D\s*O\s*S?"
               r"\s*:?|R\s*E\s*S\s*U\s*E\s*L\s*V\s*E\s*:?)\s*", r"\n\1\n", t)
    t = re.sub(r"\s*\b(V\s+I\s+S\s+T\s+O\s*S?|V\s+i\s+s\s+t\s+o\s*s?)\b", r"\n\1", t)
    t = re.sub(r"(?<=[.:;,]) +(" + _ORD_C + r"\s*\.\s)", r"\n\1", t)
    t = re.sub(r"(?<=[.:;]) +(\d{1,3}\.\s+[A-ZÁÉÍÓÚ])", r"\n\1", t)
    ls, zona, est, res = t.split("\n"), "", None, None
    for i, l in enumerate(ls):
        if re.match(r"^C\s*O\s*N\s*S\s*I\s*D\s*E\s*R\s*A\s*N\s*D\s*O", l):
            zona = "C"
        elif re.match(r"^R\s*E\s*S\s*U\s*E\s*L\s*V\s*E", l) or re.search(r"se resuelve\s*[:;]\s*$", l, re.I):
            res = i if res is None else res
            zona = "R"
        elif zona == "C" and est is None:
            m = re.match(r"^(?:" + _ORD_C + r"|\d{1,3})\s*\.\s*([^.]{2,90}?)\.", l)
            if m and len(m.group(1).split()) <= 10 and not re.match(_PROC_C, m.group(1).strip(), re.I) \
                    and m.group(1).strip()[:1].isupper():
                est = i
    if est is None or res is None:
        return ""
    cola = ls[res:]
    for j, l in enumerate(cola):
        m = re.search(r"Notif[íi]quese|Así lo resolvi|Así, por unanimidad|Así lo resolvieron", l)
        if m and j:
            cola = cola[:j] + [l[:m.start()]]
            break
    t = "\n".join(x.strip() for x in ls[:est + 1] + cola if x.strip())
    return re.sub(r"\*+(?:[ \-/.]*\*+)*", "Fulano", t)


print("\n15 · CALIBRACIÓN: 41 SENTENCIAS DEL TRIBUNAL (versiones públicas 2025-2026)")
try:
    import fitz  # noqa: F401
    hay_corpus = os.path.isdir(LECTURA)
except Exception:
    hay_corpus = False
if not hay_corpus:
    print("   (sin el corpus del OAJ o sin PyMuPDF en este equipo: se salta)")
else:
    _tipos = {"AD": "amparo_directo", "AR": "amparo_revision", "Q": "queja", "RF": "revision_fiscal"}
    hallados, leidas = set(), 0
    por_regla = {}
    for k in CALIBRACION:
        ruta = os.path.join(LECTURA, k + ".pdf")
        if not os.path.exists(ruta):
            continue
        t = _version_publica(ruta)
        if not t:
            continue
        leidas += 1
        pre, num, _ = k.split("_")
        m = re.search(r"Resoluci[óo]n\s+del?\s+(.+?Circuito)", t)
        datos = {"numero": num.replace("-", "/"), "tribunal": m.group(1) if m else "", "ciudad": "Querétaro"}
        for x in vp.revisar_detalle(t, None, datos, _tipos[pre]):
            hallados.add((k, x["regla"]))
            por_regla[x["regla"]] = por_regla.get(x["regla"], 0) + 1
    print(f"   {leidas} sentencias leídas; avisos por regla: "
          + ", ".join(f"({r}) {por_regla.get(r, 0)}" for r in "abcdefghijklmnopq"))
    ok(leidas >= 38, f"se leyeron {leidas} de {len(CALIBRACION)}")
    nuevos = hallados - ESPERADOS
    ok(not nuevos, "NINGÚN aviso nuevo sobre las sentencias buenas" + (f": {sorted(nuevos)}" if nuevos else ""))
    # SEXTA RONDA (3-oct-2026): también (o), (p) y (q), las de los asuntos
    # relacionados. Medidas además, fuera de esta prueba, sobre 118 sentencias
    # 2024-2026 del tribunal que los mencionan: 0 avisos nuevos.
    ok(all(por_regla.get(r, 0) == 0 for r in "abdegiklmnopq"),
       "cero en (a), (b), (d), (e), (g), (i), (k), (l), (m), (n), (o), (p) y (q); en (c), (f), (h) y (j) sólo los "
       "errores del engrose de arriba")

# ═══════════════════════════════════════════════════════════════════════════
# 16 · LO QUE COMPONE `resultandos_por_tipo` PASA LA VERJA (si ya existe)
# ═══════════════════════════════════════════════════════════════════════════
print("\n16 · EL COMPOSITOR POR TIPO CONTRA LA VERJA")
try:
    import resultandos_por_tipo as _rpt
    _componer = getattr(_rpt, "componer")
except Exception:
    _componer = None
if _componer is None:
    print("   (resultandos_por_tipo.componer todavía no existe: se salta)")
else:
    for tipo, ficha, datos in (("amparo_directo", FICHA_AD, DATOS_AD), ("amparo_revision", FICHA_AR, DATOS_AR),
                               ("queja", FICHA_Q, DATOS_Q), ("revision_fiscal", FICHA_RF, DATOS_RF)):
        try:
            r = _componer(tipo, ficha, dict(datos, magistrado="Luis Armando Pérez Topete", materia=ficha["materia"]))
            cuerpo = "\n".join(f"{['PRIMERO', 'SEGUNDO', 'TERCERO', 'CUARTO', 'QUINTO', 'SEXTO', 'SÉPTIMO'][i]}. "
                               f"{x['titulo'].rstrip('.')}. {x['texto']}" for i, x in enumerate(r["resultandos"][:7]))
            texto = f"V I S T O, {r['visto']}\nR E S U L T A N D O\n{cuerpo}\nC O N S I D E R A N D O\n"
            av = [x for x in vp.revisar_detalle(texto, ficha, datos, tipo)]
            ok(not av, f"{tipo}: el V I S T O y los resultandos compuestos, sin avisos"
               + (f" — {[x['aviso'][:120] for x in av[:2]]}" if av else ""))
        except Exception as ex:
            ok(False, f"{tipo}: componer lanzó {type(ex).__name__}: {ex}")
    # Ronda 2: las tres variantes que daban avisos falsos sobre lo compuesto
    # (informe del compositor, 3-oct-2026): la RF con adhesiva y la Sala de la
    # ficha en versales y sin la cola del TFJA; el AR con la clave mixta cruda.
    for nombre, tipo, ficha, datos in (
            ("RF con adhesiva y ficha.sala en versales, sin la cola del TFJA", "revision_fiscal",
             dict(FICHA_RF_ADH, sala=SALA_RF.upper(), acto=dict(FICHA_RF["acto"], organo=SALA_RF.upper())),
             dict(DATOS_RF, organo_recurrido="")),
            ("AR con acto.resolvio = «sobresee_niega» en crudo", "amparo_revision",
             dict(FICHA_AR, acto=dict(FICHA_AR["acto"], resolvio="sobresee_niega")), DATOS_AR),
            ("AR con acto.resolvio = «sobresee_concede» en crudo", "amparo_revision",
             dict(FICHA_AR, acto=dict(FICHA_AR["acto"], resolvio="sobresee_concede")), DATOS_AR),
            ("AR con «mixto» + resolvio_mixto", "amparo_revision",
             dict(FICHA_AR, acto=dict(FICHA_AR["acto"], resolvio="mixto", resolvio_mixto="sobresee_concede")),
             DATOS_AR)):
        try:
            r = _componer(tipo, ficha, dict(datos, magistrado="Luis Armando Pérez Topete", materia=ficha["materia"]))
            cuerpo = "\n".join(f"{['PRIMERO', 'SEGUNDO', 'TERCERO', 'CUARTO', 'QUINTO', 'SEXTO', 'SÉPTIMO'][i]}. "
                               f"{x['titulo'].rstrip('.')}. {x['texto']}" for i, x in enumerate(r["resultandos"][:7]))
            texto = f"V I S T O, {r['visto']}\nR E S U L T A N D O\n{cuerpo}\nC O N S I D E R A N D O\n"
            av = vp.revisar_detalle(texto, ficha, datos, tipo)
            ok(not av, f"{nombre}: lo compuesto pasa sin avisos"
               + (f" — {[x['aviso'][:120] for x in av[:2]]}" if av else ""))
        except Exception as ex:
            ok(False, f"{nombre}: componer lanzó {type(ex).__name__}: {ex}")

# ═══════════════════════════════════════════════════════════════════════════
# 17 · RONDA 3 (3-oct-2026): lo que el banco de oráculo y la punta a punta
#      encontraron y la verja callaba
# ═══════════════════════════════════════════════════════════════════════════
print("\n17 · RONDA 3: BANCO DE ORÁCULO (32 SENTENCIAS REALES) Y PUNTA A PUNTA")


def _seguro(f, *a, **k):
    """La prueba no se cae si la verja vieja no tiene la función o el argumento: falla."""
    try:
        return f(*a, **k)
    except Exception as ex:
        return ex


# ── EL PONENTE DEL ÚLTIMO RETURNO (8 AD, 6 RF y 2 Q del banco) ──
_ad_turno = con(AD_CDMX, "MAGISTRADO PONENTE: ISMAEL CAMACHO HERRERA.", "MAGISTRADO PONENTE: LUIS ARMANDO PÉREZ TOPETE.")
a = avisos_de(_ad_turno, FICHA_CDMX, DATOS_CDMX, "amparo_directo", "d")
ok(len(a) == 1 and "NO ES EL DEL RETURNO" in a[0] and "Ismael Camacho Herrera" in a[0],
   "con returno, la carátula con el ponente del TURNO se acusa (el arnés del banco: 8 de 8 AD)")
_dos_returnos = con(AD_CDMX, "SÉPTIMO. Celebración", "SÉPTIMO. Returno. Por acuerdo de ocho de junio de dos mil "
                    "veintiséis, se returnaron los autos a la ponencia de Jenica Campos Juárez, para la elaboración "
                    "del proyecto de resolución.\nOCTAVO. Celebración")
a = avisos_de(_dos_returnos, FICHA_CDMX, DATOS_CDMX, "amparo_directo", "d")
ok(len(a) == 1 and "Jenica Campos Juárez" in a[0], "dos returnos: se coteja contra el ÚLTIMO")
ok(not avisos_de(con(_dos_returnos, "MAGISTRADO PONENTE: ISMAEL CAMACHO HERRERA.", "MAGISTRADO PONENTE: JENICA CAMPOS "
                     "JUÁREZ."), FICHA_CDMX, DATOS_CDMX, "amparo_directo", "d"),
   "…y la carátula con el del último returno pasa")
_sin_nombre = con(_ad_turno, "se returnaron los autos a la ponencia de Ismael Camacho Herrera",
                  "se returnaron los autos con motivo de la nueva integración del tribunal")
a = avisos_de(_sin_nombre, FICHA_CDMX, DATOS_CDMX, "amparo_directo", "d")
ok(len(a) == 1 and "returno.ponente" in a[0] and "Ismael Camacho Herrera" in a[0],
   "el resultando de returno sin nombre: se coteja contra returno.ponente de la ficha")
ok(not avisos_de(con(_sin_nombre, "MAGISTRADO PONENTE: LUIS ARMANDO PÉREZ TOPETE.", "MAGISTRADO PONENTE: ISMAEL "
                     "CAMACHO HERRERA."), FICHA_CDMX, DATOS_CDMX, "amparo_directo", "d"), "…y si coincide, nada")
_al_magistrado = con(_ad_turno, "se returnaron los autos a la ponencia de Ismael Camacho Herrera",
                     "se returnó el asunto al Magistrado Ismael Camacho Herrera")
ok(any("NO ES EL DEL RETURNO" in x for x in avisos_de(_al_magistrado, FICHA_CDMX, DATOS_CDMX, "amparo_directo", "d")),
   "«se returnó el asunto al Magistrado X» también se lee")

# ── CONCORDANCIAS: -dora/-tora/-sora, Jefa, Legislatura…, prefijo temporal ──
for _frase in ("el Jefa de la Unidad Jurídica", "el JEFA DE LA UNIDAD JURÍDICA", "el Coordinadora de Recursos Humanos",
               "el Subdirectora de Afiliación y Vigencia de Derechos", "al Legislatura del Estado de Querétaro",
               "del Cámara de Diputados", "el Oficina Recaudadora", "el Jefatura de Gobierno",
               "el Actual Sala Regional en Querétaro", "el entonces Procuradora Fiscal"):
    t = con(AD, "Tiene ese carácter Alejandro Daniel Betancourt Velarde.",
            f"Tiene ese carácter Alejandro Daniel Betancourt Velarde y {_frase}.")
    ok(any("masculino ante un nombre femenino" in x for x in avisos_de(t, FICHA_AD, DATOS_AD, "amparo_directo", "f")),
       f"«{_frase}» se acusa (banco de oráculo: AD 128, AR 201, RF 2, 7, 26 y 49)")
for _frase in ("el acta de la audiencia", "el área de amparos", "la Coordinadora de Recursos Humanos",
               "del Director de Ingresos", "el actual Juzgado Tercero de lo Civil"):
    t = con(AD, "Tiene ese carácter Alejandro Daniel Betancourt Velarde.",
            f"Tiene ese carácter Alejandro Daniel Betancourt Velarde y {_frase}.")
    ok(not any("masculino ante un nombre femenino" in x
               for x in avisos_de(t, FICHA_AD, DATOS_AD, "amparo_directo", "f")), f"«{_frase}» no se acusa")
t = con(AD, f"dictada por la {SALA}, localizada", f"dictada por la actual {SALA}, localizado")
ok(any("localizado" in x for x in avisos_de(t, FICHA_AD, DATOS_AD, "amparo_directo", "f")),
   "«por la actual Sala…, localizado» (el prefijo temporal no esconde la Sala)")
t = con(RF, f"tramitado ante la {SALA_RF} del Tribunal Federal de Justicia Administrativa, localizada",
        f"tramitado ante el Actual {SALA_RF} del Tribunal Federal de Justicia Administrativa, localizado")
a = avisos_de(t, FICHA_RF, DATOS_RF, "revision_fiscal", "f")
ok(any("localizado" in x for x in a) and any("masculino ante un nombre femenino" in x for x in a),
   "RF 49: «ante el Actual Sala…, localizado»: el artículo y la concordancia")

# ── VERSALES TRAS «LO HIZO VALER / INTERPUESTO POR / PROMOVIDO POR» ──
t = con(AD, "La demanda de amparo fue promovida por Ruth Gabriela Larrañaga Silva, quien",
        "La demanda de amparo fue promovida por RUTH GABRIELA LARRAÑAGA SILVA, quien")
ok(any("VERSALES" in x and "RUTH GABRIELA" in x for x in avisos_de(t, FICHA_AD, DATOS_AD, "amparo_directo", "d")),
   "AD 274 de punta a punta: «promovida por GABRIEL REYES ALAMO» (una persona en versales)")
t = con(RF, f"El recurso lo interpone el {UJ}, unidad", f"El recurso fue interpuesto por parte legítima, pues lo "
        f"hizo valer el {UJ.upper()}, unidad")
ok(any("VERSALES" in x and "TITULAR DE LA UNIDAD" in x for x in avisos_de(t, FICHA_RF, DATOS_RF, "revision_fiscal",
                                                                          "d")),
   "RF (7 de 8 del banco): «interpuesto por parte legítima, pues lo hizo valer el TITULAR…» — la segunda mención "
   "no se la traga la primera")
for _x in ("el IMSS", "PARTE QUEJOSA", "INFONAVIT"):
    t = con(AD, "promovido por Ruth Gabriela Larrañaga Silva, contra", f"promovido por {_x}, contra")
    ok(not any("VERSALES" in x for x in avisos_de(t, FICHA_AD, DATOS_AD, "amparo_directo", "d")),
       f"«promovido por {_x}»: siglas o la figura, no un nombre en versales")
t = con(AR, "La determinación del crédito fiscal contenida en el oficio DI/123/2025",
        "La determinación del crédito fiscal en el procedimiento promovido por JUAN PÉREZ LÓPEZ, oficio DI/123/2025")
ok(not any("VERSALES" in x for x in avisos_de(t, FICHA_AR, DATOS_AR, "amparo_revision", "d")),
   "dentro del bloque ACTOS RECLAMADOS (la demanda tal cual) no se acusa")

# ── LA ETIQUETA DE ROL DE LA CARÁTULA EN LA PROSA (AR 208 y 72, RF 4) ──
for _et in ("(AUTORIDAD RESPONSABLE)", "(autoridad Responsable)", "(DEMANDADA)"):
    t = con(AR, f"interpuesto por {COM}, contra la sentencia", f"interpuesto por {COM} {_et}, contra la sentencia")
    ok(any("ETIQUETA DE ROL" in x for x in avisos_de(t, FICHA_AR, DATOS_AR, "amparo_revision", "d")),
       f"«{_et}» en el V I S T O se acusa")
for _et in ("(tercera interesada)", "(Quejosa)"):
    t = con(AR, f"interpuesto por {COM}, contra la sentencia", f"interpuesto por {COM} {_et}, contra la sentencia")
    ok(not any("ETIQUETA DE ROL" in x for x in avisos_de(t, FICHA_AR, DATOS_AR, "amparo_revision", "d")),
       f"«{_et}», como lo escribe el tribunal (Q 416, AD 742 de la calibración): no se acusa")

# ── LA CARÁTULA SIN EL NÚMERO DEL ASUNTO (8 de 8 RF del banco) ──
t = con(AD, "AMPARO DIRECTO CIVIL: 174/2026.\n", "")
a = avisos_de(t, FICHA_AD, DATOS_AD, "amparo_directo", "e")
ok(len(a) == 1 and "NO TRAE EL NÚMERO" in a[0] and "174/2026" in a[0], "la carátula sin «AMPARO DIRECTO CIVIL: 174/2026»")
ok(not avisos_de(t, None, DATOS_AD, "amparo_directo", "e"), "…sin ficha (camino viejo, versión pública) no se acusa")

# ── «X, EN REPRESENTACIÓN DE Y, EN REPRESENTACIÓN DE Y» Y «X, EN REPRESENTACIÓN DE X» ──
_rep = f"Inconforme, el {UJ}, en representación de la autoridad demandada,"
t = con(RF, _rep, f"Inconforme, el {UJ}, en representación del Subdelegado de Prestaciones Económicas, Delegación "
                  f"Estatal en Querétaro, en representación del Subdelegado de Prestaciones Económicas, Delegación "
                  f"Estatal en Querétaro,")
ok(any("REPRESENTACIÓN REPETIDA" in x for x in avisos_de(t, FICHA_RF, DATOS_RF, "revision_fiscal", "d")),
   "RF 2, 7 y 6-2025: la representada dos veces seguidas (con comas dentro del nombre)")
t = con(RF, _rep, "Inconforme, el Subdelegado de Prestaciones Económicas, Delegación Estatal Querétaro, del Instituto "
                  "de Seguridad y Servicios Sociales de los Trabajadores del Estado, en representación del Subdelegado "
                  "de Prestaciones, Delegación Estatal Querétaro, del Instituto de Seguridad y Servicios Sociales de los "
                  "Trabajadores del Estado,")
ok(any("SÍ MISMO" in x for x in avisos_de(t, FICHA_RF, DATOS_RF, "revision_fiscal", "d")),
   "RF 6-2026: «el Subdelegado de Prestaciones Económicas…, en representación del Subdelegado de Prestaciones…»")
for _nombre, _txt in (
        ("la unidad jurídica por el titular de la delegación (lo que escribe el compositor)",
         f"Inconforme, el {UJ}, en representación del Titular de la Delegación Estatal en Querétaro del Instituto de "
         f"Seguridad y Servicios Sociales de los Trabajadores del Estado,"),
        ("la unidad jurídica por el Instituto",
         "Inconforme, el Jefe de la Unidad Jurídica del Instituto Mexicano del Seguro Social, en representación del "
         "Instituto Mexicano del Seguro Social,"),
        ("el Departamento Jurídico por el de Pensiones",
         "Inconforme, el Jefe del Departamento Jurídico de la Delegación Estatal Querétaro del ISSSTE, en "
         "representación del Jefe del Departamento de Pensiones de la Delegación Estatal Querétaro del ISSSTE,")):
    t = con(RF, _rep, _txt)
    ok(not any("REPRESENTACIÓN REPETIDA" in x or "SÍ MISMO" in x
               for x in avisos_de(t, FICHA_RF, DATOS_RF, "revision_fiscal", "d")), f"{_nombre}: no es repetición")

# ── (l) LA CLASE DEL ACTO (AD 349 y 552 del banco) ──
a = avisos_de(AD, dict(FICHA_AD, acto=dict(FICHA_AD["acto"], clase="resolucion")), DATOS_AD, "amparo_directo", "l")
ok(len(a) == 1 and "la resolución reclamada" in a[0] and "COMPETENCIA" in a[0].upper(),
   "AD con acto.clase «resolucion» y «una sentencia definitiva / la sentencia reclamada»: un aviso con los apartados")
a = avisos_de(AD, dict(FICHA_AD, acto=dict(FICHA_AD["acto"], clase="laudo")), DATOS_AD, "amparo_directo", "l")
ok(len(a) == 1 and "el laudo reclamado" in a[0], "…con un laudo, «el laudo reclamado»")
ok(not avisos_de(AD, FICHA_AD, DATOS_AD, "amparo_directo", "l") and not avisos_de(AD, None, DATOS_AD, "amparo_directo",
                                                                                  "l"),
   "con la sentencia, o sin ficha, nada")
a = avisos_de(AR, dict(FICHA_AR, clase_recurrida="auto_sobreseimiento"), DATOS_AR, "amparo_revision", "l")
ok(len(a) == 1 and "el auto recurrido" in a[0], "AR: el auto de sobreseimiento (81-I-d) llamado «la sentencia recurrida»")
ok(not avisos_de(AR, dict(FICHA_AR, clase_recurrida="interlocutoria_suspension"), DATOS_AR, "amparo_revision", "l"),
   "AR: la interlocutoria es sentencia interlocutoria: nada")

# ── (m) LA FORMA DE NOTIFICACIÓN (quejas 172, 24 y 300 del banco) ──
a = avisos_de(AD, dict(FICHA_AD, forma_notificacion="electronica"), DATOS_AD, "amparo_directo", "m")
ok(len(a) == 1 and "de manera personal" in a[0] and "por vía electrónica" in a[0],
   "la ficha dice electrónica y el considerando «de manera personal»")
_bol = con(AD, "de manera personal y surtió", "mediante Boletín Judicial y surtió")
ok(not avisos_de(_bol, dict(FICHA_AD, forma_notificacion="lista"), DATOS_AD, "amparo_directo", "m"),
   "lista y Boletín son la misma publicación: nada")
ok(not avisos_de(_bol, FICHA_AD, DATOS_AD, "amparo_directo", "m"),
   "«personal» en la ficha es la omisión del formulario: contra el Boletín del TFJA no se acusa (integ AD Yucatán)")
_of = con(AD, "de manera personal y surtió", "por oficio y surtió")
ok(not avisos_de(_of, FICHA_AD, DATOS_AD, "amparo_directo", "m"),
   "…ni contra el «por oficio» de la autoridad que recurre (D2, art. 31, fr. I)")
ok(len(avisos_de(_of, dict(FICHA_AD, forma_notificacion="electronica"), DATOS_AD, "amparo_directo", "m")) == 1,
   "la ficha dice electrónica (fr. III) y el considerando «por oficio» (fr. I): se acusa")
ok(len(avisos_de(AD, dict(FICHA_AD, forma_notificacion="lista"), DATOS_AD, "amparo_directo", "m")) == 1,
   "Q 300: la ficha dice lista y el considerando «de manera personal»")
ok(not avisos_de(AD, dict(FICHA_AD, forma_notificacion=""), DATOS_AD, "amparo_directo", "m"),
   "sin la forma en la ficha no se coteja")

# ── (g) «SUPLETORIAMENTE», EL ARTÍCULO 2o. Y LA SEDE DE tipos_asunto (rev_2) ──
_ex = ("Documentales que en términos de los artículos 129 y 202 del Código Federal de Procedimientos Civiles, de "
       "aplicación supletoria a la Ley de Amparo, merecen")
for _nombre, _txt in (
        ("«aplicado supletoriamente a la Ley de Amparo»",
         "Documentales que en términos de los artículos 312 y 344 del Código Nacional de Procedimientos Civiles y "
         "Familiares, aplicado supletoriamente a la Ley de Amparo, merecen"),
        ("«en relación con el artículo 2o. de la Ley de Amparo»",
         "Documentales que en términos de los artículos 312 y 344 del Código Nacional de Procedimientos Civiles y "
         "Familiares, en relación con el artículo 2o. de la Ley de Amparo, merecen"),
        ("el código DESPUÉS del artículo 2o.",
         "Documentales que en términos del artículo 2o. de la Ley de Amparo, que remite al Código Nacional de "
         "Procedimientos Civiles y Familiares, merecen")):
    a = avisos_de(con(AD, _ex, _txt), FICHA_AD, DATOS_AD, "amparo_directo", "g")
    ok(len(a) == 1 and "Código Nacional" in a[0], f"el CNPCF en Querétaro con {_nombre}")
ok(not avisos_de(AD, dict(FICHA_AD, sede=dict(FICHA_AD["sede"], cdmx=True)), DATOS_AD, "amparo_directo", "g"),
   "la sede la decide tipos_asunto (Querétaro, XXII), no ficha.sede.cdmx: el CFPC no se acusa")
_t_est = con(AD, "QUINTO. Estudio.", "QUINTO. Estudio. Es un hecho notorio, en términos del artículo 269 del Código "
             "Nacional de Procedimientos Civiles y Familiares, de aplicación supletoria a la Ley de Amparo, que el "
             "toca se resolvió.")
a = _seguro(getattr(vp, "revisar_supletorio", None), _t_est, FICHA_AD, DATOS_AD, "amparo_directo")
ok(isinstance(a, list) and len(a) == 1 and "ESTUDIO" in a[0],
   "revisar_supletorio: el documento entero, sólo la regla (g) (el hecho notorio en el Estudio)")

# ── (h) EL AVISO DEL AG 1/2023 DICE BIEN QUIÉN ABROGÓ A QUIÉN ──
t = con(AR, COLA_COMPETENCIA, COLA_COMPETENCIA + " Así como en el punto quinto del Acuerdo General 1/2023 del Pleno de "
        "la Suprema Corte de Justicia de la Nación.")
a = [x for x in avisos_de(t, FICHA_AR, DATOS_AR, "amparo_revision", "h") if "1/2023" in x]
ok(len(a) == 1 and "abrogado por el Acuerdo General 11/2025" in a[0] and "que abrogó" not in a[0],
   "AG 1/2023 «abrogado por el Acuerdo General 11/2025» (rev_2: el aviso lo decía al revés)")

# ── (a) EL HUECO CON SU CLAVE, AGRUPADO Y SIN REPETIR LO YA AVISADO (Q 335: 8 avisos para 2 datos) ──
_q_sin_organo = Q.replace(f"dictado por el {JUZ3}", "dictado por *********")
d = [x for x in vp.revisar_detalle(_q_sin_organo, FICHA_Q, DATOS_Q, "queja") if x["regla"] == "a"]
ok(len(d) == 1 and d[0].get("clave") == "acto.organo" and len(d[0].get("apartados") or []) >= 3
   and "(acto.organo)" in d[0]["aviso"],
   "el órgano en hueco en el V I S T O, la interposición, la competencia y el resolutivo: UN aviso con su clave")
_prev = ["FALTA EL ÓRGANO QUE DICTÓ EL AUTO RECURRIDO (acto.organo): está en el auto recurrido. Va en hueco en «V I S "
         "T O»."]
r = _seguro(vp.revisar, _q_sin_organo, FICHA_Q, DATOS_Q, "queja", avisos_previos=_prev)
ok(isinstance(r, list) and not any(x.startswith("HUECO") for x in r),
   "avisos_previos: si el compositor ya avisó «(acto.organo)», la verja no lo repite")
_t_surte = con(AD, "conforme al artículo 126 del Código de Procedimientos Civiles del Estado de Querétaro",
               "conforme al *********")
r = _seguro(vp.revisar, _t_surte, FICHA_AD, DATOS_AD, "amparo_directo",
            avisos_previos=["EL SURTIMIENTO VA SIN PRECEPTO, EN HUECO: el considerando dice que…"])
ok(isinstance(r, list) and not any(x.startswith("HUECO") for x in r),
   "…ni el precepto del surtimiento si la oportunidad ya avisó «EL SURTIMIENTO VA SIN PRECEPTO»")
_t_adh = con(AD_CDMX, "y lo hizo dentro de los quince días del artículo 181.",
             "y el auto admisorio se le notificó el ********* y el escrito se presentó el *********, por lo que "
             "*********.")
d = [x for x in vp.revisar_detalle(_t_adh, FICHA_CDMX, DATOS_CDMX, "amparo_directo") if x["regla"] == "a"]
ok(len(d) == 3 and [x.get("clave") for x in d] == ["adhesivo.notificacion", "adhesivo.presentacion", ""]
   and "veredicto" in d[2]["aviso"],
   "AD 279: el considerando del adhesivo sin sus dos fechas: un aviso POR CAMPO (lo que documento_generado sabe "
   "callar uno a uno) y el veredicto nombrado")
r = _seguro(vp.revisar, _t_adh, FICHA_CDMX, DATOS_CDMX, "amparo_directo",
            avisos_previos=["FALTA LA FECHA DE NOTIFICACIÓN DEL AUTO ADMISORIO AL ADHERENTE (adhesivo.notificacion)",
                            "FALTA LA FECHA DE PRESENTACIÓN DEL ADHESIVO (adhesivo.presentacion)",
                            "EL VEREDICTO DE OPORTUNIDAD DEL AMPARO ADHESIVO VA EN HUECO: sin las dos fechas…"])
ok(isinstance(r, list) and not any(x.startswith("HUECO") for x in r),
   "…y con los tres ya avisados por el compositor (avisos_previos), ninguno")
r = _seguro(vp.revisar, _t_adh, FICHA_CDMX, DATOS_CDMX, "amparo_directo",
            avisos_previos=["FALTA LA FECHA DE NOTIFICACIÓN DEL AUTO ADMISORIO AL ADHERENTE (adhesivo.notificacion)"])
ok(isinstance(r, list) and any(x.startswith("HUECO") and "adhesivo.presentacion" in x for x in r)
   and not any(x.startswith("HUECO") and "adhesivo.notificacion" in x for x in r),
   "…con uno solo avisado, el otro sigue (nada se pierde)")
try:
    import documento_generado as _dg
    _hya = getattr(_dg, "_hueco_ya_avisado", None)
except Exception:
    _hya = None
if callable(_hya):
    _esp = ["FALTA LA FECHA EN QUE SE NOTIFICÓ AL ADHERENTE EL AUTO ADMISORIO DE LA DEMANDA (adhesivo.notificacion): "
            "está en la constancia. Va en hueco en «Legitimación y oportunidad del amparo adhesivo».",
            "FALTA LA FECHA DE PRESENTACIÓN DEL ESCRITO DEL ADHESIVO (adhesivo.presentacion): está en el sello. Va en "
            "hueco en «Legitimación y oportunidad del amparo adhesivo»."]
    ok(all(_hya(x, _esp) for x in d[:2]),
       "con el detalle de la verja, documento_generado._hueco_ya_avisado calla los dos campos ya avisados")

t = con(AR, f"El conocimiento del asunto correspondió al {JUZ}, que", "El conocimiento del asunto correspondió a "
        "*********, que")
d = [x for x in vp.revisar_detalle(t, FICHA_AR, DATOS_AR, "amparo_revision") if x["regla"] == "a"]
ok(len(d) == 1 and d[0].get("clave") == "acto.organo",
   "AR 307/448/60: «correspondió a *********, que lo registró…» es el juzgado (acto.organo), no «el dato que va ahí»")

# ── (j) LA CRONOLOGÍA DEL AMPARO INDIRECTO EN EL AR (307, 448 y 60 del banco) ──
t = con(AR, "el diez de abril de dos mil veintiséis se celebró la audiencia constitucional y se dictó sentencia",
        "el veinte de abril de dos mil veintiséis se celebró la audiencia constitucional y el diez de abril de dos mil "
        "veintiséis se dictó sentencia")
ok(any("audiencia" in x and "posterior a la sentencia" in x for x in avisos_de(t, FICHA_AR, DATOS_AR, "amparo_revision",
                                                                                 "j")),
   "la audiencia posterior a la sentencia")
t = con(AR, "Por escrito presentado el tres de noviembre de dos mil veinticinco",
        "Por escrito presentado el tres de mayo de dos mil veintiséis")
a = avisos_de(t, FICHA_AR, DATOS_AR, "amparo_revision", "j")
ok(any("demanda de amparo" in x and "posterior a la sentencia" in x for x in a), "la demanda posterior a la sentencia")
a = avisos_de(t, dict(FICHA_AR, fechas_imposibles=["LA DEMANDA ES POSTERIOR A LA SENTENCIA"]), DATOS_AR,
              "amparo_revision", "j")
ok(len(a) == 1 and "FICHA DE TRÁMITE" in a[0], "…y si la ficha ya lo dice, no se repite")
ok(not avisos_de(AR, FICHA_AR, DATOS_AR, "amparo_revision", "j"), "el AR bien fechado, nada")
# Si la ficha misma ya trae la audiencia después de la sentencia, `ficha_tramite.validar` lo dice (el compositor suma
# sus avisos, AR 448 del banco) y la verja no lo repite; si validar no lo dice, lo dice la verja.
_f_aud = dict(FICHA_AR, audiencia="2026-04-20")
_t_aud = con(AR, "el diez de abril de dos mil veintiséis se celebró la audiencia constitucional y se dictó sentencia",
             "el veinte de abril de dos mil veintiséis se celebró la audiencia constitucional y el diez de abril de dos "
             "mil veintiséis se dictó sentencia")
try:
    import copy as _copy
    import ficha_tramite as _ftr
    _val = " ".join(str(x).lower() for x in (_ftr.validar(_copy.deepcopy(_f_aud)) or []))
except Exception:
    _val = ""
_dice_validar = "audiencia" in _val and "posterior" in _val
a = [x for x in avisos_de(_t_aud, _f_aud, DATOS_AR, "amparo_revision", "j") if "TRÁMITE DEL JUICIO DE AMPARO" in x]
ok((not a) if _dice_validar else len(a) == 1,
   "la audiencia posterior en la ficha: " + ("la dice validar y la verja no la repite" if _dice_validar
                                             else "validar no la dice, la dice la verja"))

# ── EL RENGLÓN RECURRENTE DE LA CARÁTULA (Q 300 del banco; rev_0) ──
_f_tercero = dict(FICHA_Q, caracter="tercero", promovente="Inmobiliaria Los Arcos, S.A. de C.V.")
a = [x for x in avisos_de(Q, _f_tercero, DATOS_Q, "queja", "d") if "RENGLÓN DE QUIEN RECURRE" in x]
ok(len(a) == 1 and "JUAN PÉREZ LÓPEZ" in a[0], "recurre la tercera y la carátula dice «RECURRENTE:» con el quejoso")
_q_bien = con(Q, "RECURRENTE: JUAN PÉREZ LÓPEZ.", "QUEJOSO: JUAN PÉREZ LÓPEZ.\nTERCERA INTERESADA Y RECURRENTE: "
              "INMOBILIARIA LOS ARCOS, S.A. DE C.V.")
ok(not any("RENGLÓN DE QUIEN RECURRE" in x for x in avisos_de(_q_bien, _f_tercero, DATOS_Q, "queja", "d")),
   "…con el renglón de quien recurre, nada")
ok(not any("RENGLÓN DE QUIEN RECURRE" in x for x in avisos_de(Q, dict(_f_tercero, promovente="PARTE RECURRENTE"),
                                                              DATOS_Q, "queja", "d")),
   "con el nombre testado («PARTE RECURRENTE») no se compara")

# ── EL TITULAR Y SU ÓRGANO: «Juez/Jueza…», «Titular del Juzgado…» y «el Juzgado…» (casos de la queja) ──
for _org in (JUZ3.replace("Juzgado Tercero", "Jueza Tercera").upper(), JUZ3.replace("Juzgado Tercero", "Juez Tercero"),
             "Titular del " + JUZ3):
    ok(not avisos_de(Q, dict(FICHA_Q, acto=dict(FICHA_Q["acto"], organo=_org)), dict(DATOS_Q, organo_recurrido=""),
                     "queja", "d"), f"la ficha dice «{_org[:40]}…» y el texto «el Juzgado Tercero…»: el mismo órgano")
ok(any("otro órgano" in x for x in avisos_de(Q, dict(FICHA_Q, acto=dict(FICHA_Q["acto"], organo=JUZ3.replace(
    "Juzgado Tercero", "Juez Quinto"))), dict(DATOS_Q, organo_recurrido=""), "queja", "d")),
   "…pero el Juez QUINTO no es el Juzgado Tercero")

# ── NINGUNA REGLA TROPIEZA (las reglas que leen `out` con los avisos de hueco que ya traen su clave) ──
import contextlib as _ctx
import io as _io
_buf = _io.StringIO()
_ad_imposible = con(AD.replace("veintidós de enero de dos mil veintiséis", "veintidós de enero de dos mil veintisiete"),
                    "inciso c), de la Constitución", "inciso *********), de la Constitución")
with _ctx.redirect_stdout(_buf):
    for _t, _f, _d, _tp in ((_ad_imposible, FICHA_AD, DATOS_AD, "amparo_directo"),
                            (_t_surte, dict(FICHA_AD, forma_notificacion="lista"), DATOS_AD, "amparo_directo"),
                            (_q_sin_organo, FICHA_Q, DATOS_Q, "queja"),
                            (_t_adh, FICHA_CDMX, DATOS_CDMX, "amparo_directo"),
                            (RF, dict(FICHA_RF, forma_notificacion="personal"), DATOS_RF, "revision_fiscal"),
                            (AR, dict(FICHA_AR, clase_recurrida="auto_sobreseimiento"), DATOS_AR, "amparo_revision")):
        vp.revisar_detalle(_t, _f, _d, _tp)
ok("una regla falló" not in _buf.getvalue(), "ninguna regla falla con huecos, clase, forma y fechas a la vez"
   + (f" — {_buf.getvalue()[:200]}" if "una regla falló" in _buf.getvalue() else ""))
ok(any("antes de que existiera" in x for x in avisos_de(_ad_imposible, None, DATOS_AD, "amparo_directo", "j")),
   "la fecha imposible del texto sale aunque haya un hueco con su clave (la regla no tropieza)")
t = con(RF, "artículo 63, fracción III, de la Ley Federal de Procedimiento Contencioso Administrativo.",
        "artículo 63, fracción *********, de la Ley Federal de Procedimiento Contencioso Administrativo, toda vez que "
        "*********.")
a = avisos_de(t, FICHA_RF, DATOS_RF, "revision_fiscal", "a")
ok(len(a) == 1 and "(fraccion_63)" in a[0] and "y 1 más en el mismo apartado" in a[0],
   "RF: «fracción *********, … toda vez que *********»: la razón la da la fracción, un solo aviso con su campo")
a = avisos_de(con(t, "toda vez que *********.", "toda vez que el asunto es de importancia y trascendencia, relativa a "
                  "*********."), FICHA_RF, DATOS_RF, "revision_fiscal", "a")
ok(len(a) == 2 and any("de qué trata la resolución impugnada" in x for x in a),
   "…«relativa a *********» (fr. VI, RF 4, 26, 49 del banco) se nombra aparte: no lo da la fracción")

# ── NUNCA LENTA: 200 000 caracteres sin puntos y con representaciones ──
import time as _time
_largo = ("V I S T O, para resolver el recurso de revisión fiscal 42/2026; y,\nR E S U L T A N D O\nPRIMERO. Interposición. "
          + ("el Jefe de la Unidad Jurídica, en representación del Jefe de la Unidad Jurídica, lo hizo valer el JEFA DE "
             "LA UNIDAD " * 1500)[:200000] + "\nC O N S I D E R A N D O\nPRIMERO. Competencia. Este tribunal es competente")
_t0 = _time.time()
_r = _seguro(vp.revisar, _largo, FICHA_RF, DATOS_RF, "revision_fiscal")
ok(isinstance(_r, list) and _time.time() - _t0 < 5, f"200 000 caracteres sin puntos: {_time.time() - _t0:.2f} s")

# ═══════════════════════════════════════════════════════════════════════════
# 18 · RONDA 4 (3-oct-2026): la verificación final contra las 32 sentencias
#      reales (rev/verif_final.txt) y las decisiones E1-E12 de FIXES_R4
# ═══════════════════════════════════════════════════════════════════════════
print("\n18 · RONDA 4: VERIFICACIÓN FINAL (E2, E8, E11 Y LAS PARTES EN LA PROSA)")


def _d(texto, ficha, datos, tipo, regla, **kw):
    r = _seguro(vp.revisar_detalle, texto, ficha, datos, tipo, **kw)
    return [x["aviso"] for x in r if x["regla"] == regla] if isinstance(r, list) else [repr(r)]


# ── E11: EL ÓRGANO QUE DECIDIÓ EL COMPOSITOR MANDA SOBRE LO CRUDO DE LA FICHA (Q 261 con la ficha vieja) ──
_JUEZ_CORTO = "Juez Tercero de Distrito de Amparo y Juicios Federales en el Estado de Querétaro"
_f_q261 = dict(FICHA_Q, acto=dict(FICHA_Q["acto"], organo=_JUEZ_CORTO), responsable=JUZ3)
a = _d(Q, _f_q261, dict(DATOS_Q, procesal={"juzgado": JUZ3, "organo_acto": JUZ3}), "queja", "d")
ok(not a, "Q 261: la ficha trae el juzgado de dos formas y el compositor eligió el nombre completo: se coteja contra "
          "procesal.juzgado, no contra acto.organo (eran 3 avisos falsos)" + (f" — {a[:1]}" if a else ""))
a = _d(Q, _f_q261, dict(DATOS_Q, procesal={"juzgado": JUZ}), "queja", "d")
ok(bool(a) and all("compositor" in x for x in a if "NO SE ESCRIBE IGUAL" in x),
   "…pero el texto que se aparta de lo que decidió el compositor se acusa, y el aviso lo dice")
ok(not _d(Q, FICHA_Q, dict(DATOS_Q, procesal={}), "queja", "d"), "…y sin compositor, contra la ficha como siempre")

# ── E10/E11: EL ÓRGANO DESCARTADO QUE SIGUE EN LA COMPETENCIA Y LA EXISTENCIA (AR 307, 448 y 60, fichas viejas) ──
_SALA_FAM = "Sala Familiar del Tribunal Superior de Justicia del Estado de Querétaro"
_ar_desc = con(con(con(con(AR,
    f"por el {JUZ}, en el juicio de amparo indirecto 950/2025; y,", "por *********, en el juicio de amparo indirecto "
    "950/2025; y,"),
    f"correspondió al {JUZ}, que", "correspondió a *********, que"),
    f"en materia administrativa, por el {JUZ}, localizado", f"en materia administrativa, por la {_SALA_FAM}, localizada"),
    f"que remitió el {JUZ} en términos", f"que remitió la {_SALA_FAM} en términos")
_f_desc = dict(FICHA_AR, acto=dict(FICHA_AR["acto"], organo=_SALA_FAM),
               demanda=dict(FICHA_AR["demanda"], autoridades=[_SALA_FAM]))
for _nombre, _proc in (("procesal.juzgado vacío", {"juzgado": ""}),
                       ("procesal.juzgado en hueco", {"juzgado": "*********"}),
                       ("procesal.juzgado_descartado", {"juzgado": "*********", "juzgado_descartado": _SALA_FAM})):
    a = _d(_ar_desc, _f_desc, dict(DATOS_AR, organo_recurrido=_SALA_FAM, procesal=_proc), "amparo_revision", "d")
    ok(len(a) == 1 and "SE DESCARTÓ SIGUE EN EL TEXTO" in a[0] and "Competencia" in a[0] and "Existencia" in a[0],
       f"AR 307: el compositor descartó la Sala Familiar ({_nombre}) y la competencia y la existencia la siguen "
       f"nombrando: un aviso con los dos apartados" + (f" — {a[:2]}" if len(a) != 1 else ""))
_ar_hueco = con(con(_ar_desc, f"por la {_SALA_FAM}, localizada", "por *********, localizado"),
                f"que remitió la {_SALA_FAM} en términos", "que remitió ********* en términos")
ok(not _d(_ar_hueco, _f_desc, dict(DATOS_AR, organo_recurrido="*********", procesal={"juzgado": "*********"}),
          "amparo_revision", "d"),
   "…con el hueco en todos los apartados (E10 hecho), nada")

# ── E11: LA CRONOLOGÍA QUE YA DIJO validar NO SE REPITE (AR 448 del banco) ──
_ad_pres = con(AD, "Por escrito presentado el doce de febrero de dos mil veintiséis ante",
               "Por escrito presentado el doce de enero de dos mil veintiséis ante")
_f_pres = dict(FICHA_AD, presentacion="2026-01-12", fechas_imposibles=["presentacion < acto.fecha"])
try:
    import ficha_tramite as _ftr4
    _val4 = " ".join(_ftr4.validar(_copy.deepcopy(dict(_f_pres, fechas_imposibles=[]))) or [])
except Exception:
    _val4 = ""
_dice4 = "12/01/2026" in _val4 and "22/01/2026" in _val4
a = _d(_ad_pres, _f_pres, DATOS_AD, "amparo_directo", "j")
ok((not a) if _dice4 else (len(a) >= 1),
   ("la presentación anterior al acto, que validar ya dice con las mismas fechas: ni el par del texto ni «EN LA FICHA "
    "DE TRÁMITE: presentacion < acto.fecha»" if _dice4 else "validar no lo dice: lo dice la verja")
   + (f" — {a[:2]}" if (a and _dice4) else ""))
a = _d(_ad_pres, FICHA_AD, DATOS_AD, "amparo_directo", "j")
ok(len(a) == 1 and "antes de que existiera" in a[0],
   "…si el texto se apartó de la ficha (la ficha trae la fecha buena), validar no lo dice y la verja sí")
_prev4 = ["FECHA IMPOSIBLE: la demanda se presentó el 12/01/2026 y la sentencia reclamada es del 22/01/2026. Nadie "
          "combate una resolución que aún no existe; revisa las dos fechas."]
a = _d(_ad_pres, dict(FICHA_AD, presentacion=""), DATOS_AD, "amparo_directo", "j", avisos_previos=_prev4)
ok(not a, "…ni lo que ya avisó otra pieza con las mismas dos fechas (avisos_previos, en cifras)"
   + (f" — {a[:1]}" if a else ""))
a = _d(_ad_pres, dict(FICHA_AD, presentacion=""), DATOS_AD, "amparo_directo", "j",
       avisos_previos=["FECHA IMPOSIBLE: la demanda se presentó el 12/01/2026 y la sentencia es del 23/01/2026."])
ok(len(a) == 1, "…pero otro par de fechas no lo calla")
a = _d(_t_aud, dict(FICHA_AR, fechas_imposibles=["LA FECHA DEL TURNO ES ANTERIOR A LA ADMISIÓN"]), DATOS_AR,
       "amparo_revision", "j")
ok(any("TRÁMITE DEL JUICIO DE AMPARO" in x for x in a) and any("FICHA DE TRÁMITE" in x for x in a),
   "AR: una fecha imposible de otra cosa en la ficha ya no calla la cronología del juicio (audiencia tras sentencia)")

# ── LAS FÓRMULAS DE REPRESENTACIÓN CON MAYÚSCULA A MEDIA FRASE (AD 552, Q 342, Q 229) ──
_AGR = "Agrícola Los Pinos, Sociedad de Producción Rural de Responsabilidad Ilimitada"
t = con(AD, "promovido por Ruth Gabriela Larrañaga Silva, contra la sentencia",
        f"promovido por {_AGR}, por Conducto de Su Consejo de Administración, y Ruth Gabriela Larrañaga Silva, por "
        f"Propio Derecho, contra la sentencia")
a = [x for x in _d(t, FICHA_AD, DATOS_AD, "amparo_directo", "f") if "MAYÚSCULAS A MEDIA FRASE" in x]
ok(len(a) == 1 and "por Conducto de Su" in a[0] and "V I S T O" in a[0],
   "AD 552: «por Conducto de Su Consejo de Administración», «por Propio Derecho»")
t2 = con(t, "La demanda de amparo fue promovida por Ruth Gabriela Larrañaga Silva, quien",
         "La demanda de amparo fue promovida por Ruth Gabriela Larrañaga Silva, por Propio Derecho, quien")
a = [x for x in _d(t2, FICHA_AD, DATOS_AD, "amparo_directo", "f") if "MAYÚSCULAS A MEDIA FRASE" in x]
ok(len(a) == 1 and "V I S T O" in a[0] and "Legitimación" in a[0], "…un solo aviso con todos los apartados")
for _frase in ("la Sucesión Testamentaria a Bienes de María Pérez López, a Través de Su Albacea Juan Pérez López",
               "Juan Pérez López y Otra", "Juan Pérez López, por Su Propio Derecho y en Representación de la Menor"):
    t = con(Q, "interpuesto por Juan Pérez López, en contra del auto", f"interpuesto por {_frase}, en contra del auto")
    ok(any("MAYÚSCULAS A MEDIA FRASE" in x for x in _d(t, FICHA_Q, DATOS_Q, "queja", "f")), f"«…{_frase[-50:]}» se acusa")
for _frase in ("la Sucesión Testamentaria a Bienes de María Pérez López, a través de su albacea Juan Pérez López",
               "Servicios y Otros Conceptos del Bajío, S.A. de C.V.",
               "el Titular de la Unidad Jurídica de la Oficina de Representación del ISSSTE en Querétaro",
               "Juan Pérez López, por propio derecho y en representación de la menor"):
    t = con(Q, "interpuesto por Juan Pérez López, en contra del auto", f"interpuesto por {_frase}, en contra del auto")
    ok(not any("MAYÚSCULAS A MEDIA FRASE" in x for x in _d(t, FICHA_Q, DATOS_Q, "queja", "f")),
       f"«…{_frase[-50:]}» no se acusa")

# ── LA RACHA EN VERSALES DENTRO DE LA MENCIÓN (AD 335, AR 208, AR 60) ──
t = con(AD, "promovido por Ruth Gabriela Larrañaga Silva, contra la sentencia",
        "promovido por la Sucesión a Bienes de JOSÉ GARCÍA RUIZ, contra la sentencia")
a = [x for x in _d(t, FICHA_AD, DATOS_AD, "amparo_directo", "d") if "VERSALES" in x]
ok(len(a) == 1 and "JOSÉ GARCÍA RUIZ" in a[0] and "V I S T O" in a[0],
   "AD 335: «promovido por la Sucesión a Bienes de JOSÉ GARCÍA RUIZ» (la racha, no el principio de la mención)")
t = con(AD, "no ampara ni protege a Ruth Gabriela Larrañaga Silva, contra",
        "no ampara ni protege a Sucesión a Bienes de JOSÉ GARCÍA RUIZ, contra")
a = _d(t, FICHA_AD, DATOS_AD, "amparo_directo", "d")
ok(any("VERSALES" in x and "resolutivos" in x for x in a), "…«no ampara ni protege a … JOSÉ GARCÍA RUIZ» en el resolutivo")
t = con(AR, f"{COM}, promovió juicio de amparo indirecto",
        "PEDRO y JUAN, ambos de apellidos PÉREZ LÓPEZ, promovieron juicio de amparo indirecto")
a = _d(t, FICHA_AR, DATOS_AR, "amparo_revision", "d")
ok(any("VERSALES" in x and "PÉREZ LÓPEZ" in x for x in a),
   "AR 208: «…, ambos de apellidos PÉREZ LÓPEZ, promovieron juicio…» (el nombre antes del verbo)")
for _x in ("Comercializadora del Centro, S.A. DE C.V.", "el Sindicato de Trabajadores C.T.M.",
           "la Sala Regional del Centro II", "PARTE QUEJOSA, por conducto de REPRESENTANTE"):
    t = con(AD, "promovido por Ruth Gabriela Larrañaga Silva, contra", f"promovido por {_x}, contra")
    ok(not any("VERSALES" in x for x in _d(t, FICHA_AD, DATOS_AD, "amparo_directo", "d")),
       f"«promovido por {_x}»: siglas, romanos o la figura del banco, no un nombre en versales")
t = con(AD, "no ampara ni protege a Ruth Gabriela Larrañaga Silva, contra la sentencia dictada",
        "no ampara ni protege a Ruth Gabriela Larrañaga Silva, contra la RESOLUCIÓN DEFINITIVA dictada")
ok(not any("VERSALES" in x for x in _d(t, FICHA_AD, DATOS_AD, "amparo_directo", "d")),
   "lo que va después del nombre («contra la RESOLUCIÓN…») no es la parte")

# ── EL COLECTIVO SIN SU ARTÍCULO (AD 335, AR 60) ──
for _frase, _art in (("no ampara ni protege a Sucesión a Bienes de José García Ruiz", "la Sucesión"),
                     ("no ampara ni protege a sucesión intestamentaria a bienes de José García Ruiz", "la sucesión"),
                     ("no ampara ni protege a Ejido San Juan del Río", "el Ejido")):
    t = con(AD, "no ampara ni protege a Ruth Gabriela Larrañaga Silva", _frase)
    a = [x for x in _d(t, FICHA_AD, DATOS_AD, "amparo_directo", "f") if "ARTÍCULO DEL COLECTIVO" in x]
    ok(len(a) == 1 and _art in a[0], f"«{_frase[:55]}…»: va «{_art}»")
for _frase in ("no ampara ni protege a la Sucesión a Bienes de José García Ruiz",
               "no ampara ni protege al Ejido San Juan del Río", "no ampara ni protege a dicha sucesión"):
    t = con(AD, "no ampara ni protege a Ruth Gabriela Larrañaga Silva", _frase)
    ok(not any("ARTÍCULO DEL COLECTIVO" in x for x in _d(t, FICHA_AD, DATOS_AD, "amparo_directo", "f")),
       f"«{_frase[:55]}» no se acusa")

# ── «X, POR CONDUCTO DE SU X» Y EL CARGO IGUAL A SU ÓRGANO (E8; RF 7, 2, 26 y 6-2026 del banco) ──
_JEFA_BCS = ("la Jefa de la Unidad Jurídica de la Delegación Estatal del Instituto de Seguridad y Servicios Sociales de "
             "los Trabajadores del Estado en Baja California Sur")
for _nombre, _txt in (
        ("RF 7: «la Jefa de la Unidad Jurídica…, por conducto de su Jefa de la Unidad Jurídica»",
         f"Inconforme, {_JEFA_BCS}, por conducto de su Jefa de la Unidad Jurídica, Martha Ruiz Gómez, en "
         f"representación de la autoridad demandada,"),
        ("RF 2: «…, por conducto de su Titular de la Unidad Jurídica…»",
         f"Inconforme, {_JEFA_BCS}, por conducto de su Titular de la Unidad Jurídica…, Martha Ruiz Gómez, en "
         f"representación de la autoridad demandada,"),
        ("RF 26: «el Subdirector de lo Contencioso…, por conducto de su Subdirector de lo Contencioso»",
         "Inconforme, el Subdirector de lo Contencioso del Instituto de Seguridad y Servicios Sociales de los "
         "Trabajadores del Estado, por conducto de su Subdirector de lo Contencioso, Pedro Ruiz Gómez, en "
         "representación de la autoridad demandada,")):
    a = _d(con(RF, _rep, _txt), FICHA_RF, DATOS_RF, "revision_fiscal", "d")
    ok(any("POR CONDUCTO DE SU PROPIO CARGO" in x for x in a), _nombre)
for _nombre, _txt in (
        ("el Subdelegado por conducto de la unidad jurídica (lo que pide el art. 63)",
         "Inconforme, el Subdelegado de Prestaciones Económicas de la Delegación Estatal en Querétaro del Instituto de "
         "Seguridad y Servicios Sociales de los Trabajadores del Estado, por conducto de su Titular de la Unidad "
         "Jurídica, Martha Ruiz Gómez,"),
        ("el Instituto por conducto de su Subdirector de lo Contencioso",
         "Inconforme, el Instituto de Seguridad y Servicios Sociales de los Trabajadores del Estado, por conducto de "
         "su Subdirector de lo Contencioso, Pedro Ruiz Gómez,"),
        ("una persona moral por conducto de su apoderado",
         "Inconforme, Comercializadora del Centro, S.A. de C.V., por conducto de su apoderado legal, Pedro Ruiz Gómez,")):
    a = _d(con(RF, _rep, _txt), FICHA_RF, DATOS_RF, "revision_fiscal", "d")
    ok(not any("PROPIO CARGO" in x or "SÍ MISMO" in x for x in a), f"{_nombre}: no se acusa")
_t_sub = con(RF, _rep, "Inconforme, el Subdelegado de Prestaciones Económicas, Delegación Estatal Querétaro, del "
             "Instituto de Seguridad y Servicios Sociales de los Trabajadores del Estado, por conducto de su Titular de "
             "la Unidad de Asuntos Jurídicos de la citada delegación, en representación de la Subdelegación de "
             "Prestaciones Económicas del Instituto de Seguridad y Servicios Sociales de los Trabajadores del Estado en "
             "Querétaro,")
ok(any("SÍ MISMO" in x for x in _d(_t_sub, FICHA_RF, DATOS_RF, "revision_fiscal", "d")),
   "RF 6-2026: «el Subdelegado…, en representación de la Subdelegación…» (el cargo y su órgano son la misma autoridad)")
_t_tit = con(RF, _rep, "Inconforme, el Titular de la Subdelegación de Prestaciones Económicas de la Delegación Estatal "
             "en Querétaro, en representación del Subdelegado de Prestaciones Económicas de la Delegación Estatal en "
             "Querétaro,")
ok(any("SÍ MISMO" in x for x in _d(_t_tit, FICHA_RF, DATOS_RF, "revision_fiscal", "d")),
   "«el Titular de la Subdelegación…, en representación del Subdelegado…»: «Titular de la X» es X")
_leg = "El recurso lo interpone el " + UJ + ", unidad encargada"
_t_leg = con(RF, _leg, "El recurso fue interpuesto por parte legítima, pues lo hizo valer la Jefa de la Unidad Jurídica "
             "de la Delegación Estatal en Querétaro del Instituto de Seguridad y Servicios Sociales de los Trabajadores "
             "del Estado, por conducto de Martha Ruiz Gómez, en su carácter de unidad administrativa encargada")
ok(any("EN SU CARÁCTER DE UNIDAD ADMINISTRATIVA" in x for x in _d(_t_leg, FICHA_RF, DATOS_RF, "revision_fiscal", "d")),
   "RF 7: la legitimación «…, por conducto de Fulana, en su carácter de unidad administrativa» (una persona no es "
   "una unidad)")
_t_leg2 = con(RF, _leg, "El recurso fue interpuesto por parte legítima, pues lo hizo valer Martha Ruiz Gómez, Jefa de la "
              "Unidad Jurídica de la Delegación Estatal en Querétaro del Instituto de Seguridad y Servicios Sociales de "
              "los Trabajadores del Estado, unidad encargada")
ok(not any("UNIDAD ADMINISTRATIVA" in x or "PROPIO CARGO" in x
           for x in _d(_t_leg2, FICHA_RF, DATOS_RF, "revision_fiscal", "d")),
   "…«lo hizo valer Fulana, Jefa de la Unidad Jurídica…» (la forma del corpus) no se acusa")

# ── E2: PLURAL EN EL RESULTANDO, SINGULAR EN LA LEGITIMACIÓN Y LA CARÁTULA (AD 552) ──
_ad_pl = con(con(con(AD, "Ruth Gabriela Larrañaga Silva promovió juicio de amparo directo",
                     "Ruth Gabriela Larrañaga Silva y Alma Delia Ruiz Torres promovieron juicio de amparo directo"),
                 "La demanda de amparo fue promovida por Ruth Gabriela Larrañaga Silva, quien",
                 "La demanda de amparo fue promovida por Ruth Gabriela Larrañaga Silva y Alma Delia Ruiz Torres, quien"),
             "QUEJOSA: RUTH GABRIELA LARRAÑAGA SILVA.", "QUEJOSA: RUTH GABRIELA LARRAÑAGA SILVA Y ALMA DELIA RUIZ TORRES.")
a = [x for x in _d(_ad_pl, FICHA_AD, DATOS_AD, "amparo_directo", "f") if "VARIOS Y LUEGO QUE ES UNO" in x]
ok(len(a) == 1 and "promovieron" in a[0] and "quien está legitimada" in a[0] and "carátula" in a[0],
   "AD 552: «promovieron» en el resultando, «quien está legitimada» y «QUEJOSA:»: un aviso con los dos sitios")
_ad_pl_bien = con(con(_ad_pl, ", quien está legitimada para ello", "; la parte quejosa está legitimada para ello"),
                  "QUEJOSA: RUTH", "PARTE QUEJOSA: RUTH")
ok(not any("VARIOS Y LUEGO" in x for x in _d(_ad_pl_bien, FICHA_AD, DATOS_AD, "amparo_directo", "f")),
   "…con «la parte quejosa está legitimada» y «PARTE QUEJOSA:» (E2), nada")
_ad_pl_bien2 = con(con(_ad_pl, ", quien está legitimada para ello", ", quienes están legitimadas para ello"),
                   "QUEJOSA: RUTH", "QUEJOSAS: RUTH")
ok(not any("VARIOS Y LUEGO" in x for x in _d(_ad_pl_bien2, FICHA_AD, DATOS_AD, "amparo_directo", "f")),
   "…con «están legitimadas» y «QUEJOSAS:», nada")
_ar_pl = con(AR, f"{COM}, promovió juicio de amparo indirecto",
             f"{COM} y Juan Pérez López promovieron juicio de amparo indirecto")
ok(not any("VARIOS Y LUEGO" in x for x in _d(_ar_pl, FICHA_AR, DATOS_AR, "amparo_revision", "f")),
   "AR: varios quejosos en la demanda y UNA recurrente: el singular de la legitimación es el de quien recurre")
_q_pl = con(Q, "Juan Pérez López, quejoso en el juicio de amparo indirecto 1201/2026, interpuso recurso de queja",
            "Juan Pérez López y Ana Ruiz Gómez, quejosos en el juicio de amparo indirecto 1201/2026, interpusieron "
            "recurso de queja")
a = [x for x in _d(_q_pl, FICHA_Q, DATOS_Q, "queja", "f") if "VARIOS Y LUEGO" in x]
ok(len(a) == 1 and "por lo que está legitimado" in a[0] and "RECURRENTE:" in a[0],
   "Q: «interpusieron» y «por lo que está legitimado» / «RECURRENTE:»")

# ── E7/E11: EL HUECO DEL AUTO QUE REGISTRA LLEVA LA CLAVE DEL COMPOSITOR (registro.fecha; Q 335, RF 7) ──
_TRAM_Q = ("Por auto de Presidencia de nueve de junio de dos mil veintiséis, este Tribunal Colegiado registró el recurso "
           "con el número 355/2026 y lo admitió a trámite.")
_q_ii = con(Q, _TRAM_Q, "Por auto de Presidencia de *********, este Tribunal Colegiado registró el recurso con el número "
            "355/2026 y requirió a la autoridad responsable su informe con justificación sobre la materia de la queja "
            "(artículo 101 de la Ley de Amparo); por auto de nueve de junio de dos mil veintiséis lo tuvo por rendido y "
            "admitió el recurso.")
d = [x for x in vp.revisar_detalle(_q_ii, FICHA_Q, DATOS_Q, "queja") if x["regla"] == "a"]
ok(len(d) == 1 and d[0].get("clave") == "registro.fecha" and "(registro.fecha)" in d[0]["aviso"],
   "queja fr. II: el auto que registra y requiere el informe en hueco es registro.fecha, no admision.fecha"
   + (f" — {[x.get('clave') for x in d]}" if not (len(d) == 1 and d[0].get("clave") == "registro.fecha") else ""))
r = _seguro(vp.revisar, _q_ii, FICHA_Q, DATOS_Q, "queja", avisos_previos=[
    "FALTA LA FECHA DEL AUTO DE PRESIDENCIA QUE REGISTRÓ EL RECURSO Y REQUIRIÓ EL INFORME CON JUSTIFICACIÓN "
    "(registro.fecha): en la queja de la fracción II es otro auto… Va en hueco en «Trámite del recurso»."])
ok(isinstance(r, list) and not any(x.startswith("HUECO") for x in r),
   "…y con el aviso del compositor «(registro.fecha)» como previo, la verja no lo repite (los 2 casos fr. II del "
   "compositor: 2 avisos dobles → 0)")
_q_dos = con(Q, _TRAM_Q, "Por auto de Presidencia de nueve de junio de dos mil veintiséis, este Tribunal Colegiado "
             "registró el recurso con el número 355/2026; y por auto de ********* lo admitió a trámite.")
d = [x for x in vp.revisar_detalle(_q_dos, FICHA_Q, DATOS_Q, "queja") if x["regla"] == "a"]
ok(len(d) == 1 and d[0].get("clave") == "admision.fecha",
   "dos autos (RF 7): «…; y por auto de ********* lo admitió a trámite» es admision.fecha")
_q_uno = con(Q, _TRAM_Q, "Por auto de Presidencia de *********, este Tribunal Colegiado registró el recurso con el número "
             "355/2026 y lo admitió a trámite.")
d = [x for x in vp.revisar_detalle(_q_uno, FICHA_Q, DATOS_Q, "queja") if x["regla"] == "a"]
ok(len(d) == 1 and d[0].get("clave") == "admision.fecha", "un solo auto que registra y admite: admision.fecha, como antes")
_ar_adh = con(AR, "lo registró con el número 410/2026 y lo admitió a trámite.",
              "lo registró con el número 410/2026 y lo admitió a trámite. Por auto de *********, se tuvo a "
              "Comercializadora Ejemplo del Centro y del Bajío, S.A. de C.V. interponiendo revisión adhesiva.")
d = [x for x in vp.revisar_detalle(_ar_adh, FICHA_AR, DATOS_AR, "amparo_revision") if x["regla"] == "a"]
ok(len(d) == 1 and d[0].get("clave") == "adhesivo.admision",
   "«se tuvo a [nombre largo] interponiendo revisión adhesiva»: la clave se lee hasta el fin de la frase")

# ── EL PREFIJO TEMPORAL NO HACE OTRA SALA (RF 49 y 21 del banco, ronda 4) ──
_SALA_TFJA = f"{SALA_RF} del Tribunal Federal de Justicia Administrativa"
_rf_act = RF.replace(f"por la {_SALA_TFJA}", f"por la actual {_SALA_TFJA}")
a = _d(_rf_act, dict(FICHA_RF, sala="actual " + SALA_RF), dict(DATOS_RF, organo_recurrido=""), "revision_fiscal", "d")
ok(not a, "RF 49: la carátula «SALA REGIONAL EN QUERÉTARO…» y el dato «actual Sala Regional en Querétaro»: la misma Sala"
   + (f" — {a[:1]}" if a else ""))
_rf_ent = RF.replace(f"dictada por la {_SALA_TFJA}, en el juicio",
                     f"dictada por la entonces Sala Regional del Centro II, ahora {_SALA_TFJA}, en el juicio")
a = _d(_rf_ent, FICHA_RF, DATOS_RF, "revision_fiscal", "d")
ok(not a, "RF 21: «la entonces Sala Regional del Centro II, ahora Sala Regional en Querétaro…»: cuenta el nombre de hoy"
   + (f" — {a[:1]}" if a else ""))
a = _d(RF.replace(f"dictada por la {_SALA_TFJA}, en el juicio", "dictada por la Sala Regional del Centro II del "
                  "Tribunal Federal de Justicia Administrativa, en el juicio"), FICHA_RF, DATOS_RF, "revision_fiscal", "d")
ok(any("NO SE ESCRIBE IGUAL" in x for x in a), "…pero el nombre viejo SOLO sí es otro nombre: se acusa")

# ── E5: EL ADHESIVO SIN NINGÚN AUTO NO LLEVA CONSIDERANDO NI RESOLUTIVO (AD 274 del banco) ──
_f_sin_auto = dict(FICHA_AD, adhesivo={"quien": "Alejandro Daniel Betancourt Velarde"})
_ad_sin_auto = con(AD, "en términos del artículo 181 de la Ley de Amparo.",
                   "en términos del artículo 181 de la Ley de Amparo. Por auto de *********, se admitió el amparo "
                   "adhesivo promovido por Alejandro Daniel Betancourt Velarde.")
a = _d(_ad_sin_auto, _f_sin_auto, DATOS_AD, "amparo_directo", "k")
ok(not a, "E5: consta quién pero ni la admisión ni la presentación: el resultando lo menciona y la verja no pide su "
          "considerando ni su resolutivo (decía lo contrario que el compositor)" + (f" — {a[:1]}" if a else ""))
a = _d(AD, _f_sin_auto, DATOS_AD, "amparo_directo", "k")
ok(len(a) == 1 and "los resultandos" in a[0] and "considerandos" not in a[0],
   "…pero si ni el resultando lo menciona, eso sí se acusa (sólo el resultando)")
a = _d(_ad_sin_auto, dict(FICHA_AD, adhesivo={"quien": "Alejandro Daniel Betancourt Velarde", "admision": "2026-03-17"}),
       DATOS_AD, "amparo_directo", "k")
ok(len(a) == 1 and "considerandos" in a[0], "con el auto que lo admite, el considerando y el resolutivo se siguen pidiendo")

# ═══════════════════════════════════════════════════════════════════════════
# 19 · RONDA 5 (3-oct-2026): la segunda verificación (rev/verif_final2.txt) y
#      las decisiones F1-F6 de FIXES_R5
# ═══════════════════════════════════════════════════════════════════════════
print("\n19 · RONDA 5: SEGUNDA VERIFICACIÓN (LA VÍA EN HUECO, F1, F4, F5 Y EL PAR REPETIDO)")


def _det(texto, ficha, datos, tipo, regla, **kw):
    r = _seguro(vp.revisar_detalle, texto, ficha, datos, tipo, **kw)
    return [x for x in r if x.get("regla") == regla] if isinstance(r, list) else []


# ── LA VÍA EN HUECO NO ES EL NÚMERO (Q 335 sin fracción del 97, verif_q2 t6) ──
_VISTO_Q = "relativo al juicio de amparo indirecto 1201/2026; y,"
_INTERP_Q = "Juan Pérez López, quejoso en el juicio de amparo indirecto 1201/2026, interpuso"
_q_via = con(con(Q, _VISTO_Q, "relativo al juicio de amparo ********* 1201/2026; y,"),
             _INTERP_Q, "Juan Pérez López, quejoso en el juicio de amparo ********* 1201/2026, interpuso")
d = _det(_q_via, FICHA_Q, DATOS_Q, "queja", "a")
ok(len(d) == 1 and d[0].get("clave") == "fraccion_97" and "vía" in d[0].get("dato", "")
   and "V I S T O" in d[0]["aviso"] and "INTERPOSICIÓN" in d[0]["aviso"],
   "Q 335: «juicio de amparo ********* 1201/2026» en el V I S T O y la interposición: falta la VÍA (fraccion_97), "
   "un aviso con los dos apartados; no «falta el número (acto.expediente)» con el número a la vista"
   + (f" — {[(x.get('clave'), x.get('dato')) for x in d]}" if not (len(d) == 1 and d[0].get("clave") == "fraccion_97")
      else ""))
_AV_COMP_97 = ("FALTA LA FRACCIÓN DEL ARTÍCULO 97 (fraccion_97): sale del auto recurrido y de la vía —I si lo dictó un "
               "Juzgado de Distrito en amparo indirecto, II si lo dictó la responsable en amparo directo—. La vía del "
               "juicio («indirecto» o «directo») va en hueco en el V I S T O y en «Interposición del recurso de queja».")
d = _det(_q_via, dict(FICHA_Q, fraccion_97=""), DATOS_Q, "queja", "a", avisos_previos=[_AV_COMP_97])
ok(not d, "…y con el aviso del compositor «(fraccion_97)» como previo, la verja no lo repite (E11)"
   + (f" — {[x['aviso'][:90] for x in d]}" if d else ""))
d = _det(con(Q, _VISTO_Q, "relativo al juicio de amparo ********* TESTADO-EXPEDIENTE; y,"), FICHA_Q, DATOS_Q,
         "queja", "a")
ok(len(d) == 1 and d[0].get("clave") == "fraccion_97",
   "…igual con el número testado del banco («********* TESTADO-EXPEDIENTE»)")
_q_comp = con(Q, f"dictado por el {JUZ3}, localizado", f"dictado en un juicio de amparo *********, por el {JUZ3}, "
              f"localizado")
d = _det(con(_q_comp, _VISTO_Q, "relativo al juicio de amparo ********* 1201/2026; y,"), FICHA_Q, DATOS_Q, "queja", "a")
ok(len(d) == 1 and d[0].get("clave") == "fraccion_97" and "COMPETENCIA" in d[0]["aviso"],
   "F3: la competencia «dictado en un juicio de amparo *********, por…» (sin número) también es la vía, en el mismo "
   "aviso que el V I S T O" + (f" — {[(x.get('clave'), x['aviso'][:80]) for x in d]}" if len(d) != 1 else ""))
d = _det(con(Q, _VISTO_Q, "relativo al juicio de amparo ********* *********; y,"), FICHA_Q, DATOS_Q, "queja", "a")
ok(sorted(x.get("clave") for x in d) == ["acto.expediente", "fraccion_97"],
   "«juicio de amparo ********* *********»: la vía (fraccion_97) y el número (acto.expediente), cada uno con su "
   "clave" + (f" — {[x.get('clave') for x in d]}" if len(d) != 2 else ""))
d = _det(con(Q, _VISTO_Q, "relativo al juicio de amparo indirecto *********; y,"), FICHA_Q, DATOS_Q, "queja", "a")
ok(len(d) == 1 and d[0].get("clave") == "acto.expediente",
   "…y «juicio de amparo indirecto *********» sigue siendo el número, como antes")
d = _det(con(Q, _VISTO_Q, "relativo al juicio de amparo *********; y,"), FICHA_Q, DATOS_Q, "queja", "a")
ok(len(d) == 1 and d[0].get("clave") == "acto.expediente",
   "…y «juicio de amparo *********; y,» (nada detrás), el número, como antes")
d = _det(con(AR, "en el juicio de amparo indirecto 950/2025; y,", "en el juicio de amparo ********* 950/2025; y,"),
         FICHA_AR, DATOS_AR, "amparo_revision", "a")
ok(len(d) == 1 and d[0].get("clave") == "acto.via", "AR: la vía en hueco con el número detrás es acto.via")

# ── F4: «X, POR CONDUCTO DE SU X» CON LA ADSCRIPCIÓN ESCRITA DISTINTO (RF 7 con el nombre real, vf2_rf/t1) ──
_JEFA_DEL = ("la Jefa de la Unidad Jurídica de la Delegación Estatal del Instituto de Seguridad y Servicios Sociales de "
             "los Trabajadores del Estado en Baja California Sur")
_JEFA_REP = ("Jefa de la Unidad Jurídica de la Representación Estatal del Instituto de Seguridad y Servicios Sociales de "
             "los Trabajadores del Estado en Baja California Sur")
_SUBDEL = ("del Subdelegado de Pensiones, Seguridad e Higiene de la Delegación Estatal Baja California Sur del "
           "Instituto de Seguridad y Servicios Sociales de los Trabajadores del Estado")
_t_rf7 = con(RF, _rep, f"Inconforme, {_JEFA_DEL}, por conducto de su {_JEFA_REP}, María Luisa Gómez Ríos, en "
             f"representación {_SUBDEL},")
a = [x["aviso"] for x in _det(_t_rf7, FICHA_RF, DATOS_RF, "revision_fiscal", "d")]
ok(any("POR CONDUCTO DE SU PROPIO CARGO" in x for x in a),
   "RF 7: «la Jefa de la Unidad Jurídica de la Delegación Estatal…, por conducto de su Jefa de la Unidad Jurídica de "
   "la Representación Estatal…»: la misma unidad por su núcleo (tipos_asunto.misma_autoridad, F4)")
a = [x["aviso"] for x in _det(con(RF, _rep, f"Inconforme, {_JEFA_DEL}, en representación de la {_JEFA_REP},"),
                               FICHA_RF, DATOS_RF, "revision_fiscal", "d")]
ok(any("SÍ MISMO" in x for x in a), "…y «la Jefa… de la Delegación…, en representación de la Jefa… de la "
                                     "Representación…»: X en representación de X")
for _nombre, _txt in (
        ("la unidad jurídica por conducto de la Subdirección de lo Contencioso",
         f"Inconforme, {_JEFA_DEL}, por conducto de su Titular de la Subdirección de lo Contencioso, María Luisa Gómez "
         f"Ríos,"),
        ("la unidad jurídica en representación del Subdelegado (la forma del corpus)",
         f"Inconforme, {_JEFA_DEL}, en representación {_SUBDEL},"),
        ("el Titular de la Unidad Jurídica en representación del Titular de la Delegación",
         "Inconforme, el Titular de la Unidad Jurídica de la Delegación Estatal en Querétaro del Instituto de Seguridad "
         "y Servicios Sociales de los Trabajadores del Estado, en representación del Titular de la Delegación Estatal "
         "en Querétaro del Instituto de Seguridad y Servicios Sociales de los Trabajadores del Estado,")):
    a = [x["aviso"] for x in _det(con(RF, _rep, _txt), FICHA_RF, DATOS_RF, "revision_fiscal", "d")]
    ok(not any("PROPIO CARGO" in x or "SÍ MISMO" in x for x in a), f"{_nombre}: no se acusa"
       + (f" — {a[:1]}" if a else ""))

# ── F1: LA FORMA DE NOTIFICACIÓN SIN FUENTE QUE EL CONSIDERANDO AFIRMA (Q 335, AD 274 y 335) ──
_f_omi = dict(FICHA_AD, fuentes={"forma_notificacion": "omision"})
d = _det(AD, _f_omi, DATOS_AD, "amparo_directo", "m")
ok(len(d) == 1 and "NO CONSTA" in d[0]["aviso"] and "de manera personal" in d[0]["aviso"]
   and d[0].get("clave") == "forma_notificacion",
   "AD 274: la forma salió del valor por omisión del formulario (fuente «omision») y la oportunidad dice «de manera "
   "personal»: se acusa" + (f" — {[x['aviso'][:90] for x in d]}" if len(d) != 1 else ""))
_ad_sin_forma = con(AD, "dos mil veintiséis de manera personal y surtió efectos", "dos mil veintiséis y surtió efectos")
ok(not _det(_ad_sin_forma, _f_omi, DATOS_AD, "amparo_directo", "m"),
   "…con «se notificó a la quejosa el … y surtió efectos al día hábil siguiente» (F1, como los engroses), nada")
ok(not _det(AD, dict(FICHA_AD, fuentes={"forma_notificacion": "secretario"}), DATOS_AD, "amparo_directo", "m"),
   "…y la forma «personal» que DECLARÓ el secretario sigue sin cotejarse, como antes")
# LA CONTRACCIÓN (revisión AD, 3-oct-2026): decía «la ficha la tomó de el valor por omisión».
ok(len(d) == 1 and "la ficha la tomó del valor por omisión del formulario («personal»)" in d[0]["aviso"]
   and "tomó de el" not in d[0]["aviso"],
   "el aviso dice «la tomó del valor por omisión», no «de el»")
_q_ofi = con(Q, "El auto recurrido se le notificó el tres de junio", "El auto recurrido se le notificó por oficio el tres "
             "de junio")
_f_q_ofi = dict(FICHA_Q, forma_notificacion="oficio", fuentes={"forma_notificacion": "omision_autoridad"})
d = _det(_q_ofi, _f_q_ofi, DATOS_Q, "queja", "m")
ok(len(d) == 1 and "por oficio" in d[0]["aviso"] and "NO CONSTA" in d[0]["aviso"],
   "D2 sin papel (fuente «omision_autoridad»): «se le notificó por oficio» se acusa")
ok(not _det(_q_ofi, dict(_f_q_ofi, fuentes={"forma_notificacion": "auto"}), DATOS_Q, "queja", "m"),
   "…y el oficio que dice el auto (fuente «auto») cuadra: nada")

# ── F5: EL ARTÍCULO DEL PAPEL EN UN CARGO DE DOBLE GÉNERO (AD 128 del banco) ──
_OFICIAL = "Oficial Mayor y Coordinadora de Recursos Humanos del Municipio de Cadereyta de Montes, Querétaro"
_ter = "Tiene ese carácter Alejandro Daniel Betancourt Velarde."
_f_ofi = dict(FICHA_AD, terceros=[f"la {_OFICIAL}"])
d = _det(con(AD, _ter, f"Tiene ese carácter el {_OFICIAL}."), _f_ofi, DATOS_AD, "amparo_directo", "f")
ok(len(d) == 1 and "ARTÍCULO DEL CARGO" in d[0]["aviso"] and "TERCERO INTERESADO" in d[0]["aviso"]
   and d[0].get("clave") == "terceros",
   "AD 128: la ficha dice «la Oficial Mayor…» y el resultando «el Oficial Mayor…»: se acusa"
   + (f" — {[x['aviso'][:90] for x in d]}" if len(d) != 1 else ""))
ok(not _det(con(AD, _ter, f"Tiene ese carácter la {_OFICIAL}."), _f_ofi, DATOS_AD, "amparo_directo", "f"),
   "…con «la Oficial Mayor…», nada")
ok(not _det(con(AD, _ter, f"Tiene ese carácter el {_OFICIAL}."), dict(FICHA_AD, terceros=[_OFICIAL]), DATOS_AD,
            "amparo_directo", "f"),
   "…sin artículo en el papel no hay contra qué cotejar (es el aviso «EL GÉNERO DEL CARGO NO CONSTA» del compositor)")
_TIT = ("Titular de la Unidad Jurídica de la Oficina de Representación del Instituto de Seguridad y Servicios Sociales "
        "de los Trabajadores del Estado en Querétaro")
_rf_tit = con(RF, _rep, f"Inconforme, el {_TIT}, en representación de la autoridad demandada,")
d = _det(_rf_tit, dict(FICHA_RF, promovente=f"la {_TIT}"), DATOS_RF, "revision_fiscal", "f")
ok(len(d) == 1 and d[0].get("clave") == "promovente",
   "RF 49: la ficha dice «la Titular…» y el resultando «el Titular…»: se acusa")
ok(not _det(_rf_tit, dict(FICHA_RF, promovente=f"la {_TIT}", autoridad_demandada=f"el {_TIT}"), DATOS_RF,
            "revision_fiscal", "f"),
   "…pero si el papel trae el mismo cargo con los dos artículos (dos personas), no se sabe cuál es cuál: nada")

# ── EL PAR DE PALABRAS REPETIDO (RF 4 del banco, negativa ficta) ──
_rf_sol = con(RF, "demandó la nulidad de la resolución contenida en el oficio",
              "demandó la nulidad de la resolución negativa ficta recaída a su solicitud de solicitud de incorporación "
              "al sistema de jubilación, contenida en el oficio")
d = _det(_rf_sol, FICHA_RF, DATOS_RF, "revision_fiscal", "n")
ok(len(d) == 1 and "solicitud de solicitud de" in d[0]["aviso"] and "TRÁMITE DEL JUICIO" in d[0]["aviso"],
   "RF 4: «recaída a su solicitud de solicitud de incorporación…» se acusa con su apartado"
   + (f" — {[x['aviso'][:90] for x in d]}" if len(d) != 1 else ""))
ok(len(_det(con(Q, "en el que negó la suspensión provisional.", "en el que negó la suspensión de la de la "
                "suspensión provisional."), FICHA_Q, DATOS_Q, "queja", "n")) == 1, "«de la de la» se acusa")
for _frase in ("Juan Pérez Pérez", "aunado a que que", "caso por caso", "día a día"):
    ok(not _det(con(Q, "en el que negó la suspensión provisional.", f"en el que negó la suspensión provisional, "
                    f"{_frase}."), FICHA_Q, DATOS_Q, "queja", "n"),
       f"«{_frase}» no se acusa (apellido, palabra suelta de dedo, o no es el mismo par seguido)")

# ═══════════════════════════════════════════════════════════════════════════
# 20 · SEXTA RONDA (3-oct-2026): las respuestas de David (contrato C1-C8)
# ═══════════════════════════════════════════════════════════════════════════
print("\n20 · SEXTA RONDA: LAS FORMAS NUEVAS (C1-C7) SIN AVISOS FALSOS Y LOS ASUNTOS RELACIONADOS (o, p, q)")
import tipos_asunto as _ta6

_HN_QRO = _ta6.supletorio(TRIB, "Querétaro")["hecho_notorio"]
_HN_CDMX = _ta6.supletorio(TRIB_CDMX, "Ciudad de México")["hecho_notorio"]


def _todas(texto, ficha, datos, tipo):
    return _seguro(vp.revisar_detalle, texto, ficha, datos, tipo) or []


def _rel(tipo, numero, estado="misma_sesion"):
    return {"tipo": tipo, "numero": numero, "estado": estado}


# ── C1: EL AMPARO EN REVISIÓN SIN EXISTENCIA, CON «Procedencia.» Y EL PUNTO QUE NOMBRA ACTO Y AUTORIDAD (C5) ──
_EXIST_AR = next(l for l in AR.split("\n") if l.startswith("SEGUNDO. Existencia de la resolución recurrida."))
_SEG_AR = ("SEGUNDO. La Justicia de la Unión no ampara ni protege a {COM}, contra el acto que reclamó del Director de "
           "Ingresos del Municipio de Querétaro, precisado en el resultando primero de esta ejecutoria.").replace(
    "{COM}", COM)
_PUNTO_C5 = _ta6.con_actos_del_amparo(
    _ta6.RAMAS_REVISION["confirma_niega"]["puntos"][1], FICHA_AR["demanda"]["actos"],
    FICHA_AR["demanda"]["autoridades"], "primero", False, rama="confirma_niega").replace("{quejoso}", COM)
AR_C1 = con(con(con(con(AR, _EXIST_AR, f"SEGUNDO. Procedencia. {_ta6.procedencia_revision('sentencia')}"),
                    "PRIMERO. En la materia de la revisión, se confirma la sentencia recurrida.",
                    "PRIMERO. Se confirma la sentencia recurrida."),
                _SEG_AR, _PUNTO_C5),
            "CUARTO. Sentencia recurrida y agravios.", "CUARTO. Resolución recurrida y agravios.")
ok("respecto del acto que reclamó del Director de Ingresos del Municipio de Querétaro, consistente en la "
   "determinación del crédito fiscal" in AR_C1 and "Existencia" not in AR_C1,
   "(el fixture) el AR de C1 no lleva existencia y su punto nombra acto y autoridad, con `tipos_asunto`")
for _n, _f in (("con ficha", FICHA_AR), ("sin ficha", None)):
    a = _todas(AR_C1, _f, DATOS_AR, "amparo_revision")
    ok(not a, f"C1/C5: el AR sin existencia, con «Procedencia.» antes de la legitimación y el punto «respecto del acto "
              f"que reclamó de…», {_n}: cero avisos" + (f" — {[x['aviso'][:110] for x in a[:2]]}" if a else ""))
# El punto del sobreseimiento con el acto y la autoridad del amparo, y lo recurrido es un AUTO (81-I-d): ni la Sala
# del amparo es el órgano recurrido (d), ni «la sentencia dictada el…» del acto reclamado es la clase de lo recurrido (l).
_SALA2 = "Segunda Sala Civil del Tribunal Superior de Justicia del Estado de Querétaro"
_ACTO_S = f"La sentencia dictada el diez de mayo de dos mil veinticinco, por la {_SALA2}"
_punto_s = _ta6.con_actos_del_amparo(_ta6.RAMAS_REVISION["confirma_sobresee"]["puntos"][1], [_ACTO_S], [_SALA2],
                                     "primero", False, rama="confirma_sobresee").replace("{quejoso}", COM)
_ar_s = con(con(AR_C1, _PUNTO_C5, _punto_s), "Director de Ingresos del Municipio de Querétaro.\nACTOS RECLAMADOS:\n"
            "La determinación del crédito fiscal contenida en el oficio DI/123/2025, que le concedió un plazo para pagar.",
            f"{_SALA2}.\nACTOS RECLAMADOS:\n{_ACTO_S}.")
_f_s = dict(FICHA_AR, demanda=dict(FICHA_AR["demanda"], autoridades=[_SALA2], actos=[_ACTO_S]))
ok("contra el acto que reclamó de la Segunda Sala Civil" in _ar_s and "consistente en la sentencia dictada el diez" in _ar_s,
   "(el fixture) el punto del sobreseimiento con «contra» y el acto con su Sala")
for _n, _f in (("con ficha", _f_s), ("sin ficha", None)):
    d = [x for x in _todas(_ar_s, _f, DATOS_AR, "amparo_revision") if x["regla"] in "dc"]
    ok(not d, f"C5: «…contra el acto que reclamó de la Sala…, consistente en la sentencia dictada el … por la Sala…» "
              f"en el resolutivo no es el órgano recurrido (d) ni una fecha del acto ajena (c), {_n}"
       + (f" — {[x['aviso'][:120] for x in d[:2]]}" if d else ""))
# Y si lo recurrido es el AUTO que sobreseyó fuera de la audiencia (81-I-d), «la sentencia dictada el…» del acto
# reclamado no es la clase de lo recurrido (l).
_ar_auto = (f"V I S T O, para resolver el recurso de revisión 410/2026, interpuesto por {COM}, contra el auto de diez "
            f"de abril de dos mil veintiséis, dictado por el {JUZ}, en el juicio de amparo indirecto 950/2025; y,\n"
            f"R E S U L T A N D O\nC O N S I D E R A N D O\n"
            f"R E S U E L V E\nPRIMERO. Se confirma el auto recurrido.\n{_punto_s}")
_f_auto = dict(_f_s, clase_recurrida="auto_sobreseimiento", acto=dict(FICHA_AR["acto"], clase="auto", resolvio="sobresee"))
d = [x for x in _todas(_ar_auto, _f_auto, DATOS_AR, "amparo_revision") if x["regla"] in "dlc"]
ok(not d, "C5: con lo recurrido un AUTO de sobreseimiento, el acto reclamado «la sentencia dictada el…» del punto no "
          "es la clase de lo recurrido (l), ni el órgano (d), ni la fecha del acto (c)"
   + (f" — {[x['aviso'][:120] for x in d[:2]]}" if d else ""))
ok(any(x["regla"] == "l" for x in _todas(_ar_auto.replace("PRIMERO. Se confirma el auto recurrido.",
                                                          "PRIMERO. Se confirma la sentencia recurrida."),
                                         _f_auto, DATOS_AR, "amparo_revision")),
   "…pero «se confirma la sentencia recurrida» con lo recurrido un auto se sigue acusando (l)")
# El marcador que nadie sustituyó.
_ar_m = con(AR_C1, _PUNTO_C5, _ta6.RAMAS_REVISION["confirma_niega"]["puntos"][1].replace("{quejoso}", COM))
d = _det(_ar_m, FICHA_AR, DATOS_AR, "amparo_revision", "a")
ok(len(d) == 1 and "MARCADOR DE PLANTILLA SIN SUSTITUIR" in d[0]["aviso"] and "{actos_del_amparo}" in d[0]["aviso"]
   and "LOS RESOLUTIVOS" in d[0]["aviso"],
   "C5: «{actos_del_amparo}» que llega sin sustituir al resolutivo se acusa (a), con su apartado"
   + (f" — {[x['aviso'][:120] for x in d]}" if len(d) != 1 else ""))
ok(not any("MARCADOR" in x["aviso"] for x in _todas(AR, FICHA_AR, DATOS_AR, "amparo_revision")),
   "…y sin llaves, nada")
# El órgano descartado en el AR de C1: el aviso ya no manda a la existencia.
_ar_desc6 = con(con(con(AR_C1, f"por el {JUZ}, en el juicio de amparo indirecto 950/2025; y,",
                        "por *********, en el juicio de amparo indirecto 950/2025; y,"),
                    f"correspondió al {JUZ}, que", "correspondió a *********, que"),
                f"en materia administrativa, por el {JUZ}, localizado",
                f"en materia administrativa, por la {_SALA_FAM}, localizada")
a = _d(_ar_desc6, _f_desc, dict(DATOS_AR, organo_recurrido=_SALA_FAM, procesal={"juzgado": ""}), "amparo_revision", "d")
ok(len(a) == 1 and "SE DESCARTÓ" in a[0] and "existencia" not in a[0].split("O. Tiene que ir")[-1].lower(),
   "C1: el órgano descartado en el AR sin existencia: el aviso dice dónde va el juzgado sin nombrar la existencia"
   + (f" — {a[:1]}" if len(a) != 1 else ""))

# ── C3: LA QUEJA SIN EL RENGLÓN DEL ÓRGANO Y CON «Demanda de amparo.» PRIMERO ──
_JUEZ_CIV = "Juez Segundo de lo Civil del Distrito Judicial de Querétaro"
_ACTO_Q = f"La orden de embargo dictada el quince de mayo de dos mil veintiséis, por el {_JUEZ_CIV}"
Q_C3 = (con(con(con(con(con(Q, f"RECURRENTE: JUAN PÉREZ LÓPEZ.\nÓRGANO QUE DICTÓ EL AUTO RECURRIDO: {JUZ3.upper()}.\n",
                            "QUEJOSO Y RECURRENTE: JUAN PÉREZ LÓPEZ.\n"),
                        "PRIMERO. Interposición del recurso de queja.",
                        f"PRIMERO. Demanda de amparo. Por escrito presentado el veinte de mayo de dos mil veintiséis, "
                        f"Juan Pérez López promovió juicio de amparo indirecto en contra de la autoridad y el acto que a "
                        f"continuación se señalan:\nAUTORIDAD RESPONSABLE:\n{_JUEZ_CIV}\nACTO RECLAMADO:\n{_ACTO_Q}.\n"
                        f"SEGUNDO. Interposición del recurso de queja."),
                    "SEGUNDO. Trámite del recurso.", "TERCERO. Trámite del recurso."),
                "TERCERO. Turno del asunto.", "CUARTO. Turno del asunto."),
            "CUARTO. Verificación de la sesión", "QUINTO. Verificación de la sesión"))
FICHA_Q_C3 = dict(FICHA_Q, quejoso="Juan Pérez López",
                  demanda={"fecha": "2026-05-20", "autoridades": [_JUEZ_CIV], "actos": [_ACTO_Q]})
for _n, _f in (("con ficha", FICHA_Q_C3), ("sin ficha", None)):
    a = _todas(Q_C3, _f, DATOS_Q, "queja")
    ok(not a, f"C3: la queja sin el renglón del órgano y con «Demanda de amparo.» primero (su acto «dictada… por el "
              f"Juez…»), {_n}: cero avisos" + (f" — {[x['aviso'][:110] for x in a[:2]]}" if a else ""))
_q_dem = con(Q_C3, "presentado el veinte de mayo de dos mil veintiséis", "presentado el diez de junio de dos mil veintiséis")
d = _det(_q_dem, FICHA_Q_C3, DATOS_Q, "queja", "j")
ok(len(d) == 1 and "DEMANDA DE AMPARO" in d[0]["aviso"].upper() and "posterior al auto recurrido" in d[0]["aviso"],
   "C3: la demanda posterior al auto recurrido que se dictó en ese juicio se acusa (j)"
   + (f" — {[x['aviso'][:120] for x in d]}" if len(d) != 1 else ""))
ok(not _det(_q_dem, FICHA_Q_C3, DATOS_Q, "queja", "j",
            avisos_previos=["FECHA IMPOSIBLE: la demanda de amparo es del 10/06/2026 y el auto recurrido del 02/06/2026."]),
   "…y si otra pieza ya lo dijo con esas dos fechas, no se repite (E11)")
_f_dem = dict(FICHA_Q_C3, demanda=dict(FICHA_Q_C3["demanda"], fecha="2026-06-10"))
_val = vp._validar_copia(_f_dem)[0]
ok(not _det(_q_dem, _f_dem, DATOS_Q, "queja", "j") if any("imposible" in vp._plano(x) for x in _val) else True,
   "…ni si ya lo dice `ficha_tramite.validar` con la fecha de la ficha (demanda.fecha > acto.fecha)"
   + ("" if any("imposible" in vp._plano(x) for x in _val) else " (validar no lo dice: no aplica)"))
d = _det(con(Q_C3, "presentado el veinte de mayo de dos mil veintiséis", "presentado el *********"),
         dict(FICHA_Q_C3, demanda=dict(FICHA_Q_C3["demanda"], fecha="")), DATOS_Q, "queja", "a")
ok(len(d) == 1 and d[0].get("clave") == "demanda.fecha",
   "C3: si la fecha de la demanda de la queja llegara en hueco, su clave es demanda.fecha (no la del recurso)"
   + (f" — {[x.get('clave') for x in d]}" if len(d) != 1 else ""))

# ── C4: LA CARÁTULA DE LA REVISIÓN FISCAL («RECURRENTE:», SIN LA SALA) ──
RF_C4 = con(RF, f"AUTORIDAD RECURRENTE: {UJ.upper()}.\nPARTE ACTORA: MARÍA GÓMEZ RUIZ.\nSALA RESPONSABLE: "
                f"{SALA_RF.upper()} DEL TRIBUNAL FEDERAL DE JUSTICIA ADMINISTRATIVA.\n",
            f"RECURRENTE: {UJ.upper()}.\nPARTE ACTORA: MARÍA GÓMEZ RUIZ.\n")
for _n, _f in (("con ficha", FICHA_RF), ("sin ficha", None)):
    a = _todas(RF_C4, _f, DATOS_RF, "revision_fiscal")
    ok(not a, f"C4: la revisión fiscal con «RECURRENTE:» y sin «SALA RESPONSABLE:», {_n}: cero avisos"
       + (f" — {[x['aviso'][:110] for x in a[:2]]}" if a else ""))
d = _det(con(RF_C4, "REVISIÓN FISCAL: 42/2026.", "REVISIÓN FISCAL: *********."), FICHA_RF, DATOS_RF,
         "revision_fiscal", "e")
ok(any("NO TRAE EL NÚMERO" in x["aviso"] for x in d),
   "C4: la carátula sin el número sigue acusándose con el renglón «RECURRENTE:» debajo")
d = _det(con(RF_C4, "RECURRENTE: ", "RECURRENTE: *********.\nRECURRENTE ADHESIVO: "), FICHA_RF, DATOS_RF,
         "revision_fiscal", "a")
ok(any(x.get("clave") == "promovente" for x in d), "el hueco de «RECURRENTE:» es el promovente")
d = _det(con(RF_C4, "PARTE ACTORA: ", "RECURRENTE ADHESIVO: *********.\nPARTE ACTORA: "), FICHA_RF, DATOS_RF,
         "revision_fiscal", "a")
ok(any(x.get("clave") == "adhesivo.quien" for x in d) and not any(x.get("clave") == "promovente" for x in d),
   "…y el de «RECURRENTE ADHESIVO:», quien se adhirió (adhesivo.quien), no quien recurre")

# ── C6: LOS ASUNTOS RELACIONADOS BIEN COMPUESTOS, CON LOS TEXTOS DE `tipos_asunto` ──
def _con_relacionados(texto, encabezado, tras_numero, lista, tipo, numero, materia, hn, tras_considerando, ficha):
    """El rubro, el V I S T O y el considerando, como los pone documento_generado (C6)."""
    rot, txt = _ta6.considerando_relacionados(tipo, numero, materia, lista, hn)
    t = con(texto, encabezado, encabezado + "\n" + _ta6.rotulo_relacionados(lista, materia))
    t = con(t, tras_numero, tras_numero + ", relacionado con " + _ta6.relacionados_en_prosa(lista, materia))
    # El considerando entra tras la legitimación (o la procedencia) y el estudio se corre solo.
    lin = t.split("\n")
    k = next(i for i, l in enumerate(lin) if l.startswith(tras_considerando))
    ords = ["PRIMERO", "SEGUNDO", "TERCERO", "CUARTO", "QUINTO", "SEXTO", "SÉPTIMO", "OCTAVO"]
    j = k + 1
    while j < len(lin) and not re.match(r"^(?:" + "|".join(ords) + r")\. ", lin[j]):
        j += 1
    n_ins = ords.index(lin[k].split(".")[0]) + 1
    resto = []
    en_resolutivos = False
    for l in lin[j:]:
        if l.replace(" ", "") == "RESUELVE":
            en_resolutivos = True
        m = re.match(r"^(" + "|".join(ords) + r")\. ", l)
        if m and not en_resolutivos:
            l = ords[ords.index(m.group(1)) + 1] + l[len(m.group(1)):]
        resto.append(l)
    parrafos = txt.split("\n")
    nuevo = lin[:j] + [f"{ords[n_ins]}. {rot} {parrafos[0]}"] + parrafos[1:] + resto
    return "\n".join(nuevo), dict(ficha, relacionados=list(lista), fuentes=dict(ficha.get("fuentes") or {},
                                                                                  relacionados="secretario"))


_L_AD = [_rel("amparo_directo", "175/2026")]
AD_REL, FICHA_AD_REL = _con_relacionados(AD, "AMPARO DIRECTO CIVIL: 174/2026.", "juicio de amparo directo civil 174/2026",
                                         _L_AD, "amparo_directo", "174/2026", "civil", _HN_QRO,
                                         "TERCERO. Legitimación", FICHA_AD)
ok("RELACIONADO CON EL AMPARO DIRECTO CIVIL 175/2026." in AD_REL and "CUARTO. Conexidad. Con vista" in AD_REL
   and "QUINTO. Sentencia reclamada" in AD_REL and "relacionado con el amparo directo civil 175/2026, promovido" in AD_REL,
   "(el fixture) el AD con su relacionado: el rubro, el V I S T O y «CUARTO. Conexidad.», con el estudio corrido")
_L_Q = [_rel("amparo_revision", "298/2025", "resuelto")]
Q_REL, FICHA_Q_REL = _con_relacionados(Q_C3, "RECURSO DE QUEJA CIVIL: 355/2026.", "recurso de queja civil 355/2026",
                                       _L_Q, "queja", "355/2026", "civil", _HN_QRO, "TERCERO. Legitimación",
                                       FICHA_Q_C3)
_L_AR = [_rel("amparo_directo", "175/2026"), _rel("queja", "24/2026", "resuelto")]
AR_REL, FICHA_AR_REL = _con_relacionados(AR_C1, "AMPARO EN REVISIÓN ADMINISTRATIVA: 410/2026.",
                                         "recurso de revisión 410/2026", _L_AR, "amparo_revision", "410/2026",
                                         "administrativa", _HN_QRO, "TERCERO. Legitimación", FICHA_AR)
_L_RF = [_rel("amparo_directo", "469/2024")]
RF_REL, FICHA_RF_REL = _con_relacionados(RF_C4, "REVISIÓN FISCAL: 42/2026.", "recurso de revisión fiscal número 42/2026",
                                         _L_RF, "revision_fiscal", "42/2026", "administrativa", _HN_QRO,
                                         "TERCERO. Procedencia", FICHA_RF)
_L_CDMX = [_rel("amparo_directo", "175/2026", "resuelto")]
AD_CDMX_REL, FICHA_CDMX_REL = _con_relacionados(AD_CDMX, "AMPARO DIRECTO CIVIL: 174/2026.",
                                                "juicio de amparo directo civil 174/2026", _L_CDMX, "amparo_directo",
                                                "174/2026", "civil", _HN_CDMX, "TERCERO. Legitimación", FICHA_CDMX)
ok("Asuntos relacionados." in AR_REL and "Hecho notorio." in Q_REL and "artículo 64 de la Ley Federal" in RF_REL
   and "269 del Código Nacional" in AD_CDMX_REL and "88 del Código Federal" in Q_REL,
   "(los fixtures) «Asuntos relacionados.» (AR), «Hecho notorio.» (Q, CFPC 88), la RF con el 64 y el AD de la Ciudad "
   "de México con el CNPCF 269")
for _n, _t, _f, _dd, _tp in (("AD civil con otro AD en la misma sesión (Conexidad)", AD_REL, FICHA_AD_REL, DATOS_AD,
                              "amparo_directo"),
                             ("queja con el AR ya resuelto (Hecho notorio, CFPC 88)", Q_REL, FICHA_Q_REL, DATOS_Q, "queja"),
                             ("AR con un AD en la misma sesión y una queja resuelta (Asuntos relacionados)", AR_REL,
                              FICHA_AR_REL, DATOS_AR, "amparo_revision"),
                             ("RF con el AD de la misma sentencia (art. 64 LFPCA)", RF_REL, FICHA_RF_REL, DATOS_RF,
                              "revision_fiscal"),
                             ("AD de la Ciudad de México con un AD resuelto (CNPCF 269)", AD_CDMX_REL, FICHA_CDMX_REL,
                              DATOS_CDMX, "amparo_directo")):
    a = _todas(_t, _f, _dd, _tp)
    ok(not a, f"C6: {_n}: cero avisos" + (f" — {[x['aviso'][:120] for x in a[:2]]}" if a else ""))
    a = _todas(_t, None, _dd, _tp)
    ok(not a, f"…sin ficha, tampoco" + (f" — {[x['aviso'][:120] for x in a[:2]]}" if a else ""))
# Con la lista en `procesal` (lo que expone el compositor) y la ficha sin la clave, igual.
_f_sin = {k: v for k, v in FICHA_AD_REL.items() if k != "relacionados"}
ok(not _todas(AD_REL, _f_sin, dict(DATOS_AD, procesal={"relacionados": _L_AD}), "amparo_directo"),
   "C6: la lista que expone el compositor (procesal.relacionados) vale como la de la ficha")

# ── (e) EL RENGLÓN «RELACIONADO CON …» NO ES EL NÚMERO DEL ASUNTO ──
_ad_sin_num = con(AD_REL, "AMPARO DIRECTO CIVIL: 174/2026.", "AMPARO DIRECTO CIVIL: *********.")
d = _det(_ad_sin_num, FICHA_AD_REL, DATOS_AD, "amparo_directo", "e")
ok(any("NO TRAE EL NÚMERO" in x["aviso"] for x in d) and not any("175/2026" in x["aviso"] for x in d),
   "(e) el encabezado sin número y debajo «RELACIONADO CON EL AMPARO DIRECTO CIVIL 175/2026.»: «la carátula no trae "
   "el número», no «la carátula dice 175/2026»" + (f" — {[x['aviso'][:120] for x in d]}" if d else ""))
_ad_visto = con(AD_REL, "juicio de amparo directo civil 174/2026, relacionado",
                "juicio de amparo directo civil *********, relacionado")
d = [x for x in _det(_ad_visto, FICHA_AD_REL, DATOS_AD, "amparo_directo", "e") if "NO CUADRA" in x["aviso"]]
ok(not d, "(e) el V I S T O con el número en hueco y «relacionado con el amparo directo civil 175/2026»: el 175 no es "
          "el número del asunto" + (f" — {[x['aviso'][:120] for x in d]}" if d else ""))
ok(any("NO CUADRA" in x["aviso"] and "175/2026" in x["aviso"]
       for x in _det(con(AD_REL, "juicio de amparo directo civil 174/2026, relacionado",
                         "juicio de amparo directo civil 175/2026, relacionado"), FICHA_AD_REL, DATOS_AD,
                     "amparo_directo", "e")),
   "…pero el V I S T O con otro número EN EL LUGAR del asunto sí se acusa, como siempre")

# ── (o) EL PROPIO ASUNTO COMO RELACIONADO ──
_ad_propio = con(con(AD, "AMPARO DIRECTO CIVIL: 174/2026.",
                     "AMPARO DIRECTO CIVIL: 174/2026.\nRELACIONADO CON EL AMPARO DIRECTO CIVIL 174/2026."),
                 "juicio de amparo directo civil 174/2026, promovido",
                 "juicio de amparo directo civil 174/2026, relacionado con el amparo directo civil 174/2026, promovido")
d = _det(_ad_propio, None, DATOS_AD, "amparo_directo", "o")
ok(len(d) == 1 and "CONSIGO MISMO" in d[0]["aviso"] and "la carátula" in d[0]["aviso"] and "V I S T O" in d[0]["aviso"]
   and d[0].get("clave") == "relacionados",
   "(o) «relacionado con el amparo directo civil 174/2026» en el propio AD 174/2026: un aviso con los dos sitios"
   + (f" — {[x['aviso'][:120] for x in d]}" if len(d) != 1 else ""))
ok(not _det(con(_ad_propio, "CIVIL: 174/2026.\nRELACIONADO CON EL AMPARO DIRECTO CIVIL 174/2026",
                "CIVIL: 174/2026.\nRELACIONADO CON EL RECURSO DE REVISIÓN FISCAL 174/2026")
            .replace("relacionado con el amparo directo civil 174/2026", "relacionado con el recurso de revisión fiscal "
                     "174/2026"), None, DATOS_AD, "amparo_directo", "o"),
   "…la revisión fiscal 174/2026 es OTRO expediente (mismo número, otro tipo): no se acusa")
_cons_propio = con(AD_REL, "con el amparo directo civil 175/2026, ambos", "con el amparo directo civil 174/2026, ambos")
d = _det(_cons_propio, None, DATOS_AD, "amparo_directo", "o")
ok(len(d) == 1 and "Conexidad" in d[0]["aviso"],
   "(o) el considerando que relaciona el AD 174/2026 consigo mismo («el presente juicio… 174/2026 con el amparo "
   "directo civil 174/2026»): se acusa; «el presente» no cuenta" + (f" — {[x['aviso'][:120] for x in d]}" if len(d) != 1 else ""))

# ── (p) EL ARTÍCULO 64 DE LA LFPCA SÓLO PARA EL PAR AMPARO DIRECTO ↔ REVISIÓN FISCAL ──
_TXT_64 = ("Con vista en la conexión que guarda el presente {este} con el {otro}, ambos del índice de este Tribunal "
           "Colegiado de Circuito, se resuelven en la misma sesión, con fundamento en el artículo 64 de la Ley Federal "
           "de Procedimiento Contencioso Administrativo, porque en ambos se impugna la misma sentencia.")
_ar_64 = con(AR_REL, next(l for l in AR_REL.split("\n") if l.startswith("CUARTO. Asuntos relacionados.")),
             "CUARTO. Conexidad. " + _TXT_64.format(este="amparo en revisión administrativa 410/2026",
                                                    otro="recurso de revisión fiscal 33/2024"))
d = _det(_ar_64, None, DATOS_AR, "amparo_revision", "p")
ok(len(d) == 1 and "ARTÍCULO 64" in d[0]["aviso"] and "amparo en revisión" in d[0]["aviso"],
   "(p) el AR que funda la misma sesión con una RF en el 64 de la LFPCA: se acusa"
   + (f" — {[x['aviso'][:140] for x in d]}" if len(d) != 1 else ""))
_ad_64 = con(AD_REL, "a fin de evitar el dictado de resoluciones contradictorias.",
             "con fundamento en el artículo 64 de la Ley Federal de Procedimiento Contencioso Administrativo.")
d = _det(_ad_64, None, DATOS_AD, "amparo_directo", "p")
ok(len(d) == 1 and "175/2026" in d[0]["aviso"],
   "(p) el AD que funda en el 64 la misma sesión con OTRO amparo directo: se acusa, con el relacionado"
   + (f" — {[x['aviso'][:140] for x in d]}" if len(d) != 1 else ""))
ok(not _det(RF_REL, None, DATOS_RF, "revision_fiscal", "p"), "…la RF con el AD (el par del 64): nada")
_rot64, _txt64 = _ta6.considerando_relacionados("amparo_directo", "174/2026", "administrativa",
                                                [_rel("revision_fiscal", "33/2024"), _rel("amparo_revision", "12/2026")],
                                                _HN_QRO)
_ad_64b = con(AD_REL, next(l for l in AD_REL.split("\n") if l.startswith("CUARTO. Conexidad.")),
              f"CUARTO. {_rot64} {_txt64}")
ok("Asimismo" in _txt64 and not _det(_ad_64b, None, DATOS_AD, "amparo_directo", "p"),
   "…el AD con la RF (64) y un AR en otra oración («Asimismo…», la fórmula general): nada")
ok(not _det(con(RF, "63, párrafo primero, de la Ley Federal", "63 y 64 de la Ley Federal"), None, DATOS_RF,
            "revision_fiscal", "p"),
   "…el 64 citado fuera de una conexión (la competencia de la RF): no es esta regla")

# ── (q) EL RUBRO, EL V I S T O Y EL CONSIDERANDO, ENTRE SÍ Y CONTRA LA FICHA ──
_cons_ad = next(l for l in AD_REL.split("\n") if l.startswith("CUARTO. Conexidad."))
_sin_cons = AD_REL.replace(_cons_ad + "\n", "")
d = _det(_sin_cons, FICHA_AD_REL, DATOS_AD, "amparo_directo", "q")
ok(len(d) == 1 and "NO HAY CONSIDERANDO" in d[0]["aviso"] and "V I S T O" in d[0]["aviso"]
   and "CARÁTULA" in d[0]["aviso"],
   "(q) el V I S T O y el rubro dicen «relacionado con» y no hay considerando: se acusa"
   + (f" — {[x['aviso'][:120] for x in d]}" if len(d) != 1 else ""))
_sin_anuncio = (AD_REL.replace("\nRELACIONADO CON EL AMPARO DIRECTO CIVIL 175/2026.", "")
                .replace(", relacionado con el amparo directo civil 175/2026", ""))
d = _det(_sin_anuncio, FICHA_AD_REL, DATOS_AD, "amparo_directo", "q")
ok(len(d) == 1 and "NI EL V I S T O NI LA CARÁTULA" in d[0]["aviso"],
   "(q) …al revés: el considerando sin el V I S T O ni el rubro" + (f" — {[x['aviso'][:120] for x in d]}" if len(d) != 1 else ""))
_otro_num = AD_REL.replace(", relacionado con el amparo directo civil 175/2026", ", relacionado con el amparo directo "
                                                                                  "civil 176/2026")
d = _det(_otro_num, FICHA_AD_REL, DATOS_AD, "amparo_directo", "q")
ok(len(d) == 1 and "NO CUADRAN" in d[0]["aviso"] and "176/2026" in d[0]["aviso"],
   "(q) el V I S T O nombra otro número que el rubro y el considerando: «no cuadran»"
   + (f" — {[x['aviso'][:120] for x in d]}" if len(d) != 1 else ""))
d = _det(AD_REL, dict(FICHA_AD_REL, relacionados=[]), DATOS_AD, "amparo_directo", "q")
ok(len(d) == 1 and "NADIE MARCÓ" in d[0]["aviso"] and "automático" in d[0]["aviso"],
   "(q) la conexidad con la lista de la ficha VACÍA: «asuntos relacionados que nadie marcó» (David: nunca en automático)"
   + (f" — {[x['aviso'][:120] for x in d]}" if len(d) != 1 else ""))
d = _det(AD, FICHA_AD_REL, DATOS_AD, "amparo_directo", "q")
ok(len(d) == 1 and "NO LOS TRAE" in d[0]["aviso"] and "175/2026" in d[0]["aviso"],
   "(q) el secretario los marcó y el documento no los trae: se acusa (lo perdió el camino)"
   + (f" — {[x['aviso'][:120] for x in d]}" if len(d) != 1 else ""))
d = _det(AD_REL, dict(FICHA_AD_REL, relacionados=[_rel("amparo_directo", "180/2026")]), DATOS_AD, "amparo_directo", "q")
ok(len(d) == 1 and "NO SON LOS QUE MARCÓ" in d[0]["aviso"],
   "(q) los del documento no son los de la ficha: se acusa" + (f" — {[x['aviso'][:120] for x in d]}" if len(d) != 1 else ""))
ok(not _det(AD, dict(FICHA_AD, relacionados=[]), DATOS_AD, "amparo_directo", "q"),
   "(q) sin relacionados en la ficha ni en el texto: nada")
for _t in (_sin_cons, _sin_anuncio, _otro_num):
    ok(not _det(_t, None, DATOS_AD, "amparo_directo", "q"),
       "(q) sin la ficha nueva no se coteja: el tribunal pone el rubro sin considerando (Q 24/2026) y el considerando "
       "sin el V I S T O (AD 469/2024)")
# El AD 469/2024 del banco, como lo escribió el tribunal: el rubro en la misma línea, el V I S T O sin la mención y
# «CUARTO. Conexidad.» con el 64 (el par AD ↔ RF). Ninguna regla nueva lo acusa.
_ad469 = (AD.replace("AMPARO DIRECTO CIVIL: 174/2026.", "AMPARO DIRECTO ADMINISTRATIVO 174/2026 (RELACIONADO CON EL RECURSO "
                     "DE REVISIÓN FISCAL 33/2024).")
          .replace("CUARTO. Sentencia reclamada y conceptos de violación.",
                   "CUARTO. Conexidad. Con vista en la conexión que guardan el presente juicio de amparo directo "
                   "administrativo 174/2026, con el recurso de revisión fiscal 33/2024, ambos del índice de este Tribunal "
                   "Colegiado de Circuito, con fundamento en el artículo 64 de la Ley Federal de Procedimiento Contencioso "
                   "Administrativo, se resuelven en una misma sesión.\nQUINTO. Sentencia reclamada y conceptos de "
                   "violación.").replace("QUINTO. Estudio.", "SEXTO. Estudio."))
d = [x for x in _todas(_ad469, None, DATOS_AD, "amparo_directo") if x["regla"] in "eopq"]
ok(not d, "el AD 469/2024 como lo escribió el tribunal (rubro con paréntesis, V I S T O sin la mención, Conexidad con "
          "el 64): nada en (e), (o), (p) ni (q)" + (f" — {[x['aviso'][:120] for x in d]}" if d else ""))

# ── C7: EL CÓDIGO NACIONAL EN LO AGRARIO ES SUPLETORIO DE LA LEY AGRARIA, NO DE LA LEY DE AMPARO (g) ──
_OPORT_AD = ("conforme al artículo 126 del Código de Procedimientos Civiles del Estado de Querétaro, es decir, el veintisiete "
             "de enero de dos mil veintiséis, por lo que")
_cnpcf_agr = con(AD, _OPORT_AD, "conforme al artículo 227, fracción I, del Código Nacional de Procedimientos Civiles y "
                 "Familiares, de aplicación supletoria en términos del artículo 167 de la Ley Agraria, por lo que")
ok(not _det(_cnpcf_agr, FICHA_AD, DATOS_AD, "amparo_directo", "g"),
   "C7: «227, fracción I, del Código Nacional…, de aplicación supletoria en términos del artículo 167 de la Ley Agraria, "
   "por lo que el plazo… de la Ley de Amparo» fuera de la Ciudad de México (el secretario lo eligió: ya rige para ese "
   "juicio): no es el supletorio de la Ley de Amparo")
_cfpc_agr = con(AD_CDMX, "conforme al artículo 120 del Código de Procedimientos Civiles para el Distrito Federal, es "
                "decir, el veintisiete de enero de dos mil veintiséis, por lo que",
                "conforme al artículo 321 del Código Federal de Procedimientos Civiles, de aplicación supletoria en "
                "materia agraria, por lo que")
ok(not _det(_cfpc_agr, FICHA_CDMX, DATOS_CDMX, "amparo_directo", "g"),
   "C7: «321 del Código Federal…, de aplicación supletoria en materia agraria» en la Ciudad de México (juicio "
   "iniciado antes): tampoco")
ok(len(_det(con(AD, "los artículos 129 y 202 del Código Federal de Procedimientos Civiles",
                "los artículos 312 y 344 del Código Nacional de Procedimientos Civiles y Familiares"),
            FICHA_AD, DATOS_AD, "amparo_directo", "g")) == 1,
   "…y el Código Nacional como supletorio DE LA LEY DE AMPARO en Querétaro se sigue acusando")

# ── «AUTORIDAD RESPONSABLE:» DEL AR ES DE LA DEMANDA (revisión AR, 3-oct-2026; variante AR 448 sin demanda) ──
# El hueco del rótulo en el resultando de la demanda se atribuía a `acto.organo`
# (el juzgado que dictó la recurrida) y la verja decía «LA FICHA SÍ LO TRAE» con
# el Juez de Distrito: contradecía el aviso del compositor.
_ar_sin_aut = con(AR, "AUTORIDAD RESPONSABLE:\nDirector de Ingresos del Municipio de Querétaro.",
                  "AUTORIDAD RESPONSABLE:\n*********")
_f_sin_dem = dict(FICHA_AR, demanda={"fecha": "2025-11-03", "actos": FICHA_AR["demanda"]["actos"]})
_av_aut = [x["aviso"] for x in (_seguro(vp.revisar_detalle, _ar_sin_aut, _f_sin_dem, DATOS_AR, "amparo_revision") or [])
           if "AUTORIDAD RESPONSABLE:" in x["aviso"] or "(acto.organo" in x["aviso"]
           or "demanda.autoridades" in x["aviso"]]
ok(_av_aut and all("(demanda.autoridades)" in a and "(acto.organo" not in a and "SÍ LO TRAE" not in a for a in _av_aut),
   "AR: el hueco de «AUTORIDAD RESPONSABLE:» es de demanda.autoridades, y sin ellas en la ficha no se dice que la "
   "ficha lo traiga" + (f" — {[a[:160] for a in _av_aut]}" if _av_aut else " — (sin aviso del hueco)"))
ok(vp._k_organo({}, "ÓRGANO RECURRIDO:\n", "", "", "amparo_revision") == "acto.organo"
   and vp._k_organo({}, "AUTORIDAD RESPONSABLE:\n", "", "", "amparo_directo") == "responsable"
   and vp._k_organo({}, "AUTORIDADES RESPONSABLES:\n", "", "", "queja")[1] == "demanda.autoridades",
   "«ÓRGANO RECURRIDO» sigue siendo acto.organo; en el AD la responsable; en la queja con demanda, la demanda")

# ── EL COMPOSITOR POR TIPO CON LAS FORMAS NUEVAS (si ya las compone) ──
if _componer is not None:
    for _n, _tp, _f, _dd, _lista, _num, _mat, _tras in (
            ("queja fr. I con la demanda y un AR resuelto", "queja", dict(FICHA_Q_C3, relacionados=_L_Q),
             DATOS_Q, _L_Q, "355/2026", "civil", "Legitimación"),
            ("AR con un AD en la misma sesión", "amparo_revision", dict(FICHA_AR, relacionados=[_L_AR[0]]),
             DATOS_AR, [_L_AR[0]], "410/2026", "administrativa", "Legitimación")):
        try:
            _f = dict(_f, fuentes=dict(_f.get("fuentes") or {}, relacionados="secretario"))
            r = _componer(_tp, _f, dict(_dd, magistrado="Luis Armando Pérez Topete", materia=_f["materia"]))
            _ords = ['PRIMERO', 'SEGUNDO', 'TERCERO', 'CUARTO', 'QUINTO', 'SEXTO', 'SÉPTIMO']
            cuerpo = "\n".join(f"{_ords[i]}. {x['titulo'].rstrip('.')}. {x['texto']}"
                               for i, x in enumerate(r["resultandos"][:7]))
            _rot, _txt = _ta6.considerando_relacionados(_tp, _num, _mat, (r.get("datos_extra") or {}).get("relacionados")
                                                        or [], _HN_QRO)
            _cab = (_ta6.encabezado_de(_tp, _mat, _num) + ".\n" + _ta6.rotulo_relacionados(
                (r.get("datos_extra") or {}).get("relacionados") or [], _mat))
            texto = (f"{_cab}\nV I S T O, {r['visto']}\nR E S U L T A N D O\n{cuerpo}\nC O N S I D E R A N D O\n"
                     f"PRIMERO. {_rot} " + _txt.replace("\n", "\n") + "\n")
            _dd2 = dict(_dd, procesal=r.get("datos_extra") or {})
            av = vp.revisar_detalle(texto, _f, _dd2, _tp)
            _rel_ok = "relacionado con" in r["visto"] and bool((r.get("datos_extra") or {}).get("relacionados"))
            ok(_rel_ok and not av, f"el compositor: {_n} — el V I S T O «…, relacionado con …», los resultandos y el "
                                   f"considerando de `tipos_asunto` pasan sin avisos"
               + (f" — {[x['aviso'][:120] for x in av[:2]]}" if av else "")
               + ("" if _rel_ok else " — el compositor todavía no pone el relacionado en el V I S T O"))
        except Exception as ex:
            ok(False, f"el compositor: {_n}: {type(ex).__name__}: {ex}")

print()
if FALLOS:
    print(f"FALLAN {len(FALLOS)}: " + " · ".join(FALLOS))
    raise SystemExit(1)
print("RESULTADO: TODAS LAS COMPROBACIONES PASAN")
