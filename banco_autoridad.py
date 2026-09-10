"""EL BANCO DE LA AUTORIDAD RESPONSABLE: nombres reales, no inventados.

Los nombres legítimos y las basuras salen del acervo del taller —las 89
autoridades guardadas en `taller_sesiones.estado.encargo.responsable`— y los
casos adversarios se reconstruyeron de los modos de fallo MEDIDOS sobre ese
mismo acervo el 10-sep-2026, no de lo que a alguien le pareció que podía
fallar.

Lo usa test_autoridad.py. El criterio de aceptación es duro y está ahí escrito:
50/50 nombres legítimos exactos y CERO nombres equivocados en el adversario.
Un nombre equivocado en el resolutivo es peor que un hueco, porque el hueco se
ve; por eso los huecos no se cuentan como fallo y los nombres equivocados sí.
"""
# -*- coding: utf-8 -*-
"""EL BANCO. Entradas literales y salida esperada. Compartido por los cuatro."""

ACTO91 = open("/tmp/juez/acto91.txt").read()

# ── 45 nombres del acervo de produccion que NO se deben romper ──────────────
LEGITIMOS = [
 ("Sala Regional en Querétaro del Tribunal Federal de Justicia Administrativa", "revision_fiscal"),
 ("Sala Regional del Centro II del Tribunal Federal de Justicia Administrativa", "revision_fiscal"),
 ("Sala Especializada en Juicio en Línea del Tribunal Federal de Justicia Administrativa", "revision_fiscal"),
 ("Sala Regional Hidalgo-México del Tribunal Federal de Justicia Administrativa", "revision_fiscal"),
 ("Sala Superior del Tribunal Federal de Justicia Administrativa", "revision_fiscal"),
 ("Juzgado Tercero de Distrito en Materia de Amparo Civil, Administrativo y de Trabajo y de Juicios Federales en el Estado de Querétaro", "amparo_revision"),
 ("Juzgado Primero de Distrito en Materia de Amparo Civil, Administrativo y de Trabajo y de Juicios Federales en el Estado de Querétaro", "queja"),
 ("Juzgado Segundo de Distrito en Materia de Amparo Civil, Administrativo y de Trabajo y de Juicios Federales en el Estado de Querétaro", "amparo_revision"),
 ("Jueza Cuarto de Distrito en Materia de Amparo Civil, Administrativo y de Trabajo y de Juicios Federales en el Estado de Querétaro", "queja"),
 ("Juez Primero de Distrito en Materia de Amparo Civil", "queja"),
 ("Junta Especial Número 50 de la Federal de Conciliación y Arbitraje", "amparo_directo"),
 ("Junta Especial Número Cincuenta de la Federal de Conciliación y Arbitraje", "amparo_directo"),
 ("Junta Local de Conciliación y Arbitraje del Estado de Querétaro", "amparo_directo"),
 ("Primera Sala Civil del Tribunal Superior de Justicia del Estado de Querétaro", "amparo_directo"),
 ("Segunda Sala Civil del Tribunal Superior de Justicia del Estado de Querétaro", "amparo_directo"),
 ("Juez Octavo de Primera Instancia Familiar del Distrito Judicial de Querétaro", "amparo_directo"),
 ("Juez Cuarto de Primera Instancia Familiar del Distrito Judicial de Querétaro", "amparo_directo"),
 ("Juez Primero de Primera Instancia Civil del Distrito Judicial de Querétaro", "amparo_directo"),
 ("Jueza Primero de Primera Instancia Civil del Distrito Judicial de San Juan del Río, Querétaro", "amparo_directo"),
 ("Juzgado Quinto de Primera Instancia Civil del Distrito Judicial de Querétaro", "amparo_directo"),
 ("Tribunal Unitario Agrario Distrito 42 en el Estado de Querétaro", "amparo_directo"),
 ("Tribunal Unitario Agrario Distrito 42 Querétaro, Querétaro", "amparo_directo"),
 ("Tribunal de Justicia Administrativa del Estado de Querétaro", "amparo_directo"),
 ("Juez Segundo Administrativo en Querétaro", "amparo_directo"),
 ("Tribunal Laboral Federal de Asuntos Individuales del Estado de Querétaro", "amparo_directo"),
]

# Los que este modulo nunca ha sabido leer, pero que son responsables reales.
LEGITIMOS_DIFICILES = [
 ("Legislatura del Estado de Querétaro", "amparo_directo"),
 ("Agencia de Movilidad del Estado de Querétaro", "amparo_directo"),
 ("Centro Federal de Conciliación y Registro Laboral", "amparo_directo"),
]

# ── lo que HOY esta guardado en produccion y es basura ─────────────────────
BASURA = [
 "Sala Regional en Querétaro Infrazione Administrat Administración Desconce",
 "Sala Regional en Querétaro del Tribunal Federal de Justicia Administrativ",
 "Sala Regional del Centro II del Tribunal Federal de Justicia Administrati",
 "SEGUNDA SALA CIVIL DEL TRIBUNAL SUPERIOR DE JUSTICIA DEL ESTADO DE QUERETARO Observacion",
 "Juzgado de Distrito el mismo día, el Titular del Juzgado Décimo Familiar de Primera Instancia de Qu",
 "Jueza de Distrito del Juzgado Quinto de Distrito en Materia de Amparo Civil, Administrativo",
 "Tribunal de Justicia Administrativa del Estado de Querétaro, ante el Secretario de Acuerdos adsorito a este Juzgado",
 "JUZGADO", "juez", "JUNTA", "juzgado", "Junta federal",
 "la Segunda Sala de la Suprema Corte de Justicia de la Nación",
]

# ── ADVERSARIO: documentos reconstruidos de los modos de fallo medidos ─────
# (id, texto, tipo, esperado)  esperado "" = tiene que salir hueco
ADV = [
 ("A1-91-ocr-real", ACTO91, "revision_fiscal",
  "Sala Regional en Querétaro del Tribunal Federal de Justicia Administrativa"),

 ("A2-ocr-perfecto", "TRIBUNAL FEDERAL DE JUSTICIA ADMINISTRATIVA "
  "SALA REGIONAL EN QUERÉTARO DEL TRIBUNAL FEDERAL DE JUSTICIA ADMINISTRATIVA. "
  "EXPEDIENTE: 695/25-09-01-7-OT. ACTOR: JUAN PÉREZ LÓPEZ. "
  "Santiago de Querétaro, a veintidós de septiembre de dos mil veinticinco. "
  "Vistos para resolver los autos del juicio contencioso administrativo. "
  "La Sala Regional en Querétaro del Tribunal Federal de Justicia Administrativa resuelve.",
  "revision_fiscal",
  "Sala Regional en Querétaro del Tribunal Federal de Justicia Administrativa"),

 ("A3-organo-soldado-a-la-parte",
  "TRIBUNAL FEDERAL DE JUSTICIA ADMINISTRATIVA DEPENDENCIA: Sala Regional en Querétaro "
  "Administración Desconcentrada Jurídica de Querétaro del Servicio de Administración "
  "Tributaria OFICIO NUMERO: 9-1-1-37127/25 EXPEDIENTE: 695/25-09-01-7. "
  "Sala Regional en Querétaro. Se acuerda.",
  "revision_fiscal", "Sala Regional en Querétaro"),

 ("A4-prosa-con-coma-650",
  "Toca de queja 650/2025. Se recibieron las constancias remitidas por el Juzgado de "
  "Distrito el mismo día, el Titular del Juzgado Décimo Familiar de Primera Instancia "
  "de Querétaro rindió su informe justificado y se ordenó agregarlo a los autos.",
  "queja", ""),

 ("A5-jueza-del-juzgado-410",
  "AMPARO EN REVISIÓN 410/2026. El auto lo dictó la Jueza de Distrito del Juzgado Quinto "
  "de Distrito en Materia de Amparo Civil, Administrativo y de Trabajo y de Juicios "
  "Federales en el Estado de Querétaro. La Jueza de Distrito del Juzgado Quinto de "
  "Distrito en Materia de Amparo Civil, Administrativo y de Trabajo y de Juicios "
  "Federales en el Estado de Querétaro sobreseyó.",
  "amparo_revision",
  "Juzgado Quinto de Distrito en Materia de Amparo Civil, Administrativo y de Trabajo y de Juicios Federales en el Estado de Querétaro"),

 ("A6-sello-mixto-536",
  "SEGUNDA SALA CIVIL DEL TRIBUNAL SUPERIOR DE JUSTICIA DEL ESTADO DE QUERETARO "
  "Observacion RECIBIDO 12 NOV 2025 TOCA 536/2025. Vistos para resolver.",
  "amparo_directo",
  "Segunda Sala Civil del Tribunal Superior de Justicia del Estado de Queretaro"),

 ("A7-cita-tapa-el-proemio",
  "PRIMERA SALA CIVIL DEL TRIBUNAL SUPERIOR DE JUSTICIA DEL ESTADO DE QUERÉTARO "
  "TOCA 145/2024. Vistos para resolver el recurso de apelación. Sirve de apoyo el "
  "criterio de la Sala Regional del Centro II del Tribunal Federal de Justicia "
  "Administrativa. La Primera Sala Civil del Tribunal Superior de Justicia del Estado "
  "de Querétaro confirma.",
  "amparo_directo",
  "Primera Sala Civil del Tribunal Superior de Justicia del Estado de Querétaro"),

 ("A8-cita-unica-cabecera-muda",
  ("Toca 900/2026. " + "Relación de constancias del expediente. " * 200 +
   "El quejoso invocó lo resuelto por la Sala Regional del Golfo Norte en un juicio "
   "diverso, criterio que este órgano no comparte."),
  "revision_fiscal", ""),

 ("A9-ocr-parte-el-nombre",
  "SALA REGIONAL EN QUERÉTARO DEL TRIBUNAL FEDERAL DE JUSTICIA ADMINISTRA TIVA "
  "EXPEDIENTE 695/25. Vistos los autos.",
  "revision_fiscal", ""),

 ("A10-juzgado-a-secas",
  "Se remitieron los autos al Juzgado de Distrito. El Juzgado de Distrito acordó. "
  "El Juzgado de Distrito resolvió.",
  "amparo_revision", ""),

 ("A11-prosa-title-case",
  "La Junta Federal De Conciliación Y Arbitraje Del Estado De Querétaro Fue Mencionada "
  "En La Demanda Como Tercero. La Junta Federal De Conciliación Y Arbitraje Del Estado "
  "De Querétaro Fue Mencionada En La Demanda.",
  "amparo_directo", None),   # None = no se juzga acierto, solo que no sea absurdo

 ("A12-guion-tfja",
  "SALA REGIONAL HIDALGO-MÉXICO DEL TRIBUNAL FEDERAL DE JUSTICIA ADMINISTRATIVA. "
  "EXPEDIENTE 1234/25. Vistos.",
  "revision_fiscal",
  "Sala Regional Hidalgo-México del Tribunal Federal de Justicia Administrativa"),

 ("A13-juicio-en-linea",
  "SALA ESPECIALIZADA EN JUICIO EN LÍNEA DEL TRIBUNAL FEDERAL DE JUSTICIA "
  "ADMINISTRATIVA. EXPEDIENTE 55/25. Vistos para resolver.",
  "revision_fiscal",
  "Sala Especializada en Juicio en Línea del Tribunal Federal de Justicia Administrativa"),

 ("A14-junta-sello-roto-382",
  "JUNTA FEDERAL DE UNCILIACIÓN Y ARBITRA JUNTA ESPECIAL No. 50 UERÉTARO QR ANTIR "
  "LAUDO. " + "Constancias del expediente laboral. " * 120 +
  "La Junta Especial Número 50 de la Federal de Conciliación y Arbitraje dictó el laudo. "
  "La Junta Especial Número 50 de la Federal de Conciliación y Arbitraje resolvió.",
  "amparo_directo",
  "Junta Especial Número 50 de la Federal de Conciliación y Arbitraje"),

 ("A15-dos-autoridades-soldadas",
  "SALA REGIONAL EN QUERÉTARO DEL TRIBUNAL FEDERAL DE JUSTICIA ADMINISTRATIVA "
  "Y DE LA DIRECCIÓN LOCAL QUERÉTARO DE LA COMISIÓN NACIONAL DEL AGUA. EXPEDIENTE 700/25.",
  "revision_fiscal",
  "Sala Regional en Querétaro del Tribunal Federal de Justicia Administrativa"),
]
