# -*- coding: utf-8 -*-
"""El vocabulario canónico de la jurimetría (jurimetria_canon.py), sin red. 22-nov-2026."""
import sys
import jurimetria_canon as jc

fallos = []


def igual(nombre, obtenido, esperado):
    if obtenido != esperado:
        fallos.append(f"{nombre}: {obtenido!r} ≠ {esperado!r}")


# ── estado ↔ circuito: lo que manda la web y lo que guarda user_profiles ─────
for e, c in [("CIUDAD_DE_MEXICO", "1"), ("MEXICO", "2"), ("Estado de México", "2"), ("NUEVO_LEON", "4"),
             ("BAJA_CALIFORNIA", "15"), ("BAJA_CALIFORNIA_SUR", "26"), ("QUERETARO", "22"), ("Querétaro", "22"),
             ("SAN_LUIS_POTOSI", "9"), ("Marte", None)]:
    igual(f"circuito_de_estado({e})", jc.circuito_de_estado(e), c)
igual("estado_de_circuito(22)", jc.estado_de_circuito("22"), "Querétaro")
igual("circuito_de_consulta('022')", jc.circuito_de_consulta("022"), "22")
igual("circuito_de_consulta(estado)", jc.circuito_de_consulta("", "JALISCO"), "3")
igual("circuito_de_consulta('40')", jc.circuito_de_consulta("40"), None)

# ── ordinales ───────────────────────────────────────────────────────────────
for t, n in [("Primer", 1), ("Décimo Primer", 11), ("Decimosexto", 16), ("Décimo Sexto", 16), ("decimoctavo", 18),
             ("Vigésimo Segundo", 22), ("Trigésimo", 30), ("Trigésimo Segundo", 32), ("Undécimo", 11), ("Casa", None)]:
    igual(f"numero_ordinal({t})", jc.numero_ordinal(t), n)
igual("nombre_ordinal(11)", jc.nombre_ordinal(11), "Décimo Primer")
igual("nombre_ordinal(22)", jc.nombre_ordinal(22), "Vigésimo Segundo")

# ── tribunal: las tres grafías del acervo y el nombre oficial dan la MISMA clave ─
OF3 = ("Tercer Tribunal Colegiado en Materias Administrativa y Civil del Vigésimo Segundo Circuito, "
       "con residencia en Querétaro, Querétaro")
t = jc.tribunal_de_nombre(OF3)
igual("3TCC oficial · id", t and t["id"], "22:3:ADM-CIV")
igual("3TCC oficial · sede", t and t["sede"], "Querétaro, Querétaro")
igual("3TCC oficial · nombre", t and t["nombre"],
      "Tercer Tribunal Colegiado en Materias Administrativa y Civil del Vigésimo Segundo Circuito")
igual("3TCC clave", (jc.tribunal("3TCC", "22") or {}).get("id"), "22:3:ADM-CIV")
igual("TCC_PENAL clave", (jc.tribunal("TCC_PENAL", "22") or {}).get("id"), "22:0:ADM-PEN")
igual("TCC_ADM clave", (jc.tribunal("TCC_ADM", "22") or {}).get("id"), "22:0:ADM-LAB")
igual("8TCC_ADM clave", (jc.tribunal("8TCC_ADM", "1") or {}).get("id"), "1:8:ADM")
igual("8TCC_ADM nombre", (jc.tribunal("8TCC_ADM", "1") or {}).get("nombre"),
      "Octavo Tribunal Colegiado en Materia Administrativa del Primer Circuito")
igual("8TCC_ADM = oficial", (jc.tribunal_de_nombre(
    "Octavo Tribunal Colegiado en Materia Administrativa del Primer Circuito") or {}).get("id"), "1:8:ADM")
igual("11TCC_CIV nombre", (jc.tribunal("11TCC_CIV", "1") or {}).get("nombre"),
      "Décimo Primer Tribunal Colegiado en Materia Civil del Primer Circuito")
igual("2TCC_LAB c6", (jc.tribunal("2TCC_LAB", "6") or {}).get("id"), "6:2:LAB")
igual("c16 escrito", (jc.tribunal("Primer Tribunal Colegiado en Materia de Trabajo del Decimosexto Circuito con "
                                  "residencia en Guanajuato Gto", "16") or {}).get("id"), "16:1:LAB")
igual("c16 oficial", (jc.tribunal_de_nombre("Primer Tribunal Colegiado en Materia de Trabajo del Décimo Sexto "
                                            "Circuito, con residencia en Guanajuato, Guanajuato") or {}).get("id"),
      "16:1:LAB")
igual("Penal y Administrativa = Administrativa y Penal",
      (jc.tribunal_de_nombre("Tribunal Colegiado en Materias Administrativa y Penal del Vigésimo Segundo Circuito")
       or {}).get("id"), "22:0:ADM-PEN")
igual("clave rara no se adivina", jc.tribunal("XYZ", "1"), None)
igual("clave nula", jc.tribunal(None, "1"), None)

# ── tipo y materia ──────────────────────────────────────────────────────────
for raw, t in [("Amparo Directo", "amparo_directo"), ("amparo_directo", "amparo_directo"),
               ("Amparo en Revisión", "amparo_revision"), ("amparo_revision", "amparo_revision"),
               ("Revisión Fiscal", "revision_fiscal"), ("Queja", "queja"), ("Reclamaciones", "reclamacion"),
               ("Inconformidades", "inconformidad"), ("Conflictos Competenciales", "conflicto_competencial"),
               ("Incidente en Revisión Civil", "revision_incidental"), ("Incidentes de Inejecución", "inejecucion"),
               ("Repetición del Acto Reclamado", "repeticion_acto"), ("Revisión Contenciosa Administrativa",
                                                                      "revision_contencioso"),
               ("null", None), ("", None)]:
    igual(f"tipo_canon({raw})", jc.tipo_canon(raw), t)
igual("tipo_de_consulta('AD')", jc.tipo_de_consulta("AD"), "amparo_directo")
igual("tipo_de_consulta('Amparo en revisión')", jc.tipo_de_consulta("Amparo en revisión"), "amparo_revision")
for raw, m in [("administrativa", ("administrativa", None)), ("Administrativa", ("administrativa", None)),
               ("fiscal", ("administrativa", "fiscal")), ("Trabajo", ("laboral", None)),
               ("Civil", ("civil", None)), ("mercantil", ("civil", "mercantil")), ("comun", ("comun", None)),
               ("común", ("comun", None)), ("null", (None, None))]:
    igual(f"materia_canon({raw})", jc.materia_canon(raw), m)
igual("la web manda «administrativo»", jc.materia_de_consulta("administrativo"), "administrativa")

# ── sentido, prospera y quejoso ─────────────────────────────────────────────
igual("sentido null", jc.sentido_canon("null"), None)
igual("sentido vacío", jc.sentido_canon(""), None)
igual("sin materia", jc.sentido_canon("sin materia"), "sin_materia")
igual("parcialmente_concede", jc.sentido_canon("parcialmente_concede"), "concede_parcial")
igual("raro", jc.sentido_canon("resuelve_algo_raro"), "otro")
igual("AD concede", jc.prospera("amparo_directo", "concede"), "si")
igual("AD sobresee", jc.prospera("amparo_directo", "sobresee"), "sin_fondo")
igual("AR confirma", jc.prospera("amparo_revision", "confirma"), "no")
igual("AR modifica es parcial, no sobreseer", jc.prospera("amparo_revision", "modifica"), "parcial")
igual("AR concede no basta", jc.prospera("amparo_revision", "concede"), None)
igual("Q infundado", jc.prospera("queja", "infundado"), "no")
igual("RF confirma: no prospera la autoridad", jc.prospera("revision_fiscal", "confirma"), "no")
igual("RF confirma: gana el particular", jc.quejoso("revision_fiscal", "confirma"), "gana")
igual("RF revoca: pierde el particular", jc.quejoso("revision_fiscal", "revoca"), "pierde")
igual("AR concede: gana el quejoso", jc.quejoso("amparo_revision", "concede"), "gana")
igual("AR confirma: no se sabe sin el recurrente", jc.quejoso("amparo_revision", "confirma"), None)
igual("AD niega", jc.quejoso("amparo_directo", "niega"), "pierde")
igual("conflicto competencial", jc.prospera("conflicto_competencial", "competencia"), None)

# ── síntesis oficial de la OAJ ──────────────────────────────────────────────
for s, q in [("Confirma y Ampara", "gana"), ("Confirma y No ampara", "pierde"), ("Revoca y Sobresee", "sin_fondo"),
             ("1. Queda firme sobreseimiento. 2. Se modifica. 3. Se sobresee. 4. Ampara.", "gana_en_parte"),
             ("Primero: Se confirma.\nSegundo: Ampara.", "gana"), ("Sin materia", None), ("", None)]:
    igual(f"quejoso_de_sintesis({s!r})", jc.quejoso_de_sintesis(s), q)
for s, d in [("Confirma y Ampara", "confirma"), ("Fundado ", "fundado"), ("Es infundado el recurso", "infundado"),
             ("1. Revoca. 2. Ampara.", "revoca"), ("Único: Sin materia.", "sin_materia"),
             ("Confirma y revoca", None)]:
    igual(f"decision_de_sintesis({s!r})", jc.decision_de_sintesis(s), d)

# ── fecha y periodo ─────────────────────────────────────────────────────────
igual("fecha imposible", jc.fecha_iso("2023-02-29"), None)
igual("fecha OAJ dd-mm-aaaa", jc.fecha_iso("08-07-2024"), "2024-07-08")
igual("fecha futura", jc.fecha_iso("2031-01-01"), None)
igual("fecha basura", jc.fecha_iso("null"), None)
igual("periodo del corte", jc.periodo("2025-09-01"), "nueva_integracion")
igual("periodo de antes", jc.periodo("2025-08-31"), "anterior")
igual("periodo sin fecha", jc.periodo(None), None)
igual("ponente", jc.ponente_canon("Mgdo. JUAN BONILLA PIZANO"), "Juan Bonilla Pizano")

# ── el punto completo ───────────────────────────────────────────────────────
c = jc.campos({"circuito": "1", "tribunal": "8TCC_CIV", "tipo_asunto": "Amparo Directo", "materia": "civil",
               "sentido": "confirma", "fecha_sentencia": "2024-07-08", "file_id": "467-2024_35846084",
               "magistrado_ponente": "Armengol Alonso"})
igual("campos · estado", c["j_estado"], "Ciudad de México")
igual("campos · tribunal", c["j_tribunal"], "1:8:CIV")
igual("campos · AD confirma no se inventa", c["j_prospera"], None)
igual("campos · fecha", c["j_fecha"], "2024-07-08T00:00:00Z")
igual("campos · año", c["j_anio"], 2024)
igual("campos · fuente", c["j_fecha_fuente"], "lectura")
igual("campos · sentencia", c["j_sentencia"], "1:35846084")
o = jc.campos({"circuito": "22", "tribunal": "3TCC", "tipo_asunto": "Amparo en Revisión", "materia": "Administrativa",
               "sentido": "confirma", "neun": 25132210},
              {"organo": OF3, "fecha": "17-10-2025", "titular": "J. Guadalupe Tafoya Hernández",
               "sintesis": "Confirma y Ampara"})
igual("oficial · tribunal", o["j_tribunal"], "22:3:ADM-CIV")
igual("oficial · quejoso", o["j_quejoso"], "gana")
igual("oficial · prospera (recurrente)", o["j_prospera"], "no")
igual("oficial · fecha gana", o["j_fecha"], "2025-10-17T00:00:00Z")
igual("oficial · periodo", o["j_periodo"], "nueva_integracion")
igual("oficial · ponente", o["j_ponente"], "J. Guadalupe Tafoya Hernández")
igual("oficial · sede", o.get("j_sede"), "Querétaro, Querétaro")
igual("todos los índices son campos", set(jc.INDICES) - set(o), set())

if fallos:
    print("FALLOS:")
    for f in fallos:
        print("  ·", f)
    sys.exit(1)
print("OK · jurimetria_canon: estados, tribunales, tipos, materias, sentidos, síntesis, fechas")
