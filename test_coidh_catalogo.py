"""El catálogo de la Corte IDH y el resolvedor de citas de casos — 25-sep-2026.

    .venv/bin/python test_coidh_catalogo.py

Sin red, sin Qdrant, sin modelos: lee `datos/coidh_catalogo.json` y nada más.
Las negativas no son de adorno: el prototipo del plan se disparaba en el
30.8 % de 4,000 preguntas reales («de la» → Mapiripán 1,149 veces; «mi
cliente José García Rodríguez» → García Rodríguez Vs. México). Varias de las
de abajo salieron de escritos reales de ese mismo día (una nota al pie con
Ruiz-Mateos c. España, un «JUANA ANGEL HERNANDEZ CORZO» en una firma, el
asunto Viviana Gallardo), recortadas y sin datos de nadie.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import coidh_catalogo as K

FALLOS = []


def ok(cond, que):
    print(f"   {'PASA ' if cond else 'FALLA'}  {que}")
    if not cond:
        FALLOS.append(que)


def uno(texto):
    """El único resultado del texto, o None si hay cero o varios."""
    rs = K.resolver_citas_coidh(texto)
    return rs[0] if len(rs) == 1 else None


def nada(texto):
    return K.resolver_citas_coidh(texto) == []


cat = K.catalogo()
docs, casos = cat["docs"], cat["casos"]

print("\n1 · EL CATÁLOGO CUADRA CON EL LISTADO DEL SITIO")
cif = cat["data"]["cifras"]
ok(cif["documentos"] == {"CC": 597, "OC": 33, "SS": 905, "MP": 785},
   "597 sentencias, 33 de la Serie A, 905 supervisiones y 785 medidas provisionales")
ok(cif["serie_c"]["faltan"] == [597] and cif["serie_c"]["maximo"] == 598,
   "la Serie C va del 1 al 598 y sólo falta la 597: 598 − 1 = 597")
ok(all(f"C-{n}" in docs for n in range(1, 599) if n != 597), "cada número de la Serie C tiene su documento")
ok("C-155" in docs and "C-369" in docs,
   "Vargas Areco (C-155) y Trueba Arciniega (C-369) no se pierden: el acto vacío no tumba el parser")
ok(all(d["url_oficial"].startswith("https://www.corteidh.or.cr/docs/") for d in docs.values()),
   "las 2,320 URL oficiales normalizadas a https://www.")
ok(all(d["url_oficial"] == d["url_oficial"].lower() for d in docs.values()), "y en minúsculas")
ok(docs["C-385"]["url_oficial"].endswith("/seriec_384_esp.pdf")
   and docs["C-384"]["url_oficial"].endswith("/seriec_385_esp.pdf"),
   "384/385: la URL se toma del listado (Ruiz Fuentes vive en seriec_384), nunca se arma con el número")
ok(any(a.startswith("archivo_con_otro_numero") for a in docs["C-385"]["anomalias"]), "y queda marcada la anomalía")
ok(any("seriec_461" in a for a in docs["C-441"]["anomalias"]), "la 441 marca su enlace extra a seriec_461")
ok(sum("sufijo_esp1" in d.get("anomalias", []) for d in docs.values() if d["clase"] == "CC") == 13
   and sum("sufijo_esp2" in d.get("anomalias", []) for d in docs.values() if d["clase"] == "CC") == 2,
   "13 sentencias con _esp1 y 2 con _esp2, marcadas")
ok(docs["C-89"]["url_listado"].endswith("Seriec_89_esp.pdf") and docs["C-89"]["url_oficial"].endswith("seriec_89_esp.pdf"),
   "«Seriec_89» del listado queda en url_listado; la oficial va en minúsculas")
ok(docs["C-158"]["caso_id"] == docs["C-174"]["caso_id"], "Cesados: fondo (C-158) e interpretación (C-174) son un caso")
ok(docs["C-80"]["caso_id"] == docs["C-94"]["caso_id"] == docs["C-82"]["caso_id"],
   "Hilaire, Constantine y Benjamin (acumulados, ficha 269) son un caso")
ok(docs["C-154"]["acto_principal"] and docs["C-4"]["acto_principal"] and not docs["C-7"]["acto_principal"],
   "la resolución principal es la de fondo (Velásquez Rodríguez: C-4, no C-7)")
ok([v["slug"] for v in docs["C-158"]["votos"]] == ["garcia-ramirez", "cancado-trindade"],
   "votos con slug estable, del campo del listado (revisión B.13)")
ok(all(a["slug"] == "vio-grossi" for d in docs.values() for v in d["votos"] for a in v["autores"]
       if "vio grossi" in K._norm(a["nombre"])), "«Eduardo Vio Grossi» y «Vio Grossi» son el mismo slug")
ok("Control de convencionalidad" in (docs["C-154"].get("palabras_clave") or []),
   "Almonacid trae sus palabras clave de la ficha técnica 335")
ok(docs["SS-gelman-uruguay-2013-03-20"]["url_oficial"].endswith("/supervisiones/gelman_20_03_13.pdf")
   and docs["SS-gelman-uruguay-2013-03-20"]["caso_id"] == "gelman-uruguay",
   "la supervisión de Gelman del 20-mar-2013 está enlazada a su caso")
ok(docs["A-24"]["oc"] == "OC-24/17" and docs["A-24"]["oc_anio"] == 2017, "OC-24/17 = Serie A 24")
ok(not any(a["a"] in ("de la", "de los", "del", "la", "el", "amparo") for c in casos.values() for a in c["alias"]),
   "ningún alias es un residuo («de la», «del») ni «amparo» a secas")
ok(cat["alias"][("el", "amparo")]["clase"] == "comun", "«el amparo» es frase común: exige «Vs.» o «Caso» con mayúscula")

print("\n2 · LO QUE TIENE QUE ENCONTRAR")
r = uno("Almonacid ¶124")
ok(r and r["doc_id"] == "C-154" and r["parrafo"] == 124 and r["seg"] == "sentencia",
   "«Almonacid ¶124» → C-154, párr. 124")
ok(r and r["url_oficial"] == "https://www.corteidh.or.cr/docs/casos/articulos/seriec_154_esp.pdf",
   "con la URL oficial del listado")
ok(r and r["llaves"] == ["C-154|s|124"], "y su llave de la colección coidh («C-154|s|124»)")
ok(r and r["cita_canonica"].endswith("Serie C No. 154, párr. 124."), "y la cita canónica con el párrafo")
r = uno("caso Radilla")
ok(r and r["doc_id"] == "C-209" and r["parrafo"] is None, "«caso Radilla» → C-209, sin párrafo")
r = uno("OC-24/17 párr. 26")
ok(r and r["doc_id"] == "A-24" and r["parrafos"] == [26] and r["seg"] == "sentencia" and r["voto_autor"] is None,
   "«OC-24/17 párr. 26» → Serie A 24, párr. 26 de la opinión (no el de un voto)")
r = uno("voto razonado de García Ramírez en Trabajadores Cesados párr. 12")
ok(r and r["doc_id"] == "C-158" and r["seg"] == "voto" and r["voto_autor"] == "garcia-ramirez" and r["parrafos"] == [12],
   "«voto razonado de García Ramírez en Trabajadores Cesados párr. 12» → C-158, voto, párr. 12")
ok(r and r["llaves"] == ["C-158|v:garcia-ramirez|12"], "llave del voto: «C-158|v:garcia-ramirez|12»")
r = uno("Tzompaxtle, resolutivo 8")
ok(r and r["doc_id"] == "C-470" and r["seg"] == "resolutivos" and r["parrafos"] == [8],
   "«Tzompaxtle, resolutivo 8» → C-470, resolutivo 8")
r = uno("Campo Algodonero")
ok(r and r["doc_id"] == "C-205", "«Campo Algodonero» → C-205 (apodo con mayúsculas, sin más pista)")
r = uno("Serie C No. 154, párr. 124")
ok(r and r["doc_id"] == "C-154" and r["parrafos"] == [124] and r["via"] == "serie",
   "«Serie C No. 154, párr. 124» → C-154 por llave directa")
r = uno("Almonacid-Arellano et al. v. Chile, para. 124")
ok(r and r["doc_id"] == "C-154" and r["parrafos"] == [124], "en inglés: «Almonacid-Arellano et al. v. Chile, para. 124»")
r = uno("Corte IDH. Caso Almonacid Arellano y otros Vs. Chile. Excepciones Preliminares, Fondo, Reparaciones y "
        "Costas. Sentencia de 26 de septiembre de 2006. Series C No. 154, párr. 124")
ok(r and r["doc_id"] == "C-154" and r["confianza"] == "alta", "la cita oficial completa («Series C» también)")
r = uno("sentencia de la Corte Interamericana de Derechos Humanos: campo algodonero (ciudad juarez)")
ok(r and r["doc_id"] == "C-205", "en minúsculas, con la Corte Interamericana al lado (pregunta real)")
r = uno("opinión consultiva 21/14, párrafo 31")
ok(r and r["doc_id"] == "A-21" and r["parrafos"] == [31], "«opinión consultiva 21/14, párrafo 31» → A-21")
r = uno("Myrna Mack Chang, voto de García Ramírez, párr. 27")
ok(r and r["doc_id"] == "C-101" and r["voto_autor"] == "garcia-ramirez" and r["parrafos"] == [27],
   "Myrna Mack, voto de García Ramírez, párr. 27 (el antecedente de la línea)")
r = uno("Gelman, supervisión de cumplimiento de 20 de marzo de 2013, considerando 67")
ok(r and r["doc_id"] == "SS-gelman-uruguay-2013-03-20" and r["seg"] == "considerandos" and r["parrafos"] == [67],
   "Gelman, supervisión del 20-mar-2013, considerando 67")

print("\n3 · PÁRRAFOS: RANGOS, LISTAS, VARIOS SEGMENTOS, VARIAS CITAS")
r = uno("Radilla Pacheco vs. México, párrs. 338 a 341")
ok(r and r["parrafos"] == [338, 339, 340, 341], "«párrs. 338 a 341» → 338…341")
rs = K.resolver_citas_coidh("Tzompaxtle Tecpile y otros Vs. México, párrs. 216 a 219 y resolutivos 7 y 8")
ok(len(rs) == 2 and rs[0]["parrafos"] == [216, 217, 218, 219] and rs[1]["seg"] == "resolutivos"
   and rs[1]["parrafos"] == [7, 8], "párrafos y resolutivos de la misma cita salen por separado")
rs = K.resolver_citas_coidh("caso Radilla, párr. 338; Cabrera García y Montiel Flores, párr. 225")
ok([(r["doc_id"], r["parrafos"]) for r in rs] == [("C-209", [338]), ("C-220", [225])],
   "dos citas en una línea, cada párrafo con su caso")
r = uno("Andrade Salmón vs. Bolivia, párrs. 93, 94, 96, 100, 101 y 102")
ok(r and r["parrafos"] == [93, 94, 96, 100, 101, 102], "lista con comas e «y»")
r = uno("el párrafo 124 del caso Almonacid Arellano")
ok(r and r["doc_id"] == "C-154" and r["parrafos"] == [124], "el párrafo antes del caso («el párrafo 124 del caso…»)")
r = uno("el artículo 1o constitucional, párrafo tercero, y el caso Radilla, párrafo 3 del artículo 13")
ok(r and r["doc_id"] == "C-209" and r["parrafos"] == [], "«párrafo 3 del artículo 13» no es un párrafo de Radilla")
rs = K.resolver_citas_coidh("Corte IDH, Caso Genie Lacayo, 29 de enero de 1997, Serie C N° 30, párr. 77, donde se "
                            "cita a la Corte Europea de Derechos Humanos, Motta c. Italia, 19 de febrero de 1991, "
                            "Serie A N° 195-A, párr. 30; Corte Europea de Derechos Humanos, Ruiz- Mateos c. España, "
                            "23 de junio de 1993, Serie A N° 262, párr. 30.")
ok([(r["doc_id"], r["parrafos"]) for r in rs] == [("C-30", [77])],
   "nota al pie real: Genie Lacayo párr. 77; los párr. 30 del Tribunal Europeo no se le pegan")

print("\n4 · NUNCA ELIGE EN SILENCIO")
r = uno("¿qué resolvió la Corte IDH en el caso González?")
ok(r and r["doc_id"] is None and r["confianza"] == "ambigua" and "C-205" in [c["doc_id"] for c in r["candidatos"]]
   and len(r["candidatos"]) >= 3, "«caso González» con la Corte IDH al lado → candidatos (Campo Algodonero entre ellos)")
r = uno("Velásquez Rodríguez, párr. 166")
ok(r and r["doc_id"] is None and [c["doc_id"] for c in r["candidatos"]] == ["C-1", "C-4", "C-7"],
   "antes de 2004: excepciones, fondo y reparaciones son Series distintas → candidatos, sin elegir")
r = uno("Velásquez Rodríguez (1988), párr. 166")
ok(r and r["doc_id"] == "C-4", "…con el año se desempata: 1988 es el fondo (C-4)")
r = uno("Caso Velásquez Rodríguez, Reparaciones, párr. 30")
ok(r and r["doc_id"] == "C-7", "…con la palabra también: «Reparaciones» → C-7")
r = uno("Caso Velásquez Rodríguez, Interpretación de la Sentencia de Reparaciones, párr. 30")
ok(r and r["doc_id"] == "C-9", "…e «Interpretación» → C-9")
r = uno("Caso Hilaire, párr. 20")
ok(r and r["doc_id"] is None and {c["doc_id"] for c in r["candidatos"]} == {"C-80", "C-94"},
   "Hilaire: sólo las resoluciones que llevan ese nombre (C-80 y la acumulada C-94)")
r = uno("Trabajadores Cesados, voto razonado de García Ramírez")
ok(r and r["doc_id"] == "C-158", "«Trabajadores Cesados» es ambiguo (Congreso, Petroperú, ENAPU); el voto desempata")
r = uno("Caso del Tribunal Constitucional")
ok(r and r["doc_id"] is None and {c["doc_id"] for c in r["candidatos"]} == {"C-71", "C-268"},
   "«Caso del Tribunal Constitucional» → Perú o Ecuador: candidatos")
r = uno("Caso Almonacid Arellano y otros Vs. Chile. Serie C No. 1541")
ok(r and r["doc_id"] == "C-154" and any("nota pegada" in n for n in r["notas"]),
   "«Serie C No. 1541» (llamada a nota pegada, así vienen los cuadernillos) se lee como 154 si el nombre casa")
r = uno("OC-24/18 párr. 26")
ok(r and r["doc_id"] == "A-24" and r["confianza"] == "media", "OC-24/18: el año no cuadra → confianza media y nota")
r = uno("caso I.V. Vs. Bolivia")
ok(r and r["doc_id"] == "C-329", "«I.V.» sólo con «Caso» y «Vs.»")
rs = K.resolver_citas_coidh("Caso Furlan y familiares Vs. Argentina. Excepciones Preliminares, Fondo, Reparaciones y "
                            "Costas. Sentencia de 31 de agosto de 2012. Serie C No. 212, párr. 133.")
ok(len(rs) == 1 and rs[0]["doc_id"] is None and rs[0]["parrafos"] == [133]
   and {c["doc_id"] for c in rs[0]["candidatos"]} == {"C-246", "C-212"},
   "errata del Cuadernillo 5 (Furlan con «Serie C No. 212», que es Chitay Nech): conflicto, no se le pega el "
   "párr. 133 a otro caso")
ambiguos = [" ".join(k) for k, v in cat["alias"].items() if len(v["casos"]) > 1 and v["clase"] in ("palabra", "apellidos")]
malos = [a for a in ambiguos[:60] if any(x["doc_id"] for x in K.resolver_citas_coidh(f"Corte IDH, caso {a.title()}, párr. 10"))
         and len({x["caso_id"] for x in K.resolver_citas_coidh(f"Corte IDH, caso {a.title()}, párr. 10") if x["doc_id"]}) > 1]
ok(not malos, f"ningún alias ambiguo elige caso a ciegas ({len(ambiguos[:60])} probados)")

print("\n5 · LO QUE NO ES UN CASO")
for t, que in [
    ("mi cliente José García Rodríguez fue detenido", "un cliente que se apellida como un caso"),
    ("juicio de amparo indirecto", "«amparo» (El Amparo Vs. Venezuela)"),
    ("pena convencional en un arrendamiento", "«pena convencional»"),
    ("San Juan del Río", "«San Juan» (Comunidad Garífuna de San Juan)"),
    ("de la Ley Federal del Trabajo", "«de la» (el residuo de «Caso de la “Masacre de Mapiripán”»)"),
    ("el caso González que llevo en el juzgado", "«el caso González» sin nada interamericano: es SU cliente"),
    ("en caso de que el amparo sea procedente", "«en caso de que…» es condicional, no cita"),
    ("la Corte Suprema de Justicia de la Nación resolvió", "«Corte Suprema de Justicia» (Quintana Coello Vs. Ecuador)"),
    ("el Tribunal Constitucional de España", "«Tribunal Constitucional» sin «Caso» ni «Vs.»"),
    ("los trabajadores cesados del ayuntamiento", "«trabajadores cesados» en minúsculas"),
    ("caso fortuito", "«caso fortuito»"), ("hacer caso omiso", "«caso omiso»"),
    ("en un caso penal federal", "«caso penal» (Penal Miguel Castro Castro)"),
    ("en el caso concreto, un caso grande", "«caso grande» en minúsculas (Grande Vs. Argentina)"),
    ("Ruiz- Mateos c. España, 23 de junio de 1993, Serie A N° 262, párr. 30", "Ruiz-Mateos (TEDH) no es Ruiz Fuentes"),
    ("Golder c. Reino Unido, 21 de febrero de 1975, Serie A No. 18, párr. 36", "Serie A del TEDH no es la OC-18"),
    ("CASO: ÓRDENES DE PROTECCIÓN URGENTES", "un encabezado en mayúsculas («ÓRDENES» ≠ Órdenes Guerra)"),
    ("el artículo 14, fracción V, de la ley de Chile", "«fracción V» no es «v.»"),
    ("Campo algodonero de la región lagunera", "«Campo algodonero» sin mayúscula en «algodonero» ni pista"),
    # Las de abajo se disparaban antes de la revisión del 25-sep-2026.
    ("¿Cuál es el plazo para contestar la demanda en el caso de Hidalgo?",
     "«en el caso de Hidalgo» (el estado; Hidalgo y otros Vs. Ecuador)"),
    ("en el caso de canales de riego", "«el caso de canales» (Canales Huapaya) en minúsculas"),
    ("el caso de Ochoa en el juzgado tercero", "«el caso de Ochoa» (un cliente; Digna Ochoa)"),
    ("¿Cómo se regula el voto en Hidalgo?", "«el voto en Hidalgo»: derecho electoral, no un voto judicial"),
    ("el voto en Guerrero para la elección de gobernador", "«el voto en Guerrero»"),
    ("Ley de Hacienda de Hidalgo, párrafo 4", "«… de Hidalgo, párrafo 4»: el alias es complemento"),
    ("el Código Civil de Hidalgo, párrafo 3 del artículo 20", "«párrafo 3 del artículo 20» no es pista de caso"),
    ("la empresa Hermanos Gómez S.A. de C.V. demandó el pago", "«Hermanos Gómez» (prefijo de Gómez Paquiyauri)"),
    ("La Comunidad Campesina de San Pedro promovió amparo agrario", "«Comunidad Campesina» (prefijo de Santa Bárbara)"),
    ("la orden de compra OC-15 fue cancelada por el proveedor", "«OC-15» sin año ni Corte IDH: orden de compra"),
    ("el capital social se integra con acciones Serie C 500", "«acciones Serie C 500» sin «No.» ni Corte IDH"),
]:
    ok(nada(t), f"{que}: nada")
r = uno("¿Qué dijo la Corte IDH en el caso de Hidalgo y otros?")
ok(r and r["doc_id"] == "C-534", "…pero «el caso de Hidalgo y otros» con la Corte IDH al lado sí es el caso")
r = uno("la sentencia de Almonacid, párr. 124")
ok(r and r["doc_id"] == "C-154" and r["parrafos"] == [124], "…y «la sentencia de Almonacid, párr. 124» también")
r = uno("Corte IDH, OC-21, párr. 31")
ok(r and r["doc_id"] == "A-21" and r["parrafos"] == [31], "…y «OC-21» sin año, con la Corte IDH al lado")
rs = K.resolver_citas_coidh("JUANA ANGEL HERNANDEZ CORZO\n\nCorte IDH, Caso Familia Pacheco Tineo Vs. Bolivia, "
                            "sentencia de 25 de noviembre de 2013, Serie C No. 272, párrs. 145 y 154 a 156")
ok([(r["doc_id"], r["parrafos"]) for r in rs] == [("C-272", [145, 154, 155, 156])],
   "una firma en mayúsculas junto a una cita real: sólo Pacheco Tineo, con sus párrafos")
rs = K.resolver_citas_coidh("Corte IDH. Asunto de Viviana Gallardo y otras. Serie A No. 101.")
ok([r["doc_id"] for r in rs] == ["A-101"], "Viviana Gallardo es A-101, no un caso «Gallardo»")
rs = K.resolver_citas_coidh("Caso Maldonado Vargas y otros Vs. Chile, párr. 123")
ok([r["doc_id"] for r in rs] == ["C-300"], "«Maldonado Vargas» (el listado dice «Omar Humberto…») → C-300, no Vargas Areco")

print("\n6 · PREGUNTAS DE SEGUIMIENTO (revisión B.7)")
r = K.resolver_citas_coidh("¿y el párrafo 125?", previo="C-154|s|124")
ok(len(r) == 1 and r[0]["doc_id"] == "C-154" and r[0]["parrafos"] == [125] and r[0]["via"] == "heredado",
   "«¿y el párrafo 125?» tras Almonacid ¶124 → C-154 ¶125, heredado")
r = K.resolver_citas_coidh("¿y el párrafo 13?", previo="C-158|v:garcia-ramirez|12")
ok(len(r) == 1 and r[0]["seg"] == "voto" and r[0]["voto_autor"] == "garcia-ramirez" and r[0]["parrafos"] == [13],
   "tras un voto, el párrafo siguiente es del mismo voto")
r = K.resolver_citas_coidh("¿y el caso Radilla, párr. 338?", previo="C-154|s|124")
ok([x["doc_id"] for x in r] == ["C-209"], "si la pregunta nombra otro caso, manda el que nombra")
ok(K.resolver_citas_coidh("¿y el párrafo 125?") == [], "sin cita previa, un párrafo suelto no es nada")
ok(K.resolver_citas_coidh("¿qué dice el párrafo tercero del artículo 1o?", previo="C-154|s|124") == [],
   "el párrafo de un artículo no se hereda al caso")
ok(K.resolver_citas_coidh("¿y qué dice el artículo 14, párrafo 2?", previo="C-154|s|124") == [],
   "«el artículo 14, párrafo 2» (el artículo va ANTES del párrafo) tampoco")
ok(K.resolver_citas_coidh("dame el párrafo 5 del escrito", previo="C-154|s|124") == [],
   "ni «el párrafo 5 del escrito»")

print("\n7 · CUESTA POCO")
largo = ("En el juicio de amparo indirecto el quejoso reclama la orden de aprehensión. " * 400)[:30000]
t0 = time.perf_counter()
K.resolver_citas_coidh(largo + " Caso Radilla Pacheco Vs. México, párr. 338.")
ms = 1000 * (time.perf_counter() - t0)
ok(ms < 150, f"30,000 caracteres en {ms:.0f} ms (el plan pide < 150 ms añadidos)")

print("\n" + ("TODO PASA" if not FALLOS else f"{len(FALLOS)} FALLAN: {FALLOS}"))
raise SystemExit(1 if FALLOS else 0)
