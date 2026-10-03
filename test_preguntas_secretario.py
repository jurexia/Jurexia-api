# -*- coding: utf-8 -*-
"""Las preguntas al secretario (2-oct-2026, bandera «preguntas_al_secretario»):
indispensable por CÓDIGO, topes, ids estables, el bloque de respuestas y el
detector de frases de herramienta. Puro: sin modelo, sin red, sin base.

    .venv/bin/python test_preguntas_secretario.py
"""
import sys
sys.path.insert(0, ".")
import contexto_taller as ct
import preguntas_secretario as ps

FALLOS = []


def ok(c, q):
    print(f"   {'PASA ' if c else 'FALLA'}  {q}")
    if not c:
        FALLOS.append(q)


def prem(i, problema=1, si_si="fundado", si_no="infundado", el_acto="no_se_pronuncia", **kw):
    d = {"id": f"F{i}", "segmento": f"C1.{chr(96 + i)}", "problema": problema,
         "afirma_la_parte": f"la parte afirma el hecho número {i}",
         "cita_escrito": "", "el_acto": el_acto, "carga_de": "la_parte",
         "pregunta": f"¿Ocurrió el hecho número {i} en el juicio de origen?", "tipo": "si_no",
         "si_si": si_si, "si_no": si_no, "para_que": f"decide el punto {i}"}
    d.update(kw)
    return d


PROBS = [{"pregunta": "¿Fue legal el emplazamiento?", "jerarquia": "principal"},
         {"pregunta": "¿Procedía la condena en costas?", "jerarquia": "accesorio"}]

print("\n1 · INDISPENSABLE LO DECIDE EL CÓDIGO")
A = {"premisas": [
    prem(1),                                              # principal, cambia el desenlace, acto mudo
    prem(2, problema=2),                                  # accesorio: no detiene
    prem(3, si_si="infundado", si_no="inoperante"),       # la respuesta no cambia el desenlace
    prem(4, el_acto="lo_tuvo_por_cierto"),                # el acto ya lo dijo: no se pregunta
    prem(5, pregunta=""),                                 # sin pregunta: nada
]}
P = ps.preguntas_de(A, PROBS)
por = {p["premisa"]: p for p in P}
ok(por["F1"]["indispensable"], "principal + la respuesta cambia el desenlace + acto mudo = indispensable")
ok(not por["F2"]["indispensable"], "la de un accesorio no detiene la propuesta")
ok(not por["F3"]["indispensable"], "si «sí» y «no» llevan al mismo desenlace (infundado/inoperante), no detiene")
ok("F4" not in por and "F5" not in por, "lo que el acto ya dijo no se pregunta; sin pregunta, nada")
ok(all(p["el_acto"] == "no_se_pronuncia" and p["respuesta"] is None for p in P), "forma del contrato B")
ok(set(P[0]) >= {"id", "pregunta", "tipo", "para_que", "problema", "indispensable", "afirma_la_parte",
                 "cita_escrito", "el_acto", "si_si", "si_no", "si_no_contesta", "respuesta"},
   "todas las claves del contrato B")
ok(P[0]["pregunta"].startswith("¿") and P[0]["pregunta"].endswith("?") and len(P[0]["pregunta"]) <= 220,
   "pregunta cerrada, con sus signos y dentro del tope")
ok("carga" in P[0]["si_no_contesta"] and "infundado" in P[0]["si_no_contesta"],
   "sin respuesta, decide la carga de la prueba (le toca a la parte: lo de «no»)")
ok("contraparte" in ps.preguntas_de({"premisas": [prem(1, carga_de="contraparte")]}, PROBS)[0]["si_no_contesta"],
   "si la carga es de la contraparte, se dice")
_dec = ps.preguntas_de(A, PROBS, decide=2)
ok({p["premisa"] for p in _dec if p["indispensable"]} == {"F1", "F2"},
   "el problema que decide el asunto también cuenta (decide=2)")
ok(ps.preguntas_de({"premisas": [prem(1)]}, [{"pregunta": "x"}, {"pregunta": "y"}])[0]["indispensable"],
   "sin jerarquía, el principal es el primero (la regla del árbol)")
_seg = {"premisas": [prem(1, problema=0, segmento="C2.a")],
        "argumentos": [{"segmento": "C2.a", "combate": ["R2"]}],
        "razones": [{"id": "R2", "problemas": [1]}]}
ok(ps.preguntas_de(_seg, PROBS)[0]["problema"] == 1 and ps.preguntas_de(_seg, PROBS)[0]["indispensable"],
   "sin problema declarado, el de las razones que combate su segmento")
ok(ps.preguntas_de(None, PROBS) == [] and ps.preguntas_de({}, PROBS) == [], "sin análisis, ninguna")

print("\n2 · LOS TOPES: 3 INDISPENSABLES, 5 EN TOTAL")
B = {"premisas": [prem(i) for i in range(1, 6)] + [prem(i, problema=2) for i in range(6, 10)]}
PB = ps.preguntas_de(B, PROBS)
ok(len(PB) == 5 and sum(p["indispensable"] for p in PB) == 3, "como mucho 3 indispensables y 5 en total")
ok([p["indispensable"] for p in PB] == [True, True, True, False, False], "las indispensables primero")

print("\n3 · IDS ESTABLES")
P2 = ps.preguntas_de({"premisas": list(reversed(A["premisas"]))}, PROBS)
ok({p["premisa"]: p["id"] for p in P2} == {p["premisa"]: p["id"] for p in P},
   "la misma premisa conserva su id aunque cambie el orden de la lista")
ok(ps.id_de(prem(1)) != ps.id_de(prem(2)) and ps.id_de(prem(1)) == ps.id_de(dict(prem(1), id="F9")),
   "el id sale de lo que se pregunta, no del número que le puso el modelo")
ok(all(p["id"].startswith("P") for p in P), "ids «P…»")

print("\n4 · LAS RESPUESTAS")
ok([ps.normalizar_respuesta(x) for x in ("Sí", "si", True, "NO", False, "No consta", "", None)]
   == ["si", "si", "si", "no", "no", "no_consta", None, None], "sí / no / no consta / vacío")
ok(ps.normalizar_respuesta("La cláusula sexta dice que el pago es mensual", "texto")
   == "La cláusula sexta dice que el pago es mensual", "el texto literal se conserva")
_id1 = por["F1"]["id"]
ok(ps.pendientes(P) and ps.pendientes(P)[0]["id"] == _id1, "pendiente la indispensable sin contestar")
ok(not ps.pendientes(P, {_id1: "no"}), "contestada (aunque sea «no»), ya no detiene")
ok(not ps.pendientes(ps.preguntas_de(A, PROBS, {_id1: "no_consta"})), "«no consta» también es una respuesta")
_con = ps.preguntas_de(A, PROBS, {_id1: "si"})
ok(next(p for p in _con if p["id"] == _id1)["respuesta"] == "si", "la respuesta viaja con su pregunta")
_an = ps.con_respuestas(A, _con)
ok(next(x for x in _an["premisas"] if x["id"] == "F1")["respuesta"] == "si"
   and "respuesta" not in A["premisas"][0], "con_respuestas pone la respuesta en su premisa SIN tocar el original")

print("\n5 · LA MARCA DE RESPUESTAS")
_m, _ign = ps.marca_respuestas("H1", [], [{"id": _id1, "respuesta": "Sí"}, {"id": "Pxxxxxx", "respuesta": "no"}], P, 5.0)
ok(_m["huella"] == "H1" and len(_m["items"]) == 1 and _m["items"][0]["respuesta"] == "si"
   and _m["items"][0]["pregunta"] == por["F1"]["pregunta"] and _ign == ["Pxxxxxx"],
   "se guarda con el texto de la pregunta; un id que no existe se ignora")
_m2, _ = ps.marca_respuestas("H1", _m["items"], [{"id": _id1, "respuesta": ""}], P)
ok(_m2["items"] == [] and _m2["firma"] == "", "una respuesta vacía borra la anterior")
ok(ps.items_de_marca(_m, "H1") and ps.items_de_marca(_m, "H2") == [],
   "las respuestas de otro adelanto no se aplican")
ok(ps.firma_de_marca(_m, "H1") and ps.firma_de_marca(_m, "H1") != ps.firma_de_marca(
    ps.marca_respuestas("H1", [], [{"id": _id1, "respuesta": "no"}], P)[0], "H1"),
   "la firma cambia con la respuesta (entra en la clave de la propuesta guardada)")
ok(ps.firma([]) == "" and ps.firma(None) == "", "sin respuestas, firma vacía")

print("\n6 · EL BLOQUE «RESPUESTAS DEL SECRETARIO»")
b = ps.bloque_respuestas(_m["items"])
ok(b.startswith(ps.CABECERA_BLOQUE) and por["F1"]["pregunta"] in b and "Respuesta: SÍ" in b
   and ps.FIN_BLOQUE in b, "pregunta y respuesta, entre su cabecera y su cierre")
ok("NO ACREDITADO" in ps.bloque_respuestas([{"pregunta": "¿X?", "respuesta": "no_consta"}]),
   "«no consta» se lee como no acreditado")
ok(ps.bloque_respuestas(P) == "" and ps.bloque_respuestas([]) == "", "sin respuestas, nada")
_ctx = "ANTES\n" + b + "CONSTANCIAS QUE OBRAN EN AUTOS: largas…"
_bl, _resto = ps.separar_bloque(_ctx)
ok(_bl.strip() == b.strip() and ps.CABECERA_BLOQUE not in _resto and "CONSTANCIAS" in _resto,
   "el bloque se separa entero del contexto (para que el recorte no lo toque)")
ok(ps.separar_bloque("sin bloque") == ("", "sin bloque"), "sin bloque, el contexto tal cual")

print("\n7 · LA RESPUESTA «PREGUNTAS» (contrato A)")
R = ps.respuesta_preguntas("1/2026", P, modelo="m", origen_firma="o", firma_respuestas="")
ok(R["estado"] == "preguntas" and R["formato"] == 3 and R["propuestas"] == [] and R["global"] is None
   and R["criterios_json"] == "[]" and R["pendientes"] == 1 and R["preguntas"] == P
   and R["expediente"] == "1/2026" and isinstance(R["avisos"], list),
   "estado, formato 3, sin propuestas ni global, criterios vacíos y las preguntas")

print("\n8 · LAS FRASES DE HERRAMIENTA")
MALAS = ["La cláusula no obra en el material proporcionado.",
         "De las constancias remitidas para este estudio no se advierte la firma.",
         "Ese dato no consta en los insumos.",
         "Según lo aportado, la notificación fue personal.",
         "conforme al expediente proporcionado, la demanda se presentó tarde",
         "la documental aportada por el secretario acredita el pago",
         "según informa el secretario, el emplazamiento se practicó con la hermana"]
BUENAS = ["La información proporcionada por la autoridad fiscal no desvirtúa la presunción.",
          "Lo aportado por la quejosa como prueba no acredita la relación laboral.",
          "Los insumos de producción adquiridos por la contribuyente son deducibles.",
          "Las constancias remitidas por la responsable acreditan el emplazamiento.",
          "El material probatorio que obra en autos es insuficiente.",
          "De la resolución recurrida no se advierte que la parte demostrara el pago."]
_det = [ps.frases_de_herramienta(x) for x in MALAS]
ok(all(_det), f"detecta la voz de herramienta en las siete ({[bool(x) for x in _det]})")
_fp = [x for x in BUENAS if ps.frases_de_herramienta(x)]
ok(not _fp, f"calibrado: no acusa a una sentencia buena ({_fp})")
ok(ps.frases_de_herramienta("no obra en el material. No obra en el material.") == ["no obra en el material"],
   "sin repetir")

print("\n9 · LA BANDERA")
ct.poner(False)
ok(not ps.activa(), "para los de fuera, apagada (omisión «casa»)")
ct.poner(True, {"banderas": {"preguntas_al_secretario": True}}, pruebas=True)
ok(ps.activa(), "se enciende por petición en una cuenta de casa")
ct.poner(False)

print()
if FALLOS:
    print(f"FALLAN {len(FALLOS)}: " + " · ".join(FALLOS))
    sys.exit(1)
print("RESULTADO: TODAS LAS COMPROBACIONES PASAN")
