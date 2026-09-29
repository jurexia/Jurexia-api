# -*- coding: utf-8 -*-
"""El análisis neutral de la litis (rediseño, etapa 2): la verificación sin modelo y el cableado.

    .venv/bin/python test_analisis_litis.py
"""
import inspect, sys, types
sys.path.insert(0, ".")
import analisis_litis as al

FALLOS = []


def ok(c, q):
    print(f"   {'PASA ' if c else 'FALLA'}  {q}")
    if not c:
        FALLOS.append(q)


ACTO = ("La Sala determinó que el tercero adquirente del inmueble se sustituyó válidamente en la "
        "ejecución porque adquirió el bien durante el procedimiento y conocía el arrendamiento "
        "celebrado con la actora. Además, estimó que las prestaciones a ejecutar no cambiaron con "
        "la sustitución de la parte actora en el juicio de origen.")
ESCRITO = ("La recurrente sostiene que la sustitución procesal es ilegal porque el adquirente no "
           "acreditó la cesión del derecho contractual concreto que se ejecuta en el juicio.")
SEGS = [{"id": "A1.a", "texto": "la sustitución es ilegal"}, {"id": "A1.b", "texto": "otro"}]
CRUDO = {
    "cuestion_central": "Si el adquirente del inmueble adquirió el derecho contractual concreto que se ejecuta.",
    "razones": [
        {"id": "R1", "afirma": "se sustituyó válidamente", "conclusion": "procede la sustitución",
         "relacion": "conjunta", "con": ["R2", "R9"], "problemas": [1],
         "cita": "el tercero adquirente del inmueble se sustituyó válidamente en la ejecución porque adquirió el bien"},
        {"id": "R2", "afirma": "las prestaciones no cambiaron", "conclusion": "la sustitución no afecta",
         "relacion": "autonoma", "problemas": [1],
         "cita": "las prestaciones a ejecutar no cambiaron con la sustitución de la parte actora"},
        {"id": "R3", "afirma": "inventada", "conclusion": "x", "relacion": "raro",
         "cita": "esta frase no aparece en ninguna parte del acto reclamado ni de la sentencia"}],
    "argumentos": [{"segmento": "A1.a", "pide": "revocar", "por_que": "falta cesión", "combate": ["R1", "R77"]},
                   {"segmento": "Z9", "pide": "x", "por_que": "y", "combate": ["R2"]}],
    "hechos": [
        {"id": "H1", "que": "conocía el arrendamiento", "afirma": "a_quo", "fuente": "acto",
         "cita": "adquirió el bien durante el procedimiento y conocía el arrendamiento celebrado con la actora",
         "condicion": "tenido_por_acreditado"},
        {"id": "H2", "que": "la cesión del derecho", "afirma": "recurrente", "fuente": "escrito",
         "cita": "", "condicion": "falta_en_insumos"},
        {"id": "H3", "que": "algo sin cita", "afirma": "a_quo", "fuente": "acto",
         "cita": "una cita que no está en el acto de ninguna manera posible hoy", "condicion": "tenido_por_acreditado"}],
    "faltantes": [{"que": "el contrato de cesión", "por_que_importa": "decide la legitimación"}],
}

print("\n1 · LA VERIFICACIÓN SIN MODELO")
d = al.verificar(CRUDO, ACTO, ESCRITO, "", SEGS)
R = {r["id"]: r for r in d["razones"]}
ok(R["R1"]["verificada"] and R["R2"]["verificada"] and not R["R3"]["verificada"],
   "las citas de la recurrida se comprueban palabra por palabra; la inventada no pasa")
ok(R["R1"]["con"] == ["R2"] and R["R3"]["relacion"] == "conjunta",
   "las referencias a razones que no existen se quitan; una relación fuera del catálogo NO se da por autónoma")
ok([a["segmento"] for a in d["argumentos"]] == ["A1.a"] and d["argumentos"][0]["combate"] == ["R1"],
   "sólo segmentos del inventario, y sólo razones que existen")
H = {h["id"]: h for h in d["hechos"]}
ok(H["H1"]["condicion"] == "tenido_por_acreditado" and H["H1"]["verificada"],
   "«tenido por acreditado» con su cita verificada se conserva")
ok(H["H2"]["condicion"] == "falta_en_insumos" and H["H2"]["cita"] == "",
   "«falta en los insumos» no lleva cita (no es lo mismo que «no acreditado»)")
ok(H["H3"]["condicion"] == "sin_verificar",
   "una condición sin cita verificada no se da por buena: baja a «sin_verificar»")
ok(d["autonomas_sin_combatir"] == ["R2"],
   "las razones AUTÓNOMAS que ningún argumento combate, calculadas por código (punto 5)")
ok(d["citas_sin_verificar"] == 2, "y se cuentan las citas que no se hallaron")

print("\n1b · LO QUE EL MODELO MANDA MAL FORMADO (revisión adversarial)")
_raro = {"razones": [
            {"id": "r1", "afirma": "a", "relacion": "Autónoma", "problemas": 1, "con": "R2",
             "cita": "el tercero adquirente del inmueble se sustituyó válidamente en la ejecución porque adquirió el bien"},
            {"id": "R2", "afirma": "b", "relacion": "dependiente de R1", "problemas": ["²", "2"], "con": 7},
            {"afirma": "sin id", "relacion": "conjunta con R1"},
            {"id": "R3", "afirma": "autónoma atacada sólo en la segunda mención del segmento", "relacion": "autonoma"}],
         "argumentos": [{"segmento": "a1.A", "combate": "R1"},
                        {"segmento": "A1.a", "combate": ["R3"]}],
         "hechos": [{"id": "H1", "que": "x", "fuente": "escrito de agravios", "condicion": "tenido_por_acreditado",
                     "cita": "el adquirente no acreditó la cesión del derecho contractual concreto que se ejecuta"},
                    {"id": "H2", "que": "y", "fuente": "acto", "condicion": "falta_en_insumos",
                     "cita": "las prestaciones a ejecutar no cambiaron con la sustitución de la parte actora"},
                    {"id": "H3", "que": "z", "fuente": "constancias", "condicion": "tenido_por_acreditado",
                     "cita": "las prestaciones a ejecutar no cambiaron con la sustitución de la parte actora"}]}
try:
    _dr = al.verificar(_raro, ACTO, ESCRITO, "", SEGS)
    _crash = None
except Exception as _e:
    _dr, _crash = None, _e
ok(_crash is None, f"valores sueltos en vez de listas no rompen el análisis ({_crash!r})")
_R = {r["id"]: r for r in (_dr or {}).get("razones", [])}
ok(_R.get("R1", {}).get("relacion") == "autonoma" and _R.get("R2", {}).get("relacion") == "dependiente"
   and _R.get("R4", {}).get("relacion") == "conjunta" and _R.get("R1", {}).get("problemas") == [1]
   and _R.get("R2", {}).get("problemas") == [2] and _R.get("R1", {}).get("con") == ["R2"],
   "«Autónoma», «dependiente de R1», problemas sueltos y «con» como texto se leen bien; la razón sin id no choca")
ok((_dr or {}).get("autonomas_sin_combatir") == [] and len((_dr or {}).get("argumentos", [])) == 1
   and sorted(_dr["argumentos"][0]["combate"]) == ["R1", "R3"],
   "el segmento en otra caja y repetido SUMA sus ataques: ninguna autónoma queda falsamente sin combatir")
_H = {h["id"]: h for h in (_dr or {}).get("hechos", [])}
ok(_H.get("H1", {}).get("fuente") == "escrito" and _H.get("H1", {}).get("verificada"),
   "«escrito de agravios» es el escrito: la cita se comprueba ahí")
ok(_H.get("H2", {}).get("condicion") == "sin_verificar",
   "«falta en insumos» con una cita que SÍ está en los insumos se contradice: queda sin verificar")
ok(_H.get("H3", {}).get("fuente") == "acto",
   "sin constancias, la cita hallada en el acto dice que está en el acto, no en las constancias")

print("\n2 · LO QUE RECIBE LA PROPUESTA")
b = al.bloque_propuesta(d)
ok("CUESTIÓN CENTRAL" in b and "derecho contractual concreto" in b, "la cuestión central, concreta")
ok("R2 ← NINGUNO" in b and "RAZONES AUTÓNOMAS QUE NINGÚN ARGUMENTO COMBATE: R2." in b,
   "qué argumento ataca cada razón y el aviso de la autónoma sin combatir")
ok("NO CONSTA en los insumos" in b and "TUVO POR ACREDITADO" in b and "SIN VERIFICAR" in b,
   "cada hecho con su condición, dicha en palabras")
ok("fundado" not in b.lower().replace("infundado", ""), "el análisis no califica")
ok(al.bloque_propuesta(None) == "" and al.bloque_propuesta({}) == "", "sin análisis, nada")

print("\n3 · EL PROMPT NO LLEVA ACERVO NI PRECEDENTES")
p = al.prompt(ACTO, ESCRITO, "", SEGS, [{"pregunta": "¿Procede la sustitución?"}], "FICHA", True, "amparo_revision")
ok("A1.a" in p and "¿Procede la sustitución?" in p and "FICHA" in p, "recibe inventario, planteamientos y ficha")
ok(not any(w in p.lower() for w in ("oaj", "espejo", "precedente del propio", "tesis del acervo", "registro digital")),
   "ni la OAJ ni el acervo: el análisis es de ESTE expediente")

print("\n4 · LA HUELLA")
r1 = types.SimpleNamespace(fases=types.SimpleNamespace(fuentes=[ACTO, ESCRITO], autos="",
                                                       problemas=[{"pregunta": "p", "combate": "c", "resolvio": "r"}]))
h1 = al.huella(r1)
r1.fases.problemas[0]["pregunta"] = "otra"
ok(al.huella(r1) != h1, "cambia si cambia un planteamiento: el análisis viejo no se reutiliza")

print("\n5 · EL CABLEADO")
src = open("main.py", encoding="utf-8").read()
import ast
FN = {n.name: n for n in ast.walk(ast.parse(src)) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
pre = ast.get_source_segment(src, FN["_taller_preanalizar"])
ok('rediseno("analisis_neutral")' in pre and pre.index('rediseno("analisis_neutral")') < pre.index("_taller_guardar_marca"),
   "sin la bandera no escribe nada ni gasta la llamada")
esp = ast.get_source_segment(src, FN["_taller_esperar_analisis"])
ok("tope=ANALISIS_ESPERA_S" in esp and 'huella_analisis' in esp,
   "la propuesta lo espera con tope y sólo si es de esta versión del adelanto")
nuc = ast.get_source_segment(src, FN["_taller_proponer_nucleo"])
ok("analisis=_analisis_p" in nuc, "y lo pasa a la propuesta")
import fase5_propuesta as f5
ok("analisis" in inspect.signature(f5.proponer).parameters
   and "_bloques_rediseno(analisis, requisitos, material)" in inspect.getsource(f5.proponer)
   and "bloque_propuesta(analisis)" in inspect.getsource(f5._bloques_rediseno)
   and f5._bloques_rediseno({"razones": [{"id": "R1", "relacion": 5}]}, {"requisitos": [7]}, None) is not None,
   "la propuesta lo recibe como datos junto al contraste")
import contexto_taller as ct
ct.poner(False, {})
ok(not ct.rediseno("analisis_neutral"), "para los de fuera, apagado")

print("\n6 · LA CITA CON ERRORES DE LECTURA (103/2025: el acto escaneado)")
import plan_estudio as _pe
_src = ("de los preceptos legales citados, aplicables al caso que se analiza, se derivan lidselementos constitutivos "
        "desus pretensiones consistentes erario 1) que en autos se acredite. Sacramento Olvera Corral le habia "
        "informad quezla senora pilar felipaz huefta venancio era su dependiente economica (respuestaja la pregunta "
        "dos). El tribunal considera que la testimonial es eficaz para acreditar el concubinato")
_T = _pe.Texto(_src)
ok(al.casi_literal("De los preceptos legales citados, aplicables al caso que se analiza, se derivan los elementos "
                   "constitutivos de sus pretensiones consistentes en", _T)
   and al.casi_literal("Sacramento Olvera Corral le había informado que la señora Pilar Felipa Huerta Venancio era su "
                       "dependiente económica", _T),
   "la cita limpia de un texto escaneado se da por verificada")
ok(not al.casi_literal("El tribunal considera que la testimonial no es eficaz para acreditar el concubinato", _T)
   and not al.casi_literal("El tribunal considera que la testimonial es ineficaz para acreditar el concubinato", _T)
   and not al.casi_literal("los preceptos legales citados no son aplicables al caso y no se derivan los elementos "
                           "constitutivos de la pretensión", _T),
   "la que añade una negación, un prefijo o palabras enteras, no")
_src2 = ("Por lo expuesto el demandado no probó sus excepciones y el agravio resulta infundado porque el acto es "
         "legal y la conducta es atípica conforme al artículo 150000 del año 2024, y se condena al pago de cuando "
         "menos dos pensiones a la parte actora o a su representante legal en el juicio")
_T2 = _pe.Texto(_src2)
ok(not any(al.casi_literal(q, _T2) for q in (
       "Por lo expuesto el demandado probó sus excepciones y el agravio resulta infundado porque el acto es legal",
       "el demandado no probó sus excepciones y el agravio resulta fundado porque el acto es legal y la conducta",
       "el agravio resulta infundado porque el acto es ilegal y la conducta es atípica conforme al artículo",
       "la conducta es atípica conforme al artículo 150000 del año 2025 y se condena al pago",
       "se condena al pago de cuando más dos pensiones a la parte actora o a su representante",
       "se condena al pago de cuando menos dos pensiones a la parte actora y a su representante")),
   "ni la que QUITA una negación, un prefijo o cambia una cifra o una palabra corta (revisión adversarial)")
_v = al.verificar({"razones": [{"id": "R1", "afirma": "x", "relacion": "autonoma",
                                "cita": "Sacramento Olvera Corral le había informado que la señora Pilar Felipa "
                                        "Huerta Venancio era su dependiente económica"}]}, _src, "", "", [])
ok(_v["razones"][0]["verificada"] and _v["razones"][0].get("lectura") == "ocr"
   and "errores de lectura" in al.bloque_propuesta(_v),
   "y queda rotulada: la propuesta sabe que se cotejó con un escaneo")

print("\n7 · EL ACTO Y EL ESCRITO, ENTEROS (diagnóstico del sesgo a conceder)")
_largo = "ANTECEDENTES " + "x" * 90000 + " CONSIDERANDO ESTUDIO " + "y" * 30000 + " RESOLUTIVOS: se niega. Conste."
ok(al.recorte(_largo, al.MAX_ACTO) == (_largo, 0) and al.MAX_ACTO >= 193385 and al.MAX_ESCRITO >= 125897,
   "un acto de 120 mil (y el más largo del banco, 193 mil) va entero: antes se cortaba en 60 mil por el principio")
_r, _om = al.recorte(_largo, 40000)
ok(_om == len(_largo) - 40000 and _r.startswith("ANTECEDENTES") and _r.endswith("se niega. Conste.")
   and f"SE OMITEN {_om} CARACTERES" in _r,
   "si pasa el tope, van el principio y el FINAL (estudio y resolutivos) con la omisión marcada")
_p = al.prompt(_largo, "concepto único", "", [], [], "", False)
ok("RESOLUTIVOS: se niega. Conste." in _p and "ÍNTEGRO: " in _p and "no pidas su «texto completo»" in _p,
   "el prompt lleva el acto hasta sus resolutivos, dice que va íntegro y prohíbe pedirlo")
ok(al.VERSION == "analisis-3", "la versión cambia: las marcas del análisis anterior se recalculan")
_F = [{"que": "El texto completo de la sentencia reclamada, incluidos los puntos resolutivos."},
      {"que": "Los resolutivos y el texto íntegro de la sentencia reclamada."},
      {"que": "La parte de la sentencia reclamada posterior al fragmento entregado."},
      {"que": "La continuación íntegra de la resolución reclamada."},
      {"que": "La sentencia completa de primera instancia, más allá de sus puntos resolutivos reproducidos en "
              "la resolución reclamada."},
      {"que": "La totalidad del expediente de primera instancia, pues sólo se entregó el acto reclamado."},
      {"que": "La parte restante de la resolución reclamada y las constancias completas del expediente agrario."},
      {"que": "El texto completo del artículo 40 de la Ley de Servicios Auxiliares."},
      {"que": "El texto completo de la demanda de amparo."}]
_q, _fuera = al.depurar_faltantes(_F, "acto entregado", "escrito entregado")
ok(len(_fuera) == 5 and [x["que"][:14] for x in _q] == ["La sentencia c", "La totalidad d", "La parte resta",
                                                         "El texto compl"],
   "con el acto y el escrito íntegros se quitan los faltantes que los piden a ellos (calibrado: 18 de 128 en el "
   "banco, ningún otro); se quedan la sentencia de primera instancia, el expediente y lo que pide algo más")
_q2, _fuera2 = al.depurar_faltantes(_F, "a" * (al.MAX_ACTO + 1), "")
ok(not _fuera2 and len(_q2) == len(_F), "recortado o sin entregar, el faltante puede ser cierto y se queda")
_v7 = al.verificar({"faltantes": _F[:2]}, "acto entregado", "", "", [])
ok(_v7["faltantes"] == [] and len(_v7["faltantes_depurados"]) == 2
   and "texto completo de la sentencia" not in al.bloque_propuesta(_v7),
   "verificar los depura y la propuesta ya no los lee como una omisión de la Sala")

print()
if FALLOS:
    print("FALLAS:\n  " + "\n  ".join(FALLOS))
    sys.exit(1)
print("RESULTADO: TODAS LAS COMPROBACIONES PASAN")
