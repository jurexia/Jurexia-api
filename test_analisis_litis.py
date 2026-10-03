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
_ACTO = "… se niega el amparo. RESUELVE: ÚNICO. Notifíquese. Así lo resolvió el Tribunal. Conste."
_ESCR = "… por lo expuesto, PIDO se conceda el amparo. PROTESTO LO NECESARIO."
_F = [{"que": "El texto completo de la sentencia reclamada, incluidos los puntos resolutivos."},
      {"que": "Los resolutivos y el texto íntegro de la sentencia reclamada."},
      {"que": "La parte de la sentencia reclamada posterior al fragmento entregado."},
      {"que": "Copia íntegra de la sentencia reclamada."},
      {"que": "El texto íntegro de la sentencia reclamada, especialmente el apartado séptimo sobre legitimación, "
              "escisión y costas."},
      {"que": "La sentencia completa de primera instancia, más allá de sus puntos resolutivos reproducidos en "
              "la resolución reclamada."},
      {"que": "La totalidad del expediente de primera instancia, pues sólo se entregó el acto reclamado."},
      {"que": "La parte restante de la resolución reclamada y las constancias completas del expediente agrario."},
      {"que": "El resto de la sentencia reclamada junto con la demanda natural."},
      {"que": "El texto completo del artículo 40 de la Ley de Servicios Auxiliares."},
      {"que": "El texto completo de la demanda de amparo."}]
_q, _fuera = al.depurar_faltantes(_F, _ACTO, _ESCR, es_recurso=False)
ok(len(_fuera) == 6 and [x["que"][:14] for x in _q] == ["La sentencia c", "La totalidad d", "La parte resta",
                                                         "El resto de la", "El texto compl"],
   "amparo directo, acto y escrito íntegros: se quitan los faltantes que piden el acto o la demanda (calibrado: "
   "16 de 128 en el banco); se quedan la sentencia de primera instancia, el expediente y lo que pide algo más")
_qr, _fr = al.depurar_faltantes(_F + [{"que": "El texto completo de la sentencia recurrida."},
                                      {"que": "El texto íntegro del recurso de revisión."}], _ACTO, _ESCR, es_recurso=True)
ok(_fr == ["El texto completo de la sentencia recurrida.", "El texto íntegro del recurso de revisión."],
   "en un recurso, el acto es la sentencia recurrida y el escrito el recurso: «la demanda de amparo» y «la "
   "sentencia reclamada» son documentos que no se subieron y se quedan")
ok(al.depurar_faltantes(_F, _ACTO, _ESCR)[1] == [], "sin saber el tipo de asunto, no se quita nada")
_q2, _fuera2 = al.depurar_faltantes(_F, "a" * (al.MAX_ACTO + 1) + " Conste.", "", es_recurso=False)
ok(not _fuera2 and len(_q2) == len(_F), "recortado o sin entregar, el faltante puede ser cierto y se queda")
_incompleto = "La Sala… EN LOS CASOS DONDE EL INSTITUTO DEL FONDO NACIONAL DE LA VIVIENDA 40 i"
ok(not al.parece_completo(_incompleto) and al.parece_completo(_ACTO) and al.parece_completo(_ESCR, escrito=True)
   and not al.depurar_faltantes(_F[:1], _incompleto, _ESCR, es_recurso=False)[1]
   and [l for l in al.prompt(_incompleto, _ESCR, "", [], [], "", False).splitlines()
        if l.startswith("EL ACTO RECLAMADO (")][0].endswith("puede haberse subido incompleto):")
   and [l for l in al.prompt(_ACTO, _ESCR, "", [], [], "", False).splitlines()
        if l.startswith("EL ACTO RECLAMADO (")][0].startswith("EL ACTO RECLAMADO (ÍNTEGRO"),
   "un acto que no cierra como documento completo (el 400/2024: foja 40, sin resolutivos ni firmas) no se "
   "rotula ÍNTEGRO y su faltante cierto se queda")
_v7 = al.verificar({"faltantes": _F[:2]}, _ACTO, "", "", [], es_recurso=False)
ok(_v7["faltantes"] == [] and len(_v7["faltantes_depurados"]) == 2
   and "texto completo de la sentencia" not in al.bloque_propuesta(_v7),
   "verificar los depura y la propuesta ya no los lee como una omisión de la Sala")
ok("no te impide proponer" in al.bloque_propuesta(_v7 | {"razones": [{"id": "R1", "relacion": "autonoma", "afirma": "a",
                                                                       "conclusion": "b", "verificada": True}]}),
   "lo que falta en los insumos baja la confianza pero no impide proponer (humo del AR 631: «alcanza» en falso)")
_pc = al.prompt("a", "b", "x" * 30000 + "FIN", [], [], "", False)
ok("FIN" in _pc and "SE OMITEN" in _pc.split("CONSTANCIAS")[-1],
   "las constancias tampoco se cortan por el final sin decirlo: principio y final, con la omisión marcada")
ok(al.recorte("ABCDEFGHIJ", 0)[0].count("ABCDEFGHIJ") == 0 and al.MAX_TOKENS >= 32000,
   "un tope de cero no duplica el texto, y la salida tiene margen para el peor caso (32 mil)")
import os as _os, importlib as _il
_os.environ["ANALISIS_MAX_ACTO"] = "220k"
_al2 = _il.reload(al)
ok(_al2.MAX_ACTO == 220000, "un tope mal escrito en el entorno vale el de omisión en vez de tumbar el módulo")
del _os.environ["ANALISIS_MAX_ACTO"]
_il.reload(al)

# ═══ 8 · LAS PREMISAS DE LA PARTE (2-oct-2026, bandera «preguntas_al_secretario») ═══
print("\n8 · SIN LA BANDERA, IDÉNTICO A LA BASE (b89e049), LETRA POR LETRA")
import subprocess, importlib.util as _ilu
ct.poner(False, {})
try:
    _src_base = subprocess.run(["git", "show", "b89e049:analisis_litis.py"], capture_output=True,
                               text=True, check=True).stdout
    _spec = _ilu.spec_from_loader("analisis_litis_base", loader=None)
    al0 = _ilu.module_from_spec(_spec)
    exec(compile(_src_base, "analisis_litis_base.py", "exec"), al0.__dict__)
except Exception as _eb:
    al0 = None
    print(f"   (no se pudo leer la base: {_eb})")
if al0 is not None:
    _args = (ACTO, ESCRITO, "constancias", SEGS, [{"pregunta": "¿Procede?"}], "FICHA", True, "amparo_revision")
    ok(al.prompt(*_args) == al0.prompt(*_args), "el prompt, idéntico")
    _CR8 = dict(CRUDO, hechos=CRUDO["hechos"] + [
        {"id": "H4", "que": "lo dice la parte", "afirma": "recurrente", "fuente": "escrito",
         "condicion": "no_controvertido",
         "cita": "el adquirente no acreditó la cesión del derecho contractual concreto que se ejecuta"}],
        premisas=[{"id": "F1", "afirma_la_parte": "x", "pregunta": "¿x?"}])
    ok(al.verificar(_CR8, ACTO, ESCRITO, "", SEGS) == al0.verificar(_CR8, ACTO, ESCRITO, "", SEGS),
       "la verificación, idéntica (sin premisas ni «afirmado_por_la_parte»)")
    _d8 = al.verificar(_CR8, ACTO, ESCRITO, "", SEGS)
    ok(al.bloque_propuesta(_d8) == al0.bloque_propuesta(_d8), "el bloque de la propuesta, idéntico")
    ok(al.huella(r1) == al0.huella(r1), "la huella, idéntica: el análisis guardado sigue valiendo")
    ok(al.verificar_cita("adquirió el bien durante el procedimiento y conocía el arrendamiento",
                         al.textos(ACTO, ESCRITO, ""), "escrito")
       == al0.verificar_cita("adquirió el bien durante el procedimiento y conocía el arrendamiento",
                             al0.textos(ACTO, ESCRITO, ""), "escrito"),
       "verificar_cita sin `solo` busca como siempre (con caída a las demás fuentes)")

print("\n9 · CON LA BANDERA: LAS PREMISAS, VERIFICADAS POR FUENTE")
ct.poner(True, {"banderas": {"preguntas_al_secretario": True}}, pruebas=True)
ok(al.version() == "analisis-5" and al.VERSION == "analisis-3", "otra versión, sólo con la bandera")
ok(al.huella(r1) != al0.huella(r1) if al0 is not None else True, "y la huella lo dice: se recalcula")
_p9 = al.prompt(*(ACTO, ESCRITO, "", SEGS, [{"pregunta": "¿Procede?"}], "FICHA", True, "amparo_revision"))
ok("premisas:" in _p9 and "afirma_la_parte" in _p9 and "no_se_pronuncia" in _p9 and "carga_de" in _p9
   and "afirmado_por_la_parte" in _p9, "el prompt pide las premisas y conoce la condición nueva")
ok("faltantes:" not in _p9, "y deja de pedir faltantes abiertos (las preguntas los sustituyen)")
ok("emplazamiento" not in _p9 and "cláusula sexta" not in _p9,
   "describe la forma de la pregunta sin dar una (un ejemplo literal se copia)")
_T9 = al.textos(ACTO, ESCRITO, "")
_c = "el adquirente no acreditó la cesión del derecho contractual concreto que se ejecuta"
ok(al.verificar_cita(_c, _T9, "acto", solo=("acto", "constancias"))[1] is False
   and al.verificar_cita(_c, _T9, "acto")[1] is True,
   "con `solo`, una cita del escrito NO pasa por cita del acto (sin `solo`, sí: la caída de siempre)")
_CR9 = dict(CRUDO, faltantes=[{"que": "el contrato", "por_que_importa": "x"}],
            hechos=CRUDO["hechos"] + [
                {"id": "H4", "que": "lo dice la parte", "afirma": "recurrente", "fuente": "escrito",
                 "condicion": "no_controvertido", "cita": _c},
                {"id": "H5", "que": "lo dice el acto", "afirma": "a_quo", "fuente": "escrito",
                 "condicion": "tenido_por_acreditado",
                 "cita": "las prestaciones a ejecutar no cambiaron con la sustitución de la parte actora"}],
            premisas=[
                {"id": "F1", "segmento": "a1.a", "problema": 1, "afirma_la_parte": "no hubo cesión",
                 "cita_escrito": _c, "el_acto": "no_se_pronuncia", "cita_acto": "", "carga_de": "recurrente",
                 "pregunta": "hubo cesión del derecho contractual", "tipo": "si_no",
                 "si_si": "infundado", "si_no": "fundado", "para_que": "legitimación"},
                {"id": "F2", "segmento": "A1.a", "afirma_la_parte": "conocía el arrendamiento",
                 "cita_escrito": _c, "el_acto": "lo_tuvo_por_cierto",
                 "cita_acto": "adquirió el bien durante el procedimiento y conocía el arrendamiento celebrado"},
                # 3-oct-2026: sin pregunta, como la manda el modelo real cuando dice
                # «lo tuvo por cierto» (el prompt sólo la pide si «no se pronuncia»).
                {"id": "F3", "segmento": "A1.a", "afirma_la_parte": "la Sala lo tuvo por cierto",
                 "cita_escrito": _c, "el_acto": "lo_tuvo_por_cierto", "carga_de": "recurrente",
                 "cita_acto": _c},
                {"id": "F4", "segmento": "A1.a", "afirma_la_parte": "inventada",
                 "cita_escrito": "esta frase no aparece en el escrito de la parte por ningún lado nunca",
                 "el_acto": "no_se_pronuncia", "pregunta": "¿Pasó?", "si_si": "fundado", "si_no": "infundado"}])
_d9 = al.verificar(_CR9, ACTO, ESCRITO, "", SEGS, True)
_H9 = {h["id"]: h for h in _d9["hechos"]}
ok(_H9["H4"]["condicion"] == "afirmado_por_la_parte",
   "un «no controvertido» cuya cita sólo está en el escrito: AFIRMADO POR LA PARTE")
ok(_H9["H5"]["condicion"] == "tenido_por_acreditado" and _H9["H5"]["fuente"] == "acto",
   "si también está en el acto, la condición se queda y la fuente es el acto")
ok(_d9["faltantes"] == [], "los faltantes abiertos ya no se piden")
_F = {f["id"]: f for f in _d9["premisas"]}
ok(_F["F1"]["segmento"] == "A1.a" and _F["F1"]["cita_escrito_verificada"] and _F["F1"]["carga_de"] == "la_parte"
   and _F["F1"]["pregunta"] and _F["F1"]["tipo"] == "si_no",
   "la premisa del escrito, con su segmento del inventario, su carga y su pregunta")
ok(_F["F2"]["el_acto"] == "lo_tuvo_por_cierto" and _F["F2"]["fuente_acto"] == "acto" and "pregunta" not in _F["F2"],
   "lo que el acto tuvo por cierto, con su cita del acto, no se pregunta")
ok(_F["F3"]["el_acto"] == "lo_tuvo_por_cierto" and _F["F3"]["cita_acto"] == ""
   and _F["F3"]["cita_acto_verificada"] is False and "pregunta" not in _F["F3"],
   "la «cita del acto» que sólo está en el escrito no se cita, pero lo declarado se CONSERVA, sin verificar "
   "(antes bajaba a «no se pronuncia» y decidía la carga contra la parte)")
ok(_F["F2"]["cita_acto_verificada"] is True, "y la que sí se halló queda verificada")
ok(not _F["F4"]["cita_escrito_verificada"] and "pregunta" not in _F["F4"],
   "una premisa que la parte no escribió no se le pregunta a nadie")
import preguntas_secretario as ps9
_q9 = ps9.preguntas_de(_d9, [{"pregunta": "¿Procede?", "jerarquia": "principal"}])
ok(len(_q9) == 1 and all(q["indispensable"] for q in _q9) and _q9[0]["pregunta"].startswith("¿"),
   "de ahí sale la pregunta de F1, indispensable por código (F3 no: el acto sí se pronunció)")
_b9 = al.bloque_propuesta(ps9.con_respuestas(_d9, ps9.preguntas_de(
    _d9, [{"pregunta": "¿Procede?", "jerarquia": "principal"}], {_q9[0]["id"]: "no"})))
ok("PREMISAS DE LA PARTE, CONTRASTADAS CON LA RESOLUCIÓN" in _b9 and "[lo afirma: recurrente]" in _b9
   and "SÓLO LO AFIRMA LA PARTE" in _b9, "el bloque dice quién afirma cada hecho y la sección de premisas")
ok("respuesta del secretario: NO" in _b9 and "decide la carga de la prueba" in _b9,
   "con la respuesta del secretario, o sin ella la carga de la prueba")
ok("lo que sólo afirma la parte NO es un hecho del asunto" in _b9 and "FALTA EN LOS INSUMOS" not in _b9,
   "y su regla")
_l3 = next(i for i, x in enumerate(_b9.splitlines()) if x.strip().startswith("F3"))
_sig3 = _b9.splitlines()[_l3 + 1]
ok("LO TUVO POR CIERTO" in _b9.splitlines()[_l3] and "NO se halló" in _sig3 and "carga" in _sig3
   and "decide la carga de la prueba, que toca" not in _sig3,
   "F3 se imprime como lo declaró, con su cita sin hallar, y sin la carga de la prueba")
ok(al.estado_de_premisa(_F["F3"]).startswith("lo_tuvo_por_cierto (cita sin verificar")
   and al.estado_de_premisa(_F["F2"]) == "lo_tuvo_por_cierto",
   "la etiqueta corta para el examinador lo dice (cita sin verificar)")

print("\n10 · LAS OMISIONES Y LOS HECHOS DEL PROCEDIMIENTO NO SE DECIDEN POR CARGA (3-oct-2026)")
ok("se_verifica_en" in _p9 and "omitió pronunciarse" in _p9 and "no la listes" in _p9,
   "el prompt de las premisas deja fuera las omisiones de la resolución y pide `se_verifica_en`")
ok(al.verifica_en("", "La Sala omitió pronunciarse sobre la prescripción que hice valer") == "resolucion"
   and al.verifica_en("juicio", "la responsable no valoró la pericial en contabilidad") == "resolucion"
   and al.verifica_en("", "el juez dejó de estudiar el agravio tercero") == "resolucion",
   "una omisión de la resolución se reconoce aunque el modelo diga otra cosa")
ok(al.verifica_en("", "que hizo valer la excepción de prescripción en la apelación") == "autos"
   and al.verifica_en("autos", "que firmó el contrato") == "autos"
   and al.verifica_en("", "la autoridad omitió notificarle el crédito") == "autos",
   "un hecho del procedimiento es «autos» (y «omitió notificar» no es una omisión de la resolución)")
ok(al.verifica_en("", "que trabajó para el demandado desde 2019") == "juicio"
   and al.verifica_en("raro", "que pagó la renta") == "juicio",
   "lo demás, «juicio», como siempre")
_A = "Que la Sala omitió pronunciarse sobre la excepción de prescripción que hice valer en mi apelación"
_E10 = ESCRITO + " " + _A + "."
_CR10 = dict(CRUDO, premisas=[
    {"id": "F1", "segmento": "A1.a", "problema": 1, "afirma_la_parte": _A, "cita_escrito": _A,
     "el_acto": "no_se_pronuncia", "carga_de": "la parte quejosa", "pregunta": "¿Se pronunció la Sala?",
     "si_si": "fundado", "si_no": "infundado"},
    {"id": "F2", "segmento": "A1.a", "problema": 1, "afirma_la_parte": "que hizo valer la prescripción en la apelación",
     "cita_escrito": _A, "el_acto": "no_se_pronuncia", "carga_de": "la parte", "se_verifica_en": "autos",
     "pregunta": "¿Hizo valer la prescripción en sus agravios de apelación?", "si_si": "fundado",
     "si_no": "infundado"}])
_d10 = al.verificar(_CR10, ACTO, _E10, "", SEGS, True)
_F10 = {f["id"]: f for f in _d10["premisas"]}
ok(_F10["F1"]["se_verifica_en"] == "resolucion" and "pregunta" not in _F10["F1"],
   "la omisión de la resolución no lleva pregunta: se comprueba leyéndola")
ok(_F10["F2"]["se_verifica_en"] == "autos" and _F10["F2"]["pregunta"], "el hecho del procedimiento sí se pregunta")
_q10 = ps9.preguntas_de(_d10, [{"pregunta": "¿Procede?", "jerarquia": "principal"}])
ok(len(_q10) == 1 and "pendiente de verificar en autos" in _q10[0]["si_no_contesta"]
   and "carga" in _q10[0]["si_no_contesta"] and "es suya" not in _q10[0]["si_no_contesta"],
   "sin respuesta, el hecho de autos queda pendiente: no «la carga de probarlo es suya»")
_b10 = al.bloque_propuesta(_d10)
ok("es una omisión que se atribuye a la resolución" in _b10 and "pendiente de verificar en" in _b10
   and "decide la carga de la prueba, que toca" not in _b10,
   "el bloque no aplica la carga ni a la omisión ni al hecho de autos")
ok(not al.decide_la_carga(_F10["F1"]) and not al.decide_la_carga(_F10["F2"])
   and al.decide_la_carga({"el_acto": "no_se_pronuncia"}) and not al.decide_la_carga(_F["F3"]),
   "decide_la_carga: sólo el hecho del juicio sobre el que la resolución calla (lo guardado antes, como siempre)")

print("\n11 · LA REGLA NO MANDA ESCRIBIR IDENTIFICADORES INTERNOS; LA RED LOS QUITA (3-oct-2026)")
ok("H# o la F#" not in _b9 and "sin escribir los identificadores" in _b9,
   "la regla nombra el hecho por su contenido y su fuente, no por «H#/F#»")
ok(al.quitar_ids_internos("Es fundado porque la notificación por lista (H1) no cumplió el art. 27, y la parte "
                          "no fue llamada (F1).")
   == "Es fundado porque la notificación por lista no cumplió el art. 27, y la parte no fue llamada.",
   "quita los ids entre paréntesis")
ok(al.quitar_ids_internos("Fundado conforme a F2: la Sala no valoró.") == "Fundado: la Sala no valoró."
   and al.quitar_ids_internos("Según H1 y F2, procede.") == "Procede."
   and al.quitar_ids_internos("El hecho H3 consta en autos.") == "El hecho consta en autos.",
   "y los que van tras un conector o tras «el hecho»")
_leg = "La fórmula H2O y el art. 14, fracción II; tesis 2a./J. 172/2010, registro 2021345, AR 631/2025."
ok(al.quitar_ids_internos(_leg) == _leg and al.quitar_ids_internos(None) is None
   and al.quitar_ids_internos("Sin ids.") == "Sin ids.",
   "no toca artículos, claves de tesis, registros ni expedientes")
_pp = al.quitar_ids_de_propuesta({"global": {"sentido": "fundado", "razon": "Fundado (H1).",
                                             "registros": ["H1x"]},
                                  "propuestas": [{"razon": "véase F2", "alternativa": {"razon": "R1 basta"}}]})
ok(_pp["global"]["razon"] == "Fundado." and _pp["global"]["sentido"] == "fundado"
   and _pp["propuestas"][0]["alternativa"]["razon"] == "Basta" and _pp["propuestas"][0]["razon"] == "",
   "quitar_ids_de_propuesta limpia sólo la prosa, en una copia")

print("\n12 · CON LA PROBABILIDAD, LA REGLA NO HABLA DE «alcanza» (3-oct-2026)")
ct.poner(True, {"banderas": {"preguntas_al_secretario": True, "propuesta_por_probabilidad": True}}, pruebas=True)
_b12 = al.bloque_propuesta(_d9)
ok("alcanza" not in _b12 and "`sostenida`=false" in _b12 and "Decide siempre" in _b12,
   "con «propuesta_por_probabilidad»: decide siempre, `sostenida`=false si el acervo no lo respalda")
ct.poner(True, {"banderas": {"propuesta_por_probabilidad": True}}, pruebas=True)
ok("alcanza" not in al.bloque_propuesta(_d8) and "`sostenida`=false" in al.bloque_propuesta(_d8),
   "también en la regla sin las premisas")
ct.poner(False, {})
ok("alcanza=false queda sólo para cuando el acervo no da para sostener ningún sentido" in al.bloque_propuesta(_d8),
   "sin banderas, el texto de siempre")
ct.poner(False, {})

print()
if FALLOS:
    print("FALLAS:\n  " + "\n  ".join(FALLOS))
    sys.exit(1)
print("RESULTADO: TODAS LAS COMPROBACIONES PASAN")
