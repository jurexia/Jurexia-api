"""p2-congruencia — la congruencia interna del estudio (familia v2: v2, v3, v4).

26-sep-2026. David: «que no sea repetitivo y el proyecto mismo sea inteligente
(se le denomina congruencia interna) y que conteste todo lo efectivamente
planteado». Dos defectos vistos en el ADC 642/2024 v4, congelados aquí con su
texto real (anonimizado: datos/congruencia/):

  (1) los apartados abrían «Sobre el primer concepto…, en el que la parte
      quejosa sostiene que…» y seguían con «Lo anterior, porque…» SIN la
      calificación en medio. EL MODELO SÍ LA ESCRIBIÓ —«Es fundado.», sola en
      su renglón— y el compositor del .docx tira los párrafos de menos de seis
      palabras. Se comprueba el camino entero: estudio → pegar → compositor.
  (2) el plan ponía a cada argumento la etiqueta de SU PROBLEMA, y la
      reconvención (la Sala sí la examinó) salió «fundada». plan-4: el
      sentido del problema es del secretario; el del argumento, dentro de él,
      se razona con la regla de coherencia.

Secciones:
  1 · la calificación en una frase (engroses, v1, v2 y la atribución a la parte);
  2 · el 642 v4 congelado: el compositor se come la calificación; pegada, sobrevive;
  3 · la reparación: la calificación que asigna el criterio (o el plan), sin tocar
      nada más, y el aviso cuando no es unívoca;
  4 · la calificación de cada argumento dentro del sentido de su problema (plan-4,
      la reconvención del 642) y en el prompt de la familia v2;
  5 · la extensión sin presión (prompt y «se quedó corto»);
  6 · los EFECTOS casan con el cuerpo (en sombra);
  7 · el camino: los dos gemelos, `_terminar`, el «listo», y la v1 intacta.

    .venv/bin/python test_congruencia.py
"""
import ast
import asyncio
import json
import os
import types

os.environ.pop("ESTUDIO_PROMPT", None)
os.environ.pop("ESTUDIO_REPARAR_APERTURAS", None)

import congruencia as K
import fase6_estudio as f6
import formato_sentencia as fs
import marcas as mc
import plan_estudio as pe

FALLOS = []


def ok(cond, que):
    print(f"   {'PASA ' if cond else 'FALLA'}  {que}")
    if not cond:
        FALLOS.append(que)


AQUI = os.path.dirname(os.path.abspath(__file__))
DATOS = os.path.join(AQUI, "datos", "congruencia")
V4 = open(os.path.join(DATOS, "642_v4.estudio"), encoding="utf-8").read()
V3 = open(os.path.join(DATOS, "642_v3.estudio"), encoding="utf-8").read()
PLAN = json.load(open(os.path.join(DATOS, "642_v4.plan"), encoding="utf-8"))

# EL CRITERIO DEL 642, como lo fijó el secretario (el del plan: el problema 1
# cubre los conceptos primero y tercero; el 2, el segundo; el 3, el cuarto).
P1 = "¿La Sala tuvo por acreditada la identidad del bien reivindicado sin prueba suficiente?"
P2 = "¿La Sala debió declarar operante la cosa juzgada refleja del juicio anterior?"
P3 = "¿La Sala motivó la condena en costas de ambas instancias?"
CRIT = [f6.Criterio(P1, "fundado", "La Sala no explicó qué prueba vincula la fracción con el bien poseído.",
                    "principal"),
        f6.Criterio(P2, "inoperante", "No combate la consideración independiente.", "accesorio"),
        f6.Criterio(P3, "fundado", "No precisó la hipótesis legal de la condena.", "accesorio")]
PROB = [{"pregunta": P1, "cubre": [1, 3]}, {"pregunta": P2, "cubre": [2]}, {"pregunta": P3, "cubre": [4]}]


def componer(estudio: str) -> list:
    """El estudio por el compositor REAL del .docx: lo que el secretario lee."""
    import docx
    import documento_generado as dg
    d = docx.Document()
    dg._escribir_estudio(d, f6.parrafos(estudio), [], [], [])
    return [p.text for p in d.paragraphs if p.text.strip()]


# ═══════════════════════════════════════════════════════════════════════════
print("\n1 · LA CALIFICACIÓN EN UNA FRASE")
for frase, esperado, que in (
        ("Es fundado.", True, "la calificación sola"),
        ("Se considera infundado.", True, "la fórmula del oficio"),
        ("Sobre el primer concepto de violación, en el que la parte quejosa sostiene que X, se "
         "considera infundado.", True, "la v1: la oración principal vuelve tras la coma"),
        ("En relación con el tercero, en el que sostiene que Y, su estudio resulta innecesario.", True,
         "…también con un sujeto corto («su estudio»)"),
        ("Lo anterior es infundado porque la Sala aplicó la jurisprudencia.", True,
         "el engrose: «Lo anterior es infundado porque…» (ADC 642/2024, el de oro)"),
        ("De ahí lo infundado de la disidencia planteada.", True, "el engrose: «lo infundado»"),
        ("No le asiste la razón a la parte quejosa.", True, "«no le asiste la razón»"),
        ("Dicho argumento resulta infundado.", True, "«argumento» no es el verbo «argumentó»"),
        ("Sobre el primer concepto de violación, en el que la parte quejosa sostiene que el agravio "
         "de apelación era fundado.", False, "lo que la parte sostiene no es la calificación"),
        ("La resolución fundada en el artículo 14 constitucional.", False, "«fundada en el artículo»"),
        ("La sentencia debidamente fundada y motivada.", False, "«fundada y motivada»")):
    ok(K.califica(frase) is esperado, que)
ok(K.ordinales("En el primero concepto de violación el quejoso") == {1}
   and K.ordinales("los conceptos de violación segundo y tercero") == {2, 3}
   and K.ordinales("los conceptos que, según el contrato… las cláusulas primera y décima tercera") == set()
   and K.ordinales("el tercero interesado") == set(),
   "el ordinal va pegado a su sustantivo (la cláusula «primera» no abre un apartado)")

_resumen = ("En su segundo concepto de violación, la quejosa sostiene que la Sala omitió la cosa juzgada.\n"
            "Lo anterior, pues aduce que en el expediente anterior la acción se declaró improcedente.")
_demo = ("Sobre el segundo concepto de violación, en el que la quejosa sostiene que la Sala omitió la cosa "
         "juzgada.\nLo anterior, porque el concepto no combate la consideración independiente.")
ok(not [h for h in K.huecos(_resumen.split("\n")) if h["tipo"] == "lo_anterior"]
   and [h["tipo"] for h in K.huecos(_demo.split("\n"))] == ["apertura", "lo_anterior"],
   "«Lo anterior, pues aduce que…» resume a la parte (5 de 40 engroses): no es la demostración huérfana")

# ═══════════════════════════════════════════════════════════════════════════
print("\n2 · EL 642 v4 CONGELADO: EL COMPOSITOR SE COMÍA LA CALIFICACIÓN")
_lineas = [x.strip() for x in V4.split("\n") if x.strip()]
ok(sum(K.es_sola(x) for x in _lineas) == 4
   and [x for x in _lineas if K.es_sola(x)] == ["Es fundado.", "Es fundado.", "Resulta inoperante.", "Es fundado."],
   "el modelo SÍ escribió las cuatro calificaciones, cada una sola en su renglón")
ok(K.huecos(f6.parrafos(V4)) == [], "en el estudio que escribió el modelo no hay hueco")
_entregado = componer(V4)
ok(not any(t.strip() in ("Es fundado.", "Resulta inoperante.") for t in _entregado),
   "el compositor del .docx las tira (menos de seis palabras): así se entregó")
_h_ent = K.huecos(_entregado)
ok(sorted({h["tipo"] for h in _h_ent}) == ["apertura", "lo_anterior"]
   and sorted(h["ordinales"][0] for h in _h_ent if h["tipo"] == "apertura") == [1, 2, 3, 4]
   and sum(h["tipo"] == "lo_anterior" for h in _h_ent) == 5,
   "en lo entregado, los cuatro apartados abren sin calificación y cinco «Lo anterior» no tienen "
   "antecedente (el defecto que vio el secretario)")
_pegado, _hechas = K.pegar_calificaciones(V4)
ok(len(_hechas) == 4 and all(h[2] == "atrás" for h in _hechas), "se pegan las cuatro, cada una a su apertura")
ok("sostiene que la Sala responsable tuvo por acreditada la identidad del bien reivindicado sin prueba "
   "suficiente. Es fundado." in _pegado
   and "respecto de lo resuelto en el expediente 1000/2017. Resulta inoperante." in _pegado,
   "la apertura termina con su calificación")
ok([x for x in _pegado.split("\n") if x.strip() and not K.es_sola(x)]
   == [x if not any(x.endswith(t) for t in (". Es fundado.", ". Resulta inoperante.")) else x
       for x in _pegado.split("\n") if x.strip()]
   and len(_pegado.split()) == len(V4.split()),
   "ni una palabra más ni una menos: sólo cambia de renglón")
_compuesto = componer(_pegado)
ok(sum(t.endswith("Es fundado.") for t in _compuesto) == 3 and sum(t.endswith("Resulta inoperante.")
                                                                   for t in _compuesto) == 1,
   "EL CAMINO ENTERO: pegada, la calificación llega al .docx")
ok(K.huecos(_compuesto) == [], "y en lo que se entrega ya no queda hueco")
# LA v3 DEL MISMO ASUNTO, que calificó bien: nada que pegar, nada que tocar.
_p3, _h3 = K.pegar_calificaciones(V3)
ok(_p3 == V3 and _h3 == [] and K.huecos(f6.parrafos(V3)) == [], "la v3 del 642 sale idéntica")
_est3, _inf3 = K.reparar_aperturas(V3, CRIT, PROB)
ok(_est3 == V3 and not _inf3["anadidas"] and not _inf3["sin_reparar"] and K.aviso_aperturas(_inf3) == "",
   "…también por la reparación entera, sin aviso")
# CON MARCAS (v3/v4): la marca del renglón suelto pasa a su apertura.
_m, _ = K.pegar_calificaciones("⟦C1.a⟧ Sobre el primer concepto de violación, en el que sostiene X.\n"
                               "⟦C1.b⟧ Es fundado.\n\nLo anterior, porque sí.")
ok(_m.split("\n")[0] == "⟦C1.a C1.b⟧ Sobre el primer concepto de violación, en el que sostiene X. Es fundado."
   and mc.separar_marcas(_m)[1] == {"C1.a": [0], "C1.b": [0]},
   "con marcas: la calificación suelta lleva su marca a la apertura")
_mod, _hm = K.pegar_calificaciones("1. ¿La Sala debió admitir la prueba?\n\nSí.\n\nEl primer concepto de "
                                   "violación es fundado, porque la prueba se ofreció a tiempo.")
ok(_hm and _mod == ("1. ¿La Sala debió admitir la prueba?\n\nSí. El primer concepto de violación es fundado, "
                    "porque la prueba se ofreció a tiempo."),
   "la moderna: el «Sí.» suelto va al comienzo de su respuesta, no a la pregunta")
ok(K.pegar_calificaciones("EFECTOS DE LA CONCESIÓN\n\nEs fundado.")[0] == "EFECTOS DE LA CONCESIÓN\n\nEs fundado.",
   "detrás de un rótulo no se pega nada")

# ═══════════════════════════════════════════════════════════════════════════
print("\n3 · LA REPARACIÓN: LA CALIFICACIÓN QUE ASIGNA EL CRITERIO")
# Si el modelo NO hubiera escrito la calificación (lo entregado, en texto):
_sin = "\n".join(x for x in V4.split("\n") if not K.es_sola(x))
_rep, _inf = K.reparar_aperturas(_sin, CRIT, PROB)
ok([(x["ordinales"], x["frase"], x["fuente"]) for x in _inf["anadidas"]]
   == [([1], "Es fundado.", "criterio"), ([3], "Es fundado.", "criterio"),
       ([2], "Es inoperante.", "criterio"), ([4], "Es fundado.", "criterio")],
   "a cada apertura, la calificación del problema que la cubre (CUBRE), en el orden del estudio")
ok(K.huecos(f6.parrafos(_rep)) == [], "y no queda ningún «Lo anterior» sin antecedente")
_a, _b = [x for x in _sin.split("\n")], [x for x in _rep.split("\n")]
_cambian = [i for i in range(len(_a)) if _a[i] != _b[i]]
ok(len(_a) == len(_b) and len(_cambian) == 4
   and all(_b[i].startswith(_a[i].rstrip()) and len(_b[i].split()) - len(_a[i].split()) == 2 for i in _cambian),
   "sólo cambian los cuatro renglones de apertura, y sólo por su calificación al final")
_av = K.aviso_aperturas(_inf)
ok(_av.startswith("CALIFICACIÓN AL ABRIR") and "el primer concepto de violación su calificación «Es fundado.»"
   in _av and "la de su criterio" in _av, "el aviso dice qué añadió la máquina y de dónde lo sacó")
# Ambiguo: el primer concepto lo cubren dos problemas con sentidos distintos.
_prob_amb = [{"pregunta": P1, "cubre": [1, 3]}, {"pregunta": P2, "cubre": [1, 2]}, {"pregunta": P3, "cubre": [4]}]
_rep2, _inf2 = K.reparar_aperturas(_sin, CRIT, _prob_amb)
ok(any(x["ordinales"] == [1] and "sentidos distintos" in x["motivo"] for x in _inf2["sin_reparar"])
   and not any(x["ordinales"] == [1] for x in _inf2["anadidas"]),
   "si no es unívoca, no escribe")
ok("no se pudo añadir en el primer concepto de violación" in K.aviso_aperturas(_inf2), "…y lo avisa")
os.environ["ESTUDIO_REPARAR_APERTURAS"] = "0"
import importlib
importlib.reload(K)
_rep3, _inf3b = K.reparar_aperturas(_sin, CRIT, PROB)
ok(_rep3 == _sin and len(_inf3b["sin_reparar"]) == 4, "con el interruptor apagado sólo se pega y se avisa")
os.environ.pop("ESTUDIO_REPARAR_APERTURAS")
importlib.reload(K)
# LO QUE VA EN SOMBRA: la apertura sin calificación sin «Lo anterior» huérfano
# (el engrose que expone el concepto en varios párrafos y califica después).
_engrose = ("En su primer concepto de violación, la quejosa aduce que la Sala valoró mal la pericial.\n"
            "Sostiene que el perito no tenía cédula.\n"
            "Dicho argumento de la quejosa resulta infundado.\n"
            "Contrario a lo que afirma, el perito sí la exhibió.")
_r_eng, _i_eng = K.reparar_aperturas(_engrose, CRIT, PROB)
ok(_r_eng == _engrose and not _i_eng["anadidas"] and [x["tipo"] for x in _i_eng["sombra"]] == ["apertura"]
   and K.aviso_aperturas(_i_eng) == "",
   "la forma del engrose que califica después de exponer: se anota en sombra, no se toca ni se avisa")
_corto = _engrose.replace("Dicho argumento de la quejosa resulta infundado.", "Dicho argumento resulta infundado.")
_r_c, _i_c = K.reparar_aperturas(_corto, CRIT, PROB)
ok("Sostiene que el perito no tenía cédula. Dicho argumento resulta infundado." in _r_c
   and len(_r_c.split()) == len(_corto.split()) and not _i_c["anadidas"],
   "…pero si la calificación es de menos de seis palabras (el compositor la tiraría), se pega, sin añadir nada")
# LOS DOS CANDADOS DE LA REPARACIÓN (revisión adversarial, 26-sep-2026): la
# máquina no escribe una calificación que la demostración del apartado niega,
# ni adivina, sin el plan, cuál de los conceptos de un problema fundado lo funda.
ok(K.direcciones(["Es infundado, porque la Sala sí examinó la reconvención."]) == {"contra"}
   and K.direcciones(["No le asiste la razón a la quejosa."]) == {"contra"}
   and K.direcciones(["El planteamiento no es fundado."]) == {"contra"}
   and K.direcciones(["Es fundado pero insuficiente, pues subsiste la otra consideración."]) == {"contra"}
   and K.direcciones(["No asiste razón en una parte; sin embargo, sí la asiste en otra."]) == {"contra", "favor"}
   and K.direcciones(["Resulta fundado el concepto."]) == {"favor"}
   and K.direcciones(["COSTAS. SE DETERMINAN POR LO FUNDADO O INFUNDADO DE LOS AGRAVIOS."]) == set(),
   "la dirección de lo que el apartado ya califica (la negación y el «pero insuficiente» van en contra; "
   "el rubro de una tesis no cuenta)")
_dos = ("Los conceptos de violación son fundados.\n\n"
        "Sobre el primer concepto de violación, en el que la parte quejosa sostiene que faltó la pericial.\n\n"
        "Lo anterior, porque la Sala no explicó qué prueba identifica la fracción reclamada.\n\n"
        "Sobre el tercer concepto de violación, en el que la parte quejosa sostiene que la Sala omitió la "
        "reconvención.\n\n"
        "Lo anterior, porque la Sala sí examinó la confesional y la testimonial.\n\n"
        "De ahí que el planteamiento sea infundado.")
_rd, _id = K.reparar_aperturas(_dos, CRIT, PROB)
ok(not _id["anadidas"] and _rd == _dos
   and sorted((x["ordinales"], "otros conceptos" in x["motivo"] or "contraria" in x["motivo"])
              for x in _id["sin_reparar"]) == [([1], True), ([3], True)],
   "sin plan, el problema 1 (fundado) cubre el primero y el tercero: al primero no se le añade «fundado» "
   "porque nada dice que sea él quien lo funda, y al tercero, que se demuestra infundado, menos")
_uno = _dos.replace("identifica la fracción reclamada.",
                    "identifica la fracción reclamada.\n\nPor eso le asiste la razón a la quejosa.")
_ru, _iu = K.reparar_aperturas(_uno, CRIT, PROB)
ok([x["ordinales"] for x in _iu["anadidas"]] == [[1]] and [x["ordinales"] for x in _iu["sin_reparar"]] == [[3]],
   "…si el cuerpo del primero ya razona a su favor, se le añade; el tercero sigue sin tocarse y se avisa")
_plan_mal = {"segmentos": [{"id": "C3.a", "etiqueta": "fundado"}]}
_dos_m = _dos.replace("Sobre el tercer concepto", "⟦C3.a⟧ Sobre el tercer concepto")
_rm, _im = K.reparar_aperturas(_dos_m, CRIT, PROB, plan=_plan_mal)
ok(not any(x["ordinales"] == [3] for x in _im["anadidas"])
   and any(x["ordinales"] == [3] and "contraria" in x["motivo"] for x in _im["sin_reparar"]),
   "con plan: tampoco la etiqueta del plan se escribe si la demostración del apartado va al revés")
ok("el tercer concepto de violación («Lo anterior, porque la Sala sí examinó" in K.aviso_aperturas(_im)
   and "dirección contraria a «fundado» (la de su plan)" in K.aviso_aperturas(_im),
   "…y el secretario lo sabe por el aviso, con su porqué")
_ef = ("Sobre el primer concepto de violación, en el que sostiene X.\n\nEs fundado.\n\nLo anterior, porque sí.\n\n"
       "EFECTOS DE LA CONCESIÓN\n\n1. Deje insubsistente la sentencia.\n\nEs fundado.\n\n2. Dicte otra.")
_pe_, _he_ = K.pegar_calificaciones(_ef)
ok(len(_he_) == 1 and "1. Deje insubsistente la sentencia.\n\nEs fundado.\n\n2. Dicte otra." in _pe_,
   "el pegado sólo toca el cuerpo: en los EFECTOS un renglón corto no es la calificación de ningún apartado")
_sombra = K.informe_sombra(_inf)
ok("texto" not in json.dumps(_sombra) or all("texto" not in x for x in _sombra["anadidas"] + _sombra["sin_reparar"]
                                              + _sombra["sombra"]),
   "a la ficha van cifras e identificadores, sin el texto del estudio")

# ═══════════════════════════════════════════════════════════════════════════
print("\n4 · LA CALIFICACIÓN DE CADA ARGUMENTO DENTRO DEL SENTIDO DE SU PROBLEMA")
_segs = [dict(s) for s in PLAN["segmentos"]]
_probs = [{"id": 1, "sentido": "fundado"}, {"id": 2, "sentido": "inoperante"}, {"id": 3, "sentido": "fundado"}]
ok(all(s["etiqueta"] == {1: "fundado", 2: "inoperante", 3: "fundado"}[s["problema_id"]] for s in _segs),
   "el plan del 642 v4 (plan-3): cada argumento con la etiqueta de su problema, reconvención (C1.d) incluida")
_c1d = next(s for s in _segs if s["id"] == "C1.d")
_c1d["etiqueta"] = "infundado"
ok(pe.etiqueta_fuera("infundado", "fundado") == "" and pe.problemas_sin_quien_los_funde(_segs, _probs) == [],
   "plan-4: la reconvención INFUNDADA dentro del problema 1, fundado, cabe (C1.a-c lo fundan)")
ok(pe.PLAN_VERSION == "plan-4", "y un plan-3 guardado no se reutiliza")
_todos_inf = [dict(s, etiqueta="infundado") if s["problema_id"] == 1 else s for s in _segs]
ok(pe.problemas_sin_quien_los_funde(_todos_inf, _probs) == [1],
   "si ningún argumento funda el problema 1, las etiquetas niegan el sentido del secretario")
ok(pe.etiqueta_fuera("fundado", "infundado") != "" and pe.etiqueta_fuera("fundado_insuficiente", "infundado") == "",
   "dentro de un problema que no prospera, el que tiene razón y no alcanza es «fundado pero insuficiente»")
ok("etiqueta: infundado (dentro del problema 1, fundado)" in pe._linea_seg(_c1d, "desarrolla", "", {1: "fundado"}),
   "el guion dice la calificación del argumento y el sentido de su problema")
# EL GUION YA NO DICE LA IGUALDAD DE plan-3 (revisión adversarial): ni en su
# encabezado ni en la descripción que lee el estudio; y el RESIDUAL que trae
# dato se hace cargo de él (el planificador marca residual con dato: 16 de 18
# en el banco de plan_rep), en vez de despacharlo en una o dos frases.
_bl = " ".join(pe.bloque("APARTADO 1").split())
ok("el sentido de cada argumento es el de su criterio" not in _bl
   and "el sentido de cada problema es el del criterio y el guion no lo cambia" in _bl
   and "la etiqueta de cada argumento es su calificación dentro de él" in _bl,
   "la descripción del guion: el sentido es del PROBLEMA; la etiqueta, del argumento")
ok("si el guion le pone un dato, la respuesta se hace cargo de ese dato" in _bl,
   "RESIDUAL con dato: la respuesta se hace cargo del dato")
_vi = pe.vista({"segmentos": [dict(_c1d, trat="aplica")], "problemas": [{"id": 1, "sentido": "fundado"}],
                "unidades": [], "premisas": [], "propuestas": [], "orden": {}}, "estandar")
ok(_vi.startswith("GUION DEL ESTUDIO — organiza; el sentido de cada problema es el del criterio"),
   "el encabezado del guion habla del sentido de cada problema")
# El apartado del primer concepto mezcla C1.a (fundado) y C1.d (infundado):
ok(K.calificacion_de_apartado([1], CRIT, PROB, ["C1.a", "C1.d"], {"segmentos": _segs}) == ("fundado", "criterio"),
   "la apertura del apartado mixto lleva el sentido de su problema (lo funda C1.a)")
ok(K.calificacion_de_apartado([1], CRIT, PROB, ["C1.d"], {"segmentos": _segs}) == ("infundado", "plan"),
   "un apartado sólo de la reconvención abre con la suya, del plan")
for forma in ("estandar", "moderna"):
    _p2 = f6.prompt_estudio("ACTO", "CONC", CRIT, f6.Material(tipo_asunto="amparo_directo", formato=forma,
                                                              problemas=PROB, n_planteamientos=4, variante="v2"))
    _p1 = f6.prompt_estudio("ACTO", "CONC", CRIT, f6.Material(tipo_asunto="amparo_directo", formato=forma,
                                                              problemas=PROB, n_planteamientos=4, variante="v1"))
    _p2n = " ".join(_p2.split())
    ok("CADA ARGUMENTO LLEVA SU PROPIA CALIFICACIÓN" in _p2n and "fundado pero insuficiente" in _p2n
       and "lo fijó el secretario y no se cambia" in _p2n,
       f"v2/{forma}: el bloque del criterio da la regla de coherencia")
    ok("CADA ARGUMENTO LLEVA SU PROPIA CALIFICACIÓN" not in _p1, f"v1/{forma}: no la recibe")
    ok("Cada concepto de violación recibe la calificación que le" not in _p2,
       f"v2/{forma}: ya no dice que cada concepto hereda la de su problema")

# ═══════════════════════════════════════════════════════════════════════════
print("\n5 · LA APERTURA EN EL PROMPT Y LA EXTENSIÓN SIN PRESIÓN")
_std = f6.prompt_estudio("ACTO", "CONC", CRIT, f6.Material(tipo_asunto="amparo_directo", formato="estandar",
                                                           problemas=PROB, n_planteamientos=4, variante="v2"))
_mod = f6.prompt_estudio("ACTO", "CONC", CRIT, f6.Material(tipo_asunto="amparo_directo", formato="moderna",
                                                           problemas=PROB, n_planteamientos=4, variante="v2"))
ok("Y EN EL MISMO PÁRRAFO" in _std and "Nunca en un renglón\n       aparte" in _std
   and "EN ESE MISMO PÁRRAFO, su calificación" in _std and "sigue su calificación" not in _std,
   "estándar: la calificación en el párrafo de la apertura, dicho en el cuerpo y en el recordatorio final")
ok("EN ESE MISMO PÁRRAFO, NOMBRA" in _mod and "la respuesta no va\n  sola en su renglón" in _mod,
   "moderna: la respuesta nombra el concepto y su calificación en su mismo párrafo")
ok("cada argumento que trae un dato propio" in _std and "TECHO ALTO" in _std and "No es una meta" in _std
   and "sin pasar de" not in _mod and f"TECHO ALTO de {fs.techo_moderna(CRIT)} palabras" in _mod,
   "la medida es la respuesta de cada argumento con dato propio; el techo es alto y no es meta")
ok("un argumento menor y sin dato propio" in _std, "RESIDUAL es sólo lo que no trae dato propio")
ok(fs.referencia_estandar([CRIT[0]]) == 2000 and fs.referencia_estandar(CRIT) == 2250
   and fs.referencia_estandar(CRIT * 3) == 2500, "la referencia de la estándar: 2,000-2,500 según problemas vivos")
_m_std = f6.Material(tipo_asunto="amparo_directo", formato="estandar", problemas=PROB, variante="v2")
_corta = "Los conceptos son fundados. " + "palabra " * 1100
ok(any("se quedó" in a.lower() or "ningún engrose real" in a for a in f6.revisar(_corta, CRIT * 2, _m_std)),
   "con seis problemas vivos, 1,100 palabras se avisan (umbral 1,125)")
ok(not any("ningún engrose real" in a for a in f6.revisar(_corta, [CRIT[0]], _m_std)),
   "con uno, no (umbral 1,000: ningún engrose del banco baja de 1,303)")
ok(any("referencia de este formato es de unas 2500" in a
       for a in f6.revisar(_corta, CRIT * 2, _m_std)), "el aviso dice la referencia")
_m_v1 = f6.Material(tipo_asunto="amparo_directo", formato="estandar", problemas=PROB, variante="v1")
ok(f6._objetivo_palabras(_m_v1, CRIT) == f6.PALABRAS_ESTUDIO, "la v1 sigue midiendo contra sus 3,733")

# ═══════════════════════════════════════════════════════════════════════════
print("\n6 · LOS EFECTOS CASAN CON EL CUERPO (en sombra)")
SEGS = [{"id": "C1.a", "concepto": 1, "texto": "La Sala no explicó qué prueba identifica la fracción de dos "
                                               "hectáreas y media dentro de la parcela", "anclas": []},
        {"id": "C1.d", "concepto": 1, "texto": "La reconvención de prescripción positiva por contrato verbal "
                                               "con su padre no se valoró", "anclas": ["art 1143 cc qro"]},
        {"id": "C4.a", "concepto": 4, "texto": "La condena en costas de ambas instancias carece de la hipótesis "
                                               "legal aplicada", "anclas": ["art 769 cpc qro"]}]
_cuerpo = ["Los conceptos son en parte fundados.", "Sobre el primer concepto… Es fundado.",
           "EFECTOS DE LA CONCESIÓN"]
_ef_ok = _cuerpo + ["1. Deje insubsistente la sentencia reclamada.",
                    "2. Precise qué prueba acredita la identidad de la fracción de dos hectáreas y media dentro de la parcela.",
                    "3. Funde la condena en costas de ambas instancias en la hipótesis legal del artículo 769."]
ok(K.efectos_sin_cubrir(_ef_ok, SEGS, CRIT, PROB, {"segmentos": _segs}) == [],
   "cada argumento fundado deja su huella en los efectos; la reconvención infundada no la necesita")
_ef_mal = _cuerpo + ["1. Deje insubsistente la sentencia reclamada.",
                     "2. Precise qué prueba acredita la identidad de la fracción de dos hectáreas y media dentro de la parcela."]
ok([x["id"] for x in K.efectos_sin_cubrir(_ef_mal, SEGS, CRIT, PROB, {"segmentos": _segs})] == ["C4.a"],
   "el fundado de las costas, sin efecto que lo recoja: incongruencia")
ok(K.efectos_sin_cubrir(_cuerpo + ["1. Se concede de manera lisa y llana."], SEGS, CRIT, PROB) == [],
   "con efectos lisos y llanos no se exige nada")
ok(K.efectos_sin_cubrir(_cuerpo[:2], SEGS, CRIT, PROB) == [], "sin efectos, tampoco")
ok([x["concepto"] for x in K.efectos_sin_cubrir(_ef_mal, [], CRIT, PROB)] == [[4]],
   "sin inventario (v2), por concepto con el reparto")
ok(not K.EFECTOS_VISIBLE and "v3 2/14" in K.CALIBRACION_EFECTOS, "va en sombra, con su calibración escrita")

# ═══════════════════════════════════════════════════════════════════════════
print("\n6b · LA REPARACIÓN DIRIGIDA RESPETA LA CALIFICACIÓN DEL ARGUMENTO EN EL PLAN")
import exhaustivo as X
_mat_r = f6.Material(tipo_asunto="amparo_directo", variante="v4", inventario=SEGS, problemas=PROB)
_est_r = "Los conceptos son en parte fundados.\n\n⟦C1.d⟧ Sobre la reconvención, su estudio resulta innecesario."
_falt = [{"id": "C1.d", "parrafo": 1}]
_sin_plan = X.prompt_reparacion(_est_r, CRIT, _mat_r, _falt, "")
_con_plan = X.prompt_reparacion(_est_r, CRIT, _mat_r, _falt, "", plan={"segmentos": _segs})
ok("en el plan" not in _sin_plan, "sin plan (v3), el prompt de la reparación no cambia")
ok("calificación de este argumento en el plan: INFUNDADO" in _con_plan
   and "se contesta con ella: es la suya\n  dentro del sentido de su problema" in _con_plan,
   "con plan (v4): el argumento lleva su calificación del plan como dato, y la regla de usarla")
_POR = {s["id"]: s for s in SEGS}
_REP = X.reparto(CRIT, PROB)
_et = X.etiquetas_del_plan({"segmentos": _segs})
_pieza = "En ese sentido, el argumento de la reconvención es infundado, porque la Sala sí valoró la confesional."
ok(X.guardas(["C1.d"], _pieza, False, _est_r, _POR, _REP).startswith("califica en contra"),
   "sin plan, la guarda de siempre: infundado bajo un problema fundado se descarta")
ok(X.guardas(["C1.d"], _pieza, False, _est_r, _POR, _REP, _et) == "",
   "con el plan que la califica infundada, la pieza pasa (plan-4)")
ok(X.guardas(["C1.d"], "Al dictar la nueva resolución, valore la confesional sobre la reconvención.", True,
             _est_r, _POR, _REP, _et).startswith("nombra en los efectos un argumento que su criterio desestima"),
   "…y no va a los EFECTOS: la responsable no reexamina lo que este tribunal desestimó")
ok(X.guardas(["C1.a"], "En ese sentido, el argumento es infundado, porque la Sala sí explicó la prueba.", False,
             _est_r, _POR, _REP, _et).startswith("califica en contra"),
   "el argumento que el plan califica fundado no puede salir infundado")

# ═══════════════════════════════════════════════════════════════════════════
print("\n7 · EL CAMINO: LOS DOS GEMELOS, `_terminar`, EL «LISTO» Y LA v1")
import redactor_adelanto as ra
_r = types.SimpleNamespace(encargo=types.SimpleNamespace(plan={"estado": "usado", "plan": {"segmentos": _segs}}))
_av1, _me1 = [], {}
_out = ra._congruencia_apertura(_r, CRIT, f6.Material(tipo_asunto="amparo_directo", variante="v4",
                                                      problemas=PROB), V4, _me1, _av1)
ok(_out == _pegado and _me1["congruencia"]["pegadas"] == 4 and _av1 == [],
   "v4: pega las cuatro, lo anota en sombra y no avisa (el texto es del modelo)")
_av2, _me2 = [], {}
_out2 = ra._congruencia_apertura(_r, CRIT, f6.Material(variante="v1", problemas=PROB), V4, _me2, _av2)
ok(_out2 == V4 and _me2 == {} and _av2 == [], "v1: ni una coma")
_av3, _me3 = [], {}
ok(ra._congruencia_apertura(_r, CRIT, f6.Material(variante="v2", problemas=PROB), _sin, _me3, _av3) != _sin
   and _av3 and _av3[0].startswith("CALIFICACIÓN AL ABRIR") and len(_me3["congruencia"]["anadidas"]) == 4,
   "v2 (sin inventario): repara con el criterio y avisa")
SRC_RA = open(os.path.join(AQUI, "redactor_adelanto.py"), encoding="utf-8").read()
ARBOL = ast.parse(SRC_RA)
for nombre in ("resolver", "resolver_en_vivo"):
    fn = next(n for n in ARBOL.body if isinstance(n, ast.AsyncFunctionDef) and n.name == nombre)
    src = ast.get_source_segment(SRC_RA, fn)
    llamadas = [getattr(n.func, "id", "") for n in ast.walk(fn) if isinstance(n, ast.Call)]
    ok(llamadas.count("_congruencia_apertura") == 1
       and src.index("_completar_estudio(") < src.index("_congruencia_apertura(") < src.index("_terminar("),
       f"{nombre}: una vez, después de la reparación dirigida y antes de componer")
    ok(llamadas.count("_congruencia_pegar") == 1
       and src.index("_congruencia_pegar(") < src.index("_por_completar(") < src.index("_completar_estudio("),
       f"{nombre}: la calificación suelta se pega ANTES de la reparación dirigida")
# POR QUÉ ANTES (revisión adversarial, 26-sep-2026): la reparación dirigida
# inserta la pieza tras el último párrafo que marca un argumento de su
# concepto; si ése es la apertura, sin pegar antes la pieza queda entre la
# apertura y su «Es fundado.», y el pegado posterior se la da a la pieza.
import exhaustivo as X_ins
_ins = ("⟦C1.a⟧ Sobre el primer concepto de violación, en el que sostiene que faltó la pericial.\n\n"
        "Es fundado.\n\n"
        "Lo anterior, porque la Sala no explicó qué prueba identifica la fracción reclamada.")
_pz = [(["C1.b"], "En cuanto a la reconvención, la Sala sí examinó la confesional y la testimonial ofrecidas.")]
_spi = {"C1.a": {"id": "C1.a", "concepto": 1}, "C1.b": {"id": "C1.b", "concepto": 1}}
_tarde = K.pegar_calificaciones(X_ins.insertar(_ins, _pz, [], _spi)[0])[0]
_pronto = X_ins.insertar(K.pegar_calificaciones(_ins)[0], _pz, [], _spi)[0]
ok(_tarde.split("\n")[0].rstrip().endswith("faltó la pericial.") and "ofrecidas. Es fundado." in _tarde,
   "pegando DESPUÉS de la reparación, la calificación de la apertura acaba en la pieza insertada")
ok(_pronto.split("\n")[0].rstrip().endswith("faltó la pericial. Es fundado.") and "ofrecidas. Es fundado." not in _pronto,
   "pegando ANTES, la apertura conserva su calificación y la pieza va detrás")
_me6, _av6 = {}, []
_m6 = f6.Material(tipo_asunto="amparo_directo", variante="v4", problemas=PROB)
_p6 = ra._congruencia_pegar(_m6, V4, _me6)
_o6 = ra._congruencia_apertura(_r, CRIT, _m6, _p6, _me6, _av6)
ok(_p6 == _pegado and _o6 == _pegado and _me6["congruencia"]["pegadas"] == 4 and "_cg_pegadas" not in _me6
   and _av6 == [], "lo pegado antes cuenta en el informe, sin dejar rastro interno en la ficha")
_me7 = {}
ok(ra._congruencia_pegar(f6.Material(variante="v1", problemas=PROB), V4, _me7) == V4 and _me7 == {},
   "v1: el pegado previo tampoco toca una coma")
_term = next(n for n in ARBOL.body if isinstance(n, ast.AsyncFunctionDef) and n.name == "_terminar")
_st = ast.get_source_segment(SRC_RA, _term)
ok("_congruencia_efectos(e, material, estudio, criterios, meta_estudio)" in _st
   and _st.index("_marcas_y_cobertura(") < _st.index("_congruencia_efectos(") < _st.index("ens.Relleno("),
   "`_terminar`: los efectos se miran sobre el texto sin marcas, antes de componer")
_me4 = {"congruencia": {"pegadas": 1}}
ra._congruencia_efectos(types.SimpleNamespace(plan={"plan": {"segmentos": _segs}}),
                        f6.Material(variante="v3", inventario=SEGS, problemas=PROB),
                        "\n".join(_ef_mal), CRIT, _me4)
ok([x["id"] for x in _me4["congruencia"]["efectos_sin_cubrir"]] == ["C4.a"] and _me4["congruencia"]["pegadas"] == 1,
   "`_congruencia_efectos` anota en sombra junto a lo demás")
_me5 = {}
ra._congruencia_efectos(types.SimpleNamespace(plan={}), f6.Material(variante="v1"), "\n".join(_ef_mal), CRIT, _me5)
ok(_me5 == {}, "…y en la v1, nada")
ok("plan=_pl if isinstance(_pl, dict) else None" in ast.get_source_segment(
    SRC_RA, next(n for n in ARBOL.body if isinstance(n, ast.AsyncFunctionDef) and n.name == "_completar_estudio")),
   "`_completar_estudio` pasa el plan del encargo a la reparación")
SRC_MAIN = open(os.path.join(AQUI, "main.py"), encoding="utf-8").read()
ok('fuera["congruencia"] = dict(_m["congruencia"])' in SRC_MAIN, "el «listo» y la ficha la llevan (sólo si la hay)")

print()
if FALLOS:
    print(f"FALLAN {len(FALLOS)}: " + " · ".join(FALLOS))
    raise SystemExit(1)
print("RESULTADO: TODAS LAS COMPROBACIONES PASAN")
