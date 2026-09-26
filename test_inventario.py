"""El inventario de argumentos y la v3 del prompt (Paso 2a) — 26-sep-2026.

La v2 del estudio contesta con razón propia menos argumentos autónomos que la
v1 porque, al acortar sin saber qué argumentos hay, los funde. El remedio
aprobado por David: el estudio recibe la lista de argumentos (`inventario.py`)
y marca dónde contesta cada uno (`marcas.py`). Esto comprueba:

  1 · el escrito sin la maqueta del PDF, y la cita que se verifica contra él;
  2 · el resumen partido en argumentos, con los patrones medidos en las
      sesiones reales (verbos de alegación, «; además, afirma», «, y solicita»,
      «así como» + infinitivo, «que A y que B», el párrafo que sólo apoya), sin
      trozos sueltos y con las anomalías anotadas;
  3 · los segmentos del contrato: identificador, concepto, párrafo, texto de
      80 palabras como mucho, cita literal de 10 a 40 palabras o vacía, anclas
      sin los datos del asunto; nunca lanza;
  4 · el bloque del prompt: datos, sin frases que copiar;
  5 · las variantes: la v1 y la v2 IDÉNTICAS (instantánea de la v2 sacada de
      origin/main con su propio código); la v3 es la v2 más dos piezas y nada
      más; la v4 sin plan es la v3;
  6 · el camino: el inventario viaja por el material sólo en v3/v4, las marcas
      no llegan a la pantalla, `_terminar` las separa lo primero, el aviso
      visible sólo sale si falta la marca y no hay rastro, y el evento «listo»
      y la ficha llevan el mapa y la cobertura (y la v1/v2 ni un campo más).

    .venv/bin/python test_inventario.py
"""
import ast
import asyncio
import json
import os
import re
import types

os.environ.pop("ESTUDIO_PROMPT", None)
os.environ.pop("ESTUDIO_PROMPT_AD", None)

import fase6_estudio as f6
import fases123_pipeline as f123
import inventario as inv
import marcas as mc

FALLOS = []
AQUI = os.path.dirname(os.path.abspath(__file__))


def ok(cond, que):
    print(f"   {'PASA ' if cond else 'FALLA'}  {que}")
    if not cond:
        FALLOS.append(que)


# ═══════════════════════════════════════════════════════════════════════════
# UN ASUNTO DE PRUEBA, inventado, con la forma de los reales
# ═══════════════════════════════════════════════════════════════════════════
RESUMEN = """En contra de esas consideraciones, la parte quejosa plantea los siguientes conceptos de violación:

En el primer concepto de violación aduce que la Sala responsable transgrede el artículo 14 constitucional, porque tiene por acreditada la identidad del inmueble sin la prueba pericial en topografía que exige la acción reivindicatoria. [[p.6 §2]] Refiere que las superficies de cuatro hectáreas y de dos hectáreas y media que se mencionan en autos son contradictorias entre sí. Asimismo, sostiene que la Sala omite analizar la reconvención de prescripción positiva y las pruebas confesional y testimonial; además, afirma que la testimonial acredita la posesión derivada del contrato verbal celebrado con su padre.

Para apoyar esa postura, invoca la jurisprudencia de registro 2009789 sobre la identidad del bien reivindicado.

En el segundo concepto de violación afirma que lo resuelto en el expediente 1114/2017 produce cosa juzgada refleja, porque en aquel juicio se desestimó la misma acción por falta de identidad del predio. Por ello, sostiene que la Sala debió analizar de oficio la cosa juzgada refleja y declarar improcedente la acción.

Finalmente, en el tercer concepto de violación aduce que la condena en costas es ilegal porque actuó sin dolo ni mala fe en ninguna de las instancias, y solicita que se considere su condición de campesino que no sabe leer ni escribir."""

RENGLON = 80


def _a_columna(texto: str) -> str:
    """El escrito como llega del PDF: a 80 columnas, la palabra partida sin
    guion donde acaba el renglón y el resto relleno de espacios."""
    plano = " ".join(texto.split())
    fuera = []
    while plano:
        fuera.append(plano[:RENGLON])
        plano = plano[RENGLON:]
    return "\n".join(x.ljust(RENGLON) if len(x) < RENGLON else x for x in fuera)


CABECERA = ("TOCA CIVIL 601/2023. QUEJOSO: JUAN PÉREZ. Sentencia de diecinueve de agosto de "
            "dos mil veinticuatro dictada en el juicio 1414/2020. " + "Datos de trámite. " * 60)
C1 = ("PRIMER CONCEPTO DE VIOLACIÓN. La responsable tuvo por acreditada la identidad sin "
      "que exista en autos la prueba pericial en materia de topografía, que es la prueba "
      "idónea para acreditar la identidad de un bien en la acción reivindicatoria. "
      "Las superficies son contradictorias entre sí: se habla de cuatro hectáreas en la "
      "reconvención y de dos hectáreas y media en la demanda. "
      "La Sala omite analizar la reconvención de prescripción positiva, así como las "
      "pruebas confesional y testimonial ofrecidas en el juicio natural. "
      "La testimonial acredita la posesión derivada del contrato verbal celebrado con mi "
      "difunto padre. Sirve de apoyo la jurisprudencia de registro 2009789. ")
C2 = ("SEGUNDO CONCEPTO DE VIOLACIÓN. Lo resuelto en el diverso expediente 1114/2017 "
      "produce cosa juzgada refleja, porque en aquel juicio se desestimó la misma acción "
      "reivindicatoria por falta de identidad del predio. La Sala debió analizar de oficio "
      "la cosa juzgada refleja y declarar improcedente la acción reivindicatoria. ")
C3 = ("TERCER CONCEPTO DE VIOLACIÓN. La condena en costas es ilegal, pues el suscrito "
      "actuó sin dolo ni mala fe en ninguna de las instancias; soy campesino, no sé leer ni "
      "escribir y mi condición es vulnerable. Por lo expuesto, pido se sirva conceder. ")
ESCRITO_LIMPIO = CABECERA + C1 + C2 + C3
ESCRITO = _a_columna(ESCRITO_LIMPIO)
_i1, _i2, _i3 = (ESCRITO_LIMPIO.find(x) for x in ("PRIMER CONCEPTO", "SEGUNDO CONCEPTO", "TERCER CONCEPTO"))


def _en_crudo(pos_limpio: int) -> int:
    # Cada renglón de 80 caracteres del limpio es uno de 81 en el crudo (con el salto).
    return pos_limpio + pos_limpio // RENGLON


CONTEO = {"estado": "contado", "n": 3, "valores": [1, 2, 3],
          "tramos": [[_en_crudo(_i1), _en_crudo(_i2)], [_en_crudo(_i2), _en_crudo(_i3)],
                     [_en_crudo(_i3), len(ESCRITO)]]}
FASES = f123.Fases123(resumen_conceptos=RESUMEN, conteo=CONTEO)

# ═══════════════════════════════════════════════════════════════════════════
print("\n1 · EL ESCRITO SIN LA MAQUETA DEL PDF")
norm, pos = inv.normalizar_escrito(ESCRITO)
ok(norm == " ".join(ESCRITO_LIMPIO.split()), "renglones a 80 columnas: las palabras partidas se unen "
   "y los blancos se colapsan (el normalizado es el texto limpio)")
ok(len(pos) == len(norm) and all(ESCRITO[pos[k]] == norm[k] for k in range(len(norm)) if norm[k] != " "),
   "cada carácter del normalizado sabe de dónde viene en el crudo")
n2, _ = inv.normalizar_escrito("la pose-\nsión del bien\n\n  y   más")
ok(n2 == "la posesión del bien y más", "el guion de fin de renglón entre minúsculas se quita")
ok(inv.normalizar_escrito("")[0] == "" and inv.normalizar_escrito(None)[0] == "", "vacío o None")
ok(inv.verificar_cita("la prueba idónea para acreditar la identidad de un bien", ESCRITO),
   "una cita del escrito se verifica aunque en el crudo esté partida entre renglones")
ok(inv.verificar_cita("LA PRUEBA IDONEA para acreditar", ESCRITO), "sin distinguir mayúsculas ni acentos")
ok(not inv.verificar_cita("la prueba idónea para acreditar la propiedad", ESCRITO),
   "una cita que no está no se verifica")
ok(not inv.verificar_cita("la prueba", ESCRITO), "menos de tres palabras no es cita")

# ═══════════════════════════════════════════════════════════════════════════
print("\n2 · EL RESUMEN, PARTIDO EN ARGUMENTOS")
piezas, anom = inv.piezas_del_resumen(FASES)
textos = [p["texto"] for p in piezas]
ok(len(piezas) == 8, f"ocho argumentos en tres conceptos ({len(piezas)})")
ok([p["concepto"] for p in piezas] == [1, 1, 1, 1, 2, 2, 3, 3] and anom == [],
   "el concepto sale del ordinal que escribe el resumen; sin anomalías")
ok(not any("En contra de esas consideraciones" in t for t in textos), "la bisagra no es argumento")
ok(textos[1].startswith("Refiere que las superficies"), "una frase con verbo de alegación abre argumento")
ok(textos[2].startswith("Asimismo, sostiene que la Sala omite") and textos[2].endswith("testimonial")
   and textos[3].startswith("Además, afirma que la testimonial"),
   "«; además, afirma que…» parte la frase y el conector va con el trozo nuevo")
ok("registro 2009789" in textos[3] and piezas[3]["parrafo"] == 1,
   "el párrafo que sólo apoya («Para apoyar esa postura, invoca…») va con el argumento anterior")
ok(textos[5].startswith("Por ello, sostiene que la Sala debió analizar de oficio"),
   "«Por ello, sostiene…» con contenido propio es argumento")
ok(textos[6].startswith("Finalmente, en el tercer concepto") and textos[7].startswith(
    "Solicita que se considere su condición de campesino")
   and textos[7].endswith("leer ni escribir."), "«…, y solicita que…» parte la frase")
ok("[[" not in " ".join(textos) and piezas[0]["paginas"] == ["p.6 §2"],
   "la nota [[p.6 §2]] sale del texto y queda como página del primer argumento")
ok([p["parrafo"] for p in piezas] == [1, 1, 1, 1, 3, 3, 4, 4],
   "el párrafo es el índice en `resumen_conceptos` (renglones no vacíos)")
por_parrafo, _ = inv.piezas_del_resumen(FASES, partir=False)
ok(len(por_parrafo) == 3, "sin partir (para medir): un segmento por párrafo, el de apoyo dentro")
fr = inv._frases("No identifica el departamento 402 del edificio D. Sostiene que la inscripción "
                 "es un presupuesto. El C. Juan Pérez declaró.")
ok(len(fr) == 3 and fr[1].startswith("Sostiene"),
   "la letra suelta con punto cierra la frase si lo que sigue abre argumento (722/2025)")
ok(inv._partir_dentro("Refiere que el monto se fija al dictar la sentencia principal, que la "
                      "presunción legal no fue destruida y que durante el matrimonio el contrario "
                      "adquirió tres bienes inmuebles distintos.")[1].startswith("Refiere que durante"),
   "«que A y que B»: el trozo nuevo lleva el verbo de la frase")
ok(inv._partir_dentro("Señala que la Sala omite analizar la temporalidad de las adquisiciones, así "
                      "como pronunciarse sobre la valorización del tercer inmueble exhibido.")[1]
   .startswith("Así como pronunciarse"),
   "«así como» + infinitivo: dos omisiones, dos argumentos (174/2026)")
ok(len(inv._partir_dentro("Señala que la Sala deja de examinar las inconsistencias del testigo, así "
                          "como las declaraciones del comisariado ejidal y los colindantes.")) == 1,
   "«así como» + sustantivo: la misma omisión, no se parte")
ok(inv._partir_dentro("Afirma que ejerce los medios de defensa sin dolo, y solicita que se "
                      "considere su condición y que no sabe leer.") ==
   ["Afirma que ejerce los medios de defensa sin dolo",
    "Solicita que se considere su condición y que no sabe leer."],
   "un corte que dejaría menos de ocho palabras no se hace (642/2024)")
# Anomalías y casos raros medidos en las sesiones.
f_n1 = f123.Fases123(resumen_conceptos="Bisagra:\nSostiene que A es ilegal por muchas razones distintas.\n"
                                       "Alega que B también es ilegal por otras razones.",
                     conteo={"estado": "contado", "n": 1})
p_n1, a_n1 = inv.piezas_del_resumen(f_n1)
ok([p["concepto"] for p in p_n1] == [1, 1] and a_n1 == [], "n=1 sin ordinales: todo es del concepto 1")
f_n0 = f123.Fases123(resumen_conceptos="Sostiene que A es ilegal por muchas razones distintas.\n"
                                       "Alega que B también es ilegal por otras razones.",
                     conteo={"estado": "no_contado"})
p_n0, a_n0 = inv.piezas_del_resumen(f_n0)
ok([p["concepto"] for p in p_n0] == [1, 2] and any("sin ordinales" in a for a in a_n0),
   "sin contar y sin ordinales: el orden de párrafos (contrato), y se anota")
f_mas = f123.Fases123(resumen_conceptos="En el primer agravio sostiene que A es ilegal por mucho.\n"
                                        "En el tercer agravio alega que C es ilegal por otras razones.",
                      conteo={"estado": "contado", "n": 1})
p_mas, a_mas = inv.piezas_del_resumen(f_mas)
ok([p["concepto"] for p in p_mas] == [1, 3]
   and any("nombra el 3.º y el contador contó 1" in a for a in a_mas)
   and any("no nombra el 2.º" in a for a in a_mas),
   "manda el ordinal del resumen; se anotan el desacuerdo con el contador y el hueco (308/2026)")
ok(inv.piezas_del_resumen(f123.Fases123())[0] == [], "resumen vacío: ningún argumento")

# ═══════════════════════════════════════════════════════════════════════════
print("\n3 · LOS SEGMENTOS DEL CONTRATO")
segs = inv.segmentos(FASES, ESCRITO)
ok([s["id"] for s in segs] == ["C1.a", "C1.b", "C1.c", "C1.d", "C2.a", "C2.b", "C3.a", "C3.b"],
   "identificadores: C + concepto + letra por argumento")
ok(all(set(s) >= {"id", "concepto", "parrafo", "texto", "cita", "anclas"} for s in segs),
   "cada segmento trae los campos del contrato")
ok(all(len(s["texto"].split()) <= inv.MAX_PALABRAS_TEXTO for s in segs), "texto de 80 palabras como mucho")
ok(all(s["cita"] for s in segs), "todos anclados en el escrito")
ok(all(inv.CITA_MIN <= len(s["cita"].split()) <= inv.CITA_MAX for s in segs if s["cita"]),
   "cada cita, de 10 a 40 palabras")
ok(all(inv.verificar_cita(s["cita"], ESCRITO) for s in segs), "cada cita se verifica literal en el escrito")
ok(all(" ".join(s["cita"].split()) in norm for s in segs), "y es un pasaje del escrito normalizado, sin palabras partidas")
_pos = {s["id"]: norm.find(" ".join(s["cita"].split())) for s in segs}
_n1, _n2, _n3 = (norm.find(x) for x in ("PRIMER CONCEPTO", "SEGUNDO CONCEPTO", "TERCER CONCEPTO"))
ok(all(_n1 - 400 <= _pos[i] < _n2 + 400 for i in ("C1.a", "C1.b", "C1.c", "C1.d"))
   and all(_n2 - 400 <= _pos[i] < _n3 + 400 for i in ("C2.a", "C2.b")) and _pos["C3.a"] >= _n3 - 400,
   "cada cita cae en el tramo de su concepto")
ok("pericial" in segs[0]["cita"].lower() and "cosa juzgada refleja" in segs[4]["cita"].lower()
   and "dolo" in segs[6]["cita"].lower() and "campesino" in segs[7]["cita"].lower(),
   "y es el pasaje del argumento, no uno cualquiera del tramo")
# LA CITA ES EL PASAJE DE SU PROPIO ARGUMENTO, NO EL DEL VECINO (revisión
# adversarial, 26-sep-2026). Con la ventana cortada a 40 palabras desde su
# primer acierto, «omite analizar la reconvención» quedaba anclado en el pasaje
# de la pericial, el de las superficies se cortaba antes de las hectáreas y el
# cuarto argumento heredaba el expediente del segundo concepto.
_propio = {"C1.b": "cuatro hectáreas", "C1.c": "omite analizar la reconvención",
           "C1.d": "contrato verbal", "C2.b": "analizar de oficio la cosa juzgada"}
ok(all(frase in s["cita"] for s in segs for i, frase in _propio.items() if s["id"] == i),
   "cada cita es el pasaje de SU argumento: " + " · ".join(
       f"{s['id']}={'sí' if _propio[s['id']] in s['cita'] else 'NO'}" for s in segs if s["id"] in _propio))
ok("pericial" not in segs[2]["cita"] and "SEGUNDO CONCEPTO" not in segs[3]["cita"],
   "sin arrastrar el pasaje del argumento anterior ni el del concepto siguiente")
ok("exp 1114/2017" not in segs[3]["anclas"],
   "el argumento no hereda como dato el expediente del concepto siguiente")
ok("exp 1114/2017" in segs[4]["anclas"] and "reg 2009789" in segs[3]["anclas"],
   "anclas duras del argumento: el expediente y el registro")
ok(not any("exp 1414/2020" in s["anclas"] or "exp 601/2023" in s["anclas"] for s in segs),
   "los datos del propio asunto (en el encabezado del escrito) no son anclas")
ok(segs[0].get("pagina") == "p.6 §2", "la página del resumen viaja con el primer argumento del párrafo")
sin_escrito = inv.segmentos(FASES, "")
ok(len(sin_escrito) == 8 and all(s["cita"] == "" for s in sin_escrito),
   "sin escrito: los mismos argumentos, con la cita vacía (nunca inventada)")
ajeno = inv.segmentos(FASES, _a_columna("Texto de otro asunto sobre arrendamiento de oficinas. " * 80))
ok(all(s["cita"] == "" for s in ajeno), "con un escrito que no casa, ninguna cita")
ok(inv.segmentos(None, ESCRITO) == [] and inv.segmentos({"resumen_conceptos": ""}, ESCRITO) == [],
   "sin fases o sin resumen: [] y sin lanzar")
ok([s["id"] for s in inv.segmentos({"resumen_conceptos": RESUMEN, "conteo": CONTEO}, ESCRITO)] ==
   [s["id"] for s in segs], "acepta las fases como dict (como vienen de la sesión)")
ok(inv.segmentos(FASES, ESCRITO, es_recurso=True)[0]["id"] == "A1.a", "en un recurso, A de agravio")
ok(inv._letra(0) == "a" and inv._letra(25) == "z" and inv._letra(26) == "aa" and inv._letra(27) == "ab",
   "más de 26 argumentos en un concepto: aa, ab… (489/2026 tiene 72)")
ok(inv.anclas_duras("el artículo 1635 del Código Civil Federal y la tesis de registro 2009789")
   == ["art 1635 ccf", "reg 2009789"], "anclas normalizadas, en orden")
ok(inv.anclas_duras("el artículo 1635 del Código Civil Federal", excluir={"art 1635 ccf"}) == [],
   "`excluir` quita las del asunto")

# ═══════════════════════════════════════════════════════════════════════════
print("\n4 · EL BLOQUE DEL PROMPT: DATOS, SIN FRASES QUE COPIAR")
blq = inv.bloque_inventario(segs, "concepto de violación")
lineas = [x for x in blq.split("\n") if re.match(r"^C\d+\.[a-z]+ · ", x)]
ok(len(lineas) == len(segs), "un renglón por argumento")
ok(all(s["id"] in blq for s in segs) and "cita: «" in blq and "datos: exp 1114/2017" in blq,
   "con el identificador, la cita y los datos duros")
ok("primer concepto de violación" in lineas[0] and "tercer concepto de violación" in lineas[-1],
   "cada renglón dice a qué concepto pertenece")
_cabeza = blq.split(lineas[0])[0]
ok("«" not in _cabeza and not re.search(r"\b[CA]\d+\.[a-z]\b", _cabeza),
   "la cabecera describe los campos: sin comillas ni identificadores de ejemplo")
ok(inv.bloque_inventario([], "concepto de violación") == "", "sin argumentos, sin bloque")

# ═══════════════════════════════════════════════════════════════════════════
print("\n5 · LAS VARIANTES: v1 Y v2 IDÉNTICAS, v3 = v2 + INVENTARIO, v4 SIN PLAN = v3")
ok(f6.VARIANTES == ("v1", "v2", "v3", "v4"), "cuatro variantes")
ok(f6.normalizar_variante("v3") == "v3" and f6.normalizar_variante("V4") == "v4"
   and f6.normalizar_variante("C-lite") == "v3" and f6.normalizar_variante("c") == "v4"
   and f6.normalizar_variante("v9", "v1") == "v1", "se reconocen v3, v4 y los nombres de la propuesta")
ok(all(f6._v2(types.SimpleNamespace(variante=v)) for v in ("v2", "v3", "v4"))
   and not f6._v2(types.SimpleNamespace(variante="v1")), "la v3 y la v4 son de la familia de la v2")
ok([f6.con_inventario(types.SimpleNamespace(variante=v)) for v in f6.VARIANTES] == [False, False, True, True],
   "sólo la v3 y la v4 reciben el inventario")
ok(f6.variante_global() == "v1" and f6.variante_global("amparo_directo") == "v1",
   "sin variables, la global es la v1 (no se enciende nada en esta entrega)")
os.environ["ESTUDIO_PROMPT_AD"] = "v3"
ok(f6.variante_global("amparo_directo") == "v3" and f6.variante_global("queja") == "v1"
   and f6.variante_global() == "v1", "ESTUDIO_PROMPT_AD enciende sólo el amparo directo")
os.environ["ESTUDIO_PROMPT_AD"] = "tonteria"
ok(f6.variante_global("amparo_directo") == "v1", "un valor que no existe: la global")
os.environ.pop("ESTUDIO_PROMPT_AD", None)

# La instantánea de la v2, sacada de origin/main (de66e3d) con SU código, con
# el criterio del 93 que usa test_prompt_v2.py para la de la v1.
_CRIT = [
    {"problema": "¿La Sala debió admitir la ampliación de demanda y estudiar los argumentos "
                 "dirigidos contra el crédito fiscal?",
     "sentido": "fundado",
     "razonamiento": "La interlocutoria confirmó la preclusión exclusivamente porque reputó "
                     "aplicable el término sumario de cinco días, pese a que el acuerdo de primero "
                     "de julio dejó a salvo la ampliación con referencia al artículo 17 y a la "
                     "jurisprudencia citada. Al no esclarecer ni respetar ese contenido, cerró el "
                     "debate sobre el crédito y afectó la congruencia y exhaustividad.",
     "jerarquia": "principal"},
    {"problema": "¿La Sala debía pronunciarse sobre los argumentos formulados por la actora en "
                 "sus alegatos?",
     "sentido": "innecesario",
     "razonamiento": "Dado el sentido del estudio del problema principal, queda sin materia el "
                     "análisis de este planteamiento: La nueva sentencia deberá pronunciarse sobre "
                     "los alegatos en la medida en que la ampliación incorpore el crédito fiscal y "
                     "sus cuestiones conexas.",
     "jerarquia": "accesorio"},
]
C93 = [f6.Criterio(problema=d["problema"], sentido=d["sentido"], razonamiento=d["razonamiento"],
                   jerarquia=d["jerarquia"]) for d in _CRIT]
P93 = [{"pregunta": C93[0].problema, "cubre": [1]}, {"pregunta": C93[1].problema, "cubre": [2]}]
ACTO = "LO QUE RESOLVIÓ LA SALA (resumen de prueba)."
CONC = "LO QUE SE COMBATE (resumen de prueba)."
ESC93 = ("PRIMER CONCEPTO DE VIOLACIÓN. " + "La Sala no admitió la ampliación de la demanda. " * 20)


def mat(formato, variante, tipo="amparo_directo", **kw):
    return f6.Material(tipo_asunto=tipo, materia="administrativa", formato=formato,
                       problemas=P93, n_planteamientos=2, variante=variante, **kw)


for forma in ("estandar", "moderna"):
    for etiq, tipo, rec, esc in (("93", "amparo_directo", False, ESC93), ("rf", "revision_fiscal", True, "")):
        ruta = os.path.join(AQUI, "datos", "instantaneas", f"estudio_v2_{etiq}_{forma}.prompt")
        antes = open(ruta, encoding="utf-8").read()
        ahora = f6.prompt_estudio(ACTO, CONC, C93, mat(forma, "v2", tipo), es_recurso=rec,
                                  escrito_literal=esc)
        ok(ahora == antes, f"v2 {etiq}/{forma}: idéntica, byte por byte, a la de origin/main")
        con_inv = f6.prompt_estudio(ACTO, CONC, C93, mat(forma, "v2", tipo, inventario=segs),
                                    es_recurso=rec, escrito_literal=esc)
        ok(con_inv == antes, f"v2 {etiq}/{forma}: aunque el material traiga inventario, la v2 no lo usa")
        ok(f6.prompt_estudio(ACTO, CONC, C93, mat(forma, "v3", tipo), es_recurso=rec,
                             escrito_literal=esc) == antes,
           f"v3 {etiq}/{forma} sin inventario: escribe como la v2")
_v1 = f6.prompt_estudio(ACTO, CONC, C93, mat("estandar", "v1"))
ok(f6.prompt_estudio(ACTO, CONC, C93, mat("estandar", "v1", inventario=segs)) == _v1
   and "INVENTARIO DE ARGUMENTOS" not in _v1 and "⟦" not in _v1,
   "v1: el inventario en el material no la toca (la v1 sigue en su instantánea: test_prompt_v2.py)")

p2 = f6.prompt_estudio(ACTO, CONC, C93, mat("estandar", "v2"), escrito_literal=ESC93)
p3 = f6.prompt_estudio(ACTO, CONC, C93, mat("estandar", "v3", inventario=segs), escrito_literal=ESC93)
bloque, recordatorio = f6._partes_v3(mat("estandar", "v3", inventario=segs))
ok(bloque and recordatorio and p3.count(bloque) == 1 and p3.count(recordatorio) == 1,
   "la v3 lleva el bloque del inventario y el recordatorio, una vez cada uno")
ok(p3.replace(bloque, "", 1).replace(recordatorio, "", 1) == p2,
   "la v3 es la v2 más esas dos piezas y NADA MÁS")
ok(p3.index(bloque) > p3.index("EL ESCRITO DE LA PARTE, LITERAL")
   and p3.index(bloque) < p3.index("Escribe el estudio de fondo."),
   "el inventario va después del escrito y antes de la orden de escribir")
ok(p3.index(recordatorio) > p3.index("Escribe el estudio de fondo.")
   and p3.index(recordatorio) < p3.index("NO ESCRIBAS LA FÓRMULA FINAL"),
   "el recordatorio, entre lo último que lee el modelo")
ok(f"({len(segs)})" in recordatorio, "el recordatorio dice cuántos argumentos hay")
regla = f6._regla_marcas("concepto de violación")
ok("«" not in regla and "»" not in regla and not re.search(r"\b[CAMU]\d+(?:\.[a-z])?\b", regla),
   "la regla de las marcas no trae frases entre comillas ni identificadores de ejemplo")
_regla1 = " ".join(regla.split())
ok("⟦" in regla and "⟧" in regla and "separados por un espacio" in _regla1
   and "al comienzo del párrafo" in _regla1, "describe la marca: identificadores, espacio, ⟦ ⟧, al comienzo")
ok("no escribas identificadores" in regla.lower() or "Fuera de la marca no escribas" in regla,
   "prohíbe los identificadores en la prosa")
ok("remisión con contenido" in regla and "nombran juntos" in regla and "dato propio" in regla,
   "la regla aprobada: respuesta con su dato propio, remisión con contenido, reiterados juntos")
p4 = f6.prompt_estudio(ACTO, CONC, C93, mat("estandar", "v4", inventario=segs), escrito_literal=ESC93)
ok(p4 == p3, "la v4 sin plan escribe como la v3 (el guion lo trae la pieza del plan)")
pr = f6.prompt_estudio(ACTO, CONC, C93, mat("estandar", "v3", "revision_fiscal",
                                            inventario=inv.segmentos(FASES, ESCRITO, es_recurso=True)),
                       es_recurso=True)
ok("primer agravio" in pr and "A1.a" in pr, "en un recurso el bloque dice «agravio» y los ids son A")

# ═══════════════════════════════════════════════════════════════════════════
print("\n6 · EL CAMINO: MATERIAL, PANTALLA, _terminar Y EL EVENTO «listo»")
import redactor_adelanto as ra

r = types.SimpleNamespace()
r.fases = f123.Fases123(resumen_conceptos=RESUMEN, conteo=CONTEO, fuentes=["acto", ESCRITO])
r.encargo = types.SimpleNamespace(formato="", variante_estudio="v3", tipo_asunto="amparo_directo",
                                  es_recurso=False)
m = f6.Material()
ra._formato_al_material(r, m, None, C93)
ok(m.variante == "v3" and [s["id"] for s in m.inventario] == [s["id"] for s in segs],
   "v3: `_formato_al_material` arma el inventario con el escrito de la sesión (`fuentes[1]`)")
r.encargo.variante_estudio = "v2"
ra._formato_al_material(r, m, None, C93)
ok(m.variante == "v2" and m.inventario == [],
   "la vuelta siguiente en v2 lo vacía: el material vive en el worker y no se cuela")
r.encargo.variante_estudio = "v3"
r.fases.resumen_conceptos = None
ra._formato_al_material(r, m, None, C93)
ok(m.inventario == [], "un resumen que no se puede leer deja el inventario vacío, sin lanzar")
r.fases.resumen_conceptos = RESUMEN

# La pantalla: los trozos del flujo pasan por el filtro.
ESTUDIO_MARCADO = ("Los conceptos de violación son infundados.\n"
                   "⟦C1.a C1.b⟧ Sobre el primer concepto de violación, la identidad se acreditó "
                   "sin pericial en topografía y las superficies no se contradicen.\n"
                   "⟦C1.c C1.d⟧ La Sala sí analizó la reconvención, la confesional y la testimonial.\n"
                   "⟦C2.a C2.b⟧ Sobre el segundo concepto de violación, lo resuelto en el expediente "
                   "1114/2017 no produce cosa juzgada refleja.")
_llamadas = {}


async def _vivo_falso(*a, **kw):
    for i in range(0, len(ESTUDIO_MARCADO), 7):
        yield {"tipo": "texto", "dato": ESTUDIO_MARCADO[i:i + 7]}
    yield {"tipo": "fin", "estudio": ESTUDIO_MARCADO, "advertencias": "", "avisos": [], "meta": {"variante": "v3"}}


async def _terminar_falso(cliente, r_, e_, criterios, material, estudio, advertencias, avisos, *a, **kw):
    _llamadas["estudio"] = estudio
    _llamadas["meta"] = kw.get("meta_estudio")
    return "RESULTADO"


_orig = (ra.f6.redactar_en_vivo, ra._terminar, ra._litis_y_material)
ra.f6.redactar_en_vivo, ra._terminar = _vivo_falso, _terminar_falso
ra._litis_y_material = lambda *a, **k: []
r.encargo.propuesta_global = None
r.encargo.conceptos_violacion = ""
r.encargo.resolvio_declarado = ""
r.partes = None


async def _correr():
    textos, fin = [], None
    async for paso in ra.resolver_en_vivo(None, r, C93, f6.Material(variante="v3"), "/tmp/x.docx"):
        if paso["tipo"] == "texto":
            textos.append(paso["dato"])
        elif paso["tipo"] == "listo":
            fin = paso
    return textos, fin


try:
    _textos, _fin = asyncio.run(_correr())
finally:
    ra.f6.redactar_en_vivo, ra._terminar, ra._litis_y_material = _orig
_pantalla = "".join(_textos)
ok("⟦" not in _pantalla and "⟧" not in _pantalla and not any("C1.a" in t for t in _textos),
   "en vivo: ninguna marca llega a la pantalla, ni partida entre trozos")
ok(_pantalla == mc.sin_marcas(ESTUDIO_MARCADO), "lo que se ve escribiéndose es el texto limpio")
ok(_llamadas.get("estudio") == ESTUDIO_MARCADO and _fin and _fin["resultado"] == "RESULTADO",
   "`_terminar` recibe el estudio CON las marcas: es él quien las separa y guarda el mapa")

# `_terminar`: las separa lo primero, en los dos gemelos.
SRC_RA = open(os.path.join(AQUI, "redactor_adelanto.py"), encoding="utf-8").read()
ARBOL_RA = ast.parse(SRC_RA)
term = next(n for n in ARBOL_RA.body if isinstance(n, ast.AsyncFunctionDef) and n.name == "_terminar")
_primeras = [n for n in term.body if not (isinstance(n, ast.Expr) and isinstance(getattr(n, "value", None), ast.Constant))]
_src_prim = "\n".join(ast.get_source_segment(SRC_RA, n) for n in _primeras[:5])
ok("_marcas_y_cobertura(" in _src_prim and "import litis_normativa" not in _src_prim.split("_marcas_y_cobertura(")[0],
   "`_terminar` separa las marcas ANTES de la litis, el relleno y el .docx")
for nombre in ("resolver", "resolver_en_vivo"):
    fn = next(n for n in ARBOL_RA.body if isinstance(n, ast.AsyncFunctionDef) and n.name == nombre)
    ok(sum(1 for n in ast.walk(fn) if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "_terminar") == 1,
       f"{nombre}: termina por `_terminar` (los dos gemelos deciden igual)")
ok("FiltroMarcas()" in ast.get_source_segment(SRC_RA, next(
    n for n in ARBOL_RA.body if isinstance(n, ast.AsyncFunctionDef) and n.name == "resolver_en_vivo")),
   "resolver_en_vivo: el flujo pasa por `FiltroMarcas`")

# El control V1 con el inventario de verdad.
m3 = f6.Material(variante="v3", inventario=segs)
enc = types.SimpleNamespace(tipo_asunto="amparo_directo")
v1 = ra._marcas_y_cobertura(r, enc, m3, "SEXTO. Estudio.\n" + ESTUDIO_MARCADO + "\n⟦C9.z⟧",
                            "ADVERTENCIAS de prueba ⟦C3.a⟧ pendiente.")
ok("⟦" not in v1["estudio"] and "⟦" not in v1["advertencias"], "el estudio y las advertencias salen sin marcas")
cob = v1["meta"]["cobertura"]
ok(cob["marcados"] == 6 and cob["total"] == 8 and cob["faltan"] == ["C3.a", "C3.b"],
   "cobertura: seis de ocho marcados; faltan los del tercer concepto")
ok(cob["sin_rastro"] == ["C3.a", "C3.b"] and cob["visible"] and "C3.a" in v1["aviso"]
   and v1["aviso"].startswith("ARGUMENTOS SIN RESPUESTA IDENTIFICABLE"),
   "sin marca y sin rastro en el estudio: aviso VISIBLE (el de advertencias no cuenta como respuesta)")
ok(cob["en_advertencias"] == ["C3.a"] and cob["desconocidos"] == ["C9.z"],
   "se anota lo marcado en ADVERTENCIAS y lo marcado que no está en el inventario")
_pars = f6.parrafos(v1["estudio"])
ok(v1["meta"]["mapa"]["C1.a"] == [1] and _pars[1].startswith("Sobre el primer concepto")
   and v1["meta"]["mapa"]["C2.b"] == [3] and _pars[3].startswith("Sobre el segundo"),
   "el mapa cuenta los párrafos como los compone el documento (sin el «SEXTO. Estudio.»)")
_ps = v1["meta"]["parrafos"]
ok(isinstance(_ps, list) and len(_ps) == len(_pars)
   and all(_ps[k].split()[:5] == _pars[k].split()[:5] for k in range(len(_pars)))
   and _ps[v1["meta"]["mapa"]["C2.b"][0]].startswith("Sobre el segundo"),
   "`parrafos`: el arranque de cada párrafo, en lista y con el índice del mapa (lo que lee la pestaña)")
ok(cob["parrafos"]["1"].startswith("Sobre el primer concepto") and len(cob["segmentos"]) == 8
   and set(cob["segmentos"][0]) >= {"id", "concepto", "texto", "cita"},
   "la cobertura trae lo que la pestaña «Mapa del estudio» necesita")
_resc = ra._marcas_y_cobertura(r, enc, m3, ESTUDIO_MARCADO + "\nSobre las costas: haber actuado sin "
                               "dolo ni mala fe no releva de la condena, y la condición de campesino "
                               "que no sabe leer ni escribir tampoco.", "")
ok(_resc["meta"]["cobertura"]["rescatados"] == ["C3.a", "C3.b"] and _resc["aviso"] == ""
   and not _resc["meta"]["cobertura"]["visible"],
   "sin marca pero con rastro: se rescata y va en sombra, sin aviso")
_v2 = ra._marcas_y_cobertura(r, enc, f6.Material(variante="v2"), "Texto de la v2.\nSin marcas.", "")
ok(_v2 == {"estudio": "Texto de la v2.\nSin marcas.", "advertencias": "", "aviso": "", "meta": {}},
   "v1/v2: el texto pasa idéntico y no se añade nada a lo anotado")
_malo = ra._marcas_y_cobertura(r, enc, types.SimpleNamespace(variante="v3", inventario=[{"id": "C1.a"}]),
                               None, None)
_roto = ra._marcas_y_cobertura(r, enc, types.SimpleNamespace(variante="v3", inventario=[None]),
                               "⟦C1.a⟧ Texto.", "")
ok(_malo["estudio"] == "" and isinstance(_malo["aviso"], str)
   and _roto["estudio"] == "Texto." and _roto["aviso"] == "",
   "con datos raros no lanza; si el control revienta, el texto sale limpio igual y sin aviso")

# UNA MARCA DELANTE DEL RÓTULO no esconde las ADVERTENCIAS ni los EFECTOS
# (revisión adversarial, 26-sep-2026): `separar_advertencias` corre sobre el
# texto con marcas, antes de `_terminar`.
_c, _a = f6.separar_advertencias("Estudio.\n⟦C1.a⟧ Sobre el primero.\n⟦C3.e⟧ ADVERTENCIAS:\nRevise el tercero.")
ok(_c == "Estudio.\n⟦C1.a⟧ Sobre el primero." and _a == "Revise el tercero.",
   "«⟦…⟧ ADVERTENCIAS:» se aparta igual: no entra en la sentencia")
_c, _a = f6.separar_advertencias("Estudio.\nADVERTENCIAS:\nNota.\n⟦U1⟧ EFECTOS DE LA CONCESIÓN\nDicte otra.")
ok(_a == "Nota." and _c.endswith("⟦U1⟧ EFECTOS DE LA CONCESIÓN\nDicte otra."),
   "y los EFECTOS con su marca delante vuelven a la sentencia, con la marca para el mapa")
# El corte de origin/main, copiado tal cual, para comparar: sin «⟦» tiene que
# dar exactamente lo mismo.
_RX_ADV_VIEJO = re.compile(r"\n\s*ADVERTENCIAS?\s*[:\n]", re.I)
_RX_EF_VIEJO = re.compile(r"\n\s*(?:\*\*|__)?\s*EFECTOS(?:\s+DE\s+LA\s+(?:CONCESI[ÓO]N|PROTECCI[ÓO]N"
                          r"\s+CONSTITUCIONAL))?\s*(?:\*\*|__)?\s*[.:]?\s*\n", re.I)


def _separar_viejo(estudio):
    m_ = _RX_ADV_VIEJO.search(estudio)
    if not m_:
        return estudio.strip(), ""
    cuerpo, adv = estudio[:m_.start()].strip(), estudio[m_.end():].strip()
    me = _RX_EF_VIEJO.search("\n" + adv)
    if me:
        _a = "\n" + adv
        cuerpo = cuerpo + "\n\n" + _a[me.start():].strip()
        adv = _a[:me.start()].strip()
    return cuerpo, adv


_muestras = ["Estudio.\nADVERTENCIAS:\nNota.", "Estudio.\n\nADVERTENCIA\nNota.\nEFECTOS\nDicte.",
             "Sin nada.", "Estudio.\n**ADVERTENCIAS**:\nNota.", "Estudio.\nadvertencias:\nx\n**EFECTOS:**\ny",
             "A.\n  ADVERTENCIAS\nB.\nEFECTOS DE LA PROTECCIÓN CONSTITUCIONAL:\nC.\nD.",
             ESTUDIO_MARCADO.replace("⟦", "").replace("⟧", "") + "\nADVERTENCIAS:\nuna [[p.7 §3]]"]
ok(all(f6.separar_advertencias(t) == _separar_viejo(t) for t in _muestras),
   "sin marcas (v1, v2) el corte es exactamente el de origin/main")

# SI REVIENTA LA SEPARACIÓN MISMA, ninguna marca llega a la sentencia.
_sep_orig = mc.separar_marcas
def _revienta(*a, **k):
    raise RuntimeError("prueba")
mc.separar_marcas = _revienta
try:
    _rb = ra._marcas_y_cobertura(r, enc, m3, ESTUDIO_MARCADO, "⟦C3.a⟧ Nota.")
finally:
    mc.separar_marcas = _sep_orig
ok("⟦" not in _rb["estudio"] and "⟦" not in _rb["advertencias"] and _rb["aviso"] == ""
   and _rb["estudio"].split("\n")[1].startswith("Sobre el primer concepto"),
   "si `separar_marcas` revienta, las marcas se quitan a lo bruto y el texto sigue entero")

# main.py: el evento «listo» y la ficha.
SRC_MAIN = open(os.path.join(AQUI, "main.py"), encoding="utf-8").read()
ARBOL_MAIN = ast.parse(SRC_MAIN)
_ns = {"os": os}
_piezas = []
for n in ARBOL_MAIN.body:
    if isinstance(n, ast.Assign) and any(getattr(t, "id", "") in
                                         ("ADMIN_EMAILS", "TALLER_SIN_LIMITE", "TALLER_DOMINIO_INTERNO")
                                         for t in n.targets):
        _piezas.append(n)
    if isinstance(n, ast.FunctionDef) and n.name in ("_taller_sin_tope", "_taller_es_casa",
                                                     "_taller_variante_estudio", "_commit_desplegado",
                                                     "_taller_meta_listo"):
        _piezas.append(n)
exec(compile(ast.Module(body=_piezas, type_ignores=[]), "main.py", "exec"), _ns)
_v = _ns["_taller_variante_estudio"]
ok(_v("administracion@iurexia.com", "v3") == "v3" and _v("soporte@iurexia.com", "v4") == "v4",
   "una cuenta de casa puede pedir la v3 y la v4")
ok(_v("secretario.piloto@gmail.com", "v3") == "v1", "un secretario del piloto no")
# EL ENCENDIDO POR TIPO LLEGA DE VERDAD AL ENCARGO (revisión adversarial,
# 26-sep-2026): antes la global se leía sin tipo y `ESTUDIO_PROMPT_AD` no
# encendía nada.
os.environ["ESTUDIO_PROMPT_AD"] = "v3"
ok(_v("secretario.piloto@gmail.com", "", "amparo_directo") == "v3"
   and _v("secretario.piloto@gmail.com", "v2", "amparo_directo") == "v3"
   and _v("secretario.piloto@gmail.com", "", "revision_fiscal") == "v1"
   and _v("administracion@iurexia.com", "v2", "amparo_directo") == "v2",
   "con ESTUDIO_PROMPT_AD, el amparo directo sale con ella; los recursos, con la global; casa manda")
os.environ.pop("ESTUDIO_PROMPT_AD", None)
ok(_v("secretario.piloto@gmail.com", "", "amparo_directo") == "v1",
   "sin la variable, lo de siempre")
ok(SRC_MAIN.count('_taller_variante_estudio(\n            user_email, variante_estudio,\n'
                  '            getattr(r.encargo, "tipo_asunto", "") or "")') == 2,
   "los dos gemelos le pasan el tipo del encargo")
_con = type("Res", (), {"meta_estudio": {"variante": "v3", "finish_reason": "stop", "uso": {},
                                         "mapa": {"C1.a": [1]}, "cobertura": {"cobertura": 1.0, "total": 1},
                                         "parrafos": ["Abre.", "Sobre el primero…"]}})()
_sin = type("Res", (), {"meta_estudio": {"variante": "v2", "finish_reason": "stop", "uso": {}}})()
os.environ["RENDER_GIT_COMMIT"] = "abc123"
ml = _ns["_taller_meta_listo"](_con)
ok(ml["mapa"] == {"C1.a": [1]} and ml["cobertura"]["cobertura"] == 1.0 and ml["variante"] == "v3"
   and ml["parrafos"] == ["Abre.", "Sobre el primero…"],
   "el evento «listo» y la ficha llevan el mapa, la cobertura y el arranque de los párrafos")
ok(_ns["_taller_meta_listo"](_sin) == {"variante": "v2", "commit": "abc123", "finish_reason": "stop", "uso": {}},
   "la v1/v2 no ganan ni un campo")
os.environ.pop("RENDER_GIT_COMMIT", None)
_fg = next(n for n in ARBOL_MAIN.body if isinstance(n, ast.FunctionDef) and n.name == "_taller_guardar_proyecto")
ok("**_taller_meta_listo(res)" in ast.get_source_segment(SRC_MAIN, _fg), "la ficha los recibe por `_taller_meta_listo`")
ok(SRC_MAIN.count("**_taller_meta_listo(res),") >= 2, "y el evento «listo» del gemelo en vivo también")
ok("_meta_f = dict(meta_estudio or {})" in SRC_RA, "`_terminar` devuelve lo anotado (con el mapa) en el Resultado")

# ═══════════════════════════════════════════════════════════════════════════
print("\n7 · EL BANCO: v3 Y v4, EL MAPA, LA COBERTURA Y LA MARCA FILTRADA")
import banco_estudio as be
import comparar_estudio as ce_
import metricas_estudio as me_

ok(me_.marcas_filtradas("SEXTO. Estudio. ⟦C1.a⟧ Sobre el primero. [[C2.b]] y [[p.7 §3]]") == 2
   and me_.marcas_filtradas("Limpio, con su nota [[p.7 §3]].") == 0,
   "se cuenta la marca que llegó al documento; la nota al pie no es marca")
ok("marcas_filtradas" in me_.CLAVES and me_.MEJOR["marcas_filtradas"] == "menos",
   "es una métrica del banco (menos es mejor)")
ok([v.servidor for v in be.variantes("v2,v3,v4,v3@r")] == ["v2", "v3", "v4", "v3"],
   "el banco pide v3 y v4 (y su réplica) como cualquier otra variante")
ok(be.motivo_de_descarte(be.Variante("v3", "v3"), "v3") is None
   and be.motivo_de_descarte(be.Variante("v3", "v3"), "v1"),
   "y descarta la corrida si el servidor no corrió la que se pidió")
_acc = be.Acumulador(t0=0)
_acc.meter({"tipo": "listo", "docx_b64": "", "variante": "v3", "commit": "abc",
            "mapa": {"C1.a": [1]}, "cobertura": {"cobertura": 0.5, "total": 2, "marcados": 1,
                                                 "sin_rastro": ["C1.b"], "rescatados": []}}, 1)
_fila, _ = be.resultado(_acc, be.POR_NUMERO["103/2025"], be.Variante("v3", "v3"), 1, "estandar", "x")
ok(_fila["mapa"] == {"C1.a": [1]} and _fila["cobertura"]["total"] == 2,
   "la fila guarda el mapa y la cobertura del «listo»")
ok(me_.cobertura_de_fila(_fila) == {"cobertura": 0.5, "con_rescate": None, "total": 2, "marcados": 1,
                                   "sin_rastro": 1, "rescatados": 0, "desconocidos": 0,
                                   "visible": False, "parrafos_marcados": 1}
   and me_.cobertura_de_fila({"listo": {}}) is None, "`cobertura_de_fila` la lee (None si no la trae)")
ok("marcas 1/2" in be._linea(dict(_fila, t_total=1, ok=True)), "y la línea de progreso la enseña")
_texto = ("SEXTO. Estudio.\nLos conceptos de violación son infundados.\n"
          "Sobre el primer concepto de violación, se considera infundado. " + "Razón. " * 60 + "\n")


def _f(v, k, texto, cob=None):
    return {"caso": "999/2099", "variante": v, "k": k, "ok": True, "descartada": None, "texto": texto,
            "listo": ({"cobertura": cob} if cob else {})}


_cob = {"cobertura": 0.9, "cobertura_con_rescate": 1.0, "total": 10, "marcados": 9,
        "sin_rastro": [], "rescatados": ["C1.c"], "desconocidos": [], "visible": False}
_datos = {"999/2099": {"v2": [_f("v2", 1, _texto), _f("v2", 2, _texto)],
                       "v3": [_f("v3", 1, _texto, _cob), _f("v3", 2, _texto + "⟦C1.a⟧ resto", _cob)]}}
_par = ce_.comparar_par(_datos, ce_.Medidor(escritos={"999/2099": ""}, oros={"999/2099": None}), "v2", "v3")
ok(_par["bloquea"] and _par["bloquea"][0]["filtradas"] == 1 and ce_.detiene(_par),
   "una marca filtrada al documento BLOQUEA (w2_final §6.6)")
_md = ce_.informe("prueba", [_par], ce_.operacion(_datos))
ok("### Marcas e inventario (v3/v4)" in _md and "| 999/2099 | v3 | 2 de 2 | 10 | 0.9 · 0.9 | 1 |" in _md,
   "el informe trae la sección de las marcas con la cobertura de cada corrida")
ok("Con marcas en el .docx" in _md, "y la columna de las filtradas en la tabla de bloqueos")
_datos2 = {"999/2099": {"v1": [_f("v1", 1, _texto)], "v2": [_f("v2", 1, _texto)]}}
_md2 = ce_.informe("prueba", [ce_.comparar_par(_datos2, ce_.Medidor(escritos={"999/2099": ""},
                                                                      oros={"999/2099": None}), "v1", "v2")],
                   ce_.operacion(_datos2))
ok("Marcas e inventario" not in _md2, "sin corridas con marcas (v1 contra v2) el informe no cambia")

print()
if FALLOS:
    print(f"FALLAN {len(FALLOS)}: " + " · ".join(FALLOS))
    raise SystemExit(1)
print("RESULTADO: TODAS LAS COMPROBACIONES PASAN")
