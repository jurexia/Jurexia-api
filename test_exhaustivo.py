"""p2-exhaustivo — el «sin materia» del criterio, la marca honesta y la reparación dirigida.

26-sep-2026. El banco del localizador a ciegas midió que la v3 y la v4 dejan
más omisiones graves que la v1, con dos fallos: el estudio declara sin materia
por su cuenta (o sin que los EFECTOS lo recojan) y absorbe argumentos con dato
propio en una calificación global. Esto comprueba, con dobles del modelo:

  1 · el prompt de la familia v2 ata la escala «innecesario / cae / inoperante»
      al criterio, prohíbe el «sin materia» por cuenta del estudio y manda
      nombrar en los EFECTOS lo que la concesión deja a la responsable; la v1
      no cambia (su instantánea la cuida test_prompt_v2.py);
  2 · la declaración de «sin estudio» se reconoce en las frases reales del
      banco y no en las que no lo son (calificación general, negación, lo que
      hizo la responsable, «litigios innecesarios», el adhesivo, una tesis);
  3 · el control del punto 1 (`sin_materia_por_su_cuenta`), con marcas y sin
      ellas, y que va en sombra;
  4 · la marca honesta (`sin_su_dato`), con los casos que la calibraron;
  5 · la reparación: el prompt sin frases modelo, el parser, las guardas, la
      inserción (sólo añade), el vencimiento y el fallo, y el razonamiento
      nunca por debajo de medio;
  6 · el camino: los DOS gemelos la llaman igual, el de flujo emite
      «completando», main.py lo reenvía, y la v1/v2 no la tocan.

    .venv/bin/python test_exhaustivo.py
"""
import ast
import asyncio
import os
import re
import types

os.environ.pop("ESTUDIO_PROMPT", None)
os.environ.pop("ESFUERZO_REPARAR", None)

import exhaustivo as X
import fase6_estudio as f6
import marcas as mc

FALLOS = []


def ok(cond, que):
    print(f"   {'PASA ' if cond else 'FALLA'}  {que}")
    if not cond:
        FALLOS.append(que)


AQUI = os.path.dirname(os.path.abspath(__file__))

# ═══════════════════════════════════════════════════════════════════════════
print("\n1 · EL PROMPT: EL «SIN MATERIA» ES DEL CRITERIO")
C_CONC = [f6.Criterio("¿La Sala debía acreditar el valor de los bienes?", "fundado", "Razón A", "principal"),
          f6.Criterio("¿Valoró la doble jornada?", "innecesario",
                      "Dado el sentido del estudio del problema principal, queda sin materia.", "accesorio")]
P_CONC = [{"pregunta": C_CONC[0].problema, "cubre": [1]}, {"pregunta": C_CONC[1].problema, "cubre": [2]}]


def mat(variante, formato="estandar", **kw):
    return f6.Material(tipo_asunto="amparo_directo", materia="civil", formato=formato,
                       problemas=P_CONC, n_planteamientos=2, variante=variante, **kw)


for forma in ("estandar", "moderna"):
    p2 = f6.prompt_estudio("ACTO", "CONC", C_CONC, mat("v2", forma))
    p1 = f6.prompt_estudio("ACTO", "CONC", C_CONC, mat("v1", forma))
    ok("NINGÚN ARGUMENTO SE DECLARA SIN ESTUDIO POR CUENTA DEL ESTUDIO" in p2
       and "valen SÓLO para lo que el CRITERIO DEL SECRETARIO calificó así" in p2,
       f"v2/{forma}: la escala vale sólo para lo que el criterio calificó así")
    ok("se NOMBRA en los\n  EFECTOS, con su dato" in p2 or "se NOMBRA en los EFECTOS" in " ".join(p2.split()),
       f"v2/{forma}: lo que la concesión deja a la responsable se nombra en los EFECTOS")
    ok("sus argumentos con dato propio se nombran en los EFECTOS" in p2,
       f"v2/{forma}: la línea NO SE ESTUDIA del criterio lo dice")
    ok("una de esas órdenes dice cuáles son" in p2, f"v2/{forma}: y la descripción de los EFECTOS")
    ok("POR CUENTA DEL ESTUDIO" not in p1 and "nombran en los EFECTOS" not in p1
       and "una de esas órdenes dice cuáles son" not in p1,
       f"v1/{forma}: nada de esto (congelada)")
    ok("ésta es la ÚNICA medida para todo el estudio" in p2, f"v2/{forma}: sigue siendo la ÚNICA medida")
_segs3 = [{"id": "C1.a", "concepto": 1, "texto": "Sostiene que la Sala le impuso probar el valor de los bienes.",
           "cita": "", "anclas": []}]
p3 = f6.prompt_estudio("ACTO", "CONC", C_CONC, mat("v3", inventario=_segs3))
ok("Declararlo sin materia o innecesario tampoco es respuesta" in p3
   and "La marca va en el párrafo que aplica la razón al dato" in p3,
   "v3: la regla de las marcas dice que declararlo no es respuesta y dónde va la marca")
ok(f6.prompt_estudio("ACTO", "CONC", C_CONC, mat("v4", inventario=_segs3)) == p3,
   "v4 sin guion: la misma v3")
# Ni una frase nueva entre comillas angulares respecto de la v1 (la lista de
# test_prompt_v2 no se amplía): lo nuevo son descripciones.
_rx_q = re.compile(r"«([^«»]{1,300})»")
# (la calificación general concordada «innecesarios» es de la v2 de antes: test_prompt_v2)
_nuevo_v2 = {" ".join(x.split()).replace("innecesarios", "innecesario")
             for x in _rx_q.findall(f6.prompt_estudio("A", "C", C_CONC, mat("v2")))}
_v1c = {" ".join(x.split()) for x in _rx_q.findall(f6.prompt_estudio("A", "C", C_CONC, mat("v1")))}
_nuevas = {x for x in _nuevo_v2 - _v1c if "innecesari" in x or "materia" in x or "EFECTOS" in x}
ok(not _nuevas, "el punto 1 no mete ninguna frase entrecomillada para copiar"
   + (f": {sorted(_nuevas)}" if _nuevas else ""))

# ═══════════════════════════════════════════════════════════════════════════
print("\n2 · LA DECLARACIÓN DE «SIN ESTUDIO» (frases del banco del 26-sep-2026)")
SI = [
    "Lo anterior, toda vez que la concesión hará que la Sala emita una nueva resolución, por lo que carece "
    "de objeto examinar ahora si ponderó correctamente esas actividades o si actualizan una doble jornada.",
    "En relación con los planteamientos sobre la custodia del hijo y la doble jornada, resultan innecesarios.",
    "Finalmente, quedan sin materia los argumentos relativos a que el demandado adquirió tres bienes.",
    "Por ello, su estudio autónomo no produciría un beneficio adicional frente a la decisión que deberá dictarse.",
    "En consecuencia, tampoco es necesario estudiar la omisión atribuida a la Sala respecto de la temporalidad.",
    "Sobre el tercer concepto de violación, en el que la parte quejosa cuestiona la tacha. Resulta innecesario.",
    "el pronunciamiento específico sobre esas omisiones quedaría subordinado al nuevo examen de la Sala.",
]
NO = [
    "Los conceptos de violación son en parte fundados y en parte innecesarios.",
    "Por ello, la nulidad de esos actos no hizo innecesario examinar la legalidad del crédito fiscal.",
    "La Sala consideró innecesario estudiar los restantes conceptos de impugnación.",
    "la buena fe procesal es una conducta que evite litigios innecesarios; en este sentido, se desestima.",
    "queda cubierto por el Seguro de Vida Institucional incluyendo el beneficio adicional contratado.",
    "SÉPTIMO. Amparo adhesivo. Debe declararse sin materia el promovido por el tercero interesado.",
    "“AGRAVIOS INOPERANTES. LO SON AQUELLOS QUE SE SUSTENTAN EN PREMISAS FALSAS. Los agravios cuya "
    "construcción parte de premisas falsas son inoperantes, ya que a ningún fin práctico conduciría su análisis.",
    "El criterio aplicable establece que carece de utilidad examinar una conclusión que depende de una premisa falsa.",
    "Se estima fundado el concepto de violación, porque la Sala omitió valorar la pericial en informática.",
]
ok(all(X.declara_sin_estudio(x) for x in SI),
   "reconoce las siete formas medidas" + "".join(f" · NO VE: {x[:60]}" for x in SI if not X.declara_sin_estudio(x)))
ok(not any(X.declara_sin_estudio(x) for x in NO),
   "no la ve en la calificación general, la negación, lo que hizo la Sala, otros sentidos, el adhesivo ni una tesis"
   + "".join(f" · VE: {x[:60]}" for x in NO if X.declara_sin_estudio(x)))

# ═══════════════════════════════════════════════════════════════════════════
print("\n3 · EL CONTROL DEL PUNTO 1: SIN MATERIA POR SU CUENTA (EN SOMBRA)")
SEGS = [
    {"id": "C1.a", "concepto": 1, "texto": "Aduce que la Sala impuso a la quejosa la carga de acreditar el valor "
     "de los inmuebles mediante avalúo para demostrar el desequilibrio patrimonial.", "cita": "", "anclas": ["art 268 cc"]},
    {"id": "C1.b", "concepto": 1, "texto": "Sostiene que la Sala omitió considerar la custodia del hijo menor y "
     "el cuidado que brindaba en fines de semana y días festivos, que configuran una doble jornada.",
     "cita": "", "anclas": []},
    {"id": "C2.a", "concepto": 2, "texto": "Refiere que la Sala no valoró la pericial en informática ni la "
     "jurisprudencia sobre la fiabilidad de la plataforma bancaria.", "cita": "", "anclas": ["reg 2017826"]},
    {"id": "C2.b", "concepto": 2, "texto": "Afirma que se vulneran los artículos 14 y 16 constitucionales.",
     "cita": "", "anclas": ["art 14 cpeum", "art 16 cpeum"]},
]
CRIT = [f6.Criterio("¿Debía acreditar el valor de los bienes?", "fundado", "Razón", "principal"),
        f6.Criterio("¿Valoró la pericial en informática?", "innecesario", "Sin materia.", "accesorio")]
PROB = [{"pregunta": CRIT[0].problema, "cubre": [1]}, {"pregunta": CRIT[1].problema, "cubre": [2]}]
PS = ["Los conceptos de violación son en parte fundados y en parte innecesarios.",
      "Sobre el primer concepto de violación, la Sala exigió el avalúo. Es fundado.",
      "Lo anterior, porque el artículo 268 del Código Civil presume la contribución.",
      "Respecto de la custodia del hijo y la doble jornada, carece de objeto examinar esas actividades.",
      "Sobre el segundo concepto de violación, relativo a la pericial en informática, resulta innecesario.",
      "EFECTOS DE LA CONCESIÓN",
      "1. Deje insubsistente la sentencia reclamada.",
      "2. Dicte otra en la que no exija el avalúo como condición de la compensación."]
MAPA = {"C1.a": [1], "C1.b": [3], "C2.a": [4], "C2.b": [4]}
h = X.sin_materia_por_su_cuenta(PS, MAPA, SEGS, CRIT, PROB)
ok([x["ids"] for x in h] == [["C1.b"]],
   f"con marcas: acusa la doble jornada (criterio fundado) y no la pericial (criterio innecesario): {h}")
ok(X.sin_materia_por_su_cuenta(PS, {}, [], CRIT, PROB) and all(x["conceptos"] == [1] for x in X.sin_materia_por_su_cuenta(PS, {}, [], CRIT, PROB)),
   "sin marcas: por el ordinal del apartado abierto (el primero, fundado)")
_ps_rest = ["Se estudia primero el primer concepto, cuya solución hace innecesario examinar los restantes."]
ok(not X.sin_materia_por_su_cuenta(_ps_rest, {}, [], CRIT, PROB),
   "«hace innecesario examinar los restantes» con un criterio que tiene innecesarios: no se acusa")
ok(X.sin_materia_por_su_cuenta(_ps_rest, {}, [], [CRIT[0]], PROB),
   "y con un criterio sin ninguno que admita no estudiarse, sí")
_h_ef = X.sin_materia_por_su_cuenta(PS + ["3. Es innecesario pronunciarse sobre lo demás."], MAPA, SEGS, CRIT, PROB)
ok(all(x["parrafo"] < 5 for x in _h_ef), "los EFECTOS no se revisan: son órdenes, no estudio")
ok(X.SIN_MATERIA_VISIBLE is False and "810" in X.CALIBRACION_ORO,
   "va en sombra por omisión: acusa a un engrose bueno (ADC 810/2025)")
ok("NO" not in X.aviso_sin_materia([]) and "SIN QUE EL CRITERIO LO DIGA" in X.aviso_sin_materia(h),
   "el aviso sólo existe cuando hay hallazgos")

# ═══════════════════════════════════════════════════════════════════════════
print("\n4 · LA MARCA HONESTA: EL ARGUMENTO DECLARADO SIN SU DATO")
sd = X.sin_su_dato(PS, MAPA, SEGS)
ok([x["id"] for x in sd] == ["C1.b", "C2.a"],
   f"acusa los dos declarados sin estudio cuyo dato no está en los efectos ni en otra respuesta: {sd}")
PS_EF = PS + ["3. Al dictar la nueva resolución, examine la custodia del hijo menor, el cuidado en fines de "
              "semana y días festivos y la doble jornada."]
ok([x["id"] for x in X.sin_su_dato(PS_EF, MAPA, SEGS)] == ["C2.a"],
   "si los EFECTOS nombran su dato, la doble jornada ya no se acusa")
PS_FONDO = PS[:5] + ["La pericial en informática no acredita la fiabilidad de la plataforma bancaria, "
                     "y la jurisprudencia de registro 2017826 exige esa prueba."] + PS[5:]
ok([x["id"] for x in X.sin_su_dato(PS_FONDO, MAPA, SEGS)] == ["C1.b"],
   "si otro párrafo de fondo contesta su dato (aquí, con su ancla), la pericial ya no se acusa")
ok("C2.b" not in [x["id"] for x in sd], "el que sólo invoca los artículos 14 y 16 no tiene dato propio: no se acusa")
ok("C1.a" not in [x["id"] for x in sd], "el que se contesta de fondo no se acusa")
ok(X.Datos(SEGS).tiene_dato("C1.b") and not X.Datos(SEGS).tiene_dato("C2.b"),
   "dato propio: palabras que no comparte con más de dos, o un ancla propia no genérica")
ok(X.UMBRAL_EFECTOS == 0.4 and X.UMBRAL_SUSTANTIVO == 0.6, "los umbrales calibrados")
_r = X.revisar_texto("⟦C1.b⟧ " + PS[3] + "\n" + "\n".join(PS[5:]), SEGS, CRIT, PROB)
ok([x["id"] for x in _r["sin_dato"]] == ["C1.b"] and _r["sin_materia"],
   "`revisar_texto` lee el estudio CON sus marcas")

# ═══════════════════════════════════════════════════════════════════════════
print("\n5 · LA REPARACIÓN DIRIGIDA")
ESTUDIO = ("Los conceptos de violación son en parte fundados y en parte innecesarios.\n\n"
           "⟦C1.a⟧ Sobre el primer concepto de violación, la Sala exigió el avalúo. Es fundado.\n\n"
           "⟦C1.b⟧ Respecto de la custodia del hijo y la doble jornada, carece de objeto examinar esas actividades.\n\n"
           "⟦C2.a C2.b⟧ Sobre el segundo concepto de violación, relativo a la pericial, resulta innecesario.\n\n"
           "EFECTOS DE LA CONCESIÓN\n\n"
           "1. Deje insubsistente la sentencia reclamada.\n\n"
           "2. Dicte otra en la que no exija el avalúo como condición de la compensación.")
MAT = f6.Material(tipo_asunto="amparo_directo", variante="v3", inventario=SEGS, problemas=PROB)
FALTAN = X.revisar_texto(ESTUDIO, SEGS, CRIT, PROB)["sin_dato"]
ok([f["id"] for f in FALTAN] == ["C1.b", "C2.a"], "los que faltan en el estudio de prueba")
pr = X.prompt_reparacion(ESTUDIO, CRIT, MAT, FALTAN, "")
_instr = pr.split("EL CRITERIO DEL SECRETARIO")[0]
ok("«" not in _instr and "»" not in _instr,
   "las instrucciones no llevan ni una frase entre comillas para copiar (lo entrecomillado son los datos)")
ok("C1.b · concepto de violación 1" in pr and "C2.a" in pr and ESTUDIO in pr and "RAZÓN DEL SECRETARIO: Razón" in pr,
   "recibe el estudio con sus marcas, el criterio y los argumentos como datos")
ok("No cambias ninguna calificación" in pr and "No citas tesis" in pr and "no la omisión" in pr,
   "y los límites: ni calificación, ni tesis nuevas, ni la omisión de la responsable como objeto")

SALIDA = ("Aquí van las piezas:\n"
          "EFECTO ⟦C1.b⟧ Al dictar la nueva resolución, examine la custodia del hijo menor, el cuidado en fines de "
          "semana y días festivos y la doble jornada que la quejosa alega.\n"
          "EFECTO ⟦C2.a U3⟧ 5. Al dictar la nueva resolución, valore la pericial en informática y la fiabilidad de "
          "la plataforma bancaria que la institución de crédito ofreció.\n"
          "⟦C9.z⟧ Párrafo de un argumento que nadie pidió, con palabras de sobra para pasar el mínimo de extensión.\n")
pars, efs, desc = X.parsear(SALIDA, ["C1.b", "C2.a"])
ok(not pars and [i for i, _ in efs] == [["C1.b"], ["C2.a"]],
   "el parser toma las dos órdenes y quita de la marca la unidad del plan que no se pidió (⟦C2.a U3⟧)")
ok(efs[1][1].startswith("Al dictar") and desc and desc[0]["motivo"] == "identificador que no se pidió",
   "quita el número que el modelo puso y descarta la pieza de un argumento que no se pidió")
nuevo, ins = X.insertar(ESTUDIO, [], efs, {s["id"]: s for s in SEGS})
ok(nuevo.startswith(ESTUDIO.split("\n\n2. Dicte")[0]) and "\n\n3. Al dictar la nueva resolución, examine la custodia" in nuevo
   and "\n\n4. Al dictar la nueva resolución, valore la pericial" in nuevo,
   "las órdenes van al final de los EFECTOS, numeradas en su serie (3 y 4)")
ok([l for l in ESTUDIO.split("\n")] == [l for l in nuevo.split("\n") if not l.startswith(("3. ", "4. "))][:len(ESTUDIO.split("\n"))]
   and all(l in nuevo.split("\n") for l in ESTUDIO.split("\n")),
   "no se toca nada más: todos los renglones del estudio siguen, en su orden")
nuevo_p, ins_p = X.insertar(ESTUDIO, [(["C1.b"], "Con todo, la custodia y la doble jornada son un pilar autónomo de la Sala que la concesión no alcanza, por lo que se examinan.")], [], {s["id"]: s for s in SEGS})
_ls = [l for l in nuevo_p.split("\n") if l.strip()]
ok(_ls[_ls.index(next(l for l in _ls if l.startswith("⟦C1.b⟧ Respecto"))) + 1].startswith("⟦C1.b⟧ Con todo"),
   "el párrafo va justo después del que ya marcaba su argumento, con su marca")
_limpio, _mapa = mc.separar_marcas(nuevo_p)
ok("⟦" not in _limpio and len(_mapa.get("C1.b", [])) == 2, "y la marca nueva entra al mapa como las demás")
POR = {s["id"]: s for s in SEGS}
REP = X.reparto(CRIT, PROB)
ok(X.guardas(["C1.b"], "Por tanto, resulta innecesario examinar la doble jornada, pues la concesión la absorbe.", False,
             ESTUDIO, POR, REP).startswith("declara sin estudio"),
   "guarda: un párrafo que vuelve a declarar sin estudio lo que su criterio manda estudiar se descarta")
ok(X.guardas(["C1.b"], "En ese sentido, el argumento de la doble jornada es infundado, porque la Sala sí la valoró.",
             False, ESTUDIO, POR, REP).startswith("califica en contra"),
   "guarda: un párrafo que califica al revés que su criterio (infundado bajo un fundado) se descarta")
ok(not X.guardas(["C1.b"], "En ese sentido, el argumento resulta fundado pero insuficiente, porque subsiste el otro pilar.",
                 False, ESTUDIO, {}, []), "sin criterio que casar, la guarda de calificación no inventa uno")
_rep_fi = X.reparto([f6.Criterio(CRIT[0].problema, "fundado_insuficiente")], PROB)
ok(not X.guardas(["C1.b"], "En ese sentido, el argumento resulta fundado pero insuficiente, porque subsiste el otro pilar.",
                 False, ESTUDIO, POR, _rep_fi), "«fundado pero insuficiente» va en la dirección de su criterio (no prospera)")
ok(X.guardas(["C1.b"], "Sirve de apoyo el criterio de registro 2099999, que obliga a valorar la doble jornada del caso.",
             False, ESTUDIO, POR, REP).startswith("cita un registro"),
   "guarda: un registro que no está en el estudio ni en los datos del argumento se descarta")
ok(not X.guardas(["C2.a"], "Valore la pericial y la jurisprudencia de registro 2017826 sobre la fiabilidad de la plataforma.",
                 True, ESTUDIO, POR, REP),
   "el registro que trae el propio argumento sí se admite")


class _Resp:
    def __init__(self, t):
        self.choices = [types.SimpleNamespace(message=types.SimpleNamespace(content=t), finish_reason="stop")]
        self.usage = None


class _Cli:
    def __init__(self, salida="", demora=0.0, error=None):
        self.kw, cli = [], self

        class _C:
            @staticmethod
            async def create(**kw):
                cli.kw.append(kw)
                if demora:
                    await asyncio.sleep(demora)
                if error:
                    raise error
                return _Resp(salida)
        self.chat = types.SimpleNamespace(completions=_C)


cli = _Cli(SALIDA)
nuevo_r, inf = asyncio.run(X.reparar(cli, ESTUDIO, CRIT, MAT, FALTAN, escrito=""))
ok(inf["estado"] == "ok" and len(inf["efectos"]) == 2 and "3. Al dictar" in nuevo_r,
   "con el doble del modelo: dos órdenes insertadas")
ok(len(cli.kw) == 1 and cli.kw[0]["model"] == f6.MODELO_ESTUDIO and cli.kw[0]["reasoning_effort"] == "medium",
   "UNA llamada, al modelo del estudio, con razonamiento medio")
os.environ["ESFUERZO_REPARAR"] = "low"
ok(X.esfuerzo_reparar() == "medium", "«low» no se admite: nunca por debajo de medio")
os.environ["ESFUERZO_REPARAR"] = "minimal"
ok(X.esfuerzo_reparar() == "medium", "ni «minimal»")
os.environ["ESFUERZO_REPARAR"] = "high"
ok(X.esfuerzo_reparar() == "high", "«high», como el estudio, sí")
os.environ.pop("ESFUERZO_REPARAR", None)
_v, inf_v = asyncio.run(X.reparar(_Cli(SALIDA, demora=0.3), ESTUDIO, CRIT, MAT, FALTAN, tope_s=0.05))
ok(_v == ESTUDIO and inf_v["estado"] == "vencio", "si vence, el estudio sale como estaba")
_f, inf_f = asyncio.run(X.reparar(_Cli(error=RuntimeError("caída")), ESTUDIO, CRIT, MAT, FALTAN))
ok(_f == ESTUDIO and inf_f["estado"] == "fallo" and inf_f["error"] == "RuntimeError",
   "si falla, el estudio sale como estaba")
_n, inf_n = asyncio.run(X.reparar(_Cli("nada que ver"), ESTUDIO, CRIT, MAT, FALTAN))
ok(_n == ESTUDIO and inf_n["estado"] == "sin_piezas", "si ninguna pieza sirve, como estaba")
_z, inf_z = asyncio.run(X.reparar(None, ESTUDIO, CRIT, MAT, FALTAN))
ok(_z == ESTUDIO and inf_z["estado"] == "nada", "sin cliente no se llama a nada")
ok("ESTUDIO COMPLETADO" in X.aviso_reparacion(inf, SEGS) and "ESTUDIO SIN COMPLETAR (la llamada venció)"
   in X.aviso_reparacion(inf_v, SEGS) and X.aviso_reparacion(inf_z, SEGS) == "",
   "el aviso visible dice qué se añadió, o que no se pudo")

# ═══════════════════════════════════════════════════════════════════════════
print("\n6 · EL CAMINO: LOS DOS GEMELOS, EL EVENTO «completando» Y LA v1/v2")
import redactor_adelanto as ra

ok(ra._por_completar(f6.Material(variante="v2", inventario=SEGS, problemas=PROB), CRIT, ESTUDIO) == [],
   "v2: no se revisa ni se repara (aunque el material traiga inventario)")
ok(ra._por_completar(f6.Material(variante="v1"), CRIT, ESTUDIO) == [], "v1: tampoco")
ok([f["id"] for f in ra._por_completar(MAT, CRIT, ESTUDIO)] == ["C1.b", "C2.a"], "v3: los que faltan")
ok([f["id"] for f in ra._por_completar(f6.Material(variante="v4", inventario=SEGS, problemas=PROB), CRIT, ESTUDIO)]
   == ["C1.b", "C2.a"], "v4: igual")

SRC_RA = open(os.path.join(AQUI, "redactor_adelanto.py"), encoding="utf-8").read()
ARBOL_RA = ast.parse(SRC_RA)
for nombre in ("resolver", "resolver_en_vivo"):
    fn = next(n for n in ARBOL_RA.body if isinstance(n, ast.AsyncFunctionDef) and n.name == nombre)
    llamadas = [getattr(n.func, "id", "") for n in ast.walk(fn) if isinstance(n, ast.Call)]
    ok(llamadas.count("_por_completar") == 1 and llamadas.count("_completar_estudio") == 1,
       f"{nombre}: revisa y completa una vez, con las mismas dos funciones")
    src = ast.get_source_segment(SRC_RA, fn)
    ok(src.index("_completar_estudio(") < src.index("_efectos_de_reposicion(") < src.index("_terminar("),
       f"{nombre}: antes de los efectos, las constancias, los preceptos y `_terminar`")
_src_vivo = ast.get_source_segment(SRC_RA, next(n for n in ARBOL_RA.body if isinstance(n, ast.AsyncFunctionDef)
                                               and n.name == "resolver_en_vivo"))
ok(_src_vivo.index('{"tipo": "completando"}') < _src_vivo.index("_completar_estudio(")
   < _src_vivo.index('{"tipo": "componiendo"}'),
   "resolver_en_vivo: «completando» antes de la llamada, y antes de «componiendo»")
SRC_MAIN = open(os.path.join(AQUI, "main.py"), encoding="utf-8").read()
ok('elif tipo == "completando":\n' in SRC_MAIN and '_cola.put_nowait({"tipo": "completando"})' in SRC_MAIN,
   "main.py reenvía «completando» a la pantalla")
_term = next(n for n in ARBOL_RA.body if isinstance(n, ast.AsyncFunctionDef) and n.name == "_terminar")
ok("_marcas_y_cobertura(r, e, material, estudio, advertencias, criterios)" in ast.get_source_segment(SRC_RA, _term),
   "`_terminar` pasa el criterio al control, que queda en la cobertura (sombra)")

# El flujo de verdad, con dobles: el estudio trae la doble jornada declarada
# sin estudio; la reparación la nombra en los EFECTOS y `_terminar` lo recibe.
_llam = {}


async def _vivo_falso(*a, **kw):
    for i in range(0, len(ESTUDIO), 11):
        yield {"tipo": "texto", "dato": ESTUDIO[i:i + 11]}
    yield {"tipo": "fin", "estudio": ESTUDIO, "advertencias": "", "avisos": [], "meta": {"variante": "v3"}}


async def _redactar_falso(*a, **kw):
    if isinstance(kw.get("meta"), dict):
        kw["meta"].update({"variante": "v3"})
    return ESTUDIO, "", []


async def _terminar_falso(cliente, r_, e_, criterios, material, estudio, advertencias, avisos, *a, **kw):
    _llam["estudio"], _llam["avisos"], _llam["meta"] = estudio, list(avisos), kw.get("meta_estudio")
    return "RESULTADO"


r = types.SimpleNamespace()
import fases123_pipeline as f123
r.fases = f123.Fases123(resumen_conceptos="", fuentes=["acto", "escrito"])
r.encargo = types.SimpleNamespace(formato="", variante_estudio="v3", tipo_asunto="amparo_directo",
                                  es_recurso=False, propuesta_global=None, conceptos_violacion="",
                                  resolvio_declarado="", guion="")
r.partes = None
_orig = (ra.f6.redactar_en_vivo, ra.f6.redactar, ra._terminar, ra._litis_y_material)
ra.f6.redactar_en_vivo, ra.f6.redactar, ra._terminar = _vivo_falso, _redactar_falso, _terminar_falso
ra._litis_y_material = lambda *a, **k: []


async def _correr(material, cliente):
    tipos = []
    async for paso in ra.resolver_en_vivo(cliente, r, CRIT, material, "/tmp/x.docx"):
        tipos.append(paso["tipo"])
    return tipos


try:
    _cli_v = _Cli(SALIDA)
    _tipos = asyncio.run(_correr(f6.Material(tipo_asunto="amparo_directo", variante="v3", inventario=SEGS,
                                             problemas=PROB), _cli_v))
    ok("completando" in _tipos and _tipos.index("completando") < _tipos.index("componiendo") < _tipos.index("listo"),
       f"en vivo: texto… → completando → componiendo → listo ({[t for t in _tipos if t != 'texto']})")
    ok("3. Al dictar la nueva resolución, examine la custodia" in _llam["estudio"]
       and any(a.startswith("ESTUDIO COMPLETADO") for a in _llam["avisos"])
       and (_llam["meta"] or {}).get("completado", {}).get("estado") == "ok",
       "`_terminar` recibe el estudio completado, el aviso visible y el informe")
    _tipos2 = asyncio.run(_correr(f6.Material(tipo_asunto="amparo_directo", variante="v2", inventario=SEGS,
                                              problemas=PROB), _Cli(SALIDA)))
    ok("completando" not in _tipos2 and _llam["estudio"] == ESTUDIO, "en vivo, v2: ni evento ni llamada; el estudio intacto")
    _cli_p = _Cli(SALIDA)
    _res = asyncio.run(ra.resolver(_cli_p, r, CRIT, f6.Material(tipo_asunto="amparo_directo", variante="v3",
                                                                inventario=SEGS, problemas=PROB), "/tmp/x.docx"))
    ok(_res == "RESULTADO" and len(_cli_p.kw) == 1 and "3. Al dictar la nueva resolución" in _llam["estudio"],
       "el gemelo plano hace lo mismo: una llamada y el estudio completado")
finally:
    ra.f6.redactar_en_vivo, ra.f6.redactar, ra._terminar, ra._litis_y_material = _orig

print()
if FALLOS:
    print(f"FALLAN {len(FALLOS)}: " + " · ".join(FALLOS))
    raise SystemExit(1)
print("RESULTADO: TODAS LAS COMPROBACIONES PASAN")
