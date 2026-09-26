"""La v2 del prompt del estudio (la limpieza) y la v1 congelada — 26-sep-2026.

David aprobó el «Paso 1» de la propuesta del estudio: quitar del prompt las
órdenes que se contradicen y las capas que fabrican repetición, como VARIANTE
seleccionable, con la de producción congelada. Esto comprueba:

  1 · la v1 es el prompt de antes con UNA sola frase cambiada —el orden del
      artículo 189—, renderizado con el criterio real del ADC 93/2026;
  2 · la v2 no trae ninguna de las órdenes retiradas, en ninguna forma, tipo
      ni materia;
  3 · la v2 conserva lo que la propuesta manda conservar;
  4 · la v2 no mete frases nuevas para copiar (lección: un ejemplo del prompt
      se firma literal);
  5 · el cierre sólo se permite con tres apartados o más de resultado distinto;
  6 · la variante sólo la elige una cuenta de casa, y viaja por el material;
  7 · los dos gemelos la pasan igual (AST, como test_formato_sentencia.py);
  8 · los controles ajustados no acusan a los engroses buenos del banco
      Kingston (se salta si el corpus no está en esta máquina);
  9 · la instrumentación: variante, finish_reason y tokens salen de los dos
      redactores y llegan a la ficha y al evento «listo».

    .venv/bin/python test_prompt_v2.py
"""
import ast
import asyncio
import glob
import json
import os
import re
import sys

os.environ.pop("ESTUDIO_PROMPT", None)

import calidad_estudio as ce
import fase6_estudio as f6
import formato_sentencia as fs

FALLOS = []


def ok(cond, que):
    print(f"   {'PASA ' if cond else 'FALLA'}  {que}")
    if not cond:
        FALLOS.append(que)


AQUI = os.path.dirname(os.path.abspath(__file__))

# EL CRITERIO DEL 93, tal como lo mandó la pantalla (crit_v6.json del
# diagnóstico): el mismo con que se generaron las instantáneas.
CRIT_93 = [
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
C93 = [f6.Criterio(problema=d["problema"], sentido=d["sentido"],
                   razonamiento=d["razonamiento"], jerarquia=d["jerarquia"]) for d in CRIT_93]
P93 = [{"pregunta": C93[0].problema, "cubre": [1]}, {"pregunta": C93[1].problema, "cubre": [2]}]
ACTO = "LO QUE RESOLVIÓ LA SALA (resumen de prueba)."
CONC = "LO QUE SE COMBATE (resumen de prueba)."


def mat(formato="estandar", variante="v1", **kw):
    base = dict(tipo_asunto="amparo_directo", materia="administrativa", formato=formato,
                problemas=P93, n_planteamientos=2, variante=variante)
    base.update(kw)
    return f6.Material(**base)


# ═══════════════════════════════════════════════════════════════════════════
print("\n1 · LA v1 ESTÁ CONGELADA: SÓLO CAMBIA LA FRASE DEL ARTÍCULO 189")
FRASE_VIEJA = ("privilegiando el\n       estudio de las violaciones procesales que inciden "
               "en el sentido del fallo.»")
FRASE_NUEVA = ("privilegiando el\n       estudio de los de fondo sobre los de procedimiento y "
               "forma, como lo\n       ordena el artículo 189 de esa ley.» (Si inviertes ese "
               "orden porque\n       estudiar primero una violación procesal redunda en un "
               "mayor beneficio\n       para la parte quejosa, dilo y di en qué consiste ese "
               "beneficio.)")
for forma in ("estandar", "moderna"):
    ruta = os.path.join(AQUI, "datos", "instantaneas", f"estudio_v1_93_{forma}.prompt")
    antes = open(ruta, encoding="utf-8").read()
    ahora = f6.prompt_estudio(ACTO, CONC, C93, mat(forma))
    ok(FRASE_VIEJA in antes and antes.count(FRASE_VIEJA) == 1,
       f"{forma}: la instantánea de antes (origin/main 9833dfb) trae la fórmula vieja una vez")
    ok(FRASE_NUEVA not in antes, f"{forma}: y no la nueva")
    ok(ahora == antes.replace(FRASE_VIEJA, FRASE_NUEVA),
       f"{forma}: la v1 de hoy es la de antes con esa frase cambiada y NADA MÁS")
    ok(f6.prompt_estudio(ACTO, CONC, C93, mat(forma, variante="")) == ahora
       and f6.prompt_estudio(ACTO, CONC, C93, mat(forma, variante="v7")) == ahora,
       f"{forma}: una variante vacía o desconocida es la v1")
m_sin = f6.Material(tipo_asunto="amparo_directo", materia="administrativa", formato="estandar",
                    problemas=P93, n_planteamientos=2)
ok(m_sin.variante == "v1", "el material nace en v1")
ok("con la MISMA técnica de los cuatro pasos\nque usaste con los agravios."
   in f6._bloque_conceptos("revoca_sobreseimiento", "Primer concepto: x " * 30),
   "el bloque de conceptos de la v1 no cambia")
ok(f6._bloque_arquitectura("laboral") == f6._ARQUITECTURA_COMUN + f6._ARQUITECTURA["laboral"],
   "la arquitectura de la v1 no cambia")

# ═══════════════════════════════════════════════════════════════════════════
print("\n2 · LA v2 NO TRAE NINGUNA DE LAS ÓRDENES RETIRADAS")
RETIRADAS = {
    "TRANSCRIBE LA FUENTE": "fila 6: transcribir el precepto",
    "Tras cada transcripción": "fila 6: la regla 2 con transcripción",
    "Del precepto\n   transcrito deriva la regla": "fila 6: el molde «Del precepto transcrito»",
    "TITULA POR FUNCIÓN": "fila 7",
    "RESPONDE LA MEJOR OBJECIÓN DEL QUE PIERDE": "fila 8: la regla 9",
    "Por cada cuestión, un párrafo": "fila 8",
    "la objeción previsible": "fila 8: la objeción en la extensión y en el principal",
    "UN AGRAVIO, UN TRAMO": "fila 5: la regla 8",
    "transcribe entre comillas lo\n   que alegó": "fila 5",
    "Prohibido resolver dos con una calificación conjunta": "fila 5",
    "entre TRES y SEIS": "fila 9",
    "Invoca entre TRES y SEIS": "fila 9: el recordatorio final",
    "HAZLA HABLAR": "fila 10",
    "Del criterio transcrito se desprende": "fila 10: el ejemplo de :1978",
    "Y LO ÚLTIMO, QUE ES LO QUE MÁS SE ROMPE": "fila 10: la segunda copia de la regla",
    "Por cuestión de método": "fila 14: molde",
    "bloques\n       temáticos": "fila 14: molde",
    "se procede al análisis conjunto": "fila 14: molde",
    "unidad de\n      cuantificación": "fila 14b: el ejemplo de la moderna",
    "artículo 50 de la\n      Ley Federal": "fila 14b",
    "«1. Deje insubsistente…»": "fila 14b: los efectos de ejemplo",
    "el apartado CIERRA con lo que se sigue": "fila 16",
    "EFECTOS: si se concede, enumera": "fila 17",
    "CIERRA con UNA SOLA calificación": "fila 17",
    "Y SI EL ASUNTO ES LABORAL Y PROMUEVE EL TRABAJADOR": "fila 18: suplencia por tercera vez",
    "Y LOS ARTÍCULOS: SU NÚMERO Y SU LEY, JUNTOS, SIEMPRE": "fila 18: el duplicado final",
    "Y HAY MATERIAS DONDE LA INOPERANCIA POR DEFICIENCIA NO CABE": "fila 18: fundida en una",
    "SUPLENCIA DE LA QUEJA: si el asunto toca": "fila 18: fundida en una",
    "3733": "fila 12: la meta de 3,733",
    "Alrededor de": "fila 12: la meta como objetivo",
    "ENTRE TRES Y SIETE PÁRRAFOS": "fila 13: una de las cuatro medidas",
    "SE CITA UNA TESIS SOBRE LA INOPERANCIA": "fila 13",
    "CUATRO PASOS": "filas 2-3",
    "SIEMPRE LOS CUATRO": "filas 2-3",
    "PASO 2 — LA PREMISA NORMATIVA": "filas 2-3: la premisa en cada apartado",
    "EN SU VERSIÓN MÁS FUERTE": "fila 4: la apertura larga",
    "UN APARTADO POR CONCEPTO": "la unidad deja de ser el concepto",
    "Ningún concepto de violación sin su apartado": "fila 19 (sin plan): lo último que leía",
    "Ningún agravio sin su apartado": "ídem en recursos",
    "Tu último párrafo SÍ recapitula": "decisión 3",
    "CÓMO TERMINA EL ESTUDIO": "decisión 3: las TRES frases",
    "El último párrafo, antes del resolutivo, dice en TRES frases": "decisión 3",
    "violaciones procesales que inciden en el sentido del fallo": "el orden contrario al 189",
    "Cinco registros distintos como mínimo": "materia laboral: cuota de citas",
    "tesis transcritas como mínimo": "materia civil: cuota de citas",
    "TRANSCRIBIENDO": "materia administrativa: transcribir el precepto",
    "Título descriptivo de lo que se decide": "materia administrativa: rótulo",
    "CIERRE. Dos partes obligatorias": "materia laboral: cierre obligatorio",
    "anúnciala en la misma frase del veredicto y fúndala con precepto y tesis: 53%\n"
    "arriba contra 25%. Si no cae, no la menciones.": "materia civil: suplencia sin beneficio",
    "de los cuatro pasos": "el bloque de conceptos remitía a los cuatro pasos",
    "no los contestes por separado dentro de él": "el grupo borraba la respuesta de cada uno",
    "ni lo califiques ni lo contestes": "lo innecesario sin su calificación",
}
C_CONCEDE = [f6.Criterio("¿La Sala debió admitir la ampliación?", "fundado", "Razón A", "principal"),
             f6.Criterio("¿La multa es excesiva?", "infundado", "Razón B", "accesorio", grupo="A"),
             f6.Criterio("¿Se valoró la pericial?", "inoperante", "Razón C", "accesorio", grupo="A"),
             f6.Criterio("¿Debía atender alegatos?", "innecesario", "", "accesorio")]
C_NIEGA = [f6.Criterio("¿Prescribió la acción?", "infundado", "Razón", "principal")]
C_LAB = [f6.Criterio("¿El despido fue justificado?", "inoperante", "Razón", "principal")]
PROMPTS_V2 = {}
for tipo, es_rec, rama in (("amparo_directo", False, ""), ("revision_fiscal", True, ""),
                           ("amparo_revision", True, "revoca_sobreseimiento"),
                           ("queja", True, "")):
    for forma in ("estandar", "moderna"):
        for materia in ("", "laboral", "administrativa", "civil", "penal"):
            for nombre, crit in (("concede", C_CONCEDE), ("niega", C_NIEGA), ("laboral", C_LAB)):
                m = f6.Material(tipo_asunto=tipo, materia=materia, formato=forma,
                                problemas=P93, n_planteamientos=4, variante="v2")
                PROMPTS_V2[(tipo, forma, materia, nombre)] = f6.prompt_estudio(
                    ACTO, CONC, crit, m, es_recurso=es_rec, rama=rama,
                    conceptos_violacion="Primer concepto de violación: " + "x " * 40,
                    marco="MATERIAL CONSTITUCIONAL")
sobreviven = {}
for clave, p in PROMPTS_V2.items():
    for frase, fila in RETIRADAS.items():
        if frase in p:
            sobreviven.setdefault(frase, (fila, clave))
ok(not sobreviven, f"ninguna orden retirada sobrevive en {len(PROMPTS_V2)} combinaciones "
   f"(tipo × forma × materia × sentido)"
   + (": " + " · ".join(f"«{k[:40]}» ({v[0]}) en {v[1]}" for k, v in sobreviven.items())
      if sobreviven else ""))
_casan = []
for _viejo, _nuevo in f6._ARQUITECTURA_V2_CAMBIOS:
    _m = next((k for k, t in f6._ARQUITECTURA.items() if _viejo in t), None)
    _casan.append(_m is not None and _nuevo in f6._arquitectura_materia_v2(f6._ARQUITECTURA[_m])
                  and _viejo not in f6._arquitectura_materia_v2(f6._ARQUITECTURA[_m]))
ok(all(_casan), "cada sustitución de la arquitectura de materia casa con el texto de la v1")
ok(sum(1 for viejo, _ in f6._ARQUITECTURA_V2_CAMBIOS
       if any(viejo in t for t in f6._ARQUITECTURA.values())) == len(f6._ARQUITECTURA_V2_CAMBIOS),
   "y ninguna sustitución se quedó sin su frase de origen")

# ═══════════════════════════════════════════════════════════════════════════
print("\n3 · LA v2 CONSERVA LO QUE LA PROPUESTA MANDA CONSERVAR")
p_std = f6.prompt_estudio(ACTO, CONC, C93, mat("estandar", "v2"))
p_mod = f6.prompt_estudio(ACTO, CONC, C93, mat("moderna", "v2"))
for forma, p in (("estándar", p_std), ("moderna", p_mod)):
    for frase, que in (("FRASE de unas 35 palabras", "las medidas de frase"),
                       ("PÁRRAFO de unas 49", "y de párrafo"),
                       ("CONECTORES, por orden de uso real", "los conectores"),
                       ("LA LEY QUE GOBIERNA ESTA VÍA", "la ley de la vía"),
                       ("NUNCA inventes un registro digital", "no inventar registros"),
                       ("LA LEY AJENA NO ENTRA; EL CRITERIO AJENO SÍ", "ley ajena y criterio ajeno"),
                       ("ASÍ SE CITA, Y NO DE OTRA FORMA", "el formato de cita"),
                       ("EL REGISTRO DIGITAL VA SIEMPRE", "el registro siempre"),
                       ("EN REVISIÓN SE REVOCA; SÓLO EN AMPARO SE DEJA INSUBSISTENTE", "revoca frente a deja insubsistente"),
                       ("EFECTOS DE LA CONCESIÓN", "el rótulo EFECTOS"),
                       ("«ADVERTENCIAS»", "el rótulo ADVERTENCIAS"),
                       ("EL CUERPO NO TRANSCRIBE", "el cuerpo no transcribe"),
                       ("EL CRITERIO DEL SECRETARIO — DIRECTIVA INNEGOCIABLE", "el criterio manda"),
                       ("RAZÓN DEL SECRETARIO:", "la razón del secretario"),
                       ("ARTÍCULO 76 DE LA LEY DE AMPARO", "el anuncio de método con el 76"),
                       ("artículo 189 de la Ley de Amparo", "el orden del 189"),
                       ("mayor beneficio", "y su excepción"),
                       ("SE DECIDEN TODAS (artículos 74, fracción\n  V, y 174", "las procesales, todas"),
                       ("CADA ARGUMENTO, UNA RESPUESTA IDENTIFICABLE", "fila 5"),
                       ("SÓLO DONDE SE EXPONE POR PRIMERA VEZ", "filas 2-3"),
                       ("recuerda en una frase la proposición concreta", "filas 2-3: el recordatorio"),
                       ("LA OBJECIÓN, UNA VEZ", "fila 8"),
                       ("como máximo dos", "fila 9"),
                       ("DESPUÉS DE LA CITA, NO LA REPITAS: ÚSALA", "fila 10: una sola regla"),
                       ("ÉSTA es la ÚNICA medida".lower(), "fila 13: una sola tabla"),
                       ("SÓLO SE EXPRESA", "suplencia: art. 79, último párrafo"),
                       ("SIN PÁRRAFO DE CIERRE", "decisión 3: sin cierre por defecto")):
        ok(frase.lower() in p.lower(), f"{forma}: {que}")
    ok(p.rstrip().endswith("Nada más."), f"{forma}: termina igual")
    ok(p.count("LA OBJECIÓN, UNA VEZ") == 1, f"{forma}: la objeción se ordena en un solo sitio del cuerpo")
ok(f"TIENE UN TECHO: {f6.SOLUCION_P90} palabras" in p_std and "techo, no una meta" in p_std,
   "estándar: el techo medido, dicho como techo")
ok(f"TIENE UN TECHO: {fs.palabras_moderna(C93)} palabras" in p_mod, "moderna: su propio techo")
ok("Sobre el primer concepto de violación,\n       en el que la parte quejosa sostiene que…" in p_std,
   "estándar: la fórmula de David sigue abriendo el apartado de un solo concepto")
ok("no más de treinta\n       palabras" in p_std, "fila 4: la apertura, treinta palabras como máximo")
ok("NO EL CONCEPTO DE VIOLACIÓN" in p_std and "Sobre los conceptos de violación segundo y\n       tercero"
   in p_std, "estándar: la unidad es la consideración atacada, nombrando los conceptos que entran")
ok("FORMATO: ESTÁNDAR" in p_std and "FORMATO: VERSIÓN MODERNA" not in p_std, "estándar: su bloque")
ok("FORMATO: VERSIÓN MODERNA" in p_mod and "CADA PROBLEMA ABRE CON SU PREGUNTA" in p_mod,
   "moderna: cada apartado sigue siendo un problema con su pregunta")
ok("CUBRE: el primer concepto de violación" in p_std, "la línea CUBRE sigue (su cambio es del Paso 3)")
_p_rf = PROMPTS_V2[("revision_fiscal", "estandar", "administrativa", "concede")]
ok("la quejosa" not in _p_rf.lower().replace("la parte quejosa", ""),
   "revisión fiscal: ningún texto nuevo nombra quejosa a la autoridad")
ok("EN UN RECURSO" in _p_rf and "74, fracción" not in _p_rf,
   "revisión fiscal: el orden lo fija su técnica; el 174 es del amparo directo")
_p_lab = PROMPTS_V2[("amparo_directo", "estandar", "laboral", "laboral")]
ok("UNA SALVEDAD, Y SÓLO UNA" in _p_lab,
   "la salvedad de suplencia del criterio (de otro cambio) llega igual a la v2")
ok("Primer concepto de violación: x" in PROMPTS_V2[("amparo_revision", "estandar", "", "niega")]
   and "MISMA forma de construir cada apartado" in PROMPTS_V2[("amparo_revision", "estandar", "", "niega")],
   "revoca sobreseimiento: los conceptos llegan, sin remitir a los cuatro pasos")
_p_conc = PROMPTS_V2[("amparo_directo", "estandar", "administrativa", "concede")]
ok("SE ESTUDIA JUNTO CON LOS DEMÁS DEL GRUPO A" in _p_conc and "respuesta identificable" in _p_conc,
   "el grupo del secretario: un apartado, y cada argumento con su respuesta")
ok("REPOSICIÓN en el orden en que ha de cumplirse" in _p_conc, "los efectos, descritos")
ok(len(p_std) < len(f6.prompt_estudio(ACTO, CONC, C93, mat("estandar"))),
   f"la v2 es más corta que la v1 ({len(p_std)} contra "
   f"{len(f6.prompt_estudio(ACTO, CONC, C93, mat('estandar')))} caracteres)")

# ═══════════════════════════════════════════════════════════════════════════
print("\n4 · LA v2 NO METE FRASES NUEVAS PARA COPIAR")
# Todo lo que va entre comillas angulares en la v2 y no estaba en la v1 tiene
# que ser una de éstas: palabras sueltas, fórmulas que David pidió o frases que
# se PROHÍBEN. Una frase modelo nueva haría fallar esto.
PERMITIDAS = {"Sí.", "No.", "dicho numeral", "transcrito", "en su versión más favorable",
              "la mejor objeción a esta conclusión", "los lineamientos de esta ejecutoria",
              "Sobre el primer agravio,\n       en el que la parte recurrente sostiene que…",
              "Sobre el primer agravio,\n       en el que la autoridad recurrente sostiene que…",
              "Sobre el primer concepto de violación,\n       en el que la parte quejosa sostiene que…",
              "Sobre los agravios segundo y\n       tercero, en los que…",
              "Sobre los conceptos de violación segundo y\n       tercero, en los que…",
              "Sobre el primer agravio, en el que la parte recurrente sostiene…",
              "Sobre el primer agravio, en el que la autoridad recurrente sostiene…",
              "Sobre el primer concepto de violación, en el que la parte quejosa sostiene…"}
_rx_q = re.compile(r"«([^«»]{1,300})»")


def _citas(t: str) -> set:
    # El salto de renglón no cambia la frase: se compara con los espacios
    # normalizados, o un simple re-ajuste de línea contaría como frase nueva.
    return {" ".join(x.split()) for x in _rx_q.findall(t)}


PERMITIDAS = {" ".join(x.split()) for x in PERMITIDAS}
nuevas = set()
for clave, p in PROMPTS_V2.items():
    _v1 = f6.prompt_estudio(ACTO, CONC, [C_CONCEDE, C_NIEGA, C_LAB][
        ["concede", "niega", "laboral"].index(clave[3])],
        f6.Material(tipo_asunto=clave[0], materia=clave[2], formato=clave[1],
                    problemas=P93, n_planteamientos=4),
        es_recurso=clave[0] != "amparo_directo",
        rama="revoca_sobreseimiento" if clave[0] == "amparo_revision" else "",
        conceptos_violacion="Primer concepto de violación: " + "x " * 40,
        marco="MATERIAL CONSTITUCIONAL")
    nuevas |= _citas(p) - _citas(_v1)
ok(not (nuevas - PERMITIDAS), "ningún entrecomillado nuevo fuera de la lista"
   + (": " + " · ".join(sorted(x[:50].replace("\n", " ") for x in nuevas - PERMITIDAS))
      if nuevas - PERMITIDAS else ""))

# ═══════════════════════════════════════════════════════════════════════════
print("\n5 · EL CIERRE: SÓLO CON TRES APARTADOS O MÁS DE RESULTADO DISTINTO")
C3 = [f6.Criterio("a", "fundado"), f6.Criterio("b", "infundado"), f6.Criterio("c", "infundado")]
ok(f6._cierre_permitido(C3), "tres apartados, dos resultados → cierre breve")
ok(not f6._cierre_permitido([f6.Criterio(x, "infundado") for x in "abcd"]),
   "cuatro apartados con el mismo resultado → sin cierre")
ok(not f6._cierre_permitido(C93), "el 93 (dos apartados) → sin cierre")
ok(not f6._cierre_permitido([f6.Criterio("a", "fundado"), f6.Criterio("b", "infundado", grupo="G"),
                             f6.Criterio("c", "infundado", grupo="G")]),
   "un grupo del secretario cuenta como un apartado")
_p3 = f6.prompt_estudio(ACTO, CONC, C3, mat("estandar", "v2"))
ok("UN CIERRE BREVE" in _p3 and "tres frases como\nmáximo" in _p3 and "SIN PÁRRAFO DE CIERRE" not in _p3,
   "con cierre permitido, el prompt lo dice y lo acota")
ok("UN CIERRE BREVE" not in p_std, "sin él, el prompt lo prohíbe")

# ═══════════════════════════════════════════════════════════════════════════
print("\n6 · LA VARIANTE: SÓLO CASA LA ELIGE, Y VIAJA POR EL MATERIAL")
ok(f6.normalizar_variante("V2") == "v2" and f6.normalizar_variante("B") == "v2"
   and f6.normalizar_variante("a") == "v1" and f6.normalizar_variante("v9") == "",
   "se normaliza sin inventar ninguna")
ok(f6.variante_global() == "v1", "sin ESTUDIO_PROMPT, la global es la v1")
os.environ["ESTUDIO_PROMPT"] = "v2"
ok(f6.variante_global() == "v2", "ESTUDIO_PROMPT=v2 la cambia para todos")
os.environ["ESTUDIO_PROMPT"] = "tonteria"
ok(f6.variante_global() == "v1", "un valor que no existe no apaga nada: v1")
os.environ.pop("ESTUDIO_PROMPT", None)

# Las funciones de main.py, tomadas del archivo y ejecutadas aparte: importar
# main arranca clientes y variables que esta prueba no necesita.
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
                                                     "_taller_meta_listo", "_criterios_json_para_ficha"):
        _piezas.append(n)
    if isinstance(n, ast.Assign) and any(getattr(t, "id", "") == "_FICHA_TEXTO_MAX" for t in n.targets):
        _piezas.append(n)
exec(compile(ast.Module(body=_piezas, type_ignores=[]), "main.py", "exec"), _ns)
_ns["json"] = json
_v = _ns["_taller_variante_estudio"]
ok(_v("jdm.juridico@gmail.com", "v2") == "v2", "David pide la v2 → v2")
ok(_v("administracion@iurexia.com", "v2") == "v2" and _v("soporte@iurexia.com", "B") == "v2",
   "administracion@ y soporte@ son de casa")
ok(_v("secretario.piloto@gmail.com", "v2") == "v1", "un secretario del piloto pide v2 → se ignora, v1")
ok(_v("jdm.juridico@gmail.com", "") == "v1" and _v("jdm.juridico@gmail.com", "v9") == "v1",
   "casa sin pedir, o pidiendo algo que no existe → la global")
os.environ["ESTUDIO_PROMPT"] = "v2"
ok(_v("secretario.piloto@gmail.com", "v1") == "v2", "con la global en v2, el piloto recibe v2 aunque pida v1")
ok(_v("jdm.juridico@gmail.com", "v1") == "v1", "y casa puede volver a la v1 para comparar")
os.environ.pop("ESTUDIO_PROMPT", None)

import redactor_adelanto as ra


class _R:
    pass


r = _R()
import fases123_pipeline as f123
r.fases = f123.Fases123(conteo={"estado": "contado", "n": 2})
r.fases.problemas = P93
import types
r.encargo = types.SimpleNamespace(formato="", variante_estudio="v2", tipo_asunto="amparo_directo",
                                  es_recurso=False)
m_v = f6.Material()
ra._formato_al_material(r, m_v, None, C93)
ok(m_v.variante == "v2", "el encargo con v2 pone la v2 en el material")
r.encargo.variante_estudio = ""
ra._formato_al_material(r, m_v, None, C93)
ok(m_v.variante == "v1", "y la vuelta siguiente sin variante vuelve a la global: no se cuela la anterior")
os.environ["ESTUDIO_PROMPT"] = "v2"
ra._formato_al_material(r, m_v, None, C93)
ok(m_v.variante == "v2", "sin variante en el encargo, manda ESTUDIO_PROMPT")
os.environ.pop("ESTUDIO_PROMPT", None)
del r.encargo.variante_estudio
ra._formato_al_material(r, m_v, None, C93)
ok(m_v.variante == "v1", "un encargo viejo, sin el campo, va con la global")

# ═══════════════════════════════════════════════════════════════════════════
print("\n7 · LOS DOS GEMELOS LA PASAN IGUAL")
eps = {n.name: n for n in ast.walk(ARBOL_MAIN)
       if isinstance(n, ast.AsyncFunctionDef) and n.name in ("taller_resolver_stream", "taller_resolver")}
ok(set(eps) == {"taller_resolver_stream", "taller_resolver"}, "están los dos endpoints")
for nombre, fn in eps.items():
    args = [a.arg for a in fn.args.args]
    ok("variante_estudio" in args, f"{nombre}: recibe `variante_estudio`")
    asigna = [n for n in ast.walk(fn) if isinstance(n, ast.Assign)
              and any(isinstance(t, ast.Attribute) and t.attr == "variante_estudio" for t in n.targets)]
    ok(len(asigna) == 1 and isinstance(asigna[0].value, ast.Call)
       and getattr(asigna[0].value.func, "id", "") == "_taller_variante_estudio"
       and [getattr(a, "id", "") for a in asigna[0].value.args] == ["user_email", "variante_estudio"],
       f"{nombre}: la pone en el encargo con `_taller_variante_estudio(user_email, variante_estudio)`")
    guarda = [n for n in ast.walk(fn) if isinstance(n, ast.Call)
              and getattr(n.func, "id", "") == "_taller_guardar_proyecto"]
    ok(len(guarda) == 1 and any(k.arg == "criterios_json" for k in guarda[0].keywords),
       f"{nombre}: la ficha recibe el `criterios_json` completo")
    ok(any(isinstance(n, ast.Call) and getattr(n.func, "id", "") == "_taller_meta_listo"
           for n in ast.walk(fn)),
       f"{nombre}: devuelve variante, commit y finish_reason (evento «listo» o cabeceras)")
# El encargo se asigna en los dos DENTRO del mismo `if r.encargo is not None`
# que la forma: una vuelta anterior no se cuela.
ok(SRC_MAIN.count("r.encargo.variante_estudio = _taller_variante_estudio(") == 2,
   "las dos asignaciones, una por gemelo")

SRC_RA = open(os.path.join(AQUI, "redactor_adelanto.py"), encoding="utf-8").read()
ARBOL_RA = ast.parse(SRC_RA)
redac = {n.name: n for n in ast.walk(ARBOL_RA)
         if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name in ("resolver", "resolver_en_vivo")}
for nombre, fn in redac.items():
    term = [n for n in ast.walk(fn) if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "_terminar"]
    ok(len(term) == 1 and any(k.arg == "meta_estudio" for k in term[0].keywords),
       f"{nombre}: pasa lo anotado del estudio a `_terminar`")
    lit = [n for n in ast.walk(fn) if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "_litis_y_material"]
    ok(len(lit) == 1, f"{nombre}: pasa por `_litis_y_material` (que fija forma y variante)")
_red = [n for n in ast.walk(redac["resolver"]) if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute) and n.func.attr == "redactar"]
ok(len(_red) == 1 and any(k.arg == "meta" for k in _red[0].keywords),
   "resolver: pide a `f6.redactar` lo anotado (`meta=`)")
ok("_meta = dict(paso.get(\"meta\") or {})" in SRC_RA, "resolver_en_vivo: lo recoge del evento «fin»")
ok("material.variante = _f6v.normalizar_variante(" in SRC_RA, "`_formato_al_material` fija la variante")

SRC_F6 = open(os.path.join(AQUI, "fase6_estudio.py"), encoding="utf-8").read()
ARBOL_F6 = ast.parse(SRC_F6)
for nombre in ("redactar", "redactar_en_vivo"):
    fn = next(n for n in ast.walk(ARBOL_F6) if isinstance(n, ast.AsyncFunctionDef) and n.name == nombre)
    llamadas = [n for n in ast.walk(fn) if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "prompt_estudio"]
    ok(len(llamadas) == 1 and getattr(llamadas[0].args[3], "id", "") == "material",
       f"{nombre}: arma el prompt con el material, que es quien trae la variante")
    ok(any(isinstance(n, ast.Call) and getattr(n.func, "id", "") == "_meta_vacia" for n in ast.walk(fn)),
       f"{nombre}: anota variante, finish_reason y tokens")

# ═══════════════════════════════════════════════════════════════════════════
print("\n8 · LOS CONTROLES AJUSTADOS NO ACUSAN A LA SALIDA BUENA")
m_std2 = mat("estandar", "v2", tesis=[{"registro": "2022074", "obligatoria": True,
                                       "rubro": "RUBRO DE PRUEBA LARGO PARA QUE CUENTE COMO RUBRO",
                                       "texto": "texto"}])
m_std1 = mat("estandar", "v1", tesis=m_std2.tesis)
_una_cita = "Los conceptos son infundados. Sirve de apoyo el criterio de registro 2022074: "
corto = _una_cita + "palabra " * 1100
ok(not any("Se quedó corto" in a or "ningún engrose real" in a for a in f6.revisar(corto, C_NIEGA, m_std2)),
   "v2: 1,100 palabras no son «cortas» (ningún engrose Kingston baja de 1,303)")
ok(any("Se quedó corto" in a for a in f6.revisar(corto, C_NIEGA, m_std1)), "v1: sí lo eran (45 % de 3,733)")
ok(any("ningún engrose real" in a for a in f6.revisar(_una_cita + "p " * 900, C_NIEGA, m_std2)),
   "v2: 900 palabras sí")
ok(not any("entre tres y seis" in a for a in f6.revisar(corto, C_NIEGA, m_std2)),
   "v2: una sola cita no se acusa (cada premisa, un apoyo)")
ok(any("entre tres y seis" in a for a in f6.revisar(corto, C_NIEGA, m_std1)), "v1: sí")
ok(not any("NI UNA TESIS" in a for a in f6.revisar("Los conceptos son infundados. " + "p " * 2000,
                                                   C_NIEGA, m_std2)),
   "v2: ni siquiera cero citas es aviso visible (en sombra hasta calibrarlo)")
_osc = "Los conceptos son infundados. " + "p " * 2000 + " La Sala lo estimó fundado; es infundado."
ok(not any("oscila" in a for a in f6.revisar(_osc, C_NIEGA, m_std2)), "v2: sin cierre no hay cierre que oscile")
ok(any("oscila" in a for a in f6.revisar(_osc, C_NIEGA, m_std1)), "v1: el aviso sigue igual")
ok(not any("techo de la" in a for a in f6.revisar(_una_cita + "p " * 6000, C_NIEGA, m_std2)),
   f"v2: 6,000 palabras no pasan de {f6.SOLUCION_EXCESO}")
ok(any("techo de la" in a for a in f6.revisar(_una_cita + "p " * 7000, C_NIEGA, m_std2)),
   "v2: 7,000 sí")
ok(any("sólo el 10%" in a for a in f6.revisar(_una_cita + "p " * 7000, C_NIEGA, m_std1)),
   "v1: su aviso de exceso sigue igual")
_v2_formula = ("Los conceptos de violación son infundados.\n"
               "Sobre el primer concepto de violación, en el que la parte quejosa sostiene X. "
               "Se considera infundado. Lo anterior…\n"
               "Sobre el segundo concepto de violación, en el que sostiene Y. Se considera "
               "infundado. Lo anterior…\n"
               "Sobre el tercer concepto de violación, en el que sostiene Z. Se estima inoperante.")
ok(ce.exhaustividad(_v2_formula)["contesta_todo"] is False
   and ce.exhaustividad(_v2_formula, amplia=True)["contesta_todo"] is True,
   "la calificación de la fórmula de David («Se considera…») cuenta en la v2")

# CONTRA LOS ENGROSES DE VERDAD. Se mide la Solución de los 24 de oro sólido
# del banco Kingston con la misma regla con que se fijaron las constantes.
CASOS = "/Users/josedavidalcantarmendoza/Documents/IUREXIA-MAC/redactor-sentencias/corpus/casos"
UTIL = "/Users/josedavidalcantarmendoza/Documents/IUREXIA-MAC/redactor-sentencias/utillaje"
_RX_SOL = re.compile(r"^(?:[IVX]+\.\s*)?Soluci[óo]n\.?\s*$", re.I)
_RX_FIN = re.compile(r"R\s*E\s*S\s*U\s*E\s*L\s*V\s*E|^Por lo expuesto[^.]{0,120}(?:resuelve|RESUELVE)"
                     r"|^(?:S[ÉEé]PTIMO|OCTAVO)\.")


def solucion(oro: str) -> str:
    ls = [x.strip() for x in oro.split("\n") if x.strip()]
    ini = next((i + 1 for i, x in enumerate(ls) if _RX_SOL.match(x)), 1)
    fin = next((j for j in range(ini, len(ls)) if _RX_FIN.search(ls[j])), len(ls))
    return "\n".join(ls[ini:fin])


if os.path.isdir(CASOS) and os.path.isdir(UTIL):
    sys.path.insert(0, UTIL)
    from comparar import calificaciones
    vistos, banco = set(), []
    for f in sorted(glob.glob(f"{CASOS}/*.json")):
        if os.path.basename(f).startswith("_"):
            continue
        c = json.load(open(f))
        if not isinstance(c, dict) or not c.get("oro") or c["asunto"] in vistos:
            continue
        if sum(calificaciones(c["oro"]).values()) < 3:
            continue
        vistos.add(c["asunto"])
        banco.append((c["asunto"], solucion(c["oro"])))
    ok(len(banco) == 24, f"el banco tiene sus 24 engroses de oro sólido ({len(banco)})")
    ns = sorted(len(t.split()) for a, t in banco if not a.startswith("ADC 810"))
    k = (len(ns) - 1) * 0.9
    p90 = ns[int(k)] + (ns[min(int(k) + 1, len(ns) - 1)] - ns[int(k)]) * (k - int(k))
    ok(ns[len(ns) // 2] == f6.SOLUCION_MEDIANA and round(p90) == f6.SOLUCION_P90,
       f"las constantes son las medidas: mediana {ns[len(ns) // 2]}, p90 {round(p90)}")
    acusa = {"corto": [], "exceso": [], "citas": [], "oscila": [], "exhaust": []}
    for asunto, t in banco:
        av = f6.revisar(t, C_NIEGA, m_std2)
        if any("ningún engrose real" in a for a in av):
            acusa["corto"].append(asunto)
        if any("techo de la" in a for a in av):
            acusa["exceso"].append(asunto)
        if any("entre tres y seis" in a or "NI UNA TESIS" in a for a in av):
            acusa["citas"].append(asunto)
        if any("oscila" in a for a in av):
            acusa["oscila"].append(asunto)
        if ce.exhaustividad(t, amplia=True)["contesta_todo"] is False:
            acusa["exhaust"].append(asunto)
    ok(not acusa["corto"], "«se quedó corto» (v2): no acusa a ninguno de los 24")
    ok(not acusa["citas"], "pocas citas (v2): no acusa a ninguno (queda en sombra)")
    ok(not acusa["oscila"], "el cierre que oscila (v2): no acusa a ninguno")
    ok(sorted(a[:6] for a in acusa["exceso"]) == ["ADC 43", "ADC 59", "ADC 81"],
       f"el exceso (v2) acusa sólo a los tres que ya acusaba la v1: {acusa['exceso']}")
    ok([a[:7] for a in acusa["exhaust"]] == ["ADC 722"],
       f"la exhaustividad amplia acusa sólo al que ya acusaba la estrecha: {acusa['exhaust']}")
else:
    print("   (el corpus Kingston no está en esta máquina: la calibración no se corrió)")

# ═══════════════════════════════════════════════════════════════════════════
print("\n9 · LO QUE SE ANOTA: VARIANTE, FINISH_REASON Y TOKENS")


class _U:
    prompt_tokens = 30000
    completion_tokens = 9000
    total_tokens = 39000
    completion_tokens_details = type("D", (), {"reasoning_tokens": 5000})()
    prompt_tokens_details = {"cached_tokens": 1200}


class _Resp:
    def __init__(self, t):
        self.choices = [type("C", (), {"message": type("M", (), {"content": t})(),
                                       "finish_reason": "stop"})()]
        self.usage = _U()


class _Trozo:
    def __init__(self, t=None, fin=None, uso=None):
        self.usage = uso
        self.choices = [] if t is None and fin is None else [
            type("C", (), {"delta": type("D", (), {"content": t})(), "finish_reason": fin})()]


class _Flujo:
    def __init__(self, trozos):
        self.trozos = list(trozos)

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self.trozos:
            raise StopAsyncIteration
        return self.trozos.pop(0)


class _Cli:
    def __init__(self, rechaza_opciones=False):
        self.kw = []
        self.rechaza = rechaza_opciones
        cli = self

        class _C:
            @staticmethod
            async def create(**kw):
                cli.kw.append(dict(kw))
                if kw.get("stream"):
                    if cli.rechaza and "stream_options" in kw:
                        raise RuntimeError("Unrecognized request argument supplied: stream_options")
                    return _Flujo([_Trozo("Los conceptos "), _Trozo("son infundados."),
                                   _Trozo(None, "length"), _Trozo(uso=_U())])
                return _Resp("Los conceptos son infundados.")
        self.chat = type("Ch", (), {"completions": _C})()


async def _vivo(cli, m):
    fin = None
    async for paso in f6.redactar_en_vivo(cli, ACTO, CONC, C93, m):
        if paso["tipo"] == "fin":
            fin = paso
    return fin


_fin = asyncio.run(_vivo(_Cli(), mat("estandar", "v2")))
ok(_fin["meta"] == {"variante": "v2", "finish_reason": "length",
                    "uso": {"entrada": 30000, "salida": 9000, "total": 39000,
                            "razonamiento": 5000, "en_cache": 1200}},
   "en vivo: variante, `length` y tokens salen en el evento «fin»")
_cli_r = _Cli(rechaza_opciones=True)
_fin2 = asyncio.run(_vivo(_cli_r, mat("estandar", "v1")))
ok(_fin2["estudio"] == "Los conceptos son infundados." and len(_cli_r.kw) == 2
   and "stream_options" not in _cli_r.kw[1],
   "si el proveedor no admite `include_usage`, se llama sin él y el estudio sale igual")
ok(_cli_r.kw[0].get("reasoning_effort") == f6.ESFUERZO_ESTUDIO == "high",
   "el razonamiento del estudio sigue en «high»")
_meta = {}
asyncio.run(f6.redactar(_Cli(), ACTO, CONC, C93, mat("moderna", "v2"), meta=_meta))
ok(_meta.get("variante") == "v2" and _meta.get("finish_reason") == "stop"
   and _meta.get("uso", {}).get("razonamiento") == 5000,
   "el gemelo plano anota lo mismo")
_res = type("Res", (), {"meta_estudio": {"variante": "v2", "finish_reason": "stop", "uso": {"total": 1}}})()
os.environ["RENDER_GIT_COMMIT"] = "abc123"
ok(_ns["_taller_meta_listo"](_res) == {"variante": "v2", "commit": "abc123", "finish_reason": "stop",
                                       "uso": {"total": 1}},
   "el evento «listo» y la ficha llevan variante, commit, finish_reason y uso")
os.environ.pop("RENDER_GIT_COMMIT", None)
_cj = _ns["_criterios_json_para_ficha"](json.dumps(CRIT_93 + [{"problema": "x", "sentido": "fundado",
                                                               "tocado": True, "grupo": "A",
                                                               "razonamiento": "r" * 20000}]))
ok(_cj[0]["razonamiento"] == CRIT_93[0]["razonamiento"] and _cj[2]["tocado"] is True
   and _cj[2]["grupo"] == "A" and len(_cj[2]["razonamiento"]) == 8000,
   "la ficha guarda el criterios_json completo (razón, grupo, tocado), recortado por texto")
ok(_ns["_criterios_json_para_ficha"]("no es json") == "no es json"
   and _ns["_criterios_json_para_ficha"]("") == [], "y si no es JSON lo guarda como llegó")
_fg = next(n for n in ast.walk(ARBOL_MAIN) if isinstance(n, ast.FunctionDef)
           and n.name == "_taller_guardar_proyecto")
_src_fg = ast.get_source_segment(SRC_MAIN, _fg)
ok('"criterios_json": _criterios_json_para_ficha(criterios_json)' in _src_fg
   and "**_taller_meta_listo(res)" in _src_fg and '"razonamiento"' in _src_fg and '"grupo"' in _src_fg,
   "`_taller_guardar_proyecto` escribe todo eso en la ficha")
ok("meta_estudio: dict = field(default_factory=dict)" in SRC_RA
   and "avisos=_limpios, meta_estudio=_meta_f)" in SRC_RA,
   "`_terminar` devuelve lo anotado en el Resultado")

print()
if FALLOS:
    print(f"FALLAN {len(FALLOS)}: " + " · ".join(FALLOS))
    raise SystemExit(1)
print("RESULTADO: TODAS LAS COMPROBACIONES PASAN")
