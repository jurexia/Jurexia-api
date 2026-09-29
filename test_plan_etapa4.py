# -*- coding: utf-8 -*-
"""LA ETAPA 4 EN EL PLAN: el alias, lo que el código cambia y la relación sin
clasificar (rediseño, piezas 2, 3 y 7 del cruce; 29-sep-2026) — sin modelo.

POR QUÉ. Tres cosas del plan se hacían en silencio y la etapa 4 las hace ver,
todas detrás de la bandera `plan_con_analisis` (sólo cuentas de prueba):
  · `cae_con_principal` nombraba en el plan una SUSTRACCIÓN («queda sin
    materia») y en el árbol una CAÍDA («inoperante»). Con la bandera se emite
    `sin_materia_por_principal`; al leer valen los dos nombres, con y sin ella,
    porque los planes plan-6 guardados traen el viejo y `RAZONES[rz]` lanzaría
    KeyError.
  · `reparar`, `_resolver_en_sitio` y `_reparar_organizacion` le cambian al
    planificador calificaciones, razones, tratamientos y unidades, y sólo lo
    decían en un aviso de texto. Ahora queda en `cambios_sin_justificar`,
    calculado como DIFERENCIA contra lo que devolvió el planificador (guardado
    al normalizar), con la regla de cada cambio. Sólo registra.
  · la relación que el planificador no clasificó pasaba a «necesaria» sin
    decirlo; ahora es «sin_clasificar»: decide como necesaria y se ve.

QUÉ COMPRUEBA
  1 · la paridad con la bandera apagada: la red de test_paridad_plan.py sigue
      idéntica byte a byte; sin la bandera no sale ni un rastro de la etapa 4;
      encenderla y apagarla en el mismo proceso no deja nada detrás (gunicorn
      -w 2: nada vive entre peticiones); la clave sólo cambia con ella;
  2 · el alias con planes viejos y con planes nuevos leídos sin la bandera;
  3 · el registro de cambios: el portador, la suficiencia, la lectura de la
      razón escrita como calificación, la organización; cada regla con su
      entrada exacta (regla, campo, cuenta), también la que sólo alcanza
      `resolver_por_dependencia` sola y el registro que ésta rehace; idéntico tras tres
      pasadas, tras tres `reparar` y tras pasar por la fila (jsonb); sin una
      sola decisión distinta de la que sale sin la bandera;
  4 · «sin_clasificar»: decide como «necesaria» y se ve en el aviso, en el
      guion, en el bloque y en la ficha; el «toral» y el «fondo» por omisión
      no cambian (decisión abierta).

LOS ASUNTOS son los de test_plan_estudio.py (el amparo directo sintético) y
test_regresion_631.py (el 631 y el PLAN6), COPIADOS tal cual de la red de
paridad, que los copió de ahí: aquellas pruebas corren al importarse. Todo es
SINTÉTICO.

    .venv/bin/python test_plan_etapa4.py
"""
import copy
import json
import os
import subprocess
import sys

import contexto_taller as ct

# LAS BANDERAS LAS PONE LA PRUEBA, NO EL ENTORNO: una PLAN_CON_ANALISIS=todos
# encendería la bandera en las comprobaciones que la quieren apagada.
for _v in list(ct.BANDERAS_REDISENO.values()) + [
        "ESFUERZO_PLAN", "PLAN_TOPE_SALIDA", "PLAN_TOPE_CORRIDAS", "PLAN_ESPERA_S", "MODELO_PLAN",
        "PREGUNTA_DECISIVA_ACTIVA", "CNPCF_VIGENTE"]:
    os.environ.pop(_v, None)
ct.poner(False, {})

import fase6_estudio as f6
import fases123_pipeline as f123
import plan_estudio as pe

FALLOS = []


def ok(cond, que):
    print(f"   {'PASA ' if cond else 'FALLA'}  {que}")
    if not cond:
        FALLOS.append(que)


AQUI = os.path.dirname(os.path.abspath(__file__))


def encender():
    """Una cuenta de prueba con SÓLO esta bandera, como la pide una evaluación
    de casa: las demás del rediseño se apagan para medir ésta sola (la fuerza
    unificada, por ejemplo, cambia el índice que ve el planificador)."""
    ct.poner(True, {"banderas": {b: b == "plan_con_analisis" for b in ct.BANDERAS_REDISENO}}, pruebas=True)


def apagar():
    """Una cuenta de fuera: ninguna bandera del rediseño."""
    ct.poner(False, {})


# ═══ LOS ASUNTOS (copiados de test_paridad_plan.py, que los copió de ═════════
# test_plan_estudio.py y test_regresion_631.py) ══════════════════════════════

# De test_plan_estudio.py, líneas 60-198 (en 4c8f610), tal cual:

# ═══ EL ASUNTO DE PRUEBA (sintético) ════════════════════════════════════════
ACTO = ("La Sala responsable consideró que la identidad del inmueble quedó acreditada con la "
        "confesión del demandado de poseer dentro de la parcela de la actora, por lo que no era "
        "necesaria la prueba pericial en materia de topografía, conforme al artículo 84 del "
        "Código de Procedimientos Civiles para el Estado. Asimismo, estimó que no existe cosa "
        "juzgada, pues así lo ordenó la ejecutoria del amparo directo 312/2023, que quedó firme. "
        "Por último, condenó en costas en ambas instancias.")
ESCRITO = (
    "CONCEPTOS DE VIOLACIÓN\n"
    "PRIMERO. La responsable violó en mi perjuicio el artículo 16 constitucional porque tuvo por "
    "acreditada la identidad del inmueble sin que se desahogara la prueba pericial en materia de "
    "topografía, que era la idónea para ello. Además, en la demanda se reclamó una superficie de "
    "2.5 hectáreas y la condena se refiere a una fracción de 4-80-54.113 hectáreas, lo que "
    "demuestra que no hay identidad entre lo reclamado y lo condenado.\n\n"
    "SEGUNDO. La Sala omitió examinar de oficio la cosa juzgada refleja que deriva de lo resuelto "
    "en el juicio 1114/2017, en el que se discutió la misma parcela entre las mismas partes.\n\n"
    "TERCERO. Insisto en que sin la pericial en topografía no puede tenerse por identificado el "
    "inmueble. También solicito que se aplique por analogía el criterio del juicio 1114/2017 y la "
    "tesis de registro 196451, que obligan a respetar lo resuelto en aquel asunto.")
_o = [ESCRITO.index(x) for x in ("PRIMERO.", "SEGUNDO.", "TERCERO.")] + [len(ESCRITO)]
CONTEO = {"estado": "contado", "n": 3, "tramos": [[_o[i], _o[i + 1]] for i in range(3)]}
PREG1 = "¿La identidad del inmueble quedó acreditada sin la prueba pericial en topografía?"
PREG2 = "¿La Sala debía examinar la cosa juzgada refleja del juicio 1114/2017?"
RAZON1 = ("La identidad quedó acreditada con la confesión del demandado de poseer dentro de la "
          "parcela de la actora; la diferencia de superficies mide la extensión de la condena y no "
          "la identidad del bien. Apoya el criterio de registro 2001111.")
RAZON2 = ("La sentencia reclamada se dictó en cumplimiento de la ejecutoria del amparo directo "
          "312/2023, que ordenó reexaminar los agravios partiendo de que no se actualizaba la cosa "
          "juzgada, y esa determinación quedó firme.")


def fases(**kw):
    f = f123.Fases123(resumen_acto="La Sala tuvo por acreditada la identidad y negó la cosa juzgada.",
                      resumen_conceptos="Primero: identidad sin pericial.\nSegundo: cosa juzgada "
                                        "refleja.\nTercero: reitera y pide analogía.",
                      problemas=[{"pregunta": PREG1, "cubre": [1, 3], "clase": "fondo",
                                  "jerarquia": "principal"},
                                 {"pregunta": PREG2, "cubre": [2], "clase": "fondo",
                                  "jerarquia": "accesorio", "depende_de": None}],
                      fuentes=[ACTO, ESCRITO], conteo=dict(CONTEO))
    for k, v in kw.items():
        setattr(f, k, v)
    return f


def crit(s1="infundado", s2="inoperante", g1="", g2="", r1=RAZON1, r2=RAZON2):
    return [f6.Criterio(problema=PREG1, sentido=s1, razonamiento=r1, jerarquia="principal", grupo=g1),
            f6.Criterio(problema=PREG2, sentido=s2, razonamiento=r2, jerarquia="accesorio", grupo=g2)]


def material(**kw):
    m = f6.Material(
        tesis=[{"registro": "2001111", "rubro": "IDENTIDAD DEL INMUEBLE. PUEDE ACREDITARSE CON LA "
                "CONFESIÓN DEL DEMANDADO.", "obligatoria": True,
                "texto": "La identidad del bien reclamado puede tenerse por demostrada con la "
                         "confesión del demandado de poseer el inmueble materia del juicio, sin que "
                         "sea indispensable la prueba pericial."},
               {"registro": "2002222", "rubro": "COSA JUZGADA. LO DECIDIDO EN UNA EJECUTORIA DE "
                "AMPARO VINCULA AL TRIBUNAL.", "obligatoria": True,
                "texto": "Lo decidido en una ejecutoria de amparo que quedó firme no puede "
                         "discutirse de nuevo en otro juicio de amparo."}],
        normas=[{"cuerpo_legal": "Código de Procedimientos Civiles para el Estado", "articulo": "84",
                 "texto": "El actor debe probar los hechos constitutivos de su acción."}],
        tipo_asunto="amparo_directo", materia="civil", variante="v4")
    for k, v in kw.items():
        setattr(m, k, v)
    return m


SEGS = [
    {"id": "C1.a", "concepto": 1, "parrafo": 0, "texto": "Sin pericial no hay identidad.",
     "cita": "tuvo por acreditada la identidad del inmueble sin que se desahogara la prueba pericial",
     "anclas": ["art 16 cpeum"]},
    {"id": "C1.b", "concepto": 1, "parrafo": 0, "texto": "Las superficies no coinciden.",
     "cita": "en la demanda se reclamó una superficie de 2.5 hectáreas y la condena se refiere a "
             "una fracción de 4-80-54.113 hectáreas",
     "anclas": ["cifra 2.5 ha", "cifra 4-80-54.113 ha"]},
    {"id": "C2.a", "concepto": 2, "parrafo": 1, "texto": "Cosa juzgada refleja.",
     "cita": "La Sala omitió examinar de oficio la cosa juzgada refleja que deriva de lo resuelto "
             "en el juicio 1114/2017",
     "anclas": ["exp 1114/2017"]},
    {"id": "C3.a", "concepto": 3, "parrafo": 2, "texto": "Reitera la falta de pericial.",
     "cita": "Insisto en que sin la pericial en topografía no puede tenerse por identificado el inmueble",
     "anclas": []},
    {"id": "C3.b", "concepto": 3, "parrafo": 2, "texto": "Pide aplicar por analogía el 1114/2017.",
     "cita": "solicito que se aplique por analogía el criterio del juicio 1114/2017 y la tesis de "
             "registro 196451",
     "anclas": ["exp 1114/2017", "reg 196451"]},
]
SEGS_N = [pe.normalizar_segmento_piso(s) for s in SEGS]

PLAN_BUENO = {
    "proposiciones": [
        {"id": "P1", "dice": "La identidad quedó acreditada con la confesión del demandado",
         "caracter": "toral", "relacion": "suficiente", "fuente": "reclamada",
         "cita": "la identidad del inmueble quedó acreditada con la confesión del demandado de "
                 "poseer dentro de la parcela de la actora", "vinculada_por_ejecutoria": False},
        {"id": "P2", "dice": "No hay cosa juzgada: así lo ordenó la ejecutoria del amparo previo",
         "caracter": "toral", "relacion": "necesaria", "fuente": "reclamada",
         "cita": "no existe cosa juzgada, pues así lo ordenó la ejecutoria del amparo directo 312/2023",
         "vinculada_por_ejecutoria": True}],
    "segmentos": [
        {"id": "C1.a", "problema_id": 1, "vicio": "fondo", "ataca": "P1", "reitera": None,
         "dato": {"texto": "confesión de poseer dentro de la parcela",
                  "cita": "confesión del demandado de poseer dentro de la parcela de la actora",
                  "fuente": "reclamada"},
         "etiqueta": "infundado", "razon": "fondo_desestimado", "trat": "aplica",
         "diferencia": None, "pendiente": None, "sostiene": "sin pericial no hay identidad"},
        {"id": "C1.b", "problema_id": 1, "vicio": "fondo", "ataca": "P1", "reitera": None,
         "dato": {"texto": "las dos superficies",
                  "cita": "una superficie de 2.5 hectáreas y la condena se refiere a una fracción "
                          "de 4-80-54.113 hectáreas", "fuente": "escrito"},
         "etiqueta": "infundado", "razon": "fondo_desestimado", "trat": "aplica",
         "diferencia": None, "pendiente": None},
        {"id": "C2.a", "problema_id": 2, "vicio": "fondo", "ataca": "P2", "reitera": None,
         "dato": None, "etiqueta": "inoperante", "razon": "cosa_juzgada_amparo_previo(P2)",
         "trat": "aplica", "diferencia": None, "pendiente": None},
        {"id": "C3.a", "problema_id": 1, "vicio": "fondo", "ataca": "P1", "reitera": "C1.a",
         "dato": None, "etiqueta": "infundado", "razon": "fondo_desestimado", "trat": "remite",
         "diferencia": None, "pendiente": None},
        {"id": "C3.b", "problema_id": 1, "vicio": "fondo", "ataca": "P1", "reitera": None,
         "dato": None, "etiqueta": "infundado", "razon": "fondo_desestimado", "trat": "desarrolla",
         # plan-6: sin «pendiente: razon» (la Decisión 6 se retiró; V0 lo rechaza).
         "diferencia": "precedente", "pendiente": None}],
    "premisas": [
        {"id": "M1", "responde_a": ["P1"], "fuentes": {"tesis": ["T1"], "normas": []},
         "anclas": ["confesión", "dentro de la parcela", "extensión, no identidad"],
         "rastro": "razon",
         "rastro_cita": "confesión del demandado de poseer dentro de la parcela de la actora"},
        {"id": "M2", "responde_a": ["P2"], "fuentes": {"tesis": ["T2"], "normas": []},
         "anclas": ["amparo directo 312/2023", "firme"], "rastro": "razon",
         "rastro_cita": "se dictó en cumplimiento de la ejecutoria del amparo directo 312/2023"}],
    "unidades": [
        {"id": "U1", "problemas": [1], "segmentos": ["C1.a", "C1.b", "C3.a", "C3.b"],
         "premisa": "M1", "objecion": {"de": "la parte quejosa", "anclas": ["pericial en topografía"]}},
        {"id": "U2", "problemas": [2], "segmentos": ["C2.a"], "premisa": "M2", "objecion": None}],
    "propuestas": [], "avisos_al_secretario": [],
    "orden": {"criterio": "promovente", "por_que": "el orden del escrito ya es el lógico"},
}


def mutar(fn):
    d = copy.deepcopy(PLAN_BUENO)
    fn(d)
    return d


def seg(d, sid):
    return next(s for s in d["segmentos"] if s["id"] == sid)


# test_plan_estudio.py, líneas 563-569: el plan malo del reintento.
# EL PLAN MALO LLEVA UN DEFECTO DE CONTENIDO (un argumento del escrito que el
# planificador dejó fuera: V0 a) y, además, dos que el código ya corrige solo
# (reitera con ancla propia y premisa sin rastro, 26-sep-2026). Sólo el
# primero justifica pedir otro plan.
_malo = mutar(lambda d: (seg(d, "C3.b").update(reitera="C2.a", trat="remite", pendiente=None),
                         d["segmentos"].remove(seg(d, "C3.a"))))
_malo["premisas"][0]["rastro_cita"] = "texto que no está en ninguna parte y no se puede verificar"


# De test_regresion_631.py, líneas 56-178, tal cual salvo el sufijo _631 donde el
# nombre chocaba con el del amparo directo:

# ═══ EL ASUNTO DE PRUEBA ═══════════════════════════════════════════════════
ACTO_631 = ("El Juzgado de Distrito consideró que la sustitución de la parte ejecutante alteró la cosa "
        "juzgada, porque la adquirente del inmueble no era causahabiente del contrato de "
        "arrendamiento que sustentó la acción personal de rescisión, y concedió el amparo. "
        "Consideró innecesario estudiar los restantes conceptos de violación.")
ESCRITO_631 = (
    "AGRAVIOS\n"
    "PRIMERO. La sentencia recurrida vulnera los principios de exhaustividad y congruencia, porque "
    "la interlocutoria sólo determinó quién cuenta con legitimación para continuar la ejecución sin "
    "modificar las prestaciones de la sentencia definitiva. Además, la sentencia de amparo omite "
    "ponderar que el artículo 49 del Código Procesal Civil local permite que el causahabiente "
    "sustituya a quien transmitió el derecho controvertido. También sostiene que se vulnera el "
    "principio de igualdad jurídica porque no se aplican de manera uniforme las normas procesales "
    "a las dos partes. Asimismo afirma que la sentencia carece de fundamentación y motivación "
    "porque no aplica los artículos 2284 y 2294 del Código Civil local al caso concreto. Finalmente "
    "invoca el principio de privilegio del fondo sobre la forma para sostener que la interlocutoria "
    "atiende al fondo de la controversia incidental.\n\n"
    "SEGUNDO. Solicita que, como consecuencia de los argumentos expuestos en el primero, se tenga "
    "por reproducido su contenido para controvertir el considerando octavo de la sentencia recurrida.")
_o_631 = [ESCRITO_631.index(x) for x in ("PRIMERO.", "SEGUNDO.")] + [len(ESCRITO_631)]
CONTEO_631 = {"estado": "contado", "n": 2, "tramos": [[_o_631[i], _o_631[i + 1]] for i in range(2)]}
PREG1_631 = "¿La sustitución de la parte ejecutante alteró la cosa juzgada?"
PREG2_631 = "¿La sentencia recurrida fue exhaustiva y debidamente fundada al resolver sobre la sustitución?"
RAZON1_631 = ("La interlocutoria sólo determinó quién cuenta con legitimación para continuar la "
          "ejecución, sin modificar las prestaciones de la sentencia definitiva; el artículo 49 del "
          "Código Procesal Civil local permite al causahabiente sustituir a quien transmitió el "
          "derecho controvertido.")
RAZON2_631 = "Dado el sentido del estudio del problema principal, queda sin materia el análisis de este planteamiento."


def fases_631(**kw):
    f = f123.Fases123(
        resumen_acto=ACTO_631,
        resumen_conceptos="Primero: la interlocutoria no alteró la cosa juzgada.\nSegundo: reproduce el primero.",
        problemas=[{"pregunta": PREG1_631, "cubre": [1], "clase": "fondo", "jerarquia": "principal",
                    "combate": "La recurrente sostiene que la interlocutoria sólo determinó quién "
                               "ejecuta, sin modificar las prestaciones.",
                    "resolvio": "El Juzgado de Distrito determinó que la sustitución alteró la cosa "
                                "juzgada y concedió el amparo."},
                   {"pregunta": PREG2_631, "cubre": [2], "clase": "fondo", "jerarquia": "accesorio",
                    "depende_de": 1}],
        fuentes=[ACTO_631, ESCRITO_631], conteo=dict(CONTEO_631))
    for k, v in kw.items():
        setattr(f, k, v)
    return f


def crit_631(s1="fundado", s2="innecesario"):
    return [f6.Criterio(problema=PREG1_631, sentido=s1, razonamiento=RAZON1_631, jerarquia="principal"),
            f6.Criterio(problema=PREG2_631, sentido=s2, razonamiento=RAZON2_631, jerarquia="accesorio")]


def material_631():
    return f6.Material(tesis=[], normas=[
        {"cuerpo_legal": "Código Procesal Civil local", "articulo": "49",
         "texto": "El causahabiente puede sustituir a quien transmitió el derecho controvertido."}],
        tipo_asunto="amparo_revision", materia="civil", variante="v4")


SEGS_631 = [pe.normalizar_segmento_piso(s) for s in [
    {"id": "A1.a", "concepto": 1, "texto": "La interlocutoria sólo determinó quién ejecuta.",
     "cita": "la interlocutoria sólo determinó quién cuenta con legitimación para continuar la ejecución",
     "anclas": []},
    {"id": "A1.b", "concepto": 1, "texto": "El artículo 49 permite la sustitución del causahabiente.",
     "cita": "el artículo 49 del Código Procesal Civil local permite que el causahabiente sustituya",
     "anclas": ["art 49 cpc"]},
    {"id": "A1.c", "concepto": 1, "texto": "Igualdad jurídica.",
     "cita": "se vulnera el principio de igualdad jurídica porque no se aplican de manera uniforme",
     "anclas": []},
    {"id": "A1.d", "concepto": 1, "texto": "Falta de fundamentación: artículos 2284 y 2294.",
     "cita": "no aplica los artículos 2284 y 2294 del Código Civil local al caso concreto",
     "anclas": ["art 2284 cc", "art 2294 cc"]},
    {"id": "A1.e", "concepto": 1, "texto": "Privilegio del fondo sobre la forma.",
     "cita": "invoca el principio de privilegio del fondo sobre la forma para sostener",
     "anclas": []},
    {"id": "A2.a", "concepto": 2, "texto": "Reproduce el primero contra el considerando octavo.",
     "cita": "se tenga por reproducido su contenido para controvertir el considerando octavo",
     "anclas": []},
]]


def _s_631(sid, **kw):
    base = {"id": sid, "problema_id": 1, "vicio": "fondo", "ataca": "P1", "reitera": None,
            "dato": None, "etiqueta": "fondo_desestimado", "razon": "fondo_desestimado",
            "trat": "desarrolla", "diferencia": "norma", "pendiente": None}
    base.update(kw)
    return base


PLAN_631 = {
    "proposiciones": [
        {"id": "P1", "dice": "La sustitución alteró la cosa juzgada", "caracter": "toral",
         "relacion": "necesaria", "fuente": "recurrida",
         "cita": "la sustitución de la parte ejecutante alteró la cosa juzgada",
         "vinculada_por_ejecutoria": False}],
    "segmentos": [
        # Lo que devolvió el planificador en el 631: UNO fundado, y los demás
        # con la RAZÓN «fondo_desestimado» escrita en el campo de la etiqueta.
        _s_631("A1.a", etiqueta="fundado", razon="fundado", trat="aplica", diferencia=None),
        _s_631("A1.b"), _s_631("A1.c", vicio="fondo"), _s_631("A1.d", vicio="forma"), _s_631("A1.e"),
        _s_631("A2.a", problema_id=2, etiqueta="innecesario", razon="cae_con_principal",
           trat="no_se_estudia", diferencia=None, ataca=None)],
    "premisas": [],
    "unidades": [{"id": "U1", "problemas": [1], "segmentos": ["A1.a"], "premisa": None, "objecion": None},
                 {"id": "U2", "problemas": [1], "segmentos": ["A1.b", "A1.c", "A1.e"], "premisa": None,
                  "objecion": None},
                 {"id": "U3", "problemas": [1], "segmentos": ["A1.d"], "premisa": None, "objecion": None}],
    "propuestas": [], "avisos_al_secretario": [],
    "orden": {"criterio": "promovente", "por_que": "el orden del escrito"},
}


# LA TRADUCCIÓN SOLA (577c700, plan-5). Desde plan-6 lo que en un problema que
# prospera no lo funda es innecesario por suficiencia —la jerarquía lo
# resuelve después de traducir—; para ver la traducción sin ella, cada uno
# alega aquí una omisión, que nunca depende del principal. (Hasta la revisión
# adversarial del 28-sep-2026 pedían una «consecuencia distinta»; pero un
# infundado que la pide no da nada más y ahora es innecesario igual.)
PLAN_631_C = copy.deepcopy(PLAN_631)
for _x in PLAN_631_C["segmentos"]:
    if _x["id"] in ("A1.b", "A1.c", "A1.d", "A1.e"):
        _x["vicio"] = "omision"


# test_regresion_631.py, líneas 458-567: el PLAN6 (fundado; decide uno).
ESCRITO6 = (
    "AGRAVIOS\n"
    "PRIMERO. La sentencia recurrida vulnera los principios de exhaustividad y congruencia porque no "
    "atendió lo que se planteó sobre la ejecución de la sentencia definitiva. La interlocutoria sólo "
    "determinó quién cuenta con legitimación para continuar la ejecución sin modificar las prestaciones "
    "de la sentencia definitiva. El artículo 49 del Código Procesal Civil local permite que el "
    "causahabiente sustituya a quien transmitió el derecho controvertido. Los artículos 2284 y 2294 del "
    "Código Civil local transmiten al adquirente los derechos del arrendador sobre el inmueble "
    "arrendado. La recurrente adquirió el inmueble mediante escritura pública de fecha anterior a la "
    "sustitución que se reclama. La sentencia carece de fundamentación porque no cita el precepto que "
    "sustenta la cosa juzgada. La sentencia carece de motivación porque no expone las razones "
    "particulares del caso concreto. Se vulnera el principio de igualdad jurídica porque no se aplican "
    "de manera uniforme las normas procesales. La falta de fundamentación y motivación es una violación "
    "formal distinta de la indebida fundamentación y motivación. Invoca la jurisprudencia sobre los "
    "argumentos que deben examinarse cuando se alega la ausencia de fundamentación. REVISIÓN ADHESIVA. "
    "La quejosa adherente sostiene que el juicio de origen se tramitó con violaciones al procedimiento "
    "que la dejaron sin defensa.\n\n"
    "SEGUNDO. Solicita que, como consecuencia de los argumentos expuestos en el primero, se tenga por "
    "reproducido su contenido para controvertir el considerando octavo de la sentencia recurrida.")
_o6 = [ESCRITO6.index(x) for x in ("PRIMERO.", "SEGUNDO.")] + [len(ESCRITO6)]
CONTEO6 = {"estado": "contado", "n": 2, "tramos": [[_o6[i], _o6[i + 1]] for i in range(2)]}


def fases6():
    return fases_631(fuentes=[ACTO_631, ESCRITO6], conteo=dict(CONTEO6))


_CITAS6 = {
    "A1.a": ("forma", "vulnera los principios de exhaustividad y congruencia porque no atendió", []),
    "A1.b": ("fondo", "sólo determinó quién cuenta con legitimación para continuar la ejecución", []),
    "A1.c": ("fondo", "el artículo 49 del Código Procesal Civil local permite que el causahabiente",
             ["art 49 cpc"]),
    "A1.d": ("fondo", "Los artículos 2284 y 2294 del Código Civil local transmiten al adquirente",
             ["art 2284 cc", "art 2294 cc"]),
    "A1.e": ("fondo", "La recurrente adquirió el inmueble mediante escritura pública de fecha anterior", []),
    "A1.f": ("forma", "La sentencia carece de fundamentación porque no cita el precepto", []),
    "A1.g": ("forma", "La sentencia carece de motivación porque no expone las razones particulares", []),
    "A1.h": ("forma", "Se vulnera el principio de igualdad jurídica porque no se aplican", []),
    "A1.i": ("forma", "La falta de fundamentación y motivación es una violación formal distinta", []),
    "A1.j": ("forma", "Invoca la jurisprudencia sobre los argumentos que deben examinarse", []),
    "AD1.a": ("procesal", "el juicio de origen se tramitó con violaciones al procedimiento que la dejaron", []),
}
SEGS6 = [pe.normalizar_segmento_piso({"id": k, "concepto": 1, "texto": v[1], "cita": v[1], "anclas": v[2]})
         for k, v in _CITAS6.items()] + [SEGS_631[-1]]           # A2.a, el del segundo agravio
_FORMA6 = ("A1.a", "A1.f", "A1.g", "A1.h", "A1.i", "A1.j")


def _s6(sid, **kw):
    base = {"id": sid, "problema_id": 1, "vicio": _CITAS6[sid][0], "ataca": "P1", "reitera": None,
            "dato": None, "etiqueta": "fundado", "razon": "fundado", "trat": "aplica", "diferencia": None,
            "pendiente": None}
    base.update(kw)
    return base


PLAN6 = {
    "proposiciones": [
        {"id": "P1", "dice": "La sustitución alteró la cosa juzgada", "caracter": "toral", "relacion": "suficiente",
         "fuente": "recurrida", "cita": "la sustitución de la parte ejecutante alteró la cosa juzgada",
         "vinculada_por_ejecutoria": False},
        {"id": "P2", "dice": "La adquirente no era causahabiente", "caracter": "accesoria",
         "relacion": "dependiente_de:P1", "fuente": "recurrida",
         "cita": "la adquirente del inmueble no era causahabiente del contrato de arrendamiento",
         "vinculada_por_ejecutoria": False},
        {"id": "P3", "dice": "La acción era personal", "caracter": "accesoria", "relacion": "dependiente_de:P2",
         "fuente": "recurrida", "cita": "que sustentó la acción personal de rescisión",
         "vinculada_por_ejecutoria": False},
        {"id": "P5", "dice": "Era innecesario estudiar los demás conceptos", "caracter": "accesoria",
         "relacion": "dependiente_de:P1", "fuente": "recurrida",
         "cita": "Consideró innecesario estudiar los restantes conceptos de violación",
         "vinculada_por_ejecutoria": False}],
    "segmentos": [
        # Lo que devolvía el planificador del 631 con la Decisión 6: la forma
        # «fundada» y pendiente de una razón del secretario, uno por uno.
        _s6("A1.a", trat="desarrolla", diferencia="norma", pendiente="razon"),
        _s6("A1.b"),
        _s6("A1.c", ataca="P2", dato={"texto": "el artículo 49", "fuente": "escrito",
                                      "cita": "el artículo 49 del Código Procesal Civil local permite que el "
                                              "causahabiente sustituya"}),
        _s6("A1.d", ataca="P2", etiqueta="esencialmente_fundado", razon="esencialmente_fundado"),
        _s6("A1.e", ataca="P3", diferencia="hecho"),
        _s6("A1.f", trat="desarrolla", diferencia="norma", pendiente="razon"),
        _s6("A1.g", trat="desarrolla", diferencia="norma", pendiente="razon"),
        _s6("A1.h", etiqueta="inoperante", razon="generico", trat="residual"),
        _s6("A1.i", trat="desarrolla", diferencia="norma", pendiente="razon"),
        _s6("A1.j", trat="desarrolla", diferencia="precedente", pendiente="razon"),
        # Un suplido y la violación procesal del adhesivo: nunca dependen.
        {"id": "S1.a", "problema_id": 1, "vicio": "fondo", "ataca": "P1", "reitera": None, "dato": None,
         "etiqueta": "fundado", "razon": "fundado", "trat": "desarrolla", "diferencia": "hecho", "pendiente": None},
        _s6("AD1.a", etiqueta="infundado", razon="fondo_desestimado", trat="desarrolla", diferencia="procesal"),
        {"id": "A2.a", "problema_id": 2, "vicio": "fondo", "ataca": "P5", "reitera": None, "dato": None,
         "etiqueta": "innecesario", "razon": "cae_con_principal", "trat": "no_se_estudia", "diferencia": None,
         "pendiente": None}],
    "premisas": [{"id": "M1", "responde_a": ["P1", "P2", "P3"], "fuentes": {"tesis": [], "normas": ["art. 49"]},
                  "anclas": ["causahabiente", "legitimación"], "rastro": "razon",
                  "rastro_cita": "el artículo 49 del Código Procesal Civil local permite al causahabiente "
                                 "sustituir a quien transmitió el derecho controvertido"}],
    # Partido como en el 631: la misma premisa en tres unidades.
    "unidades": [{"id": "U1", "problemas": [1], "segmentos": ["A1.b"], "premisa": "M1",
                  "objecion": {"de": "el juzgado", "anclas": ["cosa juzgada"]}},
                 {"id": "U2", "problemas": [1], "segmentos": ["A1.c", "A1.d"], "premisa": "M1", "objecion": None},
                 {"id": "U3", "problemas": [1], "segmentos": ["A1.e"], "premisa": "M1", "objecion": None},
                 {"id": "U4", "problemas": [1], "segmentos": ["A1.a", "A1.f", "A1.g", "A1.i", "A1.j"],
                  "premisa": None, "objecion": None},
                 {"id": "U5", "problemas": [1], "segmentos": ["S1.a"], "premisa": None, "objecion": None},
                 {"id": "U6", "problemas": [1], "segmentos": ["AD1.a"], "premisa": None, "objecion": None}],
    "propuestas": [],
    "avisos_al_secretario": ["A1.a, A1.f, A1.g, A1.i y A1.j tienen pendiente la razón específica del secretario"],
    "orden": {"criterio": "prelacion", "por_que": "el principal primero"},
}


# test_plan_estudio.py §3: las variantes del problema 2 (procesal; sin materia) y el principal fundado.
_fp = fases(problemas=[{"pregunta": PREG1, "cubre": [1, 3], "clase": "fondo", "jerarquia": "principal"},
                       {"pregunta": PREG2, "cubre": [2], "clase": "procesal", "jerarquia": "accesorio"}])
_cp3 = crit(s1="infundado", s2="innecesario", r2="Queda sin materia.")
_c_f = crit(s1="fundado")


# ═══ LOS CASOS Y EL CAMINO DE MAIN ═════════════════════════════════════════
AD, REV = "amparo_directo", "amparo_revision"
_PORTADOR_631 = copy.deepcopy(PLAN_631_C)
_PORTADOR_631["segmentos"][0].update(etiqueta="fondo_desestimado", razon="fondo_desestimado",
                                     trat="desarrolla", diferencia="norma")

# Cada uno ejercita una regla distinta del código (la que dice el comentario).
CASOS = {
    "ad.bueno": (PLAN_BUENO, crit(), SEGS_N, fases(), material(), AD),              # nada que cambiar
    "ad.fundado": (PLAN_BUENO, _c_f, SEGS_N, fases(), material(), AD),              # portador y suficiencia
    "ad.malo": (_malo, crit(), SEGS_N, fases(), material(), AD),                    # reitera, referencia, premisa
    "ad.premisa_sin_rastro": (mutar(lambda d: d["premisas"][0].update(
        rastro_cita="la confesión ficta basta siempre para identificar cualquier bien")),
        crit(), SEGS_N, fases(), material(), AD),
    "ad.unidad_mezcla_vicios": (mutar(lambda d: seg(d, "C1.b").update(vicio="omision", razon="omision_inexistente")),
                                crit(), SEGS_N, fases(), material(), AD),         # unidad partida
    "ad.segmento_inventado": (mutar(lambda d: d["segmentos"].append(dict(seg(d, "C1.a"), id="C9.z"))),
                              crit(), SEGS_N, fases(), material(), AD),           # fuera del inventario
    "ad.procesal_sin_estudio_del_criterio": (mutar(lambda d: seg(d, "C2.a").update(
        etiqueta="innecesario", razon="cae_con_principal", trat="no_se_estudia", vicio="procesal")),
        _cp3, SEGS_N, _fp, material(), AD),                                        # la razón renombrada, escrita
    "ad.razon_vacia_sin_estudio": (mutar(lambda d: seg(d, "C2.a").update(
        etiqueta="innecesario", razon="", trat="no_se_estudia")),
        _cp3, SEGS_N, fases(), material(), AD),                                    # la razón renombrada, puesta
    "ad.rarezas": (mutar(lambda d: (d["proposiciones"][0].update(relacion="autonoma", caracter="principal"),
                                    d["proposiciones"][1].update(relacion="conjunta", caracter=""),
                                    seg(d, "C3.b").update(vicio="sustantivo"))),
                   crit(), SEGS_N, fases(), material(), AD),                       # lo que normalizar no entiende
    "rev.631": (PLAN_631, crit_631(), SEGS_631, fases_631(), material_631(), REV),
    "rev.631_portador": (_PORTADOR_631, crit_631(), SEGS_631, fases_631(), material_631(), REV),
    "rev.plan6": (PLAN6, crit_631(), SEGS6, fases6(), material_631(), REV),        # grupos de la jerarquía
}


# UNA REGLA POR CASO, CON SU ENTRADA EXACTA EN LA §3 (revisión adversarial del
# 29-sep-2026). Las comprobaciones de conjunto —«ningún sin_regla», «la
# organización no cuenta»— no veían trece mutaciones del registro: una regla
# sin anotar en un camino que ningún caso recorría, una con el nombre de otra,
# la lectura que se ponía a contar, el orden de las unidades que nunca se
# registraba. Cada caso de aquí recorre UNA regla y la §3 exige su entrada
# (regla, campo, cuenta); como van en CASOS, pasan también por la paridad, la
# idempotencia y «sólo registra».
def _p3_dependiente(d):
    """P3 depende de P1, la toral: lo que cae por derivar de P3 cae con P1."""
    d["proposiciones"].append(
        {"id": "P3", "dice": "La confesión basta para tener por identificado el inmueble",
         "caracter": "accesoria", "relacion": "dependiente_de:P1", "fuente": "reclamada", "cita": "",
         "vinculada_por_ejecutoria": False})


_SUPLIDO = {"id": "S1.a", "problema_id": 1, "vicio": "fondo", "ataca": "P1", "reitera": None, "dato": None,
            "etiqueta": "infundado", "razon": "fondo_desestimado", "trat": "desarrolla", "diferencia": "hecho",
            "pendiente": None}
CASOS.update({
    # fundado dentro de un problema infundado → fundado pero insuficiente
    "ad.fundado_en_infundado": (mutar(lambda d: seg(d, "C1.b").update(etiqueta="fundado", razon="fundado")),
                                crit(), SEGS_N, fases(), material(), AD),
    # el problema 2 sin sentido fijado → etiqueta vacía, pendiente del secretario
    "ad.sin_sentido": (PLAN_BUENO, crit(s2=""), SEGS_N, fases(), material(), AD),
    # el planificador lo puso en un problema que no cubre su concepto
    "ad.otro_problema": (mutar(lambda d: seg(d, "C1.b").update(problema_id=2)),
                         crit(), SEGS_N, fases(), material(), AD),
    # un suplido que no prospera → no se expresa (art. 79, penúltimo párrafo)
    "ad.suplido": (mutar(lambda d: (d["segmentos"].append(dict(_SUPLIDO)),
                                    d["unidades"].append({"id": "U3", "problemas": [1], "segmentos": ["S1.a"],
                                                          "premisa": None, "objecion": None}))),
                   crit(), SEGS_N, fases(), material(), AD),
    # su criterio dice que no se estudia y el planificador lo contestaba en su unidad
    "ad.criterio_no_estudia": (mutar(lambda d: seg(d, "C2.a").update(etiqueta="innecesario",
                                                                     razon="cae_con_principal")),
                               _cp3, SEGS_N, fases(), material(), AD),
    # el accesorio antes que su principal, con orden de prelación → el código reordena
    "ad.orden": (mutar(lambda d: (d["unidades"].reverse(), d["orden"].update(criterio="prelacion"))),
                 crit(), SEGS_N, fases(), material(), AD),
    # el inoperante que el planificador declaró derivado de P3 cae con P1, su raíz
    "ad.deriva": (mutar(lambda d: (_p3_dependiente(d), seg(d, "C3.b").update(
        etiqueta="inoperante", razon="deriva_de_desestimado(P3)", ataca="P3", trat="aplica"))),
        crit(), SEGS_N, fases(), material(), AD),
})
_NUEVO, _VIEJO = "sin_materia_por_principal", "cae_con_principal"


def _crudo(d, tipo, suplencia=None):
    """El plan crudo tal como sale de `planear`: normalizado y sellado."""
    n = pe.normalizar(copy.deepcopy(d))
    n.update({"version": pe.PLAN_VERSION, "tipo_asunto": tipo,
              "suplencia": dict(suplencia) if isinstance(suplencia, dict) else {}})
    return n


def _al_estudio(plan, razones=None):
    """El plan que llega al estudio por el camino de main (`_taller_plan_para`)."""
    return pe.resolver_por_dependencia(pe.aplicar_razones(plan, pe.leer_razones_segmento(razones or {})))


def _dump(x) -> str:
    return json.dumps(x, ensure_ascii=False, sort_keys=True)


def _corrida(crudo, c, segs, f, m, tipo) -> dict:
    n = _crudo(crudo, tipo)
    r, av = pe.reparar(n, c, segs, f, m, "", {})
    e = _al_estudio(r)
    g = pe.vista(e, "estandar")
    return {"normalizado": n, "reparado": r, "avisos": av, "v0": pe.validar(r, c, segs, f, m, "", suplencia={}),
            "al_estudio": e, "guion_estandar": g, "guion_moderna": pe.vista(e, "moderna"), "bloque": pe.bloque(g),
            "ficha": pe.para_ficha(r, "usado", "clave-de-prueba", av)}


def _prompt() -> str:
    return pe.prompt_plan(tipo_asunto=AD, probs=pe.problemas_del_criterio(crit(), fases()), segs=SEGS_N,
                          resumen_acto="r", tramos=pe._tramos_del_escrito(fases()),
                          indice=pe.indice_material(material(), fases()), n=3)


def _todo() -> dict:
    F = {k: _corrida(*v) for k, v in CASOS.items()}
    F["prompt"] = _prompt()
    F["clave"] = pe.clave(crit(), "h", "ctx", {}, "estandar")
    return F


def _sin_etapa4(x):
    """El plan sin lo que la bandera añade (el original, las reglas, el
    registro, la relación sin clasificar y su aviso) y con el nombre de
    siempre: si la etapa 4 sólo registra, esto es el plan sin la bandera."""
    x = copy.deepcopy(x)
    if isinstance(x, dict):
        for k in ("_original", "_reglas", "cambios_sin_justificar"):
            x.pop(k, None)
        for p in x.get("proposiciones") or []:
            if p.get("relacion") == "sin_clasificar":
                p["relacion"] = "necesaria"
                p.pop("relacion_crudo", None)
        if "avisos_al_secretario" in x:
            x["avisos_al_secretario"] = [a for a in x["avisos_al_secretario"] if "«sin_clasificar»" not in a]
    return json.loads(_dump(x).replace(_NUEVO, _VIEJO))


def _guion_de_siempre(g: str) -> str:
    return g.replace(_NUEVO, _VIEJO).replace(" · sin_clasificar", " · necesaria")


def _cambios(F, k) -> list:
    return F[k]["reparado"].get("cambios_sin_justificar") or []


def _entrada(F, k, obj, id_, campo):
    return next((x for x in _cambios(F, k) if x.get(obj) == id_ and x.get("campo") == campo), None)


def _guardado(k) -> dict:
    """Un plan como los de producción: reparado SIN la bandera y leído de la
    fila (jsonb). No trae original."""
    apagar()
    crudo, c, segs, f, m, tipo = CASOS[k]
    r, _ = pe.reparar(_crudo(crudo, tipo), c, segs, f, m, "", {})
    return json.loads(json.dumps(r))


# ═══════════════════════════════════════════════════════════════════════════
print("\n0 · LA BANDERA plan_con_analisis")
apagar()
ok(not ct.rediseno("plan_con_analisis") and not pe._con_analisis(), "para una cuenta de fuera, apagada")
ct.poner(True, {}, pruebas=True)
ok(ct.rediseno("plan_con_analisis") and pe._con_analisis(), "para una cuenta de prueba (@iurexia.com), encendida")
encender()
ok(pe._con_analisis() and not any(ct.rediseno(b) for b in ct.BANDERAS_REDISENO if b != "plan_con_analisis"),
   "y aquí se mide sola: las demás banderas del rediseño, apagadas")
apagar()

# ═══════════════════════════════════════════════════════════════════════════
print("\n1 · CON LA BANDERA APAGADA, EL PLAN DE SIEMPRE")
_env = {k: v for k, v in os.environ.items() if k not in ("FOTOGRAFIAR", "PARIDAD_HIJO")}
_red = subprocess.run([sys.executable, os.path.join(AQUI, "test_paridad_plan.py")], cwd=AQUI, env=_env,
                      capture_output=True, text=True, timeout=600)
ok(_red.returncode == 0 and "RESULTADO: TODAS LAS COMPROBACIONES PASAN" in _red.stdout,
   "la red de paridad (test_paridad_plan.py) sigue idéntica byte a byte: prompts, planes, V0, guiones, "
   "claves y huellas"
   + ("" if _red.returncode == 0 else
      f" (salida {_red.returncode}: {[x for x in _red.stdout.splitlines() if 'FALLA' in x][:6]})"))

apagar()
A = _todo()
_RASTROS = ("_original", "_reglas", "cambios_sin_justificar", "sin_clasificar", "relacion_crudo", _NUEVO)
_hay = sorted({r for v in A.values() for r in _RASTROS if r in _dump(v)})
ok(not _hay, f"sin la bandera no sale ni un rastro de la etapa 4 en {len(CASOS)} casos (normalizar, reparar, V0, "
             f"el plan al estudio, los dos guiones, el bloque, la ficha, el prompt)"
             + (f"; salen: {_hay}" if _hay else ""))
encender()
B = _todo()
apagar()
C = _todo()
ok(_dump(A) == _dump(C),
   "encenderla y apagarla en el mismo proceso no deja nada detrás: lo que sale después es lo de antes "
   "(gunicorn -w 2: nada de la bandera vive entre peticiones)")
ok(_dump(A) != _dump(B), "y con ella sí cambia algo (la comprobación no es trivial)")
ok(A["clave"] == C["clave"] != B["clave"],
   "la clave del plan es la de siempre sin la bandera y otra con ella: un plan hecho con la bandera no se "
   "sirve sin ella ni al revés")

# ═══════════════════════════════════════════════════════════════════════════
print("\n2 · EL ALIAS: cae_con_principal → sin_materia_por_principal")
ok(len(pe.RAZONES) == 20 and _VIEJO in pe.RAZONES and _NUEVO not in pe.RAZONES,
   "el catálogo no cambia: la clave vieja se queda (los planes guardados la traen) y la nueva no entra")
for _modo, _nombre, _d in ((apagar, _VIEJO, "sin"), (encender, _NUEVO, "con")):
    _modo()
    ok(pe._razon(_VIEJO) == (_nombre, None) and pe._razon(_NUEVO) == (_nombre, None)
       and pe._razon("Sin materia por principal (p2)") == (_nombre, "P2") and pe._razon("Cae con principal") == (_nombre, None),
       f"{_d} la bandera, al leer valen los dos nombres y sale «{_nombre}»")
    ok(pe.norm_sentido(_NUEVO) == _VIEJO and pe.clase_sentido("Sin materia por principal") == pe.NO_SE_ESTUDIA
       and pe.calificacion_conocida(_NUEVO),
       f"{_d} la bandera, como SENTIDO el nombre nuevo es el de siempre (el que conocen congruencia.py y "
       f"fase5_propuesta.py): el plan nunca lo emite como etiqueta")
    ok(pe._implica(_NUEVO) == pe._implica(_VIEJO) == "innecesario" and pe._falla_razon(_NUEVO, "innecesario") == ""
       and pe._falla_razon(_NUEVO, "fundado") == pe._falla_razon(_VIEJO, "fundado").replace(_VIEJO, _NUEVO) != "",
       f"{_d} la bandera, V0 juzga igual los dos nombres, sin KeyError")
apagar()
encender()
_p1 = _prompt()
apagar()
_p0 = _prompt()
ok(_p1.count(_NUEVO) == 3 and _VIEJO not in _p1,
   "con la bandera el planificador sólo ve el nombre nuevo: en el catálogo, en la regla de las procesales y "
   "en los tratamientos")
ok(_VIEJO in _p0 and _NUEVO not in _p0 and _p1.replace(_NUEVO, _VIEJO) == _p0,
   "sin ella, el de siempre, y el prompt no difiere en nada más")

# LOS PLANES VIEJOS: reparados sin la bandera, guardados en la fila y leídos
# con ella y sin ella (GET /taller/plan, el guion, la ficha).
for _k, _sid in (("ad.procesal_sin_estudio_del_criterio", "C2.a"), ("rev.631", "A2.a")):
    _viejo = _guardado(_k)
    crudo, c, segs, f, m, tipo = CASOS[_k]
    ok(next(s for s in _viejo["segmentos"] if s["id"] == _sid)["razon"] == _VIEJO and "_original" not in _viejo,
       f"{_k}: el plan guardado trae «{_VIEJO}» en {_sid} y no trae original")
    _res = {}
    for _modo, _d in ((apagar, "sin"), (encender, "con")):
        _modo()
        try:
            _e = _al_estudio(_viejo)
            _res[_d] = {"v0": pe.validar(_e, c, segs, f, m, "", suplencia={}), "e": _e,
                        "g": pe.vista(_e, "estandar"), "gm": pe.vista(_e, "moderna"),
                        "ficha": pe.para_ficha(_e, "usado", "k", [])}
            pe.bloque(_res[_d]["g"])
            _err = ""
        except Exception as ex:                                   # pragma: no cover
            _err = f"{type(ex).__name__}: {ex}"
        ok(not _err, f"{_k}, {_d} la bandera: se relee, se valida, se escribe su guion y su ficha sin error"
                     + (f" ({_err})" if _err else ""))
    if len(_res) == 2:
        ok(_res["con"]["v0"] == _res["sin"]["v0"] == [], f"{_k}: V0 lo da por bueno con y sin la bandera")
        _fs = {s["id"]: s for s in _res["con"]["ficha"]["segmentos"]}
        ok(f"razón: {_NUEVO}" in _res["con"]["g"] and _VIEJO not in _res["con"]["g"]
           and f"razón: {_VIEJO}" in _res["sin"]["g"] and _NUEVO not in _res["sin"]["g"]
           and _fs[_sid]["razon"] == _NUEVO,
           f"{_k}: con la bandera el guion y la ficha lo escriben «{_NUEVO}»; sin ella, «{_VIEJO}»")
        ok(next(s for s in _res["con"]["e"]["segmentos"] if s["id"] == _sid)["razon"] == _VIEJO
           and "cambios_sin_justificar" not in _res["con"]["e"],
           f"{_k}: el dato guardado no se reescribe, y a un plan sin original no se le inventa un registro "
           f"(sin lo que dijo el planificador no hay diferencia que calcular)")
        ok(_guion_de_siempre(_res["con"]["g"]) == _res["sin"]["g"]
           and _guion_de_siempre(_res["con"]["gm"]) == _res["sin"]["gm"],
           f"{_k}: el guion, en las dos formas, sólo difiere en el nombre")

# UN PLAN HECHO CON LA BANDERA Y LEÍDO SIN ELLA (una evaluación que la apaga).
encender()
crudo, c, segs, f, m, tipo = CASOS["ad.procesal_sin_estudio_del_criterio"]
_nuevo, _ = pe.reparar(_crudo(crudo, tipo), c, segs, f, m, "", {})
_nuevo = json.loads(json.dumps(_nuevo))
apagar()
ok(seg(_nuevo, "C2.a")["razon"] == _NUEVO and pe.validar(_nuevo, c, segs, f, m, "", suplencia={}) == []
   and f"razón: {_VIEJO}" in pe.vista(_al_estudio(_nuevo), "estandar"),
   "un plan hecho con la bandera trae el nombre nuevo; leído sin ella, V0 lo da por bueno y el guion dice el de siempre")
# DONDE EL CÓDIGO ELIGE LA RAZÓN (`_razon_de_la_etiqueta`, paso 2 de la organización).
ok(seg(B["ad.razon_vacia_sin_estudio"]["reparado"], "C2.a")["razon"] == _NUEVO
   and seg(A["ad.razon_vacia_sin_estudio"]["reparado"], "C2.a")["razon"] == _VIEJO
   and B["ad.razon_vacia_sin_estudio"]["v0"] == A["ad.razon_vacia_sin_estudio"]["v0"] == []
   and any(f"C2.a ({_NUEVO})" in a for a in B["ad.razon_vacia_sin_estudio"]["reparado"]["avisos_al_secretario"]),
   "la razón que pone el propio código a lo que queda sin materia por el principal sale con el nombre que rige, "
   "y así lo dice el aviso")

# ═══════════════════════════════════════════════════════════════════════════
print("\n3 · LO QUE EL CÓDIGO LE CAMBIA AL PLANIFICADOR, REGISTRADO")
ok(all("cambios_sin_justificar" not in A[k]["reparado"] and "_original" not in A[k]["normalizado"]
       and "cambios_sin_justificar" not in A[k]["al_estudio"] for k in CASOS),
   "sin la bandera ningún plan lleva el registro ni el original")
ok(all(isinstance(B[k]["reparado"].get("cambios_sin_justificar"), list)
       and isinstance(B[k]["normalizado"].get("_original"), dict) for k in CASOS),
   "con ella todos lo llevan, y el original se guarda al normalizar")
ok(_cambios(B, "ad.bueno") == [], "el plan bueno: el código no le cambió nada y la lista está vacía")
_o = B["ad.fundado"]["normalizado"]["_original"]
ok(_o["segmentos"]["C1.a"]["etiqueta"] == "infundado" and _o["unidades"]["U1"]["segmentos"] == ["C1.a", "C1.b", "C3.a", "C3.b"]
   and _o["orden"] == {"criterio": "promovente", "unidades": ["U1", "U2"]},
   "el original es lo que dijo el planificador: sus etiquetas, sus unidades y su orden")
ok(_entrada(B, "ad.fundado", "segmento", "C1.a", "etiqueta")
   == {"segmento": "C1.a", "campo": "etiqueta", "antes": "infundado", "despues": "fundado", "regla": "portador",
       "cuenta": True},
   "EL PORTADOR: el problema prospera y el planificador no dejó quien lo funde; el código invirtió la etiqueta "
   "de C1.a y queda registrado, y cuenta")
ok(B["ad.fundado"]["reparado"]["problemas"][0]["sentido"] == "fundado"
   and any("lo funda C1.a (infundado→fundado)" in a for a in B["ad.fundado"]["reparado"]["avisos_al_secretario"]),
   "el sentido del problema no se toca y el aviso de siempre sigue diciéndolo")
_suf = [x for x in _cambios(B, "ad.fundado") if x["regla"] == "suficiencia"]
ok({x["segmento"] for x in _suf} == {"C1.b", "C3.a", "C3.b"} and all(x["cuenta"] == (x["campo"] != "trat") for x in _suf),
   "LA SUFICIENCIA: los que la jerarquía deja innecesarios, registrados; la etiqueta y la razón cuentan, el "
   "tratamiento no")
_gp = [x for x in _cambios(B, "rev.plan6") if x["regla"] == "grupo_con_principal"]
ok(_gp and all(x["segmento"] == "A1.d" and x["antes"] == "esencialmente_fundado" and x["despues"] == "fundado"
               for x in _gp),
   "CON EL PRINCIPAL: el «esencialmente fundado» que pasa a la calificación del grupo, registrado")
_lec = [x for x in _cambios(B, "rev.631_portador") if x["regla"] == "razon_como_etiqueta"]
ok(len(_lec) == 4 and all(not x["cuenta"] and x["antes"] == "fondo_desestimado" and x["despues"] == "infundado"
                          for x in _lec),
   "LA RAZÓN ESCRITA COMO CALIFICACIÓN (el 631): leerla es tomar lo que el planificador sí dijo; se registra y "
   "no cuenta")
_org = [x for k in ("ad.malo", "ad.premisa_sin_rastro", "ad.unidad_mezcla_vicios") for x in _cambios(B, k)]
ok({x["regla"] for x in _org} == {"reitera_con_ancla_propia", "referencia_rota", "premisa_retirada", "partir_unidad"}
   and not any(x["cuenta"] for x in _org),
   "LA ORGANIZACIÓN (reitera con ancla propia, referencia rota, premisa retirada, unidad partida): registrada y "
   "sin contar, para no dejar pendiente casi todo proyecto")
ok(_entrada(B, "ad.segmento_inventado", "segmento", "C9.z", "presente")
   == {"segmento": "C9.z", "campo": "presente", "antes": True, "despues": False, "regla": "fuera_del_inventario",
       "cuenta": False},
   "el segmento que el planificador inventó y el código quitó, también")

# CADA REGLA, SU ENTRADA EXACTA (revisión adversarial del 29-sep-2026): el
# nombre de la regla es lo que el secretario lee para saber POR QUÉ cambió, y
# `cuenta` es lo que deja el proyecto «justificación pendiente». Una regla con el
# nombre de otra, o que cuenta sin deber, pasaba las comprobaciones de conjunto.
_EXACTAS = [
    ("ad.fundado_en_infundado", "segmento", "C1.b", "etiqueta", "fundado", "fundado_insuficiente",
     "fundado_a_insuficiente", True,
     "FUNDADO DENTRO DE UN PROBLEMA QUE NO PROSPERA: el código lo baja a «fundado pero insuficiente», una "
     "calificación que el planificador no dio; se registra con SU regla (no «etiqueta_del_problema») y cuenta"),
    ("ad.sin_sentido", "segmento", "C2.a", "etiqueta", "inoperante", "", "sin_sentido_fijado", False,
     "SIN SENTIDO FIJADO: la etiqueta queda vacía, pendiente del secretario; nada se decidió y no cuenta (si "
     "contara, un criterio a medias dejaría pendiente el proyecto por lo que nadie decidió)"),
    ("ad.otro_problema", "segmento", "C1.b", "problema_id", 2, 1, "problema_del_inventario", False,
     "EL PROBLEMA LO FIJA EL INVENTARIO: el planificador lo puso en uno que no cubre su concepto; se registra "
     "y no cuenta (es organización)"),
    ("ad.suplido", "segmento", "S1.a", "trat", "desarrolla", "no_se_expresa_art79", "suplido_sin_beneficio", False,
     "EL SUPLIDO SIN BENEFICIO no se expresa (art. 79, penúltimo párrafo): se registra con su regla y el "
     "tratamiento no cuenta"),
    ("ad.criterio_no_estudia", "segmento", "C2.a", "trat", "aplica", "no_se_estudia", "criterio_no_se_estudia",
     False,
     "LO QUE EL CRITERIO NO MANDA ESTUDIAR sale de su unidad: se registra con su regla y no cuenta"),
    ("ad.orden", "plan", "orden", "unidades", ["U2", "U1"], ["U1", "U2"], "orden_de_la_ley", False,
     "EL ORDEN DE LA LEY: el accesorio iba antes que su principal y el código reordenó las unidades; el "
     "cambio de orden se registra (no sólo la unidad que se añade) con su regla, y no cuenta"),
    ("ad.deriva", "segmento", "C3.b", "razon_p", "P3", "P1", "deriva", True,
     "CAE POR DERIVAR: la jerarquía le cambia la proposición de su razón (P3 → P1, la raíz toral); es la "
     "regla «deriva», no la suficiencia, y CUENTA: cambia la base de lo que se le contesta"),
    ("ad.deriva", "segmento", "C3.b", "trat", "aplica", "residual", "deriva", False,
     "y su tratamiento pasa a residual con la misma regla, sin contar"),
]
for _k, _obj, _id, _campo, _a, _d, _regla, _cuenta, _por_que in _EXACTAS:
    _x = _entrada(B, _k, _obj, _id, _campo)
    _debe = {_obj: _id, "campo": _campo, "antes": _a, "despues": _d, "regla": _regla, "cuenta": _cuenta}
    ok(_x == _debe, f"{_k}: {_por_que}" + ("" if _x == _debe else f" (sale {_x})"))
_solos = [k for k in ("ad.sin_sentido", "ad.otro_problema", "ad.suplido", "ad.orden") if len(_cambios(B, k)) != 1]
ok(not _solos, "y en los casos de una sola corrección, ésa es la única entrada: nada se registra dos veces "
               "ni se cuela lo que no cambió" + (f"; más de una en {_solos}" if _solos else ""))

# LO QUE LA JERARQUÍA DEVUELVE AL ESTUDIO («autonomo_al_estudio»). Dentro de
# `reparar` no se llega: (b) ya devuelve al estudio lo que la suficiencia no
# cubre antes de la jerarquía. Se llega cuando `resolver_por_dependencia` corre
# sola sobre un plan con su original —main la corre sobre el guardado en cada
# GET y al armar el guion—: aquí, el plan del planificador con los problemas
# del criterio y el adhesivo «innecesario» dentro de un problema fundado (el
# adhesivo nunca depende del principal: `_nunca_depende`).
encender()
_p6 = copy.deepcopy(PLAN6)
seg(_p6, "AD1.a").update(etiqueta="innecesario", razon="cae_con_principal", trat="no_se_estudia")
_n6 = _crudo(_p6, REV)
_n6["problemas"] = copy.deepcopy(B["rev.plan6"]["reparado"]["problemas"])
_j6 = pe.resolver_por_dependencia(_n6)
_x6 = {x["campo"]: x for x in _j6.get("cambios_sin_justificar") or [] if x.get("segmento") == "AD1.a"}
ok(_x6.get("etiqueta") == {"segmento": "AD1.a", "campo": "etiqueta", "antes": "innecesario", "despues": "fundado",
                           "regla": "autonomo_al_estudio", "cuenta": True}
   and _x6.get("trat") == {"segmento": "AD1.a", "campo": "trat", "antes": "no_se_estudia", "despues": "desarrolla",
                           "regla": "autonomo_al_estudio", "cuenta": False},
   "LA JERARQUÍA DEVUELVE AL ESTUDIO lo que la suficiencia no cubre (el adhesivo): resolver_por_dependencia, "
   "sola, lo registra con su regla; la calificación cuenta, el tratamiento no"
   + ("" if "etiqueta" in _x6 else f" (sale {sorted(_x6)} de {len(_j6.get('cambios_sin_justificar') or [])})"))
# Y REHACE EL REGISTRO, NO LO COPIA: un plan guardado cuyo registro quedó viejo
# (otra pasada, otra versión del código) sale del camino de main con el de
# verdad, calculado contra el original.
_rv = copy.deepcopy(B["ad.fundado"]["reparado"])
_rv["cambios_sin_justificar"] = []
ok(_al_estudio(_rv).get("cambios_sin_justificar") == _cambios(B, "ad.fundado") != [],
   "resolver_por_dependencia REHACE el registro contra el original: un plan guardado con uno vacío o viejo "
   "sale con el que corresponde (si lo copiara, main serviría el de otra pasada)")
_reglas = {x["regla"] for k in CASOS for x in _cambios(B, k)}
ok("sin_regla" not in _reglas and _reglas <= set(pe.REGLAS_CAMBIO),
   f"cada cambio lleva una regla con nombre del catálogo (ningún «sin_regla»: no hay camino sin anotar): "
   f"{', '.join(sorted(_reglas))}")
ok(all(set(x) == {next(iter(x)), "campo", "antes", "despues", "regla", "cuenta"}
       and next(iter(x)) in ("segmento", "unidad", "plan") for k in CASOS for x in _cambios(B, k)),
   "la forma: {segmento|unidad|plan, campo, antes, despues, regla, cuenta}")

# IDEMPOTENTE: main relee el plan en cada GET y al armar el guion.
encender()
_mal = []
for _k, (crudo, c, segs, f, m, tipo) in CASOS.items():
    _r = B[_k]["reparado"]
    _cs = _r["cambios_sin_justificar"]
    _e1 = _al_estudio(_r)
    _e2 = pe.resolver_por_dependencia(_e1)
    _e3 = pe.resolver_por_dependencia(_e2)
    _e4 = pe.resolver_por_dependencia(json.loads(json.dumps(_e3, sort_keys=True)))   # jsonb: sin orden de claves
    _r2, _ = pe.reparar(_r, c, segs, f, m, "", {})
    _r3, _ = pe.reparar(_r2, c, segs, f, m, "", {})
    if not (_cs == _e1["cambios_sin_justificar"] == _e2["cambios_sin_justificar"] == _e3["cambios_sin_justificar"]
            == _e4["cambios_sin_justificar"] == _r2["cambios_sin_justificar"] == _r3["cambios_sin_justificar"]
            and _dump(_e1.get("_reglas")) == _dump(_e3.get("_reglas"))):
        _mal.append(_k)
ok(not _mal, f"IDEMPOTENTE en los {len(CASOS)} casos: el mismo registro tras tres pasadas de "
             f"resolver_por_dependencia, tras pasar por la fila y tras tres `reparar`"
             + (f"; cambia en {_mal}" if _mal else ""))

# SÓLO REGISTRA: quitado lo que la bandera añade, el plan es el de siempre.
_dist = [k for k in CASOS
         if _dump(_sin_etapa4(B[k]["reparado"])) != _dump(_sin_etapa4(A[k]["reparado"]))
         or _dump(_sin_etapa4(B[k]["al_estudio"])) != _dump(_sin_etapa4(A[k]["al_estudio"]))
         or B[k]["v0"] != A[k]["v0"] or _sin_etapa4(B[k]["avisos"]) != A[k]["avisos"]]
ok(not _dist, "SÓLO REGISTRA: sin el original, las reglas y el registro, cada plan es el mismo que sin la bandera "
              "(etiquetas, razones, tratamientos, jerarquía, unidades, orden, propuestas, avisos y V0)"
              + (f"; distintos: {_dist}" if _dist else ""))
_gd = [k for k in CASOS if _guion_de_siempre(B[k]["guion_estandar"]) != A[k]["guion_estandar"]
       or _guion_de_siempre(B[k]["guion_moderna"]) != A[k]["guion_moderna"]]
ok(not _gd, "y el guion, en las dos formas, sólo difiere en el nombre de la razón y en la relación sin clasificar"
            + (f"; distintos: {_gd}" if _gd else ""))
_cc = _cambios(B, "ad.fundado")
ok(B["ad.fundado"]["ficha"].get("cambios_sin_justificar")
   == ([c for c in _cc if c.get("cuenta")] + [c for c in _cc if not c.get("cuenta")])[:120]
   and not any("cambios_sin_justificar" in A[k]["ficha"] or "proposiciones" in A[k]["ficha"] for k in CASOS),
   "la ficha del proyecto lo lleva con la bandera (los que cuentan, primero); sin ella, la ficha de siempre")
# REVISIÓN ADVERSARIAL (29-sep): la limpieza de una referencia rota no cuenta, y
# el tope de 120 de la ficha no deja fuera los que cuentan.
ok(pe._cuenta("segmento", "razon_p", "P9", None, "referencia_rota") is False
   and pe._cuenta("segmento", "razon_p", "P2", "P3", "portador") is True,
   "quitar una referencia rota es organización: no cuenta; cambiar la base por otra regla, sí")
_mucho = {"segmentos": [], "cambios_sin_justificar":
          [{"segmento": f"S{i}", "campo": "trat", "cuenta": False} for i in range(130)]
          + [{"segmento": "SX", "campo": "etiqueta", "cuenta": True}]}
encender()
_f130 = pe.para_ficha(_mucho).get("cambios_sin_justificar") or []
ok(len(_f130) == 120 and _f130[0].get("segmento") == "SX",
   "con más de 120, el que cuenta sigue en la ficha (va primero)")

# ═══════════════════════════════════════════════════════════════════════════
print("\n4 · LA RELACIÓN QUE EL PLANIFICADOR NO CLASIFICÓ: «sin_clasificar»")
_RELS = ["autonoma", "conjunta", "", "Necesaria", "suficiente", "dependiente de P1"]


def _relaciones():
    n = pe.normalizar({"proposiciones": [{"id": f"P{i + 1}", "caracter": "toral", "relacion": r}
                                         for i, r in enumerate(_RELS)]})
    return n["proposiciones"]


apagar()
_n0 = _relaciones()
encender()
_n1 = _relaciones()
ok([p["relacion"] for p in _n0] == ["necesaria"] * 4 + ["suficiente", "dependiente_de:P1"]
   and not any("relacion_crudo" in p for p in _n0),
   "sin la bandera lo que no se entiende es «necesaria», como siempre")
ok([p["relacion"] for p in _n1] == ["sin_clasificar"] * 3 + ["necesaria", "suficiente", "dependiente_de:P1"]
   and [p.get("relacion_crudo") for p in _n1] == ["autonoma", "conjunta", "", None, None, None],
   "con ella, «sin_clasificar» con lo que escribió el planificador (también si no escribió nada); lo que se "
   "entiende no cambia")
_nr = pe.normalizar({"proposiciones": [{"id": "P1", "caracter": "principal", "relacion": "suficiente"}],
                     "segmentos": [{"id": "C1.a", "vicio": "sustantivo"}]})
ok(_nr["proposiciones"][0]["caracter"] == "toral" and _nr["segmentos"][0]["vicio"] == "fondo",
   "el carácter «toral» y el vicio «fondo» por omisión NO cambian con la bandera (decisión abierta)")


def _s4(sid, pk):
    return {"id": sid, "ataca": pk, "etiqueta": "infundado", "trat": "aplica", "vicio": "fondo"}


def _decide(r1, r2, orden):
    props = {"P1": {"id": "P1", "caracter": "toral", "relacion": r1},
             "P2": {"id": "P2", "caracter": "toral", "relacion": r2}}
    return pe._quien_decide([_s4("C1.a", "P1"), _s4("C2.a", "P2")], props, False, orden)["id"]


_o12, _o21 = {"C1.a": 0, "C2.a": 1}, {"C1.a": 1, "C2.a": 0}
ok(all(_decide("sin_clasificar", r2, o) == _decide("necesaria", r2, o)
       for r2 in ("necesaria", "suficiente", "sin_clasificar") for o in (_o12, _o21)),
   "QUIÉN DECIDE: «sin_clasificar» pesa exactamente como «necesaria»")
ok(_decide("sin_clasificar", "suficiente", _o12) == "C2.a" and _decide("suficiente", "sin_clasificar", _o21) == "C1.a",
   "y nunca como «suficiente»: la suficiente gana aunque vaya después en el escrito")


def _con_p3(rel):
    return mutar(lambda d: d["proposiciones"].append(
        {"id": "P3", "dice": "Condenó en costas en ambas instancias", "caracter": "toral", "relacion": rel,
         "fuente": "reclamada", "cita": "condenó en costas en ambas instancias", "vinculada_por_ejecutoria": False}))


encender()
_rs, _ = pe.reparar(_crudo(_con_p3("autonoma"), AD), crit(), SEGS_N, fases(), material(), "", {})
_rf, _ = pe.reparar(_crudo(_con_p3("suficiente"), AD), crit(), SEGS_N, fases(), material(), "", {})
_av_sc = [a for a in _rs["avisos_al_secretario"] if "«sin_clasificar»" in a]
ok(len(_av_sc) == 1 and "P3 (escribió «autonoma»)" in _av_sc[0]
   and not any(a.startswith("P3 (") and "por sí sola" in a for a in _rs["avisos_al_secretario"]),
   "REPARAR lo dice en un aviso, con lo que escribió el planificador, y no la trata como suficiente: el aviso "
   "(p) de la suficiente que nadie ataca no sale")
ok(any(a.startswith("P3 (") and "sostiene lo resuelto por sí sola" in a for a in _rf["avisos_al_secretario"]),
   "(una suficiente de verdad sí lo da, como siempre)")
_gs = pe.vista(_al_estudio(_rs), "estandar")
ok("P3 · en síntesis: Condenó en costas en ambas instancias · toral · sin_clasificar" in _gs
   and "«sin_clasificar» (en una proposición)" in pe.bloque(_gs),
   "SE VE en el guion, y el bloque del estudio dice qué significa")
ok(any(p.get("id") == "P3" and p.get("relacion") == "sin_clasificar" and p.get("relacion_crudo") == "autonoma"
       for p in pe.para_ficha(_rs, "usado", "k", []).get("proposiciones") or []),
   "y en la ficha del proyecto")
apagar()
_r0, _ = pe.reparar(_crudo(_con_p3("autonoma"), AD), crit(), SEGS_N, fases(), material(), "", {})
_g0 = pe.vista(_al_estudio(_r0), "estandar")
ok("P3 · en síntesis: Condenó en costas en ambas instancias · toral · necesaria" in _g0
   and "sin_clasificar" not in _g0 + pe.bloque(_g0),
   "sin la bandera, «necesaria» como siempre, y el bloque no cambia")
ok(_dump(_sin_etapa4(_rs)) == _dump(_sin_etapa4(_r0)) and _guion_de_siempre(_gs) == _g0,
   "y con ella no cambia ninguna decisión: el plan y el guion son los de siempre salvo lo que se ve")

apagar()
print()
if FALLOS:
    print(f"FALLAN {len(FALLOS)}: " + " · ".join(FALLOS))
    sys.exit(1)
print("RESULTADO: TODAS LAS COMPROBACIONES PASAN")
