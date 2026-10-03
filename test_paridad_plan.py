# -*- coding: utf-8 -*-
"""LA RED DE PARIDAD DEL PLAN (rediseño, etapa 4, 29-sep-2026) — sin llamar a ningún modelo.

POR QUÉ. La etapa 4 mete el análisis neutral en `plan_estudio.py`: el alias
de `cae_con_principal`, `cambios_sin_justificar`, «sin_clasificar» en vez de
«necesaria», las P del análisis, la exclusión con su prueba. Todo va detrás de
banderas (`plan_con_analisis`, `exclusiones_con_prueba`, `estados_sesion`)
que sólo rigen en las cuentas de prueba. Para los demás secretarios no puede
cambiar NADA: la clave del plan vive en `taller_sesiones.plan`, y un byte de
más en el prompt, en la clave o en el guion hace que cada sesión abierta gaste
otra corrida del planificador o reciba un estudio distinto del que se midió.
Las pruebas que ya hay comprueban PROPIEDADES («decide A1.b», «la forma no
entra en la clave»); ninguna ve un renglón de más en el prompt ni un aviso
reescrito. Ésta fotografía la salida ENTERA con todas las banderas del
rediseño apagadas, ANTES de tocar plan_estudio.py (tal como está desde
24ba772; la instantánea dice sobre qué commit se hizo), y la compara byte a
byte.

CÓMO SE USA
    .venv/bin/python test_paridad_plan.py                 compara con la instantánea
    FOTOGRAFIAR=1 .venv/bin/python test_paridad_plan.py   la rehace y enseña qué cambia
La instantánea es `datos/instantaneas/plan_paridad.json`. Rehacerla es DECIR
que la salida para los de fuera cambia a propósito: sólo con la diferencia a
la vista y explicada en el commit. Un caso legítimo: si lo único que falla es
`catalogos/RAZONES` (una clave nueva que sólo se emite con la bandera) y todo
lo demás sigue igual, el catálogo creció sin que los de fuera vean nada; se
rehace y se dice.

QUÉ FOTOGRAFÍA, con `contexto_taller.poner(False, {})` y sin las variables de
entorno de las banderas ni las del plan (ESFUERZO_PLAN, PLAN_TOPE_SALIDA…):
  · catálogos: PLAN_VERSION, RAZONES y los conjuntos que usan reparar y V0; el
    vocabulario (norm_sentido, clase_sentido, _razon, _implica);
  · prompt_plan, texto completo: amparo directo, revisión y queja con lo
    mínimo; uno con TODOS los bloques opcionales (contexto, conceptos de
    violación, contraste, lista, suplencia confirmada, faltas, decisiva,
    ficha); el del 631 con decisiva y suplencia sin confirmar;
  · por cada plan crudo de las pruebas y las variantes que ellas mismas usan
    (26 casos): V0 sin reparar, reparar (plan y avisos), V0 del reparado, el
    plan AL ESTUDIO por el camino de main (aplicar_razones →
    resolver_por_dependencia) y su guion (vista) estándar y, en los que
    difieren, moderna, con concede y decisiva donde el caso los tiene;
    bloque() donde el guion trae rótulos distintos; para_ficha en tres;
    normalizar en los planes crudos distintos (en una mutación del plan bueno
    el campo cambiado ya se ve en todo lo demás), incluido uno con lo que el
    modelo escribe mal y lo que `normalizar` da por omisión sin decirlo
    («autonoma»/«conjunta» → «necesaria», carácter → «toral», vicio → «fondo»);
  · planes GUARDADOS que se releen sin reparar (el plan-4 del 631, con y sin
    una razón escrita en una pantalla anterior, y el plan-3 del 642):
    resolver_por_dependencia y su guion;
  · V0 sobre las mutaciones que test_plan_estudio.py §3 rechaza: la lista
    LITERAL de faltas, que vuelve al planificador en el reintento (es prompt);
  · preparar con un modelo de mentira: los prompts de cada llamada, sus
    parámetros menos el nombre del modelo, el plan, los avisos y el info;
  · clave() y huella_entradas() en variantes, indice_material, concede_de y
    las funciones de la fila (fila_pedir, fila_latido, fila_resultado,
    estado_para_pantalla) con relojes fijos.
Y comprueba que las banderas están de verdad apagadas, que dos corridas
seguidas dan lo mismo (nada se arrastra entre llamadas) y que otra semilla de
hash (PYTHONHASHSEED) también: si la salida dependiera del orden de un
conjunto, la instantánea fallaría a ratos y aquí se dice por qué.
De paso ve lo que plan_estudio lee de otros módulos (tipos_asunto,
fuerza_juridica en el índice, pregunta_decisiva, ficha_procesal, y
redactor_adelanto/fase_rama por concede_de): si cambia uno de ésos, cambia lo
que reciben los de fuera y también falla; el grupo que falla dice cuál
(`concede_de/…`, `indice/…`, o sólo los prompts).

QUÉ NO CUBRE
  · nada con las banderas ENCENDIDAS: eso lo prueba la prueba de cada pieza;
  · main.py no se ejecuta (_taller_plan_entradas, _taller_plan_para, GET
    /taller/plan): qué argumentos pasa lo fijan las pruebas por AST
    (test_plan_estudio §11, test_integracion_p2);
  · el nombre del modelo del planificador (lo fija test_plan_estudio §5) ni lo
    que contestaría: los planes crudos son los de las pruebas;
  · segmentos_de (el inventario) ni el contraste: entran como dato hecho;
  · el prompt del ESTUDIO (fase6_estudio v4): aquí sólo el bloque del guion;
  · la ficha procesal real: _ficha_de corre sobre un encargo de mentira;
  · la calidad (bancos Kingston y banco_estudio): esto sólo dice que nada
    cambió para los de fuera;
  · el orden de las claves de un dict: se comparan ordenadas, porque la fila
    (jsonb) no lo guarda. El orden de las listas y cada carácter de los
    textos, sí.

LOS ASUNTOS son los de test_plan_estudio.py (el sintético del amparo directo)
y test_regresion_631.py (el AR 631/2025 sintético, el PLAN6 y el plan-4
guardado), COPIADOS aquí tal cual —con el sufijo _631 donde los nombres
chocaban—, y el plan del 642 de datos/congruencia. Copiados y no importados:
aquellas pruebas corren al importarse, y si alguien afina sus datos esta red no
debe romperse por eso. Todo es SINTÉTICO: nombres, números y fechas de prueba.

    .venv/bin/python test_paridad_plan.py
"""
import asyncio
import copy
import difflib
import json
import os
import subprocess
import sys
import types

import contexto_taller as ct

# LAS BANDERAS, APAGADAS TAMBIÉN DESDE EL ENTORNO. `poner(False, {})` apaga
# las «casa», pero una variable como PLAN_CON_ANALISIS=todos las encendería
# para todos, y la foto saldría de un código que los de fuera no ven. Y las del
# plan se leen AL IMPORTAR (ESFUERZO_PLAN, PLAN_TOPE_SALIDA…): se quitan antes.
for _v in list(ct.BANDERAS_REDISENO.values()) + [
        "ESTUDIO_PROMPT", "ESTUDIO_PROMPT_AD", "PLAN_ESTUDIO", "ESFUERZO_PLAN", "PLAN_TOPE_SALIDA",
        "PLAN_TOPE_CORRIDAS", "PLAN_ESPERA_S", "MODELO_PLAN", "PREGUNTA_DECISIVA_ACTIVA", "CNPCF_VIGENTE"]:
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
RUTA = os.path.join(AQUI, "datos", "instantaneas", "plan_paridad.json")

# ═══ LOS ASUNTOS, COPIADOS DE LAS PRUEBAS QUE YA EXISTEN ═══════════════════
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


# test_plan_estudio.py, líneas 527-560: el doble del modelo y el resultado de mentira.
class _Resp:
    def __init__(self, txt, fin="stop"):
        m = types.SimpleNamespace(content=txt)
        self.choices = [types.SimpleNamespace(message=m, finish_reason=fin)]
        self.usage = None


class Modelo:
    """Doble del cliente: devuelve los planes en orden y anota cada llamada."""

    def __init__(self, *planes, espera=0.0):
        self.planes = list(planes)
        self.kw = []
        self.espera = espera
        self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self._crear))

    async def _crear(self, **kw):
        self.kw.append(kw)
        if self.espera:
            await asyncio.sleep(self.espera)
        p = self.planes.pop(0) if len(self.planes) > 1 else self.planes[0]
        return _Resp(p if isinstance(p, str) else json.dumps(p, ensure_ascii=False))


def resultado(f=None, variante="v4", suplencia=None, formato=""):
    r = types.SimpleNamespace()
    r.fases = f or fases()
    r.encargo = types.SimpleNamespace(tipo_asunto="amparo_directo", es_recurso=False,
                                      variante_estudio=variante, formato=formato,
                                      suplencia=suplencia or {}, conceptos_violacion="",
                                      propuesta_global={"checklist": [{"tema": "identidad"}]},
                                      plan={}, guion="")
    r.avisos = []
    return r


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


# test_regresion_631.py, líneas 641-687: el plan-4 «infundado» guardado.
_PEND = ("A1.d", "A1.g", "A1.j", "A1.m", "A1.n", "A1.o", "A1.p", "A1.q", "A1.r", "A1.s", "A1.t", "A1.v", "A1.w")
_ATACA = {"A1.c": "P3", "A1.d": "P3", "A1.f": "P3", "A1.g": "P3", "A1.p": "P3", "A1.x": "P2"}
_segs10 = []
for _c in "abcdefghijklmnopqrstuvwx":
    _sid = f"A1.{_c}"
    _p = _sid in _PEND
    _segs10.append({"id": _sid, "problema_id": 1, "concepto": 1,
                    "vicio": "forma" if _c in "lmpqrst" else ("procesal" if _c in "vw" else "fondo"),
                    "ataca": _ATACA.get(_sid, "P1"), "reitera": None, "dato": None,
                    "etiqueta": "inoperante" if _sid in ("A1.n", "A1.o") else "infundado",
                    "razon": "" if _p else "fondo_desestimado",
                    "trat": "residual" if _sid == "A1.k" else "desarrolla",
                    "diferencia": ("precedente" if _c in "jorstvw" else "norma") if _p else None,
                    "pendiente": "razon" if _p else None, "sin_premisa": not _p and _sid != "A1.k"})
_segs10.append({"id": "A1.y", "problema_id": 1, "concepto": 1, "vicio": "omision", "ataca": "P1", "reitera": None,
                "dato": None, "etiqueta": "infundado", "razon": "", "trat": "desarrolla", "diferencia": "hecho",
                "pendiente": "razon"})
# Un inoperante que el planificador SÍ declaró caído por derivar de lo
# desestimado, nombrando la proposición (lo único que cae así desde la
# revisión adversarial del 28-sep-2026).
_segs10.append({"id": "A1.z", "problema_id": 1, "concepto": 1, "vicio": "fondo", "ataca": "P2", "reitera": None,
                "dato": None, "etiqueta": "inoperante", "razon": "deriva_de_desestimado", "razon_p": "P2",
                "trat": "residual", "diferencia": None, "pendiente": None})
_segs10.append({"id": "A2.a", "problema_id": 2, "concepto": 2, "vicio": "omision", "ataca": "P4", "reitera": None,
                "dato": None, "etiqueta": "infundado", "razon": "omision_inexistente", "trat": "remite",
                "diferencia": None, "pendiente": None})
PLAN4_GUARDADO = {
    "version": "plan-4", "tipo_asunto": "amparo_revision",
    "problemas": [{"id": 1, "pregunta": PREG1_631, "sentido": "infundado", "clase": "fondo", "jerarquia": "principal",
                   "grupo": ""},
                  {"id": 2, "pregunta": PREG2_631, "sentido": "infundado", "clase": "fondo", "jerarquia": "accesorio",
                   "grupo": ""}],
    "proposiciones": [{"id": "P1", "dice": "x", "caracter": "toral", "relacion": "suficiente"},
                      {"id": "P2", "dice": "x", "caracter": "accesoria", "relacion": "dependiente_de:P1"},
                      {"id": "P3", "dice": "x", "caracter": "accesoria", "relacion": "dependiente_de:P1"},
                      {"id": "P4", "dice": "x", "caracter": "toral", "relacion": "suficiente"}],
    "segmentos": _segs10,
    "premisas": [{"id": "M3", "responde_a": ["P4"], "fuentes": {"tesis": [], "normas": []}, "anclas": [],
                  "rastro": "razon", "rastro_cita": "x"}],
    "unidades": [{"id": "U1", "problemas": [1], "segmentos": ["A1.a", "A1.b", "A1.e", "A1.h", "A1.i", "A1.u"],
                  "premisa": None}]
                + [{"id": f"U{i + 2}", "problemas": [1], "segmentos": [x["id"]], "premisa": None}
                   for i, x in enumerate(s_ for s_ in _segs10 if s_["id"] not in
                                         ("A1.a", "A1.b", "A1.e", "A1.h", "A1.i", "A1.u", "A1.k", "A2.a"))]
                + [{"id": "U40", "problemas": [2], "segmentos": ["A2.a"], "premisa": "M3"}],
    "propuestas": [], "avisos_al_secretario": [], "orden": {"criterio": "prelacion", "por_que": ""},
}

# ═══ LAS VARIANTES QUE LAS PRUEBAS YA USAN ═════════════════════════════════
# test_plan_estudio.py §3: el problema 2 procesal, la suplencia confirmada, el
# orden con el principal procesal.
_fp = fases(problemas=[{"pregunta": PREG1, "cubre": [1, 3], "clase": "fondo", "jerarquia": "principal"},
                       {"pregunta": PREG2, "cubre": [2], "clase": "procesal", "jerarquia": "accesorio"}])
_cp = crit(s2="fundado", r2="La Sala debió estudiarla: " + RAZON2)
_cp2 = crit(s1="fundado", s2="innecesario", r2="Queda sin materia.")
_cp3 = crit(s1="infundado", s2="innecesario", r2="Queda sin materia.")
_c_f = crit(s1="fundado")
_sup = {"fraccion": "IV-b", "a_favor_de": "la parte quejosa", "confirmada": True}
_fo = fases(problemas=[{"pregunta": PREG1, "cubre": [1, 3], "clase": "fondo", "jerarquia": "accesorio"},
                       {"pregunta": PREG2, "cubre": [2], "clase": "procesal", "jerarquia": "principal"}])
_co = [f6.Criterio(problema=PREG1, sentido="infundado", razonamiento=RAZON1, jerarquia="accesorio"),
       f6.Criterio(problema=PREG2, sentido="inoperante", razonamiento=RAZON2, jerarquia="principal")]
_FUNDADOS = ("C1.a", "C1.b", "C3.a", "C3.b")

# LA CUESTIÓN DECISIVA DEL 631 como la deja `pregunta_decisiva.formular` (los
# valores de test_pregunta_decisiva.py, sobre las preguntas de este asunto).
DECISIVA_631 = {
    "formulada": True, "version": "decisiva-2", "numero": 1, "pregunta_recurrida": PREG1_631,
    "pregunta_decisiva": ("¿El tercero adquirente del inmueble objeto de un juicio sobre una acción personal "
                          "puede sustituirse válidamente a la actora en la ejecución de la sentencia?"),
    "figura": "SUSTITUCIÓN PROCESAL DEL ADQUIRENTE EN LA EJECUCIÓN",
    "proposicion_toral": {"dice": "La sustitución alteró la cosa juzgada.",
                          "cita": "la sustitución de la parte ejecutante alteró la cosa juzgada"},
    "hechos_que_deciden": ["La compraventa del inmueble arrendado.",
                           "El juicio natural versó sobre la rescisión del arrendamiento."]}


# El material con lo que `indice_material` recorta o aparta: una tesis de
# método (no es fuente), una de la técnica y una de la figura con su cupo.
def material_rico():
    return material(
        tesis=material().tesis + [
            {"registro": "2003333", "metodo": True, "rubro": "INTERPRETACIÓN CONFORME.", "texto": "x"},
            {"registro": "2004444", "tecnica": True, "rubro": "CONCEPTOS DE VIOLACIÓN. ORDEN DE SU ESTUDIO.",
             "texto": "El estudio de los conceptos de violación de fondo precede al de los procesales."},
            {"registro": "2005555", "cupo_figura": True, "rubro": "IDENTIDAD DEL BIEN. SU PRUEBA.",
             "texto": "La identidad del bien puede probarse por cualquier medio idóneo."}],
        normas=material().normas + [
            {"cuerpo_legal": "Constitución Política de los Estados Unidos Mexicanos", "articulo": "16",
             "texto": "Nadie puede ser molestado en su persona, familia, domicilio, papeles o posesiones."}])


def _crudo(d, tipo, suplencia=None):
    """El plan crudo tal como sale de `planear`: normalizado y sellado con la
    versión, el tipo y la suplencia (plan_estudio.py, `planear`)."""
    n = pe.normalizar(copy.deepcopy(d))
    n.update({"version": pe.PLAN_VERSION, "tipo_asunto": tipo,
              "suplencia": dict(suplencia) if isinstance(suplencia, dict) else {}})
    return n


def _al_estudio(plan, razones=None):
    """El plan que llega al estudio por el camino de main (`_taller_plan_para`):
    la razón que escribió una pantalla anterior primero, luego la dependencia."""
    return pe.resolver_por_dependencia(pe.aplicar_razones(plan, pe.leer_razones_segmento(razones or {})))


def _guiones(F, nombre, plan, *, concede=(None,), decisiva=None, moderna=True, bloque=False):
    """El guion estándar por cada `concede` y, si se pide, la moderna y el
    bloque. El bloque repite el guion entero más la descripción de sus
    rótulos: con dos `concede`, sólo el del que concede (trae más rótulos)."""
    for cd in concede:
        suf = "" if len(concede) == 1 else f"_concede_{'si' if cd else 'no'}"
        g = pe.vista(plan, "estandar", concede=cd, decisiva=decisiva)
        F[f"{nombre}/guion_estandar{suf}"] = g
        if bloque and (len(concede) == 1 or cd):
            F[f"{nombre}/bloque_estandar{suf}"] = pe.bloque(g)
    if moderna:
        F[f"{nombre}/guion_moderna"] = pe.vista(plan, "moderna", concede=concede[0], decisiva=decisiva)


def caso(F, nombre, crudo, *, c, segs, f, m, tipo, contexto="", suplencia=None, tocados=None,
         concede=None, decisiva=None, moderna=True, bloque=False, ficha=False, normalizado=False):
    """Un plan crudo por todo lo que el código le hace sin modelo. `concede`,
    si no se da, el de `concede_de` sobre un encargo de ese tipo (como main).
    `normalizado` sólo donde el plan crudo es otro: en una mutación del plan
    bueno es el mismo con un campo cambiado, y ese campo ya se ve en el
    reparado, en V0 y en el guion."""
    sup = suplencia or {}
    n = _crudo(crudo, tipo, sup)
    if normalizado:
        F[f"{nombre}/normalizado"] = n
    F[f"{nombre}/v0_sin_reparar"] = pe.validar(n, c, segs, f, m, contexto, suplencia=sup, tocados=tocados)
    r, av = pe.reparar(n, c, segs, f, m, contexto, sup, tocados=tocados)
    F[f"{nombre}/reparado"] = r
    F[f"{nombre}/avisos"] = av
    F[f"{nombre}/v0"] = pe.validar(r, c, segs, f, m, contexto, suplencia=sup, tocados=tocados)
    e = _al_estudio(r)
    # Casi siempre es el mismo (resolver es idempotente): se dice, no se repite.
    F[f"{nombre}/al_estudio"] = "= reparado" if e == r else e
    if concede is None:
        concede = (pe.concede_de(types.SimpleNamespace(
            fases=f, encargo=types.SimpleNamespace(tipo_asunto=tipo, resolvio_declarado="")), c),)
    _guiones(F, nombre, e, concede=concede, decisiva=decisiva, moderna=moderna, bloque=bloque)
    if ficha:
        F[f"{nombre}/ficha"] = pe.para_ficha(r, "usado", "clave-de-prueba", av)
    return r


def ad(F, nombre, crudo, c=None, f=None, **kw):
    """Un caso del amparo directo sintético (test_plan_estudio.py)."""
    return caso(F, nombre, crudo, c=c or crit(), segs=SEGS_N, f=f or fases(), m=material(),
                tipo="amparo_directo", **kw)


def rev(F, nombre, crudo, c=None, segs=None, f=None, **kw):
    """Un caso del amparo en revisión 631 (test_regresion_631.py)."""
    return caso(F, nombre, crudo, c=c or crit_631(), segs=segs or SEGS_631, f=f or fases_631(),
                m=material_631(), tipo="amparo_revision", **kw)


def _preparar(F, nombre, cli, r, c, m, contexto, suplencia, **kw):
    """`preparar` de punta a punta con el modelo de mentira. Del modelo se
    fotografía lo que recibió: el prompt de cada llamada y sus parámetros,
    salvo el NOMBRE del modelo (no es del plan; lo fija test_plan_estudio §5)."""
    plan, av, info = asyncio.run(pe.preparar(cli, r, c, m, contexto, suplencia, **kw))
    for i, k in enumerate(cli.kw, 1):
        F[f"{nombre}/llamada{i}_prompt"] = k["messages"][0]["content"]
        F[f"{nombre}/llamada{i}_parametros"] = {
            "roles": [x.get("role") for x in k.get("messages") or []],
            **{x: v for x, v in k.items() if x not in ("model", "messages")}}
    F[f"{nombre}/plan"] = plan
    F[f"{nombre}/avisos"] = av
    F[f"{nombre}/info"] = info
    return plan


# ═══ LA FOTO ═══════════════════════════════════════════════════════════════

def generar() -> dict:
    """Todo lo que se fotografía, {«grupo/pieza»: valor}. Cada llamada arma
    sus asuntos de cero: dos corridas seguidas deben dar lo mismo."""
    F = {}

    # ── Catálogos y vocabulario ───────────────────────────────────────────
    F["catalogos/PLAN_VERSION"] = pe.PLAN_VERSION
    F["catalogos/RAZONES"] = pe.RAZONES
    F["catalogos/conjuntos"] = {
        "TRATAMIENTOS": pe.TRATAMIENTOS, "_EN_UNIDAD": pe._EN_UNIDAD, "VICIOS": pe.VICIOS,
        "DIFERENCIAS": pe.DIFERENCIAS, "CARACTERES": pe.CARACTERES, "FUENTES_DATO": pe.FUENTES_DATO,
        "ETIQUETAS_DE_FONDO": pe.ETIQUETAS_DE_FONDO,
        "_SENTIDOS_NO_SE_ESTUDIA": pe._SENTIDOS_NO_SE_ESTUDIA,
        "_RAZONES_DE_INFUNDADO": pe._RAZONES_DE_INFUNDADO,
        "_RAZONES_DE_INOPERANTE": pe._RAZONES_DE_INOPERANTE, "_RAZONES_FORMA": pe._RAZONES_FORMA,
        "_SENTIDOS_SUFICIENCIA": pe._SENTIDOS_SUFICIENCIA, "_ETIQUETAS_DE_GRUPO": pe._ETIQUETAS_DE_GRUPO}
    # Las palabras, ESCRITAS y no sacadas de RAZONES: una razón nueva (el alias
    # de la etapa 4) cambia `catalogos/RAZONES` y nada más, y lo que ya se
    # entendía sigue entendiéndose igual.
    _palabras = (
        "adhesivo_fuera_182", "adhesivo_sin_materia", "ataca_accesoria", "cae_con_principal",
        "cosa_juzgada_amparo_previo", "deriva_de_desestimado", "esencialmente_fundado", "falsa_premisa",
        "fondo_desestimado", "fundado", "fundado_insuficiente", "generico", "inatendible", "ineficaz",
        "infundado", "innecesario", "innecesario_mayor_beneficio", "innecesario_por_suficiencia",
        "inoperante", "no_combate", "no_se_estudia", "novedoso", "omision_inexistente", "procesal_171_172",
        "reitera_sin_combatir", "sin_materia",
        "", "inventada", "Fundado pero insuficiente", "parcialmente fundado", "fundado suplido",
        "Cae con principal", "no_combate(P2)", "deriva_de_desestimado (p3)", "Innecesario por suficiencia")
    F["catalogos/vocabulario"] = {
        w: {"norm_sentido": pe.norm_sentido(w), "clase_sentido": pe.clase_sentido(w),
            "calificacion_conocida": pe.calificacion_conocida(w), "_razon": pe._razon(w),
            "_implica": pe._implica(w)} for w in _palabras}

    # ── El prompt del planificador ────────────────────────────────────────
    _probs = pe.problemas_del_criterio(crit(), fases())
    _ind = pe.indice_material(material(), fases())
    for t in ("amparo_directo", "amparo_revision", "queja"):
        F[f"prompt/{t}_minimo"] = pe.prompt_plan(
            tipo_asunto=t, probs=_probs, segs=SEGS_N, resumen_acto="r",
            tramos=pe._tramos_del_escrito(fases()), indice=_ind, n=3)
    F["prompt/amparo_directo_completo"] = pe.prompt_plan(
        tipo_asunto="amparo_directo", probs=pe.problemas_del_criterio(_c_f, fases()), segs=SEGS_N,
        resumen_acto=fases().resumen_acto, tramos=pe._tramos_del_escrito(fases()),
        indice=pe.indice_material(material_rico(), fases()),
        contexto="Constancias: el demandado confesó poseer dentro de la parcela de la actora (foja 12).",
        conceptos_violacion="PRIMERO. La Sala no valoró la inspección judicial.",
        contraste=[{"problema": 1, "razon_toral": "la confesión acredita la identidad", "combate": True}],
        checklist=[{"tema": "identidad"}, {"tema": "cosa juzgada refleja"}],
        suplencia=_sup, faltas=["faltan segmentos del inventario: C3.a", "U1: mezcla vicios (fondo, omision)"],
        n=3, decisiva=dict(DECISIVA_631, pregunta_recurrida=PREG1),
        ficha="LA FICHA PROCESAL (datos): quejosa, la parte demandada; acto, la sentencia de segunda instancia.")
    F["prompt/amparo_revision_631"] = pe.prompt_plan(
        tipo_asunto="amparo_revision", probs=pe.problemas_del_criterio(crit_631(), fases_631()),
        segs=SEGS_631, resumen_acto=ACTO_631, tramos=[("escrito", ESCRITO_631)],
        indice=pe.indice_material(material_631(), fases_631()), suplencia={"fraccion": "VI"},
        decisiva=DECISIVA_631)
    F["indice/material_rico"] = pe.indice_material(material_rico(), fases())

    # ── Amparo directo sintético: el plan bueno y lo que el código le hace ─
    ad(F, "ad.bueno", PLAN_BUENO, bloque=True, ficha=True, normalizado=True)
    ad(F, "ad.prelacion", mutar(lambda d: d["orden"].update(criterio="prelacion")), moderna=False)
    ad(F, "ad.fundado", PLAN_BUENO, c=_c_f, concede=(True, False), bloque=True, ficha=True)
    ad(F, "ad.fundado_tocado", PLAN_BUENO, c=_c_f, tocados=[PREG1], moderna=False)
    ad(F, "ad.fundado_en_infundado", mutar(lambda d: seg(d, "C1.a").update(etiqueta="fundado")), moderna=False)
    ad(F, "ad.innecesario_en_infundado", mutar(lambda d: seg(d, "C1.a").update(etiqueta="innecesario")),
       moderna=False)
    # El 642: la reconvención pide una CONSECUENCIA distinta de la del principal.
    ad(F, "ad.consecuencia",
       mutar(lambda d: [seg(d, x).update(etiqueta="fundado", razon="fundado") for x in ("C1.a", "C3.a", "C3.b")]
             + [seg(d, "C1.b").update(etiqueta="infundado", razon="fondo_desestimado", diferencia="consecuencia")]),
       c=_c_f, moderna=False)
    ad(F, "ad.malo", _malo, moderna=False, normalizado=True)
    ad(F, "ad.procesal_189",
       mutar(lambda d: (seg(d, "C2.a").update(etiqueta="innecesario", razon="innecesario_mayor_beneficio",
                                              trat="no_se_estudia", vicio="procesal"),
                        [seg(d, x).update(etiqueta="fundado", razon="fundado") for x in _FUNDADOS])),
       c=_cp2, f=_fp, moderna=False)
    ad(F, "ad.procesal_sin_estudio_del_criterio",
       mutar(lambda d: seg(d, "C2.a").update(etiqueta="innecesario", razon="cae_con_principal",
                                             trat="no_se_estudia", vicio="procesal")),
       c=_cp3, f=_fp, moderna=False)
    ad(F, "ad.innecesario_del_criterio",
       mutar(lambda d: [seg(d, x).update(etiqueta="fundado", razon="fundado") for x in _FUNDADOS]),
       c=_cp2, moderna=False)
    # Los suplidos (art. 79): el que prospera, avisado; el que no, a no_se_expresa.
    ad(F, "ad.suplidos",
       mutar(lambda d: (d["segmentos"].extend([
           {"id": "S1.a", "problema_id": 1, "vicio": "procesal", "ataca": None, "reitera": None, "dato": None,
            "etiqueta": "fundado", "razon": "fundado", "trat": "aplica", "diferencia": None, "pendiente": None},
           {"id": "S1.b", "problema_id": 1, "vicio": "fondo", "ataca": "P1", "reitera": None, "dato": None,
            "etiqueta": "infundado", "razon": "fondo_desestimado", "trat": "aplica", "diferencia": None,
            "pendiente": None}]),
           [seg(d, x).update(etiqueta="fundado", razon="fundado") for x in _FUNDADOS])),
       c=_c_f, suplencia=_sup, bloque=True, normalizado=True)
    ad(F, "ad.adhesivo", mutar(lambda d: d["segmentos"].append(dict(seg(d, "C2.a"), id="AD1.a"))), moderna=False)
    # Lo organizativo (test_plan_estudio §15 y §16).
    ad(F, "ad.unidad_mezcla_vicios",
       mutar(lambda d: seg(d, "C1.b").update(vicio="omision", razon="omision_inexistente")), moderna=False)
    ad(F, "ad.premisa_sin_rastro",
       mutar(lambda d: d["premisas"][0].update(
           rastro_cita="la confesión ficta basta siempre para identificar cualquier bien")), bloque=True)
    ad(F, "ad.rastro_con_bordes",
       mutar(lambda d: d["premisas"][0].update(
           rastro="material",
           rastro_cita="La identidad del inmueble puede acreditarse con la confesión del demandado.")),
       moderna=False)
    ad(F, "ad.premisa_inexistente", mutar(lambda d: d["unidades"][1].update(premisa="M9")), moderna=False)
    ad(F, "ad.razon_no_cabe",
       mutar(lambda d: (seg(d, "C1.b").update(razon="generico"),
                        seg(d, "C1.a").update(vicio="omision", razon="no_combate(P1)"))), moderna=False)
    _d16 = mutar(lambda d: seg(d, "C1.b").update(etiqueta="inoperante", razon="no_combate(P1)", trat="aplica"))
    _d16["premisas"][0]["rastro_cita"] = "una regla que no está en ninguna parte del expediente ni de la razón"
    ad(F, "ad.premisa_retirada_y_mezcla", _d16, moderna=False)
    # LAS BASES QUE LA ETAPA 4 TOCA (mapa: reparar p y el con_p de V0): un
    # `fundado_insuficiente` que nombra la proposición que sobrevive, un
    # `no_combate` que nombra —como pide el prompt— la que deja sin combatir, y
    # una proposición suficiente que nadie nombra. Hoy las dos primeras cuentan
    # P3 como «atacada» y callan su aviso de reparar (p); sólo P4 avisa.
    ad(F, "ad.bases_suficientes",
       mutar(lambda d: (
           d["proposiciones"].extend([
               {"id": "P3", "dice": "Condenó en costas en ambas instancias", "caracter": "toral",
                "relacion": "suficiente", "fuente": "reclamada",
                "cita": "condenó en costas en ambas instancias", "vinculada_por_ejecutoria": False},
               {"id": "P4", "dice": "No era necesaria la pericial en topografía", "caracter": "accesoria",
                "relacion": "suficiente", "fuente": "reclamada",
                "cita": "no era necesaria la prueba pericial en materia de topografía",
                "vinculada_por_ejecutoria": False}]),
           seg(d, "C1.b").update(etiqueta="fundado_insuficiente", razon="fundado_insuficiente(P3)"),
           seg(d, "C3.b").update(etiqueta="inoperante", razon="no_combate(P3)", ataca="P3", trat="residual",
                                 diferencia=None))),
       normalizado=True)
    # LO QUE `normalizar` DA POR OMISIÓN SIN DECIRLO (la etapa 4 lo cambia sólo
    # con la bandera): una relación «autonoma» o «conjunta» → «necesaria», un
    # carácter desconocido → «toral», un vicio desconocido → «fondo»; y ids,
    # etiquetas y razones mal escritas.
    ad(F, "ad.rarezas_del_modelo",
       mutar(lambda d: (
           d["proposiciones"][0].update(relacion="autonoma", caracter="principal"),
           d["proposiciones"][1].update(relacion="conjunta", caracter="", vinculada_por_ejecutoria="sí"),
           d["proposiciones"].append({"id": "p3", "dice": "La condena en costas", "caracter": "Accesoria",
                                      "relacion": "Dependiente de P1", "fuente": "reclamada",
                                      "cita": "condenó en costas en ambas instancias",
                                      "vinculada_por_ejecutoria": "no"}),
           seg(d, "C1.b").update(etiqueta="Fundado pero insuficiente", razon="Fundado insuficiente (p2)"),
           seg(d, "C3.b").update(etiqueta="", razon="No combate (P3)", diferencia="otra", vicio="sustantivo",
                                 trat="otro"),
           seg(d, "C1.a").update(id="c1.A"))),
       moderna=False, normalizado=True)

    # ── V0: las faltas literales que vuelven al planificador ──────────────
    def _v0(d, c=None, f=None, suplencia=None):
        n = _crudo(d, "amparo_directo", suplencia or {})
        return pe.validar(n, c or crit(), SEGS_N, f or fases(), material(), "", suplencia=suplencia or {})

    F["v0/rechazos"] = {
        "fundado_en_infundado": _v0(mutar(lambda d: seg(d, "C1.a").update(etiqueta="fundado"))),
        "infundado_en_inoperante": _v0(mutar(lambda d: seg(d, "C2.a").update(etiqueta="infundado",
                                                                              razon="fondo_desestimado"))),
        "innecesario_en_infundado": _v0(mutar(lambda d: seg(d, "C1.a").update(etiqueta="innecesario"))),
        "procesal_no_estudiada": _v0(mutar(lambda d: seg(d, "C2.a").update(
            etiqueta="fundado", razon="cae_con_principal", trat="no_se_estudia", vicio="procesal")), c=_cp, f=_fp),
        "procesal_sin_mayor_beneficio": _v0(mutar(lambda d: (
            seg(d, "C2.a").update(etiqueta="innecesario", razon="cae_con_principal", trat="no_se_estudia",
                                  vicio="procesal"),
            [seg(d, x).update(etiqueta="fundado", razon="fundado") for x in _FUNDADOS])), c=_cp2, f=_fp),
        "suplencia_y_generico": _v0(mutar(lambda d: seg(d, "C2.a").update(razon="generico")), suplencia=_sup),
        "cosa_juzgada_no_vinculada": _v0(mutar(lambda d: d["proposiciones"][1].update(
            vinculada_por_ejecutoria=False))),
        "accesorio_antes_del_principal": _v0(mutar(lambda d: d["unidades"].reverse())),
        "procesal_antes_que_fondo": _v0(mutar(lambda d: (
            seg(d, "C2.a").update(vicio="procesal"), d["unidades"].reverse(),
            d["orden"].update(por_que="el escrito"))), c=_co, f=_fo),
        "falta_un_segmento": _v0(mutar(lambda d: d["segmentos"].pop(3))),
        "segmento_inventado": _v0(mutar(lambda d: d["segmentos"].append(dict(seg(d, "C1.a"), id="C9.z")))),
        "problema_ajeno": _v0(mutar(lambda d: seg(d, "C2.a").update(problema_id=1))),
        "razon_incompatible": _v0(mutar(lambda d: seg(d, "C1.a").update(razon="fundado"))),
        "inoperancia_en_infundado": _v0(mutar(lambda d: seg(d, "C1.a").update(razon="no_combate(P1)"))),
        "pendiente_de_razon": _v0(mutar(lambda d: seg(d, "C3.b").update(pendiente="razon"))),
        "reitera_con_ancla_propia": _v0(mutar(lambda d: seg(d, "C3.b").update(reitera="C1.a", trat="remite"))),
        "no_se_expresa_en_un_concepto": _v0(mutar(lambda d: seg(d, "C1.b").update(trat="no_se_expresa_art79"))),
        "no_se_estudia_lo_que_decide": _v0(mutar(lambda d: seg(d, "C1.b").update(trat="no_se_estudia"))),
        "unidad_mezcla_vicios": _v0(mutar(lambda d: seg(d, "C1.b").update(vicio="omision",
                                                                           razon="omision_inexistente"))),
        "premisa_sin_rastro": _v0(mutar(lambda d: d["premisas"][0].update(
            rastro_cita="la confesión ficta basta siempre para identificar cualquier bien"))),
    }

    # ── El AR 631/2025 (amparo en revisión) ───────────────────────────────
    rev(F, "rev.631", PLAN_631, moderna=False, normalizado=True)
    rev(F, "rev.631_traducido", PLAN_631_C, moderna=False, normalizado=True)
    _d2 = copy.deepcopy(PLAN_631_C)
    _d2["segmentos"][0].update(etiqueta="fondo_desestimado", razon="fondo_desestimado",
                               trat="desarrolla", diferencia="norma")
    rev(F, "rev.631_portador", _d2, moderna=False)
    _d3 = copy.deepcopy(PLAN_631_C)
    _d3["segmentos"][2].update(etiqueta="", razon="no_combate(P1)")
    rev(F, "rev.631_etiqueta_vacia", _d3, moderna=False)
    _r9 = rev(F, "rev.plan6", PLAN6, segs=SEGS6, f=fases6(), concede=(False, True), bloque=True, ficha=True,
             normalizado=True)
    F["rev.plan6/guion_estandar_con_decisiva"] = pe.vista(_al_estudio(_r9), "estandar", concede=False,
                                                          decisiva=DECISIVA_631)

    # ── Planes GUARDADOS que se releen sin reparar (GET /taller/plan) ─────
    _g4 = _al_estudio(PLAN4_GUARDADO)
    F["guardado.plan4_631/al_estudio"] = _g4
    _guiones(F, "guardado.plan4_631", _g4, bloque=True)
    _g4r = _al_estudio(PLAN4_GUARDADO, json.dumps({"a1.D": "Lo resuelto en el juicio previo no vincula aquí.",
                                                   "A1.b": "   ", "A2.a": "Otra razón escrita antes."}))
    F["guardado.plan4_631_razon_de_pantalla/al_estudio"] = _g4r
    _guiones(F, "guardado.plan4_631_razon_de_pantalla", _g4r, moderna=False)
    _642 = json.load(open(os.path.join(AQUI, "datos", "congruencia", "642_v4.plan"), encoding="utf-8"))
    _p642 = {"segmentos": _642["segmentos"], "unidades": [], "premisas": [], "propuestas": [], "orden": {},
             "problemas": [{"id": 1, "sentido": "fundado"}, {"id": 2, "sentido": "inoperante"},
                           {"id": 3, "sentido": "fundado"}]}
    _g642 = _al_estudio(_p642)
    F["guardado.plan3_642/al_estudio"] = _g642
    _guiones(F, "guardado.plan3_642", _g642)

    # ── preparar de punta a punta (modelo de mentira) ─────────────────────
    _preparar(F, "preparar.ad_reintento", Modelo(_malo, PLAN_BUENO), resultado(), crit(), material(), "", {},
              segs=SEGS_N)
    _r631 = types.SimpleNamespace(
        fases=fases6(), avisos=[],
        encargo=types.SimpleNamespace(tipo_asunto="amparo_revision", es_recurso=True, variante_estudio="v4",
                                      formato="", suplencia={}, conceptos_violacion="",
                                      propuesta_global={"checklist": [{"tema": "sustitución"}]},
                                      plan={}, guion="", resolvio_declarado="Concedió el amparo."))
    _preparar(F, "preparar.631_con_decisiva", Modelo(PLAN6), _r631, crit_631(), material_631(),
              "Constancias: la adquirente exhibió la escritura pública.", {}, segs=SEGS6,
              contraste=[{"problema": 1, "razon_toral": "la sustitución alteró la cosa juzgada", "combate": True}],
              tocados=[PREG2_631], decisiva=DECISIVA_631)

    # ── La clave, la huella, concede_de ───────────────────────────────────
    F["clave/variantes"] = {
        "base": pe.clave(crit(), "h", "ctx", {}, "estandar"),
        "moderna": pe.clave(crit(), "h", "ctx", {}, "moderna"),
        "variante_v3": pe.clave(crit(), "h", "ctx", {}, "estandar", "v3"),
        "razon_cambiada": pe.clave(crit(r1=RAZON1 + " Además."), "h", "ctx", {}, "estandar"),
        "grupo": pe.clave(crit(g1="A", g2="A"), "h", "ctx", {}, "estandar"),
        "otra_huella": pe.clave(crit(), "h2", "ctx", {}, "estandar"),
        "otro_contexto": pe.clave(crit(), "h", "otro", {}, "estandar"),
        "suplencia_sin_confirmar": pe.clave(crit(), "h", "ctx", {"fraccion": "VII", "confirmada": False}),
        "suplencia_confirmada": pe.clave(crit(), "h", "ctx", {"fraccion": "VII", "confirmada": True}),
        "631": pe.clave(crit_631(), "h", "", {}),
        "631_con_decisiva": pe.clave(crit_631(), "h", "", {}, decisiva=DECISIVA_631),
        "sin_criterio": pe.clave([], "h", "", {}),
    }
    _rh = types.SimpleNamespace(fases=fases6(), encargo=types.SimpleNamespace(tipo_asunto="amparo_revision"))
    F["huella_entradas/variantes"] = {
        "ad": pe.huella_entradas(resultado(), material(), "", SEGS_N),
        "ad_con_conceptos": pe.huella_entradas(resultado(), material(), "PRIMERO. No valoró la inspección.", SEGS_N),
        "ad_material_rico": pe.huella_entradas(resultado(), material_rico(), "", SEGS_N),
        "ad_sin_segmentos": pe.huella_entradas(resultado(), material(), "", None),
        "631": pe.huella_entradas(_rh, material_631(), "", SEGS6),
    }
    F["concede_de/variantes"] = {
        "ad_infundado": pe.concede_de(resultado(), crit()),
        "ad_fundado": pe.concede_de(resultado(), _c_f),
        "631_revoca_la_concesion": pe.concede_de(types.SimpleNamespace(
            encargo=types.SimpleNamespace(tipo_asunto="amparo_revision", resolvio_declarado="Concedió el amparo."),
            fases=fases6()), crit_631()),
    }

    # ── La fila (taller_sesiones.plan), con relojes fijos ─────────────────
    _d1, _dec1 = pe.fila_pedir(None, "k1", "h1", 1000.0)
    _dx, _decx = pe.fila_pedir(_d1, "k1", "h1", 1001.0)
    _dl, _decl = pe.fila_latido(_d1, "k1", "h1", 1010.0)
    _dr, _estr = pe.fila_resultado(_dl, "k1", "h1", {"version": pe.PLAN_VERSION, "segmentos": []},
                                   ["un aviso"], 12.345, 1020.0)
    _df, _estf = pe.fila_resultado(_dl, "k1", "h1", None, ["V0 dos veces"], 3.0, 1020.0)
    _de, _este = pe.fila_resultado(_dl, "k1", "h1", None, ["un 500"], 3.0, 1020.0, reintentable=True)
    F["fila/secuencia"] = {
        "pedir": [_d1, _dec1], "pedir_otra_vez": [_dx, _decx], "latido": [_dl, _decl],
        "resultado": [_dr, _estr], "fallo": [_df, _estf], "error": [_de, _este],
        "pedir_tras_listo": list(pe.fila_pedir(_dr, "k1", "h1", 1030.0)),
        "pedir_otro_adelanto": list(pe.fila_pedir(_dr, "k2", "h2", 1030.0)),
        "resultado_de_otro_adelanto": list(pe.fila_resultado(_dr, "k1", "h9", {"x": 1}, [], 1.0, 1030.0)),
        "pantalla_listo": pe.estado_para_pantalla(_dr, "h1", 1021.0),
        "pantalla_en_curso": pe.estado_para_pantalla(_d1, "h1", 1001.0),
        "pantalla_abandonada": pe.estado_para_pantalla(_d1, "h1", 1000.0 + pe.PLAN_ABANDONADO_S + 1),
        "pantalla_error": pe.estado_para_pantalla(_de, "h1", 1021.0),
        "pantalla_otro_adelanto": pe.estado_para_pantalla(_dr, "h2", 1021.0),
        "tope": list(pe.fila_pedir(dict(_d1, corridas=pe.TOPE_CORRIDAS, planes={}), "k3", "h1", 1040.0)),
    }
    return F


# ═══ LA COMPARACIÓN ════════════════════════════════════════════════════════

def _canon(x):
    """El valor como lo guarda la fila (JSON): claves de texto, tuplas como
    listas, conjuntos ordenados (sólo los hay en los catálogos). Lo que no es
    JSON falla aquí, no en silencio."""
    if isinstance(x, dict):
        return {str(k): _canon(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_canon(v) for v in x]
    if isinstance(x, (set, frozenset)):
        return sorted(_canon(v) for v in x)
    if x is None or isinstance(x, (bool, int, float, str)):
        return x
    raise TypeError(f"no sé fotografiar un {type(x).__name__}")


def _pieza(x):
    """Un texto va renglón a renglón para que el diff se lea; `split("\\n")` no
    pierde nada: unidos, dan el mismo texto byte a byte."""
    return {"texto": x.split("\n")} if isinstance(x, str) else {"json": _canon(x)}


def _dump(x) -> str:
    return json.dumps(x, ensure_ascii=False, indent=1, sort_keys=True)


def _piezas(F) -> dict:
    return {k: _pieza(v) for k, v in F.items()}


def _renglones(p) -> list:
    return p["texto"] if "texto" in p else _dump(p.get("json")).split("\n")


def _ventana(x: str, i: int, ancho: int = 240) -> str:
    """El renglón recortado ALREDEDOR de su primera diferencia. El renglón
    RAZONES del guion pasa de mil caracteres: recortado por el principio, el
    cambio quedaba fuera y el diff enseñaba dos renglones iguales."""
    if len(x) <= ancho:
        return x
    ini = max(0, min(i - ancho // 3, len(x) - ancho))
    return ("…" if ini else "") + x[ini:ini + ancho] + ("…" if ini + ancho < len(x) else "")


def _primera_diferencia(a: str, b: str) -> int:
    return 1 + len(os.path.commonprefix([a[1:], b[1:]]))


def _diff(viejo, nuevo, tope=60):
    d = list(difflib.unified_diff(_renglones(viejo), _renglones(nuevo), "instantánea", "hoy", n=2, lineterm=""))
    es = lambda x, c: x.startswith(c) and not x.startswith(c * 3)
    salida, k = [], 0
    while k < len(d):
        if not es(d[k], "-"):
            salida.append(_ventana(d[k], 0))
            k += 1
            continue
        # Cada renglón quitado frente al que lo sustituye, para saber dónde cambia.
        j = k
        while j < len(d) and es(d[j], "-"):
            j += 1
        m = j
        while m < len(d) and es(d[m], "+"):
            m += 1
        quitados, puestos = d[k:j], d[j:m]
        for n, x in enumerate(quitados):
            salida.append(_ventana(x, _primera_diferencia(x, puestos[n] if n < len(puestos) else "")))
        for n, y in enumerate(puestos):
            salida.append(_ventana(y, _primera_diferencia(quitados[n] if n < len(quitados) else "", y)))
        k = m
    for x in salida[:tope]:
        print("         " + x)
    if len(salida) > tope:
        print(f"         … y {len(salida) - tope} renglones más del diff")


def comparar(viejas: dict, nuevas: dict, decir=ok) -> list:
    """Una comprobación por GRUPO (lo que va antes de «/»), con el diff de
    cada pieza que cambió. Devuelve los nombres de las que no casan."""
    malas = []
    grupos = {}
    for k in sorted(set(viejas) | set(nuevas)):
        grupos.setdefault(k.split("/", 1)[0], []).append(k)
    for g, ks in grupos.items():
        mal = [k for k in ks if k not in viejas or k not in nuevas or _dump(viejas[k]) != _dump(nuevas[k])]
        decir(not mal, f"{g}: {len(ks)} pieza{'s' if len(ks) != 1 else ''} idéntica{'s' if len(ks) != 1 else ''} "
                       f"a la instantánea" if not mal else f"{g}: cambia {', '.join(k.split('/', 1)[-1] for k in mal)}")
        for k in mal:
            if k not in viejas:
                print(f"      {k}: pieza NUEVA, la instantánea no la tiene")
            elif k not in nuevas:
                print(f"      {k}: la instantánea la tiene y hoy ya no se calcula")
            else:
                print(f"      {k}:")
                _diff(viejas[k], nuevas[k])
        malas += mal
    return malas


def _nota() -> dict:
    try:
        cab = subprocess.run(["git", "-C", AQUI, "rev-parse", "--short", "HEAD"], capture_output=True,
                             text=True, timeout=10).stdout.strip()
    except Exception:
        cab = ""
    return {"que_es": ("La salida de plan_estudio.py con las banderas del rediseño apagadas. La escribe "
                       "test_paridad_plan.py con FOTOGRAFIAR=1; no se edita a mano."),
            "hecha_sobre": cab, "banderas_apagadas": sorted(ct.BANDERAS_REDISENO)}


# ═══ EL HIJO: LA MISMA FOTO CON OTRA SEMILLA DE HASH ═══════════════════════
_MARCA = "=== PIEZAS DE LA PARIDAD ==="
if os.environ.get("PARIDAD_HIJO"):
    print(_MARCA + _dump(_piezas(generar())))
    sys.exit(0)


# ═══════════════════════════════════════════════════════════════════════════
print("\n1 · LAS BANDERAS DEL REDISEÑO, APAGADAS (lo que ven las cuentas de fuera)")
ok(not ct.es_de_pruebas() and not ct.es_casa() and ct.aplicado()["banderas"] == {},
   "el contexto es el de una cuenta de fuera: ni de casa ni de prueba, sin banderas pedidas")
# C8 (3-oct-2026, David: «empuja para todos los usuarios»): «procedencia_por_tipo»
# nace encendida para TODOS (`OMISION_REDISENO` = «todos»), también para las
# cuentas de fuera. No toca el plan, los prompts, la V0 ni el guion —la foto
# sigue idéntica byte a byte con ella encendida (sección 4)—; por eso se aparta
# de esta cuenta. Antes se exigía que no rigiera ninguna.
_PARA_TODOS = {k for k, v in getattr(ct, "OMISION_REDISENO", {}).items() if v == "todos"}
_encendidas = [b for b in ct.BANDERAS_REDISENO if ct.rediseno(b) and b not in _PARA_TODOS]
ok(not _encendidas and _PARA_TODOS <= {"procedencia_por_tipo"},
   f"ninguna bandera del rediseño rige ({len(ct.BANDERAS_REDISENO)} en la lista), salvo las que nacen "
   f"para todos ({sorted(_PARA_TODOS)})" + (f"; encendidas: {_encendidas}" if _encendidas else ""))
ok({"plan_con_analisis", "exclusiones_con_prueba", "estados_sesion"} <= set(ct.BANDERAS_REDISENO),
   "las tres de la etapa 4 están en la lista (y por eso, apagadas aquí)")

print("\n2 · LOS ASUNTOS COPIADOS DICEN LO QUE DICEN EN SUS PRUEBAS")
FOTO = generar()
ok(FOTO["ad.bueno/v0"] == [] and FOTO["ad.bueno/v0_sin_reparar"] == [] and FOTO["ad.bueno/avisos"] == []
   and [s["etiqueta"] for s in FOTO["ad.bueno/reparado"]["segmentos"]]
   == ["infundado", "infundado", "inoperante", "infundado", "infundado"],
   "el plan bueno del amparo directo pasa V0 y conserva la etiqueta de cada argumento (test_plan_estudio §2)")
_j9 = next(j for j in FOTO["rev.plan6/reparado"]["jerarquia"] if j["problema"] == 1)
ok(_j9["decide"] == "A1.b" and FOTO["rev.plan6/v0"] == [],
   "en el PLAN6 del 631 decide A1.b y V0 queda limpio (test_regresion_631 §9)")
ok(next(s for s in FOTO["guardado.plan4_631/al_estudio"]["segmentos"] if s["id"] == "A1.z")["dependencia"]
   == "deriva", "en el plan-4 guardado, A1.z cae por derivar (test_regresion_631 §10)")
ok(FOTO["preparar.ad_reintento/plan"] is not None and FOTO["preparar.ad_reintento/info"]["intentos"] == 2
   and "LO QUE EL CÓDIGO RECHAZÓ" in FOTO["preparar.ad_reintento/llamada2_prompt"],
   "preparar: el primero falla V0 y el reintento, con la lista de lo rechazado, pasa (test_plan_estudio §5)")
ok(FOTO["preparar.631_con_decisiva/info"]["intentos"] == 1
   and "LA CUESTIÓN QUE DECIDE" in FOTO["preparar.631_con_decisiva/llamada1_prompt"],
   "preparar del 631: pasa al primer intento y el planificador recibe la cuestión decisiva")

print("\n3 · LA FOTO NO DEPENDE DE NADA MÁS QUE DE LA ENTRADA")
ok(_dump(_piezas(generar())) == _dump(_piezas(FOTO)),
   "dos corridas seguidas dan lo mismo: ninguna función cambia los asuntos que recibe")
_semilla = "1" if os.environ.get("PYTHONHASHSEED") == "0" else "0"
_env = {k: v for k, v in os.environ.items() if k != "FOTOGRAFIAR"}
_env.update({"PARIDAD_HIJO": "1", "PYTHONHASHSEED": _semilla})
_hijo = subprocess.run([sys.executable, os.path.abspath(__file__)], cwd=AQUI, env=_env, capture_output=True,
                       text=True, timeout=600)
_out = _hijo.stdout.split(_MARCA, 1)
ok(_hijo.returncode == 0 and len(_out) == 2,
   f"la misma foto en otro proceso con PYTHONHASHSEED={_semilla}"
   + ("" if _hijo.returncode == 0 and len(_out) == 2 else
      f" (salida {_hijo.returncode}: {_hijo.stderr.strip()[-400:]})"))
if len(_out) == 2:
    _malas_h = comparar(json.loads(_out[1]), _piezas(FOTO), decir=lambda c, q: None)
    ok(not _malas_h, "y es idéntica: nada depende del orden de un conjunto"
                     + (f" (cambian con la semilla: {', '.join(_malas_h[:8])})" if _malas_h else ""))

print("\n4 · LA SALIDA DE HOY, IDÉNTICA A LA INSTANTÁNEA")
NUEVAS = _piezas(FOTO)
if os.environ.get("FOTOGRAFIAR", "").strip() in ("1", "si", "sí", "true"):
    if os.path.exists(RUTA):
        print("   (FOTOGRAFIAR: lo que cambia respecto de la instantánea anterior)")
        _cambian = comparar(json.load(open(RUTA, encoding="utf-8")).get("piezas") or {}, NUEVAS,
                            decir=lambda c, q: print(f"   {'igual' if c else 'CAMBIA'}  {q}"))
        print(f"   ({len(_cambian)} pieza(s) cambian)")
    os.makedirs(os.path.dirname(RUTA), exist_ok=True)
    with open(RUTA, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(_dump({"_nota": _nota(), "piezas": NUEVAS}) + "\n")
    print(f"   INSTANTÁNEA ESCRITA: {os.path.relpath(RUTA, AQUI)} ({len(NUEVAS)} piezas, "
          f"{os.path.getsize(RUTA) // 1024} KB)")
if not os.path.exists(RUTA):
    ok(False, f"no hay instantánea en {os.path.relpath(RUTA, AQUI)}: se hace con FOTOGRAFIAR=1 sobre el código "
              f"que se quiere fijar")
else:
    _doc = json.load(open(RUTA, encoding="utf-8"))
    VIEJAS = _doc.get("piezas") or {}
    print(f"   (instantánea hecha sobre {(_doc.get('_nota') or {}).get('hecha_sobre') or '?'}, "
          f"{len(VIEJAS)} piezas)")
    _malas = comparar(VIEJAS, NUEVAS)
    ok(not _malas and _dump(VIEJAS) == _dump(NUEVAS),
       "TODO idéntico byte a byte: prompts, planes, V0, guiones, claves y huellas" if not _malas else
       f"{len(_malas)} pieza(s) distintas de la instantánea: si el cambio es para las cuentas de fuera, va "
       f"detrás de su bandera; si es a propósito, FOTOGRAFIAR=1 y se explica en el commit")

print()
if FALLOS:
    print(f"FALLAN {len(FALLOS)}: " + " · ".join(FALLOS))
    sys.exit(1)
print("RESULTADO: TODAS LAS COMPROBACIONES PASAN")
