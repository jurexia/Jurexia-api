# -*- coding: utf-8 -*-
"""EL RETROCESO DEL 28-SEP-2026 (AR 631/2025), SIN LLAMAR A NINGÚN MODELO.

David generó en la cuenta de soporte un amparo en revisión que la tercera
interesada interpuso contra una sentencia que CONCEDIÓ el amparo. Dictó
«fundado» (revocar) para todo el asunto. El proyecto salió así:

  1. LA RAZÓN ARGUMENTABA LO CONTRARIO DEL SENTIDO. El cuadro de la razón
     global traía la del motor para «infundado»; al pedir la de «fundado», esa
     razón viajó como «la base del secretario —desarróllala, no la discutas—» y
     el modelo la desarrolló: «la sustitución sí alteró la cosa juzgada…».
  2. DIECINUEVE ARGUMENTOS «FUNDADOS» QUE EL PLANIFICADOR HABÍA DESESTIMADO. El
     planificador escribió la RAZÓN «fondo_desestimado» en el campo de la
     calificación; `reparar` la trató como ilegible y les puso la del problema.
  3. REVOCÓ PARA VOLVER A CONCEDER. Lo que hizo el juzgado se leyó contando
     verbos en el PDF entero —la sentencia narraba un amparo ANTERIOR que «negó
     el amparo»— y salió «niega»; con el recurso fundado, la rama fue
     «revoca_fondo_concede» y el resolutivo amparó a la recurrente contra un
     juzgado cuyo acto se había sobreseído. Su propio resolutivo decía «ampara y
     protege»: revocar esa concesión es NEGAR.

Todo es SINTÉTICO: nombres, números y fechas de prueba.

    .venv/bin/python test_regresion_631.py
"""
import ast
import copy
import datetime as _dt
import os
import sys
import tempfile
import types

os.environ.pop("ESTUDIO_PROMPT", None)
os.environ.pop("PLAN_ESTUDIO", None)

import fase5_propuesta as f5
import fase6_estudio as f6
import fase_rama as fr
import fases123_pipeline as f123
import modos_decision as md
import plan_estudio as pe
import tipos_asunto as ta

FALLOS = []


def ok(cond, que):
    print(f"   {'PASA ' if cond else 'FALLA'}  {que}")
    if not cond:
        FALLOS.append(que)


AQUI = os.path.dirname(os.path.abspath(__file__))

# ═══ EL ASUNTO DE PRUEBA ═══════════════════════════════════════════════════
ACTO = ("El Juzgado de Distrito consideró que la sustitución de la parte ejecutante alteró la cosa "
        "juzgada, porque la adquirente del inmueble no era causahabiente del contrato de "
        "arrendamiento que sustentó la acción personal de rescisión, y concedió el amparo. "
        "Consideró innecesario estudiar los restantes conceptos de violación.")
ESCRITO = (
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
_o = [ESCRITO.index(x) for x in ("PRIMERO.", "SEGUNDO.")] + [len(ESCRITO)]
CONTEO = {"estado": "contado", "n": 2, "tramos": [[_o[i], _o[i + 1]] for i in range(2)]}
PREG1 = "¿La sustitución de la parte ejecutante alteró la cosa juzgada?"
PREG2 = "¿La sentencia recurrida fue exhaustiva y debidamente fundada al resolver sobre la sustitución?"
RAZON1 = ("La interlocutoria sólo determinó quién cuenta con legitimación para continuar la "
          "ejecución, sin modificar las prestaciones de la sentencia definitiva; el artículo 49 del "
          "Código Procesal Civil local permite al causahabiente sustituir a quien transmitió el "
          "derecho controvertido.")
RAZON2 = "Dado el sentido del estudio del problema principal, queda sin materia el análisis de este planteamiento."


def fases(**kw):
    f = f123.Fases123(
        resumen_acto=ACTO,
        resumen_conceptos="Primero: la interlocutoria no alteró la cosa juzgada.\nSegundo: reproduce el primero.",
        problemas=[{"pregunta": PREG1, "cubre": [1], "clase": "fondo", "jerarquia": "principal",
                    "combate": "La recurrente sostiene que la interlocutoria sólo determinó quién "
                               "ejecuta, sin modificar las prestaciones.",
                    "resolvio": "El Juzgado de Distrito determinó que la sustitución alteró la cosa "
                                "juzgada y concedió el amparo."},
                   {"pregunta": PREG2, "cubre": [2], "clase": "fondo", "jerarquia": "accesorio",
                    "depende_de": 1}],
        fuentes=[ACTO, ESCRITO], conteo=dict(CONTEO))
    for k, v in kw.items():
        setattr(f, k, v)
    return f


def crit(s1="fundado", s2="innecesario"):
    return [f6.Criterio(problema=PREG1, sentido=s1, razonamiento=RAZON1, jerarquia="principal"),
            f6.Criterio(problema=PREG2, sentido=s2, razonamiento=RAZON2, jerarquia="accesorio")]


def material():
    return f6.Material(tesis=[], normas=[
        {"cuerpo_legal": "Código Procesal Civil local", "articulo": "49",
         "texto": "El causahabiente puede sustituir a quien transmitió el derecho controvertido."}],
        tipo_asunto="amparo_revision", materia="civil", variante="v4")


SEGS = [pe.normalizar_segmento_piso(s) for s in [
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


def _s(sid, **kw):
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
        _s("A1.a", etiqueta="fundado", razon="fundado", trat="aplica", diferencia=None),
        _s("A1.b"), _s("A1.c", vicio="fondo"), _s("A1.d", vicio="forma"), _s("A1.e"),
        _s("A2.a", problema_id=2, etiqueta="innecesario", razon="cae_con_principal",
           trat="no_se_estudia", diferencia=None, ataca=None)],
    "premisas": [],
    "unidades": [{"id": "U1", "problemas": [1], "segmentos": ["A1.a"], "premisa": None, "objecion": None},
                 {"id": "U2", "problemas": [1], "segmentos": ["A1.b", "A1.c", "A1.e"], "premisa": None,
                  "objecion": None},
                 {"id": "U3", "problemas": [1], "segmentos": ["A1.d"], "premisa": None, "objecion": None}],
    "propuestas": [], "avisos_al_secretario": [],
    "orden": {"criterio": "promovente", "por_que": "el orden del escrito"},
}


def _norm(d):
    p = pe.normalizar(copy.deepcopy(d))
    p["tipo_asunto"] = "amparo_revision"
    return p


def _rep(d, c=None):
    return pe.reparar(_norm(d), c or crit(), SEGS, fases(), material(), "", {})


def seg(p, sid):
    return next(s for s in p["segmentos"] if s["id"] == sid)


# ═══════════════════════════════════════════════════════════════════════════
print("\n1 · EL PLAN: LA RAZÓN ESCRITA COMO CALIFICACIÓN NO HEREDA LA DEL PROBLEMA")
ok(pe.etiqueta_fuera("fondo_desestimado", "fundado") != "",
   "el caso reproduce la trampa: «fondo_desestimado» no es una calificación del catálogo")
_r, _av = _rep(PLAN_631)
_et = {s["id"]: s["etiqueta"] for s in _r["segmentos"]}
ok(_et["A1.a"] == "fundado", "el argumento que el planificador fundó sigue fundado")
ok(all(_et[x] == "infundado" for x in ("A1.b", "A1.c", "A1.d", "A1.e")),
   f"los desestimados salen «infundado», la calificación que su razón implica, NO «fundado»: {_et}")
ok(sum(1 for x in ("A1.a", "A1.b", "A1.c", "A1.d", "A1.e") if _et[x] == "fundado") == 1,
   "en el problema que prospera, UN argumento lo funda: los demás no heredan su sentido")
ok(all(seg(_r, x)["razon"] == "fondo_desestimado" for x in ("A1.b", "A1.c", "A1.d", "A1.e")),
   "la razón del planificador se queda (antes se «ajustaba» a «fundado»)")
ok(not any("lleva la calificación del problema" in a for a in _av),
   "ningún argumento «lleva la calificación del problema»")
ok(any("leídas como la calificación que esa razón implica" in a and "A1.b" in a
       for a in _r["avisos_al_secretario"]),
   "el panel dice qué calificaciones se leyeron de su razón")
ok(_et["A2.a"] == "innecesario", "el accesorio que no se estudia conserva su sentido")
_f = pe.validar(_r, crit(), SEGS, fases(), material(), "", suplencia={})
ok(not any("etiqueta" in x or "problema 1" in x for x in _f),
   f"V0 no rechaza las calificaciones del plan reparado: {[x for x in _f if 'etiqueta' in x][:3]}")

print("\n2 · EL PLAN: TODOS DESESTIMADOS DENTRO DE UN PROBLEMA FUNDADO → LO FUNDA UNO, NO TODOS")
_d = copy.deepcopy(PLAN_631)
_d["segmentos"][0].update(etiqueta="fondo_desestimado", razon="fondo_desestimado",
                          trat="desarrolla", diferencia="norma")
_r2, _av2 = _rep(_d)
_et2 = {s["id"]: s["etiqueta"] for s in _r2["segmentos"]}
ok(_et2["A1.a"] == "fundado" and all(_et2[x] == "infundado" for x in ("A1.b", "A1.c", "A1.d", "A1.e")),
   f"lo funda el primero que ataca la proposición toral; los demás conservan lo suyo: {_et2}")
ok(_r2["propuestas"] and _r2["propuestas"][0]["a"] == "infundado" and _r2["propuestas"][0]["de"] == "fundado",
   "lo que opinaba el planificador va a PROPUESTA del problema (sólo la aplica el secretario)")
ok(any("problema 1" in a and "A1.a" in a and "conservan" in a for a in _r2["avisos_al_secretario"]),
   "y el panel dice quién lo funda y que los demás conservan su calificación")
ok(pe.problemas_sin_quien_los_funde(_r2["segmentos"], pe.problemas_del_criterio(crit(), fases())) == [],
   "el sentido del secretario tiene quien lo funde")

print("\n3 · EL PLAN: LA ETIQUETA VACÍA SE LEE DE SU RAZÓN")
_d = copy.deepcopy(PLAN_631)
_d["segmentos"][2].update(etiqueta="", razon="no_combate(P1)")
_r3, _ = _rep(_d)
ok(seg(_r3, "A1.c")["etiqueta"] == "inoperante",
   f"vacía con razón «no_combate» → «inoperante», no «fundado»: {seg(_r3, 'A1.c')['etiqueta']}")
ok(pe.PLAN_VERSION == "plan-5", "la versión del plan sube: un plan-4 guardado con la herencia no se reutiliza")
_pr = pe.prompt_plan(tipo_asunto="amparo_revision", probs=pe.problemas_del_criterio(crit(), fases()),
                     segs=SEGS, resumen_acto=ACTO, tramos=[("escrito", ESCRITO)],
                     indice=pe.indice_material(material(), fases()))
ok("etiqueta (la calificación del segmento): fundado | esencialmente_fundado" in _pr
   and "nunca en «etiqueta»" in _pr,
   "el prompt del planificador trae el catálogo de las etiquetas y dice que la razón no va ahí")

# ═══════════════════════════════════════════════════════════════════════════
print("\n4 · LA RAZÓN DE LA OTRA VÍA NO ES LA DEL SECRETARIO")
R_MOTOR = ("La adquisición del inmueble no acredita la transmisión de los derechos litigiosos; la "
           "sustitución alteró los límites subjetivos de la cosa juzgada.")
R_ALT = ("La interlocutoria sólo sustituyó a quien ejecuta, sin alterar la cosa, la causa ni las "
         "prestaciones de la sentencia firme.")
GLOB = {"sentido": "infundado", "razon": R_MOTOR,
        "alternativa": {"sentido": "fundado", "razon": R_ALT}}
_t, _a = md.razon_de_la_otra_via(R_MOTOR, "fundado", GLOB)
ok(_t == R_ALT and "VÍA CONTRARIA" in _a,
   "el eco del motor para «infundado» pedido como base de «fundado» → la de la vía contraria ya escrita")
_t, _a = md.razon_de_la_otra_via("  " + R_MOTOR.replace(" ", "  ") + " ", "fundado", GLOB)
ok(_t == R_ALT, "los espacios no esconden el eco")
_t, _a = md.razon_de_la_otra_via(R_MOTOR, "fundado", {"sentido": "infundado", "razon": R_MOTOR})
ok(_t == "" and "no se usó" in _a, "sin vía contraria escrita, la razón se queda vacía y se dice")
_t, _a = md.razon_de_la_otra_via(R_MOTOR, "inoperante", GLOB)
ok(_t == R_MOTOR and _a == "", "la misma vía (infundado → inoperante) no es eco: se respeta")
_t, _a = md.razon_de_la_otra_via("Mi razón: la interlocutoria no modificó las prestaciones.", "fundado", GLOB)
ok(_a == "" and _t.startswith("Mi razón"), "lo que escribió el secretario se respeta siempre")
_t, _a = md.razon_de_la_otra_via(R_MOTOR, "fundado", None)
ok(_t == R_MOTOR and _a == "", "sin propuesta global no hay nada que comparar")

print("\n5 · LA RAZÓN SABE QUIÉN GANA SI PROSPERA")
_p = f5.prompt_razon(PREG1, "fundado", f6.Material(), ACTO, "", True, "amparo_revision",
                     combate=fases().problemas[0]["combate"], resolvio=fases().problemas[0]["resolvio"])
ok("QUÉ QUIERE DECIR ESA CALIFICACIÓN EN ESTE PLANTEAMIENTO" in _p
   and "prosperan los agravios: cae lo que resolvió" in _p
   and "sólo determinó quién ejecuta" in _p and "concedió el amparo" in _p,
   "«fundado» en una revisión: prosperan los agravios y cae lo resuelto, con lo que sostienen y lo resuelto")
_p = f5.prompt_razon(PREG1, "infundado", f6.Material(), ACTO, "", True, "amparo_revision",
                     combate="x sostiene y", resolvio="el juez resolvió z")
ok("no prosperan los agravios: subsiste lo que resolvió" in _p, "«infundado»: subsiste lo resuelto")
_p = f5.prompt_razon(PREG2, "innecesario", f6.Material(), ACTO, "", True, "amparo_revision",
                     combate="x", resolvio="z")
ok("QUÉ QUIERE DECIR" not in _p, "lo que no se estudia no lleva dirección")
_p = f5.prompt_razon(PREG1, "fundado", f6.Material(), ACTO, "", True, "amparo_revision",
                     directriz="Una base cualquiera del secretario.")
ok("La calificación de arriba es la decisión" in _p,
   "con directriz: la base se lleva hasta la calificación, no a la contraria")
ok("QUÉ QUIERE DECIR" not in f5.prompt_razon(PREG1, "fundado", f6.Material(), ACTO, "", True,
                                              "amparo_revision"),
   "sin datos del problema, el prompt queda como antes")

print("\n6 · LAS PUERTAS DEL SERVIDOR USAN LA REGLA (por AST)")
_src = open(os.path.join(AQUI, "main.py"), encoding="utf8").read()
_arbol = ast.parse(_src)


def _fn(nombre):
    return next(n for n in ast.walk(_arbol)
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == nombre)


def _llama(fn, atributo):
    return sum(1 for n in ast.walk(fn) if isinstance(n, ast.Attribute) and n.attr == atributo)


_raz = _fn("taller_razonar")
ok(_llama(_raz, "razon_de_la_otra_via") >= 1, "/taller/razonar limpia la directriz que es eco de la otra vía")
_kws = {k.arg for n in ast.walk(_raz) if isinstance(n, ast.Call) and getattr(n.func, "attr", "") == "razonar"
        for k in n.keywords}
ok({"combate", "resolvio"} <= _kws, "/taller/razonar le pasa al modelo qué sostienen y qué se resolvió")
ok(_llama(_fn("_taller_armar_criterio"), "razon_de_la_otra_via") >= 2,
   "el criterio común limpia el eco en el modo global y por problema (las cuatro puertas lo usan)")

# ═══════════════════════════════════════════════════════════════════════════
print("\n7 · LO QUE HIZO EL JUZGADO LO DICE SU RESOLUTIVO")
RESOL = ("La Justicia de la Unión ampara y protege a Unión Ejemplo, A.C., en contra del acto "
         "atribuido a la Sala Civil Uno del Tribunal Superior de Justicia, por los motivos "
         "expresados en el considerando séptimo de esta sentencia y para los efectos precisados en "
         "el último considerando.")
SENTENCIA = (
    "ANTECEDENTES. Inconforme, la demandada promovió amparo directo y el Tribunal Colegiado negó el "
    "amparo solicitado. Después, en otro juicio, el Tribunal negó la protección. Más tarde el "
    "Colegiado negó el amparo por segunda vez. "
    "CONSIDERANDO SÉPTIMO. Es fundado el concepto de violación y procede conceder el amparo. "
    "Por lo expuesto, se R E S U E L V E: ÚNICO. " + RESOL + " Notifíquese.")
ok(fr.resolvio_a_quo(SENTENCIA) == "niega",
   "el caso reproduce la trampa: contar verbos en todo el texto da «niega» (un amparo anterior)")
ok(fr.resolvio_segun_resolutivos(SENTENCIA) == "concede", "sus puntos resolutivos dicen «concede»")
ok(fr.resolvio_a_quo(SENTENCIA, resolutivo=fr.resolutivo_recurrida(SENTENCIA)) == "concede",
   "con el resolutivo delante, `resolvio_a_quo` dice «concede»")
_mixta = ("se R E S U E L V E: PRIMERO. Se sobresee en el juicio respecto del acto del Juzgado "
          "Quinto. SEGUNDO. " + RESOL + " Notifíquese.")
ok(fr.resolvio_segun_resolutivos(_mixta) == "sobresee_concede", "sobresee un acto y ampara por otro → mixta")
ok(fr.que_dice_el_resolutivo("La Justicia de la Unión no ampara ni protege a X contra el acto de Y.")
   == "niega", "«no ampara ni protege» → niega")
ok(fr.que_dice_el_resolutivo("ampara y protege a X; no ampara ni protege a Z") == "",
   "ampara a uno y niega a otro: no se adivina")
ok(fr.que_dice_el_resolutivo("Se estima que el juez negó el amparo.") == "",
   "la prosa en pasado no es un resolutivo")
_fx = types.SimpleNamespace(resolutivo_recurrida=RESOL, resolvio_a_quo="niega",
                            antecedentes="El juzgado negó el amparo.")
ok(fr.que_hizo_el_juzgado(_fx, "") == "concede", "el resolutivo manda sobre la lectura guardada del PDF")
_fx = types.SimpleNamespace(resolutivo_recurrida="", resolvio_a_quo="",
                            antecedentes="Seguido el juicio, el juzgado concedió el amparo a la quejosa.")
ok(fr.que_hizo_el_juzgado(_fx, "") == "concede",
   "los antecedentes son una CADENA: ya no se leen letra por letra")
_fx = types.SimpleNamespace(resolutivo_recurrida="", resolvio_a_quo="", antecedentes="")
ok(fr.que_hizo_el_juzgado(_fx, "Concedió el amparo porque la sustitución alteró la cosa juzgada.")
   == "concede", "sin papel, lo declarado por el motor")
ok(ta.rama_revision("concede", "fundado") == "revoca_fondo_niega",
   "juzgado que concede + recurso fundado → se revoca y se NIEGA")
_pts = ta.puntos_revoca_concesion(fr.resolutivo_recurrida(SENTENCIA))
ok(_pts == ["PRIMERO. Se revoca la sentencia recurrida.",
            "SEGUNDO. La Justicia de la Unión no ampara ni protege a Unión Ejemplo, A.C., en contra del "
            "acto atribuido a la Sala Civil Uno del Tribunal Superior de Justicia, por los motivos y "
            "fundamentos expuestos en el último considerando de esta ejecutoria."],
   f"revocar la concesión es negar ESO, con el sujeto y el acto del juzgado: {_pts}")
ok(ta.puntos_revoca_concesion("Se sobresee en el juicio.") == []
   and ta.puntos_revoca_concesion("La Justicia de la Unión no ampara ni protege a X contra el acto de Y.") == [],
   "si el resolutivo no es un «ampara y protege» limpio, va la fórmula genérica")

print("\n8 · EL DOCUMENTO: REVOCA Y NIEGA A QUIEN SE AMPARÓ, AUNQUE LA SESIÓN GUARDARA «niega»")
try:
    import documento_generado as dg
    import fase0_oportunidad as f0
    from docx import Document

    _est = dg.Estructura(apertura="V.", visto="para resolver.",
                         resultandos=[{"titulo": "Presentación de la demanda de amparo indirecto",
                                       "texto": "Unión Ejemplo, A.C. promovió amparo contra la Sala "
                                                "Civil Uno y el Juzgado Quinto."}],
                         competencia="", existencia="", procedencia="")
    _c = f0.computar(_dt.date(2026, 3, 2), _dt.date(2026, 3, 9), plazo=10)
    _ruta = os.path.join(tempfile.mkdtemp(prefix="regresion631_"), "ar.docx")
    dg.componer({"tipo_asunto": "amparo_revision", "numero": "1/2026",
                 "encabezado": "AMPARO EN REVISIÓN 1/2026",
                 # Lo que tecleó el formulario en el 631: la RECURRENTE en «quejoso».
                 "quejoso": "Inmobiliaria Ejemplo, S.A. de C.V.",
                 "recurrente": "Inmobiliaria Ejemplo, S.A. de C.V.",
                 "responsable": "Juzgado Séptimo de Distrito", "magistrado": "M", "secretario": "S",
                 "tribunal": "Tercer Tribunal Colegiado", "ciudad": "Querétaro",
                 "resolvio_a_quo": "niega",               # lo que guardó la sesión
                 "resolutivo_recurrida": fr.resolutivo_recurrida(SENTENCIA)},
                _est, _c, f0.fecha_en_letra, _ruta,
                estudio=["Resolución recurrida",
                         "El Juzgado de Distrito precisó que la protección se concedía para que la "
                         "Sala Civil Uno dejara insubsistente la resolución reclamada.",
                         "Solución",
                         "Es fundado el primer agravio: la interlocutoria sólo determinó quién "
                         "ejecuta, sin modificar las prestaciones; por tanto, se revoca la sentencia "
                         "recurrida y procede negar el amparo."],
                calificaciones=["fundado"], tipo_asunto="amparo_revision")
    _ps = [q.text for q in Document(_ruta).paragraphs if q.text.strip()]
    _res = " ".join(_ps[next(i for i, t in enumerate(_ps) if "R E S U E L V E" in t or t.strip() == "RESUELVE"):])
    ok("Se revoca la sentencia recurrida" in _res, "PRIMERO: se revoca")
    ok("no ampara ni protege a Unión Ejemplo, A.C." in _res and "Sala Civil Uno" in _res,
       "SEGUNDO: NIEGA, a quien amparó el juzgado y contra el acto por el que lo amparó")
    ok("ampara y protege a Inmobiliaria" not in _res and "Juzgado Quinto" not in _res,
       "no ampara a la recurrente ni contra el juzgado sobreseído")
    ok(any("SE TOMÓ DE SU PUNTO RESOLUTIVO" in a for a in _est.avisos)
       and any("revoca_fondo_niega" in a for a in _est.avisos),
       "el aviso dice que manda el resolutivo y la rama es «revoca_fondo_niega»")
    import ensamblar_adelanto as ens
    _cg = ens.revisar_congruencia(_ruta, ["fundado"], "amparo_revision")
    ok(not any("INCONGRUENCIA QUE INVALIDA" in a for a in _cg),
       f"narrar que el juzgado concedió «para que … dejara insubsistente» no son efectos de "
       f"este proyecto: sin la alarma de incongruencia {[a[:60] for a in _cg]}")
    ok(bool(ens._RX_EFECTOS.search("la autoridad responsable deberá dejar insubsistente la resolución"))
       and bool(ens._RX_EFECTOS.search("la Sala dejará insubsistente la resolución"))
       and not ens._RX_EFECTOS.search("se concedía para que la Sala dejara insubsistente la resolución"),
       "«deberá dejar» y «dejará» son órdenes; «dejara» es la narración de otra sentencia")
    # LA ALARMA VERDADERA SIGUE: efectos escritos en la Solución y un resolutivo que niega.
    _est2 = dg.Estructura(apertura="V.", visto="para resolver.",
                          resultandos=[{"titulo": "Presentación de la demanda de amparo indirecto",
                                        "texto": "Unión Ejemplo, A.C. promovió amparo contra la Sala "
                                                 "Civil Uno."}],
                          competencia="", existencia="", procedencia="")
    _ruta2 = os.path.join(os.path.dirname(_ruta), "ar2.docx")
    dg.componer({"tipo_asunto": "amparo_revision", "numero": "2/2026",
                 "encabezado": "AMPARO EN REVISIÓN 2/2026", "quejoso": "Unión Ejemplo, A.C.",
                 "responsable": "Juzgado Séptimo de Distrito", "magistrado": "M", "secretario": "S",
                 "tribunal": "Tercer Tribunal Colegiado", "ciudad": "Querétaro",
                 "resolvio_a_quo": "concede",
                 "resolutivo_recurrida": fr.resolutivo_recurrida(SENTENCIA)},
                _est2, _c, f0.fecha_en_letra, _ruta2,
                estudio=["Solución",
                         "Es fundado el agravio y la autoridad responsable deberá dejar insubsistente "
                         "la resolución reclamada."],
                calificaciones=["fundado"], tipo_asunto="amparo_revision")
    ok(any("INCONGRUENCIA QUE INVALIDA" in a
           for a in ens.revisar_congruencia(_ruta2, ["fundado"], "amparo_revision")),
       "la alarma verdadera sigue: efectos en la Solución y un resolutivo que niega")
except Exception as ex:  # pragma: no cover
    ok(False, f"el documento no se pudo componer: {type(ex).__name__}: {ex}")

print()
if FALLOS:
    print(f"FALLAN {len(FALLOS)}: " + " · ".join(FALLOS))
    sys.exit(1)
print("RESULTADO: TODAS LAS COMPROBACIONES PASAN")
